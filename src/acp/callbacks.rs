//! Internal stable-v1 ACP callback bundle.
#![cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "exposed only after prompt and service wiring are complete"
    )
)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use agent_client_protocol::v1::schema::{
    CancelNotification, CloseSessionRequest, DeleteSessionRequest, InitializeRequest,
    ListSessionsRequest, LoadSessionRequest, NewSessionRequest, PromptRequest, PromptResponse,
    StopReason,
};
use agent_client_protocol::{Agent, Client, ConnectTo, Error as ProtocolError};

use crate::model::Message;
use crate::run::{
    execute_session_turn, prepare_session_turn, turn_events, SessionTurnStatus, TurnStart,
};

use super::approval::AcpApprover;
use super::dispatcher;
use super::error::ControlError;
use super::prompt;
use super::runtime::PromptRuntime;
use super::session_control::SessionControl;

#[derive(Debug)]
enum PromptTaskError {
    Execution(anyhow::Error),
    Dispatch(ProtocolError),
}

async fn await_execution_and_dispatch<E, D>(
    execution: E,
    dispatch: D,
) -> Result<SessionTurnStatus, PromptTaskError>
where
    E: std::future::Future<Output = anyhow::Result<SessionTurnStatus>>,
    D: std::future::Future<Output = Result<(), ProtocolError>>,
{
    tokio::pin!(execution);
    tokio::pin!(dispatch);
    tokio::select! {
        result = &mut execution => {
            let status = result.map_err(PromptTaskError::Execution)?;
            dispatch.await.map_err(PromptTaskError::Dispatch)?;
            Ok(status)
        }
        result = &mut dispatch => {
            result.map_err(PromptTaskError::Dispatch)?;
            execution.await.map_err(PromptTaskError::Execution)
        }
    }
}

fn supports_notices(request: &InitializeRequest) -> bool {
    request
        .client_capabilities
        .session
        .as_ref()
        .is_some_and(|session| session.notices.is_some())
}

pub(crate) type PromptSetupHook = Arc<dyn Fn() + Send + Sync>;

/// Build the unexposed stable-v1 callbacks over the shared session control boundary.
pub(crate) fn callback_bundle(
    control: Arc<SessionControl>,
    runtime: Arc<PromptRuntime>,
) -> impl ConnectTo<Client> {
    callback_bundle_with_setup_hook(control, runtime, None)
}

pub(crate) fn callback_bundle_with_setup_hook(
    control: Arc<SessionControl>,
    runtime: Arc<PromptRuntime>,
    prompt_setup_hook: Option<PromptSetupHook>,
) -> impl ConnectTo<Client> {
    let new_control = Arc::clone(&control);
    let list_control = Arc::clone(&control);
    let notices = Arc::new(AtomicBool::new(false));
    let initialize_notices = Arc::clone(&notices);
    let prompt_notices = notices;
    let load_control = Arc::clone(&control);
    let close_control = Arc::clone(&control);
    let delete_control = Arc::clone(&control);
    let prompt_control = Arc::clone(&control);
    let prompt_runtime = runtime;
    let cancel_control = control;

    Agent
        .v1()
        .name("icode")
        .on_receive_request(
            async move |request: InitializeRequest, responder, _connection| {
                initialize_notices.store(supports_notices(&request), Ordering::Release);
                responder.respond(SessionControl::initialize(&request))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: ListSessionsRequest, responder, connection| {
                let control = Arc::clone(&list_control);
                connection.spawn(async move {
                    responder.respond_with_result(
                        control
                            .list_sessions(&request)
                            .await
                            .map_err(ProtocolError::from),
                    )
                })
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: NewSessionRequest, responder, connection| {
                let control = Arc::clone(&new_control);
                connection.spawn(async move {
                    let result = control.new_session(&request).await;
                    responder.respond_with_result(result.map_err(ProtocolError::from))
                })
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: LoadSessionRequest, responder, connection| {
                let control = Arc::clone(&load_control);
                let task_connection = connection.clone();
                connection.spawn(async move {
                    let transaction = match control.prepare_load_with_mcp(&request).await {
                        Ok(transaction) => transaction,
                        Err(error) => return responder.respond_with_error(error.into()),
                    };
                    transaction
                        .complete_with(
                            |notification| task_connection.send_notification(notification),
                            |response| responder.respond(response),
                        )
                        .map_err(ProtocolError::from)
                })
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: CloseSessionRequest, responder, connection| {
                let control = Arc::clone(&close_control);
                connection.spawn(async move {
                    responder.respond_with_result(
                        control.close(&request).await.map_err(ProtocolError::from),
                    )
                })
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: DeleteSessionRequest, responder, connection| {
                let control = Arc::clone(&delete_control);
                connection.spawn(async move {
                    responder.respond_with_result(
                        control.delete(&request).await.map_err(ProtocolError::from),
                    )
                })
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: PromptRequest, responder, connection| {
                let message = match prompt::user_message(&request.prompt) {
                    Ok(message) => message,
                    Err(error) => {
                        return responder.respond_with_error(
                            ProtocolError::invalid_params().data(error.to_string()),
                        );
                    }
                };
                let Message::User { content } = message else {
                    unreachable!("ACP prompt adapter always returns a user message");
                };
                let mut guard = match prompt_control.begin_prompt(&request.session_id) {
                    Ok(guard) => guard,
                    Err(error) => return responder.respond_with_error(error.into()),
                };
                let session = guard.session().clone();
                let tools = guard.tools();
                let additional_directories = guard.additional_directories();
                let ownership = guard.ownership();
                let setup_ownership = Arc::clone(&ownership);
                let cwd = PathBuf::from(session.cwd());
                let runtime = Arc::clone(&prompt_runtime);
                let setup_hook = prompt_setup_hook.clone();
                let phase_four_budget = runtime.phase_four_side_call_token_budget();
                let context_window = runtime.context_window();
                let supports_notices = prompt_notices.load(Ordering::Acquire);
                let request_cancellation = responder.cancellation();
                let approver = Arc::new(AcpApprover::new(
                    request.session_id.clone(),
                    connection.clone(),
                    guard.approval_policy(),
                ));
                let session_id = request.session_id;
                let task_connection = connection.clone();

                connection.spawn(async move {
                    let setup = tokio::task::spawn_blocking(move || {
                        let _ownership = setup_ownership;
                        if let Some(hook) = setup_hook {
                            hook();
                        }
                        let agent = runtime.agent_for_with_directories(
                            &cwd,
                            additional_directories,
                            tools,
                            approver,
                        );
                        let prepared = prepare_session_turn(&session, TurnStart::User(content))?;
                        Ok::<_, anyhow::Error>((agent, session, prepared))
                    })
                    .await;
                    let (agent, session, prepared) = match setup {
                        Ok(Ok(setup)) => setup,
                        Ok(Err(error)) => {
                            return responder
                                .respond_with_error(ControlError::internal(error).into());
                        }
                        Err(error) => {
                            return responder
                                .respond_with_error(ControlError::internal(error).into());
                        }
                    };

                    let (observer, events) = turn_events::channel();
                    let execution = async {
                        let output = execute_session_turn(
                            &agent,
                            session,
                            prepared,
                            observer,
                            phase_four_budget,
                            Some(ownership),
                            async {
                                tokio::select! {
                                    _ = guard.cancelled() => {}
                                    _ = request_cancellation.cancelled() => {}
                                }
                            },
                        )
                        .await?;
                        let status = output.status;
                        let has_unsaved_messages = output.has_unsaved_messages;
                        drop(output.observer);
                        anyhow::ensure!(
                            !has_unsaved_messages,
                            "ACP prompt completed with unsaved messages"
                        );
                        Ok::<_, anyhow::Error>(status)
                    };
                    let dispatch = dispatcher::dispatch_to_client(
                        session_id,
                        context_window,
                        supports_notices,
                        events,
                        task_connection,
                    );
                    let status = match await_execution_and_dispatch(execution, dispatch).await {
                        Ok(status) => status,
                        Err(PromptTaskError::Dispatch(error)) => {
                            return responder.respond_with_error(error);
                        }
                        Err(PromptTaskError::Execution(error)) => {
                            return responder
                                .respond_with_error(ControlError::internal(error).into());
                        }
                    };
                    let stop_reason = match status {
                        SessionTurnStatus::Completed(_) => StopReason::EndTurn,
                        SessionTurnStatus::Cancelled => StopReason::Cancelled,
                    };
                    let response = responder.respond(PromptResponse::new(stop_reason));
                    drop(guard);
                    response
                })
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            async move |notification: CancelNotification,
                        _connection: agent_client_protocol::v1::ConnectionTo<Client>| {
                let _matched_active_prompt = cancel_control.cancel(&notification);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::time::Duration;

    use agent_client_protocol::schema::ProtocolVersion;
    use agent_client_protocol::v1::schema::{
        AgentCapabilities, ClientCapabilities, ClientSessionCapabilities, CloseSessionRequest,
        CloseSessionResponse, ContentBlock, DeleteSessionRequest, DeleteSessionResponse,
        InitializeResponse, ListSessionsRequest, ListSessionsResponse, LoadSessionResponse,
        McpServer, McpServerStdio, NewSessionResponse, NoticeCapabilities, PromptRequest,
        SessionAdditionalDirectoriesCapabilities, SessionCapabilities, SessionCloseCapabilities,
        SessionDeleteCapabilities, SessionId, SessionListCapabilities, SessionNotification,
    };
    use agent_client_protocol::ErrorCode;
    use tempfile::TempDir;

    use super::*;
    use crate::acp::sessions::SessionRegistry;
    use crate::agent::AgentKind;
    use crate::config::ModelProfile;
    use crate::model::ModelClient;
    use crate::session::{Record, SessionStore};

    fn prompt_runtime() -> Arc<PromptRuntime> {
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_key = "test-key".to_string();
        profile.base_url = "http://127.0.0.1:1".to_string();
        profile.model = "mock-model".to_string();
        Arc::new(PromptRuntime::new(
            Arc::new(ModelClient::from_profile(&profile).unwrap()),
            AgentKind::Code,
            1,
            -1,
        ))
    }

    #[test]
    fn notice_support_requires_explicit_session_capability() {
        let without = InitializeRequest::new(ProtocolVersion::V1);
        let with = InitializeRequest::new(ProtocolVersion::V1).client_capabilities(
            ClientCapabilities::new()
                .session(ClientSessionCapabilities::new().notices(NoticeCapabilities::new())),
        );

        assert!(!supports_notices(&without));
        assert!(supports_notices(&with));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dispatch_failure_drops_pending_execution_without_waiting() {
        struct DropSignal(Arc<AtomicBool>);
        impl Drop for DropSignal {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let execution_dropped = Arc::clone(&dropped);
        let signal = DropSignal(execution_dropped);
        let execution = async move {
            let _signal = signal;
            std::future::pending::<anyhow::Result<SessionTurnStatus>>().await
        };
        let dispatch = async { Err(ProtocolError::internal_error()) };

        let result = tokio::time::timeout(
            Duration::from_secs(1),
            await_execution_and_dispatch(execution, dispatch),
        )
        .await
        .expect("dispatch failure waited for pending execution");
        assert!(matches!(result, Err(PromptTaskError::Dispatch(_))));
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn callbacks_preserve_load_order_errors_and_close_lifecycle() -> Result<(), ProtocolError>
    {
        let directory = TempDir::new().unwrap();
        let store = SessionStore::with_root(directory.path().join("sessions"));
        let sessions = Arc::new(SessionRegistry::new(store.clone()));
        let control = Arc::new(SessionControl::new(
            Arc::clone(&sessions),
            "model-a",
            "build",
        ));

        let loaded = store
            .create(directory.path().to_str().unwrap(), "model-a", "build")
            .unwrap();
        loaded
            .append(&Record::User {
                text: "question".to_string(),
            })
            .unwrap();
        loaded
            .append(&Record::Assistant {
                content: Some("answer".to_string()),
                reasoning: None,
                tool_calls: vec![],
                responses_items: vec![],
                responses_continuation: None,
            })
            .unwrap();
        let load_id = SessionId::new(loaded.id());

        let cancel_id = sessions
            .create(
                directory.path().to_str().unwrap(),
                "model-a",
                "build",
                Arc::new(crate::tools::ToolRegistry::builtin()),
            )
            .unwrap();
        let mut prompt = sessions.begin_prompt(&cancel_id).unwrap();
        let notifications = Arc::new(Mutex::new(Vec::new()));
        let received = Arc::clone(&notifications);
        let client = Client.v1().on_receive_notification(
            async move |notification: SessionNotification, _connection| {
                received
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(notification.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        );

        client
            .connect_with(
                callback_bundle(control, prompt_runtime()),
                async move |connection| {
                    let initialized: InitializeResponse = connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    assert_eq!(initialized.protocol_version, ProtocolVersion::V1);
                    assert_eq!(
                        initialized.agent_capabilities,
                        AgentCapabilities::new()
                            .load_session(true)
                            .session_capabilities(
                                SessionCapabilities::new()
                                    .additional_directories(
                                        SessionAdditionalDirectoriesCapabilities::new(),
                                    )
                                    .close(SessionCloseCapabilities::new())
                                    .delete(SessionDeleteCapabilities::new())
                                    .list(SessionListCapabilities::new()),
                            )
                    );

                    let empty_prompt = connection
                        .send_request(PromptRequest::new(cancel_id.clone(), vec![]))
                        .block_task()
                        .await
                        .unwrap_err();
                    assert_eq!(empty_prompt.code, ErrorCode::INVALID_PARAMS);

                    let missing_prompt = connection
                        .send_request(PromptRequest::new(
                            SessionId::new("missing"),
                            vec![ContentBlock::from("hello")],
                        ))
                        .block_task()
                        .await
                        .unwrap_err();
                    assert_eq!(missing_prompt.code, ErrorCode::RESOURCE_NOT_FOUND);

                    let invalid = connection
                        .send_request(NewSessionRequest::new("relative"))
                        .block_task()
                        .await
                        .unwrap_err();
                    assert_eq!(invalid.code, ErrorCode::INVALID_PARAMS);

                    #[cfg(unix)]
                    {
                        let script = concat!(
                            "read initialize; sleep 0.2; printf '%s\\n' '",
                            "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{",
                            "\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},",
                            "\"serverInfo\":{\"name\":\"slow\",\"version\":\"1\"}}}'; ",
                            "read initialized; read list; printf '%s\\n' '",
                            "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[]}}'; ",
                            "read until_closed"
                        );
                        let slow_request =
                            NewSessionRequest::new(directory.path()).mcp_servers(vec![
                                McpServer::Stdio(
                                    McpServerStdio::new("slow", "/bin/sh")
                                        .args(vec!["-c".into(), script.into()]),
                                ),
                            ]);
                        let slow = connection.send_request(slow_request).block_task();
                        tokio::pin!(slow);
                        assert!(tokio::time::timeout(Duration::from_millis(25), &mut slow)
                            .await
                            .is_err());
                        let initialized: InitializeResponse = tokio::time::timeout(
                            Duration::from_secs(1),
                            connection
                                .send_request(InitializeRequest::new(ProtocolVersion::V1))
                                .block_task(),
                        )
                        .await
                        .expect("slow MCP startup blocked ACP dispatch")?;
                        assert_eq!(initialized.protocol_version, ProtocolVersion::V1);
                        let _: NewSessionResponse = slow.await?;
                    }

                    let created: NewSessionResponse = connection
                        .send_request(NewSessionRequest::new(directory.path()))
                        .block_task()
                        .await?;
                    let created_id = created.session_id.clone();
                    assert!(store.load(&created_id.to_string()).is_ok());
                    let _: CloseSessionResponse = connection
                        .send_request(CloseSessionRequest::new(created_id.clone()))
                        .block_task()
                        .await?;
                    assert!(store.load(&created_id.to_string()).is_ok());
                    let listed: ListSessionsResponse = connection
                        .send_request(ListSessionsRequest::new().cwd(directory.path()))
                        .block_task()
                        .await?;
                    assert_eq!(listed.next_cursor, None);
                    assert!(listed.sessions.iter().any(|session| {
                        session.session_id == created_id && session.cwd == directory.path()
                    }));
                    let invalid_cursor = connection
                        .send_request(ListSessionsRequest::new().cursor("unsolicited"))
                        .block_task()
                        .await
                        .unwrap_err();
                    assert_eq!(invalid_cursor.code, ErrorCode::INVALID_PARAMS);
                    let closed_again = connection
                        .send_request(CloseSessionRequest::new(created_id.clone()))
                        .block_task()
                        .await
                        .unwrap_err();
                    assert_eq!(closed_again.code, ErrorCode::RESOURCE_NOT_FOUND);
                    let _: DeleteSessionResponse = connection
                        .send_request(DeleteSessionRequest::new(created_id.clone()))
                        .block_task()
                        .await?;
                    assert!(store.load(&created_id.to_string()).is_err());
                    let listed: ListSessionsResponse = connection
                        .send_request(ListSessionsRequest::new().cwd(directory.path()))
                        .block_task()
                        .await?;
                    assert!(listed
                        .sessions
                        .iter()
                        .all(|session| session.session_id != created_id));
                    let deleted_again = connection
                        .send_request(DeleteSessionRequest::new(created_id))
                        .block_task()
                        .await
                        .unwrap_err();
                    assert_eq!(deleted_again.code, ErrorCode::RESOURCE_NOT_FOUND);

                    let _: LoadSessionResponse = connection
                        .send_request(LoadSessionRequest::new(load_id, directory.path()))
                        .block_task()
                        .await?;
                    {
                        let updates = notifications
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        assert_eq!(updates.len(), 2);
                        assert_eq!(
                            serde_json::to_value(&updates[0]).unwrap()["sessionUpdate"],
                            "user_message_chunk"
                        );
                        assert_eq!(
                            serde_json::to_value(&updates[1]).unwrap()["sessionUpdate"],
                            "agent_message_chunk"
                        );
                    }

                    let close = connection
                        .send_request(CloseSessionRequest::new(cancel_id))
                        .block_task();
                    tokio::pin!(close);
                    assert!(tokio::time::timeout(Duration::from_millis(25), &mut close)
                        .await
                        .is_err());
                    tokio::time::timeout(Duration::from_secs(1), prompt.cancelled())
                        .await
                        .expect("close callback did not signal the active prompt");
                    drop(prompt);
                    let _: CloseSessionResponse = close.await?;
                    Ok(())
                },
            )
            .await
    }

    #[test]
    fn callback_bundle_has_the_stable_v1_component_type() {
        fn accepts_component(_component: impl ConnectTo<Client>) {}

        let directory = TempDir::new().unwrap();
        let store = SessionStore::with_root(directory.path().join("sessions"));
        let sessions = Arc::new(SessionRegistry::new(store));
        accepts_component(callback_bundle(
            Arc::new(SessionControl::new(sessions, "model-a", "build")),
            prompt_runtime(),
        ));
    }
}
