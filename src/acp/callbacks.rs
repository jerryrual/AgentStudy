//! Internal stable-v1 ACP callback bundle.
#![cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "exposed only after prompt and service wiring are complete"
    )
)]

use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol::v1::schema::{
    CancelNotification, InitializeRequest, LoadSessionRequest, NewSessionRequest, PromptRequest,
    PromptResponse, StopReason,
};
use agent_client_protocol::{Agent, Client, ConnectTo, Error as ProtocolError};

use crate::model::Message;
use crate::run::{
    execute_session_turn, prepare_session_turn, turn_events, SessionTurnStatus, TurnStart,
};

use super::dispatcher;
use super::error::ControlError;
use super::prompt;
use super::runtime::PromptRuntime;
use super::session_control::SessionControl;

/// Build the unexposed stable-v1 callbacks over the shared session control boundary.
pub(crate) fn callback_bundle(
    control: Arc<SessionControl>,
    runtime: Arc<PromptRuntime>,
) -> impl ConnectTo<Client> {
    let new_control = Arc::clone(&control);
    let load_control = Arc::clone(&control);
    let prompt_control = Arc::clone(&control);
    let prompt_runtime = runtime;
    let cancel_control = control;

    Agent
        .v1()
        .name("icode")
        .on_receive_request(
            async move |request: InitializeRequest, responder, _connection| {
                responder.respond(SessionControl::initialize(&request))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: NewSessionRequest, responder, _connection| {
                let control = Arc::clone(&new_control);
                let result = tokio::task::spawn_blocking(move || control.new_session(&request))
                    .await
                    .map_err(ControlError::internal)
                    .and_then(|result| result);
                responder.respond_with_result(result.map_err(ProtocolError::from))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: LoadSessionRequest, responder, connection| {
                let control = Arc::clone(&load_control);
                let transaction =
                    tokio::task::spawn_blocking(move || control.prepare_load(&request))
                        .await
                        .map_err(ControlError::internal)
                        .and_then(|result| result);
                let transaction = match transaction {
                    Ok(transaction) => transaction,
                    Err(error) => return responder.respond_with_error(error.into()),
                };
                transaction
                    .complete_with(
                        |notification| connection.send_notification(notification),
                        |response| responder.respond(response),
                    )
                    .map_err(ProtocolError::from)
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
                let cwd = PathBuf::from(session.cwd());
                let runtime = Arc::clone(&prompt_runtime);
                let phase_four_budget = runtime.phase_four_side_call_token_budget();
                let request_cancellation = responder.cancellation();
                let session_id = request.session_id;
                let task_connection = connection.clone();

                connection.spawn(async move {
                    let setup = tokio::task::spawn_blocking(move || {
                        let agent = runtime.agent_for(&cwd);
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
                            async move {
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
                    let dispatch =
                        dispatcher::dispatch_to_client(session_id, events, task_connection);
                    let (execution, dispatch) = tokio::join!(execution, dispatch);
                    if let Err(error) = dispatch {
                        return responder.respond_with_error(error);
                    }
                    let status = match execution {
                        Ok(status) => status,
                        Err(error) => {
                            return responder
                                .respond_with_error(ControlError::internal(error).into());
                        }
                    };
                    let stop_reason = match status {
                        SessionTurnStatus::Completed(_) => StopReason::EndTurn,
                        SessionTurnStatus::Cancelled => StopReason::Cancelled,
                    };
                    responder.respond(PromptResponse::new(stop_reason))
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
    use std::sync::Mutex;
    use std::time::Duration;

    use agent_client_protocol::schema::ProtocolVersion;
    use agent_client_protocol::v1::schema::{
        AgentCapabilities, ContentBlock, InitializeResponse, LoadSessionResponse,
        NewSessionResponse, PromptRequest, SessionId, SessionNotification,
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

    #[tokio::test(flavor = "current_thread")]
    async fn callbacks_preserve_load_order_errors_and_cancellation() -> Result<(), ProtocolError> {
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
            })
            .unwrap();
        let load_id = SessionId::new(loaded.id());

        let cancel_id = sessions
            .create(directory.path().to_str().unwrap(), "model-a", "build")
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
                    assert_eq!(initialized.agent_capabilities, AgentCapabilities::new());

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

                    let created: NewSessionResponse = connection
                        .send_request(NewSessionRequest::new(directory.path()))
                        .block_task()
                        .await?;
                    assert!(store.load(&created.session_id.to_string()).is_ok());

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

                    connection.send_notification(CancelNotification::new(cancel_id))?;
                    tokio::time::timeout(Duration::from_secs(1), prompt.cancelled())
                        .await
                        .expect("cancel callback did not signal the active prompt");
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
