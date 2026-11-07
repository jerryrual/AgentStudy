//! Side-effect boundaries for stable ACP session control requests.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming ACP protocol handlers")
)]

use std::path::Path;
use std::sync::Arc;

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::v1::schema::{
    AgentCapabilities, CancelNotification, InitializeRequest, InitializeResponse,
    LoadSessionRequest, LoadSessionResponse, NewSessionRequest, NewSessionResponse,
    SessionNotification,
};
use anyhow::{ensure, Context, Result};

use super::sessions::{PreparedLoad, SessionRegistry};

/// Validated session creation and cancellation shared by future wire handlers.
pub(crate) struct SessionControl {
    sessions: Arc<SessionRegistry>,
    model: String,
    agent: String,
}

/// A prepared load that owns its lease until replay and response complete.
pub(crate) struct LoadTransaction {
    prepared: PreparedLoad,
}

impl SessionControl {
    /// Create a control boundary with defaults for newly persisted sessions.
    pub(crate) fn new(
        sessions: Arc<SessionRegistry>,
        model: impl Into<String>,
        agent: impl Into<String>,
    ) -> Self {
        Self {
            sessions,
            model: model.into(),
            agent: agent.into(),
        }
    }

    /// Build the honest stable-v1 initialization response for the staged service.
    pub(crate) fn initialize(request: &InitializeRequest) -> InitializeResponse {
        let protocol_version = if request.protocol_version == ProtocolVersion::V1 {
            request.protocol_version
        } else {
            ProtocolVersion::LATEST
        };
        InitializeResponse::new(protocol_version).agent_capabilities(AgentCapabilities::new())
    }

    /// Validate all setup fields before creating any durable session state.
    pub(crate) fn new_session(&self, request: &NewSessionRequest) -> Result<NewSessionResponse> {
        ensure!(
            request.additional_directories.is_empty(),
            "ACP additional workspace roots are not supported"
        );
        ensure!(
            request.mcp_servers.is_empty(),
            "ACP MCP servers are not supported"
        );
        let cwd = validated_cwd(&request.cwd)?;
        let session_id = self.sessions.create(cwd, &self.model, &self.agent)?;
        Ok(NewSessionResponse::new(session_id))
    }

    /// Validate and project a load while its session remains hidden from prompts.
    pub(crate) fn prepare_load(&self, request: &LoadSessionRequest) -> Result<LoadTransaction> {
        ensure!(
            request.additional_directories.is_empty(),
            "ACP additional workspace roots are not supported"
        );
        ensure!(
            request.mcp_servers.is_empty(),
            "ACP MCP servers are not supported"
        );
        let cwd = validated_cwd(&request.cwd)?;
        let prepared = self.sessions.prepare_load(&request.session_id, cwd)?;
        Ok(LoadTransaction { prepared })
    }

    /// Forward a fire-and-forget cancellation to the matching active prompt.
    pub(crate) fn cancel(&self, notification: &CancelNotification) -> bool {
        self.sessions.cancel(&notification.session_id)
    }
}

impl LoadTransaction {
    /// Admit every replay notification before the response, committing only on success.
    pub(crate) fn complete_with(
        self,
        mut notify: impl FnMut(SessionNotification) -> Result<()>,
        respond: impl FnOnce(LoadSessionResponse) -> Result<()>,
    ) -> Result<()> {
        let session_id = self.prepared.session_id();
        let updates = self.prepared.updates().to_vec();
        let pending = self.prepared.activate()?;
        for update in updates {
            notify(SessionNotification::new(session_id.clone(), update))?;
        }
        respond(LoadSessionResponse::new())?;
        pending.commit();
        Ok(())
    }
}

/// Return the exact UTF-8 cwd after validating its durable session preconditions.
fn validated_cwd(cwd: &Path) -> Result<&str> {
    ensure!(cwd.is_absolute(), "ACP session cwd must be absolute");
    let cwd = cwd
        .to_str()
        .context("ACP session cwd must be valid UTF-8")?;
    ensure!(!cwd.contains('\0'), "ACP session cwd must not contain NUL");
    let metadata =
        std::fs::metadata(cwd).with_context(|| format!("ACP session cwd does not exist: {cwd}"))?;
    ensure!(
        metadata.is_dir(),
        "ACP session cwd is not a directory: {cwd}"
    );
    Ok(cwd)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use agent_client_protocol::v1::schema::{McpServer, McpServerStdio, SessionId};
    use tempfile::TempDir;

    use super::*;
    use crate::model::ToolCall;
    use crate::session::{Record, Session, SessionStore};

    fn control() -> (TempDir, SessionStore, SessionControl) {
        let directory = TempDir::new().unwrap();
        let store = SessionStore::with_root(directory.path().join("sessions"));
        let sessions = Arc::new(SessionRegistry::new(store.clone()));
        let control = SessionControl::new(sessions, "model-a", "build");
        (directory, store, control)
    }

    #[test]
    fn initialization_negotiates_stable_v1_without_unimplemented_capabilities() {
        for requested in [ProtocolVersion::V0, ProtocolVersion::V1] {
            let response = SessionControl::initialize(&InitializeRequest::new(requested));
            assert_eq!(response.protocol_version, ProtocolVersion::V1);
            assert_eq!(response.agent_capabilities, AgentCapabilities::new());
        }
    }

    #[test]
    fn new_session_validates_then_persists_the_requested_cwd_and_defaults() {
        let (directory, store, control) = control();
        let response = control
            .new_session(&NewSessionRequest::new(directory.path()))
            .unwrap();
        let loaded = store.load(&response.session_id.to_string()).unwrap();

        assert_eq!(loaded.meta.cwd, directory.path().to_str().unwrap());
        assert_eq!(loaded.meta.model, "model-a");
        assert_eq!(loaded.meta.agent, "build");
    }

    #[test]
    fn rejected_setup_requests_create_no_persistent_session() {
        let (directory, store, control) = control();
        let file = directory.path().join("file.txt");
        std::fs::write(&file, "not a directory").unwrap();
        let cases = [
            NewSessionRequest::new("relative/path"),
            NewSessionRequest::new(directory.path().join("missing")),
            NewSessionRequest::new(&file),
            NewSessionRequest::new(directory.path())
                .additional_directories(vec![directory.path().join("other")]),
            NewSessionRequest::new(directory.path()).mcp_servers(vec![McpServer::Stdio(
                McpServerStdio::new("tools", "/bin/echo"),
            )]),
        ];

        for request in cases {
            assert!(control.new_session(&request).is_err());
        }
        assert!(store.list().unwrap().is_empty());
    }

    fn persisted_session(
        store: &SessionStore,
        cwd: &Path,
        records: &[Record],
    ) -> (Session, SessionId) {
        let session = store
            .create(cwd.to_str().unwrap(), "model-a", "build")
            .unwrap();
        for record in records {
            session.append(record).unwrap();
        }
        let session_id = SessionId::new(session.id());
        (session, session_id)
    }

    #[test]
    fn load_admits_complete_replay_before_response_then_enables_prompts() {
        let (directory, store, control) = control();
        let records = vec![
            Record::User {
                text: "question".to_string(),
            },
            Record::Assistant {
                content: Some("checking".to_string()),
                reasoning: Some("thinking".to_string()),
                tool_calls: vec![ToolCall {
                    id: "call-1".to_string(),
                    name: "read_file".to_string(),
                    arguments: r#"{"path":"a"}"#.to_string(),
                }],
                responses_items: vec![],
            },
            Record::Tool {
                call_id: "call-1".to_string(),
                name: "read_file".to_string(),
                content: "contents".to_string(),
            },
            Record::Assistant {
                content: Some("done".to_string()),
                reasoning: None,
                tool_calls: vec![],
                responses_items: vec![],
            },
        ];
        let (_session, session_id) = persisted_session(&store, directory.path(), &records);
        let events = RefCell::new(Vec::new());
        let request = LoadSessionRequest::new(session_id.clone(), directory.path());

        control
            .prepare_load(&request)
            .unwrap()
            .complete_with(
                |notification| {
                    assert!(control
                        .sessions
                        .begin_prompt(&session_id)
                        .err()
                        .unwrap()
                        .to_string()
                        .contains("still loading"));
                    assert!(!control.sessions.close(&session_id));
                    assert!(!control.sessions.cancel(&session_id));
                    events
                        .borrow_mut()
                        .push(serde_json::to_value(notification).unwrap());
                    Ok(())
                },
                |response| {
                    assert!(control
                        .sessions
                        .begin_prompt(&session_id)
                        .err()
                        .unwrap()
                        .to_string()
                        .contains("still loading"));
                    assert!(!control.sessions.close(&session_id));
                    assert!(!control.sessions.cancel(&session_id));
                    events
                        .borrow_mut()
                        .push(serde_json::json!({"response": response}));
                    Ok(())
                },
            )
            .unwrap();

        let events = events.into_inner();
        assert_eq!(events.len(), 7);
        assert_eq!(
            events[..6]
                .iter()
                .map(|event| event["update"]["sessionUpdate"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "user_message_chunk",
                "agent_thought_chunk",
                "agent_message_chunk",
                "tool_call",
                "tool_call_update",
                "agent_message_chunk",
            ]
        );
        assert!(events[..6]
            .iter()
            .all(|event| event["sessionId"] == session_id.to_string()));
        assert_eq!(events[0]["update"]["content"]["text"], "question");
        assert_eq!(events[3]["update"]["rawInput"]["path"], "a");
        assert_eq!(events[4]["update"]["status"], "completed");
        assert_eq!(events[4]["update"]["rawOutput"], "contents");
        assert_eq!(events[6], serde_json::json!({"response": {}}));
        let guard = control.sessions.begin_prompt(&session_id).unwrap();
        assert!(store
            .try_acquire_run_lease(&session_id.to_string())
            .is_err());
        drop(guard);
    }

    #[test]
    fn load_preparation_failures_release_the_lease_without_registration() {
        let (directory, store, control) = control();
        let other = TempDir::new().unwrap();
        let (_session, session_id) = persisted_session(&store, directory.path(), &[]);
        let wrong_cwd = LoadSessionRequest::new(session_id.clone(), other.path());
        assert_eq!(
            control.prepare_load(&wrong_cwd).err().unwrap().to_string(),
            "ACP load cwd does not match persisted session cwd"
        );
        assert!(control.sessions.begin_prompt(&session_id).is_err());
        store
            .try_acquire_run_lease(&session_id.to_string())
            .unwrap();

        let corrupt_records = [Record::Assistant {
            content: None,
            reasoning: None,
            tool_calls: vec![
                ToolCall {
                    id: "same".to_string(),
                    name: "bash".to_string(),
                    arguments: "{}".to_string(),
                },
                ToolCall {
                    id: "same".to_string(),
                    name: "grep".to_string(),
                    arguments: "{}".to_string(),
                },
            ],
            responses_items: vec![],
        }];
        let (_corrupt, corrupt_id) = persisted_session(&store, directory.path(), &corrupt_records);
        let request = LoadSessionRequest::new(corrupt_id.clone(), directory.path());
        assert_eq!(
            control.prepare_load(&request).err().unwrap().to_string(),
            "duplicate ACP replay tool call ID: same"
        );
        assert!(control.sessions.begin_prompt(&corrupt_id).is_err());
        store
            .try_acquire_run_lease(&corrupt_id.to_string())
            .unwrap();
    }

    #[test]
    fn load_callback_failures_roll_back_registration_and_release_the_lease() {
        for fail_response in [false, true] {
            let (directory, store, control) = control();
            let records = [
                Record::User {
                    text: "first".to_string(),
                },
                Record::Assistant {
                    content: Some("second".to_string()),
                    reasoning: None,
                    tool_calls: vec![],
                    responses_items: vec![],
                },
            ];
            let (_session, session_id) = persisted_session(&store, directory.path(), &records);
            let events = RefCell::new(Vec::new());
            let request = LoadSessionRequest::new(session_id.clone(), directory.path());
            let mut notification_count = 0;

            let error = control
                .prepare_load(&request)
                .unwrap()
                .complete_with(
                    |notification| {
                        notification_count += 1;
                        events.borrow_mut().push(
                            serde_json::to_value(notification).unwrap()["update"]["sessionUpdate"]
                                .as_str()
                                .unwrap()
                                .to_string(),
                        );
                        if !fail_response && notification_count == 2 {
                            anyhow::bail!("notification closed");
                        }
                        Ok(())
                    },
                    |_| {
                        events.borrow_mut().push("response".to_string());
                        if fail_response {
                            anyhow::bail!("response closed");
                        }
                        Ok(())
                    },
                )
                .err()
                .unwrap();

            assert_eq!(
                error.to_string(),
                if fail_response {
                    "response closed"
                } else {
                    "notification closed"
                }
            );
            let expected = if fail_response {
                vec!["user_message_chunk", "agent_message_chunk", "response"]
            } else {
                vec!["user_message_chunk", "agent_message_chunk"]
            };
            assert_eq!(events.into_inner(), expected);
            assert!(control.sessions.begin_prompt(&session_id).is_err());
            store
                .try_acquire_run_lease(&session_id.to_string())
                .unwrap();
        }
    }

    #[tokio::test]
    async fn cancel_only_signals_the_matching_active_prompt() {
        let (directory, _store, control) = control();
        let response = control
            .new_session(&NewSessionRequest::new(directory.path()))
            .unwrap();
        let unknown = CancelNotification::new(SessionId::new("unknown"));
        assert!(!control.cancel(&unknown));

        let mut guard = control.sessions.begin_prompt(&response.session_id).unwrap();
        assert!(control.cancel(&CancelNotification::new(response.session_id.clone())));
        tokio::time::timeout(std::time::Duration::from_secs(1), guard.cancelled())
            .await
            .expect("cancel signal was not delivered");
    }
}
