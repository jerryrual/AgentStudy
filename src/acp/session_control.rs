//! Side-effect boundaries for stable ACP session control requests.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming ACP protocol handlers")
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::error::{ControlError, ControlResult, InvalidSessionRequest};
use super::mcp;
use super::sessions::{BeginPromptError, LoadError, PreparedLoad, PromptGuard, SessionRegistry};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::v1::schema::{
    AgentCapabilities, CancelNotification, InitializeRequest, InitializeResponse,
    LoadSessionRequest, LoadSessionResponse, NewSessionRequest, NewSessionResponse,
    SessionAdditionalDirectoriesCapabilities, SessionCapabilities, SessionNotification,
};
use agent_client_protocol::Error as ProtocolError;

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
        InitializeResponse::new(protocol_version).agent_capabilities(
            AgentCapabilities::new()
                .load_session(true)
                .session_capabilities(
                    SessionCapabilities::new()
                        .additional_directories(SessionAdditionalDirectoriesCapabilities::new()),
                ),
        )
    }

    /// Validate all setup fields before creating any durable session state.
    pub(crate) async fn new_session(
        &self,
        request: &NewSessionRequest,
    ) -> ControlResult<NewSessionResponse> {
        let additional_directories =
            validated_additional_directories(&request.additional_directories)?;
        let requested_cwd = request.cwd.clone();
        let cwd = tokio::task::spawn_blocking(move || {
            validated_cwd(&requested_cwd).map(|cwd| cwd.to_string())
        })
        .await
        .map_err(ControlError::internal)??;
        let specs = mcp::stdio_specs(&request.mcp_servers)
            .map_err(|_| ControlError::invalid(InvalidSessionRequest::InvalidMcpServers))?;
        let tools = mcp::start_tool_registry(&specs, std::path::Path::new(&cwd))
            .await
            .map_err(ControlError::internal)?;
        let store = self.sessions.store();
        let model = self.model.clone();
        let agent = self.agent.clone();
        let prepared = tokio::task::spawn_blocking(move || {
            SessionRegistry::prepare_create(store, &cwd, &model, &agent)
        })
        .await
        .map_err(ControlError::internal)?
        .map_err(ControlError::internal)?;
        let session_id = self
            .sessions
            .commit_create_with_directories(prepared, tools, additional_directories)
            .map_err(ControlError::internal)?;
        Ok(NewSessionResponse::new(session_id))
    }

    /// Validate and project a load while its session remains hidden from prompts.
    pub(crate) fn prepare_load(
        &self,
        request: &LoadSessionRequest,
    ) -> ControlResult<LoadTransaction> {
        let additional_directories =
            validated_additional_directories(&request.additional_directories)?;
        if !request.mcp_servers.is_empty() {
            return Err(ControlError::invalid(InvalidSessionRequest::McpServers));
        }
        let cwd = validated_cwd(&request.cwd)?;
        let mut prepared = self
            .sessions
            .prepare_load(&request.session_id, cwd)
            .map_err(|error| map_load_error(error, &request.session_id))?;
        prepared.set_additional_directories(additional_directories);
        Ok(LoadTransaction { prepared })
    }

    /// Validate, rebuild MCP tools, and project a load without blocking dispatch.
    pub(crate) async fn prepare_load_with_mcp(
        &self,
        request: &LoadSessionRequest,
    ) -> ControlResult<LoadTransaction> {
        let additional_directories =
            validated_additional_directories(&request.additional_directories)?;
        let specs = mcp::stdio_specs(&request.mcp_servers)
            .map_err(|_| ControlError::invalid(InvalidSessionRequest::InvalidMcpServers))?;
        let requested_cwd = request.cwd.clone();
        let cwd = tokio::task::spawn_blocking(move || {
            validated_cwd(&requested_cwd).map(|cwd| cwd.to_string())
        })
        .await
        .map_err(ControlError::internal)??;
        if self.sessions.is_registered(&request.session_id) {
            return Err(ControlError::invalid(
                InvalidSessionRequest::AlreadyRegistered,
            ));
        }
        let registry = Arc::downgrade(&self.sessions);
        let store = self.sessions.store();
        let session_id = request.session_id.clone();
        let expected_cwd = cwd.clone();
        let mut prepared = tokio::task::spawn_blocking(move || {
            SessionRegistry::prepare_load_detached(registry, store, session_id, expected_cwd)
        })
        .await
        .map_err(ControlError::internal)?
        .map_err(|error| map_load_error(error, &request.session_id))?;
        prepared.set_additional_directories(additional_directories);
        let tools = mcp::start_tool_registry(&specs, std::path::Path::new(&cwd))
            .await
            .map_err(ControlError::internal)?;
        prepared.set_tools(tools);
        Ok(LoadTransaction { prepared })
    }

    /// Acquire the prompt slot or classify its client-visible admission failure.
    pub(crate) fn begin_prompt(
        &self,
        session_id: &agent_client_protocol::v1::schema::SessionId,
    ) -> ControlResult<PromptGuard> {
        self.sessions
            .begin_prompt(session_id)
            .map_err(|error| match error {
                BeginPromptError::SessionNotFound => {
                    ControlError::SessionNotFound(session_id.clone())
                }
                BeginPromptError::Loading => {
                    ControlError::invalid(InvalidSessionRequest::PromptLoading)
                }
                BeginPromptError::ActivePrompt => {
                    ControlError::invalid(InvalidSessionRequest::PromptActive)
                }
            })
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
        mut notify: impl FnMut(SessionNotification) -> Result<(), ProtocolError>,
        respond: impl FnOnce(LoadSessionResponse) -> Result<(), ProtocolError>,
    ) -> ControlResult<()> {
        let session_id = self.prepared.session_id();
        let updates = self.prepared.updates().to_vec();
        let pending = self
            .prepared
            .activate()
            .map_err(|error| map_load_error(error, &session_id))?;
        for update in updates {
            notify(SessionNotification::new(session_id.clone(), update))
                .map_err(ControlError::Protocol)?;
        }
        respond(LoadSessionResponse::new()).map_err(ControlError::Protocol)?;
        pending.commit();
        Ok(())
    }
}

fn map_load_error(
    error: LoadError,
    session_id: &agent_client_protocol::v1::schema::SessionId,
) -> ControlError {
    match error {
        LoadError::InvalidSessionId => {
            ControlError::invalid(InvalidSessionRequest::InvalidSessionId)
        }
        LoadError::SessionNotFound => ControlError::SessionNotFound(session_id.clone()),
        LoadError::ActiveElsewhere => ControlError::invalid(InvalidSessionRequest::ActiveElsewhere),
        LoadError::AlreadyRegistered => {
            ControlError::invalid(InvalidSessionRequest::AlreadyRegistered)
        }
        LoadError::CwdMismatch => ControlError::invalid(InvalidSessionRequest::CwdMismatch),
        LoadError::Internal(error) => ControlError::Internal(error),
    }
}

/// Preserve the exact ordered roots after enforcing ACP's absolute-path rule.
fn validated_additional_directories(directories: &[PathBuf]) -> ControlResult<Vec<PathBuf>> {
    if directories.iter().any(|directory| !directory.is_absolute()) {
        return Err(ControlError::invalid(
            InvalidSessionRequest::AdditionalDirectoryNotAbsolute,
        ));
    }
    Ok(directories.to_vec())
}

/// Return the exact UTF-8 cwd after validating its durable session preconditions.
fn validated_cwd(cwd: &Path) -> ControlResult<&str> {
    if !cwd.is_absolute() {
        return Err(ControlError::invalid(InvalidSessionRequest::CwdNotAbsolute));
    }
    let cwd = cwd
        .to_str()
        .ok_or_else(|| ControlError::invalid(InvalidSessionRequest::CwdNotUtf8))?;
    if cwd.contains('\0') {
        return Err(ControlError::invalid(InvalidSessionRequest::CwdContainsNul));
    }
    let metadata = std::fs::metadata(cwd).map_err(|error| match error.kind() {
        std::io::ErrorKind::NotFound => ControlError::invalid(InvalidSessionRequest::CwdMissing),
        std::io::ErrorKind::NotADirectory => {
            ControlError::invalid(InvalidSessionRequest::CwdNotDirectory)
        }
        _ => ControlError::internal(error),
    })?;
    if !metadata.is_dir() {
        return Err(ControlError::invalid(
            InvalidSessionRequest::CwdNotDirectory,
        ));
    }
    Ok(cwd)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use agent_client_protocol::v1::schema::{EnvVariable, McpServer, McpServerStdio, SessionId};
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
            assert_eq!(
                response.agent_capabilities,
                AgentCapabilities::new()
                    .load_session(true)
                    .session_capabilities(
                        SessionCapabilities::new().additional_directories(
                            SessionAdditionalDirectoriesCapabilities::new(),
                        )
                    )
            );
        }
    }

    #[tokio::test]
    async fn new_session_validates_then_persists_the_requested_cwd_and_defaults() {
        let (directory, store, control) = control();
        let response = control
            .new_session(&NewSessionRequest::new(directory.path()))
            .await
            .unwrap();
        let loaded = store.load(&response.session_id.to_string()).unwrap();

        assert_eq!(loaded.meta.cwd, directory.path().to_str().unwrap());
        assert_eq!(loaded.meta.model, "model-a");
        assert_eq!(loaded.meta.agent, "build");
    }

    #[tokio::test]
    async fn new_session_keeps_ordered_additional_roots_out_of_the_transcript() {
        let (directory, store, control) = control();
        let roots = vec![
            directory.path().join("extra-alpha"),
            directory.path().join("extra-beta"),
        ];
        let response = control
            .new_session(
                &NewSessionRequest::new(directory.path()).additional_directories(roots.clone()),
            )
            .await
            .unwrap();

        let guard = control.begin_prompt(&response.session_id).unwrap();
        assert_eq!(guard.additional_directories(), roots);
        drop(guard);
        let id = response.session_id.to_string();
        let transcript = std::fs::read_to_string(store.path(&id)).unwrap();
        assert!(!transcript.contains("extra-alpha"));
        assert!(!transcript.contains("extra-beta"));
    }

    #[tokio::test]
    async fn rejected_setup_requests_create_no_persistent_session() {
        let (directory, store, control) = control();
        let file = directory.path().join("file.txt");
        std::fs::write(&file, "not a directory").unwrap();
        let cases = [
            (
                NewSessionRequest::new("relative/path"),
                InvalidSessionRequest::CwdNotAbsolute,
            ),
            (
                NewSessionRequest::new(directory.path().join("missing")),
                InvalidSessionRequest::CwdMissing,
            ),
            (
                NewSessionRequest::new(&file),
                InvalidSessionRequest::CwdNotDirectory,
            ),
            (
                NewSessionRequest::new(directory.path())
                    .additional_directories(vec![PathBuf::from("other")]),
                InvalidSessionRequest::AdditionalDirectoryNotAbsolute,
            ),
            (
                NewSessionRequest::new(directory.path())
                    .mcp_servers(vec![McpServer::Stdio(McpServerStdio::new("", "/bin/echo"))]),
                InvalidSessionRequest::InvalidMcpServers,
            ),
        ];

        for (request, reason) in cases {
            assert!(matches!(
                control.new_session(&request).await,
                Err(ControlError::Invalid(actual)) if actual == reason
            ));
        }
        #[cfg(unix)]
        assert!(matches!(
            control
                .new_session(&NewSessionRequest::new(file.join("child")))
                .await,
            Err(ControlError::Invalid(
                InvalidSessionRequest::CwdNotDirectory
            ))
        ));
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
    fn load_classifies_client_identity_and_lease_failures() {
        let (directory, store, control) = control();
        let invalid_id = SessionId::new("../outside");
        let invalid = LoadSessionRequest::new(invalid_id, directory.path());
        assert!(matches!(
            control.prepare_load(&invalid),
            Err(ControlError::Invalid(
                InvalidSessionRequest::InvalidSessionId
            ))
        ));
        assert!(!directory.path().join("outside.run.lock").exists());

        let missing_id = SessionId::new("missing-session");
        let missing = LoadSessionRequest::new(missing_id.clone(), directory.path());
        let error = control.prepare_load(&missing).err().unwrap();
        assert!(matches!(error, ControlError::SessionNotFound(_)));
        let protocol: ProtocolError = error.into();
        assert_eq!(
            protocol.code,
            agent_client_protocol::ErrorCode::RESOURCE_NOT_FOUND
        );
        assert_eq!(
            protocol.data,
            Some(serde_json::json!({"sessionId": missing_id}))
        );

        let (session, session_id) = persisted_session(&store, directory.path(), &[]);
        let _lease = store.try_acquire_run_lease(session.id()).unwrap();
        let active = LoadSessionRequest::new(session_id, directory.path());
        assert!(matches!(
            control.prepare_load(&active),
            Err(ControlError::Invalid(
                InvalidSessionRequest::ActiveElsewhere
            ))
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn load_rebuilds_tools_from_client_stdio_declarations() {
        let (directory, store, control) = control();
        let (_session, session_id) = persisted_session(&store, directory.path(), &[]);
        let script = concat!(
            "read initialize; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,",
            "\"result\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},",
            "\"serverInfo\":{\"name\":\"fixture\",\"version\":\"1\"}}}'; ",
            "read initialized; read list; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,",
            "\"result\":{\"tools\":[{\"name\":\"echo\",\"inputSchema\":{}}]}}'; ",
            "read until_closed"
        );
        let request =
            LoadSessionRequest::new(session_id.clone(), directory.path()).mcp_servers(vec![
                McpServer::Stdio(
                    McpServerStdio::new("fixture", "/bin/sh")
                        .args(vec!["-c".into(), script.into()]),
                ),
            ]);

        control
            .prepare_load_with_mcp(&request)
            .await
            .unwrap()
            .complete_with(
                |_| Ok::<_, ProtocolError>(()),
                |_| Ok::<_, ProtocolError>(()),
            )
            .unwrap();
        let guard = control.begin_prompt(&session_id).unwrap();
        assert!(guard.tools().get("read_file").is_some());
        assert!(guard.tools().get("mcp_s0_t0_echo").is_some());
    }

    #[tokio::test]
    async fn load_rejects_invalid_mcp_before_acquiring_the_session_lease() {
        let (directory, store, control) = control();
        let (session, session_id) = persisted_session(&store, directory.path(), &[]);
        let request = LoadSessionRequest::new(session_id, directory.path())
            .mcp_servers(vec![McpServer::Stdio(McpServerStdio::new("", "/bin/echo"))]);

        assert!(matches!(
            control.prepare_load_with_mcp(&request).await,
            Err(ControlError::Invalid(
                InvalidSessionRequest::InvalidMcpServers
            ))
        ));
        store.try_acquire_run_lease(session.id()).unwrap();
    }

    #[tokio::test]
    async fn async_load_classifies_an_already_registered_session() {
        let (directory, _store, control) = control();
        let created = control
            .new_session(&NewSessionRequest::new(directory.path()))
            .await
            .unwrap();
        let request = LoadSessionRequest::new(created.session_id, directory.path());

        assert!(matches!(
            control.prepare_load_with_mcp(&request).await,
            Err(ControlError::Invalid(
                InvalidSessionRequest::AlreadyRegistered
            ))
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn failed_load_mcp_startup_releases_the_prepared_lease() {
        let (directory, store, control) = control();
        let (session, session_id) = persisted_session(&store, directory.path(), &[]);
        let request = LoadSessionRequest::new(session_id, directory.path()).mcp_servers(vec![
            McpServer::Stdio(
                McpServerStdio::new("broken", "/bin/sh")
                    .args(vec!["-c".into(), "read initialize; exit 1".into()]),
            ),
        ]);

        assert!(matches!(
            control.prepare_load_with_mcp(&request).await,
            Err(ControlError::Internal(_))
        ));
        store.try_acquire_run_lease(session.id()).unwrap();
    }

    #[tokio::test]
    async fn each_load_request_replaces_the_active_additional_roots() {
        let (directory, _store, control) = control();
        let created = control
            .new_session(
                &NewSessionRequest::new(directory.path())
                    .additional_directories(vec![directory.path().join("create-root")]),
            )
            .await
            .unwrap();
        assert!(control.sessions.close(&created.session_id));

        let replacement = vec![directory.path().join("load-root")];
        control
            .prepare_load(
                &LoadSessionRequest::new(created.session_id.clone(), directory.path())
                    .additional_directories(replacement.clone()),
            )
            .unwrap()
            .complete_with(|_| Ok(()), |_| Ok(()))
            .unwrap();
        let guard = control.begin_prompt(&created.session_id).unwrap();
        assert_eq!(guard.additional_directories(), replacement);
        drop(guard);
        assert!(control.sessions.close(&created.session_id));

        control
            .prepare_load(&LoadSessionRequest::new(
                created.session_id.clone(),
                directory.path(),
            ))
            .unwrap()
            .complete_with(|_| Ok(()), |_| Ok(()))
            .unwrap();
        let guard = control.begin_prompt(&created.session_id).unwrap();
        assert!(guard.additional_directories().is_empty());
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
                responses_continuation: None,
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
                responses_continuation: None,
            },
        ];
        let (_session, session_id) = persisted_session(&store, directory.path(), &records);
        let events = RefCell::new(Vec::new());
        let roots = vec![
            directory.path().join("replacement-beta"),
            directory.path().join("replacement-alpha"),
        ];
        let request = LoadSessionRequest::new(session_id.clone(), directory.path())
            .additional_directories(roots.clone());

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
        assert_eq!(guard.additional_directories(), roots);
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
        let relative_root = LoadSessionRequest::new(session_id.clone(), directory.path())
            .additional_directories(vec![PathBuf::from("relative-extra")]);
        assert!(matches!(
            control.prepare_load(&relative_root),
            Err(ControlError::Invalid(
                InvalidSessionRequest::AdditionalDirectoryNotAbsolute
            ))
        ));
        store
            .try_acquire_run_lease(&session_id.to_string())
            .unwrap();

        let wrong_cwd = LoadSessionRequest::new(session_id.clone(), other.path());
        assert!(matches!(
            control.prepare_load(&wrong_cwd),
            Err(ControlError::Invalid(InvalidSessionRequest::CwdMismatch))
        ));
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
            responses_continuation: None,
        }];
        let (_corrupt, corrupt_id) = persisted_session(&store, directory.path(), &corrupt_records);
        let request = LoadSessionRequest::new(corrupt_id.clone(), directory.path());
        let error = control.prepare_load(&request).err().unwrap();
        assert!(matches!(error, ControlError::Internal(_)));
        let protocol: ProtocolError = error.into();
        assert_eq!(
            protocol.code,
            agent_client_protocol::ErrorCode::INTERNAL_ERROR
        );
        assert_eq!(protocol.data, None);
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
                    responses_continuation: None,
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
                            return Err(ProtocolError::internal_error().data("notification closed"));
                        }
                        Ok(())
                    },
                    |_| {
                        events.borrow_mut().push("response".to_string());
                        if fail_response {
                            return Err(ProtocolError::internal_error().data("response closed"));
                        }
                        Ok(())
                    },
                )
                .err()
                .unwrap();

            let error: ProtocolError = error.into();
            assert_eq!(error.code, agent_client_protocol::ErrorCode::INTERNAL_ERROR);
            assert_eq!(
                error.data,
                Some(serde_json::json!(if fail_response {
                    "response closed"
                } else {
                    "notification closed"
                }))
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

    #[cfg(unix)]
    #[tokio::test]
    async fn new_session_starts_mcp_tools_before_persisting() {
        let (directory, store, control) = control();
        let script = concat!(
            "test \"$ICODE_MCP_PRIVATE_TEST\" = 'not-for-transcript' || exit 2; ",
            "pwd > mcp-cwd.txt; ",
            "read initialize; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,",
            "\"result\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},",
            "\"serverInfo\":{\"name\":\"fixture\",\"version\":\"1\"}}}'; ",
            "read initialized; read list; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,",
            "\"result\":{\"tools\":[{\"name\":\"echo\",\"inputSchema\":{}}]}}'; ",
            "read until_closed"
        );
        let request = NewSessionRequest::new(directory.path()).mcp_servers(vec![McpServer::Stdio(
            McpServerStdio::new("fixture", "/bin/sh")
                .args(vec!["-c".into(), script.into()])
                .env(vec![EnvVariable::new(
                    "ICODE_MCP_PRIVATE_TEST",
                    "not-for-transcript",
                )]),
        )]);

        let response = control.new_session(&request).await.unwrap();
        let guard = control.begin_prompt(&response.session_id).unwrap();
        assert!(guard.tools().get("read_file").is_some());
        assert!(guard.tools().get("mcp_s0_t0_echo").is_some());
        assert_eq!(
            std::fs::canonicalize(
                std::fs::read_to_string(directory.path().join("mcp-cwd.txt"))
                    .unwrap()
                    .trim()
            )
            .unwrap(),
            std::fs::canonicalize(directory.path()).unwrap()
        );
        let id = response.session_id.to_string();
        assert!(store.load(&id).is_ok());
        let transcript = std::fs::read_to_string(store.path(&id)).unwrap();
        assert!(!transcript.contains("ICODE_MCP_PRIVATE_TEST"));
        assert!(!transcript.contains("not-for-transcript"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn failed_mcp_startup_creates_no_persistent_session() {
        let (directory, store, control) = control();
        let request = NewSessionRequest::new(directory.path()).mcp_servers(vec![McpServer::Stdio(
            McpServerStdio::new("broken", "/bin/sh")
                .args(vec!["-c".into(), "read initialize; exit 1".into()]),
        )]);

        assert!(matches!(
            control.new_session(&request).await,
            Err(ControlError::Internal(_))
        ));
        assert!(store.list().unwrap().is_empty());
    }

    #[tokio::test]
    async fn cancel_only_signals_the_matching_active_prompt() {
        let (directory, _store, control) = control();
        let response = control
            .new_session(&NewSessionRequest::new(directory.path()))
            .await
            .unwrap();
        let unknown = CancelNotification::new(SessionId::new("unknown"));
        assert!(!control.cancel(&unknown));

        let missing = control
            .begin_prompt(&SessionId::new("missing"))
            .err()
            .expect("missing session must be rejected");
        assert!(matches!(missing, ControlError::SessionNotFound(_)));

        let mut guard = control.begin_prompt(&response.session_id).unwrap();
        assert!(matches!(
            control.begin_prompt(&response.session_id),
            Err(ControlError::Invalid(InvalidSessionRequest::PromptActive))
        ));
        assert!(control.cancel(&CancelNotification::new(response.session_id.clone())));
        tokio::time::timeout(std::time::Duration::from_secs(1), guard.cancelled())
            .await
            .expect("cancel signal was not delivered");
    }
}
