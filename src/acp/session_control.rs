//! Side-effect boundaries for stable ACP session control requests.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming ACP protocol handlers")
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::error::{ControlError, ControlResult, InvalidSessionRequest};
use super::mcp;
use super::sessions::{
    BeginPromptError, CloseError, LoadError, PreparedLoad, PreparedRestore, PromptGuard,
    SessionRegistry, SetModeError,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::v1::schema::{
    AgentCapabilities, CancelNotification, CloseSessionRequest, CloseSessionResponse,
    DeleteSessionRequest, DeleteSessionResponse, InitializeRequest, InitializeResponse,
    ListSessionsRequest, ListSessionsResponse, LoadSessionRequest, LoadSessionResponse, McpServer,
    NewSessionRequest, NewSessionResponse, ResumeSessionRequest, ResumeSessionResponse,
    SessionAdditionalDirectoriesCapabilities, SessionCapabilities, SessionCloseCapabilities,
    SessionConfigOption, SessionConfigOptionCategory, SessionConfigOptionValue,
    SessionConfigSelectOption, SessionDeleteCapabilities, SessionId, SessionInfo,
    SessionListCapabilities, SessionNotification, SessionResumeCapabilities,
    SetSessionConfigOptionRequest, SetSessionConfigOptionResponse,
};
use agent_client_protocol::Error as ProtocolError;

use crate::agent::AgentKind;
use crate::session::RunLeaseError;

const MODE_CONFIG_ID: &str = "mode";

fn configured_agent_kind(agent: &str) -> AgentKind {
    AgentKind::parse(agent).unwrap_or_default()
}

fn mode_config_options(agent_kind: AgentKind) -> Vec<SessionConfigOption> {
    vec![SessionConfigOption::select(
        MODE_CONFIG_ID,
        "Agent mode",
        agent_kind.as_str(),
        vec![
            SessionConfigSelectOption::new(
                AgentKind::Code.as_str(),
                AgentKind::Code.display_name(),
            ),
            SessionConfigSelectOption::new(AgentKind::Qa.as_str(), AgentKind::Qa.display_name()),
        ],
    )
    .category(SessionConfigOptionCategory::Mode)]
}

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

/// A prepared resume that owns its lease until the response is admitted.
pub(crate) struct ResumeTransaction {
    prepared: PreparedRestore,
}

#[derive(Clone, Copy)]
enum RestoreMode {
    Load,
    Resume,
}

enum PreparedTransaction {
    Load(PreparedLoad),
    Resume(PreparedRestore),
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
                        .additional_directories(SessionAdditionalDirectoriesCapabilities::new())
                        .close(SessionCloseCapabilities::new())
                        .delete(SessionDeleteCapabilities::new())
                        .list(SessionListCapabilities::new())
                        .resume(SessionResumeCapabilities::new()),
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
        Ok(NewSessionResponse::new(session_id)
            .config_options(mode_config_options(configured_agent_kind(&self.agent))))
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
        match self
            .prepare_restore_with_mcp(
                &request.session_id,
                &request.cwd,
                &request.additional_directories,
                &request.mcp_servers,
                RestoreMode::Load,
            )
            .await?
        {
            PreparedTransaction::Load(prepared) => Ok(LoadTransaction { prepared }),
            PreparedTransaction::Resume(_) => unreachable!("load requested a resume restore"),
        }
    }

    /// Validate and restore a session without replaying historical notifications.
    pub(crate) async fn prepare_resume_with_mcp(
        &self,
        request: &ResumeSessionRequest,
    ) -> ControlResult<ResumeTransaction> {
        match self
            .prepare_restore_with_mcp(
                &request.session_id,
                &request.cwd,
                &request.additional_directories,
                &request.mcp_servers,
                RestoreMode::Resume,
            )
            .await?
        {
            PreparedTransaction::Resume(prepared) => Ok(ResumeTransaction { prepared }),
            PreparedTransaction::Load(_) => unreachable!("resume requested a load restore"),
        }
    }

    async fn prepare_restore_with_mcp(
        &self,
        session_id: &SessionId,
        requested_cwd: &Path,
        requested_directories: &[PathBuf],
        mcp_servers: &[McpServer],
        mode: RestoreMode,
    ) -> ControlResult<PreparedTransaction> {
        let additional_directories = validated_additional_directories(requested_directories)?;
        let specs = mcp::stdio_specs(mcp_servers)
            .map_err(|_| ControlError::invalid(InvalidSessionRequest::InvalidMcpServers))?;
        let requested_cwd = requested_cwd.to_path_buf();
        let cwd = tokio::task::spawn_blocking(move || {
            validated_cwd(&requested_cwd).map(|cwd| cwd.to_string())
        })
        .await
        .map_err(ControlError::internal)??;
        if self.sessions.is_registered(session_id) {
            return Err(ControlError::invalid(
                InvalidSessionRequest::AlreadyRegistered,
            ));
        }
        let registry = Arc::downgrade(&self.sessions);
        let store = self.sessions.store();
        let detached_session_id = session_id.clone();
        let expected_cwd = cwd.clone();
        let mut prepared = tokio::task::spawn_blocking(move || match mode {
            RestoreMode::Load => SessionRegistry::prepare_load_detached(
                registry,
                store,
                detached_session_id,
                expected_cwd,
            )
            .map(PreparedTransaction::Load),
            RestoreMode::Resume => SessionRegistry::prepare_resume_detached(
                registry,
                store,
                detached_session_id,
                expected_cwd,
            )
            .map(PreparedTransaction::Resume),
        })
        .await
        .map_err(ControlError::internal)?
        .map_err(|error| map_load_error(error, session_id))?;
        let tools = mcp::start_tool_registry(&specs, Path::new(&cwd))
            .await
            .map_err(ControlError::internal)?;
        match &mut prepared {
            PreparedTransaction::Load(prepared) => {
                prepared.set_additional_directories(additional_directories);
                prepared.set_tools(tools);
            }
            PreparedTransaction::Resume(prepared) => {
                prepared.set_additional_directories(additional_directories);
                prepared.set_tools(tools);
            }
        }
        Ok(prepared)
    }

    /// Persist one supported session config option for subsequent prompts.
    pub(crate) async fn set_config_option(
        &self,
        request: &SetSessionConfigOptionRequest,
    ) -> ControlResult<SetSessionConfigOptionResponse> {
        if request.config_id.0.as_ref() != MODE_CONFIG_ID {
            return Err(ControlError::invalid(InvalidSessionRequest::ConfigId));
        }
        let SessionConfigOptionValue::ValueId { value } = &request.value else {
            return Err(ControlError::invalid(InvalidSessionRequest::ConfigValue));
        };
        let agent_kind = match value.0.as_ref() {
            "code" => AgentKind::Code,
            "qa" => AgentKind::Qa,
            _ => return Err(ControlError::invalid(InvalidSessionRequest::ConfigValue)),
        };
        let sessions = Arc::clone(&self.sessions);
        let session_id = request.session_id.clone();
        tokio::task::spawn_blocking(move || sessions.set_agent_kind(&session_id, agent_kind))
            .await
            .map_err(ControlError::internal)?
            .map_err(|error| match error {
                SetModeError::SessionNotFound => {
                    ControlError::SessionNotFound(request.session_id.clone())
                }
                SetModeError::Loading => {
                    ControlError::invalid(InvalidSessionRequest::ConfigLoading)
                }
                SetModeError::ActivePrompt => {
                    ControlError::invalid(InvalidSessionRequest::ConfigActive)
                }
                SetModeError::Persist(error) => ControlError::internal(error),
            })?;
        Ok(SetSessionConfigOptionResponse::new(mode_config_options(
            agent_kind,
        )))
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

    /// List durable sessions with an optional exact working-directory filter.
    pub(crate) async fn list_sessions(
        &self,
        request: &ListSessionsRequest,
    ) -> ControlResult<ListSessionsResponse> {
        if request.cursor.is_some() {
            return Err(ControlError::invalid(InvalidSessionRequest::ListCursor));
        }
        if request.cwd.as_deref().is_some_and(|cwd| !cwd.is_absolute()) {
            return Err(ControlError::invalid(InvalidSessionRequest::CwdNotAbsolute));
        }
        let cwd = request.cwd.clone();
        let store = self.sessions.store();
        let summaries = tokio::task::spawn_blocking(move || store.list())
            .await
            .map_err(ControlError::internal)?
            .map_err(ControlError::internal)?;
        let sessions = summaries
            .into_iter()
            .filter(|summary| {
                let summary_cwd = Path::new(&summary.cwd);
                summary_cwd.is_absolute()
                    && cwd.as_deref().is_none_or(|filter| summary_cwd == filter)
            })
            .map(|summary| {
                SessionInfo::new(SessionId::new(summary.id), PathBuf::from(summary.cwd))
                    .title(summary.preview)
            })
            .collect();
        Ok(ListSessionsResponse::new(sessions))
    }

    /// Cancel active work, drain it, then delete the durable session and archive.
    pub(crate) async fn delete(
        &self,
        request: &DeleteSessionRequest,
    ) -> ControlResult<DeleteSessionResponse> {
        match self.sessions.close(&request.session_id).await {
            Ok(()) | Err(CloseError::SessionNotFound) => {}
            Err(CloseError::Loading) => {
                return Err(ControlError::invalid(InvalidSessionRequest::DeleteLoading));
            }
        }
        let store = self.sessions.store();
        let session_id = request.session_id.clone();
        let id = session_id.to_string();
        tokio::task::spawn_blocking(move || store.delete(&id))
            .await
            .map_err(ControlError::internal)?
            .map_err(|error| map_delete_error(error, session_id))?;
        Ok(DeleteSessionResponse::new())
    }

    /// Cancel active work, drain it, and release one registered session.
    pub(crate) async fn close(
        &self,
        request: &CloseSessionRequest,
    ) -> ControlResult<CloseSessionResponse> {
        self.sessions
            .close(&request.session_id)
            .await
            .map_err(|error| match error {
                CloseError::SessionNotFound => {
                    ControlError::SessionNotFound(request.session_id.clone())
                }
                CloseError::Loading => ControlError::invalid(InvalidSessionRequest::CloseLoading),
            })?;
        Ok(CloseSessionResponse::new())
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
        let agent_kind = self.prepared.agent_kind();
        let updates = self.prepared.updates().to_vec();
        let pending = self
            .prepared
            .activate()
            .map_err(|error| map_load_error(error, &session_id))?;
        for update in updates {
            notify(SessionNotification::new(session_id.clone(), update))
                .map_err(ControlError::Protocol)?;
        }
        respond(LoadSessionResponse::new().config_options(mode_config_options(agent_kind)))
            .map_err(ControlError::Protocol)?;
        pending.commit();
        Ok(())
    }
}

impl ResumeTransaction {
    /// Admit the response before making the resumed session prompt-visible.
    pub(crate) fn complete_with(
        self,
        respond: impl FnOnce(ResumeSessionResponse) -> Result<(), ProtocolError>,
    ) -> ControlResult<()> {
        let session_id = self.prepared.session_id();
        let agent_kind = self.prepared.agent_kind();
        let pending = self
            .prepared
            .activate()
            .map_err(|error| map_load_error(error, &session_id))?;
        respond(ResumeSessionResponse::new().config_options(mode_config_options(agent_kind)))
            .map_err(ControlError::Protocol)?;
        pending.commit();
        Ok(())
    }
}

fn map_delete_error(error: RunLeaseError, session_id: SessionId) -> ControlError {
    match error {
        RunLeaseError::InvalidSessionId(_) => {
            ControlError::invalid(InvalidSessionRequest::InvalidSessionId)
        }
        RunLeaseError::SessionNotFound(_) => ControlError::SessionNotFound(session_id),
        RunLeaseError::ActiveElsewhere { .. } => {
            ControlError::invalid(InvalidSessionRequest::ActiveElsewhere)
        }
        RunLeaseError::Internal(error) => ControlError::Internal(error),
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
                    SessionCapabilities::new()
                        .additional_directories(SessionAdditionalDirectoriesCapabilities::new(),)
                        .close(SessionCloseCapabilities::new())
                        .delete(SessionDeleteCapabilities::new())
                        .list(SessionListCapabilities::new())
                        .resume(SessionResumeCapabilities::new()),
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
    async fn list_sessions_projects_durable_summaries_and_validates_filters() {
        let (directory, store, control) = control();
        let primary_cwd = directory.path().join("primary");
        let other_cwd = directory.path().join("other");
        let primary = store
            .create(primary_cwd.to_str().unwrap(), "model-a", "build")
            .unwrap();
        primary
            .append(&Record::User {
                text: "Primary title\nmore".to_string(),
            })
            .unwrap();
        let other = store
            .create(other_cwd.to_str().unwrap(), "model-b", "plan")
            .unwrap();

        let listed = control
            .list_sessions(&ListSessionsRequest::new())
            .await
            .unwrap();
        assert_eq!(listed.sessions.len(), 2);
        assert_eq!(listed.next_cursor, None);
        let projected = listed
            .sessions
            .iter()
            .find(|session| session.session_id.to_string() == primary.id())
            .unwrap();
        assert_eq!(projected.cwd, primary_cwd);
        assert_eq!(projected.title.as_deref(), Some("Primary title"));
        assert!(projected.additional_directories.is_empty());
        assert_eq!(projected.updated_at, None);

        let filtered = control
            .list_sessions(&ListSessionsRequest::new().cwd(other_cwd.clone()))
            .await
            .unwrap();
        assert_eq!(filtered.sessions.len(), 1);
        assert_eq!(filtered.sessions[0].session_id.to_string(), other.id());
        assert_eq!(filtered.sessions[0].cwd, other_cwd);

        let relative = control
            .list_sessions(&ListSessionsRequest::new().cwd("relative"))
            .await
            .unwrap_err();
        assert!(matches!(
            relative,
            ControlError::Invalid(InvalidSessionRequest::CwdNotAbsolute)
        ));
        let cursor = control
            .list_sessions(&ListSessionsRequest::new().cursor("unexpected"))
            .await
            .unwrap_err();
        assert!(matches!(
            cursor,
            ControlError::Invalid(InvalidSessionRequest::ListCursor)
        ));
    }

    #[tokio::test]
    async fn close_cancels_drains_and_preserves_the_persisted_session() {
        let (directory, store, control) = control();
        let response = control
            .new_session(&NewSessionRequest::new(directory.path()))
            .await
            .unwrap();
        let session_id = response.session_id;
        let id = session_id.to_string();
        let mut guard = control.begin_prompt(&session_id).unwrap();
        let close_control = Arc::new(control);
        let close_request = CloseSessionRequest::new(session_id.clone());
        let close_task = {
            let close_control = Arc::clone(&close_control);
            tokio::spawn(async move { close_control.close(&close_request).await })
        };

        guard.cancelled().await;
        assert!(matches!(
            close_control.begin_prompt(&session_id),
            Err(ControlError::SessionNotFound(_))
        ));
        assert!(!close_task.is_finished());
        drop(guard);
        assert_eq!(
            close_task.await.unwrap().unwrap(),
            CloseSessionResponse::new()
        );
        store.try_acquire_run_lease(&id).unwrap();
        assert!(store.path(&id).exists());

        let error = close_control
            .close(&CloseSessionRequest::new(session_id.clone()))
            .await
            .unwrap_err();
        let protocol: ProtocolError = error.into();
        assert_eq!(
            protocol,
            agent_client_protocol::v1::schema::Error::session_not_found(session_id).into()
        );
    }

    #[tokio::test]
    async fn delete_cancels_drains_and_removes_the_durable_session() {
        let (directory, store, control) = control();
        let response = control
            .new_session(&NewSessionRequest::new(directory.path()))
            .await
            .unwrap();
        let session_id = response.session_id;
        let id = session_id.to_string();
        let archive_root = store.archive_root(&id).unwrap();
        std::fs::create_dir_all(&archive_root).unwrap();
        std::fs::write(archive_root.join("record.json"), "owned").unwrap();
        let mut guard = control.begin_prompt(&session_id).unwrap();
        let delete_control = Arc::new(control);
        let delete_request = DeleteSessionRequest::new(session_id.clone());
        let delete_task = {
            let delete_control = Arc::clone(&delete_control);
            tokio::spawn(async move { delete_control.delete(&delete_request).await })
        };

        guard.cancelled().await;
        assert!(matches!(
            delete_control.begin_prompt(&session_id),
            Err(ControlError::SessionNotFound(_))
        ));
        assert!(!delete_task.is_finished());
        drop(guard);
        assert_eq!(
            delete_task.await.unwrap().unwrap(),
            DeleteSessionResponse::new()
        );
        assert!(!store.path(&id).exists());
        assert!(!archive_root.exists());

        let error = delete_control
            .delete(&DeleteSessionRequest::new(session_id.clone()))
            .await
            .unwrap_err();
        let protocol: ProtocolError = error.into();
        assert_eq!(
            protocol,
            agent_client_protocol::v1::schema::Error::session_not_found(session_id).into()
        );
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

    #[cfg(unix)]
    #[tokio::test]
    async fn resume_rebuilds_tools_from_client_stdio_declarations() {
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
            ResumeSessionRequest::new(session_id.clone(), directory.path()).mcp_servers(vec![
                McpServer::Stdio(
                    McpServerStdio::new("fixture", "/bin/sh")
                        .args(vec!["-c".into(), script.into()]),
                ),
            ]);

        control
            .prepare_resume_with_mcp(&request)
            .await
            .unwrap()
            .complete_with(|_| Ok::<_, ProtocolError>(()))
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
        control.sessions.close(&created.session_id).await.unwrap();

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
        control.sessions.close(&created.session_id).await.unwrap();

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

    #[tokio::test]
    async fn session_mode_config_is_durable_and_applies_to_the_next_prompt() {
        let (directory, _store, control) = control();
        let created = control
            .new_session(&NewSessionRequest::new(directory.path()))
            .await
            .unwrap();
        assert_eq!(
            created.config_options,
            Some(mode_config_options(AgentKind::Code))
        );

        let select_qa =
            SetSessionConfigOptionRequest::new(created.session_id.clone(), MODE_CONFIG_ID, "qa");
        let response = control.set_config_option(&select_qa).await.unwrap();
        assert_eq!(response.config_options, mode_config_options(AgentKind::Qa));

        let guard = control.begin_prompt(&created.session_id).unwrap();
        assert_eq!(guard.agent_kind(), AgentKind::Qa);
        let active =
            SetSessionConfigOptionRequest::new(created.session_id.clone(), MODE_CONFIG_ID, "code");
        assert!(matches!(
            control.set_config_option(&active).await,
            Err(ControlError::Invalid(InvalidSessionRequest::ConfigActive))
        ));
        drop(guard);
        control.sessions.close(&created.session_id).await.unwrap();

        let transaction = control
            .prepare_resume_with_mcp(&ResumeSessionRequest::new(
                created.session_id.clone(),
                directory.path(),
            ))
            .await
            .unwrap();
        transaction
            .complete_with(|response| {
                assert_eq!(
                    response.config_options,
                    Some(mode_config_options(AgentKind::Qa))
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(
            control
                .begin_prompt(&created.session_id)
                .unwrap()
                .agent_kind(),
            AgentKind::Qa
        );

        let unknown =
            SetSessionConfigOptionRequest::new(created.session_id.clone(), "unknown", "qa");
        assert!(matches!(
            control.set_config_option(&unknown).await,
            Err(ControlError::Invalid(InvalidSessionRequest::ConfigId))
        ));
        let invalid =
            SetSessionConfigOptionRequest::new(created.session_id.clone(), MODE_CONFIG_ID, true);
        assert!(matches!(
            control.set_config_option(&invalid).await,
            Err(ControlError::Invalid(InvalidSessionRequest::ConfigValue))
        ));
        let missing =
            SetSessionConfigOptionRequest::new(SessionId::new("missing"), MODE_CONFIG_ID, "qa");
        assert!(matches!(
            control.set_config_option(&missing).await,
            Err(ControlError::SessionNotFound(_))
        ));
    }

    #[tokio::test]
    async fn resume_skips_replay_admits_response_before_ready_and_rolls_back_failure() {
        let (directory, store, control) = control();
        let duplicate = Record::Assistant {
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
        };
        let (session, session_id) =
            persisted_session(&store, directory.path(), std::slice::from_ref(&duplicate));
        let request = ResumeSessionRequest::new(session_id.clone(), directory.path());
        let transaction = control.prepare_resume_with_mcp(&request).await.unwrap();
        transaction
            .complete_with(|response| {
                assert_eq!(
                    response,
                    ResumeSessionResponse::new()
                        .config_options(mode_config_options(AgentKind::Code))
                );
                assert!(matches!(
                    control.begin_prompt(&session_id),
                    Err(ControlError::Invalid(InvalidSessionRequest::PromptLoading))
                ));
                Ok(())
            })
            .unwrap();
        drop(control.begin_prompt(&session_id).unwrap());
        control.sessions.close(&session_id).await.unwrap();

        assert!(matches!(
            control
                .prepare_load_with_mcp(&LoadSessionRequest::new(
                    session_id.clone(),
                    directory.path()
                ))
                .await,
            Err(ControlError::Internal(_))
        ));
        let failed = control.prepare_resume_with_mcp(&request).await.unwrap();
        assert!(matches!(
            failed.complete_with(|_| Err(ProtocolError::internal_error())),
            Err(ControlError::Protocol(_))
        ));
        assert!(!control.sessions.is_registered(&session_id));
        store.try_acquire_run_lease(session.id()).unwrap();
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
                    assert!(control.sessions.is_registered(&session_id));
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
                    assert!(control.sessions.is_registered(&session_id));
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
        assert_eq!(
            events[6],
            serde_json::json!({
                "response": {
                    "configOptions": mode_config_options(AgentKind::Code)
                }
            })
        );
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
