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
    NewSessionRequest, NewSessionResponse,
};
use anyhow::{ensure, Context, Result};

use super::sessions::SessionRegistry;

/// Validated session creation and cancellation shared by future wire handlers.
pub(crate) struct SessionControl {
    sessions: Arc<SessionRegistry>,
    model: String,
    agent: String,
}

impl SessionControl {
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

    /// Forward a fire-and-forget cancellation to the matching active prompt.
    pub(crate) fn cancel(&self, notification: &CancelNotification) -> bool {
        self.sessions.cancel(&notification.session_id)
    }
}

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
    use agent_client_protocol::v1::schema::{McpServer, McpServerStdio, SessionId};
    use tempfile::TempDir;

    use super::*;
    use crate::session::SessionStore;

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
