//! Agent Client Protocol integration foundations.

use std::sync::Arc;

use agent_client_protocol::{Agent as AcpAgent, ConnectTo, Stdio};
use anyhow::{Context, Result};

use crate::agent::AgentKind;
use crate::config::{ModelProfile, Settings};
use crate::model::ModelClient;
use crate::session::SessionStore;

pub(crate) mod callbacks;
pub(crate) mod dispatcher;
pub(crate) mod error;
pub(crate) mod event_mapping;
pub(crate) mod history_replay;
pub(crate) mod prompt;
pub(crate) mod runtime;
pub(crate) mod session_control;
pub(crate) mod sessions;

/// Serve one ACP client over stdin/stdout until the transport closes.
pub(crate) async fn serve_stdio(
    settings: Settings,
    profile: ModelProfile,
    agent_kind: AgentKind,
    store: SessionStore,
) -> Result<()> {
    serve(settings, profile, agent_kind, store, Stdio::new()).await
}

/// Serve one ACP client over an injected transport until it closes.
async fn serve(
    settings: Settings,
    profile: ModelProfile,
    agent_kind: AgentKind,
    store: SessionStore,
    transport: impl ConnectTo<AcpAgent>,
) -> Result<()> {
    let client = tokio::task::spawn_blocking(move || ModelClient::from_profile(&profile))
        .await
        .context("ACP model configuration task failed")??;
    let model = client.model().to_string();
    let sessions = Arc::new(sessions::SessionRegistry::new(store));
    let control = Arc::new(session_control::SessionControl::new(
        sessions,
        model,
        agent_kind.as_str(),
    ));
    let runtime = Arc::new(runtime::PromptRuntime::new(
        Arc::new(client),
        agent_kind,
        settings.max_tool_rounds,
        settings.phase4_side_call_token_budget,
    ));
    callbacks::callback_bundle(control, runtime)
        .connect_to(transport)
        .await
        .context("ACP connection failed")
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use agent_client_protocol::schema::ProtocolVersion;
    use agent_client_protocol::v1::schema::{
        InitializeRequest, InitializeResponse, NewSessionRequest, NewSessionResponse, SessionId,
    };
    use agent_client_protocol::{Channel, Client, Error as ProtocolError};
    use tempfile::TempDir;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn clean_eof_releases_connection_scoped_session_leases() {
        let directory = TempDir::new().unwrap();
        let store = SessionStore::with_root(directory.path().join("sessions"));
        let cwd = directory.path().to_path_buf();
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_key = "test-key".to_string();
        profile.base_url = "http://127.0.0.1:1".to_string();
        profile.model = "mock-model".to_string();
        let settings = Settings::default();
        let created_id = Arc::new(Mutex::new(None::<SessionId>));
        let received_id = Arc::clone(&created_id);
        let (transport, peer) = Channel::duplex();

        let server = serve(settings, profile, AgentKind::Code, store.clone(), transport);
        let client = Client.v1().connect_with(peer, async move |connection| {
            let _: InitializeResponse = connection
                .send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let created: NewSessionResponse = connection
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            *received_id.lock().unwrap() = Some(created.session_id);
            Ok::<(), ProtocolError>(())
        });

        let (server_result, client_result) =
            tokio::time::timeout(std::time::Duration::from_secs(10), async {
                tokio::join!(server, client)
            })
            .await
            .expect("ACP clean EOF test timed out");
        server_result.unwrap();
        client_result.unwrap();

        let session_id = created_id.lock().unwrap().clone().unwrap();
        store
            .try_acquire_run_lease(&session_id.to_string())
            .expect("clean EOF must release the connection-scoped run lease");
    }
}
