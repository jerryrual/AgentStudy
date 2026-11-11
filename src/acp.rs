//! Agent Client Protocol integration foundations.

use std::sync::Arc;

use agent_client_protocol::{Agent as AcpAgent, ConnectTo, Stdio};
use anyhow::{Context, Result};

use crate::agent::AgentKind;
use crate::config::{ModelProfile, Settings};
use crate::model::ModelClient;
use crate::session::SessionStore;

pub(crate) mod approval;
pub(crate) mod callbacks;
mod client_fs;
pub(crate) mod dispatcher;
pub(crate) mod error;
pub(crate) mod event_mapping;
pub(crate) mod history_replay;
pub(crate) mod mcp;
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
    serve_with_setup_hook(settings, profile, agent_kind, store, transport, None).await
}

async fn serve_with_setup_hook(
    settings: Settings,
    profile: ModelProfile,
    agent_kind: AgentKind,
    store: SessionStore,
    transport: impl ConnectTo<AcpAgent>,
    prompt_setup_hook: Option<callbacks::PromptSetupHook>,
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
    callbacks::callback_bundle_with_setup_hook(control, runtime, prompt_setup_hook)
        .connect_to(transport)
        .await
        .context("ACP connection failed")
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use agent_client_protocol::schema::ProtocolVersion;
    use agent_client_protocol::v1::schema::{
        ContentBlock, InitializeRequest, InitializeResponse, NewSessionRequest, NewSessionResponse,
        PromptRequest, SessionId,
    };
    use agent_client_protocol::{Channel, Client, Error as ProtocolError};
    use tempfile::TempDir;
    use tokio::io::AsyncReadExt;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn provider_phase_prompt_eof_releases_connection_scoped_lease() {
        let directory = TempDir::new().unwrap();
        let store = SessionStore::with_root(directory.path().join("sessions"));
        let cwd = directory.path().to_path_buf();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (provider_entered_tx, provider_entered_rx) = tokio::sync::oneshot::channel();
        let (release_provider_tx, release_provider_rx) = tokio::sync::oneshot::channel();
        let provider = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let read = socket.read(&mut request).await.unwrap();
            assert!(read > 0, "provider received an empty HTTP request");
            provider_entered_tx.send(()).unwrap();
            let _ = release_provider_rx.await;
        });
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_key = "test-key".to_string();
        profile.base_url = format!("http://{address}");
        profile.model = "mock-model".to_string();
        let created_id = Arc::new(Mutex::new(None::<SessionId>));
        let received_id = Arc::clone(&created_id);
        let (transport, peer) = Channel::duplex();

        let server = serve(
            Settings::default(),
            profile,
            AgentKind::Code,
            store.clone(),
            transport,
        );
        let client = Client.v1().connect_with(peer, async move |connection| {
            let _: InitializeResponse = connection
                .send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let created: NewSessionResponse = connection
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            *received_id.lock().unwrap() = Some(created.session_id.clone());
            let prompt = connection
                .send_request(PromptRequest::new(
                    created.session_id,
                    vec![ContentBlock::from("remain active at provider")],
                ))
                .block_task();
            connection.spawn(async move {
                let _ = prompt.await;
                Ok(())
            })?;
            provider_entered_rx.await.unwrap();
            Ok::<(), ProtocolError>(())
        });

        let (server_result, client_result) =
            tokio::time::timeout(std::time::Duration::from_secs(10), async {
                tokio::join!(server, client)
            })
            .await
            .expect("provider-phase EOF test timed out");
        server_result.unwrap();
        client_result.unwrap();

        let session_id = created_id.lock().unwrap().clone().unwrap().to_string();
        let _lease = store
            .try_acquire_run_lease(&session_id)
            .expect("provider-phase EOF must release the session run lease");
        release_provider_tx.send(()).unwrap();
        provider.await.unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn active_prompt_eof_waits_for_detached_setup_before_releasing_lease() {
        let directory = TempDir::new().unwrap();
        let store = SessionStore::with_root(directory.path().join("sessions"));
        let cwd = directory.path().to_path_buf();
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_key = "test-key".to_string();
        profile.base_url = "http://127.0.0.1:1".to_string();
        profile.model = "mock-model".to_string();
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let release_rx = Mutex::new(release_rx);
        let setup_hook: callbacks::PromptSetupHook = Arc::new(move || {
            entered_tx.send(()).unwrap();
            release_rx.lock().unwrap().recv().unwrap();
        });
        let created_id = Arc::new(Mutex::new(None::<SessionId>));
        let received_id = Arc::clone(&created_id);
        let (transport, peer) = Channel::duplex();

        let server = serve_with_setup_hook(
            Settings::default(),
            profile,
            AgentKind::Code,
            store.clone(),
            transport,
            Some(setup_hook),
        );
        let client = Client.v1().connect_with(peer, async move |connection| {
            let _: InitializeResponse = connection
                .send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let created: NewSessionResponse = connection
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            *received_id.lock().unwrap() = Some(created.session_id.clone());
            let prompt = connection
                .send_request(PromptRequest::new(
                    created.session_id,
                    vec![ContentBlock::from("remain active")],
                ))
                .block_task();
            connection.spawn(async move {
                let _ = prompt.await;
                Ok(())
            })?;
            tokio::task::spawn_blocking(move || entered_rx.recv().unwrap())
                .await
                .unwrap();
            Ok::<(), ProtocolError>(())
        });

        let (server_result, client_result) =
            tokio::time::timeout(std::time::Duration::from_secs(10), async {
                tokio::join!(server, client)
            })
            .await
            .expect("active-prompt EOF test timed out");
        server_result.unwrap();
        client_result.unwrap();

        let session_id = created_id.lock().unwrap().clone().unwrap().to_string();
        assert!(store.try_acquire_run_lease(&session_id).is_err());
        release_tx.send(()).unwrap();
        let _lease = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                if let Ok(lease) = store.try_acquire_run_lease(&session_id) {
                    break lease;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("detached setup worker did not release its run lease");
    }

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
