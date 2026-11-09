//! Agent Client Protocol integration foundations.

use std::sync::Arc;

use agent_client_protocol::{ConnectTo, Stdio};
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
        .connect_to(Stdio::new())
        .await
        .context("ACP stdio connection failed")
}
