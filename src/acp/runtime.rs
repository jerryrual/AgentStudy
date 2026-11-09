//! Shared construction policy for ACP prompt agents.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming ACP prompt handler")
)]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::agent::{Agent, AgentKind, AgentOptions};
use crate::interaction::{Approver, NoAsker};
use crate::model::ModelClient;
use crate::tools::ToolRegistry;

/// Immutable dependencies shared by prompt turns on one ACP service.
pub(crate) struct PromptRuntime {
    client: Arc<ModelClient>,
    agent_kind: AgentKind,
    max_tool_rounds: usize,
    phase_four_side_call_token_budget: i64,
    tool_timeout: Duration,
}

impl PromptRuntime {
    /// Build the baseline ACP runtime with builtin tools and fail-closed mutation approval.
    pub(crate) fn new(
        client: Arc<ModelClient>,
        agent_kind: AgentKind,
        max_tool_rounds: usize,
        phase_four_side_call_token_budget: i64,
    ) -> Self {
        Self {
            client,
            agent_kind,
            max_tool_rounds,
            phase_four_side_call_token_budget,
            tool_timeout: Duration::from_secs(120),
        }
    }

    /// Construct a fresh agent whose project memory and mutable state belong to `cwd`.
    pub(crate) fn agent_for(
        &self,
        cwd: &Path,
        tools: Arc<ToolRegistry>,
        approver: Arc<dyn Approver>,
    ) -> Agent {
        Agent::new(
            Arc::clone(&self.client),
            tools,
            approver,
            Arc::new(NoAsker),
            AgentOptions {
                kind: self.agent_kind,
                cwd: cwd.to_path_buf(),
                max_rounds: self.max_tool_rounds,
                tool_timeout: self.tool_timeout,
            },
        )
    }

    pub(crate) fn context_window(&self) -> Option<u32> {
        self.client.context_window()
    }

    /// Return the durable Phase 4 side-call budget used by each prompt sink.
    pub(crate) fn phase_four_side_call_token_budget(&self) -> i64 {
        self.phase_four_side_call_token_budget
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::config::ModelProfile;
    use crate::interaction::DenyAll;

    use super::*;

    #[test]
    fn constructs_fresh_agents_with_session_specific_project_memory() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        fs::write(first.path().join("AGENTS.md"), "first workspace rule").unwrap();
        fs::write(second.path().join("AGENTS.md"), "second workspace rule").unwrap();
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_key = "test-key".to_string();
        profile.base_url = "http://127.0.0.1:1".to_string();
        profile.model = "mock-model".to_string();
        let runtime = PromptRuntime::new(
            Arc::new(ModelClient::from_profile(&profile).unwrap()),
            AgentKind::Code,
            17,
            2048,
        );

        let first_agent = runtime.agent_for(
            first.path(),
            Arc::new(ToolRegistry::builtin()),
            Arc::new(DenyAll),
        );
        let second_agent = runtime.agent_for(
            second.path(),
            Arc::new(ToolRegistry::builtin()),
            Arc::new(DenyAll),
        );

        assert_eq!(first_agent.kind(), AgentKind::Code);
        assert_eq!(first_agent.model(), "mock-model");
        assert!(first_agent.system_prompt().contains("first workspace rule"));
        assert!(!first_agent
            .system_prompt()
            .contains("second workspace rule"));
        assert!(second_agent
            .system_prompt()
            .contains("second workspace rule"));
        assert!(!second_agent
            .system_prompt()
            .contains("first workspace rule"));
        assert_eq!(runtime.phase_four_side_call_token_budget(), 2048);
    }
}
