//! User interaction hooks used by the agent: approval of mutating operations
//! and questions addressed to the user.

use async_trait::async_trait;

use crate::model::ToolCall;

/// A mutating operation that needs consent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalRequest {
    /// Tool name, e.g. `bash`.
    pub tool: String,
    /// One-line description of what will happen.
    pub summary: String,
}

impl ApprovalRequest {
    pub fn new(tool: impl Into<String>, summary: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            summary: summary.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    /// Run this one invocation.
    AllowOnce,
    /// Run it and remember the choice for the rest of the session.
    AllowAlways,
    /// Skip the invocation.
    Deny,
}

/// Decides whether a mutating tool call may run.
#[async_trait]
pub trait Approver: Send + Sync {
    /// Decide using the legacy display-only approval request.
    async fn approve(&self, request: &ApprovalRequest) -> ApprovalDecision;

    /// Decide one exact provider tool call.
    ///
    /// Existing approvers inherit the display-only behavior. Protocol adapters
    /// can override this hook to retain the stable call id and raw arguments.
    async fn approve_tool(&self, call: &ToolCall, summary: &str) -> ApprovalDecision {
        self.approve(&ApprovalRequest::new(&call.name, summary))
            .await
    }
}

/// Approves everything, used by `icode run` (which mirrors iCode's bypass) and
/// by tests.
pub struct AllowAll;

#[async_trait]
impl Approver for AllowAll {
    async fn approve(&self, _request: &ApprovalRequest) -> ApprovalDecision {
        ApprovalDecision::AllowOnce
    }
}

/// Refuses every mutating call.
pub struct DenyAll;

#[async_trait]
impl Approver for DenyAll {
    async fn approve(&self, _request: &ApprovalRequest) -> ApprovalDecision {
        ApprovalDecision::Deny
    }
}

/// Asks the user a free-form question from the `ask_user` tool.
#[async_trait]
pub trait Asker: Send + Sync {
    async fn ask(&self, question: &str) -> String;
}

/// Used by headless runs where no user is available to answer.
pub struct NoAsker;

#[async_trait]
impl Asker for NoAsker {
    async fn ask(&self, _question: &str) -> String {
        "No user is available to answer; proceed using your best judgement.".to_string()
    }
}

/// Used by tests that pre-script the answers.
pub struct ScriptedAsker {
    answers: std::sync::Mutex<std::collections::VecDeque<String>>,
}

impl ScriptedAsker {
    pub fn new(answers: impl IntoIterator<Item = String>) -> Self {
        Self {
            answers: std::sync::Mutex::new(answers.into_iter().collect()),
        }
    }
}

#[async_trait]
impl Asker for ScriptedAsker {
    async fn ask(&self, _question: &str) -> String {
        self.answers
            .lock()
            .ok()
            .and_then(|mut answers| answers.pop_front())
            .unwrap_or_else(|| "no answer".to_string())
    }
}

/// A scripted approver for tests.
pub struct ScriptedApprover {
    decisions: std::sync::Mutex<std::collections::VecDeque<ApprovalDecision>>,
    pub requests: std::sync::Mutex<Vec<ApprovalRequest>>,
}

impl ScriptedApprover {
    pub fn new(decisions: impl IntoIterator<Item = ApprovalDecision>) -> Self {
        Self {
            decisions: std::sync::Mutex::new(decisions.into_iter().collect()),
            requests: std::sync::Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Approver for ScriptedApprover {
    async fn approve(&self, request: &ApprovalRequest) -> ApprovalDecision {
        if let Ok(mut requests) = self.requests.lock() {
            requests.push(request.clone());
        }
        self.decisions
            .lock()
            .ok()
            .and_then(|mut decisions| decisions.pop_front())
            .unwrap_or(ApprovalDecision::Deny)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn structured_approval_falls_back_to_the_legacy_request() {
        let approver = ScriptedApprover::new([ApprovalDecision::AllowOnce]);
        let call = ToolCall {
            id: "call-9".to_string(),
            name: "bash".to_string(),
            arguments: r#"{"command":"cargo test"}"#.to_string(),
        };

        let decision = approver.approve_tool(&call, "cargo test").await;

        assert_eq!(decision, ApprovalDecision::AllowOnce);
        assert_eq!(
            *approver.requests.lock().unwrap(),
            vec![ApprovalRequest::new("bash", "cargo test")]
        );
    }
}
