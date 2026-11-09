//! Stable-v1 ACP permission requests for mutating tool calls.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming per-prompt ACP wiring")
)]

use std::collections::HashMap;
use std::sync::Arc;

use agent_client_protocol::v1::schema::{
    PermissionOption, PermissionOptionKind, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, SessionId, ToolCallUpdate, ToolCallUpdateFields,
};
use agent_client_protocol::v1::ConnectionTo;
use agent_client_protocol::Client;
use async_trait::async_trait;
use serde_json::Value;

use crate::interaction::{ApprovalDecision, ApprovalRequest, Approver};
use crate::model::ToolCall;

const ALLOW_ONCE: &str = "allow_once";
const ALLOW_ALWAYS: &str = "allow_always";
const REJECT_ONCE: &str = "reject_once";
const REJECT_ALWAYS: &str = "reject_always";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RememberedPermission {
    Allow,
    Reject,
}

pub(crate) type ApprovalPolicy = Arc<tokio::sync::Mutex<HashMap<String, RememberedPermission>>>;

pub(crate) fn new_policy() -> ApprovalPolicy {
    Arc::new(tokio::sync::Mutex::new(HashMap::new()))
}

/// Sends one fail-closed permission request for each mutating tool invocation.
pub(crate) struct AcpApprover {
    session_id: SessionId,
    connection: ConnectionTo<Client>,
    policy: ApprovalPolicy,
}

impl AcpApprover {
    pub(crate) fn new(
        session_id: SessionId,
        connection: ConnectionTo<Client>,
        policy: ApprovalPolicy,
    ) -> Self {
        Self {
            session_id,
            connection,
            policy,
        }
    }

    fn request(session_id: SessionId, call: &ToolCall, summary: &str) -> RequestPermissionRequest {
        let title = if summary.is_empty() {
            call.name.clone()
        } else {
            summary.to_string()
        };
        let raw_input = call
            .parse_arguments()
            .unwrap_or_else(|_| Value::String(call.arguments.clone()));
        let fields = ToolCallUpdateFields::new()
            .title(title)
            .name(call.name.clone())
            .raw_input(raw_input);
        RequestPermissionRequest::new(
            session_id,
            ToolCallUpdate::new(call.id.clone(), fields),
            vec![
                PermissionOption::new(ALLOW_ONCE, "Allow once", PermissionOptionKind::AllowOnce),
                PermissionOption::new(
                    ALLOW_ALWAYS,
                    "Allow always",
                    PermissionOptionKind::AllowAlways,
                ),
                PermissionOption::new(REJECT_ONCE, "Reject once", PermissionOptionKind::RejectOnce),
                PermissionOption::new(
                    REJECT_ALWAYS,
                    "Reject always",
                    PermissionOptionKind::RejectAlways,
                ),
            ],
        )
    }

    fn decision(
        response: RequestPermissionResponse,
    ) -> (ApprovalDecision, Option<RememberedPermission>) {
        match response.outcome {
            RequestPermissionOutcome::Selected(selected)
                if selected.option_id.to_string() == ALLOW_ONCE =>
            {
                (ApprovalDecision::AllowOnce, None)
            }
            RequestPermissionOutcome::Selected(selected)
                if selected.option_id.to_string() == ALLOW_ALWAYS =>
            {
                (
                    ApprovalDecision::AllowAlways,
                    Some(RememberedPermission::Allow),
                )
            }
            RequestPermissionOutcome::Selected(selected)
                if selected.option_id.to_string() == REJECT_ALWAYS =>
            {
                (ApprovalDecision::Deny, Some(RememberedPermission::Reject))
            }
            RequestPermissionOutcome::Selected(_) | RequestPermissionOutcome::Cancelled => {
                (ApprovalDecision::Deny, None)
            }
            _ => (ApprovalDecision::Deny, None),
        }
    }
}

#[async_trait]
impl Approver for AcpApprover {
    async fn approve(&self, _request: &ApprovalRequest) -> ApprovalDecision {
        ApprovalDecision::Deny
    }

    async fn approve_tool(&self, call: &ToolCall, summary: &str) -> ApprovalDecision {
        let mut policy = self.policy.lock().await;
        match policy.get(&call.name) {
            Some(RememberedPermission::Allow) => return ApprovalDecision::AllowAlways,
            Some(RememberedPermission::Reject) => return ApprovalDecision::Deny,
            None => {}
        }
        let response = self
            .connection
            .send_request(Self::request(self.session_id.clone(), call, summary))
            .block_task()
            .await;
        let Ok(response) = response else {
            return ApprovalDecision::Deny;
        };
        let (decision, remembered) = Self::decision(response);
        if let Some(remembered) = remembered {
            policy.insert(call.name.clone(), remembered);
        }
        decision
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use agent_client_protocol::v1::schema::{PermissionOptionId, SelectedPermissionOutcome};
    use agent_client_protocol::{Agent, Error as ProtocolError};

    use super::*;

    fn call(arguments: &str) -> ToolCall {
        ToolCall {
            id: "provider-call-7".to_string(),
            name: "write_file".to_string(),
            arguments: arguments.to_string(),
        }
    }

    #[test]
    fn request_preserves_identity_arguments_and_once_only_options() {
        let request = AcpApprover::request(
            SessionId::new("session-1"),
            &call(r#"{"path":"a.txt","content":"x"}"#),
            "write a.txt",
        );

        assert_eq!(request.session_id, SessionId::new("session-1"));
        assert_eq!(
            request.tool_call.tool_call_id.to_string(),
            "provider-call-7"
        );
        assert_eq!(
            request.tool_call.fields.title.as_deref(),
            Some("write a.txt")
        );
        assert_eq!(request.tool_call.fields.name.as_deref(), Some("write_file"));
        assert_eq!(request.tool_call.fields.raw_input.unwrap()["path"], "a.txt");
        assert_eq!(request.options.len(), 4);
        assert_eq!(request.options[0].kind, PermissionOptionKind::AllowOnce);
        assert_eq!(request.options[1].kind, PermissionOptionKind::AllowAlways);
        assert_eq!(request.options[2].kind, PermissionOptionKind::RejectOnce);
        assert_eq!(request.options[3].kind, PermissionOptionKind::RejectAlways);
    }

    #[test]
    fn malformed_arguments_remain_visible_as_raw_text() {
        let request = AcpApprover::request(SessionId::new("session-1"), &call("not-json"), "");

        assert_eq!(
            request.tool_call.fields.title.as_deref(),
            Some("write_file")
        );
        assert_eq!(
            request.tool_call.fields.raw_input,
            Some(Value::String("not-json".into()))
        );
    }

    #[test]
    fn outcomes_remember_only_explicit_always_choices() {
        let selected = |id| {
            RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
                SelectedPermissionOutcome::new(PermissionOptionId::new(id)),
            ))
        };
        assert_eq!(
            AcpApprover::decision(selected(ALLOW_ONCE)),
            (ApprovalDecision::AllowOnce, None)
        );
        assert_eq!(
            AcpApprover::decision(selected(ALLOW_ALWAYS)),
            (
                ApprovalDecision::AllowAlways,
                Some(RememberedPermission::Allow)
            )
        );
        assert_eq!(
            AcpApprover::decision(selected(REJECT_ONCE)),
            (ApprovalDecision::Deny, None)
        );
        assert_eq!(
            AcpApprover::decision(selected(REJECT_ALWAYS)),
            (ApprovalDecision::Deny, Some(RememberedPermission::Reject))
        );
        assert_eq!(
            AcpApprover::decision(selected("unknown")),
            (ApprovalDecision::Deny, None)
        );
        assert_eq!(
            AcpApprover::decision(RequestPermissionResponse::new(
                RequestPermissionOutcome::Cancelled
            )),
            (ApprovalDecision::Deny, None)
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sends_a_typed_agent_to_client_request_and_uses_the_selected_option() {
        let client = Client.v1().on_receive_request(
            async |request: RequestPermissionRequest, responder, _connection| {
                assert_eq!(request.session_id, SessionId::new("session-1"));
                assert_eq!(
                    request.tool_call.tool_call_id.to_string(),
                    "provider-call-7"
                );
                responder.respond(RequestPermissionResponse::new(
                    RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(ALLOW_ONCE)),
                ))
            },
            agent_client_protocol::on_receive_request!(),
        );

        Agent
            .v1()
            .connect_with(client, async |connection| {
                let approver =
                    AcpApprover::new(SessionId::new("session-1"), connection, new_policy());
                assert_eq!(
                    approver.approve_tool(&call("{}"), "write").await,
                    ApprovalDecision::AllowOnce
                );
                Ok::<(), ProtocolError>(())
            })
            .await
            .unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn allow_always_is_session_scoped_and_per_tool() {
        let requests = Arc::new(AtomicUsize::new(0));
        let client_requests = Arc::clone(&requests);
        let client = Client.v1().on_receive_request(
            async move |_request: RequestPermissionRequest, responder, _connection| {
                client_requests.fetch_add(1, Ordering::SeqCst);
                responder.respond(RequestPermissionResponse::new(
                    RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                        ALLOW_ALWAYS,
                    )),
                ))
            },
            agent_client_protocol::on_receive_request!(),
        );

        Agent
            .v1()
            .connect_with(client, async |connection| {
                let approver =
                    AcpApprover::new(SessionId::new("session-1"), connection, new_policy());
                assert_eq!(
                    approver.approve_tool(&call("{}"), "write").await,
                    ApprovalDecision::AllowAlways
                );
                assert_eq!(
                    approver.approve_tool(&call("{}"), "write again").await,
                    ApprovalDecision::AllowAlways
                );
                let mut other = call("{}");
                other.name = "delete_file".into();
                assert_eq!(
                    approver.approve_tool(&other, "delete").await,
                    ApprovalDecision::AllowAlways
                );
                Ok::<(), ProtocolError>(())
            })
            .await
            .unwrap();
        assert_eq!(requests.load(Ordering::SeqCst), 2);
    }
}
