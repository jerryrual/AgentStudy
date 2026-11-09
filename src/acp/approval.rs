//! Stable-v1 ACP permission requests for mutating tool calls.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming per-prompt ACP wiring")
)]

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
const REJECT_ONCE: &str = "reject_once";

/// Sends one fail-closed permission request for each mutating tool invocation.
pub(crate) struct AcpApprover {
    session_id: SessionId,
    connection: ConnectionTo<Client>,
}

impl AcpApprover {
    pub(crate) fn new(session_id: SessionId, connection: ConnectionTo<Client>) -> Self {
        Self {
            session_id,
            connection,
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
                PermissionOption::new(REJECT_ONCE, "Reject once", PermissionOptionKind::RejectOnce),
            ],
        )
    }

    fn decision(response: RequestPermissionResponse) -> ApprovalDecision {
        match response.outcome {
            RequestPermissionOutcome::Selected(selected)
                if selected.option_id.to_string() == ALLOW_ONCE =>
            {
                ApprovalDecision::AllowOnce
            }
            RequestPermissionOutcome::Selected(_) | RequestPermissionOutcome::Cancelled => {
                ApprovalDecision::Deny
            }
            _ => ApprovalDecision::Deny,
        }
    }
}

#[async_trait]
impl Approver for AcpApprover {
    async fn approve(&self, _request: &ApprovalRequest) -> ApprovalDecision {
        ApprovalDecision::Deny
    }

    async fn approve_tool(&self, call: &ToolCall, summary: &str) -> ApprovalDecision {
        match self
            .connection
            .send_request(Self::request(self.session_id.clone(), call, summary))
            .block_task()
            .await
        {
            Ok(response) => Self::decision(response),
            Err(_) => ApprovalDecision::Deny,
        }
    }
}

#[cfg(test)]
mod tests {
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
        assert_eq!(request.options.len(), 2);
        assert_eq!(request.options[0].kind, PermissionOptionKind::AllowOnce);
        assert_eq!(request.options[1].kind, PermissionOptionKind::RejectOnce);
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
    fn outcomes_fail_closed_except_for_the_offered_allow_once_id() {
        let selected = |id| {
            RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
                SelectedPermissionOutcome::new(PermissionOptionId::new(id)),
            ))
        };
        assert_eq!(
            AcpApprover::decision(selected(ALLOW_ONCE)),
            ApprovalDecision::AllowOnce
        );
        assert_eq!(
            AcpApprover::decision(selected(REJECT_ONCE)),
            ApprovalDecision::Deny
        );
        assert_eq!(
            AcpApprover::decision(selected("unknown")),
            ApprovalDecision::Deny
        );
        assert_eq!(
            AcpApprover::decision(RequestPermissionResponse::new(
                RequestPermissionOutcome::Cancelled
            )),
            ApprovalDecision::Deny
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
                let approver = AcpApprover::new(SessionId::new("session-1"), connection);
                assert_eq!(
                    approver.approve_tool(&call("{}"), "write").await,
                    ApprovalDecision::AllowOnce
                );
                Ok::<(), ProtocolError>(())
            })
            .await
            .unwrap();
    }
}
