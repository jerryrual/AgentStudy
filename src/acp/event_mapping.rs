//! Pure mapping from internal turn events to stable ACP v1 updates.

use agent_client_protocol::v1::schema::{
    ContentChunk, SessionUpdate, ToolCall, ToolCallContent, ToolCallStatus, ToolCallUpdate,
    ToolCallUpdateFields, ToolKind, UsageUpdate,
};
use serde_json::Value;

use crate::model::ToolCall as ModelToolCall;
use crate::run::turn_events::TurnEvent;

/// Map one internal event without inventing session or message identity.
///
/// Usage and diagnostic events need session capabilities or context-window
/// state that this pure mapper intentionally does not own.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming ACP dispatcher")
)]
pub(crate) fn session_update(
    event: TurnEvent,
    context_window: Option<u32>,
) -> Option<SessionUpdate> {
    match event {
        TurnEvent::TextDelta(text) => Some(SessionUpdate::AgentMessageChunk(text_chunk(text))),
        TurnEvent::ReasoningDelta(text) => Some(SessionUpdate::AgentThoughtChunk(text_chunk(text))),
        TurnEvent::ToolStarted { call, summary } => Some(tool_started(call, summary)),
        TurnEvent::ToolFinished {
            call, ok, output, ..
        } => Some(tool_finished(call.id, ok, output)),
        TurnEvent::Usage(usage) => context_window.map(|size| {
            SessionUpdate::UsageUpdate(UsageUpdate::new(usage.prompt_tokens, u64::from(size)))
        }),
        TurnEvent::Info(_) | TurnEvent::Error(_) | TurnEvent::Warning(_) => None,
    }
}

/// Wrap one exact text payload as an ACP content chunk.
pub(super) fn text_chunk(text: String) -> ContentChunk {
    ContentChunk::new(text.into())
}

/// Project an internal tool declaration as an in-progress ACP tool call.
pub(super) fn tool_started(call: ModelToolCall, summary: String) -> SessionUpdate {
    let title = if summary.is_empty() {
        call.name.clone()
    } else {
        summary
    };
    let raw_input = call
        .parse_arguments()
        .unwrap_or_else(|_| Value::String(call.arguments.clone()));
    SessionUpdate::ToolCall(
        ToolCall::new(call.id, title)
            .name(call.name.clone())
            .kind(tool_kind(&call.name))
            .status(ToolCallStatus::InProgress)
            .raw_input(raw_input),
    )
}

/// Project an internal tool result as a terminal ACP tool update.
pub(super) fn tool_finished(call_id: String, ok: bool, output: String) -> SessionUpdate {
    let status = if ok {
        ToolCallStatus::Completed
    } else {
        ToolCallStatus::Failed
    };
    let fields = ToolCallUpdateFields::new()
        .status(status)
        .content(vec![ToolCallContent::from(output.clone())])
        .raw_output(Value::String(output));
    SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(call_id, fields))
}

fn tool_kind(name: &str) -> ToolKind {
    match name {
        "read_file" | "list_dir" => ToolKind::Read,
        "grep" | "glob" => ToolKind::Search,
        "write_file" | "edit_file" => ToolKind::Edit,
        "bash" => ToolKind::Execute,
        "todo_write" => ToolKind::Think,
        _ => ToolKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::model::{ToolCall as ModelToolCall, Usage};

    fn update(event: TurnEvent) -> Option<SessionUpdate> {
        session_update(event, None)
    }

    fn call(id: &str, name: &str, arguments: &str) -> ModelToolCall {
        ModelToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: arguments.to_string(),
        }
    }

    #[test]
    fn maps_text_and_reasoning_to_distinct_chunk_updates() {
        let text = update(TurnEvent::TextDelta("answer".to_string())).unwrap();
        let thought = update(TurnEvent::ReasoningDelta("thought".to_string())).unwrap();

        assert_eq!(
            serde_json::to_value(text).unwrap(),
            serde_json::json!({
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "answer"}
            })
        );
        assert_eq!(
            serde_json::to_value(thought).unwrap(),
            serde_json::json!({
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "thought"}
            })
        );
    }

    #[test]
    fn maps_tool_lifecycle_with_stable_identity_and_raw_payloads() {
        let first = call("call-1", "read_file", r#"{"path":"src/lib.rs"}"#);
        let second = call("call-2", "read_file", "not-json");

        let first_start = update(TurnEvent::ToolStarted {
            call: first.clone(),
            summary: "src/lib.rs".to_string(),
        })
        .unwrap();
        let second_start = update(TurnEvent::ToolStarted {
            call: second.clone(),
            summary: String::new(),
        })
        .unwrap();
        let first_finish = update(TurnEvent::ToolFinished {
            call: first,
            ok: true,
            duration: Duration::from_millis(4),
            summary: "src/lib.rs".to_string(),
            output: "contents".to_string(),
        })
        .unwrap();
        let second_finish = update(TurnEvent::ToolFinished {
            call: second,
            ok: false,
            duration: Duration::from_millis(2),
            summary: "read_file".to_string(),
            output: "failed".to_string(),
        })
        .unwrap();

        let first_start = serde_json::to_value(first_start).unwrap();
        let second_start = serde_json::to_value(second_start).unwrap();
        let first_finish = serde_json::to_value(first_finish).unwrap();
        let second_finish = serde_json::to_value(second_finish).unwrap();
        assert_eq!(first_start["toolCallId"], "call-1");
        assert_eq!(first_start["rawInput"]["path"], "src/lib.rs");
        assert_eq!(first_start["status"], "in_progress");
        assert_eq!(first_start["kind"], "read");
        assert_eq!(second_start["toolCallId"], "call-2");
        assert_eq!(second_start["title"], "read_file");
        assert_eq!(second_start["rawInput"], "not-json");
        assert_eq!(first_finish["toolCallId"], "call-1");
        assert_eq!(first_finish["status"], "completed");
        assert_eq!(first_finish["rawOutput"], "contents");
        assert_eq!(first_finish["content"][0]["content"]["text"], "contents");
        assert_eq!(second_finish["toolCallId"], "call-2");
        assert_eq!(second_finish["status"], "failed");
    }

    #[test]
    fn maps_usage_only_with_a_known_context_window_and_omits_notices() {
        let usage = Usage {
            prompt_tokens: 3,
            completion_tokens: 2,
            total_tokens: 5,
        };

        assert_eq!(
            serde_json::to_value(session_update(TurnEvent::Usage(usage), Some(128_000)).unwrap())
                .unwrap(),
            serde_json::json!({
                "sessionUpdate": "usage_update",
                "used": 3,
                "size": 128_000
            })
        );
        assert!(update(TurnEvent::Usage(usage)).is_none());
        assert!(update(TurnEvent::Info("working".to_string())).is_none());
        assert!(update(TurnEvent::Error("failed".to_string())).is_none());
        assert!(update(TurnEvent::Warning("retry".to_string())).is_none());
    }
}
