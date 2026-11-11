//! Pure projection of durable session records into stable ACP v1 replay updates.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming ACP load handler")
)]

use std::collections::HashSet;

use agent_client_protocol::v1::schema::{ContentBlock, ContentChunk, ImageContent, SessionUpdate};
use anyhow::{ensure, Result};

use super::event_mapping::{text_chunk, tool_finished, tool_started};
use crate::model::{ToolCall, UserContent, UserContentPart};
use crate::session::Record;

const INTERRUPTED_TOOL_OUTPUT: &str =
    "Error: the tool call was interrupted before it produced a result.";

/// Rebuild the retained visible transcript as ordered stable ACP v1 updates.
///
/// The returned vector is complete before any notification is sent, so a
/// validation failure cannot expose a partial history to the client.
pub(crate) fn replay_updates(records: &[Record]) -> Result<Vec<SessionUpdate>> {
    let mut updates = Vec::new();
    let mut pending = Vec::<ToolCall>::new();
    let mut seen_tool_ids = HashSet::new();

    for record in records {
        match record {
            Record::User { text } => {
                flush_interrupted(&mut updates, &mut pending);
                push_user_content(&mut updates, text);
            }
            Record::InjectedUser { text, .. } => {
                flush_interrupted(&mut updates, &mut pending);
                updates.push(SessionUpdate::UserMessageChunk(text_chunk(text.clone())));
            }
            Record::Assistant {
                content,
                reasoning,
                tool_calls,
                responses_items: _,
                responses_continuation: _,
            } => {
                flush_interrupted(&mut updates, &mut pending);
                if let Some(reasoning) = reasoning {
                    updates.push(SessionUpdate::AgentThoughtChunk(text_chunk(
                        reasoning.clone(),
                    )));
                }
                if let Some(content) = content {
                    updates.push(SessionUpdate::AgentMessageChunk(text_chunk(
                        content.clone(),
                    )));
                }
                for call in tool_calls {
                    ensure!(
                        !call.id.is_empty(),
                        "ACP replay tool call ID must not be empty"
                    );
                    ensure!(
                        seen_tool_ids.insert(call.id.clone()),
                        "duplicate ACP replay tool call ID: {}",
                        call.id
                    );
                    updates.push(tool_started(call.clone(), call.name.clone()));
                    pending.push(call.clone());
                }
            }
            Record::Tool {
                call_id,
                name,
                content,
            } => {
                let Some(position) = pending.iter().position(|call| call.id == *call_id) else {
                    continue;
                };
                let call = pending.remove(position);
                ensure!(
                    call.name == *name,
                    "ACP replay tool result name mismatch for {call_id}: expected {}, got {name}",
                    call.name
                );
                updates.push(tool_finished(
                    call_id.clone(),
                    !content.starts_with("Error:"),
                    content.clone(),
                ));
            }
            Record::Compaction {
                summary,
                compacted_messages: None,
            } => {
                updates.clear();
                pending.clear();
                seen_tool_ids.clear();
                updates.push(SessionUpdate::AgentMessageChunk(text_chunk(
                    summary.clone(),
                )));
            }
            Record::Compaction {
                compacted_messages: Some(_),
                ..
            }
            | Record::CompressionBatch { .. }
            | Record::ContextProjection { .. } => {
                flush_interrupted(&mut updates, &mut pending);
            }
            Record::Meta { .. }
            | Record::AgentMode { .. }
            | Record::Usage { .. }
            | Record::PhaseFourBreaker { .. } => {}
        }
    }
    flush_interrupted(&mut updates, &mut pending);
    Ok(updates)
}

fn push_user_content(updates: &mut Vec<SessionUpdate>, content: &UserContent) {
    match content {
        UserContent::Text(text) => {
            updates.push(SessionUpdate::UserMessageChunk(text_chunk(text.clone())));
        }
        UserContent::Parts(parts) => {
            updates.extend(parts.iter().map(|part| {
                let block = match part {
                    UserContentPart::Text { text } => ContentBlock::from(text.clone()),
                    UserContentPart::Image { image } => ContentBlock::Image(ImageContent::new(
                        image.data.clone(),
                        image.media_type.clone(),
                    )),
                };
                SessionUpdate::UserMessageChunk(ContentChunk::new(block))
            }));
        }
    }
}

fn flush_interrupted(updates: &mut Vec<SessionUpdate>, pending: &mut Vec<ToolCall>) {
    updates.extend(
        pending
            .drain(..)
            .map(|call| tool_finished(call.id, false, INTERRUPTED_TOOL_OUTPUT.to_string())),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: arguments.to_string(),
        }
    }

    fn wire(records: &[Record]) -> Vec<serde_json::Value> {
        replay_updates(records)
            .unwrap()
            .into_iter()
            .map(|update| serde_json::to_value(update).unwrap())
            .collect()
    }

    #[test]
    fn replays_messages_thoughts_and_tool_lifecycle_in_record_order() {
        let records = vec![
            Record::User {
                text: "first".into(),
            },
            Record::InjectedUser {
                injection_id: "private-id".to_string(),
                text: "follow-up".to_string(),
            },
            Record::Assistant {
                content: Some("checking".to_string()),
                reasoning: Some("reasoning".to_string()),
                tool_calls: vec![call("call-1", "read_file", r#"{"path":"a"}"#)],
                responses_items: vec![serde_json::json!({"providerSecret": "not replayed"})],
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

        let updates = wire(&records);
        assert_eq!(
            updates
                .iter()
                .map(|update| update["sessionUpdate"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "user_message_chunk",
                "user_message_chunk",
                "agent_thought_chunk",
                "agent_message_chunk",
                "tool_call",
                "tool_call_update",
                "agent_message_chunk",
            ]
        );
        assert_eq!(updates[0]["content"]["text"], "first");
        assert_eq!(updates[1]["content"]["text"], "follow-up");
        assert_eq!(updates[4]["toolCallId"], "call-1");
        assert_eq!(updates[4]["rawInput"]["path"], "a");
        assert_eq!(updates[5]["status"], "completed");
        assert_eq!(updates[5]["rawOutput"], "contents");
        assert!(!serde_json::to_string(&updates)
            .unwrap()
            .contains("providerSecret"));
    }

    #[test]
    fn replays_multimodal_user_parts_in_order() {
        let updates = wire(&[Record::User {
            text: UserContent::parts(vec![
                UserContentPart::Text {
                    text: "before".into(),
                },
                UserContentPart::Image {
                    image: crate::model::ImageInput {
                        media_type: "image/png".into(),
                        data: "iVBORw0KGgo=".into(),
                        validated_key: None,
                    },
                },
                UserContentPart::Text {
                    text: "after".into(),
                },
            ]),
        }]);

        assert_eq!(updates.len(), 3);
        assert_eq!(updates[0]["content"]["text"], "before");
        assert_eq!(updates[1]["content"]["type"], "image");
        assert_eq!(updates[1]["content"]["mimeType"], "image/png");
        assert_eq!(updates[1]["content"]["data"], "iVBORw0KGgo=");
        assert_eq!(updates[2]["content"]["text"], "after");
    }

    #[test]
    fn omits_orphans_and_fails_dangling_calls_in_declaration_order() {
        let records = vec![
            Record::Tool {
                call_id: "orphan".to_string(),
                name: "bash".to_string(),
                content: "ignored".to_string(),
            },
            Record::Assistant {
                content: None,
                reasoning: None,
                tool_calls: vec![call("one", "bash", "{}"), call("two", "grep", "{}")],
                responses_items: vec![],
                responses_continuation: None,
            },
            Record::Tool {
                call_id: "two".to_string(),
                name: "grep".to_string(),
                content: "Error: failed".to_string(),
            },
        ];

        let updates = wire(&records);
        assert_eq!(updates.len(), 4);
        assert_eq!(updates[0]["toolCallId"], "one");
        assert_eq!(updates[1]["toolCallId"], "two");
        assert_eq!(updates[2]["toolCallId"], "two");
        assert_eq!(updates[2]["status"], "failed");
        assert_eq!(updates[3]["toolCallId"], "one");
        assert_eq!(updates[3]["status"], "failed");
        assert_eq!(updates[3]["rawOutput"], INTERRUPTED_TOOL_OUTPUT);
    }

    #[test]
    fn manual_compaction_resets_before_emission_and_allows_reused_tool_ids() {
        let records = vec![
            Record::User {
                text: "discarded".into(),
            },
            Record::Assistant {
                content: None,
                reasoning: None,
                tool_calls: vec![call("same", "bash", "{}")],
                responses_items: vec![],
                responses_continuation: None,
            },
            Record::Compaction {
                summary: "retained summary".to_string(),
                compacted_messages: None,
            },
            Record::Assistant {
                content: None,
                reasoning: None,
                tool_calls: vec![call("same", "grep", "{}")],
                responses_items: vec![],
                responses_continuation: None,
            },
            Record::Tool {
                call_id: "same".to_string(),
                name: "grep".to_string(),
                content: "match".to_string(),
            },
        ];

        let updates = wire(&records);
        assert_eq!(updates.len(), 3);
        assert_eq!(updates[0]["sessionUpdate"], "agent_message_chunk");
        assert_eq!(updates[0]["content"]["text"], "retained summary");
        assert_eq!(updates[1]["name"], "grep");
        assert_eq!(updates[2]["status"], "completed");
        assert!(!serde_json::to_string(&updates)
            .unwrap()
            .contains("discarded"));
    }

    #[test]
    fn rejects_ambiguous_tool_identity_without_returning_partial_replay() {
        let duplicate = vec![Record::Assistant {
            content: None,
            reasoning: None,
            tool_calls: vec![call("same", "bash", "{}"), call("same", "grep", "{}")],
            responses_items: vec![],
            responses_continuation: None,
        }];
        assert_eq!(
            replay_updates(&duplicate).unwrap_err().to_string(),
            "duplicate ACP replay tool call ID: same"
        );

        let mismatch = vec![
            Record::Assistant {
                content: None,
                reasoning: None,
                tool_calls: vec![call("call-1", "bash", "{}")],
                responses_items: vec![],
                responses_continuation: None,
            },
            Record::Tool {
                call_id: "call-1".to_string(),
                name: "grep".to_string(),
                content: "result".to_string(),
            },
        ];
        assert_eq!(
            replay_updates(&mismatch).unwrap_err().to_string(),
            "ACP replay tool result name mismatch for call-1: expected bash, got grep"
        );

        let empty = vec![Record::Assistant {
            content: None,
            reasoning: None,
            tool_calls: vec![call("", "bash", "{}")],
            responses_items: vec![],
            responses_continuation: None,
        }];
        assert_eq!(
            replay_updates(&empty).unwrap_err().to_string(),
            "ACP replay tool call ID must not be empty"
        );
    }

    #[test]
    fn automatic_compaction_is_invisible_but_closes_interrupted_tools() {
        let records = vec![
            Record::User {
                text: "question".into(),
            },
            Record::Assistant {
                content: None,
                reasoning: None,
                tool_calls: vec![call("call-1", "bash", "{}")],
                responses_items: vec![],
                responses_continuation: None,
            },
            Record::Compaction {
                summary: "internal summary".to_string(),
                compacted_messages: Some(2),
            },
            Record::Assistant {
                content: Some("answer".to_string()),
                reasoning: None,
                tool_calls: vec![],
                responses_items: vec![],
                responses_continuation: None,
            },
        ];

        let updates = wire(&records);
        assert_eq!(updates.len(), 4);
        assert_eq!(updates[2]["status"], "failed");
        assert_eq!(updates[3]["content"]["text"], "answer");
        assert!(!serde_json::to_string(&updates)
            .unwrap()
            .contains("internal summary"));
    }
}
