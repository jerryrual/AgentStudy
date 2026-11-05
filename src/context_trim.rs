//! Deterministic request-only context projections.

use std::borrow::Cow;
use std::collections::HashMap;

use crate::model::{Message, ToolCall};

const SUMMARY_PREFIX: &str = "[Tool call: ";
const SUMMARY_SUFFIX: &str = "]";
const RESULT_SUMMARY_MAX: usize = 100;
const ARGS_SUMMARY_MAX: usize = 200;
const SUMMARY_TOTAL_MAX: usize = 4_096;
const SUMMARY_TRUNCATION_MARKER: &str = "... [truncated]";

/// Result of projecting old tool results for one model request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TrimResult<'a> {
    pub(crate) messages: Cow<'a, [Message]>,
    pub(crate) summarized: usize,
    pub(crate) estimated_tokens: u32,
}

/// Summarize completed-turn tool results oldest-first until `target_tokens`.
///
/// The returned request projection borrows unchanged history and owns only a
/// modified clone. The source history is never mutated, and the current turn
/// (starting at the last user message) is deliberately ineligible.
pub(crate) fn trim_previous_tool_results<'a, F>(
    messages: &'a [Message],
    trigger_tokens: u32,
    target_tokens: u32,
    mut estimate: F,
) -> TrimResult<'a>
where
    F: FnMut(&[Message]) -> u32,
{
    let initial = estimate(messages);
    let mut projected = Cow::Borrowed(messages);
    if initial < trigger_tokens {
        return TrimResult {
            messages: projected,
            summarized: 0,
            estimated_tokens: initial,
        };
    }
    let Some(current_turn_start) = messages
        .iter()
        .rposition(|message| matches!(message, Message::User { .. }))
    else {
        return TrimResult {
            messages: projected,
            summarized: 0,
            estimated_tokens: initial,
        };
    };

    let calls = tool_calls_by_id(messages);
    let groups = tool_result_groups(messages, current_turn_start);
    let mut summarized = 0;
    let mut estimated_tokens = initial;
    for group in groups {
        if group.iter().any(|&index| {
            matches!(&projected[index], Message::Tool { content, .. } if content.starts_with(SUMMARY_PREFIX))
        }) {
            continue;
        }
        let replacements: Vec<_> = group
            .into_iter()
            .filter_map(|index| match &projected[index] {
                Message::Tool {
                    call_id,
                    name,
                    content,
                } => {
                    let call = calls.get(call_id.as_str()).copied();
                    Some((index, tool_summary(name, content, call)))
                }
                _ => None,
            })
            .collect();
        if replacements.is_empty() {
            continue;
        }
        let mut candidate = projected.as_ref().to_vec();
        for (index, summary) in &replacements {
            let Message::Tool { content, .. } = &mut candidate[*index] else {
                unreachable!("replacement candidates are tool messages")
            };
            *content = summary.clone();
        }
        let candidate_tokens = estimate(&candidate);
        if candidate_tokens >= estimated_tokens {
            continue;
        }
        projected = Cow::Owned(candidate);
        summarized += replacements.len();
        estimated_tokens = candidate_tokens;
        if estimated_tokens <= target_tokens {
            break;
        }
    }

    TrimResult {
        messages: projected,
        summarized,
        estimated_tokens,
    }
}

fn tool_result_groups(messages: &[Message], end: usize) -> Vec<Vec<usize>> {
    let mut groups = Vec::<Vec<usize>>::new();
    let mut group_by_call = HashMap::<&str, usize>::new();
    for (index, message) in messages[..end].iter().enumerate() {
        match message {
            Message::Assistant { message } if !message.tool_calls.is_empty() => {
                let group = groups.len();
                groups.push(Vec::new());
                for call in &message.tool_calls {
                    group_by_call.insert(call.id.as_str(), group);
                }
            }
            Message::Tool { call_id, .. } => {
                if let Some(group) = group_by_call.get(call_id.as_str()).copied() {
                    groups[group].push(index);
                }
            }
            _ => {}
        }
    }
    groups.retain(|group| !group.is_empty());
    groups
}

fn tool_calls_by_id(messages: &[Message]) -> HashMap<&str, &ToolCall> {
    let mut calls = HashMap::new();
    for message in messages {
        if let Message::Assistant { message } = message {
            for call in &message.tool_calls {
                calls.insert(call.id.as_str(), call);
            }
        }
    }
    calls
}

fn tool_summary(name: &str, content: &str, call: Option<&ToolCall>) -> String {
    let name = call.map_or(name, |call| call.name.as_str());
    let arguments = call.map_or("", |call| call.arguments.trim());
    let arguments = format_arguments(arguments);
    let call = if arguments.is_empty() {
        name.to_string()
    } else {
        format!("{name}({arguments})")
    };
    let result = format_result(content);
    cap_summary(format!("{call} → {result}"))
}

fn format_arguments(raw: &str) -> String {
    if raw.is_empty() || raw == "{}" {
        return String::new();
    }
    let formatted = match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(serde_json::Value::Object(object)) => object
            .iter()
            .map(|(key, value)| {
                let value = match value {
                    serde_json::Value::String(value) => format!("{value:?}"),
                    _ => value.to_string(),
                };
                format!("{key}={value}")
            })
            .collect::<Vec<_>>()
            .join(", "),
        Ok(value) => value.to_string(),
        Err(_) => raw.to_string(),
    };
    truncate_chars(&formatted, ARGS_SUMMARY_MAX, "...")
}

fn format_result(content: &str) -> String {
    let content = content.trim();
    if content.is_empty() {
        return "completed".to_string();
    }
    let count = content.chars().count();
    if count <= RESULT_SUMMARY_MAX {
        return content.to_string();
    }
    let prefix: String = content.chars().take(RESULT_SUMMARY_MAX).collect();
    format!("{prefix}... ({count} chars)")
}

fn cap_summary(label: String) -> String {
    let summary = format!("{SUMMARY_PREFIX}{label}{SUMMARY_SUFFIX}");
    if summary.chars().count() <= SUMMARY_TOTAL_MAX {
        return summary;
    }
    let reserved = SUMMARY_PREFIX.chars().count()
        + SUMMARY_SUFFIX.chars().count()
        + SUMMARY_TRUNCATION_MARKER.chars().count();
    let label: String = label
        .chars()
        .take(SUMMARY_TOTAL_MAX.saturating_sub(reserved))
        .collect();
    format!(
        "{SUMMARY_PREFIX}{}{SUMMARY_TRUNCATION_MARKER}{SUMMARY_SUFFIX}",
        label.trim_end()
    )
}

fn truncate_chars(text: &str, max: usize, suffix: &str) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        format!("{}{suffix}", text.chars().take(max).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::AssistantMessage;

    fn call(id: &str, name: &str, arguments: &str) -> Message {
        Message::assistant(AssistantMessage {
            tool_calls: vec![ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: arguments.to_string(),
            }],
            ..AssistantMessage::default()
        })
    }

    fn tool_content_len(messages: &[Message]) -> u32 {
        messages
            .iter()
            .filter_map(|message| match message {
                Message::Tool { content, .. } => Some(content.len() as u32),
                _ => None,
            })
            .sum()
    }

    fn history() -> Vec<Message> {
        vec![
            Message::user("old one"),
            call("a", "read_file", r#"{"path":"src/main.rs"}"#),
            Message::tool("a", "read_file", "a".repeat(300)),
            Message::user("old two"),
            call("b", "grep", r#"{"pattern":"needle"}"#),
            Message::tool("b", "grep", "b".repeat(300)),
            Message::user("current"),
            call("c", "bash", r#"{"command":"cargo test"}"#),
            Message::tool("c", "bash", "current result"),
        ]
    }

    #[test]
    fn below_trigger_is_an_identity_projection() {
        let messages = history();
        let result = trim_previous_tool_results(&messages, 1_000, 100, tool_content_len);
        assert_eq!(result.messages, messages);
        assert_eq!(result.summarized, 0);
    }

    #[test]
    fn trigger_and_target_boundaries_are_inclusive() {
        let messages = history();
        let result = trim_previous_tool_results(&messages, 600, 500, |candidate| {
            if candidate.iter().any(
                |message| matches!(message, Message::Tool { content, .. } if content.starts_with(SUMMARY_PREFIX)),
            ) {
                500
            } else {
                600
            }
        });

        assert_eq!(result.summarized, 1);
        assert_eq!(result.estimated_tokens, 500);
    }

    #[test]
    fn summarizes_oldest_completed_turn_and_stops_at_target() {
        let messages = history();
        let result = trim_previous_tool_results(&messages, 500, 500, tool_content_len);
        assert_eq!(result.summarized, 1);
        let Message::Tool {
            call_id,
            name,
            content,
        } = &result.messages[2]
        else {
            panic!("expected tool result")
        };
        assert_eq!(call_id, "a");
        assert_eq!(name, "read_file");
        assert!(content.starts_with("[Tool call: read_file(path=\"src/main.rs\") → "));
        assert!(
            matches!(&result.messages[5], Message::Tool { content, .. } if content == &"b".repeat(300))
        );
        assert!(
            matches!(&result.messages[8], Message::Tool { content, .. } if content == "current result")
        );
        assert_eq!(messages, history());
    }

    #[test]
    fn summarizes_sibling_tool_results_as_one_atomic_batch() {
        let messages = vec![
            Message::user("old"),
            Message::assistant(AssistantMessage {
                tool_calls: vec![
                    ToolCall {
                        id: "a".to_string(),
                        name: "read_file".to_string(),
                        arguments: "{}".to_string(),
                    },
                    ToolCall {
                        id: "b".to_string(),
                        name: "grep".to_string(),
                        arguments: "{}".to_string(),
                    },
                ],
                ..AssistantMessage::default()
            }),
            Message::tool("a", "read_file", "a".repeat(300)),
            Message::tool("b", "grep", "ok"),
            Message::user("current"),
        ];

        let result = trim_previous_tool_results(&messages, 300, 200, tool_content_len);

        assert_eq!(result.summarized, 2);
        assert!(
            matches!(&result.messages[2], Message::Tool { content, .. } if content.starts_with(SUMMARY_PREFIX))
        );
        assert!(
            matches!(&result.messages[3], Message::Tool { content, .. } if content.starts_with(SUMMARY_PREFIX))
        );
    }

    #[test]
    fn summarizes_all_eligible_results_when_target_is_impossible() {
        let messages = history();
        let result = trim_previous_tool_results(&messages, 1, 0, tool_content_len);
        assert_eq!(result.summarized, 2);
        assert!(
            matches!(&result.messages[2], Message::Tool { content, .. } if content.starts_with(SUMMARY_PREFIX))
        );
        assert!(
            matches!(&result.messages[5], Message::Tool { content, .. } if content.starts_with(SUMMARY_PREFIX))
        );
        assert!(
            matches!(&result.messages[8], Message::Tool { content, .. } if content == "current result")
        );
        assert!(result.estimated_tokens > 0);
    }

    #[test]
    fn summary_is_utf8_safe_bounded_and_idempotent() {
        let messages = vec![
            Message::user("old"),
            call(
                "a",
                "read_file",
                &format!(r#"{{"path":"{}"}}"#, "路".repeat(250)),
            ),
            Message::tool("a", "read_file", "结果".repeat(500)),
            Message::user("current"),
        ];
        let first = trim_previous_tool_results(&messages, 1, 0, tool_content_len);
        let second = trim_previous_tool_results(&first.messages, 1, 0, tool_content_len);
        assert_eq!(first.messages, second.messages);
        assert_eq!(second.summarized, 0);
        let Message::Tool { content, .. } = &first.messages[2] else {
            panic!("expected tool result")
        };
        assert!(content.contains("...") && content.contains("1000 chars"));
        assert!(content.chars().count() <= SUMMARY_TOTAL_MAX);
    }

    #[test]
    fn skips_a_group_when_its_serialized_request_estimate_grows() {
        let messages = vec![
            Message::user("old"),
            call(
                "a",
                "read_file",
                &format!(r#"{{"path":"{}"}}"#, "路".repeat(250)),
            ),
            Message::tool("a", "read_file", "a".repeat(500)),
            Message::user("current"),
        ];
        let estimate = |candidate: &[Message]| {
            let wire = serde_json::Value::Array(candidate.iter().map(Message::to_api).collect());
            crate::context_budget::estimate_json_tokens(&wire)
        };
        let before = estimate(&messages);

        let result = trim_previous_tool_results(&messages, before, 0, estimate);

        assert_eq!(result.messages.as_ref(), messages);
        assert_eq!(result.summarized, 0);
        assert_eq!(result.estimated_tokens, before);
    }

    #[test]
    fn does_not_expand_short_results_or_trim_without_a_turn_boundary() {
        let short = vec![
            Message::user("old"),
            call("a", "read_file", "{}"),
            Message::tool("a", "read_file", "ok"),
            Message::user("current"),
        ];
        let result = trim_previous_tool_results(&short, 1, 0, tool_content_len);
        assert_eq!(result.messages, short);
        assert_eq!(result.summarized, 0);

        let no_user = vec![Message::tool("a", "read_file", "x".repeat(300))];
        let result = trim_previous_tool_results(&no_user, 1, 0, tool_content_len);
        assert_eq!(result.messages, no_user);
        assert_eq!(result.summarized, 0);
    }
}
