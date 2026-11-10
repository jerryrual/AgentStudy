//! Deterministic request-only context projections.

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};

use crate::compaction::{summary_message, CompressionBatch, CompressionFold};
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
    pub(crate) removed: usize,
    pub(crate) compressed: usize,
    pub(crate) compression: CompressionBatch,
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
            removed: 0,
            compressed: 0,
            compression: CompressionBatch::default(),
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
            removed: 0,
            compressed: 0,
            compression: CompressionBatch::default(),
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
        removed: 0,
        compressed: 0,
        compression: CompressionBatch::default(),
        estimated_tokens,
    }
}

/// Remove completed historical tool-call batches until `target_tokens`.
///
/// Each candidate removes one assistant batch atomically, including all of its
/// sibling calls, matching results, and call-associated reasoning. Ordinary
/// assistant content and provider-native message items are retained.
pub(crate) fn remove_previous_tool_groups<'a, F>(
    mut projection: TrimResult<'a>,
    target_tokens: u32,
    mut estimate: F,
) -> TrimResult<'a>
where
    F: FnMut(&[Message]) -> u32,
{
    while projection.estimated_tokens > target_tokens {
        let Some(current_turn_start) = projection
            .messages
            .iter()
            .rposition(|message| matches!(message, Message::User { .. }))
        else {
            break;
        };
        let groups = removable_tool_groups(projection.messages.as_ref(), current_turn_start);
        let mut committed = false;

        for group in groups {
            let candidate = remove_tool_group(projection.messages.as_ref(), &group);
            let candidate_tokens = estimate(&candidate);
            if candidate_tokens >= projection.estimated_tokens {
                continue;
            }
            projection.messages = Cow::Owned(candidate);
            projection.removed += 1;
            projection.estimated_tokens = candidate_tokens;
            committed = true;
            break;
        }

        if !committed {
            break;
        }
    }
    projection
}

/// Replace completed historical turns with deterministic request-only digests.
pub(crate) fn compress_previous_text_turns<'a, F>(
    mut projection: TrimResult<'a>,
    source: &[Message],
    target_tokens: u32,
    estimate: F,
) -> TrimResult<'a>
where
    F: FnMut(&[Message]) -> u32,
{
    if projection.estimated_tokens <= target_tokens {
        return projection;
    }

    let source_users = user_indices(source);
    let projected_users = user_indices(projection.messages.as_ref());
    if source_users.len() != projected_users.len() || source_users.len() < 2 {
        return projection;
    }

    let baseline = projection.clone();
    let summary_boundary = source
        .iter()
        .rposition(|message| matches!(message, Message::Summary { .. }));
    let first_eligible = summary_boundary.map_or(0, |boundary| {
        source_users
            .iter()
            .position(|index| *index > boundary)
            .unwrap_or(source_users.len())
    });
    let mut estimate = estimate;

    for ordinal in first_eligible..source_users.len() - 1 {
        if projection.estimated_tokens <= target_tokens {
            break;
        }

        let source_start = source_users[ordinal];
        let source_end = source_users[ordinal + 1];
        let projected_boundary = projection
            .messages
            .iter()
            .rposition(|message| matches!(message, Message::Summary { .. }));
        let mut projected_users = projection
            .messages
            .iter()
            .enumerate()
            .skip(projected_boundary.map_or(0, |index| index + 1))
            .filter_map(|(index, message)| {
                matches!(message, Message::User { .. }).then_some(index)
            });
        let (Some(projected_start), Some(projected_end)) =
            (projected_users.next(), projected_users.next())
        else {
            break;
        };
        if source[source_start..source_end]
            .iter()
            .chain(projection.messages[projected_start..projected_end].iter())
            .any(|message| matches!(message, Message::System { .. } | Message::Summary { .. }))
        {
            break;
        }

        let summary = automatic_turn_summary(&source[source_start..source_end]);
        let fold = CompressionFold {
            summary: summary.clone(),
            compacted_messages: source_end - source_start,
        };
        let mut candidate = projection.messages.as_ref().to_vec();
        candidate.splice(projected_start..projected_end, [summary_message(&summary)]);
        let candidate_tokens = estimate(&candidate);

        projection.messages = Cow::Owned(candidate);
        projection.compressed += 1;
        projection.compression.folds.push(fold);
        projection.estimated_tokens = candidate_tokens;
    }

    if projection.estimated_tokens >= baseline.estimated_tokens {
        baseline
    } else {
        projection
    }
}

fn user_indices(messages: &[Message]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| matches!(message, Message::User { .. }).then_some(index))
        .collect()
}

pub(crate) fn automatic_turn_summary(turn: &[Message]) -> String {
    let user_messages: Vec<_> = turn
        .iter()
        .filter_map(|message| match message {
            Message::User { content } if !content.is_empty() => Some(content.as_str()),
            _ => None,
        })
        .collect();
    let injected_user_messages: Vec<_> = turn
        .iter()
        .filter_map(|message| match message {
            Message::InjectedUser { content, .. } if !content.is_empty() => Some(content.as_str()),
            _ => None,
        })
        .collect();
    let mut parts = Vec::new();
    if let Some(content) = user_messages.first() {
        parts.push(format!("User: {}", truncate_chars(content, 100, "...")));
        if user_messages.len() > 1 {
            parts.push(format!("(+{} more messages)", user_messages.len() - 1));
        }
    }
    const MAX_INJECTED_SUMMARY_MESSAGES: usize = 3;
    parts.extend(
        injected_user_messages
            .iter()
            .take(MAX_INJECTED_SUMMARY_MESSAGES)
            .map(|content| format!("Injected user: {}", truncate_chars(content, 100, "..."))),
    );
    if injected_user_messages.len() > MAX_INJECTED_SUMMARY_MESSAGES {
        parts.push(format!(
            "(+{} more injected messages)",
            injected_user_messages.len() - MAX_INJECTED_SUMMARY_MESSAGES
        ));
    }

    let mut tools = BTreeSet::new();
    for message in turn {
        if let Message::Assistant { message } = message {
            tools.extend(message.tool_calls.iter().map(|call| call.name.as_str()));
            tools.extend(message.responses_items.iter().filter_map(|item| {
                (item.get("type").and_then(serde_json::Value::as_str) == Some("function_call"))
                    .then(|| item.get("name")?.as_str())
                    .flatten()
            }));
        }
    }
    if !tools.is_empty() {
        parts.push(format!(
            "Tools: {}",
            tools.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }

    if parts.is_empty() {
        "Earlier conversation".to_string()
    } else {
        parts.join("; ")
    }
}

#[derive(Debug)]
struct ToolGroup {
    assistant_index: usize,
    call_ids: HashSet<String>,
    result_indices: HashSet<usize>,
}

fn removable_tool_groups(messages: &[Message], end: usize) -> Vec<ToolGroup> {
    let mut groups = Vec::<ToolGroup>::new();
    let mut group_by_call = HashMap::<String, usize>::new();
    for (index, message) in messages[..end].iter().enumerate() {
        match message {
            Message::Assistant { message } => {
                let mut call_ids: HashSet<String> = message
                    .tool_calls
                    .iter()
                    .map(|call| call.id.clone())
                    .collect();
                call_ids.extend(
                    message
                        .responses_items
                        .iter()
                        .filter_map(native_function_call_id),
                );
                if call_ids.is_empty() {
                    continue;
                }
                let group_index = groups.len();
                for call_id in &call_ids {
                    group_by_call.insert(call_id.clone(), group_index);
                }
                groups.push(ToolGroup {
                    assistant_index: index,
                    call_ids,
                    result_indices: HashSet::new(),
                });
            }
            Message::Tool { call_id, .. } => {
                if let Some(group_index) = group_by_call.get(call_id).copied() {
                    groups[group_index].result_indices.insert(index);
                }
            }
            _ => {}
        }
    }
    groups.retain(|group| {
        group.call_ids.iter().all(|call_id| {
            group.result_indices.iter().any(|index| {
                matches!(&messages[*index], Message::Tool { call_id: result_id, .. } if result_id == call_id)
            })
        })
    });
    groups
}

fn native_function_call_id(item: &serde_json::Value) -> Option<String> {
    (item.get("type").and_then(serde_json::Value::as_str) == Some("function_call"))
        .then(|| item.get("call_id")?.as_str().map(str::to_owned))
        .flatten()
}

fn remove_tool_group(messages: &[Message], group: &ToolGroup) -> Vec<Message> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| match message {
            Message::Tool { .. } if group.result_indices.contains(&index) => None,
            Message::Assistant { message } if index == group.assistant_index => {
                let mut message = message.clone();
                message
                    .tool_calls
                    .retain(|call| !group.call_ids.contains(&call.id));
                message.responses_items.retain(|item| {
                    let item_type = item.get("type").and_then(serde_json::Value::as_str);
                    item_type != Some("reasoning")
                        && !(item_type == Some("function_call")
                            && item
                                .get("call_id")
                                .and_then(serde_json::Value::as_str)
                                .is_some_and(|call_id| group.call_ids.contains(call_id)))
                });
                message.reasoning = None;
                (message
                    .content
                    .as_deref()
                    .is_some_and(|content| !content.trim().is_empty())
                    || !message.tool_calls.is_empty()
                    || !message.responses_items.is_empty())
                .then_some(Message::assistant(message))
            }
            _ => Some(message.clone()),
        })
        .collect()
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
        assert_eq!(result.removed, 0);
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

    fn phase_two<'a>(messages: &'a [Message], target: u32) -> TrimResult<'a> {
        remove_previous_tool_groups(
            TrimResult {
                messages: Cow::Borrowed(messages),
                summarized: 0,
                removed: 0,
                compressed: 0,
                compression: CompressionBatch::default(),
                estimated_tokens: messages.len() as u32,
            },
            target,
            |candidate| candidate.len() as u32,
        )
    }

    #[test]
    fn phase_one_target_stops_before_phase_two() {
        let messages = history();
        let phase_one = trim_previous_tool_results(&messages, 600, 500, |candidate| {
            if candidate.iter().any(
                |message| matches!(message, Message::Tool { content, .. } if content.starts_with(SUMMARY_PREFIX)),
            ) { 500 } else { 600 }
        });
        let result = if phase_one.estimated_tokens > 500 {
            remove_previous_tool_groups(phase_one, 500, |_| 0)
        } else {
            phase_one
        };
        assert_eq!(result.summarized, 1);
        assert_eq!(result.removed, 0);
    }

    #[test]
    fn phase_two_is_oldest_first_exact_and_current_safe() {
        let messages = history();
        let source = messages.clone();
        let result = phase_two(&messages, (messages.len() - 2) as u32);
        assert_eq!(result.removed, 1);
        assert_eq!(result.estimated_tokens, (messages.len() - 2) as u32);
        assert!(!result
            .messages
            .iter()
            .any(|message| matches!(message, Message::Tool { call_id, .. } if call_id == "a")));
        assert!(result
            .messages
            .iter()
            .any(|message| matches!(message, Message::Tool { call_id, .. } if call_id == "b")));
        assert!(result
            .messages
            .iter()
            .any(|message| matches!(message, Message::Tool { call_id, .. } if call_id == "c")));
        assert_eq!(messages, source);
        assert!(matches!(result.messages, Cow::Owned(_)));
    }

    #[test]
    fn phase_two_scopes_reused_call_ids_to_the_historical_batch() {
        let messages = vec![
            Message::user("old"),
            call("shared", "read_file", "{}"),
            Message::tool("shared", "read_file", "old result"),
            Message::user("current"),
            call("shared", "read_file", "{}"),
            Message::tool("shared", "read_file", "current result"),
        ];

        let result = phase_two(&messages, 4);

        assert_eq!(result.removed, 1);
        assert_eq!(result.messages.len(), 4);
        assert!(result.messages.iter().any(
            |message| matches!(message, Message::Tool { call_id, content, .. } if call_id == "shared" && content == "current result"),
        ));
        assert!(!result.messages.iter().any(
            |message| matches!(message, Message::Tool { content, .. } if content == "old result"),
        ));
    }

    #[test]
    fn phase_two_scopes_reused_call_ids_across_historical_batches() {
        let messages = vec![
            Message::user("old one"),
            call("shared", "read_file", "{}"),
            Message::tool("shared", "read_file", "first result"),
            Message::user("old two"),
            call("shared", "read_file", "{}"),
            Message::tool("shared", "read_file", "second result"),
            Message::user("current"),
        ];

        let result = phase_two(&messages, 5);

        assert_eq!(result.removed, 1);
        assert!(!result.messages.iter().any(
            |message| matches!(message, Message::Tool { content, .. } if content == "first result"),
        ));
        assert!(result.messages.iter().any(
            |message| matches!(message, Message::Tool { content, .. } if content == "second result"),
        ));
    }

    #[test]
    fn phase_two_retains_an_incomplete_older_reused_id_batch() {
        let messages = vec![
            Message::user("old incomplete"),
            call("shared", "read_file", "{}"),
            Message::user("old complete"),
            call("shared", "read_file", "{}"),
            Message::tool("shared", "read_file", "completed result"),
            Message::user("current"),
        ];

        let result = phase_two(&messages, 4);

        assert_eq!(result.removed, 1);
        assert!(!result.messages.iter().any(
            |message| matches!(message, Message::Tool { content, .. } if content == "completed result"),
        ));
        assert!(matches!(
            &result.messages[1],
            Message::Assistant { message } if message.tool_calls.iter().any(|call| call.id == "shared")
        ));
    }

    #[test]
    fn phase_two_removes_parallel_siblings_atomically() {
        let messages = vec![
            Message::user("old"),
            Message::assistant(AssistantMessage {
                tool_calls: vec![
                    ToolCall {
                        id: "a".into(),
                        name: "read_file".into(),
                        arguments: "{}".into(),
                    },
                    ToolCall {
                        id: "b".into(),
                        name: "grep".into(),
                        arguments: "{}".into(),
                    },
                ],
                ..AssistantMessage::default()
            }),
            Message::tool("a", "read_file", "one"),
            Message::tool("b", "grep", "two"),
            Message::user("current"),
        ];
        let result = phase_two(&messages, 2);
        assert_eq!(result.removed, 1);
        assert!(!result
            .messages
            .iter()
            .any(|message| matches!(message, Message::Tool { .. })));
    }

    #[test]
    fn phase_two_keeps_an_incomplete_sibling_batch_intact() {
        let messages = vec![
            Message::user("old"),
            Message::assistant(AssistantMessage {
                tool_calls: vec![
                    ToolCall {
                        id: "a".into(),
                        name: "read_file".into(),
                        arguments: "{}".into(),
                    },
                    ToolCall {
                        id: "b".into(),
                        name: "grep".into(),
                        arguments: "{}".into(),
                    },
                ],
                ..AssistantMessage::default()
            }),
            Message::tool("a", "read_file", "one"),
            Message::user("current"),
        ];

        let result = phase_two(&messages, 0);

        assert_eq!(result.messages.as_ref(), messages);
        assert_eq!(result.removed, 0);
    }

    #[test]
    fn phase_two_preserves_fused_narration() {
        let messages = vec![
            Message::user("old"),
            Message::assistant(AssistantMessage {
                content: Some("I checked the file.".into()),
                reasoning: Some("tool-specific reasoning".into()),
                tool_calls: vec![ToolCall {
                    id: "a".into(),
                    name: "read_file".into(),
                    arguments: "{}".into(),
                }],
                responses_items: Vec::new(),
                responses_continuation: None,
            }),
            Message::tool("a", "read_file", "result"),
            Message::user("current"),
        ];
        let result = phase_two(&messages, 3);
        let Message::Assistant { message } = &result.messages[1] else {
            panic!("narration should survive")
        };
        assert_eq!(message.content.as_deref(), Some("I checked the file."));
        assert_eq!(message.reasoning, None);
        assert!(message.tool_calls.is_empty());
    }

    #[test]
    fn phase_two_filters_native_call_and_reasoning_but_keeps_text() {
        let text = serde_json::json!({"type": "message", "content": [{"type": "output_text", "text": "kept"}]});
        let messages = vec![
            Message::user("old"),
            Message::assistant(AssistantMessage {
                responses_items: vec![
                    serde_json::json!({"type": "reasoning", "encrypted_content": "secret"}),
                    serde_json::json!({"type": "function_call", "call_id": "a", "name": "read_file", "arguments": "{}"}),
                    text.clone(),
                ],
                ..AssistantMessage::default()
            }),
            Message::tool("a", "read_file", "result"),
            Message::user("current"),
        ];
        let result = phase_two(&messages, 3);
        let Message::Assistant { message } = &result.messages[1] else {
            panic!("text item should survive")
        };
        assert_eq!(message.responses_items, vec![text]);
        assert_eq!(result.removed, 1);
    }

    #[test]
    fn phase_two_stops_on_non_reducing_candidate() {
        let messages = vec![
            Message::user("old"),
            call("a", "read_file", "{}"),
            Message::tool("a", "read_file", "result"),
            Message::user("current"),
        ];
        let result = remove_previous_tool_groups(
            TrimResult {
                messages: Cow::Borrowed(&messages),
                summarized: 0,
                removed: 0,
                compressed: 0,
                compression: CompressionBatch::default(),
                estimated_tokens: 100,
            },
            10,
            |_| 100,
        );
        assert_eq!(result.messages, messages);
        assert_eq!(result.removed, 0);
        assert_eq!(result.estimated_tokens, 100);
    }

    #[test]
    fn phase_two_stops_when_non_tool_budget_is_impossible() {
        let messages = vec![Message::system("fixed overhead"), Message::user("current")];
        let result = remove_previous_tool_groups(
            TrimResult {
                messages: Cow::Borrowed(&messages),
                summarized: 0,
                removed: 0,
                compressed: 0,
                compression: CompressionBatch::default(),
                estimated_tokens: 100,
            },
            10,
            |_| 100,
        );
        assert_eq!(result.messages, messages);
        assert_eq!(result.removed, 0);
        assert_eq!(result.estimated_tokens, 100);
    }

    fn phase_three<'a>(messages: &'a [Message], tokens: u32) -> TrimResult<'a> {
        TrimResult {
            messages: Cow::Borrowed(messages),
            summarized: 0,
            removed: 0,
            compressed: 0,
            compression: CompressionBatch::default(),
            estimated_tokens: tokens,
        }
    }

    #[test]
    fn phase_three_is_oldest_first_target_safe_and_request_only() {
        let messages = vec![
            Message::system("fixed"),
            Message::user("old one"),
            Message::assistant(AssistantMessage::text("discarded")),
            Message::tool("a", "read", "discarded result"),
            Message::user("old two"),
            Message::assistant(AssistantMessage::text("also discarded")),
            Message::user("current"),
            Message::assistant(AssistantMessage::text("current answer")),
        ];
        let source = messages.clone();
        let result = compress_previous_text_turns(
            phase_three(&messages, messages.len() as u32),
            &messages,
            (messages.len() - 2) as u32,
            |candidate| candidate.len() as u32,
        );

        assert_eq!(result.compressed, 1);
        assert!(matches!(result.messages, Cow::Owned(_)));
        assert_eq!(result.messages[0], Message::system("fixed"));
        assert_eq!(result.messages[1], summary_message("User: old one"));
        assert_eq!(
            &result.messages[result.messages.len() - 2..],
            &messages[6..]
        );
        assert_eq!(messages, source);
    }

    #[test]
    fn phase_three_summary_matches_upstream_digest() {
        let long = "界".repeat(101);
        let digest = automatic_turn_summary(&[
            Message::user(format!("  {long}  ")),
            Message::assistant(AssistantMessage {
                content: Some("ordinary text omitted".into()),
                reasoning: Some("reasoning omitted".into()),
                tool_calls: vec![
                    ToolCall {
                        id: "z".into(),
                        name: "zeta".into(),
                        arguments: "{}".into(),
                    },
                    ToolCall {
                        id: "a".into(),
                        name: "alpha".into(),
                        arguments: "{}".into(),
                    },
                ],
                responses_items: vec![
                    serde_json::json!({"type": "function_call", "name": "beta"}),
                    serde_json::json!({"type": "function_call", "name": "alpha"}),
                ],
                responses_continuation: None,
            }),
        ]);
        assert_eq!(
            digest,
            format!("User:   {}...; Tools: alpha, beta, zeta", "界".repeat(98))
        );
        assert_eq!(
            automatic_turn_summary(&[Message::assistant(AssistantMessage::text("ignored"))]),
            "Earlier conversation"
        );
        assert_eq!(
            automatic_turn_summary(&[Message::user("first"), Message::user("second")]),
            "User: first; (+1 more messages)"
        );
    }

    #[test]
    fn historical_turn_summary_keeps_injected_user_content_without_a_new_boundary() {
        let digest = automatic_turn_summary(&[
            Message::user("primary request"),
            Message::assistant(AssistantMessage::text("working")),
            Message::injected_user("inject-1", "also preserve this constraint"),
        ]);

        assert_eq!(
            digest,
            "User: primary request; Injected user: also preserve this constraint"
        );
        assert_eq!(
            user_indices(&[
                Message::user("primary request"),
                Message::injected_user("inject-1", "same turn"),
            ]),
            vec![0]
        );
        assert_eq!(
            automatic_turn_summary(&[
                Message::user("primary"),
                Message::injected_user("inject-1", "one"),
                Message::injected_user("inject-2", "two"),
                Message::injected_user("inject-3", "three"),
                Message::injected_user("inject-4", "four"),
            ]),
            "User: primary; Injected user: one; Injected user: two; Injected user: three; (+1 more injected messages)"
        );
    }

    #[test]
    fn phase_three_aligns_phase_two_projection_with_canonical_source() {
        let source = vec![
            Message::user("inspect"),
            Message::assistant(AssistantMessage {
                content: Some("kept narration".into()),
                tool_calls: vec![ToolCall {
                    id: "a".into(),
                    name: "canonical_tool".into(),
                    arguments: "{}".into(),
                }],
                ..AssistantMessage::default()
            }),
            Message::tool("a", "projection_name", "result"),
            Message::user("current"),
        ];
        let phase_two = remove_previous_tool_groups(phase_three(&source, 100), 99, |_| 90);
        let result = compress_previous_text_turns(phase_two, &source, 50, |candidate| {
            if candidate.iter().any(|message| matches!(message, Message::Summary { content } if content.contains("User: inspect"))) { 40 } else { 90 }
        });

        assert_eq!(result.compressed, 1);
        assert!(matches!(
            &result.messages[0],
            Message::Summary { content }
                if content == "[Earlier conversation summary]\nUser: inspect; Tools: canonical_tool"
        ));
        assert_eq!(result.messages[1], Message::user("current"));
    }

    #[test]
    fn phase_three_stops_at_a_system_turn_boundary() {
        let source = vec![
            Message::user("system turn"),
            Message::system("embedded"),
            Message::user("nonreducing"),
            Message::assistant(AssistantMessage::text("one")),
            Message::user("reducing"),
            Message::assistant(AssistantMessage::text("two")),
            Message::user("current"),
        ];
        let result = compress_previous_text_turns(phase_three(&source, 100), &source, 50, |_| 40);

        assert_eq!(result.compressed, 0);
        assert!(result.compression.is_empty());
        assert_eq!(result.messages.as_ref(), source);
    }

    #[test]
    fn phase_three_accepts_a_nonreducing_prefix_when_the_batch_reduces() {
        let source = vec![
            Message::user("ok"),
            Message::assistant(AssistantMessage::text("done")),
            Message::user("large historical request"),
            Message::assistant(AssistantMessage::text("large historical response")),
            Message::user("current"),
        ];
        let result =
            compress_previous_text_turns(phase_three(&source, 100), &source, 50, |candidate| {
                match candidate
                    .iter()
                    .filter(|message| matches!(message, Message::Summary { .. }))
                    .count()
                {
                    1 => 110,
                    2 => 40,
                    _ => 100,
                }
            });

        assert_eq!(result.compressed, 2);
        assert_eq!(result.estimated_tokens, 40);
        assert_eq!(result.compression.folds.len(), 2);
        assert!(matches!(result.messages[0], Message::Summary { .. }));
        assert!(matches!(result.messages[1], Message::Summary { .. }));
        assert_eq!(result.messages[2], Message::user("current"));

        let no_gain = compress_previous_text_turns(phase_three(&source, 100), &source, 50, |_| 100);
        assert_eq!(no_gain.compressed, 0);
        assert!(no_gain.compression.is_empty());
        assert_eq!(no_gain.messages.as_ref(), source);
    }

    #[test]
    fn phase_three_tracks_indices_across_deletions_skips_and_multiple_commits() {
        let source = vec![
            Message::user("tool-only turn"),
            call("a", "read_file", "{}"),
            Message::tool("a", "read_file", "result"),
            Message::user("first text turn"),
            Message::assistant(AssistantMessage::text("first answer")),
            Message::user("system-bearing turn"),
            Message::system("must stay fixed"),
            Message::assistant(AssistantMessage::text("system answer")),
            Message::user("second text turn"),
            Message::assistant(AssistantMessage::text("second answer")),
            Message::user("current"),
            Message::assistant(AssistantMessage::text("current answer")),
        ];
        let projection = vec![
            Message::user("tool-only turn"),
            Message::user("first text turn"),
            Message::assistant(AssistantMessage::text("first answer")),
            Message::user("system-bearing turn"),
            Message::system("must stay fixed"),
            Message::assistant(AssistantMessage::text("system answer")),
            Message::user("second text turn"),
            Message::assistant(AssistantMessage::text("second answer")),
            Message::user("current"),
            Message::assistant(AssistantMessage::text("current answer")),
        ];
        let source_snapshot = source.clone();
        let result =
            compress_previous_text_turns(phase_three(&projection, 10), &source, 8, |candidate| {
                let summaries = candidate
                    .iter()
                    .filter(|message| matches!(message, Message::Summary { .. }))
                    .count();
                (candidate.len() - summaries) as u32
            });

        assert_eq!(result.compressed, 2);
        assert_eq!(result.estimated_tokens, 7);
        assert_eq!(
            result.compression.folds,
            vec![
                CompressionFold {
                    summary: "User: tool-only turn; Tools: read_file".into(),
                    compacted_messages: 3,
                },
                CompressionFold {
                    summary: "User: first text turn".into(),
                    compacted_messages: 2,
                },
            ]
        );
        assert_eq!(
            result.messages.as_ref(),
            &[
                summary_message("User: tool-only turn; Tools: read_file"),
                summary_message("User: first text turn"),
                Message::user("system-bearing turn"),
                Message::system("must stay fixed"),
                Message::assistant(AssistantMessage::text("system answer")),
                Message::user("second text turn"),
                Message::assistant(AssistantMessage::text("second answer")),
                Message::user("current"),
                Message::assistant(AssistantMessage::text("current answer")),
            ]
        );
        assert_eq!(source, source_snapshot);
    }

    #[test]
    fn phase_three_boundary_impossible_and_repeat_are_deterministic() {
        let source = vec![
            Message::user("before summary"),
            Message::assistant(AssistantMessage::text("old")),
            crate::compaction::summary_message("manual"),
            Message::assistant(AssistantMessage::text("boundary")),
            Message::user("after summary"),
            Message::assistant(AssistantMessage::text("eligible")),
            Message::user("current"),
        ];
        let first =
            compress_previous_text_turns(phase_three(&source, 100), &source, 0, |candidate| {
                candidate.len() as u32
            });
        assert_eq!(first.compressed, 1);
        assert!(first.messages.iter().any(
            |message| matches!(message, Message::User { content } if content == "before summary")
        ));
        assert!(first
            .messages
            .iter()
            .any(|message| matches!(message, Message::Summary { .. })));
        let repeated = compress_previous_text_turns(first.clone(), &source, 0, |candidate| {
            candidate.len() as u32
        });
        assert_eq!(repeated, first);

        let no_user = vec![Message::system("fixed")];
        let impossible =
            compress_previous_text_turns(phase_three(&no_user, 100), &no_user, 0, |_| 100);
        assert_eq!(impossible.messages.as_ref(), no_user);
        assert_eq!(impossible.compressed, 0);
    }
}
