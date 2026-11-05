//! Conversation and tool-call types shared by the client, the agent and the
//! session store.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A tool call requested by the model.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    /// Provider-assigned id, echoed back with the tool result.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Raw JSON arguments, exactly as streamed by the provider.
    pub arguments: String,
}

impl ToolCall {
    /// Parse the streamed arguments. Models occasionally emit invalid JSON; the
    /// error message is returned to the model as a tool result instead of
    /// aborting the turn.
    pub fn parse_arguments(&self) -> Result<Value, serde_json::Error> {
        let raw = self.arguments.trim();
        if raw.is_empty() {
            return Ok(Value::Object(Default::default()));
        }
        serde_json::from_str(raw)
    }
}

/// An assistant turn: optional text, optional reasoning and zero or more tool
/// calls.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AssistantMessage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Provider-native Responses output items retained for stateless replay.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub responses_items: Vec<Value>,
}

impl AssistantMessage {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            reasoning: None,
            tool_calls: Vec::new(),
            responses_items: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.content
            .as_deref()
            .map(str::trim)
            .is_none_or(str::is_empty)
            && self.tool_calls.is_empty()
    }
}

/// A message in the model conversation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: String,
    },
    /// Persisted compaction context sent to providers with the user role.
    Summary {
        content: String,
    },
    Assistant {
        message: AssistantMessage,
    },
    Tool {
        call_id: String,
        name: String,
        content: String,
    },
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    pub(crate) fn summary(content: impl Into<String>) -> Self {
        Self::Summary {
            content: content.into(),
        }
    }

    pub fn assistant(message: AssistantMessage) -> Self {
        Self::Assistant { message }
    }

    pub fn tool(
        call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self::Tool {
            call_id: call_id.into(),
            name: name.into(),
            content: content.into(),
        }
    }

    /// Serialise into the OpenAI chat-completions wire format.
    pub fn to_api(&self) -> Value {
        match self {
            Message::System { content } => json!({"role": "system", "content": content}),
            Message::User { content } | Message::Summary { content } => {
                json!({"role": "user", "content": content})
            }
            Message::Assistant { message } => {
                let mut value = json!({"role": "assistant"});
                value["content"] = match &message.content {
                    Some(content) => Value::String(content.clone()),
                    None => Value::Null,
                };
                if !message.tool_calls.is_empty() {
                    value["tool_calls"] = Value::Array(
                        message
                            .tool_calls
                            .iter()
                            .map(|call| {
                                json!({
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": call.arguments,
                                    }
                                })
                            })
                            .collect(),
                    );
                }
                // `reasoning_content` is deliberately not sent back: DeepSeek
                // documents that it must not be included in later requests.
                value
            }
            Message::Tool {
                call_id, content, ..
            } => json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            }),
        }
    }

    /// Short single-line rendering used by transcripts and tests.
    pub fn preview(&self) -> String {
        let text = match self {
            Message::System { content }
            | Message::User { content }
            | Message::Summary { content } => content.clone(),
            Message::Assistant { message } => message.content.clone().unwrap_or_default(),
            Message::Tool { content, .. } => content.clone(),
        };
        let mut line = text.lines().next().unwrap_or_default().trim().to_string();
        if line.chars().count() > 100 {
            line = line.chars().take(100).collect();
            line.push('…');
        }
        line
    }
}

/// A tool exposed to the model.
#[derive(Clone, Debug, Serialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolDef {
    pub fn to_api(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

/// Token accounting reported by the provider.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

impl Usage {
    pub fn add(&mut self, other: Usage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
    }
}

/// The result of one model call.
#[derive(Clone, Debug)]
pub struct Completion {
    pub message: AssistantMessage,
    pub usage: Usage,
    pub finish_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_truncates_multibyte_text_without_panicking() {
        let message = Message::user("中".repeat(150));
        let preview = message.preview();
        assert!(preview.chars().count() <= 101, "{preview}");
        assert!(preview.ends_with('…'));
    }

    #[test]
    fn assistant_serialisation_keeps_tool_calls_and_nulls_content() {
        let message = Message::assistant(AssistantMessage {
            content: None,
            reasoning: None,
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            }],
            responses_items: Vec::new(),
        });
        let value = message.to_api();
        assert_eq!(value["role"], "assistant");
        assert!(value["content"].is_null());
        assert_eq!(value["tool_calls"][0]["function"]["name"], "read_file");
    }
}
