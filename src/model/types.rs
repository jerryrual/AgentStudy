//! Conversation and tool-call types shared by the client, the agent and the
//! session store.

use std::collections::HashSet;
use std::io::Cursor;

use anyhow::{ensure, Context, Result};
use base64::prelude::{Engine as _, BASE64_STANDARD};
use image::{ImageFormat, ImageReader, Limits};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

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

/// A stored Responses handle, scoped to the model endpoint that created it.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResponsesContinuation {
    pub id: String,
    /// SHA-256 of endpoint, model, authentication identity and system prompt.
    pub scope_sha256: String,
    /// SHA-256 of the local message prefix sent before this response.
    #[serde(default)]
    pub input_sha256: String,
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
    /// Stored response id and its provider identity for server-side continuation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub responses_continuation: Option<Box<ResponsesContinuation>>,
}

impl AssistantMessage {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            reasoning: None,
            tool_calls: Vec::new(),
            responses_items: Vec::new(),
            responses_continuation: None,
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

pub(crate) const MAX_IMAGE_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const MAX_PROMPT_IMAGE_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_PROMPT_IMAGES: usize = 16;
pub(crate) const MAX_SESSION_IMAGE_BYTES: usize = 32 * 1024 * 1024;
pub(crate) const MAX_SESSION_IMAGES: usize = 256;
const MAX_ENCODED_IMAGE_BYTES: usize = MAX_IMAGE_BYTES.div_ceil(3) * 4;
const MAX_IMAGE_DIMENSION: u32 = 8_192;
const MAX_IMAGE_DECODE_ALLOC: u64 = 128 * 1024 * 1024;

/// One embedded image supplied by a user-facing protocol.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageInput {
    pub media_type: String,
    /// Canonical base64 payload without a data-URL prefix.
    pub data: String,
    #[serde(skip)]
    pub(crate) validated_key: Option<[u8; 32]>,
}

impl PartialEq for ImageInput {
    fn eq(&self, other: &Self) -> bool {
        self.media_type == other.media_type && self.data == other.data
    }
}

impl Eq for ImageInput {}

impl ImageInput {
    pub fn data_url(&self) -> String {
        format!("data:{};base64,{}", self.media_type, self.data)
    }

    pub(crate) fn validated(
        media_type: String,
        data: String,
        remaining_bytes: usize,
    ) -> Result<(Self, usize)> {
        let image = Self {
            media_type,
            data,
            validated_key: None,
        };
        let (format, decoded) = image.decode_payload()?;
        Self::validated_decoded(image, format, decoded, remaining_bytes)
    }

    pub(crate) fn validated_bytes(
        decoded: Vec<u8>,
        remaining_bytes: usize,
    ) -> Result<(Self, usize)> {
        ensure!(!decoded.is_empty(), "image file must not be empty");
        ensure!(
            decoded.len() <= MAX_IMAGE_BYTES,
            "image exceeds the {MAX_IMAGE_BYTES}-byte decoded limit"
        );
        let format =
            image::guess_format(&decoded).context("unsupported or unrecognized image format")?;
        let media_type = match format {
            ImageFormat::Png => "image/png",
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::Gif => "image/gif",
            ImageFormat::WebP => "image/webp",
            _ => anyhow::bail!("unsupported image format: {format:?}"),
        };
        let image = Self {
            media_type: media_type.to_string(),
            data: BASE64_STANDARD.encode(&decoded),
            validated_key: None,
        };
        Self::validated_decoded(image, format, decoded, remaining_bytes)
    }

    fn validated_decoded(
        mut image: Self,
        format: ImageFormat,
        decoded: Vec<u8>,
        remaining_bytes: usize,
    ) -> Result<(Self, usize)> {
        ensure!(
            decoded.len() <= remaining_bytes,
            "prompt images exceed their aggregate limit"
        );
        image.decode_image(format, &decoded)?;
        image.validated_key = Some(validation_cache_key(&image.media_type, &decoded));
        Ok((image, decoded.len()))
    }

    fn validate_persisted(
        &self,
        remaining_bytes: usize,
        validated: &mut HashSet<[u8; 32]>,
    ) -> Result<usize> {
        let (format, decoded) = self.decode_payload()?;
        ensure!(
            decoded.len() <= remaining_bytes,
            "image bytes exceed their remaining limit"
        );
        let key = validation_cache_key(&self.media_type, &decoded);
        if self.validated_key != Some(key) && !validated.contains(&key) {
            self.decode_image(format, &decoded)?;
        }
        validated.insert(key);
        Ok(decoded.len())
    }

    fn decode_payload(&self) -> Result<(ImageFormat, Vec<u8>)> {
        let format = match self.media_type.as_str() {
            "image/png" => ImageFormat::Png,
            "image/jpeg" => ImageFormat::Jpeg,
            "image/gif" => ImageFormat::Gif,
            "image/webp" => ImageFormat::WebP,
            _ => anyhow::bail!("unsupported ACP image MIME type: {}", self.media_type),
        };
        ensure!(!self.data.is_empty(), "ACP image data must not be empty");
        ensure!(
            self.data.len() <= MAX_ENCODED_IMAGE_BYTES,
            "ACP image exceeds the {MAX_IMAGE_BYTES}-byte decoded limit"
        );
        let decoded = BASE64_STANDARD
            .decode(&self.data)
            .context("ACP image data is not valid padded base64")?;
        ensure!(!decoded.is_empty(), "ACP image data must not be empty");
        ensure!(
            decoded.len() <= MAX_IMAGE_BYTES,
            "ACP image exceeds the {MAX_IMAGE_BYTES}-byte decoded limit"
        );
        Ok((format, decoded))
    }

    fn decode_image(&self, format: ImageFormat, decoded: &[u8]) -> Result<()> {
        let mut limits = Limits::default();
        limits.max_image_width = Some(MAX_IMAGE_DIMENSION);
        limits.max_image_height = Some(MAX_IMAGE_DIMENSION);
        limits.max_alloc = Some(MAX_IMAGE_DECODE_ALLOC);
        let mut reader = ImageReader::with_format(Cursor::new(decoded), format);
        reader.limits(limits);
        reader
            .decode()
            .with_context(|| format!("image data is not a valid {} image", self.media_type))?;
        Ok(())
    }

    fn placeholder(&self) -> String {
        format!("[image: {}]", self.media_type)
    }
}

fn validation_cache_key(media_type: &str, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(media_type.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    hasher.finalize().into()
}

/// An ordered part of a multimodal user message.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContentPart {
    Text { text: String },
    Image { image: ImageInput },
}

/// User content retains the legacy string representation for text-only turns.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Parts(Vec<UserContentPart>),
}

impl UserContent {
    pub fn parts(parts: Vec<UserContentPart>) -> Self {
        Self::Parts(parts)
    }

    pub fn is_meaningful(&self) -> bool {
        match self {
            Self::Text(text) => !text.trim().is_empty(),
            Self::Parts(parts) => parts.iter().any(|part| match part {
                UserContentPart::Text { text } => !text.trim().is_empty(),
                UserContentPart::Image { .. } => true,
            }),
        }
    }

    pub(crate) fn has_images(&self) -> bool {
        matches!(
            self,
            Self::Parts(parts)
                if parts
                    .iter()
                    .any(|part| matches!(part, UserContentPart::Image { .. }))
        )
    }

    pub(crate) fn validate_prompt_images(&self) -> Result<(usize, usize)> {
        self.validate_images(
            MAX_PROMPT_IMAGE_BYTES,
            MAX_PROMPT_IMAGES,
            &mut HashSet::new(),
        )
    }

    pub(crate) fn validate_images(
        &self,
        max_bytes: usize,
        max_count: usize,
        validated: &mut HashSet<[u8; 32]>,
    ) -> Result<(usize, usize)> {
        let Self::Parts(parts) = self else {
            return Ok((0, 0));
        };
        let mut bytes = 0usize;
        let mut count = 0usize;
        for part in parts {
            let UserContentPart::Image { image } = part else {
                continue;
            };
            count = count.checked_add(1).context("image count overflowed")?;
            ensure!(count <= max_count, "image count exceeds its limit");
            let remaining_bytes = max_bytes.saturating_sub(bytes);
            bytes = bytes
                .checked_add(image.validate_persisted(remaining_bytes, validated)?)
                .context("image byte count overflowed")?;
            ensure!(bytes <= max_bytes, "image bytes exceed their limit");
        }
        Ok((bytes, count))
    }

    pub fn display_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .map(|part| match part {
                    UserContentPart::Text { text } => text.clone(),
                    UserContentPart::Image { image } => image.placeholder(),
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
        }
    }

    pub fn chat_content(&self) -> Value {
        match self {
            Self::Text(text) => Value::String(text.clone()),
            Self::Parts(parts) => Value::Array(
                parts
                    .iter()
                    .map(|part| match part {
                        UserContentPart::Text { text } => json!({
                            "type": "text",
                            "text": text,
                        }),
                        UserContentPart::Image { image } => json!({
                            "type": "image_url",
                            "image_url": {"url": image.data_url()},
                        }),
                    })
                    .collect(),
            ),
        }
    }

    pub fn responses_content(&self) -> Vec<Value> {
        match self {
            Self::Text(text) => vec![json!({"type": "input_text", "text": text})],
            Self::Parts(parts) => parts
                .iter()
                .map(|part| match part {
                    UserContentPart::Text { text } => {
                        json!({"type": "input_text", "text": text})
                    }
                    UserContentPart::Image { image } => json!({
                        "type": "input_image",
                        "image_url": image.data_url(),
                    }),
                })
                .collect(),
        }
    }
}

impl PartialEq<str> for UserContent {
    fn eq(&self, other: &str) -> bool {
        matches!(self, Self::Text(text) if text == other)
    }
}

impl PartialEq<&str> for UserContent {
    fn eq(&self, other: &&str) -> bool {
        self == *other
    }
}

impl std::fmt::Display for UserContent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.display_text())
    }
}

impl From<String> for UserContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for UserContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

/// A message in the model conversation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: UserContent,
    },
    /// Same-turn user-role context with a durable local identity.
    InjectedUser {
        injection_id: String,
        content: String,
    },
    /// Persisted compaction context sent to providers with the user role.
    Summary {
        content: String,
    },
    /// Persisted Phase 4 progress injected with the user role.
    LastWords {
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

    pub fn user(content: impl Into<UserContent>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    pub fn user_parts(parts: Vec<UserContentPart>) -> Self {
        Self::User {
            content: UserContent::parts(parts),
        }
    }

    pub fn injected_user(injection_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::InjectedUser {
            injection_id: injection_id.into(),
            content: content.into(),
        }
    }

    pub(crate) fn summary(content: impl Into<String>) -> Self {
        Self::Summary {
            content: content.into(),
        }
    }

    pub(crate) fn last_words(content: impl Into<String>) -> Self {
        Self::LastWords {
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
            Message::User { content } => {
                json!({"role": "user", "content": content.chat_content()})
            }
            Message::InjectedUser { content, .. }
            | Message::Summary { content }
            | Message::LastWords { content } => {
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

    /// Serialise for textual side-call transcripts without embedding image bytes.
    pub(crate) fn to_transcript_api(&self) -> Value {
        match self {
            Message::User { content } => {
                json!({"role": "user", "content": content.display_text()})
            }
            _ => self.to_api(),
        }
    }

    /// Short single-line rendering used by transcripts and tests.
    pub fn preview(&self) -> String {
        let text = match self {
            Message::User { content } => content.display_text(),
            Message::System { content }
            | Message::InjectedUser { content, .. }
            | Message::Summary { content }
            | Message::LastWords { content } => content.clone(),
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
    fn injected_user_round_trips_identity_without_leaking_it_to_chat_api() {
        let message = Message::injected_user("injection-1", "updated requirements");
        let stored = serde_json::to_value(&message).unwrap();

        assert_eq!(
            stored,
            json!({
                "role": "injected_user",
                "injection_id": "injection-1",
                "content": "updated requirements",
            })
        );
        assert_eq!(serde_json::from_value::<Message>(stored).unwrap(), message);
        assert_eq!(message.preview(), "updated requirements");
        assert_eq!(
            message.to_api(),
            json!({"role": "user", "content": "updated requirements"})
        );
    }

    #[test]
    fn text_user_keeps_legacy_storage_and_chat_shapes() {
        let message = Message::user("hello");
        assert_eq!(
            serde_json::to_value(&message).unwrap(),
            json!({"role": "user", "content": "hello"})
        );
        assert_eq!(
            message.to_api(),
            json!({"role": "user", "content": "hello"})
        );
    }

    #[test]
    fn image_count_budget_rejects_before_payload_validation() {
        let content = UserContent::parts(vec![UserContentPart::Image {
            image: ImageInput {
                media_type: "image/png".into(),
                data: "not-base64".into(),
                validated_key: None,
            },
        }]);
        let error = content
            .validate_images(MAX_IMAGE_BYTES, 0, &mut HashSet::new())
            .unwrap_err();
        assert!(error.to_string().contains("count exceeds"), "{error:#}");
    }

    #[test]
    fn multimodal_user_round_trips_and_maps_ordered_chat_parts() {
        let message = Message::user_parts(vec![
            UserContentPart::Text {
                text: "before".into(),
            },
            UserContentPart::Image {
                image: ImageInput {
                    media_type: "image/png".into(),
                    data: "cG5n".into(),
                    validated_key: None,
                },
            },
            UserContentPart::Text {
                text: "after".into(),
            },
        ]);
        let stored = serde_json::to_value(&message).unwrap();

        assert_eq!(serde_json::from_value::<Message>(stored).unwrap(), message);
        assert_eq!(
            message.to_api(),
            json!({
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64,cG5n"
                    }},
                    {"type": "text", "text": "after"},
                ]
            })
        );
        assert_eq!(message.preview(), "before");
        let transcript = message.to_transcript_api();
        assert_eq!(
            transcript["content"],
            "before\n\n[image: image/png]\n\nafter"
        );
        assert!(!transcript.to_string().contains("cG5n"));
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
            responses_continuation: None,
        });
        let value = message.to_api();
        assert_eq!(value["role"], "assistant");
        assert!(value["content"].is_null());
        assert_eq!(value["tool_calls"][0]["function"]["name"], "read_file");
    }
}
