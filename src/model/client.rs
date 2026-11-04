//! OpenAI-compatible chat-completions client with SSE streaming and retries.

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use reqwest::Client as HttpClient;
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::{json, Value};

use super::sse::SseParser;
use super::types::{AssistantMessage, Completion, Message, ToolCall, ToolDef, Usage};
use crate::agent::EventSink;
use crate::config::{resolve_api_key, ApiStyle, ModelProfile};

const MAX_ATTEMPTS: usize = 3;
const MAX_ERROR_BODY: usize = 600;

/// A configured model endpoint.
pub struct ModelClient {
    http: HttpClient,
    api_style: ApiStyle,
    include_encrypted_reasoning: bool,
    endpoint: String,
    api_key: String,
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    stream: bool,
    retry_base: Duration,
}

impl ModelClient {
    /// Build a client for a profile, resolving the API key.
    pub fn from_profile(profile: &ModelProfile) -> Result<Self> {
        let api_key = resolve_api_key(&profile.api_key)
            .with_context(|| format!("cannot use model profile {:?}", profile.name))?;
        let http = HttpClient::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(600))
            .build()
            .context("failed to create HTTP client")?;
        let retry_base = std::env::var("ICODE_RETRY_BASE_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map(Duration::from_millis)
            .unwrap_or(Duration::from_millis(1000));
        let endpoint = match profile.api_style {
            ApiStyle::Chat => profile.chat_completions_url(),
            ApiStyle::Responses => profile.responses_url(),
        };
        Ok(Self {
            http,
            api_style: profile.api_style,
            include_encrypted_reasoning: profile.api_style == ApiStyle::Responses
                && profile.provider.eq_ignore_ascii_case("openai"),
            endpoint,
            api_key,
            model: profile.model.clone(),
            temperature: profile.temperature,
            max_tokens: profile.max_tokens,
            stream: profile.stream,
            retry_base,
        })
    }

    /// Model identifier, shown in the status bar.
    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Run one completion, streaming deltas through `sink`.
    ///
    /// Transient failures (HTTP 429/5xx and transport errors) are retried up to
    /// three times with exponential backoff. Failures after a partial stream are
    /// not retried so that text is never duplicated.
    pub async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        sink: &dyn EventSink,
    ) -> Result<Completion> {
        let body = self.request_body(messages, tools);
        let mut last_error: Option<anyhow::Error> = None;
        for attempt in 0..MAX_ATTEMPTS {
            match self.send_once(&body, sink).await {
                Ok(completion) => return Ok(completion),
                Err(ClientError::Fatal(error)) => return Err(error),
                Err(ClientError::Retryable(error)) => {
                    last_error = Some(error);
                    if attempt + 1 < MAX_ATTEMPTS {
                        let delay = self.retry_base * 2u32.pow(attempt as u32);
                        sink.info(&format!(
                            "model request failed ({}); retrying in {:.1}s",
                            last_error
                                .as_ref()
                                .map(|error| error.to_string())
                                .unwrap_or_default(),
                            delay.as_secs_f64()
                        ));
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow!("model request failed")))
    }

    fn request_body(&self, messages: &[Message], tools: &[ToolDef]) -> Value {
        if self.api_style == ApiStyle::Responses {
            return super::responses::request_body(
                messages,
                tools,
                &self.model,
                self.stream,
                self.temperature,
                self.max_tokens,
                self.include_encrypted_reasoning,
            );
        }
        let api_messages: Vec<Value> = messages.iter().map(Message::to_api).collect();
        let mut body = json!({
            "model": self.model,
            "messages": api_messages,
            "stream": self.stream,
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.iter().map(ToolDef::to_api).collect());
            body["tool_choice"] = json!("auto");
        }
        if self.stream {
            body["stream_options"] = json!({"include_usage": true});
        }
        if let Some(temperature) = self.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(max_tokens) = self.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        body
    }

    async fn send_once(
        &self,
        body: &Value,
        sink: &dyn EventSink,
    ) -> Result<Completion, ClientError> {
        let response = self
            .http
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|error| ClientError::Retryable(anyhow!("request to model failed: {error}")))?;

        let status = response.status();
        if !status.is_success() {
            let text = response
                .text()
                .await
                .unwrap_or_else(|error| format!("<failed to read error body: {error}>"));
            let body = truncate(&text, MAX_ERROR_BODY);
            let message = format!(
                "model request failed with HTTP {}: {}",
                status.as_u16(),
                body.trim()
            );
            if is_retryable_status(status) {
                return Err(ClientError::Retryable(anyhow!(message)));
            }
            return Err(ClientError::Fatal(anyhow!(message)));
        }

        if self.stream {
            self.consume_stream(response, sink).await
        } else {
            self.consume_json(response).await
        }
    }

    async fn consume_stream(
        &self,
        response: reqwest::Response,
        sink: &dyn EventSink,
    ) -> Result<Completion, ClientError> {
        if self.api_style == ApiStyle::Responses {
            return self.consume_responses_stream(response, sink).await;
        }
        let mut stream = response.bytes_stream();
        let mut parser = SseParser::new();
        let mut accumulator = Accumulator::default();
        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|error| {
                ClientError::Fatal(anyhow!("streaming response failed: {error}"))
            })?;
            for payload in parser
                .push(&bytes)
                .map_err(|error| ClientError::Fatal(anyhow!("{error}")))?
            {
                if accumulator.apply(&payload, sink)? {
                    return Ok(accumulator.finish());
                }
            }
        }
        for payload in parser
            .finish()
            .map_err(|error| ClientError::Fatal(anyhow!("{error}")))?
        {
            if accumulator.apply(&payload, sink)? {
                break;
            }
        }
        Ok(accumulator.finish())
    }

    async fn consume_responses_stream(
        &self,
        response: reqwest::Response,
        sink: &dyn EventSink,
    ) -> Result<Completion, ClientError> {
        let mut stream = response.bytes_stream();
        let mut parser = SseParser::new();
        let mut accumulator = super::responses::ResponsesAccumulator::default();
        while let Some(chunk) = stream.next().await {
            let bytes = chunk
                .map_err(|error| ClientError::Fatal(anyhow!("Responses stream failed: {error}")))?;
            for payload in parser
                .push(&bytes)
                .map_err(|error| ClientError::Fatal(anyhow!("{error}")))?
            {
                let trimmed = payload.trim();
                if !trimmed.is_empty()
                    && trimmed != "[DONE]"
                    && accumulator
                        .apply(&payload, sink)
                        .map_err(ClientError::Fatal)?
                {
                    return accumulator.finish().map_err(ClientError::Fatal);
                }
            }
        }
        for payload in parser
            .finish()
            .map_err(|error| ClientError::Fatal(anyhow!("{error}")))?
        {
            let trimmed = payload.trim();
            if !trimmed.is_empty() && trimmed != "[DONE]" {
                accumulator
                    .apply(&payload, sink)
                    .map_err(ClientError::Fatal)?;
            }
        }
        accumulator.finish().map_err(ClientError::Fatal)
    }

    async fn consume_json(&self, response: reqwest::Response) -> Result<Completion, ClientError> {
        if self.api_style == ApiStyle::Responses {
            let payload: Value = response.json().await.map_err(|error| {
                ClientError::Fatal(anyhow!("invalid Responses API response: {error}"))
            })?;
            return super::responses::decode_json(payload).map_err(ClientError::Fatal);
        }
        let payload: ChatResponse = response
            .json()
            .await
            .map_err(|error| ClientError::Fatal(anyhow!("invalid model response: {error}")))?;
        let choice = payload
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ClientError::Fatal(anyhow!("model returned no choices")))?;
        let message = AssistantMessage {
            content: choice.message.content.filter(|text| !text.is_empty()),
            reasoning: choice
                .message
                .reasoning_content
                .filter(|text| !text.is_empty()),
            tool_calls: choice
                .message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(|call| ToolCall {
                    id: call.id,
                    name: call.function.name,
                    arguments: call.function.arguments.unwrap_or_default(),
                })
                .collect(),
            responses_items: Vec::new(),
        };
        Ok(Completion {
            message,
            usage: payload.usage.unwrap_or_default(),
            finish_reason: choice.finish_reason,
        })
    }
}

fn is_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn truncate(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_string();
    }
    let mut end = limit;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

#[derive(Debug)]
enum ClientError {
    Retryable(anyhow::Error),
    Fatal(anyhow::Error),
}

#[derive(Default)]
struct Accumulator {
    text: String,
    reasoning: String,
    calls: BTreeMap<usize, ToolCall>,
    finish_reason: Option<String>,
    usage: Usage,
}

impl Accumulator {
    /// Apply one SSE payload. Returns `true` when the stream is finished.
    fn apply(&mut self, payload: &str, sink: &dyn EventSink) -> Result<bool, ClientError> {
        let payload = payload.trim();
        if payload == "[DONE]" {
            return Ok(true);
        }
        if payload.is_empty() {
            return Ok(false);
        }
        let chunk: Chunk = serde_json::from_str(payload).map_err(|error| {
            ClientError::Fatal(anyhow!(
                "invalid streaming event: {error}: {}",
                truncate(payload, MAX_ERROR_BODY)
            ))
        })?;
        if let Some(usage) = chunk.usage {
            self.usage = Usage {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: if usage.total_tokens == 0 {
                    usage.prompt_tokens + usage.completion_tokens
                } else {
                    usage.total_tokens
                },
            };
        }
        for choice in chunk.choices {
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }
            let Some(delta) = choice.delta else {
                continue;
            };
            if let Some(content) = delta.content {
                if !content.is_empty() {
                    self.text.push_str(&content);
                    sink.text_delta(&content);
                }
            }
            if let Some(reasoning) = delta.reasoning_content {
                if !reasoning.is_empty() {
                    self.reasoning.push_str(&reasoning);
                    sink.reasoning_delta(&reasoning);
                }
            }
            for delta_call in delta.tool_calls.unwrap_or_default() {
                let entry = self.calls.entry(delta_call.index).or_default();
                if let Some(id) = delta_call.id {
                    if entry.id.is_empty() {
                        entry.id = id;
                    }
                }
                if let Some(function) = delta_call.function {
                    if let Some(name) = function.name {
                        if entry.name.is_empty() {
                            entry.name = name;
                        }
                    }
                    if let Some(arguments) = function.arguments {
                        entry.arguments.push_str(&arguments);
                    }
                }
            }
        }
        Ok(false)
    }

    fn finish(self) -> Completion {
        let tool_calls: Vec<ToolCall> = self
            .calls
            .into_values()
            .filter(|call| !call.name.is_empty())
            .collect();
        Completion {
            message: AssistantMessage {
                content: if self.text.is_empty() {
                    None
                } else {
                    Some(self.text)
                },
                reasoning: if self.reasoning.is_empty() {
                    None
                } else {
                    Some(self.reasoning)
                },
                tool_calls,
                responses_items: Vec::new(),
            },
            usage: self.usage,
            finish_reason: self.finish_reason,
        }
    }
}

#[derive(Deserialize)]
struct Chunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: Option<Delta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionDelta>,
}

#[derive(Deserialize)]
struct FunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<ResponseChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct ResponseChoice {
    message: ResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Deserialize)]
struct ResponseToolCall {
    #[serde(default)]
    id: String,
    function: ResponseFunction,
}

#[derive(Deserialize)]
struct ResponseFunction {
    name: String,
    #[serde(default)]
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NullSink;
    impl EventSink for NullSink {}

    #[test]
    fn accumulator_merges_streamed_tool_calls() {
        let mut accumulator = Accumulator::default();
        let sink = NullSink;
        accumulator
            .apply(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"bash","arguments":"{\"comm"}}]}}]}"#,
                &sink,
            )
            .unwrap();
        accumulator
            .apply(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"and\":\"ls\"}"}}]},"finish_reason":"tool_calls"}]}"#,
                &sink,
            )
            .unwrap();
        let completion = accumulator.finish();
        assert_eq!(completion.message.tool_calls.len(), 1);
        assert_eq!(completion.message.tool_calls[0].name, "bash");
        assert_eq!(
            completion.message.tool_calls[0].arguments,
            "{\"command\":\"ls\"}"
        );
    }

    #[test]
    fn accumulator_collects_text_and_usage() {
        let mut accumulator = Accumulator::default();
        let sink = NullSink;
        accumulator
            .apply(r#"{"choices":[{"delta":{"content":"hello "}}]}"#, &sink)
            .unwrap();
        accumulator
            .apply(
                r#"{"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}"#,
                &sink,
            )
            .unwrap();
        let completion = accumulator.finish();
        assert_eq!(completion.message.content.as_deref(), Some("hello world"));
        assert_eq!(completion.usage.total_tokens, 12);
        assert_eq!(completion.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn done_marker_is_recognised() {
        let mut accumulator = Accumulator::default();
        let sink = NullSink;
        assert!(accumulator.apply("[DONE]", &sink).unwrap());
    }
}
