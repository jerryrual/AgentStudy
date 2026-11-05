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
    context_window: Option<u32>,
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
            context_window: profile.context_window,
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

    /// Whether one completed model call crossed the persisted-fold threshold.
    pub(crate) fn should_force_compression(&self, usage: Usage) -> bool {
        if usage.prompt_tokens == 0 {
            return false;
        }
        let Some(budget) =
            self.context_window
                .zip(self.max_tokens)
                .and_then(|(context_window, max_tokens)| {
                    crate::context_budget::ContextBudget::derive(context_window, max_tokens)
                })
        else {
            return false;
        };
        usage.prompt_tokens > u64::from(budget.force_tokens)
    }

    /// Check whether messages leave a valid output allowance without sending them.
    pub(crate) fn validate_request_context(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<()> {
        self.request_body(messages, tools).map(|_| ())
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
        let body = self.request_body(messages, tools)?;
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

    /// Build the provider-specific request and apply its context output cap.
    fn request_body(&self, messages: &[Message], tools: &[ToolDef]) -> Result<Value> {
        let context_budget =
            self.context_window
                .zip(self.max_tokens)
                .and_then(|(context_window, max_tokens)| {
                    crate::context_budget::ContextBudget::derive(context_window, max_tokens)
                });
        let (prepared_messages, estimated_input) = match context_budget {
            Some(budget) => {
                let phase1_result = crate::context_trim::trim_previous_tool_results(
                    messages,
                    budget.trigger_tokens,
                    budget.target_tokens,
                    |candidate| self.estimated_request_tokens(candidate, tools),
                );
                let phase1_triggered = phase1_result.summarized > 0
                    || phase1_result.estimated_tokens >= budget.trigger_tokens;
                let phase2_result = if phase1_triggered {
                    crate::context_trim::remove_previous_tool_groups(
                        phase1_result,
                        budget.target_tokens,
                        |candidate| self.estimated_request_tokens(candidate, tools),
                    )
                } else {
                    phase1_result
                };
                let phase3_result =
                    if phase1_triggered && phase2_result.estimated_tokens > budget.target_tokens {
                        crate::context_trim::compress_previous_text_turns(
                            phase2_result,
                            messages,
                            budget.target_tokens,
                            |candidate| self.estimated_request_tokens(candidate, tools),
                        )
                    } else {
                        phase2_result
                    };
                (phase3_result.messages, phase3_result.estimated_tokens)
            }
            None => (
                std::borrow::Cow::Borrowed(messages),
                self.estimated_request_tokens(messages, tools),
            ),
        };
        let mut body = if self.api_style == ApiStyle::Responses {
            super::responses::request_body(
                prepared_messages.as_ref(),
                tools,
                &self.model,
                self.stream,
                self.temperature,
                self.max_tokens,
                self.include_encrypted_reasoning,
            )
        } else {
            let api_messages: Vec<Value> = prepared_messages.iter().map(Message::to_api).collect();
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
        };
        self.apply_context_budget(&mut body, estimated_input)?;
        Ok(body)
    }

    /// Conservatively estimate the semantic history and tools sent by this API style.
    fn estimated_request_tokens(&self, messages: &[Message], tools: &[ToolDef]) -> u32 {
        let semantic_messages: Vec<Value> = messages
            .iter()
            .map(|message| match message {
                Message::Assistant { message }
                    if self.api_style == ApiStyle::Responses
                        && !message.responses_items.is_empty() =>
                {
                    json!({
                        "role": "assistant",
                        "responses_items": message.responses_items,
                    })
                }
                _ => message.to_api(),
            })
            .collect();
        let semantic_tools: Vec<Value> = tools.iter().map(ToolDef::to_api).collect();
        crate::context_budget::estimate_json_tokens(&json!({
            "messages": semantic_messages,
            "tools": semantic_tools,
        }))
    }

    /// Clamp the request output limit or reject a request with no valid output room.
    fn apply_context_budget(&self, body: &mut Value, estimated_input: u32) -> Result<()> {
        let (Some(context_window), Some(configured_output)) =
            (self.context_window, self.max_tokens)
        else {
            return Ok(());
        };
        let output = crate::context_budget::clamp_output_tokens(
            context_window,
            configured_output,
            estimated_input,
        );
        if output == 0 {
            anyhow::bail!(
                "estimated model input ({estimated_input} tokens) exhausts the configured context window ({context_window} tokens); compact the conversation or increase context_window"
            );
        }
        let minimum_output = if self.api_style == ApiStyle::Responses {
            16
        } else {
            1
        };
        if output < minimum_output {
            anyhow::bail!(
                "estimated model input leaves room for {output} output tokens, below the {} API minimum of {minimum_output}; compact the conversation or increase context_window",
                self.api_style.as_str()
            );
        }
        let field = if self.api_style == ApiStyle::Responses {
            "max_output_tokens"
        } else {
            "max_tokens"
        };
        body[field] = json!(output);
        Ok(())
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

    fn budget_client(api_style: ApiStyle) -> ModelClient {
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_style = api_style;
        profile.api_key = "test-key".to_string();
        profile.base_url = "http://127.0.0.1:1".to_string();
        profile.context_window = Some(2_000);
        profile.max_tokens = Some(500);
        ModelClient::from_profile(&profile).unwrap()
    }

    #[test]
    fn forced_compression_uses_strict_prompt_threshold_and_known_usage() {
        let mut client = budget_client(ApiStyle::Chat);
        let force_tokens = crate::context_budget::ContextBudget::derive(2_000, 500)
            .unwrap()
            .force_tokens;
        let usage = |prompt_tokens| Usage {
            prompt_tokens,
            completion_tokens: 1,
            total_tokens: prompt_tokens.saturating_add(1),
        };

        assert!(!client.should_force_compression(usage(0)));
        assert!(!client.should_force_compression(usage(u64::from(force_tokens))));
        assert!(client.should_force_compression(usage(u64::from(force_tokens) + 1)));

        client.context_window = None;
        assert!(!client.should_force_compression(usage(u64::MAX)));
        client.context_window = Some(2_000);
        client.max_tokens = None;
        assert!(!client.should_force_compression(usage(u64::MAX)));
    }

    fn phase1_client(api_style: ApiStyle) -> ModelClient {
        let mut client = budget_client(api_style);
        client.context_window = Some(4_000);
        client
    }

    fn tool_call_message(id: &str, name: &str) -> Message {
        Message::assistant(AssistantMessage {
            tool_calls: vec![ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: "{}".to_string(),
            }],
            ..AssistantMessage::default()
        })
    }

    fn projection_messages() -> Vec<Message> {
        vec![
            Message::user("old turn"),
            tool_call_message("call_old", "old_tool"),
            Message::tool("call_old", "old_tool", "OLD_TOOL_RESULT".repeat(300)),
            Message::assistant(AssistantMessage::text("old turn complete")),
            Message::user("current turn"),
            tool_call_message("call_current", "current_tool"),
            Message::tool(
                "call_current",
                "current_tool",
                "CURRENT_TOOL_RESULT_MUST_STAY_EXACT",
            ),
        ]
    }

    fn phase3_client(api_style: ApiStyle) -> ModelClient {
        let mut client = budget_client(api_style);
        client.context_window = Some(4_000);
        client
    }

    fn phase3_messages(native_responses: bool) -> Vec<Message> {
        let old_text = "OLD_FULL_TEXT_MUST_DISAPPEAR".repeat(240);
        let old_responses_items = if native_responses {
            vec![json!({
                "type": "message",
                "id": "msg_old_native_must_disappear",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": old_text,
                    "annotations": [],
                }],
            })]
        } else {
            Vec::new()
        };
        let current_responses_items = if native_responses {
            vec![json!({
                "type": "function_call",
                "id": "fc_current_must_stay_exact",
                "call_id": "call_current_must_stay_exact",
                "name": "current_tool_must_stay_exact",
                "arguments": r#"{"value":"CURRENT_ARGUMENT_MUST_STAY_EXACT"}"#,
                "status": "completed",
            })]
        } else {
            Vec::new()
        };

        vec![
            Message::user("OLD_USER_FULL_TEXT_MUST_DISAPPEAR".repeat(40)),
            Message::assistant(AssistantMessage {
                content: Some(old_text),
                responses_items: old_responses_items,
                ..AssistantMessage::default()
            }),
            Message::user("CURRENT_USER_MUST_STAY_EXACT"),
            Message::assistant(AssistantMessage {
                tool_calls: vec![ToolCall {
                    id: "call_current_must_stay_exact".to_string(),
                    name: "current_tool_must_stay_exact".to_string(),
                    arguments: r#"{"value":"CURRENT_ARGUMENT_MUST_STAY_EXACT"}"#.to_string(),
                }],
                responses_items: current_responses_items,
                ..AssistantMessage::default()
            }),
            Message::tool(
                "call_current_must_stay_exact",
                "current_tool_must_stay_exact",
                "CURRENT_RESULT_MUST_STAY_EXACT",
            ),
        ]
    }

    fn phase3_estimates(client: &ModelClient, messages: &[Message]) -> (u32, u32, u32) {
        let budget = crate::context_budget::ContextBudget::derive(
            client.context_window.unwrap(),
            client.max_tokens.unwrap(),
        )
        .unwrap();
        let phase1 = crate::context_trim::trim_previous_tool_results(
            messages,
            budget.trigger_tokens,
            budget.target_tokens,
            |candidate| client.estimated_request_tokens(candidate, &[]),
        );
        let phase1_tokens = phase1.estimated_tokens;
        assert!(phase1_tokens > budget.target_tokens);
        let phase2 = crate::context_trim::remove_previous_tool_groups(
            phase1,
            budget.target_tokens,
            |candidate| client.estimated_request_tokens(candidate, &[]),
        );
        let phase2_tokens = phase2.estimated_tokens;
        assert_eq!(phase2_tokens, phase1_tokens);
        let phase3 = crate::context_trim::compress_previous_text_turns(
            phase2,
            messages,
            budget.target_tokens,
            |candidate| client.estimated_request_tokens(candidate, &[]),
        );
        assert!(phase3.estimated_tokens < phase2_tokens);
        assert_eq!(
            phase3.estimated_tokens,
            client.estimated_request_tokens(phase3.messages.as_ref(), &[])
        );
        (phase1_tokens, phase2_tokens, phase3.estimated_tokens)
    }

    fn assert_one_compressed_marker(wire: &[Value], content_pointer: &str) {
        let markers: Vec<&str> = wire
            .iter()
            .filter_map(|item| item.pointer(content_pointer).and_then(Value::as_str))
            .filter(|content| content.starts_with("[Compressed context]\nSummary: "))
            .collect();
        assert_eq!(markers.len(), 1, "{wire:?}");
    }

    fn phase2_client(api_style: ApiStyle) -> ModelClient {
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_style = api_style;
        profile.api_key = "test-key".to_string();
        profile.base_url = "http://127.0.0.1:1".to_string();
        profile.context_window = Some(3_000);
        profile.max_tokens = Some(2_200);
        ModelClient::from_profile(&profile).unwrap()
    }

    fn phase2_messages(native_responses: bool) -> Vec<Message> {
        let old_calls = vec![
            ToolCall {
                id: "call_old_a".to_string(),
                name: "old_tool_a".to_string(),
                arguments: r#"{"path":"old-a"}"#.to_string(),
            },
            ToolCall {
                id: "call_old_b".to_string(),
                name: "old_tool_b".to_string(),
                arguments: r#"{"path":"old-b"}"#.to_string(),
            },
        ];
        let old_native_items = if native_responses {
            vec![
                json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": "OLD_NARRATION_MUST_STAY",
                        "annotations": [],
                    }],
                }),
                json!({
                    "type": "function_call",
                    "id": "fc_call_old_a",
                    "call_id": "call_old_a",
                    "name": "old_tool_a",
                    "arguments": r#"{"path":"old-a"}"#,
                    "status": "completed",
                }),
                json!({
                    "type": "function_call",
                    "id": "fc_call_old_b",
                    "call_id": "call_old_b",
                    "name": "old_tool_b",
                    "arguments": r#"{"path":"old-b"}"#,
                    "status": "completed",
                }),
            ]
        } else {
            Vec::new()
        };
        let current_native_items = if native_responses {
            vec![json!({
                "type": "function_call",
                "id": "fc_call_current",
                "call_id": "call_current",
                "name": "current_tool",
                "arguments": "{}",
                "status": "completed",
            })]
        } else {
            Vec::new()
        };

        vec![
            Message::user("old turn"),
            Message::assistant(AssistantMessage {
                content: Some("OLD_NARRATION_MUST_STAY".to_string()),
                tool_calls: old_calls,
                responses_items: old_native_items,
                ..AssistantMessage::default()
            }),
            Message::tool("call_old_a", "old_tool_a", "OLD_RESULT_A".repeat(300)),
            Message::tool("call_old_b", "old_tool_b", "OLD_RESULT_B".repeat(300)),
            Message::user("current turn"),
            Message::assistant(AssistantMessage {
                tool_calls: vec![ToolCall {
                    id: "call_current".to_string(),
                    name: "current_tool".to_string(),
                    arguments: "{}".to_string(),
                }],
                responses_items: current_native_items,
                ..AssistantMessage::default()
            }),
            Message::tool(
                "call_current",
                "current_tool",
                "CURRENT_RESULT_MUST_STAY_EXACT",
            ),
        ]
    }

    fn phase2_estimates(client: &ModelClient, messages: &[Message]) -> (u32, u32) {
        let budget = crate::context_budget::ContextBudget::derive(
            client.context_window.unwrap(),
            client.max_tokens.unwrap(),
        )
        .unwrap();
        let phase1 = crate::context_trim::trim_previous_tool_results(
            messages,
            budget.trigger_tokens,
            budget.target_tokens,
            |candidate| client.estimated_request_tokens(candidate, &[]),
        );
        let phase1_tokens = phase1.estimated_tokens;
        assert!(phase1_tokens > budget.target_tokens);
        let phase2 = crate::context_trim::remove_previous_tool_groups(
            phase1,
            budget.target_tokens,
            |candidate| client.estimated_request_tokens(candidate, &[]),
        );
        assert!(phase2.estimated_tokens < phase1_tokens);
        assert_eq!(
            phase2.estimated_tokens,
            client.estimated_request_tokens(phase2.messages.as_ref(), &[])
        );
        (phase1_tokens, phase2.estimated_tokens)
    }

    #[test]
    fn request_phase3_compresses_old_chat_text_and_uses_final_projection() {
        let client = phase3_client(ApiStyle::Chat);
        let messages = phase3_messages(false);
        let original = messages.clone();
        let (phase1_tokens, phase2_tokens, final_tokens) = phase3_estimates(&client, &messages);

        let first = client.request_body(&messages, &[]).unwrap();
        let second = client.request_body(&messages, &[]).unwrap();

        assert_eq!(first, second);
        assert_eq!(messages, original);
        assert_eq!(phase1_tokens, phase2_tokens);
        let wire = first["messages"].as_array().unwrap();
        assert_one_compressed_marker(wire, "/content");
        let serialized = serde_json::to_string(wire).unwrap();
        assert!(!serialized.contains(&"OLD_FULL_TEXT_MUST_DISAPPEAR".repeat(240)));
        assert_eq!(
            &wire[wire.len() - 3..],
            &[
                json!({"role": "user", "content": "CURRENT_USER_MUST_STAY_EXACT"}),
                json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_current_must_stay_exact",
                        "type": "function",
                        "function": {
                            "name": "current_tool_must_stay_exact",
                            "arguments": r#"{"value":"CURRENT_ARGUMENT_MUST_STAY_EXACT"}"#,
                        },
                    }],
                }),
                json!({
                    "role": "tool",
                    "tool_call_id": "call_current_must_stay_exact",
                    "content": "CURRENT_RESULT_MUST_STAY_EXACT",
                }),
            ]
        );

        let expected = crate::context_budget::clamp_output_tokens(4_000, 500, final_tokens);
        let phase2_cap = crate::context_budget::clamp_output_tokens(4_000, 500, phase2_tokens);
        assert_ne!(expected, phase2_cap);
        assert_eq!(first["max_tokens"], expected);
    }

    #[test]
    fn request_phase3_compresses_old_native_text_and_uses_final_projection() {
        let client = phase3_client(ApiStyle::Responses);
        let messages = phase3_messages(true);
        let original = messages.clone();
        let (phase1_tokens, phase2_tokens, final_tokens) = phase3_estimates(&client, &messages);

        let first = client.request_body(&messages, &[]).unwrap();
        let second = client.request_body(&messages, &[]).unwrap();

        assert_eq!(first, second);
        assert_eq!(messages, original);
        assert_eq!(phase1_tokens, phase2_tokens);
        let wire = first["input"].as_array().unwrap();
        assert_one_compressed_marker(wire, "/content/0/text");
        let summary = wire
            .iter()
            .find(|item| {
                item.pointer("/content/0/text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| text.starts_with("[Compressed context]"))
            })
            .unwrap();
        assert_eq!(summary["role"], "user");
        assert_eq!(summary["content"][0]["type"], "input_text");
        assert!(summary.get("id").is_none());
        assert!(summary.get("status").is_none());
        let serialized = serde_json::to_string(wire).unwrap();
        assert!(!serialized.contains(&"OLD_FULL_TEXT_MUST_DISAPPEAR".repeat(240)));
        assert!(!serialized.contains("msg_old_native_must_disappear"));
        assert_eq!(
            &wire[wire.len() - 3..],
            &[
                json!({
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": "CURRENT_USER_MUST_STAY_EXACT",
                    }],
                }),
                json!({
                    "type": "function_call",
                    "id": "fc_current_must_stay_exact",
                    "call_id": "call_current_must_stay_exact",
                    "name": "current_tool_must_stay_exact",
                    "arguments": r#"{"value":"CURRENT_ARGUMENT_MUST_STAY_EXACT"}"#,
                    "status": "completed",
                }),
                json!({
                    "type": "function_call_output",
                    "call_id": "call_current_must_stay_exact",
                    "output": "CURRENT_RESULT_MUST_STAY_EXACT",
                }),
            ]
        );

        let expected = crate::context_budget::clamp_output_tokens(4_000, 500, final_tokens);
        let phase2_cap = crate::context_budget::clamp_output_tokens(4_000, 500, phase2_tokens);
        assert_ne!(expected, phase2_cap);
        assert_eq!(first["max_output_tokens"], expected);
    }

    #[test]
    fn request_phase3_does_not_run_below_phase1_trigger() {
        let client = phase3_client(ApiStyle::Chat);
        let messages = vec![
            Message::user("old request ".repeat(10)),
            Message::assistant(AssistantMessage::text("old completed text ".repeat(20))),
            Message::user("CURRENT_USER_MUST_STAY_EXACT"),
        ];
        let budget = crate::context_budget::ContextBudget::derive(4_000, 500).unwrap();
        let estimated = client.estimated_request_tokens(&messages, &[]);
        assert!(estimated < budget.trigger_tokens, "{estimated} >= trigger");

        let body = client.request_body(&messages, &[]).unwrap();
        let expected: Vec<Value> = messages.iter().map(Message::to_api).collect();

        assert_eq!(body["messages"], Value::Array(expected));
        assert!(body["messages"].as_array().unwrap().iter().all(|item| {
            item["content"]
                .as_str()
                .is_none_or(|content| !content.starts_with("[Compressed context]"))
        }));
    }

    #[test]
    fn request_phase2_removes_old_chat_batch_and_uses_final_projection() {
        let client = phase2_client(ApiStyle::Chat);
        let messages = phase2_messages(false);
        let original = messages.clone();
        let (phase1_tokens, final_tokens) = phase2_estimates(&client, &messages);

        let body = client.request_body(&messages, &[]).unwrap();

        assert_eq!(messages, original);
        let wire = body["messages"].as_array().unwrap();
        assert!(wire.iter().all(|item| {
            item["tool_call_id"] != "call_old_a"
                && item["tool_call_id"] != "call_old_b"
                && item["tool_calls"].as_array().is_none_or(|calls| {
                    calls
                        .iter()
                        .all(|call| !call["id"].as_str().unwrap().starts_with("call_old"))
                })
        }));
        assert!(wire.iter().any(
            |item| item["role"] == "assistant" && item["content"] == "OLD_NARRATION_MUST_STAY"
        ));
        assert!(wire.iter().any(|item| {
            item["tool_calls"]
                .as_array()
                .is_some_and(|calls| calls.iter().any(|call| call["id"] == "call_current"))
        }));
        assert!(wire.iter().any(|item| {
            item["tool_call_id"] == "call_current"
                && item["content"] == "CURRENT_RESULT_MUST_STAY_EXACT"
        }));

        let expected = crate::context_budget::clamp_output_tokens(
            client.context_window.unwrap(),
            client.max_tokens.unwrap(),
            final_tokens,
        );
        let phase1_cap = crate::context_budget::clamp_output_tokens(
            client.context_window.unwrap(),
            client.max_tokens.unwrap(),
            phase1_tokens,
        );
        assert_ne!(expected, phase1_cap);
        assert_eq!(body["max_tokens"], expected);
    }

    #[test]
    fn request_phase2_removes_old_native_responses_batch_and_uses_final_projection() {
        let client = phase2_client(ApiStyle::Responses);
        let messages = phase2_messages(true);
        let original = messages.clone();
        let (phase1_tokens, final_tokens) = phase2_estimates(&client, &messages);

        let body = client.request_body(&messages, &[]).unwrap();

        assert_eq!(messages, original);
        let wire = body["input"].as_array().unwrap();
        assert!(wire
            .iter()
            .all(|item| { item["call_id"] != "call_old_a" && item["call_id"] != "call_old_b" }));
        assert!(wire.iter().any(|item| {
            item["type"] == "message"
                && item["role"] == "assistant"
                && item["content"][0]["text"] == "OLD_NARRATION_MUST_STAY"
        }));
        assert!(wire
            .iter()
            .any(|item| { item["type"] == "function_call" && item["call_id"] == "call_current" }));
        assert!(wire.iter().any(|item| {
            item["type"] == "function_call_output"
                && item["call_id"] == "call_current"
                && item["output"] == "CURRENT_RESULT_MUST_STAY_EXACT"
        }));

        let expected = crate::context_budget::clamp_output_tokens(
            client.context_window.unwrap(),
            client.max_tokens.unwrap(),
            final_tokens,
        );
        let phase1_cap = crate::context_budget::clamp_output_tokens(
            client.context_window.unwrap(),
            client.max_tokens.unwrap(),
            phase1_tokens,
        );
        assert_ne!(expected, phase1_cap);
        assert_eq!(body["max_output_tokens"], expected);
    }

    #[test]
    fn request_phase2_does_not_run_below_the_trigger() {
        let client = budget_client(ApiStyle::Chat);
        let messages = vec![
            Message::user("old turn"),
            tool_call_message("call_old", "old_tool"),
            Message::tool("call_old", "old_tool", "x".repeat(200)),
            Message::user("current turn"),
        ];
        let budget = crate::context_budget::ContextBudget::derive(2_000, 500).unwrap();
        let estimated = client.estimated_request_tokens(&messages, &[]);
        assert!(estimated > budget.target_tokens, "{estimated} <= target");
        assert!(estimated < budget.trigger_tokens, "{estimated} >= trigger");

        let body = client.request_body(&messages, &[]).unwrap();
        let wire = body["messages"].as_array().unwrap();

        assert!(wire.iter().any(|item| {
            item["tool_calls"]
                .as_array()
                .is_some_and(|calls| calls.iter().any(|call| call["id"] == "call_old"))
        }));
        assert!(wire.iter().any(|item| {
            item["tool_call_id"] == "call_old" && item["content"] == "x".repeat(200)
        }));
    }

    #[test]
    fn request_projection_trims_old_tools_for_chat_without_mutating_history() {
        let client = phase1_client(ApiStyle::Chat);
        let messages = projection_messages();
        let original = messages.clone();

        let first = client.request_body(&messages, &[]).unwrap();
        let second = client.request_body(&messages, &[]).unwrap();

        assert_eq!(first, second);
        assert_eq!(messages, original);
        let wire = first["messages"].as_array().unwrap();
        let old_result = wire
            .iter()
            .find(|item| item["tool_call_id"] == "call_old")
            .unwrap();
        assert!(old_result["content"]
            .as_str()
            .unwrap()
            .starts_with("[Tool call: "));
        let current_result = wire
            .iter()
            .find(|item| item["tool_call_id"] == "call_current")
            .unwrap();
        assert_eq!(
            current_result["content"],
            "CURRENT_TOOL_RESULT_MUST_STAY_EXACT"
        );

        let call_ids: Vec<&str> = wire
            .iter()
            .filter_map(|item| item.get("tool_calls"))
            .flat_map(|calls| calls.as_array().unwrap())
            .filter_map(|call| call["id"].as_str())
            .collect();
        let result_ids: Vec<&str> = wire
            .iter()
            .filter_map(|item| item["tool_call_id"].as_str())
            .collect();
        assert_eq!(call_ids, result_ids);
    }

    #[test]
    fn request_projection_trims_old_tools_for_responses_and_keeps_pairs() {
        let client = phase1_client(ApiStyle::Responses);
        let messages = projection_messages();
        let original = messages.clone();

        let first = client.request_body(&messages, &[]).unwrap();
        let second = client.request_body(&messages, &[]).unwrap();

        assert_eq!(first, second);
        assert_eq!(messages, original);
        let wire = first["input"].as_array().unwrap();
        let old_result = wire
            .iter()
            .find(|item| item["type"] == "function_call_output" && item["call_id"] == "call_old")
            .unwrap();
        assert!(old_result["output"]
            .as_str()
            .unwrap()
            .starts_with("[Tool call: "));
        let current_result = wire
            .iter()
            .find(|item| {
                item["type"] == "function_call_output" && item["call_id"] == "call_current"
            })
            .unwrap();
        assert_eq!(
            current_result["output"],
            "CURRENT_TOOL_RESULT_MUST_STAY_EXACT"
        );

        let call_ids: Vec<&str> = wire
            .iter()
            .filter(|item| item["type"] == "function_call")
            .filter_map(|item| item["call_id"].as_str())
            .collect();
        let result_ids: Vec<&str> = wire
            .iter()
            .filter(|item| item["type"] == "function_call_output")
            .filter_map(|item| item["call_id"].as_str())
            .collect();
        assert_eq!(call_ids, result_ids);
    }

    #[test]
    fn responses_projection_preserves_native_function_call_items() {
        let client = phase1_client(ApiStyle::Responses);
        let messages = vec![
            Message::user("old turn"),
            Message::assistant(AssistantMessage {
                tool_calls: vec![ToolCall {
                    id: "call_old".to_string(),
                    name: "old_tool".to_string(),
                    arguments: r#"{"path":"large.txt"}"#.to_string(),
                }],
                responses_items: vec![json!({
                    "type": "function_call",
                    "call_id": "call_old",
                    "name": "old_tool",
                    "arguments": r#"{"path":"large.txt"}"#,
                })],
                ..AssistantMessage::default()
            }),
            Message::tool("call_old", "old_tool", "OLD_TOOL_RESULT".repeat(300)),
            Message::user("current turn"),
        ];

        let body = client.request_body(&messages, &[]).unwrap();
        let wire = body["input"].as_array().unwrap();
        assert!(wire
            .iter()
            .any(|item| { item["type"] == "function_call" && item["call_id"] == "call_old" }));
        let output = wire
            .iter()
            .find(|item| item["type"] == "function_call_output" && item["call_id"] == "call_old")
            .unwrap();
        assert!(output["output"]
            .as_str()
            .unwrap()
            .starts_with("[Tool call: old_tool(path=\"large.txt\") → "));
    }

    #[test]
    fn request_projection_is_disabled_without_context_window() {
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_key = "test-key".to_string();
        profile.base_url = "http://127.0.0.1:1".to_string();
        profile.context_window = None;
        profile.max_tokens = Some(500);
        let client = ModelClient::from_profile(&profile).unwrap();
        let messages = projection_messages();

        let body = client.request_body(&messages, &[]).unwrap();
        let old_result = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["tool_call_id"] == "call_old")
            .unwrap();
        assert_eq!(old_result["content"], "OLD_TOOL_RESULT".repeat(300));
    }

    #[test]
    fn request_budget_counts_responses_native_history_once() {
        let client = budget_client(ApiStyle::Responses);
        let native_item = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "native"}],
        });
        let messages = vec![Message::assistant(AssistantMessage {
            content: Some("duplicate chat text".repeat(100)),
            responses_items: vec![native_item.clone()],
            ..AssistantMessage::default()
        })];
        let expected = json!({
            "messages": [{
                "role": "assistant",
                "responses_items": [native_item],
            }],
            "tools": [],
        });

        assert_eq!(
            client.estimated_request_tokens(&messages, &[]),
            crate::context_budget::estimate_json_tokens(&expected)
        );
    }

    #[test]
    fn request_budget_uses_chat_history_after_model_switch() {
        let client = budget_client(ApiStyle::Chat);
        let messages = vec![Message::assistant(AssistantMessage {
            content: Some("actual chat content".repeat(100)),
            responses_items: vec![json!({"type": "message", "content": "small"})],
            ..AssistantMessage::default()
        })];
        let expected = json!({
            "messages": [messages[0].to_api()],
            "tools": [],
        });

        assert_eq!(
            client.estimated_request_tokens(&messages, &[]),
            crate::context_budget::estimate_json_tokens(&expected)
        );
    }

    #[test]
    fn request_budget_clamps_chat_and_responses_output() {
        let messages = vec![Message::user("x".repeat(1_600))];
        let mut caps = Vec::new();
        for (style, field) in [
            (ApiStyle::Chat, "max_tokens"),
            (ApiStyle::Responses, "max_output_tokens"),
        ] {
            let body = budget_client(style).request_body(&messages, &[]).unwrap();
            let cap = body[field].as_u64().unwrap();
            assert!((16..500).contains(&cap), "{style:?}: {body}");
            caps.push(cap);
        }
        assert_eq!(caps[0], caps[1]);
    }

    #[test]
    fn request_budget_counts_tool_definitions() {
        let client = budget_client(ApiStyle::Chat);
        let messages = vec![Message::user("hello")];
        let uncapped = client.request_body(&messages, &[]).unwrap();
        let tool = ToolDef {
            name: "large_tool".to_string(),
            description: "x".repeat(1_600),
            parameters: json!({"type": "object"}),
        };
        let capped = client.request_body(&messages, &[tool]).unwrap();

        assert_eq!(uncapped["max_tokens"], 500);
        assert!(capped["max_tokens"].as_u64().unwrap() < 500);
    }

    #[test]
    fn request_budget_enforces_provider_output_minimums() {
        let responses = budget_client(ApiStyle::Responses);
        let mut body = json!({"max_output_tokens": 500});
        responses.apply_context_budget(&mut body, 1_984).unwrap();
        assert_eq!(body["max_output_tokens"], 16);

        for estimated_input in [1_985, 1_999] {
            let mut body = json!({"max_output_tokens": 500});
            let error = responses
                .apply_context_budget(&mut body, estimated_input)
                .unwrap_err();
            assert!(error.to_string().contains("minimum of 16"));
        }

        let chat = budget_client(ApiStyle::Chat);
        let mut body = json!({"max_tokens": 500});
        chat.apply_context_budget(&mut body, 1_999).unwrap();
        assert_eq!(body["max_tokens"], 1);
    }

    #[test]
    fn request_budget_rejects_exhausted_context_before_http() {
        let messages = vec![Message::user("x".repeat(4_000))];
        let error = budget_client(ApiStyle::Chat)
            .request_body(&messages, &[])
            .unwrap_err();
        assert!(error.to_string().contains("exhausts"));
    }

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
