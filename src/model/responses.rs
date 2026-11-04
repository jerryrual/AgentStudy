//! Stateless OpenAI Responses API wire mapping.

use std::collections::BTreeMap;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};

use super::{AssistantMessage, Completion, Message, ToolCall, ToolDef, Usage};
use crate::agent::EventSink;

pub(super) fn request_body(
    messages: &[Message],
    tools: &[ToolDef],
    model: &str,
    stream: bool,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    include_encrypted_reasoning: bool,
) -> Value {
    let mut input = Vec::new();
    for message in messages {
        match message {
            Message::System { content } | Message::User { content } => input.push(json!({
                "type": "message",
                "role": if matches!(message, Message::System { .. }) { "system" } else { "user" },
                "content": [{"type": "input_text", "text": content}],
            })),
            Message::Assistant { message } => {
                if !message.responses_items.is_empty() {
                    input.extend(message.responses_items.iter().cloned());
                    continue;
                }
                if let Some(content) = &message.content {
                    input.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content, "annotations": []}],
                    }));
                }
                for call in &message.tool_calls {
                    input.push(json!({
                        "type": "function_call",
                        "id": format!("fc_{}", call.id),
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                        "status": "completed",
                    }));
                }
            }
            Message::Tool {
                call_id, content, ..
            } => input.push(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": content,
            })),
        }
    }
    let mut body = json!({
        "model": model,
        "input": input,
        "stream": stream,
        "store": false,
    });
    if !tools.is_empty() {
        body["tools"] = Value::Array(
            tools
                .iter()
                .map(|tool| {
                    let mut parameters = tool.parameters.clone();
                    if let Some(object) = parameters.as_object_mut() {
                        object.insert("additionalProperties".to_string(), Value::Bool(false));
                    }
                    json!({
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": parameters,
                        "strict": false,
                    })
                })
                .collect(),
        );
        body["tool_choice"] = json!("auto");
        body["parallel_tool_calls"] = json!(true);
    }
    if include_encrypted_reasoning {
        body["include"] = json!(["reasoning.encrypted_content"]);
    }
    if let Some(temperature) = temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(max_tokens) = max_tokens {
        body["max_output_tokens"] = json!(max_tokens);
    }
    body
}

#[derive(Default)]
pub(super) struct ResponsesAccumulator {
    text: String,
    reasoning: String,
    calls: BTreeMap<usize, ToolCall>,
    items: BTreeMap<usize, Value>,
    usage: Usage,
    finish_reason: Option<String>,
    terminal: bool,
    incomplete: bool,
    refused: bool,
    unsupported_item: Option<String>,
}

impl ResponsesAccumulator {
    pub(super) fn apply(&mut self, payload: &str, sink: &dyn EventSink) -> Result<bool> {
        let event: Value = serde_json::from_str(payload)
            .map_err(|error| anyhow!("invalid Responses streaming event: {error}"))?;
        match event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "response.output_text.delta" => {
                self.push_text(value_str(&event, "delta"), sink);
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                self.push_reasoning(value_str(&event, "delta"), sink);
            }
            "response.output_item.added" | "response.output_item.done" => {
                if let Some(item) = event.get("item") {
                    let index = event_index(&event);
                    self.items.insert(index, item.clone());
                    self.learn_item(index, item);
                }
            }
            "response.function_call_arguments.delta" => {
                let index = event_index(&event);
                self.calls.entry(index).or_default().arguments.push_str(
                    event
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                );
            }
            "response.function_call_arguments.done" => {
                let index = event_index(&event);
                if let Some(arguments) = event.get("arguments").and_then(Value::as_str) {
                    self.calls.entry(index).or_default().arguments = arguments.to_string();
                }
            }
            "response.completed" | "response.incomplete" => {
                let response = event.get("response").unwrap_or(&event);
                self.learn_response(response);
                self.terminal = true;
            }
            "response.failed" | "response.cancelled" | "error" => {
                let message = event
                    .pointer("/response/error/message")
                    .or_else(|| event.pointer("/error/message"))
                    .or_else(|| event.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("Responses API request failed");
                bail!("Responses API request failed: {message}");
            }
            _ => {}
        }
        Ok(self.terminal)
    }

    pub(super) fn finish(self) -> Result<Completion> {
        if !self.terminal {
            bail!("Responses stream ended before a terminal event");
        }
        if self.incomplete && !self.calls.is_empty() {
            bail!("Responses API returned function calls in an incomplete response");
        }
        if let Some(item_type) = &self.unsupported_item {
            bail!("Responses API returned unsupported output item type {item_type:?}");
        }
        if self.refused && !self.calls.is_empty() {
            bail!("Responses API refused the request but also returned function calls");
        }
        Ok(self.into_completion())
    }

    fn learn_response(&mut self, response: &Value) {
        self.learn_usage(response.get("usage"));
        if let Some(output) = response.get("output").and_then(Value::as_array) {
            for (index, item) in output.iter().enumerate() {
                self.items.insert(index, item.clone());
                self.learn_item(index, item);
            }
        }
        let status = response
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("completed");
        self.incomplete = status == "incomplete";
        self.finish_reason = Some(if status == "incomplete" {
            match response
                .pointer("/incomplete_details/reason")
                .and_then(Value::as_str)
            {
                Some("max_output_tokens") => "length",
                _ => "content_filter",
            }
            .to_string()
        } else if self.calls.is_empty() {
            "stop".to_string()
        } else {
            "tool_calls".to_string()
        });
    }

    fn learn_item(&mut self, index: usize, item: &Value) {
        match item.get("type").and_then(Value::as_str).unwrap_or_default() {
            "function_call" => {
                let call = self.calls.entry(index).or_default();
                if let Some(id) = item.get("call_id").and_then(Value::as_str) {
                    call.id = id.to_string();
                }
                if let Some(name) = item.get("name").and_then(Value::as_str) {
                    call.name = name.to_string();
                }
                if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
                    call.arguments = arguments.to_string();
                }
            }
            "message" => {
                if let Some(parts) = item.get("content").and_then(Value::as_array) {
                    for part in parts {
                        match part.get("type").and_then(Value::as_str).unwrap_or_default() {
                            "output_text" => {
                                if self.text.is_empty() {
                                    self.text.push_str(value_str(part, "text"));
                                }
                            }
                            "refusal" => {
                                self.refused = true;
                                if self.text.is_empty() {
                                    self.text.push_str(value_str(part, "refusal"));
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            "reasoning" => {
                if self.reasoning.is_empty() {
                    if let Some(parts) = item.get("summary").and_then(Value::as_array) {
                        for part in parts {
                            self.reasoning.push_str(value_str(part, "text"));
                        }
                    }
                    if let Some(parts) = item.get("content").and_then(Value::as_array) {
                        for part in parts {
                            self.reasoning.push_str(value_str(part, "text"));
                        }
                    }
                }
            }
            "" => {}
            item_type => self.unsupported_item = Some(item_type.to_string()),
        }
    }

    fn learn_usage(&mut self, usage: Option<&Value>) {
        let Some(usage) = usage else { return };
        self.usage.prompt_tokens = usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        self.usage.completion_tokens = usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        self.usage.total_tokens = usage
            .get("total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(self.usage.prompt_tokens + self.usage.completion_tokens);
    }

    fn push_text(&mut self, delta: &str, sink: &dyn EventSink) {
        if !delta.is_empty() {
            self.text.push_str(delta);
            sink.text_delta(delta);
        }
    }

    fn push_reasoning(&mut self, delta: &str, sink: &dyn EventSink) {
        if !delta.is_empty() {
            self.reasoning.push_str(delta);
            sink.reasoning_delta(delta);
        }
    }

    fn into_completion(mut self) -> Completion {
        let mut text_backfilled = false;
        for item in self.items.values_mut() {
            if !text_backfilled
                && item.get("type").and_then(Value::as_str) == Some("message")
                && item
                    .get("content")
                    .and_then(Value::as_array)
                    .is_some_and(Vec::is_empty)
                && !self.text.is_empty()
            {
                item["content"] = json!([{
                    "type": "output_text",
                    "text": self.text,
                    "annotations": []
                }]);
                text_backfilled = true;
            }
        }
        let responses_items = self.items.into_values().collect();
        let tool_calls = self
            .calls
            .into_values()
            .filter(|call| !call.name.is_empty())
            .collect();
        Completion {
            message: AssistantMessage {
                content: (!self.text.is_empty()).then_some(self.text),
                reasoning: (!self.reasoning.is_empty()).then_some(self.reasoning),
                tool_calls,
                responses_items,
            },
            usage: self.usage,
            finish_reason: self.finish_reason,
        }
    }
}

pub(super) fn decode_json(payload: Value) -> Result<Completion> {
    let status = payload
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if matches!(status, "failed" | "cancelled")
        || payload.get("error").is_some_and(|error| !error.is_null())
    {
        let message = payload
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or("Responses API request failed");
        bail!("Responses API request failed: {message}");
    }
    let mut accumulator = ResponsesAccumulator::default();
    accumulator.learn_response(&payload);
    accumulator.terminal = matches!(status, "completed" | "incomplete");
    accumulator.finish()
}

fn event_index(event: &Value) -> usize {
    event
        .get("output_index")
        .and_then(Value::as_u64)
        .unwrap_or_default() as usize
}

fn value_str<'a>(value: &'a Value, field: &str) -> &'a str {
    value.get(field).and_then(Value::as_str).unwrap_or_default()
}
