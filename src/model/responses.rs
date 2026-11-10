//! Stateless OpenAI Responses API wire mapping.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};

use super::{
    AssistantMessage, Completion, Message, ResponsesContinuation, ToolCall, ToolDef, Usage,
};
use crate::agent::EventSink;

#[derive(Clone, Default)]
pub(super) struct RequestFeatures {
    pub include_encrypted_reasoning: bool,
    pub web_search: bool,
    pub store: bool,
    pub previous_response_id: Option<String>,
}

pub(super) fn request_body(
    messages: &[Message],
    tools: &[ToolDef],
    model: &str,
    stream: bool,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    features: RequestFeatures,
) -> Value {
    let mut input = Vec::new();
    for message in messages {
        match message {
            Message::System { content }
            | Message::User { content }
            | Message::InjectedUser { content, .. }
            | Message::Summary { content }
            | Message::LastWords { content } => input.push(json!({
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
        "store": features.store,
    });
    if let Some(previous_response_id) = &features.previous_response_id {
        body["previous_response_id"] = json!(previous_response_id);
    }
    let mut declarations: Vec<Value> = tools
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
        .collect();
    if features.web_search {
        declarations.push(json!({"type": "web_search"}));
    }
    if !declarations.is_empty() {
        body["tools"] = Value::Array(declarations);
    }
    if !tools.is_empty() {
        body["tool_choice"] = json!("auto");
        body["parallel_tool_calls"] = json!(true);
    }
    if features.include_encrypted_reasoning {
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
    text_by_index: BTreeMap<usize, String>,
    streamed_text: BTreeSet<usize>,
    reasoning: String,
    calls: BTreeMap<usize, ToolCall>,
    items: BTreeMap<usize, Value>,
    usage: Usage,
    finish_reason: Option<String>,
    terminal: bool,
    incomplete: bool,
    refused: bool,
    unsupported_item: Option<String>,
    response_id: Option<String>,
    response_id_mismatch: bool,
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
            "response.created" => {
                self.learn_response_id(event.get("response").unwrap_or(&event));
            }
            "response.output_text.delta" => {
                self.push_text(event_index(&event), value_str(&event, "delta"), sink);
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
        if self.response_id_mismatch {
            bail!("Responses API changed response id within one request");
        }
        if self.incomplete && !self.calls.is_empty() {
            bail!("Responses API returned function calls in an incomplete response");
        }
        if self
            .calls
            .values()
            .any(|call| call.name.trim().is_empty() || call.id.trim().is_empty())
        {
            bail!("Responses API returned a function call without a name or call_id");
        }
        if let Some(item_type) = &self.unsupported_item {
            bail!("Responses API returned unsupported output item type {item_type:?}");
        }
        if self.refused && !self.calls.is_empty() {
            bail!("Responses API refused the request but also returned function calls");
        }
        Ok(self.into_completion())
    }

    fn learn_response_id(&mut self, response: &Value) {
        let Some(id) = response
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
        else {
            return;
        };
        match &self.response_id {
            Some(existing) if existing != id => self.response_id_mismatch = true,
            None => self.response_id = Some(id.to_string()),
            Some(_) => {}
        }
    }

    fn learn_response(&mut self, response: &Value) {
        self.learn_response_id(response);
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
                    let mut text = String::new();
                    for part in parts {
                        match part.get("type").and_then(Value::as_str).unwrap_or_default() {
                            "output_text" => text.push_str(value_str(part, "text")),
                            "refusal" => {
                                self.refused = true;
                                text.push_str(value_str(part, "refusal"));
                            }
                            _ => {}
                        }
                    }
                    if !self.streamed_text.contains(&index) {
                        self.text_by_index.insert(index, text);
                        self.rebuild_text();
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
            "web_search_call" => {}
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

    fn push_text(&mut self, index: usize, delta: &str, sink: &dyn EventSink) {
        if !delta.is_empty() {
            if self.streamed_text.insert(index) {
                self.text_by_index.insert(index, String::new());
            }
            self.text_by_index.entry(index).or_default().push_str(delta);
            self.rebuild_text();
            sink.text_delta(delta);
        }
    }

    fn rebuild_text(&mut self) {
        self.text = self.text_by_index.values().map(String::as_str).collect();
    }

    fn push_reasoning(&mut self, delta: &str, sink: &dyn EventSink) {
        if !delta.is_empty() {
            self.reasoning.push_str(delta);
            sink.reasoning_delta(delta);
        }
    }

    fn into_completion(mut self) -> Completion {
        for (index, item) in &mut self.items {
            if item.get("type").and_then(Value::as_str) == Some("message")
                && item
                    .get("content")
                    .and_then(Value::as_array)
                    .is_some_and(Vec::is_empty)
            {
                let text = self
                    .text_by_index
                    .get(index)
                    .filter(|text| !text.is_empty());
                if let Some(text) = text {
                    item["content"] = json!([{
                        "type": "output_text",
                        "text": text,
                        "annotations": []
                    }]);
                }
            }
        }
        let responses_items = self.items.into_values().collect();
        let tool_calls = self.calls.into_values().collect();
        Completion {
            message: AssistantMessage {
                content: (!self.text.is_empty()).then_some(self.text),
                reasoning: (!self.reasoning.is_empty()).then_some(self.reasoning),
                tool_calls,
                responses_items,
                responses_continuation: self.response_id.map(|id| {
                    Box::new(ResponsesContinuation {
                        id,
                        scope_sha256: String::new(),
                        input_sha256: String::new(),
                    })
                }),
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
    if !matches!(status, "" | "completed" | "incomplete") {
        bail!("Responses API returned a non-terminal status {status:?}");
    }
    accumulator.terminal = true;
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

#[cfg(test)]
mod tests {
    use super::{decode_json, request_body, RequestFeatures, ResponsesAccumulator};
    use crate::agent::NullSink;
    use crate::model::Message;

    #[test]
    fn replays_hosted_web_search_items_verbatim() {
        let hosted = serde_json::json!({
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "Rust"}
        });
        let body = request_body(
            &[Message::assistant(crate::model::AssistantMessage {
                responses_items: vec![hosted.clone()],
                ..crate::model::AssistantMessage::default()
            })],
            &[],
            "model",
            true,
            None,
            None,
            RequestFeatures {
                web_search: true,
                ..RequestFeatures::default()
            },
        );

        assert_eq!(body["input"], serde_json::json!([hosted]));
        assert_eq!(body["tools"], serde_json::json!([{"type": "web_search"}]));
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn maps_last_words_as_stateless_user_context() {
        let body = request_body(
            &[
                Message::user("task"),
                Message::last_words("[LAST_WORDS] progress"),
            ],
            &[],
            "model",
            false,
            None,
            Some(128),
            RequestFeatures::default(),
        );

        assert_eq!(body["input"][1]["role"], "user");
        assert_eq!(
            body["input"][1]["content"][0]["text"],
            "[LAST_WORDS] progress"
        );
        assert_eq!(body["store"], false);
        assert!(body.get("previous_response_id").is_none());
    }

    #[test]
    fn rejects_response_id_changes_between_stream_events() {
        let mut accumulator = ResponsesAccumulator::default();
        assert!(!accumulator
            .apply(
                r#"{"type":"response.created","response":{"id":"resp_a","status":"in_progress"}}"#,
                &NullSink,
            )
            .unwrap());
        assert!(accumulator
            .apply(
                r#"{"type":"response.completed","response":{"id":"resp_b","status":"completed","output":[]}}"#,
                &NullSink,
            )
            .unwrap());

        assert_eq!(
            accumulator.finish().unwrap_err().to_string(),
            "Responses API changed response id within one request"
        );
    }

    #[test]
    fn maps_injected_user_without_leaking_identity() {
        let body = request_body(
            &[Message::injected_user(
                "injection-1",
                "updated requirements",
            )],
            &[],
            "model",
            false,
            None,
            Some(128),
            RequestFeatures::default(),
        );

        assert_eq!(
            body["input"][0],
            serde_json::json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "updated requirements"}],
            })
        );
        assert!(body["input"][0].get("injection_id").is_none());
    }

    #[test]
    fn accumulates_function_call_argument_deltas_before_completion() {
        let mut accumulator = ResponsesAccumulator::default();
        accumulator
            .apply(
                r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"weather","arguments":""}}"#,
                &NullSink,
            )
            .unwrap();
        accumulator
            .apply(
                r#"{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"city\":"}"#,
                &NullSink,
            )
            .unwrap();
        accumulator
            .apply(
                r#"{"type":"response.function_call_arguments.delta","output_index":0,"delta":"\"London\"}"}"#,
                &NullSink,
            )
            .unwrap();

        assert_eq!(
            accumulator
                .calls
                .get(&0)
                .map(|call| call.arguments.as_str()),
            Some(r#"{"city":"London"}"#)
        );
    }

    #[test]
    fn streamed_text_replaces_snapshot_fallback() {
        let mut accumulator = ResponsesAccumulator::default();
        accumulator
            .apply(
                r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","content":[{"type":"output_text","text":"snapshot"}]}}"#,
                &NullSink,
            )
            .unwrap();
        accumulator
            .apply(
                r#"{"type":"response.output_text.delta","output_index":0,"delta":"delta"}"#,
                &NullSink,
            )
            .unwrap();

        assert_eq!(accumulator.text, "delta");
    }

    #[test]
    fn streaming_one_index_preserves_other_message_snapshots() {
        let mut accumulator = ResponsesAccumulator::default();
        accumulator
            .apply(
                r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"message","content":[{"type":"output_text","text":"A"}]}}"#,
                &NullSink,
            )
            .unwrap();
        accumulator
            .apply(
                r#"{"type":"response.output_item.added","output_index":1,"item":{"type":"message","content":[]}}"#,
                &NullSink,
            )
            .unwrap();
        accumulator
            .apply(
                r#"{"type":"response.output_text.delta","output_index":1,"delta":"B"}"#,
                &NullSink,
            )
            .unwrap();

        assert_eq!(accumulator.text, "AB");
    }

    #[test]
    fn rejects_function_calls_without_identity() {
        let error = decode_json(serde_json::json!({
            "status": "completed",
            "output": [{
                "type": "function_call",
                "call_id": "call_1",
                "arguments": "{}"
            }]
        }))
        .unwrap_err();

        assert!(error.to_string().contains("without a name or call_id"));
    }

    #[test]
    fn accepts_missing_status_but_rejects_non_terminal_status() {
        let completion = decode_json(serde_json::json!({
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "done"}]
            }]
        }))
        .unwrap();
        assert_eq!(completion.message.content.as_deref(), Some("done"));

        let error =
            decode_json(serde_json::json!({"status": "in_progress", "output": []})).unwrap_err();
        assert!(error.to_string().contains("non-terminal status"));
    }
}
