mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use common::{MockResponse, MockServer, RecordingSink};
use icode::agent::{Agent, AgentKind, AgentOptions};
use icode::config::{ApiStyle, ModelProfile};
use icode::interaction::{AllowAll, NoAsker};
use icode::model::{AssistantMessage, Message, ModelClient, ToolCall, ToolDef};
use icode::tools::ToolRegistry;
use serde_json::json;

fn responses_profile(server: &MockServer, stream: bool) -> ModelProfile {
    ModelProfile {
        name: "mock-responses".to_string(),
        provider: "openai".to_string(),
        api_style: ApiStyle::Responses,
        model: "gpt-test".to_string(),
        base_url: server.base_url(),
        api_key: "test-key".to_string(),
        max_tokens: Some(512),
        context_window: None,
        temperature: None,
        stream,
    }
}

fn weather_tool() -> ToolDef {
    ToolDef {
        name: "get_weather".to_string(),
        description: "Get the weather for a city".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"],
            "additionalProperties": false
        }),
    }
}

#[tokio::test]
async fn responses_stream_maps_stateless_history_and_assembles_output() {
    let server = MockServer::new(|path, body| {
        assert_eq!(path, "/v1/responses");
        assert_eq!(body["model"], "gpt-test");
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], false);
        assert_eq!(body["max_output_tokens"], 512);
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"], true);
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        assert!(body.get("messages").is_none());
        assert!(body.get("max_tokens").is_none());

        assert_eq!(
            body["tools"],
            json!([{
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": false
                },
                "strict": false
            }])
        );
        assert_eq!(
            body["input"],
            json!([
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Be concise."}]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Weather in Paris?"}]
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I will check.", "annotations": []}]
                },
                {
                    "type": "function_call",
                    "id": "fc_call_previous",
                    "call_id": "call_previous",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Paris\"}",
                    "status": "completed"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_previous",
                    "output": "{\"temperature_c\":18}"
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "And London?"}]
                }
            ])
        );

        MockResponse::sse(&[
            r#"{"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}"#,
            r#"{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"Checking current conditions. "}"#,
            r#"{"type":"response.output_item.added","output_index":1,"item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}"#,
            r#"{"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"delta":"London is "}"#,
            r#"{"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"delta":"cooler. "}"#,
            r#"{"type":"response.output_item.added","output_index":2,"item":{"id":"fc_1","type":"function_call","call_id":"call_weather","name":"get_weather","arguments":""}}"#,
            r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":2,"delta":"{\"city\":"}"#,
            r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":2,"delta":"\"London\"}"}"#,
            r#"{"type":"response.output_item.done","output_index":2,"item":{"id":"fc_1","type":"function_call","call_id":"call_weather","name":"get_weather","arguments":"{\"city\":\"London\"}"}}"#,
            r#"{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"reasoning","id":"rs_1","encrypted_content":"opaque","summary":[{"type":"summary_text","text":"Checking current conditions. "}]},{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"London is cooler. ","annotations":[]}]},{"id":"fc_1","type":"function_call","call_id":"call_weather","name":"get_weather","arguments":"{\"city\":\"London\"}","status":"completed"}],"usage":{"input_tokens":31,"output_tokens":9,"total_tokens":40}}}"#,
        ])
    });
    let client = ModelClient::from_profile(&responses_profile(&server, true)).unwrap();
    assert_eq!(client.endpoint(), format!("{}/v1/responses", server.url()));
    let sink = RecordingSink::default();
    let messages = vec![
        Message::system("Be concise."),
        Message::user("Weather in Paris?"),
        Message::assistant(AssistantMessage {
            content: Some("I will check.".to_string()),
            reasoning: Some("Need fresh data.".to_string()),
            tool_calls: vec![ToolCall {
                id: "call_previous".to_string(),
                name: "get_weather".to_string(),
                arguments: r#"{"city":"Paris"}"#.to_string(),
            }],
            responses_items: Vec::new(),
        }),
        Message::tool("call_previous", "get_weather", r#"{"temperature_c":18}"#),
        Message::user("And London?"),
    ];

    let completion = client
        .complete(&messages, &[weather_tool()], &sink)
        .await
        .unwrap();

    assert_eq!(
        completion.message.content.as_deref(),
        Some("London is cooler. ")
    );
    assert_eq!(
        completion.message.reasoning.as_deref(),
        Some("Checking current conditions. ")
    );
    assert_eq!(
        completion.message.tool_calls,
        vec![ToolCall {
            id: "call_weather".to_string(),
            name: "get_weather".to_string(),
            arguments: r#"{"city":"London"}"#.to_string(),
        }]
    );
    assert_eq!(completion.message.responses_items.len(), 3);
    assert_eq!(completion.message.responses_items[0]["id"], "rs_1");
    assert_eq!(
        completion.message.responses_items[0]["encrypted_content"],
        "opaque"
    );
    assert_eq!(completion.message.responses_items[2]["id"], "fc_1");
    assert_eq!(completion.usage.prompt_tokens, 31);
    assert_eq!(completion.usage.completion_tokens, 9);
    assert_eq!(completion.usage.total_tokens, 40);
    assert_eq!(
        sink.events(),
        vec![
            "reasoning:Checking current conditions. ".to_string(),
            "text:London is ".to_string(),
            "text:cooler. ".to_string()
        ]
    );
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn responses_non_streaming_json_extracts_output_text_and_usage() {
    let server = MockServer::new(|path, body| {
        assert_eq!(path, "/v1/responses");
        assert_eq!(body["stream"], false);
        assert_eq!(
            body["input"],
            json!([{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}]
            }])
        );
        MockResponse::json(
            200,
            &json!({
                "id": "resp_json",
                "status": "completed",
                "error": null,
                "output": [{
                    "id": "msg_json",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello ", "annotations": []},
                        {"type": "output_text", "text": "from Responses.", "annotations": []}
                    ]
                }],
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 5,
                    "total_tokens": 9
                }
            })
            .to_string(),
        )
    });
    let client = ModelClient::from_profile(&responses_profile(&server, false)).unwrap();
    let sink = RecordingSink::default();

    let completion = client
        .complete(&[Message::user("Hello")], &[], &sink)
        .await
        .unwrap();

    assert_eq!(
        completion.message.content.as_deref(),
        Some("Hello from Responses.")
    );
    assert!(completion.message.tool_calls.is_empty());
    assert_eq!(completion.usage.prompt_tokens, 4);
    assert_eq!(completion.usage.completion_tokens, 5);
    assert_eq!(completion.usage.total_tokens, 9);
    assert!(sink.events().is_empty());
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn responses_stream_requires_a_terminal_event() {
    let server = MockServer::new(|_, _| {
        MockResponse::sse(&[
            r#"{"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}"#,
            r#"{"type":"response.output_text.delta","output_index":0,"delta":"partial"}"#,
        ])
    });
    let client = ModelClient::from_profile(&responses_profile(&server, true)).unwrap();
    let sink = RecordingSink::default();

    let error = client
        .complete(&[Message::user("Hello")], &[], &sink)
        .await
        .unwrap_err();

    assert!(
        error.to_string().contains("before a terminal event"),
        "{error:#}"
    );
}

#[tokio::test]
async fn responses_stream_ignores_empty_data_events() {
    let server = MockServer::new(|_, _| {
        MockResponse {
        status: 200,
        content_type: "text/event-stream".to_string(),
        body: concat!(
            "data: \n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"done\"}]}]}}\n\n"
        )
        .to_string(),
    }
    });
    let client = ModelClient::from_profile(&responses_profile(&server, true)).unwrap();

    let completion = client
        .complete(&[Message::user("Hello")], &[], &RecordingSink::default())
        .await
        .unwrap();

    assert_eq!(completion.message.content.as_deref(), Some("done"));
}

#[tokio::test]
async fn responses_rejects_unsupported_executable_output_items() {
    let server = MockServer::new(|_, _| {
        MockResponse::json(
            200,
            &json!({
                "id": "resp_1",
                "status": "completed",
                "output": [{"type": "web_search_call", "id": "ws_1", "status": "completed"}],
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
            })
            .to_string(),
        )
    });
    let client = ModelClient::from_profile(&responses_profile(&server, false)).unwrap();
    let sink = RecordingSink::default();

    let error = client
        .complete(&[Message::user("Search")], &[], &sink)
        .await
        .unwrap_err();

    assert!(
        error.to_string().contains("unsupported output item"),
        "{error:#}"
    );
}

#[tokio::test]
async fn responses_rejects_refusal_with_function_calls() {
    let server = MockServer::new(|_, _| {
        MockResponse::json(
            200,
            &json!({
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "Cannot comply"}]},
                    {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"}
                ],
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
            })
            .to_string(),
        )
    });
    let client = ModelClient::from_profile(&responses_profile(&server, false)).unwrap();
    let sink = RecordingSink::default();

    let error = client
        .complete(&[Message::user("Do it")], &[weather_tool()], &sink)
        .await
        .unwrap_err();

    assert!(error.to_string().contains("refused"), "{error:#}");
}

#[tokio::test]
async fn responses_replays_native_output_items_without_reconstruction() {
    let native_items = vec![
        json!({
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "opaque",
            "summary": []
        }),
        json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{\"city\":\"Paris\"}",
            "status": "completed"
        }),
    ];
    let expected_items = native_items.clone();
    let server = MockServer::new(move |_, body| {
        assert_eq!(body["input"][1], expected_items[0]);
        assert_eq!(body["input"][2], expected_items[1]);
        MockResponse::json(
            200,
            &json!({
                "id": "resp_2",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Done", "annotations": []}]
                }],
                "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}
            })
            .to_string(),
        )
    });
    let client = ModelClient::from_profile(&responses_profile(&server, false)).unwrap();
    let messages = vec![
        Message::user("Call it"),
        Message::assistant(AssistantMessage {
            content: None,
            reasoning: Some("display-only summary".to_string()),
            tool_calls: vec![ToolCall {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                arguments: "{}".to_string(),
            }],
            responses_items: native_items,
        }),
        Message::tool("call_1", "get_weather", "sunny"),
    ];

    client
        .complete(&messages, &[], &RecordingSink::default())
        .await
        .unwrap();
}

#[tokio::test]
async fn responses_rejects_function_calls_from_incomplete_output() {
    let server = MockServer::new(|_, _| {
        MockResponse::json(
            200,
            &json!({
                "id": "resp_1",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [{
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{}",
                    "status": "in_progress"
                }],
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
            })
            .to_string(),
        )
    });
    let client = ModelClient::from_profile(&responses_profile(&server, false)).unwrap();

    let error = client
        .complete(
            &[Message::user("Weather")],
            &[weather_tool()],
            &RecordingSink::default(),
        )
        .await
        .unwrap_err();

    assert!(
        error.to_string().contains("incomplete response"),
        "{error:#}"
    );
}

#[tokio::test]
async fn agent_recovers_once_from_responses_context_overflow_with_stateless_summary() {
    const CURRENT_USER: &str = "Answer the current question exactly.";

    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_server = attempts.clone();
    let server = MockServer::new(move |path, _body| {
        assert_eq!(path, "/v1/responses");
        if attempts_for_server.fetch_add(1, Ordering::SeqCst) == 0 {
            MockResponse::json(
                400,
                &json!({
                    "error": {
                        "message": "This model's maximum context length was exceeded.",
                        "type": "invalid_request_error",
                        "param": "input",
                        "code": "context_length_exceeded"
                    }
                })
                .to_string(),
            )
        } else {
            MockResponse::json(
                200,
                &json!({
                    "id": "resp_recovered",
                    "status": "completed",
                    "error": null,
                    "output": [{
                        "id": "msg_recovered",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": "Recovered answer.",
                            "annotations": []
                        }]
                    }],
                    "usage": {
                        "input_tokens": 80,
                        "output_tokens": 3,
                        "total_tokens": 83
                    }
                })
                .to_string(),
            )
        }
    });
    let mut profile = responses_profile(&server, false);
    profile.context_window = Some(20_000);
    let client = ModelClient::from_profile(&profile).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let agent = Agent::new(
        Arc::new(client),
        Arc::new(ToolRegistry::builtin()),
        Arc::new(AllowAll),
        Arc::new(NoAsker),
        AgentOptions {
            kind: AgentKind::Qa,
            cwd: directory.path().to_path_buf(),
            max_rounds: 2,
            tool_timeout: Duration::from_secs(5),
        },
    );
    let old_user = format!("old completed text turn: {}", "x".repeat(5_000));
    let mut messages = vec![
        Message::user(old_user.clone()),
        Message::assistant(AssistantMessage::text("old completed answer")),
        Message::user(CURRENT_USER),
    ];

    let turn = agent
        .run(&mut messages, &RecordingSink::default())
        .await
        .unwrap();

    assert_eq!(turn.text, "Recovered answer.");
    assert_eq!(turn.rounds, 1, "the transport retry is one agent round");
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert_eq!(server.request_count(), 2);

    let requests = server.requests();
    let first_input = requests[0]["input"].as_array().unwrap();
    assert!(first_input.iter().any(|item| {
        item.pointer("/content/0/text")
            .and_then(|text| text.as_str())
            == Some(old_user.as_str())
    }));
    assert!(!first_input.iter().any(|item| {
        item.pointer("/content/0/text")
            .and_then(|text| text.as_str())
            .is_some_and(|text| text.starts_with("[Earlier conversation summary]\n"))
    }));

    let retry = &requests[1];
    assert_eq!(retry["store"], false);
    assert!(retry.get("previous_response_id").is_none());
    let retry_input = retry["input"].as_array().unwrap();
    assert!(retry_input.iter().any(|item| {
        item["role"] == "user"
            && item
                .pointer("/content/0/text")
                .and_then(|text| text.as_str())
                .is_some_and(|text| {
                    text.starts_with(
                        "[Earlier conversation summary]\nUser: old completed text turn:",
                    )
                })
    }));
    assert!(retry_input.iter().any(|item| {
        item["role"] == "user"
            && item
                .pointer("/content/0/text")
                .and_then(|text| text.as_str())
                == Some(CURRENT_USER)
    }));
}
