mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{MockResponse, MockServer, RecordingSink};
use icode::agent::{Agent, AgentKind, AgentOptions};
use icode::config::{
    ApiStyle, ModelProfile, ResponsesFileSearchComparisonFilter,
    ResponsesFileSearchComparisonOperator, ResponsesFileSearchCompoundFilter,
    ResponsesFileSearchCompoundOperator, ResponsesFileSearchConfig, ResponsesFileSearchFilter,
    ResponsesFileSearchFilterScalar, ResponsesFileSearchFilterValue,
    ResponsesFileSearchHybridSearch, ResponsesFileSearchRanker, ResponsesFileSearchRankingOptions,
};
use icode::interaction::{AllowAll, NoAsker};
use icode::model::{AssistantMessage, Message, ModelClient, ToolCall, ToolDef};
use icode::tools::ToolRegistry;
use serde_json::json;

fn responses_profile(server: &MockServer, stream: bool) -> ModelProfile {
    ModelProfile {
        name: "mock-responses".to_string(),
        provider: "openai".to_string(),
        api_style: ApiStyle::Responses,
        responses_web_search: false,
        responses_code_interpreter: false,
        responses_file_search: None,
        responses_stored_continuation: false,
        responses_background: false,
        model: "gpt-test".to_string(),
        base_url: server.base_url(),
        api_key: "test-key".to_string(),
        max_tokens: Some(512),
        context_window: None,
        temperature: None,
        last_words_max_output_tokens: 1_024,
        last_words_supplement: String::new(),
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
async fn responses_phase_four_generates_last_words_without_tools_or_streaming() {
    let note_body = "concrete progress detail ".repeat(30);
    let note = format!("## Task\n{note_body}\n## Progress\n{note_body}\n## Next\n{note_body}");
    let note_for_server = note.clone();
    let server = MockServer::new(move |path, body| {
        assert_eq!(path, "/v1/responses");
        if body["stream"] == false {
            assert_eq!(body["store"], false);
            assert!(body.get("tools").is_none());
            assert!(body.get("tool_choice").is_none());
            assert!(body.get("parallel_tool_calls").is_none());
            assert!(body.get("include").is_none());
            return MockResponse::json(
                200,
                &json!({
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": note_for_server}]
                    }],
                    "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
                })
                .to_string(),
            );
        }
        MockResponse::sse(&[
            r#"{"type":"response.output_text.delta","output_index":0,"delta":"continued"}"#,
            r#"{"type":"response.completed","response":{"id":"resp_after_phase_four","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"continued"}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}"#,
        ])
    });
    let directory = tempfile::tempdir().unwrap();
    let mut profile = responses_profile(&server, true);
    profile.responses_web_search = true;
    profile.responses_code_interpreter = true;
    profile.responses_stored_continuation = true;
    profile.context_window = Some(30_000);
    profile.max_tokens = Some(3_000);
    let agent = Agent::new(
        Arc::new(ModelClient::from_profile(&profile).unwrap()),
        Arc::new(ToolRegistry::builtin()),
        Arc::new(AllowAll),
        Arc::new(NoAsker),
        AgentOptions {
            kind: AgentKind::Code,
            cwd: directory.path().to_path_buf(),
            max_rounds: 6,
            tool_timeout: Duration::from_secs(10),
        },
    );
    let sink = RecordingSink::default();
    let mut messages = vec![
        Message::user("inspect the large result and continue"),
        Message::assistant(AssistantMessage {
            tool_calls: vec![ToolCall {
                id: "call_large".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"large.txt"}"#.into(),
            }],
            ..AssistantMessage::default()
        }),
        Message::tool("call_large", "read_file", "result ".repeat(2_500)),
    ];

    let turn = agent.run(&mut messages, &sink).await.unwrap();

    assert_eq!(server.request_count(), 2);
    assert_eq!(turn.rounds, 1);
    assert_eq!(turn.usage.total_tokens, 25);
    assert_eq!(sink.projections.lock().unwrap().len(), 1);
    assert!(matches!(messages[2], Message::LastWords { .. }));
    let requests = server.requests();
    assert_eq!(requests[0]["stream"], false);
    assert_eq!(requests[0]["store"], false);
    assert!(requests[0].get("previous_response_id").is_none());
    assert_eq!(requests[1]["stream"], true);
    assert_eq!(requests[1]["store"], true);
    assert!(requests[1].get("previous_response_id").is_none());
    let ordinary_tools = requests[1]["tools"].as_array().unwrap();
    assert!(ordinary_tools.iter().any(|tool| tool["type"] == "function"));
    assert!(ordinary_tools
        .iter()
        .any(|tool| tool == &json!({"type": "web_search"})));
    assert_eq!(
        ordinary_tools.last().unwrap(),
        &json!({
            "type": "code_interpreter",
            "container": {
                "type": "auto",
                "network_policy": {"type": "disabled"}
            }
        })
    );
    assert_eq!(requests[1]["tool_choice"], "auto");
    assert_eq!(requests[1]["parallel_tool_calls"], true);
    assert!(requests[1]["input"].as_array().unwrap().iter().any(|item| {
        item["content"][0]["text"]
            .as_str()
            .is_some_and(|content| content.contains("[LAST_WORDS]"))
    }));
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
            }, {
                "type": "web_search"
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
            r#"{"type":"response.output_item.added","output_index":2,"item":{"id":"ws_1","type":"web_search_call","status":"in_progress"}}"#,
            r#"{"type":"response.web_search_call.searching","output_index":2,"item_id":"ws_1"}"#,
            r#"{"type":"response.output_item.done","output_index":2,"item":{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"London weather"}}}"#,
            r#"{"type":"response.output_item.added","output_index":3,"item":{"id":"fc_1","type":"function_call","call_id":"call_weather","name":"get_weather","arguments":""}}"#,
            r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":3,"delta":"{\"city\":"}"#,
            r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":3,"delta":"\"London\"}"}"#,
            r#"{"type":"response.output_item.done","output_index":3,"item":{"id":"fc_1","type":"function_call","call_id":"call_weather","name":"get_weather","arguments":"{\"city\":\"London\"}"}}"#,
            r#"{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"reasoning","id":"rs_1","encrypted_content":"opaque","summary":[{"type":"summary_text","text":"Checking current conditions. "}]},{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"London is cooler. ","annotations":[]}]},{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"London weather"}},{"id":"fc_1","type":"function_call","call_id":"call_weather","name":"get_weather","arguments":"{\"city\":\"London\"}","status":"completed"}],"usage":{"input_tokens":31,"output_tokens":9,"total_tokens":40}}}"#,
        ])
    });
    let mut profile = responses_profile(&server, true);
    profile.responses_web_search = true;
    let client = ModelClient::from_profile(&profile).unwrap();
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
            responses_continuation: None,
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
    assert_eq!(completion.message.responses_items.len(), 4);
    assert_eq!(completion.message.responses_items[0]["id"], "rs_1");
    assert_eq!(
        completion.message.responses_items[0]["encrypted_content"],
        "opaque"
    );
    assert_eq!(completion.message.responses_items[2]["id"], "ws_1");
    assert_eq!(completion.message.responses_items[2]["status"], "completed");
    assert_eq!(completion.message.responses_items[3]["id"], "fc_1");
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
        declared_length: None,
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
async fn responses_preserves_hosted_web_search_without_local_execution() {
    let server = MockServer::new(|_, body| {
        assert_eq!(body["tools"], json!([{"type": "web_search"}]));
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
        MockResponse::json(
            200,
            &json!({
                "id": "resp_search",
                "status": "completed",
                "output": [
                    {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "status": "completed",
                        "action": {"type": "search", "query": "Rust language"}
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Rust is a systems language.", "annotations": []}]
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}
            })
            .to_string(),
        )
    });
    let mut profile = responses_profile(&server, false);
    profile.responses_web_search = true;
    let client = ModelClient::from_profile(&profile).unwrap();

    let completion = client
        .complete(
            &[Message::user("What is Rust?")],
            &[],
            &RecordingSink::default(),
        )
        .await
        .unwrap();

    assert_eq!(
        completion.message.content.as_deref(),
        Some("Rust is a systems language.")
    );
    assert!(completion.message.tool_calls.is_empty());
    assert_eq!(completion.message.responses_items.len(), 2);
    assert_eq!(
        completion.message.responses_items[0],
        json!({
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "Rust language"}
        })
    );
}

#[tokio::test]
async fn responses_preserves_and_replays_hosted_code_interpreter_calls() {
    let server = MockServer::new(|_, body| {
        if body["input"].as_array().unwrap().len() == 1 {
            assert_eq!(
                body["tools"],
                json!([{
                    "type": "code_interpreter",
                    "container": {
                        "type": "auto",
                        "network_policy": {"type": "disabled"}
                    }
                }])
            );
            assert!(body["tools"][0]["container"].get("file_ids").is_none());
            assert!(body["tools"][0]["container"].get("memory_limit").is_none());
            assert!(body["tools"][0].get("allowed_callers").is_none());
            assert!(body.get("tool_choice").is_none());
            return MockResponse::json(
                200,
                &json!({
                    "id": "resp_python",
                    "status": "completed",
                    "output": [
                        {
                            "type": "code_interpreter_call",
                            "id": "ci_1",
                            "code": "print(6 * 7)",
                            "container_id": "cntr_1",
                            "outputs": [{"type":"logs","logs":"42\n"}],
                            "status": "completed"
                        },
                        {
                            "type": "message",
                            "id": "msg_1",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type":"output_text","text":"The result is 42.","annotations":[]}]
                        }
                    ],
                    "usage": {"input_tokens":4,"output_tokens":3,"total_tokens":7}
                })
                .to_string(),
            );
        }
        assert_eq!(body["input"][1]["type"], "code_interpreter_call");
        assert_eq!(body["input"][1]["container_id"], "cntr_1");
        assert_eq!(body["input"][1]["outputs"][0]["logs"], "42\n");
        MockResponse::json(
            200,
            &json!({
                "id":"resp_explanation",
                "status":"completed",
                "output":[{
                    "type":"message",
                    "role":"assistant",
                    "content":[{"type":"output_text","text":"Six times seven is 42.","annotations":[]}]
                }],
                "usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10}
            })
            .to_string(),
        )
    });
    let mut profile = responses_profile(&server, false);
    profile.responses_code_interpreter = true;
    let client = ModelClient::from_profile(&profile).unwrap();
    let first = client
        .complete(
            &[Message::user("Calculate six times seven")],
            &[],
            &RecordingSink::default(),
        )
        .await
        .unwrap();
    assert_eq!(first.message.content.as_deref(), Some("The result is 42."));
    assert!(first.message.tool_calls.is_empty());
    assert_eq!(first.message.responses_items.len(), 2);
    assert_eq!(
        first.message.responses_items[0]["type"],
        "code_interpreter_call"
    );

    let messages = vec![
        Message::user("Calculate six times seven"),
        Message::assistant(first.message),
        Message::user("Explain the result"),
    ];
    let second = client
        .complete(&messages, &[], &RecordingSink::default())
        .await
        .unwrap();
    assert_eq!(
        second.message.content.as_deref(),
        Some("Six times seven is 42.")
    );
}

#[tokio::test]
async fn responses_preserves_hosted_file_search_results_and_citations() {
    let server = MockServer::new(|_, body| {
        assert_eq!(
            body["tools"],
            json!([{
                "type": "file_search",
                "vector_store_ids": ["vs_docs"],
                "filters": {
                    "type": "and",
                    "filters": [
                        {"type": "eq", "key": "department", "value": "engineering"},
                        {
                            "type": "or",
                            "filters": [
                                {"type": "gte", "key": "year", "value": 2025.0},
                                {"type": "in", "key": "category", "value": ["guide", 7.0]}
                            ]
                        }
                    ]
                },
                "max_num_results": 10,
                "ranking_options": {
                    "hybrid_search": {
                        "embedding_weight": 0.7,
                        "text_weight": 0.3
                    },
                    "ranker": "default-2024-11-15",
                    "score_threshold": 0.25
                }
            }])
        );
        assert_eq!(
            body["include"],
            json!(["reasoning.encrypted_content", "file_search_call.results"])
        );
        assert!(body.get("tool_choice").is_none());
        MockResponse::json(
            200,
            &json!({
                "id": "resp_file_search",
                "status": "completed",
                "output": [
                    {
                        "type": "file_search_call",
                        "id": "fs_1",
                        "queries": ["Rust ownership"],
                        "status": "completed",
                        "results": [{
                            "file_id": "file_1",
                            "filename": "guide.md",
                            "score": 0.91,
                            "text": "Ownership has one owner."
                        }]
                    },
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": "Ownership has one owner.",
                            "annotations": [{
                                "type": "file_citation",
                                "file_id": "file_1",
                                "filename": "guide.md",
                                "index": 0
                            }]
                        }]
                    }
                ],
                "usage": {"input_tokens": 6, "output_tokens": 4, "total_tokens": 10}
            })
            .to_string(),
        )
    });
    let mut profile = responses_profile(&server, false);
    profile.responses_file_search = Some(ResponsesFileSearchConfig {
        vector_store_ids: vec!["vs_docs".into()],
        filters: Some(ResponsesFileSearchFilter::Compound(
            ResponsesFileSearchCompoundFilter {
                operator: ResponsesFileSearchCompoundOperator::And,
                filters: vec![
                    ResponsesFileSearchFilter::Comparison(ResponsesFileSearchComparisonFilter {
                        key: "department".into(),
                        operator: ResponsesFileSearchComparisonOperator::Eq,
                        value: ResponsesFileSearchFilterValue::String("engineering".into()),
                    }),
                    ResponsesFileSearchFilter::Compound(ResponsesFileSearchCompoundFilter {
                        operator: ResponsesFileSearchCompoundOperator::Or,
                        filters: vec![
                            ResponsesFileSearchFilter::Comparison(
                                ResponsesFileSearchComparisonFilter {
                                    key: "year".into(),
                                    operator: ResponsesFileSearchComparisonOperator::Gte,
                                    value: ResponsesFileSearchFilterValue::Number(2025.0),
                                },
                            ),
                            ResponsesFileSearchFilter::Comparison(
                                ResponsesFileSearchComparisonFilter {
                                    key: "category".into(),
                                    operator: ResponsesFileSearchComparisonOperator::In,
                                    value: ResponsesFileSearchFilterValue::List(vec![
                                        ResponsesFileSearchFilterScalar::String("guide".into()),
                                        ResponsesFileSearchFilterScalar::Number(7.0),
                                    ]),
                                },
                            ),
                        ],
                    }),
                ],
            },
        )),
        max_num_results: Some(10),
        ranking_options: Some(ResponsesFileSearchRankingOptions {
            hybrid_search: Some(ResponsesFileSearchHybridSearch {
                embedding_weight: 0.7,
                text_weight: 0.3,
            }),
            ranker: Some(ResponsesFileSearchRanker::Default20241115),
            score_threshold: Some(0.25),
        }),
    });

    let completion = ModelClient::from_profile(&profile)
        .unwrap()
        .complete(
            &[Message::user("What does the guide say about ownership?")],
            &[],
            &RecordingSink::default(),
        )
        .await
        .unwrap();

    assert_eq!(
        completion.message.content.as_deref(),
        Some("Ownership has one owner.")
    );
    assert!(completion.message.tool_calls.is_empty());
    assert_eq!(completion.message.responses_items.len(), 2);
    assert_eq!(
        completion.message.responses_items[0]["type"],
        "file_search_call"
    );
    assert_eq!(
        completion.message.responses_items[1]["content"][0]["annotations"][0]["type"],
        "file_citation"
    );
}

#[tokio::test]
async fn responses_rejects_unsupported_executable_output_items() {
    let server = MockServer::new(|_, _| {
        MockResponse::json(
            200,
            &json!({
                "id": "resp_1",
                "status": "completed",
                "output": [{"type": "mcp_call", "id": "mcp_1", "status": "completed"}],
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
            responses_continuation: None,
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

#[tokio::test]
async fn responses_stored_continuation_reuses_only_a_scope_matching_tail() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_server = Arc::clone(&calls);
    let server = MockServer::new(move |_, _| {
        let call = calls_for_server.fetch_add(1, Ordering::SeqCst) + 1;
        MockResponse::json(
            200,
            &json!({
                "id": format!("resp_{call}"),
                "status": "completed",
                "output": [{
                    "type": "message",
                    "id": format!("msg_{call}"),
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": format!("answer {call}"), "annotations": []}]
                }],
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
            })
            .to_string(),
        )
    });
    let mut profile = responses_profile(&server, false);
    profile.responses_stored_continuation = true;
    let client = ModelClient::from_profile(&profile).unwrap();
    let sink = RecordingSink::default();

    let mut messages = vec![Message::system("prompt A"), Message::user("first")];
    let first = client.complete(&messages, &[], &sink).await.unwrap();
    assert_eq!(
        first
            .message
            .responses_continuation
            .as_ref()
            .map(|handle| handle.id.as_str()),
        Some("resp_1")
    );
    messages.push(Message::assistant(first.message));
    messages.push(Message::user("second"));

    let second = client.complete(&messages, &[], &sink).await.unwrap();
    assert_eq!(
        second
            .message
            .responses_continuation
            .as_ref()
            .map(|handle| handle.id.as_str()),
        Some("resp_2")
    );
    messages.push(Message::assistant(second.message));
    messages.push(Message::user("third"));
    messages[1] = Message::user("changed first");

    let third = client.complete(&messages, &[], &sink).await.unwrap();
    messages.push(Message::assistant(third.message));
    messages.push(Message::user("fourth"));
    profile.api_key = "other-project-key".to_string();
    ModelClient::from_profile(&profile)
        .unwrap()
        .complete(&messages, &[], &sink)
        .await
        .unwrap();

    let requests = server.requests();
    assert_eq!(requests.len(), 4);
    assert_eq!(requests[0]["store"], true);
    assert!(requests[0].get("previous_response_id").is_none());
    assert_eq!(requests[0]["input"].as_array().unwrap().len(), 2);
    assert_eq!(requests[1]["store"], true);
    assert_eq!(requests[1]["previous_response_id"], "resp_1");
    assert_eq!(requests[1]["input"].as_array().unwrap().len(), 1);
    assert_eq!(requests[1]["input"][0]["content"][0]["text"], "second");
    assert!(requests[2].get("previous_response_id").is_none());
    assert_eq!(requests[2]["input"].as_array().unwrap().len(), 6);
    assert!(requests[3].get("previous_response_id").is_none());
    assert_eq!(
        requests[3]["input"].as_array().unwrap().len(),
        messages.len()
    );
}

#[tokio::test]
async fn responses_file_search_configuration_changes_force_full_stored_replay() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_server = Arc::clone(&calls);
    let server = MockServer::new(move |_, _| {
        let call = calls_for_server.fetch_add(1, Ordering::SeqCst) + 1;
        MockResponse::json(
            200,
            &json!({
                "id": format!("resp_{call}"),
                "status": "completed",
                "output": [{
                    "type": "message",
                    "id": format!("msg_{call}"),
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type":"output_text","text":"answer","annotations":[]}]
                }],
                "usage": {"input_tokens":2,"output_tokens":1,"total_tokens":3}
            })
            .to_string(),
        )
    });
    let mut profile = responses_profile(&server, false);
    profile.responses_stored_continuation = true;
    profile.responses_file_search = Some(ResponsesFileSearchConfig {
        vector_store_ids: vec!["vs_a".into()],
        filters: None,
        max_num_results: None,
        ranking_options: None,
    });
    let sink = RecordingSink::default();
    let mut messages = vec![Message::system("prompt"), Message::user("first")];
    let first = ModelClient::from_profile(&profile)
        .unwrap()
        .complete(&messages, &[], &sink)
        .await
        .unwrap();
    messages.push(Message::assistant(first.message));
    messages.push(Message::user("second"));

    profile.responses_file_search = Some(ResponsesFileSearchConfig {
        vector_store_ids: vec!["vs_a".into()],
        filters: None,
        max_num_results: None,
        ranking_options: Some(ResponsesFileSearchRankingOptions {
            hybrid_search: None,
            ranker: Some(ResponsesFileSearchRanker::Auto),
            score_threshold: Some(0.4),
        }),
    });
    let second_input_len = messages.len();
    let second = ModelClient::from_profile(&profile)
        .unwrap()
        .complete(&messages, &[], &sink)
        .await
        .unwrap();
    messages.push(Message::assistant(second.message));
    messages.push(Message::user("third"));
    profile.responses_file_search.as_mut().unwrap().filters = Some(
        ResponsesFileSearchFilter::Comparison(ResponsesFileSearchComparisonFilter {
            key: "department".into(),
            operator: ResponsesFileSearchComparisonOperator::Eq,
            value: ResponsesFileSearchFilterValue::String("engineering".into()),
        }),
    );
    let third_input_len = messages.len();
    let third = ModelClient::from_profile(&profile)
        .unwrap()
        .complete(&messages, &[], &sink)
        .await
        .unwrap();
    messages.push(Message::assistant(third.message));
    messages.push(Message::user("fourth"));
    profile.responses_code_interpreter = true;
    ModelClient::from_profile(&profile)
        .unwrap()
        .complete(&messages, &[], &sink)
        .await
        .unwrap();

    let requests = server.requests();
    assert_eq!(requests.len(), 4);
    assert!(requests[1].get("previous_response_id").is_none());
    assert_eq!(
        requests[1]["input"].as_array().unwrap().len(),
        second_input_len
    );
    assert_eq!(requests[1]["tools"][0]["vector_store_ids"], json!(["vs_a"]));
    assert_eq!(
        requests[1]["tools"][0]["ranking_options"],
        json!({"ranker":"auto","score_threshold":0.4})
    );
    assert!(requests[2].get("previous_response_id").is_none());
    assert_eq!(
        requests[2]["input"].as_array().unwrap().len(),
        third_input_len
    );
    assert_eq!(
        requests[2]["tools"][0]["filters"],
        json!({"type":"eq","key":"department","value":"engineering"})
    );
    assert!(requests[3].get("previous_response_id").is_none());
    assert_eq!(
        requests[3]["input"].as_array().unwrap().len(),
        messages.len()
    );
    assert_eq!(
        requests[3]["tools"][0],
        json!({
            "type":"code_interpreter",
            "container": {
                "type":"auto",
                "network_policy": {"type":"disabled"}
            }
        })
    );
}

#[tokio::test]
async fn responses_stored_stream_requires_and_captures_one_response_id() {
    let server = MockServer::new(|_, _| {
        MockResponse::sse(&[
            r#"{"type":"response.created","response":{"id":"resp_stream","status":"in_progress"}}"#,
            r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}}"#,
            r#"{"type":"response.completed","response":{"id":"resp_stream","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}"#,
        ])
    });
    let mut profile = responses_profile(&server, true);
    profile.responses_stored_continuation = true;
    let completion = ModelClient::from_profile(&profile)
        .unwrap()
        .complete(
            &[Message::system("prompt"), Message::user("hello")],
            &[],
            &RecordingSink::default(),
        )
        .await
        .unwrap();

    assert_eq!(
        completion
            .message
            .responses_continuation
            .as_ref()
            .map(|handle| handle.id.as_str()),
        Some("resp_stream")
    );
}

#[tokio::test]
async fn responses_background_creates_once_then_polls_to_completion() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_server = Arc::clone(&calls);
    let server = MockServer::new(move |path, body| {
        let call = calls_for_server.fetch_add(1, Ordering::SeqCst);
        match call {
            0 => {
                assert_eq!(path, "/v1/responses");
                assert_eq!(body["background"], true);
                assert_eq!(body["store"], true);
                assert_eq!(body["stream"], false);
                assert_eq!(
                    body["include"],
                    json!(["reasoning.encrypted_content", "file_search_call.results"])
                );
                MockResponse::json(
                    200,
                    &json!({"id":"resp_bg","status":"queued","output":[]}).to_string(),
                )
            }
            1 => {
                assert_eq!(
                    path,
                    "/v1/responses/resp_bg?include=reasoning.encrypted_content&include=file_search_call.results"
                );
                assert!(body.is_null());
                MockResponse::json(500, r#"{"error":{"message":"poll later"}}"#)
            }
            2 => {
                assert_eq!(
                    path,
                    "/v1/responses/resp_bg?include=reasoning.encrypted_content&include=file_search_call.results"
                );
                assert!(body.is_null());
                let mut response =
                    MockResponse::json(200, r#"{"id":"resp_bg","status":"in_progress"}"#);
                response.declared_length = Some(response.body.len() + 10);
                response
            }
            3 => {
                assert_eq!(
                    path,
                    "/v1/responses/resp_bg?include=reasoning.encrypted_content&include=file_search_call.results"
                );
                assert!(body.is_null());
                MockResponse::json(
                    200,
                    &json!({
                        "id": "resp_bg",
                        "status": "completed",
                        "output": [{
                            "type": "message",
                            "id": "msg_bg",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type":"output_text","text":"background done","annotations":[]}]
                        }],
                        "usage": {"input_tokens":2,"output_tokens":2,"total_tokens":4}
                    })
                    .to_string(),
                )
            }
            _ => panic!("unexpected background request {call}"),
        }
    });
    let mut profile = responses_profile(&server, false);
    profile.responses_stored_continuation = true;
    profile.responses_background = true;
    profile.responses_file_search = Some(ResponsesFileSearchConfig {
        vector_store_ids: vec!["vs_docs".into()],
        filters: None,
        max_num_results: None,
        ranking_options: None,
    });

    let completion = ModelClient::from_profile(&profile)
        .unwrap()
        .complete(
            &[Message::system("prompt"), Message::user("long task")],
            &[],
            &RecordingSink::default(),
        )
        .await
        .unwrap();

    assert_eq!(
        completion.message.content.as_deref(),
        Some("background done")
    );
    assert_eq!(
        completion
            .message
            .responses_continuation
            .as_ref()
            .map(|handle| handle.id.as_str()),
        Some("resp_bg")
    );
    assert_eq!(server.request_count(), 4);
}

#[tokio::test]
async fn responses_background_stream_resumes_after_known_sequence() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_server = Arc::clone(&calls);
    let server = MockServer::new(move |path, body| {
        let call = calls_for_server.fetch_add(1, Ordering::SeqCst);
        match call {
            0 => {
                assert_eq!(path, "/v1/responses");
                assert_eq!(body["background"], true);
                assert_eq!(body["store"], true);
                assert_eq!(body["stream"], true);
                MockResponse {
                    status: 200,
                    content_type: "text/event-stream".into(),
                    declared_length: None,
                    body: concat!(
                        "data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_stream_bg\",\"status\":\"queued\",\"output\":[]}}\n\n",
                        "data: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_bg\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
                        "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":2,\"output_index\":0,\"delta\":\"part \"}\n\n",
                        "data: {\"type\":\"response.output_text.delta\""
                    )
                    .into(),
                }
            }
            1 => {
                assert_eq!(
                    path,
                    "/v1/responses/resp_stream_bg?include=reasoning.encrypted_content&stream=true&starting_after=2"
                );
                assert!(body.is_null());
                MockResponse::sse(&[
                    r#"{"type":"response.output_text.delta","sequence_number":3,"output_index":0,"delta":"done"}"#,
                    r#"{"type":"response.output_item.done","sequence_number":4,"output_index":0,"item":{"type":"message","id":"msg_bg","role":"assistant","status":"completed","content":[{"type":"output_text","text":"part done","annotations":[]}]}}"#,
                    r#"{"type":"response.completed","sequence_number":5,"response":{"id":"resp_stream_bg","status":"completed","output":[{"type":"message","id":"msg_bg","role":"assistant","status":"completed","content":[{"type":"output_text","text":"part done","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":2,"total_tokens":4}}}"#,
                ])
            }
            _ => panic!("unexpected background stream request {call}"),
        }
    });
    let mut profile = responses_profile(&server, true);
    profile.responses_stored_continuation = true;
    profile.responses_background = true;
    let completion = ModelClient::from_profile(&profile)
        .unwrap()
        .complete(
            &[Message::system("prompt"), Message::user("long task")],
            &[],
            &RecordingSink::default(),
        )
        .await
        .unwrap();

    assert_eq!(completion.message.content.as_deref(), Some("part done"));
    assert_eq!(
        completion
            .message
            .responses_continuation
            .as_ref()
            .map(|handle| handle.id.as_str()),
        Some("resp_stream_bg")
    );
    assert_eq!(server.request_count(), 2);
}

#[tokio::test]
async fn responses_background_stream_does_not_retry_before_response_id() {
    let server = MockServer::new(|path, body| {
        assert_eq!(path, "/v1/responses");
        assert_eq!(body["background"], true);
        assert_eq!(body["stream"], true);
        let mut response = MockResponse {
            status: 200,
            content_type: "text/event-stream".into(),
            declared_length: None,
            body: "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":0,\"output_index\":0,\"delta\":\"unknown\"}\n\n".into(),
        };
        response.declared_length = Some(response.body.len() + 10);
        response
    });
    let mut profile = responses_profile(&server, true);
    profile.responses_stored_continuation = true;
    profile.responses_background = true;

    let error = ModelClient::from_profile(&profile)
        .unwrap()
        .complete(
            &[Message::user("long task")],
            &[],
            &RecordingSink::default(),
        )
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("ambiguous remote outcome"),
        "{error:#}"
    );
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn responses_background_local_cancellation_cancels_remote_work() {
    let paths = Arc::new(Mutex::new(Vec::new()));
    let paths_for_server = Arc::clone(&paths);
    let server = MockServer::new(move |path, body| {
        paths_for_server.lock().unwrap().push(path.to_string());
        match path {
            "/v1/responses" => {
                assert_eq!(body["background"], true);
                assert_eq!(body["stream"], true);
                let mut response = MockResponse {
                    status: 200,
                    content_type: "text/event-stream".into(),
                    declared_length: None,
                    body: concat!(
                        "data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_cancel\",\"status\":\"queued\",\"output\":[]}}\n\n",
                        "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"output_index\":0,\"delta\":\"started\"}\n\n"
                    )
                    .into(),
                };
                response.declared_length = Some(response.body.len() + 10);
                response
            }
            "/v1/responses/resp_cancel/cancel" => {
                assert!(body.is_null());
                MockResponse::json(
                    200,
                    &json!({"id":"resp_cancel","status":"cancelled","output":[]}).to_string(),
                )
            }
            _ => panic!("unexpected background cancellation request {path}"),
        }
    });
    let mut profile = responses_profile(&server, true);
    profile.responses_stored_continuation = true;
    profile.responses_background = true;
    let client = ModelClient::from_profile(&profile).unwrap();
    let sink = Arc::new(RecordingSink::default());
    let sink_for_task = Arc::clone(&sink);
    let task = tokio::spawn(async move {
        client
            .complete(&[Message::user("long task")], &[], sink_for_task.as_ref())
            .await
    });

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if sink.events().iter().any(|event| event == "text:started") {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if paths
                .lock()
                .unwrap()
                .iter()
                .any(|path| path == "/v1/responses/resp_cancel/cancel")
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        paths.lock().unwrap().as_slice(),
        ["/v1/responses", "/v1/responses/resp_cancel/cancel"]
    );
}

#[tokio::test]
async fn responses_background_poll_failure_cancels_remote_work() {
    let paths = Arc::new(Mutex::new(Vec::new()));
    let paths_for_server = Arc::clone(&paths);
    let server = MockServer::new(move |path, body| {
        paths_for_server.lock().unwrap().push(path.to_string());
        match path {
            "/v1/responses" => {
                assert_eq!(body["background"], true);
                MockResponse::json(
                    200,
                    &json!({"id":"resp_poll_error","status":"queued","output":[]}).to_string(),
                )
            }
            "/v1/responses/resp_poll_error?include=reasoning.encrypted_content" => {
                assert!(body.is_null());
                MockResponse::json(400, r#"{"error":{"message":"invalid response"}}"#)
            }
            "/v1/responses/resp_poll_error/cancel" => {
                assert!(body.is_null());
                MockResponse::json(
                    200,
                    &json!({"id":"resp_poll_error","status":"cancelled","output":[]}).to_string(),
                )
            }
            _ => panic!("unexpected background poll failure request {path}"),
        }
    });
    let mut profile = responses_profile(&server, false);
    profile.responses_stored_continuation = true;
    profile.responses_background = true;

    let error = ModelClient::from_profile(&profile)
        .unwrap()
        .complete(
            &[Message::user("long task")],
            &[],
            &RecordingSink::default(),
        )
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("retrieval failed with HTTP 400"),
        "{error:#}"
    );
    assert!(
        error
            .to_string()
            .contains("remote cancellation was accepted"),
        "{error:#}"
    );
    assert_eq!(
        paths.lock().unwrap().as_slice(),
        [
            "/v1/responses",
            "/v1/responses/resp_poll_error?include=reasoning.encrypted_content",
            "/v1/responses/resp_poll_error/cancel"
        ]
    );
}

#[tokio::test]
async fn responses_background_create_failure_is_not_retried() {
    let server = MockServer::new(|path, body| {
        assert_eq!(path, "/v1/responses");
        assert_eq!(body["background"], true);
        MockResponse::json(500, &"x".repeat(70_000))
    });
    let mut profile = responses_profile(&server, false);
    profile.responses_stored_continuation = true;
    profile.responses_background = true;

    let error = ModelClient::from_profile(&profile)
        .unwrap()
        .complete(
            &[Message::system("prompt"), Message::user("long task")],
            &[],
            &RecordingSink::default(),
        )
        .await
        .unwrap_err();

    assert!(error.to_string().contains("was not retried"));
    assert!(error
        .to_string()
        .contains("error body exceeded 65536 bytes"));
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn responses_stored_result_without_id_fails_closed() {
    let server = MockServer::new(|_, _| {
        MockResponse::json(
            200,
            &json!({
                "status": "completed",
                "output": [{
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "done", "annotations": []}]
                }]
            })
            .to_string(),
        )
    });
    let mut profile = responses_profile(&server, false);
    profile.responses_stored_continuation = true;
    let error = ModelClient::from_profile(&profile)
        .unwrap()
        .complete(
            &[Message::system("prompt"), Message::user("hello")],
            &[],
            &RecordingSink::default(),
        )
        .await
        .unwrap_err();

    assert!(error
        .to_string()
        .contains("stored Responses result is missing a response id"));
}
