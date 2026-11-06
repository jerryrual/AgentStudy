mod common;

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use common::*;
use icode::agent::{Agent, AgentKind, AgentOptions, EventSink};
use icode::compaction::ContextProjection;
use icode::interaction::{AllowAll, ApprovalDecision, DenyAll, NoAsker, ScriptedApprover};
use icode::model::{AssistantMessage, Message, ModelClient};
use icode::tools::ToolRegistry;

struct FailedCheckpointSink {
    entries: AtomicUsize,
    checkpoints: AtomicUsize,
}

impl EventSink for FailedCheckpointSink {
    fn enter_phase_four_round(&self) -> anyhow::Result<bool> {
        self.entries.fetch_add(1, Ordering::SeqCst);
        Ok(true)
    }

    fn checkpoint_phase_four_side_call(&self, _estimated_input: u64) -> anyhow::Result<bool> {
        self.checkpoints.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("simulated checkpoint failure")
    }
}

struct SideCallQuotaSink {
    allowed_attempts: usize,
    charged_attempts: AtomicUsize,
    events: std::sync::Mutex<Vec<String>>,
    projections: std::sync::Mutex<Vec<ContextProjection>>,
}

impl SideCallQuotaSink {
    fn new(allowed_attempts: usize) -> Self {
        Self {
            allowed_attempts,
            charged_attempts: AtomicUsize::new(0),
            events: std::sync::Mutex::new(Vec::new()),
            projections: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl EventSink for SideCallQuotaSink {
    fn turn_started(&self) {
        self.charged_attempts.store(0, Ordering::SeqCst);
    }

    fn charge_phase_four_side_call(&self, _estimated_input: u64) -> bool {
        self.charged_attempts.fetch_add(1, Ordering::SeqCst) < self.allowed_attempts
    }

    fn persist_context_projection(
        &self,
        _source: &[Message],
        projection: &ContextProjection,
    ) -> anyhow::Result<bool> {
        self.projections.lock().unwrap().push(projection.clone());
        Ok(true)
    }

    fn info(&self, message: &str) {
        self.events.lock().unwrap().push(message.to_string());
    }
}

fn agent_options(cwd: PathBuf, max_rounds: usize) -> AgentOptions {
    AgentOptions {
        kind: AgentKind::Code,
        cwd,
        max_rounds,
        tool_timeout: Duration::from_secs(10),
    }
}

fn build_agent(
    server: &MockServer,
    cwd: &std::path::Path,
    approver: Arc<dyn icode::interaction::Approver>,
) -> Agent {
    let client = ModelClient::from_profile(&mock_profile(server.base_url())).unwrap();
    Agent::new(
        Arc::new(client),
        Arc::new(ToolRegistry::builtin()),
        approver,
        Arc::new(NoAsker),
        agent_options(cwd.to_path_buf(), 6),
    )
}

fn build_context_limited_agent(server: &MockServer, cwd: &std::path::Path) -> Agent {
    let mut profile = mock_profile(server.base_url());
    profile.context_window = Some(140_000);
    profile.max_tokens = Some(10_000);
    let client = ModelClient::from_profile(&profile).unwrap();
    Agent::new(
        Arc::new(client),
        Arc::new(ToolRegistry::builtin()),
        Arc::new(AllowAll),
        Arc::new(NoAsker),
        agent_options(cwd.to_path_buf(), 6),
    )
}

fn context_overflow_response() -> MockResponse {
    MockResponse::json(
        400,
        r#"{"error":{"code":"context_length_exceeded","message":"This model's maximum context length is 140000 tokens."}}"#,
    )
}

#[tokio::test]
async fn streams_an_answer_and_reports_usage() {
    let server = MockServer::new(|_path, _body| {
        MockResponse::sse(&[
            &text_chunk("Hello "),
            &text_chunk("world"),
            &done_chunk("stop"),
        ])
    });
    let directory = tempfile::tempdir().unwrap();
    let agent = build_agent(&server, directory.path(), Arc::new(AllowAll));
    let sink = RecordingSink::default();
    let mut messages = vec![Message::user("say hello")];

    let turn = agent.run(&mut messages, &sink).await.unwrap();
    assert_eq!(turn.text, "Hello world");
    assert_eq!(turn.usage.total_tokens, 15);
    assert_eq!(turn.rounds, 1);
    assert_eq!(messages.len(), 3, "system, user and assistant");
    assert!(matches!(messages[0], Message::System { .. }));
    assert!(sink.events().iter().any(|event| event == "text:Hello "));
    // The request carries the system prompt and the user message.
    let request = &server.requests()[0];
    assert_eq!(request["model"], "mock-model");
    assert_eq!(request["messages"].as_array().unwrap().len(), 2);
    assert!(request["tools"].as_array().unwrap().len() >= 8);
}

#[tokio::test]
async fn executes_tool_calls_and_feeds_results_back() {
    let server = MockServer::new(|_path, body| {
        let has_tool_result = body["messages"]
            .as_array()
            .map(|messages| messages.iter().any(|message| message["role"] == "tool"))
            .unwrap_or(false);
        if has_tool_result {
            MockResponse::sse(&[&text_chunk("The file says hello"), &done_chunk("stop")])
        } else {
            MockResponse::sse(&[
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":"{\"path\":\"hello.txt\"}"}}]},"finish_reason":"tool_calls"}]}"#,
                &done_chunk("tool_calls"),
            ])
        }
    });
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("hello.txt"), "hello from file\n").unwrap();
    let approver = Arc::new(ScriptedApprover::new([]));
    let agent = build_agent(&server, directory.path(), approver.clone());
    let sink = RecordingSink::default();
    let mut messages = vec![Message::user("read hello.txt")];

    let turn = agent.run(&mut messages, &sink).await.unwrap();
    assert_eq!(turn.text, "The file says hello");
    assert_eq!(turn.tool_calls, 1);
    assert_eq!(turn.rounds, 2);

    let events = sink.events();
    assert!(events
        .iter()
        .any(|event| event.starts_with("tool_start:read_file")));
    assert!(events
        .iter()
        .any(|event| event.starts_with("tool_done:read_file:true")));

    let tool_message = messages
        .iter()
        .find(|message| matches!(message, Message::Tool { .. }))
        .expect("a tool result message");
    if let Message::Tool { content, .. } = tool_message {
        assert!(content.contains("hello from file"), "{content}");
    }
    assert!(
        approver.requests.lock().unwrap().is_empty(),
        "read-only tools are not approved"
    );
}

#[tokio::test]
async fn parallel_read_results_are_sent_back_in_provider_order() {
    let server = MockServer::new(|_path, body| {
        let tool_results = body["messages"]
            .as_array()
            .map(|messages| {
                messages
                    .iter()
                    .filter(|message| message["role"] == "tool")
                    .count()
            })
            .unwrap_or_default();
        if tool_results == 2 {
            MockResponse::sse(&[&text_chunk("both read"), &done_chunk("stop")])
        } else {
            MockResponse::sse(&[
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}},{"index":1,"id":"call_b","function":{"name":"read_file","arguments":"{\"path\":\"b.txt\"}"}}]},"finish_reason":"tool_calls"}]}"#,
                &done_chunk("tool_calls"),
            ])
        }
    });
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("a.txt"), "alpha\n").unwrap();
    std::fs::write(directory.path().join("b.txt"), "beta\n").unwrap();
    let agent = build_agent(&server, directory.path(), Arc::new(AllowAll));
    let sink = RecordingSink::default();
    let mut messages = vec![Message::user("read both")];

    let turn = agent.run(&mut messages, &sink).await.unwrap();
    assert_eq!(turn.text, "both read");
    assert_eq!(turn.tool_calls, 2);

    let requests = server.requests();
    let tool_ids: Vec<_> = requests[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "tool")
        .map(|message| message["tool_call_id"].as_str().unwrap())
        .collect();
    assert_eq!(tool_ids, ["call_a", "call_b"]);
}

#[tokio::test]
async fn mutating_tools_require_approval() {
    let server = MockServer::new(|_path, body| {
        let has_tool_result = body["messages"]
            .as_array()
            .map(|messages| messages.iter().any(|message| message["role"] == "tool"))
            .unwrap_or(false);
        if has_tool_result {
            MockResponse::sse(&[&text_chunk("done"), &done_chunk("stop")])
        } else {
            MockResponse::sse(&[
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_2","function":{"name":"bash","arguments":"{\"command\":\"echo approved\"}"}}]},"finish_reason":"tool_calls"}]}"#,
                &done_chunk("tool_calls"),
            ])
        }
    });
    let directory = tempfile::tempdir().unwrap();
    let approver = Arc::new(ScriptedApprover::new([ApprovalDecision::AllowOnce]));
    let agent = build_agent(&server, directory.path(), approver.clone());
    let sink = RecordingSink::default();
    let mut messages = vec![Message::user("echo approved")];

    let turn = agent.run(&mut messages, &sink).await.unwrap();
    assert_eq!(turn.text, "done");
    let requests = approver.requests.lock().unwrap().clone();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].tool, "bash");
    assert!(requests[0].summary.contains("echo approved"));
    assert!(sink.events().iter().any(|event| event.contains("approved")));
}

#[tokio::test]
async fn denied_tool_call_is_reported_to_the_model() {
    let server = MockServer::new(|_path, body| {
        let has_tool_result = body["messages"]
            .as_array()
            .map(|messages| messages.iter().any(|message| message["role"] == "tool"))
            .unwrap_or(false);
        if has_tool_result {
            MockResponse::sse(&[&text_chunk("understood"), &done_chunk("stop")])
        } else {
            MockResponse::sse(&[
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_3","function":{"name":"write_file","arguments":"{\"path\":\"blocked.txt\",\"content\":\"nope\"}"}}]},"finish_reason":"tool_calls"}]}"#,
                &done_chunk("tool_calls"),
            ])
        }
    });
    let directory = tempfile::tempdir().unwrap();
    let agent = build_agent(&server, directory.path(), Arc::new(DenyAll));
    let sink = RecordingSink::default();
    let mut messages = vec![Message::user("write a file")];

    let turn = agent.run(&mut messages, &sink).await.unwrap();
    assert_eq!(turn.text, "understood");
    assert!(!directory.path().join("blocked.txt").exists());
    let tool_message = messages
        .iter()
        .find(|message| matches!(message, Message::Tool { .. }))
        .unwrap();
    if let Message::Tool { content, .. } = tool_message {
        assert!(content.to_lowercase().contains("denied"), "{content}");
    }
}

#[tokio::test]
async fn retries_transient_failures() {
    // Make retries fast for the test.
    std::env::set_var("ICODE_RETRY_BASE_MS", "10");
    let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let attempts_for_handler = attempts.clone();
    let server = MockServer::new(move |_path, _body| {
        let attempt = attempts_for_handler.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if attempt == 0 {
            MockResponse::json(429, r#"{"error":{"message":"rate limited"}}"#)
        } else {
            MockResponse::sse(&[&text_chunk("recovered"), &done_chunk("stop")])
        }
    });
    let directory = tempfile::tempdir().unwrap();
    let agent = build_agent(&server, directory.path(), Arc::new(AllowAll));
    let sink = RecordingSink::default();
    let mut messages = vec![Message::user("hello")];

    let turn = agent.run(&mut messages, &sink).await.unwrap();
    assert_eq!(turn.text, "recovered");
    assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert!(sink.events().iter().any(|event| event.contains("retrying")));
}

#[tokio::test]
async fn phase_four_checkpoint_failure_prevents_side_call_provider_io() {
    let side_requests = Arc::new(AtomicUsize::new(0));
    let observed = side_requests.clone();
    let server = MockServer::new(move |_path, body| {
        if body["stream"] == false {
            observed.fetch_add(1, Ordering::SeqCst);
        }
        MockResponse::sse(&[&text_chunk("continued raw"), &done_chunk("stop")])
    });
    let directory = tempfile::tempdir().unwrap();
    let mut profile = mock_profile(server.base_url());
    profile.context_window = Some(30_000);
    profile.max_tokens = Some(3_000);
    let agent = Agent::new(
        Arc::new(ModelClient::from_profile(&profile).unwrap()),
        Arc::new(ToolRegistry::builtin()),
        Arc::new(AllowAll),
        Arc::new(NoAsker),
        agent_options(directory.path().to_path_buf(), 6),
    );
    let mut messages = vec![
        Message::user("inspect the large result and continue"),
        Message::assistant(AssistantMessage {
            tool_calls: vec![icode::model::ToolCall {
                id: "call_large".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"large.txt"}"#.into(),
            }],
            ..AssistantMessage::default()
        }),
        Message::tool("call_large", "read_file", "result ".repeat(2_500)),
    ];

    let sink = FailedCheckpointSink {
        entries: AtomicUsize::new(0),
        checkpoints: AtomicUsize::new(0),
    };
    let turn = agent.run(&mut messages, &sink).await.unwrap();

    assert_eq!(sink.entries.load(Ordering::SeqCst), 1);
    assert_eq!(sink.checkpoints.load(Ordering::SeqCst), 1);
    assert_eq!(turn.text, "continued raw");
    assert_eq!(side_requests.load(Ordering::SeqCst), 0);
    assert_eq!(server.request_count(), 1, "only the raw main request ran");
}

#[tokio::test]
async fn phase_four_quota_charges_each_physical_retry_before_provider_io() {
    std::env::set_var("ICODE_RETRY_BASE_MS", "1");
    let server = MockServer::new(move |_path, body| {
        if body["stream"] == false {
            return MockResponse::json(429, r#"{"error":{"message":"rate limited"}}"#);
        }
        MockResponse::sse(&[&text_chunk("continued raw"), &done_chunk("stop")])
    });
    let directory = tempfile::tempdir().unwrap();
    let mut profile = mock_profile(server.base_url());
    profile.context_window = Some(30_000);
    profile.max_tokens = Some(3_000);
    let agent = Agent::new(
        Arc::new(ModelClient::from_profile(&profile).unwrap()),
        Arc::new(ToolRegistry::builtin()),
        Arc::new(AllowAll),
        Arc::new(NoAsker),
        agent_options(directory.path().to_path_buf(), 6),
    );
    let sink = SideCallQuotaSink::new(1);
    let mut messages = vec![
        Message::user("inspect the large result and continue"),
        Message::assistant(AssistantMessage {
            tool_calls: vec![icode::model::ToolCall {
                id: "call_large".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"large.txt"}"#.into(),
            }],
            ..AssistantMessage::default()
        }),
        Message::tool("call_large", "read_file", "result ".repeat(2_500)),
    ];

    let turn = agent.run(&mut messages, &sink).await.unwrap();

    assert_eq!(turn.text, "continued raw");
    assert_eq!(server.request_count(), 2, "denied retry never reached HTTP");
    assert_eq!(sink.charged_attempts.load(Ordering::SeqCst), 2);
    assert!(sink.projections.lock().unwrap().is_empty());
    assert!(sink
        .events
        .lock()
        .unwrap()
        .iter()
        .any(|event| event.contains("side-call input-token budget exhausted")));
}

#[tokio::test]
async fn generates_last_words_before_retrying_a_large_current_tool_turn() {
    let note_body = "concrete progress detail ".repeat(30);
    let note = format!("## Task\n{note_body}\n## Progress\n{note_body}\n## Next\n{note_body}");
    let note_for_server = note.clone();
    let server = MockServer::new(move |_path, body| {
        if body["stream"] == false {
            assert!(body.get("tools").is_none());
            assert!(body.get("tool_choice").is_none());
            return MockResponse::json(
                200,
                &serde_json::json!({
                    "choices": [{
                        "message": {"role": "assistant", "content": note_for_server},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
                })
                .to_string(),
            );
        }
        MockResponse::sse(&[&text_chunk("continued"), &done_chunk("stop")])
    });
    let directory = tempfile::tempdir().unwrap();
    let mut profile = mock_profile(server.base_url());
    profile.context_window = Some(30_000);
    profile.max_tokens = Some(3_000);
    let agent = Agent::new(
        Arc::new(ModelClient::from_profile(&profile).unwrap()),
        Arc::new(ToolRegistry::builtin()),
        Arc::new(AllowAll),
        Arc::new(NoAsker),
        agent_options(directory.path().to_path_buf(), 6),
    );
    let sink = RecordingSink::default();
    let mut messages = vec![
        Message::user("inspect the large result and continue"),
        Message::assistant(AssistantMessage {
            tool_calls: vec![icode::model::ToolCall {
                id: "call_large".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"large.txt"}"#.into(),
            }],
            ..AssistantMessage::default()
        }),
        Message::tool("call_large", "read_file", "result ".repeat(2_500)),
    ];

    let turn = agent.run(&mut messages, &sink).await.unwrap();

    assert_eq!(
        server.request_count(),
        2,
        "one side call and one agent call"
    );
    assert_eq!(turn.rounds, 1);
    assert_eq!(turn.usage.total_tokens, 25);
    assert_eq!(sink.projections.lock().unwrap().len(), 1);
    assert!(matches!(messages[2], Message::LastWords { .. }));
    assert!(matches!(
        messages.last(),
        Some(Message::Assistant { message }) if message.content.as_deref() == Some("continued")
    ));
    let requests = server.requests();
    assert_eq!(requests[0]["stream"], false);
    assert_eq!(requests[1]["stream"], true);
    assert!(requests[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|message| {
            message["content"]
                .as_str()
                .is_some_and(|content| content.contains("[LAST_WORDS]"))
        }));
}

#[tokio::test]
async fn provider_overflow_phase_four_does_not_recurse_after_retry() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls_for_server = calls.clone();
    let note_body = "concrete progress detail ".repeat(30);
    let note = format!("## Task\n{note_body}\n## Progress\n{note_body}\n## Next\n{note_body}");
    let server = MockServer::new(move |_path, body| {
        match calls_for_server.fetch_add(1, std::sync::atomic::Ordering::SeqCst) {
            0 => {
                assert_eq!(body["stream"], true);
                context_overflow_response()
            }
            1 => {
                assert_eq!(body["stream"], false);
                MockResponse::json(
                    200,
                    &serde_json::json!({
                        "choices": [{
                            "message": {"role": "assistant", "content": note},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
                    })
                    .to_string(),
                )
            }
            _ => context_overflow_response(),
        }
    });
    let directory = tempfile::tempdir().unwrap();
    let agent = build_context_limited_agent(&server, directory.path());
    let sink = RecordingSink::default();
    let mut messages = vec![
        Message::user("continue after inspecting the result"),
        Message::assistant(AssistantMessage {
            tool_calls: vec![icode::model::ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            }],
            ..AssistantMessage::default()
        }),
        Message::tool("call_1", "read_file", "current result ".repeat(500)),
    ];

    let error = agent.run(&mut messages, &sink).await.unwrap_err();

    assert!(
        error.to_string().contains("context_length_exceeded"),
        "{error:#}"
    );
    assert_eq!(server.request_count(), 3);
    assert_eq!(
        server
            .requests()
            .iter()
            .map(|request| request["stream"].as_bool().unwrap())
            .collect::<Vec<_>>(),
        vec![true, false, true]
    );
    assert_eq!(sink.projections.lock().unwrap().len(), 1);
    assert!(messages
        .iter()
        .any(|message| matches!(message, Message::LastWords { .. })));
}

#[tokio::test]
async fn stale_last_words_never_drops_new_current_tool_history() {
    let note_body = "prior concrete progress detail ".repeat(30);
    let old_note = format!("## Task\n{note_body}\n## Progress\n{note_body}\n## Next\n{note_body}");
    let stale = old_note.replace("##", "###").replace("detail ", "detail  ");
    let server = MockServer::new(move |_path, body| {
        if body["stream"] == true {
            return MockResponse::sse(&[&text_chunk("continued safely"), &done_chunk("stop")]);
        }
        MockResponse::json(
            200,
            &serde_json::json!({
                "choices": [{
                    "message": {"role": "assistant", "content": stale},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
            })
            .to_string(),
        )
    });
    let directory = tempfile::tempdir().unwrap();
    let mut profile = mock_profile(server.base_url());
    profile.context_window = Some(30_000);
    profile.max_tokens = Some(3_000);
    let agent = Agent::new(
        Arc::new(ModelClient::from_profile(&profile).unwrap()),
        Arc::new(ToolRegistry::builtin()),
        Arc::new(AllowAll),
        Arc::new(NoAsker),
        agent_options(directory.path().to_path_buf(), 6),
    );
    let sink = RecordingSink::default();
    let mut messages = vec![
        Message::user("inspect the large result and continue"),
        Message::LastWords {
            content: format!(
                "<system-reminder>\n[LAST_WORDS] prior progress\n\n{old_note}\n</system-reminder>"
            ),
        },
        Message::assistant(AssistantMessage {
            tool_calls: vec![icode::model::ToolCall {
                id: "call_large".into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            }],
            ..AssistantMessage::default()
        }),
        Message::tool("call_large", "read_file", "result ".repeat(2_500)),
    ];

    let turn = agent.run(&mut messages, &sink).await.unwrap();

    assert_eq!(turn.text, "continued safely");
    assert_eq!(turn.rounds, 1);
    assert_eq!(turn.usage.total_tokens, 35);
    assert_eq!(
        server.request_count(),
        3,
        "two side calls and one normal request"
    );
    assert!(sink.projections.lock().unwrap().is_empty());
    assert_eq!(
        messages
            .iter()
            .filter(|message| matches!(message, Message::LastWords { .. }))
            .count(),
        1
    );
    assert!(messages.iter().any(
        |message| matches!(message, Message::Tool { content, .. } if content.starts_with("result "))
    ));
}

#[tokio::test]
async fn overflow_falls_back_to_historical_compression_when_last_words_is_invalid() {
    let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let attempts_for_handler = attempts.clone();
    let server = MockServer::new(move |_path, body| {
        match attempts_for_handler.fetch_add(1, std::sync::atomic::Ordering::SeqCst) {
            0 => context_overflow_response(),
            1 | 2 => {
                assert_eq!(body["stream"], false);
                MockResponse::json(
                    200,
                    &serde_json::json!({
                        "choices": [{
                            "message": {"role": "assistant", "content": "too short"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
                    })
                    .to_string(),
                )
            }
            _ => MockResponse::sse(&[&text_chunk("recovered"), &done_chunk("stop")]),
        }
    });
    let directory = tempfile::tempdir().unwrap();
    let agent = build_context_limited_agent(&server, directory.path());
    let sink = RecordingSink::default();
    let prior_user = format!(
        "prior completed turn {}",
        "historical detail ".repeat(4_500)
    );
    let current_user = "keep this current request byte-for-byte";
    let mut messages = vec![
        Message::user(prior_user.clone()),
        Message::assistant(AssistantMessage::text("prior answer")),
        Message::user(current_user),
        Message::assistant(AssistantMessage {
            tool_calls: vec![icode::model::ToolCall {
                id: "call_current".into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            }],
            ..AssistantMessage::default()
        }),
        Message::tool("call_current", "read_file", "current result"),
    ];

    let turn = agent.run(&mut messages, &sink).await.unwrap();

    assert_eq!(turn.text, "recovered");
    assert_eq!(turn.rounds, 1, "the retry is part of one model round");
    assert_eq!(turn.usage.total_tokens, 35);
    assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 4);
    let requests = server.requests();
    assert_eq!(requests.len(), 4, "overflow, two side calls, and one retry");

    let first_messages = requests[0]["messages"].as_array().unwrap();
    assert!(first_messages
        .iter()
        .any(|message| message["content"] == prior_user));
    assert!(!first_messages.iter().any(|message| {
        message["content"]
            .as_str()
            .is_some_and(|content| content.starts_with("[Earlier conversation summary]"))
    }));

    let retry_messages = requests[3]["messages"].as_array().unwrap();
    assert!(retry_messages.iter().any(|message| {
        message["content"]
            .as_str()
            .is_some_and(|content| content.starts_with("[Earlier conversation summary]"))
    }));
    assert!(!retry_messages
        .iter()
        .any(|message| message["content"] == prior_user));
    assert_eq!(
        retry_messages
            .iter()
            .filter(|message| message["role"] == "user" && message["content"] == current_user)
            .count(),
        1,
        "the current user message is unchanged"
    );
    assert_eq!(
        messages
            .iter()
            .filter(|message| matches!(message, Message::Assistant { message } if message.content.as_deref() == Some("recovered")))
            .count(),
        1,
        "the successful assistant response is appended once"
    );
    let events = sink.events();
    assert!(events
        .iter()
        .any(|event| event == "info:model context overflow; compacting and retrying once"));
    assert!(!events.iter().any(|event| event.contains("retrying in")));
}

#[tokio::test]
async fn fails_after_a_second_context_overflow() {
    let server = MockServer::new(|_path, _body| context_overflow_response());
    let directory = tempfile::tempdir().unwrap();
    let agent = build_context_limited_agent(&server, directory.path());
    let sink = RecordingSink::default();
    let prior_user = format!(
        "prior completed turn {}",
        "historical detail ".repeat(4_500)
    );
    let mut messages = vec![
        Message::user(prior_user),
        Message::assistant(AssistantMessage::text("prior answer")),
        Message::user("current request"),
    ];

    let error = agent.run(&mut messages, &sink).await.unwrap_err();

    assert_eq!(
        server.request_count(),
        2,
        "only one recovery retry is allowed"
    );
    assert!(
        error.to_string().contains("context_length_exceeded"),
        "{error}"
    );
}

#[tokio::test]
async fn does_not_retry_context_overflow_without_a_smaller_projection() {
    let server = MockServer::new(|_path, _body| context_overflow_response());
    let directory = tempfile::tempdir().unwrap();
    let agent = build_context_limited_agent(&server, directory.path());
    let sink = RecordingSink::default();
    let mut messages = vec![Message::user("only the current request")];

    let error = agent.run(&mut messages, &sink).await.unwrap_err();

    assert_eq!(server.request_count(), 1);
    assert!(
        error.to_string().contains("context_length_exceeded"),
        "{error}"
    );
    assert!(!sink
        .events()
        .iter()
        .any(|event| event.contains("compacting and retrying")));
}

#[tokio::test]
async fn stops_at_the_round_budget() {
    let server = MockServer::new(|_path, _body| {
        MockResponse::sse(&[
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_x","function":{"name":"read_file","arguments":"{\"path\":\"loop.txt\"}"}}]},"finish_reason":"tool_calls"}]}"#,
            &done_chunk("tool_calls"),
        ])
    });
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("loop.txt"), "content\n").unwrap();
    let client = ModelClient::from_profile(&mock_profile(server.base_url())).unwrap();
    let mut options = agent_options(directory.path().to_path_buf(), 3);
    options.max_rounds = 3;
    let agent = Agent::new(
        Arc::new(client),
        Arc::new(ToolRegistry::builtin()),
        Arc::new(AllowAll),
        Arc::new(NoAsker),
        options,
    );
    let sink = RecordingSink::default();
    let mut messages = vec![Message::user("loop forever")];

    let error = agent.run(&mut messages, &sink).await.unwrap_err();
    assert!(error.to_string().contains("maximum"), "{error}");
}

#[tokio::test]
async fn empty_model_response_is_an_error() {
    let server = MockServer::new(|_path, _body| MockResponse::sse(&[&done_chunk("stop")]));
    let directory = tempfile::tempdir().unwrap();
    let agent = build_agent(&server, directory.path(), Arc::new(AllowAll));
    let sink = RecordingSink::default();
    let mut messages = vec![Message::user("hello")];

    let error = agent.run(&mut messages, &sink).await.unwrap_err();
    assert!(error.to_string().contains("empty"), "{error}");
}

#[tokio::test]
async fn qa_agent_is_read_only() {
    let server = MockServer::new(|_path, body| {
        let has_tool_result = body["messages"]
            .as_array()
            .map(|messages| messages.iter().any(|message| message["role"] == "tool"))
            .unwrap_or(false);
        if has_tool_result {
            MockResponse::sse(&[&text_chunk("understood"), &done_chunk("stop")])
        } else {
            MockResponse::sse(&[
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_w","function":{"name":"write_file","arguments":"{\"path\":\"blocked.txt\",\"content\":\"nope\"}"}}]},"finish_reason":"tool_calls"}]}"#,
                &done_chunk("tool_calls"),
            ])
        }
    });
    let directory = tempfile::tempdir().unwrap();
    let client = ModelClient::from_profile(&mock_profile(server.base_url())).unwrap();
    let agent = Agent::new(
        Arc::new(client),
        Arc::new(ToolRegistry::builtin()),
        Arc::new(AllowAll),
        Arc::new(NoAsker),
        AgentOptions {
            kind: AgentKind::Qa,
            cwd: directory.path().to_path_buf(),
            max_rounds: 4,
            tool_timeout: Duration::from_secs(5),
        },
    );
    let sink = RecordingSink::default();
    let mut messages = vec![Message::user("write a file")];

    let turn = agent.run(&mut messages, &sink).await.unwrap();
    assert_eq!(turn.text, "understood");
    assert!(!directory.path().join("blocked.txt").exists());

    let request = &server.requests()[0];
    let names: Vec<&str> = request["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["function"]["name"].as_str().unwrap())
        .collect();
    assert!(!names.contains(&"write_file"), "{names:?}");
    assert!(!names.contains(&"bash"), "{names:?}");
    assert!(names.contains(&"read_file"), "{names:?}");

    let tool_message = messages
        .iter()
        .find(|message| matches!(message, Message::Tool { .. }))
        .expect("tool result");
    if let Message::Tool { content, .. } = tool_message {
        assert!(content.contains("not available"), "{content}");
    }
}
