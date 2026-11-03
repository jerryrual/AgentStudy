mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use common::*;
use icode::agent::{Agent, AgentKind, AgentOptions};
use icode::interaction::{AllowAll, ApprovalDecision, DenyAll, NoAsker, ScriptedApprover};
use icode::model::{Message, ModelClient};
use icode::tools::ToolRegistry;

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
