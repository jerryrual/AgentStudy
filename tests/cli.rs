mod common;

use std::process::Command;

use common::*;
use serde_json::{json, Value};

fn settings_json(base_url: &str) -> String {
    json!({
        "models": [{
            "name": "mock",
            "provider": "openai",
            "api_style": "chat",
            "model": "mock-model",
            "base_url": base_url,
            "api_key": "test-key",
            "stream": true
        }],
        "current_model": "mock",
        "approval_mode": "auto",
        "agent": "code",
        "max_tool_rounds": 10
    })
    .to_string()
}

fn icode() -> Command {
    Command::new(env!("CARGO_BIN_EXE_icode"))
}

fn write_settings(home: &std::path::Path, base_url: &str) {
    std::fs::write(home.join("settings.json"), settings_json(base_url)).unwrap();
}

#[test]
fn run_json_returns_a_machine_readable_result() {
    let server = MockServer::new(|_path, _body| {
        MockResponse::sse(&[&text_chunk("Hello from mock"), &done_chunk("stop")])
    });
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    std::fs::write(
        work.path().join("AGENTS.md"),
        vec![b'a'; icode::memory::DEFAULT_MAX_MEMORY_TOKENS * 4 + 1],
    )
    .unwrap();
    write_settings(home.path(), &server.base_url());

    let output = icode()
        .env("ICODE_HOME", home.path())
        .args([
            "run",
            "hello",
            "--json",
            "--cwd",
            work.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("35,000-token"), "{stderr}");
    assert_eq!(payload["ok"], true);
    assert_eq!(payload["text"], "Hello from mock");
    assert_eq!(payload["tool_calls"], 0);
    assert_eq!(payload["rounds"], 1);
    assert!(payload["usage"]["total_tokens"].as_u64().unwrap() > 0);
    assert!(!payload["session_id"].as_str().unwrap().is_empty());

    let sessions: Vec<_> = std::fs::read_dir(home.path().join("sessions"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "jsonl")
        })
        .collect();
    assert_eq!(sessions.len(), 1, "one session file was written");

    // The request carried the user prompt and the tool definitions.
    let requests = server.requests();
    assert_eq!(requests.len(), 1);
    let messages = requests[0]["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[1]["content"], "hello");
    assert!(!requests[0]["tools"].as_array().unwrap().is_empty());
}

#[test]
fn run_streams_text_to_stdout_and_progress_to_stderr() {
    let server = MockServer::new(|_path, _body| {
        MockResponse::sse(&[&text_chunk("streamed answer"), &done_chunk("stop")])
    });
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    write_settings(home.path(), &server.base_url());

    let output = icode()
        .env("ICODE_HOME", home.path())
        .args(["run", "hello", "--cwd", work.path().to_str().unwrap()])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "streamed answer");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Done"), "{stderr}");
}

#[test]
fn models_lists_configured_profiles() {
    let home = tempfile::tempdir().unwrap();
    std::fs::write(
        home.path().join("settings.json"),
        settings_json("http://127.0.0.1:9/v1"),
    )
    .unwrap();
    let output = icode()
        .env("ICODE_HOME", home.path())
        .arg("models")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("mock"), "{stdout}");
    assert!(stdout.contains("mock-model"), "{stdout}");
    // The key itself is never printed.
    assert!(!stdout.contains("test-key"), "{stdout}");
}

#[test]
fn missing_environment_key_is_reported_clearly() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let settings = json!({
        "models": [{
            "name": "mock",
            "provider": "openai",
            "model": "mock-model",
            "base_url": "http://127.0.0.1:9/v1",
            "api_key": "{{ICODE_TEST_KEY_THAT_DOES_NOT_EXIST}}",
            "stream": true
        }],
        "current_model": "mock"
    })
    .to_string();
    std::fs::write(home.path().join("settings.json"), settings).unwrap();

    let output = icode()
        .env("ICODE_HOME", home.path())
        .args(["run", "hello", "--cwd", work.path().to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("ICODE_TEST_KEY_THAT_DOES_NOT_EXIST"),
        "{stderr}"
    );
}

#[test]
fn unknown_model_is_rejected() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    std::fs::write(
        home.path().join("settings.json"),
        settings_json("http://127.0.0.1:9/v1"),
    )
    .unwrap();
    let output = icode()
        .env("ICODE_HOME", home.path())
        .args([
            "run",
            "hello",
            "--model",
            "does-not-exist",
            "--cwd",
            work.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("does-not-exist"), "{stderr}");
    assert!(stderr.contains("mock"), "{stderr}");
}

#[test]
fn version_flag_prints_the_version() {
    let output = icode().arg("--version").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(env!("CARGO_PKG_VERSION")), "{stdout}");
}

#[test]
fn run_continue_uses_the_latest_session() {
    let server = MockServer::new(|_path, _body| {
        MockResponse::sse(&[&text_chunk("continued"), &done_chunk("stop")])
    });
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    write_settings(home.path(), &server.base_url());
    let sessions = home.path().join("sessions");
    std::fs::create_dir_all(&sessions).unwrap();
    let meta = json!({
        "type": "meta",
        "meta": {
            "id": "1a1a0000000-00001",
            "created_ms": 1,
            "cwd": work.path(),
            "model": "mock-model",
            "agent": "code"
        }
    });
    let previous = json!({"type": "user", "text": "previous context"});
    std::fs::write(
        sessions.join("1a1a0000000-00001.jsonl"),
        format!("{}\n{}\n", meta, previous),
    )
    .unwrap();

    let output = icode()
        .env("ICODE_HOME", home.path())
        .args([
            "run",
            "follow up",
            "--continue",
            "--json",
            "--cwd",
            work.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["ok"], true);
    assert_eq!(payload["text"], "continued");
    let request = &server.requests()[0];
    let contents: Vec<&str> = request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|message| message["content"].as_str())
        .collect();
    assert!(contents.contains(&"previous context"), "{contents:?}");
    assert!(contents.contains(&"follow up"), "{contents:?}");
}

#[test]
fn resume_turn_continues_only_from_a_complete_tool_checkpoint() {
    let server = MockServer::new(|_path, _body| {
        MockResponse::sse(&[&text_chunk("resumed"), &done_chunk("stop")])
    });
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    write_settings(home.path(), &server.base_url());
    let sessions = home.path().join("sessions");
    std::fs::create_dir_all(&sessions).unwrap();
    let id = "1a1a0000000-00002";
    let records = [
        json!({
            "type": "meta",
            "meta": {
                "id": id,
                "created_ms": 2,
                "cwd": work.path(),
                "model": "mock-model",
                "agent": "code"
            }
        }),
        json!({"type": "user", "text": "inspect the result"}),
        json!({
            "type": "assistant",
            "content": null,
            "reasoning": null,
            "tool_calls": [{
                "id": "call_1",
                "name": "read_file",
                "arguments": "{\"path\":\"a.txt\"}"
            }],
            "responses_items": []
        }),
        json!({
            "type": "tool",
            "call_id": "call_1",
            "name": "read_file",
            "content": "durable result"
        }),
    ];
    let transcript = records
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    let path = sessions.join(format!("{id}.jsonl"));
    std::fs::write(&path, format!("{transcript}\n")).unwrap();

    let output = icode()
        .env("ICODE_HOME", home.path())
        .args([
            "resume-turn",
            "--session",
            id,
            "--json",
            "--cwd",
            work.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let request = &server.requests()[0];
    let contents: Vec<&str> = request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|message| message["content"].as_str())
        .collect();
    assert!(contents.contains(&"inspect the result"), "{contents:?}");
    assert!(contents.contains(&"durable result"), "{contents:?}");

    let persisted = std::fs::read_to_string(path).unwrap();
    assert_eq!(
        persisted
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|record| record["type"] == "user")
            .count(),
        1,
        "same-turn resume must not append a User"
    );
}

#[test]
fn resume_turn_rejects_an_ambiguous_user_only_tail() {
    let server = MockServer::new(|_path, _body| panic!("provider must not be called"));
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    write_settings(home.path(), &server.base_url());
    let sessions = home.path().join("sessions");
    std::fs::create_dir_all(&sessions).unwrap();
    let id = "1a1a0000000-00003";
    let meta = json!({
        "type": "meta",
        "meta": {
            "id": id,
            "created_ms": 3,
            "cwd": work.path(),
            "model": "mock-model",
            "agent": "code"
        }
    });
    let user = json!({"type": "user", "text": "ambiguous work"});
    std::fs::write(
        sessions.join(format!("{id}.jsonl")),
        format!("{meta}\n{user}\n"),
    )
    .unwrap();

    let output = icode()
        .env("ICODE_HOME", home.path())
        .args(["resume-turn", "--session", id])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no complete tool or Last Words checkpoint"),
        "{stderr}"
    );
    assert_eq!(server.request_count(), 0);
}
