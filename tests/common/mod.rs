//! Shared helpers for integration tests: a tiny HTTP/SSE mock model server.

#![allow(dead_code)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;

/// One scripted response.
#[derive(Clone)]
pub struct MockResponse {
    pub status: u16,
    pub body: String,
    pub content_type: String,
}

impl MockResponse {
    /// An SSE chat-completions stream built from raw JSON chunks.
    pub fn sse(chunks: &[&str]) -> Self {
        let mut body = String::new();
        for chunk in chunks {
            body.push_str("data: ");
            body.push_str(chunk);
            body.push_str("\n\n");
        }
        body.push_str("data: [DONE]\n\n");
        Self {
            status: 200,
            body,
            content_type: "text/event-stream".to_string(),
        }
    }

    pub fn json(status: u16, body: &str) -> Self {
        Self {
            status,
            body: body.to_string(),
            content_type: "application/json".to_string(),
        }
    }
}

type Handler = Box<dyn Fn(&str, &Value) -> MockResponse + Send + Sync>;

/// A local HTTP server that records request bodies and replies with scripted
/// SSE or JSON bodies.
pub struct MockServer {
    url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    hits: Arc<AtomicUsize>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl MockServer {
    pub fn new(handler: impl Fn(&str, &Value) -> MockResponse + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let address = listener.local_addr().expect("mock server address");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let hits = Arc::new(AtomicUsize::new(0));
        let handler: Arc<Handler> = Arc::new(Box::new(handler));
        let requests_for_thread = requests.clone();
        let hits_for_thread = hits.clone();
        let handle = std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else {
                    continue;
                };
                hits_for_thread.fetch_add(1, Ordering::SeqCst);
                serve_connection(stream, handler.as_ref(), &requests_for_thread);
            }
        });
        Self {
            url: format!("http://{address}"),
            requests,
            hits,
            handle: Some(handle),
        }
    }

    pub fn url(&self) -> String {
        self.url.clone()
    }

    /// Base URL including a conventional `/v1` prefix.
    pub fn base_url(&self) -> String {
        format!("{}/v1", self.url)
    }

    pub fn requests(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }

    pub fn hits(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }

    pub fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        // Closing the listener ends the accept loop; the thread is detached so
        // tests never block on shutdown.
        self.handle.take();
    }
}

fn serve_connection(
    mut stream: std::net::TcpStream,
    handler: &Handler,
    requests: &Arc<Mutex<Vec<Value>>>,
) -> Option<()> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).ok()?;
    let mut content_length = 0usize;
    let mut path = String::from("/");
    if let Some(target) = request_line.split_whitespace().nth(1) {
        path = target.to_string();
    }
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            break;
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body).ok()?;
    let value: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    // Record before writing the response so a test that waits for the reply
    // can always inspect the request.
    if let Ok(mut recorded) = requests.lock() {
        recorded.push(value.clone());
    }
    let response = handler(&path, &value);

    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        status_text(response.status),
        response.content_type,
        response.body.len()
    );
    stream.write_all(head.as_bytes()).ok()?;
    stream.write_all(response.body.as_bytes()).ok()?;
    stream.flush().ok()?;
    Some(())
}

fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        _ => "Status",
    }
}

/// Create a model profile pointed at a mock server.
pub fn mock_profile(base_url: String) -> icode::config::ModelProfile {
    icode::config::ModelProfile {
        name: "mock".to_string(),
        provider: "openai".to_string(),
        api_style: icode::config::ApiStyle::Chat,
        model: "mock-model".to_string(),
        base_url,
        api_key: "test-key".to_string(),
        max_tokens: None,
        context_window: None,
        temperature: Some(0.0),
        stream: true,
    }
}

/// A tool context rooted at `cwd`.
pub fn tool_context(cwd: &std::path::Path) -> icode::tools::ToolContext {
    icode::tools::ToolContext {
        cwd: cwd.to_path_buf(),
        timeout: std::time::Duration::from_secs(10),
        asker: std::sync::Arc::new(icode::interaction::ScriptedAsker::new(vec![
            "scripted answer".to_string(),
        ])),
        todos: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
    }
}

/// Sink that records every agent event as a string.
#[derive(Default)]
pub struct RecordingSink {
    pub events: Mutex<Vec<String>>,
    pub projections: Mutex<Vec<icode::compaction::ContextProjection>>,
}

impl RecordingSink {
    pub fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }

    fn push(&self, event: String) {
        self.events.lock().unwrap().push(event);
    }
}

impl icode::agent::EventSink for RecordingSink {
    fn persist_context_projection(
        &self,
        projection: &icode::compaction::ContextProjection,
    ) -> anyhow::Result<bool> {
        self.projections.lock().unwrap().push(projection.clone());
        Ok(true)
    }

    fn text_delta(&self, delta: &str) {
        self.push(format!("text:{delta}"));
    }

    fn reasoning_delta(&self, delta: &str) {
        self.push(format!("reasoning:{delta}"));
    }

    fn tool_started(&self, name: &str, summary: &str) {
        self.push(format!("tool_start:{name}:{summary}"));
    }

    fn tool_finished(
        &self,
        name: &str,
        ok: bool,
        _duration: std::time::Duration,
        _summary: &str,
        output: &str,
    ) {
        self.push(format!("tool_done:{name}:{ok}:{output}"));
    }

    fn info(&self, message: &str) {
        self.push(format!("info:{message}"));
    }

    fn error(&self, message: &str) {
        self.push(format!("error:{message}"));
    }
}

/// JSON for an assistant text delta chunk.
pub fn text_chunk(text: &str) -> String {
    serde_json::json!({
        "choices": [{"delta": {"content": text}}]
    })
    .to_string()
}

/// JSON for a final chunk with a finish reason and usage.
pub fn done_chunk(finish_reason: &str) -> String {
    serde_json::json!({
        "choices": [{"delta": {}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    })
    .to_string()
}
