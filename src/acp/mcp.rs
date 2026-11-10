//! Strict projection and runtime primitives for stable ACP MCP declarations.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming MCP runtime wiring")
)]

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::v1::schema::{McpServer, McpServerHttp};
use anyhow::{bail, Context};
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::tools::{truncate_output, Tool, ToolContext, ToolRegistry};

const MAX_SERVERS: usize = 32;
const MAX_ARGS: usize = 256;
const MAX_ENV: usize = 256;
const MAX_HTTP_HEADERS: usize = 128;
const MAX_HTTP_HEADER_BYTES: usize = 64 * 1024;
const MAX_MESSAGE_BYTES: usize = 4 * 1024 * 1024;
const MCP_PROTOCOL_VERSION: &str = "2025-06-18";
const MAX_TOOL_PAGES: usize = 64;
const MAX_DISCOVERED_TOOLS: usize = 1024;
const MAX_MODEL_TOOL_NAME_BYTES: usize = 64;
const STARTUP_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StdioServerSpec {
    pub(crate) name: String,
    pub(crate) command: PathBuf,
    pub(crate) args: Vec<String>,
    pub(crate) env: Vec<(String, String)>,
}

#[derive(Clone)]
pub(crate) struct HttpServerSpec {
    pub(crate) name: String,
    url: reqwest::Url,
    headers: HeaderMap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpecError {
    TooManyServers,
    UnsupportedTransport,
    InvalidName,
    DuplicateName,
    InvalidCommand,
    TooManyArguments,
    InvalidArgument,
    TooManyEnvironmentVariables,
    InvalidEnvironmentVariable,
    DuplicateEnvironmentVariable,
    InvalidUrl,
    TooManyHttpHeaders,
    InvalidHttpHeader,
    DuplicateHttpHeader,
    ReservedHttpHeader,
}

pub(crate) fn stdio_specs(servers: &[McpServer]) -> Result<Vec<StdioServerSpec>, SpecError> {
    if servers.len() > MAX_SERVERS {
        return Err(SpecError::TooManyServers);
    }
    let mut names = HashSet::new();
    servers
        .iter()
        .map(|server| {
            let McpServer::Stdio(server) = server else {
                return Err(SpecError::UnsupportedTransport);
            };
            if server.name.is_empty() || server.name.contains('\0') {
                return Err(SpecError::InvalidName);
            }
            if !names.insert(server.name.clone()) {
                return Err(SpecError::DuplicateName);
            }
            if !server.command.is_absolute()
                || server.command.as_os_str().is_empty()
                || server.command.to_string_lossy().contains('\0')
            {
                return Err(SpecError::InvalidCommand);
            }
            if server.args.len() > MAX_ARGS {
                return Err(SpecError::TooManyArguments);
            }
            if server.args.iter().any(|argument| argument.contains('\0')) {
                return Err(SpecError::InvalidArgument);
            }
            if server.env.len() > MAX_ENV {
                return Err(SpecError::TooManyEnvironmentVariables);
            }
            let mut environment_names = HashSet::new();
            let mut env = Vec::with_capacity(server.env.len());
            for variable in &server.env {
                if variable.name.is_empty()
                    || variable.name.contains(['\0', '='])
                    || variable.value.contains('\0')
                {
                    return Err(SpecError::InvalidEnvironmentVariable);
                }
                if !environment_names.insert(variable.name.clone()) {
                    return Err(SpecError::DuplicateEnvironmentVariable);
                }
                env.push((variable.name.clone(), variable.value.clone()));
            }
            Ok(StdioServerSpec {
                name: server.name.clone(),
                command: server.command.clone(),
                args: server.args.clone(),
                env,
            })
        })
        .collect()
}

impl std::fmt::Debug for HttpServerSpec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HttpServerSpec")
            .field("name", &self.name)
            .field("origin", &self.url.origin().ascii_serialization())
            .field("header_count", &self.headers.len())
            .finish()
    }
}

pub(crate) fn http_spec(server: &McpServerHttp) -> Result<HttpServerSpec, SpecError> {
    if server.name.is_empty() || server.name.contains('\0') {
        return Err(SpecError::InvalidName);
    }
    let url = reqwest::Url::parse(&server.url).map_err(|_| SpecError::InvalidUrl)?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.has_host()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(SpecError::InvalidUrl);
    }
    if server.headers.len() > MAX_HTTP_HEADERS {
        return Err(SpecError::TooManyHttpHeaders);
    }
    let mut headers = HeaderMap::with_capacity(server.headers.len());
    let mut header_bytes = 0usize;
    for header in &server.headers {
        header_bytes = header_bytes
            .saturating_add(header.name.len())
            .saturating_add(header.value.len());
        if header_bytes > MAX_HTTP_HEADER_BYTES {
            return Err(SpecError::TooManyHttpHeaders);
        }
        let name = HeaderName::from_bytes(header.name.as_bytes())
            .map_err(|_| SpecError::InvalidHttpHeader)?;
        if matches!(
            name.as_str(),
            "accept"
                | "connection"
                | "content-length"
                | "content-type"
                | "host"
                | "mcp-protocol-version"
                | "mcp-session-id"
                | "origin"
                | "transfer-encoding"
        ) {
            return Err(SpecError::ReservedHttpHeader);
        }
        if headers.contains_key(&name) {
            return Err(SpecError::DuplicateHttpHeader);
        }
        let mut value =
            HeaderValue::from_str(&header.value).map_err(|_| SpecError::InvalidHttpHeader)?;
        value.set_sensitive(true);
        headers.insert(name, value);
    }
    Ok(HttpServerSpec {
        name: server.name.clone(),
        url,
        headers,
    })
}

pub(crate) struct StdioServerProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Option<ChildStdout>,
}

impl StdioServerProcess {
    pub(crate) fn spawn(spec: &StdioServerSpec) -> std::io::Result<Self> {
        Self::spawn_with_cwd(spec, None)
    }

    fn spawn_in(spec: &StdioServerSpec, cwd: &Path) -> std::io::Result<Self> {
        Self::spawn_with_cwd(spec, Some(cwd))
    }

    fn spawn_with_cwd(spec: &StdioServerSpec, cwd: Option<&Path>) -> std::io::Result<Self> {
        let mut command = Command::new(&spec.command);
        command.args(&spec.args).envs(spec.env.iter().cloned());
        if let Some(cwd) = cwd {
            command.current_dir(cwd);
        }
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let mut child = command.spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("MCP child stdin was not piped"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("MCP child stdout was not piped"))?;
        Ok(Self {
            child,
            stdin: Some(stdin),
            stdout: Some(stdout),
        })
    }

    pub(crate) fn take_stdin(&mut self) -> Option<ChildStdin> {
        self.stdin.take()
    }

    pub(crate) fn take_stdout(&mut self) -> Option<ChildStdout> {
        self.stdout.take()
    }

    pub(crate) async fn shutdown(mut self, grace: Duration) -> std::io::Result<ExitStatus> {
        drop(self.stdin.take());
        match tokio::time::timeout(grace, self.child.wait()).await {
            Ok(status) => status,
            Err(_) => {
                self.child.kill().await?;
                self.child.wait().await
            }
        }
    }
}

impl Drop for StdioServerProcess {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

#[derive(Debug)]
pub(crate) enum TransportError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Oversized,
    Truncated,
    InvalidEnvelope,
    Http(reqwest::Error),
    HttpStatus(reqwest::StatusCode),
    SessionExpired,
    InvalidContentType,
    InvalidSessionId,
    MissingResponse,
    MultipleResponses,
    UnexpectedServerRequest,
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "MCP stdio I/O failed: {error}"),
            Self::Json(error) => write!(formatter, "invalid MCP JSON: {error}"),
            Self::Oversized => formatter.write_str("MCP message exceeds the size limit"),
            Self::Truncated => formatter.write_str("MCP message is not newline terminated"),
            Self::InvalidEnvelope => formatter.write_str("invalid MCP JSON-RPC envelope"),
            Self::Http(error) => write!(formatter, "MCP HTTP request failed: {error}"),
            Self::HttpStatus(status) => write!(formatter, "MCP HTTP status was {status}"),
            Self::SessionExpired => formatter.write_str("MCP HTTP session expired"),
            Self::InvalidContentType => formatter.write_str("invalid MCP HTTP content type"),
            Self::InvalidSessionId => formatter.write_str("invalid MCP HTTP session ID"),
            Self::MissingResponse => formatter.write_str("MCP HTTP response was empty"),
            Self::MultipleResponses => {
                formatter.write_str("MCP HTTP response contained multiple messages")
            }
            Self::UnexpectedServerRequest => {
                formatter.write_str("MCP HTTP server requests are not supported")
            }
        }
    }
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Http(error) => Some(error),
            Self::Oversized
            | Self::Truncated
            | Self::InvalidEnvelope
            | Self::HttpStatus(_)
            | Self::SessionExpired
            | Self::InvalidContentType
            | Self::InvalidSessionId
            | Self::MissingResponse
            | Self::MultipleResponses
            | Self::UnexpectedServerRequest => None,
        }
    }
}

pub(crate) struct StdioJsonRpc {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl StdioJsonRpc {
    pub(crate) fn from_process(process: &mut StdioServerProcess) -> std::io::Result<Self> {
        let stdin = process
            .take_stdin()
            .ok_or_else(|| std::io::Error::other("MCP child stdin already taken"))?;
        let stdout = process
            .take_stdout()
            .ok_or_else(|| std::io::Error::other("MCP child stdout already taken"))?;
        Ok(Self {
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    pub(crate) async fn send(&mut self, message: &Value) -> Result<(), TransportError> {
        let encoded = encode_frame(message)?;
        self.stdin
            .write_all(&encoded)
            .await
            .map_err(TransportError::Io)?;
        self.stdin.flush().await.map_err(TransportError::Io)
    }

    pub(crate) async fn receive(&mut self) -> Result<Value, TransportError> {
        let mut encoded = Vec::new();
        let read = (&mut self.stdout)
            .take((MAX_MESSAGE_BYTES + 1) as u64)
            .read_until(b'\n', &mut encoded)
            .await
            .map_err(TransportError::Io)?;
        if read == 0 {
            return Err(TransportError::Truncated);
        }
        decode_frame(encoded)
    }
}

pub(crate) struct StreamableHttpJsonRpc {
    client: reqwest::Client,
    spec: HttpServerSpec,
    session_id: Option<HeaderValue>,
    negotiated: bool,
}

impl StreamableHttpJsonRpc {
    pub(crate) fn new(spec: HttpServerSpec) -> Result<Self, TransportError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs(60))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(TransportError::Http)?;
        Ok(Self {
            client,
            spec,
            session_id: None,
            negotiated: false,
        })
    }

    pub(crate) async fn request(&mut self, message: &Value) -> Result<Value, TransportError> {
        validate_envelope(message)?;
        let Some(object) = message.as_object() else {
            return Err(TransportError::InvalidEnvelope);
        };
        if !object.contains_key("id") || !object.contains_key("method") {
            return Err(TransportError::InvalidEnvelope);
        }
        let expected_id = object
            .get("id")
            .cloned()
            .ok_or(TransportError::InvalidEnvelope)?;
        let initialize = object.get("method").and_then(Value::as_str) == Some("initialize");
        let response = self.send(message).await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND && self.session_id.is_some() {
            return Err(TransportError::SessionExpired);
        }
        if !response.status().is_success() {
            return Err(TransportError::HttpStatus(response.status()));
        }
        let response_headers = response.headers().clone();
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .map(str::trim)
            .map(str::to_owned)
            .ok_or(TransportError::InvalidContentType)?;
        let message = if content_type.eq_ignore_ascii_case("application/json") {
            decode_http_json(&bounded_http_body(response).await?)?
        } else if content_type.eq_ignore_ascii_case("text/event-stream") {
            receive_http_sse(response, &expected_id).await?
        } else {
            return Err(TransportError::InvalidContentType);
        };
        let response_object = message.as_object().ok_or(TransportError::InvalidEnvelope)?;
        if response_object.contains_key("method") || response_object.get("id") != Some(&expected_id)
        {
            return Err(TransportError::InvalidEnvelope);
        }
        let valid_initialize = initialize
            && message
                .get("result")
                .and_then(|result| result.get("protocolVersion"))
                .and_then(Value::as_str)
                == Some(MCP_PROTOCOL_VERSION);
        self.capture_session_id(&response_headers, valid_initialize)?;
        if valid_initialize {
            self.negotiated = true;
        }
        Ok(message)
    }

    pub(crate) async fn notify(&mut self, message: &Value) -> Result<(), TransportError> {
        validate_envelope(message)?;
        let Some(object) = message.as_object() else {
            return Err(TransportError::InvalidEnvelope);
        };
        if object.contains_key("id") || !object.contains_key("method") {
            return Err(TransportError::InvalidEnvelope);
        }
        let response = self.send(message).await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND && self.session_id.is_some() {
            return Err(TransportError::SessionExpired);
        }
        if response.status() != reqwest::StatusCode::ACCEPTED {
            return Err(TransportError::HttpStatus(response.status()));
        }
        self.capture_session_id(response.headers(), false)?;
        if !bounded_http_body(response).await?.is_empty() {
            return Err(TransportError::MultipleResponses);
        }
        Ok(())
    }

    async fn send(&self, message: &Value) -> Result<reqwest::Response, TransportError> {
        let mut request = self
            .client
            .post(self.spec.url.clone())
            .headers(self.spec.headers.clone())
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/json, text/event-stream")
            .json(message);
        if self.negotiated {
            request = request.header("mcp-protocol-version", MCP_PROTOCOL_VERSION);
        }
        if let Some(session_id) = &self.session_id {
            request = request.header("mcp-session-id", session_id.clone());
        }
        request
            .send()
            .await
            .map_err(|error| TransportError::Http(error.without_url()))
    }

    fn capture_session_id(
        &mut self,
        headers: &HeaderMap,
        allow_new: bool,
    ) -> Result<(), TransportError> {
        let Some(value) = headers.get("mcp-session-id") else {
            return Ok(());
        };
        if value.is_empty()
            || value
                .as_bytes()
                .iter()
                .any(|byte| !(0x21..=0x7e).contains(byte))
        {
            return Err(TransportError::InvalidSessionId);
        }
        if self.session_id.is_none() && !allow_new {
            return Err(TransportError::InvalidSessionId);
        }
        if self.session_id.as_ref().is_some_and(|known| known != value) {
            return Err(TransportError::InvalidSessionId);
        }
        let mut value = value.clone();
        value.set_sensitive(true);
        self.session_id = Some(value);
        Ok(())
    }
}

async fn bounded_http_body(response: reqwest::Response) -> Result<Vec<u8>, TransportError> {
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| TransportError::Http(error.without_url()))?;
        if body.len().saturating_add(chunk.len()) > MAX_MESSAGE_BYTES {
            return Err(TransportError::Oversized);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn decode_http_json(body: &[u8]) -> Result<Value, TransportError> {
    if body.is_empty() {
        return Err(TransportError::MissingResponse);
    }
    let message = serde_json::from_slice(body).map_err(TransportError::Json)?;
    validate_envelope(&message)?;
    Ok(message)
}

#[derive(Default)]
struct HttpSseDecoder {
    buffer: Vec<u8>,
    data: String,
    total_bytes: usize,
}

impl HttpSseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<Value>, TransportError> {
        self.total_bytes = self.total_bytes.saturating_add(bytes.len());
        if self.total_bytes > MAX_MESSAGE_BYTES {
            return Err(TransportError::Oversized);
        }
        self.buffer.extend_from_slice(bytes);
        let mut messages = Vec::new();
        let mut buffer = std::mem::take(&mut self.buffer);
        let mut start = 0;
        while let Some(offset) = buffer[start..].iter().position(|byte| *byte == b'\n') {
            let end = start + offset;
            let raw = &buffer[start..end];
            let line = raw.strip_suffix(b"\r").unwrap_or(raw);
            self.consume_line(line, &mut messages)?;
            start = end + 1;
        }
        buffer.drain(..start);
        self.buffer = buffer;
        Ok(messages)
    }

    fn finish(&mut self) -> Result<Vec<Value>, TransportError> {
        let mut messages = Vec::new();
        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            self.consume_line(&line, &mut messages)?;
        }
        self.flush_data(&mut messages)?;
        Ok(messages)
    }

    fn consume_line(
        &mut self,
        line: &[u8],
        messages: &mut Vec<Value>,
    ) -> Result<(), TransportError> {
        let line = std::str::from_utf8(line).map_err(|_| TransportError::InvalidEnvelope)?;
        if line.is_empty() {
            return self.flush_data(messages);
        }
        if line.starts_with(':') {
            return Ok(());
        }
        if let Some(value) = line.strip_prefix("data:") {
            if !self.data.is_empty() {
                self.data.push('\n');
            }
            self.data.push_str(value.strip_prefix(' ').unwrap_or(value));
        }
        Ok(())
    }

    fn flush_data(&mut self, messages: &mut Vec<Value>) -> Result<(), TransportError> {
        if !self.data.is_empty() {
            messages.push(decode_http_json(self.data.as_bytes())?);
            self.data.clear();
        }
        Ok(())
    }
}

async fn receive_http_sse(
    response: reqwest::Response,
    expected_id: &Value,
) -> Result<Value, TransportError> {
    let mut stream = response.bytes_stream();
    let mut decoder = HttpSseDecoder::default();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| TransportError::Http(error.without_url()))?;
        if let Some(message) = select_http_sse_response(decoder.push(&chunk)?, expected_id)? {
            return Ok(message);
        }
    }
    if let Some(message) = select_http_sse_response(decoder.finish()?, expected_id)? {
        return Ok(message);
    }
    Err(TransportError::MissingResponse)
}

fn select_http_sse_response(
    messages: Vec<Value>,
    expected_id: &Value,
) -> Result<Option<Value>, TransportError> {
    for message in messages {
        let object = message.as_object().ok_or(TransportError::InvalidEnvelope)?;
        if object.contains_key("method") {
            if object.contains_key("id") {
                return Err(TransportError::UnexpectedServerRequest);
            }
            continue;
        }
        if object.get("id") != Some(expected_id) {
            return Err(TransportError::InvalidEnvelope);
        }
        return Ok(Some(message));
    }
    Ok(None)
}

fn encode_frame(message: &Value) -> Result<Vec<u8>, TransportError> {
    validate_envelope(message)?;
    let mut encoded = serde_json::to_vec(message).map_err(TransportError::Json)?;
    if encoded.len() >= MAX_MESSAGE_BYTES {
        return Err(TransportError::Oversized);
    }
    encoded.push(b'\n');
    Ok(encoded)
}

fn decode_frame(mut encoded: Vec<u8>) -> Result<Value, TransportError> {
    if encoded.len() > MAX_MESSAGE_BYTES {
        return Err(TransportError::Oversized);
    }
    if encoded.last() != Some(&b'\n') {
        return Err(TransportError::Truncated);
    }
    encoded.pop();
    if encoded.last() == Some(&b'\r') {
        encoded.pop();
    }
    let message = serde_json::from_slice(&encoded).map_err(TransportError::Json)?;
    validate_envelope(&message)?;
    Ok(message)
}

fn validate_envelope(message: &Value) -> Result<(), TransportError> {
    let Some(object) = message.as_object() else {
        return Err(TransportError::InvalidEnvelope);
    };
    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(TransportError::InvalidEnvelope);
    }
    if !object.contains_key("method")
        && !object.contains_key("result")
        && !object.contains_key("error")
    {
        return Err(TransportError::InvalidEnvelope);
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) enum RequestError {
    Transport(TransportError),
    Unusable,
    IdExhausted,
    UnexpectedMessage,
    MismatchedId,
    InvalidResponse,
    Remote(Value),
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(error) => error.fmt(formatter),
            Self::Unusable => formatter.write_str("MCP client stream is no longer reusable"),
            Self::IdExhausted => formatter.write_str("MCP request ID space is exhausted"),
            Self::UnexpectedMessage => {
                formatter.write_str("unexpected MCP message while awaiting response")
            }
            Self::MismatchedId => formatter.write_str("MCP response ID does not match request"),
            Self::InvalidResponse => formatter.write_str("invalid MCP response payload"),
            Self::Remote(error) => write!(formatter, "MCP server returned an error: {error}"),
        }
    }
}

impl std::error::Error for RequestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(error) => Some(error),
            Self::Unusable
            | Self::IdExhausted
            | Self::UnexpectedMessage
            | Self::MismatchedId
            | Self::InvalidResponse
            | Self::Remote(_) => None,
        }
    }
}

fn valid_remote_error(value: &Value) -> bool {
    let Some(error) = value.as_object() else {
        return false;
    };
    let integer_code = error
        .get("code")
        .and_then(Value::as_f64)
        .is_some_and(|code| code.is_finite() && code.fract() == 0.0);
    integer_code && error.get("message").and_then(Value::as_str).is_some()
}

pub(crate) struct StdioMcpClient {
    transport: StdioJsonRpc,
    next_id: u64,
    usable: bool,
    initialized: bool,
    discovered_tools: HashSet<String>,
}

impl StdioMcpClient {
    pub(crate) fn from_process(process: &mut StdioServerProcess) -> std::io::Result<Self> {
        Ok(Self {
            transport: StdioJsonRpc::from_process(process)?,
            next_id: 1,
            usable: true,
            initialized: false,
            discovered_tools: HashSet::new(),
        })
    }

    pub(crate) async fn request(
        &mut self,
        method: &str,
        params: Value,
    ) -> Result<Value, RequestError> {
        if !self.usable {
            return Err(RequestError::Unusable);
        }
        let id = self.next_id;
        self.next_id = id.checked_add(1).ok_or(RequestError::IdExhausted)?;
        self.usable = false;
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        if let Err(error) = self.transport.send(&request).await {
            self.usable = false;
            return Err(RequestError::Transport(error));
        }
        let response = match self.transport.receive().await {
            Ok(response) => response,
            Err(error) => {
                self.usable = false;
                return Err(RequestError::Transport(error));
            }
        };
        let object = response.as_object().ok_or(RequestError::InvalidResponse)?;
        if object.contains_key("method") {
            self.usable = false;
            return Err(RequestError::UnexpectedMessage);
        }
        if object.get("id") != Some(&Value::from(id)) {
            return Err(RequestError::MismatchedId);
        }
        match (object.get("result"), object.get("error")) {
            (Some(result), None) => {
                self.usable = true;
                Ok(result.clone())
            }
            (None, Some(error)) if valid_remote_error(error) => {
                self.usable = true;
                Err(RequestError::Remote(error.clone()))
            }
            _ => Err(RequestError::InvalidResponse),
        }
    }
    async fn notify(&mut self, method: &str, params: Value) -> Result<(), RequestError> {
        if !self.usable {
            return Err(RequestError::Unusable);
        }
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.usable = false;
        if let Err(error) = self.transport.send(&notification).await {
            return Err(RequestError::Transport(error));
        }
        self.usable = true;
        Ok(())
    }

    pub(crate) async fn initialize(&mut self) -> Result<ServerInitialization, InitializeError> {
        let result = self
            .request(
                "initialize",
                serde_json::json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "icode",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                }),
            )
            .await
            .map_err(|error| {
                self.usable = false;
                InitializeError::Request(error)
            })?;
        let initialization = match ServerInitialization::from_result(&result) {
            Ok(initialization) => initialization,
            Err(error) => {
                self.usable = false;
                return Err(error);
            }
        };
        self.notify("notifications/initialized", serde_json::json!({}))
            .await
            .map_err(InitializeError::Request)?;
        self.initialized = true;
        Ok(initialization)
    }

    pub(crate) async fn list_tools(&mut self) -> Result<Vec<DiscoveredTool>, DiscoveryError> {
        if !self.initialized {
            return Err(DiscoveryError::NotInitialized);
        }
        let mut cursor = None;
        let mut seen_cursors = HashSet::new();
        let mut seen_names = HashSet::new();
        let mut tools = Vec::new();
        for _ in 0..MAX_TOOL_PAGES {
            let params = match &cursor {
                Some(cursor) => serde_json::json!({"cursor": cursor}),
                None => serde_json::json!({}),
            };
            let result = self
                .request("tools/list", params)
                .await
                .map_err(DiscoveryError::Request)?;
            let page = ToolPage::from_result(&result)?;
            if tools.len() + page.tools.len() > MAX_DISCOVERED_TOOLS {
                return Err(DiscoveryError::TooManyTools);
            }
            for tool in page.tools {
                if !seen_names.insert(tool.name.clone()) {
                    return Err(DiscoveryError::DuplicateName(tool.name));
                }
                tools.push(tool);
            }
            let Some(next_cursor) = page.next_cursor else {
                self.discovered_tools = tools.iter().map(|tool| tool.name.clone()).collect();
                return Ok(tools);
            };
            if !seen_cursors.insert(next_cursor.clone()) {
                return Err(DiscoveryError::RepeatedCursor);
            }
            cursor = Some(next_cursor);
        }
        Err(DiscoveryError::TooManyPages)
    }

    pub(crate) async fn call_tool(
        &mut self,
        name: &str,
        arguments: Value,
    ) -> Result<ToolCallResult, ToolCallError> {
        if !self.initialized {
            return Err(ToolCallError::NotInitialized);
        }
        if !self.discovered_tools.contains(name) {
            return Err(ToolCallError::UnknownTool(name.to_string()));
        }
        if !arguments.is_object() {
            return Err(ToolCallError::InvalidArguments);
        }
        let result = self
            .request(
                "tools/call",
                serde_json::json!({"name": name, "arguments": arguments}),
            )
            .await
            .map_err(ToolCallError::Request)?;
        ToolCallResult::from_result(&result)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ServerInitialization {
    pub(crate) capabilities: Value,
    pub(crate) server_name: String,
    pub(crate) server_version: String,
    pub(crate) instructions: Option<String>,
}

impl ServerInitialization {
    fn from_result(result: &Value) -> Result<Self, InitializeError> {
        let object = result.as_object().ok_or(InitializeError::InvalidResult)?;
        let version = object
            .get("protocolVersion")
            .and_then(Value::as_str)
            .ok_or(InitializeError::InvalidResult)?;
        if version != MCP_PROTOCOL_VERSION {
            return Err(InitializeError::UnsupportedVersion(version.to_string()));
        }
        let capabilities = object
            .get("capabilities")
            .filter(|value| value.is_object())
            .cloned()
            .ok_or(InitializeError::InvalidResult)?;
        let server_info = object
            .get("serverInfo")
            .and_then(Value::as_object)
            .ok_or(InitializeError::InvalidResult)?;
        let server_name = nonempty_string(server_info.get("name"))?;
        let server_version = nonempty_string(server_info.get("version"))?;
        let instructions = match object.get("instructions") {
            None | Some(Value::Null) => None,
            Some(Value::String(instructions)) => Some(instructions.clone()),
            Some(_) => return Err(InitializeError::InvalidResult),
        };
        Ok(Self {
            capabilities,
            server_name,
            server_version,
            instructions,
        })
    }
}

fn nonempty_string(value: Option<&Value>) -> Result<String, InitializeError> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or(InitializeError::InvalidResult)
}

#[derive(Debug)]
pub(crate) enum InitializeError {
    Request(RequestError),
    UnsupportedVersion(String),
    InvalidResult,
}

impl std::fmt::Display for InitializeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request(error) => error.fmt(formatter),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported MCP protocol version: {version}")
            }
            Self::InvalidResult => formatter.write_str("invalid MCP initialize result"),
        }
    }
}

impl std::error::Error for InitializeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Request(error) => Some(error),
            Self::UnsupportedVersion(_) | Self::InvalidResult => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DiscoveredTool {
    pub(crate) name: String,
    pub(crate) title: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) input_schema: Value,
    pub(crate) output_schema: Option<Value>,
    pub(crate) annotations: Option<Value>,
    pub(crate) meta: Option<Value>,
}

struct ToolPage {
    tools: Vec<DiscoveredTool>,
    next_cursor: Option<String>,
}

impl ToolPage {
    fn from_result(result: &Value) -> Result<Self, DiscoveryError> {
        let object = result.as_object().ok_or(DiscoveryError::InvalidResult)?;
        let values = object
            .get("tools")
            .and_then(Value::as_array)
            .ok_or(DiscoveryError::InvalidResult)?;
        let tools = values
            .iter()
            .map(DiscoveredTool::from_value)
            .collect::<Result<_, _>>()?;
        let next_cursor = match object.get("nextCursor") {
            None | Some(Value::Null) => None,
            Some(Value::String(cursor)) if !cursor.is_empty() => Some(cursor.clone()),
            Some(_) => return Err(DiscoveryError::InvalidResult),
        };
        Ok(Self { tools, next_cursor })
    }
}

impl DiscoveredTool {
    fn from_value(value: &Value) -> Result<Self, DiscoveryError> {
        let object = value.as_object().ok_or(DiscoveryError::InvalidResult)?;
        let name = object
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .ok_or(DiscoveryError::InvalidResult)?;
        let title = optional_string(object.get("title"))?;
        let description = optional_string(object.get("description"))?;
        let input_schema =
            optional_object(object.get("inputSchema"))?.ok_or(DiscoveryError::InvalidResult)?;
        let output_schema = optional_object(object.get("outputSchema"))?;
        let annotations = optional_object(object.get("annotations"))?;
        let meta = optional_object(object.get("_meta"))?;
        Ok(Self {
            name,
            title,
            description,
            input_schema,
            output_schema,
            annotations,
            meta,
        })
    }
}

fn optional_string(value: Option<&Value>) -> Result<Option<String>, DiscoveryError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(DiscoveryError::InvalidResult),
    }
}

fn optional_object(value: Option<&Value>) -> Result<Option<Value>, DiscoveryError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) if value.is_object() => Ok(Some(value.clone())),
        Some(_) => Err(DiscoveryError::InvalidResult),
    }
}

#[derive(Debug)]
pub(crate) enum DiscoveryError {
    Request(RequestError),
    NotInitialized,
    InvalidResult,
    RepeatedCursor,
    TooManyPages,
    TooManyTools,
    DuplicateName(String),
}

impl std::fmt::Display for DiscoveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request(error) => error.fmt(formatter),
            Self::NotInitialized => formatter.write_str("MCP client is not initialized"),
            Self::InvalidResult => formatter.write_str("invalid MCP tools/list result"),
            Self::RepeatedCursor => formatter.write_str("MCP tools/list cursor did not advance"),
            Self::TooManyPages => formatter.write_str("MCP tools/list exceeds page limit"),
            Self::TooManyTools => formatter.write_str("MCP tools/list exceeds tool limit"),
            Self::DuplicateName(name) => write!(formatter, "duplicate MCP tool name: {name}"),
        }
    }
}

impl std::error::Error for DiscoveryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Request(error) => Some(error),
            Self::NotInitialized
            | Self::InvalidResult
            | Self::RepeatedCursor
            | Self::TooManyPages
            | Self::TooManyTools
            | Self::DuplicateName(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolCallResult {
    pub(crate) content: Vec<Value>,
    pub(crate) structured_content: Option<Value>,
    pub(crate) is_error: bool,
    pub(crate) meta: Option<Value>,
}

impl ToolCallResult {
    fn from_result(result: &Value) -> Result<Self, ToolCallError> {
        let object = result.as_object().ok_or(ToolCallError::InvalidResult)?;
        let content = object
            .get("content")
            .and_then(Value::as_array)
            .ok_or(ToolCallError::InvalidResult)?;
        if content.iter().any(|item| {
            item.as_object()
                .and_then(|item| item.get("type"))
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
        }) {
            return Err(ToolCallError::InvalidResult);
        }
        let structured_content = call_result_object(object.get("structuredContent"))?;
        let meta = call_result_object(object.get("_meta"))?;
        let is_error = match object.get("isError") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(is_error)) => *is_error,
            Some(_) => return Err(ToolCallError::InvalidResult),
        };
        Ok(Self {
            content: content.clone(),
            structured_content,
            is_error,
            meta,
        })
    }
}

fn call_result_object(value: Option<&Value>) -> Result<Option<Value>, ToolCallError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) if value.is_object() => Ok(Some(value.clone())),
        Some(_) => Err(ToolCallError::InvalidResult),
    }
}

#[derive(Debug)]
pub(crate) enum ToolCallError {
    Request(RequestError),
    NotInitialized,
    UnknownTool(String),
    InvalidArguments,
    InvalidResult,
}

impl std::fmt::Display for ToolCallError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request(error) => error.fmt(formatter),
            Self::NotInitialized => formatter.write_str("MCP client is not initialized"),
            Self::UnknownTool(name) => write!(formatter, "unknown MCP tool: {name}"),
            Self::InvalidArguments => formatter.write_str("MCP tool arguments must be an object"),
            Self::InvalidResult => formatter.write_str("invalid MCP tools/call result"),
        }
    }
}

impl std::error::Error for ToolCallError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Request(error) => Some(error),
            Self::NotInitialized
            | Self::UnknownTool(_)
            | Self::InvalidArguments
            | Self::InvalidResult => None,
        }
    }
}

async fn bounded_blocking<T, F>(timeout: Duration, operation: F) -> anyhow::Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::time::timeout(timeout, tokio::task::spawn_blocking(operation))
        .await
        .context("blocking operation timed out")?
        .context("blocking operation task failed")
}

pub(crate) async fn start_tool_registry(
    specs: &[StdioServerSpec],
    cwd: &Path,
) -> anyhow::Result<Arc<ToolRegistry>> {
    start_tool_registry_with_timeout(specs, cwd, STARTUP_TIMEOUT).await
}

async fn start_tool_registry_with_timeout(
    specs: &[StdioServerSpec],
    cwd: &Path,
    startup_timeout: Duration,
) -> anyhow::Result<Arc<ToolRegistry>> {
    let mut adapters = Vec::new();
    for (server_index, spec) in specs.iter().enumerate() {
        let started = tokio::time::Instant::now();
        let owned_spec = spec.clone();
        let owned_cwd = cwd.to_path_buf();
        let mut process = bounded_blocking(startup_timeout, move || {
            StdioServerProcess::spawn_in(&owned_spec, &owned_cwd)
        })
        .await
        .with_context(|| format!("MCP server {:?} process spawn timed out", spec.name))?
        .with_context(|| format!("failed to start MCP server {:?}", spec.name))?;
        let remaining = startup_timeout.saturating_sub(started.elapsed());
        let mut client = StdioMcpClient::from_process(&mut process)
            .with_context(|| format!("failed to connect MCP server {:?}", spec.name))?;
        let tools = tokio::time::timeout(remaining, async {
            client
                .initialize()
                .await
                .with_context(|| format!("failed to initialize MCP server {:?}", spec.name))?;
            client
                .list_tools()
                .await
                .with_context(|| format!("failed to discover MCP server {:?}", spec.name))
        })
        .await
        .with_context(|| format!("MCP server {:?} startup timed out", spec.name))??;
        adapters.extend(tool_adapters(server_index, process, client, tools));
    }
    ToolRegistry::builtin_with(adapters)
        .map(Arc::new)
        .context("failed to compose MCP tool registry")
}

pub(crate) struct McpServerHandle {
    client: Mutex<StdioMcpClient>,
    _process: StdioServerProcess,
}

pub(crate) fn tool_adapters(
    server_index: usize,
    process: StdioServerProcess,
    client: StdioMcpClient,
    tools: Vec<DiscoveredTool>,
) -> Vec<Box<dyn Tool>> {
    let handle = Arc::new(McpServerHandle {
        client: Mutex::new(client),
        _process: process,
    });
    tools
        .into_iter()
        .enumerate()
        .map(|(tool_index, tool)| {
            let exposed_name = model_tool_name(server_index, tool_index, &tool.name);
            Box::new(McpTool::new(exposed_name, tool, Arc::clone(&handle))) as Box<dyn Tool>
        })
        .collect()
}

fn model_tool_name(server_index: usize, tool_index: usize, remote_name: &str) -> String {
    let mut name = format!("mcp_s{server_index}_t{tool_index}_");
    for character in remote_name.chars() {
        if name.len() == MAX_MODEL_TOOL_NAME_BYTES {
            break;
        }
        name.push(
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            },
        );
    }
    name
}

struct McpTool {
    exposed_name: String,
    remote_name: String,
    description: String,
    parameters: Value,
    handle: Arc<McpServerHandle>,
}

impl McpTool {
    fn new(exposed_name: String, tool: DiscoveredTool, handle: Arc<McpServerHandle>) -> Self {
        let description = tool
            .description
            .or(tool.title)
            .unwrap_or_else(|| "Tool provided by an MCP server".to_string());
        Self {
            exposed_name,
            remote_name: tool.name,
            description,
            parameters: tool.input_schema,
            handle,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &'static str {
        "mcp_tool"
    }

    fn description(&self) -> &'static str {
        "Tool provided by an MCP server"
    }

    fn model_name(&self) -> &str {
        &self.exposed_name
    }

    fn model_description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    fn mutates(&self) -> bool {
        true
    }

    fn summarize(&self, _raw_arguments: &str) -> String {
        format!("{} via MCP", self.exposed_name)
    }

    async fn run(&self, arguments: &Value, _context: &ToolContext) -> anyhow::Result<String> {
        let result = self
            .handle
            .client
            .lock()
            .await
            .call_tool(&self.remote_name, arguments.clone())
            .await?;
        finish_tool_result(result)
    }
}

fn finish_tool_result(result: ToolCallResult) -> anyhow::Result<String> {
    let is_error = result.is_error;
    let rendered = render_tool_result(result)?;
    if is_error {
        bail!("MCP tool returned an error: {}", truncate_output(&rendered));
    }
    Ok(rendered)
}

fn render_tool_result(result: ToolCallResult) -> serde_json::Result<String> {
    let mut rendered = serde_json::Map::new();
    rendered.insert("content".to_string(), Value::Array(result.content));
    rendered.insert("isError".to_string(), Value::Bool(result.is_error));
    if let Some(structured) = result.structured_content {
        rendered.insert("structuredContent".to_string(), structured);
    }
    if let Some(meta) = result.meta {
        rendered.insert("_meta".to_string(), meta);
    }
    serde_json::to_string(&Value::Object(rendered))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use agent_client_protocol::v1::schema::{
        EnvVariable, HttpHeader, McpServerHttp, McpServerStdio,
    };

    use crate::interaction::NoAsker;
    use crate::tools::ToolRegistry;

    use super::*;

    fn stdio(name: &str) -> McpServer {
        McpServer::Stdio(
            McpServerStdio::new(name, "/usr/bin/mcp")
                .args(vec!["--stdio".into(), "".into()])
                .env(vec![EnvVariable::new("TOKEN", "secret")]),
        )
    }

    #[test]
    fn validates_http_declarations_without_exposing_credentials() {
        let spec = http_spec(
            &McpServerHttp::new("remote", "https://example.invalid/mcp?tenant=private")
                .headers(vec![HttpHeader::new("Authorization", "Bearer secret")]),
        )
        .unwrap();
        let debug = format!("{spec:?}");
        assert!(debug.contains("https://example.invalid"));
        assert!(!debug.contains("tenant"));
        assert!(!debug.contains("secret"));

        for url in [
            "relative",
            "ftp://example.invalid/mcp",
            "https://user:secret@example.invalid/mcp",
            "https://example.invalid/mcp#fragment",
        ] {
            assert_eq!(
                http_spec(&McpServerHttp::new("remote", url)).unwrap_err(),
                SpecError::InvalidUrl
            );
        }
        assert_eq!(
            http_spec(
                &McpServerHttp::new("remote", "https://example.invalid/mcp")
                    .headers(vec![HttpHeader::new("Content-Type", "text/plain")])
            )
            .unwrap_err(),
            SpecError::ReservedHttpHeader
        );
        assert_eq!(
            http_spec(
                &McpServerHttp::new("remote", "https://example.invalid/mcp").headers(vec![
                    HttpHeader::new("X-Test", "first"),
                    HttpHeader::new("x-test", "second"),
                ])
            )
            .unwrap_err(),
            SpecError::DuplicateHttpHeader
        );
    }

    #[test]
    fn preserves_valid_stdio_declarations_exactly() {
        let specs = stdio_specs(&[stdio("tools")]).unwrap();
        assert_eq!(
            specs,
            vec![StdioServerSpec {
                name: "tools".into(),
                command: PathBuf::from("/usr/bin/mcp"),
                args: vec!["--stdio".into(), "".into()],
                env: vec![("TOKEN".into(), "secret".into())],
            }]
        );
    }

    #[test]
    fn rejects_unsupported_transport_and_duplicate_server_names() {
        let http = McpServer::Http(McpServerHttp::new("remote", "https://example.invalid"));
        assert_eq!(stdio_specs(&[http]), Err(SpecError::UnsupportedTransport));
        assert_eq!(
            stdio_specs(&[stdio("same"), stdio("same")]),
            Err(SpecError::DuplicateName)
        );
    }

    #[test]
    fn rejects_unsafe_process_fields() {
        assert_eq!(
            stdio_specs(&[McpServer::Stdio(McpServerStdio::new("tools", "relative"))]),
            Err(SpecError::InvalidCommand)
        );
        assert_eq!(
            stdio_specs(&[McpServer::Stdio(
                McpServerStdio::new("tools", "/usr/bin/mcp").args(vec!["bad\0argument".into()])
            )]),
            Err(SpecError::InvalidArgument)
        );
        assert_eq!(
            stdio_specs(&[McpServer::Stdio(
                McpServerStdio::new("tools", "/usr/bin/mcp").env(vec![
                    EnvVariable::new("TOKEN", "first"),
                    EnvVariable::new("TOKEN", "second"),
                ])
            )]),
            Err(SpecError::DuplicateEnvironmentVariable)
        );
    }

    #[test]
    fn enforces_collection_bounds() {
        let servers: Vec<_> = (0..=MAX_SERVERS)
            .map(|index| stdio(&format!("server-{index}")))
            .collect();
        assert_eq!(stdio_specs(&servers), Err(SpecError::TooManyServers));

        let arguments = vec!["argument".to_string(); MAX_ARGS + 1];
        assert_eq!(
            stdio_specs(&[McpServer::Stdio(
                McpServerStdio::new("tools", "/usr/bin/mcp").args(arguments)
            )]),
            Err(SpecError::TooManyArguments)
        );
    }

    #[test]
    fn streamable_http_handles_many_heartbeats_and_a_split_tail() {
        let mut decoder = HttpSseDecoder::default();
        let mut first_chunk = ":\n".repeat(32 * 1024).into_bytes();
        first_chunk.extend_from_slice(b"data: {\"jsonrpc\":\"2.0\",\"id\":9,\"result\":");
        assert!(decoder.push(&first_chunk).unwrap().is_empty());
        let messages = decoder.push(b"{}}\n\n").unwrap();
        let response = select_http_sse_response(messages, &serde_json::json!(9))
            .unwrap()
            .unwrap();
        assert_eq!(response["result"], serde_json::json!({}));
    }

    #[test]
    fn streamable_http_preserves_multiple_messages_and_rejects_session_rotation() {
        let mut decoder = HttpSseDecoder::default();
        let messages = decoder
            .push(
                b"data: {\"jsonrpc\":\"2.0\",\"method\":\"notice\"}\n\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n",
            )
            .unwrap();
        let response = select_http_sse_response(messages, &serde_json::json!(1))
            .unwrap()
            .unwrap();
        assert_eq!(response["result"], serde_json::json!({}));
        assert!(matches!(
            select_http_sse_response(
                vec![serde_json::json!({
                    "jsonrpc": "2.0", "id": "server-1", "method": "sampling/createMessage"
                })],
                &serde_json::json!(1),
            ),
            Err(TransportError::UnexpectedServerRequest)
        ));

        let spec = http_spec(&McpServerHttp::new("remote", "https://example.invalid/mcp")).unwrap();
        let mut transport = StreamableHttpJsonRpc::new(spec).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert("mcp-session-id", HeaderValue::from_static("session-1"));
        transport.capture_session_id(&headers, true).unwrap();
        headers.insert("mcp-session-id", HeaderValue::from_static("session-2"));
        assert!(matches!(
            transport.capture_session_id(&headers, false),
            Err(TransportError::InvalidSessionId)
        ));
        headers.insert("mcp-session-id", HeaderValue::from_static("bad id"));
        assert!(matches!(
            transport.capture_session_id(&headers, false),
            Err(TransportError::InvalidSessionId)
        ));
    }

    async fn read_http_request(stream: &mut tokio::net::TcpStream) -> String {
        let mut request = Vec::new();
        let header_end = loop {
            let mut buffer = [0u8; 1024];
            let read = stream.read(&mut buffer).await.unwrap();
            assert!(read > 0);
            request.extend_from_slice(&buffer[..read]);
            if let Some(position) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break position + 4;
            }
        };
        let headers = String::from_utf8(request[..header_end].to_vec()).unwrap();
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .unwrap();
        while request.len() < header_end + content_length {
            let mut buffer = vec![0u8; header_end + content_length - request.len()];
            let read = stream.read(&mut buffer).await.unwrap();
            assert!(read > 0);
            request.extend_from_slice(&buffer[..read]);
        }
        String::from_utf8(request).unwrap()
    }

    #[tokio::test]
    async fn streamable_http_round_trips_json_sse_and_session_headers() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let mut requests = Vec::new();
            let bodies = [
                (
                    "200 OK",
                    "Content-Type: application/json\r\nMcp-Session-Id: session-1\r\n",
                    r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18"}}"#,
                ),
                ("202 Accepted", "", ""),
                (
                    "200 OK",
                    "Content-Type: Text/Event-Stream; charset=utf-8\r\n",
                    ": keepalive\r\ndata: {\"jsonrpc\":\"2.0\",\r\ndata: \"id\":2,\"result\":{\"tools\":[]}}\r\n\r\n",
                ),
            ];
            for (status, extra_headers, body) in bodies {
                let (mut stream, _) = listener.accept().await.unwrap();
                requests.push(read_http_request(&mut stream).await);
                let response = format!(
                    "HTTP/1.1 {status}\r\n{extra_headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
            requests
        });
        let spec = http_spec(
            &McpServerHttp::new("remote", format!("http://{address}/mcp"))
                .headers(vec![HttpHeader::new("Authorization", "Bearer test")]),
        )
        .unwrap();
        let mut transport = StreamableHttpJsonRpc::new(spec).unwrap();
        let initialized = transport
            .request(&serde_json::json!({
                "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}
            }))
            .await
            .unwrap();
        assert_eq!(
            initialized["result"]["protocolVersion"],
            MCP_PROTOCOL_VERSION
        );
        transport
            .notify(&serde_json::json!({
                "jsonrpc": "2.0", "method": "notifications/initialized", "params": {}
            }))
            .await
            .unwrap();
        let listed = transport
            .request(&serde_json::json!({
                "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}
            }))
            .await
            .unwrap();
        assert_eq!(listed["result"]["tools"], serde_json::json!([]));

        let requests = server.await.unwrap();
        for request in &requests {
            let lower = request.to_ascii_lowercase();
            assert!(lower.starts_with("post /mcp http/1.1"));
            assert!(lower.contains("content-type: application/json"));
            assert!(lower.contains("accept: application/json, text/event-stream"));
            assert!(lower.contains("authorization: bearer test"));
        }
        let first = requests[0].to_ascii_lowercase();
        assert!(!first.contains("mcp-session-id:"));
        assert!(!first.contains("mcp-protocol-version:"));
        for request in &requests[1..] {
            let lower = request.to_ascii_lowercase();
            assert!(lower.contains("mcp-session-id: session-1"));
            assert!(lower.contains("mcp-protocol-version: 2025-06-18"));
        }
    }

    #[tokio::test]
    async fn streamable_http_stops_after_matching_sse_response() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (close_sender, close_receiver) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_http_request(&mut stream).await;
            let body = concat!(
                "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{}}\n\n",
                "data: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"tools\":[]}}\n\n"
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{body}\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.flush().await.unwrap();
            let _ = close_receiver.await;
        });
        let spec = http_spec(&McpServerHttp::new(
            "remote",
            format!("http://{address}/mcp"),
        ))
        .unwrap();
        let mut transport = StreamableHttpJsonRpc::new(spec).unwrap();
        let response = tokio::time::timeout(
            Duration::from_secs(1),
            transport.request(&serde_json::json!({
                "jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {}
            })),
        )
        .await
        .expect("matching SSE response should not wait for EOF")
        .unwrap();
        assert_eq!(response["result"]["tools"], serde_json::json!([]));
        let _ = close_sender.send(());
        server.await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn child_inherits_parent_environment_and_overrides_declared_values() {
        let inherited_path = std::env::var("PATH").expect("test process has PATH");
        let spec = StdioServerSpec {
            name: "env".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec![
                "-c".into(),
                "printf '%s|%s' \"$PATH\" \"$ACP_MCP_EXPLICIT\"".into(),
            ],
            env: vec![("ACP_MCP_EXPLICIT".into(), "declared".into())],
        };
        let mut process = StdioServerProcess::spawn(&spec).unwrap();
        let mut stdout = process.take_stdout().unwrap();
        let mut output = String::new();
        stdout.read_to_string(&mut output).await.unwrap();
        let status = process.shutdown(Duration::from_secs(1)).await.unwrap();

        assert!(status.success());
        assert_eq!(output, format!("{inherited_path}|declared"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn child_exposes_pipes_and_shutdown_closes_stdin() {
        let spec = StdioServerSpec {
            name: "echo".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "cat".into()],
            env: Vec::new(),
        };
        let mut process = StdioServerProcess::spawn(&spec).unwrap();
        let mut stdin = process.take_stdin().unwrap();
        let mut stdout = process.take_stdout().unwrap();
        stdin.write_all(b"ping").await.unwrap();
        drop(stdin);
        let mut output = Vec::new();
        stdout.read_to_end(&mut output).await.unwrap();
        let status = process.shutdown(Duration::from_secs(1)).await.unwrap();

        assert!(status.success());
        assert_eq!(output, b"ping");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn json_rpc_transport_round_trips_one_complete_envelope() {
        let spec = StdioServerSpec {
            name: "rpc".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "cat".into()],
            env: Vec::new(),
        };
        let mut process = StdioServerProcess::spawn(&spec).unwrap();
        let mut transport = StdioJsonRpc::from_process(&mut process).unwrap();
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        });
        transport.send(&request).await.unwrap();
        assert_eq!(transport.receive().await.unwrap(), request);
        drop(transport);
        assert!(process
            .shutdown(Duration::from_secs(1))
            .await
            .unwrap()
            .success());
    }

    #[test]
    fn json_rpc_transport_round_trips_exact_frame_limit() {
        let mut request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "boundary",
            "params": ""
        });
        let baseline = serde_json::to_vec(&request).unwrap().len();
        request["params"] = Value::String("x".repeat(MAX_MESSAGE_BYTES - 1 - baseline));
        assert_eq!(
            serde_json::to_vec(&request).unwrap().len() + 1,
            MAX_MESSAGE_BYTES
        );

        let encoded = encode_frame(&request).unwrap();
        assert_eq!(encoded.len(), MAX_MESSAGE_BYTES);
        assert_eq!(decode_frame(encoded).unwrap(), request);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn json_rpc_transport_rejects_invalid_truncated_and_oversized_messages() {
        for script in ["printf 'not-json\\n'", "printf '{\"jsonrpc\":\"2.0\"}'"] {
            let spec = StdioServerSpec {
                name: "invalid".into(),
                command: PathBuf::from("/bin/sh"),
                args: vec!["-c".into(), script.into()],
                env: Vec::new(),
            };
            let mut process = StdioServerProcess::spawn(&spec).unwrap();
            let mut transport = StdioJsonRpc::from_process(&mut process).unwrap();
            let error = transport.receive().await.unwrap_err();
            assert!(matches!(
                error,
                TransportError::Json(_) | TransportError::Truncated
            ));
            drop(transport);
            process.shutdown(Duration::from_secs(1)).await.unwrap();
        }

        let inbound_spec = StdioServerSpec {
            name: "inbound-oversized".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "head -c 4194305 /dev/zero".into()],
            env: Vec::new(),
        };
        let mut inbound_process = StdioServerProcess::spawn(&inbound_spec).unwrap();
        let mut inbound_transport = StdioJsonRpc::from_process(&mut inbound_process).unwrap();
        assert!(matches!(
            inbound_transport.receive().await,
            Err(TransportError::Oversized)
        ));
        drop(inbound_transport);
        inbound_process
            .shutdown(Duration::from_secs(1))
            .await
            .unwrap();

        let spec = StdioServerSpec {
            name: "oversized".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "cat".into()],
            env: Vec::new(),
        };
        let mut process = StdioServerProcess::spawn(&spec).unwrap();
        let mut transport = StdioJsonRpc::from_process(&mut process).unwrap();
        let oversized = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "oversized",
            "params": "x".repeat(MAX_MESSAGE_BYTES)
        });
        assert!(matches!(
            transport.send(&oversized).await,
            Err(TransportError::Oversized)
        ));
        drop(transport);
        process.shutdown(Duration::from_secs(1)).await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_client_correlates_sequential_response_ids() {
        let script = concat!(
            "read first; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,",
            "\"result\":{\"step\":1}}'; ",
            "read second; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,",
            "\"result\":{\"step\":2}}'"
        );
        let spec = StdioServerSpec {
            name: "requests".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), script.into()],
            env: Vec::new(),
        };
        let mut process = StdioServerProcess::spawn(&spec).unwrap();
        let mut client = StdioMcpClient::from_process(&mut process).unwrap();

        assert_eq!(
            client
                .request("first", serde_json::json!({}))
                .await
                .unwrap(),
            serde_json::json!({"step": 1})
        );
        assert_eq!(
            client
                .request("second", serde_json::json!({}))
                .await
                .unwrap(),
            serde_json::json!({"step": 2})
        );
        drop(client);
        assert!(process
            .shutdown(Duration::from_secs(1))
            .await
            .unwrap()
            .success());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_client_rejects_mismatched_and_unsolicited_messages() {
        let cases = [
            (r#"{"jsonrpc":"2.0","id":99,"result":{}}"#, "mismatched"),
            (
                r#"{"jsonrpc":"2.0","method":"notice","params":{}}"#,
                "unexpected",
            ),
            (
                r#"{"jsonrpc":"2.0","id":1,"result":{},"error":{"code":-1}}"#,
                "invalid",
            ),
            (r#"{"jsonrpc":"2.0","id":1,"error":null}"#, "invalid"),
            (
                r#"{"jsonrpc":"2.0","id":1,"error":{"code":1.5,"message":"bad"}}"#,
                "invalid",
            ),
        ];
        for (response, expected) in cases {
            let script = format!("read request; printf '%s\\n' '{response}'");
            let spec = StdioServerSpec {
                name: expected.into(),
                command: PathBuf::from("/bin/sh"),
                args: vec!["-c".into(), script],
                env: Vec::new(),
            };
            let mut process = StdioServerProcess::spawn(&spec).unwrap();
            let mut client = StdioMcpClient::from_process(&mut process).unwrap();
            let error = client
                .request("test", serde_json::json!({}))
                .await
                .unwrap_err();
            match expected {
                "mismatched" => assert!(matches!(error, RequestError::MismatchedId)),
                "unexpected" => assert!(matches!(error, RequestError::UnexpectedMessage)),
                "invalid" => assert!(matches!(error, RequestError::InvalidResponse)),
                _ => unreachable!(),
            }
            assert!(matches!(
                client.request("again", serde_json::json!({})).await,
                Err(RequestError::Unusable)
            ));
            drop(client);
            process.shutdown(Duration::from_secs(1)).await.unwrap();
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelled_request_leaves_mcp_client_unusable() {
        let spec = StdioServerSpec {
            name: "cancelled".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec![
                "-c".into(),
                "read request; sleep 1; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}'"
                    .into(),
            ],
            env: Vec::new(),
        };
        let mut process = StdioServerProcess::spawn(&spec).unwrap();
        let mut client = StdioMcpClient::from_process(&mut process).unwrap();

        assert!(tokio::time::timeout(
            Duration::from_millis(50),
            client.request("slow", serde_json::json!({}))
        )
        .await
        .is_err());
        assert!(matches!(
            client.request("again", serde_json::json!({})).await,
            Err(RequestError::Unusable)
        ));
        drop(client);
        process.shutdown(Duration::from_millis(10)).await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_client_remains_reusable_after_remote_error() {
        let script = concat!(
            "read first; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,",
            "\"error\":{\"code\":-32603.0,\"message\":\"no\"}}'; ",
            "read second; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,",
            "\"result\":{\"ok\":true}}'"
        );
        let spec = StdioServerSpec {
            name: "remote-error".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), script.into()],
            env: Vec::new(),
        };
        let mut process = StdioServerProcess::spawn(&spec).unwrap();
        let mut client = StdioMcpClient::from_process(&mut process).unwrap();

        assert!(matches!(
            client.request("first", serde_json::json!({})).await,
            Err(RequestError::Remote(_))
        ));
        assert_eq!(
            client
                .request("second", serde_json::json!({}))
                .await
                .unwrap(),
            serde_json::json!({"ok": true})
        );
        drop(client);
        process.shutdown(Duration::from_secs(1)).await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_client_initializes_then_notifies_with_supported_version() {
        let script = concat!(
            "read initialize; ",
            "case \"$initialize\" in *'\"method\":\"initialize\"'*) ;; *) exit 2;; esac; ",
            "case \"$initialize\" in *'\"protocolVersion\":\"2025-06-18\"'*) ;; *) exit 3;; esac; ",
            "printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{",
            "\"protocolVersion\":\"2025-06-18\",\"capabilities\":{\"tools\":{}},",
            "\"serverInfo\":{\"name\":\"fixture\",\"version\":\"1.2.3\"},",
            "\"instructions\":\"use carefully\"}}'; ",
            "read initialized; ",
            "case \"$initialized\" in *'\"method\":\"notifications/initialized\"'*) exit 0;; *) exit 4;; esac"
        );
        let spec = StdioServerSpec {
            name: "initialize".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), script.into()],
            env: Vec::new(),
        };
        let mut process = StdioServerProcess::spawn(&spec).unwrap();
        let mut client = StdioMcpClient::from_process(&mut process).unwrap();

        assert_eq!(
            client.initialize().await.unwrap(),
            ServerInitialization {
                capabilities: serde_json::json!({"tools": {}}),
                server_name: "fixture".into(),
                server_version: "1.2.3".into(),
                instructions: Some("use carefully".into()),
            }
        );
        drop(client);
        assert!(process
            .shutdown(Duration::from_secs(1))
            .await
            .unwrap()
            .success());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_client_rejects_unsupported_version_and_becomes_unusable() {
        let script = concat!(
            "read initialize; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,",
            "\"result\":{\"protocolVersion\":\"2099-01-01\",\"capabilities\":{},",
            "\"serverInfo\":{\"name\":\"future\",\"version\":\"1\"}}}'"
        );
        let spec = StdioServerSpec {
            name: "unsupported".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), script.into()],
            env: Vec::new(),
        };
        let mut process = StdioServerProcess::spawn(&spec).unwrap();
        let mut client = StdioMcpClient::from_process(&mut process).unwrap();

        assert!(matches!(
            client.initialize().await,
            Err(InitializeError::UnsupportedVersion(version)) if version == "2099-01-01"
        ));
        assert!(matches!(
            client.request("again", serde_json::json!({})).await,
            Err(RequestError::Unusable)
        ));
        drop(client);
        process.shutdown(Duration::from_secs(1)).await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_client_remote_initialize_error_becomes_unusable() {
        let script = concat!(
            "read initialize; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,",
            "\"error\":{\"code\":-1,\"message\":\"denied\"}}'"
        );
        let spec = StdioServerSpec {
            name: "remote-initialize-error".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), script.into()],
            env: Vec::new(),
        };
        let mut process = StdioServerProcess::spawn(&spec).unwrap();
        let mut client = StdioMcpClient::from_process(&mut process).unwrap();

        assert!(matches!(
            client.initialize().await,
            Err(InitializeError::Request(RequestError::Remote(_)))
        ));
        assert!(matches!(
            client.request("again", serde_json::json!({})).await,
            Err(RequestError::Unusable)
        ));
        drop(client);
        process.shutdown(Duration::from_secs(1)).await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_client_discovers_valid_tools_across_pages() {
        let script = concat!(
            "read initialize; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,",
            "\"result\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},",
            "\"serverInfo\":{\"name\":\"fixture\",\"version\":\"1\"}}}'; ",
            "read initialized; read page1; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,",
            "\"result\":{\"tools\":[{\"name\":\"read\",\"title\":\"Read\",",
            "\"inputSchema\":{\"type\":\"object\"}}],\"nextCursor\":\"page-2\"}}'; ",
            "read page2; case \"$page2\" in *'\"cursor\":\"page-2\"'*) ;; *) exit 5;; esac; ",
            "printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{",
            "\"tools\":[{\"name\":\"write\",\"description\":\"Writes\",",
            "\"inputSchema\":{},\"outputSchema\":{},\"annotations\":{},\"_meta\":{}}]}}'"
        );
        let spec = StdioServerSpec {
            name: "tools".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), script.into()],
            env: Vec::new(),
        };
        let mut process = StdioServerProcess::spawn(&spec).unwrap();
        let mut client = StdioMcpClient::from_process(&mut process).unwrap();
        client.initialize().await.unwrap();

        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "read");
        assert_eq!(tools[0].title.as_deref(), Some("Read"));
        assert_eq!(tools[1].name, "write");
        assert_eq!(tools[1].description.as_deref(), Some("Writes"));
        assert!(tools[1].output_schema.as_ref().unwrap().is_object());
        drop(client);
        assert!(process
            .shutdown(Duration::from_secs(1))
            .await
            .unwrap()
            .success());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_client_rejects_repeated_tool_cursor() {
        let script = concat!(
            "read initialize; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,",
            "\"result\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},",
            "\"serverInfo\":{\"name\":\"fixture\",\"version\":\"1\"}}}'; ",
            "read initialized; read page1; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,",
            "\"result\":{\"tools\":[],\"nextCursor\":\"same\"}}'; ",
            "read page2; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":3,",
            "\"result\":{\"tools\":[],\"nextCursor\":\"same\"}}'"
        );
        let spec = StdioServerSpec {
            name: "cursor".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), script.into()],
            env: Vec::new(),
        };
        let mut process = StdioServerProcess::spawn(&spec).unwrap();
        let mut client = StdioMcpClient::from_process(&mut process).unwrap();
        client.initialize().await.unwrap();

        assert!(matches!(
            client.list_tools().await,
            Err(DiscoveryError::RepeatedCursor)
        ));
        drop(client);
        process.shutdown(Duration::from_secs(1)).await.unwrap();
    }

    #[test]
    fn initialize_result_requires_capabilities_and_server_identity() {
        assert!(matches!(
            ServerInitialization::from_result(&serde_json::json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": [],
                "serverInfo": {"name": "", "version": "1"}
            })),
            Err(InitializeError::InvalidResult)
        ));
    }

    #[test]
    fn tool_page_rejects_invalid_schema_and_cursor() {
        assert!(matches!(
            ToolPage::from_result(&serde_json::json!({
                "tools": [{"name": "bad", "inputSchema": []}],
                "nextCursor": ""
            })),
            Err(DiscoveryError::InvalidResult)
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_client_calls_only_discovered_tools_and_preserves_result() {
        let script = concat!(
            "read initialize; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,",
            "\"result\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},",
            "\"serverInfo\":{\"name\":\"fixture\",\"version\":\"1\"}}}'; ",
            "read initialized; read list; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,",
            "\"result\":{\"tools\":[{\"name\":\"echo\",\"inputSchema\":{}}]}}'; ",
            "read call; case \"$call\" in *'\"name\":\"echo\"'*'\"text\":\"hi\"'*) ;; *) exit 6;; esac; ",
            "printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{",
            "\"content\":[{\"type\":\"text\",\"text\":\"hi\"}],",
            "\"structuredContent\":{\"echo\":\"hi\"},\"isError\":false,\"_meta\":{\"source\":\"fixture\"}}}'"
        );
        let spec = StdioServerSpec {
            name: "call".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), script.into()],
            env: Vec::new(),
        };
        let mut process = StdioServerProcess::spawn(&spec).unwrap();
        let mut client = StdioMcpClient::from_process(&mut process).unwrap();
        client.initialize().await.unwrap();
        client.list_tools().await.unwrap();

        assert!(matches!(
            client.call_tool("missing", serde_json::json!({})).await,
            Err(ToolCallError::UnknownTool(name)) if name == "missing"
        ));
        let result = client
            .call_tool("echo", serde_json::json!({"text": "hi"}))
            .await
            .unwrap();
        assert_eq!(
            result.content,
            vec![serde_json::json!({"type": "text", "text": "hi"})]
        );
        assert_eq!(
            result.structured_content,
            Some(serde_json::json!({"echo": "hi"}))
        );
        assert!(!result.is_error);
        assert_eq!(result.meta, Some(serde_json::json!({"source": "fixture"})));
        drop(client);
        assert!(process
            .shutdown(Duration::from_secs(1))
            .await
            .unwrap()
            .success());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_adapter_registers_dynamic_metadata_and_preserves_full_result() {
        let script = concat!(
            "read initialize; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,",
            "\"result\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},",
            "\"serverInfo\":{\"name\":\"fixture\",\"version\":\"1\"}}}'; ",
            "read initialized; read list; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,",
            "\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"Echoes\",",
            "\"inputSchema\":{\"type\":\"object\"}}]}}'; ",
            "read call; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{",
            "\"content\":[{\"type\":\"text\",\"text\":\"hi\"}],",
            "\"structuredContent\":{\"echo\":\"hi\"},\"_meta\":{\"source\":\"fixture\"}}}'"
        );
        let spec = StdioServerSpec {
            name: "adapter".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), script.into()],
            env: Vec::new(),
        };
        let mut process = StdioServerProcess::spawn(&spec).unwrap();
        let mut client = StdioMcpClient::from_process(&mut process).unwrap();
        client.initialize().await.unwrap();
        let discovered = client.list_tools().await.unwrap();
        let registry =
            ToolRegistry::builtin_with(tool_adapters(3, process, client, discovered)).unwrap();
        let model_name = "mcp_s3_t0_echo";
        let tool = registry.get(model_name).unwrap();

        assert_eq!(tool.name(), "mcp_tool");
        assert_eq!(tool.model_name(), model_name);
        assert_eq!(tool.model_description(), "Echoes");
        assert_eq!(tool.parameters(), serde_json::json!({"type": "object"}));
        assert!(tool.mutates());
        assert!(!tool.parallel_safe());
        assert!(registry.model_names().contains(&model_name));
        assert!(registry.get("read_file").is_some());
        let context = ToolContext {
            cwd: PathBuf::from("/tmp"),
            timeout: Duration::from_secs(1),
            asker: Arc::new(NoAsker),
            todos: Arc::new(Mutex::new(Vec::new())),
        };
        let output = tool
            .run(&serde_json::json!({"text": "hi"}), &context)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&output).unwrap(),
            serde_json::json!({
                "content": [{"type": "text", "text": "hi"}],
                "structuredContent": {"echo": "hi"},
                "isError": false,
                "_meta": {"source": "fixture"}
            })
        );
    }

    #[tokio::test]
    async fn timed_out_blocking_operation_drops_its_late_output() {
        struct DropProbe(Arc<AtomicBool>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let late_drop = Arc::clone(&dropped);
        assert!(bounded_blocking(Duration::from_millis(10), move || {
            std::thread::sleep(Duration::from_millis(50));
            DropProbe(late_drop)
        })
        .await
        .is_err());
        tokio::time::timeout(Duration::from_secs(1), async {
            while !dropped.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("late blocking output must be dropped");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn aggregate_mcp_startup_times_out_a_silent_server() {
        let specs = vec![StdioServerSpec {
            name: "silent".into(),
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "read initialize; sleep 10".into()],
            env: Vec::new(),
        }];

        let error =
            start_tool_registry_with_timeout(&specs, Path::new("/"), Duration::from_millis(25))
                .await
                .err()
                .expect("silent MCP server must time out");
        assert!(error.to_string().contains("silent"));
        assert!(error.to_string().contains("timed out"));
    }

    #[test]
    fn mcp_model_names_are_bounded_safe_and_ordinally_unique() {
        let first = model_tool_name(0, 0, "same tool/工具");
        let other_server = model_tool_name(1, 0, "same tool/工具");
        let other_tool = model_tool_name(0, 1, "same tool/工具");
        assert_eq!(first, "mcp_s0_t0_same_tool___");
        assert_ne!(first, other_server);
        assert_ne!(first, other_tool);
        for name in [first, other_server, other_tool] {
            assert!(name.len() <= MAX_MODEL_TOOL_NAME_BYTES);
            assert!(name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')));
        }
        assert_eq!(
            model_tool_name(31, 1023, &"x".repeat(200)).len(),
            MAX_MODEL_TOOL_NAME_BYTES
        );
    }

    #[test]
    fn tool_call_result_rejects_untyped_content_and_non_object_extensions() {
        for result in [
            serde_json::json!({"content": [{}]}),
            serde_json::json!({"content": [], "structuredContent": []}),
            serde_json::json!({"content": [], "isError": "yes"}),
        ] {
            assert!(matches!(
                ToolCallResult::from_result(&result),
                Err(ToolCallError::InvalidResult)
            ));
        }
    }

    #[test]
    fn mcp_tool_error_payload_is_bounded_before_agent_propagation() {
        let error = finish_tool_result(ToolCallResult {
            content: vec![serde_json::json!({
                "type": "text",
                "text": "x".repeat(100_000)
            })],
            structured_content: None,
            is_error: true,
            meta: None,
        })
        .unwrap_err()
        .to_string();

        assert!(error.len() < 31_000);
        assert!(error.contains("characters omitted"));
        assert!(error.starts_with("MCP tool returned an error:"));
    }
}
