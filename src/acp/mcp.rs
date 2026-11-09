//! Strict projection of stable ACP stdio MCP server declarations.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming MCP runtime wiring")
)]

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::v1::schema::McpServer;
use anyhow::bail;
use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::tools::{truncate_output, Tool, ToolContext};

const MAX_SERVERS: usize = 32;
const MAX_ARGS: usize = 256;
const MAX_ENV: usize = 256;
const MAX_MESSAGE_BYTES: usize = 4 * 1024 * 1024;
const MCP_PROTOCOL_VERSION: &str = "2025-06-18";
const MAX_TOOL_PAGES: usize = 64;
const MAX_DISCOVERED_TOOLS: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StdioServerSpec {
    pub(crate) name: String,
    pub(crate) command: PathBuf,
    pub(crate) args: Vec<String>,
    pub(crate) env: Vec<(String, String)>,
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

pub(crate) struct StdioServerProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Option<ChildStdout>,
}

impl StdioServerProcess {
    pub(crate) fn spawn(spec: &StdioServerSpec) -> std::io::Result<Self> {
        let mut command = Command::new(&spec.command);
        command
            .args(&spec.args)
            .envs(spec.env.iter().cloned())
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
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "MCP stdio I/O failed: {error}"),
            Self::Json(error) => write!(formatter, "invalid MCP JSON: {error}"),
            Self::Oversized => formatter.write_str("MCP message exceeds the size limit"),
            Self::Truncated => formatter.write_str("MCP message is not newline terminated"),
            Self::InvalidEnvelope => formatter.write_str("invalid MCP JSON-RPC envelope"),
        }
    }
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Oversized | Self::Truncated | Self::InvalidEnvelope => None,
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
    let integer_code = error.get("code").and_then(Value::as_i64).is_some()
        || error.get("code").and_then(Value::as_u64).is_some();
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

pub(crate) struct McpServerHandle {
    client: Mutex<StdioMcpClient>,
    _process: StdioServerProcess,
}

pub(crate) fn tool_adapters(
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
        .map(|tool| Box::new(McpTool::new(tool, Arc::clone(&handle))) as Box<dyn Tool>)
        .collect()
}

struct McpTool {
    exposed_name: String,
    remote_name: String,
    description: String,
    parameters: Value,
    handle: Arc<McpServerHandle>,
}

impl McpTool {
    fn new(tool: DiscoveredTool, handle: Arc<McpServerHandle>) -> Self {
        let description = tool
            .description
            .or(tool.title)
            .unwrap_or_else(|| "Tool provided by an MCP server".to_string());
        Self {
            exposed_name: tool.name.clone(),
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
    use agent_client_protocol::v1::schema::{EnvVariable, McpServerHttp, McpServerStdio};

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
            "\"error\":{\"code\":-1,\"message\":\"no\"}}'; ",
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
            ToolRegistry::try_with_tools(tool_adapters(process, client, discovered)).unwrap();
        let tool = registry.get("echo").unwrap();

        assert_eq!(tool.name(), "mcp_tool");
        assert_eq!(tool.model_name(), "echo");
        assert_eq!(tool.model_description(), "Echoes");
        assert_eq!(tool.parameters(), serde_json::json!({"type": "object"}));
        assert!(tool.mutates());
        assert!(!tool.parallel_safe());
        assert_eq!(registry.model_names(), vec!["echo"]);
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
