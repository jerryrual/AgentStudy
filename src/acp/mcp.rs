//! Strict projection of stable ACP stdio MCP server declarations.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming MCP runtime wiring")
)]

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use agent_client_protocol::v1::schema::McpServer;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

const MAX_SERVERS: usize = 32;
const MAX_ARGS: usize = 256;
const MAX_ENV: usize = 256;
const MAX_MESSAGE_BYTES: usize = 4 * 1024 * 1024;
const MCP_PROTOCOL_VERSION: &str = "2025-06-18";

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

pub(crate) struct StdioMcpClient {
    transport: StdioJsonRpc,
    next_id: u64,
    usable: bool,
}

impl StdioMcpClient {
    pub(crate) fn from_process(process: &mut StdioServerProcess) -> std::io::Result<Self> {
        Ok(Self {
            transport: StdioJsonRpc::from_process(process)?,
            next_id: 1,
            usable: true,
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
            self.usable = false;
            return Err(RequestError::MismatchedId);
        }
        match (object.get("result"), object.get("error")) {
            (Some(result), None) => Ok(result.clone()),
            (None, Some(error)) => Err(RequestError::Remote(error.clone())),
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
        if let Err(error) = self.transport.send(&notification).await {
            self.usable = false;
            return Err(RequestError::Transport(error));
        }
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
            .map_err(InitializeError::Request)?;
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
        Ok(initialization)
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

#[cfg(test)]
mod tests {
    use agent_client_protocol::v1::schema::{EnvVariable, McpServerHttp, McpServerStdio};

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
}
