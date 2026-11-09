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
        validate_envelope(message)?;
        let mut encoded = serde_json::to_vec(message).map_err(TransportError::Json)?;
        if encoded.len() > MAX_MESSAGE_BYTES {
            return Err(TransportError::Oversized);
        }
        encoded.push(b'\n');
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
        if encoded.len() > MAX_MESSAGE_BYTES {
            return Err(TransportError::Oversized);
        }
        if read == 0 || encoded.last() != Some(&b'\n') {
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
}
