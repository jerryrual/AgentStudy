//! Strict projection of stable ACP stdio MCP server declarations.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming MCP runtime wiring")
)]

use std::collections::HashSet;
use std::path::PathBuf;

use agent_client_protocol::v1::schema::McpServer;

const MAX_SERVERS: usize = 32;
const MAX_ARGS: usize = 256;
const MAX_ENV: usize = 256;

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
}
