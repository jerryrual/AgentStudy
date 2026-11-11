//! Client-backed `read_file` override for ACP editor integrations.

use agent_client_protocol::v1::schema::{ReadTextFileRequest, ReadTextFileResponse, SessionId};
use agent_client_protocol::v1::ConnectionTo;
use agent_client_protocol::Client;
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::tools::fs::format_client_read;
use crate::tools::{
    display_path, json_path_property, optional_usize, required_str, Tool, ToolContext,
};

/// Forwards one prompt's `read_file` calls to the connected ACP client.
pub(crate) struct AcpReadFile {
    session_id: SessionId,
    connection: ConnectionTo<Client>,
}

impl AcpReadFile {
    pub(crate) fn new(session_id: SessionId, connection: ConnectionTo<Client>) -> Self {
        Self {
            session_id,
            connection,
        }
    }
}

#[async_trait]
impl Tool for AcpReadFile {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read a UTF-8 text file and return it with 1-based line numbers. \
         Use offset/limit to page through large files."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": json_path_property("File path, relative to the working directory or absolute."),
                "offset": {"type": "integer", "minimum": 1, "description": "First line to return (1-based, default 1)."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5000, "description": "Maximum number of lines (default 2000)."}
            },
            "required": ["path"]
        })
    }

    fn parallel_safe(&self) -> bool {
        true
    }

    fn summarize(&self, raw: &str) -> String {
        let Ok(arguments) = serde_json::from_str::<Value>(raw) else {
            return raw.chars().take(80).collect();
        };
        let path = arguments
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("<path>");
        let offset = arguments.get("offset").and_then(Value::as_u64).unwrap_or(1);
        if offset > 1 {
            format!("{path} (from line {offset})")
        } else {
            path.to_string()
        }
    }

    async fn run(&self, arguments: &Value, context: &ToolContext) -> Result<String> {
        let path = required_str(arguments, "path")?;
        let offset = optional_usize(arguments, "offset").unwrap_or(1).max(1);
        let limit = optional_usize(arguments, "limit")
            .unwrap_or(2000)
            .clamp(1, 5000);
        let full = context.resolve(path);
        let display = display_path(&context.cwd, &full);
        let request = ReadTextFileRequest::new(self.session_id.clone(), &full)
            .line(u32::try_from(offset).unwrap_or(u32::MAX))
            .limit(u32::try_from(limit).unwrap_or(5000));
        let response: ReadTextFileResponse = self
            .connection
            .send_request(request)
            .block_task()
            .await
            .with_context(|| format!("ACP client cannot read {display}"))?;
        format_client_read(&display, offset, limit, &response.content)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;

    use agent_client_protocol::{Agent, Error as ProtocolError};
    use tokio::sync::Mutex;

    use super::*;
    use crate::interaction::NoAsker;
    use crate::tools::TodoItem;

    fn context(cwd: PathBuf) -> ToolContext {
        ToolContext {
            cwd,
            timeout: Duration::from_secs(1),
            asker: Arc::new(NoAsker),
            todos: Arc::new(Mutex::new(Vec::<TodoItem>::new())),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn forwards_absolute_range_and_formats_client_content() {
        let directory = tempfile::tempdir().unwrap();
        let expected = directory.path().join("src/main.rs");
        let client = Client.v1().on_receive_request(
            async move |request: ReadTextFileRequest, responder, _connection| {
                assert_eq!(request.session_id, SessionId::new("session-1"));
                assert_eq!(request.path, expected);
                assert_eq!(request.line, Some(3));
                assert_eq!(request.limit, Some(2));
                responder.respond(ReadTextFileResponse::new("alpha\nbeta\n"))
            },
            agent_client_protocol::on_receive_request!(),
        );
        let cwd = directory.path().to_path_buf();

        Agent
            .v1()
            .connect_with(client, async move |connection| {
                let tool = AcpReadFile::new(SessionId::new("session-1"), connection);
                let output = tool
                    .run(
                        &json!({"path": "src/main.rs", "offset": 3, "limit": 2}),
                        &context(cwd),
                    )
                    .await
                    .unwrap();
                assert!(output.contains("client lines 3-4"));
                assert!(output.contains("3\talpha\n4\tbeta"));
                Ok::<(), ProtocolError>(())
            })
            .await
            .unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn client_errors_do_not_fall_back_to_local_files() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("local.txt"), "local fallback").unwrap();
        let client = Client.v1().on_receive_request(
            async |_request: ReadTextFileRequest, responder, _connection| {
                responder.respond_with_error(ProtocolError::internal_error())
            },
            agent_client_protocol::on_receive_request!(),
        );
        let cwd = directory.path().to_path_buf();

        Agent
            .v1()
            .connect_with(client, async move |connection| {
                let tool = AcpReadFile::new(SessionId::new("session-1"), connection);
                let error = tool
                    .run(&json!({"path": "local.txt"}), &context(cwd))
                    .await
                    .unwrap_err();
                assert!(error
                    .to_string()
                    .contains("ACP client cannot read local.txt"));
                Ok::<(), ProtocolError>(())
            })
            .await
            .unwrap();
    }
}
