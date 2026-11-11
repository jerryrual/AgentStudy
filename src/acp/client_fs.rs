//! Client-backed `read_file` override for ACP editor integrations.

use std::time::Duration;

use agent_client_protocol::v1::schema::{
    ReadTextFileRequest, ReadTextFileResponse, SessionId, WriteTextFileRequest,
    WriteTextFileResponse,
};
use agent_client_protocol::v1::ConnectionTo;
use agent_client_protocol::Client;
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};

use crate::tools::fs::{format_client_read, format_write_result, ReadFile, WriteFile};
use crate::tools::{display_path, optional_usize, required_str, Tool, ToolContext};

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
        ReadFile.name()
    }

    fn description(&self) -> &'static str {
        ReadFile.description()
    }

    fn parameters(&self) -> Value {
        ReadFile.parameters()
    }

    fn parallel_safe(&self) -> bool {
        ReadFile.parallel_safe()
    }

    fn summarize(&self, raw: &str) -> String {
        ReadFile.summarize(raw)
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

struct WriteJob {
    request: WriteTextFileRequest,
    response: oneshot::Sender<Result<WriteTextFileResponse, agent_client_protocol::Error>>,
}

/// FIFO queue whose worker outlives a cancelled prompt's tool future.
#[derive(Clone)]
pub(crate) struct AcpWriteDispatcher {
    sender: mpsc::Sender<WriteJob>,
}

impl AcpWriteDispatcher {
    /// Start one connection-scoped worker that completes writes in admission order.
    pub(crate) fn new(
        connection: ConnectionTo<Client>,
    ) -> Result<Self, agent_client_protocol::Error> {
        Self::with_timeout(connection, Duration::from_secs(30))
    }

    fn with_timeout(
        connection: ConnectionTo<Client>,
        request_timeout: Duration,
    ) -> Result<Self, agent_client_protocol::Error> {
        let (sender, mut receiver) = mpsc::channel::<WriteJob>(64);
        let worker_connection = connection.clone();
        connection.spawn(async move {
            while let Some(job) = receiver.recv().await {
                if job.response.is_closed() {
                    continue;
                }
                let request = worker_connection.send_request(job.request).block_task();
                match tokio::time::timeout(request_timeout, request).await {
                    Ok(result) => {
                        let _ = job.response.send(result);
                    }
                    Err(_) => {
                        let _ = job
                            .response
                            .send(Err(agent_client_protocol::Error::internal_error()));
                        break;
                    }
                }
            }
            Ok(())
        })?;
        Ok(Self { sender })
    }

    async fn write(&self, request: WriteTextFileRequest) -> Result<WriteTextFileResponse> {
        let (response, receiver) = oneshot::channel();
        self.sender
            .send(WriteJob { request, response })
            .await
            .map_err(|_| anyhow::anyhow!("ACP client write dispatcher is unavailable"))?;
        receiver
            .await
            .context("ACP client write dispatcher stopped")?
            .map_err(Into::into)
    }
}

/// Forwards one prompt's `write_file` calls through the connection FIFO.
pub(crate) struct AcpWriteFile {
    session_id: SessionId,
    dispatcher: AcpWriteDispatcher,
}

impl AcpWriteFile {
    /// Bind a session ID to the connection's ordered write dispatcher.
    pub(crate) fn new(session_id: SessionId, dispatcher: AcpWriteDispatcher) -> Self {
        Self {
            session_id,
            dispatcher,
        }
    }
}

#[async_trait]
impl Tool for AcpWriteFile {
    fn name(&self) -> &'static str {
        WriteFile.name()
    }

    fn description(&self) -> &'static str {
        WriteFile.description()
    }

    fn parameters(&self) -> Value {
        WriteFile.parameters()
    }

    fn mutates(&self) -> bool {
        true
    }

    fn summarize(&self, raw: &str) -> String {
        WriteFile.summarize(raw)
    }

    async fn run(&self, arguments: &Value, context: &ToolContext) -> Result<String> {
        let path = required_str(arguments, "path")?;
        let content = required_str(arguments, "content")?;
        let full = context.resolve(path);
        let display = display_path(&context.cwd, &full);
        let request = WriteTextFileRequest::new(self.session_id.clone(), &full, content);
        let _: WriteTextFileResponse = self
            .dispatcher
            .write(request)
            .await
            .with_context(|| format!("ACP client cannot write {display}"))?;
        Ok(format_write_result(&display, content))
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
                        &serde_json::json!({"path": "src/main.rs", "offset": 3, "limit": 2}),
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
    async fn forwards_exact_write_request_and_reports_known_facts() {
        let directory = tempfile::tempdir().unwrap();
        let expected = directory.path().join("src/main.rs");
        let client = Client.v1().on_receive_request(
            async move |request: WriteTextFileRequest, responder, _connection| {
                assert_eq!(request.session_id, SessionId::new("session-1"));
                assert_eq!(request.path, expected);
                assert_eq!(request.content, "alpha\nbeta\n");
                responder.respond(WriteTextFileResponse::new())
            },
            agent_client_protocol::on_receive_request!(),
        );
        let cwd = directory.path().to_path_buf();

        Agent
            .v1()
            .connect_with(client, async move |connection| {
                let dispatcher = AcpWriteDispatcher::new(connection).unwrap();
                let tool = AcpWriteFile::new(SessionId::new("session-1"), dispatcher);
                assert!(tool.mutates());
                assert!(!tool.parallel_safe());
                let output = tool
                    .run(
                        &serde_json::json!({
                            "path": "src/main.rs",
                            "content": "alpha\nbeta\n"
                        }),
                        &context(cwd),
                    )
                    .await
                    .unwrap();
                assert_eq!(output, "Wrote src/main.rs: 2 lines, 11 bytes.");
                Ok::<(), ProtocolError>(())
            })
            .await
            .unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelled_write_waiter_does_not_release_later_writes() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let first_entered = Arc::new(tokio::sync::Notify::new());
        let release_first = Arc::new(tokio::sync::Notify::new());
        let first_completed = Arc::new(AtomicBool::new(false));
        let handler_entered = Arc::clone(&first_entered);
        let handler_release = Arc::clone(&release_first);
        let handler_completed = Arc::clone(&first_completed);
        let client = Client.v1().on_receive_request(
            async move |request: WriteTextFileRequest, responder, _connection| {
                if request.content == "old" {
                    handler_entered.notify_one();
                    handler_release.notified().await;
                    handler_completed.store(true, Ordering::SeqCst);
                } else {
                    assert!(
                        handler_completed.load(Ordering::SeqCst),
                        "later write reached the client before cancelled write completed"
                    );
                }
                responder.respond(WriteTextFileResponse::new())
            },
            agent_client_protocol::on_receive_request!(),
        );

        Agent
            .v1()
            .connect_with(client, async move |connection| {
                let dispatcher = AcpWriteDispatcher::new(connection).unwrap();
                let first_dispatcher = dispatcher.clone();
                let first = tokio::spawn(async move {
                    first_dispatcher
                        .write(WriteTextFileRequest::new("session-1", "/tmp/file", "old"))
                        .await
                });
                first_entered.notified().await;
                first.abort();

                let second_dispatcher = dispatcher.clone();
                let second = tokio::spawn(async move {
                    second_dispatcher
                        .write(WriteTextFileRequest::new("session-1", "/tmp/file", "new"))
                        .await
                });
                tokio::task::yield_now().await;
                assert!(!second.is_finished());
                release_first.notify_one();
                second.await.unwrap().unwrap();
                Ok::<(), ProtocolError>(())
            })
            .await
            .unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelled_queued_write_is_never_sent() {
        let first_entered = Arc::new(tokio::sync::Notify::new());
        let release_first = Arc::new(tokio::sync::Notify::new());
        let received = Arc::new(Mutex::new(Vec::new()));
        let handler_entered = Arc::clone(&first_entered);
        let handler_release = Arc::clone(&release_first);
        let handler_received = Arc::clone(&received);
        let client = Client.v1().on_receive_request(
            async move |request: WriteTextFileRequest, responder, _connection| {
                handler_received.lock().await.push(request.content.clone());
                if request.content == "old" {
                    handler_entered.notify_one();
                    handler_release.notified().await;
                }
                responder.respond(WriteTextFileResponse::new())
            },
            agent_client_protocol::on_receive_request!(),
        );

        Agent
            .v1()
            .connect_with(client, async move |connection| {
                let dispatcher = AcpWriteDispatcher::new(connection).unwrap();
                let first_dispatcher = dispatcher.clone();
                let first = tokio::spawn(async move {
                    first_dispatcher
                        .write(WriteTextFileRequest::new("session-1", "/tmp/file", "old"))
                        .await
                });
                first_entered.notified().await;

                let (response, cancelled_waiter) = oneshot::channel();
                dispatcher
                    .sender
                    .send(WriteJob {
                        request: WriteTextFileRequest::new("session-1", "/tmp/file", "cancelled"),
                        response,
                    })
                    .await
                    .unwrap();
                drop(cancelled_waiter);
                release_first.notify_one();
                first.await.unwrap().unwrap();
                dispatcher
                    .write(WriteTextFileRequest::new("session-1", "/tmp/file", "new"))
                    .await
                    .unwrap();

                assert_eq!(&*received.lock().await, &["old", "new"]);
                Ok::<(), ProtocolError>(())
            })
            .await
            .unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn write_timeout_fuses_dispatcher_before_later_requests() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let received = Arc::new(AtomicUsize::new(0));
        let handler_received = Arc::clone(&received);
        let client = Client.v1().on_receive_request(
            async move |_request: WriteTextFileRequest, responder, _connection| {
                handler_received.fetch_add(1, Ordering::SeqCst);
                std::future::pending::<()>().await;
                responder.respond(WriteTextFileResponse::new())
            },
            agent_client_protocol::on_receive_request!(),
        );

        Agent
            .v1()
            .connect_with(client, async move |connection| {
                let dispatcher =
                    AcpWriteDispatcher::with_timeout(connection, Duration::from_millis(10))
                        .unwrap();
                assert!(dispatcher
                    .write(WriteTextFileRequest::new("session-1", "/tmp/file", "old"))
                    .await
                    .is_err());
                assert!(dispatcher
                    .write(WriteTextFileRequest::new("session-1", "/tmp/file", "new"))
                    .await
                    .is_err());
                assert_eq!(received.load(Ordering::SeqCst), 1);
                Ok::<(), ProtocolError>(())
            })
            .await
            .unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn write_errors_do_not_fall_back_to_local_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("local.txt");
        std::fs::write(&path, "original").unwrap();
        let client = Client.v1().on_receive_request(
            async |_request: WriteTextFileRequest, responder, _connection| {
                responder.respond_with_error(ProtocolError::internal_error())
            },
            agent_client_protocol::on_receive_request!(),
        );
        let cwd = directory.path().to_path_buf();

        Agent
            .v1()
            .connect_with(client, async move |connection| {
                let dispatcher = AcpWriteDispatcher::new(connection).unwrap();
                let tool = AcpWriteFile::new(SessionId::new("session-1"), dispatcher);
                let error = tool
                    .run(
                        &serde_json::json!({"path": "local.txt", "content": "replacement"}),
                        &context(cwd),
                    )
                    .await
                    .unwrap_err();
                assert!(error
                    .to_string()
                    .contains("ACP client cannot write local.txt"));
                assert_eq!(std::fs::read_to_string(path).unwrap(), "original");
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
                    .run(&serde_json::json!({"path": "local.txt"}), &context(cwd))
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
