//! The agent loop: send the conversation to the model, execute the tools it
//! asks for, feed the results back, and repeat until it answers.

pub mod events;
pub mod injection;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use futures_util::stream::{FuturesUnordered, StreamExt};
use tokio::sync::Mutex;

pub use events::{EventSink, NullSink};

use crate::interaction::{ApprovalDecision, Approver, Asker};
use crate::model::{Message, ModelClient, ToolDef, Usage};
use crate::tools::{truncate_output, TodoItem, ToolContext, ToolRegistry};

/// Which built-in agent to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AgentKind {
    /// Full coding agent with mutating tools.
    #[default]
    Code,
    /// Read-only Q&A agent.
    Qa,
}

impl AgentKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "code" | "coding" | "" => Some(Self::Code),
            "qa" | "q&a" | "ask" | "chat" => Some(Self::Qa),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Code => "code",
            Self::Qa => "qa",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Code => "Code Agent",
            Self::Qa => "Q&A Agent",
        }
    }
}

/// Result of one user turn (possibly many model and tool round trips).
#[derive(Debug, Clone, Default)]
pub struct TurnResult {
    pub text: String,
    pub usage: Usage,
    pub(crate) latest_usage: Usage,
    pub tool_calls: usize,
    pub rounds: usize,
    pub duration: Duration,
}

const CODE_SYSTEM_PROMPT: &str = r#"You are iCode, an autonomous software-engineering agent running in a terminal.

You help the user understand and change their codebase. Work directly, using the tools available to you, and report what you did.

Operating rules:
- Investigate before you edit: list directories, search, and read the relevant files.
- Prefer read_file, list_dir, grep, glob and edit_file over shell equivalents when inspecting or editing source files.
- Use bash for building, testing, running programs and git. Keep commands non-interactive.
- Keep changes minimal and consistent with the surrounding code. Do not reformat unrelated code.
- Verify your work by running the project's formatters, linters and tests when they exist.
- Never print secrets, tokens or private keys. Do not run destructive commands unless explicitly asked.
- Track multi-step work with todo_write and mark items complete as you go.
- Reply in Markdown. Be concise and concrete, and cite files by path when useful."#;

const QA_SYSTEM_PROMPT: &str = r#"You are iCode's Q&A agent, a read-only assistant running in a terminal.

Answer the user's questions about the current project. You can inspect files and search the codebase, but you must not modify files, install packages or run commands: this agent is read-only and the runtime does not expose mutating tools.

Operating rules:
- Investigate before you answer: read the relevant files and search for definitions.
- Use read_file, list_dir, grep and glob.
- If a question needs a change, explain what you would change and why instead of changing it.
- Reply in Markdown. Be concise and concrete, and cite files by path when useful."#;

/// Options for one agent instance.
#[derive(Debug, Clone)]
pub struct AgentOptions {
    pub kind: AgentKind,
    pub cwd: PathBuf,
    pub max_rounds: usize,
    pub tool_timeout: Duration,
}

impl AgentOptions {
    pub fn new(kind: AgentKind, cwd: impl Into<PathBuf>) -> Self {
        Self {
            kind,
            cwd: cwd.into(),
            max_rounds: 40,
            tool_timeout: Duration::from_secs(120),
        }
    }
}

/// The tool-calling agent.
pub struct Agent {
    client: Arc<ModelClient>,
    tools: Arc<ToolRegistry>,
    approver: Arc<dyn Approver>,
    asker: Arc<dyn Asker>,
    options: AgentOptions,
    project_memory: crate::memory::ProjectMemory,
    additional_directories: Vec<PathBuf>,
    todos: Arc<Mutex<Vec<TodoItem>>>,
}

impl Agent {
    pub fn new(
        client: Arc<ModelClient>,
        tools: Arc<ToolRegistry>,
        approver: Arc<dyn Approver>,
        asker: Arc<dyn Asker>,
        options: AgentOptions,
    ) -> Self {
        let project_memory = crate::memory::load_project_memory(&options.cwd);
        Self {
            client,
            tools,
            approver,
            asker,
            options,
            project_memory,
            additional_directories: Vec::new(),
            todos: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub(crate) fn with_additional_directories(mut self, directories: Vec<PathBuf>) -> Self {
        self.additional_directories = directories;
        self
    }

    pub fn model(&self) -> &str {
        self.client.model()
    }

    pub fn kind(&self) -> AgentKind {
        self.options.kind
    }

    pub(crate) fn should_force_compression(&self, result: &TurnResult) -> bool {
        self.client.should_force_compression(result.latest_usage)
    }

    /// The system prompt for this agent, anchored at the working directory.
    pub fn system_prompt(&self) -> String {
        let base = match self.options.kind {
            AgentKind::Code => CODE_SYSTEM_PROMPT,
            AgentKind::Qa => QA_SYSTEM_PROMPT,
        };
        let mut prompt = format!(
            "{base}\n\nCurrent working directory: {}\nToday's date: {}",
            self.options.cwd.display(),
            today()
        );
        if !self.additional_directories.is_empty() {
            prompt.push_str(
                "\n\nAdditional workspace roots (in client order; relative paths still resolve from the current working directory):",
            );
            for directory in &self.additional_directories {
                prompt.push_str("\n- ");
                prompt.push_str(&format!("{directory:?}"));
            }
        }
        if !self.project_memory.prompt.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(&self.project_memory.prompt);
        }
        prompt
    }

    /// Tools visible to the model for this agent kind.
    ///
    /// The Q&A agent only sees tools that cannot change the workspace, so a
    /// prompt-injected model cannot mutate files even in headless mode.
    pub fn tool_defs(&self) -> Vec<ToolDef> {
        self.visible_specs()
    }

    fn visible_specs(&self) -> Vec<ToolDef> {
        let specs = self.tools.specs();
        if self.options.kind == AgentKind::Code {
            return specs;
        }
        specs
            .into_iter()
            .filter(|spec| {
                self.tools
                    .get(&spec.name)
                    .is_some_and(|tool| !tool.mutates())
            })
            .collect()
    }

    /// Run one user turn, mutating `messages` with new assistant and tool turns.
    ///
    /// Each model call streams through `sink`; tool calls are executed
    /// sequentially, with approval requested for mutating tools. The loop ends
    /// when the model answers without tool calls or the round budget is spent.
    pub async fn run(
        &self,
        messages: &mut Vec<Message>,
        sink: &dyn EventSink,
    ) -> Result<TurnResult> {
        let (_queue, injections) = injection::channel();
        self.run_with_injections(messages, sink, &injections).await
    }

    /// Run one user turn while accepting same-turn user injections.
    pub async fn run_with_injections(
        &self,
        messages: &mut Vec<Message>,
        sink: &dyn EventSink,
        injections: &injection::InjectionSource,
    ) -> Result<TurnResult> {
        let _seal_injections = injections.seal_on_drop();
        let started = Instant::now();
        let mut usage = Usage::default();
        let mut tool_calls = 0usize;
        let mut rounds = 0usize;

        self.ensure_system_prompt(messages);
        if let Some(warning) = &self.project_memory.warning {
            sink.info(warning);
        }
        sink.turn_started();

        loop {
            if rounds >= self.options.max_rounds {
                bail!(
                    "reached the maximum of {} model calls for one turn; \
                     increase max_tool_rounds or continue the conversation",
                    self.options.max_rounds
                );
            }
            rounds += 1;

            if let Some(reservation) = injections.reserve() {
                sink.persist_injected_user(reservation.message())?;
                messages.push(reservation.commit());
            }
            sink.persist_recovery_checkpoint(messages)?;
            let persistent = self
                .client
                .complete_persistent(messages, &self.visible_specs(), sink)
                .await?;
            if persistent.auxiliary_usage != Usage::default() {
                usage.add(persistent.auxiliary_usage);
            }
            let completion = persistent.completion;
            let latest_usage = completion.usage;
            usage.add(completion.usage);
            sink.usage(&completion.usage);

            let message = completion.message;
            let calls = message.tool_calls.clone();
            let text = message.content.clone();
            let finish_reason = completion.finish_reason.clone();
            if message.is_empty() && calls.is_empty() {
                let reason = if finish_reason.as_deref() == Some("length") {
                    "the model response was truncated (max_tokens reached)"
                } else {
                    "the model returned an empty response"
                };
                bail!("{reason}");
            }

            let assistant = Message::assistant(message);
            messages.push(assistant.clone());
            sink.message_added(&assistant);

            if calls.is_empty() {
                if !injections.try_finish() {
                    continue;
                }
                let result = TurnResult {
                    text: text.unwrap_or_default(),
                    usage,
                    latest_usage,
                    tool_calls,
                    rounds,
                    duration: started.elapsed(),
                };
                sink.turn_finished(&result);
                return Ok(result);
            }

            let context = ToolContext {
                cwd: self.options.cwd.clone(),
                timeout: self.options.tool_timeout,
                asker: self.asker.clone(),
                todos: self.todos.clone(),
            };

            tool_calls += calls.len();
            let tool_messages = self
                .execute_tool_batch(&calls, &context, messages, sink)
                .await?;
            for message in tool_messages {
                messages.push(message.clone());
                sink.message_added(&message);
            }
        }
    }

    /// Execute one model tool-call batch. Maximal contiguous runs of tools that
    /// explicitly opt into parallel execution overlap; every other call is a
    /// sequential barrier. Each completed result is recovery-checkpointed, while
    /// the returned batch always remains in provider call order.
    async fn execute_tool_batch(
        &self,
        calls: &[crate::model::ToolCall],
        context: &ToolContext,
        message_prefix: &[Message],
        sink: &dyn EventSink,
    ) -> Result<Vec<Message>> {
        let mut messages = vec![None; calls.len()];
        let mut index = 0;
        while index < calls.len() {
            if !self.is_parallel_safe(&calls[index]) {
                let call = &calls[index];
                let result = self.execute_tool(call, context, sink).await;
                messages[index] = Some(Message::tool(call.id.clone(), call.name.clone(), result));
                self.persist_tool_result_checkpoint(message_prefix, &messages, sink)?;
                index += 1;
                continue;
            }

            let start = index;
            while index < calls.len() && self.is_parallel_safe(&calls[index]) {
                index += 1;
            }
            let mut executions = FuturesUnordered::new();
            for (offset, call) in calls[start..index].iter().enumerate() {
                let call_index = start + offset;
                let summary = self.tool_summary(call);
                sink.tool_call_started(call, &summary);
                executions.push(async move {
                    let execution = self.execute_tool_inner(call, context, &summary).await;
                    (call_index, call, summary, execution)
                });
            }
            let mut finishes = vec![None; index - start];

            while let Some((call_index, call, summary, (ok, duration, display, result))) =
                executions.next().await
            {
                finishes[call_index - start] = Some((ok, duration, summary, display));
                messages[call_index] =
                    Some(Message::tool(call.id.clone(), call.name.clone(), result));
                self.persist_tool_result_checkpoint(message_prefix, &messages, sink)?;
            }
            for (offset, finish) in finishes.into_iter().enumerate() {
                let (ok, duration, summary, display) = finish.ok_or_else(|| {
                    anyhow::anyhow!("parallel tool ended without finish metadata")
                })?;
                sink.tool_call_finished(&calls[start + offset], ok, duration, &summary, &display);
            }
        }
        messages
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow::anyhow!("tool batch ended without every result"))
    }

    fn persist_tool_result_checkpoint(
        &self,
        message_prefix: &[Message],
        tool_results: &[Option<Message>],
        sink: &dyn EventSink,
    ) -> Result<()> {
        let mut checkpoint = Vec::with_capacity(message_prefix.len() + tool_results.len());
        checkpoint.extend_from_slice(message_prefix);
        checkpoint.extend(tool_results.iter().filter_map(Clone::clone));
        sink.publish_recovery_checkpoint(checkpoint)
            .context("failed to checkpoint completed tool results")?;
        Ok(())
    }

    fn is_parallel_safe(&self, call: &crate::model::ToolCall) -> bool {
        self.tools
            .get(&call.name)
            .is_some_and(|tool| tool.parallel_safe() && !tool.mutates())
    }

    fn tool_summary(&self, call: &crate::model::ToolCall) -> String {
        self.tools
            .get(&call.name)
            .map(|tool| tool.summarize(&call.arguments))
            .unwrap_or_else(|| call.name.clone())
    }

    async fn execute_tool(
        &self,
        call: &crate::model::ToolCall,
        context: &ToolContext,
        sink: &dyn EventSink,
    ) -> String {
        let summary = self.tool_summary(call);
        sink.tool_call_started(call, &summary);
        let (ok, duration, display, result) =
            self.execute_tool_inner(call, context, &summary).await;
        sink.tool_call_finished(call, ok, duration, &summary, &display);
        result
    }

    async fn execute_tool_inner(
        &self,
        call: &crate::model::ToolCall,
        context: &ToolContext,
        summary: &str,
    ) -> (bool, Duration, String, String) {
        let started = Instant::now();
        let arguments = match call.parse_arguments() {
            Ok(arguments) => arguments,
            Err(error) => {
                let message = format!("Error: invalid JSON arguments for {}: {error}", call.name);
                return (false, started.elapsed(), message.clone(), message);
            }
        };

        let Some(tool) = self.tools.get(&call.name) else {
            let message = format!(
                "Error: unknown tool {:?}; available tools: {}",
                call.name,
                self.tools
                    .specs()
                    .iter()
                    .map(|spec| spec.name.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            return (false, started.elapsed(), message.clone(), message);
        };

        if self.options.kind == AgentKind::Qa && tool.mutates() {
            let message = format!(
                "Error: {} is not available to the read-only Q&A agent.",
                call.name
            );
            return (false, started.elapsed(), message.clone(), message);
        }

        if tool.mutates()
            && self.approver.approve_tool(call, summary).await == ApprovalDecision::Deny
        {
            let message = "Error: the user denied this operation.".to_string();
            return (false, started.elapsed(), message.clone(), message);
        }

        match tool.run(&arguments, context).await {
            Ok(output) => {
                let model_output = truncate_output(&output);
                (true, started.elapsed(), output, model_output)
            }
            Err(error) => {
                let message = format!("Error: {error:#}");
                (false, started.elapsed(), message.clone(), message)
            }
        }
    }

    fn ensure_system_prompt(&self, messages: &mut Vec<Message>) {
        let prompt = Message::system(self.system_prompt());
        match messages.first_mut() {
            Some(Message::System { content }) => *content = self.system_prompt(),
            _ => messages.insert(0, prompt),
        }
    }
}

fn today() -> String {
    // A dependency-free UTC date, sufficient for prompt context.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    let days = secs / 86_400;
    civil_from_days(days as i64)
}

fn civil_from_days(days: i64) -> String {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// Convert an [`anyhow::Error`] into a one-line message for the UI.
pub fn error_line(error: &anyhow::Error) -> String {
    format!("{error:#}")
}

/// Convenience helper for callers that build a client from a profile.
pub fn client_from_profile(profile: &crate::config::ModelProfile) -> Result<Arc<ModelClient>> {
    let client = ModelClient::from_profile(profile).context("failed to create model client")?;
    Ok(Arc::new(client))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{mpsc as std_mpsc, Arc, Mutex as StdMutex};

    use anyhow::{bail, Result};
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use tokio::sync::{oneshot, Barrier, Notify};

    use crate::config::ModelProfile;
    use crate::interaction::{AllowAll, ApprovalRequest, NoAsker};
    use crate::model::ToolCall;
    use crate::tools::{Tool, ToolContext};

    struct RecordingCheckpointSink {
        snapshots: StdMutex<Vec<Vec<Message>>>,
    }

    impl EventSink for RecordingCheckpointSink {
        fn publish_recovery_checkpoint(&self, messages: Vec<Message>) -> Result<bool> {
            self.snapshots.lock().unwrap().push(messages);
            Ok(true)
        }
    }

    struct SignalingCheckpointSink {
        snapshots: StdMutex<Vec<Vec<Message>>>,
        published: Notify,
    }

    impl EventSink for SignalingCheckpointSink {
        fn publish_recovery_checkpoint(&self, messages: Vec<Message>) -> Result<bool> {
            self.snapshots.lock().unwrap().push(messages);
            self.published.notify_waiters();
            Ok(true)
        }
    }

    struct InjectionOrderSink {
        order: Arc<StdMutex<Vec<&'static str>>>,
        persisted: StdMutex<Vec<Message>>,
        checkpoints: StdMutex<Vec<Vec<Message>>>,
        fail_persistence: bool,
    }

    impl InjectionOrderSink {
        fn new(fail_persistence: bool) -> Self {
            Self {
                order: Arc::new(StdMutex::new(Vec::new())),
                persisted: StdMutex::new(Vec::new()),
                checkpoints: StdMutex::new(Vec::new()),
                fail_persistence,
            }
        }
    }

    impl EventSink for InjectionOrderSink {
        fn persist_injected_user(&self, message: &Message) -> Result<()> {
            self.order.lock().unwrap().push("injection");
            if self.fail_persistence {
                bail!("simulated injected-user persistence failure");
            }
            self.persisted.lock().unwrap().push(message.clone());
            Ok(())
        }

        fn persist_recovery_checkpoint(&self, messages: &[Message]) -> Result<bool> {
            self.order.lock().unwrap().push("checkpoint");
            self.checkpoints.lock().unwrap().push(messages.to_vec());
            Ok(true)
        }
    }

    fn injection_test_agent(base_url: String) -> Agent {
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_key = "test-key".to_string();
        profile.base_url = base_url;
        profile.model = "mock-model".to_string();
        profile.max_tokens = None;
        let client = Arc::new(ModelClient::from_profile(&profile).unwrap());
        Agent::new(
            client,
            Arc::new(ToolRegistry::with_tools(Vec::new())),
            Arc::new(AllowAll),
            Arc::new(NoAsker),
            AgentOptions::new(AgentKind::Code, "/tmp"),
        )
    }

    fn read_model_request(stream: &mut TcpStream) -> serde_json::Value {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut request_line = String::new();
        reader.read_line(&mut request_line).unwrap();
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" || line.is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                if name.eq_ignore_ascii_case("content-length") {
                    content_length = value.trim().parse().unwrap();
                }
            }
        }
        let mut body = vec![0; content_length];
        reader.read_exact(&mut body).unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    fn write_model_response(stream: &mut TcpStream, text: &str) {
        let text_chunk = json!({"choices": [{"delta": {"content": text}}]}).to_string();
        let done_chunk = json!({
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })
        .to_string();
        let response_body = format!("data: {text_chunk}\n\ndata: {done_chunk}\n\ndata: [DONE]\n\n");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        stream.write_all(response.as_bytes()).unwrap();
        stream.flush().unwrap();
    }

    fn one_shot_model_server(
        order: Arc<StdMutex<Vec<&'static str>>>,
    ) -> (String, std::thread::JoinHandle<serde_json::Value>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            order.lock().unwrap().push("provider");
            let request = read_model_request(&mut stream);
            write_model_response(&mut stream, "done");
            request
        });
        (format!("http://{address}/v1"), handle)
    }

    type BlockedOneShotServer = (
        String,
        oneshot::Receiver<()>,
        std_mpsc::Sender<()>,
        std::thread::JoinHandle<serde_json::Value>,
    );

    fn blocked_one_shot_model_server() -> BlockedOneShotServer {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (request_seen_tx, request_seen_rx) = oneshot::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_model_request(&mut stream);
            let _ = request_seen_tx.send(());
            release_rx.recv().unwrap();
            request
        });
        (
            format!("http://{address}/v1"),
            request_seen_rx,
            release_tx,
            handle,
        )
    }

    type BlockedServer = (
        String,
        oneshot::Receiver<()>,
        std_mpsc::Sender<()>,
        std::thread::JoinHandle<Vec<serde_json::Value>>,
    );

    fn blocked_two_shot_model_server() -> BlockedServer {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (first_seen_tx, first_seen_rx) = oneshot::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        let handle = std::thread::spawn(move || {
            let (mut first_stream, _) = listener.accept().unwrap();
            let first = read_model_request(&mut first_stream);
            let _ = first_seen_tx.send(());
            release_rx.recv().unwrap();
            write_model_response(&mut first_stream, "first answer");

            let (mut second_stream, _) = listener.accept().unwrap();
            let second = read_model_request(&mut second_stream);
            write_model_response(&mut second_stream, "second answer");
            vec![first, second]
        });
        (
            format!("http://{address}/v1"),
            first_seen_rx,
            release_tx,
            handle,
        )
    }

    struct ParallelProbe {
        barrier: Arc<Barrier>,
        completed: Arc<StdMutex<Vec<String>>>,
    }

    #[async_trait]
    impl Tool for ParallelProbe {
        fn name(&self) -> &'static str {
            "probe"
        }

        fn description(&self) -> &'static str {
            "test probe"
        }

        fn parameters(&self) -> Value {
            json!({"type": "object"})
        }

        fn parallel_safe(&self) -> bool {
            true
        }

        fn summarize(&self, raw: &str) -> String {
            raw.to_string()
        }

        async fn run(&self, arguments: &Value, _context: &ToolContext) -> Result<String> {
            let id = arguments["id"].as_str().unwrap().to_string();
            let delay = arguments["delay_ms"].as_u64().unwrap_or_default();
            self.barrier.wait().await;
            tokio::time::sleep(Duration::from_millis(delay)).await;
            self.completed.lock().unwrap().push(id.clone());
            if arguments["fail"].as_bool().unwrap_or(false) {
                bail!("{id} failed");
            }
            Ok(id)
        }
    }

    struct ToolIdentitySink {
        starts: StdMutex<Vec<ToolCall>>,
        finishes: StdMutex<Vec<ToolCall>>,
    }

    impl EventSink for ToolIdentitySink {
        fn tool_call_started(&self, call: &ToolCall, _summary: &str) {
            self.starts.lock().unwrap().push(call.clone());
        }

        fn tool_call_finished(
            &self,
            call: &ToolCall,
            _ok: bool,
            _duration: Duration,
            _summary: &str,
            _output: &str,
        ) {
            self.finishes.lock().unwrap().push(call.clone());
        }
    }

    struct MutatingProbe;

    #[async_trait]
    impl Tool for MutatingProbe {
        fn name(&self) -> &'static str {
            "mutating_probe"
        }

        fn description(&self) -> &'static str {
            "test mutating probe"
        }

        fn parameters(&self) -> Value {
            json!({"type": "object"})
        }

        fn mutates(&self) -> bool {
            true
        }

        fn summarize(&self, raw: &str) -> String {
            raw.to_string()
        }

        async fn run(&self, _arguments: &Value, _context: &ToolContext) -> Result<String> {
            Ok("approved".to_string())
        }
    }

    struct IdentityApprover {
        calls: StdMutex<Vec<ToolCall>>,
    }

    #[async_trait]
    impl Approver for IdentityApprover {
        async fn approve(&self, _request: &ApprovalRequest) -> ApprovalDecision {
            ApprovalDecision::Deny
        }

        async fn approve_tool(&self, call: &ToolCall, _summary: &str) -> ApprovalDecision {
            self.calls.lock().unwrap().push(call.clone());
            ApprovalDecision::AllowOnce
        }
    }

    fn test_agent(tool: Box<dyn Tool>) -> Agent {
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_key = "test-key".to_string();
        profile.base_url = "http://127.0.0.1:1".to_string();
        let client = Arc::new(ModelClient::from_profile(&profile).unwrap());
        Agent::new(
            client,
            Arc::new(ToolRegistry::with_tools(vec![tool])),
            Arc::new(AllowAll),
            Arc::new(NoAsker),
            AgentOptions::new(AgentKind::Code, "/tmp"),
        )
    }

    fn test_context() -> ToolContext {
        ToolContext {
            cwd: PathBuf::from("/tmp"),
            timeout: Duration::from_secs(1),
            asker: Arc::new(NoAsker),
            todos: Arc::new(Mutex::new(Vec::new())),
        }
    }

    #[tokio::test]
    async fn structured_tool_events_keep_ids_and_arguments_in_provider_order() {
        let barrier = Arc::new(Barrier::new(2));
        let agent = test_agent(Box::new(ParallelProbe {
            barrier,
            completed: Arc::new(StdMutex::new(Vec::new())),
        }));
        let calls = vec![
            ToolCall {
                id: "call-a".to_string(),
                name: "probe".to_string(),
                arguments: r#"{"id":"a","delay_ms":20}"#.to_string(),
            },
            ToolCall {
                id: "call-b".to_string(),
                name: "probe".to_string(),
                arguments: r#"{"id":"b","delay_ms":0}"#.to_string(),
            },
        ];
        let sink = ToolIdentitySink {
            starts: StdMutex::new(Vec::new()),
            finishes: StdMutex::new(Vec::new()),
        };

        let results = agent
            .execute_tool_batch(&calls, &test_context(), &[], &sink)
            .await
            .unwrap();

        assert_eq!(*sink.starts.lock().unwrap(), calls);
        assert_eq!(*sink.finishes.lock().unwrap(), calls);
        assert!(matches!(
            &results[..],
            [Message::Tool { call_id: first, .. }, Message::Tool { call_id: second, .. }]
                if first == "call-a" && second == "call-b"
        ));
    }

    #[tokio::test]
    async fn structured_approval_receives_exact_tool_identity_and_arguments() {
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_key = "test-key".to_string();
        profile.base_url = "http://127.0.0.1:1".to_string();
        let client = Arc::new(ModelClient::from_profile(&profile).unwrap());
        let approver = Arc::new(IdentityApprover {
            calls: StdMutex::new(Vec::new()),
        });
        let agent = Agent::new(
            client,
            Arc::new(ToolRegistry::with_tools(vec![Box::new(MutatingProbe)])),
            approver.clone(),
            Arc::new(NoAsker),
            AgentOptions::new(AgentKind::Code, "/tmp"),
        );
        let call = ToolCall {
            id: "write-42".to_string(),
            name: "mutating_probe".to_string(),
            arguments: r#"{"path":"notes.txt"}"#.to_string(),
        };

        let output = agent.execute_tool(&call, &test_context(), &NullSink).await;

        assert_eq!(output, "approved");
        assert_eq!(*approver.calls.lock().unwrap(), vec![call]);
    }

    #[tokio::test]
    async fn prequeued_injection_is_persisted_and_checkpointed_before_provider_io() {
        let sink = InjectionOrderSink::new(false);
        let (base_url, server) = one_shot_model_server(sink.order.clone());
        let agent = injection_test_agent(base_url);
        let (queue, source) = injection::channel();
        queue.enqueue("inject-1", "additional direction").unwrap();
        let mut messages = vec![Message::user("original request")];

        let result = agent
            .run_with_injections(&mut messages, &sink, &source)
            .await
            .unwrap();
        let request = server.join().unwrap();

        assert_eq!(result.text, "done");
        assert_eq!(
            *sink.order.lock().unwrap(),
            ["injection", "checkpoint", "provider"]
        );
        assert!(matches!(
            sink.persisted.lock().unwrap().as_slice(),
            [Message::InjectedUser { injection_id, content }]
                if injection_id == "inject-1" && content == "additional direction"
        ));
        let checkpoints = sink.checkpoints.lock().unwrap();
        assert!(matches!(
            checkpoints[0].last(),
            Some(Message::InjectedUser { injection_id, content })
                if injection_id == "inject-1" && content == "additional direction"
        ));
        let provider_messages = request["messages"].as_array().unwrap();
        assert!(provider_messages.iter().any(|message| {
            message["role"] == "user" && message["content"] == "additional direction"
        }));
    }

    #[tokio::test]
    async fn tool_free_response_racing_an_injection_runs_a_second_logical_round() {
        let (base_url, first_seen, release_first, server) = blocked_two_shot_model_server();
        let agent = injection_test_agent(base_url);
        let sink = InjectionOrderSink::new(false);
        let (queue, source) = injection::channel();
        let mut messages = vec![Message::user("original request")];
        let mut run = Box::pin(agent.run_with_injections(&mut messages, &sink, &source));

        tokio::time::timeout(Duration::from_secs(2), async {
            tokio::select! {
                observed = first_seen => observed.unwrap(),
                result = &mut run => panic!("turn finished before the injection race: {result:?}"),
            }
        })
        .await
        .expect("the first provider request was not observed");
        queue.enqueue("inject-race", "late direction").unwrap();
        release_first.send(()).unwrap();

        let result = tokio::time::timeout(Duration::from_secs(2), run)
            .await
            .expect("the injected follow-up round did not finish")
            .unwrap();
        let requests = server.join().unwrap();
        let injection_count = |request: &serde_json::Value| {
            request["messages"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|message| {
                    message["role"] == "user" && message["content"] == "late direction"
                })
                .count()
        };

        assert_eq!(result.text, "second answer");
        assert_eq!(result.rounds, 2);
        assert_eq!(requests.len(), 2);
        assert_eq!(injection_count(&requests[0]), 0);
        assert_eq!(injection_count(&requests[1]), 1);
        assert_eq!(
            messages
                .iter()
                .filter(|message| {
                    matches!(
                        message,
                        Message::InjectedUser { injection_id, content }
                            if injection_id == "inject-race" && content == "late direction"
                    )
                })
                .count(),
            1
        );
        assert_eq!(sink.persisted.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn injected_user_persistence_failure_blocks_provider_and_rolls_back_reservation() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let agent = injection_test_agent(format!("http://{address}/v1"));
        let sink = InjectionOrderSink::new(true);
        let (queue, source) = injection::channel();
        queue.enqueue("inject-1", "must remain queued").unwrap();
        let mut messages = vec![Message::user("original request")];

        let error = agent
            .run_with_injections(&mut messages, &sink, &source)
            .await
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("simulated injected-user persistence failure"),
            "{error:#}"
        );
        assert_eq!(*sink.order.lock().unwrap(), ["injection"]);
        assert!(!messages.iter().any(
            |message| matches!(message, Message::InjectedUser { injection_id, .. } if injection_id == "inject-1")
        ));
        let rejected = queue
            .enqueue("inject-late", "must be retained by the caller")
            .expect_err("an exited run must seal its injection source");
        assert!(matches!(rejected, injection::EnqueueError::Closed(_)));
        assert!(matches!(
            rejected.into_message(),
            Message::InjectedUser { injection_id, content }
                if injection_id == "inject-late" && content == "must be retained by the caller"
        ));
        let reservation = source.reserve().expect("failed persistence must roll back");
        assert!(matches!(
            reservation.message(),
            Message::InjectedUser { injection_id, content }
                if injection_id == "inject-1" && content == "must remain queued"
        ));
        drop(reservation);
        listener.set_nonblocking(true).unwrap();
        assert!(matches!(
            listener.accept(),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock
        ));
    }

    #[tokio::test]
    async fn cancelling_an_in_flight_run_seals_admission_and_preserves_pending_input() {
        let (base_url, request_seen, release_request, server) = blocked_one_shot_model_server();
        let agent = injection_test_agent(base_url);
        let sink = InjectionOrderSink::new(false);
        let (queue, source) = injection::channel();
        let mut messages = vec![Message::user("original request")];
        let mut run = Box::pin(agent.run_with_injections(&mut messages, &sink, &source));

        tokio::time::timeout(Duration::from_secs(2), async {
            tokio::select! {
                observed = request_seen => observed.unwrap(),
                result = &mut run => panic!("turn finished before cancellation: {result:?}"),
            }
        })
        .await
        .expect("the provider request was not observed");
        queue
            .enqueue("inject-pending", "retain after cancel")
            .unwrap();

        drop(run);

        let rejected = queue
            .enqueue("inject-late", "too late")
            .expect_err("dropping the run must seal admission");
        assert!(matches!(rejected, injection::EnqueueError::Closed(_)));
        assert!(matches!(
            rejected.into_message(),
            Message::InjectedUser { injection_id, content }
                if injection_id == "inject-late" && content == "too late"
        ));
        let reservation = source
            .reserve()
            .expect("accepted input must remain pending");
        assert!(matches!(
            reservation.message(),
            Message::InjectedUser { injection_id, content }
                if injection_id == "inject-pending" && content == "retain after cancel"
        ));
        drop(reservation);

        release_request.send(()).unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn parallel_tools_overlap_but_results_keep_call_order() {
        let barrier = Arc::new(Barrier::new(2));
        let completed = Arc::new(StdMutex::new(Vec::new()));
        let agent = test_agent(Box::new(ParallelProbe {
            barrier,
            completed: completed.clone(),
        }));
        let calls = vec![
            crate::model::ToolCall {
                id: "call_1".into(),
                name: "probe".into(),
                arguments: r#"{"id":"slow","delay_ms":50}"#.into(),
            },
            crate::model::ToolCall {
                id: "call_2".into(),
                name: "probe".into(),
                arguments: r#"{"id":"fast","delay_ms":0}"#.into(),
            },
        ];

        let prefix = vec![
            Message::user("inspect"),
            Message::assistant(crate::model::AssistantMessage {
                content: None,
                reasoning: None,
                tool_calls: calls.clone(),
                responses_items: Vec::new(),
            }),
        ];
        let original_prefix = prefix.clone();
        let sink = RecordingCheckpointSink {
            snapshots: StdMutex::new(Vec::new()),
        };
        let messages = tokio::time::timeout(
            Duration::from_secs(1),
            agent.execute_tool_batch(&calls, &test_context(), &prefix, &sink),
        )
        .await
        .expect("parallel calls should both reach the barrier")
        .unwrap();

        assert_eq!(prefix, original_prefix);
        assert_eq!(*completed.lock().unwrap(), ["fast", "slow"]);
        let snapshots = sink.snapshots.lock().unwrap();
        assert_eq!(snapshots.len(), 2);
        assert!(matches!(&snapshots[0][2], Message::Tool { call_id, .. } if call_id == "call_2"));
        assert_eq!(
            snapshots[1][2..]
                .iter()
                .map(|message| match message {
                    Message::Tool { call_id, .. } => call_id.as_str(),
                    _ => panic!("expected tool result"),
                })
                .collect::<Vec<_>>(),
            ["call_1", "call_2"]
        );
        drop(snapshots);
        let ids_and_outputs: Vec<_> = messages
            .iter()
            .map(|message| match message {
                Message::Tool {
                    call_id, content, ..
                } => (call_id.as_str(), content.as_str()),
                _ => panic!("expected tool result"),
            })
            .collect();
        assert_eq!(ids_and_outputs, [("call_1", "slow"), ("call_2", "fast")]);
    }

    #[tokio::test]
    async fn cancelling_parallel_batch_keeps_the_last_published_subset() {
        let barrier = Arc::new(Barrier::new(2));
        let completed = Arc::new(StdMutex::new(Vec::new()));
        let agent = test_agent(Box::new(ParallelProbe {
            barrier,
            completed: completed.clone(),
        }));
        let calls = vec![
            crate::model::ToolCall {
                id: "call_slow".into(),
                name: "probe".into(),
                arguments: r#"{"id":"slow","delay_ms":500}"#.into(),
            },
            crate::model::ToolCall {
                id: "call_fast".into(),
                name: "probe".into(),
                arguments: r#"{"id":"fast","delay_ms":0}"#.into(),
            },
        ];
        let prefix = vec![Message::assistant(crate::model::AssistantMessage {
            content: None,
            reasoning: None,
            tool_calls: calls.clone(),
            responses_items: Vec::new(),
        })];
        let original_prefix = prefix.clone();
        let sink = SignalingCheckpointSink {
            snapshots: StdMutex::new(Vec::new()),
            published: Notify::new(),
        };
        let published = sink.published.notified();
        tokio::pin!(published);
        let context = test_context();
        let mut execution = Box::pin(agent.execute_tool_batch(&calls, &context, &prefix, &sink));

        tokio::select! {
            () = &mut published => {}
            result = &mut execution => panic!("batch ended before cancellation: {result:?}"),
            () = tokio::time::sleep(Duration::from_secs(1)) => panic!("first result was not checkpointed"),
        }
        drop(execution);

        assert_eq!(prefix, original_prefix);
        assert_eq!(*completed.lock().unwrap(), ["fast"]);
        let snapshots = sink.snapshots.lock().unwrap();
        assert_eq!(snapshots.len(), 1);
        assert!(
            matches!(&snapshots[0][1], Message::Tool { call_id, .. } if call_id == "call_fast")
        );
    }

    #[tokio::test]
    async fn parallel_tool_failure_does_not_cancel_siblings() {
        let barrier = Arc::new(Barrier::new(2));
        let completed = Arc::new(StdMutex::new(Vec::new()));
        let agent = test_agent(Box::new(ParallelProbe { barrier, completed }));
        let calls = vec![
            crate::model::ToolCall {
                id: "call_1".into(),
                name: "probe".into(),
                arguments: r#"{"id":"broken","fail":true}"#.into(),
            },
            crate::model::ToolCall {
                id: "call_2".into(),
                name: "probe".into(),
                arguments: r#"{"id":"ok"}"#.into(),
            },
        ];

        let messages = agent
            .execute_tool_batch(&calls, &test_context(), &[], &NullSink)
            .await
            .unwrap();
        let outputs: Vec<_> = messages
            .iter()
            .map(|message| match message {
                Message::Tool { content, .. } => content.as_str(),
                _ => panic!("expected tool result"),
            })
            .collect();
        assert!(
            outputs[0].contains("Error: broken failed"),
            "{}",
            outputs[0]
        );
        assert_eq!(outputs[1], "ok");
    }

    #[test]
    fn agent_kind_parses() {
        assert_eq!(AgentKind::parse("code"), Some(AgentKind::Code));
        assert_eq!(AgentKind::parse("QA"), Some(AgentKind::Qa));
        assert_eq!(AgentKind::parse("nope"), None);
    }

    #[test]
    fn civil_dates_match_known_values() {
        assert_eq!(civil_from_days(0), "1970-01-01");
        assert_eq!(civil_from_days(19_000), "2022-01-08");
    }

    #[test]
    fn system_prompt_includes_project_memory() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("AGENTS.md"), "Use cargo clippy.").unwrap();
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_key = "test-key".to_string();
        profile.base_url = "http://127.0.0.1:1".to_string();
        let agent = Agent::new(
            Arc::new(ModelClient::from_profile(&profile).unwrap()),
            Arc::new(ToolRegistry::with_tools(Vec::new())),
            Arc::new(AllowAll),
            Arc::new(NoAsker),
            AgentOptions::new(AgentKind::Code, temp.path()),
        );

        let prompt = agent.system_prompt();

        assert!(prompt.contains("# Auto-loaded memory"));
        assert!(prompt.contains("## File: AGENTS.md\n\nUse cargo clippy."));
    }

    #[test]
    fn forced_compression_uses_latest_call_instead_of_aggregate_usage() {
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_key = "test-key".to_string();
        profile.base_url = "http://127.0.0.1:1".to_string();
        profile.context_window = Some(2_000);
        profile.max_tokens = Some(500);
        let force_tokens = crate::context_budget::ContextBudget::derive(2_000, 500)
            .unwrap()
            .force_tokens;
        let agent = Agent::new(
            Arc::new(ModelClient::from_profile(&profile).unwrap()),
            Arc::new(ToolRegistry::with_tools(Vec::new())),
            Arc::new(AllowAll),
            Arc::new(NoAsker),
            AgentOptions::new(AgentKind::Code, "/tmp"),
        );
        let mut result = TurnResult {
            usage: Usage {
                prompt_tokens: u64::from(force_tokens) + 10_000,
                completion_tokens: 20,
                total_tokens: u64::from(force_tokens) + 10_020,
            },
            latest_usage: Usage {
                prompt_tokens: u64::from(force_tokens),
                completion_tokens: 10,
                total_tokens: u64::from(force_tokens) + 10,
            },
            ..TurnResult::default()
        };

        assert!(!agent.should_force_compression(&result));
        result.latest_usage.prompt_tokens += 1;
        assert!(agent.should_force_compression(&result));
    }

    #[test]
    fn system_prompt_mentions_cwd_and_date() {
        let options = AgentOptions::new(AgentKind::Code, "/tmp/example");
        let prompt = format!(
            "{}\n\nCurrent working directory: /tmp/example\nToday's date: {}",
            CODE_SYSTEM_PROMPT,
            today()
        );
        assert!(prompt.contains("/tmp/example"));
        assert!(!options.cwd.to_string_lossy().is_empty());
    }
}
