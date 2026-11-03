//! The agent loop: send the conversation to the model, execute the tools it
//! asks for, feed the results back, and repeat until it answers.

pub mod events;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use tokio::sync::Mutex;

pub use events::{EventSink, NullSink};

use crate::interaction::{ApprovalDecision, ApprovalRequest, Approver, Asker};
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

Answer the user's questions about the current project. You may inspect files, search the codebase and run read-only shell commands, but you must not modify files, install packages or run destructive commands.

Operating rules:
- Investigate before you answer: read the relevant files and search for definitions.
- Use read_file, list_dir, grep and glob; avoid mutating tools entirely.
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
        Self {
            client,
            tools,
            approver,
            asker,
            options,
            todos: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn model(&self) -> &str {
        self.client.model()
    }

    pub fn kind(&self) -> AgentKind {
        self.options.kind
    }

    /// The system prompt for this agent, anchored at the working directory.
    pub fn system_prompt(&self) -> String {
        let base = match self.options.kind {
            AgentKind::Code => CODE_SYSTEM_PROMPT,
            AgentKind::Qa => QA_SYSTEM_PROMPT,
        };
        format!(
            "{base}\n\nCurrent working directory: {}\nToday's date: {}",
            self.options.cwd.display(),
            today()
        )
    }

    pub fn tool_defs(&self) -> Vec<ToolDef> {
        self.tools.specs()
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
        let started = Instant::now();
        let mut usage = Usage::default();
        let mut tool_calls = 0usize;
        let mut rounds = 0usize;

        self.ensure_system_prompt(messages);
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

            let completion = self
                .client
                .complete(messages, &self.tools.specs(), sink)
                .await?;
            usage.add(completion.usage);
            sink.usage(&completion.usage);

            let message = completion.message;
            let calls = message.tool_calls.clone();
            let text = message.content.clone();
            let finish_reason = completion.finish_reason.clone();
            if message.is_empty() && calls.is_empty() {
                let reason = if finish_reason == "length" {
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
                let result = TurnResult {
                    text: text.unwrap_or_default(),
                    usage,
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

            for call in calls {
                tool_calls += 1;
                let result = self.execute_tool(&call, &context, sink).await;
                let message = Message::tool(call.id.clone(), call.name.clone(), result);
                messages.push(message.clone());
                sink.message_added(&message);
            }
        }
    }

    async fn execute_tool(
        &self,
        call: &crate::model::ToolCall,
        context: &ToolContext,
        sink: &dyn EventSink,
    ) -> String {
        let summary = self
            .tools
            .get(&call.name)
            .map(|tool| tool.summarize(&call.arguments))
            .unwrap_or_else(|| call.name.clone());
        sink.tool_started(&call.name, &summary);

        let started = Instant::now();
        let arguments = match call.parse_arguments() {
            Ok(arguments) => arguments,
            Err(error) => {
                let message = format!("Error: invalid JSON arguments for {}: {error}", call.name);
                sink.tool_finished(&call.name, false, started.elapsed(), &summary, &message);
                return message;
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
            sink.tool_finished(&call.name, false, started.elapsed(), &summary, &message);
            return message;
        };

        if tool.mutates() {
            let request = ApprovalRequest::new(&call.name, &summary);
            match self.approver.approve(&request).await {
                ApprovalDecision::Deny => {
                    let message = "Error: the user denied this operation.".to_string();
                    sink.tool_finished(&call.name, false, started.elapsed(), &summary, &message);
                    return message;
                }
                ApprovalDecision::AllowOnce | ApprovalDecision::AllowAlways => {}
            }
        }

        match tool.run(&arguments, context).await {
            Ok(output) => {
                sink.tool_finished(&call.name, true, started.elapsed(), &summary, &output);
                truncate_output(&output)
            }
            Err(error) => {
                let message = format!("Error: {error:#}");
                sink.tool_finished(&call.name, false, started.elapsed(), &summary, &message);
                message
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
