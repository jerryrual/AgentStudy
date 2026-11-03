//! Shell tool.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use super::{optional_usize, required_str, Tool, ToolContext};

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_TIMEOUT_SECS: u64 = 600;

pub struct Bash;

#[async_trait]
impl Tool for Bash {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn description(&self) -> &'static str {
        "Run a command with bash in the working directory and return its exit code, \
         stdout and stderr. Commands run non-interactively; use this for builds, tests, \
         git and programs."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "description": {"type": "string", "description": "Short description of what the command does."},
                "timeout_secs": {"type": "integer", "minimum": 1, "maximum": 600, "description": "Timeout in seconds (default 120)."}
            },
            "required": ["command"]
        })
    }

    fn mutates(&self) -> bool {
        true
    }

    fn summarize(&self, raw: &str) -> String {
        match serde_json::from_str::<Value>(raw) {
            Ok(arguments) => {
                let command = arguments
                    .get("command")
                    .and_then(Value::as_str)
                    .unwrap_or("<command>");
                if let Some(description) = arguments.get("description").and_then(Value::as_str) {
                    if !description.trim().is_empty() {
                        // Always show the command: the description is model-supplied and
                        // must not be able to hide what will actually run.
                        return format!("{}: {}", description.trim(), one_line(command));
                    }
                }
                one_line(command)
            }
            Err(_) => short(raw),
        }
    }

    async fn run(&self, arguments: &Value, context: &ToolContext) -> Result<String> {
        let command = required_str(arguments, "command")?;
        let timeout_secs = optional_usize(arguments, "timeout_secs")
            .map(|value| value as u64)
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .clamp(1, MAX_TIMEOUT_SECS);
        run_command(command, &context.cwd, Duration::from_secs(timeout_secs)).await
    }
}

/// Execute a shell command and render its result for the model.
pub async fn run_command(command: &str, cwd: &Path, timeout: Duration) -> Result<String> {
    let command = command.trim();
    if command.is_empty() {
        bail!("command must not be empty");
    }
    let child = tokio::process::Command::new("bash")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to start command: {command}"))?;

    let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(output) => output.with_context(|| format!("failed to run command: {command}"))?,
        Err(_) => bail!(
            "command timed out after {}s and was killed: {command}",
            timeout.as_secs()
        ),
    };

    let code = output
        .status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "terminated by signal".to_string());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut result = format!("exit code: {code}");
    if stdout.trim().is_empty() {
        result.push_str("\nstdout: (empty)");
    } else {
        result.push_str(&format!("\nstdout:\n{stdout}"));
    }
    if !stderr.trim().is_empty() {
        result.push_str(&format!("\nstderr:\n{stderr}"));
    }
    Ok(result)
}

fn one_line(text: &str) -> String {
    let mut line = text.lines().next().unwrap_or_default().trim().to_string();
    if line.len() > 80 {
        let mut end = 80;
        while end > 0 && !line.is_char_boundary(end) {
            end -= 1;
        }
        line.truncate(end);
        line.push('…');
    }
    line
}

fn short(raw: &str) -> String {
    let mut text = raw.trim().to_string();
    if text.len() > 60 {
        let mut end = 60;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push('…');
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_never_hides_the_command() {
        let summary = Bash
            .summarize(r#"{"command":"rm -rf build","description":"clean the build directory"}"#);
        assert!(summary.contains("rm -rf build"), "{summary}");
        assert!(summary.contains("clean the build directory"), "{summary}");
    }

    #[test]
    fn summary_uses_the_command_without_a_description() {
        let summary = Bash.summarize(r#"{"command":"cargo test"}"#);
        assert_eq!(summary, "cargo test");
    }
}
