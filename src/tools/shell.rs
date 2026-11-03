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
                        return format!("{}: {}", description.trim(), escape_newlines(command));
                    }
                }
                escape_newlines(command)
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
///
/// The command runs in its own process group (Unix) so a timeout can kill the
/// whole tree, and stdout/stderr are read concurrently with a byte cap so an
/// unbounded producer cannot exhaust memory.
pub async fn run_command(command: &str, cwd: &Path, timeout: Duration) -> Result<String> {
    let command = command.trim();
    if command.is_empty() {
        bail!("command must not be empty");
    }
    let mut process = tokio::process::Command::new("bash");
    process
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    process.process_group(0);

    let mut child = process
        .spawn()
        .with_context(|| format!("failed to start command: {command}"))?;
    let pid = child.id();
    let stdout = child.stdout.take().expect("stdout is piped");
    let stderr = child.stderr.take().expect("stderr is piped");
    let stdout_task = tokio::spawn(read_capped(stdout));
    let stderr_task = tokio::spawn(read_capped(stderr));

    let status = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(status) => status.with_context(|| format!("failed to run command: {command}"))?,
        Err(_) => {
            #[cfg(unix)]
            if let Some(pid) = pid {
                // Negative pid targets the whole process group.
                // SAFETY: `pid` is the direct child we just spawned.
                unsafe {
                    libc::kill(-(pid as i32), libc::SIGKILL);
                }
            }
            let _ = child.kill().await;
            let (stdout, _) = join_reader(stdout_task).await?;
            let (stderr, _) = join_reader(stderr_task).await?;
            let stdout = String::from_utf8_lossy(&stdout);
            let stderr = String::from_utf8_lossy(&stderr);
            let mut message = format!(
                "command timed out after {}s and was killed: {command}",
                timeout.as_secs()
            );
            if !stdout.trim().is_empty() {
                message.push_str(&format!("\n\npartial stdout:\n{}", preview(&stdout)));
            }
            if !stderr.trim().is_empty() {
                message.push_str(&format!("\n\npartial stderr:\n{}", preview(&stderr)));
            }
            bail!("{message}");
        }
    };

    let (stdout, stdout_truncated) = join_reader(stdout_task).await?;
    let (stderr, stderr_truncated) = join_reader(stderr_task).await?;
    let stdout = String::from_utf8_lossy(&stdout);
    let stderr = String::from_utf8_lossy(&stderr);
    let code = status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "terminated by signal".to_string());
    let mut result = format!("exit code: {code}");
    if stdout.trim().is_empty() {
        result.push_str("\nstdout: (empty)");
    } else {
        result.push_str(&format!("\nstdout:\n{stdout}"));
        if stdout_truncated {
            result.push_str(&format!(
                "\n[stdout truncated at {} bytes; the process kept running]",
                OUTPUT_CAP
            ));
        }
    }
    if !stderr.trim().is_empty() {
        result.push_str(&format!("\nstderr:\n{stderr}"));
        if stderr_truncated {
            result.push_str(&format!(
                "\n[stderr truncated at {} bytes; the process kept running]",
                OUTPUT_CAP
            ));
        }
    }
    Ok(result)
}

const OUTPUT_CAP: usize = 64 * 1024;
const PARTIAL_PREVIEW: usize = 4_000;

async fn read_capped<R>(mut reader: R) -> std::io::Result<(Vec<u8>, bool)>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;
    let mut kept = Vec::new();
    let mut truncated = false;
    let mut buffer = [0u8; 8192];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        if kept.len() < OUTPUT_CAP {
            let take = (OUTPUT_CAP - kept.len()).min(read);
            kept.extend_from_slice(&buffer[..take]);
            if take < read {
                truncated = true;
            }
        } else {
            truncated = true;
        }
    }
    Ok((kept, truncated))
}

async fn join_reader(
    handle: tokio::task::JoinHandle<std::io::Result<(Vec<u8>, bool)>>,
) -> std::io::Result<(Vec<u8>, bool)> {
    handle
        .await
        .map_err(|error| std::io::Error::other(error.to_string()))?
}

fn preview(text: &str) -> String {
    if text.chars().count() <= PARTIAL_PREVIEW {
        return text.to_string();
    }
    let mut end = PARTIAL_PREVIEW;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

fn escape_newlines(text: &str) -> String {
    text.trim()
        .replace("\r\n", "⏎ ")
        .replace(['\r', '\n'], "⏎ ")
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

    #[test]
    fn summary_keeps_multiline_and_long_commands() {
        let long = format!("echo {}", "x".repeat(200));
        let summary = Bash.summarize(&format!("{{\"command\":\"{long}\"}}"));
        assert_eq!(summary, long);

        let multiline = Bash.summarize(r#"{"command":"echo one\necho two"}"#);
        assert!(multiline.contains("echo one"), "{multiline}");
        assert!(multiline.contains("echo two"), "{multiline}");
        assert!(multiline.contains('⏎'), "{multiline}");
    }
}
