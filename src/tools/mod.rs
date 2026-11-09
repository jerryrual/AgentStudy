//! Built-in tools available to the agent.
//!
//! Tool implementations live in submodules grouped by theme; this module owns
//! the [`Tool`] trait, the [`ToolContext`] and the [`ToolRegistry`].

pub mod fs;
pub mod misc;
pub mod search;
pub mod shell;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::interaction::Asker;
use crate::model::ToolDef;

/// Longest tool output handed to the model (in characters).
pub const MAX_TOOL_OUTPUT: usize = 30_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

impl TodoStatus {
    pub fn marker(self) -> &'static str {
        match self {
            Self::Pending => "[ ]",
            Self::InProgress => "[~]",
            Self::Completed => "[x]",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub content: String,
    pub status: TodoStatus,
}

/// Runtime state tools operate against.
pub struct ToolContext {
    pub cwd: PathBuf,
    pub timeout: Duration,
    pub asker: Arc<dyn Asker>,
    pub todos: Arc<Mutex<Vec<TodoItem>>>,
}

impl ToolContext {
    pub fn resolve(&self, path: &str) -> PathBuf {
        resolve_path(&self.cwd, path)
    }
}

/// Expand `~`, and resolve relative paths against `cwd`.
pub fn resolve_path(cwd: &Path, raw: &str) -> PathBuf {
    let raw = raw.trim();
    if raw == "~" || raw.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            return if raw == "~" {
                home
            } else {
                home.join(&raw[2..])
            };
        }
    }
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    }
}

/// Render a path relative to the working directory when possible.
pub fn display_path(cwd: &Path, path: &Path) -> String {
    match path.strip_prefix(cwd) {
        Ok(relative) if !relative.as_os_str().is_empty() => relative.display().to_string(),
        _ => path.display().to_string(),
    }
}

/// Keep tool output bounded while preserving both head and tail.
pub fn truncate_output(text: &str) -> String {
    let count = text.chars().count();
    if count <= MAX_TOOL_OUTPUT {
        return text.to_string();
    }
    let head: String = text.chars().take(20_000).collect();
    let tail: String = text.chars().skip(count.saturating_sub(8_000)).collect();
    format!(
        "{head}\n\n[... {} characters omitted ...]\n\n{tail}",
        count - 28_000
    )
}

/// A tool callable by the model.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    /// Model-visible name; dynamic adapters may override this without leaking strings.
    fn model_name(&self) -> &str {
        self.name()
    }
    /// Model-visible description; dynamic adapters may override this.
    fn model_description(&self) -> &str {
        self.description()
    }
    fn parameters(&self) -> Value;
    /// Whether invoking this tool can change the workspace.
    fn mutates(&self) -> bool {
        false
    }
    /// Whether this tool may overlap other calls from the same model response.
    ///
    /// This is deliberately opt-in. A tool can be non-mutating yet still be
    /// stateful or interactive (for example `todo_write` or `ask_user`).
    fn parallel_safe(&self) -> bool {
        false
    }
    /// One-line summary shown in the TUI and the `icode run` progress output.
    fn summarize(&self, raw_arguments: &str) -> String;
    async fn run(&self, arguments: &Value, context: &ToolContext) -> Result<String>;
}

/// Parse model-supplied JSON arguments.
pub fn parse_arguments(raw: &str) -> Result<Value> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(Value::Object(Default::default()));
    }
    serde_json::from_str(raw).map_err(|error| anyhow!("invalid JSON arguments: {error}"))
}

pub fn required_str<'a>(arguments: &'a Value, key: &str) -> Result<&'a str> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string argument {key:?}"))
}

pub fn optional_str<'a>(arguments: &'a Value, key: &str) -> Option<&'a str> {
    arguments.get(key).and_then(Value::as_str)
}

pub fn optional_usize(arguments: &Value, key: &str) -> Option<usize> {
    arguments
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
}

pub fn optional_bool(arguments: &Value, key: &str) -> Option<bool> {
    arguments.get(key).and_then(Value::as_bool)
}

/// All tools registered for a run.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
    index: HashMap<String, usize>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

impl ToolRegistry {
    /// Register the complete built-in tool set.
    pub fn builtin() -> Self {
        Self::try_with_tools(Self::builtin_tools()).expect("built-in tool names are unique")
    }

    /// Register built-ins plus additional tools, rejecting every name collision.
    pub fn builtin_with(mut additional: Vec<Box<dyn Tool>>) -> Result<Self> {
        let mut tools = Self::builtin_tools();
        tools.append(&mut additional);
        Self::try_with_tools(tools)
    }

    fn builtin_tools() -> Vec<Box<dyn Tool>> {
        vec![
            Box::new(fs::ReadFile),
            Box::new(fs::WriteFile),
            Box::new(fs::EditFile),
            Box::new(fs::ListDir),
            Box::new(search::Grep),
            Box::new(search::Glob),
            Box::new(shell::Bash),
            Box::new(misc::TodoWrite),
            Box::new(misc::AskUser),
        ]
    }

    pub(crate) fn try_with_tools(tools: Vec<Box<dyn Tool>>) -> Result<Self> {
        let mut index = HashMap::with_capacity(tools.len());
        for (position, tool) in tools.iter().enumerate() {
            let name = tool.model_name();
            if name.is_empty() {
                return Err(anyhow!("tool name must not be empty"));
            }
            if index.insert(name.to_string(), position).is_some() {
                return Err(anyhow!("duplicate tool name {name:?}"));
            }
        }
        Ok(Self { tools, index })
    }

    #[cfg(test)]
    pub(crate) fn with_tools(tools: Vec<Box<dyn Tool>>) -> Self {
        Self::try_with_tools(tools).expect("test tool names are unique")
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.index
            .get(name)
            .and_then(|index| self.tools.get(*index))
            .map(|tool| tool.as_ref())
    }

    pub fn specs(&self) -> Vec<ToolDef> {
        self.tools
            .iter()
            .map(|tool| ToolDef {
                name: tool.model_name().to_string(),
                description: tool.model_description().to_string(),
                parameters: tool.parameters(),
            })
            .collect()
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.tools.iter().map(|tool| tool.name()).collect()
    }

    pub fn model_names(&self) -> Vec<&str> {
        self.tools.iter().map(|tool| tool.model_name()).collect()
    }
}

/// Helper used by argument builders.
pub fn json_path_property(description: &str) -> Value {
    json!({"type": "string", "description": description})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_output_keeps_head_and_tail() {
        let text = "a".repeat(100_000);
        let truncated = truncate_output(&text);
        assert!(truncated.len() < 100_000);
        assert!(truncated.contains("characters omitted"));
        assert!(truncated.starts_with("aaaa"));
        assert!(truncated.ends_with("aaaa"));
    }

    #[test]
    fn registry_exposes_all_tools() {
        let registry = ToolRegistry::builtin();
        let names = registry.names();
        for expected in [
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "grep",
            "glob",
            "bash",
            "todo_write",
            "ask_user",
        ] {
            assert!(names.contains(&expected), "missing tool {expected}");
        }
        assert!(registry.get("bash").is_some());
        assert!(registry.get("nope").is_none());
        assert_eq!(registry.specs().len(), names.len());

        for parallel in ["read_file", "list_dir", "grep", "glob"] {
            assert!(
                registry.get(parallel).unwrap().parallel_safe(),
                "{parallel} should opt into parallel execution"
            );
        }
        for sequential in ["write_file", "edit_file", "bash", "todo_write", "ask_user"] {
            assert!(
                !registry.get(sequential).unwrap().parallel_safe(),
                "{sequential} must remain a sequential barrier"
            );
        }
    }

    struct DynamicTool {
        name: String,
    }

    #[async_trait]
    impl Tool for DynamicTool {
        fn name(&self) -> &'static str {
            "dynamic_tool"
        }

        fn description(&self) -> &'static str {
            "dynamic fallback"
        }

        fn model_name(&self) -> &str {
            &self.name
        }

        fn model_description(&self) -> &str {
            "dynamic description"
        }

        fn parameters(&self) -> Value {
            json!({"type": "object"})
        }

        fn summarize(&self, _raw_arguments: &str) -> String {
            self.name.clone()
        }

        async fn run(&self, _arguments: &Value, _context: &ToolContext) -> Result<String> {
            Ok(String::new())
        }
    }

    #[test]
    fn registry_indexes_dynamic_names_and_rejects_collisions() {
        let registry = ToolRegistry::try_with_tools(vec![Box::new(DynamicTool {
            name: "remote".into(),
        })])
        .unwrap();
        assert!(registry.get("remote").is_some());
        assert_eq!(registry.model_names(), vec!["remote"]);
        assert_eq!(registry.specs()[0].description, "dynamic description");
        assert!(ToolRegistry::try_with_tools(vec![
            Box::new(DynamicTool {
                name: "same".into(),
            }),
            Box::new(DynamicTool {
                name: "same".into(),
            }),
        ])
        .is_err());
        assert!(ToolRegistry::builtin_with(vec![Box::new(DynamicTool {
            name: "read_file".into(),
        })])
        .is_err());
        let combined = ToolRegistry::builtin_with(vec![Box::new(DynamicTool {
            name: "remote".into(),
        })])
        .unwrap();
        assert!(combined.get("read_file").is_some());
        assert!(combined.get("remote").is_some());
    }

    #[test]
    fn resolve_path_expands_tilde_and_relative_paths() {
        let cwd = Path::new("/tmp/work");
        assert_eq!(resolve_path(cwd, "src/main.rs"), cwd.join("src/main.rs"));
        assert_eq!(resolve_path(cwd, "/etc/hosts"), PathBuf::from("/etc/hosts"));
        assert_eq!(resolve_path(cwd, "~"), dirs::home_dir().unwrap());
    }

    #[test]
    fn argument_helpers_validate_types() {
        let args = json!({"path": "a", "count": 3, "flag": true});
        assert_eq!(required_str(&args, "path").unwrap(), "a");
        assert!(required_str(&args, "count").is_err());
        assert_eq!(optional_usize(&args, "count"), Some(3));
        assert_eq!(optional_bool(&args, "flag"), Some(true));
        assert_eq!(optional_str(&args, "missing"), None);
    }
}
