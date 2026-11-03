//! Session-local tools: the todo list and user questions.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use super::{required_str, TodoItem, TodoStatus, Tool, ToolContext};

pub struct TodoWrite;

#[async_trait]
impl Tool for TodoWrite {
    fn name(&self) -> &'static str {
        "todo_write"
    }

    fn description(&self) -> &'static str {
        "Replace the session todo list. Use it to plan multi-step work and to show progress. \
         Keep one item in_progress at a time."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The complete todo list.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "What needs to be done."},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                        },
                        "required": ["content"]
                    }
                }
            },
            "required": ["todos"]
        })
    }

    fn summarize(&self, raw: &str) -> String {
        match serde_json::from_str::<Value>(raw) {
            Ok(arguments) => {
                let todos = arguments
                    .get("todos")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let done = todos
                    .iter()
                    .filter(|todo| todo.get("status").and_then(Value::as_str) == Some("completed"))
                    .count();
                format!("{}/{} items completed", done, todos.len())
            }
            Err(_) => "todo list".to_string(),
        }
    }

    async fn run(&self, arguments: &Value, context: &ToolContext) -> Result<String> {
        let todos = arguments
            .get("todos")
            .and_then(Value::as_array)
            .context("missing array argument \"todos\"")?;
        let mut items = Vec::with_capacity(todos.len());
        for (index, todo) in todos.iter().enumerate() {
            let content = todo
                .get("content")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|content| !content.is_empty())
                .with_context(|| format!("todo item {index} needs a non-empty content"))?;
            let status = match todo
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("pending")
            {
                "pending" => TodoStatus::Pending,
                "in_progress" | "in-progress" | "inprogress" => TodoStatus::InProgress,
                "completed" | "done" => TodoStatus::Completed,
                other => bail!(
                    "todo item {index} has unknown status {other:?}; \
                     expected pending, in_progress or completed"
                ),
            };
            items.push(TodoItem {
                content: content.to_string(),
                status,
            });
        }
        let completed = items
            .iter()
            .filter(|item| item.status == TodoStatus::Completed)
            .count();
        let total = items.len();
        *context.todos.lock().await = items.clone();
        if items.is_empty() {
            return Ok("Todo list cleared.".to_string());
        }
        let mut output = format!("Todo list updated ({completed}/{total} completed):\n");
        for (index, item) in items.iter().enumerate() {
            output.push_str(&format!(
                "{}. {} {}\n",
                index + 1,
                item.status.marker(),
                item.content
            ));
        }
        Ok(output.trim_end().to_string())
    }
}

pub struct AskUser;

#[async_trait]
impl Tool for AskUser {
    fn name(&self) -> &'static str {
        "ask_user"
    }

    fn description(&self) -> &'static str {
        "Ask the user a question and wait for the answer. Use it only when a decision \
         genuinely needs human input; otherwise make a reasonable choice yourself."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to ask."}
            },
            "required": ["question"]
        })
    }

    fn summarize(&self, raw: &str) -> String {
        match serde_json::from_str::<Value>(raw) {
            Ok(arguments) => arguments
                .get("question")
                .and_then(Value::as_str)
                .unwrap_or("question")
                .lines()
                .next()
                .unwrap_or("question")
                .to_string(),
            Err(_) => "question".to_string(),
        }
    }

    async fn run(&self, arguments: &Value, context: &ToolContext) -> Result<String> {
        let question = required_str(arguments, "question")?;
        let answer = context.asker.ask(question).await;
        Ok(format!("User's answer: {answer}"))
    }
}
