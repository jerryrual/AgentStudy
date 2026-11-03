//! File-system tools: read, write, edit and list.

use std::path::Path;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use similar::{ChangeTag, TextDiff};

use super::{
    display_path, json_path_property, optional_bool, optional_usize, required_str, Tool,
    ToolContext,
};

/// Files larger than this are refused rather than loaded into memory.
const MAX_READ_BYTES: u64 = 8 * 1024 * 1024;
const MAX_LINE_BYTES: usize = 4_000;

pub struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
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

    fn summarize(&self, raw: &str) -> String {
        match serde_json::from_str::<Value>(raw) {
            Ok(arguments) => {
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
            Err(_) => short(raw),
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

        let metadata = tokio::fs::metadata(&full)
            .await
            .with_context(|| format!("cannot read {display}"))?;
        if metadata.is_dir() {
            bail!("{display} is a directory; use list_dir instead");
        }
        if metadata.len() > MAX_READ_BYTES {
            bail!(
                "{display} is {} bytes; files larger than {} bytes are not read. Use grep to search it.",
                metadata.len(),
                MAX_READ_BYTES
            );
        }
        let bytes = tokio::fs::read(&full)
            .await
            .with_context(|| format!("cannot read {display}"))?;
        if bytes.contains(&0) {
            bail!("{display} looks like a binary file; only UTF-8 text is supported");
        }
        let text =
            String::from_utf8(bytes).map_err(|_| anyhow!("{display} is not valid UTF-8 text"))?;
        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len();
        if total == 0 {
            return Ok(format!("{display} is empty (0 lines)"));
        }
        if offset > total {
            bail!("offset {offset} is past the end of {display} ({total} lines)");
        }
        let start = offset - 1;
        let end = (start + limit).min(total);
        let width = end.to_string().len();
        let mut body = String::new();
        for (index, line) in lines[start..end].iter().enumerate() {
            let number = start + index + 1;
            let line = *line;
            if line.len() > MAX_LINE_BYTES {
                let mut end = MAX_LINE_BYTES;
                while end > 0 && !line.is_char_boundary(end) {
                    end -= 1;
                }
                body.push_str(&format!("{number:>width$}\t{}…\n", &line[..end]));
                continue;
            }
            body.push_str(&format!("{number:>width$}\t{line}\n"));
        }
        if end < total {
            body.push_str(&format!(
                "\n[{} more lines; pass offset={} to continue]\n",
                total - end,
                end + 1
            ));
        }
        if offset > 1 || end < total {
            Ok(format!(
                "{display} (lines {}-{} of {total})\n{body}",
                offset, end
            ))
        } else {
            Ok(format!("{display} ({total} lines)\n{body}"))
        }
    }
}

pub struct WriteFile;

#[async_trait]
impl Tool for WriteFile {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn description(&self) -> &'static str {
        "Create or overwrite a UTF-8 text file. Missing parent directories are created. \
         For small edits prefer edit_file, which keeps the rest of the file untouched."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": json_path_property("File path, relative to the working directory or absolute."),
                "content": {"type": "string", "description": "Complete new contents of the file."}
            },
            "required": ["path", "content"]
        })
    }

    fn mutates(&self) -> bool {
        true
    }

    fn summarize(&self, raw: &str) -> String {
        match serde_json::from_str::<Value>(raw) {
            Ok(arguments) => {
                let path = arguments
                    .get("path")
                    .and_then(Value::as_str)
                    .unwrap_or("<path>");
                let lines = arguments
                    .get("content")
                    .and_then(Value::as_str)
                    .map(|content| content.lines().count())
                    .unwrap_or(0);
                format!("{path} ({lines} lines)")
            }
            Err(_) => short(raw),
        }
    }

    async fn run(&self, arguments: &Value, context: &ToolContext) -> Result<String> {
        let path = required_str(arguments, "path")?;
        let content = required_str(arguments, "content")?;
        let full = context.resolve(path);
        let display = display_path(&context.cwd, &full);
        if full.is_dir() {
            bail!("{display} is a directory");
        }
        if let Some(parent) = full.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("cannot create directory {}", parent.display()))?;
        }
        let previous = tokio::fs::read_to_string(&full).await.ok();
        tokio::fs::write(&full, content)
            .await
            .with_context(|| format!("cannot write {display}"))?;
        let lines = content.lines().count();
        let mut message = format!("Wrote {display}: {lines} lines, {} bytes.", content.len());
        if let Some(previous) = previous {
            if previous != content {
                let (added, removed) = diff_counts(&previous, content);
                message.push_str(&format!(
                    " Replaced previous content (+{added} −{removed} lines)."
                ));
            }
        }
        Ok(message)
    }
}

pub struct EditFile;

#[async_trait]
impl Tool for EditFile {
    fn name(&self) -> &'static str {
        "edit_file"
    }

    fn description(&self) -> &'static str {
        "Replace an exact text fragment in a file. old_string must match exactly once \
         unless replace_all is true. Include enough surrounding context to make the match unique."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": json_path_property("File path, relative to the working directory or absolute."),
                "old_string": {"type": "string", "description": "Exact text to replace; include surrounding context."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring a unique match."}
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    fn mutates(&self) -> bool {
        true
    }

    fn summarize(&self, raw: &str) -> String {
        match serde_json::from_str::<Value>(raw) {
            Ok(arguments) => {
                let path = arguments
                    .get("path")
                    .and_then(Value::as_str)
                    .unwrap_or("<path>");
                path.to_string()
            }
            Err(_) => short(raw),
        }
    }

    async fn run(&self, arguments: &Value, context: &ToolContext) -> Result<String> {
        let path = required_str(arguments, "path")?;
        let old = required_str(arguments, "old_string")?;
        let new = required_str(arguments, "new_string")?;
        let replace_all = optional_bool(arguments, "replace_all").unwrap_or(false);
        if old.is_empty() {
            bail!("old_string must not be empty");
        }
        if old == new {
            bail!("old_string and new_string are identical");
        }
        let full = context.resolve(path);
        let display = display_path(&context.cwd, &full);
        let content = tokio::fs::read_to_string(&full)
            .await
            .with_context(|| format!("cannot read {display}"))?;
        let occurrences = content.matches(old).count();
        if occurrences == 0 {
            bail!("old_string was not found in {display}; read the file and copy the exact text");
        }
        if occurrences > 1 && !replace_all {
            bail!(
                "old_string occurs {occurrences} times in {display}; include more surrounding \
                 context or set replace_all=true"
            );
        }
        let updated = if replace_all {
            content.replace(old, new)
        } else {
            content.replacen(old, new, 1)
        };
        tokio::fs::write(&full, &updated)
            .await
            .with_context(|| format!("cannot write {display}"))?;
        let (added, removed) = diff_counts(&content, &updated);
        Ok(format!(
            "Edited {display}: {occurrences} replacement(s) (+{added} −{removed} lines)."
        ))
    }
}

pub struct ListDir;

#[async_trait]
impl Tool for ListDir {
    fn name(&self) -> &'static str {
        "list_dir"
    }

    fn description(&self) -> &'static str {
        "List the contents of a directory as a tree. Hidden files are included; \
         .git, node_modules and target are skipped."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": json_path_property("Directory to list (default: the working directory)."),
                "depth": {"type": "integer", "minimum": 1, "maximum": 3, "description": "Recursion depth (default 1)."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Maximum entries to return (default 300)."}
            }
        })
    }

    fn summarize(&self, raw: &str) -> String {
        match serde_json::from_str::<Value>(raw) {
            Ok(arguments) => arguments
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or(".")
                .to_string(),
            Err(_) => short(raw),
        }
    }

    async fn run(&self, arguments: &Value, context: &ToolContext) -> Result<String> {
        let path = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
        let depth = optional_usize(arguments, "depth").unwrap_or(1).clamp(1, 3);
        let limit = optional_usize(arguments, "limit")
            .unwrap_or(300)
            .clamp(1, 1000);
        let root = context.resolve(path);
        if !root.is_dir() {
            bail!("{} is not a directory", display_path(&context.cwd, &root));
        }
        let display = display_path(&context.cwd, &root);
        let mut entries = Vec::new();
        let mut truncated = false;
        walk_dir(&root, 0, depth, limit, &mut entries, &mut truncated)?;
        entries.sort();
        if entries.is_empty() {
            return Ok(format!("{display} is empty"));
        }
        let mut output = format!("{display}:\n");
        for entry in entries {
            output.push_str(&entry);
            output.push('\n');
        }
        if truncated {
            output.push_str("\n[entry limit reached; use a smaller depth or a path]\n");
        }
        Ok(output)
    }
}

fn walk_dir(
    directory: &Path,
    level: usize,
    max_depth: usize,
    limit: usize,
    entries: &mut Vec<String>,
    truncated: &mut bool,
) -> Result<()> {
    let skip = [".git", "node_modules", "target", ".venv", "__pycache__"];
    let mut children: Vec<_> = std::fs::read_dir(directory)
        .with_context(|| format!("cannot list {}", directory.display()))?
        .filter_map(|entry| entry.ok())
        .collect();
    children.sort_by_key(|entry| entry.file_name());
    for child in children {
        if entries.len() >= limit {
            *truncated = true;
            return Ok(());
        }
        let name = child.file_name().to_string_lossy().to_string();
        if skip.contains(&name.as_str()) {
            continue;
        }
        let file_type = child
            .file_type()
            .with_context(|| format!("cannot stat {}", child.path().display()))?;
        let indent = "  ".repeat(level);
        if file_type.is_dir() {
            entries.push(format!("{indent}{name}/"));
            if level + 1 < max_depth {
                walk_dir(
                    &child.path(),
                    level + 1,
                    max_depth,
                    limit,
                    entries,
                    truncated,
                )?;
            }
        } else {
            let size = child.metadata().map(|meta| meta.len()).unwrap_or(0);
            entries.push(format!("{indent}{name} ({size} B)"));
        }
    }
    Ok(())
}

fn diff_counts(old: &str, new: &str) -> (usize, usize) {
    let diff = TextDiff::from_lines(old, new);
    let mut added = 0;
    let mut removed = 0;
    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Insert => added += 1,
            ChangeTag::Delete => removed += 1,
            ChangeTag::Equal => {}
        }
    }
    (added, removed)
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

/// Convenience wrapper used by the TUI's direct shell mode.
pub async fn run_shell_command(command: &str, cwd: &Path, timeout: Duration) -> Result<String> {
    super::shell::run_command(command, cwd, timeout).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_counts_reports_changes() {
        let (added, removed) = diff_counts("a\nb\n", "a\nc\nd\n");
        assert_eq!(added, 2);
        assert_eq!(removed, 1);
    }
}
