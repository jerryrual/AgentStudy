//! Search tools: content search with regular expressions and path matching.

use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use globset::{GlobBuilder, GlobMatcher};
use ignore::WalkBuilder;
use regex::RegexBuilder;
use serde_json::{json, Value};

use super::{
    display_path, json_path_property, optional_bool, optional_usize, required_str, Tool,
    ToolContext,
};

const MAX_SEARCH_BYTES: u64 = 2 * 1024 * 1024;
const MAX_LINE_WIDTH: usize = 400;

pub struct Grep;

#[async_trait]
impl Tool for Grep {
    fn name(&self) -> &'static str {
        "grep"
    }

    fn description(&self) -> &'static str {
        "Search file contents with a regular expression. Respects .gitignore and skips \
         hidden files, binary files and files larger than 2 MiB. Returns path:line: text."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Rust/PCRE-style regular expression."},
                "path": json_path_property("File or directory to search (default: the working directory)."),
                "glob": {"type": "string", "description": "Only search files matching this glob, e.g. \"*.rs\"."},
                "case_sensitive": {"type": "boolean", "description": "Match case sensitively (default false)."},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 500, "description": "Maximum matches to return (default 60)."}
            },
            "required": ["pattern"]
        })
    }

    fn parallel_safe(&self) -> bool {
        true
    }

    fn summarize(&self, raw: &str) -> String {
        match serde_json::from_str::<Value>(raw) {
            Ok(arguments) => {
                let pattern = arguments
                    .get("pattern")
                    .and_then(Value::as_str)
                    .unwrap_or("<pattern>");
                let path = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
                format!("{pattern:?} in {path}")
            }
            Err(_) => short(raw),
        }
    }

    async fn run(&self, arguments: &Value, context: &ToolContext) -> Result<String> {
        let pattern = required_str(arguments, "pattern")?;
        let path = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
        let case_sensitive = optional_bool(arguments, "case_sensitive").unwrap_or(false);
        let max_results = optional_usize(arguments, "max_results")
            .unwrap_or(60)
            .clamp(1, 500);
        let glob = arguments.get("glob").and_then(Value::as_str);
        let regex = RegexBuilder::new(pattern)
            .case_insensitive(!case_sensitive)
            .build()
            .map_err(|error| anyhow!("invalid regular expression: {error}"))?;
        let matcher = match glob {
            Some(glob) => Some(compile_glob(glob)?),
            None => None,
        };
        let root = context.resolve(path);
        if !root.exists() {
            bail!("{} does not exist", display_path(&context.cwd, &root));
        }

        let mut results = Vec::new();
        let mut truncated = false;
        let mut search_file = |file: &Path, results: &mut Vec<String>| -> bool {
            if results.len() >= max_results {
                return false;
            }
            if let Some(matcher) = &matcher {
                let relative = file.strip_prefix(&root).unwrap_or(file);
                if !matcher.is_match(relative) && !matcher.is_match(file) {
                    return true;
                }
            }
            let metadata = match std::fs::metadata(file) {
                Ok(metadata) => metadata,
                Err(_) => return true,
            };
            if metadata.len() > MAX_SEARCH_BYTES {
                return true;
            }
            let bytes = match std::fs::read(file) {
                Ok(bytes) => bytes,
                Err(_) => return true,
            };
            if is_binary(&bytes) {
                return true;
            }
            let Ok(text) = String::from_utf8(bytes) else {
                return true;
            };
            let display = display_path(&context.cwd, file);
            for (index, line) in text.lines().enumerate() {
                if regex.is_match(line) {
                    let line = truncate_line(line);
                    results.push(format!("{display}:{}: {line}", index + 1));
                    if results.len() >= max_results {
                        truncated = true;
                        return false;
                    }
                }
            }
            true
        };

        if root.is_file() {
            search_file(&root, &mut results);
        } else {
            let mut builder = WalkBuilder::new(&root);
            builder.standard_filters(true);
            for entry in builder.build() {
                let entry = entry.with_context(|| "failed while walking the directory")?;
                let Some(file_type) = entry.file_type() else {
                    continue;
                };
                if !file_type.is_file() {
                    continue;
                }
                if !search_file(entry.path(), &mut results) {
                    break;
                }
            }
        }

        if results.is_empty() {
            return Ok(format!(
                "No matches for {pattern:?} in {}.",
                display_path(&context.cwd, &root)
            ));
        }
        let mut output = results.join("\n");
        if truncated {
            output.push_str(&format!(
                "\n[limited to {max_results} matches; refine the pattern or path]"
            ));
        }
        Ok(output)
    }
}

pub struct Glob;

#[async_trait]
impl Tool for Glob {
    fn name(&self) -> &'static str {
        "glob"
    }

    fn description(&self) -> &'static str {
        "Find files by glob pattern, e.g. \"**/*.rs\" or \"src/**/mod.rs\". \
          Respects .gitignore and skips hidden files."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern such as \"**/*.rs\"."},
                "path": json_path_property("Directory to search (default: the working directory)."),
                "max_results": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Maximum paths to return (default 200)."}
            },
            "required": ["pattern"]
        })
    }

    fn parallel_safe(&self) -> bool {
        true
    }

    fn summarize(&self, raw: &str) -> String {
        match serde_json::from_str::<Value>(raw) {
            Ok(arguments) => arguments
                .get("pattern")
                .and_then(Value::as_str)
                .unwrap_or("<pattern>")
                .to_string(),
            Err(_) => short(raw),
        }
    }

    async fn run(&self, arguments: &Value, context: &ToolContext) -> Result<String> {
        let pattern = required_str(arguments, "pattern")?;
        let path = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
        let max_results = optional_usize(arguments, "max_results")
            .unwrap_or(200)
            .clamp(1, 1000);
        let matcher = compile_glob(pattern)?;
        let root = context.resolve(path);
        if !root.exists() {
            bail!("{} does not exist", display_path(&context.cwd, &root));
        }
        let mut matches = Vec::new();
        let mut truncated = false;
        let consider = |file: &Path, matches: &mut Vec<String>| {
            let relative = file.strip_prefix(&root).unwrap_or(file);
            if matcher.is_match(relative) || matcher.is_match(file) {
                matches.push(display_path(&context.cwd, file));
            }
        };
        if root.is_file() {
            consider(&root, &mut matches);
        } else {
            let mut builder = WalkBuilder::new(&root);
            builder.standard_filters(true);
            for entry in builder.build() {
                let entry = entry.with_context(|| "failed while walking the directory")?;
                if entry
                    .file_type()
                    .is_none_or(|file_type| !file_type.is_file())
                {
                    continue;
                }
                consider(entry.path(), &mut matches);
                if matches.len() >= max_results {
                    truncated = true;
                    break;
                }
            }
        }
        if matches.is_empty() {
            return Ok(format!("No files match {pattern:?}."));
        }
        matches.sort();
        let mut output = matches.join("\n");
        if truncated {
            output.push_str(&format!("\n[limited to {max_results} paths]"));
        }
        Ok(output)
    }
}

fn compile_glob(pattern: &str) -> Result<GlobMatcher> {
    GlobBuilder::new(pattern)
        .literal_separator(false)
        .build()
        .map(|glob| glob.compile_matcher())
        .map_err(|error| anyhow!("invalid glob pattern: {error}"))
}

fn is_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8_000).any(|byte| *byte == 0)
}

fn truncate_line(line: &str) -> String {
    let line = line.trim_end();
    if line.len() <= MAX_LINE_WIDTH {
        return line.to_string();
    }
    let mut end = MAX_LINE_WIDTH;
    while end > 0 && !line.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &line[..end])
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
