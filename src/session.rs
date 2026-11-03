//! Append-only JSONL session transcripts.
//!
//! Every turn is written as it happens so a session can be inspected or
//! resumed after a crash, mirroring iCode's session history. The format is a
//! sequence of JSON lines; the first line is the session metadata.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::model::{AssistantMessage, Message, ToolCall};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    /// Creation time in milliseconds since the Unix epoch.
    pub created_ms: u64,
    pub cwd: String,
    pub model: String,
    pub agent: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Record {
    Meta {
        meta: SessionMeta,
    },
    User {
        text: String,
    },
    Assistant {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        call_id: String,
        name: String,
        content: String,
    },
    Usage {
        prompt_tokens: u64,
        completion_tokens: u64,
    },
}

impl Record {
    pub fn from_message(message: &Message) -> Option<Self> {
        match message {
            Message::System { .. } => None,
            Message::User { content } => Some(Self::User {
                text: content.clone(),
            }),
            Message::Assistant { message } => Some(Self::Assistant {
                content: message.content.clone(),
                reasoning: message.reasoning.clone(),
                tool_calls: message.tool_calls.clone(),
            }),
            Message::Tool {
                call_id,
                name,
                content,
            } => Some(Self::Tool {
                call_id: call_id.clone(),
                name: name.clone(),
                content: content.clone(),
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Session {
    pub meta: SessionMeta,
    path: PathBuf,
}

impl Session {
    /// Append one record, flushing immediately.
    pub fn append(&self, record: &Record) -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)
            .with_context(|| format!("failed to open session {}", self.path.display()))?;
        let line = serde_json::to_string(record)?;
        writeln!(file, "{line}")
            .with_context(|| format!("failed to write session {}", self.path.display()))?;
        Ok(())
    }

    /// Append a conversation message, skipping system prompts.
    pub fn append_message(&self, message: &Message) -> Result<()> {
        if let Some(record) = Record::from_message(message) {
            self.append(&record)?;
        }
        Ok(())
    }

    pub fn id(&self) -> &str {
        &self.meta.id
    }
}

#[derive(Debug, Clone)]
pub struct LoadedSession {
    pub meta: SessionMeta,
    pub records: Vec<Record>,
}

impl LoadedSession {
    /// Rebuild the model conversation (system prompt excluded).
    pub fn messages(&self) -> Vec<Message> {
        self.records
            .iter()
            .filter_map(|record| match record {
                Record::User { text } => Some(Message::user(text)),
                Record::Assistant {
                    content,
                    reasoning,
                    tool_calls,
                } => Some(Message::assistant(AssistantMessage {
                    content: content.clone(),
                    reasoning: reasoning.clone(),
                    tool_calls: tool_calls.clone(),
                })),
                Record::Tool {
                    call_id,
                    name,
                    content,
                } => Some(Message::tool(call_id, name, content)),
                Record::Meta { .. } | Record::Usage { .. } => None,
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub id: String,
    pub created_ms: u64,
    pub cwd: String,
    pub model: String,
    /// First line of the first user message.
    pub preview: String,
}

/// Directory of JSONL session files.
#[derive(Debug, Clone)]
pub struct SessionStore {
    root: PathBuf,
}

impl SessionStore {
    /// Store under the iCode data directory.
    pub fn open() -> Self {
        let root = crate::config::Settings::data_dir().join("sessions");
        Self { root }
    }

    /// Store under an explicit directory (used by tests).
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Start a new session and write its metadata line.
    pub fn create(&self, cwd: &str, model: &str, agent: &str) -> Result<Session> {
        let created_ms = now_ms();
        let id = next_id(created_ms);
        std::fs::create_dir_all(&self.root)
            .with_context(|| format!("failed to create {}", self.root.display()))?;
        let meta = SessionMeta {
            id: id.clone(),
            created_ms,
            cwd: cwd.to_string(),
            model: model.to_string(),
            agent: agent.to_string(),
        };
        let session = Session {
            meta: meta.clone(),
            path: self.path(&id),
        };
        session.append(&Record::Meta { meta })?;
        Ok(session)
    }

    pub fn path(&self, id: &str) -> PathBuf {
        self.root.join(format!("{id}.jsonl"))
    }

    /// Open an existing session for appending.
    pub fn reopen(&self, id: &str) -> Result<Session> {
        let loaded = self.load(id)?;
        Ok(Session {
            meta: loaded.meta,
            path: self.path(id),
        })
    }

    /// Load a session by id.
    pub fn load(&self, id: &str) -> Result<LoadedSession> {
        if id.is_empty()
            || !id.chars().all(|character| {
                character.is_ascii_alphanumeric() || character == '-' || character == '_'
            })
        {
            bail!("invalid session id {id:?}");
        }
        let path = self.path(id);
        let file =
            std::fs::File::open(&path).with_context(|| format!("session {id:?} was not found"))?;
        let reader = BufReader::new(file);
        let mut meta: Option<SessionMeta> = None;
        let mut records = Vec::new();
        for (index, line) in reader.lines().enumerate() {
            let line = line.with_context(|| format!("failed to read session {id:?}"))?;
            if line.trim().is_empty() {
                continue;
            }
            let record: Record = serde_json::from_str(&line).with_context(|| {
                format!("invalid record on line {} of session {id:?}", index + 1)
            })?;
            if index == 0 {
                match &record {
                    Record::Meta { meta: first } => meta = Some(first.clone()),
                    _ => bail!("session {id:?} does not start with metadata"),
                }
            }
            records.push(record);
        }
        let meta = meta.with_context(|| format!("session {id:?} is empty"))?;
        Ok(LoadedSession { meta, records })
    }

    /// List sessions, newest first.
    pub fn list(&self) -> Result<Vec<SessionSummary>> {
        let mut summaries = Vec::new();
        let directory = match std::fs::read_dir(&self.root) {
            Ok(directory) => directory,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(summaries),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to list {}", self.root.display()));
            }
        };
        for entry in directory {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("jsonl") {
                continue;
            }
            let Some(id) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if let Ok(loaded) = self.load(id) {
                let preview = loaded
                    .records
                    .iter()
                    .find_map(|record| match record {
                        Record::User { text } => Some(preview_text(text)),
                        _ => None,
                    })
                    .unwrap_or_else(|| "(no messages)".to_string());
                summaries.push(SessionSummary {
                    id: loaded.meta.id.clone(),
                    created_ms: loaded.meta.created_ms,
                    cwd: loaded.meta.cwd.clone(),
                    model: loaded.meta.model.clone(),
                    preview,
                });
            }
        }
        summaries.sort_by_key(|summary| std::cmp::Reverse(summary.created_ms));
        Ok(summaries)
    }

    /// Id of the most recent session, if any.
    pub fn latest(&self) -> Result<Option<String>> {
        Ok(self.list()?.into_iter().next().map(|summary| summary.id))
    }
}

fn preview_text(text: &str) -> String {
    let mut preview = text.lines().next().unwrap_or_default().trim().to_string();
    if preview.chars().count() > 80 {
        preview = preview.chars().take(80).collect::<String>() + "…";
    }
    preview
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

fn next_id(created_ms: u64) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.subsec_nanos())
        .unwrap_or_default();
    let pid = std::process::id();
    format!("{created_ms:x}-{:05x}", (nanos ^ pid) & 0xfffff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_session() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store
            .create("/tmp/project", "deepseek-chat", "code")
            .unwrap();
        session
            .append(&Record::User {
                text: "hello".into(),
            })
            .unwrap();
        session
            .append(&Record::Assistant {
                content: Some("hi".into()),
                reasoning: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "bash".into(),
                    arguments: "{}".into(),
                }],
            })
            .unwrap();
        session
            .append(&Record::Tool {
                call_id: "call_1".into(),
                name: "bash".into(),
                content: "ok".into(),
            })
            .unwrap();

        let loaded = store.load(session.id()).unwrap();
        assert_eq!(loaded.meta.cwd, "/tmp/project");
        let messages = loaded.messages();
        assert_eq!(messages.len(), 3);
        assert!(matches!(&messages[0], Message::User { content } if content == "hello"));
        assert!(matches!(&messages[2], Message::Tool { name, .. } if name == "bash"));

        let summaries = store.list().unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].preview, "hello");
        assert_eq!(store.latest().unwrap().as_deref(), Some(session.id()));
    }

    #[test]
    fn invalid_ids_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        assert!(store.load("../escape").is_err());
        assert!(store.load("").is_err());
    }

    #[test]
    fn empty_store_lists_nothing() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        assert!(store.list().unwrap().is_empty());
        assert_eq!(store.latest().unwrap(), None);
    }
}
