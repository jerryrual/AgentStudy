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
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        responses_items: Vec<serde_json::Value>,
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
    Compaction {
        summary: String,
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
                responses_items: message.responses_items.clone(),
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
    ///
    /// A cancelled turn can leave an assistant `tool_calls` record without all
    /// of its tool results. OpenAI-compatible providers reject such a history,
    /// so unanswered calls are completed with an explanatory synthetic result
    /// and orphan tool records are dropped.
    pub fn messages(&self) -> Vec<Message> {
        let mut messages = Vec::new();
        let mut pending: Vec<(String, String)> = Vec::new();
        let flush = |messages: &mut Vec<Message>, pending: &mut Vec<(String, String)>| {
            for (id, name) in pending.drain(..) {
                messages.push(Message::tool(
                    id,
                    name,
                    "Error: the tool call was interrupted before it produced a result.",
                ));
            }
        };
        for record in &self.records {
            match record {
                Record::User { text } => {
                    flush(&mut messages, &mut pending);
                    messages.push(Message::user(text));
                }
                Record::Assistant {
                    content,
                    reasoning,
                    tool_calls,
                    responses_items,
                } => {
                    flush(&mut messages, &mut pending);
                    pending = tool_calls
                        .iter()
                        .map(|call| (call.id.clone(), call.name.clone()))
                        .collect();
                    messages.push(Message::assistant(AssistantMessage {
                        content: content.clone(),
                        reasoning: reasoning.clone(),
                        tool_calls: tool_calls.clone(),
                        responses_items: responses_items.clone(),
                    }));
                }
                Record::Tool {
                    call_id,
                    name,
                    content,
                } => {
                    if let Some(position) = pending.iter().position(|(id, _)| id == call_id) {
                        pending.remove(position);
                        messages.push(Message::tool(call_id, name, content));
                    }
                }
                Record::Compaction { summary } => {
                    pending.clear();
                    messages.clear();
                    messages.push(crate::compaction::summary_message(summary));
                }
                Record::Meta { .. } | Record::Usage { .. } => {}
            }
        }
        flush(&mut messages, &mut pending);
        messages
    }
}

/// Buffers a tool-calling assistant message until all of its tool results have
/// been recorded, so an interrupted turn never leaves an unpaired tool call in
/// the session file.
#[derive(Debug, Default)]
pub struct RecordBuffer {
    pending: Option<PendingGroup>,
}

#[derive(Debug)]
struct PendingGroup {
    expected: usize,
    assistant: Message,
    tools: Vec<Message>,
}

impl RecordBuffer {
    /// Record one conversation message, grouping tool calls with their results.
    pub fn record(&mut self, session: &Session, message: &Message) -> Result<()> {
        if let Message::Assistant { message: assistant } = message {
            if !assistant.tool_calls.is_empty() {
                if self.pending.take().is_some() {
                    // An incomplete group is dropped rather than persisted in a
                    // form providers reject.
                }
                self.pending = Some(PendingGroup {
                    expected: assistant.tool_calls.len(),
                    assistant: message.clone(),
                    tools: Vec::new(),
                });
                return Ok(());
            }
        }
        if matches!(message, Message::Tool { .. }) {
            remove_pending_if_orphan(session, &mut self.pending, message)?;
            return Ok(());
        }
        // Any other message ends an interrupted tool group without persisting it.
        self.pending = None;
        session.append_message(message)
    }

    /// Drop the buffered group (used when switching sessions).
    pub fn reset(&mut self) {
        self.pending = None;
    }

    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    fn flush_pending(session: &Session, group: PendingGroup) -> Result<()> {
        session.append_message(&group.assistant)?;
        for tool in group.tools {
            session.append_message(&tool)?;
        }
        Ok(())
    }
}

fn remove_pending_if_orphan(
    session: &Session,
    pending: &mut Option<PendingGroup>,
    message: &Message,
) -> Result<()> {
    let Some(group) = pending.as_mut() else {
        // A tool record without an assistant tool-call record cannot be replayed.
        return Ok(());
    };
    group.tools.push(message.clone());
    if group.tools.len() < group.expected {
        return Ok(());
    }
    let group = pending.take().expect("checked above");
    RecordBuffer::flush_pending(session, group)
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
                responses_items: vec![serde_json::json!({
                    "type": "function_call",
                    "id": "fc_call_1",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": "{}",
                    "status": "completed"
                })],
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
        assert!(matches!(
            &messages[1],
            Message::Assistant { message }
                if message.responses_items[0]["id"] == "fc_call_1"
        ));
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

    #[test]
    fn messages_repair_dangling_tool_calls() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session
            .append(&Record::Assistant {
                content: Some("partial".into()),
                reasoning: None,
                tool_calls: vec![
                    ToolCall {
                        id: "call_1".into(),
                        name: "read_file".into(),
                        arguments: "{}".into(),
                    },
                    ToolCall {
                        id: "call_2".into(),
                        name: "bash".into(),
                        arguments: "{}".into(),
                    },
                ],
                responses_items: Vec::new(),
            })
            .unwrap();
        session
            .append(&Record::Tool {
                call_id: "call_1".into(),
                name: "read_file".into(),
                content: "ok".into(),
            })
            .unwrap();
        let loaded = store.load(session.id()).unwrap();
        let messages = loaded.messages();
        assert_eq!(messages.len(), 3, "{messages:?}");
        match &messages[2] {
            Message::Tool {
                call_id, content, ..
            } => {
                assert_eq!(call_id, "call_2");
                assert!(content.contains("interrupted"));
            }
            other => panic!("expected a synthetic tool result, got {other:?}"),
        }
    }

    #[test]
    fn messages_replay_only_the_latest_compacted_context() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session
            .append_message(&Message::user("old request"))
            .unwrap();
        session
            .append(&Record::Compaction {
                summary: "Changed src/main.rs and tests pass.".into(),
            })
            .unwrap();
        session.append_message(&Message::user("continue")).unwrap();

        let messages = store.load(session.id()).unwrap().messages();
        assert_eq!(messages.len(), 2, "{messages:?}");
        assert!(matches!(
            &messages[0],
            Message::User { content } if content.contains("Changed src/main.rs")
        ));
        assert_eq!(messages[1], Message::user("continue"));
    }

    #[test]
    fn record_buffer_defers_tool_call_groups() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        let mut buffer = RecordBuffer::default();
        let assistant = Message::assistant(AssistantMessage {
            content: Some("calling tools".into()),
            reasoning: None,
            tool_calls: vec![
                ToolCall {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    arguments: "{}".into(),
                },
                ToolCall {
                    id: "call_2".into(),
                    name: "bash".into(),
                    arguments: "{}".into(),
                },
            ],
            responses_items: Vec::new(),
        });
        buffer.record(&session, &assistant).unwrap();
        buffer
            .record(&session, &Message::tool("call_1", "read_file", "one"))
            .unwrap();
        let written = std::fs::read_to_string(store.path(session.id())).unwrap();
        assert_eq!(
            written.lines().count(),
            1,
            "only the metadata is written yet"
        );

        buffer
            .record(&session, &Message::tool("call_2", "bash", "two"))
            .unwrap();
        let written = std::fs::read_to_string(store.path(session.id())).unwrap();
        assert_eq!(written.lines().count(), 4, "assistant and both results now");
        assert!(written.contains("call_1"));
        assert!(written.contains("call_2"));
    }

    #[test]
    fn record_buffer_drops_an_interrupted_group() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        let mut buffer = RecordBuffer::default();
        let assistant = Message::assistant(AssistantMessage {
            content: None,
            reasoning: None,
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "bash".into(),
                arguments: "{}".into(),
            }],
            responses_items: Vec::new(),
        });
        buffer.record(&session, &assistant).unwrap();
        buffer
            .record(&session, &Message::user("next turn"))
            .unwrap();
        let loaded = store.load(session.id()).unwrap();
        assert!(loaded
            .records
            .iter()
            .all(|record| !matches!(record, Record::Assistant { .. })));
        let messages = loaded.messages();
        assert_eq!(messages.len(), 1);
        assert!(matches!(messages[0], Message::User { .. }));
    }
}
