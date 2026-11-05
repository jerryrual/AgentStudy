//! Headless execution: `icode run`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::json;

use crate::agent::{Agent, AgentKind, AgentOptions, EventSink};
use crate::compaction::{CompressionBatch, ContextProjection};
use crate::config::{ApprovalMode, ModelProfile, Settings};
use crate::interaction::{AllowAll, Approver, DenyAll, NoAsker};
use crate::model::{Message, ModelClient};
use crate::session::{Record, RecordBuffer, Session, SessionStore};
use crate::tools::ToolRegistry;

pub struct RunOptions {
    pub settings: Settings,
    pub profile: ModelProfile,
    pub agent: AgentKind,
    pub cwd: PathBuf,
    pub task: String,
    pub approval: ApprovalMode,
    /// Emit a single JSON object instead of streamed text and progress.
    pub json: bool,
    /// Continue an existing session instead of creating a new one.
    pub session: Option<String>,
    pub store: SessionStore,
}

struct RunSink {
    json: bool,
    recorded: Mutex<RecordedSession>,
    wrote_text: AtomicBool,
    save_failed: AtomicBool,
}

struct RecordedSession {
    session: Session,
    buffer: RecordBuffer,
}

impl RunSink {
    fn text_written(&self) -> bool {
        self.wrote_text.load(Ordering::Relaxed)
    }

    fn has_unsaved_messages(&self) -> bool {
        self.save_failed.load(Ordering::Relaxed)
    }
}

impl EventSink for RunSink {
    fn persist_compression(&self, batch: &CompressionBatch) -> Result<bool> {
        if self.has_unsaved_messages() {
            return Ok(false);
        }
        let mut recorded = self
            .recorded
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        recorded.session.append_compression_batch(batch)?;
        recorded.buffer.reset();
        Ok(true)
    }

    fn persist_context_projection(
        &self,
        source: &[Message],
        projection: &ContextProjection,
    ) -> Result<bool> {
        if self.has_unsaved_messages() {
            return Ok(false);
        }
        let mut recorded = self
            .recorded
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if recorded.buffer.has_pending() {
            return Ok(false);
        }
        recorded
            .session
            .append_archived_context_projection_if(source, projection, || true)?;
        recorded.buffer.reset();
        Ok(true)
    }

    fn text_delta(&self, delta: &str) {
        if self.json {
            return;
        }
        self.wrote_text.store(true, Ordering::Relaxed);
        print!("{delta}");
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }

    fn tool_started(&self, name: &str, summary: &str) {
        if !self.json {
            eprintln!("→ {name}  {summary}");
        }
    }

    fn tool_finished(
        &self,
        name: &str,
        ok: bool,
        duration: Duration,
        _summary: &str,
        _output: &str,
    ) {
        if !self.json {
            let mark = if ok { "✓" } else { "✗" };
            eprintln!("  {mark} {name} {:.1}s", duration.as_secs_f64());
        }
    }

    fn message_added(&self, message: &Message) {
        if let Ok(mut recorded) = self.recorded.lock() {
            let RecordedSession { session, buffer } = &mut *recorded;
            if let Err(error) = buffer.record(session, message) {
                self.save_failed.store(true, Ordering::Relaxed);
                eprintln!("warning: failed to save session: {error:#}");
            }
        }
    }

    fn info(&self, message: &str) {
        eprintln!("  · {message}");
    }

    fn error(&self, message: &str) {
        eprintln!("✗ {message}");
    }
}

/// Execute one task without the TUI.
///
/// Like iCode's `icode run`, no approval prompt is shown: mutating tools are
/// allowed unless `--approval deny` was requested.
pub async fn execute(options: RunOptions) -> Result<()> {
    let profile = options.profile.clone();
    let client = std::sync::Arc::new(
        tokio::task::spawn_blocking(move || ModelClient::from_profile(&profile))
            .await
            .map_err(|error| {
                anyhow::anyhow!("failed to resolve the model configuration: {error}")
            })??,
    );
    let store = options.store;
    let (session, mut messages): (Session, Vec<Message>) = match &options.session {
        Some(id) => {
            let loaded = store.load(id)?;
            (store.reopen(id)?, loaded.messages())
        }
        None => (
            store.create(
                &options.cwd.display().to_string(),
                &options.profile.model,
                options.agent.as_str(),
            )?,
            Vec::new(),
        ),
    };

    let task = options.task.trim().to_string();
    if task.is_empty() {
        anyhow::bail!("the task prompt must not be empty");
    }
    messages.push(Message::user(task.clone()));
    session.append(&Record::User { text: task.clone() })?;

    let approver: std::sync::Arc<dyn Approver> = match options.approval {
        ApprovalMode::Deny => std::sync::Arc::new(DenyAll),
        ApprovalMode::Ask | ApprovalMode::Auto => std::sync::Arc::new(AllowAll),
    };
    let agent = Agent::new(
        client,
        std::sync::Arc::new(ToolRegistry::builtin()),
        approver,
        std::sync::Arc::new(NoAsker),
        AgentOptions {
            kind: options.agent,
            cwd: options.cwd.clone(),
            max_rounds: options.settings.max_tool_rounds,
            tool_timeout: Duration::from_secs(120),
        },
    );

    let sink = RunSink {
        json: options.json,
        recorded: Mutex::new(RecordedSession {
            session: session.clone(),
            buffer: RecordBuffer::default(),
        }),
        wrote_text: AtomicBool::new(false),
        save_failed: AtomicBool::new(false),
    };

    let mut run = Box::pin(agent.run(&mut messages, &sink));
    let result = tokio::select! {
        result = run.as_mut() => result,
        _ = tokio::signal::ctrl_c() => {
            if !options.json {
                eprintln!();
            }
            eprintln!("✗ Interrupted");
            std::process::exit(130);
        }
    };
    drop(run);

    match result {
        Ok(turn) => {
            if !sink.has_unsaved_messages() && agent.should_force_compression(&turn) {
                if let Some(plan) = crate::compaction::plan_forced_compression(&messages) {
                    let checkpoint = Record::Compaction {
                        summary: plan.summary.clone(),
                        compacted_messages: Some(plan.compacted_messages),
                    };
                    match session.append(&checkpoint) {
                        Ok(()) => {
                            let mut recorded = sink
                                .recorded
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                            recorded.buffer.reset();
                        }
                        Err(error) => {
                            eprintln!("warning: failed to save automatic compaction: {error:#}");
                        }
                    }
                }
            }
            if options.json {
                let payload = json!({
                    "ok": true,
                    "agent": options.agent.as_str(),
                    "model": options.profile.model,
                    "session_id": session.id(),
                    "text": turn.text,
                    "usage": {
                        "prompt_tokens": turn.usage.prompt_tokens,
                        "completion_tokens": turn.usage.completion_tokens,
                        "total_tokens": turn.usage.total_tokens,
                    },
                    "tool_calls": turn.tool_calls,
                    "rounds": turn.rounds,
                    "duration_ms": turn.duration.as_millis(),
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else {
                if sink.text_written() {
                    println!();
                }
                eprintln!(
                    "✓ Done · {:.1}s · {} tool call(s) · session {}",
                    turn.duration.as_secs_f64(),
                    turn.tool_calls,
                    session.id()
                );
            }
            Ok(())
        }
        Err(error) => {
            if options.json {
                let payload = json!({
                    "ok": false,
                    "agent": options.agent.as_str(),
                    "model": options.profile.model,
                    "session_id": session.id(),
                    "error": format!("{error:#}"),
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
            }
            Err(error).context("the agent run failed")
        }
    }
}

/// Print the configured model profiles without exposing secrets.
pub fn list_models(settings: &Settings) -> Result<()> {
    if settings.models.is_empty() {
        println!(
            "No model profiles configured. Edit {}",
            Settings::path().display()
        );
        return Ok(());
    }
    let current = settings.current_model.as_deref().unwrap_or_default();
    println!(
        "    {:<16} {:<12} {:<32} {:<20} KEY",
        "NAME", "PROVIDER", "MODEL", "BASE URL"
    );
    for profile in &settings.models {
        let marker = if profile.name == current { "*" } else { " " };
        println!(
            "{:<3} {:<16} {:<12} {:<32} {:<20} {}",
            marker,
            truncate(&profile.name, 16),
            truncate(
                if profile.provider.is_empty() {
                    "-"
                } else {
                    &profile.provider
                },
                12
            ),
            truncate(&profile.model, 32),
            truncate(&profile.resolved_base_url(), 44),
            profile.key_source(),
        );
    }
    Ok(())
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut result: String = text.chars().take(limit.saturating_sub(1)).collect();
    result.push('…');
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_sink_tracks_unsaved_messages_for_compaction_guard() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        std::fs::remove_dir_all(directory.path()).unwrap();
        let sink = RunSink {
            json: true,
            recorded: Mutex::new(RecordedSession {
                session,
                buffer: RecordBuffer::default(),
            }),
            wrote_text: AtomicBool::new(false),
            save_failed: AtomicBool::new(false),
        };

        sink.message_added(&Message::assistant(crate::model::AssistantMessage::text(
            "answer",
        )));

        assert!(sink.has_unsaved_messages());
        assert!(!sink
            .persist_compression(&CompressionBatch::default())
            .unwrap());
    }

    #[test]
    fn run_sink_persists_projection_after_buffered_tool_group() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("current")).unwrap();
        let sink = RunSink {
            json: true,
            recorded: Mutex::new(RecordedSession {
                session: session.clone(),
                buffer: RecordBuffer::default(),
            }),
            wrote_text: AtomicBool::new(false),
            save_failed: AtomicBool::new(false),
        };
        sink.message_added(&Message::assistant(crate::model::AssistantMessage {
            content: Some("assistant context".into()),
            reasoning: Some("private reasoning".into()),
            tool_calls: vec![crate::model::ToolCall {
                id: "call_1".into(),
                name: "grep".into(),
                arguments: r#"{"pattern":"private-token"}"#.into(),
            }],
            responses_items: vec![serde_json::json!({"encrypted_content": "opaque-secret"})],
        }));
        sink.message_added(&Message::tool("call_1", "grep", "large result"));
        let projection = ContextProjection {
            historical_folds: Vec::new(),
            current_turn: Some(crate::compaction::CurrentTurnFold {
                note: "fresh progress".into(),
                compacted_messages: 2,
            }),
        };

        let source = vec![
            Message::user("current"),
            Message::assistant(crate::model::AssistantMessage {
                content: Some("assistant context".into()),
                reasoning: Some("private reasoning".into()),
                tool_calls: vec![crate::model::ToolCall {
                    id: "call_1".into(),
                    name: "grep".into(),
                    arguments: r#"{"pattern":"private-token"}"#.into(),
                }],
                responses_items: vec![serde_json::json!({"encrypted_content": "opaque-secret"})],
            }),
            Message::tool("call_1", "grep", "large result"),
        ];
        assert!(sink
            .persist_context_projection(&source, &projection)
            .unwrap());

        let loaded = store.load(session.id()).unwrap();
        assert!(matches!(
            loaded.records.last(),
            Some(Record::ContextProjection { .. })
        ));
        assert!(matches!(
            loaded.messages().last(),
            Some(Message::LastWords { .. })
        ));
        let archive = crate::compaction_archive::ArchiveStore::with_root(session.archive_root())
            .reconcile()
            .unwrap();
        assert_eq!(archive.live_records.len(), 1);
        let catalog = &archive.live_records[0];
        let archived =
            std::fs::read_to_string(session.archive_root().join(&catalog.relative_path)).unwrap();
        let record_id_line = format!("Record ID: `{}`", catalog.record_id);
        for expected in [
            "assistant context",
            "private reasoning",
            "\"pattern\": \"private-token\"",
            "opaque-secret",
            "large result",
            "Group ID: `call_1`",
            record_id_line.as_str(),
        ] {
            assert!(
                archived.contains(expected),
                "missing archive detail: {expected}"
            );
        }
    }

    #[test]
    fn run_sink_preserves_raw_history_when_archive_fails() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("current")).unwrap();
        let sink = RunSink {
            json: true,
            recorded: Mutex::new(RecordedSession {
                session: session.clone(),
                buffer: RecordBuffer::default(),
            }),
            wrote_text: AtomicBool::new(false),
            save_failed: AtomicBool::new(false),
        };
        let assistant = Message::assistant(crate::model::AssistantMessage {
            tool_calls: vec![crate::model::ToolCall {
                id: "call_1".into(),
                name: "grep".into(),
                arguments: "{}".into(),
            }],
            ..crate::model::AssistantMessage::default()
        });
        let result = Message::tool("call_1", "grep", "complete result");
        sink.message_added(&assistant);
        sink.message_added(&result);
        std::fs::create_dir_all(session.archive_root()).unwrap();
        std::fs::write(session.archive_root().join("compactions"), "blocked").unwrap();
        let source = vec![Message::user("current"), assistant, result];
        let projection = ContextProjection {
            historical_folds: Vec::new(),
            current_turn: Some(crate::compaction::CurrentTurnFold {
                note: "fresh progress".into(),
                compacted_messages: 2,
            }),
        };

        assert!(sink
            .persist_context_projection(&source, &projection)
            .is_err());

        let loaded = store.load(session.id()).unwrap();
        assert!(loaded
            .records
            .iter()
            .all(|record| !matches!(record, Record::ContextProjection { .. })));
        assert!(matches!(
            loaded.messages().last(),
            Some(Message::Tool { content, .. }) if content == "complete result"
        ));
    }

    #[test]
    fn run_sink_returns_error_when_compression_append_fails() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        std::fs::remove_dir_all(directory.path()).unwrap();
        let sink = RunSink {
            json: true,
            recorded: Mutex::new(RecordedSession {
                session,
                buffer: RecordBuffer::default(),
            }),
            wrote_text: AtomicBool::new(false),
            save_failed: AtomicBool::new(false),
        };

        assert!(sink
            .persist_compression(&CompressionBatch::default())
            .is_err());
    }
}
