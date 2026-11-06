//! Headless execution: `icode run`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::json;

use crate::agent::{Agent, AgentKind, AgentOptions, EventSink};
use crate::compaction::{CompressionBatch, ContextProjection};
use crate::compaction_archive::CurrentTurnArchiveManifest;
use crate::config::{ApprovalMode, ModelProfile, Settings};
use crate::interaction::{AllowAll, Approver, DenyAll, NoAsker};
use crate::model::{Message, ModelClient};
use crate::phase4_breaker::PhaseFourBreaker;
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
    /// Resume the latest user turn without appending a new User record.
    pub resume_turn: bool,
    pub store: SessionStore,
}

struct RunSink {
    json: bool,
    recorded: Mutex<RecordedSession>,
    wrote_text: AtomicBool,
    save_failed: AtomicBool,
    archive_pointer: Mutex<Option<String>>,
    current_turn_manifest: Mutex<CurrentTurnArchiveManifest>,
    phase_four_turn: u64,
    phase_four_breaker: PhaseFourBreaker,
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
    fn turn_started(&self) {
        let pointer = self
            .recorded
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .session
            .archive_catalog_pointer()
            .ok()
            .flatten();
        *self
            .archive_pointer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = pointer;
        *self
            .current_turn_manifest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            CurrentTurnArchiveManifest::default();
    }

    fn enter_phase_four_round(&self) -> anyhow::Result<bool> {
        let result = self.phase_four_breaker.enter_round_and_persist(|_, limit| {
            let recorded = self
                .recorded
                .lock()
                .map_err(|_| anyhow::anyhow!("recorded session lock is poisoned"))?;
            recorded
                .session
                .enter_phase_four_round_if(self.phase_four_turn, limit, || true)
        });
        if result.is_err() {
            self.save_failed.store(true, Ordering::Relaxed);
        }
        result
    }

    fn record_phase_four_failure(&self, disable: bool) -> anyhow::Result<()> {
        let result = self
            .phase_four_breaker
            .record_failure_and_persist(disable, |_, disable| {
                let recorded = self
                    .recorded
                    .lock()
                    .map_err(|_| anyhow::anyhow!("recorded session lock is poisoned"))?;
                recorded
                    .session
                    .record_phase_four_failure_if(self.phase_four_turn, disable, || true)
            });
        if result.is_err() {
            self.save_failed.store(true, Ordering::Relaxed);
        }
        result
    }

    fn record_phase_four_progress(&self, meaningful: bool) -> anyhow::Result<()> {
        let result =
            self.phase_four_breaker
                .record_progress_and_persist(meaningful, |_, meaningful| {
                    let recorded = self
                        .recorded
                        .lock()
                        .map_err(|_| anyhow::anyhow!("recorded session lock is poisoned"))?;
                    recorded.session.record_phase_four_progress_if(
                        self.phase_four_turn,
                        meaningful,
                        || true,
                    )
                });
        if result.is_err() {
            self.save_failed.store(true, Ordering::Relaxed);
        }
        result
    }

    fn checkpoint_phase_four_side_call(&self, estimated_input: u64) -> anyhow::Result<bool> {
        let result =
            self.phase_four_breaker
                .charge_and_persist(estimated_input, |previous, proposed| {
                    let recorded = self
                        .recorded
                        .lock()
                        .map_err(|_| anyhow::anyhow!("recorded session lock is poisoned"))?;
                    recorded.session.append_phase_four_breaker_if(
                        self.phase_four_turn,
                        previous,
                        proposed,
                        || true,
                    )
                });
        if result.is_err() {
            self.save_failed.store(true, Ordering::Relaxed);
        }
        result
    }

    fn archive_pointer(&self) -> Option<String> {
        self.archive_pointer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn current_turn_archive_manifest(&self, file_read_available: bool) -> Option<String> {
        let root = self
            .recorded
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .session
            .archive_root()
            .to_path_buf();
        self.current_turn_manifest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .render(&root, file_read_available)
    }

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
        let archived = {
            let mut recorded = self
                .recorded
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if recorded.buffer.has_pending() {
                return Ok(false);
            }
            let Some(archived) = recorded.session.append_archived_context_projection_if(
                source,
                projection,
                || true,
            )?
            else {
                return Ok(false);
            };
            recorded.buffer.reset();
            archived
        };
        self.current_turn_manifest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .append(archived);
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
    let (session, _run_lease) = match &options.session {
        Some(id) => {
            let lease = store.try_acquire_run_lease(id)?;
            (store.reopen(id)?, lease)
        }
        None if options.resume_turn => {
            anyhow::bail!("same-turn resume requires an existing session")
        }
        None => {
            let session = store.create(
                &options.cwd.display().to_string(),
                &options.profile.model,
                options.agent.as_str(),
            )?;
            let lease = store.try_acquire_run_lease(session.id())?;
            (session, lease)
        }
    };
    let loaded = options
        .session
        .as_deref()
        .map(|id| store.load(id))
        .transpose()?;

    let (mut messages, phase_four_turn, phase_four_state) = if options.resume_turn {
        session.load_resumable_turn()?
    } else {
        let mut messages = loaded.map(|loaded| loaded.messages()).unwrap_or_default();
        let task = options.task.trim().to_string();
        if task.is_empty() {
            anyhow::bail!("the task prompt must not be empty");
        }
        messages.push(Message::user(task.clone()));
        let (turn, state) = session.append_user_and_load_phase_four_breaker(&task)?;
        (messages, turn, state)
    };

    let approver: std::sync::Arc<dyn Approver> = if options.resume_turn {
        // The prior process may have exited after a mutating tool took effect but
        // before its result reached the durable transcript. Never replay writes.
        std::sync::Arc::new(DenyAll)
    } else {
        match options.approval {
            ApprovalMode::Deny => std::sync::Arc::new(DenyAll),
            ApprovalMode::Ask | ApprovalMode::Auto => std::sync::Arc::new(AllowAll),
        }
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
        archive_pointer: Mutex::new(None),
        current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
        phase_four_turn,
        phase_four_breaker: PhaseFourBreaker::new(
            options.settings.phase4_side_call_token_budget,
            phase_four_state,
        ),
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
    fn run_sink_checkpoints_side_call_spend_before_admission() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("current")).unwrap();
        let (turn, state) = session.load_phase_four_breaker().unwrap();
        let sink = RunSink {
            json: true,
            recorded: Mutex::new(RecordedSession {
                session: session.clone(),
                buffer: RecordBuffer::default(),
            }),
            wrote_text: AtomicBool::new(false),
            save_failed: AtomicBool::new(false),
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
            phase_four_turn: turn,
            phase_four_breaker: PhaseFourBreaker::new(100, state),
        };

        assert!(sink.checkpoint_phase_four_side_call(40).unwrap());
        assert!(!sink.checkpoint_phase_four_side_call(60).unwrap());
        let (_, restored) = session.load_phase_four_breaker().unwrap();
        assert_eq!(restored.side_call_tokens, 100);
        assert_eq!(
            std::fs::read_to_string(session.phase_four_breaker_path())
                .unwrap()
                .lines()
                .count(),
            2
        );
    }

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
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
            phase_four_turn: 0,
            phase_four_breaker: PhaseFourBreaker::new(-1, Default::default()),
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
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
            phase_four_turn: 0,
            phase_four_breaker: PhaseFourBreaker::new(-1, Default::default()),
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
        let manifest = sink.current_turn_archive_manifest(true).unwrap();
        assert!(manifest.contains("--- Dropped this turn"));
        assert!(manifest.contains("r1 001 grep(…) → unknown, 12 chars"));
        assert!(manifest.contains("To inspect archived input/output"));

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
        sink.turn_started();
        assert_eq!(sink.current_turn_archive_manifest(true), None);
        let frozen = sink.archive_pointer().unwrap();
        assert!(frozen.contains("archived 1 record from previous turns"));
        crate::compaction_archive::ArchiveStore::with_root(session.archive_root())
            .archive_round(&[crate::compaction_archive::ArchiveRecord {
                turn: 2,
                round: 1,
                sequence: 1,
                tool: "grep".into(),
                created_at: "2026-01-02T03:04:05Z".into(),
                kind: "group".into(),
                group_id: "later".into(),
                session_id: session.id().into(),
                markdown: "later record".into(),
            }])
            .unwrap();
        assert_eq!(sink.archive_pointer().as_deref(), Some(frozen.as_str()));
        assert!(session
            .archive_catalog_pointer()
            .unwrap()
            .unwrap()
            .contains("archived 2 records from previous turns"));
        sink.turn_started();
        assert!(sink
            .archive_pointer()
            .unwrap()
            .contains("archived 2 records from previous turns"));
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
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
            phase_four_turn: 0,
            phase_four_breaker: PhaseFourBreaker::new(-1, Default::default()),
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
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
            phase_four_turn: 0,
            phase_four_breaker: PhaseFourBreaker::new(-1, Default::default()),
        };

        assert!(sink
            .persist_compression(&CompressionBatch::default())
            .is_err());
    }
}
