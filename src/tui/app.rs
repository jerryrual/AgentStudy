//! TUI state and event handling.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::Rect;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};

use crate::agent::injection::{self, EnqueueError, InjectionQueue};
use crate::agent::{Agent, AgentKind, AgentOptions, EventSink, TurnResult};
use crate::compaction::{Compaction, CompressionBatch, ContextProjection};
use crate::compaction_archive::{ArchivedContextProjection, CurrentTurnArchiveManifest};
use crate::config::{ApprovalMode, ModelProfile, Settings};
use crate::interaction::{AllowAll, ApprovalDecision, ApprovalRequest, Approver, Asker, DenyAll};
use crate::model::{Message, ModelClient, Usage};
use crate::phase4_breaker::{PhaseFourBreaker, PhaseFourBreakerState};
use crate::recovery_writer::{RecoveryCheckpointHandle, RecoveryCheckpointWriter};
use crate::session::{Record, RecordBuffer, Session, SessionRunLease, SessionStore};
use crate::theme::Theme;
use crate::tools::{truncate_output, ToolRegistry};
use crate::APP_NAME;

/// One rendered entry in the transcript.
#[derive(Debug, Clone)]
pub enum ViewItem {
    User(String),
    Assistant {
        text: String,
        reasoning: String,
        streaming: bool,
    },
    Tool {
        name: String,
        summary: String,
        status: ToolStatus,
        ok: Option<bool>,
        duration: Option<Duration>,
        output: String,
    },
    Info(String),
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolStatus {
    Running,
    Done,
}

#[derive(Debug)]
pub enum Modal {
    Approval {
        request: ApprovalRequest,
        reply: oneshot::Sender<ApprovalDecision>,
        allow_always: bool,
        recovery: bool,
    },
    Help,
}

/// Events sent from the agent task and the interaction handlers.
pub(crate) enum UiEvent {
    Turn {
        id: u64,
        event: Box<UiEvent>,
    },
    TextDelta(String),
    ReasoningDelta(String),
    ToolStarted {
        name: String,
        summary: String,
    },
    ToolFinished {
        name: String,
        ok: bool,
        duration: Duration,
        summary: String,
        output: String,
    },
    Usage(Usage),
    MessageAdded(Message),
    PersistInjectedUser {
        message: Message,
        reply: std::sync::mpsc::Sender<std::result::Result<(), String>>,
    },
    PersistContextProjection {
        source: Vec<Message>,
        projection: ContextProjection,
        reply:
            std::sync::mpsc::Sender<std::result::Result<Option<ArchivedContextProjection>, String>>,
    },
    Info(String),
    Error(String),
    Approval {
        request: ApprovalRequest,
        reply: oneshot::Sender<ApprovalDecision>,
        allow_always: bool,
        recovery: bool,
    },
    Ask {
        question: String,
        reply: oneshot::Sender<String>,
    },
    TurnFinished(std::result::Result<TurnResult, String>, Vec<Message>, bool),
    CompactionFinished {
        result: std::result::Result<Compaction, String>,
        duration: Duration,
    },
}

/// Construction options, separated so tests can build an app without a
/// terminal or network access.
pub struct AppOptions {
    pub settings: Settings,
    pub profile: ModelProfile,
    pub agent: AgentKind,
    pub cwd: PathBuf,
    pub store: SessionStore,
    pub resume: Option<String>,
    pub theme: Theme,
    /// Effective approval mode for this run; a `--approval` override stays out
    /// of the persisted settings unless the user changes it with `/approval`.
    pub approval: ApprovalMode,
}

#[derive(Debug, Clone)]
struct PendingInjection {
    id: String,
    raw: String,
}

/// One direct shell command waiting for the event loop to own the terminal.
#[derive(Debug, Clone)]
pub(crate) struct DirectShellRequest {
    pub id: u64,
    pub command: String,
    pub cwd: PathBuf,
    pub started: Instant,
}

pub struct App {
    pub settings: Settings,
    pub profile: ModelProfile,
    pub agent_kind: AgentKind,
    pub approval_mode: ApprovalMode,
    pub cwd: PathBuf,
    pub theme: Theme,
    pub store: SessionStore,
    pub session: Session,
    session_run_lease: SessionRunLease,
    recorder: Arc<Mutex<RecordBuffer>>,
    pub(crate) unsaved_messages: bool,
    pub(crate) requires_session_reload: bool,
    pub messages: Vec<Message>,
    pub view: Vec<ViewItem>,
    pub input: String,
    pub cursor: usize,
    pub follow: bool,
    pub scroll_back: u16,
    pub running: bool,
    pub continue_available: bool,
    pub(crate) recovery_probe_failed: bool,
    turn_id: u64,
    pub turn_started: Option<Instant>,
    pub elapsed: Duration,
    pub tokens: Usage,
    pub spinner: usize,
    pub modal: Option<Modal>,
    pub pending_ask: Option<oneshot::Sender<String>>,
    pub history: Vec<String>,
    pub history_index: Option<usize>,
    pub quit: bool,
    events: mpsc::UnboundedReceiver<UiEvent>,
    events_tx: mpsc::UnboundedSender<UiEvent>,
    cancel_handle: Option<tokio::task::JoinHandle<()>>,
    injection_queue: Option<InjectionQueue>,
    pending_injection: Option<PendingInjection>,
    deferred_submission: bool,
    next_injection_id: u64,
    persistence_gate: Option<Arc<AtomicBool>>,
    recovery_writer: Option<RecoveryCheckpointWriter>,
    pending_direct_shell: Option<DirectShellRequest>,
    active_direct_shell: Option<u64>,
    pub approver: Arc<dyn Approver>,
    pub asker: Arc<dyn Asker>,
    pub tools: Arc<ToolRegistry>,
    pub runtime: Handle,
}

impl App {
    pub fn new(runtime: Handle, options: AppOptions) -> Result<Self> {
        let (events_tx, events) = mpsc::unbounded_channel();
        let approver: Arc<dyn Approver> = match options.approval {
            ApprovalMode::Auto => Arc::new(AllowAll),
            ApprovalMode::Deny => Arc::new(DenyAll),
            ApprovalMode::Ask => Arc::new(TuiApprover::new(events_tx.clone())),
        };
        let asker: Arc<dyn Asker> = Arc::new(TuiAsker {
            tx: events_tx.clone(),
        });
        let (session, session_run_lease, messages, mut view, opened_existing) =
            match options.resume.as_deref() {
                Some(id) => {
                    let lease = options.store.try_acquire_run_lease(id)?;
                    let loaded = options.store.load(id)?;
                    let messages = loaded.messages();
                    let view = view_from_records(&loaded.records);
                    let session = options.store.reopen(id)?;
                    (session, lease, messages, view, true)
                }
                None => {
                    let session = options.store.create(
                        &options.cwd.display().to_string(),
                        &options.profile.model,
                        options.agent.as_str(),
                    )?;
                    let lease = options.store.try_acquire_run_lease(session.id())?;
                    (session, lease, Vec::new(), welcome(), false)
                }
            };
        let (continue_available, recovery_probe_failed) = if opened_existing {
            inspect_continue_available(&session, &mut view)
        } else {
            (false, false)
        };
        if continue_available {
            view.push(continue_available_notice(options.approval));
        }
        Ok(Self {
            settings: options.settings,
            profile: options.profile,
            agent_kind: options.agent,
            approval_mode: options.approval,
            cwd: options.cwd,
            theme: options.theme,
            store: options.store,
            session,
            session_run_lease,
            recorder: Arc::new(Mutex::new(RecordBuffer::default())),
            unsaved_messages: false,
            requires_session_reload: false,
            messages,
            view,
            input: String::new(),
            cursor: 0,
            follow: true,
            scroll_back: 0,
            running: false,
            continue_available,
            recovery_probe_failed,
            turn_id: 0,
            turn_started: None,
            elapsed: Duration::default(),
            tokens: Usage::default(),
            spinner: 0,
            modal: None,
            pending_ask: None,
            history: Vec::new(),
            history_index: None,
            quit: false,
            events,
            events_tx,
            cancel_handle: None,
            injection_queue: None,
            pending_injection: None,
            deferred_submission: false,
            next_injection_id: 0,
            persistence_gate: None,
            recovery_writer: None,
            pending_direct_shell: None,
            active_direct_shell: None,
            approver,
            asker,
            tools: Arc::new(ToolRegistry::builtin()),
            runtime,
        })
    }

    /// Send an event into the app (used by tests).
    #[cfg(test)]
    pub(crate) fn push_event(&self, event: UiEvent) {
        let _ = self.events_tx.send(event);
    }

    fn record_message(&self, message: &Message) -> Result<()> {
        let mut recorder = self
            .recorder
            .lock()
            .map_err(|_| anyhow::anyhow!("session recorder lock is poisoned"))?;
        recorder.record(&self.session, message)
    }

    fn reset_recorder(&self) {
        self.recorder
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .reset();
    }

    #[cfg(test)]
    pub(crate) fn push_stale_event(&self, event: UiEvent) {
        let _ = self.events_tx.send(UiEvent::Turn {
            id: self.turn_id.wrapping_sub(1),
            event: Box::new(event),
        });
    }

    /// Drain pending events; returns true when the UI should redraw.
    pub fn pump(&mut self) -> bool {
        let mut changed = false;
        while let Ok(event) = self.events.try_recv() {
            self.apply(event);
            changed = true;
        }
        changed
    }

    pub fn tick(&mut self) {
        self.spinner = self.spinner.wrapping_add(1);
        if let Some(started) = self.turn_started {
            self.elapsed = started.elapsed();
        }
    }

    fn apply(&mut self, event: UiEvent) {
        match event {
            UiEvent::Turn { id, event } => {
                if id == self.turn_id {
                    self.apply(*event);
                }
            }
            UiEvent::TextDelta(delta) => self.append_assistant_text(&delta),
            UiEvent::ReasoningDelta(delta) => self.append_reasoning(&delta),
            UiEvent::ToolStarted { name, summary } => {
                if !self.running {
                    return;
                }
                self.view.push(ViewItem::Tool {
                    name,
                    summary,
                    status: ToolStatus::Running,
                    ok: None,
                    duration: None,
                    output: String::new(),
                });
            }
            UiEvent::ToolFinished {
                name,
                ok,
                duration,
                summary,
                output,
            } => {
                if !self.running {
                    return;
                }
                let preview = preview_output(&output);
                // Parallel batches emit starts and finishes in provider order.
                // Match the oldest identical running card so duplicate calls do
                // not swap their outputs.
                let found = self.view.iter_mut().find_map(|item| match item {
                    ViewItem::Tool {
                        name: tool_name,
                        summary: tool_summary,
                        status,
                        ok: tool_ok,
                        duration: tool_duration,
                        output: tool_output,
                    } if *status == ToolStatus::Running
                        && *tool_name == name
                        && *tool_summary == summary =>
                    {
                        *status = ToolStatus::Done;
                        *tool_ok = Some(ok);
                        *tool_duration = Some(duration);
                        *tool_output = preview.clone();
                        Some(())
                    }
                    _ => None,
                });
                if found.is_none() {
                    self.view.push(ViewItem::Tool {
                        name,
                        summary,
                        status: ToolStatus::Done,
                        ok: Some(ok),
                        duration: Some(duration),
                        output: preview,
                    });
                }
            }
            UiEvent::Usage(usage) => self.tokens.add(usage),
            UiEvent::MessageAdded(message) => {
                if let Err(error) = self.record_message(&message) {
                    self.unsaved_messages = true;
                    self.view.push(ViewItem::Error(format!(
                        "failed to save session: {error:#}"
                    )));
                }
                if let Message::Assistant { message } = &message {
                    if let Some(ViewItem::Assistant {
                        text, streaming, ..
                    }) = self.view.iter_mut().rev().find(
                        |item| matches!(item, ViewItem::Assistant { streaming, .. } if *streaming),
                    ) {
                        *streaming = false;
                        if let Some(content) = &message.content {
                            if text.is_empty() {
                                text.push_str(content);
                            }
                        }
                    }
                }
            }
            UiEvent::PersistInjectedUser { message, reply } => {
                let result = self
                    .record_message(&message)
                    .map_err(|error| format!("failed to save injected user message: {error:#}"));
                if result.is_err() {
                    self.unsaved_messages = true;
                } else if let Message::InjectedUser {
                    injection_id,
                    content,
                } = &message
                {
                    let matching = self
                        .pending_injection
                        .as_ref()
                        .is_some_and(|pending| pending.id == *injection_id);
                    let display = if matching {
                        self.pending_injection
                            .take()
                            .map(|pending| pending.raw)
                            .unwrap_or_else(|| content.clone())
                    } else {
                        content.clone()
                    };
                    self.view.push(ViewItem::User(display));
                    if matching && self.pending_ask.is_none() {
                        self.input.clear();
                        self.cursor = 0;
                    }
                }
                let _ = reply.send(result);
            }
            UiEvent::PersistContextProjection {
                source,
                projection,
                reply,
            } => {
                let result = persist_context_projection(
                    &self.session,
                    &self.recorder,
                    self.persistence_gate.as_ref(),
                    self.unsaved_messages,
                    &source,
                    &projection,
                );
                let _ = reply.send(result);
            }
            UiEvent::Info(message) => self.view.push(ViewItem::Info(message)),
            UiEvent::Error(message) => self.view.push(ViewItem::Error(message)),
            UiEvent::Approval {
                request,
                reply,
                allow_always,
                recovery,
            } => {
                self.modal = Some(Modal::Approval {
                    request,
                    reply,
                    allow_always,
                    recovery,
                });
            }
            UiEvent::Ask { question, reply } => {
                self.pending_ask = Some(reply);
                if self.pending_injection.is_some() {
                    self.input.clear();
                    self.cursor = 0;
                }
                self.view
                    .push(ViewItem::Info(format!("The assistant asks: {question}")));
                self.view.push(ViewItem::Info(
                    "Type your answer and press Enter.".to_string(),
                ));
            }
            UiEvent::TurnFinished(result, messages, force_compression) => {
                self.injection_queue = None;
                if let Some(pending) = self.pending_injection.take() {
                    if self.pending_ask.is_none() {
                        self.input = pending.raw;
                        self.cursor = self.char_len();
                    }
                }
                let submit_deferred = std::mem::take(&mut self.deferred_submission);
                let mut recovery_writer = self.recovery_writer.take();
                self.persistence_gate = None;
                self.messages = messages;
                self.running = false;
                self.turn_started = None;
                for item in self.view.iter_mut().rev() {
                    if let ViewItem::Assistant { streaming, .. } = item {
                        *streaming = false;
                    }
                }
                match result {
                    Ok(turn) => {
                        self.elapsed = turn.duration;
                        if force_compression && !self.unsaved_messages {
                            match self.store.load(self.session.id()) {
                                Ok(loaded) => {
                                    let persisted_messages = loaded.messages();
                                    if let Some(plan) = crate::compaction::plan_forced_compression(
                                        &persisted_messages,
                                    ) {
                                        let record = Record::Compaction {
                                            summary: plan.summary.clone(),
                                            compacted_messages: Some(plan.compacted_messages),
                                        };
                                        match self.session.append(&record) {
                                            Ok(()) => {
                                                self.messages = plan.projected(&persisted_messages);
                                                self.reset_recorder();
                                            }
                                            Err(error) => self.view.push(ViewItem::Error(format!(
                                                "failed to save automatic compaction: {error:#}"
                                            ))),
                                        }
                                    }
                                }
                                Err(error) => self.view.push(ViewItem::Error(format!(
                                    "failed to reload session for automatic compaction: {error:#}"
                                ))),
                            }
                        }
                        if let Some(writer) = recovery_writer.as_mut() {
                            let finish = if self.unsaved_messages {
                                writer.preserve()
                            } else {
                                writer.clear()
                            };
                            if let Err(error) = finish {
                                self.view.push(ViewItem::Error(format!(
                                    "failed to finish recovery checkpoint writer: {error:#}"
                                )));
                            }
                        } else if !self.unsaved_messages {
                            if let Err(error) = self.session.clear_recovery_checkpoint() {
                                self.view.push(ViewItem::Error(format!(
                                    "failed to clear recovery checkpoint: {error:#}"
                                )));
                            }
                        }
                        if turn.tool_calls > 0 {
                            self.view.push(ViewItem::Info(format!(
                                "Done in {:.1}s · {} tool call(s) · {} model calls",
                                turn.duration.as_secs_f64(),
                                turn.tool_calls,
                                turn.rounds
                            )));
                        }
                    }
                    Err(error) => {
                        if let Some(writer) = recovery_writer.as_mut() {
                            if let Err(persist_error) = writer.preserve() {
                                self.view.push(ViewItem::Error(format!(
                                    "failed to preserve recovery checkpoint: {persist_error:#}"
                                )));
                            }
                        }
                        self.view.push(ViewItem::Error(error));
                    }
                }
                self.refresh_continue_available(false);
                if submit_deferred {
                    self.submit();
                }
            }
            UiEvent::CompactionFinished { result, duration } => {
                self.running = false;
                self.turn_started = None;
                self.elapsed = duration;
                match result {
                    Ok(compaction) => {
                        let record = Record::Compaction {
                            summary: compaction.summary.clone(),
                            compacted_messages: None,
                        };
                        if let Err(error) = self.session.append(&record) {
                            self.view.push(ViewItem::Error(format!(
                                "failed to save compaction summary: {error:#}"
                            )));
                            return;
                        }
                        self.messages =
                            vec![crate::compaction::summary_message(&compaction.summary)];
                        self.reset_recorder();
                        self.unsaved_messages = false;
                        self.tokens.add(compaction.usage);
                        let usage_error = self
                            .session
                            .append(&Record::Usage {
                                prompt_tokens: compaction.usage.prompt_tokens,
                                completion_tokens: compaction.usage.completion_tokens,
                            })
                            .err();
                        self.view.clear();
                        self.view.push(ViewItem::Info(
                            "Compressed earlier conversation into this summary:".to_string(),
                        ));
                        self.view.push(ViewItem::Assistant {
                            text: compaction.summary,
                            reasoning: String::new(),
                            streaming: false,
                        });
                        self.view.push(ViewItem::Info(format!(
                            "Compacted in {:.1}s.",
                            duration.as_secs_f64()
                        )));
                        if let Some(error) = usage_error {
                            self.view.push(ViewItem::Error(format!(
                                "failed to save compaction usage: {error:#}"
                            )));
                        }
                    }
                    Err(error) => self.view.push(ViewItem::Error(error)),
                }
            }
        }
    }

    fn append_assistant_text(&mut self, delta: &str) {
        if let Some(ViewItem::Assistant {
            text, streaming, ..
        }) = self.view.last_mut()
        {
            if *streaming {
                text.push_str(delta);
                return;
            }
        }
        self.view.push(ViewItem::Assistant {
            text: delta.to_string(),
            reasoning: String::new(),
            streaming: true,
        });
    }

    fn append_reasoning(&mut self, delta: &str) {
        if let Some(ViewItem::Assistant {
            reasoning,
            streaming,
            ..
        }) = self.view.last_mut()
        {
            if *streaming {
                reasoning.push_str(delta);
                return;
            }
        }
        self.view.push(ViewItem::Assistant {
            text: String::new(),
            reasoning: delta.to_string(),
            streaming: true,
        });
    }

    // ----- input handling -------------------------------------------------

    fn injection_editor_locked(&self) -> bool {
        self.pending_ask.is_none() && (self.pending_injection.is_some() || self.deferred_submission)
    }

    fn queue_injection(&mut self) {
        let raw = self.input.trim().to_string();
        if raw.is_empty() || self.pending_injection.is_some() || self.deferred_submission {
            return;
        }
        let Some(queue) = self.injection_queue.clone() else {
            return;
        };
        self.next_injection_id = self.next_injection_id.wrapping_add(1);
        let id = format!(
            "{}-{}-{}",
            self.session.id(),
            self.turn_id,
            self.next_injection_id
        );
        let content = self.expand_mentions(&raw);
        match queue.enqueue(id.clone(), content) {
            Ok(()) => {
                self.history.push(raw.clone());
                self.history_index = None;
                self.pending_injection = Some(PendingInjection { id, raw });
            }
            Err(EnqueueError::Closed(_)) => {
                self.deferred_submission = true;
            }
            Err(EnqueueError::Full(_)) => self.view.push(ViewItem::Error(
                "Another injected message is already pending.".to_string(),
            )),
        }
    }

    fn withdraw_pending_injection(&mut self) -> bool {
        if self.deferred_submission {
            self.deferred_submission = false;
            return true;
        }
        let Some(pending) = self.pending_injection.as_ref() else {
            return false;
        };
        let cancelled = self
            .injection_queue
            .as_ref()
            .and_then(|queue| queue.cancel(&pending.id));
        if cancelled.is_some() {
            self.pending_injection = None;
            return true;
        }
        false
    }

    pub fn handle_key(&mut self, key: KeyEvent, screen: Rect) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => {
                if self.running {
                    self.cancel();
                } else {
                    self.quit = true;
                }
                return;
            }
            KeyCode::Char('d') if ctrl => {
                if self.input.is_empty() && !self.running {
                    self.quit = true;
                }
                return;
            }
            _ => {}
        }

        if let Some(modal) = self.modal.take() {
            self.handle_modal_key(modal, key, screen);
            return;
        }

        match key.code {
            KeyCode::Esc => {
                if self.pending_ask.is_some() {
                    self.answer_question(String::new());
                } else if self.withdraw_pending_injection() {
                    // The first Escape only withdraws queued input.
                } else if self.running {
                    self.cancel();
                } else {
                    self.input.clear();
                    self.cursor = 0;
                }
            }
            KeyCode::Enter => {
                if self.pending_ask.is_some() {
                    let answer = std::mem::take(&mut self.input);
                    self.cursor = 0;
                    self.answer_question(answer);
                } else if self.running {
                    let trimmed = self.input.trim_start();
                    if !self.injection_editor_locked()
                        && (trimmed.starts_with('/') || trimmed.starts_with('!'))
                    {
                        self.submit();
                    } else if !self.injection_editor_locked()
                        && self.injection_queue.is_none()
                        && !trimmed.is_empty()
                    {
                        self.view.push(ViewItem::Info(
                            "A turn is already running; wait for it or press Esc to cancel."
                                .to_string(),
                        ));
                    } else {
                        self.queue_injection();
                    }
                } else if self.continue_available && self.input.trim().is_empty() {
                    self.continue_turn();
                } else {
                    self.submit();
                }
            }
            KeyCode::Char(character) if !self.injection_editor_locked() => {
                self.insert_char(character);
            }
            KeyCode::Backspace if !self.injection_editor_locked() => self.backspace(),
            KeyCode::Delete if !self.injection_editor_locked() => self.delete(),
            KeyCode::Left if !self.injection_editor_locked() => {
                self.cursor = self.cursor.saturating_sub(1)
            }
            KeyCode::Right if !self.injection_editor_locked() => {
                self.cursor = (self.cursor + 1).min(self.char_len());
            }
            KeyCode::Home if !self.injection_editor_locked() => self.cursor = 0,
            KeyCode::End if !self.injection_editor_locked() => self.cursor = self.char_len(),
            KeyCode::Up if !self.injection_editor_locked() => self.history_previous(),
            KeyCode::Down if !self.injection_editor_locked() => self.history_next(),
            KeyCode::PageUp => {
                self.scroll_back = self.scroll_back.saturating_add(10);
                self.follow = false;
            }
            KeyCode::PageDown => {
                self.scroll_back = self.scroll_back.saturating_sub(10);
                if self.scroll_back == 0 {
                    self.follow = true;
                }
            }
            KeyCode::Tab if !self.injection_editor_locked() => self.complete_command(),
            _ => {}
        }
    }

    fn handle_modal_key(&mut self, modal: Modal, key: KeyEvent, screen: Rect) {
        let allow_always = matches!(
            &modal,
            Modal::Approval {
                allow_always: true,
                ..
            }
        );
        self.modal = Some(modal);
        match key.code {
            KeyCode::Esc => self.decide_approval(ApprovalDecision::Deny, screen),
            // Enter is deliberately not bound: an accidental Enter must not
            // approve a tool call the user has not read.
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                self.decide_approval(ApprovalDecision::AllowOnce, screen)
            }
            KeyCode::Char('n') | KeyCode::Char('N') => {
                self.decide_approval(ApprovalDecision::Deny, screen)
            }
            KeyCode::Char('a') | KeyCode::Char('A') if allow_always => {
                self.decide_approval(ApprovalDecision::AllowAlways, screen)
            }
            _ => {}
        }
    }

    /// Approve or deny the pending request. Allow decisions are refused while
    /// the dialog has to clip the command, so the user cannot approve text they
    /// cannot read.
    fn decide_approval(&mut self, decision: ApprovalDecision, screen: Rect) {
        let Some(Modal::Approval {
            request,
            allow_always,
            ..
        }) = &self.modal
        else {
            self.modal = None;
            return;
        };
        if decision == ApprovalDecision::AllowAlways && !*allow_always {
            return;
        }
        if matches!(
            decision,
            ApprovalDecision::AllowOnce | ApprovalDecision::AllowAlways
        ) && approval_layout(screen, &request.tool, &request.summary, *allow_always).clipped
        {
            return;
        }
        if let Some(Modal::Approval { reply, .. }) = self.modal.take() {
            let _ = reply.send(decision);
        }
    }

    fn insert_char(&mut self, character: char) {
        let index = self.byte_index(self.cursor);
        self.input.insert(index, character);
        self.cursor += 1;
        self.history_index = None;
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let end = self.byte_index(self.cursor);
        let start = self.byte_index(self.cursor - 1);
        self.input.replace_range(start..end, "");
        self.cursor -= 1;
        self.history_index = None;
    }

    fn delete(&mut self) {
        if self.cursor >= self.char_len() {
            return;
        }
        let start = self.byte_index(self.cursor);
        let end = self.byte_index(self.cursor + 1);
        self.input.replace_range(start..end, "");
    }

    fn char_len(&self) -> usize {
        self.input.chars().count()
    }

    fn byte_index(&self, character_index: usize) -> usize {
        self.input
            .char_indices()
            .nth(character_index)
            .map(|(byte, _)| byte)
            .unwrap_or(self.input.len())
    }

    fn history_previous(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let index = match self.history_index {
            Some(index) => index.saturating_sub(1),
            None => self.history.len() - 1,
        };
        self.history_index = Some(index);
        self.input = self.history[index].clone();
        self.cursor = self.char_len();
    }

    fn history_next(&mut self) {
        if let Some(index) = self.history_index {
            if index + 1 < self.history.len() {
                self.history_index = Some(index + 1);
                self.input = self.history[index + 1].clone();
            } else {
                self.history_index = None;
                self.input.clear();
            }
            self.cursor = self.char_len();
        }
    }

    fn complete_command(&mut self) {
        if !self.input.starts_with('/') || self.input.contains(' ') {
            return;
        }
        let commands = [
            "/help",
            "/new",
            "/clear",
            "/quit",
            "/models",
            "/model",
            "/approval",
            "/sessions",
            "/resume",
            "/continue-turn",
            "/cwd",
            "/compact",
        ];
        if let Some(command) = commands
            .iter()
            .find(|command| command.starts_with(&self.input))
        {
            self.input = command.to_string();
            self.cursor = self.char_len();
        }
    }

    // ----- turn lifecycle -------------------------------------------------

    fn refresh_continue_available(&mut self, announce: bool) {
        let (available, failed) = inspect_continue_available(&self.session, &mut self.view);
        self.continue_available = available;
        self.recovery_probe_failed = failed;
        if announce && available {
            self.view
                .push(continue_available_notice(self.approval_mode));
        }
    }

    fn reload_session_before_turn(&mut self) -> bool {
        if !self.requires_session_reload {
            return true;
        }
        match self.session.load_messages() {
            Ok(messages) => {
                self.messages = messages;
                self.reset_recorder();
                self.requires_session_reload = false;
                self.refresh_continue_available(false);
                true
            }
            Err(error) => {
                self.view.push(ViewItem::Error(format!(
                    "cannot start a new turn until the session reloads: {error:#}"
                )));
                false
            }
        }
    }

    pub fn submit(&mut self) {
        let submitted = std::mem::take(&mut self.input);
        self.cursor = 0;
        self.history_index = None;
        let raw = submitted.trim().to_string();
        if raw.is_empty() {
            return;
        }
        self.history.push(raw.clone());
        if raw.starts_with('/') {
            self.run_command(&raw);
            return;
        }
        if let Some(command) = raw.strip_prefix('!') {
            self.run_shell(command.trim().to_string());
            return;
        }
        if self.running {
            self.view.push(ViewItem::Info(
                "A turn is already running; wait for it or press Esc to cancel.".to_string(),
            ));
            return;
        }
        if !self.reload_session_before_turn() {
            self.history.pop();
            self.input = submitted;
            self.cursor = self.char_len();
            return;
        }
        if self.recovery_probe_failed {
            self.history.pop();
            self.input = submitted;
            self.cursor = self.char_len();
            self.view.push(ViewItem::Error(
                "Cannot start a new turn while recovery inspection is failing. Use /new or /resume instead."
                    .to_string(),
            ));
            return;
        }
        let message_text = self.expand_mentions(&raw);
        let mut recovery_writer = if self.unsaved_messages {
            None
        } else {
            let next_turn = match self.session.load_phase_four_breaker() {
                Ok((turn, _)) => turn.saturating_add(1),
                Err(error) => {
                    self.history.pop();
                    self.input = submitted;
                    self.cursor = self.char_len();
                    self.view.push(ViewItem::Error(format!(
                        "failed to prepare recovery checkpoint writer: {error:#}"
                    )));
                    return;
                }
            };
            match RecoveryCheckpointWriter::new(self.session.clone(), next_turn) {
                Ok(writer) => Some(writer),
                Err(error) => {
                    self.history.pop();
                    self.input = submitted;
                    self.cursor = self.char_len();
                    self.view.push(ViewItem::Error(format!(
                        "failed to start recovery checkpoint writer: {error:#}"
                    )));
                    return;
                }
            }
        };
        self.continue_available = false;
        let user = Message::user(message_text.clone());
        self.messages.push(user);
        let phase_four = match self
            .session
            .append_user_and_load_phase_four_breaker(&message_text)
        {
            Ok(restored) => restored,
            Err(error) => {
                recovery_writer = None;
                self.unsaved_messages = true;
                self.view.push(ViewItem::Error(format!(
                    "failed to save session: {error:#}"
                )));
                (0, Default::default())
            }
        };
        self.view.push(ViewItem::User(raw));
        self.view.push(ViewItem::Assistant {
            text: String::new(),
            reasoning: String::new(),
            streaming: true,
        });
        self.start_agent(phase_four, self.approver.clone(), recovery_writer);
    }

    fn start_agent(
        &mut self,
        phase_four: (u64, PhaseFourBreakerState),
        approver: Arc<dyn Approver>,
        recovery_writer: Option<RecoveryCheckpointWriter>,
    ) {
        self.running = true;
        self.turn_id = self.turn_id.wrapping_add(1);
        let turn_id = self.turn_id;
        self.turn_started = Some(Instant::now());
        self.tokens = Usage::default();
        let tx = self.events_tx.clone();
        let profile = self.profile.clone();
        let kind = self.agent_kind;
        let cwd = self.cwd.clone();
        let max_rounds = self.settings.max_tool_rounds;
        let phase_four_side_call_token_budget = self.settings.phase4_side_call_token_budget;
        let asker = self.asker.clone();
        let tools = self.tools.clone();
        let mut messages = self.messages.clone();
        let (phase_four_turn, phase_four_state) = phase_four;
        let recovery_enabled = recovery_writer.is_some();
        let persistence_gate = Arc::new(AtomicBool::new(recovery_enabled));
        let recovery_handle = recovery_writer
            .as_ref()
            .map(RecoveryCheckpointWriter::handle);
        self.persistence_gate = Some(persistence_gate.clone());
        self.recovery_writer = recovery_writer;
        let (injection_queue, injection_source) = injection::channel();
        self.injection_queue = Some(injection_queue);
        let sink: Arc<dyn EventSink> = Arc::new(TuiSink {
            tx: tx.clone(),
            turn_id,
            session: self.session.clone(),
            recorder: self.recorder.clone(),
            persistence_gate,
            recovery_enabled,
            recovery_writer: recovery_handle,
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
            phase_four_turn,
            phase_four_breaker: PhaseFourBreaker::new(
                phase_four_side_call_token_budget,
                phase_four_state,
            ),
        });
        let handle = self.runtime.clone();
        let task = handle.spawn(async move {
            let completed = async {
                let client = Arc::new(
                    tokio::task::spawn_blocking(move || ModelClient::from_profile(&profile))
                        .await
                        .map_err(|error| {
                            anyhow::anyhow!("failed to resolve the model configuration: {error}")
                        })??,
                );
                let options = AgentOptions {
                    kind,
                    cwd,
                    max_rounds,
                    tool_timeout: Duration::from_secs(120),
                };
                let agent = Agent::new(client, tools, approver, asker, options);
                let turn = agent
                    .run_with_injections(&mut messages, sink.as_ref(), &injection_source)
                    .await?;
                let force_compression = agent.should_force_compression(&turn);
                Ok::<_, anyhow::Error>((turn, force_compression))
            }
            .await;
            let (result, force_compression) = match completed {
                Ok((turn, force)) => (Ok(turn), force),
                Err(error) => (Err(format!("{error:#}")), false),
            };
            let _ = tx.send(UiEvent::Turn {
                id: turn_id,
                event: Box::new(UiEvent::TurnFinished(result, messages, force_compression)),
            });
        });
        self.cancel_handle = Some(task);
    }

    pub fn cancel(&mut self) {
        if !self.running {
            return;
        }
        if self.active_direct_shell.take().is_some() {
            self.pending_direct_shell = None;
            self.running = false;
            self.turn_started = None;
            for item in &mut self.view {
                if let ViewItem::Tool {
                    name,
                    status,
                    ok,
                    output,
                    ..
                } = item
                {
                    if name == "bash" && *status == ToolStatus::Running {
                        *status = ToolStatus::Done;
                        *ok = Some(false);
                        *output = "Cancelled.".to_string();
                    }
                }
            }
            return;
        }
        if self.pending_ask.is_some() {
            if let Some(pending) = self.pending_injection.as_ref() {
                self.input = pending.raw.clone();
                self.cursor = self.char_len();
            }
        }
        self.pending_injection = None;
        self.deferred_submission = false;
        let preserve = self
            .recovery_writer
            .as_ref()
            .map(RecoveryCheckpointWriter::begin_preserve)
            .transpose();
        if let Some(queue) = self.injection_queue.take() {
            queue.close();
        }
        if let Some(gate) = self.persistence_gate.take() {
            gate.store(false, Ordering::SeqCst);
        }
        if let Some(handle) = self.cancel_handle.take() {
            handle.abort();
        }
        if let Some(mut writer) = self.recovery_writer.take() {
            let result = match preserve {
                Ok(Some(receiver)) => writer.wait_preserved(receiver),
                Ok(None) => Ok(()),
                Err(error) => Err(error),
            };
            if let Err(error) = result {
                self.view.push(ViewItem::Error(format!(
                    "failed to preserve recovery checkpoint: {error:#}"
                )));
            }
        }
        if !self.unsaved_messages {
            self.requires_session_reload = true;
            match self.session.load_messages() {
                Ok(messages) => {
                    self.messages = messages;
                    self.reset_recorder();
                    self.requires_session_reload = false;
                }
                Err(error) => self.view.push(ViewItem::Error(format!(
                    "failed to reload session after cancellation: {error:#}"
                ))),
            }
        }
        self.refresh_continue_available(false);
        if let Some(Modal::Approval { reply, .. }) = self.modal.take() {
            let _ = reply.send(ApprovalDecision::Deny);
        }
        if let Some(reply) = self.pending_ask.take() {
            let _ = reply.send(String::new());
        }
        self.active_direct_shell = None;
        self.pending_direct_shell = None;
        self.running = false;
        self.turn_id = self.turn_id.wrapping_add(1);
        self.turn_started = None;
        for item in &mut self.view {
            match item {
                ViewItem::Assistant { streaming, .. } => *streaming = false,
                ViewItem::Tool {
                    status, ok, output, ..
                } if *status == ToolStatus::Running => {
                    *status = ToolStatus::Done;
                    *ok = Some(false);
                    *output = "Cancelled.".to_string();
                }
                _ => {}
            }
        }
        self.view.push(ViewItem::Info("Cancelled.".to_string()));
    }

    fn answer_question(&mut self, answer: String) {
        if let Some(reply) = self.pending_ask.take() {
            let _ = reply.send(answer);
        }
        if let Some(pending) = self.pending_injection.as_ref() {
            self.input = pending.raw.clone();
            self.cursor = self.char_len();
        } else {
            self.input.clear();
            self.cursor = 0;
        }
    }

    pub(crate) fn compaction_messages(&self) -> Result<Vec<Message>> {
        Ok(self.store.load(self.session.id())?.messages())
    }

    fn start_compaction(&mut self) {
        if self.unsaved_messages {
            self.view.push(ViewItem::Error(
                "Cannot compact: some conversation messages were not saved.".to_string(),
            ));
            return;
        }
        let messages = match self.compaction_messages() {
            Ok(messages) => messages,
            Err(error) => {
                self.view.push(ViewItem::Error(format!(
                    "cannot load conversation for compaction: {error:#}"
                )));
                return;
            }
        };
        if !messages
            .iter()
            .any(|message| !matches!(message, Message::System { .. }))
        {
            self.view.push(ViewItem::Error(
                "There is no conversation to compact.".to_string(),
            ));
            return;
        }
        self.running = true;
        self.turn_id = self.turn_id.wrapping_add(1);
        let turn_id = self.turn_id;
        self.turn_started = Some(Instant::now());
        self.tokens = Usage::default();
        self.view
            .push(ViewItem::Info("Compacting conversation...".to_string()));

        let tx = self.events_tx.clone();
        let profile = self.profile.clone();
        let handle = self.runtime.clone();
        let task = handle.spawn(async move {
            let started = Instant::now();
            let result = async {
                let client =
                    tokio::task::spawn_blocking(move || ModelClient::from_profile(&profile))
                        .await
                        .map_err(|error| {
                            anyhow::anyhow!("failed to resolve the model configuration: {error}")
                        })??;
                crate::compaction::compact(&client, &messages).await
            }
            .await;
            let _ = tx.send(UiEvent::Turn {
                id: turn_id,
                event: Box::new(UiEvent::CompactionFinished {
                    result: result.map_err(|error| format!("compaction failed: {error:#}")),
                    duration: started.elapsed(),
                }),
            });
        });
        self.cancel_handle = Some(task);
    }

    fn run_shell(&mut self, command: String) {
        if self.running {
            self.view.push(ViewItem::Info(
                "A turn is already running; press Esc to cancel.".to_string(),
            ));
            return;
        }
        if command.is_empty() {
            return;
        }
        self.running = true;
        self.turn_id = self.turn_id.wrapping_add(1);
        let id = self.turn_id;
        let started = Instant::now();
        self.turn_started = Some(started);
        self.view.push(ViewItem::Tool {
            name: "bash".to_string(),
            summary: command.clone(),
            status: ToolStatus::Running,
            ok: None,
            duration: None,
            output: String::new(),
        });
        self.active_direct_shell = Some(id);
        self.pending_direct_shell = Some(DirectShellRequest {
            id,
            command,
            cwd: self.cwd.clone(),
            started,
        });
    }

    /// Hand the one pending direct command to the terminal event loop.
    pub(crate) fn take_direct_shell(&mut self) -> Option<DirectShellRequest> {
        self.pending_direct_shell.take()
    }

    /// Complete the matching direct command without touching conversation state.
    pub(crate) fn complete_direct_shell(
        &mut self,
        id: u64,
        command: &str,
        duration: Duration,
        outcome: super::direct_shell::DirectShellOutcome,
    ) -> bool {
        if self.active_direct_shell != Some(id) {
            return false;
        }
        self.active_direct_shell = None;
        self.pending_direct_shell = None;
        self.running = false;
        self.turn_started = None;
        self.elapsed = duration;
        let ok = outcome.succeeded();
        let output = outcome.summary();
        if let Some(ViewItem::Tool {
            status,
            ok: tool_ok,
            duration: tool_duration,
            output: tool_output,
            ..
        }) = self.view.iter_mut().rev().find(|item| {
            matches!(
                item,
                ViewItem::Tool { name, summary, status: ToolStatus::Running, .. }
                    if name == "bash" && summary == command
            )
        }) {
            *status = ToolStatus::Done;
            *tool_ok = Some(ok);
            *tool_duration = Some(duration);
            *tool_output = output;
        }
        true
    }

    fn expand_mentions(&self, input: &str) -> String {
        let mut attachments = String::new();
        let mut seen = std::collections::HashSet::new();
        for token in input.split_whitespace() {
            let Some(path) = token.strip_prefix('@') else {
                continue;
            };
            if path.is_empty() || !seen.insert(path.to_string()) {
                continue;
            }
            let full = self.cwd.join(path);
            let Ok(content) = std::fs::read_to_string(&full) else {
                continue;
            };
            let content = if content.chars().count() > 20_000 {
                let head: String = content.chars().take(20_000).collect();
                format!("{head}\n[... file truncated at 20000 characters ...]")
            } else {
                content
            };
            attachments.push_str(&format!("\n\nAttached file `{path}`:\n```\n{content}\n```"));
        }
        if attachments.is_empty() {
            input.to_string()
        } else {
            format!("{input}{attachments}")
        }
    }

    // ----- slash commands -------------------------------------------------

    fn run_command(&mut self, raw: &str) {
        let mut parts = raw.splitn(2, char::is_whitespace);
        let command = parts.next().unwrap_or_default();
        let argument = parts.next().unwrap_or_default().trim();
        if self.running
            && matches!(
                command,
                "/new" | "/clear" | "/resume" | "/continue-turn" | "/compact"
            )
        {
            self.view.push(ViewItem::Info(
                "Cancel the running turn (Esc) before changing the session.".to_string(),
            ));
            return;
        }
        match command {
            "/help" => self.modal = Some(Modal::Help),
            "/quit" | "/exit" => self.quit = true,
            "/new" | "/clear" => self.new_session(),
            "/models" => self.list_models(),
            "/model" => {
                if argument.is_empty() {
                    self.view.push(ViewItem::Info(format!(
                        "Current model: {} ({}). Use /model <name> to switch.",
                        self.profile.name, self.profile.model
                    )));
                } else if let Some(profile) = self
                    .settings
                    .models
                    .iter()
                    .find(|profile| profile.name == argument)
                    .cloned()
                {
                    self.profile = profile;
                    self.settings.current_model = Some(argument.to_string());
                    if let Err(error) = self.settings.save() {
                        self.view.push(ViewItem::Error(format!(
                            "failed to save settings: {error:#}"
                        )));
                    }
                    self.view.push(ViewItem::Info(format!(
                        "Switched to model profile {argument} ({}).",
                        self.profile.model
                    )));
                } else {
                    self.view.push(ViewItem::Error(format!(
                        "Unknown model profile {argument:?}; use /models to list them."
                    )));
                }
            }
            "/approval" => {
                if argument.is_empty() {
                    self.view.push(ViewItem::Info(format!(
                        "Approval mode: {} (ask, auto or deny).",
                        self.approval_mode
                    )));
                } else if let Some(mode) = ApprovalMode::parse(argument) {
                    self.approval_mode = mode;
                    // An explicit /approval change is persisted; a CLI override
                    // alone is not.
                    self.settings.approval_mode = mode;
                    self.approver = match mode {
                        ApprovalMode::Auto => Arc::new(AllowAll),
                        ApprovalMode::Deny => Arc::new(DenyAll),
                        ApprovalMode::Ask => Arc::new(TuiApprover::new(self.events_tx.clone())),
                    };
                    self.view
                        .push(ViewItem::Info(format!("Approval mode set to {mode}.")));
                } else {
                    self.view.push(ViewItem::Error(
                        "Usage: /approval ask|auto|deny".to_string(),
                    ));
                }
            }
            "/sessions" => match self.store.list() {
                Ok(sessions) if !sessions.is_empty() => {
                    self.view
                        .push(ViewItem::Info("Recent sessions:".to_string()));
                    for session in sessions.iter().take(10) {
                        self.view.push(ViewItem::Info(format!(
                            "{} · {} · {}",
                            session.id, session.model, session.preview
                        )));
                    }
                    self.view.push(ViewItem::Info(
                        "Use /resume <id> to continue a session.".to_string(),
                    ));
                }
                Ok(_) => self
                    .view
                    .push(ViewItem::Info("No sessions yet.".to_string())),
                Err(error) => self
                    .view
                    .push(ViewItem::Error(format!("cannot list sessions: {error:#}"))),
            },
            "/resume" => {
                if argument.is_empty() {
                    self.view
                        .push(ViewItem::Error("Usage: /resume <session-id>".to_string()));
                } else {
                    self.resume_session(argument);
                }
            }
            "/continue-turn" => {
                if argument.is_empty() {
                    self.continue_turn();
                } else {
                    self.view
                        .push(ViewItem::Error("Usage: /continue-turn".to_string()));
                }
            }
            "/cwd" => {
                if argument.is_empty() {
                    self.view.push(ViewItem::Info(format!(
                        "Working directory: {}",
                        self.cwd.display()
                    )));
                } else {
                    let candidate = self.cwd.join(argument);
                    match candidate.canonicalize() {
                        Ok(path) if path.is_dir() => {
                            self.cwd = path.clone();
                            self.view.push(ViewItem::Info(format!(
                                "Working directory is now {}.",
                                path.display()
                            )));
                        }
                        _ => self
                            .view
                            .push(ViewItem::Error(format!("{argument:?} is not a directory"))),
                    }
                }
            }
            "/compact" => self.start_compaction(),
            _ => self.view.push(ViewItem::Error(format!(
                "Unknown command {command:?}; use /help for the command list."
            ))),
        }
    }

    fn continue_turn(&mut self) {
        if self.unsaved_messages {
            self.view.push(ViewItem::Error(
                "Cannot continue: some conversation messages were not saved.".to_string(),
            ));
            return;
        }
        let (messages, turn, breaker) = match self.session.load_resumable_turn() {
            Ok(restored) => restored,
            Err(error) => {
                self.view.push(ViewItem::Error(format!(
                    "cannot continue current turn: {error:#}"
                )));
                return;
            }
        };
        let recovery_writer = match RecoveryCheckpointWriter::new(self.session.clone(), turn) {
            Ok(writer) => writer,
            Err(error) => {
                self.view.push(ViewItem::Error(format!(
                    "cannot start recovery checkpoint writer: {error:#}"
                )));
                return;
            }
        };
        let (approver, safety_message): (Arc<dyn Approver>, &str) = match self.approval_mode {
            ApprovalMode::Deny => (
                Arc::new(DenyAll),
                "Continuing the current user turn with mutating tools denied.",
            ),
            ApprovalMode::Ask | ApprovalMode::Auto => (
                Arc::new(TuiApprover::recovery(self.events_tx.clone())),
                "Continuing the current user turn; every mutating tool requires confirmation.",
            ),
        };

        self.continue_available = false;
        self.messages = messages;
        self.reset_recorder();
        self.requires_session_reload = false;
        self.view.push(ViewItem::Info(safety_message.to_string()));
        self.view.push(ViewItem::Assistant {
            text: String::new(),
            reasoning: String::new(),
            streaming: true,
        });
        self.start_agent((turn, breaker), approver, Some(recovery_writer));
    }

    fn resume_session(&mut self, id: &str) {
        let replacement_lease = if id == self.session.id() {
            None
        } else {
            match self.store.try_acquire_run_lease(id) {
                Ok(lease) => Some(lease),
                Err(error) => {
                    self.view
                        .push(ViewItem::Error(format!("cannot resume: {error:#}")));
                    return;
                }
            }
        };
        let loaded = match self.store.load(id) {
            Ok(loaded) => loaded,
            Err(error) => {
                self.view
                    .push(ViewItem::Error(format!("cannot resume: {error:#}")));
                return;
            }
        };
        let session = match self.store.reopen(id) {
            Ok(session) => session,
            Err(error) => {
                self.view
                    .push(ViewItem::Error(format!("cannot resume session: {error:#}")));
                return;
            }
        };

        self.messages = loaded.messages();
        self.view = view_from_records(&loaded.records);
        self.cwd = PathBuf::from(&loaded.meta.cwd);
        self.session = session;
        if let Some(lease) = replacement_lease {
            self.session_run_lease = lease;
        }
        self.reset_recorder();
        self.unsaved_messages = false;
        self.requires_session_reload = false;
        self.view.push(ViewItem::Info(format!(
            "Resumed session {} ({}).",
            loaded.meta.id, loaded.meta.cwd
        )));
        self.refresh_continue_available(true);
    }

    fn new_session(&mut self) {
        let session = match self.store.create(
            &self.cwd.display().to_string(),
            &self.profile.model,
            self.agent_kind.as_str(),
        ) {
            Ok(session) => session,
            Err(error) => {
                self.view.push(ViewItem::Error(format!(
                    "failed to create session: {error:#}"
                )));
                return;
            }
        };
        let lease = match self.store.try_acquire_run_lease(session.id()) {
            Ok(lease) => lease,
            Err(error) => {
                self.view.push(ViewItem::Error(format!(
                    "failed to lease new session: {error:#}"
                )));
                return;
            }
        };

        self.session = session;
        self.session_run_lease = lease;
        self.messages.clear();
        self.view = welcome();
        self.reset_recorder();
        self.unsaved_messages = false;
        self.requires_session_reload = false;
        self.continue_available = false;
        self.recovery_probe_failed = false;
        self.view
            .push(ViewItem::Info("Started a new session.".to_string()));
    }

    fn list_models(&mut self) {
        self.view
            .push(ViewItem::Info("Model profiles:".to_string()));
        for profile in &self.settings.models {
            let marker = if profile.name == self.profile.name {
                "* "
            } else {
                "  "
            };
            self.view.push(ViewItem::Info(format!(
                "{marker}{} · {} · {}",
                profile.name,
                profile.model,
                profile.resolved_base_url()
            )));
        }
    }

    pub fn status_line(&self) -> String {
        let mut status = format!(
            " {} · {} · {} · {} ",
            self.agent_kind.display_name(),
            self.profile.name,
            self.approval_mode,
            self.cwd.display()
        );
        if self.tokens.total_tokens > 0 {
            status.push_str(&format!("· {} tok ", self.tokens.total_tokens));
        }
        if self.running {
            status.push_str("· working ");
            if self.pending_injection.is_some() || self.deferred_submission {
                status.push_str("· queued ");
            }
        } else if self.elapsed > Duration::ZERO {
            status.push_str(&format!("· {:.1}s ", self.elapsed.as_secs_f64()));
        }
        status.push_str(&format!("· {} ", self.session.id()));
        status
    }
}

/// Geometry shared by the approval dialog renderer and the allow guard.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ApprovalLayout {
    /// Text width inside the popup borders.
    pub interior: usize,
    /// True when the command and hints cannot fit in the popup.
    pub clipped: bool,
}

/// Return the keyboard hint rendered for this approval mode.
pub(crate) fn approval_hint(allow_always: bool) -> &'static str {
    if allow_always {
        "  y allow once · a always allow · n deny · esc deny"
    } else {
        "  y allow once · n deny · esc deny"
    }
}

/// Compute the approval popup geometry for the current terminal size.
///
/// `ui::draw_approval` renders with this layout, and `decide_approval` refuses
/// allow decisions when `clipped` is set, so approval can never be granted for
/// text the user cannot see.
pub(crate) fn approval_layout(
    screen: Rect,
    tool: &str,
    summary: &str,
    allow_always: bool,
) -> ApprovalLayout {
    // Keep in sync with `centered(area, 70, height)` in the renderer.
    let popup_width = (screen.width as u32 * 70 / 100) as u16;
    let interior = popup_width.saturating_sub(2) as usize;
    if interior < 10 {
        return ApprovalLayout {
            interior: interior.max(1),
            clipped: true,
        };
    }
    let summary_lines = wrapped_cell_lines(&format!("  {tool}: {summary}"), interior);
    let hint_lines = wrapped_cell_lines(approval_hint(allow_always), interior);
    let needed = summary_lines + 1 + hint_lines + 2;
    ApprovalLayout {
        interior,
        clipped: needed > screen.height as usize,
    }
}

/// Count the display rows `wrap_line` would produce, counting wide characters
/// as two cells.
fn wrapped_cell_lines(text: &str, width: usize) -> usize {
    if width == 0 {
        return 1;
    }
    let mut total = 0;
    for line in text.split('\n') {
        let mut rows = 0usize;
        let mut current = 0usize;
        for character in line.chars() {
            let character_width = unicode_width::UnicodeWidthChar::width(character)
                .unwrap_or(0)
                .max(1);
            if current + character_width > width && current > 0 {
                rows += 1;
                current = 0;
            }
            current += character_width;
            if current >= width {
                rows += 1;
                current = 0;
            }
        }
        if current > 0 || rows == 0 {
            rows += 1;
        }
        total += rows;
    }
    total.max(1)
}

fn inspect_continue_available(session: &Session, view: &mut Vec<ViewItem>) -> (bool, bool) {
    match session.resumable_turn_available() {
        Ok(available) => (available, false),
        Err(error) => {
            view.push(ViewItem::Error(format!(
                "cannot inspect interrupted turn: {error:#}"
            )));
            (false, true)
        }
    }
}

fn continue_available_notice(approval: ApprovalMode) -> ViewItem {
    let message = match approval {
        ApprovalMode::Deny => "Interrupted turn available. Continuing keeps mutating tools denied.",
        ApprovalMode::Ask | ApprovalMode::Auto => {
            "Interrupted turn available. Every mutating tool requires confirmation."
        }
    };
    ViewItem::Info(message.to_string())
}

fn welcome() -> Vec<ViewItem> {
    let mut items = vec![ViewItem::Info(format!(
        "Welcome to {APP_NAME} {}. A Rust port of iCode.",
        crate::APP_VERSION
    ))];
    items.push(ViewItem::Info(
        "Ask a coding question, or try: explain this project; add tests for src/main.rs"
            .to_string(),
    ));
    items.push(ViewItem::Info(
        "Commands: /help · !<shell> · @<file> to attach a file".to_string(),
    ));
    items.push(ViewItem::Info(String::new()));
    items
}

fn view_from_records(records: &[crate::session::Record]) -> Vec<ViewItem> {
    let mut view = Vec::new();
    for record in records {
        match record {
            crate::session::Record::User { text } => view.push(ViewItem::User(text.clone())),
            crate::session::Record::Assistant { content, .. } => {
                if let Some(content) = content {
                    view.push(ViewItem::Assistant {
                        text: content.clone(),
                        reasoning: String::new(),
                        streaming: false,
                    });
                }
            }
            crate::session::Record::Tool { name, content, .. } => view.push(ViewItem::Tool {
                name: name.clone(),
                summary: String::new(),
                status: ToolStatus::Done,
                ok: Some(!content.starts_with("Error:")),
                duration: None,
                output: preview_output(content),
            }),
            crate::session::Record::Compaction {
                summary,
                compacted_messages,
            } => {
                if compacted_messages.is_some() {
                    continue;
                }
                view.clear();
                view.push(ViewItem::Info(
                    "Compressed earlier conversation into this summary:".to_string(),
                ));
                view.push(ViewItem::Assistant {
                    text: summary.clone(),
                    reasoning: String::new(),
                    streaming: false,
                });
            }
            crate::session::Record::Meta { .. }
            | crate::session::Record::AgentMode { .. }
            | crate::session::Record::InjectedUser { .. }
            | crate::session::Record::Usage { .. }
            | crate::session::Record::PhaseFourBreaker { .. }
            | crate::session::Record::CompressionBatch { .. }
            | crate::session::Record::ContextProjection { .. } => {}
        }
    }
    view
}

fn preview_output(output: &str) -> String {
    let truncated = truncate_output(output);
    let mut lines: Vec<&str> = truncated.lines().collect();
    if lines.len() > 8 {
        lines.truncate(6);
        let mut preview = lines.join("\n");
        preview.push_str("\n… (output truncated)");
        preview
    } else {
        lines.join("\n")
    }
}

fn persist_context_projection(
    session: &Session,
    recorder: &Mutex<RecordBuffer>,
    persistence_gate: Option<&Arc<AtomicBool>>,
    unsaved_messages: bool,
    source: &[Message],
    projection: &ContextProjection,
) -> std::result::Result<Option<ArchivedContextProjection>, String> {
    if unsaved_messages {
        return Ok(None);
    }
    let mut recorder = recorder
        .lock()
        .map_err(|_| "session recorder lock is poisoned".to_string())?;
    let archived = session
        .append_archived_context_projection_if(source, projection, || {
            persistence_gate.is_some_and(|gate| gate.load(Ordering::SeqCst))
        })
        .map_err(|error| format!("{error:#}"))?;
    if archived.is_some() {
        recorder.reset();
    }
    Ok(archived)
}

struct TuiSink {
    tx: mpsc::UnboundedSender<UiEvent>,
    turn_id: u64,
    session: Session,
    recorder: Arc<Mutex<RecordBuffer>>,
    persistence_gate: Arc<AtomicBool>,
    recovery_enabled: bool,
    recovery_writer: Option<RecoveryCheckpointHandle>,
    archive_pointer: Mutex<Option<String>>,
    current_turn_manifest: Mutex<CurrentTurnArchiveManifest>,
    phase_four_turn: u64,
    phase_four_breaker: PhaseFourBreaker,
}

impl TuiSink {
    fn send(&self, event: UiEvent) {
        let _ = self.tx.send(UiEvent::Turn {
            id: self.turn_id,
            event: Box::new(event),
        });
    }
}

impl EventSink for TuiSink {
    fn turn_started(&self) {
        let pointer = self.session.archive_catalog_pointer().ok().flatten();
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

    fn persist_injected_user(&self, message: &Message) -> anyhow::Result<()> {
        if !matches!(message, Message::InjectedUser { .. }) {
            anyhow::bail!("injected transcript persistence requires an InjectedUser message");
        }
        if !self.persistence_gate.load(Ordering::SeqCst) {
            anyhow::bail!("turn was cancelled before injected transcript persistence");
        }
        let (reply, response) = std::sync::mpsc::channel();
        self.tx
            .send(UiEvent::Turn {
                id: self.turn_id,
                event: Box::new(UiEvent::PersistInjectedUser {
                    message: message.clone(),
                    reply,
                }),
            })
            .map_err(|_| {
                anyhow::anyhow!("TUI event loop closed before injected transcript persistence")
            })?;
        loop {
            match response.recv_timeout(Duration::from_millis(100)) {
                Ok(result) => return result.map_err(anyhow::Error::msg),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected)
                    if !self.persistence_gate.load(Ordering::SeqCst) =>
                {
                    anyhow::bail!("turn was cancelled during injected transcript persistence")
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    anyhow::bail!("TUI injected transcript persistence was cancelled")
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                    if !self.persistence_gate.load(Ordering::SeqCst) =>
                {
                    anyhow::bail!("turn was cancelled during injected transcript persistence")
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }

    fn persist_recovery_checkpoint(&self, messages: &[Message]) -> anyhow::Result<bool> {
        if !self.recovery_enabled {
            return Ok(false);
        }
        if !self.persistence_gate.load(Ordering::SeqCst) {
            anyhow::bail!("turn was cancelled before the recovery checkpoint");
        }
        let breaker = self.phase_four_breaker.snapshot()?;
        self.recovery_writer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("recovery checkpoint writer is unavailable"))?
            .barrier(messages, breaker)?;
        if !self.persistence_gate.load(Ordering::SeqCst) {
            anyhow::bail!("turn was cancelled during the recovery checkpoint");
        }
        Ok(true)
    }

    fn publish_recovery_checkpoint(&self, messages: Vec<Message>) -> anyhow::Result<bool> {
        if !self.recovery_enabled {
            return Ok(false);
        }
        if !self.persistence_gate.load(Ordering::SeqCst) {
            anyhow::bail!("turn was cancelled before the recovery checkpoint");
        }
        let breaker = self.phase_four_breaker.snapshot()?;
        self.recovery_writer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("recovery checkpoint writer is unavailable"))?
            .publish(messages, breaker)?;
        Ok(true)
    }

    fn enter_phase_four_round(&self) -> anyhow::Result<bool> {
        self.phase_four_breaker.enter_round_and_persist(|_, limit| {
            self.session
                .enter_phase_four_round_if(self.phase_four_turn, limit, || {
                    self.persistence_gate.load(Ordering::SeqCst)
                })
        })
    }

    fn record_phase_four_failure(&self, disable: bool) -> anyhow::Result<()> {
        self.phase_four_breaker
            .record_failure_and_persist(disable, |_, disable| {
                self.session
                    .record_phase_four_failure_if(self.phase_four_turn, disable, || {
                        self.persistence_gate.load(Ordering::SeqCst)
                    })
            })
    }

    fn phase_four_cancelled(&self) -> bool {
        !self.persistence_gate.load(Ordering::SeqCst)
    }

    fn record_phase_four_progress(&self, meaningful: bool) -> anyhow::Result<()> {
        self.phase_four_breaker
            .record_progress_and_persist(meaningful, |_, meaningful| {
                self.session
                    .record_phase_four_progress_if(self.phase_four_turn, meaningful, || {
                        self.persistence_gate.load(Ordering::SeqCst)
                    })
            })
    }

    fn checkpoint_phase_four_side_call(&self, estimated_input: u64) -> anyhow::Result<bool> {
        self.phase_four_breaker
            .charge_and_persist(estimated_input, |previous, proposed| {
                self.session.append_phase_four_breaker_if(
                    self.phase_four_turn,
                    previous,
                    proposed,
                    || self.persistence_gate.load(Ordering::SeqCst),
                )
            })
    }

    fn archive_pointer(&self) -> Option<String> {
        self.archive_pointer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn current_turn_archive_manifest(&self, file_read_available: bool) -> Option<String> {
        self.current_turn_manifest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .render(self.session.archive_root(), file_read_available)
    }

    fn persist_compression(&self, batch: &CompressionBatch) -> Result<bool> {
        let _recorder = self
            .recorder
            .lock()
            .map_err(|_| anyhow::anyhow!("session recorder lock is poisoned"))?;
        self.session
            .append_compression_batch_if(batch, || self.persistence_gate.load(Ordering::SeqCst))
    }

    fn persist_context_projection(
        &self,
        source: &[Message],
        projection: &ContextProjection,
    ) -> Result<bool> {
        if !self.persistence_gate.load(Ordering::SeqCst) {
            return Ok(false);
        }
        let (reply, response) = std::sync::mpsc::channel();
        self.tx
            .send(UiEvent::Turn {
                id: self.turn_id,
                event: Box::new(UiEvent::PersistContextProjection {
                    source: source.to_vec(),
                    projection: projection.clone(),
                    reply,
                }),
            })
            .map_err(|_| anyhow::anyhow!("TUI event loop closed before projection persistence"))?;
        loop {
            match response.recv_timeout(Duration::from_millis(100)) {
                Ok(result) => {
                    let Some(archived) = result.map_err(anyhow::Error::msg)? else {
                        return Ok(false);
                    };
                    self.current_turn_manifest
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .append(archived);
                    return Ok(true);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected)
                    if !self.persistence_gate.load(Ordering::SeqCst) =>
                {
                    return Ok(false);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    anyhow::bail!("TUI projection persistence was cancelled")
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                    if !self.persistence_gate.load(Ordering::SeqCst) =>
                {
                    return Ok(false);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }

    fn text_delta(&self, delta: &str) {
        self.send(UiEvent::TextDelta(delta.to_string()));
    }

    fn reasoning_delta(&self, delta: &str) {
        self.send(UiEvent::ReasoningDelta(delta.to_string()));
    }

    fn tool_started(&self, name: &str, summary: &str) {
        self.send(UiEvent::ToolStarted {
            name: name.to_string(),
            summary: summary.to_string(),
        });
    }

    fn tool_finished(&self, name: &str, ok: bool, duration: Duration, summary: &str, output: &str) {
        self.send(UiEvent::ToolFinished {
            name: name.to_string(),
            ok,
            duration,
            summary: summary.to_string(),
            output: output.to_string(),
        });
    }

    fn message_added(&self, message: &Message) {
        self.send(UiEvent::MessageAdded(message.clone()));
    }

    fn usage(&self, usage: &Usage) {
        self.send(UiEvent::Usage(*usage));
    }

    fn info(&self, message: &str) {
        self.send(UiEvent::Info(message.to_string()));
    }

    fn error(&self, message: &str) {
        self.send(UiEvent::Error(message.to_string()));
    }
}

struct TuiApprover {
    tx: mpsc::UnboundedSender<UiEvent>,
    always: std::sync::Mutex<std::collections::HashSet<String>>,
    recovery: bool,
}

impl TuiApprover {
    fn new(tx: mpsc::UnboundedSender<UiEvent>) -> Self {
        Self {
            tx,
            always: std::sync::Mutex::new(std::collections::HashSet::new()),
            recovery: false,
        }
    }

    fn recovery(tx: mpsc::UnboundedSender<UiEvent>) -> Self {
        Self {
            tx,
            always: std::sync::Mutex::new(std::collections::HashSet::new()),
            recovery: true,
        }
    }

    fn always_allowed(&self, tool: &str) -> bool {
        self.always
            .lock()
            .map(|always| always.contains(tool))
            .unwrap_or(false)
    }

    fn remember(&self, tool: &str) {
        if let Ok(mut always) = self.always.lock() {
            always.insert(tool.to_string());
        }
    }
}

#[async_trait::async_trait]
impl Approver for TuiApprover {
    async fn approve(&self, request: &ApprovalRequest) -> ApprovalDecision {
        if !self.recovery && self.always_allowed(&request.tool) {
            return ApprovalDecision::AllowOnce;
        }
        let (reply, response) = oneshot::channel();
        if self
            .tx
            .send(UiEvent::Approval {
                request: request.clone(),
                reply,
                allow_always: !self.recovery,
                recovery: self.recovery,
            })
            .is_err()
        {
            return ApprovalDecision::Deny;
        }
        let decision = response.await.unwrap_or(ApprovalDecision::Deny);
        if self.recovery && decision == ApprovalDecision::AllowAlways {
            return ApprovalDecision::AllowOnce;
        }
        if decision == ApprovalDecision::AllowAlways {
            self.remember(&request.tool);
        }
        decision
    }
}

struct TuiAsker {
    tx: mpsc::UnboundedSender<UiEvent>,
}

#[async_trait::async_trait]
impl Asker for TuiAsker {
    async fn ask(&self, question: &str) -> String {
        let (reply, response) = oneshot::channel();
        if self
            .tx
            .send(UiEvent::Ask {
                question: question.to_string(),
                reply,
            })
            .is_err()
        {
            return String::new();
        }
        response.await.unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn injection_input_test_app() -> App {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().to_path_buf();
        std::mem::forget(directory);
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_key = "test-key".to_string();
        profile.base_url = "http://127.0.0.1:1".to_string();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let handle = runtime.handle().clone();
        std::mem::forget(runtime);
        App::new(
            handle,
            AppOptions {
                settings: Settings::default(),
                profile,
                agent: AgentKind::Code,
                cwd: PathBuf::from("/tmp/project"),
                store: SessionStore::with_root(path.join("sessions")),
                resume: None,
                theme: Theme::default(),
                approval: ApprovalMode::Ask,
            },
        )
        .unwrap()
    }

    #[test]
    fn running_enter_queues_one_locked_draft_and_escape_withdraws_first() {
        let mut app = injection_input_test_app();
        let (queue, _source) = injection::channel();
        app.injection_queue = Some(queue);
        app.running = true;
        app.turn_id = 1;
        app.input = "Also check the tests".to_string();
        app.cursor = app.char_len();
        let screen = Rect::new(0, 0, 80, 24);

        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), screen);

        assert!(app.pending_injection.is_some());
        assert!(app.status_line().contains("queued"));
        let locked = app.input.clone();
        let cursor = app.cursor;
        app.handle_key(
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE),
            screen,
        );
        app.handle_key(
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
            screen,
        );
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), screen);
        assert_eq!(app.input, locked);
        assert_eq!(app.cursor, cursor);

        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE), screen);
        assert!(app.running);
        assert!(app.pending_injection.is_none());
        assert_eq!(app.input, locked);
        assert!(!app.status_line().contains("queued"));

        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE), screen);
        assert!(!app.running);
        assert_eq!(app.input, locked);
    }

    #[test]
    fn direct_shell_request_and_completion_are_identity_scoped() {
        let mut app = injection_input_test_app();
        app.input = "!printf hello".to_string();
        app.submit();

        let request = app.take_direct_shell().unwrap();
        assert_eq!(request.command, "printf hello");
        assert_eq!(request.cwd, app.cwd);
        assert!(app.take_direct_shell().is_none());
        assert!(app.running);
        assert!(matches!(
            app.view.last(),
            Some(ViewItem::Tool {
                name,
                summary,
                status: ToolStatus::Running,
                ..
            }) if name == "bash" && summary == "printf hello"
        ));

        assert!(!app.complete_direct_shell(
            request.id.wrapping_add(1),
            &request.command,
            Duration::from_secs(1),
            super::super::direct_shell::DirectShellOutcome::Exited(0),
        ));
        assert!(app.running);
        assert!(app.complete_direct_shell(
            request.id,
            &request.command,
            Duration::from_secs(2),
            super::super::direct_shell::DirectShellOutcome::Exited(7),
        ));
        assert!(!app.running);
        assert!(matches!(
            app.view.last(),
            Some(ViewItem::Tool {
                status: ToolStatus::Done,
                ok: Some(false),
                duration: Some(duration),
                output,
                ..
            }) if *duration == Duration::from_secs(2) && output.contains("code 7")
        ));
    }

    #[test]
    fn cancelling_pending_direct_shell_does_not_touch_session_recovery() {
        let mut app = injection_input_test_app();
        app.input = "!sleep 1".to_string();
        app.submit();

        app.cancel();

        assert!(!app.running);
        assert!(app.take_direct_shell().is_none());
        assert!(!app.requires_session_reload);
        assert!(matches!(
            app.view.last(),
            Some(ViewItem::Tool {
                status: ToolStatus::Done,
                ok: Some(false),
                output,
                ..
            }) if output == "Cancelled."
        ));
    }

    #[test]
    fn running_local_commands_are_dispatched_instead_of_injected() {
        let mut app = injection_input_test_app();
        let (queue, source) = injection::channel();
        app.injection_queue = Some(queue);
        app.running = true;
        app.input = "!echo local".to_string();
        app.cursor = app.char_len();
        let screen = Rect::new(0, 0, 80, 24);

        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), screen);

        assert!(app.pending_injection.is_none());
        assert!(source.reserve().is_none());
        assert!(app.input.is_empty());
        assert!(matches!(
            app.view.last(),
            Some(ViewItem::Info(text)) if text.contains("already running")
        ));

        app.input = "/help".to_string();
        app.cursor = app.char_len();
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), screen);

        assert!(matches!(app.modal, Some(Modal::Help)));
        assert!(app.pending_injection.is_none());
        assert!(source.reserve().is_none());

        app.modal = None;
        app.injection_queue = None;
        app.input = "/help".to_string();
        app.cursor = app.char_len();
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), screen);
        assert!(matches!(app.modal, Some(Modal::Help)));

        app.modal = None;
        app.input = "keep this draft".to_string();
        app.cursor = app.char_len();
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), screen);
        assert_eq!(app.input, "keep this draft");
        assert!(matches!(
            app.view.last(),
            Some(ViewItem::Info(text)) if text.contains("already running")
        ));
    }

    #[test]
    fn consumed_injection_renders_raw_text_and_clears_its_owned_draft() {
        let mut app = injection_input_test_app();
        app.session
            .append_message(&Message::user("original request"))
            .unwrap();
        let (queue, _source) = injection::channel();
        app.injection_queue = Some(queue);
        app.running = true;
        app.turn_id = 2;
        app.input = "same-turn direction".to_string();
        app.cursor = app.char_len();
        let screen = Rect::new(0, 0, 80, 24);
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), screen);
        let injection_id = app.pending_injection.as_ref().unwrap().id.clone();
        let (reply, response) = std::sync::mpsc::channel();

        app.apply(UiEvent::PersistInjectedUser {
            message: Message::injected_user(injection_id, "same-turn direction"),
            reply,
        });

        response.recv().unwrap().unwrap();
        assert!(app.pending_injection.is_none());
        assert!(app.input.is_empty());
        assert!(matches!(
            app.view.last(),
            Some(ViewItem::User(text)) if text == "same-turn direction"
        ));
        let loaded = app.store.load(app.session.id()).unwrap();
        assert_eq!(
            loaded
                .records
                .iter()
                .filter(|record| matches!(record, Record::InjectedUser { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn ask_editor_does_not_overwrite_a_queued_injection() {
        let mut app = injection_input_test_app();
        app.session
            .append_message(&Message::user("original request"))
            .unwrap();
        let (queue, _source) = injection::channel();
        app.injection_queue = Some(queue);
        app.running = true;
        app.turn_id = 3;
        app.input = "queued constraint".to_string();
        app.cursor = app.char_len();
        let screen = Rect::new(0, 0, 80, 24);
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), screen);
        let injection_id = app.pending_injection.as_ref().unwrap().id.clone();
        let (ask_reply, mut ask_response) = oneshot::channel();
        app.apply(UiEvent::Ask {
            question: "Proceed?".to_string(),
            reply: ask_reply,
        });
        assert!(app.input.is_empty());
        for character in "yes".chars() {
            app.handle_key(
                KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE),
                screen,
            );
        }
        let (persist_reply, persist_response) = std::sync::mpsc::channel();
        app.apply(UiEvent::PersistInjectedUser {
            message: Message::injected_user(injection_id, "queued constraint"),
            reply: persist_reply,
        });
        persist_response.recv().unwrap().unwrap();
        assert_eq!(app.input, "yes");

        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), screen);

        assert_eq!(ask_response.try_recv().unwrap(), "yes");
        assert!(app.input.is_empty());
        assert!(app.pending_injection.is_none());
    }

    #[test]
    fn input_racing_a_sealed_turn_is_submitted_once_after_terminal_state() {
        let mut app = injection_input_test_app();
        let (queue, source) = injection::channel();
        assert!(source.try_finish());
        app.injection_queue = Some(queue);
        app.running = true;
        app.turn_id = 4;
        app.input = "fresh follow-up".to_string();
        app.cursor = app.char_len();
        let screen = Rect::new(0, 0, 80, 24);

        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), screen);
        assert!(app.deferred_submission);
        let messages = app.messages.clone();
        app.apply(UiEvent::TurnFinished(
            Ok(TurnResult::default()),
            messages,
            false,
        ));

        assert!(app.running);
        assert!(!app.deferred_submission);
        assert!(app.input.is_empty());
        assert_eq!(
            app.messages
                .iter()
                .filter(|message| matches!(message, Message::User { content } if content == "fresh follow-up"))
                .count(),
            1
        );
        app.cancel();
    }

    #[test]
    fn automatic_compaction_records_do_not_change_the_visible_transcript() {
        let records = vec![
            Record::User {
                text: "old request".to_string(),
            },
            Record::Compaction {
                summary: "automatic summary".to_string(),
                compacted_messages: Some(1),
            },
            Record::Assistant {
                content: Some("latest answer".to_string()),
                reasoning: None,
                tool_calls: Vec::new(),
                responses_items: Vec::new(),
                responses_continuation: None,
            },
        ];

        let view = view_from_records(&records);

        assert_eq!(view.len(), 2);
        assert!(matches!(&view[0], ViewItem::User(text) if text == "old request"));
        assert!(matches!(
            &view[1],
            ViewItem::Assistant { text, .. } if text == "latest answer"
        ));
    }

    #[test]
    fn tui_sink_checkpoints_side_call_spend_under_the_turn_gate() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("current")).unwrap();
        let (turn, state) = session.load_phase_four_breaker().unwrap();
        let gate = Arc::new(AtomicBool::new(true));
        let (tx, _rx) = mpsc::unbounded_channel();
        let sink = TuiSink {
            tx,
            turn_id: 1,
            session: session.clone(),
            recorder: Arc::new(Mutex::new(RecordBuffer::default())),
            persistence_gate: gate.clone(),
            recovery_enabled: true,
            recovery_writer: None,
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
            phase_four_turn: turn,
            phase_four_breaker: PhaseFourBreaker::new(100, state),
        };

        assert!(sink.checkpoint_phase_four_side_call(40).unwrap());
        gate.store(false, Ordering::SeqCst);
        assert!(sink.checkpoint_phase_four_side_call(10).is_err());
        let (_, restored) = session.load_phase_four_breaker().unwrap();
        assert_eq!(restored.side_call_tokens, 40);
    }

    #[test]
    fn tui_sink_orders_compression_before_a_buffered_tool_group() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("old")).unwrap();
        session
            .append_message(&Message::assistant(crate::model::AssistantMessage::text(
                "old answer",
            )))
            .unwrap();
        session.append_message(&Message::user("current")).unwrap();
        let recorder = Arc::new(Mutex::new(RecordBuffer::default()));
        let pending = Message::assistant(crate::model::AssistantMessage {
            tool_calls: vec![crate::model::ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            }],
            ..crate::model::AssistantMessage::default()
        });
        recorder.lock().unwrap().record(&session, &pending).unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let sink = TuiSink {
            tx,
            turn_id: 1,
            session: session.clone(),
            recorder: recorder.clone(),
            persistence_gate: Arc::new(AtomicBool::new(true)),
            recovery_enabled: true,
            recovery_writer: None,
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
            phase_four_turn: 0,
            phase_four_breaker: PhaseFourBreaker::new(-1, Default::default()),
        };

        assert!(sink
            .persist_compression(&CompressionBatch {
                folds: vec![crate::compaction::CompressionFold {
                    summary: "old summary".into(),
                    compacted_messages: 2,
                }],
            })
            .unwrap());
        recorder
            .lock()
            .unwrap()
            .record(
                &session,
                &Message::tool("call_1", "read_file", "real result"),
            )
            .unwrap();

        let loaded = store.load(session.id()).unwrap();
        let batch_index = loaded
            .records
            .iter()
            .position(|record| matches!(record, Record::CompressionBatch { .. }))
            .unwrap();
        let tool_assistant_index = loaded
            .records
            .iter()
            .rposition(|record| matches!(record, Record::Assistant { tool_calls, .. } if !tool_calls.is_empty()))
            .unwrap();
        assert!(batch_index < tool_assistant_index);
        assert!(loaded.messages().iter().any(
            |message| matches!(message, Message::Tool { content, .. } if content == "real result")
        ));
    }

    #[test]
    fn tui_injected_persistence_follows_queued_tool_messages_exactly_once() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("current")).unwrap();
        let recorder = Arc::new(Mutex::new(RecordBuffer::default()));
        let gate = Arc::new(AtomicBool::new(true));
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sink = Arc::new(TuiSink {
            tx,
            turn_id: 9,
            session: session.clone(),
            recorder: recorder.clone(),
            persistence_gate: gate,
            recovery_enabled: true,
            recovery_writer: None,
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
            phase_four_turn: 0,
            phase_four_breaker: PhaseFourBreaker::new(-1, Default::default()),
        });
        let assistant = Message::assistant(crate::model::AssistantMessage {
            tool_calls: vec![crate::model::ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            }],
            ..crate::model::AssistantMessage::default()
        });
        sink.message_added(&assistant);
        sink.message_added(&Message::tool("call_1", "read_file", "completed result"));
        let injected = Message::injected_user("inject-1", "same-turn direction");
        let worker = {
            let sink = sink.clone();
            let injected = injected.clone();
            std::thread::spawn(move || sink.persist_injected_user(&injected))
        };

        for _ in 0..3 {
            let UiEvent::Turn { id, event } = rx.blocking_recv().expect("queued turn event") else {
                panic!("expected a turn-scoped event");
            };
            assert_eq!(id, 9);
            match *event {
                UiEvent::MessageAdded(message) => {
                    recorder.lock().unwrap().record(&session, &message).unwrap()
                }
                UiEvent::PersistInjectedUser { message, reply } => {
                    let result = recorder
                        .lock()
                        .unwrap()
                        .record(&session, &message)
                        .map_err(|error| format!("{error:#}"));
                    let _ = reply.send(result);
                }
                _ => panic!("unexpected queued event"),
            }
        }
        worker.join().unwrap().unwrap();

        let loaded = store.load(session.id()).unwrap();
        let tool_index = loaded
            .records
            .iter()
            .position(|record| matches!(record, Record::Tool { content, .. } if content == "completed result"))
            .unwrap();
        let injected_indices = loaded
            .records
            .iter()
            .enumerate()
            .filter_map(|(index, record)| {
                matches!(
                    record,
                    Record::InjectedUser { injection_id, text }
                        if injection_id == "inject-1" && text == "same-turn direction"
                )
                .then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(injected_indices.len(), 1);
        assert!(tool_index < injected_indices[0]);
    }

    #[test]
    fn tui_injected_persistence_cancelled_before_dispatch_writes_nothing() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("current")).unwrap();
        let gate = Arc::new(AtomicBool::new(true));
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sink = Arc::new(TuiSink {
            tx,
            turn_id: 11,
            session: session.clone(),
            recorder: Arc::new(Mutex::new(RecordBuffer::default())),
            persistence_gate: gate.clone(),
            recovery_enabled: true,
            recovery_writer: None,
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
            phase_four_turn: 0,
            phase_four_breaker: PhaseFourBreaker::new(-1, Default::default()),
        });
        let worker = {
            let sink = sink.clone();
            std::thread::spawn(move || {
                sink.persist_injected_user(&Message::injected_user(
                    "inject-cancelled",
                    "do not persist",
                ))
            })
        };
        let UiEvent::Turn { id, event } = rx.blocking_recv().expect("queued turn event") else {
            panic!("expected a turn-scoped event");
        };
        assert_eq!(id, 11);

        gate.store(false, Ordering::SeqCst);
        drop(event);

        let error = worker.join().unwrap().unwrap_err();
        assert!(error.to_string().contains("cancelled"), "{error:#}");
        let loaded = store.load(session.id()).unwrap();
        assert!(!loaded.records.iter().any(|record| matches!(
            record,
            Record::InjectedUser { injection_id, .. } if injection_id == "inject-cancelled"
        )));
    }

    #[test]
    fn tui_projection_barrier_persists_queued_messages_first() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("current")).unwrap();
        let recorder = Arc::new(Mutex::new(RecordBuffer::default()));
        let gate = Arc::new(AtomicBool::new(true));
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sink = Arc::new(TuiSink {
            tx,
            turn_id: 7,
            session: session.clone(),
            recorder: recorder.clone(),
            persistence_gate: gate.clone(),
            recovery_enabled: true,
            recovery_writer: None,
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
            phase_four_turn: 0,
            phase_four_breaker: PhaseFourBreaker::new(-1, Default::default()),
        });
        let assistant = Message::assistant(crate::model::AssistantMessage {
            tool_calls: vec![crate::model::ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            }],
            ..crate::model::AssistantMessage::default()
        });
        sink.message_added(&assistant);
        let tool_result = Message::tool("call_1", "read_file", "large result");
        sink.message_added(&tool_result);
        let source = vec![Message::user("current"), assistant.clone(), tool_result];
        let projection = ContextProjection {
            historical_folds: Vec::new(),
            current_turn: Some(crate::compaction::CurrentTurnFold {
                note: "fresh progress".into(),
                compacted_messages: 2,
            }),
        };
        let worker = {
            let sink = sink.clone();
            let projection = projection.clone();
            let source = source.clone();
            std::thread::spawn(move || sink.persist_context_projection(&source, &projection))
        };

        for _ in 0..3 {
            let UiEvent::Turn { id, event } = rx.blocking_recv().expect("queued turn event") else {
                panic!("expected a turn-scoped event");
            };
            assert_eq!(id, 7);
            match *event {
                UiEvent::MessageAdded(message) => {
                    recorder.lock().unwrap().record(&session, &message).unwrap()
                }
                UiEvent::PersistContextProjection {
                    source,
                    projection,
                    reply,
                } => {
                    let result = persist_context_projection(
                        &session,
                        &recorder,
                        Some(&gate),
                        false,
                        &source,
                        &projection,
                    );
                    let _ = reply.send(result);
                }
                _ => panic!("unexpected queued event"),
            }
        }
        assert!(worker.join().unwrap().unwrap());
        let manifest = sink.current_turn_archive_manifest(true).unwrap();
        assert!(manifest.contains("r1 001 read_file(…) → unknown, 12 chars"));
        assert!(manifest.contains("To inspect archived input/output"));

        let loaded = store.load(session.id()).unwrap();
        let projection_index = loaded
            .records
            .iter()
            .position(|record| matches!(record, Record::ContextProjection { .. }))
            .unwrap();
        assert!(matches!(
            loaded.records[projection_index - 1],
            Record::Tool { .. }
        ));
        assert_eq!(projection_index, loaded.records.len() - 1);
        assert!(matches!(
            loaded.messages().last(),
            Some(Message::LastWords { .. })
        ));
        let archive = crate::compaction_archive::ArchiveStore::with_root(session.archive_root())
            .reconcile()
            .unwrap();
        assert_eq!(archive.live_records.len(), 1);
        let archived = std::fs::read_to_string(
            session
                .archive_root()
                .join(&archive.live_records[0].relative_path),
        )
        .unwrap();
        assert!(archived.contains("large result"));
        assert!(archived.contains("Group ID: `call_1`"));
    }

    #[test]
    fn tui_sink_skips_compression_when_turn_started_unsaved() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let sink = TuiSink {
            tx,
            turn_id: 1,
            session,
            recorder: Arc::new(Mutex::new(RecordBuffer::default())),
            persistence_gate: Arc::new(AtomicBool::new(false)),
            recovery_enabled: false,
            recovery_writer: None,
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
            phase_four_turn: 0,
            phase_four_breaker: PhaseFourBreaker::new(-1, Default::default()),
        };

        assert!(!sink
            .persist_compression(&CompressionBatch::default())
            .unwrap());
        assert!(!sink
            .persist_recovery_checkpoint(&[Message::user("unsaved")])
            .unwrap());
    }

    #[test]
    fn tui_sink_declines_projection_after_turn_gate_closes() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sink = TuiSink {
            tx,
            turn_id: 1,
            session,
            recorder: Arc::new(Mutex::new(RecordBuffer::default())),
            persistence_gate: Arc::new(AtomicBool::new(false)),
            recovery_enabled: true,
            recovery_writer: None,
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
            phase_four_turn: 0,
            phase_four_breaker: PhaseFourBreaker::new(-1, Default::default()),
        };
        let projection = ContextProjection {
            historical_folds: vec![crate::compaction::CompressionFold {
                summary: "old".into(),
                compacted_messages: 2,
            }],
            current_turn: None,
        };

        assert!(!sink.persist_context_projection(&[], &projection).unwrap());
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn queued_projection_barrier_declines_when_turn_is_cancelled() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        let gate = Arc::new(AtomicBool::new(true));
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sink = Arc::new(TuiSink {
            tx,
            turn_id: 1,
            session,
            recorder: Arc::new(Mutex::new(RecordBuffer::default())),
            persistence_gate: gate.clone(),
            recovery_enabled: true,
            recovery_writer: None,
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
            phase_four_turn: 0,
            phase_four_breaker: PhaseFourBreaker::new(-1, Default::default()),
        });
        let projection = ContextProjection {
            historical_folds: vec![crate::compaction::CompressionFold {
                summary: "old".into(),
                compacted_messages: 2,
            }],
            current_turn: None,
        };
        let worker = {
            let sink = sink.clone();
            std::thread::spawn(move || sink.persist_context_projection(&[], &projection))
        };
        let queued = rx.blocking_recv().expect("queued projection barrier");

        gate.store(false, Ordering::SeqCst);
        drop(queued);

        assert!(!worker.join().unwrap().unwrap());
    }

    #[test]
    fn clipping_uses_display_width_and_popup_width() {
        let wide = "写".repeat(30);
        // At 40x7 a character-count heuristic would claim the command fits,
        // but the wide characters need one more rendered row.
        assert!(
            approval_layout(Rect::new(0, 0, 40, 7), "bash", &wide, true).clipped,
            "{wide}"
        );
        assert!(!approval_layout(Rect::new(0, 0, 120, 40), "bash", "cargo test", true).clipped);
        // A very narrow popup has too little room for the text: refuse.
        assert!(approval_layout(Rect::new(0, 0, 12, 20), "bash", "cargo test", true).clipped);
        // The shorter recovery hint fits one row where the ordinary hint wraps.
        assert!(approval_layout(Rect::new(0, 0, 60, 5), "bash", "cargo test", true).clipped);
        assert!(!approval_layout(Rect::new(0, 0, 60, 5), "bash", "cargo test", false).clipped);
    }

    #[tokio::test]
    async fn recovery_approver_never_remembers_mutating_tools() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let approver = Arc::new(TuiApprover::recovery(tx));
        let request = ApprovalRequest::new("bash", "cargo test");

        for response in [ApprovalDecision::AllowAlways, ApprovalDecision::Deny] {
            let pending = {
                let approver = approver.clone();
                let request = request.clone();
                tokio::spawn(async move { approver.approve(&request).await })
            };
            match rx.recv().await.expect("recovery approval prompt") {
                UiEvent::Approval {
                    reply,
                    allow_always,
                    recovery,
                    ..
                } => {
                    assert!(!allow_always);
                    assert!(recovery);
                    let _ = reply.send(response);
                }
                _ => panic!("expected a recovery approval event"),
            }
            let expected = if response == ApprovalDecision::AllowAlways {
                ApprovalDecision::AllowOnce
            } else {
                response
            };
            assert_eq!(pending.await.unwrap(), expected);
        }

        drop(rx);
        assert_eq!(
            approver.approve(&request).await,
            ApprovalDecision::Deny,
            "closed recovery prompt channel must fail closed"
        );
    }

    #[tokio::test]
    async fn always_allow_is_remembered_per_tool() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let approver = Arc::new(TuiApprover::new(tx));
        let request = ApprovalRequest::new("bash", "cargo test");
        let first = {
            let approver = approver.clone();
            let request = request.clone();
            tokio::spawn(async move { approver.approve(&request).await })
        };
        let event = rx.recv().await.expect("approval prompt");
        match event {
            UiEvent::Approval { reply, .. } => {
                let _ = reply.send(ApprovalDecision::AllowAlways);
            }
            _ => panic!("expected an approval event"),
        }
        assert_eq!(first.await.unwrap(), ApprovalDecision::AllowAlways);

        // The same tool is approved without asking again.
        assert_eq!(
            approver.approve(&request).await,
            ApprovalDecision::AllowOnce
        );
        assert!(rx.try_recv().is_err(), "no second prompt for the same tool");

        // A different tool still prompts.
        let other = ApprovalRequest::new("write_file", "notes.txt");
        let pending = {
            let approver = approver.clone();
            tokio::spawn(async move { approver.approve(&other).await })
        };
        match rx.recv().await.expect("second prompt") {
            UiEvent::Approval { reply, .. } => {
                let _ = reply.send(ApprovalDecision::Deny);
            }
            _ => panic!("expected an approval event"),
        }
        assert_eq!(pending.await.unwrap(), ApprovalDecision::Deny);
    }
}
