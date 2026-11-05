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

use crate::agent::{Agent, AgentKind, AgentOptions, EventSink, TurnResult};
use crate::compaction::{Compaction, CompressionBatch, ContextProjection};
use crate::compaction_archive::{ArchivedContextProjection, CurrentTurnArchiveManifest};
use crate::config::{ApprovalMode, ModelProfile, Settings};
use crate::interaction::{AllowAll, ApprovalDecision, ApprovalRequest, Approver, Asker, DenyAll};
use crate::model::{Message, ModelClient, Usage};
use crate::session::{Record, RecordBuffer, Session, SessionStore};
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
    ShellFinished {
        ok: bool,
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

pub struct App {
    pub settings: Settings,
    pub profile: ModelProfile,
    pub agent_kind: AgentKind,
    pub approval_mode: ApprovalMode,
    pub cwd: PathBuf,
    pub theme: Theme,
    pub store: SessionStore,
    pub session: Session,
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
    persistence_gate: Option<Arc<AtomicBool>>,
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
        let (session, messages, view) = match options.resume.as_deref() {
            Some(id) => {
                let loaded = options.store.load(id)?;
                let messages = loaded.messages();
                let view = view_from_records(&loaded.records);
                let session = options.store.reopen(id)?;
                (session, messages, view)
            }
            None => {
                let session = options.store.create(
                    &options.cwd.display().to_string(),
                    &options.profile.model,
                    options.agent.as_str(),
                )?;
                (session, Vec::new(), welcome())
            }
        };
        Ok(Self {
            settings: options.settings,
            profile: options.profile,
            agent_kind: options.agent,
            approval_mode: options.approval,
            cwd: options.cwd,
            theme: options.theme,
            store: options.store,
            session,
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
            persistence_gate: None,
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
            UiEvent::Approval { request, reply } => {
                self.modal = Some(Modal::Approval { request, reply });
            }
            UiEvent::Ask { question, reply } => {
                self.pending_ask = Some(reply);
                self.view
                    .push(ViewItem::Info(format!("The assistant asks: {question}")));
                self.view.push(ViewItem::Info(
                    "Type your answer and press Enter.".to_string(),
                ));
            }
            UiEvent::TurnFinished(result, messages, force_compression) => {
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
                        if turn.tool_calls > 0 {
                            self.view.push(ViewItem::Info(format!(
                                "Done in {:.1}s · {} tool call(s) · {} model calls",
                                turn.duration.as_secs_f64(),
                                turn.tool_calls,
                                turn.rounds
                            )));
                        }
                    }
                    Err(error) => self.view.push(ViewItem::Error(error)),
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
            UiEvent::ShellFinished { ok, duration } => {
                self.running = false;
                self.turn_started = None;
                self.elapsed = duration;
                if !ok {
                    self.view.push(ViewItem::Info(
                        "Shell command failed; its output is not part of the conversation."
                            .to_string(),
                    ));
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
                } else if !self.running {
                    self.submit();
                }
            }
            KeyCode::Char(character) => {
                self.insert_char(character);
            }
            KeyCode::Backspace => self.backspace(),
            KeyCode::Delete => self.delete(),
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => {
                self.cursor = (self.cursor + 1).min(self.char_len());
            }
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.char_len(),
            KeyCode::Up => self.history_previous(),
            KeyCode::Down => self.history_next(),
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
            KeyCode::Tab => self.complete_command(),
            _ => {}
        }
    }

    fn handle_modal_key(&mut self, modal: Modal, key: KeyEvent, screen: Rect) {
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
            KeyCode::Char('a') | KeyCode::Char('A') => {
                self.decide_approval(ApprovalDecision::AllowAlways, screen)
            }
            _ => {}
        }
    }

    /// Approve or deny the pending request. Allow decisions are refused while
    /// the dialog has to clip the command, so the user cannot approve text they
    /// cannot read.
    fn decide_approval(&mut self, decision: ApprovalDecision, screen: Rect) {
        let Some(Modal::Approval { request, .. }) = &self.modal else {
            self.modal = None;
            return;
        };
        if matches!(
            decision,
            ApprovalDecision::AllowOnce | ApprovalDecision::AllowAlways
        ) && approval_layout(screen, &request.tool, &request.summary).clipped
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

    fn reload_session_before_turn(&mut self) -> bool {
        if !self.requires_session_reload {
            return true;
        }
        match self.session.load_messages() {
            Ok(messages) => {
                self.messages = messages;
                self.reset_recorder();
                self.requires_session_reload = false;
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
        let message_text = self.expand_mentions(&raw);
        let user = Message::user(message_text);
        self.messages.push(user.clone());
        if let Err(error) = self.session.append_message(&user) {
            self.unsaved_messages = true;
            self.view.push(ViewItem::Error(format!(
                "failed to save session: {error:#}"
            )));
        }
        self.view.push(ViewItem::User(raw));
        self.view.push(ViewItem::Assistant {
            text: String::new(),
            reasoning: String::new(),
            streaming: true,
        });
        self.start_agent();
    }

    fn start_agent(&mut self) {
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
        let approver = self.approver.clone();
        let asker = self.asker.clone();
        let tools = self.tools.clone();
        let mut messages = self.messages.clone();
        let persistence_gate = Arc::new(AtomicBool::new(!self.unsaved_messages));
        self.persistence_gate = Some(persistence_gate.clone());
        let sink: Arc<dyn EventSink> = Arc::new(TuiSink {
            tx: tx.clone(),
            turn_id,
            session: self.session.clone(),
            recorder: self.recorder.clone(),
            persistence_gate,
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
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
                let turn = agent.run(&mut messages, sink.as_ref()).await?;
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
        if let Some(gate) = self.persistence_gate.take() {
            gate.store(false, Ordering::SeqCst);
        }
        if let Some(handle) = self.cancel_handle.take() {
            handle.abort();
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
        if let Some(Modal::Approval { reply, .. }) = self.modal.take() {
            let _ = reply.send(ApprovalDecision::Deny);
        }
        if let Some(reply) = self.pending_ask.take() {
            let _ = reply.send(String::new());
        }
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
        let turn_id = self.turn_id;
        self.turn_started = Some(Instant::now());
        self.view.push(ViewItem::Tool {
            name: "bash".to_string(),
            summary: command.clone(),
            status: ToolStatus::Running,
            ok: None,
            duration: None,
            output: String::new(),
        });
        let tx = self.events_tx.clone();
        let cwd = self.cwd.clone();
        let handle = self.runtime.clone();
        let task = handle.spawn(async move {
            let started = Instant::now();
            let result =
                crate::tools::shell::run_command(&command, &cwd, Duration::from_secs(120)).await;
            let ok = result.is_ok();
            match result {
                Ok(output) => {
                    let _ = tx.send(UiEvent::Turn {
                        id: turn_id,
                        event: Box::new(UiEvent::ToolFinished {
                            name: "bash".to_string(),
                            ok: true,
                            duration: started.elapsed(),
                            summary: command.clone(),
                            output,
                        }),
                    });
                }
                Err(error) => {
                    let _ = tx.send(UiEvent::Turn {
                        id: turn_id,
                        event: Box::new(UiEvent::ToolFinished {
                            name: "bash".to_string(),
                            ok: false,
                            duration: started.elapsed(),
                            summary: command.clone(),
                            output: format!("{error:#}"),
                        }),
                    });
                }
            }
            let _ = tx.send(UiEvent::Turn {
                id: turn_id,
                event: Box::new(UiEvent::ShellFinished {
                    ok,
                    duration: started.elapsed(),
                }),
            });
        });
        self.cancel_handle = Some(task);
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
        if self.running && matches!(command, "/new" | "/clear" | "/resume" | "/compact") {
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
                    match self.store.load(argument) {
                        Ok(loaded) => {
                            self.messages = loaded.messages();
                            self.view = view_from_records(&loaded.records);
                            self.cwd = PathBuf::from(&loaded.meta.cwd);
                            self.session = match self.store.reopen(argument) {
                                Ok(session) => session,
                                Err(error) => {
                                    self.view.push(ViewItem::Error(format!(
                                        "cannot resume session: {error:#}"
                                    )));
                                    return;
                                }
                            };
                            self.reset_recorder();
                            self.unsaved_messages = false;
                            self.requires_session_reload = false;
                            self.view.push(ViewItem::Info(format!(
                                "Resumed session {} ({}).",
                                loaded.meta.id, loaded.meta.cwd
                            )));
                        }
                        Err(error) => self
                            .view
                            .push(ViewItem::Error(format!("cannot resume: {error:#}"))),
                    }
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

    fn new_session(&mut self) {
        self.messages.clear();
        self.view = welcome();
        match self.store.create(
            &self.cwd.display().to_string(),
            &self.profile.model,
            self.agent_kind.as_str(),
        ) {
            Ok(session) => {
                self.session = session;
                self.reset_recorder();
                self.unsaved_messages = false;
                self.requires_session_reload = false;
            }
            Err(error) => self.view.push(ViewItem::Error(format!(
                "failed to create session: {error:#}"
            ))),
        }
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

/// Compute the approval popup geometry for the current terminal size.
///
/// `ui::draw_approval` renders with this layout, and `decide_approval` refuses
/// allow decisions when `clipped` is set, so approval can never be granted for
/// text the user cannot see.
pub(crate) fn approval_layout(screen: Rect, tool: &str, summary: &str) -> ApprovalLayout {
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
    let hint_lines = wrapped_cell_lines(
        "  y allow once · a always allow · n deny · esc deny",
        interior,
    );
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
            | crate::session::Record::Usage { .. }
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
    archive_pointer: Mutex<Option<String>>,
    current_turn_manifest: Mutex<CurrentTurnArchiveManifest>,
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
}

impl TuiApprover {
    fn new(tx: mpsc::UnboundedSender<UiEvent>) -> Self {
        Self {
            tx,
            always: std::sync::Mutex::new(std::collections::HashSet::new()),
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
        if self.always_allowed(&request.tool) {
            return ApprovalDecision::AllowOnce;
        }
        let (reply, response) = oneshot::channel();
        if self
            .tx
            .send(UiEvent::Approval {
                request: request.clone(),
                reply,
            })
            .is_err()
        {
            return ApprovalDecision::Deny;
        }
        let decision = response.await.unwrap_or(ApprovalDecision::Deny);
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
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
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
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
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
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
        };

        assert!(!sink
            .persist_compression(&CompressionBatch::default())
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
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
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
            archive_pointer: Mutex::new(None),
            current_turn_manifest: Mutex::new(CurrentTurnArchiveManifest::default()),
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
            approval_layout(Rect::new(0, 0, 40, 7), "bash", &wide).clipped,
            "{wide}"
        );
        assert!(!approval_layout(Rect::new(0, 0, 120, 40), "bash", "cargo test").clipped);
        // A very narrow popup has too little room for the text: refuse.
        assert!(approval_layout(Rect::new(0, 0, 12, 20), "bash", "cargo test").clipped);
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
