//! Per-turn ordering and coalescing for recovery sidecar writes.

use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use anyhow::{anyhow, bail, Context, Result};

use crate::model::Message;
use crate::phase4_breaker::PhaseFourBreakerState;
use crate::session::Session;

#[derive(Clone)]
struct RecoverySnapshot {
    messages: Vec<Message>,
    breaker: PhaseFourBreakerState,
}

trait RecoveryCheckpointIo: Send + Sync + 'static {
    fn write(&self, snapshot: &RecoverySnapshot, strict: bool) -> Result<()>;
    fn clear(&self) -> Result<()>;
}

struct SessionRecoveryIo {
    session: Session,
    turn: u64,
}

impl RecoveryCheckpointIo for SessionRecoveryIo {
    fn write(&self, snapshot: &RecoverySnapshot, strict: bool) -> Result<()> {
        let persisted = self.session.persist_recovery_checkpoint_candidate(
            &snapshot.messages,
            self.turn,
            &snapshot.breaker,
        )?;
        if strict && !persisted {
            bail!("strict recovery checkpoint was superseded by the durable transcript");
        }
        Ok(())
    }

    fn clear(&self) -> Result<()> {
        self.session.clear_recovery_checkpoint()
    }
}

enum Control {
    Write(RecoverySnapshot, bool, Sender<Result<()>>),
    Preserve(Sender<Result<()>>),
    Clear(Sender<Result<()>>),
}

struct WriterState {
    background: Option<RecoverySnapshot>,
    controls: VecDeque<Control>,
    closed: bool,
    terminated: bool,
    last_error: Option<String>,
}

struct WriterInner {
    io: Arc<dyn RecoveryCheckpointIo>,
    state: Mutex<WriterState>,
    wake: Condvar,
}

/// Clonable submission side held by an active run sink.
#[derive(Clone)]
pub(crate) struct RecoveryCheckpointHandle {
    inner: Arc<WriterInner>,
}

impl RecoveryCheckpointHandle {
    /// Replace the pending best-effort snapshot with this newer state.
    pub(crate) fn publish(
        &self,
        messages: Vec<Message>,
        breaker: PhaseFourBreakerState,
    ) -> Result<()> {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        ensure_open(&state)?;
        state.background = Some(RecoverySnapshot { messages, breaker });
        self.inner.wake.notify_one();
        Ok(())
    }

    /// Supersede queued background state and wait for this exact snapshot.
    pub(crate) fn barrier(
        &self,
        messages: &[Message],
        breaker: PhaseFourBreakerState,
    ) -> Result<()> {
        let (tx, rx) = mpsc::channel();
        {
            let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
            ensure_open(&state)?;
            state.background = None;
            state.controls.push_back(Control::Write(
                RecoverySnapshot {
                    messages: messages.to_vec(),
                    breaker,
                },
                true,
                tx,
            ));
            self.inner.wake.notify_one();
        }
        wait_for(rx, "strict recovery checkpoint")
    }
}

/// Owner side that terminates and joins one turn's writer.
pub(crate) struct RecoveryCheckpointWriter {
    handle: RecoveryCheckpointHandle,
    worker: Option<JoinHandle<()>>,
}

impl RecoveryCheckpointWriter {
    pub(crate) fn new(session: Session, turn: u64) -> Result<Self> {
        let name = format!("icode-recovery-{}-{turn}", session.id());
        Self::with_io(Arc::new(SessionRecoveryIo { session, turn }), name)
    }

    fn with_io(io: Arc<dyn RecoveryCheckpointIo>, name: String) -> Result<Self> {
        let inner = Arc::new(WriterInner {
            io,
            state: Mutex::new(WriterState {
                background: None,
                controls: VecDeque::new(),
                closed: false,
                terminated: false,
                last_error: None,
            }),
            wake: Condvar::new(),
        });
        let worker_inner = inner.clone();
        let worker = std::thread::Builder::new()
            .name(name)
            .spawn(move || worker_entry(worker_inner))
            .context("failed to start recovery checkpoint writer")?;
        Ok(Self {
            handle: RecoveryCheckpointHandle { inner },
            worker: Some(worker),
        })
    }

    pub(crate) fn handle(&self) -> RecoveryCheckpointHandle {
        self.handle.clone()
    }

    /// Seal submissions and enqueue a terminal preserve after accepted state.
    pub(crate) fn begin_preserve(&self) -> Result<Receiver<Result<()>>> {
        let (tx, rx) = mpsc::channel();
        let mut state = self
            .handle
            .inner
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        ensure_open(&state)?;
        state.closed = true;
        if let Some(snapshot) = state.background.take() {
            let (ignored_tx, _ignored_rx) = mpsc::channel();
            state
                .controls
                .push_back(Control::Write(snapshot, false, ignored_tx));
        }
        state.controls.push_back(Control::Preserve(tx));
        self.handle.inner.wake.notify_one();
        Ok(rx)
    }

    pub(crate) fn wait_preserved(&mut self, receiver: Receiver<Result<()>>) -> Result<()> {
        let result = wait_for(receiver, "recovery checkpoint preserve");
        let join = self.join_worker();
        result.and(join)
    }

    pub(crate) fn preserve(&mut self) -> Result<()> {
        let receiver = self.begin_preserve()?;
        self.wait_preserved(receiver)
    }

    /// Seal submissions, obsolete pending background state, and enqueue clear last.
    fn begin_clear(&self) -> Result<Receiver<Result<()>>> {
        let (tx, rx) = mpsc::channel();
        let mut state = self
            .handle
            .inner
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        ensure_open(&state)?;
        state.closed = true;
        state.background = None;
        state.controls.push_back(Control::Clear(tx));
        self.handle.inner.wake.notify_one();
        Ok(rx)
    }

    fn wait_cleared(&mut self, receiver: Receiver<Result<()>>) -> Result<()> {
        let result = wait_for(receiver, "recovery checkpoint cleanup");
        let join = self.join_worker();
        result.and(join)
    }

    /// Seal submissions, obsolete pending background state, and clear last.
    pub(crate) fn clear(&mut self) -> Result<()> {
        let receiver = self.begin_clear()?;
        self.wait_cleared(receiver)
    }

    fn join_worker(&mut self) -> Result<()> {
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| anyhow!("recovery checkpoint writer panicked"))?;
        }
        Ok(())
    }
}

impl Drop for RecoveryCheckpointWriter {
    fn drop(&mut self) {
        if self.worker.is_none() {
            return;
        }
        if let Ok(receiver) = self.begin_preserve() {
            let _ = self.wait_preserved(receiver);
        } else {
            let _ = self.join_worker();
        }
    }
}

fn ensure_open(state: &WriterState) -> Result<()> {
    if state.closed || state.terminated {
        bail!("recovery checkpoint writer is closed");
    }
    Ok(())
}

fn wait_for(receiver: Receiver<Result<()>>, operation: &str) -> Result<()> {
    receiver
        .recv()
        .with_context(|| format!("{operation} worker stopped without replying"))?
}

fn worker_entry(inner: Arc<WriterInner>) {
    let worker = inner.clone();
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker_loop(worker))).is_err() {
        fail_waiters(&inner, "recovery checkpoint writer panicked");
    }
}

fn worker_loop(inner: Arc<WriterInner>) {
    loop {
        let work = {
            let mut state = inner.state.lock().unwrap_or_else(|p| p.into_inner());
            loop {
                if let Some(control) = state.controls.pop_front() {
                    break Some(control);
                }
                if let Some(snapshot) = state.background.take() {
                    let (ignored_tx, _ignored_rx) = mpsc::channel();
                    break Some(Control::Write(snapshot, false, ignored_tx));
                }
                if state.terminated || (state.closed && state.controls.is_empty()) {
                    break None;
                }
                state = inner.wake.wait(state).unwrap_or_else(|p| p.into_inner());
            }
        };
        let Some(work) = work else {
            return;
        };
        match work {
            Control::Write(snapshot, strict, reply) => {
                let result = inner.io.write(&snapshot, strict);
                record_write_result(&inner, &result);
                let _ = reply.send(result);
            }
            Control::Preserve(reply) => {
                let result = terminal_result(&inner);
                let _ = reply.send(result);
                mark_terminated(&inner);
                return;
            }
            Control::Clear(reply) => {
                let result = inner.io.clear();
                let _ = reply.send(result);
                mark_terminated(&inner);
                return;
            }
        }
    }
}

fn fail_waiters(inner: &WriterInner, message: &str) {
    let mut state = inner.state.lock().unwrap_or_else(|p| p.into_inner());
    state.closed = true;
    state.terminated = true;
    state.last_error = Some(message.to_string());
    state.background = None;
    while let Some(control) = state.controls.pop_front() {
        let reply = match control {
            Control::Write(_, _, reply) | Control::Preserve(reply) | Control::Clear(reply) => reply,
        };
        let _ = reply.send(Err(anyhow!(message.to_string())));
    }
    inner.wake.notify_all();
}

fn record_write_result(inner: &WriterInner, result: &Result<()>) {
    let mut state = inner.state.lock().unwrap_or_else(|p| p.into_inner());
    state.last_error = result.as_ref().err().map(|error| format!("{error:#}"));
}

fn terminal_result(inner: &WriterInner) -> Result<()> {
    let state = inner.state.lock().unwrap_or_else(|p| p.into_inner());
    match &state.last_error {
        Some(error) => Err(anyhow!(error.clone())),
        None => Ok(()),
    }
}

fn mark_terminated(inner: &WriterInner) {
    let mut state = inner.state.lock().unwrap_or_else(|p| p.into_inner());
    state.terminated = true;
    inner.wake.notify_all();
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::mpsc::TryRecvError;

    use super::*;

    #[derive(Default)]
    struct TestState {
        operations: Vec<String>,
        durable: Option<String>,
    }

    struct TestIo {
        state: Mutex<TestState>,
        blocked: HashSet<String>,
        failing: HashSet<String>,
        started: Sender<String>,
        release: Mutex<Receiver<()>>,
    }

    impl RecoveryCheckpointIo for TestIo {
        fn write(&self, snapshot: &RecoverySnapshot, strict: bool) -> Result<()> {
            let label = snapshot
                .messages
                .iter()
                .rev()
                .find_map(|message| match message {
                    Message::User { content } => Some(content.clone()),
                    _ => None,
                })
                .expect("test snapshot label");
            let operation = format!("{}:{label}", if strict { "strict" } else { "background" });
            self.state
                .lock()
                .unwrap()
                .operations
                .push(operation.clone());
            self.started.send(operation).unwrap();
            if self.blocked.contains(&label) {
                self.release.lock().unwrap().recv().unwrap();
            }
            if self.failing.contains(&label) {
                bail!("injected failure for {label}");
            }
            self.state.lock().unwrap().durable = Some(label);
            Ok(())
        }

        fn clear(&self) -> Result<()> {
            let mut state = self.state.lock().unwrap();
            state.operations.push("clear".into());
            state.durable = None;
            Ok(())
        }
    }

    fn fixture(
        blocked: &[&str],
        failing: &[&str],
    ) -> (
        RecoveryCheckpointWriter,
        Arc<TestIo>,
        Receiver<String>,
        Sender<()>,
    ) {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let io = Arc::new(TestIo {
            state: Mutex::new(TestState::default()),
            blocked: blocked.iter().map(|value| (*value).to_string()).collect(),
            failing: failing.iter().map(|value| (*value).to_string()).collect(),
            started: started_tx,
            release: Mutex::new(release_rx),
        });
        let writer = RecoveryCheckpointWriter::with_io(io.clone(), "test-recovery".into())
            .expect("writer starts");
        (writer, io, started_rx, release_tx)
    }

    fn publish(handle: &RecoveryCheckpointHandle, label: &str) {
        handle
            .publish(vec![Message::user(label)], PhaseFourBreakerState::default())
            .unwrap();
    }

    #[test]
    fn burst_publications_keep_only_the_latest_pending_snapshot() {
        let (mut writer, io, started, release) = fixture(&["A"], &[]);
        let handle = writer.handle();
        publish(&handle, "A");
        assert_eq!(started.recv().unwrap(), "background:A");
        publish(&handle, "B");
        publish(&handle, "C");
        publish(&handle, "D");
        release.send(()).unwrap();

        writer.preserve().unwrap();
        let state = io.state.lock().unwrap();
        assert_eq!(state.operations, ["background:A", "background:D"]);
        assert_eq!(state.durable.as_deref(), Some("D"));
    }

    #[test]
    fn strict_barrier_waits_for_in_flight_write_and_wins_last() {
        let (mut writer, io, started, release) = fixture(&["A"], &[]);
        let handle = writer.handle();
        publish(&handle, "A");
        assert_eq!(started.recv().unwrap(), "background:A");
        let (result_tx, result_rx) = mpsc::channel();
        let barrier = handle.clone();
        let thread = std::thread::spawn(move || {
            let result = barrier.barrier(&[Message::user("B")], PhaseFourBreakerState::default());
            result_tx.send(result).unwrap();
        });
        while writer
            .handle
            .inner
            .state
            .lock()
            .unwrap()
            .controls
            .is_empty()
        {
            std::thread::yield_now();
        }
        assert!(matches!(result_rx.try_recv(), Err(TryRecvError::Empty)));
        release.send(()).unwrap();
        assert_eq!(started.recv().unwrap(), "strict:B");
        result_rx.recv().unwrap().unwrap();
        thread.join().unwrap();

        writer.preserve().unwrap();
        let state = io.state.lock().unwrap();
        assert_eq!(state.operations, ["background:A", "strict:B"]);
        assert_eq!(state.durable.as_deref(), Some("B"));
    }

    #[test]
    fn terminal_clear_rejects_publication_and_cannot_be_resurrected() {
        let (mut writer, io, started, release) = fixture(&["A"], &[]);
        let handle = writer.handle();
        publish(&handle, "A");
        assert_eq!(started.recv().unwrap(), "background:A");
        publish(&handle, "B");
        let cleared = writer.begin_clear().unwrap();
        assert!(handle
            .publish(vec![Message::user("C")], PhaseFourBreakerState::default(),)
            .is_err());
        release.send(()).unwrap();
        writer.wait_cleared(cleared).unwrap();

        let state = io.state.lock().unwrap();
        assert_eq!(state.operations, ["background:A", "clear"]);
        assert_eq!(state.durable, None);
    }

    #[test]
    fn later_strict_barrier_repairs_a_background_failure() {
        let (mut writer, io, started, _release) = fixture(&[], &["A"]);
        let handle = writer.handle();
        publish(&handle, "A");
        assert_eq!(started.recv().unwrap(), "background:A");
        handle
            .barrier(&[Message::user("B")], PhaseFourBreakerState::default())
            .unwrap();
        assert_eq!(started.recv().unwrap(), "strict:B");
        writer.preserve().unwrap();

        let state = io.state.lock().unwrap();
        assert_eq!(state.operations, ["background:A", "strict:B"]);
        assert_eq!(state.durable.as_deref(), Some("B"));
    }

    #[test]
    fn preserve_reports_an_unrepaired_background_failure() {
        let (mut writer, _io, started, _release) = fixture(&[], &["A"]);
        publish(&writer.handle(), "A");
        assert_eq!(started.recv().unwrap(), "background:A");
        let error = writer.preserve().unwrap_err();
        assert!(format!("{error:#}").contains("injected failure for A"));
    }
}
