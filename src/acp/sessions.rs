//! Persistent session ownership and prompt coordination for ACP handlers.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming ACP session handlers")
)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use agent_client_protocol::v1::schema::{SessionId, SessionUpdate};
use anyhow::{ensure, Result};
use tokio::sync::watch;

use crate::session::{RunLeaseError, Session, SessionRunLease, SessionStore};
use crate::tools::ToolRegistry;

use super::approval::{new_policy, ApprovalPolicy};
use super::history_replay::replay_updates;

/// A classified failure while acquiring one ACP prompt slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BeginPromptError {
    SessionNotFound,
    Loading,
    ActivePrompt,
}

impl std::fmt::Display for BeginPromptError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SessionNotFound => formatter.write_str("ACP session not found"),
            Self::Loading => formatter.write_str("ACP session is still loading"),
            Self::ActivePrompt => formatter.write_str("ACP session already has an active prompt"),
        }
    }
}

impl std::error::Error for BeginPromptError {}

/// A classified failure while preparing or activating an ACP session load.
#[derive(Debug)]
pub(crate) enum LoadError {
    InvalidSessionId,
    SessionNotFound,
    ActiveElsewhere,
    AlreadyRegistered,
    CwdMismatch,
    Internal(anyhow::Error),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSessionId => f.write_str("invalid ACP session ID"),
            Self::SessionNotFound => f.write_str("ACP session not found"),
            Self::ActiveElsewhere => f.write_str("ACP session is active in another process"),
            Self::AlreadyRegistered => f.write_str("ACP session is already registered"),
            Self::CwdMismatch => f.write_str("ACP load cwd does not match persisted session cwd"),
            Self::Internal(_) => f.write_str("internal ACP load error"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Internal(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}

/// In-process ACP sessions backed by the existing persistent session store.
pub(crate) struct SessionRegistry {
    store: SessionStore,
    sessions: Mutex<HashMap<SessionId, Arc<RegisteredSession>>>,
}

struct RegisteredSession {
    session: Session,
    _run_lease: SessionRunLease,
    ready: AtomicBool,
    approval_policy: ApprovalPolicy,
    tools: Arc<ToolRegistry>,
    active_prompt: Mutex<Option<watch::Sender<bool>>>,
}

/// A newly persisted session that is removed unless registered and committed.
pub(crate) struct PreparedCreate {
    store: SessionStore,
    session_id: String,
    session: Option<Session>,
    run_lease: Option<SessionRunLease>,
    committed: bool,
}

/// A validated load whose replay is complete but whose session is not exposed.
pub(crate) struct PreparedLoad {
    registry: Arc<SessionRegistry>,
    session: Session,
    run_lease: SessionRunLease,
    updates: Vec<SessionUpdate>,
}

/// A load registered as pending while its response is admitted.
pub(crate) struct PendingLoad {
    registry: Arc<SessionRegistry>,
    session_id: SessionId,
    registered: Arc<RegisteredSession>,
    committed: bool,
}

/// Exclusive permission to run one prompt for a registered session.
pub(crate) struct PromptGuard {
    registered: Arc<RegisteredSession>,
    cancellation: watch::Receiver<bool>,
}

/// Keeps a registered session and its run lease alive without owning its prompt slot.
pub(crate) struct SessionOwnership {
    _registered: Arc<RegisteredSession>,
}

impl SessionRegistry {
    /// Create an empty in-process registry over the persistent session store.
    pub(crate) fn new(store: SessionStore) -> Self {
        Self {
            store,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Clone the persistent store without retaining this registry.
    pub(crate) fn store(&self) -> SessionStore {
        self.store.clone()
    }

    /// Persist and lease a session that rolls back unless committed.
    pub(crate) fn prepare_create(
        store: SessionStore,
        cwd: &str,
        model: &str,
        agent: &str,
    ) -> Result<PreparedCreate> {
        let session = store.create(cwd, model, agent)?;
        let session_id = session.id().to_string();
        let mut prepared = PreparedCreate {
            store,
            session_id,
            session: Some(session),
            run_lease: None,
            committed: false,
        };
        prepared.run_lease = Some(prepared.store.try_acquire_run_lease(&prepared.session_id)?);
        Ok(prepared)
    }

    /// Register a prepared session and disable its persistence rollback.
    pub(crate) fn commit_create(
        &self,
        mut prepared: PreparedCreate,
        tools: Arc<ToolRegistry>,
    ) -> Result<SessionId> {
        let session = prepared.session.take().expect("prepared session missing");
        let run_lease = prepared
            .run_lease
            .take()
            .expect("prepared session run lease missing");
        let (session_id, _) = self.register(session, run_lease, true, tools)?;
        prepared.committed = true;
        Ok(session_id)
    }

    /// Create and exclusively own a persistent session until this registry drops.
    pub(crate) fn create(
        &self,
        cwd: &str,
        model: &str,
        agent: &str,
        tools: Arc<ToolRegistry>,
    ) -> Result<SessionId> {
        let prepared = Self::prepare_create(self.store.clone(), cwd, model, agent)?;
        self.commit_create(prepared, tools)
    }

    /// Validate and project a persisted session while holding its run lease.
    pub(crate) fn prepare_load(
        self: &Arc<Self>,
        session_id: &SessionId,
        expected_cwd: &str,
    ) -> std::result::Result<PreparedLoad, LoadError> {
        if lock(&self.sessions).contains_key(session_id) {
            return Err(LoadError::AlreadyRegistered);
        }
        let id = session_id.to_string();
        let run_lease = self
            .store
            .try_acquire_run_lease(&id)
            .map_err(|error| match error {
                RunLeaseError::InvalidSessionId(_) => LoadError::InvalidSessionId,
                RunLeaseError::SessionNotFound(_) => LoadError::SessionNotFound,
                RunLeaseError::ActiveElsewhere { .. } => LoadError::ActiveElsewhere,
                RunLeaseError::Internal(error) => LoadError::Internal(error),
            })?;
        let loaded = self.store.load(&id).map_err(|error| {
            if error.chain().any(|cause| {
                cause
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound)
            }) {
                LoadError::SessionNotFound
            } else {
                LoadError::Internal(error)
            }
        })?;
        if loaded.meta.id != id {
            return Err(LoadError::Internal(anyhow::anyhow!(
                "ACP load session id does not match persisted metadata"
            )));
        }
        if loaded.meta.cwd != expected_cwd {
            return Err(LoadError::CwdMismatch);
        }
        let updates = replay_updates(&loaded.records).map_err(LoadError::Internal)?;
        let session = self
            .store
            .reopen_loaded(&id, loaded)
            .map_err(LoadError::Internal)?;
        Ok(PreparedLoad {
            registry: Arc::clone(self),
            session,
            run_lease,
            updates,
        })
    }

    fn register(
        &self,
        session: Session,
        run_lease: SessionRunLease,
        ready: bool,
        tools: Arc<ToolRegistry>,
    ) -> Result<(SessionId, Arc<RegisteredSession>)> {
        let session_id = SessionId::new(session.id());
        let mut sessions = lock(&self.sessions);
        ensure!(
            !sessions.contains_key(&session_id),
            "ACP session {session_id} is already registered"
        );
        let registered = Arc::new(RegisteredSession {
            session,
            _run_lease: run_lease,
            ready: AtomicBool::new(ready),
            approval_policy: new_policy(),
            tools,
            active_prompt: Mutex::new(None),
        });
        sessions.insert(session_id.clone(), Arc::clone(&registered));
        Ok((session_id, registered))
    }

    /// Acquire the only active prompt slot for a session.
    pub(crate) fn begin_prompt(
        &self,
        session_id: &SessionId,
    ) -> std::result::Result<PromptGuard, BeginPromptError> {
        let sessions = lock(&self.sessions);
        let registered = sessions
            .get(session_id)
            .cloned()
            .ok_or(BeginPromptError::SessionNotFound)?;
        if !registered.ready.load(Ordering::Acquire) {
            return Err(BeginPromptError::Loading);
        }
        let (cancel, cancellation) = watch::channel(false);
        {
            let mut active_prompt = lock(&registered.active_prompt);
            if active_prompt.is_some() {
                return Err(BeginPromptError::ActivePrompt);
            }
            *active_prompt = Some(cancel);
        }
        drop(sessions);
        Ok(PromptGuard {
            registered,
            cancellation,
        })
    }

    /// Release an idle registered session and its run lease.
    pub(crate) fn close(&self, session_id: &SessionId) -> bool {
        let mut sessions = lock(&self.sessions);
        let Some(registered) = sessions.get(session_id) else {
            return false;
        };
        if !registered.ready.load(Ordering::Acquire) || lock(&registered.active_prompt).is_some() {
            return false;
        }
        sessions.remove(session_id);
        true
    }

    /// Request cancellation without marking the prompt idle before it exits.
    pub(crate) fn cancel(&self, session_id: &SessionId) -> bool {
        let registered = lock(&self.sessions).get(session_id).cloned();
        let Some(registered) = registered else {
            return false;
        };
        if !registered.ready.load(Ordering::Acquire) {
            return false;
        }
        let active_prompt = lock(&registered.active_prompt);
        let Some(cancel) = active_prompt.as_ref() else {
            return false;
        };
        cancel.send_replace(true);
        true
    }
}

impl Drop for PreparedCreate {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        self.run_lease.take();
        self.session.take();
        let _ = std::fs::remove_file(self.store.path(&self.session_id));
        let _ = std::fs::remove_file(
            self.store
                .root()
                .join(format!("{}.run.lock", self.session_id)),
        );
        if let Ok(archive_root) = self.store.archive_root(&self.session_id) {
            let _ = std::fs::remove_dir(archive_root);
        }
    }
}

impl PreparedLoad {
    /// Return the stable ACP session ID that will be registered on activation.
    pub(crate) fn session_id(&self) -> SessionId {
        SessionId::new(self.session.id())
    }

    /// Return the fully validated replay projection without exposing the session.
    pub(crate) fn updates(&self) -> &[SessionUpdate] {
        &self.updates
    }

    /// Register this load as pending while its protocol response is admitted.
    pub(crate) fn activate(self) -> std::result::Result<PendingLoad, LoadError> {
        let PreparedLoad {
            registry,
            session,
            run_lease,
            updates: _,
        } = self;
        let (session_id, registered) = registry
            .register(session, run_lease, false, Arc::new(ToolRegistry::builtin()))
            .map_err(|_| LoadError::AlreadyRegistered)?;
        Ok(PendingLoad {
            registry,
            session_id,
            registered,
            committed: false,
        })
    }
}

impl PendingLoad {
    /// Make the session prompt-visible after its load response was admitted.
    pub(crate) fn commit(mut self) {
        let sessions = lock(&self.registry.sessions);
        let owns_registration = sessions
            .get(&self.session_id)
            .is_some_and(|registered| Arc::ptr_eq(registered, &self.registered));
        assert!(
            owns_registration,
            "pending ACP load registration was replaced"
        );
        self.registered.ready.store(true, Ordering::Release);
        self.committed = true;
    }
}

impl Drop for PendingLoad {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let mut sessions = lock(&self.registry.sessions);
        let owns_registration = sessions
            .get(&self.session_id)
            .is_some_and(|registered| Arc::ptr_eq(registered, &self.registered));
        if owns_registration {
            sessions.remove(&self.session_id);
        }
    }
}

impl PromptGuard {
    /// Return the persistent session exclusively owned by this prompt slot.
    pub(crate) fn session(&self) -> &Session {
        &self.registered.session
    }

    /// Clone ownership that keeps the session run lease alive for blocking work.
    pub(crate) fn ownership(&self) -> Arc<dyn Send + Sync> {
        Arc::new(SessionOwnership {
            _registered: Arc::clone(&self.registered),
        })
    }

    pub(crate) fn approval_policy(&self) -> ApprovalPolicy {
        Arc::clone(&self.registered.approval_policy)
    }

    pub(crate) fn tools(&self) -> Arc<ToolRegistry> {
        Arc::clone(&self.registered.tools)
    }

    /// Wait until the matching session receives a cancellation request.
    pub(crate) async fn cancelled(&mut self) {
        let _ = self.cancellation.wait_for(|cancelled| *cancelled).await;
    }
}

impl Drop for PromptGuard {
    fn drop(&mut self) {
        lock(&self.registered.active_prompt).take();
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    use tempfile::TempDir;
    use tokio::sync::oneshot;

    use super::*;

    fn registry() -> (TempDir, SessionStore, SessionRegistry) {
        let directory = TempDir::new().unwrap();
        let store = SessionStore::with_root(directory.path().join("sessions"));
        let registry = SessionRegistry::new(store.clone());
        (directory, store, registry)
    }

    #[test]
    fn creates_a_persistent_session_with_the_same_acp_id() {
        let (_directory, store, registry) = registry();

        let session_id = registry
            .create(
                "/workspace",
                "model-a",
                "build",
                Arc::new(ToolRegistry::builtin()),
            )
            .unwrap();
        let id = session_id.to_string();
        let loaded = store.load(&id).unwrap();

        assert_eq!(loaded.meta.id, id);
        assert_eq!(loaded.meta.cwd, "/workspace");
        assert_eq!(loaded.meta.model, "model-a");
        assert_eq!(loaded.meta.agent, "build");
        let guard = registry.begin_prompt(&session_id).unwrap();
        assert_eq!(guard.session().id(), id);
    }

    #[test]
    fn abandoned_prepared_create_removes_persistence_and_lease_files() {
        let (_directory, store, _registry) = registry();
        let prepared =
            SessionRegistry::prepare_create(store.clone(), "/workspace", "model-a", "build")
                .unwrap();
        let id = prepared.session_id.clone();
        assert!(store.path(&id).exists());
        assert!(store.root().join(format!("{id}.run.lock")).exists());

        drop(prepared);

        assert!(!store.path(&id).exists());
        assert!(!store.root().join(format!("{id}.run.lock")).exists());
        assert!(store.list().unwrap().is_empty());
    }

    #[test]
    fn holds_the_cross_process_run_lease_for_the_registered_lifetime() {
        let (_directory, store, registry) = registry();
        let session_id = registry
            .create(
                "/workspace",
                "model-a",
                "build",
                Arc::new(ToolRegistry::builtin()),
            )
            .unwrap();
        let id = session_id.to_string();

        let error = store.try_acquire_run_lease(&id).unwrap_err();
        assert!(matches!(error, RunLeaseError::ActiveElsewhere { .. }));

        drop(registry);
        store.try_acquire_run_lease(&id).unwrap();
    }

    #[test]
    fn preparing_a_load_fails_closed_while_an_external_run_lease_exists() {
        let (_directory, store, registry) = registry();
        let registry = Arc::new(registry);
        let session = store.create("/workspace", "model-a", "build").unwrap();
        let session_id = SessionId::new(session.id());
        let external_lease = store.try_acquire_run_lease(session.id()).unwrap();

        let error = registry
            .prepare_load(&session_id, "/workspace")
            .err()
            .unwrap();
        assert!(matches!(error, LoadError::ActiveElsewhere));
        assert!(registry.begin_prompt(&session_id).is_err());

        drop(external_lease);
        let prepared = registry.prepare_load(&session_id, "/workspace").unwrap();
        let pending = prepared.activate().unwrap();
        let duplicate = registry
            .prepare_load(&session_id, "/workspace")
            .err()
            .unwrap();
        assert!(matches!(duplicate, LoadError::AlreadyRegistered));
        assert!(registry.begin_prompt(&session_id).is_err());
        pending.commit();
        registry.begin_prompt(&session_id).unwrap();
    }

    #[test]
    fn preparing_a_load_rejects_mismatched_metadata_id_without_state() {
        let (directory, store, registry) = registry();
        let registry = Arc::new(registry);
        let session = store.create("/workspace", "model-a", "build").unwrap();
        let id = session.id().to_string();
        let session_id = SessionId::new(id.as_str());
        let path = directory
            .path()
            .join("sessions")
            .join(format!("{id}.jsonl"));
        let mut meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        meta["meta"]["id"] = serde_json::json!("different-id");
        std::fs::write(
            &path,
            format!("{}\n", serde_json::to_string(&meta).unwrap()),
        )
        .unwrap();

        let error = registry
            .prepare_load(&session_id, "/workspace")
            .err()
            .unwrap();
        assert!(matches!(error, LoadError::Internal(_)));
        assert!(registry.begin_prompt(&session_id).is_err());
        store.try_acquire_run_lease(&id).unwrap();
    }

    #[test]
    fn close_refuses_an_active_prompt_then_releases_the_idle_run_lease() {
        let (_directory, store, registry) = registry();
        let tools = Arc::new(ToolRegistry::builtin());
        let session_id = registry
            .create("/workspace", "model-a", "build", Arc::clone(&tools))
            .unwrap();
        assert_eq!(Arc::strong_count(&tools), 2);
        let id = session_id.to_string();
        let guard = registry.begin_prompt(&session_id).unwrap();

        assert!(!registry.close(&session_id));
        assert_eq!(Arc::strong_count(&tools), 2);
        assert!(store.try_acquire_run_lease(&id).is_err());

        drop(guard);
        assert!(registry.close(&session_id));
        assert_eq!(Arc::strong_count(&tools), 1);
        store.try_acquire_run_lease(&id).unwrap();
        assert!(!registry.close(&session_id));
    }

    #[test]
    fn concurrent_prompt_admission_has_exactly_one_winner() {
        let (_directory, _store, registry) = registry();
        let registry = Arc::new(registry);
        let session_id = registry
            .create(
                "/workspace",
                "model-a",
                "build",
                Arc::new(ToolRegistry::builtin()),
            )
            .unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let mut attempts = Vec::new();
        for _ in 0..2 {
            let registry = Arc::clone(&registry);
            let session_id = session_id.clone();
            let barrier = Arc::clone(&barrier);
            attempts.push(std::thread::spawn(move || {
                barrier.wait();
                registry.begin_prompt(&session_id)
            }));
        }

        barrier.wait();
        let results: Vec<_> = attempts
            .into_iter()
            .map(|attempt| attempt.join().unwrap())
            .collect();

        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(results.iter().filter(|result| result.is_err()).count(), 1);
    }

    #[tokio::test]
    async fn cancellation_wakes_the_prompt_but_keeps_the_slot_busy_until_drop() {
        let (_directory, _store, registry) = registry();
        let session_id = registry
            .create(
                "/workspace",
                "model-a",
                "build",
                Arc::new(ToolRegistry::builtin()),
            )
            .unwrap();
        let mut guard = registry.begin_prompt(&session_id).unwrap();
        let (ready_tx, ready_rx) = oneshot::channel();
        let waiter = tokio::spawn(async move {
            ready_tx.send(()).unwrap();
            guard.cancelled().await;
            guard
        });

        ready_rx.await.unwrap();
        assert!(matches!(
            registry.begin_prompt(&session_id),
            Err(BeginPromptError::ActivePrompt)
        ));
        assert!(registry.cancel(&session_id));
        let guard = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("cancellation signal was not delivered")
            .unwrap();
        assert!(registry.begin_prompt(&session_id).is_err());

        drop(guard);
        let mut next = registry.begin_prompt(&session_id).unwrap();
        assert!(!*next.cancellation.borrow());
        assert!(registry.cancel(&session_id));
        tokio::time::timeout(Duration::from_secs(1), next.cancelled())
            .await
            .expect("new prompt cancellation signal was not delivered");
    }

    #[test]
    fn detached_prompt_ownership_keeps_run_lease_until_worker_finishes() {
        let (_directory, store, registry) = registry();
        let session_id = registry
            .create(
                "/workspace",
                "model-a",
                "build",
                Arc::new(ToolRegistry::builtin()),
            )
            .unwrap();
        let guard = registry.begin_prompt(&session_id).unwrap();
        let ownership = guard.ownership();

        drop(guard);
        drop(registry);
        assert!(store
            .try_acquire_run_lease(&session_id.to_string())
            .is_err());

        drop(ownership);
        store
            .try_acquire_run_lease(&session_id.to_string())
            .unwrap();
    }

    #[test]
    fn unknown_or_idle_sessions_are_not_reported_as_cancelled() {
        let (_directory, _store, registry) = registry();
        let unknown = SessionId::new("missing-session");
        assert!(matches!(
            registry.begin_prompt(&unknown),
            Err(BeginPromptError::SessionNotFound)
        ));
        assert!(!registry.cancel(&unknown));

        let session_id = registry
            .create(
                "/workspace",
                "model-a",
                "build",
                Arc::new(ToolRegistry::builtin()),
            )
            .unwrap();
        assert!(!registry.cancel(&session_id));
        let guard = registry.begin_prompt(&session_id).unwrap();
        drop(guard);
        assert!(!registry.cancel(&session_id));
    }
}
