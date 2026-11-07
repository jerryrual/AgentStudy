//! Persistent session ownership and prompt coordination for ACP handlers.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming ACP session handlers")
)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use agent_client_protocol::v1::schema::SessionId;
use anyhow::{ensure, Context, Result};
use tokio::sync::watch;

use crate::session::{Session, SessionRunLease, SessionStore};

/// In-process ACP sessions backed by the existing persistent session store.
pub(crate) struct SessionRegistry {
    store: SessionStore,
    sessions: Mutex<HashMap<SessionId, Arc<RegisteredSession>>>,
}

struct RegisteredSession {
    session: Session,
    _run_lease: SessionRunLease,
    active_prompt: Mutex<Option<watch::Sender<bool>>>,
}

/// Exclusive permission to run one prompt for a registered session.
pub(crate) struct PromptGuard {
    registered: Arc<RegisteredSession>,
    cancellation: watch::Receiver<bool>,
}

impl SessionRegistry {
    pub(crate) fn new(store: SessionStore) -> Self {
        Self {
            store,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Create and exclusively own a persistent session until this registry drops.
    pub(crate) fn create(&self, cwd: &str, model: &str, agent: &str) -> Result<SessionId> {
        let session = self.store.create(cwd, model, agent)?;
        let run_lease = self.store.try_acquire_run_lease(session.id())?;
        self.register(session, run_lease)
    }

    /// Open and exclusively own an existing persistent session.
    pub(crate) fn open(&self, session_id: &SessionId) -> Result<()> {
        ensure!(
            !lock(&self.sessions).contains_key(session_id),
            "ACP session {session_id} is already registered"
        );
        let id = session_id.to_string();
        let run_lease = self.store.try_acquire_run_lease(&id)?;
        let session = self.store.reopen(&id)?;
        self.register(session, run_lease)?;
        Ok(())
    }

    fn register(&self, session: Session, run_lease: SessionRunLease) -> Result<SessionId> {
        let session_id = SessionId::new(session.id());
        let mut sessions = lock(&self.sessions);
        ensure!(
            !sessions.contains_key(&session_id),
            "ACP session {session_id} is already registered"
        );
        sessions.insert(
            session_id.clone(),
            Arc::new(RegisteredSession {
                session,
                _run_lease: run_lease,
                active_prompt: Mutex::new(None),
            }),
        );
        Ok(session_id)
    }

    /// Acquire the only active prompt slot for a session.
    pub(crate) fn begin_prompt(&self, session_id: &SessionId) -> Result<PromptGuard> {
        let sessions = lock(&self.sessions);
        let registered = sessions
            .get(session_id)
            .cloned()
            .with_context(|| format!("unknown ACP session {session_id}"))?;
        let (cancel, cancellation) = watch::channel(false);
        {
            let mut active_prompt = lock(&registered.active_prompt);
            ensure!(
                active_prompt.is_none(),
                "ACP session {session_id} already has an active prompt"
            );
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
        if lock(&registered.active_prompt).is_some() {
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
        let active_prompt = lock(&registered.active_prompt);
        let Some(cancel) = active_prompt.as_ref() else {
            return false;
        };
        cancel.send_replace(true);
        true
    }
}

impl PromptGuard {
    pub(crate) fn session(&self) -> &Session {
        &self.registered.session
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

        let session_id = registry.create("/workspace", "model-a", "build").unwrap();
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
    fn holds_the_cross_process_run_lease_for_the_registered_lifetime() {
        let (_directory, store, registry) = registry();
        let session_id = registry.create("/workspace", "model-a", "build").unwrap();
        let id = session_id.to_string();

        let error = store.try_acquire_run_lease(&id).unwrap_err();
        assert!(error.to_string().contains("active in another process"));

        drop(registry);
        store.try_acquire_run_lease(&id).unwrap();
    }

    #[test]
    fn opening_a_session_fails_closed_while_an_external_run_lease_exists() {
        let (_directory, store, registry) = registry();
        let session = store.create("/workspace", "model-a", "build").unwrap();
        let session_id = SessionId::new(session.id());
        let external_lease = store.try_acquire_run_lease(session.id()).unwrap();

        let error = registry.open(&session_id).unwrap_err();
        assert!(error.to_string().contains("active in another process"));
        assert!(registry.begin_prompt(&session_id).is_err());

        drop(external_lease);
        registry.open(&session_id).unwrap();
        let duplicate = registry.open(&session_id).unwrap_err();
        assert!(duplicate.to_string().contains("already registered"));
        registry.begin_prompt(&session_id).unwrap();
    }

    #[test]
    fn close_refuses_an_active_prompt_then_releases_the_idle_run_lease() {
        let (_directory, store, registry) = registry();
        let session_id = registry.create("/workspace", "model-a", "build").unwrap();
        let id = session_id.to_string();
        let guard = registry.begin_prompt(&session_id).unwrap();

        assert!(!registry.close(&session_id));
        assert!(store.try_acquire_run_lease(&id).is_err());

        drop(guard);
        assert!(registry.close(&session_id));
        store.try_acquire_run_lease(&id).unwrap();
        assert!(!registry.close(&session_id));
    }

    #[test]
    fn concurrent_prompt_admission_has_exactly_one_winner() {
        let (_directory, _store, registry) = registry();
        let registry = Arc::new(registry);
        let session_id = registry.create("/workspace", "model-a", "build").unwrap();
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
        let session_id = registry.create("/workspace", "model-a", "build").unwrap();
        let mut guard = registry.begin_prompt(&session_id).unwrap();
        let (ready_tx, ready_rx) = oneshot::channel();
        let waiter = tokio::spawn(async move {
            ready_tx.send(()).unwrap();
            guard.cancelled().await;
            guard
        });

        ready_rx.await.unwrap();
        assert!(registry.begin_prompt(&session_id).is_err());
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
    fn unknown_or_idle_sessions_are_not_reported_as_cancelled() {
        let (_directory, _store, registry) = registry();
        let unknown = SessionId::new("missing-session");
        assert!(registry.begin_prompt(&unknown).is_err());
        assert!(!registry.cancel(&unknown));

        let session_id = registry.create("/workspace", "model-a", "build").unwrap();
        assert!(!registry.cancel(&session_id));
        let guard = registry.begin_prompt(&session_id).unwrap();
        drop(guard);
        assert!(!registry.cancel(&session_id));
    }
}
