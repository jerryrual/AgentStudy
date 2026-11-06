//! Append-only JSONL session transcripts.
//!
//! Every turn is written as it happens so a session can be inspected or
//! resumed after a crash, mirroring iCode's session history. The format is a
//! sequence of JSON lines; the first line is the session metadata.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};

use crate::compaction::{CompressionBatch, CompressionFold, ContextProjection};
use crate::compaction_archive::{
    ensure_private_directory, open_append_existing_nofollow, open_append_private_nofollow,
    open_read_nofollow, replace_private_file, sync_directory, ArchiveStore,
    ArchivedContextProjection,
};
use crate::model::{AssistantMessage, Message, ToolCall};
use crate::phase4_breaker::PhaseFourBreakerState;

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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compacted_messages: Option<usize>,
    },
    CompressionBatch {
        folds: Vec<CompressionFold>,
    },
    ContextProjection {
        projection: ContextProjection,
    },
    /// Legacy PR #25 checkpoint; new writes use the breaker sidecar.
    PhaseFourBreaker {
        turn: u64,
        state: serde_json::Value,
    },
}

impl Record {
    pub fn from_message(message: &Message) -> Option<Self> {
        match message {
            Message::System { .. } | Message::Summary { .. } | Message::LastWords { .. } => None,
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

const RECOVERY_CHECKPOINT_VERSION: u32 = 2;
const MAX_RECOVERY_CHECKPOINT_BYTES: u64 = 64 * 1024 * 1024;

fn ensure_recovery_checkpoint_size(size: u64) -> Result<()> {
    if size > MAX_RECOVERY_CHECKPOINT_BYTES {
        bail!(
            "recovery checkpoint is too large: {size} bytes exceeds {MAX_RECOVERY_CHECKPOINT_BYTES}"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RecoveryCheckpoint {
    version: u32,
    session_id: String,
    turn: u64,
    #[serde(default)]
    transcript_bytes: Option<u64>,
    messages: Vec<Message>,
    phase_four_breaker: PhaseFourBreakerState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhaseFourBreakerCheckpoint {
    turn: u64,
    state: serde_json::Value,
}

#[derive(Debug)]
pub(crate) struct SessionRunLease {
    _file: std::fs::File,
}

fn is_lock_contended(error: &std::io::Error) -> bool {
    let expected = fs2::lock_contended_error();
    error.kind() == expected.kind()
        && (expected.raw_os_error().is_none() || error.raw_os_error() == expected.raw_os_error())
}

#[derive(Debug, Clone)]
pub struct Session {
    pub meta: SessionMeta,
    path: PathBuf,
    archive_root: PathBuf,
    append_lock: Arc<Mutex<()>>,
}

struct ResumableTurnPlan {
    messages: Vec<Message>,
    turn: u64,
    breaker: PhaseFourBreakerState,
    records: Vec<Record>,
}

impl Session {
    /// Directory sibling of the JSONL transcript used for derived archive files.
    pub fn archive_root(&self) -> &Path {
        &self.archive_root
    }

    pub(crate) fn archive_catalog_pointer(&self) -> Result<Option<String>> {
        ArchiveStore::with_root(&self.archive_root).catalog_pointer_text()
    }

    fn lock_exclusive(&self) -> Result<std::fs::File> {
        let lock_path = self.path.with_extension("lock");
        let lock_file = open_append_private_nofollow(&lock_path)
            .with_context(|| format!("failed to open session lock {}", lock_path.display()))?;
        FileExt::lock_exclusive(&lock_file)
            .with_context(|| format!("failed to lock session {}", self.meta.id))?;
        Ok(lock_file)
    }

    pub(crate) fn phase_four_breaker_path(&self) -> PathBuf {
        self.archive_root.join("phase4-breaker.jsonl")
    }

    /// Append one record, flushing immediately.
    pub fn append(&self, record: &Record) -> Result<()> {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        self.append_unlocked(record)
    }

    fn append_unlocked(&self, record: &Record) -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)
            .with_context(|| format!("failed to open session {}", self.path.display()))?;
        let previous_len = file
            .metadata()
            .with_context(|| format!("failed to inspect session {}", self.path.display()))?
            .len();
        let mut bytes = serde_json::to_vec(record)?;
        bytes.push(b'\n');
        if let Err(error) = file.write_all(&bytes) {
            let write_error = anyhow::Error::new(error)
                .context(format!("failed to write session {}", self.path.display()));
            let rollback = std::fs::OpenOptions::new()
                .write(true)
                .open(&self.path)
                .and_then(|handle| handle.set_len(previous_len));
            if let Err(rollback_error) = rollback {
                return Err(write_error.context(format!(
                    "also failed to roll back session {}: {rollback_error}",
                    self.path.display()
                )));
            }
            return Err(write_error);
        }
        Ok(())
    }

    /// Append a compression batch as one atomic JSONL record.
    pub fn append_compression_batch(&self, batch: &CompressionBatch) -> Result<()> {
        self.append(&Record::CompressionBatch {
            folds: batch.folds.clone(),
        })
    }

    /// Append a batch only if its turn is still active after acquiring the append lock.
    pub(crate) fn append_compression_batch_if<F>(
        &self,
        batch: &CompressionBatch,
        should_append: F,
    ) -> Result<bool>
    where
        F: FnOnce() -> bool,
    {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        if !should_append() {
            return Ok(false);
        }
        self.append_unlocked(&Record::CompressionBatch {
            folds: batch.folds.clone(),
        })?;
        Ok(true)
    }

    /// Append historical folds and a current-turn replacement as one JSONL record.
    pub fn append_context_projection(&self, projection: &ContextProjection) -> Result<()> {
        if projection.is_empty() {
            anyhow::bail!("cannot append an empty context projection");
        }
        self.append_context_projection_if(projection, || true)
            .and_then(|appended| {
                appended
                    .then_some(())
                    .ok_or_else(|| anyhow::anyhow!("context projection append was declined"))
            })
    }

    /// Append a projection only if its owning turn remains active under the append lock.
    pub(crate) fn append_context_projection_if<F>(
        &self,
        projection: &ContextProjection,
        should_append: F,
    ) -> Result<bool>
    where
        F: FnOnce() -> bool,
    {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        if !should_append() {
            return Ok(false);
        }
        self.append_unlocked(&Record::ContextProjection {
            projection: projection.clone(),
        })?;
        Ok(true)
    }

    /// Archive the raw current-turn groups, then append their projection under one append lock.
    pub(crate) fn append_archived_context_projection_if<F>(
        &self,
        source: &[Message],
        projection: &ContextProjection,
        should_append: F,
    ) -> Result<Option<ArchivedContextProjection>>
    where
        F: FnOnce() -> bool,
    {
        if projection.is_empty() {
            anyhow::bail!("cannot append an empty context projection");
        }
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        if !should_append() {
            return Ok(None);
        }
        let loaded = load_session_file(&self.path, self.id())?;
        let current_user = loaded
            .records
            .iter()
            .rposition(|record| matches!(record, Record::User { .. }))
            .context("cannot archive a context projection without a persisted user turn")?;
        let turn = loaded.records[..=current_user]
            .iter()
            .filter(|record| matches!(record, Record::User { .. }))
            .count() as u64;
        let round = loaded.records[current_user + 1..]
            .iter()
            .filter(|record| {
                matches!(
                    record,
                    Record::ContextProjection {
                        projection: ContextProjection {
                            current_turn: Some(_),
                            ..
                        }
                    }
                )
            })
            .count() as u64
            + 1;
        let archived = ArchiveStore::with_root(&self.archive_root)
            .archive_context_projection_with_manifest(self.id(), turn, round, source, projection)?;
        self.append_unlocked(&Record::ContextProjection {
            projection: projection.clone(),
        })?;
        Ok(Some(archived))
    }

    /// Reload this session after all append work from cloned handles completes.
    pub fn load_messages(&self) -> Result<Vec<Message>> {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        Ok(load_session_file(&self.path, self.id())?.messages())
    }

    /// Append a conversation message, skipping system prompts.
    pub fn append_message(&self, message: &Message) -> Result<()> {
        if let Some(record) = Record::from_message(message) {
            self.append(&record)?;
        }
        Ok(())
    }

    pub(crate) fn append_user_and_load_phase_four_breaker(
        &self,
        text: &str,
    ) -> Result<(u64, PhaseFourBreakerState)> {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        self.append_unlocked(&Record::User {
            text: text.to_string(),
        })?;
        let turn = load_session_file(&self.path, &self.meta.id)?.user_turn_count();
        Ok((turn, PhaseFourBreakerState::default()))
    }

    fn append_records_unlocked(&self, records: &[Record]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let mut file = open_append_existing_nofollow(&self.path)
            .with_context(|| format!("failed to open session {}", self.path.display()))?;
        let previous_len = file
            .metadata()
            .with_context(|| format!("failed to inspect session {}", self.path.display()))?
            .len();
        let mut bytes = Vec::new();
        for record in records {
            serde_json::to_writer(&mut bytes, record)?;
            bytes.push(b'\n');
        }
        if let Err(error) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
            let write_error = anyhow::Error::new(error)
                .context(format!("failed to adopt recovery state for {}", self.id()));
            if let Err(rollback_error) = open_append_existing_nofollow(&self.path)
                .and_then(|handle| handle.set_len(previous_len))
                .and_then(|()| open_append_existing_nofollow(&self.path)?.sync_all())
            {
                return Err(write_error.context(format!(
                    "also failed to roll back session {}: {rollback_error}",
                    self.path.display()
                )));
            }
            return Err(write_error);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn persist_recovery_checkpoint(
        &self,
        messages: &[Message],
        turn: u64,
        phase_four_breaker: &PhaseFourBreakerState,
    ) -> Result<()> {
        self.persist_recovery_checkpoint_if(messages, turn, phase_four_breaker, || true)?
            .then_some(())
            .ok_or_else(|| anyhow::anyhow!("recovery checkpoint was declined"))
    }

    pub(crate) fn persist_recovery_checkpoint_candidate(
        &self,
        messages: &[Message],
        turn: u64,
        phase_four_breaker: &PhaseFourBreakerState,
    ) -> Result<bool> {
        self.persist_recovery_checkpoint_if(messages, turn, phase_four_breaker, || true)
    }

    pub(crate) fn persist_recovery_checkpoint_if<F>(
        &self,
        messages: &[Message],
        turn: u64,
        phase_four_breaker: &PhaseFourBreakerState,
        should_persist: F,
    ) -> Result<bool>
    where
        F: FnOnce() -> bool,
    {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        if !should_persist() {
            return Ok(false);
        }
        let loaded = load_session_file(&self.path, self.id())?;
        let current_turn = loaded.user_turn_count();
        if turn == 0 || turn != current_turn {
            bail!("recovery checkpoint turn {turn} does not match current turn {current_turn}");
        }
        let shaped_messages = shape_recovery_messages(messages)?;
        let recovery_messages = canonical_recovery_messages(&shaped_messages)?;
        if !recovery_messages.starts_with(&loaded.messages()) {
            return Ok(false);
        }
        let transcript = open_append_existing_nofollow(&self.path)
            .with_context(|| format!("failed to open session {}", self.path.display()))?;
        transcript
            .sync_all()
            .with_context(|| format!("failed to sync session {}", self.path.display()))?;
        let transcript_bytes = transcript
            .metadata()
            .with_context(|| format!("failed to inspect session {}", self.path.display()))?
            .len();
        sync_directory(self.path.parent().unwrap_or_else(|| Path::new(".")))
            .with_context(|| format!("failed to sync session directory for {}", self.id()))?;
        ensure_private_directory(&self.archive_root).with_context(|| {
            format!(
                "failed to prepare session archive {}",
                self.archive_root.display()
            )
        })?;
        let checkpoint = RecoveryCheckpoint {
            version: RECOVERY_CHECKPOINT_VERSION,
            session_id: self.id().to_string(),
            turn,
            transcript_bytes: Some(transcript_bytes),
            messages: messages.to_vec(),
            phase_four_breaker: phase_four_breaker.clone(),
        };
        let payload = serde_json::to_vec(&checkpoint)?;
        ensure_recovery_checkpoint_size(payload.len() as u64)?;
        replace_private_file(&self.recovery_checkpoint_path(), &payload).with_context(|| {
            format!(
                "failed to persist recovery checkpoint {}",
                self.recovery_checkpoint_path().display()
            )
        })?;
        match self.load_recovery_checkpoint_unlocked()? {
            Some(persisted) if persisted == checkpoint => Ok(true),
            _ => bail!("recovery checkpoint read-back did not match the written snapshot"),
        }
    }

    #[cfg(test)]
    pub(crate) fn load_recovery_checkpoint(&self) -> Result<Option<RecoveryCheckpoint>> {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        self.load_recovery_checkpoint_unlocked()
    }

    fn load_recovery_checkpoint_unlocked(&self) -> Result<Option<RecoveryCheckpoint>> {
        let checkpoint = self.read_recovery_checkpoint_unlocked()?;
        if let Some(checkpoint) = &checkpoint {
            let current_turn = load_session_file(&self.path, self.id())?.user_turn_count();
            if checkpoint.turn != current_turn {
                bail!(
                    "recovery checkpoint {} is stale",
                    self.recovery_checkpoint_path().display()
                );
            }
        }
        Ok(checkpoint)
    }

    fn read_recovery_checkpoint_unlocked(&self) -> Result<Option<RecoveryCheckpoint>> {
        let path = self.recovery_checkpoint_path();
        let file = match open_read_nofollow(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error).with_context(|| format!("failed to open {}", path.display()))
            }
        };
        ensure_recovery_checkpoint_size(file.metadata()?.len())?;
        let checkpoint: RecoveryCheckpoint = serde_json::from_reader(file)
            .with_context(|| format!("failed to parse recovery checkpoint {}", path.display()))?;
        let supported_version = match checkpoint.version {
            1 => checkpoint.transcript_bytes.is_none(),
            RECOVERY_CHECKPOINT_VERSION => checkpoint.transcript_bytes.is_some(),
            _ => false,
        };
        if !supported_version
            || checkpoint.session_id != self.id()
            || checkpoint.turn == 0
            || checkpoint.messages.is_empty()
            || PhaseFourBreakerState::from_value(&serde_json::to_value(
                &checkpoint.phase_four_breaker,
            )?)
            .is_none()
        {
            bail!("recovery checkpoint {} is invalid", path.display());
        }
        Ok(Some(checkpoint))
    }

    pub(crate) fn clear_recovery_checkpoint(&self) -> Result<()> {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        let path = self.recovery_checkpoint_path();
        match std::fs::symlink_metadata(&path) {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(error).with_context(|| format!("failed to inspect {}", path.display()))
            }
        }
        open_append_existing_nofollow(&self.path)
            .with_context(|| format!("failed to open session {}", self.path.display()))?
            .sync_all()
            .with_context(|| format!("failed to sync session {}", self.path.display()))?;
        sync_directory(self.path.parent().unwrap_or_else(|| Path::new(".")))
            .with_context(|| format!("failed to sync session directory for {}", self.id()))?;
        std::fs::remove_file(&path)
            .with_context(|| format!("failed to remove {}", path.display()))?;
        sync_directory(&self.archive_root)
    }

    fn recovery_checkpoint_path(&self) -> PathBuf {
        self.archive_root.join("session.recovery.json")
    }

    /// Check whether the current durable state can safely resume without adopting a sidecar.
    pub(crate) fn resumable_turn_available(&self) -> Result<bool> {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        let loaded = load_session_file(&self.path, &self.meta.id)?;
        Ok(self.prepare_resumable_turn_unlocked(loaded).is_ok())
    }

    /// Load the current resumable turn, atomically adopting a validated sidecar suffix if needed.
    pub(crate) fn load_resumable_turn(&self) -> Result<(Vec<Message>, u64, PhaseFourBreakerState)> {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        let loaded = load_session_file(&self.path, &self.meta.id)?;
        let plan = self.prepare_resumable_turn_unlocked(loaded)?;
        self.append_records_unlocked(&plan.records)?;
        Ok((plan.messages, plan.turn, plan.breaker))
    }

    fn prepare_resumable_turn_unlocked(&self, loaded: LoadedSession) -> Result<ResumableTurnPlan> {
        let transcript_bytes = open_read_nofollow(&self.path)
            .with_context(|| format!("failed to open session {}", self.path.display()))?
            .metadata()
            .with_context(|| format!("failed to inspect session {}", self.path.display()))?
            .len();
        if let Some(checkpoint) = self.read_recovery_checkpoint_unlocked()? {
            if checkpoint.turn == loaded.user_turn_count()
                && checkpoint.version == RECOVERY_CHECKPOINT_VERSION
                && checkpoint.transcript_bytes == Some(transcript_bytes)
            {
                return self.prepare_recovery_checkpoint_unlocked(loaded, checkpoint);
            }
        }
        let messages = loaded.resumable_turn_messages()?;
        let (turn, breaker) = self.load_phase_four_breaker_from(&loaded)?;
        Ok(ResumableTurnPlan {
            messages,
            turn,
            breaker,
            records: Vec::new(),
        })
    }

    fn prepare_recovery_checkpoint_unlocked(
        &self,
        mut loaded: LoadedSession,
        checkpoint: RecoveryCheckpoint,
    ) -> Result<ResumableTurnPlan> {
        let shaped_messages = shape_recovery_messages(&checkpoint.messages)?;
        let recovery_messages = canonical_recovery_messages(&shaped_messages)?;
        validate_recovery_provider_boundary(recovery_messages)?;
        let durable_messages = loaded.messages();
        let suffix = recovery_messages
            .strip_prefix(durable_messages.as_slice())
            .ok_or_else(|| {
                anyhow::anyhow!("recovery checkpoint does not extend the durable transcript")
            })?;
        let records = suffix
            .iter()
            .map(|message| match message {
                Message::Assistant { .. } | Message::Tool { .. } => {
                    Record::from_message(message).context("recovery suffix is not persistable")
                }
                _ => bail!("recovery checkpoint suffix crosses a user-turn boundary"),
            })
            .collect::<Result<Vec<_>>>()?;
        loaded.records.extend(records.iter().cloned());
        if loaded.messages() != recovery_messages {
            bail!("recovery checkpoint suffix does not reconstruct the saved model state");
        }
        let (turn, breaker) = self.load_phase_four_breaker_from(&loaded)?;
        Ok(ResumableTurnPlan {
            messages: shaped_messages,
            turn,
            breaker,
            records,
        })
    }

    pub(crate) fn load_phase_four_breaker(&self) -> Result<(u64, PhaseFourBreakerState)> {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        self.load_phase_four_breaker_unlocked()
    }

    fn load_phase_four_breaker_unlocked(&self) -> Result<(u64, PhaseFourBreakerState)> {
        let loaded = load_session_file(&self.path, &self.meta.id)?;
        self.load_phase_four_breaker_from(&loaded)
    }

    fn load_phase_four_breaker_from(
        &self,
        loaded: &LoadedSession,
    ) -> Result<(u64, PhaseFourBreakerState)> {
        let turn = loaded.user_turn_count();
        let legacy_state = loaded.records.iter().rev().find_map(|record| match record {
            Record::PhaseFourBreaker {
                turn: checkpoint_turn,
                state,
            } if *checkpoint_turn == turn => PhaseFourBreakerState::from_value(state),
            _ => None,
        });
        ensure_private_directory(&self.archive_root).with_context(|| {
            format!(
                "failed to prepare session archive {}",
                self.archive_root.display()
            )
        })?;
        let path = self.phase_four_breaker_path();
        let file = match open_read_nofollow(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok((turn, legacy_state.unwrap_or_default()));
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to open breaker state {}", path.display()));
            }
        };
        let mut checkpoints = Vec::new();
        for line in BufReader::new(file).lines() {
            let line =
                line.with_context(|| format!("failed to read breaker state {}", path.display()))?;
            match serde_json::from_str::<PhaseFourBreakerCheckpoint>(&line) {
                Ok(checkpoint) => checkpoints.push(checkpoint),
                Err(_) => {
                    serde_json::from_str::<serde_json::Value>(&line).with_context(|| {
                        format!("invalid breaker state JSON {}", path.display())
                    })?;
                }
            }
        }
        let state = checkpoints
            .into_iter()
            .rev()
            .find_map(|checkpoint| {
                (checkpoint.turn == turn)
                    .then(|| PhaseFourBreakerState::from_value(&checkpoint.state))
                    .flatten()
            })
            .unwrap_or_else(|| legacy_state.unwrap_or_default());
        Ok((turn, state))
    }

    pub(crate) fn enter_phase_four_round_if<F>(
        &self,
        turn: u64,
        side_call_limit: i64,
        should_append: F,
    ) -> Result<Option<(PhaseFourBreakerState, bool)>>
    where
        F: FnOnce() -> bool,
    {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        let (persisted_turn, mut committed) = self.load_phase_four_breaker_unlocked()?;
        if persisted_turn != turn || !should_append() {
            return Ok(None);
        }
        let admitted = committed.enter_round(side_call_limit);
        self.append_phase_four_breaker_unlocked(turn, &committed)?;
        Ok(Some((committed, admitted)))
    }

    pub(crate) fn record_phase_four_progress_if<F>(
        &self,
        turn: u64,
        meaningful: bool,
        should_append: F,
    ) -> Result<Option<PhaseFourBreakerState>>
    where
        F: FnOnce() -> bool,
    {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        let (persisted_turn, mut committed) = self.load_phase_four_breaker_unlocked()?;
        if persisted_turn != turn || !should_append() {
            return Ok(None);
        }
        committed.record_progress(meaningful);
        self.append_phase_four_breaker_unlocked(turn, &committed)?;
        Ok(Some(committed))
    }

    pub(crate) fn record_phase_four_failure_if<F>(
        &self,
        turn: u64,
        disable: bool,
        should_append: F,
    ) -> Result<Option<PhaseFourBreakerState>>
    where
        F: FnOnce() -> bool,
    {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        let (persisted_turn, mut committed) = self.load_phase_four_breaker_unlocked()?;
        if persisted_turn != turn || !should_append() {
            return Ok(None);
        }
        committed.record_failure(disable);
        self.append_phase_four_breaker_unlocked(turn, &committed)?;
        Ok(Some(committed))
    }

    pub(crate) fn append_phase_four_breaker_if<F>(
        &self,
        turn: u64,
        previous: &PhaseFourBreakerState,
        proposed: &PhaseFourBreakerState,
        should_append: F,
    ) -> Result<Option<PhaseFourBreakerState>>
    where
        F: FnOnce() -> bool,
    {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("session append lock is poisoned"))?;
        let _file_guard = self.lock_exclusive()?;
        let (persisted_turn, mut committed) = self.load_phase_four_breaker_unlocked()?;
        if persisted_turn != turn || !should_append() {
            return Ok(None);
        }
        let delta = proposed
            .side_call_tokens
            .checked_sub(previous.side_call_tokens)
            .context("Phase 4 breaker spend cannot decrease")?;
        let mut expected = previous.clone();
        expected.side_call_tokens = proposed.side_call_tokens;
        if expected != *proposed {
            bail!("side-call checkpoint cannot change another breaker field");
        }
        committed.side_call_tokens = committed
            .side_call_tokens
            .saturating_add(delta)
            .min(i64::MAX as u64);
        self.append_phase_four_breaker_unlocked(turn, &committed)?;
        Ok(Some(committed))
    }

    fn append_phase_four_breaker_unlocked(
        &self,
        turn: u64,
        state: &PhaseFourBreakerState,
    ) -> Result<()> {
        ensure_private_directory(&self.archive_root).with_context(|| {
            format!(
                "failed to create session archive {}",
                self.archive_root.display()
            )
        })?;
        let path = self.phase_four_breaker_path();
        let mut file = open_append_private_nofollow(&path)
            .with_context(|| format!("failed to open breaker state {}", path.display()))?;
        let previous_len = file.metadata()?.len();
        let checkpoint = PhaseFourBreakerCheckpoint {
            turn,
            state: serde_json::to_value(state)?,
        };
        let mut bytes = serde_json::to_vec(&checkpoint)?;
        bytes.push(b'\n');
        if let Err(error) = file.write_all(&bytes) {
            let write_error = anyhow::Error::new(error)
                .context(format!("failed to write breaker state {}", path.display()));
            if let Err(rollback_error) =
                open_append_private_nofollow(&path).and_then(|handle| handle.set_len(previous_len))
            {
                return Err(write_error.context(format!(
                    "also failed to roll back breaker state {}: {rollback_error}",
                    path.display()
                )));
            }
            return Err(write_error);
        }
        Ok(())
    }

    pub fn id(&self) -> &str {
        &self.meta.id
    }
}

fn shape_recovery_messages(messages: &[Message]) -> Result<Vec<Message>> {
    let mut shaped = Vec::with_capacity(messages.len());
    let mut index = 0;
    while index < messages.len() {
        let Message::Assistant { message: assistant } = &messages[index] else {
            if matches!(messages[index], Message::Tool { .. }) {
                bail!("recovery checkpoint has an orphan tool result");
            }
            shaped.push(messages[index].clone());
            index += 1;
            continue;
        };
        if assistant.tool_calls.is_empty() {
            shaped.push(messages[index].clone());
            index += 1;
            continue;
        }

        let mut calls = std::collections::BTreeMap::new();
        for call in &assistant.tool_calls {
            if calls.insert(call.id.as_str(), call).is_some() {
                bail!("recovery checkpoint has duplicate tool-call IDs");
            }
        }
        let mut results = std::collections::BTreeMap::new();
        let mut next = index + 1;
        while let Some(Message::Tool {
            call_id,
            name,
            content: _,
        }) = messages.get(next)
        {
            let Some(call) = calls.get(call_id.as_str()) else {
                bail!("recovery checkpoint has an orphan tool result");
            };
            if name != &call.name {
                bail!("recovery checkpoint tool result name does not match its call");
            }
            if results
                .insert(call_id.as_str(), messages[next].clone())
                .is_some()
            {
                bail!("recovery checkpoint has duplicate tool results");
            }
            next += 1;
        }

        let complete = results.len() == assistant.tool_calls.len();
        if complete {
            shaped.extend(messages[index..next].iter().cloned());
            index = next;
            continue;
        }
        if next != messages.len() {
            bail!("recovery checkpoint has a non-trailing incomplete tool-call group");
        }
        let mut retained_calls = Vec::new();
        let mut retained_results = Vec::new();
        for call in &assistant.tool_calls {
            if let Some(result) = results.remove(call.id.as_str()) {
                retained_calls.push(call.clone());
                retained_results.push(result);
            }
        }
        if retained_calls.is_empty() {
            bail!("recovery checkpoint has no completed tool results to retain");
        }
        let retained_responses_items = shape_partial_responses_items(
            &assistant.responses_items,
            &assistant.tool_calls,
            &retained_calls,
        )?;
        let mut retained_assistant = assistant.clone();
        retained_assistant.tool_calls = retained_calls;
        retained_assistant.responses_items = retained_responses_items;
        shaped.push(Message::assistant(retained_assistant));
        shaped.extend(retained_results);
        index = next;
    }
    Ok(shaped)
}

fn shape_partial_responses_items(
    items: &[serde_json::Value],
    calls: &[ToolCall],
    retained_calls: &[ToolCall],
) -> Result<Vec<serde_json::Value>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let call_ids = calls
        .iter()
        .map(|call| call.id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let retained_ids = retained_calls
        .iter()
        .map(|call| call.id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let mut seen = std::collections::BTreeSet::new();
    let mut shaped = Vec::with_capacity(items.len());
    for item in items {
        if item.get("type").and_then(serde_json::Value::as_str) != Some("function_call") {
            shaped.push(item.clone());
            continue;
        }
        let call_id = item
            .get("call_id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("native Responses function call has no call_id"))?;
        if !call_ids.contains(call_id) {
            bail!("native Responses function call does not match the assistant calls");
        }
        if !seen.insert(call_id) {
            bail!("native Responses output has duplicate function-call IDs");
        }
        if retained_ids.contains(call_id) {
            shaped.push(item.clone());
        }
    }
    if seen.len() != call_ids.len() {
        bail!("native Responses output is missing an assistant function call");
    }
    Ok(shaped)
}

fn canonical_recovery_messages(messages: &[Message]) -> Result<&[Message]> {
    let messages = match messages.split_first() {
        Some((Message::System { .. }, rest)) => rest,
        Some(_) => messages,
        None => bail!("recovery checkpoint has no messages"),
    };
    if messages
        .iter()
        .any(|message| matches!(message, Message::System { .. }))
    {
        bail!("recovery checkpoint has a misplaced system message");
    }
    Ok(messages)
}

fn validate_recovery_provider_boundary(messages: &[Message]) -> Result<()> {
    let mut pending = std::collections::BTreeSet::new();
    for message in messages {
        match message {
            Message::Assistant { message } if !message.tool_calls.is_empty() => {
                if !pending.is_empty() {
                    bail!("recovery checkpoint has overlapping tool-call groups");
                }
                for call in &message.tool_calls {
                    if !pending.insert(call.id.as_str()) {
                        bail!("recovery checkpoint has duplicate tool-call IDs");
                    }
                }
            }
            Message::Tool { call_id, .. } => {
                if !pending.remove(call_id.as_str()) {
                    bail!("recovery checkpoint has an orphan tool result");
                }
            }
            _ if !pending.is_empty() => {
                bail!("recovery checkpoint has an incomplete tool-call group");
            }
            _ => {}
        }
    }
    if !pending.is_empty() {
        bail!("recovery checkpoint has an incomplete tool-call group");
    }
    if matches!(
        messages.last(),
        Some(Message::Assistant { message }) if message.tool_calls.is_empty()
    ) {
        bail!("recovery checkpoint already contains a completed turn");
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct LoadedSession {
    pub meta: SessionMeta,
    pub records: Vec<Record>,
}

impl LoadedSession {
    fn user_turn_count(&self) -> u64 {
        self.records
            .iter()
            .filter(|record| matches!(record, Record::User { .. }))
            .count() as u64
    }

    pub(crate) fn resumable_turn_messages(&self) -> Result<Vec<Message>> {
        let Some(user) = self
            .records
            .iter()
            .rposition(|record| matches!(record, Record::User { .. }))
        else {
            anyhow::bail!("session has no user turn to resume");
        };
        let mut pending = std::collections::BTreeSet::new();
        let mut safe_checkpoint = false;
        for record in &self.records[user + 1..] {
            match record {
                Record::Assistant { tool_calls, .. } if tool_calls.is_empty() => {
                    anyhow::bail!("the latest user turn already completed");
                }
                Record::Assistant { tool_calls, .. } => {
                    if !pending.is_empty() {
                        anyhow::bail!("the latest user turn has an interrupted tool group");
                    }
                    pending.extend(tool_calls.iter().map(|call| call.id.clone()));
                }
                Record::Tool { call_id, .. } if pending.remove(call_id) => {
                    safe_checkpoint = pending.is_empty();
                }
                Record::ContextProjection { projection }
                    if projection.current_turn.is_some() && pending.is_empty() =>
                {
                    safe_checkpoint = true;
                }
                Record::User { .. } => unreachable!("latest User was selected"),
                _ => {}
            }
        }
        if !pending.is_empty() {
            anyhow::bail!("the latest user turn has an interrupted tool group");
        }
        if !safe_checkpoint {
            anyhow::bail!(
                "the latest user turn has no complete tool or Last Words checkpoint to resume"
            );
        }
        Ok(self.messages())
    }

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
                Record::Compaction {
                    summary,
                    compacted_messages,
                } => match compacted_messages {
                    None => {
                        pending.clear();
                        messages.clear();
                        messages.push(crate::compaction::summary_message(summary));
                    }
                    Some(count) => {
                        flush(&mut messages, &mut pending);
                        apply_compression_fold(&mut messages, summary, *count);
                    }
                },
                Record::CompressionBatch { folds } => {
                    flush(&mut messages, &mut pending);
                    if let Ok(projected) = CompressionBatch::project_folds(folds, &messages) {
                        messages = projected;
                    }
                }
                Record::ContextProjection { projection } => {
                    let mut candidate_messages = messages.clone();
                    let mut candidate_pending = pending.clone();
                    flush(&mut candidate_messages, &mut candidate_pending);
                    if let Ok(projected) = projection.projected(&candidate_messages) {
                        messages = projected;
                        pending = candidate_pending;
                    }
                }
                Record::Meta { .. } | Record::Usage { .. } | Record::PhaseFourBreaker { .. } => {}
            }
        }
        flush(&mut messages, &mut pending);
        messages
    }
}

fn apply_compression_fold(messages: &mut Vec<Message>, summary: &str, count: usize) {
    let summary_end = messages
        .iter()
        .rposition(|message| matches!(message, Message::Summary { .. }))
        .map_or(0, |position| position + 1);
    let Some(range_start) =
        messages
            .iter()
            .enumerate()
            .skip(summary_end)
            .find_map(|(position, message)| {
                matches!(message, Message::User { .. }).then_some(position)
            })
    else {
        return;
    };
    let available = messages.len() - range_start;
    // A corrupt persisted count must not discard an unknown tail.
    if count == 0 || count > available {
        return;
    }
    messages.splice(
        range_start..range_start + count,
        std::iter::once(crate::compaction::summary_message(summary)),
    );
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

    pub(crate) fn try_acquire_run_lease(&self, id: &str) -> Result<SessionRunLease> {
        validate_session_id(id)?;
        let transcript = self.path(id);
        open_read_nofollow(&transcript).with_context(|| format!("failed to open session {id}"))?;
        let path = self.root.join(format!("{id}.run.lock"));
        let file = open_append_private_nofollow(&path)
            .with_context(|| format!("failed to open session run lease {}", path.display()))?;
        match FileExt::try_lock_exclusive(&file) {
            Ok(()) => Ok(SessionRunLease { _file: file }),
            Err(error) if is_lock_contended(&error) => {
                anyhow::bail!("session {id} is active in another process")
            }
            Err(error) => Err(error)
                .with_context(|| format!("failed to acquire session run lease {}", path.display())),
        }
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
            archive_root: self.archive_root(&id)?,
            append_lock: Arc::new(Mutex::new(())),
        };
        session.append(&Record::Meta { meta })?;
        Ok(session)
    }

    pub fn path(&self, id: &str) -> PathBuf {
        self.root.join(format!("{id}.jsonl"))
    }

    /// Resolve a validated session ID to its sibling archive directory.
    pub fn archive_root(&self, id: &str) -> Result<PathBuf> {
        validate_session_id(id)?;
        Ok(self.root.join(id))
    }

    /// Open an existing session for appending.
    pub fn reopen(&self, id: &str) -> Result<Session> {
        let loaded = self.load(id)?;
        let archive_root = self.archive_root(id)?;
        let session = Session {
            meta: loaded.meta,
            path: self.path(id),
            archive_root,
            append_lock: Arc::new(Mutex::new(())),
        };
        let _file_guard = session.lock_exclusive()?;
        ArchiveStore::with_root(&session.archive_root)
            .reconcile()
            .with_context(|| format!("failed to reconcile archive for session {id}"))?;
        drop(_file_guard);
        Ok(session)
    }

    /// Load a session by id.
    pub fn load(&self, id: &str) -> Result<LoadedSession> {
        validate_session_id(id)?;
        load_session_file(&self.path(id), id)
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

fn load_session_file(path: &Path, id: &str) -> Result<LoadedSession> {
    let file =
        std::fs::File::open(path).with_context(|| format!("session {id:?} was not found"))?;
    let reader = BufReader::new(file);
    let mut meta: Option<SessionMeta> = None;
    let mut records = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read session {id:?}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(&line)
            .with_context(|| format!("invalid record on line {} of session {id:?}", index + 1))?;
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

fn preview_text(text: &str) -> String {
    let mut preview = text.lines().next().unwrap_or_default().trim().to_string();
    if preview.chars().count() > 80 {
        preview = preview.chars().take(80).collect::<String>() + "…";
    }
    preview
}

fn validate_session_id(id: &str) -> Result<()> {
    if id.is_empty()
        || !id.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == '-' || character == '_'
        })
    {
        bail!("invalid session id {id:?}");
    }
    Ok(())
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

    fn loaded(records: Vec<Record>) -> LoadedSession {
        LoadedSession {
            meta: SessionMeta {
                id: "test".into(),
                created_ms: 0,
                cwd: "/tmp".into(),
                model: "model".into(),
                agent: "code".into(),
            },
            records,
        }
    }

    fn assistant_record(content: &str) -> Record {
        Record::Assistant {
            content: Some(content.into()),
            reasoning: None,
            tool_calls: Vec::new(),
            responses_items: Vec::new(),
        }
    }

    #[test]
    fn recovery_checkpoint_size_is_bounded_before_io() {
        assert!(ensure_recovery_checkpoint_size(MAX_RECOVERY_CHECKPOINT_BYTES).is_ok());
        assert!(ensure_recovery_checkpoint_size(MAX_RECOVERY_CHECKPOINT_BYTES + 1).is_err());
    }

    #[test]
    fn recovery_checkpoint_round_trips_replaces_and_clears() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session
            .append_message(&Message::user("continue work"))
            .unwrap();
        let first_messages = vec![Message::user("continue work")];
        let mut first_breaker = PhaseFourBreakerState::default();
        first_breaker.attempts = 1;
        assert!(!session
            .persist_recovery_checkpoint_if(&first_messages, 1, &first_breaker, || false)
            .unwrap());
        assert!(!session.recovery_checkpoint_path().exists());
        session
            .persist_recovery_checkpoint(&first_messages, 1, &first_breaker)
            .unwrap();
        let first = session.load_recovery_checkpoint().unwrap().unwrap();
        assert_eq!(first.messages, first_messages);
        assert_eq!(first.phase_four_breaker, first_breaker);

        let second_messages = vec![
            Message::user("continue work"),
            Message::assistant(AssistantMessage {
                content: Some("newer snapshot".into()),
                reasoning: None,
                tool_calls: Vec::new(),
                responses_items: Vec::new(),
            }),
        ];
        let mut second_breaker = first_breaker.clone();
        second_breaker.attempts = 2;
        session
            .persist_recovery_checkpoint(&second_messages, 1, &second_breaker)
            .unwrap();
        let second = session.load_recovery_checkpoint().unwrap().unwrap();
        assert_eq!(second.messages, second_messages);
        assert_eq!(second.phase_four_breaker, second_breaker);
        assert!(!session
            .recovery_checkpoint_path()
            .with_extension("tmp")
            .exists());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(session.recovery_checkpoint_path())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }

        session.clear_recovery_checkpoint().unwrap();
        assert!(session.load_recovery_checkpoint().unwrap().is_none());
    }

    #[test]
    fn delayed_partial_checkpoint_cannot_rebase_onto_newer_transcript() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        let user = Message::user("inspect");
        session.append_message(&user).unwrap();
        session
            .persist_recovery_checkpoint(std::slice::from_ref(&user), 1, &Default::default())
            .unwrap();
        let original_sidecar = std::fs::read(session.recovery_checkpoint_path()).unwrap();
        let assistant = Message::assistant(AssistantMessage {
            tool_calls: vec![
                ToolCall {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    arguments: "{}".into(),
                },
                ToolCall {
                    id: "call_2".into(),
                    name: "read_file".into(),
                    arguments: "{}".into(),
                },
            ],
            ..AssistantMessage::default()
        });
        let first = Message::tool("call_1", "read_file", "first");
        let second = Message::tool("call_2", "read_file", "second");
        let mut buffer = RecordBuffer::default();
        buffer.record(&session, &assistant).unwrap();
        buffer.record(&session, &first).unwrap();
        buffer.record(&session, &second).unwrap();

        assert!(!session
            .persist_recovery_checkpoint_candidate(
                &[user, assistant, first],
                1,
                &Default::default(),
            )
            .unwrap());
        assert_eq!(
            std::fs::read(session.recovery_checkpoint_path()).unwrap(),
            original_sidecar
        );
    }

    #[test]
    fn recovery_checkpoint_equal_to_its_transcript_baseline_can_resume() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session
            .append_message(&Message::user("retry request"))
            .unwrap();
        let (breaker, admitted) = session
            .enter_phase_four_round_if(1, -1, || true)
            .unwrap()
            .unwrap();
        assert!(admitted);
        let checkpoint_messages = vec![
            Message::system("captured system"),
            Message::user("retry request"),
        ];
        session
            .persist_recovery_checkpoint(&checkpoint_messages, 1, &breaker)
            .unwrap();
        let before = std::fs::read(&session.path).unwrap();
        let sidecar_before = std::fs::read(session.recovery_checkpoint_path()).unwrap();

        assert!(session.resumable_turn_available().unwrap());
        assert_eq!(std::fs::read(&session.path).unwrap(), before);
        assert_eq!(
            std::fs::read(session.recovery_checkpoint_path()).unwrap(),
            sidecar_before
        );

        let (messages, turn, restored_breaker) = session.load_resumable_turn().unwrap();

        assert_eq!(messages, checkpoint_messages);
        assert_eq!(turn, 1);
        assert_eq!(restored_breaker, breaker);
        assert_eq!(std::fs::read(&session.path).unwrap(), before);
    }

    #[test]
    fn resumable_probe_tracks_durable_turn_boundaries_without_mutation() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("inspect")).unwrap();
        assert!(!session.resumable_turn_available().unwrap());

        let assistant = Message::assistant(AssistantMessage {
            content: None,
            reasoning: None,
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"README.md"}"#.into(),
            }],
            responses_items: Vec::new(),
        });
        let mut buffer = RecordBuffer::default();
        buffer.record(&session, &assistant).unwrap();
        buffer
            .record(&session, &Message::tool("call_1", "read_file", "contents"))
            .unwrap();
        let resumable_bytes = std::fs::read(&session.path).unwrap();

        assert!(session.resumable_turn_available().unwrap());
        assert_eq!(std::fs::read(&session.path).unwrap(), resumable_bytes);

        buffer
            .record(
                &session,
                &Message::assistant(AssistantMessage::text("done")),
            )
            .unwrap();
        let completed_bytes = std::fs::read(&session.path).unwrap();
        assert!(!session.resumable_turn_available().unwrap());
        assert_eq!(std::fs::read(&session.path).unwrap(), completed_bytes);
    }

    #[test]
    fn recovery_checkpoint_atomically_adopts_a_complete_tool_suffix() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("inspect")).unwrap();
        let assistant = Message::assistant(AssistantMessage {
            content: None,
            reasoning: None,
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"README.md"}"#.into(),
            }],
            responses_items: Vec::new(),
        });
        let tool = Message::tool("call_1", "read_file", "contents");
        let checkpoint_messages = vec![
            Message::system("captured system"),
            Message::user("inspect"),
            assistant.clone(),
            tool.clone(),
        ];
        session
            .persist_recovery_checkpoint(&checkpoint_messages, 1, &PhaseFourBreakerState::default())
            .unwrap();

        let (messages, turn, _) = session.load_resumable_turn().unwrap();
        assert_eq!(messages, checkpoint_messages);
        assert_eq!(turn, 1);
        assert_eq!(
            store.load(session.id()).unwrap().messages(),
            vec![Message::user("inspect"), assistant, tool,]
        );
        assert_eq!(
            session.load_resumable_turn().unwrap().0,
            vec![
                Message::user("inspect"),
                checkpoint_messages[2].clone(),
                checkpoint_messages[3].clone(),
            ]
        );
    }

    #[test]
    fn recovery_checkpoint_preserves_a_complete_durable_group_result_order() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        let user = Message::user("inspect");
        let assistant = Message::assistant(AssistantMessage {
            content: None,
            reasoning: None,
            tool_calls: vec![
                ToolCall {
                    id: "call_a".into(),
                    name: "read_file".into(),
                    arguments: "{}".into(),
                },
                ToolCall {
                    id: "call_c".into(),
                    name: "grep".into(),
                    arguments: "{}".into(),
                },
            ],
            responses_items: Vec::new(),
        });
        let tool_c = Message::tool("call_c", "grep", "C result");
        let tool_a = Message::tool("call_a", "read_file", "A result");
        for message in [&user, &assistant, &tool_c, &tool_a] {
            session.append_message(message).unwrap();
        }
        let checkpoint = vec![
            Message::system("captured system"),
            user,
            assistant,
            tool_c,
            tool_a,
        ];
        session
            .persist_recovery_checkpoint(&checkpoint, 1, &Default::default())
            .unwrap();
        let before = std::fs::read(&session.path).unwrap();

        assert_eq!(session.load_resumable_turn().unwrap().0, checkpoint);
        assert_eq!(std::fs::read(&session.path).unwrap(), before);
    }

    #[test]
    fn recovery_checkpoint_keeps_only_completed_calls_from_a_partial_tail() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session
            .append_message(&Message::user("inspect three"))
            .unwrap();
        let calls = [
            ToolCall {
                id: "call_a".into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            },
            ToolCall {
                id: "call_b".into(),
                name: "write_file".into(),
                arguments: "{}".into(),
            },
            ToolCall {
                id: "call_c".into(),
                name: "grep".into(),
                arguments: "{}".into(),
            },
        ];
        let assistant = Message::assistant(AssistantMessage {
            content: Some("I checked the available results.".into()),
            reasoning: None,
            tool_calls: calls.to_vec(),
            responses_items: Vec::new(),
        });
        let checkpoint = vec![
            Message::system("captured system"),
            Message::user("inspect three"),
            assistant,
            Message::tool("call_c", "grep", "C result"),
            Message::tool("call_a", "read_file", "A result"),
        ];
        session
            .persist_recovery_checkpoint(&checkpoint, 1, &Default::default())
            .unwrap();

        let (messages, turn, _) = session.load_resumable_turn().unwrap();

        assert_eq!(turn, 1);
        let Message::Assistant { message } = &messages[2] else {
            panic!("expected the shaped assistant tool group");
        };
        assert_eq!(
            message.content.as_deref(),
            Some("I checked the available results.")
        );
        assert_eq!(
            message
                .tool_calls
                .iter()
                .map(|call| call.id.as_str())
                .collect::<Vec<_>>(),
            ["call_a", "call_c"]
        );
        assert!(
            matches!(&messages[3], Message::Tool { call_id, content, .. }
            if call_id == "call_a" && content == "A result")
        );
        assert!(
            matches!(&messages[4], Message::Tool { call_id, content, .. }
            if call_id == "call_c" && content == "C result")
        );
        assert_eq!(store.load(session.id()).unwrap().messages(), messages[1..]);
        assert_eq!(session.load_resumable_turn().unwrap().0, messages[1..]);
    }

    #[test]
    fn partial_recovery_without_completed_results_or_with_malformed_native_items_fails_closed() {
        let assistant = AssistantMessage {
            content: Some("visible text".into()),
            reasoning: None,
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            }],
            responses_items: Vec::new(),
        };
        assert!(shape_recovery_messages(&[
            Message::user("inspect"),
            Message::assistant(assistant.clone()),
        ])
        .is_err());

        let mut native = assistant;
        native.tool_calls.push(ToolCall {
            id: "call_2".into(),
            name: "grep".into(),
            arguments: "{}".into(),
        });
        native
            .responses_items
            .push(serde_json::json!({"type": "function_call"}));
        assert!(shape_recovery_messages(&[
            Message::user("inspect"),
            Message::assistant(native),
            Message::tool("call_1", "read_file", "result"),
        ])
        .is_err());
    }

    #[test]
    fn partial_recovery_filters_native_responses_calls_and_preserves_other_items() {
        let messages = shape_recovery_messages(&[
            Message::user("inspect"),
            Message::assistant(AssistantMessage {
                content: Some("visible text".into()),
                reasoning: None,
                tool_calls: vec![
                    ToolCall {
                        id: "call_a".into(),
                        name: "read_file".into(),
                        arguments: "{}".into(),
                    },
                    ToolCall {
                        id: "call_b".into(),
                        name: "grep".into(),
                        arguments: "{}".into(),
                    },
                ],
                responses_items: vec![
                    serde_json::json!({"type": "reasoning", "id": "rs_1"}),
                    serde_json::json!({"type": "function_call", "call_id": "call_a"}),
                    serde_json::json!({"type": "function_call", "call_id": "call_b"}),
                    serde_json::json!({"type": "message", "id": "msg_1"}),
                ],
            }),
            Message::tool("call_b", "grep", "B result"),
        ])
        .unwrap();

        let Message::Assistant { message } = &messages[1] else {
            panic!("expected a shaped native assistant message");
        };
        assert_eq!(message.tool_calls.len(), 1);
        assert_eq!(message.tool_calls[0].id, "call_b");
        assert_eq!(
            message
                .responses_items
                .iter()
                .map(|item| item["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["reasoning", "function_call", "message"]
        );
        assert_eq!(message.responses_items[1]["call_id"], "call_b");
    }

    #[test]
    fn partial_recovery_rejects_orphan_duplicate_and_mismatched_results() {
        let assistant = Message::assistant(AssistantMessage {
            content: None,
            reasoning: None,
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            }],
            responses_items: Vec::new(),
        });
        for messages in [
            vec![Message::tool("orphan", "read_file", "result")],
            vec![
                assistant.clone(),
                Message::tool("call_1", "read_file", "first"),
                Message::tool("call_1", "read_file", "duplicate"),
            ],
            vec![
                assistant.clone(),
                Message::tool("call_1", "grep", "wrong tool"),
            ],
        ] {
            assert!(shape_recovery_messages(&messages).is_err());
        }
    }

    #[test]
    fn legacy_or_divergent_recovery_checkpoint_is_not_adopted() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("original")).unwrap();
        session
            .persist_recovery_checkpoint(
                &[Message::system("system"), Message::user("original")],
                1,
                &PhaseFourBreakerState::default(),
            )
            .unwrap();
        let path = session.recovery_checkpoint_path();
        let mut checkpoint: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        checkpoint["messages"][1]["content"] = serde_json::json!("different");
        replace_private_file(&path, &serde_json::to_vec(&checkpoint).unwrap()).unwrap();
        assert!(session.load_resumable_turn().is_err());

        checkpoint["messages"][1]["content"] = serde_json::json!("original");
        checkpoint["messages"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::to_value(Message::user("injected turn")).unwrap());
        replace_private_file(&path, &serde_json::to_vec(&checkpoint).unwrap()).unwrap();
        assert!(session.load_resumable_turn().is_err());
        checkpoint["messages"].as_array_mut().unwrap().pop();

        checkpoint["version"] = serde_json::json!(1);
        checkpoint
            .as_object_mut()
            .unwrap()
            .remove("transcript_bytes");
        checkpoint["messages"][1]["content"] = serde_json::json!("original");
        replace_private_file(&path, &serde_json::to_vec(&checkpoint).unwrap()).unwrap();
        assert!(session.load_resumable_turn().is_err());
        assert_eq!(
            store.load(session.id()).unwrap().messages(),
            vec![Message::user("original")]
        );
    }

    #[test]
    fn recovery_checkpoint_rejects_stale_and_malformed_state() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("first")).unwrap();
        session
            .persist_recovery_checkpoint(
                &[Message::user("first")],
                1,
                &PhaseFourBreakerState::default(),
            )
            .unwrap();
        session.append_message(&Message::user("second")).unwrap();
        assert!(session.load_recovery_checkpoint().is_err());
        let assistant = Message::assistant(AssistantMessage {
            content: None,
            reasoning: None,
            tool_calls: vec![ToolCall {
                id: "call_2".into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            }],
            responses_items: Vec::new(),
        });
        session.append_message(&assistant).unwrap();
        session
            .append_message(&Message::tool("call_2", "read_file", "done"))
            .unwrap();
        assert_eq!(session.load_resumable_turn().unwrap().1, 2);

        replace_private_file(&session.recovery_checkpoint_path(), b"{truncated").unwrap();
        assert!(session.load_recovery_checkpoint().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn recovery_checkpoint_does_not_follow_a_destination_symlink() {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("work")).unwrap();
        ensure_private_directory(&session.archive_root).unwrap();
        let outside = directory.path().join("outside");
        std::fs::write(&outside, "sentinel").unwrap();
        symlink(&outside, session.recovery_checkpoint_path()).unwrap();

        session
            .persist_recovery_checkpoint(
                &[Message::user("work")],
                1,
                &PhaseFourBreakerState::default(),
            )
            .unwrap();

        assert_eq!(std::fs::read_to_string(&outside).unwrap(), "sentinel");
        assert!(
            !std::fs::symlink_metadata(session.recovery_checkpoint_path())
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }

    #[cfg(unix)]
    #[test]
    fn recovery_cleanup_preserves_sidecar_when_transcript_sync_fails() {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("work")).unwrap();
        session
            .persist_recovery_checkpoint(
                &[Message::user("work")],
                1,
                &PhaseFourBreakerState::default(),
            )
            .unwrap();
        let original = session.path.with_extension("saved");
        std::fs::rename(&session.path, &original).unwrap();
        let outside = directory.path().join("outside-transcript");
        std::fs::write(&outside, "sentinel").unwrap();
        symlink(&outside, &session.path).unwrap();

        assert!(session.clear_recovery_checkpoint().is_err());
        assert!(session.recovery_checkpoint_path().exists());
        assert_eq!(std::fs::read_to_string(outside).unwrap(), "sentinel");
    }

    #[test]
    fn resumable_turn_requires_a_complete_durable_checkpoint() {
        let user = Record::User {
            text: "continue work".into(),
        };
        assert!(loaded(vec![user.clone()])
            .resumable_turn_messages()
            .is_err());

        let call = Record::Assistant {
            content: None,
            reasoning: None,
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            }],
            responses_items: Vec::new(),
        };
        assert!(loaded(vec![user.clone(), call.clone()])
            .resumable_turn_messages()
            .unwrap_err()
            .to_string()
            .contains("interrupted tool group"));

        let tool = Record::Tool {
            call_id: "call_1".into(),
            name: "read_file".into(),
            content: "result".into(),
        };
        let messages = loaded(vec![user.clone(), call.clone(), tool.clone()])
            .resumable_turn_messages()
            .unwrap();
        assert!(matches!(messages.last(), Some(Message::Tool { .. })));

        assert!(loaded(vec![user, call, tool, assistant_record("done")])
            .resumable_turn_messages()
            .unwrap_err()
            .to_string()
            .contains("already completed"));
    }

    #[test]
    fn resumable_turn_restores_breaker_state_without_a_new_user() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session
            .append_message(&Message::user("continue work"))
            .unwrap();
        session
            .append(&Record::Assistant {
                content: None,
                reasoning: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    arguments: "{}".into(),
                }],
                responses_items: Vec::new(),
            })
            .unwrap();
        session
            .append(&Record::Tool {
                call_id: "call_1".into(),
                name: "read_file".into(),
                content: "result".into(),
            })
            .unwrap();
        session
            .enter_phase_four_round_if(1, -1, || true)
            .unwrap()
            .unwrap();

        let (messages, turn, breaker) = session.load_resumable_turn().unwrap();
        assert_eq!(turn, 1);
        assert_eq!(breaker.attempts, 1);
        assert!(matches!(messages.last(), Some(Message::Tool { .. })));
    }

    #[test]
    fn run_lease_excludes_headless_owners_and_releases_on_drop() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        let other_store = SessionStore::with_root(directory.path());

        let lease = store.try_acquire_run_lease(session.id()).unwrap();
        let error = other_store.try_acquire_run_lease(session.id()).unwrap_err();
        assert!(error.to_string().contains("active in another process"));
        session
            .append_message(&Message::user("still writable"))
            .unwrap();

        drop(lease);
        let reacquired = other_store.try_acquire_run_lease(session.id()).unwrap();
        drop(reacquired);
        assert!(directory
            .path()
            .join(format!("{}.run.lock", session.id()))
            .exists());
    }

    #[test]
    fn run_leases_are_independent_per_session_and_validate_ids() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let first = store.create("/tmp", "model", "code").unwrap();
        let second = store.create("/tmp", "model", "code").unwrap();

        let _first_lease = store.try_acquire_run_lease(first.id()).unwrap();
        let _second_lease = store.try_acquire_run_lease(second.id()).unwrap();
        assert!(store.try_acquire_run_lease("../outside").is_err());
    }

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
    fn archive_root_is_a_directory_sibling_without_migrating_jsonl() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp/project", "model", "code").unwrap();
        let jsonl = store.path(session.id());
        let expected = directory.path().join(session.id());

        assert_eq!(session.archive_root(), expected);
        assert_eq!(store.archive_root(session.id()).unwrap(), expected);
        assert!(jsonl.is_file());
        assert!(!expected.exists(), "archive storage is created lazily");

        let reopened = store.reopen(session.id()).unwrap();
        assert_eq!(reopened.archive_root(), expected);
        assert!(jsonl.is_file());
    }

    #[test]
    fn reopen_reconciles_archive_without_rewriting_jsonl() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp/project", "model", "code").unwrap();
        let jsonl = store.path(session.id());
        let before = std::fs::read(&jsonl).unwrap();
        let turn = session.archive_root().join("compactions/dropped/turn001");
        std::fs::create_dir_all(&turn).unwrap();
        let orphan = turn.join("001_bash_1234abcd.md");
        std::fs::write(&orphan, "orphan").unwrap();

        let reopened = store.reopen(session.id()).unwrap();

        assert_eq!(reopened.archive_root(), session.archive_root());
        assert!(!orphan.exists());
        assert_eq!(std::fs::read(jsonl).unwrap(), before);
    }

    #[test]
    fn invalid_ids_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        assert!(store.load("../escape").is_err());
        assert!(store.archive_root("../escape").is_err());
        assert!(store.load("").is_err());
        assert!(store.archive_root("").is_err());
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
                compacted_messages: None,
            })
            .unwrap();
        session.append_message(&Message::user("continue")).unwrap();

        let messages = store.load(session.id()).unwrap().messages();
        assert_eq!(messages.len(), 2, "{messages:?}");
        assert!(matches!(
            &messages[0],
            Message::Summary { content } if content.contains("Changed src/main.rs")
        ));
        assert_eq!(messages[1], Message::user("continue"));
    }

    #[test]
    fn old_compaction_json_keeps_manual_reset_semantics() {
        let record: Record =
            serde_json::from_str(r#"{"type":"compaction","summary":"legacy summary"}"#).unwrap();
        assert!(!serde_json::to_string(&record)
            .unwrap()
            .contains("compacted_messages"));

        let messages = loaded(vec![
            Record::User {
                text: "discard me".into(),
            },
            record,
            Record::User {
                text: "keep me".into(),
            },
        ])
        .messages();
        assert_eq!(messages.len(), 2, "{messages:?}");
        assert!(matches!(
            &messages[0],
            Message::Summary { content }
                if content == "[Earlier conversation summary]\nlegacy summary"
        ));
        assert_eq!(messages[1], Message::user("keep me"));
    }

    #[test]
    fn persisted_prefix_fold_preserves_a_later_turn() {
        let messages = loaded(vec![
            Record::User {
                text: "old request".into(),
            },
            assistant_record("old answer"),
            Record::User {
                text: "later request".into(),
            },
            assistant_record("later answer"),
            Record::Compaction {
                summary: "folded old turn".into(),
                compacted_messages: Some(2),
            },
        ])
        .messages();

        assert_eq!(messages.len(), 3, "{messages:?}");
        assert!(matches!(
            &messages[0],
            Message::Summary { content } if content.ends_with("folded old turn")
        ));
        assert_eq!(messages[1], Message::user("later request"));
        assert!(matches!(
            &messages[2],
            Message::Assistant { message }
                if message.content.as_deref() == Some("later answer")
        ));
    }

    #[test]
    fn persisted_prefix_fold_starts_at_the_first_user_after_leading_messages() {
        let leading_text = "output whose user record was not saved";
        let messages = loaded(vec![
            assistant_record(leading_text),
            Record::User {
                text: "old request".into(),
            },
            assistant_record("old answer"),
            Record::User {
                text: "current request".into(),
            },
            assistant_record("current answer"),
            Record::Compaction {
                summary: "folded old turn".into(),
                compacted_messages: Some(2),
            },
        ])
        .messages();

        assert_eq!(
            messages,
            vec![
                Message::assistant(AssistantMessage::text(leading_text)),
                crate::compaction::summary_message("folded old turn"),
                Message::user("current request"),
                Message::assistant(AssistantMessage::text("current answer")),
            ]
        );
    }

    #[test]
    fn stacked_prefix_folds_preserve_summaries_and_current_tail() {
        let messages = loaded(vec![
            Record::User {
                text: "first request".into(),
            },
            assistant_record("first answer"),
            Record::Compaction {
                summary: "first summary".into(),
                compacted_messages: Some(2),
            },
            Record::User {
                text: "second request".into(),
            },
            assistant_record("second answer"),
            Record::User {
                text: "current request".into(),
            },
            assistant_record("current answer"),
            Record::Compaction {
                summary: "second summary".into(),
                compacted_messages: Some(2),
            },
        ])
        .messages();

        assert_eq!(messages.len(), 4, "{messages:?}");
        assert!(matches!(
            &messages[0],
            Message::Summary { content } if content.ends_with("first summary")
        ));
        assert!(matches!(
            &messages[1],
            Message::Summary { content } if content.ends_with("second summary")
        ));
        assert_eq!(messages[2], Message::user("current request"));
        assert!(matches!(
            &messages[3],
            Message::Assistant { message }
                if message.content.as_deref() == Some("current answer")
        ));
    }

    #[test]
    fn stacked_prefix_folds_ignore_user_text_that_spoofs_the_summary_prefix() {
        let prefixed_user = format!("{}literal user task", crate::compaction::SUMMARY_PREFIX);
        let messages = loaded(vec![
            Record::User {
                text: prefixed_user,
            },
            assistant_record("first answer"),
            Record::User {
                text: "second request".into(),
            },
            assistant_record("second answer"),
            Record::Compaction {
                summary: "first structural summary".into(),
                compacted_messages: Some(2),
            },
            Record::User {
                text: "current request".into(),
            },
            assistant_record("current answer"),
            Record::Compaction {
                summary: "second structural summary".into(),
                compacted_messages: Some(2),
            },
        ])
        .messages();

        assert_eq!(
            messages,
            vec![
                crate::compaction::summary_message("first structural summary"),
                crate::compaction::summary_message("second structural summary"),
                Message::user("current request"),
                Message::assistant(AssistantMessage::text("current answer")),
            ]
        );
    }

    #[test]
    fn prefix_fold_flushes_pending_tool_state_before_splicing() {
        let messages = loaded(vec![
            Record::User {
                text: "interrupted request".into(),
            },
            Record::Assistant {
                content: Some("calling".into()),
                reasoning: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "bash".into(),
                    arguments: "{}".into(),
                }],
                responses_items: Vec::new(),
            },
            Record::Compaction {
                summary: "interrupted turn folded".into(),
                compacted_messages: Some(3),
            },
            Record::Tool {
                call_id: "call_1".into(),
                name: "bash".into(),
                content: "late result".into(),
            },
            Record::User {
                text: "next request".into(),
            },
        ])
        .messages();

        assert_eq!(messages.len(), 2, "{messages:?}");
        assert!(matches!(
            &messages[0],
            Message::Summary { content } if content.ends_with("interrupted turn folded")
        ));
        assert_eq!(messages[1], Message::user("next request"));
        assert!(!messages.iter().any(|message| matches!(
            message,
            Message::Assistant { message } if !message.tool_calls.is_empty()
        )));
        assert!(!messages
            .iter()
            .any(|message| matches!(message, Message::Tool { .. })));
    }

    #[test]
    fn corrupt_prefix_count_preserves_history_and_raw_records() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("keep one")).unwrap();
        session.append_message(&Message::user("keep two")).unwrap();
        session
            .append(&Record::Compaction {
                summary: "must not replace history".into(),
                compacted_messages: Some(usize::MAX),
            })
            .unwrap();

        let path = store.path(session.id());
        let raw_before = std::fs::read_to_string(&path).unwrap();
        let loaded = store.load(session.id()).unwrap();
        let messages = loaded.messages();
        let raw_after = std::fs::read_to_string(&path).unwrap();

        assert_eq!(
            messages,
            vec![Message::user("keep one"), Message::user("keep two")]
        );
        assert_eq!(
            raw_after, raw_before,
            "replay must never rewrite the transcript"
        );
        assert_eq!(loaded.records.len(), 4, "all append-only records remain");
        assert!(raw_after.contains(&format!(r#""compacted_messages":{}"#, usize::MAX)));
    }

    #[test]
    fn compression_batch_round_trips_as_one_record() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        let batch = CompressionBatch {
            folds: vec![CompressionFold {
                summary: "folded".into(),
                compacted_messages: 2,
            }],
        };
        session.append_compression_batch(&batch).unwrap();

        let raw = std::fs::read_to_string(store.path(session.id())).unwrap();
        assert_eq!(raw.lines().count(), 2);
        let record: Record = serde_json::from_str(raw.lines().nth(1).unwrap()).unwrap();
        assert!(matches!(record, Record::CompressionBatch { folds } if folds == batch.folds));
    }

    #[test]
    fn compression_batch_replays_sequential_folds_after_leading_messages() {
        let messages = loaded(vec![
            assistant_record("leading"),
            Record::User {
                text: "first".into(),
            },
            assistant_record("first answer"),
            Record::User {
                text: "second".into(),
            },
            assistant_record("second answer"),
            Record::User {
                text: "current".into(),
            },
            assistant_record("current answer"),
            Record::CompressionBatch {
                folds: vec![
                    CompressionFold {
                        summary: "first summary".into(),
                        compacted_messages: 2,
                    },
                    CompressionFold {
                        summary: "second summary".into(),
                        compacted_messages: 2,
                    },
                ],
            },
        ])
        .messages();

        assert_eq!(
            messages,
            vec![
                Message::assistant(AssistantMessage::text("leading")),
                crate::compaction::summary_message("first summary"),
                crate::compaction::summary_message("second summary"),
                Message::user("current"),
                Message::assistant(AssistantMessage::text("current answer")),
            ]
        );
    }

    #[test]
    fn corrupt_first_batch_fold_preserves_the_entire_history() {
        let expected = vec![
            Message::user("first"),
            Message::assistant(AssistantMessage::text("first answer")),
            Message::user("current"),
            Message::assistant(AssistantMessage::text("current answer")),
        ];
        for count in [0, usize::MAX] {
            let messages = loaded(vec![
                Record::User {
                    text: "first".into(),
                },
                assistant_record("first answer"),
                Record::User {
                    text: "current".into(),
                },
                assistant_record("current answer"),
                Record::CompressionBatch {
                    folds: vec![
                        CompressionFold {
                            summary: "bad".into(),
                            compacted_messages: count,
                        },
                        CompressionFold {
                            summary: "must not apply".into(),
                            compacted_messages: 2,
                        },
                    ],
                },
            ])
            .messages();
            assert_eq!(messages, expected, "count {count}");
        }

        let missing_user = loaded(vec![
            assistant_record("leading"),
            Record::CompressionBatch {
                folds: vec![CompressionFold {
                    summary: "bad".into(),
                    compacted_messages: 1,
                }],
            },
        ])
        .messages();
        assert_eq!(
            missing_user,
            vec![Message::assistant(AssistantMessage::text("leading"))]
        );
    }

    #[test]
    fn disabled_turn_gate_prevents_a_late_batch_before_reload() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        let gate = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let writer_session = session.clone();
        let writer_gate = gate.clone();
        let writer = std::thread::spawn(move || {
            writer_session.append_compression_batch_if(
                &CompressionBatch {
                    folds: vec![CompressionFold {
                        summary: "late".into(),
                        compacted_messages: 1,
                    }],
                },
                || {
                    entered_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    writer_gate.load(std::sync::atomic::Ordering::SeqCst)
                },
            )
        });

        entered_rx.recv().unwrap();
        gate.store(false, std::sync::atomic::Ordering::SeqCst);
        let loader_session = session.clone();
        let loader = std::thread::spawn(move || loader_session.load_messages());
        release_tx.send(()).unwrap();

        assert!(!writer.join().unwrap().unwrap());
        assert!(loader.join().unwrap().unwrap().is_empty());
        assert_eq!(store.load(session.id()).unwrap().records.len(), 1);
    }

    #[test]
    fn cloned_sessions_serialize_concurrent_appends() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        let workers: Vec<_> = (0..6)
            .map(|worker| {
                let session = session.clone();
                std::thread::spawn(move || {
                    for item in 0..20 {
                        session
                            .append_compression_batch(&CompressionBatch {
                                folds: vec![CompressionFold {
                                    summary: format!("{worker}:{item}:{}", "x".repeat(8_192)),
                                    compacted_messages: 1,
                                }],
                            })
                            .unwrap();
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }

        assert_eq!(store.load(session.id()).unwrap().records.len(), 121);
        assert!(session.load_messages().unwrap().is_empty());
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
    fn context_projection_round_trip_replaces_only_the_latest_tool_tail() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session
            .append(&Record::User {
                text: "prior turn".into(),
            })
            .unwrap();
        session.append(&assistant_record("prior answer")).unwrap();
        session
            .append(&Record::User {
                text: "current task".into(),
            })
            .unwrap();
        session
            .append(&Record::Assistant {
                content: Some("inline progress".into()),
                reasoning: Some("private".into()),
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    arguments: "{}".into(),
                }],
                responses_items: vec![serde_json::json!({
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": "{}"
                })],
            })
            .unwrap();
        session
            .append(&Record::Tool {
                call_id: "call_1".into(),
                name: "read_file".into(),
                content: "large result".into(),
            })
            .unwrap();
        let projection = ContextProjection {
            historical_folds: Vec::new(),
            current_turn: Some(crate::compaction::CurrentTurnFold {
                note: "## Task\nCurrent task\n## Progress\nRead the file\n## Next\nContinue".into(),
                compacted_messages: 2,
            }),
        };

        session.append_context_projection(&projection).unwrap();
        let loaded = store.load(session.id()).unwrap();

        assert!(matches!(
            loaded.records.last(),
            Some(Record::ContextProjection { projection: stored }) if stored == &projection
        ));
        assert_eq!(loaded.messages().len(), 4);
        assert!(
            matches!(loaded.messages()[0], Message::User { ref content } if content == "prior turn")
        );
        assert!(matches!(loaded.messages()[1], Message::Assistant { .. }));
        assert!(
            matches!(loaded.messages()[2], Message::User { ref content } if content == "current task")
        );
        assert!(matches!(loaded.messages()[3], Message::LastWords { .. }));
    }

    #[test]
    fn corrupt_context_projection_does_not_discard_unknown_history() {
        let records = vec![
            Record::User {
                text: "current task".into(),
            },
            Record::Assistant {
                content: None,
                reasoning: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "grep".into(),
                    arguments: "{}".into(),
                }],
                responses_items: Vec::new(),
            },
            Record::Tool {
                call_id: "call_1".into(),
                name: "grep".into(),
                content: "result".into(),
            },
            Record::ContextProjection {
                projection: ContextProjection {
                    historical_folds: Vec::new(),
                    current_turn: Some(crate::compaction::CurrentTurnFold {
                        note: "fresh note".into(),
                        compacted_messages: 1,
                    }),
                },
            },
        ];

        let messages = loaded(records).messages();

        assert_eq!(messages.len(), 3);
        assert!(matches!(messages[1], Message::Assistant { .. }));
        assert!(matches!(messages[2], Message::Tool { .. }));
    }

    #[test]
    fn corrupt_projection_preserves_pending_tool_for_a_later_result() {
        let records = vec![
            Record::User {
                text: "current task".into(),
            },
            Record::Assistant {
                content: None,
                reasoning: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "grep".into(),
                    arguments: "{}".into(),
                }],
                responses_items: Vec::new(),
            },
            Record::ContextProjection {
                projection: ContextProjection {
                    historical_folds: Vec::new(),
                    current_turn: Some(crate::compaction::CurrentTurnFold {
                        note: "fresh note".into(),
                        compacted_messages: 1,
                    }),
                },
            },
            Record::Tool {
                call_id: "call_1".into(),
                name: "grep".into(),
                content: "real result".into(),
            },
        ];

        let messages = loaded(records).messages();

        assert_eq!(messages.len(), 3);
        assert!(matches!(
            &messages[2],
            Message::Tool { content, .. } if content == "real result"
        ));
    }

    #[test]
    fn context_projection_gate_declines_without_writing() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        let projection = ContextProjection {
            historical_folds: vec![CompressionFold {
                summary: "old".into(),
                compacted_messages: 2,
            }],
            current_turn: None,
        };

        assert!(!session
            .append_context_projection_if(&projection, || false)
            .unwrap());
        assert!(store
            .load(session.id())
            .unwrap()
            .records
            .iter()
            .all(|record| !matches!(record, Record::ContextProjection { .. })));
    }

    #[test]
    fn phase_four_breaker_replays_only_the_latest_user_turn() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session
            .append(&Record::User {
                text: "first".into(),
            })
            .unwrap();
        let (first_turn, previous) = session.load_phase_four_breaker().unwrap();
        assert_eq!(first_turn, 1);
        let mut proposed = previous.clone();
        proposed.side_call_tokens = 40;
        assert_eq!(
            session
                .append_phase_four_breaker_if(first_turn, &previous, &proposed, || true)
                .unwrap(),
            Some(proposed.clone())
        );
        assert_eq!(
            session.load_phase_four_breaker().unwrap(),
            (1, proposed.clone())
        );
        assert_eq!(
            store
                .reopen(session.id())
                .unwrap()
                .load_phase_four_breaker()
                .unwrap(),
            (1, proposed.clone())
        );
        let transcript = std::fs::read_to_string(store.path(session.id())).unwrap();
        assert!(!transcript.contains("phase_four_breaker"));
        session
            .append(&Record::PhaseFourBreaker {
                turn: first_turn,
                state: serde_json::to_value(&proposed).unwrap(),
            })
            .unwrap();
        assert_eq!(store.load(session.id()).unwrap().messages().len(), 1);

        session
            .append(&Record::User {
                text: "second".into(),
            })
            .unwrap();
        assert_eq!(
            session.load_phase_four_breaker().unwrap(),
            (2, PhaseFourBreakerState::default())
        );
        assert_eq!(
            session
                .append_phase_four_breaker_if(first_turn, &previous, &proposed, || true)
                .unwrap(),
            None
        );

        let malformed = PhaseFourBreakerCheckpoint {
            turn: 2,
            state: serde_json::json!({"version": 99}),
        };
        let mut sidecar = std::fs::OpenOptions::new()
            .append(true)
            .open(session.phase_four_breaker_path())
            .unwrap();
        serde_json::to_writer(&mut sidecar, &malformed).unwrap();
        sidecar.write_all(b"\n").unwrap();
        assert_eq!(
            session.load_phase_four_breaker().unwrap(),
            (2, PhaseFourBreakerState::default())
        );
        assert_eq!(store.load(session.id()).unwrap().messages().len(), 2);
    }

    #[test]
    fn phase_four_breaker_migrates_legacy_transcript_state() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("current")).unwrap();
        let mut legacy = PhaseFourBreakerState::default();
        legacy.side_call_tokens = 40;
        session
            .append(&Record::PhaseFourBreaker {
                turn: 1,
                state: serde_json::to_value(&legacy).unwrap(),
            })
            .unwrap();

        assert_eq!(
            session.load_phase_four_breaker().unwrap(),
            (1, legacy.clone())
        );
        let mut proposed = legacy.clone();
        proposed.side_call_tokens = 50;
        assert_eq!(
            session
                .append_phase_four_breaker_if(1, &legacy, &proposed, || true)
                .unwrap(),
            Some(proposed.clone())
        );
        assert_eq!(session.load_phase_four_breaker().unwrap(), (1, proposed));
    }

    #[test]
    fn phase_four_breaker_rejects_corrupt_sidecar_json() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("current")).unwrap();
        let previous = PhaseFourBreakerState::default();
        let mut proposed = previous.clone();
        proposed.side_call_tokens = 40;
        session
            .append_phase_four_breaker_if(1, &previous, &proposed, || true)
            .unwrap();
        let mut sidecar = open_append_private_nofollow(&session.phase_four_breaker_path()).unwrap();
        sidecar.write_all(b"{broken\n").unwrap();

        assert!(session.load_phase_four_breaker().is_err());
        assert!(session
            .append_phase_four_breaker_if(1, &proposed, &proposed, || true)
            .is_err());
    }

    #[test]
    fn phase_four_budget_failure_disables_on_first_strike() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("current")).unwrap();

        let state = session
            .record_phase_four_failure_if(1, true, || true)
            .unwrap()
            .unwrap();
        assert_eq!(state.consecutive_no_progress, 1);
        assert!(state.tail_override);
        assert!(state.disabled);
        assert_eq!(session.load_phase_four_breaker().unwrap().1, state);
    }

    #[test]
    fn phase_four_progress_transitions_persist_for_current_turn() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("current")).unwrap();

        let first = session
            .record_phase_four_progress_if(1, false, || true)
            .unwrap()
            .unwrap();
        assert_eq!(first.consecutive_no_progress, 1);
        assert!(first.tail_override);
        assert!(!first.disabled);
        let second = session
            .record_phase_four_progress_if(1, false, || true)
            .unwrap()
            .unwrap();
        assert_eq!(second.consecutive_no_progress, 2);
        assert!(second.disabled);
        assert_eq!(session.load_phase_four_breaker().unwrap().1, second);
    }

    #[test]
    fn phase_four_round_entry_disables_after_one_hundred_attempts() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("current")).unwrap();

        for expected in 1..=100 {
            let (state, admitted) = session
                .enter_phase_four_round_if(1, -1, || true)
                .unwrap()
                .unwrap();
            assert!(admitted);
            assert_eq!(state.attempts, expected);
        }
        let (state, admitted) = session
            .enter_phase_four_round_if(1, -1, || true)
            .unwrap()
            .unwrap();
        assert!(!admitted);
        assert!(state.disabled);
        assert_eq!(state.attempts, 100);
        assert_eq!(session.load_phase_four_breaker().unwrap().1, state);
    }

    #[test]
    fn phase_four_breaker_merges_spend_from_reopened_handles() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::with_root(directory.path());
        let session = store.create("/tmp", "model", "code").unwrap();
        session.append_message(&Message::user("current")).unwrap();
        let first = store.reopen(session.id()).unwrap();
        let second = store.reopen(session.id()).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let charge = |session: Session, barrier: Arc<std::sync::Barrier>| {
            std::thread::spawn(move || {
                let previous = PhaseFourBreakerState::default();
                let mut proposed = previous.clone();
                proposed.side_call_tokens = 40;
                barrier.wait();
                session
                    .append_phase_four_breaker_if(1, &previous, &proposed, || true)
                    .unwrap()
                    .unwrap()
                    .side_call_tokens
            })
        };
        let first_handle = charge(first, barrier.clone());
        let second_handle = charge(second, barrier.clone());
        barrier.wait();
        let mut totals = vec![first_handle.join().unwrap(), second_handle.join().unwrap()];
        totals.sort_unstable();

        assert_eq!(totals, vec![40, 80]);
        assert_eq!(
            session
                .load_phase_four_breaker()
                .unwrap()
                .1
                .side_call_tokens,
            80
        );
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
