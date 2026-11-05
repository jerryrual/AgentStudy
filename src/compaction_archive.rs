//! Durable storage for model context removed by Phase 4.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::compaction::ContextProjection;
use crate::model::{AssistantMessage, Message, ToolCall};

/// Maximum bytes stored for one Markdown record.
pub const PER_RECORD_CAP_BYTES: usize = 2 * 1024 * 1024;
/// Maximum bytes written by one Phase 4 round.
pub const PER_ROUND_CAP_BYTES: usize = 64 * 1024 * 1024;
/// Maximum live archive bytes retained for one session.
pub const PER_SESSION_QUOTA_BYTES: u64 = 256 * 1024 * 1024;
const RECORD_TERMINATOR: &str = "<!-- end of record -->\n";
const CATALOG_RELATIVE_PATH: &str = "compactions/dropped/catalog.jsonl";
const RECORD_ID_ATTEMPTS: usize = 64;
static RECORD_NONCE: AtomicU64 = AtomicU64::new(0);
static ARCHIVE_LOCKS: OnceLock<Mutex<HashMap<PathBuf, Weak<Mutex<()>>>>> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CatalogRecord {
    pub record_id: String,
    pub relative_path: String,
    pub turn: u64,
    pub round: u64,
    pub tool: String,
    pub bytes: u64,
    pub created_at: String,
    #[serde(default = "default_kind")]
    pub kind: String,
}

fn default_kind() -> String {
    "group".into()
}

fn catalog_record_from_value(value: &serde_json::Value) -> Option<CatalogRecord> {
    let record_id = value.get("record_id")?.as_str()?.to_string();
    let relative_path = value.get("relative_path")?.as_str()?.to_string();
    let turn = value.get("turn")?.as_u64()?;
    let round = value.get("round")?.as_u64()?;
    let tool = value.get("tool")?.as_str()?.to_string();
    let bytes = value.get("bytes")?.as_u64()?;
    let created_at = value.get("created_at")?.as_str()?.to_string();
    if record_id.is_empty() || !safe_record_relative_path(Path::new(&relative_path)) {
        return None;
    }
    let kind = value
        .get("kind")
        .and_then(|kind| kind.as_str())
        .filter(|kind| !kind.is_empty())
        .unwrap_or("group")
        .to_string();
    Some(CatalogRecord {
        record_id,
        relative_path,
        turn,
        round,
        tool,
        bytes,
        created_at,
        kind,
    })
}

#[derive(Debug, Clone)]
pub struct ArchiveRecord {
    pub turn: u64,
    pub round: u64,
    pub sequence: u64,
    pub tool: String,
    pub created_at: String,
    pub kind: String,
    /// Stable group identifier, normally the first provider call ID.
    pub group_id: String,
    /// Session identifier included in the human-readable record header.
    pub session_id: String,
    /// Complete, unredacted Markdown body; storage adds metadata and the terminator.
    pub markdown: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveWrite {
    pub record: Option<CatalogRecord>,
    pub no_record_reason: Option<&'static str>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArchiveState {
    pub live_records: Vec<CatalogRecord>,
    pub live_total_bytes: u64,
    pub available_relative_paths: HashSet<String>,
    pub writable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ManifestEntry {
    pub record_id: Option<String>,
    pub group_id: String,
    pub record_dir: String,
    pub relative_path: Option<String>,
    pub turn: u64,
    pub round: u64,
    pub sequence: u64,
    pub tool: String,
    pub display_argument: String,
    pub outcome: String,
    pub size_chars: usize,
    pub assistant_text: bool,
    pub available: bool,
    pub no_record_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArchivedContextProjection {
    pub entries: Vec<ManifestEntry>,
    pub available_relative_paths: HashSet<String>,
    records: Vec<CatalogRecord>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CurrentTurnArchiveManifest {
    pub entries: Vec<ManifestEntry>,
}

impl CurrentTurnArchiveManifest {
    pub fn append(&mut self, projection: ArchivedContextProjection) {
        for entry in &mut self.entries {
            entry.available = entry
                .relative_path
                .as_ref()
                .is_some_and(|path| projection.available_relative_paths.contains(path));
        }
        self.entries.extend(projection.entries);
        if self.entries.len() > 500 {
            self.entries.drain(..self.entries.len() - 500);
        }
    }

    pub fn render(&self, session_root: &Path, file_read_available: bool) -> Option<String> {
        let last = self.entries.last()?;
        let absolute_root = if session_root.is_absolute() {
            session_root.to_path_buf()
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(session_root)
        };
        let absolute_dir =
            manifest_path_text(&absolute_posix_path(&absolute_root.join(&last.record_dir)));
        let heading = format!(
            "--- Dropped this turn (records under {}/) ---",
            absolute_dir.trim_end_matches('/')
        );
        let show_affordance = file_read_available
            && self.entries.iter().any(|entry| {
                entry.available
                    && entry
                        .relative_path
                        .as_ref()
                        .is_some_and(|path| !path.is_empty())
            });
        let affordance = "To inspect archived input/output, read the listed file with read_file; capped records contain an explicit middle-truncation marker.";
        let mut first = 0usize;
        loop {
            let omitted = first;
            let mut lines = Vec::with_capacity(self.entries.len() - first + 3);
            lines.push(heading.clone());
            if omitted > 0 {
                lines.push(format!("… and {omitted} earlier records — see manifest.md"));
            }
            lines.extend(self.entries[first..].iter().map(render_manifest_entry));
            if show_affordance {
                lines.push(affordance.into());
            }
            let rendered = lines.join("\n");
            if rendered.lines().count() <= 50 && rendered.chars().count() <= 6000 {
                return Some(rendered);
            }
            if first < self.entries.len() {
                first += 1;
                continue;
            }
            let mut truncated: String = rendered.chars().take(5999).collect();
            truncated.push('…');
            return Some(truncated);
        }
    }
}

fn render_manifest_entry(entry: &ManifestEntry) -> String {
    let size = if entry.size_chars < 1000 {
        entry.size_chars.to_string()
    } else {
        format!("{:.1}k", entry.size_chars as f64 / 1000.0)
    };
    let mut row = if entry.assistant_text {
        format!(
            "r{} {:03} assistant text, {size} chars",
            entry.round, entry.sequence
        )
    } else {
        let argument = if entry.display_argument.is_empty() {
            "…"
        } else {
            &entry.display_argument
        };
        format!(
            "r{} {:03} {}({argument}) → {}, {size} chars",
            entry.round, entry.sequence, entry.tool, entry.outcome
        )
    };
    if let Some(reason) = &entry.no_record_reason {
        row.push_str(&format!(" — (no record: {reason})"));
    } else if let Some(path) = &entry.relative_path {
        let basename = Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(path);
        row.push_str(&format!(" — {basename}"));
        if !entry.available {
            row.push_str(" (record missing)");
        }
    }
    row
}

#[derive(Debug, Clone)]
pub struct ArchiveStore {
    root: PathBuf,
    catalog_lock: Arc<Mutex<()>>,
    record_cap: usize,
    round_cap: usize,
    session_quota: u64,
}

fn archive_lock(root: &Path) -> Arc<Mutex<()>> {
    let absolute = if root.is_absolute() {
        root.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(root)
    };
    let key = absolute.canonicalize().unwrap_or_else(|_| {
        absolute
            .parent()
            .and_then(|parent| parent.canonicalize().ok())
            .and_then(|parent| absolute.file_name().map(|name| parent.join(name)))
            .unwrap_or(absolute)
    });
    let registry = ARCHIVE_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut locks = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(lock) = locks.get(&key).and_then(Weak::upgrade) {
        return lock;
    }
    locks.retain(|_, lock| lock.strong_count() > 0);
    let lock = Arc::new(Mutex::new(()));
    locks.insert(key, Arc::downgrade(&lock));
    lock
}

impl ArchiveStore {
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let catalog_lock = archive_lock(&root);
        Self {
            root,
            catalog_lock,
            record_cap: PER_RECORD_CAP_BYTES,
            round_cap: PER_ROUND_CAP_BYTES,
            session_quota: PER_SESSION_QUOTA_BYTES,
        }
    }

    /// Rebuild derived manifests and return the catalog-authoritative live set.
    pub fn reconcile(&self) -> Result<ArchiveState> {
        let _guard = self
            .catalog_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.reconcile_unlocked()
    }

    fn reconcile_unlocked(&self) -> Result<ArchiveState> {
        match std::fs::symlink_metadata(&self.root) {
            Ok(metadata) if metadata.file_type().is_dir() && !metadata_is_link_like(&metadata) => {}
            Ok(_) => return Ok(ArchiveState::default()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ArchiveState {
                    writable: true,
                    ..ArchiveState::default()
                });
            }
            Err(_) => return Ok(ArchiveState::default()),
        }
        match self.catalog_path_is_safe() {
            Ok(true) => {}
            Ok(false) | Err(_) => return Ok(ArchiveState::default()),
        }
        let (mut records, catalog_safe_for_cleanup) = match self.read_live_catalog() {
            Ok(replay) => replay,
            Err(_) => return Ok(ArchiveState::default()),
        };
        let mut available = HashSet::new();
        let mut total = 0u64;
        for record in &mut records {
            let relative = Path::new(&record.relative_path);
            if !safe_record_relative_path(relative)
                || !matches!(self.path_has_symlink(relative), Ok(false))
            {
                continue;
            }
            let path = self.root.join(relative);
            let Ok(metadata) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            if metadata.file_type().is_file() {
                record.bytes = metadata.len();
                total = total.saturating_add(metadata.len());
                available.insert(record.relative_path.clone());
            }
        }
        let cleanup_failed =
            catalog_safe_for_cleanup && self.delete_owned_orphans(&records).is_err();
        if cleanup_failed || self.rebuild_manifests(&records).is_err() {
            return Ok(ArchiveState {
                live_records: records,
                live_total_bytes: total,
                available_relative_paths: available,
                writable: false,
            });
        }
        Ok(ArchiveState {
            live_records: records,
            live_total_bytes: total,
            available_relative_paths: available,
            writable: true,
        })
    }

    /// Return the model-facing pointer to the current live archive catalog.
    ///
    /// Callers snapshot this once at a logical turn boundary so records
    /// archived later in that turn do not change the reminder. The catalog
    /// itself remains the session's append-only live index, matching upstream.
    pub fn catalog_pointer_text(&self) -> Result<Option<String>> {
        let _guard = self
            .catalog_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let state = self.reconcile_unlocked()?;
        if !state.writable || state.live_records.is_empty() || !self.catalog_path_is_safe()? {
            return Ok(None);
        }
        let catalog = self.root.join(CATALOG_RELATIVE_PATH);
        let metadata = std::fs::symlink_metadata(&catalog)?;
        if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
            return Ok(None);
        }
        let resolved_root = self.root.canonicalize()?;
        let resolved_catalog = catalog.canonicalize()?;
        if !resolved_catalog.starts_with(&resolved_root) {
            return Ok(None);
        }
        let path = absolute_posix_path(&resolved_catalog);
        let count = state.live_records.len();
        let label = if count == 1 { "record" } else { "records" };
        Ok(Some(format!(
            "Earlier context compaction archived {count} {label} from previous turns; catalog: {path} (contains each record's relative path)."
        )))
    }

    /// Serialize and archive every raw current-turn tool group replaced by a projection.
    pub fn archive_context_projection(
        &self,
        session_id: &str,
        turn: u64,
        round: u64,
        source: &[Message],
        projection: &ContextProjection,
    ) -> Result<Vec<CatalogRecord>> {
        Ok(self
            .archive_context_projection_with_manifest(session_id, turn, round, source, projection)?
            .records)
    }

    pub(crate) fn archive_context_projection_with_manifest(
        &self,
        session_id: &str,
        turn: u64,
        round: u64,
        source: &[Message],
        projection: &ContextProjection,
    ) -> Result<ArchivedContextProjection> {
        let records = projection_archive_records(session_id, turn, round, source, projection)?;
        let _guard = self
            .catalog_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.preflight_projection_batch_unlocked(&records)?;
        let writes = self.archive_round_unlocked(&records)?;
        let mut archived = Vec::with_capacity(writes.len());
        for write in writes {
            let Some(record) = write.record else {
                bail!(
                    "context projection archive refused: {}",
                    write.no_record_reason.unwrap_or("unknown reason")
                );
            };
            archived.push(record);
        }
        let final_state = self.reconcile_unlocked()?;
        let entries = projection_manifest_entries(
            source,
            &records,
            &archived,
            &final_state.available_relative_paths,
        );
        Ok(ArchivedContextProjection {
            entries,
            available_relative_paths: final_state.available_relative_paths,
            records: archived,
        })
    }

    /// Persist a batch in input order. Policy refusals are returned per record.
    pub fn archive_round(&self, records: &[ArchiveRecord]) -> Result<Vec<ArchiveWrite>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let _guard = self
            .catalog_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.archive_round_unlocked(records)
    }

    fn preflight_projection_batch_unlocked(&self, records: &[ArchiveRecord]) -> Result<()> {
        let state = self.reconcile_unlocked()?;
        if !state.writable {
            bail!("context projection archive storage is unavailable");
        }
        let mut total = 0usize;
        for record in records {
            let bytes = render_record_payload(record, "00000000", self.record_cap).len();
            if bytes as u64 > self.session_quota {
                bail!("context projection archive exceeds the session quota");
            }
            total = total
                .checked_add(bytes)
                .context("context projection archive size overflow")?;
            if total > self.round_cap {
                bail!("context projection archive exceeds the round cap");
            }
        }
        if total as u64 > self.session_quota {
            bail!("context projection archive batch exceeds the session quota");
        }
        Ok(())
    }

    fn archive_round_unlocked(&self, records: &[ArchiveRecord]) -> Result<Vec<ArchiveWrite>> {
        let mut state = self.reconcile_unlocked()?;
        if !state.writable {
            return Ok(refusals(records.len(), "storage unavailable"));
        }
        self.ensure_archive_directories()?;
        let mut writes = Vec::with_capacity(records.len());
        let mut round_bytes = 0usize;
        let mut protected = HashSet::new();
        let mut round_refused = false;
        for input in records {
            if round_refused {
                writes.push(refusal("round cap"));
                continue;
            }
            let payload = render_record_payload(input, "00000000", self.record_cap);
            if round_bytes.saturating_add(payload.len()) > self.round_cap {
                round_refused = true;
                writes.push(refusal("round cap"));
                continue;
            }
            if payload.len() as u64 > self.session_quota {
                writes.push(refusal("quota"));
                continue;
            }
            if !self.make_quota_room(&mut state, payload.len() as u64, &protected)? {
                writes.push(refusal("quota"));
                continue;
            }
            let Some((record, path)) = self.write_record(input, &state.live_records)? else {
                writes.push(refusal("record ID exhaustion"));
                continue;
            };
            if record.bytes != payload.len() as u64 {
                let _ = std::fs::remove_file(&path);
                bail!("archive record size changed after ID allocation");
            }
            if let Err(error) = self.append_catalog_record(&record) {
                let _ = std::fs::remove_file(&path);
                return Err(error);
            }
            round_bytes += record.bytes as usize;
            state.live_total_bytes += record.bytes;
            state
                .available_relative_paths
                .insert(record.relative_path.clone());
            protected.insert(record.record_id.clone());
            state.live_records.push(record.clone());
            writes.push(ArchiveWrite {
                record: Some(record),
                no_record_reason: None,
            });
        }
        self.rebuild_manifests(&state.live_records)?;
        Ok(writes)
    }

    fn ensure_archive_directories(&self) -> Result<()> {
        ensure_private_directory(&self.root)?;
        ensure_private_directory(&self.root.join("compactions"))?;
        ensure_private_directory(&self.root.join("compactions/dropped"))?;
        if !self.catalog_path_is_safe()? {
            bail!("unsafe archive catalog path");
        }
        Ok(())
    }

    fn write_record(
        &self,
        input: &ArchiveRecord,
        live: &[CatalogRecord],
    ) -> Result<Option<(CatalogRecord, PathBuf)>> {
        let relative_dir = PathBuf::from(format!("compactions/dropped/turn{:03}", input.turn));
        if self.path_has_symlink(&relative_dir)? {
            bail!("unsafe archive record directory");
        }
        let directory = self.root.join(&relative_dir);
        ensure_private_directory(&directory)?;
        let existing_ids: HashSet<_> = live
            .iter()
            .map(|record| record.record_id.as_str())
            .collect();
        for _ in 0..RECORD_ID_ATTEMPTS {
            let record_id = next_record_id();
            if existing_ids.contains(record_id.as_str()) {
                continue;
            }
            let payload = render_record_payload(input, &record_id, self.record_cap);
            let filename = build_record_filename(input.sequence, &input.tool, &record_id);
            let relative_path = relative_dir.join(filename);
            let path = self.root.join(&relative_path);
            let mut file = match open_new_private(&path) {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("failed to create {}", path.display()))
                }
            };
            if let Err(error) = file.write_all(&payload).and_then(|_| file.sync_all()) {
                drop(file);
                let _ = std::fs::remove_file(&path);
                return Err(error).with_context(|| format!("failed to write {}", path.display()));
            }
            sync_directory(&directory)?;
            let record = CatalogRecord {
                record_id,
                relative_path: path_to_slashes(&relative_path),
                turn: input.turn,
                round: input.round,
                tool: input.tool.clone(),
                bytes: payload.len() as u64,
                created_at: input.created_at.clone(),
                kind: if input.kind.is_empty() {
                    default_kind()
                } else {
                    input.kind.clone()
                },
            };
            return Ok(Some((record, path)));
        }
        Ok(None)
    }

    fn make_quota_room(
        &self,
        state: &mut ArchiveState,
        incoming: u64,
        protected: &HashSet<String>,
    ) -> Result<bool> {
        while state.live_total_bytes.saturating_add(incoming) > self.session_quota {
            let Some(index) = state
                .live_records
                .iter()
                .position(|record| !protected.contains(&record.record_id))
            else {
                return Ok(false);
            };
            let record = state.live_records.remove(index);
            self.append_tombstone(&record.record_id)?;
            let path = self.root.join(&record.relative_path);
            match std::fs::remove_file(&path) {
                Ok(()) => {
                    sync_directory(path.parent().unwrap_or(&self.root))?;
                    if state.available_relative_paths.remove(&record.relative_path) {
                        state.live_total_bytes =
                            state.live_total_bytes.saturating_sub(record.bytes);
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    if state.available_relative_paths.remove(&record.relative_path) {
                        state.live_total_bytes =
                            state.live_total_bytes.saturating_sub(record.bytes);
                    }
                }
                Err(_) => {
                    self.append_catalog_record(&record)?;
                    state.live_records.push(record);
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn read_live_catalog(&self) -> Result<(Vec<CatalogRecord>, bool)> {
        let path = self.root.join(CATALOG_RELATIVE_PATH);
        let mut bytes = Vec::new();
        match open_read_nofollow(&path) {
            Ok(mut file) => file.read_to_end(&mut bytes)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok((Vec::new(), true))
            }
            Err(error) => {
                return Err(error).with_context(|| format!("failed to read {}", path.display()))
            }
        };
        let mut live = Vec::<Option<CatalogRecord>>::new();
        let mut positions = HashMap::<String, usize>::new();
        let mut paths = HashMap::<String, String>::new();
        let terminated = bytes.ends_with(b"\n");
        let segment_count = bytes.split(|byte| *byte == b'\n').count();
        let mut safe_for_cleanup = true;
        for (index, raw) in bytes.split(|byte| *byte == b'\n').enumerate() {
            if raw.is_empty() {
                continue;
            }
            let Ok(value) = serde_json::from_slice::<serde_json::Value>(raw) else {
                if terminated || index + 1 != segment_count {
                    safe_for_cleanup = false;
                }
                continue;
            };
            let Some(record_id) = value
                .get("record_id")
                .and_then(|value| value.as_str())
                .filter(|record_id| !record_id.is_empty())
            else {
                safe_for_cleanup = false;
                continue;
            };
            if value.get("tombstone").and_then(|value| value.as_bool()) == Some(true) {
                if let Some(index) = positions.remove(record_id) {
                    if let Some(removed) = live[index].take() {
                        paths.remove(&removed.relative_path);
                    }
                }
                continue;
            }
            if positions.contains_key(record_id) {
                continue;
            }
            let Some(record) = catalog_record_from_value(&value) else {
                safe_for_cleanup = false;
                continue;
            };
            if paths.contains_key(&record.relative_path) {
                continue;
            }
            paths.insert(record.relative_path.clone(), record.record_id.clone());
            positions.insert(record.record_id.clone(), live.len());
            live.push(Some(record));
        }
        Ok((live.into_iter().flatten().collect(), safe_for_cleanup))
    }

    fn append_catalog_record(&self, record: &CatalogRecord) -> Result<()> {
        self.append_catalog_value(&serde_json::to_value(record)?)
    }

    fn append_tombstone(&self, record_id: &str) -> Result<()> {
        self.append_catalog_value(&serde_json::json!({
            "created_at": unix_timestamp(),
            "record_id": record_id,
            "tombstone": true
        }))
    }

    fn append_catalog_value(&self, value: &serde_json::Value) -> Result<()> {
        if !self.catalog_path_is_safe()? {
            bail!("unsafe archive catalog path");
        }
        ensure_private_directory(&self.root.join("compactions/dropped"))?;
        let path = self.root.join(CATALOG_RELATIVE_PATH);
        let mut file = open_append_private_nofollow(&path)?;
        let length = file.seek(SeekFrom::End(0))?;
        if length > 0 {
            file.seek(SeekFrom::End(-1))?;
            let mut last = [0u8; 1];
            file.read_exact(&mut last)?;
            file.seek(SeekFrom::End(0))?;
            if last[0] != b'\n' {
                file.write_all(b"\n")?;
            }
        }
        let mut serialized = serde_json::to_vec(value)?;
        serialized.push(b'\n');
        file.write_all(&serialized)?;
        file.sync_all()?;
        sync_directory(path.parent().unwrap_or(&self.root))?;
        Ok(())
    }

    fn catalog_path_is_safe(&self) -> Result<bool> {
        Ok(!self.path_has_symlink(Path::new(CATALOG_RELATIVE_PATH))?)
    }

    fn path_has_symlink(&self, relative: &Path) -> Result<bool> {
        let mut current = self.root.clone();
        if let Ok(metadata) = std::fs::symlink_metadata(&current) {
            if metadata_is_link_like(&metadata) {
                return Ok(true);
            }
        }
        for component in relative.components() {
            let Component::Normal(component) = component else {
                return Ok(true);
            };
            current.push(component);
            match std::fs::symlink_metadata(&current) {
                Ok(metadata) if metadata_is_link_like(&metadata) => return Ok(true),
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
                Err(error) => return Err(error.into()),
            }
        }
        Ok(false)
    }

    fn append_turn_directories(
        &self,
        relative_root: &Path,
        output: &mut Vec<PathBuf>,
    ) -> Result<()> {
        if self.path_has_symlink(relative_root)? {
            bail!("unsafe archive turn root {}", relative_root.display());
        }
        let root = self.root.join(relative_root);
        let Ok(entries) = std::fs::read_dir(&root) else {
            return Ok(());
        };
        for entry in entries {
            let entry = entry?;
            let relative = entry.path().strip_prefix(&self.root)?.to_path_buf();
            if self.path_has_symlink(&relative)? {
                bail!("unsafe archive turn directory {}", relative.display());
            }
            if entry.file_type()?.is_dir()
                && valid_turn_directory(&entry.file_name().to_string_lossy())
            {
                output.push(entry.path());
            }
        }
        Ok(())
    }

    fn owned_turn_directories(&self) -> Result<Vec<PathBuf>> {
        let mut directories = Vec::new();
        self.append_turn_directories(Path::new("compactions/dropped"), &mut directories)?;
        let sub_agents_relative = Path::new("compactions/sub_agents");
        if self.path_has_symlink(sub_agents_relative)? {
            bail!("unsafe sub-agent archive root");
        }
        let sub_agents = self.root.join(sub_agents_relative);
        let Ok(tools) = std::fs::read_dir(sub_agents) else {
            return Ok(directories);
        };
        for tool in tools {
            let tool = tool?;
            let tool_relative = tool.path().strip_prefix(&self.root)?.to_path_buf();
            if self.path_has_symlink(&tool_relative)? {
                bail!("unsafe sub-agent tool archive directory");
            }
            if !tool.file_type()?.is_dir() {
                continue;
            }
            for invocation in std::fs::read_dir(tool.path())? {
                let invocation = invocation?;
                let invocation_relative = invocation.path().strip_prefix(&self.root)?.to_path_buf();
                if self.path_has_symlink(&invocation_relative)? {
                    bail!("unsafe sub-agent invocation archive directory");
                }
                if !invocation.file_type()?.is_dir() {
                    continue;
                }
                self.append_turn_directories(
                    &invocation_relative.join("dropped"),
                    &mut directories,
                )?;
            }
        }
        Ok(directories)
    }

    fn delete_owned_orphans(&self, live: &[CatalogRecord]) -> Result<()> {
        let owned: HashSet<_> = live
            .iter()
            .map(|record| record.relative_path.as_str())
            .collect();
        for turn_path in self.owned_turn_directories()? {
            for entry in std::fs::read_dir(&turn_path)? {
                let entry = entry?;
                if !entry.file_type()?.is_file() || entry.file_name() == "manifest.md" {
                    continue;
                }
                let relative = entry.path().strip_prefix(&self.root)?.to_path_buf();
                if valid_record_filename(&entry.file_name().to_string_lossy())
                    && !owned.contains(path_to_slashes(&relative).as_str())
                {
                    std::fs::remove_file(entry.path())?;
                    sync_directory(&turn_path)?;
                }
            }
        }
        Ok(())
    }

    fn rebuild_manifests(&self, records: &[CatalogRecord]) -> Result<()> {
        let mut by_directory = HashMap::<PathBuf, Vec<&CatalogRecord>>::new();
        for directory in self.owned_turn_directories()? {
            let relative = directory.strip_prefix(&self.root)?.to_path_buf();
            by_directory.entry(relative).or_default();
        }
        for record in records {
            let relative = Path::new(&record.relative_path);
            if let Some(parent) = relative.parent() {
                by_directory
                    .entry(parent.to_path_buf())
                    .or_default()
                    .push(record);
            }
        }
        for (relative_dir, entries) in by_directory {
            if self.path_has_symlink(&relative_dir)? {
                bail!("unsafe archive manifest directory");
            }
            let directory = self.root.join(&relative_dir);
            ensure_private_directory(&directory)?;
            let mut text = String::from("# Dropped context records\n\n");
            let mut count = 0usize;
            for record in entries {
                let filename = Path::new(&record.relative_path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("record.md");
                text.push_str(&format!(
                    "- r{} `{}` — [{}]({}) — {} bytes — {}\n",
                    record.round, record.tool, filename, filename, record.bytes, record.created_at
                ));
                count += 1;
            }
            if count == 0 {
                text.push_str("(No live records.)\n");
            }
            replace_private_file(&directory.join("manifest.md"), text.as_bytes())?;
        }
        Ok(())
    }
}

fn projection_archive_records(
    session_id: &str,
    turn: u64,
    round: u64,
    source: &[Message],
    projection: &ContextProjection,
) -> Result<Vec<ArchiveRecord>> {
    let fold = projection
        .current_turn
        .as_ref()
        .context("context projection has no current-turn fold to archive")?;
    let user = source
        .iter()
        .rposition(|message| matches!(message, Message::User { .. }))
        .context("context projection archive has no current user turn")?;
    let tail = &source[user + 1..];
    if fold.compacted_messages == 0 || fold.compacted_messages != tail.len() {
        bail!("context projection archive does not match the complete current tail");
    }

    let created_at = unix_timestamp();
    let mut records = Vec::new();
    let mut cursor = 0usize;
    while cursor < tail.len() {
        let Message::Assistant { message } = &tail[cursor] else {
            cursor += 1;
            continue;
        };
        if message.tool_calls.is_empty() {
            cursor += 1;
            continue;
        }
        let end = tail[cursor + 1..]
            .iter()
            .position(|candidate| {
                matches!(candidate, Message::Assistant { message } if !message.tool_calls.is_empty())
            })
            .map_or(tail.len(), |offset| cursor + 1 + offset);
        let group = &tail[cursor..end];
        let sequence = records.len() as u64 + 1;
        records.push(ArchiveRecord {
            turn,
            round,
            sequence,
            tool: message
                .tool_calls
                .first()
                .map(|call| call.name.clone())
                .unwrap_or_else(|| "assistant".into()),
            created_at: created_at.clone(),
            kind: default_kind(),
            group_id: message
                .tool_calls
                .first()
                .map(|call| call.id.clone())
                .unwrap_or_else(|| format!("turn-{turn}-round-{round}-group-{sequence}")),
            session_id: session_id.to_string(),
            markdown: render_tool_group(message, group),
        });
        cursor = end;
    }
    if records.is_empty() {
        bail!("context projection archive contains no tool-call groups");
    }
    Ok(records)
}

fn unsafe_manifest_character(character: char) -> bool {
    character.is_control()
        || matches!(
            character,
            '\u{0085}'
                | '\u{061c}'
                | '\u{200e}'
                | '\u{200f}'
                | '\u{2028}'..='\u{202e}'
                | '\u{2066}'..='\u{2069}'
        )
}

fn manifest_inline_text(value: &str, max_chars: usize) -> String {
    value
        .chars()
        .map(|character| {
            if unsafe_manifest_character(character) {
                ' '
            } else {
                character
            }
        })
        .take(max_chars)
        .collect()
}

fn manifest_path_text(value: &str) -> String {
    let mut rendered = String::with_capacity(value.len());
    for character in value.chars() {
        if unsafe_manifest_character(character) {
            rendered.extend(character.escape_default());
        } else {
            rendered.push(character);
        }
    }
    rendered
}

fn manifest_tool_name(name: &str) -> String {
    let rendered = manifest_inline_text(name, 255);
    if rendered.trim().is_empty() {
        "unknown".into()
    } else {
        rendered
    }
}

fn manifest_display_argument(call: &ToolCall) -> String {
    let Ok(serde_json::Value::Object(arguments)) = serde_json::from_str(&call.arguments) else {
        return String::new();
    };
    let keys: &[&str] = match call.name.as_str() {
        "read_file" | "write_file" | "edit_file" | "list_dir" => &["path"],
        "grep" | "glob" => &["pattern", "path"],
        "bash" => &["command"],
        _ => &[],
    };
    let Some((key, value)) = keys
        .iter()
        .find_map(|key| arguments.get(*key).map(|value| (*key, value)))
    else {
        return String::new();
    };
    let Ok(encoded) = serde_json::to_string(value) else {
        return String::new();
    };
    let lower = encoded.to_ascii_lowercase();
    if [
        "token",
        "password",
        "passwd",
        "secret",
        "api_key",
        "api-key",
        "authorization",
        "credential",
        "bearer ",
        "ghp_",
        "github_pat",
        "sk-",
        "xox",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
    {
        return String::new();
    }
    let rendered = manifest_inline_text(&format!("{key}={encoded}"), usize::MAX);
    let cap = if matches!(key, "path" | "file_path") {
        256
    } else {
        120
    };
    if rendered.chars().count() <= cap {
        return rendered;
    }
    if matches!(key, "path" | "file_path") {
        let head = (cap - 1) / 2;
        let tail = cap - 1 - head;
        let start: String = rendered.chars().take(head).collect();
        let end: String = rendered
            .chars()
            .rev()
            .take(tail)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        format!("{start}…{end}")
    } else {
        let mut truncated: String = rendered.chars().take(cap - 1).collect();
        truncated.push('…');
        truncated
    }
}

fn projection_manifest_entries(
    source: &[Message],
    inputs: &[ArchiveRecord],
    archived: &[CatalogRecord],
    available_relative_paths: &HashSet<String>,
) -> Vec<ManifestEntry> {
    let Some(user) = source
        .iter()
        .rposition(|message| matches!(message, Message::User { .. }))
    else {
        return Vec::new();
    };
    let tail = &source[user + 1..];
    let mut entries = Vec::new();
    let mut cursor = 0usize;
    let mut group_index = 0usize;
    while cursor < tail.len() {
        let Message::Assistant { message } = &tail[cursor] else {
            cursor += 1;
            continue;
        };
        if message.tool_calls.is_empty() {
            cursor += 1;
            continue;
        }
        let end = tail[cursor + 1..]
            .iter()
            .position(|candidate| {
                matches!(candidate, Message::Assistant { message } if !message.tool_calls.is_empty())
            })
            .map_or(tail.len(), |offset| cursor + 1 + offset);
        let Some((input, record)) = inputs.get(group_index).zip(archived.get(group_index)) else {
            break;
        };
        let results: Vec<_> = tail[cursor + 1..end]
            .iter()
            .filter_map(|message| match message {
                Message::Tool {
                    call_id, content, ..
                } => Some((call_id, content)),
                _ => None,
            })
            .collect();
        let record_dir = Path::new(&record.relative_path)
            .parent()
            .map(path_to_slashes)
            .unwrap_or_default();
        for call in &message.tool_calls {
            let size_chars = results
                .iter()
                .find(|(call_id, _)| call_id.as_str() == call.id)
                .map_or(0, |(_, content)| content.chars().count());
            entries.push(ManifestEntry {
                record_id: Some(record.record_id.clone()),
                group_id: input.group_id.clone(),
                record_dir: record_dir.clone(),
                relative_path: Some(record.relative_path.clone()),
                turn: input.turn,
                round: input.round,
                sequence: input.sequence,
                tool: manifest_tool_name(&call.name),
                display_argument: manifest_display_argument(call),
                // Message::Tool has no structured execution stamp; upstream's fallback is unknown.
                outcome: "unknown".into(),
                size_chars,
                assistant_text: false,
                available: available_relative_paths.contains(&record.relative_path),
                no_record_reason: None,
            });
        }
        group_index += 1;
        cursor = end;
    }
    entries
}

fn render_tool_group(assistant: &AssistantMessage, group: &[Message]) -> String {
    let mut output = String::new();
    let mut context = Vec::new();
    if let Some(content) = assistant
        .content
        .as_deref()
        .filter(|content| !content.is_empty())
    {
        context.push(content.to_string());
    }
    if let Some(reasoning) = assistant
        .reasoning
        .as_deref()
        .filter(|reasoning| !reasoning.is_empty())
    {
        context.push(format!("[reasoning]\n{reasoning}"));
    }
    if !context.is_empty() {
        output.push_str("## Assistant context\n\n");
        output.push_str(&context.join("\n"));
        output.push_str("\n\n");
    }
    if !assistant.responses_items.is_empty() {
        output.push_str("## Native response items\n\n```json\n");
        output.push_str(
            &serde_json::to_string_pretty(&assistant.responses_items)
                .unwrap_or_else(|_| "[]".into()),
        );
        output.push_str("\n```\n\n");
    }

    let results: Vec<_> = group
        .iter()
        .filter_map(|message| match message {
            Message::Tool {
                call_id,
                name,
                content,
            } => Some((call_id, name, content)),
            _ => None,
        })
        .collect();
    let mut matched = HashSet::new();
    for call in &assistant.tool_calls {
        output.push_str(&render_tool_call(call, &results, &mut matched));
    }
    for (index, (call_id, name, content)) in results.iter().enumerate() {
        if matched.contains(&index) {
            continue;
        }
        output.push_str("## Additional group content\n\n");
        output.push_str(&format!(
            "### Unmatched tool result: {name}\n\nCall ID: `{call_id}`\n\n{content}\n\n"
        ));
    }
    output
}

fn render_tool_call(
    call: &ToolCall,
    results: &[(&String, &String, &String)],
    matched: &mut HashSet<usize>,
) -> String {
    let arguments = serde_json::from_str::<serde_json::Value>(&call.arguments)
        .ok()
        .and_then(|value| serde_json::to_string_pretty(&value).ok())
        .unwrap_or_else(|| call.arguments.clone());
    let result = results
        .iter()
        .enumerate()
        .find(|(index, (call_id, _, _))| !matched.contains(index) && call_id.as_str() == call.id)
        .map(|(index, (_, _, content))| {
            matched.insert(index);
            content.as_str()
        });
    format!(
        "## Call: {}\n\n### Arguments (as requested)\n```json\n{}\n```\n\nOutcome: `unknown`\n\n## Result\n\n{}\n\n",
        call.name,
        arguments,
        result.unwrap_or("(missing result)")
    )
}

fn render_record_payload(input: &ArchiveRecord, record_id: &str, cap: usize) -> Vec<u8> {
    if input.group_id.is_empty() && input.session_id.is_empty() {
        let mut rendered = input.markdown.as_bytes().to_vec();
        if !rendered.ends_with(RECORD_TERMINATOR.as_bytes()) {
            if !rendered.ends_with(b"\n") {
                rendered.push(b'\n');
            }
            rendered.extend_from_slice(RECORD_TERMINATOR.as_bytes());
        }
        return cap_record_bytes(&rendered, cap);
    }

    let body = input
        .markdown
        .strip_suffix(RECORD_TERMINATOR)
        .unwrap_or(&input.markdown)
        .trim_end_matches('\n');
    let rendered = format!(
        "# Dropped context record\n\n- Record ID: `{record_id}`\n- Group ID: `{}`\n- Turn: {}\n- Round: {}\n- Timestamp: {}\n- Session ID: `{}`\n\n{body}\n{}",
        input.group_id,
        input.turn,
        input.round,
        input.created_at,
        input.session_id,
        RECORD_TERMINATOR,
    );
    cap_record_bytes(rendered.as_bytes(), cap)
}

fn refusals(count: usize, reason: &'static str) -> Vec<ArchiveWrite> {
    (0..count).map(|_| refusal(reason)).collect()
}

fn refusal(reason: &'static str) -> ArchiveWrite {
    ArchiveWrite {
        record: None,
        no_record_reason: Some(reason),
    }
}

fn cap_record_bytes(payload: &[u8], cap: usize) -> Vec<u8> {
    if payload.len() <= cap {
        return payload.to_vec();
    }
    let marker = format!("\n[… middle-truncated from {} bytes …]\n", payload.len());
    let terminator = RECORD_TERMINATOR.as_bytes();
    let body = payload.strip_suffix(terminator).unwrap_or(payload);
    let keep = cap.saturating_sub(marker.len() + terminator.len());
    let head_target = keep.div_ceil(2);
    let tail_target = keep - head_target;
    let head_end = utf8_floor(body, head_target);
    let tail_start = utf8_ceil(body, body.len().saturating_sub(tail_target));
    let mut capped = Vec::with_capacity(cap);
    capped.extend_from_slice(&body[..head_end]);
    capped.extend_from_slice(marker.as_bytes());
    capped.extend_from_slice(&body[tail_start..]);
    capped.extend_from_slice(terminator);
    debug_assert!(capped.len() <= cap);
    capped
}

fn utf8_floor(bytes: &[u8], mut index: usize) -> usize {
    index = index.min(bytes.len());
    while index > 0 && index < bytes.len() && (bytes[index] & 0b1100_0000) == 0b1000_0000 {
        index -= 1;
    }
    index
}

fn utf8_ceil(bytes: &[u8], mut index: usize) -> usize {
    index = index.min(bytes.len());
    while index < bytes.len() && (bytes[index] & 0b1100_0000) == 0b1000_0000 {
        index += 1;
    }
    index
}

fn build_record_filename(sequence: u64, tool: &str, record_id: &str) -> String {
    let safe = safe_component(tool);
    let suffix = format!("_{record_id}.md");
    let prefix = format!("{sequence:03}_");
    let available = 255usize.saturating_sub(prefix.len() + suffix.len());
    let bounded = truncate_utf8(&safe, available).trim_end_matches('_');
    format!(
        "{prefix}{}{suffix}",
        if bounded.is_empty() {
            "assistant"
        } else {
            bounded
        }
    )
}

fn safe_component(value: &str) -> String {
    let mut output = String::new();
    let mut separator = false;
    for character in value.chars() {
        if character.is_alphanumeric() || matches!(character, '-' | '_') {
            output.push(character);
            separator = false;
        } else if !separator && !output.is_empty() {
            output.push('_');
            separator = true;
        }
    }
    output.trim_matches('_').to_string()
}

fn truncate_utf8(value: &str, max_bytes: usize) -> &str {
    let mut end = value.len().min(max_bytes);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn safe_record_relative_path(path: &Path) -> bool {
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return false;
    }
    let components: Vec<_> = path.components().collect();
    let main = components.len() == 4
        && components[0].as_os_str() == "compactions"
        && components[1].as_os_str() == "dropped"
        && valid_turn_directory(&components[2].as_os_str().to_string_lossy())
        && catalog_record_filename(&components[3].as_os_str().to_string_lossy());
    let sub_agent = components.len() == 7
        && components[0].as_os_str() == "compactions"
        && components[1].as_os_str() == "sub_agents"
        && components[4].as_os_str() == "dropped"
        && valid_turn_directory(&components[5].as_os_str().to_string_lossy())
        && catalog_record_filename(&components[6].as_os_str().to_string_lossy());
    main || sub_agent
}

fn catalog_record_filename(value: &str) -> bool {
    value != "manifest.md" && value.ends_with(".md")
}

fn valid_turn_directory(value: &str) -> bool {
    value
        .strip_prefix("turn")
        .is_some_and(|digits| digits.len() >= 3 && digits.chars().all(|c| c.is_ascii_digit()))
}

fn valid_record_filename(value: &str) -> bool {
    if value.len() > 255 || !value.ends_with(".md") {
        return false;
    }
    let Some(stem) = value.strip_suffix(".md") else {
        return false;
    };
    let Some((prefix, id)) = stem.rsplit_once('_') else {
        return false;
    };
    valid_record_id(id)
        && prefix.split_once('_').is_some_and(|(sequence, tool)| {
            sequence.len() >= 3
                && sequence.chars().all(|c| c.is_ascii_digit())
                && !tool.is_empty()
                && tool
                    .chars()
                    .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_'))
        })
}

fn valid_record_id(value: &str) -> bool {
    value.len() == 8
        && value
            .chars()
            .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase())
}

fn next_record_id() -> String {
    let mut hasher = DefaultHasher::new();
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .hash(&mut hasher);
    RECORD_NONCE
        .fetch_add(1, Ordering::Relaxed)
        .hash(&mut hasher);
    std::process::id().hash(&mut hasher);
    format!("{:08x}", hasher.finish() as u32)
}

fn unix_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = seconds / 86_400;
    let remainder = seconds % 86_400;
    let hour = remainder / 3_600;
    let minute = (remainder % 3_600) / 60;
    let second = remainder % 60;
    format!(
        "{}T{hour:02}:{minute:02}:{second:02}Z",
        civil_from_days(days as i64)
    )
}

fn civil_from_days(days: i64) -> String {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = (z - era * 146_097) as u64;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era as i64 + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}-{month:02}-{day:02}")
}

#[cfg(unix)]
fn absolute_posix_path(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(windows)]
fn absolute_posix_path(path: &Path) -> String {
    let rendered = path.to_string_lossy().replace('\\', "/");
    if let Some(unc) = rendered.strip_prefix("//?/UNC/") {
        return format!("//{unc}");
    }
    rendered
        .strip_prefix("//?/")
        .unwrap_or(&rendered)
        .to_string()
}

#[cfg(not(any(unix, windows)))]
fn absolute_posix_path(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn path_to_slashes(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(unix)]
fn metadata_is_link_like(metadata: &std::fs::Metadata) -> bool {
    metadata.file_type().is_symlink()
}

#[cfg(windows)]
fn metadata_is_link_like(metadata: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
    metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(any(unix, windows)))]
fn metadata_is_link_like(metadata: &std::fs::Metadata) -> bool {
    metadata.file_type().is_symlink()
}

fn safe_existing_directory(path: &Path) -> Result<bool> {
    let metadata = std::fs::symlink_metadata(path)?;
    Ok(metadata.file_type().is_dir() && !metadata_is_link_like(&metadata))
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn ensure_private_directory(path: &Path) -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    match std::fs::symlink_metadata(path) {
        Ok(_) if safe_existing_directory(path)? => return Ok(()),
        Ok(_) => bail!("unsafe archive directory {}", path.display()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let mut builder = std::fs::DirBuilder::new();
    builder.mode(0o700).create(path)?;
    if let Some(parent) = path.parent() {
        sync_directory(parent)?;
    }
    sync_directory(path)?;
    Ok(())
}

#[cfg(not(unix))]
fn ensure_private_directory(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(_) if safe_existing_directory(path)? => return Ok(()),
        Ok(_) => bail!("unsafe archive directory {}", path.display()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    std::fs::create_dir(path)?;
    Ok(())
}

#[cfg(unix)]
fn open_new_private(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
}

#[cfg(not(unix))]
fn open_new_private(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().write(true).create_new(true).open(path)
}

#[cfg(unix)]
fn open_append_private_nofollow(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
}

#[cfg(not(unix))]
fn open_append_private_nofollow(path: &Path) -> std::io::Result<File> {
    OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(path)
}

#[cfg(unix)]
fn open_read_nofollow(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
}

#[cfg(not(unix))]
fn open_read_nofollow(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().read(true).open(path)
}

#[cfg(not(windows))]
fn replace_path(temporary: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::rename(temporary, destination)
}

#[cfg(windows)]
fn replace_path(temporary: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
    #[link(name = "Kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    }

    let temporary: Vec<u16> = temporary.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    // Both paths are validated session-owned siblings; this performs one
    // replace-existing transition and asks Windows to flush it before return.
    let replaced = unsafe {
        MoveFileExW(
            temporary.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if replaced == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn replace_private_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension("tmp");
    let _ = std::fs::remove_file(&temporary);
    let mut file = open_new_private(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    replace_path(&temporary, path)?;
    sync_directory(path.parent().unwrap_or_else(|| Path::new(".")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(turn: u64, round: u64, sequence: u64, tool: &str, body: &str) -> ArchiveRecord {
        ArchiveRecord {
            turn,
            round,
            sequence,
            tool: tool.into(),
            created_at: "2026-01-02T03:04:05Z".into(),
            kind: "group".into(),
            group_id: String::new(),
            session_id: String::new(),
            markdown: format!("# Dropped context record\n\n{body}\n{RECORD_TERMINATOR}"),
        }
    }

    fn tiny_store(root: &Path, record_cap: usize, round_cap: usize, quota: u64) -> ArchiveStore {
        let mut store = ArchiveStore::with_root(root);
        store.record_cap = record_cap;
        store.round_cap = round_cap;
        store.session_quota = quota;
        store
    }

    fn manifest_entry(sequence: u64, relative_path: Option<&str>) -> ManifestEntry {
        ManifestEntry {
            record_id: relative_path.map(|_| format!("record-{sequence}")),
            group_id: format!("group-{sequence}"),
            record_dir: "compactions/dropped/turn001".into(),
            relative_path: relative_path.map(str::to_string),
            turn: 1,
            round: 2,
            sequence,
            tool: "read_file".into(),
            display_argument: String::new(),
            outcome: "unknown".into(),
            size_chars: sequence as usize,
            assistant_text: false,
            available: relative_path.is_some(),
            no_record_reason: None,
        }
    }

    #[test]
    fn missing_root_is_empty_and_writable() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("session");
        let state = ArchiveStore::with_root(&root).reconcile().unwrap();
        assert!(state.writable);
        assert!(state.live_records.is_empty());
        assert!(!root.exists());
    }

    #[test]
    fn catalog_pointer_names_the_validated_absolute_live_catalog() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("session");
        let store = ArchiveStore::with_root(&root);
        assert_eq!(store.catalog_pointer_text().unwrap(), None);

        store
            .archive_round(&[record(1, 1, 1, "read_file", "one")])
            .unwrap();
        let catalog = root.join(CATALOG_RELATIVE_PATH).canonicalize().unwrap();
        assert_eq!(
            store.catalog_pointer_text().unwrap().unwrap(),
            format!(
                "Earlier context compaction archived 1 record from previous turns; catalog: {} (contains each record's relative path).",
                absolute_posix_path(&catalog)
            )
        );

        store
            .archive_round(&[record(2, 1, 1, "grep", "two")])
            .unwrap();
        assert!(store
            .catalog_pointer_text()
            .unwrap()
            .unwrap()
            .contains("archived 2 records from previous turns"));
    }

    #[cfg(unix)]
    #[test]
    fn catalog_pointer_preserves_unix_backslashes_and_reads_the_same_file() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("session\\with-backslash");
        let store = ArchiveStore::with_root(&root);
        store
            .archive_round(&[record(1, 1, 1, "read_file", "one")])
            .unwrap();
        let pointer = store.catalog_pointer_text().unwrap().unwrap();
        let rendered = pointer
            .strip_prefix(
                "Earlier context compaction archived 1 record from previous turns; catalog: ",
            )
            .unwrap()
            .strip_suffix(" (contains each record's relative path).")
            .unwrap();
        let catalog = root.join(CATALOG_RELATIVE_PATH).canonicalize().unwrap();

        assert_eq!(rendered, catalog.to_string_lossy());
        assert_eq!(
            std::fs::read_to_string(rendered).unwrap(),
            std::fs::read_to_string(catalog).unwrap()
        );
    }

    #[cfg(windows)]
    #[test]
    fn absolute_posix_path_preserves_windows_drive_and_unc_roots() {
        assert_eq!(
            absolute_posix_path(Path::new(r"\\?\C:\sessions\catalog.jsonl")),
            "C:/sessions/catalog.jsonl"
        );
        assert_eq!(
            absolute_posix_path(Path::new(r"\\?\UNC\server\share\sessions\catalog.jsonl")),
            "//server/share/sessions/catalog.jsonl"
        );
    }

    #[test]
    fn stores_for_one_root_share_a_catalog_transaction_lock() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("session");
        let first = ArchiveStore::with_root(&root);
        let second = ArchiveStore::with_root(&root);
        assert!(Arc::ptr_eq(&first.catalog_lock, &second.catalog_lock));
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut workers = Vec::new();
        for (store, tool) in [(first, "one"), (second, "two")] {
            let barrier = barrier.clone();
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                store
                    .archive_round(&[record(1, 1, 1, tool, "body")])
                    .unwrap();
            }));
        }
        barrier.wait();
        for worker in workers {
            worker.join().unwrap();
        }

        let state = ArchiveStore::with_root(&root).reconcile().unwrap();
        assert_eq!(state.live_records.len(), 2);
    }

    #[test]
    fn archives_records_and_rebuilds_manifest() {
        let directory = tempfile::tempdir().unwrap();
        let store = ArchiveStore::with_root(directory.path().join("session"));
        let writes = store
            .archive_round(&[record(2, 3, 1, "read_file", "complete secret")])
            .unwrap();
        let catalog = writes[0].record.as_ref().unwrap();
        assert!(store.root.join(&catalog.relative_path).is_file());
        let manifest =
            std::fs::read_to_string(store.root.join("compactions/dropped/turn002/manifest.md"))
                .unwrap();
        assert!(manifest.contains("r3 `read_file`"));
        assert!(manifest.ends_with("2026-01-02T03:04:05Z\n"));
        let state = store.reconcile().unwrap();
        assert_eq!(state.live_records.len(), 1);
        assert_eq!(state.live_total_bytes, catalog.bytes);
    }

    #[test]
    fn projection_archive_groups_parallel_calls_and_skips_prior_last_words() {
        let directory = tempfile::tempdir().unwrap();
        let store = ArchiveStore::with_root(directory.path().join("session"));
        let source = vec![
            Message::user("task"),
            Message::last_words("old progress should not be archived"),
            Message::assistant(AssistantMessage {
                content: Some("assistant context".into()),
                reasoning: Some("full reasoning".into()),
                tool_calls: vec![
                    ToolCall {
                        id: "call_a".into(),
                        name: "read_file".into(),
                        arguments: r#"{"path":"secret-a"}"#.into(),
                    },
                    ToolCall {
                        id: "call_b".into(),
                        name: "grep".into(),
                        arguments: r#"{"pattern":"secret-b"}"#.into(),
                    },
                ],
                responses_items: vec![serde_json::json!({"encrypted": "native-secret"})],
            }),
            Message::tool("call_a", "read_file", "result-a"),
            Message::tool("call_b", "grep", "result-b"),
        ];
        let projection = ContextProjection {
            historical_folds: Vec::new(),
            current_turn: Some(crate::compaction::CurrentTurnFold {
                note: "fresh progress".into(),
                compacted_messages: 4,
            }),
        };

        let projection_archive = store
            .archive_context_projection_with_manifest("session-id", 3, 2, &source, &projection)
            .unwrap();
        let records = &projection_archive.records;

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].tool, "read_file");
        let archived = std::fs::read_to_string(store.root.join(&records[0].relative_path)).unwrap();
        for expected in [
            "Group ID: `call_a`",
            "Turn: 3",
            "Round: 2",
            "Session ID: `session-id`",
            "assistant context",
            "full reasoning",
            "\"path\": \"secret-a\"",
            "\"pattern\": \"secret-b\"",
            "native-secret",
            "result-a",
            "result-b",
        ] {
            assert!(
                archived.contains(expected),
                "missing archive detail: {expected}"
            );
        }
        assert!(!archived.contains("old progress should not be archived"));
        assert!(archived.ends_with(RECORD_TERMINATOR));

        assert_eq!(projection_archive.entries.len(), 2);
        assert_eq!(projection_archive.entries[0].tool, "read_file");
        assert_eq!(projection_archive.entries[1].tool, "grep");
        assert_eq!(
            projection_archive.entries[0].record_id,
            projection_archive.entries[1].record_id
        );
        assert_eq!(
            projection_archive.entries[0].relative_path,
            projection_archive.entries[1].relative_path
        );
        assert_eq!(
            projection_archive.entries[0].sequence,
            projection_archive.entries[1].sequence
        );
        assert_eq!(
            projection_archive.entries[0].size_chars,
            "result-a".chars().count()
        );
        assert_eq!(
            projection_archive.entries[1].size_chars,
            "result-b".chars().count()
        );
        assert!(projection_archive
            .entries
            .iter()
            .all(|entry| entry.available));
    }

    #[test]
    fn current_turn_manifest_appends_invalidates_and_retains_newest_500() {
        let mut manifest = CurrentTurnArchiveManifest::default();
        for batch in 0..6u64 {
            let start = batch * 100 + 1;
            let entries = (start..start + 100)
                .map(|sequence| {
                    manifest_entry(
                        sequence,
                        Some(&format!(
                            "compactions/dropped/turn001/{sequence:03}_read_file_deadbeef.md"
                        )),
                    )
                })
                .collect::<Vec<_>>();
            let available_relative_paths = entries
                .iter()
                .filter_map(|entry| entry.relative_path.clone())
                .collect();
            manifest.append(ArchivedContextProjection {
                entries,
                available_relative_paths,
                records: Vec::new(),
            });
        }

        assert_eq!(manifest.entries.len(), 500);
        assert_eq!(manifest.entries.first().unwrap().sequence, 101);
        assert_eq!(manifest.entries.last().unwrap().sequence, 600);

        let newest_path = manifest
            .entries
            .last()
            .unwrap()
            .relative_path
            .clone()
            .unwrap();
        manifest.append(ArchivedContextProjection {
            entries: Vec::new(),
            available_relative_paths: HashSet::from([newest_path]),
            records: Vec::new(),
        });
        assert!(!manifest.entries.first().unwrap().available);
        assert!(manifest.entries.last().unwrap().available);
    }

    #[test]
    fn current_turn_manifest_render_obeys_line_and_unicode_bounds() {
        let mut manifest = CurrentTurnArchiveManifest {
            entries: (1..=100)
                .map(|sequence| {
                    let mut entry = manifest_entry(
                        sequence,
                        Some(&format!(
                            "compactions/dropped/turn001/{sequence:03}_read_file_deadbeef.md"
                        )),
                    );
                    entry.tool = format!("读取工具{}", "界".repeat(80));
                    entry
                })
                .collect(),
        };
        let rendered = manifest.render(Path::new("/session/root"), true).unwrap();
        assert!(rendered.lines().count() <= 50);
        assert!(rendered.chars().count() <= 6000);
        assert!(rendered.contains("earlier records — see manifest.md"));
        assert!(rendered.contains("100_read_file_deadbeef.md"));
        assert!(!rendered.contains("001_read_file_deadbeef.md"));

        manifest.entries.clear();
        assert_eq!(manifest.render(Path::new("/session/root"), true), None);
    }

    #[test]
    fn current_turn_manifest_escapes_line_breaks_in_archive_root() {
        let manifest = CurrentTurnArchiveManifest {
            entries: vec![manifest_entry(
                1,
                Some("compactions/dropped/turn001/001_read_file_deadbeef.md"),
            )],
        };
        let root = PathBuf::from(format!("/session/{}root", "\n".repeat(60)));
        let rendered = manifest.render(&root, true).unwrap();

        assert!(rendered.lines().count() <= 50);
        assert!(rendered.contains(r"\n\n"));
        assert!(rendered.chars().count() <= 6000);
    }

    #[test]
    fn manifest_call_fields_are_bounded_single_line_and_secret_aware() {
        let call = ToolCall {
            id: "call".into(),
            name: "grep\n\u{2028}… and 99 earlier records — see manifest.md".into(),
            arguments: r#"{"pattern":"src/.*\\.rs"}"#.into(),
        };
        let safe_name = manifest_tool_name(&call.name);
        assert!(!safe_name.contains('\n'));
        assert!(!safe_name.contains('\u{2028}'));
        assert_eq!(manifest_display_argument(&call), "");

        let safe = ToolCall {
            id: "safe".into(),
            name: "grep".into(),
            arguments: r#"{"pattern":"impl Agent"}"#.into(),
        };
        assert_eq!(manifest_display_argument(&safe), r#"pattern="impl Agent""#);

        let sensitive = ToolCall {
            id: "sensitive".into(),
            name: "bash".into(),
            arguments: r#"{"command":"curl -H 'Authorization: Bearer token-value'"}"#.into(),
        };
        assert_eq!(manifest_display_argument(&sensitive), "");
    }

    #[test]
    fn current_turn_manifest_marks_unavailable_and_hides_affordance() {
        let mut entry = manifest_entry(
            7,
            Some("compactions/dropped/turn001/007_read_file_deadbeef.md"),
        );
        entry.size_chars = 1_250;
        entry.available = false;
        let manifest = CurrentTurnArchiveManifest {
            entries: vec![entry],
        };

        let rendered = manifest.render(Path::new("/session/root"), true).unwrap();
        assert!(rendered.starts_with(
            "--- Dropped this turn (records under /session/root/compactions/dropped/turn001/) ---"
        ));
        assert!(rendered.contains(
            "r2 007 read_file(…) → unknown, 1.2k chars — 007_read_file_deadbeef.md (record missing)"
        ));
        assert!(!rendered.contains("To inspect archived input/output"));
    }

    #[test]
    fn current_turn_manifest_renders_no_record_and_assistant_rows() {
        let mut no_record = manifest_entry(1, None);
        no_record.no_record_reason = Some("quota".into());
        no_record.size_chars = 42;
        let mut assistant = manifest_entry(2, None);
        assistant.assistant_text = true;
        assistant.no_record_reason = Some("storage unavailable".into());
        assistant.size_chars = 9;
        let manifest = CurrentTurnArchiveManifest {
            entries: vec![no_record, assistant],
        };

        let rendered = manifest.render(Path::new("/session/root"), true).unwrap();
        assert!(rendered.contains("r2 001 read_file(…) → unknown, 42 chars — (no record: quota)"));
        assert!(
            rendered.contains("r2 002 assistant text, 9 chars — (no record: storage unavailable)")
        );
        assert!(!rendered.contains("To inspect archived input/output"));
    }

    #[test]
    fn projection_archive_writes_one_record_per_sequential_group() {
        let directory = tempfile::tempdir().unwrap();
        let store = ArchiveStore::with_root(directory.path().join("session"));
        let call = |id: &str, name: &str| {
            Message::assistant(AssistantMessage {
                tool_calls: vec![ToolCall {
                    id: id.into(),
                    name: name.into(),
                    arguments: "{}".into(),
                }],
                ..AssistantMessage::default()
            })
        };
        let source = vec![
            Message::user("task"),
            call("call_a", "read_file"),
            Message::tool("call_a", "read_file", "first"),
            call("call_b", "grep"),
            Message::tool("call_b", "grep", "second"),
        ];
        let projection = ContextProjection {
            historical_folds: Vec::new(),
            current_turn: Some(crate::compaction::CurrentTurnFold {
                note: "fresh progress".into(),
                compacted_messages: 4,
            }),
        };

        let records = store
            .archive_context_projection("session-id", 1, 1, &source, &projection)
            .unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].tool, "read_file");
        assert_eq!(records[1].tool, "grep");
        assert!(records[0].relative_path.contains("001_read_file_"));
        assert!(records[1].relative_path.contains("002_grep_"));
    }

    #[test]
    fn projection_archive_preflights_the_whole_batch_before_mutation() {
        let source = vec![
            Message::user("task"),
            Message::assistant(AssistantMessage {
                tool_calls: vec![ToolCall {
                    id: "call_a".into(),
                    name: "read_file".into(),
                    arguments: "{}".into(),
                }],
                ..AssistantMessage::default()
            }),
            Message::tool("call_a", "read_file", "first"),
            Message::assistant(AssistantMessage {
                tool_calls: vec![ToolCall {
                    id: "call_b".into(),
                    name: "grep".into(),
                    arguments: "{}".into(),
                }],
                ..AssistantMessage::default()
            }),
            Message::tool("call_b", "grep", "second"),
        ];
        let projection = ContextProjection {
            historical_folds: Vec::new(),
            current_turn: Some(crate::compaction::CurrentTurnFold {
                note: "fresh progress".into(),
                compacted_messages: 4,
            }),
        };

        let round_directory = tempfile::tempdir().unwrap();
        let round_root = round_directory.path().join("session");
        let round_store = tiny_store(&round_root, 1_000, 400, 10_000);
        assert!(round_store
            .archive_context_projection("session-id", 1, 1, &source, &projection)
            .is_err());
        assert!(
            !round_root.exists(),
            "preflight must not create partial records"
        );

        let quota_directory = tempfile::tempdir().unwrap();
        let quota_root = quota_directory.path().join("session");
        let quota_store = tiny_store(&quota_root, 1_000, 10_000, 400);
        let seeded = quota_store
            .archive_round(&[record(1, 1, 1, "old", "old record")])
            .unwrap();
        let old = seeded[0].record.as_ref().unwrap().clone();
        assert!(quota_store
            .archive_context_projection("session-id", 2, 1, &source, &projection)
            .is_err());
        let state = quota_store.reconcile().unwrap();
        assert_eq!(state.live_records, vec![old.clone()]);
        assert!(quota_root.join(old.relative_path).exists());
    }

    #[test]
    fn record_filename_preserves_unicode_tool_names() {
        let directory = tempfile::tempdir().unwrap();
        let store = ArchiveStore::with_root(directory.path().join("session"));
        let write = store
            .archive_round(&[record(1, 1, 1, "读取Tool", "body")])
            .unwrap();
        let filename = Path::new(&write[0].record.as_ref().unwrap().relative_path)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap();
        assert!(filename.contains("读取Tool"));
        assert!(filename.len() <= 255);
    }

    #[test]
    fn catalog_tombstones_and_torn_lines_replay_in_order() {
        let directory = tempfile::tempdir().unwrap();
        let store = ArchiveStore::with_root(directory.path().join("session"));
        let writes = store
            .archive_round(&[record(1, 1, 1, "one", "a"), record(1, 1, 2, "two", "b")])
            .unwrap();
        let first = writes[0].record.as_ref().unwrap();
        store.append_tombstone(&first.record_id).unwrap();
        let catalog = store.root.join(CATALOG_RELATIVE_PATH);
        OpenOptions::new()
            .append(true)
            .open(&catalog)
            .unwrap()
            .write_all(b"{torn")
            .unwrap();
        let state = store.reconcile().unwrap();
        assert_eq!(state.live_records.len(), 1);
        assert_eq!(state.live_records[0].tool, "two");
    }

    #[test]
    fn tolerant_catalog_replay_preserves_legacy_and_sub_agent_records() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("session");
        let main_relative = "compactions/dropped/turn001/001_Bash_1234abcd.md";
        let sub_relative =
            "compactions/sub_agents/worker/invoke-1/dropped/turn002/legacy-output.md";
        for relative in [main_relative, sub_relative] {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "record").unwrap();
        }
        let catalog = root.join(CATALOG_RELATIVE_PATH);
        std::fs::create_dir_all(catalog.parent().unwrap()).unwrap();
        let rows = [
            serde_json::json!({
                "record_id": "legacy-id",
                "relative_path": main_relative,
                "turn": 1,
                "round": 2,
                "tool": "Bash",
                "bytes": 999,
                "created_at": "2026-01-02T03:04:05Z",
                "kind": 7
            }),
            serde_json::json!({
                "record_id": "sub-agent-id",
                "relative_path": sub_relative,
                "turn": 2,
                "round": 1,
                "tool": "worker",
                "bytes": 999,
                "created_at": "2026-01-02T03:04:06Z"
            }),
        ];
        let payload = rows
            .iter()
            .map(serde_json::Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(catalog, payload).unwrap();

        let state = ArchiveStore::with_root(&root).reconcile().unwrap();

        assert!(state.writable);
        assert_eq!(state.live_records.len(), 2);
        assert_eq!(state.live_records[0].kind, "group");
        assert_eq!(state.live_total_bytes, 12);
        assert!(root.join(main_relative).exists());
        assert!(root.join(sub_relative).exists());
        assert!(root
            .join("compactions/sub_agents/worker/invoke-1/dropped/turn002/manifest.md")
            .is_file());
    }

    #[test]
    fn missing_cataloged_record_becomes_unavailable() {
        let directory = tempfile::tempdir().unwrap();
        let store = ArchiveStore::with_root(directory.path().join("session"));
        let write = store
            .archive_round(&[record(1, 1, 1, "bash", "complete output")])
            .unwrap();
        let relative = &write[0].record.as_ref().unwrap().relative_path;
        std::fs::remove_file(store.root.join(relative)).unwrap();

        let state = store.reconcile().unwrap();

        assert_eq!(state.live_records.len(), 1);
        assert!(state.available_relative_paths.is_empty());
        assert_eq!(state.live_total_bytes, 0);
        let manifest =
            std::fs::read_to_string(store.root.join("compactions/dropped/turn001/manifest.md"))
                .unwrap();
        assert!(manifest.contains("r1 `bash`"));
        assert!(manifest.contains(Path::new(relative).file_name().unwrap().to_str().unwrap()));
    }

    #[test]
    fn storage_adds_the_record_terminator() {
        let directory = tempfile::tempdir().unwrap();
        let store = ArchiveStore::with_root(directory.path().join("session"));
        let mut input = record(1, 1, 1, "bash", "complete output");
        input.markdown = "# Dropped context record\n\ncomplete output\n".into();
        let write = store.archive_round(&[input]).unwrap();
        let bytes = std::fs::read(
            store
                .root
                .join(&write[0].record.as_ref().unwrap().relative_path),
        )
        .unwrap();
        assert!(bytes.ends_with(RECORD_TERMINATOR.as_bytes()));
    }

    #[test]
    fn caps_utf8_in_the_middle_and_preserves_terminator() {
        let directory = tempfile::tempdir().unwrap();
        let store = tiny_store(&directory.path().join("session"), 180, 1_000, 1_000);
        let writes = store
            .archive_round(&[record(1, 1, 1, "bash", &"界".repeat(200))])
            .unwrap();
        let catalog = writes[0].record.as_ref().unwrap();
        let bytes = std::fs::read(store.root.join(&catalog.relative_path)).unwrap();
        assert!(bytes.len() <= 180);
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("middle-truncated from"));
        assert!(text.ends_with(RECORD_TERMINATOR));
    }

    #[test]
    fn round_cap_refuses_triggering_and_remaining_records() {
        let directory = tempfile::tempdir().unwrap();
        let store = tiny_store(&directory.path().join("session"), 1_000, 150, 10_000);
        let writes = store
            .archive_round(&[
                record(1, 1, 1, "a", &"x".repeat(10)),
                record(1, 1, 2, "b", &"y".repeat(100)),
                record(1, 1, 3, "c", "small"),
            ])
            .unwrap();
        assert!(writes[0].record.is_some());
        assert_eq!(writes[1].no_record_reason, Some("round cap"));
        assert_eq!(writes[2].no_record_reason, Some("round cap"));
    }

    #[test]
    fn quota_does_not_reclaim_catalog_bytes_for_a_missing_record() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("session");
        let initial = ArchiveStore::with_root(&root);
        let writes = initial
            .archive_round(&[
                record(1, 1, 1, "missing", &"a".repeat(80)),
                record(2, 1, 1, "existing", &"b".repeat(80)),
            ])
            .unwrap();
        let missing = writes[0].record.as_ref().unwrap();
        let existing = writes[1].record.as_ref().unwrap();
        let existing_path = existing.relative_path.clone();
        std::fs::remove_file(root.join(&missing.relative_path)).unwrap();
        let incoming = record(3, 1, 1, "incoming", &"c".repeat(80));
        let incoming_bytes = incoming.markdown.len() as u64;
        let quota = existing.bytes + incoming_bytes - 1;
        let constrained = tiny_store(&root, 1_000, 2_000, quota);

        let result = constrained.archive_round(&[incoming]).unwrap();

        assert!(result[0].record.is_some());
        assert!(
            !root.join(existing_path).exists(),
            "the actual existing record must be evicted after the missing row reclaims zero"
        );
        assert!(constrained.reconcile().unwrap().live_total_bytes <= quota);
    }

    #[test]
    fn quota_evicts_oldest_catalog_record() {
        let directory = tempfile::tempdir().unwrap();
        let store = tiny_store(&directory.path().join("session"), 1_000, 2_000, 220);
        let first = store
            .archive_round(&[record(1, 1, 1, "old", &"a".repeat(50))])
            .unwrap();
        let first_path = first[0].record.as_ref().unwrap().relative_path.clone();
        let second = store
            .archive_round(&[record(2, 1, 1, "new", &"b".repeat(80))])
            .unwrap();
        assert!(second[0].record.is_some());
        assert!(!store.root.join(first_path).exists());
        assert_eq!(
            std::fs::read_to_string(store.root.join("compactions/dropped/turn001/manifest.md"))
                .unwrap(),
            "# Dropped context records\n\n(No live records.)\n"
        );
        let state = store.reconcile().unwrap();
        assert_eq!(state.live_records.len(), 1);
        assert_eq!(state.live_records[0].tool, "new");
    }

    #[test]
    fn malformed_complete_catalog_line_preserves_owned_files() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("session");
        let turn = root.join("compactions/dropped/turn001");
        std::fs::create_dir_all(&turn).unwrap();
        let record_path = turn.join("001_bash_1234abcd.md");
        std::fs::write(&record_path, "record").unwrap();
        std::fs::write(
            root.join(CATALOG_RELATIVE_PATH),
            b"{\"record_id\":\"unsupported\"}\n",
        )
        .unwrap();

        let state = ArchiveStore::with_root(&root).reconcile().unwrap();

        assert!(state.writable);
        assert!(record_path.exists());
    }

    #[test]
    fn reconciliation_deletes_only_owned_orphans() {
        let directory = tempfile::tempdir().unwrap();
        let store = ArchiveStore::with_root(directory.path().join("session"));
        let write = store
            .archive_round(&[record(1, 1, 1, "kept", "body")])
            .unwrap();
        let turn = store.root.join("compactions/dropped/turn001");
        let orphan = turn.join("999_orphan_1234abcd.md");
        let unrelated = turn.join("notes.txt");
        std::fs::write(&orphan, "orphan").unwrap();
        std::fs::write(&unrelated, "keep").unwrap();
        store.reconcile().unwrap();
        assert!(store
            .root
            .join(&write[0].record.as_ref().unwrap().relative_path)
            .exists());
        assert!(!orphan.exists());
        assert!(unrelated.exists());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_catalog_disables_storage_without_deleting_records() {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("session");
        let dropped = root.join("compactions/dropped");
        std::fs::create_dir_all(root.join("compactions")).unwrap();
        let outside = directory.path().join("outside");
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(outside.join("keep.md"), "keep").unwrap();
        symlink(&outside, &dropped).unwrap();
        let store = ArchiveStore::with_root(&root);
        let state = store.reconcile().unwrap();
        assert!(!state.writable);
        let writes = store
            .archive_round(&[record(1, 1, 1, "bash", "secret")])
            .unwrap();
        assert_eq!(writes[0].no_record_reason, Some("storage unavailable"));
        assert_eq!(store.catalog_pointer_text().unwrap(), None);
        assert!(outside.join("keep.md").exists());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_catalog_file_has_no_model_pointer() {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("session");
        let dropped = root.join("compactions/dropped");
        std::fs::create_dir_all(&dropped).unwrap();
        let outside = directory.path().join("outside-catalog.jsonl");
        std::fs::write(&outside, "{}\n").unwrap();
        symlink(&outside, root.join(CATALOG_RELATIVE_PATH)).unwrap();

        let store = ArchiveStore::with_root(&root);
        assert_eq!(store.catalog_pointer_text().unwrap(), None);
        assert_eq!(std::fs::read_to_string(outside).unwrap(), "{}\n");
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_sub_agent_root_disables_storage_without_external_mutation() {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("session");
        std::fs::create_dir_all(root.join("compactions")).unwrap();
        let outside_turn = directory
            .path()
            .join("outside/worker/invoke/dropped/turn001");
        std::fs::create_dir_all(&outside_turn).unwrap();
        let outside_record = outside_turn.join("001_bash_1234abcd.md");
        std::fs::write(&outside_record, "keep").unwrap();
        symlink(
            directory.path().join("outside"),
            root.join("compactions/sub_agents"),
        )
        .unwrap();

        let state = ArchiveStore::with_root(&root).reconcile().unwrap();

        assert!(!state.writable);
        assert!(outside_record.exists());
        assert!(!outside_turn.join("manifest.md").exists());
    }

    #[cfg(unix)]
    #[test]
    fn records_and_catalog_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let store = ArchiveStore::with_root(directory.path().join("session"));
        let write = store
            .archive_round(&[record(1, 1, 1, "bash", "secret")])
            .unwrap();
        let record_mode = std::fs::metadata(
            store
                .root
                .join(&write[0].record.as_ref().unwrap().relative_path),
        )
        .unwrap()
        .permissions()
        .mode()
            & 0o777;
        let catalog_mode = std::fs::metadata(store.root.join(CATALOG_RELATIVE_PATH))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(record_mode, 0o600);
        assert_eq!(catalog_mode, 0o600);
    }
}
