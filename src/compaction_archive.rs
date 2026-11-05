//! Durable storage for model context removed by Phase 4.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone)]
pub struct ArchiveRecord {
    pub turn: u64,
    pub round: u64,
    pub sequence: u64,
    pub tool: String,
    pub created_at: String,
    pub kind: String,
    /// Complete, unredacted Markdown body; storage adds the record terminator if absent.
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

#[derive(Debug, Clone)]
pub struct ArchiveStore {
    root: PathBuf,
    record_cap: usize,
    round_cap: usize,
    session_quota: u64,
}

impl ArchiveStore {
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            record_cap: PER_RECORD_CAP_BYTES,
            round_cap: PER_ROUND_CAP_BYTES,
            session_quota: PER_SESSION_QUOTA_BYTES,
        }
    }

    /// Rebuild derived manifests and return the catalog-authoritative live set.
    pub fn reconcile(&self) -> Result<ArchiveState> {
        match std::fs::symlink_metadata(&self.root) {
            Ok(metadata) if metadata.file_type().is_dir() && !metadata.file_type().is_symlink() => {
            }
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
        let mut records = match self.read_live_catalog() {
            Ok(records) => records,
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
        if self.delete_owned_orphans(&records).is_err()
            || self.rebuild_manifests(&records, &available).is_err()
        {
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

    /// Persist a batch in input order. Policy refusals are returned per record.
    pub fn archive_round(&self, records: &[ArchiveRecord]) -> Result<Vec<ArchiveWrite>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let mut state = self.reconcile()?;
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
            let mut rendered = input.markdown.as_bytes().to_vec();
            if !rendered.ends_with(RECORD_TERMINATOR.as_bytes()) {
                if !rendered.ends_with(b"\n") {
                    rendered.push(b'\n');
                }
                rendered.extend_from_slice(RECORD_TERMINATOR.as_bytes());
            }
            let payload = cap_record_bytes(&rendered, self.record_cap);
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
            let Some((record, path)) = self.write_record(input, &payload, &state.live_records)?
            else {
                writes.push(refusal("record ID exhaustion"));
                continue;
            };
            if let Err(error) = self.append_catalog_record(&record) {
                let _ = std::fs::remove_file(&path);
                return Err(error);
            }
            round_bytes += payload.len();
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
        self.rebuild_manifests(&state.live_records, &state.available_relative_paths)?;
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
        payload: &[u8],
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
            if let Err(error) = file.write_all(payload).and_then(|_| file.sync_all()) {
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
                    state.live_total_bytes = state.live_total_bytes.saturating_sub(record.bytes);
                    state.available_relative_paths.remove(&record.relative_path);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    state.live_total_bytes = state.live_total_bytes.saturating_sub(record.bytes);
                    state.available_relative_paths.remove(&record.relative_path);
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

    fn read_live_catalog(&self) -> Result<Vec<CatalogRecord>> {
        let path = self.root.join(CATALOG_RELATIVE_PATH);
        let mut bytes = Vec::new();
        match open_read_nofollow(&path) {
            Ok(mut file) => file.read_to_end(&mut bytes)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(error).with_context(|| format!("failed to read {}", path.display()))
            }
        };
        let mut live = Vec::<CatalogRecord>::new();
        let mut positions = HashMap::<String, usize>::new();
        let mut paths = HashMap::<String, String>::new();
        for raw in bytes.split(|byte| *byte == b'\n') {
            if raw.is_empty() {
                continue;
            }
            let Ok(value) = serde_json::from_slice::<serde_json::Value>(raw) else {
                continue;
            };
            let Some(record_id) = value.get("record_id").and_then(|value| value.as_str()) else {
                continue;
            };
            if value.get("tombstone").and_then(|value| value.as_bool()) == Some(true) {
                if let Some(index) = positions.remove(record_id) {
                    let removed = live.remove(index);
                    paths.remove(&removed.relative_path);
                    positions = live
                        .iter()
                        .enumerate()
                        .map(|(index, record)| (record.record_id.clone(), index))
                        .collect();
                }
                continue;
            }
            if positions.contains_key(record_id) {
                continue;
            }
            let Ok(mut record) = serde_json::from_value::<CatalogRecord>(value) else {
                continue;
            };
            if record.kind.is_empty() {
                record.kind = default_kind();
            }
            if !valid_record_id(&record.record_id)
                || !safe_record_relative_path(Path::new(&record.relative_path))
                || paths.contains_key(&record.relative_path)
            {
                continue;
            }
            paths.insert(record.relative_path.clone(), record.record_id.clone());
            positions.insert(record.record_id.clone(), live.len());
            live.push(record);
        }
        Ok(live)
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
            if metadata.file_type().is_symlink() {
                return Ok(true);
            }
        }
        for component in relative.components() {
            let Component::Normal(component) = component else {
                return Ok(true);
            };
            current.push(component);
            match std::fs::symlink_metadata(&current) {
                Ok(metadata) if metadata.file_type().is_symlink() => return Ok(true),
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
                Err(error) => return Err(error.into()),
            }
        }
        Ok(false)
    }

    fn delete_owned_orphans(&self, live: &[CatalogRecord]) -> Result<()> {
        let owned: HashSet<_> = live
            .iter()
            .map(|record| record.relative_path.as_str())
            .collect();
        let dropped = self.root.join("compactions/dropped");
        let Ok(turns) = std::fs::read_dir(&dropped) else {
            return Ok(());
        };
        for turn in turns {
            let turn = turn?;
            let file_type = turn.file_type()?;
            if !file_type.is_dir()
                || file_type.is_symlink()
                || !valid_turn_directory(&turn.file_name().to_string_lossy())
            {
                continue;
            }
            for entry in std::fs::read_dir(turn.path())? {
                let entry = entry?;
                if !entry.file_type()?.is_file() || entry.file_name() == "manifest.md" {
                    continue;
                }
                let relative = entry.path().strip_prefix(&self.root)?.to_path_buf();
                if valid_record_filename(&entry.file_name().to_string_lossy())
                    && !owned.contains(path_to_slashes(&relative).as_str())
                {
                    std::fs::remove_file(entry.path())?;
                    sync_directory(&turn.path())?;
                }
            }
        }
        Ok(())
    }

    fn rebuild_manifests(
        &self,
        records: &[CatalogRecord],
        available: &HashSet<String>,
    ) -> Result<()> {
        let mut by_directory = HashMap::<PathBuf, Vec<&CatalogRecord>>::new();
        let dropped = self.root.join("compactions/dropped");
        if let Ok(turns) = std::fs::read_dir(&dropped) {
            for turn in turns {
                let turn = turn?;
                if turn.file_type()?.is_dir()
                    && !turn.file_type()?.is_symlink()
                    && valid_turn_directory(&turn.file_name().to_string_lossy())
                {
                    let relative = turn.path().strip_prefix(&self.root)?.to_path_buf();
                    by_directory.entry(relative).or_default();
                }
            }
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
                if !available.contains(&record.relative_path) {
                    continue;
                }
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
        if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
            output.push(character.to_ascii_lowercase());
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
    components.len() == 4
        && components[0].as_os_str() == "compactions"
        && components[1].as_os_str() == "dropped"
        && valid_turn_directory(&components[2].as_os_str().to_string_lossy())
        && valid_record_filename(&components[3].as_os_str().to_string_lossy())
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
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
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

fn path_to_slashes(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn safe_existing_directory(path: &Path) -> Result<bool> {
    let metadata = std::fs::symlink_metadata(path)?;
    Ok(metadata.file_type().is_dir() && !metadata.file_type().is_symlink())
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

fn replace_private_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension("tmp");
    let _ = std::fs::remove_file(&temporary);
    let mut file = open_new_private(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    std::fs::rename(&temporary, path)?;
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
            markdown: format!("# Dropped context record\n\n{body}\n{RECORD_TERMINATOR}"),
        }
    }

    fn tiny_store(root: &Path, record_cap: usize, round_cap: usize, quota: u64) -> ArchiveStore {
        ArchiveStore {
            root: root.to_path_buf(),
            record_cap,
            round_cap,
            session_quota: quota,
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
        assert_eq!(
            std::fs::read_to_string(store.root.join("compactions/dropped/turn001/manifest.md"))
                .unwrap(),
            "# Dropped context records\n\n(No live records.)\n"
        );
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
        assert!(outside.join("keep.md").exists());
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
