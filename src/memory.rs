//! Loads project-local reference context from `AGENTS.md`.

use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Approximation of upstream's aggregate memory token budget.
pub const DEFAULT_MAX_MEMORY_TOKENS: usize = 35_000;

const DEFAULT_MEMORY_FILE: &str = "AGENTS.md";
const MEMORY_HEADER: &str = "# Auto-loaded memory\n\nThe following files were auto-loaded as reference context for the current task.\nRelative paths are resolved from the workspace working directory.\n\n";

/// Project memory prepared for injection into the agent's system prompt.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProjectMemory {
    pub prompt: String,
    pub files: Vec<PathBuf>,
    pub warning: Option<String>,
}

/// Load the default `AGENTS.md` memory source for a workspace.
///
/// Missing, unreadable and non-regular files are a silent no-op. Symlinked
/// `AGENTS.md` entries are ignored so implicit memory cannot disclose another
/// workspace file. Text decoding is lossy, matching upstream's nonfatal
/// decoding behavior.
pub fn load_project_memory(cwd: &Path) -> ProjectMemory {
    load_project_memory_with_limit(cwd, DEFAULT_MAX_MEMORY_TOKENS)
}

/// Resolve, contain, decode, render and budget the default memory source.
fn load_project_memory_with_limit(cwd: &Path, max_tokens: usize) -> ProjectMemory {
    let Ok(cwd) = cwd.canonicalize() else {
        return ProjectMemory::default();
    };
    if !cwd.is_dir() {
        return ProjectMemory::default();
    }

    let path = cwd.join(DEFAULT_MEMORY_FILE);
    let Ok(metadata) = std::fs::symlink_metadata(&path) else {
        return ProjectMemory::default();
    };
    if !metadata.file_type().is_file() {
        return ProjectMemory::default();
    }

    let contents = match read_lossy_bounded(&path, max_tokens.saturating_mul(4)) {
        BoundedRead::Contents(contents) => contents,
        BoundedRead::TooLarge => return over_budget(),
        BoundedRead::Unavailable => return ProjectMemory::default(),
    };
    let prompt = format!("{MEMORY_HEADER}## File: {DEFAULT_MEMORY_FILE}\n\n{contents}");
    if estimate_tokens(&prompt) > max_tokens {
        return over_budget();
    }

    ProjectMemory {
        prompt,
        files: vec![path],
        warning: None,
    }
}

/// Return the nonfatal warning used when the memory source cannot fit.
fn over_budget() -> ProjectMemory {
    ProjectMemory {
        warning: Some(format!(
            "Project memory {DEFAULT_MEMORY_FILE} was skipped because it exceeds the approximate 35,000-token limit."
        )),
        ..ProjectMemory::default()
    }
}

/// Estimate tokens as four ASCII characters or one non-ASCII character each.
fn estimate_tokens(text: &str) -> usize {
    let (ascii, non_ascii) = text.chars().fold((0usize, 0usize), |counts, character| {
        if character.is_ascii() {
            (counts.0 + 1, counts.1)
        } else {
            (counts.0, counts.1 + 1)
        }
    });
    ascii.div_ceil(4) + non_ascii
}

enum BoundedRead {
    Contents(String),
    TooLarge,
    Unavailable,
}

/// Read a memory file without allowing an unexpectedly large allocation.
fn read_lossy_bounded(path: &Path, max_bytes: usize) -> BoundedRead {
    let Ok(mut file) = open_without_following_symlinks(path) else {
        return BoundedRead::Unavailable;
    };
    read_lossy_bounded_from(&mut file, max_bytes)
}

#[cfg(unix)]
fn open_without_following_symlinks(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
}

#[cfg(windows)]
fn open_without_following_symlinks(path: &Path) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
}

#[cfg(not(any(unix, windows)))]
fn open_without_following_symlinks(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().read(true).open(path)
}

fn read_lossy_bounded_from(reader: &mut impl Read, max_bytes: usize) -> BoundedRead {
    let mut bytes = Vec::new();
    if reader
        .take(max_bytes.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .is_err()
    {
        return BoundedRead::Unavailable;
    }
    if bytes.len() > max_bytes {
        return BoundedRead::TooLarge;
    }
    BoundedRead::Contents(String::from_utf8_lossy(&bytes).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_only_agents_md_from_the_workspace_root() {
        let temp = tempfile::tempdir().unwrap();
        let child = temp.path().join("project").join("src");
        std::fs::create_dir_all(&child).unwrap();
        std::fs::write(temp.path().join("AGENTS.md"), "ancestor ignored").unwrap();
        std::fs::write(child.join("AGENTS.md"), "cwd instructions").unwrap();
        std::fs::write(child.join("ICODE.md"), "icode ignored").unwrap();

        let loaded = load_project_memory(&child);

        assert_eq!(
            loaded.files,
            [child.join("AGENTS.md").canonicalize().unwrap()]
        );
        assert!(loaded.prompt.starts_with(MEMORY_HEADER));
        assert!(loaded
            .prompt
            .contains("## File: AGENTS.md\n\ncwd instructions"));
        assert!(!loaded.prompt.contains("ancestor ignored"));
        assert!(!loaded.prompt.contains("icode ignored"));
    }

    #[test]
    fn missing_unreadable_and_invalid_utf8_are_nonfatal() {
        let temp = tempfile::tempdir().unwrap();
        assert_eq!(load_project_memory(temp.path()), ProjectMemory::default());

        std::fs::write(temp.path().join("AGENTS.md"), [b'a', 0xff, b'b']).unwrap();
        let loaded = load_project_memory(temp.path());
        assert!(loaded.prompt.contains("a\u{fffd}b"));
    }

    #[test]
    fn read_failures_are_unavailable_not_over_budget() {
        struct FailingReader;

        impl Read for FailingReader {
            fn read(&mut self, _buffer: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("unreadable"))
            }
        }

        assert!(matches!(
            read_lossy_bounded_from(&mut FailingReader, 1024),
            BoundedRead::Unavailable
        ));
    }

    #[test]
    fn skips_the_whole_file_when_the_rendered_budget_is_exceeded() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("AGENTS.md"), "content").unwrap();

        let loaded = load_project_memory_with_limit(
            temp.path(),
            estimate_tokens(MEMORY_HEADER).saturating_sub(1),
        );

        assert!(loaded.prompt.is_empty());
        assert!(loaded.files.is_empty());
        assert!(loaded.warning.unwrap().contains("35,000-token"));
        assert_eq!(estimate_tokens("abcdefgh"), 2);
        assert_eq!(estimate_tokens("项目记忆"), 4);
    }

    #[cfg(unix)]
    #[test]
    fn no_follow_open_rejects_a_symlink() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target.md");
        let link = temp.path().join("AGENTS.md");
        std::fs::write(&target, "target").unwrap();
        symlink("target.md", &link).unwrap();

        assert!(open_without_following_symlinks(&link).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_memory_files() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let workspace = temp.path().join("project");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::write(workspace.join("instructions.md"), "inside").unwrap();
        symlink("instructions.md", workspace.join("AGENTS.md")).unwrap();
        assert_eq!(load_project_memory(&workspace), ProjectMemory::default());

        std::fs::remove_file(workspace.join("AGENTS.md")).unwrap();
        std::fs::write(temp.path().join("outside.md"), "outside").unwrap();
        symlink(temp.path().join("outside.md"), workspace.join("AGENTS.md")).unwrap();
        assert_eq!(load_project_memory(&workspace), ProjectMemory::default());
    }
}
