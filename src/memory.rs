//! Loads project-local reference context from `AGENTS.md`.

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
/// Missing, unreadable and non-regular files are a silent no-op. A symlink is
/// accepted only when its physical target remains inside the workspace. Text
/// decoding is lossy, matching upstream's nonfatal decoding behavior.
pub fn load_project_memory(cwd: &Path) -> ProjectMemory {
    load_project_memory_with_limit(cwd, DEFAULT_MAX_MEMORY_TOKENS)
}

fn load_project_memory_with_limit(cwd: &Path, max_tokens: usize) -> ProjectMemory {
    let Ok(cwd) = cwd.canonicalize() else {
        return ProjectMemory::default();
    };
    if !cwd.is_dir() {
        return ProjectMemory::default();
    }

    let candidate = cwd.join(DEFAULT_MEMORY_FILE);
    let Ok(path) = candidate.canonicalize() else {
        return ProjectMemory::default();
    };
    if !path.starts_with(&cwd) || !path.is_file() {
        return ProjectMemory::default();
    }

    let Some(contents) = read_lossy_bounded(&path, max_tokens.saturating_mul(4)) else {
        return over_budget();
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

fn over_budget() -> ProjectMemory {
    ProjectMemory {
        warning: Some(format!(
            "Project memory {DEFAULT_MEMORY_FILE} was skipped because it exceeds the approximate 35,000-token limit."
        )),
        ..ProjectMemory::default()
    }
}

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

fn read_lossy_bounded(path: &Path, max_bytes: usize) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut bytes = Vec::new();
    file.by_ref()
        .take(max_bytes.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() > max_bytes {
        return None;
    }
    Some(String::from_utf8_lossy(&bytes).into_owned())
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
    fn accepts_contained_symlinks_but_rejects_workspace_escape() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let workspace = temp.path().join("project");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::write(workspace.join("instructions.md"), "inside").unwrap();
        symlink("instructions.md", workspace.join("AGENTS.md")).unwrap();
        assert!(load_project_memory(&workspace).prompt.ends_with("inside"));

        std::fs::remove_file(workspace.join("AGENTS.md")).unwrap();
        std::fs::write(temp.path().join("outside.md"), "outside").unwrap();
        symlink(temp.path().join("outside.md"), workspace.join("AGENTS.md")).unwrap();
        assert_eq!(load_project_memory(&workspace), ProjectMemory::default());
    }
}
