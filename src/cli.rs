//! Command-line interface.

use std::io::Read as _;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

use crate::agent::AgentKind;
use crate::config::{ApprovalMode, Settings};
use crate::model::types::{MAX_IMAGE_BYTES, MAX_PROMPT_IMAGES, MAX_PROMPT_IMAGE_BYTES};
use crate::model::{ImageInput, UserContent, UserContentPart};
use crate::run::{self, RunOptions};
use crate::session::SessionStore;
use crate::theme::Theme;
use crate::tui::{self, TuiOptions};

#[derive(Debug, Parser)]
#[command(
    name = "icode",
    version,
    about = "iCode — a terminal coding agent, implemented in Rust",
    long_about = "A Rust implementation of the iCode terminal coding agent. Run without a \
                  subcommand for the TUI, or use `icode run` for headless execution."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Model profile to use (see `icode models`).
    #[arg(long, global = true)]
    pub model: Option<String>,

    /// Agent to run: code or qa.
    #[arg(long, global = true)]
    pub agent: Option<String>,

    /// Approval policy for mutating tools: ask, auto or deny.
    #[arg(long, global = true, value_enum)]
    pub approval: Option<ApprovalArg>,

    /// Working directory (default: the current directory).
    #[arg(long, global = true)]
    pub cwd: Option<PathBuf>,

    /// Continue the most recent session.
    #[arg(long = "continue", global = true)]
    pub continue_session: bool,

    /// Maximum model calls per turn.
    #[arg(long, global = true)]
    pub max_rounds: Option<usize>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run a task without the TUI.
    Run {
        /// Task prompt, e.g. "explain this project".
        task: String,
        /// Local image file to attach (repeatable; resolved relative to --cwd).
        #[arg(long = "image", value_name = "FILE")]
        images: Vec<PathBuf>,
        /// Emit one JSON object instead of streamed text.
        #[arg(long)]
        json: bool,
        /// Continue an existing session by id.
        #[arg(long)]
        session: Option<String>,
    },
    /// Resume an interrupted user turn under an exclusive session lease.
    /// Mutating tools are denied because prior unrecorded effects are uncertain.
    ResumeTurn {
        /// Emit one JSON object instead of streamed text.
        #[arg(long)]
        json: bool,
        /// Session to resume; use global --continue for the latest session.
        #[arg(long)]
        session: Option<String>,
    },
    /// Serve the Agent Client Protocol over stdin/stdout.
    Acp,
    /// List the configured model profiles.
    Models,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ApprovalArg {
    Ask,
    Auto,
    Deny,
}

impl From<ApprovalArg> for ApprovalMode {
    fn from(value: ApprovalArg) -> Self {
        match value {
            ApprovalArg::Ask => ApprovalMode::Ask,
            ApprovalArg::Auto => ApprovalMode::Auto,
            ApprovalArg::Deny => ApprovalMode::Deny,
        }
    }
}

/// Parse arguments and execute the requested command.
pub fn execute(cli: Cli) -> Result<()> {
    let mut settings = Settings::load().context("failed to load settings")?;
    let cwd = resolve_cwd(cli.cwd)?;
    let agent = match &cli.agent {
        Some(agent) => AgentKind::parse(agent)
            .with_context(|| format!("unknown agent {agent:?}; expected \"code\" or \"qa\""))?,
        None => AgentKind::parse(&settings.agent).unwrap_or(AgentKind::Code),
    };
    let approval = cli
        .approval
        .map(ApprovalMode::from)
        .unwrap_or(settings.approval_mode);
    if let Some(max_rounds) = cli.max_rounds {
        if max_rounds == 0 {
            bail!("--max-rounds must be greater than zero");
        }
        settings.max_tool_rounds = max_rounds;
    }
    let profile = settings
        .profile(cli.model.as_deref())
        .context("failed to select a model")?;
    let store = SessionStore::open();

    match cli.command {
        Some(Command::ResumeTurn { json, session }) => {
            let session = match session {
                Some(id) => id,
                None if cli.continue_session => store
                    .latest()
                    .context("failed to find the latest session")?
                    .context("no session is available to resume")?,
                None => bail!("resume-turn requires --session <id> or --continue"),
            };
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("failed to start the async runtime")?;
            runtime.block_on(run::execute(RunOptions {
                settings,
                profile,
                agent,
                cwd,
                task: String::new(),
                approval,
                json,
                session: Some(session),
                resume_turn: true,
                store,
            }))
        }
        Some(Command::Acp) => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("failed to start the async runtime")?;
            runtime.block_on(crate::acp::serve_stdio(settings, profile, agent, store))
        }
        Some(Command::Models) => run::list_models(&settings),
        Some(Command::Run {
            task,
            images,
            json,
            session,
        }) => {
            let content = build_user_content(&task, &images, &cwd, profile.vision)?;
            let session = match session {
                Some(id) => Some(id),
                None if cli.continue_session => store
                    .latest()
                    .context("failed to find the latest session")?,
                None => None,
            };
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("failed to start the async runtime")?;
            runtime.block_on(run::execute_with_content(
                RunOptions {
                    settings,
                    profile,
                    agent,
                    cwd,
                    task,
                    approval,
                    json,
                    session,
                    resume_turn: false,
                    store,
                },
                content,
            ))
        }
        None => {
            let resume = if cli.continue_session {
                store
                    .latest()
                    .context("failed to find the latest session")?
            } else {
                None
            };
            tui::run(TuiOptions {
                settings,
                profile,
                agent,
                cwd,
                resume,
                store,
                theme: Theme::default(),
                approval,
            })
        }
    }
}

fn build_user_content(
    task: &str,
    images: &[PathBuf],
    cwd: &std::path::Path,
    vision: bool,
) -> Result<UserContent> {
    let task = task.trim().to_string();
    if images.is_empty() {
        return Ok(UserContent::Text(task));
    }
    if !vision {
        bail!("the selected model profile does not enable vision image input");
    }
    if images.len() > MAX_PROMPT_IMAGES {
        bail!("prompt exceeds the {MAX_PROMPT_IMAGES}-image limit");
    }

    let mut parts = Vec::with_capacity(images.len() + 1);
    parts.push(UserContentPart::Text { text: task });
    let mut image_bytes = 0usize;
    for path in images {
        let resolved = if path.is_absolute() {
            path.clone()
        } else {
            cwd.join(path)
        };
        let file = open_local_image(&resolved)
            .with_context(|| format!("failed to open image {}", resolved.display()))?;
        let metadata = file
            .metadata()
            .with_context(|| format!("failed to inspect image {}", resolved.display()))?;
        if !metadata.is_file() {
            bail!("image path is not a regular file: {}", resolved.display());
        }
        if metadata.len() > MAX_IMAGE_BYTES as u64 {
            bail!(
                "image {} exceeds the {MAX_IMAGE_BYTES}-byte decoded limit",
                resolved.display()
            );
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take((MAX_IMAGE_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .with_context(|| format!("failed to read image {}", resolved.display()))?;
        let remaining_bytes = MAX_PROMPT_IMAGE_BYTES.saturating_sub(image_bytes);
        let (image, decoded_bytes) = ImageInput::validated_bytes(bytes, remaining_bytes)
            .with_context(|| format!("invalid image {}", resolved.display()))?;
        image_bytes += decoded_bytes;
        parts.push(UserContentPart::Image { image });
    }
    Ok(UserContent::parts(parts))
}

#[cfg(unix)]
fn open_local_image(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;

    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)
}

#[cfg(not(unix))]
fn open_local_image(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    std::fs::File::open(path)
}

fn resolve_cwd(cwd: Option<PathBuf>) -> Result<PathBuf> {
    let cwd = match cwd {
        Some(cwd) => cwd,
        None => std::env::current_dir().context("cannot determine the current directory")?,
    };
    let cwd = cwd
        .canonicalize()
        .with_context(|| format!("working directory {} does not exist", cwd.display()))?;
    if !cwd.is_dir() {
        bail!("{} is not a directory", cwd.display());
    }
    Ok(cwd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_acp_stdio_subcommand() {
        let cli = Cli::try_parse_from(["icode", "acp"]).unwrap();
        assert!(matches!(cli.command, Some(Command::Acp)));
    }

    #[test]
    fn parses_ordered_run_image_flags() {
        let cli = Cli::try_parse_from([
            "icode", "run", "describe", "--image", "one.png", "--image", "two.webp",
        ])
        .unwrap();
        let Some(Command::Run { images, .. }) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(
            images,
            vec![PathBuf::from("one.png"), PathBuf::from("two.webp")]
        );
    }

    #[test]
    fn text_only_profile_rejects_images_before_reading_paths() {
        let directory = tempfile::tempdir().unwrap();
        let error = build_user_content(
            "describe",
            &[PathBuf::from("missing.png")],
            directory.path(),
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("does not enable vision"));
    }

    #[test]
    fn image_count_limit_precedes_path_reads() {
        let directory = tempfile::tempdir().unwrap();
        let images = (0..=MAX_PROMPT_IMAGES)
            .map(|index| PathBuf::from(format!("missing-{index}.png")))
            .collect::<Vec<_>>();
        let error = build_user_content("describe", &images, directory.path(), true).unwrap_err();
        assert!(error.to_string().contains("image limit"));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_to_regular_image_is_accepted() {
        use std::io::Cursor;
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.png");
        let mut bytes = Cursor::new(Vec::new());
        image::DynamicImage::new_rgba8(1, 1)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        std::fs::write(&target, bytes.into_inner()).unwrap();
        let link = directory.path().join("linked.png");
        symlink(target, &link).unwrap();

        let content = build_user_content("describe", &[link], directory.path(), true).unwrap();
        assert!(content.has_images());
    }

    #[cfg(unix)]
    #[test]
    fn image_fifo_is_rejected_without_reading() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let directory = tempfile::tempdir().unwrap();
        let fifo = directory.path().join("image.png");
        let path = CString::new(fifo.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);

        let error = build_user_content("describe", &[fifo], directory.path(), true).unwrap_err();
        assert!(error.to_string().contains("not a regular file"));
    }
}
