//! Command-line interface.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

use crate::agent::AgentKind;
use crate::config::{ApprovalMode, Settings};
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
        /// Emit one JSON object instead of streamed text.
        #[arg(long)]
        json: bool,
        /// Continue an existing session by id.
        #[arg(long)]
        session: Option<String>,
    },
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
        Some(Command::Models) => run::list_models(&settings),
        Some(Command::Run {
            task,
            json,
            session,
        }) => {
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
            runtime.block_on(run::execute(RunOptions {
                settings,
                profile,
                agent,
                cwd,
                task,
                approval,
                json,
                session,
                store,
            }))
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
