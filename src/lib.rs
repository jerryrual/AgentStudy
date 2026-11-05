//! icode — a Rust implementation of the iCode terminal coding agent.
//!
//! The crate is organised in layers:
//!
//! * [`config`] — model profiles and user settings.
//! * [`model`] — an OpenAI-compatible chat-completions client with SSE streaming.
//! * [`agent`] — the tool-calling loop that drives a conversation.
//! * [`tools`] — the built-in tool set (file editing, search, shell, todos).
//! * [`session`] — append-only JSONL conversation sessions.
//! * [`tui`] — the ratatui terminal user interface.
//! * [`run`] — headless, non-interactive execution (`icode run`).

pub mod agent;
pub mod cli;
pub mod compaction;
pub mod config;
pub mod context_budget;
pub mod interaction;
pub mod markdown;
pub mod memory;
pub mod model;
pub mod run;
pub mod session;
pub mod theme;
pub mod tools;
pub mod tui;

/// The user agent name reported to the model and persisted in sessions.
pub const APP_NAME: &str = "icode";

/// Version of the Rust port, shown in the TUI and `icode --version`.
pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
