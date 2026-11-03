//! The terminal user interface.

pub mod app;
pub mod ui;

pub use app::App;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use ratatui::crossterm::event::{self, Event, KeyEventKind};

use crate::agent::AgentKind;
use crate::config::{ModelProfile, Settings};
use crate::session::SessionStore;
use crate::theme::Theme;

/// Inputs needed to start the TUI.
pub struct TuiOptions {
    pub settings: Settings,
    pub profile: ModelProfile,
    pub agent: AgentKind,
    pub cwd: PathBuf,
    pub resume: Option<String>,
    pub store: SessionStore,
    pub theme: Theme,
}

/// Run the TUI until the user quits.
pub fn run(options: TuiOptions) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to start the async runtime")?;
    let mut app = App::new(
        runtime.handle().clone(),
        app::AppOptions {
            settings: options.settings,
            profile: options.profile,
            agent: options.agent,
            cwd: options.cwd,
            store: options.store,
            resume: options.resume,
            theme: options.theme,
        },
    )?;
    let mut terminal = ratatui::try_init().context("failed to initialize the terminal")?;
    let result = event_loop(&mut terminal, &mut app);
    app.cancel();
    ratatui::restore();
    runtime.shutdown_timeout(Duration::from_millis(500));
    result
}

fn event_loop(terminal: &mut ratatui::DefaultTerminal, app: &mut App) -> Result<()> {
    loop {
        terminal
            .draw(|frame| ui::draw(frame, app))
            .context("failed to draw the interface")?;
        if event::poll(Duration::from_millis(40)).context("failed to poll for terminal input")? {
            if let Event::Key(key) = event::read().context("failed to read terminal input")? {
                if key.kind == KeyEventKind::Press {
                    app.handle_key(key);
                }
            }
        }
        app.pump();
        app.tick();
        if app.quit {
            break;
        }
    }
    Ok(())
}
