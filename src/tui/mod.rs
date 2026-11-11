//! The terminal user interface.

pub mod app;
mod direct_shell;
pub mod ui;

pub use app::App;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use ratatui::crossterm::event::{self, Event, KeyEventKind};

use crate::agent::AgentKind;
use crate::config::{ApprovalMode, ModelProfile, Settings};
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
    /// Effective approval mode (may come from `--approval`).
    pub approval: ApprovalMode,
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
            approval: options.approval,
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
                    let size = terminal
                        .size()
                        .context("failed to read the terminal size")?;
                    app.handle_key(
                        key,
                        ratatui::layout::Rect::new(0, 0, size.width, size.height),
                    );
                }
            }
        }
        if let Some(request) = app.take_direct_shell() {
            let outcome = execute_direct_shell(terminal, &request)?;
            app.complete_direct_shell(
                request.id,
                &request.command,
                request.started.elapsed(),
                outcome,
            );
            continue;
        }
        app.pump();
        app.tick();
        if app.quit {
            break;
        }
    }
    Ok(())
}

fn execute_direct_shell(
    terminal: &mut ratatui::DefaultTerminal,
    request: &app::DirectShellRequest,
) -> Result<direct_shell::DirectShellOutcome> {
    let mut replacement = None;
    let outcome = with_terminal_suspended(
        || {
            ratatui::restore();
            Ok(())
        },
        || direct_shell::run(&request.command, &request.cwd),
        || {
            replacement = Some(
                ratatui::try_init()
                    .context("failed to restore the terminal after shell command")?,
            );
            Ok(())
        },
    )?;
    *terminal = replacement.expect("terminal resume succeeded without replacement");
    Ok(outcome)
}

fn with_terminal_suspended<T>(
    leave: impl FnOnce() -> Result<()>,
    run: impl FnOnce() -> T,
    enter: impl FnOnce() -> Result<()>,
) -> Result<T> {
    leave()?;
    let result = run();
    enter()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    #[test]
    fn suspended_terminal_reenters_after_every_runner_outcome() {
        for outcome in [
            direct_shell::DirectShellOutcome::Exited(0),
            direct_shell::DirectShellOutcome::Exited(9),
            direct_shell::DirectShellOutcome::Signaled(2),
            direct_shell::DirectShellOutcome::Error("spawn failed".to_string()),
        ] {
            let order = RefCell::new(Vec::new());
            let actual = with_terminal_suspended(
                || {
                    order.borrow_mut().push("leave");
                    Ok(())
                },
                || {
                    order.borrow_mut().push("run");
                    outcome.clone()
                },
                || {
                    order.borrow_mut().push("enter");
                    Ok(())
                },
            )
            .unwrap();

            assert_eq!(actual, outcome);
            assert_eq!(&*order.borrow(), &["leave", "run", "enter"]);
        }
    }

    #[test]
    fn resume_failure_is_reported_after_runner_finishes() {
        let order = RefCell::new(Vec::new());
        let error = with_terminal_suspended(
            || {
                order.borrow_mut().push("leave");
                Ok(())
            },
            || order.borrow_mut().push("run"),
            || {
                order.borrow_mut().push("enter");
                anyhow::bail!("cannot enter")
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("cannot enter"));
        assert_eq!(&*order.borrow(), &["leave", "run", "enter"]);
    }
}
