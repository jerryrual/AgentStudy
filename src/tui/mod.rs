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
            discard_pending_events()?;
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

/// Drop keys crossterm parsed beyond the Enter that submitted a direct command.
fn discard_pending_events() -> Result<()> {
    while event::poll(Duration::ZERO).context("failed to inspect buffered terminal input")? {
        event::read().context("failed to discard buffered terminal input")?;
    }
    Ok(())
}

/// Leave the TUI, run one terminal-owning command, and re-enter the TUI.
fn execute_direct_shell(
    terminal: &mut ratatui::DefaultTerminal,
    request: &app::DirectShellRequest,
) -> Result<direct_shell::DirectShellOutcome> {
    let mut replacement = None;
    let outcome = with_terminal_suspended(
        || ratatui::try_restore().context("failed to suspend the terminal for shell command"),
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

/// Run a value-producing operation between fallible leave and enter steps.
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

    #[cfg(unix)]
    #[test]
    fn pty_helper_discards_input_buffered_after_direct_command_enter() {
        if std::env::var_os("ICODE_DIRECT_SHELL_INPUT_HELPER").is_none() {
            return;
        }

        ratatui::crossterm::terminal::enable_raw_mode().unwrap();
        let mut submitted = String::new();
        loop {
            if let Event::Key(key) = event::read().unwrap() {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    ratatui::crossterm::event::KeyCode::Char(character) => {
                        submitted.push(character)
                    }
                    ratatui::crossterm::event::KeyCode::Enter => break,
                    _ => {}
                }
            }
        }
        discard_pending_events().unwrap();
        ratatui::crossterm::terminal::disable_raw_mode().unwrap();
        let command = submitted.strip_prefix('!').unwrap();
        let outcome = direct_shell::run(command, std::path::Path::new("."));
        assert!(
            matches!(outcome, direct_shell::DirectShellOutcome::Exited(code) if code != 0),
            "timed read unexpectedly consumed buffered sensitive input: {outcome:?}"
        );
        ratatui::crossterm::terminal::enable_raw_mode().unwrap();
        assert!(!event::poll(Duration::from_millis(100)).unwrap());
        ratatui::crossterm::terminal::disable_raw_mode().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn direct_shell_handoff_discards_crossterm_read_ahead_on_a_pty() {
        use std::fs::File;
        use std::io::{Read, Write};
        use std::os::fd::{FromRawFd, RawFd};
        use std::os::unix::process::CommandExt;
        use std::process::{Command, Stdio};

        let mut master: RawFd = -1;
        let mut slave: RawFd = -1;
        let opened = unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(
            opened,
            0,
            "openpty failed: {}",
            std::io::Error::last_os_error()
        );
        let mut master = unsafe { File::from_raw_fd(master) };
        let slave = unsafe { File::from_raw_fd(slave) };
        let mut helper = Command::new(std::env::current_exe().unwrap());
        helper
            .arg("--exact")
            .arg("tui::tests::pty_helper_discards_input_buffered_after_direct_command_enter")
            .env("ICODE_DIRECT_SHELL_INPUT_HELPER", "1")
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave.try_clone().unwrap()))
            .stderr(Stdio::from(slave));
        unsafe {
            helper.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::ioctl(libc::STDIN_FILENO, libc::TIOCSCTTY, 0) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::tcsetpgrp(libc::STDIN_FILENO, libc::getpgrp()) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }

        let mut child = helper.spawn().unwrap();
        drop(helper);
        std::thread::sleep(Duration::from_millis(100));
        master
            .write_all(b"!read -t 1 -r secret\rSENSITIVE-VALUE\r")
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if std::time::Instant::now() >= deadline {
                unsafe {
                    libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL);
                }
                let _ = child.wait();
                panic!("PTY input-isolation helper did not finish");
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let mut output = Vec::new();
        let _ = master.read_to_end(&mut output);
        assert!(
            status.success(),
            "PTY helper failed with {status}: {}",
            String::from_utf8_lossy(&output)
        );
    }

    #[test]
    fn suspended_terminal_reenters_after_every_runner_outcome() {
        for outcome in [
            direct_shell::DirectShellOutcome::Exited(0),
            direct_shell::DirectShellOutcome::Exited(9),
            direct_shell::DirectShellOutcome::Signaled(2),
            direct_shell::DirectShellOutcome::Stopped(20),
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
    fn leave_failure_does_not_run_command_or_reenter() {
        let ran = std::cell::Cell::new(false);
        let entered = std::cell::Cell::new(false);
        let error = with_terminal_suspended(
            || anyhow::bail!("cannot leave"),
            || ran.set(true),
            || {
                entered.set(true);
                Ok(())
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("cannot leave"));
        assert!(!ran.get());
        assert!(!entered.get());
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
