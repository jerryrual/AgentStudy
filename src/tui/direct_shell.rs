//! Foreground terminal execution for direct TUI `!command` requests.

use std::path::Path;
use std::process::{Command, Stdio};

/// Structured result shown on the direct-command tool card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DirectShellOutcome {
    Exited(i32),
    Signaled(i32),
    Stopped(i32),
    Error(String),
}

impl DirectShellOutcome {
    /// Whether the command exited successfully.
    pub(crate) fn succeeded(&self) -> bool {
        matches!(self, Self::Exited(0))
    }

    /// Render the bounded status retained by the TUI card.
    pub(crate) fn summary(&self) -> String {
        match self {
            Self::Exited(code) => format!("Exited with code {code}."),
            Self::Signaled(signal) => format!("Terminated by signal {signal}."),
            Self::Stopped(signal) => format!("Stopped by signal {signal}."),
            Self::Error(error) => error.clone(),
        }
    }
}

/// Run `bash -c` with inherited terminal streams and wait for completion.
pub(crate) fn run(command: &str, cwd: &Path) -> DirectShellOutcome {
    #[cfg(unix)]
    let foreground = match ForegroundLease::prepare() {
        Ok(foreground) => foreground,
        Err(error) => {
            return DirectShellOutcome::Error(format!(
                "Failed to prepare terminal control for the command: {error}"
            ));
        }
    };

    let mut child_command = Command::new("bash");
    child_command
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    #[cfg(unix)]
    configure_child(&mut child_command, foreground.is_some());
    #[cfg(not(unix))]
    configure_child(&mut child_command, false);

    let mut child = match child_command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return DirectShellOutcome::Error(format!("Failed to start shell command: {error}"));
        }
    };

    #[cfg(unix)]
    let outcome = wait_for_child(&mut child);
    #[cfg(not(unix))]
    let outcome = match child.wait() {
        Ok(status) => classify_status(status),
        Err(error) => {
            DirectShellOutcome::Error(format!("Failed to wait for shell command: {error}"))
        }
    };

    #[cfg(unix)]
    if let Some(foreground) = foreground {
        if let Err(error) = foreground.restore() {
            return DirectShellOutcome::Error(format!(
                "Failed to restore terminal control to icode: {error}"
            ));
        }
    }

    outcome
}

/// Put the child in a dedicated process group before it executes `bash`.
#[cfg(unix)]
fn configure_child(command: &mut Command, claim_terminal: bool) {
    use std::os::unix::process::CommandExt;

    unsafe {
        command.pre_exec(move || {
            if libc::setpgid(0, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            if claim_terminal {
                claim_foreground_in_child(libc::STDIN_FILENO)?;
            }
            Ok(())
        });
    }
}

/// Keep the inherited-stream setup unchanged where Unix job control is absent.
#[cfg(not(unix))]
fn configure_child(_command: &mut Command, _claim_terminal: bool) {}

/// Give the post-fork child terminal control before it can read from stdin.
#[cfg(unix)]
fn claim_foreground_in_child(terminal: libc::c_int) -> std::io::Result<()> {
    unsafe {
        let mut blocked = std::mem::zeroed::<libc::sigset_t>();
        let mut previous = std::mem::zeroed::<libc::sigset_t>();
        if libc::sigemptyset(&mut blocked) == -1
            || libc::sigaddset(&mut blocked, libc::SIGTTOU) == -1
            || libc::sigprocmask(libc::SIG_BLOCK, &blocked, &mut previous) == -1
        {
            return Err(std::io::Error::last_os_error());
        }
        let changed = libc::tcsetpgrp(terminal, libc::getpgrp());
        let change_error = (changed == -1).then(std::io::Error::last_os_error);
        let restore_mask = libc::sigprocmask(libc::SIG_SETMASK, &previous, std::ptr::null_mut());
        if let Some(error) = change_error {
            return Err(error);
        }
        if restore_mask == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

/// Convert a non-Unix process status into the direct-shell result.
#[cfg(not(unix))]
fn classify_status(status: std::process::ExitStatus) -> DirectShellOutcome {
    status.code().map_or_else(
        || {
            DirectShellOutcome::Error(
                "Shell command ended without an exit code or signal.".to_string(),
            )
        },
        DirectShellOutcome::Exited,
    )
}

/// Wait for exit, termination, or suspension of the foreground child.
#[cfg(unix)]
fn wait_for_child(child: &mut std::process::Child) -> DirectShellOutcome {
    let pid = child.id() as libc::pid_t;
    loop {
        let mut status = 0;
        let waited = unsafe { libc::waitpid(pid, &mut status, libc::WUNTRACED | libc::WCONTINUED) };
        if waited == -1 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return DirectShellOutcome::Error(format!("Failed to wait for shell command: {error}"));
        }
        if libc::WIFEXITED(status) {
            return DirectShellOutcome::Exited(libc::WEXITSTATUS(status));
        }
        if libc::WIFSIGNALED(status) {
            return DirectShellOutcome::Signaled(libc::WTERMSIG(status));
        }
        if libc::WIFSTOPPED(status) {
            let signal = libc::WSTOPSIG(status);
            terminate_process_group(child.id());
            if let Err(error) = reap_child(pid) {
                return DirectShellOutcome::Error(format!(
                    "Failed to reap stopped shell command: {error}"
                ));
            }
            return DirectShellOutcome::Stopped(signal);
        }
    }
}

/// Reap a child killed after it reported a stopped status.
#[cfg(unix)]
fn reap_child(pid: libc::pid_t) -> std::io::Result<()> {
    loop {
        let waited = unsafe { libc::waitpid(pid, std::ptr::null_mut(), 0) };
        if waited != -1 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

/// Restores the original foreground process group on every return path.
#[cfg(unix)]
struct ForegroundLease {
    terminal: libc::c_int,
    parent_group: libc::pid_t,
    active: bool,
}

#[cfg(unix)]
impl ForegroundLease {
    /// Capture the TUI process group before the child takes terminal control.
    fn prepare() -> std::io::Result<Option<Self>> {
        let terminal = libc::STDIN_FILENO;
        if unsafe { libc::isatty(terminal) } != 1 {
            return Ok(None);
        }
        let parent_group = unsafe { libc::tcgetpgrp(terminal) };
        if parent_group == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Some(Self {
            terminal,
            parent_group,
            active: true,
        }))
    }

    /// Restore terminal ownership, leaving Drop armed if the attempt fails.
    fn restore(mut self) -> std::io::Result<()> {
        set_foreground_group(self.terminal, self.parent_group)?;
        self.active = false;
        Ok(())
    }
}

#[cfg(unix)]
impl Drop for ForegroundLease {
    fn drop(&mut self) {
        if self.active {
            let _ = set_foreground_group(self.terminal, self.parent_group);
        }
    }
}

/// Change terminal ownership without allowing SIGTTOU to stop the caller.
#[cfg(unix)]
fn set_foreground_group(terminal: libc::c_int, group: libc::pid_t) -> std::io::Result<()> {
    unsafe {
        let mut blocked = std::mem::zeroed::<libc::sigset_t>();
        let mut previous = std::mem::zeroed::<libc::sigset_t>();
        if libc::sigemptyset(&mut blocked) == -1
            || libc::sigaddset(&mut blocked, libc::SIGTTOU) == -1
        {
            return Err(std::io::Error::last_os_error());
        }
        let block_result = libc::pthread_sigmask(libc::SIG_BLOCK, &blocked, &mut previous);
        if block_result != 0 {
            return Err(std::io::Error::from_raw_os_error(block_result));
        }
        let changed = libc::tcsetpgrp(terminal, group);
        let change_error = (changed == -1).then(std::io::Error::last_os_error);
        let restore_mask =
            libc::pthread_sigmask(libc::SIG_SETMASK, &previous, std::ptr::null_mut());
        if let Some(error) = change_error {
            return Err(error);
        }
        if restore_mask != 0 {
            return Err(std::io::Error::from_raw_os_error(restore_mask));
        }
        Ok(())
    }
}

/// Kill every process in the direct command's foreground group.
#[cfg(unix)]
fn terminate_process_group(child_id: u32) {
    unsafe {
        libc::kill(-(child_id as libc::pid_t), libc::SIGKILL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_serial(command: &str, cwd: &Path) -> DirectShellOutcome {
        static TERMINAL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = TERMINAL_LOCK.lock().unwrap();
        run(command, cwd)
    }

    #[test]
    fn runs_in_requested_directory_without_capturing_output() {
        let directory = tempfile::tempdir().unwrap();
        let outcome = run_serial("pwd > direct-shell-cwd", directory.path());

        assert_eq!(outcome, DirectShellOutcome::Exited(0));
        assert_eq!(
            std::fs::read_to_string(directory.path().join("direct-shell-cwd"))
                .unwrap()
                .trim(),
            directory.path().to_string_lossy()
        );
    }

    #[test]
    fn classifies_nonzero_exit() {
        assert_eq!(
            run_serial("exit 7", Path::new(".")),
            DirectShellOutcome::Exited(7)
        );
    }

    #[cfg(unix)]
    #[test]
    fn pty_helper_checks_foreground_transfer_and_restoration() {
        if std::env::var_os("ICODE_DIRECT_SHELL_PTY_HELPER").is_none() {
            return;
        }

        assert_eq!(unsafe { libc::tcgetpgrp(libc::STDIN_FILENO) }, unsafe {
            libc::getpgrp()
        });
        let outcome = run_serial(
            "IFS= read -r value && test \"$value\" = ready",
            Path::new("."),
        );
        assert_eq!(outcome, DirectShellOutcome::Exited(0));
        assert_eq!(unsafe { libc::tcgetpgrp(libc::STDIN_FILENO) }, unsafe {
            libc::getpgrp()
        });
    }

    #[cfg(unix)]
    #[test]
    fn transfers_and_restores_foreground_process_group_on_a_pty() {
        use std::fs::File;
        use std::io::Write;
        use std::os::fd::{FromRawFd, RawFd};
        use std::os::unix::process::CommandExt;

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
            .arg("tui::direct_shell::tests::pty_helper_checks_foreground_transfer_and_restoration")
            .env("ICODE_DIRECT_SHELL_PTY_HELPER", "1")
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

        let mut helper = helper.spawn().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));
        master.write_all(b"ready\n").unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let status = loop {
            if let Some(status) = helper.try_wait().unwrap() {
                break status;
            }
            if std::time::Instant::now() >= deadline {
                unsafe {
                    libc::kill(-(helper.id() as libc::pid_t), libc::SIGKILL);
                }
                let _ = helper.wait();
                panic!("PTY helper did not finish after receiving input");
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        };
        drop(master);
        assert!(status.success(), "PTY helper failed with {status}");
    }

    #[cfg(unix)]
    #[test]
    fn stops_and_reaps_a_suspended_command() {
        assert_eq!(
            run_serial("kill -STOP $$", Path::new(".")),
            DirectShellOutcome::Stopped(libc::SIGSTOP)
        );
    }

    #[cfg(unix)]
    #[test]
    fn classifies_signal_exit() {
        assert_eq!(
            run_serial("kill -TERM $$", Path::new(".")),
            DirectShellOutcome::Signaled(libc::SIGTERM)
        );
    }
}
