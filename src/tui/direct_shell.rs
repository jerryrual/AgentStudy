//! Foreground terminal execution for direct TUI `!command` requests.

use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};

/// Structured result shown on the direct-command tool card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DirectShellOutcome {
    Exited(i32),
    Signaled(i32),
    Error(String),
}

impl DirectShellOutcome {
    pub(crate) fn succeeded(&self) -> bool {
        matches!(self, Self::Exited(0))
    }

    pub(crate) fn summary(&self) -> String {
        match self {
            Self::Exited(code) => format!("Exited with code {code}."),
            Self::Signaled(signal) => format!("Terminated by signal {signal}."),
            Self::Error(error) => error.clone(),
        }
    }
}

/// Run `bash -c` with inherited terminal streams and wait for completion.
pub(crate) fn run(command: &str, cwd: &Path) -> DirectShellOutcome {
    let mut child_command = Command::new("bash");
    child_command
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    configure_child(&mut child_command);

    let mut child = match child_command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return DirectShellOutcome::Error(format!("Failed to start shell command: {error}"));
        }
    };

    #[cfg(unix)]
    let foreground = match ForegroundLease::acquire(child.id()) {
        Ok(foreground) => foreground,
        Err(error) => {
            terminate_process_group(child.id());
            let _ = child.wait();
            return DirectShellOutcome::Error(format!(
                "Failed to give the command terminal control: {error}"
            ));
        }
    };

    let status = child.wait();

    #[cfg(unix)]
    if let Some(foreground) = foreground {
        if let Err(error) = foreground.restore() {
            return DirectShellOutcome::Error(format!(
                "Failed to restore terminal control to icode: {error}"
            ));
        }
    }

    match status {
        Ok(status) => classify_status(status),
        Err(error) => {
            DirectShellOutcome::Error(format!("Failed to wait for shell command: {error}"))
        }
    }
}

#[cfg(unix)]
fn configure_child(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    unsafe {
        command.pre_exec(|| {
            if libc::setpgid(0, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn configure_child(_command: &mut Command) {}

fn classify_status(status: ExitStatus) -> DirectShellOutcome {
    if let Some(code) = status.code() {
        return DirectShellOutcome::Exited(code);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return DirectShellOutcome::Signaled(signal);
        }
    }
    DirectShellOutcome::Error("Shell command ended without an exit code or signal.".to_string())
}

#[cfg(unix)]
struct ForegroundLease {
    terminal: libc::c_int,
    parent_group: libc::pid_t,
    active: bool,
}

#[cfg(unix)]
impl ForegroundLease {
    fn acquire(child_id: u32) -> std::io::Result<Option<Self>> {
        let terminal = libc::STDIN_FILENO;
        if unsafe { libc::isatty(terminal) } != 1 {
            return Ok(None);
        }
        let parent_group = unsafe { libc::tcgetpgrp(terminal) };
        if parent_group == -1 {
            return Err(std::io::Error::last_os_error());
        }
        set_foreground_group(terminal, child_id as libc::pid_t)?;
        Ok(Some(Self {
            terminal,
            parent_group,
            active: true,
        }))
    }

    fn restore(mut self) -> std::io::Result<()> {
        let result = set_foreground_group(self.terminal, self.parent_group);
        self.active = false;
        result
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

#[cfg(unix)]
fn terminate_process_group(child_id: u32) {
    unsafe {
        libc::kill(-(child_id as libc::pid_t), libc::SIGKILL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_in_requested_directory_without_capturing_output() {
        let directory = tempfile::tempdir().unwrap();
        let outcome = run("pwd > direct-shell-cwd", directory.path());

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
        assert_eq!(run("exit 7", Path::new(".")), DirectShellOutcome::Exited(7));
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
        let outcome = run(
            "test \"$(ps -o tpgid= -p $$ | tr -d ' ')\" = \"$$\"",
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
        let master = unsafe { File::from_raw_fd(master) };
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

        let status = helper.spawn().unwrap().wait().unwrap();
        drop(master);
        assert!(status.success(), "PTY helper failed with {status}");
    }

    #[cfg(unix)]
    #[test]
    fn classifies_signal_exit() {
        assert_eq!(
            run("kill -TERM $$", Path::new(".")),
            DirectShellOutcome::Signaled(libc::SIGTERM)
        );
    }
}
