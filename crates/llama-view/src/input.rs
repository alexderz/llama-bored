//! Reading the keyboard without a line discipline (#27): `q` / Ctrl-C quit,
//! and focus reports arrive as soon as they are sent.

use std::io;
use std::os::fd::{AsFd, BorrowedFd};
use std::time::Duration;

use rustix::event::{PollFd, PollFlags, Timespec, poll};
use rustix::termios::{
    LocalModes, OptionalActions, SpecialCodeIndex, Termios, isatty, tcgetattr, tcsetattr,
};

/// What [`wait_input`] saw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Wait {
    /// The timeout passed with nothing to read.
    Timeout,
    /// Bytes are ready.
    Readable,
    /// The terminal hung up or stdin was closed.
    Closed,
}

/// Wait up to `timeout` for `fd` to be readable. This is also the frame
/// timer: a key or a focus report ends the wait early.
pub fn wait_input(fd: BorrowedFd<'_>, timeout: Duration) -> io::Result<Wait> {
    let spec = Timespec {
        tv_sec: i64::try_from(timeout.as_secs()).unwrap_or(i64::MAX),
        tv_nsec: i64::from(timeout.subsec_nanos()),
    };
    let mut fds = [PollFd::new(&fd, PollFlags::IN)];
    let n = match poll(&mut fds, Some(&spec)) {
        Ok(n) => n,
        Err(rustix::io::Errno::INTR) => 0,
        Err(err) => return Err(err.into()),
    };
    if n == 0 {
        return Ok(Wait::Timeout);
    }
    let revents = fds[0].revents();
    if revents.contains(PollFlags::IN) {
        Ok(Wait::Readable)
    } else if revents.intersects(PollFlags::HUP | PollFlags::ERR | PollFlags::NVAL) {
        Ok(Wait::Closed)
    } else {
        Ok(Wait::Timeout)
    }
}

/// The terminal settings to put back on exit: the ones found at start, with
/// canonical input, echo, signals and extended input forced on. Forcing them
/// heals a terminal that an earlier llama-view left raw when it was killed
/// (SIGTERM cannot be caught without `unsafe`).
pub fn restored(mut found: Termios) -> Termios {
    found.local_modes |=
        LocalModes::ICANON | LocalModes::ECHO | LocalModes::ISIG | LocalModes::IEXTEN;
    found
}

/// Non-canonical, no echo, no signal keys (Ctrl-C arrives as a byte and
/// quits cleanly), reads return as soon as one byte is there.
pub fn raw(mut found: Termios) -> Termios {
    found
        .local_modes
        .remove(LocalModes::ICANON | LocalModes::ECHO | LocalModes::ISIG | LocalModes::IEXTEN);
    found.special_codes[SpecialCodeIndex::VMIN] = 1;
    found.special_codes[SpecialCodeIndex::VTIME] = 0;
    found
}

/// stdin in raw mode for as long as this lives. [`RawInput::enter`] returns
/// `None` when stdin is not a terminal (a pipe, `/dev/null`, a test).
pub struct RawInput<Fd: AsFd> {
    fd: Fd,
    saved: Termios,
    restored: bool,
}

impl<Fd: AsFd> RawInput<Fd> {
    pub fn enter(fd: Fd) -> io::Result<Option<Self>> {
        if !isatty(&fd) {
            return Ok(None);
        }
        let saved = tcgetattr(&fd)?;
        tcsetattr(&fd, OptionalActions::Now, &raw(saved.clone()))?;
        Ok(Some(Self {
            fd,
            saved,
            restored: false,
        }))
    }

    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    /// Put the settings back (see [`restored`]). Idempotent.
    pub fn restore(&mut self) -> io::Result<()> {
        if self.restored {
            return Ok(());
        }
        self.restored = true;
        tcsetattr(
            &self.fd,
            OptionalActions::Now,
            &restored(self.saved.clone()),
        )?;
        Ok(())
    }
}

impl<Fd: AsFd> Drop for RawInput<Fd> {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn some_termios() -> Option<Termios> {
        // Any tty will do for the flag arithmetic; skip when there is none.
        let tty = std::fs::File::open("/dev/tty").ok()?;
        tcgetattr(&tty).ok()
    }

    #[test]
    fn raw_turns_off_the_line_discipline_and_restored_turns_it_on() {
        let Some(found) = some_termios() else {
            eprintln!("no controlling tty; skipped");
            return;
        };
        let r = raw(found.clone());
        assert!(!r.local_modes.intersects(
            LocalModes::ICANON | LocalModes::ECHO | LocalModes::ISIG | LocalModes::IEXTEN
        ));
        assert_eq!(r.special_codes[SpecialCodeIndex::VMIN], 1);
        assert_eq!(r.special_codes[SpecialCodeIndex::VTIME], 0);
        let back = restored(r);
        assert!(back.local_modes.contains(
            LocalModes::ICANON | LocalModes::ECHO | LocalModes::ISIG | LocalModes::IEXTEN
        ));
    }

    #[test]
    fn a_pipe_is_not_put_in_raw_mode_and_polls() {
        use std::io::{Read, Write};
        let (mut read, mut write) = std::io::pipe().expect("pipe");
        assert!(RawInput::enter(&read).expect("enter").is_none());
        assert_eq!(
            wait_input(read.as_fd(), Duration::from_millis(1)).unwrap(),
            Wait::Timeout
        );
        write.write_all(b"q").unwrap();
        assert_eq!(
            wait_input(read.as_fd(), Duration::from_millis(100)).unwrap(),
            Wait::Readable
        );
        let mut buf = [0u8; 4];
        assert_eq!(read.read(&mut buf).unwrap(), 1);
        drop(write);
        // A closed writer: readable with EOF, or HUP; either way not Timeout.
        assert_ne!(
            wait_input(read.as_fd(), Duration::from_millis(100)).unwrap(),
            Wait::Timeout
        );
    }
}
