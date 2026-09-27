//! HID reports on the Kraken's hidraw node.
//!
//! [`HidLink::send`] accepts only an [`EncodedReport`](crate::device::proto::EncodedReport).
//! The node is opened read/write and non-blocking; replies wait on `poll`.

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::time::{Duration, Instant};

use rustix::event::{PollFd, PollFlags, Timespec, poll};

use super::PortError;
use crate::device::proto::{self, Cmd, EncodedReport};
use crate::log::{self, Priority};

/// How long one command waits for a matching reply.
pub(crate) const REPLY_BUDGET: Duration = Duration::from_millis(500);

/// Status reports skipped before the matching reply fails the step.
pub(crate) const REPLY_LIMIT: usize = 16;

const DRAIN_LIMIT: usize = 256;

/// What a HID exchange can fail with, before it is mapped onto [`super::SinkError`].
#[derive(Debug)]
pub(crate) enum ExchangeError {
    NoReply,
    Refused,
    Unavailable,
    Failed,
    Fence,
    Protocol,
}

impl std::fmt::Display for ExchangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NoReply => "no reply",
            Self::Refused => "device refused the command",
            Self::Unavailable => "hid port unavailable",
            Self::Failed => "hid port failed",
            Self::Fence => "command fence",
            Self::Protocol => "hid reply could not be parsed",
        })
    }
}

impl std::error::Error for ExchangeError {}

/// Byte transport for one hidraw node. Tests supply a fake; production uses [`HidLink`].
pub trait HidPort {
    /// Write one encoded report. There is no byte-slice send method.
    fn send(&mut self, report: &EncodedReport) -> Result<(), PortError>;

    /// Read and discard reports already queued. `Ok` stops at would-block.
    fn drain(&mut self) -> Result<Vec<[u8; proto::REPORT_LEN]>, PortError>;

    /// Read one report, or `Ok(None)` when `timeout` elapses with nothing queued.
    fn read_report(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<[u8; proto::REPORT_LEN]>, PortError>;
}

/// The real hidraw file. Constructed by [`HidLink::open`], never by tests.
pub struct HidLink {
    file: File,
}

impl HidLink {
    /// Open `path` read/write and non-blocking.
    ///
    /// Visible only inside `device`. `path` is the hidraw node `pre_open`
    /// resolved from sysfs, not a free path.
    pub(in crate::device) fn open(path: &Path) -> Result<Self, PortError> {
        let mut options = OpenOptions::new();
        options.read(true).write(true);
        options.custom_flags(rustix::fs::OFlags::NONBLOCK.bits() as i32);
        let file = options.open(path).map_err(PortError::from_io)?;
        Ok(Self { file })
    }
}

impl HidPort for HidLink {
    fn send(&mut self, report: &EncodedReport) -> Result<(), PortError> {
        write_report(&mut self.file, report.as_bytes())
    }

    fn drain(&mut self) -> Result<Vec<[u8; proto::REPORT_LEN]>, PortError> {
        let mut drained = Vec::new();
        loop {
            if drained.len() == DRAIN_LIMIT {
                return Err(PortError::unavailable("hid drain did not idle"));
            }
            match read_nonblock(&mut self.file)? {
                Some(report) => drained.push(report),
                None => return Ok(drained),
            }
        }
    }

    fn read_report(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<[u8; proto::REPORT_LEN]>, PortError> {
        let deadline = Instant::now() + timeout;
        loop {
            match read_nonblock(&mut self.file)? {
                Some(report) => return Ok(Some(report)),
                None => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() || !poll_ready(&self.file, PollFlags::IN, remaining)? {
                        return Ok(None);
                    }
                }
            }
        }
    }
}

/// Drain, send `cmd`, then accept the first matching reply inside the budget.
pub(crate) fn transact<P: HidPort + ?Sized>(
    port: &mut P,
    cmd: &Cmd,
    trace: bool,
) -> Result<[u8; proto::REPORT_LEN], ExchangeError> {
    let drained = port.drain().map_err(ExchangeError::from)?;
    for report in &drained {
        trace_report("read", report, trace);
    }
    let encoded = proto::encode(cmd).map_err(|_| ExchangeError::Fence)?;
    trace_report("write", encoded.as_bytes(), trace);
    port.send(&encoded).map_err(ExchangeError::from)?;
    let deadline = Instant::now() + REPLY_BUDGET;
    for _ in 0..REPLY_LIMIT {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(ExchangeError::NoReply);
        }
        match port.read_report(remaining).map_err(ExchangeError::from)? {
            None => return Err(ExchangeError::NoReply),
            Some(report) => {
                trace_report("read", &report, trace);
                if !proto::prefix_matches(cmd, &report) {
                    continue;
                }
                if proto::expects_success_flag(cmd) {
                    match proto::success_flag(&report) {
                        Ok(true) => return Ok(report),
                        Ok(false) => return Err(ExchangeError::Refused),
                        Err(_) => return Err(ExchangeError::Protocol),
                    }
                }
                return Ok(report);
            }
        }
    }
    Err(ExchangeError::NoReply)
}

fn trace_report(dir: &str, report: &[u8; proto::REPORT_LEN], trace: bool) {
    if !trace {
        return;
    }
    let mut hex = String::with_capacity(report.len() * 3);
    for (index, byte) in report.iter().enumerate() {
        if index > 0 {
            hex.push(' ');
        }
        hex.push_str(&format!("{byte:02x}"));
    }
    log::emit(
        &mut log::Stderr,
        Priority::Debug,
        &format!("hid {dir} {hex}"),
    );
}

/// One `write` of the whole report. A short count is not retried: the next
/// byte would be framed as a new opcode.
fn write_one_report(
    bytes: &[u8; proto::REPORT_LEN],
    mut write_once: impl FnMut(&[u8]) -> std::io::Result<usize>,
) -> Result<(), PortError> {
    for _ in 0..8 {
        match write_once(bytes) {
            Ok(n) if n == bytes.len() => return Ok(()),
            Ok(_) => return Err(PortError::unavailable("short hid report write")),
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) if err.kind() == ErrorKind::WouldBlock => continue,
            Err(err) => return Err(PortError::from_io(err)),
        }
    }
    Err(PortError::unavailable("hid report write timed out"))
}

/// One `read` is one report. A short read is zero-padded, never completed
/// with a second read.
fn read_one_report(
    mut read_once: impl FnMut(&mut [u8]) -> std::io::Result<usize>,
) -> Result<Option<[u8; proto::REPORT_LEN]>, PortError> {
    let mut buf = [0u8; proto::REPORT_LEN];
    for _ in 0..8 {
        match read_once(&mut buf) {
            Ok(0) => return Err(PortError::unavailable("hid report read closed")),
            Ok(_) => return Ok(Some(buf)),
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) if err.kind() == ErrorKind::WouldBlock => return Ok(None),
            Err(err) => return Err(PortError::from_io(err)),
        }
    }
    Err(PortError::unavailable("hid report read interrupted"))
}

fn write_report(file: &mut File, bytes: &[u8; proto::REPORT_LEN]) -> Result<(), PortError> {
    write_one_report(bytes, |buf| match file.write(buf) {
        Err(err) if err.kind() == ErrorKind::WouldBlock => {
            poll_ready(file, PollFlags::OUT, REPLY_BUDGET)
                .map_err(|err| std::io::Error::other(err.to_string()))?;
            Err(std::io::Error::new(
                ErrorKind::WouldBlock,
                "hid report write would block",
            ))
        }
        other => other,
    })
}

fn read_nonblock(file: &mut File) -> Result<Option<[u8; proto::REPORT_LEN]>, PortError> {
    read_one_report(|buf| file.read(buf))
}

fn poll_ready(file: &File, flags: PollFlags, timeout: Duration) -> Result<bool, PortError> {
    let mut fds = [PollFd::new(file, flags)];
    let spec = Timespec {
        tv_sec: i64::try_from(timeout.as_secs()).unwrap_or(i64::MAX),
        tv_nsec: i64::from(timeout.subsec_nanos()),
    };
    let ready =
        poll(&mut fds, Some(&spec)).map_err(|err| PortError::from_io(std::io::Error::from(err)))?;
    if ready == 0 {
        return Ok(false);
    }
    let revents = fds[0].revents();
    if revents.intersects(PollFlags::ERR | PollFlags::HUP | PollFlags::NVAL) {
        return Err(PortError::unavailable("hid poll status"));
    }
    Ok(revents.contains(flags))
}

impl From<PortError> for ExchangeError {
    fn from(err: PortError) -> Self {
        if err.is_unavailable() {
            Self::Unavailable
        } else {
            Self::Failed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{read_one_report, write_one_report};

    #[test]
    fn short_write_is_unavailable_and_does_not_resend_the_tail() {
        let mut report = [0u8; 64];
        report[0] = 0x36;
        report[1] = 0x03;
        report[2] = 0x11;
        let mut calls = Vec::new();
        let err = write_one_report(&report, |buf| {
            calls.push(buf.to_vec());
            Ok(32)
        })
        .expect_err("short write");
        assert!(err.is_unavailable(), "{err}");
        assert_eq!(calls.len(), 1, "the tail must not be written: {calls:?}");
        assert_eq!(calls[0].len(), 64);
        assert_eq!(&calls[0][..3], &[0x36, 0x03, 0x11]);
    }

    #[test]
    fn one_short_read_is_zero_padded_and_not_joined() {
        let mut calls = 0usize;
        let report = read_one_report(|buf| {
            calls += 1;
            assert_eq!(buf.len(), 64);
            buf[..4].copy_from_slice(&[0x31, 0x01, 0xaa, 0xbb]);
            Ok(4)
        })
        .expect("read")
        .expect("a report");
        assert_eq!(calls, 1, "a second read would join two reports");
        assert_eq!(&report[..4], &[0x31, 0x01, 0xaa, 0xbb]);
        assert!(report[4..].iter().all(|byte| *byte == 0));
    }
}
