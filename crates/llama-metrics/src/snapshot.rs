//! The one file llama-metrics reads: the llama-watch snapshot.
//!
//! Same pattern as the LCD writer's snapshot reader: `O_NOFOLLOW`, a regular
//! file, the `MAX_BYTES` cap before any parse, then
//! [`llama_core::wire::parse_validated`].

use std::path::{Path, PathBuf};

use llama_core::wire::{self, WireSnapshot};

/// Why no snapshot was accepted. Exported as a `reason` label.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadError {
    Missing,
    NotRegular,
    TooLarge,
    /// The file could not be opened or read.
    Invalid,
    /// [`wire::parse_validated`] refused the bytes.
    Rejected(wire::WireError),
}

impl ReadError {
    /// Stable label value.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::NotRegular => "not_regular",
            Self::TooLarge => "too_large",
            Self::Invalid => "invalid",
            Self::Rejected(_) => "rejected",
        }
    }

    /// Why, for the log: the label, or the validator's reason.
    #[must_use]
    pub fn reason(self) -> String {
        match self {
            Self::Missing => "missing".to_owned(),
            Self::NotRegular => "not a regular file".to_owned(),
            Self::TooLarge => "too large".to_owned(),
            Self::Invalid => "unreadable".to_owned(),
            Self::Rejected(err) => err.to_string(),
        }
    }
}

/// Reads one snapshot file.
#[derive(Clone, Debug)]
pub struct SnapshotFile {
    path: PathBuf,
}

impl SnapshotFile {
    /// Production: `llama_core::wire::SNAPSHOT_PATH`.
    #[must_use]
    pub fn published() -> Self {
        Self::at(wire::SNAPSHOT_PATH)
    }

    /// `path` is the snapshot. Tests pass a scratch file.
    #[must_use]
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Open without following a symlink, cap, parse, validate.
    pub fn read(&self) -> Result<WireSnapshot, ReadError> {
        use rustix::fs::{Mode, OFlags};
        let fd = match rustix::fs::open(
            &self.path,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        ) {
            Ok(fd) => fd,
            Err(err) if err == rustix::io::Errno::NOENT => return Err(ReadError::Missing),
            Err(err) if err == rustix::io::Errno::LOOP => return Err(ReadError::NotRegular),
            Err(_) => return Err(ReadError::Invalid),
        };
        let stat = rustix::fs::fstat(&fd).map_err(|_| ReadError::Invalid)?;
        if !rustix::fs::FileType::from_raw_mode(stat.st_mode).is_file() {
            return Err(ReadError::NotRegular);
        }
        let max = i64::try_from(wire::MAX_BYTES).map_err(|_| ReadError::TooLarge)?;
        if stat.st_size < 0 || stat.st_size > max {
            return Err(ReadError::TooLarge);
        }
        let mut buf = vec![0_u8; wire::MAX_BYTES + 1];
        let mut filled = 0_usize;
        while filled < buf.len() {
            match rustix::io::read(&fd, &mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(err) if err == rustix::io::Errno::INTR => continue,
                Err(_) => return Err(ReadError::Invalid),
            }
        }
        if filled > wire::MAX_BYTES {
            return Err(ReadError::TooLarge);
        }
        wire::parse_validated(&buf[..filled]).map_err(ReadError::Rejected)
    }
}

/// Host `CLOCK_MONOTONIC` in nanoseconds, the clock of `t_mono_ns`.
#[must_use]
pub fn mono_now_ns() -> u64 {
    let ts = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    let sec = u64::try_from(ts.tv_sec).unwrap_or(0);
    let nsec = u64::try_from(ts.tv_nsec).unwrap_or(0);
    sec.saturating_mul(1_000_000_000).saturating_add(nsec)
}
