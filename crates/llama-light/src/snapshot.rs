//! Read `/run/llama-watch/snapshot.json` through llama-core's validated parser.
//!
//! The path is `llama_core::wire::SNAPSHOT_PATH`, not configurable. The
//! file is opened `O_NOFOLLOW | O_NONBLOCK`, must be a regular file no
//! larger than `wire::MAX_BYTES`, and is accepted only through
//! `wire::parse_validated`.

use std::path::PathBuf;

use llama_core::wire::{self, SnapshotV1};

/// Why no snapshot was accepted this tick.
pub const MISSING: &str = "snapshot missing";
/// Not a regular file.
pub const NOT_REGULAR: &str = "snapshot not a regular file";
/// Over the size cap.
pub const TOO_LARGE: &str = "snapshot too large";
/// Failed to read, parse or validate.
pub const INVALID: &str = "snapshot invalid";

/// Where snapshots come from. Tests supply a fake.
pub trait SnapshotSource {
    /// The current snapshot, validated.
    fn read(&mut self) -> Result<SnapshotV1, &'static str>;
}

/// The published snapshot file.
pub struct SnapshotFile {
    path: PathBuf,
}

impl SnapshotFile {
    /// The production file, `llama_core::wire::SNAPSHOT_PATH`.
    #[must_use]
    pub fn published() -> Self {
        Self {
            path: PathBuf::from(wire::SNAPSHOT_PATH),
        }
    }

    /// A scratch file, for tests.
    #[must_use]
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl SnapshotSource for SnapshotFile {
    fn read(&mut self) -> Result<SnapshotV1, &'static str> {
        let fd = match rustix::fs::open(
            &self.path,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::NONBLOCK
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        ) {
            Ok(fd) => fd,
            Err(err) if err == rustix::io::Errno::NOENT => return Err(MISSING),
            Err(_) => return Err(INVALID),
        };
        let stat = rustix::fs::fstat(&fd).map_err(|_| INVALID)?;
        if !rustix::fs::FileType::from_raw_mode(stat.st_mode).is_file() {
            return Err(NOT_REGULAR);
        }
        let max = i64::try_from(wire::MAX_BYTES).map_err(|_| TOO_LARGE)?;
        if stat.st_size < 0 || stat.st_size > max {
            return Err(TOO_LARGE);
        }
        let mut buf = vec![0_u8; wire::MAX_BYTES + 1];
        let mut filled = 0_usize;
        while filled < buf.len() {
            match rustix::io::read(&fd, &mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(err) if err == rustix::io::Errno::INTR => continue,
                Err(_) => return Err(INVALID),
            }
        }
        if filled > wire::MAX_BYTES {
            return Err(TOO_LARGE);
        }
        wire::parse_validated(&buf[..filled]).map_err(|_| INVALID)
    }
}
