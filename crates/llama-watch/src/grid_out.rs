//! The grid file writer (#107): each laid-out frame, published for llama-view.
//!
//! The file is `llama_core::gridfile`: the dashboard's cells with their
//! original characters (`Cell::rich`) and the palette's RGB. Off unless
//! `[grid] enabled`; with it off nothing here is built and no file is
//! touched.
//!
//! [`GridWriter`] is the part with no thread: it decides whether a frame is
//! written (it differs from the last one written, or [`HEARTBEAT`] has
//! passed so a reader can tell a live writer from a stale file) and writes
//! it atomically (a temp file in the same directory, then `rename`).
//!
//! [`GridTap`] puts it on its own thread, the way the tty writer thread
//! (`tty/writer.rs`, #42) keeps the console off the tick loop. The tap's
//! [`GridTap::offer`] only replaces a one-frame slot ("newest frame wins")
//! and returns, so a slow or failing disk never delays the console, the
//! snapshot or the watchdog ping. A failed write is logged once per distinct
//! error and retried at most once per heartbeat; the dashboard carries on.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use llama_core::color::Rgb;
use llama_core::glyph::is_showable;
use llama_core::gridfile::{self, GridCell, GridImage, HEARTBEAT};
use llama_core::log::{self, Priority, Sink};
use llama_core::palette::Palette;

use crate::tty::grid::{C16, Grid};

/// Mode of the grid file: owner and group read. The unit's group (`llama-view`)
/// comes from the directory's setgid bit, not from here.
const FILE_MODE: u32 = 0o640;

fn rgb_fg(palette: Palette, color: C16) -> Rgb {
    palette.rgb(color.index())
}

/// The console cannot draw a bright background (SGR 40..47 only), so tty11
/// shows the normal colour of the slot; the file says the same.
fn rgb_bg(palette: Palette, color: C16) -> Rgb {
    palette.rgb(color.index() & 7)
}

/// `grid` as a file image. A cell shows its `rich` scalar when that is safe
/// as one cell (`is_showable`), else its console `ch`.
fn image_of(grid: &Grid, palette: Palette, seq: u32, unix_ms: u64) -> GridImage {
    GridImage {
        seq,
        unix_ms,
        cols: grid.cols(),
        rows: grid.rows(),
        cells: grid
            .cells()
            .iter()
            .map(|cell| GridCell {
                ch: if is_showable(cell.rich) {
                    cell.rich
                } else {
                    cell.ch
                },
                fg: rgb_fg(palette, cell.fg),
                bg: rgb_bg(palette, cell.bg),
            })
            .collect(),
    }
}

/// Decides and does the write; no thread, no clock of its own.
pub struct GridWriter {
    path: PathBuf,
    tmp: PathBuf,
    palette: Palette,
    seq: u32,
    /// The newest frame offered, written or not: what a heartbeat or a retry
    /// writes.
    latest: Option<Grid>,
    /// The frame in the file now.
    written: Option<Grid>,
    /// When the file was last written.
    written_at: Option<Instant>,
    /// When the last write failed, so a bad directory is retried once per
    /// heartbeat, not once per frame.
    failed_at: Option<Instant>,
    buf: Vec<u8>,
}

impl GridWriter {
    #[must_use]
    pub fn new(path: &Path, palette: Palette) -> Self {
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        Self {
            path: path.to_path_buf(),
            tmp: PathBuf::from(tmp),
            palette,
            seq: 0,
            latest: None,
            written: None,
            written_at: None,
            failed_at: None,
            buf: Vec::new(),
        }
    }

    /// Write `grid` if it differs from the last one written or the heartbeat
    /// is due. `Ok(true)` when a file was written, `Ok(false)` when not
    /// needed (or a failed write is still waiting out its retry delay).
    pub fn offer(&mut self, grid: &Grid, now: Instant, unix_ms: u64) -> io::Result<bool> {
        self.latest = Some(grid.clone());
        self.flush(now, unix_ms)
    }

    /// [`Self::offer`] of the newest frame again: the heartbeat, and the
    /// retry after a failed write when no newer frame has come.
    pub fn heartbeat(&mut self, now: Instant, unix_ms: u64) -> io::Result<bool> {
        self.flush(now, unix_ms)
    }

    fn flush(&mut self, now: Instant, unix_ms: u64) -> io::Result<bool> {
        let Some(grid) = self.latest.take() else {
            return Ok(false);
        };
        let result = self.write_if_needed(&grid, now, unix_ms);
        self.latest = Some(grid);
        result
    }

    fn write_if_needed(&mut self, grid: &Grid, now: Instant, unix_ms: u64) -> io::Result<bool> {
        let due = self
            .written_at
            .is_none_or(|at| now.saturating_duration_since(at) >= HEARTBEAT);
        let changed = self.written.as_ref() != Some(grid);
        if !(changed || due) {
            return Ok(false);
        }
        if self
            .failed_at
            .is_some_and(|at| now.saturating_duration_since(at) < HEARTBEAT)
        {
            return Ok(false);
        }
        let seq = self.seq.wrapping_add(1);
        let image = image_of(grid, self.palette, seq, unix_ms);
        gridfile::encode(&image, &mut self.buf)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
        if let Err(err) = write_atomic(&self.path, &self.tmp, &self.buf) {
            self.failed_at = Some(now);
            return Err(err);
        }
        self.seq = seq;
        self.failed_at = None;
        self.written_at = Some(now);
        if changed {
            self.written = Some(grid.clone());
        }
        Ok(true)
    }

    /// Time until the next heartbeat is due, if a grid was ever written.
    fn heartbeat_in(&self, now: Instant) -> Option<std::time::Duration> {
        let base = self.failed_at.or(self.written_at)?;
        Some(HEARTBEAT.saturating_sub(now.saturating_duration_since(base)))
    }

    /// Sequence number of the last file written (0 before the first).
    #[must_use]
    pub fn seq(&self) -> u32 {
        self.seq
    }
}

/// Temp file in `path`'s directory, then `rename` onto `path`. The temp name
/// is unlinked first and created with `O_EXCL` (which does not follow a
/// symlink), so a planted link cannot redirect the write.
fn write_atomic(path: &Path, tmp: &Path, bytes: &[u8]) -> io::Result<()> {
    match fs::remove_file(tmp) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => return Err(err),
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(FILE_MODE)
        .open(tmp)?;
    let result = file.write_all(bytes).and_then(|()| {
        drop(file);
        fs::rename(tmp, path)
    });
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}

type Failure = (io::ErrorKind, String);

#[derive(Default)]
struct State {
    pending: Option<Grid>,
    closed: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Handle to the grid writer thread. Dropping it stops the thread; the last
/// file stays (it goes stale, which is the reader's cue).
pub struct GridTap {
    shared: Arc<Shared>,
}

impl GridTap {
    /// Start the thread with `writer`. `log` gets one line per distinct
    /// failure and one when the writes recover.
    pub fn spawn<L: Sink + Send + 'static>(writer: GridWriter, log: L) -> io::Result<Self> {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
        });
        let theirs = Arc::clone(&shared);
        // Detached, like the tty writer: nothing ever joins it.
        std::thread::Builder::new()
            .name("grid-writer".to_owned())
            .spawn(move || writer_loop(writer, log, &theirs))?;
        Ok(Self { shared })
    }

    /// Hand over the newest frame and return at once. A frame still waiting
    /// is replaced.
    pub fn offer(&self, grid: &Grid) {
        let old = self.shared.lock().pending.replace(grid.clone());
        drop(old);
        self.shared.wake.notify_one();
    }
}

impl Drop for GridTap {
    fn drop(&mut self) {
        let mut st = self.shared.lock();
        st.closed = true;
        st.pending = None;
        drop(st);
        self.shared.wake.notify_all();
    }
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn writer_loop<L: Sink>(mut writer: GridWriter, mut log: L, shared: &Shared) {
    let mut failing: Option<Failure> = None;
    loop {
        let frame = {
            let mut st = shared.lock();
            loop {
                if st.closed {
                    return;
                }
                if let Some(grid) = st.pending.take() {
                    break Some(grid);
                }
                match writer.heartbeat_in(Instant::now()) {
                    Some(left) if left.is_zero() => break None,
                    Some(left) => {
                        st = shared
                            .wake
                            .wait_timeout(st, left)
                            .unwrap_or_else(PoisonError::into_inner)
                            .0;
                    }
                    None => {
                        st = shared.wake.wait(st).unwrap_or_else(PoisonError::into_inner);
                    }
                }
            }
        };
        let now = Instant::now();
        let result = match &frame {
            Some(grid) => writer.offer(grid, now, unix_ms()),
            None => writer.heartbeat(now, unix_ms()),
        };
        match result {
            Ok(true) => {
                if failing.take().is_some() {
                    log::emit(&mut log, Priority::Info, "grid: writing again");
                }
            }
            Ok(false) => {}
            Err(err) => {
                let failure = (err.kind(), err.to_string());
                if failing.as_ref() != Some(&failure) {
                    log::emit(
                        &mut log,
                        Priority::Err,
                        &format!(
                            "grid: cannot write {}: {}",
                            writer.path.display(),
                            failure.1
                        ),
                    );
                    failing = Some(failure);
                }
            }
        }
    }
}
