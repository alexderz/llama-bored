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
//! [`GridTap::offer_frame`] only replaces a one-frame slot ("newest frame
//! wins") and returns, so a slow or failing disk never delays the console,
//! the snapshot or the watchdog ping. A failed write is logged once per
//! distinct error and retried at most once per heartbeat; the dashboard
//! carries on.
//!
//! `[grid] size` (#115) is optional. Unset, or equal to the console frame,
//! and with exact colours off, the slot holds that frame and this thread
//! writes it. A different size, or `[grid] exact_colors` (#116), puts a
//! clone of the [`TtyModel`] in the slot instead: the model's owned
//! strings and vectors, one frame, dropped when a newer frame replaces it.
//! This thread then lays that model out and writes the result. Exact
//! colours are that second layout even when the size matches. The console
//! grid stores none of them: with the file off no plane is allocated, and
//! a painted cell returns before any index work when the plane is absent.
//! The layout runs after
//! the slot is taken and the lock is dropped,
//! so however long it takes it does not delay the console write, the snapshot
//! or the watchdog ping. Layout reads cells only. The block digits are sized
//! from the column budget (`fit_pixel`), and eighths versus halves come from
//! `chart_glyphs` on the model, not from the console font or from a cell's
//! size in pixels. The second size needs no font of its own.

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

use crate::config::{GridFile, TtySize};
use crate::tty::grid::{C16, Grid};
use crate::tty::layout::{self, TtyModel};

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
/// as one cell (`is_showable`), else its console `ch`. An exact colour, when
/// the cell has one, replaces the palette RGB. The format is unchanged.
///
/// Public for the grid tests. The writer is the production caller.
#[doc(hidden)]
pub fn image_of(grid: &Grid, palette: Palette, seq: u32, unix_ms: u64) -> GridImage {
    let cols = usize::from(grid.cols());
    GridImage {
        seq,
        unix_ms,
        cols: grid.cols(),
        rows: grid.rows(),
        cells: grid
            .cells()
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let col = u16::try_from(i % cols.max(1)).unwrap_or(0);
                let row = u16::try_from(i / cols.max(1)).unwrap_or(0);
                GridCell {
                    ch: if is_showable(cell.rich) {
                        cell.rich
                    } else {
                        cell.ch
                    },
                    fg: grid
                        .exact_fg(col, row)
                        .unwrap_or_else(|| rgb_fg(palette, cell.fg)),
                    bg: grid
                        .exact_bg(col, row)
                        .unwrap_or_else(|| rgb_bg(palette, cell.bg)),
                }
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

/// One waiting frame. The newest replaces whatever is still here.
enum Queued {
    /// The console frame, written as it stands. No second layout.
    Console(Grid),
    /// The same data, laid out on the writer thread at this size. The bool
    /// is `[grid] exact_colors`: the layout then records the ramp. Boxed so
    /// a waiting frame is not the size of a [`TtyModel`].
    Apart(Box<TtyModel>, u16, u16, bool),
}

#[derive(Default)]
struct State {
    pending: Option<Queued>,
    closed: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    /// Test builds only. Production has no gate. When held, the writer
    /// parks after it has taken a second-layout frame and before it lays
    /// that frame out. The tests show that offers made while it is parked
    /// return, that the newest of those offers is the one left waiting,
    /// and that a frame replaced while it is parked is never laid out.
    /// The frame already taken is laid out when the gate opens.
    #[cfg(test)]
    gate: TestGate,
}

impl Shared {
    fn bare() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
            #[cfg(test)]
            gate: TestGate::running(),
        })
    }

    #[cfg(test)]
    fn paused() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
            gate: TestGate::held(),
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Pauses the writer after a second-layout frame has been taken and
/// before that frame is laid out. Absent from a production build.
#[cfg(test)]
struct TestGate {
    paused: Mutex<bool>,
    pause_cv: Condvar,
    parked: Mutex<bool>,
    parked_cv: Condvar,
    /// Hosts that reached the second layout, in the order they did.
    laid: Mutex<Vec<String>>,
}

#[cfg(test)]
impl TestGate {
    fn running() -> Self {
        Self::at(false)
    }

    fn held() -> Self {
        Self::at(true)
    }

    fn at(paused: bool) -> Self {
        Self {
            paused: Mutex::new(paused),
            pause_cv: Condvar::new(),
            parked: Mutex::new(false),
            parked_cv: Condvar::new(),
            laid: Mutex::new(Vec::new()),
        }
    }

    /// Park until [`GridTap::release`]. The caller has already dropped the
    /// slot lock: this waits on the gate, not on the slot.
    fn wait_if_paused(&self) {
        let mut paused = self.paused.lock().unwrap_or_else(PoisonError::into_inner);
        if !*paused {
            return;
        }
        {
            let mut parked = self.parked.lock().unwrap_or_else(PoisonError::into_inner);
            *parked = true;
            drop(parked);
            self.parked_cv.notify_all();
        }
        while *paused {
            paused = self
                .pause_cv
                .wait(paused)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *self.parked.lock().unwrap_or_else(PoisonError::into_inner) = false;
    }

    fn note(&self, host: &str) {
        self.laid
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(host.to_owned());
    }
}

fn spawn_thread<L: Sink + Send + 'static>(
    writer: GridWriter,
    log: L,
    size: Option<TtySize>,
    shared: Arc<Shared>,
) -> io::Result<GridTap> {
    let theirs = Arc::clone(&shared);
    // Detached, like the tty writer: nothing ever joins it.
    std::thread::Builder::new()
        .name("grid-writer".to_owned())
        .spawn(move || writer_loop(writer, log, &theirs))?;
    Ok(GridTap {
        shared,
        size,
        exact: false,
    })
}

/// Handle to the grid writer thread. Dropping it stops the thread; the last
/// file stays (it goes stale, which is the reader's cue).
pub struct GridTap {
    shared: Arc<Shared>,
    /// `[grid] size`. `None` publishes every console frame as it stands,
    /// unless [`Self::exact`] is set.
    size: Option<TtySize>,
    /// `[grid] exact_colors`. The console frame stays palette-only; the
    /// writer thread lays the model out again so the file can carry the ramp.
    exact: bool,
}

impl GridTap {
    /// Start the thread with `writer`. `log` gets one line per distinct
    /// failure and one when the writes recover. `size` is `[grid] size`:
    /// `None` publishes the console frame, unless [`Self::with_exact`] is set.
    /// Exact colours stay off: the service uses [`Self::from_config`].
    pub fn spawn<L: Sink + Send + 'static>(
        writer: GridWriter,
        log: L,
        size: Option<TtySize>,
    ) -> io::Result<Self> {
        spawn_thread(writer, log, size, Shared::bare())
    }

    /// Start the writer from `[grid]`. [`GridFile::size`] and
    /// [`GridFile::exact_colors`] are the fields this reads. The service
    /// calls this when the file is enabled.
    pub fn from_config<L: Sink + Send + 'static>(
        writer: GridWriter,
        log: L,
        grid: &GridFile,
    ) -> io::Result<Self> {
        Ok(Self::spawn(writer, log, grid.size)?.with_exact(grid.exact_colors))
    }

    /// `[grid] exact_colors`. Off unless this is called. On, every frame is
    /// a second layout on the writer thread, even when the size matches.
    #[must_use]
    pub fn with_exact(mut self, exact: bool) -> Self {
        self.exact = exact;
        self
    }

    /// Publish the console frame, or a second layout when `[grid] size`
    /// differs from it or exact colours are on. Returns once the frame is
    /// in the slot.
    ///
    /// The copy is the console grid when the size is unset or equal and
    /// exact colours are off, and a clone of `model` otherwise. The layout
    /// of that model happens on the writer thread, not here.
    pub fn offer_frame(&self, model: &TtyModel, console: &Grid) {
        let (cols, rows) = match self.size {
            Some(size) if size.cols != console.cols() || size.rows != console.rows() => {
                (size.cols, size.rows)
            }
            _ => (console.cols(), console.rows()),
        };
        let apart = self.exact || cols != console.cols() || rows != console.rows();
        let queued = if apart {
            Queued::Apart(Box::new(model.clone()), cols, rows, self.exact)
        } else {
            Queued::Console(console.clone())
        };
        self.enqueue(queued);
    }

    fn enqueue(&self, queued: Queued) {
        let old = self.shared.lock().pending.replace(queued);
        drop(old);
        self.shared.wake.notify_one();
    }
}

#[cfg(test)]
impl GridTap {
    /// Start with the gate held. The thread parks once it has taken a
    /// second-layout frame, before laying it out. [`Self::release`] lets
    /// it continue. Dropping the tap releases the hold too, so the thread
    /// can see that it was closed.
    fn spawn_paused<L: Sink + Send + 'static>(
        writer: GridWriter,
        log: L,
        size: Option<TtySize>,
    ) -> io::Result<Self> {
        spawn_thread(writer, log, size, Shared::paused())
    }

    fn wait_until_parked(&self) {
        let gate = &self.shared.gate;
        let mut parked = gate.parked.lock().unwrap_or_else(PoisonError::into_inner);
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        while !*parked {
            let left = deadline.saturating_duration_since(Instant::now());
            assert!(!left.is_zero(), "grid thread did not park");
            let (next, status) = gate
                .parked_cv
                .wait_timeout(parked, left)
                .unwrap_or_else(PoisonError::into_inner);
            parked = next;
            if status.timed_out() && !*parked {
                panic!("grid thread did not park");
            }
        }
    }

    fn release(&self) {
        let mut paused = self
            .shared
            .gate
            .paused
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        *paused = false;
        drop(paused);
        self.shared.gate.pause_cv.notify_all();
    }

    fn laid_out(&self) -> Vec<String> {
        self.shared
            .gate
            .laid
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn is_parked(&self) -> bool {
        *self
            .shared
            .gate
            .parked
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Host of the second-layout frame waiting in the slot, if that is
    /// what is waiting.
    fn pending_host(&self) -> Option<String> {
        let st = self.shared.lock();
        match &st.pending {
            Some(Queued::Apart(model, _, _, _)) => Some(model.host.clone()),
            _ => None,
        }
    }
}

impl Drop for GridTap {
    fn drop(&mut self) {
        #[cfg(test)]
        self.release();
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
                if let Some(queued) = st.pending.take() {
                    break Some(queued);
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
        // Layout, when there is one, runs without the slot lock held.
        let result = match frame {
            Some(Queued::Console(grid)) => writer.offer(&grid, now, unix_ms()),
            Some(Queued::Apart(model, cols, rows, exact)) => {
                // Test builds park here: the frame is already out of the
                // slot, and the slot lock is not held. A production build
                // has no gate.
                #[cfg(test)]
                shared.gate.wait_if_paused();
                // Cells only: `fit_pixel` sizes the block digits from the
                // column budget, and eighths versus halves come from
                // `model.chart_glyphs`. The console font's pixel size is
                // not an input. Exact colours are the same layout with the
                // ramp plane; the console frame never carries it.
                #[cfg(test)]
                shared.gate.note(&model.host);
                let grid = if exact {
                    layout::layout_exact(&model, cols, rows)
                } else {
                    layout::layout(&model, cols, rows)
                };
                writer.offer(&grid, now, unix_ms())
            }
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant, UNIX_EPOCH};

    use llama_core::gridfile;
    use llama_core::log::Sink;
    use llama_core::palette::Palette;

    use crate::config::{ChartGlyphs, TtySize};
    use crate::tty::grid::Grid;
    use crate::tty::layout::{self, TtyModel, WatchState};

    use super::{GridTap, GridWriter};

    struct Discard;

    impl Sink for Discard {
        fn write_line(&mut self, _line: &str) {}
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("llama-grid-hold-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        dir
    }

    fn host_model(host: &str) -> TtyModel {
        TtyModel {
            state: WatchState::Generating,
            host: host.to_owned(),
            model_name: "Model".to_owned(),
            model_detail: String::new(),
            model_stuck: false,
            slots_line: String::new(),
            swap_line: String::new(),
            cool_c: None,
            cpu_c: None,
            gpu_c: None,
            clock: "12:00:00".to_owned(),
            cpu_pct: None,
            cpu_cores: None,
            gpu_pct: None,
            vram_used_gb: None,
            vram_total_gb: None,
            mem_used_gb: None,
            mem_total_gb: None,
            power_w: None,
            power_limit_w: None,
            load_pct: None,
            activity_pct: None,
            activity_src: None,
            activity_w: None,
            gen_tps: None,
            prompt_tps: None,
            prompt_last: None,
            gen_ceiling: 250.0,
            prompt_ceiling: 1500.0,
            slots: Vec::new(),
            backend_lines: Vec::new(),
            text_note: String::new(),
            requests: Vec::new(),
            in_title: String::new(),
            out_title: String::new(),
            in_lines: Vec::new(),
            out_lines: Vec::new(),
            out_shown: 0,
            replay_frame: None,
            show_text: true,
            down_since: String::new(),
            health: Vec::new(),
            snapshot: None,
            snapshot_age: String::new(),
            errors: 0,
            uptime: String::new(),
            chart: Vec::new(),
            chart_bucket_s: 2,
            chart_glyphs: ChartGlyphs::Halves,
            fans: None,
            temps: None,
            ctx_history_h: 6,
            setup: None,
            job: None,
            job_note: None,
            model_id: String::new(),
            claims: Vec::new(),
            now: UNIX_EPOCH,
            tz_offset_s: 0,
        }
    }

    fn paused_tap(dir: &std::path::Path) -> GridTap {
        GridTap::spawn_paused(
            GridWriter::new(&dir.join("grid"), Palette::Llama),
            Discard,
            Some(TtySize {
                cols: 213,
                rows: 60,
            }),
        )
        .expect("tap")
    }

    #[test]
    fn the_console_layout_stores_no_exact_plane() {
        let mut model = host_model("box");
        model.cpu_pct = Some(50.0);
        model.gen_tps = Some(40.0);
        let plain = layout::layout(&model, 192, 60);
        let exact = layout::layout_exact(&model, 192, 60);
        assert!(!plain.records_exact());
        assert!(exact.records_exact());
        assert!(plain.same_console(&exact));
        let palette = super::image_of(&plain, Palette::Llama, 1, 0);
        let ramped = super::image_of(&exact, Palette::Llama, 1, 0);
        assert_ne!(
            palette.cells, ramped.cells,
            "a filled meter should carry a ramp"
        );
    }

    /// #115: a second layout is paused after its frame has been taken.
    /// Two further offers replace the slot. The middle model is never
    /// laid out. The frame already taken is laid out when the gate opens,
    /// and then the newest one waiting.
    #[test]
    fn the_newest_model_wins_the_second_layout() {
        let dir = scratch("newest");
        let path = dir.join("grid");
        let tap = paused_tap(&dir);
        let console = Grid::blank(192, 60);
        tap.offer_frame(&host_model("aaa"), &console);
        tap.wait_until_parked();
        tap.offer_frame(&host_model("bbb"), &console);
        let last = host_model("ccc");
        tap.offer_frame(&last, &console);
        assert!(tap.is_parked(), "the hold ended before the offers");
        assert!(
            tap.laid_out().is_empty(),
            "a frame was laid out while the thread was held"
        );
        assert_eq!(tap.pending_host().as_deref(), Some("ccc"));
        tap.release();
        let want = super::image_of(&layout::layout(&last, 213, 60), Palette::Llama, 1, 0);
        let got = wait_seq(&path, 2);
        assert_eq!(
            tap.laid_out(),
            vec!["aaa".to_owned(), "ccc".to_owned()],
            "the middle model was laid out"
        );
        assert_eq!(
            got.seq, 2,
            "the frame already taken is written, then the newest"
        );
        assert_eq!((got.cols, got.rows), (want.cols, want.rows));
        assert_eq!(got.cells, want.cells);
        drop(tap);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// #115: offers made while a second layout has been taken, and has not
    /// run, return. The newest of them is the frame left waiting. The
    /// deadline is a deadlock bound: a slot lock held across that layout
    /// stops the offers until the gate opens.
    #[test]
    fn an_offer_returns_while_the_second_layout_is_held() {
        let dir = scratch("held");
        let tap = Arc::new(paused_tap(&dir));
        let console = Grid::blank(192, 60);
        tap.offer_frame(&host_model("held"), &console);
        tap.wait_until_parked();
        assert!(
            tap.laid_out().is_empty(),
            "layout ran before the thread parked"
        );
        // The offers below are what must return. This thread does not take
        // the slot lock again until they have: a layout that still held it
        // would block them, and the deadline below would fire.

        let offers = Arc::clone(&tap);
        let console = console.clone();
        let handle = std::thread::spawn(move || {
            offers.offer_frame(&host_model("one"), &console);
            offers.offer_frame(&host_model("two"), &console);
            offers.offer_frame(&host_model("three"), &console);
        });
        let deadline = Instant::now() + Duration::from_secs(1);
        while !handle.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let returned = handle.is_finished();
        if !returned {
            // The offers are blocked on the slot lock. Open the gate so
            // they can finish and this process can unwind.
            tap.release();
            let _ = handle.join();
        } else {
            handle.join().expect("offer thread");
        }
        assert!(
            returned,
            "offers blocked while a second layout was in progress"
        );
        assert!(
            tap.is_parked(),
            "the second layout finished during the offers"
        );
        assert!(tap.laid_out().is_empty());
        assert_eq!(tap.pending_host().as_deref(), Some("three"));
        drop(tap);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn wait_seq(path: &std::path::Path, seq: u32) -> gridfile::GridImage {
        let until = Instant::now() + Duration::from_secs(5);
        while Instant::now() < until {
            if let Ok(bytes) = std::fs::read(path)
                && let Ok(image) = gridfile::decode(&bytes)
                && image.seq >= seq
            {
                return image;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("timed out waiting for seq {seq} at {}", path.display());
    }
}
