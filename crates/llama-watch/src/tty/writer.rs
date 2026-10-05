//! The tty writer thread (#42): the tick loop never writes the console.
//!
//! A `write` to tty11 can block for as long as the console wants. The main
//! trigger seen in practice is the console waking from blank on a keypress
//! (Space, so no XOFF): `con_write` waits on the console lock while fbcon and
//! nvidia-drm unblank and modeset the monitor, which took from 1 s to over
//! 10 s. Scroll Lock (VT hold) also stops output whatever the line settings
//! say. When the tick loop did the write itself, a blocked write stopped the
//! snapshot (kraken-lcd and llama-light logged `snapshot stale`) and the
//! `WATCHDOG=1` ping, and systemd killed the watcher.
//!
//! [`FrameWriter`] is the loop's [`RenderStep`]. Its `draw` only puts the
//! frame in a one-frame slot and returns; a dedicated thread owns the real
//! renderer (the [`Term`](super::term::Term) on stdout), takes the newest
//! frame and draws it. The two share one mutex that neither holds across a
//! draw, so the loop never waits on the console:
//!
//! - **Latest frame wins.** A frame still waiting when the next one comes is
//!   dropped. Memory is one waiting frame plus the one being drawn.
//! - **Repaint after a stall.** A draw that took [`STALL_AFTER`] or longer
//!   asks the renderer for a full repaint next: unblank may have redrawn the
//!   console from its own buffer, and a held console may show a partly
//!   drawn frame. Neither stays on screen.
//! - **Logged once.** The loop logs `tty: output stalled` once while a draw
//!   has been running for [`STALL_AFTER`], and `tty: output resumed` once
//!   when the writer is free again.
//! - **Shutdown never joins a stuck thread.** [`RenderStep::finish`] asks
//!   the thread to restore the console and waits at most [`FINISH_WAIT`];
//!   when it is stuck the loop exits anyway and the unit's
//!   `ExecStopPost=llama-watch tty-reset` restores the console.

use std::io;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use llama_core::log::{self, Priority, Sink};

use super::layout::TtyModel;
use crate::service::RenderStep;

/// A draw running this long counts as a stall: logged, and followed by a
/// full repaint. Catches the 1 s unblank blips too; well over a normal full
/// repaint and well under the unit's `WatchdogSec=10`.
pub const STALL_AFTER: Duration = Duration::from_secs(1);

/// How long a clean stop waits for the writer to restore the console.
pub const FINISH_WAIT: Duration = Duration::from_millis(500);

struct Frame {
    model: TtyModel,
    now: Instant,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Close {
    /// Draw frames.
    #[default]
    Open,
    /// Run [`RenderStep::finish`], report it, then end.
    Finish,
    /// The [`FrameWriter`] is gone: end without touching the console.
    Drop,
}

type Failure = (io::ErrorKind, String);

#[derive(Default)]
struct State {
    pending: Option<Frame>,
    /// Real time the current draw started. `None` while the writer waits.
    busy_since: Option<Instant>,
    dropped: u64,
    drawn: u64,
    repaints: u64,
    /// The error of the last finished draw, if it failed.
    last: Option<Failure>,
    close: Close,
    finished: Option<Result<(), Failure>>,
}

struct Shared {
    state: Mutex<State>,
    /// The writer waits here for a frame or a close.
    wake: Condvar,
    /// [`FrameWriter::finish`] waits here, bounded, for `finished`.
    done: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Counters for tests and for the stall log.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WriterStats {
    /// Frames replaced by a newer one before the writer took them.
    pub dropped: u64,
    /// Draws the writer finished (ok or not).
    pub drawn: u64,
    /// Full repaints asked for after a stalled draw.
    pub repaints: u64,
    /// A draw is running now.
    pub busy: bool,
}

/// The tick loop's [`RenderStep`]: hands frames to the tty writer thread.
pub struct FrameWriter<L: Sink> {
    shared: Arc<Shared>,
    log: L,
    stall_after: Duration,
    finish_wait: Duration,
    /// When the stall was logged, and the drop count then.
    stalled: Option<(Instant, u64)>,
}

impl<L: Sink> FrameWriter<L> {
    /// Start the writer thread with `render` and the production timings.
    pub fn spawn<R: RenderStep + Send + 'static>(render: R, log: L) -> io::Result<Self> {
        Self::with_timing(render, log, STALL_AFTER, FINISH_WAIT)
    }

    /// [`Self::spawn`] with test timings.
    pub fn with_timing<R: RenderStep + Send + 'static>(
        render: R,
        log: L,
        stall_after: Duration,
        finish_wait: Duration,
    ) -> io::Result<Self> {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
            done: Condvar::new(),
        });
        let theirs = Arc::clone(&shared);
        // Detached: the handle is dropped, so nothing can ever join a
        // writer that is stuck in the console.
        std::thread::Builder::new()
            .name("tty-writer".to_owned())
            .spawn(move || writer_loop(render, &theirs, stall_after))?;
        Ok(Self {
            shared,
            log,
            stall_after,
            finish_wait,
            stalled: None,
        })
    }

    /// Current counters.
    pub fn stats(&self) -> WriterStats {
        let st = self.shared.lock();
        WriterStats {
            dropped: st.dropped,
            drawn: st.drawn,
            repaints: st.repaints,
            busy: st.busy_since.is_some(),
        }
    }

    fn note_stall(&mut self, busy_since: Option<Instant>, dropped: u64) {
        let stuck = busy_since.is_some_and(|since| since.elapsed() >= self.stall_after);
        match (self.stalled, stuck) {
            (None, true) => {
                log::emit(
                    &mut self.log,
                    Priority::Warning,
                    "tty: output stalled; frames are dropped until the console takes output again",
                );
                self.stalled = Some((Instant::now(), dropped));
            }
            (Some((at, before)), false) => {
                log::emit(
                    &mut self.log,
                    Priority::Info,
                    &format!(
                        "tty: output resumed after {} s, {} frames dropped; full repaint",
                        at.elapsed().as_secs() + self.stall_after.as_secs(),
                        dropped.saturating_sub(before),
                    ),
                );
                self.stalled = None;
            }
            _ => {}
        }
    }
}

impl<L: Sink> RenderStep for FrameWriter<L> {
    /// Never waits on the console: replace the waiting frame and return the
    /// result of the last finished draw.
    fn draw(&mut self, model: &TtyModel, now: Instant) -> io::Result<()> {
        let frame = Frame {
            model: model.clone(),
            now,
        };
        let (old, busy_since, dropped, last) = {
            let mut st = self.shared.lock();
            let old = st.pending.replace(frame);
            if old.is_some() {
                st.dropped = st.dropped.saturating_add(1);
            }
            (old, st.busy_since, st.dropped, st.last.clone())
        };
        drop(old);
        self.shared.wake.notify_one();
        self.note_stall(busy_since, dropped);
        match last {
            None => Ok(()),
            Some((kind, msg)) => Err(io::Error::new(kind, msg)),
        }
    }

    /// Ask the writer to restore the console and wait at most
    /// [`FINISH_WAIT`]. A stuck writer is left behind, not joined.
    fn finish(&mut self) -> io::Result<()> {
        {
            let mut st = self.shared.lock();
            st.close = Close::Finish;
            st.pending = None;
        }
        self.shared.wake.notify_all();
        let deadline = Instant::now() + self.finish_wait;
        let mut st = self.shared.lock();
        loop {
            if let Some(result) = st.finished.take() {
                return result.map_err(|(kind, msg)| io::Error::new(kind, msg));
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "tty writer is stuck in the console; tty-reset restores it",
                ));
            }
            st = self
                .shared
                .done
                .wait_timeout(st, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
}

impl<L: Sink> Drop for FrameWriter<L> {
    fn drop(&mut self) {
        {
            let mut st = self.shared.lock();
            if st.close == Close::Open {
                st.close = Close::Drop;
            }
            st.pending = None;
        }
        self.shared.wake.notify_all();
    }
}

fn writer_loop<R: RenderStep>(mut render: R, shared: &Shared, stall_after: Duration) {
    loop {
        let next = {
            let mut st = shared.lock();
            loop {
                if st.close != Close::Open {
                    break None;
                }
                if let Some(frame) = st.pending.take() {
                    st.busy_since = Some(Instant::now());
                    break Some(frame);
                }
                st = shared.wake.wait(st).unwrap_or_else(PoisonError::into_inner);
            }
        };
        let Some(frame) = next else { break };
        let started = Instant::now();
        let result = render.draw(&frame.model, frame.now);
        drop(frame);
        let stalled = started.elapsed() >= stall_after;
        if stalled {
            render.repaint();
        }
        let mut st = shared.lock();
        st.busy_since = None;
        st.drawn = st.drawn.saturating_add(1);
        if stalled {
            st.repaints = st.repaints.saturating_add(1);
        }
        st.last = result.err().map(|err| (err.kind(), err.to_string()));
    }
    if shared.lock().close != Close::Finish {
        return;
    }
    let result = render.finish().map_err(|err| (err.kind(), err.to_string()));
    shared.lock().finished = Some(result);
    shared.done.notify_all();
}
