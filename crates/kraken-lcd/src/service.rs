//! Main loop, state machine, start-up self-checks, and sd-notify.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use sd_notify::NotifyState;
use thiserror::Error;

use crate::config::{Config, InvalidConfig, UploadMode, ValidConfig};
use crate::device::{
    HidPort, KrakenLcd, LcdSink, NoBulk, OpenRequest, PortError, STATE_DIR, SYS_ROOT, SinkError,
    UploadFailed, open_resolved_hid,
};
use crate::history::History;
use crate::log::{self, Priority};
use crate::policy::{Decision, Policy};
use crate::present::{View, present};
use crate::render::{self, AssetError, Assets};
use crate::snapshot_reader::SnapshotReader;
use llama_core::sample::Snapshot;

/// Host facts the self-checks consult. Production fills this from the real
/// process and `/sys`. Tests build one by hand so they never open the host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckEnv {
    /// Must be [`SYS_ROOT`] (`/sys`) for production `run` / `restore-stock`.
    pub sys_root: PathBuf,
    /// Directory that holds the HALTED latch.
    pub state_dir: PathBuf,
    /// `true` when `geteuid() == 0`.
    pub euid_is_root: bool,
    /// `true` when `statvfs("/sys")` reports `ST_RDONLY`.
    pub sys_is_readonly: bool,
    /// `true` when a hwmon named `z53` exists under [`Self::sys_root`].
    pub z53_present: bool,
    /// `true` when the state directory is writable.
    pub state_dir_writable: bool,
}

impl CheckEnv {
    /// Probe the real host. `run` and `restore-stock` always use this.
    #[must_use]
    pub fn production() -> Self {
        let sys_root = PathBuf::from(SYS_ROOT);
        let state_dir = PathBuf::from(STATE_DIR);
        Self {
            z53_present: z53_exists(&sys_root),
            state_dir_writable: dir_writable(&state_dir),
            sys_root,
            state_dir,
            euid_is_root: rustix::process::geteuid().is_root(),
            sys_is_readonly: sys_is_readonly(),
        }
    }
}

/// Why start-up refused to continue.
#[derive(Debug, Error)]
pub enum CheckError {
    /// [`Config::validate`] failed.
    #[error(transparent)]
    Invalid(#[from] InvalidConfig),
    /// `run` / `restore-stock` were pointed at a sys root other than `/sys`.
    #[error("sys root must be /sys")]
    SysRootNotDefault,
    /// `/sys` is not mounted read-only.
    #[error("/sys is not mounted read-only")]
    SysNotReadonly,
    /// Effective uid is 0.
    #[error("refusing to run as root")]
    RunningAsRoot,
    /// No hwmon named `z53`.
    #[error("no z53 hwmon")]
    NoZ53,
    /// The HALTED latch directory cannot be written.
    #[error("state directory is not writable")]
    StateDirNotWritable,
    /// Bundled fonts or sprites failed to decode.
    #[error(transparent)]
    Assets(#[from] AssetError),
}

/// Config limits, default roots, hardening, assets. Does not open a device.
pub fn self_check(config: &Config, env: &CheckEnv) -> Result<Assets, CheckError> {
    config.validate()?;
    if env.sys_root != Path::new(SYS_ROOT) {
        return Err(CheckError::SysRootNotDefault);
    }
    if !env.sys_is_readonly {
        return Err(CheckError::SysNotReadonly);
    }
    if env.euid_is_root {
        return Err(CheckError::RunningAsRoot);
    }
    if !env.z53_present {
        return Err(CheckError::NoZ53);
    }
    if !env.state_dir_writable {
        return Err(CheckError::StateDirNotWritable);
    }
    Ok(Assets::load()?)
}

/// `true` when `sys_root/class/hwmon/hwmonN/name` is `z53`.
///
/// The read is [`crate::device::z53_exists`], which reuses the guard's
/// `find_z53`. This wrapper keeps the self-check call site in the service.
#[must_use]
pub fn z53_exists(sys_root: &Path) -> bool {
    crate::device::z53_exists(sys_root)
}

/// Fail-closed latch: any result other than `NotFound` means the latch is set.
#[must_use]
pub fn latch_present(state_dir: &Path) -> bool {
    match state_dir.join("halted").symlink_metadata() {
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) | Ok(_) => true,
    }
}

fn sys_is_readonly() -> bool {
    rustix::fs::statvfs(SYS_ROOT)
        .map(|stat| stat.f_flag.contains(rustix::fs::StatVfsMountFlags::RDONLY))
        .unwrap_or(false)
}

fn dir_writable(path: &Path) -> bool {
    rustix::fs::access(path, rustix::fs::Access::WRITE_OK).is_ok()
}

/// Path of the HALTED latch: `<STATE_DIR>/halted`.
#[must_use]
pub fn halt_latch_path() -> PathBuf {
    PathBuf::from(STATE_DIR).join("halted")
}

/// Wall-clock jump that counts as a resume, in units of the tick.
pub const WALL_JUMP_TICKS: u32 = 3;

/// First reopen wait after DETACHED.
pub const BACKOFF_START: Duration = Duration::from_secs(10);

/// Cap of the DETACHED reopen wait.
pub const BACKOFF_CAP: Duration = Duration::from_secs(300);

/// Double `current`, capped at [`BACKOFF_CAP`].
#[must_use]
pub fn next_backoff(current: Duration) -> Duration {
    current.saturating_mul(2).min(BACKOFF_CAP)
}

/// How long a wall jump must be to detach.
#[must_use]
pub fn wall_jump_limit(tick: Duration) -> Duration {
    tick.saturating_mul(WALL_JUMP_TICKS)
        .max(Duration::from_secs(2))
}

/// Resume when wall ran ahead of monotonic by more than [`wall_jump_limit`].
#[must_use]
pub fn is_wall_resume(wall_delta: Duration, mono_delta: Duration, tick: Duration) -> bool {
    wall_delta.saturating_sub(mono_delta) > wall_jump_limit(tick)
}

/// Advance `from` by `delta` for gap accounting. A backwards clock uses `delta`.
#[must_use]
pub fn wall_delta(from: SystemTime, to: SystemTime, fallback: Duration) -> Duration {
    match to.duration_since(from) {
        Ok(delta) => delta,
        Err(_) => fallback,
    }
}

/// Tick period from a validated `writer.tick_s`.
#[must_use]
pub fn tick_duration(tick_s: f64) -> Duration {
    Duration::from_secs_f64(tick_s)
}

/// Frame period for a validated `upload.stream_fps` (`1..=12`).
///
/// Nanoseconds are `1e9 / fps`, rounded to the nearest nanosecond. At 10 fps
/// that is exactly 100 ms.
fn stream_period(fps: u8) -> Duration {
    let fps = u64::from(fps);
    let ns = (1_000_000_000 + fps / 2) / fps;
    Duration::from_nanos(ns)
}

/// Injected clock. Production sleeps; tests advance a fake instant.
pub trait Clock {
    /// CLOCK_MONOTONIC.
    fn mono(&self) -> Instant;
    /// Wall clock, for suspend detection.
    fn wall(&self) -> SystemTime;
    /// Wait `d`, or advance a fake clock by `d`.
    fn sleep(&mut self, d: Duration);
}

impl<C: Clock + ?Sized> Clock for &mut C {
    fn mono(&self) -> Instant {
        (**self).mono()
    }

    fn wall(&self) -> SystemTime {
        (**self).wall()
    }

    fn sleep(&mut self, d: Duration) {
        (**self).sleep(d);
    }
}

/// One collector tick. Tests feed scripted snapshots.
pub trait Sampler {
    /// `mono` and `wall` come from the injected clock.
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot;
}

impl<S: Sampler + ?Sized> Sampler for &mut S {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        (**self).sample(mono, wall)
    }
}

/// sd-notify seam. Unset `NOTIFY_SOCKET` makes the production impl a no-op.
pub trait Notifier {
    /// `READY=1` after start-up checks.
    fn ready(&mut self);
    /// `WATCHDOG=1` after steps 1–4 returned.
    fn watchdog(&mut self);
    /// `STOPPING=1` on a requested stop.
    fn stopping(&mut self);
}

impl<N: Notifier + ?Sized> Notifier for &mut N {
    fn ready(&mut self) {
        (**self).ready();
    }

    fn watchdog(&mut self) {
        (**self).watchdog();
    }

    fn stopping(&mut self) {
        (**self).stopping();
    }
}

/// Stop request. Tests trip this after N ticks (SIGTERM/SIGINT).
pub trait Stop {
    /// `true` when the loop should send `STOPPING=1` and leave.
    fn requested(&self) -> bool;
}

impl<S: Stop + ?Sized> Stop for &S {
    fn requested(&self) -> bool {
        (**self).requested()
    }
}

/// Host clock.
pub struct RealClock;

impl Clock for RealClock {
    fn mono(&self) -> Instant {
        Instant::now()
    }

    fn wall(&self) -> SystemTime {
        SystemTime::now()
    }

    fn sleep(&mut self, d: Duration) {
        std::thread::sleep(d);
    }
}

/// `sd_notify`. No-op when `NOTIFY_SOCKET` is unset.
pub struct SdNotify;

impl Notifier for SdNotify {
    fn ready(&mut self) {
        let _ = sd_notify::notify(&[NotifyState::Ready]);
    }

    fn watchdog(&mut self) {
        let _ = sd_notify::notify(&[NotifyState::Watchdog]);
    }

    fn stopping(&mut self) {
        let _ = sd_notify::notify(&[NotifyState::Stopping]);
    }
}

/// Always continues. Tests and callers without a stop flag use it.
/// SIGTERM keeps its default action: `unsafe` is forbidden and no signal crate
/// is pinned. Production `run` uses [`StopFlag`] instead.
pub struct NeverStop;

impl Stop for NeverStop {
    fn requested(&self) -> bool {
        false
    }
}

/// Stop request file the unit's `ExecStop=` creates (GitHub #59).
///
/// `systemctl stop` / `try-restart` first touches this file, then waits for
/// the writer to exit, and only then sends SIGTERM. The loop reads it at the
/// top of every tick, between uploads (an upload runs to its end inside one
/// tick), so a requested stop never cuts an upload between `WriteStart` and
/// `WriteEnd`; a SIGTERM in the middle of one left the Kraken refusing
/// `DeleteBucket` for over a minute. systemd
/// creates `/run/kraken-lcd` (`RuntimeDirectory=`) empty at each start and
/// removes it at stop, so the writer itself never creates or removes it.
pub const STOP_FLAG: &str = "/run/kraken-lcd/stop";

/// [`Stop`] that is requested while a file exists. Read-only: one
/// `symlink_metadata` per check, nothing is created or removed.
pub struct StopFlag {
    path: PathBuf,
}

impl StopFlag {
    /// Watch `path`. Production passes [`STOP_FLAG`].
    #[must_use]
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }
}

impl Stop for StopFlag {
    /// Only a file that is there asks for a stop. Any other answer keeps the
    /// writer running; SIGTERM after `ExecStop=` stays the backstop.
    fn requested(&self) -> bool {
        self.path.symlink_metadata().is_ok()
    }
}

/// Why the loop ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoopExit {
    /// Stop requested. Exit 0.
    Stopped,
    /// `restore_stock` then exit 1.
    Restored,
    /// Fence. Exit 2, no restore.
    Fence,
    /// `SinkError::Fatal`. Exit 2, no restore.
    Fatal,
}

impl LoopExit {
    /// Process exit code.
    #[must_use]
    pub fn code(self) -> i32 {
        match self {
            Self::Stopped => 0,
            Self::Restored => 1,
            Self::Fence | Self::Fatal => 2,
        }
    }
}

/// Inputs for [`run_loop`]. Production and tests share this path.
pub struct LoopInput<'a, Samp, Clk, Ntf, Stp, Lg, Op> {
    /// Validated config. The loop does not re-validate.
    pub config: &'a ValidConfig,
    /// Collector or a scripted sampler.
    pub sampler: Samp,
    /// Tick clock.
    pub clock: Clk,
    /// sd-notify.
    pub notify: Ntf,
    /// Stop request. Production uses [`StopFlag`]; tests inject their own.
    pub stop: Stp,
    /// Log sink.
    pub log: Lg,
    /// Opens the LCD. Tests return `FakeLcd`.
    pub open: Op,
    /// Fonts and sprites, already decoded by [`self_check`].
    pub assets: &'a mut Assets,
    /// Latch was present before the first open. Skips all device I/O.
    pub latch_at_start: bool,
}

#[derive(Clone, Copy)]
enum Phase {
    Opening,
    Ours,
    Detached { until: Instant, backoff: Duration },
    Bootloader,
    Halted,
}

/// Main loop. `READY=1` is sent here; start-up checks happen before the call.
pub fn run_loop<S, Samp, Clk, Ntf, Stp, Lg, Op>(
    mut input: LoopInput<'_, Samp, Clk, Ntf, Stp, Lg, Op>,
) -> LoopExit
where
    S: LcdSink,
    Samp: Sampler,
    Clk: Clock,
    Ntf: Notifier,
    Stp: Stop,
    Lg: log::Sink,
    Op: FnMut() -> Result<S, SinkError>,
{
    input.notify.ready();
    let tick = tick_duration(input.config.writer.tick_s);
    let streaming = input.config.upload.mode == UploadMode::Stream;
    let frame_period = stream_period(input.config.upload.stream_fps);
    // Change mode keeps the upload gap. Stream mode is paced by `stream_fps`.
    let upload_gap = if streaming {
        Duration::ZERO
    } else {
        Duration::from_secs(input.config.upload.min_interval_s)
    };
    let fail_limit = input.config.upload.fail_limit;
    let mut deadline: Option<Instant> = None;
    let mut state: LoopState<S> = LoopState {
        policy: Policy::new(upload_gap),
        history: History::new(input.clock.mono()),
        sink: None,
        fail_streak: 0,
        watch: WatchDown::default(),
        op_at: None,
        follow_up_pending: false,
        last_pixels: None,
        stream_slot: 0,
        guard_at: None,
        stream_retry: None,
        grace_until: None,
        phase: if input.latch_at_start {
            crit(
                &mut input.log,
                "cooling guard latch is present; collect only",
            );
            Phase::Halted
        } else {
            Phase::Opening
        },
    };
    let mut prev_view: Option<View> = None;
    let stock_after = Duration::from_secs(input.config.snapshot.watch_down_stock_after_s);
    let restore_min = Duration::from_secs(input.config.snapshot.watch_down_restore_min_s);
    let mut last_wall: Option<SystemTime> = None;
    let mut last_mono: Option<Instant> = None;

    loop {
        // Checked here, between ticks, so no upload is in flight when the
        // loop leaves (GitHub #59). Dropping the sink closes both ports.
        if input.stop.requested() {
            info(&mut input.log, "stop requested; no upload in flight");
            input.notify.stopping();
            return LoopExit::Stopped;
        }

        let tick_start = input.clock.mono();
        let wall = input.clock.wall();
        let snap = input.sampler.sample(tick_start, wall);

        if matches!(
            state.phase,
            Phase::Opening | Phase::Ours | Phase::Detached { .. }
        ) && let (Some(prev_wall), Some(prev_mono)) = (last_wall, last_mono)
        {
            let wall_d = wall_delta(prev_wall, snap.t_wall, tick);
            let mono_d = snap.t_mono.saturating_duration_since(prev_mono);
            if is_wall_resume(wall_d, mono_d, tick) {
                let gap = wall_d.saturating_sub(mono_d);
                state.history.mark_gap(gap);
                info(
                    &mut input.log,
                    &format!(
                        "resume: wall jumped {wall_d:?} (mono {mono_d:?}), marking gap {gap:?}"
                    ),
                );
                state.sink = None;
                state.follow_up_pending = false;
                state.phase = Phase::Detached {
                    until: tick_start + BACKOFF_START,
                    backoff: BACKOFF_START,
                };
            }
        }
        last_wall = Some(snap.t_wall);
        last_mono = Some(snap.t_mono);

        state.history.add(snap.t_mono, snap.load);
        let view = presented_view(&snap, &state.history, &prev_view, input.config, streaming);
        prev_view = Some(view.clone());

        let opened = match state.phase {
            Phase::Halted | Phase::Bootloader => TickDev::Continue,
            Phase::Opening => try_open(&mut input, &mut state, tick_start),
            Phase::Detached { until, .. } => {
                if tick_start >= until {
                    try_open(&mut input, &mut state, tick_start)
                } else {
                    TickDev::Continue
                }
            }
            Phase::Ours => TickDev::Continue,
        };
        if matches!(state.phase, Phase::Ours) && state.sink.as_mut().is_some_and(LcdSink::blocked) {
            enter_halt(&mut state, &mut input.log);
        }
        if matches!(opened, TickDev::Continue) {
            apply_watch_down(
                &mut state,
                snap.ai != llama_core::sample::AiState::NoData,
                tick_start,
                stock_after,
                restore_min,
            );
        }
        let display = input.config.display;
        let outcome = match opened {
            TickDev::Exit(exit) => TickDev::Exit(exit),
            TickDev::Continue if matches!(state.phase, Phase::Ours) => ours_tick(
                &mut input, &mut state, &view, tick_start, fail_limit, &display, streaming,
            ),
            TickDev::Continue => TickDev::Continue,
        };
        match outcome {
            TickDev::Continue => {}
            TickDev::Exit(exit) => return exit,
        }

        input.notify.watchdog();
        if streaming {
            pace_stream(&mut input.clock, &mut deadline, tick_start, frame_period);
        } else {
            let elapsed = input.clock.mono().saturating_duration_since(tick_start);
            if let Some(rest) = tick.checked_sub(elapsed)
                && !rest.is_zero()
            {
                input.clock.sleep(rest);
            }
        }
    }
}

/// One stream frame's deadline. An overrun sleeps nothing and the next deadline
/// is one period from now, so missed frames are not replayed.
fn pace_stream(
    clock: &mut impl Clock,
    deadline: &mut Option<Instant>,
    tick_start: Instant,
    period: Duration,
) {
    let due = *deadline.get_or_insert(tick_start + period);
    let now = clock.mono();
    if let Some(wait) = due.checked_duration_since(now) {
        if !wait.is_zero() {
            clock.sleep(wait);
        }
        *deadline = Some(due + period);
    } else {
        *deadline = Some(now + period);
    }
}

fn presented_view(
    snap: &Snapshot,
    history: &History,
    previous: &Option<View>,
    config: &Config,
    streaming: bool,
) -> View {
    if !streaming {
        return present(snap, history, previous.as_ref(), config);
    }
    // The dial already has no hysteresis. Drop the ring hold so the gauge
    // tracks the latest mean each frame. Percent and temperature keep theirs.
    let mut carried = previous.clone();
    if let Some(view) = carried.as_mut() {
        view.ring_pct = None;
        view.ring_band = None;
    }
    present(snap, history, carried.as_ref(), config)
}

enum TickDev {
    Continue,
    Exit(LoopExit),
}

#[derive(Default)]
struct WatchDown {
    /// First tick of the current not-fresh episode.
    down_since: Option<Instant>,
    /// Uploads are held after a watch-down `ShowLiquid` until the snapshot is fresh.
    suppressed: bool,
    /// Last watch-down `ShowLiquid`.
    last_restore: Option<Instant>,
}

struct LoopState<S> {
    policy: Policy,
    history: History,
    sink: Option<S>,
    fail_streak: u32,
    watch: WatchDown,
    /// When the latest device operation started. The follow-up check waits.
    op_at: Option<Instant>,
    follow_up_pending: bool,
    /// Pixels of the frame last accepted by the sink. Stream mode skips a
    /// byte-identical successor.
    last_pixels: Option<Vec<u8>>,
    /// Next ping-pong slot, `0` or `1`.
    stream_slot: u8,
    /// Last stream-mode cooling check.
    guard_at: Option<Instant>,
    /// Stream mode after `UploadFailed`: no upload before `.0`; `.1` is the
    /// wait that set it. Cleared by a success and by a new open.
    stream_retry: Option<(Instant, Duration)>,
    /// Stream mode: end of the startup grace ([`START_GRACE`]) set by each
    /// open. Cleared by the first success and by the first failure after it.
    grace_until: Option<Instant>,
    phase: Phase,
}

fn note_operation<S>(state: &mut LoopState<S>, now: Instant) {
    state.op_at = Some(now);
    state.follow_up_pending = true;
}

fn try_open<S, Samp, Clk, Ntf, Stp, Lg, Op>(
    input: &mut LoopInput<'_, Samp, Clk, Ntf, Stp, Lg, Op>,
    state: &mut LoopState<S>,
    now: Instant,
) -> TickDev
where
    S: LcdSink,
    Op: FnMut() -> Result<S, SinkError>,
    Lg: log::Sink,
{
    match (input.open)() {
        Ok(mut lcd) => {
            lcd.needs_reupload();
            state.policy.reset_on_open();
            state.last_pixels = None;
            state.stream_slot = 0;
            state.guard_at = None;
            state.stream_retry = None;
            state.grace_until = Some(now + START_GRACE);
            state.sink = Some(lcd);
            state.phase = Phase::Ours;
            note_operation(state, now);
            TickDev::Continue
        }
        Err(SinkError::DeviceUnavailable) => {
            detach_unavailable(state, &mut input.log, now);
            TickDev::Continue
        }
        Err(SinkError::DeviceInBootloader) => {
            crit(&mut input.log, "device is in the bootloader; collect only");
            state.sink = None;
            state.phase = Phase::Bootloader;
            TickDev::Continue
        }
        Err(SinkError::Halted) => {
            enter_halt(state, &mut input.log);
            TickDev::Continue
        }
        Err(SinkError::Fence) => {
            crit(&mut input.log, "command fence");
            TickDev::Exit(LoopExit::Fence)
        }
        Err(SinkError::Fatal(reason)) => {
            crit(&mut input.log, &format!("fatal: {reason}"));
            TickDev::Exit(LoopExit::Fatal)
        }
        Err(SinkError::UploadFailed(_) | SinkError::TransferAborted) => {
            detach_unavailable(state, &mut input.log, now);
            TickDev::Continue
        }
    }
}

const STREAM_GUARD: Duration = Duration::from_secs(1);

/// First wait before a stream upload is retried after `UploadFailed`.
///
/// Doubles on each failure in a row, capped at [`STREAM_RETRY_CAP`]. Change
/// mode needs no such wait: its policy already spaces failed attempts by
/// `min_interval_s` (LLD Open 0f).
const STREAM_RETRY_START: Duration = Duration::from_secs(2);

/// Cap of the stream retry wait.
const STREAM_RETRY_CAP: Duration = Duration::from_secs(30);

/// Stream mode: how long after an open a refused or unanswered upload does not
/// count toward `fail_limit`, until the first success. Right after the previous
/// process restored stock the Kraken refuses `DeleteBucket` for more than 6 s
/// but less than a minute (GitHub #14).
const START_GRACE: Duration = Duration::from_secs(60);

/// The wait after one more stream `UploadFailed` in a row.
fn next_stream_retry(last: Option<Duration>) -> Duration {
    last.map_or(STREAM_RETRY_START, |last| {
        last.saturating_mul(2).min(STREAM_RETRY_CAP)
    })
}

fn stream_frame<S, Samp, Clk, Ntf, Stp, Lg, Op>(
    input: &mut LoopInput<'_, Samp, Clk, Ntf, Stp, Lg, Op>,
    state: &mut LoopState<S>,
    view: &View,
    now: Instant,
    fail_limit: u32,
    display: &crate::config::DisplayCfg,
) -> TickDev
where
    S: LcdSink,
    Lg: log::Sink,
{
    if let TickDev::Exit(exit) = stream_guard(input, state, now) {
        return TickDev::Exit(exit);
    }
    if !matches!(state.phase, Phase::Ours) || state.watch.suppressed {
        return TickDev::Continue;
    }
    if state.stream_retry.is_some_and(|(at, _)| now < at) {
        return TickDev::Continue;
    }
    let frame = render::render(view, display, input.assets);
    if state.last_pixels.as_deref() == Some(frame.0.data()) {
        return TickDev::Continue;
    }
    let slot = state.stream_slot;
    state.policy.mark_attempted(now);
    note_operation(state, now);
    let shown = state
        .sink
        .as_mut()
        .map(|lcd| lcd.show_slot(slot, &frame))
        .unwrap_or(Err(SinkError::DeviceUnavailable));
    match shown {
        Ok(()) => {
            state.stream_slot ^= 1;
            state.last_pixels = Some(frame.0.data().to_vec());
            state.policy.mark_uploaded(view, now);
            state.fail_streak = 0;
            state.stream_retry = None;
            state.grace_until = None;
            TickDev::Continue
        }
        Err(SinkError::UploadFailed(
            err @ (UploadFailed::Refused(_) | UploadFailed::NoReply(_)),
        )) if state.grace_until.is_some_and(|end| now < end) => {
            // Startup grace: back off without counting. The retry never lands
            // past the window's end, so the counted phase starts on time.
            let end = state.grace_until.unwrap_or(now);
            let left = end.saturating_duration_since(now);
            let wait = next_stream_retry(state.stream_retry.map(|(_, last)| last)).min(left);
            state.stream_retry = Some((now + wait, wait));
            log::emit(
                &mut input.log,
                Priority::Warning,
                &format!(
                    "upload failed: {err}; device settling, retry in {}s (grace, {}s left)",
                    wait.as_secs(),
                    left.as_secs()
                ),
            );
            TickDev::Continue
        }
        Err(SinkError::UploadFailed(err)) => {
            if let TickDev::Exit(exit) = count_upload_fail(state, &mut input.log, fail_limit) {
                return TickDev::Exit(exit);
            }
            // The first failure after the grace window starts the
            // backoff afresh, so counting runs exactly as without a grace.
            let last = if state.grace_until.is_some_and(|end| now >= end) {
                state.grace_until = None;
                None
            } else {
                state.stream_retry.map(|(_, last)| last)
            };
            // Back off instead of retrying on the next frame. At 10 fps an
            // unspaced retry spent `fail_limit` within 300 ms, so a device
            // still settling after the previous process restored stock
            // exited the writer (GitHub #14).
            let wait = next_stream_retry(last);
            state.stream_retry = Some((now + wait, wait));
            log::emit(
                &mut input.log,
                Priority::Warning,
                &format!(
                    "upload failed: {err}; retry in {}s ({}/{fail_limit})",
                    wait.as_secs(),
                    state.fail_streak
                ),
            );
            TickDev::Continue
        }
        Err(SinkError::TransferAborted) => {
            if let TickDev::Exit(exit) = count_upload_fail(state, &mut input.log, fail_limit) {
                return TickDev::Exit(exit);
            }
            detach_unavailable(state, &mut input.log, now);
            TickDev::Continue
        }
        Err(SinkError::DeviceUnavailable) => {
            detach_unavailable(state, &mut input.log, now);
            TickDev::Continue
        }
        Err(SinkError::Halted) => {
            enter_halt(state, &mut input.log);
            TickDev::Continue
        }
        Err(SinkError::Fence) => {
            crit(&mut input.log, "command fence");
            TickDev::Exit(LoopExit::Fence)
        }
        Err(SinkError::Fatal(reason)) => {
            crit(&mut input.log, &format!("fatal: {reason}"));
            TickDev::Exit(LoopExit::Fatal)
        }
        Err(SinkError::DeviceInBootloader) => {
            crit(&mut input.log, "device is in the bootloader; collect only");
            state.sink = None;
            state.phase = Phase::Bootloader;
            TickDev::Continue
        }
    }
}

fn stream_guard<S, Samp, Clk, Ntf, Stp, Lg, Op>(
    input: &mut LoopInput<'_, Samp, Clk, Ntf, Stp, Lg, Op>,
    state: &mut LoopState<S>,
    now: Instant,
) -> TickDev
where
    S: LcdSink,
    Lg: log::Sink,
{
    let due = state
        .guard_at
        .is_none_or(|at| now.saturating_duration_since(at) >= STREAM_GUARD);
    if !due {
        return TickDev::Continue;
    }
    state.guard_at = Some(now);
    match state.sink.as_mut().map(LcdSink::pace_guard) {
        Some(Ok(())) | None => TickDev::Continue,
        Some(Err(SinkError::Halted)) => {
            enter_halt(state, &mut input.log);
            TickDev::Continue
        }
        Some(Err(SinkError::DeviceUnavailable)) => {
            detach_unavailable(state, &mut input.log, now);
            TickDev::Continue
        }
        Some(Err(SinkError::Fence)) => {
            crit(&mut input.log, "command fence");
            TickDev::Exit(LoopExit::Fence)
        }
        Some(Err(SinkError::Fatal(reason))) => {
            crit(&mut input.log, &format!("fatal: {reason}"));
            TickDev::Exit(LoopExit::Fatal)
        }
        Some(Err(_)) => TickDev::Continue,
    }
}

fn ours_tick<S, Samp, Clk, Ntf, Stp, Lg, Op>(
    input: &mut LoopInput<'_, Samp, Clk, Ntf, Stp, Lg, Op>,
    state: &mut LoopState<S>,
    view: &View,
    now: Instant,
    fail_limit: u32,
    display: &crate::config::DisplayCfg,
    streaming: bool,
) -> TickDev
where
    S: LcdSink,
    Lg: log::Sink,
{
    if state.sink.is_none() {
        state.phase = Phase::Opening;
        return TickDev::Continue;
    }
    if streaming {
        if let TickDev::Exit(exit) = stream_frame(input, state, view, now, fail_limit, display) {
            return TickDev::Exit(exit);
        }
    } else if !state.watch.suppressed && state.policy.decide(view, now) == Decision::Upload {
        let frame = render::render(view, display, input.assets);
        state.policy.mark_attempted(now);
        note_operation(state, now);
        let shown = state
            .sink
            .as_mut()
            .map(|lcd| lcd.show(&frame))
            .unwrap_or(Err(SinkError::DeviceUnavailable));
        match shown {
            Ok(()) => {
                state.policy.mark_uploaded(view, now);
                state.fail_streak = 0;
            }
            Err(SinkError::UploadFailed(_)) => {
                if let TickDev::Exit(exit) = count_upload_fail(state, &mut input.log, fail_limit) {
                    return TickDev::Exit(exit);
                }
            }
            Err(SinkError::TransferAborted) => {
                if let TickDev::Exit(exit) = count_upload_fail(state, &mut input.log, fail_limit) {
                    return TickDev::Exit(exit);
                }
                detach_unavailable(state, &mut input.log, now);
                return TickDev::Continue;
            }
            Err(SinkError::DeviceUnavailable) => {
                detach_unavailable(state, &mut input.log, now);
                return TickDev::Continue;
            }
            Err(SinkError::Halted) => {
                enter_halt(state, &mut input.log);
                return TickDev::Continue;
            }
            Err(SinkError::Fence) => {
                crit(&mut input.log, "command fence");
                return TickDev::Exit(LoopExit::Fence);
            }
            Err(SinkError::Fatal(reason)) => {
                crit(&mut input.log, &format!("fatal: {reason}"));
                return TickDev::Exit(LoopExit::Fatal);
            }
            Err(SinkError::DeviceInBootloader) => {
                crit(&mut input.log, "device is in the bootloader; collect only");
                state.sink = None;
                state.phase = Phase::Bootloader;
                return TickDev::Continue;
            }
        }
    }

    // Stream mode checks cooling on its own 1 Hz cadence (`pace_guard`), which
    // keeps a baseline between checks. The 2 s follow-up consumes that baseline,
    // so it stays on the change-mode path.
    if !streaming
        && matches!(state.phase, Phase::Ours)
        && state.follow_up_pending
        && state
            .op_at
            .is_some_and(|at| crate::device::follow_up_due(at, now))
    {
        state.follow_up_pending = false;
        let ticked = state.sink.as_mut().map(LcdSink::tick);
        match ticked {
            Some(Ok(())) | None => {}
            Some(Err(SinkError::Halted)) => enter_halt(state, &mut input.log),
            Some(Err(SinkError::DeviceUnavailable)) => {
                detach_unavailable(state, &mut input.log, now);
            }
            Some(Err(SinkError::Fence)) => {
                crit(&mut input.log, "command fence");
                return TickDev::Exit(LoopExit::Fence);
            }
            Some(Err(SinkError::Fatal(reason))) => {
                crit(&mut input.log, &format!("fatal: {reason}"));
                return TickDev::Exit(LoopExit::Fatal);
            }
            Some(Err(_)) => {}
        }
    }
    TickDev::Continue
}

fn count_upload_fail<S: LcdSink>(
    state: &mut LoopState<S>,
    log: &mut impl log::Sink,
    fail_limit: u32,
) -> TickDev {
    state.fail_streak = state.fail_streak.saturating_add(1);
    if state.fail_streak >= fail_limit {
        crit(log, "upload fail_limit reached; restoring stock");
        if let Some(lcd) = state.sink.as_mut() {
            lcd.restore_stock();
        }
        TickDev::Exit(LoopExit::Restored)
    } else {
        TickDev::Continue
    }
}

fn detach_unavailable<S>(state: &mut LoopState<S>, log: &mut impl log::Sink, now: Instant) {
    let backoff = match state.phase {
        Phase::Detached { backoff, .. } => next_backoff(backoff),
        _ => BACKOFF_START,
    };
    info(
        log,
        &format!(
            "device unavailable; detaching, backoff {}s",
            backoff.as_secs()
        ),
    );
    state.sink = None;
    state.follow_up_pending = false;
    state.phase = Phase::Detached {
        until: now + backoff,
        backoff,
    };
}

fn apply_watch_down<S: LcdSink>(
    state: &mut LoopState<S>,
    fresh: bool,
    now: Instant,
    stock_after: Duration,
    restore_min: Duration,
) {
    if fresh {
        if state.watch.down_since.is_some() {
            state.watch.down_since = None;
            state.watch.suppressed = false;
            state.policy.force();
        }
        return;
    }
    if state.watch.down_since.is_none() {
        state.watch.down_since = Some(now);
    }
    let since = state.watch.down_since.unwrap_or(now);
    if now.saturating_duration_since(since) < stock_after || state.watch.suppressed {
        return;
    }
    let restore_ok = state
        .watch
        .last_restore
        .is_none_or(|at| now.saturating_duration_since(at) >= restore_min);
    // The cap blocks ShowLiquid only. The no-data frame still uploads until
    // a restore is actually sent.
    if !restore_ok {
        return;
    }
    if !matches!(state.phase, Phase::Ours) || state.sink.is_none() {
        return;
    }
    if !state.policy.allows_attempt(now) {
        return;
    }
    if let Some(lcd) = state.sink.as_mut() {
        lcd.restore_stock();
    }
    state.policy.mark_attempted(now);
    state.watch.last_restore = Some(now);
    state.watch.suppressed = true;
    note_operation(state, now);
}

fn enter_halt<S>(state: &mut LoopState<S>, log: &mut impl log::Sink) {
    crit(log, "cooling guard halted; collect only");
    state.sink = None;
    state.phase = Phase::Halted;
}

fn crit(log: &mut impl log::Sink, message: &str) {
    log::emit(log, Priority::Crit, message);
}

fn info(log: &mut impl log::Sink, message: &str) {
    log::emit(log, Priority::Info, message);
}

/// Production `run`. Always uses [`CheckEnv::production`].
pub fn run(config_path: &Path, trace_hid: bool) -> i32 {
    let env = CheckEnv::production();
    let config = match Config::load_validated(config_path) {
        Ok(config) => config,
        Err(err) => {
            crit(&mut log::Stderr, &err.to_string());
            return 1;
        }
    };
    let mut assets = match self_check(&config, &env) {
        Ok(assets) => assets,
        Err(err) => {
            crit(&mut log::Stderr, &err.to_string());
            return 1;
        }
    };
    let latch = latch_present(&env.state_dir);
    let rotate = config.display.rotate_deg;
    let mut reader = SnapshotReader::open(Duration::from_secs_f64(config.snapshot.stale_after_s));
    let mut clock = RealClock;
    let mut notify = SdNotify;
    run_loop(LoopInput {
        config: &config,
        sampler: &mut reader,
        clock: &mut clock,
        notify: &mut notify,
        stop: StopFlag::new(Path::new(STOP_FLAG)),
        log: log::Stderr,
        open: || KrakenLcd::connect(rotate, trace_hid),
        assets: &mut assets,
        latch_at_start: latch,
    })
    .code()
}

/// Production `restore-stock`. Never constructs NVML.
pub fn restore_stock(config_path: &Path, trace_hid: bool) -> i32 {
    let state_dir = Path::new(STATE_DIR);
    if latch_present(state_dir) {
        crit(
            &mut log::Stderr,
            "restore skipped; cooling guard latch is present",
        );
        return 0;
    }
    let env = CheckEnv::production();
    let config = match Config::load_validated(config_path) {
        Ok(config) => config,
        Err(err) => {
            crit(&mut log::Stderr, &err.to_string());
            return 1;
        }
    };
    if let Err(err) = self_check(&config, &env) {
        crit(&mut log::Stderr, &err.to_string());
        return 1;
    }
    let request = OpenRequest {
        sys_root: Path::new(SYS_ROOT),
        state_dir,
        rotate_deg: config.display.rotate_deg,
        trace_hid,
    };
    restore_stock_on(&request, open_resolved_hid)
}

/// Injected `restore-stock`. Latch present → 0, no open. Device absent → 0.
pub fn restore_stock_on<H: HidPort>(
    request: &OpenRequest<'_>,
    open_hid: impl FnOnce(&Path) -> Result<H, PortError>,
) -> i32 {
    if latch_present(request.state_dir) {
        crit(
            &mut log::Stderr,
            "restore skipped; cooling guard latch is present",
        );
        return 0;
    }
    let _lcd = KrakenLcd::<H, NoBulk>::open_restore(request, open_hid);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injected_state_dir_latch_skips_the_open() {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let pid = rustix::process::getpid().as_raw_nonzero().get();
        let dir = std::env::temp_dir().join(format!("kraken-lcd-restore-latch-{pid}-{n}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("halted"), b"halted\n").unwrap();
        let sys = dir.join("sys");
        std::fs::create_dir_all(&sys).unwrap();
        let request = OpenRequest {
            sys_root: &sys,
            state_dir: &dir,
            rotate_deg: 0,
            trace_hid: false,
        };
        let mut opened = false;
        let code = restore_stock_on(&request, |_path| {
            opened = true;
            Err::<crate::device::HidLink, _>(PortError::unavailable("test must not open"))
        });
        assert_eq!(code, 0);
        assert!(!opened, "a latched state dir must not open the device");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn backoff_doubles_until_the_cap() {
        let mut wait = BACKOFF_START;
        let mut seen = vec![wait];
        while wait < BACKOFF_CAP {
            wait = next_backoff(wait);
            seen.push(wait);
        }
        assert_eq!(
            seen,
            [
                Duration::from_secs(10),
                Duration::from_secs(20),
                Duration::from_secs(40),
                Duration::from_secs(80),
                Duration::from_secs(160),
                Duration::from_secs(300),
            ]
        );
        assert_eq!(next_backoff(BACKOFF_CAP), BACKOFF_CAP);
    }

    #[test]
    fn stream_retry_doubles_until_the_cap() {
        let mut wait = next_stream_retry(None);
        let mut seen = vec![wait];
        while wait < STREAM_RETRY_CAP {
            wait = next_stream_retry(Some(wait));
            seen.push(wait);
        }
        assert_eq!(seen, [2, 4, 8, 16, 30].map(Duration::from_secs));
        assert_eq!(next_stream_retry(Some(STREAM_RETRY_CAP)), STREAM_RETRY_CAP);
    }

    #[test]
    fn latch_present_is_fail_closed() {
        let dir = std::env::temp_dir().join(format!(
            "t14-latch-unit-{}",
            rustix::process::getpid().as_raw_nonzero().get()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        assert!(!latch_present(&dir));
        std::fs::write(dir.join("halted"), b"halt\n").expect("latch");
        assert!(latch_present(&dir));
        std::fs::remove_file(dir.join("halted")).expect("clear");
        assert!(!latch_present(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wall_jump_limit_is_three_ticks() {
        let tick = Duration::from_secs(2);
        assert_eq!(wall_jump_limit(tick), Duration::from_secs(6));
        assert_eq!(
            wall_jump_limit(Duration::from_millis(500)),
            Duration::from_secs(2),
            "resume detection is at least 2s, even when 3 ticks is shorter"
        );
        let t0 = SystemTime::UNIX_EPOCH;
        let t1 = t0 + Duration::from_secs(7);
        assert_eq!(wall_delta(t0, t1, tick), Duration::from_secs(7));
        assert_eq!(wall_delta(t1, t0, tick), tick);
    }

    #[test]
    fn resume_is_wall_minus_mono_over_three_ticks() {
        let tick = Duration::from_secs(2);
        let ten = Duration::from_secs(10);
        let forty = Duration::from_secs(40);
        assert!(
            !is_wall_resume(ten, ten, tick),
            "a 10s slow tick with no suspend is not a resume"
        );
        assert!(
            is_wall_resume(forty, ten, tick),
            "10s of work then a 30s suspend is a resume"
        );
    }
}
