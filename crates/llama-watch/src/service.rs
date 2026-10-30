//! 10 Hz watcher loop: poll, sample, publish, draw, then notify.
//!
//! `READY=1` is sent after start-up. `WATCHDOG=1` is sent only when the four
//! tick steps return. `STOPPING=1` is sent when [`Stop::requested`] becomes
//! true, which is how a test stands in for SIGINT or SIGTERM.
//!
//! In [`run`] the draw step is a [`FrameWriter`]: it hands the frame to the
//! tty writer thread and returns, so a console write that blocks (unblank,
//! Scroll Lock) never holds back publish or the watchdog (#42).

use std::collections::{BTreeMap, HashMap};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sd_notify::NotifyState;
use thiserror::Error;

use crate::activity::ActivityRow;
use crate::collector::{GpuProcess, GpuProcesses, WatchCollector, WatchSample};
use crate::config::{ChartGlyphs, Config, ConfigError, ValidWatchConfig};
use crate::grid_out::{GridTap, GridWriter};
use crate::localtime::{HostZone, Zone, format_local, local_request_time};
use crate::metrics::EngineLive;
use crate::poller::{self, CaptureView, LlamaDetail, ModelEngineLive, ModelSetup, PollLatencies};
use crate::publish::{Extras, PublishError, Publisher, SlotCtx};
use crate::setup_rules::{LiveCtx, Rules};
use crate::slots::{SlotView, pick_slot};
use crate::sources::Roots;
use crate::sources::gpu::{GpuBackend, NvidiaGpu};
use crate::sources::proc::count_cpus;
use crate::tty::chart::TokenChart;
use crate::tty::ctx_history::CtxBook;
use crate::tty::grid::Cell;
use crate::tty::layout::{
    self, Activity, HeaderClaim, HeaderDisplaced, HeaderGpu, HealthSeg, HealthStatus, InFlight,
    SetupView, Slot, TtyModel, WatchState,
};
use crate::tty::sanitize::sanitize;
use crate::tty::term::{self, ConsoleBlank, Term};
use crate::tty::writer::FrameWriter;
use llama_core::backend::{self, Backend, KvUsage};
use llama_core::log::{self, Priority, Sink};
use llama_core::sample::{AiState, LlamaView, ModelInfo, Snapshot, SourceId};
use llama_core::wire::{SourceWire, Sources};

/// Injected clock. Production sleeps; tests advance a fake instant.
pub trait Clock {
    /// `CLOCK_MONOTONIC`.
    fn mono(&self) -> Instant;
    /// Wall clock used for the header and for suspend-unrelated timestamps.
    fn wall(&self) -> SystemTime;
    /// Wait `d`, or advance a fake clock by `d`.
    fn sleep(&mut self, d: Duration);
    /// Zone for the header clock, the down label and RECENT's TIME (#48).
    /// `mono` is now; the host clock rereads its zone at most once a
    /// minute. Fake clocks keep the default, UTC.
    fn zone(&mut self, mono: Instant) -> &Zone {
        let _ = mono;
        Zone::utc()
    }
}

/// sd-notify seam. Unset `NOTIFY_SOCKET` makes the production impl a no-op.
pub trait Notifier {
    /// `READY=1` after start-up.
    fn ready(&mut self);
    /// `WATCHDOG=1` after steps 1–4 returned.
    fn watchdog(&mut self);
    /// `STOPPING=1` when a stop was requested.
    fn stopping(&mut self);
}

/// Stop request. Tests trip this in place of SIGINT or SIGTERM.
pub trait Stop {
    /// `true` when the loop should send `STOPPING=1` and leave.
    fn requested(&self) -> bool;
}

/// Latest llama sample. [`Self::poll`] yields at most one queued item.
///
/// The loop drains the feed each tick and keeps the newest item, so a feed
/// that queued two samples paints the second one.
pub trait LlamaFeed {
    /// The next unread sample, or `None` when the feed is empty.
    fn poll(&mut self) -> Option<(LlamaView, LlamaDetail)>;
}

/// Step 2. A test impl may panic or block; the watchdog is then skipped.
pub trait SampleStep {
    /// One collector tick for `llama`.
    fn sample(&mut self, mono: Instant, wall: SystemTime, llama: &LlamaView) -> WatchSample;
}

/// Step 3.
pub trait PublishStep {
    /// Validate and publish. [`PublishError::Write`] is logged by the loop.
    /// `extras` are the llama-metrics numbers that are not on `snapshot` (#11).
    fn publish(
        &mut self,
        snapshot: &Snapshot,
        llama: &LlamaView,
        extras: &Extras,
    ) -> Result<(), PublishError>;
}

/// Step 4. Builds nothing itself: the loop has already mapped the [`TtyModel`].
pub trait RenderStep {
    /// Diff `model` to the console. An unchanged frame writes no bytes.
    fn draw(&mut self, model: &TtyModel, now: Instant) -> io::Result<()>;

    /// Clean exit: undo what drawing changed on the console beyond the
    /// cells (the llama palette, #26). Called once when the loop stops.
    fn finish(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Make the next [`Self::draw`] a full repaint. The tty writer calls it
    /// after a stalled write (#42).
    fn repaint(&mut self) {}
}

impl<B, L> SampleStep for WatchCollector<'_, B, L>
where
    B: GpuBackend,
    L: Sink,
{
    fn sample(&mut self, mono: Instant, wall: SystemTime, llama: &LlamaView) -> WatchSample {
        WatchCollector::sample(self, mono, wall, llama)
    }
}

impl<L: Sink> PublishStep for Publisher<L> {
    fn publish(
        &mut self,
        snapshot: &Snapshot,
        llama: &LlamaView,
        extras: &Extras,
    ) -> Result<(), PublishError> {
        Publisher::publish_with(self, snapshot, llama, extras)
    }
}

impl<W: Write> RenderStep for Term<W> {
    fn draw(&mut self, model: &TtyModel, now: Instant) -> io::Result<()> {
        let grid = layout::layout(model, self.cols(), self.rows());
        // The tap copies this grid when the file is the console frame.
        // Exact colours, or a different `[grid] size`, clone the model
        // instead and lay it out on the tap's thread, so this write does
        // not compute the ramp and does not wait.
        if let Some(tap) = self.grid_tap() {
            tap.offer_frame(model, &grid);
        }
        self.render(&grid, now)
    }

    fn finish(&mut self) -> io::Result<()> {
        self.restore_console()
    }

    fn repaint(&mut self) {
        self.force_repaint();
    }
}

/// Host clock, in the host's local zone.
#[derive(Default)]
pub struct RealClock {
    zone: HostZone,
}

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

    fn zone(&mut self, mono: Instant) -> &Zone {
        self.zone.get(mono)
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

/// Always continues. Same as the writer's `NeverStop`: no signal crate is
/// pinned and `unsafe` is forbidden, so SIGTERM keeps the default action.
/// `TTYReset` and `TTYVHangup` restore the console. `STOPPING=1` is sent
/// only when a test (or other caller) sets a different [`Stop`].
pub struct NeverStop;

impl Stop for NeverStop {
    fn requested(&self) -> bool {
        false
    }
}

/// Takes the newest poller sample. A second take is empty until the next publish.
///
/// [`Self::disabled`] never yields: `[llama] enabled = false` starts no poller.
pub struct SlotFeed {
    rx: Option<poller::SampleRx>,
}

impl SlotFeed {
    /// Wrap the slot from [`poller::spawn`].
    #[must_use]
    pub fn new(rx: poller::SampleRx) -> Self {
        Self { rx: Some(rx) }
    }

    /// A feed with no poller behind it.
    #[must_use]
    pub fn disabled() -> Self {
        Self { rx: None }
    }
}

impl LlamaFeed for SlotFeed {
    fn poll(&mut self) -> Option<(LlamaView, LlamaDetail)> {
        self.rx.as_ref()?.take()
    }
}

/// Why [`prepare`] refused to start the loop.
#[derive(Debug, Error)]
pub enum PrepareError {
    /// `watch.toml` could not be read or validated.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// The snapshot directory could not be opened.
    #[error(transparent)]
    Snapshot(#[from] PublishError),
}

/// Validated config and an open snapshot directory.
///
/// The directory fd is open. Dropping this drops the fd. [`run`] keeps it
/// for the life of the process.
pub struct Prepared {
    /// Validated watcher config.
    pub config: ValidWatchConfig,
    publisher: Publisher<log::Stderr>,
}

/// Read `path`, validate it against `nproc`, and open `snapshot_dir`.
///
/// Does not start the poller and does not sample the host. Failing to open
/// the directory is fatal for [`run`].
pub fn prepare(path: &Path, nproc: u32, snapshot_dir: &Path) -> Result<Prepared, PrepareError> {
    let config = Config::load_validated(path, nproc)?;
    let publisher = Publisher::open(snapshot_dir, log::Stderr)?;
    Ok(Prepared { config, publisher })
}

/// Online logical CPUs from the `cpuN` lines of `{proc}/stat`, or 1 when that read fails.
///
/// `cpu_top_k` validation uses this. A unit pinned with `CPUAffinity` still
/// sees every `cpuN` line the kernel publishes.
#[must_use]
pub fn host_nproc(proc_root: &Path) -> u32 {
    count_cpus(proc_root).unwrap_or(1)
}

/// `Some(2)` when the process effective uid is root.
#[must_use]
pub fn root_exit(euid_is_root: bool) -> Option<i32> {
    euid_is_root.then_some(2)
}

/// Parsed `llama-watch run` arguments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunArgs {
    /// `--config PATH`.
    pub config: PathBuf,
    /// `--no-text`: force `tty.show_text = false` whatever the file says.
    pub no_text: bool,
}

/// `llama-watch run --config PATH [--no-text]`, flags in either order.
/// Anything else is [`Usage`].
pub fn parse_args<I, S>(args: I) -> Result<RunArgs, Usage>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let args: Vec<String> = args.into_iter().map(|s| s.as_ref().to_owned()).collect();
    let Some((command, rest)) = args.split_first() else {
        return Err(Usage);
    };
    if command != "run" {
        return Err(Usage);
    }
    let mut config: Option<PathBuf> = None;
    let mut no_text = false;
    let mut rest = rest.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--config" if config.is_none() => {
                let path = rest.next().ok_or(Usage)?;
                if path.is_empty() || path.starts_with("--") {
                    return Err(Usage);
                }
                config = Some(PathBuf::from(path));
            }
            "--no-text" if !no_text => no_text = true,
            _ => return Err(Usage),
        }
    }
    Ok(RunArgs {
        config: config.ok_or(Usage)?,
        no_text,
    })
}

/// The process was not invoked as `run --config PATH [--no-text]`.
#[derive(Debug, Error)]
#[error("usage: llama-watch run --config PATH [--no-text]")]
pub struct Usage;

/// Inputs for [`run_loop`]. The caller owns `config` for the collector's borrow.
pub struct LoopInput<'a, Feed, Samp, Pub, Rend, Clk, Ntf, Stp, Lg> {
    /// Validated config. The tick period is `collector.tick_s`.
    pub config: &'a ValidWatchConfig,
    /// Poller hand-off. Drained to the newest sample each tick.
    pub feed: Feed,
    /// Host sampler.
    pub sampler: Samp,
    /// Snapshot publisher.
    pub publisher: Pub,
    /// Console diff.
    pub render: Rend,
    /// Tick clock.
    pub clock: Clk,
    /// sd-notify.
    pub notify: Ntf,
    /// SIGINT / SIGTERM stand-in.
    pub stop: Stp,
    /// Journal sink for the write-error transition.
    pub log: Lg,
    /// Proc and sys roots for the TTY's memory and host lines.
    pub roots: &'a Roots,
    /// Monotonic time at process start, for the uptime field.
    pub started: Instant,
}

/// Why the loop ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoopExit {
    /// Stop requested. Exit 0.
    Stopped,
}

impl LoopExit {
    /// Process exit code.
    #[must_use]
    pub fn code(self) -> i32 {
        match self {
            Self::Stopped => 0,
        }
    }
}

/// Main loop. Sends `READY=1` before the first tick.
///
/// Each tick: drain the llama feed to its newest sample, sample the host,
/// publish, draw, then `WATCHDOG=1`. A panic or a hang in those steps skips
/// the watchdog because control never reaches it.
pub fn run_loop<Feed, Samp, Pub, Rend, Clk, Ntf, Stp, Lg>(
    mut input: LoopInput<'_, Feed, Samp, Pub, Rend, Clk, Ntf, Stp, Lg>,
) -> LoopExit
where
    Feed: LlamaFeed,
    Samp: SampleStep,
    Pub: PublishStep,
    Rend: RenderStep,
    Clk: Clock,
    Ntf: Notifier,
    Stp: Stop,
    Lg: Sink,
{
    input.notify.ready();
    let tick = Duration::from_secs_f64(input.config.collector.tick_s);
    let roots = input.roots.clone();
    let ctx = FrameCtx {
        output_cap: usize::try_from(input.config.llama.output_tail_chars).unwrap_or(usize::MAX),
        input_cap: usize::try_from(input.config.llama.input_tail_chars).unwrap_or(usize::MAX),
        gen_ceiling: input.config.tty.gen_ceiling_tps,
        prompt_ceiling: input.config.tty.prompt_ceiling_tps,
        chart_bucket_s: input.config.tty.chart_bucket_s,
        chart_glyphs: input.config.tty.chart_glyphs,
        bar_segments: input.config.tty.bar_segments,
        ctx_history_h: input.config.tty.ctx_history_h,
        started: input.started,
        host: host_label(&roots.proc),
        cpu_cores: cpu_cores(&roots.proc),
        mem_total_bytes: mem_total_bytes(&roots.proc),
        llama_enabled: input.config.llama.enabled,
        show_text: input.config.tty.show_text,
        setup: Rules::compile(&input.config.setup).unwrap_or_default(),
        temps_look: crate::sources::temps::TempLook::new(&input.config.temps),
        jobs_stale_s: input.config.jobs.stale_s,
        disk_ceiling_bps: input.config.tty.disk_ceiling_bps,
        net_ceiling_bps: input.config.tty.net_ceiling_bps,
    };
    let mut state = TickState::new(
        input.config.tty.chart_bucket_s,
        input.config.tty.ctx_history_h,
    );
    let mut next_deadline = input.clock.mono();

    loop {
        if input.stop.requested() {
            input.notify.stopping();
            if let Err(err) = input.render.finish() {
                log::emit(
                    &mut input.log,
                    Priority::Warning,
                    &format!("console restore failed: {err}"),
                );
            }
            return LoopExit::Stopped;
        }
        let mono = input.clock.mono();
        let wall = input.clock.wall();
        tick_once(&mut input, &mut state, mono, wall, &ctx);
        input.notify.watchdog();
        next_deadline += tick;
        let now = input.clock.mono();
        if now > next_deadline {
            // Overrun: start the next tick immediately. Later periods are
            // measured from this moment.
            next_deadline = now;
        } else {
            let wait = next_deadline.saturating_duration_since(now);
            if !wait.is_zero() {
                input.clock.sleep(wait);
            }
        }
    }
}

struct TickState {
    llama: LlamaView,
    detail: LlamaDetail,
    heard: bool,
    published: u64,
    write_down: bool,
    draw_down: bool,
    replay: OutReplay,
    down_since: Option<SystemTime>,
    /// Monotonic time of the last publish that replaced the snapshot file.
    published_at: Option<Instant>,
    chart: TokenChart,
    /// Tick `mono` of the last poll that replaced `llama`.
    llama_at: Option<Instant>,
    /// Per-slot context history for the SLOTS sparklines (T53).
    ctx_history: CtxBook,
    /// When each model (by name) entered llama-swap `stopping`.
    stopping_since: HashMap<String, Instant>,
    /// When each model (by name) entered llama-swap `starting`.
    starting_since: HashMap<String, Instant>,
    /// Requests still running, for RECENT (#75).
    flights: crate::inflight::Tracker,
    /// Each model's context size by llama-swap key, kept after it unloads
    /// so its RECENT rows keep their scale (#75).
    ctx_sizes: BTreeMap<String, u64>,
    /// Last non-zero decode rate for the dim GEN digits (#163).
    gen_last: Option<f64>,
    /// IN/OUT burst scroll (#188).
    motion: crate::tty::scroll::Motion,
}

impl TickState {
    fn new(chart_bucket_s: u64, ctx_history_h: u32) -> Self {
        Self {
            llama: LlamaView {
                ai: AiState::Down,
                models: Vec::new(),
                decoded_total: None,
                prompt_total: None,
            },
            detail: LlamaDetail {
                slots: Vec::new(),
                activity: Vec::new(),
                gen_tps: None,
                prompt_tps: None,
                latencies: PollLatencies::default(),
                prompt_cache: Vec::new(),
                capture: None,
                setup: Vec::new(),
                engine_live: Vec::new(),
                suspected_loads: Vec::new(),
                series: Vec::new(),
                requests: Vec::new(),
                jobs: Vec::new(),
                claims: Vec::new(),
                gpu: None,
                displaced: Vec::new(),
            },
            heard: false,
            published: 0,
            write_down: false,
            draw_down: false,
            replay: OutReplay::default(),
            down_since: None,
            published_at: None,
            chart: TokenChart::new(chart_bucket_s),
            llama_at: None,
            ctx_history: CtxBook::new(ctx_history_h),
            stopping_since: HashMap::new(),
            starting_since: HashMap::new(),
            flights: crate::inflight::Tracker::default(),
            ctx_sizes: BTreeMap::new(),
            gen_last: None,
            motion: crate::tty::scroll::Motion::default(),
        }
    }
}

struct FrameCtx {
    output_cap: usize,
    input_cap: usize,
    gen_ceiling: f64,
    prompt_ceiling: f64,
    chart_bucket_s: u64,
    chart_glyphs: ChartGlyphs,
    bar_segments: bool,
    ctx_history_h: u32,
    started: Instant,
    host: String,
    cpu_cores: Option<u32>,
    mem_total_bytes: Option<u64>,
    /// `[llama] enabled`. False draws [`WatchState::NoLlama`] from the first frame.
    llama_enabled: bool,
    /// `tty.show_text`. False hands the layout no llama text at all.
    show_text: bool,
    /// `[setup]` rules, for the SETUP rows (#52).
    setup: Rules,
    /// `[temps]` renames and thresholds, for the TEMPS panel (#74).
    temps_look: crate::sources::temps::TempLook,
    /// `[jobs] stale_s`.
    jobs_stale_s: u64,
    /// `[tty] disk_ceiling_bps`.
    disk_ceiling_bps: f64,
    /// `[tty] net_ceiling_bps`. `None` uses the sampled link speed.
    net_ceiling_bps: Option<f64>,
}

fn tick_once<Feed, Samp, Pub, Rend, Clk, Ntf, Stp, Lg>(
    input: &mut LoopInput<'_, Feed, Samp, Pub, Rend, Clk, Ntf, Stp, Lg>,
    state: &mut TickState,
    mono: Instant,
    wall: SystemTime,
    ctx: &FrameCtx,
) where
    Feed: LlamaFeed,
    Samp: SampleStep,
    Pub: PublishStep,
    Rend: RenderStep,
    Clk: Clock,
    Ntf: Notifier,
    Stp: Stop,
    Lg: Sink,
{
    if let Some((view, detail)) = take_latest(&mut input.feed) {
        state.llama = view;
        state.detail = detail;
        state.heard = true;
        state.llama_at = Some(mono);
        note_poll(state, mono, wall);
    }
    state.flights.retire(mono);
    let sample = input.sampler.sample(mono, wall, &state.llama);
    let extras = publish_extras(&sample, state, ctx);
    note_publish(input, state, &sample, &extras, mono);
    match sample.snapshot.ai {
        AiState::Down | AiState::NoData => {
            if state.down_since.is_none() {
                state.down_since = Some(wall);
            }
        }
        AiState::Idle | AiState::Loaded => state.down_since = None,
    }
    let decoded = match (state.llama_at, state.llama.decoded_total) {
        (Some(at), Some(count)) => Some((at, count)),
        _ => None,
    };
    state.chart.ingest(mono, decoded, state.detail.prompt_tps);
    let running: Vec<String> = state
        .llama
        .models
        .iter()
        .map(|model| model.name.clone())
        .collect();
    let mut held = state.detail.slots.clone();
    held.extend(strata_views(&state.detail.engine_live));
    held.extend(tqllm_views(&state.detail.engine_live));
    state.ctx_history.record(mono, &running, &held);
    let zone = input.clock.zone(mono);
    let model = tty_model(&sample, state, mono, wall, zone, ctx);
    note_draw(input, state, &model, mono);
}

fn note_publish<Feed, Samp, Pub, Rend, Clk, Ntf, Stp, Lg>(
    input: &mut LoopInput<'_, Feed, Samp, Pub, Rend, Clk, Ntf, Stp, Lg>,
    state: &mut TickState,
    sample: &WatchSample,
    extras: &Extras,
    mono: Instant,
) where
    Feed: LlamaFeed,
    Samp: SampleStep,
    Pub: PublishStep,
    Rend: RenderStep,
    Clk: Clock,
    Ntf: Notifier,
    Stp: Stop,
    Lg: Sink,
{
    match input
        .publisher
        .publish(&sample.snapshot, &state.llama, extras)
    {
        Ok(()) => {
            if state.write_down {
                log::emit(&mut input.log, Priority::Info, "snapshot write recovered");
                state.write_down = false;
            }
            state.published = state.published.saturating_add(1);
            state.published_at = Some(mono);
        }
        Err(PublishError::Write(err)) => {
            if !state.write_down {
                log::emit(
                    &mut input.log,
                    Priority::Err,
                    &format!("snapshot write failed: {err}"),
                );
                state.write_down = true;
            }
        }
        Err(_) => {}
    }
}

fn note_draw<Feed, Samp, Pub, Rend, Clk, Ntf, Stp, Lg>(
    input: &mut LoopInput<'_, Feed, Samp, Pub, Rend, Clk, Ntf, Stp, Lg>,
    state: &mut TickState,
    model: &TtyModel,
    mono: Instant,
) where
    Feed: LlamaFeed,
    Samp: SampleStep,
    Pub: PublishStep,
    Rend: RenderStep,
    Clk: Clock,
    Ntf: Notifier,
    Stp: Stop,
    Lg: Sink,
{
    match input.render.draw(model, mono) {
        Ok(()) => {
            if state.draw_down {
                log::emit(&mut input.log, Priority::Info, "console draw recovered");
                state.draw_down = false;
            }
        }
        Err(err) => {
            if !state.draw_down {
                log::emit(
                    &mut input.log,
                    Priority::Err,
                    &format!("console draw failed: {err}"),
                );
                state.draw_down = true;
            }
        }
    }
}

fn take_latest(feed: &mut impl LlamaFeed) -> Option<(LlamaView, LlamaDetail)> {
    let mut latest = None;
    while let Some(item) = feed.poll() {
        latest = Some(item);
    }
    latest
}

#[derive(Default)]
struct OutReplay {
    tail: Vec<char>,
    shown: usize,
    frame: Option<u32>,
}

/// Cap `next` to its last `cap` sanitised chars and adjust `shown`.
///
/// `prev` is the tail already handed to the layout. `shown` counts sanitised
/// chars, including line ends, from the start of that tail. `shown` keeps
/// the larger of the head-trim rewind and the shared prefix, so a one-character
/// accidental overlap cannot collapse the panel. A tail that shares no prefix
/// resets `shown` to 0.
#[must_use]
pub fn advance_output_tail(
    prev: &[char],
    next: &[char],
    shown: usize,
    cap: usize,
) -> (Vec<char>, usize) {
    let next_tail = cap_chars(next, cap);
    let shown = align_shown(prev, &next_tail, shown);
    (next_tail, shown)
}

fn align_shown(prev: &[char], next: &[char], shown: usize) -> usize {
    let shown = shown.min(prev.len());
    let prefix = common_prefix_len(prev, next);
    let aligned = aligned_drop(prev, next);
    // A new request shares neither a continuation nor a prefix.
    if aligned.is_none() && prefix == 0 {
        return 0;
    }
    // With no continuation only the shared prefix stays shown. A
    // 1-character suffix match must not beat a longer shared prefix.
    let by_drop = aligned.map_or(0, |dropped| shown.saturating_sub(dropped));
    let by_prefix = shown.min(prefix);
    by_drop.max(by_prefix).min(next.len())
}

fn common_prefix_len(prev: &[char], next: &[char]) -> usize {
    prev.iter()
        .zip(next)
        .take_while(|(left, right)| left == right)
        .count()
}

fn note_output(replay: &mut OutReplay, incoming: &[char], cap: usize) {
    let next = cap_chars(incoming, cap);
    if next == replay.tail {
        advance_frame(replay);
        return;
    }
    let already = revealed(replay.shown, replay.tail.len(), replay.frame);
    replay.shown = align_shown(&replay.tail, &next, already);
    replay.tail = next;
    replay.frame = Some(0);
}

fn advance_frame(replay: &mut OutReplay) {
    let Some(frame) = replay.frame else {
        return;
    };
    let next = frame.saturating_add(1);
    if next >= 10 {
        replay.shown = replay.tail.len();
        replay.frame = None;
    } else {
        replay.frame = Some(next);
    }
}

fn revealed(shown: usize, total: usize, frame: Option<u32>) -> usize {
    match frame {
        None => total,
        Some(frame) => {
            let before = shown.min(total);
            let delta = total - before;
            before + layout::replay_shown(delta, frame.min(10)).min(delta)
        }
    }
}

fn cap_chars(chars: &[char], cap: usize) -> Vec<char> {
    if cap == 0 {
        return Vec::new();
    }
    if chars.len() <= cap {
        chars.to_vec()
    } else {
        chars[chars.len() - cap..].to_vec()
    }
}

/// Smallest drop of `prev`'s head that leaves a prefix of `next`, if any.
fn aligned_drop(prev: &[char], next: &[char]) -> Option<usize> {
    if prev.is_empty() {
        return Some(0);
    }
    if next.is_empty() {
        return None;
    }
    let lps = prefix_table(next);
    let mut matched = 0usize;
    for (i, ch) in prev.iter().enumerate() {
        while matched > 0 && next.get(matched) != Some(ch) {
            matched = lps[matched - 1];
        }
        if next.get(matched) == Some(ch) {
            matched += 1;
        }
        if matched == next.len() && i + 1 != prev.len() {
            matched = lps[matched - 1];
        }
        if i + 1 == prev.len() {
            if matched == 0 {
                return None;
            }
            return Some(prev.len() - matched);
        }
    }
    None
}

fn prefix_table(pat: &[char]) -> Vec<usize> {
    let mut table = vec![0; pat.len()];
    let mut len = 0usize;
    let mut i = 1usize;
    while i < pat.len() {
        if pat[i] == pat[len] {
            len += 1;
            table[i] = len;
            i += 1;
        } else if len > 0 {
            len = table[len - 1];
        } else {
            table[i] = 0;
            i += 1;
        }
    }
    table
}

/// Process entry. Refuses root, opens the snapshot directory, then loops.
///
/// The snapshot directory is opened before the poller starts. A failure
/// there exits 1 and does not dial llama-swap. `args.no_text` turns
/// `tty.show_text` off before the poller starts, so no text is ever kept.
pub fn run(args: &RunArgs) -> i32 {
    let path = args.config.as_path();
    if root_exit(rustix::process::geteuid().is_root()).is_some() {
        eprintln!("refusing to run as root");
        return 2;
    }
    let roots = Roots::default();
    let mut prepared = match prepare(
        path,
        host_nproc(&roots.proc),
        Path::new(llama_core::wire::SNAPSHOT_DIR),
    ) {
        Ok(prepared) => prepared,
        Err(PrepareError::Config(err)) => {
            eprintln!("{err}");
            return 2;
        }
        Err(PrepareError::Snapshot(err)) => {
            eprintln!("{err}");
            return 1;
        }
    };
    if args.no_text {
        prepared.config = prepared.config.with_text_off();
    }
    let now = Instant::now();
    // #42: keys on tty11 must not pause or draw on the dashboard. Only acts
    // when stdout is tty11; a failure is logged and the watcher carries on.
    let modes = match term::quiet_stdout() {
        Ok(modes) => modes,
        Err(err) => {
            log::emit(
                &mut log::Stderr,
                Priority::Warning,
                &format!("console: line settings unchanged: {err}"),
            );
            None
        }
    };
    let term = match Term::for_stdout(Duration::from_secs(prepared.config.tty.full_redraw_s), now) {
        Ok(term) => {
            let term = term
                .with_console_blank(ConsoleBlank::from_minutes(
                    prepared.config.tty.blank_min,
                    prepared.config.tty.sleep_min,
                ))
                .with_palette(prepared.config.tty.palette)
                .with_saved_modes(modes);
            with_grid_file(term, &prepared.config)
        }
        Err(err) => {
            eprintln!("console: {err}");
            return 1;
        }
    };
    // #42: the console is written only by the tty writer thread, so a
    // blocked write (unblank, VT hold) never stops publish or the watchdog.
    let render = match FrameWriter::spawn(term, log::Stderr) {
        Ok(render) => render,
        Err(err) => {
            eprintln!("console: tty writer: {err}");
            return 1;
        }
    };
    let (poller, feed) = if prepared.config.llama.enabled || prepared.config.jobs.dir.is_some() {
        if !prepared.config.llama.enabled {
            log::emit(
                &mut log::Stderr,
                Priority::Info,
                "llama-swap polling is off ([llama] enabled = false)",
            );
        }
        match poller::spawn(&prepared.config, log::Stderr) {
            Ok((poller, rx)) => (Some(poller), SlotFeed::new(rx)),
            Err(err) => {
                eprintln!("poller: {err}");
                return 1;
            }
        }
    } else {
        log::emit(
            &mut log::Stderr,
            Priority::Info,
            "llama-swap polling is off ([llama] enabled = false)",
        );
        (None, SlotFeed::disabled())
    };
    let collector = WatchCollector::new(
        roots.clone(),
        NvidiaGpu::new(),
        &prepared.config,
        log::Stderr,
    );
    let exit = run_loop(LoopInput {
        config: &prepared.config,
        feed,
        sampler: collector,
        publisher: prepared.publisher,
        render,
        clock: RealClock::default(),
        notify: SdNotify,
        stop: NeverStop,
        log: log::Stderr,
        roots: &roots,
        started: now,
    });
    drop(poller);
    exit.code()
}

/// `[grid] enabled` (#107): start the grid writer thread and tap the
/// terminal's frames into it. A failure to start is logged and the dashboard
/// runs without the file; it never stops llama-watch.
fn with_grid_file<W: Write>(term: Term<W>, config: &Config) -> Term<W> {
    if !config.grid.enabled {
        return term;
    }
    let writer = GridWriter::new(&config.grid.path, config.tty.palette);
    match GridTap::from_config(writer, log::Stderr, &config.grid) {
        Ok(tap) => term.with_grid_tap(tap),
        Err(err) => {
            log::emit(
                &mut log::Stderr,
                Priority::Err,
                &format!("grid: writer thread not started: {err}"),
            );
            term
        }
    }
}

fn tty_model(
    sample: &WatchSample,
    tick: &mut TickState,
    mono: Instant,
    wall: SystemTime,
    zone: &Zone,
    ctx: &FrameCtx,
) -> TtyModel {
    note_stopping(&mut tick.stopping_since, &sample.snapshot.models, mono);
    note_starting(&mut tick.starting_since, &sample.snapshot.models, mono);
    if let Some(rate) = tick.detail.gen_tps.filter(|v| v.is_finite() && *v > 0.0) {
        tick.gen_last = Some(rate);
    }
    let watch = watch_state(sample, &tick.detail, tick.heard, ctx.llama_enabled);
    let (model_loading, model_stopping) = header_swap(sample, watch);
    let no_slots = first_without_slots(sample, watch);
    // The header model has no `/slots`, or `/slots` without text (#66): IN
    // and OUT from its last capture (#5).
    let header_up = matches!(watch, WatchState::Generating | WatchState::Ready);
    let capture = tick.detail.capture.as_ref().filter(|capture| {
        ((no_slots && tick.detail.slots.is_empty()) || (header_up && capture.over_slots))
            && sample
                .snapshot
                .models
                .first()
                .is_some_and(|model| model.name == capture.model)
    });
    // With text off the poller keeps none, and nothing here reads the slot
    // tails either, so no llama text can reach the frame (RR-LV1).
    let mut text = if !ctx.show_text {
        FrameText::default()
    } else if let Some(capture) = capture {
        capture_text(capture, &mut tick.replay, ctx.input_cap, ctx.output_cap)
    } else {
        frame_text(
            &tick.detail.slots,
            &mut tick.replay,
            ctx.input_cap,
            ctx.output_cap,
        )
    };
    if ctx.show_text {
        apply_scroll(
            &mut text,
            &mut tick.motion,
            &tick.detail,
            capture,
            &sample.snapshot.models,
            mono,
        );
    }
    let mem_total = ctx.mem_total_bytes.map(gib);
    let mem_used = mem_used_bytes(sample.snapshot.mem_pct, ctx.mem_total_bytes).map(gib);
    let job_and_note = job_and_note(
        &tick.detail.jobs,
        serving_models(sample),
        ctx.jobs_stale_s,
        wall,
    );
    let outside = outside_engine(
        &sample.snapshot.models,
        job_and_note.0.is_some(),
        &sample.gpu_processes,
        ctx.llama_enabled,
        tick.heard,
        sample.snapshot.ai,
    );
    let gpu_busy = outside.is_some();
    TtyModel {
        state: watch,
        host: ctx.host.clone(),
        model_name: if gpu_busy {
            OUTSIDE_MODEL.to_owned()
        } else {
            model_name(sample, watch)
        },
        model_detail: outside.unwrap_or_else(|| model_detail(sample, watch)),
        model_stuck: model_stuck(sample, watch, &tick.stopping_since, mono),
        model_loading,
        model_stopping,
        gpu_busy,
        slots_line: if gpu_busy || ((model_loading || model_stopping) && header_up) {
            // No slots to count while the header model is still swapping,
            // or while the card is busy outside llama-swap.
            "--".to_owned()
        } else if no_slots && tick.detail.slots.is_empty() {
            "--".to_owned()
        } else {
            slots_line(&tick.detail, watch)
        },
        swap_line: if gpu_busy {
            "--".to_owned()
        } else {
            swap_line(
                watch,
                &sample.snapshot.models,
                &tick.starting_since,
                &tick.stopping_since,
                mono,
            )
        },
        cool_c: temp_i(sample.snapshot.coolant_c),
        cpu_c: temp_i(sample.snapshot.cpu_c),
        gpu_c: temp_i(sample.snapshot.gpu_c),
        clock: format_wall(wall, zone),
        cpu_pct: pct(sample.snapshot.cpu_pct),
        cpu_cores: ctx.cpu_cores,
        gpu_pct: pct(sample.snapshot.gpu_pct),
        vram_used_gb: sample.gpu.vram_used.map(gib),
        vram_total_gb: sample.gpu.vram_total.map(gib),
        mem_used_gb: mem_used,
        mem_total_gb: mem_total,
        power_w: sample.gpu.power_mw.map(|mw| f64::from(mw) / 1000.0),
        power_limit_w: sample.gpu.power_limit_mw.map(|mw| f64::from(mw) / 1000.0),
        load_pct: pct(sample.snapshot.load),
        activity_pct: pct(sample.snapshot.activity),
        activity_src: sample
            .snapshot
            .activity
            .is_some()
            .then_some(sample.load_source),
        activity_w: sample.activity_w,
        gen_tps: tick.detail.gen_tps,
        prompt_tps: tick.detail.prompt_tps,
        prompt_last: prompt_last(&tick.detail.slots),
        gen_last: tick.gen_last,
        gen_ceiling: ctx.gen_ceiling,
        prompt_ceiling: ctx.prompt_ceiling,
        slots: {
            let mut slots = layout_slots(&tick.detail.slots, &tick.ctx_history);
            slots.extend(strata_slot_rows(
                &tick.detail.engine_live,
                &tick.llama.models,
                &tick.ctx_history,
            ));
            slots.extend(tqllm_slot_rows(&tick.detail.engine_live, &tick.ctx_history));
            slots
        },
        backend_lines: engine_lines(sample, &tick.detail.engine_live, watch),
        text_note: if ctx.show_text && no_slots && tick.detail.slots.is_empty() && capture.is_none()
        {
            NO_SLOTS_TEXT.to_owned()
        } else {
            String::new()
        },
        requests: layout_requests(
            &tick.detail.activity,
            watch,
            zone,
            &RecentCtx {
                flights: &tick.flights.flights(),
                ctx_sizes: &tick.ctx_sizes,
                ids: &model_ids(&tick.llama, &tick.detail),
            },
        ),
        in_title: text.in_title,
        out_title: text.out_title,
        in_lines: text.in_lines,
        out_lines: text.out_lines,
        out_shown: text.out_shown,
        replay_frame: text.replay_frame,
        in_scroll: text.in_scroll,
        out_scroll: text.out_scroll,
        show_text: ctx.show_text,
        down_since: tick
            .down_since
            .map(|since| down_label(since, wall, zone))
            .unwrap_or_default(),
        health: health(sample, &tick.detail, watch),
        snapshot: (tick.published > 0).then_some(tick.published),
        snapshot_age: match tick.published_at {
            Some(at) => format_age(mono.saturating_duration_since(at)),
            None => "0.0s".to_owned(),
        },
        errors: u64::try_from(sample.snapshot.errors.len()).unwrap_or(u64::MAX),
        uptime: format_span(mono.saturating_duration_since(ctx.started)),
        chart: tick.chart.buckets(),
        chart_bucket_s: ctx.chart_bucket_s,
        chart_glyphs: ctx.chart_glyphs,
        bar_segments: ctx.bar_segments,
        fans: sample.fans.clone(),
        temps: sample
            .temps
            .as_ref()
            .map(|readings| crate::sources::temps::panel(readings, &ctx.temps_look)),
        ctx_history_h: ctx.ctx_history_h,
        setup: setup_view(sample, &tick.detail, watch, &ctx.setup),
        job: job_and_note.0,
        job_note: job_and_note.1,
        model_id: header_model_id(sample, watch, &tick.detail),
        claims: tick
            .detail
            .claims
            .iter()
            .map(|claim| HeaderClaim {
                model: claim.model.clone(),
                holder: claim.holder.clone(),
                in_use: claim.in_use,
                expires_at: claim.expires_at,
                hold_until: claim.hold_until,
                hold_tz_offset_s: claim
                    .hold_until
                    .map(|until| zone.offset_at(unix_secs_of(until)))
                    .unwrap_or(0),
                hold_reason: claim.hold_reason.clone(),
                conditional: claim.conditional,
            })
            .collect(),
        gpu: tick.detail.gpu.as_ref().map(|gpu| HeaderGpu {
            holder: gpu.holder.clone(),
            external: gpu.external,
            draining: gpu.draining,
            in_use: gpu.in_use,
            expires_at: gpu.expires_at,
            hold_until: gpu.hold_until,
            hold_tz_offset_s: gpu
                .hold_until
                .map(|until| zone.offset_at(unix_secs_of(until)))
                .unwrap_or(0),
            hold_reason: gpu.hold_reason.clone(),
        }),
        displaced: tick
            .detail
            .displaced
            .iter()
            .map(|note| HeaderDisplaced {
                model: note.model.clone(),
                to: note.to.clone(),
                by: note.by.clone(),
                at: note.at,
                tz_offset_s: zone.offset_at(unix_secs_of(note.at)),
            })
            .collect(),
        now: wall,
        disk_read_bps: sample.host_io.disk_read_bps,
        disk_write_bps: sample.host_io.disk_write_bps,
        net_rx_bps: sample.host_io.net_rx_bps,
        net_tx_bps: sample.host_io.net_tx_bps,
        net_ceiling_bps: ctx.net_ceiling_bps.or(sample.host_io.net_ceiling_bps),
        swap_used_gb: sample.host_io.swap_used_bytes.map(gib),
        swap_total_gb: sample.host_io.swap_total_bytes.map(gib),
        io_psi: sample.host_io.io_psi.map(f64::from),
        mem_psi: sample.host_io.mem_psi.map(f64::from),
        disk_ceiling_bps: ctx.disk_ceiling_bps,
        tz_offset_s: zone.offset_at(unix_secs_of(wall)),
    }
}

/// llama-swap id of the model SETUP names, at the claims cap
/// ([`crate::sources::events::claim_model`]). Same pairing of
/// `detail.setup` and `sample.snapshot.models` as [`setup_view`]. Empty
/// when that view is collapsed, or when the pair is stale (no `swap_id`).
fn header_model_id(sample: &WatchSample, state: WatchState, detail: &LlamaDetail) -> String {
    paired_setup(sample, detail, state)
        .map(|(setup, _)| setup.swap_id)
        .unwrap_or_default()
}

fn unix_secs_of(time: SystemTime) -> i64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(d) => i64::try_from(d.as_secs()).unwrap_or(i64::MAX),
        Err(_) => 0,
    }
}

fn serving_models(sample: &WatchSample) -> bool {
    !sample.snapshot.models.is_empty()
}

fn job_and_note(
    jobs: &[crate::jobs::Job],
    serving: bool,
    stale_s: u64,
    now: SystemTime,
) -> (Option<layout::JobView>, Option<String>) {
    match crate::jobs::pick_job(jobs, serving, stale_s, now) {
        crate::jobs::JobPick::Live {
            job,
            more,
            serving: true,
        } => (None, Some(layout::job_note_line(job, more, true))),
        crate::jobs::JobPick::Live {
            job,
            more,
            serving: false,
        } => (
            Some(layout::JobView {
                job: job.clone(),
                more,
            }),
            None,
        ),
        crate::jobs::JobPick::Finished { job } => {
            (None, Some(layout::job_note_line(job, 0, false)))
        }
        crate::jobs::JobPick::None => (None, None),
    }
}

/// The SETUP block (#52): the model generating now, else the one RECENT
/// saw last, else the first loaded. `None` with nothing loaded.
fn setup_view(
    sample: &WatchSample,
    detail: &LlamaDetail,
    state: WatchState,
    rules: &Rules,
) -> Option<SetupView> {
    let (setup, model) = paired_setup(sample, detail, state)?;
    let live = LiveCtx {
        backend: model.backend.map(|info| info.kind).unwrap_or_default(),
        detail: model.detail.as_ref(),
        info: model.backend.as_ref(),
        engine: Some(&setup.engine),
    };
    let rows = rules.rows(&setup.found, &live);
    Some(SetupView {
        id: setup.id,
        name: setup.name,
        more: sample.snapshot.models.len() - 1,
        rows,
    })
}

/// The SETUP entry that [`setup_view`] would show, and the snapshot model
/// it is paired with. A length mismatch falls back to the snapshot, with
/// an empty `swap_id`.
fn paired_setup<'a>(
    sample: &'a WatchSample,
    detail: &'a LlamaDetail,
    state: WatchState,
) -> Option<(ModelSetup, &'a llama_core::sample::ModelInfo)> {
    if !matches!(state, WatchState::Generating | WatchState::Ready) {
        return None;
    }
    let models = &sample.snapshot.models;
    if models.is_empty() {
        return None;
    }
    // The poller sends one entry per model, in the same order; anything
    // else is a stale pair and shows only what the snapshot has.
    let setups: &[ModelSetup] = if detail.setup.len() == models.len() {
        &detail.setup
    } else {
        &[]
    };
    let busy = models.iter().position(|model| {
        model
            .backend
            .is_some_and(|info| info.running.is_some_and(|n| n > 0))
            || detail
                .slots
                .iter()
                .any(|slot| slot.is_processing && slot.model == model.name)
    });
    let recent = detail
        .activity
        .first()
        .and_then(|row| setups.iter().position(|setup| setup.key == row.model));
    let at = busy.or(recent).unwrap_or(0);
    let model = &models[at];
    let setup = setups.get(at).cloned().unwrap_or_else(|| ModelSetup {
        id: model.name.clone(),
        name: model.full_name.clone().unwrap_or_default(),
        ..ModelSetup::default()
    });
    Some((setup, model))
}

fn text_chars(lines: &[String]) -> Vec<char> {
    lines.iter().flat_map(|line| line.chars()).collect()
}

/// Live tok/s for scroll (#188). A burst engine's held prefill speed and its
/// wall-clock generation spike are not live. The llama.cpp headline is
/// a completed-turn average, not a live slot measurement. tqLLM's own
/// request rate is live when it is the only engine on screen.
fn live_token_rates(detail: &LlamaDetail, models: &[ModelInfo]) -> (Option<f64>, Option<f64>) {
    let kinds: Vec<Backend> = models
        .iter()
        .filter_map(|model| model.backend.map(|info| info.kind))
        .collect();
    let burst_only = !kinds.is_empty()
        && kinds
            .iter()
            .all(|kind| matches!(kind, Backend::Vllm | Backend::SgLang));
    let prompt_live = if !kinds.is_empty() && kinds.iter().all(|kind| *kind == Backend::Tqllm) {
        detail
            .prompt_tps
            .filter(|rate| rate.is_finite() && *rate > 0.0)
    } else {
        None
    };
    let gen_live = if burst_only {
        None
    } else if kinds
        .iter()
        .any(|kind| matches!(kind, Backend::LlamaCpp | Backend::Strata | Backend::Tqllm))
        || (kinds.is_empty() && detail.slots.iter().any(|slot| slot.is_processing))
    {
        detail
            .gen_tps
            .filter(|rate| rate.is_finite() && *rate > 0.0)
    } else {
        None
    };
    (prompt_live, gen_live)
}

fn scroll_identity(
    detail: &LlamaDetail,
    capture: Option<&CaptureView>,
) -> (String, u64, Option<u64>, Option<u64>) {
    if let Some(capture) = capture {
        let row = detail.activity.iter().find(|row| row.id == capture.id);
        let model = row
            .map(|row| row.model.clone())
            .unwrap_or_else(|| capture.model.clone());
        let request = u64::try_from(capture.id.max(0)).unwrap_or(0);
        return (
            model,
            request,
            row.and_then(|row| row.input_tokens),
            row.and_then(|row| row.output_tokens),
        );
    }
    let Some(slot) = pick_slot(&detail.slots) else {
        let model = detail
            .setup
            .first()
            .map(|setup| setup.key.clone())
            .unwrap_or_default();
        return (model, 0, None, None);
    };
    let model = detail
        .setup
        .iter()
        .find(|setup| setup.name == slot.model || setup.id == slot.model || setup.key == slot.model)
        .map(|setup| setup.key.clone())
        .or_else(|| (detail.setup.len() == 1).then(|| detail.setup[0].key.clone()))
        .unwrap_or_else(|| slot.model.clone());
    let request = if slot.id_task < 0 {
        0
    } else {
        (slot.id as u64)
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add(slot.id_task as u64)
            .max(1)
    };
    (
        model,
        request,
        (slot.n_prompt_tokens > 0).then_some(slot.n_prompt_tokens),
        (slot.n_decoded > 0).then_some(slot.n_decoded),
    )
}

fn apply_scroll(
    text: &mut FrameText,
    motion: &mut crate::tty::scroll::Motion,
    detail: &LlamaDetail,
    capture: Option<&CaptureView>,
    models: &[ModelInfo],
    now: Instant,
) {
    let (model, request, input_tokens, output_tokens) = scroll_identity(detail, capture);
    let input = text_chars(&text.in_lines);
    let output = text_chars(&text.out_lines);
    let rows: Vec<crate::tty::scroll::Turn> = detail
        .activity
        .iter()
        .map(|row| crate::tty::scroll::Turn {
            seq: row.seq,
            model: row.model.clone(),
            prompt_tps: row.prompt_tps.or(row.engine_prompt_tps),
            gen_tps: row.gen_tps.or(row.engine_gen_tps),
        })
        .collect();
    let (live_prompt, live_gen) = live_token_rates(detail, models);
    let view = motion.update(
        &crate::tty::scroll::Sample {
            model: &model,
            request,
            input: &input,
            output: &output,
            input_tokens,
            output_tokens,
            live_prompt,
            live_gen,
            rows: &rows,
        },
        now,
    );
    text.in_scroll = view.inn;
    text.out_scroll = view.out;
}

/// IN and OUT for one frame, both from [`pick_slot`]. Advances the OUT
/// replay cursor. IN keeps no cursor: it is rebuilt from the picked slot on
/// every frame, so a slot switch swaps it at once. OUT's cursor goes through
/// [`align_shown`], which drops to 0 when the new slot's tail shares nothing
/// with the old one.
fn frame_text(
    slots: &[SlotView],
    replay: &mut OutReplay,
    input_cap: usize,
    output_cap: usize,
) -> FrameText {
    let picked = pick_slot(slots);
    let output = picked.map_or_else(Vec::new, |slot| sanitised_chars(&slot.output));
    note_output(replay, &output, output_cap);
    let input = picked.map_or_else(Vec::new, |slot| sanitised_chars(&slot.input));
    let input = cap_chars(&input, input_cap);
    FrameText {
        in_title: "IN".to_owned(),
        out_title: "OUT".to_owned(),
        in_lines: chars_lines(&input),
        out_lines: chars_lines(&replay.tail),
        out_shown: replay.shown,
        replay_frame: replay.frame,
        in_scroll: None,
        out_scroll: None,
    }
}

/// IN and OUT from a llama-swap capture (#5): the last finished exchange,
/// and the titles say so. OUT replays like a slot's does when a new
/// capture arrives.
fn capture_text(
    capture: &CaptureView,
    replay: &mut OutReplay,
    input_cap: usize,
    output_cap: usize,
) -> FrameText {
    let output = sanitised_chars(&capture.output);
    note_output(replay, &output, output_cap);
    let input = cap_chars(&sanitised_chars(&capture.input), input_cap);
    FrameText {
        in_title: capture_in_title(&capture.input_note),
        out_title: CAPTURE_OUT_TITLE.to_owned(),
        in_lines: chars_lines(&input),
        out_lines: chars_lines(&replay.tail),
        out_shown: replay.shown,
        replay_frame: replay.frame,
        in_scroll: None,
        out_scroll: None,
    }
}

/// IN and OUT titles over a capture (#5).
const CAPTURE_IN_TITLE: &str = "IN (last request)";

/// `IN (last request)`, or `IN (last request · 3 tool results)` for a tool
/// loop (#38).
fn capture_in_title(note: &str) -> String {
    if note.is_empty() {
        CAPTURE_IN_TITLE.to_owned()
    } else {
        format!("IN (last request \u{00B7} {note})")
    }
}
const CAPTURE_OUT_TITLE: &str = "OUT (last response)";

/// The IN/OUT part of a [`TtyModel`]. Empty when `tty.show_text = false`.
#[derive(Default)]
struct FrameText {
    in_title: String,
    out_title: String,
    in_lines: Vec<String>,
    out_lines: Vec<String>,
    out_shown: usize,
    replay_frame: Option<u32>,
    in_scroll: Option<usize>,
    out_scroll: Option<usize>,
}

fn watch_state(
    sample: &WatchSample,
    detail: &LlamaDetail,
    heard: bool,
    llama_enabled: bool,
) -> WatchState {
    if !llama_enabled {
        return WatchState::NoLlama;
    }
    if !heard {
        return WatchState::Starting;
    }
    match sample.snapshot.ai {
        AiState::Down | AiState::NoData => WatchState::AiDown,
        AiState::Idle => WatchState::Ready,
        AiState::Loaded => {
            let backend_busy = sample.snapshot.models.iter().any(|model| {
                model
                    .backend
                    .is_some_and(|info| info.running.is_some_and(|n| n > 0))
            });
            if backend_busy || detail.slots.iter().any(|slot| slot.is_processing) {
                WatchState::Generating
            } else {
                WatchState::Ready
            }
        }
    }
}

fn model_name(sample: &WatchSample, state: WatchState) -> String {
    match state {
        WatchState::Starting => "...".to_owned(),
        WatchState::AiDown | WatchState::NoLlama => "--".to_owned(),
        WatchState::Generating | WatchState::Ready => sample
            .snapshot
            .models
            .first()
            .map(|model| {
                model
                    .full_name
                    .clone()
                    .unwrap_or_else(|| model.name.clone())
            })
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "--".to_owned()),
    }
}

/// Header detail of the first model: its engine, always (#33), llama.cpp
/// when the snapshot names none. Its settings (ctx, KV, block, prefix…)
/// moved to the SETUP block (#52), so the header no longer runs into the
/// clock. With no model shown it is [`NO_ENGINE`].
fn model_detail(sample: &WatchSample, state: WatchState) -> String {
    let model = match state {
        WatchState::Generating | WatchState::Ready => sample.snapshot.models.first(),
        WatchState::Starting | WatchState::AiDown | WatchState::NoLlama => None,
    };
    let Some(model) = model else {
        return NO_ENGINE.to_owned();
    };
    let engine = model.backend.map(|info| info.kind).unwrap_or_default();
    engine.display_name().to_owned()
}

/// The header's engine item with no model shown (#33).
const NO_ENGINE: &str = "engine --";
/// A compute process at least this large makes the header `GPU BUSY` (#139).
const GPU_BUSY_MIN_BYTES: u64 = 256 * 1024 * 1024;
/// Model field while that header is up.
const OUTSIDE_MODEL: &str = "outside llama-swap";
/// IN and OUT while the header model has no `/slots` to take text from.
const NO_SLOTS_TEXT: &str = "text needs llama.cpp /slots";
/// How long a model may sit in `stopping` before the header calls it stuck.
const STUCK_STOPPING: Duration = Duration::from_secs(60);

/// True when the header (first) model is served by a backend without `/slots`.
fn first_without_slots(sample: &WatchSample, state: WatchState) -> bool {
    matches!(state, WatchState::Generating | WatchState::Ready)
        && sample
            .snapshot
            .models
            .first()
            .and_then(|model| model.backend)
            .is_some_and(|info| !info.kind.has_slots())
}

/// Remember `now` the first time a model is seen in `phase`, and forget a
/// model that has left it. The clock is the caller's.
fn note_phase(
    since: &mut HashMap<String, Instant>,
    models: &[ModelInfo],
    phase: &str,
    now: Instant,
) {
    since.retain(|name, _| {
        models
            .iter()
            .any(|model| model.name == *name && model.state == phase)
    });
    for model in models.iter().filter(|model| model.state == phase) {
        since.entry(model.name.clone()).or_insert(now);
    }
}

/// Track when each model entered `stopping`; forget the rest.
fn note_stopping(since: &mut HashMap<String, Instant>, models: &[ModelInfo], now: Instant) {
    note_phase(since, models, "stopping", now);
}

/// Track when each model entered `starting`; forget the rest.
fn note_starting(since: &mut HashMap<String, Instant>, models: &[ModelInfo], now: Instant) {
    note_phase(since, models, "starting", now);
}

/// Engine field when the card is busy outside llama-swap, or `None` when
/// the header stays as it is.
///
/// Llama-swap must have answered with an empty list: it is enabled, this
/// tick has heard it, and `ai` is [`AiState::Idle`] (the same `Idle` that
/// [`watch_state`] turns into `Ready`). Not-yet-heard, down, no-data and
/// disabled keep today's header. Also `None` when a job view is up, when
/// any model is listed, when the process read failed, or when no process
/// holds [`GPU_BUSY_MIN_BYTES`]. The clock is not an argument: the collector
/// already rate-limits the list this receives. A failure suppresses the
/// header even if `list` is stale.
fn outside_engine(
    models: &[ModelInfo],
    job_up: bool,
    processes: &GpuProcesses,
    llama_enabled: bool,
    heard: bool,
    ai: AiState,
) -> Option<String> {
    if job_up || !models.is_empty() || processes.failed {
        return None;
    }
    if !llama_enabled || !heard || ai != AiState::Idle {
        return None;
    }
    busiest(&processes.list).map(process_field)
}

/// Largest process at or above [`GPU_BUSY_MIN_BYTES`]. Equal memory keeps
/// the lower pid.
fn busiest(list: &[GpuProcess]) -> Option<&GpuProcess> {
    let mut best: Option<&GpuProcess> = None;
    for process in list {
        if process.memory_bytes < GPU_BUSY_MIN_BYTES {
            continue;
        }
        let replace = match best {
            None => true,
            Some(kept) => {
                process.memory_bytes > kept.memory_bytes
                    || (process.memory_bytes == kept.memory_bytes && process.pid < kept.pid)
            }
        };
        if replace {
            best = Some(process);
        }
    }
    best
}

/// `name 22.4 GB`, or `pid N 22.4 GB` when the name sanitises to nothing.
/// The name is untrusted and is cut again here.
fn process_field(process: &GpuProcess) -> String {
    let name = crate::sources::gpu::sanitize_process_name(&process.name);
    let who = if name.is_empty() {
        format!("pid {}", process.pid)
    } else {
        name
    };
    format!("{who} {}", format_gb(process.memory_bytes))
}

/// `{n:.1} GB`. Same unit and precision as one side of the VRAM meter.
fn format_gb(bytes: u64) -> String {
    format!("{:.1} GB", gib(bytes))
}

/// The header (first) model is llama-swap `starting` or `stopping`.
/// False while the header is not up, so `AI DOWN` and the rest keep their words.
fn header_swap(sample: &WatchSample, state: WatchState) -> (bool, bool) {
    if !matches!(state, WatchState::Generating | WatchState::Ready) {
        return (false, false);
    }
    match sample
        .snapshot
        .models
        .first()
        .map(|model| model.state.as_str())
    {
        Some("starting") => (true, false),
        Some("stopping") => (false, true),
        _ => (false, false),
    }
}

/// The header model has been `stopping` for more than [`STUCK_STOPPING`].
fn model_stuck(
    sample: &WatchSample,
    state: WatchState,
    since: &HashMap<String, Instant>,
    now: Instant,
) -> bool {
    matches!(state, WatchState::Generating | WatchState::Ready)
        && sample
            .snapshot
            .models
            .first()
            .and_then(|model| since.get(&model.name))
            .is_some_and(|at| now.saturating_duration_since(*at) > STUCK_STOPPING)
}

/// `sglang  running 1/4 · queued 0 · KV 75k/204k tok 37 % · hit 80 %` for
/// each ready model without `/slots` (#79: the KV item is [`kv_text`]
/// where the engine reports tokens, else `KV 37 %`, its own ratio).
/// Unknown gauges are `--`. Strata has no ratio of its own: `strata
/// running 1/1 · queued 0`, plus its KV tokens when it reports them. Engine numbers follow when the server
/// reports them (#31): `· spec 78 % · 2.9/step · ttft 420ms · itl  31ms ·
/// e2e 12.5 s · preempt 3` (preemptions only once there are some), then
/// the engine-measured speeds of the latest window with finished requests
/// (#35): `· prefill 2,134/s · decode 41.2/s`. A sleeping engine says so
/// first: `vllm  sleeping · running 0 · …`. The line is cut at the panel's
/// edge, so the speeds show where it fits.
#[cfg(test)]
fn backend_lines(sample: &WatchSample, state: WatchState) -> Vec<String> {
    engine_lines(sample, &[], state)
}

/// One line per ready model without `/slots`: its gauges and engine
/// numbers, and for an engine that reports what it is doing (Strata, #54)
/// a second line with its phase and the request's progress. Its KV item
/// is [`kv_text`] where the engine reports tokens (#79). A ready llama.cpp
/// model with KV numbers gets one line too, after them all: `llamacpp  KV
/// 91k/262k tok shared 35 % · 1 session` (its slots are the rows above).
fn engine_lines(sample: &WatchSample, live: &[ModelEngineLive], state: WatchState) -> Vec<String> {
    if !matches!(state, WatchState::Generating | WatchState::Ready) {
        return Vec::new();
    }
    let mut lines = metrics_engine_lines(sample, live);
    for info in sample
        .snapshot
        .models
        .iter()
        .filter(|model| model.state == "ready")
        .filter_map(|model| model.backend)
        .filter(|info| info.kind.has_slots())
    {
        if let Some(text) = info.kv.as_ref().and_then(|kv| kv_text(kv, true)) {
            lines.push(format!("{}  {text}", info.kind.as_str()));
        }
    }
    lines
}

/// The KV item of a SLOTS engine line (#79): `KV 54k/204k tok 26 %`, with
/// `≈` before a count derived from a block-rounded ratio (vLLM), `shared`
/// before the percent for a unified llama.cpp cache (`shared?` when that
/// was assumed from the slots), then `· 2 sessions` when `sessions` (the
/// slot engines; the others show their running requests already), and
/// `· +31k cached` for tokens kept only as reusable cache (SGLang). `None`
/// when neither the use nor the capacity is known.
fn kv_text(kv: &KvUsage, sessions: bool) -> Option<String> {
    let sep = llama_core::detail::SEPARATOR;
    let approx = if kv.approx { "\u{2248}" } else { "" };
    let mut text = match (kv.used, kv.capacity) {
        (Some(used), Some(capacity)) => format!(
            "KV {approx}{}/{} tok",
            layout::compact_k(used),
            layout::compact_k(capacity)
        ),
        (Some(used), None) => format!("KV {approx}{} tok", layout::compact_k(used)),
        (None, Some(capacity)) => format!("KV --/{} tok", layout::compact_k(capacity)),
        (None, None) => return None,
    };
    match (kv.unified, kv.unified_assumed) {
        (Some(true), false) => text.push_str(" shared"),
        (Some(true), true) => text.push_str(" shared?"),
        _ => {}
    }
    if let Some(permille) = kv.permille() {
        text.push_str(&format!(" {} %", (u32::from(permille) + 5) / 10));
    }
    if sessions && let Some(n) = kv.sessions {
        let noun = if n == 1 { "session" } else { "sessions" };
        text.push_str(&format!("{sep}{n} {noun}"));
    }
    if let Some(cached) = kv.cached.filter(|n| *n > 0) {
        text.push_str(&format!("{sep}+{} cached", layout::compact_k(cached)));
    }
    Some(text)
}

/// [`engine_lines`] for the models without `/slots`.
fn metrics_engine_lines(sample: &WatchSample, live: &[ModelEngineLive]) -> Vec<String> {
    let sep = llama_core::detail::SEPARATOR;
    let num = |value: Option<u16>| value.map_or_else(|| "--".to_owned(), |n| n.to_string());
    let pct = |permille: Option<u16>| {
        permille.map_or_else(
            || "--".to_owned(),
            |p| format!("{} %", (u32::from(p) + 5) / 10),
        )
    };
    let mut lines = Vec::new();
    for (model, info) in sample
        .snapshot
        .models
        .iter()
        .filter(|model| model.state == "ready")
        .filter_map(|model| Some((model, model.backend?)))
        .filter(|(_, info)| !info.kind.has_slots())
    {
        lines.push({
            let running = match (info.running, info.max_running) {
                (Some(n), Some(max)) => format!("{n}/{max}"),
                (running, _) => num(running),
            };
            let engine = &info.engine;
            let asleep = if engine.sleeping == Some(true) {
                format!("sleeping{sep}")
            } else {
                String::new()
            };
            let mut line = format!(
                "{}  {asleep}running {running}{sep}queued {}",
                info.kind.as_str(),
                num(info.queued),
            );
            if let Some(text) = info
                .kv
                .as_ref()
                .and_then(|kv| kv_text(kv, info.kind == backend::Backend::Strata))
            {
                line.push_str(&format!("{sep}{text}"));
            } else if info.kind.has_kv_gauge() {
                line.push_str(&format!("{sep}KV {}", pct(info.kv_permille)));
            }
            if info.hit_permille.is_some() {
                line.push_str(&format!("{sep}hit {}", pct(info.hit_permille)));
            }
            if let Some(permille) = engine.spec_permille {
                line.push_str(&format!("{sep}{}", backend::spec_text(permille)));
            }
            if let Some(centi) = engine.spec_len_centi {
                let tenths = (u32::from(centi) + 5) / 10;
                line.push_str(&format!("{sep}{}.{}/step", tenths / 10, tenths % 10));
            }
            for (label, us) in [
                ("ttft", engine.ttft_us),
                ("itl", engine.itl_us),
                ("e2e", engine.e2e_us),
            ] {
                if let Some(us) = us {
                    line.push_str(&format!("{sep}{label} {}", latency_text(us)));
                }
            }
            if let Some(n) = engine.preemptions.filter(|n| *n > 0) {
                line.push_str(&format!("{sep}preempt {n}"));
            }
            if let Some(tenths) = engine.prefill_tps_tenths {
                let whole = (u64::from(tenths) + 5) / 10;
                line.push_str(&format!("{sep}prefill {}/s", layout::commas(whole)));
            }
            if let Some(tenths) = engine.decode_tps_tenths {
                line.push_str(&format!("{sep}decode {}.{}/s", tenths / 10, tenths % 10));
            }
            line
        });
        if let Some(entry) = live.iter().find(|entry| entry.model == model.name)
            && let Some(text) = live_line(&entry.live)
        {
            let indent = " ".repeat(info.kind.as_str().len() + 2);
            lines.push(format!("{indent}{text}"));
        }
    }
    lines
}

/// What a busy engine is doing (#54): its phase (or state word), then the
/// prompt read so far while it reads, or the tokens generated and the
/// live tok/s while it generates. `None` while idle.
fn live_line(live: &EngineLive) -> Option<String> {
    let sep = llama_core::detail::SEPARATOR;
    let state = live.state.as_deref()?;
    if !matches!(state, "reading" | "generating") {
        return None;
    }
    let mut items: Vec<String> = vec![live.phase.clone().unwrap_or_else(|| state.to_owned())];
    match state {
        "reading" => {
            if let (Some(done), Some(total)) = (live.prompt_read, live.prompt_total)
                && total > 0
            {
                let done = done.min(total);
                items.push(format!(
                    "prompt {} ({} %)",
                    layout::comma_pair(done, total),
                    (done * 100 + total / 2) / total
                ));
            } else if let Some(prompt) = live.prompt_tokens {
                items.push(format!("prompt {}", layout::commas(prompt)));
            }
        }
        _ => {
            if let Some(generated) = live.generated {
                items.push(match live.max_tokens.filter(|max| *max > 0) {
                    Some(max) => format!("gen {}", layout::comma_pair(generated, max)),
                    None => format!("gen {}", layout::commas(generated)),
                });
            }
            if let Some(tps) = live.tok_s {
                let tenths = (tps * 10.0).round() as u64;
                items.push(format!("{}.{}/s", tenths / 10, tenths % 10));
            }
        }
    }
    Some(items.join(sep))
}

/// `420 ms` below a second, `12.5 s` below 100 s, else whole seconds.
/// A latency in five cells (#167), for the health line and the engine
/// lines alike: `  9ms`, ` 10ms` and `312ms` under a second, then ` 1.0s`
/// to `99.9s`, then whole seconds (` 100s`).
fn latency_text(us: u32) -> String {
    if us < 999_500 {
        format!("{:>3}ms", (us + 500) / 1000)
    } else if us < 99_950_000 {
        let tenths = (us + 50_000) / 100_000;
        format!("{:>2}.{}s", tenths / 10, tenths % 10)
    } else {
        format!("{:>4}s", (us + 500_000) / 1_000_000)
    }
}

/// Cells [`latency_text`] takes.
const LATENCY_CELLS: usize = 5;

fn slots_line(detail: &LlamaDetail, state: WatchState) -> String {
    match state {
        WatchState::Starting => "...".to_owned(),
        WatchState::AiDown | WatchState::NoLlama => "--".to_owned(),
        WatchState::Generating | WatchState::Ready => {
            let total = detail.slots.len();
            let busy = detail
                .slots
                .iter()
                .filter(|slot| slot.is_processing)
                .count();
            let word = if busy > 0 { "busy" } else { "idle" };
            format!("{busy}/{total} {word}")
        }
    }
}

/// Swap-field word for llama-swap `stopping`.
const STOPPING_SWAP: &str = "stopping";

/// `loading M:SS` while any model is `starting`, else `stopping M:SS`
/// while any is `stopping`, else `none`. The time is how long the one
/// seen longest has been in that state: `M:SS` under an hour, `H:MM:SS`
/// from an hour on. `...` and `--` stay for a header that is not up.
fn swap_line(
    state: WatchState,
    models: &[ModelInfo],
    starting_since: &HashMap<String, Instant>,
    stopping_since: &HashMap<String, Instant>,
    now: Instant,
) -> String {
    match state {
        WatchState::Starting => "...".to_owned(),
        WatchState::AiDown | WatchState::NoLlama => "--".to_owned(),
        WatchState::Generating | WatchState::Ready => {
            if let Some(age) = phase_age(models, "starting", starting_since, now) {
                format!("loading {}", format_phase_age(age))
            } else if let Some(age) = phase_age(models, "stopping", stopping_since, now) {
                format!("{STOPPING_SWAP} {}", format_phase_age(age))
            } else {
                "none".to_owned()
            }
        }
    }
}

/// How long the model in `phase` that was seen first has been there.
/// One not yet in `since` counts as just seen (`now`).
fn phase_age(
    models: &[ModelInfo],
    phase: &str,
    since: &HashMap<String, Instant>,
    now: Instant,
) -> Option<Duration> {
    models
        .iter()
        .filter(|model| model.state == phase)
        .map(|model| {
            let at = since.get(&model.name).copied().unwrap_or(now);
            now.saturating_duration_since(at)
        })
        .max()
}

/// `M:SS` under an hour, `H:MM:SS` from an hour on. Hours are not padded.
fn format_phase_age(duration: Duration) -> String {
    let secs = duration.as_secs();
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let rem = secs % 60;
    if hours > 0 {
        format!("{hours}:{mins:02}:{rem:02}")
    } else {
        format!("{mins}:{rem:02}")
    }
}

/// Strata's batch slots (#81) as numbers-only slot views for the SLOTS
/// sparklines: each slot's held tokens, busy or idle. No text, no task id.
fn strata_views(live: &[ModelEngineLive]) -> Vec<SlotView> {
    live.iter()
        .flat_map(|entry| {
            entry.live.slots().iter().map(|slot| {
                let held = slot.held();
                let decoded = if slot.busy() {
                    slot.generated.unwrap_or(0)
                } else {
                    0
                };
                SlotView {
                    model: entry.model.clone(),
                    id: i64::from(slot.slot),
                    id_task: -1,
                    is_processing: slot.busy(),
                    n_prompt_tokens: held,
                    n_prompt_tokens_processed: 0,
                    n_decoded: decoded,
                    n_ctx: None,
                    ctx_prompt: Some(held - decoded.min(held)),
                    prompt_cached: None,
                    input: Vec::new(),
                    output: Vec::new(),
                    ctx_used: Some(held),
                    resets: Default::default(),
                    last_reset: None,
                }
            })
        })
        .collect()
}

/// One SLOTS row per Strata batch slot (#81): an idle slot's held tokens
/// on its ctx meter; a busy one's prompt against itself once decoding,
/// the newest request's `prompt_read` while it reads, and `--` for any
/// other slot still reading (Strata gives no progress for it).
fn strata_slot_rows(
    live: &[ModelEngineLive],
    models: &[ModelInfo],
    history: &CtxBook,
) -> Vec<Slot> {
    let mut rows = Vec::new();
    for entry in live {
        let n_ctx = models
            .iter()
            .find(|model| model.name == entry.model)
            .and_then(|model| model.detail.as_ref())
            .and_then(|detail| detail.ctx)
            .map(u64::from);
        let newest = entry.live.newest_slot();
        for slot in entry.live.slots() {
            let held = slot.held();
            let prompt = slot.prompt_tokens.unwrap_or(0);
            let busy = slot.busy();
            let decoding = slot.decoding();
            let (done, done_known) = if !busy {
                (0, true)
            } else if decoding {
                (prompt, true)
            } else if newest == Some(slot.slot) {
                (entry.live.prompt_read.unwrap_or(0).min(prompt), true)
            } else {
                (0, false)
            };
            let decoded = if busy { slot.generated.unwrap_or(0) } else { 0 };
            rows.push(Slot {
                id: slot.slot,
                generating: busy,
                done,
                cached: 0,
                total: if busy { prompt } else { 0 },
                open: false,
                done_known,
                decoded,
                ctx_prompt: Some(held - decoded.min(held)),
                n_ctx,
                ctx_history: history.points(&entry.model, i64::from(slot.slot)),
            });
        }
    }
    rows
}

/// One tqLLM SLOTS line (#180): the request llama-watch put on it, if any.
struct TqLine<'a> {
    slot: u32,
    request: Option<&'a crate::metrics::TqRequest>,
}

/// tqLLM's lines (#180): `engine.slots` of them, each with the running
/// request that holds it.
fn tqllm_lines(live: &EngineLive) -> Vec<TqLine<'_>> {
    let count = live
        .request_slots
        .unwrap_or(1)
        .min(crate::tqllm::MAX_TQ_SLOTS);
    (0..count)
        .map(|slot| TqLine {
            slot,
            request: live
                .requests
                .iter()
                .find(|entry| entry.slot == Some(slot))
                .map(|entry| &entry.request),
        })
        .collect()
}

/// A tqLLM request's context (#180): `(prompt part, generated)` of its
/// `context_tokens`, else of the prompt read so far.
fn tqllm_context(req: &crate::metrics::TqRequest) -> (u64, u64) {
    let output = req.output_tokens.unwrap_or(0);
    let context = req
        .context_tokens
        .unwrap_or_else(|| req.prompt_read().saturating_add(output));
    let decoded = output.min(context);
    (context - decoded, decoded)
}

/// tqLLM's SLOTS lines (#180) as numbers-only slot views for the ctx
/// sparklines: a busy line's `context_tokens`, an idle one's 0.
fn tqllm_views(live: &[ModelEngineLive]) -> Vec<SlotView> {
    live.iter()
        .filter(|entry| entry.live.instance.is_some())
        .flat_map(|entry| {
            tqllm_lines(&entry.live).into_iter().map(|line| {
                let (prompt, decoded) = line.request.map_or((0, 0), tqllm_context);
                SlotView {
                    model: entry.model.clone(),
                    id: i64::from(line.slot),
                    id_task: -1,
                    is_processing: line.request.is_some(),
                    n_prompt_tokens: prompt + decoded,
                    n_prompt_tokens_processed: 0,
                    n_decoded: decoded,
                    n_ctx: entry.live.slot_ctx,
                    ctx_prompt: Some(prompt),
                    prompt_cached: None,
                    input: Vec::new(),
                    output: Vec::new(),
                    ctx_used: Some(prompt + decoded),
                    resets: Default::default(),
                    last_reset: None,
                }
            })
        })
        .collect()
}

/// One SLOTS row per tqLLM line (#180), up to `engine.slots`: a busy one
/// shows its request's progress (`prefill_processed_tokens` over
/// `prompt_tokens`, the cached part apart) and its `context_tokens` against
/// `engine.context_per_slot` on the ctx meter; the rest are idle.
fn tqllm_slot_rows(live: &[ModelEngineLive], history: &CtxBook) -> Vec<Slot> {
    let mut rows = Vec::new();
    for entry in live.iter().filter(|entry| entry.live.instance.is_some()) {
        for line in tqllm_lines(&entry.live) {
            let ctx_history = history.points(&entry.model, i64::from(line.slot));
            let Some(req) = line.request else {
                rows.push(Slot {
                    id: line.slot,
                    generating: false,
                    done: 0,
                    cached: 0,
                    total: 0,
                    open: false,
                    done_known: true,
                    decoded: 0,
                    ctx_prompt: Some(0),
                    n_ctx: entry.live.slot_ctx,
                    ctx_history,
                });
                continue;
            };
            let total = req.prompt_tokens.unwrap_or(0);
            let cached = req.cached_read().min(total);
            let done = req.prompt_read().min(total).saturating_sub(cached);
            let (prompt, decoded) = tqllm_context(req);
            rows.push(Slot {
                id: line.slot,
                generating: true,
                done,
                cached,
                total,
                open: false,
                done_known: req.prompt_tokens.is_some(),
                decoded,
                ctx_prompt: Some(prompt),
                n_ctx: entry.live.slot_ctx,
                ctx_history,
            });
        }
    }
    rows
}

fn layout_slots(slots: &[SlotView], history: &CtxBook) -> Vec<Slot> {
    slots
        .iter()
        .filter_map(|slot| {
            let id = u32::try_from(slot.id).ok()?;
            // #82: the exact prompt once decoding; in prefill what the
            // slot holds so far, a lower bound (`n_prompt_tokens` also
            // holds the output, #78).
            let exact = slot.prompt_total();
            Some(Slot {
                id,
                generating: slot.is_processing,
                done: slot.n_prompt_tokens_processed,
                cached: slot.prompt_cached.unwrap_or(0),
                total: exact.unwrap_or(slot.n_prompt_tokens),
                open: exact.is_none() && slot.is_processing,
                done_known: true,
                decoded: slot.n_decoded,
                ctx_prompt: slot.ctx_prompt,
                n_ctx: slot.n_ctx,
                ctx_history: history.points(&slot.model, slot.id),
            })
        })
        .collect()
}

/// Most context sizes remembered (#75).
const MAX_CTX_SIZES: usize = 64;

/// `(display name, llama-swap key)` of each loaded model, when the poller's
/// SETUP list pairs with the view's models.
fn model_ids(llama: &LlamaView, detail: &LlamaDetail) -> Vec<(String, String)> {
    if detail.setup.len() != llama.models.len() {
        return Vec::new();
    }
    llama
        .models
        .iter()
        .zip(&detail.setup)
        .map(|(model, setup)| (model.name.clone(), setup.key.clone()))
        .collect()
}

/// A new poll arrived: feed the in-flight rows and learn context sizes
/// (#75).
fn note_poll(state: &mut TickState, mono: Instant, wall: SystemTime) {
    let ids = model_ids(&state.llama, &state.detail);
    for (model, (name, key)) in state.llama.models.iter().zip(&ids) {
        let from_slots = state
            .detail
            .slots
            .iter()
            .filter(|slot| slot.model == *name)
            .filter_map(|slot| slot.n_ctx)
            .max();
        let size = model
            .detail
            .as_ref()
            .and_then(|d| d.ctx)
            .map(u64::from)
            .or(from_slots)
            .filter(|n| *n > 0);
        if let Some(size) = size {
            if state.ctx_sizes.len() >= MAX_CTX_SIZES && !state.ctx_sizes.contains_key(key) {
                state.ctx_sizes.pop_first();
            }
            state.ctx_sizes.insert(key.clone(), size);
        }
    }
    let poll = crate::inflight::Poll {
        slots: &state.detail.slots,
        engine_live: &state.detail.engine_live,
        requests: &state.detail.requests,
        activity: &state.detail.activity,
        ids: &ids,
        // The llama.cpp headline is a completed-turn average, not this
        // in-flight slot's own measurement.
        prompt_tps: None,
        gen_tps: state.detail.gen_tps,
    };
    state.flights.poll(&poll, mono, wall);
}

/// What RECENT needs beyond the activity rows (#75).
struct RecentCtx<'a> {
    flights: &'a [&'a crate::inflight::Flight],
    ctx_sizes: &'a BTreeMap<String, u64>,
    ids: &'a [(String, String)],
}

fn layout_requests(
    rows: &[ActivityRow],
    state: WatchState,
    zone: &Zone,
    recent: &RecentCtx<'_>,
) -> Vec<Activity> {
    let flights = recent.flights.iter().map(|flight| {
        let key = crate::inflight::model_key(recent.ids, &flight.model);
        let elapsed = flight.polled.saturating_duration_since(flight.started_mono);
        Activity {
            live: false,
            id: 0,
            time: format_wall(flight.started_wall, zone),
            source: String::new(),
            n_ctx: flight.n_ctx.or_else(|| recent.ctx_sizes.get(&key).copied()),
            model: key,
            input_tok: flight.prompt,
            cached_tok: flight.cached,
            cached_unknown: flight.cached_unknown,
            output_tok: flight.decoded,
            prompt_tps: flight.prompt_tps,
            gen_tps: flight.gen_tps,
            prompt_measured: flight.prompt_tps.is_some(),
            gen_measured: flight.gen_tps.is_some(),
            dur: format!("{:.1}s", elapsed.as_secs_f64()),
            err: false,
            inflight: Some(InFlight {
                decoding: flight.decoding,
                processed: flight.processed,
                // #80: a vLLM or SGLang IN is exact or unknown, never a
                // lower bound.
                open: !flight.prompt_known
                    && !flight.prompt_unknown
                    && !matches!(flight.engine, Some(Backend::Vllm | Backend::SgLang)),
                target: flight.progress,
                approx: flight.approx,
                no_prompt: flight.prompt_unknown,
                no_output: flight.output_unknown,
                held: flight.held,
                bytes: flight.bytes,
                reset: flight.reset,
                beats: flight.beats,
            }),
        }
    });
    let flying = !recent.flights.is_empty();
    let finished = rows.iter().enumerate().map(|(index, row)| {
        // #82: the whole prompt and its cached part, per engine.
        let prompt = row.prompt().unwrap_or_default();
        Activity {
            n_ctx: recent.ctx_sizes.get(&row.model).copied(),
            inflight: None,
            live: !flying && index == 0 && state == WatchState::Generating,
            id: u32::try_from(row.id).unwrap_or(0),
            time: local_request_time(&row.time, zone),
            source: row.source.clone(),
            model: row.model.clone(),
            input_tok: prompt.whole,
            cached_tok: prompt.cached,
            // #133: llama-swap's `-1` is already `None`. A reported 0 is
            // `Some(0)` and stays a number. An engine fill lands here as
            // `Some` before this copy is drawn.
            cached_unknown: row.cached_tokens.is_none(),
            output_tok: row.output_tokens.unwrap_or(0),
            prompt_tps: row.prompt_tps.or(row.engine_prompt_tps),
            gen_tps: row.gen_tps.or(row.engine_gen_tps),
            prompt_measured: row.prompt_tps.is_none() && row.engine_prompt_tps.is_some(),
            gen_measured: row.gen_tps.is_none() && row.engine_gen_tps.is_some(),
            dur: match row.duration_ms {
                Some(ms) => format!("{:.1}s", ms as f64 / 1000.0),
                None => "--".to_owned(),
            },
            err: row.status.is_some_and(|code| !(200..300).contains(&code)),
        }
    });
    flights.chain(finished).collect()
}

fn prompt_last(slots: &[SlotView]) -> Option<u64> {
    // #78: the whole prompt once known, else what the slot holds.
    pick_slot(slots).map(|slot| slot.prompt_total().unwrap_or(slot.n_prompt_tokens))
}

fn health(sample: &WatchSample, detail: &LlamaDetail, state: WatchState) -> Vec<HealthSeg> {
    if state == WatchState::Starting {
        return [
            "llama-swap",
            "/running",
            "/slots",
            "metrics",
            "activity",
            "nvml",
            "hwmon",
            "proc",
            "disk",
            "net",
            "psi",
        ]
        .into_iter()
        .map(|name| seg(name, HealthStatus::Pending, ""))
        .collect();
    }
    let down = state == WatchState::AiDown;
    let off = state == WatchState::NoLlama;
    let errors = &sample.snapshot.errors;
    let slots_status = if down || off {
        HealthStatus::Absent
    } else if detail.slots.iter().any(|slot| slot.is_processing) {
        HealthStatus::Ok
    } else {
        HealthStatus::Idle
    };
    let metric_status = if down || off {
        HealthStatus::Absent
    } else {
        HealthStatus::Ok
    };
    vec![
        timed_seg(
            "llama-swap",
            if off {
                HealthStatus::Absent
            } else if down {
                HealthStatus::Down
            } else {
                HealthStatus::Ok
            },
            &if off {
                "off".to_owned()
            } else {
                ms(detail.latencies.running)
            },
        ),
        seg(
            "/running",
            if off {
                HealthStatus::Absent
            } else if down {
                HealthStatus::Down
            } else {
                HealthStatus::Ok
            },
            "",
        ),
        // `ok` and `idle` take the same room (#167).
        HealthSeg {
            word_cells: "idle".len(),
            ..timed_seg("/slots", slots_status, &ms(detail.latencies.slots))
        },
        timed_seg("metrics", metric_status, &ms(detail.latencies.metrics)),
        timed_seg(
            "activity",
            if down || off {
                HealthStatus::Absent
            } else {
                HealthStatus::Ok
            },
            &ms(detail.latencies.activity),
        ),
        seg(
            "nvml",
            if errors.contains(&SourceId::Gpu) {
                HealthStatus::Down
            } else {
                HealthStatus::Ok
            },
            "",
        ),
        seg(
            "hwmon",
            if errors.contains(&SourceId::HwmonCoolant) || errors.contains(&SourceId::HwmonCpu) {
                HealthStatus::Down
            } else {
                HealthStatus::Ok
            },
            "",
        ),
        seg(
            "proc",
            if errors.contains(&SourceId::ProcCpu) || errors.contains(&SourceId::ProcMem) {
                HealthStatus::Down
            } else {
                HealthStatus::Ok
            },
            "",
        ),
        host_seg("disk", sample.host_io.disk),
        host_seg("net", sample.host_io.net),
        host_seg("psi", sample.host_io.psi),
    ]
}

fn host_seg(name: &str, health: crate::sources::hostio::SourceHealth) -> HealthSeg {
    let status = match health {
        crate::sources::hostio::SourceHealth::Ok => HealthStatus::Ok,
        crate::sources::hostio::SourceHealth::Down => HealthStatus::Down,
        crate::sources::hostio::SourceHealth::Absent => HealthStatus::Absent,
    };
    seg(name, status, "")
}

fn seg(name: &str, status: HealthStatus, note: &str) -> HealthSeg {
    HealthSeg {
        name: name.to_owned(),
        status,
        note: note.to_owned(),
        word_cells: 0,
        note_cells: 0,
    }
}

/// A segment whose note is a latency: it keeps [`LATENCY_CELLS`] for it,
/// also before the first read (#167).
fn timed_seg(name: &str, status: HealthStatus, note: &str) -> HealthSeg {
    HealthSeg {
        note_cells: LATENCY_CELLS,
        ..seg(name, status, note)
    }
}

fn ms(latency: Option<Duration>) -> String {
    match latency {
        Some(latency) => latency_text(u32::try_from(latency.as_micros()).unwrap_or(u32::MAX)),
        None => String::new(),
    }
}

fn sanitised_chars(cells: &[Cell]) -> Vec<char> {
    let raw: String = cells.iter().map(|cell| cell.ch).collect();
    let mut sanitised = Vec::new();
    sanitize(&raw, &mut sanitised);
    sanitised
        .into_iter()
        .map(|cell| if cell.is_line_end() { '\n' } else { cell.ch })
        .collect()
}

fn chars_lines(chars: &[char]) -> Vec<String> {
    if chars.is_empty() {
        Vec::new()
    } else {
        vec![chars.iter().collect()]
    }
}

fn pct(value: Option<f32>) -> Option<f64> {
    let value = value?;
    value.is_finite().then_some(f64::from(value))
}

fn temp_i(value: Option<f32>) -> Option<i32> {
    let value = value?;
    value.is_finite().then_some(value.round() as i32)
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / 1_073_741_824.0
}

/// Used memory as the tty shows it: `mem_pct` of `MemTotal`.
fn mem_used_bytes(mem_pct: Option<f32>, total: Option<u64>) -> Option<u64> {
    match (mem_pct, total) {
        (Some(pct), Some(total)) if pct.is_finite() => {
            Some((total as f64 * f64::from(pct.clamp(0.0, 100.0)) / 100.0).round() as u64)
        }
        _ => None,
    }
}

/// The numbers llama-metrics exports beyond the [`Snapshot`] (#11), taken
/// from what this tick's tty frame shows.
fn publish_extras(sample: &WatchSample, tick: &TickState, ctx: &FrameCtx) -> Extras {
    let watch = watch_state(sample, &tick.detail, tick.heard, ctx.llama_enabled);
    let mut slots: Vec<(String, usize, usize)> = Vec::new();
    for slot in &tick.detail.slots {
        match slots.iter_mut().find(|(model, _, _)| *model == slot.model) {
            Some(entry) => {
                entry.1 += usize::from(slot.is_processing);
                entry.2 += 1;
            }
            None => slots.push((slot.model.clone(), usize::from(slot.is_processing), 1)),
        }
    }
    Extras {
        gpu_w: sample.gpu.power_mw.map(|mw| f64::from(mw) / 1000.0),
        gpu_limit_w: sample.gpu.power_limit_mw.map(|mw| f64::from(mw) / 1000.0),
        cpu_w: sample.cpu_w,
        vram_used: sample.gpu.vram_used,
        vram_total: sample.gpu.vram_total,
        mem_used: mem_used_bytes(sample.snapshot.mem_pct, ctx.mem_total_bytes),
        mem_total: ctx.mem_total_bytes,
        slots,
        prompt_cache: tick
            .detail
            .prompt_cache
            .iter()
            .map(|entry| (entry.model.clone(), entry.prompt, entry.cached))
            .collect(),
        slot_ctx: tick
            .detail
            .slots
            .iter()
            .map(|slot| SlotCtx {
                model: slot.model.clone(),
                slot: slot.id,
                used: slot.ctx_used,
                resets: slot.resets,
            })
            .collect(),
        fans: sample
            .fans
            .iter()
            .flat_map(|panel| &panel.fans)
            .map(|fan| {
                (
                    fan.chip.clone(),
                    fan.channel,
                    fan.label.clone(),
                    fan.rpm,
                    fan.pwm,
                )
            })
            .collect(),
        temps: sample
            .temps
            .iter()
            .flatten()
            .map(|t| (t.chip.clone(), t.sensor.clone(), t.tenths))
            .collect(),
        sources: wire_sources(&health(sample, &tick.detail, watch), &tick.detail.latencies),
        suspected_loads: tick.detail.suspected_loads.clone(),
        series: tick.detail.series.clone(),
        disk_read_bps: sample.host_io.disk_read_bps,
        disk_write_bps: sample.host_io.disk_write_bps,
        net_rx_bps: sample.host_io.net_rx_bps,
        net_tx_bps: sample.host_io.net_tx_bps,
        io_psi: sample.host_io.io_psi,
        mem_psi: sample.host_io.mem_psi,
        cpu_psi: sample.host_io.cpu_psi,
        swap_used_bytes: sample.host_io.swap_used_bytes,
        swap_total_bytes: sample.host_io.swap_total_bytes,
    }
}

/// The health line as wire sources: OK and idle are up, down is down, and
/// pending or absent (llama-swap off or down, still starting) is left out.
/// Latencies are the ones the line shows; `running` shares llama-swap's.
fn wire_sources(segs: &[HealthSeg], latencies: &PollLatencies) -> Option<Sources> {
    let mut sources = Sources::default();
    let mut any = false;
    for seg in segs {
        let up = match seg.status {
            HealthStatus::Ok | HealthStatus::Idle => true,
            HealthStatus::Down => false,
            HealthStatus::Pending | HealthStatus::Absent => continue,
        };
        let (slot, latency) = match seg.name.as_str() {
            "llama-swap" => (&mut sources.llama_swap, latencies.running),
            "/running" => (&mut sources.running, latencies.running),
            "/slots" => (&mut sources.slots, latencies.slots),
            "metrics" => (&mut sources.metrics, latencies.metrics),
            "activity" => (&mut sources.activity, latencies.activity),
            "nvml" => (&mut sources.gpu, None),
            "hwmon" => (&mut sources.hwmon, None),
            "proc" => (&mut sources.proc, None),
            _ => continue,
        };
        *slot = Some(SourceWire {
            up,
            latency_s: latency.map(|d| d.as_secs_f32()),
        });
        any = true;
    }
    any.then_some(sources)
}

fn mem_total_bytes(proc_root: &Path) -> Option<u64> {
    let text = std::fs::read_to_string(proc_root.join("meminfo")).ok()?;
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        if key.trim() != "MemTotal" {
            continue;
        }
        let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
        if kb == 0 {
            return None;
        }
        return kb.checked_mul(1024);
    }
    None
}

fn cpu_cores(proc_root: &Path) -> Option<u32> {
    count_cpus(proc_root).ok()
}

fn host_label(proc_root: &Path) -> String {
    let Ok(text) = std::fs::read_to_string(proc_root.join("sys/kernel/hostname")) else {
        return "HOST".to_owned();
    };
    let mut out = String::new();
    for ch in text.trim().chars().take(32) {
        if ch.is_ascii_alphanumeric() || ch == '-' {
            out.push(ch.to_ascii_uppercase());
        }
    }
    if out.is_empty() {
        "HOST".to_owned()
    } else {
        out
    }
}

fn down_label(since: SystemTime, now: SystemTime, zone: &Zone) -> String {
    let stamp = format_wall(since, zone);
    let tod = stamp.get(11..).unwrap_or("--");
    let age = now.duration_since(since).unwrap_or_default();
    format!("{tod} ({})", format_span(age))
}

fn format_wall(wall: SystemTime, zone: &Zone) -> String {
    match wall.duration_since(UNIX_EPOCH) {
        Ok(duration) => format_local(i64::try_from(duration.as_secs()).unwrap_or(i64::MAX), zone),
        Err(_) => "--".to_owned(),
    }
}

fn format_age(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs < 10.0 {
        format!("{secs:.1}s")
    } else {
        format_span(duration)
    }
}

fn format_span(duration: Duration) -> String {
    let secs = duration.as_secs();
    let days = secs / 86_400;
    let hours = (secs % 86_400) / 3_600;
    let mins = (secs % 3_600) / 60;
    let rem = secs % 60;
    if days > 0 {
        format!("{days}d{hours:02}h{mins:02}m")
    } else if hours > 0 {
        format!("{hours}h{mins:02}m{rem:02}s")
    } else if mins > 0 {
        format!("{mins}m{rem:02}s")
    } else {
        format!("{secs}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::{Job, JobState};
    use llama_core::backend::{Backend, BackendInfo};

    fn job_at(name: &str, state: JobState, mtime: SystemTime) -> Job {
        Job {
            file: name.to_owned(),
            name: name.to_owned(),
            kind: Some("training".to_owned()),
            title: None,
            state,
            phase: None,
            step: None,
            total: None,
            elapsed_s: None,
            eta_s: None,
            rate: None,
            rates: Vec::new(),
            values: Vec::new(),
            series: Vec::new(),
            phases: Vec::new(),
            mtime,
        }
    }

    #[test]
    fn job_and_note_serving_keeps_the_note_idle_gets_the_view() {
        let now = UNIX_EPOCH + Duration::from_secs(1_000);
        let fresh = now - Duration::from_secs(10);
        let running = job_at("run", JobState::Running, fresh);

        let (view, note) = job_and_note(std::slice::from_ref(&running), true, 180, now);
        assert!(view.is_none(), "serving keeps inference: {view:?}");
        let note = note.expect("serving live job is a one-line note");
        assert!(note.starts_with("run"), "{note}");

        let (view, note) = job_and_note(std::slice::from_ref(&running), false, 180, now);
        let view = view.expect("idle with a live job is the job view");
        assert_eq!(view.job.name, "run");
        assert!(note.is_none(), "idle live job has no note: {note:?}");

        let done = job_at("wrap", JobState::Done, fresh);
        let (view, note) = job_and_note(std::slice::from_ref(&done), false, 180, now);
        assert!(view.is_none(), "done is not the job view: {view:?}");
        let note = note.expect("done job is a one-line note");
        assert!(note.contains("wrap"), "{note}");
        assert!(note.contains("done"), "{note}");

        let failed = job_at("boom", JobState::Failed, fresh);
        let (view, note) = job_and_note(std::slice::from_ref(&failed), false, 180, now);
        assert!(view.is_none(), "failed is not the job view: {view:?}");
        let note = note.expect("failed job is a one-line note");
        assert!(note.contains("boom"), "{note}");
        assert!(note.contains("failed"), "{note}");
    }

    #[test]
    fn tty_model_takes_serving_from_the_sample() {
        let now = UNIX_EPOCH + Duration::from_secs(1_000);
        let fresh = now - Duration::from_secs(10);
        let running = job_at("run", JobState::Running, fresh);
        let mut tick = TickState::new(2, 6);
        tick.detail.jobs = vec![running];
        let ctx = frame_ctx(None, true);
        let mono = Instant::now();

        let serving = backend_sample(vec![served("m", "ready", None)]);
        let model = tty_model(&serving, &mut tick, mono, now, Zone::utc(), &ctx);
        assert!(
            model.job.is_none(),
            "a serving model keeps inference: {:?}",
            model.job
        );
        let note = model
            .job_note
            .as_deref()
            .expect("a serving model with a live job is a one-line note");
        assert!(note.starts_with("run"), "{note}");

        let idle = backend_sample(Vec::new());
        let model = tty_model(&idle, &mut tick, mono, now, Zone::utc(), &ctx);
        let view = model.job.expect("idle with a live job is the job view");
        assert_eq!(view.job.name, "run");
        assert!(
            model.job_note.is_none(),
            "idle live job has no note: {:?}",
            model.job_note
        );
    }

    /// #188: a llama.cpp headline average is not a live slot rate; scroll
    /// starts at the top and uses this model's completed-turn history.
    #[test]
    fn llama_headline_average_does_not_masquerade_as_live_scroll_rate() {
        let llama = BackendInfo {
            kind: Backend::LlamaCpp,
            ..BackendInfo::default()
        };
        let sample = backend_sample(vec![served("m", "ready", Some(llama))]);
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        tick.detail.setup = vec![ModelSetup {
            key: "m".to_owned(),
            id: "m".to_owned(),
            name: "m".to_owned(),
            ..ModelSetup::default()
        }];
        let mut ctx = frame_ctx(None, true);
        ctx.show_text = true;
        ctx.input_cap = 4_000;
        ctx.output_cap = 4_000;

        let mut first = slot(0, 1, true, "aaaa", "");
        first.n_prompt_tokens = 4;
        first.n_decoded = 0;
        tick.detail.slots = vec![first];
        tick.detail.activity = vec![scroll_row(1, 5.0)];
        let t0 = Instant::now();
        let _ = tty_model(&sample, &mut tick, t0, UNIX_EPOCH, Zone::utc(), &ctx);

        let prompt = "B".repeat(2_000);
        let mut second = slot(0, 2, true, &prompt, "");
        second.n_prompt_tokens = 2_000;
        second.n_decoded = 0;
        tick.detail.slots = vec![second];
        tick.detail.activity.push(scroll_row(2, 5.0));
        tick.detail.prompt_tps = Some(400.0);
        let started = tty_model(&sample, &mut tick, t0, UNIX_EPOCH, Zone::utc(), &ctx);
        assert_eq!(
            started.in_scroll,
            Some(0),
            "the new burst starts at the top"
        );

        let moved = tty_model(
            &sample,
            &mut tick,
            t0 + Duration::from_secs(1),
            UNIX_EPOCH,
            Zone::utc(),
            &ctx,
        );
        let at = moved.in_scroll.expect("the burst is still scrolling");
        assert!(
            (at as f64 - 5.0).abs() < 2.0,
            "the per-model completed-turn history is 5 tok/s, not headline 400: {at}"
        );

        let grown = format!("{prompt}CCCC");
        tick.detail.slots[0].input = secret_cells(&grown);
        let streamed = tty_model(
            &sample,
            &mut tick,
            t0 + Duration::from_secs(1),
            UNIX_EPOCH,
            Zone::utc(),
            &ctx,
        );
        assert_eq!(
            streamed.in_scroll, None,
            "a streaming growth stays on the tail"
        );
    }

    fn scroll_row(seq: u64, speed: f64) -> crate::activity::ActivityRow {
        crate::activity::ActivityRow {
            id: i64::try_from(seq).unwrap_or(0),
            seq,
            time: String::new(),
            source: String::new(),
            model: "m".to_owned(),
            input_tokens: None,
            cached_tokens: None,
            output_tokens: None,
            prompt_tps: Some(speed),
            gen_tps: Some(speed),
            engine_prompt_tps: None,
            engine_gen_tps: None,
            duration_ms: None,
            status: None,
            draft_tokens: None,
            draft_accepted: None,
            captured: false,
            engine: None,
        }
    }

    fn claim_tick(
        claims: Vec<crate::sources::events::ModelClaim>,
        setup_id: &str,
    ) -> (WatchSample, TickState, FrameCtx) {
        let sample = backend_sample(vec![served("m", "ready", None)]);
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        tick.detail.setup = vec![ModelSetup {
            key: crate::activity::model_key(setup_id),
            id: llama_core::names::sanitize(setup_id, llama_core::detail::MAX_FULL_NAME_CHARS),
            swap_id: crate::sources::events::claim_model(setup_id).unwrap_or_default(),
            name: "m".to_owned(),
            ..ModelSetup::default()
        }];
        tick.detail.claims = claims;
        (sample, tick, frame_ctx(None, true))
    }

    fn apply_claim(model: &str, holder: &str) -> crate::sources::events::ModelClaim {
        let mut claims = crate::sources::events::Claims::default();
        let inner = serde_json::json!({
            "claims": [{
                "model": model,
                "holder": holder,
                "auto": false,
                "in_use": true,
            }],
        })
        .to_string();
        let envelope = serde_json::json!({"type": "claims", "data": inner});
        claims.apply(envelope.to_string().as_bytes());
        claims.view().0.into_iter().next().expect("claim")
    }

    fn claim_entry(model: &str, holder: &str) -> crate::sources::events::ModelClaim {
        crate::sources::events::ModelClaim {
            model: model.to_owned(),
            holder: holder.to_owned(),
            auto: false,
            in_use: false,
            expires_at: None,
            hold_until: None,
            hold_reason: String::new(),
            conditional: false,
        }
    }

    /// The countdown follows the wall clock passed into [`tty_model`], not
    /// the Unix epoch.
    #[test]
    fn tty_model_countdown_follows_the_time_passed_in() {
        let wall = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let mut claim = claim_entry("m", "eval-runner");
        claim.expires_at = Some(wall + Duration::from_secs(4 * 60 + 12));
        let (sample, mut tick, ctx) = claim_tick(vec![claim], "m");
        let model = tty_model(&sample, &mut tick, Instant::now(), wall, Zone::utc(), &ctx);
        assert_eq!(model.now, wall);
        let grid = layout::layout(&model, 192, 60);
        let header: String = (0..grid.cols())
            .map(|c| grid.get(c, 0).unwrap().ch)
            .collect();
        assert!(
            header.contains("4:12 left"),
            "countdown must follow {wall:?}: {header}"
        );
    }

    /// [`header_model_id`] pairs `detail.setup` and `models` the way
    /// [`setup_view`] does, and uses the sanitised llama-swap id, not
    /// [`crate::activity::model_key`].
    #[test]
    fn header_model_id_pairs_setup_like_setup_view() {
        let rules = Rules::builtin();
        let llama = BackendInfo {
            kind: Backend::LlamaCpp,
            ..BackendInfo::default()
        };
        let mut sample = backend_sample(vec![
            served("Qwen 35B", "ready", Some(llama)),
            served("flash", "ready", Some(sglang(Some(1)))),
        ]);
        let mut detail = TickState::new(2, 6).detail;
        let q4 = "unsloth-Qwen3.6-35B-A3B-Instruct-UD-Q4_K_XL";
        let q6 = "unsloth-Qwen3.6-35B-A3B-Instruct-UD-Q6_K_XL";
        detail.setup = vec![
            ModelSetup {
                key: crate::activity::model_key(q4),
                id: llama_core::names::sanitize(q4, llama_core::detail::MAX_FULL_NAME_CHARS),
                swap_id: crate::sources::events::claim_model(q4).unwrap_or_default(),
                ..ModelSetup::default()
            },
            ModelSetup {
                key: crate::activity::model_key(q6),
                id: llama_core::names::sanitize(q6, llama_core::detail::MAX_FULL_NAME_CHARS),
                swap_id: crate::sources::events::claim_model(q6).unwrap_or_default(),
                ..ModelSetup::default()
            },
        ];
        assert_eq!(
            crate::activity::model_key(q4),
            crate::activity::model_key(q6),
            "the 32-character key collides"
        );
        assert_eq!(
            header_model_id(&sample, WatchState::Ready, &detail),
            q6,
            "the busy model is the second"
        );
        assert_eq!(
            setup_view(&sample, &detail, WatchState::Ready, &rules)
                .map(|v| v.id)
                .as_deref(),
            Some(q6)
        );
        detail.setup.pop();
        assert!(
            header_model_id(&sample, WatchState::Ready, &detail).is_empty(),
            "a stale pair has no swap_id"
        );
        assert_eq!(
            setup_view(&sample, &detail, WatchState::Ready, &rules)
                .map(|v| v.id)
                .as_deref(),
            Some("flash"),
            "SETUP still falls back to the snapshot name"
        );
        assert!(header_model_id(&sample, WatchState::AiDown, &detail).is_empty());
        sample.snapshot.models.clear();
        assert!(header_model_id(&sample, WatchState::Ready, &detail).is_empty());
    }

    /// A llama-swap id longer than the 48-character display form still
    /// matches its claim: `Claims::apply` → `tty_model` → `layout`.
    #[test]
    fn a_claim_matches_ids_of_48_49_58_and_128_characters() {
        for n in [48usize, 49, 58, 128] {
            let id = "m".repeat(n);
            let holder = format!("h{n}");
            let claim = apply_claim(&id, &holder);
            assert_eq!(claim.model, id, "{n}");
            let (sample, mut tick, ctx) = claim_tick(vec![claim], &id);
            let model = tty_model(
                &sample,
                &mut tick,
                Instant::now(),
                UNIX_EPOCH + Duration::from_secs(1_700_000_000),
                Zone::utc(),
                &ctx,
            );
            assert_eq!(model.model_id, id, "{n}");
            let grid = layout::layout(&model, 192, 60);
            let header: String = (0..grid.cols())
                .map(|c| grid.get(c, 0).unwrap().ch)
                .collect();
            assert!(
                header.contains(&holder),
                "{n}-character id must show its claim: {header}"
            );
        }
    }

    /// Two ids equal in their first 60 characters show only their own claim.
    #[test]
    /// #167 draws every claim; the header model's comes first.
    fn two_ids_equal_in_their_first_60_characters_put_their_own_claim_first() {
        let a = format!("{}A{}", "x".repeat(60), "y".repeat(7));
        let b = format!("{}B{}", "x".repeat(60), "y".repeat(7));
        assert_eq!(&a[..60], &b[..60]);
        assert_eq!(a.chars().count(), 68);
        assert_eq!(b.chars().count(), 68);
        let mut claims = crate::sources::events::Claims::default();
        let inner = serde_json::json!({
            "claims": [
                {"model": a, "holder": "eval-runner", "auto": false, "in_use": true},
                {"model": b, "holder": "chat-ui", "auto": false, "in_use": true},
            ],
        })
        .to_string();
        let envelope = serde_json::json!({"type": "claims", "data": inner});
        claims.apply(envelope.to_string().as_bytes());
        let list = claims.view().0;
        assert_eq!(list.len(), 2);
        let wall = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        for (id, keep, drop) in [
            (a.as_str(), "eval-runner", "chat-ui"),
            (b.as_str(), "chat-ui", "eval-runner"),
        ] {
            let (sample, mut tick, ctx) = claim_tick(list.clone(), id);
            let model = tty_model(&sample, &mut tick, Instant::now(), wall, Zone::utc(), &ctx);
            assert_eq!(model.model_id, id);
            let grid = layout::layout(&model, 192, 60);
            let header: String = (0..grid.cols())
                .map(|c| grid.get(c, 0).unwrap().ch)
                .collect();
            let kept = header.find(keep).expect("own claim");
            assert!(
                header.find(drop).is_none_or(|other| other > kept),
                "{id}: {header}"
            );
        }
    }

    /// A hold that ends after a daylight-saving change uses the zone
    /// offset at that end time.
    #[test]
    fn tty_model_hold_until_uses_the_offset_at_the_end() {
        let zone = Zone::from_posix("EST5EDT,M3.2.0,M11.1.0").expect("posix");
        let now = UNIX_EPOCH
            + Duration::from_secs(
                u64::try_from(crate::localtime::parse_rfc3339("2026-03-08T06:00:00Z").unwrap())
                    .unwrap(),
            );
        let until = UNIX_EPOCH
            + Duration::from_secs(
                u64::try_from(crate::localtime::parse_rfc3339("2026-03-08T12:00:00Z").unwrap())
                    .unwrap(),
            );
        let mut claim = claim_entry("m", "operator");
        claim.hold_until = Some(until);
        claim.hold_reason = "overnight eval".to_owned();
        let (sample, mut tick, ctx) = claim_tick(vec![claim], "m");
        let model = tty_model(&sample, &mut tick, Instant::now(), now, &zone, &ctx);
        assert_eq!(model.tz_offset_s, -5 * 3600);
        assert_eq!(model.claims[0].hold_tz_offset_s, -4 * 3600);
        let grid = layout::layout(&model, 192, 60);
        let header: String = (0..grid.cols())
            .map(|c| grid.get(c, 0).unwrap().ch)
            .collect();
        assert!(header.contains("until 08:00"), "end time in EDT: {header}");
        assert!(!header.contains("until 07:00"), "{header}");
    }

    /// #167: the reservation, a guest claim and a displaced note go from
    /// [`LlamaDetail`] through `tty_model` to header row 0, with the local
    /// zone at the hold's end and at the note's time.
    #[test]
    fn reservation_guest_and_note_reach_the_header_in_local_time() {
        let at = |text: &str| {
            UNIX_EPOCH
                + Duration::from_secs(
                    u64::try_from(crate::localtime::parse_rfc3339(text).unwrap()).unwrap(),
                )
        };
        let zone = Zone::from_posix("CST6CDT,M3.2.0,M11.1.0").expect("posix");
        let now = at("2026-10-20T21:00:00Z");
        let mut guest = claim_entry("m", "guest-ui");
        guest.conditional = true;
        guest.hold_until = Some(at("2026-10-20T22:00:00Z"));
        let (sample, mut tick, ctx) = claim_tick(vec![guest], "m");
        tick.detail.gpu = Some(crate::sources::events::GpuReservation {
            holder: "sample-holder".to_owned(),
            external: true,
            draining: true,
            in_use: false,
            expires_at: None,
            hold_until: Some(at("2026-10-20T23:30:00Z")),
            hold_reason: "invented queue".to_owned(),
        });
        tick.detail.displaced = vec![crate::sources::events::DisplacedNote {
            holder: "former-ui".to_owned(),
            model: "tiel".to_owned(),
            to: "m".to_owned(),
            by: "sample-holder".to_owned(),
            at: at("2026-10-20T20:55:00Z"),
        }];
        let model = tty_model(&sample, &mut tick, Instant::now(), now, &zone, &ctx);
        let grid = layout::layout(&model, 320, 60);
        let header: String = (0..grid.cols())
            .map(|c| grid.get(c, 0).unwrap().ch)
            .collect();
        assert!(
            header.contains(
                "gpu sample-holder \u{00B7} external \u{00B7} draining \u{00B7} hold until 18:30 \u{00B7} invented queue"
            ),
            "{header}"
        );
        // `guest` wins the label; the hold's end still shows.
        assert!(
            header.contains("guest guest-ui on m \u{00B7} until 17:00"),
            "{header}"
        );
        assert!(
            header.contains("displaced tiel -> m by sample-holder 15:55"),
            "{header}"
        );
    }

    /// #35: RECENT takes the engine's speeds only where llama-swap gave
    /// none, and marks them; a llama.cpp row keeps its own.
    /// #75: a busy slot is RECENT's first row, keyed and scaled like the
    /// finished rows, and the newest finished row loses the `gen` mark.
    #[test]
    fn inflight_rows_lead_recent_with_their_model_key_and_context() {
        let mut busy = slot(0, 41, true, "", "");
        busy.model = "Qwen 35B".to_owned();
        // #78: in prefill the slot holds cached + processed so far.
        (busy.n_prompt_tokens, busy.n_prompt_tokens_processed) = (20_000, 20_000);
        busy.prompt_cached = Some(0);
        busy.n_ctx = None;
        let ids = vec![("Qwen 35B".to_owned(), "qwen3.6-35b-a3b".to_owned())];
        let page = br#"{"data":[{"id":7,"timestamp":"2026-10-03T10:00:07Z","model":"qwen3.6-35b-a3b","tokens":{"input_tokens":900}}]}"#;
        let rows = crate::activity::parse_activity(page).expect("page");
        let mut tracker = crate::inflight::Tracker::default();
        let now = Instant::now();
        tracker.poll(
            &crate::inflight::Poll {
                slots: &[busy],
                engine_live: &[],
                requests: &[],
                activity: &rows,
                ids: &ids,
                prompt_tps: Some(2_000.0),
                gen_tps: None,
            },
            now,
            UNIX_EPOCH,
        );
        let flights = tracker.flights();
        let sizes: BTreeMap<String, u64> = [("qwen3.6-35b-a3b".to_owned(), 262_144)].into();
        let recent = RecentCtx {
            flights: &flights,
            ctx_sizes: &sizes,
            ids: &ids,
        };
        let shown = layout_requests(&rows, WatchState::Generating, Zone::utc(), &recent);
        assert_eq!(shown.len(), 2);
        let live = &shown[0];
        assert_eq!(live.model, "qwen3.6-35b-a3b");
        assert_eq!(
            (live.input_tok, live.cached_tok, live.output_tok),
            (20_000, 0, 0)
        );
        assert_eq!(live.n_ctx, Some(262_144));
        assert_eq!(
            live.inflight.map(|f| (f.decoding, f.processed, f.open)),
            Some((false, 20_000, true)),
            "the whole prompt is not known in prefill"
        );
        assert_eq!(
            (live.prompt_tps, live.prompt_measured),
            (Some(2_000.0), true)
        );
        assert!(!shown[1].live, "the gen mark moved to the in-flight row");
        assert_eq!(shown[1].n_ctx, Some(262_144));
        // Without flights the newest finished row keeps today's mark.
        let none = RecentCtx {
            flights: &[],
            ctx_sizes: &sizes,
            ids: &ids,
        };
        assert!(layout_requests(&rows, WatchState::Generating, Zone::utc(), &none)[0].live);
    }

    /// #82: IN is the whole prompt and the bar's cached run its reused
    /// part, whatever the engine's `input_tokens` meant.
    #[test]
    fn recent_in_and_the_bar_read_input_tokens_per_engine() {
        let page = br#"{"data":[
            {"id":3,"timestamp":"2026-10-03T10:00:03Z","model":"strata","tokens":{"input_tokens":21,"cache_tokens":102231}},
            {"id":2,"timestamp":"2026-10-03T10:00:02Z","model":"vllm","tokens":{"input_tokens":9000,"cache_tokens":8192}},
            {"id":1,"timestamp":"2026-10-03T10:00:01Z","model":"lcpp","tokens":{"input_tokens":500,"cache_tokens":100}}
        ]}"#;
        let mut rows = crate::activity::parse_activity(page).expect("page");
        rows[0].engine = Some(Backend::Strata);
        rows[1].engine = Some(Backend::Vllm);
        rows[2].engine = Some(Backend::LlamaCpp);
        let recent = RecentCtx {
            flights: &[],
            ctx_sizes: &BTreeMap::new(),
            ids: &[],
        };
        let shown = layout_requests(&rows, WatchState::Ready, Zone::utc(), &recent);
        let ins: Vec<(u64, u64)> = shown
            .iter()
            .map(|req| (req.input_tok, req.cached_tok))
            .collect();
        assert_eq!(ins, vec![(102_252, 102_231), (9_000, 8_192), (600, 100)]);
    }

    /// #133: a missing cached count stays unknown on the row. A reported 0
    /// is a measurement. The split's cached number is still 0 either way.
    #[test]
    fn recent_cached_unknown_is_not_a_reported_zero() {
        let page = br#"{"data":[
            {"id":6,"timestamp":"2026-10-03T10:00:06Z","model":"sglang","tokens":{"input_tokens":1000,"output_tokens":40,"cache_tokens":-1}},
            {"id":5,"timestamp":"2026-10-03T10:00:05Z","model":"sglang","tokens":{"input_tokens":1000,"output_tokens":40,"cache_tokens":0}},
            {"id":4,"timestamp":"2026-10-03T10:00:04Z","model":"vllm","tokens":{"input_tokens":900,"output_tokens":20,"cache_tokens":-1}},
            {"id":3,"timestamp":"2026-10-03T10:00:03Z","model":"vllm","tokens":{"input_tokens":900,"output_tokens":20,"cache_tokens":0}},
            {"id":2,"timestamp":"2026-10-03T10:00:02Z","model":"lcpp","tokens":{"input_tokens":500,"output_tokens":10,"cache_tokens":-1}},
            {"id":1,"timestamp":"2026-10-03T10:00:01Z","model":"lcpp","tokens":{"input_tokens":500,"output_tokens":10,"cache_tokens":0}}
        ]}"#;
        let mut rows = crate::activity::parse_activity(page).expect("page");
        for (row, engine) in rows.iter_mut().zip([
            Backend::SgLang,
            Backend::SgLang,
            Backend::Vllm,
            Backend::Vllm,
            Backend::LlamaCpp,
            Backend::LlamaCpp,
        ]) {
            row.engine = Some(engine);
        }
        let recent = RecentCtx {
            flights: &[],
            ctx_sizes: &BTreeMap::new(),
            ids: &[],
        };
        let shown = layout_requests(&rows, WatchState::Ready, Zone::utc(), &recent);
        let got: Vec<(u64, u64, bool)> = shown
            .iter()
            .map(|req| (req.input_tok, req.cached_tok, req.cached_unknown))
            .collect();
        assert_eq!(
            got,
            vec![
                (1_000, 0, true),
                (1_000, 0, false),
                (900, 0, true),
                (900, 0, false),
                (500, 0, true),
                (500, 0, false),
            ]
        );
    }

    /// #133: an omitted in-flight cache is unknown. Strata's explicit 0,
    /// and a llama.cpp slot that reported 0, stay 0.
    #[test]
    fn inflight_omitted_cache_is_unknown_and_a_reported_zero_is_not() {
        use crate::live::LiveTokens;
        use crate::poller::LiveRequest;
        let mut omitted = slot(0, 1, true, "", "");
        omitted.model = "cpp".to_owned();
        omitted.n_prompt_tokens = 100;
        omitted.n_decoded = 5;
        omitted.prompt_cached = None;
        let mut zero = slot(1, 2, true, "", "");
        zero.model = "cpp0".to_owned();
        zero.n_prompt_tokens = 80;
        zero.n_prompt_tokens_processed = 80;
        zero.n_decoded = 3;
        zero.prompt_cached = Some(0);
        let vllm = LiveRequest {
            model: "vllm".to_owned(),
            id: "a".to_owned(),
            engine: Backend::Vllm,
            started: Instant::now(),
            started_wall: UNIX_EPOCH,
            resp_bytes: 10,
            tokens: Some(LiveTokens {
                prompt: Some(900),
                cached: None,
                output: Some(4),
                ..LiveTokens::default()
            }),
        };
        let strata = ModelEngineLive {
            model: "strata".to_owned(),
            live: EngineLive {
                state: Some("generating".to_owned()),
                prompt_tokens: Some(400),
                generated: Some(8),
                ..EngineLive::default()
            },
        };
        let mut tracker = crate::inflight::Tracker::default();
        let now = Instant::now();
        tracker.poll(
            &crate::inflight::Poll {
                slots: &[omitted, zero],
                engine_live: &[strata],
                requests: &[vllm],
                activity: &[],
                ids: &[],
                prompt_tps: None,
                gen_tps: None,
            },
            now,
            UNIX_EPOCH,
        );
        let flights = tracker.flights();
        let recent = RecentCtx {
            flights: &flights,
            ctx_sizes: &BTreeMap::new(),
            ids: &[],
        };
        let shown = layout_requests(&[], WatchState::Ready, Zone::utc(), &recent);
        let row = |name: &str| {
            shown
                .iter()
                .find(|req| req.model == name)
                .unwrap_or_else(|| panic!("missing {name}"))
        };
        let cpp = row("cpp");
        assert!(cpp.cached_unknown);
        assert_eq!(cpp.cached_tok, 0);
        assert_eq!(cpp.input_tok, 96, "held 100, five decoded, cache omitted");
        let cpp0 = row("cpp0");
        assert!(!cpp0.cached_unknown);
        assert_eq!((cpp0.input_tok, cpp0.cached_tok), (80, 0));
        let vllm = row("vllm");
        assert!(vllm.cached_unknown);
        assert_eq!((vllm.input_tok, vllm.cached_tok), (900, 0));
        let strata = row("strata");
        assert!(
            !strata.cached_unknown,
            "Strata counts every live token as new"
        );
        assert_eq!((strata.input_tok, strata.cached_tok), (400, 0));
    }

    #[test]
    fn recent_rows_use_engine_speeds_only_where_llama_swap_had_none() {
        let page = br#"{"data":[
            {"id":3,"timestamp":"2026-10-03T10:00:03Z","model":"fast","tokens":{"prompt_per_second":1193.6,"tokens_per_second":50.5}},
            {"id":2,"timestamp":"2026-10-03T10:00:02Z","model":"vllm","tokens":{"prompt_per_second":-1,"tokens_per_second":-1}},
            {"id":1,"timestamp":"2026-10-03T10:00:01Z","model":"vllm","tokens":{"prompt_per_second":-1,"tokens_per_second":-1}}
        ]}"#;
        let mut rows = crate::activity::parse_activity(page).expect("page");
        rows[0].engine_prompt_tps = Some(9.0);
        rows[0].engine_gen_tps = Some(9.0);
        rows[1].engine_prompt_tps = Some(2134.4);
        rows[1].engine_gen_tps = Some(41.25);
        let recent = RecentCtx {
            flights: &[],
            ctx_sizes: &BTreeMap::new(),
            ids: &[],
        };
        let shown = layout_requests(&rows, WatchState::Ready, Zone::utc(), &recent);
        assert_eq!(shown[0].prompt_tps, Some(1193.6), "llama.cpp keeps its own");
        assert_eq!(shown[0].gen_tps, Some(50.5));
        assert!(!shown[0].prompt_measured && !shown[0].gen_measured);
        assert_eq!(shown[1].prompt_tps, Some(2134.4));
        assert_eq!(shown[1].gen_tps, Some(41.25));
        assert!(shown[1].prompt_measured && shown[1].gen_measured);
        assert_eq!((shown[2].prompt_tps, shown[2].gen_tps), (None, None));
        assert!(!shown[2].prompt_measured && !shown[2].gen_measured);
        assert_eq!(
            layout::prompt_rate_text(shown[1].prompt_tps, shown[1].prompt_measured),
            "~2,134"
        );
        assert_eq!(
            layout::gen_rate_text(shown[1].gen_tps, shown[1].gen_measured),
            "~41.3"
        );
        assert_eq!(layout::prompt_rate_text(None, false), "--");
        assert_eq!(layout::gen_rate_text(None, false), "--");
    }

    /// #48: RECENT's TIME and the header clock are in the same zone; a
    /// timestamp that is not RFC 3339 is shown as it came.
    #[test]
    fn recent_time_and_the_header_clock_share_the_host_zone() {
        let page = br#"{"data":[
            {"id":2,"timestamp":"2026-10-04T05:52:17Z","model":"m"},
            {"id":1,"timestamp":"18:47:01","model":"m"}
        ]}"#;
        let sample = backend_sample(vec![served("m", "ready", None)]);
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        tick.detail.activity = crate::activity::parse_activity(page).expect("page");
        let ctx = frame_ctx(None, true);
        let wall = UNIX_EPOCH + Duration::from_secs(1_791_093_137);
        let zone = Zone::fixed(-5 * 3600);
        let model = tty_model(&sample, &mut tick, Instant::now(), wall, &zone, &ctx);
        assert_eq!(model.clock, "2026-10-04 00:52:17");
        assert_eq!(model.requests[0].time, "2026-10-04 00:52:17");
        assert_eq!(model.requests[1].time, "18:47:01");
        let grid = layout::layout(&model, 240, 67);
        let rows: Vec<String> = (0..grid.rows())
            .map(|row| {
                (0..grid.cols())
                    .map(|col| grid.get(col, row).map_or(' ', |cell| cell.ch))
                    .collect()
            })
            .collect();
        assert!(
            rows.iter()
                .any(|row| row.contains("2026-10-04 00:52:17") && !row.contains("COOL")),
            "RECENT shows local time"
        );
        assert!(rows.iter().any(|row| row.contains("18:47:01")));
        assert!(
            !rows.iter().any(|row| row.contains("05:52:17")),
            "no UTC left"
        );
        // The raw text stays for sorting and the row fingerprint.
        assert_eq!(tick.detail.activity[0].time, "2026-10-04T05:52:17Z");
    }

    #[test]
    fn layout_slots_keeps_context_fields() {
        let view = SlotView {
            model: "m".to_owned(),
            id: 0,
            id_task: 1,
            is_processing: true,
            n_prompt_tokens: 91_000,
            n_prompt_tokens_processed: 91_000,
            n_decoded: 816,
            n_ctx: Some(262_144),
            ctx_prompt: Some(91_000),
            prompt_cached: None,
            ctx_used: None,
            resets: Default::default(),
            last_reset: None,
            input: Vec::new(),
            output: Vec::new(),
        };
        let slots = layout_slots(&[view], &CtxBook::new(6));
        assert_eq!(slots[0].ctx_prompt, Some(91_000));
        assert_eq!(slots[0].n_ctx, Some(262_144));
        assert_eq!(slots[0].decoded, 816);
    }

    /// #82: SLOTS counts the processed tokens against the exact prompt once
    /// the slot decodes, and against what it holds so far, a lower bound,
    /// in prefill; b11429 bodies.
    #[test]
    fn layout_slots_counts_against_the_exact_prompt_or_a_lower_bound() {
        let fixture = |name: &str| {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/llama")
                .join(name);
            std::fs::read(path).expect("fixture")
        };
        let mut book = crate::slots::SlotBook::without_text();
        assert!(book.apply("m", "m", &fixture("slots-b11429-prefill.json"), 0, 0));
        let prefill = layout_slots(&book.slots(), &CtxBook::new(6));
        // 61,440 cached + 8,192 processed so far; the prompt is longer.
        assert_eq!(
            (prefill[0].done, prefill[0].total, prefill[0].open),
            (8_192, 69_632, true)
        );
        let mut book = crate::slots::SlotBook::without_text();
        assert!(book.apply("m", "m", &fixture("slots-b11429-decode.json"), 0, 0));
        let view = book.slots();
        let decode = layout_slots(&view, &CtxBook::new(6));
        let slot = &view[0];
        assert!(slot.n_decoded > 0);
        let exact = slot.prompt_total().expect("decoding: exact");
        assert_eq!((decode[0].total, decode[0].open), (exact, false));
        assert!(
            exact < slot.n_prompt_tokens,
            "the held count includes output"
        );
        assert_eq!(decode[0].done, slot.n_prompt_tokens_processed);
        assert_eq!(decode[0].cached, slot.prompt_cached.unwrap_or(0));

        let render = |slots: Vec<SlotView>| -> Vec<String> {
            let sample = backend_sample(vec![served("m", "ready", None)]);
            let mut tick = TickState::new(2, 6);
            tick.heard = true;
            tick.detail.slots = slots;
            let ctx = frame_ctx(None, true);
            let model = tty_model(
                &sample,
                &mut tick,
                Instant::now(),
                UNIX_EPOCH,
                Zone::utc(),
                &ctx,
            );
            let grid = layout::layout(&model, 240, 67);
            (0..grid.rows())
                .map(|row| {
                    (0..grid.cols())
                        .map(|col| grid.get(col, row).map_or(' ', |cell| cell.ch))
                        .collect()
                })
                .collect()
        };
        let mut book = crate::slots::SlotBook::without_text();
        assert!(book.apply("m", "m", &fixture("slots-b11429-prefill.json"), 0, 0));
        let text = render(book.slots());
        assert!(
            text.iter().any(|row| row.contains("8,192/69,632+")),
            "lower bound"
        );
        let pair = layout::comma_pair(decode[0].done, exact);
        let text = render(view);
        assert!(
            text.iter()
                .any(|row| row.contains(&pair) && !row.contains(&format!("{pair}+"))),
            "exact: {pair}"
        );
    }

    fn slot(id: i64, id_task: i64, busy: bool, input: &str, output: &str) -> SlotView {
        SlotView {
            model: "m".to_owned(),
            id,
            id_task,
            is_processing: busy,
            n_prompt_tokens: 0,
            n_prompt_tokens_processed: 0,
            n_decoded: 0,
            n_ctx: None,
            ctx_prompt: None,
            prompt_cached: None,
            ctx_used: None,
            resets: Default::default(),
            last_reset: None,
            input: secret_cells(input),
            output: secret_cells(output),
        }
    }

    fn text_of(lines: &[String]) -> String {
        lines.concat()
    }

    /// T49: a live tty11 photo. A stale idle "hi" slot sat after the busy one.
    #[test]
    fn in_and_out_follow_the_busy_slot_not_a_stale_idle_one() {
        let long = format!(
            "{}\nSuccessfully wrote to tests/test_live.py",
            "x".repeat(5_000)
        );
        let slots = [
            slot(0, 50, false, "system\nhi", "Hello! How can I help?"),
            slot(1, -1, false, "", ""),
            slot(2, 40, true, &long, "BUSY-OUT"),
            slot(3, 10, false, "system\nhi", "Hi there"),
        ];
        assert_eq!(pick_slot(&slots).map(|s| s.id), Some(2));
        let mut replay = OutReplay::default();
        let text = frame_text(&slots, &mut replay, 8192, 8192);
        let inn = text_of(&text.in_lines);
        let out = text_of(&text.out_lines);
        assert!(
            inn.ends_with("Successfully wrote to tests/test_live.py"),
            "{inn:?}"
        );
        assert!(!inn.contains("hi"), "{inn:?}");
        assert_eq!(out, "BUSY-OUT");
        assert_eq!(prompt_last(&slots), Some(0));
    }

    #[test]
    fn two_busy_slots_show_the_newest_task_in_both_panels() {
        let slots = [
            slot(0, 3, false, "IDLE-IN", "IDLE-OUT"),
            slot(1, 9, true, "NEW-IN", "NEW-OUT"),
            slot(2, 7, true, "OLD-IN", "OLD-OUT"),
        ];
        let mut replay = OutReplay::default();
        let text = frame_text(&slots, &mut replay, 8192, 8192);
        assert_eq!(text_of(&text.in_lines), "NEW-IN");
        assert_eq!(text_of(&text.out_lines), "NEW-OUT");
    }

    #[test]
    fn all_idle_shows_the_task_that_finished_last() {
        let slots = [
            slot(0, 12, false, "MID-IN", "MID-OUT"),
            slot(1, 30, false, "LAST-IN", "LAST-OUT"),
            slot(2, 5, false, "FIRST-IN", "FIRST-OUT"),
            slot(3, -1, false, "", ""),
        ];
        let mut replay = OutReplay::default();
        let text = frame_text(&slots, &mut replay, 8192, 8192);
        assert_eq!(text_of(&text.in_lines), "LAST-IN");
        assert_eq!(text_of(&text.out_lines), "LAST-OUT");
        assert!(pick_slot(&[]).is_none());
    }

    #[test]
    fn switching_slots_restarts_the_out_replay_and_swaps_in() {
        let mut replay = OutReplay::default();
        let first = [
            slot(0, 4, true, "PROMPT-A", "alpha output"),
            slot(1, 2, false, "PROMPT-B-OLD", "zzz"),
        ];
        frame_text(&first, &mut replay, 8192, 8192);
        // Let the replay finish, then grow the same slot: the cursor keeps
        // what is already on screen.
        for _ in 0..10 {
            frame_text(&first, &mut replay, 8192, 8192);
        }
        assert_eq!(replay.frame, None);
        let grown = [
            slot(0, 4, true, "PROMPT-A", "alpha output more"),
            slot(1, 2, false, "PROMPT-B-OLD", "zzz"),
        ];
        let text = frame_text(&grown, &mut replay, 8192, 8192);
        assert_eq!(text.out_shown, "alpha output".len());
        assert_eq!(text.replay_frame, Some(0));
        assert_eq!(text_of(&text.in_lines), "PROMPT-A");

        // A newer task starts on slot 1: IN and OUT move together and the
        // OUT cursor starts from zero, not from slot 0's count.
        let switched = [
            slot(0, 4, false, "PROMPT-A", "alpha output more"),
            slot(1, 6, true, "PROMPT-B", "beta"),
        ];
        let text = frame_text(&switched, &mut replay, 8192, 8192);
        assert_eq!(text_of(&text.in_lines), "PROMPT-B");
        assert_eq!(text_of(&text.out_lines), "beta");
        assert_eq!(text.out_shown, 0);
        assert_eq!(text.replay_frame, Some(0));
    }

    #[test]
    fn unix_epoch_is_utc_midnight() {
        assert_eq!(format_wall(UNIX_EPOCH, Zone::utc()), "1970-01-01 00:00:00");
    }

    #[test]
    fn a_short_overlap_keeps_the_common_prefix_of_one_two() {
        let prev: Vec<char> = " one two\nsee: ".chars().collect();
        let next: Vec<char> = " one two three\nsee: 4".chars().collect();
        let (_tail, shown) = advance_output_tail(&prev, &next, prev.len(), 10_000);
        assert!(shown >= 8, "shown {shown}");
    }

    #[test]
    fn a_short_overlap_keeps_the_common_prefix_of_sure() {
        let prev: Vec<char> = "Sure, here it is\nOk. S".chars().collect();
        let next: Vec<char> = "Sure, here it is:\nOk. Su".chars().collect();
        let (_tail, shown) = advance_output_tail(&prev, &next, prev.len(), 10_000);
        assert!(shown >= 16, "shown {shown}");
    }

    #[test]
    fn trimmed_tail_moves_shown_back_by_the_dropped_chars() {
        let prev: Vec<char> = "abcdefghij".chars().collect();
        let next: Vec<char> = "abcdefghijXYZ".chars().collect();
        let (tail, shown) = advance_output_tail(&prev, &next, 10, 10);
        assert_eq!(tail.iter().collect::<String>(), "defghijXYZ");
        assert_eq!(shown, 7);
    }

    #[test]
    fn a_line_end_counts_as_a_shown_char() {
        let prev = ['a', '\n', 'b', 'c'];
        let next = ['a', '\n', 'b', 'c', 'X'];
        let (tail, shown) = advance_output_tail(&prev, &next, 4, 4);
        assert_eq!(tail, ['\n', 'b', 'c', 'X']);
        assert_eq!(shown, 3);
    }

    #[test]
    fn a_shared_prefix_without_continuation_shows_only_the_prefix() {
        // T22 nit: with no continuation match, `shown` must not survive
        // past the shared prefix, or new text appears without the reveal.
        let prev: Vec<char> = "abcdefXY".chars().collect();
        let next: Vec<char> = "abcdQQ".chars().collect();
        assert_eq!(align_shown(&prev, &next, 8), 4);
    }

    #[test]
    fn a_discontinuous_tail_resets_shown() {
        let prev: Vec<char> = "abcdefghij".chars().collect();
        let next: Vec<char> = "ZZZZZZZZZZ".chars().collect();
        let (tail, shown) = advance_output_tail(&prev, &next, 10, 10);
        assert_eq!(tail.iter().collect::<String>(), "ZZZZZZZZZZ");
        assert_eq!(shown, 0);
    }

    #[test]
    fn host_nproc_counts_a_fake_stat_of_32_under_affinity_of_one() {
        const NAME: &str =
            "service::tests::host_nproc_counts_a_fake_stat_of_32_under_affinity_of_one";
        // rustix's affinity calls need the `thread` feature, which this crate
        // does not enable. Re-exec under `taskset -c 0` so this process has
        // an affinity of one CPU, the same shape as `CPUAffinity=31`.
        if std::env::var_os("LLAMA_WATCH_AFFINITY_ONE").is_none() {
            let exe = std::env::current_exe().expect("test executable");
            let output = std::process::Command::new("taskset")
                .args(["-c", "0"])
                .arg(&exe)
                .arg(NAME)
                .arg("--exact")
                .env("LLAMA_WATCH_AFFINITY_ONE", "1")
                .output()
                .expect("taskset -c 0");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success() && stdout.contains("1 passed"),
                "affinity-1 child failed: {:?}\n{stdout}\n{stderr}",
                output.status
            );
            return;
        }
        let affinity = std::thread::available_parallelism()
            .expect("available_parallelism")
            .get();
        assert_eq!(affinity, 1, "this process is pinned to one CPU");

        let scratch = CpuScratch::new("host-cpus");
        let proc_root = scratch.path().join("proc");
        std::fs::create_dir_all(&proc_root).expect("proc dir");
        std::fs::write(proc_root.join("stat"), stat_with_cpus(32)).expect("stat");

        let nproc = host_nproc(&proc_root);
        assert_eq!(nproc, 32, "cpuN lines, ignoring the affinity mask");

        let cfg = scratch.path().join("watch.toml");
        std::fs::write(&cfg, "[collector]\ncpu_top_k = 8\n").expect("toml");
        let config = Config::load_validated(&cfg, nproc)
            .expect("cpu_top_k 8 is inside 1..=32 from /proc/stat");
        assert_eq!(config.collector.cpu_top_k, 8);
        assert_eq!(cpu_cores(&proc_root), Some(32));

        let roots = Roots {
            proc: proc_root,
            sys: scratch.path().to_path_buf(),
        };
        let models = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        run_loop(LoopInput {
            config: &config,
            feed: IdleOnce { sent: false },
            sampler: FixedCpu,
            publisher: AcceptPublish,
            render: RecordModels {
                models: std::sync::Arc::clone(&models),
            },
            clock: StepClock {
                mono: Instant::now(),
                wall: SystemTime::UNIX_EPOCH,
            },
            notify: QuietNotify,
            stop: OneTick(std::sync::atomic::AtomicU64::new(0)),
            log: QuietLog,
            roots: &roots,
            started: Instant::now(),
        });
        let models = models.lock().expect("models").clone();
        let model = models.first().expect("one drawn frame");
        assert_eq!(model.cpu_cores, Some(32));
        let grid = layout::layout(model, 200, 60);
        let painted = grid_text(&grid);
        assert!(
            painted.contains("32c"),
            "CPU header should show the host count\n{painted}"
        );
    }

    struct NeverFeed;

    impl LlamaFeed for NeverFeed {
        fn poll(&mut self) -> Option<(LlamaView, crate::poller::LlamaDetail)> {
            None
        }
    }

    /// One tick of [`run_loop`] with `toml`, a feed that never yields, and
    /// fixed host numbers. Returns the drawn model.
    fn one_frame(label: &str, toml: &str) -> TtyModel {
        let scratch = CpuScratch::new(label);
        let cfg = scratch.path().join("watch.toml");
        std::fs::write(&cfg, toml).expect("toml");
        let config = Config::load_validated(&cfg, 8).expect("config");
        let roots = Roots {
            proc: scratch.path().join("proc"),
            sys: scratch.path().to_path_buf(),
        };
        let models = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        run_loop(LoopInput {
            config: &config,
            feed: NeverFeed,
            sampler: FixedCpu,
            publisher: AcceptPublish,
            render: RecordModels {
                models: std::sync::Arc::clone(&models),
            },
            clock: StepClock {
                mono: Instant::now(),
                wall: SystemTime::UNIX_EPOCH,
            },
            notify: QuietNotify,
            stop: OneTick(std::sync::atomic::AtomicU64::new(0)),
            log: QuietLog,
            roots: &roots,
            started: Instant::now(),
        });
        let models = models.lock().expect("models").clone();
        models.first().expect("one drawn frame").clone()
    }

    /// #170: `[tty] bar_segments` reaches the drawn model; solid without it.
    #[test]
    fn bar_segments_reach_the_frame() {
        let model = one_frame("solid-bars", "[llama]\nenabled = false\n");
        assert!(!model.bar_segments);
        let model = one_frame(
            "segment-bars",
            "[llama]\nenabled = false\n[tty]\nbar_segments = true\n",
        );
        assert!(model.bar_segments);
    }

    #[test]
    fn llama_disabled_draws_no_llama_at_once() {
        let model = one_frame("no-llama", "[llama]\nenabled = false\n");
        assert_eq!(model.state, WatchState::NoLlama);
        assert_eq!(model.model_name, "--");
        assert_eq!(model.slots_line, "--");
        assert_eq!(model.swap_line, "--");
        for seg in &model.health {
            let llama = matches!(
                seg.name.as_str(),
                "llama-swap" | "/running" | "/slots" | "metrics" | "activity"
            );
            if llama {
                assert_eq!(seg.status, HealthStatus::Absent, "{}", seg.name);
            }
        }
        let painted = grid_text(&layout::layout(&model, 200, 60));
        assert!(painted.contains("NO LLAMA"), "{painted}");
        assert!(painted.contains("[llama] enabled = false"), "{painted}");
        assert!(!painted.contains("AI DOWN"), "{painted}");
        assert!(!painted.contains("STARTING"), "{painted}");
        assert!(!painted.contains("unreachable"), "{painted}");
    }

    struct SecretFeed {
        sent: bool,
    }

    const SECRET_IN: &str = "SECRET-PROMPT-T45";
    const SECRET_OUT: &str = "SECRET-OUTPUT-T45";

    fn secret_cells(text: &str) -> Vec<Cell> {
        let mut cells = Vec::new();
        sanitize(text, &mut cells);
        cells
    }

    impl LlamaFeed for SecretFeed {
        fn poll(&mut self) -> Option<(LlamaView, crate::poller::LlamaDetail)> {
            if self.sent {
                return None;
            }
            self.sent = true;
            // Text as a text-on poller would hand it over. With show_text
            // off the loop must still drop it before the model.
            Some((
                LlamaView {
                    ai: AiState::Loaded,
                    models: Vec::new(),
                    decoded_total: None,
                    prompt_total: None,
                },
                crate::poller::LlamaDetail {
                    slots: vec![SlotView {
                        model: "m".to_owned(),
                        id: 0,
                        id_task: 1,
                        is_processing: true,
                        n_prompt_tokens: 9_000,
                        n_prompt_tokens_processed: 9_000,
                        n_decoded: 42,
                        n_ctx: Some(32_768),
                        ctx_prompt: Some(9_000),
                        prompt_cached: None,
                        ctx_used: None,
                        resets: Default::default(),
                        last_reset: None,
                        input: secret_cells(SECRET_IN),
                        output: secret_cells(SECRET_OUT),
                    }],
                    activity: Vec::new(),
                    gen_tps: Some(40.0),
                    prompt_tps: Some(0.0),
                    latencies: crate::poller::PollLatencies::default(),
                    prompt_cache: Vec::new(),
                    capture: None,
                    setup: Vec::new(),
                    engine_live: Vec::new(),
                    suspected_loads: Vec::new(),
                    series: Vec::new(),
                    requests: Vec::new(),
                    jobs: Vec::new(),
                    claims: Vec::new(),
                    gpu: None,
                    displaced: Vec::new(),
                },
            ))
        }
    }

    fn secret_frame(label: &str, toml: &str) -> TtyModel {
        let scratch = CpuScratch::new(label);
        let cfg = scratch.path().join("watch.toml");
        std::fs::write(&cfg, toml).expect("toml");
        let config = Config::load_validated(&cfg, 8).expect("config");
        let roots = Roots {
            proc: scratch.path().join("proc"),
            sys: scratch.path().to_path_buf(),
        };
        let models = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        run_loop(LoopInput {
            config: &config,
            feed: SecretFeed { sent: false },
            sampler: FixedCpu,
            publisher: AcceptPublish,
            render: RecordModels {
                models: std::sync::Arc::clone(&models),
            },
            clock: StepClock {
                mono: Instant::now(),
                wall: SystemTime::UNIX_EPOCH,
            },
            notify: QuietNotify,
            stop: OneTick(std::sync::atomic::AtomicU64::new(0)),
            log: QuietLog,
            roots: &roots,
            started: Instant::now(),
        });
        let models = models.lock().expect("models").clone();
        models.first().expect("one drawn frame").clone()
    }

    #[test]
    fn text_off_keeps_prompt_and_output_out_of_the_model_and_every_cell() {
        // Control: with text on the same feed does reach the frame.
        let on = secret_frame("text-on", "");
        assert!(on.show_text);
        assert!(
            format!("{on:?}").contains(SECRET_OUT),
            "control frame lost the text"
        );

        let off = secret_frame("text-off", "[tty]\nshow_text = false\n");
        assert!(!off.show_text);
        let dump = format!("{off:?}");
        assert!(!dump.contains("SECRET"), "llama text in the model: {dump}");
        assert!(off.in_lines.is_empty() && off.out_lines.is_empty());
        assert!(off.in_title.is_empty() && off.out_title.is_empty());
        assert_eq!(off.out_shown, 0);
        assert_eq!(off.replay_frame, None);
        // Busy state and ctx fill still come through.
        assert_eq!(off.slots.len(), 1);
        assert!(off.slots[0].generating);
        assert_eq!(off.slots[0].n_ctx, Some(32_768));
        assert_eq!(off.slots_line, "1/1 busy");
        for (cols, rows) in [(160, 48), (240, 67), (286, 60), (480, 135)] {
            let painted = grid_text(&layout::layout(&off, cols, rows));
            assert!(!painted.contains("SECRET"), "{cols}x{rows}\n{painted}");
            assert!(painted.contains("text off"), "{cols}x{rows}\n{painted}");
        }
    }

    #[test]
    fn llama_enabled_without_a_poll_is_still_starting() {
        let model = one_frame("llama-wait", "");
        assert_eq!(model.state, WatchState::Starting);
    }

    #[test]
    fn host_nproc_is_one_when_proc_stat_cannot_be_read() {
        let missing = std::env::temp_dir().join(format!(
            "kraken-lcd-missing-stat-{}-absent",
            std::process::id()
        ));
        assert!(!missing.exists());
        assert_eq!(host_nproc(&missing), 1);
        assert_eq!(cpu_cores(&missing), None);
    }

    struct CpuScratch(std::path::PathBuf);

    impl CpuScratch {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "kraken-lcd-{label}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .expect("clock")
                    .as_nanos()
            ));
            std::fs::create_dir_all(&path).expect("scratch");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for CpuScratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn stat_with_cpus(n: u32) -> String {
        let mut text = String::from("cpu  0 0 0 0 0 0 0 0\n");
        for index in 0..n {
            text.push_str(&format!("cpu{index} 10 0 0 10 0 0 0 0\n"));
        }
        text.push_str("intr 1 2 3\nctxt 4\n");
        text
    }

    struct IdleOnce {
        sent: bool,
    }

    impl LlamaFeed for IdleOnce {
        fn poll(&mut self) -> Option<(LlamaView, crate::poller::LlamaDetail)> {
            if self.sent {
                return None;
            }
            self.sent = true;
            Some((
                LlamaView {
                    ai: AiState::Idle,
                    models: Vec::new(),
                    decoded_total: None,
                    prompt_total: None,
                },
                crate::poller::LlamaDetail {
                    slots: Vec::new(),
                    activity: Vec::new(),
                    gen_tps: None,
                    prompt_tps: None,
                    latencies: crate::poller::PollLatencies::default(),
                    prompt_cache: Vec::new(),
                    capture: None,
                    setup: Vec::new(),
                    engine_live: Vec::new(),
                    suspected_loads: Vec::new(),
                    series: Vec::new(),
                    requests: Vec::new(),
                    jobs: Vec::new(),
                    claims: Vec::new(),
                    gpu: None,
                    displaced: Vec::new(),
                },
            ))
        }
    }

    #[test]
    fn header_uses_the_full_name_and_the_detail_line() {
        let mut sample = FixedCpu.sample(
            Instant::now(),
            SystemTime::UNIX_EPOCH,
            &LlamaView {
                ai: AiState::Loaded,
                models: Vec::new(),
                decoded_total: None,
                prompt_total: None,
            },
        );
        sample.snapshot.ai = AiState::Loaded;
        sample.snapshot.models = vec![llama_core::sample::ModelInfo {
            backend: None,
            name: "Ternary Bon…".to_owned(),
            state: "ready".to_owned(),
            full_name: Some("Ternary Bonsai 2 27B".to_owned()),
            detail: Some(llama_core::detail::ModelDetail {
                ctx: Some(262_144),
                kv_k: Some("q8_0".to_owned()),
                kv_v: Some("q8_0".to_owned()),
                quant: Some("PTQ1_0".to_owned()),
                ..llama_core::detail::ModelDetail::default()
            }),
        }];
        assert_eq!(
            model_name(&sample, WatchState::Ready),
            "Ternary Bonsai 2 27B"
        );
        assert_eq!(
            model_detail(&sample, WatchState::Ready),
            "llama.cpp",
            "#33: no backend reads as llama.cpp; #52: the settings are in SETUP"
        );
        assert_eq!(model_detail(&sample, WatchState::AiDown), "engine --");
        assert_eq!(model_detail(&sample, WatchState::Starting), "engine --");
        assert_eq!(model_detail(&sample, WatchState::NoLlama), "engine --");
        // An older snapshot: the canonical name and no detail.
        sample.snapshot.models[0].full_name = None;
        sample.snapshot.models[0].detail = None;
        assert_eq!(model_name(&sample, WatchState::Ready), "Ternary Bon…");
        assert_eq!(model_detail(&sample, WatchState::Ready), "llama.cpp");
        // #33: no model loaded.
        sample.snapshot.models.clear();
        assert_eq!(model_detail(&sample, WatchState::Ready), "engine --");
    }

    /// #52: SETUP shows the model generating now, else the one RECENT saw
    /// last, else the first; `+N` counts the others; nothing loaded or
    /// llama-swap down collapses it.
    #[test]
    fn setup_view_follows_the_busy_then_the_recent_model() {
        let rules = Rules::builtin();
        let llama = BackendInfo {
            kind: Backend::LlamaCpp,
            ..BackendInfo::default()
        };
        let mut sample = backend_sample(vec![
            served("Qwen 35B", "ready", Some(llama)),
            served("flash", "ready", Some(sglang(Some(0)))),
        ]);
        let setup = |key: &str, id: &str, cmd: &str| ModelSetup {
            key: key.to_owned(),
            id: id.to_owned(),
            name: String::new(),
            found: rules.extract(cmd),
            ..ModelSetup::default()
        };
        let mut detail = TickState::new(2, 6).detail;
        detail.setup = vec![
            setup("qwen-35b", "qwen-35b", "llama-server -c 8192"),
            setup(
                "flash",
                "flash",
                "python3 -m sglang.launch_server --context-length 204800",
            ),
        ];
        let shown = |sample: &WatchSample, detail: &LlamaDetail| {
            setup_view(sample, detail, WatchState::Ready, &rules)
                .map(|view| (view.id, view.more, view.rows[1].items[0].text.clone()))
        };
        // Nothing busy, no RECENT row: the first model.
        assert_eq!(
            shown(&sample, &detail),
            Some(("qwen-35b".to_owned(), 1, "8,192".to_owned()))
        );
        // RECENT's newest row names the second.
        let mut row = crate::activity::ActivityRow {
            id: 1,
            seq: 1,
            time: String::new(),
            source: String::new(),
            model: "flash".to_owned(),
            input_tokens: None,
            cached_tokens: None,
            output_tokens: None,
            prompt_tps: None,
            gen_tps: None,
            engine_prompt_tps: None,
            engine_gen_tps: None,
            duration_ms: None,
            status: None,
            draft_tokens: None,
            draft_accepted: None,
            captured: false,
            engine: None,
        };
        detail.activity = vec![row.clone()];
        assert_eq!(
            shown(&sample, &detail).map(|v| v.0),
            Some("flash".to_owned())
        );
        // A busy model wins over RECENT.
        row.model = "qwen-35b".to_owned();
        detail.activity = vec![row];
        sample.snapshot.models[1].backend = Some(sglang(Some(1)));
        assert_eq!(
            shown(&sample, &detail).map(|v| v.0),
            Some("flash".to_owned())
        );
        // A busy llama.cpp slot of the first model wins too.
        sample.snapshot.models[1].backend = Some(sglang(Some(0)));
        detail.activity.clear();
        detail.slots = vec![SlotView {
            model: "Qwen 35B".to_owned(),
            id: 0,
            id_task: 1,
            is_processing: true,
            n_prompt_tokens: 0,
            n_prompt_tokens_processed: 0,
            n_decoded: 0,
            n_ctx: None,
            ctx_prompt: None,
            prompt_cached: None,
            ctx_used: None,
            resets: Default::default(),
            last_reset: None,
            input: Vec::new(),
            output: Vec::new(),
        }];
        assert_eq!(
            shown(&sample, &detail).map(|v| v.0),
            Some("qwen-35b".to_owned())
        );
        // A stale setup list (another length) falls back to the snapshot.
        detail.setup.pop();
        let view = setup_view(&sample, &detail, WatchState::Ready, &rules).expect("view");
        assert_eq!(view.id, "Qwen 35B");
        assert_eq!(view.rows[0].items[0].text, "llama.cpp");
        // Collapsed without a model or with llama-swap down.
        assert!(setup_view(&sample, &detail, WatchState::AiDown, &rules).is_none());
        sample.snapshot.models.clear();
        assert!(setup_view(&sample, &detail, WatchState::Ready, &rules).is_none());
    }

    fn backend_sample(models: Vec<ModelInfo>) -> WatchSample {
        let mut sample = FixedCpu.sample(
            Instant::now(),
            SystemTime::UNIX_EPOCH,
            &LlamaView {
                ai: AiState::Loaded,
                models: Vec::new(),
                decoded_total: None,
                prompt_total: None,
            },
        );
        sample.snapshot.ai = AiState::Loaded;
        sample.snapshot.models = models;
        sample
    }

    fn served(name: &str, state: &str, backend: Option<BackendInfo>) -> ModelInfo {
        ModelInfo {
            name: name.to_owned(),
            state: state.to_owned(),
            full_name: None,
            detail: None,
            backend,
        }
    }

    fn sglang(running: Option<u16>) -> BackendInfo {
        BackendInfo {
            kind: Backend::SgLang,
            max_running: Some(4),
            running,
            queued: Some(0),
            kv_permille: Some(372),
            hit_permille: None,
            engine: Default::default(),
            kv: None,
        }
    }

    /// #180: tqLLM's engine line has no KV item. Its SLOTS lines are
    /// `engine.slots` long: each running request on the line the book gave
    /// it, its progress against its prompt with the cached part apart, its
    /// context against `context_per_slot`; a free line is idle.
    #[test]
    fn tqllm_engine_line_and_slot_lines() {
        let info = BackendInfo {
            kind: Backend::Tqllm,
            max_running: Some(2),
            running: Some(2),
            queued: Some(1),
            kv_permille: None,
            hit_permille: None,
            engine: llama_core::backend::EngineStats {
                spec_permille: Some(660),
                prefill_tps_tenths: Some(3099),
                decode_tps_tenths: Some(842),
                ..Default::default()
            },
            kv: None,
        };
        let sample = backend_sample(vec![served("tq", "ready", Some(info))]);
        assert_eq!(
            backend_lines(&sample, WatchState::Generating),
            vec!["tqllm  running 2/2 · queued 1 · spec 66 % · prefill 310/s · decode 84.2/s"]
        );
        let body = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/llama/tqllm-snapshot.json"
        ))
        .expect("tqllm fixture");
        let mut snap = crate::metrics::parse_tqllm(&body).expect("snapshot");
        snap.slots = Some(3);
        let now = SystemTime::now();
        let mut book = crate::tqllm::TqllmBook::default();
        let observed = book.observe("tq", &snap, Instant::now(), now, now);
        let entry = ModelEngineLive {
            model: "tq".to_owned(),
            live: observed.facts.live.expect("live"),
        };
        let rows = tqllm_slot_rows(std::slice::from_ref(&entry), &CtxBook::new(6));
        let got: Vec<_> = rows
            .iter()
            .map(|slot| {
                (
                    slot.id,
                    slot.generating,
                    (slot.done, slot.cached, slot.total, slot.decoded),
                    (slot.ctx_prompt, slot.n_ctx),
                    (slot.done_known, slot.open),
                )
            })
            .collect();
        assert_eq!(
            got,
            [
                // Request 4 decoding: 288 read beyond 512 cached of 800.
                (
                    0,
                    true,
                    (288, 512, 800, 150),
                    (Some(800), Some(32_768)),
                    (true, false)
                ),
                // Request 5 reading: 2,048 of 6,000 read, 1,024 of them cached.
                (
                    1,
                    true,
                    (1_024, 1_024, 6_000, 0),
                    (Some(6_000), Some(32_768)),
                    (true, false)
                ),
                (
                    2,
                    false,
                    (0, 0, 0, 0),
                    (Some(0), Some(32_768)),
                    (true, false)
                ),
            ]
        );
        // The sparkline views: each line's context, 0 when idle.
        let used: Vec<_> = tqllm_views(std::slice::from_ref(&entry))
            .iter()
            .map(|view| (view.id, view.is_processing, view.ctx_used))
            .collect();
        assert_eq!(
            used,
            [
                (0, true, Some(950)),
                (1, true, Some(6_000)),
                (2, false, Some(0))
            ]
        );
    }

    fn frame_ctx(mem_total_bytes: Option<u64>, llama_enabled: bool) -> FrameCtx {
        FrameCtx {
            output_cap: 100,
            input_cap: 100,
            gen_ceiling: 250.0,
            prompt_ceiling: 1500.0,
            chart_bucket_s: 2,
            chart_glyphs: ChartGlyphs::default(),
            bar_segments: false,
            ctx_history_h: 6,
            started: Instant::now(),
            host: "box".to_owned(),
            cpu_cores: None,
            mem_total_bytes,
            llama_enabled,
            show_text: false,
            setup: Rules::builtin(),
            temps_look: crate::sources::temps::TempLook::default(),
            jobs_stale_s: 180,
            disk_ceiling_bps: 7e9,
            net_ceiling_bps: None,
        }
    }

    /// #11: what the tty frame shows goes on the wire for llama-metrics.
    #[test]
    fn publish_extras_mirror_the_frame() {
        let mut sample = backend_sample(vec![served("m", "ready", None)]);
        sample.snapshot.mem_pct = Some(25.0);
        sample.gpu = crate::collector::GpuExtra {
            vram_used: Some(1 << 30),
            vram_total: Some(4 << 30),
            power_mw: Some(312_500),
            power_limit_mw: Some(600_000),
        };
        sample.cpu_w = Some(90.5);
        sample.fans = Some(crate::sources::fans::FanPanel {
            chip: "nct6798".to_owned(),
            present: true,
            fans: vec![crate::sources::fans::FanReading {
                chip: "nct6798".to_owned(),
                channel: 2,
                label: "CPU".to_owned(),
                rpm: Some(1100),
                pwm: Some(128),
                mode: Some(5),
            }],
        });
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        let mut other = slot(0, 1, false, "", "");
        other.model = "n".to_owned();
        tick.detail.slots = vec![
            slot(0, 5, true, "", ""),
            slot(1, 4, false, "", ""),
            slot(2, 3, true, "", ""),
            other,
        ];
        tick.detail.latencies.running = Some(Duration::from_millis(3));
        tick.detail.latencies.slots = Some(Duration::from_millis(7));
        let extras = publish_extras(&sample, &tick, &frame_ctx(Some(8 << 30), true));
        assert_eq!(extras.gpu_w, Some(312.5));
        assert_eq!(extras.gpu_limit_w, Some(600.0));
        assert_eq!(extras.cpu_w, Some(90.5));
        assert_eq!(
            (extras.vram_used, extras.vram_total),
            (Some(1 << 30), Some(4 << 30))
        );
        assert_eq!(
            (extras.mem_used, extras.mem_total),
            (Some(2 << 30), Some(8 << 30))
        );
        assert_eq!(
            extras.slots,
            vec![("m".to_owned(), 2, 3), ("n".to_owned(), 0, 1)]
        );
        assert_eq!(
            extras.fans,
            vec![(
                "nct6798".to_owned(),
                2,
                "CPU".to_owned(),
                Some(1100),
                Some(128)
            )]
        );
        let sources = extras.sources.expect("sources");
        let names: Vec<(&str, Option<bool>)> = sources
            .entries()
            .iter()
            .map(|(name, source)| (*name, source.map(|s| s.up)))
            .collect();
        assert_eq!(
            names,
            [
                ("llama-swap", Some(true)),
                ("running", Some(true)),
                ("slots", Some(true)),
                ("metrics", Some(true)),
                ("activity", Some(true)),
                ("gpu", Some(true)),
                ("hwmon", Some(true)),
                ("proc", Some(true)),
            ]
        );
        assert_eq!(sources.llama_swap.and_then(|s| s.latency_s), Some(0.003));
        assert_eq!(sources.running.and_then(|s| s.latency_s), Some(0.003));
        assert_eq!(sources.slots.and_then(|s| s.latency_s), Some(0.007));
        assert_eq!(sources.gpu.and_then(|s| s.latency_s), None);
    }

    /// #163: a distinct number in every host field, so a swap between two
    /// of them shows.
    fn host_io_sample() -> WatchSample {
        let mut sample = backend_sample(vec![served("m", "ready", None)]);
        sample.host_io = crate::sources::hostio::HostIo {
            disk_read_bps: Some(1.0e9),
            disk_write_bps: Some(2.0e6),
            net_rx_bps: Some(3.0e6),
            net_tx_bps: Some(4.0e3),
            net_ceiling_bps: Some(312_500_000.0),
            swap_used_bytes: Some(1 << 30),
            swap_total_bytes: Some(16 << 30),
            cpu_psi: Some(5.0),
            io_psi: Some(6.0),
            mem_psi: Some(7.0),
            ..Default::default()
        };
        sample
    }

    fn idle_view() -> LlamaView {
        LlamaView {
            ai: AiState::Loaded,
            models: Vec::new(),
            decoded_total: None,
            prompt_total: None,
        }
    }

    /// #163: each host number reaches the frame, the extras and the wire
    /// under its own name.
    #[test]
    fn host_io_reaches_the_frame_and_the_wire_field_by_field() {
        let sample = host_io_sample();
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        let ctx = frame_ctx(None, true);
        let model = tty_model(
            &sample,
            &mut tick,
            Instant::now(),
            UNIX_EPOCH,
            Zone::utc(),
            &ctx,
        );
        assert_eq!(
            (model.disk_read_bps, model.disk_write_bps),
            (Some(1.0e9), Some(2.0e6))
        );
        assert_eq!(
            (model.net_rx_bps, model.net_tx_bps),
            (Some(3.0e6), Some(4.0e3))
        );
        assert_eq!(model.net_ceiling_bps, Some(312_500_000.0));
        assert_eq!(
            (model.swap_used_gb, model.swap_total_gb),
            (Some(1.0), Some(16.0))
        );
        assert_eq!((model.io_psi, model.mem_psi), (Some(6.0), Some(7.0)));

        let extras = publish_extras(&sample, &tick, &ctx);
        assert_eq!(
            (extras.io_psi, extras.mem_psi, extras.cpu_psi),
            (Some(6.0), Some(7.0), Some(5.0))
        );
        let host = crate::publish::build(&sample.snapshot, &idle_view(), &extras, 1, 1).host;
        assert_eq!(
            (host.disk_read_bps, host.disk_write_bps),
            (Some(1.0e9), Some(2.0e6))
        );
        assert_eq!(
            (host.net_rx_bps, host.net_tx_bps),
            (Some(3.0e6), Some(4.0e3))
        );
        assert_eq!(
            (host.io_psi_pct, host.mem_psi_pct, host.cpu_psi_pct),
            (Some(6.0), Some(7.0), Some(5.0))
        );
        assert_eq!(
            (host.swap_used_bytes, host.swap_total_bytes),
            (Some(1 << 30), Some(16 << 30))
        );
    }

    /// An absent host reading is absent on the frame and the wire, never 0.
    #[test]
    fn absent_host_io_is_absent_on_the_frame_and_the_wire() {
        let mut sample = host_io_sample();
        sample.host_io = crate::sources::hostio::HostIo::default();
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        let ctx = frame_ctx(None, true);
        let model = tty_model(
            &sample,
            &mut tick,
            Instant::now(),
            UNIX_EPOCH,
            Zone::utc(),
            &ctx,
        );
        assert_eq!(
            (
                model.disk_read_bps,
                model.disk_write_bps,
                model.net_rx_bps,
                model.net_tx_bps,
                model.net_ceiling_bps,
            ),
            (None, None, None, None, None)
        );
        assert_eq!((model.swap_used_gb, model.swap_total_gb), (None, None));
        assert_eq!((model.io_psi, model.mem_psi), (None, None));
        let extras = publish_extras(&sample, &tick, &ctx);
        let host = crate::publish::build(&sample.snapshot, &idle_view(), &extras, 1, 1).host;
        assert_eq!(
            (
                host.disk_read_bps,
                host.disk_write_bps,
                host.net_rx_bps,
                host.net_tx_bps,
            ),
            (None, None, None, None)
        );
        assert_eq!(
            (host.io_psi_pct, host.mem_psi_pct, host.cpu_psi_pct),
            (None, None, None)
        );
        assert_eq!((host.swap_used_bytes, host.swap_total_bytes), (None, None));
    }

    /// `[tty] net_ceiling_bps` wins over the sampled link speed; unset, the
    /// link speed is used. `[tty] disk_ceiling_bps` reaches the frame.
    #[test]
    fn configured_ceilings_win_over_the_sampled_link_speed() {
        let sample = host_io_sample();
        let mut tick = TickState::new(2, 6);
        let mut ctx = frame_ctx(None, true);
        ctx.net_ceiling_bps = Some(5.0e7);
        ctx.disk_ceiling_bps = 2.0e9;
        let model = tty_model(
            &sample,
            &mut tick,
            Instant::now(),
            UNIX_EPOCH,
            Zone::utc(),
            &ctx,
        );
        assert_eq!(model.net_ceiling_bps, Some(5.0e7));
        assert_eq!(model.disk_ceiling_bps, 2.0e9);
        ctx.net_ceiling_bps = None;
        let model = tty_model(
            &sample,
            &mut tick,
            Instant::now(),
            UNIX_EPOCH,
            Zone::utc(),
            &ctx,
        );
        assert_eq!(model.net_ceiling_bps, Some(312_500_000.0));

        let model = one_frame(
            "ceilings",
            "[tty]\ndisk_ceiling_bps = 2000000000\nnet_ceiling_bps = 50000000\n",
        );
        assert_eq!(model.disk_ceiling_bps, 2.0e9);
        assert_eq!(model.net_ceiling_bps, Some(5.0e7));
    }

    /// #163: the disk, net and psi health words each read their own
    /// source.
    #[test]
    fn host_health_words_follow_their_own_source() {
        use crate::sources::hostio::SourceHealth;
        let mut sample = host_io_sample();
        sample.host_io.disk = SourceHealth::Down;
        sample.host_io.net = SourceHealth::Ok;
        sample.host_io.psi = SourceHealth::Absent;
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        let ctx = frame_ctx(None, true);
        let model = tty_model(
            &sample,
            &mut tick,
            Instant::now(),
            UNIX_EPOCH,
            Zone::utc(),
            &ctx,
        );
        let word = |name: &str| {
            model
                .health
                .iter()
                .find(|seg| seg.name == name)
                .map(|seg| &seg.status)
        };
        assert_eq!(word("disk"), Some(&HealthStatus::Down));
        assert_eq!(word("net"), Some(&HealthStatus::Ok));
        assert_eq!(word("psi"), Some(&HealthStatus::Absent));
    }

    #[test]
    fn wire_sources_leave_out_what_the_health_line_does_not_poll() {
        let mut sample = backend_sample(Vec::new());
        sample.snapshot.errors.insert(SourceId::Gpu);
        let mut tick = TickState::new(2, 6);
        // Starting: every segment pending, no sources at all.
        let extras = publish_extras(&sample, &tick, &frame_ctx(None, true));
        assert_eq!(extras.sources, None);
        assert_eq!(extras.mem_used, None);
        // llama-swap down: it and /running are down, its taps absent.
        tick.heard = true;
        sample.snapshot.ai = AiState::Down;
        let sources = publish_extras(&sample, &tick, &frame_ctx(None, true))
            .sources
            .expect("sources");
        assert_eq!(sources.llama_swap.map(|s| s.up), Some(false));
        assert_eq!(sources.running.map(|s| s.up), Some(false));
        assert_eq!(sources.slots, None);
        assert_eq!(sources.metrics, None);
        assert_eq!(sources.activity, None);
        assert_eq!(sources.gpu.map(|s| s.up), Some(false));
        assert_eq!(sources.proc.map(|s| s.up), Some(true));
        // [llama] off: the llama taps are absent, the host ones stay.
        let sources = publish_extras(&sample, &tick, &frame_ctx(None, false))
            .sources
            .expect("sources");
        assert_eq!(sources.llama_swap, None);
        assert_eq!(sources.hwmon.map(|s| s.up), Some(true));
    }

    #[test]
    fn non_llamacpp_backend_leads_the_detail_and_gets_a_slots_line() {
        let mut flash = served("flash", "ready", Some(sglang(Some(1))));
        flash.detail = Some(llama_core::detail::ModelDetail {
            ctx: Some(204_800),
            kv_k: Some("fp8_e4m3".to_owned()),
            kv_v: Some("fp8_e4m3".to_owned()),
            quant: Some("exl3".to_owned()),
            ..llama_core::detail::ModelDetail::default()
        });
        let tabby = served(
            "tabby",
            "ready",
            Some(BackendInfo {
                kind: Backend::OpenAi,
                ..BackendInfo::default()
            }),
        );
        let sample = backend_sample(vec![flash, tabby.clone()]);
        assert_eq!(
            model_detail(&sample, WatchState::Ready),
            "SGLang",
            "#52: the header keeps only the engine"
        );
        assert_eq!(
            backend_lines(&sample, WatchState::Ready),
            vec![
                "sglang  running 1/4 · queued 0 · KV 37 %".to_owned(),
                "openai  running -- · queued -- · KV --".to_owned(),
            ]
        );
        assert!(backend_lines(&sample, WatchState::AiDown).is_empty());
        assert!(first_without_slots(&sample, WatchState::Ready));
        assert_eq!(
            watch_state(&sample, &TickState::new(2, 6).detail, true, true),
            WatchState::Generating,
            "a running SGLang request is generating"
        );
        let idle = backend_sample(vec![served("flash", "ready", Some(sglang(Some(0))))]);
        assert_eq!(
            watch_state(&idle, &TickState::new(2, 6).detail, true, true),
            WatchState::Ready
        );
        let sample = backend_sample(vec![tabby]);
        assert_eq!(
            model_detail(&sample, WatchState::Ready),
            "OpenAI-compatible"
        );
        // llama.cpp and an older snapshot: llama.cpp leads (#33), no line.
        let llama = BackendInfo {
            kind: Backend::LlamaCpp,
            ..sglang(Some(1))
        };
        let sample = backend_sample(vec![
            served("q", "ready", Some(llama)),
            served("o", "ready", None),
        ]);
        assert_eq!(model_detail(&sample, WatchState::Ready), "llama.cpp");
        assert!(backend_lines(&sample, WatchState::Ready).is_empty());
        assert!(!first_without_slots(&sample, WatchState::Ready));
    }

    /// #31: a vLLM model found by its metrics: its cache facts close the
    /// tuning line and its engine numbers follow the gauges.
    /// #79: the KV item on each engine line, and a llama.cpp line after
    /// the metrics engines' lines.
    #[test]
    fn kv_items_on_the_engine_lines() {
        use llama_core::backend::KvUsage;
        let info = |kind: Backend, running: Option<u16>, kv: KvUsage| {
            Some(BackendInfo {
                kind,
                running,
                queued: Some(0),
                kv_permille: Some(999),
                kv: Some(kv),
                ..BackendInfo::default()
            })
        };
        let llama = served(
            "qwen",
            "ready",
            info(
                Backend::LlamaCpp,
                None,
                KvUsage {
                    used: Some(91_204),
                    capacity: Some(262_144),
                    sessions: Some(1),
                    unified: Some(true),
                    ..KvUsage::default()
                },
            ),
        );
        let sglang = served(
            "flash",
            "ready",
            info(
                Backend::SgLang,
                Some(2),
                KvUsage {
                    used: Some(54_272),
                    capacity: Some(204_800),
                    cached: Some(31_200),
                    sessions: Some(2),
                    ..KvUsage::default()
                },
            ),
        );
        let vllm = served(
            "q",
            "ready",
            info(
                Backend::Vllm,
                Some(1),
                KvUsage {
                    used: Some(77_319),
                    capacity: Some(187_440),
                    sessions: Some(1),
                    approx: true,
                    ..KvUsage::default()
                },
            ),
        );
        let strata = served(
            "b",
            "ready",
            info(
                Backend::Strata,
                Some(1),
                KvUsage {
                    used: Some(6960),
                    capacity: Some(524_288),
                    sessions: Some(2),
                    ..KvUsage::default()
                },
            ),
        );
        let sample = backend_sample(vec![llama, sglang, vllm, strata]);
        assert_eq!(
            backend_lines(&sample, WatchState::Ready),
            vec![
                "sglang  running 2 · queued 0 · KV 54k/204k tok 27 % · +31k cached".to_owned(),
                "vllm  running 1 · queued 0 · KV \u{2248}77k/187k tok 41 %".to_owned(),
                "strata  running 1 · queued 0 · KV 6k/524k tok 1 % · 2 sessions".to_owned(),
                "llamacpp  KV 91k/262k tok shared 35 % · 1 session".to_owned(),
            ]
        );
        // Per-slot llama.cpp says nothing of sharing; an assumed pool asks.
        let kv = |unified, unified_assumed| KvUsage {
            used: Some(38_000),
            capacity: Some(65_536),
            sessions: Some(2),
            unified: Some(unified),
            unified_assumed,
            ..KvUsage::default()
        };
        assert_eq!(
            kv_text(&kv(false, false), true).as_deref(),
            Some("KV 38k/65k tok 58 % · 2 sessions")
        );
        assert_eq!(
            kv_text(&kv(true, true), true).as_deref(),
            Some("KV 38k/65k tok shared? 58 % · 2 sessions")
        );
        let unknown = KvUsage {
            capacity: Some(65_536),
            ..KvUsage::default()
        };
        assert_eq!(kv_text(&unknown, false).as_deref(), Some("KV --/65k tok"));
        assert_eq!(kv_text(&KvUsage::default(), true), None);
    }

    #[test]
    fn vllm_engine_numbers_on_the_backend_line() {
        use llama_core::backend::EngineStats;
        let engine = EngineStats {
            spec_permille: Some(781),
            spec_len_centi: Some(294),
            preemptions: Some(3),
            sleeping: Some(false),
            ttft_us: Some(420_400),
            itl_us: Some(31_000),
            e2e_us: Some(12_460_000),
            prefill_tps_tenths: Some(21_342),
            decode_tps_tenths: Some(412),
            ..EngineStats::default()
        };
        let mut qwen = served(
            "qwen3.8-27b…",
            "ready",
            Some(BackendInfo {
                kind: Backend::Vllm,
                running: Some(1),
                queued: Some(0),
                kv_permille: Some(413),
                hit_permille: Some(750),
                engine,
                ..BackendInfo::default()
            }),
        );
        qwen.detail = Some(llama_core::detail::ModelDetail {
            kv_k: Some("fp8_e4m3".to_owned()),
            kv_v: Some("fp8_e4m3".to_owned()),
            kv_block: Some(16),
            prefix_cache: Some(true),
            ..llama_core::detail::ModelDetail::default()
        });
        let sample = backend_sample(vec![qwen.clone()]);
        assert_eq!(
            model_detail(&sample, WatchState::Ready),
            "vLLM",
            "#52: kv, block and prefix moved to SETUP"
        );
        assert_eq!(
            backend_lines(&sample, WatchState::Ready),
            vec![
                "vllm  running 1 · queued 0 · KV 41 % · hit 75 % · spec 78 % · 2.9/step · ttft 420ms · itl  31ms · e2e 12.5s · preempt 3 · prefill 2,134/s · decode 41.2/s"
                    .to_owned()
            ]
        );
        // Asleep, no preemptions yet, no spec: the line says so and stops short.
        let info = qwen.backend.as_mut().expect("backend");
        info.engine = EngineStats {
            sleeping: Some(true),
            preemptions: Some(0),
            e2e_us: Some(250_000_000),
            ..EngineStats::default()
        };
        let sample = backend_sample(vec![qwen]);
        assert_eq!(
            backend_lines(&sample, WatchState::Ready),
            vec![
                "vllm  sleeping · running 1 · queued 0 · KV 41 % · hit 75 % · e2e  250s".to_owned()
            ]
        );
    }

    #[test]
    fn latency_text_units() {
        // #167: five cells, so a number's growth moves nothing after it.
        assert_eq!(latency_text(0), "  0ms");
        assert_eq!(latency_text(9_499), "  9ms");
        assert_eq!(latency_text(9_500), " 10ms");
        assert_eq!(latency_text(31_499), " 31ms");
        assert_eq!(latency_text(99_499), " 99ms");
        assert_eq!(latency_text(99_500), "100ms");
        assert_eq!(latency_text(999_499), "999ms");
        assert_eq!(latency_text(999_500), " 1.0s");
        assert_eq!(latency_text(12_460_000), "12.5s");
        assert_eq!(latency_text(99_940_000), "99.9s");
        assert_eq!(latency_text(100_000_000), " 100s");
        assert_eq!(latency_text(3_600_000_000), "3600s");
        for us in (0..4_000_000_000u32).step_by(7_919_111) {
            assert_eq!(latency_text(us).chars().count(), LATENCY_CELLS, "{us}");
        }
        // The health line's notes are the same text.
        assert_eq!(ms(Some(Duration::from_micros(9_400))), "  9ms");
        assert_eq!(ms(Some(Duration::from_millis(312))), "312ms");
        assert_eq!(ms(None), "");
    }

    /// #167: the health row as the watcher builds it, from 9 to 10 ms and
    /// 99 to 100 ms on every timed source: only the digits change.
    #[test]
    fn health_latencies_keep_their_width_on_the_row() {
        let wall = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let row = |ms: u64| {
            let (sample, mut tick, ctx) = claim_tick(Vec::new(), "m");
            let latency = Some(Duration::from_millis(ms));
            tick.detail.latencies.running = latency;
            tick.detail.latencies.slots = latency;
            tick.detail.latencies.metrics = latency;
            tick.detail.latencies.activity = latency;
            let model = tty_model(&sample, &mut tick, Instant::now(), wall, Zone::utc(), &ctx);
            let grid = layout::layout(&model, 213, 60);
            (0..grid.cols())
                .map(|c| grid.get(c, 58).unwrap().ch)
                .collect::<String>()
        };
        for (a, b) in [(9, 10), (99, 100)] {
            let (ra, rb) = (row(a), row(b));
            assert!(ra.contains(&format!("llama-swap ok {:>3}ms", a)), "{ra}");
            let same: Vec<bool> = ra.chars().zip(rb.chars()).map(|(x, y)| x == y).collect();
            for (col, (x, y)) in ra.chars().zip(rb.chars()).enumerate() {
                if x != y {
                    assert!(
                        x.is_ascii_digit() || y.is_ascii_digit(),
                        "{a}->{b} col {col}\n{ra}\n{rb}"
                    );
                }
            }
            assert_eq!(ra.chars().count(), rb.chars().count());
            // Four sources, at most three cells each.
            assert!(same.iter().filter(|s| !**s).count() <= 12, "{ra}\n{rb}");
        }
    }

    /// #167: `/slots` takes the room of `idle` when it says `ok`, so nothing
    /// after it moves when a slot starts or stops.
    #[test]
    fn slots_ok_and_idle_take_the_same_room() {
        let wall = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let (sample, mut tick, ctx) = claim_tick(Vec::new(), "m");
        tick.detail.latencies.slots = Some(Duration::from_millis(9));
        let idle = tty_model(&sample, &mut tick, Instant::now(), wall, Zone::utc(), &ctx);
        let slots = |model: &TtyModel| {
            model
                .health
                .iter()
                .position(|seg| seg.name == "/slots")
                .expect("/slots")
        };
        assert_eq!(idle.health[slots(&idle)].status, HealthStatus::Idle);
        let mut ok = idle.clone();
        let at = slots(&ok);
        ok.health[at].status = HealthStatus::Ok;
        let row = |model: &TtyModel| {
            let grid = layout::layout(model, 213, 60);
            (0..grid.cols())
                .map(|c| grid.get(c, 58).unwrap().ch)
                .collect::<String>()
        };
        let (ri, ro) = (row(&idle), row(&ok));
        let after = ri.find("/slots idle").expect("idle") + "/slots idle".len();
        assert_eq!(&ri[after..], &ro[after..], "\n{ri}\n{ro}");
    }

    #[test]
    fn strata_leads_the_detail_and_its_line_has_no_kv() {
        let mut flash = served(
            "flash",
            "ready",
            Some(BackendInfo {
                kind: Backend::Strata,
                max_running: Some(1),
                running: Some(1),
                queued: Some(0),
                ..BackendInfo::default()
            }),
        );
        flash.detail = Some(llama_core::detail::ModelDetail {
            ctx: Some(262_144),
            kv_k: Some("q8".to_owned()),
            kv_v: Some("q8".to_owned()),
            ..llama_core::detail::ModelDetail::default()
        });
        let sample = backend_sample(vec![flash]);
        assert_eq!(model_detail(&sample, WatchState::Ready), "Strata");
        assert_eq!(
            backend_lines(&sample, WatchState::Ready),
            vec!["strata  running 1/1 · queued 0".to_owned()]
        );
        assert!(first_without_slots(&sample, WatchState::Ready));
        assert_eq!(
            watch_state(&sample, &TickState::new(2, 6).detail, true, true),
            WatchState::Generating
        );
    }

    /// #54: Strata's engine numbers, then a second line with its live
    /// phase and progress: the prompt read while reading, the tokens and
    /// tok/s while generating, nothing while idle.
    /// The rows of one frame as text.
    fn frame_rows(grid: &crate::tty::grid::Grid) -> Vec<String> {
        (0..grid.rows())
            .map(|row| {
                (0..grid.cols())
                    .map(|col| grid.get(col, row).map_or(' ', |cell| cell.ch))
                    .collect()
            })
            .collect()
    }

    /// #81: Strata in parallel mode, one slot decoding and one reading the
    /// newest request: two in-flight RECENT rows, and one SLOTS line per
    /// slot with what it holds.
    #[test]
    fn strata_parallel_slots_lead_recent_and_fill_slots() {
        let detail = llama_core::detail::ModelDetail {
            ctx: Some(262_144),
            ..llama_core::detail::ModelDetail::default()
        };
        let mut flash = served(
            "flash",
            "ready",
            Some(BackendInfo {
                kind: Backend::Strata,
                max_running: Some(2),
                running: Some(2),
                queued: Some(0),
                ..BackendInfo::default()
            }),
        );
        flash.detail = Some(detail);
        let sample = backend_sample(vec![flash.clone()]);
        let doc = serde_json::json!({
            "engine": {"context": 262_144, "batch_slots": 2},
            "live": {
                "state": "reading", "parallel": 2, "running": 2, "waiting": 0,
                "prompt_tokens": 5_965, "prompt_read": 1_024, "prompt_total": 5_965,
                "prefill_tok_s_mean": 900.0,
                "slots": [
                    {"slot": 0, "state": "decoding", "prompt_tokens": 7_853,
                     "generated": 51, "elapsed_s": 1.7, "tok_s": 34.4},
                    {"slot": 1, "state": "reading", "prompt_tokens": 5_965,
                     "generated": 0, "elapsed_s": 0.4},
                ],
            },
        });
        let (_, facts) = crate::metrics::parse_strata(&doc.to_string());
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        tick.llama.ai = AiState::Loaded;
        tick.llama.models = vec![flash];
        tick.detail.engine_live = vec![ModelEngineLive {
            model: "flash".to_owned(),
            live: facts.live.expect("live"),
        }];
        tick.detail.setup = vec![crate::poller::ModelSetup {
            key: "flash".to_owned(),
            ..crate::poller::ModelSetup::default()
        }];
        let mono = Instant::now();
        let wall = UNIX_EPOCH + Duration::from_secs(1_791_417_180);
        note_poll(&mut tick, mono, wall);
        let ctx = frame_ctx(None, true);
        let model = tty_model(&sample, &mut tick, mono, wall, Zone::utc(), &ctx);
        assert_eq!(model.slots.len(), 2);
        assert_eq!(
            (
                model.slots[0].done,
                model.slots[0].total,
                model.slots[0].decoded
            ),
            (7_853, 7_853, 51)
        );
        assert_eq!(
            (
                model.slots[1].done,
                model.slots[1].total,
                model.slots[1].ctx_prompt
            ),
            (1_024, 5_965, Some(5_965))
        );
        let flying: Vec<_> = model
            .requests
            .iter()
            .filter(|r| r.inflight.is_some())
            .collect();
        assert_eq!(flying.len(), 2);
        assert_eq!(
            (flying[0].input_tok, flying[0].inflight.map(|f| f.target)),
            (5_965, Some(true))
        );
        assert_eq!((flying[1].output_tok, flying[1].dur.as_str()), (51, "1.7s"));
        let rows = frame_rows(&layout::layout(&model, 192, 60));
        let at = rows
            .iter()
            .position(|r| r.contains("SLOTS"))
            .expect("SLOTS");
        let recent = rows
            .iter()
            .position(|r| r.contains("RECENT"))
            .expect("RECENT");
        assert!(rows[at + 1].contains("s0 gen") && rows[at + 1].contains("7,853/7,853"));
        assert!(rows[at + 2].contains("s1 gen") && rows[at + 2].contains("1,024/5,965"));
        assert!(rows[recent + 1].starts_with('>') && rows[recent + 1].trim_end().ends_with("pp"));
        assert!(rows[recent + 2].starts_with('>') && rows[recent + 2].trim_end().ends_with("gen"));
    }

    /// #80: vLLM and SGLang requests from llama-swap's event stream lead
    /// RECENT: exact numbers, a shared count marked `≈`, SGLang's held KV
    /// as the bar with `--` columns, and a request with no engine numbers
    /// yet showing llama-swap's bytes.
    #[test]
    fn vllm_and_sglang_requests_lead_recent() {
        use crate::live::LiveTokens;
        use crate::poller::LiveRequest;
        let detail = |ctx: u32| llama_core::detail::ModelDetail {
            ctx: Some(ctx),
            ..llama_core::detail::ModelDetail::default()
        };
        let backend = |kind| BackendInfo {
            kind,
            running: Some(1),
            queued: Some(0),
            ..BackendInfo::default()
        };
        let mut vllm = served("qwen-vllm", "ready", Some(backend(Backend::Vllm)));
        vllm.detail = Some(detail(131_072));
        let mut sg = served("bonsai-sg", "ready", Some(backend(Backend::SgLang)));
        sg.detail = Some(detail(65_536));
        let sample = backend_sample(vec![vllm.clone(), sg.clone()]);
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        tick.llama.ai = AiState::Loaded;
        tick.llama.models = vec![vllm, sg];
        tick.detail.setup = ["qwen-vllm", "bonsai-sg"]
            .iter()
            .map(|key| crate::poller::ModelSetup {
                key: (*key).to_owned(),
                ..crate::poller::ModelSetup::default()
            })
            .collect();
        let mono = Instant::now();
        let wall = UNIX_EPOCH + Duration::from_secs(1_791_417_180);
        let ago = |ms: u64| mono - Duration::from_millis(ms);
        let req = |model: &str, id: &str, engine, ms: u64, bytes, tokens| LiveRequest {
            model: model.to_owned(),
            id: id.to_owned(),
            engine,
            started: ago(ms),
            started_wall: wall - Duration::from_millis(ms),
            resp_bytes: bytes,
            tokens,
        };
        tick.detail.requests = vec![
            req(
                "qwen-vllm",
                "30",
                Backend::Vllm,
                8_751,
                4_096,
                Some(LiveTokens {
                    prompt: Some(9_000),
                    cached: Some(8_192),
                    output: Some(64),
                    ..LiveTokens::default()
                }),
            ),
            req(
                "qwen-vllm",
                "31",
                Backend::Vllm,
                2_300,
                0,
                Some(LiveTokens {
                    output: Some(12),
                    approx: true,
                    ..LiveTokens::default()
                }),
            ),
            req(
                "bonsai-sg",
                "32",
                Backend::SgLang,
                1_200,
                2_048,
                Some(LiveTokens {
                    held: Some(30_000),
                    gen_tps: Some(41.5),
                    ..LiveTokens::default()
                }),
            ),
            req("bonsai-sg", "33", Backend::SgLang, 400, 4_120, None),
        ];
        note_poll(&mut tick, mono, wall);
        let ctx = frame_ctx(None, true);
        let model = tty_model(&sample, &mut tick, mono, wall, Zone::utc(), &ctx);
        let flying: Vec<_> = model
            .requests
            .iter()
            .filter(|r| r.inflight.is_some())
            .collect();
        assert_eq!(flying.len(), 4);
        let rows = frame_rows(&layout::layout(&model, 192, 60));
        let recent = rows
            .iter()
            .position(|r| r.contains("RECENT"))
            .expect("RECENT");
        let shown: Vec<&String> = rows[recent + 1..recent + 5].iter().collect();
        // Newest first.
        let streaming = shown[0];
        assert!(streaming.contains("bonsai-sg") && streaming.contains("streaming 4.1 kB"));
        assert!(streaming.trim_end().ends_with("gen"), "{streaming}");
        let held = shown[1];
        assert!(held.contains("~41.5") && held.contains("1.2s"), "{held}");
        assert_eq!(
            held.matches("--").count(),
            5,
            "PROMPT, CACHED, NEW, OUTPUT and READ: {held}"
        );
        let shared = shown[2];
        assert!(
            shared.contains("\u{2248}12") && shared.trim_end().ends_with("pp"),
            "{shared}"
        );
        let exact = shown[3];
        for text in ["9,000", "8,192", "64", "8.8s"] {
            assert!(exact.contains(text), "{text}: {exact}");
        }
        assert!(
            !exact.contains('\u{2248}') && exact.trim_end().ends_with("gen"),
            "{exact}"
        );
        assert!(shown.iter().all(|row| row.starts_with('>')));
    }

    #[test]
    fn strata_live_line_shows_phase_and_progress() {
        use llama_core::backend::EngineStats;
        let flash = served(
            "flash",
            "ready",
            Some(BackendInfo {
                kind: Backend::Strata,
                max_running: Some(1),
                running: Some(1),
                queued: Some(0),
                engine: EngineStats {
                    spec_permille: Some(700),
                    prefill_tps_tenths: Some(9677),
                    decode_tps_tenths: Some(309),
                    expert_hit_permille: Some(874),
                    pcie_share_permille: Some(92),
                    ..EngineStats::default()
                },
                ..BackendInfo::default()
            }),
        );
        let sample = backend_sample(vec![flash]);
        let mut live = vec![ModelEngineLive {
            model: "flash".to_owned(),
            live: EngineLive {
                state: Some("generating".to_owned()),
                phase: Some("writing a tool call: edit".to_owned()),
                prompt_tokens: Some(1800),
                generated: Some(14_558),
                max_tokens: Some(32_000),
                tok_s: Some(29.64),
                ..EngineLive::default()
            },
        }];
        // The expert cache numbers are SETUP's (`live:expert_hit`).
        let first = "strata  running 1/1 · queued 0 · spec 70 % · prefill 968/s · decode 30.9/s";
        assert_eq!(
            engine_lines(&sample, &live, WatchState::Ready),
            vec![
                first.to_owned(),
                "        writing a tool call: edit · gen 14,558/32,000 · 29.6/s".to_owned(),
            ]
        );
        live[0].live = EngineLive {
            state: Some("reading".to_owned()),
            prompt_read: Some(4096),
            prompt_total: Some(10_240),
            ..EngineLive::default()
        };
        assert_eq!(
            engine_lines(&sample, &live, WatchState::Ready)[1],
            "        reading · prompt 4,096/10,240 (40 %)"
        );
        live[0].live.prompt_read = None;
        live[0].live.prompt_tokens = Some(1800);
        assert_eq!(
            engine_lines(&sample, &live, WatchState::Ready)[1],
            "        reading · prompt 1,800"
        );
        live[0].live.state = Some("idle".to_owned());
        assert_eq!(
            engine_lines(&sample, &live, WatchState::Ready),
            vec![first.to_owned()]
        );
        // Another model's report is not this one's.
        live[0].model = "other".to_owned();
        live[0].live.state = Some("generating".to_owned());
        assert_eq!(engine_lines(&sample, &live, WatchState::Ready).len(), 1);
    }

    /// #5: a header model without `/slots` shows its last capture, titled
    /// as the last finished exchange; with text off, nothing.
    #[test]
    fn a_capture_fills_in_and_out_for_a_model_without_slots() {
        let cells = |text: &str| -> Vec<Cell> {
            text.chars()
                .map(|ch| Cell::new(ch, crate::tty::C16::White, crate::tty::C16::Black))
                .collect()
        };
        let sample = backend_sample(vec![served("flash", "ready", Some(sglang(Some(0))))]);
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        let mut ctx = frame_ctx(None, true);
        ctx.show_text = true;
        let now = Instant::now();
        let wall = SystemTime::now();
        let model = tty_model(&sample, &mut tick, now, wall, Zone::utc(), &ctx);
        assert_eq!(model.text_note, NO_SLOTS_TEXT, "no capture yet");
        tick.detail.capture = Some(CaptureView {
            model: "flash".to_owned(),
            id: 4,
            input: cells("An invented question?"),
            input_note: String::new(),
            output: cells("An invented answer."),
            over_slots: false,
        });
        let model = tty_model(&sample, &mut tick, now, wall, Zone::utc(), &ctx);
        assert_eq!(model.in_title, CAPTURE_IN_TITLE);
        assert_eq!(model.out_title, CAPTURE_OUT_TITLE);
        assert_eq!(model.in_lines, vec!["An invented question?".to_owned()]);
        // #38: a tool loop's IN says what it holds.
        tick.detail.capture.as_mut().expect("capture").input_note = "3 tool results".to_owned();
        let model = tty_model(&sample, &mut tick, now, wall, Zone::utc(), &ctx);
        assert_eq!(model.in_title, "IN (last request \u{00B7} 3 tool results)");
        tick.detail.capture.as_mut().expect("capture").input_note = String::new();
        assert_eq!(model.out_lines, vec!["An invented answer.".to_owned()]);
        assert!(model.text_note.is_empty());
        // A capture of another model is not this header's.
        tick.detail.capture.as_mut().expect("capture").model = "other".to_owned();
        let model = tty_model(&sample, &mut tick, now, wall, Zone::utc(), &ctx);
        assert_eq!(model.text_note, NO_SLOTS_TEXT);
        assert!(model.in_lines.is_empty());
        // Text off: nothing, whatever the detail holds.
        tick.detail.capture.as_mut().expect("capture").model = "flash".to_owned();
        ctx.show_text = false;
        let model = tty_model(&sample, &mut tick, now, wall, Zone::utc(), &ctx);
        assert!(model.in_lines.is_empty() && model.out_lines.is_empty());
        assert!(model.text_note.is_empty());
    }

    /// #66: a llama.cpp header model whose `/slots` has no text shows its
    /// last capture in IN and OUT; SLOTS still counts its slots.
    #[test]
    fn a_capture_over_textless_slots_fills_in_and_out() {
        let cells = |text: &str| -> Vec<Cell> {
            text.chars()
                .map(|ch| Cell::new(ch, crate::tty::C16::White, crate::tty::C16::Black))
                .collect()
        };
        let lcpp = BackendInfo {
            kind: Backend::LlamaCpp,
            ..BackendInfo::default()
        };
        let sample = backend_sample(vec![served("lcpp", "ready", Some(lcpp))]);
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        let mut textless = slot(0, 7, true, "", "");
        textless.model = "lcpp".to_owned();
        tick.detail.slots = vec![textless];
        let mut ctx = frame_ctx(None, true);
        ctx.show_text = true;
        let now = Instant::now();
        let wall = SystemTime::now();
        let mut capture = CaptureView {
            model: "lcpp".to_owned(),
            id: 9,
            input: cells("An invented question?"),
            input_note: String::new(),
            output: cells("An invented answer."),
            over_slots: false,
        };
        // A capture not meant to stand in for slots is not used.
        tick.detail.capture = Some(capture.clone());
        let model = tty_model(&sample, &mut tick, now, wall, Zone::utc(), &ctx);
        assert_ne!(model.in_title, CAPTURE_IN_TITLE);
        capture.over_slots = true;
        tick.detail.capture = Some(capture);
        let model = tty_model(&sample, &mut tick, now, wall, Zone::utc(), &ctx);
        assert_eq!(model.in_title, CAPTURE_IN_TITLE);
        assert_eq!(model.in_lines, vec!["An invented question?".to_owned()]);
        assert_eq!(model.out_lines, vec!["An invented answer.".to_owned()]);
        assert_eq!(model.slots_line, "1/1 busy");
        assert!(model.text_note.is_empty());
        // Another model's capture is not this header's.
        tick.detail.capture.as_mut().expect("capture").model = "other".to_owned();
        let model = tty_model(&sample, &mut tick, now, wall, Zone::utc(), &ctx);
        assert_ne!(model.in_title, CAPTURE_IN_TITLE);
    }

    #[test]
    fn stopping_for_over_a_minute_is_stuck() {
        let t0 = Instant::now();
        let mut since = HashMap::new();
        let stopping = backend_sample(vec![served("flash", "stopping", None)]);
        note_stopping(&mut since, &stopping.snapshot.models, t0);
        let later = t0 + Duration::from_secs(30);
        note_stopping(&mut since, &stopping.snapshot.models, later);
        assert!(!model_stuck(&stopping, WatchState::Ready, &since, later));
        let at = t0 + Duration::from_secs(61);
        assert!(model_stuck(&stopping, WatchState::Ready, &since, at));
        assert!(!model_stuck(&stopping, WatchState::AiDown, &since, at));
        // Back to ready and stopping again: the clock restarts.
        let ready = backend_sample(vec![served("flash", "ready", None)]);
        note_stopping(&mut since, &ready.snapshot.models, at);
        assert!(since.is_empty());
        note_stopping(&mut since, &stopping.snapshot.models, at);
        assert!(!model_stuck(
            &stopping,
            WatchState::Ready,
            &since,
            at + Duration::from_secs(10)
        ));
    }

    fn named(name: &str, state: &str) -> ModelInfo {
        let mut model = served(name, state, Some(BackendInfo::default()));
        model.full_name = Some(format!("Full {name}"));
        model
    }

    /// One frame. `tick` keeps the phase clocks across calls.
    fn phase_frame(models: Vec<ModelInfo>, tick: &mut TickState, now: Instant) -> TtyModel {
        let sample = backend_sample(models);
        tty_model(
            &sample,
            tick,
            now,
            SystemTime::UNIX_EPOCH,
            Zone::utc(),
            &frame_ctx(None, true),
        )
    }

    fn header_of(model: &TtyModel) -> (String, crate::tty::grid::Grid) {
        let grid = layout::layout(model, 240, 67);
        let text: String = (0..grid.cols())
            .map(|col| grid.get(col, 0).expect("cell").ch)
            .collect();
        (text, grid)
    }

    fn assert_word(
        grid: &crate::tty::grid::Grid,
        header: &str,
        word: &str,
        fg: crate::tty::C16,
        bg: crate::tty::C16,
    ) {
        let at = header
            .find(word)
            .unwrap_or_else(|| panic!("missing {word}: {header}"));
        // `█` is three bytes and one column.
        let col = u16::try_from(header[..at].chars().count()).expect("column");
        let pip = grid.get(24, 0).expect("pip");
        assert_eq!(pip.ch, '█', "{header}");
        assert_eq!((pip.fg, pip.bg), (fg, bg), "pip of {word}");
        for (i, ch) in word.chars().enumerate() {
            let cell = grid
                .get(col + u16::try_from(i).expect("offset"), 0)
                .expect("word");
            assert_eq!(cell.ch, ch);
            assert_eq!((cell.fg, cell.bg), (fg, bg), "{word} cell {i}");
        }
    }

    /// `AI DOWN` is a black-on-yellow badge. Its pip stays yellow on black,
    /// so [`assert_word`] (pip and word the same colour) does not fit.
    fn assert_ai_down(grid: &crate::tty::grid::Grid, header: &str) {
        use crate::tty::C16;
        let word = " AI DOWN ";
        let at = header
            .find(word)
            .unwrap_or_else(|| panic!("missing {word}: {header}"));
        let col = u16::try_from(header[..at].chars().count()).expect("column");
        let pip = grid.get(24, 0).expect("pip");
        assert_eq!((pip.ch, pip.fg, pip.bg), ('█', C16::Yellow, C16::Black));
        for (i, ch) in word.chars().enumerate() {
            let cell = grid
                .get(col + u16::try_from(i).expect("offset"), 0)
                .expect("badge");
            assert_eq!(cell.ch, ch);
            assert_eq!((cell.fg, cell.bg), (C16::Black, C16::Yellow));
        }
    }

    /// #135: the word and the two fields for starting, stopping, ready and
    /// an unknown state. The name and the engine stay.
    #[test]
    fn header_word_and_fields_follow_the_swap_state() {
        use crate::tty::C16;
        let t0 = Instant::now();
        let cases = [
            (
                "starting",
                true,
                false,
                "LOADING",
                "--",
                "loading 0:00",
                C16::Yellow,
                C16::Black,
            ),
            (
                "stopping",
                false,
                true,
                "STOPPING",
                "--",
                "stopping 0:00",
                C16::Yellow,
                C16::Black,
            ),
            (
                "ready",
                false,
                false,
                "READY",
                "0/0 idle",
                "none",
                C16::Green,
                C16::Black,
            ),
            (
                "other",
                false,
                false,
                "READY",
                "0/0 idle",
                "none",
                C16::Green,
                C16::Black,
            ),
        ];
        for (state, loading, stopping, word, slots, swap, fg, bg) in cases {
            let mut tick = TickState::new(2, 6);
            tick.heard = true;
            let model = phase_frame(vec![named("cedar-70b", state)], &mut tick, t0);
            assert_eq!(model.model_loading, loading, "{state}");
            assert_eq!(model.model_stopping, stopping, "{state}");
            assert_eq!(model.slots_line, slots, "{state}");
            assert_eq!(model.swap_line, swap, "{state}");
            assert_eq!(model.model_name, "Full cedar-70b", "{state}");
            assert_eq!(model.model_detail, "llama.cpp", "{state}");
            assert_eq!(model.state, WatchState::Ready, "{state}");
            let (header, grid) = header_of(&model);
            assert_word(&grid, &header, word, fg, bg);
            assert!(header.contains("Full cedar-70b"), "{state}: {header}");
            assert!(header.contains("llama.cpp"), "{state}: {header}");
            assert!(
                header.contains(&format!("slots {slots}")),
                "{state}: {header}"
            );
            assert!(
                header.contains(&format!("swap {swap}")),
                "{state}: {header}"
            );
        }
        // A busy slot does not bring the count back while the header model
        // is still starting, and the word stays LOADING.
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        tick.detail.slots = vec![slot(0, 1, true, "", "")];
        let model = phase_frame(vec![named("cedar-70b", "starting")], &mut tick, t0);
        assert_eq!(model.state, WatchState::Generating);
        assert!(model.model_loading);
        assert_eq!(model.slots_line, "--");
        assert_eq!(model.swap_line, "loading 0:00");
        let (header, grid) = header_of(&model);
        assert_word(&grid, &header, "LOADING", C16::Yellow, C16::Black);
        assert!(
            !header.contains("READY") && !header.contains("GENERATING"),
            "{header}"
        );
    }

    fn gpu_proc(pid: u32, memory_bytes: u64, name: &str) -> GpuProcess {
        GpuProcess {
            pid,
            memory_bytes,
            name: name.to_owned(),
        }
    }

    fn gpu_seen(list: &[GpuProcess]) -> GpuProcesses {
        GpuProcesses {
            list: list.to_vec(),
            failed: false,
        }
    }

    /// #139: the outside-engine field. A listed model, a job view, a read
    /// error, or a process under 256 MiB leaves it unset. The largest
    /// process wins; equal memory keeps the lower pid.
    #[test]
    fn gpu_busy_names_the_largest_process_outside_llama_swap() {
        let min = 256 * 1024 * 1024;
        let big = gpu_proc(9, 24_051_816_858, "example-bench");
        let mid = gpu_proc(2, 2 * 1_073_741_824, "smaller");
        let under = gpu_proc(3, min - 1, "small");
        let listed = gpu_seen(&[big.clone(), mid.clone(), under.clone()]);
        for state in ["ready", "starting", "stopping", "stopped", "other"] {
            let models = vec![served("cedar-70b", state, None)];
            assert!(
                outside_when_idle(&models, false, &listed).is_none(),
                "{state}"
            );
        }
        assert!(outside_when_idle(&[], true, &listed).is_none(), "job view");
        let failed = GpuProcesses {
            list: vec![big.clone()],
            failed: true,
        };
        assert!(
            outside_when_idle(&[], false, &failed).is_none(),
            "a read error is not busy, even with a stale list"
        );
        assert!(outside_when_idle(&[], false, &gpu_seen(&[])).is_none());
        assert!(outside_when_idle(&[], false, &gpu_seen(&[under])).is_none());
        let edge = gpu_seen(&[gpu_proc(1, min, "edge")]);
        assert_eq!(
            outside_when_idle(&[], false, &edge).as_deref(),
            Some("edge 0.2 GB"),
            "256 MiB is 0.2 GB in the VRAM meter's precision"
        );
        assert_eq!(
            outside_when_idle(&[], false, &listed).as_deref(),
            Some("example-bench 22.4 GB")
        );
        let tied = gpu_seen(&[gpu_proc(10, min, "alpha"), gpu_proc(4, min, "beta")]);
        assert_eq!(
            outside_when_idle(&[], false, &tied).as_deref(),
            Some("beta 0.2 GB"),
            "equal memory keeps the lower pid"
        );
        let dirty = gpu_seen(&[gpu_proc(8, min, "../weird name!")]);
        assert_eq!(
            outside_when_idle(&[], false, &dirty).as_deref(),
            Some("..weirdname 0.2 GB")
        );
        let unnamed = gpu_seen(&[gpu_proc(42, min, "!!!")]);
        assert_eq!(
            outside_when_idle(&[], false, &unnamed).as_deref(),
            Some("pid 42 0.2 GB")
        );
        let long = gpu_seen(&[gpu_proc(1, min, &"a".repeat(40))]);
        let cut = format!("{} 0.2 GB", "a".repeat(24));
        assert_eq!(
            outside_when_idle(&[], false, &long).as_deref(),
            Some(cut.as_str())
        );
    }

    /// The process-selection cases above are an answered idle list.
    fn outside_when_idle(
        models: &[ModelInfo],
        job_up: bool,
        processes: &GpuProcesses,
    ) -> Option<String> {
        outside_engine(models, job_up, processes, true, true, AiState::Idle)
    }

    /// #139: each input of "llama-swap answered with an empty list". Dropping
    /// llama-enabled, heard, or `ai == Idle` makes one of these fail.
    #[test]
    fn gpu_busy_only_after_llama_swap_answers_idle() {
        let big = gpu_seen(&[gpu_proc(9, 24_051_816_858, "example-bench")]);
        let show = |enabled, heard, ai| outside_engine(&[], false, &big, enabled, heard, ai);
        assert!(show(true, false, AiState::Idle).is_none(), "not heard");
        assert!(show(true, true, AiState::Down).is_none(), "down");
        assert!(show(true, true, AiState::NoData).is_none(), "no data");
        assert!(show(false, true, AiState::Idle).is_none(), "llama disabled");
        assert!(
            show(true, true, AiState::Loaded).is_none(),
            "loaded is not idle"
        );
        assert_eq!(
            show(true, true, AiState::Idle).as_deref(),
            Some("example-bench 22.4 GB"),
            "enabled, heard, and idle"
        );
    }

    /// #139: the header word and fields when llama-swap answered idle.
    /// Not heard, down, no-data and llama disabled keep their own words.
    /// A live job keeps its view.
    #[test]
    fn gpu_busy_header_replaces_an_idle_header_and_yields_to_a_job() {
        use crate::tty::C16;
        let min = 256 * 1024 * 1024;
        let processes = GpuProcesses {
            list: vec![gpu_proc(9, 24_051_816_858, "example-bench")],
            failed: false,
        };
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        let ctx = frame_ctx(None, true);
        let mono = Instant::now();
        let wall = SystemTime::UNIX_EPOCH;

        // `backend_sample` marks the view loaded. An answered empty list is Idle.
        let mut idle = backend_sample(Vec::new());
        idle.snapshot.ai = AiState::Idle;
        idle.gpu_processes = processes;
        let model = tty_model(&idle, &mut tick, mono, wall, Zone::utc(), &ctx);
        assert!(model.gpu_busy);
        assert!(!model.model_loading && !model.model_stopping);
        assert_eq!(model.state, WatchState::Ready);
        assert_eq!(model.model_name, "outside llama-swap");
        assert_eq!(model.model_detail, "example-bench 22.4 GB");
        assert_eq!(model.slots_line, "--");
        assert_eq!(model.swap_line, "--");
        let (header, grid) = header_of(&model);
        assert_word(&grid, &header, "GPU BUSY", C16::Yellow, C16::Black);
        assert!(header.contains("outside llama-swap"), "{header}");
        assert!(header.contains("example-bench 22.4 GB"), "{header}");
        assert!(header.contains("slots --"), "{header}");
        assert!(header.contains("swap --"), "{header}");
        assert!(!header.contains("READY"), "{header}");

        // A failed read, even with a large process still in the list, does not.
        idle.gpu_processes.failed = true;
        let quiet = tty_model(&idle, &mut tick, mono, wall, Zone::utc(), &ctx);
        assert!(!quiet.gpu_busy);
        assert_eq!(quiet.model_name, "--");
        assert_eq!(quiet.model_detail, "engine --");
        assert_eq!(quiet.slots_line, "0/0 idle");
        assert_eq!(quiet.swap_line, "none");

        // Before the first answer the word stays STARTING, including its fields.
        tick.heard = false;
        idle.gpu_processes.failed = false;
        let starting = tty_model(&idle, &mut tick, mono, wall, Zone::utc(), &ctx);
        assert!(!starting.gpu_busy);
        assert_eq!(starting.state, WatchState::Starting);
        assert_eq!(starting.model_name, "...");
        assert_eq!(starting.model_detail, "engine --");
        assert_eq!(starting.slots_line, "...");
        assert_eq!(starting.swap_line, "...");
        let (header, grid) = header_of(&starting);
        assert_word(&grid, &header, "STARTING", C16::BrightBlack, C16::Black);
        assert!(!header.contains("GPU BUSY"), "{header}");

        tick.heard = true;
        idle.snapshot.ai = AiState::Down;
        let down = tty_model(&idle, &mut tick, mono, wall, Zone::utc(), &ctx);
        assert!(!down.gpu_busy);
        assert_eq!(down.state, WatchState::AiDown);
        assert_eq!(down.model_name, "--");
        assert_eq!(down.model_detail, "engine --");
        assert_eq!(down.slots_line, "--");
        assert_eq!(down.swap_line, "--");
        let (header, grid) = header_of(&down);
        assert_ai_down(&grid, &header);
        assert!(!header.contains("GPU BUSY"), "{header}");

        idle.snapshot.ai = AiState::NoData;
        let no_data = tty_model(&idle, &mut tick, mono, wall, Zone::utc(), &ctx);
        assert!(!no_data.gpu_busy);
        assert_eq!(no_data.state, WatchState::AiDown);
        assert_eq!(no_data.model_name, "--");
        assert_eq!(no_data.model_detail, "engine --");
        let (header, grid) = header_of(&no_data);
        assert_ai_down(&grid, &header);
        assert!(!header.contains("GPU BUSY"), "{header}");

        // Heard and idle, so only the disabled flag keeps today's word.
        idle.snapshot.ai = AiState::Idle;
        let off = frame_ctx(None, false);
        let no_llama = tty_model(&idle, &mut tick, mono, wall, Zone::utc(), &off);
        assert!(!no_llama.gpu_busy);
        assert_eq!(no_llama.state, WatchState::NoLlama);
        assert_eq!(no_llama.model_name, "--");
        assert_eq!(no_llama.model_detail, "engine --");
        assert_eq!(no_llama.slots_line, "--");
        assert_eq!(no_llama.swap_line, "--");
        let (header, grid) = header_of(&no_llama);
        assert_word(&grid, &header, "NO LLAMA", C16::BrightBlack, C16::Black);
        assert!(!header.contains("GPU BUSY"), "{header}");

        // A listed model wins, including one that is only starting.
        idle.snapshot.ai = AiState::Idle;
        let mut listed = backend_sample(vec![named("cedar-70b", "starting")]);
        listed.gpu_processes = GpuProcesses {
            list: vec![gpu_proc(9, min, "example-bench")],
            failed: false,
        };
        let model = tty_model(&listed, &mut tick, mono, wall, Zone::utc(), &ctx);
        assert!(!model.gpu_busy);
        assert!(model.model_loading);
        assert_eq!(model.model_name, "Full cedar-70b");

        // A live job view wins. A finished job is a note, so the busy header stays.
        // The file's mtime is one second before the clock passed into the frame.
        let fresh = wall + Duration::from_secs(10);
        tick.detail.jobs = vec![job_at("run", JobState::Running, fresh)];
        let model = tty_model(
            &idle,
            &mut tick,
            mono,
            fresh + Duration::from_secs(1),
            Zone::utc(),
            &ctx,
        );
        assert!(model.job.is_some(), "idle live job is the job view");
        assert!(!model.gpu_busy);
        assert_ne!(model.model_name, "outside llama-swap");

        tick.detail.jobs = vec![job_at("wrap", JobState::Done, fresh)];
        let model = tty_model(
            &idle,
            &mut tick,
            mono,
            fresh + Duration::from_secs(1),
            Zone::utc(),
            &ctx,
        );
        assert!(model.job.is_none(), "a finished job is not a view");
        assert!(model.gpu_busy);
        assert!(model.job_note.is_some());
    }

    /// #135: the swap clock starts when the state is first seen, restarts
    /// when a model leaves and comes back, takes the longest of several,
    /// and prefers `starting` over `stopping`.
    #[test]
    fn swap_time_counts_from_first_seen() {
        let t0 = Instant::now();
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        let cedar = |state: &str| named("cedar-70b", state);

        let first = phase_frame(vec![cedar("starting")], &mut tick, t0);
        assert_eq!(first.swap_line, "loading 0:00");
        let later = phase_frame(
            vec![cedar("starting")],
            &mut tick,
            t0 + Duration::from_secs(102),
        );
        assert_eq!(later.swap_line, "loading 1:42");
        assert_eq!(later.slots_line, "--");
        assert!(later.model_loading);

        // Ready clears the clock. Starting again counts from the new sight.
        let ready = phase_frame(
            vec![cedar("ready")],
            &mut tick,
            t0 + Duration::from_secs(102),
        );
        assert_eq!(ready.swap_line, "none");
        assert_eq!(ready.slots_line, "0/0 idle");
        assert!(!ready.model_loading);
        let back = phase_frame(
            vec![cedar("starting")],
            &mut tick,
            t0 + Duration::from_secs(130),
        );
        assert_eq!(back.swap_line, "loading 0:00");
        let again = phase_frame(
            vec![cedar("starting")],
            &mut tick,
            t0 + Duration::from_secs(140),
        );
        assert_eq!(again.swap_line, "loading 0:10");

        // The hour edge, and a second model seen later: the longer one.
        let mut since = HashMap::new();
        let stopping = HashMap::new();
        let one = vec![served("cedar-70b", "starting", None)];
        note_starting(&mut since, &one, t0);
        let at_hour = t0 + Duration::from_secs(3599);
        assert_eq!(
            swap_line(WatchState::Ready, &one, &since, &stopping, at_hour),
            "loading 59:59"
        );
        let hour = t0 + Duration::from_secs(3600);
        note_starting(&mut since, &one, hour);
        assert_eq!(
            swap_line(WatchState::Ready, &one, &since, &stopping, hour),
            "loading 1:00:00"
        );
        let past = t0 + Duration::from_secs(3600 + 62);
        assert_eq!(
            swap_line(WatchState::Ready, &one, &since, &stopping, past),
            "loading 1:01:02"
        );
        let long = t0 + Duration::from_secs(10 * 3600 + 12 * 60 + 9);
        assert_eq!(
            swap_line(WatchState::Ready, &one, &since, &stopping, long),
            "loading 10:12:09"
        );
        let both = vec![
            served("cedar-70b", "starting", None),
            served("birch-4b", "starting", None),
        ];
        note_starting(&mut since, &both, t0 + Duration::from_secs(90));
        // cedar from t0 (90s), birch from 90s (0s).
        assert_eq!(
            swap_line(
                WatchState::Ready,
                &both,
                &since,
                &stopping,
                t0 + Duration::from_secs(90)
            ),
            "loading 1:30"
        );
        // The one seen longest is not listed first.
        let later_first = vec![
            served("birch-4b", "starting", None),
            served("cedar-70b", "starting", None),
        ];
        assert_eq!(
            swap_line(
                WatchState::Ready,
                &later_first,
                &since,
                &stopping,
                t0 + Duration::from_secs(90)
            ),
            "loading 1:30"
        );
        // birch leaves. cedar's clock keeps going.
        let only = vec![served("cedar-70b", "starting", None)];
        note_starting(&mut since, &only, t0 + Duration::from_secs(100));
        assert_eq!(
            swap_line(
                WatchState::Ready,
                &only,
                &since,
                &stopping,
                t0 + Duration::from_secs(100)
            ),
            "loading 1:40"
        );

        // A later `starting` model wins the swap line over a longer `stopping`.
        // The header word follows the first model.
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        let mixed_at = t0 + Duration::from_secs(5);
        let _ = phase_frame(vec![named("birch-4b", "stopping")], &mut tick, t0);
        let mixed = phase_frame(
            vec![
                named("birch-4b", "stopping"),
                named("cedar-70b", "starting"),
            ],
            &mut tick,
            mixed_at,
        );
        assert!(mixed.model_stopping && !mixed.model_loading);
        assert_eq!(mixed.slots_line, "--");
        assert_eq!(mixed.swap_line, "loading 0:00");
        let (header, _) = header_of(&mixed);
        assert!(header.contains("STOPPING"), "{header}");
        assert!(header.contains("swap loading 0:00"), "{header}");
        assert!(!header.contains("READY"), "{header}");

        // Header ready, another model starting: the word stays READY and
        // the swap line says loading.
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        let _ = phase_frame(vec![named("cedar-70b", "starting")], &mut tick, t0);
        let ready_header = phase_frame(
            vec![named("birch-4b", "ready"), named("cedar-70b", "starting")],
            &mut tick,
            t0 + Duration::from_secs(75),
        );
        assert!(!ready_header.model_loading && !ready_header.model_stopping);
        assert_eq!(ready_header.model_name, "Full birch-4b");
        assert_eq!(ready_header.slots_line, "0/0 idle");
        assert_eq!(ready_header.swap_line, "loading 1:15");

        // Stopping for over a minute is still stuck, and the word is STOPPING.
        let mut tick = TickState::new(2, 6);
        tick.heard = true;
        let _ = phase_frame(vec![cedar("stopping")], &mut tick, t0);
        let early = phase_frame(
            vec![cedar("stopping")],
            &mut tick,
            t0 + Duration::from_secs(30),
        );
        assert!(!early.model_stuck);
        assert_eq!(early.swap_line, "stopping 0:30");
        let stuck = phase_frame(
            vec![cedar("stopping")],
            &mut tick,
            t0 + Duration::from_secs(61),
        );
        assert!(stuck.model_stuck && stuck.model_stopping);
        assert_eq!(stuck.slots_line, "--");
        assert_eq!(stuck.swap_line, "stopping 1:01");

        // Two `stopping` models: the one seen longest is listed second.
        let mut stop_since = HashMap::new();
        let cedar_stopping = vec![served("cedar-70b", "stopping", None)];
        note_stopping(&mut stop_since, &cedar_stopping, t0);
        let birch_first = vec![
            served("birch-4b", "stopping", None),
            served("cedar-70b", "stopping", None),
        ];
        note_stopping(&mut stop_since, &birch_first, t0 + Duration::from_secs(25));
        assert_eq!(
            swap_line(
                WatchState::Ready,
                &birch_first,
                &since,
                &stop_since,
                t0 + Duration::from_secs(25)
            ),
            "stopping 0:25"
        );
    }

    struct FixedCpu;

    impl SampleStep for FixedCpu {
        fn sample(&mut self, mono: Instant, wall: SystemTime, _llama: &LlamaView) -> WatchSample {
            WatchSample {
                snapshot: Snapshot {
                    t_mono: mono,
                    t_wall: wall,
                    load: None,
                    activity: None,
                    cpu_pct: Some(41.0),
                    cpu_topk_pct: None,
                    gpu_pct: None,
                    mem_pct: None,
                    coolant_c: None,
                    cpu_c: None,
                    gpu_c: None,
                    ai: AiState::Idle,
                    models: Vec::new(),
                    tokens: None,
                    errors: std::collections::BTreeSet::new(),
                },
                gpu: crate::collector::GpuExtra::default(),
                cpu_w: None,
                activity_w: None,
                load_source: crate::collector::LoadSource::Util,
                fans: None,
                temps: None,
                gpu_processes: crate::collector::GpuProcesses::default(),
                host_io: crate::sources::hostio::HostIo::default(),
            }
        }
    }

    struct AcceptPublish;

    impl PublishStep for AcceptPublish {
        fn publish(
            &mut self,
            _snapshot: &Snapshot,
            _llama: &LlamaView,
            _extras: &Extras,
        ) -> Result<(), crate::publish::PublishError> {
            Ok(())
        }
    }

    struct RecordModels {
        models: std::sync::Arc<std::sync::Mutex<Vec<TtyModel>>>,
    }

    impl RenderStep for RecordModels {
        fn draw(&mut self, model: &TtyModel, _now: Instant) -> std::io::Result<()> {
            self.models.lock().expect("models").push(model.clone());
            Ok(())
        }
    }

    struct StepClock {
        mono: Instant,
        wall: SystemTime,
    }

    impl Clock for StepClock {
        fn mono(&self) -> Instant {
            self.mono
        }

        fn wall(&self) -> SystemTime {
            self.wall
        }

        fn sleep(&mut self, d: Duration) {
            self.mono += d;
            self.wall += d;
        }
    }

    struct QuietNotify;

    impl Notifier for QuietNotify {
        fn ready(&mut self) {}
        fn watchdog(&mut self) {}
        fn stopping(&mut self) {}
    }

    struct OneTick(std::sync::atomic::AtomicU64);

    impl Stop for OneTick {
        fn requested(&self) -> bool {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 1
        }
    }

    struct QuietLog;

    impl llama_core::log::Sink for QuietLog {
        fn write_line(&mut self, _line: &str) {}
    }

    /// #115: `[grid] enabled = false` starts no writer, even when a size is
    /// set, so a frame builds nothing for the file.
    #[test]
    fn grid_disabled_does_not_start_a_writer() {
        let config =
            Config::from_toml("[grid]\nenabled = false\nsize = \"213x60\"\n").expect("parse");
        assert!(config.validate(8).is_ok());
        let now = Instant::now();
        let term = crate::tty::term::Term::new(
            Vec::new(),
            || {
                Ok(crate::tty::term::Size {
                    cols: 192,
                    rows: 60,
                })
            },
            Duration::from_secs(30),
            now,
        )
        .expect("term");
        let term = super::with_grid_file(term, &config);
        assert!(term.grid_tap().is_none());
    }

    fn grid_text(grid: &crate::tty::grid::Grid) -> String {
        let mut text = String::new();
        for row in 0..grid.rows() {
            for col in 0..grid.cols() {
                text.push(grid.get(col, row).expect("cell").ch);
            }
            text.push('\n');
        }
        text
    }
}
