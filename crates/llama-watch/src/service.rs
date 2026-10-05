//! 10 Hz watcher loop: poll, sample, publish, draw, then notify.
//!
//! `READY=1` is sent after start-up. `WATCHDOG=1` is sent only when the four
//! tick steps return. `STOPPING=1` is sent when [`Stop::requested`] becomes
//! true, which is how a test stands in for SIGINT or SIGTERM.
//!
//! In [`run`] the draw step is a [`FrameWriter`]: it hands the frame to the
//! tty writer thread and returns, so a console write that blocks (unblank,
//! Scroll Lock) never holds back publish or the watchdog (#42).

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sd_notify::NotifyState;
use thiserror::Error;

use crate::activity::ActivityRow;
use crate::collector::{WatchCollector, WatchSample};
use crate::config::{ChartGlyphs, Config, ConfigError, ValidWatchConfig};
use crate::poller::{self, CaptureView, LlamaDetail, ModelSetup, PollLatencies};
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
    self, Activity, HealthSeg, HealthStatus, SetupView, Slot, TtyModel, WatchState,
};
use crate::tty::sanitize::sanitize;
use crate::tty::term::{self, ConsoleBlank, Term};
use crate::tty::writer::FrameWriter;
use llama_core::backend;
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
        self.render(&grid, now)
    }

    fn finish(&mut self) -> io::Result<()> {
        self.restore_console()
    }

    fn repaint(&mut self) {
        self.force_repaint();
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
        ctx_history_h: input.config.tty.ctx_history_h,
        started: input.started,
        host: host_label(&roots.proc),
        cpu_cores: cpu_cores(&roots.proc),
        mem_total_bytes: mem_total_bytes(&roots.proc),
        llama_enabled: input.config.llama.enabled,
        show_text: input.config.tty.show_text,
        setup: Rules::compile(&input.config.setup).unwrap_or_default(),
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
    /// Per-slot context history for the SLOTS sparklines (T53).
    ctx_history: CtxBook,
    /// When each model (by name) entered llama-swap `stopping`.
    stopping_since: HashMap<String, Instant>,
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
            },
            heard: false,
            published: 0,
            write_down: false,
            draw_down: false,
            replay: OutReplay::default(),
            down_since: None,
            published_at: None,
            chart: TokenChart::new(chart_bucket_s),
            ctx_history: CtxBook::new(ctx_history_h),
            stopping_since: HashMap::new(),
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
    }
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
    state
        .chart
        .ingest(mono, state.detail.gen_tps, state.detail.prompt_tps);
    let running: Vec<String> = state
        .llama
        .models
        .iter()
        .map(|model| model.name.clone())
        .collect();
    state
        .ctx_history
        .record(mono, &running, &state.detail.slots);
    note_stopping(&mut state.stopping_since, &sample.snapshot.models, mono);
    let model = tty_model(&sample, state, mono, wall, ctx);
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
        Ok(term) => term
            .with_console_blank(ConsoleBlank::from_minutes(
                prepared.config.tty.blank_min,
                prepared.config.tty.sleep_min,
            ))
            .with_palette(prepared.config.tty.palette)
            .with_saved_modes(modes),
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
    let (poller, feed) = if prepared.config.llama.enabled {
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
        clock: RealClock,
        notify: SdNotify,
        stop: NeverStop,
        log: log::Stderr,
        roots: &roots,
        started: now,
    });
    drop(poller);
    exit.code()
}

fn tty_model(
    sample: &WatchSample,
    tick: &mut TickState,
    mono: Instant,
    wall: SystemTime,
    ctx: &FrameCtx,
) -> TtyModel {
    let watch = watch_state(sample, &tick.detail, tick.heard, ctx.llama_enabled);
    let no_slots = first_without_slots(sample, watch);
    // The header model has no `/slots`: IN and OUT from its last capture (#5).
    let capture = tick.detail.capture.as_ref().filter(|capture| {
        no_slots
            && tick.detail.slots.is_empty()
            && sample
                .snapshot
                .models
                .first()
                .is_some_and(|model| model.name == capture.model)
    });
    // With text off the poller keeps none, and nothing here reads the slot
    // tails either, so no llama text can reach the frame (RR-LV1).
    let text = if !ctx.show_text {
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
    let mem_total = ctx.mem_total_bytes.map(gib);
    let mem_used = mem_used_bytes(sample.snapshot.mem_pct, ctx.mem_total_bytes).map(gib);
    TtyModel {
        state: watch,
        host: ctx.host.clone(),
        model_name: model_name(sample, watch),
        model_detail: model_detail(sample, watch),
        model_stuck: model_stuck(sample, watch, &tick.stopping_since, mono),
        slots_line: if no_slots && tick.detail.slots.is_empty() {
            "--".to_owned()
        } else {
            slots_line(&tick.detail, watch)
        },
        swap_line: swap_line(watch),
        cool_c: temp_i(sample.snapshot.coolant_c),
        cpu_c: temp_i(sample.snapshot.cpu_c),
        gpu_c: temp_i(sample.snapshot.gpu_c),
        clock: format_wall(wall),
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
        gen_ceiling: ctx.gen_ceiling,
        prompt_ceiling: ctx.prompt_ceiling,
        slots: layout_slots(&tick.detail.slots, &tick.ctx_history),
        backend_lines: backend_lines(sample, watch),
        text_note: if ctx.show_text && no_slots && tick.detail.slots.is_empty() && capture.is_none()
        {
            NO_SLOTS_TEXT.to_owned()
        } else {
            String::new()
        },
        requests: layout_requests(&tick.detail.activity, watch),
        in_title: text.in_title,
        out_title: text.out_title,
        in_lines: text.in_lines,
        out_lines: text.out_lines,
        out_shown: text.out_shown,
        replay_frame: text.replay_frame,
        show_text: ctx.show_text,
        down_since: tick
            .down_since
            .map(|since| down_label(since, wall))
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
        fans: sample.fans.clone(),
        ctx_history_h: ctx.ctx_history_h,
        setup: setup_view(sample, &tick.detail, watch, &ctx.setup),
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
    let live = LiveCtx {
        backend: model.backend.map(|info| info.kind).unwrap_or_default(),
        detail: model.detail.as_ref(),
        info: model.backend.as_ref(),
    };
    Some(SetupView {
        id: setup.id,
        name: setup.name,
        more: models.len() - 1,
        rows: rules.rows(&setup.found, &live),
    })
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

/// Track when each model entered `stopping`; forget the rest.
fn note_stopping(since: &mut HashMap<String, Instant>, models: &[ModelInfo], now: Instant) {
    since.retain(|name, _| {
        models
            .iter()
            .any(|model| model.name == *name && model.state == "stopping")
    });
    for model in models.iter().filter(|model| model.state == "stopping") {
        since.entry(model.name.clone()).or_insert(now);
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

/// `sglang  running 1/4 · queued 0 · KV 37 % · hit 80 %` for each ready model
/// without `/slots`. Unknown gauges are `--`. Strata has no KV gauge:
/// `strata  running 1/1 · queued 0`. Engine numbers follow when the server
/// reports them (#31): `· spec 78 % · 2.9/step · ttft 420 ms · itl 31 ms ·
/// e2e 12.5 s · preempt 3` (preemptions only once there are some), then
/// the engine-measured speeds of the latest window with finished requests
/// (#35): `· prefill 2,134/s · decode 41.2/s`. A sleeping engine says so
/// first: `vllm  sleeping · running 0 · …`. The line is cut at the panel's
/// edge, so the speeds show where it fits.
fn backend_lines(sample: &WatchSample, state: WatchState) -> Vec<String> {
    if !matches!(state, WatchState::Generating | WatchState::Ready) {
        return Vec::new();
    }
    let sep = llama_core::detail::SEPARATOR;
    let num = |value: Option<u16>| value.map_or_else(|| "--".to_owned(), |n| n.to_string());
    let pct = |permille: Option<u16>| {
        permille.map_or_else(
            || "--".to_owned(),
            |p| format!("{} %", (u32::from(p) + 5) / 10),
        )
    };
    sample
        .snapshot
        .models
        .iter()
        .filter(|model| model.state == "ready")
        .filter_map(|model| model.backend)
        .filter(|info| !info.kind.has_slots())
        .map(|info| {
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
            if info.kind.has_kv_gauge() {
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
        })
        .collect()
}

/// `420 ms` below a second, `12.5 s` below 100 s, else whole seconds.
fn latency_text(us: u32) -> String {
    if us < 999_500 {
        format!("{} ms", (us + 500) / 1000)
    } else if us < 99_950_000 {
        let tenths = (us + 50_000) / 100_000;
        format!("{}.{} s", tenths / 10, tenths % 10)
    } else {
        format!("{} s", (us + 500_000) / 1_000_000)
    }
}

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

fn swap_line(state: WatchState) -> String {
    match state {
        WatchState::Starting => "...".to_owned(),
        WatchState::AiDown | WatchState::NoLlama => "--".to_owned(),
        WatchState::Generating | WatchState::Ready => "none".to_owned(),
    }
}

fn layout_slots(slots: &[SlotView], history: &CtxBook) -> Vec<Slot> {
    slots
        .iter()
        .filter_map(|slot| {
            let id = u32::try_from(slot.id).ok()?;
            Some(Slot {
                id,
                generating: slot.is_processing,
                done: slot.n_prompt_tokens_processed,
                total: slot.n_prompt_tokens,
                decoded: slot.n_decoded,
                ctx_prompt: slot.ctx_prompt,
                n_ctx: slot.n_ctx,
                ctx_history: history.points(&slot.model, slot.id),
            })
        })
        .collect()
}

fn layout_requests(rows: &[ActivityRow], state: WatchState) -> Vec<Activity> {
    rows.iter()
        .enumerate()
        .map(|(index, row)| Activity {
            live: index == 0 && state == WatchState::Generating,
            id: u32::try_from(row.id).unwrap_or(0),
            time: row.time.clone(),
            source: row.source.clone(),
            model: row.model.clone(),
            input_tok: row.input_tokens.unwrap_or(0),
            cached_tok: row.cached_tokens.unwrap_or(0),
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
        })
        .collect()
}

fn prompt_last(slots: &[SlotView]) -> Option<u64> {
    pick_slot(slots).map(|slot| slot.n_prompt_tokens)
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
        seg(
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
        seg("/slots", slots_status, &ms(detail.latencies.slots)),
        seg("metrics", metric_status, &ms(detail.latencies.metrics)),
        seg(
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
    ]
}

fn seg(name: &str, status: HealthStatus, note: &str) -> HealthSeg {
    HealthSeg {
        name: name.to_owned(),
        status,
        note: note.to_owned(),
    }
}

fn ms(latency: Option<Duration>) -> String {
    match latency {
        Some(latency) => format!("{}ms", latency.as_millis()),
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
            .map(|fan| (fan.channel, fan.label.clone(), fan.rpm, fan.pwm))
            .collect(),
        sources: wire_sources(&health(sample, &tick.detail, watch), &tick.detail.latencies),
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

fn down_label(since: SystemTime, now: SystemTime) -> String {
    let stamp = format_wall(since);
    let tod = stamp.get(11..).unwrap_or("--");
    let age = now.duration_since(since).unwrap_or_default();
    format!("{tod} ({})", format_span(age))
}

fn format_wall(wall: SystemTime) -> String {
    match wall.duration_since(UNIX_EPOCH) {
        Ok(duration) => format_unix(duration.as_secs()),
        Err(_) => "--".to_owned(),
    }
}

fn format_unix(secs: u64) -> String {
    let days = i64::try_from(secs / 86_400).unwrap_or(i64::MAX);
    let tod = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02} {hour:02}:{min:02}:{sec:02}",
        hour = tod / 3600,
        min = (tod % 3600) / 60,
        sec = tod % 60
    )
}

/// Howard Hinnant's `civil_from_days`. `days` is days since 1970-01-01.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = u64::try_from(z - era * 146_097).unwrap_or(0);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let year = i64::try_from(yoe).unwrap_or(0) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    (
        year,
        u32::try_from(month).unwrap_or(1),
        u32::try_from(day).unwrap_or(1),
    )
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
    use llama_core::backend::{Backend, BackendInfo};

    /// #35: RECENT takes the engine's speeds only where llama-swap gave
    /// none, and marks them; a llama.cpp row keeps its own.
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
        let shown = layout_requests(&rows, WatchState::Ready);
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
        assert_eq!(format_unix(0), "1970-01-01 00:00:00");
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
            captured: false,
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
        }
    }

    fn frame_ctx(mem_total_bytes: Option<u64>, llama_enabled: bool) -> FrameCtx {
        FrameCtx {
            output_cap: 100,
            input_cap: 100,
            gen_ceiling: 250.0,
            prompt_ceiling: 1500.0,
            chart_bucket_s: 2,
            chart_glyphs: ChartGlyphs::default(),
            ctx_history_h: 6,
            started: Instant::now(),
            host: "box".to_owned(),
            cpu_cores: None,
            mem_total_bytes,
            llama_enabled,
            show_text: false,
            setup: Rules::builtin(),
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
            vec![(2, "CPU".to_owned(), Some(1100), Some(128))]
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
                "vllm  running 1 · queued 0 · KV 41 % · hit 75 % · spec 78 % · 2.9/step · ttft 420 ms · itl 31 ms · e2e 12.5 s · preempt 3 · prefill 2,134/s · decode 41.2/s"
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
                "vllm  sleeping · running 1 · queued 0 · KV 41 % · hit 75 % · e2e 250 s".to_owned()
            ]
        );
    }

    #[test]
    fn latency_text_units() {
        assert_eq!(latency_text(0), "0 ms");
        assert_eq!(latency_text(31_499), "31 ms");
        assert_eq!(latency_text(999_499), "999 ms");
        assert_eq!(latency_text(999_500), "1.0 s");
        assert_eq!(latency_text(12_460_000), "12.5 s");
        assert_eq!(latency_text(99_940_000), "99.9 s");
        assert_eq!(latency_text(100_000_000), "100 s");
        assert_eq!(latency_text(3_600_000_000), "3600 s");
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
        let model = tty_model(&sample, &mut tick, now, wall, &ctx);
        assert_eq!(model.text_note, NO_SLOTS_TEXT, "no capture yet");
        tick.detail.capture = Some(CaptureView {
            model: "flash".to_owned(),
            id: 4,
            input: cells("An invented question?"),
            input_note: String::new(),
            output: cells("An invented answer."),
        });
        let model = tty_model(&sample, &mut tick, now, wall, &ctx);
        assert_eq!(model.in_title, CAPTURE_IN_TITLE);
        assert_eq!(model.out_title, CAPTURE_OUT_TITLE);
        assert_eq!(model.in_lines, vec!["An invented question?".to_owned()]);
        // #38: a tool loop's IN says what it holds.
        tick.detail.capture.as_mut().expect("capture").input_note = "3 tool results".to_owned();
        let model = tty_model(&sample, &mut tick, now, wall, &ctx);
        assert_eq!(model.in_title, "IN (last request \u{00B7} 3 tool results)");
        tick.detail.capture.as_mut().expect("capture").input_note = String::new();
        assert_eq!(model.out_lines, vec!["An invented answer.".to_owned()]);
        assert!(model.text_note.is_empty());
        // A capture of another model is not this header's.
        tick.detail.capture.as_mut().expect("capture").model = "other".to_owned();
        let model = tty_model(&sample, &mut tick, now, wall, &ctx);
        assert_eq!(model.text_note, NO_SLOTS_TEXT);
        assert!(model.in_lines.is_empty());
        // Text off: nothing, whatever the detail holds.
        tick.detail.capture.as_mut().expect("capture").model = "flash".to_owned();
        ctx.show_text = false;
        let model = tty_model(&sample, &mut tick, now, wall, &ctx);
        assert!(model.in_lines.is_empty() && model.out_lines.is_empty());
        assert!(model.text_note.is_empty());
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
