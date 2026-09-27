//! LLD state table, one test per row, driven with FakeLcd and a fake clock.

use std::collections::{BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use kraken_lcd::collector::{AiState, Snapshot};
use kraken_lcd::config::{Config, ValidConfig};
use kraken_lcd::device::{FakeLcd, LcdSink, Record, SinkError};
use kraken_lcd::log::{self, Priority};
use kraken_lcd::present::Ai;
use kraken_lcd::render::Frame;
use kraken_lcd::service::{self, Clock, LoopExit, LoopInput, Notifier, Sampler, Stop};
use kraken_lcd::sources::SourceId;

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("t14-table-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch");
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn load_config(dir: &Path, min_interval_s: u64, fail_limit: u32, tick_s: f64) -> ValidConfig {
    let path = dir.join("config.toml");
    let text = format!(
        "[writer]\ntick_s = {tick_s}\n[upload]\nmin_interval_s = {min_interval_s}\nfail_limit = {fail_limit}\n"
    );
    std::fs::write(&path, text).expect("config");
    Config::load_validated(&path).expect("valid")
}

fn snapshot(mono: Instant, wall: SystemTime) -> Snapshot {
    Snapshot {
        t_mono: mono,
        t_wall: wall,
        load: Some(12.0),
        activity: None,
        cpu_pct: Some(10.0),
        cpu_topk_pct: Some(12.0),
        gpu_pct: Some(8.0),
        mem_pct: Some(40.0),
        coolant_c: Some(36.0),
        cpu_c: Some(42.0),
        gpu_c: Some(51.0),
        ai: AiState::Idle,
        models: Vec::new(),
        tokens: None,
        errors: BTreeSet::new(),
    }
}

#[derive(Clone)]
struct SharedLcd(Arc<Mutex<FakeLcd>>);

impl SharedLcd {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(FakeLcd::new())))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, FakeLcd> {
        self.0.lock().unwrap_or_else(|err| err.into_inner())
    }
}

impl LcdSink for SharedLcd {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        self.lock().show(frame)
    }

    fn restore_stock(&mut self) {
        self.lock().restore_stock();
    }

    fn needs_reupload(&mut self) {
        self.lock().needs_reupload();
    }

    fn tick(&mut self) -> Result<(), SinkError> {
        self.lock().tick()
    }

    fn blocked(&mut self) -> bool {
        false
    }
}

struct FakeClock {
    mono: Arc<Mutex<Instant>>,
    wall: Arc<Mutex<SystemTime>>,
}

impl FakeClock {
    fn new() -> Self {
        Self {
            mono: Arc::new(Mutex::new(Instant::now())),
            wall: Arc::new(Mutex::new(
                SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000),
            )),
        }
    }

    fn elapsed_since(&self, start: Instant) -> Duration {
        self.mono().saturating_duration_since(start)
    }
}

impl Clock for FakeClock {
    fn mono(&self) -> Instant {
        *self.mono.lock().unwrap_or_else(|err| err.into_inner())
    }

    fn wall(&self) -> SystemTime {
        *self.wall.lock().unwrap_or_else(|err| err.into_inner())
    }

    fn sleep(&mut self, d: Duration) {
        *self.mono.lock().unwrap_or_else(|err| err.into_inner()) += d;
        *self.wall.lock().unwrap_or_else(|err| err.into_inner()) += d;
    }
}

struct Repeat(fn(&mut Snapshot));

impl Sampler for Repeat {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        let mut snap = snapshot(mono, wall);
        (self.0)(&mut snap);
        snap
    }
}

struct Idle;

impl Sampler for Idle {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        snapshot(mono, wall)
    }
}

#[derive(Default)]
struct CaptureNotify {
    events: Vec<&'static str>,
}

impl Notifier for CaptureNotify {
    fn ready(&mut self) {
        self.events.push("READY");
    }

    fn watchdog(&mut self) {
        self.events.push("WATCHDOG");
    }

    fn stopping(&mut self) {
        self.events.push("STOPPING");
    }
}

struct After {
    left: std::cell::Cell<u32>,
}

impl After {
    fn ticks(n: u32) -> Self {
        Self {
            left: std::cell::Cell::new(n),
        }
    }
}

impl Stop for After {
    fn requested(&self) -> bool {
        let n = self.left.get();
        if n == 0 {
            true
        } else {
            self.left.set(n - 1);
            false
        }
    }
}

#[derive(Default)]
struct CaptureLog {
    lines: Arc<Mutex<Vec<String>>>,
}

impl Clone for CaptureLog {
    fn clone(&self) -> Self {
        Self {
            lines: Arc::clone(&self.lines),
        }
    }
}

impl log::Sink for CaptureLog {
    fn write_line(&mut self, line: &str) {
        self.lines
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(line.to_owned());
    }
}

impl CaptureLog {
    fn lines(&self) -> Vec<String> {
        self.lines
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone()
    }

    fn crits(&self) -> Vec<String> {
        self.lines()
            .into_iter()
            .filter(|line| line.starts_with(log::prefix(Priority::Crit)))
            .collect()
    }
}

struct Opens {
    lcd: SharedLcd,
    script: VecDeque<Result<(), SinkError>>,
    calls: Arc<Mutex<u32>>,
    times: Arc<Mutex<Vec<Instant>>>,
    mono: Arc<Mutex<Instant>>,
}

impl Opens {
    fn ok(lcd: SharedLcd, mono: Arc<Mutex<Instant>>) -> Self {
        Self {
            lcd,
            script: VecDeque::new(),
            calls: Arc::new(Mutex::new(0)),
            times: Arc::new(Mutex::new(Vec::new())),
            mono,
        }
    }

    fn with_script(
        lcd: SharedLcd,
        script: Vec<Result<(), SinkError>>,
        mono: Arc<Mutex<Instant>>,
    ) -> Self {
        Self {
            lcd,
            script: script.into(),
            calls: Arc::new(Mutex::new(0)),
            times: Arc::new(Mutex::new(Vec::new())),
            mono,
        }
    }

    fn count(&self) -> u32 {
        *self.calls.lock().unwrap_or_else(|err| err.into_inner())
    }

    fn delays(&self) -> Vec<Duration> {
        let times = self.times.lock().unwrap_or_else(|err| err.into_inner());
        times
            .windows(2)
            .map(|pair| pair[1].saturating_duration_since(pair[0]))
            .collect()
    }

    fn open(&mut self) -> Result<SharedLcd, SinkError> {
        *self.calls.lock().unwrap_or_else(|err| err.into_inner()) += 1;
        let now = *self.mono.lock().unwrap_or_else(|err| err.into_inner());
        self.times
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(now);
        match self.script.pop_front() {
            Some(Err(err)) => Err(err),
            Some(Ok(())) | None => Ok(self.lcd.clone()),
        }
    }
}

fn show_count(lcd: &SharedLcd) -> usize {
    lcd.lock()
        .records()
        .iter()
        .filter(|record| matches!(record, Record::Cmd(_) | Record::Bulk(_)))
        .count()
}

fn liquid_count(lcd: &SharedLcd) -> usize {
    lcd.lock()
        .records()
        .iter()
        .filter(|record| {
            matches!(record, Record::Cmd(cmd) if {
                use kraken_lcd::device::proto::Cmd;
                *cmd == Cmd::ShowLiquid
            })
        })
        .count()
}

struct Harness {
    _scratch: Scratch,
    config: ValidConfig,
    lcd: SharedLcd,
    opens: Opens,
    clock: FakeClock,
    notify: CaptureNotify,
    log: CaptureLog,
}

impl Harness {
    fn new(label: &str, min_interval_s: u64, fail_limit: u32, tick_s: f64) -> Self {
        let scratch = Scratch::new(label);
        let config = load_config(&scratch.0, min_interval_s, fail_limit, tick_s);
        let lcd = SharedLcd::new();
        let clock = FakeClock::new();
        let opens = Opens::ok(lcd.clone(), Arc::clone(&clock.mono));
        Self {
            _scratch: scratch,
            config,
            lcd,
            opens,
            clock,
            notify: CaptureNotify::default(),
            log: CaptureLog::default(),
        }
    }

    fn run<Samp: Sampler>(
        &mut self,
        sampler: Samp,
        ticks: u32,
        latch: bool,
    ) -> Result<LoopExit, kraken_lcd::render::AssetError> {
        let mut assets = kraken_lcd::render::Assets::load()?;
        let config = &self.config;
        let clock = &mut self.clock;
        let notify = &mut self.notify;
        let log = self.log.clone();
        let opens = &mut self.opens;
        let exit = service::run_loop(LoopInput {
            config,
            sampler,
            clock,
            notify,
            stop: After::ticks(ticks),
            log,
            open: || opens.open(),
            assets: &mut assets,
            latch_at_start: latch,
        });
        Ok(exit)
    }
}

#[test]
fn opening_to_ours() {
    let mut h = Harness::new("opening", 10, 3, 1.0);
    let exit = h.run(Idle, 1, false).expect("assets");
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(h.opens.count(), 1);
    assert!(show_count(&h.lcd) > 0, "first frame uploaded");
    assert_eq!(h.notify.events[0], "READY");
    assert!(h.notify.events.contains(&"WATCHDOG"));
    assert_eq!(*h.notify.events.last().expect("stop"), "STOPPING");
}

#[test]
fn llama_swap_down() {
    let mut h = Harness::new("llama-down", 10, 3, 1.0);
    let exit = h
        .run(
            Repeat(|snap| {
                snap.ai = AiState::Down;
            }),
            1,
            false,
        )
        .expect("assets");
    assert_eq!(exit, LoopExit::Stopped);
    assert!(show_count(&h.lcd) > 0);
}

#[test]
fn reachable_zero_models() {
    let mut h = Harness::new("idle-models", 10, 3, 1.0);
    let exit = h.run(Idle, 1, false).expect("assets");
    assert_eq!(exit, LoopExit::Stopped);
    assert!(show_count(&h.lcd) > 0);
    let _ = Ai::Idle;
}

#[test]
fn one_source_fails() {
    let mut h = Harness::new("one-source", 10, 3, 1.0);
    let exit = h
        .run(
            Repeat(|snap| {
                snap.gpu_pct = None;
                snap.gpu_c = None;
                snap.errors.insert(SourceId::Gpu);
            }),
            1,
            false,
        )
        .expect("assets");
    assert_eq!(exit, LoopExit::Stopped);
    assert!(show_count(&h.lcd) > 0);
}

#[test]
fn host_source_failures_do_not_exit() {
    let mut h = Harness::new("host-fail", 10, 3, 1.0);
    let exit = h
        .run(
            Repeat(|snap| {
                snap.load = None;
                snap.cpu_pct = None;
                snap.cpu_topk_pct = None;
                snap.gpu_pct = None;
                snap.mem_pct = None;
                snap.coolant_c = None;
                snap.cpu_c = None;
                snap.gpu_c = None;
                for id in [
                    SourceId::ProcCpu,
                    SourceId::ProcMem,
                    SourceId::HwmonCoolant,
                    SourceId::HwmonCpu,
                    SourceId::Gpu,
                ] {
                    snap.errors.insert(id);
                }
            }),
            5,
            false,
        )
        .expect("assets");
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(liquid_count(&h.lcd), 0, "host failure is not watch down");
}

#[test]
fn upload_failed_hits_fail_limit() {
    let mut h = Harness::new("upload-fail", 10, 3, 1.0);
    {
        let mut lcd = h.lcd.lock();
        lcd.script(Err(SinkError::UploadFailed(
            kraken_lcd::device::UploadFailed::NoReply(kraken_lcd::device::Step::PreTransfer),
        )));
        lcd.script(Err(SinkError::UploadFailed(
            kraken_lcd::device::UploadFailed::Refused(kraken_lcd::device::Step::SetupBucket),
        )));
        lcd.script(Err(SinkError::UploadFailed(
            kraken_lcd::device::UploadFailed::NoReply(kraken_lcd::device::Step::PreTransfer),
        )));
    }
    let exit = h.run(Idle, 25, false).expect("assets");
    assert_eq!(exit, LoopExit::Restored);
    assert_eq!(liquid_count(&h.lcd), 1);
}

#[test]
fn transfer_aborted_detaches_and_counts() {
    let mut h = Harness::new("abort", 10, 3, 1.0);
    {
        let mut lcd = h.lcd.lock();
        lcd.script(Err(SinkError::TransferAborted));
        lcd.script(Err(SinkError::TransferAborted));
        lcd.script(Err(SinkError::TransferAborted));
    }
    let exit = h.run(Idle, 40, false).expect("assets");
    assert_eq!(exit, LoopExit::Restored);
    assert!(h.opens.count() >= 2, "DETACHED reopens after backoff");
    assert!(liquid_count(&h.lcd) >= 1);
}

#[test]
fn cooling_guard_deviation_halts() {
    let mut h = Harness::new("halt", 10, 3, 1.0);
    h.lcd.lock().script_tick(Err(SinkError::Halted));
    let after_halt = Arc::new(Mutex::new(0u32));
    let watch = HaltWatch {
        inner: h.lcd.clone(),
        halted: Arc::new(Mutex::new(false)),
        after: Arc::clone(&after_halt),
    };
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let config = &h.config;
    let clock = &mut h.clock;
    let notify = &mut h.notify;
    let log = h.log.clone();
    let mut opened = false;
    let exit = service::run_loop(LoopInput {
        config,
        sampler: Idle,
        clock,
        notify,
        stop: After::ticks(4),
        log,
        open: || {
            if opened {
                *after_halt.lock().unwrap_or_else(|err| err.into_inner()) += 1;
            }
            opened = true;
            Ok(watch.clone())
        },
        assets: &mut assets,
        latch_at_start: false,
    });
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(liquid_count(&h.lcd), 0, "HALTED never sends ShowLiquid");
    assert_eq!(
        *after_halt.lock().unwrap_or_else(|err| err.into_inner()),
        0,
        "no device I/O after the halt in the same run"
    );
    assert!(h.notify.events.iter().filter(|e| **e == "WATCHDOG").count() >= 2);
}

#[derive(Clone)]
struct HaltWatch {
    inner: SharedLcd,
    halted: Arc<Mutex<bool>>,
    after: Arc<Mutex<u32>>,
}

impl HaltWatch {
    fn note_after(&self) {
        if *self.halted.lock().unwrap_or_else(|err| err.into_inner()) {
            *self.after.lock().unwrap_or_else(|err| err.into_inner()) += 1;
        }
    }

    fn trip(&self) {
        *self.halted.lock().unwrap_or_else(|err| err.into_inner()) = true;
    }
}

impl LcdSink for HaltWatch {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        self.note_after();
        let result = self.inner.show(frame);
        if matches!(result, Err(SinkError::Halted)) {
            self.trip();
        }
        result
    }

    fn restore_stock(&mut self) {
        self.note_after();
        self.inner.restore_stock();
    }

    fn needs_reupload(&mut self) {
        self.note_after();
        self.inner.needs_reupload();
    }

    fn tick(&mut self) -> Result<(), SinkError> {
        self.note_after();
        let result = self.inner.tick();
        if matches!(result, Err(SinkError::Halted)) {
            self.trip();
        }
        result
    }

    fn blocked(&mut self) -> bool {
        false
    }
}

#[test]
fn device_unavailable_detaches_with_backoff() {
    let mut h = Harness::new("gone", 10, 3, 1.0);
    h.opens = Opens::with_script(
        h.lcd.clone(),
        vec![
            Ok(()),
            Err(SinkError::DeviceUnavailable),
            Err(SinkError::DeviceUnavailable),
            Ok(()),
        ],
        Arc::clone(&h.clock.mono),
    );
    h.lcd.lock().script(Err(SinkError::DeviceUnavailable));
    let exit = h.run(Idle, 80, false).expect("assets");
    assert_eq!(exit, LoopExit::Stopped);
    let delays = h.opens.delays();
    assert!(
        delays.len() >= 3,
        "open times={delays:?} calls={}",
        h.opens.count()
    );
    assert_eq!(delays[0], Duration::from_secs(10));
    assert_eq!(delays[1], Duration::from_secs(20));
    assert_eq!(delays[2], Duration::from_secs(40));
}

#[test]
fn device_in_bootloader_collect_only() {
    let mut h = Harness::new("bootloader", 10, 3, 1.0);
    h.opens = Opens::with_script(
        h.lcd.clone(),
        vec![Err(SinkError::DeviceInBootloader)],
        Arc::clone(&h.clock.mono),
    );
    let exit = h.run(Idle, 3, false).expect("assets");
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(h.opens.count(), 1, "BOOTLOADER does not retry open");
    assert_eq!(show_count(&h.lcd), 0);
    assert_eq!(h.log.crits().len(), 1);
    assert!(h.notify.events.iter().filter(|e| **e == "WATCHDOG").count() >= 3);
}

#[test]
fn wall_clock_jump_detaches() {
    let mut h = Harness::new("suspend", 60, 3, 1.0);
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let clock_wall = Arc::clone(&h.clock.wall);
    let config = &h.config;
    let clock = &mut h.clock;
    let notify = &mut h.notify;
    let log = h.log.clone();
    let opens = &mut h.opens;
    let exit = service::run_loop(LoopInput {
        config,
        sampler: JumpWall {
            once: std::cell::Cell::new(false),
            hold: std::cell::Cell::new(0),
            amount: Duration::from_secs(7),
            wall: clock_wall,
        },
        clock,
        notify,
        stop: After::ticks(15),
        log,
        open: || opens.open(),
        assets: &mut assets,
        latch_at_start: false,
    });
    assert_eq!(exit, LoopExit::Stopped);
    assert!(
        h.opens.count() >= 2,
        "resume reopens, calls={}",
        h.opens.count()
    );
    let lines = h.log.lines();
    assert!(
        lines.iter().any(|line| line.contains("marking gap")),
        "resume logs mark_gap: {lines:?}"
    );
    let slots = h
        .lcd
        .lock()
        .records()
        .iter()
        .filter(|record| {
            matches!(record, Record::Cmd(cmd) if {
                use kraken_lcd::device::proto::Cmd;
                matches!(cmd, Cmd::ShowSlot(_))
            })
        })
        .count();
    assert_eq!(
        slots, 1,
        "forced upload after resume still waits out min_interval"
    );
}

fn run_clock_shift(
    h: &mut Harness,
    ticks: u32,
    mono_shift: Duration,
    wall_shift: Duration,
) -> LoopExit {
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let config = &h.config;
    let clock = &mut h.clock;
    let notify = &mut h.notify;
    let log = h.log.clone();
    let opens = &mut h.opens;
    let sampler = ClockShift {
        once: std::cell::Cell::new(false),
        mono_shift,
        wall_shift,
        mono: Arc::clone(&clock.mono),
        wall: Arc::clone(&clock.wall),
    };
    service::run_loop(LoopInput {
        config,
        sampler,
        clock,
        notify,
        stop: After::ticks(ticks),
        log,
        open: || opens.open(),
        assets: &mut assets,
        latch_at_start: false,
    })
}

/// Wall and mono both advance 10 s: a slow tick, not a suspend.
#[test]
fn slow_tick_is_not_a_resume() {
    let mut h = Harness::new("slow-tick", 60, 3, 2.0);
    let exit = run_clock_shift(&mut h, 2, Duration::from_secs(10), Duration::from_secs(10));
    assert_eq!(exit, LoopExit::Stopped);
    let gaps: Vec<_> = h
        .log
        .lines()
        .into_iter()
        .filter(|line| line.contains("marking gap"))
        .collect();
    assert!(gaps.is_empty(), "slow tick with no suspend: {gaps:?}");
}

/// 10 s of monotonic work plus a 30 s wall-only suspend in the same interval.
/// The gap is `wall_delta − mono_delta` = 30 s, not `wall_delta − tick`.
#[test]
fn long_tick_then_30s_suspend_is_a_resume() {
    let mut h = Harness::new("work-then-suspend", 60, 3, 2.0);
    let exit = run_clock_shift(&mut h, 2, Duration::from_secs(10), Duration::from_secs(40));
    assert_eq!(exit, LoopExit::Stopped);
    let gaps: Vec<_> = h
        .log
        .lines()
        .into_iter()
        .filter(|line| line.contains("marking gap"))
        .collect();
    assert_eq!(gaps.len(), 1, "30s suspend after a 10s tick: {gaps:?}");
    assert!(
        gaps[0].contains("marking gap 30s"),
        "gap is wall_delta − mono_delta, got {}",
        gaps[0]
    );
}

struct ClockShift {
    once: std::cell::Cell<bool>,
    mono_shift: Duration,
    wall_shift: Duration,
    mono: Arc<Mutex<Instant>>,
    wall: Arc<Mutex<SystemTime>>,
}

impl Sampler for ClockShift {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        let snap = snapshot(mono, wall);
        if !self.once.replace(true) {
            *self.mono.lock().unwrap_or_else(|err| err.into_inner()) += self.mono_shift;
            *self.wall.lock().unwrap_or_else(|err| err.into_inner()) += self.wall_shift;
        }
        snap
    }
}

struct JumpWall {
    once: std::cell::Cell<bool>,
    /// Samples to skip before the wall jump. The follow-up halt is 2 s after
    /// open, so a jump on the first sample detaches before that halt.
    hold: std::cell::Cell<u32>,
    amount: Duration,
    wall: Arc<Mutex<SystemTime>>,
}

impl Sampler for JumpWall {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        let snap = snapshot(mono, wall);
        let hold = self.hold.get();
        if hold > 0 {
            self.hold.set(hold - 1);
        } else if !self.once.replace(true) {
            let mut stored = self.wall.lock().unwrap_or_else(|err| err.into_inner());
            *stored += self.amount;
        }
        snap
    }
}

#[test]
fn first_frame_after_open_uploads_at_once() {
    let mut h = Harness::new("first-frame", 60, 3, 1.0);
    let exit = h.run(Idle, 1, false).expect("assets");
    assert_eq!(exit, LoopExit::Stopped);
    assert!(show_count(&h.lcd) > 0, "CAM overwrite is immediate");
}

#[test]
fn forced_reopen_upload_respects_min_interval() {
    let mut h = Harness::new("gap", 60, 3, 1.0);
    h.lcd.lock().script_tick(Err(SinkError::DeviceUnavailable));
    let exit = h.run(Idle, 30, false).expect("assets");
    assert_eq!(exit, LoopExit::Stopped);
    // t=0 upload, t=1 unavailable → DETACHED, reopen at t=11. min_interval 60
    // so the forced upload must still Wait. Only the first success is in the log.
    let slots = h
        .lcd
        .lock()
        .records()
        .iter()
        .filter(|record| {
            matches!(record, Record::Cmd(cmd) if {
                use kraken_lcd::device::proto::Cmd;
                matches!(cmd, Cmd::ShowSlot(_))
            })
        })
        .count();
    assert_eq!(slots, 1, "reopen inside min_interval must not upload");
}

#[test]
fn fence_violation_exits_2_without_restore() {
    let mut h = Harness::new("fence", 10, 3, 1.0);
    h.lcd.lock().script(Err(SinkError::Fence));
    let exit = h.run(Idle, 3, false).expect("assets");
    assert_eq!(exit, LoopExit::Fence);
    assert_eq!(exit.code(), 2);
    assert_eq!(liquid_count(&h.lcd), 0);
    assert_eq!(h.log.crits().len(), 1);
}

#[test]
fn fatal_exits_nonzero_without_retry() {
    let mut h = Harness::new("fatal", 10, 3, 1.0);
    h.opens = Opens::with_script(
        h.lcd.clone(),
        vec![Err(SinkError::Fatal("post-open sysfs"))],
        Arc::clone(&h.clock.mono),
    );
    let exit = h.run(Idle, 5, false).expect("assets");
    assert_eq!(exit, LoopExit::Fatal);
    assert_eq!(exit.code(), 2);
    assert_eq!(h.opens.count(), 1, "Fatal is not retried");
    assert_eq!(liquid_count(&h.lcd), 0);
    assert_eq!(h.log.crits().len(), 1);
}

#[test]
fn watchdog_follows_a_completed_tick() {
    let mut h = Harness::new("watchdog", 10, 3, 1.0);
    let _ = h.run(Idle, 2, false).expect("assets");
    let wd = h.notify.events.iter().filter(|e| **e == "WATCHDOG").count();
    assert_eq!(wd, 2);
}

#[test]
fn injected_stop_sends_stopping_then_exits_0() {
    let mut h = Harness::new("injected-stop", 10, 3, 1.0);
    let exit = h.run(Idle, 1, false).expect("assets");
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(exit.code(), 0);
    assert_eq!(*h.notify.events.last().expect("end"), "STOPPING");
}

fn run_jumping(h: &mut Harness, ticks: u32, latch: bool, amount: Duration, hold: u32) -> LoopExit {
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let clock_wall = Arc::clone(&h.clock.wall);
    let config = &h.config;
    let clock = &mut h.clock;
    let notify = &mut h.notify;
    let log = h.log.clone();
    let opens = &mut h.opens;
    service::run_loop(LoopInput {
        config,
        sampler: JumpWall {
            once: std::cell::Cell::new(false),
            hold: std::cell::Cell::new(hold),
            amount,
            wall: clock_wall,
        },
        clock,
        notify,
        stop: After::ticks(ticks),
        log,
        open: || opens.open(),
        assets: &mut assets,
        latch_at_start: latch,
    })
}

#[test]
fn latch_at_start_ignores_a_wall_jump() {
    let mut h = Harness::new("latch-jump", 10, 3, 1.0);
    let exit = run_jumping(&mut h, 25, true, Duration::from_secs(120), 0);
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(
        h.opens.count(),
        0,
        "HALTED is terminal; resume does not open"
    );
}

#[test]
fn bootloader_stays_bootloader_across_a_wall_jump() {
    let mut h = Harness::new("boot-jump", 10, 3, 1.0);
    h.opens = Opens::with_script(
        h.lcd.clone(),
        vec![Err(SinkError::DeviceInBootloader)],
        Arc::clone(&h.clock.mono),
    );
    let exit = run_jumping(&mut h, 25, false, Duration::from_secs(120), 0);
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(
        h.opens.count(),
        1,
        "BOOTLOADER is terminal; resume does not reopen"
    );
}

#[test]
fn guard_halt_ignores_a_wall_jump() {
    let mut h = Harness::new("halt-jump", 10, 3, 1.0);
    h.lcd.lock().script_tick(Err(SinkError::Halted));
    let exit = run_jumping(&mut h, 25, false, Duration::from_secs(120), 3);
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(
        h.opens.count(),
        1,
        "HALTED is terminal; resume does not reopen"
    );
}

#[test]
fn failed_uploads_are_rate_limited_over_ten_minutes() {
    let mut h = Harness::new("fail-pace", 60, 3, 2.0);
    let shows = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let start = h.clock.mono();
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let config = &h.config;
    let clock = &mut h.clock;
    let notify = &mut h.notify;
    let log = h.log.clone();
    let lcd = AlwaysFail {
        shows: Arc::clone(&shows),
    };
    let exit = service::run_loop(LoopInput {
        config,
        sampler: Idle,
        clock,
        notify,
        stop: After::ticks(300),
        log,
        open: {
            let lcd = lcd.clone();
            move || Ok(lcd.clone())
        },
        assets: &mut assets,
        latch_at_start: false,
    });
    assert_eq!(exit, LoopExit::Restored);
    let attempts = shows.load(std::sync::atomic::Ordering::Relaxed);
    assert!(attempts <= 11, "attempts={attempts}");
    assert_eq!(attempts, 3, "fail_limit 3 on a 60s gate");
    assert!(
        h.clock.elapsed_since(start) >= Duration::from_secs(120),
        "fail_limit fires on the spaced schedule, elapsed={:?}",
        h.clock.elapsed_since(start)
    );
    assert!(
        h.log.crits().iter().any(|line| line.contains("fail_limit")),
        "Restored logs CRITICAL with the reason: {:?}",
        h.log.crits()
    );
}

#[derive(Clone)]
struct AlwaysFail {
    shows: Arc<std::sync::atomic::AtomicU32>,
}

impl LcdSink for AlwaysFail {
    fn show(&mut self, _frame: &Frame) -> Result<(), SinkError> {
        self.shows
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Err(SinkError::UploadFailed(
            kraken_lcd::device::UploadFailed::NoReply(kraken_lcd::device::Step::PreTransfer),
        ))
    }

    fn restore_stock(&mut self) {}

    fn needs_reupload(&mut self) {}

    fn tick(&mut self) -> Result<(), SinkError> {
        Ok(())
    }

    fn blocked(&mut self) -> bool {
        false
    }
}
