//! Stream mode: deadline pacing, ping-pong slots, and the unchanged safety gates.
//!
//! Fake device and clock only. Change mode is not retuned here.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use kraken_lcd::config::{Config, ValidConfig};
use kraken_lcd::device::proto::Cmd;
use kraken_lcd::device::{FakeLcd, LcdSink, Record, SinkError, UploadFailed};
use kraken_lcd::log;
use kraken_lcd::render::Frame;
use kraken_lcd::service::{self, Clock, LoopExit, LoopInput, Notifier, Sampler, Stop};
use llama_core::sample::{AiState, Snapshot};

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("t36-{label}-{}", std::process::id()));
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

fn load_config(dir: &Path, text: &str) -> ValidConfig {
    let path = dir.join("config.toml");
    std::fs::write(&path, text).expect("config");
    Config::load_validated(&path).expect("valid")
}

fn stream_config(dir: &Path, fps: u8, min_interval_s: u64, tick_s: f64) -> ValidConfig {
    load_config(
        dir,
        &format!(
            "[writer]\ntick_s = {tick_s}\n[upload]\nmode = \"stream\"\nstream_fps = {fps}\nmin_interval_s = {min_interval_s}\nfail_limit = 3\n"
        ),
    )
}

fn snap(mono: Instant, wall: SystemTime, load: f32, cpu: f32, fresh: bool) -> Snapshot {
    Snapshot {
        t_mono: mono,
        t_wall: wall,
        load: fresh.then_some(load),
        activity: None,
        cpu_pct: fresh.then_some(cpu),
        cpu_topk_pct: fresh.then_some(load),
        gpu_pct: fresh.then_some(8.0),
        mem_pct: fresh.then_some(40.0),
        coolant_c: fresh.then_some(36.0),
        cpu_c: fresh.then_some(42.0),
        gpu_c: fresh.then_some(51.0),
        ai: if fresh {
            AiState::Idle
        } else {
            AiState::NoData
        },
        models: Vec::new(),
        tokens: None,
        errors: BTreeSet::new(),
    }
}

struct FakeClock {
    mono: Arc<Mutex<Instant>>,
    wall: Arc<Mutex<SystemTime>>,
}

impl FakeClock {
    fn new() -> (Self, Instant) {
        let origin = Instant::now();
        let clock = Self {
            mono: Arc::new(Mutex::new(origin)),
            wall: Arc::new(Mutex::new(
                SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000),
            )),
        };
        (clock, origin)
    }

    fn advance(&self, d: Duration) {
        *self.mono.lock().unwrap_or_else(|err| err.into_inner()) += d;
        *self.wall.lock().unwrap_or_else(|err| err.into_inner()) += d;
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
        self.advance(d);
    }
}

struct After(std::cell::Cell<u32>);

impl After {
    fn ticks(n: u32) -> Self {
        Self(std::cell::Cell::new(n))
    }
}

impl Stop for After {
    fn requested(&self) -> bool {
        let n = self.0.get();
        if n == 0 {
            true
        } else {
            self.0.set(n - 1);
            false
        }
    }
}

struct Quiet;

impl Notifier for Quiet {
    fn ready(&mut self) {}
    fn watchdog(&mut self) {}
    fn stopping(&mut self) {}
}

struct CaptureLog(Arc<Mutex<Vec<String>>>);

impl log::Sink for CaptureLog {
    fn write_line(&mut self, line: &str) {
        self.0
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(line.to_owned());
    }
}

impl log::Sink for &mut CaptureLog {
    fn write_line(&mut self, line: &str) {
        (*self).write_line(line);
    }
}

fn quiet_log() -> CaptureLog {
    CaptureLog(Arc::new(Mutex::new(Vec::new())))
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
    fn show_slot(&mut self, slot: u8, frame: &Frame) -> Result<(), SinkError> {
        self.lock().show_slot(slot, frame)
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
        self.lock().blocked()
    }
}

fn slots_of(lcd: &SharedLcd) -> Vec<u8> {
    lcd.lock()
        .records()
        .iter()
        .filter_map(|record| match record {
            Record::Cmd(Cmd::ShowSlot(slot)) => Some(slot.get()),
            _ => None,
        })
        .collect()
}

fn bulks_of(lcd: &SharedLcd) -> Vec<Vec<u8>> {
    lcd.lock()
        .records()
        .iter()
        .filter_map(|record| match record {
            Record::Bulk(bytes) => Some(bytes.clone()),
            _ => None,
        })
        .collect()
}

fn liquids_of(lcd: &SharedLcd) -> usize {
    lcd.lock()
        .records()
        .iter()
        .filter(|record| matches!(record, Record::Cmd(Cmd::ShowLiquid)))
        .count()
}

struct Steady {
    samples: Arc<Mutex<Vec<Duration>>>,
    origin: Instant,
    load_of: fn(u32) -> f32,
    cpu_of: fn(u32) -> f32,
    fresh: bool,
    n: std::cell::Cell<u32>,
}

impl Sampler for Steady {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        let at = mono.saturating_duration_since(self.origin);
        self.samples
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(at);
        let i = self.n.get();
        self.n.set(i.saturating_add(1));
        snap(mono, wall, (self.load_of)(i), (self.cpu_of)(i), self.fresh)
    }
}

fn run<S: LcdSink + Clone + 'static>(
    config: &ValidConfig,
    ticks: u32,
    sampler: impl Sampler,
    clock: &mut FakeClock,
    lcd: S,
    log: &mut CaptureLog,
) -> LoopExit {
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    service::run_loop(LoopInput {
        config,
        sampler,
        clock,
        notify: &mut Quiet,
        stop: After::ticks(ticks),
        log,
        open: move || Ok(lcd.clone()),
        assets: &mut assets,
        latch_at_start: false,
    })
}

#[test]
fn stream_paces_on_the_frame_deadline() {
    let scratch = Scratch::new("pace");
    let config = stream_config(&scratch.0, 10, 60, 2.0);
    let (mut clock, origin) = FakeClock::new();
    let samples = Arc::new(Mutex::new(Vec::new()));
    let exit = run(
        &config,
        5,
        Steady {
            samples: Arc::clone(&samples),
            origin,
            load_of: |_| 10.0,
            cpu_of: |_| 10.0,
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        &mut clock,
        SharedLcd::new(),
        &mut quiet_log(),
    );
    assert_eq!(exit, LoopExit::Stopped);
    let samples = samples.lock().unwrap_or_else(|err| err.into_inner());
    let period = Duration::from_millis(100);
    assert_eq!(samples.len(), 5, "one sample per frame: {samples:?}");
    for pair in samples.windows(2) {
        assert_eq!(
            pair[1] - pair[0],
            period,
            "deadline spacing, not writer.tick_s: {samples:?}"
        );
    }
}

#[test]
fn stream_overrun_starts_the_next_frame_without_a_burst() {
    let scratch = Scratch::new("overrun");
    let config = stream_config(&scratch.0, 10, 60, 2.0);
    let (mut clock, origin) = FakeClock::new();
    let starts = Arc::new(Mutex::new(Vec::new()));
    let stalled = Arc::new(Mutex::new(false));
    let lcd = StallLcd {
        inner: SharedLcd::new(),
        clock: clock.mono.clone(),
        wall: clock.wall.clone(),
        origin,
        starts: Arc::clone(&starts),
        stalled: Arc::clone(&stalled),
    };
    let exit = run(
        &config,
        4,
        Steady {
            samples: Arc::new(Mutex::new(Vec::new())),
            origin,
            load_of: |i| 20.0 + i as f32 * 30.0,
            cpu_of: |i| 20.0 + i as f32 * 30.0,
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        &mut clock,
        lcd,
        &mut quiet_log(),
    );
    assert_eq!(exit, LoopExit::Stopped);
    let starts = starts.lock().unwrap_or_else(|err| err.into_inner());
    assert_eq!(
        starts.len(),
        4,
        "one upload per frame while the picture changes: {starts:?}"
    );
    assert_eq!(
        starts[1] - starts[0],
        Duration::from_millis(500),
        "{starts:?}"
    );
    let period = Duration::from_millis(100);
    assert_eq!(
        starts[2] - starts[1],
        period,
        "no catch-up burst: {starts:?}"
    );
    assert_eq!(
        starts[3] - starts[2],
        period,
        "back on the deadline: {starts:?}"
    );
    let burst = starts.windows(2).filter(|pair| pair[1] == pair[0]).count();
    assert_eq!(burst, 0, "missed deadlines are not replayed: {starts:?}");
}

#[derive(Clone)]
struct StallLcd {
    inner: SharedLcd,
    clock: Arc<Mutex<Instant>>,
    wall: Arc<Mutex<SystemTime>>,
    origin: Instant,
    starts: Arc<Mutex<Vec<Duration>>>,
    stalled: Arc<Mutex<bool>>,
}

impl LcdSink for StallLcd {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        self.show_slot(0, frame)
    }
    fn show_slot(&mut self, slot: u8, frame: &Frame) -> Result<(), SinkError> {
        let now = *self.clock.lock().unwrap_or_else(|err| err.into_inner());
        self.starts
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(now.saturating_duration_since(self.origin));
        let mut stalled = self.stalled.lock().unwrap_or_else(|err| err.into_inner());
        if !*stalled {
            *stalled = true;
            *self.clock.lock().unwrap_or_else(|err| err.into_inner()) += Duration::from_millis(500);
            *self.wall.lock().unwrap_or_else(|err| err.into_inner()) += Duration::from_millis(500);
        }
        drop(stalled);
        self.inner.show_slot(slot, frame)
    }
    fn restore_stock(&mut self) {
        self.inner.restore_stock();
    }
    fn needs_reupload(&mut self) {
        self.inner.needs_reupload();
    }
    fn tick(&mut self) -> Result<(), SinkError> {
        self.inner.tick()
    }
    fn blocked(&mut self) -> bool {
        false
    }
}

#[test]
fn identical_frames_are_not_uploaded_again() {
    let scratch = Scratch::new("identical");
    let config = stream_config(&scratch.0, 10, 60, 2.0);
    let (mut clock, origin) = FakeClock::new();
    let lcd = SharedLcd::new();
    let exit = run(
        &config,
        15,
        Steady {
            samples: Arc::new(Mutex::new(Vec::new())),
            origin,
            load_of: |_| 40.0,
            cpu_of: |_| 10.0,
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        &mut clock,
        lcd.clone(),
        &mut quiet_log(),
    );
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(
        slots_of(&lcd),
        vec![0, 1],
        "the ring appearing is a new frame; repeats are skipped"
    );
}

#[test]
fn stream_ping_pongs_slots_zero_and_one() {
    let scratch = Scratch::new("pong");
    let config = stream_config(&scratch.0, 10, 60, 2.0);
    let (mut clock, origin) = FakeClock::new();
    let lcd = SharedLcd::new();
    let exit = run(
        &config,
        4,
        Steady {
            samples: Arc::new(Mutex::new(Vec::new())),
            origin,
            load_of: |i| i as f32 * 25.0,
            cpu_of: |i| i as f32 * 25.0,
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        &mut clock,
        lcd.clone(),
        &mut quiet_log(),
    );
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(slots_of(&lcd), vec![0, 1, 0, 1]);
    assert!(
        clock.mono().saturating_duration_since(origin) < Duration::from_secs(10),
        "min_interval_s must not space stream uploads"
    );
}

#[test]
fn guard_ticks_at_least_once_a_second() {
    let scratch = Scratch::new("guard-hz");
    let config = stream_config(&scratch.0, 10, 60, 2.0);
    let (mut clock, origin) = FakeClock::new();
    let guards = Arc::new(Mutex::new(Vec::new()));
    let lcd = GuardLcd {
        inner: SharedLcd::new(),
        clock: clock.mono.clone(),
        origin,
        guards: Arc::clone(&guards),
        halt_at: None,
    };
    let exit = run(
        &config,
        12,
        Steady {
            samples: Arc::new(Mutex::new(Vec::new())),
            origin,
            load_of: |_| 10.0,
            cpu_of: |_| 10.0,
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        &mut clock,
        lcd,
        &mut quiet_log(),
    );
    assert_eq!(exit, LoopExit::Stopped);
    let guards = guards.lock().unwrap_or_else(|err| err.into_inner());
    assert_eq!(
        guards.as_slice(),
        &[Duration::ZERO, Duration::from_secs(1)],
        "once a second across a 1.1 s stream, not once a frame and not the 2 s follow-up: {guards:?}"
    );
}

#[derive(Clone)]
struct GuardLcd {
    inner: SharedLcd,
    clock: Arc<Mutex<Instant>>,
    origin: Instant,
    guards: Arc<Mutex<Vec<Duration>>>,
    halt_at: Option<usize>,
}

impl LcdSink for GuardLcd {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        self.inner.show(frame)
    }
    fn show_slot(&mut self, slot: u8, frame: &Frame) -> Result<(), SinkError> {
        self.inner.show_slot(slot, frame)
    }
    fn restore_stock(&mut self) {
        self.inner.restore_stock();
    }
    fn needs_reupload(&mut self) {
        self.inner.needs_reupload();
    }
    fn tick(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
    fn pace_guard(&mut self) -> Result<(), SinkError> {
        let now = *self.clock.lock().unwrap_or_else(|err| err.into_inner());
        let at = now.saturating_duration_since(self.origin);
        let mut guards = self.guards.lock().unwrap_or_else(|err| err.into_inner());
        guards.push(at);
        let n = guards.len();
        drop(guards);
        if self.halt_at == Some(n) {
            self.inner.lock().halt();
            return Err(SinkError::Halted);
        }
        Ok(())
    }
    fn blocked(&mut self) -> bool {
        self.inner.blocked()
    }
}

#[test]
fn guard_trip_mid_stream_latches_and_stops() {
    let scratch = Scratch::new("guard-trip");
    let config = stream_config(&scratch.0, 10, 60, 2.0);
    let (mut clock, origin) = FakeClock::new();
    let guards = Arc::new(Mutex::new(Vec::new()));
    let lcd = GuardLcd {
        inner: SharedLcd::new(),
        clock: clock.mono.clone(),
        origin,
        guards: Arc::clone(&guards),
        halt_at: Some(2),
    };
    let mut log = quiet_log();
    let exit = run(
        &config,
        15,
        Steady {
            samples: Arc::new(Mutex::new(Vec::new())),
            origin,
            load_of: |i| i as f32 * 5.0,
            cpu_of: |i| i as f32 * 5.0,
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        &mut clock,
        lcd.clone(),
        &mut log,
    );
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(
        slots_of(&lcd.inner).len(),
        10,
        "uploads stop when the guard trips"
    );
    assert_eq!(
        liquids_of(&lcd.inner),
        0,
        "a guard trip sends no ShowLiquid"
    );
    let text = log
        .0
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .join("\n");
    assert!(
        text.contains("cooling guard halted"),
        "latch path logs the halt: {text}"
    );
}

#[test]
fn blocked_latch_mid_stream_stops_without_restore() {
    let scratch = Scratch::new("latch");
    let config = stream_config(&scratch.0, 10, 60, 2.0);
    let (mut clock, origin) = FakeClock::new();
    let samples = Arc::new(Mutex::new(Vec::new()));
    let lcd = LatchLcd {
        inner: SharedLcd::new(),
        shows: Arc::new(Mutex::new(0)),
    };
    let mut log = quiet_log();
    let exit = run(
        &config,
        6,
        Steady {
            samples: Arc::clone(&samples),
            origin,
            load_of: |i| i as f32 * 20.0,
            cpu_of: |i| i as f32 * 20.0,
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        &mut clock,
        lcd.clone(),
        &mut log,
    );
    assert_eq!(exit, LoopExit::Stopped);
    let samples = samples.lock().unwrap_or_else(|err| err.into_inner());
    assert_eq!(samples.len(), 6);
    assert_eq!(samples[1] - samples[0], Duration::from_millis(100));
    assert_eq!(slots_of(&lcd.inner), vec![0]);
    assert_eq!(liquids_of(&lcd.inner), 0);
    let text = log
        .0
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .join("\n");
    assert!(text.contains("cooling guard halted"), "{text}");
}

#[derive(Clone)]
struct LatchLcd {
    inner: SharedLcd,
    shows: Arc<Mutex<u32>>,
}

impl LcdSink for LatchLcd {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        self.show_slot(0, frame)
    }
    fn show_slot(&mut self, slot: u8, frame: &Frame) -> Result<(), SinkError> {
        *self.shows.lock().unwrap_or_else(|err| err.into_inner()) += 1;
        self.inner.show_slot(slot, frame)
    }
    fn restore_stock(&mut self) {
        self.inner.restore_stock();
    }
    fn needs_reupload(&mut self) {
        self.inner.needs_reupload();
    }
    fn tick(&mut self) -> Result<(), SinkError> {
        self.inner.tick()
    }
    fn blocked(&mut self) -> bool {
        *self.shows.lock().unwrap_or_else(|err| err.into_inner()) >= 1
    }
}

/// `service::STREAM_RETRY_START`, private to keep S11's public surface fixed.
const RETRY_START: Duration = Duration::from_secs(2);

fn refused(step: kraken_lcd::device::Step) -> Result<(), SinkError> {
    Err(SinkError::UploadFailed(UploadFailed::Refused(step)))
}

/// Records when each upload is attempted, then defers to the shared fake.
#[derive(Clone)]
struct TimedLcd {
    inner: SharedLcd,
    clock: Arc<Mutex<Instant>>,
    origin: Instant,
    attempts: Arc<Mutex<Vec<Duration>>>,
}

impl TimedLcd {
    fn new(clock: &FakeClock, origin: Instant) -> Self {
        Self {
            inner: SharedLcd::new(),
            clock: clock.mono.clone(),
            origin,
            attempts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn attempts(&self) -> Vec<Duration> {
        self.attempts
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone()
    }
}

impl LcdSink for TimedLcd {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        self.show_slot(0, frame)
    }
    fn show_slot(&mut self, slot: u8, frame: &Frame) -> Result<(), SinkError> {
        let now = *self.clock.lock().unwrap_or_else(|err| err.into_inner());
        self.attempts
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(now.saturating_duration_since(self.origin));
        self.inner.show_slot(slot, frame)
    }
    fn restore_stock(&mut self) {
        self.inner.restore_stock();
    }
    fn needs_reupload(&mut self) {
        self.inner.needs_reupload();
    }
    fn tick(&mut self) -> Result<(), SinkError> {
        self.inner.tick()
    }
    fn blocked(&mut self) -> bool {
        self.inner.blocked()
    }
}

#[test]
fn stream_fail_limit_restores_without_the_change_interval() {
    let scratch = Scratch::new("fail");
    let config = stream_config(&scratch.0, 10, 60, 2.0);
    let (mut clock, origin) = FakeClock::new();
    let lcd = TimedLcd::new(&clock, origin);
    for _ in 0..3 {
        lcd.inner
            .lock()
            .script(refused(kraken_lcd::device::Step::SetupBucket));
    }
    let mut log = quiet_log();
    let exit = run(
        &config,
        600,
        Steady {
            samples: Arc::new(Mutex::new(Vec::new())),
            origin,
            load_of: |i| (i % 5) as f32 * 20.0,
            cpu_of: |i| (i % 5) as f32 * 20.0,
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        &mut clock,
        lcd.clone(),
        &mut log,
    );
    assert_eq!(exit, LoopExit::Restored, "persistent failures still exit");
    assert_eq!(liquids_of(&lcd.inner), 1);
    assert!(slots_of(&lcd.inner).is_empty(), "nothing was ever shown");
    let attempts = lcd.attempts();
    assert_eq!(
        attempts.len(),
        3,
        "exactly fail_limit attempts: {attempts:?}"
    );
    assert_eq!(attempts[1] - attempts[0], RETRY_START, "{attempts:?}");
    assert_eq!(attempts[2] - attempts[1], RETRY_START * 2, "{attempts:?}");
    assert!(
        clock.mono().saturating_duration_since(origin) < Duration::from_secs(60),
        "failures are spaced by the retry backoff, not by min_interval_s"
    );
    let lines = log.0.lock().unwrap_or_else(|err| err.into_inner()).clone();
    assert!(
        lines.iter().any(|line| line.contains("upload failed")),
        "each failure is logged: {lines:?}"
    );
}

/// GitHub #14: the device refuses the first uploads right after open (the
/// previous process has just restored stock). The writer backs off, survives,
/// and then streams, without restoring stock or exiting.
#[test]
fn stream_survives_failures_right_after_open_then_streams() {
    let scratch = Scratch::new("settle");
    let config = stream_config(&scratch.0, 10, 60, 2.0);
    let (mut clock, origin) = FakeClock::new();
    let lcd = TimedLcd::new(&clock, origin);
    // fail_limit - 1 refusals in a row, the most a transient may cost.
    lcd.inner
        .lock()
        .script(refused(kraken_lcd::device::Step::PreTransfer));
    lcd.inner
        .lock()
        .script(refused(kraken_lcd::device::Step::DeleteBucket));
    let exit = run(
        &config,
        100,
        Steady {
            samples: Arc::new(Mutex::new(Vec::new())),
            origin,
            load_of: |i| (i % 5) as f32 * 20.0,
            cpu_of: |i| (i % 5) as f32 * 20.0,
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        &mut clock,
        lcd.clone(),
        &mut quiet_log(),
    );
    assert_eq!(exit, LoopExit::Stopped, "the writer survives");
    assert_eq!(liquids_of(&lcd.inner), 0, "no restore to stock");
    let attempts = lcd.attempts();
    assert!(attempts.len() > 10, "then it streams: {attempts:?}");
    assert_eq!(attempts[0], Duration::ZERO, "first upload right after open");
    assert_eq!(attempts[1] - attempts[0], RETRY_START);
    assert_eq!(attempts[2] - attempts[1], RETRY_START * 2);
    for pair in attempts[2..].windows(2) {
        assert_eq!(
            pair[1] - pair[0],
            Duration::from_millis(100),
            "after a success, back on the frame deadline: {attempts:?}"
        );
    }
    let slots = slots_of(&lcd.inner);
    assert_eq!(&slots[..4], &[0, 1, 0, 1], "ping-pong from slot 0");
}

/// A success resets the streak and the backoff, as in change mode.
#[test]
fn stream_success_resets_the_fail_streak() {
    let scratch = Scratch::new("reset");
    let config = stream_config(&scratch.0, 10, 60, 2.0);
    let (mut clock, origin) = FakeClock::new();
    let lcd = TimedLcd::new(&clock, origin);
    for script in [
        refused(kraken_lcd::device::Step::SetupBucket),
        refused(kraken_lcd::device::Step::SetupBucket),
        Ok(()),
        refused(kraken_lcd::device::Step::SetupBucket),
        refused(kraken_lcd::device::Step::SetupBucket),
    ] {
        lcd.inner.lock().script(script);
    }
    let exit = run(
        &config,
        200,
        Steady {
            samples: Arc::new(Mutex::new(Vec::new())),
            origin,
            load_of: |i| (i % 5) as f32 * 20.0,
            cpu_of: |i| (i % 5) as f32 * 20.0,
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        &mut clock,
        lcd.clone(),
        &mut quiet_log(),
    );
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(liquids_of(&lcd.inner), 0);
    let attempts = lcd.attempts();
    assert_eq!(
        attempts[4] - attempts[3],
        RETRY_START,
        "the backoff restarts after a success: {attempts:?}"
    );
}

struct Episode {
    origin: std::cell::Cell<Option<Instant>>,
}

impl Sampler for Episode {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        let start = self.origin.get().unwrap_or(mono);
        if self.origin.get().is_none() {
            self.origin.set(Some(mono));
        }
        let elapsed = mono.saturating_duration_since(start);
        let fresh = elapsed.is_zero()
            || (elapsed >= Duration::from_secs(7) && elapsed < Duration::from_secs(8));
        snap(mono, wall, 22.0, 11.0, fresh)
    }
}

#[test]
fn watch_down_in_stream_restores_on_its_own_budget() {
    let scratch = Scratch::new("watch");
    let config = load_config(
        &scratch.0,
        "[writer]\ntick_s = 1.0\n[upload]\nmode = \"stream\"\nstream_fps = 1\nmin_interval_s = 60\nfail_limit = 3\n[snapshot]\nwatch_down_stock_after_s = 5\nwatch_down_restore_min_s = 60\n",
    );
    let (mut clock, origin) = FakeClock::new();
    let lcd = SharedLcd::new();
    let exit = run(
        &config,
        15,
        Episode {
            origin: std::cell::Cell::new(None),
        },
        &mut clock,
        lcd.clone(),
        &mut quiet_log(),
    );
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(
        liquids_of(&lcd),
        1,
        "one ShowLiquid inside the restore budget"
    );
    assert!(
        slots_of(&lcd).len() >= 2,
        "the stale snapshot is drawn as no data before stock"
    );
    let elapsed = clock.mono().saturating_duration_since(origin);
    assert!(
        elapsed < Duration::from_secs(20),
        "stock does not wait out min_interval_s, elapsed {elapsed:?}"
    );
}

#[test]
fn stream_keeps_percent_hysteresis() {
    let scratch = Scratch::new("hysteresis");
    let config = stream_config(&scratch.0, 10, 60, 2.0);
    let (mut clock, origin) = FakeClock::new();
    let lcd = SharedLcd::new();
    let exit = run(
        &config,
        10,
        Steady {
            samples: Arc::new(Mutex::new(Vec::new())),
            origin,
            load_of: |_| 40.0,
            cpu_of: |i| if i < 5 { 10.0 } else { 11.0 },
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        &mut clock,
        lcd.clone(),
        &mut quiet_log(),
    );
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(
        slots_of(&lcd).len(),
        2,
        "cpu 10 then 11 stays on the shown percent; the ring still appears"
    );
}

#[test]
fn stream_ring_updates_inside_the_hold_band() {
    let scratch = Scratch::new("ring");
    let (mut clock, origin) = FakeClock::new();
    let stream_lcd = SharedLcd::new();
    let stream_cfg = stream_config(&scratch.0, 10, 60, 0.5);
    let exit = run(
        &stream_cfg,
        14,
        Steady {
            samples: Arc::new(Mutex::new(Vec::new())),
            origin,
            load_of: |i| if i < 5 { 10.0 } else { 14.0 },
            cpu_of: |_| 10.0,
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        &mut clock,
        stream_lcd.clone(),
        &mut quiet_log(),
    );
    assert_eq!(exit, LoopExit::Stopped);
    let stream_frames = bulks_of(&stream_lcd);

    let change_dir = Scratch::new("ring-change");
    let change_cfg = load_config(
        &change_dir.0,
        "[writer]\ntick_s = 0.5\n[upload]\nmode = \"change\"\nstream_fps = 10\nmin_interval_s = 10\nfail_limit = 3\n",
    );
    let (mut change_clock, change_origin) = FakeClock::new();
    let change_lcd = SharedLcd::new();
    let exit = run(
        &change_cfg,
        21,
        Steady {
            samples: Arc::new(Mutex::new(Vec::new())),
            origin: change_origin,
            load_of: |i| if i < 5 { 10.0 } else { 14.0 },
            cpu_of: |_| 10.0,
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        &mut change_clock,
        change_lcd.clone(),
        &mut quiet_log(),
    );
    assert_eq!(exit, LoopExit::Stopped);
    let change_frames = bulks_of(&change_lcd);
    assert_eq!(change_frames.len(), 2, "change mode stays on its interval");
    // T54: the stream ring is smoothed and drawn in 1-point steps, so the
    // live mean can slide through more than one step.
    assert!(
        stream_frames.len() >= 3,
        "empty ring, held step, then the live step(s): {}",
        stream_frames.len()
    );
    assert_eq!(
        stream_frames[1], change_frames[1],
        "the first drawn ring matches change mode"
    );
    assert_ne!(
        stream_frames[stream_frames.len() - 1],
        stream_frames[1],
        "a later mean inside the hold band still moves the stream ring"
    );
}

#[test]
fn bootloader_in_stream_sends_nothing() {
    let scratch = Scratch::new("boot");
    let config = stream_config(&scratch.0, 10, 60, 0.5);
    let (mut clock, _) = FakeClock::new();
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let opened = std::cell::Cell::new(false);
    let exit = service::run_loop(LoopInput {
        config: &config,
        sampler: Steady {
            samples: Arc::new(Mutex::new(Vec::new())),
            origin: clock.mono(),
            load_of: |_| 10.0,
            cpu_of: |_| 10.0,
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        clock: &mut clock,
        notify: &mut Quiet,
        stop: After::ticks(3),
        log: &mut quiet_log(),
        open: || {
            opened.set(true);
            Err::<SharedLcd, _>(SinkError::DeviceInBootloader)
        },
        assets: &mut assets,
        latch_at_start: false,
    });
    assert_eq!(exit, LoopExit::Stopped);
    assert!(opened.get(), "bootloader is detected at open");
}

#[test]
fn latch_at_start_skips_stream_open() {
    let scratch = Scratch::new("latched");
    let config = stream_config(&scratch.0, 10, 60, 0.5);
    let (mut clock, _) = FakeClock::new();
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let opened = std::cell::Cell::new(false);
    let exit = service::run_loop(LoopInput {
        config: &config,
        sampler: Steady {
            samples: Arc::new(Mutex::new(Vec::new())),
            origin: clock.mono(),
            load_of: |_| 10.0,
            cpu_of: |_| 10.0,
            fresh: true,
            n: std::cell::Cell::new(0),
        },
        clock: &mut clock,
        notify: &mut Quiet,
        stop: After::ticks(2),
        log: &mut quiet_log(),
        open: || {
            opened.set(true);
            Ok(SharedLcd::new())
        },
        assets: &mut assets,
        latch_at_start: true,
    });
    assert_eq!(exit, LoopExit::Stopped);
    assert!(!opened.get(), "a present latch does not open the device");
}
