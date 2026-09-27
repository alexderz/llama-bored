//! Watch down: no-data view, one stock restore, then a forced upload on the gap.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use kraken_lcd::config::{Config, ValidConfig};
use kraken_lcd::device::proto::Cmd;
use kraken_lcd::device::{FakeLcd, LcdSink, Record, SinkError};
use kraken_lcd::log;
use kraken_lcd::render::Frame;
use kraken_lcd::service::{self, Clock, LoopExit, LoopInput, Notifier, Sampler, Stop};
use llama_core::sample::{AiState, Snapshot};

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("t23-down-{label}-{}", std::process::id()));
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

fn load_config(dir: &Path, tick_s: f64, min_interval_s: u64) -> ValidConfig {
    let path = dir.join("config.toml");
    let text = format!(
        "[writer]\ntick_s = {tick_s}\n[upload]\nmin_interval_s = {min_interval_s}\nfail_limit = 3\n[snapshot]\nwatch_down_stock_after_s = 30\nwatch_down_restore_min_s = 600\n"
    );
    std::fs::write(&path, text).expect("config");
    Config::load_validated(&path).expect("valid")
}

fn snap(mono: Instant, wall: SystemTime, fresh: bool) -> Snapshot {
    if fresh {
        Snapshot {
            t_mono: mono,
            t_wall: wall,
            load: Some(22.0),
            activity: None,
            cpu_pct: Some(11.0),
            cpu_topk_pct: Some(22.0),
            gpu_pct: Some(9.0),
            mem_pct: Some(33.0),
            coolant_c: Some(36.0),
            cpu_c: Some(48.0),
            gpu_c: Some(55.0),
            ai: AiState::Idle,
            models: Vec::new(),
            tokens: None,
            errors: BTreeSet::new(),
        }
    } else {
        Snapshot {
            t_mono: mono,
            t_wall: wall,
            load: None,
            activity: None,
            cpu_pct: None,
            cpu_topk_pct: None,
            gpu_pct: None,
            mem_pct: None,
            coolant_c: Some(36.0),
            cpu_c: None,
            gpu_c: None,
            ai: AiState::NoData,
            models: Vec::new(),
            tokens: None,
            errors: BTreeSet::new(),
        }
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

struct QuietLog;

impl log::Sink for QuietLog {
    fn write_line(&mut self, _line: &str) {}
}

/// Fresh for the opening tick, stale until 35s, then fresh again.
struct Episode {
    origin: std::cell::Cell<Option<Instant>>,
}

impl Sampler for Episode {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        let origin = self.origin.get().unwrap_or(mono);
        if self.origin.get().is_none() {
            self.origin.set(Some(mono));
        }
        let elapsed = mono.saturating_duration_since(origin);
        let fresh = elapsed < Duration::from_millis(500) || elapsed >= Duration::from_secs(35);
        snap(mono, wall, fresh)
    }
}

struct AlwaysStale;

impl Sampler for AlwaysStale {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        snap(mono, wall, false)
    }
}

fn cmds(lcd: &SharedLcd) -> Vec<Cmd> {
    lcd.lock()
        .records()
        .iter()
        .filter_map(|record| match record {
            Record::Cmd(cmd) => Some(*cmd),
            Record::Bulk(_) => None,
        })
        .collect()
}

#[derive(Clone)]
struct TimedLcd {
    inner: SharedLcd,
    mono: Arc<Mutex<Instant>>,
    origin: Instant,
    events: Arc<Mutex<Vec<(Duration, &'static str)>>>,
}

impl LcdSink for TimedLcd {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        let result = self.inner.show(frame);
        if result.is_ok() {
            self.mark("slot");
        }
        result
    }
    fn restore_stock(&mut self) {
        self.inner.restore_stock();
        self.mark("liquid");
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

impl TimedLcd {
    fn mark(&self, kind: &'static str) {
        let now = *self.mono.lock().unwrap_or_else(|err| err.into_inner());
        self.events
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push((now.saturating_duration_since(self.origin), kind));
    }
}

#[test]
fn fresh_then_stale_restores_stock_once_then_forces_the_next_slot() {
    let scratch = Scratch::new("episode");
    let config = load_config(&scratch.0, 0.5, 10);
    let lcd = SharedLcd::new();
    let mut clock = FakeClock::new();
    let events = Arc::new(Mutex::new(Vec::new()));
    let timed = TimedLcd {
        inner: lcd.clone(),
        mono: Arc::clone(&clock.mono),
        origin: clock.mono(),
        events: Arc::clone(&events),
    };
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let exit = service::run_loop(LoopInput {
        config: &config,
        sampler: Episode {
            origin: std::cell::Cell::new(None),
        },
        clock: &mut clock,
        notify: &mut Quiet,
        stop: After::ticks(82),
        log: QuietLog,
        open: {
            let timed = timed.clone();
            move || Ok(timed.clone())
        },
        assets: &mut assets,
        latch_at_start: false,
    });
    assert_eq!(
        exit,
        LoopExit::Stopped,
        "watch down does not exit the writer"
    );
    let got = cmds(&lcd);
    let slots: Vec<&Cmd> = got
        .iter()
        .filter(|cmd| matches!(cmd, Cmd::ShowSlot(_)))
        .collect();
    let liquids: Vec<&Cmd> = got
        .iter()
        .filter(|cmd| matches!(cmd, Cmd::ShowLiquid))
        .collect();
    assert_eq!(liquids.len(), 1, "one ShowLiquid after 30s, cmds={got:?}");
    assert_eq!(
        slots.len(),
        3,
        "data, no-data, then the forced frame: {got:?}"
    );
    let liquid_at = got.iter().position(|cmd| matches!(cmd, Cmd::ShowLiquid));
    let slots_at: Vec<usize> = got
        .iter()
        .enumerate()
        .filter_map(|(index, cmd)| matches!(cmd, Cmd::ShowSlot(_)).then_some(index))
        .collect();
    assert_eq!(slots_at.len(), 3);
    assert!(
        slots_at[1] < liquid_at.expect("liquid") && liquid_at.unwrap() < slots_at[2],
        "no-data upload, then stock, then the forced upload: {got:?}"
    );
    let times = events.lock().unwrap_or_else(|err| err.into_inner()).clone();
    assert_eq!(
        times,
        vec![
            (Duration::from_secs(0), "slot"),
            (Duration::from_secs(10), "slot"),
            (Duration::from_millis(30_500), "liquid"),
            (Duration::from_millis(40_500), "slot"),
        ],
        "the forced frame waits for the upload gap after ShowLiquid"
    );
}

#[test]
fn halted_watch_down_sends_nothing() {
    let scratch = Scratch::new("halted");
    let config = load_config(&scratch.0, 1.0, 10);
    let lcd = SharedLcd::new();
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let opens = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let exit = service::run_loop(LoopInput {
        config: &config,
        sampler: AlwaysStale,
        clock: &mut FakeClock::new(),
        notify: &mut Quiet,
        stop: After::ticks(40),
        log: QuietLog,
        open: {
            let lcd = lcd.clone();
            let opens = Arc::clone(&opens);
            move || {
                opens.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(lcd.clone())
            }
        },
        assets: &mut assets,
        latch_at_start: true,
    });
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(opens.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert!(lcd.lock().records().is_empty(), "HALTED does no device I/O");
}

/// Fresh only while a forced frame can land between two stale episodes.
struct CappedEpisodes {
    origin: std::cell::Cell<Option<Instant>>,
}

impl Sampler for CappedEpisodes {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        if self.origin.get().is_none() {
            self.origin.set(Some(mono));
        }
        let origin = self.origin.get().unwrap_or(mono);
        let elapsed = mono.saturating_duration_since(origin);
        let fresh = (Duration::from_secs(70)..Duration::from_secs(121)).contains(&elapsed);
        snap(mono, wall, fresh)
    }
}

#[test]
fn restore_cap_still_uploads_no_data_until_the_next_show_liquid() {
    let scratch = Scratch::new("cap");
    let config = load_config(&scratch.0, 1.0, 60);
    let mut clock = FakeClock::new();
    let events = Arc::new(Mutex::new(Vec::new()));
    let timed = TimedLcd {
        inner: SharedLcd::new(),
        mono: Arc::clone(&clock.mono),
        origin: clock.mono(),
        events: Arc::clone(&events),
    };
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let exit = service::run_loop(LoopInput {
        config: &config,
        sampler: CappedEpisodes {
            origin: std::cell::Cell::new(None),
        },
        clock: &mut clock,
        notify: &mut Quiet,
        stop: After::ticks(661),
        log: QuietLog,
        open: {
            let timed = timed.clone();
            move || Ok(timed.clone())
        },
        assets: &mut assets,
        latch_at_start: false,
    });
    assert_eq!(exit, LoopExit::Stopped);
    let times = events.lock().unwrap_or_else(|err| err.into_inner()).clone();
    assert_eq!(
        times,
        vec![
            (Duration::from_secs(0), "slot"),
            (Duration::from_secs(60), "liquid"),
            (Duration::from_secs(120), "slot"),
            (Duration::from_secs(180), "slot"),
            (Duration::from_secs(660), "liquid"),
        ],
        "while the 600s cap blocks ShowLiquid the no-data frame still uploads"
    );
}

/// Stale the whole run. From 1 s on, an external latch file exists.
struct ArmLatch {
    dir: PathBuf,
    origin: std::cell::Cell<Option<Instant>>,
}

impl Sampler for ArmLatch {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        if self.origin.get().is_none() {
            self.origin.set(Some(mono));
        }
        let origin = self.origin.get().unwrap_or(mono);
        if mono.saturating_duration_since(origin) >= Duration::from_secs(1) {
            std::fs::write(self.dir.join("halted"), b"external\n").expect("latch");
        }
        snap(mono, wall, false)
    }
}

#[derive(Clone)]
struct LatchSink {
    inner: SharedLcd,
    dir: PathBuf,
}

impl LcdSink for LatchSink {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        self.inner.show(frame)
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
        service::latch_present(&self.dir)
    }
}

#[test]
fn external_latch_halts_before_the_follow_up() {
    let scratch = Scratch::new("ext-latch");
    let config = load_config(&scratch.0, 1.0, 10);
    let lcd = SharedLcd::new();
    let sink = LatchSink {
        inner: lcd.clone(),
        dir: scratch.0.clone(),
    };
    let opens = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let exit = service::run_loop(LoopInput {
        config: &config,
        sampler: ArmLatch {
            dir: scratch.0.clone(),
            origin: std::cell::Cell::new(None),
        },
        clock: &mut FakeClock::new(),
        notify: &mut Quiet,
        stop: After::ticks(40),
        log: QuietLog,
        open: {
            let sink = sink.clone();
            let opens = Arc::clone(&opens);
            move || {
                opens.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(sink.clone())
            }
        },
        assets: &mut assets,
        latch_at_start: false,
    });
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(opens.load(std::sync::atomic::Ordering::Relaxed), 1);
    let got = cmds(&lcd);
    assert!(
        got.iter().any(|cmd| matches!(cmd, Cmd::ShowSlot(_))),
        "the frame before the latch still uploads"
    );
    assert!(
        got.iter().all(|cmd| !matches!(cmd, Cmd::ShowLiquid)),
        "HALTED before the 2s follow-up and before watch-down stock: {got:?}"
    );
}

#[test]
fn bootloader_watch_down_sends_no_show_liquid() {
    let scratch = Scratch::new("boot");
    let config = load_config(&scratch.0, 1.0, 10);
    let lcd = SharedLcd::new();
    let opens = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let exit = service::run_loop(LoopInput {
        config: &config,
        sampler: AlwaysStale,
        clock: &mut FakeClock::new(),
        notify: &mut Quiet,
        stop: After::ticks(40),
        log: QuietLog,
        open: {
            let lcd = lcd.clone();
            let opens = Arc::clone(&opens);
            let gave_bootloader = std::sync::atomic::AtomicBool::new(false);
            move || {
                opens.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if gave_bootloader.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    Ok(lcd.clone())
                } else {
                    Err(SinkError::DeviceInBootloader)
                }
            }
        },
        assets: &mut assets,
        latch_at_start: false,
    });
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(
        opens.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "BOOTLOADER does not retry open"
    );
    assert!(
        cmds(&lcd).is_empty(),
        "watch down must not ShowLiquid in BOOTLOADER: {:?}",
        cmds(&lcd)
    );
}
