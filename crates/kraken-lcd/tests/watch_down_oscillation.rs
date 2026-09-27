//! An oscillating watcher must stay inside the device-write budget.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use kraken_lcd::config::Config;
use kraken_lcd::device::{FakeLcd, LcdSink, SinkError};
use kraken_lcd::log;
use kraken_lcd::render::Frame;
use kraken_lcd::service::{self, Clock, LoopExit, LoopInput, Notifier, Sampler, Stop};
use llama_core::sample::{AiState, Snapshot};

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone)]
struct BudgetLcd {
    inner: Arc<Mutex<FakeLcd>>,
    mono: Arc<Mutex<Instant>>,
    origin: Instant,
    writes: Arc<Mutex<Vec<Duration>>>,
    liquids: Arc<Mutex<Vec<Duration>>>,
}

impl BudgetLcd {
    fn stamp(&self) -> Duration {
        let now = *self.mono.lock().unwrap_or_else(|err| err.into_inner());
        now.saturating_duration_since(self.origin)
    }
}

impl LcdSink for BudgetLcd {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        let result = self
            .inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .show(frame);
        if result.is_ok() {
            self.writes
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .push(self.stamp());
        }
        result
    }

    fn restore_stock(&mut self) {
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .restore_stock();
        let at = self.stamp();
        self.writes
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(at);
        self.liquids
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(at);
    }

    fn needs_reupload(&mut self) {
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .needs_reupload();
    }

    fn tick(&mut self) -> Result<(), SinkError> {
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .tick()
    }

    fn blocked(&mut self) -> bool {
        false
    }
}

struct FakeClock {
    mono: Arc<Mutex<Instant>>,
    wall: SystemTime,
}

impl Clock for FakeClock {
    fn mono(&self) -> Instant {
        *self.mono.lock().unwrap_or_else(|err| err.into_inner())
    }
    fn wall(&self) -> SystemTime {
        self.wall
    }
    fn sleep(&mut self, d: Duration) {
        *self.mono.lock().unwrap_or_else(|err| err.into_inner()) += d;
        self.wall += d;
    }
}

/// 31 s stale, 1 s fresh, repeating.
struct Oscillate {
    origin: std::cell::Cell<Option<Instant>>,
}

impl Sampler for Oscillate {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        if self.origin.get().is_none() {
            self.origin.set(Some(mono));
        }
        let origin = self.origin.get().unwrap_or(mono);
        let pos = mono.saturating_duration_since(origin).as_nanos() % 32_000_000_000;
        let fresh = pos >= 31_000_000_000;
        Snapshot {
            t_mono: mono,
            t_wall: wall,
            load: fresh.then_some(20.0),
            activity: None,
            cpu_pct: fresh.then_some(10.0),
            cpu_topk_pct: fresh.then_some(20.0),
            gpu_pct: fresh.then_some(8.0),
            mem_pct: fresh.then_some(30.0),
            coolant_c: Some(36.0),
            cpu_c: fresh.then_some(40.0),
            gpu_c: fresh.then_some(50.0),
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
}

struct After(std::cell::Cell<u32>);

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

fn windows_hold(times: &[Duration], width: Duration) -> bool {
    times.iter().enumerate().all(|(index, start)| {
        times[index..]
            .iter()
            .filter(|at| (**at).saturating_sub(*start) < width)
            .count()
            <= 1
    })
}

#[test]
fn two_hours_of_oscillation_stays_inside_the_write_budget() {
    let scratch = Scratch(
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("t23-osc-{}", std::process::id())),
    );
    let _ = std::fs::remove_dir_all(&scratch.0);
    std::fs::create_dir_all(&scratch.0).expect("scratch");
    let path = scratch.0.join("config.toml");
    std::fs::write(
        &path,
        "[writer]\ntick_s = 0.5\n[upload]\nmin_interval_s = 60\nfail_limit = 3\n[snapshot]\nstale_after_s = 1.0\nwatch_down_stock_after_s = 30\nwatch_down_restore_min_s = 600\n",
    )
    .expect("config");
    let config = Config::load_validated(&path).expect("valid");
    let origin = Instant::now();
    let mono = Arc::new(Mutex::new(origin));
    let writes = Arc::new(Mutex::new(Vec::new()));
    let liquids = Arc::new(Mutex::new(Vec::new()));
    let lcd = BudgetLcd {
        inner: Arc::new(Mutex::new(FakeLcd::new())),
        mono: Arc::clone(&mono),
        origin,
        writes: Arc::clone(&writes),
        liquids: Arc::clone(&liquids),
    };
    let mut clock = FakeClock {
        mono,
        wall: SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000),
    };
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let exit = service::run_loop(LoopInput {
        config: &config,
        sampler: Oscillate {
            origin: std::cell::Cell::new(None),
        },
        clock: &mut clock,
        notify: &mut Quiet,
        stop: After(std::cell::Cell::new(14_400)),
        log: QuietLog,
        open: {
            let lcd = lcd.clone();
            move || Ok(lcd.clone())
        },
        assets: &mut assets,
        latch_at_start: false,
    });
    assert_eq!(exit, LoopExit::Stopped);
    let writes = writes.lock().unwrap_or_else(|err| err.into_inner()).clone();
    let liquids = liquids
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .clone();
    assert!(
        writes.len() <= 120,
        "writes={} exceeds 120 in 2h",
        writes.len()
    );
    assert!(
        windows_hold(&writes, Duration::from_secs(60)),
        "a 60s window held more than one device write: {writes:?}"
    );
    assert!(
        windows_hold(&liquids, Duration::from_secs(600)),
        "ShowLiquid more than once per 600s: {liquids:?}"
    );
}
