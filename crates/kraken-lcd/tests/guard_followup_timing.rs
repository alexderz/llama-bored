//! At a 0.5 s tick the cooling follow-up runs on the first tick at least 2 s later.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use kraken_lcd::config::Config;
use kraken_lcd::device::{FakeLcd, LcdSink, SinkError};
use kraken_lcd::log;
use kraken_lcd::render::Frame;
use kraken_lcd::service::{self, Clock, LoopInput, Notifier, Sampler, Stop};
use llama_core::sample::{AiState, Snapshot};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("t23-follow-{}", std::process::id()));
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

#[derive(Clone)]
struct CountingLcd {
    inner: Arc<Mutex<FakeLcd>>,
    ticks: Arc<Mutex<Vec<Duration>>>,
    origin: Arc<Mutex<Instant>>,
    mono: Arc<Mutex<Instant>>,
}

impl LcdSink for CountingLcd {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .show(frame)
    }
    fn restore_stock(&mut self) {
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .restore_stock();
    }
    fn needs_reupload(&mut self) {
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .needs_reupload();
    }
    fn tick(&mut self) -> Result<(), SinkError> {
        let now = *self.mono.lock().unwrap_or_else(|err| err.into_inner());
        let origin = *self.origin.lock().unwrap_or_else(|err| err.into_inner());
        self.ticks
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(now.saturating_duration_since(origin));
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

struct Idle;

impl Sampler for Idle {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        Snapshot {
            t_mono: mono,
            t_wall: wall,
            load: Some(10.0),
            activity: None,
            cpu_pct: Some(10.0),
            cpu_topk_pct: Some(10.0),
            gpu_pct: Some(10.0),
            mem_pct: Some(10.0),
            coolant_c: Some(30.0),
            cpu_c: Some(40.0),
            gpu_c: Some(40.0),
            ai: AiState::Idle,
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

#[test]
fn follow_up_runs_at_the_first_tick_at_least_two_seconds_later() {
    let scratch = Scratch::new();
    let path = scratch.0.join("config.toml");
    std::fs::write(
        &path,
        "[writer]\ntick_s = 0.5\n[upload]\nmin_interval_s = 60\nfail_limit = 3\n",
    )
    .expect("config");
    let config = Config::load_validated(&path).expect("valid");
    let origin = Instant::now();
    let mono = Arc::new(Mutex::new(origin));
    let ticks = Arc::new(Mutex::new(Vec::new()));
    let lcd = CountingLcd {
        inner: Arc::new(Mutex::new(FakeLcd::new())),
        ticks: Arc::clone(&ticks),
        origin: Arc::new(Mutex::new(origin)),
        mono: Arc::clone(&mono),
    };
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let _ = service::run_loop(LoopInput {
        config: &config,
        sampler: Idle,
        clock: &mut FakeClock {
            mono: Arc::clone(&mono),
            wall: SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000),
        },
        notify: &mut Quiet,
        stop: After(std::cell::Cell::new(5)),
        log: QuietLog,
        open: {
            let lcd = lcd.clone();
            move || Ok(lcd.clone())
        },
        assets: &mut assets,
        latch_at_start: false,
    });
    let seen = ticks.lock().unwrap_or_else(|err| err.into_inner()).clone();
    assert_eq!(
        seen,
        vec![Duration::from_secs(2)],
        "0.5s ticks at 0, 0.5, 1.0 and 1.5 must not run the follow-up"
    );
}
