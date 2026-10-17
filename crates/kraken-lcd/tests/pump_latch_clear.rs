//! A pump latch clears itself after 600 s of normal pump reads (#158).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use kraken_lcd::collector::{AiState, Snapshot};
use kraken_lcd::config::{Config, ValidConfig};
use kraken_lcd::device::{
    CoolingSnapshot, FakeLcd, LcdSink, PUMP_AUTO_CLEARED_FILE, PUMP_LATCH_CLEAR_AFTER,
    PUMP_LATCH_MAX_GAP, SinkError,
};
use kraken_lcd::log::{self, Priority};
use kraken_lcd::render::Frame;
use kraken_lcd::service::{self, Clock, HaltWatch, LoopExit, LoopInput, Notifier, Sampler, Stop};

const START_UNIX: u64 = 1_700_000_000;

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("pump-clear-{label}-{}", std::process::id()));
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

fn load_config(dir: &Path) -> ValidConfig {
    let path = dir.join("config.toml");
    std::fs::write(
        &path,
        "[writer]\ntick_s = 1.0\n[upload]\nmin_interval_s = 10\nfail_limit = 3\n",
    )
    .expect("config");
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
        errors: std::collections::BTreeSet::new(),
    }
}

fn cooling(fan: u32, pwm1_enable: u32, devnum: u32) -> CoolingSnapshot {
    CoolingSnapshot {
        z53: true,
        pwm1_enable: Some(pwm1_enable),
        pwm2_enable: Some(0),
        pwm1: Some(64),
        pwm2: Some(51),
        fan1_rpm: Some(fan),
        devnum: Some(devnum),
        bootloader: false,
    }
}

fn normal() -> CoolingSnapshot {
    cooling(1000, 0, 2)
}

fn pump_latch() -> String {
    "CRITICAL time=1700000000 reason=fan1_input\n\
baseline z53=yes pwm1_enable=0 pwm2_enable=0 pwm1=64 pwm2=51 fan1_rpm=1000 devnum=2 bootloader=no\n\
current z53=yes pwm1_enable=0 pwm2_enable=0 pwm1=64 pwm2=51 fan1_rpm=0 devnum=2 bootloader=no\n"
        .to_owned()
}

fn latch_for(reason: &str) -> String {
    format!(
        "CRITICAL time=1 reason={reason}\n\
baseline z53=yes pwm1_enable=0 pwm2_enable=0 pwm1=64 pwm2=51 fan1_rpm=1000 devnum=2 bootloader=no\n\
current z53=yes pwm1_enable=0 pwm2_enable=0 pwm1=64 pwm2=51 fan1_rpm=1000 devnum=2 bootloader=no\n"
    )
}

struct FakeClock {
    mono: Instant,
    wall: SystemTime,
}

impl FakeClock {
    fn new() -> Self {
        Self {
            mono: Instant::now(),
            wall: SystemTime::UNIX_EPOCH + Duration::from_secs(START_UNIX),
        }
    }
}

impl Clock for FakeClock {
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

/// Adds `PUMP_LATCH_MAX_GAP + 1 s` of wall time on sleep 300.
struct GappedClock {
    mono: Instant,
    wall: SystemTime,
    sleeps: u32,
}

impl GappedClock {
    fn new() -> Self {
        Self {
            mono: Instant::now(),
            wall: SystemTime::UNIX_EPOCH + Duration::from_secs(START_UNIX),
            sleeps: 0,
        }
    }
}

impl Clock for GappedClock {
    fn mono(&self) -> Instant {
        self.mono
    }

    fn wall(&self) -> SystemTime {
        self.wall
    }

    fn sleep(&mut self, d: Duration) {
        self.sleeps += 1;
        self.mono += d;
        self.wall += d;
        if self.sleeps == 300 {
            self.wall += PUMP_LATCH_MAX_GAP + Duration::from_secs(1);
        }
    }
}

struct Idle;

impl Sampler for Idle {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        snapshot(mono, wall)
    }
}

struct QuietNotify;

impl Notifier for QuietNotify {
    fn ready(&mut self) {}
    fn watchdog(&mut self) {}
    fn stopping(&mut self) {}
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

#[derive(Default, Clone)]
struct CaptureLog(Arc<Mutex<Vec<String>>>);

impl log::Sink for CaptureLog {
    fn write_line(&mut self, line: &str) {
        self.0
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(line.to_owned());
    }
}

impl CaptureLog {
    fn crits(&self) -> Vec<String> {
        self.0
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .iter()
            .filter(|line| line.starts_with(log::prefix(Priority::Crit)))
            .cloned()
            .collect()
    }
}

struct SeqWatch {
    dir: PathBuf,
    reads: Arc<AtomicU32>,
    next: Box<dyn FnMut(u32) -> Option<CoolingSnapshot>>,
}

impl SeqWatch {
    fn new(dir: &Path, next: impl FnMut(u32) -> Option<CoolingSnapshot> + 'static) -> Self {
        Self {
            dir: dir.to_path_buf(),
            reads: Arc::new(AtomicU32::new(0)),
            next: Box::new(next),
        }
    }

    fn reads(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.reads)
    }
}

impl HaltWatch for SeqWatch {
    fn state_dir(&self) -> &Path {
        &self.dir
    }

    fn read_cooling(&mut self) -> Option<CoolingSnapshot> {
        let n = self.reads.fetch_add(1, Ordering::Relaxed);
        (self.next)(n)
    }
}

struct Steady;

impl Steady {
    fn watch(dir: &Path) -> SeqWatch {
        SeqWatch::new(dir, |_n| Some(normal()))
    }
}

struct LatchSink {
    dir: PathBuf,
    arm: bool,
    inner: FakeLcd,
}

impl LcdSink for LatchSink {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        if self.arm {
            self.arm = false;
            std::fs::write(self.dir.join("halted"), pump_latch()).expect("latch");
            return Err(SinkError::Halted);
        }
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
        self.inner.blocked()
    }
}

struct Ran {
    exit: LoopExit,
    opens: u32,
    log: CaptureLog,
    reads: u32,
}

fn run_with<Clk, W, S>(
    config: &ValidConfig,
    ticks: u32,
    mut clock: Clk,
    mut watch: W,
    latch_at_start: bool,
    mut open: impl FnMut() -> Result<S, SinkError>,
) -> Ran
where
    Clk: Clock,
    W: HaltWatch,
    S: LcdSink,
{
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let mut notify = QuietNotify;
    let log = CaptureLog::default();
    let mut opens = 0u32;
    let exit = service::run_loop(LoopInput {
        config,
        sampler: Idle,
        clock: &mut clock,
        notify: &mut notify,
        stop: After::ticks(ticks),
        log: log.clone(),
        open: || {
            opens += 1;
            open()
        },
        assets: &mut assets,
        latch_at_start,
        halt: &mut watch,
    });
    Ran {
        exit,
        opens,
        log,
        reads: 0,
    }
}

fn run_steady(config: &ValidConfig, dir: &Path, ticks: u32, latch_at_start: bool) -> Ran {
    let watch = Steady::watch(dir);
    let reads = watch.reads();
    let mut ran = run_with(
        config,
        ticks,
        FakeClock::new(),
        watch,
        latch_at_start,
        || Ok(FakeLcd::new()),
    );
    ran.reads = reads.load(Ordering::Relaxed);
    ran
}

fn clear_line(n: usize) -> String {
    let after = PUMP_LATCH_CLEAR_AFTER.as_secs();
    format!(
        "cooling guard latch cleared after {after} s of normal pump operation (auto-clear {n} of 3 in 24 h)"
    )
}

fn assert_cleared(dir: &Path, ran: &Ran, unix: u64, which: usize, opens: u32) {
    assert_eq!(ran.exit, LoopExit::Stopped);
    assert_eq!(ran.opens, opens);
    assert!(!dir.join("halted").exists(), "latch removed");
    assert_eq!(
        std::fs::read(dir.join(PUMP_AUTO_CLEARED_FILE)).expect("record"),
        format!("{unix}\n").into_bytes()
    );
    let crits = ran.log.crits();
    assert!(
        crits.iter().any(|line| line.contains(&clear_line(which))),
        "{crits:?}"
    );
}

fn assert_stays(dir: &Path, ran: &Ran) {
    assert_eq!(ran.exit, LoopExit::Stopped);
    assert_eq!(ran.opens, 0);
    assert!(dir.join("halted").exists(), "latch stays");
    assert!(
        ran.log
            .crits()
            .iter()
            .all(|line| !line.contains("latch cleared")),
        "{:?}",
        ran.log.crits()
    );
}

#[test]
fn pump_latch_found_at_start_clears_after_600s() {
    let scratch = Scratch::new("found");
    std::fs::write(scratch.0.join("halted"), pump_latch()).expect("latch");
    let config = load_config(&scratch.0);
    let ran = run_steady(&config, &scratch.0, 601, true);
    assert_eq!(ran.reads, 601);
    assert_cleared(&scratch.0, &ran, START_UNIX + 600, 1, 1);
    assert!(
        ran.log
            .crits()
            .iter()
            .any(|line| line.contains("cooling guard latch is present; collect only"))
    );
}

#[test]
fn pump_latch_written_in_process_clears_after_600s() {
    let scratch = Scratch::new("in-process");
    let config = load_config(&scratch.0);
    let dir = scratch.0.clone();
    let watch = Steady::watch(&scratch.0);
    let reads = watch.reads();
    let mut armed = true;
    let ran = run_with(&config, 602, FakeClock::new(), watch, false, move || {
        let arm = armed;
        armed = false;
        Ok(LatchSink {
            dir: dir.clone(),
            arm,
            inner: FakeLcd::new(),
        })
    });
    assert_eq!(reads.load(Ordering::Relaxed), 601);
    assert_cleared(&scratch.0, &ran, START_UNIX + 601, 1, 2);
    assert!(
        ran.log
            .crits()
            .iter()
            .any(|line| line.contains("cooling guard halted; collect only"))
    );
}

#[test]
fn abnormal_read_at_599s_does_not_clear_at_600s() {
    let scratch = Scratch::new("abnormal-short");
    std::fs::write(scratch.0.join("halted"), pump_latch()).expect("latch");
    let config = load_config(&scratch.0);
    let watch = SeqWatch::new(&scratch.0, |n| {
        if n == 599 {
            Some(cooling(0, 0, 2))
        } else {
            Some(normal())
        }
    });
    let ran = run_with(&config, 601, FakeClock::new(), watch, true, || {
        Ok(FakeLcd::new())
    });
    assert_stays(&scratch.0, &ran);
    assert!(!scratch.0.join(PUMP_AUTO_CLEARED_FILE).exists());
}

#[test]
fn abnormal_read_at_599s_restarts_the_window() {
    let scratch = Scratch::new("abnormal-restart");
    std::fs::write(scratch.0.join("halted"), pump_latch()).expect("latch");
    let config = load_config(&scratch.0);
    let watch = SeqWatch::new(&scratch.0, |n| {
        if n == 599 {
            Some(cooling(0, 0, 2))
        } else {
            Some(normal())
        }
    });
    let ran = run_with(&config, 1202, FakeClock::new(), watch, true, || {
        Ok(FakeLcd::new())
    });
    assert_cleared(&scratch.0, &ran, START_UNIX + 1201, 1, 1);
}

#[test]
fn gap_between_reads_does_not_clear_at_the_original_600s() {
    let scratch = Scratch::new("gap-short");
    std::fs::write(scratch.0.join("halted"), pump_latch()).expect("latch");
    let config = load_config(&scratch.0);
    let ran = run_with(
        &config,
        601,
        GappedClock::new(),
        Steady::watch(&scratch.0),
        true,
        || Ok(FakeLcd::new()),
    );
    assert_stays(&scratch.0, &ran);
    assert!(!scratch.0.join(PUMP_AUTO_CLEARED_FILE).exists());
}

#[test]
fn gap_between_reads_restarts_the_window() {
    let scratch = Scratch::new("gap");
    std::fs::write(scratch.0.join("halted"), pump_latch()).expect("latch");
    let config = load_config(&scratch.0);
    let ran = run_with(
        &config,
        901,
        GappedClock::new(),
        Steady::watch(&scratch.0),
        true,
        || Ok(FakeLcd::new()),
    );
    assert_cleared(&scratch.0, &ran, START_UNIX + 906, 1, 1);
}

#[test]
fn service_restart_restarts_the_window() {
    let scratch = Scratch::new("restart");
    std::fs::write(scratch.0.join("halted"), pump_latch()).expect("latch");
    let config = load_config(&scratch.0);
    let first = run_steady(&config, &scratch.0, 400, true);
    assert_stays(&scratch.0, &first);
    assert!(!scratch.0.join(PUMP_AUTO_CLEARED_FILE).exists());
    let second = run_steady(&config, &scratch.0, 400, true);
    assert_stays(&scratch.0, &second);
    let third = run_steady(&config, &scratch.0, 601, true);
    assert_cleared(&scratch.0, &third, START_UNIX + 600, 1, 1);
}

#[test]
fn non_pump_reasons_never_auto_clear() {
    for reason in [
        "z53_missing",
        "pwm1_enable",
        "pwm2_enable",
        "devnum",
        "bootloader",
        "unreadable",
    ] {
        let scratch = Scratch::new(&format!("reason-{reason}"));
        std::fs::write(scratch.0.join("halted"), latch_for(reason)).expect("latch");
        let config = load_config(&scratch.0);
        let ran = run_steady(&config, &scratch.0, 601, true);
        assert_eq!(ran.reads, 0, "{reason}");
        assert_stays(&scratch.0, &ran);
    }
}

#[test]
fn unparsable_latch_never_auto_clears() {
    let scratch = Scratch::new("unparsable");
    std::fs::write(scratch.0.join("halted"), b"not a latch\n").expect("latch");
    let config = load_config(&scratch.0);
    let ran = run_steady(&config, &scratch.0, 601, true);
    assert_eq!(ran.reads, 0);
    assert_stays(&scratch.0, &ran);
}

#[test]
fn pump_latch_without_baseline_line_never_auto_clears() {
    let scratch = Scratch::new("no-baseline");
    std::fs::write(
        scratch.0.join("halted"),
        b"CRITICAL time=1 reason=fan1_input\n",
    )
    .expect("latch");
    let config = load_config(&scratch.0);
    let ran = run_steady(&config, &scratch.0, 601, true);
    assert_eq!(ran.reads, 0);
    assert_stays(&scratch.0, &ran);
}

#[test]
fn read_that_differs_in_pwm1_enable_never_auto_clears() {
    let scratch = Scratch::new("pwm1-enable");
    std::fs::write(scratch.0.join("halted"), pump_latch()).expect("latch");
    let config = load_config(&scratch.0);
    let watch = SeqWatch::new(&scratch.0, |_n| Some(cooling(1000, 1, 2)));
    let ran = run_with(&config, 601, FakeClock::new(), watch, true, || {
        Ok(FakeLcd::new())
    });
    assert_stays(&scratch.0, &ran);
    assert!(!scratch.0.join(PUMP_AUTO_CLEARED_FILE).exists());
}

#[test]
fn read_that_differs_in_devnum_never_auto_clears() {
    let scratch = Scratch::new("devnum");
    std::fs::write(scratch.0.join("halted"), pump_latch()).expect("latch");
    let config = load_config(&scratch.0);
    let watch = SeqWatch::new(&scratch.0, |_n| Some(cooling(1000, 0, 9)));
    let ran = run_with(&config, 601, FakeClock::new(), watch, true, || {
        Ok(FakeLcd::new())
    });
    assert_stays(&scratch.0, &ran);
    assert!(!scratch.0.join(PUMP_AUTO_CLEARED_FILE).exists());
}

#[test]
fn three_clears_in_24h_keep_the_fourth_latch() {
    let scratch = Scratch::new("backoff");
    std::fs::write(scratch.0.join("halted"), pump_latch()).expect("latch");
    let body = format!(
        "{}\n{}\n{}\n",
        START_UNIX - 3000,
        START_UNIX - 2000,
        START_UNIX - 1000
    );
    std::fs::write(scratch.0.join(PUMP_AUTO_CLEARED_FILE), &body).expect("record");
    let config = load_config(&scratch.0);
    let ran = run_steady(&config, &scratch.0, 601, true);
    assert_stays(&scratch.0, &ran);
    assert_eq!(
        std::fs::read_to_string(scratch.0.join(PUMP_AUTO_CLEARED_FILE)).expect("record"),
        body
    );
}

#[test]
fn clear_older_than_24h_does_not_count() {
    let scratch = Scratch::new("aged");
    std::fs::write(scratch.0.join("halted"), pump_latch()).expect("latch");
    let old = START_UNIX - 25 * 3600;
    let recent_a = START_UNIX - 1000;
    let recent_b = START_UNIX - 2000;
    std::fs::write(
        scratch.0.join(PUMP_AUTO_CLEARED_FILE),
        format!("{old}\n{recent_a}\n{recent_b}\n"),
    )
    .expect("record");
    let config = load_config(&scratch.0);
    let ran = run_steady(&config, &scratch.0, 601, true);
    assert_eq!(ran.exit, LoopExit::Stopped);
    assert_eq!(ran.opens, 1);
    assert!(!scratch.0.join("halted").exists());
    let written = std::fs::read_to_string(scratch.0.join(PUMP_AUTO_CLEARED_FILE)).expect("record");
    let cleared_at = START_UNIX + 600;
    assert_eq!(written, format!("{recent_a}\n{recent_b}\n{cleared_at}\n"));
    assert!(
        ran.log
            .crits()
            .iter()
            .any(|line| line.contains(&clear_line(3))),
        "{:?}",
        ran.log.crits()
    );
}

#[test]
fn corrupt_auto_clear_state_means_no_clear() {
    let scratch = Scratch::new("corrupt");
    std::fs::write(scratch.0.join("halted"), pump_latch()).expect("latch");
    std::fs::write(scratch.0.join(PUMP_AUTO_CLEARED_FILE), b"nope\n").expect("record");
    let config = load_config(&scratch.0);
    let ran = run_steady(&config, &scratch.0, 601, true);
    assert_eq!(ran.reads, 0);
    assert_stays(&scratch.0, &ran);
    assert_eq!(
        std::fs::read(scratch.0.join(PUMP_AUTO_CLEARED_FILE)).expect("untouched"),
        b"nope\n"
    );
    let bad: Vec<_> = ran
        .log
        .crits()
        .into_iter()
        .filter(|line| line.contains("unreadable or corrupt"))
        .collect();
    assert_eq!(bad.len(), 1, "{:?}", ran.log.crits());
}

#[test]
fn unreadable_auto_clear_state_means_no_clear() {
    let scratch = Scratch::new("unreadable");
    std::fs::write(scratch.0.join("halted"), pump_latch()).expect("latch");
    std::fs::create_dir(scratch.0.join(PUMP_AUTO_CLEARED_FILE)).expect("dir");
    let config = load_config(&scratch.0);
    let ran = run_steady(&config, &scratch.0, 601, true);
    assert_eq!(ran.reads, 0);
    assert_stays(&scratch.0, &ran);
    let bad: Vec<_> = ran
        .log
        .crits()
        .into_iter()
        .filter(|line| line.contains("unreadable or corrupt"))
        .collect();
    assert_eq!(bad.len(), 1, "{:?}", ran.log.crits());
}
