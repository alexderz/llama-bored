//! GitHub #59: a requested stop never cuts an upload in half.
//!
//! The unit's `ExecStop=` creates [`service::STOP_FLAG`] and waits for the
//! writer to leave before systemd sends SIGTERM. These tests plant the flag
//! from inside an upload (the worst moment) and check that the upload runs to
//! its end, no further upload starts, and the writer itself sends no
//! `ShowLiquid` (`ExecStopPost` restores stock). Fake device and clock only.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use kraken_lcd::config::{Config, ValidConfig};
use kraken_lcd::device::proto::Cmd;
use kraken_lcd::device::{FakeLcd, LcdSink, Record, SinkError};
use kraken_lcd::log;
use kraken_lcd::render::Frame;
use kraken_lcd::service::{
    self, Clock, LoopExit, LoopInput, Notifier, STOP_FLAG, Sampler, Stop, StopFlag,
};
use llama_core::sample::{AiState, Snapshot};

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("gh59-{label}-{}", std::process::id()));
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

fn config(dir: &Path, upload: &str) -> ValidConfig {
    let path = dir.join("config.toml");
    std::fs::write(
        &path,
        format!("[writer]\ntick_s = 0.5\n[upload]\n{upload}fail_limit = 3\n"),
    )
    .expect("config");
    Config::load_validated(&path).expect("valid")
}

struct FakeClock {
    mono: Instant,
    wall: SystemTime,
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

/// Fresh data whose load changes every tick, so change mode wants an upload.
struct Moving(u32);

impl Sampler for Moving {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        self.0 = self.0.wrapping_add(1);
        let load = (self.0 % 5) as f32 * 20.0;
        Snapshot {
            t_mono: mono,
            t_wall: wall,
            load: Some(load),
            activity: None,
            cpu_pct: Some(load),
            cpu_topk_pct: Some(load),
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
}

struct Quiet;

impl Notifier for Quiet {
    fn ready(&mut self) {}
    fn watchdog(&mut self) {}
    fn stopping(&mut self) {}
}

#[derive(Clone, Default)]
struct Lines(Arc<Mutex<Vec<String>>>);

impl log::Sink for Lines {
    fn write_line(&mut self, line: &str) {
        self.0
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(line.to_owned());
    }
}

/// A [`FakeLcd`] whose `n`-th upload creates the stop flag while it runs,
/// as `ExecStop=` would during a `systemctl try-restart`.
#[derive(Clone)]
struct StopDuringUpload {
    inner: Arc<Mutex<FakeLcd>>,
    flag: PathBuf,
    at_upload: usize,
    uploads: Arc<Mutex<usize>>,
    /// The flag was present when an upload started.
    started_after_stop: Arc<Mutex<bool>>,
}

impl StopDuringUpload {
    fn new(flag: &Path, at_upload: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(FakeLcd::new())),
            flag: flag.to_path_buf(),
            at_upload,
            uploads: Arc::new(Mutex::new(0)),
            started_after_stop: Arc::new(Mutex::new(false)),
        }
    }

    fn lcd(&self) -> std::sync::MutexGuard<'_, FakeLcd> {
        self.inner.lock().unwrap_or_else(|err| err.into_inner())
    }

    fn upload(&mut self, slot: Option<u8>, frame: &Frame) -> Result<(), SinkError> {
        if self.flag.exists() {
            *self
                .started_after_stop
                .lock()
                .unwrap_or_else(|err| err.into_inner()) = true;
        }
        let n = {
            let mut uploads = self.uploads.lock().unwrap_or_else(|err| err.into_inner());
            *uploads += 1;
            *uploads
        };
        if n == self.at_upload {
            std::fs::write(&self.flag, b"").expect("plant stop flag");
        }
        match slot {
            Some(slot) => self.lcd().show_slot(slot, frame),
            None => self.lcd().show(frame),
        }
    }
}

impl LcdSink for StopDuringUpload {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        self.upload(None, frame)
    }
    fn show_slot(&mut self, slot: u8, frame: &Frame) -> Result<(), SinkError> {
        self.upload(Some(slot), frame)
    }
    fn restore_stock(&mut self) {
        self.lcd().restore_stock();
    }
    fn needs_reupload(&mut self) {
        self.lcd().needs_reupload();
    }
    fn tick(&mut self) -> Result<(), SinkError> {
        self.lcd().tick()
    }
    fn blocked(&mut self) -> bool {
        self.lcd().blocked()
    }
}

/// Upper bound on ticks, so a stop that is never seen fails the test
/// instead of hanging it.
struct FlagOrLimit {
    flag: StopFlag,
    left: std::cell::Cell<u32>,
}

impl Stop for FlagOrLimit {
    fn requested(&self) -> bool {
        let left = self.left.get();
        assert!(left > 0, "the stop flag was never honoured");
        self.left.set(left - 1);
        self.flag.requested()
    }
}

fn run(config: &ValidConfig, lcd: &StopDuringUpload, log: &Lines) -> LoopExit {
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let mut clock = FakeClock {
        mono: Instant::now(),
        wall: SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000),
    };
    let sink = lcd.clone();
    service::run_loop(LoopInput {
        config,
        sampler: Moving(0),
        clock: &mut clock,
        notify: &mut Quiet,
        stop: FlagOrLimit {
            flag: StopFlag::new(&lcd.flag),
            left: std::cell::Cell::new(10_000),
        },
        log: log.clone(),
        open: move || Ok(sink.clone()),
        assets: &mut assets,
        latch_at_start: false,
    })
}

fn shown(lcd: &StopDuringUpload) -> (usize, usize, usize) {
    let fake = lcd.lcd();
    let records = fake.records();
    let slots = records
        .iter()
        .filter(|r| matches!(r, Record::Cmd(Cmd::ShowSlot(_))))
        .count();
    let bulks = records
        .iter()
        .filter(|r| matches!(r, Record::Bulk(_)))
        .count();
    let liquids = records
        .iter()
        .filter(|r| matches!(r, Record::Cmd(Cmd::ShowLiquid)))
        .count();
    (slots, bulks, liquids)
}

fn assert_clean_stop(lcd: &StopDuringUpload, log: &Lines, exit: LoopExit, uploads: usize) {
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(exit.code(), 0, "a requested stop is a clean exit");
    assert_eq!(
        *lcd.uploads.lock().unwrap_or_else(|err| err.into_inner()),
        uploads,
        "no upload starts after the stop request"
    );
    assert!(
        !*lcd
            .started_after_stop
            .lock()
            .unwrap_or_else(|err| err.into_inner()),
        "an upload started while the flag was present"
    );
    let (slots, bulks, liquids) = shown(lcd);
    assert_eq!(
        (slots, bulks),
        (uploads, uploads),
        "the upload in flight ran to its end (data and ShowSlot)"
    );
    assert_eq!(liquids, 0, "stock is ExecStopPost's job, not the loop's");
    let lines = log.0.lock().unwrap_or_else(|err| err.into_inner()).clone();
    assert!(
        lines
            .iter()
            .any(|line| line.contains("stop requested; no upload in flight")),
        "{lines:?}"
    );
}

#[test]
fn stop_flag_is_requested_only_while_the_file_exists() {
    let scratch = Scratch::new("flag");
    let path = scratch.0.join("stop");
    let flag = StopFlag::new(&path);
    assert!(!flag.requested(), "absent file");
    std::fs::write(&path, b"").expect("flag");
    assert!(flag.requested(), "present file");
    std::fs::remove_file(&path).expect("clear");
    assert!(!flag.requested(), "removed again");
    let missing_dir = StopFlag::new(&scratch.0.join("gone").join("stop"));
    assert!(!missing_dir.requested(), "missing runtime directory");
}

#[test]
fn stream_stop_during_an_upload_finishes_it_and_starts_no_other() {
    let scratch = Scratch::new("stream");
    let config = config(&scratch.0, "mode = \"stream\"\nstream_fps = 10\n");
    let lcd = StopDuringUpload::new(&scratch.0.join("stop"), 3);
    let log = Lines::default();
    let exit = run(&config, &lcd, &log);
    assert_clean_stop(&lcd, &log, exit, 3);
}

#[test]
fn change_mode_stop_during_an_upload_finishes_it_and_starts_no_other() {
    let scratch = Scratch::new("change");
    let config = config(&scratch.0, "min_interval_s = 10\n");
    let lcd = StopDuringUpload::new(&scratch.0.join("stop"), 2);
    let log = Lines::default();
    let exit = run(&config, &lcd, &log);
    assert_clean_stop(&lcd, &log, exit, 2);
}

#[test]
fn stop_flag_before_the_first_tick_opens_nothing() {
    let scratch = Scratch::new("early");
    let config = config(&scratch.0, "mode = \"stream\"\nstream_fps = 10\n");
    let lcd = StopDuringUpload::new(&scratch.0.join("stop"), usize::MAX);
    std::fs::write(&lcd.flag, b"").expect("flag");
    let exit = run(&config, &lcd, &Lines::default());
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(
        *lcd.uploads.lock().unwrap_or_else(|err| err.into_inner()),
        0
    );
    assert!(lcd.lcd().records().is_empty(), "no device I/O at all");
}

/// The unit's `ExecStop=` creates exactly the path the writer watches, inside
/// the runtime directory systemd creates for it, waits for the main PID, and
/// keeps the `ExecStopPost` restore after it.
#[test]
fn unit_exec_stop_creates_the_flag_the_writer_watches() {
    let unit_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packaging/kraken-lcd.service");
    let unit = std::fs::read_to_string(unit_path).expect("unit");
    let directive = |key: &str| -> Vec<String> {
        unit.lines()
            .filter_map(|line| line.strip_prefix(key))
            .map(str::to_owned)
            .collect()
    };
    let stops = directive("ExecStop=");
    assert_eq!(stops.len(), 1, "{stops:?}");
    let stop = &stops[0];
    assert!(
        stop.starts_with('-'),
        "a failed ExecStop must not fail the stop"
    );
    assert!(
        stop.contains(&format!("/usr/bin/touch {STOP_FLAG} ")),
        "{stop}"
    );
    assert!(stop.contains("--pid=${MAINPID}"), "{stop}");
    assert!(stop.contains("/usr/bin/timeout 5 "), "bounded wait: {stop}");
    let dir = Path::new(STOP_FLAG)
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .expect("runtime dir name");
    assert_eq!(directive("RuntimeDirectory="), [dir.to_owned()]);
    assert_eq!(directive("RuntimeDirectoryMode="), ["0700".to_owned()]);
    assert!(STOP_FLAG.starts_with("/run/"));
    assert!(
        directive("RuntimeDirectoryPreserve=").is_empty(),
        "the flag must not survive into the next start"
    );
    assert_eq!(directive("ExecStopPost=").len(), 1, "restore-stock stays");
    let timeout_stop: u64 = directive("TimeoutStopSec=")[0].parse().expect("secs");
    assert!(
        timeout_stop > 5,
        "systemd waits longer than the ExecStop bound"
    );
}
