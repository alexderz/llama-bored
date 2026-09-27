//! S7 process level: latch survives drop and a fresh loop on the same state dir.

#[path = "device_fixture.rs"]
mod fixture;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use kraken_lcd::collector::{AiState, Snapshot};
use kraken_lcd::config::{Config, ValidConfig};
use kraken_lcd::device::{FakeLcd, LcdSink, OpenRequest, SinkError};
use kraken_lcd::log::{self, Priority};
use kraken_lcd::render::Frame;
use kraken_lcd::service::{
    self, Clock, LoopExit, LoopInput, Notifier, Sampler, Stop, latch_present,
};

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("t14-latch-{label}-{}", std::process::id()));
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
    mono: Instant,
    wall: SystemTime,
}

impl FakeClock {
    fn new() -> Self {
        Self {
            mono: Instant::now(),
            wall: SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000),
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

fn run_collect_only(
    config: &ValidConfig,
    lcd: SharedLcd,
    log: CaptureLog,
    ticks: u32,
) -> (LoopExit, u32) {
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    let mut clock = FakeClock::new();
    let mut notify = QuietNotify;
    let mut opens = 0u32;
    let exit = service::run_loop(LoopInput {
        config,
        sampler: Idle,
        clock: &mut clock,
        notify: &mut notify,
        stop: After::ticks(ticks),
        log,
        open: || {
            opens += 1;
            Ok(lcd.clone())
        },
        assets: &mut assets,
        latch_at_start: true,
    });
    (exit, opens)
}

#[test]
fn run_with_latch_is_collect_only_and_logs_one_critical() {
    let scratch = Scratch::new("run");
    std::fs::write(scratch.0.join("halted"), b"halted\n").expect("latch");
    assert!(latch_present(&scratch.0));
    let config = load_config(&scratch.0);
    let lcd = SharedLcd::new();
    let log = CaptureLog::default();
    let (exit, opens) = run_collect_only(&config, lcd.clone(), log.clone(), 3);
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(opens, 0, "latch skips every open");
    assert!(lcd.lock().records().is_empty());
    assert_eq!(log.crits().len(), 1);
}

#[test]
fn latch_survives_drop_and_a_fresh_loop() {
    let scratch = Scratch::new("reboot");
    std::fs::write(scratch.0.join("halted"), b"halted\n").expect("latch");
    let config = load_config(&scratch.0);
    let lcd = SharedLcd::new();
    let log = CaptureLog::default();
    let (exit, opens) = run_collect_only(&config, lcd.clone(), log.clone(), 2);
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(opens, 0);
    drop(log);

    let log2 = CaptureLog::default();
    let lcd2 = SharedLcd::new();
    let (exit, opens) = run_collect_only(&config, lcd2.clone(), log2.clone(), 2);
    assert_eq!(exit, LoopExit::Stopped);
    assert_eq!(opens, 0, "a fresh loop on the same latch still skips I/O");
    assert!(lcd2.lock().records().is_empty());
    assert_eq!(log2.crits().len(), 1);
}

fn restore_stock_delegates(src: &str) -> bool {
    let src = strip_rust_comments(src);
    let Some(start) = src.find("pub fn restore_stock(") else {
        return false;
    };
    let rest = &src[start..];
    let Some(end) = rest.find("\npub fn restore_stock_on") else {
        return false;
    };
    let body = &rest[..end];
    let Some(latch) = body.find("latch_present") else {
        return false;
    };
    let Some(call) = body.find("restore_stock_on(") else {
        return false;
    };
    latch < call
        && body.contains("SYS_ROOT")
        && body.contains("STATE_DIR")
        && body.contains("open_resolved_hid")
}

fn strip_rust_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut chars = src.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '/' && chars.peek() == Some(&'/') {
            chars.next();
            for next in chars.by_ref() {
                if next == '\n' {
                    out.push('\n');
                    break;
                }
            }
            continue;
        }
        if ch == '/' && chars.peek() == Some(&'*') {
            chars.next();
            while let Some(next) = chars.next() {
                if next == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    break;
                }
                if next == '\n' {
                    out.push('\n');
                }
            }
            continue;
        }
        if ch == '"' {
            out.push(ch);
            while let Some(next) = chars.next() {
                out.push(next);
                if next == '\\' {
                    if let Some(escaped) = chars.next() {
                        out.push(escaped);
                    }
                    continue;
                }
                if next == '"' {
                    break;
                }
            }
            continue;
        }
        out.push(ch);
    }
    out
}

#[test]
fn restore_stock_delegates_to_restore_stock_on() {
    let src = include_str!("../src/service.rs");
    assert!(
        restore_stock_delegates(src),
        "production restore_stock must call restore_stock_on on the constant roots"
    );
}

#[test]
fn a_commented_out_restore_stock_on_call_does_not_count() {
    let planted = "\
pub fn restore_stock() {
    if latch_present(state_dir) { return 0; }
    let _ = (SYS_ROOT, STATE_DIR);
    // restore_stock_on(&request, open_resolved_hid)
    0
}
pub fn restore_stock_on() {}
";
    assert!(
        !restore_stock_delegates(planted),
        "a commented-out call must not count as delegation"
    );
}

#[test]
fn restore_stock_on_refuses_a_latched_dir_before_the_opener() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "restore-on-latch-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("state dir");
    std::fs::write(dir.join("halted"), b"halted\n").expect("latch");
    let sys = dir.join("sys");
    std::fs::create_dir_all(&sys).expect("sys");
    let request = OpenRequest {
        sys_root: &sys,
        state_dir: &dir,
        rotate_deg: 0,
        trace_hid: false,
    };
    let code = service::restore_stock_on(
        &request,
        |_path| -> Result<fixture::FakeHid, kraken_lcd::device::PortError> {
            panic!("restore_stock_on called the opener while the latch was present");
        },
    );
    assert_eq!(code, 0);
    let _ = std::fs::remove_dir_all(&dir);

    let src = strip_rust_comments(include_str!("../src/service.rs"));
    let start = src
        .find("pub fn restore_stock_on")
        .expect("restore_stock_on");
    let rest = &src[start..];
    let end = rest.find("\n#[cfg").unwrap_or(rest.len());
    let body = &rest[..end];
    let latch = body
        .find("latch_present")
        .expect("restore_stock_on checks the latch itself");
    let open = body.find("open_restore(").expect("open_restore");
    assert!(
        latch < open,
        "restore_stock_on's latch check is before open_restore"
    );
}

#[test]
fn restore_stock_with_latch_sends_nothing() {
    let tree = fixture::Tree::new("restore-latch");
    std::fs::write(tree.state.join("halted"), b"halted\n").expect("latch");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let code = service::restore_stock_on(&tree.request(0, false), {
        let hid = Arc::clone(&hid);
        move |_path| Ok(fixture::FakeHid::attach(hid))
    });
    assert_eq!(code, 0);
    assert!(hid.sent().is_empty(), "latch: zero HID reports");
}

#[test]
fn restore_stock_without_latch_sends_show_liquid() {
    let tree = fixture::Tree::new("restore-ok");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    hid.push(fixture::ack(kraken_lcd::device::proto::expected_prefix(
        &kraken_lcd::device::proto::Cmd::ShowLiquid,
    )));
    let request = OpenRequest {
        sys_root: &tree.sys,
        state_dir: &tree.state,
        rotate_deg: 0,
        trace_hid: false,
    };
    let code = service::restore_stock_on(&request, {
        let hid = Arc::clone(&hid);
        move |_path| Ok(fixture::FakeHid::attach(hid))
    });
    assert_eq!(code, 0);
    let sent = hid.sent();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0][0..4], [0x38, 0x01, 0x02, 0x00]);
}

#[test]
fn clear_halt_as_non_root_refuses() {
    if rustix::process::geteuid().is_root() {
        return;
    }
    let out = Command::new(env!("CARGO_BIN_EXE_kraken-lcd"))
        .arg("clear-halt")
        .output()
        .expect("spawn");
    assert!(
        !out.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("clear-halt refuses unless euid is 0"),
        "stderr={stderr}"
    );
}
