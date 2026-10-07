#![allow(dead_code)]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use llama_core::log::Sink;
use llama_core::wire::{Ai, AiWire, Host, SCHEMA, SnapshotV1, Tokens};
use llama_light::aura::device::{AuraOpener, AuraPort, OpenError, PortError};
use llama_light::aura::proto::{EncodedReport, REPORT_LEN};
use llama_light::config::{ConfigError, ConfigSource, LightConfig, Stamp};
use llama_light::service::{Clock, Notifier};
use llama_light::snapshot::SnapshotSource;
use std::borrow::Cow;

pub const SEC: u64 = 1_000_000_000;

pub fn snap(seq: u64, t_mono_ns: u64) -> SnapshotV1 {
    SnapshotV1 {
        schema: SCHEMA,
        run_id: 7,
        seq,
        t_mono_ns,
        t_wall_ms: 1,
        host: Host {
            load_pct: Some(10.0),
            activity_pct: Some(50.0),
            cpu_pct: Some(20.0),
            cpu_topk_pct: Some(30.0),
            gpu_pct: Some(40.0),
            mem_pct: Some(60.0),
            coolant_c: Some(33.0),
            cpu_c: Some(55.0),
            gpu_c: Some(60.0),
            gpu_w: None,
            gpu_limit_w: None,
            cpu_w: None,
            vram_used_bytes: None,
            vram_total_bytes: None,
            mem_used_bytes: None,
            mem_total_bytes: None,
        },
        ai: Ai {
            state: AiWire::Idle,
            models: Vec::new(),
        },
        tokens: Tokens {
            decoded_total: Some(1000),
            prompt_total: None,
        },
        fans: Vec::new(),
        sources: None,
        suspected_loads: Vec::new(),
    }
}

/// Captured log lines.
#[derive(Clone, Default)]
pub struct Lines(pub Rc<RefCell<Vec<String>>>);

impl Lines {
    pub fn all(&self) -> Vec<String> {
        self.0.borrow().clone()
    }
    pub fn count(&self, needle: &str) -> usize {
        self.0
            .borrow()
            .iter()
            .filter(|l| l.contains(needle))
            .count()
    }
}

impl Sink for Lines {
    fn write_line(&mut self, line: &str) {
        self.0.borrow_mut().push(line.to_owned());
    }
}

/// Manual clock: `sleep` advances it.
#[derive(Clone)]
pub struct ManualClock(pub Rc<RefCell<u64>>);

impl ManualClock {
    pub fn at(ns: u64) -> Self {
        Self(Rc::new(RefCell::new(ns)))
    }
    pub fn now(&self) -> u64 {
        *self.0.borrow()
    }
}

impl Clock for ManualClock {
    fn now_ns(&self) -> u64 {
        *self.0.borrow()
    }
    fn sleep(&mut self, d: Duration) {
        *self.0.borrow_mut() += d.as_nanos() as u64;
    }
}

pub struct NoNotify;
impl Notifier for NoNotify {
    fn ready(&mut self) {}
    fn watchdog(&mut self) {}
}

/// Snapshot source that returns what the test put in it.
#[derive(Clone)]
pub struct Feed(pub Rc<RefCell<Result<SnapshotV1, &'static str>>>);

impl Feed {
    pub fn new(first: SnapshotV1) -> Self {
        Self(Rc::new(RefCell::new(Ok(first))))
    }
    pub fn missing() -> Self {
        Self(Rc::new(RefCell::new(Err("snapshot missing"))))
    }
    pub fn set(&self, value: Result<SnapshotV1, &'static str>) {
        *self.0.borrow_mut() = value;
    }
}

impl SnapshotSource for Feed {
    fn read(&mut self) -> Result<SnapshotV1, Cow<'static, str>> {
        self.0.borrow().clone().map_err(Cow::Borrowed)
    }
}

/// Config source the test edits: text, a stamp generation, and whether
/// the file exists.
#[derive(Clone)]
pub struct FakeConfig(pub Rc<RefCell<(String, u64, bool)>>);

impl FakeConfig {
    pub fn new(text: &str) -> Self {
        Self(Rc::new(RefCell::new((text.to_owned(), 1, true))))
    }
    /// The file disappears: no stamp, and loading fails.
    pub fn remove(&self) {
        self.0.borrow_mut().2 = false;
    }
    pub fn edit(&self, text: &str) {
        let mut inner = self.0.borrow_mut();
        inner.0 = text.to_owned();
        inner.1 += 1;
        inner.2 = true;
    }
}

impl ConfigSource for FakeConfig {
    fn stamp(&self) -> Option<Stamp> {
        let inner = self.0.borrow();
        inner.2.then(|| Stamp::from_parts(0, inner.1, 1))
    }
    fn load(&self) -> Result<LightConfig, ConfigError> {
        let inner = self.0.borrow();
        if !inner.2 {
            return Err(ConfigError("light.toml: NotFound".to_owned()));
        }
        llama_light::config::parse(&inner.0)
    }
}

/// What the fake Aura saw.
#[derive(Default)]
pub struct AuraLog {
    pub reports: Vec<[u8; REPORT_LEN]>,
    pub opens: usize,
    pub present: bool,
    pub fail_writes: bool,
}

#[derive(Clone)]
pub struct FakeOpener(pub Rc<RefCell<AuraLog>>);

impl FakeOpener {
    pub fn present() -> Self {
        Self(Rc::new(RefCell::new(AuraLog {
            present: true,
            ..AuraLog::default()
        })))
    }
    pub fn absent() -> Self {
        Self(Rc::new(RefCell::new(AuraLog::default())))
    }
    pub fn reports(&self) -> Vec<[u8; REPORT_LEN]> {
        self.0.borrow().reports.clone()
    }
    pub fn direct_reports(&self) -> usize {
        self.0
            .borrow()
            .reports
            .iter()
            .filter(|r| r[1] == 0x40)
            .count()
    }
    pub fn set_present(&self, present: bool) {
        self.0.borrow_mut().present = present;
        self.0.borrow_mut().fail_writes = !present;
    }
}

pub struct FakePort(Rc<RefCell<AuraLog>>);

impl AuraPort for FakePort {
    fn send(&mut self, report: &EncodedReport) -> Result<(), PortError> {
        let mut log = self.0.borrow_mut();
        if log.fail_writes {
            return Err(PortError("gone".to_owned()));
        }
        log.reports.push(*report.as_bytes());
        Ok(())
    }
}

impl AuraOpener for FakeOpener {
    type Port = FakePort;
    fn open(&mut self) -> Result<FakePort, OpenError> {
        let mut log = self.0.borrow_mut();
        if !log.present {
            return Err(OpenError::NoPin);
        }
        log.opens += 1;
        Ok(FakePort(self.0.clone()))
    }
}

/// A fresh scratch directory.
pub fn scratch(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("llama-light-{label}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("scratch");
    path
}

/// What the fake keyboard saw.
#[derive(Default)]
pub struct KbLog {
    pub reports: Vec<[u8; llama_light::keyboard::proto::REPORT_LEN]>,
    pub opens: usize,
    pub present: bool,
    pub fail_writes: bool,
    /// When set, `open` fails with this instead of `NoPin`.
    pub refuse: Option<OpenError>,
}

#[derive(Clone)]
pub struct FakeKbOpener(pub Rc<RefCell<KbLog>>);

impl FakeKbOpener {
    pub fn present() -> Self {
        Self(Rc::new(RefCell::new(KbLog {
            present: true,
            ..KbLog::default()
        })))
    }
    pub fn absent() -> Self {
        Self(Rc::new(RefCell::new(KbLog::default())))
    }
    pub fn reports(&self) -> Vec<[u8; llama_light::keyboard::proto::REPORT_LEN]> {
        self.0.borrow().reports.clone()
    }
    pub fn opens(&self) -> usize {
        self.0.borrow().opens
    }
    /// 24-bit commits of blue: one per frame.
    pub fn frames(&self) -> usize {
        self.0
            .borrow()
            .reports
            .iter()
            .filter(|r| r[1] == 0x07 && r[2] == 0x28 && r[3] == 3)
            .count()
    }
    pub fn set_present(&self, present: bool) {
        self.0.borrow_mut().present = present;
        self.0.borrow_mut().fail_writes = !present;
    }
    pub fn refuse(&self, err: Option<OpenError>) {
        self.0.borrow_mut().refuse = err;
    }
}

pub struct FakeKbPort(Rc<RefCell<KbLog>>);

impl llama_light::keyboard::device::KeyboardPort for FakeKbPort {
    fn send(
        &mut self,
        report: &llama_light::keyboard::proto::EncodedReport,
    ) -> Result<(), PortError> {
        let mut log = self.0.borrow_mut();
        if log.fail_writes {
            return Err(PortError("gone".to_owned()));
        }
        log.reports.push(*report.as_bytes());
        Ok(())
    }
}

impl llama_light::keyboard::device::KeyboardOpener for FakeKbOpener {
    type Port = FakeKbPort;
    fn open(&mut self) -> Result<FakeKbPort, OpenError> {
        let mut log = self.0.borrow_mut();
        if let Some(err) = log.refuse.clone() {
            return Err(err);
        }
        if !log.present {
            return Err(OpenError::NoPin);
        }
        log.opens += 1;
        Ok(FakeKbPort(self.0.clone()))
    }
}
