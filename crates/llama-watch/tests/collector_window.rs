//! Windowed collector: a 1 s CPU window at 10 Hz, an injected llama view,
//! and watcher-only GPU extras.
//!
//! Host numbers come from a fake `/proc` and the fixture `/sys`. Nothing
//! here opens a device node or calls llama-swap.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use llama_core::sample::{AiState, LlamaView, ModelInfo};
use llama_watch::collector::{GpuExtra, WatchCollector, WatchSample};
use llama_watch::config::{Config, ValidWatchConfig};
use llama_watch::sources::gpu::{GpuBackend, GpuError};
use llama_watch::sources::{Roots, SourceId};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "collector-window-{label}-{}-{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("scratch dir");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct ProcDir {
    dir: PathBuf,
}

impl ProcDir {
    fn new(scratch: &Scratch) -> Self {
        let dir = scratch.path().join("proc");
        std::fs::create_dir_all(&dir).expect("proc dir");
        std::fs::copy(fixtures().join("proc/mem/meminfo"), dir.join("meminfo")).expect("meminfo");
        Self { dir }
    }

    fn set_stat(&self, text: &str) {
        std::fs::write(self.dir.join("stat"), text).expect("write stat");
    }

    fn roots(&self) -> Roots {
        Roots {
            proc: self.dir.clone(),
            sys: fixtures().join("sys"),
        }
    }
}

/// Cumulative `/proc/stat` after `steps` intervals of 100 ticks.
///
/// Step 0 is 100 idle ticks on every CPU. Each later step adds 100 ticks
/// split between user and idle, so a delta across any number of steps is
/// exactly `busy_pct`.
fn stat_after(busy_pct: &[u32], steps: u64) -> String {
    let mut text = String::from("cpu  0 0 0 0 0 0 0 0\n");
    for (index, pct) in busy_pct.iter().enumerate() {
        assert!(*pct <= 100, "busy percent {pct}");
        let busy = u64::from(*pct) * steps;
        let idle = 100 + (100 - u64::from(*pct)) * steps;
        text.push_str(&format!("cpu{index} {busy} 0 0 {idle} 0 0 0 0\n"));
    }
    text
}

fn one_cpu(user: u64, idle: u64) -> String {
    format!("cpu  0 0 0 0 0 0 0 0\ncpu0 {user} 0 0 {idle} 0 0 0 0\n")
}

fn stray_thread() -> Vec<u32> {
    let mut busy = vec![0; 32];
    busy[0] = 100;
    busy
}

#[derive(Clone, Copy)]
struct FakeGpu {
    util: Result<f32, &'static str>,
    temp: Result<f32, &'static str>,
    memory: Result<(u64, u64), &'static str>,
    power: Result<u32, &'static str>,
    limit: Result<u32, &'static str>,
}

impl FakeGpu {
    fn steady(util: f32, temp: f32) -> Self {
        Self {
            util: Ok(util),
            temp: Ok(temp),
            memory: Ok((1_000, 2_000)),
            power: Ok(30_000),
            limit: Ok(100_000),
        }
    }

    fn failing() -> Self {
        Self {
            util: Err("gpu lost"),
            temp: Ok(50.0),
            memory: Ok((1, 2)),
            power: Ok(3),
            limit: Ok(4),
        }
    }
}

impl GpuBackend for FakeGpu {
    fn init(&mut self) -> Result<(), GpuError> {
        Ok(())
    }

    fn util_pct(&mut self) -> Result<f32, GpuError> {
        self.util.map_err(GpuError::new)
    }

    fn temp_c(&mut self) -> Result<f32, GpuError> {
        self.temp.map_err(GpuError::new)
    }

    fn memory_info(&mut self) -> Result<(u64, u64), GpuError> {
        self.memory.map_err(GpuError::new)
    }

    fn power_usage(&mut self) -> Result<u32, GpuError> {
        self.power.map_err(GpuError::new)
    }

    fn enforced_power_limit(&mut self) -> Result<u32, GpuError> {
        self.limit.map_err(GpuError::new)
    }

    fn disconnect(&mut self) {}
}

#[derive(Clone, Default)]
struct Capture {
    lines: Arc<Mutex<Vec<String>>>,
}

impl Capture {
    fn lines(&self) -> Vec<String> {
        self.lines
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone()
    }
}

impl llama_core::log::Sink for Capture {
    fn write_line(&mut self, line: &str) {
        self.lines
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(line.to_owned());
    }
}

fn watch_config(dir: &Path, cpu_top_k: u32, cpu_window_s: f64) -> ValidWatchConfig {
    let path = dir.join("watch.toml");
    let text = format!(
        "[collector]\ntick_s = 0.1\ncpu_window_s = {cpu_window_s}\ncpu_top_k = {cpu_top_k}\n\
         [llama]\nurl = \"http://127.0.0.1:9\"\n\
         [load]\nsmooth_s = 0\n"
    );
    std::fs::write(&path, text).expect("write watch.toml");
    Config::load_validated(&path, 32).expect("valid watch config")
}

fn idle_view() -> LlamaView {
    LlamaView {
        ai: AiState::Idle,
        models: Vec::new(),
        decoded_total: None,
        prompt_total: None,
    }
}

fn sample_steps(busy_pct: &[u32], gpu: FakeGpu, cpu_top_k: u32, steps: u64) -> WatchSample {
    let scratch = Scratch::new("steps");
    let proc_dir = ProcDir::new(&scratch);
    let config = watch_config(scratch.path(), cpu_top_k, 1.0);
    let mut collector = WatchCollector::new(proc_dir.roots(), gpu, &config, Capture::default());
    let start = Instant::now();
    let wall = SystemTime::UNIX_EPOCH + Duration::from_millis(1_700_000_000_000);
    let mut sampled = None;
    for nth in 0..=steps {
        proc_dir.set_stat(&stat_after(busy_pct, nth));
        let now = start + Duration::from_millis(100 * nth);
        sampled = Some(collector.sample(now, wall, &idle_view()));
    }
    sampled.expect("at least the step-0 sample")
}

#[test]
fn cpu_percent_is_none_until_the_window_then_uses_that_delta() {
    // Nine idle steps, then one fully busy step. The adjacent delta at 1.0 s
    // is 100%. The 1 s window is 10%.
    let scratch = Scratch::new("window");
    let proc_dir = ProcDir::new(&scratch);
    let config = watch_config(scratch.path(), 1, 1.0);
    let mut collector = WatchCollector::new(
        proc_dir.roots(),
        FakeGpu::steady(0.0, 40.0),
        &config,
        Capture::default(),
    );
    let start = Instant::now();
    let wall = SystemTime::UNIX_EPOCH + Duration::from_millis(50);
    let step = Duration::from_millis(100);
    let view = idle_view();
    for nth in 0u32..=9 {
        let idle = u64::from(nth) * 100;
        let (user, idle) = if nth == 0 { (0, 0) } else { (0, idle) };
        proc_dir.set_stat(&one_cpu(user, idle));
        let sampled = collector.sample(start + step * nth, wall, &view);
        assert_eq!(
            sampled.snapshot.cpu_pct, None,
            "step {nth} is still inside the first second"
        );
        assert_eq!(sampled.snapshot.cpu_topk_pct, None, "step {nth}");
        assert_eq!(
            sampled.snapshot.load,
            Some(0.0),
            "gpu alone until the window fills"
        );
    }
    proc_dir.set_stat(&one_cpu(100, 900));
    let at_one = collector.sample(start + step * 10, wall, &view);
    assert_eq!(at_one.snapshot.cpu_pct, Some(10.0));
    assert_eq!(at_one.snapshot.cpu_topk_pct, Some(10.0));
    assert_eq!(at_one.snapshot.load, Some(10.0));
    assert!(!at_one.snapshot.errors.contains(&SourceId::ProcCpu));

    // 1.1 s. Baseline slides to the 0.1 s sample (user 0, idle 100).
    // Delta to (user 200, idle 900) is 20%, not the adjacent 100%.
    proc_dir.set_stat(&one_cpu(200, 900));
    let slid = collector.sample(start + step * 11, wall + step, &view);
    assert_eq!(slid.snapshot.cpu_pct, Some(20.0));
    assert_eq!(slid.snapshot.load, Some(20.0));
    assert_eq!(slid.snapshot.t_mono, start + step * 11);
    assert_eq!(slid.snapshot.t_wall, wall + step);
}

#[test]
fn shorter_cpu_window_from_config_fills_sooner() {
    let scratch = Scratch::new("half");
    let proc_dir = ProcDir::new(&scratch);
    let config = watch_config(scratch.path(), 1, 0.5);
    let mut collector = WatchCollector::new(
        proc_dir.roots(),
        FakeGpu::steady(0.0, 40.0),
        &config,
        Capture::default(),
    );
    let start = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    let view = idle_view();
    for nth in 0u64..=5 {
        proc_dir.set_stat(&stat_after(&[50], nth));
        let sampled = collector.sample(start + Duration::from_millis(100 * nth), wall, &view);
        if nth < 5 {
            assert_eq!(sampled.snapshot.cpu_pct, None, "step {nth} is under 0.5 s");
        } else {
            assert_eq!(sampled.snapshot.cpu_pct, Some(50.0));
        }
    }
}

#[test]
fn injected_llama_view_is_used_and_running_is_not_called() {
    let scratch = Scratch::new("llama");
    let proc_dir = ProcDir::new(&scratch);
    let config = watch_config(scratch.path(), 1, 1.0);
    let log = Capture::default();
    let mut collector = WatchCollector::new(
        proc_dir.roots(),
        FakeGpu::steady(1.0, 40.0),
        &config,
        log.clone(),
    );
    proc_dir.set_stat(&one_cpu(0, 100));
    let now = Instant::now();
    let wall = SystemTime::UNIX_EPOCH + Duration::from_secs(10);
    let loaded = LlamaView {
        ai: AiState::Loaded,
        models: vec![ModelInfo {
            backend: None,
            name: "Qwen\t三五B".to_owned(),
            state: "ready".to_owned(),
            full_name: None,
            detail: None,
        }],
        decoded_total: Some(42),
        prompt_total: None,
    };
    let sampled = collector.sample(now, wall, &loaded);
    assert_eq!(sampled.snapshot.ai, AiState::Loaded);
    assert_eq!(
        sampled.snapshot.models,
        vec![ModelInfo {
            backend: None,
            name: "Qwen B".to_owned(),
            state: "ready".to_owned(),
            full_name: None,
            detail: None
        }]
    );
    // The watcher snapshot does not carry the token counter. Publish does.
    assert_eq!(sampled.snapshot.tokens, None);
    assert!(!sampled.snapshot.errors.contains(&SourceId::Llama));
    assert_eq!(sampled.snapshot.t_mono, now);
    assert_eq!(sampled.snapshot.t_wall, wall);
    assert_eq!(log.lines(), vec!["<6>llama loaded".to_owned()]);

    let again = collector.sample(now, wall, &loaded);
    assert_eq!(again.snapshot.ai, AiState::Loaded);
    assert_eq!(log.lines(), vec!["<6>llama loaded".to_owned()]);

    let idle = collector.sample(now, wall, &idle_view());
    assert_eq!(idle.snapshot.ai, AiState::Idle);
    assert!(idle.snapshot.models.is_empty());
    assert_eq!(
        log.lines(),
        vec!["<6>llama loaded".to_owned(), "<6>llama idle".to_owned()]
    );
}

#[test]
fn watcher_never_produces_no_data() {
    let scratch = Scratch::new("nodata");
    let proc_dir = ProcDir::new(&scratch);
    let config = watch_config(scratch.path(), 1, 1.0);
    let mut collector = WatchCollector::new(
        proc_dir.roots(),
        FakeGpu::steady(1.0, 40.0),
        &config,
        Capture::default(),
    );
    proc_dir.set_stat(&one_cpu(0, 100));
    let view = LlamaView {
        ai: AiState::NoData,
        models: vec![ModelInfo {
            backend: None,
            name: "should-drop".to_owned(),
            state: "ready".to_owned(),
            full_name: None,
            detail: None,
        }],
        decoded_total: Some(9),
        prompt_total: None,
    };
    let sampled = collector.sample(Instant::now(), SystemTime::UNIX_EPOCH, &view);
    assert_ne!(sampled.snapshot.ai, AiState::NoData);
    assert_eq!(sampled.snapshot.ai, AiState::Down);
    assert!(sampled.snapshot.models.is_empty());
    assert!(sampled.snapshot.errors.contains(&SourceId::Llama));
    assert_eq!(sampled.snapshot.tokens, None);
}

#[test]
fn load_is_gpu_when_gpu_exceeds_the_windowed_cpu_topk() {
    let snap = sample_steps(&stray_thread(), FakeGpu::steady(40.0, 61.0), 8, 10).snapshot;
    assert_eq!(snap.cpu_pct, Some(3.125));
    assert_eq!(snap.cpu_topk_pct, Some(12.5));
    assert_eq!(snap.gpu_pct, Some(40.0));
    assert_eq!(snap.load, Some(40.0));
}

#[test]
fn load_is_cpu_topk_when_it_exceeds_gpu() {
    let snap = sample_steps(&stray_thread(), FakeGpu::steady(5.0, 61.0), 8, 10).snapshot;
    assert_eq!(snap.cpu_topk_pct, Some(12.5));
    assert_eq!(snap.gpu_pct, Some(5.0));
    assert_eq!(snap.load, Some(12.5));
    // No zenergy here, so activity is the GPU's share alone: 30 W at the
    // 30 W default floor is 0%, not util and not the CPU top-k.
    assert_eq!(snap.cpu_pct, Some(3.125));
    assert_eq!(snap.activity, Some(0.0));
}

#[test]
fn load_is_cpu_topk_when_gpu_is_missing() {
    let snap = sample_steps(&stray_thread(), FakeGpu::failing(), 8, 10).snapshot;
    assert_eq!(snap.gpu_pct, None);
    assert_eq!(snap.gpu_c, None);
    assert!(snap.errors.contains(&SourceId::Gpu));
    assert_eq!(snap.cpu_topk_pct, Some(12.5));
    assert_eq!(snap.load, Some(12.5));
    assert_eq!(snap.cpu_pct, Some(3.125));
    assert_eq!(snap.activity, Some(3.125));
}

#[test]
fn load_is_gpu_when_the_cpu_window_is_not_full() {
    let snap = sample_steps(&stray_thread(), FakeGpu::steady(40.0, 61.0), 8, 0).snapshot;
    assert_eq!(snap.cpu_topk_pct, None);
    assert!(!snap.errors.contains(&SourceId::ProcCpu));
    assert_eq!(snap.gpu_pct, Some(40.0));
    assert_eq!(snap.load, Some(40.0));
}

#[test]
fn load_is_none_when_gpu_and_cpu_topk_are_both_missing() {
    let snap = sample_steps(&stray_thread(), FakeGpu::failing(), 8, 0).snapshot;
    assert_eq!(snap.cpu_topk_pct, None);
    assert_eq!(snap.gpu_pct, None);
    assert_eq!(snap.load, None);
}

#[test]
fn cpu_topk_follows_cpu_top_k_from_the_watch_config() {
    let snap = sample_steps(&stray_thread(), FakeGpu::steady(0.0, 50.0), 4, 10).snapshot;
    assert_eq!(snap.cpu_pct, Some(3.125));
    assert_eq!(snap.cpu_topk_pct, Some(25.0));
}

#[test]
fn host_sources_match_the_fixtures() {
    let snap = sample_steps(&stray_thread(), FakeGpu::steady(40.0, 61.0), 8, 10).snapshot;
    assert_eq!(snap.mem_pct, Some(75.0));
    assert_eq!(snap.coolant_c, Some(31.25));
    assert_eq!(snap.cpu_c, Some(77.5));
    assert_eq!(snap.gpu_c, Some(61.0));
    assert!(snap.errors.is_empty(), "{:?}", snap.errors);
}

#[test]
fn gpu_extra_reports_values_and_a_failed_call_is_none() {
    let scratch = Scratch::new("extra");
    let proc_dir = ProcDir::new(&scratch);
    let config = watch_config(scratch.path(), 1, 1.0);
    let mut gpu = FakeGpu::steady(12.0, 55.0);
    gpu.memory = Ok((111, 222));
    gpu.power = Err("power sensor");
    gpu.limit = Ok(333);
    let mut collector = WatchCollector::new(proc_dir.roots(), gpu, &config, Capture::default());
    proc_dir.set_stat(&one_cpu(0, 100));
    let sampled = collector.sample(Instant::now(), SystemTime::UNIX_EPOCH, &idle_view());
    assert_eq!(sampled.snapshot.gpu_pct, Some(12.0));
    assert_eq!(sampled.snapshot.gpu_c, Some(55.0));
    assert!(!sampled.snapshot.errors.contains(&SourceId::Gpu));
    assert_eq!(
        sampled.gpu,
        GpuExtra {
            vram_used: Some(111),
            vram_total: Some(222),
            power_mw: None,
            power_limit_mw: Some(333),
        }
    );
}

#[derive(Clone)]
struct ToggleMemory {
    error: Arc<Mutex<Option<&'static str>>>,
}

impl ToggleMemory {
    fn failing(message: &'static str) -> Self {
        Self {
            error: Arc::new(Mutex::new(Some(message))),
        }
    }

    fn succeed(&self) {
        *self.error.lock().unwrap_or_else(|err| err.into_inner()) = None;
    }
}

impl GpuBackend for ToggleMemory {
    fn init(&mut self) -> Result<(), GpuError> {
        Ok(())
    }

    fn util_pct(&mut self) -> Result<f32, GpuError> {
        Ok(12.0)
    }

    fn temp_c(&mut self) -> Result<f32, GpuError> {
        Ok(55.0)
    }

    fn memory_info(&mut self) -> Result<(u64, u64), GpuError> {
        match *self.error.lock().unwrap_or_else(|err| err.into_inner()) {
            Some(message) => Err(GpuError::new(message)),
            None => Ok((9, 10)),
        }
    }

    fn power_usage(&mut self) -> Result<u32, GpuError> {
        Ok(1)
    }

    fn enforced_power_limit(&mut self) -> Result<u32, GpuError> {
        Ok(2)
    }

    fn disconnect(&mut self) {}
}

#[test]
fn gpu_extra_failure_is_logged_once_then_recovers() {
    let scratch = Scratch::new("extra-log");
    let proc_dir = ProcDir::new(&scratch);
    let config = watch_config(scratch.path(), 1, 1.0);
    let log = Capture::default();
    let toggle = ToggleMemory::failing("vram lost");
    let handle = toggle.clone();
    let mut collector = WatchCollector::new(proc_dir.roots(), toggle, &config, log.clone());
    proc_dir.set_stat(&one_cpu(0, 100));
    let now = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    let view = idle_view();
    collector.sample(now, wall, &view);
    collector.sample(now, wall, &view);
    let vram: Vec<_> = log
        .lines()
        .into_iter()
        .filter(|line| line.contains("vram"))
        .collect();
    assert_eq!(vram, vec!["<3>gpu failed: vram: vram lost".to_owned()]);

    handle.succeed();
    collector.sample(now, wall, &view);
    collector.sample(now, wall, &view);
    let vram: Vec<_> = log
        .lines()
        .into_iter()
        .filter(|line| line.contains("vram"))
        .collect();
    assert_eq!(
        vram,
        vec![
            "<3>gpu failed: vram: vram lost".to_owned(),
            "<6>gpu vram recovered".to_owned(),
        ]
    );
}

#[test]
fn a_repeated_host_failure_is_logged_once() {
    let scratch = Scratch::new("mem-log");
    let proc_dir = ProcDir::new(&scratch);
    std::fs::remove_file(proc_dir.dir.join("meminfo")).expect("drop meminfo");
    let config = watch_config(scratch.path(), 1, 1.0);
    let log = Capture::default();
    let mut collector = WatchCollector::new(
        proc_dir.roots(),
        FakeGpu::steady(1.0, 40.0),
        &config,
        log.clone(),
    );
    proc_dir.set_stat(&one_cpu(0, 100));
    let now = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    let view = idle_view();
    let first = collector.sample(now, wall, &view);
    collector.sample(now, wall, &view);
    assert_eq!(first.snapshot.mem_pct, None);
    assert!(first.snapshot.errors.contains(&SourceId::ProcMem));
    assert_eq!(first.snapshot.gpu_pct, Some(1.0), "gpu still samples");
    let mem: Vec<_> = log
        .lines()
        .into_iter()
        .filter(|line| line.contains("proc.mem"))
        .collect();
    assert_eq!(mem.len(), 1, "{mem:?}");
    assert!(mem[0].starts_with("<3>proc.mem failed:"), "{mem:?}");

    std::fs::copy(
        fixtures().join("proc/mem/meminfo"),
        proc_dir.dir.join("meminfo"),
    )
    .expect("restore meminfo");
    collector.sample(now, wall, &view);
    let mem: Vec<_> = log
        .lines()
        .into_iter()
        .filter(|line| line.contains("proc.mem"))
        .collect();
    assert_eq!(mem.len(), 2, "{mem:?}");
    assert!(mem[0].starts_with("<3>proc.mem failed:"), "{mem:?}");
    assert_eq!(mem[1], "<6>proc.mem recovered");
}

#[test]
fn host_percentages_are_clamped_into_0_to_100() {
    let scratch = Scratch::new("clamp");
    let proc_dir = ProcDir::new(&scratch);
    std::fs::write(
        proc_dir.dir.join("meminfo"),
        "MemTotal:       1000 kB\nMemAvailable:    2000 kB\n",
    )
    .expect("write meminfo");
    let config = watch_config(scratch.path(), 1, 1.0);
    let mut collector = WatchCollector::new(
        proc_dir.roots(),
        FakeGpu::steady(150.0, 40.0),
        &config,
        Capture::default(),
    );
    proc_dir.set_stat(&one_cpu(0, 100));
    let sampled = collector.sample(Instant::now(), SystemTime::UNIX_EPOCH, &idle_view());
    assert_eq!(sampled.snapshot.mem_pct, Some(0.0));
    assert_eq!(sampled.snapshot.gpu_pct, Some(100.0));
    assert_eq!(sampled.snapshot.load, Some(100.0));
    assert!(!sampled.snapshot.errors.contains(&SourceId::ProcMem));
}

fn view_with(ai: AiState) -> LlamaView {
    LlamaView {
        ai,
        models: Vec::new(),
        decoded_total: None,
        prompt_total: None,
    }
}

fn lines_containing(log: &Capture, needle: &str) -> Vec<String> {
    log.lines()
        .into_iter()
        .filter(|line| line.contains(needle))
        .collect()
}

#[test]
fn llama_down_is_logged_once() {
    let scratch = Scratch::new("llama-down");
    let proc_dir = ProcDir::new(&scratch);
    let config = watch_config(scratch.path(), 1, 1.0);
    let log = Capture::default();
    let mut collector = WatchCollector::new(
        proc_dir.roots(),
        FakeGpu::steady(1.0, 40.0),
        &config,
        log.clone(),
    );
    proc_dir.set_stat(&one_cpu(0, 100));
    let now = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    let down = view_with(AiState::Down);
    collector.sample(now, wall, &down);
    collector.sample(now, wall, &down);
    assert_eq!(
        lines_containing(&log, "llama"),
        vec!["<3>llama down".to_owned()]
    );
}

#[test]
fn llama_down_reason_change_is_quiet() {
    let scratch = Scratch::new("llama-reason");
    let proc_dir = ProcDir::new(&scratch);
    let config = watch_config(scratch.path(), 1, 1.0);
    let log = Capture::default();
    let mut collector = WatchCollector::new(
        proc_dir.roots(),
        FakeGpu::steady(1.0, 40.0),
        &config,
        log.clone(),
    );
    proc_dir.set_stat(&one_cpu(0, 100));
    let now = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    collector.sample(now, wall, &view_with(AiState::Down));
    collector.sample(now, wall, &view_with(AiState::NoData));
    assert_eq!(
        lines_containing(&log, "llama"),
        vec!["<3>llama down".to_owned()]
    );
}

struct MessageGpu {
    message: Arc<Mutex<&'static str>>,
    up: Arc<Mutex<bool>>,
}

impl GpuBackend for MessageGpu {
    fn init(&mut self) -> Result<(), GpuError> {
        Ok(())
    }

    fn util_pct(&mut self) -> Result<f32, GpuError> {
        if *self.up.lock().unwrap_or_else(|err| err.into_inner()) {
            Ok(12.0)
        } else {
            Err(GpuError::new(
                *self.message.lock().unwrap_or_else(|err| err.into_inner()),
            ))
        }
    }

    fn temp_c(&mut self) -> Result<f32, GpuError> {
        Ok(40.0)
    }

    fn memory_info(&mut self) -> Result<(u64, u64), GpuError> {
        Ok((1, 2))
    }

    fn power_usage(&mut self) -> Result<u32, GpuError> {
        Ok(3)
    }

    fn enforced_power_limit(&mut self) -> Result<u32, GpuError> {
        Ok(4)
    }

    fn disconnect(&mut self) {}
}

#[test]
fn a_different_gpu_failure_text_is_not_logged_again() {
    let scratch = Scratch::new("gpu-text");
    let proc_dir = ProcDir::new(&scratch);
    let config = watch_config(scratch.path(), 1, 1.0);
    let log = Capture::default();
    let message = Arc::new(Mutex::new("sensor a"));
    let up = Arc::new(Mutex::new(false));
    let mut collector = WatchCollector::new(
        proc_dir.roots(),
        MessageGpu {
            message: Arc::clone(&message),
            up: Arc::clone(&up),
        },
        &config,
        log.clone(),
    );
    proc_dir.set_stat(&one_cpu(0, 100));
    let now = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    let view = idle_view();
    collector.sample(now, wall, &view);
    *message.lock().unwrap_or_else(|err| err.into_inner()) = "sensor b";
    collector.sample(now, wall, &view);
    let gpu = lines_containing(&log, "gpu failed");
    assert_eq!(gpu.len(), 1, "{gpu:?}");
    assert!(gpu[0].contains("sensor a"), "{gpu:?}");
    assert!(!gpu[0].contains("sensor b"), "{gpu:?}");
}

#[test]
fn gpu_util_failure_and_recovery_are_logged_once() {
    let scratch = Scratch::new("gpu-util");
    let proc_dir = ProcDir::new(&scratch);
    let config = watch_config(scratch.path(), 1, 1.0);
    let log = Capture::default();
    let up = Arc::new(Mutex::new(false));
    let mut collector = WatchCollector::new(
        proc_dir.roots(),
        MessageGpu {
            message: Arc::new(Mutex::new("util lost")),
            up: Arc::clone(&up),
        },
        &config,
        log.clone(),
    );
    proc_dir.set_stat(&one_cpu(0, 100));
    let now = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    let view = idle_view();
    collector.sample(now, wall, &view);
    collector.sample(now, wall, &view);
    let failed = lines_containing(&log, "gpu failed");
    assert_eq!(failed.len(), 1, "{failed:?}");
    assert!(failed[0].contains("util lost"), "{failed:?}");

    *up.lock().unwrap_or_else(|err| err.into_inner()) = true;
    collector.sample(now, wall, &view);
    collector.sample(now, wall, &view);
    let recovered = lines_containing(&log, "gpu recovered");
    assert_eq!(recovered, vec!["<6>gpu recovered".to_owned()]);
}

fn copy_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("mkdir");
    for entry in std::fs::read_dir(src).expect("read_dir") {
        let entry = entry.expect("entry");
        let to = dst.join(entry.file_name());
        if entry.file_type().expect("file type").is_dir() {
            copy_tree(&entry.path(), &to);
        } else {
            std::fs::copy(entry.path(), &to).expect("copy");
        }
    }
}

fn roots_with_sys(proc_dir: &ProcDir, sys: &Path) -> Roots {
    Roots {
        proc: proc_dir.dir.clone(),
        sys: sys.to_path_buf(),
    }
}

#[test]
fn coolant_cpu_temp_and_proc_cpu_failures_leave_the_other_sources() {
    let scratch = Scratch::new("host-fail");
    let proc_dir = ProcDir::new(&scratch);
    let config = watch_config(scratch.path(), 1, 1.0);
    let now = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    let view = idle_view();

    let coolant_sys = scratch.path().join("sys-coolant");
    copy_tree(&fixtures().join("sys"), &coolant_sys);
    std::fs::remove_dir_all(coolant_sys.join("class/hwmon/hwmon5")).expect("drop z53");
    let mut collector = WatchCollector::new(
        roots_with_sys(&proc_dir, &coolant_sys),
        FakeGpu::steady(10.0, 40.0),
        &config,
        Capture::default(),
    );
    proc_dir.set_stat(&one_cpu(0, 100));
    let snap = collector.sample(now, wall, &view).snapshot;
    assert!(
        snap.errors.contains(&SourceId::HwmonCoolant),
        "{:?}",
        snap.errors
    );
    assert_eq!(snap.coolant_c, None);
    assert_eq!(snap.mem_pct, Some(75.0));
    assert_eq!(snap.gpu_pct, Some(10.0));
    assert_eq!(snap.cpu_c, Some(77.5));

    let cpu_sys = scratch.path().join("sys-cpu");
    copy_tree(&fixtures().join("sys"), &cpu_sys);
    std::fs::remove_file(cpu_sys.join("class/hwmon/hwmon3/temp1_label")).expect("drop Tctl");
    let mut collector = WatchCollector::new(
        roots_with_sys(&proc_dir, &cpu_sys),
        FakeGpu::steady(10.0, 40.0),
        &config,
        Capture::default(),
    );
    let snap = collector.sample(now, wall, &view).snapshot;
    assert!(
        snap.errors.contains(&SourceId::HwmonCpu),
        "{:?}",
        snap.errors
    );
    assert_eq!(snap.cpu_c, None);
    assert_eq!(snap.coolant_c, Some(31.25));
    assert_eq!(snap.mem_pct, Some(75.0));

    let mut collector = WatchCollector::new(
        proc_dir.roots(),
        FakeGpu::steady(10.0, 40.0),
        &config,
        Capture::default(),
    );
    proc_dir.set_stat("intr 1\nctxt 2\n");
    let snap = collector.sample(now, wall, &view).snapshot;
    assert!(
        snap.errors.contains(&SourceId::ProcCpu),
        "{:?}",
        snap.errors
    );
    assert_eq!(snap.cpu_pct, None);
    assert_eq!(snap.cpu_topk_pct, None);
    assert_eq!(snap.mem_pct, Some(75.0));
    assert_eq!(snap.coolant_c, Some(31.25));
    assert_eq!(snap.gpu_pct, Some(10.0));
}
