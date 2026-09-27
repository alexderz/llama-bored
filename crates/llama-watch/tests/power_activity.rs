//! Bottleneck activity: zenergy socket energy, NVML watts, and a fake clock.
//!
//! Nothing here opens a device node. The sys root is a scratch directory.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

use llama_core::log::Sink;
use llama_core::sample::{AiState, LlamaView};
use llama_watch::collector::{LoadSource, WatchCollector};
use llama_watch::config::{Config, ValidWatchConfig};
use llama_watch::sources::Roots;
use llama_watch::sources::gpu::{GpuBackend, GpuError};

struct Quiet;

impl Sink for Quiet {
    fn write_line(&mut self, _line: &str) {}
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("power-activity-{label}-{}-{n}", std::process::id()));
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

struct Rig {
    scratch: Scratch,
    hwmon: PathBuf,
}

impl Rig {
    fn new(label: &str) -> Self {
        let scratch = Scratch::new(label);
        let hwmon = scratch.path().join("sys/class/hwmon/hwmon0");
        std::fs::create_dir_all(&hwmon).expect("hwmon dir");
        std::fs::write(hwmon.join("name"), "zenergy\n").expect("name");
        std::fs::write(hwmon.join("energy1_label"), "Esocket0 energy\n").expect("socket label");
        std::fs::write(hwmon.join("energy2_label"), "Ecore0 energy\n").expect("core label");
        Self { scratch, hwmon }
    }

    fn set_energy(&self, socket_uj: u64, core_uj: u64, rapl_uj: u64) {
        std::fs::write(self.hwmon.join("energy1_input"), format!("{socket_uj}\n")).expect("socket");
        std::fs::write(self.hwmon.join("energy2_input"), format!("{core_uj}\n")).expect("core");
        // RAPL `energy_uj` is root-only and must not be a load input.
        std::fs::write(self.hwmon.join("energy_uj"), format!("{rapl_uj}\n")).expect("rapl");
    }

    fn roots(&self) -> Roots {
        Roots {
            proc: self.scratch.path().join("proc"),
            sys: self.scratch.path().join("sys"),
        }
    }

    /// `load` is the `[load]` body. Unless it sets `nominal_frac`, the device
    /// fractions here are taken on the full limit (`nominal_frac = 1.0`), so
    /// the arithmetic below is plain `(w - idle) / (limit - idle)`.
    fn config(&self, load: &str) -> ValidWatchConfig {
        let path = self.scratch.path().join("watch.toml");
        let load = if load.contains("nominal_frac") {
            load.to_owned()
        } else {
            format!("{load}\nnominal_frac = 1.0\n")
        };
        let text = format!(
            "[collector]\ntick_s = 0.1\ncpu_window_s = 1.0\ncpu_top_k = 1\n\
             [llama]\nurl = \"http://127.0.0.1:9\"\n\
             [load]\n{load}\n"
        );
        std::fs::write(&path, text).expect("watch.toml");
        Config::load_validated(&path, 1).expect("valid watch config")
    }
}

#[derive(Clone, Copy)]
struct FakeGpu {
    util: f32,
    power_mw: Result<u32, &'static str>,
    limit_mw: Result<u32, &'static str>,
}

impl FakeGpu {
    fn power(util: f32, power_mw: u32, limit_mw: u32) -> Self {
        Self {
            util,
            power_mw: Ok(power_mw),
            limit_mw: Ok(limit_mw),
        }
    }
}

impl GpuBackend for FakeGpu {
    fn init(&mut self) -> Result<(), GpuError> {
        Ok(())
    }

    fn util_pct(&mut self) -> Result<f32, GpuError> {
        Ok(self.util)
    }

    fn temp_c(&mut self) -> Result<f32, GpuError> {
        Ok(40.0)
    }

    fn memory_info(&mut self) -> Result<(u64, u64), GpuError> {
        Ok((1, 2))
    }

    fn power_usage(&mut self) -> Result<u32, GpuError> {
        self.power_mw.map_err(GpuError::new)
    }

    fn enforced_power_limit(&mut self) -> Result<u32, GpuError> {
        self.limit_mw.map_err(GpuError::new)
    }

    fn disconnect(&mut self) {}
}

fn idle_view() -> LlamaView {
    LlamaView {
        ai: AiState::Idle,
        models: Vec::new(),
        decoded_total: None,
    }
}

/// Fixed floors of 0 W and no smoothing: fractions are plain watts over limits.
const ZERO_FLOORS: &str =
    "cpu_limit_w = 100\nidle = \"fixed\"\ngpu_idle_w = 0\ncpu_idle_w = 0\nsmooth_s = 0\n";

#[test]
fn socket_energy_wraparound_is_a_forward_delta() {
    // 100 W across 1 s is 100_000_000 µJ. The counter crosses u64::MAX.
    // GPU draw is 0 W, cpu_limit is 200 W and the floors are 0, so the CPU
    // fraction is 100/200 = 50% and wins. Smoothing is off.
    // An Ecore jump and a changing energy_uj must not move that number:
    // only Esocket* counts. A saturating delta would report 0 W and 0%.
    let rig = Rig::new("wrap");
    let before = u64::MAX - 25_000_000;
    let after = 74_999_999;
    rig.set_energy(before, 0, 0);
    let config = rig.config(
        "cpu_limit_w = 200\nidle = \"fixed\"\ngpu_idle_w = 0\ncpu_idle_w = 0\nsmooth_s = 0\n",
    );
    let mut collector =
        WatchCollector::new(rig.roots(), FakeGpu::power(1.0, 0, 100_000), &config, Quiet);
    let start = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    let view = idle_view();
    collector.sample(start, wall, &view);

    rig.set_energy(after, 50_000_000_000, 50_000_000_000);
    let second = collector.sample(start + Duration::from_secs(1), wall, &view);
    assert_eq!(
        second.snapshot.activity,
        Some(50.0),
        "wrapped socket joules should be 50% activity, gpu util is only 1%"
    );
    assert_eq!(second.snapshot.load, Some(1.0));
    assert_eq!(second.snapshot.gpu_pct, Some(1.0));
    assert_eq!(second.cpu_w, Some(100.0));
    assert_eq!(second.load_source, LoadSource::Cpu);
    assert_eq!(second.load_source.label(), "cpu");
}

fn stat_after(busy_pct: &[u32], steps: u64) -> String {
    let mut text = String::from("cpu  0 0 0 0 0 0 0 0\n");
    for (index, pct) in busy_pct.iter().enumerate() {
        let busy = u64::from(*pct) * steps;
        let idle = 100 + (100 - u64::from(*pct)) * steps;
        text.push_str(&format!("cpu{index} {busy} 0 0 {idle} 0 0 0 0\n"));
    }
    text
}

#[test]
fn neither_power_source_falls_back_to_max_of_gpu_util_and_cpu_mean() {
    // One busy CPU on a 32-thread box: mean 3.125%, top-8 mean 12.5%.
    // GPU util is 10%, NVML power fails and zenergy is absent. The ring
    // must follow max(util, cpu mean) = 10, not the top-8 mean.
    let rig = Rig::new("fallback");
    std::fs::remove_file(rig.hwmon.join("name")).expect("drop zenergy name");
    let proc_dir = rig.scratch.path().join("proc");
    std::fs::create_dir_all(&proc_dir).expect("proc");
    let mut busy = vec![0; 32];
    busy[0] = 100;
    let path = rig.scratch.path().join("watch.toml");
    std::fs::write(
        &path,
        "[collector]\ntick_s = 0.1\ncpu_window_s = 1.0\ncpu_top_k = 8\n\
         [llama]\nurl = \"http://127.0.0.1:9\"\n\
         [load]\nsmooth_s = 0\n",
    )
    .expect("watch.toml");
    let config = Config::load_validated(&path, 32).expect("config");
    let mut gpu = FakeGpu::power(10.0, 300_000, 350_000);
    gpu.power_mw = Err("power sensor");
    let mut collector = WatchCollector::new(rig.roots(), gpu, &config, Quiet);
    let start = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    let view = idle_view();
    let mut last = None;
    for nth in 0..=10 {
        std::fs::write(proc_dir.join("stat"), stat_after(&busy, nth)).expect("stat");
        last = Some(collector.sample(start + Duration::from_millis(100 * nth), wall, &view));
    }
    let sampled = last.expect("sample");
    let snap = &sampled.snapshot;
    assert_eq!(snap.cpu_pct, Some(3.125));
    assert_eq!(snap.cpu_topk_pct, Some(12.5));
    assert_eq!(snap.gpu_pct, Some(10.0));
    assert_eq!(snap.load, Some(12.5));
    assert_eq!(
        snap.activity,
        Some(10.0),
        "fallback must be max(gpu, cpu mean)"
    );
    assert_eq!(sampled.load_source, LoadSource::Util);
    assert_eq!(sampled.cpu_w, None);
    assert_eq!(sampled.load_source.label(), "util");
}

/// NVML whose draw a test turns like a dial.
#[derive(Clone)]
struct DialGpu {
    power_mw: Arc<AtomicU32>,
    limit_mw: u32,
}

impl DialGpu {
    fn new(limit_w: u32) -> Self {
        Self {
            power_mw: Arc::new(AtomicU32::new(0)),
            limit_mw: limit_w * 1000,
        }
    }
}

impl GpuBackend for DialGpu {
    fn init(&mut self) -> Result<(), GpuError> {
        Ok(())
    }

    fn util_pct(&mut self) -> Result<f32, GpuError> {
        Ok(3.0)
    }

    fn temp_c(&mut self) -> Result<f32, GpuError> {
        Ok(40.0)
    }

    fn memory_info(&mut self) -> Result<(u64, u64), GpuError> {
        Ok((1, 2))
    }

    fn power_usage(&mut self) -> Result<u32, GpuError> {
        Ok(self.power_mw.load(Ordering::Relaxed))
    }

    fn enforced_power_limit(&mut self) -> Result<u32, GpuError> {
        Ok(self.limit_mw)
    }

    fn disconnect(&mut self) {}
}

/// Fake clock, NVML dial and zenergy counter for long runs.
struct Drive<'r> {
    rig: &'r Rig,
    dial: Arc<AtomicU32>,
    start: Instant,
    at: Duration,
    socket_uj: u64,
}

impl<'r> Drive<'r> {
    fn new(rig: &'r Rig, gpu: &DialGpu) -> Self {
        rig.set_energy(0, 0, 0);
        Self {
            rig,
            dial: Arc::clone(&gpu.power_mw),
            start: Instant::now(),
            at: Duration::ZERO,
            socket_uj: 0,
        }
    }

    /// Prime the counter so the next step has a socket rate. The GPU draws
    /// `gpu_w`, which the auto floor sees like any other read.
    fn prime<B: GpuBackend>(&mut self, collector: &mut WatchCollector<'_, B, Quiet>, gpu_w: u32) {
        self.dial.store(gpu_w * 1000, Ordering::Relaxed);
        collector.sample(self.start, SystemTime::UNIX_EPOCH, &idle_view());
    }

    /// Hold `gpu_w` and `cpu_w` for one second.
    fn step<B: GpuBackend>(
        &mut self,
        collector: &mut WatchCollector<'_, B, Quiet>,
        gpu_w: u32,
        cpu_w: u64,
    ) -> llama_watch::collector::WatchSample {
        self.dial.store(gpu_w * 1000, Ordering::Relaxed);
        self.socket_uj += cpu_w * 1_000_000;
        self.rig.set_energy(self.socket_uj, 0, 0);
        self.at += Duration::from_secs(1);
        collector.sample(self.start + self.at, SystemTime::UNIX_EPOCH, &idle_view())
    }
}

fn pct(sample: &llama_watch::collector::WatchSample) -> f64 {
    f64::from(sample.snapshot.activity.expect("activity"))
}

#[test]
fn sustained_gpu_bottleneck_holds_for_thirty_minutes() {
    // Observed on a reference box: 340 of 350 W on the GPU read 1–7%. A minute idle at
    // 20 W GPU / 20 W CPU sets both floors to 20 W. Then 30 min of 340 W GPU
    // with the CPU nearly idle at 30 W. The GPU fraction is
    // (340 - 20) / (350 - 20) = 96.97%. Summing reads (370 - 40) / (580 - 40)
    // = 61%, and a rolling floor learns 340 W as idle and decays to 0.
    // Defaults: cpu_limit_w 230, idle auto, smooth_s 0.3.
    let rig = Rig::new("sustained-gpu");
    let config = rig.config("");
    let gpu = DialGpu::new(350);
    let mut drive = Drive::new(&rig, &gpu);
    let mut collector = WatchCollector::new(rig.roots(), gpu.clone(), &config, Quiet);
    drive.prime(&mut collector, 20);
    for _ in 0..60 {
        drive.step(&mut collector, 20, 20);
    }
    let idle = drive.step(&mut collector, 20, 20);
    assert!(pct(&idle) < 0.5, "idle reads ~0, got {}", pct(&idle));

    let expected = (340.0 - 20.0) / (350.0 - 20.0) * 100.0;
    let mut settled = None;
    for second in 0..30 * 60 {
        let sampled = drive.step(&mut collector, 340, 30);
        if second < 5 {
            continue;
        }
        let now = pct(&sampled);
        assert!(
            now >= 95.0,
            "activity {now} at {second} s of sustained 340 W GPU"
        );
        assert_eq!(sampled.load_source, LoadSource::Gpu, "at {second} s");
        let first = *settled.get_or_insert(now);
        assert!(
            now >= first - 0.01,
            "decayed from {first} to {now} at {second} s"
        );
    }
    let last = drive.step(&mut collector, 340, 30);
    assert!(
        (pct(&last) - expected).abs() < 0.05,
        "after 30 min {} expected {expected}",
        pct(&last)
    );
}

#[test]
fn cpu_bottleneck_reads_full_with_source_cpu() {
    // GPU idle at 20 W, CPU at its 230 W limit. Floors learn 20 W each.
    let rig = Rig::new("cpu-bottleneck");
    let config = rig.config("smooth_s = 0\n");
    let gpu = DialGpu::new(350);
    let mut drive = Drive::new(&rig, &gpu);
    let mut collector = WatchCollector::new(rig.roots(), gpu.clone(), &config, Quiet);
    drive.prime(&mut collector, 20);
    for _ in 0..5 {
        drive.step(&mut collector, 20, 20);
    }
    for _ in 0..60 {
        let sampled = drive.step(&mut collector, 20, 230);
        assert!(
            pct(&sampled) >= 99.5,
            "cpu at limit reads {}",
            pct(&sampled)
        );
        assert_eq!(sampled.load_source, LoadSource::Cpu);
        assert_eq!(sampled.cpu_w, Some(230.0));
    }
}

#[test]
fn both_idle_reads_zero() {
    // At the default floors (30 W GPU, 25 W CPU) and below them.
    let rig = Rig::new("both-idle");
    let config = rig.config("smooth_s = 0\n");
    let gpu = DialGpu::new(350);
    let mut drive = Drive::new(&rig, &gpu);
    let mut collector = WatchCollector::new(rig.roots(), gpu.clone(), &config, Quiet);
    drive.prime(&mut collector, 30);
    for (gpu_w, cpu_w) in [(30, 25), (30, 25), (20, 15), (20, 15), (18, 12)] {
        let sampled = drive.step(&mut collector, gpu_w, cpu_w);
        assert!(
            pct(&sampled) < 1.0,
            "{gpu_w} W GPU / {cpu_w} W CPU reads {}",
            pct(&sampled)
        );
        assert_ne!(sampled.load_source, LoadSource::Util);
    }
}

#[test]
fn idle_floor_never_rises_under_sustained_load() {
    // Floors learn 10 W each from 45 s idle (30 s settle, then a 10 s mean),
    // then 30 min of heavy load on both. A GPU probe
    // at 60 W must still read (60 - 10) / (350 - 10), and a CPU probe at
    // 115 W (115 - 10) / (230 - 10). A floor that learned the load reads 0.
    let rig = Rig::new("floor-monotonic");
    let config = rig.config("smooth_s = 0\n");
    let gpu = DialGpu::new(350);
    let mut drive = Drive::new(&rig, &gpu);
    let mut collector = WatchCollector::new(rig.roots(), gpu.clone(), &config, Quiet);
    drive.prime(&mut collector, 10);
    for _ in 0..45 {
        drive.step(&mut collector, 10, 10);
    }
    for _ in 0..30 * 60 {
        drive.step(&mut collector, 340, 200);
    }
    let gpu_probe = drive.step(&mut collector, 60, 10);
    let expected = (60.0 - 10.0) / (350.0 - 10.0) * 100.0;
    assert!(
        (pct(&gpu_probe) - expected).abs() < 0.05,
        "gpu probe {} expected {expected}",
        pct(&gpu_probe)
    );
    assert_eq!(gpu_probe.load_source, LoadSource::Gpu);
    let cpu_probe = drive.step(&mut collector, 10, 115);
    let expected = (115.0 - 10.0) / (230.0 - 10.0) * 100.0;
    assert!(
        (pct(&cpu_probe) - expected).abs() < 0.05,
        "cpu probe {} expected {expected}",
        pct(&cpu_probe)
    );
    assert_eq!(cpu_probe.load_source, LoadSource::Cpu);
}

#[test]
fn a_configured_floor_starts_above_what_was_observed_only_when_fixed() {
    // 20 W GPU is held for 45 s, then a 60 W probe. Auto learns the 20 W
    // mean and reads (60 - 20) / (350 - 20). Fixed at 40 W reads (60 - 40) / (350 - 40).
    for (mode, floor) in [("auto", 20.0), ("fixed", 40.0)] {
        let rig = Rig::new(&format!("floor-{mode}"));
        let config = rig.config(&format!(
            "idle = \"{mode}\"\ngpu_idle_w = 40\ncpu_idle_w = 25\nsmooth_s = 0\n"
        ));
        let gpu = DialGpu::new(350);
        let mut drive = Drive::new(&rig, &gpu);
        let mut collector = WatchCollector::new(rig.roots(), gpu.clone(), &config, Quiet);
        drive.prime(&mut collector, 20);
        for _ in 0..45 {
            drive.step(&mut collector, 20, 20);
        }
        let probe = drive.step(&mut collector, 60, 20);
        let expected = (60.0 - floor) / (350.0 - floor) * 100.0;
        assert!(
            (pct(&probe) - expected).abs() < 0.05,
            "{mode}: {} expected {expected}",
            pct(&probe)
        );
    }
}

/// T56: found live. The first zenergy delta after a restart spans a partial
/// interval and read low; the raw-sample minimum pinned the CPU floor there
/// and a 37 W idle read about 20 %. The GPU idles at its default here.
#[test]
fn a_glitch_low_first_socket_read_does_not_poison_the_cpu_floor() {
    let rig = Rig::new("glitch");
    let config = rig.config(
        "smooth_s = 0
nominal_frac = 0.8
",
    );
    let gpu = DialGpu::new(350);
    let mut drive = Drive::new(&rig, &gpu);
    let mut collector = WatchCollector::new(rig.roots(), gpu.clone(), &config, Quiet);
    drive.prime(&mut collector, 30);
    let glitch = drive.step(&mut collector, 30, 5);
    assert_eq!(glitch.cpu_w, Some(5.0));
    for _ in 0..90 {
        drive.step(&mut collector, 30, 37);
    }
    let idle = drive.step(&mut collector, 30, 37);
    assert!(pct(&idle) < 0.5, "37 W idle CPU reads {}", pct(&idle));
}

/// T56: a Zen package idles near 41 W, above the 25 W default. Auto learns it.
#[test]
fn a_cpu_idle_above_the_default_is_learned_and_reads_zero() {
    let rig = Rig::new("idle-above-default");
    let config = rig.config(
        "smooth_s = 0
cpu_idle_w = 25
nominal_frac = 0.8
",
    );
    let gpu = DialGpu::new(350);
    let mut drive = Drive::new(&rig, &gpu);
    let mut collector = WatchCollector::new(rig.roots(), gpu.clone(), &config, Quiet);
    drive.prime(&mut collector, 30);
    let early = drive.step(&mut collector, 30, 41);
    let expected = (41.0 - 25.0) / (0.8 * 230.0 - 25.0) * 100.0;
    assert!(
        (pct(&early) - expected).abs() < 0.05,
        "before learning the default holds: {} expected {expected}",
        pct(&early)
    );
    for _ in 0..60 {
        drive.step(&mut collector, 30, 41);
    }
    let idle = drive.step(&mut collector, 30, 41);
    assert!(pct(&idle) < 0.5, "41 W idle CPU reads {}", pct(&idle));
    let probe = drive.step(&mut collector, 30, 112);
    let expected = (112.0 - 41.0) / (0.8 * 230.0 - 41.0) * 100.0;
    assert!(
        (pct(&probe) - expected).abs() < 0.05,
        "probe {} expected {expected}",
        pct(&probe)
    );
}

#[test]
fn missing_zenergy_uses_the_gpu_only() {
    // 340 of 350 W with the default 30 W floor: (340 - 30) / (350 - 30).
    let rig = Rig::new("no-zenergy");
    std::fs::remove_file(rig.hwmon.join("name")).expect("drop zenergy name");
    let config = rig.config("smooth_s = 0\n");
    let gpu = DialGpu::new(350);
    let mut drive = Drive::new(&rig, &gpu);
    let mut collector = WatchCollector::new(rig.roots(), gpu.clone(), &config, Quiet);
    drive.prime(&mut collector, 340);
    let sampled = drive.step(&mut collector, 340, 0);
    let expected = (340.0 - 30.0) / (350.0 - 30.0) * 100.0;
    assert!(
        (pct(&sampled) - expected).abs() < 0.05,
        "{} expected {expected}",
        pct(&sampled)
    );
    assert_eq!(sampled.load_source, LoadSource::Gpu);
    assert_eq!(sampled.cpu_w, None);
}

#[test]
fn missing_nvml_power_uses_the_cpu_only() {
    // 130 W on the socket with the default 25 W floor and 230 W limit.
    // Either NVML call failing takes the GPU out; GPU util does not count.
    for (power, limit) in [
        (Err("power sensor"), Ok(350_000)),
        (Ok(340_000), Err("limit")),
    ] {
        let rig = Rig::new("no-nvml-power");
        let config = rig.config("smooth_s = 0\n");
        let gpu = FakeGpu {
            util: 90.0,
            power_mw: power,
            limit_mw: limit,
        };
        let mut collector = WatchCollector::new(rig.roots(), gpu, &config, Quiet);
        let start = Instant::now();
        let wall = SystemTime::UNIX_EPOCH;
        let view = idle_view();
        rig.set_energy(0, 0, 0);
        collector.sample(start, wall, &view);
        rig.set_energy(130_000_000, 0, 0);
        let sampled = collector.sample(start + Duration::from_secs(1), wall, &view);
        let expected = (130.0 - 25.0) / (230.0 - 25.0) * 100.0;
        let got = f64::from(sampled.snapshot.activity.expect("activity"));
        assert!((got - expected).abs() < 0.05, "{got} expected {expected}");
        assert_eq!(sampled.snapshot.gpu_pct, Some(90.0));
        assert_eq!(sampled.load_source, LoadSource::Cpu);
        assert_eq!(sampled.cpu_w, Some(130.0));
    }
}

#[test]
fn activity_clamps_above_one_and_below_zero() {
    let start = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    let view = idle_view();

    let high = Rig::new("clamp-high");
    let config = high.config(ZERO_FLOORS);
    let mut collector = WatchCollector::new(
        high.roots(),
        FakeGpu::power(90.0, 150_000, 100_000),
        &config,
        Quiet,
    );
    high.set_energy(0, 0, 0);
    collector.sample(start, wall, &view);
    high.set_energy(150_000_000, 0, 0);
    let hot = collector.sample(start + Duration::from_secs(1), wall, &view);
    assert_eq!(
        hot.snapshot.activity,
        Some(125.0),
        "1.5 of nominal clamps to the 125% peg"
    );
    assert_eq!(hot.snapshot.gpu_pct, Some(90.0));

    let low = Rig::new("clamp-low");
    let config = low.config(
        "cpu_limit_w = 100\nidle = \"fixed\"\ngpu_idle_w = 40\ncpu_idle_w = 40\nsmooth_s = 0\n",
    );
    let mut collector = WatchCollector::new(
        low.roots(),
        FakeGpu::power(90.0, 0, 100_000),
        &config,
        Quiet,
    );
    low.set_energy(0, 0, 0);
    collector.sample(start, wall, &view);
    low.set_energy(0, 0, 0);
    let cold = collector.sample(start + Duration::from_secs(1), wall, &view);
    assert_eq!(
        cold.snapshot.activity,
        Some(0.0),
        "power below the floor clamps to 0"
    );
}

/// T54: each device's fraction is taken on idle→(nominal_frac × limit), so
/// sustained heavy load reads about 100 and a spike reads up to 125.
#[test]
fn nominal_frac_puts_sustained_load_at_100_and_spikes_above() {
    let start = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    let view = idle_view();
    // GPU limit 350 W, floor 30 W fixed, nominal 0.8: headroom is 280 - 30.
    let load =
        "idle = \"fixed\"\ngpu_idle_w = 30\ncpu_idle_w = 25\nsmooth_s = 0\nnominal_frac = 0.8\n";
    for (gpu_w, expected) in [
        (30_u32, 0.0_f64),
        (155, 50.0),
        (280, 100.0),
        (310, 112.0),
        (342, 124.8),
        (350, 125.0),
    ] {
        let rig = Rig::new(&format!("nominal-{gpu_w}"));
        let config = rig.config(load);
        let mut collector = WatchCollector::new(
            rig.roots(),
            FakeGpu::power(50.0, gpu_w * 1000, 350_000),
            &config,
            Quiet,
        );
        rig.set_energy(0, 0, 0);
        collector.sample(start, wall, &view);
        rig.set_energy(25_000_000, 0, 0);
        let sampled = collector.sample(start + Duration::from_secs(1), wall, &view);
        let got = f64::from(sampled.snapshot.activity.expect("activity"));
        assert!(
            (got - expected).abs() < 0.05,
            "{gpu_w} W reads {got}, expected {expected}"
        );
        assert_eq!(sampled.load_source, LoadSource::Gpu, "{gpu_w} W");
    }
}

/// The default `nominal_frac` is 0.8: 340 of 350 W over a 20 W floor is
/// (340 - 20) / (280 - 20) = 123 %, not the 97 % of the full limit.
#[test]
fn default_nominal_frac_reads_a_near_limit_draw_in_the_redline() {
    let start = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    let view = idle_view();
    let rig = Rig::new("nominal-default");
    let config =
        rig.config("idle = \"fixed\"\ngpu_idle_w = 20\nsmooth_s = 0\nnominal_frac = 0.8\n");
    assert_eq!(config.load.nominal_frac, 0.8);
    let mut collector = WatchCollector::new(
        rig.roots(),
        FakeGpu::power(97.0, 340_000, 350_000),
        &config,
        Quiet,
    );
    rig.set_energy(0, 0, 0);
    collector.sample(start, wall, &view);
    rig.set_energy(25_000_000, 0, 0);
    let sampled = collector.sample(start + Duration::from_secs(1), wall, &view);
    let got = f64::from(sampled.snapshot.activity.expect("activity"));
    let expected = (340.0 - 20.0) / (280.0 - 20.0) * 100.0;
    assert!((got - expected).abs() < 0.05, "{got} expected {expected}");
    assert!(got > 100.0);
}

#[test]
fn ema_tracks_a_step_at_the_configured_time_constant() {
    // 0% then 100% held. τ = 0.3 s, Δt = 1 s.
    // α = 1 - exp(-Δt/τ). The second step must keep the previous EMA.
    let start = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    let view = idle_view();
    let at = |secs| start + Duration::from_secs(secs);
    let tau = 0.3_f64;
    let dt = 1.0_f64;
    let alpha = 1.0 - (-dt / tau).exp();
    let ema1 = alpha * 100.0;
    let ema2 = alpha * 100.0 + (1.0 - alpha) * ema1;

    let rig = Rig::new("ema-step");
    let config = rig.config(
        "cpu_limit_w = 100\nidle = \"fixed\"\ngpu_idle_w = 0\ncpu_idle_w = 0\nsmooth_s = 0.3\n",
    );
    let mut collector = WatchCollector::new(
        rig.roots(),
        StepGpu {
            draws: std::cell::Cell::new(0),
        },
        &config,
        Quiet,
    );
    rig.set_energy(0, 0, 0);
    collector.sample(start, wall, &view);
    rig.set_energy(0, 0, 0);
    let quiet = collector.sample(at(1), wall, &view);
    assert_eq!(
        quiet.snapshot.activity,
        Some(0.0),
        "baseline before the step"
    );
    rig.set_energy(100_000_000, 0, 0);
    let first = collector.sample(at(2), wall, &view);
    let first_load = f64::from(first.snapshot.activity.expect("first step"));
    assert!(
        (first_load - ema1).abs() < 1e-3,
        "first step {first_load} expected {ema1}"
    );
    rig.set_energy(200_000_000, 0, 0);
    let second = collector.sample(at(3), wall, &view);
    let second_load = f64::from(second.snapshot.activity.expect("second step"));
    assert!(
        (second_load - ema2).abs() < 1e-3,
        "second step {second_load} expected {ema2}"
    );
}

/// GPU draw stays 0 W for two reads, then 100 W. Limit is 100 W throughout.
struct StepGpu {
    draws: std::cell::Cell<u32>,
}

impl GpuBackend for StepGpu {
    fn init(&mut self) -> Result<(), GpuError> {
        Ok(())
    }

    fn util_pct(&mut self) -> Result<f32, GpuError> {
        Ok(0.0)
    }

    fn temp_c(&mut self) -> Result<f32, GpuError> {
        Ok(40.0)
    }

    fn power_usage(&mut self) -> Result<u32, GpuError> {
        let n = self.draws.get();
        self.draws.set(n + 1);
        // Reads 0 and 1 are the baseline. Later reads are the 100 W step.
        Ok(if n < 2 { 0 } else { 100_000 })
    }

    fn enforced_power_limit(&mut self) -> Result<u32, GpuError> {
        Ok(100_000)
    }

    fn disconnect(&mut self) {}
}
