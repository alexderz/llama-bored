//! Every host source is optional. A host without NVIDIA, zenergy, a CPU
//! temperature sensor, or llama-swap still gets a snapshot, with the missing
//! fields empty and the other sources untouched.
//!
//! Fake `/proc` and `/sys` trees and a fake GPU only. Nothing here opens a
//! device node, loads a GPU library, or dials llama-swap.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use llama_core::sample::{AiState, LlamaView, ModelInfo};
use llama_watch::collector::{GpuExtra, LoadSource, WatchCollector, WatchSample};
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
            "optional-sources-{label}-{}-{n}",
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

/// Fake host: `/proc` with meminfo, `/sys` copied from the fixture tree.
struct Host {
    scratch: Scratch,
}

impl Host {
    fn new(label: &str) -> Self {
        let scratch = Scratch::new(label);
        let proc_dir = scratch.path().join("proc");
        std::fs::create_dir_all(&proc_dir).expect("proc");
        std::fs::copy(
            fixtures().join("proc/mem/meminfo"),
            proc_dir.join("meminfo"),
        )
        .expect("meminfo");
        copy_tree(&fixtures().join("sys"), &scratch.path().join("sys"));
        Self { scratch }
    }

    fn sys(&self) -> PathBuf {
        self.scratch.path().join("sys")
    }

    fn roots(&self) -> Roots {
        Roots {
            proc: self.scratch.path().join("proc"),
            sys: self.sys(),
        }
    }

    fn set_stat(&self, busy_pct: &[u32], steps: u64) {
        let mut text = String::from("cpu  0 0 0 0 0 0 0 0\n");
        for (index, pct) in busy_pct.iter().enumerate() {
            let busy = u64::from(*pct) * steps;
            let idle = 100 + (100 - u64::from(*pct)) * steps;
            text.push_str(&format!("cpu{index} {busy} 0 0 {idle} 0 0 0 0\n"));
        }
        std::fs::write(self.scratch.path().join("proc/stat"), text).expect("stat");
    }

    /// AMD socket energy, as the zenergy driver exposes it.
    fn add_zenergy(&self, socket_uj: u64) {
        let dir = self.sys().join("class/hwmon/hwmon7");
        std::fs::create_dir_all(&dir).expect("zenergy dir");
        std::fs::write(dir.join("name"), "zenergy\n").expect("name");
        std::fs::write(dir.join("energy1_label"), "Esocket0\n").expect("label");
        std::fs::write(dir.join("energy1_input"), format!("{socket_uj}\n")).expect("input");
    }

    fn config(&self, extra: &str) -> ValidWatchConfig {
        let path = self.scratch.path().join("watch.toml");
        let text = format!(
            "[collector]\ntick_s = 0.1\ncpu_window_s = 1.0\ncpu_top_k = 1\n\
             [load]\ncpu_limit_w = 100\nidle = \"fixed\"\ngpu_idle_w = 0\ncpu_idle_w = 0\nsmooth_s = 0\n{extra}"
        );
        std::fs::write(&path, text).expect("watch.toml");
        Config::load_validated(&path, 4).expect("valid watch config")
    }
}

/// No NVIDIA driver: every init fails, as `NvidiaGpu` does when no
/// `libnvidia-ml.so.1` exists.
struct NoNvidia;

impl GpuBackend for NoNvidia {
    fn init(&mut self) -> Result<(), GpuError> {
        Err(GpuError::new("no NVIDIA library loaded"))
    }

    fn util_pct(&mut self) -> Result<f32, GpuError> {
        panic!("util read without a session")
    }

    fn temp_c(&mut self) -> Result<f32, GpuError> {
        panic!("temp read without a session")
    }

    fn disconnect(&mut self) {}
}

struct SteadyGpu;

impl GpuBackend for SteadyGpu {
    fn init(&mut self) -> Result<(), GpuError> {
        Ok(())
    }

    fn util_pct(&mut self) -> Result<f32, GpuError> {
        Ok(10.0)
    }

    fn temp_c(&mut self) -> Result<f32, GpuError> {
        Ok(40.0)
    }

    fn memory_info(&mut self) -> Result<(u64, u64), GpuError> {
        Ok((1, 2))
    }

    fn power_usage(&mut self) -> Result<u32, GpuError> {
        Ok(50_000)
    }

    fn enforced_power_limit(&mut self) -> Result<u32, GpuError> {
        Ok(100_000)
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

fn view(ai: AiState) -> LlamaView {
    LlamaView {
        ai,
        models: Vec::new(),
        decoded_total: None,
    }
}

/// Eleven ticks at 10 Hz so the 1 s CPU window is full.
fn run<B: GpuBackend>(
    host: &Host,
    gpu: B,
    config: &ValidWatchConfig,
    llama: &LlamaView,
    busy: &[u32],
    log: Capture,
) -> WatchSample {
    let mut collector = WatchCollector::new(host.roots(), gpu, config, log);
    let start = Instant::now();
    let wall = SystemTime::UNIX_EPOCH;
    let mut last = None;
    for nth in 0..=10 {
        host.set_stat(busy, nth);
        last = Some(collector.sample(start + Duration::from_millis(100 * nth), wall, llama));
    }
    last.expect("sample")
}

#[test]
fn no_nvidia_leaves_gpu_fields_empty_and_activity_follows_the_cpu() {
    let host = Host::new("no-nvidia");
    host.add_zenergy(0);
    let config = host.config("");
    let sampled = run(
        &host,
        NoNvidia,
        &config,
        &view(AiState::Idle),
        &[40, 0, 0, 0],
        Capture::default(),
    );
    let snap = &sampled.snapshot;
    assert_eq!(snap.gpu_pct, None);
    assert_eq!(snap.gpu_c, None);
    assert_eq!(sampled.gpu, GpuExtra::default());
    assert!(snap.errors.contains(&SourceId::Gpu), "{:?}", snap.errors);
    // CPU mean over four CPUs is 10%. Without GPU watts the ring is the
    // socket's share alone: the counter is still, so 0 W is 0%, not util.
    assert_eq!(snap.cpu_pct, Some(10.0));
    assert_eq!(sampled.cpu_w, Some(0.0));
    assert_eq!(snap.activity, Some(0.0));
    assert_eq!(sampled.load_source, LoadSource::Cpu);
    // The other sources are unaffected.
    assert_eq!(snap.coolant_c, Some(31.25));
    assert_eq!(snap.cpu_c, Some(77.5));
    assert_eq!(snap.mem_pct, Some(75.0));
}

#[test]
fn no_zenergy_with_a_gpu_uses_gpu_watts_without_an_error() {
    let host = Host::new("no-zenergy");
    let config = host.config("");
    let sampled = run(
        &host,
        SteadyGpu,
        &config,
        &view(AiState::Idle),
        &[80, 0, 0, 0],
        Capture::default(),
    );
    let snap = &sampled.snapshot;
    assert_eq!(sampled.cpu_w, None);
    assert_eq!(sampled.load_source, LoadSource::Gpu);
    // 50 of 100 W with a 0 W floor on the default 0.8 nominal ceiling:
    // 50 / 80 = 62.5 %, not max(gpu 10%, cpu mean 20%).
    assert_eq!(snap.activity, Some(62.5));
    assert!(snap.errors.is_empty(), "{:?}", snap.errors);
}

#[test]
fn no_cpu_temperature_driver_leaves_cpu_c_empty_only() {
    let host = Host::new("no-cputemp");
    // The fixture's hwmon3 is k10temp. Remove it; no coretemp either.
    std::fs::remove_dir_all(host.sys().join("class/hwmon/hwmon3")).expect("drop k10temp");
    let config = host.config("");
    let sampled = run(
        &host,
        SteadyGpu,
        &config,
        &view(AiState::Idle),
        &[0, 0, 0, 0],
        Capture::default(),
    );
    let snap = &sampled.snapshot;
    assert_eq!(snap.cpu_c, None);
    assert_eq!(
        snap.errors.iter().copied().collect::<Vec<_>>(),
        vec![SourceId::HwmonCpu]
    );
    assert_eq!(snap.coolant_c, Some(31.25));
    assert_eq!(snap.gpu_c, Some(40.0));
}

#[test]
fn intel_coretemp_replaces_a_missing_k10temp() {
    let host = Host::new("coretemp");
    std::fs::remove_dir_all(host.sys().join("class/hwmon/hwmon3")).expect("drop k10temp");
    let dir = host.sys().join("class/hwmon/hwmon8");
    std::fs::create_dir_all(&dir).expect("coretemp dir");
    std::fs::write(dir.join("name"), "coretemp\n").expect("name");
    std::fs::write(dir.join("temp1_label"), "Package id 0\n").expect("label");
    std::fs::write(dir.join("temp1_input"), "48000\n").expect("input");
    let config = host.config("");
    let sampled = run(
        &host,
        SteadyGpu,
        &config,
        &view(AiState::Idle),
        &[0, 0, 0, 0],
        Capture::default(),
    );
    assert_eq!(sampled.snapshot.cpu_c, Some(48.0));
    assert!(sampled.snapshot.errors.is_empty());
}

#[test]
fn llama_disabled_is_idle_with_no_error_and_no_llama_log() {
    let host = Host::new("no-llama");
    let config = host.config("[llama]\nenabled = false\n");
    let log = Capture::default();
    // With polling off no view ever arrives. Even a stale Loaded view or the
    // loop's initial Down view must not leak into the snapshot.
    for stale in [AiState::Down, AiState::NoData, AiState::Loaded] {
        let mut llama = view(stale);
        llama.models = vec![ModelInfo {
            name: "stale".to_owned(),
            state: "ready".to_owned(),
            full_name: None,
            detail: None,
        }];
        let sampled = run(&host, SteadyGpu, &config, &llama, &[0; 4], log.clone());
        let snap = &sampled.snapshot;
        assert_eq!(snap.ai, AiState::Idle, "{stale:?}");
        assert!(snap.models.is_empty(), "{stale:?}");
        assert!(!snap.errors.contains(&SourceId::Llama), "{:?}", snap.errors);
    }
    let lines = log.lines();
    assert!(
        !lines.iter().any(|line| line.contains("llama")),
        "{lines:?}"
    );
}

#[test]
fn llama_enabled_but_down_is_still_an_error() {
    let host = Host::new("llama-down");
    let config = host.config("");
    let sampled = run(
        &host,
        SteadyGpu,
        &config,
        &view(AiState::Down),
        &[0; 4],
        Capture::default(),
    );
    assert_eq!(sampled.snapshot.ai, AiState::Down);
    assert!(sampled.snapshot.errors.contains(&SourceId::Llama));
}
