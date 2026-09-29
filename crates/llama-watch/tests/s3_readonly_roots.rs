//! S3 collect half: a read-only fake proc and sys root still completes a collect.
//!
//! `chmod` runs only on a directory under `CARGO_TARGET_TMPDIR`. Nothing here
//! opens a device node or the real `/proc` or `/sys`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime};

use llama_core::log::Sink;
use llama_core::sample::{AiState, LlamaView};
use llama_watch::collector::WatchCollector;
use llama_watch::config::{Config, ValidWatchConfig};
use llama_watch::sources::Roots;
use llama_watch::sources::gpu::{GpuBackend, GpuError};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "s3-readonly-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).expect("scratch dir");
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = Command::new("chmod")
            .arg("-R")
            .arg("u+w")
            .arg(&self.0)
            .status();
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct QuietLog;

impl Sink for QuietLog {
    fn write_line(&mut self, _line: &str) {}
}

struct SteadyGpu;

impl GpuBackend for SteadyGpu {
    fn init(&mut self) -> Result<(), GpuError> {
        Ok(())
    }

    fn util_pct(&mut self) -> Result<f32, GpuError> {
        Ok(40.0)
    }

    fn temp_c(&mut self) -> Result<f32, GpuError> {
        Ok(55.0)
    }

    fn disconnect(&mut self) {}
}

fn copy_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap_or_else(|err| panic!("mkdir {}: {err}", dst.display()));
    for entry in
        std::fs::read_dir(src).unwrap_or_else(|err| panic!("read {}: {err}", src.display()))
    {
        let entry = entry.unwrap_or_else(|err| panic!("read {}: {err}", src.display()));
        let to = dst.join(entry.file_name());
        let kind = entry
            .file_type()
            .unwrap_or_else(|err| panic!("type {}: {err}", entry.path().display()));
        if kind.is_dir() {
            copy_tree(&entry.path(), &to);
        } else if kind.is_file() {
            std::fs::copy(entry.path(), &to)
                .unwrap_or_else(|err| panic!("copy {}: {err}", entry.path().display()));
        } else {
            panic!("fixture is not a regular file: {}", entry.path().display());
        }
    }
}

fn chmod_readonly(path: &Path) {
    let status = Command::new("chmod")
        .arg("-R")
        .arg("a-w")
        .arg(path)
        .status()
        .unwrap_or_else(|err| panic!("chmod: {err}"));
    assert!(status.success(), "chmod -R a-w {} failed", path.display());
}

fn load_config(dir: &Path) -> ValidWatchConfig {
    let path = dir.join("watch.toml");
    std::fs::write(
        &path,
        "[collector]\ntick_s = 0.1\ncpu_window_s = 1.0\ncpu_top_k = 8\n\
         [llama]\nurl = \"http://127.0.0.1:9\"\n",
    )
    .expect("write config");
    Config::load_validated(&path, 32).expect("valid config")
}

fn euid_is_root() -> bool {
    std::process::Command::new("id")
        .arg("-u")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .is_some_and(|text| text.trim() == "0")
}

#[test]
fn readonly_roots_complete_collect() {
    if euid_is_root() {
        eprintln!(
            "skipping readonly_roots_complete_collect: euid is 0, so chmod a-w does not block writes"
        );
        return;
    }
    let scratch = Scratch::new();
    let proc_root = scratch.0.join("proc");
    let sys_root = scratch.0.join("sys");
    copy_tree(&fixtures().join("proc"), &proc_root);
    copy_tree(&fixtures().join("sys"), &sys_root);
    std::fs::copy(proc_root.join("mem/meminfo"), proc_root.join("meminfo")).expect("meminfo");
    std::fs::copy(proc_root.join("cpu/tick0/stat"), proc_root.join("stat")).expect("stat");
    chmod_readonly(&proc_root);
    chmod_readonly(&sys_root);

    let denied = std::fs::write(proc_root.join("probe"), b"no");
    assert!(
        denied.is_err(),
        "the fake proc root must reject writes after chmod"
    );

    let config = load_config(&scratch.0);
    let mut collector = WatchCollector::new(
        Roots {
            proc: proc_root,
            sys: sys_root,
        },
        SteadyGpu,
        &config,
        QuietLog,
    );
    let view = LlamaView {
        ai: AiState::Idle,
        models: Vec::new(),
        decoded_total: None,
        prompt_total: None,
    };
    let snapshot = collector
        .sample(Instant::now(), SystemTime::UNIX_EPOCH, &view)
        .snapshot;
    assert!(
        snapshot.mem_pct.is_some(),
        "read-only proc meminfo should still be readable"
    );
    assert!(
        snapshot.coolant_c.is_some(),
        "read-only z53 temp should still be readable"
    );
    assert!(
        snapshot.cpu_c.is_some(),
        "read-only k10temp should still be readable"
    );
    assert!(
        snapshot.gpu_c.is_some(),
        "fake gpu should report a temperature"
    );
}
