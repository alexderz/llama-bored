//! Host sources on fake `/proc` and `/sys` roots.
//!
//! Fixtures live under `fixtures/`. Nothing here opens the real
//! `/proc`, `/sys`, or a device node.

use llama_watch::sources::hwmon;
use llama_watch::sources::proc;
use llama_watch::sources::{Roots, SourceId};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn cpu_roots(tick: &str) -> Roots {
    Roots {
        proc: fixtures().join(format!("proc/cpu/{tick}")),
        sys: fixtures().join("sys"),
    }
}

#[test]
fn first_cpu_sample_has_no_busy_percent() {
    let sample = proc::read_cpu(&cpu_roots("tick0"), None).expect("tick0 stat");
    assert!(sample.busy_pct.is_none());
    assert_eq!(sample.counters.len(), 32);
    for (index, cpu) in sample.counters.iter().enumerate() {
        assert_eq!(cpu.index, index as u32, "cpu{index}");
        assert_eq!(cpu.total, 200, "cpu{index} total");
        assert_eq!(cpu.idle, 100, "cpu{index} idle");
        assert_eq!(cpu.iowait, 0, "cpu{index} iowait");
    }
}

#[test]
fn cpu_busy_is_delta_of_total_minus_idle_and_iowait() {
    let first = proc::read_cpu(&cpu_roots("tick0"), None).expect("tick0");
    let second = proc::read_cpu(&cpu_roots("tick1"), Some(&first)).expect("tick1");
    let busy = second.busy_pct.expect("second sample has a delta");
    assert_eq!(busy.len(), 32);
    for (index, pct) in busy.iter().enumerate() {
        let expected = if index < 8 { 100.0 } else { 0.0 };
        assert_eq!(*pct, expected, "cpu{index}");
    }
}

#[test]
fn plain_mean_and_top8_mean_of_the_cpu_fixture() {
    let first = proc::read_cpu(&cpu_roots("tick0"), None).expect("tick0");
    let second = proc::read_cpu(&cpu_roots("tick1"), Some(&first)).expect("tick1");
    let busy = second.busy_pct.expect("delta");
    // Eight saturated CPUs out of 32: plain mean 25, mean of the 8 busiest 100.
    assert_eq!(proc::plain_mean(&busy), Some(25.0));
    assert_eq!(proc::busiest_mean(&busy, 8), Some(100.0));
}

#[test]
fn memory_percent_is_one_minus_available_over_total() {
    let roots = Roots {
        proc: fixtures().join("proc/mem"),
        sys: fixtures().join("sys"),
    };
    // 100 * (1 - 8_000_000 / 32_000_000) = 75. Other meminfo keys are ignored.
    let pct = proc::read_mem(&roots).expect("meminfo");
    assert_eq!(pct, 75.0);
}

fn sys_roots() -> Roots {
    Roots {
        proc: fixtures().join("proc/mem"),
        sys: fixtures().join("sys"),
    }
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "kraken-lcd-hwmon-{label}-{}-{n}",
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

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("mkdir");
    for entry in std::fs::read_dir(src).expect("read_dir") {
        let entry = entry.expect("dir entry");
        let to = dst.join(entry.file_name());
        if entry.file_type().expect("file type").is_dir() {
            copy_dir(&entry.path(), &to);
        } else {
            std::fs::copy(entry.path(), &to).expect("copy");
        }
    }
}

fn copied_sys() -> (Scratch, Roots) {
    let scratch = Scratch::new("sys");
    let sys = scratch.path().join("sys");
    copy_dir(&fixtures().join("sys"), &sys);
    let roots = Roots {
        proc: scratch.path().to_path_buf(),
        sys,
    };
    (scratch, roots)
}

#[test]
fn coolant_is_z53_temp1_in_celsius() {
    // hwmon1 is nvme at 45°C. Coolant is z53 temp1, labeled "Coolant temp".
    assert_eq!(hwmon::read_coolant(&sys_roots()).expect("coolant"), 31.25);
}

#[test]
fn cpu_temp_is_k10temp_tctl_in_celsius() {
    // Tctl is temp1 at 77.5°C. Tccd1 is temp3 and Tccd2 is temp4.
    assert_eq!(hwmon::read_cpu_temp(&sys_roots()).expect("tctl"), 77.5);
}

#[test]
fn missing_tctl_label_is_a_hwmon_cpu_error() {
    let (_scratch, roots) = copied_sys();
    std::fs::remove_file(roots.sys.join("class/hwmon/hwmon3/temp1_label")).expect("drop label");
    let err = hwmon::read_cpu_temp(&roots).expect_err("missing Tctl");
    assert_eq!(err.id, SourceId::HwmonCpu);
    assert!(err.message.contains("Tctl"), "{}", err.message);
}

#[test]
fn renumbered_z53_is_found_on_the_next_read() {
    let (_scratch, roots) = copied_sys();
    assert_eq!(hwmon::read_coolant(&roots).expect("hwmon5"), 31.25);
    let from = roots.sys.join("class/hwmon/hwmon5");
    let to = roots.sys.join("class/hwmon/hwmon6");
    std::fs::rename(&from, &to).expect("renumber hwmon5 -> hwmon6");
    assert_eq!(hwmon::read_coolant(&roots).expect("hwmon6"), 31.25);
}

#[test]
fn proc_and_hwmon_production_code_only_reads() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let banned = [
        "std::fs::write",
        "File::create",
        "OpenOptions",
        "write_all",
        "remove_file",
        "remove_dir",
        "std::fs::copy",
        "std::fs::rename",
        "create_dir",
        "set_len",
        "hard_link",
    ];
    for rel in ["src/sources/proc.rs", "src/sources/hwmon.rs"] {
        let text = std::fs::read_to_string(root.join(rel)).expect(rel);
        let production = text.split("#[cfg(test)]").next().expect("source");
        assert!(
            production.contains("std::fs::read_to_string"),
            "{rel} does not read with read_to_string"
        );
        for needle in banned {
            assert!(!production.contains(needle), "{rel} contains {needle}");
        }
    }
}
