//! `/proc/stat` and `/proc/meminfo`, rooted at [`Roots`](crate::sources::Roots).
//!
//! Files are read with [`std::fs::read_to_string`]. This module has no write API.

use std::path::Path;

use crate::sources::{Roots, SourceError, SourceId};

/// Cumulative tick counters for one logical CPU (`cpuN` in `/proc/stat`).
///
/// `total` is user + nice + system + idle + iowait + irq + softirq + steal.
/// Guest and guest_nice are not included: the kernel already counts them
/// inside user and nice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuCounters {
    pub index: u32,
    pub total: u64,
    pub idle: u64,
    pub iowait: u64,
}

/// One sample of every logical CPU.
///
/// Pass this as `prev` on the next [`read_cpu`] call.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuSample {
    pub counters: Vec<CpuCounters>,
    /// Per-CPU busy percent, ordered by CPU index (the same order as
    /// [`Self::counters`]). `None` on the first sample and when the set of
    /// online CPUs changed, because there is no paired delta.
    pub busy_pct: Option<Vec<f32>>,
}

/// Read `{proc}/stat` lines `cpuN`.
///
/// Busy percent is `100 * Δ(total − idle − iowait) / Δtotal`, clamped to
/// 0..=100. Each field delta uses saturating subtraction, because a per-CPU
/// counter (iowait in particular) can decrease. The aggregate `cpu` line is
/// ignored.
///
/// The first call (`prev == None`), and any call whose online CPU set differs
/// from `prev`, returns [`CpuSample::busy_pct`] = `None` together with the
/// counters just read. The next call can delta against those counters.
pub fn read_cpu(roots: &Roots, prev: Option<&CpuSample>) -> Result<CpuSample, SourceError> {
    let counters = parse_stat(&read_text(SourceId::ProcCpu, &roots.proc.join("stat"))?)?;
    let busy_pct = prev.and_then(|prev| busy_deltas(&prev.counters, &counters));
    Ok(CpuSample { counters, busy_pct })
}

/// Host logical CPUs: the number of `cpuN` lines in `{proc}/stat`.
///
/// The aggregate `cpu` line is not a CPU. This is the online set the kernel
/// publishes, so a process `CPUAffinity` mask does not change the result.
pub fn count_cpus(proc_root: &Path) -> Result<u32, SourceError> {
    let text = read_text(SourceId::ProcCpu, &proc_root.join("stat"))?;
    let n = parse_stat(&text)?.len();
    Ok(u32::try_from(n).unwrap_or(u32::MAX))
}

/// Arithmetic mean of per-CPU busy percents.
///
/// This is the display CPU percent: every logical CPU is weighted equally.
/// `None` when `busy_pct` is empty.
pub fn plain_mean(busy_pct: &[f32]) -> Option<f32> {
    if busy_pct.is_empty() {
        return None;
    }
    let sum: f64 = busy_pct.iter().map(|value| f64::from(*value)).sum();
    Some((sum / busy_pct.len() as f64) as f32)
}

/// Mean of the `k` highest per-CPU busy percents.
///
/// Used as the CPU term of composite load. `None` when `k` is 0 or
/// `busy_pct` is empty. When `k` is greater than the number of CPUs, every
/// CPU is included.
pub fn busiest_mean(busy_pct: &[f32], k: usize) -> Option<f32> {
    if busy_pct.is_empty() || k == 0 {
        return None;
    }
    let mut ranked = busy_pct.to_vec();
    ranked.sort_by(|left, right| right.total_cmp(left));
    let take = k.min(ranked.len());
    plain_mean(&ranked[..take])
}

/// Memory used percent from `{proc}/meminfo`: `100 * (1 - MemAvailable / MemTotal)`.
///
/// The result is not clamped. A zero `MemTotal` is an error.
pub fn read_mem(roots: &Roots) -> Result<f32, SourceError> {
    let text = read_text(SourceId::ProcMem, &roots.proc.join("meminfo"))?;
    let mut total = None;
    let mut available = None;
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        if key != "MemTotal" && key != "MemAvailable" {
            continue;
        }
        let value = parse_mem_value(key, rest)?;
        if key == "MemTotal" {
            total = Some(value);
        } else {
            available = Some(value);
        }
    }
    let total = total.ok_or_else(|| SourceError::new(SourceId::ProcMem, "missing MemTotal"))?;
    let available =
        available.ok_or_else(|| SourceError::new(SourceId::ProcMem, "missing MemAvailable"))?;
    if total == 0 {
        return Err(SourceError::new(SourceId::ProcMem, "MemTotal is zero"));
    }
    let pct = 100.0 * (1.0 - available as f64 / total as f64);
    Ok(pct as f32)
}

fn parse_mem_value(key: &str, rest: &str) -> Result<u64, SourceError> {
    let Some(token) = rest.split_whitespace().next() else {
        return Err(SourceError::new(
            SourceId::ProcMem,
            format!("bad {key} value"),
        ));
    };
    token
        .parse()
        .map_err(|_| SourceError::new(SourceId::ProcMem, format!("bad {key} value")))
}

fn busy_deltas(prev: &[CpuCounters], next: &[CpuCounters]) -> Option<Vec<f32>> {
    if prev.len() != next.len()
        || prev
            .iter()
            .zip(next)
            .any(|(old, new)| old.index != new.index)
    {
        return None;
    }
    Some(
        prev.iter()
            .zip(next)
            .map(|(old, new)| one_busy(old, new))
            .collect(),
    )
}

fn one_busy(prev: &CpuCounters, next: &CpuCounters) -> f32 {
    let delta_total = next.total.saturating_sub(prev.total);
    let delta_idle = next.idle.saturating_sub(prev.idle);
    let delta_iowait = next.iowait.saturating_sub(prev.iowait);
    if delta_total == 0 {
        return 0.0;
    }
    let busy_ticks = i128::from(delta_total) - i128::from(delta_idle) - i128::from(delta_iowait);
    let ratio = (busy_ticks as f64 * 100.0 / delta_total as f64) as f32;
    ratio.clamp(0.0, 100.0)
}

fn read_text(id: SourceId, path: &Path) -> Result<String, SourceError> {
    std::fs::read_to_string(path)
        .map_err(|err| SourceError::new(id, format!("read {}: {err}", path.display())))
}

fn parse_stat(text: &str) -> Result<Vec<CpuCounters>, SourceError> {
    let mut counters = Vec::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let Some(label) = parts.next() else {
            continue;
        };
        let Some(index) = cpu_index(label) else {
            continue;
        };
        let fields: Vec<&str> = parts.collect();
        if fields.len() < 4 {
            return Err(SourceError::new(
                SourceId::ProcCpu,
                format!("{label} has {} fields, need at least 4", fields.len()),
            ));
        }
        let mut nums = [0u64; 8];
        for (nth, field) in fields.iter().take(8).enumerate() {
            nums[nth] = field.parse::<u64>().map_err(|_| {
                SourceError::new(
                    SourceId::ProcCpu,
                    format!("{label} field {} is not a counter", nth + 1),
                )
            })?;
        }
        let idle = nums[3];
        let iowait = nums[4];
        let total = sum_counters(label, &nums)?;
        counters.push(CpuCounters {
            index,
            total,
            idle,
            iowait,
        });
    }
    if counters.is_empty() {
        return Err(SourceError::new(SourceId::ProcCpu, "no cpuN lines in stat"));
    }
    counters.sort_by_key(|cpu| cpu.index);
    for pair in counters.windows(2) {
        if pair[0].index == pair[1].index {
            return Err(SourceError::new(
                SourceId::ProcCpu,
                format!("duplicate cpu{}", pair[0].index),
            ));
        }
    }
    Ok(counters)
}

fn cpu_index(label: &str) -> Option<u32> {
    let rest = label.strip_prefix("cpu")?;
    if rest.is_empty() || !rest.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

fn sum_counters(label: &str, nums: &[u64; 8]) -> Result<u64, SourceError> {
    nums.iter()
        .try_fold(0u64, |acc, value| acc.checked_add(*value))
        .ok_or_else(|| SourceError::new(SourceId::ProcCpu, format!("{label} counters overflow")))
}

#[cfg(test)]
mod tests {
    use super::{busiest_mean, plain_mean, read_cpu, read_mem};
    use crate::sources::{Roots, SourceId};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "kraken-lcd-proc-{label}-{}-{n}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).expect("scratch dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn roots(&self, stat: &str) -> Roots {
            std::fs::write(self.path().join("stat"), stat).expect("write stat");
            Roots {
                proc: self.path().to_path_buf(),
                sys: self.path().to_path_buf(),
            }
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn assert_proc_cpu(err: &crate::sources::SourceError) {
        assert_eq!(err.id, SourceId::ProcCpu);
        assert!(!err.message.is_empty());
    }

    #[test]
    fn busiest_mean_uses_the_highest_values_not_file_order() {
        // The saturated CPUs are not a prefix, so taking the first k is wrong.
        let busy = [0.0, 10.0, 100.0, 0.0, 50.0];
        assert_eq!(busiest_mean(&busy, 2), Some(75.0));
    }

    #[test]
    fn means_are_none_without_cpus_or_without_a_selection() {
        assert_eq!(plain_mean(&[]), None);
        assert_eq!(busiest_mean(&[], 8), None);
        assert_eq!(busiest_mean(&[10.0, 20.0], 0), None);
    }

    #[test]
    fn busiest_mean_uses_every_cpu_when_k_exceeds_the_count() {
        assert_eq!(busiest_mean(&[10.0, 30.0, 20.0], 8), Some(20.0));
    }

    #[test]
    fn zero_delta_is_zero_percent() {
        let scratch = Scratch::new("zero");
        let stat = "cpu0 10 0 0 10 0 0 0 0\n";
        let roots = scratch.roots(stat);
        let first = read_cpu(&roots, None).expect("first");
        let second = read_cpu(&roots, Some(&first)).expect("second");
        assert_eq!(second.busy_pct, Some(vec![0.0]));
    }

    #[test]
    fn iowait_and_guest_are_outside_busy() {
        let scratch = Scratch::new("iowait-guest");
        let roots = scratch.roots("cpu0 0 0 0 0 0 0 0 0 0 0\n");
        let first = read_cpu(&roots, None).expect("first");
        // 50 idle + 50 iowait, and a guest column that must not enlarge total.
        std::fs::write(scratch.path().join("stat"), "cpu0 0 0 0 50 50 0 0 0 40 0\n")
            .expect("rewrite stat");
        let second = read_cpu(&roots, Some(&first)).expect("second");
        assert_eq!(second.busy_pct, Some(vec![0.0]));
        assert_eq!(second.counters[0].total, 100);
        assert_eq!(second.counters[0].iowait, 50);
    }

    #[test]
    fn iowait_decrease_keeps_the_new_counters() {
        let scratch = Scratch::new("iowait-down");
        // user 100 + idle 100 + iowait 100.
        let roots = scratch.roots("cpu0 100 0 0 100 100 0 0 0\n");
        let first = read_cpu(&roots, None).expect("first");
        // iowait fell 100 → 50 while user rose 100 → 200. Net total 300 → 350.
        // A signed iowait delta would push the ratio above 100; the result stays in range.
        std::fs::write(scratch.path().join("stat"), "cpu0 200 0 0 100 50 0 0 0\n")
            .expect("rewrite stat");
        let second = read_cpu(&roots, Some(&first)).expect("iowait decreased");
        assert_eq!(second.counters[0].iowait, 50);
        assert_eq!(second.counters[0].total, 350);
        assert_eq!(second.busy_pct, Some(vec![100.0]));

        // Idle grew by more than total, which would make the raw ratio negative.
        std::fs::write(scratch.path().join("stat"), "cpu0 150 0 0 250 50 0 0 0\n")
            .expect("rewrite stat");
        let clamped = read_cpu(&roots, Some(&second)).expect("clamped");
        assert_eq!(clamped.counters[0].idle, 250);
        assert_eq!(clamped.busy_pct, Some(vec![0.0]));
    }

    #[test]
    fn cpu_appearing_or_disappearing_starts_a_new_baseline() {
        let scratch = Scratch::new("hotplug");
        let roots = scratch.roots("cpu0 10 0 0 10 0 0 0 0\ncpu1 10 0 0 10 0 0 0 0\n");
        let first = read_cpu(&roots, None).expect("first");

        // cpu1 went offline. Keep this sample's counters and skip the percent.
        std::fs::write(scratch.path().join("stat"), "cpu0 110 0 0 10 0 0 0 0\n")
            .expect("drop cpu1");
        let disappeared = read_cpu(&roots, Some(&first)).expect("disappeared");
        assert!(disappeared.busy_pct.is_none());
        assert_eq!(disappeared.counters.len(), 1);
        assert_eq!(disappeared.counters[0].index, 0);
        // user 110 + idle 10.
        assert_eq!(disappeared.counters[0].total, 120);

        // cpu1 came back. Again no percent, and the new sample is what we keep.
        std::fs::write(
            scratch.path().join("stat"),
            "cpu0 210 0 0 10 0 0 0 0\ncpu1 20 0 0 10 0 0 0 0\n",
        )
        .expect("cpu1 online");
        let appeared = read_cpu(&roots, Some(&disappeared)).expect("appeared");
        assert!(appeared.busy_pct.is_none());
        assert_eq!(
            appeared
                .counters
                .iter()
                .map(|cpu| (cpu.index, cpu.total))
                .collect::<Vec<_>>(),
            vec![(0, 220), (1, 30)]
        );

        // The following tick has a stable set, so busy percent resumes.
        std::fs::write(
            scratch.path().join("stat"),
            "cpu0 310 0 0 10 0 0 0 0\ncpu1 20 0 0 10 0 0 0 0\n",
        )
        .expect("stable set");
        let resumed = read_cpu(&roots, Some(&appeared)).expect("resumed");
        assert_eq!(resumed.busy_pct, Some(vec![100.0, 0.0]));
    }

    #[test]
    fn missing_or_empty_stat_is_a_proc_cpu_error() {
        let scratch = Scratch::new("missing-stat");
        let roots = Roots {
            proc: scratch.path().to_path_buf(),
            sys: scratch.path().to_path_buf(),
        };
        let err = read_cpu(&roots, None).expect_err("missing file");
        assert_proc_cpu(&err);

        let roots = scratch.roots("intr 1\nctxt 2\n");
        let err = read_cpu(&roots, None).expect_err("no cpuN");
        assert_proc_cpu(&err);
        assert!(err.message.contains("no cpuN"), "{}", err.message);
    }

    #[test]
    fn cpu_lines_match_by_index_when_the_file_is_unsorted() {
        let scratch = Scratch::new("order");
        let roots = scratch.roots("cpu1 0 0 0 0 0 0 0 0\ncpu0 0 0 0 0 0 0 0 0\n");
        let first = read_cpu(&roots, None).expect("first");
        assert_eq!(first.counters[0].index, 0);
        assert_eq!(first.counters[1].index, 1);
        std::fs::write(
            scratch.path().join("stat"),
            "cpu1 0 0 0 100 0 0 0 0\ncpu0 100 0 0 0 0 0 0 0\n",
        )
        .expect("rewrite stat");
        let second = read_cpu(&roots, Some(&first)).expect("second");
        assert_eq!(second.busy_pct, Some(vec![100.0, 0.0]));
    }

    fn roots_mem(scratch: &Scratch, meminfo: &str) -> Roots {
        std::fs::write(scratch.path().join("meminfo"), meminfo).expect("write meminfo");
        Roots {
            proc: scratch.path().to_path_buf(),
            sys: scratch.path().to_path_buf(),
        }
    }

    #[test]
    fn memory_percent_follows_the_available_ratio() {
        let scratch = Scratch::new("mem-order");
        // MemAvailable before MemTotal, and a non-round ratio: 100 * (1 - 1/3).
        let roots = roots_mem(
            &scratch,
            "MemAvailable: 1000 kB\nCached: 5 kB\nMemTotal: 3000 kB\n",
        );
        let pct = read_mem(&roots).expect("mem");
        let expected = (100.0 * (1.0 - 1000.0 / 3000.0)) as f32;
        assert!((pct - expected).abs() < 1e-4, "{pct} != {expected}");
    }

    #[test]
    fn memory_percent_is_not_clamped() {
        let scratch = Scratch::new("mem-over");
        let roots = roots_mem(&scratch, "MemTotal: 1000 kB\nMemAvailable: 1500 kB\n");
        assert_eq!(read_mem(&roots).expect("over"), -50.0);
    }

    #[test]
    fn memory_errors_are_proc_mem() {
        let scratch = Scratch::new("mem-err");
        let cases = [
            ("", "MemTotal"),
            ("MemAvailable: 10 kB\n", "MemTotal"),
            ("MemTotal: 10 kB\n", "MemAvailable"),
            ("MemTotal: 0 kB\nMemAvailable: 0 kB\n", "zero"),
            ("MemTotal: no kB\nMemAvailable: 1 kB\n", "bad"),
        ];
        for (body, needle) in cases {
            let roots = roots_mem(&scratch, body);
            let err = read_mem(&roots).expect_err(needle);
            assert_eq!(err.id, SourceId::ProcMem, "{needle}");
            assert!(
                err.message
                    .to_ascii_lowercase()
                    .contains(&needle.to_ascii_lowercase()),
                "{needle}: {}",
                err.message
            );
        }
        let roots = Roots {
            proc: scratch.path().join("absent"),
            sys: scratch.path().to_path_buf(),
        };
        let err = read_mem(&roots).expect_err("missing file");
        assert_eq!(err.id, SourceId::ProcMem);
    }
}
