//! In-process sample types shared by the watcher and the writer.

use std::collections::BTreeSet;
use std::time::{Instant, SystemTime};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SourceId {
    ProcCpu,
    ProcMem,
    HwmonCoolant,
    HwmonCpu,
    Gpu,
    Llama,
}

impl SourceId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ProcCpu => "proc.cpu",
            Self::ProcMem => "proc.mem",
            Self::HwmonCoolant => "hwmon.coolant",
            Self::HwmonCpu => "hwmon.cpu",
            Self::Gpu => "gpu",
            Self::Llama => "llama",
        }
    }
}

impl std::fmt::Display for SourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{id}: {message}")]
pub struct SourceError {
    pub id: SourceId,
    pub message: String,
}

impl SourceError {
    pub fn new(id: SourceId, message: impl Into<String>) -> Self {
        Self {
            id,
            message: message.into(),
        }
    }
}

/// Display-safe model name plus the upstream state word.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInfo {
    /// Printable ASCII, whitespace collapsed, truncated with `…`.
    pub name: String,
    /// Upstream state (`"ready"`, `"starting"`, …). Empty when the field is absent.
    pub state: String,
    /// Untruncated display name, sanitised to [`crate::detail::MAX_FULL_NAME_CHARS`].
    /// `None` from an older watcher.
    pub full_name: Option<String>,
    /// Tuning detail from the launch command. `None` from an older watcher.
    pub detail: Option<crate::detail::ModelDetail>,
    /// Server kind and its live gauges (T72). `None` from an older watcher.
    pub backend: Option<crate::backend::BackendInfo>,
}

/// llama-swap reachability for one tick.
///
/// [`Self::Down`] carries no reason. The collector logs the reason.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AiState {
    /// Any llama-swap client or parse failure.
    Down,
    /// HTTP 200 and an empty `running` list.
    Idle,
    /// HTTP 200 and one or more models.
    Loaded,
    /// No fresh watcher snapshot. The watcher does not produce this; the writer's reader does.
    NoData,
}

/// Token counter from one accepted wire snapshot.
///
/// `decoded_total` is `None` when the watcher had no measured counter.
/// The watcher leaves [`Snapshot::tokens`] empty; the writer's reader fills it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenReading {
    /// Watcher process id, random at start.
    pub run_id: u64,
    /// Publish sequence within [`Self::run_id`].
    pub seq: u64,
    /// `CLOCK_MONOTONIC` at sample time, in nanoseconds.
    pub t_mono_ns: u64,
    /// Box decoded-token counter, when the watcher measured one.
    pub decoded_total: Option<u64>,
}

/// Poller output the collector samples. TTY-only detail is not on this type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaView {
    /// Reachability for this poll.
    pub ai: AiState,
    /// Empty unless `ai` is [`AiState::Loaded`].
    pub models: Vec<ModelInfo>,
    /// Running decoded-token total, when every ready model was measured.
    pub decoded_total: Option<u64>,
    /// Running prompt-token total, on the same rule as `decoded_total` (#11).
    pub prompt_total: Option<u64>,
}

/// One collector tick. Every `None` means that source failed or is stale.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    /// CLOCK_MONOTONIC
    pub t_mono: Instant,
    /// logs + suspend detection only
    pub t_wall: SystemTime,
    /// composite 0..100: `max(gpu_pct, cpu_topk_pct)`
    pub load: Option<f32>,
    /// Power-weighted activity, 0..=125 (100 is nominal sustained load).
    /// `None` on an older snapshot.
    pub activity: Option<f32>,
    /// plain mean over all CPUs (display)
    pub cpu_pct: Option<f32>,
    /// mean of k busiest CPUs (load term)
    pub cpu_topk_pct: Option<f32>,
    /// NVML utilization.gpu
    pub gpu_pct: Option<f32>,
    /// 100 * (1 - MemAvailable/MemTotal)
    pub mem_pct: Option<f32>,
    /// z53 temp1_input / 1000
    pub coolant_c: Option<f32>,
    /// k10temp Tctl / 1000
    pub cpu_c: Option<f32>,
    /// NVML GPU temp
    pub gpu_c: Option<f32>,
    /// llama-swap state for this tick
    pub ai: AiState,
    /// empty unless ai == Loaded
    pub models: Vec<ModelInfo>,
    /// Token reading from a fresh wire snapshot.
    ///
    /// The watcher sets this to `None`. The writer's reader fills it.
    pub tokens: Option<TokenReading>,
    /// sources that failed this tick
    pub errors: BTreeSet<SourceId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_data_is_distinct_from_down() {
        assert_ne!(AiState::NoData, AiState::Down);
        assert_ne!(AiState::NoData, AiState::Idle);
        assert_ne!(AiState::NoData, AiState::Loaded);
    }

    #[test]
    fn token_reading_and_llama_view_carry_the_shared_fields() {
        let reading = TokenReading {
            run_id: 7,
            seq: 8,
            t_mono_ns: 9,
            decoded_total: Some(10),
        };
        assert_eq!(
            (
                reading.run_id,
                reading.seq,
                reading.t_mono_ns,
                reading.decoded_total
            ),
            (7, 8, 9, Some(10))
        );
        let view = LlamaView {
            ai: AiState::NoData,
            models: Vec::new(),
            decoded_total: None,
            prompt_total: None,
        };
        assert_eq!(view.ai, AiState::NoData);
        assert!(view.models.is_empty());
        assert_eq!(view.decoded_total, None);
    }
}
