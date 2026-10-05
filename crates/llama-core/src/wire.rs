//! Snapshot v1, the watcher-to-writer contract.
//!
//! The writer accepts bytes only through [`parse_validated`]: the size cap,
//! then the parse, then [`validate`]. [`from_bytes`] parses without those
//! rules and is crate-private.
//!
//! # Compatibility
//!
//! A reader ignores fields it does not know, within the same [`SCHEMA`]
//! number. A newer watcher may add an optional field to schema 1, and an
//! older kraken-lcd, llama-light or llama-metrics still accepts the file and
//! simply does not see it (#12; before 0.2 every unknown field was a reject,
//! which took old readers down on an upgrade). Known fields keep every check:
//! the exact `schema`, [`MAX_BYTES`] before the parse, types, ranges, and
//! the name and token allowlists in [`validate`]. An unknown field's value is
//! dropped at the parse, so it reaches no display and no export. A change
//! that an older reader must not misread (a renamed field, a new meaning, a
//! tighter range the old reader would not enforce) needs a new schema number.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::backend::{Backend, MAX_ENGINE_LATENCY_S, MAX_ENGINE_TPS, MAX_REQS, MAX_SPEC_LEN};
pub use crate::detail::{MAX_FULL_NAME_CHARS, ModelDetail};

/// Wire schema version this crate reads and writes.
pub const SCHEMA: u8 = 1;
/// Largest snapshot accepted before parsing.
pub const MAX_BYTES: usize = 16 * 1024;
/// Top of `host.activity_pct`. 100 is nominal sustained load; the watcher
/// pins spikes at this value. Every other percent tops out at 100.
pub const ACTIVITY_MAX_PCT: f32 = 125.0;
/// Most models a snapshot may carry.
pub const MAX_MODELS: usize = 8;
/// Cheap length pre-check, in Unicode scalars.
///
/// The real limit is [`CANONICAL_NAME_CHARS`] (12), and that count includes
/// the trailing `…`. A 13-scalar name fails the canonical check.
pub const MAX_NAME_CHARS: usize = 13;
/// Directory of the published snapshot. Not configurable.
pub const SNAPSHOT_DIR: &str = "/run/llama-watch";
/// Path of the published snapshot. Not configurable.
pub const SNAPSHOT_PATH: &str = "/run/llama-watch/snapshot.json";

/// Top of every watts field (GPU, GPU limit, CPU socket).
pub const MAX_WATTS: f32 = 5000.0;
/// Top of every byte-count field (VRAM, system memory): 1 PiB.
pub const MAX_MEM_BYTES: u64 = 1 << 50;
/// Most slots a model may report.
pub const MAX_SLOTS: u16 = 1024;
/// Most fans a snapshot may carry (`[fans] channels` allows 8).
pub const MAX_FANS: usize = 8;
/// Highest fan channel number (`[fans] channels` are 1..=16).
pub const MAX_FAN_CHANNEL: u8 = 16;
/// Longest fan label, in characters (`[fans] labels`).
pub const MAX_FAN_LABEL_CHARS: usize = 10;
/// Top of a fan's rpm.
pub const MAX_FAN_RPM: u32 = 100_000;
/// Top of a source's last poll latency, seconds.
pub const MAX_LATENCY_S: f32 = 600.0;
/// Most per-slot context rows one snapshot carries, across every model
/// (#10). Keeps the worst case under [`MAX_BYTES`]; the watcher sends the
/// first models' lowest slots and drops the rest.
pub const MAX_SLOT_CTX: usize = 32;
/// Top of a slot's context tokens (#10).
pub const MAX_CTX_TOKENS: u64 = u32::MAX as u64;

/// Canonical model-name width, including the trailing `…`.
///
/// [`MAX_NAME_CHARS`] is only a cheap length pre-check. A name that
/// [`crate::names::sanitize_wire`] emits is at most this long.
pub const CANONICAL_NAME_CHARS: usize = 12;

/// Why a buffer was not accepted as a v1 snapshot.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum WireError {
    /// `bytes.len()` is greater than [`MAX_BYTES`].
    #[error("snapshot length {len} exceeds the maximum")]
    TooLong {
        /// Length of the rejected buffer.
        len: usize,
    },
    /// The bytes are not a snapshot object.
    #[error("snapshot JSON is not a v1 object")]
    Parse,
    /// `schema` is not [`SCHEMA`].
    #[error("snapshot schema is not 1")]
    Schema,
    /// A percent or temperature is non-finite or outside its range.
    #[error("snapshot field {field} is out of range")]
    OutOfRange {
        /// Static host field name.
        field: &'static str,
    },
    /// `ai.models` has more than [`MAX_MODELS`] entries.
    #[error("snapshot has more than 8 models")]
    TooManyModels,
    /// Models are present while `ai.state` is not `loaded`.
    #[error("snapshot models must be empty unless state is loaded")]
    ModelsNotLoaded,
    /// A model name is empty, too long, or not canonical.
    #[error("snapshot model name is not canonical")]
    Name,
    /// A model's `running` or `queued` is above [`MAX_REQS`], its
    /// `kv_fill` is non-finite or outside 0..=1, its engine numbers are out
    /// of range ([`EngineWire::is_valid`]), its cached prompt tokens
    /// exceed its prompt tokens, or a slot context row is out of range,
    /// repeated, or past [`MAX_SLOT_CTX`] (#10).
    #[error("snapshot model gauge is out of range")]
    Gauge,
    /// A full name is not canonical, or a detail token is not allowlisted.
    #[error("snapshot model detail is not canonical")]
    Detail,
    /// More than [`MAX_FANS`] fans, a repeated or out-of-range channel, a
    /// label that is not short printable ASCII, or an rpm or pwm out of range.
    #[error("snapshot fan is out of range")]
    Fan,
    /// The snapshot could not be encoded. Not signalled by an empty buffer.
    #[error("snapshot could not be encoded")]
    Encode,
}

/// Validated snapshot v1. Produced by [`parse_validated`].
pub type SnapshotV1 = WireSnapshot;

/// One published sample.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WireSnapshot {
    /// Must be [`SCHEMA`].
    pub schema: u8,
    /// Random at watcher start.
    pub run_id: u64,
    /// Plus one per publish within [`Self::run_id`].
    pub seq: u64,
    /// `CLOCK_MONOTONIC` at sample time, in nanoseconds.
    pub t_mono_ns: u64,
    /// Wall clock in milliseconds. Logs only.
    pub t_wall_ms: u64,
    /// Host telemetry. `None` means that source failed.
    pub host: Host,
    /// llama-swap state and display names.
    pub ai: Ai,
    /// Decoded-token counter.
    pub tokens: Tokens,
    /// Fan speeds when `[fans]` is on (#11). Omitted when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fans: Vec<FanWire>,
    /// Source health, the tty health line (#11). Omitted by an older watcher.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<Sources>,
}

/// Host numbers. Percents are 0..=100, except `activity_pct`, which is
/// 0..=[`ACTIVITY_MAX_PCT`]. Temperatures are −20..=150 °C.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Host {
    /// Composite load percent: `max(gpu, cpu_topk)`.
    #[serde(with = "finite_f32")]
    pub load_pct: Option<f32>,
    /// Power-weighted activity percent, 0..=[`ACTIVITY_MAX_PCT`]. 100 is the
    /// nominal sustained ceiling; spikes read above it. Omitted by an older watcher.
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub activity_pct: Option<f32>,
    /// Mean CPU percent.
    #[serde(with = "finite_f32")]
    pub cpu_pct: Option<f32>,
    /// Mean of the busiest CPUs, percent.
    #[serde(with = "finite_f32")]
    pub cpu_topk_pct: Option<f32>,
    /// GPU utilisation percent.
    #[serde(with = "finite_f32")]
    pub gpu_pct: Option<f32>,
    /// Memory percent.
    #[serde(with = "finite_f32")]
    pub mem_pct: Option<f32>,
    /// Coolant temperature, °C.
    #[serde(with = "finite_f32")]
    pub coolant_c: Option<f32>,
    /// CPU temperature, °C.
    #[serde(with = "finite_f32")]
    pub cpu_c: Option<f32>,
    /// GPU temperature, °C.
    #[serde(with = "finite_f32")]
    pub gpu_c: Option<f32>,
    /// GPU power draw, watts, 0..=[`MAX_WATTS`] (#11). Every field below is
    /// optional on schema v1 and omitted when unknown.
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub gpu_w: Option<f32>,
    /// GPU enforced power limit, watts.
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub gpu_limit_w: Option<f32>,
    /// CPU socket power, watts.
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub cpu_w: Option<f32>,
    /// VRAM in use, bytes, at most [`MAX_MEM_BYTES`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_used_bytes: Option<u64>,
    /// VRAM total, bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_total_bytes: Option<u64>,
    /// System memory in use (`MemTotal - MemAvailable`), bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mem_used_bytes: Option<u64>,
    /// System memory total, bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mem_total_bytes: Option<u64>,
}

/// llama-swap state and the models to draw.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ai {
    /// Reachability.
    pub state: AiWire,
    /// Empty unless [`Self::state`] is [`AiWire::Loaded`].
    pub models: Vec<ModelWire>,
}

/// llama-swap reachability on the wire.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AiWire {
    /// llama-swap could not be read.
    Down,
    /// Reachable, nothing loaded.
    Idle,
    /// One or more models loaded.
    Loaded,
}

/// One display name and its upstream state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelWire {
    /// Canonical display name. See [`validate`].
    pub name: String,
    /// Upstream lifecycle word.
    pub state: ModelState,
    /// Untruncated display name, at most [`MAX_FULL_NAME_CHARS`]. Omitted
    /// by an older watcher and when it equals [`Self::name`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_name: Option<String>,
    /// Tuning detail from the launch command. Omitted by an older watcher.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ModelDetail>,
    /// Server kind (T72). Omitted by an older watcher; absent reads as llama.cpp.
    /// A word this reader does not know reads as [`Backend::OpenAi`], so a
    /// newer watcher's backend never rejects the snapshot.
    #[serde(
        default,
        deserialize_with = "Backend::deserialize_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub backend: Option<Backend>,
    /// Requests running, from a backend without `/slots`. At most [`MAX_REQS`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub running: Option<u16>,
    /// Requests waiting, from a backend without `/slots`. At most [`MAX_REQS`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queued: Option<u16>,
    /// KV cache fill 0..=1, from a backend without `/slots`.
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub kv_fill: Option<f32>,
    /// Prefix cache hit ratio 0..=1, from a backend without `/slots` (#11).
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub cache_hit: Option<f32>,
    /// llama.cpp slots busy now, at most [`Self::slots_total`] (#11).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slots_busy: Option<u16>,
    /// llama.cpp slots the server has, at most [`MAX_SLOTS`] (#11).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slots_total: Option<u16>,
    /// Prompt tokens of the model's finished requests since the watcher
    /// started, cached ones included (#10). A counter: it restarts with the
    /// watcher. Omitted when not measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Of [`Self::prompt_tokens`], the tokens served from the prompt cache
    /// (#10). Never above it, and absent without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cached_tokens: Option<u64>,
    /// llama.cpp per-slot context (#10). At most [`MAX_SLOT_CTX`] rows in
    /// the whole snapshot. Omitted when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slot_ctx: Vec<SlotCtxWire>,
    /// Engine numbers from a backend without `/slots` (#31): speculative
    /// decoding, preemptions, sleep, latency means. Omitted when none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<EngineWire>,
}

/// A server's own engine numbers (#31). Every field is optional. Window
/// means cover the latest metrics poll window that saw activity; counters
/// count since the watcher started.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EngineWire {
    /// Speculative acceptance, accepted / draft tokens, 0..=1.
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub spec_accept: Option<f32>,
    /// Mean tokens per speculative step, 1..=[`MAX_SPEC_LEN`].
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub spec_len: Option<f32>,
    /// Speculative draft rounds since the watcher started.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_drafts: Option<u64>,
    /// Tokens drafted since the watcher started.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_draft_tokens: Option<u64>,
    /// Of [`Self::spec_draft_tokens`], those accepted. Never above it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_accepted_tokens: Option<u64>,
    /// Preemptions since the watcher started.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preemptions: Option<u64>,
    /// The engine is asleep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sleeping: Option<bool>,
    /// Mean time to first token, seconds, 0..=[`MAX_ENGINE_LATENCY_S`].
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub ttft_s: Option<f32>,
    /// Mean inter-token latency, seconds.
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub itl_s: Option<f32>,
    /// Mean end-to-end request latency, seconds.
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub e2e_s: Option<f32>,
    /// Prefill tokens per second of the requests that finished in the
    /// latest metrics window with any (#35), 0..=[`MAX_ENGINE_TPS`].
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub prefill_tps: Option<f32>,
    /// Decode tokens per second over the same window, 0..=[`MAX_ENGINE_TPS`].
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub decode_tps: Option<f32>,
    /// Expert cache hit rate of the newest finished request, 0..=1 (#54,
    /// Strata). Omitted by an older watcher and by other engines.
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub expert_hit: Option<f32>,
    /// Share of that request's expert reads served over PCIe, 0..=1 (#54).
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub pcie_share: Option<f32>,
}

impl EngineWire {
    /// True when the numbers are in range: ratios in 0..=1, a step length
    /// in 1..=[`MAX_SPEC_LEN`], accepted tokens not above draft tokens,
    /// latencies in 0..=[`MAX_ENGINE_LATENCY_S`], and speeds in
    /// 0..=[`MAX_ENGINE_TPS`].
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let in_range = |value: Option<f32>, low: f64, high: f64| {
            value.is_none_or(|v| v.is_finite() && (low..=high).contains(&f64::from(v)))
        };
        [self.spec_accept, self.expert_hit, self.pcie_share]
            .into_iter()
            .all(|ratio| in_range(ratio, 0.0, 1.0))
            && in_range(self.spec_len, 1.0, MAX_SPEC_LEN)
            && [self.ttft_s, self.itl_s, self.e2e_s]
                .into_iter()
                .all(|latency| in_range(latency, 0.0, MAX_ENGINE_LATENCY_S))
            && [self.prefill_tps, self.decode_tps]
                .into_iter()
                .all(|tps| in_range(tps, 0.0, MAX_ENGINE_TPS))
            && self
                .spec_accepted_tokens
                .is_none_or(|accepted| self.spec_draft_tokens.is_some_and(|all| accepted <= all))
    }
}

/// One llama.cpp slot's context (#10). Numbers only.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SlotCtxWire {
    /// Slot id, below [`MAX_SLOTS`], unique within its model.
    pub slot: u16,
    /// Context tokens the slot holds: prompt plus decoded. An idle slot
    /// keeps its last busy value, as its KV cache does. At most
    /// [`MAX_CTX_TOKENS`].
    pub used: u64,
    /// Context drops seen since the watcher started (the SLOTS sparkline's
    /// reset rule), by the watcher's best-guess reason (#9). Counters: they
    /// restart with the watcher.
    #[serde(default)]
    pub resets: SlotResetsWire,
}

/// A slot's context drops by reason (#9). Every reason is a guess from
/// token counts; `llama_watch::resets` states the rules. A zero is omitted.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SlotResetsWire {
    /// Same conversation, shorter prompt, mostly cached.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub compacted: u64,
    /// A different conversation took the slot.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub new: u64,
    /// A conversation came back with its cache gone.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub evicted: u64,
    /// Not enough evidence.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub unknown: u64,
}

impl SlotResetsWire {
    /// Every count with its label word, in export order.
    #[must_use]
    pub fn entries(&self) -> [(&'static str, u64); 4] {
        [
            ("compacted", self.compacted),
            ("new", self.new),
            ("evicted", self.evicted),
            ("unknown", self.unknown),
        ]
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_zero(value: &u64) -> bool {
    *value == 0
}

/// Upstream model lifecycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelState {
    /// Serving.
    Ready,
    /// Load in progress.
    Starting,
    /// Unload in progress.
    Stopping,
    /// Any other upstream word.
    Other,
}

/// Box token counters since watcher start.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Tokens {
    /// `None` when the counter was not measured. No numeric range.
    pub decoded_total: Option<u64>,
    /// Prompt tokens processed, like `decoded_total` (#11). Omitted when
    /// not measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_total: Option<u64>,
}

/// One configured fan (#11). Read-only numbers from the watcher's hwmon.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FanWire {
    /// Channel `N`, 1..=[`MAX_FAN_CHANNEL`], unique in the snapshot.
    pub channel: u8,
    /// The `[fans] labels` entry, or `fanN`: 1..=[`MAX_FAN_LABEL_CHARS`]
    /// printable ASCII characters.
    pub label: String,
    /// `fanN_input`, rpm, at most [`MAX_FAN_RPM`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rpm: Option<u32>,
    /// `pwmN` as 0..=1.
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub pwm: Option<f32>,
}

/// One source on the tty health line (#11).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceWire {
    /// `true` when the source answered (tty OK or idle), `false` when down.
    pub up: bool,
    /// Last poll duration, seconds, 0..=[`MAX_LATENCY_S`]. Omitted when the
    /// tty shows none.
    #[serde(default, with = "finite_f32", skip_serializing_if = "Option::is_none")]
    pub latency_s: Option<f32>,
}

/// Source health by fixed name. A source the watcher did not poll this tick
/// (llama-swap off or down, or still starting) is absent. A fixed set of
/// named fields, so a source added later is an unknown field to an older
/// reader, not an error.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Sources {
    /// llama-swap reachability and its `/running` latency.
    #[serde(
        rename = "llama-swap",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub llama_swap: Option<SourceWire>,
    /// `/running`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub running: Option<SourceWire>,
    /// `/upstream/<model>/slots`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slots: Option<SourceWire>,
    /// `/upstream/<model>/metrics`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<SourceWire>,
    /// `/api/metrics/activity`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activity: Option<SourceWire>,
    /// The GPU (NVML; `nvml` on the tty health line). Named `gpu` here and
    /// in the export, as the S11 and S16 fences keep that word out of the
    /// core and the exporter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<SourceWire>,
    /// hwmon (coolant, CPU temperature).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hwmon: Option<SourceWire>,
    /// `/proc` (CPU, memory).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proc: Option<SourceWire>,
}

impl Sources {
    /// Every source, by its wire and label name, in health-line order.
    #[must_use]
    pub fn entries(&self) -> [(&'static str, Option<SourceWire>); 8] {
        [
            ("llama-swap", self.llama_swap),
            ("running", self.running),
            ("slots", self.slots),
            ("metrics", self.metrics),
            ("activity", self.activity),
            ("gpu", self.gpu),
            ("hwmon", self.hwmon),
            ("proc", self.proc),
        ]
    }
}

/// Serialise `snapshot` as compact JSON.
///
/// A non-finite host number is written as JSON null. [`parse_validated`]
/// reads that null back as `None`. A failure is [`WireError::Encode`], never
/// an empty buffer.
pub fn to_json(snapshot: &WireSnapshot) -> Result<Vec<u8>, WireError> {
    serde_json::to_vec(snapshot).map_err(|_| WireError::Encode)
}

/// Parse a snapshot. The length is checked before any parse.
///
/// Does not apply [`validate`]. Crate-private so the writer cannot skip
/// [`parse_validated`].
pub(crate) fn from_bytes(bytes: &[u8]) -> Result<WireSnapshot, WireError> {
    if bytes.len() > MAX_BYTES {
        return Err(WireError::TooLong { len: bytes.len() });
    }
    serde_json::from_slice(bytes).map_err(|_| WireError::Parse)
}

/// The only function the writer may use to accept snapshot bytes.
///
/// Size cap, then parse, then [`validate`].
pub fn parse_validated(bytes: &[u8]) -> Result<SnapshotV1, WireError> {
    let snapshot = from_bytes(bytes)?;
    validate(&snapshot)?;
    Ok(snapshot)
}

/// Range, count, and canonical-name rules. Run this after [`from_bytes`].
pub fn validate(snapshot: &WireSnapshot) -> Result<(), WireError> {
    if snapshot.schema != SCHEMA {
        return Err(WireError::Schema);
    }
    check_pct("load_pct", snapshot.host.load_pct)?;
    check_range(
        "activity_pct",
        snapshot.host.activity_pct,
        0.0,
        ACTIVITY_MAX_PCT,
    )?;
    check_pct("cpu_pct", snapshot.host.cpu_pct)?;
    check_pct("cpu_topk_pct", snapshot.host.cpu_topk_pct)?;
    check_pct("gpu_pct", snapshot.host.gpu_pct)?;
    check_pct("mem_pct", snapshot.host.mem_pct)?;
    check_temp("coolant_c", snapshot.host.coolant_c)?;
    check_temp("cpu_c", snapshot.host.cpu_c)?;
    check_temp("gpu_c", snapshot.host.gpu_c)?;
    check_range("gpu_w", snapshot.host.gpu_w, 0.0, MAX_WATTS)?;
    check_range("gpu_limit_w", snapshot.host.gpu_limit_w, 0.0, MAX_WATTS)?;
    check_range("cpu_w", snapshot.host.cpu_w, 0.0, MAX_WATTS)?;
    for (field, bytes) in [
        ("vram_used_bytes", snapshot.host.vram_used_bytes),
        ("vram_total_bytes", snapshot.host.vram_total_bytes),
        ("mem_used_bytes", snapshot.host.mem_used_bytes),
        ("mem_total_bytes", snapshot.host.mem_total_bytes),
    ] {
        if bytes.is_some_and(|b| b > MAX_MEM_BYTES) {
            return Err(WireError::OutOfRange { field });
        }
    }
    validate_fans(&snapshot.fans)?;
    if let Some(sources) = &snapshot.sources {
        for (_, source) in sources.entries() {
            check_range(
                "latency_s",
                source.and_then(|s| s.latency_s),
                0.0,
                MAX_LATENCY_S,
            )?;
        }
    }
    if snapshot.ai.models.len() > MAX_MODELS {
        return Err(WireError::TooManyModels);
    }
    let slot_rows: usize = snapshot.ai.models.iter().map(|m| m.slot_ctx.len()).sum();
    if slot_rows > MAX_SLOT_CTX {
        return Err(WireError::Gauge);
    }
    if snapshot.ai.state != AiWire::Loaded && !snapshot.ai.models.is_empty() {
        return Err(WireError::ModelsNotLoaded);
    }
    for model in &snapshot.ai.models {
        validate_name(&model.name)?;
        if let Some(full) = &model.full_name {
            validate_full_name(full)?;
        }
        if let Some(detail) = &model.detail
            && !crate::detail::is_valid(detail)
        {
            return Err(WireError::Detail);
        }
        let not_ratio = |value: Option<f32>| {
            value.is_some_and(|ratio| !ratio.is_finite() || !(0.0..=1.0).contains(&ratio))
        };
        if model.running.is_some_and(|n| n > MAX_REQS)
            || model.queued.is_some_and(|n| n > MAX_REQS)
            || not_ratio(model.kv_fill)
            || not_ratio(model.cache_hit)
            || model.slots_total.is_some_and(|n| n > MAX_SLOTS)
            || model
                .slots_busy
                .is_some_and(|busy| model.slots_total.is_none_or(|total| busy > total))
            || model
                .prompt_cached_tokens
                .is_some_and(|cached| model.prompt_tokens.is_none_or(|all| cached > all))
        {
            return Err(WireError::Gauge);
        }
        if model
            .engine
            .as_ref()
            .is_some_and(|engine| !engine.is_valid())
        {
            return Err(WireError::Gauge);
        }
        validate_slot_ctx(&model.slot_ctx)?;
    }
    Ok(())
}

fn validate_slot_ctx(rows: &[SlotCtxWire]) -> Result<(), WireError> {
    for (i, row) in rows.iter().enumerate() {
        if row.slot >= MAX_SLOTS
            || row.used > MAX_CTX_TOKENS
            || rows[..i].iter().any(|other| other.slot == row.slot)
        {
            return Err(WireError::Gauge);
        }
    }
    Ok(())
}

fn validate_fans(fans: &[FanWire]) -> Result<(), WireError> {
    if fans.len() > MAX_FANS {
        return Err(WireError::Fan);
    }
    for (i, fan) in fans.iter().enumerate() {
        if !(1..=MAX_FAN_CHANNEL).contains(&fan.channel)
            || fans[..i].iter().any(|other| other.channel == fan.channel)
            || fan.label.is_empty()
            || fan.label.chars().count() > MAX_FAN_LABEL_CHARS
            || !fan
                .label
                .chars()
                .all(|c| ('\u{20}'..='\u{7e}').contains(&c))
            || fan.rpm.is_some_and(|rpm| rpm > MAX_FAN_RPM)
            || fan
                .pwm
                .is_some_and(|pwm| !pwm.is_finite() || !(0.0..=1.0).contains(&pwm))
        {
            return Err(WireError::Fan);
        }
    }
    Ok(())
}

fn check_pct(field: &'static str, value: Option<f32>) -> Result<(), WireError> {
    check_range(field, value, 0.0, 100.0)
}

fn check_temp(field: &'static str, value: Option<f32>) -> Result<(), WireError> {
    check_range(field, value, -20.0, 150.0)
}

fn check_range(
    field: &'static str,
    value: Option<f32>,
    low: f32,
    high: f32,
) -> Result<(), WireError> {
    if let Some(value) = value
        && (!value.is_finite() || !(low..=high).contains(&value))
    {
        return Err(WireError::OutOfRange { field });
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<(), WireError> {
    if name.is_empty()
        || name.chars().count() > MAX_NAME_CHARS
        || !name.chars().all(is_name_char)
        || name != crate::names::sanitize_wire(name)
    {
        return Err(WireError::Name);
    }
    Ok(())
}

fn validate_full_name(name: &str) -> Result<(), WireError> {
    if name.is_empty()
        || name.chars().count() > MAX_FULL_NAME_CHARS
        || !name.chars().all(is_name_char)
        || name != crate::names::sanitize(name, MAX_FULL_NAME_CHARS)
    {
        return Err(WireError::Detail);
    }
    Ok(())
}

fn is_name_char(c: char) -> bool {
    ('\u{20}'..='\u{7e}').contains(&c) || c == '…'
}

/// `Option<f32>` that writes non-finite numbers as null.
mod finite_f32 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &Option<f32>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match *value {
            Some(number) if number.is_finite() => serializer.serialize_some(&number),
            _ => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<f32>::deserialize(deserializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn down() -> WireSnapshot {
        WireSnapshot {
            schema: SCHEMA,
            run_id: 1,
            seq: 1,
            t_mono_ns: 1,
            t_wall_ms: 1,
            host: Host {
                load_pct: None,
                activity_pct: None,
                cpu_pct: None,
                cpu_topk_pct: None,
                gpu_pct: None,
                mem_pct: None,
                coolant_c: None,
                cpu_c: None,
                gpu_c: None,
                gpu_w: None,
                gpu_limit_w: None,
                cpu_w: None,
                vram_used_bytes: None,
                vram_total_bytes: None,
                mem_used_bytes: None,
                mem_total_bytes: None,
            },
            ai: Ai {
                state: AiWire::Down,
                models: Vec::new(),
            },
            tokens: Tokens {
                decoded_total: None,
                prompt_total: None,
            },
            fans: Vec::new(),
            sources: None,
        }
    }

    #[test]
    fn from_bytes_is_unvalidated() {
        let mut snap = down();
        snap.schema = 2;
        let bytes = to_json(&snap).expect("encode");
        let parsed = from_bytes(&bytes).expect("schema 2 still parses");
        assert_eq!(parsed.schema, 2);
        assert_eq!(parse_validated(&bytes), Err(WireError::Schema));
    }
}
