//! Snapshot v1, the watcher-to-writer contract.
//!
//! The writer accepts bytes only through [`parse_validated`]: the size cap,
//! then the parse, then [`validate`]. [`from_bytes`] parses without those
//! rules and is crate-private.

use serde::{Deserialize, Serialize};
use thiserror::Error;

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

/// Canonical model-name width, including the trailing `…`.
///
/// [`MAX_NAME_CHARS`] is only a cheap length pre-check. A name that
/// [`crate::names::sanitize_wire`] emits is at most this long.
pub const CANONICAL_NAME_CHARS: usize = 12;

/// Why a buffer was not accepted as a v1 snapshot.
#[derive(Debug, Error, PartialEq, Eq)]
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
    /// A full name is not canonical, or a detail token is not allowlisted.
    #[error("snapshot model detail is not canonical")]
    Detail,
    /// The snapshot could not be encoded. Not signalled by an empty buffer.
    #[error("snapshot could not be encoded")]
    Encode,
}

/// Validated snapshot v1. Produced by [`parse_validated`].
pub type SnapshotV1 = WireSnapshot;

/// One published sample.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
}

/// Host numbers. Percents are 0..=100, except `activity_pct`, which is
/// 0..=[`ACTIVITY_MAX_PCT`]. Temperatures are −20..=150 °C.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
}

/// llama-swap state and the models to draw.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

/// Box decoded-token counter since watcher start.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tokens {
    /// `None` when the counter was not measured. No numeric range.
    pub decoded_total: Option<u64>,
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
    if snapshot.ai.models.len() > MAX_MODELS {
        return Err(WireError::TooManyModels);
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
            },
            ai: Ai {
                state: AiWire::Down,
                models: Vec::new(),
            },
            tokens: Tokens {
                decoded_total: None,
            },
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
