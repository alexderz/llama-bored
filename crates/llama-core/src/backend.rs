//! Which inference server sits behind a llama-swap model (T72).
//!
//! llama.cpp (and its forks) answer `/slots` and `llamacpp:*` metrics.
//! SGLang and vLLM have their own Prometheus names and no `/slots`. Any
//! other OpenAI-compatible server (TabbyAPI, ExLlamaV3) has neither, and
//! the watcher falls back to llama-swap's activity log for it.

use serde::{Deserialize, Serialize};

/// Most requests a `running` or `queued` gauge may carry on the wire.
pub const MAX_REQS: u16 = 4096;

/// Server kind. The wire and `watch.toml` use the lowercase word.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// llama-server, mainline or a fork.
    #[default]
    LlamaCpp,
    /// SGLang (`sglang.launch_server`, `sglang serve`).
    SgLang,
    /// vLLM (`vllm serve`, `vllm.entrypoints`).
    Vllm,
    /// Any other OpenAI-compatible server.
    OpenAi,
}

impl Backend {
    /// The wire and config word.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LlamaCpp => "llamacpp",
            Self::SgLang => "sglang",
            Self::Vllm => "vllm",
            Self::OpenAi => "openai",
        }
    }

    /// Only llama.cpp answers `/slots`.
    #[must_use]
    pub fn has_slots(self) -> bool {
        self == Self::LlamaCpp
    }

    /// Whether `/upstream/<id>/metrics` is worth a GET.
    #[must_use]
    pub fn has_metrics(self) -> bool {
        self != Self::OpenAi
    }
}

/// Backend kind plus the live gauges its `/metrics` gave. Gauges are `None`
/// when unknown; ratios are per mille so the type stays `Eq`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BackendInfo {
    /// Server kind.
    pub kind: Backend,
    /// `--max-running-requests` / `--max-num-seqs` from the launch command.
    pub max_running: Option<u16>,
    /// Requests running now, at most [`MAX_REQS`].
    pub running: Option<u16>,
    /// Requests waiting, at most [`MAX_REQS`].
    pub queued: Option<u16>,
    /// KV cache fill, 0..=1000.
    pub kv_permille: Option<u16>,
    /// Prefix cache hit rate, 0..=1000.
    pub hit_permille: Option<u16>,
}

/// A request count clamped to [`MAX_REQS`]. Non-finite or negative is `None`.
#[must_use]
pub fn reqs(value: f64) -> Option<u16> {
    (value.is_finite() && value >= 0.0).then(|| value.min(f64::from(MAX_REQS)) as u16)
}

/// A 0..=1 ratio as 0..=1000. Anything else is `None`.
#[must_use]
pub fn permille(ratio: f64) -> Option<u16> {
    (ratio.is_finite() && (0.0..=1.0).contains(&ratio)).then(|| (ratio * 1000.0).round() as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_round_trip_through_serde() {
        for kind in [
            Backend::LlamaCpp,
            Backend::SgLang,
            Backend::Vllm,
            Backend::OpenAi,
        ] {
            let json = serde_json::to_string(&kind).expect("encode");
            assert_eq!(json, format!("\"{}\"", kind.as_str()));
            assert_eq!(
                serde_json::from_str::<Backend>(&json).expect("decode"),
                kind
            );
        }
        assert!(serde_json::from_str::<Backend>("\"tabby\"").is_err());
        assert!(serde_json::from_str::<Backend>("\"SGLang\"").is_err());
    }

    #[test]
    fn only_llamacpp_has_slots_and_openai_has_no_metrics() {
        assert!(Backend::LlamaCpp.has_slots());
        assert!(!Backend::SgLang.has_slots());
        assert!(!Backend::Vllm.has_slots());
        assert!(Backend::SgLang.has_metrics());
        assert!(!Backend::OpenAi.has_metrics());
    }

    #[test]
    fn gauges_are_bounded() {
        assert_eq!(reqs(3.0), Some(3));
        assert_eq!(reqs(1e9), Some(MAX_REQS));
        assert_eq!(reqs(-1.0), None);
        assert_eq!(reqs(f64::NAN), None);
        assert_eq!(permille(0.3712), Some(371));
        assert_eq!(permille(1.0), Some(1000));
        assert_eq!(permille(1.2), None);
        assert_eq!(permille(-0.1), None);
    }
}
