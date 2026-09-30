//! Which inference server sits behind a llama-swap model (T72).
//!
//! llama.cpp (and its forks) answer `/slots` and `llamacpp:*` metrics.
//! SGLang and vLLM have their own Prometheus names and no `/slots`. Any
//! other OpenAI-compatible server (TabbyAPI, ExLlamaV3) has neither, and
//! the watcher falls back to llama-swap's activity log for it. Strata serves
//! one request at a time and answers `/metrics` with JSON, not Prometheus.

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
    /// Strata (`serve/server.py --engine strata`): JSON `/metrics`, one
    /// request at a time, no usable `/slots`.
    Strata,
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
            Self::Strata => "strata",
            Self::OpenAi => "openai",
        }
    }

    /// The kind for a wire or config word; `None` for an unknown word.
    #[must_use]
    pub fn from_word(word: &str) -> Option<Self> {
        [
            Self::LlamaCpp,
            Self::SgLang,
            Self::Vllm,
            Self::Strata,
            Self::OpenAi,
        ]
        .into_iter()
        .find(|kind| kind.as_str() == word)
    }

    /// Wire read of a backend word. A word this reader does not know (a
    /// newer watcher's backend) reads as [`Self::OpenAi`], a server with no
    /// gauges this reader can show, instead of rejecting the snapshot.
    /// `watch.toml` stays strict: it uses the derived `Deserialize`.
    pub fn deserialize_lenient<'de, D>(deserializer: D) -> Result<Option<Self>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let word = Option::<std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        Ok(word.map(|word| Self::from_word(&word).unwrap_or(Self::OpenAi)))
    }

    /// Whether the server reports a KV cache fill. Strata has none.
    #[must_use]
    pub fn has_kv_gauge(self) -> bool {
        self != Self::Strata
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
            Backend::Strata,
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
        assert_eq!(Backend::from_word("strata"), Some(Backend::Strata));
        assert_eq!(Backend::from_word("Strata"), None);
    }

    #[derive(serde::Deserialize)]
    struct Lenient {
        #[serde(default, deserialize_with = "Backend::deserialize_lenient")]
        backend: Option<Backend>,
    }

    #[test]
    fn lenient_read_maps_unknown_words_to_openai() {
        let read = |json: &str| serde_json::from_str::<Lenient>(json).map(|l| l.backend);
        assert_eq!(
            read(r#"{"backend":"strata"}"#).ok(),
            Some(Some(Backend::Strata))
        );
        assert_eq!(
            read(r#"{"backend":"sglang"}"#).ok(),
            Some(Some(Backend::SgLang))
        );
        assert_eq!(
            read(r#"{"backend":"tabby"}"#).ok(),
            Some(Some(Backend::OpenAi))
        );
        assert_eq!(
            read(r#"{"backend":"strata\u0031"}"#).ok(),
            Some(Some(Backend::OpenAi))
        );
        assert_eq!(read(r#"{"backend":null}"#).ok(), Some(None));
        assert_eq!(read("{}").ok(), Some(None));
        assert!(read(r#"{"backend":7}"#).is_err());
    }

    #[test]
    fn only_llamacpp_has_slots_and_openai_has_no_metrics() {
        assert!(Backend::LlamaCpp.has_slots());
        assert!(!Backend::SgLang.has_slots());
        assert!(!Backend::Vllm.has_slots());
        assert!(!Backend::Strata.has_slots());
        assert!(Backend::Strata.has_metrics());
        assert!(!Backend::Strata.has_kv_gauge());
        assert!(Backend::SgLang.has_kv_gauge());
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
