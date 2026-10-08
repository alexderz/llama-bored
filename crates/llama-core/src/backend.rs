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

    /// Display name (#33): `llama.cpp`, `SGLang`, `vLLM`, `Strata`,
    /// `OpenAI-compatible`. A llama.cpp fork stays `llama.cpp`.
    #[must_use]
    pub fn display_name(self) -> &'static str {
        match self {
            Self::LlamaCpp => "llama.cpp",
            Self::SgLang => "SGLang",
            Self::Vllm => "vLLM",
            Self::Strata => "Strata",
            Self::OpenAi => "OpenAI-compatible",
        }
    }

    /// Short display name for tight spots (#33): `llama.cpp`, `sglang`,
    /// `vllm`, `strata`, `openai`.
    #[must_use]
    pub fn short_name(self) -> &'static str {
        match self {
            Self::LlamaCpp => "llama.cpp",
            other => other.as_str(),
        }
    }

    /// The kind whose [`Self::display_name`] is `name`; `None` otherwise.
    #[must_use]
    pub fn from_display_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.display_name() == name)
    }

    /// Every kind, in declaration order.
    pub const ALL: [Self; 5] = [
        Self::LlamaCpp,
        Self::SgLang,
        Self::Vllm,
        Self::Strata,
        Self::OpenAi,
    ];

    /// The kind for a wire or config word; `None` for an unknown word.
    #[must_use]
    pub fn from_word(word: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == word)
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

    /// Whether the server reports a KV cache fill ratio of its own. Strata
    /// has none; its KV comes as tokens (#79).
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
    /// Engine numbers beyond the gauges: speculative decoding, preemptions,
    /// sleep and latency means (#31). vLLM, and SGLang where its names allow.
    pub engine: EngineStats,
    /// KV cache in use across every session, and its capacity (#79).
    /// `None` when the engine gave neither.
    pub kv: Option<KvUsage>,
}

/// A model's KV cache across all its sessions (#79), in tokens. Every
/// number is `None` when the engine does not report it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KvUsage {
    /// Tokens the KV cache holds for live sessions: llama.cpp's and
    /// Strata's slots (idle ones keep their context), SGLang's
    /// `kv_used_tokens`, vLLM's usage ratio times its capacity.
    pub used: Option<u64>,
    /// Tokens the KV cache can hold.
    pub capacity: Option<u64>,
    /// Tokens held only as reusable cache, not by a live session (SGLang's
    /// radix cache, `kv_evictable_tokens`). Not part of [`Self::used`].
    pub cached: Option<u64>,
    /// Sessions holding KV: slots with tokens (llama.cpp, Strata), else
    /// running requests (SGLang, vLLM).
    pub sessions: Option<u16>,
    /// llama.cpp only: `true` when every slot shares one KV pool
    /// (`--kv-unified`, the default with an automatic slot count).
    pub unified: Option<bool>,
    /// [`Self::unified`] was not in the launch command: it is assumed
    /// because every slot reports the same `n_ctx`. Not on the wire.
    pub unified_assumed: bool,
    /// [`Self::used`] is derived from a block-rounded ratio (vLLM).
    pub approx: bool,
}

impl KvUsage {
    /// `used / capacity` as 0..=1000, when both are known and the
    /// capacity is not zero.
    #[must_use]
    pub fn permille(&self) -> Option<u16> {
        let (used, capacity) = (self.used?, self.capacity?);
        (capacity > 0).then(|| {
            let used = u128::from(used.min(capacity));
            ((used * 1000 + u128::from(capacity) / 2) / u128::from(capacity)) as u16
        })
    }

    /// True when no field is known.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.used.is_none()
            && self.capacity.is_none()
            && self.cached.is_none()
            && self.sessions.is_none()
            && self.unified.is_none()
    }
}

/// Longest mean accepted length per speculative step kept.
pub const MAX_SPEC_LEN: f64 = 64.0;
/// Longest mean latency (TTFT, ITL, request) kept, seconds.
pub const MAX_ENGINE_LATENCY_S: f64 = 3600.0;
/// Fastest engine-measured prefill or decode speed kept, tokens per second
/// (#35). Anything above is a broken read, not a speed.
pub const MAX_ENGINE_TPS: f64 = 1_000_000.0;

/// Speculative-decoding counters since the watcher started (#31).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpecCounts {
    /// Draft rounds (`vllm:spec_decode_num_drafts`). `None` for an engine
    /// that counts drafted and accepted tokens but not rounds (Strata, #54).
    pub drafts: Option<u64>,
    /// Tokens drafted (`vllm:spec_decode_num_draft_tokens`).
    pub draft_tokens: u64,
    /// Drafted tokens accepted, never above [`Self::draft_tokens`].
    pub accepted: u64,
}

/// Engine numbers from a server's `/metrics` (#31). Every field is `None`
/// when the server does not report it. Window values cover the latest
/// metrics poll window that saw any activity, so an idle server keeps its
/// last numbers instead of reading zero.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EngineStats {
    /// Speculative acceptance (accepted / draft tokens), 0..=1000.
    pub spec_permille: Option<u16>,
    /// Mean tokens per speculative step (1 + accepted / drafts), in
    /// hundredths, 100..=[`MAX_SPEC_LEN`] × 100.
    pub spec_len_centi: Option<u16>,
    /// Speculative counters since the watcher started.
    pub spec_counts: Option<SpecCounts>,
    /// Preemptions since the watcher started; a rising count is KV pressure.
    pub preemptions: Option<u64>,
    /// The engine is asleep (`vllm:engine_sleep_state{sleep_state="awake"} 0`).
    pub sleeping: Option<bool>,
    /// Mean time to first token, microseconds.
    pub ttft_us: Option<u32>,
    /// Mean inter-token latency, microseconds.
    pub itl_us: Option<u32>,
    /// Mean end-to-end request latency, microseconds.
    pub e2e_us: Option<u32>,
    /// Prefill speed of the requests that finished in the latest metrics
    /// window with any (#35): computed (uncached) prompt tokens over prefill
    /// time, in tenths of a token per second.
    pub prefill_tps_tenths: Option<u32>,
    /// Decode speed over the same window: tokens after the first over
    /// decode time, speculative decoding included, in tenths.
    pub decode_tps_tenths: Option<u32>,
    /// Expert cache hit rate of the newest finished request, 0..=1000
    /// (Strata `requests[0].hit_rate`, #54).
    pub expert_hit_permille: Option<u16>,
    /// Share of that request's expert reads that crossed PCIe, 0..=1000
    /// (Strata `requests[0].pcie_share`, #54).
    pub pcie_share_permille: Option<u16>,
}

impl EngineStats {
    /// True when no field is known.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Seconds as microseconds, when finite and within 0..=[`MAX_ENGINE_LATENCY_S`].
#[must_use]
pub fn micros(seconds: f64) -> Option<u32> {
    (seconds.is_finite() && (0.0..=MAX_ENGINE_LATENCY_S).contains(&seconds))
        .then(|| (seconds * 1e6).round() as u32)
}

/// Tokens per second as tenths, when finite and within
/// 0..=[`MAX_ENGINE_TPS`] (#35).
#[must_use]
pub fn tps_tenths(tps: f64) -> Option<u32> {
    (tps.is_finite() && (0.0..=MAX_ENGINE_TPS).contains(&tps)).then(|| (tps * 10.0).round() as u32)
}

/// A mean accepted length as hundredths, when within 1..=[`MAX_SPEC_LEN`].
#[must_use]
pub fn centi_len(len: f64) -> Option<u16> {
    (len.is_finite() && (1.0..=MAX_SPEC_LEN).contains(&len)).then(|| (len * 100.0).round() as u16)
}

/// `spec 78 %`: the acceptance text the tty and the LCD draw (#31).
#[must_use]
pub fn spec_text(permille: u16) -> String {
    format!("spec {} %", (u32::from(permille.min(1000)) + 5) / 10)
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
    fn tps_tenths_is_bounded() {
        assert_eq!(tps_tenths(41.25), Some(413));
        assert_eq!(tps_tenths(0.0), Some(0));
        assert_eq!(tps_tenths(MAX_ENGINE_TPS), Some(10_000_000));
        assert_eq!(tps_tenths(MAX_ENGINE_TPS + 1.0), None);
        assert_eq!(tps_tenths(-0.1), None);
        assert_eq!(tps_tenths(f64::NAN), None);
    }

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

    /// #33: one long and one short display name per engine.
    #[test]
    fn display_names() {
        let names: Vec<_> = Backend::ALL
            .iter()
            .map(|kind| (kind.display_name(), kind.short_name()))
            .collect();
        assert_eq!(
            names,
            [
                ("llama.cpp", "llama.cpp"),
                ("SGLang", "sglang"),
                ("vLLM", "vllm"),
                ("Strata", "strata"),
                ("OpenAI-compatible", "openai"),
            ]
        );
        assert_eq!(Backend::default().display_name(), "llama.cpp");
        for kind in Backend::ALL {
            assert_eq!(Backend::from_display_name(kind.display_name()), Some(kind));
        }
        assert_eq!(Backend::from_display_name("sglang"), None);
        assert_eq!(Backend::from_display_name("SGLa"), None);
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

    #[test]
    fn engine_numbers_are_bounded() {
        assert_eq!(micros(0.4215), Some(421_500));
        assert_eq!(micros(3600.0), Some(3_600_000_000));
        assert_eq!(micros(3600.5), None);
        assert_eq!(micros(-0.1), None);
        assert_eq!(micros(f64::NAN), None);
        assert_eq!(centi_len(2.904), Some(290));
        assert_eq!(centi_len(1.0), Some(100));
        assert_eq!(centi_len(0.99), None);
        assert_eq!(centi_len(65.0), None);
        assert_eq!(spec_text(781), "spec 78 %");
        assert_eq!(spec_text(1000), "spec 100 %");
        assert_eq!(spec_text(4000), "spec 100 %");
        assert!(EngineStats::default().is_empty());
    }
}
