//! `/api/metrics/activity`: llama-swap's newest requests. Unknown fields are dropped.

use llama_core::backend::Backend;
use llama_core::names::sanitize;
use serde::Deserialize;

/// One activity row safe to hand to the console.
///
/// Strings are printable ASCII. A negative tok/s (`-1` means unknown on
/// llama-swap v256) is `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct ActivityRow {
    /// Activity id: llama-swap's, which starts again at 0 when llama-swap
    /// restarts. Shown, and used for its capture; never a key across reads.
    pub id: i64,
    /// llama-watch's own row number (#44): unique and increasing for the
    /// whole run, across llama-swap restarts. 0 until
    /// [`crate::recent::Recent`] numbers the row; never parsed.
    pub seq: u64,
    /// Timestamp text, sanitised.
    pub time: String,
    /// Client label, sanitised. Empty when the field is absent.
    pub source: String,
    /// Model id, sanitised. Never empty.
    pub model: String,
    /// Prompt tokens, when the value is finite and ≥ 0.
    pub input_tokens: Option<u64>,
    /// Cached prompt tokens.
    pub cached_tokens: Option<u64>,
    /// Generated tokens.
    pub output_tokens: Option<u64>,
    /// Prompt tok/s. `None` when missing, non-finite, or negative.
    pub prompt_tps: Option<f64>,
    /// Generation tok/s.
    pub gen_tps: Option<f64>,
    /// Prefill tok/s the engine measured over the window this request
    /// finished in (#35), when llama-swap gave no [`Self::prompt_tps`].
    /// Never parsed: the poller fills it.
    pub engine_prompt_tps: Option<f64>,
    /// Decode tok/s the engine measured, when there is no [`Self::gen_tps`].
    pub engine_gen_tps: Option<f64>,
    /// Request duration in milliseconds.
    pub duration_ms: Option<u64>,
    /// HTTP status of the upstream response.
    pub status: Option<u16>,
    /// Speculative draft tokens llama.cpp's timings gave (#71). `None`
    /// when llama-swap sent `-1` (no draft model) or nothing.
    pub draft_tokens: Option<u64>,
    /// Of [`Self::draft_tokens`], those accepted.
    pub draft_accepted: Option<u64>,
    /// llama-swap kept this request's bodies (`has_capture`, #5).
    pub captured: bool,
    /// The engine that served it, when the poller knew it (#82): it says
    /// what [`Self::input_tokens`] counts. Never parsed.
    pub engine: Option<Backend>,
}

/// A request's prompt (#82): the whole prompt and the part of it reused
/// from the cache, whatever the engine's llama-swap `input_tokens` meant.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PromptSplit {
    /// Every prompt token, cached ones included.
    pub whole: u64,
    /// Of [`Self::whole`], the tokens reused from the cache.
    pub cached: u64,
}

impl PromptSplit {
    /// Prompt tokens computed for this request: `whole − cached`.
    #[must_use]
    pub fn new_tokens(self) -> u64 {
        self.whole - self.cached
    }
}

/// llama-swap's `input_tokens` leaves the cached tokens out for this
/// engine (#82). llama-swap takes it from llama-server's `timings.prompt_n`
/// when the reply has `timings` (llama.cpp, and Strata, which sends them),
/// and from OpenAI `usage.prompt_tokens`, which counts them, otherwise
/// (vLLM, SGLang, any OpenAI-compatible server).
#[must_use]
pub fn input_excludes_cache(engine: Backend) -> bool {
    matches!(engine, Backend::LlamaCpp | Backend::Strata)
}

/// The one place llama-swap's `input_tokens` and `cache_tokens` become a
/// prompt (#82): RECENT's IN and CACHED, the context bar, the prompt
/// counters fed from activity rows, the in-flight handover and the reset
/// matching all read it.
///
/// With the engine unknown (a row of a model this run never saw loaded),
/// a cached count above the input can only mean the excluding form; any
/// other row reads as the OpenAI form, input = whole prompt. `None` when
/// llama-swap gave no input count.
#[must_use]
pub fn prompt_split(
    input: Option<u64>,
    cached: Option<u64>,
    engine: Option<Backend>,
) -> Option<PromptSplit> {
    let input = input?;
    let cached = cached.unwrap_or(0);
    let excludes = match engine {
        Some(engine) => input_excludes_cache(engine),
        None => cached > input,
    };
    Some(if excludes {
        PromptSplit {
            whole: input.saturating_add(cached),
            cached,
        }
    } else {
        PromptSplit {
            whole: input,
            cached: cached.min(input),
        }
    })
}

impl ActivityRow {
    /// This row's prompt by its own engine ([`prompt_split`]).
    #[must_use]
    pub fn prompt(&self) -> Option<PromptSplit> {
        self.prompt_for(None)
    }

    /// [`Self::prompt`], reading the row as `engine`'s when the poller did
    /// not stamp one.
    #[must_use]
    pub fn prompt_for(&self, engine: Option<Backend>) -> Option<PromptSplit> {
        prompt_split(
            self.input_tokens,
            self.cached_tokens,
            self.engine.or(engine),
        )
    }
}

/// Rows [`parse_activity`] keeps.
pub const ROWS: usize = 8;
/// Rows the poller parses from one page for counting, before it keeps the
/// newest few for RECENT. llama-swap v256 pages hold 25.
pub const MAX_PAGE_ROWS: usize = 100;

/// Parse a page. `None` means the body is not an activity object.
///
/// At most [`ROWS`] rows are returned, newest timestamp first.
#[must_use]
pub fn parse_activity(body: &[u8]) -> Option<Vec<ActivityRow>> {
    parse_activity_rows(body, ROWS)
}

/// [`parse_activity`] keeping at most `max_rows`. `tty.show_text = false`
/// gives RECENT the text panels' rows, so the poller keeps more.
#[must_use]
pub fn parse_activity_rows(body: &[u8], max_rows: usize) -> Option<Vec<ActivityRow>> {
    let page: ActivityPage = serde_json::from_slice(body).ok()?;
    let mut rows: Vec<ActivityRow> = page.data.into_iter().map(ActivityRow::from).collect();
    rows.sort_by(|left, right| right.time.cmp(&left.time).then(right.id.cmp(&left.id)));
    rows.truncate(max_rows);
    Some(rows)
}

impl From<ActivityJson> for ActivityRow {
    fn from(row: ActivityJson) -> Self {
        let model = model_key(row.model.as_deref().unwrap_or(""));
        let tokens = row.tokens.unwrap_or_default();
        Self {
            id: row.id,
            seq: 0,
            time: sanitize(row.timestamp.as_deref().unwrap_or(""), 40),
            source: sanitize(row.src.as_deref().unwrap_or(""), 64),
            model,
            input_tokens: nonneg_u64(tokens.input_tokens),
            cached_tokens: nonneg_u64(tokens.cache_tokens),
            output_tokens: nonneg_u64(tokens.output_tokens),
            prompt_tps: nonneg_f64(tokens.prompt_per_second),
            gen_tps: nonneg_f64(tokens.tokens_per_second),
            engine_prompt_tps: None,
            engine_gen_tps: None,
            duration_ms: nonneg_u64(row.duration_ms),
            draft_tokens: nonneg_u64(tokens.draft_tokens),
            draft_accepted: nonneg_u64(tokens.draft_acc_tokens),
            captured: row.has_capture,
            engine: None,
            status: row.resp_status_code.and_then(|code| {
                if (0.0..65536.0).contains(&code) && code.fract() == 0.0 {
                    Some(code as u16)
                } else {
                    None
                }
            }),
        }
    }
}

const MODEL_CHARS: usize = 32;

/// A model id as [`ActivityRow::model`] holds it, for matching rows to a
/// llama-swap model.
#[must_use]
pub fn model_key(id: &str) -> String {
    let model = sanitize(id, MODEL_CHARS);
    if model.is_empty() {
        "model".to_owned()
    } else {
        model
    }
}

fn nonneg_u64(value: Option<f64>) -> Option<u64> {
    let value = value?;
    if !value.is_finite() || value < 0.0 || value >= u64::MAX as f64 {
        None
    } else {
        Some(value as u64)
    }
}

fn nonneg_f64(value: Option<f64>) -> Option<f64> {
    let value = value?;
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

#[derive(Debug, Deserialize)]
struct ActivityPage {
    #[serde(default)]
    data: Vec<ActivityJson>,
}

#[derive(Debug, Deserialize)]
struct ActivityJson {
    #[serde(default)]
    id: i64,
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    src: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    tokens: Option<TokensJson>,
    #[serde(default)]
    duration_ms: Option<f64>,
    #[serde(default)]
    resp_status_code: Option<f64>,
    #[serde(default)]
    has_capture: bool,
}

#[derive(Debug, Default, Deserialize)]
struct TokensJson {
    #[serde(default)]
    cache_tokens: Option<f64>,
    #[serde(default)]
    input_tokens: Option<f64>,
    #[serde(default)]
    output_tokens: Option<f64>,
    #[serde(default)]
    prompt_per_second: Option<f64>,
    #[serde(default)]
    tokens_per_second: Option<f64>,
    #[serde(default)]
    draft_tokens: Option<f64>,
    #[serde(default)]
    draft_acc_tokens: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_keeps_the_last_eight_and_ignores_unknown_fields() {
        let body = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/llama/activity-v256.json"
        ));
        let rows = parse_activity(body).expect("activity page");
        assert_eq!(rows.len(), 8);
        assert_eq!(
            rows.iter().map(|row| row.id).collect::<Vec<_>>(),
            vec![20, 19, 18, 17, 16, 15, 14, 13]
        );
        assert_eq!(rows[0].time, "2026-09-24T01:47:25Z");
        let raw = std::str::from_utf8(body).expect("utf-8");
        assert!(
            !raw.contains("raise_exception"),
            "template error text remains"
        );
        // The capture is scrubbed: every client is a documentation address.
        for src in raw.split("\"src\":").skip(1) {
            let value = src.trim_start().trim_start_matches('"');
            assert!(
                value.starts_with("ip:192.0.2.") || value.starts_with("ip:2001:db8:"),
                "non-documentation address in fixture: {}",
                &value[..value.len().min(40)]
            );
        }
        assert_eq!(rows[0].source, "ip:192.0.2.1");
        assert_eq!(rows[0].model, "devstral-small-2-24b");
        assert_eq!(rows[0].input_tokens, Some(69));
        assert_eq!(rows[0].cached_tokens, Some(553));
        assert_eq!(rows[0].output_tokens, Some(12));
        assert!((rows[0].prompt_tps.unwrap() - 1193.6477182299418).abs() < 1e-6);
        assert!((rows[0].gen_tps.unwrap() - 50.46450280995528).abs() < 1e-6);
        assert_eq!(rows[0].duration_ms, Some(280));
        assert_eq!(rows[0].status, Some(200));
        assert_eq!(rows[3].status, Some(500));
        let blob = format!("{rows:?}");
        assert!(!blob.contains("raise_exception"), "{blob}");
        assert!(!blob.contains("error_msg"), "{blob}");
        assert!(!blob.contains("has_capture"), "{blob}");

        let unknown = br#"{
            "data": [{
                "id": 3,
                "timestamp": "2026-09-24T00:00:00Z",
                "model": "\u4e09\u4e94",
                "src": "ip:\u001b[2J",
                "tokens": {
                    "input_tokens": 4,
                    "output_tokens": -1,
                    "cache_tokens": 1,
                    "prompt_per_second": -1,
                    "tokens_per_second": 9.5,
                    "draft_tokens": -1
                },
                "duration_ms": 10,
                "resp_status_code": 200,
                "error_msg": "SECRET_ACTIVITY_TEXT",
                "req_path": "/v1/chat/completions"
            }],
            "page": 1,
            "limit": 25,
            "total": 1
        }"#;
        let rows = parse_activity(unknown).expect("synthetic");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model, "model");
        assert_eq!(rows[0].output_tokens, None);
        assert_eq!(rows[0].prompt_tps, None);
        assert_eq!(rows[0].gen_tps, Some(9.5));
        assert!(!rows[0].source.chars().any(|ch| ch == '\u{1b}'));
        assert!(!format!("{rows:?}").contains("SECRET_ACTIVITY_TEXT"));

        assert!(parse_activity(b"not-json").is_none());
        assert_eq!(rows[0].engine, None, "the poller stamps the engine");
        assert_eq!(parse_activity(br#"{"data":[]}"#).unwrap().len(), 0);
    }

    /// #82: one helper reads `input_tokens` per engine. llama.cpp and
    /// Strata (llama-server `timings`) leave the cache out; vLLM, SGLang
    /// and OpenAI `usage` count it.
    #[test]
    fn prompt_split_reads_input_tokens_per_engine() {
        let split = |input, cached, engine| prompt_split(Some(input), cached, engine);
        // Seen live on Strata: 21 new, 102,231 reused.
        for engine in [Backend::LlamaCpp, Backend::Strata] {
            assert_eq!(
                split(21, Some(102_231), Some(engine)),
                Some(PromptSplit {
                    whole: 102_252,
                    cached: 102_231
                })
            );
            assert_eq!(
                split(500, Some(100), Some(engine)).map(PromptSplit::new_tokens),
                Some(500)
            );
        }
        for engine in [Backend::Vllm, Backend::SgLang, Backend::OpenAi] {
            let got = split(9_000, Some(8_192), Some(engine)).expect("split");
            assert_eq!(
                (got.whole, got.cached, got.new_tokens()),
                (9_000, 8_192, 808)
            );
            // A cached count past the prompt is clamped, never negative.
            let odd = split(10, Some(50), Some(engine)).expect("split");
            assert_eq!((odd.whole, odd.cached), (10, 10));
        }
        // No cached count: everything is new.
        assert_eq!(
            split(64, None, Some(Backend::LlamaCpp)),
            Some(PromptSplit {
                whole: 64,
                cached: 0
            })
        );
        // Engine unknown: only a cached count above the input proves the
        // excluding form.
        assert_eq!(split(69, Some(553), None).map(|p| p.whole), Some(622));
        assert_eq!(split(900, Some(300), None).map(|p| p.whole), Some(900));
        assert_eq!(prompt_split(None, Some(5), Some(Backend::Vllm)), None);

        // A row stamped by the poller wins over the caller's guess.
        let page = br#"{"data":[{"id":1,"timestamp":"t","model":"m","tokens":{"input_tokens":69,"cache_tokens":553}}]}"#;
        let mut row = parse_activity(page).expect("page").remove(0);
        assert_eq!(
            row.prompt_for(Some(Backend::Vllm)).map(|p| p.whole),
            Some(69)
        );
        row.engine = Some(Backend::LlamaCpp);
        assert_eq!(
            row.prompt_for(Some(Backend::Vllm)).map(|p| p.whole),
            Some(622)
        );
        assert_eq!(row.prompt().map(|p| p.cached), Some(553));
    }
}
