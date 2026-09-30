//! `/api/metrics/activity`: the last eight requests. Unknown fields are dropped.

use llama_core::names::sanitize;
use serde::Deserialize;

/// One activity row safe to hand to the console.
///
/// Strings are printable ASCII. A negative tok/s (`-1` means unknown on
/// llama-swap v256) is `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct ActivityRow {
    /// Activity id.
    pub id: i64,
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
    /// Request duration in milliseconds.
    pub duration_ms: Option<u64>,
    /// HTTP status of the upstream response.
    pub status: Option<u16>,
    /// llama-swap kept this request's bodies (`has_capture`, #5).
    pub captured: bool,
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
            time: sanitize(row.timestamp.as_deref().unwrap_or(""), 40),
            source: sanitize(row.src.as_deref().unwrap_or(""), 64),
            model,
            input_tokens: nonneg_u64(tokens.input_tokens),
            cached_tokens: nonneg_u64(tokens.cache_tokens),
            output_tokens: nonneg_u64(tokens.output_tokens),
            prompt_tps: nonneg_f64(tokens.prompt_per_second),
            gen_tps: nonneg_f64(tokens.tokens_per_second),
            duration_ms: nonneg_u64(row.duration_ms),
            captured: row.has_capture,
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
        assert_eq!(parse_activity(br#"{"data":[]}"#).unwrap().len(), 0);
    }
}
