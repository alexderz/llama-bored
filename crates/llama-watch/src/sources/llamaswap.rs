//! llama-swap `GET /running`.
//!
//! The reply is untrusted. Serde structs name only `model`, `name`, and
//! `state`, plus a [`ModelDetail`] that the launch command is parsed into
//! while the body is decoded ([`super::cmdline`]). The command string itself
//! and the upstream proxy URL are never stored or returned. Invalid UTF-8 in the body is
//! [`RunningStatus::Down`], because the body is parsed as JSON and never lossily
//! decoded. This module does not log; the collector logs a failure once per
//! state change.

use std::collections::HashMap;
use std::time::Duration;

use llama_core::detail::{MAX_FULL_NAME_CHARS, ModelDetail};
use llama_core::names::{sanitize, sanitize_wire};
use serde::{Deserialize, Deserializer};

/// One model from a single `/running` body: raw id, display name, and state.
///
/// `id` is the upstream path segment. `name` is the sanitised label. They
/// come from the same array element, so a later response cannot re-pair them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunningModel {
    /// Raw `model` field. Empty when the field is absent.
    pub id: String,
    /// Alias, else `name`, else the model id, after the wire sanitiser.
    pub name: String,
    /// Upstream state word. Empty when the field is absent.
    pub state: String,
    /// The same label as [`Self::name`], sanitised to [`MAX_FULL_NAME_CHARS`].
    pub full_name: String,
    /// Tuning detail from the launch command. `None` when there is no command.
    pub detail: Option<ModelDetail>,
}

/// Whether llama-swap is unreachable, up with nothing loaded, or serving models.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunningStatus {
    /// Client or parse failure. The label is static and never bytes from the reply.
    Down(&'static str),
    /// HTTP 200 and an empty `running` array.
    Idle,
    /// HTTP 200 and one or more `running` entries.
    Loaded,
}

/// One read of `{url}/running`.
///
/// `models` is empty unless `ai` is [`RunningStatus::Loaded`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Reading {
    pub ai: RunningStatus,
    pub models: Vec<RunningModel>,
}

/// Agent for llama-swap. No proxy, no redirects, status codes are bodies.
///
/// The poller keeps one of these for every tap. Callers that are not the
/// poller build their own and pass it to [`read_with`]. Timeouts are per call.
#[must_use]
pub fn new_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .proxy(None)
        .max_redirects(0)
        .http_status_as_error(false)
        .build()
        .into()
}

/// `GET {url}/running` on `agent` and classify that one body.
///
/// `aliases` is keyed by the raw `model` field. `timeout` is the deadline for
/// this call. Display names go through [`sanitize_wire`](llama_core::names::sanitize_wire).
/// Does not log. On failure, [`RunningStatus::Down`] carries one of
/// `"timeout"`, `"connection refused"`, `"request failed"`, `"http status"`,
/// `"oversized body"`, or `"malformed json"`. Invalid UTF-8 in the body is
/// `"malformed json"`.
#[must_use]
pub fn read_with(
    agent: &ureq::Agent,
    url: &str,
    timeout: Duration,
    aliases: &HashMap<String, String>,
) -> Reading {
    let endpoint = running_endpoint(url);
    let mut response = match agent
        .get(&endpoint)
        .config()
        .timeout_global(Some(timeout))
        .build()
        .call()
    {
        Ok(response) => response,
        Err(err) => return down_reading(transport_reason(&err)),
    };

    let status = response.status().as_u16();
    if status != 200 {
        return down_reading("http status");
    }

    let bytes = match response
        .body_mut()
        .with_config()
        // ureq errors once this many bytes have been consumed, so a body of
        // exactly 65_536 is rejected along with anything larger.
        .limit(65_536)
        .read_to_vec()
    {
        Ok(bytes) => bytes,
        Err(err) => return down_reading(body_reason(&err)),
    };

    match parse_running(&bytes, aliases) {
        Ok(reading) => reading,
        Err(()) => down_reading("malformed json"),
    }
}

fn down_reading(reason: &'static str) -> Reading {
    Reading {
        ai: RunningStatus::Down(reason),
        models: Vec::new(),
    }
}

fn running_endpoint(url: &str) -> String {
    format!("{}/running", url.trim_end_matches('/'))
}

fn transport_reason(err: &ureq::Error) -> &'static str {
    match err {
        ureq::Error::Timeout(_) => "timeout",
        ureq::Error::ConnectionFailed => "connection refused",
        ureq::Error::Io(io) => io_reason(io),
        _ => "request failed",
    }
}

fn body_reason(err: &ureq::Error) -> &'static str {
    match err {
        ureq::Error::Timeout(_) => "timeout",
        ureq::Error::BodyExceedsLimit(_) => "oversized body",
        ureq::Error::Io(io) if io.kind() == std::io::ErrorKind::TimedOut => "timeout",
        _ => "request failed",
    }
}

fn io_reason(err: &std::io::Error) -> &'static str {
    match err.kind() {
        std::io::ErrorKind::ConnectionRefused => "connection refused",
        std::io::ErrorKind::TimedOut => "timeout",
        _ => "request failed",
    }
}

fn parse_running(bytes: &[u8], aliases: &HashMap<String, String>) -> Result<Reading, ()> {
    let parsed: RunningResponse = serde_json::from_slice(bytes).map_err(|_| ())?;
    if parsed.running.is_empty() {
        return Ok(Reading {
            ai: RunningStatus::Idle,
            models: Vec::new(),
        });
    }
    let models = parsed
        .running
        .iter()
        .take(MAX_MODELS)
        .map(|entry| RunningModel {
            id: entry.model.clone().unwrap_or_default(),
            name: display_name(entry, aliases, sanitize_wire),
            state: entry.state.clone().unwrap_or_default(),
            full_name: display_name(entry, aliases, |raw| sanitize(raw, MAX_FULL_NAME_CHARS)),
            detail: entry.detail.clone(),
        })
        .collect();
    Ok(Reading {
        ai: RunningStatus::Loaded,
        models,
    })
}

const MAX_MODELS: usize = 8;

/// Alias for the raw model id, else `name`, else `model`. If that sanitises
/// to empty, the sanitised model id is used instead.
fn display_name(
    entry: &RunningEntry,
    aliases: &HashMap<String, String>,
    clean: impl Fn(&str) -> String,
) -> String {
    let model_raw = entry.model.as_deref().unwrap_or("");
    let chosen = choose_label(entry, aliases);
    let sanitised = clean(chosen);
    if sanitised.is_empty() && chosen != model_raw {
        clean(model_raw)
    } else {
        sanitised
    }
}

fn choose_label<'a>(entry: &'a RunningEntry, aliases: &'a HashMap<String, String>) -> &'a str {
    if let Some(model) = entry.model.as_deref()
        && let Some(alias) = aliases.get(model)
    {
        return alias.as_str();
    }
    match entry.name.as_deref() {
        Some(name) if !name.is_empty() => name,
        _ => entry.model.as_deref().unwrap_or(""),
    }
}

#[derive(Debug, Deserialize)]
struct RunningResponse {
    running: Vec<RunningEntry>,
}

/// Only `model`, `name`, `state`, and the detail parsed from the launch
/// command. Other members, including the upstream proxy URL, are skipped.
/// The command is borrowed by [`detail_from_command`] and never kept, so it
/// cannot reach a log or the screen.
#[derive(Debug, Deserialize)]
struct RunningEntry {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    state: Option<String>,
    #[serde(default, rename = "cmd", deserialize_with = "detail_from_command")]
    detail: Option<ModelDetail>,
}

/// Parse the command in place. A non-string command gives no detail.
fn detail_from_command<'de, D>(deserializer: D) -> Result<Option<ModelDetail>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(value.as_str().map(super::cmdline::parse))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_names_use_the_wire_width_not_config() {
        let reading = parse_running(
            br#"{"running":[{"model":"abcdefghijklmnopqrstuvwxyz","state":"ready"}]}"#,
            &HashMap::new(),
        )
        .expect("parse");
        assert_eq!(reading.models[0].name, "abcdefghijk…");
        assert_eq!(reading.models[0].name.chars().count(), 12);
    }

    #[test]
    fn real_running_body_gives_full_name_and_detail() {
        let body = br#"{"running":[{"model":"bonsai2-27b","state":"ready","cmd":"/models/prism/llama-server --host 127.0.0.1 --port 5800 -fa on --metrics\n-ctk q8_0 -ctv q8_0\n-m /models/llm/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf\n-ngl 999 -c 262144\n","proxy":"http://localhost:5800","ttl":900,"name":"Ternary Bonsai 2 27B","description":"Qwen3.8-27B compressed to ternary weights"}]}"#;
        let reading = parse_running(body, &HashMap::new()).expect("parse");
        let model = &reading.models[0];
        assert_eq!(model.name, "Ternary Bon…");
        assert_eq!(model.full_name, "Ternary Bonsai 2 27B");
        let detail = model.detail.as_ref().expect("detail");
        assert_eq!(detail.ctx, Some(262_144));
        assert_eq!(detail.kv_k.as_deref(), Some("q8_0"));
        assert_eq!(detail.kv_v.as_deref(), Some("q8_0"));
        assert_eq!(detail.quant.as_deref(), Some("PTQ1_0"));
        assert_eq!(detail.fa, Some(true));
        let debug = format!("{reading:?}");
        for leak in [
            "/models",
            "localhost",
            "5800",
            "llama-server",
            "ternary weights",
        ] {
            assert!(!debug.contains(leak), "{leak} leaked: {debug}");
        }

        let no_cmd = parse_running(
            br#"{"running":[{"model":"m","state":"ready","cmd":7}]}"#,
            &HashMap::new(),
        )
        .expect("a non-string command is not an error");
        assert_eq!(no_cmd.models[0].detail, None);
    }

    #[test]
    fn running_endpoint_appends_path_and_trims_slash() {
        assert_eq!(
            running_endpoint("http://127.0.0.1:8080"),
            "http://127.0.0.1:8080/running"
        );
        assert_eq!(
            running_endpoint("http://127.0.0.1:8080/"),
            "http://127.0.0.1:8080/running"
        );
        assert_eq!(
            running_endpoint("http://127.0.0.1:8080///"),
            "http://127.0.0.1:8080/running"
        );
    }

    #[test]
    fn parse_empty_running_is_idle() {
        let reading = parse_running(br#"{"running":[]}"#, &HashMap::new()).expect("parse");
        assert_eq!(reading.ai, RunningStatus::Idle);
        assert!(reading.models.is_empty());
    }

    #[test]
    fn parse_uses_alias_then_name_then_model() {
        let mut aliases = HashMap::new();
        aliases.insert("qwen3.6-35b-a3b".to_owned(), "Qwen   35B 🔥".to_owned());
        aliases.insert("blank-alias".to_owned(), String::new());

        let aliased = parse_running(
            br#"{"running":[{"model":"qwen3.6-35b-a3b","name":"raw-name","state":"ready","cmd":"CANARY_CMD_9f3a2c7e","proxy":"CANARY_PROXY_1b6d4e8a"}]}"#,
            &aliases,
        )
        .expect("parse");
        assert_eq!(aliased.ai, RunningStatus::Loaded);
        assert_eq!(aliased.models[0].id, "qwen3.6-35b-a3b");
        assert_eq!(aliased.models[0].name, "Qwen 35B");
        assert_eq!(aliased.models[0].state, "ready");
        assert!(!format!("{aliased:?}").contains("CANARY_CMD_9f3a2c7e"));
        assert!(!format!("{aliased:?}").contains("CANARY_PROXY_1b6d4e8a"));
        assert!(!format!("{aliased:?}").contains("raw-name"));

        let named = parse_running(
            br#"{"running":[{"model":"other","name":"DeepSeek","state":"starting"}]}"#,
            &HashMap::new(),
        )
        .expect("parse");
        assert_eq!(named.models[0].name, "DeepSeek");
        assert_eq!(named.models[0].state, "starting");

        let modeled = parse_running(
            br#"{"running":[{"model":"plain-model","state":"ready"}]}"#,
            &HashMap::new(),
        )
        .expect("parse");
        assert_eq!(modeled.models[0].name, "plain-model");

        let empty_name = parse_running(
            br#"{"running":[{"model":"fallback","name":"","state":"ready"}]}"#,
            &HashMap::new(),
        )
        .expect("parse");
        assert_eq!(empty_name.models[0].name, "fallback");

        // A name that sanitises to empty falls back to the sanitised model id.
        let blank_name = parse_running(
            br#"{"running":[{"model":"fallback","name":"   ","state":"ready"}]}"#,
            &HashMap::new(),
        )
        .expect("parse");
        assert_eq!(blank_name.models[0].name, "fallback");

        let unicode_name = parse_running(
            r#"{"running":[{"model":"qwen-id","name":"三五","state":"ready"}]}"#.as_bytes(),
            &HashMap::new(),
        )
        .expect("parse");
        assert_eq!(unicode_name.models[0].name, "qwen-id");

        // An alias that sanitises to empty likewise falls back to the model id.
        let blank_alias = parse_running(
            br#"{"running":[{"model":"blank-alias","name":"Visible","state":"ready"}]}"#,
            &aliases,
        )
        .expect("parse");
        assert_eq!(blank_alias.models[0].name, "blank-alias");
    }

    #[test]
    fn parse_keeps_at_most_eight_models() {
        let mut entries = String::from(r#"{"running":["#);
        for index in 0..20 {
            if index > 0 {
                entries.push(',');
            }
            entries.push_str(&format!(r#"{{"model":"model-{index}","state":"ready"}}"#));
        }
        entries.push_str("]}");
        let reading = parse_running(entries.as_bytes(), &HashMap::new()).expect("parse");
        assert_eq!(reading.ai, RunningStatus::Loaded);
        assert_eq!(reading.models.len(), 8);
        let names: Vec<_> = reading
            .models
            .iter()
            .map(|model| model.name.as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                "model-0", "model-1", "model-2", "model-3", "model-4", "model-5", "model-6",
                "model-7",
            ]
        );
    }

    #[test]
    fn parse_errors_are_down() {
        let cases = [
            br#"{"running":"#.as_slice(),
            br#"{"nope":[]}"#.as_slice(),
            br#"{"running":{"model":"x"}}"#.as_slice(),
            br#"{"running":"no"}"#.as_slice(),
            br#"{"running":null}"#.as_slice(),
            br#"{"running":[1]}"#.as_slice(),
            &[0xff, 0xfe],
        ];
        for bytes in cases {
            assert!(
                parse_running(bytes, &HashMap::new()).is_err(),
                "{}",
                String::from_utf8_lossy(bytes)
            );
        }
    }

    #[test]
    fn serde_structs_do_not_declare_cmd_or_proxy() {
        let src = include_str!("llamaswap.rs");
        // Split so this test's own source does not contain the needle.
        let cmd_field = concat!("cm", "d:");
        let proxy_field = concat!("pro", "xy:");
        assert!(
            !src.contains(cmd_field),
            "launch command must not be a declared field"
        );
        assert!(
            !src.contains(proxy_field),
            "proxy URL must not be a declared field"
        );
    }
}
