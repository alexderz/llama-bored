//! Tokens of the vLLM and SGLang requests llama-swap has in flight (#80).
//!
//! Neither engine reports a request while it runs, so each in-flight row
//! gets what the engine's totals say, at the reads the poller already
//! makes, through the fresh gate (#70). Nothing is extrapolated: a row's
//! numbers change only when a read does.
//!
//! - **vLLM** (`/metrics`, read every metrics period): at each read, the
//!   deltas of `generation_tokens_total`, `prompt_tokens_total` and the
//!   cached-prompt counter since the read before are shared among the
//!   requests that were in flight at any time in between, by llama-swap's
//!   `/api/events` ([`crate::sources::events`]), or among
//!   `num_requests_running` if that is more. One request: its own tokens,
//!   exactly. More: an even split, marked approximate. vLLM adds a
//!   request's whole prompt, cached part included, to `prompt_tokens`
//!   when its prefill completes (checked against vLLM 0.30's
//!   `PromptTokenStats` and a live read: prompt = local compute + cache
//!   hit), so IN is unknown until then. A request first credited after it
//!   had already run a window (the stream connected late) only has a
//!   lower bound, marked approximate too. A window in which no request is
//!   listed stays open up to [`HOLD`], so a request the stream lists a
//!   moment after it began still gets its first tokens.
//! - **SGLang** (`GET /upstream/<id>/v1/loads?include=core`, read at the
//!   metrics cadence and only while a request is in flight): its
//!   `/metrics` gauges lag up to 40 decode steps, `/v1/loads` is computed
//!   on demand. Summed over its per-DP-rank entries, `num_used_tokens` is
//!   the KV the running requests hold and `gen_throughput` their decode
//!   tok/s; one request gets them whole, more an even share, marked
//!   approximate.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Deserializer};

use crate::metrics::MetricsSample;
use crate::sources::events::EventsView;

/// Most requests whose tokens are kept.
pub const MAX_SHARES: usize = 128;
/// How long a vLLM window with no request listed stays open (#80).
pub const HOLD: Duration = Duration::from_secs(5);
/// Most bytes of a `/v1/loads` body read.
pub const LOADS_CAP: usize = 64 * 1024;
/// Most per-DP-rank entries of `/v1/loads` summed.
const MAX_RANKS: usize = 64;

/// One request's tokens as the engine's totals give them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LiveTokens {
    /// The whole prompt (vLLM, once its prefill completed).
    pub prompt: Option<u64>,
    /// Of it, reused from the prefix cache.
    pub cached: Option<u64>,
    /// Tokens generated so far (vLLM).
    pub output: Option<u64>,
    /// KV tokens it holds now (SGLang).
    pub held: Option<u64>,
    /// Its decode tok/s (SGLang).
    pub gen_tps: Option<f64>,
    /// A share of several requests' totals, or a lower bound.
    pub approx: bool,
}

/// `/v1/loads?include=core`, summed over its DP ranks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Loads {
    /// `num_running_reqs`.
    pub running: u64,
    /// `num_used_tokens`: KV tokens the running requests hold.
    pub used_tokens: u64,
    /// `gen_throughput`: decode tok/s.
    pub gen_throughput: f64,
}

#[derive(Deserialize)]
struct LoadsDoc {
    #[serde(deserialize_with = "ranks")]
    loads: Vec<RankJson>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct RankJson {
    num_running_reqs: Option<f64>,
    num_used_tokens: Option<f64>,
    gen_throughput: Option<f64>,
}

/// `loads`: the first [`MAX_RANKS`] objects of the array.
fn ranks<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<RankJson>, D::Error> {
    let serde_json::Value::Array(items) = serde_json::Value::deserialize(deserializer)? else {
        return Err(serde::de::Error::custom("loads is not an array"));
    };
    Ok(items
        .into_iter()
        .take(MAX_RANKS)
        .filter_map(|item| serde_json::from_value(item).ok())
        .collect())
}

/// Parse a `/v1/loads` body (SGLang `entrypoints/v1_loads.py`): a JSON
/// object whose `loads` array has one entry per DP rank. `None` when the
/// body is not that, or no rank has a running count.
#[must_use]
pub fn parse_loads(body: &[u8]) -> Option<Loads> {
    let doc: LoadsDoc = serde_json::from_slice(body).ok()?;
    let count = |value: Option<f64>| {
        value
            .filter(|n| n.is_finite() && *n >= 0.0 && *n < 1e15)
            .map(|n| n as u64)
    };
    let mut out = Loads::default();
    let mut any = false;
    for rank in &doc.loads {
        if let Some(running) = count(rank.num_running_reqs) {
            any = true;
            out.running = out.running.saturating_add(running);
        }
        out.used_tokens = out
            .used_tokens
            .saturating_add(count(rank.num_used_tokens).unwrap_or(0));
        out.gen_throughput += rank
            .gen_throughput
            .filter(|tps| {
                tps.is_finite() && *tps >= 0.0 && *tps <= llama_core::backend::MAX_ENGINE_TPS
            })
            .unwrap_or(0.0);
    }
    any.then_some(out)
}

/// One vLLM read's counters.
#[derive(Clone, Copy, Debug)]
struct Counters {
    at: Instant,
    generated: u64,
    prompt: Option<u64>,
    cached: Option<u64>,
}

/// One request's running shares.
#[derive(Clone, Copy, Debug, Default)]
struct Share {
    generated: u64,
    prompt: u64,
    cached: Option<u64>,
    /// A positive prompt delta was credited: its prefill completed.
    prompted: bool,
    held: Option<u64>,
    gen_tps: Option<f64>,
    approx: bool,
}

/// The requests' shares across reads.
#[derive(Debug, Default)]
pub struct LiveBook {
    /// The last vLLM read per model id.
    last: HashMap<String, Counters>,
    /// Per `(model key, in-flight id)`.
    shares: HashMap<(String, String), Share>,
}

/// The requests of model `key` in flight at any time in `(from, to]`:
/// listed now and started by `to`, or ended after `from`. Ids and starts.
fn alive<'a>(
    view: &'a EventsView,
    key: &str,
    from: Option<Instant>,
    to: Instant,
) -> Vec<(&'a str, Instant)> {
    let mut out: Vec<(&str, Instant)> = view
        .requests
        .iter()
        .filter(|req| req.model == key && req.started <= to)
        .map(|req| (req.id.as_str(), req.started))
        .collect();
    out.extend(
        view.ended
            .iter()
            .filter(|req| {
                req.model == key && req.started <= to && from.is_none_or(|from| req.ended > from)
            })
            .map(|req| (req.id.as_str(), req.started)),
    );
    out.sort_by_key(|(_, started)| *started);
    out.dedup_by(|a, b| a.0 == b.0);
    out
}

/// `total` split over `n`, the oldest getting the remainder.
fn split(total: u64, n: u64, index: usize) -> u64 {
    let n = n.max(1);
    total / n + u64::from(index == 0) * (total % n)
}

impl LiveBook {
    /// One vLLM `/metrics` read of model `id` (activity key `key`) at
    /// `at`. The first read, or one after a counter went down (a restart),
    /// is only a baseline.
    pub fn observe_vllm(
        &mut self,
        id: &str,
        key: &str,
        at: Instant,
        sample: &MetricsSample,
        view: &EventsView,
    ) {
        let Some(generated) = sample.n_decode_total else {
            self.last.remove(id);
            return;
        };
        let now = Counters {
            at,
            generated,
            prompt: sample.prompt_total,
            cached: sample.cached_total,
        };
        let Some(before) = self.last.get(id).copied() else {
            self.last.insert(id.to_owned(), now);
            return;
        };
        if !view.connected || generated < before.generated {
            self.last.insert(id.to_owned(), now);
            return;
        }
        let delta = |new: Option<u64>, old: Option<u64>| match (new, old) {
            (Some(new), Some(old)) if new >= old => Some(new - old),
            _ => None,
        };
        let d_gen = generated - before.generated;
        let d_prompt = delta(now.prompt, before.prompt);
        let d_cached = delta(now.cached, before.cached);
        let reqs = alive(view, key, Some(before.at), at);
        if reqs.is_empty() {
            // Nobody to give them to yet: the window stays open a while,
            // so a request listed a moment after it began still gets its
            // first tokens, not a lower bound.
            if at.saturating_duration_since(before.at) > HOLD {
                self.last.insert(id.to_owned(), now);
            }
            return;
        }
        self.last.insert(id.to_owned(), now);
        let running = sample
            .requests_processing
            .filter(|n| n.is_finite() && *n >= 0.0)
            .map_or(0, |n| n.round() as u64);
        let n = (reqs.len() as u64).max(running);
        for (index, (req, started)) in reqs.iter().enumerate() {
            if !self
                .shares
                .contains_key(&(key.to_owned(), (*req).to_owned()))
                && self.shares.len() >= MAX_SHARES
            {
                continue;
            }
            let fresh = !self
                .shares
                .contains_key(&(key.to_owned(), (*req).to_owned()));
            let share = self
                .shares
                .entry((key.to_owned(), (*req).to_owned()))
                .or_default();
            // First credited, but it ran before the window: tokens are missing.
            if fresh && *started < before.at {
                share.approx = true;
            }
            if n > 1 {
                share.approx = true;
            }
            share.generated = share.generated.saturating_add(split(d_gen, n, index));
            if let Some(d) = d_prompt.filter(|d| *d > 0) {
                share.prompt = share.prompt.saturating_add(split(d, n, index));
                share.prompted = true;
            }
            if let Some(d) = d_cached {
                share.cached = Some(share.cached.unwrap_or(0).saturating_add(split(d, n, index)));
            }
        }
    }

    /// One SGLang `/v1/loads` read of model `key` at `at`: each request
    /// in flight now gets its share of the KV held and the decode rate.
    pub fn observe_loads(&mut self, key: &str, at: Instant, loads: &Loads, view: &EventsView) {
        if !view.connected {
            return;
        }
        let reqs: Vec<&str> = view
            .requests
            .iter()
            .filter(|req| req.model == key && req.started <= at)
            .map(|req| req.id.as_str())
            .collect();
        if reqs.is_empty() {
            return;
        }
        let n = (reqs.len() as u64).max(loads.running);
        for (index, req) in reqs.iter().enumerate() {
            if !self
                .shares
                .contains_key(&(key.to_owned(), (*req).to_owned()))
                && self.shares.len() >= MAX_SHARES
            {
                continue;
            }
            let share = self
                .shares
                .entry((key.to_owned(), (*req).to_owned()))
                .or_default();
            share.held = Some(split(loads.used_tokens, n, index));
            share.gen_tps = Some(loads.gen_throughput / n.max(1) as f64);
            share.approx = n > 1;
        }
    }

    /// What is known of request `id` of model `key`. `None` before any
    /// read credited it.
    #[must_use]
    pub fn tokens(&self, key: &str, id: &str) -> Option<LiveTokens> {
        let share = self.shares.get(&(key.to_owned(), id.to_owned()))?;
        let vllm = share.held.is_none();
        Some(LiveTokens {
            prompt: (vllm && share.prompted).then_some(share.prompt),
            cached: if vllm && share.prompted {
                share.cached.map(|c| c.min(share.prompt))
            } else {
                None
            },
            output: vllm.then_some(share.generated),
            held: share.held,
            gen_tps: share.gen_tps,
            approx: share.approx,
        })
    }

    /// Forget requests the stream no longer lists, in flight or ended.
    pub fn prune(&mut self, view: &EventsView) {
        self.shares.retain(|(key, id), _| {
            view.requests
                .iter()
                .any(|req| req.model == *key && req.id == *id)
                || view
                    .ended
                    .iter()
                    .any(|req| req.model == *key && req.id == *id)
        });
    }

    /// Model `id`'s process is gone or no longer read: its next read is a
    /// baseline.
    pub fn forget(&mut self, id: &str) {
        self.last.remove(id);
    }
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use super::*;
    use crate::sources::events::{EndedRequest, InflightRequest};

    fn req(id: &str, model: &str, started: Instant) -> InflightRequest {
        InflightRequest {
            id: id.to_owned(),
            model: model.to_owned(),
            started,
            started_wall: SystemTime::UNIX_EPOCH,
            resp_bytes: 0,
        }
    }

    fn read(generated: u64, prompt: u64, cached: u64, running: f64) -> MetricsSample {
        MetricsSample {
            n_decode_total: Some(generated),
            prompt_total: Some(prompt),
            cached_total: Some(cached),
            requests_processing: Some(running),
            ..MetricsSample::default()
        }
    }

    fn view(requests: Vec<InflightRequest>, ended: Vec<EndedRequest>) -> EventsView {
        EventsView {
            connected: true,
            requests,
            ended,
            failure: None,
        }
    }

    /// #80: one request running: the counter deltas since it started are
    /// exactly its own. vLLM adds the whole prompt (cached part included)
    /// when the prefill completes.
    #[test]
    fn vllm_one_request_gets_its_own_tokens_exactly() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut book = LiveBook::default();
        let one = view(vec![req("30", "qwen", at(150))], Vec::new());
        book.observe_vllm(
            "qwen",
            "qwen",
            at(100),
            &read(1_000, 50_000, 40_000, 0.0),
            &one,
        );
        assert_eq!(
            book.tokens("qwen", "30"),
            None,
            "a baseline credits nothing"
        );
        // Prefill still running: nothing counted yet.
        book.observe_vllm(
            "qwen",
            "qwen",
            at(300),
            &read(1_000, 50_000, 40_000, 1.0),
            &one,
        );
        let t = book.tokens("qwen", "30").expect("credited");
        assert_eq!(
            (t.prompt, t.cached, t.output, t.approx),
            (None, None, Some(0), false)
        );
        // Prefill done (9,000 prompt, 8,192 cached), 12 tokens out.
        book.observe_vllm(
            "qwen",
            "qwen",
            at(500),
            &read(1_012, 59_000, 48_192, 1.0),
            &one,
        );
        book.observe_vllm(
            "qwen",
            "qwen",
            at(700),
            &read(1_060, 59_000, 48_192, 1.0),
            &one,
        );
        let t = book.tokens("qwen", "30").expect("credited");
        assert_eq!(
            (t.prompt, t.cached, t.output, t.held, t.approx),
            (Some(9_000), Some(8_192), Some(60), None, false)
        );
        // Another model's request is not this one's.
        assert_eq!(book.tokens("other", "30"), None);
    }

    /// #80: two requests at once: an even split, marked approximate; one
    /// that ended inside the window still takes its share; a request the
    /// stream first lists after it had run is a lower bound.
    #[test]
    fn vllm_several_requests_share_evenly_and_are_marked() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut book = LiveBook::default();
        let two = view(
            vec![req("1", "q", at(50)), req("2", "q", at(60))],
            Vec::new(),
        );
        book.observe_vllm("q", "q", at(100), &read(0, 0, 0, 2.0), &two);
        book.observe_vllm("q", "q", at(300), &read(101, 0, 0, 2.0), &two);
        let one = book.tokens("q", "1").expect("1");
        let other = book.tokens("q", "2").expect("2");
        assert_eq!(
            (one.output, other.output),
            (Some(51), Some(50)),
            "the oldest takes the remainder"
        );
        assert!(one.approx && other.approx);
        assert!(
            one.output.unwrap() + other.output.unwrap() == 101,
            "nothing lost"
        );
        // "8" ended inside the window and "9" started in it: both ran.
        let mut book = LiveBook::default();
        let solo = view(
            vec![req("9", "q", at(250))],
            vec![EndedRequest {
                id: "8".to_owned(),
                model: "q".to_owned(),
                started: at(10),
                ended: at(200),
            }],
        );
        book.observe_vllm("q", "q", at(100), &read(0, 0, 0, 1.0), &solo);
        book.observe_vllm("q", "q", at(300), &read(90, 0, 0, 1.0), &solo);
        assert_eq!(book.tokens("q", "9").and_then(|t| t.output), Some(45));
        assert_eq!(book.tokens("q", "8").and_then(|t| t.output), Some(45));
        assert!(book.tokens("q", "8").is_some_and(|t| t.approx));
        // "5" ran before the first window it is credited in (a lower
        // bound), and the running gauge counts two requests llama-swap
        // does not list.
        let mut book = LiveBook::default();
        let late = view(vec![req("5", "q", at(0))], Vec::new());
        book.observe_vllm("q", "q", at(100), &read(0, 0, 0, 3.0), &late);
        book.observe_vllm("q", "q", at(300), &read(30, 0, 0, 3.0), &late);
        let t = book.tokens("q", "5").expect("5");
        assert_eq!(
            (t.output, t.approx),
            (Some(10), true),
            "a third of three running"
        );
        // A restart (counters down) is a new baseline, not a negative.
        book.observe_vllm("q", "q", at(500), &read(3, 0, 0, 1.0), &late);
        assert_eq!(book.tokens("q", "5").and_then(|t| t.output), Some(10));
        book.prune(&view(Vec::new(), Vec::new()));
        assert_eq!(book.tokens("q", "5"), None);
    }

    /// #80: SGLang's `/v1/loads` JSON (entrypoints/v1_loads.py,
    /// load_snapshot.py): one entry per DP rank, summed; every other
    /// section skipped.
    #[test]
    fn sglang_loads_parse_and_share() {
        let body = br#"{
            "timestamp": "2026-10-07T10:00:00+00:00", "version": "0.5.9",
            "accelerator": "INVENTED GPU", "num_accelerators": 1,
            "loads": [
                {"timestamp": 1791417180.1, "dp_rank": 0, "num_running_reqs": 1,
                 "num_waiting_reqs": 0, "num_waiting_uncached_tokens": 0,
                 "num_used_tokens": 30000, "num_total_tokens": 30000,
                 "num_active_tokens": 30000, "num_prealloc_ready_tokens": 0,
                 "max_total_num_tokens": 204000, "max_running_requests": 4,
                 "token_usage": 0.147, "gen_throughput": 41.5, "cache_hit_rate": 0.5,
                 "utilization": 0.2, "total_prefill_uncached_tokens": 123456,
                 "total_prefill_busy_us": 999, "decode_moments": [1.0, 2.0, 3.0]},
                {"dp_rank": 1, "num_running_reqs": 1, "num_used_tokens": 12000,
                 "gen_throughput": 40.0}
            ]
        }"#;
        let loads = parse_loads(body).expect("loads");
        assert_eq!(
            loads,
            Loads {
                running: 2,
                used_tokens: 42_000,
                gen_throughput: 81.5
            }
        );
        assert_eq!(parse_loads(br#"{"loads": []}"#), None);
        assert_eq!(parse_loads(br#"{"loads": {}}"#), None);
        assert_eq!(
            parse_loads(b"sglang_num_running_reqs{dp_rank=\"0\"} 1"),
            None
        );

        let t0 = Instant::now();
        let mut book = LiveBook::default();
        let one = view(vec![req("31", "sg", t0)], Vec::new());
        let single = Loads {
            running: 1,
            used_tokens: 30_000,
            gen_throughput: 41.5,
        };
        book.observe_loads("sg", t0 + Duration::from_millis(10), &single, &one);
        let t = book.tokens("sg", "31").expect("credited");
        assert_eq!(
            (t.held, t.gen_tps, t.approx),
            (Some(30_000), Some(41.5), false)
        );
        assert_eq!(
            (t.prompt, t.output),
            (None, None),
            "SGLang gives no per-request counts"
        );
        book.observe_loads("sg", t0 + Duration::from_millis(20), &loads, &one);
        let t = book.tokens("sg", "31").expect("credited");
        assert_eq!((t.held, t.approx), (Some(21_000), true));
    }

    /// #80: the stream lists a request a moment after it began: the
    /// window with nobody listed stays open, so it still gets its first
    /// tokens, exactly; a window open past [`HOLD`] starts over.
    #[test]
    fn a_request_listed_late_still_gets_its_first_tokens() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut book = LiveBook::default();
        let none = view(Vec::new(), Vec::new());
        book.observe_vllm("q", "q", at(100), &read(0, 0, 0, 0.0), &none);
        // It began at 250 ms; the 300 ms read does not list it yet.
        book.observe_vllm("q", "q", at(300), &read(5, 0, 0, 1.0), &none);
        let listed = view(vec![req("40", "q", at(250))], Vec::new());
        book.observe_vllm("q", "q", at(500), &read(12, 0, 0, 1.0), &listed);
        let t = book.tokens("q", "40").expect("credited");
        assert_eq!((t.output, t.approx), (Some(12), false));
        // Idle past the hold: the old window closes, nothing carried.
        let late = at(500) + HOLD + Duration::from_secs(1);
        book.observe_vllm("q", "q", late, &read(20, 0, 0, 0.0), &none);
        let next = view(
            vec![req("41", "q", late + Duration::from_millis(10))],
            Vec::new(),
        );
        book.observe_vllm(
            "q",
            "q",
            late + Duration::from_millis(200),
            &read(23, 0, 0, 1.0),
            &next,
        );
        assert_eq!(book.tokens("q", "41").and_then(|t| t.output), Some(3));
    }
}
