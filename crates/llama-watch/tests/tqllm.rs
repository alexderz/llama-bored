//! #180: tqLLM's monitor snapshot, its strict parse, and the counters,
//! slots and row fills across reads. Invented values only.

use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

use llama_watch::metrics::{
    TqError, TqPhase, TqRequest, TqSnapshot, depth_ok, is_strata_json, is_tqllm_json, parse_tqllm,
    parse_tqllm_health,
};
use llama_watch::tqllm::{Fill, RowClaim, TqllmBook};

fn fixture() -> String {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/llama/tqllm-snapshot.json");
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

/// The fixture with one member of request `id` replaced (`"key": value`).
fn with_field(id: u64, key: &str, value: &str) -> String {
    let mut doc: serde_json::Value = serde_json::from_str(&fixture()).expect("fixture json");
    let requests = doc["requests"].as_array_mut().expect("requests");
    let req = requests
        .iter_mut()
        .find(|req| req["id"] == id)
        .expect("request");
    req[key] = serde_json::from_str(value).expect("value json");
    doc.to_string()
}

#[test]
fn the_real_shape_parses_and_keeps_numbers_only() {
    let body = fixture();
    assert!(is_tqllm_json(&body));
    assert!(!is_strata_json(&body));
    let snap = parse_tqllm(&body).expect("snapshot");
    assert_eq!(snap.instance, "4242000000000000007");
    assert_eq!(snap.context_per_slot, Some(32_768));
    assert_eq!(snap.slots, Some(2));
    assert_eq!(snap.build.as_deref(), Some("0a1b2c3"));
    let phases: Vec<_> = snap.requests.iter().map(|req| req.phase).collect();
    assert_eq!(
        phases,
        [
            Some(TqPhase::Complete),
            Some(TqPhase::Complete),
            Some(TqPhase::Failed),
            Some(TqPhase::Decode),
            Some(TqPhase::Prefill),
            Some(TqPhase::Queued),
        ]
    );
    let prefill = &snap.requests[4];
    assert_eq!(prefill.id, 5);
    assert_eq!(prefill.prompt_tokens, Some(6_000));
    assert_eq!(prefill.prefill_processed, Some(2_048));
    assert_eq!(prefill.prefill_cached, Some(1_024));
    assert_eq!(prefill.output_tokens, None, "null stays unknown");
    assert_eq!(prefill.decode_tps, None);
    assert_eq!(prefill.prompt_read(), 2_048);
    assert_eq!(prefill.cached_read(), 1_024);
    let queued = &snap.requests[5];
    assert_eq!((queued.prompt_tokens, queued.prompt_read()), (None, 0));
    let failed = &snap.requests[2];
    assert_eq!(failed.output_tokens, None);
    // The identity fields and the hardware are never read.
    let kept = format!("{snap:?}");
    for leak in [
        "invented-host",
        "0000:00:00.0",
        "invented-moe-alias",
        "Invented GPU",
        "future",
    ] {
        assert!(!kept.contains(leak), "{leak}: {kept}");
    }
}

#[test]
fn another_schema_is_no_snapshot() {
    let body = fixture();
    for (from, to, want) in [
        (
            r#""schema_version": 1"#,
            r#""schema_version": 2"#,
            TqError::Schema(Some(2)),
        ),
        (
            r#""schema_version": 1"#,
            r#""schema_version": 0"#,
            TqError::Schema(Some(0)),
        ),
        (
            r#""schema_version": 1"#,
            r#""schema_version": "1""#,
            TqError::Schema(None),
        ),
        (
            r#""schema_version": 1"#,
            r#""schema_version": 1.5"#,
            TqError::Schema(None),
        ),
        (
            r#""schema_version": 1"#,
            r#""schema": 1"#,
            TqError::Malformed,
        ),
    ] {
        let changed = body.replacen(from, to, 1);
        assert_ne!(changed, body, "{to}");
        assert_eq!(parse_tqllm(&changed), Err(want), "{to}");
        assert!(!is_tqllm_json(&changed), "{to}");
    }
}

#[test]
fn the_parse_is_strict_on_the_fields_it_uses() {
    for (id, key, value) in [
        (1, "prompt_tokens", r#""20""#),
        (1, "prompt_tokens", "-1"),
        (1, "prompt_tokens", "20.5"),
        (1, "output_tokens", "true"),
        (1, "elapsed_ms", "null"),
        (1, "elapsed_ms", "-3"),
        (1, "decode_tokens_per_second", r#""fast""#),
        (1, "id", "-1"),
        (1, "id", r#""1""#),
        (1, "phase", "7"),
    ] {
        assert_eq!(
            parse_tqllm(&with_field(id, key, value)),
            Err(TqError::Malformed),
            "{key}: {value}"
        );
    }
    // A whole number printed as a float is a count.
    let snap = parse_tqllm(&with_field(1, "prompt_tokens", "20.0")).expect("20.0");
    assert_eq!(snap.requests[0].prompt_tokens, Some(20));
    // A missing optional count is unknown, like null.
    let mut doc: serde_json::Value = serde_json::from_str(&fixture()).expect("json");
    doc["requests"][0]
        .as_object_mut()
        .expect("object")
        .remove("drafted");
    let snap = parse_tqllm(&doc.to_string()).expect("no drafted");
    assert_eq!(snap.requests[0].drafted, None);
    // The engine's numbers and the instance must be there and be right.
    for (path, value) in [
        ("context_per_slot", "null"),
        ("slots", r#""2""#),
        ("slots", "-2"),
    ] {
        let mut doc: serde_json::Value = serde_json::from_str(&fixture()).expect("json");
        doc["engine"][path] = serde_json::from_str(value).expect("value");
        assert_eq!(
            parse_tqllm(&doc.to_string()),
            Err(TqError::Malformed),
            "{path}"
        );
    }
    for instance in [
        serde_json::json!(""),
        serde_json::json!(42),
        serde_json::json!("x".repeat(65)),
        serde_json::json!("a b"),
    ] {
        let mut doc: serde_json::Value = serde_json::from_str(&fixture()).expect("json");
        doc["instance_id"] = instance.clone();
        assert_eq!(
            parse_tqllm(&doc.to_string()),
            Err(TqError::Malformed),
            "{instance}"
        );
    }
    // An unknown phase word: that request is left out, the rest is read.
    let snap = parse_tqllm(&with_field(1, "phase", r#""paused""#)).expect("unknown phase");
    assert_eq!(snap.requests.len(), 5);
    assert!(snap.requests.iter().all(|req| req.id != 1));
    assert_eq!(parse_tqllm("not json"), Err(TqError::Malformed));
    assert_eq!(parse_tqllm("[]"), Err(TqError::Malformed));
}

#[test]
fn json_depth_is_capped() {
    let mut deep = String::new();
    for _ in 0..9 {
        deep.push('[');
    }
    for _ in 0..9 {
        deep.push(']');
    }
    let body = fixture().replacen(
        r#""a_future_member": "ignored""#,
        &format!(r#""a_future_member": {deep}"#),
        1,
    );
    assert_eq!(parse_tqllm(&body), Err(TqError::Depth));
    assert!(!is_tqllm_json(&body));
    assert!(
        depth_ok(r#"{"a":"[[[[[[[[[[[[[[["}"#, 2),
        "brackets in a string"
    );
    assert!(depth_ok(r#"{"a":"\"[[[[[[[[[[["}"#, 2), "an escaped quote");
    assert!(!depth_ok("[[[]]]", 2));
    assert!(depth_ok("[[]]", 2));
    assert_eq!(
        parse_tqllm_health(&format!(r#"{{"fit":"x","deep":{deep}}}"#)),
        Err(TqError::Depth)
    );
}

#[test]
fn health_gives_the_fit_text() {
    let fit = parse_tqllm_health(
        r#"{"status":"ok","engine":"0a1b2c3","fit":"tqserve: fit: to fit 1 slot(s) x 131072 in GPU memory:\n KV cache q4/tcq2 (asked bf16)\u001b[2J"}"#,
    );
    assert_eq!(
        fit,
        Ok(Some(
            "tqserve: fit: to fit 1 slot(s) x 131072 in GPU memory: KV cache q4/tcq2 (asked bf16)[2J"
                .to_owned()
        ))
    );
    assert_eq!(parse_tqllm_health(r#"{"status":"ok","fit":""}"#), Ok(None));
    assert_eq!(
        parse_tqllm_health(r#"{"status":"ok","fit":"  "}"#),
        Ok(None)
    );
    assert_eq!(parse_tqllm_health(r#"{"status":"ok"}"#), Ok(None));
    assert_eq!(parse_tqllm_health(r#"{"fit":7}"#), Err(TqError::Malformed));
    assert_eq!(parse_tqllm_health("ok"), Err(TqError::Malformed));
    let long = parse_tqllm_health(&format!(r#"{{"fit":"{}"}}"#, "y".repeat(5000)))
        .expect("long")
        .expect("text");
    assert_eq!(long.chars().count(), 1024, "cut, not dropped");
}

// ---- the book ----------------------------------------------------------------

const BASE: u64 = 1_800_000_000; // 2027-01-15T08:00:00Z

fn wall(secs: f64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs_f64(BASE as f64 + secs)
}

fn stamp(secs: u64) -> String {
    // 2027-01-15T08:00:00Z plus whole seconds, under a minute.
    assert!(secs < 60);
    format!("2027-01-15T08:00:{secs:02}Z")
}

fn snap(instance: &str, slots: u64, requests: Vec<TqRequest>) -> TqSnapshot {
    TqSnapshot {
        instance: instance.to_owned(),
        context_per_slot: Some(16_384),
        slots: Some(slots),
        build: Some("0a1b2c3".to_owned()),
        requests,
    }
}

fn req(id: u64, phase: TqPhase) -> TqRequest {
    TqRequest {
        id,
        phase: Some(phase),
        elapsed_ms: 100.0,
        ..TqRequest::default()
    }
}

fn prefill(id: u64, prompt: u64, read: u64, cached: u64) -> TqRequest {
    TqRequest {
        prompt_tokens: Some(prompt),
        prefill_processed: Some(read),
        prefill_cached: Some(cached),
        ..req(id, TqPhase::Prefill)
    }
}

fn decode(id: u64, prompt: u64, cached: u64, output: u64) -> TqRequest {
    TqRequest {
        prompt_tokens: Some(prompt),
        prefill_processed: Some(prompt),
        prefill_cached: Some(cached),
        output_tokens: Some(output),
        context_tokens: Some(prompt + output),
        ..req(id, TqPhase::Decode)
    }
}

fn complete(id: u64, prompt: u64, cached: u64, output: u64) -> TqRequest {
    TqRequest {
        phase: Some(TqPhase::Complete),
        prefill_ms: Some(500.0),
        decode_ms: Some(1_000.0),
        prefill_tps: Some(800.0),
        decode_tps: Some(42.5),
        ..decode(id, prompt, cached, output)
    }
}

struct Reads {
    book: TqllmBook,
    at: Instant,
    t: f64,
}

impl Reads {
    fn new() -> Self {
        Self {
            book: TqllmBook::default(),
            at: Instant::now(),
            t: 0.0,
        }
    }

    /// One read of model `m` one second after the last.
    fn read(&mut self, snap: &TqSnapshot) -> llama_watch::tqllm::Observed {
        self.t += 1.0;
        self.at += Duration::from_secs(1);
        self.book
            .observe("m", snap, self.at, wall(self.t - 0.01), wall(self.t))
    }

    fn totals(observed: &llama_watch::tqllm::Observed) -> (u64, u64, u64, u64, u64) {
        let sample = &observed.sample;
        let spec = sample.spec.expect("spec counts");
        (
            sample.prompt_total.expect("prompt"),
            sample.cached_total.expect("cached"),
            sample.n_decode_total.expect("generated"),
            spec.draft_tokens,
            spec.accepted,
        )
    }
}

#[test]
fn counters_follow_each_request_once() {
    let mut reads = Reads::new();
    // Read 1: one request in prefill, a tenth read, half of that cached.
    let one = reads.read(&snap("i1", 2, vec![prefill(1, 1_000, 100, 50)]));
    assert_eq!(Reads::totals(&one), (100, 50, 0, 0, 0));
    assert_eq!(one.sample.requests_processing, Some(1.0));
    assert_eq!(one.sample.queued, Some(0.0));
    assert!(!one.restarted && one.gap == 0);
    // Read 2: it decodes (the whole prompt is read), a second one queues.
    let two = reads.read(&snap(
        "i1",
        2,
        vec![decode(1, 1_000, 50, 20), req(2, TqPhase::Queued)],
    ));
    assert_eq!(Reads::totals(&two), (1_000, 50, 20, 0, 0));
    assert_eq!(
        (two.sample.requests_processing, two.sample.queued),
        (Some(1.0), Some(1.0))
    );
    // Read 3: the first finishes with its drafts, the second reads.
    let mut done = complete(1, 1_000, 50, 40);
    done.drafted = Some(30);
    done.accepted = Some(20);
    let three = reads.read(&snap("i1", 2, vec![done.clone(), prefill(2, 900, 300, 0)]));
    assert_eq!(Reads::totals(&three), (1_300, 50, 40, 30, 20));
    // Read 4: the first is counted already; the second fails after more
    // prefill, which was not seen: only what was seen counts. A third,
    // first seen finished, counts whole, once.
    let mut failed = prefill(2, 900, 500, 0);
    failed.phase = Some(TqPhase::Failed);
    let late = complete(3, 10, 0, 5);
    let four = reads.read(&snap(
        "i1",
        2,
        vec![done.clone(), failed.clone(), late.clone()],
    ));
    assert_eq!(Reads::totals(&four), (1_310, 50, 45, 30, 20));
    let five = reads.read(&snap("i1", 2, vec![done, failed, late]));
    assert_eq!(Reads::totals(&five), Reads::totals(&four), "never twice");
    // A cancelled request first seen cancelled adds nothing.
    let mut cancelled = decode(4, 70, 0, 9);
    cancelled.phase = Some(TqPhase::Cancelled);
    let six = reads.read(&snap("i1", 2, vec![cancelled]));
    assert_eq!(Reads::totals(&six), Reads::totals(&four));
    // Engine speeds from the finished requests' own times.
    let speeds = six.sample.speeds.expect("speeds");
    let prefill = speeds.prefill.expect("prefill");
    assert_eq!(prefill.seconds.count, 2.0);
    assert!((prefill.seconds.sum - 1.0).abs() < 1e-9);
    assert_eq!(prefill.tokens.sum, 950.0 + 10.0, "uncached prompt tokens");
    let decode = speeds.decode.expect("decode");
    assert_eq!((decode.tokens.sum, decode.tokens.count), (45.0, 2.0));
}

#[test]
fn a_new_instance_is_a_restart() {
    let mut reads = Reads::new();
    reads.read(&snap("i1", 1, vec![complete(1, 100, 0, 10)]));
    let again = snap("i2", 1, vec![complete(1, 7, 0, 3)]);
    assert!(reads.book.is_restart("m", &again));
    assert!(!reads.book.is_restart("m", &snap("i1", 1, Vec::new())));
    assert!(!reads.book.is_restart("other", &again), "never read");
    // Read straight on: the counts start again from the new process.
    let observed = reads.read(&again);
    assert!(observed.restarted);
    assert_eq!(Reads::totals(&observed), (7, 0, 3, 0, 0));
    // As the poller does it: forget, then read; request 1 of the new
    // instance is a new request, not the old one.
    let mut reads = Reads::new();
    reads.read(&snap("i1", 1, vec![complete(1, 100, 0, 10)]));
    reads.book.forget("m");
    let observed = reads.read(&again);
    assert!(!observed.restarted);
    assert_eq!(Reads::totals(&observed), (7, 0, 3, 0, 0));
}

#[test]
fn history_eviction_is_a_gap_not_a_spike() {
    let mut reads = Reads::new();
    let first = reads.read(&snap(
        "i1",
        1,
        vec![
            complete(1, 10, 0, 1),
            complete(2, 10, 0, 1),
            complete(3, 10, 0, 1),
        ],
    ));
    assert_eq!(first.gap, 0, "the first read is the baseline");
    // Requests 4 and 5 came and went between two reads.
    let second = reads.read(&snap(
        "i1",
        1,
        vec![complete(6, 10, 0, 1), complete(7, 10, 0, 1)],
    ));
    assert_eq!(second.gap, 2);
    assert!(second.log_gap);
    assert_eq!(Reads::totals(&second).0, 50, "only the seen ones");
    // An active request that vanished is a gap too; logged once only.
    let third = reads.read(&snap(
        "i1",
        1,
        vec![complete(7, 10, 0, 1), decode(8, 40, 0, 2)],
    ));
    assert_eq!(third.gap, 0);
    let fourth = reads.read(&snap("i1", 1, vec![complete(9, 10, 0, 1)]));
    assert_eq!(fourth.gap, 1);
    assert!(!fourth.log_gap, "once per instance");
    assert_eq!(Reads::totals(&fourth).0, 50 + 40 + 10);
}

#[test]
fn running_requests_keep_their_slot_lines() {
    let mut reads = Reads::new();
    let lines = |observed: &llama_watch::tqllm::Observed| -> Vec<(u64, Option<u32>)> {
        observed
            .facts
            .live
            .as_ref()
            .expect("live")
            .requests
            .iter()
            .map(|entry| (entry.request.id, entry.slot))
            .collect()
    };
    let one = reads.read(&snap(
        "i1",
        2,
        vec![
            decode(1, 10, 0, 1),
            prefill(2, 10, 1, 0),
            req(3, TqPhase::Queued),
        ],
    ));
    assert_eq!(lines(&one), [(1, Some(0)), (2, Some(1)), (3, None)]);
    let live = one.facts.live.as_ref().expect("live");
    assert_eq!(live.request_slots, Some(2));
    assert_eq!(live.slot_ctx, Some(16_384));
    assert_eq!(live.instance.as_deref(), Some("i1"));
    // 1 finished: 3 takes its line; 2 keeps its own.
    let two = reads.read(&snap(
        "i1",
        2,
        vec![
            complete(1, 10, 0, 2),
            decode(2, 10, 0, 1),
            prefill(3, 10, 0, 0),
        ],
    ));
    assert_eq!(lines(&two), [(2, Some(1)), (3, Some(0))]);
    // More running than slots: the extra one has no line.
    let three = reads.read(&snap(
        "i1",
        1,
        vec![decode(2, 10, 0, 2), decode(3, 10, 0, 1)],
    ));
    assert_eq!(lines(&three), [(2, None), (3, Some(0))]);
}

#[test]
fn facts_carry_ctx_slots_build_and_the_fit() {
    let mut reads = Reads::new();
    let observed = reads.read(&snap("i1", 2, Vec::new()));
    let facts = &observed.facts;
    assert_eq!(facts.ctx, Some(16_384));
    assert!(
        facts.ctx_exact,
        "the snapshot's ctx replaces --context / --np"
    );
    assert_eq!(facts.kv, None, "no KV report");
    assert_eq!(
        facts.values.get("version").map(String::as_str),
        Some("0a1b2c3")
    );
    assert_eq!(facts.values.get("slots").map(String::as_str), Some("2"));
    assert_eq!(
        facts.values.get("context_per_slot").map(String::as_str),
        Some("16384")
    );
    assert!(!facts.values.contains_key("fit"));
    assert!(observed.sample.kv.capacity.is_none() && observed.sample.kv_fill.is_none());
    assert!(reads.book.health_due("m"));
    reads
        .book
        .set_fit("m", Some("KV cache q4/tcq2 (asked bf16)".to_owned()));
    assert!(!reads.book.health_due("m"));
    let observed = reads.read(&snap("i1", 2, Vec::new()));
    assert_eq!(
        observed.facts.values.get("fit").map(String::as_str),
        Some("KV cache q4/tcq2 (asked bf16)")
    );
    // A new instance asks again.
    reads.book.forget("m");
    reads.read(&snap("i2", 2, Vec::new()));
    assert!(reads.book.health_due("m"));
    assert!(!reads.book.health_due("other"));
}

/// A finished row takes its request's numbers only when exactly one
/// finished request matches its counts and ended within 2 s of it.
#[test]
fn a_finished_row_takes_the_one_request_that_matches() {
    let claim = |seq: u64, input: u64, output: u64, at: u64| RowClaim {
        seq,
        model: "m".to_owned(),
        input,
        output,
        timestamp: Some(stamp(at)),
    };
    let mut reads = Reads::new();
    // t = 1: request 1 decoding for 0.5 s; t = 2: done after 1.5 s, so it
    // ended at about t = 2.0. Request 2 is first seen done at t = 2.
    let mut running = decode(1, 900, 600, 10);
    running.elapsed_ms = 500.0;
    reads.read(&snap("i1", 2, vec![running]));
    let mut done = complete(1, 900, 600, 30);
    done.elapsed_ms = 1_500.0;
    let other = complete(2, 900, 0, 31);
    reads.read(&snap("i1", 2, vec![done.clone(), other.clone()]));
    // Rows: one matching request 1, one matching nothing, and one with
    // its counts that ended 4 s after it.
    reads.book.activity(
        reads.at,
        &[
            claim(11, 900, 30, 2),
            claim(12, 900, 29, 2),
            claim(13, 900, 30, 6),
        ],
    );
    assert_eq!(
        reads.book.fill(11),
        None,
        "waits for a read after its window"
    );
    for _ in 0..4 {
        reads.read(&snap("i1", 2, vec![done.clone(), other.clone()]));
    }
    assert_eq!(
        reads.book.fill(11),
        Some(Fill {
            cached: 600,
            prompt_tps: Some(800.0),
            gen_tps: Some(42.5),
        })
    );
    assert_eq!(reads.book.fill(12), None, "no request has its counts");
    // Two finished requests with the same counts: neither is taken.
    let twin_a = complete(3, 50, 0, 5);
    let twin_b = complete(4, 50, 10, 5);
    reads.read(&snap("i1", 2, vec![twin_a.clone(), twin_b.clone()]));
    let now = reads.t as u64;
    reads.book.activity(reads.at, &[claim(14, 50, 5, now)]);
    for _ in 0..4 {
        reads.read(&snap("i1", 2, vec![twin_a.clone(), twin_b.clone()]));
    }
    assert_eq!(reads.book.fill(14), None, "two candidates");
    assert_eq!(reads.book.fill(13), None, "ended too far from it");
    // A row of a model this book never read waits for nothing.
    let mut stranger = claim(15, 900, 30, 2);
    stranger.model = "x".to_owned();
    reads.book.activity(reads.at, &[stranger]);
    assert_eq!(reads.book.fill(15), None);
}

/// #180: a count past 2^53 - 1, or a time or rate past it, is refused, so
/// no sum of them can become infinite; at the cap it still reads.
#[test]
fn absurd_counts_and_times_are_malformed() {
    for (key, value) in [
        ("output_tokens", "9007199254740992"),
        ("output_tokens", "18446744073709551615"),
        ("prompt_tokens", "1e17"),
        ("elapsed_ms", "1e18"),
        ("prefill_ms", "1e300"),
        ("decode_ms", "1e16"),
        ("decode_tokens_per_second", "1e20"),
    ] {
        assert_eq!(
            parse_tqllm(&with_field(4, key, value)).err(),
            Some(TqError::Malformed),
            "{key} {value}"
        );
    }
    let snap = parse_tqllm(&with_field(4, "output_tokens", "9007199254740991")).expect("cap");
    assert_eq!(snap.requests[3].output_tokens, Some(9_007_199_254_740_991));
    let snap = parse_tqllm(&with_field(4, "decode_ms", "9e15")).expect("under the cap");
    assert_eq!(snap.requests[3].decode_ms, Some(9e15));
}

/// #180: tqLLM writes a newline or a tab inside `fit` raw, which JSON
/// forbids; `/health` reads it as a space.
#[test]
fn a_raw_newline_or_tab_in_the_fit_is_a_space() {
    let body = "{\"status\":\"ok\",\"engine\":\"0a1b2c3\",\"fit\":\"to fit 1 slot(s):\n\tKV cache q4/tcq2 (asked bf16)\"}";
    assert!(body.contains('\n') && body.contains('\t'));
    assert!(
        serde_json::from_str::<serde_json::Value>(body).is_err(),
        "not JSON as sent"
    );
    assert_eq!(
        parse_tqllm_health(body),
        Ok(Some(
            "to fit 1 slot(s): KV cache q4/tcq2 (asked bf16)".to_owned()
        ))
    );
}

/// A request first seen finished at t = 31, after a 1.5 s run.
fn first_seen_finished(reads: &mut Reads) -> TqRequest {
    reads.t = 30.0;
    let mut done = complete(1, 70, 0, 7);
    done.elapsed_ms = 1_500.0;
    reads.read(&snap("i1", 1, vec![done.clone()]));
    done
}

fn row_at(seq: u64, at: u64) -> RowClaim {
    RowClaim {
        seq,
        model: "m".to_owned(),
        input: 70,
        output: 7,
        timestamp: Some(stamp(at)),
    }
}

/// The same request seen finished on the second read, at t = 31; the
/// first read, at t = 30, listed nothing.
fn seen_finished_after_a_read(reads: &mut Reads) -> TqRequest {
    reads.t = 29.0;
    reads.read(&snap("i1", 1, Vec::new()));
    let mut done = complete(1, 70, 0, 7);
    done.elapsed_ms = 1_500.0;
    reads.read(&snap("i1", 1, vec![done.clone()]));
    done
}

/// #180: a request seen only as finished on the first read ended at a
/// time not known, perhaps before llama-watch started: it is counted, but
/// it fills no row, not even one stamped the second before the read. Seen
/// finished on a later read, it ended between the two, and fills.
#[test]
fn a_request_seen_only_finished_on_the_first_read_fills_no_row() {
    let mut reads = Reads::new();
    let done = first_seen_finished(&mut reads);
    reads
        .book
        .activity(reads.at, &[row_at(21, 30), row_at(22, 6)]);
    for _ in 0..4 {
        let observed = reads.read(&snap("i1", 1, vec![done.clone()]));
        assert_eq!(Reads::totals(&observed), (70, 0, 7, 0, 0), "counted once");
    }
    assert_eq!(reads.book.fill(21), None);
    assert_eq!(reads.book.fill(22), None);
    let mut reads = Reads::new();
    let done = seen_finished_after_a_read(&mut reads);
    reads.book.activity(reads.at, &[row_at(23, 30)]);
    for _ in 0..4 {
        reads.read(&snap("i1", 1, vec![done.clone()]));
    }
    assert!(reads.book.fill(23).is_some(), "ended between two reads");
}

/// #180: when llama-swap's row ids start again (#44), the fills and the
/// rows still waiting for one go.
#[test]
fn row_fills_go_when_llama_swap_ids_start_again() {
    let mut reads = Reads::new();
    let done = seen_finished_after_a_read(&mut reads);
    reads.book.activity(reads.at, &[row_at(22, 30)]);
    for _ in 0..4 {
        reads.read(&snap("i1", 1, vec![done.clone()]));
    }
    assert!(reads.book.fill(22).is_some());
    let mut later = complete(2, 80, 0, 8);
    later.elapsed_ms = 500.0;
    reads.read(&snap("i1", 1, vec![done.clone(), later.clone()]));
    let mut waiting = row_at(23, reads.t as u64);
    (waiting.input, waiting.output) = (80, 8);
    reads.book.activity(reads.at, &[waiting]);
    reads.book.forget_rows();
    assert_eq!(reads.book.fill(22), None, "a fill of the old ids");
    for _ in 0..4 {
        reads.read(&snap("i1", 1, vec![done.clone(), later.clone()]));
    }
    assert_eq!(reads.book.fill(23), None, "a row of the old ids");
}

/// #180: an escaped `\n` and a multi-byte character in `fit` come through
/// as JSON reads them: the newline as a space, the `é` unchanged.
#[test]
fn an_escaped_newline_and_a_multibyte_character_in_the_fit() {
    let body = r#"{"status":"ok","fit":"réduit à 1 slot\nKV q4"}"#;
    assert_eq!(
        parse_tqllm_health(body),
        Ok(Some("réduit à 1 slot KV q4".to_owned()))
    );
}
