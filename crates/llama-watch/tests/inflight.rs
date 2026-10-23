//! #75: RECENT's in-flight rows from live per-request numbers.

use std::time::{Duration, Instant, SystemTime};

use llama_core::backend::Backend;
use llama_watch::activity::{ActivityRow, parse_activity};
use llama_watch::inflight::{Flight, HANDOVER, Poll, Tracker};
use llama_watch::live::LiveTokens;
use llama_watch::metrics::EngineLive;
use llama_watch::poller::{LiveRequest, ModelEngineLive};
use llama_watch::resets::{ResetCounts, ResetReason};
use llama_watch::slots::{SlotBook, SlotView};

/// A busy llama.cpp slot of the `Qwen 35B` model, as b11429's `/slots`
/// reads (#78): `n_prompt_tokens` is what the slot holds, cached +
/// processed so far, plus every sampled token but the newest.
fn busy(task: i64, cached: Option<u64>, processed: u64, decoded: u64) -> SlotView {
    let held = cached.unwrap_or(0) + processed + decoded.saturating_sub(1);
    SlotView {
        model: "Qwen 35B".to_owned(),
        id: 0,
        id_task: task,
        is_processing: true,
        n_prompt_tokens: held,
        n_prompt_tokens_processed: processed,
        n_decoded: decoded,
        n_ctx: Some(262_144),
        ctx_prompt: Some(held - decoded.min(held)),
        prompt_cached: cached,
        input: Vec::new(),
        output: Vec::new(),
        ctx_used: Some(held),
        resets: ResetCounts::default(),
        last_reset: None,
    }
}

/// The slots of a b11429-shaped fixture body, text off.
fn fixture_slots(name: &str) -> Vec<SlotView> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/llama")
        .join(name);
    let body = std::fs::read(path).expect("fixture");
    let mut book = SlotBook::without_text();
    assert!(book.apply("qwen3.6-35b-a3b", "Qwen 35B", &body, 0, 0));
    book.slots()
}

fn idle(mut slot: SlotView) -> SlotView {
    slot.is_processing = false;
    slot
}

/// One finished llama-swap row of `model` with `input` prompt tokens,
/// numbered `seq` by the poller.
fn row(seq: u64, model: &str, input: u64) -> ActivityRow {
    row_cached(seq, model, input, 0)
}

/// [`row`] with `cache` tokens reused: llama.cpp's prompt is input + cache.
fn row_cached(seq: u64, model: &str, input: u64, cache: u64) -> ActivityRow {
    let page = format!(
        r#"{{"data":[{{"id":{seq},"timestamp":"2026-10-07T18:47:30Z","model":"{model}","tokens":{{"input_tokens":{input},"cache_tokens":{cache},"output_tokens":10}}}}]}}"#
    );
    let mut rows = parse_activity(page.as_bytes()).expect("row");
    rows[0].seq = seq;
    rows.remove(0)
}

const IDS: [(&str, &str); 1] = [("Qwen 35B", "qwen3.6-35b-a3b")];

fn ids() -> Vec<(String, String)> {
    IDS.iter()
        .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
        .collect()
}

struct Feed {
    tracker: Tracker,
    t0: Instant,
    ids: Vec<(String, String)>,
}

impl Feed {
    fn new() -> Self {
        Self {
            tracker: Tracker::default(),
            t0: Instant::now(),
            ids: ids(),
        }
    }

    fn poll(&mut self, at_ms: u64, slots: &[SlotView], activity: &[ActivityRow]) -> Vec<Flight> {
        self.poll_with(at_ms, slots, &[], activity)
    }

    fn poll_with(
        &mut self,
        at_ms: u64,
        slots: &[SlotView],
        engine: &[ModelEngineLive],
        activity: &[ActivityRow],
    ) -> Vec<Flight> {
        let poll = Poll {
            slots,
            engine_live: engine,
            requests: &[],
            activity,
            ids: &self.ids,
            prompt_tps: Some(2_400.0),
            gen_tps: Some(54.2),
        };
        let mono = self.t0 + Duration::from_millis(at_ms);
        self.tracker.poll(&poll, mono, SystemTime::UNIX_EPOCH);
        self.flights()
    }

    /// One poll with vLLM or SGLang requests in flight (#80).
    fn poll_requests(
        &mut self,
        at_ms: u64,
        requests: &[LiveRequest],
        activity: &[ActivityRow],
    ) -> Vec<Flight> {
        let poll = Poll {
            slots: &[],
            engine_live: &[],
            requests,
            activity,
            ids: &self.ids,
            prompt_tps: None,
            gen_tps: None,
        };
        let mono = self.t0 + Duration::from_millis(at_ms);
        self.tracker.poll(&poll, mono, SystemTime::UNIX_EPOCH);
        self.flights()
    }

    fn flights(&self) -> Vec<Flight> {
        self.tracker.flights().into_iter().cloned().collect()
    }
}

#[test]
fn a_busy_slot_in_prefill_then_decode_is_one_flight() {
    let mut feed = Feed::new();
    let history = [row(7, "qwen3.6-35b-a3b", 900)];
    let got = feed.poll(0, &[busy(41, Some(20_000), 28_000, 0)], &history);
    assert_eq!(got.len(), 1);
    let f = &got[0];
    // #78: prefill shows what the slot holds, with no target.
    assert_eq!(
        (f.prompt, f.prompt_known, f.cached, f.processed, f.decoded),
        (48_000, false, 20_000, 28_000, 0)
    );
    assert!(!f.decoding, "prefill");
    assert_eq!(f.n_ctx, Some(262_144));
    assert_eq!(f.prompt_tps, Some(2_400.0));
    // All prompt tokens computed, still no token out: still prefill.
    let got = feed.poll(1_000, &[busy(41, Some(20_000), 100_000, 0)], &history);
    assert_eq!(
        (got[0].prompt, got[0].prompt_known, got[0].decoding),
        (120_000, false, false)
    );
    // Decoding: the prompt is exact and the output is not in IN.
    let slot = busy(41, Some(20_000), 100_000, 512);
    assert_eq!(slot.n_prompt_tokens, 120_511);
    let got = feed.poll(2_000, &[slot], &history);
    assert_eq!(got.len(), 1, "the same task stays one flight");
    assert_eq!(
        (got[0].prompt, got[0].prompt_known, got[0].cached),
        (120_000, true, 20_000)
    );
    assert_eq!(got[0].processed, 100_000);
    assert_eq!((got[0].decoded, got[0].decoding), (512, true));
    assert_eq!(got[0].polled - got[0].started_mono, Duration::from_secs(2));
}

#[test]
fn a_flight_changes_only_on_a_poll() {
    let mut feed = Feed::new();
    let before = feed.poll(0, &[busy(41, Some(0), 48_000, 0)], &[]);
    for ms in [100, 400, 900, 2_900] {
        feed.tracker.retire(feed.t0 + Duration::from_millis(ms));
        assert_eq!(feed.flights(), before, "frame at {ms} ms");
    }
}

/// #96: `beats` drives the RECENT bar's head spinner, one step per poll in
/// which the growing count actually rose — `processed` in prefill,
/// `decoded` once decoding — never on a tie, and not per frame.
#[test]
fn beats_bump_only_when_the_growing_count_rises() {
    let mut feed = Feed::new();
    // Prefill: 0 -> 1,000 processed is a rise.
    let got = feed.poll(0, &[busy(41, Some(20_000), 1_000, 0)], &[]);
    assert_eq!(got[0].beats, 1);
    // Same processed count, a later poll: no rise, no bump.
    let got = feed.poll(100, &[busy(41, Some(20_000), 1_000, 0)], &[]);
    assert_eq!(got[0].beats, 1);
    // Processed grows again: another beat.
    let got = feed.poll(200, &[busy(41, Some(20_000), 2_000, 0)], &[]);
    assert_eq!(got[0].beats, 2);
    // Decoding starts: 0 -> 1 decoded is itself a rise, its own metric now.
    let got = feed.poll(300, &[busy(41, Some(20_000), 2_000, 1)], &[]);
    assert_eq!(got[0].beats, 3);
    // Decoding, the same decoded count: no bump.
    let got = feed.poll(400, &[busy(41, Some(20_000), 2_000, 1)], &[]);
    assert_eq!(got[0].beats, 3);
    // Decoded grows again: another beat.
    let got = feed.poll(500, &[busy(41, Some(20_000), 2_000, 4)], &[]);
    assert_eq!(got[0].beats, 4);
}

/// #96: the vLLM/SGLang path (#80) bumps `beats` the same way, from
/// `decoded` once a byte streamed.
#[test]
fn beats_bump_on_the_live_request_path_too() {
    let mut feed = Feed::new();
    let tokens = |output: u64| LiveTokens {
        prompt: Some(9_000),
        cached: Some(8_192),
        output: Some(output),
        ..LiveTokens::default()
    };
    let got = feed.poll_requests(0, &[vllm(&feed, "30", 0, 4_096, Some(tokens(10)))], &[]);
    assert_eq!(got[0].beats, 1);
    let got = feed.poll_requests(100, &[vllm(&feed, "30", 0, 4_096, Some(tokens(10)))], &[]);
    assert_eq!(got[0].beats, 1, "same output: no bump");
    let got = feed.poll_requests(200, &[vllm(&feed, "30", 0, 8_192, Some(tokens(20)))], &[]);
    assert_eq!(got[0].beats, 2);
}

#[test]
fn a_reset_before_a_cold_prefill_shows_its_reason() {
    let mut feed = Feed::new();
    let mut warm = idle(busy(40, Some(88_000), 2_000, 300));
    feed.poll(0, std::slice::from_ref(&warm), &[]);
    // The next task starts from zero; the slot records a compaction.
    let mut cold = busy(41, Some(0), 12_000, 0);
    cold.resets.add(ResetReason::Compacted);
    cold.last_reset = Some(ResetReason::Compacted);
    let got = feed.poll(1_000, std::slice::from_ref(&cold), &[]);
    assert_eq!(got[0].reset, Some(ResetReason::Compacted));
    // A task with cache hits is no reset, whatever the slot recorded.
    warm.id_task = 42;
    warm.is_processing = true;
    warm.resets = cold.resets;
    warm.last_reset = cold.last_reset;
    let got = feed.poll(2_000, &[warm], &[]);
    assert_eq!(got.iter().find(|f| f.cached > 0).unwrap().reset, None);
}

#[test]
fn the_finished_row_replaces_the_flight_without_a_duplicate() {
    let mut feed = Feed::new();
    let old = [row(7, "qwen3.6-35b-a3b", 120_000)];
    feed.poll(0, &[busy(41, Some(0), 48_000, 0)], &old);
    // An older row with the same count is not it.
    assert_eq!(
        feed.poll(500, &[busy(41, Some(0), 90_000, 0)], &old).len(),
        1
    );
    // The slot goes idle before llama-swap's row: kept to bridge the gap.
    let done = idle(busy(41, Some(0), 120_000, 700));
    assert_eq!(feed.poll(1_000, std::slice::from_ref(&done), &old).len(), 1);
    // The row arrives: the flight goes, so the request shows once.
    let new = [row(8, "qwen3.6-35b-a3b", 120_000), old[0].clone()];
    assert!(
        feed.poll(1_500, std::slice::from_ref(&done), &new)
            .is_empty()
    );
    // A row that comes while the slot is still busy also replaces it:
    // its input + cache is the flight's whole prompt (#78).
    let mut feed = Feed::new();
    feed.poll(0, &[busy(50, Some(3_800), 200, 10)], &old);
    let racing = [row_cached(9, "qwen3.6-35b-a3b", 200, 3_800)];
    assert!(
        feed.poll(800, &[busy(50, Some(3_800), 200, 12)], &racing)
            .is_empty()
    );
    // A prefill's held count is a lower bound: a row of that size is not
    // its finished row.
    let mut feed = Feed::new();
    feed.poll(0, &[busy(60, Some(0), 5_000, 0)], &old);
    let other = [row(10, "qwen3.6-35b-a3b", 5_000)];
    let got = feed.poll(500, &[busy(60, Some(0), 5_000, 0)], &other);
    assert_eq!(got.len(), 1, "{got:?}");
}

#[test]
fn a_flight_without_its_row_goes_after_the_handover() {
    let mut feed = Feed::new();
    feed.poll(0, &[busy(41, None, 1_000, 5)], &[]);
    feed.poll(100, &[idle(busy(41, None, 1_000, 9))], &[]);
    feed.tracker
        .retire(feed.t0 + Duration::from_millis(100) + HANDOVER - Duration::from_millis(1));
    assert_eq!(feed.flights().len(), 1);
    feed.tracker
        .retire(feed.t0 + Duration::from_millis(100) + HANDOVER);
    assert!(feed.flights().is_empty());
}

#[test]
fn several_busy_slots_are_newest_first() {
    let mut feed = Feed::new();
    let mut second = busy(60, Some(7_000), 1_000, 0);
    second.id = 1;
    feed.poll(0, &[busy(41, Some(0), 48_000, 0)], &[]);
    let got = feed.poll(500, &[busy(41, Some(0), 60_000, 0), second], &[]);
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].prompt, 8_000, "newest first");
}

#[test]
fn strata_live_is_a_flight_with_no_cached_part() {
    let mut feed = Feed::new();
    let live = |state: &str, read: u64, generated: u64| ModelEngineLive {
        model: "strata".to_owned(),
        live: EngineLive {
            state: Some(state.to_owned()),
            prompt_total: Some(20_000),
            prompt_read: Some(read),
            generated: Some(generated),
            prefill_tok_s_mean: Some(800.0),
            tok_s: Some(31.5),
            ..EngineLive::default()
        },
    };
    let got = feed.poll_with(0, &[], &[live("reading", 5_000, 0)], &[]);
    assert_eq!(got.len(), 1);
    assert_eq!(
        (got[0].prompt, got[0].cached, got[0].processed),
        (20_000, 0, 5_000)
    );
    assert!(got[0].prompt_known, "Strata gives its prompt's length");
    assert!(!got[0].decoding);
    assert_eq!(got[0].prompt_tps, Some(800.0));
    let got = feed.poll_with(1_000, &[], &[live("generating", 20_000, 64)], &[]);
    assert_eq!(got.len(), 1);
    assert_eq!(
        (got[0].processed, got[0].decoded, got[0].decoding),
        (20_000, 64, true)
    );
    assert_eq!(got[0].gen_tps, Some(31.5));
    // Idle: it ends and waits for its row.
    let got = feed.poll_with(2_000, &[], &[live("idle", 0, 0)], &[]);
    assert_eq!(got.len(), 1);
}

#[test]
fn engines_without_live_numbers_have_no_flight() {
    // vLLM, SGLang and OpenAI-compatible servers: no slots, no live report.
    let mut feed = Feed::new();
    let rows = [row(3, "vllm", 4_000)];
    assert!(feed.poll(0, &[], &rows).is_empty());
    let mut strata_idle = Feed::new();
    let quiet = ModelEngineLive {
        model: "strata".to_owned(),
        live: EngineLive::default(),
    };
    assert!(strata_idle.poll_with(0, &[], &[quiet], &[]).is_empty());
}

/// #78: one request through b11429's `/slots` (fixtures, invented
/// numbers): in prefill the row holds cached + processed so far with no
/// target; decoding, the prompt is exact and IN leaves the output out; its
/// llama-swap row (input + cache) replaces it.
#[test]
fn b11429_prefill_then_decode_reads_held_tokens_right() {
    let mut feed = Feed::new();
    let prefill = fixture_slots("slots-b11429-prefill.json");
    assert_eq!(prefill.len(), 2);
    assert_eq!(
        prefill[0].prompt_total(),
        None,
        "/slots gives no prompt length"
    );
    let got = feed.poll(0, &prefill, &[]);
    assert_eq!(got.len(), 1, "the idle slot is no flight");
    let f = &got[0];
    assert_eq!(
        (
            f.prompt,
            f.prompt_known,
            f.cached,
            f.processed,
            f.decoded,
            f.decoding
        ),
        (69_632, false, 61_440, 8_192, 0, false)
    );

    let decode = fixture_slots("slots-b11429-decode.json");
    let slot = &decode[0];
    assert_eq!((slot.n_prompt_tokens, slot.n_decoded), (84_529, 517));
    // cached + processed is the prompt; the slot holds all but the newest
    // sampled token besides.
    assert_eq!(slot.prompt_total(), Some(84_013));
    assert_eq!(slot.n_prompt_tokens - slot.n_decoded + 1, 84_013);
    let got = feed.poll(1_000, &decode, &[]);
    assert_eq!(got.len(), 1, "the same task");
    let f = &got[0];
    assert_eq!(
        (
            f.prompt,
            f.prompt_known,
            f.cached,
            f.processed,
            f.decoded,
            f.decoding
        ),
        (84_013, true, 61_440, 22_573, 517, true)
    );
    // The idle slot's finished task: its prompt is known too.
    assert_eq!(decode[1].prompt_total(), Some(12_000));

    let finished = [row_cached(11, "qwen3.6-35b-a3b", 22_573, 61_440)];
    assert!(feed.poll(1_500, &decode, &finished).is_empty());
}

/// #78: a prompt-cache restore: most of the prompt reused, little
/// computed yet. Prefill shows the held count, all but 64 of it cached.
#[test]
fn b11429_cache_restore_is_mostly_cached_with_no_target() {
    let mut feed = Feed::new();
    let got = feed.poll(0, &fixture_slots("slots-b11429-restore.json"), &[]);
    assert_eq!(got.len(), 1);
    let f = &got[0];
    assert_eq!(
        (f.prompt, f.prompt_known, f.cached, f.processed, f.decoding),
        (118_336, false, 118_272, 64, false)
    );
    // The first token out: the whole prompt was one more batch.
    let mut first = busy(901, Some(118_272), 1_200, 1);
    first.id = 0;
    let got = feed.poll(500, &[first], &[]);
    assert_eq!(
        (got[0].prompt, got[0].prompt_known, got[0].processed),
        (119_472, true, 1_200)
    );
}

/// A Strata `/metrics` document in parallel mode (#81), parsed as the
/// poller parses it: `live` as given, two batch slots.
fn strata_parallel(live: serde_json::Value) -> ModelEngineLive {
    let doc = serde_json::json!({
        "engine": {"context": 262_144, "batch_slots": 2},
        "live": live,
    });
    let (_, facts) = llama_watch::metrics::parse_strata(&doc.to_string());
    ModelEngineLive {
        model: "strata".to_owned(),
        live: facts.live.expect("live"),
    }
}

/// #81: each busy batch slot is its own row. The newest request (the one
/// reading slot whose prompt is `live.prompt_tokens`) gets the top-level
/// `prompt_read` / `prompt_total` and a target; DUR runs from the slot's
/// own `elapsed_s`; no cached part while it runs.
#[test]
fn strata_parallel_slots_are_one_flight_each() {
    let mut feed = Feed::new();
    let live = |read: u64, gen0: u64| {
        strata_parallel(serde_json::json!({
            "state": "reading", "parallel": 2, "running": 2, "waiting": 0,
            "outside_slots": 0, "prompt_tokens": 5_965, "prompt_read": read,
            "prompt_total": 5_965, "generated": 0, "prefill_tok_s_mean": 900.0,
            "slots": [
                {"slot": 0, "state": "decoding", "prompt_tokens": 7_853,
                 "generated": gen0, "elapsed_s": 1.7, "tok_s": 34.4},
                {"slot": 1, "state": "reading", "prompt_tokens": 5_965,
                 "generated": 0, "elapsed_s": 0.4, "tok_s": null},
            ],
        }))
    };
    let got = feed.poll_with(0, &[], &[live(1_024, 51)], &[]);
    assert_eq!(got.len(), 2);
    let (reading, decoding) = (&got[0], &got[1]);
    assert_eq!(
        (
            reading.prompt,
            reading.cached,
            reading.processed,
            reading.decoded
        ),
        (5_965, 0, 1_024, 0),
        "newest first: slot 1"
    );
    assert!(reading.prompt_known && reading.progress && !reading.decoding);
    assert_eq!(reading.prompt_tps, Some(900.0));
    assert_eq!(
        (decoding.prompt, decoding.processed, decoding.decoded),
        (7_853, 7_853, 51)
    );
    assert!(decoding.decoding && !decoding.progress);
    assert_eq!(decoding.gen_tps, Some(34.4));
    assert_eq!(decoding.engine, Some(llama_core::backend::Backend::Strata));
    // DUR from the engine's clock: 1.7 s before it was first seen.
    let dur = decoding
        .polled
        .saturating_duration_since(decoding.started_mono);
    assert_eq!(dur, Duration::from_millis(1_700));

    // The next poll: the same two requests, moved on.
    let got = feed.poll_with(500, &[], &[live(4_096, 60)], &[]);
    assert_eq!(got.len(), 2);
    assert_eq!((got[0].processed, got[1].decoded), (4_096, 60));

    // A reading slot that is not the newest shows its prompt, no target.
    let other = strata_parallel(serde_json::json!({
        "state": "reading", "parallel": 2, "running": 2,
        "prompt_tokens": 900, "prompt_read": 100, "prompt_total": 900,
        "slots": [
            {"slot": 0, "state": "reading", "prompt_tokens": 7_000, "generated": 0, "elapsed_s": 2.0},
            {"slot": 1, "state": "reading", "prompt_tokens": 900, "generated": 0, "elapsed_s": 0.1},
        ],
    }));
    let mut fresh = Feed::new();
    let got = fresh.poll_with(0, &[], &[other], &[]);
    let old = got.iter().find(|f| f.prompt == 7_000).expect("slot 0");
    assert!(!old.progress && !old.decoding && old.prompt_known);
    let new = got.iter().find(|f| f.prompt == 900).expect("slot 1");
    assert_eq!((new.processed, new.progress), (100, true));
}

/// #81: a slot that runs a new request is a new row; each finished row
/// (llama-swap's Strata input leaves the cache out, #82) replaces exactly
/// one flight, never two, and none is shown twice.
#[test]
fn strata_slots_hand_over_one_row_each() {
    let two = |gen0: u64, gen1: u64| {
        strata_parallel(serde_json::json!({
            "state": "generating", "parallel": 2, "running": 2,
            "slots": [
                {"slot": 0, "state": "decoding", "prompt_tokens": 63, "generated": gen0, "elapsed_s": 3.0},
                {"slot": 1, "state": "decoding", "prompt_tokens": 63, "generated": gen1, "elapsed_s": 2.0},
            ],
        }))
    };
    let mut feed = Feed::new();
    let history = [row(1, "strata", 10)];
    assert_eq!(feed.poll_with(0, &[], &[two(5, 4)], &history).len(), 2);
    // One of the two 63-token requests finished: one row, one flight gone.
    let done = [row_cached(2, "strata", 3, 60), row(1, "strata", 10)];
    let got = feed.poll_with(500, &[], &[two(9, 8)], &done);
    assert_eq!(got.len(), 1, "one row replaces one flight: {got:?}");
    // Slot 0 now runs a new request: a new row, started 0.2 s ago.
    let next = strata_parallel(serde_json::json!({
        "state": "generating", "parallel": 2, "running": 2,
        "slots": [
            {"slot": 0, "state": "reading", "prompt_tokens": 2_000, "generated": 0, "elapsed_s": 0.2},
            {"slot": 1, "state": "decoding", "prompt_tokens": 63, "generated": 12, "elapsed_s": 2.5},
        ],
    }));
    let got = feed.poll_with(1_000, &[], &[next], &done);
    let prompts: Vec<u64> = got.iter().map(|f| f.prompt).collect();
    assert!(prompts.contains(&2_000), "{prompts:?}");
    // Both finish: two rows, every flight replaced, nothing doubled.
    let idle = strata_parallel(serde_json::json!({
        "state": "idle", "parallel": 2, "running": 0,
        "slots": [
            {"slot": 0, "state": "idle", "held_tokens": 2_040},
            {"slot": 1, "state": "idle", "held_tokens": 80},
        ],
    }));
    let all = [
        row_cached(4, "strata", 40, 1_960),
        row_cached(3, "strata", 1, 62),
        row_cached(2, "strata", 3, 60),
        row(1, "strata", 10),
    ];
    assert!(feed.poll_with(1_500, &[], &[idle], &all).is_empty());
}

/// A vLLM request llama-swap lists (#80), `bytes` streamed so far.
fn vllm(
    feed: &Feed,
    id: &str,
    started_ms: u64,
    bytes: u64,
    tokens: Option<LiveTokens>,
) -> LiveRequest {
    LiveRequest {
        model: "qwen-vllm".to_owned(),
        id: id.to_owned(),
        engine: Backend::Vllm,
        started: feed.t0 + Duration::from_millis(started_ms),
        started_wall: SystemTime::UNIX_EPOCH,
        resp_bytes: bytes,
        tokens,
    }
}

/// #80: a vLLM request is `pp` until a byte streams, `gen` after; its DUR
/// runs from llama-swap's start; before any metrics read credits it, it
/// has only llama-swap's byte count.
#[test]
fn a_vllm_request_is_a_flight_from_the_event_stream() {
    let mut feed = Feed::new();
    let got = feed.poll_requests(500, &[vllm(&feed, "30", 0, 0, None)], &[]);
    assert_eq!(got.len(), 1);
    let f = &got[0];
    assert!(!f.decoding, "nothing streamed: prefill");
    assert!(f.prompt_unknown && f.output_unknown && !f.prompt_known);
    assert_eq!(f.bytes, Some(0));
    assert_eq!(f.engine, Some(Backend::Vllm));
    assert_eq!(
        f.polled.saturating_duration_since(f.started_mono),
        Duration::from_millis(500)
    );
    let tokens = LiveTokens {
        prompt: Some(9_000),
        cached: Some(8_192),
        output: Some(64),
        ..LiveTokens::default()
    };
    let got = feed.poll_requests(1_000, &[vllm(&feed, "30", 0, 4_096, Some(tokens))], &[]);
    let f = &got[0];
    assert!(f.decoding, "bytes streamed: decoding");
    assert_eq!(
        (f.prompt, f.cached, f.processed, f.decoded),
        (9_000, 8_192, 808, 64)
    );
    assert!(f.prompt_known && !f.approx && f.bytes.is_none());
    // A shared count is drawn ≈ and never matches a row by its prompt.
    let shared = LiveTokens {
        approx: true,
        ..tokens
    };
    let got = feed.poll_requests(1_500, &[vllm(&feed, "30", 0, 8_000, Some(shared))], &[]);
    assert!(got[0].approx && !got[0].prompt_known);
}

/// #80: llama-swap's in-flight id is not its activity id, so an ended
/// request takes the oldest later row of its model that no other flight
/// took; two requests that end together take two rows, never one twice,
/// and an exact vLLM prompt takes its own row first.
#[test]
fn vllm_flights_hand_over_one_row_each() {
    let mut feed = Feed::new();
    let exact = LiveTokens {
        prompt: Some(9_000),
        cached: Some(8_192),
        output: Some(10),
        ..LiveTokens::default()
    };
    let history = [row(5, "qwen-vllm", 100)];
    let both = [
        vllm(&feed, "30", 0, 10, None),
        vllm(&feed, "31", 100, 10, Some(exact)),
    ];
    assert_eq!(feed.poll_requests(200, &both, &history).len(), 2);
    // Both end; one row arrives, with 31's whole prompt (vLLM's input
    // counts the cache, #82).
    let one = [
        row_cached(6, "qwen-vllm", 9_000, 8_192),
        row(5, "qwen-vllm", 100),
    ];
    let got = feed.poll_requests(400, &[], &one);
    assert_eq!(got.len(), 1, "31 took its own row: {got:?}");
    assert!(got[0].bytes.is_some(), "30 waits for its row");
    // 30's row comes: nothing is left, and nothing comes back.
    let two = [
        row(7, "qwen-vllm", 300),
        row_cached(6, "qwen-vllm", 9_000, 8_192),
        row(5, "qwen-vllm", 100),
    ];
    assert!(feed.poll_requests(600, &[], &two).is_empty());
    // A late frame still listing 30 does not bring it back.
    assert!(
        feed.poll_requests(700, &[vllm(&feed, "30", 0, 10, None)], &two)
            .is_empty()
    );
}

// ---- #180: tqLLM ---------------------------------------------------------------

/// The tqLLM requests of one snapshot, read as the poller reads them
/// (through the book, which gives each running request its line).
fn tqllm_live(
    book: &mut llama_watch::tqllm::TqllmBook,
    requests: Vec<llama_watch::metrics::TqRequest>,
) -> ModelEngineLive {
    let snap = llama_watch::metrics::TqSnapshot {
        instance: "4242000000000000007".to_owned(),
        context_per_slot: Some(32_768),
        slots: Some(2),
        build: None,
        requests,
    };
    let now = SystemTime::now();
    let observed = book.observe("tq", &snap, Instant::now(), now, now);
    ModelEngineLive {
        model: "tq".to_owned(),
        live: observed.facts.live.expect("live"),
    }
}

fn tq_request(
    id: u64,
    phase: llama_watch::metrics::TqPhase,
    elapsed_ms: f64,
) -> llama_watch::metrics::TqRequest {
    llama_watch::metrics::TqRequest {
        id,
        phase: Some(phase),
        elapsed_ms,
        ..llama_watch::metrics::TqRequest::default()
    }
}

fn tq_prefill(id: u64, read: u64) -> llama_watch::metrics::TqRequest {
    llama_watch::metrics::TqRequest {
        prompt_tokens: Some(6_000),
        prefill_processed: Some(read),
        prefill_cached: Some(1_024),
        prefill_tps: Some(1_137.5),
        context_tokens: Some(6_000),
        ..tq_request(id, llama_watch::metrics::TqPhase::Prefill, 900.0)
    }
}

fn tq_decode(id: u64, output: u64, elapsed_ms: f64) -> llama_watch::metrics::TqRequest {
    llama_watch::metrics::TqRequest {
        prompt_tokens: Some(800),
        prefill_processed: Some(800),
        prefill_cached: Some(512),
        prefill_tps: Some(443.0),
        output_tokens: Some(output),
        context_tokens: Some(800 + output),
        decode_tps: Some(83.5),
        ..tq_request(id, llama_watch::metrics::TqPhase::Decode, elapsed_ms)
    }
}

/// #180: one row per active tqLLM request. `pp` is `prefill_processed`
/// against the prompt with the cached part apart, `gen` the exact output,
/// each with its own live rate; DUR is the request's `elapsed_ms`; a
/// queued request is a row too, its prompt unknown.
#[test]
fn tqllm_requests_are_one_flight_each() {
    use llama_watch::metrics::TqPhase;
    let mut book = llama_watch::tqllm::TqllmBook::default();
    let mut feed = Feed::new();
    let live = tqllm_live(
        &mut book,
        vec![tq_decode(1, 150, 2_500.0), tq_prefill(2, 2_048)],
    );
    let got = feed.poll_with(0, &[], &[live], &[]);
    assert_eq!(got.len(), 2);
    let (reading, decoding) = (&got[0], &got[1]);
    assert_eq!(
        (
            reading.prompt,
            reading.cached,
            reading.processed,
            reading.decoded
        ),
        (6_000, 1_024, 1_024, 0),
        "newest first: the prefill"
    );
    assert!(reading.prompt_known && reading.progress && !reading.decoding);
    assert!(!reading.cached_unknown && !reading.prompt_unknown);
    assert_eq!(reading.prompt_tps, Some(1_137.5));
    assert_eq!(reading.n_ctx, Some(32_768));
    assert_eq!(reading.engine, Some(Backend::Tqllm));
    assert_eq!(
        (
            decoding.prompt,
            decoding.cached,
            decoding.processed,
            decoding.decoded
        ),
        (800, 512, 288, 150)
    );
    assert!(decoding.decoding);
    assert_eq!(decoding.gen_tps, Some(83.5));
    let dur = |flight: &Flight| flight.polled.saturating_duration_since(flight.started_mono);
    assert_eq!(dur(reading), Duration::from_millis(900));
    assert_eq!(dur(decoding), Duration::from_millis(2_500));

    // pp → gen: the prefill finishes reading and decodes; DUR follows the
    // engine's clock. A queued request joins with its prompt unknown.
    let mut handover = tq_decode(2, 5, 1_400.0);
    handover.prompt_tokens = Some(6_000);
    handover.prefill_processed = Some(6_000);
    handover.prefill_cached = Some(1_024);
    let live = tqllm_live(
        &mut book,
        vec![
            tq_decode(1, 190, 3_000.0),
            handover,
            tq_request(3, TqPhase::Queued, 100.0),
        ],
    );
    let got = feed.poll_with(500, &[], &[live], &[]);
    assert_eq!(got.len(), 3);
    let queued = &got[0];
    assert!(queued.prompt_unknown && !queued.decoding);
    let two = got.iter().find(|f| f.prompt == 6_000).expect("request 2");
    assert!(two.decoding);
    assert_eq!((two.processed, two.cached, two.decoded), (4_976, 1_024, 5));
    assert_eq!(dur(two), Duration::from_millis(1_400));
    assert!(two.beats >= 1, "its count rose");
}

/// #180: handover to llama-swap's finished row is one flight per row,
/// oldest first; a row whose prompt matches a flight takes that one.
#[test]
fn tqllm_flights_hand_over_one_per_finished_row() {
    use llama_watch::metrics::TqPhase;
    let mut book = llama_watch::tqllm::TqllmBook::default();
    let mut feed = Feed::new();
    let live = tqllm_live(
        &mut book,
        vec![tq_decode(1, 150, 2_500.0), tq_prefill(2, 2_048)],
    );
    feed.poll_with(0, &[], &[live], &[]);
    // Both finish. Their rows: one has request 2's prompt (6,000, the
    // whole prompt: tqLLM's input_tokens counts the cache), one another.
    let mut one = tq_decode(1, 200, 3_000.0);
    one.phase = Some(TqPhase::Complete);
    let mut two = tq_decode(2, 10, 2_000.0);
    two.phase = Some(TqPhase::Complete);
    two.prompt_tokens = Some(6_000);
    let live = tqllm_live(&mut book, vec![one.clone(), two.clone()]);
    let got = feed.poll_with(500, &[], std::slice::from_ref(&live), &[]);
    assert_eq!(got.len(), 2, "ended, waiting for their rows");
    let rows = [row(2, "tq", 6_000)];
    let got = feed.poll_with(1_000, &[], std::slice::from_ref(&live), &rows);
    assert_eq!(got.len(), 1, "the matching row took request 2: {got:?}");
    assert_eq!(got[0].prompt, 800);
    // The next row matches nothing exactly: the oldest ended flight goes.
    let rows = [row(3, "tq", 801), row(2, "tq", 6_000)];
    let got = feed.poll_with(1_500, &[], std::slice::from_ref(&live), &rows);
    assert!(got.is_empty(), "{got:?}");
    // Never back: a later snapshot still listing them adds no flight.
    let got = feed.poll_with(2_000, &[], &[live], &rows);
    assert!(got.is_empty());
    // A restarted engine's request 1 is a new request.
    let mut fresh = llama_watch::tqllm::TqllmBook::default();
    let mut live = tqllm_live(&mut fresh, vec![tq_decode(1, 3, 100.0)]);
    live.live.instance = Some("4242000000000000099".to_owned());
    let got = feed.poll_with(2_500, &[], &[live], &rows);
    assert_eq!(got.len(), 1);
}

/// #180: a tqLLM request still running takes no finished row, not even one
/// with its own prompt: the row goes to the request that ended.
#[test]
fn a_running_tqllm_flight_never_takes_another_requests_row() {
    use llama_watch::metrics::TqPhase;
    let mut book = llama_watch::tqllm::TqllmBook::default();
    let mut feed = Feed::new();
    // A (1) is older; B (2) has the same 800-token prompt.
    let live = tqllm_live(
        &mut book,
        vec![tq_decode(1, 150, 3_000.0), tq_decode(2, 100, 1_000.0)],
    );
    feed.poll_with(0, &[], &[live], &[]);
    // B finishes first, A decodes on; the row is B's.
    let mut b_done = tq_decode(2, 120, 1_400.0);
    b_done.phase = Some(TqPhase::Complete);
    let live = tqllm_live(&mut book, vec![tq_decode(1, 180, 3_500.0), b_done]);
    let rows = [row(1, "tq", 800)];
    let got = feed.poll_with(500, &[], std::slice::from_ref(&live), &rows);
    assert_eq!(got.len(), 1, "{got:?}");
    assert!(got[0].decoding);
    assert_eq!(got[0].decoded, 180, "A is still a row: {got:?}");
}

/// #180: 20 requests on 2 slots: the 2 running ones are rows, then the
/// next queued ones up to the cap; poll after poll the same ones, and no
/// queued request pushes a running one out.
#[test]
fn many_queued_tqllm_requests_never_push_out_the_running_ones() {
    use llama_watch::inflight::MAX_TQ_QUEUED;
    use llama_watch::metrics::TqPhase;
    let mut book = llama_watch::tqllm::TqllmBook::default();
    let mut feed = Feed::new();
    let dur = |flight: &Flight| flight.polled.saturating_duration_since(flight.started_mono);
    for poll in 0..6_u64 {
        let mut requests = vec![
            tq_decode(1, 100 + poll, 2_000.0),
            tq_prefill(2, 1_024 + poll),
        ];
        // Queued 3..=20, each waiting 50 ms × its id.
        requests.extend((3..=20).map(|id| tq_request(id, TqPhase::Queued, 50.0 * id as f64)));
        let live = tqllm_live(&mut book, requests);
        let got = feed.poll_with(poll * 500, &[], &[live], &[]);
        assert_eq!(got.len(), 2 + MAX_TQ_QUEUED, "{poll}: {got:?}");
        assert!(
            got.iter().any(|f| f.decoding && f.decoded == 100 + poll),
            "{poll}: the decoding one"
        );
        assert!(
            got.iter().any(|f| f.prompt == 6_000 && !f.decoding),
            "{poll}: the reading one"
        );
        let mut waits: Vec<Duration> = got.iter().filter(|f| f.prompt_unknown).map(dur).collect();
        waits.sort();
        let next: Vec<Duration> = (3..3 + MAX_TQ_QUEUED as u64)
            .map(|id| Duration::from_millis(50 * id))
            .collect();
        assert_eq!(waits, next, "{poll}: the next to run");
    }
}
