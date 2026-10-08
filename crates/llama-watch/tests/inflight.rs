//! #75: RECENT's in-flight rows from live per-request numbers.

use std::time::{Duration, Instant, SystemTime};

use llama_watch::activity::{ActivityRow, parse_activity};
use llama_watch::inflight::{Flight, HANDOVER, Poll, Tracker};
use llama_watch::metrics::EngineLive;
use llama_watch::poller::ModelEngineLive;
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
            activity,
            ids: &self.ids,
            prompt_tps: Some(2_400.0),
            gen_tps: Some(54.2),
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
