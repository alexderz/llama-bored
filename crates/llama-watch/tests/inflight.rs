//! #75: RECENT's in-flight rows from live per-request numbers.

use std::time::{Duration, Instant, SystemTime};

use llama_watch::activity::{ActivityRow, parse_activity};
use llama_watch::inflight::{Flight, HANDOVER, Poll, Tracker};
use llama_watch::metrics::EngineLive;
use llama_watch::poller::ModelEngineLive;
use llama_watch::resets::{ResetCounts, ResetReason};
use llama_watch::slots::SlotView;

/// A busy llama.cpp slot of the `Qwen 35B` model, as `/slots` reads.
fn busy(task: i64, prompt: u64, cached: Option<u64>, processed: u64, decoded: u64) -> SlotView {
    SlotView {
        model: "Qwen 35B".to_owned(),
        id: 0,
        id_task: task,
        is_processing: true,
        n_prompt_tokens: prompt,
        n_prompt_tokens_processed: processed,
        n_decoded: decoded,
        n_ctx: Some(262_144),
        ctx_prompt: Some(prompt),
        prompt_cached: cached,
        input: Vec::new(),
        output: Vec::new(),
        ctx_used: Some(prompt + decoded),
        resets: ResetCounts::default(),
        last_reset: None,
    }
}

fn idle(mut slot: SlotView) -> SlotView {
    slot.is_processing = false;
    slot
}

/// One finished llama-swap row of `model` with `input` prompt tokens,
/// numbered `seq` by the poller.
fn row(seq: u64, model: &str, input: u64) -> ActivityRow {
    let page = format!(
        r#"{{"data":[{{"id":{seq},"timestamp":"2026-10-07T18:47:30Z","model":"{model}","tokens":{{"input_tokens":{input},"cache_tokens":0,"output_tokens":10}}}}]}}"#
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
    let got = feed.poll(0, &[busy(41, 120_000, Some(0), 48_000, 0)], &history);
    assert_eq!(got.len(), 1);
    let f = &got[0];
    assert_eq!(
        (f.prompt, f.cached, f.processed, f.decoded),
        (120_000, 0, 48_000, 0)
    );
    assert!(!f.decoding, "prefill");
    assert_eq!(f.n_ctx, Some(262_144));
    assert_eq!(f.prompt_tps, Some(2_400.0));
    // Cached tokens count once, in the cached part.
    let got = feed.poll(
        1_000,
        &[busy(41, 120_000, Some(20_000), 100_000, 0)],
        &history,
    );
    assert_eq!(
        (got[0].cached, got[0].processed, got[0].decoding),
        (20_000, 100_000, true)
    );
    let got = feed.poll(
        2_000,
        &[busy(41, 120_000, Some(20_000), 100_000, 512)],
        &history,
    );
    assert_eq!(got.len(), 1, "the same task stays one flight");
    assert_eq!((got[0].decoded, got[0].decoding), (512, true));
    assert_eq!(got[0].polled - got[0].started_mono, Duration::from_secs(2));
}

#[test]
fn a_flight_changes_only_on_a_poll() {
    let mut feed = Feed::new();
    let before = feed.poll(0, &[busy(41, 120_000, Some(0), 48_000, 0)], &[]);
    for ms in [100, 400, 900, 2_900] {
        feed.tracker.retire(feed.t0 + Duration::from_millis(ms));
        assert_eq!(feed.flights(), before, "frame at {ms} ms");
    }
}

#[test]
fn a_reset_before_a_cold_prefill_shows_its_reason() {
    let mut feed = Feed::new();
    let mut warm = idle(busy(40, 90_000, Some(88_000), 2_000, 300));
    feed.poll(0, std::slice::from_ref(&warm), &[]);
    // The next task starts from zero; the slot records a compaction.
    let mut cold = busy(41, 60_000, Some(0), 12_000, 0);
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
    feed.poll(0, &[busy(41, 120_000, Some(0), 48_000, 0)], &old);
    // An older row with the same count is not it.
    assert_eq!(
        feed.poll(500, &[busy(41, 120_000, Some(0), 90_000, 0)], &old)
            .len(),
        1
    );
    // The slot goes idle before llama-swap's row: kept to bridge the gap.
    let done = idle(busy(41, 120_000, Some(0), 120_000, 700));
    assert_eq!(feed.poll(1_000, std::slice::from_ref(&done), &old).len(), 1);
    // The row arrives: the flight goes, so the request shows once.
    let new = [row(8, "qwen3.6-35b-a3b", 120_000), old[0].clone()];
    assert!(
        feed.poll(1_500, std::slice::from_ref(&done), &new)
            .is_empty()
    );
    // A row that comes while the slot is still busy also replaces it.
    let mut feed = Feed::new();
    feed.poll(0, &[busy(50, 4_000, Some(3_800), 200, 10)], &old);
    let racing = [row(9, "qwen3.6-35b-a3b", 4_000)];
    assert!(
        feed.poll(800, &[busy(50, 4_000, Some(3_800), 200, 12)], &racing)
            .is_empty()
    );
}

#[test]
fn a_flight_without_its_row_goes_after_the_handover() {
    let mut feed = Feed::new();
    feed.poll(0, &[busy(41, 1_000, None, 1_000, 5)], &[]);
    feed.poll(100, &[idle(busy(41, 1_000, None, 1_000, 9))], &[]);
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
    let mut second = busy(60, 8_000, Some(0), 1_000, 0);
    second.id = 1;
    feed.poll(0, &[busy(41, 120_000, Some(0), 48_000, 0)], &[]);
    let got = feed.poll(500, &[busy(41, 120_000, Some(0), 60_000, 0), second], &[]);
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
