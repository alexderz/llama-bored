//! llama.cpp WRITE in tokens, not decode calls (DER-451).
//!
//! `llamacpp:n_decode_total` counts `llama_decode` calls. With N slots
//! decoding, one call carries N tokens, so a rate of that counter is the
//! per-stream step rate, about the aggregate ÷ N; speculative decoding
//! under-reads it too. This book counts tokens instead:
//!
//! - **`/slots`**: each `(slot, id_task)` credits the rise of its
//!   `n_decoded` since the last read. A task first seen after the model's
//!   first read counts its whole `n_decoded`. The tasks of that first read
//!   are a baseline only, so a long decode already running at startup
//!   adds nothing at once.
//! - **Between `/slots` reads** (every second by default, `/metrics` every
//!   0.25 s) the rise of `n_decode_total` times the slots that were
//!   decoding at the last read is credited ahead, so a single stream moves
//!   at the metrics cadence as before. Every later confirmed credit pays
//!   that advance back first, so the total stays exact: an advance that
//!   overshot makes the next confirmed tokens add nothing until it is paid.
//! - **Tasks that end between reads**: `tokens_predicted_total` adds a
//!   task's `n_decoded` when it is released. Its rise, less what the
//!   retired tasks were already credited with, is the tail the `/slots`
//!   reads missed, including whole requests that started and ended
//!   between two reads. It is settled after a `/slots` read (which follows
//!   the `/metrics` read whose rise it settles, so every task in it is
//!   retired by then) and when `/metrics` reports nothing processing.
//!
//! When `/slots` is unavailable while the model is busy, `n_decode_total`
//! is used only while the busy slots per call over the last reads is at
//! most [`ONE_STREAM`]: one stream, where a call is a token. Otherwise
//! the total is unknown ([`LcppTokens::observe_metrics`] returns `None`),
//! never a ÷N number. A server without `tokens_predicted_total` cannot be
//! settled, so it is read the same way: decode calls while one stream
//! decodes, unknown otherwise.
//!
//! A counter that goes down is a new process under the same id, and a
//! model that is loaded again ([`LcppTokens::forget`]) is one too: its
//! counters are read from 0, so what it already did is counted in full,
//! as [`crate::metrics::DecodedCounter`] does.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use llama_core::rate::counter_delta;

/// Busy slots per decode call still read as one stream.
pub const ONE_STREAM: f64 = 1.05;

/// Fewest decode calls between two reads for their busy-slot ratio to be
/// taken; fewer keep the previous ratio.
const RATIO_CALLS: u64 = 8;

/// One slot's numbers from a `/slots` read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotTokens {
    /// Slot index.
    pub id: i64,
    /// Task id; below 0 when the slot has none.
    pub id_task: i64,
    /// The server's processing flag.
    pub is_processing: bool,
    /// `next_token[0].n_decoded`.
    pub n_decoded: u64,
}

/// What one llama.cpp `/metrics` read says.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MetricsRead {
    /// `llamacpp:n_decode_total`.
    pub calls: Option<u64>,
    /// `llamacpp:tokens_predicted_total`.
    pub predicted: Option<u64>,
    /// `llamacpp:n_busy_slots_per_decode`: busy slots per call since the
    /// server started.
    pub busy_per_decode: Option<f64>,
    /// `llamacpp:requests_processing`.
    pub processing: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Mode {
    /// Tokens from `/slots`, settled with `tokens_predicted_total`.
    #[default]
    Slots,
    /// `/slots` is unavailable and one stream decodes: decode calls.
    Calls,
    /// `/slots` is unavailable and several streams decode: unknown.
    Unknown,
}

#[derive(Debug, Default)]
struct Book {
    total: u64,
    mode: Mode,
    /// A `/slots` read (or a read with nothing processing) gave the
    /// baseline that `tasks` and `balance` count from.
    synced: bool,
    /// `n_decoded` accounted to each live `(slot, id_task)`: credited, or
    /// the baseline of the first read.
    tasks: HashMap<(i64, i64), u64>,
    /// `tokens_predicted_total` rise less the accounted tokens of the
    /// tasks retired since the last settle. Negative only by what the
    /// last `/slots` round retired (see [`Book::drop_old_debt`]).
    balance: i64,
    /// Tokens credited ahead of the `/slots` reads, not yet confirmed.
    advance: u64,
    /// Slots decoding at the last `/slots` read.
    streams: u64,
    calls: Option<u64>,
    predicted: Option<u64>,
    /// `busy_per_decode × calls` at the last read.
    busy_calls: Option<f64>,
    /// Busy slots per call between the last reads with enough calls.
    ratio: Option<f64>,
    slots_at: Option<Instant>,
    busy_since: Option<Instant>,
    /// The last `/metrics` read had no `tokens_predicted_total`.
    unsettled: bool,
}

fn signed(tokens: u64) -> i64 {
    i64::try_from(tokens).unwrap_or(i64::MAX)
}

impl Book {
    /// A book for a new process: its counters count from 0 and every task
    /// is new.
    fn new_process(total: u64, mode: Mode) -> Self {
        Self {
            total,
            mode,
            synced: true,
            calls: Some(0),
            predicted: Some(0),
            busy_calls: Some(0.0),
            ..Self::default()
        }
    }

    /// Decode calls while one stream decodes, unknown otherwise. Without
    /// a busy-slot ratio (no `n_busy_slots_per_decode`, or too few calls
    /// yet) more than one `requests_processing` is unknown (review L4); a
    /// server without that gauge either is read as before.
    fn calls_mode(&self, processing: Option<f64>) -> Mode {
        let one = match self.ratio {
            Some(ratio) => ratio <= ONE_STREAM,
            None => processing.is_none_or(|n| n <= 1.0),
        };
        if one { Mode::Calls } else { Mode::Unknown }
    }

    /// Credit `tokens` that are confirmed: they pay the advance first.
    fn confirm(&mut self, tokens: u64) {
        let paid = tokens.min(self.advance);
        self.advance -= paid;
        self.total = self.total.saturating_add(tokens - paid);
    }

    /// Retire `accounted` tokens of a released task: they are what
    /// `tokens_predicted_total` adds for it, less its tail.
    fn retire(&mut self, accounted: u64) {
        self.balance = self.balance.saturating_sub(signed(accounted));
    }

    /// Before a round's retires: debt the `/metrics` read since the last
    /// round has not paid is dropped. The tasks that round retired were
    /// released after the read before it, so this read's
    /// `tokens_predicted_total` has them; what is still owed is a task
    /// llama.cpp never adds, cancelled or aborted, and must not eat later
    /// tasks' tails (review M2). This round's debt is kept whole.
    fn drop_old_debt(&mut self) {
        self.balance = self.balance.max(0);
    }

    /// Settle the tail: a positive balance is tokens no read credited. A
    /// negative one is this round's tasks, owed until the next read.
    fn settle(&mut self) {
        if self.balance > 0 {
            self.confirm(u64::try_from(self.balance).unwrap_or(0));
            self.balance = 0;
        }
    }

    /// Drop the `/slots` baseline; the next `/slots` read sets a new one.
    fn unsync(&mut self) {
        self.synced = false;
        self.tasks.clear();
        self.balance = 0;
        self.streams = 0;
    }
}

/// Per-model llama.cpp token totals. See the module docs.
#[derive(Debug)]
pub struct LcppTokens {
    models: HashMap<String, Book>,
    /// How long `/slots` may go unread while the model is busy before the
    /// total falls back to decode calls or becomes unknown.
    stale: Duration,
}

impl Default for LcppTokens {
    fn default() -> Self {
        Self::new(Duration::from_secs(3))
    }
}

impl LcppTokens {
    /// A book whose `/slots` reads go stale after `stale`.
    #[must_use]
    pub fn new(stale: Duration) -> Self {
        Self {
            models: HashMap::new(),
            stale,
        }
    }

    /// Fold in one `/metrics` read of `model`. Returns the model's token
    /// total, or `None` while it is unknown.
    pub fn observe_metrics(
        &mut self,
        model: &str,
        read: &MetricsRead,
        now: Instant,
    ) -> Option<u64> {
        let stale = self.stale;
        let book = self.models.entry(model.to_owned()).or_default();
        let restarted = matches!((book.calls, read.calls), (Some(last), Some(value)) if value < last)
            || matches!((book.predicted, read.predicted), (Some(last), Some(value)) if value < last);
        if restarted {
            // A new process under the same id: its counters from 0.
            *book = Book::new_process(book.total, book.mode);
        }
        let calls = read.calls.map(|value| counter_delta(book.calls, value));
        let predicted = read
            .predicted
            .map(|value| counter_delta(book.predicted, value));
        let busy_calls = read
            .busy_per_decode
            .zip(read.calls)
            .map(|(per, calls)| per * calls as f64);
        if let (Some(delta), Some(busy_now), Some(busy_last)) = (calls, busy_calls, book.busy_calls)
            && delta >= RATIO_CALLS
        {
            book.ratio = Some((busy_now - busy_last) / delta as f64);
        }
        if read.calls.is_some() {
            book.calls = read.calls;
        }
        if read.predicted.is_some() {
            book.predicted = read.predicted;
        }
        if busy_calls.is_some() {
            book.busy_calls = busy_calls;
        }
        let calls = calls.unwrap_or(0);
        book.unsettled = read.predicted.is_none();
        if book.unsettled {
            book.unsync();
            book.busy_since = None;
            book.mode = book.calls_mode(read.processing);
            return match book.mode {
                Mode::Unknown => None,
                _ => {
                    book.confirm(calls);
                    Some(book.total)
                }
            };
        }
        let predicted = predicted.unwrap_or(0);
        if !read.processing.is_some_and(|n| n > 0.0) {
            // Nothing processing: every task is released and in
            // `tokens_predicted_total`, so the book settles exactly.
            book.busy_since = None;
            if book.mode == Mode::Slots && book.synced {
                book.balance = book.balance.saturating_add(signed(predicted));
                book.drop_old_debt();
                let retired: Vec<u64> = book.tasks.drain().map(|(_, n)| n).collect();
                for accounted in retired {
                    book.retire(accounted);
                }
                book.settle();
            } else if book.mode == Mode::Calls {
                book.confirm(calls);
            }
            book.unsync();
            book.synced = true;
            book.mode = Mode::Slots;
            return Some(book.total);
        }
        let since = *book.busy_since.get_or_insert(now);
        let slots_stale = book
            .slots_at
            .is_none_or(|at| now.saturating_duration_since(at) > stale);
        if slots_stale && now.saturating_duration_since(since) > stale {
            book.unsync();
            book.mode = book.calls_mode(read.processing);
        }
        match book.mode {
            Mode::Slots => {
                if book.synced {
                    book.balance = book.balance.saturating_add(signed(predicted));
                    let ahead = calls.saturating_mul(book.streams);
                    book.total = book.total.saturating_add(ahead);
                    book.advance = book.advance.saturating_add(ahead);
                }
                Some(book.total)
            }
            Mode::Calls => {
                book.confirm(calls);
                Some(book.total)
            }
            Mode::Unknown => None,
        }
    }

    /// Fold in one `/slots` read of `model`, after the `/metrics` read
    /// whose `tokens_predicted_total` it settles.
    pub fn observe_slots(&mut self, model: &str, slots: &[SlotTokens], now: Instant) {
        let book = self.models.entry(model.to_owned()).or_default();
        if book.unsettled {
            return;
        }
        book.slots_at = Some(now);
        book.drop_old_debt();
        if book.mode != Mode::Slots {
            book.mode = Mode::Slots;
            book.unsync();
        }
        let baseline = !book.synced;
        let mut round = 0u64;
        let mut live = HashMap::with_capacity(slots.len());
        let mut streams = 0u64;
        for slot in slots.iter().filter(|slot| slot.id_task >= 0) {
            let key = (slot.id, slot.id_task);
            let accounted = match book.tasks.remove(&key) {
                Some(seen) => {
                    round = round.saturating_add(slot.n_decoded.saturating_sub(seen));
                    seen.max(slot.n_decoded)
                }
                // An idle slot's task that was never seen live: the
                // settle with `tokens_predicted_total` counts it.
                None if !slot.is_processing => continue,
                None => {
                    if !baseline {
                        round = round.saturating_add(slot.n_decoded);
                    }
                    slot.n_decoded
                }
            };
            if slot.is_processing {
                live.insert(key, accounted);
                if slot.n_decoded > 0 {
                    streams += 1;
                }
            } else {
                // Released: its final count is in, and so will its
                // `tokens_predicted_total` share be.
                book.retire(accounted);
            }
        }
        // Tasks no longer listed as processing were released.
        let gone: Vec<u64> = book.tasks.drain().map(|(_, n)| n).collect();
        for accounted in gone {
            book.retire(accounted);
        }
        book.tasks = live;
        book.streams = streams;
        if baseline {
            // Whatever `tokens_predicted_total` rose by before this read
            // was for tasks this book never saw live.
            book.balance = 0;
            book.synced = true;
            return;
        }
        book.confirm(round);
        book.settle();
    }

    /// `model`'s process is gone: the next one counts from 0.
    pub fn forget(&mut self, model: &str) {
        if let Some(book) = self.models.get_mut(model) {
            *book = Book::new_process(0, Mode::Slots);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(id: i64, id_task: i64, busy: bool, decoded: u64) -> SlotTokens {
        SlotTokens {
            id,
            id_task,
            is_processing: busy,
            n_decoded: decoded,
        }
    }

    fn read(calls: u64, predicted: u64, processing: f64) -> MetricsRead {
        MetricsRead {
            calls: Some(calls),
            predicted: Some(predicted),
            busy_per_decode: None,
            processing: Some(processing),
        }
    }

    /// CCR-3 corrected: two slots decoding share one call per step. 50
    /// steps are 50 calls and 100 tokens; the book counts 100.
    #[test]
    fn two_streams_count_tokens_not_decode_calls() {
        let mut book = LcppTokens::default();
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        assert_eq!(book.observe_metrics("m", &read(1000, 0, 2.0), t0), Some(0));
        book.observe_slots("m", &[slot(0, 1, true, 10), slot(1, 2, true, 10)], t0);
        let at = t0 + s;
        book.observe_metrics("m", &read(1050, 0, 2.0), at);
        book.observe_slots("m", &[slot(0, 1, true, 60), slot(1, 2, true, 60)], at);
        assert_eq!(
            book.observe_metrics("m", &read(1050, 0, 2.0), at),
            Some(100)
        );
    }

    /// N = 1, 2 and 4 streams, ten steps per quarter second for 3 s, read
    /// every 0.25 s with `/slots` every second: the total is N × steps at
    /// each `/slots` read, and between them it never runs ahead of the
    /// tokens the calls carried.
    #[test]
    fn one_two_and_four_streams_total_exactly() {
        for n in [1u64, 2, 4] {
            let mut book = LcppTokens::default();
            let t0 = Instant::now();
            let quarter = Duration::from_millis(250);
            let streams = |decoded: u64| -> Vec<SlotTokens> {
                (0..n)
                    .map(|i| {
                        let i = i64::try_from(i).unwrap();
                        slot(i, 10 + i, true, decoded)
                    })
                    .collect()
            };
            book.observe_metrics("m", &read(0, 0, n as f64), t0);
            book.observe_slots("m", &streams(1), t0);
            let mut last = 0;
            for q in 1u32..=12 {
                let at = t0 + quarter * q;
                let calls = u64::from(q) * 10;
                let total = book
                    .observe_metrics("m", &read(calls, 0, n as f64), at)
                    .expect("known");
                assert!(total >= last, "n {n} q {q}: monotonic");
                assert!(total <= n * calls, "n {n} q {q}: {total} ahead");
                last = total;
                if q % 4 == 0 {
                    book.observe_slots("m", &streams(1 + calls), at);
                    let total = book
                        .observe_metrics("m", &read(calls, 0, n as f64), at)
                        .expect("known");
                    assert_eq!(total, n * calls, "n {n} at {q}");
                    last = total;
                }
            }
        }
    }

    /// A single stream between `/slots` reads moves with the decode calls,
    /// as WRITE did before: the advance is exact when a call is a token.
    #[test]
    fn a_single_stream_moves_at_the_metrics_cadence() {
        let mut book = LcppTokens::default();
        let t0 = Instant::now();
        book.observe_metrics("m", &read(100, 0, 1.0), t0);
        book.observe_slots("m", &[slot(0, 7, true, 50)], t0);
        let q = Duration::from_millis(250);
        assert_eq!(
            book.observe_metrics("m", &read(110, 0, 1.0), t0 + q),
            Some(10)
        );
        assert_eq!(
            book.observe_metrics("m", &read(120, 0, 1.0), t0 + q * 2),
            Some(20)
        );
        book.observe_slots("m", &[slot(0, 7, true, 70)], t0 + q * 2);
        assert_eq!(
            book.observe_metrics("m", &read(120, 0, 1.0), t0 + q * 2),
            Some(20)
        );
    }

    /// Speculative decoding: three accepted tokens per call. The advance
    /// under-reads and the `/slots` read brings the rest.
    #[test]
    fn speculative_tokens_arrive_with_the_slots_read() {
        let mut book = LcppTokens::default();
        let t0 = Instant::now();
        book.observe_metrics("m", &read(0, 0, 1.0), t0);
        book.observe_slots("m", &[slot(0, 1, true, 3)], t0);
        let s = Duration::from_secs(1);
        assert_eq!(
            book.observe_metrics("m", &read(20, 0, 1.0), t0 + s),
            Some(20)
        );
        book.observe_slots("m", &[slot(0, 1, true, 63)], t0 + s);
        assert_eq!(
            book.observe_metrics("m", &read(20, 0, 1.0), t0 + s),
            Some(60)
        );
    }

    /// A task that ends between two reads: its last tokens come from the
    /// rise of `tokens_predicted_total`, once.
    #[test]
    fn a_task_ending_between_reads_is_settled_by_tokens_predicted() {
        let mut book = LcppTokens::default();
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        book.observe_metrics("m", &read(0, 0, 2.0), t0);
        book.observe_slots("m", &[slot(0, 1, true, 100), slot(1, 2, true, 10)], t0);
        // Task 1 decodes 30 more and is released (130 in
        // tokens_predicted_total); task 2 goes on.
        book.observe_metrics("m", &read(0, 130, 1.0), t0 + s);
        book.observe_slots("m", &[slot(0, 1, false, 100), slot(1, 2, true, 50)], t0 + s);
        assert_eq!(
            book.observe_metrics("m", &read(0, 130, 1.0), t0 + s),
            Some(70)
        );
        // Task 2 ends with 60; nothing is processing.
        assert_eq!(
            book.observe_metrics("m", &read(0, 190, 0.0), t0 + s * 2),
            Some(80)
        );
    }

    /// A whole request that starts and ends between two `/slots` reads is
    /// counted from `tokens_predicted_total`.
    #[test]
    fn a_request_never_seen_live_is_counted_once() {
        let mut book = LcppTokens::default();
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        book.observe_metrics("m", &read(0, 0, 1.0), t0);
        book.observe_slots("m", &[slot(0, 1, true, 5)], t0);
        book.observe_metrics("m", &read(0, 40, 1.0), t0 + s);
        book.observe_slots("m", &[slot(0, 1, true, 25), slot(1, 9, false, 40)], t0 + s);
        assert_eq!(
            book.observe_metrics("m", &read(0, 40, 1.0), t0 + s),
            Some(60)
        );
        book.observe_slots(
            "m",
            &[slot(0, 1, true, 25), slot(1, 9, false, 40)],
            t0 + s * 2,
        );
        assert_eq!(
            book.observe_metrics("m", &read(0, 40, 1.0), t0 + s * 2),
            Some(60)
        );
    }

    /// Startup mid-decode: the running task's earlier tokens are a
    /// baseline, and so are tasks released before the first read.
    #[test]
    fn startup_mid_decode_adds_only_new_tokens() {
        let mut book = LcppTokens::default();
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        book.observe_metrics("m", &read(9000, 5000, 1.0), t0);
        book.observe_metrics("m", &read(9000, 8000, 1.0), t0 + s / 2);
        book.observe_slots("m", &[slot(0, 3, true, 4000)], t0 + s / 2);
        assert_eq!(
            book.observe_metrics("m", &read(9000, 8000, 1.0), t0 + s / 2),
            Some(0)
        );
        book.observe_slots("m", &[slot(0, 3, true, 4040)], t0 + s);
        assert_eq!(
            book.observe_metrics("m", &read(9000, 8000, 1.0), t0 + s),
            Some(40)
        );
        // Released at 4050: only its last 10 are new.
        assert_eq!(
            book.observe_metrics("m", &read(9000, 12050, 0.0), t0 + s * 2),
            Some(50)
        );
    }

    /// The same slot moves to a new task between reads: the old task is
    /// retired and the new one counts in full.
    #[test]
    fn a_new_task_in_the_same_slot_counts_in_full() {
        let mut book = LcppTokens::default();
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        book.observe_metrics("m", &read(0, 0, 1.0), t0);
        book.observe_slots("m", &[slot(0, 1, true, 500)], t0);
        book.observe_metrics("m", &read(0, 510, 1.0), t0 + s);
        book.observe_slots("m", &[slot(0, 2, true, 30)], t0 + s);
        assert_eq!(
            book.observe_metrics("m", &read(0, 510, 1.0), t0 + s),
            Some(40)
        );
    }

    /// Without `/slots`, one stream falls back to decode calls; several
    /// streams are unknown, never a ÷N number.
    #[test]
    fn without_slots_one_stream_uses_calls_and_several_are_unknown() {
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        for (per, expect) in [(1.0, Some(100)), (2.0, None)] {
            let mut book = LcppTokens::new(Duration::from_secs(3));
            let mut got = None;
            for i in 0u32..=5 {
                let calls = 1000 + u64::from(i) * 50;
                let mut sample = read(calls, 0, 2.0);
                sample.busy_per_decode = Some(per);
                got = book.observe_metrics("m", &sample, t0 + s * i);
            }
            // Busy since t0, stale after 3 s: the reads at 4 s and 5 s count.
            assert_eq!(got, expect, "busy per call {per}");
        }
    }

    /// The busy-slot ratio is read between reads, not since the server
    /// started: an old episode of concurrency does not block the fallback.
    #[test]
    fn the_busy_ratio_is_windowed() {
        let mut book = LcppTokens::new(Duration::from_secs(3));
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        let mut got = None;
        for i in 0u32..=5 {
            let calls = 1000 + u64::from(i) * 50;
            // 1000 calls at 2 busy slots, then one slot: the lifetime mean
            // stays above 1, the windowed ratio is 1.
            let busy = 2000.0 + f64::from(i) * 50.0;
            let sample = MetricsRead {
                busy_per_decode: Some(busy / calls as f64),
                ..read(calls, 0, 1.0)
            };
            got = book.observe_metrics("m", &sample, t0 + s * i);
        }
        assert_eq!(got, Some(100));
    }

    /// `/slots` coming back after the fallback sets a new baseline, so the
    /// calls already counted are not counted again.
    #[test]
    fn slots_returning_after_the_fallback_rebase() {
        let mut book = LcppTokens::new(Duration::from_secs(1));
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        book.observe_metrics("m", &read(0, 0, 1.0), t0);
        assert_eq!(
            book.observe_metrics("m", &read(40, 0, 1.0), t0 + s * 2),
            Some(40)
        );
        assert_eq!(
            book.observe_metrics("m", &read(80, 0, 1.0), t0 + s * 3),
            Some(80)
        );
        book.observe_slots("m", &[slot(0, 1, true, 80)], t0 + s * 3);
        // `/metrics` is read before `/slots` in a pass.
        assert_eq!(
            book.observe_metrics("m", &read(120, 0, 1.0), t0 + s * 4),
            Some(120)
        );
        book.observe_slots("m", &[slot(0, 1, true, 120)], t0 + s * 4);
        assert_eq!(
            book.observe_metrics("m", &read(120, 0, 1.0), t0 + s * 4),
            Some(120)
        );
    }

    /// A restarted server under the same id lowers its counters: new
    /// baselines, and the total does not jump.
    #[test]
    fn a_restart_keeps_the_total() {
        let mut book = LcppTokens::default();
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        book.observe_metrics("m", &read(0, 0, 1.0), t0);
        book.observe_slots("m", &[slot(0, 1, true, 0)], t0);
        book.observe_slots("m", &[slot(0, 1, true, 30)], t0 + s);
        assert_eq!(
            book.observe_metrics("m", &read(30, 0, 0.0), t0 + s),
            Some(30)
        );
        assert_eq!(
            book.observe_metrics("m", &read(2, 0, 0.0), t0 + s * 2),
            Some(30)
        );
        assert_eq!(
            book.observe_metrics("m", &read(4, 0, 0.0), t0 + s * 3),
            Some(30)
        );
        // The new process's finished tokens count in full.
        assert_eq!(
            book.observe_metrics("m", &read(9, 12, 0.0), t0 + s * 4),
            Some(42)
        );
    }

    /// A model loaded again counts its new process from 0, so what it did
    /// before the first read counts in full (the box counter does the
    /// same after its forget).
    #[test]
    fn a_reloaded_model_counts_from_zero() {
        let mut book = LcppTokens::default();
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        book.observe_metrics("m", &read(500, 400, 0.0), t0);
        assert_eq!(
            book.observe_metrics("m", &read(600, 450, 0.0), t0 + s),
            Some(50)
        );
        book.forget("m");
        assert_eq!(
            book.observe_metrics("m", &read(80, 70, 0.0), t0 + s * 2),
            Some(70),
            "a new book: its 70 counted from 0"
        );
        // A model never seen has no book: its first read is a baseline.
        book.forget("n");
        assert_eq!(book.observe_metrics("n", &read(80, 70, 0.0), t0), Some(0));
    }

    /// Without `tokens_predicted_total` nothing can be settled: one stream
    /// counts decode calls, several are unknown, and `/slots` is ignored.
    #[test]
    fn without_tokens_predicted_one_stream_uses_calls() {
        let no_predicted = |calls: u64, busy: f64, processing: f64| MetricsRead {
            calls: Some(calls),
            predicted: None,
            busy_per_decode: Some(busy),
            processing: Some(processing),
        };
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        for n in [1u64, 2, 4] {
            let mut book = LcppTokens::default();
            book.observe_metrics("m", &no_predicted(100, n as f64, n as f64), t0);
            book.observe_slots("m", &[slot(0, 1, true, 5)], t0);
            let got = book.observe_metrics("m", &no_predicted(150, n as f64, n as f64), t0 + s);
            book.observe_slots("m", &[slot(0, 1, true, 500)], t0 + s);
            let after =
                book.observe_metrics("m", &no_predicted(160, n as f64, n as f64), t0 + s * 2);
            if n == 1 {
                assert_eq!((got, after), (Some(50), Some(60)), "{n} stream");
            } else {
                assert_eq!((got, after), (None, None), "{n} streams");
            }
        }
    }

    /// M2: llama.cpp does not add a cancelled task to
    /// `tokens_predicted_total`. Its retired tokens are debt for one round
    /// only, then dropped, so the next task's tail still counts. A task
    /// released after the `/metrics` read is paid one round later.
    #[test]
    fn a_cancelled_task_does_not_eat_a_later_tail() {
        for cancelled in [true, false] {
            let mut book = LcppTokens::default();
            let t0 = Instant::now();
            let s = Duration::from_secs(1);
            book.observe_metrics("m", &read(0, 0, 2.0), t0);
            book.observe_slots("m", &[slot(0, 1, true, 10), slot(1, 2, true, 10)], t0);
            book.observe_metrics("m", &read(0, 0, 2.0), t0 + s);
            book.observe_slots("m", &[slot(0, 1, true, 100), slot(1, 2, true, 50)], t0 + s);
            // Task 1 is released after this /metrics read: cancelled at
            // 100, or finished at 110.
            let last = if cancelled { 100 } else { 110 };
            book.observe_metrics("m", &read(0, 0, 2.0), t0 + s * 2);
            book.observe_slots(
                "m",
                &[slot(0, 1, false, last), slot(1, 2, true, 80)],
                t0 + s * 2,
            );
            let predicted = if cancelled { 0 } else { 110 };
            book.observe_metrics("m", &read(0, predicted, 1.0), t0 + s * 3);
            book.observe_slots("m", &[slot(1, 2, true, 120)], t0 + s * 3);
            // Task 2 ends at 150, 30 past the last /slots read.
            let total = book.observe_metrics("m", &read(0, predicted + 150, 0.0), t0 + s * 4);
            let task1 = last - 10;
            assert_eq!(total, Some(task1 + 140), "cancelled {cancelled}");
        }
    }

    /// L4: without `n_busy_slots_per_decode` there is no busy-slot ratio,
    /// so stale `/slots` reads decode calls only while at most one request
    /// is processing; with more the total is unknown.
    #[test]
    fn without_a_busy_ratio_several_requests_are_unknown() {
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        for (processing, expect) in [(1.0, Some(100)), (2.0, None)] {
            let mut book = LcppTokens::new(Duration::from_secs(3));
            let mut got = None;
            for i in 0u32..=5 {
                let calls = 1000 + u64::from(i) * 50;
                got = book.observe_metrics("m", &read(calls, 0, processing), t0 + s * i);
            }
            // Busy since t0, stale after 3 s: the reads at 4 s and 5 s count.
            assert_eq!(got, expect, "processing {processing}");
        }
    }

    /// Review of M2: a cancelled task, then rounds that each retire a task,
    /// released before the `/metrics` read (paid that round), after it
    /// (paid the next), or with a tail past their last `/slots` read. The
    /// cancelled task's debt is dropped once and eats no later tail: every
    /// task counts exactly what it decoded.
    #[test]
    fn a_cancelled_tasks_debt_does_not_follow_later_retires() {
        let mut book = LcppTokens::default();
        let t0 = Instant::now();
        let at = |k: u32| t0 + Duration::from_secs(1) * k;
        book.observe_metrics("m", &read(0, 0, 4.0), at(0));
        let start: Vec<SlotTokens> = (0..4).map(|i| slot(i, 1 + i, true, 10)).collect();
        book.observe_slots("m", &start, at(0));
        // Task 1 is cancelled at 100: never in tokens_predicted_total.
        book.observe_metrics("m", &read(0, 0, 4.0), at(1));
        let round = [
            slot(0, 1, false, 100),
            slot(1, 2, true, 50),
            slot(2, 3, true, 50),
            slot(3, 4, true, 50),
        ];
        book.observe_slots("m", &round, at(1));
        // Task 2 ends at 90 before this read.
        book.observe_metrics("m", &read(0, 90, 3.0), at(2));
        let round = [
            slot(1, 2, false, 90),
            slot(2, 3, true, 70),
            slot(3, 4, true, 70),
        ];
        book.observe_slots("m", &round, at(2));
        // Task 3 ends at 80 after this read.
        book.observe_metrics("m", &read(0, 90, 2.0), at(3));
        let round = [
            slot(1, 2, false, 90),
            slot(2, 3, false, 80),
            slot(3, 4, true, 90),
        ];
        book.observe_slots("m", &round, at(3));
        // Task 4 ends at 120, 30 past its last read; task 6 has its slot.
        book.observe_metrics("m", &read(0, 90 + 80 + 120, 1.0), at(4));
        book.observe_slots("m", &[slot(3, 6, true, 3)], at(4));
        // Task 6 ends at 10 between reads.
        let total = book.observe_metrics("m", &read(0, 90 + 80 + 120 + 10, 0.0), at(5));
        let decoded = (100 - 10) + (90 - 10) + (80 - 10) + (120 - 10) + 10;
        assert_eq!(total, Some(decoded));
    }
}
