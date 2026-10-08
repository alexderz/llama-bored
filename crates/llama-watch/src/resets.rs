//! Why a llama.cpp slot's context dropped (#9). Numbers only: no prompt
//! text is kept or needed, so this works with `tty.show_text = false`.
//!
//! # What the watcher can see
//!
//! llama-swap v256's `/api/metrics/activity` rows carry no session id (their
//! `metadata` holds only `fifo_priority`), even when the client sent
//! `X-Session-Id`; only a request capture keeps the headers, and reading
//! captures for this would be a large extra GET per request. So every
//! reason here is a **best guess** from token counts:
//!
//! - the drop itself, from `/slots` (the sparkline rule: a busy slot's
//!   context falls by more than 30 % and 2k tokens);
//! - the new task's whole prompt `P` and how much of it was reused from
//!   the cache `C`. `/slots` gives `P` only once the task decodes
//!   ([`crate::slots::whole_prompt`]; its `n_prompt_tokens` is what the
//!   slot holds, #78), so a drop seen in prefill waits for that; one whose
//!   task is never seen decoding is [`ResetReason::Unknown`]. `C` comes
//!   from the finished request's llama-swap activity row (`cache_tokens`; llama.cpp's timings make
//!   `input_tokens + cache_tokens` the whole prompt, so the row whose sum is
//!   `P` is this task's), or, when no row matches in time, from `/slots`
//!   `n_prompt_tokens_cache` if the server sends it.
//!
//! # Rules (all guesses)
//!
//! - [`ResetReason::Compacted`]: `C/P` ≥ [`COMPACT_SHARE`]. Most of the
//!   shorter prompt was already cached, so it shares a long prefix with the
//!   old context: the same conversation, summarised or rewound. A new
//!   conversation of the same agent can share a large system prompt and
//!   read as compacted too.
//! - [`ResetReason::Evicted`]: `C/P` ≤ [`EVICT_SHARE`] (the whole prompt
//!   processed again) and `P` fits a conversation this model's slots lost
//!   in the last [`DISPLACED_TTL`]: from 0.7× to 1.25× (at least +8k) of the
//!   context it held when it left. Conversations are matched by size only.
//! - [`ResetReason::New`]: any other drop with a known `C`: little of the
//!   prompt was cached and it fits no lost conversation.
//! - [`ResetReason::Unknown`]: no `C` within [`PENDING_TTL`] of the drop
//!   (activity off or too slow, and no `n_prompt_tokens_cache`).
//!
//! A drop is counted once, when its reason is decided, so a counter can lag
//! the drop by up to [`PENDING_TTL`].

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Cached share at or above which a drop reads as compacted.
pub const COMPACT_SHARE: f64 = 0.5;
/// Cached share at or below which a prompt counts as fully reprocessed.
pub const EVICT_SHARE: f64 = 0.1;
/// How long a drop waits for its activity row before it is decided without.
pub const PENDING_TTL: Duration = Duration::from_secs(300);
/// How long a conversation that left a slot is remembered.
pub const DISPLACED_TTL: Duration = Duration::from_secs(3_600);
/// Conversations remembered per model.
pub const DISPLACED_MAX: usize = 16;
/// Models whose lost conversations are remembered at once. They outlive an
/// unload and a llama-swap restart (#45), so the oldest model goes first.
pub const DISPLACED_MODELS: usize = 64;

/// Why a context dropped. Every value is a guess; see the module docs.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ResetReason {
    /// Same conversation, shorter prompt.
    Compacted,
    /// A different conversation took the slot.
    New,
    /// A conversation came back and its cache was gone.
    Evicted,
    /// Not enough evidence.
    Unknown,
}

impl ResetReason {
    /// Every reason, in export order.
    pub const ALL: [Self; 4] = [Self::Compacted, Self::New, Self::Evicted, Self::Unknown];

    /// The wire and label word.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Compacted => "compacted",
            Self::New => "new",
            Self::Evicted => "evicted",
            Self::Unknown => "unknown",
        }
    }

    /// This reason's bit in a set (the sparkline's per-bucket reasons).
    #[must_use]
    pub fn bit(self) -> u8 {
        match self {
            Self::Compacted => 1,
            Self::New => 2,
            Self::Evicted => 4,
            Self::Unknown => 8,
        }
    }

    /// The reason a set of bits shows as: evicted, then new, compacted,
    /// unknown. `None` for an empty set.
    #[must_use]
    pub fn strongest(bits: u8) -> Option<Self> {
        [Self::Evicted, Self::New, Self::Compacted, Self::Unknown]
            .into_iter()
            .find(|reason| bits & reason.bit() != 0)
    }
}

/// Drops by reason since the watcher started.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResetCounts {
    pub compacted: u64,
    pub new: u64,
    pub evicted: u64,
    pub unknown: u64,
}

impl ResetCounts {
    /// The count for `reason`.
    #[must_use]
    pub fn get(&self, reason: ResetReason) -> u64 {
        match reason {
            ResetReason::Compacted => self.compacted,
            ResetReason::New => self.new,
            ResetReason::Evicted => self.evicted,
            ResetReason::Unknown => self.unknown,
        }
    }

    /// Count one drop.
    pub fn add(&mut self, reason: ResetReason) {
        let slot = match reason {
            ResetReason::Compacted => &mut self.compacted,
            ResetReason::New => &mut self.new,
            ResetReason::Evicted => &mut self.evicted,
            ResetReason::Unknown => &mut self.unknown,
        };
        *slot = slot.saturating_add(1);
    }

    /// Every drop.
    #[must_use]
    pub fn total(&self) -> u64 {
        ResetReason::ALL
            .into_iter()
            .fold(0u64, |sum, reason| sum.saturating_add(self.get(reason)))
    }
}

/// Context sizes of conversations that left one model's slots.
#[derive(Debug, Default)]
pub struct Displaced {
    sizes: VecDeque<(u64, Instant)>,
}

impl Displaced {
    /// Remember a conversation of `size` tokens that left a slot at `now`.
    pub fn push(&mut self, size: u64, now: Instant) {
        self.sizes.push_front((size, now));
        self.sizes.truncate(DISPLACED_MAX);
    }

    /// Take the newest remembered conversation a `prompt`-token prompt could
    /// be the return of. Expired ones are dropped first.
    pub fn take_match(&mut self, prompt: u64, now: Instant) -> bool {
        self.prune(now);
        let fits = |size: u64| {
            let low = size.saturating_mul(7) / 10;
            let high = size.saturating_add((size / 4).max(8_192));
            (low..=high).contains(&prompt)
        };
        match self.sizes.iter().position(|(size, _)| fits(*size)) {
            Some(index) => {
                self.sizes.remove(index);
                true
            }
            None => false,
        }
    }

    /// Forget conversations older than [`DISPLACED_TTL`].
    pub fn prune(&mut self, now: Instant) {
        self.sizes
            .retain(|(_, at)| now.saturating_duration_since(*at) <= DISPLACED_TTL);
    }

    /// When the newest remembered conversation left.
    #[must_use]
    pub fn newest(&self) -> Option<Instant> {
        self.sizes.iter().map(|(_, at)| *at).max()
    }

    /// Conversations remembered now.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sizes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }
}

/// Decide one drop. `prompt` is the new task's prompt tokens, `cached` the
/// part reused from the cache (`None`: no evidence). `displaced` is the
/// model's lost conversations; an evicted match is taken out of it.
#[must_use]
pub fn classify(
    prompt: u64,
    cached: Option<u64>,
    displaced: &mut Displaced,
    now: Instant,
) -> ResetReason {
    let Some(cached) = cached else {
        return ResetReason::Unknown;
    };
    if prompt == 0 {
        return ResetReason::Unknown;
    }
    let share = cached.min(prompt) as f64 / prompt as f64;
    if share >= COMPACT_SHARE {
        ResetReason::Compacted
    } else if share <= EVICT_SHARE && displaced.take_match(prompt, now) {
        ResetReason::Evicted
    } else {
        ResetReason::New
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_high_cached_share_is_compacted_and_a_low_one_new() {
        let now = Instant::now();
        let mut lost = Displaced::default();
        assert_eq!(
            classify(20_000, Some(14_000), &mut lost, now),
            ResetReason::Compacted
        );
        assert_eq!(
            classify(20_000, Some(10_000), &mut lost, now),
            ResetReason::Compacted
        );
        assert_eq!(
            classify(20_000, Some(9_999), &mut lost, now),
            ResetReason::New
        );
        assert_eq!(classify(20_000, Some(0), &mut lost, now), ResetReason::New);
        assert_eq!(classify(20_000, None, &mut lost, now), ResetReason::Unknown);
        assert_eq!(classify(0, Some(0), &mut lost, now), ResetReason::Unknown);
        // A cached count above the prompt is read as all of it.
        assert_eq!(
            classify(100, Some(500), &mut lost, now),
            ResetReason::Compacted
        );
    }

    #[test]
    fn a_returning_conversation_with_no_cache_is_evicted_once() {
        let t0 = Instant::now();
        let mut lost = Displaced::default();
        lost.push(80_000, t0);
        let later = t0 + Duration::from_secs(60);
        // Too small or too big to be the 80k conversation: new.
        assert_eq!(
            classify(40_000, Some(0), &mut lost, later),
            ResetReason::New
        );
        assert_eq!(
            classify(120_000, Some(100), &mut lost, later),
            ResetReason::New
        );
        // Mostly cached: compacted, and the lost one stays remembered.
        assert_eq!(
            classify(82_000, Some(60_000), &mut lost, later),
            ResetReason::Compacted
        );
        assert_eq!(lost.len(), 1);
        // Back with its cache gone.
        assert_eq!(
            classify(83_500, Some(1_200), &mut lost, later),
            ResetReason::Evicted
        );
        assert!(lost.is_empty());
        assert_eq!(
            classify(83_500, Some(1_200), &mut lost, later),
            ResetReason::New
        );
    }

    #[test]
    fn lost_conversations_expire_and_are_bounded() {
        let t0 = Instant::now();
        let mut lost = Displaced::default();
        lost.push(50_000, t0);
        let late = t0 + DISPLACED_TTL + Duration::from_secs(1);
        assert_eq!(classify(50_000, Some(0), &mut lost, late), ResetReason::New);
        for size in 0..(DISPLACED_MAX as u64 + 10) {
            lost.push(size * 100_000, t0);
        }
        assert_eq!(lost.len(), DISPLACED_MAX);
    }

    #[test]
    fn counts_and_bits() {
        let mut counts = ResetCounts::default();
        counts.add(ResetReason::New);
        counts.add(ResetReason::New);
        counts.add(ResetReason::Evicted);
        assert_eq!(counts.get(ResetReason::New), 2);
        assert_eq!(counts.total(), 3);
        let bits = ResetReason::Compacted.bit() | ResetReason::Evicted.bit();
        assert_eq!(ResetReason::strongest(bits), Some(ResetReason::Evicted));
        assert_eq!(ResetReason::strongest(0), None);
        assert_eq!(
            ResetReason::ALL.map(ResetReason::as_str),
            ["compacted", "new", "evicted", "unknown"]
        );
    }
}
