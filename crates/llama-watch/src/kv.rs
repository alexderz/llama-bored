//! KV cache in use per model, across all its sessions, and its capacity
//! (#79).
//!
//! - **llama.cpp**: from `/slots`. In use is the sum of every slot's held
//!   tokens ([`SlotView::kv_tokens`]); an idle slot keeps its context. The
//!   sessions are the slots holding any. With a unified cache (one pool
//!   every slot shares, [`KvLayout::unified`]) every slot reports the
//!   whole pool as its `n_ctx`, so the capacity is one slot's `n_ctx`;
//!   with `--kv-unified-per-slot` each slot's `n_ctx` is capped, so it is
//!   their sum, at most `-c`. Without a unified cache each slot has its
//!   own part: the capacity is the sum. When the launch command could not
//!   be read, the cache is assumed unified if every slot reports the same
//!   `n_ctx`, and [`KvUsage::unified_assumed`] says so.
//! - **SGLang, vLLM, Strata**: from `/metrics` ([`KvTokens`]). The
//!   sessions are the running requests, or Strata's slots holding tokens.

use llama_core::backend::{KvUsage, MAX_REQS};

use crate::metrics::KvTokens;
use crate::slots::SlotView;
use crate::sources::cmdline::KvLayout;

impl SlotView {
    /// Tokens this slot holds in KV: `/slots` `n_prompt_tokens`, its whole
    /// cached sequence (prompt and generated). An idle slot keeps it until
    /// llama-server clears the slot. The one place #79 reads a slot's
    /// tokens, so a change to what `/slots` means lands here.
    #[must_use]
    pub fn kv_tokens(&self) -> u64 {
        self.n_prompt_tokens
    }
}

/// A llama.cpp model's KV cache from its `slots`, its launch command's
/// `layout` (`None` when it was not read) and its `-c` (`ctx`). `None`
/// when it has no slots.
#[must_use]
pub fn llamacpp(
    slots: &[&SlotView],
    layout: Option<KvLayout>,
    ctx: Option<u32>,
) -> Option<KvUsage> {
    if slots.is_empty() {
        return None;
    }
    let used = slots
        .iter()
        .fold(0u64, |sum, slot| sum.saturating_add(slot.kv_tokens()));
    let sessions = slots.iter().filter(|slot| slot.kv_tokens() > 0).count();
    let n_ctx: Option<Vec<u64>> = slots
        .iter()
        .map(|slot| slot.n_ctx.filter(|n| *n > 0))
        .collect();
    let same = n_ctx
        .as_ref()
        .is_some_and(|all| all.windows(2).all(|pair| pair[0] == pair[1]));
    let (unified, per_slot_cap, assumed) = match layout {
        Some(layout) => (layout.unified, layout.per_slot_cap, false),
        None => (same, false, true),
    };
    let capacity = n_ctx.and_then(|all| {
        let sum = all.iter().try_fold(0u64, |sum, n| sum.checked_add(*n))?;
        match (unified, per_slot_cap) {
            (true, false) => all.iter().copied().max(),
            (true, true) => Some(ctx.map_or(sum, |ctx| sum.min(u64::from(ctx)))),
            (false, _) => Some(sum),
        }
    });
    Some(KvUsage {
        used: Some(capacity.map_or(used, |capacity| used.min(capacity))),
        capacity,
        cached: None,
        sessions: Some(u16::try_from(sessions).unwrap_or(u16::MAX).min(MAX_REQS)),
        unified: Some(unified),
        unified_assumed: assumed,
        approx: false,
    })
}

/// A `/metrics` engine's KV cache from its tokens and its `running`
/// requests. `None` when it reports no tokens.
#[must_use]
pub fn from_metrics(kv: KvTokens, running: Option<u16>) -> Option<KvUsage> {
    if kv.used.is_none() && kv.capacity.is_none() && kv.cached.is_none() {
        return None;
    }
    let sessions = match kv.sessions {
        Some(n) => Some(u16::try_from(n).unwrap_or(u16::MAX).min(MAX_REQS)),
        None => running,
    };
    Some(KvUsage {
        used: kv.used,
        capacity: kv.capacity,
        cached: kv.cached,
        sessions,
        unified: None,
        unified_assumed: false,
        approx: kv.approx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(id: i64, held: u64, n_ctx: u64) -> SlotView {
        SlotView {
            model: "m".to_owned(),
            id,
            id_task: id,
            is_processing: false,
            n_prompt_tokens: held,
            n_prompt_tokens_processed: 0,
            n_decoded: 0,
            n_ctx: Some(n_ctx),
            ctx_prompt: Some(held),
            prompt_cached: None,
            input: Vec::new(),
            output: Vec::new(),
            ctx_used: Some(held),
            resets: crate::resets::ResetCounts::default(),
            last_reset: None,
        }
    }

    const SHARED: KvLayout = KvLayout {
        unified: true,
        per_slot_cap: false,
    };
    const PER_SLOT: KvLayout = KvLayout {
        unified: false,
        per_slot_cap: false,
    };

    #[test]
    fn unified_capacity_is_one_slots_n_ctx() {
        let slots = [
            slot(0, 41_000, 131_072),
            slot(1, 0, 131_072),
            slot(2, 9_000, 131_072),
        ];
        let refs: Vec<&SlotView> = slots.iter().collect();
        let kv = llamacpp(&refs, Some(SHARED), None).expect("kv");
        assert_eq!(kv.used, Some(50_000));
        assert_eq!(kv.capacity, Some(131_072));
        assert_eq!(kv.sessions, Some(2));
        assert_eq!(kv.unified, Some(true));
        assert!(!kv.unified_assumed);
        assert_eq!(kv.permille(), Some(381));
    }

    #[test]
    fn per_slot_capacity_is_the_sum() {
        let slots = [slot(0, 8_000, 32_768), slot(1, 30_000, 32_768)];
        let refs: Vec<&SlotView> = slots.iter().collect();
        let kv = llamacpp(&refs, Some(PER_SLOT), None).expect("kv");
        assert_eq!(kv.used, Some(38_000));
        assert_eq!(kv.capacity, Some(65_536));
        assert_eq!(kv.unified, Some(false));
    }

    #[test]
    fn a_per_slot_cap_sums_the_slots_up_to_the_pool() {
        let capped = KvLayout {
            unified: true,
            per_slot_cap: true,
        };
        let slots = [slot(0, 1, 16_384), slot(1, 2, 16_384), slot(2, 3, 16_384)];
        let refs: Vec<&SlotView> = slots.iter().collect();
        let kv = llamacpp(&refs, Some(capped), None).expect("kv");
        assert_eq!(kv.capacity, Some(49_152), "pool sized to n_parallel × N");
        let kv = llamacpp(&refs, Some(capped), Some(40_000)).expect("kv");
        assert_eq!(kv.capacity, Some(40_000), "-c pins the pool");
    }

    #[test]
    fn an_unknown_layout_is_assumed_from_the_slots() {
        let same = [slot(0, 5, 65_536), slot(1, 7, 65_536)];
        let refs: Vec<&SlotView> = same.iter().collect();
        let kv = llamacpp(&refs, None, None).expect("kv");
        assert_eq!((kv.unified, kv.unified_assumed), (Some(true), true));
        assert_eq!(kv.capacity, Some(65_536));
        let differ = [slot(0, 5, 65_536), slot(1, 7, 32_768)];
        let refs: Vec<&SlotView> = differ.iter().collect();
        let kv = llamacpp(&refs, None, None).expect("kv");
        assert_eq!((kv.unified, kv.unified_assumed), (Some(false), true));
        assert_eq!(kv.capacity, Some(98_304));
    }

    #[test]
    fn missing_n_ctx_leaves_the_capacity_unknown_and_no_slots_is_none() {
        let mut one = slot(0, 500, 0);
        one.n_ctx = None;
        let kv = llamacpp(&[&one], Some(SHARED), None).expect("kv");
        assert_eq!((kv.used, kv.capacity), (Some(500), None));
        assert_eq!(llamacpp(&[], Some(SHARED), None), None);
    }

    #[test]
    fn metrics_sessions_are_the_running_requests_unless_the_engine_counts_slots() {
        let tokens = KvTokens {
            used: Some(10),
            capacity: Some(100),
            cached: Some(20),
            sessions: None,
            approx: false,
        };
        let kv = from_metrics(tokens, Some(3)).expect("kv");
        assert_eq!((kv.sessions, kv.cached), (Some(3), Some(20)));
        let strata = KvTokens {
            sessions: Some(2),
            ..tokens
        };
        assert_eq!(
            from_metrics(strata, Some(0)).and_then(|kv| kv.sessions),
            Some(2)
        );
        assert_eq!(from_metrics(KvTokens::default(), Some(1)), None);
    }
}
