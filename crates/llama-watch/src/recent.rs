//! RECENT's own rows (#44): a bounded ring llama-watch keeps, so a
//! llama-swap restart (its `/api/metrics/activity` list is in memory and
//! comes back empty) or a model swap does not blank RECENT.
//!
//! Each activity read is merged in. A row llama-swap shows for the first
//! time goes on the front, newest first; a row already held is updated in
//! place (its fields can change, e.g. `has_capture` once llama-swap evicts
//! the capture). Nothing else leaves the ring but its oldest rows, past
//! [`RING`].
//!
//! # Generations
//!
//! A llama-swap row is `(generation, id)`: a restarted llama-swap numbers
//! its rows from 0 again. A new generation starts when a read shows
//!
//! - a newest id below the newest one seen, or an empty list after rows
//!   (llama-swap's list never shrinks while it runs);
//! - a row whose id is held from the current generation but whose
//!   fingerprint (model, timestamp, input and output tokens, duration)
//!   differs: the id was reused, even if the top id is not below the old
//!   one;
//! - after `/running` went down ([`Recent::note_down`]), no row the ring
//!   already holds: llama-swap may have restarted while it was away.
//!
//! In a new generation every row of the page is new. Rows of older
//! generations stay in the ring with what was learnt about them (engine
//! speeds, #35), and are never matched to a new row with the same id.
//!
//! Every row the ring numbers gets a [`ActivityRow::seq`]: llama-watch's own
//! row number, increasing and unique for the whole run. Per-row state
//! elsewhere ([`crate::speeds::SpeedBook`], [`crate::slots::SlotBook`]) is
//! keyed by it, never by llama-swap's id.

use std::collections::VecDeque;

use crate::activity::ActivityRow;

/// Most rows the ring keeps: the most RECENT ever shows
/// ([`crate::tty::layout::RECENT_ROWS_TEXT_OFF`]).
pub const RING: usize = 32;

/// What tells two rows with one id apart.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Fingerprint {
    model: String,
    time: String,
    input: Option<u64>,
    output: Option<u64>,
    duration: Option<u64>,
}

impl Fingerprint {
    fn of(row: &ActivityRow) -> Self {
        Self {
            model: row.model.clone(),
            time: row.time.clone(),
            input: row.input_tokens,
            output: row.output_tokens,
            duration: row.duration_ms,
        }
    }
}

#[derive(Debug)]
struct Kept {
    generation: u32,
    fingerprint: Fingerprint,
    row: ActivityRow,
}

/// One activity read, merged ([`Recent::merge`]).
#[derive(Debug, Default)]
pub struct Merged {
    /// The page as read, newest first, each row with its
    /// [`ActivityRow::seq`] when the ring numbered it (0 for an older row
    /// it never held). Every row is of the current generation.
    pub page: Vec<ActivityRow>,
    /// The rows this read showed for the first time, newest first. Empty
    /// on the first read, which is only the baseline.
    pub new: Vec<ActivityRow>,
    /// The first read: its rows are history, not new.
    pub baseline: bool,
    /// This read started a new llama-swap generation.
    pub restarted: bool,
}

/// RECENT's ring of rows across llama-swap generations.
#[derive(Debug, Default)]
pub struct Recent {
    /// Newest first, at most [`RING`].
    rows: VecDeque<Kept>,
    generation: u32,
    /// Newest id seen in the current generation; `None` before any read.
    seen: Option<i64>,
    next_seq: u64,
    /// `/running` went down since the last read.
    down: bool,
}

impl Recent {
    /// `/running` failed: the next read checks for a new generation even
    /// when its ids look like a continuation.
    pub fn note_down(&mut self) {
        self.down = true;
    }

    /// The current llama-swap generation, counted from 0 by this watcher.
    #[must_use]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Merge one parsed page, newest first.
    pub fn merge(&mut self, mut page: Vec<ActivityRow>) -> Merged {
        let baseline = self.seen.is_none();
        let restarted = !baseline && self.restarted(&page);
        if restarted {
            self.generation = self.generation.wrapping_add(1);
            self.seen = Some(-1);
        }
        self.down = false;
        let seen = self.seen.unwrap_or(-1);
        let generation = self.generation;
        let mut fresh: Vec<usize> = Vec::new();
        for (index, row) in page.iter_mut().enumerate() {
            if let Some(kept) = self
                .rows
                .iter_mut()
                .find(|kept| kept.generation == generation && kept.row.id == row.id)
            {
                row.seq = kept.row.seq;
                kept.row = row.clone();
            } else if baseline || row.id > seen {
                fresh.push(index);
            }
        }
        if baseline {
            // History from before the watcher started: shown, not counted.
            fresh.truncate(RING);
        }
        // Oldest first, so a newer row has a larger number and lands in
        // front of an older one.
        for &index in fresh.iter().rev() {
            self.next_seq += 1;
            let row = &mut page[index];
            row.seq = self.next_seq;
            self.rows.push_front(Kept {
                generation,
                fingerprint: Fingerprint::of(row),
                row: row.clone(),
            });
        }
        self.rows.truncate(RING);
        let newest = page.iter().map(|row| row.id).max().unwrap_or(-1);
        self.seen = Some(seen.max(newest));
        let new = if baseline {
            Vec::new()
        } else {
            fresh.iter().map(|&index| page[index].clone()).collect()
        };
        Merged {
            page,
            new,
            baseline,
            restarted,
        }
    }

    /// The newest `count` rows, newest first, across generations.
    #[must_use]
    pub fn rows(&self, count: usize) -> Vec<ActivityRow> {
        self.rows
            .iter()
            .take(count)
            .map(|kept| kept.row.clone())
            .collect()
    }

    /// Rows held, every generation.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Does `page` come from a llama-swap that restarted since the last read?
    fn restarted(&self, page: &[ActivityRow]) -> bool {
        let Some(seen) = self.seen else {
            return false;
        };
        let Some(newest) = page.iter().map(|row| row.id).max() else {
            return seen >= 0;
        };
        if newest < seen {
            return true;
        }
        let mut matched = false;
        for row in page.iter().filter(|row| row.id <= seen) {
            let Some(kept) = self
                .rows
                .iter()
                .find(|kept| kept.generation == self.generation && kept.row.id == row.id)
            else {
                continue;
            };
            if kept.fingerprint != Fingerprint::of(row) {
                return true;
            }
            matched = true;
        }
        self.down && !matched
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: i64, time: &str, output: u64) -> ActivityRow {
        ActivityRow {
            id,
            seq: 0,
            time: time.to_owned(),
            source: String::new(),
            model: "m".to_owned(),
            input_tokens: Some(10),
            cached_tokens: None,
            output_tokens: Some(output),
            prompt_tps: None,
            gen_tps: None,
            engine_prompt_tps: None,
            engine_gen_tps: None,
            duration_ms: Some(100),
            status: Some(200),
            draft_tokens: None,
            draft_accepted: None,
            captured: false,
        }
    }

    fn ids(rows: &[ActivityRow]) -> Vec<i64> {
        rows.iter().map(|row| row.id).collect()
    }

    #[test]
    fn the_first_read_is_history_and_later_rows_are_new_once() {
        let mut ring = Recent::default();
        let first = ring.merge(vec![row(1, "t1", 5), row(0, "t0", 4)]);
        assert!(first.baseline && first.new.is_empty());
        assert_eq!(ids(&ring.rows(8)), vec![1, 0]);
        let seqs: Vec<u64> = first.page.iter().map(|row| row.seq).collect();
        assert_eq!(seqs, vec![2, 1], "older rows get smaller numbers");

        let next = ring.merge(vec![row(3, "t3", 7), row(2, "t2", 6), row(1, "t1", 5)]);
        assert!(!next.baseline && !next.restarted);
        assert_eq!(ids(&next.new), vec![3, 2]);
        assert_eq!(next.new[0].seq, 4);
        assert_eq!(next.page[2].seq, 2, "a held row keeps its number");
        assert_eq!(ids(&ring.rows(8)), vec![3, 2, 1, 0]);
        // The same page again: nothing new.
        let again = ring.merge(vec![row(3, "t3", 7), row(2, "t2", 6)]);
        assert!(again.new.is_empty() && !again.restarted);
    }

    #[test]
    fn a_held_row_is_updated_in_place() {
        let mut ring = Recent::default();
        ring.merge(vec![row(0, "t0", 4)]);
        let mut changed = row(0, "t0", 4);
        changed.captured = true;
        changed.status = Some(500);
        let merged = ring.merge(vec![changed]);
        assert!(merged.new.is_empty() && !merged.restarted);
        let kept = &ring.rows(1)[0];
        assert!(kept.captured);
        assert_eq!(kept.status, Some(500));
        assert_eq!(kept.seq, 1);
    }

    #[test]
    fn a_restart_with_reused_ids_keeps_old_rows_and_counts_every_new_one() {
        let mut ring = Recent::default();
        ring.merge(vec![row(0, "a0", 4)]);
        ring.merge(vec![row(1, "a1", 5), row(0, "a0", 4)]);
        // Restarted: ids from 0 again, and the top id is not below the old.
        let merged = ring.merge(vec![row(2, "b2", 9), row(1, "b1", 8), row(0, "b0", 7)]);
        assert!(merged.restarted);
        assert_eq!(ids(&merged.new), vec![2, 1, 0]);
        assert_eq!(ring.generation(), 1);
        assert_eq!(ids(&ring.rows(8)), vec![2, 1, 0, 1, 0]);
        let shown = ring.rows(8);
        assert_eq!(shown[1].output_tokens, Some(8), "the new id 1");
        assert_eq!(shown[3].output_tokens, Some(5), "the old id 1 stays");
        assert_ne!(shown[1].seq, shown[3].seq);
        // The next read of the new generation updates its own rows only.
        let merged = ring.merge(vec![row(2, "b2", 9), row(1, "b1", 8)]);
        assert!(merged.new.is_empty() && !merged.restarted);
        assert_eq!(ring.len(), 5);
    }

    #[test]
    fn an_empty_list_after_rows_is_a_restart_and_keeps_the_ring() {
        let mut ring = Recent::default();
        ring.merge(vec![row(0, "a0", 4)]);
        let merged = ring.merge(Vec::new());
        assert!(merged.restarted);
        assert_eq!(ids(&ring.rows(8)), vec![0]);
        // More empty reads are not more restarts.
        assert!(!ring.merge(Vec::new()).restarted);
        // The new process's first row, id 0 again, is new.
        let merged = ring.merge(vec![row(0, "b0", 7)]);
        assert!(!merged.restarted);
        assert_eq!(ids(&merged.new), vec![0]);
        assert_eq!(ids(&ring.rows(8)), vec![0, 0]);
        // Before any row, an empty list is just empty.
        let mut ring = Recent::default();
        ring.merge(Vec::new());
        assert!(!ring.merge(Vec::new()).restarted);
    }

    #[test]
    fn a_lower_top_id_is_a_restart() {
        let mut ring = Recent::default();
        ring.merge(vec![row(5, "a5", 1)]);
        let merged = ring.merge(vec![row(1, "b1", 2), row(0, "b0", 3)]);
        assert!(merged.restarted);
        assert_eq!(ids(&merged.new), vec![1, 0]);
    }

    #[test]
    fn after_running_was_down_only_a_held_row_proves_the_same_llama_swap() {
        let mut ring = Recent::default();
        ring.merge(vec![row(1, "a1", 1), row(0, "a0", 1)]);
        ring.note_down();
        let merged = ring.merge(vec![row(2, "a2", 1), row(1, "a1", 1)]);
        assert!(!merged.restarted, "row 1 is the one held");
        assert_eq!(ids(&merged.new), vec![2]);
        // Down again, and nothing on the page is held: a new generation.
        ring.note_down();
        let merged = ring.merge(vec![row(9, "b9", 1)]);
        assert!(merged.restarted);
        assert_eq!(ids(&merged.new), vec![9]);
    }

    #[test]
    fn the_ring_is_bounded_and_drops_its_oldest() {
        let mut ring = Recent::default();
        ring.merge(Vec::new());
        for id in 0..(RING as i64 + 10) {
            let merged = ring.merge(vec![row(id, &format!("t{id:03}"), 1)]);
            assert_eq!(merged.new.len(), 1);
        }
        assert_eq!(ring.len(), RING);
        assert_eq!(ring.rows(1)[0].id, RING as i64 + 9);
        assert_eq!(ring.rows(RING)[RING - 1].id, 10);
        // A baseline page longer than the ring keeps the newest rows.
        let mut ring = Recent::default();
        let page: Vec<ActivityRow> = (0..100)
            .rev()
            .map(|id| row(id, &format!("t{id:03}"), 1))
            .collect();
        let merged = ring.merge(page);
        assert_eq!(ring.len(), RING);
        assert_eq!(ring.rows(1)[0].id, 99);
        assert_eq!(merged.page[RING].seq, 0, "never held");
    }
}
