//! Change-only upload policy. The caller passes `now`; this module does not read a clock.

use std::time::{Duration, Instant};

use crate::present::View;

/// What the service should do with the current view on this tick.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[must_use]
pub enum Decision {
    /// Send the current view now.
    Upload,
    /// A send is wanted, and the minimum interval since the last success has not elapsed.
    Wait,
    /// The panel already shows this view, and nothing forces a resend.
    Nothing,
}

/// Last successful upload, last attempt, and any forced reupload.
///
/// The success and attempt timestamps are independent of the remembered view.
/// [`Policy::reset_on_open`] and [`Policy::force`] never clear them.
#[derive(Debug)]
pub struct Policy {
    min_interval: Duration,
    /// Set only by [`Policy::mark_uploaded`].
    last_success_at: Option<Instant>,
    /// Set by [`Policy::mark_attempted`] on every show, success or failure.
    last_attempt_at: Option<Instant>,
    /// View from the last success. [`Policy::reset_on_open`] forgets it.
    last_view: Option<View>,
    force: bool,
}

impl Policy {
    /// `min_interval` is the minimum gap between successful uploads, including
    /// forced uploads and uploads after a reopen.
    ///
    /// The 10 s floor is [`crate::config::Config::validate`]'s job.
    #[must_use]
    pub fn new(min_interval: Duration) -> Self {
        Self {
            min_interval,
            last_success_at: None,
            last_attempt_at: None,
            last_view: None,
            force: false,
        }
    }

    /// Decide whether `view` should be sent at `now`.
    ///
    /// `Nothing` when `view` matches the remembered view and nothing forces a
    /// resend. `Upload` when a send is wanted — first frame, the view changed,
    /// [`Policy::force`] was called, or [`Policy::reset_on_open`] forgot the
    /// view — and at least `min_interval` has elapsed since
    /// `max(last_success, last_attempt)`, or nothing has ever been sent.
    /// `Wait` when a send is wanted and that gap has not elapsed.
    ///
    /// Call [`Policy::mark_attempted`] on every `show`, success or failure.
    pub fn decide(&self, view: &View, now: Instant) -> Decision {
        if !self.wants_upload(view) {
            return Decision::Nothing;
        }
        if self.gap_elapsed(now) {
            Decision::Upload
        } else {
            Decision::Wait
        }
    }

    fn wants_upload(&self, view: &View) -> bool {
        self.force
            || self
                .last_view
                .as_ref()
                .is_none_or(|previous| previous != view)
    }

    /// `true` when a device write is allowed at `now`.
    ///
    /// The gap matches [`Self::decide`]: `min_interval` since the later of the
    /// last success and the last attempt, or immediately when nothing has been
    /// sent. A watch-down `ShowLiquid` checks this, then calls
    /// [`Self::mark_attempted`]. [`Self::force`] does not bypass the gap.
    #[must_use]
    pub fn allows_attempt(&self, now: Instant) -> bool {
        self.gap_elapsed(now)
    }

    fn gap_elapsed(&self, now: Instant) -> bool {
        self.last_gate()
            .is_none_or(|at| now.saturating_duration_since(at) >= self.min_interval)
    }

    fn last_gate(&self) -> Option<Instant> {
        match (self.last_success_at, self.last_attempt_at) {
            (None, None) => None,
            (Some(success), None) => Some(success),
            (None, Some(attempt)) => Some(attempt),
            (Some(success), Some(attempt)) => Some(success.max(attempt)),
        }
    }

    /// Record a successful upload of `view` at `now` and clear a forced reupload.
    ///
    /// Call this only after the sink has accepted the frame.
    pub fn mark_uploaded(&mut self, view: &View, now: Instant) {
        self.last_view = Some(view.clone());
        self.last_success_at = Some(now);
        self.force = false;
    }

    /// Record a show attempt at `now`, success or failure.
    ///
    /// The next [`decide`] returns [`Decision::Upload`] only once `min_interval`
    /// has elapsed since this attempt or a later success.
    pub fn mark_attempted(&mut self, now: Instant) {
        self.last_attempt_at = Some(now);
    }

    /// Forget the remembered view, as when the device is opened again.
    ///
    /// The last successful upload time stays. The next [`decide`] returns
    /// [`Decision::Upload`] only once `min_interval` has elapsed, or immediately
    /// when nothing has ever been uploaded. A pending [`Policy::force`] is cleared.
    pub fn reset_on_open(&mut self) {
        self.last_view = None;
        self.force = false;
    }

    /// Ask for an upload even when the view is unchanged.
    ///
    /// The gap since the last successful upload still applies. Cleared by
    /// [`Policy::mark_uploaded`] and [`Policy::reset_on_open`]. Does not clear
    /// the success timestamp.
    pub fn force(&mut self) {
        self.force = true;
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use crate::present::{Ai, Band, View};

    use super::{Decision, Policy};

    fn origin() -> Instant {
        Instant::now()
    }

    fn view(ring_pct: u8) -> View {
        View {
            ring_pct: Some(ring_pct),
            ring_band: None,
            blocks: [Band::Quiet; 3],
            coolant_c: Some(36),
            cpu_c: Some(42),
            gpu_c: Some(51),
            cpu_pct: Some(3),
            mem_pct: Some(40),
            ai: Ai::Idle,
            models: Vec::new(),
            model_count: 0,
            ..View::default()
        }
    }

    #[test]
    fn fresh_policy_uploads_and_ignores_the_interval() {
        let policy = Policy::new(Duration::from_secs(60));
        let now = origin();
        assert_eq!(policy.decide(&view(0), now), Decision::Upload);
        assert_eq!(
            policy.decide(&view(0), now + Duration::from_secs(1)),
            Decision::Upload
        );
    }

    #[test]
    fn same_view_is_nothing_until_it_changes_and_the_interval_elapses() {
        let interval = Duration::from_secs(60);
        let mut policy = Policy::new(interval);
        let t0 = origin();
        let first = view(0);
        let later = view(40);
        policy.mark_uploaded(&first, t0);

        assert_eq!(policy.decide(&first, t0 + interval), Decision::Nothing);
        assert_eq!(
            policy.decide(&later, t0 + interval - Duration::from_nanos(1)),
            Decision::Wait
        );
        assert_eq!(policy.decide(&later, t0 + interval), Decision::Upload);
        policy.mark_uploaded(&later, t0 + interval);
        assert_eq!(
            policy.decide(&first, t0 + interval),
            Decision::Wait,
            "the view from this tick is the one that stuck"
        );
    }

    #[test]
    fn force_and_reset_wait_until_the_success_gap_has_elapsed() {
        let interval = Duration::from_secs(60);
        let mut policy = Policy::new(interval);
        let t0 = origin();
        let same = view(0);
        policy.mark_uploaded(&same, t0);

        policy.force();
        let t1 = t0 + Duration::from_secs(1);
        assert_eq!(policy.decide(&same, t1), Decision::Wait);
        assert_eq!(policy.decide(&view(7), t1), Decision::Wait);
        let sent = t0 + interval;
        assert_eq!(policy.decide(&same, sent), Decision::Upload);
        policy.mark_uploaded(&same, sent);
        assert_eq!(policy.decide(&same, sent), Decision::Nothing);
        assert_eq!(
            policy.decide(&view(7), sent + interval - Duration::from_nanos(1)),
            Decision::Wait
        );

        policy.force();
        policy.reset_on_open();
        assert_eq!(
            policy.decide(&same, sent + Duration::from_secs(1)),
            Decision::Wait
        );
        let reopened = sent + interval;
        assert_eq!(policy.decide(&same, reopened), Decision::Upload);
        policy.mark_uploaded(&same, reopened);
        assert_eq!(
            policy.decide(&same, reopened + Duration::from_secs(1)),
            Decision::Nothing
        );
    }

    #[test]
    fn allows_attempt_uses_the_same_gap_as_decide() {
        let interval = Duration::from_secs(60);
        let mut policy = Policy::new(interval);
        let t0 = origin();
        assert!(policy.allows_attempt(t0));
        policy.mark_attempted(t0);
        assert!(!policy.allows_attempt(t0 + interval - Duration::from_nanos(1)));
        assert!(policy.allows_attempt(t0 + interval));
        policy.mark_uploaded(&view(0), t0 + interval);
        let sent = t0 + interval;
        assert!(!policy.allows_attempt(sent + interval - Duration::from_nanos(1)));
        assert!(policy.allows_attempt(sent + interval));
    }
}
