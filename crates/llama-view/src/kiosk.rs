//! `--kiosk` (#104): llama-view as the only program on a dedicated console
//! under kmscon, which respawns it the instant it exits. Two things the
//! rest of the crate does not need to know about: the keyboard is ignored
//! (that lives in [`crate::session::Session::input`]), and a missing or
//! unreadable console source is not fatal. `llama-watch`'s unit frees tty11
//! for a moment on every restart (`TTYVTDisallocate`), opening or reading
//! `/dev/vcsa11` then fails with `ENXIO`; exiting would just make kmscon
//! spin in a tight respawn loop instead.
//!
//! [`SourceWatch`] is the retry/wait state machine: pure and driven by a
//! `now` the caller passes in, so a test can walk it through time without
//! sleeping.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::pane::sanitize;

/// How often to retry opening the console source while it is down.
pub const RETRY_PERIOD: Duration = Duration::from_secs(2);

/// Longest reason shown in the status line, after [`sanitize`].
pub const REASON_CAP: usize = 48;

/// Whether to turn focus reporting on (`ESC[?1004h`): only when stdin is a
/// terminal, and never in kiosk mode. Kiosk ignores every byte it reads
/// (including a focus report), and there is nothing else on the console to
/// hand focus to.
pub fn wants_focus_reports(kiosk: bool, stdin_is_tty: bool) -> bool {
    stdin_is_tty && !kiosk
}

/// Names what main opened, for the status line: `tty<N>` from `--vt`, or
/// the `--device` path when one was given.
pub fn source_name(vt: u8, device: Option<&Path>) -> String {
    match device {
        Some(path) => path.display().to_string(),
        None => format!("tty{vt}"),
    }
}

/// Whether the console source is readable, and when to retry it next.
#[derive(Clone, Debug, Default)]
pub struct SourceWatch {
    down: Option<Down>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Down {
    line: String,
    next_attempt: Instant,
}

impl SourceWatch {
    /// Whether the source is currently considered down.
    pub fn is_down(&self) -> bool {
        self.down.is_some()
    }

    /// Whether an attempt to (re)open the source is due: immediately the
    /// first time, then every [`RETRY_PERIOD`] after a failure.
    pub fn due(&self, now: Instant) -> bool {
        self.down
            .as_ref()
            .is_none_or(|down| now >= down.next_attempt)
    }

    /// Record a failed attempt to open or read `source` (as named by
    /// [`source_name`]) because of `reason` (typically an error's
    /// `Display`). `reason` is sanitised to printable ASCII and capped at
    /// [`REASON_CAP`] here, so a raw OS error string can be passed straight
    /// through. Returns the status line to show; it stays the same text
    /// across retries that fail the same way, so the caller's renderer sees
    /// an unchanged frame and does not flicker.
    pub fn failed(&mut self, source: &str, reason: &str, now: Instant) -> &str {
        let line = format!(
            "llama-view: waiting for {source} \u{2014} {}",
            sanitize(reason, REASON_CAP)
        );
        self.down = Some(Down {
            line,
            next_attempt: now + RETRY_PERIOD,
        });
        &self.down.as_ref().expect("just set").line
    }

    /// Clear the down state. `true` when it had been down, so the caller
    /// knows to force a full repaint (the renderer must not think the
    /// status screen is still what is drawn).
    pub fn recovered(&mut self) -> bool {
        self.down.take().is_some()
    }

    /// The status line while down, if any.
    pub fn status_line(&self) -> Option<&str> {
        self.down.as_ref().map(|down| down.line.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focus_reports_only_outside_kiosk_and_with_a_tty() {
        assert!(wants_focus_reports(false, true));
        assert!(!wants_focus_reports(false, false));
        assert!(
            !wants_focus_reports(true, true),
            "kiosk never reports focus"
        );
        assert!(!wants_focus_reports(true, false));
    }

    #[test]
    fn source_name_prefers_the_device_path() {
        assert_eq!(source_name(11, None), "tty11");
        assert_eq!(
            source_name(11, Some(Path::new("/tmp/fix/vcsa11"))),
            "/tmp/fix/vcsa11"
        );
    }

    #[test]
    fn a_fresh_watch_is_up_and_due_immediately() {
        let watch = SourceWatch::default();
        assert!(!watch.is_down());
        assert!(watch.status_line().is_none());
        assert!(watch.due(Instant::now()), "first attempt is never delayed");
    }

    #[test]
    fn failed_sets_a_stable_line_and_a_retry_timer() {
        let mut watch = SourceWatch::default();
        let t0 = Instant::now();
        let line = watch
            .failed("tty11", "No such device or address (os error 6)", t0)
            .to_string();
        assert_eq!(
            line,
            "llama-view: waiting for tty11 \u{2014} No such device or address (os error 6)"
        );
        assert!(watch.is_down());
        assert!(!watch.due(t0), "not due right after a failure");
        assert!(!watch.due(t0 + Duration::from_millis(500)));
        assert!(watch.due(t0 + RETRY_PERIOD), "due once the period passes");

        // Retrying with the same reason keeps the same line (no flicker).
        let t1 = t0 + RETRY_PERIOD;
        let again = watch.failed("tty11", "No such device or address (os error 6)", t1);
        assert_eq!(again, line);
        assert!(!watch.due(t1), "the timer restarted from t1");
        assert!(watch.due(t1 + RETRY_PERIOD));
    }

    #[test]
    fn reason_is_sanitised_and_capped() {
        let mut watch = SourceWatch::default();
        let dirty = format!("line one\nline two\x1b[31m{}", "x".repeat(100));
        let line = watch.failed("tty11", &dirty, Instant::now());
        assert!(!line.contains('\n'), "{line:?}");
        assert!(!line.contains('\x1b'), "{line:?}");
        // "llama-view: waiting for tty11 — " plus at most REASON_CAP chars.
        let prefix = "llama-view: waiting for tty11 \u{2014} ";
        assert!(line.starts_with(prefix), "{line:?}");
        assert!(
            line[prefix.len()..].chars().count() <= REASON_CAP,
            "{line:?}"
        );
    }

    #[test]
    fn recovered_is_true_only_once() {
        let mut watch = SourceWatch::default();
        assert!(!watch.recovered(), "never down");
        watch.failed("tty11", "gone", Instant::now());
        assert!(watch.recovered());
        assert!(!watch.is_down());
        assert!(!watch.recovered(), "already up");
    }
}
