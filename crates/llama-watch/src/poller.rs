//! One thread polling llama-swap into a [`LlamaView`] and a [`LlamaDetail`].
//!
//! The thread owns a single [`ureq::Agent`] (`proxy(None)`, `max_redirects(0)`,
//! per-call `timeout_global`) and uses it for every tap, including `/running`
//! via [`crate::sources::llamaswap::read_with`]. One `/running` body supplies
//! each model's raw id and state. Tails use `input_tail_chars` and
//! `output_tail_chars`.
//! With `tty.show_text = false` the `/slots` bodies are parsed for numbers
//! only: no prompt or generated text is kept (RR-LV1).
//!
//! The taps run in series on this thread. With eight ready models the worst
//! case is one `/running` timeout, eight `/metrics` timeouts, eight `/slots`
//! timeouts, and one activity timeout. At the default timeouts that is
//! 0.25 + 8×0.2 + 8×0.5 + 0.25 = 6.1 s when every call hangs. The consumer
//! calls [`SampleRx::take`], which does not wait, so that hang cannot stall
//! the main loop.
//!
//! Samples leave in a one-deep slot. A new sample replaces an unread one, so
//! a stalled consumer holds exactly the newest publish and nothing older.
//!
//! The thread never touches the console or the snapshot file.

use std::collections::HashMap;
use std::io::Read;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use llama_core::log::{self, Priority, Sink};
use llama_core::names::sanitize;
use llama_core::sample::{AiState, LlamaView, ModelInfo};
use llama_core::wire::CANONICAL_NAME_CHARS;

use crate::activity::{self, ActivityRow};
use crate::config::{PromptView, ValidWatchConfig};
use crate::metrics::{DecodedCounter, GenRate, parse_metrics};
use crate::slots::{SlotBook, SlotView};
use crate::sources::llamaswap;

const RUNNING_CAP: usize = 64 * 1024;
const METRICS_CAP: usize = 64 * 1024;
const ACTIVITY_CAP: usize = 256 * 1024;
const MODEL_PLACEHOLDER: &str = "model";

/// How long the last attempt at each tap took.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PollLatencies {
    /// `/running`, including the model-id fetch when that runs.
    pub running: Option<Duration>,
    /// `/upstream/<model>/metrics` phase.
    pub metrics: Option<Duration>,
    /// `/upstream/<model>/slots` phase.
    pub slots: Option<Duration>,
    /// `/api/metrics/activity`.
    pub activity: Option<Duration>,
}

/// TTY-facing llama detail. Text leaves the poller only as sanitised cells.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaDetail {
    /// Per-slot numbers and sanitised tails.
    pub slots: Vec<SlotView>,
    /// Newest eight activity rows.
    pub activity: Vec<ActivityRow>,
    /// Generation tok/s from `decoded_total` over the last second.
    pub gen_tps: Option<f64>,
    /// Prompt tok/s from `/slots`, or zero when nothing is in flight.
    pub prompt_tps: Option<f64>,
    /// Last poll durations.
    pub latencies: PollLatencies,
}

/// Latest sample from the poller. [`Self::put`] replaces an unread value.
///
/// The poller thread is the only other owner of the slot. [`Self::try_recv`]
/// treats `strong_count == 1` as "this consumer handle was dropped". The type
/// is not cloneable: an extra consumer owner would keep that count above 1
/// and the thread would not stop.
pub struct SampleRx {
    slot: Arc<Mutex<Option<(LlamaView, LlamaDetail)>>>,
}

impl SampleRx {
    /// Empty slot.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slot: Arc::new(Mutex::new(None)),
        }
    }

    /// Store `item`, dropping any sample the consumer has not taken.
    ///
    /// The previous sample is released after the guard, so its drop does not
    /// run while the lock is held. Callers build `item` before calling.
    pub fn put(&self, item: (LlamaView, LlamaDetail)) {
        let old = self.lock().replace(item);
        drop(old);
    }

    /// Second owner for the poller thread. Not a public [`Clone`].
    fn share_with_thread(&self) -> Self {
        Self {
            slot: Arc::clone(&self.slot),
        }
    }

    /// Take the held sample, leaving the slot empty.
    pub fn take(&self) -> Option<(LlamaView, LlamaDetail)> {
        self.lock().take()
    }

    /// [`Self::take`], or [`TryRecvError::Empty`] / [`TryRecvError::Disconnected`].
    ///
    /// Disconnected means every producer clone has been dropped and the slot
    /// is empty.
    pub fn try_recv(&self) -> Result<(LlamaView, LlamaDetail), TryRecvError> {
        if let Some(item) = self.take() {
            return Ok(item);
        }
        if Arc::strong_count(&self.slot) == 1 {
            Err(TryRecvError::Disconnected)
        } else {
            Err(TryRecvError::Empty)
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<(LlamaView, LlamaDetail)>> {
        self.slot.lock().unwrap_or_else(|err| err.into_inner())
    }
}

impl Default for SampleRx {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for the poller thread. Drop stops the thread and joins it.
pub struct Poller {
    stop: Sender<()>,
    join: Option<JoinHandle<()>>,
}

impl Drop for Poller {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(join) = self.join.take()
            && let Err(panic) = join.join()
            && !thread::panicking()
        {
            std::panic::resume_unwind(panic);
        }
    }
}

/// Start the poller. [`SampleRx::take`] yields the newest `(LlamaView, LlamaDetail)`.
///
/// Drop the [`Poller`] to stop the thread. An in-flight HTTP call finishes
/// within its own timeout before the thread exits.
pub fn spawn(
    config: &ValidWatchConfig,
    log: impl Sink + Send + 'static,
) -> std::io::Result<(Poller, SampleRx)> {
    let limits = Limits::from_config(config);
    let rx = SampleRx::new();
    let tx = rx.share_with_thread();
    let (stop_tx, stop_rx) = mpsc::channel();
    let agent = llamaswap::new_agent();
    let join = thread::Builder::new()
        .name("llama-poller".to_owned())
        .spawn(move || run(limits, agent, log, tx, stop_rx))?;
    Ok((
        Poller {
            stop: stop_tx,
            join: Some(join),
        },
        rx,
    ))
}

struct Limits {
    url: String,
    running_interval: Duration,
    running_timeout: Duration,
    metrics_interval: Duration,
    metrics_timeout: Duration,
    slots_interval: Duration,
    slots_timeout: Duration,
    slots_cap: usize,
    activity_interval: Duration,
    activity_timeout: Duration,
    input_tail: usize,
    output_tail: usize,
    /// `tty.show_text`. False keeps no slot text and more activity rows.
    show_text: bool,
    /// `tty.prompt_view`.
    prompt_view: PromptView,
    max_name_chars: usize,
    aliases: HashMap<String, String>,
}

impl Limits {
    fn from_config(config: &ValidWatchConfig) -> Self {
        Self {
            url: config.llama.url.clone(),
            running_interval: secs(config.llama.running_interval_s),
            running_timeout: secs(config.llama.running_timeout_s),
            metrics_interval: secs(config.llama.metrics_interval_s),
            metrics_timeout: secs(config.llama.metrics_timeout_s),
            slots_interval: secs(config.llama.slots_interval_s),
            slots_timeout: secs(config.llama.slots_timeout_s),
            slots_cap: usize::try_from(config.llama.slots_max_bytes).unwrap_or(RUNNING_CAP),
            activity_interval: secs(config.llama.activity_interval_s),
            activity_timeout: secs(config.llama.activity_timeout_s),
            input_tail: config.llama.input_tail_chars as usize,
            output_tail: config.llama.output_tail_chars as usize,
            show_text: config.tty.show_text,
            prompt_view: config.tty.prompt_view,
            max_name_chars: config.models.max_name_chars as usize,
            aliases: config
                .models
                .aliases
                .iter()
                .map(|(model, name)| (model.clone(), name.clone()))
                .collect(),
        }
    }
}

#[derive(Clone)]
struct ReadyModel {
    id: String,
    name: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Reach {
    Unknown,
    Up,
    Down(&'static str),
}

struct State<L> {
    limits: Limits,
    agent: ureq::Agent,
    log: L,
    tx: SampleRx,
    counter: DecodedCounter,
    gen_rate: GenRate,
    slots: SlotBook,
    activity: Vec<ActivityRow>,
    ai: AiState,
    models: Vec<ModelInfo>,
    ready: Vec<ReadyModel>,
    running_up: bool,
    unmetered: bool,
    latencies: PollLatencies,
    reach: Reach,
    slots_oversize: bool,
    metrics_failed: bool,
    slots_failed: bool,
    activity_failed: bool,
}

fn run<L: Sink>(limits: Limits, agent: ureq::Agent, log: L, tx: SampleRx, stop: Receiver<()>) {
    let slots = if limits.show_text {
        SlotBook::with_prompt_view(limits.prompt_view)
    } else {
        SlotBook::without_text()
    };
    let mut state = State {
        limits,
        agent,
        log,
        tx,
        counter: DecodedCounter::default(),
        gen_rate: GenRate::default(),
        slots,
        activity: Vec::new(),
        ai: AiState::Down,
        models: Vec::new(),
        ready: Vec::new(),
        running_up: false,
        unmetered: false,
        latencies: PollLatencies::default(),
        reach: Reach::Unknown,
        slots_oversize: false,
        metrics_failed: false,
        slots_failed: false,
        activity_failed: false,
    };
    let mut next_running = Instant::now();
    let mut next_metrics = Instant::now();
    let mut next_slots = Instant::now();
    let mut next_activity = Instant::now();
    loop {
        if stopped(&stop) {
            break;
        }
        let now = Instant::now();
        let mut worked = false;
        if now >= next_running {
            state.poll_running();
            next_running = Instant::now() + state.limits.running_interval;
            worked = true;
        }
        if now >= next_metrics {
            state.poll_metrics();
            next_metrics = Instant::now() + state.limits.metrics_interval;
            worked = true;
        }
        if now >= next_slots {
            state.poll_slots();
            next_slots = Instant::now() + state.limits.slots_interval;
            worked = true;
        }
        if now >= next_activity {
            state.poll_activity();
            next_activity = Instant::now() + state.limits.activity_interval;
            worked = true;
        }
        if worked && state.publish().is_err() {
            break;
        }
        let mut next = next_running;
        for candidate in [next_metrics, next_slots, next_activity] {
            if candidate < next {
                next = candidate;
            }
        }
        let wait = next.saturating_duration_since(Instant::now());
        match stop.recv_timeout(wait) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}

impl<L: Sink> State<L> {
    fn poll_running(&mut self) {
        let started = Instant::now();
        let reading = llamaswap::read_with(
            &self.agent,
            &self.limits.url,
            self.limits.running_timeout,
            &self.limits.aliases,
        );
        let (ai, down) = match reading.ai {
            llamaswap::RunningStatus::Down(reason) => (AiState::Down, Some(reason)),
            llamaswap::RunningStatus::Idle => (AiState::Idle, None),
            llamaswap::RunningStatus::Loaded => (AiState::Loaded, None),
        };
        self.note_reach(down);
        self.ai = ai;
        self.running_up = down.is_none();
        self.models = if self.running_up {
            reading
                .models
                .iter()
                .map(|info| ModelInfo {
                    name: snapshot_name(&info.name, self.limits.max_name_chars),
                    state: info.state.clone(),
                    full_name: full_name(&info.full_name, self.limits.max_name_chars),
                    detail: info.detail.clone(),
                })
                .collect()
        } else {
            Vec::new()
        };

        self.ready.clear();
        self.unmetered = false;
        if self.running_up {
            for info in &reading.models {
                if info.state != "ready" {
                    continue;
                }
                let name = snapshot_name(&info.name, self.limits.max_name_chars);
                match upstream_id(&info.id) {
                    Some(id) => self.ready.push(ReadyModel { id, name }),
                    None => self.unmetered = true,
                }
            }
        }
        if !self.running_up || self.ready.is_empty() {
            self.slots.clear();
        } else {
            let ids: Vec<&str> = self.ready.iter().map(|model| model.id.as_str()).collect();
            self.slots.retain_models(&ids);
        }
        self.latencies.running = Some(started.elapsed());
    }

    fn poll_metrics(&mut self) {
        if self.ready.is_empty() {
            self.latencies.metrics = None;
            return;
        }
        let started = Instant::now();
        let ready = self.ready.clone();
        let mut failure: Option<&'static str> = None;
        for model in &ready {
            let url = upstream(&self.limits.url, &model.id, "metrics");
            match get_exact(&self.agent, &url, self.limits.metrics_timeout, METRICS_CAP) {
                Ok(bytes) => match std::str::from_utf8(&bytes) {
                    Ok(text) => {
                        let sample = parse_metrics(text);
                        if let Some(value) = sample.n_decode_total {
                            self.counter.observe(
                                &model.id,
                                value,
                                sample.requests_processing,
                                Instant::now(),
                            );
                        } else if failure.is_none() {
                            failure = Some("malformed");
                        }
                    }
                    Err(_) => {
                        if failure.is_none() {
                            failure = Some("malformed");
                        }
                    }
                },
                Err(err) => {
                    if failure.is_none() {
                        failure = Some(err.label());
                    }
                }
            }
        }
        note_flag(&mut self.log, &mut self.metrics_failed, failure, "metrics");
        self.latencies.metrics = Some(started.elapsed());
    }

    fn poll_slots(&mut self) {
        let now = Instant::now();
        let busy: Vec<ReadyModel> = self
            .ready
            .iter()
            .filter(|model| {
                self.counter
                    .requests_processing(&model.id, now)
                    .is_some_and(|value| value > 0.0)
            })
            .cloned()
            .collect();
        if busy.is_empty() {
            self.latencies.slots = None;
            return;
        }
        let started = Instant::now();
        self.slots.begin_round();
        let mut failure: Option<&'static str> = None;
        let mut oversize = false;
        for model in &busy {
            let url = upstream(&self.limits.url, &model.id, "slots");
            match get_limited(
                &self.agent,
                &url,
                self.limits.slots_timeout,
                self.limits.slots_cap,
            ) {
                Ok(Limited::Exact(bytes)) => {
                    if !self.slots.apply(
                        &model.id,
                        &model.name,
                        &bytes,
                        self.limits.input_tail,
                        self.limits.output_tail,
                    ) && failure.is_none()
                    {
                        failure = Some("malformed");
                    }
                }
                Ok(Limited::Oversize) => {
                    oversize = true;
                }
                Err(err) => {
                    if failure.is_none() {
                        failure = Some(err.label());
                    }
                }
            }
        }
        self.slots.finish_round(Instant::now());
        if oversize {
            if !self.slots_oversize {
                log::emit(&mut self.log, Priority::Warning, "slots: body exceeds cap");
                self.slots_oversize = true;
            }
        } else {
            self.slots_oversize = false;
        }
        note_flag(&mut self.log, &mut self.slots_failed, failure, "slots");
        self.latencies.slots = Some(started.elapsed());
    }

    fn poll_activity(&mut self) {
        if !self.running_up {
            self.latencies.activity = None;
            return;
        }
        let started = Instant::now();
        let url = join_url(&self.limits.url, "api/metrics/activity");
        let failure = match get_exact(
            &self.agent,
            &url,
            self.limits.activity_timeout,
            ACTIVITY_CAP,
        ) {
            Ok(bytes) => match activity::parse_activity_rows(&bytes, self.activity_rows()) {
                Some(rows) => {
                    self.activity = rows;
                    None
                }
                None => Some("malformed"),
            },
            Err(err) => Some(err.label()),
        };
        note_flag(
            &mut self.log,
            &mut self.activity_failed,
            failure,
            "activity",
        );
        self.latencies.activity = Some(started.elapsed());
    }

    /// RECENT rows worth keeping: eight with the text panels, more without.
    fn activity_rows(&self) -> usize {
        if self.limits.show_text {
            activity::ROWS
        } else {
            usize::from(crate::tty::layout::RECENT_ROWS_TEXT_OFF)
        }
    }

    fn publish(&mut self) -> Result<(), ()> {
        let now = Instant::now();
        let ready_ids: Vec<&str> = self.ready.iter().map(|model| model.id.as_str()).collect();
        let decoded = if self.unmetered {
            None
        } else {
            self.counter
                .total_if_fresh(self.running_up, &ready_ids, now)
        };
        let gen_tps = if let Some(total) = decoded {
            self.gen_rate.observe(now, total)
        } else {
            self.gen_rate = GenRate::default();
            None
        };
        let busy = self.ready.iter().any(|model| {
            self.counter
                .requests_processing(&model.id, now)
                .is_some_and(|value| value > 0.0)
        });
        let prompt_tps = if !self.running_up {
            None
        } else if !busy {
            Some(0.0)
        } else {
            self.slots.prompt_tps()
        };
        let view = LlamaView {
            ai: self.ai,
            models: self.models.clone(),
            decoded_total: decoded,
        };
        let detail = LlamaDetail {
            slots: self.slots.slots(),
            activity: self.activity.clone(),
            gen_tps,
            prompt_tps,
            latencies: self.latencies,
        };
        if Arc::strong_count(&self.tx.slot) == 1 {
            return Err(());
        }
        self.tx.put((view, detail));
        Ok(())
    }

    fn note_reach(&mut self, down: Option<&'static str>) {
        let next = match down {
            Some(reason) => Reach::Down(reason),
            None => Reach::Up,
        };
        if self.reach == next {
            return;
        }
        match next {
            Reach::Down(reason) => log::emit(
                &mut self.log,
                Priority::Err,
                &format!("llama: down ({reason})"),
            ),
            Reach::Up => log::emit(&mut self.log, Priority::Info, "llama: reachable"),
            Reach::Unknown => {}
        }
        self.reach = next;
    }
}

fn note_flag<L: Sink>(
    log: &mut L,
    flag: &mut bool,
    failure: Option<&'static str>,
    tap: &'static str,
) {
    match failure {
        Some(reason) => {
            if !*flag {
                log::emit(log, Priority::Warning, &format!("{tap}: {reason}"));
                *flag = true;
            }
        }
        None => {
            if *flag {
                log::emit(log, Priority::Info, &format!("{tap}: recovered"));
            }
            *flag = false;
        }
    }
}

enum Limited {
    Exact(Vec<u8>),
    /// Body was longer than the cap. The bytes are dropped, not parsed.
    Oversize,
}

enum TapError {
    Timeout,
    Refused,
    Status,
    Oversize,
    Failed,
}

impl TapError {
    fn label(self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::Refused => "connection refused",
            Self::Status => "http status",
            Self::Oversize => "oversized body",
            Self::Failed => "request failed",
        }
    }
}

fn get_exact(
    agent: &ureq::Agent,
    url: &str,
    timeout: Duration,
    cap: usize,
) -> Result<Vec<u8>, TapError> {
    match get_limited(agent, url, timeout, cap)? {
        Limited::Exact(bytes) => Ok(bytes),
        Limited::Oversize => Err(TapError::Oversize),
    }
}

fn get_limited(
    agent: &ureq::Agent,
    url: &str,
    timeout: Duration,
    cap: usize,
) -> Result<Limited, TapError> {
    let mut response = agent
        .get(url)
        .config()
        .timeout_global(Some(timeout))
        .build()
        .call()
        .map_err(classify)?;
    if response.status().as_u16() != 200 {
        return Err(TapError::Status);
    }
    let mut buf = Vec::new();
    let mut reader = response
        .body_mut()
        .as_reader()
        .take((cap as u64).saturating_add(1));
    reader
        .read_to_end(&mut buf)
        .map_err(|err| classify_io(&err))?;
    if buf.len() > cap {
        Ok(Limited::Oversize)
    } else {
        Ok(Limited::Exact(buf))
    }
}

fn classify(err: ureq::Error) -> TapError {
    match err {
        ureq::Error::Timeout(_) => TapError::Timeout,
        ureq::Error::ConnectionFailed => TapError::Refused,
        ureq::Error::Io(io) => classify_io(&io),
        _ => TapError::Failed,
    }
}

fn classify_io(err: &std::io::Error) -> TapError {
    match err.kind() {
        std::io::ErrorKind::TimedOut => TapError::Timeout,
        std::io::ErrorKind::ConnectionRefused => TapError::Refused,
        _ => TapError::Failed,
    }
}

fn join_url(base: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

fn upstream(base: &str, model: &str, leaf: &str) -> String {
    join_url(base, &format!("upstream/{model}/{leaf}"))
}

/// Ids that are a single safe path segment. `.` and `..` are rejected.
fn upstream_id(id: &str) -> Option<String> {
    if id.is_empty() || id == "." || id == ".." {
        return None;
    }
    if id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~'))
    {
        Some(id.to_owned())
    } else {
        None
    }
}

/// Narrow a wire name to the config width. An empty result is `model`,
/// truncated to that same width.
/// The untruncated label, unless `models.max_name_chars` asks for less than
/// the canonical width or the label is empty.
fn full_name(from_read: &str, max_name_chars: usize) -> Option<String> {
    (max_name_chars >= CANONICAL_NAME_CHARS && !from_read.is_empty()).then(|| from_read.to_owned())
}

fn snapshot_name(from_read: &str, max_name_chars: usize) -> String {
    let width = max_name_chars.min(CANONICAL_NAME_CHARS);
    let name = if from_read.is_empty() {
        String::new()
    } else if width >= CANONICAL_NAME_CHARS {
        from_read.to_owned()
    } else {
        sanitize(from_read, width)
    };
    if name.is_empty() {
        let placeholder = sanitize(MODEL_PLACEHOLDER, width);
        if placeholder.is_empty() {
            MODEL_PLACEHOLDER.to_owned()
        } else {
            placeholder
        }
    } else {
        name
    }
}

fn secs(value: f64) -> Duration {
    Duration::try_from_secs_f64(value).unwrap_or(Duration::from_millis(1))
}

fn stopped(stop: &Receiver<()>) -> bool {
    match stop.try_recv() {
        Ok(()) | Err(TryRecvError::Disconnected) => true,
        Err(TryRecvError::Empty) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_name_is_kept_at_the_canonical_width_only() {
        assert_eq!(
            full_name("Ternary Bonsai 2 27B", 12).as_deref(),
            Some("Ternary Bonsai 2 27B")
        );
        assert_eq!(full_name("Ternary Bonsai 2 27B", 8), None);
        assert_eq!(full_name("", 12), None);
    }

    #[test]
    fn poller_upstream_id_rejects_dot_dot() {
        assert_eq!(
            upstream_id("qwen3.6-35b-a3b").as_deref(),
            Some("qwen3.6-35b-a3b")
        );
        assert!(upstream_id("..").is_none());
        assert!(upstream_id("../metrics").is_none());
        assert!(upstream_id("a b").is_none());
        assert!(upstream_id("").is_none());
        assert!(upstream_id("a/b").is_none());
    }

    #[test]
    fn poller_name_narrows_and_is_never_empty() {
        let wire = format!("abcdefghijk{}", '\u{2026}');
        assert_eq!(snapshot_name(&wire, 12), wire);
        assert_eq!(snapshot_name(&wire, 6), format!("abcde{}", '\u{2026}'));
        assert_eq!(snapshot_name("", 12), "model");
        assert_eq!(snapshot_name("", 4), format!("mod{}", '\u{2026}'));
        assert_eq!(snapshot_name("", 2), format!("m{}", '\u{2026}'));
    }
}
