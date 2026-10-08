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
//! timeouts, two `/running` timeouts around each of those sixteen (the
//! fresh gate, #70), one activity timeout and one capture timeout. At the
//! default timeouts that is 0.25 + 8×0.2 + 8×0.5 + 32×0.25 + 0.25 + 1 =
//! 15.1 s when every call hangs. The consumer calls [`SampleRx::take`], which does not wait, so that
//! hang cannot stall the main loop.
//!
//! IN and OUT for a model without `/slots` come from llama-swap's request
//! capture of its newest finished activity row (#5, [`crate::capture`]),
//! and so do a llama.cpp model's while its `/slots` carries no prompt or
//! generated text (#66: current llama-server sends them only with
//! `LLAMA_SERVER_SLOTS_DEBUG=1`). That is re-checked on every `/slots`
//! read, so text that appears switches IN and OUT back to live; the
//! numbers keep coming from `/slots` either way. Captures are
//! `GET /api/captures/<id>` once per new row id, only with
//! `tty.show_text = true`, only for a row marked `has_capture`, at most
//! [`CAPTURE_CAP`] bytes. The capture's bodies become sanitised tails here
//! and are dropped.
//!
//! Samples leave in a one-deep slot. A new sample replaces an unread one, so
//! a stalled consumer holds exactly the newest publish and nothing older.
//!
//! Each ready model has a [`Backend`] (T72), from its launch command or
//! `[llama.backends]`. Only llama.cpp gets `/slots`. SGLang and vLLM get the
//! same `/metrics` GET with their own names; a model with no usable
//! `/metrics` (and any OpenAI-compatible server) falls back to llama-swap's
//! activity rows: each finished request adds its `output_tokens` to the
//! decoded counter once, so the counter still moves, at request end.
//! Strata's `/metrics` is JSON ([`crate::metrics::parse_strata`]); its
//! `engine` facts fill the model's ctx and KV, which its launch command
//! cannot give, and SETUP's `engine:<key>` values; its `live` report (the
//! phase, prompt progress, tok/s) rides to tty11 in
//! [`LlamaDetail::engine_live`] while its gauges are fresh (#54). A 401 or 403 (Strata started with an API key; llama-bored
//! keeps no secrets) falls back like any missing `/metrics`.
//!
//! A model whose launch command names no server (`openai`: a container
//! whose image runs `vllm serve`, a wrapper script around llama-server) and
//! that `[llama.backends]` does not name gets one `/upstream/<id>/metrics`
//! GET per load, with the usual cap and timeout (#31). A JSON object with
//! `engine` and `live` objects is Strata (#54, a container started by image
//! digest); else the first sample's metric prefix decides: `vllm:`,
//! `sglang:` or `llamacpp:` (which then also gets `/slots`); anything else,
//! or no `/metrics`, stays `openai`. Each outcome is logged once. A probe
//! that names a server re-reads `/running` at once, and that read and each
//! one after it pass the server to the launch command parser, which reads
//! a container's flags after its image (#67).
//!
//! llama-swap loads (or swaps in) a model on any `/upstream/<id>/…` request
//! for a model that is not loaded, so polling must never make one (#31,
//! #70). Every upstream GET, the probe and `/slots` included, goes through
//! [`State::upstream_read`], the fresh gate: it reads `/running` itself,
//! as the request just before that one GET, and sends it only if that
//! read lists the model as `ready` and lists no model in any other state
//! (a model starting or stopping is a swap in progress, and the rest of
//! the round makes no upstream request at all). A read the gate drops is
//! not retried. A `/running` failure clears the ready list, so nothing
//! upstream is read until llama-swap answers again.
//!
//! The gate then reads `/running` again at once and checks itself: a
//! model `starting` there, or a GET slower than 2 s, is a suspected load.
//! It is logged as a warning with the model id and path, counted per
//! model on the snapshot (`suspected_loads`), and that model gets no
//! upstream reads for 5 minutes.
//!
//! A 409 (what llama-swap answers when the path is in its optional
//! `upstream.ignorePaths` and the model is not loaded), or any other
//! non-2xx answer, means "not available now": the model's engine numbers
//! are dropped (as on unload) and it gets no more reads that round. A 409
//! is logged once per change and is not a source failure.
//!
//! RECENT's PROMPT and GEN for a vLLM request come from the engine (#35,
//! [`crate::speeds`]): llama-swap gives `-1` for them, so each new row of a
//! model without `/slots` gets the prefill and decode tok/s of the requests
//! that finished between the metrics read before the previous activity
//! read and the first metrics read after the one that showed the row. A
//! row llama-swap gave real speeds keeps them.
//!
//! RECENT is llama-watch's own ring of rows ([`crate::recent`], #44), not a
//! copy of llama-swap's in-memory list: a llama-swap restart keeps it, and
//! a restart that reuses row ids (told by a lower top id, an empty list, a
//! changed fingerprint, or `/running` having been down) starts a new
//! generation whose rows are all new. Per-row state is keyed by the ring's
//! row number, and a capture is only fetched for a row of the page just
//! read.
//!
//! The thread never touches the console or the snapshot file.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use llama_core::backend::{self, Backend, BackendInfo, EngineStats};
use llama_core::log::{self, Priority, Sink};
use llama_core::names::sanitize;
use llama_core::sample::{AiState, LlamaView, ModelInfo};
use llama_core::wire::CANONICAL_NAME_CHARS;

use crate::activity::{self, ActivityRow};
use crate::capture::{CAPTURE_CAP, parse_capture};
use crate::config::{PromptView, ValidWatchConfig};
use crate::metrics::{
    DecodedCounter, EngineBook, EngineFacts, EngineLive, EngineValues, GenRate, KvTokens,
    MetricsSample, PromptCache, detect_backend, parse_metrics_full,
};
use crate::recent::{Merged, Recent};
use crate::series::{ModelSeries, SeriesBook};
use crate::setup_rules::{Found, Rules};
use crate::slots::{SlotBook, SlotView, prompt_cells, tail_cells};
use crate::sources::cmdline::KvLayout;
use crate::sources::llamaswap;
use crate::speeds::SpeedBook;
use crate::tty::grid::Cell;

const RUNNING_CAP: usize = 64 * 1024;
/// llama-server's `/metrics` is a few KiB.
const LLAMACPP_METRICS_CAP: usize = 64 * 1024;
/// SGLang and vLLM export latency histograms per label set; SGLang's is
/// about 70 KiB with one model loaded and grows with its labels.
const SERVER_METRICS_CAP: usize = 1024 * 1024;
const ACTIVITY_CAP: usize = 256 * 1024;
/// A capture read may take longer than the activity page: it can be MiBs.
const CAPTURE_TIMEOUT: Duration = Duration::from_secs(1);
const MODEL_PLACEHOLDER: &str = "model";
/// Backend gauges older than this (or two metrics periods) are not shown.
const FRESH_GAUGES: Duration = Duration::from_secs(1);
/// An upstream read slower than this is a suspected model load (#70): a
/// loaded server answers `/metrics` and `/slots` in milliseconds.
const SUSPECT_SLOW: Duration = Duration::from_secs(2);
/// No upstream reads for a model this long after a suspected load (#70).
const SUSPECT_BACKOFF: Duration = Duration::from_secs(300);
/// Most model ids whose suspected loads are counted (#70).
const MAX_SUSPECTS: usize = 8;

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
    /// RECENT: the newest activity rows llama-watch kept, newest first,
    /// across llama-swap restarts (#44). Eight with the text panels, up to
    /// [`crate::recent::RING`] without.
    pub activity: Vec<ActivityRow>,
    /// Generation tok/s from `decoded_total` over the last second.
    pub gen_tps: Option<f64>,
    /// Prompt tok/s from `/slots`, or zero when nothing is in flight.
    pub prompt_tps: Option<f64>,
    /// Last poll durations.
    pub latencies: PollLatencies,
    /// Prompt and cached-prompt counters of each ready model (#10).
    pub prompt_cache: Vec<ModelPromptCache>,
    /// The last finished exchange of a model without `/slots`, from a
    /// llama-swap capture (#5). Always `None` with `tty.show_text = false`.
    pub capture: Option<CaptureView>,
    /// What the `[setup]` rules read from each model's launch command
    /// (#52), in the order of the view's models.
    pub setup: Vec<ModelSetup>,
    /// What each ready model's engine reports it is doing now, from a
    /// fresh `/metrics` read (Strata's `live`, #54).
    pub engine_live: Vec<ModelEngineLive>,
    /// Suspected model loads this run (#70): `(llama-swap id, sanitised,
    /// count)`, at most [`MAX_SUSPECTS`] ids. Empty when there were none.
    pub suspected_loads: Vec<(String, u64)>,
    /// Each listed model's cumulative numbers for llama-metrics (#71), in
    /// the order of the view's models.
    pub series: Vec<ModelSeries>,
}

/// One ready model's [`EngineLive`] (#54).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelEngineLive {
    /// Display name, as [`SlotView::model`] and the snapshot use it.
    pub model: String,
    /// The engine's own report: numbers and a sanitised phase.
    pub live: EngineLive,
}

/// One loaded model's SETUP values (#52).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelSetup {
    /// [`activity::model_key`] of the llama-swap id, to match RECENT rows.
    pub key: String,
    /// The llama-swap id, sanitised for display.
    pub id: String,
    /// The llama-swap `name` (or alias), sanitised.
    pub name: String,
    /// Values the rules took from the launch command and the model name.
    pub found: Vec<Found>,
    /// Settings the engine reports about itself, for `engine:<key>` rules
    /// (#54). Empty for an engine that reports none.
    pub engine: EngineValues,
}

/// IN and OUT from one llama-swap capture (#5): sanitised tails only.
#[derive(Clone, Debug, PartialEq)]
pub struct CaptureView {
    /// Display name of the model that served it.
    pub model: String,
    /// The activity row id it belongs to.
    pub id: i64,
    /// The last user message, or a tool loop's newest user and tool
    /// messages (#38), as IN shows it (`tty.prompt_view`).
    pub input: Vec<Cell>,
    /// What a tool-loop IN holds (`3 tool results`), for its title; empty
    /// for a single user message. Built by llama-watch, never request text.
    pub input_note: String,
    /// The answer.
    pub output: Vec<Cell>,
    /// The model has `/slots` without text (#66): IN and OUT show this
    /// capture over its slots.
    pub over_slots: bool,
}

/// One ready model's prompt token counters since the watcher started (#10).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelPromptCache {
    /// Display name, as [`SlotView::model`] and the snapshot use it.
    pub model: String,
    /// Prompt tokens of finished requests, cached ones included.
    pub prompt: u64,
    /// Of those, served from the prompt cache. `None` when no source says.
    pub cached: Option<u64>,
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
    /// `[llama.backends]`: model id to backend, over the launch command.
    backends: HashMap<String, Backend>,
    /// `[setup]` rules read from each launch command (#52).
    setup: Rules,
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
            backends: config
                .llama
                .backends
                .iter()
                .map(|(model, kind)| (model.clone(), *kind))
                .collect(),
            setup: Rules::compile(&config.setup).unwrap_or_default(),
        }
    }
}

#[derive(Clone)]
struct ReadyModel {
    id: String,
    name: String,
    backend: Backend,
    /// The launch command named no server and `[llama.backends]` has no
    /// entry: `/metrics` decides, once per load (#31).
    probe: bool,
}

/// Live gauges from one backend `/metrics` read.
#[derive(Clone)]
struct Gauges {
    running: Option<u16>,
    queued: Option<u16>,
    kv_permille: Option<u16>,
    hit_permille: Option<u16>,
    /// KV cache tokens (#79).
    kv: KvTokens,
    engine: EngineStats,
    /// The engine's own `live` report (Strata, #54).
    live: Option<EngineLive>,
    at: Instant,
}

impl Gauges {
    fn of(
        sample: &MetricsSample,
        engine: EngineStats,
        live: Option<EngineLive>,
        at: Instant,
    ) -> Self {
        Self {
            running: sample.requests_processing.and_then(backend::reqs),
            queued: sample.queued.and_then(backend::reqs),
            kv_permille: sample.kv_fill.and_then(backend::permille),
            hit_permille: sample.cache_hit.and_then(backend::permille),
            kv: sample.kv,
            engine,
            live,
            at,
        }
    }
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
    /// RECENT's own rows across llama-swap restarts (#44).
    recent: Recent,
    ai: AiState,
    models: Vec<ModelInfo>,
    /// Raw id of each entry of `models`, in the same order.
    model_ids: Vec<String>,
    /// llama.cpp KV layout of each entry of `models`, from its launch
    /// command (#79), in the same order.
    model_kv: Vec<Option<KvLayout>>,
    /// SETUP values of each entry of `models`, in the same order (#52).
    model_setup: Vec<ModelSetup>,
    ready: Vec<ReadyModel>,
    /// While `/running` fails: the models ready at the last good read and
    /// their servers, for the first good read after it (#46).
    down_ready: Vec<(String, Backend)>,
    /// Gauges per model id from the last backend `/metrics` read.
    gauges: HashMap<String, Gauges>,
    /// Engine facts per ready model id from its last `/metrics` read (Strata).
    engine_facts: HashMap<String, EngineFacts>,
    /// Ready model ids counted from activity rows instead of `/metrics`.
    fallback: HashSet<String>,
    /// Model ids whose missing `/metrics` was logged this run.
    no_metrics_logged: HashSet<String>,
    /// SGLang/vLLM prompt-token counter, for prompt tok/s without `/slots`.
    prompt_counter: DecodedCounter,
    prompt_rate: GenRate,
    /// Box prompt-token total for the snapshot (#11): every metered model's
    /// prompt counter, or its activity rows' `input_tokens` in fallback.
    prompt_box: DecodedCounter,
    /// Per-model prompt and cached-prompt counters (#10).
    prompt_cache: PromptCache,
    /// Per-model engine numbers: spec decoding, preemptions, latency (#31).
    engines: EngineBook,
    /// Engine-measured speeds given to RECENT rows (#35).
    speeds: SpeedBook,
    /// Per-model cumulative numbers for llama-metrics (#71).
    series: SeriesBook,
    /// Server kind `/metrics` gave each loaded model the launch command did
    /// not name (#31); `openai` when it gave none. Dropped when the model
    /// leaves `ready`, so the next load is probed again.
    detected: HashMap<String, Backend>,
    /// `id:kind` probe outcomes already logged this run.
    detect_logged: HashSet<String>,
    /// A probe found a server: read `/running` again at once, so the
    /// launch command's flags are read as that server's (#67).
    running_due: bool,
    running_up: bool,
    unmetered: bool,
    latencies: PollLatencies,
    reach: Reach,
    slots_oversize: bool,
    metrics_failed: bool,
    slots_failed: bool,
    activity_failed: bool,
    /// The `(llama-swap generation, id)` of the newest activity row a
    /// capture was considered for (#5, #44).
    capture_seen: Option<(u32, i64)>,
    /// The capture shown, with the raw id of its model.
    capture: Option<(String, CaptureView)>,
    /// Ready llama.cpp models whose `/slots` carries no text: IN and OUT
    /// come from their captures (#66).
    slots_textless: HashSet<String>,
    /// Models whose textless `/slots` was logged this run.
    textless_logged: HashSet<String>,
    capture_failed: bool,
    capture_oversize: bool,
    /// Models that answered an upstream read with a non-2xx this round
    /// (#70): no more upstream reads for them until the next round.
    round_skip: HashSet<String>,
    /// Ready models whose last upstream read got llama-swap's 409 "not
    /// loaded" (#70), so it is logged once until a read succeeds or the
    /// model leaves `ready`.
    not_loaded: HashSet<String>,
    /// The last good `/running` read listed a model in a state other than
    /// `ready`: llama-swap is loading or unloading one (#70).
    swapping: bool,
    /// A gate read this round saw a swap: no more upstream reads until the
    /// next round (#70).
    round_blocked: bool,
    /// Models with no upstream reads until the given time, after a
    /// suspected load (#70).
    backoff: HashMap<String, Instant>,
    /// Suspected loads per model id this run (#70), at most
    /// [`MAX_SUSPECTS`] ids.
    suspected: Vec<(String, u64)>,
}

/// What [`State::upstream_read`] did.
enum Gate {
    /// No GET: the fresh `/running` read does not list the model as
    /// `ready`, or it already had a non-2xx this round.
    Skipped,
    /// llama-swap answered 409: the model is not loaded. Already handled.
    NotLoaded,
    /// The GET was made. The model is as the fresh `/running` read gave it.
    Read(ReadyModel, Result<Limited, TapError>),
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
        recent: Recent::default(),
        ai: AiState::Down,
        models: Vec::new(),
        model_ids: Vec::new(),
        model_kv: Vec::new(),
        model_setup: Vec::new(),
        ready: Vec::new(),
        down_ready: Vec::new(),
        gauges: HashMap::new(),
        engine_facts: HashMap::new(),
        fallback: HashSet::new(),
        no_metrics_logged: HashSet::new(),
        prompt_counter: DecodedCounter::default(),
        prompt_rate: GenRate::default(),
        prompt_box: DecodedCounter::default(),
        prompt_cache: PromptCache::default(),
        engines: EngineBook::default(),
        speeds: SpeedBook::default(),
        series: SeriesBook::default(),
        detected: HashMap::new(),
        detect_logged: HashSet::new(),
        running_due: false,
        running_up: false,
        unmetered: false,
        latencies: PollLatencies::default(),
        reach: Reach::Unknown,
        slots_oversize: false,
        metrics_failed: false,
        slots_failed: false,
        activity_failed: false,
        capture_seen: None,
        capture: None,
        slots_textless: HashSet::new(),
        textless_logged: HashSet::new(),
        capture_failed: false,
        capture_oversize: false,
        round_skip: HashSet::new(),
        not_loaded: HashSet::new(),
        swapping: false,
        round_blocked: false,
        backoff: HashMap::new(),
        suspected: Vec::new(),
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
        // One round of upstream reads per pass: a model that answered one
        // with a non-2xx is not read again until the next pass (#70).
        state.round_skip.clear();
        state.round_blocked = false;
        if now >= next_running || state.running_due {
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
        let mut next = if state.running_due {
            Instant::now()
        } else {
            next_running
        };
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
        self.running_due = false;
        // Each model's server where the config or this load's probe named
        // one, so a container's flags are read as that server's (#67).
        let mut known = self.limits.backends.clone();
        for (id, kind) in &self.detected {
            if *kind != Backend::OpenAi {
                known.entry(id.clone()).or_insert(*kind);
            }
        }
        let reading = llamaswap::read_with(
            &self.agent,
            &self.limits.url,
            self.limits.running_timeout,
            &self.limits.aliases,
            &self.limits.setup,
            &known,
        );
        let (ai, down) = match reading.ai {
            llamaswap::RunningStatus::Down(reason) => (AiState::Down, Some(reason)),
            llamaswap::RunningStatus::Idle => (AiState::Idle, None),
            llamaswap::RunningStatus::Loaded => (AiState::Loaded, None),
        };
        self.note_reach(down);
        self.ai = ai;
        self.running_up = down.is_none();
        // Any model starting, stopping or in another state: a swap is in
        // progress, and no upstream read is made until it is over (#70).
        self.swapping = self.running_up && reading.models.iter().any(|info| info.state != "ready");
        if !self.running_up {
            self.recent.note_down();
        }
        let models: &[llamaswap::RunningModel] = if self.running_up {
            &reading.models
        } else {
            &[]
        };
        self.models = models
            .iter()
            .map(|info| ModelInfo {
                name: snapshot_name(&info.name, self.limits.max_name_chars),
                state: info.state.clone(),
                full_name: full_name(&info.full_name, self.limits.max_name_chars),
                detail: info.detail.clone(),
                backend: Some(BackendInfo {
                    kind: self.backend_of(info),
                    max_running: info.max_running.or_else(|| {
                        // Strata serves one request at a time, whichever
                        // way it was told (#54).
                        (self.backend_of(info) == Backend::Strata).then_some(1)
                    }),
                    ..BackendInfo::default()
                }),
            })
            .collect();
        self.model_ids = models.iter().map(|info| info.id.clone()).collect();
        self.model_kv = models.iter().map(|info| info.kv).collect();
        self.model_setup = models
            .iter()
            .map(|info| ModelSetup {
                key: activity::model_key(&info.id),
                id: sanitize(&info.id, llama_core::detail::MAX_FULL_NAME_CHARS),
                name: info.full_name.clone(),
                found: info.setup.clone(),
                engine: EngineValues::default(),
            })
            .collect();

        // The models ready at the last good read, with their servers. A
        // failed read clears `ready` but keeps them in `down_ready`, so the
        // first good read after it still sees what left (#46).
        let mut was_ready: Vec<(String, Backend)> = std::mem::take(&mut self.down_ready);
        for model in &self.ready {
            if !was_ready.iter().any(|(id, _)| *id == model.id) {
                was_ready.push((model.id.clone(), model.backend));
            }
        }
        self.ready.clear();
        self.unmetered = false;
        if self.running_up {
            for info in &reading.models {
                if info.state != "ready" {
                    continue;
                }
                let name = snapshot_name(&info.name, self.limits.max_name_chars);
                match upstream_id(&info.id) {
                    Some(id) => self.ready.push(ReadyModel {
                        id,
                        name,
                        backend: self.backend_of(info),
                        probe: self.probes(info),
                    }),
                    None => self.unmetered = true,
                }
            }
        }
        let ready: HashSet<String> = self.ready.iter().map(|model| model.id.clone()).collect();
        if self.running_up {
            // A model that left `ready` in a good read was unloaded (or is
            // loading again), and one whose server changed is a new
            // process: its next load is probed afresh, its engine windows
            // start over, its counters count the next process from 0 and
            // its prompt counters go back to activity rows until `/metrics`
            // takes over again (#46).
            self.detected.retain(|id, _| ready.contains(id));
            let gone: Vec<String> = was_ready
                .iter()
                .filter(|(id, backend)| {
                    !self
                        .ready
                        .iter()
                        .any(|model| model.id == *id && model.backend == *backend)
                })
                .map(|(id, _)| id.clone())
                .collect();
            for id in &gone {
                self.forget_engine(id);
                self.forget_process(id);
            }
        } else {
            // llama-swap is not answering. Its engines' windows go now, so
            // the old process's numbers do not outlive it (#46). A refused
            // connection means llama-swap itself is gone, and with it every
            // server it started: their counters count the next process from
            // 0. A timeout may be a busy but live llama-swap, whose servers
            // still run, so their baselines stay until a good read says
            // whether each is still ready.
            for (id, _) in &was_ready {
                self.forget_engine(id);
                if down == Some(llamaswap::REFUSED) {
                    self.forget_process(id);
                }
            }
            self.down_ready = was_ready;
        }
        if self
            .capture
            .as_ref()
            .is_some_and(|(id, _)| !ready.contains(id.as_str()))
        {
            self.capture = None;
        }
        self.fallback.retain(|id| ready.contains(id.as_str()));
        self.slots_textless.retain(|id| ready.contains(id.as_str()));
        self.not_loaded.retain(|id| ready.contains(id.as_str()));
        self.gauges.retain(|id, _| ready.contains(id.as_str()));
        self.engine_facts
            .retain(|id, _| ready.contains(id.as_str()));
        for model in &self.ready {
            if !model.backend.has_metrics() {
                self.fallback.insert(model.id.clone());
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

    /// Drop `id`'s engine windows and attributed-speed reads (#31, #35).
    fn forget_engine(&mut self, id: &str) {
        self.engines.forget(id);
        self.speeds.forget(id);
        self.series.forget_gauges(id);
    }

    /// `id`'s process is gone: the next one starts its counters at 0 and
    /// its prompt counters come from activity rows until `/metrics` gives
    /// both again (#46).
    fn forget_process(&mut self, id: &str) {
        self.counter.forget(id);
        self.prompt_counter.forget(id);
        self.prompt_box.forget(id);
        self.prompt_cache.forget(id);
        self.series.forget_process(id);
    }

    /// `[llama.backends]` first, then the launch command, then what this
    /// load's `/metrics` probe found (#31).
    fn backend_of(&self, info: &llamaswap::RunningModel) -> Backend {
        if let Some(kind) = self.limits.backends.get(&info.id) {
            return *kind;
        }
        if self.probes(info) {
            return self
                .detected
                .get(&info.id)
                .copied()
                .unwrap_or(Backend::OpenAi);
        }
        info.backend
    }

    /// The launch command named no server and the config does not either.
    fn probes(&self, info: &llamaswap::RunningModel) -> bool {
        info.backend == Backend::OpenAi && !self.limits.backends.contains_key(&info.id)
    }

    /// One `/metrics` GET for each ready model that still needs a probe
    /// this load (#31), through the fresh gate (#70). A model the gate
    /// skips, or that llama-swap says is not loaded, keeps its probe for a
    /// later round.
    fn probe_backends(&mut self) {
        let todo: Vec<String> = self
            .ready
            .iter()
            .filter(|model| model.probe && !self.detected.contains_key(&model.id))
            .map(|model| model.id.clone())
            .collect();
        for id in todo {
            let read = match self.upstream_read(&id, "metrics", self.limits.metrics_timeout, |_| {
                SERVER_METRICS_CAP
            }) {
                Gate::Read(model, read) if model.probe && !self.detected.contains_key(&id) => read,
                Gate::Read(..) | Gate::Skipped | Gate::NotLoaded => continue,
            };
            let read = exact(read)
                .map_err(TapError::label)
                .and_then(|bytes| String::from_utf8(bytes).map_err(|_| "malformed"));
            let (kind, note) = match read.as_deref().map(detect_backend) {
                Ok(Some(kind)) => (kind, format!("/metrics says {}", kind.as_str())),
                Ok(None) => (
                    Backend::OpenAi,
                    "no known metric names; staying openai".to_owned(),
                ),
                Err(label) => (
                    Backend::OpenAi,
                    format!("no /metrics ({label}); staying openai"),
                ),
            };
            self.detected.insert(id.clone(), kind);
            // The detail and SETUP values were read before the server was
            // known: read `/running` again now, so its container flags are
            // read as this server's (#67).
            if kind != Backend::OpenAi {
                self.running_due = true;
            }
            if self.detect_logged.insert(format!("{id}:{}", kind.as_str())) {
                log::emit(
                    &mut self.log,
                    Priority::Info,
                    &format!("{id}: launch command names no server; {note}"),
                );
            }
            for model in self.ready.iter_mut().filter(|model| model.id == id) {
                model.backend = kind;
            }
            for (model, model_id) in self.models.iter_mut().zip(&self.model_ids) {
                if *model_id == id
                    && let Some(info) = model.backend.as_mut()
                {
                    info.kind = kind;
                    if kind == Backend::Strata {
                        info.max_running = info.max_running.or(Some(1));
                    }
                }
            }
            if kind.has_metrics() {
                self.fallback.remove(&id);
            }
        }
    }

    fn poll_metrics(&mut self) {
        self.probe_backends();
        let ready: Vec<ReadyModel> = self
            .ready
            .iter()
            .filter(|model| model.backend.has_metrics())
            .cloned()
            .collect();
        if ready.is_empty() {
            self.latencies.metrics = None;
            return;
        }
        let started = Instant::now();
        let mut failure: Option<&'static str> = None;
        for listed in &ready {
            let (model, read) = match self.upstream_read(
                &listed.id,
                "metrics",
                self.limits.metrics_timeout,
                metrics_cap,
            ) {
                Gate::Read(model, read) if model.backend.has_metrics() => (model, read),
                Gate::Read(..) | Gate::Skipped | Gate::NotLoaded => continue,
            };
            let model = &model;
            let read = exact(read)
                .map_err(TapError::label)
                .and_then(|bytes| String::from_utf8(bytes).map_err(|_| "malformed"))
                .map(|text| parse_metrics_full(model.backend, &text));
            let now = Instant::now();
            let mut live = None;
            let read = read.map(|(sample, facts)| {
                if let Some(mut facts) = facts {
                    live = facts.live.take();
                    self.engine_facts.insert(model.id.clone(), facts);
                }
                sample
            });
            if let Ok(sample) = &read {
                self.series
                    .observe_metrics(&model.id, model.backend, sample, now);
            }
            let decoded = match &read {
                Ok(sample) => sample.n_decode_total,
                Err(_) => None,
            };
            let prompt = read.as_ref().ok().and_then(|sample| sample.prompt_total);
            if let Ok(sample) = &read
                && !model.backend.has_slots()
            {
                self.prompt_cache.observe_metrics(
                    &model.id,
                    sample.prompt_total,
                    sample.cached_total,
                );
                let engine = self.engines.observe(&model.id, sample);
                self.speeds.observe(&model.id, now, sample.speeds);
                self.gauges
                    .insert(model.id.clone(), Gauges::of(sample, engine, live, now));
                if let Some(prompt) = sample.prompt_total {
                    self.prompt_counter.observe(&model.id, prompt, None, now);
                }
            }
            match (decoded, read) {
                (Some(value), Ok(sample)) => {
                    self.fallback.remove(&model.id);
                    self.counter
                        .observe(&model.id, value, sample.requests_processing, now);
                }
                (_, read) if model.backend == Backend::LlamaCpp => {
                    if failure.is_none() {
                        failure = Some(match read {
                            Ok(_) => "malformed",
                            Err(label) => label,
                        });
                    }
                }
                (_, read) => {
                    let reason = match read {
                        Ok(_) => "no token counter",
                        Err(label) => label,
                    };
                    self.use_activity(model, reason);
                }
            }
            // A fallback model's prompt tokens come from activity rows.
            if let Some(prompt) = prompt
                && !self.fallback.contains(&model.id)
            {
                self.prompt_box.observe(&model.id, prompt, None, now);
            }
        }
        note_flag(&mut self.log, &mut self.metrics_failed, failure, "metrics");
        self.latencies.metrics = Some(started.elapsed());
    }

    /// The fresh gate (#70), and the one place an `/upstream/<id>/<leaf>`
    /// GET is made. llama-swap loads any model an upstream request names
    /// that is not loaded, so:
    ///
    /// 1. It reads `/running` itself, as the request just before the GET.
    /// 2. If that read lists any model in a state other than `ready`, a
    ///    swap is in progress: no upstream read for the rest of the round.
    /// 3. It GETs only if that read lists `id` as `ready`, `id` had no
    ///    non-2xx answer this round, and `id` is not backed off.
    /// 4. It reads `/running` again at once. A model `starting` there
    ///    began loading during our GET (step 2 means none was before), and
    ///    a GET slower than [`SUSPECT_SLOW`] may have waited for one: either
    ///    is a suspected load. It is logged, counted, and `id` gets no
    ///    upstream reads for [`SUSPECT_BACKOFF`].
    ///
    /// A dropped read is not retried; the next round asks again. `cap` is
    /// given the model's server as the read in step 1 has it.
    ///
    /// A 409 (llama-swap's `upstream.ignorePaths` answer for a model that
    /// is not loaded) or any other non-2xx answer: the model is not
    /// available now. Its engine numbers go, as on unload, and it gets no
    /// more reads this round. A 409 is logged once until a read succeeds or
    /// the model leaves `ready`, and is no failure of any tap.
    fn upstream_read(
        &mut self,
        id: &str,
        leaf: &'static str,
        timeout: Duration,
        cap: impl FnOnce(Backend) -> usize,
    ) -> Gate {
        if self.round_blocked || self.round_skip.contains(id) || self.backed_off(id) {
            return Gate::Skipped;
        }
        self.poll_running();
        if self.swapping {
            self.round_blocked = true;
            return Gate::Skipped;
        }
        let Some(model) = self.ready.iter().find(|model| model.id == id).cloned() else {
            return Gate::Skipped;
        };
        let url = upstream(&self.limits.url, &model.id, leaf);
        let started = Instant::now();
        let read = get_limited(&self.agent, &url, timeout, cap(model.backend));
        let took = started.elapsed();
        if let Err(TapError::NotLoaded) = &read {
            // llama-swap refused without starting anything.
            self.not_available(id);
            if self.not_loaded.insert(id.to_owned()) {
                log::emit(
                    &mut self.log,
                    Priority::Info,
                    &format!("{id}: llama-swap says not loaded; skipping until ready"),
                );
            }
            return Gate::NotLoaded;
        }
        self.poll_running();
        let why = if self.running_up && self.swapping_to_start() {
            Some("a model is starting after it".to_owned())
        } else if took > SUSPECT_SLOW {
            Some(format!("it took {:.1} s", took.as_secs_f64()))
        } else {
            None
        };
        if let Some(why) = why {
            self.suspect_load(id, leaf, &why);
        }
        match &read {
            Err(TapError::Status | TapError::Unauthorized) => self.not_available(id),
            Ok(_) => {
                self.not_loaded.remove(id);
            }
            Err(_) => {}
        }
        Gate::Read(model, read)
    }

    /// The last `/running` read lists a model as `starting`.
    fn swapping_to_start(&self) -> bool {
        self.models.iter().any(|model| model.state == "starting")
    }

    /// `id` is backed off after a suspected load. An expired entry goes.
    fn backed_off(&mut self, id: &str) -> bool {
        let now = Instant::now();
        self.backoff.retain(|_, until| *until > now);
        self.backoff.contains_key(id)
    }

    /// Our upstream read of `id` may have made llama-swap load a model
    /// (#70): warn, count it, and leave `id` alone for [`SUSPECT_BACKOFF`].
    fn suspect_load(&mut self, id: &str, leaf: &str, why: &str) {
        self.backoff
            .insert(id.to_owned(), Instant::now() + SUSPECT_BACKOFF);
        self.round_skip.insert(id.to_owned());
        if let Some((_, count)) = self.suspected.iter_mut().find(|(model, _)| model == id) {
            *count = count.saturating_add(1);
        } else if self.suspected.len() < MAX_SUSPECTS {
            self.suspected.push((id.to_owned(), 1));
        }
        log::emit(
            &mut self.log,
            Priority::Warning,
            &format!(
                "{id}: suspected model load after GET /upstream/{id}/{leaf} ({why}); no upstream reads for it for {} s",
                SUSPECT_BACKOFF.as_secs()
            ),
        );
    }

    /// `id` answered an upstream read with a non-2xx (#70): forget its
    /// engine numbers and slots, as on unload, and skip it for the rest of
    /// this round. Its counters stay: the process may well be the same.
    fn not_available(&mut self, id: &str) {
        self.round_skip.insert(id.to_owned());
        self.gauges.remove(id);
        self.engine_facts.remove(id);
        self.forget_engine(id);
        let keep: Vec<&str> = self
            .ready
            .iter()
            .map(|model| model.id.as_str())
            .filter(|model| *model != id)
            .collect();
        self.slots.retain_models(&keep);
    }

    /// A non-llama.cpp model with no decode counter: count it from activity
    /// rows. Logged once per model per run; never a `metrics` failure.
    fn use_activity(&mut self, model: &ReadyModel, reason: &str) {
        self.fallback.insert(model.id.clone());
        if self.no_metrics_logged.insert(model.id.clone()) {
            log::emit(
                &mut self.log,
                Priority::Info,
                &format!(
                    "{}: no /metrics from {} ({reason}); using llama-swap activity",
                    model.id,
                    model.backend.as_str()
                ),
            );
        }
    }

    fn poll_slots(&mut self) {
        let now = Instant::now();
        let busy: Vec<ReadyModel> = self
            .ready
            .iter()
            .filter(|model| {
                model.backend.has_slots()
                    && self
                        .counter
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
        let cap = self.limits.slots_cap;
        for listed in &busy {
            let (model, read) =
                match self.upstream_read(&listed.id, "slots", self.limits.slots_timeout, |_| cap) {
                    Gate::Read(model, read) if model.backend.has_slots() => (model, read),
                    Gate::Read(..) | Gate::Skipped | Gate::NotLoaded => continue,
                };
            let model = &model;
            match read {
                Ok(Limited::Exact(bytes)) => {
                    if self.slots.apply(
                        &model.id,
                        &model.name,
                        &bytes,
                        self.limits.input_tail,
                        self.limits.output_tail,
                    ) {
                        self.note_slot_text(&model.id);
                    } else if failure.is_none() {
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

    /// Use captures for `id`'s IN and OUT while its `/slots` has no text,
    /// and its live text again once it has (#66). Logged once per model.
    fn note_slot_text(&mut self, id: &str) {
        let Some(text) = self.slots.has_text(id) else {
            return;
        };
        if text {
            if self.slots_textless.remove(id)
                && self.capture.as_ref().is_some_and(|(model, _)| model == id)
            {
                self.capture = None;
            }
        } else if self.slots_textless.insert(id.to_owned())
            && self.textless_logged.insert(id.to_owned())
        {
            log::emit(
                &mut self.log,
                Priority::Info,
                &format!(
                    "{id}: /slots has no prompt text; start llama-server with LLAMA_SERVER_SLOTS_DEBUG=1 for live IN/OUT"
                ),
            );
        }
    }

    /// IN and OUT come from captures: no `/slots`, or no text on it (#66).
    fn textless(&self, model: &ReadyModel) -> bool {
        !model.backend.has_slots() || self.slots_textless.contains(&model.id)
    }

    fn poll_activity(&mut self) {
        if !self.running_up {
            self.latencies.activity = None;
            return;
        }
        let started = Instant::now();
        let url = join_url(&self.limits.url, "api/metrics/activity");
        let mut page: Vec<ActivityRow> = Vec::new();
        let failure = match get_exact(
            &self.agent,
            &url,
            self.limits.activity_timeout,
            ACTIVITY_CAP,
        ) {
            Ok(bytes) => match activity::parse_activity_rows(&bytes, activity::MAX_PAGE_ROWS) {
                Some(rows) => {
                    let merged = self.recent.merge(rows);
                    if merged.restarted {
                        log::emit(
                            &mut self.log,
                            Priority::Info,
                            "activity: llama-swap restarted; its row ids start again",
                        );
                    }
                    self.count_activity(&merged);
                    self.slots.note_activity(&merged.page, Instant::now());
                    page = merged.page;
                    None
                }
                None => Some("malformed"),
            },
            Err(err) => Some(err.label()),
        };
        let activity_took = started.elapsed();
        // After the activity timing, so a capture read does not count as
        // activity latency on the health line.
        self.poll_capture(&page);
        if failure.is_some() {
            // Drops waiting for their row still age out (#9).
            self.slots.note_activity(&[], Instant::now());
        }
        note_flag(
            &mut self.log,
            &mut self.activity_failed,
            failure,
            "activity",
        );
        self.latencies.activity = Some(activity_took);
    }

    /// Add each new row's `output_tokens` to its fallback model, once, and
    /// every ready model's prompt and cached tokens to its counters (#10).
    ///
    /// New rows are the ones [`Recent::merge`] saw for the first time, so a
    /// llama-swap restart that reuses ids is counted in full (#44). The
    /// first read only sets the baseline, so history from before the
    /// watcher started is not back-filled. Each fallback model is then
    /// fresh until the next activity read is due.
    fn count_activity(&mut self, merged: &Merged) {
        for model in &self.ready {
            self.prompt_cache
                .touch(&model.id, model.backend.has_slots());
            self.series.touch(&model.id);
        }
        let now = Instant::now();
        if merged.baseline {
            self.speeds.activity(now, &[]);
            return;
        }
        let rows = &merged.new;
        // #35: new rows of a model without `/slots` that llama-swap gave
        // no speeds wait for the engine's.
        let mut measure: Vec<(u64, String)> = Vec::new();
        for model in self.ready.iter().filter(|model| !model.backend.has_slots()) {
            let key = activity::model_key(&model.id);
            measure.extend(
                rows.iter()
                    .filter(|row| row.model == key)
                    .filter(|row| row.prompt_tps.is_none() || row.gen_tps.is_none())
                    .map(|row| (row.seq, model.id.clone())),
            );
        }
        self.speeds.activity(now, &measure);
        for model in &self.ready {
            let key = activity::model_key(&model.id);
            for row in rows.iter().filter(|row| row.model == key) {
                self.prompt_cache
                    .add_row(&model.id, row.input_tokens, row.cached_tokens);
                self.series
                    .add_row(&model.id, row, model.backend.has_slots());
            }
        }
        let window = self.limits.activity_interval + self.limits.activity_timeout;
        let fallback: Vec<String> = self.fallback.iter().cloned().collect();
        for id in &fallback {
            let key = activity::model_key(id);
            let new_rows = || rows.iter().filter(|row| row.model == key);
            let tokens = new_rows()
                .filter_map(|row| row.output_tokens)
                .fold(0u64, u64::saturating_add);
            self.counter.add(id, tokens, now, window);
            self.series.add_generation(id, tokens);
            let prompt = new_rows()
                .filter_map(|row| row.input_tokens)
                .fold(0u64, u64::saturating_add);
            self.prompt_box.add(id, prompt, now, window);
        }
    }

    /// Fetch the capture of the newest finished row of a ready model
    /// without `/slots`, once per row (#5). Nothing with text off; a row
    /// without `has_capture` (captures off, or already evicted from
    /// llama-swap's buffer) keeps what is shown.
    ///
    /// `rows` is the page just read, so every id in it is the current
    /// llama-swap's: a row RECENT keeps from before a restart is never
    /// fetched, as its id may now name another request (#44). A row is
    /// `(generation, id)`, so a reused id is fetched for its new request.
    fn poll_capture(&mut self, rows: &[ActivityRow]) {
        if !self.limits.show_text {
            return;
        }
        let textless: Vec<(String, &ReadyModel)> = self
            .ready
            .iter()
            .filter(|model| self.textless(model))
            .map(|model| (activity::model_key(&model.id), model))
            .collect();
        let Some((row, model)) = rows
            .iter()
            .filter_map(|row| {
                let (_, model) = textless.iter().find(|(key, _)| *key == row.model)?;
                Some((row, *model))
            })
            .max_by_key(|(row, _)| row.id)
        else {
            return;
        };
        let key = (self.recent.generation(), row.id);
        if self.capture_seen == Some(key) {
            return;
        }
        self.capture_seen = Some(key);
        if !row.captured || row.id < 0 {
            return;
        }
        let url = join_url(&self.limits.url, &format!("api/captures/{}", row.id));
        let failure = match get_limited(&self.agent, &url, CAPTURE_TIMEOUT, CAPTURE_CAP) {
            Ok(Limited::Exact(bytes)) => match parse_capture(&bytes) {
                Some(text) => {
                    let view = CaptureView {
                        model: model.name.clone(),
                        id: row.id,
                        input: prompt_cells(
                            &text.input,
                            self.limits.input_tail,
                            self.limits.prompt_view,
                        ),
                        input_note: text.input_note,
                        output: tail_cells(&text.output, self.limits.output_tail),
                        over_slots: model.backend.has_slots(),
                    };
                    self.capture = Some((model.id.clone(), view));
                    None
                }
                None => Some("malformed"),
            },
            Ok(Limited::Oversize) => {
                if !self.capture_oversize {
                    log::emit(
                        &mut self.log,
                        Priority::Info,
                        &format!("captures: {} exceeds {CAPTURE_CAP} bytes; skipped", row.id),
                    );
                    self.capture_oversize = true;
                }
                None
            }
            Err(err) => Some(err.label()),
        };
        note_flag(&mut self.log, &mut self.capture_failed, failure, "captures");
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
        let prompt_total = if self.unmetered {
            None
        } else {
            self.prompt_box
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
        // Prompt tok/s from SGLang/vLLM's prompt counter, for when no
        // `/slots` gave one.
        let prompt_rate = self.prompt_rate.observe(now, self.prompt_counter.total());
        let prompt_tps = if !self.running_up {
            None
        } else if !busy {
            Some(0.0)
        } else {
            self.slots.prompt_tps().or(prompt_rate)
        };
        let view = LlamaView {
            ai: self.ai,
            models: self.models_with_gauges(now),
            decoded_total: decoded,
            prompt_total,
        };
        let detail = LlamaDetail {
            slots: self.slots.slots(),
            activity: self.measured_activity(),
            gen_tps,
            prompt_tps,
            latencies: self.latencies,
            prompt_cache: self
                .ready
                .iter()
                .filter_map(|model| {
                    let (prompt, cached) = self.prompt_cache.get(&model.id)?;
                    Some(ModelPromptCache {
                        model: model.name.clone(),
                        prompt,
                        cached,
                    })
                })
                .collect(),
            capture: self.capture.as_ref().map(|(_, view)| view.clone()),
            setup: self.setup_with_engine(),
            engine_live: self.engine_live(now),
            suspected_loads: self
                .suspected
                .iter()
                .map(|(id, count)| {
                    (
                        sanitize(id, llama_core::detail::MAX_FULL_NAME_CHARS),
                        *count,
                    )
                })
                .collect(),
            series: self.model_series(now),
        };
        if Arc::strong_count(&self.tx.slot) == 1 {
            return Err(());
        }
        self.tx.put((view, detail));
        Ok(())
    }

    /// [`Self::model_setup`] with each model's engine-reported values (#54).
    fn setup_with_engine(&self) -> Vec<ModelSetup> {
        let mut setup = self.model_setup.clone();
        for (entry, id) in setup.iter_mut().zip(&self.model_ids) {
            if let Some(facts) = self.engine_facts.get(id) {
                entry.engine = facts.values.clone();
            }
        }
        setup
    }

    /// Each listed model's cumulative numbers (#71), in the order of
    /// [`Self::models`], with the version its engine reports.
    fn model_series(&self, now: Instant) -> Vec<ModelSeries> {
        let fresh = FRESH_GAUGES.max(self.limits.metrics_interval * 2);
        self.models
            .iter()
            .zip(&self.model_ids)
            .map(|(model, id)| ModelSeries {
                model: model.name.clone(),
                version: self
                    .engine_facts
                    .get(id)
                    .and_then(|facts| facts.values.get("version").cloned()),
                ..self.series.get(id, now, fresh)
            })
            .collect()
    }

    /// Each ready model's engine `live` report from a fresh read (#54).
    fn engine_live(&self, now: Instant) -> Vec<ModelEngineLive> {
        let fresh = FRESH_GAUGES.max(self.limits.metrics_interval * 2);
        self.ready
            .iter()
            .filter_map(|model| {
                let gauges = self
                    .gauges
                    .get(&model.id)
                    .filter(|gauges| now.saturating_duration_since(gauges.at) <= fresh)?;
                Some(ModelEngineLive {
                    model: model.name.clone(),
                    live: gauges.live.clone()?,
                })
            })
            .collect()
    }

    /// RECENT rows, newest first from llama-watch's own ring (#44), with
    /// the engine's speeds where llama-swap gave none (#35).
    fn measured_activity(&self) -> Vec<ActivityRow> {
        let mut rows = self.recent.rows(self.activity_rows());
        for row in &mut rows {
            let Some(speeds) = self.speeds.speeds(row.seq) else {
                continue;
            };
            if row.prompt_tps.is_none() {
                row.engine_prompt_tps = speeds.prefill;
            }
            if row.gen_tps.is_none() {
                row.engine_gen_tps = speeds.decode;
            }
        }
        rows
    }

    /// [`Self::models`] with each backend's fresh gauges filled in, and
    /// engine facts where the launch command gave no ctx or KV. A
    /// llama.cpp model's KV comes from its slots (#79).
    fn models_with_gauges(&self, now: Instant) -> Vec<ModelInfo> {
        let fresh = FRESH_GAUGES.max(self.limits.metrics_interval * 2);
        let mut models = self.models.clone();
        let slots = self.slots.slots();
        for ((model, id), layout) in models.iter_mut().zip(&self.model_ids).zip(&self.model_kv) {
            if let Some(facts) = self.engine_facts.get(id) {
                with_facts(&mut model.detail, facts);
            }
            let ctx = model.detail.as_ref().and_then(|detail| detail.ctx);
            let Some(info) = model.backend.as_mut() else {
                continue;
            };
            if info.kind.has_slots() {
                let own: Vec<&SlotView> = slots
                    .iter()
                    .filter(|slot| slot.model == model.name)
                    .collect();
                info.kv = crate::kv::llamacpp(&own, *layout, ctx);
            }
            let Some(gauges) = self
                .gauges
                .get(id)
                .filter(|gauges| now.saturating_duration_since(gauges.at) <= fresh)
            else {
                continue;
            };
            info.running = gauges.running;
            info.queued = gauges.queued;
            info.kv_permille = gauges.kv_permille;
            info.hit_permille = gauges.hit_permille;
            info.engine = gauges.engine;
            info.kv = crate::kv::from_metrics(gauges.kv, gauges.running);
        }
        models
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
    /// 409 on an upstream read: llama-swap says the model is not loaded
    /// (its `upstream.ignorePaths` answer, #70).
    NotLoaded,
    Status,
    /// 401 or 403: the server wants an API key llama-bored does not keep.
    Unauthorized,
    Oversize,
    Failed,
}

impl TapError {
    fn label(self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::Refused => "connection refused",
            Self::NotLoaded => "not loaded",
            Self::Status => "http status",
            Self::Unauthorized => "unauthorized, API key set",
            Self::Oversize => "oversized body",
            Self::Failed => "request failed",
        }
    }
}

/// Fill `detail`'s ctx when the launch command gave none, its KV when it
/// is unknown or `auto`, and the KV block size and prefix caching (#31),
/// from the server's own report.
fn with_facts(detail: &mut Option<llama_core::detail::ModelDetail>, facts: &EngineFacts) {
    if *facts == EngineFacts::default() {
        return;
    }
    let detail = detail.get_or_insert_with(Default::default);
    if detail.ctx.is_none() {
        detail.ctx = facts.ctx;
    }
    if detail.kv_block.is_none() {
        detail.kv_block = facts.kv_block;
    }
    if detail.prefix_cache.is_none() {
        detail.prefix_cache = facts.prefix_cache;
    }
    if let Some(kv) = &facts.kv {
        for side in [&mut detail.kv_k, &mut detail.kv_v] {
            if side.as_deref().is_none_or(|old| old == "auto") {
                *side = Some(kv.clone());
            }
        }
    }
}

fn metrics_cap(backend: Backend) -> usize {
    if backend.has_slots() {
        LLAMACPP_METRICS_CAP
    } else {
        SERVER_METRICS_CAP
    }
}

fn get_exact(
    agent: &ureq::Agent,
    url: &str,
    timeout: Duration,
    cap: usize,
) -> Result<Vec<u8>, TapError> {
    exact(get_limited(agent, url, timeout, cap))
}

/// An oversized body is an error.
fn exact(read: Result<Limited, TapError>) -> Result<Vec<u8>, TapError> {
    match read? {
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
    let status = response.status().as_u16();
    if status == 401 || status == 403 {
        return Err(TapError::Unauthorized);
    }
    if status == 409 {
        return Err(TapError::NotLoaded);
    }
    if status != 200 {
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
        ureq::Error::StatusCode(401 | 403) => TapError::Unauthorized,
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
