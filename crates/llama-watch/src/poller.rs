//! One thread polling llama-swap into a [`LlamaView`] and a [`LlamaDetail`].
//!
//! The thread owns a single [`ureq::Agent`] (`proxy(None)`, `max_redirects(0)`,
//! per-call `timeout_global`) and uses it for every tap, including `/running`
//! via [`crate::sources::llamaswap::fetch_running`]. One `/running` body supplies
//! each model's raw id and state, and the spec files for those ids are read
//! here, off the frame path (#92). Tails use `input_tail_chars` and
//! `output_tail_chars`.
//! With `tty.show_text = false` the `/slots` bodies are parsed for numbers
//! only: no prompt or generated text is kept (RR-LV1).
//!
//! The taps run in series on this thread. With eight ready models the worst
//! case is one `/running` timeout, eight `/metrics` timeouts, eight `/slots`
//! timeouts, two `/running` timeouts around each of those sixteen (the
//! fresh gate, #70), one activity timeout and one capture timeout. At the
//! default timeouts that is 0.25 + 8×0.2 + 8×0.5 + 32×0.25 + 0.25 + 1 =
//! 15.1 s when every call hangs (an SGLang model with a request in flight
//! adds one `/v1/loads` read and its two gate reads to that, #80; the
//! `/api/events` stream has its own thread and adds nothing). The consumer calls [`SampleRx::take`], which does not wait, so that
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
//! tqLLM (#180) has no `/metrics`: at the metrics cadence it gets
//! `GET /upstream/<id>/api/monitor/v1/snapshot` instead, capped at
//! [`TQLLM_CAP`], its JSON depth capped, and read strictly
//! ([`crate::metrics::parse_tqllm`]); [`crate::tqllm::TqllmBook`] turns
//! its per-request numbers into the counters every engine has. A
//! `schema_version` other than 1 is "no metrics", logged once per load.
//! `GET /upstream/<id>/health` is read once per load and once per
//! `instance_id`, for its `fit` (SETUP's `engine:fit`). Nothing else is
//! sent to tqLLM: never `POST /admin/slots` (it changes the live slot
//! count) or the monitor's chat bridge.
//!
//! A model whose launch command names no server (`openai`: a container
//! whose image runs `vllm serve`, a wrapper script around llama-server) and
//! that `[llama.backends]` does not name gets one `/upstream/<id>/metrics`
//! GET per load, with the usual cap and timeout (#31). A JSON object with
//! `engine` and `live` objects is Strata (#54, a container started by image
//! digest); else the first sample's metric prefix decides: `vllm:`,
//! `sglang:` or `llamacpp:` (which then also gets `/slots`). When
//! `/metrics` gives none of these (#180), one
//! `GET /upstream/<id>/api/monitor/v1/snapshot` follows, in the same round
//! or the next: a body with `schema_version` 1, an `engine` object and a
//! `requests` array is tqLLM. Anything else stays `openai`. Each outcome
//! is logged once. A probe
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
//! **In flight on vLLM and SGLang (#80).** While llama-swap answers (until
//! #167, only while a model was ready), its `GET /api/events` stream runs
//! on a thread of its own
//! ([`crate::sources::events`]): its `inflight` frames list the requests
//! llama-swap is serving, for any engine, and a llama-swap build with
//! model claims sends `claims` frames on the same stream (#128). Each vLLM
//! `/metrics` read shares its counter deltas among them
//! ([`crate::live`]); for an SGLang model with a request in flight the
//! round adds one gated `GET /upstream/<id>/v1/loads?include=core`, at the
//! metrics cadence. Each such request is a [`LiveRequest`] for RECENT, and
//! each listed model's in-flight count rides [`ModelSeries::inflight`]
//! while the stream is up. The claims list rides [`LlamaDetail::claims`]
//! to the tty model, and the GPU reservation and displaced notes
//! [`LlamaDetail::gpu`] and [`LlamaDetail::displaced`] (#167); none of
//! them enters the snapshot.
//!
//! The thread never touches the console or the snapshot file.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime};

use llama_core::backend::{self, Backend, BackendInfo, EngineStats};
use llama_core::log::{self, Priority, Sink};
use llama_core::names::sanitize;
use llama_core::sample::{AiState, LlamaView, ModelInfo};
use llama_core::wire::CANONICAL_NAME_CHARS;

use crate::activity::{self, ActivityRow};
use crate::capture::{CAPTURE_CAP, MessageMark, new_input_chars, parse_capture, shown_new};
use crate::config::{PromptView, ValidWatchConfig};
use crate::jobs::{self, Job, Jobs};
use crate::lcpp_tokens::{LcppTokens, MetricsRead};
use crate::live::{LOADS_CAP, LiveBook, LiveTokens, parse_loads};
use crate::metrics::{
    DecodedCounter, EngineBook, EngineFacts, EngineLive, EngineValues, GenRate, KvTokens,
    MetricsSample, PromptCache, TqError, detect_backend, is_tqllm_json, parse_metrics_full,
    parse_tqllm, parse_tqllm_health,
};
use crate::recent::{Merged, Recent};
use crate::series::{ModelSeries, SeriesBook};
use crate::setup_rules::{Found, Rules};
use crate::setup_specs::Specs;
use crate::sglang_rows::{Claim, SglangBook, Totals};
use crate::slots::{ChartPhase, SlotBook, SlotView, prompt_cells, tail_cells};
use crate::sources::cmdline::{KvLayout, TqKv};
use crate::sources::events::{
    DisplacedNote, Events, EventsView, GpuReservation, ModelClaim, claim_model,
};
use crate::sources::llamaswap;
use crate::speeds::SpeedBook;
use crate::tqllm::{RowClaim, TqllmBook};
use crate::tty::grid::Cell;
use crate::vllm_phase::VllmPhaseBook;

const RUNNING_CAP: usize = 64 * 1024;
/// Captured requests per model whose message marks are kept for the IN
/// diff (DER-451). Several, so conversations that interleave on one model
/// each still find their own last request.
const REQUEST_MARKS: usize = 8;
/// Most models with kept request marks.
const REQUEST_MARK_MODELS: usize = 16;
/// llama-server's `/metrics` is a few KiB.
const LLAMACPP_METRICS_CAP: usize = 64 * 1024;
/// SGLang and vLLM export latency histograms per label set; SGLang's is
/// about 70 KiB with one model loaded and grows with its labels.
const SERVER_METRICS_CAP: usize = 1024 * 1024;
const ACTIVITY_CAP: usize = 256 * 1024;
/// tqLLM's monitor snapshot and its `/health` (#180): 32 finished requests
/// and the active ones are a few KiB each.
pub const TQLLM_CAP: usize = 256 * 1024;
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
/// Most models whose server RECENT remembers for its rows (#82).
const MAX_ROW_ENGINES: usize = 64;
/// SGLang's live load report (#80), read only while a request is in
/// flight: the leaf [`State::poll_loads`] names at its gate call (S17
/// reads it there). A non-2xx answer to it says nothing of the model.
const LOADS_LEAF: &str = "v1/loads?include=core";
/// tqLLM's `/health` leaf (#180): a non-2xx there only loses the `fit`.
const HEALTH_LEAF: &str = "health";
/// The shortest connect and head timeout for `/api/events` (#80).
const EVENTS_TIMEOUT: Duration = Duration::from_secs(1);

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
    /// Chart input: phase-gated historical estimate for active llama.cpp prefill,
    /// true zero for confirmed decode, and unknown for missing phase. Other
    /// engines retain their existing rate semantics; this is not a measurement.
    pub chart_prompt_tps: Option<f64>,
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
    /// vLLM and SGLang requests llama-swap has in flight (#80), oldest
    /// first, with what the engine's totals say of each.
    pub requests: Vec<LiveRequest>,
    /// Job status files from `[jobs] dir`, newest mtime first. Empty when
    /// the directory is unset. Not written to the snapshot.
    pub jobs: Vec<Job>,
    /// Latest model claims from llama-swap's `/api/events` (#128). Empty
    /// when the stream is down or has not sent a `claims` frame. Not
    /// written to the snapshot.
    pub claims: Vec<ModelClaim>,
    /// The GPU reservation from the same stream (#167). Not written to the
    /// snapshot.
    pub gpu: Option<GpuReservation>,
    /// Recent displaced notes, newest first (#167). Not written to the
    /// snapshot.
    pub displaced: Vec<DisplacedNote>,
}

/// One vLLM or SGLang request in flight (#80), from llama-swap's
/// `/api/events` and the engine's totals.
#[derive(Clone, Debug, PartialEq)]
pub struct LiveRequest {
    /// Display name of its model, as [`SlotView::model`] has it.
    pub model: String,
    /// llama-swap's in-flight id (not the activity id).
    pub id: String,
    /// Its engine.
    pub engine: Backend,
    /// When it started, by llama-swap's `elapsed_ms`.
    pub started: Instant,
    /// The same on the wall clock.
    pub started_wall: SystemTime,
    /// Response bytes written: 0 while it is still in prefill.
    pub resp_bytes: u64,
    /// Its tokens, once a read credited it.
    pub tokens: Option<LiveTokens>,
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
    /// The llama-swap id, sanitised for display (48 characters).
    pub id: String,
    /// The llama-swap id at the claims cap, from [`crate::sources::events::claim_model`].
    /// The header matches a claim against this, not [`Self::id`].
    pub swap_id: String,
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
    /// Characters at the end of [`Self::input`] that this request appended
    /// (DER-451, [`new_input_chars`]); `None` when its messages give no
    /// answer and the uncached token count has to.
    pub input_new: Option<usize>,
    /// Characters of the whole answer [`Self::output`] is the tail of
    /// (DER-451). IN has no such count: it is the newest messages, not a
    /// tail of the prompt its token count covers.
    pub output_chars: Option<usize>,
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
    /// `[setup] specs_dir`. `None` reads no spec files (#92).
    specs_dir: Option<PathBuf>,
    /// `[jobs] dir`. `None` reads no job files.
    jobs_dir: Option<PathBuf>,
    /// `[llama] enabled`. False skips every llama-swap GET.
    llama_enabled: bool,
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
            specs_dir: config.setup.specs_dir.clone(),
            jobs_dir: config.jobs.dir.clone(),
            llama_enabled: config.llama.enabled,
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
    /// llama.cpp tokens from `/slots`, settled with
    /// `tokens_predicted_total`: what `counter` adds for those models,
    /// not their decode calls (DER-451).
    lcpp_tokens: LcppTokens,
    /// RECENT's own rows across llama-swap restarts (#44).
    recent: Recent,
    ai: AiState,
    models: Vec<ModelInfo>,
    /// Raw id of each entry of `models`, in the same order.
    model_ids: Vec<String>,
    /// llama.cpp KV layout of each entry of `models`, from its launch
    /// command (#79), in the same order.
    model_kv: Vec<Option<KvLayout>>,
    /// What each listed tqLLM command said about its KV (#180).
    model_tq_kv: Vec<Option<TqKv>>,
    /// SETUP values of each entry of `models`, in the same order (#52).
    model_setup: Vec<ModelSetup>,
    /// Per-model spec files, read with `/running` (#92).
    specs: Specs,
    /// Job status files, read at most once a second.
    jobs: Jobs,
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
    /// SGLang cached tokens and speeds from a single-request window (#133).
    sglang: SglangBook,
    /// Prefill or decode of each vLLM model's running requests, for the
    /// READ chart (DER-451).
    vllm_phase: VllmPhaseBook,
    /// Per-model cumulative numbers for llama-metrics (#71).
    series: SeriesBook,
    /// Server kind `/metrics` gave each loaded model the launch command did
    /// not name (#31); `openai` when it gave none. Dropped when the model
    /// leaves `ready`, so the next load is probed again.
    detected: HashMap<String, Backend>,
    /// `id:kind` probe outcomes already logged this run.
    detect_logged: HashSet<String>,
    /// Ready models whose `/metrics` probe found no known server this load
    /// (#180): why, for the log. Their probe goes on to the tqLLM snapshot.
    probe_snapshot: HashMap<String, String>,
    /// tqLLM counters, slots and row fills (#180).
    tqllm: TqllmBook,
    /// Ready tqLLM models whose snapshot `schema_version` was logged this
    /// load (#180).
    schema_logged: HashSet<String>,
    /// tqLLM models whose `/health` failed this load, logged (#180).
    health_logged: HashSet<String>,
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
    /// Message marks of the last [`REQUEST_MARKS`] captured requests per
    /// model id, for IN's new part (DER-451). Hashes and roles, no text.
    request_marks: RequestMarks,
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
    /// The server of each model this run saw ready, by
    /// [`activity::model_key`], at most [`MAX_ROW_ENGINES`] (#82): an
    /// activity row's `input_tokens` is read by its engine, also after the
    /// model unloads.
    row_engines: HashMap<String, Backend>,
    /// llama-swap's `/api/events` stream while llama-swap answers (#80,
    /// #128, #167), and what it said at the start of this round.
    events: Option<Events>,
    events_view: EventsView,
    events_failed: bool,
    /// A malformed `claims` frame was logged this session (#128).
    claims_failed: bool,
    /// A malformed `gpu` frame was logged this session (#167).
    gpu_failed: bool,
    /// A vLLM or SGLang model was ready on the last good `/running` read.
    /// The inflight snapshot field follows this, not the events thread, so
    /// a llama.cpp-only box does not gain `inflight` when the stream runs
    /// for claims (#128).
    inflight_engines: bool,
    /// vLLM and SGLang requests' token shares (#80).
    live: LiveBook,
    /// Ready SGLang models whose `/v1/loads` answered non-2xx this load:
    /// not asked again until they reload (#80).
    loads_missing: HashSet<String>,
    loads_failed: bool,
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
    let specs = Specs::new(limits.specs_dir.clone());
    let jobs = Jobs::new(limits.jobs_dir.clone());
    // Three `/slots` rounds, at least 3 s, before llama.cpp's WRITE falls
    // back to decode calls or becomes unknown (DER-451).
    let lcpp_tokens = LcppTokens::new((limits.slots_interval * 3).max(Duration::from_secs(3)));
    let mut state = State {
        limits,
        agent,
        log,
        tx,
        counter: DecodedCounter::default(),
        gen_rate: GenRate::default(),
        slots,
        lcpp_tokens,
        recent: Recent::default(),
        ai: AiState::Down,
        models: Vec::new(),
        model_ids: Vec::new(),
        model_kv: Vec::new(),
        model_tq_kv: Vec::new(),
        model_setup: Vec::new(),
        specs,
        jobs,
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
        sglang: SglangBook::default(),
        vllm_phase: VllmPhaseBook::default(),
        series: SeriesBook::default(),
        detected: HashMap::new(),
        detect_logged: HashSet::new(),
        probe_snapshot: HashMap::new(),
        tqllm: TqllmBook::default(),
        schema_logged: HashSet::new(),
        health_logged: HashSet::new(),
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
        request_marks: RequestMarks::default(),
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
        row_engines: HashMap::new(),
        events: None,
        events_view: EventsView::default(),
        events_failed: false,
        claims_failed: false,
        gpu_failed: false,
        inflight_engines: false,
        live: LiveBook::default(),
        loads_missing: HashSet::new(),
        loads_failed: false,
    };
    let mut next_running = Instant::now();
    let mut next_metrics = Instant::now();
    let mut next_slots = Instant::now();
    let mut next_activity = Instant::now();
    let jobs_on = state.limits.jobs_dir.is_some();
    let mut next_jobs = Instant::now();
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
        if state.limits.llama_enabled && (now >= next_running || state.running_due) {
            state.poll_running();
            state.sync_events();
            next_running = Instant::now() + state.limits.running_interval;
            worked = true;
        }
        if state.limits.llama_enabled && now >= next_metrics {
            state.poll_metrics();
            next_metrics = Instant::now() + state.limits.metrics_interval;
            worked = true;
        }
        if state.limits.llama_enabled && now >= next_slots {
            state.poll_slots();
            next_slots = Instant::now() + state.limits.slots_interval;
            worked = true;
        }
        if state.limits.llama_enabled && now >= next_activity {
            state.poll_activity();
            next_activity = Instant::now() + state.limits.activity_interval;
            worked = true;
        }
        if jobs_on && now >= next_jobs {
            state.jobs.refresh(Instant::now(), &mut state.log);
            next_jobs = Instant::now() + jobs::CHECK_EVERY;
            worked = true;
        }
        if worked && state.publish().is_err() {
            break;
        }
        let mut next: Option<Instant> = None;
        if state.limits.llama_enabled {
            next = Some(if state.running_due {
                Instant::now()
            } else {
                next_running
            });
            for candidate in [next_metrics, next_slots, next_activity] {
                if next.is_none_or(|at| candidate < at) {
                    next = Some(candidate);
                }
            }
        }
        if jobs_on && next.is_none_or(|at| next_jobs < at) {
            next = Some(next_jobs);
        }
        let wait = next
            .unwrap_or_else(|| Instant::now() + Duration::from_secs(3600))
            .saturating_duration_since(Instant::now());
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
        let reading = match llamaswap::fetch_running(
            &self.agent,
            &self.limits.url,
            self.limits.running_timeout,
        ) {
            llamaswap::Fetch::Down(reason) => llamaswap::down_reading(reason),
            llamaswap::Fetch::Body(bytes) => {
                // Spec files are local and slow, so they stay on this thread.
                // The command is still in the body, so a rule's `match` applies.
                let ids = llamaswap::running_ids(&bytes);
                self.specs.refresh(&ids, Instant::now(), &mut self.log);
                let mut docs = HashMap::new();
                for id in &ids {
                    if let Some(doc) = self.specs.value(id) {
                        docs.insert(id.clone(), doc.clone());
                    }
                }
                llamaswap::parse_body(
                    &bytes,
                    &self.limits.aliases,
                    &self.limits.setup,
                    &known,
                    &docs,
                )
                .unwrap_or_else(|_| llamaswap::down_reading("malformed json"))
            }
        };
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
                        // way it was told (#54), unless its `live` reports
                        // batch slots (#81, set with the gauges).
                        (self.backend_of(info) == Backend::Strata).then_some(1)
                    }),
                    ..BackendInfo::default()
                }),
            })
            .collect();
        self.model_ids = models.iter().map(|info| info.id.clone()).collect();
        self.model_kv = models.iter().map(|info| info.kv).collect();
        self.model_tq_kv = models.iter().map(|info| info.tq_kv).collect();
        self.model_setup = models
            .iter()
            .map(|info| ModelSetup {
                key: activity::model_key(&info.id),
                id: sanitize(&info.id, llama_core::detail::MAX_FULL_NAME_CHARS),
                swap_id: claim_model(&info.id).unwrap_or_default(),
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
            self.inflight_engines = self
                .ready
                .iter()
                .any(|model| matches!(model.backend, Backend::Vllm | Backend::SgLang));
        }
        self.note_row_engines();
        if self.running_up {
            // A model that left `ready` in a good read was unloaded (or is
            // loading again), and one whose server changed is a new
            // process: its next load is probed afresh, its engine windows
            // start over, its counters count the next process from 0 and
            // its prompt counters go back to activity rows until `/metrics`
            // takes over again (#46).
            self.detected.retain(|id, _| ready.contains(id));
            self.probe_snapshot.retain(|id, _| ready.contains(id));
            self.schema_logged.retain(|id| ready.contains(id));
            self.health_logged.retain(|id| ready.contains(id));
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

    /// Remember each ready model's server for its activity rows (#82). A
    /// model still waiting for its `/metrics` probe is not known yet.
    fn note_row_engines(&mut self) {
        for model in &self.ready {
            if model.probe && !self.detected.contains_key(&model.id) {
                continue;
            }
            let key = activity::model_key(&model.id);
            if self.row_engines.len() >= MAX_ROW_ENGINES && !self.row_engines.contains_key(&key) {
                continue;
            }
            self.row_engines.insert(key, model.backend);
        }
    }

    /// Drop `id`'s engine windows and attributed-speed reads (#31, #35).
    fn forget_engine(&mut self, id: &str) {
        self.engines.forget(id);
        self.speeds.forget(id);
        self.sglang.forget(id);
        self.vllm_phase.forget(id);
        self.series.forget_gauges(id);
    }

    /// `id`'s process is gone: the next one starts its counters at 0 and
    /// its prompt counters come from activity rows until `/metrics` gives
    /// both again (#46).
    fn forget_process(&mut self, id: &str) {
        self.live.forget(id);
        self.loads_missing.remove(id);
        self.counter.forget(id);
        self.lcpp_tokens.forget(id);
        self.prompt_counter.forget(id);
        self.prompt_box.forget(id);
        self.prompt_cache.forget(id);
        self.series.forget_process(id);
        self.tqllm.forget(id);
        // A new process starts with a cold prefix cache: its first request
        // is all new, not a diff against the old process's last one.
        self.request_marks.forget(id);
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
            // #180: `/metrics` already said nothing this load.
            if self.probe_snapshot.contains_key(&id) {
                self.probe_tqllm(&id);
                continue;
            }
            let read = match self.upstream_read(&id, "metrics", self.limits.metrics_timeout, |_| {
                SERVER_METRICS_CAP
            }) {
                Gate::Read(model, read) if model.probe && !self.detected.contains_key(&id) => read,
                Gate::Read(..) | Gate::Skipped | Gate::NotLoaded => continue,
            };
            let read = exact(read)
                .map_err(TapError::label)
                .and_then(|bytes| String::from_utf8(bytes).map_err(|_| "malformed"));
            let why = match read.as_deref().map(detect_backend) {
                Ok(Some(kind)) => {
                    self.settle_probe(&id, kind, &format!("/metrics says {}", kind.as_str()));
                    continue;
                }
                Ok(None) => "no known metric names".to_owned(),
                Err(label) => format!("no /metrics ({label})"),
            };
            // A non-2xx answer skips the model for the rest of this round:
            // the snapshot then waits for the next one.
            self.probe_snapshot.insert(id.clone(), why);
            self.probe_tqllm(&id);
        }
    }

    /// #180: the `/metrics` probe found no known server, so one
    /// `GET /upstream/<id>/api/monitor/v1/snapshot` through the fresh gate
    /// tells tqLLM. A model the gate skips keeps this for a later round.
    fn probe_tqllm(&mut self, id: &str) {
        let read = match self.upstream_read(
            id,
            "api/monitor/v1/snapshot",
            self.limits.metrics_timeout,
            |_| TQLLM_CAP,
        ) {
            Gate::Read(model, read) if model.probe && !self.detected.contains_key(id) => read,
            Gate::Read(..) | Gate::Skipped | Gate::NotLoaded => return,
        };
        let why = self.probe_snapshot.remove(id).unwrap_or_default();
        let read = exact(read)
            .map_err(TapError::label)
            .and_then(|bytes| String::from_utf8(bytes).map_err(|_| "malformed"));
        let (kind, note) = match read.as_deref() {
            Ok(body) if is_tqllm_json(body) => (
                Backend::Tqllm,
                format!("{why}; /api/monitor/v1/snapshot says tqllm"),
            ),
            Ok(_) => (
                Backend::OpenAi,
                format!("{why}; no tqLLM snapshot; staying openai"),
            ),
            Err(label) => (
                Backend::OpenAi,
                format!("{why}; no tqLLM snapshot ({label}); staying openai"),
            ),
        };
        self.settle_probe(id, kind, &note);
    }

    /// This load's probe found `kind` for `id` (#31, #180): remember it,
    /// log it once, and read `/running` again for its container flags.
    fn settle_probe(&mut self, id: &str, kind: Backend, note: &str) {
        let id = id.to_owned();
        {
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
        // The requests in flight as of now, for this round's token shares
        // (#80): a request listed late would only get a lower bound.
        if let Some(events) = &self.events {
            self.events_view = events.view();
        }
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
            // #180: tqLLM's snapshot in place of `/metrics`.
            if listed.backend == Backend::Tqllm {
                self.poll_tqllm(&listed.id);
                continue;
            }
            // The engine reads its counters during the request. The stamp
            // below is after the parse, so a finish in between can be absent
            // from this scrape and still have a stamp later than it.
            let before = SystemTime::now();
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
            let mut ttft_labeled = true;
            let read = exact(read)
                .map_err(TapError::label)
                .and_then(|bytes| String::from_utf8(bytes).map_err(|_| "malformed"))
                .map(|text| {
                    ttft_labeled = crate::metrics::sglang_ttft_labeled(&text);
                    parse_metrics_full(model.backend, &text)
                });
            let now = Instant::now();
            let wall = SystemTime::now();
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
                if model.backend == Backend::Vllm {
                    let key = activity::model_key(&model.id);
                    self.live
                        .observe_vllm(&model.id, &key, now, sample, &self.events_view);
                }
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
                if model.backend == Backend::Vllm {
                    self.vllm_phase.observe(&model.id, sample, now);
                }
                if model.backend == Backend::SgLang {
                    self.sglang.observe_scrape(
                        &model.id,
                        now,
                        before,
                        wall,
                        Totals::from_sample(sample),
                        ttft_labeled,
                    );
                }
                self.gauges
                    .insert(model.id.clone(), Gauges::of(sample, engine, live, now));
                if let Some(prompt) = sample.prompt_total {
                    self.prompt_counter.observe(&model.id, prompt, None, now);
                }
            }
            match (decoded, read) {
                (Some(_), Ok(sample)) if model.backend == Backend::LlamaCpp => {
                    self.fallback.remove(&model.id);
                    self.observe_lcpp(&model.id, &sample, now);
                }
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
        self.poll_loads();
        self.latencies.metrics = Some(started.elapsed());
    }

    /// A llama.cpp `/metrics` read into the box counter: its tokens from
    /// [`LcppTokens`], not its decode calls (DER-451). While they are
    /// unknown the model's gauge is kept and the box total is not fresh.
    fn observe_lcpp(&mut self, id: &str, sample: &MetricsSample, now: Instant) {
        let read = MetricsRead {
            calls: sample.n_decode_total,
            predicted: sample.predicted_total,
            busy_per_decode: sample.busy_per_decode,
            processing: sample.requests_processing,
        };
        match self.lcpp_tokens.observe_metrics(id, &read, now) {
            Some(tokens) => self
                .counter
                .observe(id, tokens, sample.requests_processing, now),
            None => self
                .counter
                .observe_unknown(id, sample.requests_processing, now),
        }
    }

    /// One tqLLM snapshot read (#180), through the fresh gate, in place of
    /// `/metrics`: the counters, gauges, engine facts and active requests,
    /// then `/health` when this load or instance has not had it.
    fn poll_tqllm(&mut self, id: &str) {
        let before = SystemTime::now();
        let (model, read) = match self.upstream_read(
            id,
            "api/monitor/v1/snapshot",
            self.limits.metrics_timeout,
            |_| TQLLM_CAP,
        ) {
            Gate::Read(model, read) if model.backend == Backend::Tqllm => (model, read),
            Gate::Read(..) | Gate::Skipped | Gate::NotLoaded => return,
        };
        let text = exact(read)
            .map_err(TapError::label)
            .and_then(|bytes| String::from_utf8(bytes).map_err(|_| "malformed"));
        let snap = match text.map(|text| parse_tqllm(&text)) {
            Ok(Ok(snap)) => snap,
            Ok(Err(TqError::Schema(version))) => {
                // Not a schema this reader knows: no metrics, as if absent.
                self.fallback.insert(model.id.clone());
                if self.schema_logged.insert(model.id.clone()) {
                    let version = version.map_or_else(|| "?".to_owned(), |v| v.to_string());
                    log::emit(
                        &mut self.log,
                        Priority::Info,
                        &format!(
                            "{}: tqLLM snapshot schema_version {version} is not 1; using llama-swap activity",
                            model.id
                        ),
                    );
                }
                return;
            }
            Ok(Err(TqError::Depth)) => {
                return self.use_activity(&model, "snapshot nested too deep");
            }
            Ok(Err(TqError::Malformed)) => return self.use_activity(&model, "malformed snapshot"),
            Err(label) => return self.use_activity(&model, label),
        };
        let now = Instant::now();
        let wall = SystemTime::now();
        if self.tqllm.is_restart(&model.id, &snap) {
            self.tqllm_restarted(&model.id);
        }
        let observed = self.tqllm.observe(&model.id, &snap, now, before, wall);
        if observed.log_gap {
            log::emit(
                &mut self.log,
                Priority::Info,
                &format!(
                    "{}: {} tqLLM request(s) left its history before they were seen finished; their unseen tokens are not counted",
                    model.id, observed.gap
                ),
            );
        }
        let mut facts = observed.facts;
        let live = facts.live.take();
        self.engine_facts.insert(model.id.clone(), facts);
        let sample = observed.sample;
        self.series
            .observe_metrics(&model.id, model.backend, &sample, now);
        self.prompt_cache
            .observe_metrics(&model.id, sample.prompt_total, sample.cached_total);
        let engine = self.engines.observe(&model.id, &sample);
        self.gauges
            .insert(model.id.clone(), Gauges::of(&sample, engine, live, now));
        if let Some(prompt) = sample.prompt_total {
            self.prompt_counter.observe(&model.id, prompt, None, now);
        }
        if let Some(value) = sample.n_decode_total {
            self.fallback.remove(&model.id);
            self.counter
                .observe(&model.id, value, sample.requests_processing, now);
        }
        if let Some(prompt) = sample.prompt_total {
            self.prompt_box.observe(&model.id, prompt, None, now);
        }
        if self.tqllm.health_due(&model.id) {
            self.poll_health(&model.id);
        }
    }

    /// A new tqLLM `instance_id` for `id` (#180): its counters, engine
    /// numbers and book start again, logged.
    fn tqllm_restarted(&mut self, id: &str) {
        log::emit(
            &mut self.log,
            Priority::Info,
            &format!("{id}: tqLLM restarted (new instance_id); its counters start again"),
        );
        self.forget_engine(id);
        self.forget_process(id);
    }

    /// tqLLM's `GET /upstream/<id>/health` (#180), once per load and once
    /// per `instance_id`: its `fit` becomes SETUP's `engine:fit`. A failed
    /// read counts as read; the next instance asks again.
    fn poll_health(&mut self, id: &str) {
        let read =
            match self.upstream_read(id, "health", self.limits.metrics_timeout, |_| TQLLM_CAP) {
                Gate::Read(model, read) if model.backend == Backend::Tqllm => read,
                Gate::Read(..) | Gate::Skipped | Gate::NotLoaded => return,
            };
        let fit = match exact(read).map_err(TapError::label) {
            Ok(bytes) => String::from_utf8(bytes)
                .ok()
                .and_then(|text| parse_tqllm_health(&text).ok())
                .flatten(),
            Err(label) => {
                if self.health_logged.insert(id.to_owned()) {
                    log::emit(
                        &mut self.log,
                        Priority::Info,
                        &format!("{id}: no tqLLM /health ({label}); no fit shown"),
                    );
                }
                None
            }
        };
        self.note_fit(id, fit);
    }

    /// Keep `/health`'s `fit` for `id` and show it now (#180).
    fn note_fit(&mut self, id: &str, fit: Option<String>) {
        self.tqllm.set_fit(id, fit.clone());
        if let Some(facts) = self.engine_facts.get_mut(id) {
            match fit {
                Some(fit) => {
                    facts.values.insert("fit", fit);
                }
                None => {
                    facts.values.remove("fit");
                }
            }
        }
    }

    /// Start llama-swap's `/api/events` stream while llama-swap answers (or
    /// is away and it ran), and take what it says for this round (#80,
    /// #128). It runs with nothing loaded too: the GPU reservation and the
    /// claims are shown then (#167).
    fn sync_events(&mut self) {
        let wanted = self.running_up || self.events.is_some();
        if !wanted {
            self.events = None;
        } else if self.events.is_none() {
            match Events::spawn(
                &self.limits.url,
                self.limits.running_timeout.max(EVENTS_TIMEOUT),
            ) {
                Ok(events) => self.events = Some(events),
                Err(err) => log::emit(
                    &mut self.log,
                    Priority::Warning,
                    &format!("events: cannot start its thread: {err}"),
                ),
            }
        }
        self.events_view = self.events.as_ref().map(Events::view).unwrap_or_default();
        let failure = self.events_view.failure;
        note_flag(&mut self.log, &mut self.events_failed, failure, "events");
        if self.events_view.claims_error.is_some() {
            if !self.claims_failed {
                log::emit(
                    &mut self.log,
                    Priority::Warning,
                    "events: a claims frame did not parse",
                );
                self.claims_failed = true;
            }
        } else {
            self.claims_failed = false;
        }
        if self.events_view.gpu_error.is_some() {
            if !self.gpu_failed {
                log::emit(
                    &mut self.log,
                    Priority::Warning,
                    "events: a gpu frame did not parse",
                );
                self.gpu_failed = true;
            }
        } else {
            self.gpu_failed = false;
        }
        self.live.prune(&self.events_view);
    }

    /// One gated `/v1/loads` read for each ready SGLang model with a
    /// request in flight (#80), at the metrics cadence.
    fn poll_loads(&mut self) {
        let todo: Vec<String> = self
            .ready
            .iter()
            .filter(|model| model.backend == Backend::SgLang)
            .filter(|model| !self.loads_missing.contains(&model.id))
            .filter(|model| {
                let key = activity::model_key(&model.id);
                self.events_view.requests.iter().any(|req| req.model == key)
            })
            .map(|model| model.id.clone())
            .collect();
        let mut failure: Option<&'static str> = None;
        for id in todo {
            let (model, read) = match self.upstream_read(
                &id,
                "v1/loads?include=core",
                self.limits.metrics_timeout,
                |_| LOADS_CAP,
            ) {
                Gate::Read(model, read) if model.backend == Backend::SgLang => (model, read),
                Gate::Read(..) | Gate::Skipped | Gate::NotLoaded => continue,
            };
            let loads = match exact(read) {
                Ok(bytes) => parse_loads(&bytes),
                Err(TapError::Status) => {
                    self.loads_missing.insert(model.id.clone());
                    log::emit(
                        &mut self.log,
                        Priority::Info,
                        &format!(
                            "{}: no /v1/loads from sglang; its in-flight rows show no tokens",
                            model.id
                        ),
                    );
                    continue;
                }
                Err(err) => {
                    failure = failure.or(Some(err.label()));
                    continue;
                }
            };
            match loads {
                Some(loads) => {
                    let key = activity::model_key(&model.id);
                    self.live
                        .observe_loads(&key, Instant::now(), &loads, &self.events_view);
                }
                None => {
                    failure = failure.or(Some("malformed"));
                }
            }
        }
        note_flag(&mut self.log, &mut self.loads_failed, failure, "loads");
    }

    /// The vLLM and SGLang requests in flight now (#80), oldest first.
    fn live_requests(&self, view: &EventsView) -> Vec<LiveRequest> {
        let mut out = Vec::new();
        for req in &view.requests {
            let Some(model) = self.ready.iter().find(|model| {
                matches!(model.backend, Backend::Vllm | Backend::SgLang)
                    && activity::model_key(&model.id) == req.model
            }) else {
                continue;
            };
            out.push(LiveRequest {
                model: model.name.clone(),
                id: req.id.clone(),
                engine: model.backend,
                started: req.started,
                started_wall: req.started_wall,
                resp_bytes: req.resp_bytes,
                tokens: self.live.tokens(&req.model, &req.id),
            });
        }
        out
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
            // An SGLang without `/v1/loads` answers 404: that says nothing
            // of the model, whose gauges stay (#80).
            Err(TapError::Status) if leaf == LOADS_LEAF => {}
            // #180: a tqLLM without `/health`, or one that refuses it,
            // loses only its `fit`; its gauges stay.
            Err(TapError::Status | TapError::Unauthorized) if leaf == HEALTH_LEAF => {}
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
            // Nothing is processing, so `/slots` is not read this round
            // (#98): whatever prefill episode was running is over, and the
            // held rate must not stay published as if it still were.
            self.slots.idle();
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
                        self.lcpp_tokens.observe_slots(
                            &model.id,
                            &self.slots.tokens(&model.id),
                            Instant::now(),
                        );
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
                Some(mut rows) => {
                    for row in &mut rows {
                        row.engine = self.row_engines.get(&row.model).copied();
                    }
                    let merged = self.recent.merge(rows);
                    if merged.restarted {
                        log::emit(
                            &mut self.log,
                            Priority::Info,
                            "activity: llama-swap restarted; its row ids start again",
                        );
                        // #180: no tqLLM fill carries over to the new ids.
                        self.tqllm.forget_rows();
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
            self.sglang.activity(now, &[]);
            return;
        }
        let rows = &merged.new;
        // #35: new rows of a model without `/slots` that llama-swap gave
        // no speeds wait for the engine's.
        let mut measure: Vec<(u64, String)> = Vec::new();
        // #180: a tqLLM row takes only its own request's numbers, below.
        for model in self
            .ready
            .iter()
            .filter(|model| !model.backend.has_slots() && model.backend != Backend::Tqllm)
        {
            let key = activity::model_key(&model.id);
            measure.extend(
                rows.iter()
                    .filter(|row| row.model == key)
                    .filter(|row| row.prompt_tps.is_none() || row.gen_tps.is_none())
                    .map(|row| (row.seq, model.id.clone())),
            );
        }
        self.speeds.activity(now, &measure);
        let mut claims = Vec::new();
        for model in self
            .ready
            .iter()
            .filter(|model| model.backend == Backend::SgLang)
        {
            let key = activity::model_key(&model.id);
            for row in rows.iter().filter(|row| row.model == key) {
                let Some(input) = row.input_tokens else {
                    continue;
                };
                claims.push(Claim {
                    seq: row.seq,
                    model: model.id.clone(),
                    input,
                    output: row.output_tokens,
                    duration_ms: row.duration_ms,
                    prompt_known: row.prompt_tps.is_some(),
                    gen_known: row.gen_tps.is_some(),
                    timestamp: Some(row.time.clone()),
                });
            }
        }
        self.sglang.activity(now, &claims);
        // #180: tqLLM rows wait for the one finished request that matches.
        let mut tq_rows = Vec::new();
        for model in self
            .ready
            .iter()
            .filter(|model| model.backend == Backend::Tqllm)
        {
            let key = activity::model_key(&model.id);
            for row in rows.iter().filter(|row| row.model == key) {
                let (Some(prompt), Some(output)) =
                    (row.prompt_for(Some(Backend::Tqllm)), row.output_tokens)
                else {
                    continue;
                };
                tq_rows.push(RowClaim {
                    seq: row.seq,
                    model: model.id.clone(),
                    input: prompt.whole,
                    output,
                    timestamp: Some(row.time.clone()),
                });
            }
        }
        self.tqllm.activity(now, &tq_rows);
        for model in &self.ready {
            let key = activity::model_key(&model.id);
            for row in rows.iter().filter(|row| row.model == key) {
                self.prompt_cache.add_row(
                    &model.id,
                    row.prompt_for(Some(model.backend)),
                    row.cached_tokens.is_some(),
                );
                self.series
                    .add_row(&model.id, row, model.backend.has_slots());
            }
        }
        let window = self.limits.activity_interval + self.limits.activity_timeout;
        let fallback: Vec<(String, Option<Backend>)> = self
            .fallback
            .iter()
            .map(|id| {
                let backend = self.ready.iter().find(|m| m.id == *id).map(|m| m.backend);
                (id.clone(), backend)
            })
            .collect();
        for (id, backend) in &fallback {
            let key = activity::model_key(id);
            let new_rows = || rows.iter().filter(|row| row.model == key);
            let tokens = new_rows()
                .filter_map(|row| row.output_tokens)
                .fold(0u64, u64::saturating_add);
            self.counter.add(id, tokens, now, window);
            self.series.add_generation(id, tokens);
            let prompt = new_rows()
                .filter_map(|row| row.prompt_for(*backend))
                .map(|prompt| prompt.whole)
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
                Some(mut text) => {
                    let input_new = self.request_marks.diff_and_keep(&model.id, &mut text);
                    let input =
                        prompt_cells(&text.input, self.limits.input_tail, self.limits.prompt_view);
                    let input_new = input_new.map(|new| {
                        shown_new(
                            new,
                            text.input.chars().count(),
                            self.limits.input_tail,
                            input.len(),
                        )
                    });
                    let view = CaptureView {
                        model: model.name.clone(),
                        id: row.id,
                        input,
                        input_new,
                        input_note: text.input_note,
                        output: tail_cells(&text.output, self.limits.output_tail),
                        output_chars: Some(text.output.chars().count()).filter(|chars| *chars > 0),
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
        // One fresh look at the event stream for the whole sample (#80).
        let events = self.events.as_ref().map(Events::view).unwrap_or_default();
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
        // Prompt tok/s from the counter, for a streaming engine with no
        // `/slots` rate. A burst engine does not use it (#188).
        let prompt_rate = self.prompt_rate.observe(now, self.prompt_counter.total());
        // An idle llama.cpp slot stores 0. That is not a live prefill, so
        // it must not hide a vLLM or SGLang rate or bring the wall-clock
        // spike back. A slot that is actually in flight still reports its
        // rate, including 0 while it is decoding (#188).
        let slots_rate = self.slots_prompt_rate(now);
        let burst = self.burst_prompt_engine(now);
        let llama_busy = self.ready.iter().any(|model| {
            model.backend == Backend::LlamaCpp
                && self
                    .counter
                    .requests_processing(&model.id, now)
                    .is_some_and(|n| n > 0.0)
        });
        let mut chart_prompt_tps = None;
        let prompt_tps = if !self.running_up {
            None
        } else if llama_busy {
            // /slots supplies cumulative processed tokens, not elapsed prefill
            // time. Poll-to-poll deltas can land a whole batch in a tiny gap.
            // Prefer completed per-request engine timings for each busy model.
            let rows = self.measured_activity_limit(crate::recent::RING);
            let mut rates = Vec::new();
            let mut missing = false;
            let mut chart_sum = 0.0;
            let mut chart_known = true;
            for model in &self.ready {
                if !self
                    .counter
                    .requests_processing(&model.id, now)
                    .is_some_and(|n| n > 0.0)
                {
                    continue;
                }
                let rate = match model.backend {
                    Backend::LlamaCpp => {
                        let key = activity::model_key(&model.id);
                        crate::prompt_head::last_five(
                            rows.iter()
                                .filter(|row| row.model == key)
                                .map(|row| row.prompt_tps.or(row.engine_prompt_tps)),
                        )
                    }
                    Backend::Vllm => self.engines.prefill_tps(&model.id),
                    Backend::SgLang => self.sglang.prompt_rate(&model.id),
                    Backend::Tqllm => self
                        .gauges
                        .get(&model.id)
                        .filter(|gauges| {
                            now.saturating_duration_since(gauges.at)
                                <= FRESH_GAUGES.max(self.limits.metrics_interval * 2)
                        })
                        .and_then(|gauges| gauges.live.as_ref())
                        .and_then(|live| live.prefill_tps_sum()),
                    _ => None,
                };
                let chart_rate = match model.backend {
                    Backend::LlamaCpp => {
                        let active = self
                            .counter
                            .requests_processing(&model.id, now)
                            .unwrap_or(0.0);
                        match self.slots.chart_phase(
                            &model.id,
                            active,
                            now,
                            self.limits.slots_interval * 2,
                        ) {
                            Some(ChartPhase::Decode) => Some(0.0),
                            Some(ChartPhase::Prefill) => rate,
                            None => None,
                        }
                    }
                    Backend::Vllm | Backend::SgLang => self.burst_chart_rate(model, now),
                    _ => rate,
                };
                if let Some(value) = chart_rate {
                    chart_sum += value;
                } else {
                    chart_known = false;
                }
                if let Some(rate) = rate {
                    rates.push(rate);
                } else {
                    missing = true;
                }
            }
            if chart_known {
                chart_prompt_tps = Some(chart_sum);
            }
            if missing || rates.is_empty() {
                None
            } else {
                Some(rates.iter().sum())
            }
        } else {
            crate::prompt_head::headline(
                self.running_up,
                busy,
                slots_rate,
                prompt_rate,
                burst,
                self.held_burst_prompt(now),
            )
        };
        // #180: with only tqLLM loaded, GEN and PROMPT are its requests'
        // own rates, `--` until it has them.
        let tq_rates = self.tqllm_rates(now).filter(|_| self.running_up);
        let tq_rated = tq_rates.is_some();
        let (gen_tps, prompt_tps) = tq_rates.unwrap_or((gen_tps, prompt_tps));
        if !llama_busy {
            // DER-451: the headline holds a burst engine's last prefill
            // speed through the whole request. The chart shows it only
            // while a request is prefilling, and 0 while all of them decode.
            let slots_live = slots_rate.is_some_and(|rate| rate.is_finite() && rate >= 0.0);
            chart_prompt_tps = if self.running_up && busy && burst && !slots_live && !tq_rated {
                self.burst_chart_prompt(now)
            } else {
                prompt_tps
            };
        }
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
            chart_prompt_tps,
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
            series: self.model_series(now, &events),
            requests: self.live_requests(&events),
            jobs: self.jobs.jobs().into_iter().cloned().collect(),
            claims: events.claims.clone(),
            gpu: events.gpu.clone(),
            displaced: events.displaced.clone(),
        };
        if Arc::strong_count(&self.tx.slot) == 1 {
            return Err(());
        }
        self.tx.put((view, detail));
        Ok(())
    }

    /// GEN and PROMPT from tqLLM's requests (#180) when every ready model
    /// with telemetry is tqLLM and each has a fresh snapshot: the sum of
    /// `decode_tokens_per_second` over the requests in decode, and of
    /// `prefill_tokens_per_second` over those in prefill. Each is 0 with
    /// no such request and `None` while they have no rate yet. `None` for
    /// any other mix of models: the counters' rates stay.
    fn tqllm_rates(&self, now: Instant) -> Option<(Option<f64>, Option<f64>)> {
        let fresh = FRESH_GAUGES.max(self.limits.metrics_interval * 2);
        let mut metered = self
            .ready
            .iter()
            .filter(|model| model.backend.has_metrics())
            .peekable();
        metered.peek()?;
        let mut decode = Some(0.0);
        let mut prompt = Some(0.0);
        for model in metered {
            if model.backend != Backend::Tqllm {
                return None;
            }
            let live = self
                .gauges
                .get(&model.id)
                .filter(|gauges| now.saturating_duration_since(gauges.at) <= fresh)?
                .live
                .as_ref()?;
            decode = decode.zip(live.decode_tps_sum()).map(|(a, b)| a + b);
            prompt = prompt.zip(live.prefill_tps_sum()).map(|(a, b)| a + b);
        }
        Some((decode, prompt))
    }

    /// `/slots` prefill while a llama.cpp model is actually in flight.
    /// `idle()` stores 0 when nothing with `/slots` is processing, and that
    /// stored 0 must not hide a burst engine (#188). Decoding really is 0,
    /// so an in-flight slot still reports it, including 0.
    fn slots_prompt_rate(&self, now: Instant) -> Option<f64> {
        let in_flight = self.ready.iter().any(|model| {
            model.backend.has_slots()
                && self
                    .counter
                    .requests_processing(&model.id, now)
                    .is_some_and(|value| value > 0.0)
        });
        if !in_flight {
            return None;
        }
        self.slots
            .prompt_tps()
            .filter(|rate| rate.is_finite() && *rate >= 0.0)
    }

    /// The models busy right now are all vLLM or SGLang (#188). An idle
    /// llama.cpp or Strata sitting loaded does not count, so its stored 0
    /// cannot bring the wall-clock spike back onto a burst prefill. A busy
    /// streaming engine does. tqLLM is left out: its own rates replace the
    /// headline just after this choice.
    fn burst_prompt_engine(&self, now: Instant) -> bool {
        let mut saw = false;
        for model in &self.ready {
            if !model.backend.has_metrics() || model.backend == Backend::Tqllm {
                continue;
            }
            let busy = self
                .counter
                .requests_processing(&model.id, now)
                .is_some_and(|value| value > 0.0);
            if !busy {
                continue;
            }
            if !matches!(model.backend, Backend::Vllm | Backend::SgLang) {
                return false;
            }
            saw = true;
        }
        saw
    }

    /// Honest prefill tok/s of the burst engines that are busy now (#188).
    /// vLLM's is the held histogram window. SGLang's is the last rate from
    /// one finished request that was alone. Several busy models add.
    fn held_burst_prompt(&self, now: Instant) -> Option<f64> {
        let mut sum = 0.0;
        let mut n = 0u32;
        for model in &self.ready {
            let processing = self
                .counter
                .requests_processing(&model.id, now)
                .unwrap_or(0.0);
            if processing <= 0.0 {
                continue;
            }
            let rate = match model.backend {
                Backend::Vllm => self.engines.prefill_tps(&model.id),
                Backend::SgLang => self.sglang.prompt_rate(&model.id),
                _ => None,
            };
            if let Some(rate) = rate.filter(|rate| rate.is_finite() && *rate >= 0.0) {
                sum += rate;
                n += 1;
            }
        }
        (n > 0).then_some(sum)
    }

    /// READ chart input of one busy burst-engine model (DER-451): its held
    /// prefill speed while one of its requests is prefilling, 0 while every
    /// running one decodes, `None` when the phase is unknown. vLLM's phase
    /// is [`VllmPhaseBook`]'s. SGLang's is never known here: its `/metrics`
    /// gauges lag its requests by up to 40 decode steps ([`crate::live`]),
    /// so its running count cannot be set against its first-token count.
    /// Its chart is unknown while it is busy, not its held speed.
    fn burst_chart_rate(&self, model: &ReadyModel, now: Instant) -> Option<f64> {
        if model.backend != Backend::Vllm {
            return None;
        }
        let fresh = FRESH_GAUGES.max(self.limits.metrics_interval * 2);
        match self.vllm_phase.phase(&model.id, now, fresh)? {
            ChartPhase::Decode => Some(0.0),
            ChartPhase::Prefill => self
                .engines
                .prefill_tps(&model.id)
                .filter(|rate| rate.is_finite() && *rate >= 0.0),
        }
    }

    /// [`Self::burst_chart_rate`] of every busy vLLM and SGLang model,
    /// added (DER-451). One unknown makes the sum unknown.
    fn burst_chart_prompt(&self, now: Instant) -> Option<f64> {
        let mut sum = 0.0;
        for model in &self.ready {
            if !matches!(model.backend, Backend::Vllm | Backend::SgLang)
                || !self
                    .counter
                    .requests_processing(&model.id, now)
                    .is_some_and(|n| n > 0.0)
            {
                continue;
            }
            sum += self.burst_chart_rate(model, now)?;
        }
        Some(sum)
    }

    /// [`Self::model_setup`] with each model's engine-reported values (#54),
    /// and for tqLLM the KV its command asked for (#180).
    fn setup_with_engine(&self) -> Vec<ModelSetup> {
        let mut setup = self.model_setup.clone();
        for (((entry, id), model), tq_kv) in setup
            .iter_mut()
            .zip(&self.model_ids)
            .zip(&self.models)
            .zip(&self.model_tq_kv)
        {
            if let Some(facts) = self.engine_facts.get(id) {
                entry.engine = facts.values.clone();
            }
            if !model
                .backend
                .is_some_and(|info| info.kind == Backend::Tqllm)
            {
                continue;
            }
            // No KV flag: no value, and SETUP draws the default dim.
            let asked = match tq_kv {
                Some(TqKv::Default) => None,
                Some(TqKv::Named) => crate::tqllm::kv_asked(model.detail.as_ref()),
                Some(TqKv::Unreadable) | None => Some(crate::tqllm::KV_UNKNOWN.to_owned()),
            };
            if let Some(kv) = asked {
                entry.engine.insert("kv_asked", kv);
            }
        }
        setup
    }

    /// Each listed model's cumulative numbers (#71), in the order of
    /// [`Self::models`], with the version its engine reports.
    fn model_series(&self, now: Instant, events: &EventsView) -> Vec<ModelSeries> {
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
                // #80: llama-swap's own count, while its stream is up
                // for vLLM or SGLang. Claims may open the stream for
                // llama.cpp too (#128); that must not add an inflight
                // field to the snapshot.
                inflight: (events.connected && self.inflight_engines).then(|| {
                    let key = activity::model_key(id);
                    let n = events
                        .requests
                        .iter()
                        .filter(|req| req.model == key)
                        .count();
                    u16::try_from(n).unwrap_or(u16::MAX)
                }),
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
    /// the engine's speeds where llama-swap gave none (#35) and, for one
    /// SGLang request, the counters of that request (#133). The fill is
    /// applied on this copy: the prompt counters and the snapshot never
    /// see it. A value llama-swap reported is left as it was. A missing
    /// cached count is filled even when a rate was reported, and a missing
    /// rate is filled the same way. The one exception is tqLLM's prompt
    /// rate (#180): llama-swap's is the whole prompt, cached part
    /// included, over the prefill time, so the matched request's own
    /// uncached rate replaces it.
    fn measured_activity(&self) -> Vec<ActivityRow> {
        self.measured_activity_limit(self.activity_rows())
    }

    fn measured_activity_limit(&self, limit: usize) -> Vec<ActivityRow> {
        let mut rows = self.recent.rows(limit);
        for row in &mut rows {
            if let Some(speeds) = self.speeds.speeds(row.seq) {
                if row.prompt_tps.is_none() {
                    row.engine_prompt_tps = speeds.prefill;
                }
                if row.gen_tps.is_none() {
                    row.engine_gen_tps = speeds.decode;
                }
            }
            // #180: a tqLLM row's one matching finished request. Its own
            // prompt rate replaces llama-swap's, which counts the cache.
            if let Some(fill) = self.tqllm.fill(row.seq) {
                if row.cached_tokens.is_none() {
                    row.cached_tokens = Some(fill.cached);
                }
                if fill.prompt_tps.is_some() {
                    row.prompt_tps = fill.prompt_tps;
                }
                if row.gen_tps.is_none() && row.engine_gen_tps.is_none() {
                    row.engine_gen_tps = fill.gen_tps;
                }
            }
            let Some(fill) = self.sglang.fill(row.seq) else {
                continue;
            };
            if row.cached_tokens.is_none() {
                row.cached_tokens = Some(fill.cached);
            }
            if row.prompt_tps.is_none() && row.engine_prompt_tps.is_none() {
                row.engine_prompt_tps = fill.prompt_tps;
            }
            if row.gen_tps.is_none() && row.engine_gen_tps.is_none() {
                row.engine_gen_tps = fill.gen_tps;
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
        for (((model, id), layout), tq_kv) in models
            .iter_mut()
            .zip(&self.model_ids)
            .zip(&self.model_kv)
            .zip(&self.model_tq_kv)
        {
            if let Some(facts) = self.engine_facts.get(id) {
                with_facts(&mut model.detail, facts);
            }
            // #180: a tqLLM command that named no KV flag at all runs
            // tqLLM's default unless the fit picked another; the snapshot
            // does not say. An unreadable one, or flags not read as
            // tqLLM's, stay unknown.
            if model
                .backend
                .is_some_and(|info| info.kind == Backend::Tqllm)
                && *tq_kv == Some(TqKv::Default)
                && let Some(detail) = model.detail.as_mut()
            {
                for side in [&mut detail.kv_k, &mut detail.kv_v] {
                    side.get_or_insert_with(|| crate::sources::cmdline::KV_TQLLM.to_owned());
                }
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
            // #81: Strata in parallel mode runs one request per batch slot.
            if let Some(parallel) = gauges
                .live
                .as_ref()
                .and_then(|live| live.parallel)
                .filter(|n| *n >= 2)
            {
                info.max_running = backend::reqs(f64::from(parallel));
            }
            // #180: tqLLM's `engine.slots` replaces `--np`.
            if let Some(slots) = gauges.live.as_ref().and_then(|live| live.request_slots) {
                info.max_running = backend::reqs(f64::from(slots));
            }
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

/// Fill `detail`'s ctx when the launch command gave none (or replace it
/// with tqLLM's per-slot context, #180), its KV when it is unknown or
/// `auto`, and the KV block size and prefix caching (#31), from the
/// server's own report.
fn with_facts(detail: &mut Option<llama_core::detail::ModelDetail>, facts: &EngineFacts) {
    if *facts == EngineFacts::default() {
        return;
    }
    let detail = detail.get_or_insert_with(Default::default);
    if detail.ctx.is_none() || (facts.ctx_exact && facts.ctx.is_some()) {
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

/// Message marks of the last [`REQUEST_MARKS`] captured requests per model
/// id, for IN's new part (DER-451). Hashes and roles, no text. At most
/// [`REQUEST_MARK_MODELS`] models; a new one evicts the least recently used.
#[derive(Debug, Default)]
struct RequestMarks {
    tick: u64,
    models: HashMap<String, (u64, std::collections::VecDeque<Vec<MessageMark>>)>,
}

impl RequestMarks {
    /// IN's new characters in `text` against `model`'s kept requests, which
    /// this one then joins. Its marks are moved out of `text`.
    fn diff_and_keep(
        &mut self,
        model: &str,
        text: &mut crate::capture::CaptureText,
    ) -> Option<usize> {
        let new = new_input_chars(
            self.models
                .get(model)
                .into_iter()
                .flat_map(|(_, list)| list)
                .map(Vec::as_slice),
            text,
        );
        let marks = std::mem::take(&mut text.messages);
        if marks.is_empty() {
            return new;
        }
        if !self.models.contains_key(model)
            && self.models.len() >= REQUEST_MARK_MODELS
            && let Some(oldest) = self
                .models
                .iter()
                .min_by_key(|(_, (used, _))| *used)
                .map(|(id, _)| id.clone())
        {
            self.models.remove(&oldest);
        }
        self.tick += 1;
        let (used, list) = self.models.entry(model.to_owned()).or_default();
        *used = self.tick;
        list.push_back(marks);
        while list.len() > REQUEST_MARKS {
            list.pop_front();
        }
        new
    }

    /// `model`'s process is gone: its next request has nothing to diff.
    fn forget(&mut self, model: &str) {
        self.models.remove(model);
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

    #[test]
    fn llama_headline_uses_completed_engine_rates_not_batch_poll_spike() {
        let mut state = blank_state();
        state.ready = vec![ReadyModel {
            id: "cpp".into(),
            name: "cpp".into(),
            backend: Backend::LlamaCpp,
            probe: false,
        }];
        state.running_up = true;
        let now = Instant::now();
        state.counter.observe("cpp", 1, Some(1.0), now);
        let rows = activity::parse_activity(br#"{"data":[
            {"id":8,"timestamp":"2026-10-29T01:00:08Z","model":"other","tokens":{"input_tokens":1000,"prompt_per_second":34000}},
            {"id":7,"timestamp":"2026-10-29T01:00:07Z","model":"cpp","tokens":{"input_tokens":100}},
            {"id":6,"timestamp":"2026-10-29T01:00:06Z","model":"cpp","tokens":{"input_tokens":100}},
            {"id":5,"timestamp":"2026-10-29T01:00:05Z","model":"cpp","tokens":{"input_tokens":100}},
            {"id":4,"timestamp":"2026-10-29T01:00:04Z","model":"cpp","tokens":{"input_tokens":100}},
            {"id":3,"timestamp":"2026-10-29T01:00:03Z","model":"cpp","tokens":{"input_tokens":100}},
            {"id":2,"timestamp":"2026-10-29T01:00:02Z","model":"cpp","tokens":{"input_tokens":1000,"cache_tokens":9000,"prompt_per_second":900}},
            {"id":1,"timestamp":"2026-10-29T01:00:01Z","model":"cpp","tokens":{"input_tokens":100,"cache_tokens":9900,"prompt_per_second":300}}
        ]}"#).unwrap();
        assert_eq!(
            rows[6].prompt_for(Some(Backend::LlamaCpp)).unwrap().whole,
            10000
        );
        state.recent.merge(rows);
        for (processed, at) in [(100, now - Duration::from_millis(100)), (3500, now)] {
            let body = format!(
                r#"[{{"id":0,"id_task":1,"is_processing":true,"n_prompt_tokens":{processed},"n_prompt_tokens_processed":{processed}}}]"#
            );
            state.slots.begin_round();
            assert!(state.slots.apply("cpp", "cpp", body.as_bytes(), 0, 0));
            state.slots.finish_round(at);
        }
        assert_eq!(state.slots.prompt_tps(), Some(34_000.0));
        let rx = state.tx.share_with_thread();
        state.publish().expect("publish");
        let (_, detail) = rx.take().expect("sample");
        assert_eq!(detail.prompt_tps, Some(600.0));
    }

    #[test]
    fn chart_feed_separates_prefill_estimate_from_decode_and_headline() {
        use crate::tty::chart::TokenChart;
        let mut state = blank_state();
        state.running_up = true;
        state.ready = vec![ReadyModel {
            id: "cpp".into(),
            name: "cpp".into(),
            backend: Backend::LlamaCpp,
            probe: false,
        }];
        let now = Instant::now();
        state.counter.observe("cpp", 1, Some(1.0), now);
        state.recent.merge(activity::parse_activity(br#"{"data":[{"id":1,"timestamp":"2026-10-29T01:00:01Z","model":"cpp","tokens":{"input_tokens":100,"cache_tokens":900,"prompt_per_second":600}}]}"#).unwrap());
        let rx = state.tx.share_with_thread();
        let mut chart = TokenChart::new(2);
        let apply = |state: &mut State<llama_core::log::Stderr>,
                     task: i64,
                     decoded: Option<u64>| {
            let next = decoded.map_or(String::new(), |n| {
                format!(r#", "next_token":[{{"n_decoded":{n}}}]"#)
            });
            let body = format!(
                r#"[{{"id":0,"id_task":{task},"is_processing":true,"n_prompt_tokens_processed":100{next}}}]"#
            );
            state.slots.begin_round();
            assert!(state.slots.apply("cpp", "cpp", body.as_bytes(), 0, 0));
            state.slots.finish_round(Instant::now());
        };
        apply(&mut state, 1, Some(0));
        state.publish().unwrap();
        let (_, prefill) = rx.take().unwrap();
        assert_eq!(prefill.prompt_tps, Some(600.0));
        assert_eq!(prefill.chart_prompt_tps, Some(600.0));
        chart.ingest(now, None, prefill.chart_prompt_tps);
        apply(&mut state, 1, Some(1));
        state.publish().unwrap();
        let (_, decode) = rx.take().unwrap();
        assert_eq!(decode.prompt_tps, Some(600.0));
        assert_eq!(decode.chart_prompt_tps, Some(0.0));
        chart.ingest(now + Duration::from_secs(2), None, decode.chart_prompt_tps);
        chart.ingest(now + Duration::from_secs(4), None, decode.chart_prompt_tps);
        let buckets = chart.buckets();
        assert_eq!(buckets[buckets.len() - 2].prompt_tps, Some(0.0));
        assert_eq!(buckets[buckets.len() - 1].prompt_tps, Some(0.0));
        apply(&mut state, 2, Some(8)); // possibly stale previous-task output
        state.publish().unwrap();
        assert_eq!(rx.take().unwrap().1.chart_prompt_tps, None);
        apply(&mut state, 2, None); // missing next_token, not a zero
        state.publish().unwrap();
        assert_eq!(rx.take().unwrap().1.chart_prompt_tps, None);
        apply(&mut state, 2, Some(0));
        state.publish().unwrap();
        assert_eq!(rx.take().unwrap().1.chart_prompt_tps, Some(600.0));
        state.counter.observe("cpp", 1, Some(0.0), Instant::now());
        state.publish().unwrap();
        assert_eq!(rx.take().unwrap().1.chart_prompt_tps, Some(0.0));
    }

    #[test]
    fn chart_phase_mixed_slots_and_models_requires_every_active_phase() {
        let mut book = SlotBook::default();
        let now = Instant::now();
        book.begin_round();
        assert!(book.apply("a", "a", br#"[{"id":0,"id_task":1,"is_processing":true,"next_token":[{"n_decoded":0}]},{"id":1,"id_task":2,"is_processing":true,"next_token":[{"n_decoded":0}]}]"#, 0, 0));
        assert!(book.apply(
            "b",
            "b",
            br#"[{"id":0,"id_task":3,"is_processing":true,"next_token":[{"n_decoded":0}]}]"#,
            0,
            0
        ));
        book.finish_round(now);
        book.begin_round();
        assert!(book.apply("a", "a", br#"[{"id":0,"id_task":1,"is_processing":true,"next_token":[{"n_decoded":5}]},{"id":1,"id_task":2,"is_processing":true,"next_token":[{"n_decoded":0}]}]"#, 0, 0));
        assert!(book.apply(
            "b",
            "b",
            br#"[{"id":0,"id_task":3,"is_processing":true,"next_token":[{"n_decoded":9}]}]"#,
            0,
            0
        ));
        book.finish_round(now);
        assert_eq!(
            book.chart_phase("a", 2.0, now, Duration::from_secs(2)),
            Some(crate::slots::ChartPhase::Prefill)
        );
        assert_eq!(
            book.chart_phase("b", 1.0, now, Duration::from_secs(2)),
            Some(crate::slots::ChartPhase::Decode)
        );
        assert_eq!(
            book.chart_phase("a", 3.0, now, Duration::from_secs(2)),
            None
        );
        assert_eq!(
            book.chart_phase(
                "a",
                2.0,
                now + Duration::from_secs(3),
                Duration::from_secs(2)
            ),
            None
        );
    }

    #[test]
    fn chart_feed_mixed_models_does_not_hide_another_prefill() {
        let mut state = blank_state();
        state.running_up = true;
        for id in ["a", "b"] {
            state.ready.push(ReadyModel {
                id: id.into(),
                name: id.into(),
                backend: Backend::LlamaCpp,
                probe: false,
            });
            state.counter.observe(
                id,
                1,
                Some(if id == "a" { 2.0 } else { 1.0 }),
                Instant::now(),
            );
        }
        state.recent.merge(activity::parse_activity(br#"{"data":[
          {"id":2,"timestamp":"2026-10-29T01:00:02Z","model":"a","tokens":{"input_tokens":100,"prompt_per_second":400}},
          {"id":1,"timestamp":"2026-10-29T01:00:01Z","model":"b","tokens":{"input_tokens":100,"prompt_per_second":1000}}]}"#).unwrap());
        let rx = state.tx.share_with_thread();
        for decode in [0, 1] {
            state.slots.begin_round();
            let a = format!(
                r#"[{{"id":0,"id_task":1,"is_processing":true,"next_token":[{{"n_decoded":{decode}}}]}},{{"id":1,"id_task":2,"is_processing":true,"next_token":[{{"n_decoded":0}}]}}]"#
            );
            let b = format!(
                r#"[{{"id":0,"id_task":3,"is_processing":true,"next_token":[{{"n_decoded":{decode}}}]}}]"#
            );
            assert!(state.slots.apply("a", "a", a.as_bytes(), 0, 0));
            assert!(state.slots.apply("b", "b", b.as_bytes(), 0, 0));
            state.slots.finish_round(Instant::now());
        }
        state.publish().unwrap();
        let (_, detail) = rx.take().unwrap();
        assert_eq!(detail.prompt_tps, Some(1400.0));
        assert_eq!(detail.chart_prompt_tps, Some(400.0));
        state.slots.begin_round();
        assert!(state.slots.apply("a", "a", br#"[{"id":0,"id_task":1,"is_processing":true},{"id":1,"id_task":2,"is_processing":true,"next_token":[{"n_decoded":0}]}]"#, 0, 0));
        assert!(state.slots.apply(
            "b",
            "b",
            br#"[{"id":0,"id_task":3,"is_processing":true,"next_token":[{"n_decoded":2}]}]"#,
            0,
            0
        ));
        state.slots.finish_round(Instant::now());
        state.publish().unwrap();
        assert_eq!(rx.take().unwrap().1.chart_prompt_tps, None);
    }

    /// A llama.cpp model `cpp` in `state`, ready and read.
    fn lcpp_ready(state: &mut State<llama_core::log::Stderr>) {
        state.ready = vec![ReadyModel {
            id: "cpp".into(),
            name: "cpp".into(),
            backend: Backend::LlamaCpp,
            probe: false,
        }];
        state.running_up = true;
    }

    /// One llama.cpp `/metrics` read of `cpp` at `at`.
    fn lcpp_metrics(
        state: &mut State<llama_core::log::Stderr>,
        at: Instant,
        calls: u64,
        predicted: u64,
        processing: f64,
    ) {
        let sample = crate::metrics::parse_metrics_full(
            Backend::LlamaCpp,
            &format!(
                "llamacpp:n_decode_total {calls}\nllamacpp:tokens_predicted_total {predicted}\nllamacpp:requests_processing {processing}\nllamacpp:n_busy_slots_per_decode {processing}\n"
            ),
        )
        .0;
        state.observe_lcpp("cpp", &sample, at);
    }

    /// One llama.cpp `/slots` round of `cpp` at `at`: one decoding slot
    /// per entry of `decoded`, task 10 + slot.
    fn lcpp_slots(state: &mut State<llama_core::log::Stderr>, at: Instant, decoded: &[u64]) {
        let body: Vec<String> = decoded
            .iter()
            .enumerate()
            .map(|(id, n)| {
                format!(
                    r#"{{"id":{id},"id_task":{},"is_processing":true,"n_prompt_tokens_processed":100,"next_token":[{{"n_decoded":{n}}}]}}"#,
                    10 + id
                )
            })
            .collect();
        state.slots.begin_round();
        assert!(state.slots.apply(
            "cpp",
            "cpp",
            format!("[{}]", body.join(",")).as_bytes(),
            0,
            0
        ));
        state.slots.finish_round(at);
        let tokens = state.slots.tokens("cpp");
        state.lcpp_tokens.observe_slots("cpp", &tokens, at);
    }

    /// CCR-3 corrected, through the poller: N slots decoding share one
    /// `llama_decode` call per step. 50 steps in a second are 50 calls and
    /// 50 × N tokens; WRITE (the box counter's rate, which also feeds the
    /// timeline and the LCD) shows 50 × N, for N = 1, 2 and 4.
    #[test]
    fn llamacpp_write_counts_every_streams_tokens() {
        for n in [1usize, 2, 4] {
            let mut state = blank_state();
            lcpp_ready(&mut state);
            let t0 = Instant::now();
            let s = Duration::from_secs(1);
            let streams = |decoded: u64| vec![decoded; n];
            lcpp_metrics(&mut state, t0, 1000, 0, n as f64);
            lcpp_slots(&mut state, t0, &streams(10));
            let mut rate = GenRate::default();
            let base = state
                .counter
                .total_if_fresh(true, &["cpp"], t0)
                .expect("fresh");
            let _ = rate.observe(t0, base);
            lcpp_metrics(&mut state, t0 + s, 1050, 0, n as f64);
            lcpp_slots(&mut state, t0 + s, &streams(60));
            let total = state
                .counter
                .total_if_fresh(true, &["cpp"], t0 + s)
                .expect("fresh");
            assert_eq!(
                rate.observe(t0 + s, total),
                Some(50.0 * n as f64),
                "{n} streams"
            );
        }
    }

    /// A long single-stream decode with `/slots` every second and
    /// `/metrics` every 0.25 s: WRITE moves at every `/metrics` read with
    /// the decode calls, as before DER-451, and its one-second rate stays
    /// at the stream's 40 tok/s.
    #[test]
    fn llamacpp_single_stream_write_is_unchanged() {
        let mut state = blank_state();
        lcpp_ready(&mut state);
        let t0 = Instant::now();
        let q = Duration::from_millis(250);
        let mut rate = GenRate::default();
        lcpp_metrics(&mut state, t0, 0, 0, 1.0);
        lcpp_slots(&mut state, t0, &[1]);
        let _ = rate.observe(t0, state.counter.total());
        let mut last = state.counter.total();
        for i in 1u32..=40 {
            let at = t0 + q * i;
            let calls = u64::from(i) * 10;
            lcpp_metrics(&mut state, at, calls, 0, 1.0);
            if i % 4 == 0 {
                lcpp_slots(&mut state, at, &[1 + calls]);
            }
            let total = state.counter.total();
            assert_eq!(total - last, 10, "read {i} moves by its 10 calls");
            last = total;
            if i >= 4 {
                assert_eq!(rate.observe(at, total), Some(40.0), "read {i}");
            } else {
                let _ = rate.observe(at, total);
            }
        }
    }

    /// Several llama.cpp streams with `/slots` gone: WRITE is unknown, not
    /// ÷N; the model's gauge stays, so `/slots` is still tried.
    #[test]
    fn llamacpp_write_without_slots_is_unknown_under_concurrency() {
        let mut state = blank_state();
        lcpp_ready(&mut state);
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        for i in 0u32..=4 {
            lcpp_metrics(&mut state, t0 + s * i, 1000 + u64::from(i) * 50, 0, 2.0);
        }
        let at = t0 + s * 4;
        assert_eq!(state.counter.total_if_fresh(true, &["cpp"], at), None);
        assert_eq!(state.counter.requests_processing("cpp", at), Some(2.0));
    }

    /// One vLLM `/metrics` read of model `v` at `at`, as the scrape folds
    /// it in: `running` requests, the TTFT and e2e counts, the generated
    /// counter, and one finished prefill of 4,000 tokens in 2 s (2,000 tok/s).
    fn vllm_read(
        state: &mut State<llama_core::log::Stderr>,
        at: Instant,
        running: f64,
        (ttft, e2e): (f64, f64),
        generated: u64,
    ) {
        use crate::metrics::{Hist, MetricsSample};
        use crate::speeds::{Phase, SpeedTotals};
        let count = |n: f64| Some(Hist { sum: n, count: n });
        let sample = MetricsSample {
            requests_processing: Some(running),
            queued: Some(0.0),
            ttft: count(ttft),
            e2e: count(e2e),
            n_decode_total: Some(generated),
            speeds: Some(SpeedTotals {
                prefill: Some(Phase {
                    seconds: Hist {
                        sum: 2.0,
                        count: 1.0,
                    },
                    tokens: Hist {
                        sum: 4_000.0,
                        count: 1.0,
                    },
                }),
                decode: None,
            }),
            ..MetricsSample::default()
        };
        state.engines.observe("v", &sample);
        state.vllm_phase.observe("v", &sample, at);
        state.counter.observe("v", generated, Some(running), at);
    }

    fn vllm_model() -> ReadyModel {
        ReadyModel {
            id: "v".into(),
            name: "v".into(),
            backend: Backend::Vllm,
            probe: false,
        }
    }

    /// Request marks (DER-451): a reload forgets the model's, and past the
    /// model cap the least recently used model makes room.
    #[test]
    fn request_marks_forget_on_reload_and_evict_the_least_recent() {
        use crate::capture::CaptureText;
        let text = |hash: u64| CaptureText {
            input: "new".into(),
            messages: vec![MessageMark {
                hash,
                in_role: true,
            }],
            input_parts: vec![(0, 0)],
            ..CaptureText::default()
        };
        let mut marks = RequestMarks::default();
        assert_eq!(marks.diff_and_keep("a", &mut text(1)), None, "no history");
        assert_eq!(marks.diff_and_keep("a", &mut text(1)), Some(0), "a retry");
        marks.forget("a");
        assert_eq!(marks.diff_and_keep("a", &mut text(1)), None, "forgotten");
        for index in 0..REQUEST_MARK_MODELS {
            marks.diff_and_keep(&format!("m{index}"), &mut text(1));
        }
        // "a" is the oldest; touching it makes "m0" the oldest instead.
        marks.diff_and_keep("a", &mut text(1));
        marks.diff_and_keep("new", &mut text(1));
        assert_eq!(marks.models.len(), REQUEST_MARK_MODELS);
        assert!(marks.models.contains_key("a"));
        assert!(marks.models.contains_key("new"), "a new model is kept");
        assert!(!marks.models.contains_key("m0"), "the least recent went");
    }

    /// vLLM's READ chart follows its requests' phase (DER-451): the held
    /// prefill speed while one is prefilling, 0 while every one decodes,
    /// unknown when the counts do not add up or the phase read is stale.
    /// The big number keeps the held speed through all of it.
    #[test]
    fn vllm_chart_follows_the_request_phase_not_the_held_headline() {
        let mut state = blank_state();
        state.running_up = true;
        state.ready = vec![
            ReadyModel {
                id: "cpp".into(),
                name: "cpp".into(),
                backend: Backend::LlamaCpp,
                probe: false,
            },
            vllm_model(),
        ];
        state.slots.idle();
        let rx = state.tx.share_with_thread();
        let publish = |state: &mut State<llama_core::log::Stderr>| {
            state.publish().unwrap();
            let (_, detail) = rx.take().unwrap();
            (detail.prompt_tps, detail.chart_prompt_tps)
        };
        vllm_read(&mut state, Instant::now(), 0.0, (1.0, 1.0), 50);
        assert_eq!(publish(&mut state), (Some(0.0), Some(0.0)), "idle");
        vllm_read(&mut state, Instant::now(), 1.0, (1.0, 1.0), 50);
        assert_eq!(
            publish(&mut state),
            (Some(2_000.0), Some(2_000.0)),
            "prefilling"
        );
        vllm_read(&mut state, Instant::now(), 1.0, (2.0, 1.0), 51);
        assert_eq!(
            publish(&mut state),
            (Some(2_000.0), Some(0.0)),
            "decoding: the headline holds, the chart is 0"
        );
        vllm_read(&mut state, Instant::now(), 1.0, (2.0, 1.0), 90);
        assert_eq!(publish(&mut state), (Some(2_000.0), Some(0.0)));
        vllm_read(&mut state, Instant::now(), 2.0, (2.0, 1.0), 120);
        assert_eq!(
            publish(&mut state),
            (Some(2_000.0), Some(2_000.0)),
            "a second request prefills beside the decode"
        );
        vllm_read(&mut state, Instant::now(), 1.0, (2.0, 3.0), 150);
        assert_eq!(
            publish(&mut state),
            (Some(2_000.0), None),
            "more finished than started: unknown, not a fabricated phase"
        );
        vllm_read(
            &mut state,
            Instant::now() - Duration::from_secs(5),
            1.0,
            (3.0, 2.0),
            160,
        );
        state.counter.observe("v", 160, Some(1.0), Instant::now());
        assert_eq!(publish(&mut state), (Some(2_000.0), None), "stale phase");
        vllm_read(&mut state, Instant::now(), 0.0, (3.0, 3.0), 170);
        assert_eq!(publish(&mut state), (Some(0.0), Some(0.0)), "done");
    }

    /// A busy llama.cpp model beside vLLM (DER-451): each model's chart
    /// input is gated by its own phase and they add.
    #[test]
    fn vllm_chart_adds_to_a_busy_llama_cpp_chart() {
        let mut state = blank_state();
        state.running_up = true;
        state.ready = vec![
            ReadyModel {
                id: "cpp".into(),
                name: "cpp".into(),
                backend: Backend::LlamaCpp,
                probe: false,
            },
            vllm_model(),
        ];
        state.counter.observe("cpp", 1, Some(1.0), Instant::now());
        state.recent.merge(activity::parse_activity(br#"{"data":[{"id":1,"timestamp":"2026-10-29T01:00:01Z","model":"cpp","tokens":{"input_tokens":100,"prompt_per_second":600}}]}"#).unwrap());
        state.slots.begin_round();
        assert!(state.slots.apply(
            "cpp",
            "cpp",
            br#"[{"id":0,"id_task":1,"is_processing":true,"next_token":[{"n_decoded":0}]}]"#,
            0,
            0
        ));
        state.slots.finish_round(Instant::now());
        let rx = state.tx.share_with_thread();
        vllm_read(&mut state, Instant::now(), 0.0, (0.0, 0.0), 0);
        vllm_read(&mut state, Instant::now(), 1.0, (1.0, 0.0), 1);
        state.publish().unwrap();
        let (_, detail) = rx.take().unwrap();
        assert_eq!(detail.prompt_tps, Some(2_600.0));
        assert_eq!(detail.chart_prompt_tps, Some(600.0), "vLLM decodes");
        vllm_read(&mut state, Instant::now(), 2.0, (1.0, 0.0), 9);
        state.publish().unwrap();
        let (_, detail) = rx.take().unwrap();
        assert_eq!(detail.chart_prompt_tps, Some(2_600.0), "both prefill");
    }

    /// One finished SGLang request, alone, holds its prefill on the big
    /// number while llama.cpp sits idle beside it (#188). Replacing the
    /// SGLang arm of [`State::held_burst_prompt`] with `None` loses the 200.
    #[test]
    fn one_finished_sglang_request_holds_its_rate_beside_an_idle_slot() {
        let mut state = blank_state();
        lone_sglang_finish(&mut state.sglang);
        assert_eq!(
            state.sglang.prompt_rate("flash"),
            Some(200.0),
            "one lone finish is (1000 - 600) / 2"
        );
        state.slots.idle();
        assert_eq!(
            state.slots.prompt_tps(),
            Some(0.0),
            "an idle llama.cpp slot stored 0"
        );
        state.ready = vec![
            ReadyModel {
                id: "cpp".to_owned(),
                name: "llama".to_owned(),
                backend: Backend::LlamaCpp,
                probe: false,
            },
            ReadyModel {
                id: "flash".to_owned(),
                name: "flash".to_owned(),
                backend: Backend::SgLang,
                probe: false,
            },
        ];
        state.running_up = true;
        let now = Instant::now();
        let _ = state.prompt_rate.observe(now - Duration::from_secs(1), 0);
        state
            .prompt_counter
            .observe("flash", 0, None, now - Duration::from_secs(1));
        state
            .prompt_counter
            .observe("flash", 20_000, Some(1.0), now);
        assert_eq!(state.prompt_counter.total(), 20_000, "the wall-clock spike");
        state.counter.observe("flash", 1, Some(1.0), now);

        let rx = state.tx.share_with_thread();
        state.publish().expect("publish");
        let (_view, detail) = rx.take().expect("sample");
        assert_eq!(
            detail.prompt_tps,
            Some(200.0),
            "the big number is the lone SGLang prefill, not the idle 0 or the spike"
        );
        // DER-451: SGLang's phase is never known, so its chart is unknown
        // while it is busy, not the held speed through its decode.
        assert_eq!(detail.chart_prompt_tps, None);
    }

    fn blank_state() -> State<llama_core::log::Stderr> {
        State {
            limits: Limits {
                url: "http://127.0.0.1:9".to_owned(),
                running_interval: Duration::from_secs(1),
                running_timeout: Duration::from_secs(1),
                metrics_interval: Duration::from_secs(1),
                metrics_timeout: Duration::from_secs(1),
                slots_interval: Duration::from_secs(1),
                slots_timeout: Duration::from_secs(1),
                slots_cap: 1024,
                activity_interval: Duration::from_secs(1),
                activity_timeout: Duration::from_secs(1),
                input_tail: 0,
                output_tail: 0,
                show_text: false,
                prompt_view: PromptView::default(),
                max_name_chars: 48,
                aliases: HashMap::new(),
                backends: HashMap::new(),
                setup: Rules::default(),
                specs_dir: None,
                jobs_dir: None,
                llama_enabled: true,
            },
            agent: llamaswap::new_agent(),
            log: llama_core::log::Stderr,
            tx: SampleRx::new(),
            counter: DecodedCounter::default(),
            gen_rate: GenRate::default(),
            slots: SlotBook::default(),
            lcpp_tokens: LcppTokens::default(),
            recent: Recent::default(),
            ai: AiState::Loaded,
            models: Vec::new(),
            model_ids: Vec::new(),
            model_kv: Vec::new(),
            model_tq_kv: Vec::new(),
            model_setup: Vec::new(),
            specs: Specs::new(None),
            jobs: Jobs::new(None),
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
            sglang: SglangBook::default(),
            vllm_phase: VllmPhaseBook::default(),
            series: SeriesBook::default(),
            detected: HashMap::new(),
            detect_logged: HashSet::new(),
            probe_snapshot: HashMap::new(),
            tqllm: TqllmBook::default(),
            schema_logged: HashSet::new(),
            health_logged: HashSet::new(),
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
            request_marks: RequestMarks::default(),
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
            row_engines: HashMap::new(),
            events: None,
            events_view: EventsView::default(),
            events_failed: false,
            claims_failed: false,
            gpu_failed: false,
            inflight_engines: false,
            live: LiveBook::default(),
            loads_missing: HashSet::new(),
            loads_failed: false,
        }
    }

    /// The SGLang book's one-request window: 1,000 prompt tokens, 600
    /// cached, two seconds to the first token. The honest rate is 200.
    fn lone_sglang_finish(book: &mut SglangBook) {
        let origin = Instant::now();
        let at = |ms: u64| origin + Duration::from_millis(ms);
        let wall = |ms: u64| SystemTime::UNIX_EPOCH + Duration::from_millis(ms);
        let see = |book: &mut SglangBook, ms: u64, totals: Totals| {
            book.observe("flash", at(ms), wall(ms), Some(totals));
        };
        let zeros = Totals {
            requests: 0,
            batch: 0,
            prompt: 0,
            cached: 0,
            generated: 0,
            e2e_sum: 0.0,
            e2e_count: 0.0,
            ttft_sum: 0.0,
            ttft_count: 0.0,
            queue_sum: 0.0,
            queue_count: 0.0,
            running: 0.0,
        };
        let mut ms = 0;
        loop {
            see(book, ms, zeros);
            if ms >= 8_000 {
                break;
            }
            let next = ms.saturating_add(2_000);
            ms = if next > 8_000 { 8_000 } else { next };
        }
        book.activity(at(100), &[]);
        see(
            book,
            10_000,
            Totals {
                ttft_sum: 2.0,
                ttft_count: 1.0,
                running: 1.0,
                ..zeros
            },
        );
        let after = Totals {
            requests: 1,
            batch: 0,
            prompt: 1_000,
            cached: 600,
            generated: 401,
            e2e_sum: 10.0,
            e2e_count: 1.0,
            ttft_sum: 2.0,
            ttft_count: 1.0,
            queue_sum: 0.0,
            queue_count: 0.0,
            running: 0.0,
        };
        see(book, 12_000, after);
        let stamp = format!(
            "{}Z",
            crate::localtime::format_local(12, crate::localtime::Zone::utc())
        );
        book.activity(
            at(12_500),
            &[Claim {
                seq: 1,
                model: "flash".to_owned(),
                input: 1_000,
                output: Some(401),
                duration_ms: Some(10_000),
                prompt_known: false,
                gen_known: false,
                timestamp: Some(stamp),
            }],
        );
        ms = 14_000;
        loop {
            see(book, ms, after);
            if ms > 18_000 {
                break;
            }
            ms = ms.saturating_add(2_000);
        }
    }
}
