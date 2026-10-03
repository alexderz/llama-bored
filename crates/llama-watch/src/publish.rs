//! Atomic snapshot publish.
//!
//! The directory is a parameter. Production passes
//! [`llama_core::wire::SNAPSHOT_DIR`]. The directory fd is opened once with
//! `O_DIRECTORY|O_NOFOLLOW`. Each publish validates, skips a body longer than
//! the cap, writes `snapshot.json.tmp` without following a symlink, and
//! `renameat`s it onto `snapshot.json`. The temp name is unlinked first, then
//! created with `O_EXCL` and `fchmod`ed to `0640`, so a planted symlink, FIFO,
//! or hard link cannot redirect or truncate the write. There is no fsync: the
//! rename is the publication.

use std::io::Error as IoError;
use std::os::fd::OwnedFd;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use llama_core::backend::EngineStats;
use llama_core::detail::{self, MAX_FULL_NAME_CHARS};
use llama_core::log::{self, Priority, Sink};
use llama_core::names::{sanitize, sanitize_wire};
use llama_core::sample::{AiState, LlamaView, Snapshot};
use llama_core::wire::{
    self, Ai, AiWire, EngineWire, FanWire, Host, ModelState, ModelWire, SlotCtxWire,
    SlotResetsWire, Sources, Tokens, WireError, WireSnapshot,
};

use crate::resets::ResetCounts;
use thiserror::Error;

const TMP_NAME: &str = "snapshot.json.tmp";
const FINAL_NAME: &str = "snapshot.json";

/// Why a snapshot was not published.
#[derive(Debug, Error)]
pub enum PublishError {
    /// The directory could not be opened, or the run id could not be drawn.
    #[error("could not open the snapshot directory")]
    Directory(#[source] IoError),
    /// [`wire::validate`] or encoding failed. The previous file is unchanged.
    #[error("snapshot failed validation")]
    Invalid(#[source] WireError),
    /// The encoded body is longer than the cap. The previous file is unchanged.
    #[error("snapshot exceeds the maximum length")]
    TooLarge,
    /// The temp file could not be written or renamed. The previous file is unchanged.
    ///
    /// T22 must log [`PublishError::Write`] once per transition: one line when
    /// the failure starts, and one when a later publish succeeds. Do not log
    /// it on every tick.
    #[error("could not write the snapshot")]
    Write(#[source] IoError),
}

/// Watcher numbers that are not on [`Snapshot`] or [`LlamaView`] but go on
/// the wire for llama-metrics (#11). `Default` is "nothing known".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Extras {
    /// GPU power, watts.
    pub gpu_w: Option<f64>,
    /// GPU enforced power limit, watts.
    pub gpu_limit_w: Option<f64>,
    /// CPU socket power, watts.
    pub cpu_w: Option<f64>,
    /// VRAM used and total, bytes.
    pub vram_used: Option<u64>,
    pub vram_total: Option<u64>,
    /// System memory used and total, bytes.
    pub mem_used: Option<u64>,
    pub mem_total: Option<u64>,
    /// llama.cpp slots per model: `(display name, busy, total)`. The name is
    /// the [`llama_core::sample::ModelInfo::name`] the slots belong to.
    pub slots: Vec<(String, usize, usize)>,
    /// Per-model prompt token counters (#10): `(display name, prompt,
    /// cached)`, the name as in `slots`.
    pub prompt_cache: Vec<(String, u64, Option<u64>)>,
    /// llama.cpp slot context (#10), one entry per slot.
    pub slot_ctx: Vec<SlotCtx>,
    /// Configured fans: `(channel, label, rpm, pwm 0..=255)`.
    pub fans: Vec<(u32, String, Option<u32>, Option<u8>)>,
    /// The tty health line, as wire sources. `None` while starting.
    pub sources: Option<Sources>,
}

/// One llama.cpp slot's context numbers for the wire (#10).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlotCtx {
    /// Display name of the model that owns the slot.
    pub model: String,
    /// llama-server slot id.
    pub slot: i64,
    /// Context tokens held, `None` when `/slots` gave no prompt count.
    pub used: Option<u64>,
    /// Context drops seen this run, by reason (#9).
    pub resets: ResetCounts,
}

/// Publishes validated snapshots into one directory.
pub struct Publisher<L> {
    dir: OwnedFd,
    run_id: u64,
    seq: u64,
    max_bytes: usize,
    oversize_logged: bool,
    invalid_logged: bool,
    log: L,
}

impl<L: Sink> Publisher<L> {
    /// Open `dir` once. Fails if `dir` is a symlink or not a directory.
    pub fn open(dir: &Path, log: L) -> Result<Self, PublishError> {
        let opened = rustix::fs::open(
            dir,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )
        .map_err(|err| PublishError::Directory(IoError::from(err)))?;
        let run_id = random_u64().map_err(PublishError::Directory)?;
        Ok(Self {
            dir: opened,
            run_id,
            seq: 0,
            max_bytes: wire::MAX_BYTES,
            oversize_logged: false,
            invalid_logged: false,
            log,
        })
    }

    /// Reject encoded snapshots longer than `max_bytes`.
    ///
    /// The default is [`wire::MAX_BYTES`]. A valid v1 snapshot stays under that
    /// cap; tests lower it to prove the branch.
    pub fn set_max_bytes(&mut self, max_bytes: usize) {
        self.max_bytes = max_bytes;
    }

    /// Validate, then replace `snapshot.json`. A refusal leaves the previous file.
    pub fn publish(&mut self, snapshot: &Snapshot, llama: &LlamaView) -> Result<(), PublishError> {
        self.publish_with(snapshot, llama, &Extras::default())
    }

    /// [`Self::publish`] with the llama-metrics extras (#11).
    pub fn publish_with(
        &mut self,
        snapshot: &Snapshot,
        llama: &LlamaView,
        extras: &Extras,
    ) -> Result<(), PublishError> {
        let seq = self.seq.saturating_add(1);
        let wire_snapshot = build(snapshot, llama, extras, self.run_id, seq);
        if let Err(err) = wire::validate(&wire_snapshot) {
            self.log_invalid(&err);
            return Err(PublishError::Invalid(err));
        }
        let bytes = match wire::to_json(&wire_snapshot) {
            Ok(bytes) => bytes,
            Err(err) => {
                self.log_invalid(&err);
                return Err(PublishError::Invalid(err));
            }
        };
        if bytes.len() > self.max_bytes {
            self.log_oversize();
            return Err(PublishError::TooLarge);
        }
        self.write_replace(&bytes)?;
        self.seq = seq;
        self.invalid_logged = false;
        self.oversize_logged = false;
        Ok(())
    }

    fn write_replace(&self, bytes: &[u8]) -> Result<(), PublishError> {
        match rustix::fs::unlinkat(&self.dir, TMP_NAME, rustix::fs::AtFlags::empty()) {
            Ok(()) => {}
            Err(rustix::io::Errno::NOENT) => {}
            Err(err) => return Err(PublishError::Write(IoError::from(err))),
        }
        let file = rustix::fs::openat(
            &self.dir,
            TMP_NAME,
            rustix::fs::OFlags::WRONLY
                | rustix::fs::OFlags::CREATE
                | rustix::fs::OFlags::EXCL
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            file_mode(),
        )
        .map_err(|err| PublishError::Write(IoError::from(err)))?;
        rustix::fs::fchmod(&file, file_mode())
            .map_err(|err| PublishError::Write(IoError::from(err)))?;
        write_full(&file, bytes)?;
        drop(file);
        rustix::fs::renameat(&self.dir, TMP_NAME, &self.dir, FINAL_NAME)
            .map_err(|err| PublishError::Write(IoError::from(err)))?;
        Ok(())
    }

    fn log_oversize(&mut self) {
        if self.oversize_logged {
            return;
        }
        self.oversize_logged = true;
        log::emit(
            &mut self.log,
            Priority::Err,
            "snapshot exceeds maximum length; not published",
        );
    }

    fn log_invalid(&mut self, err: &WireError) {
        if self.invalid_logged {
            return;
        }
        self.invalid_logged = true;
        log::emit(
            &mut self.log,
            Priority::Err,
            &format!("snapshot rejected: {err}"),
        );
    }
}

/// Assemble the wire snapshot. Every extra that is out of its wire range is
/// left out rather than failing [`wire::validate`], so one odd reading never
/// stops the publish.
pub fn build(
    snapshot: &Snapshot,
    llama: &LlamaView,
    extras: &Extras,
    run_id: u64,
    seq: u64,
) -> WireSnapshot {
    WireSnapshot {
        schema: wire::SCHEMA,
        run_id,
        seq,
        t_mono_ns: mono_ns(),
        t_wall_ms: wall_ms(snapshot.t_wall),
        host: Host {
            load_pct: clamp_pct(snapshot.load),
            activity_pct: clamp_activity(snapshot.activity),
            cpu_pct: clamp_pct(snapshot.cpu_pct),
            cpu_topk_pct: clamp_pct(snapshot.cpu_topk_pct),
            gpu_pct: clamp_pct(snapshot.gpu_pct),
            mem_pct: clamp_pct(snapshot.mem_pct),
            coolant_c: snapshot.coolant_c,
            cpu_c: snapshot.cpu_c,
            gpu_c: snapshot.gpu_c,
            gpu_w: watts(extras.gpu_w),
            gpu_limit_w: watts(extras.gpu_limit_w),
            cpu_w: watts(extras.cpu_w),
            vram_used_bytes: mem_bytes(extras.vram_used),
            vram_total_bytes: mem_bytes(extras.vram_total),
            mem_used_bytes: mem_bytes(extras.mem_used),
            mem_total_bytes: mem_bytes(extras.mem_total),
        },
        ai: ai_of(snapshot, extras),
        tokens: Tokens {
            decoded_total: llama.decoded_total,
            prompt_total: llama.prompt_total,
        },
        fans: fans_of(&extras.fans),
        sources: extras.sources.map(sources_of),
    }
}

fn watts(value: Option<f64>) -> Option<f32> {
    value
        .filter(|w| w.is_finite() && (0.0..=f64::from(wire::MAX_WATTS)).contains(w))
        .map(|w| w as f32)
}

fn mem_bytes(value: Option<u64>) -> Option<u64> {
    value.filter(|b| *b <= wire::MAX_MEM_BYTES)
}

/// Fans in config order; a channel or label the wire refuses is dropped.
fn fans_of(fans: &[(u32, String, Option<u32>, Option<u8>)]) -> Vec<FanWire> {
    let mut out: Vec<FanWire> = Vec::new();
    for (channel, label, rpm, pwm) in fans {
        let Some(channel) = u8::try_from(*channel)
            .ok()
            .filter(|c| (1..=wire::MAX_FAN_CHANNEL).contains(c))
        else {
            continue;
        };
        let label = sanitize(label, wire::MAX_FAN_LABEL_CHARS);
        if label.is_empty()
            || !label.is_ascii()
            || out.len() >= wire::MAX_FANS
            || out.iter().any(|fan| fan.channel == channel)
        {
            continue;
        }
        out.push(FanWire {
            channel,
            label,
            rpm: rpm.filter(|r| *r <= wire::MAX_FAN_RPM),
            pwm: pwm.map(|p| f32::from(p) / 255.0),
        });
    }
    out
}

/// Latencies over [`wire::MAX_LATENCY_S`] are dropped; the up flag stays.
fn sources_of(mut sources: Sources) -> Sources {
    for source in [
        &mut sources.llama_swap,
        &mut sources.running,
        &mut sources.slots,
        &mut sources.metrics,
        &mut sources.activity,
        &mut sources.gpu,
        &mut sources.hwmon,
        &mut sources.proc,
    ]
    .into_iter()
    .flatten()
    {
        source.latency_s = source
            .latency_s
            .filter(|s| s.is_finite() && (0.0..=wire::MAX_LATENCY_S).contains(s));
    }
    sources
}

fn ai_of(snapshot: &Snapshot, extras: &Extras) -> Ai {
    let slots = &extras.slots;
    let mut slot_rows_left = wire::MAX_SLOT_CTX;
    match snapshot.ai {
        AiState::Loaded => Ai {
            state: AiWire::Loaded,
            models: snapshot
                .models
                .iter()
                .take(wire::MAX_MODELS)
                .map(|model| {
                    let slot_counts = slots
                        .iter()
                        .find(|(owner, _, total)| *owner == model.name && *total > 0)
                        .and_then(|(_, busy, total)| {
                            let total = u16::try_from(*total)
                                .ok()
                                .filter(|t| *t <= wire::MAX_SLOTS)?;
                            let busy = u16::try_from(*busy).ok().filter(|b| *b <= total)?;
                            Some((busy, total))
                        });
                    let name = sanitize_wire(&model.name);
                    // Gauges only for a backend without `/slots`; llama.cpp
                    // keeps its slot view on the tty.
                    let gauges = model.backend.filter(|info| !info.kind.has_slots());
                    let (prompt_tokens, prompt_cached_tokens) = extras
                        .prompt_cache
                        .iter()
                        .find(|(owner, _, _)| *owner == model.name)
                        .map_or((None, None), |(_, prompt, cached)| {
                            (Some(*prompt), cached.map(|cached| cached.min(*prompt)))
                        });
                    let slot_ctx = slot_ctx_of(&extras.slot_ctx, &model.name, &mut slot_rows_left);
                    ModelWire {
                        full_name: wire_full_name(model.full_name.as_deref(), &name),
                        detail: model.detail.clone().filter(detail::is_valid),
                        name,
                        state: model_state(&model.state),
                        backend: model.backend.map(|info| info.kind),
                        running: gauges.and_then(|info| info.running).map(cap_reqs),
                        queued: gauges.and_then(|info| info.queued).map(cap_reqs),
                        kv_fill: gauges.and_then(|info| info.kv_permille).and_then(ratio),
                        cache_hit: gauges.and_then(|info| info.hit_permille).and_then(ratio),
                        slots_busy: slot_counts.map(|(busy, _)| busy),
                        slots_total: slot_counts.map(|(_, total)| total),
                        prompt_tokens,
                        prompt_cached_tokens,
                        slot_ctx,
                        engine: gauges.and_then(|info| engine_of(&info.engine)),
                    }
                })
                .collect(),
        },
        AiState::Idle => Ai {
            state: AiWire::Idle,
            models: Vec::new(),
        },
        AiState::Down | AiState::NoData => Ai {
            state: AiWire::Down,
            models: Vec::new(),
        },
    }
}

/// `model`'s slots, lowest id first, at most `left` of them; `left` goes
/// down by what was taken. A slot id the wire refuses, a repeat, or a slot
/// with no context count is skipped.
fn slot_ctx_of(rows: &[SlotCtx], model: &str, left: &mut usize) -> Vec<SlotCtxWire> {
    let mut out: Vec<SlotCtxWire> = rows
        .iter()
        .filter(|row| row.model == model)
        .filter_map(|row| {
            let slot = u16::try_from(row.slot)
                .ok()
                .filter(|slot| *slot < wire::MAX_SLOTS)?;
            Some(SlotCtxWire {
                slot,
                used: row.used?.min(wire::MAX_CTX_TOKENS),
                resets: SlotResetsWire {
                    compacted: row.resets.compacted,
                    new: row.resets.new,
                    evicted: row.resets.evicted,
                    unknown: row.resets.unknown,
                },
            })
        })
        .collect();
    out.sort_by_key(|row| row.slot);
    out.dedup_by_key(|row| row.slot);
    out.truncate(*left);
    *left -= out.len();
    out
}

/// The wire form of a model's engine numbers (#31); `None` when it has
/// none. Values outside the wire's ranges are left out, and accepted spec
/// tokens never exceed drafted ones.
fn engine_of(stats: &EngineStats) -> Option<EngineWire> {
    let seconds = |us: Option<u32>| {
        us.map(|us| (f64::from(us) / 1e6) as f32)
            .filter(|s| (0.0..=wire::MAX_ENGINE_LATENCY_S as f32).contains(s))
    };
    let counts = stats.spec_counts;
    let engine = EngineWire {
        spec_accept: stats.spec_permille.and_then(ratio),
        spec_len: stats
            .spec_len_centi
            .map(|centi| f32::from(centi) / 100.0)
            .filter(|len| (1.0..=wire::MAX_SPEC_LEN as f32).contains(len)),
        spec_drafts: counts.map(|c| c.drafts),
        spec_draft_tokens: counts.map(|c| c.draft_tokens),
        spec_accepted_tokens: counts.map(|c| c.accepted.min(c.draft_tokens)),
        preemptions: stats.preemptions,
        sleeping: stats.sleeping,
        ttft_s: seconds(stats.ttft_us),
        itl_s: seconds(stats.itl_us),
        e2e_s: seconds(stats.e2e_us),
        prefill_tps: tps(stats.prefill_tps_tenths),
        decode_tps: tps(stats.decode_tps_tenths),
    };
    (engine != EngineWire::default()).then_some(engine)
}

/// Tenths of a token per second as the wire's tok/s, inside
/// 0..=[`wire::MAX_ENGINE_TPS`] (#35).
fn tps(tenths: Option<u32>) -> Option<f32> {
    tenths
        .map(|tenths| (f64::from(tenths) / 10.0) as f32)
        .filter(|tps| (0.0..=wire::MAX_ENGINE_TPS as f32).contains(tps))
}

fn ratio(permille: u16) -> Option<f32> {
    (permille <= 1000).then(|| f32::from(permille) / 1000.0)
}

fn cap_reqs(n: u16) -> u16 {
    n.min(wire::MAX_REQS)
}

/// The full name when it adds something to `name` and is canonical.
fn wire_full_name(full: Option<&str>, name: &str) -> Option<String> {
    let full = sanitize(full?, MAX_FULL_NAME_CHARS);
    (!full.is_empty() && full != name).then_some(full)
}

fn file_mode() -> rustix::fs::Mode {
    rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR | rustix::fs::Mode::RGRP
}

/// Finite host percentages enter the wire inside 0..=100. Anything else is absent.
fn clamp_pct(value: Option<f32>) -> Option<f32> {
    clamp_to(value, 100.0)
}

/// Activity keeps its redline: 0..=[`wire::ACTIVITY_MAX_PCT`].
fn clamp_activity(value: Option<f32>) -> Option<f32> {
    clamp_to(value, wire::ACTIVITY_MAX_PCT)
}

fn clamp_to(value: Option<f32>, high: f32) -> Option<f32> {
    let value = value?;
    if !value.is_finite() {
        return None;
    }
    Some(value.clamp(0.0, high))
}

fn model_state(state: &str) -> ModelState {
    match state {
        "ready" => ModelState::Ready,
        "starting" => ModelState::Starting,
        "stopping" => ModelState::Stopping,
        _ => ModelState::Other,
    }
}

fn mono_ns() -> u64 {
    let timespec = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    let sec = u64::try_from(timespec.tv_sec).unwrap_or(0);
    let nsec = u64::try_from(timespec.tv_nsec).unwrap_or(0);
    sec.saturating_mul(1_000_000_000).saturating_add(nsec)
}

fn wall_ms(wall: SystemTime) -> u64 {
    match wall.duration_since(UNIX_EPOCH) {
        Ok(duration) => u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
        Err(_) => 0,
    }
}

fn random_u64() -> Result<u64, IoError> {
    let mut buf = [0u8; 8];
    let mut filled = 0;
    while filled < buf.len() {
        let n = rustix::rand::getrandom(&mut buf[filled..], rustix::rand::GetRandomFlags::empty())
            .map_err(IoError::from)?;
        if n == 0 {
            return Err(IoError::other("getrandom returned no bytes"));
        }
        filled += n;
    }
    Ok(u64::from_ne_bytes(buf))
}

fn write_full(fd: &OwnedFd, mut bytes: &[u8]) -> Result<(), PublishError> {
    while !bytes.is_empty() {
        match rustix::io::write(fd, bytes) {
            Ok(0) => {
                return Err(PublishError::Write(IoError::new(
                    std::io::ErrorKind::WriteZero,
                    "snapshot write made no progress",
                )));
            }
            Ok(n) => bytes = &bytes[n..],
            Err(rustix::io::Errno::INTR) => {}
            Err(err) => return Err(PublishError::Write(IoError::from(err))),
        }
    }
    Ok(())
}
