//! Read the watcher snapshot. The LCD coolant is not taken from that file.
//!
//! `llama_core::wire::from_bytes` is crate-private. The landed writer entry
//! is [`llama_core::wire::parse_validated`], which parses and then
//! [`llama_core::wire::validate`] (canonical names go through
//! [`llama_core::names`]).

use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

use llama_core::log::{self, Priority, Sink};
use llama_core::sample::{AiState, ModelInfo, Snapshot, TokenReading};
use llama_core::wire::{self, AiWire, ModelState, WireSnapshot};

use crate::device::{self, read_coolant_c};
use crate::service::Sampler;

const MISSING: &str = "snapshot missing";
const NOT_REGULAR: &str = "snapshot not a regular file";
const TOO_LARGE: &str = "snapshot too large";
const INVALID: &str = "snapshot invalid";
const SEQ_REGRESSION: &str = "snapshot seq regression";
const FROM_THE_FUTURE: &str = "snapshot from the future";
const STALE: &str = "snapshot stale";
const FRESH: &str = "snapshot fresh";

/// `t_mono_ns` may sit this far ahead of the writer's clock and still be valid.
const FUTURE_SLACK: Duration = Duration::from_millis(50);

/// `CLOCK_MONOTONIC` in nanoseconds. Production is [`HostMono`].
pub trait MonoNow {
    /// Nanoseconds since an arbitrary monotonic epoch shared with the watcher.
    fn now_ns(&self) -> u64;
}

/// Host `CLOCK_MONOTONIC`.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostMono;

impl MonoNow for HostMono {
    fn now_ns(&self) -> u64 {
        let ts = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
        let sec = u64::try_from(ts.tv_sec).unwrap_or(0);
        let nsec = u64::try_from(ts.tv_nsec).unwrap_or(0);
        sec.saturating_mul(1_000_000_000).saturating_add(nsec)
    }
}

/// Test clock. [`SnapshotReader::set_now`] moves it.
#[derive(Clone, Copy, Debug)]
pub struct ManualMono {
    now_ns: u64,
}

impl ManualMono {
    #[must_use]
    pub fn new(now_ns: u64) -> Self {
        Self { now_ns }
    }
}

impl MonoNow for ManualMono {
    fn now_ns(&self) -> u64 {
        self.now_ns
    }
}

struct Accepted {
    run_id: u64,
    seq: u64,
}

/// One snapshot file plus the writer's own coolant read.
pub struct SnapshotReader<L, C = HostMono> {
    path: PathBuf,
    stale_after: Duration,
    sys_root: PathBuf,
    log: L,
    clock: C,
    last: Option<Accepted>,
    logged: Option<&'static str>,
}

impl SnapshotReader<log::Stderr, HostMono> {
    /// Production reader. The path is `llama_core::wire::SNAPSHOT_PATH`.
    ///
    /// Tests use [`SnapshotReader::new`] with a scratch file.
    #[must_use]
    pub fn open(stale_after: Duration) -> Self {
        Self::new(
            llama_core::wire::SNAPSHOT_PATH,
            stale_after,
            PathBuf::from(device::SYS_ROOT),
            log::Stderr,
            HostMono,
        )
    }
}

impl<L, C> SnapshotReader<L, C> {
    /// `path` is the snapshot file. Tests pass a scratch file, not `/run`.
    #[must_use]
    pub fn new(
        path: impl Into<PathBuf>,
        stale_after: Duration,
        sys_root: impl Into<PathBuf>,
        log: L,
        clock: C,
    ) -> Self {
        Self {
            path: path.into(),
            stale_after,
            sys_root: sys_root.into(),
            log,
            clock,
            last: None,
            logged: None,
        }
    }
}

impl<L> SnapshotReader<L, ManualMono> {
    /// Move the test clock. Freshness uses this, not [`Instant`].
    pub fn set_now(&mut self, now_ns: u64) {
        self.clock.now_ns = now_ns;
    }
}

impl<L, C> SnapshotReader<L, C>
where
    L: Sink,
    C: MonoNow,
{
    fn transition(&mut self, label: &'static str) {
        if self.logged == Some(label) {
            return;
        }
        self.logged = Some(label);
        let priority = if label == FRESH {
            Priority::Info
        } else {
            Priority::Err
        };
        log::emit(&mut self.log, priority, label);
    }

    fn blank(
        &mut self,
        mono: Instant,
        wall: SystemTime,
        label: &'static str,
        coolant: Option<f32>,
    ) -> Snapshot {
        self.transition(label);
        blank_snapshot(mono, wall, coolant)
    }

    fn read_file(&mut self) -> Result<WireSnapshot, &'static str> {
        let fd = match rustix::fs::open(
            &self.path,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::NONBLOCK
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        ) {
            Ok(fd) => fd,
            Err(err) if err == rustix::io::Errno::NOENT => return Err(MISSING),
            Err(_) => return Err(INVALID),
        };
        let stat = rustix::fs::fstat(&fd).map_err(|_| INVALID)?;
        let kind = rustix::fs::FileType::from_raw_mode(stat.st_mode);
        if !kind.is_file() {
            return Err(NOT_REGULAR);
        }
        let max = i64::try_from(wire::MAX_BYTES).map_err(|_| TOO_LARGE)?;
        if stat.st_size < 0 || stat.st_size > max {
            return Err(TOO_LARGE);
        }
        let mut buf = vec![0_u8; wire::MAX_BYTES + 1];
        let mut filled = 0_usize;
        while filled < buf.len() {
            match rustix::io::read(&fd, &mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(err) if err == rustix::io::Errno::INTR => continue,
                Err(_) => return Err(INVALID),
            }
        }
        if filled > wire::MAX_BYTES {
            return Err(TOO_LARGE);
        }
        wire::parse_validated(&buf[..filled]).map_err(|_| INVALID)
    }

    fn accept(&mut self, wire: &WireSnapshot, now_ns: u64) -> Result<(), &'static str> {
        if let Some(prev) = &self.last
            && wire.run_id == prev.run_id
            && wire.seq < prev.seq
        {
            return Err(SEQ_REGRESSION);
        }
        let slack = u64::try_from(FUTURE_SLACK.as_nanos()).unwrap_or(u64::MAX);
        if wire.t_mono_ns > now_ns.saturating_add(slack) {
            return Err(FROM_THE_FUTURE);
        }
        self.last = Some(Accepted {
            run_id: wire.run_id,
            seq: wire.seq,
        });
        let age = now_ns.saturating_sub(wire.t_mono_ns);
        let limit = u64::try_from(self.stale_after.as_nanos()).unwrap_or(u64::MAX);
        if age <= limit { Ok(()) } else { Err(STALE) }
    }
}

impl<L, C> Sampler for SnapshotReader<L, C>
where
    L: Sink,
    C: MonoNow,
{
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        let coolant = read_coolant_c(&self.sys_root);
        let now_ns = self.clock.now_ns();
        let wire = match self.read_file() {
            Ok(wire) => wire,
            Err(label) => return self.blank(mono, wall, label, coolant),
        };
        match self.accept(&wire, now_ns) {
            Ok(()) => {
                self.transition(FRESH);
                fresh_snapshot(&wire, mono, wall, coolant)
            }
            Err(label) => self.blank(mono, wall, label, coolant),
        }
    }
}

fn fresh_snapshot(
    wire: &WireSnapshot,
    mono: Instant,
    wall: SystemTime,
    coolant: Option<f32>,
) -> Snapshot {
    Snapshot {
        t_mono: mono,
        t_wall: wall,
        load: wire.host.load_pct,
        activity: wire.host.activity_pct,
        cpu_pct: wire.host.cpu_pct,
        cpu_topk_pct: wire.host.cpu_topk_pct,
        gpu_pct: wire.host.gpu_pct,
        mem_pct: wire.host.mem_pct,
        coolant_c: coolant,
        cpu_c: wire.host.cpu_c,
        gpu_c: wire.host.gpu_c,
        ai: map_ai(wire.ai.state),
        models: wire
            .ai
            .models
            .iter()
            .map(|model| ModelInfo {
                name: model.name.clone(),
                state: model_state(model.state).to_owned(),
                full_name: model.full_name.clone(),
                detail: model.detail.clone(),
                // The LCD draws no backend gauges, only the engine (#33)
                // and the speculative acceptance (#31) on the detail line.
                backend: Some(engine_and_spec(model)),
            })
            .collect(),
        tokens: Some(TokenReading {
            run_id: wire.run_id,
            seq: wire.seq,
            t_mono_ns: wire.t_mono_ns,
            decoded_total: wire.tokens.decoded_total,
        }),
        errors: std::collections::BTreeSet::new(),
    }
}

/// The model's engine (absent is llama.cpp, #33) with its speculative
/// acceptance as the only number, when the snapshot has one. Validation
/// already bounded it to 0..=1.
fn engine_and_spec(model: &llama_core::wire::ModelWire) -> llama_core::backend::BackendInfo {
    let spec_permille = model
        .engine
        .as_ref()
        .and_then(|engine| engine.spec_accept)
        .and_then(|accept| llama_core::backend::permille(f64::from(accept)));
    llama_core::backend::BackendInfo {
        kind: model.backend.unwrap_or_default(),
        engine: llama_core::backend::EngineStats {
            spec_permille,
            ..llama_core::backend::EngineStats::default()
        },
        ..llama_core::backend::BackendInfo::default()
    }
}

fn blank_snapshot(mono: Instant, wall: SystemTime, coolant: Option<f32>) -> Snapshot {
    Snapshot {
        t_mono: mono,
        t_wall: wall,
        load: None,
        activity: None,
        cpu_pct: None,
        cpu_topk_pct: None,
        gpu_pct: None,
        mem_pct: None,
        coolant_c: coolant,
        cpu_c: None,
        gpu_c: None,
        ai: AiState::NoData,
        models: Vec::new(),
        tokens: None,
        errors: std::collections::BTreeSet::new(),
    }
}

fn map_ai(state: AiWire) -> AiState {
    match state {
        AiWire::Down => AiState::Down,
        AiWire::Idle => AiState::Idle,
        AiWire::Loaded => AiState::Loaded,
    }
}

fn model_state(state: ModelState) -> &'static str {
    match state {
        ModelState::Ready => "ready",
        ModelState::Starting => "starting",
        ModelState::Stopping => "stopping",
        ModelState::Other => "other",
    }
}
