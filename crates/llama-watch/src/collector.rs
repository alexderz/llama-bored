//! Sources to one [`Snapshot`] per tick, including power-weighted load.
//!
//! [`WatchCollector`] takes a [`ValidWatchConfig`] and an injected
//! [`LlamaView`]. CPU percent is the delta about `cpu_window_s` back, so the
//! first window is empty. It never calls `/running` and never returns
//! [`AiState::NoData`].

use std::collections::{BTreeSet, VecDeque};
use std::marker::PhantomData;
use std::time::{Duration, Instant, SystemTime};

use crate::sources::fans::{FanPanel, FanSource};
use crate::sources::gpu::{GpuBackend, GpuSource};
use crate::sources::proc::{self, CpuSample};
use crate::sources::temps::{TempReading, TempSource};
use crate::sources::{self, Roots, SourceId};
use llama_core::log::{self, Priority, Sink};
use llama_core::names::sanitize_wire;
use llama_core::sample::LlamaView;
use llama_core::wire::ACTIVITY_MAX_PCT;

use crate::config::{IdleMode, ValidWatchConfig};
use crate::load::{IdleFloor, Winner};

pub use llama_core::sample::{AiState, ModelInfo, Snapshot};

/// VRAM and power for the TTY. Not on [`Snapshot`]; the publisher puts
/// them on the wire for llama-metrics (#11).
///
/// Each field is `None` when that NVML call failed. Bytes are from
/// `memory_info`; power is milliwatts from `power_usage` and
/// `enforced_power_limit`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuExtra {
    /// Used VRAM, bytes.
    pub vram_used: Option<u64>,
    /// Total VRAM, bytes.
    pub vram_total: Option<u64>,
    /// Current power, milliwatts.
    pub power_mw: Option<u32>,
    /// Enforced power limit, milliwatts.
    pub power_limit_mw: Option<u32>,
}

/// What set [`Snapshot::activity`] on this tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoadSource {
    /// The GPU's share of its NVML headroom was the larger.
    Gpu,
    /// The zenergy socket's share of `cpu_limit_w` was the larger.
    Cpu,
    /// `max(gpu_util, cpu_mean)`. Neither device had watts.
    Util,
}

impl LoadSource {
    /// `gpu`, `cpu` or `util`, as the tty ACTIVITY bar shows it.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
            Self::Util => "util",
        }
    }
}

/// One windowed tick: the snapshot plus TTY-only GPU numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct WatchSample {
    /// Host, AI, and power-weighted load for this tick.
    pub snapshot: Snapshot,
    /// VRAM and power. Absent from [`Snapshot`].
    pub gpu: GpuExtra,
    /// Socket power in watts. `None` until two zenergy reads span a tick.
    pub cpu_w: Option<f64>,
    /// Watts of the device in [`Self::load_source`]. `None` for `Util`.
    pub activity_w: Option<f64>,
    /// What set [`Snapshot::activity`] on this tick.
    pub load_source: LoadSource,
    /// Read-only fan speeds for the tty FANS panel. `None` when `[fans]` is
    /// off. Absent from [`Snapshot`]; the publisher copies rpm and pwm to the
    /// wire (#11).
    pub fans: Option<FanPanel>,
    /// Every shown temperature, GPU included, most important first (#74).
    /// `None` when `[temps]` is off. Read at most once a second.
    pub temps: Option<Vec<TempReading>>,
}

/// 10 Hz collector. [`Self::sample`] takes the clock and a llama view.
pub struct WatchCollector<'a, B, L> {
    inner: Collector<'a, B, L>,
}

impl<'a, B, L> WatchCollector<'a, B, L>
where
    B: GpuBackend,
    L: Sink,
{
    /// `backend` is the GPU session. Tests pass a fake.
    ///
    /// `config` is borrowed for the life of the collector. CPU percent uses
    /// `collector.cpu_window_s`.
    pub fn new(roots: Roots, backend: B, config: &'a ValidWatchConfig, log: L) -> Self {
        Self {
            inner: Collector::watch(roots, backend, config, log),
        }
    }

    /// One tick from `llama`. Does not call `/running`.
    ///
    /// CPU percent is `None` until a stored sample is at least `cpu_window_s` old.
    /// [`AiState::NoData`] in the view is reported as [`AiState::Down`].
    pub fn sample(&mut self, mono: Instant, wall: SystemTime, llama: &LlamaView) -> WatchSample {
        self.inner.sample_at(mono, wall, llama)
    }
}

struct Activity {
    load: Option<f32>,
    source: LoadSource,
    cpu_w: Option<f64>,
    device_w: Option<f64>,
}

/// Host percentages are 0..=100 on the snapshot. Non-finite values are absent.
fn clamp_pct(value: Option<f32>) -> Option<f32> {
    clamp_to(value, 100.0)
}

/// Activity is 0..=[`ACTIVITY_MAX_PCT`]: 100 is nominal, spikes read above.
fn clamp_activity(value: Option<f32>) -> Option<f32> {
    clamp_to(value, ACTIVITY_MAX_PCT)
}

fn clamp_to(value: Option<f32>, high: f32) -> Option<f32> {
    let value = value?;
    if !value.is_finite() {
        return None;
    }
    Some(value.clamp(0.0, high))
}

/// `max(gpu_pct, cpu_topk_pct)`. One missing term leaves the other.
fn composite_load(gpu_pct: Option<f32>, other: Option<f32>) -> Option<f32> {
    match (gpu_pct, other) {
        (Some(gpu), Some(cpu)) => Some(gpu.max(cpu)),
        (Some(gpu), None) => Some(gpu),
        (None, Some(cpu)) => Some(cpu),
        (None, None) => None,
    }
}

/// Host and GPU reads for one [`WatchCollector`] tick.
pub(crate) struct Collector<'a, B, L> {
    roots: Roots,
    gpu: GpuSource<B>,
    cpu_top_k: usize,
    /// CPU delta is taken against the newest sample at least this old.
    cpu_window: Duration,
    cpu_limit_w: f64,
    nominal_frac: f64,
    gpu_idle: IdleFloor,
    cpu_idle: IdleFloor,
    smooth_s: f64,
    ema: Option<f64>,
    ema_at: Option<Instant>,
    /// `(mono, Esocket µJ counters)` from the previous successful read.
    energy_prev: Option<(Instant, Vec<u64>)>,
    log: L,
    /// `(mono, counters)` from oldest to newest.
    cpu_history: VecDeque<(Instant, CpuSample)>,
    /// Sources currently failed. The set, not the message, decides the next log.
    failed: BTreeSet<SourceId>,
    vram_failed: bool,
    power_failed: bool,
    limit_failed: bool,
    llama: Option<LlamaSeen>,
    /// `[llama] enabled`. When false every view is ignored.
    llama_enabled: bool,
    /// `[fans]`, read on every tick like the other hwmon sources.
    fans: Option<FanSource>,
    /// `[temps]` (#74): discovered every 30 s, read at most once a second.
    temps: Option<TempSource>,
    _config: PhantomData<&'a ()>,
}

/// Last llama-swap state that was logged. The Down reason is not part of it.
#[derive(Clone, Copy, Eq, PartialEq)]
enum LlamaSeen {
    Down,
    Idle,
    Loaded,
}

impl<'a, B, L> Collector<'a, B, L>
where
    B: GpuBackend,
    L: Sink,
{
    fn sample_result(
        &mut self,
        id: SourceId,
        errors: &mut BTreeSet<SourceId>,
        read: impl FnOnce(&Roots) -> Result<f32, sources::SourceError>,
    ) -> Option<f32> {
        match read(&self.roots) {
            Ok(value) => {
                self.note(id, Ok(()));
                Some(value)
            }
            Err(err) => {
                errors.insert(id);
                self.note(id, Err(err.message));
                None
            }
        }
    }

    /// `failed` is true when init or a util/temp read failed. Both percents are then `None`.
    fn read_gpu(&mut self, errors: &mut BTreeSet<SourceId>) -> (Option<f32>, Option<f32>, bool) {
        let sample = self.gpu.read();
        if let Some(err) = sample.error {
            errors.insert(SourceId::Gpu);
            self.note(SourceId::Gpu, Err(err.message));
            (None, None, true)
        } else {
            self.note(SourceId::Gpu, Ok(()));
            (sample.gpu_pct, sample.gpu_c, false)
        }
    }

    /// Log once on entering failure, once on recovery.
    ///
    /// The first failure's message is included. Later failures of the same
    /// source stay quiet even when the text changes.
    fn note(&mut self, id: SourceId, outcome: Result<(), String>) {
        match outcome {
            Ok(()) => {
                if self.failed.remove(&id) {
                    log::emit(&mut self.log, Priority::Info, &format!("{id} recovered"));
                }
            }
            Err(message) => {
                if !self.failed.insert(id) {
                    return;
                }
                log::emit(
                    &mut self.log,
                    Priority::Err,
                    &format!("{id} failed: {message}"),
                );
            }
        }
    }

    /// Same once-per-transition rule as [`Self::note`], for one NVML extra call.
    ///
    /// `message` is `None` on success. Returns whether the call is now failed.
    fn note_part(&mut self, was_failed: bool, label: &'static str, message: Option<&str>) -> bool {
        match message {
            None => {
                if was_failed {
                    log::emit(
                        &mut self.log,
                        Priority::Info,
                        &format!("gpu {label} recovered"),
                    );
                }
                false
            }
            Some(message) => {
                if !was_failed {
                    log::emit(
                        &mut self.log,
                        Priority::Err,
                        &format!("gpu failed: {label}: {message}"),
                    );
                }
                true
            }
        }
    }
}

impl<'a, B, L> Collector<'a, B, L>
where
    B: GpuBackend,
    L: Sink,
{
    /// `backend` is the GPU session. Tests pass a fake.
    ///
    /// `config` is borrowed for the life of the collector. CPU percent uses
    /// `config.collector.cpu_window_s`, which [`ValidWatchConfig`] keeps finite.
    ///
    fn watch(roots: Roots, backend: B, config: &'a ValidWatchConfig, log: L) -> Self {
        let learn = config.load.idle == IdleMode::Auto;
        Self {
            roots,
            gpu: GpuSource::new(backend),
            cpu_top_k: config.collector.cpu_top_k as usize,
            cpu_window: Duration::from_secs_f64(config.collector.cpu_window_s),
            cpu_limit_w: config.load.cpu_limit_w,
            nominal_frac: config.load.nominal_frac,
            gpu_idle: IdleFloor::new(config.load.gpu_idle_w, learn, config.load.nominal_frac),
            cpu_idle: IdleFloor::new(config.load.cpu_idle_w, learn, config.load.nominal_frac),
            smooth_s: config.load.smooth_s,
            ema: None,
            ema_at: None,
            energy_prev: None,
            log,
            cpu_history: VecDeque::new(),
            failed: BTreeSet::new(),
            vram_failed: false,
            power_failed: false,
            limit_failed: false,
            llama: None,
            llama_enabled: config.llama.enabled,
            fans: FanSource::new(&config.fans),
            temps: TempSource::new(&config.temps),
            _config: PhantomData,
        }
    }

    fn sample_at(&mut self, mono: Instant, wall: SystemTime, llama: &LlamaView) -> WatchSample {
        let mut errors = BTreeSet::new();
        let (cpu_pct, cpu_topk_pct) = self.sample_cpu_window(mono, &mut errors);
        let mem_pct = self.sample_result(SourceId::ProcMem, &mut errors, |roots| {
            sources::proc::read_mem(roots)
        });
        let coolant_c = self.sample_result(SourceId::HwmonCoolant, &mut errors, |roots| {
            sources::hwmon::read_coolant(roots)
        });
        let cpu_c = self.sample_result(SourceId::HwmonCpu, &mut errors, |roots| {
            sources::hwmon::read_cpu_temp(roots)
        });
        let fans = self
            .fans
            .as_mut()
            .map(|source| source.read(&self.roots, mono, &mut self.log));
        let (gpu_pct, gpu_c, gpu_failed) = self.read_gpu(&mut errors);
        let temps = self
            .temps
            .as_mut()
            .map(|source| source.read(&self.roots, mono, gpu_c, &mut self.log));
        let gpu = if gpu_failed {
            GpuExtra::default()
        } else {
            self.read_gpu_extra()
        };
        let (ai, models) = self.apply_view(llama, &mut errors);
        let activity = self.activity_load(mono, gpu_pct, cpu_pct, &gpu);
        WatchSample {
            snapshot: Snapshot {
                t_mono: mono,
                t_wall: wall,
                load: clamp_pct(composite_load(gpu_pct, cpu_topk_pct)),
                activity: clamp_activity(activity.load),
                cpu_pct: clamp_pct(cpu_pct),
                cpu_topk_pct: clamp_pct(cpu_topk_pct),
                gpu_pct: clamp_pct(gpu_pct),
                mem_pct: clamp_pct(mem_pct),
                coolant_c,
                cpu_c,
                gpu_c,
                ai,
                models,
                tokens: None,
                errors,
            },
            gpu,
            cpu_w: activity.cpu_w,
            activity_w: activity.device_w,
            load_source: activity.source,
            fans,
            temps,
        }
    }

    /// `max(gpu_frac, cpu_frac)`; one device's fraction when the other has no
    /// watts; `max(gpu_pct, cpu_mean)` when neither has.
    fn activity_load(
        &mut self,
        mono: Instant,
        gpu_pct: Option<f32>,
        cpu_pct: Option<f32>,
        gpu: &GpuExtra,
    ) -> Activity {
        let cpu_w = self.poll_socket_watts(mono);
        let gpu_w = gpu.power_mw.map(|mw| f64::from(mw) / 1000.0);
        let gpu_limit_w = gpu.power_limit_mw.map(|mw| f64::from(mw) / 1000.0);
        let gpu_pair = gpu_w.zip(gpu_limit_w);
        self.gpu_idle
            .observe(mono, gpu_pair.map(|(watts, _)| watts));
        self.cpu_idle.observe(mono, cpu_w);
        let gpu_frac = gpu_pair.and_then(|(watts, limit)| {
            crate::load::device_frac(watts, self.gpu_idle.watts(limit), limit, self.nominal_frac)
        });
        let cpu_frac = cpu_w.and_then(|watts| {
            crate::load::device_frac(
                watts,
                self.cpu_idle.watts(self.cpu_limit_w),
                self.cpu_limit_w,
                self.nominal_frac,
            )
        });
        let (raw, source, device_w) = match crate::load::bottleneck(gpu_frac, cpu_frac) {
            Some((pct, Winner::Gpu)) => (Some(pct), LoadSource::Gpu, gpu_w),
            Some((pct, Winner::Cpu)) => (Some(pct), LoadSource::Cpu, cpu_w),
            None => (
                composite_load(gpu_pct, cpu_pct).map(f64::from),
                LoadSource::Util,
                None,
            ),
        };
        Activity {
            load: raw.map(|sample| self.push_ema(mono, sample) as f32),
            source,
            cpu_w,
            device_w,
        }
    }

    fn push_ema(&mut self, mono: Instant, sample: f64) -> f64 {
        let dt = self
            .ema_at
            .map(|then| mono.saturating_duration_since(then).as_secs_f64())
            .unwrap_or(0.0);
        let next = crate::load::ema_step(self.ema, sample, dt, self.smooth_s);
        self.ema = Some(next);
        self.ema_at = Some(mono);
        next
    }

    fn poll_socket_watts(&mut self, mono: Instant) -> Option<f64> {
        let now = sources::hwmon::read_socket_energy_uj(&self.roots).ok()?;
        if now.is_empty() {
            return None;
        }
        if self
            .energy_prev
            .as_ref()
            .is_some_and(|(then, _)| mono < *then)
        {
            return None;
        }
        let watts = self.energy_prev.as_ref().and_then(|(then, prev)| {
            let dt = mono.saturating_duration_since(*then).as_secs_f64();
            crate::load::socket_watts(prev, &now, dt)
        });
        self.energy_prev = Some((mono, now));
        watts
    }

    /// Delta against the newest sample at least `cpu_window` old.
    fn sample_cpu_window(
        &mut self,
        mono: Instant,
        errors: &mut BTreeSet<SourceId>,
    ) -> (Option<f32>, Option<f32>) {
        let window = self.cpu_window;
        let baseline = self.cpu_baseline(mono, window);
        let sample = match proc::read_cpu(&self.roots, baseline.as_ref()) {
            Ok(sample) => sample,
            Err(err) => {
                errors.insert(SourceId::ProcCpu);
                self.note(SourceId::ProcCpu, Err(err.message));
                self.cpu_history.clear();
                return (None, None);
            }
        };
        self.note(SourceId::ProcCpu, Ok(()));
        let means = sample.busy_pct.as_deref().map(|busy| {
            (
                proc::plain_mean(busy),
                proc::busiest_mean(busy, self.cpu_top_k),
            )
        });
        self.push_cpu(mono, sample, window);
        match means {
            Some((plain, topk)) => (plain, topk),
            None => (None, None),
        }
    }

    fn cpu_baseline(&self, mono: Instant, window: Duration) -> Option<CpuSample> {
        self.cpu_history
            .iter()
            .rev()
            .find(|(stamp, _)| mono.saturating_duration_since(*stamp) >= window)
            .map(|(_, sample)| sample.clone())
    }

    fn push_cpu(&mut self, mono: Instant, sample: CpuSample, window: Duration) {
        self.cpu_history.retain(|(stamp, _)| *stamp <= mono);
        self.cpu_history.push_back((mono, sample));
        let Some(keep) = self
            .cpu_history
            .iter()
            .rposition(|(stamp, _)| mono.saturating_duration_since(*stamp) >= window)
        else {
            return;
        };
        for _ in 0..keep {
            self.cpu_history.pop_front();
        }
    }

    fn read_gpu_extra(&mut self) -> GpuExtra {
        let reading = self.gpu.read_extra();
        let (vram_used, vram_total) = match reading.memory {
            Ok((used, total)) => {
                self.vram_failed = self.note_part(self.vram_failed, "vram", None);
                (Some(used), Some(total))
            }
            Err(err) => {
                let message = err.to_string();
                self.vram_failed = self.note_part(self.vram_failed, "vram", Some(&message));
                (None, None)
            }
        };
        let power_mw = match reading.power_mw {
            Ok(mw) => {
                self.power_failed = self.note_part(self.power_failed, "power", None);
                Some(mw)
            }
            Err(err) => {
                let message = err.to_string();
                self.power_failed = self.note_part(self.power_failed, "power", Some(&message));
                None
            }
        };
        let power_limit_mw = match reading.power_limit_mw {
            Ok(mw) => {
                self.limit_failed = self.note_part(self.limit_failed, "power_limit", None);
                Some(mw)
            }
            Err(err) => {
                let message = err.to_string();
                self.limit_failed =
                    self.note_part(self.limit_failed, "power_limit", Some(&message));
                None
            }
        };
        GpuExtra {
            vram_used,
            vram_total,
            power_mw,
            power_limit_mw,
        }
    }

    /// Map the view onto the snapshot. [`AiState::NoData`] becomes [`AiState::Down`].
    ///
    /// With `[llama] enabled = false` the view is ignored: the snapshot is
    /// [`AiState::Idle`] with no models, no llama error, and no log line.
    fn apply_view(
        &mut self,
        llama: &LlamaView,
        errors: &mut BTreeSet<SourceId>,
    ) -> (AiState, Vec<ModelInfo>) {
        if !self.llama_enabled {
            return (AiState::Idle, Vec::new());
        }
        match llama.ai {
            AiState::NoData => {
                self.note_view(AiState::Down, Some("no data"));
                errors.insert(SourceId::Llama);
                (AiState::Down, Vec::new())
            }
            AiState::Down => {
                self.note_view(AiState::Down, None);
                errors.insert(SourceId::Llama);
                (AiState::Down, Vec::new())
            }
            AiState::Idle => {
                self.note_view(AiState::Idle, None);
                (AiState::Idle, Vec::new())
            }
            AiState::Loaded => {
                self.note_view(AiState::Loaded, None);
                let models = llama
                    .models
                    .iter()
                    .map(|model| ModelInfo {
                        name: sanitize_wire(&model.name),
                        state: model.state.clone(),
                        full_name: model.full_name.clone(),
                        detail: model.detail.clone(),
                        backend: model.backend,
                    })
                    .collect();
                (AiState::Loaded, models)
            }
        }
    }

    fn note_view(&mut self, ai: AiState, down_reason: Option<&'static str>) {
        let (seen, priority, line) = match ai {
            AiState::Down | AiState::NoData => {
                let line = match down_reason {
                    Some(reason) => format!("llama down: {reason}"),
                    None => "llama down".to_owned(),
                };
                (LlamaSeen::Down, Priority::Err, line)
            }
            AiState::Idle => (LlamaSeen::Idle, Priority::Info, "llama idle".to_owned()),
            AiState::Loaded => (LlamaSeen::Loaded, Priority::Info, "llama loaded".to_owned()),
        };
        if self.llama == Some(seen) {
            return;
        }
        self.llama = Some(seen);
        log::emit(&mut self.log, priority, &line);
    }
}

#[cfg(test)]
mod tests {
    use super::composite_load;

    #[test]
    fn composite_load_is_the_max_of_the_two_terms() {
        assert_eq!(composite_load(Some(70.0), Some(12.5)), Some(70.0));
        assert_eq!(composite_load(Some(5.0), Some(12.5)), Some(12.5));
        assert_eq!(composite_load(None, Some(12.5)), Some(12.5));
        assert_eq!(composite_load(Some(40.0), None), Some(40.0));
        assert_eq!(composite_load(None, None), None);
    }
}
