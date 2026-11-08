//! Sources to one [`Snapshot`] per tick, including power-weighted load.
//!
//! [`WatchCollector`] takes a [`ValidWatchConfig`] and an injected
//! [`LlamaView`]. CPU percent is the delta about `cpu_window_s` back, so the
//! first window is empty. It never calls `/running` and never returns
//! [`AiState::NoData`].

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::marker::PhantomData;
use std::time::{Duration, Instant, SystemTime};

use crate::sources::fans::{FanPanel, FanSource};
use crate::sources::gpu::{GpuBackend, GpuDeviceIdentity, GpuSource, ProcessRead};
use crate::sources::hostio::{
    self, DiskCounters, HostIo, LinkSpeeds, NetCounters, Probe, RateWindow, SourceHealth,
};
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

/// One compute process on the GPU, named from `/proc/<pid>/comm`.
///
/// Absent from [`Snapshot`]. The header uses it; metrics and the LCD do not.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuProcess {
    pub pid: u32,
    /// Used GPU memory, bytes.
    pub memory_bytes: u64,
    /// Sanitised executable name. Empty when `comm` could not be read.
    pub name: String,
    /// Short labels of the devices this process occupies. A model that spans
    /// more than one GPU keeps every label here for a later per-GPU display.
    /// Empty when the device has no label.
    pub device_labels: Vec<String>,
}

/// One selected GPU's independently sampled telemetry.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuDeviceTelemetry {
    pub identity: GpuDeviceIdentity,
    pub label: String,
    pub util_pct: Option<f32>,
    pub temp_c: Option<f32>,
    pub extra: GpuExtra,
}

/// Compute processes from the last NVML attempt.
///
/// `failed` stays true for the whole second after an error, including ticks
/// that reuse the cached failure. A down GPU session clears `list` and does
/// not set `failed`: util or temperature already logged that.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GpuProcesses {
    pub list: Vec<GpuProcess>,
    pub failed: bool,
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
    /// Selected devices for per-device publishing.
    pub gpu_devices: Vec<GpuDeviceTelemetry>,
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
    /// Compute processes on the GPU. Read at most once a second. Absent
    /// from [`Snapshot`].
    pub gpu_processes: GpuProcesses,
    /// Disk, network, swap and PSI from this tick. Rates are over the
    /// trailing `cpu_window_s` and stay `None` until a stored sample is that
    /// old; health says whether the file was readable.
    pub host_io: HostIo,
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

fn device_key(id: &GpuDeviceIdentity, now: Instant, single: bool) -> String {
    if let Some(uuid) = id.uuid.as_deref().filter(|s| !s.is_empty()) {
        format!("uuid:{}", uuid.to_ascii_lowercase())
    } else if let Some(pci) = id.pci_bus_id.as_deref().filter(|s| !s.is_empty()) {
        format!("pci:{}", pci.to_ascii_lowercase())
    } else if single {
        // Legacy single-device backends have no identity; retain their idle learning.
        "anonymous-single".into()
    } else {
        // Never carry an unidentified device's state across multi-device reads.
        format!("anonymous-tick:{now:?}:{}", id.index)
    }
}
fn sum_u64(values: impl Iterator<Item = Option<u64>>) -> Option<u64> {
    values.flatten().reduce(u64::saturating_add)
}
fn sum_u32(values: impl Iterator<Item = Option<u32>>) -> Option<u32> {
    values.flatten().reduce(u32::saturating_add)
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
    gpu_idle: BTreeMap<String, IdleFloor>,
    gpu_config: crate::config::GpuConfig,
    gpu_idle_w: f64,
    gpu_learn: bool,
    process_cache: BTreeMap<String, Vec<GpuProcess>>,
    process_errors: BTreeMap<String, String>,
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
    /// Last process list. Cleared when the read fails or the session drops.
    gpu_processes: Vec<GpuProcess>,
    processes_failed: bool,
    /// Whole-disk counters over `cpu_window`, for the byte rate.
    disk_rates: RateWindow<DiskCounters>,
    /// Physical-interface counters over `cpu_window`, for the byte rate.
    net_rates: RateWindow<NetCounters>,
    /// NET ceiling, re-read every few seconds.
    links: LinkSpeeds,
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
        let window = Duration::from_secs_f64(config.collector.cpu_window_s);
        Self {
            roots,
            gpu: GpuSource::new(backend),
            cpu_top_k: config.collector.cpu_top_k as usize,
            cpu_window: window,
            cpu_limit_w: config.load.cpu_limit_w,
            nominal_frac: config.load.nominal_frac,
            gpu_idle: BTreeMap::new(),
            gpu_config: config.gpu.clone(),
            gpu_idle_w: config.load.gpu_idle_w,
            gpu_learn: learn,
            process_cache: BTreeMap::new(),
            process_errors: BTreeMap::new(),
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
            gpu_processes: Vec::new(),
            processes_failed: false,
            disk_rates: RateWindow::new(window),
            net_rates: RateWindow::new(window),
            links: LinkSpeeds::default(),
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
        let devices = self.read_selected_devices(mono, &mut errors);
        let gpu_pct = devices.iter().filter_map(|d| d.util_pct).reduce(f32::max);
        let gpu_c = devices.iter().filter_map(|d| d.temp_c).reduce(f32::max);
        let temps = self
            .temps
            .as_mut()
            .map(|source| source.read(&self.roots, mono, gpu_c, &mut self.log));
        let gpu = GpuExtra {
            vram_used: sum_u64(devices.iter().map(|d| d.extra.vram_used)),
            vram_total: sum_u64(devices.iter().map(|d| d.extra.vram_total)),
            power_mw: sum_u32(devices.iter().map(|d| d.extra.power_mw)),
            power_limit_mw: sum_u32(devices.iter().map(|d| d.extra.power_limit_mw)),
        };
        let (ai, models) = self.apply_view(llama, &mut errors);
        let activity = self.activity_load(mono, gpu_pct, cpu_pct, &devices);
        let host_io = self.sample_host_io(mono, &mut errors);
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
            gpu_devices: devices,
            cpu_w: activity.cpu_w,
            activity_w: activity.device_w,
            load_source: activity.source,
            fans,
            temps,
            gpu_processes: GpuProcesses {
                list: self.gpu_processes.clone(),
                failed: self.processes_failed,
            },
            host_io,
        }
    }

    /// Disk, net, swap and PSI. Missing files stay absent and out of
    /// `errors`. A present file that does not parse is one error. Disk and
    /// net rates use the CPU window, so they read as steadily as CPU %.
    fn sample_host_io(&mut self, mono: Instant, errors: &mut BTreeSet<SourceId>) -> HostIo {
        let mut io = HostIo::default();
        match hostio::read_disk(&self.roots) {
            Probe::Ready(now) => {
                self.note(SourceId::ProcDisk, Ok(()));
                if let Some((dt, prev)) = self.disk_rates.push(mono, now) {
                    io.disk_read_bps = hostio::sector_rate(prev.read_sectors, now.read_sectors, dt);
                    io.disk_write_bps =
                        hostio::sector_rate(prev.write_sectors, now.write_sectors, dt);
                }
                io.disk = SourceHealth::Ok;
            }
            Probe::Missing => {
                self.clear_source(SourceId::ProcDisk);
                self.disk_rates.clear();
                io.disk = SourceHealth::Absent;
            }
            Probe::Failed(message) => {
                errors.insert(SourceId::ProcDisk);
                self.note(SourceId::ProcDisk, Err(message));
                self.disk_rates.clear();
                io.disk = SourceHealth::Down;
            }
        }
        match hostio::read_net(&self.roots, &mut self.links, mono) {
            Probe::Ready(now) => {
                self.note(SourceId::ProcNet, Ok(()));
                if let Some((dt, prev)) = self.net_rates.push(mono, now) {
                    io.net_rx_bps = hostio::counter_rate(prev.rx_bytes, now.rx_bytes, dt);
                    io.net_tx_bps = hostio::counter_rate(prev.tx_bytes, now.tx_bytes, dt);
                }
                io.net_ceiling_bps = now.ceiling_bps.map(|bps| bps as f64);
                io.net = SourceHealth::Ok;
            }
            Probe::Missing => {
                self.clear_source(SourceId::ProcNet);
                self.net_rates.clear();
                io.net = SourceHealth::Absent;
            }
            Probe::Failed(message) => {
                errors.insert(SourceId::ProcNet);
                self.note(SourceId::ProcNet, Err(message));
                self.net_rates.clear();
                io.net = SourceHealth::Down;
            }
        }
        match hostio::read_swap(&self.roots) {
            Probe::Ready(swap) => {
                io.swap_used_bytes = Some(swap.used);
                io.swap_total_bytes = Some(swap.total);
            }
            // Swap rides on meminfo. A bad or missing pair leaves the meter
            // absent; proc health already covers a missing meminfo.
            Probe::Missing | Probe::Failed(_) => {}
        }
        match hostio::read_psi(&self.roots) {
            Probe::Ready(psi) => {
                self.note(SourceId::ProcPsi, Ok(()));
                io.cpu_psi = psi.cpu;
                io.io_psi = psi.io;
                io.mem_psi = psi.memory;
                io.psi = SourceHealth::Ok;
            }
            Probe::Missing => {
                self.clear_source(SourceId::ProcPsi);
                io.psi = SourceHealth::Absent;
            }
            Probe::Failed(message) => {
                errors.insert(SourceId::ProcPsi);
                self.note(SourceId::ProcPsi, Err(message));
                io.psi = SourceHealth::Down;
            }
        }
        io
    }

    /// A source that was down and is now simply absent stops being an error.
    fn clear_source(&mut self, id: SourceId) {
        if self.failed.contains(&id) {
            self.note(id, Ok(()));
        }
    }

    fn read_selected_devices(
        &mut self,
        now: Instant,
        errors: &mut BTreeSet<SourceId>,
    ) -> Vec<GpuDeviceTelemetry> {
        let rows = match self.gpu.read_devices(now) {
            Ok(rows) => rows,
            Err(err) => {
                errors.insert(SourceId::Gpu);
                self.note(SourceId::Gpu, Err(err.to_string()));
                self.process_cache.clear();
                self.process_errors.clear();
                self.gpu_processes.clear();
                for idle in self.gpu_idle.values_mut() {
                    idle.observe(now, None);
                }
                return Vec::new();
            }
        };
        let rows = rows
            .into_iter()
            .filter(|r| self.gpu_config.selects(&r.identity))
            .collect::<Vec<_>>();
        let mut failed = None;
        let mut vram_error = None;
        let mut power_error = None;
        let mut limit_error = None;
        let mut live = BTreeSet::new();
        let mut result = Vec::new();
        let single = rows.len() == 1;
        for row in rows {
            let key = device_key(&row.identity, now, single);
            live.insert(key.clone());
            let r = row.reading;
            if let Err(e) = &r.util_pct {
                failed = Some(e.to_string());
            }
            if let Err(e) = &r.temp_c {
                failed = Some(e.to_string());
            }
            if let Err(e) = &r.memory {
                vram_error = Some(e.to_string());
            }
            if let Err(e) = &r.power_mw {
                power_error = Some(e.to_string());
            }
            if let Err(e) = &r.power_limit_mw {
                limit_error = Some(e.to_string());
            }
            match r.processes {
                ProcessRead::Ready(items) => {
                    self.process_errors.remove(&key);
                    let device_label = self.gpu_config.label(&row.identity);
                    self.process_cache.insert(
                        key,
                        items
                            .into_iter()
                            .map(|p| GpuProcess {
                                pid: p.pid,
                                memory_bytes: p.memory_bytes,
                                name: crate::sources::gpu::read_process_name(
                                    &self.roots.proc,
                                    p.pid,
                                ),
                                device_labels: if device_label.is_empty() {
                                    Vec::new()
                                } else {
                                    vec![device_label.clone()]
                                },
                            })
                            .collect(),
                    );
                }
                ProcessRead::Failed(message) => {
                    self.process_cache.remove(&key);
                    self.process_errors.insert(key, message);
                }
                ProcessRead::Unavailable => {
                    self.process_cache.remove(&key);
                    self.process_errors.remove(&key);
                }
                ProcessRead::Cached => {}
            }
            let memory = r.memory.ok();
            result.push(GpuDeviceTelemetry {
                label: self.gpu_config.label(&row.identity),
                identity: row.identity,
                util_pct: r.util_pct.ok(),
                temp_c: r.temp_c.ok(),
                extra: GpuExtra {
                    vram_used: memory.map(|m| m.0),
                    vram_total: memory.map(|m| m.1),
                    power_mw: r.power_mw.ok(),
                    power_limit_mw: r.power_limit_mw.ok(),
                },
            });
        }
        self.process_cache.retain(|key, _| live.contains(key));
        self.gpu_idle.retain(|key, _| live.contains(key));
        self.process_errors.retain(|key, _| live.contains(key));
        if let Some(message) = failed {
            errors.insert(SourceId::Gpu);
            self.note(SourceId::Gpu, Err(message));
        } else {
            self.note(SourceId::Gpu, Ok(()));
        }
        if !result.is_empty() {
            self.vram_failed = self.note_part(self.vram_failed, "vram", vram_error.as_deref());
            self.power_failed = self.note_part(self.power_failed, "power", power_error.as_deref());
            self.limit_failed =
                self.note_part(self.limit_failed, "power limit", limit_error.as_deref());
        }
        let proc_error = self.process_errors.values().next().cloned();
        self.processes_failed =
            self.note_part(self.processes_failed, "processes", proc_error.as_deref());
        let mut merged: Vec<GpuProcess> = Vec::new();
        for process in self.process_cache.values().flatten() {
            if let Some(p) = merged.iter_mut().find(|p| p.pid == process.pid) {
                p.memory_bytes = p.memory_bytes.saturating_add(process.memory_bytes);
                for label in &process.device_labels {
                    if !p.device_labels.iter().any(|have| have == label) {
                        p.device_labels.push(label.clone());
                    }
                }
            } else {
                merged.push(process.clone());
            }
        }
        self.gpu_processes = merged;
        result
    }

    /// `max(gpu_frac, cpu_frac)`; one device's fraction when the other has no
    /// watts; `max(gpu_pct, cpu_mean)` when neither has.
    fn activity_load(
        &mut self,
        mono: Instant,
        gpu_pct: Option<f32>,
        cpu_pct: Option<f32>,
        devices: &[GpuDeviceTelemetry],
    ) -> Activity {
        let cpu_w = self.poll_socket_watts(mono);
        let mut gpu_best: Option<(f64, f64)> = None;
        for device in devices {
            let key = device_key(&device.identity, mono, devices.len() == 1);
            let idle_w = self.gpu_config.idle_w(&device.identity, self.gpu_idle_w);
            let has_override =
                self.gpu_config.idle_w.iter().any(|(selector, _)| {
                    crate::config::GpuConfig::matches(selector, &device.identity)
                });
            let idle = self.gpu_idle.entry(key).or_insert_with(|| {
                IdleFloor::new(idle_w, self.gpu_learn && !has_override, self.nominal_frac)
            });
            let pair = device
                .extra
                .power_mw
                .zip(device.extra.power_limit_mw)
                .map(|(w, limit)| (f64::from(w) / 1000.0, f64::from(limit) / 1000.0));
            idle.observe(mono, pair.map(|(watts, _)| watts));
            if let Some((watts, limit)) = pair {
                if let Some(frac) =
                    crate::load::device_frac(watts, idle.watts(limit), limit, self.nominal_frac)
                {
                    if gpu_best.is_none_or(|(best, _)| frac > best) {
                        gpu_best = Some((frac, watts));
                    }
                }
            }
        }
        let gpu_frac = gpu_best.map(|(frac, _)| frac);
        let gpu_w = gpu_best.map(|(_, watts)| watts);
        self.cpu_idle.observe(mono, cpu_w);
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
