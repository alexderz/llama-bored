//! GPU utilisation, temperature, and the compute processes on the card.
//!
//! [`GpuSource`] owns a [`GpuBackend`]. It initialises the backend once.
//! An init failure waits 1, 2, 4, … reads (capped at 60) before trying again,
//! and a successful init resets that wait. Any later read error drops the
//! session so the next read initialises immediately, and clears both fields.
//! [`NvidiaGpu`] loads the host library; unit tests use a fake and do not.
//!
//! Compute processes are a separate read. It runs at most once a second,
//! only while the session is up, and a failure there clears the list without
//! dropping the session. A tick inside that second returns [`ProcessRead::Cached`]
//! and does not call the backend or open `/proc`.

use super::{SourceError, SourceId};
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::enums::device::{SampleValue, UsedGpuMemory};
use nvml_wrapper::struct_wrappers::device::{FieldValueSample, ProcessInfo};
use nvml_wrapper::structs::device::FieldId;
use nvml_wrapper::sys_exports::field_id::NVML_FI_DEV_MEMORY_TEMP;
use nvml_wrapper::{Device, Nvml};
use std::ffi::OsStr;
use std::path::Path;
use std::time::{Duration, Instant};

/// Failure from a [`GpuBackend`] call, before it is tagged with [`SourceId::Gpu`].
#[derive(Debug)]
pub struct GpuError {
    message: String,
}

impl GpuError {
    /// Backend error text, not yet tagged with [`SourceId::Gpu`].
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for GpuError {}

impl From<nvml_wrapper::error::NvmlError> for GpuError {
    fn from(err: nvml_wrapper::error::NvmlError) -> Self {
        Self::new(err.to_string())
    }
}

/// One GPU read: utilisation and temperature in the snapshot's units.
///
/// `error` is set when init or either read failed. A read failure clears both
/// `gpu_pct` and `gpu_c`.
#[derive(Debug)]
#[must_use]
pub struct GpuSample {
    pub gpu_pct: Option<f32>,
    pub gpu_c: Option<f32>,
    pub error: Option<SourceError>,
}

/// Backend for GPU utilisation, temperature, VRAM, and power.
///
/// [`NvidiaGpu`] is the host implementation. Tests supply a fake.
/// `disconnect` drops the live library handle. [`GpuSource`] calls it after a
/// util or temperature error. A VRAM or power error does not drop the session:
/// that field is `None` and the next tick tries again.
pub trait GpuBackend {
    /// Start a session. Not called again until [`Self::disconnect`].
    fn init(&mut self) -> Result<(), GpuError>;

    /// Utilisation of the GPU engines, in percent.
    fn util_pct(&mut self) -> Result<f32, GpuError>;

    /// GPU temperature in degrees Celsius.
    fn temp_c(&mut self) -> Result<f32, GpuError>;

    /// Used and total VRAM, in bytes. `memory_info` on the host.
    fn memory_info(&mut self) -> Result<(u64, u64), GpuError> {
        Err(GpuError::new("memory_info unsupported"))
    }

    /// Current draw, in milliwatts. `power_usage` on the host.
    fn power_usage(&mut self) -> Result<u32, GpuError> {
        Err(GpuError::new("power_usage unsupported"))
    }

    /// Enforced cap, in milliwatts. `enforced_power_limit` on the host.
    fn enforced_power_limit(&mut self) -> Result<u32, GpuError> {
        Err(GpuError::new("enforced_power_limit unsupported"))
    }

    /// Running compute processes: pid and used GPU memory in bytes.
    ///
    /// `running_compute_processes` on the host. Memory NVML reports as
    /// unavailable is left out by [`NvidiaGpu`]. The default is an empty
    /// list, so a fake that does not care stays quiet and logs nothing.
    fn compute_processes(&mut self) -> Result<Vec<ComputeProcess>, GpuError> {
        Ok(Vec::new())
    }

    /// Enumerate devices; legacy backends expose a single anonymous device.
    fn devices(&mut self) -> Result<Vec<GpuDeviceIdentity>, GpuError> {
        Ok(vec![GpuDeviceIdentity {
            index: 0,
            uuid: None,
            pci_bus_id: None,
            name: None,
        }])
    }
    /// Read a device independently; legacy backends implement device zero.
    fn device_reading(&mut self, index: u32, processes: bool) -> GpuDeviceReading {
        if index != 0 {
            return GpuDeviceReading::unavailable(index, "unknown GPU device");
        }
        GpuDeviceReading {
            index,
            util_pct: self.util_pct(),
            temp_c: self.temp_c(),
            memory: self.memory_info(),
            power_mw: self.power_usage(),
            power_limit_mw: self.enforced_power_limit(),
            // Legacy backends have no memory-junction field.
            mem_temp_c: None,
            processes: if processes {
                match self.compute_processes() {
                    Ok(items) => ProcessRead::Ready(items),
                    Err(err) => ProcessRead::Failed(err.to_string()),
                }
            } else {
                ProcessRead::Cached
            },
        }
    }

    /// Drop the live library handle. Safe to call when nothing is loaded.
    fn disconnect(&mut self);
}

/// Physical device identity. Missing metadata does not hide the device.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuDeviceIdentity {
    /// NVML index for reads; not a stable identity.
    pub index: u32,
    pub uuid: Option<String>,
    pub pci_bus_id: Option<String>,
    pub name: Option<String>,
}
/// Independent per-device fields; failures do not hide sibling fields or devices.
#[derive(Debug)]
pub struct GpuDeviceReading {
    pub index: u32,
    pub util_pct: Result<f32, GpuError>,
    pub temp_c: Result<f32, GpuError>,
    pub memory: Result<(u64, u64), GpuError>,
    pub power_mw: Result<u32, GpuError>,
    pub power_limit_mw: Result<u32, GpuError>,
    /// Memory-junction temperature in °C from `NVML_FI_DEV_MEMORY_TEMP`.
    /// `None` when the card has no such sensor, the read failed, or the
    /// value is not a plausible temperature. Never `Some(0)` for a miss:
    /// the line omits the number. This wrapper has no hotspot field;
    /// hotspot comes from hwmon, not from an invented NVML query.
    pub mem_temp_c: Option<f32>,
    pub processes: ProcessRead,
}
impl GpuDeviceReading {
    fn unavailable(index: u32, reason: &str) -> Self {
        Self {
            index,
            util_pct: Err(GpuError::new(reason)),
            temp_c: Err(GpuError::new(reason)),
            memory: Err(GpuError::new(reason)),
            power_mw: Err(GpuError::new(reason)),
            power_limit_mw: Err(GpuError::new(reason)),
            mem_temp_c: None,
            processes: ProcessRead::Failed(reason.into()),
        }
    }
}
#[derive(Debug)]
pub struct GpuDeviceSample {
    pub identity: GpuDeviceIdentity,
    pub reading: GpuDeviceReading,
}

/// One compute process on the GPU. Memory is bytes, or absent when NVML
/// does not report it (those are dropped before this struct is built).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub struct ComputeProcess {
    pub pid: u32,
    /// Used GPU memory, bytes.
    pub memory_bytes: u64,
}

/// How often [`GpuSource::read_processes`] calls the backend.
const PROCESS_EVERY: Duration = Duration::from_secs(1);

/// One process-list attempt. A tick inside [`PROCESS_EVERY`] is [`Self::Cached`]
/// and does not call the backend.
#[derive(Clone, Debug, Eq, PartialEq)]
#[must_use]
pub enum ProcessRead {
    /// The last attempt is still inside the second. The caller keeps its list.
    Cached,
    /// The session is down, so NVML was not called. The caller drops its list.
    /// This is not a process-read failure: util or temperature already logged.
    Unavailable,
    /// A fresh read. The caller replaces its list.
    Ready(Vec<ComputeProcess>),
    /// A fresh read failed. The caller drops its list.
    Failed(String),
}

/// Longest `/proc/<pid>/comm` read. The kernel's comm is 15 bytes plus a
/// newline; the cap only bounds a stand-in file.
const COMM_READ_MAX: usize = 64;
/// Characters kept from a process name. The header cuts here.
const PROCESS_NAME_MAX: usize = 24;

/// Keep only printable ASCII `[A-Za-z0-9._+-]`, at most [`PROCESS_NAME_MAX`].
/// Anything else, including a newline from `comm`, is dropped.
#[must_use]
pub(crate) fn sanitize_process_name(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.chars() {
        if out.len() == PROCESS_NAME_MAX {
            break;
        }
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '+' | '-') {
            out.push(ch);
        }
    }
    out
}

/// `{proc_root}/<pid>/comm`, sanitised. Empty when the file cannot be read
/// or nothing allowed remains.
///
/// The installed unit sets `ProtectProc=invisible`, which hides another
/// user's `/proc/<pid>`. That read fails and the header shows `pid N`.
/// A process of the llama-watch user still yields its comm.
pub(crate) fn read_process_name(proc_root: &Path, pid: u32) -> String {
    let path = proc_root.join(pid.to_string()).join("comm");
    let mut file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return String::new(),
    };
    let mut buf = [0u8; COMM_READ_MAX];
    let n = match std::io::Read::read(&mut file, &mut buf) {
        Ok(n) => n,
        Err(_) => return String::new(),
    };
    let Ok(text) = std::str::from_utf8(&buf[..n]) else {
        return String::new();
    };
    sanitize_process_name(text)
}

/// Drop processes whose memory NVML did not report.
fn retain_compute(listed: Vec<ProcessInfo>) -> Vec<ComputeProcess> {
    let mut processes = Vec::new();
    for info in listed {
        let UsedGpuMemory::Used(memory_bytes) = info.used_gpu_memory else {
            continue;
        };
        processes.push(ComputeProcess {
            pid: info.pid,
            memory_bytes,
        });
    }
    processes
}

/// Merge engine lists for one device without double-counting a process that
/// appears in both. Either engine may be unsupported on a given device.
fn merge_engine_processes(
    compute: Result<Vec<ComputeProcess>, GpuError>,
    graphics: Result<Vec<ComputeProcess>, GpuError>,
) -> Result<Vec<ComputeProcess>, GpuError> {
    let (compute, graphics) = match (compute, graphics) {
        (Err(err), Err(_)) => return Err(err),
        (compute, graphics) => (compute.unwrap_or_default(), graphics.unwrap_or_default()),
    };
    let mut merged = compute;
    for process in graphics {
        if let Some(existing) = merged.iter_mut().find(|row| row.pid == process.pid) {
            existing.memory_bytes = existing.memory_bytes.max(process.memory_bytes);
        } else {
            merged.push(process);
        }
    }
    Ok(merged)
}

/// VRAM and power from one extra read. Independent of util and temperature.
#[derive(Debug)]
#[must_use]
pub struct GpuReading {
    /// `(used, total)` bytes.
    pub memory: Result<(u64, u64), GpuError>,
    /// Milliwatts.
    pub power_mw: Result<u32, GpuError>,
    /// Milliwatts.
    pub power_limit_mw: Result<u32, GpuError>,
}

/// Longest init backoff, counted in reads.
const INIT_BACKOFF_MAX: u32 = 60;

/// `Some` when either read failed. Both reasons are kept so a log line names each.
fn read_failure_message(
    util: &Result<f32, GpuError>,
    temp: &Result<f32, GpuError>,
) -> Option<String> {
    match (util, temp) {
        (Ok(_), Ok(_)) => None,
        (Err(util_err), Ok(_)) => Some(format!("util: {util_err}")),
        (Ok(_), Err(temp_err)) => Some(format!("temp: {temp_err}")),
        (Err(util_err), Err(temp_err)) => Some(format!("util: {util_err}; temp: {temp_err}")),
    }
}

/// Session around a [`GpuBackend`]: init once, drop the handle on any error.
#[derive(Debug)]
pub struct GpuSource<B> {
    backend: B,
    ready: bool,
    /// Reads to skip before the next init attempt.
    init_wait: u32,
    /// Backoff, in reads, applied after the next init failure.
    init_backoff: u32,
    init_error: String,
    /// When the last process attempt finished. Cleared with the session,
    /// so the next live tick reads at once instead of waiting out the second.
    processes_at: Option<Instant>,
}

impl<B: GpuBackend> GpuSource<B> {
    /// Holds `backend` with no live session.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            ready: false,
            init_wait: 0,
            init_backoff: 1,
            init_error: String::new(),
            processes_at: None,
        }
    }

    /// Read utilisation and temperature for one tick.
    ///
    /// Initialises the backend on the first call. After an init failure, later
    /// attempts wait 1, then 2, 4, … reads, capped at 60. A successful init
    /// resets that sequence to one read.
    pub fn read(&mut self) -> GpuSample {
        if !self.ensure_ready() {
            return self.init_backoff_sample();
        }
        let util = self.backend.util_pct();
        let temp = self.backend.temp_c();
        let gpu_pct = util.as_ref().ok().copied();
        let gpu_c = temp.as_ref().ok().copied();
        let Some(message) = read_failure_message(&util, &temp) else {
            return GpuSample {
                gpu_pct,
                gpu_c,
                error: None,
            };
        };
        self.backend.disconnect();
        self.ready = false;
        GpuSample {
            gpu_pct: None,
            gpu_c: None,
            error: Some(SourceError::new(SourceId::Gpu, message)),
        }
    }

    fn note_session_failure(&mut self, message: String, backoff: bool) {
        self.backend.disconnect();
        self.ready = false;
        self.processes_at = None;
        self.init_error = message;
        if backoff {
            self.init_wait = self.init_backoff;
            self.init_backoff = self.init_backoff.saturating_mul(2).min(INIT_BACKOFF_MAX);
        }
    }

    fn ensure_ready(&mut self) -> bool {
        if !self.ready {
            if self.init_wait > 0 {
                self.init_wait -= 1;
                return false;
            }
            if let Err(err) = self.backend.init() {
                // A failed init may still have kept a handle. Drop it.
                self.backend.disconnect();
                self.init_error = format!("init: {err}");
                self.init_wait = self.init_backoff;
                self.init_backoff = self.init_backoff.saturating_mul(2).min(INIT_BACKOFF_MAX);
                return false;
            }
            self.ready = true;
            self.init_backoff = 1;
            self.init_wait = 0;
            self.init_error.clear();
        }
        true
    }
    /// Read all devices in PCI order; session enumeration failures disconnect,
    /// while individual device and field failures remain isolated.
    pub fn read_devices(&mut self, now: Instant) -> Result<Vec<GpuDeviceSample>, GpuError> {
        if !self.ensure_ready() {
            return Err(GpuError::new(self.init_error.clone()));
        }
        let mut devices = match self.backend.devices() {
            Ok(devices) => devices,
            Err(err) => {
                // A card falling off the bus must not reopen NVML every tick.
                self.note_session_failure(err.to_string(), true);
                return Err(err);
            }
        };
        devices.sort_by(|a, b| match (&a.pci_bus_id, &b.pci_bus_id) {
            (Some(a), Some(b)) => a.cmp(b),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.index.cmp(&b.index),
        });
        let processes = self
            .processes_at
            .is_none_or(|at| now.saturating_duration_since(at) >= PROCESS_EVERY);
        if processes {
            self.processes_at = Some(now);
        }
        let rows: Vec<_> = devices
            .into_iter()
            .map(|identity| {
                let reading = self.backend.device_reading(identity.index, processes);
                GpuDeviceSample { identity, reading }
            })
            .collect();
        // One failed device stays. If every selected device failed util or
        // temperature, drop the session as the one-GPU path does. The next
        // read opens NVML immediately; enumeration failure is what backs off.
        if !rows.is_empty()
            && rows.iter().all(|row| {
                read_failure_message(&row.reading.util_pct, &row.reading.temp_c).is_some()
            })
        {
            let message = rows
                .iter()
                .filter_map(|row| read_failure_message(&row.reading.util_pct, &row.reading.temp_c))
                .collect::<Vec<_>>()
                .join("; ");
            self.note_session_failure(message.clone(), false);
            return Err(GpuError::new(message));
        }
        Ok(rows)
    }

    /// VRAM and power. Does not drop the session when a call fails.
    ///
    /// When the session is down, no backend call is made and every field is an error.
    pub fn read_extra(&mut self) -> GpuReading {
        if !self.ready {
            let unavailable = || GpuError::new("gpu session is not ready");
            return GpuReading {
                memory: Err(unavailable()),
                power_mw: Err(unavailable()),
                power_limit_mw: Err(unavailable()),
            };
        }
        GpuReading {
            memory: self.backend.memory_info(),
            power_mw: self.backend.power_usage(),
            power_limit_mw: self.backend.enforced_power_limit(),
        }
    }

    /// Compute processes, at most once a second.
    ///
    /// Inside the second this returns [`ProcessRead::Cached`] and does not
    /// call the backend. A down session returns [`ProcessRead::Unavailable`]
    /// without a call. An NVML error returns [`ProcessRead::Failed`] and does
    /// not drop the session: the next util read still uses it.
    pub fn read_processes(&mut self, now: Instant) -> ProcessRead {
        if !self.ready {
            self.processes_at = None;
            return ProcessRead::Unavailable;
        }
        if let Some(at) = self.processes_at
            && now.saturating_duration_since(at) < PROCESS_EVERY
        {
            return ProcessRead::Cached;
        }
        self.processes_at = Some(now);
        match self.backend.compute_processes() {
            Ok(list) => ProcessRead::Ready(list),
            Err(err) => ProcessRead::Failed(err.to_string()),
        }
    }

    fn init_backoff_sample(&self) -> GpuSample {
        GpuSample {
            gpu_pct: None,
            gpu_c: None,
            error: Some(SourceError::new(SourceId::Gpu, self.init_error.clone())),
        }
    }
}

/// Absolute paths tried in order for the NVIDIA management library.
///
/// Fedora/RHEL, Debian/Ubuntu (x86_64), then Arch. A fixed list, never a
/// library-search-path lookup, so `LD_LIBRARY_PATH` cannot pick the library.
const NVML_LIB_PATHS: &[&str] = &[
    "/usr/lib64/libnvidia-ml.so.1",
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
    "/usr/lib/libnvidia-ml.so.1",
];

/// First `load(path)` that succeeds. When all fail, one error names each path.
fn load_first<T>(
    paths: &[&str],
    mut load: impl FnMut(&str) -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    let mut reasons = Vec::with_capacity(paths.len());
    for path in paths {
        match load(path) {
            Ok(value) => return Ok(value),
            Err(err) => reasons.push(format!("{path}: {err}")),
        }
    }
    Err(GpuError::new(format!(
        "no NVIDIA library loaded ({})",
        reasons.join("; ")
    )))
}

/// Host GPU. [`GpuBackend::init`] loads the first of [`NVML_LIB_PATHS`] by
/// absolute path. With no NVIDIA driver every init fails, GPU fields stay
/// empty, and activity falls back to utilisation.
///
/// The device handle borrows the library object, so device 0 is resolved on each
/// read and only the library is kept. [`GpuBackend::disconnect`] drops it.
#[derive(Debug, Default)]
pub struct NvidiaGpu {
    library: Option<Nvml>,
}

impl NvidiaGpu {
    /// Does not load the library.
    pub fn new() -> Self {
        Self::default()
    }

    fn device(&self) -> Result<Device<'_>, GpuError> {
        let Some(library) = self.library.as_ref() else {
            return Err(GpuError::new("GPU library is not initialised"));
        };
        library.device_by_index(0).map_err(GpuError::from)
    }
}

fn identities_from_count(
    count: u32,
    mut open: impl FnMut(u32) -> Result<GpuDeviceIdentity, ()>,
) -> Result<Vec<GpuDeviceIdentity>, GpuError> {
    let mut devices = Vec::new();
    let mut failed = 0u32;
    for index in 0..count {
        match open(index) {
            Ok(identity) => devices.push(identity),
            Err(()) => {
                failed += 1;
                devices.push(GpuDeviceIdentity {
                    index,
                    uuid: None,
                    pci_bus_id: None,
                    name: None,
                });
            }
        }
    }
    if count > 0 && failed == count {
        return Err(GpuError::new("every GPU device open failed"));
    }
    Ok(devices)
}

impl GpuBackend for NvidiaGpu {
    fn init(&mut self) -> Result<(), GpuError> {
        self.library = None;
        self.library = Some(load_first(NVML_LIB_PATHS, |path| {
            Ok(Nvml::builder()
                // The builder's path argument is `&OsStr`.
                .lib_path(OsStr::new(path))
                .init()?)
        })?);
        Ok(())
    }

    fn util_pct(&mut self) -> Result<f32, GpuError> {
        Ok(self.device()?.utilization_rates()?.gpu as f32)
    }

    fn temp_c(&mut self) -> Result<f32, GpuError> {
        Ok(self.device()?.temperature(TemperatureSensor::Gpu)? as f32)
    }

    fn memory_info(&mut self) -> Result<(u64, u64), GpuError> {
        let info = self.device()?.memory_info()?;
        Ok((info.used, info.total))
    }

    fn power_usage(&mut self) -> Result<u32, GpuError> {
        let mw = self.device()?.power_usage()?;
        Ok(mw)
    }

    fn enforced_power_limit(&mut self) -> Result<u32, GpuError> {
        let mw = self.device()?.enforced_power_limit()?;
        Ok(mw)
    }

    fn compute_processes(&mut self) -> Result<Vec<ComputeProcess>, GpuError> {
        let device = self.device()?;
        merge_engine_processes(
            device
                .running_compute_processes()
                .map(retain_compute)
                .map_err(GpuError::from),
            device
                .running_graphics_processes()
                .map(retain_compute)
                .map_err(GpuError::from),
        )
    }

    fn devices(&mut self) -> Result<Vec<GpuDeviceIdentity>, GpuError> {
        let library = self
            .library
            .as_ref()
            .ok_or_else(|| GpuError::new("GPU library is not initialised"))?;
        let count = library.device_count()?;
        identities_from_count(count, |index| match library.device_by_index(index) {
            Ok(device) => Ok(GpuDeviceIdentity {
                index,
                uuid: device.uuid().ok(),
                pci_bus_id: device.pci_info().ok().map(|info| info.bus_id),
                name: device.name().ok(),
            }),
            Err(_) => Err(()),
        })
    }
    fn device_reading(&mut self, index: u32, processes: bool) -> GpuDeviceReading {
        let device = self
            .library
            .as_ref()
            .ok_or_else(|| GpuError::new("GPU library is not initialised"))
            .and_then(|library| library.device_by_index(index).map_err(GpuError::from));
        let device = match device {
            Ok(device) => device,
            Err(err) => return GpuDeviceReading::unavailable(index, &err.to_string()),
        };
        GpuDeviceReading {
            index,
            util_pct: device
                .utilization_rates()
                .map(|v| v.gpu as f32)
                .map_err(GpuError::from),
            temp_c: device
                .temperature(TemperatureSensor::Gpu)
                .map(|v| v as f32)
                .map_err(GpuError::from),
            memory: device
                .memory_info()
                .map(|v| (v.used, v.total))
                .map_err(GpuError::from),
            power_mw: device.power_usage().map_err(GpuError::from),
            power_limit_mw: device.enforced_power_limit().map_err(GpuError::from),
            // A miss is None. It does not drop the session and it is not 0.
            mem_temp_c: device
                .field_values_for(&[FieldId(NVML_FI_DEV_MEMORY_TEMP)])
                .ok()
                .and_then(memory_celsius),
            processes: if processes {
                match merge_engine_processes(
                    device
                        .running_compute_processes()
                        .map(retain_compute)
                        .map_err(GpuError::from),
                    device
                        .running_graphics_processes()
                        .map(retain_compute)
                        .map_err(GpuError::from),
                ) {
                    Ok(items) => ProcessRead::Ready(items),
                    Err(err) => ProcessRead::Failed(err.to_string()),
                }
            } else {
                ProcessRead::Cached
            },
        }
    }

    fn disconnect(&mut self) {
        self.library = None;
    }
}

/// Degrees from one `NVML_FI_DEV_MEMORY_TEMP` sample list.
///
/// The outer call already failed into `None` before this runs. An
/// unsupported field, a sample error, a non-finite value, zero, or anything
/// outside 5..127 °C is omitted. Zero is not a reading: a failed sensor
/// must not paint `0°` or leave a hole where a number was.
fn memory_celsius(
    samples: Vec<Result<FieldValueSample, nvml_wrapper::error::NvmlError>>,
) -> Option<f32> {
    let sample = samples.into_iter().next()?.ok()?;
    let degrees = match sample.value.ok()? {
        SampleValue::U32(v) => f64::from(v),
        SampleValue::U64(v) => v as f64,
        SampleValue::I64(v) => v as f64,
        SampleValue::F64(v) => v,
    };
    if !degrees.is_finite() {
        return None;
    }
    let rounded = degrees.round();
    if !(5.0..127.0).contains(&rounded) {
        return None;
    }
    Some(rounded as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nvml_wrapper::enums::device::UsedGpuMemory;
    use nvml_wrapper::error::NvmlError;
    use nvml_wrapper::struct_wrappers::device::ProcessInfo;
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn memory_sample(value: Result<SampleValue, NvmlError>) -> FieldValueSample {
        FieldValueSample {
            field: FieldId(NVML_FI_DEV_MEMORY_TEMP),
            timestamp: 0,
            latency: 0,
            value,
        }
    }

    #[test]
    fn memory_temp_omits_errors_zero_and_implausible() {
        assert_eq!(memory_celsius(vec![]), None);
        assert_eq!(memory_celsius(vec![Err(NvmlError::NotSupported)]), None);
        assert_eq!(
            memory_celsius(vec![Ok(memory_sample(Err(NvmlError::NotSupported)))]),
            None
        );
        assert_eq!(
            memory_celsius(vec![Ok(memory_sample(Ok(SampleValue::U32(0))))]),
            None
        );
        assert_eq!(
            memory_celsius(vec![Ok(memory_sample(Ok(SampleValue::I64(4))))]),
            None
        );
        assert_eq!(
            memory_celsius(vec![Ok(memory_sample(Ok(SampleValue::U32(127))))]),
            None
        );
        assert_eq!(
            memory_celsius(vec![Ok(memory_sample(Ok(SampleValue::I64(-3))))]),
            None
        );
        assert_eq!(
            memory_celsius(vec![Ok(memory_sample(Ok(SampleValue::F64(f64::NAN))))]),
            None
        );
        assert_eq!(
            memory_celsius(vec![Ok(memory_sample(Ok(SampleValue::F64(92.2))))]),
            Some(92.0)
        );
        assert_eq!(
            memory_celsius(vec![Ok(memory_sample(Ok(SampleValue::U32(92))))]),
            Some(92.0)
        );
    }

    #[derive(Default)]
    struct Probe {
        init_calls: AtomicUsize,
        util_calls: AtomicUsize,
        temp_calls: AtomicUsize,
        disconnects: AtomicUsize,
    }

    impl Probe {
        fn init_calls(&self) -> usize {
            self.init_calls.load(Ordering::Relaxed)
        }

        fn util_calls(&self) -> usize {
            self.util_calls.load(Ordering::Relaxed)
        }

        fn temp_calls(&self) -> usize {
            self.temp_calls.load(Ordering::Relaxed)
        }

        fn disconnects(&self) -> usize {
            self.disconnects.load(Ordering::Relaxed)
        }
    }

    struct FakeGpu {
        inits: VecDeque<Result<(), &'static str>>,
        utils: VecDeque<Result<f32, &'static str>>,
        temps: VecDeque<Result<f32, &'static str>>,
        probe: Arc<Probe>,
    }

    impl FakeGpu {
        fn pop(
            queue: &mut VecDeque<Result<f32, &'static str>>,
            what: &str,
        ) -> Result<f32, GpuError> {
            match queue.pop_front() {
                Some(Ok(value)) => Ok(value),
                Some(Err(message)) => Err(GpuError::new(message)),
                None => Err(GpuError::new(format!("unscripted {what}"))),
            }
        }
    }

    impl GpuBackend for FakeGpu {
        fn init(&mut self) -> Result<(), GpuError> {
            self.probe.init_calls.fetch_add(1, Ordering::Relaxed);
            match self.inits.pop_front() {
                Some(Ok(())) => Ok(()),
                Some(Err(message)) => Err(GpuError::new(message)),
                None => Err(GpuError::new("unscripted init")),
            }
        }

        fn util_pct(&mut self) -> Result<f32, GpuError> {
            self.probe.util_calls.fetch_add(1, Ordering::Relaxed);
            Self::pop(&mut self.utils, "util")
        }

        fn temp_c(&mut self) -> Result<f32, GpuError> {
            self.probe.temp_calls.fetch_add(1, Ordering::Relaxed);
            Self::pop(&mut self.temps, "temp")
        }

        fn disconnect(&mut self) {
            self.probe.disconnects.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn scripted(
        inits: &[Result<(), &'static str>],
        utils: &[Result<f32, &'static str>],
        temps: &[Result<f32, &'static str>],
    ) -> (GpuSource<FakeGpu>, Arc<Probe>) {
        let probe = Arc::new(Probe::default());
        let backend = FakeGpu {
            inits: inits.iter().copied().collect(),
            utils: utils.iter().copied().collect(),
            temps: temps.iter().copied().collect(),
            probe: Arc::clone(&probe),
        };
        (GpuSource::new(backend), probe)
    }

    #[test]
    fn engine_processes_merge_and_tolerate_one_unsupported_engine() {
        let p = |pid, memory_bytes| ComputeProcess { pid, memory_bytes };
        assert_eq!(
            merge_engine_processes(Ok(vec![p(1, 100)]), Ok(vec![p(1, 150), p(2, 40)])).unwrap(),
            vec![p(1, 150), p(2, 40)]
        );
        assert_eq!(
            merge_engine_processes(Err(GpuError::new("unsupported")), Ok(vec![p(2, 40)])).unwrap(),
            vec![p(2, 40)]
        );
        assert!(
            merge_engine_processes(
                Err(GpuError::new("compute")),
                Err(GpuError::new("graphics"))
            )
            .is_err()
        );
    }

    #[test]
    fn successful_read_returns_util_and_temp_and_inits_once() {
        let (mut source, probe) = scripted(&[Ok(())], &[Ok(12.0), Ok(12.0)], &[Ok(61.0), Ok(61.0)]);

        let first = source.read();
        assert_eq!(first.gpu_pct, Some(12.0));
        assert_eq!(first.gpu_c, Some(61.0));
        assert!(first.error.is_none());
        assert_eq!(probe.init_calls(), 1);
        assert_eq!(probe.disconnects(), 0);

        let second = source.read();
        assert_eq!(second.gpu_pct, Some(12.0));
        assert_eq!(second.gpu_c, Some(61.0));
        assert!(second.error.is_none());
        assert_eq!(
            probe.init_calls(),
            1,
            "init happens once while reads succeed"
        );
        assert_eq!(probe.util_calls(), 2);
        assert_eq!(probe.temp_calls(), 2);
        assert_eq!(probe.disconnects(), 0);
    }

    fn assert_gpu_error<'a>(sample: &'a GpuSample, message_part: &str) -> &'a str {
        let error = sample.error.as_ref().expect("expected a GPU source error");
        assert_eq!(error.id, SourceId::Gpu);
        assert!(
            error.message.contains(message_part),
            "error message {:?} should contain {message_part:?}",
            error.message
        );
        assert_eq!(
            error.to_string(),
            format!("gpu: {}", error.message),
            "SourceError display is tagged with the gpu source id"
        );
        error.message.as_str()
    }

    #[test]
    fn init_failure_skips_the_next_read_then_retries() {
        let (mut source, probe) =
            scripted(&[Err("driver not loaded"), Ok(())], &[Ok(4.0)], &[Ok(40.0)]);

        let failed = source.read();
        assert_eq!(failed.gpu_pct, None);
        assert_eq!(failed.gpu_c, None);
        assert_gpu_error(&failed, "driver not loaded");
        assert_eq!(probe.init_calls(), 1);
        assert_eq!(probe.util_calls(), 0);
        assert_eq!(probe.temp_calls(), 0);
        assert_eq!(probe.disconnects(), 1);

        let skipped = source.read();
        assert_eq!(skipped.gpu_pct, None);
        assert_eq!(skipped.gpu_c, None);
        assert_gpu_error(&skipped, "driver not loaded");
        assert_eq!(probe.init_calls(), 1, "init is not called on the next read");
        assert_eq!(probe.util_calls(), 0);
        assert_eq!(probe.disconnects(), 1);

        let recovered = source.read();
        assert_eq!(recovered.gpu_pct, Some(4.0));
        assert_eq!(recovered.gpu_c, Some(40.0));
        assert!(recovered.error.is_none());
        assert_eq!(
            probe.init_calls(),
            2,
            "init is called again after the backoff"
        );
        assert_eq!(probe.disconnects(), 1);
    }

    /// Reads skipped between successive init attempts.
    fn init_gaps(source: &mut GpuSource<FakeGpu>, probe: &Probe, attempts: usize) -> Vec<usize> {
        let mut gaps = Vec::new();
        let mut last_attempt: Option<usize> = None;
        let mut seen = 0;
        for tick in 0..500 {
            let before = probe.init_calls();
            let sample = source.read();
            assert_eq!(sample.gpu_pct, None);
            assert_eq!(sample.gpu_c, None);
            assert_eq!(
                sample
                    .error
                    .as_ref()
                    .expect("init failure stays visible")
                    .id,
                SourceId::Gpu
            );
            if probe.init_calls() > before {
                if let Some(prev) = last_attempt {
                    gaps.push(tick - prev - 1);
                }
                last_attempt = Some(tick);
                seen += 1;
                if seen == attempts {
                    break;
                }
            }
        }
        assert_eq!(seen, attempts, "init was not attempted often enough");
        gaps
    }

    #[test]
    fn init_backoff_doubles_after_each_failure() {
        let (mut source, probe) = scripted(&[Err("down"), Err("down"), Err("down")], &[], &[]);
        let gaps = init_gaps(&mut source, &probe, 3);
        assert_eq!(gaps, vec![1, 2]);
    }

    #[test]
    fn init_backoff_caps_at_60_reads() {
        let failures = [Err("down"); 9];
        let (mut source, probe) = scripted(&failures, &[], &[]);
        let gaps = init_gaps(&mut source, &probe, 9);
        assert_eq!(gaps, vec![1, 2, 4, 8, 16, 32, 60, 60]);
    }

    #[test]
    fn successful_init_resets_the_backoff() {
        let (mut source, probe) = scripted(
            &[Err("down"), Ok(()), Err("down"), Ok(())],
            &[Ok(1.0), Err("lost"), Ok(2.0)],
            &[Ok(10.0), Ok(11.0), Ok(12.0)],
        );

        assert!(source.read().error.is_some());
        assert_eq!(probe.init_calls(), 1);
        assert!(source.read().error.is_some());
        assert_eq!(probe.init_calls(), 1, "first failure still skips one read");
        let recovered = source.read();
        assert_eq!(recovered.gpu_pct, Some(1.0));
        assert_eq!(probe.init_calls(), 2);

        let dropped = source.read();
        assert!(dropped.error.is_some());
        assert_eq!(probe.init_calls(), 2, "a later read failure does not init");

        assert!(source.read().error.is_some());
        assert_eq!(
            probe.init_calls(),
            3,
            "the next read initialises immediately"
        );
        assert!(source.read().error.is_some());
        assert_eq!(probe.init_calls(), 3, "the reset backoff skips one read");
        let again = source.read();
        assert_eq!(again.gpu_pct, Some(2.0));
        assert_eq!(again.gpu_c, Some(12.0));
        assert!(again.error.is_none());
        assert_eq!(
            probe.init_calls(),
            4,
            "init runs again after one skipped read"
        );
    }

    #[test]
    fn read_failure_drops_the_session_and_the_next_read_inits_again() {
        let (mut source, probe) = scripted(
            &[Ok(()), Ok(())],
            &[Err("gpu lost"), Ok(9.0)],
            &[Ok(55.0), Ok(56.0)],
        );

        let failed = source.read();
        assert_eq!(failed.gpu_pct, None);
        assert_eq!(failed.gpu_c, None);
        assert_gpu_error(&failed, "gpu lost");
        assert_eq!(probe.init_calls(), 1);
        assert_eq!(probe.util_calls(), 1);
        assert_eq!(probe.temp_calls(), 1, "temperature is still read");
        assert_eq!(probe.disconnects(), 1);

        let recovered = source.read();
        assert_eq!(recovered.gpu_pct, Some(9.0));
        assert_eq!(recovered.gpu_c, Some(56.0));
        assert!(recovered.error.is_none());
        assert_eq!(
            probe.init_calls(),
            2,
            "the handle is re-initialised on the next read"
        );
        assert_eq!(probe.disconnects(), 1);
    }

    #[test]
    fn temperature_only_failure_clears_both_fields_and_drops_the_session() {
        let (mut source, probe) = scripted(
            &[Ok(()), Ok(())],
            &[Ok(33.0), Ok(34.0)],
            &[Err("sensor unavailable"), Ok(62.0)],
        );

        let failed = source.read();
        assert_eq!(failed.gpu_pct, None);
        assert_eq!(failed.gpu_c, None);
        let message = assert_gpu_error(&failed, "sensor unavailable");
        assert!(
            !message.contains("util:"),
            "a temperature-only failure should not be reported as a util failure"
        );
        assert_eq!(probe.disconnects(), 1);
        assert_eq!(probe.init_calls(), 1);

        let recovered = source.read();
        assert_eq!(recovered.gpu_pct, Some(34.0));
        assert_eq!(recovered.gpu_c, Some(62.0));
        assert!(recovered.error.is_none());
        assert_eq!(probe.init_calls(), 2);
        assert_eq!(probe.disconnects(), 1);
    }

    #[test]
    fn both_reads_failing_reports_each_reason_and_drops_the_session() {
        let (mut source, probe) =
            scripted(&[Ok(())], &[Err("gpu lost")], &[Err("sensor unavailable")]);

        let failed = source.read();
        assert_eq!(failed.gpu_pct, None);
        assert_eq!(failed.gpu_c, None);
        assert_gpu_error(&failed, "gpu lost");
        assert_gpu_error(&failed, "sensor unavailable");
        assert_eq!(probe.disconnects(), 1);
        assert_eq!(probe.util_calls(), 1);
        assert_eq!(probe.temp_calls(), 1);
    }

    #[derive(Default)]
    struct ExtraProbe {
        init_calls: AtomicUsize,
        memory_calls: AtomicUsize,
        power_calls: AtomicUsize,
        limit_calls: AtomicUsize,
    }

    struct ScriptedExtra {
        memory: Result<(u64, u64), &'static str>,
        power: Result<u32, &'static str>,
        limit: Result<u32, &'static str>,
        probe: Arc<ExtraProbe>,
    }

    impl GpuBackend for ScriptedExtra {
        fn init(&mut self) -> Result<(), GpuError> {
            self.probe.init_calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn util_pct(&mut self) -> Result<f32, GpuError> {
            Ok(8.0)
        }

        fn temp_c(&mut self) -> Result<f32, GpuError> {
            Ok(44.0)
        }

        fn memory_info(&mut self) -> Result<(u64, u64), GpuError> {
            self.probe.memory_calls.fetch_add(1, Ordering::Relaxed);
            self.memory.map_err(GpuError::new)
        }

        fn power_usage(&mut self) -> Result<u32, GpuError> {
            self.probe.power_calls.fetch_add(1, Ordering::Relaxed);
            self.power.map_err(GpuError::new)
        }

        fn enforced_power_limit(&mut self) -> Result<u32, GpuError> {
            self.probe.limit_calls.fetch_add(1, Ordering::Relaxed);
            self.limit.map_err(GpuError::new)
        }

        fn disconnect(&mut self) {}
    }

    #[test]
    fn extra_read_returns_vram_and_power() {
        let probe = Arc::new(ExtraProbe::default());
        let mut source = GpuSource::new(ScriptedExtra {
            memory: Ok((111, 222)),
            power: Ok(3_000),
            limit: Ok(4_500),
            probe: Arc::clone(&probe),
        });
        let sample = source.read();
        assert_eq!(sample.gpu_pct, Some(8.0));
        assert!(sample.error.is_none());
        let extra = source.read_extra();
        assert_eq!(extra.memory.as_ref().expect("memory"), &(111, 222));
        assert_eq!(extra.power_mw.as_ref().expect("power"), &3_000);
        assert_eq!(extra.power_limit_mw.as_ref().expect("limit"), &4_500);
        assert_eq!(probe.memory_calls.load(Ordering::Relaxed), 1);
        assert_eq!(probe.power_calls.load(Ordering::Relaxed), 1);
        assert_eq!(probe.limit_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn extra_read_error_leaves_that_field_failed_and_keeps_the_session() {
        let probe = Arc::new(ExtraProbe::default());
        let mut source = GpuSource::new(ScriptedExtra {
            memory: Err("vram lost"),
            power: Ok(10),
            limit: Err("limit unavailable"),
            probe: Arc::clone(&probe),
        });
        assert!(source.read().error.is_none());
        let extra = source.read_extra();
        assert!(
            extra
                .memory
                .as_ref()
                .expect_err("vram")
                .to_string()
                .contains("vram lost")
        );
        assert_eq!(extra.power_mw.as_ref().expect("power"), &10);
        assert!(
            extra
                .power_limit_mw
                .as_ref()
                .expect_err("limit")
                .to_string()
                .contains("limit unavailable")
        );
        let again = source.read();
        assert_eq!(again.gpu_pct, Some(8.0));
        assert_eq!(again.gpu_c, Some(44.0));
        assert!(
            again.error.is_none(),
            "a VRAM or power error does not drop util"
        );
        assert_eq!(
            probe.init_calls.load(Ordering::Relaxed),
            1,
            "a VRAM or power error does not drop the session"
        );
        let again_extra = source.read_extra();
        assert!(again_extra.memory.is_err());
        assert_eq!(probe.memory_calls.load(Ordering::Relaxed), 2);
        assert_eq!(probe.init_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn extra_read_does_not_call_the_backend_while_the_session_is_down() {
        let (mut source, probe) = scripted(&[Err("down")], &[], &[]);
        let failed = source.read();
        assert!(failed.error.is_some());
        assert_eq!(probe.init_calls(), 1);
        let extra = source.read_extra();
        assert!(
            extra
                .memory
                .as_ref()
                .expect_err("no session")
                .to_string()
                .contains("not ready")
        );
        assert!(extra.power_mw.is_err());
        assert!(extra.power_limit_mw.is_err());
        assert_eq!(
            probe.util_calls(),
            0,
            "util is not read during init backoff"
        );
    }

    #[test]
    fn uninitialised_host_backend_reports_that_and_does_not_query() {
        let mut backend = NvidiaGpu::new();
        let util = backend.util_pct().expect_err("no session yet");
        let temp = backend.temp_c().expect_err("no session yet");
        assert_eq!(util.to_string(), "GPU library is not initialised");
        assert_eq!(temp.to_string(), "GPU library is not initialised");
        assert_eq!(
            backend.memory_info().expect_err("no session").to_string(),
            "GPU library is not initialised"
        );
        assert_eq!(
            backend.power_usage().expect_err("no session").to_string(),
            "GPU library is not initialised"
        );
        assert_eq!(
            backend
                .enforced_power_limit()
                .expect_err("no session")
                .to_string(),
            "GPU library is not initialised"
        );
        assert_eq!(
            backend
                .compute_processes()
                .expect_err("no session")
                .to_string(),
            "GPU library is not initialised"
        );
        backend.disconnect();
        let after = backend
            .util_pct()
            .expect_err("disconnect leaves it unloaded");
        assert_eq!(after.to_string(), "GPU library is not initialised");
    }

    #[test]
    fn nvml_candidates_are_fixed_absolute_system_paths() {
        assert_eq!(NVML_LIB_PATHS[0], "/usr/lib64/libnvidia-ml.so.1");
        for path in NVML_LIB_PATHS {
            assert!(path.starts_with("/usr/lib"), "{path}");
            assert!(path.ends_with("/libnvidia-ml.so.1"), "{path}");
            assert!(!path.contains(".."), "{path}");
        }
    }

    #[test]
    fn load_first_stops_at_the_first_loadable_path() {
        let mut tried = Vec::new();
        let got = load_first(&["/a", "/b", "/c"], |path| {
            tried.push(path.to_owned());
            if path == "/b" {
                Ok(7)
            } else {
                Err(GpuError::new("missing"))
            }
        });
        assert_eq!(got.expect("second path"), 7);
        assert_eq!(tried, ["/a", "/b"]);
    }

    #[test]
    fn no_nvidia_library_is_one_error_naming_every_path() {
        let err = load_first::<()>(&["/a", "/b"], |_| Err(GpuError::new("missing")))
            .expect_err("no library");
        let text = err.to_string();
        assert!(text.contains("/a") && text.contains("/b"), "{text}");
    }

    struct ProcFake {
        calls: Arc<AtomicUsize>,
        script: VecDeque<Result<Vec<ComputeProcess>, &'static str>>,
        util_fail: bool,
    }

    impl GpuBackend for ProcFake {
        fn init(&mut self) -> Result<(), GpuError> {
            Ok(())
        }

        fn util_pct(&mut self) -> Result<f32, GpuError> {
            if self.util_fail {
                Err(GpuError::new("util lost"))
            } else {
                Ok(1.0)
            }
        }

        fn temp_c(&mut self) -> Result<f32, GpuError> {
            Ok(40.0)
        }

        fn compute_processes(&mut self) -> Result<Vec<ComputeProcess>, GpuError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            match self.script.pop_front() {
                Some(Ok(list)) => Ok(list),
                Some(Err(message)) => Err(GpuError::new(message)),
                None => Err(GpuError::new("unscripted processes")),
            }
        }

        fn disconnect(&mut self) {}
    }

    fn proc_source(
        script: Vec<Result<Vec<ComputeProcess>, &'static str>>,
    ) -> (GpuSource<ProcFake>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let source = GpuSource::new(ProcFake {
            calls: Arc::clone(&calls),
            script: script.into(),
            util_fail: false,
        });
        (source, calls)
    }

    fn one_proc(pid: u32, memory_bytes: u64) -> ComputeProcess {
        ComputeProcess { pid, memory_bytes }
    }

    #[test]
    fn process_read_is_at_most_once_a_second_and_an_error_clears_it() {
        let listed = vec![one_proc(4, 100), one_proc(9, 200)];
        let (mut source, calls) = proc_source(vec![
            Ok(listed.clone()),
            Err("nvml busy"),
            Ok(vec![one_proc(4, 100)]),
        ]);
        let t0 = Instant::now();
        // No session yet: no backend call.
        assert_eq!(source.read_processes(t0), ProcessRead::Unavailable);
        assert_eq!(calls.load(Ordering::Relaxed), 0);

        assert!(source.read().error.is_none());
        assert_eq!(source.read_processes(t0), ProcessRead::Ready(listed));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            source.read_processes(t0 + Duration::from_millis(999)),
            ProcessRead::Cached,
            "a tick inside the second does not call NVML"
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let at = t0 + PROCESS_EVERY;
        assert_eq!(
            source.read_processes(at),
            ProcessRead::Failed("nvml busy".to_owned())
        );
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(
            source.read_processes(at + Duration::from_millis(500)),
            ProcessRead::Cached
        );
        assert_eq!(calls.load(Ordering::Relaxed), 2);

        // The error did not drop the session: util is still the first init.
        let again = source.read();
        assert_eq!(again.gpu_pct, Some(1.0));
        assert!(again.error.is_none());
        assert_eq!(
            source.read_processes(at + PROCESS_EVERY),
            ProcessRead::Ready(vec![one_proc(4, 100)])
        );
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn a_dropped_session_forgets_the_process_interval() {
        let (mut source, calls) =
            proc_source(vec![Ok(vec![one_proc(1, 5)]), Ok(vec![one_proc(2, 6)])]);
        let t0 = Instant::now();
        assert!(source.read().error.is_none());
        assert!(matches!(source.read_processes(t0), ProcessRead::Ready(_)));
        source.backend.util_fail = true;
        let lost = source.read();
        assert!(lost.error.is_some());
        assert_eq!(
            source.read_processes(t0 + Duration::from_millis(10)),
            ProcessRead::Unavailable
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "a down session does not query"
        );

        source.backend.util_fail = false;
        assert!(source.read().error.is_none(), "the next tick inits again");
        assert_eq!(
            source.read_processes(t0 + Duration::from_millis(20)),
            ProcessRead::Ready(vec![one_proc(2, 6)]),
            "the interval does not survive the dropped session"
        );
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn a_fake_that_omits_processes_reports_an_empty_list() {
        let (mut source, _probe) = scripted(&[Ok(())], &[Ok(3.0)], &[Ok(30.0)]);
        assert!(source.read().error.is_none());
        assert_eq!(
            source.read_processes(Instant::now()),
            ProcessRead::Ready(Vec::new())
        );
    }

    #[test]
    fn unavailable_process_memory_is_dropped() {
        let kept = retain_compute(vec![
            ProcessInfo {
                pid: 1,
                used_gpu_memory: UsedGpuMemory::Unavailable,
                gpu_instance_id: None,
                compute_instance_id: None,
            },
            ProcessInfo {
                pid: 2,
                used_gpu_memory: UsedGpuMemory::Used(50),
                gpu_instance_id: None,
                compute_instance_id: None,
            },
        ]);
        assert_eq!(kept, vec![one_proc(2, 50)]);
    }

    #[test]
    fn process_names_keep_only_the_allowed_characters() {
        assert_eq!(sanitize_process_name("llama-server\n"), "llama-server");
        assert_eq!(sanitize_process_name("ok.name+v2_1-b"), "ok.name+v2_1-b");
        assert_eq!(sanitize_process_name("../weird name!"), "..weirdname");
        assert_eq!(sanitize_process_name("!!!"), "");
        assert_eq!(
            sanitize_process_name(&"a".repeat(40)).len(),
            PROCESS_NAME_MAX
        );
        assert_eq!(
            sanitize_process_name("has a space"),
            "hasaspace",
            "spaces are not in the allowed set"
        );
    }

    #[test]
    fn process_name_reads_comm_and_falls_back_to_empty() {
        let root = std::env::temp_dir().join(format!("gpu-comm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("4242")).expect("pid dir");
        std::fs::write(root.join("4242").join("comm"), "example-bench\n").expect("comm");
        assert_eq!(read_process_name(&root, 4242), "example-bench");
        assert_eq!(read_process_name(&root, 7), "", "a missing comm is empty");
        std::fs::write(root.join("4242").join("comm"), "bad/name!\n").expect("comm");
        assert_eq!(read_process_name(&root, 4242), "badname");
        std::fs::write(root.join("4242").join("comm"), "a".repeat(200)).expect("long");
        assert_eq!(read_process_name(&root, 4242).len(), PROCESS_NAME_MAX);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Manual, on a host with an NVIDIA GPU, as a normal user. Not part of `scripts/check.sh`.
    #[test]
    #[ignore = "loads the host GPU library; run only by hand"]
    fn ignored_manual_read_of_host_gpu() {
        let mut source = GpuSource::new(NvidiaGpu::new());
        let first = source.read();
        assert!(
            first.error.is_none(),
            "host read failed: {}",
            first
                .error
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default()
        );
        assert!(first.gpu_pct.is_some());
        assert!(first.gpu_c.is_some());

        let second = source.read();
        assert!(
            second.error.is_none(),
            "second host read failed: {}",
            second
                .error
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default()
        );
        assert!(second.gpu_pct.is_some());
        assert!(second.gpu_c.is_some());
    }
    struct MultiFake {
        init_count: usize,
        disconnect_count: usize,
        fail_enum: bool,
        fail_reads: bool,
    }
    impl GpuBackend for MultiFake {
        fn init(&mut self) -> Result<(), GpuError> {
            self.init_count += 1;
            Ok(())
        }
        fn util_pct(&mut self) -> Result<f32, GpuError> {
            Ok(1.0)
        }
        fn temp_c(&mut self) -> Result<f32, GpuError> {
            Ok(2.0)
        }
        fn disconnect(&mut self) {
            self.disconnect_count += 1;
        }
        fn devices(&mut self) -> Result<Vec<GpuDeviceIdentity>, GpuError> {
            if self.fail_enum {
                self.fail_enum = false;
                return Err(GpuError::new("enumeration"));
            }
            Ok(vec![
                GpuDeviceIdentity {
                    index: 0,
                    uuid: Some("zero".into()),
                    pci_bus_id: Some("0000:20:00.0".into()),
                    name: Some("first".into()),
                },
                GpuDeviceIdentity {
                    index: 1,
                    uuid: Some("one".into()),
                    pci_bus_id: Some("0000:02:00.0".into()),
                    name: Some("second".into()),
                },
            ])
        }
        fn device_reading(&mut self, index: u32, processes: bool) -> GpuDeviceReading {
            let mut reading = GpuDeviceReading::unavailable(index, "failure");
            if !self.fail_reads && index == 1 {
                reading.util_pct = Ok(25.0);
                reading.temp_c = Ok(55.0);
                reading.mem_temp_c = None;
            }
            reading.processes = if processes {
                ProcessRead::Ready(vec![ComputeProcess {
                    pid: index,
                    memory_bytes: 7,
                }])
            } else {
                ProcessRead::Cached
            };
            reading
        }
    }
    #[test]
    fn multi_device_sort_isolation_and_process_throttle() {
        let mut source = GpuSource::new(MultiFake {
            init_count: 0,
            disconnect_count: 0,
            fail_enum: false,
            fail_reads: false,
        });
        let now = Instant::now();
        let samples = source.read_devices(now).unwrap();
        assert_eq!(
            samples.iter().map(|v| v.identity.index).collect::<Vec<_>>(),
            vec![1, 0]
        );
        assert_eq!(samples[0].identity.uuid.as_deref(), Some("one"));
        assert_eq!(samples[0].reading.util_pct.as_ref().unwrap(), &25.0);
        assert!(samples[0].reading.memory.is_err());
        assert!(samples[1].reading.util_pct.is_err());
        assert!(matches!(
            samples[0].reading.processes,
            ProcessRead::Ready(_)
        ));
        let cached = source
            .read_devices(now + Duration::from_millis(500))
            .unwrap();
        assert!(
            cached
                .iter()
                .all(|v| matches!(v.reading.processes, ProcessRead::Cached))
        );
        assert!(matches!(
            source.read_devices(now + Duration::from_secs(1)).unwrap()[0]
                .reading
                .processes,
            ProcessRead::Ready(_)
        ));
        assert_eq!(source.backend.init_count, 1);
        assert_eq!(source.backend.disconnect_count, 0);
    }
    #[test]
    fn every_selected_device_failing_drops_the_session_immediately() {
        let mut source = GpuSource::new(MultiFake {
            init_count: 0,
            disconnect_count: 0,
            fail_enum: false,
            fail_reads: true,
        });
        let now = Instant::now();
        assert!(source.read_devices(now).is_err());
        assert_eq!(source.backend.disconnect_count, 1);
        source.backend.fail_reads = false;
        assert_eq!(source.read_devices(now).unwrap().len(), 2);
        assert_eq!(source.backend.init_count, 2);
    }
    #[test]
    fn every_device_open_failing_is_a_session_error() {
        let err = identities_from_count(2, |_| Err(())).unwrap_err();
        assert!(err.to_string().contains("every GPU device open failed"));
        let rows = identities_from_count(2, |index| {
            if index == 0 {
                Err(())
            } else {
                Ok(GpuDeviceIdentity {
                    index,
                    uuid: Some("u".into()),
                    pci_bus_id: None,
                    name: None,
                })
            }
        })
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows[0].uuid.is_none());
        assert_eq!(rows[1].uuid.as_deref(), Some("u"));
    }
    #[test]
    fn multi_device_enumeration_failure_disconnects_then_reconnects() {
        let mut source = GpuSource::new(MultiFake {
            init_count: 0,
            disconnect_count: 0,
            fail_enum: true,
            fail_reads: false,
        });
        let now = Instant::now();
        assert!(source.read_devices(now).is_err());
        assert_eq!(source.backend.disconnect_count, 1);
        assert_eq!(source.backend.init_count, 1);
        assert!(source.read_devices(now).is_err());
        assert_eq!(source.backend.init_count, 1);
        assert_eq!(source.read_devices(now).unwrap().len(), 2);
        assert_eq!(source.backend.init_count, 2);
    }
    #[test]
    fn legacy_backend_multi_read_keeps_init_backoff_and_defaults() {
        let (mut source, probe) = scripted(&[Err("offline"), Ok(())], &[Ok(42.0)], &[Ok(60.0)]);
        let now = Instant::now();
        assert!(source.read_devices(now).is_err());
        assert!(source.read_devices(now).is_err());
        assert_eq!(probe.init_calls(), 1);
        let rows = source.read_devices(now).unwrap();
        assert_eq!(probe.init_calls(), 2);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].identity.index, 0);
        assert_eq!(*rows[0].reading.util_pct.as_ref().unwrap(), 42.0);
        assert!(matches!(&rows[0].reading.processes, ProcessRead::Ready(p) if p.is_empty()));
    }
}
