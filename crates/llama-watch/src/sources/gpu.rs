//! GPU utilisation and temperature.
//!
//! [`GpuSource`] owns a [`GpuBackend`]. It initialises the backend once.
//! An init failure waits 1, 2, 4, … reads (capped at 60) before trying again,
//! and a successful init resets that wait. Any later read error drops the
//! session so the next read initialises immediately, and clears both fields.
//! [`NvidiaGpu`] loads the host library; unit tests use a fake and do not.

use super::{SourceError, SourceId};
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::{Device, Nvml};
use std::ffi::OsStr;

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

    /// Drop the live library handle. Safe to call when nothing is loaded.
    fn disconnect(&mut self);
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
        }
    }

    /// Read utilisation and temperature for one tick.
    ///
    /// Initialises the backend on the first call. After an init failure, later
    /// attempts wait 1, then 2, 4, … reads, capped at 60. A successful init
    /// resets that sequence to one read.
    pub fn read(&mut self) -> GpuSample {
        if !self.ready {
            if self.init_wait > 0 {
                self.init_wait -= 1;
                return self.init_backoff_sample();
            }
            if let Err(err) = self.backend.init() {
                // A failed init may still have kept a handle. Drop it.
                self.backend.disconnect();
                self.init_error = format!("init: {err}");
                self.init_wait = self.init_backoff;
                self.init_backoff = self.init_backoff.saturating_mul(2).min(INIT_BACKOFF_MAX);
                return self.init_backoff_sample();
            }
            self.ready = true;
            self.init_backoff = 1;
            self.init_wait = 0;
            self.init_error.clear();
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

    fn disconnect(&mut self) {
        self.library = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
}
