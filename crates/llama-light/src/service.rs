//! The llama-light loop: snapshot → frame → devices, with staleness,
//! presence and config reload. Every seam is a trait so tests run it with
//! fakes and a manual clock.
//!
//! Stop: production relies on the default SIGTERM action. systemd treats
//! death by SIGTERM as a clean stop, and the unit's `ExecStopPost` runs
//! `llama-light restore`, which leaves the header on a neutral static
//! colour and hands the keyboard back to its own lighting (see [`restore`]).

use std::time::Duration;

use llama_core::color::Rgb;
use llama_core::log::{self, Priority, Sink};
use llama_core::wire::SnapshotV1;
use sd_notify::NotifyState;

use crate::backend::Backend;
use crate::config::{ConfigSource, LightConfig, Stamp};
use crate::mapping::{Frames, Renderer, fade, keyboard_neutral_frame, neutral_frame};
use crate::snapshot::SnapshotSource;

/// A snapshot older than this is stale: the last frame is held.
pub const HOLD_AFTER: Duration = Duration::from_secs(5);
/// Stale this long: fade to the dim neutral.
pub const FADE_AFTER: Duration = Duration::from_secs(30);
/// The fade itself.
pub const FADE_TIME: Duration = Duration::from_secs(5);
/// An absent device is looked for again this often, silently.
pub const RESCAN: Duration = Duration::from_secs(10);
/// The config file is stat'ed this often.
pub const CONFIG_CHECK: Duration = Duration::from_secs(2);
/// `t_mono_ns` may lead the local clock by this much.
pub const FUTURE_SLACK: Duration = Duration::from_millis(50);

/// `CLOCK_MONOTONIC` and sleep. Tests use a manual clock.
pub trait Clock {
    /// Nanoseconds on the clock the watcher stamps `t_mono_ns` with.
    fn now_ns(&self) -> u64;
    /// Sleep `d`.
    fn sleep(&mut self, d: Duration);
}

/// Host `CLOCK_MONOTONIC`.
pub struct HostClock;

impl Clock for HostClock {
    fn now_ns(&self) -> u64 {
        let ts = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
        let sec = u64::try_from(ts.tv_sec).unwrap_or(0);
        let nsec = u64::try_from(ts.tv_nsec).unwrap_or(0);
        sec.saturating_mul(1_000_000_000).saturating_add(nsec)
    }

    fn sleep(&mut self, d: Duration) {
        std::thread::sleep(d);
    }
}

/// sd-notify seam.
pub trait Notifier {
    /// `READY=1`.
    fn ready(&mut self);
    /// `WATCHDOG=1`.
    fn watchdog(&mut self);
}

/// `sd_notify`. A no-op without `NOTIFY_SOCKET`.
pub struct SdNotify;

impl Notifier for SdNotify {
    fn ready(&mut self) {
        let _ = sd_notify::notify(&[NotifyState::Ready]);
    }

    fn watchdog(&mut self) {
        let _ = sd_notify::notify(&[NotifyState::Watchdog]);
    }
}

/// What the stale policy says to do this tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    /// Fresh: draw the snapshot.
    Live,
    /// Stale up to [`FADE_AFTER`]: keep the last frame, send nothing new.
    Hold,
    /// Stale past [`FADE_AFTER`]: this far (0..=1) toward the neutral.
    Fade(f32),
}

/// The stale policy. `age` is how old the last accepted snapshot is (or,
/// before the first one, how long the service has run).
#[must_use]
pub fn phase(age: Duration) -> Phase {
    if age <= HOLD_AFTER {
        Phase::Live
    } else if age <= FADE_AFTER {
        Phase::Hold
    } else {
        let into = (age - FADE_AFTER).as_secs_f32() / FADE_TIME.as_secs_f32();
        Phase::Fade(into.min(1.0))
    }
}

/// One line when a device goes absent, one when it returns. Never more.
pub struct Presence {
    name: &'static str,
    present: Option<bool>,
}

impl Presence {
    /// A device not yet seen either way.
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            present: None,
        }
    }

    /// Record the state; log only on a change.
    pub fn update(&mut self, present: bool, reason: &str, sink: &mut impl Sink) {
        if self.present == Some(present) {
            return;
        }
        self.present = Some(present);
        if present {
            log::emit(sink, Priority::Info, &format!("{} present", self.name));
        } else {
            log::emit(
                sink,
                Priority::Warning,
                &format!(
                    "{} absent ({reason}); looking again every {} s without further logs",
                    self.name,
                    RESCAN.as_secs()
                ),
            );
        }
    }
}

struct Device {
    backend: Box<dyn Backend>,
    presence: Presence,
    last_scan: Option<u64>,
    /// When a frame was last written.
    last_sent: Option<u64>,
}

impl Device {
    fn new(backend: Box<dyn Backend>) -> Self {
        let name = backend.name();
        Self {
            backend,
            presence: Presence::new(name),
            last_scan: None,
            last_sent: None,
        }
    }

    /// Whether a frame may be written now under a cap of `fps` frames per
    /// second, when the loop runs at `tick_hz`. A cap at or above the loop
    /// rate never holds a frame back.
    fn may_send(&self, now: u64, fps: u8, tick_hz: u8) -> bool {
        if fps >= tick_hz {
            return true;
        }
        let period = 1_000_000_000 / u64::from(fps.max(1));
        self.last_sent
            .is_none_or(|last| now.saturating_sub(last) + SEND_SLACK_NS >= period)
    }

    fn due(&self, now: u64) -> bool {
        self.last_scan
            .is_none_or(|last| now.saturating_sub(last) >= nanos(RESCAN))
    }

    /// Probe when closed and due. `true` when open afterwards.
    fn ensure_open(&mut self, now: u64, sink: &mut impl Sink) -> bool {
        if self.backend.is_open() {
            return true;
        }
        if !self.due(now) {
            return false;
        }
        self.last_scan = Some(now);
        match self.backend.probe() {
            Ok(()) => {
                self.presence.update(true, "", sink);
                true
            }
            Err(reason) => {
                self.presence.update(false, &reason, sink);
                false
            }
        }
    }

    fn show(&mut self, frame: &[Rgb], now: u64, sink: &mut impl Sink) -> bool {
        match self.backend.show(frame) {
            Ok(sent) => {
                if sent {
                    self.last_sent = Some(now);
                }
                sent
            }
            Err(reason) => {
                self.last_scan = Some(now);
                self.presence.update(false, &reason, sink);
                false
            }
        }
    }
}

/// Clock jitter allowed when checking a device's frame cap.
const SEND_SLACK_NS: u64 = 1_000_000;

fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// The running service.
pub struct Light<C, S, F, N, L> {
    clock: C,
    snapshots: S,
    config_source: F,
    notify: N,
    sink: L,
    renderer: Renderer,
    aura: Option<Device>,
    keyboard: Option<Device>,
    started: u64,
    last_tick: Option<u64>,
    last_config_check: u64,
    stamp: Option<Stamp>,
    /// The stamp of the last rejected file (`Some(None)`: the file was gone).
    rejected: Option<Option<Stamp>>,
    latest: Option<SnapshotV1>,
    snapshot_label: Option<&'static str>,
    live_frame: Option<Frames>,
    frames_sent: u64,
    keyboard_frames_sent: u64,
    restart: bool,
}

/// Why [`Light::run`] returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunEnd {
    /// The tick budget ran out (tests).
    Ticks,
    /// A device is present that only a restart of the unit can open.
    /// `main` exits non-zero so systemd restarts the unit.
    Restart,
}

/// Everything [`Light::new`] needs.
pub struct Parts<C, S, F, N, L> {
    pub clock: C,
    pub snapshots: S,
    pub config_source: F,
    pub notify: N,
    pub sink: L,
    pub config: LightConfig,
    pub aura: Option<Box<dyn Backend>>,
    pub keyboard: Option<Box<dyn Backend>>,
}

impl<C, S, F, N, L> Light<C, S, F, N, L>
where
    C: Clock,
    S: SnapshotSource,
    F: ConfigSource,
    N: Notifier,
    L: Sink,
{
    /// Build the service. `config` is the one already loaded at start.
    pub fn new(parts: Parts<C, S, F, N, L>) -> Self {
        let now = parts.clock.now_ns();
        let stamp = parts.config_source.stamp();
        Self {
            clock: parts.clock,
            snapshots: parts.snapshots,
            config_source: parts.config_source,
            notify: parts.notify,
            sink: parts.sink,
            renderer: Renderer::new(parts.config),
            aura: parts.aura.map(Device::new),
            keyboard: parts.keyboard.map(Device::new),
            started: now,
            last_tick: None,
            last_config_check: now,
            stamp,
            rejected: None,
            latest: None,
            snapshot_label: None,
            live_frame: None,
            frames_sent: 0,
            keyboard_frames_sent: 0,
            restart: false,
        }
    }

    /// Frames written to the keyboard so far.
    #[must_use]
    pub fn keyboard_frames_sent(&self) -> u64 {
        self.keyboard_frames_sent
    }

    /// Frames written to the Aura so far.
    #[must_use]
    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    /// The running config.
    #[must_use]
    pub fn config(&self) -> &LightConfig {
        self.renderer.config()
    }

    /// The log sink (tests read captured lines through it).
    pub fn sink(&mut self) -> &mut L {
        &mut self.sink
    }

    /// The clock (tests move a manual clock through it).
    pub fn clock(&mut self) -> &mut C {
        &mut self.clock
    }

    /// Run forever, `max_ticks` ticks, or until a restart is needed.
    pub fn run(&mut self, max_ticks: Option<u64>) -> RunEnd {
        self.notify.ready();
        let mut ticks = 0_u64;
        loop {
            if max_ticks.is_some_and(|max| ticks >= max) {
                return RunEnd::Ticks;
            }
            self.tick();
            if self.restart {
                return RunEnd::Restart;
            }
            ticks += 1;
            let fps = u64::from(self.config().engine.tick_hz.max(1));
            self.clock.sleep(Duration::from_nanos(1_000_000_000 / fps));
        }
    }

    /// One tick: config check, snapshot, frame, devices, watchdog.
    pub fn tick(&mut self) {
        let now = self.clock.now_ns();
        let dt_s = self
            .last_tick
            .map_or(0.0, |last| now.saturating_sub(last) as f32 / 1e9);
        self.last_tick = Some(now);
        if now.saturating_sub(self.last_config_check) >= nanos(CONFIG_CHECK) {
            self.last_config_check = now;
            self.check_config();
        }
        self.read_snapshot(now);
        let frames = self.frames(now, dt_s);
        let aura_enabled = self.config().aura.enabled;
        let tick_hz = self.config().engine.tick_hz;
        let aura_fps = self.config().aura.fps;
        let keyboard_fps = self.config().engine.tween_fps;
        if let Some(device) = self.aura.as_mut()
            && aura_enabled
            && device.ensure_open(now, &mut self.sink)
            && let Some(frames) = &frames
            && device.may_send(now, aura_fps, tick_hz)
            && device.show(&frames.aura, now, &mut self.sink)
        {
            self.frames_sent += 1;
        }
        // The keyboard is independent of the fans: absent, refused or
        // failing, it never stops the Aura frame above.
        let keyboard_enabled = self.config().keyboard.enabled;
        if let Some(device) = self.keyboard.as_mut()
            && keyboard_enabled
        {
            if device.ensure_open(now, &mut self.sink) {
                if let Some(frames) = &frames
                    && device.may_send(now, keyboard_fps, tick_hz)
                    && device.show(&frames.keyboard, now, &mut self.sink)
                {
                    self.keyboard_frames_sent += 1;
                }
            } else if device.backend.wants_restart() && !self.restart {
                self.restart = true;
                log::emit(
                    &mut self.sink,
                    Priority::Warning,
                    "keyboard: attached after the unit started, so the unit's device list does not include it; exiting so systemd restarts llama-light",
                );
            }
        }
        self.notify.watchdog();
    }

    fn check_config(&mut self) {
        let stamp = self.config_source.stamp();
        if stamp == self.stamp || self.rejected == Some(stamp) {
            return;
        }
        match self.config_source.load() {
            Ok(config) => {
                self.stamp = stamp;
                self.rejected = None;
                let was_enabled = self.config().aura.enabled;
                let keyboard_was_enabled = self.config().keyboard.enabled;
                let neutral = neutral_frame(&self.config().aura);
                self.renderer = Renderer::new(config);
                self.live_frame = None;
                log::emit(&mut self.sink, Priority::Info, "config reloaded");
                if was_enabled
                    && !self.config().aura.enabled
                    && let Some(device) = self.aura.as_mut()
                    && device.backend.is_open()
                {
                    let now = self.clock.now_ns();
                    device.show(&neutral, now, &mut self.sink);
                }
                if keyboard_was_enabled
                    && !self.config().keyboard.enabled
                    && let Some(device) = self.keyboard.as_mut()
                    && device.backend.is_open()
                    && let Err(reason) = device.backend.release()
                {
                    log::emit(
                        &mut self.sink,
                        Priority::Warning,
                        &format!("keyboard: hand-back to its own lighting failed ({reason})"),
                    );
                }
            }
            Err(err) => {
                self.rejected = Some(stamp);
                log::emit(
                    &mut self.sink,
                    Priority::Err,
                    &format!("config reload rejected, keeping the running config: {err}"),
                );
            }
        }
    }

    fn read_snapshot(&mut self, now: u64) {
        let label = match self.snapshots.read() {
            Ok(snapshot) if snapshot.t_mono_ns > now.saturating_add(nanos(FUTURE_SLACK)) => {
                "snapshot from the future"
            }
            Ok(snapshot) => {
                let fresh = now.saturating_sub(snapshot.t_mono_ns) <= nanos(HOLD_AFTER);
                self.latest = Some(snapshot);
                if fresh {
                    "snapshot fresh"
                } else {
                    "snapshot stale"
                }
            }
            Err(label) => label,
        };
        if self.snapshot_label != Some(label) {
            self.snapshot_label = Some(label);
            let priority = if label == "snapshot fresh" {
                Priority::Info
            } else {
                Priority::Warning
            };
            log::emit(&mut self.sink, priority, label);
        }
    }

    /// The frames to show now, or `None` to send nothing.
    fn frames(&mut self, now: u64, dt_s: f32) -> Option<Frames> {
        let age_ns = match &self.latest {
            Some(snapshot) => now.saturating_sub(snapshot.t_mono_ns),
            None => now.saturating_sub(self.started).max(nanos(HOLD_AFTER) + 1),
        };
        let neutral = Frames {
            aura: neutral_frame(&self.config().aura),
            keyboard: keyboard_neutral_frame(&self.config().keyboard),
        };
        match phase(Duration::from_nanos(age_ns)) {
            Phase::Live => {
                let snapshot = self.latest.as_ref()?;
                let frames = self.renderer.frames(snapshot, dt_s);
                self.live_frame = Some(frames.clone());
                Some(frames)
            }
            Phase::Hold => self.live_frame.clone(),
            Phase::Fade(amount) => {
                let from = self.live_frame.clone().unwrap_or_else(|| neutral.clone());
                Some(Frames {
                    aura: fade(&from.aura, &neutral.aura, amount),
                    keyboard: fade(&from.keyboard, &neutral.keyboard, amount),
                })
            }
        }
    }
}

/// `ExecStopPost`: leave header 1 on [`crate::mapping::NEUTRAL`] (after the
/// brightness cap), in Direct mode, RAM only; then hand the keyboard back
/// to its own lighting.
///
/// Aura: the protocol has no documented command that re-loads the board's
/// stored effect without a power cycle, and the effect commands that do
/// exist would either overwrite what is stored (with a commit) or guess at
/// it. So this writes a neutral static frame and never commits: the stored
/// effect comes back at the next power cycle, as observed on a test board.
///
/// Keyboard: the legacy protocol has a RAM-only switch back to hardware
/// lighting (`07 05 01`), so the keyboard shows its own stored lighting
/// again, and keeps showing it while llama-light is stopped. `keyboard` is
/// `None` when the config leaves the keyboard off.
///
/// An absent device is one log line and success. Exit status 1 if a write
/// to a present device failed.
pub fn restore(
    config: &LightConfig,
    aura: &mut dyn Backend,
    keyboard: Option<&mut dyn Backend>,
    sink: &mut impl Sink,
) -> u8 {
    let mut status = restore_aura(config, aura, sink);
    if let Some(keyboard) = keyboard {
        match keyboard.release() {
            Ok(true) => log::emit(
                sink,
                Priority::Info,
                "restore: keyboard handed back to its own (hardware) lighting",
            ),
            Ok(false) => log::emit(
                sink,
                Priority::Info,
                "restore: keyboard absent; nothing sent",
            ),
            Err(reason) => {
                log::emit(
                    sink,
                    Priority::Err,
                    &format!("restore: keyboard write failed ({reason})"),
                );
                status = 1;
            }
        }
    }
    status
}

fn restore_aura(config: &LightConfig, aura: &mut dyn Backend, sink: &mut impl Sink) -> u8 {
    if !config.aura.enabled {
        log::emit(
            sink,
            Priority::Info,
            "restore: aura disabled in config; nothing sent",
        );
        return 0;
    }
    if let Err(reason) = aura.probe() {
        log::emit(
            sink,
            Priority::Info,
            &format!("restore: aura absent ({reason}); nothing sent"),
        );
        return 0;
    }
    match aura.show(&neutral_frame(&config.aura)) {
        Ok(_) => {
            log::emit(
                sink,
                Priority::Info,
                "restore: header 1 set to the neutral static colour",
            );
            0
        }
        Err(reason) => {
            log::emit(
                sink,
                Priority::Err,
                &format!("restore: aura write failed ({reason})"),
            );
            1
        }
    }
}
