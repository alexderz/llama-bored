//! Bottleneck activity for the LCD ring and the tty ACTIVITY bar.
//!
//! Generation is usually bound by one device, so each is normalised on its own:
//! `frac = clamp((w - idle) / (nominal_frac × limit - idle), 0, 1.25)` and
//! `activity = max(gpu_frac, cpu_frac)`. 100 % is sustained heavy load
//! (`nominal_frac` of the limit, 0.8 by default); spikes read up to the 125 % peg.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Largest device fraction: the 125 % peg.
pub(crate) const FRAC_MAX: f64 = 1.25;

/// Watts from a µJ counter step. A decrease is a `u64` wrap, not a reset.
pub(crate) fn socket_watts(prev: &[u64], now: &[u64], dt_s: f64) -> Option<f64> {
    if prev.is_empty() || prev.len() != now.len() || !(dt_s.is_finite() && dt_s > 0.0) {
        return None;
    }
    let mut microjoules: u128 = 0;
    for (before, after) in prev.iter().zip(now) {
        microjoules += u128::from(after.wrapping_sub(*before));
    }
    let watts = (microjoules as f64) / dt_s / 1_000_000.0;
    watts.is_finite().then_some(watts)
}

/// One device's share of its nominal headroom, 0..=[`FRAC_MAX`].
///
/// The headroom runs from `idle_w` to `nominal_frac × limit_w`. `None` when a
/// number is non-finite or that ceiling is not above `idle_w`.
pub(crate) fn device_frac(watts: f64, idle_w: f64, limit_w: f64, nominal_frac: f64) -> Option<f64> {
    let headroom = nominal_frac * limit_w - idle_w;
    if !watts.is_finite() || !idle_w.is_finite() || !headroom.is_finite() || headroom <= 0.0 {
        return None;
    }
    Some(((watts - idle_w) / headroom).clamp(0.0, FRAC_MAX))
}

/// Which device set the activity on this tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Winner {
    Gpu,
    Cpu,
}

/// `max(gpu, cpu)` as a percent with the winner. The GPU wins a tie.
/// One missing fraction leaves the other; both missing is `None`.
pub(crate) fn bottleneck(gpu: Option<f64>, cpu: Option<f64>) -> Option<(f64, Winner)> {
    let (frac, winner) = match (gpu, cpu) {
        (Some(gpu), Some(cpu)) if cpu > gpu => (cpu, Winner::Cpu),
        (Some(gpu), _) => (gpu, Winner::Gpu),
        (None, Some(cpu)) => (cpu, Winner::Cpu),
        (None, None) => return None,
    };
    Some((frac * 100.0, winner))
}

/// Quiet time after start, or after a source (re)appears, before the auto
/// floor learns. The first zenergy delta spans a partial interval and reads low.
pub(crate) const IDLE_SETTLE: Duration = Duration::from_secs(30);

/// Span of the rolling mean the auto floor learns from. Raw samples never count.
pub(crate) const IDLE_MEAN_WINDOW: Duration = Duration::from_secs(10);

/// Share of the limit the learned floor must stay below `nominal_frac × limit`.
pub(crate) const IDLE_CLAMP_FRAC: f64 = 0.1;

/// A device's idle floor.
///
/// Auto: the configured watts until the first valid [`IDLE_MEAN_WINDOW`]
/// mean, then the lowest such mean seen since start, which may sit above the
/// configured watts. Reads in the [`IDLE_SETTLE`] after start or after the
/// source reappears are ignored. The floor never rises, so sustained load
/// cannot be learned as idle. It is capped at
/// `(nominal_frac - IDLE_CLAMP_FRAC) × limit`, so a box loaded from boot
/// cannot learn a floor that leaves no headroom; such a box wants `fixed`.
///
/// Fixed: the configured watts.
#[derive(Clone, Debug)]
pub(crate) struct IdleFloor {
    configured_w: f64,
    learn: bool,
    nominal_frac: f64,
    /// When the source (re)appeared. `None` while it is absent.
    seen_since: Option<Instant>,
    /// Settled reads, oldest first, and when the unbroken run of them began.
    window: VecDeque<(Instant, f64)>,
    window_start: Option<Instant>,
    lowest_mean_w: Option<f64>,
}

impl IdleFloor {
    pub(crate) fn new(configured_w: f64, learn: bool, nominal_frac: f64) -> Self {
        Self {
            configured_w,
            learn,
            nominal_frac,
            seen_since: None,
            window: VecDeque::new(),
            window_start: None,
            lowest_mean_w: None,
        }
    }

    /// Record one tick. `None` or a non-finite draw means the source is absent,
    /// which restarts the settle time. Negatives count as 0.
    pub(crate) fn observe(&mut self, mono: Instant, watts: Option<f64>) {
        if !self.learn {
            return;
        }
        let Some(watts) = watts.filter(|w| w.is_finite()) else {
            self.seen_since = None;
            self.window.clear();
            self.window_start = None;
            return;
        };
        let since = *self.seen_since.get_or_insert(mono);
        if mono.saturating_duration_since(since) < IDLE_SETTLE {
            return;
        }
        if self.window.back().is_some_and(|(then, _)| mono < *then) {
            return;
        }
        let start = *self.window_start.get_or_insert(mono);
        self.window.push_back((mono, watts.max(0.0)));
        while self
            .window
            .front()
            .is_some_and(|(then, _)| mono.saturating_duration_since(*then) >= IDLE_MEAN_WINDOW)
        {
            self.window.pop_front();
        }
        if mono.saturating_duration_since(start) < IDLE_MEAN_WINDOW || self.window.is_empty() {
            return;
        }
        let mean = self.window.iter().map(|(_, w)| w).sum::<f64>() / self.window.len() as f64;
        if mean.is_finite() {
            let lowest = self.lowest_mean_w.map_or(mean, |low| low.min(mean));
            self.lowest_mean_w = Some(lowest);
        }
    }

    /// Floor in watts against this device's `limit_w`.
    pub(crate) fn watts(&self, limit_w: f64) -> f64 {
        if !self.learn {
            return self.configured_w;
        }
        let floor = self.lowest_mean_w.unwrap_or(self.configured_w);
        let cap = (self.nominal_frac - IDLE_CLAMP_FRAC) * limit_w;
        if cap.is_finite() {
            floor.min(cap.max(0.0))
        } else {
            floor
        }
    }
}

/// One EMA step. `tau_s == 0` tracks `sample`. A missing previous value is `sample`.
pub(crate) fn ema_step(prev: Option<f64>, sample: f64, dt_s: f64, tau_s: f64) -> f64 {
    let Some(prev) = prev.filter(|value| value.is_finite()) else {
        return sample;
    };
    if !sample.is_finite() {
        return prev;
    }
    let alpha = if tau_s <= 0.0 || !tau_s.is_finite() {
        1.0
    } else if !dt_s.is_finite() || dt_s <= 0.0 {
        0.0
    } else {
        1.0 - (-dt_s / tau_s).exp()
    };
    let next = alpha * sample + (1.0 - alpha) * prev;
    if next.is_finite() { next } else { sample }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_frac_normalises_each_device_on_its_own_headroom() {
        let gpu = device_frac(340.0, 20.0, 350.0, 1.0).expect("gpu");
        assert!((gpu - 320.0 / 330.0).abs() < 1e-12, "{gpu}");
        assert_eq!(device_frac(10.0, 20.0, 350.0, 1.0), Some(0.0));
        assert_eq!(device_frac(500.0, 20.0, 350.0, 1.0), Some(FRAC_MAX));
        assert_eq!(device_frac(100.0, 350.0, 350.0, 1.0), None);
        assert_eq!(device_frac(f64::NAN, 20.0, 350.0, 1.0), None);
    }

    #[test]
    fn device_frac_is_taken_on_the_nominal_ceiling_and_pins_at_125() {
        // 0.8 × 350 = 280 W reads 1.0; 310 W reads 1.12; 350 W pins at 1.25.
        assert_eq!(device_frac(280.0, 30.0, 350.0, 0.8), Some(1.0));
        let spike = device_frac(310.0, 30.0, 350.0, 0.8).expect("spike");
        assert!((spike - 1.12).abs() < 1e-12, "{spike}");
        assert_eq!(device_frac(350.0, 30.0, 350.0, 0.8), Some(FRAC_MAX));
        assert_eq!(device_frac(30.0, 30.0, 350.0, 0.8), Some(0.0));
        assert_eq!(device_frac(100.0, 280.0, 350.0, 0.8), None, "no headroom");
        let (pct, _) = bottleneck(Some(FRAC_MAX), Some(0.1)).expect("peg");
        assert_eq!(pct, 125.0);
    }

    #[test]
    fn bottleneck_takes_the_busier_device_and_falls_through_to_either() {
        assert_eq!(
            bottleneck(Some(0.97), Some(0.05)),
            Some((97.0, Winner::Gpu))
        );
        assert_eq!(bottleneck(Some(0.0), Some(1.0)), Some((100.0, Winner::Cpu)));
        assert_eq!(bottleneck(Some(0.5), Some(0.5)), Some((50.0, Winner::Gpu)));
        assert_eq!(bottleneck(None, Some(0.25)), Some((25.0, Winner::Cpu)));
        assert_eq!(bottleneck(Some(0.25), None), Some((25.0, Winner::Gpu)));
        assert_eq!(bottleneck(None, None), None);
    }

    /// Feeds `floor` one read per `step` from `from` for `secs`, all `watts`.
    fn feed(floor: &mut IdleFloor, t0: Instant, from: u64, secs: u64, watts: f64) {
        for s in from..from + secs {
            floor.observe(t0 + Duration::from_secs(s), Some(watts));
        }
    }

    #[test]
    fn a_glitch_low_first_sample_does_not_poison_the_floor() {
        // The first zenergy delta after a restart reads 5 W; the box idles at 37 W.
        let t0 = Instant::now();
        let mut floor = IdleFloor::new(25.0, true, 0.8);
        floor.observe(t0, Some(5.0));
        feed(&mut floor, t0, 1, 120, 37.0);
        assert!(
            (floor.watts(230.0) - 37.0).abs() < 1e-9,
            "{}",
            floor.watts(230.0)
        );
        let frac = device_frac(37.0, floor.watts(230.0), 230.0, 0.8).expect("frac");
        assert!(frac < 0.005, "37 W idle reads {frac}");
    }

    #[test]
    fn a_glitch_after_the_source_reappears_is_ignored_too() {
        let t0 = Instant::now();
        let mut floor = IdleFloor::new(25.0, true, 0.8);
        feed(&mut floor, t0, 0, 60, 41.0);
        floor.observe(t0 + Duration::from_secs(60), None);
        floor.observe(t0 + Duration::from_secs(61), Some(3.0));
        feed(&mut floor, t0, 62, 60, 41.0);
        assert!(
            (floor.watts(230.0) - 41.0).abs() < 1e-9,
            "{}",
            floor.watts(230.0)
        );
    }

    #[test]
    fn idle_above_the_default_is_learned_and_reads_zero() {
        let t0 = Instant::now();
        let mut floor = IdleFloor::new(25.0, true, 0.8);
        feed(&mut floor, t0, 0, 30, 41.0);
        assert_eq!(
            floor.watts(230.0),
            25.0,
            "the default seeds the settle time"
        );
        feed(&mut floor, t0, 30, 60, 41.0);
        assert!(
            (floor.watts(230.0) - 41.0).abs() < 1e-9,
            "{}",
            floor.watts(230.0)
        );
        let frac = device_frac(41.0, floor.watts(230.0), 230.0, 0.8).expect("frac");
        assert!(frac < 0.005, "41 W idle reads {frac}");
    }

    #[test]
    fn thirty_minutes_of_load_after_an_idle_start_keeps_the_idle_floor() {
        let t0 = Instant::now();
        let mut floor = IdleFloor::new(25.0, true, 0.8);
        feed(&mut floor, t0, 0, 60, 41.0);
        feed(&mut floor, t0, 60, 30 * 60, 200.0);
        floor.observe(t0 + Duration::from_secs(60 + 30 * 60), Some(f64::NAN));
        feed(&mut floor, t0, 61 + 30 * 60, 20, 200.0);
        assert!(
            (floor.watts(230.0) - 41.0).abs() < 1e-9,
            "{}",
            floor.watts(230.0)
        );
    }

    #[test]
    fn a_box_loaded_from_boot_is_capped_by_the_sanity_clamp() {
        // 200 W from boot on a 230 W limit: cap is (0.8 - 0.1) × 230 = 161 W.
        let t0 = Instant::now();
        let mut floor = IdleFloor::new(25.0, true, 0.8);
        feed(&mut floor, t0, 0, 120, 200.0);
        let cap = (0.8 - IDLE_CLAMP_FRAC) * 230.0;
        assert!(
            (floor.watts(230.0) - cap).abs() < 1e-9,
            "{}",
            floor.watts(230.0)
        );
        assert!(device_frac(230.0, floor.watts(230.0), 230.0, 0.8).is_some());
    }

    #[test]
    fn fixed_mode_ignores_learning() {
        let t0 = Instant::now();
        let mut fixed = IdleFloor::new(40.0, false, 0.8);
        feed(&mut fixed, t0, 0, 120, 10.0);
        assert_eq!(fixed.watts(230.0), 40.0);
        let mut high = IdleFloor::new(40.0, false, 0.8);
        feed(&mut high, t0, 0, 120, 300.0);
        assert_eq!(high.watts(230.0), 40.0, "no clamp in fixed mode");
    }
}
