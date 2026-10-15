//! A span as a heated metal bar.
//!
//! One thousand cells, in percent of the entry's range (`f64`). The first
//! LED is the bottom, held by a sink at 0; the last LED is the top, where
//! the heater sits. Above the foot the cells shift toward the bottom at a
//! constant speed (whole cells, the fraction carried over, no smearing);
//! what reaches the foot is added to the foot's top cell. One backward-Euler
//! conduction step then carries that heat through the foot into the sink.
//! The heater follows its input with [`crate::mapping::ema`]'s exponential,
//! in `f64`: `attack_s` while the input is above the heater, `smooth_s`
//! while it is not, and a time constant of 0 jumps. `ema` is that same step
//! in `f32`, so the bar does not round the heater through it.
//!
//! A span of one LED cannot hold a foot (`foot_leds` is at least 1 and must
//! be fewer than the span), so [`HeatBar::new`] returns an error. Two LEDs
//! work with `foot_leds = 1`. The config check reports the same cases; the
//! renderer skips a span it cannot build.

/// Cells along the bar.
pub const CELLS: usize = 1000;

/// A gap longer than this is cut, so a resume from suspend cannot spin.
pub const MAX_GAP_S: f64 = 60.0;

/// Longest sub-step. A larger `dt` is split so the shift and the conduction
/// stay a small step.
pub const MAX_SUBSTEP_S: f64 = 0.5;

/// Shortest `conduction_s`, in seconds. The config accepts this same floor.
/// A smaller diffusion time makes the implicit step overflow; below about
/// 1e-11 s the cells stop being finite and stay that way.
const MIN_CONDUCTION_S: f64 = 1.0;

/// The bar's numbers. Times are seconds; `foot_leds` is a count of LEDs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeatParams {
    /// Seconds for heat to travel from the top of the bar to the top of the foot.
    pub transit_s: f64,
    /// LEDs at the bottom that conduct into the sink.
    pub foot_leds: usize,
    /// Diffusion time of the bar above the foot (`L²/D`).
    pub conduction_s: f64,
    /// Heater time constant while the input is above the heater.
    pub attack_s: f64,
    /// Heater time constant while the input is not above the heater.
    pub smooth_s: f64,
}

impl Default for HeatParams {
    fn default() -> Self {
        Self {
            transit_s: 60.0,
            foot_leds: 5,
            conduction_s: 20_000.0,
            attack_s: 0.0,
            smooth_s: 0.0,
        }
    }
}

/// Why a bar cannot be built. The config check reports the same limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeatError {
    /// `foot_leds` must be at least 1 and fewer than `leds`, and `leds`
    /// must be at least 2.
    Span { leds: usize, foot_leds: usize },
    /// `transit_s` is not a finite number above 0, `conduction_s` is not a
    /// finite number of at least 1 second, or a heater time constant is
    /// negative or non-finite. Below 1 s the conduction step can overflow.
    Time,
}

/// One span's bar: cells, heater temperature, and the shift remainder.
#[derive(Clone, Debug)]
pub struct HeatBar {
    led_count: usize,
    /// Cell temperatures, bottom (sink end) first. Index 0 is next to the sink.
    cells: Vec<f64>,
    heater: f64,
    /// Cells of shift not yet applied, in `[0, 1)` except while a step adds.
    shift: f64,
    /// First cell above the foot.
    ie: usize,
    /// Flow speed, bar lengths per second.
    speed: f64,
    dx: f64,
    /// Conductivity of each cell.
    kc: Vec<f64>,
    /// Conductivity of the face above cell `i` (harmonic mean), length `CELLS - 1`.
    kf: Vec<f64>,
    /// Tent weights, one list per LED: `(cell, normalised weight)`.
    weights: Vec<Vec<(usize, f64)>>,
    attack_s: f64,
    smooth_s: f64,
    diag: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    rhs: Vec<f64>,
    cp: Vec<f64>,
    dp: Vec<f64>,
}

impl HeatBar {
    /// A cold bar of `leds` LEDs. `leds` of 1 is [`HeatError::Span`]. Two
    /// LEDs need `params.foot_leds == 1`.
    pub fn new(leds: usize, params: HeatParams) -> Result<Self, HeatError> {
        if !params.transit_s.is_finite()
            || params.transit_s <= 0.0
            || !params.conduction_s.is_finite()
            || params.conduction_s < MIN_CONDUCTION_S
            || !params.attack_s.is_finite()
            || params.attack_s < 0.0
            || !params.smooth_s.is_finite()
            || params.smooth_s < 0.0
        {
            return Err(HeatError::Time);
        }
        if leds < 2 || params.foot_leds < 1 || params.foot_leds >= leds {
            return Err(HeatError::Span {
                leds,
                foot_leds: params.foot_leds,
            });
        }
        let dx = 1.0 / CELLS as f64;
        let xe = params.foot_leds as f64 / leds as f64;
        // First cell whose centre sits at or above the foot.
        let ie = (0..CELLS).find(|&i| (i as f64 + 0.5) * dx >= xe);
        let Some(ie) = ie else {
            return Err(HeatError::Span {
                leds,
                foot_leds: params.foot_leds,
            });
        };
        // The shift dumps into the cell under `ie`. A foot that covers no
        // cell has nowhere to put that heat.
        if ie == 0 {
            return Err(HeatError::Span {
                leds,
                foot_leds: params.foot_leds,
            });
        }
        let diffusion = 1.0 / params.conduction_s;
        let speed = (1.0 - xe) / params.transit_s;
        let mut kc = vec![0.0; CELLS];
        for (i, cell) in kc.iter_mut().enumerate() {
            *cell = if i < ie { speed * xe } else { diffusion };
        }
        let mut kf = vec![0.0; CELLS - 1];
        for i in 0..CELLS - 1 {
            kf[i] = 2.0 * kc[i] * kc[i + 1] / (kc[i] + kc[i + 1]);
        }
        Ok(Self {
            led_count: leds,
            cells: vec![0.0; CELLS],
            heater: 0.0,
            shift: 0.0,
            ie,
            speed,
            dx,
            kc,
            kf,
            weights: tent_weights(leds),
            attack_s: params.attack_s,
            smooth_s: params.smooth_s,
            diag: vec![0.0; CELLS],
            lower: vec![0.0; CELLS],
            upper: vec![0.0; CELLS],
            rhs: vec![0.0; CELLS],
            cp: vec![0.0; CELLS],
            dp: vec![0.0; CELLS],
        })
    }

    /// LEDs in this bar.
    #[must_use]
    pub fn led_count(&self) -> usize {
        self.led_count
    }

    /// Heater temperature, percent of range.
    #[must_use]
    pub fn heater(&self) -> f64 {
        self.heater
    }

    /// Advance by `dt` seconds with the heater driven by `input_pct`
    /// (percent of range). A non-finite or non-positive `dt` does nothing.
    /// A gap above [`MAX_GAP_S`] is cut to that, then split into sub-steps
    /// of at most [`MAX_SUBSTEP_S`].
    pub fn step(&mut self, input_pct: f64, dt: f64) {
        if !dt.is_finite() || dt <= 0.0 || !input_pct.is_finite() {
            return;
        }
        let mut left = dt.min(MAX_GAP_S);
        while left > 1e-12 {
            let h = left.min(MAX_SUBSTEP_S);
            self.step_once(input_pct, h);
            left -= h;
        }
    }

    /// Tent-filtered LED values, bottom first, percent of range.
    #[must_use]
    pub fn leds(&self) -> Vec<f64> {
        self.weights
            .iter()
            .map(|weights| {
                weights
                    .iter()
                    .map(|(cell, weight)| self.cells[*cell] * weight)
                    .sum()
            })
            .collect()
    }

    fn step_once(&mut self, input: f64, dt: f64) {
        self.heater = heater_follow(self.heater, input, self.attack_s, self.smooth_s, dt);
        let above = CELLS - self.ie;
        let add = self.speed * dt / self.dx;
        if add.is_finite() {
            self.shift = (self.shift + add).min(above as f64);
        } else {
            self.shift = above as f64;
        }
        let n = self.shift.floor() as usize;
        self.shift -= n as f64;
        if n > 0 {
            let mut dump = 0.0;
            for cell in &self.cells[self.ie..self.ie + n] {
                dump += *cell;
            }
            self.cells.copy_within(self.ie + n..CELLS, self.ie);
            for cell in &mut self.cells[CELLS - n..] {
                *cell = self.heater;
            }
            self.cells[self.ie - 1] += dump;
        }
        self.conduct(dt);
    }

    /// One backward-Euler step. The sink under cell 0 is held at 0.
    fn conduct(&mut self, dt: f64) {
        let c = dt / (self.dx * self.dx);
        let sink = 2.0 * self.kc[0] * c;
        let n = CELLS;
        for i in 0..n {
            let left = if i > 0 { self.kf[i - 1] * c } else { sink };
            let right = if i + 1 < n { self.kf[i] * c } else { 0.0 };
            self.diag[i] = 1.0 + left + right;
            self.lower[i] = if i > 0 { -self.kf[i - 1] * c } else { 0.0 };
            self.upper[i] = if i + 1 < n { -self.kf[i] * c } else { 0.0 };
            self.rhs[i] = self.cells[i];
        }
        let diag = &self.diag;
        let lower = &self.lower;
        let upper = &self.upper;
        let rhs = &self.rhs;
        let cp = &mut self.cp;
        let dp = &mut self.dp;
        cp[0] = upper[0] / diag[0];
        dp[0] = rhs[0] / diag[0];
        for i in 1..n {
            let m = diag[i] - lower[i] * cp[i - 1];
            cp[i] = upper[i] / m;
            dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / m;
        }
        self.cells[n - 1] = dp[n - 1];
        for i in (0..n - 1).rev() {
            self.cells[i] = dp[i] - cp[i] * self.cells[i + 1];
        }
    }
}

/// `prev + (value - prev) * (1 - exp(-dt / tau))`, with `tau` chosen as
/// `attack_s` or `smooth_s`. A `tau` of 0 jumps to `value`.
fn heater_follow(prev: f64, value: f64, attack_s: f64, smooth_s: f64, dt: f64) -> f64 {
    let tau = if value > prev { attack_s } else { smooth_s };
    if tau <= 0.0 {
        return value;
    }
    let alpha = 1.0 - (-dt.max(0.0) / tau).exp();
    prev + (value - prev) * alpha
}

/// Weight of each cell on LED `j`: 1 at the LED's centre, 0 one pitch away,
/// normalised.
fn tent_weights(leds: usize) -> Vec<Vec<(usize, f64)>> {
    let dx = 1.0 / CELLS as f64;
    let pitch = leds as f64;
    (0..leds)
        .map(|j| {
            let centre = (j as f64 + 0.5) / pitch;
            let mut weights = Vec::new();
            let mut total = 0.0;
            for i in 0..CELLS {
                let x = (i as f64 + 0.5) * dx;
                let weight = 1.0 - (x - centre).abs() * pitch;
                if weight > 0.0 {
                    weights.push((i, weight));
                    total += weight;
                }
            }
            for (_, weight) in &mut weights {
                *weight /= total;
            }
            weights
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum_cells(bar: &HeatBar) -> f64 {
        bar.cells.iter().sum()
    }

    fn hottest(leds: &[f64]) -> usize {
        leds.iter()
            .enumerate()
            .max_by(|left, right| left.1.partial_cmp(right.1).expect("finite"))
            .map(|(index, _)| index)
            .expect("leds")
    }

    #[test]
    fn no_input_never_gains_heat_and_a_hot_bar_decays() {
        let mut cold = HeatBar::new(28, HeatParams::default()).expect("bar");
        let mut last = 0.0;
        for _ in 0..40 {
            cold.step(0.0, 0.5);
            let sum = sum_cells(&cold);
            assert!(sum <= last + 1e-8, "{last} -> {sum}");
            last = sum;
        }
        assert!(last.abs() < 1e-8, "{last}");

        let mut hot = HeatBar::new(28, HeatParams::default()).expect("bar");
        for _ in 0..40 {
            hot.step(125.0, 0.5);
        }
        let mut last = sum_cells(&hot);
        assert!(last > 100.0, "the burst did not heat the bar ({last})");
        for _ in 0..4000 {
            hot.step(0.0, 0.5);
            let sum = sum_cells(&hot);
            assert!(
                sum <= last + 1e-6,
                "heat rose from {last} to {sum} with no input"
            );
            last = sum;
        }
        assert!(last < 1.0, "still {last} after cooling");
    }

    #[test]
    fn steady_125_for_five_minutes_fills_from_the_top() {
        let mut bar = HeatBar::new(28, HeatParams::default()).expect("bar");
        for _ in 0..600 {
            bar.step(125.0, 0.5);
        }
        let leds = bar.leds();
        assert!((leds[27] - 125.0).abs() <= 0.5, "top LED {}", leds[27]);
        for (index, value) in leds.iter().enumerate().skip(4) {
            assert!(
                *value >= 100.0,
                "LED {} = {value}, wanted >= 100",
                index + 1
            );
        }
        assert!((13.0..17.0).contains(&leds[0]), "LED 1 = {}", leds[0]);
    }

    #[test]
    fn a_two_second_burst_travels_toward_the_bottom() {
        let mut bar = HeatBar::new(28, HeatParams::default()).expect("bar");
        let mut t = 0.0;
        while t < 2.0 - 1e-12 {
            bar.step(125.0, 0.05);
            t += 0.05;
        }
        // The band starts at the heater. A single 0.05 s sample can swap two
        // LEDs that differ by a few hundredths while the peak sits on the
        // boundary between them; the travel is the position every quarter
        // second, about one LED per 2.6 s, only toward the bottom.
        let mut last = hottest(&bar.leds());
        assert!(last >= 26, "the burst did not stay at the top ({last})");
        let mut moved = false;
        for _ in 0..400 {
            bar.step(0.0, 0.25);
            let leds = bar.leds();
            let peak = leds.iter().copied().fold(0.0, f64::max);
            if peak < 20.0 {
                break;
            }
            let hot = hottest(&leds);
            assert!(hot <= last, "hottest LED moved from {last} up to {hot}");
            if hot < last {
                moved = true;
            }
            last = hot;
        }
        assert!(moved, "the band never left the top");
        assert!(last < 20, "hottest LED ended at {last}");
    }

    #[test]
    fn foot_of_one_and_foot_of_len_minus_one_run() {
        for foot in [1, 27] {
            let params = HeatParams {
                foot_leds: foot,
                ..HeatParams::default()
            };
            let mut bar = HeatBar::new(28, params).expect("bar");
            bar.step(125.0, 1.0);
            let leds = bar.leds();
            assert_eq!(leds.len(), 28);
            assert!(
                leds.iter().all(|value| value.is_finite()),
                "{foot}: {leds:?}"
            );
        }
    }

    #[test]
    fn one_led_is_rejected_and_two_leds_need_a_foot_of_one() {
        let err = HeatBar::new(
            1,
            HeatParams {
                foot_leds: 1,
                ..HeatParams::default()
            },
        );
        assert!(matches!(err, Err(HeatError::Span { leds: 1, .. })));
        assert!(HeatBar::new(2, HeatParams::default()).is_err());
        let mut bar = HeatBar::new(
            2,
            HeatParams {
                foot_leds: 1,
                ..HeatParams::default()
            },
        )
        .expect("two LEDs with a foot of 1");
        bar.step(80.0, 0.5);
        let leds = bar.leds();
        assert_eq!(leds.len(), 2);
        assert!(leds.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn a_coarse_step_stays_close_to_a_fine_one_and_does_not_blow_up() {
        fn run(dt: f64) -> Vec<f64> {
            let mut bar = HeatBar::new(28, HeatParams::default()).expect("bar");
            let mut left = 2.0;
            while left > 1e-12 {
                let h = dt.min(left);
                bar.step(125.0, h);
                left -= h;
            }
            bar.leds()
        }
        let fine = run(0.05);
        let mid = run(0.1);
        let gap = run(2.0);
        for (name, leds) in [("0.05", &fine), ("0.1", &mid), ("2.0", &gap)] {
            assert!(
                leds.iter()
                    .all(|value| value.is_finite() && *value > -1.0 && *value < 200.0),
                "{name}: {leds:?}"
            );
        }
        for (index, (a, b)) in fine.iter().zip(&mid).enumerate() {
            assert!((a - b).abs() < 1.0, "LED {index}: 0.05 {a} vs 0.1 {b}");
        }
        for (index, (a, c)) in fine.iter().zip(&gap).enumerate() {
            assert!(
                (a - c).abs() < 15.0,
                "LED {index}: 0.05 {a} vs one 2 s gap {c}"
            );
        }
    }

    #[test]
    fn attack_rises_and_smooth_falls() {
        let mut bar = HeatBar::new(
            8,
            HeatParams {
                foot_leds: 1,
                attack_s: 2.0,
                smooth_s: 10.0,
                ..HeatParams::default()
            },
        )
        .expect("bar");
        bar.step(125.0, 2.0);
        let risen = 125.0 * (1.0 - (-1.0_f64).exp());
        assert!(
            (bar.heater() - risen).abs() < 1e-9,
            "heater {} wanted {risen}",
            bar.heater()
        );
        bar.step(0.0, 10.0);
        let fallen = risen * (-1.0_f64).exp();
        assert!(
            (bar.heater() - fallen).abs() < 1e-9,
            "heater {} wanted {fallen}",
            bar.heater()
        );
    }

    #[test]
    fn conduction_below_one_second_is_rejected() {
        let slow = HeatParams {
            conduction_s: 0.5,
            ..HeatParams::default()
        };
        assert!(matches!(HeatBar::new(8, slow), Err(HeatError::Time)));
        let floor = HeatParams {
            conduction_s: 1.0,
            ..HeatParams::default()
        };
        assert!(HeatBar::new(8, floor).is_ok());
    }

    #[test]
    fn a_fast_transit_stays_finite_and_the_sink_only_removes_heat() {
        let mut bar = HeatBar::new(
            28,
            HeatParams {
                transit_s: 0.01,
                attack_s: 0.0,
                smooth_s: 0.0,
                ..HeatParams::default()
            },
        )
        .expect("bar");
        bar.step(80.0, 0.5);
        let before = sum_cells(&bar);
        assert!(before.is_finite() && before > 0.0, "{before}");
        // One sub-step. The heater jumps to 0, so the shift injects nothing
        // and the sink is the only place heat can go.
        bar.step(0.0, 0.5);
        let after = sum_cells(&bar);
        assert!(after.is_finite(), "{after}");
        assert!(bar.heater().is_finite());
        assert!(
            bar.leds().iter().all(|value| value.is_finite()),
            "{:?}",
            bar.leds()
        );
        assert!(
            after < before,
            "sink flux had the wrong sign: {before} -> {after}"
        );
    }

    #[test]
    fn a_huge_gap_matches_a_sixty_second_step() {
        let mut huge = HeatBar::new(28, HeatParams::default()).expect("bar");
        let mut capped = HeatBar::new(28, HeatParams::default()).expect("bar");
        huge.step(40.0, 1.0e6);
        capped.step(40.0, MAX_GAP_S);
        assert_eq!(huge.heater(), capped.heater());
        assert_eq!(huge.leds(), capped.leds());
    }

    #[test]
    fn a_one_second_step_is_two_half_second_steps_and_not_one_unsplit_step() {
        let params = HeatParams {
            foot_leds: 1,
            conduction_s: 1.0,
            attack_s: 0.0,
            smooth_s: 0.0,
            ..HeatParams::default()
        };
        let mut split = HeatBar::new(14, params).expect("bar");
        split.step(125.0, 1.0);
        let mut halves = HeatBar::new(14, params).expect("bar");
        halves.step(125.0, 0.5);
        halves.step(125.0, 0.5);
        assert_eq!(split.heater(), halves.heater());
        assert_eq!(split.leds(), halves.leds());

        let mut unsplit = HeatBar::new(14, params).expect("bar");
        unsplit.step_once(125.0, 1.0);
        assert_ne!(
            split.leds(),
            unsplit.leds(),
            "a 1 s step must be split at {MAX_SUBSTEP_S} s"
        );
    }

    #[test]
    fn a_non_finite_input_leaves_the_bar_unchanged() {
        let mut bar = HeatBar::new(14, HeatParams::default()).expect("bar");
        bar.step(40.0, 0.5);
        let mut twin = bar.clone();
        bar.step(f64::NAN, 0.5);
        assert_eq!(bar.heater(), twin.heater());
        assert_eq!(bar.leds(), twin.leds());
        bar.step(10.0, 0.5);
        twin.step(10.0, 0.5);
        assert_eq!(bar.heater(), twin.heater());
        assert_eq!(bar.leds(), twin.leds());
    }
}
