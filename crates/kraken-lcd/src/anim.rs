//! Frame clock, smoke and embers at the 125 % peg, and the bar flicker.
//!
//! Everything here is deterministic. The step is one frame, `dt = 0.1 s`;
//! positions are Euler-integrated per frame. Randomness is xorshift32 with a
//! fixed seed, so a given seed and frame count always give the same picture.
//! Coordinates are frame pixels; angles are degrees clockwise from 12.

/// One frame of the 10 fps clock, seconds.
pub const DT: f32 = 0.1;
/// Seed of the peg's particle system.
pub const PARTICLE_SEED: u32 = 20_260_926;

/// Ring start, degrees clockwise from 12 (7:30).
pub const RING_START_DEG: f32 = 225.0;
/// Ring sweep for 0..=125, degrees.
pub const RING_SWEEP_DEG: f32 = 270.0;
/// Top of the ring scale.
pub const RING_MAX: f32 = 125.0;

/// Angle of activity `p` on the ring, degrees clockwise from 12. `p` is
/// clamped to `0..=125`: 0 → 225°, 50 → 333°, 100 → 441° (81°), 125 → 495° (135°).
/// The result is not wrapped, so it grows monotonically with `p`.
#[must_use]
pub fn ring_angle(p: f32) -> f32 {
    let p = if p.is_finite() {
        p.clamp(0.0, RING_MAX)
    } else {
        0.0
    };
    RING_START_DEG + p * (RING_SWEEP_DEG / RING_MAX)
}

/// xorshift32, as the design's reference (`s ^= s << 13; s ^= s >> 17; s ^= s << 5`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XorShift32(u32);

impl XorShift32 {
    /// A zero seed becomes 1.
    #[must_use]
    pub fn new(seed: u32) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    /// Next raw state.
    pub fn next_u32(&mut self) -> u32 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        self.0 = s;
        s
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (f64::from(self.next_u32()) / 4_294_967_296.0) as f32
    }
}

/// Smoke intensity at activity `v`: nothing below 115, full at the 125 peg.
#[must_use]
pub fn smoke_intensity(v: f32) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    ((v - 115.0) / 10.0).clamp(0.0, 1.0)
}

/// Glow pulse at frame time `t`: `0.7 + 0.3·sin(2π·0.7·t)`.
#[must_use]
pub fn pulse(t: f32) -> f32 {
    0.7 + 0.3 * (std::f32::consts::TAU * 0.7 * t).sin()
}

/// Bar `index`'s flicker noise at time `t`, in `-1..=1`. Two phases per bar
/// come from xorshift32 seeded with `1000 + 7919·index`, so neighbours differ
/// and a bar is always the same.
#[must_use]
pub fn flicker(index: usize, t: f32) -> f32 {
    let seed = 1000_u32.wrapping_add(7919_u32.wrapping_mul(index as u32));
    let mut rng = XorShift32::new(seed);
    let phi = rng.next_f32() * std::f32::consts::TAU;
    let psi = rng.next_f32() * std::f32::consts::TAU;
    let tau = std::f32::consts::TAU;
    0.6 * (tau * 1.3 * t + phi).sin() + 0.4 * (tau * 3.1 * t + psi).sin()
}

/// A translucent smoke wisp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wisp {
    /// Position.
    pub x: f32,
    /// Position.
    pub y: f32,
    /// Seconds alive.
    pub age: f32,
    /// Lifetime, seconds.
    pub life: f32,
    /// Clockwise tangent at the head when spawned.
    pub tx: f32,
    /// Clockwise tangent at the head when spawned.
    pub ty: f32,
    /// Curl phase.
    pub phase: f32,
    /// Curl side, ±1.
    pub side: f32,
}

/// A hot ember.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ember {
    /// Position.
    pub x: f32,
    /// Position.
    pub y: f32,
    /// Seconds alive.
    pub age: f32,
    /// Lifetime, seconds.
    pub life: f32,
    /// Velocity, px/s.
    pub vx: f32,
    /// Velocity, px/s.
    pub vy: f32,
}

/// Wisps: spawn chance per frame at full intensity, and the most alive.
pub const WISP_SPAWN: f32 = 0.6;
/// Most wisps alive.
pub const WISP_CAP: usize = 12;
/// Embers: spawn chance per frame at full intensity.
pub const EMBER_SPAWN: f32 = 0.35;
/// Most embers alive.
pub const EMBER_CAP: usize = 8;

/// The peg's smoke and embers.
#[derive(Clone, Debug, PartialEq)]
pub struct Particles {
    rng: XorShift32,
    /// Alive wisps, oldest first.
    pub wisps: Vec<Wisp>,
    /// Alive embers, oldest first.
    pub embers: Vec<Ember>,
}

impl Default for Particles {
    fn default() -> Self {
        Self::new(PARTICLE_SEED)
    }
}

impl Particles {
    /// Nothing alive; the generator starts at `seed`.
    #[must_use]
    pub fn new(seed: u32) -> Self {
        Self {
            rng: XorShift32::new(seed),
            wisps: Vec::with_capacity(WISP_CAP),
            embers: Vec::with_capacity(EMBER_CAP),
        }
    }

    /// Drop everything alive. The generator keeps its state.
    pub fn clear(&mut self) {
        self.wisps.clear();
        self.embers.clear();
    }

    /// Alive particles.
    #[must_use]
    pub fn alive(&self) -> usize {
        self.wisps.len() + self.embers.len()
    }

    /// One 0.1 s frame with the ring at activity `v` on a ring of `ring_radius`.
    pub fn step(&mut self, v: f32, ring_radius: f32) {
        let s = smoke_intensity(v);
        let head = ring_angle(v);
        let (hx, hy) = polar(ring_radius, head);
        for wisp in &mut self.wisps {
            wisp.age += DT;
        }
        self.wisps.retain(|wisp| wisp.age < wisp.life);
        for ember in &mut self.embers {
            ember.age += DT;
        }
        self.embers.retain(|ember| ember.age < ember.life);

        let th = head.to_radians();
        let (tx, ty) = (th.cos(), th.sin());
        if s > 0.0 {
            if self.wisps.len() < WISP_CAP && self.rng.next_f32() < WISP_SPAWN * s {
                let angle = head - self.rng.next_f32() * 6.0;
                let radius = ring_radius + (self.rng.next_f32() * 4.0 - 2.0);
                let (x, y) = polar(radius, angle);
                let life = 1.5 + self.rng.next_f32() * 0.6;
                let phase = self.rng.next_f32() * std::f32::consts::TAU;
                let side = if self.rng.next_f32() < 0.5 { -1.0 } else { 1.0 };
                self.wisps.push(Wisp {
                    x,
                    y,
                    age: 0.0,
                    life,
                    tx,
                    ty,
                    phase,
                    side,
                });
            }
            if self.embers.len() < EMBER_CAP && self.rng.next_f32() < EMBER_SPAWN * s {
                let jitter = (self.rng.next_f32() * 2.0 - 1.0) * 30_f32.to_radians();
                let (c, sn) = (jitter.cos(), jitter.sin());
                let speed = 22.0 + self.rng.next_f32() * 16.0;
                let x = hx + (self.rng.next_f32() * 6.0 - 3.0);
                let y = hy + (self.rng.next_f32() * 6.0 - 3.0);
                let life = 0.6 + self.rng.next_f32() * 0.3;
                self.embers.push(Ember {
                    x,
                    y,
                    age: 0.0,
                    life,
                    vx: (tx * c - ty * sn) * speed,
                    vy: (tx * sn + ty * c) * speed,
                });
            }
        }
        for wisp in &mut self.wisps {
            // Along the tangent into the gap, a little inward, and screen-up.
            let dx = 160.0 - wisp.x;
            let dy = 160.0 - wisp.y;
            let len = (dx * dx + dy * dy).sqrt().max(1.0);
            wisp.x += (wisp.tx * 20.0 + dx / len * 2.0) * DT;
            wisp.y += (wisp.ty * 20.0 + dy / len * 2.0 - 6.0) * DT;
        }
        for ember in &mut self.embers {
            ember.vx *= 0.88;
            ember.vy *= 0.88;
            ember.x += ember.vx * DT;
            ember.y += (ember.vy - 6.0) * DT;
        }
    }
}

/// Frame memory for the animated parts: the frame counter, the smoothed ring
/// value, and the particles.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Anim {
    /// Frames presented since start.
    pub frame: u64,
    /// Ring value after smoothing, before quantising. Stream mode only.
    pub shown: Option<f32>,
    /// Smoke and embers.
    pub particles: Particles,
}

impl Anim {
    /// Frame time, seconds.
    #[must_use]
    pub fn t(&self) -> f32 {
        (self.frame % 1_000_000) as f32 * DT
    }

    /// A still frame: `frames` steps at `v` from a fresh seed, as the design's
    /// static faces show the peg after four seconds.
    #[must_use]
    pub fn still(v: f32, ring_radius: f32, frames: u64, seed: u32) -> Self {
        let mut particles = Particles::new(seed);
        for _ in 0..frames {
            particles.step(v, ring_radius);
        }
        Self {
            frame: frames,
            shown: None,
            particles,
        }
    }
}

fn polar(radius: f32, deg: f32) -> (f32, f32) {
    let theta = deg.to_radians();
    (160.0 + radius * theta.sin(), 160.0 - radius * theta.cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift_matches_the_reference_sequence() {
        // JS: s^=s<<13; s>>>=0; s^=s>>>17; s^=s<<5; s>>>=0 from seed 1.
        let mut rng = XorShift32::new(1);
        assert_eq!(rng.next_u32(), 270_369);
        assert_eq!(rng.next_u32(), 67_634_689);
        assert_eq!(XorShift32::new(0), XorShift32::new(1));
    }
}
