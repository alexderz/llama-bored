//! V3b Plasma Blackbody: the colour function, the 270° scale, the particles,
//! and the incandescent bars.

use kraken_lcd::anim::{
    EMBER_CAP, Particles, RING_MAX, WISP_CAP, XorShift32, flicker, ring_angle, smoke_intensity,
};
use kraken_lcd::config::DisplayCfg;
use kraken_lcd::present::{Ai, View, warm_still};
use kraken_lcd::render::color::{ACT_STOPS, BB_STOPS, act_color, hex};
use kraken_lcd::render::ring::{cap_deg, head_color, readout_color};
use kraken_lcd::render::{self, Assets, Frame, Rgb};

fn rgb(frame: &Frame, x: f32, y: f32) -> (u8, u8, u8) {
    let pixel = frame
        .0
        .pixel((x - 0.5).round() as u32, (y - 0.5).round() as u32)
        .expect("on the frame");
    (pixel.red(), pixel.green(), pixel.blue())
}

fn polar(radius: f32, deg: f32) -> (f32, f32) {
    let theta = deg.to_radians();
    (160.0 + radius * theta.sin(), 160.0 - radius * theta.cos())
}

fn at(frame: &Frame, radius: f32, deg: f32) -> (u8, u8, u8) {
    let (x, y) = polar(radius, deg);
    rgb(frame, x, y)
}

fn near(got: (u8, u8, u8), expect: Rgb, tol: u8) -> bool {
    got.0.abs_diff(expect.r) <= tol
        && got.1.abs_diff(expect.g) <= tol
        && got.2.abs_diff(expect.b) <= tol
}

fn view(ring: Option<u8>) -> View {
    View {
        ring_pct: ring,
        ai: Ai::Idle,
        ..View::default()
    }
}

#[test]
fn act_color_hits_every_stop_and_interpolates_between() {
    let stops = [
        (0.0, 0x4A55C8),
        (20.0, 0x7550D8),
        (40.0, 0xA64ACF),
        (60.0, 0xD044A8),
        (80.0, 0xF4466A),
        (100.0, 0xFF3A22),
        (107.0, 0xFF6E1A),
        (114.0, 0xFFB02A),
        (120.0, 0xFFEA9A),
        (125.0, 0xFFFFFF),
    ];
    for (p, value) in stops {
        assert_eq!(act_color(p), hex(value), "stop {p}");
    }
    assert_eq!(ACT_STOPS.len() + BB_STOPS.len(), 11, "eleven stops");
    // Halfway L1 → L2, rounded per channel.
    assert_eq!(
        act_color(10.0),
        Rgb {
            r: 96,
            g: 83,
            b: 208
        }
    );
    // Halfway 114 → 120: #FFB02A → #FFEA9A.
    assert_eq!(
        act_color(117.0),
        Rgb {
            r: 255,
            g: 205,
            b: 98
        }
    );
    // Clamped to 0–125; NaN reads as 0.
    assert_eq!(act_color(-5.0), hex(0x4A55C8));
    assert_eq!(act_color(400.0), hex(0xFFFFFF));
    assert_eq!(act_color(f32::NAN), hex(0x4A55C8));
    // Red at 100 is continuous with L6.
    assert_eq!(act_color(100.0), act_color(100.000_01));
}

#[test]
fn readout_and_head_colours_follow_the_spec() {
    // Up to 100: act_color mixed 35 % toward white.
    assert_eq!(
        readout_color(0.0),
        Rgb {
            r: 137,
            g: 145,
            b: 219
        }
    );
    // Above 100: act_color(max(112, v)).
    assert_eq!(readout_color(104.0), act_color(112.0));
    assert_eq!(readout_color(125.0), hex(0xFFFFFF));
    // The comet head is act_color(v) mixed halfway to white.
    assert_eq!(
        head_color(100.0),
        Rgb {
            r: 255,
            g: 157,
            b: 145
        }
    );
}

#[test]
fn the_scale_is_270_degrees_from_seven_thirty() {
    assert_eq!(ring_angle(0.0), 225.0);
    assert!((ring_angle(50.0) - 333.0).abs() < 1e-4);
    assert!(
        (ring_angle(100.0) - 441.0).abs() < 1e-4,
        "100 is 81° (2:40)"
    );
    assert!(
        (ring_angle(125.0) - 495.0).abs() < 1e-4,
        "125 is 135° (4:30)"
    );
    assert_eq!(ring_angle(-3.0), 225.0);
    assert_eq!(ring_angle(200.0), ring_angle(RING_MAX));
    assert!((ring_angle(1.0) - ring_angle(0.0) - 2.16).abs() < 1e-4);
    // The leading cap's outer edge is the value angle: 6 px at r 151.
    assert!((cap_deg(151.0, 12.0) - 2.2769).abs() < 1e-3);
}

#[test]
fn the_arc_covers_its_value_by_position_and_leaves_the_rest_dim() {
    let mut assets = Assets::load().expect("assets");
    let frame = render::render(&view(Some(50)), &DisplayCfg::default(), &mut assets);
    // 40 on the scale is 311.4°: inside a 50 % arc, before the comet head.
    let lit = at(&frame, 151.0, ring_angle(15.0));
    assert!(near(lit, act_color(15.0), 24), "15 got {lit:?}");
    // 60 is past the value: the dim ramp.
    let dim = at(&frame, 151.0, ring_angle(62.0));
    assert!(
        dim.0 < 70 && dim.1 < 30 && dim.2 < 60,
        "past the value is the dim track, got {dim:?}"
    );
    // The gap at 6 o'clock stays black and the readout says 50%.
    assert_eq!(at(&frame, 151.0, 190.0), (0, 0, 0));
    let mut readout = 0;
    for y in 302..316 {
        for x in 140..180 {
            let got = rgb(&frame, x as f32 + 0.5, y as f32 + 0.5);
            if near(got, readout_color(50.0), 40) {
                readout += 1;
            }
        }
    }
    assert!(readout > 10, "the readout sits in the gap, saw {readout}");
    // The 100 gate is a light tick even when the arc stops short of it.
    let gate = at(&frame, 151.0, 81.0);
    assert!(gate.0 > 150 && gate.1 > 150, "gate {gate:?}");
}

#[test]
fn the_peg_goes_white_hot() {
    let mut assets = Assets::load().expect("assets");
    let frame = render::render(&view(Some(125)), &DisplayCfg::default(), &mut assets);
    let head = at(&frame, 151.0, ring_angle(123.0));
    assert!(
        head.0 > 240 && head.1 > 230 && head.2 > 220,
        "the last 8° are white at 125, got {head:?}"
    );
    let orange = at(&frame, 151.0, ring_angle(106.0));
    assert!(
        orange.0 > 230 && orange.1 > 80 && orange.2 < 80,
        "106 is blackbody orange-red, got {orange:?}"
    );
}

#[test]
fn particles_are_deterministic_for_a_seed() {
    let run = |seed: u32| {
        let mut particles = Particles::new(seed);
        for _ in 0..60 {
            particles.step(125.0, 151.0);
        }
        particles
    };
    assert_eq!(run(7), run(7), "same seed, same frames, same smoke");
    assert_ne!(run(7), run(8), "a different seed differs");
    let pinned = run(7);
    assert!(pinned.alive() > 0);
    assert!(pinned.wisps.len() <= WISP_CAP && pinned.embers.len() <= EMBER_CAP);
    assert!(pinned.alive() <= 20, "at most 20 particles");
    // Embers fly into the gap band, never inside r 114.
    for ember in &pinned.embers {
        let r = ((ember.x - 160.0).powi(2) + (ember.y - 160.0).powi(2)).sqrt();
        assert!(r > 114.0, "ember at r {r}");
    }
    // Nothing below 115; full at the peg.
    assert_eq!(smoke_intensity(115.0), 0.0);
    assert_eq!(smoke_intensity(125.0), 1.0);
    let mut cool = Particles::new(7);
    for _ in 0..60 {
        cool.step(114.0, 151.0);
    }
    assert_eq!(cool.alive(), 0);
    // The generator is the reference xorshift32.
    let mut rng = XorShift32::new(1);
    assert_eq!(rng.next_u32(), 270_369);
}

#[test]
fn flicker_is_fixed_per_bar_and_bounded() {
    for index in 0..24 {
        for frame in 0..50 {
            let t = frame as f32 * 0.1;
            let n = flicker(index, t);
            assert!((-1.0..=1.0).contains(&n), "bar {index} at {t}: {n}");
            assert_eq!(n, flicker(index, t), "same bar, same time");
        }
    }
    assert_ne!(flicker(0, 1.0), flicker(1, 1.0), "neighbours differ");
}

#[test]
fn a_still_view_smokes_only_at_the_peg() {
    let mut pinned = view(Some(125));
    warm_still(&mut pinned);
    assert!(pinned.dial_state.anim.particles.alive() > 0);
    let mut busy = view(Some(100));
    warm_still(&mut busy);
    assert_eq!(busy.dial_state.anim.particles.alive(), 0);
    let mut stale = view(Some(125));
    stale.ai = Ai::NoData;
    warm_still(&mut stale);
    assert_eq!(stale.dial_state.anim.particles.alive(), 0, "off when stale");
}

/// Bar geometry: slot centre of the oldest (24°) bar.
const OLDEST_DEG: f32 = 360.0 - 12.0;

#[test]
fn bars_use_act_color_and_turn_incandescent_over_100() {
    let mut assets = Assets::load().expect("assets");
    let mut dial = [None; 24];
    dial[23] = Some(60);
    let mut sixty = view(None);
    sixty.dial = dial;
    let frame = render::render(&sixty, &DisplayCfg::default(), &mut assets);
    // L = 3 + 21 × 0.6 = 15.6; 72 % of it is r 129.2, the pure colour.
    let body = at(&frame, 129.2, OLDEST_DEG);
    assert!(
        near(body, act_color(60.0), 30),
        "60 is #D044A8, got {body:?}"
    );
    // Tip at r 133.6, past it is empty.
    assert_eq!(at(&frame, 136.0, OLDEST_DEG), (0, 0, 0));

    dial[23] = Some(125);
    let mut hot = view(None);
    hot.dial = dial;
    let frame = render::render(&hot, &DisplayCfg::default(), &mut assets);
    // Full length at 125 (tip r 142), ember-dark base, white-hot tip.
    let base = at(&frame, 119.5, OLDEST_DEG - 6.0);
    let tip = at(&frame, 140.5, OLDEST_DEG - 6.0);
    assert!(
        base.0 < 190 && base.1 < 60,
        "coals are dark at the bottom, got {base:?}"
    );
    assert!(
        tip.0 > 240 && tip.1 > 200 && tip.2 > 150,
        "the tip is near white at 125, got {tip:?}"
    );
    // Bloom: Plus halo just outside the bar's edge.
    let halo = at(&frame, 143.5, OLDEST_DEG - 6.0);
    assert!(
        halo.0 > 20 && halo.1 > 15,
        "a hot bar blooms past its tip, got {halo:?}"
    );
    // At 100 the bar is red, full length, and does not bloom.
    dial[23] = Some(100);
    let mut full = view(None);
    full.dial = dial;
    let frame = render::render(&full, &DisplayCfg::default(), &mut assets);
    let body = at(&frame, 118.0 + 0.72 * 24.0, OLDEST_DEG);
    assert!(near(body, act_color(100.0), 30), "100 is L6, got {body:?}");
    assert_eq!(at(&frame, 143.5, OLDEST_DEG - 6.0), (0, 0, 0));
}

#[test]
fn the_time_scale_labels_a1_and_sweeps_only_with_a_fill() {
    let mut assets = Assets::load().expect("assets");
    let mut plain = view(Some(40));
    plain.dial = [Some(40); 24];
    let mut swept = plain.clone();
    for tier in swept.scale.iter_mut().skip(3) {
        tier.fill = Some(500);
    }
    let a = render::render(&plain, &DisplayCfg::default(), &mut assets);
    let b = render::render(&swept, &DisplayCfg::default(), &mut assets);
    // The 1 min tier's newest bar spans 150°–172.5°; the sweep is at r 115.6.
    // The 1.2 px arc is anti-aliased; take the brightest pixel across it.
    let sweep = [115.0, 115.6, 116.2]
        .map(|r| at(&b, r, 152.0))
        .into_iter()
        .max_by_key(|c| u16::from(c.0) + u16::from(c.1) + u16::from(c.2))
        .unwrap_or_default();
    assert!(
        sweep.0 > 80 && sweep.0.abs_diff(sweep.2) < 12 && sweep.0.abs_diff(sweep.1) < 8,
        "the fill sweep is grey #C8C8CC, got {sweep:?}"
    );
    let bare = at(&a, 115.6, 152.0);
    assert!(bare.0 < 40, "no fill, no sweep, got {bare:?}");
    // Half full: nothing past the middle of the bar.
    let past = at(&b, 115.6, 168.0);
    assert!(past.0 < 40, "the sweep stops at half, got {past:?}");
    // Labels: "15s" sits at (269, 163) on A1 and not on A3.
    let label_ink = |frame: &Frame| {
        let mut ink = 0;
        for y in 155..166 {
            for x in 262..278 {
                let got = rgb(frame, x as f32 + 0.5, y as f32 + 0.5);
                if near(got, hex(0x7A7A82), 40) {
                    ink += 1;
                }
            }
        }
        ink
    };
    assert!(label_ink(&a) > 5, "A1 draws the 15s label");
    let mut a3 = plain.clone();
    a3.variant = kraken_lcd::present::Variant::A3;
    let c = render::render(&a3, &DisplayCfg::default(), &mut assets);
    assert_eq!(label_ink(&c), 0, "A3 has no labels");
}
