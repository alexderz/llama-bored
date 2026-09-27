//! `act_color` and friends. The ramp lives in [`llama_core::color`] so the
//! LCD writer and the RGB writer (llama-light) share one definition.

pub use llama_core::color::{ACT_STOPS, BB_STOPS, BLACK, HOT_GOLD, WHITE, act_color, hex, mix};

/// How far a tip mixes toward white: at most 15 % under 50 so cold bars and
/// the cold comet head stay saturated (T62); `lift` unchanged at 50 and up.
#[must_use]
pub fn cold_tip_lift(v: f32, lift: f32) -> f32 {
    if v < 50.0 { lift.min(0.15) } else { lift }
}
