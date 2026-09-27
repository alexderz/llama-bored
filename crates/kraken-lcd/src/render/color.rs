//! `act_color` and friends. The ramp lives in [`llama_core::color`] so the
//! LCD writer and the RGB writer (llama-light) share one definition.

pub use llama_core::color::{ACT_STOPS, BB_STOPS, BLACK, WHITE, act_color, hex, mix};
