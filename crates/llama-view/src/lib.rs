#![forbid(unsafe_code)]

mod cli;
mod color;
mod input;
mod pane;
mod render;
mod screen;
mod session;

pub use cli::{
    CliError, ColorSettings, HELP, Options, color_settings, device_paths, parse_args,
    tmux_window_name as tmux_window_wanted,
};

pub use color::{ColorChoice, ColorMode, SgrTable, nearest_xterm256, xterm256_rgb};
pub use input::{RawInput, Wait, raw, restored, wait_input};
pub use llama_core::palette::Palette;
pub use pane::{
    Fit, HOST_CAP, InputEvent, InputParser, MODEL_CAP, Titler, frame_period, header_model,
    osc_title, place, sanitize, title_text, tmux_window_name,
};
pub use render::{ENTER, FOCUS_ON, RESTORE, Renderer, SYNC_BEGIN, SYNC_END, TermGuard, crop};
pub use screen::{
    AttrLayout, Cell, Color, DecodeError, MAX_VCSA_BYTES, Screen, decode_screen,
    decode_screen_with, detect_layout, vcsa_geometry, vga_attr, vga_attr_512,
};
pub use session::{Next, Session};
