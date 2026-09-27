#![forbid(unsafe_code)]

mod cli;
mod render;
mod screen;

pub use cli::{CliError, HELP, Options, device_paths, parse_args};

pub use render::{ENTER, RESTORE, Renderer, TermGuard, crop};
pub use screen::{Cell, Color, DecodeError, Screen, decode_screen, vga_attr};
