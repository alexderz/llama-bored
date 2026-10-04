//! Console output: sanitiser, cell grid, and the allowlisted byte emitter,
//! the root pre-step that sets tty11's font and size (`setup`), and the
//! writer thread that keeps the tick loop off the console (`writer`, #42).

pub mod chart;
pub mod chat_template;
pub mod ctx_history;
pub mod grid;
pub mod layout;
pub mod sanitize;
pub mod setup;
pub mod term;
pub mod writer;

pub use grid::{C16, Cell, Grid};
