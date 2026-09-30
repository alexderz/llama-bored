//! Console output: sanitiser, cell grid, and the allowlisted byte emitter,
//! and the root pre-step that sets tty11's font and size (`setup`).

pub mod chart;
pub mod chat_template;
pub mod ctx_history;
pub mod grid;
pub mod layout;
pub mod sanitize;
pub mod setup;
pub mod term;

pub use grid::{C16, Cell, Grid};
