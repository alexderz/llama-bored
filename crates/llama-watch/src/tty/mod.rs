//! Console output: sanitiser, cell grid, and the allowlisted byte emitter.

pub mod chart;
pub mod chat_template;
pub mod ctx_history;
pub mod grid;
pub mod layout;
pub mod sanitize;
pub mod term;

pub use grid::{C16, Cell, Grid};
