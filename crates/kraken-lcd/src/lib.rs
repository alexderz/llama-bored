#![forbid(unsafe_code)]

pub mod activity;
pub mod anim;
pub mod config;
pub mod device;
pub mod history;
pub mod policy;
pub mod present;
pub mod render;
pub mod service;
pub mod snapshot_reader;
pub mod tokens;

pub use llama_core::log;

/// Sample types [`present`] still imports on this path.
///
/// The collector lives in `llama-watch`. These are the shared sample types.
pub mod collector {
    pub use llama_core::sample::{AiState, ModelInfo, Snapshot};
}

/// Source-id types tests still name on this path.
///
/// `Roots` lives in `llama-watch`. Its default includes `/proc`, which the
/// writer does not read.
pub mod sources {
    pub use llama_core::sample::{SourceError, SourceId};
}
