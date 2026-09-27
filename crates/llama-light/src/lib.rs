#![forbid(unsafe_code)]
//! llama-light: RGB lighting from the llama-watch snapshot.
//!
//! A separate writer from kraken-lcd, with its own uid and unit. It reads
//! only the published snapshot, drives colour only, and speaks to the ASUS
//! Aura USB controller and the Corsair STRAFE RGB MK.2 keyboard, each
//! through a closed command table that has no save.

pub mod aura;
pub mod backend;
pub mod config;
pub mod hidraw;
pub mod keyboard;
pub mod mapping;
pub mod metric;
pub mod palette;
pub mod service;
pub mod snapshot;

pub use llama_core::log;
