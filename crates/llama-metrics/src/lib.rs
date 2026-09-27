//! llama-metrics: a Prometheus exporter for the llama-watch snapshot.
//!
//! It is the only process in this workspace with a listening socket, and it
//! can reach nothing sensitive: it reads one file, the snapshot, through
//! llama-core's validated parser, and serves `GET /metrics` to an in-process
//! CIDR allowlist. It opens no device, reads no `/proc` or `/sys`, and dials
//! nothing. S16 (`tests/s16_metrics_scan.rs`) fences the source.

#![forbid(unsafe_code)]

pub mod acl;
pub mod config;
pub mod expo;
pub mod http;
pub mod service;
pub mod snapshot;
