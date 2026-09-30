//! llama-cast: the tty11 dashboard as a live video on the LAN, over
//! DLNA/UPnP, for smart TVs (Roku Media Player on TCL Roku TVs is tested).
//!
//! It announces one MediaServer over SSDP, answers ContentDirectory Browse
//! with one live `object.item.videoItem`, and streams `/live.ts`: tty11
//! (`/dev/vcsa11`, read-only) rendered with the console font at `fps`,
//! encoded to MPEG-TS by the configured ffmpeg, one encoder per client.
//!
//! It is one of two processes in this workspace with a listening socket
//! (llama-metrics is the other). It binds one TCP listener and one UDP
//! socket on port 1900, checks a CIDR allowlist in process before reading a
//! byte, reads only its config, `/dev/vcsa11`, the font and
//! `/etc/machine-id`, writes no file, opens no other device, dials nothing,
//! and spawns only ffmpeg. S18 (`tests/s18_cast_scan.rs`) fences the source.

#![forbid(unsafe_code)]

pub mod acl;
pub mod config;
pub mod discovery;
pub mod dlna;
pub mod encoder;
pub mod font;
pub mod http;
pub mod render;
pub mod service;
pub mod sha256;
pub mod source;
pub mod ssdp;
