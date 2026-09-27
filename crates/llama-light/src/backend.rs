//! The device seam the service drives. One implementation per device.

use llama_core::color::Rgb;

/// A lighting device.
///
/// `probe` finds and opens the device; an `Err` means "absent" and carries
/// the reason for the one transition log line. `show` sends a frame only
/// when it differs from the last one sent on this open (`Ok(true)`), and
/// `Ok(false)` otherwise. An `Err` from `show` closes the device: the
/// service logs it absent and probes again later.
pub trait Backend {
    /// Short name for log lines.
    fn name(&self) -> &'static str;
    /// Whether the device is open.
    fn is_open(&self) -> bool;
    /// Find and open the device.
    fn probe(&mut self) -> Result<(), String>;
    /// Show `frame` if it changed.
    fn show(&mut self, frame: &[Rgb]) -> Result<bool, String>;
}
