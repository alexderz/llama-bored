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
    /// Hand the device back to its own lighting and close it. `Ok(true)`
    /// when something was sent, `Ok(false)` when there is nothing to do
    /// (the default) or the device is absent.
    fn release(&mut self) -> Result<bool, String> {
        Ok(false)
    }
    /// Whether the device is present but only a restart of the unit would
    /// let the process open it.
    fn wants_restart(&self) -> bool {
        false
    }
}
