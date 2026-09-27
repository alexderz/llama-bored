//! Corsair STRAFE RGB MK.2 (`1b1c:1b48`): per-key colour.
//!
//! Open on probe and put the keyboard in software lighting mode, then one
//! frame (12 reports) per change. [`Backend::release`] hands the keyboard
//! back to its own (hardware) lighting, which it keeps showing without
//! llama-light. Nothing is ever saved to the keyboard.

pub mod device;
pub mod keymap;
pub mod proto;

use llama_core::color::{BLACK, Rgb};

use crate::backend::Backend;
use device::{KeyboardOpener, KeyboardPort, OpenError};
use keymap::{KEYS, LED_SLOTS};

pub use keymap::key_index;

/// The 144 channel slots for a frame in [`KEYS`] order. Slots with no named
/// key are black. A frame of the wrong length is an error.
pub fn slots(frame: &[Rgb]) -> Result<[Rgb; LED_SLOTS], String> {
    if frame.len() != KEYS.len() {
        return Err(format!(
            "keyboard frame has {} keys, expected {}",
            frame.len(),
            KEYS.len()
        ));
    }
    let mut out = [BLACK; LED_SLOTS];
    for (key, color) in KEYS.iter().zip(frame) {
        out[usize::from(key.led)] = *color;
    }
    Ok(out)
}

/// The keyboard backend.
pub struct KeyboardBackend<O: KeyboardOpener> {
    opener: O,
    port: Option<O::Port>,
    last_sent: Option<Vec<Rgb>>,
    restart: bool,
}

impl<O: KeyboardOpener> KeyboardBackend<O> {
    /// A closed backend. Nothing is opened until [`Backend::probe`].
    pub fn new(opener: O) -> Self {
        Self {
            opener,
            port: None,
            last_sent: None,
            restart: false,
        }
    }

    fn close(&mut self) {
        self.port = None;
        self.last_sent = None;
    }

    fn open(&mut self) -> Result<O::Port, String> {
        self.opener.open().map_err(|err| {
            if err == OpenError::NeedsRestart {
                self.restart = true;
            }
            err.to_string()
        })
    }
}

impl<O: KeyboardOpener> Backend for KeyboardBackend<O> {
    fn name(&self) -> &'static str {
        "keyboard"
    }

    fn is_open(&self) -> bool {
        self.port.is_some()
    }

    fn probe(&mut self) -> Result<(), String> {
        if self.port.is_some() {
            return Ok(());
        }
        let mut port = self.open()?;
        let enter = proto::encode(&proto::Cmd::SoftwareMode).map_err(|err| err.to_string())?;
        port.send(&enter).map_err(|err| err.to_string())?;
        self.port = Some(port);
        self.last_sent = None;
        Ok(())
    }

    fn show(&mut self, frame: &[Rgb]) -> Result<bool, String> {
        if self.last_sent.as_deref() == Some(frame) {
            return Ok(false);
        }
        let reports = proto::frame_reports(&slots(frame)?).map_err(|err| err.to_string())?;
        let Some(port) = self.port.as_mut() else {
            return Err("not open".to_owned());
        };
        for report in &reports {
            if let Err(err) = port.send(report) {
                self.close();
                return Err(err.to_string());
            }
        }
        self.last_sent = Some(frame.to_vec());
        Ok(true)
    }

    /// Hardware lighting mode, then close. Opens the node for this if it is
    /// closed, without entering software mode first. `Ok(false)`: absent.
    fn release(&mut self) -> Result<bool, String> {
        let mut port = match self.port.take() {
            Some(port) => port,
            None => match self.open() {
                Ok(port) => port,
                Err(_) => {
                    self.close();
                    return Ok(false);
                }
            },
        };
        self.close();
        let back = proto::encode(&proto::Cmd::HardwareMode).map_err(|err| err.to_string())?;
        port.send(&back).map_err(|err| err.to_string())?;
        Ok(true)
    }

    fn wants_restart(&self) -> bool {
        self.restart
    }
}
