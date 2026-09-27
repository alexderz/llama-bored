//! ASUS Aura USB mainboard controller, addressable header 1.

pub mod device;
pub mod proto;

use llama_core::color::Rgb;

use crate::backend::Backend;
use device::{AuraOpener, AuraPort};

/// The Aura backend: open on probe, Direct mode once per open, then one
/// frame per change.
pub struct AuraBackend<O: AuraOpener> {
    opener: O,
    port: Option<O::Port>,
    last_sent: Option<Vec<Rgb>>,
}

impl<O: AuraOpener> AuraBackend<O> {
    /// A closed backend. Nothing is opened until [`Backend::probe`].
    pub fn new(opener: O) -> Self {
        Self {
            opener,
            port: None,
            last_sent: None,
        }
    }

    fn close(&mut self) {
        self.port = None;
        self.last_sent = None;
    }

    fn write_frame(&mut self, frame: &[Rgb]) -> Result<(), String> {
        let reports = proto::frame_reports(frame).map_err(|err| err.to_string())?;
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
        Ok(())
    }
}

impl<O: AuraOpener> Backend for AuraBackend<O> {
    fn name(&self) -> &'static str {
        "aura"
    }

    fn is_open(&self) -> bool {
        self.port.is_some()
    }

    fn probe(&mut self) -> Result<(), String> {
        if self.port.is_some() {
            return Ok(());
        }
        let mut port = self.opener.open().map_err(|err| err.to_string())?;
        let enter = proto::encode(&proto::Cmd::EnterDirect).map_err(|err| err.to_string())?;
        port.send(&enter).map_err(|err| err.to_string())?;
        self.port = Some(port);
        self.last_sent = None;
        Ok(())
    }

    fn show(&mut self, frame: &[Rgb]) -> Result<bool, String> {
        if self.last_sent.as_deref() == Some(frame) {
            return Ok(false);
        }
        self.write_frame(frame).map(|()| true)
    }
}
