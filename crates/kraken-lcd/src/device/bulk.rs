//! Bulk image bytes on interface 0, endpoint `0x02`.
//!
//! [`BulkLink`] lists the device, opens it, claims interface 0, and keeps the
//! interface alive for as long as the endpoint exists. The device handle is
//! dropped after the endpoint is open.

use std::time::Duration;

use super::PortError;

const BULK_ENDPOINT: u8 = 0x02;
const BULK_TIMEOUT: Duration = Duration::from_secs(1);

/// One bulk OUT pipe. Tests supply a fake; production uses [`BulkLink`].
pub trait BulkPort {
    /// Write one transfer. The header is 20 bytes; each pixel transfer is 512.
    fn write_chunk(&mut self, data: &[u8]) -> Result<(), PortError>;
}

/// No bulk claim. The restore path uses this so interface 0 stays untouched.
#[derive(Debug, Default)]
pub struct NoBulk;

impl BulkPort for NoBulk {
    fn write_chunk(&mut self, _data: &[u8]) -> Result<(), PortError> {
        Err(PortError::unavailable("bulk port is closed"))
    }
}

/// The real bulk endpoint. Constructed by [`BulkLink::open`], never by tests.
pub struct BulkLink {
    /// The endpoint keeps the interface claim alive after `open` drops its
    /// own interface handle.
    endpoint: nusb::Endpoint<nusb::transfer::Bulk, nusb::transfer::Out>,
}

impl BulkLink {
    /// Match `busnum` and `devnum`, claim interface 0, and open endpoint `0x02`.
    pub fn open(busnum: u8, devnum: u8) -> Result<Self, PortError> {
        use nusb::MaybeFuture;
        use nusb::transfer::{Bulk, Out};

        let mut listed = nusb::list_devices()
            .wait()
            .map_err(|err| PortError::unavailable(err.to_string()))?;
        let info = listed
            .find(|device| device.busnum() == busnum && device.device_address() == devnum)
            .ok_or_else(|| PortError::unavailable("kraken usb device is not listed"))?;
        let device = info
            .open()
            .wait()
            .map_err(|err| PortError::unavailable(err.to_string()))?;
        let interface = device
            .claim_interface(0)
            .wait()
            .map_err(|err| PortError::unavailable(err.to_string()))?;
        let endpoint = interface
            .endpoint::<Bulk, Out>(BULK_ENDPOINT)
            .map_err(|err| PortError::unavailable(err.to_string()))?;
        // `endpoint` holds its own interface handle, which keeps the claim.
        drop(interface);
        drop(device);
        Ok(Self { endpoint })
    }
}

impl BulkPort for BulkLink {
    fn write_chunk(&mut self, data: &[u8]) -> Result<(), PortError> {
        let completion = self
            .endpoint
            .transfer_blocking(data.to_vec().into(), BULK_TIMEOUT);
        if let Err(err) = &completion.status {
            return Err(PortError::unavailable(err.to_string()));
        }
        if completion.actual_len != data.len() {
            return Err(PortError::unavailable("short bulk transfer"));
        }
        Ok(())
    }
}
