//! Closed encoder for the Corsair "legacy" K-series / STRAFE lighting
//! protocol (STRAFE RGB MK.2, `1b1c:1b48`).
//!
//! Written from protocol descriptions, not from anyone's code (OpenRGB and
//! ckb-next are GPL; this crate is MIT). What we rely on:
//!
//! - Packets are 64 bytes on the vendor HID interface (usage page 0xFFC2),
//!   zero-filled. The interface uses no report ids, so a hidraw write is
//!   `0x00` then the 64 bytes (Linux `Documentation/hid/hidraw.rst`: the
//!   first byte of a write is the report number, 0 when unnumbered).
//! - Byte 0 is the command: `0x07` write a property, `0x0E` read one,
//!   `0x7F` stream a data buffer. Byte 1 of a write is the property
//!   (ckb-next's protocol notes and OpenRGB's "Corsair Peripheral" wiki
//!   page describe the same command table).
//! - `07 05 <mode> 00 03`: lighting control. Mode `0x02` is software
//!   (host-streamed colours), `0x01` is hardware (the keyboard's own
//!   lighting from its stored profile). It is a RAM mode switch: it is what
//!   a host driver sends when it takes the keyboard and when it lets go,
//!   and it stores nothing.
//! - `7F <n> <len> 00 <data…>`: stream buffer packet `n` (1-based) of
//!   `len` data bytes. One colour channel is 144 bytes, one per LED slot:
//!   packets 1 and 2 carry 60 bytes (`0x3C`), packet 3 carries 24 (`0x18`).
//! - `07 28 <channel> 03 <flag>`: take the 3 streamed packets as 24-bit
//!   colour channel `<channel>` (1 red, 2 green, 3 blue). `<flag>` is `0x01`
//!   for red and green and `0x02` on blue, the last one, which applies the
//!   frame.
//! - On-device (OpenRGB 1.0 bench, 2026-09-27, firmware 3.36): Direct mode
//!   with per-key colours shows correctly, so the keyboard takes 24-bit
//!   per-key colour on this interface.
//!
//! Never built here, and refused by the allowlist: `07 02` (reset, which
//! can drop into the bootloader), `07 04` (special-function control: moves
//! the brightness and Windows-lock keys into software), `07 0C`/`07 0D`
//! (firmware update), `07 13`, `07 14`, `07 15`, `07 16` and `07 17`
//! (hardware profile and stored lighting writes), `07 40` (per-key input
//! routing, which can stop keys typing), `07 27` (9-bit colour commit),
//! `07 0A` (poll rate), and every `0x0E` read.
//!
//! The encoder can emit exactly four shapes: software mode, hardware mode,
//! a stream packet 1..=3 of the fixed lengths, and a 24-bit channel commit.
//! Every report is checked against that table before it exists.

use llama_core::color::Rgb;
use thiserror::Error;

use super::keymap::LED_SLOTS;

/// Bytes per hidraw write: the `0x00` report number, then 64.
pub const REPORT_LEN: usize = 65;
/// Report number written first: the interface has none.
pub const REPORT_ID: u8 = 0x00;
/// Write-property command.
pub const OP_WRITE: u8 = 0x07;
/// Stream-buffer command.
pub const OP_STREAM: u8 = 0x7F;
/// The closed command table: the only second bytes a report may carry.
pub const OPCODES: [u8; 2] = [OP_WRITE, OP_STREAM];
/// Lighting-control property.
pub const PROP_LIGHTING: u8 = 0x05;
/// 24-bit colour-channel commit property.
pub const PROP_COMMIT: u8 = 0x28;
/// The closed property table for [`OP_WRITE`].
pub const PROPERTIES: [u8; 2] = [PROP_LIGHTING, PROP_COMMIT];
/// Lighting control: the keyboard's own (hardware) lighting.
pub const MODE_HARDWARE: u8 = 0x01;
/// Lighting control: host-streamed (software) colours.
pub const MODE_SOFTWARE: u8 = 0x02;
/// Fixed fourth byte of a lighting-control write.
pub const LIGHTING_ARG: u8 = 0x03;
/// Stream packets per colour channel.
pub const PACKETS: u8 = 3;
/// Data bytes in stream packets 1, 2 and 3.
pub const PACKET_LEN: [usize; 3] = [60, 60, 24];
/// Commit flag on red and green.
pub const COMMIT_MORE: u8 = 0x01;
/// Commit flag on blue, the last channel: apply the frame.
pub const COMMIT_APPLY: u8 = 0x02;

/// A colour channel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Channel {
    Red = 1,
    Green = 2,
    Blue = 3,
}

/// A report that failed the table.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[error("report is outside the keyboard command allowlist")]
pub struct FenceViolation;

/// One keyboard command. These variants are the only bytes `encode` can emit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cmd<'a> {
    /// `07 05 02 00 03`: host-streamed colours. RAM only.
    SoftwareMode,
    /// `07 05 01 00 03`: back to the keyboard's own lighting. RAM only.
    HardwareMode,
    /// `7F <packet> <len> 00 <data>`: part of one channel's 144 bytes.
    Stream {
        /// 1..=3.
        packet: u8,
        /// Exactly `PACKET_LEN[packet - 1]` bytes.
        data: &'a [u8],
    },
    /// `07 28 <channel> 03 <01|02>`: commit the streamed channel. Blue
    /// commits with [`COMMIT_APPLY`], the others with [`COMMIT_MORE`].
    Commit(Channel),
}

mod report {
    /// A 65-byte report. Only [`encode`](super::encode) can build one.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct EncodedReport([u8; super::REPORT_LEN]);

    impl EncodedReport {
        pub(super) fn from_checked(
            bytes: [u8; super::REPORT_LEN],
        ) -> Result<Self, super::FenceViolation> {
            if !super::allowed(&bytes) {
                return Err(super::FenceViolation);
            }
            Ok(Self(bytes))
        }

        /// The bytes written to hidraw, report number first.
        #[must_use]
        pub fn as_bytes(&self) -> &[u8; super::REPORT_LEN] {
            &self.0
        }
    }
}

pub use report::EncodedReport;

/// Run raw bytes through the allowlist. Tests use this to show that any
/// report outside the table is refused.
pub fn check_raw(bytes: [u8; REPORT_LEN]) -> Result<EncodedReport, FenceViolation> {
    EncodedReport::from_checked(bytes)
}

fn zero_from(bytes: &[u8; REPORT_LEN], at: usize) -> bool {
    bytes[at..].iter().all(|byte| *byte == 0)
}

/// The byte-level allowlist.
fn allowed(bytes: &[u8; REPORT_LEN]) -> bool {
    if bytes[0] != REPORT_ID || !OPCODES.contains(&bytes[1]) {
        return false;
    }
    match bytes[1] {
        OP_WRITE => match bytes[2] {
            PROP_LIGHTING => {
                (bytes[3] == MODE_SOFTWARE || bytes[3] == MODE_HARDWARE)
                    && bytes[4] == 0
                    && bytes[5] == LIGHTING_ARG
                    && zero_from(bytes, 6)
            }
            PROP_COMMIT => {
                let flag_ok = match bytes[3] {
                    1 | 2 => bytes[5] == COMMIT_MORE,
                    3 => bytes[5] == COMMIT_APPLY,
                    _ => false,
                };
                flag_ok && bytes[4] == PACKETS && zero_from(bytes, 6)
            }
            _ => false,
        },
        OP_STREAM => {
            let packet = usize::from(bytes[2]);
            if !(1..=PACKET_LEN.len()).contains(&packet) {
                return false;
            }
            let len = PACKET_LEN[packet - 1];
            usize::from(bytes[3]) == len && bytes[4] == 0 && zero_from(bytes, 5 + len)
        }
        _ => false,
    }
}

/// A command whose fields are out of range.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum EncodeError {
    /// A stream packet number or length that is not in the table.
    #[error("stream packet is out of bounds")]
    Range,
    /// The bytes failed the allowlist.
    #[error(transparent)]
    Fence(#[from] FenceViolation),
}

/// Encode one command.
pub fn encode(cmd: &Cmd<'_>) -> Result<EncodedReport, EncodeError> {
    let mut bytes = [0u8; REPORT_LEN];
    bytes[0] = REPORT_ID;
    match cmd {
        Cmd::SoftwareMode | Cmd::HardwareMode => {
            bytes[1] = OP_WRITE;
            bytes[2] = PROP_LIGHTING;
            bytes[3] = if *cmd == Cmd::SoftwareMode {
                MODE_SOFTWARE
            } else {
                MODE_HARDWARE
            };
            bytes[5] = LIGHTING_ARG;
        }
        Cmd::Stream { packet, data } => {
            let index = usize::from(*packet);
            if !(1..=PACKET_LEN.len()).contains(&index) || data.len() != PACKET_LEN[index - 1] {
                return Err(EncodeError::Range);
            }
            bytes[1] = OP_STREAM;
            bytes[2] = *packet;
            bytes[3] = u8::try_from(data.len()).map_err(|_| EncodeError::Range)?;
            bytes[5..5 + data.len()].copy_from_slice(data);
        }
        Cmd::Commit(channel) => {
            bytes[1] = OP_WRITE;
            bytes[2] = PROP_COMMIT;
            bytes[3] = *channel as u8;
            bytes[4] = PACKETS;
            bytes[5] = if *channel == Channel::Blue {
                COMMIT_APPLY
            } else {
                COMMIT_MORE
            };
        }
    }
    Ok(EncodedReport::from_checked(bytes)?)
}

/// The 12 reports for one frame of [`LED_SLOTS`] colours: for red, green
/// and blue in turn, three stream packets and a commit. The blue commit
/// applies the frame.
pub fn frame_reports(slots: &[Rgb; LED_SLOTS]) -> Result<Vec<EncodedReport>, EncodeError> {
    let mut out = Vec::with_capacity(12);
    for channel in [Channel::Red, Channel::Green, Channel::Blue] {
        let values: Vec<u8> = slots
            .iter()
            .map(|led| match channel {
                Channel::Red => led.r,
                Channel::Green => led.g,
                Channel::Blue => led.b,
            })
            .collect();
        let mut at = 0;
        for (index, len) in PACKET_LEN.iter().enumerate() {
            let packet = u8::try_from(index + 1).map_err(|_| EncodeError::Range)?;
            out.push(encode(&Cmd::Stream {
                packet,
                data: &values[at..at + len],
            })?);
            at += len;
        }
        out.push(encode(&Cmd::Commit(channel))?);
    }
    Ok(out)
}
