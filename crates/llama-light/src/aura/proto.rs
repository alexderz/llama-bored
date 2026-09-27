//! Closed encoder for the ASUS Aura USB mainboard controller (`0b05:18f3`).
//!
//! Written from protocol facts, not from anyone's code. What we rely on:
//!
//! - HID output reports are 65 bytes: report id `0xEC`, then 64 bytes,
//!   zero-filled (OpenRGB wiki, "ASUS Aura USB" page; liquidctl's
//!   `docs/asus-aura-led-guide.md` and its driver notes: 65-byte reports,
//!   `0xEC` prefix).
//! - `EC 35 <effect-channel> 00 00 <mode>` sets the effect of one channel.
//!   Mode `0xFF` is Direct (host-streamed colours). The addressable header 1
//!   is effect channel `0x01` (the mainboard's 12 V headers are `0x00`).
//! - `EC 40 <flags|direct-channel> <start> <count> <R G B> * count` streams
//!   up to 20 LEDs. Addressable header 1 is direct channel `0x00`. Bit
//!   `0x80` in byte 2 marks the last report of a frame ("apply").
//! - `EC 3F 55` commits the current effect to the controller's flash
//!   ("save"), and `EC 3E ..` writes header configuration (Gen1/Gen2).
//!   Neither is ever built here. `EC 36`, `EC B0` and `EC 82` are not needed
//!   and are not built either.
//! - Direct frames and effect changes without a commit live in RAM: on
//!   the test board the stored effect came back after a power cycle
//!   (on-device findings, 2026-09-26).
//!
//! The encoder can emit exactly two opcodes: `0x35` with mode `0xFF` on
//! channel `0x01`, and `0x40` on direct channel `0x00`. Every report is
//! checked against that table before it exists.

use llama_core::color::Rgb;
use thiserror::Error;

/// Bytes per HID output report, including the report id.
pub const REPORT_LEN: usize = 65;
/// Report id of every Aura USB report.
pub const REPORT_ID: u8 = 0xEC;
/// Set-effect opcode. Only ever sent with [`MODE_DIRECT`].
pub const OP_EFFECT: u8 = 0x35;
/// Direct-colour opcode.
pub const OP_DIRECT: u8 = 0x40;
/// The closed opcode table: the only second bytes a report may carry.
pub const OPCODES: [u8; 2] = [OP_EFFECT, OP_DIRECT];
/// Effect mode "Direct".
pub const MODE_DIRECT: u8 = 0xFF;
/// Effect channel of addressable header 1.
pub const ARGB1_EFFECT_CHANNEL: u8 = 0x01;
/// Direct channel of addressable header 1.
pub const ARGB1_DIRECT_CHANNEL: u8 = 0x00;
/// Byte-2 flag on the last direct report of a frame.
pub const APPLY: u8 = 0x80;
/// LEDs per direct report: 5 header bytes plus 20 × 3 fit in 65.
pub const LEDS_PER_REPORT: usize = 20;
/// Largest LED index range this writer addresses on the header.
pub const MAX_LEDS: usize = 120;

/// A report that failed the opcode table.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[error("report is outside the Aura command allowlist")]
pub struct FenceViolation;

/// One Aura command. These variants are the only bytes `encode` can emit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cmd<'a> {
    /// `EC 35 01 00 00 FF`: header 1 into Direct mode. RAM only.
    EnterDirect,
    /// `EC 40 <apply|00> <start> <n> <rgb…>`: `leds` (1..=20) from `start`.
    Direct {
        /// First LED index.
        start: u8,
        /// Colours, 1..=[`LEDS_PER_REPORT`].
        leds: &'a [Rgb],
        /// Last report of the frame.
        apply: bool,
    },
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

        /// The bytes written to hidraw, report id first.
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

/// The byte-level allowlist.
fn allowed(bytes: &[u8; REPORT_LEN]) -> bool {
    if bytes[0] != REPORT_ID || !OPCODES.contains(&bytes[1]) {
        return false;
    }
    match bytes[1] {
        OP_EFFECT => {
            bytes[2] == ARGB1_EFFECT_CHANNEL
                && bytes[3] == 0
                && bytes[4] == 0
                && bytes[5] == MODE_DIRECT
                && bytes[6..].iter().all(|byte| *byte == 0)
        }
        OP_DIRECT => {
            let channel_ok =
                bytes[2] == ARGB1_DIRECT_CHANNEL || bytes[2] == APPLY | ARGB1_DIRECT_CHANNEL;
            let start = usize::from(bytes[3]);
            let count = usize::from(bytes[4]);
            channel_ok
                && (1..=LEDS_PER_REPORT).contains(&count)
                && start + count <= MAX_LEDS
                && bytes[5 + count * 3..].iter().all(|byte| *byte == 0)
        }
        _ => false,
    }
}

/// A command whose fields are out of range.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum EncodeError {
    /// 0 or more than [`LEDS_PER_REPORT`] LEDs, or past [`MAX_LEDS`].
    #[error("direct report LED range is out of bounds")]
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
        Cmd::EnterDirect => {
            bytes[1] = OP_EFFECT;
            bytes[2] = ARGB1_EFFECT_CHANNEL;
            bytes[5] = MODE_DIRECT;
        }
        Cmd::Direct { start, leds, apply } => {
            let count = leds.len();
            if !(1..=LEDS_PER_REPORT).contains(&count) || usize::from(*start) + count > MAX_LEDS {
                return Err(EncodeError::Range);
            }
            bytes[1] = OP_DIRECT;
            bytes[2] = if *apply {
                APPLY | ARGB1_DIRECT_CHANNEL
            } else {
                ARGB1_DIRECT_CHANNEL
            };
            bytes[3] = *start;
            bytes[4] = u8::try_from(count).map_err(|_| EncodeError::Range)?;
            for (index, led) in leds.iter().enumerate() {
                let at = 5 + index * 3;
                bytes[at] = led.r;
                bytes[at + 1] = led.g;
                bytes[at + 2] = led.b;
            }
        }
    }
    Ok(EncodedReport::from_checked(bytes)?)
}

/// Every report for one frame on header 1, in order. The last carries
/// [`APPLY`]. An empty frame, or one longer than [`MAX_LEDS`], is an error.
pub fn frame_reports(frame: &[Rgb]) -> Result<Vec<EncodedReport>, EncodeError> {
    if frame.is_empty() || frame.len() > MAX_LEDS {
        return Err(EncodeError::Range);
    }
    let chunks: Vec<&[Rgb]> = frame.chunks(LEDS_PER_REPORT).collect();
    let last = chunks.len() - 1;
    let mut out = Vec::with_capacity(chunks.len());
    for (index, chunk) in chunks.into_iter().enumerate() {
        let start = u8::try_from(index * LEDS_PER_REPORT).map_err(|_| EncodeError::Range)?;
        out.push(encode(&Cmd::Direct {
            start,
            leds: chunk,
            apply: index == last,
        })?);
    }
    Ok(out)
}
