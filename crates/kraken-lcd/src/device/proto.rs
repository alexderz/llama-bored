//! Closed encoder for the nine LCD commands.
//!
//! Absent from this module: init (`0x70`), pump or fan duty (`0x72`),
//! brightness and orientation writes, gif bulk type, the built-in animation
//! whose third byte is `01` after `38 01`, and firmware.

use thiserror::Error;

use crate::render::{FRAME_H, FRAME_W, Frame};

/// HID report length. Every encoded command is zero-padded to this size.
pub const REPORT_LEN: usize = 64;

/// Packed frame size: 320 × 320 pixels, four bytes each.
pub const FRAME_BYTES: usize = {
    let width = FRAME_W as usize;
    let height = FRAME_H as usize;
    width * height * 4
};

/// Bulk-transfer magic from the static-image upload capture.
pub const BULK_MAGIC: [u8; 12] = [
    0x12, 0xFA, 0x01, 0xE8, 0xAB, 0xCD, 0xEF, 0x98, 0x76, 0x54, 0x32, 0x10,
];

/// Twenty-byte bulk header: magic, static-image type, then `FRAME_BYTES` little-endian.
pub const BULK_HEADER: [u8; 20] = [
    0x12, 0xFA, 0x01, 0xE8, 0xAB, 0xCD, 0xEF, 0x98, 0x76, 0x54, 0x32, 0x10, 0x02, 0x00, 0x00, 0x00,
    0x00, 0x40, 0x06, 0x00,
];

/// Bucket size of one packed frame, in 1 KiB units (`ceil((20 + FRAME_BYTES) / 1024)`).
pub const SLOT_UNITS: u16 = 401;

/// Slots in the upload ring.
pub const SLOT_COUNT: u8 = 8;

/// Buckets the device reports. Queries and deletes address all of them.
pub const BUCKET_COUNT: u8 = 16;

/// Memory our slots occupy, in 1 KiB units: `[0, 3208)`.
pub const SLOT_REGION: std::ops::Range<u16> = 0..((SLOT_COUNT as u16) * SLOT_UNITS);

/// Two-byte prefixes `encode` will emit, excluding the `38 01` switch forms.
///
/// `38 01` is allowed only when its third byte is `02` or `04`.
pub const ALLOWED: [[u8; 2]; 7] = [
    [0x30, 0x01],
    [0x30, 0x04],
    [0x32, 0x01],
    [0x32, 0x02],
    [0x36, 0x01],
    [0x36, 0x02],
    [0x36, 0x03],
];

/// A bucket or slot index was outside its closed range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("index is outside the LCD command range")]
pub struct IndexError;

/// A report prefix is not one of the nine LCD commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("command prefix is outside the LCD allowlist")]
pub struct FenceViolation;

/// A reply was too short, or its orientation byte was outside `0..=3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ParseError {
    #[error("reply is shorter than the parsed field")]
    Truncated,
    #[error("orientation byte is outside 0..=3")]
    Orientation,
}

/// `pack` was given an angle or a frame it cannot encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PackError {
    #[error("orientation must be 0..=3")]
    Orientation,
    #[error("rotate_deg must be 0, 90, 180, or 270")]
    RotateDeg,
    #[error("frame must be 320 by 320")]
    FrameSize,
}

/// Bucket index in `0..BUCKET_COUNT` (queries and deletes).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BucketId(u8);

impl BucketId {
    /// Bucket `id` in `0..BUCKET_COUNT`.
    pub const fn try_new(id: u8) -> Result<Self, IndexError> {
        if id < BUCKET_COUNT {
            Ok(Self(id))
        } else {
            Err(IndexError)
        }
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

/// Upload-ring slot in `0..SLOT_COUNT`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SlotId(u8);

impl SlotId {
    /// Ring slot `id` in `0..SLOT_COUNT`.
    pub const fn try_new(id: u8) -> Result<Self, IndexError> {
        if id < SLOT_COUNT {
            Ok(Self(id))
        } else {
            Err(IndexError)
        }
    }

    pub const fn get(self) -> u8 {
        self.0
    }

    /// The next slot after this one. Slot 7 wraps to slot 0.
    pub const fn next(self) -> Self {
        Self((self.0 + 1) % SLOT_COUNT)
    }

    /// Fixed start address of this slot, in 1 KiB units.
    pub const fn start_units(self) -> u16 {
        (self.0 as u16) * SLOT_UNITS
    }
}

/// One LCD command. These nine variants are the only bytes `encode` can emit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cmd {
    LcdInfo,
    PreTransfer,
    QueryBucket(BucketId),
    DeleteBucket(BucketId),
    SetupBucket { slot: SlotId },
    WriteStart(SlotId),
    WriteEnd,
    ShowSlot(SlotId),
    ShowLiquid,
}

/// Fields read from an LCD-info reply after its `31 01` prefix matches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LcdInfoReply {
    /// Read-only level at reply index 0x18.
    ///
    /// Brightness writes are absent from this module.
    pub brightness: u8,
    /// Quarter-turns from reply index 0x1A, range 0..=3.
    pub orientation: u8,
}

/// One bucket-query reply, with addresses in 1 KiB units.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BucketTable {
    pub empty: bool,
    /// Start address in 1 KiB units, bytes 17–18 little-endian.
    pub start_kib: u16,
    /// Size in 1 KiB units, bytes 19–20 little-endian.
    pub size_kib: u16,
}

mod report {
    /// A 64-byte HID report. Only [`encode`](super::encode) can build one.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct EncodedReport([u8; super::REPORT_LEN]);

    impl EncodedReport {
        pub(super) fn from_checked(
            bytes: [u8; super::REPORT_LEN],
        ) -> Result<Self, super::FenceViolation> {
            if !super::prefix_allowed(&bytes) {
                return Err(super::FenceViolation);
            }
            Ok(Self(bytes))
        }

        pub fn as_bytes(&self) -> &[u8; super::REPORT_LEN] {
            &self.0
        }

        /// Test-only constructor. Prefixes outside the allowlist are rejected.
        #[cfg(test)]
        pub(super) fn try_from_raw(
            bytes: [u8; super::REPORT_LEN],
        ) -> Result<Self, super::FenceViolation> {
            Self::from_checked(bytes)
        }
    }
}

pub use report::EncodedReport;

fn prefix_allowed(bytes: &[u8; REPORT_LEN]) -> bool {
    let pair = [bytes[0], bytes[1]];
    if ALLOWED.contains(&pair) {
        return true;
    }
    bytes[0] == 0x38 && bytes[1] == 0x01 && matches!(bytes[2], 0x02 | 0x04)
}

fn layout(cmd: &Cmd) -> [u8; REPORT_LEN] {
    let mut bytes = [0u8; REPORT_LEN];
    match cmd {
        Cmd::LcdInfo => {
            bytes[0] = 0x30;
            bytes[1] = 0x01;
        }
        Cmd::PreTransfer => {
            bytes[0] = 0x36;
            bytes[1] = 0x03;
        }
        Cmd::QueryBucket(id) => {
            bytes[0] = 0x30;
            bytes[1] = 0x04;
            bytes[2] = id.get();
        }
        Cmd::DeleteBucket(id) => {
            bytes[0] = 0x32;
            bytes[1] = 0x02;
            bytes[2] = id.get();
        }
        Cmd::SetupBucket { slot } => {
            let id = slot.get();
            let address = slot.start_units().to_le_bytes();
            let units = SLOT_UNITS.to_le_bytes();
            bytes[0] = 0x32;
            bytes[1] = 0x01;
            bytes[2] = id;
            bytes[3] = id + 1;
            bytes[4] = address[0];
            bytes[5] = address[1];
            bytes[6] = units[0];
            bytes[7] = units[1];
            bytes[8] = 0x01;
        }
        Cmd::WriteStart(slot) => {
            bytes[0] = 0x36;
            bytes[1] = 0x01;
            bytes[2] = slot.get();
        }
        Cmd::WriteEnd => {
            bytes[0] = 0x36;
            bytes[1] = 0x02;
        }
        Cmd::ShowSlot(slot) => {
            bytes[0] = 0x38;
            bytes[1] = 0x01;
            bytes[2] = 0x04;
            bytes[3] = slot.get();
        }
        Cmd::ShowLiquid => {
            bytes[0] = 0x38;
            bytes[1] = 0x01;
            bytes[2] = 0x02;
            bytes[3] = 0x00;
        }
    }
    bytes
}

/// Encode one LCD command into a 64-byte report.
///
/// The bytes are checked against [`ALLOWED`] (and the `38 01` third-byte
/// rule) before the report is returned. Every [`Cmd`] passes that check.
pub fn encode(cmd: &Cmd) -> Result<EncodedReport, FenceViolation> {
    EncodedReport::from_checked(layout(cmd))
}

/// First two bytes of the reply that belongs to `cmd`.
#[must_use]
pub fn expected_prefix(cmd: &Cmd) -> [u8; 2] {
    match cmd {
        Cmd::LcdInfo => [0x31, 0x01],
        Cmd::PreTransfer => [0x37, 0x03],
        Cmd::QueryBucket(_) => [0x31, 0x04],
        Cmd::DeleteBucket(_) => [0x33, 0x02],
        Cmd::SetupBucket { .. } => [0x33, 0x01],
        Cmd::WriteStart(_) => [0x37, 0x01],
        Cmd::WriteEnd => [0x37, 0x02],
        Cmd::ShowSlot(_) | Cmd::ShowLiquid => [0x39, 0x01],
    }
}

/// Whether `reply` begins with [`expected_prefix`] for `cmd`.
#[must_use]
pub fn prefix_matches(cmd: &Cmd, reply: &[u8]) -> bool {
    let prefix = expected_prefix(cmd);
    reply.len() >= prefix.len() && reply[0] == prefix[0] && reply[1] == prefix[1]
}

/// `true` when reply byte 14 is `0x01`.
pub fn success_flag(reply: &[u8]) -> Result<bool, ParseError> {
    match reply.get(14) {
        Some(0x01) => Ok(true),
        Some(_) => Ok(false),
        None => Err(ParseError::Truncated),
    }
}

/// Whether this command's success check is reply byte 14, not the prefix alone.
#[must_use]
pub fn expects_success_flag(cmd: &Cmd) -> bool {
    matches!(
        cmd,
        Cmd::DeleteBucket(_) | Cmd::SetupBucket { .. } | Cmd::ShowSlot(_) | Cmd::ShowLiquid
    )
}

/// Prefix match, plus byte 14 when [`expects_success_flag`] is set.
#[must_use]
pub fn reply_ok(cmd: &Cmd, reply: &[u8]) -> bool {
    if !prefix_matches(cmd, reply) {
        return false;
    }
    if expects_success_flag(cmd) {
        return matches!(success_flag(reply), Ok(true));
    }
    true
}

/// Read byte 0x18 and the orientation at byte 0x1A.
///
/// Callers match the `31 01` prefix first. Orientation outside `0..=3` is an error.
pub fn parse_lcd_info(reply: &[u8]) -> Result<LcdInfoReply, ParseError> {
    let Some(brightness) = reply.get(0x18).copied() else {
        return Err(ParseError::Truncated);
    };
    let Some(orientation) = reply.get(0x1A).copied() else {
        return Err(ParseError::Truncated);
    };
    if orientation > 3 {
        return Err(ParseError::Orientation);
    }
    Ok(LcdInfoReply {
        brightness,
        orientation,
    })
}

/// Parse a bucket-query reply.
///
/// The bucket is empty when every byte from index 15 onward is zero.
/// Bytes 17–18 and 19–20 are the start and size, little-endian, in 1 KiB units.
///
/// The 21-byte length check is what makes those fixed indexes safe.
pub fn parse_bucket_table(reply: &[u8]) -> Result<BucketTable, ParseError> {
    if reply.len() < 21 {
        return Err(ParseError::Truncated);
    }
    let empty = reply[15..].iter().all(|byte| *byte == 0);
    Ok(BucketTable {
        empty,
        start_kib: u16::from_le_bytes([reply[17], reply[18]]),
        size_kib: u16::from_le_bytes([reply[19], reply[20]]),
    })
}

fn quarter_turns(orientation: u8, rotate_deg: u16) -> Result<u8, PackError> {
    if orientation > 3 {
        return Err(PackError::Orientation);
    }
    let extra = match rotate_deg {
        0 => 0,
        90 => 1,
        180 => 2,
        270 => 3,
        _ => return Err(PackError::RotateDeg),
    };
    Ok((orientation + extra) % 4)
}

/// Pack `frame` as row-major R, G, B, `0x00`.
///
/// Pixmap samples are premultiplied. Each pixel is demultiplied first, so a
/// partial alpha does not darken the color. A fully transparent pixel is
/// black. The fourth byte is still zero.
///
/// Rotation is clockwise. `orientation` (0..=3 quarter-turns, from the LCD
/// info reply) and `rotate_deg` (0, 90, 180, or 270) are added, then wrapped
/// to a full turn.
pub fn pack(frame: &Frame, orientation: u8, rotate_deg: u16) -> Result<Vec<u8>, PackError> {
    let turns = quarter_turns(orientation, rotate_deg)?;
    if frame.0.width() != FRAME_W || frame.0.height() != FRAME_H {
        return Err(PackError::FrameSize);
    }
    let pixels = frame.0.pixels();
    if pixels.len() != (FRAME_W as usize) * (FRAME_H as usize) {
        return Err(PackError::FrameSize);
    }
    let mut out = vec![0u8; FRAME_BYTES];
    let last = FRAME_W - 1;
    for y in 0..FRAME_H {
        for x in 0..FRAME_W {
            let (src_x, src_y) = match turns {
                0 => (x, y),
                1 => (y, last - x),
                2 => (last - x, last - y),
                _ => (last - y, x),
            };
            let src = (src_y * FRAME_W + src_x) as usize;
            let dst = ((y * FRAME_W + x) * 4) as usize;
            let (red, green, blue) = straight_rgb(&pixels[src]);
            out[dst] = red;
            out[dst + 1] = green;
            out[dst + 2] = blue;
        }
    }
    Ok(out)
}

fn straight_rgb(pixel: &tiny_skia::PremultipliedColorU8) -> (u8, u8, u8) {
    if pixel.alpha() == 0 {
        return (0, 0, 0);
    }
    let color = pixel.demultiply();
    (color.red(), color.green(), color.blue())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiny_skia::{Color, Pixmap, PremultipliedColorU8};

    fn slot(id: u8) -> SlotId {
        SlotId::try_new(id).expect("slot")
    }

    fn bucket(id: u8) -> BucketId {
        BucketId::try_new(id).expect("bucket")
    }

    fn raw(prefix: &[u8]) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[..prefix.len()].copy_from_slice(prefix);
        out
    }

    fn absent_opcode(bytes: &[u8; 64]) -> bool {
        // forbidden opcodes: must be rejected
        bytes[0] == 0x70
            || bytes[0] == 0x72
            || (bytes[0] == 0x30 && bytes[1] == 0x02)
            || (bytes[0] == 0x38 && bytes[1] == 0x01 && bytes[2] == 0x01)
    }

    fn allowed_prefix(bytes: &[u8; 64]) -> bool {
        ALLOWED.contains(&[bytes[0], bytes[1]])
            || (bytes[0] == 0x38 && bytes[1] == 0x01 && matches!(bytes[2], 0x02 | 0x04))
    }

    #[track_caller]
    fn assert_report(cmd: &Cmd, prefix: &[u8]) {
        let encoded = encode(cmd).expect("in-table command encodes");
        let bytes = encoded.as_bytes();
        assert_eq!(bytes.len(), 64);
        assert_eq!(&bytes[..prefix.len()], prefix, "{cmd:?}");
        assert!(
            bytes[prefix.len()..].iter().all(|b| *b == 0),
            "{cmd:?} must be zero-padded"
        );
        assert!(allowed_prefix(bytes), "{cmd:?} prefix not allowed");
        assert!(!absent_opcode(bytes), "{cmd:?} uses an absent opcode");
    }

    fn all_cmds() -> Vec<Cmd> {
        let mut cmds = vec![
            Cmd::LcdInfo,
            Cmd::PreTransfer,
            Cmd::WriteEnd,
            Cmd::ShowLiquid,
        ];
        for id in 0..=15 {
            cmds.push(Cmd::QueryBucket(bucket(id)));
            cmds.push(Cmd::DeleteBucket(bucket(id)));
        }
        for id in 0..=7 {
            let slot = slot(id);
            cmds.push(Cmd::SetupBucket { slot });
            cmds.push(Cmd::WriteStart(slot));
            cmds.push(Cmd::ShowSlot(slot));
        }
        cmds
    }

    #[test]
    fn constants_match_the_lcd_table() {
        assert_eq!(REPORT_LEN, 64);
        assert_eq!(FRAME_BYTES, 409_600);
        assert_eq!(FRAME_BYTES, (FRAME_W * FRAME_H * 4) as usize);
        assert_eq!(
            BULK_MAGIC,
            [
                0x12, 0xFA, 0x01, 0xE8, 0xAB, 0xCD, 0xEF, 0x98, 0x76, 0x54, 0x32, 0x10
            ]
        );
        assert_eq!(BULK_HEADER.len(), 20);
        assert_eq!(&BULK_HEADER[..12], &BULK_MAGIC);
        assert_eq!(&BULK_HEADER[12..16], &[0x02, 0x00, 0x00, 0x00]);
        assert_eq!(&BULK_HEADER[16..], &(FRAME_BYTES as u32).to_le_bytes());
        assert_eq!(&BULK_HEADER[16..], &[0x00, 0x40, 0x06, 0x00]);
        assert_eq!(SLOT_UNITS, 401);
        assert_eq!(SLOT_COUNT, 8);
        assert_eq!(BUCKET_COUNT, 16);
        assert_eq!(SLOT_REGION, 0..3208);
        assert_eq!(
            u16::from(SLOT_COUNT) * SLOT_UNITS,
            SLOT_REGION.end - SLOT_REGION.start
        );
        assert_eq!(
            ALLOWED,
            [
                [0x30, 0x01],
                [0x30, 0x04],
                [0x32, 0x01],
                [0x32, 0x02],
                [0x36, 0x01],
                [0x36, 0x02],
                [0x36, 0x03],
            ]
        );
        // forbidden opcodes: must be rejected
        assert!(
            ALLOWED
                .iter()
                .all(|pair| pair[0] != 0x70 && pair[0] != 0x72)
        );
        assert!(!ALLOWED.contains(&[0x30, 0x02]));
        assert!(!ALLOWED.contains(&[0x38, 0x01]));
    }

    #[test]
    fn bucket_and_slot_ids_cover_only_the_closed_ranges() {
        for id in 0..=15 {
            assert_eq!(bucket(id).get(), id);
        }
        assert!(BucketId::try_new(16).is_err());
        for id in 0..=7 {
            assert_eq!(slot(id).get(), id);
        }
        assert!(SlotId::try_new(8).is_err());
    }

    #[test]
    fn slots_sit_at_401_unit_steps_inside_the_region() {
        let starts = [0u16, 401, 802, 1203, 1604, 2005, 2406, 2807];
        for (id, expected) in starts.iter().copied().enumerate() {
            let slot = slot(id as u8);
            assert_eq!(slot.start_units(), expected);
            assert!(SLOT_REGION.contains(&expected));
            let end = u32::from(slot.start_units()) + 401;
            assert!(end <= u32::from(SLOT_REGION.end));
        }
        assert_eq!(slot(7).start_units() + 401, 3208);
    }

    #[test]
    fn next_slot_rotates_0_through_7_and_back() {
        let mut seen = Vec::new();
        let mut current = slot(0);
        for _ in 0..8 {
            seen.push(current.get());
            let advanced = current.next();
            assert_ne!(advanced.get(), current.get());
            current = advanced;
        }
        assert_eq!(seen, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(current.get(), 0);
    }

    #[test]
    fn each_command_matches_the_poc_byte_layout() {
        assert_report(&Cmd::LcdInfo, &[0x30, 0x01]);
        assert_report(&Cmd::PreTransfer, &[0x36, 0x03]);
        assert_report(&Cmd::WriteEnd, &[0x36, 0x02]);
        assert_report(&Cmd::ShowLiquid, &[0x38, 0x01, 0x02, 0x00]);
        for id in 0..=15 {
            assert_report(&Cmd::QueryBucket(bucket(id)), &[0x30, 0x04, id]);
            assert_report(&Cmd::DeleteBucket(bucket(id)), &[0x32, 0x02, id]);
        }
        for id in 0..=7 {
            assert_report(&Cmd::WriteStart(slot(id)), &[0x36, 0x01, id]);
            assert_report(&Cmd::ShowSlot(slot(id)), &[0x38, 0x01, 0x04, id]);
            let start = u16::from(id) * 401;
            let le = start.to_le_bytes();
            assert_report(
                &Cmd::SetupBucket { slot: slot(id) },
                &[0x32, 0x01, id, id + 1, le[0], le[1], 0x91, 0x01, 0x01],
            );
        }
    }

    #[test]
    fn setup_bucket_encodes_only_the_slot_start() {
        for id in 0..=7 {
            let slot = slot(id);
            let report = encode(&Cmd::SetupBucket { slot }).expect("setup");
            let bytes = report.as_bytes();
            let start = u16::from_le_bytes([bytes[4], bytes[5]]);
            assert_eq!(start, slot.start_units());
            assert_eq!(start, u16::from(id) * 401);
        }
        // 0x1234 is an out-of-layout address. SetupBucket has no start field,
        // so that address cannot be constructed or encoded.
        let report = encode(&Cmd::SetupBucket { slot: slot(1) }).expect("slot 1");
        let start = u16::from_le_bytes([report.as_bytes()[4], report.as_bytes()[5]]);
        assert_ne!(start, 0x1234);
    }

    #[test]
    fn every_constructible_cmd_encodes_inside_the_allowlist() {
        let cmds = all_cmds();
        assert_eq!(cmds.len(), 4 + 16 * 2 + 8 * 3);
        for cmd in &cmds {
            let bytes = *encode(cmd).expect("in-table").as_bytes();
            assert_eq!(bytes.len(), 64);
            assert!(
                matches!(bytes[0], 0x30 | 0x32 | 0x36 | 0x38),
                "{cmd:?} encoded {bytes:02x?}"
            );
            assert!(allowed_prefix(&bytes), "{cmd:?}");
            assert!(!absent_opcode(&bytes), "{cmd:?}");
        }
    }

    #[test]
    fn out_of_table_prefix_is_a_fence_violation() {
        // forbidden opcodes: must be rejected
        assert_eq!(
            EncodedReport::try_from_raw(raw(&[0x70, 0x01])),
            Err(FenceViolation)
        );
        assert_eq!(
            EncodedReport::try_from_raw(raw(&[0x70, 0x02])),
            Err(FenceViolation)
        );
        assert_eq!(
            EncodedReport::try_from_raw(raw(&[0x72, 0x00])),
            Err(FenceViolation)
        );
        assert_eq!(
            EncodedReport::try_from_raw(raw(&[0x30, 0x02, 0x01])),
            Err(FenceViolation)
        );
        assert_eq!(
            EncodedReport::try_from_raw(raw(&[0x38, 0x01, 0x01, 0x00])),
            Err(FenceViolation)
        );
        for third in [0x00, 0x01, 0x03, 0x05, 0xff] {
            assert_eq!(
                EncodedReport::try_from_raw(raw(&[0x38, 0x01, third])),
                Err(FenceViolation),
                "third {third:#04x}"
            );
        }
        assert_eq!(
            EncodedReport::try_from_raw(raw(&[0x31, 0x01])),
            Err(FenceViolation)
        );
        assert!(EncodedReport::try_from_raw(raw(&[0x38, 0x01, 0x02])).is_ok());
        assert!(EncodedReport::try_from_raw(raw(&[0x38, 0x01, 0x04])).is_ok());
        for pair in ALLOWED {
            assert!(
                EncodedReport::try_from_raw(raw(&pair)).is_ok(),
                "{pair:02x?}"
            );
        }
    }

    #[test]
    fn reply_prefixes_and_success_flag_follow_the_table() {
        let cases = [
            (Cmd::LcdInfo, [0x31, 0x01], false),
            (Cmd::PreTransfer, [0x37, 0x03], false),
            (Cmd::QueryBucket(bucket(0)), [0x31, 0x04], false),
            (Cmd::QueryBucket(bucket(15)), [0x31, 0x04], false),
            (Cmd::DeleteBucket(bucket(0)), [0x33, 0x02], true),
            (Cmd::DeleteBucket(bucket(15)), [0x33, 0x02], true),
            (Cmd::SetupBucket { slot: slot(0) }, [0x33, 0x01], true),
            (Cmd::WriteStart(slot(3)), [0x37, 0x01], false),
            (Cmd::WriteEnd, [0x37, 0x02], false),
            (Cmd::ShowSlot(slot(7)), [0x39, 0x01], true),
            (Cmd::ShowLiquid, [0x39, 0x01], true),
        ];
        for (cmd, prefix, flag) in cases {
            assert_eq!(expected_prefix(&cmd), prefix, "{cmd:?}");
            assert!(prefix_matches(&cmd, &prefix), "{cmd:?}");
            assert!(!prefix_matches(&cmd, &[prefix[0]]), "{cmd:?} short");
            assert!(
                !prefix_matches(&cmd, &[prefix[0], prefix[1].wrapping_add(1)]),
                "{cmd:?} mismatch"
            );
            assert_eq!(expects_success_flag(&cmd), flag, "{cmd:?}");
        }

        let mut ok = [0u8; 64];
        ok[0] = 0x33;
        ok[1] = 0x02;
        ok[14] = 0x01;
        let delete = Cmd::DeleteBucket(bucket(2));
        assert_eq!(success_flag(&ok), Ok(true));
        assert!(reply_ok(&delete, &ok));

        ok[14] = 0x00;
        assert_eq!(success_flag(&ok), Ok(false));
        assert!(!reply_ok(&delete, &ok));

        ok[14] = 0x09;
        assert_eq!(success_flag(&ok), Ok(false));
        assert!(!reply_ok(&delete, &ok));

        assert_eq!(success_flag(&[0u8; 14]), Err(ParseError::Truncated));
        assert!(!reply_ok(&delete, &[0x33, 0x02]));

        let mut write_end = [0u8; 64];
        write_end[0] = 0x37;
        write_end[1] = 0x02;
        assert!(reply_ok(&Cmd::WriteEnd, &write_end));
        assert!(!reply_ok(&Cmd::WriteEnd, &ok));

        let mut info = [0u8; 64];
        info[0] = 0x31;
        info[1] = 0x01;
        assert!(reply_ok(&Cmd::LcdInfo, &info));
        assert!(!reply_ok(&Cmd::ShowLiquid, &info));
    }

    #[test]
    fn lcd_info_reads_byte_18_and_orientation_0_to_3() {
        let mut reply = [0u8; 64];
        reply[0x18] = 0x32;
        for orientation in 0..=3 {
            reply[0x1A] = orientation;
            let parsed = parse_lcd_info(&reply).expect("orientation in range");
            assert_eq!(parsed.brightness, 0x32);
            assert_eq!(parsed.orientation, orientation);
        }
        reply[0x18] = 0x00;
        reply[0x1A] = 0;
        assert_eq!(parse_lcd_info(&reply).expect("zero").brightness, 0x00);
        reply[0x18] = 0x64;
        reply[0x1A] = 3;
        let parsed = parse_lcd_info(&reply).expect("top of range");
        assert_eq!(parsed.brightness, 0x64);
        assert_eq!(parsed.orientation, 3);

        reply[0x1A] = 4;
        assert_eq!(parse_lcd_info(&reply), Err(ParseError::Orientation));
        reply[0x1A] = 255;
        assert_eq!(parse_lcd_info(&reply), Err(ParseError::Orientation));
        assert_eq!(parse_lcd_info(&[0u8; 26]), Err(ParseError::Truncated));
        let mut exact = [0u8; 27];
        exact[0x18] = 0x32;
        exact[0x1A] = 1;
        assert_eq!(
            parse_lcd_info(&exact).expect("27 bytes"),
            LcdInfoReply {
                brightness: 0x32,
                orientation: 1,
            }
        );
    }

    #[test]
    fn bucket_table_empty_when_tail_is_zero_and_addresses_are_le_kib() {
        let mut empty = [0u8; 64];
        empty[0] = 0x31;
        empty[1] = 0x04;
        empty[14] = 0x01;
        assert_eq!(
            parse_bucket_table(&empty).expect("empty"),
            BucketTable {
                empty: true,
                start_kib: 0,
                size_kib: 0,
            }
        );

        let mut occupied = [0u8; 64];
        occupied[15] = 0x02;
        occupied[16] = 0x03;
        occupied[17] = 0x91;
        occupied[18] = 0x01;
        occupied[19] = 0x91;
        occupied[20] = 0x01;
        assert_eq!(
            parse_bucket_table(&occupied).expect("occupied"),
            BucketTable {
                empty: false,
                start_kib: 401,
                size_kib: 401,
            }
        );

        occupied[17] = 0xF7;
        occupied[18] = 0x0A;
        occupied[19] = 0x34;
        occupied[20] = 0x12;
        let parsed = parse_bucket_table(&occupied).expect("endian");
        assert!(!parsed.empty);
        assert_eq!(parsed.start_kib, 0x0AF7);
        assert_eq!(parsed.size_kib, 0x1234);

        let mut late = [0u8; 64];
        late[63] = 0x01;
        assert_eq!(
            parse_bucket_table(&late).expect("late occupancy"),
            BucketTable {
                empty: false,
                start_kib: 0,
                size_kib: 0,
            }
        );

        assert_eq!(parse_bucket_table(&[0u8; 20]), Err(ParseError::Truncated));
        let mut bare = [0u8; 21];
        assert!(parse_bucket_table(&bare).expect("21 zero bytes").empty);
        bare[15] = 0x01;
        bare[17] = 0x02;
        bare[18] = 0x00;
        bare[19] = 0x03;
        bare[20] = 0x00;
        assert_eq!(
            parse_bucket_table(&bare).expect("21 occupied"),
            BucketTable {
                empty: false,
                start_kib: 2,
                size_kib: 3,
            }
        );
    }

    fn paint(frame: &mut Frame, x: u32, y: u32, rgb: [u8; 3]) {
        let idx = (y * FRAME_W + x) as usize;
        frame.0.pixels_mut()[idx] =
            PremultipliedColorU8::from_rgba(rgb[0], rgb[1], rgb[2], 255).expect("opaque color");
    }

    fn marked_frame() -> Frame {
        let mut frame = Frame::new();
        frame.0.fill(Color::from_rgba8(0, 0, 0, 255));
        paint(&mut frame, 0, 0, [255, 0, 0]);
        paint(&mut frame, 1, 0, [0, 255, 0]);
        paint(&mut frame, 0, 1, [0, 0, 255]);
        paint(&mut frame, 319, 319, [255, 255, 255]);
        frame
    }

    fn pixel(buf: &[u8], at: (u32, u32)) -> [u8; 4] {
        let index = ((at.1 * 320 + at.0) * 4) as usize;
        [buf[index], buf[index + 1], buf[index + 2], buf[index + 3]]
    }

    #[test]
    fn pack_is_rgb_zero_row_major_after_clockwise_rotation() {
        let frame = marked_frame();
        let cases = [
            (0, 0, (0, 0), (1, 0), (0, 1), (319, 319)),
            (0, 90, (319, 0), (319, 1), (318, 0), (0, 319)),
            (0, 180, (319, 319), (318, 319), (319, 318), (0, 0)),
            (0, 270, (0, 319), (0, 318), (1, 319), (319, 0)),
            (1, 0, (319, 0), (319, 1), (318, 0), (0, 319)),
            (1, 90, (319, 319), (318, 319), (319, 318), (0, 0)),
            (1, 180, (0, 319), (0, 318), (1, 319), (319, 0)),
            (1, 270, (0, 0), (1, 0), (0, 1), (319, 319)),
            (2, 0, (319, 319), (318, 319), (319, 318), (0, 0)),
            (2, 90, (0, 319), (0, 318), (1, 319), (319, 0)),
            (2, 180, (0, 0), (1, 0), (0, 1), (319, 319)),
            (2, 270, (319, 0), (319, 1), (318, 0), (0, 319)),
            (3, 0, (0, 319), (0, 318), (1, 319), (319, 0)),
            (3, 90, (0, 0), (1, 0), (0, 1), (319, 319)),
            (3, 180, (319, 0), (319, 1), (318, 0), (0, 319)),
            (3, 270, (319, 319), (318, 319), (319, 318), (0, 0)),
        ];
        for (orientation, rotate_deg, red, green, blue, white) in cases {
            let buf = pack(&frame, orientation, rotate_deg).expect("angles in range");
            assert_eq!(buf.len(), 409_600, "ori {orientation} deg {rotate_deg}");
            assert!(
                buf.iter().skip(3).step_by(4).all(|b| *b == 0),
                "ori {orientation} deg {rotate_deg} fourth byte"
            );
            assert_eq!(
                pixel(&buf, red),
                [255, 0, 0, 0],
                "red {orientation} {rotate_deg}"
            );
            assert_eq!(
                pixel(&buf, green),
                [0, 255, 0, 0],
                "green {orientation} {rotate_deg}"
            );
            assert_eq!(
                pixel(&buf, blue),
                [0, 0, 255, 0],
                "blue {orientation} {rotate_deg}"
            );
            assert_eq!(
                pixel(&buf, white),
                [255, 255, 255, 0],
                "white {orientation} {rotate_deg}"
            );
        }
        let identity = pack(&frame, 0, 0).expect("identity");
        assert_eq!(pixel(&identity, (2, 2)), [0, 0, 0, 0]);
    }

    #[test]
    fn pack_demultiplies_partial_alpha_before_zeroing_the_fourth_byte() {
        let mut frame = Frame::new();
        frame.0.fill(Color::from_rgba8(0, 0, 0, 255));
        let pixels = frame.0.pixels_mut();
        pixels[0] = PremultipliedColorU8::from_rgba(128, 0, 0, 128).expect("half red");
        pixels[1] = PremultipliedColorU8::from_rgba(153, 99, 54, 180).expect("partial");
        pixels[2] = PremultipliedColorU8::from_rgba(0, 0, 0, 0).expect("clear");
        let buf = pack(&frame, 0, 0).expect("identity");
        assert_eq!(pixel(&buf, (0, 0)), [255, 0, 0, 0]);
        assert_eq!(pixel(&buf, (1, 0)), [217, 140, 77, 0]);
        assert_eq!(pixel(&buf, (2, 0)), [0, 0, 0, 0]);
    }

    #[test]
    fn pack_rejects_orientation_outside_0_to_3() {
        let frame = marked_frame();
        assert_eq!(pack(&frame, 4, 0), Err(PackError::Orientation));
        assert_eq!(pack(&frame, 255, 90), Err(PackError::Orientation));
    }

    #[test]
    fn pack_rejects_rotate_deg_not_a_quarter_turn() {
        let frame = marked_frame();
        assert_eq!(pack(&frame, 0, 45), Err(PackError::RotateDeg));
        assert_eq!(pack(&frame, 1, 360), Err(PackError::RotateDeg));
        assert_eq!(pack(&frame, 0, 1), Err(PackError::RotateDeg));
    }

    #[test]
    fn pack_rejects_a_frame_that_is_not_320_square() {
        let pixmap = Pixmap::new(16, 16).expect("small pixmap");
        let frame = Frame(pixmap);
        assert_eq!(pack(&frame, 0, 0), Err(PackError::FrameSize));
    }
}
