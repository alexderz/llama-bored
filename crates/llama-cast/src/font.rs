//! PSF2 console fonts (the llama-hack `.psfu` files). `/dev/vcsa11` stores a
//! glyph index, so the tty source draws by position. The grid file stores a
//! Unicode scalar, so the same font's Unicode table is parsed into a
//! scalar-to-glyph map. A font with no table maps a scalar below its glyph
//! count to itself.

use thiserror::Error;

/// Largest font file read.
pub const MAX_FONT_BYTES: usize = 256 * 1024;
const MAGIC: [u8; 4] = [0x72, 0xb5, 0x4a, 0x86];
const HEADER_BYTES: usize = 32;
/// PSF2 `PSF2_HAS_UNICODE_TABLE`.
const UNICODE_FLAG: usize = 1;
/// Ends one glyph's Unicode list.
const UNICODE_END: u8 = 0xFF;
/// Starts a combining sequence. Those scalars are not mapped: the grid is
/// one scalar per cell.
const UNICODE_SEQ: u8 = 0xFE;

/// Why a font was refused.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum FontError {
    #[error("font is larger than 256 KiB")]
    TooLarge,
    #[error("font is not PSF2")]
    Magic,
    #[error("font header is inconsistent: {0}")]
    Header(&'static str),
    #[error("font unicode table is inconsistent: {0}")]
    Unicode(&'static str),
}

/// A parsed PSF2 font.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Psf2 {
    width: usize,
    height: usize,
    count: usize,
    row_bytes: usize,
    glyphs: Vec<u8>,
    /// Sorted `(scalar, glyph)`. `None` means the font has no Unicode table.
    unicode: Option<Vec<(u32, u16)>>,
}

fn word(bytes: &[u8], at: usize) -> usize {
    let raw = u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    usize::try_from(raw).unwrap_or(usize::MAX)
}

impl Psf2 {
    /// Parse and validate. A truncated or malformed Unicode table is an
    /// error; bytes after a complete table are ignored. No input panics.
    pub fn parse(bytes: &[u8]) -> Result<Self, FontError> {
        if bytes.len() > MAX_FONT_BYTES {
            return Err(FontError::TooLarge);
        }
        if bytes.len() < HEADER_BYTES || bytes[..4] != MAGIC {
            return Err(FontError::Magic);
        }
        let header = word(bytes, 8);
        let flags = word(bytes, 12);
        let count = word(bytes, 16);
        let char_size = word(bytes, 20);
        let height = word(bytes, 24);
        let width = word(bytes, 28);
        if !(HEADER_BYTES..=bytes.len()).contains(&header) {
            return Err(FontError::Header("header size"));
        }
        if !(1..=512).contains(&count) {
            return Err(FontError::Header("glyph count"));
        }
        if !(1..=64).contains(&width) || !(1..=64).contains(&height) {
            return Err(FontError::Header("glyph size"));
        }
        let row_bytes = width.div_ceil(8);
        if char_size != row_bytes * height {
            return Err(FontError::Header("bytes per glyph"));
        }
        let end = header + count * char_size;
        if end > bytes.len() {
            return Err(FontError::Header("glyphs past the end"));
        }
        let unicode = if flags & UNICODE_FLAG == 0 {
            None
        } else {
            Some(parse_unicode(&bytes[end..], count)?)
        };
        Ok(Self {
            width,
            height,
            count,
            row_bytes,
            glyphs: bytes[header..end].to_vec(),
            unicode,
        })
    }

    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of glyphs (256 or 512 for console fonts).
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// True when pixel (`x`, `y`) of glyph `index` is set. An index past the
    /// end has no pixels.
    #[must_use]
    pub fn pixel(&self, index: usize, x: usize, y: usize) -> bool {
        if index >= self.count || x >= self.width || y >= self.height {
            return false;
        }
        let row = index * self.row_bytes * self.height + y * self.row_bytes;
        let byte = self.glyphs[row + x / 8];
        byte & (0x80 >> (x % 8)) != 0
    }

    /// Glyph index for `scalar`. The first mapping in the table wins. A font
    /// with no table maps scalars below [`Self::count`] to themselves.
    #[must_use]
    pub fn glyph_for(&self, scalar: u32) -> Option<usize> {
        match &self.unicode {
            Some(map) => map
                .binary_search_by_key(&scalar, |entry| entry.0)
                .ok()
                .map(|at| usize::from(map[at].1)),
            None if u32::try_from(self.count).is_ok_and(|count| scalar < count) => {
                Some(usize::try_from(scalar).unwrap_or(0))
            }
            None => None,
        }
    }

    /// Glyph for `scalar`, or U+FFFD when the font has that glyph and not
    /// `scalar`. `None` means draw a blank cell: the font has neither.
    #[must_use]
    pub fn glyph_for_cell(&self, scalar: u32) -> Option<usize> {
        if let Some(index) = self.glyph_for(scalar) {
            return Some(index);
        }
        if scalar == u32::from('\u{FFFD}') {
            return None;
        }
        self.glyph_for(u32::from('\u{FFFD}'))
    }
}

/// One Unicode scalar, or why these bytes are not one.
fn one_scalar(bytes: &[u8]) -> Result<(u32, usize), FontError> {
    let Some(&lead) = bytes.first() else {
        return Err(FontError::Unicode("truncated"));
    };
    if lead < 0x80 {
        return Ok((u32::from(lead), 1));
    }
    let (need, min, mut scalar) = if lead & 0xE0 == 0xC0 {
        (2, 0x80_u32, u32::from(lead & 0x1F))
    } else if lead & 0xF0 == 0xE0 {
        (3, 0x800, u32::from(lead & 0x0F))
    } else if lead & 0xF8 == 0xF0 {
        (4, 0x1_0000, u32::from(lead & 0x07))
    } else {
        return Err(FontError::Unicode("bad utf-8"));
    };
    if bytes.len() < need {
        return Err(FontError::Unicode("truncated"));
    }
    for &byte in &bytes[1..need] {
        if byte & 0xC0 != 0x80 {
            return Err(FontError::Unicode("bad utf-8"));
        }
        scalar = (scalar << 6) | u32::from(byte & 0x3F);
    }
    if scalar < min || scalar > 0x10_FFFF || (0xD800..=0xDFFF).contains(&scalar) {
        return Err(FontError::Unicode("bad utf-8"));
    }
    Ok((scalar, need))
}

/// Skip a combining sequence. `0xFF` ends the glyph's list. Another `0xFE`
/// starts a further sequence, still unmapped.
fn skip_sequence(table: &[u8], i: &mut usize) -> Result<(), FontError> {
    loop {
        if *i >= table.len() {
            return Err(FontError::Unicode("truncated"));
        }
        match table[*i] {
            UNICODE_END => {
                *i += 1;
                return Ok(());
            }
            UNICODE_SEQ => *i += 1,
            _ => {
                let (_, len) = one_scalar(&table[*i..])?;
                *i += len;
            }
        }
    }
}

fn remember(map: &mut Vec<(u32, u16)>, scalar: u32, glyph: u16) {
    if let Err(at) = map.binary_search_by_key(&scalar, |entry| entry.0) {
        map.insert(at, (scalar, glyph));
    }
}

fn parse_unicode(table: &[u8], count: usize) -> Result<Vec<(u32, u16)>, FontError> {
    let mut map = Vec::new();
    let mut i = 0;
    for glyph in 0..count {
        let glyph = u16::try_from(glyph).unwrap_or(u16::MAX);
        loop {
            if i >= table.len() {
                return Err(FontError::Unicode("truncated"));
            }
            match table[i] {
                UNICODE_END => {
                    i += 1;
                    break;
                }
                UNICODE_SEQ => {
                    i += 1;
                    skip_sequence(table, &mut i)?;
                    break;
                }
                _ => {
                    let (scalar, len) = one_scalar(&table[i..])?;
                    i += len;
                    remember(&mut map, scalar, glyph);
                }
            }
        }
    }
    Ok(map)
}
