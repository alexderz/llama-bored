//! PSF2 console fonts (the llama-hack `.psfu` files tty11 uses). Glyphs
//! are indexed by font position, which is what `/dev/vcsa11` stores; the
//! Unicode table after the glyphs is not needed and not read.

use thiserror::Error;

/// Largest font file read.
pub const MAX_FONT_BYTES: usize = 256 * 1024;
const MAGIC: [u8; 4] = [0x72, 0xb5, 0x4a, 0x86];
const HEADER_BYTES: usize = 32;

/// Why a font was refused.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum FontError {
    #[error("font is larger than 256 KiB")]
    TooLarge,
    #[error("font is not PSF2")]
    Magic,
    #[error("font header is inconsistent: {0}")]
    Header(&'static str),
}

/// A parsed PSF2 font.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Psf2 {
    width: usize,
    height: usize,
    count: usize,
    row_bytes: usize,
    glyphs: Vec<u8>,
}

fn word(bytes: &[u8], at: usize) -> usize {
    let raw = u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    usize::try_from(raw).unwrap_or(usize::MAX)
}

impl Psf2 {
    /// Parse and validate.
    pub fn parse(bytes: &[u8]) -> Result<Self, FontError> {
        if bytes.len() > MAX_FONT_BYTES {
            return Err(FontError::TooLarge);
        }
        if bytes.len() < HEADER_BYTES || bytes[..4] != MAGIC {
            return Err(FontError::Magic);
        }
        let header = word(bytes, 8);
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
        Ok(Self {
            width,
            height,
            count,
            row_bytes,
            glyphs: bytes[header..end].to_vec(),
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
}
