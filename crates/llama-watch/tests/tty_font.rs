//! The committed tty11 font draws every scalar `term.rs` may emit.
//!
//! Parses the PSF2 Unicode table of `packaging/fonts/llama-hack-12x24.psfu`.
//! `packaging/fonts/build-psf.py --self-test` (run from `scripts/check.sh`)
//! checks that the file is a byte-for-byte rebuild from Hack.

use std::collections::BTreeSet;
use std::path::PathBuf;

use llama_watch::tty::term::GLYPHS;

const PSF2_MAGIC: u32 = 0x864A_B572;
const HAS_UNICODE_TABLE: u32 = 1;

struct Psf {
    width: u32,
    height: u32,
    glyphs: u32,
    /// Every scalar the Unicode table maps to some glyph.
    mapped: BTreeSet<char>,
}

fn word(bytes: &[u8], index: usize) -> u32 {
    let at = index * 4;
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"))
}

fn parse_psf2(bytes: &[u8]) -> Result<Psf, String> {
    if bytes.len() < 32 {
        return Err("shorter than a PSF2 header".into());
    }
    if word(bytes, 0) != PSF2_MAGIC {
        return Err("not PSF2".into());
    }
    let header = word(bytes, 2) as usize;
    let flags = word(bytes, 3);
    let glyphs = word(bytes, 4);
    let per_glyph = word(bytes, 5) as usize;
    let height = word(bytes, 6);
    let width = word(bytes, 7);
    if flags & HAS_UNICODE_TABLE == 0 {
        return Err("no Unicode table".into());
    }
    let table_at = header + glyphs as usize * per_glyph;
    let table = bytes
        .get(table_at..)
        .ok_or_else(|| "glyph bitmaps run past the end".to_string())?;
    let mut mapped = BTreeSet::new();
    let mut rest = table;
    // One entry per glyph, ended by 0xFF. Before the first 0xFE the entry is
    // single scalars; after it come combining sequences, which the tty never
    // emits and which do not count as coverage.
    for entry in 0..glyphs {
        let end = rest
            .iter()
            .position(|b| *b == 0xFF)
            .ok_or_else(|| format!("table entry {entry} of {glyphs} is not terminated"))?;
        let singles = rest[..end].split(|b| *b == 0xFE).next().unwrap_or(&[]);
        let text = std::str::from_utf8(singles)
            .map_err(|err| format!("entry {entry} is not UTF-8: {err}"))?;
        mapped.extend(text.chars());
        rest = &rest[end + 1..];
    }
    if !rest.is_empty() {
        return Err(format!("{} bytes after the last table entry", rest.len()));
    }
    Ok(Psf {
        width,
        height,
        glyphs,
        mapped,
    })
}

fn font() -> Psf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packaging/fonts/llama-hack-12x24.psfu");
    let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    parse_psf2(&bytes).unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

#[test]
fn committed_font_is_a_12x24_psf2_with_a_unicode_table() {
    let psf = font();
    assert_eq!((psf.width, psf.height), (12, 24));
    assert_eq!(psf.glyphs, 512);
}

/// Every printable ASCII byte, `?` (the S12 replacement) and every
/// `term::GLYPHS` scalar must map to a glyph, or tty11 shows tofu.
#[test]
fn committed_font_has_every_glyph_term_may_emit() {
    let psf = font();
    let missing: Vec<char> = ('\u{20}'..='\u{7e}')
        .chain(GLYPHS.iter().copied())
        .filter(|ch| !psf.mapped.contains(ch))
        .collect();
    assert!(
        missing.is_empty(),
        "llama-hack-12x24.psfu lacks {missing:?}"
    );
}

#[test]
fn parser_rejects_a_table_that_is_short_an_entry() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packaging/fonts/llama-hack-12x24.psfu");
    let mut bytes = std::fs::read(&path).expect("font");
    let last = bytes.iter().rposition(|b| *b == 0xFF).expect("terminator");
    bytes.truncate(last);
    assert!(parse_psf2(&bytes).is_err());
}
