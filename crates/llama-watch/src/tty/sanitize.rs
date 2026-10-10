//! Llama text to cells. Bytes are not emitted here.
//!
//! #90: model text is full of typographic Unicode (curly quotes, en/em
//! dashes, an ellipsis, a bullet, arrows) and accented Latin. Before this,
//! every one of those collapsed to a literal `?`, so prose read "Calliope?s"
//! and a real `?` could not be told apart from a missing glyph. [`sanitize`]
//! now runs one transliteration pass before the old fallback:
//!
//! 1. A small table turns common typography into either the real glyph (when
//!    [`super::term::GLYPHS`] and all three committed consoles fonts already
//!    have it — curly quotes, en/em dash, the ellipsis, the bullet) or a
//!    single ASCII character that keeps the column count exact.
//! 2. Latin-1 Supplement and Latin Extended-A letters with diacritics become
//!    their base ASCII letter: the same result `t.chars().nfkd()` plus
//!    dropping combining marks would give, written out by hand (see
//!    [`latin_base_letter`]) rather than pulling in `unicode-normalization`.
//!    A handful of letters in that range (`Æ ß Ð Ø Þ Đ Ħ Ł Ŋ Œ Ŧ ı ĸ`, …)
//!    have no Unicode decomposition at all; those get the same
//!    one-ASCII-letter treatment as a ligature (below), by hand.
//! 3. `ﬁ`/`ﬂ` become `f`/`l`: the issue allows expanding a ligature to two
//!    cells, but taking the first letter keeps every other rule in this file
//!    ("one cell in, one cell out") true without exception.
//! 4. Zero-width joiners/spacers and the BOM are dropped (zero cells): they
//!    carry no column of their own. Bidi direction overrides are
//!    deliberately NOT here — a Trojan-Source-style reorder stays visible as
//!    the placeholder below, never silently vanishes.
//! 5. Anything left — CJK, emoji, Cyrillic, bidi overrides, standalone
//!    combining marks, symbols this table does not cover — becomes
//!    [`UNKNOWN`], not `?`: a `?` in the input now always means a literal
//!    `?` was there.
//!
//! Printable ASCII is kept. `\n` appends an end-of-line marker and resets the
//! tab column. `\t` expands to the next multiple of 4 columns (1 to 4 spaces).
//! `\r` is dropped and does not move the column. Other C0, DEL, and C1 are
//! dropped. Each call starts at column 0; cells already in `out` stay.

use super::grid::{C16, Cell};

/// Placeholder for a scalar this console cannot show and that
/// [`transliterate`] has no mapping for. The Unicode replacement character
/// (U+FFFD) rather than `·`: it is already in [`super::term::GLYPHS`] (and in
/// all three committed fonts, and in eurlatgr, #90), it is the standard
/// "something was here and this console cannot show it" glyph, and it does
/// not collide with `·`'s other on-screen meanings (the chart dot, the
/// bullet fallback). A real `?` (ASCII 0x3F) is handled by the printable-
/// ASCII arm above and never reaches this constant.
const UNKNOWN: char = '\u{fffd}';

/// Append sanitised cells for `input`.
pub fn sanitize(input: &str, out: &mut Vec<Cell>) {
    let mut col = 0usize;
    for ch in input.chars() {
        match ch {
            '\n' => {
                push(out, '\n');
                col = 0;
            }
            '\t' => {
                let spaces = 4 - (col % 4);
                for _ in 0..spaces {
                    push(out, ' ');
                    col += 1;
                }
            }
            '\r' => {}
            c if ('\u{20}'..='\u{7e}').contains(&c) => {
                push(out, c);
                col += 1;
            }
            // Zero-width format characters: genuinely no column of their
            // own, so dropping them (like `\r` and the controls below)
            // keeps width exact. U+00AD is a hyphenation hint that is
            // normally invisible outside a line break; it is grouped here
            // rather than with the dash table because, unlike a visible
            // dash, it should not appear as anything at all.
            c if dropped(c) => {}
            c => {
                push(out, transliterate(c));
                col += 1;
            }
        }
    }
}

/// Zero-width format characters and controls: [`sanitize`] gives them no
/// cell at all, and neither does [`detail_char`].
fn dropped(c: char) -> bool {
    matches!(
        c,
        '\u{ad}' | '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{feff}'
    ) || c.is_control()
}

/// One scalar of single-line text to its cell, the way [`sanitize`] maps it:
/// `None` for what it drops (zero-width characters and controls, including
/// tab and newline), printable ASCII as itself, anything else through
/// [`transliterate`]. For [`super::layout`]'s detail painters (#90).
pub(crate) fn detail_char(c: char) -> Option<char> {
    if dropped(c) {
        None
    } else if ('\u{20}'..='\u{7e}').contains(&c) {
        Some(c)
    } else {
        Some(transliterate(c))
    }
}

/// One scalar outside ASCII to one cell: a mapped character, or [`UNKNOWN`].
fn transliterate(c: char) -> char {
    typography(c)
        .or_else(|| latin_base_letter(c))
        .unwrap_or(UNKNOWN)
}

/// Common typography, and the handful of symbols the issue calls out by
/// name. Where the real glyph is already allowed ([`super::term::GLYPHS`],
/// backed by all three committed fonts and by eurlatgr — verified by
/// `tests/tty_font.rs` and `tests/tty_term.rs`), it is returned as itself so
/// `'s` reads as `'s` rather than 's. Where it is not, the closest single
/// ASCII character is returned instead; never two characters, so a column is
/// never lost or gained.
fn typography(c: char) -> Option<char> {
    Some(match c {
        // Right/left single quote, single high-reversed-9, prime: the first
        // two are real glyphs; the last two have no font slot.
        '\u{2018}' | '\u{2019}' => c,
        '\u{201b}' | '\u{2032}' => '\'',
        // Left/right double quote: real glyphs. Low-9 and double prime: not.
        '\u{201c}' | '\u{201d}' => c,
        '\u{201e}' | '\u{2033}' => '"',
        // En dash, em dash: real glyphs. Plain hyphen (U+2010, distinct from
        // ASCII `-`), non-breaking hyphen, and the horizontal bar: not.
        '\u{2013}' | '\u{2014}' => c,
        '\u{2010}' | '\u{2011}' | '\u{2015}' => '-',
        // Horizontal ellipsis: already a `term::GLYPHS` entry.
        '\u{2026}' => c,
        // Bullet: also already allowed, so it stays itself rather than
        // falling back to the chart's `·`.
        '\u{2022}' => c,
        // Arrows: none of the three committed fonts draw these, so each
        // becomes the closest single-column ASCII stand-in. The double
        // arrow shares `>` with the single one; there is no single-column
        // "implies" character to tell them apart.
        '\u{2192}' | '\u{21d2}' => '>',
        '\u{2190}' => '<',
        '\u{2191}' => '^',
        '\u{2193}' => 'v',
        // Multiplication sign.
        '\u{d7}' => 'x',
        // NBSP, figure space, thin space, narrow NBSP: all become a plain
        // space, one column, same as the space they stand in for.
        '\u{a0}' | '\u{2007}' | '\u{2009}' | '\u{202f}' => ' ',
        // `ﬁ`/`ﬂ`: the issue's one named exception to single-cell width,
        // taken as "use the first letter only" so every rule above stays a
        // true one-cell mapping (see the module doc).
        '\u{fb01}' => 'f',
        '\u{fb02}' => 'l',
        _ => return None,
    })
}

/// Latin-1 Supplement (U+00C0-00FF) and Latin Extended-A (U+0100-017F)
/// letters, mapped to their base ASCII letter: what
/// `c.nfkd().next().filter(|b| b.is_ascii_alphabetic())` would give for the
/// ~170 of these that have a real Unicode decomposition, written out by hand
/// (`×` U+00D7 and `÷` U+00F7 are not letters and are not here; `×` is
/// handled by [`typography`], `÷` has no mapping and becomes [`UNKNOWN`]).
///
/// A dozen or so letters in these two blocks have no decomposition at all —
/// ligatures (`Æ Œ Ĳ`) and "stroke" letters (`Đ Ħ Ł Ŋ Ŧ Ø`, Icelandic `Ð Þ`,
/// German `ß`, Turkish dotless `ı`) — those get the nearest single ASCII
/// letter by hand, the same "first letter, one cell" rule `typography` uses
/// for `ﬁ`/`ﬂ`.
fn latin_base_letter(c: char) -> Option<char> {
    Some(match c {
        '\u{00c0}' | '\u{00c1}' | '\u{00c2}' | '\u{00c3}' | '\u{00c4}' | '\u{00c5}'
        | '\u{00c6}' => 'A',
        '\u{00c7}' => 'C',
        '\u{00c8}' | '\u{00c9}' | '\u{00ca}' | '\u{00cb}' => 'E',
        '\u{00cc}' | '\u{00cd}' | '\u{00ce}' | '\u{00cf}' => 'I',
        '\u{00d0}' => 'D',
        '\u{00d1}' => 'N',
        '\u{00d2}' | '\u{00d3}' | '\u{00d4}' | '\u{00d5}' | '\u{00d6}' | '\u{00d8}' => 'O',
        '\u{00d9}' | '\u{00da}' | '\u{00db}' | '\u{00dc}' => 'U',
        '\u{00dd}' => 'Y',
        '\u{00de}' => 'T',
        '\u{00df}' => 's',
        '\u{00e0}' | '\u{00e1}' | '\u{00e2}' | '\u{00e3}' | '\u{00e4}' | '\u{00e5}'
        | '\u{00e6}' => 'a',
        '\u{00e7}' => 'c',
        '\u{00e8}' | '\u{00e9}' | '\u{00ea}' | '\u{00eb}' => 'e',
        '\u{00ec}' | '\u{00ed}' | '\u{00ee}' | '\u{00ef}' => 'i',
        '\u{00f0}' => 'd',
        '\u{00f1}' => 'n',
        '\u{00f2}' | '\u{00f3}' | '\u{00f4}' | '\u{00f5}' | '\u{00f6}' | '\u{00f8}' => 'o',
        '\u{00f9}' | '\u{00fa}' | '\u{00fb}' | '\u{00fc}' => 'u',
        '\u{00fd}' | '\u{00ff}' => 'y',
        '\u{00fe}' => 't',
        '\u{0100}' | '\u{0102}' | '\u{0104}' => 'A',
        '\u{0101}' | '\u{0103}' | '\u{0105}' => 'a',
        '\u{0106}' | '\u{0108}' | '\u{010a}' | '\u{010c}' => 'C',
        '\u{0107}' | '\u{0109}' | '\u{010b}' | '\u{010d}' => 'c',
        '\u{010e}' | '\u{0110}' => 'D',
        '\u{010f}' | '\u{0111}' => 'd',
        '\u{0112}' | '\u{0114}' | '\u{0116}' | '\u{0118}' | '\u{011a}' => 'E',
        '\u{0113}' | '\u{0115}' | '\u{0117}' | '\u{0119}' | '\u{011b}' => 'e',
        '\u{011c}' | '\u{011e}' | '\u{0120}' | '\u{0122}' => 'G',
        '\u{011d}' | '\u{011f}' | '\u{0121}' | '\u{0123}' => 'g',
        '\u{0124}' | '\u{0126}' => 'H',
        '\u{0125}' | '\u{0127}' => 'h',
        '\u{0128}' | '\u{012a}' | '\u{012c}' | '\u{012e}' | '\u{0130}' | '\u{0132}' => 'I',
        '\u{0129}' | '\u{012b}' | '\u{012d}' | '\u{012f}' | '\u{0131}' | '\u{0133}' => 'i',
        '\u{0134}' => 'J',
        '\u{0135}' => 'j',
        '\u{0136}' => 'K',
        '\u{0137}' | '\u{0138}' => 'k',
        '\u{0139}' | '\u{013b}' | '\u{013d}' | '\u{013f}' | '\u{0141}' => 'L',
        '\u{013a}' | '\u{013c}' | '\u{013e}' | '\u{0140}' | '\u{0142}' => 'l',
        '\u{0143}' | '\u{0145}' | '\u{0147}' | '\u{014a}' => 'N',
        '\u{0144}' | '\u{0146}' | '\u{0148}' | '\u{0149}' | '\u{014b}' => 'n',
        '\u{014c}' | '\u{014e}' | '\u{0150}' | '\u{0152}' => 'O',
        '\u{014d}' | '\u{014f}' | '\u{0151}' | '\u{0153}' => 'o',
        '\u{0154}' | '\u{0156}' | '\u{0158}' => 'R',
        '\u{0155}' | '\u{0157}' | '\u{0159}' => 'r',
        '\u{015a}' | '\u{015c}' | '\u{015e}' | '\u{0160}' => 'S',
        '\u{015b}' | '\u{015d}' | '\u{015f}' | '\u{0161}' | '\u{017f}' => 's',
        '\u{0162}' | '\u{0164}' | '\u{0166}' => 'T',
        '\u{0163}' | '\u{0165}' | '\u{0167}' => 't',
        '\u{0168}' | '\u{016a}' | '\u{016c}' | '\u{016e}' | '\u{0170}' | '\u{0172}' => 'U',
        '\u{0169}' | '\u{016b}' | '\u{016d}' | '\u{016f}' | '\u{0171}' | '\u{0173}' => 'u',
        '\u{0174}' => 'W',
        '\u{0175}' => 'w',
        '\u{0176}' | '\u{0178}' => 'Y',
        '\u{0177}' => 'y',
        '\u{0179}' | '\u{017b}' | '\u{017d}' => 'Z',
        '\u{017a}' | '\u{017c}' | '\u{017e}' => 'z',
        _ => return None,
    })
}

fn push(out: &mut Vec<Cell>, ch: char) {
    out.push(Cell::new(ch, C16::White, C16::Black));
}
