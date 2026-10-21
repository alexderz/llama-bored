//! Which scalars the grid source may show as one cell (#107).
//!
//! llama-watch's grid file carries the original character of each cell, and
//! llama-view draws it on a full-Unicode terminal. Both sides ask the same
//! question, so it lives here: [`is_showable`] is true for a scalar a
//! terminal draws as exactly one column, with nothing that can move the
//! cursor, reorder text or combine with its neighbour.
//!
//! It is an allowlist of hand-written ranges, not a Unicode database (the
//! repo adds no dependency for this). Conservative on purpose: a scalar that
//! is not listed falls back to the console glyph (watch) or U+FFFD (view),
//! which is always safe. Left out, whole: C0, DEL and C1 controls; format
//! and zero-width characters (U+00AD, U+200B..U+200F, bidi embeddings,
//! overrides and isolates U+202A..U+202E and U+2066..U+2069, U+FEFF, tags);
//! combining marks; East Asian Wide and Fullwidth forms; emoji and
//! pictographs, regional indicators and variation selectors; right-to-left
//! and complex scripts; surrogates, private use and every supplementary
//! plane.

/// Inclusive scalar ranges, ascending and disjoint.
const SHOWABLE: &[(u32, u32)] = &[
    // Basic Latin (printable), Latin-1 Supplement without NBSP and the
    // soft hyphen, Latin Extended-A/B, IPA, spacing modifier letters.
    (0x0020, 0x007e),
    (0x00a1, 0x00ac),
    (0x00ae, 0x02ff),
    // Greek and Coptic (assigned letters), Cyrillic without its combining
    // marks (U+0483..U+0489), Cyrillic Supplement.
    (0x0370, 0x0373),
    (0x0375, 0x0377),
    (0x037a, 0x037f),
    (0x0384, 0x038a),
    (0x038c, 0x038c),
    (0x038e, 0x03a1),
    (0x03a3, 0x03ff),
    (0x0400, 0x0482),
    (0x048a, 0x052f),
    // Latin Extended Additional.
    (0x1e00, 0x1eff),
    // General Punctuation without spaces, format characters (U+200B..
    // U+200F, U+2028..U+202F, U+2060..U+206F) and the combining marks.
    (0x2010, 0x2027),
    (0x2030, 0x205e),
    // Super/subscripts, currency, letterlike symbols, number forms.
    (0x2070, 0x2071),
    (0x2074, 0x208e),
    (0x2090, 0x209c),
    (0x20a0, 0x20bf),
    (0x2100, 0x214f),
    (0x2150, 0x2189),
    // Arrows, Mathematical Operators.
    (0x2190, 0x22ff),
    // Miscellaneous Technical without everything that has an emoji form
    // (U+231A..U+231B, U+2328, U+2329..U+232A, U+23CF, U+23E9 onward).
    (0x2300, 0x2319),
    (0x231c, 0x2327),
    (0x232b, 0x23ce),
    (0x23d0, 0x23e8),
    // Control Pictures, Enclosed Alphanumerics, Box Drawing, Block
    // Elements, Geometric Shapes without the emoji-width U+25FD..U+25FE.
    (0x2400, 0x2426),
    (0x2460, 0x24ff),
    (0x2500, 0x25fc),
    (0x25ff, 0x25ff),
    // A few symbols with no emoji form: stars, music, check and ballot marks.
    (0x2605, 0x2606),
    (0x2669, 0x266f),
    (0x2713, 0x2713),
    (0x2717, 0x2717),
    // Miscellaneous Mathematical Symbols-A, Supplemental Arrows-A, Braille
    // Patterns, Supplemental Arrows-B, Miscellaneous Mathematical
    // Symbols-B, Supplemental Mathematical Operators.
    (0x27c0, 0x27ff),
    (0x2800, 0x28ff),
    (0x2900, 0x2aff),
    // Latin Extended-C, alphabetic presentation forms (ligatures), the
    // replacement character.
    (0x2c60, 0x2c7f),
    (0xfb00, 0xfb06),
    (0xfffd, 0xfffd),
];

/// llama-bored's gauge segment (#165): the lower half block `▄` with its last
/// pixel column blank, so a row of them reads as separate segments with a
/// hairline between. No standard scalar draws that, so it is a private-use
/// one, and only a reader that knows it draws it: tty11's llama-hack font
/// carries the glyph, and llama-cast's rasteriser draws the gap itself.
/// Every other reader shows [`SEGMENT_PLAIN`] in its place ([`plain`]).
///
/// It is not [`is_showable`]: a terminal font may draw anything there. The
/// grid file carries it ([`is_carried`]), and [`crate::gridfile::decode`]
/// turns it back into [`SEGMENT_PLAIN`] unless the caller asks to keep it.
pub const SEGMENT: char = '\u{e584}';

/// What a reader without the segment glyph draws instead: the lower half
/// block, the same bar without the hairline.
pub const SEGMENT_PLAIN: char = '\u{2584}';

/// `c`, with [`SEGMENT`] as [`SEGMENT_PLAIN`].
#[must_use]
pub fn plain(c: char) -> char {
    if c == SEGMENT { SEGMENT_PLAIN } else { c }
}

/// A scalar the grid file may carry: [`is_showable`], or [`SEGMENT`].
#[must_use]
pub fn is_carried(c: char) -> bool {
    c == SEGMENT || is_showable(c)
}

/// Whether `c` is safe to draw as one cell on a full-Unicode terminal. See
/// the module doc for what is left out and why.
#[must_use]
pub fn is_showable(c: char) -> bool {
    let n = u32::from(c);
    SHOWABLE
        .binary_search_by(|&(lo, hi)| {
            if n < lo {
                std::cmp::Ordering::Greater
            } else if n > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_sorted_and_disjoint() {
        for &(lo, hi) in SHOWABLE {
            assert!(lo <= hi, "{lo:x}..{hi:x}");
            assert!(char::from_u32(lo).is_some() && char::from_u32(hi).is_some());
        }
        for pair in SHOWABLE.windows(2) {
            assert!(
                pair[0].1 < pair[1].0,
                "{:x?} overlaps {:x?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn no_listed_scalar_is_a_control_or_format_character() {
        for &(lo, hi) in SHOWABLE {
            for n in lo..=hi {
                let c = char::from_u32(n).expect("scalar");
                assert!(!c.is_control(), "{n:x}");
                assert!(!c.is_whitespace() || c == ' ', "{n:x}");
            }
        }
    }

    #[test]
    fn ascii_latin_greek_and_cyrillic_are_showable() {
        for c in [
            'a', 'Z', '0', ' ', '~', '?', '\u{e9}', '\u{d1}', '\u{141}', '\u{1ea1}',
        ] {
            assert!(is_showable(c), "{c:?}");
        }
        for c in ['\u{3b1}', '\u{3a9}', '\u{44f}', '\u{416}', '\u{451}'] {
            assert!(is_showable(c), "{c:?}");
        }
        for c in
            "caf\u{e9} \u{2018}\u{2019}\u{201c}\u{201d} \u{2013}\u{2014}\u{2026}\u{2022}".chars()
        {
            assert!(is_showable(c), "{c:?}");
        }
        // Arrows, the multiplication sign, box drawing, blocks, replacement.
        for c in [
            '\u{2192}', '\u{d7}', '\u{2248}', '\u{2500}', '\u{2588}', '\u{2587}', '\u{fffd}',
        ] {
            assert!(is_showable(c), "{c:?}");
        }
    }

    #[test]
    fn the_segment_is_carried_but_not_showable_and_plain_is_the_half_block() {
        assert!(
            !is_showable(SEGMENT),
            "a terminal font may draw anything there"
        );
        assert!(is_carried(SEGMENT));
        assert!(is_showable(SEGMENT_PLAIN) && is_carried(SEGMENT_PLAIN));
        assert_eq!(plain(SEGMENT), '\u{2584}');
        assert_eq!(plain('a'), 'a');
        assert_eq!(plain('\u{2584}'), '\u{2584}');
        // Any other private-use scalar is neither.
        for n in [0xe000, 0xe583, 0xe585, 0xf584, 0xf8ff] {
            let c = char::from_u32(n).expect("scalar");
            assert!(!is_carried(c), "{n:x}");
        }
    }

    #[test]
    fn controls_are_not_showable() {
        for n in (0x00..=0x1f).chain(0x7f..=0x9f) {
            assert!(!is_showable(char::from_u32(n).expect("scalar")), "{n:x}");
        }
        assert!(!is_showable('\n') && !is_showable('\t') && !is_showable('\u{1b}'));
    }

    #[test]
    fn zero_width_and_format_characters_are_not_showable() {
        for n in [
            0xad, 0x200b, 0x200c, 0x200d, 0x200e, 0x200f, 0x2028, 0x2029, 0x2060, 0x2061, 0xfeff,
            0x061c, 0x180e, 0xe0001, 0xe0020, 0xe007f, 0xfff9,
        ] {
            assert!(!is_showable(char::from_u32(n).expect("scalar")), "{n:x}");
        }
    }

    #[test]
    fn bidi_overrides_embeddings_and_isolates_are_not_showable() {
        for n in (0x202a..=0x202e).chain(0x2066..=0x2069) {
            assert!(!is_showable(char::from_u32(n).expect("scalar")), "{n:x}");
        }
    }

    #[test]
    fn combining_marks_are_not_showable() {
        for n in [
            0x300, 0x301, 0x308, 0x36f, 0x483, 0x489, 0x591, 0x64b, 0x93c, 0xe31, 0x20d0, 0x1ab0,
        ] {
            assert!(!is_showable(char::from_u32(n).expect("scalar")), "{n:x}");
        }
    }

    #[test]
    fn wide_emoji_and_pictographs_are_not_showable() {
        for c in [
            '\u{4e2d}',
            '\u{3042}',
            '\u{ac00}',
            '\u{ff21}',
            '\u{ff0c}',
            '\u{3000}',
            '\u{1f600}',
            '\u{1f680}',
            '\u{1f1e6}',
            '\u{2705}',
            '\u{274c}',
            '\u{231a}',
            '\u{23f0}',
            '\u{26a1}',
            '\u{2b50}',
            '\u{fe0f}',
            '\u{fe0e}',
            '\u{e0100}',
            '\u{1f3fb}',
        ] {
            assert!(!is_showable(c), "{c:?}");
        }
    }

    #[test]
    fn other_planes_and_odd_ranges_are_not_showable() {
        for n in [
            0xa0,
            0xd800 - 1,
            0xe000,
            0xf8ff,
            0xfdd0,
            0xfffe,
            0xffff,
            0x10000,
            0x1d400,
            0x10ffff,
            0x5d0,
            0x627,
            0x378,
        ] {
            if let Some(c) = char::from_u32(n) {
                assert!(!is_showable(c), "{n:x}");
            }
        }
    }
}
