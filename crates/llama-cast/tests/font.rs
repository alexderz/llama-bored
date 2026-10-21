//! The PSF2 Unicode table: first mapping wins, sequences are skipped, and a
//! truncated or malformed table is refused without a panic.

mod common;

use llama_cast::font::{FontError, Psf2};

fn psf(count: usize, flags: u32, glyphs: &[u8], table: &[u8]) -> Vec<u8> {
    let width: usize = 8;
    let height: usize = 8;
    let char_size = width.div_ceil(8) * height;
    assert_eq!(glyphs.len(), count * char_size);
    let mut out = Vec::new();
    out.extend_from_slice(&0x864A_B572_u32.to_le_bytes());
    out.extend_from_slice(&0_u32.to_le_bytes());
    out.extend_from_slice(&32_u32.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&u32::try_from(count).unwrap().to_le_bytes());
    out.extend_from_slice(&u32::try_from(char_size).unwrap().to_le_bytes());
    out.extend_from_slice(&u32::try_from(height).unwrap().to_le_bytes());
    out.extend_from_slice(&u32::try_from(width).unwrap().to_le_bytes());
    out.extend_from_slice(glyphs);
    if flags & 1 == 1 {
        out.extend_from_slice(table);
    }
    out
}

fn blank(count: usize) -> Vec<u8> {
    vec![0; count * 8]
}

#[test]
fn a_font_without_a_table_maps_scalars_below_its_count() {
    let font = Psf2::parse(&psf(128, 0, &blank(128), &[])).unwrap();
    assert_eq!(font.glyph_for(65), Some(65));
    assert_eq!(font.glyph_for(127), Some(127));
    assert_eq!(font.glyph_for(128), None);
    assert_eq!(font.glyph_for(0xFFFD), None);
    assert_eq!(font.glyph_for_cell(65), Some(65));
    assert_eq!(font.glyph_for_cell(200), None);
}

#[test]
fn the_first_mapping_wins_and_sequences_are_skipped() {
    // Glyph 0: U+0041, then a sequence that also names U+0042 (not mapped).
    // Glyph 1: U+0041 again (lost) and U+0042 (kept).
    let table = [
        0x41, 0xFE, 0x42, 0xCC, 0x88, 0xFF, // glyph 0
        0x41, 0x42, 0xFF, // glyph 1
    ];
    let font = Psf2::parse(&psf(2, 1, &blank(2), &table)).unwrap();
    assert_eq!(font.glyph_for(u32::from('A')), Some(0));
    assert_eq!(font.glyph_for(u32::from('B')), Some(1));
    assert_eq!(
        font.glyph_for(0x0308),
        None,
        "a sequence scalar is not a cell"
    );
}

#[test]
fn a_sequence_does_not_hide_a_later_glyphs_scalar() {
    let table = [
        0xFE, 0x41, 0xFF, // glyph 0: U+0041 only inside a sequence
        0x41, 0xFF, // glyph 1
    ];
    let font = Psf2::parse(&psf(2, 1, &blank(2), &table)).unwrap();
    assert_eq!(font.glyph_for(u32::from('A')), Some(1));
}

#[test]
fn bytes_after_a_complete_table_are_ignored() {
    let mut table = vec![0x41, 0xFF, 0x42, 0xFF];
    table.extend_from_slice(&[0x00, 0x01, 0xFF]);
    let font = Psf2::parse(&psf(2, 1, &blank(2), &table)).unwrap();
    assert_eq!(font.glyph_for(u32::from('A')), Some(0));
    assert_eq!(font.glyph_for(u32::from('B')), Some(1));
}

#[test]
fn a_truncated_or_malformed_table_is_refused() {
    let cut = psf(1, 1, &blank(1), &[0x41]);
    assert_eq!(Psf2::parse(&cut), Err(FontError::Unicode("truncated")));
    let bare_fe = psf(1, 1, &blank(1), &[0xFE, 0x41]);
    assert_eq!(Psf2::parse(&bare_fe), Err(FontError::Unicode("truncated")));
    let empty = psf(1, 1, &blank(1), &[]);
    assert_eq!(Psf2::parse(&empty), Err(FontError::Unicode("truncated")));
    // Overlong U+0041, a surrogate, and a bad lead.
    for table in [
        vec![0xC0, 0x81, 0xFF],
        vec![0xED, 0xA0, 0x80, 0xFF],
        vec![0x80, 0xFF],
        vec![0xF5, 0x80, 0x80, 0x80, 0xFF],
    ] {
        let bytes = psf(1, 1, &blank(1), &table);
        assert_eq!(
            Psf2::parse(&bytes),
            Err(FontError::Unicode("bad utf-8")),
            "{table:?}"
        );
    }
}

#[test]
fn a_scalar_past_unicode_is_refused() {
    // U+110000 is the scalar after U+10FFFF. Its UTF-8 is a 4-byte sequence
    // whose lead is F4, so a check that only rejects a lead above F4 accepts it.
    let table = [0xF4, 0x90, 0x80, 0x80, 0xFF];
    let bytes = psf(1, 1, &blank(1), &table);
    assert_eq!(Psf2::parse(&bytes), Err(FontError::Unicode("bad utf-8")));
}

#[test]
fn hostile_fonts_do_not_panic() {
    let samples: [&[u8]; 6] = [
        &[],
        &[0x72, 0xb5, 0x4a, 0x86],
        &[0x72, 0xb5, 0x4a, 0x86, 0, 0, 0, 0],
        &[0xFF; 64],
        &[0x00; 80],
        &[0xFE, 0xFF, 0x41],
    ];
    for bytes in samples {
        let _ = Psf2::parse(bytes);
    }
    let mut flagged = psf(1, 1, &blank(1), &[0xFF]);
    flagged.truncate(20);
    let _ = Psf2::parse(&flagged);
}

/// Scalars llama-watch draws itself (not free text such as a model name).
/// Same list as `llama_watch::tty::term::GLYPHS`. The chart's lower eighths
/// are U+2581..=U+2588, which this list already contains.
const DRAWN: &[char] = &[
    '█', '▌', '▐', '░', '▒', '▓', '▀', '▄', '·', '…', '≈', '’', '‘', '“', '”', '–', '—', '•',
    '\u{fffd}', '▁', '▂', '▃', '▅', '▆', '▇', '\u{e584}',
];

fn shipped(size: &str) -> Psf2 {
    let path = common::workspace_root().join(format!("packaging/fonts/llama-hack-{size}.psfu"));
    Psf2::parse(&std::fs::read(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display())))
        .unwrap_or_else(|err| panic!("{size}: {err}"))
}

#[test]
fn every_shipped_font_has_the_glyphs_the_dashboard_draws() {
    let mut block = DRAWN.to_vec();
    block.extend('\u{2581}'..='\u{2588}');
    let mut slot_of_full = None;
    for size in ["12x24", "12x22", "10x18", "9x18"] {
        let font = shipped(size);
        for ch in &block {
            let Some(index) = font.glyph_for(u32::from(*ch)) else {
                panic!("{size} has no glyph for {ch:?}");
            };
            let lit =
                (0..font.height()).any(|y| (0..font.width()).any(|x| font.pixel(index, x, y)));
            assert!(lit, "{size} draws {ch:?} as a blank cell");
        }
        let full = font.glyph_for(u32::from('█'));
        match slot_of_full {
            None => slot_of_full = Some(full),
            Some(first) => assert_eq!(full, first, "{size} moved the full-block slot"),
        }
        assert_eq!(font.glyph_for(u32::from('A')), Some(usize::from(b'A')));
    }
}

/// #165: every shipped font's gauge segment is its lower half block with
/// the last pixel column blank, and nothing else differs.
#[test]
fn every_shipped_font_draws_the_segment_as_a_half_block_one_column_short() {
    for size in ["12x24", "12x22", "10x18", "9x18"] {
        let font = shipped(size);
        let half = font.glyph_for(0x2584).expect("half block");
        let seg = font.glyph_for(0xe584).expect("segment");
        let last = font.width() - 1;
        for y in 0..font.height() {
            for x in 0..font.width() {
                let want = x != last && font.pixel(half, x, y);
                assert_eq!(font.pixel(seg, x, y), want, "{size} at ({x}, {y})");
            }
            assert!(!font.pixel(seg, last, y), "{size}: the hairline at row {y}");
        }
    }
}
