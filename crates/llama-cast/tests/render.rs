//! vcsa -> frame: the font parser, the vcsa parser, the attribute layout,
//! and a golden PNG of a synthetic screen rendered with the shipped font.
//!
//! Regenerate the golden after a deliberate change with
//! `LLAMA_CAST_BLESS=1 cargo test -p llama-cast --test render`, then look at
//! crates/llama-cast/tests/golden/synthetic-12x24.png before committing it.

mod common;

use std::path::PathBuf;

use common::fixture;
use llama_cast::font::{FontError, Psf2};
use llama_cast::render::{
    self, FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH, Screen, VcsaError, decode_cell, rgb24,
    vga_to_ansi,
};
use llama_cast::source::{FrameSource, VcsaSource, load_font, read_capped};
use llama_core::palette::Palette;

fn shipped_font(size: &str) -> PathBuf {
    common::workspace_root().join(format!("packaging/fonts/llama-hack-{size}.psfu"))
}

fn pixel(frame: &[u8], x: usize, y: usize) -> [u8; 3] {
    let at = (y * FRAME_WIDTH + x) * 3;
    [frame[at], frame[at + 1], frame[at + 2]]
}

#[test]
fn shipped_fonts_parse() {
    for (size, w, h) in [("12x24", 12, 24), ("12x22", 12, 22)] {
        let font = load_font(&shipped_font(size)).expect(size);
        assert_eq!((font.width(), font.height()), (w, h), "{size}");
        assert_eq!(font.count(), 512, "{size}");
        // Space is blank; '#' is not.
        assert!((0..w).all(|x| (0..h).all(|y| !font.pixel(32, x, y))));
        assert!((0..w).any(|x| (0..h).any(|y| font.pixel(usize::from(b'#'), x, y))));
        // Past the end: no pixels, no panic.
        assert!(!font.pixel(512, 0, 0));
        assert!(!font.pixel(65, w, 0));
    }
}

#[test]
fn bad_fonts_are_refused() {
    let good = std::fs::read(shipped_font("12x24")).unwrap();
    assert_eq!(Psf2::parse(&good[..31]), Err(FontError::Magic));
    let mut psf1 = good.clone();
    psf1[..4].copy_from_slice(&[0x36, 0x04, 0, 0]);
    assert_eq!(Psf2::parse(&psf1), Err(FontError::Magic));
    let mut truncated = good.clone();
    truncated.truncate(32 + 511 * 48);
    assert!(matches!(Psf2::parse(&truncated), Err(FontError::Header(_))));
    let mut wrong_size = good.clone();
    wrong_size[20] = 47;
    assert!(matches!(
        Psf2::parse(&wrong_size),
        Err(FontError::Header(_))
    ));
    let mut huge = good.clone();
    huge.resize(300 * 1024, 0);
    assert_eq!(Psf2::parse(&huge), Err(FontError::TooLarge));
}

#[test]
fn attribute_layouts() {
    // VGA order (bit 0 blue) to ANSI (bit 0 red).
    assert_eq!(vga_to_ansi(1), 4);
    assert_eq!(vga_to_ansi(4), 1);
    assert_eq!(vga_to_ansi(6), 3);
    assert_eq!(vga_to_ansi(9), 12);
    assert_eq!(vga_to_ansi(15), 15);
    // 512 glyphs: bit 0 is glyph bit 8, fg bits 1-4, bg bits 5-7.
    let attr = (1 << 5) | (12 << 1) | 1; // bg VGA blue, fg VGA bright red
    assert_eq!(decode_cell(0x41, attr, true), (0x141, 9, 4));
    // 256 glyphs: fg bits 0-3, bg bits 4-6, blink ignored.
    assert_eq!(decode_cell(0x41, 0x80 | (1 << 4) | 12, false), (0x41, 9, 4));
}

#[test]
fn vcsa_geometry() {
    let dump = std::fs::read(fixture("synthetic.vcsa")).unwrap();
    let screen = Screen::parse(&dump).unwrap();
    assert_eq!((screen.rows, screen.cols), (10, 40));
    assert_eq!(
        Screen::parse(&dump[..dump.len() - 2]),
        Err(VcsaError::Geometry)
    );
    assert_eq!(
        Screen::parse(&dump[..dump.len() - 1]),
        Err(VcsaError::Short)
    );
    assert_eq!(Screen::parse(&dump[..3]), Err(VcsaError::Short));
    // The kernel truncates rows and columns to 8 bits: 49 x 320 reads as
    // 49 x 64, and the cell count gives the real width back.
    let mut wide = vec![49_u8, 64, 0, 0];
    wide.resize(4 + 2 * 49 * 320, 0);
    let screen = Screen::parse(&wide).unwrap();
    assert_eq!((screen.rows, screen.cols), (49, 320));
    let mut huge = vec![0_u8; render::MAX_VCSA_BYTES + 2];
    huge[0] = 1;
    assert_eq!(Screen::parse(&huge), Err(VcsaError::TooLarge));
}

#[test]
fn synthetic_screen_matches_the_golden_png() {
    let font = load_font(&shipped_font("12x24")).unwrap();
    // The default palette (cast.toml `palette = "llama"`, #26).
    let source = VcsaSource::new(fixture("synthetic.vcsa"), font, Palette::Llama);
    let mut frame = vec![0_u8; FRAME_BYTES];
    source.frame(&mut frame).unwrap();
    let palette = rgb24(Palette::Llama);
    assert_eq!(
        palette[4],
        [0x14, 0x28, 0xd8],
        "blue is L1 of the heat ramp"
    );
    // 40x10 cells of 12x24 = 480x240, centred: origin (720, 420).
    assert_eq!(pixel(&frame, 719, 420), [0, 0, 0]);
    assert_eq!(
        pixel(&frame, 720, 420),
        palette[4],
        "row 0 has a blue background"
    );
    // The X at row 9, col 39 has a red background.
    assert_eq!(pixel(&frame, 720 + 39 * 12, 420 + 9 * 24), palette[1]);
    assert_eq!(pixel(&frame, 1200, 660), [0, 0, 0]);

    let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/synthetic-12x24.png");
    if std::env::var_os("LLAMA_CAST_BLESS").is_some() {
        let file = std::fs::File::create(&golden).unwrap();
        let mut enc = png::Encoder::new(
            std::io::BufWriter::new(file),
            FRAME_WIDTH as u32,
            FRAME_HEIGHT as u32,
        );
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_compression(png::Compression::High);
        let mut w = enc.write_header().unwrap();
        w.write_image_data(&frame).unwrap();
    }
    let decoder = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(&golden).expect("golden PNG (LLAMA_CAST_BLESS=1 to create)"),
    ));
    let mut reader = decoder.read_info().unwrap();
    let mut want = vec![0_u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut want).unwrap();
    assert_eq!(
        (info.width, info.height),
        (FRAME_WIDTH as u32, FRAME_HEIGHT as u32)
    );
    assert_eq!(info.color_type, png::ColorType::Rgb);
    want.truncate(info.buffer_size());
    assert!(
        frame == want,
        "rendered frame differs from {}",
        golden.display()
    );
}

#[test]
fn a_larger_console_is_cropped_from_the_top_left() {
    let font = load_font(&shipped_font("12x22")).unwrap();
    // 200x60 cells of 12x22 = 2400x1320, larger than the frame both ways.
    let (rows, cols) = (60_usize, 200_usize);
    let mut dump = vec![rows as u8, cols as u8, 0, 0];
    for r in 0..rows {
        for c in 0..cols {
            // Background colour by quadrant; VGA 1 (blue) top-left, VGA 4 (red) elsewhere.
            let bg: u8 = if r < 49 && c < 160 { 1 } else { 4 };
            dump.extend_from_slice(&[b' ', bg << 5]);
        }
    }
    let screen = Screen::parse(&dump).unwrap();
    let frame = render::render(&screen, &font, Palette::Vga);
    let vga = rgb24(Palette::Vga);
    assert_eq!(pixel(&frame, 0, 0), vga[4]);
    assert_eq!(pixel(&frame, 1919, 1077), vga[4]);
    assert_eq!(pixel(&frame, 1919, 1078), vga[1], "row 49 starts at y 1078");
}

#[test]
fn reads_are_capped() {
    let dir = common::scratch("read-capped");
    let path = dir.join("f");
    std::fs::write(&path, [7_u8; 100]).unwrap();
    assert_eq!(read_capped(&path, 100).unwrap().len(), 100);
    assert!(read_capped(&path, 99).is_err());
    // Symlinks are not followed.
    let link = dir.join("link");
    std::os::unix::fs::symlink(&path, &link).unwrap();
    assert!(read_capped(&link, 100).is_err());
}

/// #26: the same cells in each palette take exactly that palette's RGB.
#[test]
fn every_attribute_colour_renders_with_the_configured_palette() {
    let font = load_font(&shipped_font("12x24")).unwrap();
    // One row: 8 blank cells with background VGA 0..=7, then 16 cells of '#'
    // with foreground VGA 0..=15 on black (512-glyph attribute layout).
    let mut dump = vec![1_u8, 24, 0, 0];
    for bg in 0..8_u8 {
        dump.extend_from_slice(&[b' ', bg << 5]);
    }
    for fg in 0..16_u8 {
        dump.extend_from_slice(&[b'#', fg << 1]);
    }
    let screen = Screen::parse(&dump).unwrap();
    for palette in [Palette::Llama, Palette::Vga] {
        let colours = rgb24(palette);
        let frame = render::render(&screen, &font, palette);
        // 24 cells of 12 px = 288 px wide, centred: x 816, y 528.
        for bg in 0..8_usize {
            let ansi = usize::from(vga_to_ansi(bg as u8));
            assert_eq!(
                pixel(&frame, 816 + bg * 12, 528),
                colours[ansi],
                "{palette:?} bg {bg}"
            );
        }
        // A lit pixel of '#' shows the foreground.
        let (gx, gy) = (0..12)
            .flat_map(|x| (0..24).map(move |y| (x, y)))
            .find(|&(x, y)| font.pixel(usize::from(b'#'), x, y))
            .unwrap();
        for fg in 0..16_usize {
            let ansi = usize::from(vga_to_ansi(fg as u8));
            let at = pixel(&frame, 816 + (8 + fg) * 12 + gx, 528 + gy);
            assert_eq!(at, colours[ansi], "{palette:?} fg {fg}");
        }
    }
}
