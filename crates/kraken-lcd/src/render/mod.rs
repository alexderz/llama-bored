use tiny_skia::Pixmap;

use crate::config::DisplayCfg;
use crate::present::{Band, Variant, View};

mod chart;
pub mod color;
mod dial;
mod geometry;
pub mod layout_a;
mod particles;
pub mod ring;
pub mod text;

/// One sRGB channel triple from the mockup palette. Alpha is applied at the draw site.
pub use llama_core::color::Rgb;

/// LCD off-pixels. `#000000`.
pub const BACKGROUND: Rgb = Rgb {
    r: 0x00,
    g: 0x00,
    b: 0x00,
};

/// Block edge and empty-segment dashes. `#3A3A3A`.
pub const OUTLINE: Rgb = Rgb {
    r: 0x3A,
    g: 0x3A,
    b: 0x3A,
};
/// Numerals, model name, and state words. `#F4F4F2`.
pub const TEXT: Rgb = Rgb {
    r: 0xF4,
    g: 0xF4,
    b: 0xF2,
};
/// Column labels. `#9A9A9A`.
pub const LABEL: Rgb = Rgb {
    r: 0x9A,
    g: 0x9A,
    b: 0x9A,
};
/// The failed-source dash. `#666666`.
pub const NO_DATA: Rgb = Rgb {
    r: 0x66,
    g: 0x66,
    b: 0x66,
};
/// Quiet band. `#2B5C8A`.
pub const QUIET: Rgb = Rgb {
    r: 0x2B,
    g: 0x5C,
    b: 0x8A,
};
/// Light band. `#2EB88A`.
pub const LIGHT: Rgb = Rgb {
    r: 0x2E,
    g: 0xB8,
    b: 0x8A,
};
/// Busy band. `#FFB030`.
pub const BUSY: Rgb = Rgb {
    r: 0xFF,
    g: 0xB0,
    b: 0x30,
};
/// Flat-out band. `#FFF3E0`.
pub const FLAT_OUT: Rgb = Rgb {
    r: 0xFF,
    g: 0xF3,
    b: 0xE0,
};

/// Visible disc radius. Pixel centres farther than this from the frame centre are cleared.
pub const DISC_RADIUS: f32 = 160.0;

/// Colour of a load band. [`Band::Filling`] has no colour.
#[must_use]
pub fn band_color(band: Band) -> Option<Rgb> {
    match band {
        Band::Quiet => Some(QUIET),
        Band::Light => Some(LIGHT),
        Band::Busy => Some(BUSY),
        Band::FlatOut => Some(FLAT_OUT),
        Band::Filling => None,
    }
}

/// Ring midline radius of a layout: 151 for A1, 148 for A3.
#[must_use]
pub fn ring_radius(variant: Variant) -> f32 {
    geometry::for_variant(variant).ring_radius
}

/// Paint the frame. No clock, I/O, device, or network.
///
/// Background is black, then the V3b ring, the activity dial, layout text
/// (with the tokens chart), the ring readout, the peg's smoke and embers,
/// then the circular mask. `View::variant` selects the geometry. The frame
/// clock and the particles come from the view's memory. `cfg.rotate_deg` is
/// applied when the frame is packed, not here. `assets` is mutable so layout
/// can fill the bounded glyph cache without a `RefCell`.
#[must_use]
pub fn render(view: &View, _cfg: &DisplayCfg, assets: &mut Assets) -> Frame {
    let geom = geometry::for_variant(view.variant);
    let mut frame = Frame::new();
    frame.0.fill(tiny_skia::Color::from_rgba8(
        BACKGROUND.r,
        BACKGROUND.g,
        BACKGROUND.b,
        255,
    ));
    let anim = &view.dial_state.anim;
    let t = anim.t();
    let value = layout_a::ring(view).map(f32::from);
    ring::draw(&mut frame.0, value, geom, t);
    dial::draw(&mut frame.0, view, geom, t, assets.text());
    layout_a::draw(&mut frame.0, view, assets, geom);
    ring::draw_readout(&mut frame.0, assets.text(), value, geom);
    if value.is_some() {
        particles::draw(&mut frame.0, &anim.particles);
    }
    apply_disc_mask(&mut frame.0);
    frame
}

pub const FRAME_W: u32 = 320;
pub const FRAME_H: u32 = 320;

pub struct Frame(pub Pixmap);

impl Frame {
    pub fn new() -> Self {
        match Pixmap::new(FRAME_W, FRAME_H) {
            Some(pixmap) => Self(pixmap),
            None => unreachable!("FRAME_W and FRAME_H are non-zero"),
        }
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}

/// Pixels whose centre is farther than 160 px from the frame centre become black.
fn apply_disc_mask(pixmap: &mut Pixmap) {
    let width = pixmap.width();
    for (index, pixel) in pixmap.pixels_mut().iter_mut().enumerate() {
        let x = index as u32 % width;
        let y = index as u32 / width;
        if outside_disc(x, y) {
            *pixel = opaque(BACKGROUND);
        }
    }
}

fn outside_disc(x: u32, y: u32) -> bool {
    let dx = f64::from(x) + 0.5 - f64::from(DISC_RADIUS);
    let dy = f64::from(y) + 0.5 - f64::from(DISC_RADIUS);
    dx * dx + dy * dy > f64::from(DISC_RADIUS) * f64::from(DISC_RADIUS)
}

fn opaque(color: Rgb) -> tiny_skia::PremultipliedColorU8 {
    match tiny_skia::PremultipliedColorU8::from_rgba(color.r, color.g, color.b, 255) {
        Some(pixel) => pixel,
        None => tiny_skia::PremultipliedColorU8::TRANSPARENT,
    }
}

/// Which pre-rendered brain sprite to draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Brain {
    /// Model loaded.
    Awake,
    /// llama-swap up, nothing loaded.
    Sleeping,
    /// llama-swap unreachable.
    Dead,
}

/// Bundled fonts or sprites failed to decode.
#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    /// A face rejected by fontdue.
    #[error("failed to decode {face} font: {reason}")]
    Font {
        /// ExtraBold, Bold, or SemiBold.
        face: &'static str,
        /// fontdue's reason.
        reason: &'static str,
    },
    /// A brain PNG rejected by the png crate.
    #[error("failed to decode {which} sprite: {detail}")]
    Sprite {
        /// awake, sleeping, or dead.
        which: &'static str,
        /// Decoder detail.
        detail: String,
    },
}

/// Fonts and brain sprites, decoded once at start.
///
/// Faces are Inter 4.1 static TTFs (SIL OFL 1.1):
/// <https://github.com/rsms/inter/releases/tag/v4.1>.
pub struct Assets {
    text: text::GlyphCache,
    sprites: [Pixmap; 3],
}

impl Assets {
    /// Decode the bundled faces and the three 40 px brain sprites.
    ///
    /// This is start-up self-check 3. A bad file returns an error instead of panicking.
    pub fn load() -> Result<Self, AssetError> {
        Self::from_bytes(
            include_bytes!("../../assets/fonts/Inter-ExtraBold.ttf"),
            include_bytes!("../../assets/fonts/Inter-Bold.ttf"),
            include_bytes!("../../assets/fonts/Inter-SemiBold.ttf"),
            include_bytes!("../../assets/icons/brain-awake-40.png"),
            include_bytes!("../../assets/icons/brain-sleeping-40.png"),
            include_bytes!("../../assets/icons/brain-dead-40.png"),
        )
    }

    fn from_bytes(
        extra_bold: &[u8],
        bold: &[u8],
        semi_bold: &[u8],
        awake: &[u8],
        sleeping: &[u8],
        dead: &[u8],
    ) -> Result<Self, AssetError> {
        let text = text::GlyphCache::from_fonts(extra_bold, bold, semi_bold).map_err(|err| {
            let text::FontError::Decode { face, reason } = err;
            AssetError::Font { face, reason }
        })?;
        Ok(Self {
            text,
            sprites: [
                decode_sprite("awake", awake)?,
                decode_sprite("sleeping", sleeping)?,
                decode_sprite("dead", dead)?,
            ],
        })
    }

    /// Glyph cache over the three bundled faces. Mutable because drawing inserts glyphs.
    #[must_use]
    pub fn text(&mut self) -> &mut text::GlyphCache {
        &mut self.text
    }

    /// One 40 px brain sprite.
    #[must_use]
    pub fn sprite(&self, brain: Brain) -> &Pixmap {
        match brain {
            Brain::Awake => &self.sprites[0],
            Brain::Sleeping => &self.sprites[1],
            Brain::Dead => &self.sprites[2],
        }
    }
}

/// Awake, sleeping, or dead sprite for the view's AI state.
///
/// [`crate::present::Ai::NoData`] has no sprite.
#[must_use]
pub fn brain_for(ai: crate::present::Ai) -> Option<Brain> {
    match ai {
        crate::present::Ai::Loaded => Some(Brain::Awake),
        crate::present::Ai::Idle => Some(Brain::Sleeping),
        crate::present::Ai::Down => Some(Brain::Dead),
        crate::present::Ai::NoData => None,
    }
}

fn decode_sprite(which: &'static str, bytes: &[u8]) -> Result<Pixmap, AssetError> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().map_err(|err| sprite_err(which, err))?;
    let buffer_len = reader
        .output_buffer_size()
        .ok_or_else(|| AssetError::Sprite {
            which,
            detail: "image is too large".to_owned(),
        })?;
    let mut buffer = vec![0; buffer_len];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|err| sprite_err(which, err))?;
    if info.bit_depth != png::BitDepth::Eight || info.color_type != png::ColorType::Rgba {
        return Err(AssetError::Sprite {
            which,
            detail: format!(
                "need 8-bit RGBA, got {:?} {:?}",
                info.bit_depth, info.color_type
            ),
        });
    }
    let pixel_count = usize::try_from(info.width)
        .ok()
        .and_then(|width| usize::try_from(info.height).ok()?.checked_mul(width))
        .ok_or_else(|| AssetError::Sprite {
            which,
            detail: "image is too large".to_owned(),
        })?;
    let byte_len = pixel_count
        .checked_mul(4)
        .ok_or_else(|| AssetError::Sprite {
            which,
            detail: "image is too large".to_owned(),
        })?;
    if buffer.len() < byte_len {
        return Err(AssetError::Sprite {
            which,
            detail: "decoded buffer is short".to_owned(),
        });
    }
    buffer.truncate(byte_len);
    for pixel in buffer.chunks_mut(4) {
        let alpha = pixel[3];
        pixel[0] = text::premultiply(pixel[0], alpha);
        pixel[1] = text::premultiply(pixel[1], alpha);
        pixel[2] = text::premultiply(pixel[2], alpha);
    }
    let size =
        tiny_skia::IntSize::from_wh(info.width, info.height).ok_or_else(|| AssetError::Sprite {
            which,
            detail: "invalid image size".to_owned(),
        })?;
    Pixmap::from_vec(buffer, size).ok_or_else(|| AssetError::Sprite {
        which,
        detail: "could not build pixmap".to_owned(),
    })
}

fn sprite_err(which: &'static str, err: png::DecodingError) -> AssetError {
    AssetError::Sprite {
        which,
        detail: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiny_skia::Color;

    fn sample(pixmap: &Pixmap, turns: f32) -> (u8, u8, u8) {
        let theta = turns * std::f32::consts::TAU;
        let x = 160.0 + 148.0 * theta.sin();
        let y = 160.0 - 148.0 * theta.cos();
        let ix = (x - 0.5).round() as u32;
        let iy = (y - 0.5).round() as u32;
        let pixel = pixmap.pixel(ix, iy).expect("sample on the frame");
        (pixel.red(), pixel.green(), pixel.blue())
    }

    fn near(got: (u8, u8, u8), expect: (u8, u8, u8), tol: u8) -> bool {
        abs_diff(got.0, expect.0) <= tol
            && abs_diff(got.1, expect.1) <= tol
            && abs_diff(got.2, expect.2) <= tol
    }

    fn abs_diff(left: u8, right: u8) -> u8 {
        left.abs_diff(right)
    }

    fn black_frame() -> Pixmap {
        let mut pixmap = Pixmap::new(FRAME_W, FRAME_H).expect("320 is non-zero");
        pixmap.fill(Color::from_rgba8(0, 0, 0, 255));
        pixmap
    }

    #[test]
    fn frame_is_320_square() {
        assert_eq!(FRAME_W, 320);
        assert_eq!(FRAME_H, 320);
        let frame = Frame::new();
        assert_eq!(frame.0.width(), 320);
        assert_eq!(frame.0.height(), 320);
    }

    #[test]
    fn disc_mask_clears_pixels_farther_than_160_from_centre() {
        let mut pixmap = Pixmap::new(FRAME_W, FRAME_H).expect("320 is non-zero");
        pixmap.fill(Color::from_rgba8(255, 255, 255, 255));
        apply_disc_mask(&mut pixmap);

        let outside = pixmap.pixel(0, 146).expect("in bounds");
        assert_eq!(
            (outside.red(), outside.green(), outside.blue()),
            (0, 0, 0),
            "pixel (0, 146) is just outside r = 160"
        );
        let inside = pixmap.pixel(0, 147).expect("in bounds");
        assert_eq!(
            (inside.red(), inside.green(), inside.blue()),
            (255, 255, 255),
            "pixel (0, 147) is just inside r = 160"
        );
        let corner = pixmap.pixel(0, 0).expect("in bounds");
        assert_eq!((corner.red(), corner.green(), corner.blue()), (0, 0, 0));
        let centre = pixmap.pixel(160, 160).expect("in bounds");
        assert_eq!(
            (centre.red(), centre.green(), centre.blue()),
            (255, 255, 255)
        );
    }

    #[test]
    fn no_value_leaves_the_bare_track_and_zero_draws_the_start_dot() {
        let geom = &geometry::A1;
        let start = ring::polar(geom.ring_radius, 225.0);
        let pixel = |pixmap: &Pixmap, (x, y): (f32, f32)| {
            let p = pixmap
                .pixel((x - 0.5).round() as u32, (y - 0.5).round() as u32)
                .expect("on frame");
            (p.red(), p.green(), p.blue())
        };
        let mut bare = black_frame();
        ring::draw(&mut bare, None, geom, 0.0);
        let dim = color::mix(color::act_color(0.0), color::BLACK, 0.8);
        let got = pixel(&bare, ring::polar(geom.ring_radius, 227.0));
        assert!(near(got, (dim.r, dim.g, dim.b), 12), "dim track {got:?}");
        let mut zero = black_frame();
        ring::draw(&mut zero, Some(0.0), geom, 0.0);
        let l1 = color::act_color(0.0);
        let got = pixel(&zero, start);
        assert!(
            near(got, (l1.r, l1.g, l1.b), 12),
            "0 % is a dot at the start, got {got:?}"
        );
        let got = pixel(&bare, (start.0 - 4.0, start.1 + 1.0));
        assert!(
            !near(got, (l1.r, l1.g, l1.b), 30),
            "no data has no start dot, got {got:?}"
        );
    }

    #[test]
    fn bundled_fonts_and_sprites_decode() {
        let mut assets = Assets::load().expect("startup self-check decodes bundled assets");
        for brain in [Brain::Awake, Brain::Sleeping, Brain::Dead] {
            let sprite = assets.sprite(brain);
            assert_eq!((sprite.width(), sprite.height()), (40, 40), "{brain:?}");
            assert!(
                sprite.pixels().iter().any(|pixel| pixel.alpha() > 0),
                "{brain:?} sprite is blank"
            );
        }
        let mut pixmap = black_frame();
        assets.text().draw_missing(
            &mut pixmap,
            text::Pen {
                x: 40.0,
                baseline: 80.0,
            },
            34.0,
            text::Weight::ExtraBold,
        );
        let ink = pixmap.pixels().iter().any(|pixel| pixel.red() > 20);
        assert!(ink, "decoded ExtraBold can draw the missing-source dash");
    }

    #[test]
    fn a_bad_sprite_or_font_is_an_error() {
        let extra = include_bytes!("../../assets/fonts/Inter-ExtraBold.ttf");
        let bold = include_bytes!("../../assets/fonts/Inter-Bold.ttf");
        let semi = include_bytes!("../../assets/fonts/Inter-SemiBold.ttf");
        let awake = include_bytes!("../../assets/icons/brain-awake-40.png");
        let sleeping = include_bytes!("../../assets/icons/brain-sleeping-40.png");
        let dead = include_bytes!("../../assets/icons/brain-dead-40.png");
        let bad_font = Assets::from_bytes(&[0, 1, 2, 3], bold, semi, awake, sleeping, dead);
        match bad_font {
            Err(AssetError::Font {
                face: "ExtraBold", ..
            }) => {}
            Err(AssetError::Font { face, .. }) => panic!("bad font reported {face}"),
            Err(AssetError::Sprite { which, .. }) => panic!("bad font reported sprite {which}"),
            Ok(_) => panic!("truncated font decoded"),
        }
        let bad_sprite = Assets::from_bytes(extra, bold, semi, b"not-a-png", sleeping, dead);
        match bad_sprite {
            Err(AssetError::Sprite { which: "awake", .. }) => {}
            Err(AssetError::Sprite { which, .. }) => panic!("bad sprite reported {which}"),
            Err(AssetError::Font { face, .. }) => panic!("bad sprite reported font {face}"),
            Ok(_) => panic!("truncated sprite decoded"),
        }
    }

    #[test]
    fn brain_sprite_follows_ai_state() {
        use crate::present::Ai;
        assert_eq!(brain_for(Ai::Loaded), Some(Brain::Awake));
        assert_eq!(brain_for(Ai::Idle), Some(Brain::Sleeping));
        assert_eq!(brain_for(Ai::Down), Some(Brain::Dead));
        assert_eq!(brain_for(Ai::NoData), None);
    }

    #[test]
    fn no_data_draws_the_circle_mark_and_no_brain() {
        use crate::present::{Ai, Band, View};

        let view = View {
            ring_pct: Some(100),
            ring_band: Some(Band::FlatOut),
            blocks: [Band::Quiet; 3],
            coolant_c: None,
            cpu_c: None,
            gpu_c: None,
            cpu_pct: None,
            mem_pct: None,
            ai: Ai::NoData,
            models: Vec::new(),
            model_count: 0,
            ..View::default()
        };
        let mut assets = Assets::load().expect("assets");
        let frame = render(&view, &DisplayCfg::default(), &mut assets);

        // Left edge of the A1 mark (T51 V1 slot: circle centred near (94, 78), radius ~8.3).
        let stroke = frame.0.pixel(86, 78).expect("mark");
        assert!(
            stroke.red().abs_diff(LABEL.r) <= 40
                && stroke.green().abs_diff(LABEL.g) <= 40
                && stroke.blue().abs_diff(LABEL.b) <= 40,
            "no data draws a #9A9A9A circle, got {:02x}{:02x}{:02x}",
            stroke.red(),
            stroke.green(),
            stroke.blue()
        );
        let hole = frame.0.pixel(94, 73).expect("mark interior");
        assert!(
            hole.red() <= 16 && hole.green() <= 16 && hole.blue() <= 16,
            "the mark is not a filled brain, got {:02x}{:02x}{:02x}",
            hole.red(),
            hole.green(),
            hole.blue()
        );

        let mut words = 0usize;
        for y in 78..92 {
            for x in 108..210 {
                let pixel = frame.0.pixel(x, y).expect("pixel");
                if pixel.red().abs_diff(TEXT.r) <= 24
                    && pixel.green().abs_diff(TEXT.g) <= 24
                    && pixel.blue().abs_diff(TEXT.b) <= 24
                {
                    words += 1;
                }
            }
        }
        assert!(
            words > 12,
            "\"no data\" is the text colour, saw {words} pixels"
        );

        // 3 o'clock is 104 on the scale: the bare redline track, no arc.
        let three = sample(&frame.0, 0.25);
        assert!(
            near(three, (0x3A, 0x12, 0x16), 16),
            "no data leaves the ring on the track, got {three:?}"
        );
    }
}
