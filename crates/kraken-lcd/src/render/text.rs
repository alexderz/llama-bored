//! Fontdue glyph cache and coverage blending onto a pixmap.

use std::collections::HashMap;

use fontdue::{Font, FontSettings};
use tiny_skia::{Pixmap, PremultipliedColorU8};

use super::{NO_DATA, Rgb};

/// Coverage 0 leaves `dst`. Coverage 255 replaces it with `color`. In between, source-over.
///
/// `color` is straight (not premultiplied). `coverage` is the glyph's alpha.
pub(crate) fn blend_coverage(
    dst: PremultipliedColorU8,
    color: Rgb,
    coverage: u8,
) -> PremultipliedColorU8 {
    if coverage == 0 {
        return dst;
    }
    let src_a = coverage;
    let src_r = premultiply(color.r, src_a);
    let src_g = premultiply(color.g, src_a);
    let src_b = premultiply(color.b, src_a);
    let inv = 255 - src_a;
    let out_a = src_a.saturating_add(premultiply(dst.alpha(), inv));
    let out_r = src_r.saturating_add(premultiply(dst.red(), inv)).min(out_a);
    let out_g = src_g
        .saturating_add(premultiply(dst.green(), inv))
        .min(out_a);
    let out_b = src_b
        .saturating_add(premultiply(dst.blue(), inv))
        .min(out_a);
    match PremultipliedColorU8::from_rgba(out_r, out_g, out_b, out_a) {
        Some(pixel) => pixel,
        None => dst,
    }
}

/// `a * b / 255`, rounded the same way as tiny-skia's pixmap premultiply.
pub(crate) fn premultiply(channel: u8, alpha: u8) -> u8 {
    let prod = u32::from(channel) * u32::from(alpha) + 128;
    ((prod + (prod >> 8)) >> 8) as u8
}

/// Where one line of text sits. Y grows downward.
#[derive(Clone, Copy, Debug)]
pub struct Pen {
    /// Left edge of the line, in pixels.
    pub x: f32,
    /// Baseline, in pixels.
    pub baseline: f32,
}

/// Size, face, and colour for one run of text.
#[derive(Clone, Copy, Debug)]
pub struct TextStyle {
    /// Pixels per em.
    pub px: f32,
    /// Bundled Inter face.
    pub weight: Weight,
    /// Straight RGB. Glyph coverage supplies the alpha.
    pub color: Rgb,
    /// Extra space between glyphs, as a fraction of [`Self::px`]. Labels use 0.06.
    pub tracking_em: f32,
}

/// Width of one run and how high its ink sits above the baseline.
#[derive(Clone, Copy, Debug)]
pub struct Measured {
    /// Pen travel, including kerning and tracking. Matches [`GlyphCache::draw`].
    pub advance: f32,
    /// Right edge of the glyph bitmaps, relative to the starting pen.
    pub ink_right: f32,
    /// Top of the ink above the baseline, at least the cap height of `H`.
    pub above_baseline: f32,
}

/// Which bundled Inter face to draw with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Weight {
    /// Inter ExtraBold, 800. Numerals.
    ExtraBold,
    /// Inter Bold, 700.
    Bold,
    /// Inter SemiBold, 600. Labels and the model name.
    SemiBold,
}

/// How many rasterised glyphs to keep. Model names are untrusted, so a run of
/// distinct characters cannot grow the map without bound.
const GLYPH_CACHE_CAP: usize = 512;

/// Rasterised glyphs for one process. Drawing takes `&mut self` and `render`
/// takes `&mut Assets`, so the map needs no `RefCell`. It is cleared when a new
/// glyph would push it past [`GLYPH_CACHE_CAP`].
pub struct GlyphCache {
    fonts: [Font; 3],
    glyphs: HashMap<GlyphKey, Raster>,
}

/// A bundled face did not decode.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FontError {
    /// `from_bytes` rejected the face.
    #[error("failed to decode {face}: {reason}")]
    Decode {
        /// ExtraBold, Bold, or SemiBold.
        face: &'static str,
        /// fontdue's reason.
        reason: &'static str,
    },
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
struct GlyphKey {
    weight: u8,
    px_bits: u32,
    ch: char,
}

struct Raster {
    xmin: i32,
    ymin: i32,
    width: usize,
    height: usize,
    advance: f32,
    coverage: Vec<u8>,
}

impl GlyphCache {
    /// Decode the three faces. A bad file is an error, not a panic.
    pub fn from_fonts(extra_bold: &[u8], bold: &[u8], semi_bold: &[u8]) -> Result<Self, FontError> {
        let settings = FontSettings {
            scale: 34.0,
            ..FontSettings::default()
        };
        Ok(Self {
            fonts: [
                decode("ExtraBold", extra_bold, settings)?,
                decode("Bold", bold, settings)?,
                decode("SemiBold", semi_bold, settings)?,
            ],
            glyphs: HashMap::new(),
        })
    }

    /// Draw `text` on one line. Tracking from [`TextStyle::tracking_em`] is inserted
    /// between glyphs, after kerning.
    pub fn draw(&mut self, pixmap: &mut Pixmap, text: &str, mut pen: Pen, style: TextStyle) {
        if !style.px.is_finite() || style.px <= 0.0 {
            return;
        }
        let font_index = face_index(style.weight);
        let tracking = style.px * style.tracking_em;
        let mut previous = None;
        for ch in text.chars() {
            if let Some(prev) = previous {
                if let Some(kern) = self.fonts[font_index].horizontal_kern(prev, ch, style.px) {
                    pen.x += kern;
                }
                pen.x += tracking;
            }
            pen.x += self.place(pixmap, style, ch, pen);
            previous = Some(ch);
        }
    }

    /// Measure `text` without drawing it.
    ///
    /// [`Measured::above_baseline`] is the higher of the run's bitmap tops and the
    /// cap height of `H`, so a fit check can use the top of the line.
    #[must_use]
    pub fn measure(&self, text: &str, px: f32, weight: Weight, tracking_em: f32) -> Measured {
        if !px.is_finite() || px <= 0.0 {
            return Measured {
                advance: 0.0,
                ink_right: 0.0,
                above_baseline: 0.0,
            };
        }
        let font = &self.fonts[face_index(weight)];
        let tracking = px * tracking_em;
        let mut above = bitmap_top(&font.metrics('H', px));
        let mut x = 0.0;
        let mut ink_right = 0.0_f32;
        let mut previous = None;
        for ch in text.chars() {
            if let Some(prev) = previous {
                if let Some(kern) = font.horizontal_kern(prev, ch, px) {
                    x += kern;
                }
                x += tracking;
            }
            let metrics = font.metrics(ch, px);
            let right = x + metrics.xmin as f32 + metrics.width as f32;
            if right > ink_right {
                ink_right = right;
            }
            let top = bitmap_top(&metrics);
            if top > above {
                above = top;
            }
            x += metrics.advance_width;
            previous = Some(ch);
        }
        Measured {
            advance: x,
            ink_right,
            above_baseline: above,
        }
    }

    /// Draw "—" in `#666666` for a failed source.
    pub fn draw_missing(&mut self, pixmap: &mut Pixmap, pen: Pen, px: f32, weight: Weight) {
        self.draw(
            pixmap,
            "\u{2014}",
            pen,
            TextStyle {
                px,
                weight,
                color: NO_DATA,
                tracking_em: 0.0,
            },
        );
    }

    #[cfg(test)]
    fn cached_glyphs(&self) -> usize {
        self.glyphs.len()
    }

    fn place(&mut self, pixmap: &mut Pixmap, style: TextStyle, ch: char, pen: Pen) -> f32 {
        let key = GlyphKey {
            weight: face_index(style.weight) as u8,
            px_bits: style.px.to_bits(),
            ch,
        };
        if let Some(advance) = self.glyphs.get(&key).map(|raster| {
            blit(pixmap, raster, pen.x, pen.baseline, style.color);
            raster.advance
        }) {
            return advance;
        }
        if self.glyphs.len() >= GLYPH_CACHE_CAP {
            self.glyphs.clear();
        }
        let raster = rasterize(&self.fonts[face_index(style.weight)], style.px, ch);
        blit(pixmap, &raster, pen.x, pen.baseline, style.color);
        let advance = raster.advance;
        self.glyphs.insert(key, raster);
        advance
    }
}

fn decode(face: &'static str, bytes: &[u8], settings: FontSettings) -> Result<Font, FontError> {
    Font::from_bytes(bytes, settings).map_err(|reason| FontError::Decode { face, reason })
}

fn bitmap_top(metrics: &fontdue::Metrics) -> f32 {
    let top = metrics.ymin as f32 + metrics.height as f32;
    if top.is_finite() && top > 0.0 {
        top
    } else {
        0.0
    }
}

fn face_index(weight: Weight) -> usize {
    match weight {
        Weight::ExtraBold => 0,
        Weight::Bold => 1,
        Weight::SemiBold => 2,
    }
}

fn rasterize(font: &Font, px: f32, ch: char) -> Raster {
    let (metrics, coverage) = font.rasterize(ch, px);
    Raster {
        xmin: metrics.xmin,
        ymin: metrics.ymin,
        width: metrics.width,
        height: metrics.height,
        advance: metrics.advance_width,
        coverage,
    }
}

fn blit(pixmap: &mut Pixmap, raster: &Raster, pen_x: f32, baseline_y: f32, color: Rgb) {
    if raster.width == 0 || raster.height == 0 {
        return;
    }
    let Some(cells) = raster.width.checked_mul(raster.height) else {
        return;
    };
    if raster.coverage.len() < cells {
        return;
    }
    let origin_x = pen_x + raster.xmin as f32;
    let origin_y = baseline_y - raster.ymin as f32 - raster.height as f32;
    let width = pixmap.width() as i32;
    let height = pixmap.height() as i32;
    let row_stride = pixmap.width();
    let pixels = pixmap.pixels_mut();
    for row in 0..raster.height {
        let y = origin_y.round() as i32 + row as i32;
        if y < 0 || y >= height {
            continue;
        }
        for col in 0..raster.width {
            let x = origin_x.round() as i32 + col as i32;
            if x < 0 || x >= width {
                continue;
            }
            let coverage = raster.coverage[row * raster.width + col];
            if coverage == 0 {
                continue;
            }
            let index = (y as u32 * row_stride + x as u32) as usize;
            pixels[index] = blend_coverage(pixels[index], color, coverage);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::NO_DATA;
    use tiny_skia::PremultipliedColorU8;

    fn channels(pixel: PremultipliedColorU8) -> (u8, u8, u8, u8) {
        (pixel.red(), pixel.green(), pixel.blue(), pixel.alpha())
    }

    #[test]
    fn full_coverage_replaces_the_destination() {
        let dst = PremultipliedColorU8::from_rgba(255, 0, 0, 255).expect("opaque red");
        let out = blend_coverage(dst, NO_DATA, 255);
        assert_eq!(channels(out), (0x66, 0x66, 0x66, 255));
    }

    #[test]
    fn zero_coverage_leaves_the_destination() {
        let dst = PremultipliedColorU8::from_rgba(255, 0, 0, 255).expect("opaque red");
        let out = blend_coverage(dst, NO_DATA, 0);
        assert_eq!(channels(out), (255, 0, 0, 255));
    }

    #[test]
    fn half_coverage_of_white_over_red() {
        let dst = PremultipliedColorU8::from_rgba(255, 0, 0, 255).expect("opaque red");
        let white = Rgb {
            r: 255,
            g: 255,
            b: 255,
        };
        let out = blend_coverage(dst, white, 128);
        assert_eq!(
            channels(out),
            (255, 128, 128, 255),
            "premultiplied source-over at coverage 128"
        );
    }

    use crate::render::{FRAME_H, TEXT};
    use tiny_skia::{Color, Pixmap};

    fn cache() -> GlyphCache {
        GlyphCache::from_fonts(
            include_bytes!("../../assets/fonts/Inter-ExtraBold.ttf"),
            include_bytes!("../../assets/fonts/Inter-Bold.ttf"),
            include_bytes!("../../assets/fonts/Inter-SemiBold.ttf"),
        )
        .expect("bundled Inter decodes")
    }

    fn black() -> Pixmap {
        let mut pixmap = Pixmap::new(320, FRAME_H).expect("frame");
        pixmap.fill(Color::from_rgba8(0, 0, 0, 255));
        pixmap
    }

    fn ink_right(pixmap: &Pixmap) -> u32 {
        let mut max_x = 0;
        for y in 0..pixmap.height() {
            for x in 0..pixmap.width() {
                let pixel = pixmap.pixel(x, y).expect("pixel");
                if pixel.red() > 20 {
                    max_x = max_x.max(x);
                }
            }
        }
        max_x
    }

    #[test]
    fn missing_mark_is_the_em_dash_in_no_data_grey() {
        let mut cache = cache();
        let mut missing = black();
        let pen = Pen {
            x: 16.0,
            baseline: 60.0,
        };
        cache.draw_missing(&mut missing, pen, 34.0, Weight::ExtraBold);
        let mut direct = black();
        cache.draw(
            &mut direct,
            "\u{2014}",
            pen,
            TextStyle {
                px: 34.0,
                weight: Weight::ExtraBold,
                color: NO_DATA,
                tracking_em: 0.0,
            },
        );
        assert_eq!(missing.data(), direct.data());
        let mut coloured = 0;
        let mut min_y = u32::MAX;
        let mut max_y = 0;
        for (index, pixel) in missing.pixels().iter().enumerate() {
            let (r, g, b) = (pixel.red(), pixel.green(), pixel.blue());
            if r > 8 || g > 8 || b > 8 {
                coloured += 1;
                let spread = r.max(g).max(b) - r.min(g).min(b);
                assert!(spread <= 2, "dash pixel {r:02x}{g:02x}{b:02x} is not grey");
                let y = (index as u32) / missing.width();
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
        assert!(
            coloured > 10,
            "the dash should cover more than a speck, saw {coloured}"
        );
        let mid = (min_y + max_y) / 2;
        assert!(mid < 60, "dash centre {mid} should sit above the baseline");
        assert!(
            60 - mid < 34,
            "dash centre {mid} should stay within one em of the baseline"
        );
    }

    #[test]
    fn a_repeated_glyph_is_cached_once() {
        let mut cache = cache();
        let mut pixmap = black();
        cache.draw(
            &mut pixmap,
            "——",
            Pen {
                x: 8.0,
                baseline: 60.0,
            },
            TextStyle {
                px: 28.0,
                weight: Weight::SemiBold,
                color: TEXT,
                tracking_em: 0.0,
            },
        );
        assert_eq!(cache.cached_glyphs(), 1);
    }

    #[test]
    fn two_marks_advance_past_one() {
        let mut cache = cache();
        let mut one = black();
        let pen = Pen {
            x: 8.0,
            baseline: 60.0,
        };
        cache.draw_missing(&mut one, pen, 34.0, Weight::Bold);
        let mut two = black();
        cache.draw(
            &mut two,
            "\u{2014}\u{2014}",
            pen,
            TextStyle {
                px: 34.0,
                weight: Weight::Bold,
                color: NO_DATA,
                tracking_em: 0.0,
            },
        );
        assert!(ink_right(&two) > ink_right(&one));
    }

    #[test]
    fn a_truncated_font_is_an_error() {
        let err = GlyphCache::from_fonts(
            &[0, 1, 2, 3],
            include_bytes!("../../assets/fonts/Inter-Bold.ttf"),
            include_bytes!("../../assets/fonts/Inter-SemiBold.ttf"),
        );
        assert!(err.is_err(), "a truncated ExtraBold face must not decode");
    }

    #[test]
    fn measure_reports_advance_ink_and_cap_height() {
        let cache = cache();
        let one = cache.measure("W", 19.0, Weight::SemiBold, 0.0);
        let two = cache.measure("WW", 19.0, Weight::SemiBold, 0.0);
        assert!(two.advance > one.advance);
        assert!(two.ink_right + 0.01 >= one.ink_right);
        assert!(
            one.above_baseline > 8.0 && one.above_baseline < 19.0,
            "cap height {:?}",
            one.above_baseline
        );
        let tracked = cache.measure("COOL", 13.0, Weight::SemiBold, 0.06);
        let plain = cache.measure("COOL", 13.0, Weight::SemiBold, 0.0);
        assert!(tracked.advance > plain.advance);
    }

    #[test]
    fn ten_thousand_distinct_chars_stay_within_the_glyph_cache_cap() {
        let mut cache = cache();
        let mut pixmap = Pixmap::new(8, 8).expect("pixmap");
        pixmap.fill(tiny_skia::Color::from_rgba8(0, 0, 0, 255));
        let style = TextStyle {
            px: 8.0,
            weight: Weight::SemiBold,
            color: TEXT,
            tracking_em: 0.0,
        };
        let pen = Pen {
            x: 0.0,
            baseline: 7.0,
        };
        for code in 1u32..=10_000 {
            let Some(ch) = char::from_u32(code) else {
                continue;
            };
            cache.draw(&mut pixmap, ch.encode_utf8(&mut [0; 4]), pen, style);
        }
        assert!(
            cache.cached_glyphs() <= 512,
            "cache grew to {}",
            cache.cached_glyphs()
        );
    }
}
