//! Centre content, between the dial and the disc mask.
//!
//! Positions come from [`super::geometry::LayoutGeometry`] (A1 or A3).
//! Fontdue's advances differ from Chromium; the baselines, sizes and palette
//! are the token-dial mockup's.

use tiny_skia::{
    FilterQuality, LineCap, LineJoin, Paint, Path, PathBuilder, Pixmap, PixmapPaint, Stroke,
    Transform,
};

use super::geometry::LayoutGeometry;
use super::text::{GlyphCache, Pen, TextStyle, Weight};
use super::{Assets, Brain, Frame, LABEL, NO_DATA, Rgb, TEXT, brain_for};
use crate::present::{Ai, View};
use llama_core::detail::ModelDetail;

/// Names are capped at this many characters before fitting.
/// `max_name_chars` has no ceiling.
const NAME_CHAR_CAP: usize = 64;
/// The no-data mark is drawn in this box, then scaled into the brain slot.
const MARK_BOX: f32 = 64.0;
/// Brain sprites are 40 px. [`LayoutGeometry::brain_px`] is the drawn size.
const SPRITE_PX: f32 = 40.0;

struct StatusLine {
    text: String,
    px: f32,
    x: f32,
    baseline: f32,
    middle: bool,
    color: Rgb,
}

/// Where the brain (or the no-data mark) goes, and how big.
#[derive(Clone, Copy)]
struct BrainSlot {
    x: f32,
    y: f32,
    px: f32,
}

struct Status {
    lines: Vec<StatusLine>,
    brain: BrainSlot,
}

struct Anchor {
    x: f32,
    baseline: f32,
    px: f32,
    tracking_em: f32,
    middle: bool,
}

/// Draw the centre onto `pixmap`. The caller has painted the ring and the dial
/// and applies the disc mask afterwards.
pub(super) fn draw(pixmap: &mut Pixmap, view: &View, assets: &mut Assets, geom: &LayoutGeometry) {
    draw_status(pixmap, view, assets, geom);
    draw_temps(pixmap, view, assets, geom);
    super::chart::draw(pixmap, view, geom, assets.text());
    draw_footer(pixmap, view, assets, geom);
}

fn draw_status(pixmap: &mut Pixmap, view: &View, assets: &mut Assets, geom: &LayoutGeometry) {
    let status = plan_status(view, assets.text(), geom);
    if view.ai == Ai::NoData {
        draw_mark(pixmap, status.brain);
    } else if let Some(brain) = brain_for(view.ai) {
        blit_brain(pixmap, assets, brain, status.brain);
    }
    let cache = assets.text();
    for line in &status.lines {
        draw_runs(
            cache,
            pixmap,
            &[(&line.text, Weight::SemiBold, line.color)],
            Anchor {
                x: line.x,
                baseline: line.baseline,
                px: line.px,
                tracking_em: 0.0,
                middle: line.middle,
            },
        );
    }
}

/// Text of every status line, top to bottom: names, then the detail line.
#[doc(hidden)]
pub fn status_text(view: &View, assets: &mut Assets) -> Vec<String> {
    let geom = super::geometry::for_variant(view.variant);
    plan_status(view, assets.text(), geom)
        .lines
        .into_iter()
        .map(|line| line.text)
        .collect()
}

fn fixed_brain(geom: &LayoutGeometry) -> BrainSlot {
    BrainSlot {
        x: geom.brain_x,
        y: geom.brain_y,
        px: geom.brain_px,
    }
}

fn plan_status(view: &View, cache: &GlyphCache, geom: &LayoutGeometry) -> Status {
    if view.ai == Ai::NoData {
        return single_line(cache, "no data", geom);
    }
    if view.ai == Ai::Down {
        return single_line(cache, "AI down", geom);
    }
    if view.ai == Ai::Idle || view.models.is_empty() {
        return single_line(cache, "no model", geom);
    }
    if view.models.len() == 2 && stacked_pair_fits(cache, &view.models[0], &view.models[1], geom) {
        return Status {
            lines: vec![
                text_line(
                    &view.models[0],
                    geom.stack_px,
                    geom.stack_baselines[0],
                    geom,
                ),
                text_line(
                    &view.models[1],
                    geom.stack_px,
                    geom.stack_baselines[1],
                    geom,
                ),
            ],
            brain: fixed_brain(geom),
        };
    }
    if view.models.len() == 1 {
        return plan_model(cache, &view.models[0], view.detail.as_ref(), geom);
    }
    single_line(cache, &count_label(view), geom)
}

fn text_line(text: &str, px: f32, baseline: f32, geom: &LayoutGeometry) -> StatusLine {
    StatusLine {
        text: text.to_owned(),
        px,
        x: geom.model_x,
        baseline,
        middle: geom.model_middle,
        color: TEXT,
    }
}

fn single_line(cache: &GlyphCache, text: &str, geom: &LayoutGeometry) -> Status {
    Status {
        lines: vec![fit_model_line(cache, text, geom)],
        brain: fixed_brain(geom),
    }
}

/// One loaded model: the name on one or two lines, then the detail line.
///
/// This is the accepted "Head & tail" design (T51 V1). The name fits one
/// line, else the most even two-line split at a space, else `head…` over
/// `tail`: the longest word prefix and the longest word suffix that fit.
/// The middle is what gets dropped; the end is never cut.
fn plan_model(
    cache: &GlyphCache,
    name: &str,
    detail: Option<&ModelDetail>,
    geom: &LayoutGeometry,
) -> Status {
    let (single, pair) = if detail.is_some() {
        (geom.name_single_baseline, geom.name_baselines)
    } else {
        (geom.model_baseline, geom.stack_baselines)
    };
    let px = if detail.is_some() {
        geom.name_px
    } else {
        geom.model_px
    };
    let name = cap_name(name.trim());
    let fits = |slot: usize, text: &str| line_fits(cache, text, geom, pair[slot], px);
    let width = |text: &str| cache.measure(text, px, Weight::SemiBold, 0.0).advance;
    let lines = wrap_name(&name, &fits, &width);
    let baselines: Vec<f32> = if lines.len() == 1 {
        vec![single]
    } else {
        pair.to_vec()
    };
    let mut out: Vec<StatusLine> = lines
        .iter()
        .zip(&baselines)
        .map(|(text, baseline)| text_line(text, px, *baseline, geom))
        .collect();
    let brain = match detail {
        Some(_) if geom.model_middle => brain_above(cache, &out, geom),
        _ => fixed_brain(geom),
    };
    if let Some(detail) = detail {
        let fits_detail = |text: &str| {
            line_fits_at(
                cache,
                text,
                geom,
                (geom.detail_x, true),
                geom.detail_baseline,
                geom.detail_px,
            )
        };
        let text = llama_core::detail::fitted(detail, fits_detail);
        if !text.is_empty() {
            out.push(StatusLine {
                text,
                px: geom.detail_px,
                x: geom.detail_x,
                baseline: geom.detail_baseline,
                middle: true,
                color: LABEL,
            });
        }
    }
    Status { lines: out, brain }
}

/// At most [`NAME_CHAR_CAP`] characters. A longer name keeps its start and
/// its end around `…`, so the end survives the cap too.
fn cap_name(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    if chars.len() <= NAME_CHAR_CAP {
        return name.to_owned();
    }
    let tail = NAME_CHAR_CAP / 2 - 1;
    let head = NAME_CHAR_CAP - tail - 1;
    let mut out: String = chars[..head].iter().collect();
    out.push('\u{2026}');
    out.extend(&chars[chars.len() - tail..]);
    out
}

/// Head-and-tail wrap. `fits(line, text)` checks line 0 or line 1, and
/// `width` is the drawn advance.
///
/// One line when the name fits line 0. Else the split at a space with the
/// smallest width difference where both halves fit. Else `head…` on line 0
/// and `tail` on line 1, from whole words; a word wider than its line is cut
/// by characters, the head keeping its start and the tail its end.
fn wrap_name(
    name: &str,
    fits: &impl Fn(usize, &str) -> bool,
    width: &impl Fn(&str) -> f32,
) -> Vec<String> {
    if name.is_empty() {
        return vec![String::new()];
    }
    if fits(0, name) {
        return vec![name.to_owned()];
    }
    let words: Vec<&str> = name.split(' ').filter(|word| !word.is_empty()).collect();
    if words.len() == 1 {
        return split_word(name, fits);
    }
    let mut best: Option<(f32, [String; 2])> = None;
    for split in 1..words.len() {
        let head = words[..split].join(" ");
        let tail = words[split..].join(" ");
        if !fits(0, &head) || !fits(1, &tail) {
            continue;
        }
        let score = (width(&head) - width(&tail)).abs();
        if best.as_ref().is_none_or(|(held, _)| score < *held) {
            best = Some((score, [head, tail]));
        }
    }
    if let Some((_, pair)) = best {
        return pair.to_vec();
    }
    let head_words = (1..=words.len())
        .take_while(|count| fits(0, &format!("{}\u{2026}", words[..*count].join(" "))))
        .last()
        .unwrap_or(0);
    let tail_words = (1..=words.len())
        .take_while(|count| fits(1, &words[words.len() - count..].join(" ")))
        .last()
        .unwrap_or(0);
    if head_words + tail_words >= words.len() {
        let split = (words.len() - tail_words).max(1);
        return vec![words[..split].join(" "), words[split..].join(" ")];
    }
    let head = if head_words > 0 {
        format!("{}\u{2026}", words[..head_words].join(" "))
    } else {
        cut_head(words[0], &|text| fits(0, text))
    };
    let tail = if tail_words > 0 {
        words[words.len() - tail_words..].join(" ")
    } else {
        cut_tail(words[words.len() - 1], &|text| fits(1, text))
    };
    vec![head, tail]
}

/// A single word too wide for line 0: its start on line 0 and the rest on
/// line 1, or `start…` over `…end` when the rest does not fit either.
fn split_word(word: &str, fits: &impl Fn(usize, &str) -> bool) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let head = (1..chars.len())
        .take_while(|count| fits(0, &chars[..*count].iter().collect::<String>()))
        .last()
        .unwrap_or(0);
    let rest: String = chars[head..].iter().collect();
    if head > 0 && fits(1, &rest) {
        return vec![chars[..head].iter().collect(), rest];
    }
    vec![
        cut_head(word, &|text| fits(0, text)),
        cut_tail(word, &|text| fits(1, text)),
    ]
}

/// Longest `start…` of `word` that fits. At least one character.
fn cut_head(word: &str, fits: &impl Fn(&str) -> bool) -> String {
    let chars: Vec<char> = word.chars().collect();
    let mut keep = chars.len();
    loop {
        let text = format!("{}\u{2026}", chars[..keep].iter().collect::<String>());
        if keep <= 1 || fits(&text) {
            return text;
        }
        keep -= 1;
    }
}

/// Longest `…end` of `word` that fits. At least one character.
fn cut_tail(word: &str, fits: &impl Fn(&str) -> bool) -> String {
    let chars: Vec<char> = word.chars().collect();
    let mut keep = chars.len();
    loop {
        let text = format!(
            "\u{2026}{}",
            chars[chars.len() - keep..].iter().collect::<String>()
        );
        if keep <= 1 || fits(&text) {
            return text;
        }
        keep -= 1;
    }
}

/// A3 puts the brain above the first name, at [`LayoutGeometry::flow_brain_px`].
fn brain_above(cache: &GlyphCache, lines: &[StatusLine], geom: &LayoutGeometry) -> BrainSlot {
    let Some(first) = lines.first() else {
        return fixed_brain(geom);
    };
    let top = first.baseline
        - cache
            .measure(&first.text, first.px, Weight::SemiBold, 0.0)
            .above_baseline;
    let px = geom.flow_brain_px;
    BrainSlot {
        x: 160.0 - px * 0.5,
        y: top - geom.flow_brain_gap - px,
        px,
    }
}

/// Shrink from the layout's model size down to its floor, then truncate with `…`.
///
/// Only the first [`NAME_CHAR_CAP`] characters are considered. The longest
/// prefix that fits is found by binary search, which needs the predicate to
/// be monotonic in prefix length.
fn fit_model_line(cache: &GlyphCache, text: &str, geom: &LayoutGeometry) -> StatusLine {
    let text = first_chars(text, NAME_CHAR_CAP);
    let mut px = geom.model_px;
    loop {
        if line_fits(cache, text, geom, geom.model_baseline, px) {
            return StatusLine {
                text: text.to_owned(),
                px,
                x: geom.model_x,
                baseline: geom.model_baseline,
                middle: geom.model_middle,
                color: TEXT,
            };
        }
        if px <= geom.model_px_floor {
            break;
        }
        px -= 1.0;
    }
    StatusLine {
        text: truncate_with_ellipsis(cache, text, geom, geom.model_px_floor),
        px: geom.model_px_floor,
        x: geom.model_x,
        baseline: geom.model_baseline,
        middle: geom.model_middle,
        color: TEXT,
    }
}

fn first_chars(text: &str, cap: usize) -> &str {
    match text.char_indices().nth(cap) {
        Some((index, _)) => &text[..index],
        None => text,
    }
}

fn truncate_with_ellipsis(
    cache: &GlyphCache,
    text: &str,
    geom: &LayoutGeometry,
    px: f32,
) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut best: Option<usize> = None;
    let mut lo = 0;
    let mut hi = chars.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if prefix_fits(cache, &chars, mid, geom, px) {
            best = Some(mid);
            lo = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            hi = mid;
        }
    }
    match best {
        Some(keep) => {
            let mut out: String = chars[..keep].iter().collect();
            out.push('\u{2026}');
            out
        }
        None => "\u{2026}".to_owned(),
    }
}

fn prefix_fits(
    cache: &GlyphCache,
    chars: &[char],
    keep: usize,
    geom: &LayoutGeometry,
    px: f32,
) -> bool {
    let mut candidate: String = chars[..keep].iter().collect();
    candidate.push('\u{2026}');
    line_fits(cache, &candidate, geom, geom.model_baseline, px)
}

fn count_label(view: &View) -> String {
    let count = if view.model_count >= 2 {
        u32::from(view.model_count)
    } else {
        u32::try_from(view.models.len()).unwrap_or(u32::from(u8::MAX))
    };
    format!("{count} models")
}

fn stacked_pair_fits(cache: &GlyphCache, top: &str, bottom: &str, geom: &LayoutGeometry) -> bool {
    line_fits(cache, top, geom, geom.stack_baselines[0], geom.stack_px)
        && line_fits(cache, bottom, geom, geom.stack_baselines[1], geom.stack_px)
}

/// The limit is the circle at the cap-height top of the line, where it is narrower
/// than at the baseline. `ink_right` includes the glyph bitmap, not only the advance.
fn line_fits(
    cache: &GlyphCache,
    text: &str,
    geom: &LayoutGeometry,
    baseline: f32,
    px: f32,
) -> bool {
    line_fits_at(
        cache,
        text,
        geom,
        (geom.model_x, geom.model_middle),
        baseline,
        px,
    )
}

/// [`line_fits`] for a line anchored at `(x, middle)` instead of the model anchor.
fn line_fits_at(
    cache: &GlyphCache,
    text: &str,
    geom: &LayoutGeometry,
    (x, middle): (f32, bool),
    baseline: f32,
    px: f32,
) -> bool {
    let measured = cache.measure(text, px, Weight::SemiBold, 0.0);
    let top = baseline - measured.above_baseline;
    let limit = chord_right(top, geom.content_radius);
    if middle {
        let left = x - measured.advance * 0.5;
        let right = left + measured.ink_right;
        left >= 320.0 - limit && right <= limit
    } else {
        x >= 320.0 - limit && x + measured.ink_right <= limit
    }
}

fn chord_right(y: f32, radius: f32) -> f32 {
    let dy = y - 160.0;
    let remain = radius * radius - dy * dy;
    if remain <= 1.0 {
        return 160.0;
    }
    160.0 + remain.sqrt()
}

fn draw_temps(pixmap: &mut Pixmap, view: &View, assets: &mut Assets, geom: &LayoutGeometry) {
    let labels = ["COOL", "CPU", "GPU"];
    let values = [view.coolant_c, view.cpu_c, view.gpu_c];
    let cache = assets.text();
    for (index, label) in labels.iter().enumerate() {
        let x = geom.temp_x[index];
        draw_runs(
            cache,
            pixmap,
            &[(label, Weight::SemiBold, LABEL)],
            Anchor {
                x,
                baseline: geom.temp_label_baseline,
                px: geom.temp_label_px,
                tracking_em: geom.temp_label_track,
                middle: true,
            },
        );
        let degrees = if cleared(view) { None } else { values[index] };
        let (text, color) = match degrees {
            Some(degrees) => (format!("{degrees}°"), TEXT),
            None => ("\u{2014}".to_owned(), NO_DATA),
        };
        draw_runs(
            cache,
            pixmap,
            &[(&text, Weight::ExtraBold, color)],
            Anchor {
                x,
                baseline: geom.temp_value_baseline,
                px: geom.temp_value_px,
                tracking_em: 0.0,
                middle: true,
            },
        );
    }
}

fn draw_footer(pixmap: &mut Pixmap, view: &View, assets: &mut Assets, geom: &LayoutGeometry) {
    let cpu_pct = if cleared(view) { None } else { view.cpu_pct };
    let mem_pct = if cleared(view) { None } else { view.mem_pct };
    let cpu = percent_text(cpu_pct);
    let mem = percent_text(mem_pct);
    // Parent run is label grey, weight 600. Tspans recolour the values only.
    draw_runs(
        assets.text(),
        pixmap,
        &[
            ("CPU ", Weight::SemiBold, LABEL),
            (&cpu, Weight::SemiBold, value_color(cpu_pct)),
            (" \u{00B7} MEM ", Weight::SemiBold, LABEL),
            (&mem, Weight::SemiBold, value_color(mem_pct)),
        ],
        Anchor {
            x: 160.0,
            baseline: geom.footer_baseline,
            px: geom.footer_px,
            tracking_em: geom.footer_track,
            middle: true,
        },
    );
}

fn value_color(value: Option<impl Copy>) -> Rgb {
    if value.is_some() { TEXT } else { NO_DATA }
}

/// Watch-down blanks the live numbers. The mark, the empty blocks, and the
/// status words are drawn from [`Ai::NoData`] itself.
fn cleared(view: &View) -> bool {
    view.ai == Ai::NoData
}

/// Ring value for this frame. Watch-down keeps the track and drops the arc,
/// even when the view still carries a percent.
pub(super) fn ring(view: &View) -> Option<u8> {
    if cleared(view) { None } else { view.ring_pct }
}

fn percent_text(value: Option<u8>) -> String {
    match value {
        Some(value) => format!("{value}%"),
        None => "\u{2014}".to_owned(),
    }
}

fn solid(color: Rgb) -> Paint<'static> {
    let mut paint = Paint::default();
    paint.set_color_rgba8(color.r, color.g, color.b, 255);
    paint.anti_alias = true;
    paint
}

fn blit_brain(pixmap: &mut Pixmap, assets: &Assets, brain: Brain, slot: BrainSlot) {
    let scale = slot.px / SPRITE_PX;
    let transform = Transform::from_scale(scale, scale).post_translate(slot.x, slot.y);
    let paint = PixmapPaint {
        quality: FilterQuality::Bilinear,
        ..PixmapPaint::default()
    };
    pixmap.draw_pixmap(0, 0, assets.sprite(brain).as_ref(), &paint, transform, None);
}

/// Circle with a horizontal bar, in the brain slot. Icon space is 64 px.
fn draw_mark(pixmap: &mut Pixmap, slot: BrainSlot) {
    let scale = slot.px / MARK_BOX;
    let cx = slot.x + 32.0 * scale;
    let cy = slot.y + 32.0 * scale;
    let radius = 22.0 * scale;
    let width = 5.0 * scale;
    if let Some(path) = circle(cx, cy, radius) {
        stroke_mark(pixmap, &path, width, LineCap::Butt);
    }
    let mut bar = PathBuilder::new();
    bar.move_to(slot.x + 21.0 * scale, cy);
    bar.line_to(slot.x + 43.0 * scale, cy);
    if let Some(path) = bar.finish() {
        stroke_mark(pixmap, &path, width, LineCap::Round);
    }
}

fn circle(cx: f32, cy: f32, radius: f32) -> Option<Path> {
    if radius <= 0.0 {
        return None;
    }
    let mut path = PathBuilder::new();
    path.push_circle(cx, cy, radius);
    path.finish()
}

fn stroke_mark(pixmap: &mut Pixmap, path: &Path, width: f32, cap: LineCap) {
    let stroke = Stroke {
        width,
        miter_limit: 4.0,
        line_cap: cap,
        line_join: LineJoin::Round,
        dash: None,
    };
    pixmap.stroke_path(path, &solid(LABEL), &stroke, Transform::identity(), None);
}

fn draw_runs(
    cache: &mut GlyphCache,
    pixmap: &mut Pixmap,
    parts: &[(&str, Weight, Rgb)],
    anchor: Anchor,
) {
    if !anchor.px.is_finite() || anchor.px <= 0.0 {
        return;
    }
    let tracking = anchor.px * anchor.tracking_em;
    let mut advance = 0.0;
    let mut runs = 0_u32;
    for (text, weight, _) in parts {
        if text.is_empty() {
            continue;
        }
        if runs > 0 {
            advance += tracking;
        }
        advance += cache
            .measure(text, anchor.px, *weight, anchor.tracking_em)
            .advance;
        runs += 1;
    }
    let mut x = if anchor.middle {
        anchor.x - advance * 0.5
    } else {
        anchor.x
    };
    let mut drawn = false;
    for (text, weight, color) in parts {
        if text.is_empty() {
            continue;
        }
        if drawn {
            x += tracking;
        }
        drawn = true;
        let width = cache
            .measure(text, anchor.px, *weight, anchor.tracking_em)
            .advance;
        cache.draw(
            pixmap,
            text,
            Pen {
                x,
                baseline: anchor.baseline,
            },
            TextStyle {
                px: anchor.px,
                weight: *weight,
                color: *color,
                tracking_em: anchor.tracking_em,
            },
        );
        x += width;
    }
}

// TODO(dev-deps): move these helpers into the integration test once serde_json
// and png are dev-dependencies. Cargo.toml is outside this ticket's allowed paths.
/// Parse one [`View`] JSON snapshot.
#[doc(hidden)]
pub fn view_from_json(bytes: &[u8]) -> Result<View, String> {
    serde_json::from_slice(bytes).map_err(|err| err.to_string())
}

/// Encode one frame as an 8-bit RGBA PNG. Alpha is opaque, so the premultiplied
/// buffer is straight RGBA.
#[doc(hidden)]
pub fn frame_png(frame: &Frame) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, frame.0.width(), frame.0.height());
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|err| err.to_string())?;
    writer
        .write_image_data(frame.0.data())
        .map_err(|err| err.to_string())?;
    writer.finish().map_err(|err| err.to_string())?;
    Ok(bytes)
}

/// Count pixels where any channel of `frame` differs from an 8-bit RGBA PNG by
/// more than `tolerance`. Alpha is a channel. The frame is opaque, so its
/// premultiplied buffer matches straight RGBA.
#[doc(hidden)]
pub fn png_channel_disagreements(
    frame: &Frame,
    png_bytes: &[u8],
    tolerance: u8,
) -> Result<usize, String> {
    let decoded = decode_rgba(png_bytes)?;
    if decoded.width != frame.0.width() || decoded.height != frame.0.height() {
        return Err(format!(
            "png is {}x{}, frame is {}x{}",
            decoded.width,
            decoded.height,
            frame.0.width(),
            frame.0.height()
        ));
    }
    let got = frame.0.data();
    let (got_px, got_tail) = got.as_chunks::<4>();
    let (png_px, png_tail) = decoded.bytes.as_chunks::<4>();
    if !got_tail.is_empty() || !png_tail.is_empty() || got_px.len() != png_px.len() {
        return Err(format!(
            "png has {} bytes, frame has {}",
            decoded.bytes.len(),
            got.len()
        ));
    }
    let mut count = 0;
    for (left, right) in got_px.iter().zip(png_px) {
        if left
            .iter()
            .zip(right)
            .any(|(a, b)| a.abs_diff(*b) > tolerance)
        {
            count += 1;
        }
    }
    Ok(count)
}

struct Decoded {
    width: u32,
    height: u32,
    bytes: Vec<u8>,
}

fn decode_rgba(bytes: &[u8]) -> Result<Decoded, String> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().map_err(|err| err.to_string())?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| "png is too large".to_owned())?;
    let mut buffer = vec![0; size];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|err| err.to_string())?;
    if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
        return Err(format!(
            "need 8-bit RGBA, got {:?} {:?}",
            info.bit_depth, info.color_type
        ));
    }
    buffer.truncate(info.buffer_size());
    Ok(Decoded {
        width: info.width,
        height: info.height,
        bytes: buffer,
    })
}
