//! Layout numbers for A1 and A3. Drawing code reads these fields.
//!
//! Both variants grow bars outward and run time clockwise. The fields are the
//! mockup's "Bar geometry" and the measured text positions in `token-dial.html`.

use crate::present::Variant;

/// One layout. Dial placement, direction and radii are fields, not branches.
#[derive(Clone, Copy)]
pub(super) struct LayoutGeometry {
    /// Ring stroke midline.
    pub ring_radius: f32,
    /// Ring stroke width.
    pub ring_stroke: f32,
    /// Inner edge of a bar.
    pub base_radius: f32,
    /// Bar length at t = 0. `L = length_min + length_span · t`.
    pub length_min: f32,
    /// Added length at t = 1.
    pub length_span: f32,
    /// Gap between bars, as a px arc at [`Self::base_radius`].
    pub gap_px: f32,
    /// Corner radius cap. The drawn radius is `min(corner_px, width/2, L/2)`.
    pub corner_px: f32,
    /// Slot angle of each bar, degrees, newest first. Sums to 360.
    pub bar_deg: [f32; 24],
    /// "Now" dot, px from the centre toward 12 o'clock.
    pub now_radius: f32,
    /// "Now" dot circle radius.
    pub now_dot_radius: f32,
    /// Measured-zero dot circle radius. Its centre is one radius out from the base.
    pub zero_dot_radius: f32,
    /// Model names stay inside this radius.
    pub content_radius: f32,
    /// Top-left of the brain or the no-data mark.
    pub brain_x: f32,
    /// Top-left of the brain or the no-data mark.
    pub brain_y: f32,
    /// Drawn size of the 40 px brain sprite, and of the mark's 64 px box.
    pub brain_px: f32,
    /// Model line anchor. The centre when [`Self::model_middle`] is set.
    pub model_x: f32,
    /// Model baseline.
    pub model_baseline: f32,
    /// Model size before shrinking.
    pub model_px: f32,
    /// Smallest model size. Below this the name is truncated.
    pub model_px_floor: f32,
    /// Centre the model and the two-name stack on [`Self::model_x`].
    pub model_middle: bool,
    /// Size of each line when two names are stacked.
    pub stack_px: f32,
    /// Baselines of the two stacked names.
    pub stack_baselines: [f32; 2],
    /// Name baselines when a detail line follows and the name takes two lines.
    pub name_baselines: [f32; 2],
    /// Name baseline when a detail line follows and the name fits one line.
    pub name_single_baseline: f32,
    /// Name size when a detail line follows.
    pub name_px: f32,
    /// Detail line baseline (ctx, kv, quant, moe, fa).
    pub detail_baseline: f32,
    /// Detail line anchor, left edge or centre as [`Self::model_middle`].
    pub detail_x: f32,
    /// Detail line size.
    pub detail_px: f32,
    /// Brain size next to a name with detail.
    pub flow_brain_px: f32,
    /// Gap between the brain and the first name when the brain sits above it.
    pub flow_brain_gap: f32,
    /// Coolant, CPU, and GPU column centres.
    pub temp_x: [f32; 3],
    /// Temperature label baseline.
    pub temp_label_baseline: f32,
    /// Temperature value baseline.
    pub temp_value_baseline: f32,
    /// Temperature label size.
    pub temp_label_px: f32,
    /// Temperature value size.
    pub temp_value_px: f32,
    /// Temperature label tracking, em.
    pub temp_label_track: f32,
    /// A1 draws the tokens · 24 h chart where the load blocks were; A3 has no slot.
    pub show_chart: bool,
    /// Left edge of the chart plot area.
    pub chart_x: f32,
    /// Top of the chart plot area.
    pub chart_y: f32,
    /// Plot width.
    pub chart_w: f32,
    /// Plot height.
    pub chart_h: f32,
    /// Title row baseline (ceiling on the left, "tokens · 24h" on the right).
    pub chart_title_baseline: f32,
    /// Tick label baseline ("now · 1h · 6h · 24h").
    pub chart_tick_baseline: f32,
    /// Dial time-scale labels (5s · 15s · 1m · 5m · 30m) inside the bar bases.
    pub scale_labels: bool,
    /// Ring readout baseline, in the gap.
    pub readout_baseline: f32,
    /// Ring readout size.
    pub readout_px: f32,
    /// CPU/MEM baseline.
    pub footer_baseline: f32,
    /// CPU/MEM size.
    pub footer_px: f32,
    /// CPU/MEM tracking, em.
    pub footer_track: f32,
}

const BAR_DEG: [f32; 24] = [
    6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 15.0, 15.0, 20.0, 20.0, 20.0, 22.5, 22.5,
    22.5, 22.5, 24.0, 24.0, 24.0, 24.0, 24.0,
];

/// A1 Halo. Bars from r 118, the 12 px V3b ring (r 145–157), centre packed inside r 114.
pub(super) const A1: LayoutGeometry = LayoutGeometry {
    ring_radius: 151.0,
    ring_stroke: 12.0,
    base_radius: 118.0,
    length_min: 3.0,
    length_span: 21.0,
    gap_px: 2.4,
    corner_px: 3.0,
    bar_deg: BAR_DEG,
    now_radius: 112.0,
    now_dot_radius: 1.8,
    zero_dot_radius: 1.5,
    content_radius: 114.0,
    brain_x: 82.0,
    brain_y: 66.0,
    brain_px: 24.0,
    model_x: 110.0,
    model_baseline: 88.0,
    model_px: 13.0,
    model_px_floor: 11.0,
    model_middle: false,
    stack_px: 13.0,
    stack_baselines: [78.0, 93.0],
    name_baselines: [78.0, 93.0],
    name_px: 13.0,
    name_single_baseline: 88.0,
    detail_baseline: 106.0,
    detail_x: 160.0,
    detail_px: 10.0,
    flow_brain_px: 24.0,
    flow_brain_gap: 4.0,
    temp_x: [92.0, 160.0, 228.0],
    temp_label_baseline: 124.0,
    temp_value_baseline: 158.0,
    temp_label_px: 12.0,
    temp_value_px: 32.0,
    temp_label_track: 0.06,
    show_chart: true,
    chart_x: 85.0,
    chart_y: 174.0,
    chart_w: 150.0,
    chart_h: 32.0,
    chart_title_baseline: 170.0,
    chart_tick_baseline: 219.0,
    scale_labels: true,
    readout_baseline: 313.0,
    readout_px: 12.0,
    footer_baseline: 244.0,
    footer_px: 14.0,
    footer_track: 0.04,
};

/// A3 Dial-first. Bars from r 104, the 16 px ring, no chart. The time-scale
/// labels are off: at r 95 they would touch the name and GPU columns.
pub(super) const A3: LayoutGeometry = LayoutGeometry {
    ring_radius: 148.0,
    ring_stroke: 16.0,
    base_radius: 104.0,
    length_min: 4.0,
    length_span: 28.0,
    gap_px: 2.4,
    corner_px: 3.0,
    bar_deg: BAR_DEG,
    now_radius: 98.0,
    now_dot_radius: 1.8,
    zero_dot_radius: 1.5,
    content_radius: 100.0,
    brain_x: 147.0,
    brain_y: 76.0,
    brain_px: 26.0,
    model_x: 160.0,
    model_baseline: 122.0,
    model_px: 15.0,
    model_px_floor: 11.0,
    model_middle: true,
    stack_px: 15.0,
    stack_baselines: [114.0, 130.0],
    name_baselines: [99.0, 114.0],
    name_px: 13.0,
    name_single_baseline: 114.0,
    detail_baseline: 129.0,
    detail_x: 160.0,
    detail_px: 10.0,
    flow_brain_px: 20.0,
    flow_brain_gap: 4.0,
    temp_x: [91.0, 160.0, 229.0],
    temp_label_baseline: 150.0,
    temp_value_baseline: 187.0,
    temp_label_px: 13.0,
    temp_value_px: 34.0,
    temp_label_track: 0.06,
    show_chart: false,
    chart_x: 85.0,
    chart_y: 174.0,
    chart_w: 150.0,
    chart_h: 32.0,
    chart_title_baseline: 170.0,
    chart_tick_baseline: 219.0,
    scale_labels: false,
    readout_baseline: 313.0,
    readout_px: 12.0,
    footer_baseline: 222.0,
    footer_px: 14.0,
    footer_track: 0.04,
};

/// Geometry for the frame's variant. A1 is the default.
pub(super) fn for_variant(variant: Variant) -> &'static LayoutGeometry {
    match variant {
        Variant::A1 => &A1,
        Variant::A3 => &A3,
    }
}
