//! tty11 dashboard. A pure function of [`TtyModel`], columns and rows.
//!
//! Phase B builds the grid from [`Term::cols`](super::term::Term::cols) and
//! [`Term::rows`](super::term::Term::rows). The last row and the last column
//! stay blank so the console cannot scroll.

use super::chart::{self, ChartBucket, Ink};
use super::ctx_history::{self, CtxPoint};
use super::grid::{C16, Cell, Grid};
use super::sanitize::sanitize;
use crate::collector::LoadSource;
use crate::config::{ChartGlyphs, MAX_FAN_LABEL};
use crate::resets::ResetReason;
use crate::sources::fans::{FanPanel, FanReading, mode_word};
use crate::sources::temps::{Level, TempGroup, TempPanel};

/// How many of `new_chars` are visible on frame `frame` of a 10-frame second.
///
/// Frame 0 shows nothing new. Frame 10 shows every new character. The counts
/// in between are spaced with integer division, so the characters land evenly.
pub fn replay_shown(new_chars: usize, frame: u32) -> usize {
    new_chars.saturating_mul(frame as usize) / 10
}

/// What the header pip says.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WatchState {
    Generating,
    Ready,
    AiDown,
    Starting,
    /// `[llama] enabled = false`: llama-swap is never polled.
    NoLlama,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HealthStatus {
    Ok,
    Idle,
    Down,
    Pending,
    Absent,
}

#[derive(Clone, Debug)]
pub struct HealthSeg {
    pub name: String,
    pub status: HealthStatus,
    pub note: String,
}

#[derive(Clone, Debug)]
pub struct Slot {
    pub id: u32,
    pub generating: bool,
    pub done: u64,
    pub total: u64,
    pub decoded: u64,
    /// Prompt tokens counted toward context. `None` when `/slots` omitted both
    /// `n_prompt_tokens` and the cached-prompt count.
    pub ctx_prompt: Option<u64>,
    /// Slot `n_ctx`. `None` when the key is absent. `Some(0)` is a real zero.
    pub n_ctx: Option<u64>,
    /// Context history buckets, newest first (T53). Empty draws a blank
    /// sparkline.
    pub ctx_history: Vec<CtxPoint>,
}

#[derive(Clone, Debug, Default)]
pub struct Activity {
    pub live: bool,
    pub id: u32,
    pub time: String,
    pub source: String,
    pub model: String,
    pub input_tok: u64,
    pub cached_tok: u64,
    pub output_tok: u64,
    /// `None` when llama-swap reported no rate (a backend without timings)
    /// and the engine measured none either.
    pub prompt_tps: Option<f64>,
    pub gen_tps: Option<f64>,
    /// The rate is the engine's window measure, not the request's own
    /// timing (#35): drawn as `~1,234` / `~45.3`.
    pub prompt_measured: bool,
    pub gen_measured: bool,
    pub dur: String,
    pub err: bool,
    /// The model's context size (#75): the bar's full width. `None` scales
    /// the bar to the largest row shown and marks it `~`.
    pub n_ctx: Option<u64>,
    /// A request still running (#75), from live per-request numbers.
    /// `input_tok` is then its whole prompt (or, while `open`, the prompt
    /// tokens held so far), `output_tok` the tokens so far.
    pub inflight: Option<InFlight>,
}

/// The live half of an in-flight RECENT row (#75).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InFlight {
    /// Decoding (`gen`); else still in prefill (`pp`).
    pub decoding: bool,
    /// Prompt tokens computed so far, beyond the cached part.
    pub processed: u64,
    /// The whole prompt is not known yet (#78): `input_tok` is a lower
    /// bound, drawn `12,345+`, and the bar has no target track.
    pub open: bool,
    /// A context reset just before this request, which started it from
    /// zero: its letter shows after `pp`, as in SLOTS.
    pub reset: Option<ResetReason>,
}

/// One item on a SETUP row (#52): `kv q8_0`, drawn after `sep`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SetupItem {
    /// Separator before the item; the row's first item draws none.
    pub sep: String,
    /// Display text from a `[setup]` rule. Printable ASCII or `·`.
    pub text: String,
    /// A rule's `default` (nothing was set): drawn grey.
    pub dim: bool,
}

/// One SETUP row: its label and items, most important first.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SetupRow {
    pub label: String,
    pub items: Vec<SetupItem>,
}

/// The SETUP block (#52) under the meters: the model generating now, or
/// else the one used last, and its settings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SetupView {
    /// llama-swap model id, sanitised.
    pub id: String,
    /// llama-swap `name` (or alias). Drawn after the id when it fits and
    /// differs from it.
    pub name: String,
    /// Other models loaded: `+N` at the right of the title.
    pub more: usize,
    /// Rows in order; the block drops the last ones when short of room.
    pub rows: Vec<SetupRow>,
}

/// Plain data for one frame. T22 fills it. Text fields are untrusted and are
/// passed through [`sanitize`] before they are placed.
#[derive(Clone, Debug)]
pub struct TtyModel {
    pub state: WatchState,
    pub host: String,
    pub model_name: String,
    /// Header detail of the first model: its engine (#33). The settings
    /// themselves are in [`Self::setup`] since #52. Empty draws nothing.
    pub model_detail: String,
    /// The first model has sat in llama-swap `stopping` for over a minute.
    pub model_stuck: bool,
    pub slots_line: String,
    pub swap_line: String,
    pub cool_c: Option<i32>,
    pub cpu_c: Option<i32>,
    pub gpu_c: Option<i32>,
    pub clock: String,
    pub cpu_pct: Option<f64>,
    pub cpu_cores: Option<u32>,
    pub gpu_pct: Option<f64>,
    pub vram_used_gb: Option<f64>,
    pub vram_total_gb: Option<f64>,
    pub mem_used_gb: Option<f64>,
    pub mem_total_gb: Option<f64>,
    pub power_w: Option<f64>,
    pub power_limit_w: Option<f64>,
    pub load_pct: Option<f64>,
    /// Bottleneck activity for the ACTIVITY meter. `None` draws `--`.
    pub activity_pct: Option<f64>,
    /// Device that set `activity_pct`: `gpu`, `cpu` or `util`.
    pub activity_src: Option<LoadSource>,
    /// Watts of that device, shown on the ACTIVITY meter.
    pub activity_w: Option<f64>,
    pub gen_tps: Option<f64>,
    pub prompt_tps: Option<f64>,
    pub prompt_last: Option<u64>,
    pub gen_ceiling: f64,
    pub prompt_ceiling: f64,
    pub slots: Vec<Slot>,
    /// One SLOTS line per ready model without `/slots` (T72), such as
    /// `sglang  running 1 · queued 0 · KV 37 %`. Drawn under the slot rows.
    pub backend_lines: Vec<String>,
    /// Drawn in IN and OUT while they are empty. Empty draws nothing.
    pub text_note: String,
    pub requests: Vec<Activity>,
    pub in_title: String,
    pub out_title: String,
    pub in_lines: Vec<String>,
    pub out_lines: Vec<String>,
    /// Sanitised cells of the output tail already on screen before this poll.
    /// With [`Self::replay_frame`], frame 0 keeps this prefix and the new
    /// characters spread over the following frames. `None` on the frame shows
    /// the whole tail.
    pub out_shown: usize,
    /// Which frame of the 10-frame replay is on screen. `None` shows the tail
    /// in full.
    pub replay_frame: Option<u32>,
    /// `tty.show_text`. False draws no IN/OUT panels, gives their rows to the
    /// chart and RECENT, and tags the header `text off`.
    pub show_text: bool,
    pub down_since: String,
    pub health: Vec<HealthSeg>,
    pub snapshot: Option<u64>,
    pub snapshot_age: String,
    pub errors: u64,
    pub uptime: String,
    /// Oldest-first token-rate buckets for the activity chart. Empty is no data.
    pub chart: Vec<ChartBucket>,
    /// Seconds of history in one chart column. Drives the "-Nm" axis label.
    pub chart_bucket_s: u64,
    /// Half-block or eighth-block chart (`tty.chart_glyphs`).
    pub chart_glyphs: ChartGlyphs,
    /// FANS panel. `None` when `[fans]` is off, and nothing is drawn.
    pub fans: Option<FanPanel>,
    /// TEMPS panel (#74). `None` when `[temps]` is off, and nothing is drawn.
    pub temps: Option<TempPanel>,
    /// Hours the SLOTS sparklines span (`tty.ctx_history_h`).
    pub ctx_history_h: u32,
    /// SETUP block under the meters (#52). `None` collapses it.
    pub setup: Option<SetupView>,
}

/// Narrowest console the dashboard draws: the RECENT columns' floor.
pub const MIN_COLS: u16 = 160;
/// Shortest console the dashboard draws (#7). Rows 0-14 are the header, a
/// rule, the six meters, ACTIVITY and the SETUP rows that fit beside SLOTS
/// (#52); then a rule, the RECENT header, 4 request rows, the legend and
/// the RECENT rule; then the health rule, the health line and the blank
/// last row: 26. Each SLOTS row past the first pushes RECENT down a row,
/// and it shows fewer requests. From [`FULL_ROWS`] SETUP takes up to two
/// more rows.
pub const MIN_ROWS: u16 = 26;
/// From this height the text-on layout keeps IN/OUT: the chart shrinks or
/// hides first. Below it panels drop in order: IN/OUT, then FANS, then the
/// chart shrinks (the text-off split of the freed rows).
pub const FULL_ROWS: u16 = 48;

/// Draw one frame. Below [`MIN_COLS`]×[`MIN_ROWS`] the grid is the single
/// "tty too small" line.
pub fn layout(model: &TtyModel, cols: u16, rows: u16) -> Grid {
    let mut grid = Grid::new(cols, rows);
    if cols < MIN_COLS || rows < MIN_ROWS {
        let msg = format!("llama-watch: tty too small ({cols}x{rows}, need {MIN_COLS}x{MIN_ROWS})");
        paint_str(&mut grid, 0, 0, &msg, C16::White, C16::Black);
        return grid;
    }
    let mut g = Geom::new(cols, rows);
    g.slot_last = g.slot_label.saturating_add(slot_body_rows(model));
    draw_header(&mut grid, model, &g);
    draw_rule(&mut grid, 1, g.cols);
    draw_meters(&mut grid, model, &g);
    draw_rates(&mut grid, model, &g);
    draw_slots(&mut grid, model, &g);
    let stats_end = g.meter_last.max(g.slot_last);
    let mid_rule = draw_setup(&mut grid, model, &g, stats_end + 2);
    draw_rule(&mut grid, mid_rule, g.cols);
    let req_header = mid_rule + 1;
    if model.show_text && (rows >= FULL_ROWS || text_fits(model, &g, req_header)) {
        let req_rows = request_slots(rows);
        draw_requests(&mut grid, model, &g, req_header, req_rows);
        let req_rule = req_header + 1 + req_rows + 1;
        draw_rule(&mut grid, req_rule, g.cols);
        let chart_h = chart_height(req_rule, rows);
        if chart_h >= 5 {
            draw_chart(&mut grid, model, &g, req_rule + 1, chart_h);
        }
        draw_text(&mut grid, model, &g, req_rule.saturating_add(chart_h));
    } else {
        // Text off, or a short screen without room for IN/OUT under a
        // full chart: the IN/OUT rows go to the chart and RECENT.
        draw_text_off(&mut grid, model, &g, req_header);
    }
    let health_rule = rows - 3;
    // A long SLOTS list on a short screen may reach these rows. The health
    // rule and line always draw on blank cells.
    for row in [health_rule, rows - 2] {
        fill_span(&mut grid, 0, cols - 2, row, ' ', C16::White, C16::Black);
    }
    draw_rule(&mut grid, health_rule, g.cols);
    draw_health(&mut grid, model, &g, rows - 2);
    grid
}

/// Below [`FULL_ROWS`]: IN/OUT stay only while a full chart, IN and OUT
/// with three rows each and (on a narrow screen) the FANS block all fit.
fn text_fits(model: &TtyModel, g: &Geom, req_header: u16) -> bool {
    let req_rule = req_header.saturating_add(request_slots(g.rows) + 2);
    let below = g
        .rows
        .saturating_sub(3)
        .saturating_sub(req_rule.saturating_add(1));
    let fans = match &model.fans {
        Some(panel) if g.cols < FANS_SIDE_COLS => 1 + fans_block_rows(panel),
        _ => 0,
    };
    below >= CHART_PREFERRED + TEXT_FLOOR_SPAN + fans
}

struct Geom {
    cols: u16,
    rows: u16,
    tall: bool,
    pitch: u16,
    bar_rows: u16,
    meter_last: u16,
    left_bar: u16,
    left_bar_w: u16,
    right: u16,
    half: u16,
    slot_label: u16,
    slot_last: u16,
}

impl Geom {
    fn new(cols: u16, rows: u16) -> Self {
        let tall = rows >= 90;
        // #52: one meter per row; the bars' short glyph keeps the rows
        // apart. Tall screens keep two-row bars with a blank row between.
        let pitch: u16 = if tall { 3 } else { 1 };
        let bar_rows: u16 = if tall { 2 } else { 1 };
        let meter_last = 3 + 5 * pitch + (bar_rows - 1);
        let left_bar = 27u16;
        let left_bar_end = cols / 2 - 3;
        let left_bar_w = left_bar_end - left_bar + 1;
        let right = cols / 2 + 1;
        let right_end = cols - 3;
        let avail = right_end - right + 1;
        let half = (avail - 3) / 2;
        let label_row = 3u16;
        let digit_row = if tall { label_row + 2 } else { label_row + 1 };
        let digit_h: u16 = if tall { 10 } else { 5 };
        let rate_bar = digit_row + digit_h + 1;
        let tick_row = rate_bar + bar_rows;
        let slot_label = tick_row + bar_rows;
        let slot_last = slot_label + 1;
        Self {
            cols,
            rows,
            tall,
            pitch,
            bar_rows,
            meter_last,
            left_bar,
            left_bar_w,
            right,
            half,
            slot_label,
            slot_last,
        }
    }

    fn prompt_x(&self) -> u16 {
        self.right.saturating_add(self.half).saturating_add(3)
    }

    /// Row of the ACTIVITY bar, right under LOAD.
    fn activity_row(&self) -> u16 {
        3 + 5 * self.pitch + self.bar_rows
    }

    /// Last column of the left half: where the meter bars end.
    fn left_end(&self) -> u16 {
        self.left_bar + self.left_bar_w - 1
    }

    /// Single-spaced meters draw whole cells of a short block (#52).
    fn meter_cells(&self, glyphs: ChartGlyphs) -> Option<char> {
        (self.pitch == 1).then_some(meter_glyph(glyphs))
    }
}

/// The lower seven-eighths block `▇` with a llama-hack font, which leaves
/// an eighth of a cell (3 pixels) between stacked meters. eurlatgr lacks
/// it, so `chart_glyphs = "halves"` (the setting that says which font is
/// loaded) uses the lower half `▄` it does have (#52).
#[must_use]
pub fn meter_glyph(glyphs: ChartGlyphs) -> char {
    match glyphs {
        ChartGlyphs::Eighths => '\u{2587}',
        ChartGlyphs::Halves => '\u{2584}',
    }
}

fn slot_body_rows(model: &TtyModel) -> u16 {
    match model.state {
        WatchState::AiDown | WatchState::Starting | WatchState::NoLlama => 1,
        _ => u16::try_from(model.slots.len() + model.backend_lines.len()).unwrap_or(u16::MAX),
    }
}

fn draw_header(grid: &mut Grid, model: &TtyModel, g: &Geom) {
    let host_len = model.host.chars().count().min(32);
    paint_fit(
        grid,
        1,
        0,
        &model.host,
        C16::BrightWhite,
        C16::Black,
        host_len,
    );
    let product_at = col_u16(1usize.saturating_add(host_len).saturating_add(2));
    paint_str(
        grid,
        product_at,
        0,
        "llama-bored",
        C16::BrightBlack,
        C16::Black,
    );
    let (pip_fg, word, word_fg, word_bg) = match model.state {
        WatchState::Generating => (C16::BrightRed, "GENERATING", C16::BrightRed, C16::Black),
        WatchState::Ready => (C16::Green, "READY", C16::Green, C16::Black),
        WatchState::AiDown => (C16::Yellow, " AI DOWN ", C16::Black, C16::Yellow),
        WatchState::Starting => (C16::BrightBlack, "STARTING", C16::BrightBlack, C16::Black),
        WatchState::NoLlama => (C16::BrightBlack, "NO LLAMA", C16::BrightBlack, C16::Black),
    };
    paint(grid, 24, 0, '█', pip_fg, C16::Black);
    paint_str(grid, 26, 0, word, word_fg, word_bg);
    // The badge already includes its trailing pad, so it needs one less gap.
    let gap: usize = if word_bg == C16::Black { 2 } else { 1 };
    let mut x = col_u16(
        26usize
            .saturating_add(word.chars().count())
            .saturating_add(gap),
    );
    let temps = if g.cols >= 200 {
        format!(
            "COOL {}   CPU {}   GPU {}     {}",
            temp(model.cool_c),
            temp(model.cpu_c),
            temp(model.gpu_c),
            model.clock
        )
    } else {
        model.clock.clone()
    };
    let fg = if model.state == WatchState::Starting {
        C16::BrightBlack
    } else {
        C16::BrightWhite
    };
    let end = usize::from(g.cols.saturating_sub(2));
    let len = temps.chars().count();
    let start = col_u16(end.saturating_add(1).saturating_sub(len));
    // #39: header text left of the clock ends two cells before it.
    let limit = usize::from(start).saturating_sub(CLOCK_GAP);
    x = paint_model(grid, x, model, limit);
    x = paint_field(
        grid,
        x,
        0,
        "slots",
        &model.slots_line,
        name_fg(model),
        limit,
    );
    let tag_at = paint_field(grid, x, 0, "swap", &model.swap_line, name_fg(model), limit);
    paint_str(grid, start, 0, &temps, fg, C16::Black);
    if !model.show_text {
        let tag_end = usize::from(tag_at).saturating_add(TEXT_OFF_TAG.chars().count());
        if tag_end.saturating_add(2) <= usize::from(start) {
            paint_str(grid, tag_at, 0, TEXT_OFF_TAG, C16::BrightBlack, C16::Black);
        }
    }
}

/// Header tag while `tty.show_text = false`.
const TEXT_OFF_TAG: &str = "text off";

fn name_fg(model: &TtyModel) -> C16 {
    match model.state {
        WatchState::AiDown | WatchState::Starting | WatchState::NoLlama => C16::BrightBlack,
        _ => C16::BrightWhite,
    }
}

fn temp(v: Option<i32>) -> String {
    match v {
        Some(n) => format!("{n}C"),
        None => "--".to_string(),
    }
}

/// Longest model name in the header, the full-name width of the snapshot.
const MODEL_NAME_CAP: usize = llama_core::detail::MAX_FULL_NAME_CHARS;
/// Longest detail string in the header.
const MODEL_DETAIL_CAP: usize = 48;

/// Header tag for a model stuck in llama-swap `stopping`.
const STUCK_TAG: &str = "stopping (stuck?)";

/// Cells kept blank between the header text and the clock (#39).
const CLOCK_GAP: usize = 2;

/// `model <full name>  <detail>`, the detail in grey. The detail ends by
/// column `limit` (exclusive), dropping whole trailing ` · item`s (#39).
fn paint_model(grid: &mut Grid, x: u16, model: &TtyModel, limit: usize) -> u16 {
    let label = "model";
    let label_len = label.chars().count();
    paint_fit(grid, x, 0, label, C16::BrightBlack, C16::Black, label_len);
    let vx = usize::from(x).saturating_add(label_len).saturating_add(1);
    let drawn = model.model_name.chars().count().min(MODEL_NAME_CAP);
    paint_fit(
        grid,
        col_u16(vx),
        0,
        &model.model_name,
        name_fg(model),
        C16::Black,
        drawn,
    );
    let mut end = vx.saturating_add(drawn);
    if model.model_stuck {
        let sx = end.saturating_add(2);
        paint_str(grid, col_u16(sx), 0, STUCK_TAG, C16::Yellow, C16::Black);
        end = sx.saturating_add(STUCK_TAG.chars().count());
    }
    if !model.model_detail.is_empty() {
        let dx = end.saturating_add(2);
        let room = limit.saturating_sub(dx).min(MODEL_DETAIL_CAP);
        // One item longer than the room is cut at the room, as before.
        let detail = match fit_items(&model.model_detail, room) {
            "" => model.model_detail.as_str(),
            items => items,
        };
        let shown = paint_detail(grid, dx, 0, detail, room);
        if shown > 0 {
            end = dx.saturating_add(shown);
        }
    }
    col_u16(end.saturating_add(3))
}

/// Grey detail text. Printable ASCII, the `·` separator and `…` are drawn,
/// any other scalar is `?`. Returns the cells used.
fn paint_detail(grid: &mut Grid, col: usize, row: usize, text: &str, cap: usize) -> usize {
    paint_detail_fg(grid, col, row, text, cap, C16::BrightBlack)
}

/// [`paint_detail`] in `fg`.
fn paint_detail_fg(
    grid: &mut Grid,
    col: usize,
    row: usize,
    text: &str,
    cap: usize,
    fg: C16,
) -> usize {
    let mut drawn = 0;
    for ch in text.chars().take(cap) {
        let ch = if matches!(ch, '\u{00B7}' | '\u{2026}') || ('\u{20}'..='\u{7e}').contains(&ch) {
            ch
        } else {
            '?'
        };
        paint_at(grid, col.saturating_add(drawn), row, ch, fg, C16::Black);
        drawn += 1;
    }
    drawn
}

/// The longest run of whole leading ` · `-separated items of `text` that
/// fits in `room` cells. Empty when not even the first item fits.
fn fit_items(text: &str, room: usize) -> &str {
    const SEP: &str = " \u{00B7} ";
    let mut fit = "";
    let mut from = 0;
    loop {
        let cut = text[from..].find(SEP).map_or(text.len(), |at| from + at);
        if text[..cut].chars().count() > room {
            return fit;
        }
        fit = &text[..cut];
        if cut == text.len() {
            return fit;
        }
        from = cut + SEP.len();
    }
}

/// `label value`, skipped when it would end past column `limit` (#39).
fn paint_field(
    grid: &mut Grid,
    x: u16,
    row: u16,
    label: &str,
    value: &str,
    value_fg: C16,
    limit: usize,
) -> u16 {
    let label_len = label.chars().count();
    let field_end = usize::from(x)
        .saturating_add(label_len)
        .saturating_add(1)
        .saturating_add(value.chars().count().min(32));
    if field_end > limit {
        return x;
    }
    paint_fit(grid, x, row, label, C16::BrightBlack, C16::Black, label_len);
    let vx = usize::from(x).saturating_add(label_len).saturating_add(1);
    let drawn = value.chars().count().min(32);
    paint_fit(grid, col_u16(vx), row, value, value_fg, C16::Black, drawn);
    col_u16(vx.saturating_add(drawn).saturating_add(3))
}

fn col_u16(col: usize) -> u16 {
    u16::try_from(col).unwrap_or(u16::MAX)
}

fn draw_rule(grid: &mut Grid, row: u16, cols: u16) {
    if row + 1 >= grid.rows() {
        return;
    }
    for col in 1..cols.saturating_sub(1) {
        paint(grid, col, row, '-', C16::BrightBlack, C16::Black);
    }
}

/// SETUP rows (title included) on a screen under 60 rows, and from 60.
const SETUP_ROWS: u16 = 6;
const SETUP_ROWS_TALL: u16 = 8;
/// SETUP row labels start here; their items at [`SETUP_VALUE_X`].
const SETUP_LABEL_X: u16 = 4;
const SETUP_VALUE_X: u16 = 13;
const SETUP_TITLE: &str = "SETUP";

/// The SETUP block (#52) under ACTIVITY, a blank row below it, left
/// aligned under the meters. Returns the mid rule's row: `base_rule`
/// (the row the meters and SLOTS want) or just under the block.
///
/// The rows beside SLOTS are free. From [`FULL_ROWS`] the block may take
/// more, up to [`SETUP_ROWS`] in all ([`SETUP_ROWS_TALL`] from 60 rows),
/// pushing RECENT down; a shorter screen keeps its panels (#7) and the
/// block gets only the free rows. The last rows drop first. No view
/// (nothing loaded, llama-swap down) draws nothing and leaves the rule
/// where it was.
fn draw_setup(grid: &mut Grid, model: &TtyModel, g: &Geom, base_rule: u16) -> u16 {
    let Some(view) = &model.setup else {
        return base_rule;
    };
    if !matches!(model.state, WatchState::Generating | WatchState::Ready) {
        return base_rule;
    }
    let top = g
        .activity_row()
        .saturating_add(g.bar_rows)
        .saturating_add(1);
    let free = base_rule.saturating_sub(top);
    let cap = if g.rows >= 60 {
        SETUP_ROWS_TALL
    } else {
        SETUP_ROWS
    };
    let budget = if g.rows >= FULL_ROWS {
        free.max(cap)
    } else {
        free
    };
    if budget == 0 {
        return base_rule;
    }
    let right = usize::from(g.left_end());
    draw_setup_title(grid, view, top, right);
    let mut used: u16 = 1;
    for row in &view.rows {
        if used >= budget {
            break;
        }
        draw_setup_row(grid, row, top.saturating_add(used), right);
        used += 1;
    }
    base_rule.max(top.saturating_add(used))
}

/// `SETUP  <id> · <name>` with `+N` at the right end. The name only when
/// it fits whole; the id is cut with `…` when even it does not.
fn draw_setup_title(grid: &mut Grid, view: &SetupView, row: u16, right: usize) {
    paint_str(grid, 2, row, SETUP_TITLE, C16::White, C16::Black);
    let x = 2 + SETUP_TITLE.len() + 2;
    let mut end = right.saturating_add(1);
    if view.more > 0 {
        let more = format!("+{}", view.more);
        let at = end.saturating_sub(more.chars().count());
        paint_str(grid, col_u16(at), row, &more, C16::BrightYellow, C16::Black);
        end = at.saturating_sub(2);
    }
    let room = end.saturating_sub(x);
    let id = cut_text(&view.id, room);
    paint_detail_fg(grid, x, usize::from(row), &id, room, C16::BrightWhite);
    let name = &view.name;
    if name.is_empty() || *name == view.id {
        return;
    }
    let after = x + id.chars().count();
    let text = format!("{}{name}", llama_core::detail::SEPARATOR);
    if after + text.chars().count() <= end {
        paint_detail(grid, after, usize::from(row), &text, end - after);
    }
}

/// `  label    item · item`: the items that fit whole, else the first cut.
fn draw_setup_row(grid: &mut Grid, setup: &SetupRow, row: u16, right: usize) {
    let label: String = setup.label.chars().take(8).collect();
    paint_detail(
        grid,
        usize::from(SETUP_LABEL_X),
        usize::from(row),
        &label,
        8,
    );
    let x = usize::from(SETUP_VALUE_X);
    let room = right.saturating_add(1).saturating_sub(x);
    let mut keep = setup.items.len();
    while keep > 1 && setup_items_width(&setup.items[..keep]) > room {
        keep -= 1;
    }
    let mut col = x;
    for (i, item) in setup.items[..keep].iter().enumerate() {
        if i > 0 {
            col += paint_detail(grid, col, usize::from(row), &item.sep, room - (col - x));
        }
        let left = room.saturating_sub(col - x);
        let text = cut_text(&item.text, left);
        let fg = if item.dim {
            C16::BrightBlack
        } else {
            C16::BrightWhite
        };
        col += paint_detail_fg(grid, col, usize::from(row), &text, left, fg);
    }
}

/// `text`, or its first `width - 1` characters and `…`. No sanitising:
/// [`paint_detail_fg`] draws only printable ASCII, `·` and `…`.
fn cut_text(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_owned();
    }
    match width {
        0 => String::new(),
        n => text
            .chars()
            .take(n - 1)
            .chain(std::iter::once('\u{2026}'))
            .collect(),
    }
}

fn setup_items_width(items: &[SetupItem]) -> usize {
    items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let sep = if i > 0 { item.sep.chars().count() } else { 0 };
            sep + item.text.chars().count()
        })
        .sum()
}

fn draw_meters(grid: &mut Grid, model: &TtyModel, g: &Geom) {
    // The bool is "a level": a spectrum along the bar. VRAM and MEM are
    // capacity and keep one band colour for the whole bar.
    let meters = [
        ("CPU", cpu_value(model), model.cpu_pct, true),
        ("GPU", pct_value(model.gpu_pct), model.gpu_pct, true),
        (
            "VRAM",
            gb_value(model.vram_used_gb, model.vram_total_gb),
            percent_of(model.vram_used_gb, model.vram_total_gb),
            false,
        ),
        (
            "MEM",
            gb_value(model.mem_used_gb, model.mem_total_gb),
            percent_of(model.mem_used_gb, model.mem_total_gb),
            false,
        ),
        (
            "POWER",
            power_value(model),
            percent_of(model.power_w, model.power_limit_w),
            true,
        ),
        ("LOAD", pct_value(model.load_pct), model.load_pct, true),
    ];
    let starting = model.state == WatchState::Starting;
    for (i, (label, value, percent, level)) in meters.into_iter().enumerate() {
        let row = 3 + i as u16 * g.pitch;
        paint_str(grid, 2, row, label, C16::White, C16::Black);
        let shown = if starting { "--".to_string() } else { value };
        let value_fg = if starting || shown == "--" {
            C16::BrightBlack
        } else {
            C16::BrightWhite
        };
        right_align(grid, 24, row, &shown, value_fg, C16::Black);
        let pct = if starting {
            None
        } else {
            percent.map(f64::round)
        };
        let frac = pct.and_then(|p| (p > 0.0).then_some(p / 100.0));
        let ink = if level {
            level_ink(pct)
        } else {
            let (top, bot) = band(pct.unwrap_or(0.0));
            BarInk::Fixed(top, bot)
        };
        draw_h_bar(
            grid,
            HBar {
                x: g.left_bar,
                row,
                width: g.left_bar_w,
                frac,
                ink,
                rows: g.bar_rows,
                cells: g.meter_cells(model.chart_glyphs),
            },
        );
    }
    draw_one_meter(
        grid,
        g,
        model,
        "ACTIVITY",
        activity_value(model),
        model.activity_pct,
        g.activity_row(),
    );
}

fn draw_one_meter(
    grid: &mut Grid,
    g: &Geom,
    model: &TtyModel,
    label: &str,
    value: String,
    percent: Option<f64>,
    row: u16,
) {
    if row + 1 >= grid.rows() {
        return;
    }
    let starting = model.state == WatchState::Starting;
    let cells = g.meter_cells(model.chart_glyphs);
    paint_str(grid, 2, row, label, C16::White, C16::Black);
    let shown = if starting { "--".to_string() } else { value };
    let value_fg = if starting || shown == "--" {
        C16::BrightBlack
    } else {
        C16::BrightWhite
    };
    right_align(grid, 24, row, &shown, value_fg, C16::Black);
    let pct = if starting {
        None
    } else {
        percent.map(f64::round)
    };
    // Activity reads up to 125: over 100 the bar is full and its top step
    // turns white-hot.
    let frac = pct.and_then(|p| (p > 0.0).then_some((p / 100.0).min(1.0)));
    draw_h_bar(
        grid,
        HBar {
            x: g.left_bar,
            row,
            width: g.left_bar_w,
            frac,
            ink: level_ink(pct),
            rows: g.bar_rows,
            cells,
        },
    );
}

fn cpu_value(model: &TtyModel) -> String {
    match (model.cpu_pct, model.cpu_cores) {
        (Some(pct), Some(cores)) => format!("{} %  {cores}c", pct_num(pct)),
        (Some(pct), None) => format!("{} %", pct_num(pct)),
        _ => "--".to_string(),
    }
}

fn pct_value(pct: Option<f64>) -> String {
    match pct {
        Some(pct) => format!("{} %", pct_num(pct)),
        None => "--".to_string(),
    }
}

fn pct_num(pct: f64) -> String {
    (pct.round() as i64).to_string()
}

fn gb_value(used: Option<f64>, total: Option<f64>) -> String {
    match (used, total) {
        (Some(u), Some(t)) => format!("{u:.1}/{t:.1} GB"),
        _ => "--".to_string(),
    }
}

fn power_value(model: &TtyModel) -> String {
    match (model.power_w, model.power_limit_w) {
        (Some(w), Some(limit)) => format!("{}/{} W", w.round() as i64, limit.round() as i64),
        _ => "--".to_string(),
    }
}

/// `78 % gpu 340 W`. Watts are dropped when they would touch the label.
fn activity_value(model: &TtyModel) -> String {
    let Some(pct) = model.activity_pct else {
        return "--".to_string();
    };
    let mut head = format!("{} %", pct_num(pct));
    if let Some(src) = model.activity_src {
        head = format!("{head} {}", src.label());
    }
    let Some(watts) = model.activity_w.filter(|watts| watts.is_finite()) else {
        return head;
    };
    let full = format!("{head} {} W", watts.round() as i64);
    if full.chars().count() <= METER_VALUE_CHARS {
        full
    } else {
        head
    }
}

/// Meter values end at column 24 and must leave a blank after "ACTIVITY".
const METER_VALUE_CHARS: usize = 14;

/// Used/total as a percent. `None` when either side is missing.
fn percent_of(used: Option<f64>, total: Option<f64>) -> Option<f64> {
    match (used, total) {
        (Some(used), Some(total)) if total > 0.0 => Some(used / total * 100.0),
        _ => None,
    }
}

/// Spectrum ink for a level meter. Past 100 % the top step is white-hot.
fn level_ink(pct: Option<f64>) -> BarInk {
    BarInk::Spectrum {
        hot: pct.is_some_and(|p| p > 100.0),
    }
}

/// LCD ring bands on a percent: quiet, light, busy, flat-out, then red
/// over 100. Only the capacity meters (VRAM, MEM) still use them.
fn band(percent: f64) -> (C16, C16) {
    if percent > 100.0 {
        (C16::BrightRed, C16::Red)
    } else if percent < 15.0 {
        (C16::BrightBlue, C16::Blue)
    } else if percent < 40.0 {
        (C16::BrightGreen, C16::Green)
    } else if percent < 70.0 {
        (C16::BrightYellow, C16::Yellow)
    } else {
        (C16::BrightWhite, C16::White)
    }
}

/// The console's five steps of the shared activity ramp (L1..L5), low to
/// high: indigo, blue, magenta, pink, red.
const STEPS: [C16; 5] = [
    C16::Blue,
    C16::BrightBlue,
    C16::Magenta,
    C16::BrightMagenta,
    C16::BrightRed,
];

/// How the lit cells of a bar are coloured. The empty track is always 90.
#[derive(Clone, Copy)]
enum BarInk {
    /// One top/bottom pair for every lit cell: progress, capacity, alarms.
    Fixed(C16, C16),
    /// Each lit cell takes the step of its own position (its fifth of full
    /// scale), so a bar starts blue and a full bar shows every step. `hot` is
    /// a value past full scale: the top step turns white-hot (97 on 91).
    Spectrum { hot: bool },
}

struct HBar {
    x: u16,
    row: u16,
    width: u16,
    frac: Option<f64>,
    ink: BarInk,
    rows: u16,
    /// `Some(glyph)`: whole cells of `glyph` (#52's single-spaced meters),
    /// no half-cell ends. `None`: half-cell resolution with full blocks.
    cells: Option<char>,
}

fn draw_h_bar(grid: &mut Grid, bar: HBar) {
    let width = usize::from(bar.width);
    let lit = match bar.cells {
        Some(_) => lit_cells(bar.frac, width),
        None => lit_halves(bar.frac, width),
    };
    for i in 0..width {
        let ch = match bar.cells {
            Some(glyph) if i < lit => glyph,
            Some(_) => '░',
            None => bar_glyph(i, lit),
        };
        let (fg_top, fg_bot) = match bar.ink {
            _ if ch == '░' => (C16::BrightBlack, C16::BrightBlack),
            BarInk::Fixed(top, bot) => (top, bot),
            BarInk::Spectrum { hot } => spectrum_colours(i, width, hot),
        };
        let bot_ch = match ch {
            '█' => '▓',
            other => other,
        };
        let col = bar.x + i as u16;
        paint(grid, col, bar.row, ch, fg_top, C16::Black);
        if bar.rows > 1 {
            paint(grid, col, bar.row + 1, bot_ch, fg_bot, C16::Black);
        }
    }
}

/// Top and bottom colour of spectrum cell `cell` in a bar `width` cells long.
fn spectrum_colours(cell: usize, width: usize, hot: bool) -> (C16, C16) {
    let step = fifth(cell, width);
    if hot && step == STEPS.len() - 1 {
        (C16::BrightWhite, C16::BrightRed)
    } else {
        (STEPS[step], STEPS[step])
    }
}

fn fifth(cell: usize, width: usize) -> usize {
    if width == 0 {
        return 0;
    }
    let edge = |step: usize| ((step as f64) * width as f64 / 5.0).round() as usize;
    (0..5).find(|step| cell < edge(step + 1)).unwrap_or(4)
}

/// Lit half cells `n = clamp(round(frac × 2W), 2, 2W)`; 0 for no fill.
fn lit_halves(frac: Option<f64>, width: usize) -> usize {
    match frac {
        Some(frac) if frac > 0.0 && width > 0 => {
            let tw = width * 2;
            ((frac * tw as f64).round() as usize).clamp(2, tw)
        }
        _ => 0,
    }
}

/// Lit whole cells `clamp(round(frac × W), 1, W)`; 0 for no fill.
fn lit_cells(frac: Option<f64>, width: usize) -> usize {
    match frac {
        Some(frac) if frac > 0.0 && width > 0 => {
            ((frac * width as f64).round() as usize).clamp(1, width)
        }
        _ => 0,
    }
}

/// Glyph of `cell` when halves `1..lit` are lit (half 0 is the rounded start).
fn bar_glyph(cell: usize, lit: usize) -> char {
    let left = cell * 2;
    let lit_l = (1..lit).contains(&left);
    let lit_r = (1..lit).contains(&(left + 1));
    match (lit_l, lit_r) {
        (true, true) => '█',
        (false, true) => '▐',
        (true, false) => '▌',
        (false, false) => '░',
    }
}

fn draw_rates(grid: &mut Grid, model: &TtyModel, g: &Geom) {
    paint_label_unit(grid, g.right, 3, "GENERATION");
    paint_label_unit(grid, g.prompt_x(), 3, "PROMPT");
    if let Some(last) = model.prompt_last {
        let text = format!("last {last}");
        right_align(
            grid,
            g.prompt_x().saturating_add(g.half).saturating_sub(1),
            3,
            &text,
            C16::BrightBlack,
            C16::Black,
        );
    }
    let digit_row = if g.tall { 5 } else { 4 };
    let ph = if g.tall { 2u16 } else { 1 };
    let base = if g.tall { 4u16 } else { 2 };
    draw_digits(
        grid,
        g.right,
        digit_row,
        model.gen_tps,
        model.gen_ceiling,
        fit_pixel(base, digit_len(model.gen_tps), g.half),
        ph,
    );
    draw_digits(
        grid,
        g.prompt_x(),
        digit_row,
        model.prompt_tps,
        model.prompt_ceiling,
        fit_pixel(base, digit_len(model.prompt_tps), g.half),
        ph,
    );
    let bar_row = digit_row + if g.tall { 10 } else { 5 } + 1;
    let gen_frac = rate_frac(model.gen_tps, model.gen_ceiling);
    let prompt_frac = rate_frac(model.prompt_tps, model.prompt_ceiling);
    draw_rate_bar(grid, g.right, bar_row, g.half, gen_frac, g.bar_rows);
    draw_rate_bar(grid, g.prompt_x(), bar_row, g.half, prompt_frac, g.bar_rows);
    let tick_row = bar_row + g.bar_rows;
    draw_ticks(grid, g.right, tick_row, g.half, model.gen_ceiling);
    draw_ticks(grid, g.prompt_x(), tick_row, g.half, model.prompt_ceiling);
}

fn paint_label_unit(grid: &mut Grid, x: u16, row: u16, name: &str) {
    paint_str(grid, x, row, name, C16::White, C16::Black);
    let unit_at = col_u16(
        usize::from(x)
            .saturating_add(name.chars().count())
            .saturating_add(1),
    );
    paint_str(grid, unit_at, row, "tok/s", C16::BrightBlack, C16::Black);
}

/// Largest pixel width in `start, start/2, …, 1` whose digits fit in `avail`.
fn fit_pixel(start: u16, digits: usize, avail: u16) -> u16 {
    if digits == 0 || avail == 0 {
        return 0;
    }
    let mut pw = start.max(1);
    loop {
        let need = u32::from(pw)
            .saturating_mul(4)
            .saturating_mul(u32::try_from(digits).unwrap_or(u32::MAX));
        if need <= u32::from(avail) {
            return pw;
        }
        if pw == 1 {
            return 0;
        }
        pw /= 2;
    }
}

fn digit_len(value: Option<f64>) -> usize {
    match value {
        None => 2,
        Some(v) if v <= 0.0 => 1,
        Some(v) => (v.round() as i64).to_string().chars().count(),
    }
}

fn draw_rate_bar(grid: &mut Grid, x: u16, row: u16, width: u16, frac: Option<f64>, rows: u16) {
    let hot = frac.is_some_and(|frac| frac >= 1.0);
    draw_h_bar(
        grid,
        HBar {
            x,
            row,
            width,
            frac,
            ink: BarInk::Spectrum { hot },
            rows,
            cells: None,
        },
    );
}

fn rate_frac(value: Option<f64>, ceiling: f64) -> Option<f64> {
    let v = value?;
    if v <= 0.0 || ceiling <= 0.0 {
        return Some(0.0);
    }
    let frac = (v / ceiling).log2() + 5.0;
    Some((frac / 5.0).clamp(0.0, 1.0))
}

fn draw_ticks(grid: &mut Grid, x: u16, row: u16, width: u16, ceiling: f64) {
    let edges = [32.0, 16.0, 8.0, 4.0, 2.0, 1.0];
    let w = width as usize;
    let last = w.saturating_sub(1);
    for (i, div) in edges.into_iter().enumerate() {
        let label = tick_label(ceiling / div);
        let len = label.chars().count();
        let center = (i as f64 * last as f64 / 5.0).round() as usize;
        let mut start = center as isize - (len / 2) as isize;
        if start < 0 {
            start = 0;
        }
        if start as usize + len > w {
            start = w as isize - len as isize;
        }
        if start < 0 {
            start = 0;
        }
        paint_str(
            grid,
            x + start as u16,
            row,
            &label,
            C16::BrightBlack,
            C16::Black,
        );
    }
}

fn tick_label(v: f64) -> String {
    if v < 10.0 {
        format!("{:.1}", (v * 10.0).round() / 10.0)
    } else {
        format!("{}", v.round() as i64)
    }
}

const DIGITS: [&[&str]; 11] = [
    &["###", "#.#", "#.#", "#.#", "###"],
    &["..#", "..#", "..#", "..#", "..#"],
    &["###", "..#", "###", "#..", "###"],
    &["###", "..#", "###", "..#", "###"],
    &["#.#", "#.#", "###", "..#", "..#"],
    &["###", "#..", "###", "..#", "###"],
    &["###", "#..", "###", "#.#", "###"],
    &["###", "..#", "..#", "..#", "..#"],
    &["###", "#.#", "###", "#.#", "###"],
    &["###", "#.#", "###", "..#", "###"],
    &["...", "...", "###", "...", "..."],
];

fn draw_digits(
    grid: &mut Grid,
    x: u16,
    row: u16,
    value: Option<f64>,
    ceiling: f64,
    pw: u16,
    ph: u16,
) {
    if pw == 0 {
        return;
    }
    let (text, top, body) = match value {
        None => ("--".to_string(), C16::BrightBlack, C16::BrightBlack),
        Some(v) if v <= 0.0 => ("0".to_string(), C16::BrightBlack, C16::BrightBlack),
        Some(v) => {
            let step = rate_step(v, ceiling);
            let (top, body) = digit_colours(step);
            ((v.round() as i64).to_string(), top, body)
        }
    };
    let mut cx = x;
    for ch in text.chars() {
        let glyph = match ch {
            '0'..='9' => DIGITS[(ch as u8 - b'0') as usize],
            '-' => DIGITS[10],
            _ => continue,
        };
        for (py, bits) in glyph.iter().enumerate() {
            // The top two pixel rows take the step's cap colour. Below the
            // ceiling that colour is the same as the body.
            let colour = if py < 2 { top } else { body };
            for (px, bit) in bits.chars().enumerate() {
                if bit != '#' {
                    continue;
                }
                for dy in 0..ph {
                    for dx in 0..pw {
                        paint(
                            grid,
                            cx + px as u16 * pw + dx,
                            row + py as u16 * ph + dy,
                            '█',
                            colour,
                            C16::Black,
                        );
                    }
                }
            }
        }
        cx += 3 * pw + pw;
    }
}

fn digit_colours(step: usize) -> (C16, C16) {
    if step >= 5 {
        (C16::BrightWhite, C16::BrightRed)
    } else {
        let colour = STEPS[step];
        (colour, colour)
    }
}

fn rate_step(v: f64, ceiling: f64) -> usize {
    let edges = [
        ceiling / 32.0,
        ceiling / 16.0,
        ceiling / 8.0,
        ceiling / 4.0,
        ceiling / 2.0,
        ceiling,
    ];
    edges
        .iter()
        .enumerate()
        .rev()
        .find(|(_, e)| v >= **e)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn draw_slots(grid: &mut Grid, model: &TtyModel, g: &Geom) {
    paint_str(grid, g.right, g.slot_label, "SLOTS", C16::White, C16::Black);
    paint_str(
        grid,
        g.right + 7,
        g.slot_label,
        "prompt processed",
        C16::BrightBlack,
        C16::Black,
    );
    paint_str(
        grid,
        g.prompt_x(),
        g.slot_label,
        "decoded",
        C16::BrightBlack,
        C16::Black,
    );
    let row = g.slot_label + 1;
    match model.state {
        WatchState::AiDown | WatchState::NoLlama => {
            paint_str(grid, g.right, row, "--", C16::BrightBlack, C16::Black);
            return;
        }
        WatchState::Starting => {
            paint_str(grid, g.right, row, "...", C16::BrightBlack, C16::Black);
            return;
        }
        WatchState::Generating | WatchState::Ready => {}
    }
    let spark = spark_plan(model, g);
    if let Some(plan) = &spark
        && !model.slots.is_empty()
    {
        draw_spark_header(grid, model, g, plan);
        // The legend only when there is a marker to explain.
        if model
            .slots
            .iter()
            .any(|slot| slot.ctx_history.iter().any(|point| point.reset))
        {
            draw_reset_legend(grid, g, plan);
        }
    }
    // The count sits in a 13-column field ending 3 columns before `decoded`,
    // and the bar keeps a 7-column gap in front of that field.
    let field_end = g.prompt_x().saturating_sub(4);
    let bar_x = g.right + 7;
    let bar_end = field_end.saturating_sub(13 + 7);
    let bar_w = bar_end.saturating_sub(bar_x).saturating_add(1);
    for (i, slot) in model.slots.iter().enumerate() {
        let Ok(offset) = u16::try_from(i) else {
            break;
        };
        let row = g.slot_label.saturating_add(1).saturating_add(offset);
        paint_str(
            grid,
            g.right,
            row,
            &format!("s{}", slot.id),
            C16::BrightWhite,
            C16::Black,
        );
        let (state, state_fg) = if slot.generating {
            ("gen", C16::BrightRed)
        } else {
            ("idle", C16::BrightBlack)
        };
        paint_str(grid, g.right + 3, row, state, state_fg, C16::Black);
        let frac = (slot.total > 0).then_some(slot.done as f64 / slot.total as f64);
        let fill = if slot.generating {
            C16::White
        } else {
            C16::BrightBlack
        };
        draw_h_bar(
            grid,
            HBar {
                x: bar_x,
                row,
                width: bar_w,
                frac,
                ink: BarInk::Fixed(fill, fill),
                rows: 1,
                cells: None,
            },
        );
        let (count, decoded, fg) = if slot.total == 0 && !slot.generating {
            ("idle".to_string(), "--".to_string(), C16::BrightBlack)
        } else {
            (
                comma_pair(slot.done, slot.total),
                format!("{} tok", commas(slot.decoded)),
                C16::BrightWhite,
            )
        };
        right_align(grid, field_end, row, &count, fg, C16::Black);
        paint_str(grid, g.prompt_x(), row, &decoded, fg, C16::Black);
        let decoded_w = u16::try_from(decoded.chars().count()).unwrap_or(u16::MAX);
        let after_decoded = g.prompt_x().saturating_add(decoded_w);
        match &spark {
            Some(plan) => paint_slot_spark(grid, row, after_decoded, plan, slot, model),
            None => paint_slot_ctx(grid, row, after_decoded, g.cols, slot),
        }
    }
    let first = u16::try_from(model.slots.len()).unwrap_or(u16::MAX);
    let cap = usize::from(g.cols.saturating_sub(2).saturating_sub(g.right));
    for (i, line) in model.backend_lines.iter().enumerate() {
        let Ok(offset) = u16::try_from(i) else {
            break;
        };
        let row = g
            .slot_label
            .saturating_add(1)
            .saturating_add(first)
            .saturating_add(offset);
        paint_detail(grid, usize::from(g.right), usize::from(row), line, cap);
    }
}

/// Columns `decoded` keeps for `999,999 tok`, and the gap after it.
const DECODED_FIELD: u16 = 11 + 2;
/// Width of the per-slot ctx fill meter.
const CTX_METER_W: u16 = 12;
/// Narrowest sparkline worth drawing. Narrower hides it.
const SPARK_MIN: u16 = 12;

/// Where the fixed ctx block and the sparkline sit on every slot row (T53).
struct SparkPlan {
    /// `ctx` label column.
    ctx_x: u16,
    /// Width of the right-aligned `91k/262k` field.
    value_w: u16,
    /// First sparkline column: the newest bucket.
    spark_x: u16,
    /// Last column the sparkline may use.
    end: u16,
    /// Columns in the sparkline.
    width: u16,
}

/// `None` when the row has no room for a [`SPARK_MIN`] sparkline; the ctx
/// block then keeps its right-aligned T34 place.
fn spark_plan(model: &TtyModel, g: &Geom) -> Option<SparkPlan> {
    let ctx_x = g.prompt_x().saturating_add(DECODED_FIELD);
    let value_w = model
        .slots
        .iter()
        .map(|slot| ctx_value(slot).chars().count())
        .max()
        .unwrap_or(0)
        .max(9);
    let value_w = u16::try_from(value_w).ok()?;
    // `ctx`, gap, meter, gap, value, two blanks.
    let spark_x = ctx_x
        .saturating_add(3 + 1 + CTX_METER_W + 1)
        .saturating_add(value_w)
        .saturating_add(2);
    let end = g.cols.saturating_sub(2);
    let width = end.checked_sub(spark_x)?.saturating_add(1);
    (width >= SPARK_MIN).then_some(SparkPlan {
        ctx_x,
        value_w,
        spark_x,
        end,
        width,
    })
}

/// `91k/262k`, or `--` when either input is missing.
fn ctx_value(slot: &Slot) -> String {
    match slot_ctx(slot) {
        Some(ctx) => format!("{}/{}", compact_k(ctx.used), compact_k(ctx.n_ctx)),
        None => "--".to_string(),
    }
}

/// `ctx · 6h` in dim text over the sparklines. Painted cell by cell: the
/// sanitiser would turn `·` into `?`.
fn draw_spark_header(grid: &mut Grid, model: &TtyModel, g: &Geom, plan: &SparkPlan) {
    let text = format!("ctx \u{b7} {}h", model.ctx_history_h);
    for (i, ch) in text.chars().take(usize::from(plan.width)).enumerate() {
        let Ok(offset) = u16::try_from(i) else {
            break;
        };
        paint(
            grid,
            plan.spark_x.saturating_add(offset),
            g.slot_label,
            ch,
            C16::BrightBlack,
            C16::Black,
        );
    }
}

/// The reset marker legend (#9), longest first: each is the reasons'
/// marker letters in order (compacted, new, evicted, other drop) and a
/// word each.
const RESET_LEGENDS: [[&str; 4]; 2] = [
    ["compacted", "new", "evicted", "drop"],
    ["compact", "new", "evict", "drop"],
];

/// The legend on the SLOTS label row, right-aligned to end two columns
/// before the sparkline, over the ctx meters: the longest that fits between
/// `decoded` and there, or none.
fn draw_reset_legend(grid: &mut Grid, g: &Geom, plan: &SparkPlan) {
    let reasons = [
        Some(ResetReason::Compacted),
        Some(ResetReason::New),
        Some(ResetReason::Evicted),
        None,
    ];
    let start = g.prompt_x().saturating_add(7 + 2);
    let end = plan.spark_x.saturating_sub(2);
    let room = usize::from(end.saturating_sub(start)) + 1;
    let Some(words) = RESET_LEGENDS.iter().find(|words| {
        words.iter().map(|word| word.len() + 2).sum::<usize>() + words.len() - 1 <= room
    }) else {
        return;
    };
    let mut cells: Vec<(char, C16)> = Vec::new();
    for (reason, word) in reasons.into_iter().zip(words) {
        if !cells.is_empty() {
            cells.push((' ', C16::BrightBlack));
        }
        let (mark, fg) = reset_mark(reason);
        cells.push((mark, fg));
        cells.extend(format!(" {word}").chars().map(|ch| (ch, C16::BrightBlack)));
    }
    let width = u16::try_from(cells.len()).unwrap_or(u16::MAX);
    let x = end.saturating_add(1).saturating_sub(width);
    for (i, (ch, fg)) in cells.into_iter().enumerate() {
        let Ok(offset) = u16::try_from(i) else {
            break;
        };
        paint(
            grid,
            x.saturating_add(offset),
            g.slot_label,
            ch,
            fg,
            C16::Black,
        );
    }
}

/// A reset marker's glyph and colour by reason (#9). A drop with no reason
/// yet, an unknown one and a model swap keep T53's red `v`.
fn reset_mark(reason: Option<ResetReason>) -> (char, C16) {
    match reason {
        Some(ResetReason::Compacted) => ('c', C16::BrightGreen),
        Some(ResetReason::New) => ('n', C16::BrightCyan),
        Some(ResetReason::Evicted) => ('e', C16::BrightYellow),
        Some(ResetReason::Unknown) | None => ('v', C16::BrightRed),
    }
}

/// The ctx block at its fixed column, then this slot's sparkline, newest on
/// the left. A row whose decoded count overran its field shifts right and
/// loses its oldest columns; every column keeps the same age on every row.
fn paint_slot_spark(
    grid: &mut Grid,
    row: u16,
    after_decoded: u16,
    plan: &SparkPlan,
    slot: &Slot,
    model: &TtyModel,
) {
    let shift = after_decoded.saturating_add(1).saturating_sub(plan.ctx_x);
    let x = plan.ctx_x.saturating_add(shift);
    paint_str(grid, x, row, "ctx", C16::BrightBlack, C16::Black);
    let known = slot_ctx(slot);
    let frac = known.and_then(|ctx| ctx_fill_frac(ctx.used, ctx.n_ctx));
    if let Some(frac) = frac {
        draw_h_bar(
            grid,
            HBar {
                x: x.saturating_add(4),
                row,
                width: CTX_METER_W,
                frac: Some(frac),
                ink: BarInk::Spectrum { hot: false },
                rows: 1,
                cells: None,
            },
        );
    }
    let value_end = x
        .saturating_add(4 + CTX_METER_W + 1)
        .saturating_add(plan.value_w)
        .saturating_sub(1);
    let value_fg = if known.is_some() {
        C16::BrightWhite
    } else {
        C16::BrightBlack
    };
    if frac.is_some() {
        right_align(grid, value_end, row, &ctx_value(slot), value_fg, C16::Black);
    } else {
        // No meter: the value takes its place, `ctx --` as in T34.
        paint_fit(
            grid,
            x.saturating_add(4),
            row,
            &ctx_value(slot),
            value_fg,
            C16::Black,
            usize::from(value_end.saturating_sub(x).saturating_sub(3)),
        );
    }

    // The last decided reason, between the value and the sparkline (#9).
    if let Some(reason) = ctx_history::last_reason(&slot.ctx_history) {
        let (ch, fg) = reset_mark(Some(reason));
        paint(grid, value_end.saturating_add(2), row, ch, fg, C16::Black);
    }

    let span_ms = u64::from(model.ctx_history_h.max(1)) * 3_600_000;
    let columns = ctx_history::columns(&slot.ctx_history, usize::from(plan.width), span_ms);
    let n_ctx = slot.n_ctx.filter(|n| *n > 0);
    let start = plan.spark_x.saturating_add(shift);
    for (i, point) in columns.iter().enumerate() {
        let Ok(offset) = u16::try_from(i) else {
            break;
        };
        let col = start.saturating_add(offset);
        if col > plan.end {
            break;
        }
        if point.reset {
            let (ch, fg) = reset_mark(point.reason());
            paint(grid, col, row, ch, fg, C16::Black);
            continue;
        }
        let (Some(used), Some(n_ctx)) = (point.max, n_ctx) else {
            continue;
        };
        let frac = (used as f64 / n_ctx as f64).clamp(0.0, 1.0);
        if let Some(ch) = spark_glyph(frac, used > 0, model.chart_glyphs) {
            paint(grid, col, row, ch, ctx_colour(frac), C16::Black);
        }
    }
}

/// One-row sparkline cell: `▄ █` in halves, `▁`..`█` in eighths. Any used
/// context shows at least the smallest step.
fn spark_glyph(frac: f64, any: bool, glyphs: ChartGlyphs) -> Option<char> {
    let (level, top) = match glyphs {
        ChartGlyphs::Halves => (chart::halves(frac, 1), 2),
        ChartGlyphs::Eighths => (chart::eighths(frac, 1), 8),
    };
    let level = if any { level.max(1) } else { level };
    match (glyphs, level) {
        (_, 0) => None,
        (_, l) if l >= top => Some('\u{2588}'),
        (ChartGlyphs::Halves, _) => Some('\u{2584}'),
        (ChartGlyphs::Eighths, l) => char::from_u32(0x2580 + l),
    }
}

/// Prompt tokens plus decoded, when both context inputs were in the slots body.
#[derive(Clone, Copy)]
struct SlotCtx {
    used: u64,
    n_ctx: u64,
}

fn slot_ctx(slot: &Slot) -> Option<SlotCtx> {
    let prompt = slot.ctx_prompt?;
    let n_ctx = slot.n_ctx?;
    Some(SlotCtx {
        used: prompt.saturating_add(slot.decoded),
        n_ctx,
    })
}

/// Meter fraction in `0.0..=1.0`. `None` when `n_ctx` is zero so the bar is not drawn.
/// A used count above `n_ctx` clamps to `1.0`.
fn ctx_fill_frac(used: u64, n_ctx: u64) -> Option<f64> {
    if n_ctx == 0 {
        return None;
    }
    Some((used as f64 / n_ctx as f64).clamp(0.0, 1.0))
}

/// Thousands as `91k`. Smaller counts stay decimal. Truncates, so 91_816 is `91k`.
fn compact_k(n: u64) -> String {
    if n < 1000 {
        n.to_string()
    } else {
        format!("{}k", n / 1000)
    }
}

/// Step of a context fill, one colour per sparkline column. The top step is red.
fn ctx_colour(frac: f64) -> C16 {
    let step = ((frac.max(0.0) * 5.0).floor() as usize).min(STEPS.len() - 1);
    STEPS[step]
}

/// `ctx 91k/262k` and a fill meter, right of the decoded count. Missing inputs
/// draw `ctx --` and no meter. The console has no em dash, so the gap is `--`.
fn paint_slot_ctx(grid: &mut Grid, row: u16, after_decoded: u16, cols: u16, slot: &Slot) {
    let right = usize::from(cols.saturating_sub(2));
    let left_limit = usize::from(after_decoded.saturating_add(1));
    if left_limit > right {
        return;
    }
    let known = slot_ctx(slot);
    let value = match known {
        Some(ctx) => format!("{}/{}", compact_k(ctx.used), compact_k(ctx.n_ctx)),
        None => "--".to_string(),
    };
    let value_w = value.chars().count();
    if value_w == 0 || value_start_overflows(right, value_w) {
        return;
    }
    let value_start = right + 1 - value_w;
    if value_start < left_limit {
        return;
    }
    let frac = known.and_then(|ctx| ctx_fill_frac(ctx.used, ctx.n_ctx));
    let label = "ctx";
    let label_w = label.chars().count();
    let gap = 1usize;
    let room = value_start
        .saturating_sub(gap)
        .saturating_sub(left_limit.saturating_add(label_w).saturating_add(gap));
    let meter_w = match frac {
        Some(_) if room >= 4 => room.min(12),
        _ => 0,
    };
    let value_fg = if known.is_some() {
        C16::BrightWhite
    } else {
        C16::BrightBlack
    };
    paint_str(
        grid,
        col_u16(value_start),
        row,
        &value,
        value_fg,
        C16::Black,
    );
    let label_x = if let Some(frac) = frac.filter(|_| meter_w > 0) {
        let meter_x = value_start - gap - meter_w;
        draw_h_bar(
            grid,
            HBar {
                x: col_u16(meter_x),
                row,
                width: col_u16(meter_w),
                frac: Some(frac),
                ink: BarInk::Spectrum { hot: false },
                rows: 1,
                cells: None,
            },
        );
        meter_x.saturating_sub(gap + label_w)
    } else {
        value_start.saturating_sub(gap + label_w)
    };
    if label_x >= left_limit {
        paint_str(
            grid,
            col_u16(label_x),
            row,
            label,
            C16::BrightBlack,
            C16::Black,
        );
    }
}

fn value_start_overflows(right: usize, value_w: usize) -> bool {
    value_w > right + 1
}

/// `MM-DD HH:MM:SS`. The floor so a timestamp is never clipped.
const TIME_SHORT: usize = 14;
/// `YYYY-MM-DD HH:MM:SS`. Extra columns widen the time up to this before the model.
const TIME_FULL: usize = 19;
const MODEL_FLOOR: usize = 8;
/// Activity names are already capped at 32. Past that, width goes to the bar.
const MODEL_CAP: usize = 32;
const BAR_FLOOR: usize = 8;
/// Past this, leftover columns become gaps so the row still reaches the edge.
const BAR_CAP: usize = 160;
const SOURCE_FLOOR: usize = 15;
/// Column 0 is the live marker. Column 1 stays blank.
const REQUEST_GUTTER: usize = 2;

const LEGEND_NARROW: &str = "PROMPT = prompt processing (prefill) · GEN = token generation (decode) · CACHED = prompt tokens reused from KV cache";
const LEGEND_WIDE: &str = "PROMPT tok/s = prompt processing speed · GEN tok/s = generation speed";
/// Added to the legend while a shown row has an engine-measured rate (#35).
const LEGEND_MEASURED: &str = " · ~ = engine-measured";
/// #75: the bar's colours, painted over the `█`s.
const LEGEND_KEY: &str = " · █cached █new █out";
/// #75: a bar without a context size is scaled to the largest row.
const LEGEND_RELATIVE: &str = " · ~bar = vs largest row";

/// The RECENT legend: the column words, then what fits of the measured
/// note, the bar's colour key and the relative note, dropped in reverse
/// order of that list when the row is short. The rainbow order is the key
/// in the header row (#77).
fn paint_legend(grid: &mut Grid, row: u16, cols: u16, base: &str, measured: bool, relative: bool) {
    let room = usize::from(cols.saturating_sub(3));
    let mut items: Vec<&str> = Vec::new();
    if measured {
        items.push(LEGEND_MEASURED);
    }
    items.push(LEGEND_KEY);
    if relative {
        items.push(LEGEND_RELATIVE);
    }
    let len = |items: &[&str]| {
        base.chars().count() + items.iter().map(|i| i.chars().count()).sum::<usize>()
    };
    while !items.is_empty() && len(&items) > room {
        let drop = [LEGEND_RELATIVE, LEGEND_MEASURED, LEGEND_KEY]
            .iter()
            .find_map(|d| items.iter().position(|i| i == d));
        match drop {
            Some(at) => {
                items.remove(at);
            }
            None => break,
        }
    }
    let mut col = 2u16;
    paint_fixed(grid, col, row, base, C16::BrightBlack);
    col += u16_from(base.chars().count());
    for item in items {
        let mut key = 0usize;
        for ch in item.chars() {
            match ch {
                '█' => {
                    let fg = [C16::Blue, C16::BrightCyan, C16::BrightMagenta][key.min(2)];
                    paint(grid, col, row, '█', fg, C16::Black);
                    key += 1;
                }
                _ => paint_fixed(grid, col, row, &ch.to_string(), C16::BrightBlack),
            }
            col = col.saturating_add(1);
        }
    }
}

/// #75: the order of the remainder cell's colours, by palette slot. Under
/// the llama palette it runs around the hue wheel: violet, indigo, blue,
/// sky, green, light green, light yellow, orange. It skips the warning
/// yellow and reds and the output magenta; blue and sky are also the
/// cached and new colours, told apart by the cell's place (the first of a
/// run). The same slots in every mode (tty11, llama-view 16 / 256 /
/// truecolor, llama-cast), so the order reads the same everywhere.
pub const RAINBOW: [C16; 8] = [
    C16::Magenta,
    C16::BrightBlue,
    C16::Blue,
    C16::BrightCyan,
    C16::Green,
    C16::BrightGreen,
    C16::BrightYellow,
    C16::Cyan,
];

/// RECENT's rainbow key (#77): [`RAINBOW`] as full blocks, one cell per
/// colour, from the bar's first cell, so key cell `i` sits over bar cell
/// `i` and a remainder cell's colour reads as the start, middle or end of
/// its eighth. Full blocks in halves mode too. A bar narrower than the
/// key, or one that runs off the screen, shows what fits.
fn paint_rainbow_key(grid: &mut Grid, bar: Span, row: u16) {
    for (i, fg) in RAINBOW.iter().take(usize::from(bar.w)).enumerate() {
        paint(
            grid,
            bar.x.saturating_add(u16_from(i)),
            row,
            '\u{2588}',
            *fg,
            C16::Black,
        );
    }
}

/// Which part of a RECENT context bar a cell draws (#75).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CtxPart {
    /// Cached input.
    Cached,
    /// New input (input minus cached, never below 0).
    New,
    /// In flight: prompt not yet computed, the target the new part fills.
    Pending,
    /// Output tokens.
    Out,
    /// Empty: the dim baseline.
    Track,
}

/// One cell of a context bar.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CtxCell {
    /// What it shows.
    pub part: CtxPart,
    /// Height in eighths, 1..=8 (8 is a full cell).
    pub eighths: u8,
    /// A segment's remainder cell: its index in [`RAINBOW`].
    pub shade: Option<u8>,
}

/// The cells of a context bar `width` cells wide where the full width is
/// `scale` tokens (#75). Each segment is `value / scale × width` cells,
/// in order. A segment with a fractional part (or under one cell) starts
/// with its remainder cell: `e = ⌊frac × 8⌋` eighths high (at least one)
/// and coloured `RAINBOW[⌊(frac × 8 − e) × 8⌋]`; then its whole cells.
/// Exact integer maths: the same counts always give the same cells.
/// Pending draws its cells (rounded up) as a low track. The rest is
/// [`CtxPart::Track`], and nothing passes `width`.
#[must_use]
pub fn ctx_cells(parts: &[(CtxPart, u64)], scale: u64, width: usize) -> Vec<CtxCell> {
    const K: u128 = RAINBOW.len() as u128;
    let scale = u128::from(scale.max(1));
    let mut cells: Vec<CtxCell> = Vec::with_capacity(width);
    for (part, value) in parts {
        if *value == 0 {
            continue;
        }
        // Sub-steps: one cell is 8 eighths of K colours each.
        let units = u128::from(*value) * width as u128 * 8 * K / scale;
        let whole = usize::try_from(units / (8 * K)).unwrap_or(usize::MAX);
        let rem = units % (8 * K);
        if *part == CtxPart::Pending {
            let n = whole.saturating_add(usize::from(rem > 0));
            cells.extend(std::iter::repeat_n(
                CtxCell {
                    part: *part,
                    eighths: 1,
                    shade: None,
                },
                n.min(width),
            ));
        } else {
            if rem > 0 || whole == 0 {
                let e = u8::try_from(rem / K).unwrap_or(0).max(1);
                let s = u8::try_from(rem % K).unwrap_or(0);
                cells.push(CtxCell {
                    part: *part,
                    eighths: e,
                    shade: Some(s),
                });
            }
            cells.extend(std::iter::repeat_n(
                CtxCell {
                    part: *part,
                    eighths: 8,
                    shade: None,
                },
                whole.min(width),
            ));
        }
        if cells.len() >= width {
            break;
        }
    }
    cells.truncate(width);
    cells.resize(
        width,
        CtxCell {
            part: CtxPart::Track,
            eighths: 1,
            shade: None,
        },
    );
    cells
}

/// A row's bar segments: cached, new (computed so far, in flight), the
/// prompt still to compute (in flight), output.
fn ctx_parts(req: &Activity) -> [(CtxPart, u64); 4] {
    let cached = req.cached_tok.min(req.input_tok);
    let new = req.input_tok - cached;
    let (done, pending) = match req.inflight {
        Some(flight) if !flight.decoding && !flight.open => {
            let done = flight.processed.min(new);
            (done, new - done)
        }
        _ => (new, 0),
    };
    [
        (CtxPart::Cached, cached),
        (CtxPart::New, done),
        (CtxPart::Pending, pending),
        (CtxPart::Out, req.output_tok),
    ]
}

/// The low line of an empty or pending cell.
fn track_glyph(glyphs: ChartGlyphs) -> char {
    match glyphs {
        ChartGlyphs::Eighths => '\u{2581}',
        ChartGlyphs::Halves => '_',
    }
}

/// `e` eighths: `▁`..`▇`, `█`; halves: `▄` up to four, else `█`.
fn eighth_glyph(eighths: u8, glyphs: ChartGlyphs) -> char {
    match glyphs {
        ChartGlyphs::Eighths if eighths < 8 => {
            char::from_u32(0x2580 + u32::from(eighths.max(1))).unwrap_or('\u{2588}')
        }
        ChartGlyphs::Halves if eighths <= 4 => '\u{2584}',
        _ => '\u{2588}',
    }
}

/// RECENT's bar (#75): the request's context, cached / new / output,
/// against the model's context size; `~` before a bar scaled to the
/// largest row instead; a yellow `!` after one at 90 % of the context.
fn draw_ctx_bar(
    grid: &mut Grid,
    req: &Activity,
    span: Span,
    row: u16,
    relative_scale: u64,
    dim: bool,
    glyphs: ChartGlyphs,
) {
    let parts = ctx_parts(req);
    let total: u64 = parts.iter().map(|(_, v)| *v).sum();
    let n_ctx = req.n_ctx.filter(|n| *n > 0);
    let scale = n_ctx.unwrap_or(relative_scale);
    let width = usize::from(span.w);
    let cells = ctx_cells(&parts, scale, width);
    for (i, cell) in cells.iter().enumerate() {
        let (ch, fg) = match cell.part {
            CtxPart::Track => (track_glyph(glyphs), C16::BrightBlack),
            CtxPart::Pending => (track_glyph(glyphs), C16::BrightCyan),
            part => {
                let base = match part {
                    CtxPart::Cached => C16::Blue,
                    CtxPart::New => C16::BrightCyan,
                    _ => C16::BrightMagenta,
                };
                let fg = cell.shade.map_or(base, |s| RAINBOW[usize::from(s)]);
                (eighth_glyph(cell.eighths, glyphs), fg)
            }
        };
        let fg = if dim { C16::BrightBlack } else { fg };
        paint(grid, span.x + u16_from(i), row, ch, fg, C16::Black);
    }
    if n_ctx.is_none() && span.x > 0 {
        paint(grid, span.x - 1, row, '~', C16::BrightBlack, C16::Black);
    }
    if let Some(n) = n_ctx
        && u128::from(total) * 10 >= u128::from(n) * 9
        && width > 0
    {
        let used = cells
            .iter()
            .rposition(|c| c.part != CtxPart::Track)
            .map_or(0, |i| i + 1);
        let at = used.min(width - 1);
        let fg = if dim { C16::BrightBlack } else { C16::Yellow };
        paint(grid, span.x + u16_from(at), row, '!', fg, C16::Black);
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Grow {
    None,
    Time,
    Model,
    Bar,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Side {
    Left,
    Right,
}

#[derive(Clone, Copy)]
struct ColSpec {
    header: &'static str,
    floor: usize,
    grow: Grow,
    side: Side,
}

#[derive(Clone, Copy)]
struct Span {
    x: u16,
    w: u16,
}

struct Plan {
    wide: bool,
    full_time: bool,
    id: Span,
    time: Span,
    source: Span,
    model: Span,
    prompt_tok: Span,
    cached: Span,
    output: Span,
    prompt_tps: Span,
    gen_tps: Span,
    /// Left of the bar so a row's numbers read together (#8).
    dur: Span,
    bar: Span,
    status: Span,
    headers: &'static [ColSpec],
}

fn request_slots(rows: u16) -> u16 {
    if rows < 60 { 4 } else { 8 }
}

/// Most RECENT rows drawn with `tty.show_text = false`. The poller keeps
/// this many activity rows in that mode.
pub const RECENT_ROWS_TEXT_OFF: u16 = 32;
/// Chart rows with text off: 6 up, the axis, 6 down.
const CHART_TEXT_OFF: u16 = 13;
/// RECENT rows kept before the text-off chart shrinks.
const RECENT_FLOOR: u16 = 4;

/// Text-off rows between the RECENT header and the health rule.
#[derive(Debug, Eq, PartialEq)]
struct TextOffRows {
    recent: u16,
    chart: u16,
    blank: u16,
}

/// Split the rows the IN/OUT panels would have used. The chart gets 13 rows
/// and RECENT the rest, up to [`RECENT_ROWS_TEXT_OFF`]; anything beyond that
/// stays blank at the bottom. Short of rows, RECENT keeps [`RECENT_FLOOR`]
/// and the chart shrinks to the largest odd height of at least
/// [`CHART_MIN`], or hides.
fn text_off_rows(req_header: u16, rows: u16, reserve: u16) -> TextOffRows {
    let health_rule = rows.saturating_sub(3).saturating_sub(reserve);
    // Request rows, the legend and the RECENT rule sit under the header.
    let body = health_rule
        .saturating_sub(req_header.saturating_add(1))
        .saturating_sub(2);
    let spare = body.saturating_sub(RECENT_FLOOR);
    let chart = if spare >= CHART_TEXT_OFF {
        CHART_TEXT_OFF
    } else if spare >= CHART_MIN {
        spare - (1 - spare % 2)
    } else {
        0
    };
    let recent = (body - chart).min(RECENT_ROWS_TEXT_OFF);
    TextOffRows {
        recent,
        chart,
        blank: body - chart - recent,
    }
}

/// `tty.show_text = false`: RECENT, the chart and blank rows. No IN/OUT.
fn draw_text_off(grid: &mut Grid, model: &TtyModel, g: &Geom, req_header: u16) {
    // FANS and TEMPS sit at the bottom of the freed rows, above the health
    // rule. RECENT keeps its floor first; without room they shrink to the
    // TEMPS line, or hide.
    let (fans, temps) = (model.fans.as_ref(), model.temps.as_ref());
    let spare = text_off_rows(req_header, g.rows, 0)
        .recent
        .saturating_sub(RECENT_FLOOR);
    let strip = strip_plan(fans, temps, spare);
    let reserve = match strip {
        Some(Strip::Block(height)) => 1 + height,
        Some(Strip::Line) => 1,
        None => 0,
    };
    let plan = text_off_rows(req_header, g.rows, reserve);
    draw_requests(grid, model, g, req_header, plan.recent);
    let req_rule = req_header + 1 + plan.recent + 1;
    draw_rule(grid, req_rule, g.cols);
    if plan.chart >= CHART_MIN {
        draw_chart(grid, model, g, req_rule + 1, plan.chart);
    }
    match strip {
        Some(Strip::Block(height)) => {
            let strip_rule = g.rows - 3 - reserve;
            draw_rule(grid, strip_rule, g.cols);
            draw_strip(grid, g, fans, temps, strip_rule + 1, height);
        }
        Some(Strip::Line) => {
            if let Some(panel) = temps {
                draw_temps_line(grid, panel, 2, g.cols - 3, g.rows - 4);
            }
        }
        None => {}
    }
    let note = match model.state {
        WatchState::NoLlama => {
            "llama-swap polling is off ([llama] enabled = false in watch.toml)".to_owned()
        }
        WatchState::AiDown => format!(
            "llama-swap unreachable since {} - retrying every 1 s",
            model.down_since
        ),
        _ => return,
    };
    // In the blank rows when there are any, else on the last RECENT row.
    let (top, height) = if plan.blank > 0 {
        (req_rule + 1 + plan.chart, plan.blank)
    } else if plan.recent > 0 {
        (req_header + plan.recent, 1)
    } else {
        return;
    };
    let mid = top + height.saturating_sub(1) / 2;
    if model.state == WatchState::AiDown {
        let band = if height >= 3 {
            mid - 1..=mid + 1
        } else {
            mid..=mid
        };
        for row in band {
            fill_span(grid, 2, g.cols - 3, row, ' ', C16::Black, C16::Yellow);
        }
        center_in(grid, 2, g.cols - 3, mid, &note, C16::Black, C16::Yellow);
    } else {
        fill_span(grid, 2, g.cols - 3, mid, ' ', C16::BrightBlack, C16::Black);
        center_in(
            grid,
            2,
            g.cols - 3,
            mid,
            &note,
            C16::BrightBlack,
            C16::Black,
        );
    }
}

fn short_specs() -> &'static [ColSpec] {
    &SHORT_SPECS
}

fn long_specs() -> &'static [ColSpec] {
    &LONG_SPECS
}

const SHORT_SPECS: [ColSpec; 12] = [
    ColSpec {
        header: "RECENT",
        floor: 6,
        grow: Grow::None,
        side: Side::Left,
    },
    ColSpec {
        header: "TIME",
        floor: TIME_SHORT,
        grow: Grow::Time,
        side: Side::Left,
    },
    ColSpec {
        header: "SOURCE",
        floor: SOURCE_FLOOR,
        grow: Grow::None,
        side: Side::Left,
    },
    ColSpec {
        header: "MODEL",
        floor: MODEL_FLOOR,
        grow: Grow::Model,
        side: Side::Left,
    },
    ColSpec {
        header: "IN",
        floor: 8,
        grow: Grow::None,
        side: Side::Right,
    },
    ColSpec {
        header: "CACHED",
        floor: 8,
        grow: Grow::None,
        side: Side::Right,
    },
    ColSpec {
        header: "OUT",
        floor: 8,
        grow: Grow::None,
        side: Side::Right,
    },
    ColSpec {
        header: "PROMPT",
        floor: 7,
        grow: Grow::None,
        side: Side::Right,
    },
    ColSpec {
        header: "GEN",
        floor: 6,
        grow: Grow::None,
        side: Side::Right,
    },
    ColSpec {
        header: "DUR",
        floor: 8,
        grow: Grow::None,
        side: Side::Right,
    },
    ColSpec {
        header: "",
        floor: BAR_FLOOR,
        grow: Grow::Bar,
        side: Side::Left,
    },
    ColSpec {
        header: "",
        floor: 5,
        grow: Grow::None,
        side: Side::Left,
    },
];

const LONG_SPECS: [ColSpec; 12] = [
    ColSpec {
        header: "RECENT",
        floor: 6,
        grow: Grow::None,
        side: Side::Left,
    },
    ColSpec {
        header: "TIME",
        floor: TIME_SHORT,
        grow: Grow::Time,
        side: Side::Left,
    },
    ColSpec {
        header: "SOURCE",
        floor: SOURCE_FLOOR,
        grow: Grow::None,
        side: Side::Left,
    },
    ColSpec {
        header: "MODEL",
        floor: MODEL_FLOOR,
        grow: Grow::Model,
        side: Side::Left,
    },
    ColSpec {
        header: "PROMPT tokens",
        floor: 8,
        grow: Grow::None,
        side: Side::Right,
    },
    ColSpec {
        header: "CACHED tokens",
        floor: 8,
        grow: Grow::None,
        side: Side::Right,
    },
    ColSpec {
        header: "OUTPUT tokens",
        floor: 8,
        grow: Grow::None,
        side: Side::Right,
    },
    ColSpec {
        header: "PROMPT tok/s",
        floor: 7,
        grow: Grow::None,
        side: Side::Right,
    },
    ColSpec {
        header: "GEN tok/s",
        floor: 6,
        grow: Grow::None,
        side: Side::Right,
    },
    ColSpec {
        header: "DURATION",
        floor: 8,
        grow: Grow::None,
        side: Side::Right,
    },
    ColSpec {
        header: "",
        floor: BAR_FLOOR,
        grow: Grow::Bar,
        side: Side::Left,
    },
    ColSpec {
        header: "",
        floor: 5,
        grow: Grow::None,
        side: Side::Left,
    },
];

fn col_floor(spec: &ColSpec) -> usize {
    spec.header.chars().count().max(spec.floor)
}

fn specs_min(specs: &[ColSpec]) -> usize {
    let columns: usize = specs.iter().map(col_floor).sum();
    let gaps = specs.len().saturating_sub(1);
    REQUEST_GUTTER + columns + gaps
}

/// Columns span every paintable cell. Extra width goes to the timestamp (up to
/// the year form), then the model (up to the display-name cap), then the
/// generation bar, then the gaps.
fn request_plan(cols: u16) -> Plan {
    let budget = usize::from(cols.saturating_sub(1));
    let wide = cols > 160 && specs_min(long_specs()) <= budget;
    let headers = if wide { long_specs() } else { short_specs() };
    let mut widths: Vec<usize> = headers.iter().map(col_floor).collect();
    let mut extra = budget.saturating_sub(specs_min(headers));
    grow(&mut widths, headers, Grow::Time, TIME_FULL, &mut extra);
    grow(&mut widths, headers, Grow::Model, MODEL_CAP, &mut extra);
    grow(&mut widths, headers, Grow::Bar, BAR_CAP, &mut extra);
    let gap_count = widths.len().saturating_sub(1);
    let mut gaps = vec![1usize; gap_count];
    if let (Some(add), Some(rem)) = (extra.checked_div(gap_count), extra.checked_rem(gap_count)) {
        for (i, gap) in gaps.iter_mut().enumerate() {
            *gap += add;
            if i >= gap_count - rem {
                *gap += 1;
            }
        }
    }
    let mut x = REQUEST_GUTTER;
    let mut spans = Vec::with_capacity(widths.len());
    for (i, width) in widths.iter().enumerate() {
        spans.push(Span {
            x: u16_from(x),
            w: u16_from(*width),
        });
        x += width;
        if let Some(gap) = gaps.get(i) {
            x += gap;
        }
    }
    debug_assert_eq!(x, budget);
    let full_time = usize::from(spans[1].w) >= TIME_FULL;
    Plan {
        wide,
        full_time,
        id: spans[0],
        time: spans[1],
        source: spans[2],
        model: spans[3],
        prompt_tok: spans[4],
        cached: spans[5],
        output: spans[6],
        prompt_tps: spans[7],
        gen_tps: spans[8],
        dur: spans[9],
        bar: spans[10],
        status: spans[11],
        headers,
    }
}

fn grow(widths: &mut [usize], specs: &[ColSpec], kind: Grow, cap: usize, extra: &mut usize) {
    let Some(index) = specs.iter().position(|spec| spec.grow == kind) else {
        return;
    };
    let add = (*extra).min(cap.saturating_sub(widths[index]));
    widths[index] += add;
    *extra -= add;
}

fn u16_from(n: usize) -> u16 {
    u16::try_from(n).unwrap_or(u16::MAX)
}

fn draw_requests(grid: &mut Grid, model: &TtyModel, g: &Geom, header: u16, slots: u16) {
    let plan = request_plan(g.cols);
    paint_request_headers(grid, header, &plan);
    paint_rainbow_key(grid, plan.bar, header);
    let legend = if plan.wide {
        LEGEND_WIDE
    } else {
        LEGEND_NARROW
    };
    let measured = model
        .requests
        .iter()
        .take(usize::from(slots))
        .any(|req| req.prompt_measured || req.gen_measured);
    let relative = model
        .requests
        .iter()
        .take(usize::from(slots))
        .any(|req| req.n_ctx.is_none_or(|n| n == 0));
    let starting = model.state == WatchState::Starting;
    paint_legend(
        grid,
        header + 1 + slots,
        g.cols,
        legend,
        measured && !starting,
        relative && !starting,
    );
    if model.state == WatchState::Starting {
        paint_str(
            grid,
            2,
            header + 1,
            "waiting for llama-swap /api/metrics/activity ...",
            C16::BrightBlack,
            C16::Black,
        );
        return;
    }
    let dim = matches!(model.state, WatchState::AiDown | WatchState::NoLlama);
    let plain = if dim { C16::BrightBlack } else { C16::White };
    let hot = if dim {
        C16::BrightBlack
    } else {
        C16::BrightWhite
    };
    // #75: in-flight rows first, at most half the rows, so finished rows
    // still show.
    let flight_cap = usize::from(slots / 2).max(1);
    let shown: Vec<&Activity> = model
        .requests
        .iter()
        .filter(|req| req.inflight.is_some())
        .take(flight_cap)
        .chain(model.requests.iter().filter(|req| req.inflight.is_none()))
        .take(usize::from(slots))
        .collect();
    let relative_scale = shown
        .iter()
        .filter(|req| req.n_ctx.is_none_or(|n| n == 0))
        .map(|req| ctx_parts(req).iter().map(|(_, v)| *v).sum::<u64>())
        .max()
        .unwrap_or(0)
        .max(1);
    for (i, req) in shown.into_iter().enumerate() {
        let Ok(offset) = u16::try_from(i) else {
            break;
        };
        let row = header + 1 + offset;
        if (req.live || req.inflight.is_some()) && !dim {
            paint(grid, 0, row, '>', C16::BrightRed, C16::Black);
        }
        // An in-flight request has no llama-swap id yet (#75).
        if req.inflight.is_none() {
            paint_span_right(grid, plan.id, row, &req.id.to_string(), plain, C16::Black);
        }
        let time = format_request_time(&req.time, plan.full_time);
        paint_span_left(grid, plan.time, row, &time, plain, C16::Black);
        paint_ellipsis(grid, plan.source, row, &format_source(&req.source), plain);
        paint_ellipsis(grid, plan.model, row, &req.model, plain);
        let open = req.inflight.is_some_and(|flight| flight.open);
        let input = if open {
            format!("{}+", commas(req.input_tok))
        } else {
            commas(req.input_tok)
        };
        paint_span_right(grid, plan.prompt_tok, row, &input, hot, C16::Black);
        paint_span_right(
            grid,
            plan.cached,
            row,
            &commas(req.cached_tok),
            plain,
            C16::Black,
        );
        paint_span_right(
            grid,
            plan.output,
            row,
            &commas(req.output_tok),
            hot,
            C16::Black,
        );
        paint_span_right(
            grid,
            plan.prompt_tps,
            row,
            &prompt_rate_text(req.prompt_tps, req.prompt_measured),
            plain,
            C16::Black,
        );
        paint_span_right(
            grid,
            plan.gen_tps,
            row,
            &gen_rate_text(req.gen_tps, req.gen_measured),
            hot,
            C16::Black,
        );
        paint_span_right(grid, plan.dur, row, &req.dur, plain, C16::Black);
        draw_ctx_bar(
            grid,
            req,
            plan.bar,
            row,
            relative_scale,
            dim,
            model.chart_glyphs,
        );
        if req.err {
            paint_span_left(grid, plan.status, row, " ERR ", C16::Black, C16::Yellow);
        } else if let Some(flight) = req.inflight.filter(|_| !dim) {
            let word = if flight.decoding { "gen" } else { "pp" };
            paint_span_left(grid, plan.status, row, word, C16::BrightRed, C16::Black);
            if let (false, Some(reason)) = (flight.decoding, flight.reset) {
                let (mark, fg) = reset_mark(Some(reason));
                paint(grid, plan.status.x + 3, row, mark, fg, C16::Black);
            }
        } else if req.live && !dim {
            paint_span_left(grid, plan.status, row, "gen", C16::BrightRed, C16::Black);
        }
    }
}

fn paint_request_headers(grid: &mut Grid, row: u16, plan: &Plan) {
    let spans = [
        plan.id,
        plan.time,
        plan.source,
        plan.model,
        plan.prompt_tok,
        plan.cached,
        plan.output,
        plan.prompt_tps,
        plan.gen_tps,
        plan.dur,
        plan.bar,
        plan.status,
    ];
    for (spec, span) in plan.headers.iter().zip(spans) {
        if spec.header.is_empty() {
            continue;
        }
        let fg = if spec.header == "RECENT" {
            C16::White
        } else {
            C16::BrightBlack
        };
        match spec.side {
            Side::Left => paint_span_left(grid, span, row, spec.header, fg, C16::Black),
            Side::Right => paint_span_right(grid, span, row, spec.header, fg, C16::Black),
        }
    }
}

/// Clock digits from an activity timestamp, to the second.
///
/// The service hands RECENT `YYYY-MM-DD HH:MM:SS` already in the host's
/// zone, like the header clock (#48); that becomes `09-25 15:57:08`, or the
/// year form when `full` is set. A timestamp the service could not convert
/// keeps its own digits, and a value that is not a timestamp is returned
/// sanitised and unchanged.
fn format_request_time(raw: &str, full: bool) -> String {
    let clean = one_line(raw);
    let Some(clock) = parse_clock(&clean) else {
        return clean;
    };
    if full {
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            clock.year, clock.month, clock.day, clock.hour, clock.min, clock.sec
        )
    } else {
        format!(
            "{:02}-{:02} {:02}:{:02}:{:02}",
            clock.month, clock.day, clock.hour, clock.min, clock.sec
        )
    }
}

struct Clock {
    year: u16,
    month: u16,
    day: u16,
    hour: u16,
    min: u16,
    sec: u16,
}

fn parse_clock(text: &str) -> Option<Clock> {
    let bytes = text.as_bytes();
    if bytes.len() < TIME_FULL {
        return None;
    }
    let year = parse_digits(&bytes[0..4])?;
    if bytes[4] != b'-' {
        return None;
    }
    let month = parse_digits(&bytes[5..7])?;
    if bytes[7] != b'-' {
        return None;
    }
    let day = parse_digits(&bytes[8..10])?;
    if bytes[10] != b'T' && bytes[10] != b' ' {
        return None;
    }
    let hour = parse_digits(&bytes[11..13])?;
    if bytes[13] != b':' {
        return None;
    }
    let min = parse_digits(&bytes[14..16])?;
    if bytes[16] != b':' {
        return None;
    }
    let sec = parse_digits(&bytes[17..19])?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || min > 59 || sec > 59 {
        return None;
    }
    if !clock_tail(&bytes[TIME_FULL..]) {
        return None;
    }
    Some(Clock {
        year,
        month,
        day,
        hour,
        min,
        sec,
    })
}

fn clock_tail(rest: &[u8]) -> bool {
    let Some(rest) = strip_fraction(rest) else {
        return false;
    };
    zone_suffix(rest)
}

fn strip_fraction(rest: &[u8]) -> Option<&[u8]> {
    if rest.first() != Some(&b'.') {
        return Some(rest);
    }
    let digits = rest[1..].iter().take_while(|b| b.is_ascii_digit()).count();
    if digits == 0 {
        return None;
    }
    Some(&rest[1 + digits..])
}

fn zone_suffix(rest: &[u8]) -> bool {
    if rest.is_empty() || rest == b"Z" || rest == b"z" {
        return true;
    }
    let Some((sign, body)) = rest.split_first() else {
        return false;
    };
    if *sign != b'+' && *sign != b'-' {
        return false;
    }
    let colon_at = body.iter().position(|b| *b == b':');
    let digits: Vec<u8> = body.iter().copied().filter(|b| *b != b':').collect();
    let colon_ok = match colon_at {
        None => true,
        Some(2) if body.iter().filter(|b| **b == b':').count() == 1 => true,
        _ => false,
    };
    colon_ok && digits.len() == 4 && digits.iter().all(u8::is_ascii_digit)
}

fn parse_digits(bytes: &[u8]) -> Option<u16> {
    if bytes.is_empty() || !bytes.iter().all(u8::is_ascii_digit) {
        return None;
    }
    std::str::from_utf8(bytes).ok()?.parse().ok()
}

fn format_source(raw: &str) -> String {
    let clean = one_line(raw);
    clean.strip_prefix("ip:").unwrap_or(&clean).to_string()
}

fn fit_ellipsis(text: &str, width: usize) -> String {
    let chars: Vec<char> = one_line(text).chars().collect();
    if chars.len() <= width {
        return chars.into_iter().collect();
    }
    match width {
        0 => String::new(),
        1 => "…".to_string(),
        n => {
            let mut out: String = chars.into_iter().take(n - 1).collect();
            out.push('…');
            out
        }
    }
}

fn paint_ellipsis(grid: &mut Grid, span: Span, row: u16, text: &str, fg: C16) {
    // `…` is our truncation mark. Sanitise the text first, then add it.
    // A second pass through the sanitiser would turn the mark into `?`.
    let fitted = fit_ellipsis(text, usize::from(span.w));
    paint_fixed(grid, span.x, row, &fitted, fg);
}

/// Paint a fixed string, including the legend dot and the truncation mark.
///
/// Those two scalars are in the console glyph set. The sanitiser maps both to
/// `?`, and this string is layout copy, not llama text.
fn paint_fixed(grid: &mut Grid, mut col: u16, row: u16, text: &str, fg: C16) {
    for ch in text.chars() {
        if ch != '·' && ch != '…' && !(ch.is_ascii() && !ch.is_ascii_control()) {
            continue;
        }
        paint(grid, col, row, ch, fg, C16::Black);
        col = col.saturating_add(1);
    }
}

fn paint_span_left(grid: &mut Grid, span: Span, row: u16, text: &str, fg: C16, bg: C16) {
    let width = usize::from(span.w);
    for (i, ch) in one_line(text).chars().take(width).enumerate() {
        let Ok(dx) = u16::try_from(i) else {
            break;
        };
        paint(grid, span.x.saturating_add(dx), row, ch, fg, bg);
    }
}

fn paint_span_right(grid: &mut Grid, span: Span, row: u16, text: &str, fg: C16, bg: C16) {
    let width = usize::from(span.w);
    if width == 0 {
        return;
    }
    let chars: Vec<char> = one_line(text).chars().collect();
    let skip = chars.len().saturating_sub(width);
    let start = width.saturating_sub(chars.len());
    for (i, ch) in chars.iter().skip(skip).enumerate() {
        let Ok(dx) = u16::try_from(start + i) else {
            break;
        };
        paint(grid, span.x.saturating_add(dx), row, *ch, fg, bg);
    }
}

/// RECENT's PROMPT: `1,234`, or `~1,234` when the engine measured it
/// (#35), `~123k` from 100,000 so the mark fits the column; `--` unknown.
#[must_use]
pub fn prompt_rate_text(tps: Option<f64>, measured: bool) -> String {
    let Some(tps) = tps else {
        return "--".to_owned();
    };
    let whole = tps.round() as u64;
    if !measured {
        commas(whole)
    } else if whole < 100_000 {
        format!("~{}", commas(whole))
    } else {
        format!("~{}k", commas(whole / 1000))
    }
}

/// RECENT's GEN: `45.3`, or `~45.3` when the engine measured it (#35),
/// `~1,234` from 1,000 and `~12k` from 10,000 so the mark fits the
/// column; `--` unknown.
#[must_use]
pub fn gen_rate_text(tps: Option<f64>, measured: bool) -> String {
    let Some(tps) = tps else {
        return "--".to_owned();
    };
    if !measured {
        one_decimal(tps)
    } else if tps < 999.95 {
        format!("~{}", one_decimal(tps))
    } else if tps < 9_999.5 {
        format!("~{}", commas(tps.round() as u64))
    } else {
        format!("~{}k", (tps / 1000.0).round() as u64)
    }
}

fn one_decimal(v: f64) -> String {
    let scaled = (v * 10.0).round() / 10.0;
    format!("{scaled:.1}")
}

const CHART_PREFERRED: u16 = 9;
const CHART_MIN: u16 = 5;
const CHART_INOUT_FLOOR: u16 = 8;

fn chart_height(req_rule: u16, rows: u16) -> u16 {
    let health_rule = rows.saturating_sub(3);
    let available = health_rule.saturating_sub(req_rule.saturating_add(1));
    if available < CHART_MIN + CHART_INOUT_FLOOR {
        return 0;
    }
    let raw = if available >= CHART_PREFERRED + CHART_INOUT_FLOOR {
        CHART_PREFERRED
    } else {
        available
            .saturating_sub(CHART_INOUT_FLOOR)
            .clamp(CHART_MIN, CHART_PREFERRED)
    };
    let half = ((raw - 1) / 2).max(2);
    (half * 2 + 1).min(available)
}

fn draw_chart(grid: &mut Grid, model: &TtyModel, g: &Geom, start: u16, height: u16) {
    let half = (height - 1) / 2;
    if half == 0 {
        return;
    }
    let axis = start + half;
    let band_left = 2u16;
    let band_right = g.cols.saturating_sub(3);
    let gen_label = tick_label(model.gen_ceiling);
    let prompt_label = tick_label(model.prompt_ceiling);
    // "now", then one blank column, then the newest bucket.
    let now = "now";
    let data_left = band_left
        .saturating_add(u16::try_from(now.chars().count()).unwrap_or(u16::MAX))
        .saturating_add(1);
    let series_w = "gen".chars().count().max("prompt".chars().count());
    // Ceilings, series names and the age label share the right margin.
    let right_w = gen_label
        .chars()
        .count()
        .max(prompt_label.chars().count())
        .max(series_w);
    let fit = |reserve: usize| -> Option<usize> {
        let reserve = u16::try_from(reserve).ok()?;
        let data_right = band_right.saturating_sub(reserve);
        if data_right < data_left {
            return None;
        }
        Some(usize::from(data_right - data_left + 1))
    };
    let minutes = |width: usize| {
        let secs = width as u64 * model.chart_bucket_s.max(1);
        (secs + 30) / 60
    };
    let Some(mut width) = fit(right_w) else {
        return;
    };
    let mut mins = minutes(width);
    let age_w = format!("-{mins}m").chars().count();
    if age_w > right_w {
        let Some(next_width) = fit(age_w) else {
            return;
        };
        width = next_width;
        mins = minutes(width);
    }
    let len = model.chart.len();
    let show = len.min(width);
    for age in 0..show {
        let col = data_left.saturating_add(u16::try_from(age).unwrap_or(u16::MAX));
        draw_chart_col(
            grid,
            col,
            axis,
            half,
            model.chart[len - 1 - age],
            model.gen_ceiling,
            model.prompt_ceiling,
            model.chart_glyphs,
        );
    }
    paint_str(grid, band_left, axis, now, C16::BrightBlack, C16::Black);
    right_align(
        grid,
        band_right,
        axis,
        &format!("-{mins}m"),
        C16::BrightBlack,
        C16::Black,
    );
    right_align(
        grid,
        band_right,
        start,
        &gen_label,
        C16::BrightBlack,
        C16::Black,
    );
    right_align(
        grid,
        band_right,
        start + height - 1,
        &prompt_label,
        C16::BrightBlack,
        C16::Black,
    );
    if axis > 0 {
        right_align(grid, band_right, axis - 1, "gen", C16::White, C16::Black);
    }
    right_align(grid, band_right, axis + 1, "prompt", C16::White, C16::Black);
}

#[allow(clippy::too_many_arguments)]
fn draw_chart_col(
    grid: &mut Grid,
    col: u16,
    axis: u16,
    half: u16,
    bucket: ChartBucket,
    gen_ceiling: f64,
    prompt_ceiling: f64,
    glyphs: ChartGlyphs,
) {
    match bucket.gen_tps {
        None => {}
        Some(v) if v <= 0.0 => {
            paint(grid, col, axis, '·', C16::BrightBlack, C16::Black);
        }
        Some(v) => {
            let frac = rate_frac(Some(v), gen_ceiling).unwrap_or(0.0);
            let colour = digit_colours(rate_step(v, gen_ceiling)).1;
            let cells = match glyphs {
                ChartGlyphs::Halves => {
                    let level = chart::halves(frac, half).max(1);
                    half_inks(chart::rise_glyphs(level, half))
                }
                ChartGlyphs::Eighths => {
                    let level = chart::eighths(frac, half).max(1);
                    chart::rise_eighths(level, half)
                }
            };
            for (i, ink) in cells.into_iter().enumerate() {
                paint_ink(grid, col, axis - half + i as u16, ink, colour);
            }
        }
    }
    match bucket.prompt_tps {
        None => {}
        Some(v) if v <= 0.0 => {
            paint(grid, col, axis, '·', C16::BrightBlack, C16::Black);
        }
        Some(v) => {
            let frac = rate_frac(Some(v), prompt_ceiling).unwrap_or(0.0);
            let colour = digit_colours(rate_step(v, prompt_ceiling)).1;
            let cells = match glyphs {
                ChartGlyphs::Halves => {
                    let level = chart::halves(frac, half).max(1);
                    half_inks(chart::fall_glyphs(level, half))
                }
                ChartGlyphs::Eighths => {
                    let level = chart::eighths(frac, half).max(1);
                    chart::fall_eighths(level, half)
                }
            };
            for (i, ink) in cells.into_iter().enumerate() {
                paint_ink(grid, col, axis + 1 + i as u16, ink, colour);
            }
        }
    }
}

fn half_inks(glyphs: Vec<char>) -> Vec<Ink> {
    glyphs
        .into_iter()
        .map(|ch| if ch == ' ' { Ink::Empty } else { Ink::Fg(ch) })
        .collect()
}

fn paint_ink(grid: &mut Grid, col: u16, row: u16, ink: Ink, colour: C16) {
    match ink {
        Ink::Empty => {}
        Ink::Fg(ch) => paint(grid, col, row, ch, colour, C16::Black),
        Ink::Inverse(ch) => paint(grid, col, row, ch, C16::Black, colour),
    }
}

fn draw_text(grid: &mut Grid, model: &TtyModel, g: &Geom, req_rule: u16) {
    let in_label = req_rule + 1;
    let mut health_rule = g.rows - 3;
    if health_rule <= in_label + 4 {
        return;
    }
    // Last column of IN/OUT text. FANS and TEMPS may take the right side.
    let mut right = g.cols - 3;
    let (fans, temps) = (model.fans.as_ref(), model.temps.as_ref());
    if fans.is_some() || temps.is_some() {
        if g.cols >= FANS_SIDE_COLS {
            let split = fans_split(g.cols);
            right = split - 2;
            for row in in_label..health_rule {
                paint(grid, split, row, '|', C16::BrightBlack, C16::Black);
            }
            let (left, end) = fans_span(g);
            draw_side_column(
                grid,
                fans,
                temps,
                left,
                end,
                in_label,
                health_rule - in_label,
            );
        } else {
            // Under IN/OUT: a rule, then the blocks side by side, if IN and
            // OUT keep three rows each; else TEMPS on one line (#74).
            let spare = health_rule.saturating_sub(in_label + TEXT_FLOOR_SPAN);
            match strip_plan(fans, temps, spare) {
                Some(Strip::Block(height)) => {
                    health_rule -= 1 + height;
                    draw_rule(grid, health_rule, g.cols);
                    draw_strip(grid, g, fans, temps, health_rule + 1, height);
                }
                Some(Strip::Line) => {
                    health_rule -= 1;
                    if let Some(panel) = temps {
                        draw_temps_line(grid, panel, 2, g.cols - 3, health_rule);
                    }
                }
                None => {}
            }
        }
    }
    let span = health_rule - in_label - 1;
    let text_rows = span.saturating_sub(2);
    let in_rows = (text_rows as usize * 3 / 10).max(3).min(text_rows as usize);
    let in_rows = in_rows as u16;
    let out_label = in_label + 1 + in_rows + 1;
    let out_rows = text_rows - in_rows;
    paint_title(grid, in_label, &model.in_title);
    paint_title(grid, out_label, &model.out_title);
    let in_start = in_label + 1;
    let out_start = out_label + 1;
    place_lines(
        grid,
        in_start,
        in_rows,
        &logical_lines(&model.in_lines),
        false,
        false,
        right,
    );
    place_lines(
        grid,
        out_start,
        out_rows,
        &revealed_lines(model),
        model.state == WatchState::Generating,
        true,
        right,
    );
    if !model.text_note.is_empty() && model.in_lines.is_empty() && model.out_lines.is_empty() {
        for (start, rows) in [(in_start, in_rows), (out_start, out_rows)] {
            let mid = start.saturating_add(rows.saturating_sub(1) / 2);
            center_in(
                grid,
                2,
                right,
                mid,
                &model.text_note,
                C16::BrightBlack,
                C16::Black,
            );
        }
    }
    if model.state == WatchState::Starting {
        let note = "llama-watch starting - first reads in ~1 s; llama text appears after the first /slots poll";
        let mid = in_start
            .saturating_add(out_start)
            .saturating_add(out_rows)
            .saturating_sub(1)
            / 2;
        center_in(grid, 2, right, mid, note, C16::BrightBlack, C16::Black);
    }
    if model.state == WatchState::NoLlama {
        let note = "llama-swap polling is off ([llama] enabled = false in watch.toml)";
        let mid = in_start
            .saturating_add(out_start)
            .saturating_add(out_rows)
            .saturating_sub(1)
            / 2;
        center_in(grid, 2, right, mid, note, C16::BrightBlack, C16::Black);
    }
    if model.state == WatchState::AiDown {
        let top = out_start + out_rows / 2;
        let msg = format!(
            "llama-swap unreachable since {} - retrying every 1 s; last text kept above and below",
            model.down_since
        );
        for row in top.saturating_sub(1)..=top + 1 {
            fill_span(grid, 2, right, row, ' ', C16::Black, C16::Yellow);
        }
        center_in(grid, 2, right, top, &msg, C16::Black, C16::Yellow);
    }
}

/// From this width the FANS panel sits beside IN/OUT, not under it. 190
/// takes in 192x60, the 10x18 font on a 1920x1080 screen (#73): IN/OUT
/// keep 109 columns and FANS gets 77.
pub const FANS_SIDE_COLS: u16 = 190;
/// IN/OUT rows kept under FANS on a narrow screen: two titles and 3 + 3 text
/// rows, plus the row under the IN title.
const TEXT_FLOOR_SPAN: u16 = 9;
/// Fixed cells of one fan row besides the meter.
const FAN_ROW_FIXED: usize = MAX_FAN_LABEL + 2 + 5 + 4 + 2 + 1 + 4 + 2 + 6;
const FAN_BAR_MIN: usize = 4;
const FAN_PANEL_MAX: u16 = 100;

/// Column of the `|` between IN/OUT and FANS: 58% of the width.
fn fans_split(cols: u16) -> u16 {
    u16::try_from(u32::from(cols) * 58 / 100).unwrap_or(cols)
}

/// First and last column of the FANS panel.
fn fans_span(g: &Geom) -> (u16, u16) {
    if g.cols >= FANS_SIDE_COLS {
        (fans_split(g.cols) + 2, g.cols - 3)
    } else {
        (2, (g.cols - 3).min(2 + FAN_PANEL_MAX - 1))
    }
}

/// How FANS and TEMPS fit under IN/OUT (or RECENT, text off).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Strip {
    /// A rule, then this many rows with the blocks side by side.
    Block(u16),
    /// TEMPS as one line (the hottest of each row); FANS hidden.
    Line,
}

/// Rows FANS and TEMPS can have: `spare` counts the rule. The full blocks;
/// else FANS whole with TEMPS cut to its height; else (no FANS) a cut
/// TEMPS block of at least three rows; else the TEMPS line; else nothing.
fn strip_plan(fans: Option<&FanPanel>, temps: Option<&TempPanel>, spare: u16) -> Option<Strip> {
    let fans_need = fans.map(fans_block_rows);
    let temps_need = temps.map(temps_block_rows);
    let full = fans_need.unwrap_or(0).max(temps_need.unwrap_or(0));
    if full > 0 && full < spare {
        return Some(Strip::Block(full));
    }
    if let (Some(need), Some(_)) = (fans_need, temps)
        && need < spare
    {
        return Some(Strip::Block(need));
    }
    if temps.is_some() && fans.is_none() && spare > TEMPS_CUT_MIN {
        return Some(Strip::Block(spare - 1));
    }
    temps
        .is_some()
        .then_some(Strip::Line)
        .filter(|_| spare >= 1)
}

/// The fewest rows a cut TEMPS block takes: the header, a row and the
/// summary of the rest.
const TEMPS_CUT_MIN: u16 = 3;

/// Under IN/OUT or RECENT: FANS left, TEMPS right, with a `|` between. On
/// a wide screen (text off) FANS keeps its right-hand place and TEMPS takes
/// the left. Alone, each takes its usual span.
fn draw_strip(
    grid: &mut Grid,
    g: &Geom,
    fans: Option<&FanPanel>,
    temps: Option<&TempPanel>,
    top: u16,
    rows: u16,
) {
    let wide = g.cols >= FANS_SIDE_COLS;
    match (fans, temps) {
        (Some(fans), Some(temps)) => {
            let (fans_span, temps_span, sep) = if wide {
                let split = fans_split(g.cols);
                ((split + 2, g.cols - 3), (2, split - 2), split)
            } else {
                let width = (u32::from(g.cols - 4) * 55 / 100).min(u32::from(FAN_PANEL_MAX));
                let fans_end = 2 + u16::try_from(width).unwrap_or(FAN_PANEL_MAX) - 1;
                ((2, fans_end), (fans_end + 4, g.cols - 3), fans_end + 2)
            };
            for row in top..top + rows {
                paint(grid, sep, row, '|', C16::BrightBlack, C16::Black);
            }
            draw_fans(grid, fans, fans_span.0, fans_span.1, top, rows);
            draw_temps(grid, temps, temps_span.0, temps_span.1, top, rows);
        }
        (Some(fans), None) => {
            let (left, right) = fans_span(g);
            draw_fans(grid, fans, left, right, top, rows);
        }
        (None, Some(temps)) => draw_temps(grid, temps, 2, g.cols - 3, top, rows),
        (None, None) => {}
    }
}

/// Wide screens, right of IN/OUT: FANS, a blank row, then TEMPS in the
/// rows left (cut to fit).
fn draw_side_column(
    grid: &mut Grid,
    fans: Option<&FanPanel>,
    temps: Option<&TempPanel>,
    left: u16,
    right: u16,
    top: u16,
    rows: u16,
) {
    let mut used = 0;
    if let Some(panel) = fans {
        used = fans_block_rows(panel).min(rows);
        draw_fans(grid, panel, left, right, top, used);
        used += 1;
    }
    if let Some(panel) = temps {
        let left_rows = rows.saturating_sub(used);
        if left_rows >= 2 {
            draw_temps(grid, panel, left, right, top + used, left_rows);
        }
    }
}

/// Header plus one row per TEMPS row, or one note row when there is none.
fn temps_block_rows(panel: &TempPanel) -> u16 {
    1 + u16::try_from(panel.groups.len()).unwrap_or(u16::MAX).max(1)
}

/// TEMPS row names take this many cells, then a gap.
const TEMP_NAME_W: u16 = 8;

/// Colour of a TEMPS value.
fn level_fg(level: Level) -> C16 {
    match level {
        Level::Normal => C16::BrightWhite,
        Level::Warn => C16::Yellow,
        Level::Crit => C16::BrightRed,
    }
}

/// `TEMPS`, then one row per device: its name, then `label value` items
/// with ` · ` between, cut at a whole item. Too many rows for `rows`: the
/// last row sums up the rest as `+ name hottest · ...`.
fn draw_temps(grid: &mut Grid, panel: &TempPanel, left: u16, right: u16, top: u16, rows: u16) {
    if rows == 0 || right <= left {
        return;
    }
    let width = usize::from(right - left) + 1;
    paint_fit(grid, left, top, "TEMPS", C16::White, C16::Black, width);
    // The colour key: values turn these colours at the warn and crit marks.
    if width >= 17 {
        paint_str(
            grid,
            left + 7,
            top,
            "warn",
            level_fg(Level::Warn),
            C16::Black,
        );
        paint_str(
            grid,
            left + 12,
            top,
            "crit",
            level_fg(Level::Crit),
            C16::Black,
        );
    }
    if panel.groups.is_empty() {
        if rows > 1 {
            paint_fit(
                grid,
                left,
                top + 1,
                "no temperature inputs found",
                C16::BrightBlack,
                C16::Black,
                width,
            );
        }
        return;
    }
    let body = usize::from(rows - 1);
    let whole = if panel.groups.len() <= body {
        panel.groups.len()
    } else {
        body.saturating_sub(1)
    };
    let mut row = top + 1;
    for group in &panel.groups[..whole] {
        draw_temp_group(grid, group, left, right, row);
        row += 1;
    }
    if whole < panel.groups.len() && body > 0 {
        paint_str(grid, left, row, "+", C16::BrightBlack, C16::Black);
        paint_summary(grid, &panel.groups[whole..], left + 2, right, row);
    }
}

fn draw_temp_group(grid: &mut Grid, group: &TempGroup, left: u16, right: u16, row: u16) {
    let name_w = usize::from(TEMP_NAME_W).min(usize::from(right - left) + 1);
    paint_fit(grid, left, row, &group.name, C16::White, C16::Black, name_w);
    let items: Vec<Vec<(String, C16)>> = group
        .items
        .iter()
        .map(|item| {
            let mut parts = Vec::new();
            if !item.label.is_empty() {
                parts.push((format!("{} ", item.label), C16::BrightBlack));
            }
            parts.push((celsius(item.tenths), level_fg(item.level)));
            parts
        })
        .collect();
    paint_items(grid, &items, left + TEMP_NAME_W + 1, right, row);
}

/// One TEMPS line: `TEMPS  CPU 68 · GPU 71 · NVMe0 67 ...`, the hottest
/// value of each row.
fn draw_temps_line(grid: &mut Grid, panel: &TempPanel, left: u16, right: u16, row: u16) {
    if right <= left + 7 {
        return;
    }
    paint_str(grid, left, row, "TEMPS", C16::White, C16::Black);
    paint_summary(grid, &panel.groups, left + 7, right, row);
}

fn paint_summary(grid: &mut Grid, groups: &[TempGroup], left: u16, right: u16, row: u16) {
    let items: Vec<Vec<(String, C16)>> = groups
        .iter()
        .filter_map(|group| {
            let hot = group.hottest()?;
            Some(vec![
                (format!("{} ", group.name), C16::BrightBlack),
                (celsius(hot.tenths), level_fg(hot.level)),
            ])
        })
        .collect();
    paint_items(grid, &items, left, right, row);
}

/// Items with ` · ` between, from `left` up to `right`; an item that does
/// not fit whole ends the row with `…`.
fn paint_items(grid: &mut Grid, items: &[Vec<(String, C16)>], left: u16, right: u16, row: u16) {
    const SEP: &str = " \u{00B7} ";
    let end = usize::from(right) + 1;
    let mut col = usize::from(left);
    for (i, parts) in items.iter().enumerate() {
        let sep = if i == 0 { 0 } else { SEP.chars().count() };
        let len: usize = parts.iter().map(|(t, _)| t.chars().count()).sum();
        if col + sep + len > end {
            if col + 2 <= end && i > 0 {
                paint_at(
                    grid,
                    col + 1,
                    usize::from(row),
                    '\u{2026}',
                    C16::BrightBlack,
                    C16::Black,
                );
            }
            return;
        }
        if sep > 0 {
            // `·` is a console glyph the text painter would replace.
            paint(
                grid,
                col_u16(col + 1),
                row,
                '\u{00B7}',
                C16::BrightBlack,
                C16::Black,
            );
            col += sep;
        }
        for (text, fg) in parts {
            paint_str(grid, col_u16(col), row, text, *fg, C16::Black);
            col += text.chars().count();
        }
    }
}

/// Whole degrees: `68`.
fn celsius(tenths: i32) -> String {
    format!("{}", (f64::from(tenths) / 10.0).round() as i64)
}

/// Header plus one row per fan, or one note row when the chip is absent.
fn fans_block_rows(panel: &FanPanel) -> u16 {
    let body = if panel.present {
        u16::try_from(panel.fans.len()).unwrap_or(u16::MAX).max(1)
    } else {
        1
    };
    1 + body
}

/// `FANS  <chip>`, then one row per fan: label, rpm, pwm meter, percent and
/// mode. A stalled fan (0 rpm while pwm > 0) is drawn in dim red.
fn draw_fans(grid: &mut Grid, panel: &FanPanel, left: u16, right: u16, top: u16, rows: u16) {
    if rows == 0 || right <= left {
        return;
    }
    let width = usize::from(right - left) + 1;
    paint_fit(grid, left, top, "FANS", C16::White, C16::Black, width);
    paint_fit(
        grid,
        left + 6,
        top,
        &panel.chip,
        C16::BrightBlack,
        C16::Black,
        width.saturating_sub(6),
    );
    if !panel.present {
        if rows > 1 {
            let note = if panel.chip.is_empty() {
                "no fan inputs found".to_owned()
            } else {
                format!("no single hwmon named {}", panel.chip)
            };
            paint_fit(
                grid,
                left,
                top + 1,
                &note,
                C16::BrightBlack,
                C16::Black,
                width,
            );
        }
        return;
    }
    for (i, fan) in panel.fans.iter().enumerate() {
        let Ok(i) = u16::try_from(i) else {
            break;
        };
        if i + 1 >= rows {
            break;
        }
        draw_fan_row(grid, fan, left, width, top + 1 + i);
    }
}

/// 0 rpm while the controller asks for some pwm.
#[must_use]
pub fn fan_stalled(fan: &FanReading) -> bool {
    fan.rpm == Some(0) && fan.pwm.is_some_and(|pwm| pwm > 0)
}

fn draw_fan_row(grid: &mut Grid, fan: &FanReading, left: u16, width: usize, row: u16) {
    let stalled = fan_stalled(fan);
    let (label_fg, value_fg, dim_fg) = if stalled {
        (C16::Red, C16::Red, C16::Red)
    } else {
        (C16::White, C16::BrightWhite, C16::BrightBlack)
    };
    let mut col = left;
    paint_fit(
        grid,
        col,
        row,
        &fan.label,
        label_fg,
        C16::Black,
        MAX_FAN_LABEL.min(width),
    );
    col += col_u16(MAX_FAN_LABEL + 2);
    let rpm = fan
        .rpm
        .map_or_else(|| "--".to_owned(), |rpm| rpm.to_string());
    right_align(grid, col + 4, row, &rpm, value_fg, C16::Black);
    paint_str(grid, col + 5, row, " rpm", dim_fg, C16::Black);
    col += 5 + 4 + 2;
    let bar_w = width.saturating_sub(FAN_ROW_FIXED);
    let frac = fan.pwm.map(|pwm| f64::from(pwm) / 255.0);
    if bar_w >= FAN_BAR_MIN {
        draw_h_bar(
            grid,
            HBar {
                x: col,
                row,
                width: col_u16(bar_w),
                frac,
                ink: if stalled {
                    BarInk::Fixed(C16::Red, C16::Red)
                } else {
                    BarInk::Spectrum { hot: false }
                },
                rows: 1,
                cells: None,
            },
        );
        col += col_u16(bar_w) + 1;
    }
    let pct = frac.map_or_else(
        || "--".to_owned(),
        |frac| format!("{}%", (frac * 100.0).round() as i64),
    );
    right_align(grid, col + 3, row, &pct, value_fg, C16::Black);
    col += 4 + 2;
    paint_str(grid, col, row, mode_word(fan.mode), dim_fg, C16::Black);
}

fn revealed_lines(model: &TtyModel) -> Vec<String> {
    let stream = tail_stream(&model.out_lines);
    let total = stream.len();
    let keep = match model.replay_frame {
        None => total,
        Some(frame) => {
            let before = model.out_shown.min(total);
            let delta = total - before;
            before + replay_shown(delta, frame.min(10)).min(delta)
        }
    };
    if keep == 0 {
        return Vec::new();
    }
    split_chars(&stream[..keep])
}

fn logical_lines(parts: &[String]) -> Vec<String> {
    let stream = tail_stream(parts);
    if stream.is_empty() {
        return Vec::new();
    }
    split_chars(&stream)
}

/// Sanitised tail. Parts are joined with a line-end, and a `\n` inside a
/// part is a line-end too. The result uses `\n` only as that marker.
fn tail_stream(parts: &[String]) -> Vec<char> {
    let mut out = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let mut cells = Vec::new();
        sanitize(part, &mut cells);
        for cell in cells {
            if cell.is_line_end() {
                out.push('\n');
            } else {
                out.push(cell.ch);
            }
        }
    }
    out
}

fn split_chars(chars: &[char]) -> Vec<String> {
    let mut lines = vec![String::new()];
    for ch in chars {
        if *ch == '\n' {
            lines.push(String::new());
        } else {
            lines.last_mut().expect("line").push(*ch);
        }
    }
    lines
}

fn place_lines(
    grid: &mut Grid,
    start: u16,
    rows: u16,
    lines: &[String],
    live: bool,
    out: bool,
    right: u16,
) {
    // Content occupies columns 2..=right. The layout owns the left margin.
    let width = usize::from(right).saturating_sub(1).max(1);
    let view = visual_tail(lines, width, usize::from(rows));
    let rows_n = usize::from(rows);
    let pad = rows_n.saturating_sub(view.len());
    let total = view
        .iter()
        .filter(|line| line.chars().any(|ch| ch != ' '))
        .count();
    let third = total.div_ceil(3);
    let mut seen = 0usize;
    for (offset, line) in view.iter().enumerate() {
        if !line.chars().any(|ch| ch != ' ') {
            continue;
        }
        seen += 1;
        let from_end = total - seen;
        let fg = if out && live && from_end == 0 {
            C16::BrightWhite
        } else if from_end == 0 || (out && live && from_end < third) {
            C16::White
        } else {
            C16::BrightBlack
        };
        let row = start.saturating_add(u16::try_from(pad + offset).unwrap_or(u16::MAX));
        paint_line(grid, row, line, fg);
        if out && live && from_end == 0 {
            paint(grid, cursor_col(line), row, '█', C16::BrightRed, C16::Black);
        }
    }
}

fn cursor_col(text: &str) -> u16 {
    let mut last = None;
    for (i, ch) in text.chars().enumerate() {
        if ch != ' ' {
            last = Some(i);
        }
    }
    match last {
        Some(last) => col_u16(2usize.saturating_add(last).saturating_add(1)),
        None => 2,
    }
}

/// Last `limit` visual rows. Wrapping walks back from the end and stops
/// once the window is full, so a long tail is not wrapped in full.
fn visual_tail(lines: &[String], width: usize, limit: usize) -> Vec<String> {
    if limit == 0 || lines.is_empty() {
        return Vec::new();
    }
    let width = width.max(1);
    let mut picked: Vec<&str> = Vec::new();
    let mut rows = 0usize;
    for line in lines.iter().rev() {
        picked.push(line.as_str());
        rows = rows.saturating_add(visual_row_count(line, width));
        if rows >= limit {
            break;
        }
    }
    picked.reverse();
    let mut wrapped = Vec::new();
    for (i, line) in picked.iter().enumerate() {
        let skip = if i == 0 {
            rows.saturating_sub(limit)
        } else {
            0
        };
        wrapped.extend(hard_wrap_skip(line, width, skip));
    }
    if wrapped.len() > limit {
        let drop = wrapped.len() - limit;
        wrapped.drain(0..drop);
    }
    wrapped
}

fn visual_row_count(line: &str, width: usize) -> usize {
    let n = line.chars().count();
    if n == 0 { 1 } else { n.div_ceil(width.max(1)) }
}

fn hard_wrap_skip(text: &str, width: usize, skip_rows: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return if skip_rows == 0 {
            vec![String::new()]
        } else {
            Vec::new()
        };
    }
    let start = skip_rows.saturating_mul(width).min(chars.len());
    if start >= chars.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut i = start;
    while i < chars.len() {
        let end = (i + width).min(chars.len());
        out.push(chars[i..end].iter().collect());
        i = end;
    }
    out
}

fn paint_line(grid: &mut Grid, row: u16, text: &str, fg: C16) {
    let chars: Vec<char> = text.chars().collect();
    let Some(last) = chars.iter().rposition(|ch| *ch != ' ') else {
        return;
    };
    for (i, ch) in chars.iter().enumerate().take(last + 1) {
        paint_at(grid, 2 + i, usize::from(row), *ch, fg, C16::Black);
    }
}

fn paint_title(grid: &mut Grid, row: u16, title: &str) {
    let text = title_line(title);
    let chars: Vec<char> = text.chars().collect();
    let key = if has_word(&chars, "OUT") { "OUT" } else { "IN" };
    let Some(at) = find_word(&chars, key) else {
        paint_detail(grid, 0, usize::from(row), &text, usize::MAX);
        return;
    };
    paint_str(grid, at as u16, row, key, C16::White, C16::Black);
    let mut rest = at + key.len();
    while rest < chars.len() && chars[rest] == ' ' {
        rest += 1;
    }
    if rest < chars.len() {
        let subtitle: String = chars[rest..].iter().collect();
        paint_detail(grid, rest, usize::from(row), &subtitle, usize::MAX);
    }
}

/// A panel title's first line, sanitised, keeping the `·` separator of
/// `IN (last request · 3 tool results)` (#38).
fn title_line(title: &str) -> String {
    let first = title.split(['\n', '\r']).next().unwrap_or_default();
    first
        .split('\u{00B7}')
        .map(one_line)
        .collect::<Vec<_>>()
        .join("\u{00B7}")
}

fn has_word(chars: &[char], word: &str) -> bool {
    find_word(chars, word).is_some()
}

fn find_word(chars: &[char], word: &str) -> Option<usize> {
    let needle: Vec<char> = word.chars().collect();
    if needle.is_empty() || chars.len() < needle.len() {
        return None;
    }
    (0..=chars.len() - needle.len()).find(|&at| {
        let before = at == 0 || chars[at - 1] == ' ';
        let after = at + needle.len() == chars.len() || chars[at + needle.len()] == ' ';
        before && after && chars[at..at + needle.len()] == needle[..]
    })
}

fn draw_health(grid: &mut Grid, model: &TtyModel, g: &Geom, row: u16) {
    let mut x = 2usize;
    let stop = usize::from(g.cols.saturating_sub(40));
    for seg in &model.health {
        let piece = if seg.note.is_empty() {
            format!("{} {}", seg.name, status_word(seg.status))
        } else {
            format!("{} {} {}", seg.name, status_word(seg.status), seg.note)
        };
        let next = x.saturating_add(piece.chars().count());
        if next >= stop {
            break;
        }
        paint_str(grid, col_u16(x), row, &seg.name, C16::White, C16::Black);
        let sx = x.saturating_add(seg.name.chars().count()).saturating_add(1);
        let (fg, bg) = status_colour(seg.status);
        paint_str(grid, col_u16(sx), row, status_word(seg.status), fg, bg);
        if !seg.note.is_empty() {
            let nx = sx
                .saturating_add(status_word(seg.status).chars().count())
                .saturating_add(1);
            paint_str(
                grid,
                col_u16(nx),
                row,
                &seg.note,
                C16::BrightBlack,
                C16::Black,
            );
        }
        x = next.saturating_add(3);
    }
    let tail = match model.snapshot {
        Some(seq) => format!(
            "snapshot #{seq} {}   errors {}   up {}",
            model.snapshot_age, model.errors, model.uptime
        ),
        None => format!(
            "snapshot --   errors {}   up {}",
            model.errors, model.uptime
        ),
    };
    let end = usize::from(g.cols.saturating_sub(2));
    let start = col_u16(end.saturating_add(1).saturating_sub(tail.chars().count()));
    let tail_fg = if model.errors > 0 {
        C16::Yellow
    } else {
        C16::BrightBlack
    };
    paint_str(grid, start, row, &tail, tail_fg, C16::Black);
}

fn status_word(status: HealthStatus) -> &'static str {
    match status {
        HealthStatus::Ok => "ok",
        HealthStatus::Idle => "idle",
        HealthStatus::Down => "DOWN",
        HealthStatus::Pending => "..",
        HealthStatus::Absent => "--",
    }
}

fn status_colour(status: HealthStatus) -> (C16, C16) {
    match status {
        HealthStatus::Ok | HealthStatus::Idle => (C16::Green, C16::Black),
        HealthStatus::Down => (C16::Black, C16::Yellow),
        HealthStatus::Pending | HealthStatus::Absent => (C16::BrightBlack, C16::Black),
    }
}

fn right_align(grid: &mut Grid, end: u16, row: u16, text: &str, fg: C16, bg: C16) {
    let chars: Vec<char> = one_line(text).chars().collect();
    let end_u = usize::from(end);
    let len = chars.len().min(end_u.saturating_add(1));
    let start = end_u.saturating_add(1).saturating_sub(len);
    let skip = chars.len().saturating_sub(len);
    for (i, ch) in chars.iter().skip(skip).enumerate() {
        paint_at(grid, start + i, usize::from(row), *ch, fg, bg);
    }
}

fn center_in(grid: &mut Grid, left: u16, right: u16, row: u16, text: &str, fg: C16, bg: C16) {
    let width = usize::from(right.saturating_sub(left).saturating_add(1));
    let chars: Vec<char> = one_line(text).chars().collect();
    let len = chars.len().min(width);
    let start = usize::from(left) + width.saturating_sub(len) / 2;
    for (i, ch) in chars.iter().take(len).enumerate() {
        paint_at(grid, start + i, usize::from(row), *ch, fg, bg);
    }
}

fn fill_span(grid: &mut Grid, left: u16, right: u16, row: u16, ch: char, fg: C16, bg: C16) {
    for col in left..=right {
        paint(grid, col, row, ch, fg, bg);
    }
}

fn one_line(text: &str) -> String {
    let mut cells = Vec::new();
    sanitize(text, &mut cells);
    cells
        .into_iter()
        .take_while(|cell| !cell.is_line_end())
        .map(|cell| cell.ch)
        .collect()
}

fn paint_str(grid: &mut Grid, col: u16, row: u16, text: &str, fg: C16, bg: C16) {
    paint_fit(grid, col, row, text, fg, bg, usize::MAX);
}

fn paint_fit(grid: &mut Grid, col: u16, row: u16, text: &str, fg: C16, bg: C16, cap: usize) {
    let origin = usize::from(col);
    let room = usize::from(grid.cols().saturating_sub(1)).saturating_sub(origin);
    let limit = room.min(cap);
    let mut cells = Vec::new();
    sanitize(text, &mut cells);
    for (drawn, cell) in cells.iter().take(limit).enumerate() {
        if cell.is_line_end() {
            break;
        }
        paint_at(grid, origin + drawn, usize::from(row), cell.ch, fg, bg);
    }
}

fn paint_at(grid: &mut Grid, col: usize, row: usize, ch: char, fg: C16, bg: C16) {
    let Ok(col) = u16::try_from(col) else {
        return;
    };
    let Ok(row) = u16::try_from(row) else {
        return;
    };
    paint(grid, col, row, ch, fg, bg);
}

fn paint(grid: &mut Grid, col: u16, row: u16, ch: char, fg: C16, bg: C16) {
    if row >= grid.rows().saturating_sub(1) || col >= grid.cols().saturating_sub(1) {
        return;
    }
    grid.put(col, row, Cell::new(ch, fg, bg));
}

pub(crate) fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

pub(crate) fn comma_pair(done: u64, total: u64) -> String {
    format!("{}/{}", commas(done), commas(total))
}

#[cfg(test)]
mod tests {

    /// #26: the layout's colour roles sit on the slots the palette module
    /// documents, so the llama palette paints them as designed.
    #[test]
    fn colour_roles_match_the_palette_role_map() {
        use llama_core::palette::{LLAMA, role};
        let steps: Vec<u8> = STEPS.iter().map(|c| c.index()).collect();
        assert_eq!(steps, role::SPECTRUM);
        assert_eq!(digit_colours(5), (C16::BrightWhite, C16::BrightRed));
        assert_eq!(C16::BrightWhite.index(), role::BRIGHT);
        assert_eq!(C16::BrightRed.index(), role::LIVE);
        assert_eq!(status_colour(HealthStatus::Ok).0.index(), role::OK);
        assert_eq!(status_colour(HealthStatus::Down).1.index(), role::WARN);
        assert_eq!(band(150.0).1.index(), role::ERROR);
        assert_eq!(C16::White.index(), role::TEXT);
        assert_eq!(C16::BrightBlack.index(), role::DIM);
        assert_eq!(C16::Black.index(), role::BACKGROUND);
        // The spectrum climbs the ramp: red channel never falls step to step.
        let reds: Vec<u8> = STEPS
            .iter()
            .map(|c| LLAMA[usize::from(c.index())].r)
            .collect();
        assert!(reds.windows(2).all(|w| w[0] <= w[1]), "{reds:?}");
    }
    use super::*;

    #[test]
    fn rainbow_key_clips_to_a_bar_narrower_than_the_key() {
        let mut grid = Grid::new(40, 4);
        paint_rainbow_key(&mut grid, Span { x: 10, w: 5 }, 1);
        let row: Vec<(char, C16)> = (0..40)
            .map(|col| {
                let cell = grid.get(col, 1).unwrap();
                (cell.ch, cell.fg)
            })
            .collect();
        let want: Vec<(char, C16)> = RAINBOW[..5].iter().map(|fg| ('\u{2588}', *fg)).collect();
        assert_eq!(&row[10..15], want.as_slice());
        assert!(row[15..].iter().all(|(ch, _)| *ch == ' '), "{row:?}");
        assert!(row[..10].iter().all(|(ch, _)| *ch == ' '), "{row:?}");
        // A bar running off the screen: what fits before the last column.
        let mut grid = Grid::new(16, 4);
        paint_rainbow_key(&mut grid, Span { x: 12, w: 30 }, 1);
        let drawn: Vec<C16> = (0..16)
            .map(|col| grid.get(col, 1).unwrap())
            .filter(|cell| cell.ch == '\u{2588}')
            .map(|cell| cell.fg)
            .collect();
        assert_eq!(drawn, RAINBOW[..3]);
    }

    #[test]
    fn request_time_formats_to_the_second() {
        assert_eq!(
            format_request_time("2026-09-25T15:57:08Z", false),
            "09-25 15:57:08"
        );
        assert_eq!(
            format_request_time("2026-09-25T15:57:08Z", true),
            "2026-09-25 15:57:08"
        );
        assert_eq!(
            format_request_time("2026-09-25T15:57:08.5Z", true),
            "2026-09-25 15:57:08"
        );
        assert_eq!(
            format_request_time("2026-09-25T15:57:08.123-05:00", false),
            "09-25 15:57:08"
        );
        assert_eq!(
            format_request_time("2026-09-25 15:57:08", true),
            "2026-09-25 15:57:08"
        );
        assert_eq!(format_request_time("18:47:01", true), "18:47:01");
        assert_eq!(
            format_request_time("2026-09-25T15:57:08Z\u{1b}[2J", true),
            "2026-09-25T15:57:08Z[2J"
        );
    }

    #[test]
    fn source_drops_the_ip_prefix_after_sanitising() {
        assert_eq!(format_source("ip:192.0.2.83"), "192.0.2.83");
        assert_eq!(format_source("ip:\u{1b}[2J10.1.2.3"), "[2J10.1.2.3");
        assert_eq!(format_source("10.0.0.1"), "10.0.0.1");
    }

    #[test]
    fn ellipsis_is_only_the_truncated_tail() {
        assert_eq!(fit_ellipsis("GLM-4.7 Flash", 32), "GLM-4.7 Flash");
        assert_eq!(
            fit_ellipsis(&"M".repeat(80), 8),
            format!("{}…", "M".repeat(7))
        );
        assert_eq!(fit_ellipsis("PRE\u{1b}[2JPOST\u{db}", 32), "PRE[2JPOST?");
        assert_eq!(fit_ellipsis("ab", 1), "…");
    }

    #[test]
    fn recent_plan_fills_the_width_and_upgrades_the_timestamp_first() {
        for cols in [160u16, 240, 286, 480] {
            let plan = request_plan(cols);
            assert_eq!(
                plan.status.x + plan.status.w,
                cols - 1,
                "{cols} does not reach the last paintable column"
            );
            assert!(plan.full_time, "{cols}");
            assert_eq!(usize::from(plan.time.w), TIME_FULL, "{cols}");
            assert_eq!(plan.wide, cols > 160, "{cols}");
            assert!(usize::from(plan.model.w) <= MODEL_CAP, "{cols}");
        }
        let narrow = request_plan(160);
        let mid = request_plan(240);
        let wide = request_plan(480);
        assert!(mid.bar.w > narrow.bar.w);
        assert!(wide.bar.w >= mid.bar.w);
        assert!(wide.model.w >= narrow.model.w);
        assert!(!narrow.wide);
        assert!(wide.wide);
    }

    fn slot(prompt: Option<u64>, decoded: u64, n_ctx: Option<u64>) -> Slot {
        Slot {
            id: 0,
            generating: true,
            done: 0,
            total: 0,
            decoded,
            ctx_prompt: prompt,
            n_ctx,
            ctx_history: Vec::new(),
        }
    }

    #[test]
    fn ctx_fill_clamps_used_above_n_ctx() {
        assert_eq!(ctx_fill_frac(300_000, 262_144), Some(1.0));
    }

    #[test]
    fn ctx_fill_is_absent_when_n_ctx_is_zero() {
        assert_eq!(ctx_fill_frac(15, 0), None);
    }

    #[test]
    fn missing_context_field_is_not_a_zero() {
        assert!(slot_ctx(&slot(None, 612, Some(262_144))).is_none());
        assert!(slot_ctx(&slot(Some(91_204), 612, None)).is_none());
    }

    #[test]
    fn context_used_is_prompt_plus_decoded_as_thousands() {
        let ctx = slot_ctx(&slot(Some(91_204), 612, Some(262_144))).expect("ctx");
        assert_eq!(ctx.used, 91_816);
        assert_eq!(compact_k(ctx.used), "91k");
        assert_eq!(compact_k(ctx.n_ctx), "262k");
        assert_eq!(compact_k(0), "0");
    }
}
