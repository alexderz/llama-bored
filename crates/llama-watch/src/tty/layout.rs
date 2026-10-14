//! tty11 dashboard. A pure function of [`TtyModel`], columns and rows.
//!
//! Phase B builds the grid from [`Term::cols`](super::term::Term::cols) and
//! [`Term::rows`](super::term::Term::rows). The last row and the last column
//! stay blank so the console cannot scroll.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use llama_core::color::{self, Rgb, act_color};

use super::chart::{self, ChartBucket, Ink};
use super::ctx_history::{self, CtxPoint};
use super::grid::{C16, Cell, Grid};
use super::sanitize::{detail_char, detail_glyph, sanitize};
use crate::collector::LoadSource;
use crate::config::{ChartGlyphs, MAX_FAN_LABEL};
use crate::resets::ResetReason;
use crate::sources::fans::{FanPanel, FanReading, mode_word};
use crate::sources::temps::{Level, TempGroup, TempItem, TempPanel};

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
    /// Prompt tokens computed beyond the cache.
    pub done: u64,
    /// Prompt tokens reused from the cache; with [`Self::done`], the part
    /// of the prompt the bar fills.
    pub cached: u64,
    /// The prompt (#82): exact once the slot decodes; while it is still in
    /// prefill ([`Self::open`]) what it holds so far, a lower bound.
    pub total: u64,
    /// [`Self::total`] is a lower bound: drawn `8,192/69,632+`, and the bar
    /// has no fill, since no honest fraction exists (#78, #82).
    pub open: bool,
    /// [`Self::done`] is known. False for a Strata slot reading a prompt
    /// that is not the newest request (#81): drawn `--/1,800`, no fill.
    pub done_known: bool,
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
    /// The engine did not report a cached count (#133). [`Self::cached_tok`]
    /// is then 0 for the counters that still read a number; RECENT prints
    /// `--`. A finished row draws the prompt as one dim run. An in-flight
    /// row keeps its progress and its cursor. A reported 0 stays 0.
    pub cached_unknown: bool,
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
    /// `processed` is the engine's count against the whole prompt, so the
    /// rest of it is drawn as the target track in prefill. False with
    /// `open`, and for a Strata slot that is not the newest request (#81).
    pub target: bool,
    /// A context reset just before this request, which started it from
    /// zero: its letter shows after `pp`, as in SLOTS.
    pub reset: Option<ResetReason>,
    /// The numbers are a share of several requests' totals, or a lower
    /// bound (#80): IN, CACHED and OUT are drawn `≈12,345`.
    pub approx: bool,
    /// IN and CACHED are not known (#80): drawn `--`.
    pub no_prompt: bool,
    /// OUT is not known (#80): drawn `--`.
    pub no_output: bool,
    /// The KV the request holds (#80, SGLang): the bar is this one run.
    pub held: Option<u64>,
    /// No engine numbers at all (#80): the bar's place says `streaming`
    /// and llama-swap's response bytes so far, or stays empty before the
    /// first byte.
    pub bytes: Option<u64>,
    /// Polls in which the growing count rose (#96): the bar's head
    /// spinner cycles [`SPIN`] by this, once per such poll.
    pub beats: u32,
}

/// One item on a SETUP row (#52): `kv q8_0`, drawn after `sep`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SetupItem {
    /// Separator before the item; the row's first item draws none.
    pub sep: String,
    /// Display text from a `[setup]` rule. Rule text is printable ASCII or
    /// `·`. A `text` value may also hold the spec string's own characters;
    /// [`paint_detail_fg`] maps them when it draws.
    pub text: String,
    /// A rule's `default` (nothing was set): drawn grey.
    pub dim: bool,
    /// A `kind = "text"` value (#114), not its dim default. Items after
    /// this one drop before it is cut to the row.
    pub text_value: bool,
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
    /// The header model is llama-swap `starting`: the state word is `LOADING`.
    /// The rest of the frame stays on [`Self::state`].
    pub model_loading: bool,
    /// The header model is llama-swap `stopping`: the state word is `STOPPING`.
    pub model_stopping: bool,
    /// A compute process outside llama-swap holds at least 256 MiB, no model
    /// is listed, and no job view is up: the state word is `GPU BUSY`.
    /// The rest of the frame stays on [`Self::state`].
    pub gpu_busy: bool,
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
    /// `sglang  running 1 · queued 0 · KV 75k/204k tok 37 %`, then one per
    /// llama.cpp model with KV numbers (#79). Drawn under the slot rows.
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
    /// Job view replacing GENERATION, PROMPT, the charts, SLOTS and RECENT.
    /// `None` keeps the inference layout.
    pub job: Option<JobView>,
    /// One-line job note in the header when the inference view is showing.
    pub job_note: Option<String>,
    /// llama-swap id of the model the header names. Empty when the header
    /// names no model; the claim is then not drawn (#128).
    pub model_id: String,
    /// Latest model claims from llama-swap's `/api/events` (#128).
    pub claims: Vec<HeaderClaim>,
    /// Wall clock for the claim countdown. Tests inject this.
    pub now: SystemTime,
    /// Seconds east of UTC, for a hold's `until HH:MM` in local time.
    pub tz_offset_s: i32,
}

/// One model claim as the header draws it (#128).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeaderClaim {
    /// llama-swap model id, matched against [`TtyModel::model_id`].
    pub model: String,
    /// Holder, already sanitised as SETUP `text`, at most 24 characters.
    pub holder: String,
    pub in_use: bool,
    pub expires_at: Option<SystemTime>,
    /// Set for an explicit hold.
    pub hold_until: Option<SystemTime>,
    /// Seconds east of UTC at [`Self::hold_until`], so a hold that ends
    /// after a daylight-saving change shows that hour. Unused otherwise.
    pub hold_tz_offset_s: i32,
    /// Already sanitised. Empty when there is none.
    pub hold_reason: String,
}

/// The job shown in place of the inference panels.
#[derive(Clone, Debug, PartialEq)]
pub struct JobView {
    pub job: crate::jobs::Job,
    /// Extra live jobs besides this one, drawn `+N` after the name.
    pub more: usize,
}

/// Narrowest console the dashboard draws: the RECENT columns' floor.
pub const MIN_COLS: u16 = 160;
/// Shortest console the dashboard draws (#7). Rows 0-14 are the header, a
/// rule, the six meters, ACTIVITY and the SETUP rows that fit beside SLOTS
/// (#52); then a rule, the RECENT header, 4 request rows, the legend and
/// the RECENT rule; then the health rule, the health line and the blank
/// last row: 26. Each SLOTS row past the first pushes RECENT down a row,
/// and it shows fewer requests. From [`FULL_ROWS`] SETUP takes up to two
/// more rows, and from 60 rows one more than that so a label row does not
/// push the last row out (#114).
pub const MIN_ROWS: u16 = 26;
/// From this height the text-on layout keeps IN/OUT: the chart shrinks or
/// hides first. Below it panels drop in order: IN/OUT, then FANS, then the
/// chart shrinks (the text-off split of the freed rows).
pub const FULL_ROWS: u16 = 48;

/// Draw one frame. Below [`MIN_COLS`]×[`MIN_ROWS`] the grid is the single
/// "tty too small" line. Palette colours only: the console grid does not
/// record exact colours.
pub fn layout(model: &TtyModel, cols: u16, rows: u16) -> Grid {
    layout_with(model, cols, rows, false)
}

/// [`layout`] plus the exact-colour plane (#116). The grid file's writer
/// calls this; the console tick does not.
///
/// Public for the grid tests.
#[doc(hidden)]
pub fn layout_exact(model: &TtyModel, cols: u16, rows: u16) -> Grid {
    layout_with(model, cols, rows, true)
}

fn layout_with(model: &TtyModel, cols: u16, rows: u16, exact: bool) -> Grid {
    let mut grid = if exact {
        Grid::with_exact(cols, rows)
    } else {
        Grid::new(cols, rows)
    };
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
    if model.job.is_some() {
        draw_job(&mut grid, model, &g);
    } else {
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
    let job_word = model.job.as_ref().map(|view| {
        view.job
            .kind
            .as_deref()
            .map(|kind| kind.to_ascii_uppercase())
            .unwrap_or_else(|| "JOB".to_owned())
    });
    // A job word wins, as before. LOADING, STOPPING and GPU BUSY are yellow
    // text on black (the pip the same colour), not the black-on-yellow AI DOWN badge.
    let (pip_fg, word, word_fg, word_bg) = if let Some(word) = job_word.as_deref() {
        (C16::BrightRed, word, C16::BrightRed, C16::Black)
    } else if model.model_loading {
        (C16::Yellow, "LOADING", C16::Yellow, C16::Black)
    } else if model.model_stopping {
        (C16::Yellow, "STOPPING", C16::Yellow, C16::Black)
    } else if model.gpu_busy {
        (C16::Yellow, "GPU BUSY", C16::Yellow, C16::Black)
    } else {
        match model.state {
            WatchState::Generating => (C16::BrightRed, "GENERATING", C16::BrightRed, C16::Black),
            WatchState::Ready => (C16::Green, "READY", C16::Green, C16::Black),
            WatchState::AiDown => (C16::Yellow, " AI DOWN ", C16::Black, C16::Yellow),
            WatchState::Starting => (C16::BrightBlack, "STARTING", C16::BrightBlack, C16::Black),
            WatchState::NoLlama => (C16::BrightBlack, "NO LLAMA", C16::BrightBlack, C16::Black),
        }
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
    let after_slots = x;
    let after_swap = paint_field(grid, x, 0, "swap", &model.swap_line, name_fg(model), limit);
    // The claim sits after `swap`. When `swap` itself was not drawn, the
    // claim is not drawn either: it must not take that field's place.
    // Two claims on the same model: the first is shown.
    let tag_at = if after_swap == after_slots || model.model_id.is_empty() {
        after_swap
    } else if let Some(claim) = model
        .claims
        .iter()
        .find(|claim| claim.model == model.model_id)
    {
        paint_claim(grid, after_swap, claim, model.now, model.tz_offset_s, limit)
    } else {
        after_swap
    };
    let mut note_at = tag_at;
    if !model.show_text {
        let tag_end = usize::from(tag_at).saturating_add(TEXT_OFF_TAG.chars().count());
        if tag_end.saturating_add(2) <= usize::from(start) {
            paint_str(grid, tag_at, 0, TEXT_OFF_TAG, C16::BrightBlack, C16::Black);
            note_at = col_u16(tag_end.saturating_add(2));
        }
    }
    if let Some(note) = &model.job_note {
        let room = limit.saturating_sub(usize::from(note_at));
        if room >= 4 {
            let shown = cut_text(note, room);
            let n = shown.chars().count();
            paint_fit(grid, note_at, 0, &shown, C16::BrightWhite, C16::Black, n);
        }
    }
    paint_str(grid, start, 0, &temps, fg, C16::Black);
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

/// `model <full name>  <detail>`, the detail in grey; `job <name>` while
/// the job view is up. The detail ends by column `limit` (exclusive),
/// dropping whole trailing ` · item`s (#39).
fn paint_model(grid: &mut Grid, x: u16, model: &TtyModel, limit: usize) -> u16 {
    let label = if model.job.is_some() { "job" } else { "model" };
    let label_len = label.chars().count();
    paint_fit(grid, x, 0, label, C16::BrightBlack, C16::Black, label_len);
    let vx = usize::from(x).saturating_add(label_len).saturating_add(1);
    let job_name = model.job.as_ref().map(|view| view.job.name.as_str());
    let name = job_name.unwrap_or(model.model_name.as_str());
    let drawn = name.chars().count().min(MODEL_NAME_CAP);
    paint_fit(
        grid,
        col_u16(vx),
        0,
        name,
        name_fg(model),
        C16::Black,
        drawn,
    );
    let mut end = vx.saturating_add(drawn);
    if job_name.is_some() {
        return col_u16(end.saturating_add(3));
    }
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

/// Grey detail text: SETUP's title, rows, and `model.backend_lines` (#90).
/// Printable ASCII is drawn as itself. `·` (the SETUP/[`cut_text`] separator)
/// and `≈` (an approximate KV count, #79) are this crate's own glyphs, not
/// model text, and stay literal here for the same reason [`sanitize`] leaves
/// them out of its table (its module doc: a model-sent one must not be
/// confused with the chart dot or the bullet fallback). Every other
/// character goes through [`detail_char`], the same per-character mapping
/// [`sanitize`] uses for IN/OUT and RECENT (zero-width characters and
/// controls take no cell; `…` and other typography are mapped), so `'s`,
/// an em dash or an accented letter in a model id or name reads correctly
/// here too, and an unmapped scalar draws the placeholder rather than a
/// literal `?` (never ambiguous with a `?` the text actually had). `cap`
/// counts cells drawn, not characters read. Returns the cells used.
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
    for ch in text.chars() {
        if drawn == cap {
            break;
        }
        let glyph = if matches!(ch, '\u{00B7}' | '\u{2248}') {
            (ch, ch)
        } else if let Some(glyph) = detail_glyph(ch) {
            glyph
        } else {
            continue;
        };
        paint_glyph(grid, col.saturating_add(drawn), row, glyph, fg, C16::Black);
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

const SEP_CELLS: usize = 3;

/// Header claim for the model the header names (#128). Sheds in order:
/// full; reason cut with the ellipsis (only when at least two cells of
/// reason text remain before the ellipsis); no reason; no label word;
/// nothing. The reason is never drawn once the label is dropped. Never
/// cuts the holder; if the holder alone does not fit, draws nothing.
fn paint_claim(
    grid: &mut Grid,
    x: u16,
    claim: &HeaderClaim,
    now: SystemTime,
    tz_offset_s: i32,
    limit: usize,
) -> u16 {
    let hold = claim.hold_until.is_some();
    let label = if hold { "hold" } else { "claim" };
    let holder = claim.holder.as_str();
    if holder.is_empty() {
        return x;
    }
    let status = claim_status(claim, now, tz_offset_s);
    let reason = claim.hold_reason.as_str();
    let holder_w = holder.chars().count();
    let status_w = status
        .as_ref()
        .map(|s| SEP_CELLS + s.text.chars().count())
        .unwrap_or(0);
    let core = holder_w.saturating_add(status_w);
    let room = limit.saturating_sub(usize::from(x));
    if core > room {
        return x;
    }
    let label_w = label.chars().count().saturating_add(1);
    let keep_label = label_w.saturating_add(core) <= room;
    let used = if keep_label {
        label_w.saturating_add(core)
    } else {
        core
    };
    let leftover = room.saturating_sub(used);
    // The reason rides with the label. Once the label is dropped, the
    // reason is gone too. A cut needs at least two cells of reason text
    // before the ellipsis, so a one-cell leftover is not a lone `…`.
    let shown_reason = if !keep_label || reason.is_empty() {
        None
    } else {
        fit_reason(reason, leftover)
    };

    let mut col = usize::from(x);
    if keep_label {
        let fg = if hold { C16::Yellow } else { C16::BrightBlack };
        paint_fit(
            grid,
            col_u16(col),
            0,
            label,
            fg,
            C16::Black,
            label.chars().count(),
        );
        col += label.chars().count() + 1;
    }
    paint_fit(
        grid,
        col_u16(col),
        0,
        holder,
        C16::BrightCyan,
        C16::Black,
        holder_w,
    );
    col += holder_w;
    if let Some(status) = &status {
        col += paint_sep(grid, col);
        paint_fit(
            grid,
            col_u16(col),
            0,
            &status.text,
            status.fg,
            C16::Black,
            status.text.chars().count(),
        );
        col += status.text.chars().count();
    }
    if let Some(reason) = shown_reason.as_deref().filter(|r| !r.is_empty()) {
        col += paint_sep(grid, col);
        let n = paint_detail_fg(
            grid,
            col,
            0,
            reason,
            leftover.saturating_sub(SEP_CELLS),
            C16::White,
        );
        col += n;
    }
    col_u16(col.saturating_add(3))
}

struct ClaimStatus {
    text: String,
    fg: C16,
}

/// Reason text that fits in `leftover` cells after the core, or none.
/// A cut is drawn only when at least two cells of reason text remain
/// before the ellipsis.
fn fit_reason(reason: &str, leftover: usize) -> Option<String> {
    if leftover <= SEP_CELLS {
        return None;
    }
    let room = leftover - SEP_CELLS;
    let n = reason.chars().count();
    if n <= room {
        return Some(reason.to_owned());
    }
    if room < 3 {
        return None;
    }
    let cut = cut_text(reason, room);
    let head = cut.strip_suffix('\u{2026}').unwrap_or(cut.as_str());
    if head.chars().filter(|c| *c != ' ').count() < 2 {
        return None;
    }
    Some(cut)
}

fn claim_status(claim: &HeaderClaim, now: SystemTime, tz_offset_s: i32) -> Option<ClaimStatus> {
    if let Some(until) = claim.hold_until {
        return Some(ClaimStatus {
            text: format_until(until, now, tz_offset_s, claim.hold_tz_offset_s),
            fg: C16::BrightWhite,
        });
    }
    if claim.in_use {
        return Some(ClaimStatus {
            text: "in use".to_owned(),
            fg: C16::BrightGreen,
        });
    }
    let expires = claim.expires_at?;
    let left = expires.duration_since(now).unwrap_or(Duration::ZERO);
    let secs = left.as_secs();
    Some(ClaimStatus {
        text: format!("{} left", format_left(secs)),
        fg: if secs < 60 {
            C16::Yellow
        } else {
            C16::BrightWhite
        },
    })
}

fn format_left(secs: u64) -> String {
    if secs < 3600 {
        format!("{}:{:02}", secs / 60, secs % 60)
    } else {
        format!("{} h {:02} min", secs / 3600, (secs % 3600) / 60)
    }
}

fn format_until(until: SystemTime, now: SystemTime, now_off: i32, until_off: i32) -> String {
    let until_s = unix_secs(until);
    let now_s = unix_secs(now);
    let until_local =
        crate::localtime::format_local(until_s, &crate::localtime::Zone::fixed(until_off));
    let now_local = crate::localtime::format_local(now_s, &crate::localtime::Zone::fixed(now_off));
    let hm = until_local.get(11..16).unwrap_or("--:--");
    if until_local.get(..10) == now_local.get(..10) {
        format!("until {hm}")
    } else {
        let date = until_local.get(5..10).unwrap_or("--");
        format!("until {date} {hm}")
    }
}

fn unix_secs(time: SystemTime) -> i64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(d) => i64::try_from(d.as_secs()).unwrap_or(i64::MAX),
        Err(_) => 0,
    }
}

fn paint_sep(grid: &mut Grid, col: usize) -> usize {
    paint(grid, col_u16(col), 0, ' ', C16::BrightBlack, C16::Black);
    paint(
        grid,
        col_u16(col + 1),
        0,
        '\u{00B7}',
        C16::BrightBlack,
        C16::Black,
    );
    paint(grid, col_u16(col + 2), 0, ' ', C16::BrightBlack, C16::Black);
    SEP_CELLS
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

/// SETUP rows (title included) on a screen under 60 rows.
const SETUP_ROWS: u16 = 6;
/// From 60 rows: one more than the eight the block had, so a label row
/// does not push the last row out (#114).
const SETUP_ROWS_TALL: u16 = 9;
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
    let after = x + paint_detail_fg(grid, x, usize::from(row), &id, room, C16::BrightWhite);
    let name = &view.name;
    if name.is_empty() || *name == view.id {
        return;
    }
    let text = format!("{}{name}", llama_core::detail::SEPARATOR);
    if after + detail_width(&text) <= end {
        paint_detail(grid, after, usize::from(row), &text, end - after);
    }
}

/// `  label    item · item`. Trailing items drop until the row fits, then
/// the last one is cut. A `text` value stays: items after it drop first,
/// and the value is cut to the room rather than dropped (#114). A text
/// value that is not the row's first item, with fewer than two cells left,
/// drops with its separator: a bare ` · ` or a lone `…` is not drawn.
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
    let keep = setup_items_kept(&setup.items, room);
    let mut col = x;
    for (i, item) in setup.items[..keep].iter().enumerate() {
        let used = col - x;
        if i > 0 {
            let sep_room = room.saturating_sub(used);
            let sep_w = detail_width(&item.sep).min(sep_room);
            if item.text_value && sep_room.saturating_sub(sep_w) < 2 {
                break;
            }
            col += paint_detail(grid, col, usize::from(row), &item.sep, sep_room);
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

/// `text`, or its first `width - 1` characters and `…`. Trailing spaces of
/// that prefix are dropped, so the cut is `hello…` rather than `hello …`.
/// The cut itself does not sanitise: [`paint_detail_fg`] maps every
/// non-ASCII scalar it is handed (including the `…` this appends) when it
/// draws the result. [`crate::setup_rules`] uses the same cut for a `text`
/// value's `max_chars`.
pub(crate) fn cut_text(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    let mut out: String = text.chars().take(width - 1).collect();
    let end = out.trim_end_matches(' ').len();
    out.truncate(end);
    out.push('\u{2026}');
    out
}

/// Leading items to draw in `room` cells. A `text` value is the floor:
/// items after it drop while the row is too wide, and the value itself
/// is left to be cut. A text value that is not first, with fewer than two
/// cells left after its separator, drops with that separator. Any other
/// row drops down to its first item.
fn setup_items_kept(items: &[SetupItem], room: usize) -> usize {
    let mut keep = items.len();
    let text_at = items.iter().position(|item| item.text_value);
    let floor = text_at.map_or(1, |at| at.saturating_add(1));
    while keep > floor && setup_items_width(&items[..keep]) > room {
        keep -= 1;
    }
    if let Some(at) = text_at
        && at > 0
        && keep > at
        && text_item_room(items, at, room) < 2
    {
        keep = at;
    }
    keep
}

/// Cells left for `items[at]` after the items before it and its separator.
fn text_item_room(items: &[SetupItem], at: usize, room: usize) -> usize {
    let sep = detail_width(&items[at].sep);
    room.saturating_sub(setup_items_width(&items[..at]).saturating_add(sep))
}

fn setup_items_width(items: &[SetupItem]) -> usize {
    items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let sep = if i > 0 { detail_width(&item.sep) } else { 0 };
            sep + detail_width(&item.text)
        })
        .sum()
}

/// Cells [`paint_detail_fg`] draws for `text`: characters [`detail_char`]
/// drops (zero-width, controls) take none (#90).
fn detail_width(text: &str) -> usize {
    text.chars()
        .filter(|&ch| matches!(ch, '\u{00B7}' | '\u{2248}') || detail_char(ch).is_some())
        .count()
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

/// Spectrum ink for a level meter. Past 100 % the top step is white-hot,
/// and the exact ramp is stretched to end at the level.
fn level_ink(pct: Option<f64>) -> BarInk {
    BarInk::Spectrum {
        hot: pct.is_some_and(|p| p > 100.0),
        top: pct.map_or(color::spectrum::FULL, color::spectrum::top_of),
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
    /// `top` is the ramp position of the bar's right edge for the exact
    /// colours: [`color::spectrum::FULL`], or the level itself past 100 %.
    Spectrum { hot: bool, top: f32 },
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
            BarInk::Spectrum { hot, .. } => spectrum_colours(i, width, hot),
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
        // Filled spectrum cells take the ramp sampled at the ends: the
        // first cell is the first stop and the last cell of a full bar is
        // `top` (#130). The empty track and a fixed-ink bar (capacity, a
        // stalled fan) stay on the palette. Both rows of a tall bar are
        // that colour.
        if grid.records_exact()
            && ch != '░'
            && let BarInk::Spectrum { top, .. } = bar.ink
        {
            let exact = act_color(color::spectrum::place_to(i, width, top));
            grid.set_exact_fg(col, bar.row, exact);
            if bar.rows > 1 {
                grid.set_exact_fg(col, bar.row + 1, exact);
            }
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
    let gen_top = rate_top(model.gen_tps, model.gen_ceiling);
    let prompt_top = rate_top(model.prompt_tps, model.prompt_ceiling);
    draw_rate_bar(
        grid, g.right, bar_row, g.half, gen_frac, g.bar_rows, gen_top,
    );
    draw_rate_bar(
        grid,
        g.prompt_x(),
        bar_row,
        g.half,
        prompt_frac,
        g.bar_rows,
        prompt_top,
    );
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

/// Ramp position of a rate bar's right edge for the exact colours: full
/// scale up to the ceiling, and past it the value's own position, the one
/// the digits above the bar take (#123).
fn rate_top(value: Option<f64>, ceiling: f64) -> f32 {
    value.map_or(color::spectrum::FULL, |v| color::spectrum::rate(v, ceiling))
}

fn draw_rate_bar(
    grid: &mut Grid,
    x: u16,
    row: u16,
    width: u16,
    frac: Option<f64>,
    rows: u16,
    top: f32,
) {
    let hot = frac.is_some_and(|frac| frac >= 1.0);
    draw_h_bar(
        grid,
        HBar {
            x,
            row,
            width,
            frac,
            ink: BarInk::Spectrum { hot, top },
            rows,
            // #95: GENERATION and PROMPT each draw one isolated bar (the
            // tick row sits right below, no other bar directly above or
            // under it), so the half-cell caps may keep full height here.
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
    let (text, top, body, pos) = match value {
        None => ("--".to_string(), C16::BrightBlack, C16::BrightBlack, None),
        // NaN is not a rate: the same dim zero as a non-positive value.
        // Positive infinity stays live, at the end of the tail (#130).
        Some(v) if v.is_nan() || v <= 0.0 => {
            ("0".to_string(), C16::BrightBlack, C16::BrightBlack, None)
        }
        Some(v) => {
            let step = rate_step(v, ceiling);
            let (top, body) = digit_colours(step);
            // Dim values (above) stay on the palette. A live value takes
            // the ramp position of the same scale that picked `step`.
            let pos = grid
                .records_exact()
                .then(|| color::spectrum::rate(v, ceiling));
            ((v.round() as i64).to_string(), top, body, pos)
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
                        let y = row + py as u16 * ph + dy;
                        paint(grid, cx + px as u16 * pw + dx, y, '█', colour, C16::Black);
                        // The top console row of the whole digit is 14 ramp
                        // points above the value, the bottom row 14 below.
                        if let Some(pos) = pos {
                            let rows = 5 * usize::from(ph);
                            let at = py * usize::from(ph) + usize::from(dy);
                            grid.set_exact_fg(
                                cx + px as u16 * pw + dx,
                                y,
                                act_color(color::spectrum::shade(pos, at, rows)),
                            );
                        }
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
        // #82: the prompt in the KV against the whole prompt; nothing
        // while the whole prompt is not known.
        let frac = (slot.total > 0 && !slot.open && slot.done_known)
            .then(|| (slot.cached.saturating_add(slot.done) as f64 / slot.total as f64).min(1.0));
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
                // #95: slot rows stack with no blank row between. The
                // half-cell caps (`▐`/`▌`) are full height and a solid `█`
                // would merge with the row above or below, so this bar
                // keeps #52's whole-cell glyph instead.
                cells: Some(meter_glyph(model.chart_glyphs)),
            },
        );
        let (count, decoded, fg) = if slot.total == 0 && !slot.generating {
            ("idle".to_string(), "--".to_string(), C16::BrightBlack)
        } else {
            let pair = if slot.done_known {
                comma_pair(slot.done, slot.total)
            } else {
                format!("--/{}", commas(slot.total))
            };
            (
                if slot.open { format!("{pair}+") } else { pair },
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
            None => paint_slot_ctx(grid, row, after_decoded, g.cols, slot, model.chart_glyphs),
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
                ink: BarInk::Spectrum {
                    hot: false,
                    top: color::spectrum::FULL,
                },
                rows: 1,
                // #95: one ctx meter per slot row, stacked with no blank
                // row between, so it needs #52's whole-cell glyph too.
                cells: Some(meter_glyph(model.chart_glyphs)),
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

/// One-row sparkline cell: `▄` in halves, `▁`..`▇` in eighths. Any used
/// context shows at least the smallest step. #95: one sparkline per slot
/// row, stacked with no blank row between, so a full cell never draws the
/// full-height `█` either — the same non-bridging fallback #52 gave the
/// meters (`▇` in eighths, `▄` in halves).
fn spark_glyph(frac: f64, any: bool, glyphs: ChartGlyphs) -> Option<char> {
    let (level, top) = match glyphs {
        ChartGlyphs::Halves => (chart::halves(frac, 1), 2),
        ChartGlyphs::Eighths => (chart::eighths(frac, 1), 8),
    };
    let level = if any { level.max(1) } else { level };
    match (glyphs, level) {
        (_, 0) => None,
        (ChartGlyphs::Halves, _) => Some('\u{2584}'),
        (ChartGlyphs::Eighths, l) if l >= top => Some('\u{2587}'),
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
#[must_use]
pub fn compact_k(n: u64) -> String {
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
fn paint_slot_ctx(
    grid: &mut Grid,
    row: u16,
    after_decoded: u16,
    cols: u16,
    slot: &Slot,
    glyphs: ChartGlyphs,
) {
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
                ink: BarInk::Spectrum {
                    hot: false,
                    top: color::spectrum::FULL,
                },
                rows: 1,
                // #95: stacks with the slot rows above and below it.
                cells: Some(meter_glyph(glyphs)),
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
/// #75: a bar without a context size is scaled to the largest row.
const LEGEND_RELATIVE: &str = " · ~bar = vs largest row";

/// The RECENT legend: the column words, then what fits of the measured
/// note and the relative note, dropped in reverse order of that list when
/// the row is short. The bar's own colour key is the header row (#96).
fn paint_legend(grid: &mut Grid, row: u16, cols: u16, base: &str, measured: bool, relative: bool) {
    let room = usize::from(cols.saturating_sub(3));
    let mut items: Vec<&str> = Vec::new();
    if measured {
        items.push(LEGEND_MEASURED);
    }
    if relative {
        items.push(LEGEND_RELATIVE);
    }
    let len = |items: &[&str]| {
        base.chars().count() + items.iter().map(|i| i.chars().count()).sum::<usize>()
    };
    while !items.is_empty() && len(&items) > room {
        let drop = [LEGEND_RELATIVE, LEGEND_MEASURED]
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
        paint_fixed(grid, col, row, item, C16::BrightBlack);
        col += u16_from(item.chars().count());
    }
}

/// The head spinner's cursor colour cycle (#96): red, light green, white,
/// magenta. One step per poll in which the growing segment's token count
/// rose ([`crate::inflight::Flight::beats`] / [`InFlight::beats`]), not
/// per frame, so the cursor animates only when the request made progress.
pub const SPIN: [C16; 4] = [
    C16::Red,
    C16::BrightGreen,
    C16::BrightWhite,
    C16::BrightMagenta,
];

/// RECENT's header key (#75, #96): a fixed marker per segment with its dim
/// label, in the bar's first cells, replacing the rainbow strip (#77) now
/// that a segment is always one colour. Below 17 cells "cached" shortens
/// to "cach" so the key still fits a 160-column console; narrower still,
/// it draws what fits and the rest runs off the edge of the grid. The
/// marker is [`meter_glyph`]: `▇` with a llama-hack font, `▄` in halves
/// mode, since eurlatgr lacks the eighths block.
fn paint_segment_key(grid: &mut Grid, bar: Span, row: u16, glyphs: ChartGlyphs) {
    let mark = meter_glyph(glyphs).to_string();
    let cached_label = if bar.w >= 17 { "cached" } else { "cach" };
    let items: [(&str, C16); 8] = [
        (mark.as_str(), C16::Blue),
        (cached_label, C16::BrightBlack),
        (" ", C16::BrightBlack),
        (mark.as_str(), C16::BrightCyan),
        ("new", C16::BrightBlack),
        (" ", C16::BrightBlack),
        (mark.as_str(), C16::Yellow),
        ("out", C16::BrightBlack),
    ];
    let mut col = bar.x;
    for (text, fg) in items {
        for ch in text.chars() {
            paint(grid, col, row, ch, fg, C16::Black);
            col = col.saturating_add(1);
        }
    }
}

/// Which part of a RECENT context bar a cell draws (#75).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CtxPart {
    /// Cached input.
    Cached,
    /// The whole prompt when the cached count was not reported (#133).
    /// One full-height run in the dim grey of a `--` cell, not the new
    /// colour and not the lower step.
    Prompt,
    /// New input (input minus cached, never below 0). Drawn a step lower
    /// than the other segments (#96) so cached → new → out read as steps.
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
    /// This is the growing segment's head: a full cell in this colour
    /// instead of [`CtxPart`]'s fixed one (#96), the spinner cursor.
    pub cursor: Option<C16>,
}

/// `▇`: a whole cell, the hairline rule (#95). Never `█` (#96).
const FULL: u8 = 8;
/// A non-zero segment is at least `▂`, whatever its size (#96).
const MIN_E: u8 = 2;

/// Which flight segment is growing, and the head spinner's step (#96).
/// `None` (a finished row, or a row with no engine numbers) draws no
/// cursor and no segment spins.
#[derive(Clone, Copy, Debug)]
pub struct Spin {
    /// The growing segment is `out` while decoding, else `new`.
    pub decoding: bool,
    /// Polls in which the growing count rose: the index into [`SPIN`].
    pub beats: u32,
}

/// One segment's heights in eighths, in bar order (cached, new, out):
/// whole cells are [`FULL`], then a remainder cell of at least [`MIN_E`];
/// a non-zero segment under one eighth still shows one [`MIN_E`] cell.
/// Exact integer maths: the same count always gives the same heights.
fn seg_heights(value: u64, scale: u64, width: usize) -> Vec<u8> {
    let scale = u128::from(scale.max(1));
    let units = u128::from(value) * width as u128 * 8 / scale;
    let whole = usize::try_from(units / 8).unwrap_or(usize::MAX).min(width);
    let rem = u8::try_from(units % 8).unwrap_or(0);
    let mut heights = vec![FULL; whole];
    if rem > 0 {
        heights.push(rem.max(MIN_E));
    } else if whole == 0 && value > 0 {
        heights.push(MIN_E);
    }
    heights
}

/// The pending target's cell count: the eighths it would take (rounded up),
/// then floored to whole cells, at least one while `value > 0`. Drawn as a
/// low `▁` track, never stepped or spun.
fn pending_cells(value: u64, scale: u64, width: usize) -> usize {
    let scale = u128::from(scale.max(1));
    let eighths = u128::from(value) * width as u128 * 8;
    let ceil_eighths = eighths.div_ceil(scale);
    usize::try_from((ceil_eighths / 8).max(1))
        .unwrap_or(usize::MAX)
        .min(width)
}

/// The step option (#96): the new-input chunk sits lower than cached and
/// out, so a whole cell is `▄` (4 eighths) and a partial one is `▂`/`▃`
/// (clamped to 2..=3, rounded from its eighths height).
fn step_new(heights: &mut [u8]) {
    for h in heights {
        *h = if *h == FULL {
            4
        } else {
            MIN_E.max((*h).div_ceil(2).min(3))
        };
    }
}

/// The cells of a context bar `width` cells wide where the full width is
/// `scale` tokens (#75, #96). Segments draw in order (cached, new, out),
/// each a run of [`seg_heights`]; pending draws its rounded cell count as a
/// low track between new and out. `spin` names the growing segment (new
/// while in prefill, out while decoding): its last cell becomes a full
/// cursor in [`SPIN`]`[beats % 4]`, or, if the segment has no cells yet, a
/// single cursor cell is inserted (before pending, when it is shown). The
/// rest is [`CtxPart::Track`], and nothing passes `width`.
#[must_use]
pub fn ctx_cells(
    parts: &[(CtxPart, u64)],
    scale: u64,
    width: usize,
    spin: Option<Spin>,
) -> Vec<CtxCell> {
    enum Seg {
        Part(CtxPart, Vec<u8>),
        Pending(usize),
    }
    let mut segs: Vec<Seg> = Vec::new();
    for (part, value) in parts {
        if *value == 0 {
            continue;
        }
        segs.push(match part {
            CtxPart::Pending => Seg::Pending(pending_cells(*value, scale, width)),
            _ => Seg::Part(*part, seg_heights(*value, scale, width)),
        });
    }
    let growing = spin.map(|s| {
        if s.decoding {
            CtxPart::Out
        } else {
            CtxPart::New
        }
    });
    if let Some(growing) = growing {
        let present = segs
            .iter()
            .any(|s| matches!(s, Seg::Part(part, _) if *part == growing));
        if !present {
            let at = segs
                .iter()
                .position(|s| matches!(s, Seg::Pending(_)))
                .unwrap_or(segs.len());
            segs.insert(at, Seg::Part(growing, Vec::new()));
        }
    }
    for seg in &mut segs {
        if let Seg::Part(CtxPart::New, heights) = seg {
            step_new(heights);
        }
    }
    let mut cells: Vec<CtxCell> = Vec::with_capacity(width);
    'segs: for seg in &segs {
        match seg {
            Seg::Pending(n) => cells.extend(std::iter::repeat_n(
                CtxCell {
                    part: CtxPart::Pending,
                    eighths: 1,
                    cursor: None,
                },
                *n,
            )),
            Seg::Part(part, heights) => {
                let mut drawn: Vec<CtxCell> = heights
                    .iter()
                    .map(|h| CtxCell {
                        part: *part,
                        eighths: *h,
                        cursor: None,
                    })
                    .collect();
                if Some(*part) == growing {
                    let cursor = spin.map(|s| SPIN[(s.beats % 4) as usize]);
                    let head = CtxCell {
                        part: *part,
                        eighths: FULL,
                        cursor,
                    };
                    match drawn.last_mut() {
                        Some(last) => *last = head,
                        None => drawn.push(head),
                    }
                }
                cells.extend(drawn);
            }
        }
        if cells.len() >= width {
            break 'segs;
        }
    }
    cells.truncate(width);
    cells.resize(
        width,
        CtxCell {
            part: CtxPart::Track,
            eighths: 1,
            cursor: None,
        },
    );
    cells
}

/// A row's bar segments: cached, new (computed so far, in flight), the
/// prompt still to compute (in flight), output.
fn ctx_parts(req: &Activity) -> [(CtxPart, u64); 4] {
    // #80: what an SGLang request holds is one run; a request with no
    // engine numbers draws nothing.
    if let Some(flight) = req.inflight {
        if let Some(held) = flight.held {
            return [
                (CtxPart::Cached, 0),
                (CtxPart::New, held),
                (CtxPart::Pending, 0),
                (CtxPart::Out, 0),
            ];
        }
        if flight.bytes.is_some() {
            return [
                (CtxPart::Cached, 0),
                (CtxPart::New, 0),
                (CtxPart::Pending, 0),
                (CtxPart::Out, 0),
            ];
        }
    }
    // #133: a finished row with no cached count draws the prompt as one
    // dim run. An in-flight row keeps the done/pending split and the
    // cursor it had when an omitted count was read as zero; only the
    // CACHED cell is `--`. A reported 0 takes the path below.
    if req.cached_unknown && req.inflight.is_none() {
        return [
            (CtxPart::Prompt, req.input_tok),
            (CtxPart::New, 0),
            (CtxPart::Pending, 0),
            (CtxPart::Out, req.output_tok),
        ];
    }
    let cached = if req.cached_unknown {
        0
    } else {
        req.cached_tok.min(req.input_tok)
    };
    let new = req.input_tok - cached;
    let (done, pending) = match req.inflight {
        Some(flight) if !flight.decoding && !flight.open && flight.target => {
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

/// `e` eighths: `▁`..`▇`; halves: `▄`. #95: RECENT rows stack with no blank
/// row between, same as the meters (#52), so a "whole" cell (`e == 8`) never
/// draws the full-height `█` either — it keeps the lower seven-eighths block
/// (or, in halves, the same `▄` fallback the meters use), leaving the same
/// hairline gap above every row.
fn eighth_glyph(eighths: u8, glyphs: ChartGlyphs) -> char {
    match glyphs {
        ChartGlyphs::Eighths if eighths < 8 => {
            char::from_u32(0x2580 + u32::from(eighths.max(1))).unwrap_or('\u{2587}')
        }
        ChartGlyphs::Eighths => '\u{2587}',
        ChartGlyphs::Halves => '\u{2584}',
    }
}

/// RECENT's bar (#75, #96): the request's context, cached / new / output in
/// fixed colours (the header key), against the model's context size; `~`
/// before a bar scaled to the largest row instead; a `!` after one at 90 %
/// of the context. An in-flight row's growing segment carries the head
/// spinner (#96): its cursor cell cycles [`SPIN`] a step per poll its
/// token count rose. A finished row never spins.
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
    let spin = req.inflight.map(|flight| Spin {
        decoding: flight.decoding,
        beats: flight.beats,
    });
    let cells = ctx_cells(&parts, scale, width, spin);
    for (i, cell) in cells.iter().enumerate() {
        let (ch, fg) = match cell.part {
            CtxPart::Track => (track_glyph(glyphs), C16::BrightBlack),
            CtxPart::Pending => (track_glyph(glyphs), C16::BrightCyan),
            part => {
                let base = match part {
                    CtxPart::Cached => C16::Blue,
                    CtxPart::New => C16::BrightCyan,
                    CtxPart::Prompt => C16::BrightBlack,
                    _ => C16::Yellow,
                };
                (
                    eighth_glyph(cell.eighths, glyphs),
                    cell.cursor.unwrap_or(base),
                )
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
        // #96: the out segment is now yellow, so the warning takes the
        // brighter step of that colour to stay distinct from it.
        let fg = if dim {
            C16::BrightBlack
        } else {
            C16::BrightYellow
        };
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
            draw_strip(
                grid,
                g,
                fans,
                temps,
                strip_rule + 1,
                height,
                model.chart_glyphs,
            );
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
    paint_segment_key(grid, plan.bar, header, model.chart_glyphs);
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
        let flight = req.inflight.unwrap_or_default();
        // #80: `--` for what the engine does not say, `≈` for a share.
        let input = if flight.no_prompt {
            "--".to_owned()
        } else if open {
            format!("{}+", commas(req.input_tok))
        } else {
            commas(req.input_tok)
        };
        let known = |unknown: bool, n: u64| {
            if unknown {
                ("--".to_owned(), false)
            } else {
                (commas(n), flight.approx)
            }
        };
        let approx_in = flight.approx && !flight.no_prompt;
        paint_span_approx(grid, plan.prompt_tok, row, &input, approx_in, hot);
        let (cached, approx) = if req.cached_unknown {
            ("--".to_owned(), false)
        } else {
            known(flight.no_prompt, req.cached_tok)
        };
        paint_span_approx(grid, plan.cached, row, &cached, approx, plain);
        let (output, approx) = known(flight.no_output, req.output_tok);
        paint_span_approx(grid, plan.output, row, &output, approx, hot);
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
        if let Some(bytes) = flight.bytes.filter(|_| req.inflight.is_some()) {
            // #80: no engine numbers; llama-swap's bytes are the progress.
            if bytes > 0 {
                let text = format!("streaming {}", byte_text(bytes));
                paint_span_left(grid, plan.bar, row, &text, C16::BrightBlack, C16::Black);
            }
        } else {
            draw_ctx_bar(
                grid,
                req,
                plan.bar,
                row,
                relative_scale,
                dim,
                model.chart_glyphs,
            );
        }
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

/// [`paint_span_right`], with `≈` before the text when `approx` (#80).
/// Painted as a cell: the sanitiser would turn `≈` into `?`.
fn paint_span_approx(grid: &mut Grid, span: Span, row: u16, text: &str, approx: bool, fg: C16) {
    paint_span_right(grid, span, row, text, fg, C16::Black);
    let len = u16::try_from(text.chars().count()).unwrap_or(u16::MAX);
    if approx && len < span.w {
        paint(
            grid,
            span.x + span.w - len - 1,
            row,
            '\u{2248}',
            fg,
            C16::Black,
        );
    }
}

/// llama-swap's response bytes as `812 B`, `12.3 kB`, `4.6 MB` (#80).
#[must_use]
pub fn byte_text(bytes: u64) -> String {
    if bytes < 1_000 {
        format!("{bytes} B")
    } else if bytes < 1_000_000 {
        format!("{:.1} kB", bytes as f64 / 1e3)
    } else {
        format!("{:.1} MB", bytes as f64 / 1e6)
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
    for (i, glyph) in one_line_glyphs(text).into_iter().take(width).enumerate() {
        let Ok(dx) = u16::try_from(i) else {
            break;
        };
        paint_rich(grid, span.x.saturating_add(dx), row, glyph, fg, bg);
    }
}

fn paint_span_right(grid: &mut Grid, span: Span, row: u16, text: &str, fg: C16, bg: C16) {
    let width = usize::from(span.w);
    if width == 0 {
        return;
    }
    let chars = one_line_glyphs(text);
    let skip = chars.len().saturating_sub(width);
    let start = width.saturating_sub(chars.len());
    for (i, glyph) in chars.iter().skip(skip).enumerate() {
        let Ok(dx) = u16::try_from(start + i) else {
            break;
        };
        paint_rich(grid, span.x.saturating_add(dx), row, *glyph, fg, bg);
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
            let rows_n = usize::from(half);
            for (i, ink) in cells.into_iter().enumerate() {
                // Index 0 is the far row. The baseline is the cool end.
                let from_axis = rows_n - 1 - i;
                paint_ink(
                    grid,
                    col,
                    axis - half + i as u16,
                    ink,
                    colour,
                    chart_exact(grid, ink, from_axis, rows_n),
                );
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
            let rows_n = usize::from(half);
            for (i, ink) in cells.into_iter().enumerate() {
                // Index 0 is the row nearest the axis, the cool end.
                paint_ink(
                    grid,
                    col,
                    axis + 1 + i as u16,
                    ink,
                    colour,
                    chart_exact(grid, ink, i, rows_n),
                );
            }
        }
    }
}

/// The ramp colour of the level this row stands for, or `None` for an
/// empty cell and for a grid that does not record exact colours.
fn chart_exact(grid: &Grid, ink: Ink, from_axis: usize, rows: usize) -> Option<Rgb> {
    if matches!(ink, Ink::Empty) || !grid.records_exact() {
        return None;
    }
    Some(act_color(color::spectrum::place(from_axis, rows)))
}

fn half_inks(glyphs: Vec<char>) -> Vec<Ink> {
    glyphs
        .into_iter()
        .map(|ch| if ch == ' ' { Ink::Empty } else { Ink::Fg(ch) })
        .collect()
}

fn paint_ink(grid: &mut Grid, col: u16, row: u16, ink: Ink, colour: C16, exact: Option<Rgb>) {
    match ink {
        Ink::Empty => {}
        Ink::Fg(ch) => {
            paint(grid, col, row, ch, colour, C16::Black);
            if let Some(rgb) = exact {
                grid.set_exact_fg(col, row, rgb);
            }
        }
        // The falling tip draws black on the bar colour, so the ramp is
        // the background: that is the colour the cell shows.
        Ink::Inverse(ch) => {
            paint(grid, col, row, ch, C16::Black, colour);
            if let Some(rgb) = exact {
                grid.set_exact_bg(col, row, rgb);
            }
        }
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
            draw_side_column(
                grid,
                fans,
                temps,
                fans_span(g),
                in_label,
                health_rule - in_label,
                model.chart_glyphs,
            );
        } else {
            // Under IN/OUT: a rule, then the blocks side by side, if IN and
            // OUT keep three rows each; else TEMPS on one line (#74).
            let spare = health_rule.saturating_sub(in_label + TEXT_FLOOR_SPAN);
            match strip_plan(fans, temps, spare) {
                Some(Strip::Block(height)) => {
                    health_rule -= 1 + height;
                    draw_rule(grid, health_rule, g.cols);
                    draw_strip(
                        grid,
                        g,
                        fans,
                        temps,
                        health_rule + 1,
                        height,
                        model.chart_glyphs,
                    );
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
    glyphs: ChartGlyphs,
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
            draw_fans(grid, fans, fans_span.0, fans_span.1, top, rows, glyphs);
            draw_temps(grid, temps, temps_span.0, temps_span.1, top, rows);
        }
        (Some(fans), None) => {
            let (left, right) = fans_span(g);
            draw_fans(grid, fans, left, right, top, rows, glyphs);
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
    span: (u16, u16),
    top: u16,
    rows: u16,
    glyphs: ChartGlyphs,
) {
    let (left, right) = span;
    let mut used = 0;
    if let Some(panel) = fans {
        used = fans_block_rows(panel).min(rows);
        draw_fans(grid, panel, left, right, top, used, glyphs);
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
    let items: Vec<Vec<InkText>> = group
        .items
        .iter()
        .map(|item| {
            let mut parts = Vec::new();
            if !item.label.is_empty() {
                parts.push(InkText::plain(format!("{} ", item.label), C16::BrightBlack));
            }
            parts.push(temp_value(grid, item));
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
    let items: Vec<Vec<InkText>> = groups
        .iter()
        .filter_map(|group| {
            let hot = group.hottest()?;
            Some(vec![
                InkText::plain(format!("{} ", group.name), C16::BrightBlack),
                temp_value(grid, hot),
            ])
        })
        .collect();
    paint_items(grid, &items, left, right, row);
}

/// One run of TEMPS text. `exact` is the value's ramp colour when this
/// grid records it. Labels leave it empty.
struct InkText {
    text: String,
    fg: C16,
    exact: Option<Rgb>,
}

impl InkText {
    fn plain(text: String, fg: C16) -> Self {
        Self {
            text,
            fg,
            exact: None,
        }
    }
}

/// A TEMPS value: the palette level, and the threshold ramp when this
/// grid records exact colours and the reading has a usable pair.
fn temp_value(grid: &Grid, item: &TempItem) -> InkText {
    InkText {
        text: celsius(item.tenths),
        fg: level_fg(item.level),
        exact: temp_exact(grid, item),
    }
}

fn temp_exact(grid: &Grid, item: &TempItem) -> Option<Rgb> {
    if !grid.records_exact() {
        return None;
    }
    let (warn, crit) = item.thresholds?;
    color::temp::color(
        item.tenths as f32 / 10.0,
        warn as f32 / 10.0,
        crit as f32 / 10.0,
    )
}

/// Items with ` · ` between, from `left` up to `right`; an item that does
/// not fit whole ends the row with `…`.
fn paint_items(grid: &mut Grid, items: &[Vec<InkText>], left: u16, right: u16, row: u16) {
    const SEP: &str = " \u{00B7} ";
    let end = usize::from(right) + 1;
    let mut col = usize::from(left);
    for (i, parts) in items.iter().enumerate() {
        let sep = if i == 0 { 0 } else { SEP.chars().count() };
        let len: usize = parts.iter().map(|part| part.text.chars().count()).sum();
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
        for part in parts {
            let start = col;
            paint_str(grid, col_u16(col), row, &part.text, part.fg, C16::Black);
            if let Some(rgb) = part.exact {
                for dx in 0..part.text.chars().count() {
                    grid.set_exact_fg(col_u16(start + dx), row, rgb);
                }
            }
            col += part.text.chars().count();
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
fn draw_fans(
    grid: &mut Grid,
    panel: &FanPanel,
    left: u16,
    right: u16,
    top: u16,
    rows: u16,
    glyphs: ChartGlyphs,
) {
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
        draw_fan_row(grid, fan, left, width, top + 1 + i, glyphs);
    }
}

/// 0 rpm while the controller asks for some pwm.
#[must_use]
pub fn fan_stalled(fan: &FanReading) -> bool {
    fan.rpm == Some(0) && fan.pwm.is_some_and(|pwm| pwm > 0)
}

fn draw_fan_row(
    grid: &mut Grid,
    fan: &FanReading,
    left: u16,
    width: usize,
    row: u16,
    glyphs: ChartGlyphs,
) {
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
                    BarInk::Spectrum {
                        hot: false,
                        top: color::spectrum::FULL,
                    }
                },
                rows: 1,
                // #95: one fan per row, stacked with no blank row between.
                cells: Some(meter_glyph(glyphs)),
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

/// One sanitised character of model text: the console `ch` and the `rich`
/// scalar (#107). `'\n'` is the line-end marker in a stream.
type Glyph = (char, char);

fn revealed_lines(model: &TtyModel) -> Vec<Vec<Glyph>> {
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

fn logical_lines(parts: &[String]) -> Vec<Vec<Glyph>> {
    let stream = tail_stream(parts);
    if stream.is_empty() {
        return Vec::new();
    }
    split_chars(&stream)
}

/// Sanitised tail. Parts are joined with a line-end, and a `\n` inside a
/// part is a line-end too. The result uses `\n` only as that marker.
fn tail_stream(parts: &[String]) -> Vec<Glyph> {
    let mut out = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            out.push(('\n', '\n'));
        }
        let mut cells = Vec::new();
        sanitize(part, &mut cells);
        for cell in cells {
            if cell.is_line_end() {
                out.push(('\n', '\n'));
            } else {
                out.push((cell.ch, cell.rich));
            }
        }
    }
    out
}

fn split_chars(chars: &[Glyph]) -> Vec<Vec<Glyph>> {
    let mut lines = vec![Vec::new()];
    for glyph in chars {
        if glyph.0 == '\n' {
            lines.push(Vec::new());
        } else {
            lines.last_mut().expect("line").push(*glyph);
        }
    }
    lines
}

fn place_lines(
    grid: &mut Grid,
    start: u16,
    rows: u16,
    lines: &[Vec<Glyph>],
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
        .filter(|line| line.iter().any(|g| g.0 != ' '))
        .count();
    let third = total.div_ceil(3);
    let mut seen = 0usize;
    for (offset, line) in view.iter().enumerate() {
        if !line.iter().any(|g| g.0 != ' ') {
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

fn cursor_col(text: &[Glyph]) -> u16 {
    let mut last = None;
    for (i, glyph) in text.iter().enumerate() {
        if glyph.0 != ' ' {
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
fn visual_tail(lines: &[Vec<Glyph>], width: usize, limit: usize) -> Vec<Vec<Glyph>> {
    if limit == 0 || lines.is_empty() {
        return Vec::new();
    }
    let width = width.max(1);
    let mut picked: Vec<&[Glyph]> = Vec::new();
    let mut rows = 0usize;
    for line in lines.iter().rev() {
        picked.push(line.as_slice());
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

fn visual_row_count(line: &[Glyph], width: usize) -> usize {
    let n = line.len();
    if n == 0 { 1 } else { n.div_ceil(width.max(1)) }
}

fn hard_wrap_skip(chars: &[Glyph], width: usize, skip_rows: usize) -> Vec<Vec<Glyph>> {
    if chars.is_empty() {
        return if skip_rows == 0 {
            vec![Vec::new()]
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
        out.push(chars[i..end].to_vec());
        i = end;
    }
    out
}

fn paint_line(grid: &mut Grid, row: u16, chars: &[Glyph], fg: C16) {
    let Some(last) = chars.iter().rposition(|g| g.0 != ' ') else {
        return;
    };
    for (i, glyph) in chars.iter().enumerate().take(last + 1) {
        paint_glyph(grid, 2 + i, usize::from(row), *glyph, fg, C16::Black);
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
    let chars = one_line_glyphs(text);
    let end_u = usize::from(end);
    let len = chars.len().min(end_u.saturating_add(1));
    let start = end_u.saturating_add(1).saturating_sub(len);
    let skip = chars.len().saturating_sub(len);
    for (i, glyph) in chars.iter().skip(skip).enumerate() {
        paint_glyph(grid, start + i, usize::from(row), *glyph, fg, bg);
    }
}

fn center_in(grid: &mut Grid, left: u16, right: u16, row: u16, text: &str, fg: C16, bg: C16) {
    let width = usize::from(right.saturating_sub(left).saturating_add(1));
    let chars = one_line_glyphs(text);
    let len = chars.len().min(width);
    let start = usize::from(left) + width.saturating_sub(len) / 2;
    for (i, glyph) in chars.iter().take(len).enumerate() {
        paint_glyph(grid, start + i, usize::from(row), *glyph, fg, bg);
    }
}

fn fill_span(grid: &mut Grid, left: u16, right: u16, row: u16, ch: char, fg: C16, bg: C16) {
    for col in left..=right {
        paint(grid, col, row, ch, fg, bg);
    }
}

fn one_line(text: &str) -> String {
    one_line_glyphs(text)
        .into_iter()
        .map(|(ch, _)| ch)
        .collect()
}

/// [`one_line`] keeping each cell's `rich` scalar: `(ch, rich)` pairs (#107).
fn one_line_glyphs(text: &str) -> Vec<Glyph> {
    let mut cells = Vec::new();
    sanitize(text, &mut cells);
    cells
        .into_iter()
        .take_while(|cell| !cell.is_line_end())
        .map(|cell| (cell.ch, cell.rich))
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
        paint_glyph(
            grid,
            origin + drawn,
            usize::from(row),
            (cell.ch, cell.rich),
            fg,
            bg,
        );
    }
}

fn paint_at(grid: &mut Grid, col: usize, row: usize, ch: char, fg: C16, bg: C16) {
    paint_glyph(grid, col, row, (ch, ch), fg, bg);
}

/// [`paint_at`] for a `(ch, rich)` pair (#107).
fn paint_glyph(grid: &mut Grid, col: usize, row: usize, glyph: Glyph, fg: C16, bg: C16) {
    let Ok(col) = u16::try_from(col) else {
        return;
    };
    let Ok(row) = u16::try_from(row) else {
        return;
    };
    paint_rich(grid, col, row, glyph, fg, bg);
}

fn paint(grid: &mut Grid, col: u16, row: u16, ch: char, fg: C16, bg: C16) {
    paint_rich(grid, col, row, (ch, ch), fg, bg);
}

fn paint_rich(grid: &mut Grid, col: u16, row: u16, (ch, rich): Glyph, fg: C16, bg: C16) {
    if row >= grid.rows().saturating_sub(1) || col >= grid.cols().saturating_sub(1) {
        return;
    }
    grid.put(col, row, Cell::with_rich(ch, rich, fg, bg));
    grid.clear_exact(col, row);
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

/// Elapsed or remaining time for a job: `43 min`, `4 h 16 min`, `12 s`.
pub(crate) fn job_duration(secs: u64) -> String {
    let secs = secs.min(crate::jobs::MAX_DURATION_S);
    if secs < 60 {
        return format!("{secs} s");
    }
    let mins = secs.saturating_add(30) / 60;
    let hours = mins / 60;
    let mins = mins % 60;
    if hours > 0 {
        if mins == 0 {
            format!("{hours} h")
        } else {
            format!("{hours} h {mins} min")
        }
    } else {
        format!("{mins} min")
    }
}

/// Scale applied to [`job_digits`] so the caption carries k / M / `x1e-N`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DigitScale {
    One,
    Kilo,
    Mega,
    Exp(i32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DigitFmt {
    pub text: String,
    pub scale: DigitScale,
}

/// Big-digit text: at most four characters (`16.6`, `0.25`, `1540`, `9999`).
/// Never contains `e`. A value of 10,000 or more uses a k/M scale in the
/// caption; above the M range, and under 0.01, the caption is `x1eN`.
/// A magnitude so small the scaled text would be empty or non-finite is `0`.
#[allow(dead_code)]
pub(crate) fn job_digits(value: f64) -> String {
    job_digit_fmt(value).text
}

/// Largest magnitude Mega can show as four digits (`9999` × 1e6).
const MEGA_MAX: f64 = 9_999.5 * 1_000_000.0;

pub(crate) fn job_digit_fmt(value: f64) -> DigitFmt {
    if !value.is_finite() {
        return DigitFmt {
            text: "--".to_owned(),
            scale: DigitScale::One,
        };
    }
    let sign = if value < 0.0 { "-" } else { "" };
    let abs = value.abs();
    if abs == 0.0 {
        return zero_digits();
    }
    if (1_000_000.0..MEGA_MAX).contains(&abs) {
        return scaled_digits(sign, four_char(abs / 1_000_000.0), DigitScale::Mega);
    }
    if (9999.5..1_000_000.0).contains(&abs) {
        return scaled_digits(sign, four_char(abs / 1_000.0), DigitScale::Kilo);
    }
    if (0.01..9999.5).contains(&abs) {
        return scaled_digits(sign, four_char(abs), DigitScale::One);
    }
    exp_digits(sign, abs)
}

fn zero_digits() -> DigitFmt {
    DigitFmt {
        text: "0".to_owned(),
        scale: DigitScale::One,
    }
}

fn scaled_digits(sign: &str, body: String, scale: DigitScale) -> DigitFmt {
    if !digit_body_ok(&body) {
        return zero_digits();
    }
    DigitFmt {
        text: format!("{sign}{body}"),
        scale,
    }
}

fn digit_body_ok(body: &str) -> bool {
    !body.is_empty()
        && body.chars().count() <= 4
        && body.chars().all(|c| c.is_ascii_digit() || c == '.')
}

fn exp_digits(sign: &str, abs: f64) -> DigitFmt {
    let log = abs.log10();
    if !log.is_finite() {
        return zero_digits();
    }
    let mut exp = log.floor() as i32;
    let Some(mut mantissa) = mantissa_at(abs, exp) else {
        return zero_digits();
    };
    if mantissa >= 9.995 {
        exp = exp.saturating_add(1);
        match mantissa_at(abs, exp) {
            Some(next) => mantissa = next,
            None => return zero_digits(),
        }
    }
    let body = format!("{mantissa:.2}");
    if !digit_body_ok(&body) {
        return zero_digits();
    }
    DigitFmt {
        text: format!("{sign}{body}"),
        scale: DigitScale::Exp(exp),
    }
}

fn mantissa_at(abs: f64, exp: i32) -> Option<f64> {
    let pow = 10f64.powi(exp);
    if !pow.is_finite() || pow == 0.0 {
        return None;
    }
    let m = abs / pow;
    m.is_finite().then_some(m)
}

/// At most four characters, no exponent. `abs` is in a range that fits.
fn four_char(abs: f64) -> String {
    if abs >= 100.0 {
        format!("{:.0}", abs)
    } else if abs >= 10.0 {
        let text = format!("{abs:.1}");
        if text.len() > 4 {
            format!("{:.0}", abs.round())
        } else {
            text
        }
    } else if abs >= 1.0 {
        format!("{abs:.1}")
    } else {
        format!("{abs:.2}")
    }
}

fn scale_caption(base: &str, scale: DigitScale) -> String {
    match scale {
        DigitScale::One => base.to_owned(),
        DigitScale::Kilo => format!("K {base}"),
        DigitScale::Mega => format!("M {base}"),
        DigitScale::Exp(exp) => format!("{base} x1e{exp}"),
    }
}

/// Compact number for the VALUES block (`1.28e-5`, `0.8465`, `0.96`).
pub(crate) fn job_compact(value: f64) -> String {
    if !value.is_finite() {
        return "--".to_owned();
    }
    let abs = value.abs();
    if abs != 0.0 && (abs < 1e-3 || abs >= 1e5) {
        let sign = if value < 0.0 { "-" } else { "" };
        return format!("{sign}{}", job_sci(abs));
    }
    let mut text = format!("{value:.4}");
    if text.contains('.') {
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
    }
    if text == "-0" { "0".to_owned() } else { text }
}

fn job_sci(abs: f64) -> String {
    let text = format!("{abs:.2e}");
    text.replace("e-0", "e-")
        .replace("e+0", "e")
        .replace("e+", "e")
}

/// Caption beside the rate digits: `s/step` becomes `SECONDS / STEP`.
pub(crate) fn job_unit_caption(unit: &str) -> String {
    if unit.eq_ignore_ascii_case("s/step") {
        "SECONDS / STEP".to_owned()
    } else {
        unit.to_ascii_uppercase()
    }
}

fn draw_job(grid: &mut Grid, model: &TtyModel, g: &Geom) {
    let Some(view) = &model.job else {
        return;
    };
    let job = &view.job;
    let health_rule = g.rows.saturating_sub(3);
    draw_job_title(grid, view, g);
    let mut right_y = 4u16;
    if g.rows >= MIN_ROWS {
        draw_job_progress(grid, job, g, right_y, model.chart_glyphs);
        right_y = right_y.saturating_add(1).saturating_add(g.bar_rows);
    }
    let digit_h: u16 = if g.tall { 10 } else { 5 };
    let digits_span = 1 + digit_h;
    let mut values_y = None;
    if right_y.saturating_add(digits_span) < health_rule.saturating_sub(4) {
        draw_job_digits(grid, job, g, right_y);
        values_y = Some(right_y.saturating_add(digits_span).saturating_add(1));
    }
    let setup_top = g
        .activity_row()
        .saturating_add(g.bar_rows)
        .saturating_add(1);
    let mut mid = g.meter_last.max(g.slot_last).saturating_add(2);
    if setup_top < health_rule.saturating_sub(5) {
        let used = draw_job_setup(grid, view, setup_top, usize::from(g.left_end()));
        mid = mid.max(setup_top.saturating_add(used));
    }
    let values_h = job_values_rows(job, g);
    let without_values = mid;
    let with_values = values_y.map_or(mid, |top| mid.max(top.saturating_add(values_h)));
    let (fans, temps) = (model.fans.as_ref(), model.temps.as_ref());
    let side = g.cols >= FANS_SIDE_COLS && (fans.is_some() || temps.is_some());
    let chart_min: u16 = if job.series.is_empty() { 0 } else { 5 };
    let lower_with = health_rule.saturating_sub(with_values.saturating_add(1));
    let keep_values =
        values_h > 0 && values_y.is_some() && lower_with >= chart_min.saturating_add(2);
    if keep_values && let Some(top) = values_y {
        let used = draw_job_values(grid, job, g, top);
        mid = mid.max(top.saturating_add(used));
    } else {
        mid = without_values;
    }
    if mid + 1 >= health_rule {
        return;
    }
    draw_rule(grid, mid, g.cols);
    let mut y = mid + 1;
    let spare_for_strip = health_rule.saturating_sub(y.saturating_add(6));
    let strip = if side {
        None
    } else {
        strip_plan(fans, temps, spare_for_strip)
    };
    let reserve = match strip {
        Some(Strip::Block(height)) => 1 + height,
        Some(Strip::Line) => 1,
        None => 0,
    };
    let body_end = health_rule.saturating_sub(reserve);
    let content_right = if side {
        fans_split(g.cols).saturating_sub(2)
    } else {
        g.cols.saturating_sub(3)
    };
    let body = body_end.saturating_sub(y);
    let phase_n = u16::try_from(job.phases.len()).unwrap_or(0);
    let phase_need = if phase_n == 0 { 0 } else { 1 + phase_n + 1 };
    // PHASES stays when charts still get at least five rows after it.
    // Charts outrank PHASES only when both will not fit.
    let keep_phases = phase_need > 0 && body >= phase_need.saturating_add(chart_min);
    if keep_phases {
        let take = phase_n.min(body_end.saturating_sub(y + 1));
        if take > 0 {
            draw_job_phases(grid, job, g, y, take, content_right);
            y = y.saturating_add(1).saturating_add(take).saturating_add(1);
            if y < body_end {
                draw_rule(grid, y.saturating_sub(1), g.cols);
            }
        }
    }
    let rest = body_end.saturating_sub(y);
    if rest >= 5 && chart_min > 0 {
        draw_job_charts(grid, job, g, y, rest, model.chart_glyphs, content_right);
        y = y.saturating_add(rest);
    }
    let _ = y;
    if side {
        let split = fans_split(g.cols);
        let top = mid + 1;
        for row in top..health_rule {
            paint(grid, split, row, '|', C16::BrightBlack, C16::Black);
        }
        draw_side_column(
            grid,
            fans,
            temps,
            fans_span(g),
            top,
            health_rule.saturating_sub(top),
            model.chart_glyphs,
        );
    } else {
        match strip {
            Some(Strip::Block(height)) => {
                let strip_rule = g.rows - 3 - reserve;
                draw_rule(grid, strip_rule, g.cols);
                draw_strip(
                    grid,
                    g,
                    fans,
                    temps,
                    strip_rule + 1,
                    height,
                    model.chart_glyphs,
                );
            }
            Some(Strip::Line) => {
                if let Some(panel) = temps {
                    draw_temps_line(grid, panel, 2, g.cols - 3, g.rows - 4);
                }
            }
            None => {}
        }
    }
}

fn draw_job_title(grid: &mut Grid, view: &JobView, g: &Geom) {
    let job = &view.job;
    let mut parts: Vec<String> = Vec::new();
    if let Some(kind) = &job.kind {
        parts.push(kind.to_ascii_uppercase());
    }
    let mut who = job.name.clone();
    if view.more > 0 {
        who.push_str(&format!(" +{}", view.more));
    }
    parts.push(who);
    if let Some(title) = &job.title {
        parts.push(title.clone());
    }
    if let Some(phase) = &job.phase {
        let mut text = phase.label.clone();
        if let (Some(index), Some(of)) = (phase.index, phase.of) {
            text.push_str(&format!(", {index} of {of}"));
        }
        parts.push(text);
    }
    let line = parts.join(" \u{00B7} ");
    let x = usize::from(g.right);
    let end = usize::from(g.cols.saturating_sub(3));
    let room = end.saturating_add(1).saturating_sub(x);
    let shown = cut_text(&line, room);
    paint_detail_fg(grid, x, 3, &shown, room, C16::BrightWhite);
}

fn draw_job_progress(
    grid: &mut Grid,
    job: &crate::jobs::Job,
    g: &Geom,
    row: u16,
    glyphs: ChartGlyphs,
) {
    let mut bits: Vec<String> = Vec::new();
    match (job.step, job.total) {
        (Some(step), Some(total)) => {
            bits.push(format!("STEP {} / {}", commas(step), commas(total)))
        }
        (Some(step), None) => bits.push(format!("STEP {}", commas(step))),
        (None, Some(total)) => bits.push(format!("STEP -- / {}", commas(total))),
        (None, None) => {}
    }
    if let (Some(step), Some(total)) = (job.step, job.total)
        && total > 0
    {
        let pct = ((step as f64 / total as f64) * 100.0).round();
        if pct.is_finite() {
            bits.push(format!("{} %", pct as i64));
        }
    }
    if let Some(eta) = job.eta_s {
        bits.push(format!("{} left", job_duration(eta)));
    }
    let line = bits.join("   ");
    let x = usize::from(g.right);
    let end = usize::from(g.cols.saturating_sub(3));
    let room = end.saturating_add(1).saturating_sub(x);
    paint_str(
        grid,
        g.right,
        row,
        &cut_text(&line, room),
        C16::White,
        C16::Black,
    );
    let frac = match (job.step, job.total) {
        (Some(step), Some(total)) if total > 0 => {
            Some((step as f64 / total as f64).clamp(0.0, 1.0))
        }
        _ => None,
    };
    let bar_row = row + 1;
    let width = g.cols.saturating_sub(3).saturating_sub(g.right);
    if width == 0 {
        return;
    }
    draw_h_bar(
        grid,
        HBar {
            x: g.right,
            row: bar_row,
            width,
            frac,
            ink: BarInk::Spectrum {
                hot: frac.is_some_and(|f| f >= 1.0),
                top: color::spectrum::FULL,
            },
            rows: g.bar_rows,
            cells: Some(meter_glyph(glyphs)),
        },
    );
}

fn draw_job_digits(grid: &mut Grid, job: &crate::jobs::Job, g: &Geom, row: u16) {
    let ph = if g.tall { 2u16 } else { 1 };
    let base = if g.tall { 4u16 } else { 2 };
    if let Some(rate) = &job.rate {
        let fmt = job_digit_fmt(rate.value);
        let base_cap = rate
            .unit
            .as_deref()
            .map(job_unit_caption)
            .unwrap_or_else(|| "RATE".to_owned());
        let caption = scale_caption(&base_cap, fmt.scale);
        paint_str(grid, g.right, row, &caption, C16::White, C16::Black);
        let pw = fit_pixel(base, fmt.text.chars().count().max(1), g.half);
        draw_value_digits(grid, g.right, row + 1, &fmt.text, rate.value, pw, ph);
    }
    if let Some(first) = job.values.first() {
        let x = g.prompt_x();
        let fmt = job_digit_fmt(first.value);
        let caption = scale_caption(&first.label.to_ascii_uppercase(), fmt.scale);
        paint_str(grid, x, row, &caption, C16::White, C16::Black);
        let pw = fit_pixel(base, fmt.text.chars().count().max(1), g.half);
        draw_value_digits(grid, x, row + 1, &fmt.text, first.value, pw, ph);
    }
}

fn draw_value_digits(grid: &mut Grid, x: u16, row: u16, text: &str, value: f64, pw: u16, ph: u16) {
    if pw == 0 {
        return;
    }
    let ceiling = value_ceiling(value);
    let step = rate_step(value.abs(), ceiling);
    let (top, body) = digit_colours(step);
    let pos = grid
        .records_exact()
        .then(|| color::spectrum::rate(value.abs(), ceiling));
    let mut cx = x;
    for ch in text.chars() {
        match ch {
            '0'..='9' => {
                paint_digit_glyph(
                    grid,
                    cx,
                    row,
                    DIGITS[(ch as u8 - b'0') as usize],
                    top,
                    body,
                    pos,
                    pw,
                    ph,
                );
                cx += 3 * pw + pw;
            }
            '-' => {
                paint_digit_glyph(grid, cx, row, DIGITS[10], top, body, pos, pw, ph);
                cx += 3 * pw + pw;
            }
            '.' => {
                for dy in 0..ph {
                    paint(grid, cx, row + 4 * ph + dy, '█', body, C16::Black);
                    if let Some(pos) = pos {
                        let rows = 5 * usize::from(ph);
                        let at = 4 * usize::from(ph) + usize::from(dy);
                        grid.set_exact_fg(
                            cx,
                            row + 4 * ph + dy,
                            act_color(color::spectrum::shade(pos, at, rows)),
                        );
                    }
                }
                cx += pw + pw / 2 + 1;
            }
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_digit_glyph(
    grid: &mut Grid,
    cx: u16,
    row: u16,
    glyph: &[&str],
    top: C16,
    body: C16,
    pos: Option<f32>,
    pw: u16,
    ph: u16,
) {
    for (py, bits) in glyph.iter().enumerate() {
        let colour = if py < 2 { top } else { body };
        for (px, bit) in bits.chars().enumerate() {
            if bit != '#' {
                continue;
            }
            for dy in 0..ph {
                for dx in 0..pw {
                    let y = row + py as u16 * ph + dy;
                    paint(grid, cx + px as u16 * pw + dx, y, '█', colour, C16::Black);
                    if let Some(pos) = pos {
                        let rows = 5 * usize::from(ph);
                        let at = py * usize::from(ph) + usize::from(dy);
                        grid.set_exact_fg(
                            cx + px as u16 * pw + dx,
                            y,
                            act_color(color::spectrum::shade(pos, at, rows)),
                        );
                    }
                }
            }
        }
    }
}

fn value_ceiling(value: f64) -> f64 {
    let abs = value.abs();
    if abs <= 0.0 {
        return 1.0;
    }
    let exp = abs.log10().ceil();
    if !exp.is_finite() {
        return abs.max(1.0);
    }
    10_f64.powf(exp).max(1.0)
}

fn draw_job_setup(grid: &mut Grid, view: &JobView, top: u16, right: usize) -> u16 {
    let job = &view.job;
    paint_str(grid, 2, top, "JOB", C16::White, C16::Black);
    let state = job.state.as_str();
    let fg = match job.state {
        crate::jobs::JobState::Running => C16::BrightRed,
        crate::jobs::JobState::Done => C16::Green,
        crate::jobs::JobState::Failed => C16::Yellow,
    };
    paint_str(grid, 8, top, state, fg, C16::Black);
    let mut used = 1u16;
    if let Some(elapsed) = job.elapsed_s {
        paint_detail(grid, 4, usize::from(top + 1), "elapsed", 8);
        let text = job_duration(elapsed);
        paint_detail_fg(
            grid,
            13,
            usize::from(top + 1),
            &text,
            right.saturating_sub(12),
            C16::BrightWhite,
        );
        used += 1;
    }
    used
}

fn job_values_rows(job: &crate::jobs::Job, g: &Geom) -> u16 {
    let n = job.rates.len() + job.values.len();
    if n == 0 {
        return 0;
    }
    let width = usize::from(g.cols.saturating_sub(3).saturating_sub(g.right));
    let cols = (width / 18).max(1);
    let rows = n.div_ceil(cols);
    1 + u16::try_from(rows).unwrap_or(u16::MAX)
}

fn draw_job_values(grid: &mut Grid, job: &crate::jobs::Job, g: &Geom, top: u16) -> u16 {
    paint_str(grid, g.right, top, "VALUES", C16::White, C16::Black);
    let mut items: Vec<String> = Vec::new();
    for n in job.rates.iter().chain(job.values.iter()) {
        items.push(format!("{} {}", n.label, job_compact(n.value)));
    }
    if items.is_empty() {
        return 1;
    }
    let width = usize::from(g.cols.saturating_sub(3).saturating_sub(g.right));
    let col_w = 18usize;
    let cols = (width / col_w).max(1);
    let mut row = 1u16;
    let mut col = 0usize;
    for item in items {
        let x = usize::from(g.right) + col * col_w;
        paint_detail_fg(
            grid,
            x,
            usize::from(top + row),
            &item,
            col_w.saturating_sub(1),
            C16::BrightWhite,
        );
        col += 1;
        if col >= cols {
            col = 0;
            row += 1;
        }
    }
    if col == 0 { row } else { row + 1 }
}

fn draw_job_phases(
    grid: &mut Grid,
    job: &crate::jobs::Job,
    g: &Geom,
    header: u16,
    take: u16,
    right: u16,
) {
    paint_str(grid, 2, header, "PHASES", C16::White, C16::Black);
    let take_n = usize::from(take).min(job.phases.len());
    for (i, phase) in job.phases.iter().take(take_n).enumerate() {
        let row = header + 1 + i as u16;
        let (mark, fg) = match phase.mark {
            Some(crate::jobs::PhaseMark::Done) => ('•', C16::Green),
            Some(crate::jobs::PhaseMark::Running) => ('>', C16::BrightRed),
            Some(crate::jobs::PhaseMark::Queued) => ('·', C16::BrightBlack),
            Some(crate::jobs::PhaseMark::Failed) => ('x', C16::Yellow),
            None => ('·', C16::BrightBlack),
        };
        paint(grid, 2, row, mark, fg, C16::Black);
        let mut x = 4usize;
        let label = cut_text(&phase.label, 16);
        x += paint_detail_fg(grid, x, usize::from(row), &label, 16, C16::BrightWhite);
        x = x.max(22);
        if let Some(elapsed) = phase.elapsed_s {
            let text = job_duration(elapsed);
            x += paint_detail_fg(grid, x, usize::from(row), &text, 16, C16::BrightBlack);
            x += 2;
        }
        if let Some(result) = &phase.result {
            let room = usize::from(right).saturating_sub(x);
            paint_detail_fg(grid, x, usize::from(row), result, room, C16::White);
        }
        let _ = g;
    }
}

/// Narrowest chart when pairing two series side by side.
const JOB_CHART_PAIR_W: u16 = 60;
const JOB_CHART_H_MIN: u16 = 5;

fn draw_job_charts(
    grid: &mut Grid,
    job: &crate::jobs::Job,
    g: &Geom,
    start: u16,
    height: u16,
    glyphs: ChartGlyphs,
    right: u16,
) {
    let series: Vec<&crate::jobs::JobSeries> = job.series.iter().take(4).collect();
    if series.is_empty() || height < JOB_CHART_H_MIN {
        return;
    }
    let left = 2u16;
    let width = right.saturating_sub(left).saturating_add(1);
    // Two series sit side by side when the frame is wide enough. Three or
    // four series use two rows of pairs when each chart still gets 5 rows.
    let pair = series.len() >= 2 && width >= JOB_CHART_PAIR_W * 2;
    let rows_n = if pair {
        series.len().div_ceil(2)
    } else {
        series.len()
    };
    let heights = split_chart_heights(u16::try_from(rows_n).unwrap_or(1), height);
    if heights.is_empty() {
        return;
    }
    let mut y = start;
    if pair {
        let gap = 2u16;
        let inner = width.saturating_sub(gap);
        let each_w = inner / 2;
        let mid = left + each_w;
        for (row_i, h) in heights.iter().enumerate() {
            let a = row_i * 2;
            if a >= series.len() {
                break;
            }
            draw_one_job_chart(
                grid,
                series[a],
                g,
                y,
                *h,
                glyphs,
                left,
                mid.saturating_sub(1),
            );
            if a + 1 < series.len() {
                draw_one_job_chart(grid, series[a + 1], g, y, *h, glyphs, mid + gap, right);
            }
            y = y.saturating_add(*h).saturating_add(1);
        }
    } else {
        for (i, h) in heights.iter().enumerate() {
            if i >= series.len() {
                break;
            }
            draw_one_job_chart(grid, series[i], g, y, *h, glyphs, left, right);
            y = y.saturating_add(*h).saturating_add(1);
        }
    }
}

fn split_chart_heights(count: u16, rows: u16) -> Vec<u16> {
    if count == 0 || rows < JOB_CHART_H_MIN {
        return Vec::new();
    }
    let mut fit = 0u16;
    while fit < count {
        let need = (fit + 1) * JOB_CHART_H_MIN + fit;
        if need > rows {
            break;
        }
        fit += 1;
    }
    if fit == 0 {
        return Vec::new();
    }
    let gaps = fit.saturating_sub(1);
    let usable = rows.saturating_sub(gaps);
    let base = (usable / fit).max(JOB_CHART_H_MIN);
    let mut heights = vec![base; usize::from(fit)];
    let mut leftover = usable.saturating_sub(base * fit);
    for h in &mut heights {
        if leftover == 0 {
            break;
        }
        *h += 1;
        leftover -= 1;
    }
    heights
}

#[allow(clippy::too_many_arguments)]
fn draw_one_job_chart(
    grid: &mut Grid,
    series: &crate::jobs::JobSeries,
    g: &Geom,
    start: u16,
    height: u16,
    glyphs: ChartGlyphs,
    left: u16,
    right: u16,
) {
    if height < 3 {
        return;
    }
    let _ = g;
    let last = series.points.last().copied();
    let mut head = series.label.clone();
    if let Some(v) = last {
        head.push(' ');
        head.push_str(&job_compact(v));
    }
    paint_str(grid, left, start, &head, C16::White, C16::Black);
    let (min, max) = series_bounds(&series.points);
    right_align(
        grid,
        right,
        start,
        &job_compact(max),
        C16::BrightBlack,
        C16::Black,
    );
    right_align(
        grid,
        right,
        start + height - 1,
        &job_compact(min),
        C16::BrightBlack,
        C16::Black,
    );
    let data_top = start + 1;
    let data_h = height.saturating_sub(2);
    if data_h == 0 {
        return;
    }
    let data_left = left;
    let data_right = right.saturating_sub(8);
    if data_right <= data_left {
        return;
    }
    let width = usize::from(data_right - data_left + 1);
    let span = (max - min).abs().max(1e-12);
    for col in 0..width {
        let Some(v) = column_sample(&series.points, col, width) else {
            continue;
        };
        let frac = if (max - min).abs() < 1e-12 {
            0.5
        } else {
            ((v - min) / span).clamp(0.0, 1.0)
        };
        let colour = digit_colours(rate_step(frac * 5.0 + 0.01, 5.0)).1;
        let cells = match glyphs {
            ChartGlyphs::Halves => {
                let level = crate::tty::chart::halves(frac, data_h).max(1);
                half_inks(crate::tty::chart::rise_glyphs(level, data_h))
            }
            ChartGlyphs::Eighths => {
                let level = crate::tty::chart::eighths(frac, data_h).max(1);
                crate::tty::chart::rise_eighths(level, data_h)
            }
        };
        let rows_n = usize::from(data_h);
        for (i, ink) in cells.into_iter().enumerate() {
            let from_axis = rows_n.saturating_sub(1).saturating_sub(i);
            paint_ink(
                grid,
                data_left + col as u16,
                data_top + i as u16,
                ink,
                colour,
                chart_exact(grid, ink, from_axis, rows_n),
            );
        }
    }
}

fn series_bounds(points: &[f64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in points {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
    }
    if !min.is_finite() {
        (0.0, 1.0)
    } else {
        (min, max)
    }
}

/// Value drawn in one chart column. A long series takes the column's
/// maximum so a loss spike is not skipped.
fn column_sample(points: &[f64], col: usize, width: usize) -> Option<f64> {
    let n = points.len();
    if n == 0 || width == 0 {
        return None;
    }
    if n == 1 {
        return points[0].is_finite().then_some(points[0]);
    }
    if n > width {
        let start = col * n / width;
        let end = ((col + 1) * n / width).max(start + 1).min(n);
        points[start..end]
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .max_by(|a, b| a.total_cmp(b))
    } else {
        let idx = if width <= 1 {
            0
        } else {
            (col * (n - 1) + (width - 1) / 2) / (width - 1)
        };
        let v = points[idx.min(n - 1)];
        v.is_finite().then_some(v)
    }
}

/// Header one-line note for a live or finished job.
pub(crate) fn job_note_line(job: &crate::jobs::Job, more: usize, live: bool) -> String {
    let mut out = job.name.clone();
    if more > 0 {
        out.push_str(&format!(" +{more}"));
    }
    if live {
        match (job.step, job.total) {
            (Some(step), Some(total)) => {
                out.push(' ');
                out.push_str(&comma_pair(step, total));
            }
            (Some(step), None) => {
                out.push(' ');
                out.push_str(&commas(step));
            }
            _ => {}
        }
        if let Some(eta) = job.eta_s {
            out.push(' ');
            out.push_str(&job_duration(eta));
            out.push_str(" left");
        }
    } else {
        out.push(' ');
        out.push_str(job.state.as_str());
        if let Some(elapsed) = job.elapsed_s {
            out.push(' ');
            out.push_str(&job_duration(elapsed));
        }
    }
    out
}

#[cfg(test)]
mod tests {

    #[test]
    fn job_digits_fit_four_characters() {
        assert_eq!(job_digits(16.62), "16.6");
        assert_eq!(job_digits(0.2544), "0.25");
        assert_eq!(job_digits(1540.0), "1540");
        assert_eq!(job_digits(3.285), "3.3");
        assert_eq!(job_compact(1.276e-5), "1.28e-5");
        assert_eq!(job_compact(0.8465), "0.8465");
        assert_eq!(job_compact(0.96), "0.96");
        assert_eq!(job_duration(2559), "43 min");
        assert_eq!(job_duration(15_381), "4 h 16 min");
        assert_eq!(job_duration(10_186), "2 h 50 min");
        assert_eq!(job_duration(12), "12 s");
        assert_eq!(
            job_duration(u64::MAX),
            job_duration(crate::jobs::MAX_DURATION_S)
        );
        assert_ne!(job_duration(u64::MAX), "0 min");
    }

    fn paint_digits(value: f64) -> (DigitFmt, usize) {
        let fmt = job_digit_fmt(value);
        let body = fmt.text.trim_start_matches('-');
        assert!(
            fmt.text
                .chars()
                .all(|c| c.is_ascii_digit() || c == '-' || c == '.'),
            "digits contain a non-paintable char: {}",
            fmt.text
        );
        assert!(
            !fmt.text.contains('e') && !fmt.text.contains('E'),
            "{}",
            fmt.text
        );
        assert!(body.chars().count() <= 4, "{}", fmt.text);
        let mut grid = Grid::new(80, 12);
        draw_value_digits(&mut grid, 0, 0, &fmt.text, value, 1, 1);
        let ink = grid.cells().iter().filter(|cell| cell.ch == '█').count();
        assert!(ink > 0, "painter drew nothing for {value} ({})", fmt.text);
        assert!(
            !grid
                .cells()
                .iter()
                .any(|cell| cell.ch == 'e' || cell.ch == 'E'),
            "painter kept an e for {value}"
        );
        (fmt, ink)
    }

    #[test]
    fn job_digits_through_the_painter() {
        let (fmt, _) = paint_digits(12345.0);
        assert_eq!(fmt.text, "12.3");
        assert_eq!(fmt.scale, DigitScale::Kilo);
        let (fmt, _) = paint_digits(0.005);
        assert_eq!(fmt.text, "5.00");
        assert_eq!(fmt.scale, DigitScale::Exp(-3));
        let (fmt, _) = paint_digits(1.276e-5);
        assert_eq!(fmt.text, "1.28");
        assert_eq!(fmt.scale, DigitScale::Exp(-5));
        let (fmt, _) = paint_digits(99.96);
        assert_eq!(fmt.text, "100");
        assert_eq!(fmt.scale, DigitScale::One);
        let (fmt, _) = paint_digits(-16.62);
        assert_eq!(fmt.text, "-16.6");
        let (fmt, _) = paint_digits(0.0);
        assert_eq!(fmt.text, "0");
        assert_eq!(scale_caption("ROWS / S", DigitScale::Kilo), "K ROWS / S");
        assert_eq!(scale_caption("LR", DigitScale::Exp(-5)), "LR x1e-5");

        let (fmt, _) = paint_digits(9.9994e9);
        assert_eq!(fmt.text, "9999");
        assert_eq!(fmt.scale, DigitScale::Mega);
        assert_shown_is_true(9.9994e9, &fmt);
        let (fmt, _) = paint_digits(1e10);
        assert_eq!(fmt.text, "1.00");
        assert_eq!(fmt.scale, DigitScale::Exp(10));
        assert_shown_is_true(1e10, &fmt);
        let (fmt, _) = paint_digits(1e12);
        assert_eq!(fmt.text, "1.00");
        assert_eq!(fmt.scale, DigitScale::Exp(12));
        assert_shown_is_true(1e12, &fmt);
        let (fmt, _) = paint_digits(f64::MAX);
        assert_eq!(fmt.text, "1.80");
        assert_eq!(fmt.scale, DigitScale::Exp(308));
        assert_shown_is_true(f64::MAX, &fmt);
        let (fmt, _) = paint_digits(-9.9994e9);
        assert_eq!(fmt.text, "-9999");
        assert_eq!(fmt.scale, DigitScale::Mega);
        assert_shown_is_true(-9.9994e9, &fmt);
        let (fmt, _) = paint_digits(-1e10);
        assert_eq!(fmt.text, "-1.00");
        assert_eq!(fmt.scale, DigitScale::Exp(10));
        let (fmt, _) = paint_digits(-1e12);
        assert_eq!(fmt.text, "-1.00");
        assert_eq!(fmt.scale, DigitScale::Exp(12));
        let (fmt, _) = paint_digits(-f64::MAX);
        assert_eq!(fmt.text, "-1.80");
        assert_eq!(fmt.scale, DigitScale::Exp(308));
        let (fmt, _) = paint_digits(5e-310);
        assert_eq!(fmt.text, "0");
        assert_eq!(fmt.scale, DigitScale::One);
    }

    fn assert_shown_is_true(value: f64, fmt: &DigitFmt) {
        let n: f64 = fmt.text.parse().expect("digit text");
        let got = match fmt.scale {
            DigitScale::One => n.abs().log10(),
            DigitScale::Kilo => n.abs().log10() + 3.0,
            DigitScale::Mega => n.abs().log10() + 6.0,
            DigitScale::Exp(exp) => n.abs().log10() + f64::from(exp),
        };
        let want = value.abs().log10();
        assert!(
            (got - want).abs() < 0.01,
            "digits+caption {} {:?} are not the true value {value} (log10 {got} vs {want})",
            fmt.text,
            fmt.scale
        );
        assert_eq!(fmt.text.starts_with('-'), value < 0.0);
    }

    #[test]
    fn job_chart_heights_share_every_row() {
        assert_eq!(split_chart_heights(1, 24), vec![24]);
        assert_eq!(split_chart_heights(2, 24), vec![12, 11]);
        assert_eq!(split_chart_heights(2, 11), vec![5, 5]);
        assert_eq!(split_chart_heights(2, 10), vec![10]);
        assert!(split_chart_heights(1, 24)[0] > 16);
    }

    #[test]
    fn a_constant_series_prints_its_real_bounds() {
        assert_eq!(series_bounds(&[5.0, 5.0, 5.0]), (5.0, 5.0));
        assert_eq!(series_bounds(&[5.0]), (5.0, 5.0));
        assert_eq!(series_bounds(&[1.0, 3.0, 2.0]), (1.0, 3.0));
    }

    #[test]
    fn a_long_series_takes_each_column_maximum() {
        let mut points = vec![1.0; 20];
        points[3] = 9.0;
        let v = column_sample(&points, 0, 4).unwrap();
        assert_eq!(v, 9.0, "the spike in the first quarter is kept");
        assert_eq!(column_sample(&[2.0, 4.0, 3.0], 0, 10), Some(2.0));
    }

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

    use llama_core::color::{self, act_color};
    use llama_core::palette::Palette;

    /// A spectrum bar at `pct` percent of full scale, inked like a level meter.
    fn spectrum_bar(grid: &mut Grid, width: u16, rows: u16, pct: f64) {
        let frac = (pct / 100.0).min(1.0);
        draw_h_bar(
            grid,
            HBar {
                x: 0,
                row: 0,
                width,
                frac: Some(frac),
                ink: level_ink(Some(pct)),
                rows,
                cells: Some('█'),
            },
        );
    }

    /// Largest channel difference between two colours.
    fn gap(a: llama_core::color::Rgb, b: llama_core::color::Rgb) -> u8 {
        a.r.abs_diff(b.r)
            .max(a.g.abs_diff(b.g))
            .max(a.b.abs_diff(b.b))
    }

    #[test]
    fn a_spectrum_bar_is_one_ramp_from_the_first_stop_to_the_last() {
        let first = color::ACT_STOPS[0].1;
        let last = color::ACT_STOPS[color::ACT_STOPS.len() - 1].1;
        // Width 2 and 5 are short enough that centre sampling misses both
        // stops. The ends are the stops at every width.
        for width in [2u16, 5, 20, 67] {
            let mut grid = Grid::with_exact(width + 2, 4);
            spectrum_bar(&mut grid, width, 2, 100.0);
            assert_eq!(grid.exact_fg(0, 0), Some(first), "width {width} start");
            assert_eq!(grid.exact_fg(width - 1, 0), Some(last), "width {width} end");
            for i in 0..width {
                assert_eq!(grid.exact_fg(i, 1), grid.exact_fg(i, 0), "row {i}");
                assert_eq!(grid.get(i, 1).unwrap().fg, grid.get(i, 0).unwrap().fg);
            }
        }
        // A wide bar has no seam. The console still paints its five steps.
        let width = 67u16;
        let mut wide = Grid::with_exact(width + 2, 3);
        spectrum_bar(&mut wide, width, 1, 100.0);
        let mut steps = vec![wide.get(0, 0).unwrap().fg];
        for i in 1..width {
            let here = wide.exact_fg(i, 0).expect("exact");
            let before = wide.exact_fg(i - 1, 0).expect("exact");
            assert!(gap(here, before) <= 16, "a seam at cell {i}");
            let fg = wide.get(i, 0).unwrap().fg;
            if steps.last() != Some(&fg) {
                steps.push(fg);
            }
        }
        assert_eq!(steps.len(), 5, "five spectrum steps on the console");
    }

    #[test]
    fn a_hot_bar_stretches_into_the_blackbody_tail_without_a_seam() {
        let mut grid = Grid::with_exact(8, 4);
        spectrum_bar(&mut grid, 5, 2, 125.0);
        // The console is as before: the last step white-hot over red.
        assert_eq!(grid.get(4, 0).unwrap().fg, C16::BrightWhite);
        assert_eq!(grid.get(4, 1).unwrap().fg, C16::BrightRed);
        // A five-cell bar at the end of the tail: first stop, last stop.
        assert_eq!(grid.exact_fg(0, 0), Some(color::ACT_STOPS[0].1));
        assert_eq!(
            grid.exact_fg(4, 0),
            Some(color::BB_STOPS[color::BB_STOPS.len() - 1].1)
        );
        for i in 0..5u16 {
            let pos = color::spectrum::place_to(usize::from(i), 5, 125.0);
            assert_eq!(grid.exact_fg(i, 0), Some(act_color(pos)));
            assert_eq!(grid.exact_fg(i, 1), Some(act_color(pos)));
        }
        assert!(color::spectrum::place_to(4, 5, 125.0) > color::spectrum::FULL);

        let width = 67u16;
        let bar = |pct: f64| {
            let mut grid = Grid::with_exact(width + 2, 3);
            spectrum_bar(&mut grid, width, 1, pct);
            (0..width)
                .map(|i| grid.exact_fg(i, 0).expect("exact"))
                .collect::<Vec<_>>()
        };
        for pct in [101.0, 110.0, 118.0, 125.0, 300.0] {
            let cells = bar(pct);
            for pair in cells.windows(2) {
                assert!(gap(pair[0], pair[1]) <= 16, "{pct} %: a seam");
            }
        }
        // Crossing 100 % moves no cell by more than a shade.
        let full = bar(100.0);
        let above = bar(100.5);
        for (a, b) in full.iter().zip(&above) {
            assert!(gap(*a, *b) <= 4);
        }
        // Past the tail's end the bar stops changing.
        assert_eq!(bar(125.0), bar(300.0));
        assert_ne!(bar(125.0), full);
    }

    #[test]
    fn a_rate_bar_past_its_ceiling_stretches_like_its_digits() {
        let ceiling = 250.0;
        let tail = color::BB_STOPS[color::BB_STOPS.len() - 1].0;
        let mut at_tail = ceiling;
        while color::spectrum::rate(at_tail, ceiling) < tail {
            at_tail *= 2.0;
            assert!(at_tail < ceiling * 64.0, "the tail stays reachable");
        }
        // The stop the digits take at each of the three values.
        let stop_of = |value: f64| {
            if value == ceiling {
                color::ACT_STOPS[color::ACT_STOPS.len() - 1].1
            } else if value == ceiling * 2.0 {
                color::BB_STOPS[3].1
            } else {
                color::BB_STOPS[color::BB_STOPS.len() - 1].1
            }
        };
        // Widths `Geom` gives the rate bars at 192 and 213 columns.
        for cols in [192u16, 213] {
            let g = Geom::new(cols, 60);
            let digit_row = if g.tall { 5 } else { 4 };
            let ph = if g.tall { 2u16 } else { 1 };
            let base = if g.tall { 4u16 } else { 2 };
            let bar_row = digit_row + if g.tall { 10 } else { 5 } + 1;
            assert_eq!(ph, 1, "60 rows draw the short digits");
            let draw = |value: f64| {
                let mut grid = Grid::with_exact(g.cols, 60);
                let pw = fit_pixel(base, digit_len(Some(value)), g.half);
                draw_digits(&mut grid, g.right, digit_row, Some(value), ceiling, pw, ph);
                draw_rate_bar(
                    &mut grid,
                    g.right,
                    bar_row,
                    g.half,
                    rate_frac(Some(value), ceiling),
                    g.bar_rows,
                    rate_top(Some(value), ceiling),
                );
                grid
            };
            let mut full_first = None;
            for value in [ceiling, ceiling * 2.0, at_tail] {
                let grid = draw(value);
                let last = grid
                    .exact_fg(g.right + g.half - 1, bar_row)
                    .expect("last cell of a full scale");
                let mut digit = None;
                // The middle glyph row is the value, unshaded.
                for col in g.right..g.right + g.half {
                    if let Some(colour) = grid.exact_fg(col, digit_row + 2) {
                        digit = Some(colour);
                        break;
                    }
                }
                let digit = digit.expect("digit colour");
                assert_eq!(last, digit, "{cols} cols at {value}");
                assert_eq!(digit, stop_of(value), "{cols} cols at {value}");
                if value == ceiling {
                    full_first = grid.exact_fg(g.right, bar_row);
                }
            }
            // Below the ceiling the lit cells are the full-scale ramp.
            let low = draw(ceiling / 4.0);
            assert_eq!(low.exact_fg(g.right, bar_row), full_first, "{cols}");
        }
        assert_eq!(rate_top(None, ceiling), color::spectrum::FULL);
    }

    #[test]
    fn an_empty_track_and_a_fixed_bar_carry_no_exact_colour() {
        let mut grid = Grid::with_exact(8, 3);
        spectrum_bar(&mut grid, 5, 1, 40.0);
        assert!(grid.exact_fg(0, 0).is_some());
        assert_eq!(grid.get(4, 0).unwrap().ch, '░');
        assert!(grid.exact_fg(4, 0).is_none());
        draw_h_bar(
            &mut grid,
            HBar {
                x: 0,
                row: 0,
                width: 5,
                frac: Some(1.0),
                ink: BarInk::Fixed(C16::Green, C16::Green),
                rows: 1,
                cells: Some('█'),
            },
        );
        assert_eq!(grid.get(0, 0).unwrap().fg, C16::Green);
        assert!(
            grid.exact_fg(0, 0).is_none(),
            "a fixed bar overwrites the ramp"
        );
        let mut plain = Grid::new(8, 3);
        spectrum_bar(&mut plain, 5, 1, 125.0);
        assert!(!plain.records_exact());
        assert!(plain.exact_fg(4, 0).is_none());
    }

    #[test]
    fn digits_shade_around_the_value_and_a_dim_value_stays_plain() {
        let ceiling = 250.0;
        let value = ceiling * 2.0_f64.powf(-2.5);
        let mut grid = Grid::with_exact(24, 8);
        draw_digits(&mut grid, 0, 0, Some(value), ceiling, 1, 1);
        let pos = color::spectrum::rate(value, ceiling);
        // Column 2 of '4' is a lit pixel on every glyph row.
        for row in 0..5u16 {
            let cell = grid.get(2, row).unwrap();
            assert_eq!(cell.ch, '█', "row {row}");
            assert_eq!(cell.fg, C16::Magenta, "row {row}");
            assert_eq!(
                grid.exact_fg(2, row).unwrap(),
                act_color(color::spectrum::shade(pos, usize::from(row), 5))
            );
        }
        // The middle row is the value itself: the middle of the third band,
        // half a band above the step the console paints.
        assert_eq!(grid.exact_fg(2, 2).unwrap(), act_color(pos));
        assert!((pos - 50.0).abs() < 1.0e-3, "{pos}");
        let mut dim = Grid::with_exact(8, 8);
        draw_digits(&mut dim, 0, 0, Some(0.0), ceiling, 1, 1);
        assert_eq!(dim.get(0, 0).unwrap().fg, C16::BrightBlack);
        assert!(dim.exact_fg(0, 0).is_none());
        draw_digits(&mut dim, 0, 0, None, ceiling, 1, 1);
        assert!(dim.exact_fg(0, 0).is_none());
        // NaN is the dim zero, not a live value on the first stop.
        let mut nan = Grid::with_exact(8, 8);
        draw_digits(&mut nan, 0, 0, Some(f64::NAN), ceiling, 1, 1);
        assert_eq!(nan.get(0, 0).unwrap().ch, '█');
        assert_eq!(nan.get(0, 0).unwrap().fg, C16::BrightBlack);
        assert!(nan.exact_fg(0, 0).is_none());
    }

    #[test]
    fn a_chart_column_is_coloured_by_its_row_not_its_value() {
        // `chart_height` draws 2, 3 or 4 rows a side. Five never happens.
        let mut halves = Vec::new();
        for rows in MIN_ROWS..=512 {
            for req in 0..rows {
                let height = chart_height(req, rows);
                if height == 0 {
                    continue;
                }
                let half = (height - 1) / 2;
                if !halves.contains(&half) {
                    halves.push(half);
                }
            }
        }
        halves.sort_unstable();
        assert_eq!(halves, vec![2, 3, 4], "rows a side chart_height produces");
        let first = color::ACT_STOPS[0].1;
        let last = color::ACT_STOPS[color::ACT_STOPS.len() - 1].1;
        let ceiling = 250.0;
        for half in halves {
            for above in [true, false] {
                let mut grid = Grid::with_exact(4, half * 2 + 4);
                let axis = half + 1;
                draw_chart_col(
                    &mut grid,
                    1,
                    axis,
                    half,
                    ChartBucket {
                        gen_tps: above.then_some(ceiling),
                        prompt_tps: (!above).then_some(ceiling),
                    },
                    ceiling,
                    ceiling,
                    ChartGlyphs::Eighths,
                );
                // The row at the axis is the first stop; the far row the last.
                let (near, far) = if above {
                    (axis - 1, axis - half)
                } else {
                    (axis + 1, axis + half)
                };
                assert_eq!(
                    grid.exact_fg(1, near),
                    Some(first),
                    "half {half} above {above}: axis"
                );
                assert_eq!(
                    grid.exact_fg(1, far),
                    Some(last),
                    "half {half} above {above}: far"
                );
                // The console colour is the column's value, on every row.
                assert_eq!(grid.get(1, near).unwrap().fg, C16::BrightRed);
                assert_eq!(grid.get(1, far).unwrap().fg, C16::BrightRed);
                assert_ne!(first, Palette::Llama.rgb(C16::BrightRed.index()));
            }
        }
        // A measured zero is the dim dot, with no ramp. The usual height.
        let half = 4u16;
        let axis = half + 1;
        let mut grid = Grid::with_exact(4, half * 2 + 4);
        draw_chart_col(
            &mut grid,
            1,
            axis,
            half,
            ChartBucket {
                gen_tps: Some(0.0),
                prompt_tps: None,
            },
            ceiling,
            ceiling,
            ChartGlyphs::Eighths,
        );
        assert_eq!(grid.get(1, axis).unwrap().ch, '·');
        assert!(grid.exact_fg(1, axis).is_none());

        // One eighth: the tip is the row at the axis, the first stop.
        let mut value = ceiling / 32.0;
        let mut level = 0u32;
        while value < ceiling {
            level = chart::eighths(rate_frac(Some(value), ceiling).unwrap_or(0.0), half);
            if level == 1 {
                break;
            }
            value *= 1.05;
        }
        assert_eq!(level, 1, "a one-eighth column at {value}");
        draw_chart_col(
            &mut grid,
            1,
            axis,
            half,
            ChartBucket {
                gen_tps: None,
                prompt_tps: Some(value),
            },
            ceiling,
            ceiling,
            ChartGlyphs::Eighths,
        );
        let tip = axis + 1;
        assert_eq!(grid.get(1, tip).unwrap().fg, C16::Black);
        assert_eq!(grid.get(1, tip).unwrap().bg, C16::Blue);
        assert!(grid.exact_fg(1, tip).is_none());
        assert_eq!(grid.exact_bg(1, tip).unwrap(), first);
    }

    #[test]
    fn temperatures_follow_the_threshold_ramp() {
        let marks = Some((800, 900));
        let item = |c: i32, level: Level, thresholds: Option<(i32, i32)>| TempItem {
            label: String::new(),
            tenths: c * 10,
            level,
            thresholds,
        };
        let panel = TempPanel {
            groups: vec![TempGroup {
                name: "CPU".to_owned(),
                items: vec![
                    item(45, Level::Normal, marks),
                    item(80, Level::Warn, marks),
                    item(90, Level::Crit, marks),
                    item(70, Level::Normal, None),
                ],
            }],
        };
        let mut grid = Grid::with_exact(48, 4);
        draw_temps(&mut grid, &panel, 0, 46, 0, 3);
        let row = 1u16;
        // Name takes TEMP_NAME_W, then a gap, then the values.
        let at = |col: u16| grid.exact_fg(col, row);
        let fg = |col: u16| grid.get(col, row).unwrap().fg;
        assert_eq!(grid.get(9, row).unwrap().ch, '4');
        assert_eq!(fg(9), C16::BrightWhite);
        assert_eq!(at(9), Some(color::temp::NORMAL));
        assert_eq!(grid.get(14, row).unwrap().ch, '8');
        assert_eq!(fg(14), C16::Yellow);
        assert_eq!(at(14), Some(color::temp::WARN));
        assert_eq!(at(14).unwrap(), Palette::Llama.rgb(C16::Yellow.index()));
        assert_eq!(grid.get(19, row).unwrap().ch, '9');
        assert_eq!(fg(19), C16::BrightRed);
        assert_eq!(at(19), Some(color::temp::CRIT));
        // Crit in the file is the ramp's 100 stop (#FF2A14). The console
        // stays BrightRed (#F21E5A). That difference is intended.
        assert_eq!(at(19).unwrap(), llama_core::color::hex(0xFF2A14));
        assert_ne!(at(19).unwrap(), Palette::Llama.rgb(C16::BrightRed.index()));
        assert_eq!(grid.get(24, row).unwrap().ch, '7');
        assert_eq!(fg(24), C16::BrightWhite);
        assert!(at(24).is_none(), "no thresholds, no ramp");
    }

    /// `rate_frac` is the console's log fraction, clamped to the five bands.
    /// [`color::spectrum::rate`] is that fraction turned into a ramp
    /// position, and it is not clamped: above the ceiling it climbs into
    /// the blackbody tail. On the scale the two are the same fraction.
    #[test]
    fn the_rate_fraction_matches_the_ramp_across_the_scale() {
        let ceiling = 250.0;
        let frac_of = |value: f64| {
            let pos = color::spectrum::rate(value, ceiling);
            f64::from(pos / color::spectrum::FULL)
        };
        let mut value = ceiling / 32.0;
        while value < ceiling {
            let console = rate_frac(Some(value), ceiling).expect("on the scale");
            let ramp = frac_of(value);
            assert!(
                (ramp - console).abs() < 1e-4,
                "value={value} ramp={ramp} console={console}"
            );
            assert!((0.0..=1.0).contains(&console));
            value *= 1.05;
        }
        let at_ceiling = rate_frac(Some(ceiling), ceiling).expect("ceiling");
        assert!((frac_of(ceiling) - at_ceiling).abs() < 1e-4);
        assert!((at_ceiling - 1.0).abs() < 1e-12);

        // Off the scale they part on purpose. Clamping the ramp to match
        // the console would drop the blackbody tail.
        assert_eq!(rate_frac(Some(ceiling * 2.0), ceiling), Some(1.0));
        assert!(
            color::spectrum::rate(ceiling * 2.0, ceiling) > color::spectrum::rate(ceiling, ceiling)
        );
        assert_eq!(rate_frac(Some(ceiling / 64.0), ceiling), Some(0.0));
        assert!(
            color::spectrum::rate(ceiling / 64.0, ceiling)
                < color::spectrum::rate(ceiling / 32.0, ceiling)
        );
    }

    #[test]
    fn cut_text_drops_the_space_before_the_ellipsis() {
        assert_eq!(cut_text("hello world", 7), "hello\u{2026}");
        assert_eq!(cut_text("hello  world", 8), "hello\u{2026}");
        assert_eq!(cut_text("hello world", 6), "hello\u{2026}");
        assert_eq!(cut_text("hello world", 11), "hello world");
        assert_eq!(cut_text("hello", 0), "");
    }

    #[test]
    fn segment_key_is_fixed_colour_and_shortens_below_17_cells() {
        let mut grid = Grid::new(40, 4);
        paint_segment_key(&mut grid, Span { x: 10, w: 17 }, 1, ChartGlyphs::Eighths);
        let text: String = (10..40).map(|c| grid.get(c, 1).unwrap().ch).collect();
        assert!(
            text.starts_with("\u{2587}cached \u{2587}new \u{2587}out"),
            "{text:?}"
        );
        assert_eq!(grid.get(10, 1).unwrap().fg, C16::Blue, "cached ▇");
        assert_eq!(grid.get(11, 1).unwrap().fg, C16::BrightBlack, "dim label");
        assert_eq!(grid.get(18, 1).unwrap().fg, C16::BrightCyan, "new ▇");
        assert_eq!(grid.get(23, 1).unwrap().fg, C16::Yellow, "out ▇");

        // Narrower than 17 cells: "cached" shortens to "cach" (#96).
        let mut grid = Grid::new(40, 4);
        paint_segment_key(&mut grid, Span { x: 10, w: 16 }, 1, ChartGlyphs::Eighths);
        let text: String = (10..40).map(|c| grid.get(c, 1).unwrap().ch).collect();
        assert!(
            text.starts_with("\u{2587}cach \u{2587}new \u{2587}out"),
            "{text:?}"
        );

        // A key running off the right edge draws only what the grid holds.
        let mut grid = Grid::new(16, 4);
        paint_segment_key(&mut grid, Span { x: 12, w: 30 }, 1, ChartGlyphs::Eighths);
        assert_eq!(grid.get(12, 1).unwrap().ch, '\u{2587}');
        assert_eq!(grid.get(13, 1).unwrap().ch, 'c');
        assert_eq!(grid.get(14, 1).unwrap().ch, 'a');

        // Halves: `▄`, not `▇` — eurlatgr lacks the eighths block.
        let mut grid = Grid::new(40, 4);
        paint_segment_key(&mut grid, Span { x: 10, w: 17 }, 1, ChartGlyphs::Halves);
        assert_eq!(grid.get(10, 1).unwrap().ch, '\u{2584}');
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
        // #90: Û (U+00DB) now transliterates to its base letter 'U'.
        assert_eq!(fit_ellipsis("PRE\u{1b}[2JPOST\u{db}", 32), "PRE[2JPOSTU");
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
            cached: 0,
            open: false,
            done_known: true,
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

    fn t(secs: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(secs)
    }

    fn claim_row(claim: &HeaderClaim, now: SystemTime, tz: i32, limit: usize) -> String {
        let mut grid = Grid::new(80, 2);
        paint_claim(&mut grid, 0, claim, now, tz, limit);
        (0..80).map(|c| grid.get(c, 0).unwrap().ch).collect()
    }

    fn idle_claim(holder: &str, expires_in: u64, now: SystemTime) -> HeaderClaim {
        HeaderClaim {
            model: "big".to_owned(),
            holder: holder.to_owned(),
            in_use: false,
            expires_at: Some(now + Duration::from_secs(expires_in)),
            hold_until: None,
            hold_tz_offset_s: 0,
            hold_reason: String::new(),
        }
    }

    #[test]
    fn claim_time_left_is_m_ss_then_h_min() {
        assert_eq!(format_left(0), "0:00");
        assert_eq!(format_left(41), "0:41");
        assert_eq!(format_left(4 * 60 + 12), "4:12");
        assert_eq!(format_left(59), "0:59");
        assert_eq!(format_left(60), "1:00");
        assert_eq!(format_left(3599), "59:59");
        assert_eq!(format_left(3600), "1 h 00 min");
        assert_eq!(format_left(3600 + 5 * 60), "1 h 05 min");
        assert_eq!(format_left(10 * 3600 + 12 * 60), "10 h 12 min");
    }

    fn from_rfc(text: &str) -> SystemTime {
        let unix = crate::localtime::parse_rfc3339(text).expect("rfc3339");
        UNIX_EPOCH + Duration::from_secs(u64::try_from(unix).unwrap())
    }

    #[test]
    fn hold_until_adds_the_date_when_it_is_not_today() {
        let now = from_rfc("2026-10-12T18:00:00Z");
        assert_eq!(
            format_until(from_rfc("2026-10-12T21:00:00Z"), now, 0, 0),
            "until 21:00"
        );
        assert_eq!(
            format_until(from_rfc("2026-10-14T00:00:00Z"), now, 0, 0),
            "until 10-14 00:00"
        );
        // Offset +5:30: 18:00 UTC is 23:30 local; 21:00 UTC is 02:30 next local day.
        assert_eq!(
            format_until(
                from_rfc("2026-10-12T21:00:00Z"),
                now,
                5 * 3600 + 30 * 60,
                5 * 3600 + 30 * 60
            ),
            "until 10-13 02:30"
        );
    }

    /// A hold that ends after a daylight-saving change uses the offset at
    /// that end time, not the offset at now.
    #[test]
    fn hold_until_uses_the_offset_at_the_end_time() {
        let now = from_rfc("2026-03-08T06:00:00Z");
        let until = from_rfc("2026-03-08T12:00:00Z");
        // Now is still EST (UTC−5); the hold ends in EDT (UTC−4).
        assert_eq!(
            format_until(until, now, -5 * 3600, -4 * 3600),
            "until 08:00"
        );
        // The same instants with now's offset on the end time would be 07:00.
        assert_eq!(
            format_until(until, now, -5 * 3600, -5 * 3600),
            "until 07:00"
        );
    }

    #[test]
    fn claim_status_goes_amber_under_a_minute_and_stays_at_zero() {
        let now = t(1_000);
        let under = idle_claim("eval-runner", 59, now);
        let on = idle_claim("eval-runner", 60, now);
        let past = HeaderClaim {
            expires_at: Some(now - Duration::from_secs(5)),
            ..idle_claim("eval-runner", 0, now)
        };
        let s59 = claim_status(&under, now, 0).expect("status");
        let s60 = claim_status(&on, now, 0).expect("status");
        let s0 = claim_status(&past, now, 0).expect("status");
        assert_eq!(s59.text, "0:59 left");
        assert_eq!(s59.fg, C16::Yellow);
        assert_eq!(s60.text, "1:00 left");
        assert_eq!(s60.fg, C16::BrightWhite);
        assert_eq!(s0.text, "0:00 left");
        assert_eq!(s0.fg, C16::Yellow);
    }

    #[test]
    fn claim_sheds_reason_then_label_then_nothing() {
        let now = t(1_000);
        let claim = HeaderClaim {
            model: "big".to_owned(),
            holder: "operator".to_owned(),
            in_use: false,
            expires_at: None,
            hold_until: Some(now + Duration::from_secs(3600)),
            hold_tz_offset_s: 0,
            hold_reason: "overnight eval, do not unload".to_owned(),
        };
        let full = claim_row(&claim, now, 0, 80);
        assert!(
            full.contains("hold operator") && full.contains("overnight eval"),
            "{full:?}"
        );

        // `hold operator · until HH:MM · ` is 4+1+8+3+11+3 = 30, plus reason.
        let cut = claim_row(&claim, now, 0, 30 + 8);
        let cut: String = cut.trim_end().chars().collect();
        assert!(cut.contains("hold operator"), "{cut:?}");
        assert!(cut.contains('\u{2026}'), "{cut:?}");
        assert!(!cut.contains("unload"), "{cut:?}");

        let no_reason = claim_row(&claim, now, 0, 30);
        let no_reason: String = no_reason.trim_end().chars().collect();
        assert!(no_reason.starts_with("hold operator"), "{no_reason:?}");
        assert!(!no_reason.contains("overnight"), "{no_reason:?}");
        assert!(!no_reason.contains('\u{2026}'), "{no_reason:?}");

        // holder (8) + sep (3) + until HH:MM (11) = 22. Label does not fit.
        let no_label = claim_row(&claim, now, 0, 22);
        let no_label: String = no_label.trim_end().chars().collect();
        assert!(no_label.starts_with("operator"), "{no_label:?}");
        assert!(!no_label.contains("hold"), "{no_label:?}");
        assert!(
            !no_label.contains("overnight"),
            "reason without label: {no_label:?}"
        );

        let nothing = claim_row(&claim, now, 0, 7);
        assert!(nothing.chars().all(|c| c == ' '), "{nothing:?}");
    }

    #[test]
    fn a_cut_reason_needs_two_non_space_characters() {
        // leftover = SEP (3) + room. room 3 is `cut_text(_, 3)`: two chars,
        // then a trailing space is trimmed, so `a b c` would be `a…`.
        assert_eq!(fit_reason("a b c", 6), None);
        // `abc` is three cells: it fits whole in room 3. A longer string
        // is cut to two cells and `…`.
        assert_eq!(fit_reason("abc", 6).as_deref(), Some("abc"));
        assert_eq!(fit_reason("abcd", 6).as_deref(), Some("ab\u{2026}"));
        assert_eq!(fit_reason("ab", 6).as_deref(), Some("ab"));
        assert_eq!(fit_reason("a", 6).as_deref(), Some("a"));
        assert_eq!(fit_reason("a b c d e f", 6), None);
        assert_eq!(fit_reason("a b c d e f", 8).as_deref(), Some("a b\u{2026}"));
    }
}
