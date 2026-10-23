//! Text goldens for the tty11 layout. Frames are compared cell by cell.

use std::path::PathBuf;
use std::time::{Duration, Instant, UNIX_EPOCH};

use llama_core::backend::{Backend, BackendInfo, EngineStats};
use llama_core::detail::ModelDetail;
use llama_watch::collector::LoadSource;
use llama_watch::config::{ChartGlyphs, Config};
use llama_watch::jobs::{
    Job, JobNumber, JobPhase, JobPhaseRow, JobRate, JobSeries, JobState, PhaseMark,
};
use llama_watch::localtime::parse_rfc3339;
use llama_watch::setup_rules::{LiveCtx, Rules};
use llama_watch::tty::chart::ChartBucket;
use llama_watch::tty::grid::C16;
use llama_watch::tty::layout::{
    Activity, HeaderClaim, HeaderDisplaced, HeaderGpu, HealthSeg, HealthStatus, JobView, SetupItem,
    SetupRow, SetupView, Slot, TtyModel, WatchState, layout, layout_exact, replay_shown,
};
use llama_watch::tty::term::{GLYPHS, Size, Term};

#[derive(serde::Deserialize)]
struct Fix {
    w: u16,
    h: u16,
    rows: Vec<String>,
    cols: Vec<Vec<[u16; 3]>>,
}

#[test]
fn golden_frames_match_character_and_colour() {
    let generating = load("generating-480.json");
    let idle = load("idle-480.json");
    let inn = region(&generating, "IN ", Some("OUT "));
    let out_gen = region(&generating, "OUT ", None);
    let out_idle = region(&idle, "OUT ", None);
    let frames = [
        ("generating-480.json", WatchState::Generating, &out_gen),
        ("idle-480.json", WatchState::Ready, &out_idle),
        ("down-480.json", WatchState::AiDown, &out_idle),
        ("starting-480.json", WatchState::Starting, &out_gen),
        ("generating-240.json", WatchState::Generating, &out_gen),
        ("idle-240.json", WatchState::Ready, &out_idle),
        ("down-240.json", WatchState::AiDown, &out_idle),
        ("starting-240.json", WatchState::Starting, &out_gen),
    ];
    for (name, state, out_lines) in frames {
        let fix = load(name);
        let mut model = sample(state);
        model.in_title = title_line(&fix, "IN ");
        model.out_title = title_line(&fix, "OUT ");
        if state != WatchState::Starting {
            model.in_lines = inn.clone();
            model.out_lines = out_lines.clone();
        }
        assert_frame(name, &fix, &model);
    }
    let small = load("small-150.json");
    assert_frame("small-150.json", &small, &sample(WatchState::Ready));
}

#[test]
fn replay_spreads_new_chars_over_ten_frames() {
    assert_eq!(replay_shown(0, 5), 0);
    assert_eq!(replay_shown(10, 0), 0);
    assert_eq!(replay_shown(10, 1), 1);
    assert_eq!(replay_shown(10, 5), 5);
    assert_eq!(replay_shown(10, 10), 10);
    assert_eq!(replay_shown(11, 1), 1);
    assert_eq!(replay_shown(11, 5), 5);

    let mut model = sample(WatchState::Ready);
    // Two spaces keep the payload inside the text region (columns 0 and 1
    // are outside it). Joined "  ~~\n  ^^" is 9 chars.
    model.out_lines = vec!["  ~~".to_string(), "  ^^".to_string()];
    model.replay_frame = Some(4); // 3 chars: two spaces and one '~'
    assert_eq!(counts(&model, 160, 48), (1, 0), "frame 4 of 9 chars");
    model.replay_frame = Some(6); // 5 chars: "  ~~\n"
    assert_eq!(
        counts(&model, 160, 48),
        (2, 0),
        "frame 6 stops on the newline"
    );
    model.replay_frame = Some(10);
    assert_eq!(counts(&model, 160, 48), (2, 2), "frame 10 shows both lines");
}

#[test]
fn llama_text_is_placed_only_as_sanitised_cells() {
    let mut model = sample(WatchState::Ready);
    model.in_lines = vec!["PRE\u{1b}[2JPOST\u{db}END".to_string()];
    let grid = draw(&model, 160, 48);
    let mut chars = Vec::new();
    for row in 0..grid.rows() {
        for col in 0..grid.cols() {
            let cell = grid.get(col, row).expect("cell");
            assert!(
                is_console_char(cell.ch),
                "cell {ch:?} at {col},{row} is not a console char",
                ch = cell.ch
            );
            if cell.ch != ' ' {
                chars.push(cell.ch);
            }
        }
    }
    let text: String = chars.iter().collect();
    assert!(
        // #90: Û (U+00DB) now transliterates to its base letter 'U' rather
        // than becoming '?'.
        text.contains("PRE[2JPOSTUEND"),
        "sanitised llama text missing from {text}"
    );
    assert!(!text.contains('\u{1b}'));
    assert!(!text.contains('\u{db}'));

    let mut term = sized(160, 48);
    term.render(&grid, Instant::now()).expect("render");
    let bytes = term.out();
    assert!(
        !bytes.contains(&0xdb),
        "latin-1 leaked into the byte stream"
    );
    assert!(
        bytes.windows(14).any(|w| w == b"PRE[2JPOSTUEND"),
        "sanitised text was not emitted"
    );
}

fn assert_frame(name: &str, fix: &Fix, model: &TtyModel) {
    let term = sized(fix.w, fix.h);
    assert_eq!(term.cols(), fix.w, "{name} term cols");
    assert_eq!(term.rows(), fix.h, "{name} term rows");
    let grid = layout(model, term.cols(), term.rows());
    assert_eq!(grid.cols(), fix.w, "{name} grid cols");
    assert_eq!(grid.rows(), fix.h, "{name} grid rows");
    assert_eq!(fix.rows.len(), usize::from(fix.h), "{name} row count");
    let mut mismatches = Vec::new();
    let mut mismatch_count = 0usize;
    for row in 0..fix.h {
        let expect = &fix.rows[usize::from(row)];
        assert_eq!(
            expect.chars().count(),
            usize::from(fix.w),
            "{name} row {row} width"
        );
        let colour = expand(&fix.cols[usize::from(row)], usize::from(fix.w));
        for (col, ch) in expect.chars().enumerate() {
            let cell = grid.get(col as u16, row).expect("cell");
            let (fg, bg) = colour[col];
            if cell.ch != ch || fg_sgr(cell.fg) != fg || bg_sgr(cell.bg) != bg {
                mismatch_count += 1;
                if mismatches.len() < 12 {
                    mismatches.push(format!(
                        "r{row} c{col}: expected {ch:?} {fg}/{} got {:?} {}/{}",
                        bg,
                        cell.ch,
                        fg_sgr(cell.fg),
                        bg_sgr(cell.bg)
                    ));
                }
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{name} differs in {mismatch_count} cells:\n{}",
        mismatches.join("\n")
    );
    // Exact colours do not change the bytes tty11 writes, including a
    // frame that differs only in those colours.
    let exact = layout_exact(model, grid.cols(), grid.rows());
    assert_eq!(console_bytes(&grid), console_bytes(&exact), "{name}");
    let mut term = sized(grid.cols(), grid.rows());
    let now = Instant::now();
    term.render(&grid, now).expect("plain");
    term.out_mut().clear();
    term.render(&exact, now).expect("exact");
    assert!(
        term.out().is_empty(),
        "{name}: an exact-only change wrote {} bytes",
        term.out().len()
    );
}

fn console_bytes(grid: &llama_watch::tty::grid::Grid) -> Vec<u8> {
    let mut term = sized(grid.cols(), grid.rows());
    term.render(grid, Instant::now()).expect("render");
    term.out().to_vec()
}

fn counts(model: &TtyModel, cols: u16, rows: u16) -> (usize, usize) {
    let grid = draw(model, cols, rows);
    let mut tilde = 0;
    let mut caret = 0;
    for row in 0..grid.rows() {
        for col in 0..grid.cols() {
            match grid.get(col, row).expect("cell").ch {
                '~' => tilde += 1,
                '^' => caret += 1,
                _ => {}
            }
        }
    }
    (tilde, caret)
}

fn draw(model: &TtyModel, cols: u16, rows: u16) -> llama_watch::tty::grid::Grid {
    let term = sized(cols, rows);
    layout(model, term.cols(), term.rows())
}

fn sized(cols: u16, rows: u16) -> Term<Vec<u8>> {
    Term::new(
        Vec::new(),
        move || Ok(Size { cols, rows }),
        Duration::from_secs(60),
        Instant::now(),
    )
    .expect("size")
}

#[test]
fn raw_multiline_tail_keeps_every_character_inside_the_margin() {
    let mut model = sample(WatchState::Ready);
    let width = 240usize - 4;
    let long = "L".repeat(width + 10);
    model.in_lines = vec![format!("Hello world\nsecond line\n{long}")];
    model.out_lines = vec!["Xyz tail".to_string()];
    let grid = draw(&model, 240, 67);

    let hello = row_with(&grid, "Hello world");
    let left = left_rows(&grid, hello, 8);
    assert!(left[0].contains("Hello world"), "{left:?}");
    assert!(
        left.iter().any(|line| line.contains("second line")),
        "{left:?}"
    );
    let mid = grid.cols() / 2;
    let ells: usize = (hello..hello + 8)
        .map(|row| {
            (2..mid)
                .filter(|col| grid.get(*col, row).unwrap().ch == 'L')
                .count()
        })
        .sum();
    assert_eq!(ells, long.len(), "every L stayed in the left pane");
    let out = row_with(&grid, "Xyz tail");
    assert!(row_string(&grid, out).contains("Xyz tail"));
    for row in [hello, hello + 1, hello + 2, hello + 3, out] {
        assert_eq!(grid.get(0, row).unwrap().ch, ' ', "margin col 0 row {row}");
        assert_eq!(grid.get(1, row).unwrap().ch, ' ', "margin col 1 row {row}");
    }
}

#[test]
fn slot_context_fits_at_160x48_and_240x67() {
    for (cols, rows) in [(160u16, 48u16), (240, 67)] {
        let grid = draw(&sample(WatchState::Generating), cols, rows);
        assert_blank_edges(&grid);
        let row = row_with(&grid, "91k/262k");
        let text = row_string(&grid, row);
        let tok = char_at(&text, "612 tok");
        // The slot's own `ctx`, right of decoded (SETUP's ctx row may
        // share the line on the left, #52).
        let after: String = text.chars().skip(tok).collect();
        let label = tok + char_at(&after, "ctx");
        let value = char_at(&text, "91k/262k");
        assert!(
            tok < label && label < value,
            "{cols}x{rows} ctx sits right of decoded: {text}"
        );
        let meter = ctx_meter_cells(&grid, row, value as u16);
        assert!(
            meter.iter().any(|(ch, _)| is_lit_meter_cell(*ch)),
            "{cols}x{rows} context meter missing: {text}"
        );
        let lit: Vec<C16> = meter
            .iter()
            .filter(|(ch, _)| is_lit_meter_cell(*ch))
            .map(|(_, fg)| *fg)
            .collect();
        assert_eq!(
            steps_of(&lit),
            [C16::Blue, C16::BrightBlue],
            "{cols}x{rows} 91k/262k is a spectrum up to the bright-blue step, {meter:?}"
        );
    }
}

#[test]
fn missing_context_field_shows_ctx_dashes() {
    let mut model = sample(WatchState::Generating);
    model.slots[0].ctx_prompt = None;
    model.slots[0].n_ctx = Some(262_144);
    let grid = draw(&model, 240, 67);
    let text = row_string(&grid, row_with(&grid, "s0"));
    assert!(text.contains("ctx --"), "{text}");
    assert!(!text.contains("91k"), "{text}");
    assert!(!text.contains("/262"), "{text}");
}

#[test]
fn zero_n_ctx_shows_the_ratio_without_a_fill() {
    let mut model = sample(WatchState::Ready);
    model.slots = vec![Slot {
        id: 0,
        generating: false,
        done: 0,
        cached: 0,
        open: false,
        done_known: true,
        total: 0,
        decoded: 3,
        ctx_prompt: Some(12),
        n_ctx: Some(0),
        ctx_history: Vec::new(),
    }];
    let grid = draw(&model, 160, 48);
    let row = row_with(&grid, "15/0");
    let text = row_string(&grid, row);
    assert!(text.contains("ctx 15/0"), "{text}");
    let value = char_at(&text, "15/0");
    let meter = ctx_meter_cells(&grid, row, value as u16);
    assert!(
        meter
            .iter()
            .all(|(ch, _)| !is_lit_meter_cell(*ch) && *ch != '░'),
        "zero n_ctx must not draw a meter: {meter:?} in {text}"
    );
}

#[test]
fn context_above_n_ctx_clamps_the_meter_to_a_full_spectrum() {
    let mut model = sample(WatchState::Generating);
    model.slots = vec![Slot {
        id: 0,
        generating: true,
        done: 1,
        cached: 0,
        open: false,
        done_known: true,
        total: 1,
        decoded: 1,
        ctx_prompt: Some(300_000),
        n_ctx: Some(262_144),
        ctx_history: Vec::new(),
    }];
    let grid = draw(&model, 240, 67);
    let row = row_with(&grid, "300k/262k");
    let text = row_string(&grid, row);
    let value = char_at(&text, "300k/262k");
    let meter = ctx_meter_cells(&grid, row, value as u16);
    assert!(
        meter.iter().any(|(ch, _)| is_lit_meter_cell(*ch)),
        "overfull context should still draw a meter: {text}"
    );
    assert!(
        meter.iter().all(|(ch, _)| *ch != '░'),
        "overfull meter must be clamped full, got {meter:?}"
    );
    let lit: Vec<C16> = meter
        .iter()
        .filter(|(ch, _)| is_lit_meter_cell(*ch))
        .map(|(_, fg)| *fg)
        .collect();
    assert_eq!(
        steps_of(&lit),
        STEPS,
        "overfull meter shows every step and ends red, {meter:?}"
    );
}

/// Char column of `needle` in a row. Block glyphs are multibyte, so a byte index is not a column.
fn char_at(text: &str, needle: &str) -> usize {
    let byte = text
        .find(needle)
        .unwrap_or_else(|| panic!("missing {needle} in {text}"));
    text[..byte].chars().count()
}

/// Cells of the context meter: the glyph run immediately left of `value_col`.
/// #95: the slot rows stack with no blank row between, so the meter keeps
/// #52's whole-cell glyph (`▇`/`▄`) instead of the half-cell `█`/`▐`/`▌`.
fn ctx_meter_cells(
    grid: &llama_watch::tty::grid::Grid,
    row: u16,
    value_col: u16,
) -> Vec<(char, C16)> {
    let mut cells = Vec::new();
    let mut col = value_col;
    while col > 0 {
        col -= 1;
        let cell = grid.get(col, row).expect("cell");
        if cell.ch == ' ' && cells.is_empty() {
            continue;
        }
        if !matches!(cell.ch, '█' | '▐' | '▌' | '▇' | '▄' | '░' | '▓' | '▒' | '·')
        {
            break;
        }
        cells.push((cell.ch, cell.fg));
    }
    cells.reverse();
    cells
}

/// A meter's lit cells (#95: `▇` in eighths mode, `▄` in halves).
fn is_lit_meter_cell(ch: char) -> bool {
    matches!(ch, '█' | '▐' | '▌' | '▇' | '▄')
}

#[test]
fn four_slots_stay_above_the_requests_header() {
    for (cols, rows) in [(240u16, 67u16), (480, 135)] {
        let mut model = sample(WatchState::Generating);
        model.slots = (0..4)
            .map(|id| Slot {
                id,
                generating: true,
                done: 10,
                cached: 0,
                open: false,
                done_known: true,
                total: 20,
                decoded: 5,
                ctx_prompt: None,
                n_ctx: None,
                ctx_history: Vec::new(),
            })
            .collect();
        let grid = draw(&model, cols, rows);
        let recent = row_with(&grid, "RECENT");
        for id in 0..4 {
            let slot_row = row_with(&grid, &format!("s{id}"));
            assert!(
                slot_row < recent,
                "{cols}x{rows}: s{id} on row {slot_row} overlaps RECENT on {recent}"
            );
        }
        let header = row_string(&grid, recent);
        assert!(header.contains("RECENT"), "{header}");
        assert!(!header.contains("s0"), "{header}");
    }
}

const RECENT_WIDTHS: [(u16, u16); 4] = [(160, 48), (240, 67), (286, 67), (480, 135)];
const TIME_FULL: &str = "2026-09-25 15:57:08";
const TIME_SHORT: &str = "09-25 15:57:08";
#[allow(dead_code)]
const LEGEND_NARROW: &str = "PROMPT = prompt processing (prefill) · GEN = token generation (decode) · CACHED = prompt tokens reused from KV cache";

/// ISO activity timestamps render as a whole local time, the client IP has no
/// `ip:` prefix, and the columns of one row do not share cells.
///
/// Extra width lengthens the timestamp, then the model, then the generation
/// bar, then the gaps. A wider screen therefore has a wider bar, or wider
/// gaps once the bar has reached its cap.
#[test]
fn recent_time_is_complete_and_columns_do_not_overlap() {
    let mut bars = Vec::new();
    let mut gaps = Vec::new();
    for (cols, rows) in RECENT_WIDTHS {
        let grid = draw(&recent_model(), cols, rows);
        let header_row = row_with(&grid, "RECENT");
        let header = full_row(&grid, header_row);
        let data_row = header_row + 1;
        let data = full_row(&grid, data_row);
        let shown = if data.contains(TIME_FULL) {
            TIME_FULL
        } else if data.contains(TIME_SHORT) {
            TIME_SHORT
        } else {
            panic!("{cols}: time is truncated or still ISO: {data}");
        };
        // The year form is the upgrade. Every tested width has room for it
        // after the 14-column floor, because extra width goes to timestamps
        // before the model column.
        assert_eq!(shown, TIME_FULL, "{cols}: {data}");
        assert!(
            !data.contains("2026-09-25T"),
            "{cols}: ISO timestamp ran into the next column: {data}"
        );
        let time_at = find_chars(&data, shown).expect("time");
        let source_at = find_chars(&data, "192.0.2.83").expect("source ip");
        assert!(
            source_at > time_at + shown.chars().count(),
            "{cols}: time [{time_at}] collides with source [{source_at}]: {data}"
        );
        assert!(
            !data.contains("ip:"),
            "{cols}: source still has the ip: prefix: {data}"
        );
        let model_at = find_chars(&data, "GLM-4.7 Flash").unwrap_or_else(|| {
            panic!("{cols}: model name was truncated below a width that fits it: {data}")
        });
        assert!(
            model_at > source_at + "192.0.2.83".chars().count(),
            "{cols}: source collides with model: {data}"
        );
        let fields = [
            shown,
            "192.0.2.83",
            "GLM-4.7 Flash",
            "91,204",
            "88,960",
            "612",
            "1,212",
            "54.2",
            "13.2s",
        ];
        let mut spans = Vec::new();
        for field in fields {
            let start = find_chars(&data, field).unwrap_or_else(|| {
                panic!("{cols}: missing {field} in {data}");
            });
            spans.push((start, start + field.chars().count(), field));
        }
        for pair in spans.windows(2) {
            assert!(
                pair[0].1 <= pair[1].0,
                "{cols}: {:?} overlaps {:?}: {data}",
                pair[0].2,
                pair[1].2
            );
        }
        // The rate bar is the last filled column (#8 moved DUR left of it).
        let bar_end = data
            .chars()
            .collect::<Vec<_>>()
            .iter()
            .rposition(|ch| is_bar_char(*ch))
            .expect("bar");
        assert!(
            bar_end > usize::from(cols) * 3 / 4,
            "{cols}: RECENT stops at {bar_end}, short of the full width"
        );
        assert!(
            header.contains("TIME") && header.contains("SOURCE") && header.contains("MODEL"),
            "{cols}: header {header}"
        );
        assert!(
            !header.contains("PP t/s") && !header.contains("TG t/s"),
            "{cols}: old speed headers remain: {header}"
        );
        let bar = data.chars().filter(|ch| is_bar_char(*ch)).count();
        bars.push(bar);
        gaps.push(source_at - (time_at + shown.chars().count()));
    }
    assert!(
        bars[1] > bars[0] && bars[3] >= bars[2] && bars[2] >= bars[1],
        "generation bar should widen before gaps do: {bars:?} gaps {gaps:?}"
    );
    assert!(
        gaps[3] >= gaps[2] && gaps[2] >= gaps[1] && gaps[1] >= gaps[0],
        "column gaps should not shrink as the screen widens: bars {bars:?} gaps {gaps:?}"
    );
    assert!(
        bars[3] > bars[0] || gaps[3] > gaps[0],
        "480 should spend width on the bar or the gaps: bars {bars:?} gaps {gaps:?}"
    );
}

#[test]
fn recent_headers_use_plain_words_and_the_narrow_legend() {
    let narrow = draw(&recent_model(), 160, 48);
    let header = full_row(&narrow, row_with(&narrow, "RECENT"));
    assert!(
        !header.contains("tokens") && !header.contains("tok/s") && !header.contains("DURATION"),
        "160 should use short headers: {header}"
    );
    assert!(header.contains("PROMPT"), "{header}");
    assert!(header.contains("CACHED"), "{header}");
    assert!(header.contains("GEN"), "{header}");
    assert!(header.contains("DUR"), "{header}");
    assert!(header.contains("SL"), "SL column: {header}");
    let all = whole(&narrow);
    assert!(
        !all.contains("prompt processing (prefill)"),
        "legend stayed: {all}"
    );
    for (cols, rows) in [(240u16, 67u16), (286, 67), (480, 135)] {
        let grid = draw(&recent_model(), cols, rows);
        let header = full_row(&grid, row_with(&grid, "RECENT"));
        for word in [
            "PROMPT tokens",
            "CACHED tokens",
            "OUTPUT tokens",
            "PROMPT tok/s",
            "GEN tok/s",
            "DURATION",
        ] {
            assert!(
                header.contains(word),
                "{cols} header missing {word}: {header}"
            );
        }
        let all = whole(&grid);
        assert!(
            !all.contains("prompt processing speed"),
            "{cols}: legend stayed"
        );
    }
}

#[test]
fn recent_model_and_source_ellipsis_only_when_the_column_is_short() {
    let mut model = recent_model();
    model.requests[0].model = "GLM-4.7 Flash".to_string();
    model.requests[1].model = "M".repeat(80);
    model.requests[1].source = "ip:2001:0db8:0000:0000:0000:0000:0000:0001".to_string();
    model.requests[1].time = "2026-09-25T18:09:08Z".to_string();
    model.requests[2].model = "PRE\u{1b}[2JPOST\u{db}".to_string();
    model.requests[2].source = "ip:\u{1b}[2J10.1.2.3".to_string();
    for (cols, rows) in RECENT_WIDTHS {
        let grid = draw(&model, cols, rows);
        let header = row_with(&grid, "RECENT");
        let short_name = full_row(&grid, header + 1);
        assert!(
            short_name.contains("GLM-4.7 Flash") && !short_name.contains('…'),
            "{cols}: a name that fits was cut: {short_name}"
        );
        let long_name = full_row(&grid, header + 2);
        assert!(
            long_name.contains('…'),
            "{cols}: overlong model/source was cut without an ellipsis: {long_name}"
        );
        assert!(
            !long_name.contains(&"M".repeat(40)),
            "{cols}: model spilled: {long_name}"
        );
        assert!(
            long_name.contains("2026-09-25 18:09:08"),
            "{cols}: second timestamp truncated: {long_name}"
        );
        let hostile = full_row(&grid, header + 3);
        // #90: Û (U+00DB) now transliterates to its base letter 'U'.
        assert!(hostile.contains("PRE[2JPOSTU"), "{cols}: {hostile}");
        assert!(hostile.contains("[2J10.1.2.3"), "{cols}: {hostile}");
        assert!(!hostile.contains("ip:"), "{cols}: {hostile}");
        assert!(!hostile.contains('\u{1b}') && !hostile.contains('\u{db}'));
    }
}

#[test]
fn recent_numbers_stay_right_aligned_under_their_headers() {
    for (cols, rows) in RECENT_WIDTHS {
        let grid = draw(&recent_model(), cols, rows);
        let header_row = row_with(&grid, "RECENT");
        let header = full_row(&grid, header_row);
        let data = full_row(&grid, header_row + 1);
        let pairs: &[(&str, &str)] = if cols == 160 {
            &[
                ("IN", "91,204"),
                ("CACHED", "88,960"),
                ("OUT", "612"),
                ("PROMPT", "1,212"),
                ("GEN", "54.2"),
                ("DUR", "13.2s"),
            ]
        } else {
            &[
                ("PROMPT tokens", "91,204"),
                ("CACHED tokens", "88,960"),
                ("OUTPUT tokens", "612"),
                ("PROMPT tok/s", "1,212"),
                ("GEN tok/s", "54.2"),
                ("DURATION", "13.2s"),
            ]
        };
        for (label, value) in pairs {
            let label_at = find_chars(&header, label).unwrap_or_else(|| {
                panic!("{cols}: missing header {label}: {header}");
            });
            let label_end = label_at + label.chars().count();
            let value_at = find_chars(&data, value).unwrap_or_else(|| {
                panic!("{cols}: missing {value}: {data}");
            });
            let value_end = value_at + value.chars().count();
            assert_eq!(
                value_end, label_end,
                "{cols}: {value} ends at {value_end}, {label} ends at {label_end}"
            );
        }
    }
}

/// #133: an unreported cached count is `--`, right-aligned like the other
/// `--` cells, and the prompt run is the dim full-height block. A reported
/// 0 stays `0` and is drawn as new. A known cached count is unchanged.
#[test]
fn recent_unknown_cache_is_dim_and_a_reported_zero_is_new() {
    let model = cache_rows_model();
    let grid = draw(&model, 240, 67);
    let header_row = row_with(&grid, "RECENT");
    let header = full_row(&grid, header_row);
    let label = "CACHED tokens";
    let label_end = find_chars(&header, label).expect("cached header") + label.chars().count();
    let prompt_label = "PROMPT tok/s";
    let prompt_end =
        find_chars(&header, prompt_label).expect("prompt header") + prompt_label.chars().count();
    let gen_label = "GEN tok/s";
    let gen_end = find_chars(&header, gen_label).expect("gen header") + gen_label.chars().count();
    // (model, cached text, unknown bar, reported zero, filled)
    let cases = [
        ("lcpp-miss", "--", BarKind::Unknown),
        ("lcpp-zero", "0", BarKind::NewOnly),
        ("vllm-miss", "--", BarKind::Unknown),
        ("vllm-zero", "0", BarKind::NewOnly),
        ("sg-miss", "--", BarKind::Unknown),
        ("sg-fill", "600", BarKind::Split),
    ];
    for (name, cached, kind) in cases {
        let row = row_with(&grid, name);
        let text = full_row(&grid, row);
        let start = label_end - cached.chars().count();
        let got: String = text
            .chars()
            .skip(start)
            .take(cached.chars().count())
            .collect();
        assert_eq!(got, cached, "{name}: CACHED cell: {text}");
        assert_eq!(
            text.chars().nth(start - 1),
            Some(' '),
            "{name}: {cached} is not right-aligned in its own cell: {text}"
        );
        if cached == "--" {
            let col = u16::try_from(start).expect("col");
            assert_eq!(
                grid.get(col, row).expect("cell").fg,
                C16::White,
                "{name}: CACHED `--` is the column's plain colour"
            );
        }
        let bar = bar_cells(&grid, row);
        match kind {
            BarKind::Unknown => {
                assert!(
                    bar.iter()
                        .any(|(ch, fg)| *ch == '\u{2587}' && *fg == C16::BrightBlack),
                    "{name}: prompt is not a dim full block: {bar:?}"
                );
                assert!(
                    bar.iter()
                        .all(|(_, fg)| *fg != C16::Blue && *fg != C16::BrightCyan),
                    "{name}: unknown cache was painted cached or new: {bar:?}"
                );
                assert!(
                    bar.iter()
                        .any(|(ch, fg)| *ch == '\u{2587}' && *fg == C16::Yellow),
                    "{name}: output run changed: {bar:?}"
                );
            }
            BarKind::NewOnly => {
                assert!(
                    bar.iter()
                        .any(|(ch, fg)| *ch == '\u{2584}' && *fg == C16::BrightCyan),
                    "{name}: a reported 0 is all new: {bar:?}"
                );
                assert!(
                    bar.iter().all(|(_, fg)| *fg != C16::Blue),
                    "{name}: a reported 0 drew a cached run: {bar:?}"
                );
            }
            BarKind::Split => {
                assert!(
                    bar.iter()
                        .any(|(ch, fg)| *ch == '\u{2587}' && *fg == C16::Blue),
                    "{name}: cached run: {bar:?}"
                );
                assert!(
                    bar.iter()
                        .any(|(ch, fg)| *ch == '\u{2584}' && *fg == C16::BrightCyan),
                    "{name}: new run: {bar:?}"
                );
                assert!(
                    bar.iter()
                        .any(|(ch, fg)| *ch == '\u{2587}' && *fg == C16::Yellow),
                    "{name}: output run: {bar:?}"
                );
                assert!(
                    find_chars(&text, "~200").is_some_and(|at| at + 4 == prompt_end),
                    "{name}: prompt rate: {text}"
                );
                assert!(
                    find_chars(&text, "~50.0").is_some_and(|at| at + 5 == gen_end),
                    "{name}: gen rate: {text}"
                );
                assert!(
                    !text.contains("~600"),
                    "{name}: the cached count is not marked: {text}"
                );
            }
        }
    }
}

#[derive(Clone, Copy)]
enum BarKind {
    Unknown,
    NewOnly,
    Split,
}

fn bar_cells(grid: &llama_watch::tty::grid::Grid, row: u16) -> Vec<(char, C16)> {
    let start = (0..grid.cols())
        .find(|col| is_bar_char(grid.get(*col, row).expect("cell").ch))
        .unwrap_or_else(|| panic!("no bar on {row}"));
    (start..grid.cols())
        .map(|col| grid.get(col, row).expect("cell"))
        .take_while(|cell| is_bar_char(cell.ch) && cell.ch != '\u{2581}')
        .map(|cell| (cell.ch, cell.fg))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn cache_row(
    id: u32,
    model: &str,
    input: u64,
    cached: u64,
    unknown: bool,
    output: u64,
    prompt: Option<f64>,
    gen_rate: Option<f64>,
    measured: bool,
) -> Activity {
    Activity {
        id,
        time: format!("18:10:{id:02}"),
        source: "192.0.2.10".to_owned(),
        model: model.to_owned(),
        input_tok: input,
        cached_tok: cached,
        cached_unknown: unknown,
        output_tok: output,
        prompt_tps: prompt,
        gen_tps: gen_rate,
        prompt_measured: measured && prompt.is_some(),
        gen_measured: measured && gen_rate.is_some(),
        dur: "4.0s".to_owned(),
        n_ctx: Some(8_192),
        ..Activity::default()
    }
}

/// Unknown and reported-zero rows for llama.cpp, vLLM and SGLang, plus one
/// SGLang row filled from a single-request window (600 cached of 1,000,
/// 200 prompt tok/s, 50 gen tok/s).
fn cache_rows_model() -> TtyModel {
    let mut model = sample(WatchState::Ready);
    model.chart_glyphs = ChartGlyphs::Eighths;
    model.requests = vec![
        cache_row(
            6,
            "lcpp-miss",
            1_200,
            0,
            true,
            300,
            Some(400.0),
            Some(30.0),
            false,
        ),
        cache_row(
            5,
            "lcpp-zero",
            1_200,
            0,
            false,
            300,
            Some(400.0),
            Some(30.0),
            false,
        ),
        cache_row(4, "vllm-miss", 1_000, 0, true, 200, None, None, false),
        cache_row(3, "vllm-zero", 1_000, 0, false, 200, None, None, false),
        cache_row(2, "sg-miss", 800, 0, true, 100, None, None, false),
        cache_row(
            1,
            "sg-fill",
            1_000,
            600,
            false,
            401,
            Some(200.0),
            Some(50.0),
            true,
        ),
    ];
    model
}

#[test]
fn recent_cache_golden_matches_character_and_colour() {
    assert_frame(
        "recent-cache-240.json",
        &load("recent-cache-240.json"),
        &cache_rows_model(),
    );
}

#[test]
#[ignore = "run with --ignored to write the #133 cache goldens"]
fn dump_recent_cache_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    let grid = draw(&cache_rows_model(), 240, 67);
    std::fs::write(dir.join("recent-cache-240.json"), dump_grid(&grid)).expect("write tty golden");
    let gdir = grid_fix_dir();
    std::fs::create_dir_all(&gdir).expect("fixtures/grid");
    std::fs::write(
        gdir.join("recent-cache-192.json"),
        grid_fix_text(&grid_fix_of(&cache_rows_model())),
    )
    .expect("write grid golden");
}

/// #8: DURATION reads with the other numbers, left of the rate bar:
/// `... IN CACHED OUT PROMPT GEN DUR [bar] STATUS`, in both column sets and
/// with text on or off.
#[test]
fn recent_duration_sits_between_the_numbers_and_the_bar() {
    for show_text in [true, false] {
        for (cols, rows) in RECENT_WIDTHS {
            let mut model = recent_model();
            model.show_text = show_text;
            let grid = draw(&model, cols, rows);
            let header_row = row_with(&grid, "RECENT");
            let header = full_row(&grid, header_row);
            let data = full_row(&grid, header_row + 1);
            let (gen_label, dur_label) = if cols == 160 {
                ("GEN", "DUR")
            } else {
                ("GEN tok/s", "DURATION")
            };
            let gen_end = find_chars(&header, gen_label).expect("gen header") + gen_label.len();
            let dur_at = find_chars(&header, dur_label).expect("dur header");
            let dur_end = dur_at + dur_label.len();
            let value_at = find_chars(&data, "13.2s").expect("dur value");
            let bar_at = data.chars().position(is_bar_char).expect("bar");
            let bar_end = data
                .chars()
                .collect::<Vec<_>>()
                .iter()
                .rposition(|ch| is_bar_char(*ch))
                .expect("bar end");
            let after_bar: String = data.chars().skip(bar_end + 1).collect();
            let tag = format!("{cols} text {show_text}");
            assert!(gen_end < dur_at, "{tag}: DUR is not after GEN: {header}");
            assert!(dur_end < bar_at, "{tag}: DUR header is not left of the bar");
            assert!(
                value_at + 5 < bar_at,
                "{tag}: duration is right of the bar: {data}"
            );
            assert!(
                after_bar.contains("gen"),
                "{tag}: status is not after the bar: {data}"
            );
        }
    }
}

/// A RECENT bar cell (#75): blocks, eighths, the `_` / `▁` track.
fn is_bar_char(ch: char) -> bool {
    matches!(ch, '█' | '▌' | '▐' | '░' | '▄' | '_') || ('\u{2581}'..='\u{2587}').contains(&ch)
}

fn recent_model() -> TtyModel {
    let mut model = sample(WatchState::Generating);
    model.requests[0].time = "2026-09-25T15:57:08.5Z".to_string();
    model.requests[0].source = "ip:192.0.2.83".to_string();
    model.requests[0].model = "GLM-4.7 Flash".to_string();
    model
}

fn full_row(grid: &llama_watch::tty::grid::Grid, row: u16) -> String {
    (0..grid.cols())
        .map(|col| grid.get(col, row).expect("cell").ch)
        .collect()
}

fn find_chars(hay: &str, needle: &str) -> Option<usize> {
    let hay: Vec<char> = hay.chars().collect();
    let needle: Vec<char> = needle.chars().collect();
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len())
        .position(|window| window == needle.as_slice())
}

#[test]
fn replay_frame_zero_keeps_the_previous_poll() {
    let mut first = sample(WatchState::Generating);
    first.out_lines = vec!["Hello world".to_string()];
    first.out_shown = 0;
    first.replay_frame = Some(10);
    let done = draw(&first, 240, 67);
    let row = row_with(&done, "Hello world");
    assert!(
        row_string(&done, row).contains('█'),
        "{}",
        row_string(&done, row)
    );

    let mut second = first.clone();
    second.out_lines = vec!["Hello world\nNEWCHARS".to_string()];
    second.out_shown = "Hello world".chars().count();
    second.replay_frame = Some(0);
    let held = draw(&second, 240, 67);
    let row = row_with(&held, "Hello world");
    assert!(
        row_string(&held, row).contains('█'),
        "frame 0 must not blank OUT"
    );
    assert!(
        !whole(&held).contains("NEWCHARS"),
        "the new delta waits for a later frame"
    );

    second.replay_frame = Some(10);
    let revealed = draw(&second, 240, 67);
    assert!(
        whole(&revealed).contains("NEWCHARS"),
        "frame 10 shows the new line"
    );
}

#[test]
fn huge_model_name_does_not_panic_or_spill() {
    let mut model = sample(WatchState::Generating);
    model.model_name = "M".repeat(70_000);
    model.requests[0].source = "S".repeat(300);
    model.requests[0].model = "Qwen 35B".to_string();
    let grid = draw(&model, 240, 67);
    let header = row_string(&grid, 0);
    assert!(header.contains("2026-09-23"), "clock overwritten: {header}");
    assert!(
        !header.contains(&"M".repeat(49)),
        "model name spilled across the header"
    );
    assert!(
        header.contains("slots"),
        "slots label was pushed off: {header}"
    );
    let request = full_row(&grid, row_with(&grid, "RECENT") + 1);
    let qwen = find_chars(&request, "Qwen 35B").expect("model column overwritten");
    let mark = find_chars(&request, "…").expect("overlong source was not clipped");
    assert!(mark < qwen, "source ellipsis ran into the model: {request}");
    assert!(
        !request.chars().skip(qwen).any(|ch| ch == 'S'),
        "source spilled into the model: {request}"
    );
    let run = request
        .chars()
        .skip_while(|ch| *ch != 'S')
        .take_while(|ch| *ch == 'S')
        .count();
    assert!(run < 20, "source column did not clip the run of S ({run})");
}

#[test]
fn ceiling_rate_caps_the_bar_and_the_digits() {
    let mut model = sample(WatchState::Generating);
    model.gen_tps = Some(300.0);
    let grid = draw(&model, 240, 67);
    // 240x67 is a short screen: the rate bar is a lower-half cell so it
    // cannot bridge the blank rows above and below it.
    let row = (0..grid.rows())
        .find(|row| grid.get(2, *row).unwrap().ch == '▄')
        .expect("GEN bar");
    assert_eq!(grid.get(2, row).unwrap().fg, C16::Blue, "bar starts cold");
    let last = (2..28)
        .rev()
        .find(|col| matches!(grid.get(*col, row).unwrap().ch, '█' | '▄'))
        .expect("bar end");
    assert_eq!(
        fg_sgr(grid.get(last, row).unwrap().fg),
        97,
        "ceiling cap is white-hot"
    );
    // Digits sit in the fixed GEN columns and use the hot step.
    let digit = (3..8)
        .find(|row| grid.get(2, *row).unwrap().ch == '█')
        .expect("digit");
    assert_eq!(fg_sgr(grid.get(2, digit).unwrap().fg), 97);
}

#[test]
fn tall_narrow_digits_stay_in_their_half() {
    let mut model = sample(WatchState::Generating);
    model.gen_tps = Some(123.0);
    model.prompt_tps = Some(2345.0);
    let grid = draw(&model, 160, 90);
    let digit_row = (2..16)
        .find(|row| (2..28).any(|col| grid.get(col, *row).unwrap().ch == '█'))
        .expect("GEN digits");
    assert!(
        (2..28).any(|col| grid.get(col, digit_row).unwrap().ch == '█'),
        "GEN digits draw in columns 2-27"
    );
    let gen_right = (2..62)
        .rev()
        .find(|col| *col < 32 && grid.get(*col, digit_row).unwrap().ch == '█')
        .expect("GEN");
    let prompt_left = (32..62)
        .find(|col| grid.get(*col, digit_row).unwrap().ch == '█')
        .expect("PROMPT");
    assert!(
        gen_right < prompt_left,
        "GEN {gen_right} meets PROMPT {prompt_left}"
    );
    assert!(
        (32..62).any(|col| grid.get(col, digit_row).unwrap().ch == '█'),
        "PROMPT digits draw in columns 32-61"
    );
}

#[test]
fn oversized_window_does_not_panic_and_keeps_the_edges_blank() {
    let model = sample(WatchState::Starting);
    for (cols, rows) in [(400u16, u16::MAX), (u16::MAX, 300)] {
        let term = sized(cols, rows);
        assert_eq!(term.cols(), cols.min(1024), "term cols clamp");
        assert_eq!(term.rows(), rows.min(512), "term rows clamp");
        let clamped = layout(&model, term.cols(), term.rows());
        assert_blank_edges(&clamped);

        let raw = layout(&model, cols, rows);
        assert_eq!((raw.cols(), raw.rows()), (cols, rows));
        assert_blank_edges(&raw);
    }
}

fn chart_model() -> TtyModel {
    let mut model = sample(WatchState::Generating);
    model.in_title = "IN   prompt tail".to_string();
    model.out_title = "OUT  live".to_string();
    model.chart_bucket_s = 2;
    model
}

fn req_rule_row(grid: &llama_watch::tty::grid::Grid) -> u16 {
    let recent = row_with(grid, "RECENT");
    (recent + 1..grid.rows())
        .find(|row| is_rule_row(grid, *row))
        .expect("rule after RECENT")
}

fn is_rule_row(grid: &llama_watch::tty::grid::Grid, row: u16) -> bool {
    grid.get(1, row).unwrap().ch == '-' && grid.get(2, row).unwrap().ch == '-'
}

fn timeline_rule(grid: &llama_watch::tty::grid::Grid) -> u16 {
    (0..grid.rows())
        .find(|row| row_string(grid, *row).contains("TIMELINE"))
        .unwrap_or_else(|| panic!("missing TIMELINE"))
}

/// Chart rows between the timeline rule and the next rule.
fn chart_height_of(grid: &llama_watch::tty::grid::Grid) -> u16 {
    let start = timeline_rule(grid) + 1;
    (start..grid.rows())
        .take_while(|row| !is_rule_row(grid, *row))
        .count() as u16
}

/// Axis row: `now` in the left margin and the age label on the right.
/// The timeline rule also contains the word "now".
fn axis_row(grid: &llama_watch::tty::grid::Grid) -> u16 {
    (0..grid.rows())
        .find(|row| {
            let text = row_string(grid, *row);
            text.contains("now")
                && text.contains('-')
                && text.contains('m')
                && !text.contains("TIMELINE")
        })
        .unwrap_or_else(|| panic!("chart axis missing"))
}

#[test]
fn chart_is_nine_rows_under_recent_at_240x67() {
    let mut model = chart_model();
    model.chart = vec![
        ChartBucket {
            gen_tps: Some(54.2),
            prompt_tps: Some(0.0),
        };
        40
    ];
    let grid = draw(&model, 240, 67);
    assert_eq!(
        chart_height_of(&grid),
        14,
        "60-class band gives the extra RECENT row to the timeline"
    );
    let axis = axis_row(&grid);
    assert!(row_string(&grid, axis).contains("now"), "axis missing now");
    let start = timeline_rule(&grid) + 1;
    // 14 rows: 7 above the axis, 6 below.
    assert_eq!(axis, start + 7, "axis sits under seven gen rows");
    assert!(row_string(&grid, start).contains("250"), "gen ceiling");
    assert!(
        row_string(&grid, start + 13).contains("1500"),
        "prompt ceiling"
    );
    assert!(row_string(&grid, axis - 1).contains("gen"), "gen label");
    assert!(
        row_string(&grid, axis + 1).contains("prompt"),
        "prompt label"
    );
    let inn = row_with(&grid, "prompt tail");
    assert!(axis < inn, "chart must sit above IN");
}

#[test]
fn minimum_160x48_still_fits_the_chart() {
    let mut model = chart_model();
    model.chart = vec![ChartBucket {
        gen_tps: Some(20.0),
        prompt_tps: Some(0.0),
    }];
    let grid = draw(&model, 160, 48);
    let line = row_string(&grid, 0);
    assert!(
        !line.contains("too small"),
        "160x48 must still draw the dashboard, got {line}"
    );
    let axis = (0..grid.rows())
        .find(|row| {
            let text = row_string(&grid, *row);
            text.contains("now") && text.contains('m') && text.contains('-')
        })
        .expect("chart axis missing at 160x48");
    assert!(chart_height_of(&grid) >= 5);
    assert!(row_with(&grid, "prompt tail") > axis);
}

#[test]
fn sixteen_slots_at_minimum_hides_the_chart_and_keeps_in_out() {
    let mut model = chart_model();
    model.slots = (0..16)
        .map(|id| Slot {
            id,
            generating: true,
            done: 10,
            cached: 0,
            open: false,
            done_known: true,
            total: 20,
            decoded: 5,
            ctx_prompt: None,
            n_ctx: None,
            ctx_history: Vec::new(),
        })
        .collect();
    model.chart = vec![ChartBucket {
        gen_tps: Some(20.0),
        prompt_tps: Some(100.0),
    }];
    let grid = draw(&model, 160, 48);
    // Slot rows are clipped inside the top band, so they no longer steal
    // the timeline. IN and OUT share a row.
    assert!(
        chart_height_of(&grid) >= 5,
        "chart stays with the waterfall"
    );
    let inn = row_with(&grid, "prompt tail");
    let out = row_with(&grid, "OUT  live");
    assert!(inn > row_with(&grid, "RECENT") || inn == out);
    assert_eq!(inn, out, "IN and OUT are side by side");
}

#[test]
fn chart_shrinks_to_five_rows_before_in_out() {
    let mut model = chart_model();
    model.slots = (0..10)
        .map(|id| Slot {
            id,
            generating: true,
            done: 10,
            cached: 0,
            open: false,
            done_known: true,
            total: 20,
            decoded: 5,
            ctx_prompt: None,
            n_ctx: None,
            ctx_history: Vec::new(),
        })
        .collect();
    model.chart = vec![ChartBucket {
        gen_tps: Some(20.0),
        prompt_tps: Some(100.0),
    }];
    let grid = draw(&model, 160, 48);
    assert_eq!(
        chart_height_of(&grid),
        10,
        "extra slots stay in the top band"
    );
    let inn = row_with(&grid, "prompt tail");
    let out = row_with(&grid, "OUT  live");
    assert_eq!(inn, out, "IN and OUT share the text band");
}

#[test]
fn measured_zero_is_a_dot_and_no_data_is_blank() {
    let mut model = chart_model();
    // Oldest first: a measured zero, then a newer gap. The gap is drawn on the left.
    model.chart = vec![
        ChartBucket {
            gen_tps: Some(0.0),
            prompt_tps: Some(0.0),
        },
        ChartBucket {
            gen_tps: None,
            prompt_tps: None,
        },
    ];
    let grid = draw(&model, 240, 67);
    let axis = axis_row(&grid);
    let zero_col = (0..grid.cols())
        .find(|col| grid.get(*col, axis).unwrap().ch == '·')
        .expect("measured zero");
    assert!(
        zero_col < 16,
        "older zero moves right of the newest gap, got column {zero_col}"
    );
    assert_eq!(
        fg_sgr(grid.get(zero_col, axis).unwrap().fg),
        90,
        "zero is dim"
    );
    let gap_col = zero_col - 1;
    assert_eq!(
        grid.get(gap_col, axis).unwrap().ch,
        ' ',
        "no-data draws nothing"
    );
    for dy in 1..=4 {
        assert_eq!(grid.get(gap_col, axis - dy).unwrap().ch, ' ');
        assert_eq!(grid.get(gap_col, axis + dy).unwrap().ch, ' ');
        assert_eq!(grid.get(zero_col, axis - dy).unwrap().ch, ' ');
        assert_eq!(grid.get(zero_col, axis + dy).unwrap().ch, ' ');
    }
}

#[test]
fn history_flows_from_the_newest_bucket_on_the_left() {
    let mut model = chart_model();
    model.chart = vec![
        ChartBucket {
            gen_tps: Some(0.0),
            prompt_tps: Some(0.0),
        },
        ChartBucket {
            gen_tps: Some(54.0),
            prompt_tps: None,
        },
    ];
    let grid = draw(&model, 240, 67);
    let axis = axis_row(&grid);
    let axis_text = row_string(&grid, axis);
    let now_at = col_of(&grid, axis, "now");
    let age_at = col_of(&grid, axis, "-8m");
    assert!(
        (2..8).contains(&now_at),
        "now stays in the left margin, got {axis_text}"
    );
    assert!(
        axis_text.ends_with("-8m"),
        "age label is the right axis label, got {axis_text}"
    );
    assert!(now_at < age_at, "now at {now_at}, -8m at {age_at}");

    // Left margin is cols 2..=7, then a gutter, then the newest column.
    let data_left = 9u16;
    assert_eq!(
        grid.get(data_left - 1, axis).unwrap().ch,
        ' ',
        "gutter between now and the newest column"
    );
    let newer = (0..grid.cols())
        .find(|col| {
            let ch = grid.get(*col, axis - 1).unwrap().ch;
            ch == '█' || ch == '▄'
        })
        .expect("newer gen column");
    let older = (0..grid.cols())
        .find(|col| grid.get(*col, axis).unwrap().ch == '·')
        .expect("older zero");
    assert_eq!(
        newer, data_left,
        "newest bucket is the left edge of the data"
    );
    assert!(
        newer < older,
        "newer column {newer} should be left of older {older}"
    );

    let gen_at = col_of(&grid, axis - 1, "gen");
    let prompt_at = col_of(&grid, axis + 1, "prompt");
    assert!(
        gen_at < newer,
        "gen label at {gen_at} covers column {newer}"
    );
    assert!(
        prompt_at < newer,
        "prompt label at {prompt_at} covers column {newer}"
    );
    let start = timeline_rule(&grid) + 1;
    let height = chart_height_of(&grid);
    let tick = col_of(&grid, start, "250");
    let prompt_tick = col_of(&grid, start + height - 1, "1500");
    assert!(tick < newer, "ceiling covers the newest column");
    assert!(
        prompt_tick < newer,
        "prompt ceiling covers the newest column"
    );

    let wide = draw(&model, 480, 135);
    let wide_row = axis_row(&wide);
    let wide_axis = row_string(&wide, wide_row);
    assert!(
        (2..8).contains(&col_of(&wide, wide_row, "now")),
        "now stays in the left margin at 480, got {wide_axis}"
    );
    assert!(
        wide_axis.ends_with("-16m"),
        "480 band is still about 16 minutes, got {wide_axis}"
    );
}

#[test]
fn age_label_stays_clear_of_a_wide_band() {
    let mut model = chart_model();
    model.chart_bucket_s = 60;
    model.chart = vec![
        ChartBucket {
            gen_tps: Some(250.0),
            prompt_tps: Some(0.0),
        };
        70_000
    ];
    let grid = draw(&model, u16::MAX, 48);
    let axis = axis_row(&grid);
    let age_at = (0..grid.cols())
        .rev()
        .find(|col| grid.get(*col, axis).unwrap().ch == '-')
        .expect("age label");
    assert_eq!(grid.get(age_at, axis).unwrap().ch, '-');
    let above = grid.get(age_at, axis - 1).unwrap().ch;
    assert!(
        above != '█' && above != '▄' && above != '▀',
        "age label covers a bar at column {age_at}: {above:?}"
    );
    assert_eq!(
        grid.get(age_at - 1, axis).unwrap().ch,
        '·',
        "oldest column should sit against the age label"
    );
}

#[test]
fn oldest_bucket_falls_off_the_right_edge() {
    let mut model = chart_model();
    let mut chart = vec![
        ChartBucket {
            gen_tps: Some(0.0),
            prompt_tps: None,
        };
        400
    ];
    chart[0].gen_tps = Some(400.0);
    chart[399] = ChartBucket {
        gen_tps: Some(1.0),
        prompt_tps: None,
    };
    model.chart = chart;
    let grid = draw(&model, 240, 67);
    let axis = axis_row(&grid);
    let newest = (0..grid.cols())
        .find(|col| {
            let ch = grid.get(*col, axis - 1).unwrap().ch;
            ch == '▄' || ch == '█'
        })
        .expect("newest half-cell");
    assert_eq!(
        grid.get(newest, axis - 1).unwrap().ch,
        '▄',
        "newest bucket stays a single half-cell"
    );
    assert_eq!(grid.get(newest, axis - 2).unwrap().ch, ' ');
    for col in 0..grid.cols() {
        let full = (1..=4).all(|dy| grid.get(col, axis - dy).unwrap().ch == '█');
        assert!(!full, "oldest full column still on screen at {col}");
    }
    let older = (newest + 1..grid.cols())
        .find(|col| grid.get(*col, axis).unwrap().ch == '·')
        .expect("older dots to the right of newest");
    let rightmost = (0..grid.cols())
        .rev()
        .find(|col| grid.get(*col, axis).unwrap().ch == '·')
        .expect("rightmost older dot");
    assert!(older > newest);
    for col in newest + 1..=rightmost {
        assert_eq!(
            grid.get(col, axis).unwrap().ch,
            '·',
            "history gap at column {col}"
        );
    }
}

#[test]
fn ceiling_clamps_to_full_blocks() {
    let mut model = chart_model();
    model.chart = vec![ChartBucket {
        gen_tps: Some(400.0),
        prompt_tps: Some(10_000.0),
    }];
    let grid = draw(&model, 240, 67);
    let axis = axis_row(&grid);
    let col = (0..grid.cols())
        .rev()
        .find(|c| grid.get(*c, axis - 1).unwrap().ch == '█')
        .expect("gen ceiling column");
    for dy in 1..=4 {
        assert_eq!(grid.get(col, axis - dy).unwrap().ch, '█', "gen row {dy}");
        assert_eq!(grid.get(col, axis + dy).unwrap().ch, '█', "prompt row {dy}");
    }
    assert_eq!(
        fg_sgr(grid.get(col, axis - 1).unwrap().fg),
        91,
        "gen ceiling is the top step"
    );
    assert_eq!(
        fg_sgr(grid.get(col, axis + 1).unwrap().fg),
        91,
        "prompt ceiling is the top step"
    );
}

#[test]
fn chart_golden_240x67_includes_the_band() {
    let generating = load("generating-480.json");
    let mut model = sample(WatchState::Generating);
    let fix = load("chart-240.json");
    model.in_title = title_line(&fix, "IN ");
    model.out_title = title_line(&fix, "OUT ");
    model.in_lines = region(&generating, "IN ", Some("OUT "));
    model.out_lines = region(&generating, "OUT ", None);
    model.chart = chart_story();
    assert_frame("chart-240.json", &fix, &model);
}

#[test]
fn chart_golden_240x67_in_eighths_mode() {
    let generating = load("generating-480.json");
    let mut model = sample(WatchState::Generating);
    let fix = load("chart-eighths-240.json");
    model.in_title = title_line(&fix, "IN ");
    model.out_title = title_line(&fix, "OUT ");
    model.in_lines = region(&generating, "IN ", Some("OUT "));
    model.out_lines = region(&generating, "OUT ", None);
    model.chart = chart_story();
    model.chart_glyphs = ChartGlyphs::Eighths;
    assert_frame("chart-eighths-240.json", &fix, &model);
}

#[test]
fn a_positive_rate_draws_at_least_one_half_cell() {
    let mut model = chart_model();
    model.chart = vec![ChartBucket {
        gen_tps: Some(1.0),
        prompt_tps: Some(1.0),
    }];
    let grid = draw(&model, 240, 67);
    let axis = axis_row(&grid);
    let col = (0..grid.cols())
        .rev()
        .find(|c| {
            let up = grid.get(*c, axis - 1).unwrap().ch;
            up == '▄' || up == '█'
        })
        .expect("gen half-cell");
    let up = grid.get(col, axis - 1).unwrap().ch;
    let down = grid.get(col, axis + 1).unwrap().ch;
    assert!(up == '▄' || up == '█', "gen {up:?}");
    assert!(down == '▀' || down == '█', "prompt {down:?}");
}

#[test]
fn ceiling_labels_do_not_overwrite_data() {
    let mut model = chart_model();
    model.prompt_ceiling = 100_000.0;
    model.chart = vec![
        ChartBucket {
            gen_tps: Some(250.0),
            prompt_tps: Some(100_000.0),
        };
        400
    ];
    let grid = draw(&model, 240, 67);
    let axis = axis_row(&grid);
    let bottom = timeline_rule(&grid) + chart_height_of(&grid);
    let bottom_text = row_string(&grid, bottom);
    assert!(
        bottom_text.contains("100000"),
        "prompt ceiling in the margin: {bottom_text}"
    );
    let is_bar =
        |col: u16, row: u16| matches!(grid.get(col, row).unwrap().ch, '█' | '▀' | '▄' | '▌' | '▐');
    for col in 2..8 {
        assert!(!is_bar(col, bottom), "margin col {col} holds a bar");
        assert!(!is_bar(col, axis - 1), "gen margin col {col} holds a bar");
    }
    let newest = (9..grid.cols())
        .find(|col| is_bar(*col, bottom))
        .expect("prompt bars");
    assert_eq!(newest, 9, "newest column is the first data column");
}

#[test]
fn chart_band_is_stable_until_the_bucket_value_changes() {
    let mut model = chart_model();
    model.chart = vec![ChartBucket {
        gen_tps: Some(54.2),
        prompt_tps: Some(0.0),
    }];
    let first = draw(&model, 240, 67);
    model.chart[0].gen_tps = Some(54.3);
    let same = draw(&model, 240, 67);
    let start = timeline_rule(&first) + 1;
    let height = chart_height_of(&first);
    for row in start..start + height {
        assert_eq!(
            row_string(&first, row),
            row_string(&same, row),
            "row {row} changed without a level change"
        );
    }
    model.chart[0].gen_tps = Some(250.0);
    let changed = draw(&model, 240, 67);
    let band_changed =
        (start..start + height).any(|row| row_string(&first, row) != row_string(&changed, row));
    assert!(band_changed, "ceiling rate should redraw the column");

    let t0 = Instant::now();
    let mut term = sized(240, 67);
    term.render(&first, t0).expect("first");
    term.out_mut().clear();
    term.render(&same, t0 + Duration::from_millis(10))
        .expect("same");
    assert_eq!(term.out(), b"", "an unchanged chart must not emit bytes");
}

/// The rate that fills `frac` of the chart band: five octaves under the ceiling.
fn rate_at(frac: f64, ceiling: f64) -> f64 {
    ceiling * 2f64.powf(5.0 * frac - 5.0)
}

const LOWER_EIGHTHS: [char; 6] = ['▁', '▂', '▃', '▅', '▆', '▇'];

fn chart_cells(grid: &llama_watch::tty::grid::Grid) -> Vec<(u16, u16)> {
    let start = timeline_rule(grid) + 1;
    let height = chart_height_of(grid);
    (start..start + height)
        .flat_map(|row| (0..grid.cols()).map(move |col| (col, row)))
        .collect()
}

#[test]
fn halves_mode_draws_only_eurlatgr_glyphs() {
    let mut model = chart_model();
    model.chart = chart_story();
    model.chart.push(ChartBucket {
        gen_tps: Some(250.0 * 9.0 / 32.0),
        prompt_tps: Some(1500.0 * 9.0 / 32.0),
    });
    let grid = draw(&model, 240, 67);
    for row in 0..grid.rows() {
        for col in 0..grid.cols() {
            let cell = grid.get(col, row).unwrap();
            assert!(
                is_console_char(cell.ch),
                "halves mode drew {:?} at ({col},{row})",
                cell.ch
            );
        }
    }
}

#[test]
fn eighths_mode_rises_in_lower_eighths_and_falls_in_inverse_video() {
    let mut model = chart_model();
    model.chart_glyphs = ChartGlyphs::Eighths;
    // 9/32 of each (log-scale) band: one full cell and one eighth each side.
    model.chart = vec![ChartBucket {
        gen_tps: Some(rate_at(9.0 / 32.0, 250.0)),
        prompt_tps: Some(rate_at(9.0 / 32.0, 1500.0)),
    }];
    let grid = draw(&model, 160, 40);
    let axis = axis_row(&grid);
    let col = (9..grid.cols())
        .find(|c| grid.get(*c, axis - 1).unwrap().ch == '█')
        .expect("gen column");
    let gen_full = grid.get(col, axis - 1).unwrap();
    let gen_tip = grid.get(col, axis - 2).unwrap();
    assert_eq!(gen_tip.ch, '▁', "gen tip");
    assert_eq!(gen_tip.fg, gen_full.fg, "gen tip keeps the bar colour");
    assert_eq!(gen_tip.bg, C16::Black);
    assert_eq!(grid.get(col, axis - 3).unwrap().ch, ' ');

    let prompt_full = grid.get(col, axis + 1).unwrap();
    let prompt_tip = grid.get(col, axis + 2).unwrap();
    assert_eq!(prompt_full.ch, '█', "prompt full cell");
    // Top 1/8 in the bar colour: the lower 7/8 drawn black on the colour.
    assert_eq!(prompt_tip.ch, '▇', "prompt tip");
    assert_eq!(prompt_tip.fg, C16::Black);
    assert_eq!(bg_sgr(prompt_tip.bg), 40 + fg_sgr(prompt_full.fg) % 10);
    assert_eq!(grid.get(col, axis + 3).unwrap().ch, ' ');
}

#[test]
fn eighths_mode_shows_a_tiny_rate_as_one_eighth() {
    let mut model = chart_model();
    model.chart_glyphs = ChartGlyphs::Eighths;
    model.chart = vec![ChartBucket {
        gen_tps: Some(0.01),
        prompt_tps: Some(0.01),
    }];
    let grid = draw(&model, 240, 67);
    let axis = axis_row(&grid);
    let col = (0..grid.cols())
        .find(|c| grid.get(*c, axis - 1).unwrap().ch == '▁')
        .expect("gen eighth");
    let down = grid.get(col, axis + 1).unwrap();
    assert_eq!((down.ch, down.fg), ('▇', C16::Black), "prompt eighth");
}

#[test]
fn eighths_mode_story_uses_eighths_and_stays_on_the_term_allowlist() {
    let mut model = chart_model();
    model.chart_glyphs = ChartGlyphs::Eighths;
    model.chart = chart_story();
    let grid = draw(&model, 240, 67);
    let mut eighths = 0;
    for (col, row) in chart_cells(&grid) {
        let ch = grid.get(col, row).unwrap().ch;
        assert!(
            (' '..='~').contains(&ch) || GLYPHS.contains(&ch),
            "{ch:?} at ({col},{row}) is not on the term allowlist"
        );
        if LOWER_EIGHTHS.contains(&ch) {
            eighths += 1;
        }
    }
    assert!(eighths > 0, "the story drew no eighth-blocks");
}

fn chart_story() -> Vec<ChartBucket> {
    let mut buckets = Vec::new();
    let push = |buckets: &mut Vec<ChartBucket>,
                n: usize,
                gen_tps: Option<f64>,
                prompt_tps: Option<f64>| {
        buckets.extend(std::iter::repeat_n(
            ChartBucket {
                gen_tps,
                prompt_tps,
            },
            n,
        ));
    };
    push(&mut buckets, 8, Some(0.0), Some(0.0));
    push(&mut buckets, 6, Some(0.0), Some(1800.0));
    push(&mut buckets, 4, None, None);
    push(&mut buckets, 20, Some(54.0), Some(0.0));
    push(&mut buckets, 5, Some(54.0), Some(2200.0));
    push(&mut buckets, 10, Some(54.0), Some(0.0));
    push(&mut buckets, 6, Some(0.0), Some(0.0));
    push(&mut buckets, 5, None, None);
    push(&mut buckets, 12, Some(54.0), Some(0.0));
    buckets
}

fn assert_blank_edges(grid: &llama_watch::tty::grid::Grid) {
    let last_col = grid.cols() - 1;
    let last_row = grid.rows() - 1;
    for col in 0..grid.cols() {
        assert_eq!(
            grid.get(col, last_row).unwrap().ch,
            ' ',
            "last row col {col}"
        );
    }
    for row in 0..grid.rows() {
        assert_eq!(
            grid.get(last_col, row).unwrap().ch,
            ' ',
            "last col row {row}"
        );
    }
}

#[allow(dead_code)]
fn content(grid: &llama_watch::tty::grid::Grid, row: u16) -> String {
    let mut text = String::new();
    for col in 2..grid.cols().saturating_sub(1) {
        text.push(grid.get(col, row).unwrap().ch);
    }
    text.trim_end().to_string()
}

fn col_of(grid: &llama_watch::tty::grid::Grid, row: u16, needle: &str) -> u16 {
    let text = row_string(grid, row);
    let byte = text
        .find(needle)
        .unwrap_or_else(|| panic!("missing {needle} on row {row}: {text}"));
    u16::try_from(text[..byte].chars().count()).expect("column")
}

fn row_string(grid: &llama_watch::tty::grid::Grid, row: u16) -> String {
    let mut text = String::new();
    for col in 0..grid.cols() {
        text.push(grid.get(col, row).unwrap().ch);
    }
    text.trim_end().to_string()
}

fn row_with(grid: &llama_watch::tty::grid::Grid, needle: &str) -> u16 {
    (0..grid.rows())
        .find(|row| row_string(grid, *row).contains(needle))
        .unwrap_or_else(|| panic!("missing {needle}"))
}

fn is_console_char(ch: char) -> bool {
    matches!(
        ch,
        ' '..='~'
            | '█'
            | '▌'
            | '▐'
            | '░'
            | '▒'
            | '▓'
            | '▀'
            | '▄'
            | '·'
            | '…'
            | '≈'
            // #90: typography the tty sanitiser lets through as itself, and
            // the distinct unknown-scalar placeholder.
            | '’'
            | '‘'
            | '“'
            | '”'
            | '–'
            | '—'
            | '•'
            | '\u{fffd}'
    )
}

fn load(name: &str) -> Fix {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/tty")
        .join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("parse {name}: {err}"))
}

fn dump_grid(grid: &llama_watch::tty::grid::Grid) -> String {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for row in 0..grid.rows() {
        let mut text = String::new();
        let mut runs: Vec<[u16; 3]> = Vec::new();
        for col in 0..grid.cols() {
            let cell = grid.get(col, row).expect("cell");
            text.push(cell.ch);
            let fg = fg_sgr(cell.fg);
            let bg = bg_sgr(cell.bg);
            if let Some(last) = runs.last_mut()
                && last[0] == fg
                && last[1] == bg
            {
                last[2] += 1;
                continue;
            }
            runs.push([fg, bg, 1]);
        }
        rows.push(text);
        cols.push(runs);
    }
    serde_json::to_string(&serde_json::json!({
        "w": grid.cols(),
        "h": grid.rows(),
        "rows": rows,
        "cols": cols,
    }))
    .expect("json")
}

#[test]
#[ignore = "run with --ignored to rewrite fixtures/tty goldens"]
fn dump_tty_goldens() {
    let generating = load("generating-480.json");
    let idle = load("idle-480.json");
    let inn = region(&generating, "IN ", Some("OUT "));
    let out_gen = region(&generating, "OUT ", None);
    let out_idle = region(&idle, "OUT ", None);
    let frames = [
        ("generating-480.json", WatchState::Generating, &out_gen),
        ("idle-480.json", WatchState::Ready, &out_idle),
        ("down-480.json", WatchState::AiDown, &out_idle),
        ("starting-480.json", WatchState::Starting, &out_gen),
        ("generating-240.json", WatchState::Generating, &out_gen),
        ("idle-240.json", WatchState::Ready, &out_idle),
        ("down-240.json", WatchState::AiDown, &out_idle),
        ("starting-240.json", WatchState::Starting, &out_gen),
    ];
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, state, out_lines) in frames {
        let old = load(name);
        let mut model = sample(state);
        model.in_title = title_line(&old, "IN ");
        model.out_title = title_line(&old, "OUT ");
        if state != WatchState::Starting {
            model.in_lines = inn.clone();
            model.out_lines = out_lines.clone();
        }
        let grid = draw(&model, old.w, old.h);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write golden");
    }
    let old = load("generating-240.json");
    let mut model = sample(WatchState::Generating);
    model.in_title = title_line(&old, "IN ");
    model.out_title = title_line(&old, "OUT ");
    model.in_lines = inn;
    model.out_lines = out_gen;
    model.chart = chart_story();
    let grid = draw(&model, 240, 67);
    std::fs::write(dir.join("chart-240.json"), dump_grid(&grid)).expect("write chart golden");
    model.chart_glyphs = ChartGlyphs::Eighths;
    let grid = draw(&model, 240, 67);
    std::fs::write(dir.join("chart-eighths-240.json"), dump_grid(&grid))
        .expect("write eighths chart golden");
}

fn title_row(fix: &Fix, key: &str) -> usize {
    if let Some(row) = side_by_side_title(fix) {
        return row;
    }
    fix.rows
        .iter()
        .position(|row| {
            let trimmed = row.trim_start();
            trimmed.starts_with(key)
        })
        .unwrap_or_else(|| panic!("missing {key} title"))
}

fn title_line(fix: &Fix, key: &str) -> String {
    if let Some(row) = side_by_side_title(fix) {
        let (left, right) = panes(&fix.rows[row], fix.w);
        return if key.starts_with("IN") { left } else { right };
    }
    line(fix, title_row(fix, key))
}

/// Title row of the side-by-side IN/OUT band, when this frame has one.
fn side_by_side_title(fix: &Fix) -> Option<usize> {
    let mid = usize::from(fix.w / 2);
    fix.rows.iter().position(|row| {
        let chars: Vec<char> = row.chars().collect();
        chars.get(mid) == Some(&'|') && row.trim_start().starts_with("IN")
    })
}

fn panes(row: &str, cols: u16) -> (String, String) {
    let chars: Vec<char> = row.chars().collect();
    let mid = usize::from(cols / 2);
    let take = |range: std::ops::Range<usize>| {
        range
            .filter_map(|i| chars.get(i).copied())
            .collect::<String>()
            .trim_end()
            .replace('█', "")
    };
    (take(2..mid), take(mid + 1..chars.len()))
}

fn region(fix: &Fix, start_key: &str, end_key: Option<&str>) -> Vec<String> {
    if let Some(title) = side_by_side_title(fix) {
        let left = start_key.starts_with("IN");
        return pane_lines(fix, title, left);
    }
    let start = title_row(fix, start_key).saturating_add(1);
    let mut end = match end_key {
        Some(key) => title_row(fix, key),
        None => usize::from(fix.h.saturating_sub(3)),
    };
    while end > start && fix.rows[end - 1].trim().is_empty() {
        end -= 1;
    }
    if end <= start {
        return Vec::new();
    }
    window(fix, start, end - 1)
}

/// Text of one side-by-side pane, without the top and bottom padding and
/// without the AI-down banner (yellow), which the layout paints itself.
fn pane_lines(fix: &Fix, title: usize, left: bool) -> Vec<String> {
    let end = usize::from(fix.h.saturating_sub(3));
    let mut lines = Vec::new();
    for (idx, row) in fix.rows.iter().enumerate().take(end).skip(title + 1) {
        if is_fix_rule(row) || is_banner(fix, idx) {
            continue;
        }
        let (pane_left, pane_right) = panes(row, fix.w);
        lines.push(if left { pane_left } else { pane_right });
    }
    while lines.first().is_some_and(|line| line.trim().is_empty()) {
        lines.remove(0);
    }
    while lines.last().is_some_and(|line| line.trim().is_empty()) {
        lines.pop();
    }
    lines
}

fn is_fix_rule(row: &str) -> bool {
    let mut chars = row.chars();
    chars.next() == Some(' ') && chars.next() == Some('-') && chars.next() == Some('-')
}

fn is_banner(fix: &Fix, row: usize) -> bool {
    fix.cols
        .get(row)
        .is_some_and(|runs| runs.iter().any(|run| run[1] == 43))
}

fn window(fix: &Fix, start: usize, end: usize) -> Vec<String> {
    // Screen rows include the two-column margin and, on a live line, the
    // cursor. The layout adds both, so the model receives the raw tail.
    fix.rows[start..=end]
        .iter()
        .map(|row| {
            let trimmed = row.trim_end().replace('█', "");
            trimmed.chars().skip(2).collect()
        })
        .collect()
}

fn line(fix: &Fix, row: usize) -> String {
    fix.rows[row].trim_end().to_string()
}

fn sample(state: WatchState) -> TtyModel {
    let (model_name, slots_line, swap_line, cool, cpu_c, gpu_c) = match state {
        WatchState::Generating | WatchState::Ready => (
            "Qwen 35B",
            if state == WatchState::Generating {
                "1/1 busy"
            } else {
                "1/1 idle"
            },
            "none (last 16:02)",
            Some(if state == WatchState::Generating {
                40
            } else {
                34
            }),
            Some(if state == WatchState::Generating {
                79
            } else {
                46
            }),
            Some(if state == WatchState::Generating {
                80
            } else {
                38
            }),
        ),
        WatchState::AiDown => ("--", "--", "--", Some(34), Some(46), Some(38)),
        // A host without llama-swap or NVIDIA: no GPU temperature either.
        WatchState::NoLlama => ("--", "--", "--", Some(33), Some(45), None),
        WatchState::Starting => ("...", "...", "...", None, None, None),
    };
    let (cpu, gpu, vram_u, vram_t, mem_u, mem_t, power, limit, load) = match state {
        WatchState::Generating => (
            Some(41.0),
            Some(97.0),
            Some(22.8),
            Some(24.0),
            Some(38.1),
            Some(62.6),
            Some(312.0),
            Some(350.0),
            Some(70.0),
        ),
        WatchState::Ready => (
            Some(3.0),
            Some(0.0),
            Some(22.8),
            Some(24.0),
            Some(38.1),
            Some(62.6),
            Some(31.0),
            Some(350.0),
            Some(6.0),
        ),
        WatchState::AiDown => (
            Some(2.0),
            Some(0.0),
            Some(0.4),
            Some(24.0),
            Some(11.2),
            Some(62.6),
            Some(21.0),
            Some(350.0),
            Some(8.0),
        ),
        WatchState::NoLlama => (
            Some(4.0),
            None,
            None,
            None,
            Some(11.2),
            Some(62.6),
            None,
            None,
            Some(9.0),
        ),
        WatchState::Starting => (None, None, None, None, None, None, None, None, None),
    };
    let (gen_tps, prompt_tps, prompt_last) = match state {
        WatchState::Generating => (Some(54.2), Some(0.0), Some(1212)),
        WatchState::Ready => (Some(0.0), Some(0.0), Some(1187)),
        WatchState::AiDown | WatchState::Starting | WatchState::NoLlama => (None, None, None),
    };
    let slots = match state {
        WatchState::Generating => vec![Slot {
            id: 0,
            generating: true,
            done: 91_204,
            cached: 0,
            open: false,
            done_known: true,
            total: 91_204,
            decoded: 612,
            ctx_prompt: Some(91_204),
            n_ctx: Some(262_144),
            ctx_history: Vec::new(),
        }],
        WatchState::Ready => vec![Slot {
            id: 0,
            generating: false,
            done: 0,
            cached: 0,
            open: false,
            done_known: true,
            total: 0,
            decoded: 0,
            ctx_prompt: None,
            n_ctx: None,
            ctx_history: Vec::new(),
        }],
        WatchState::AiDown | WatchState::Starting | WatchState::NoLlama => Vec::new(),
    };
    let requests = match state {
        WatchState::Generating => requests(true),
        WatchState::Ready | WatchState::AiDown => requests(false),
        WatchState::Starting | WatchState::NoLlama => Vec::new(),
    };
    let (snapshot, age, errors, uptime, down_since) = match state {
        WatchState::Generating => (Some(48_213), "0.1s", 0, "3d04h12m", ""),
        WatchState::Ready => (Some(51_004), "0.1s", 0, "3d04h12m", ""),
        WatchState::AiDown => (Some(51_920), "0.1s", 143, "3d04h12m", "18:44:51 (2m23s)"),
        WatchState::Starting => (None, "", 0, "0s", ""),
        WatchState::NoLlama => (Some(51_004), "0.1s", 1, "3d04h12m", ""),
    };
    TtyModel {
        state,
        host: "AIBOX".to_string(),
        model_name: model_name.to_string(),
        // #33: the engine always leads the detail; `engine --` with no model.
        model_detail: match state {
            WatchState::Generating | WatchState::Ready => "llama.cpp",
            WatchState::AiDown | WatchState::Starting | WatchState::NoLlama => "engine --",
        }
        .to_string(),
        model_stuck: false,
        model_loading: false,
        model_stopping: false,
        gpu_busy: false,
        slots_line: slots_line.to_string(),
        swap_line: swap_line.to_string(),
        cool_c: cool,
        cpu_c,
        gpu_c,
        clock: "2026-09-23 18:47:14".to_string(),
        cpu_pct: cpu,
        cpu_cores: if state == WatchState::Starting {
            None
        } else {
            Some(16)
        },
        gpu_pct: gpu,
        vram_used_gb: vram_u,
        vram_total_gb: vram_t,
        mem_used_gb: mem_u,
        mem_total_gb: mem_t,
        power_w: power,
        power_limit_w: limit,
        load_pct: load,
        activity_pct: activity_pct_for(state),
        activity_src: activity_pct_for(state).map(|_| LoadSource::Gpu),
        activity_w: activity_w_for(state),
        gen_tps,
        prompt_tps,
        prompt_last,
        gen_last: None,
        gen_ceiling: 250.0,
        prompt_ceiling: 1500.0,
        slots,
        backend_lines: Vec::new(),
        text_note: String::new(),
        requests,
        in_title: String::new(),
        out_title: String::new(),
        in_lines: Vec::new(),
        out_lines: Vec::new(),
        out_shown: 0,
        replay_frame: None,
        show_text: true,
        down_since: down_since.to_string(),
        health: health(state),
        snapshot,
        snapshot_age: age.to_string(),
        errors,
        uptime: uptime.to_string(),
        chart: Vec::new(),
        chart_bucket_s: 2,
        chart_glyphs: ChartGlyphs::Halves,
        bar_segments: false,
        fans: None,
        temps: None,
        ctx_history_h: 6,
        setup: match state {
            WatchState::Generating | WatchState::Ready => Some(qwen_setup()),
            WatchState::AiDown | WatchState::Starting | WatchState::NoLlama => None,
        },
        job: None,
        job_note: None,
        model_id: String::new(),
        claims: Vec::new(),
        gpu: None,
        displaced: Vec::new(),
        now: UNIX_EPOCH,
        disk_read_bps: None,
        disk_write_bps: None,
        net_rx_bps: None,
        net_tx_bps: None,
        net_ceiling_bps: None,
        swap_used_gb: None,
        swap_total_gb: None,
        io_psi: None,
        mem_psi: None,
        disk_ceiling_bps: 7e9,
        tz_offset_s: 0,
    }
}

/// An invented llama.cpp launch for the sample model (#52).
const QWEN_CMD: &str = "/opt/llama.cpp/bin/llama-server --port 5810 -m /models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 99 -fa on -c 262144 -ctk q8_0 -ctv q8_0 -ncmoe 16 --spec-type draft-mtp --spec-draft-n-max 3 --reasoning-budget 24000 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0";

/// SETUP for `cmd` through the built-in rules, as llama-watch builds it.
fn setup_of(
    id: &str,
    name: &str,
    cmd: &str,
    backend: Backend,
    detail: Option<&ModelDetail>,
    info: Option<&BackendInfo>,
) -> SetupView {
    let rules = Rules::builtin();
    let live = LiveCtx {
        backend,
        detail,
        info,
        engine: None,
    };
    SetupView {
        id: id.to_owned(),
        name: name.to_owned(),
        more: 0,
        rows: rules.rows(&rules.extract(cmd), &live),
    }
}

fn qwen_setup() -> SetupView {
    setup_of(
        "qwen3.6-35b-a3b",
        "Qwen 35B",
        QWEN_CMD,
        Backend::LlamaCpp,
        None,
        None,
    )
}

fn activity_pct_for(state: WatchState) -> Option<f64> {
    match state {
        WatchState::Generating => Some(42.0),
        WatchState::Ready => Some(8.0),
        WatchState::AiDown => Some(5.0),
        WatchState::NoLlama => Some(4.0),
        WatchState::Starting => None,
    }
}

fn activity_w_for(state: WatchState) -> Option<f64> {
    match state {
        WatchState::Generating => Some(312.0),
        WatchState::Ready => Some(46.0),
        WatchState::AiDown => Some(21.0),
        WatchState::NoLlama => None,
        WatchState::Starting => None,
    }
}

#[test]
fn no_llama_and_no_gpu_draw_dashes_and_a_quiet_note() {
    let model = sample(WatchState::NoLlama);
    for (cols, rows) in [(160, 48), (240, 67)] {
        let grid = draw(&model, cols, rows);
        let header = full_row(&grid, 0);
        assert!(header.contains("NO LLAMA"), "{header}");
        assert!(!header.contains("AI DOWN"), "{header}");
        for label in ["GPU", "VRAM", "POWER"] {
            let row = meter_row(&grid, label);
            let text = row_string(&grid, row);
            assert!(
                text.contains("--"),
                "{label} at {cols}x{rows} should dash: {text}"
            );
        }
        let note = row_with(&grid, "[llama] enabled = false");
        // The note is dim text, not the yellow AI DOWN band.
        for col in 0..cols {
            let cell = grid.get(col, note).expect("cell");
            assert_ne!(cell.bg, C16::Yellow, "yellow band at {col},{note}");
        }
        assert_blank_edges(&grid);
    }
}

/// #165: ACT and LOAD share the first gauge row. ACT keeps its source; the
/// device watts are POWER's.
#[test]
fn act_and_load_share_the_first_gauge_row_and_act_names_its_source() {
    let model = sample(WatchState::Generating);
    let grid = draw(&model, 240, 67);
    let row = meter_row(&grid, "ACT");
    assert_eq!(meter_row(&grid, "LOAD"), row, "LOAD is beside ACT");
    let text = row_string(&grid, row);
    assert!(text.contains("ACT       42 % gpu"), "{text}");
    assert!(text.contains("LOAD          70 %"), "{text}");
    let power = row_string(&grid, meter_row(&grid, "POWER"));
    assert!(power.contains("312/350 W"), "{power}");
}

#[test]
fn act_value_names_each_source_and_keeps_its_field() {
    let cases: [(f64, LoadSource, Option<f64>, &str); 5] = [
        (78.0, LoadSource::Gpu, Some(340.0), "78 % gpu"),
        (100.0, LoadSource::Cpu, Some(230.0), "100 % cpu"),
        (5.0, LoadSource::Cpu, Some(31.0), "5 % cpu"),
        (12.0, LoadSource::Util, None, "12 % util"),
        (99.6, LoadSource::Gpu, Some(349.0), "100 % gpu"),
    ];
    for (pct, src, watts, want) in cases {
        let mut model = sample(WatchState::Generating);
        model.activity_pct = Some(pct);
        model.activity_src = Some(src);
        model.activity_w = watts;
        let grid = draw(&model, 240, 67);
        let (row, col) = label_hit(&grid, "ACT");
        let text: String = (col..col + 19)
            .map(|c| grid.get(c, row).unwrap().ch)
            .collect();
        assert_eq!(
            text,
            format!("ACT{want:>15} "),
            "{}",
            row_string(&grid, row)
        );
    }
}

/// T54: activity is nominal-relative and reads up to 125. Over 100 the text
/// keeps the real number, the bar is full, and it turns red.
#[test]
fn activity_over_100_is_a_full_bar_with_a_hot_top_step_and_the_real_number() {
    let bar = |pct: f64| {
        let mut model = sample(WatchState::Generating);
        model.activity_pct = Some(pct);
        model.activity_src = Some(LoadSource::Gpu);
        model.activity_w = Some(340.0);
        let grid = draw(&model, 240, 67);
        let row = meter_row(&grid, "ACT");
        let full = row_string(&grid, row);
        let at = full.find("ACT ").expect("ACT");
        let text: String = full[at..].chars().take_while(|ch| ch.is_ascii()).collect();
        let bar = meter_bar_col(&grid, row, "ACT");
        let cells: Vec<_> = (bar..bar + 36)
            .map(|col| grid.get(col, row).expect("cell"))
            .filter(|cell| matches!(cell.ch, '█' | '░' | '▉'..='▏' | '▄' | '▇' | SEG))
            .collect();
        (text, cells)
    };
    let (full_text, full) = bar(100.0);
    let (hot_text, hot) = bar(118.0);
    let (pinned_text, pinned) = bar(125.0);
    assert_eq!(
        hot_text.split_whitespace().collect::<Vec<_>>(),
        ["ACT", "118", "%", "gpu"],
        "{hot_text}"
    );
    assert!(full_text.contains("100 %"), "{full_text}");
    assert!(pinned_text.contains("125 %"), "{pinned_text}");
    assert!(!hot.is_empty(), "the bar is drawn");
    assert_eq!(
        hot.iter().map(|cell| cell.ch).collect::<String>(),
        full.iter().map(|cell| cell.ch).collect::<String>(),
        "over 100 the bar is as full as at 100, not wider"
    );
    assert_eq!(
        pinned.iter().map(|cell| cell.ch).collect::<String>(),
        full.iter().map(|cell| cell.ch).collect::<String>()
    );
    let fgs = |cells: &[llama_watch::tty::grid::Cell]| -> Vec<C16> {
        cells
            .iter()
            .filter(|cell| cell.ch != '░')
            .map(|cell| cell.fg)
            .collect()
    };
    assert_eq!(steps_of(&fgs(&full)), STEPS, "100 % is every step");
    let mut hot_steps = STEPS.to_vec();
    hot_steps[4] = C16::BrightWhite;
    assert_eq!(
        steps_of(&fgs(&hot)),
        hot_steps,
        "over 100 the top step turns white-hot"
    );
    assert_eq!(steps_of(&fgs(&pinned)), hot_steps);
}

fn requests(live_head: bool) -> Vec<Activity> {
    let mut rows = Vec::new();
    if live_head {
        rows.push(activity(
            true,
            4822,
            "18:47:01",
            "192.0.2.83",
            "Qwen 35B",
            91_204,
            88_960,
            612,
            1212.0,
            54.2,
            "13.2s",
            false,
        ));
    }
    rows.extend([
        activity(
            false,
            4821,
            "18:42:47",
            "192.0.2.83",
            "Qwen 35B",
            90_511,
            86_016,
            1904,
            1187.0,
            55.8,
            "38.0s",
            false,
        ),
        activity(
            false,
            4820,
            "18:41:05",
            "127.0.0.1",
            "Qwen 35B",
            2210,
            0,
            388,
            1604.0,
            61.3,
            "7.7s",
            false,
        ),
        activity(
            false,
            4819,
            "18:30:12",
            "192.0.2.83",
            "Qwen 35B",
            88_930,
            0,
            2210,
            217.0,
            52.9,
            "451.7s",
            false,
        ),
        activity(
            false,
            4818,
            "18:22:40",
            "192.0.2.51",
            "Gemma 4B",
            512,
            0,
            96,
            2380.0,
            141.0,
            "0.9s",
            false,
        ),
        activity(
            false,
            4817,
            "18:22:31",
            "192.0.2.51",
            "Gemma 4B",
            498,
            0,
            211,
            2295.0,
            138.6,
            "1.7s",
            false,
        ),
        activity(
            false,
            4816,
            "17:58:03",
            "192.0.2.83",
            "Qwen 35B",
            71_022,
            70_144,
            740,
            1330.0,
            56.1,
            "14.1s",
            false,
        ),
        activity(
            false,
            4815,
            "17:55:48",
            "192.0.2.83",
            "Qwen 35B",
            70_410,
            0,
            1502,
            238.0,
            54.7,
            "323.3s",
            live_head,
        ),
    ]);
    if !live_head {
        rows.push(activity(
            false,
            4814,
            "17:51:20",
            "192.0.2.83",
            "Qwen 35B",
            69_980,
            0,
            1200,
            240.0,
            55.0,
            "312.5s",
            false,
        ));
    }
    rows
}

#[allow(clippy::too_many_arguments)]
fn activity(
    live: bool,
    id: u32,
    time: &str,
    source: &str,
    model: &str,
    input_tok: u64,
    cached_tok: u64,
    output_tok: u64,
    prompt_tps: f64,
    gen_tps: f64,
    dur: &str,
    err: bool,
) -> Activity {
    Activity {
        live,
        id,
        time: time.to_string(),
        source: source.to_string(),
        model: model.to_string(),
        input_tok,
        cached_tok,
        output_tok,
        prompt_tps: Some(prompt_tps),
        gen_tps: Some(gen_tps),
        prompt_measured: false,
        gen_measured: false,
        dur: dur.to_string(),
        err,
        // #75: the context the bar is drawn against.
        n_ctx: match model {
            "Qwen 35B" => Some(262_144),
            "Gemma 4B" | "Gemma 27B" => Some(131_072),
            _ => None,
        },
        ..Activity::default()
    }
}

fn health(state: WatchState) -> Vec<HealthSeg> {
    let seg = |name: &str, status: HealthStatus, note: &str| HealthSeg {
        name: name.to_string(),
        status,
        note: note.to_string(),
        word_cells: 0,
        note_cells: 0,
    };
    match state {
        WatchState::Generating => vec![
            seg("llama-swap", HealthStatus::Ok, "3ms"),
            seg("/running", HealthStatus::Ok, ""),
            seg("/slots", HealthStatus::Ok, "14ms 1Hz"),
            seg("metrics", HealthStatus::Ok, ""),
            seg("activity", HealthStatus::Ok, ""),
            seg("nvml", HealthStatus::Ok, ""),
            seg("hwmon", HealthStatus::Ok, ""),
            seg("proc", HealthStatus::Ok, ""),
        ],
        WatchState::Ready => vec![
            seg("llama-swap", HealthStatus::Ok, "2ms"),
            seg("/running", HealthStatus::Ok, ""),
            seg("/slots", HealthStatus::Idle, ""),
            seg("metrics", HealthStatus::Ok, ""),
            seg("activity", HealthStatus::Ok, ""),
            seg("nvml", HealthStatus::Ok, ""),
            seg("hwmon", HealthStatus::Ok, ""),
            seg("proc", HealthStatus::Ok, ""),
        ],
        WatchState::AiDown => vec![
            seg("llama-swap", HealthStatus::Down, "refused"),
            seg("/running", HealthStatus::Down, ""),
            seg("/slots", HealthStatus::Absent, ""),
            seg("metrics", HealthStatus::Absent, ""),
            seg("activity", HealthStatus::Absent, ""),
            seg("nvml", HealthStatus::Ok, ""),
            seg("hwmon", HealthStatus::Ok, ""),
            seg("proc", HealthStatus::Ok, ""),
        ],
        WatchState::NoLlama => vec![
            seg("llama-swap", HealthStatus::Absent, "off"),
            seg("/running", HealthStatus::Absent, ""),
            seg("/slots", HealthStatus::Absent, ""),
            seg("metrics", HealthStatus::Absent, ""),
            seg("activity", HealthStatus::Absent, ""),
            seg("nvml", HealthStatus::Down, ""),
            seg("hwmon", HealthStatus::Ok, ""),
            seg("proc", HealthStatus::Ok, ""),
        ],
        WatchState::Starting => vec![
            seg("llama-swap", HealthStatus::Pending, ""),
            seg("/running", HealthStatus::Pending, ""),
            seg("/slots", HealthStatus::Pending, ""),
            seg("metrics", HealthStatus::Pending, ""),
            seg("activity", HealthStatus::Pending, ""),
            seg("nvml", HealthStatus::Pending, ""),
            seg("hwmon", HealthStatus::Pending, ""),
            seg("proc", HealthStatus::Pending, ""),
        ],
    }
}

fn expand(runs: &[[u16; 3]], width: usize) -> Vec<(u16, u16)> {
    let mut out = Vec::with_capacity(width);
    for run in runs {
        for _ in 0..run[2] {
            out.push((run[0], run[1]));
        }
    }
    assert_eq!(out.len(), width, "colour run does not cover the row");
    out
}

fn fg_sgr(colour: C16) -> u16 {
    match colour {
        C16::Black => 30,
        C16::Red => 31,
        C16::Green => 32,
        C16::Yellow => 33,
        C16::Blue => 34,
        C16::Magenta => 35,
        C16::Cyan => 36,
        C16::White => 37,
        C16::BrightBlack => 90,
        C16::BrightRed => 91,
        C16::BrightGreen => 92,
        C16::BrightYellow => 93,
        C16::BrightBlue => 94,
        C16::BrightMagenta => 95,
        C16::BrightCyan => 96,
        C16::BrightWhite => 97,
    }
}

fn bg_sgr(colour: C16) -> u16 {
    match colour {
        C16::Black | C16::BrightBlack => 40,
        C16::Red | C16::BrightRed => 41,
        C16::Green | C16::BrightGreen => 42,
        C16::Yellow | C16::BrightYellow => 43,
        C16::Blue | C16::BrightBlue => 44,
        C16::Magenta | C16::BrightMagenta => 45,
        C16::Cyan | C16::BrightCyan => 46,
        C16::White | C16::BrightWhite => 47,
    }
}

// ---- T45: `tty.show_text = false` ----

/// Every size T45 must fit, with the RECENT rows and trailing blank rows the
/// text-off allocation gives it (one slot, 40 requests on hand).
/// `(cols, rows, RECENT rows, blank rows)` with text off. Since #52 the
/// SETUP block under the meters takes two rows from RECENT at 160x48 and
/// three from 60 rows; 4K has the rows to spare.
const TEXT_OFF_SIZES: [(u16, u16, u16, u16); 4] = [
    (160, 48, 11, 0),
    (240, 67, 29, 0),
    (286, 60, 22, 0),
    (480, 135, 32, 53),
];

fn many_requests(n: u32) -> Vec<Activity> {
    (0..n)
        .map(|i| {
            let secs = 3_000 - i * 37;
            activity(
                i == 0,
                4_900 - i,
                &format!("18:{:02}:{:02}", (secs / 60) % 60, secs % 60),
                "192.0.2.83",
                if i % 3 == 0 { "Qwen 35B" } else { "Gemma 27B" },
                u64::from(1_000 + i * 131),
                u64::from(i * 97),
                u64::from(200 + i * 13),
                800.0 + f64::from(i) * 11.0,
                30.0 + f64::from(i % 7) * 4.5,
                &format!("{}.{}s", 2 + i % 9, i % 10),
                i % 11 == 5,
            )
        })
        .collect()
}

fn text_off_model(state: WatchState) -> TtyModel {
    let mut model = sample(state);
    model.show_text = false;
    // Even if a caller hands text over, a text-off frame must not draw it.
    model.in_title = "IN   prompt tail".to_string();
    model.out_title = "OUT  live".to_string();
    model.in_lines = vec!["LEAKED-PROMPT".to_string()];
    model.out_lines = vec!["LEAKED-OUTPUT".to_string()];
    if state != WatchState::Starting && state != WatchState::NoLlama {
        model.requests = many_requests(40);
    }
    model.chart = chart_story();
    model
}

#[test]
fn text_off_gives_the_in_out_rows_to_a_13_row_chart_and_recent() {
    for (cols, rows, _, _) in TEXT_OFF_SIZES {
        let at = format!("{cols}x{rows}");
        let grid = draw(&text_off_model(WatchState::Generating), cols, rows);
        let all = whole(&grid);
        for gone in ["prompt tail", "OUT  live", "LEAKED"] {
            assert!(!all.contains(gone), "{at}: {gone} drawn");
        }
        assert!(
            recent_rows_shown(&grid) >= 7,
            "{at} recent {}",
            recent_rows_shown(&grid)
        );
        assert!(
            chart_height_of(&grid) >= 9,
            "{at} chart {}",
            chart_height_of(&grid)
        );
        assert!(row_string(&grid, axis_row(&grid)).contains("now"), "{at}");
        assert_blank_edges(&grid);
    }
}

#[test]
fn text_off_short_of_rows_keeps_recent_and_shrinks_the_chart() {
    let mut model = text_off_model(WatchState::Generating);
    model.slots = (0..16)
        .map(|id| Slot {
            id,
            generating: true,
            done: 10,
            cached: 0,
            open: false,
            done_known: true,
            total: 20,
            decoded: 5,
            ctx_prompt: None,
            n_ctx: None,
            ctx_history: Vec::new(),
        })
        .collect();
    let grid = draw(&model, 160, 48);
    // Extra slots stay inside the top band. RECENT keeps the waterfall's 7.
    assert_eq!(recent_rows_shown(&grid), 7, "{}", whole(&grid));
    assert!(chart_height_of(&grid) >= 9, "{}", chart_height_of(&grid));
    assert!(row_string(&grid, axis_row(&grid)).contains("now"));
}

#[test]
fn text_off_header_tag_is_small_and_dim() {
    for state in [
        WatchState::Generating,
        WatchState::Ready,
        WatchState::AiDown,
        WatchState::Starting,
        WatchState::NoLlama,
    ] {
        for (cols, rows, _, _) in TEXT_OFF_SIZES {
            let grid = draw(&text_off_model(state), cols, rows);
            let top = row_string(&grid, 0);
            let at = col_of(&grid, 0, "text off");
            for col in at..at + 8 {
                let cell = grid.get(col, 0).unwrap();
                assert_eq!(cell.fg, C16::BrightBlack, "{state:?} {cols}x{rows}: {top}");
                assert_eq!(cell.bg, C16::Black);
            }
            assert!(top.contains("swap"), "{top}");
            assert!(top.contains("2026-09-23 18:47:14"), "{top}");
            assert!(col_of(&grid, 0, "swap") < at, "{top}");
        }
    }
    let on = draw(&sample(WatchState::Generating), 240, 67);
    assert!(!row_string(&on, 0).contains("text off"));
}

#[test]
fn text_off_still_says_why_llama_is_down_or_off() {
    let grid = draw(&text_off_model(WatchState::AiDown), 480, 135);
    let row = row_with(&grid, "llama-swap unreachable since 18:44:51");
    assert!(!row_string(&grid, row).contains("text kept"));
    let grid = draw(&text_off_model(WatchState::AiDown), 240, 67);
    row_with(&grid, "llama-swap unreachable since 18:44:51");
    let grid = draw(&text_off_model(WatchState::NoLlama), 240, 67);
    row_with(&grid, "[llama] enabled = false");
    let grid = draw(&text_off_model(WatchState::Starting), 240, 67);
    let all: String = (0..grid.rows())
        .map(|row| row_string(&grid, row) + "\n")
        .collect();
    assert!(!all.contains("llama text appears"), "{all}");
    row_with(&grid, "waiting for llama-swap");
}

const TEXT_OFF_GOLDENS: [(&str, u16, u16); 2] = [
    ("text-off-240.json", 240, 67),
    ("text-off-286.json", 286, 60),
];

#[test]
fn text_off_goldens_match_character_and_colour() {
    for (name, _, _) in TEXT_OFF_GOLDENS {
        let fix = load(name);
        assert_frame(name, &fix, &text_off_model(WatchState::Generating));
    }
}

#[test]
#[ignore = "run with --ignored to write the T45 text-off goldens"]
fn dump_text_off_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, cols, rows) in TEXT_OFF_GOLDENS {
        let grid = draw(&text_off_model(WatchState::Generating), cols, rows);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write text-off golden");
    }
}

#[test]
fn header_shows_the_full_model_name_and_detail() {
    let mut model = sample(WatchState::Ready);
    model.model_name = "Ternary Bonsai 2 27B".to_string();
    model.model_detail = "256k · kv q8 · UD-Q4_K_M · moe 16".to_string();
    for cols in [160, 240] {
        let grid = draw(&model, cols, 67);
        let header = row_string(&grid, 0);
        assert!(
            header.contains("model Ternary Bonsai 2 27B  256k · kv q8 · UD-Q4_K_M · moe 16"),
            "{cols}: {header}"
        );
        assert!(header.contains("slots"), "{cols}: {header}");
        let start = header.find("256k").expect("detail");
        let col = header[..start].chars().count();
        let cell = grid.get(u16::try_from(col).expect("col"), 0).expect("cell");
        assert_eq!(cell.fg, C16::BrightBlack, "detail is grey");
    }
}

/// T49: IN fills every row it has with the tail of the cleaned prompt, and
/// the role labels are dim.
#[test]
fn in_fills_its_rows_with_the_tail_and_dims_role_labels() {
    for (cols, rows) in [(160u16, 48u16), (240, 67), (480, 135)] {
        let mut model = sample(WatchState::Generating);
        let mut text = String::new();
        for turn in 0..60 {
            let role = ["user", "assistant", "tool"][turn % 3];
            text.push_str(&format!("-- {role} --\nturn {turn}\n"));
        }
        text.push_str("the last real line");
        model.in_lines = vec![text];
        model.out_lines = vec!["out".to_string()];
        model.in_title = "IN  T49-IN-TITLE".to_string();
        model.out_title = "OUT  T49-OUT-TITLE".to_string();
        let grid = draw(&model, cols, rows);
        let title = row_with(&grid, "T49-IN-TITLE");
        let lines = left_rows(&grid, title, grid.rows() - title);
        assert!(
            lines.iter().any(|line| line.contains("the last real line")),
            "{cols}x{rows} {lines:?}"
        );
        assert!(
            lines.iter().any(|line| line.contains("-- user --")),
            "{cols}x{rows} no role label"
        );
    }
}

// ---- T52: FANS panel ------------------------------------------------------

use llama_watch::sources::fans::{FanPanel, FanReading};

fn fan(
    channel: u32,
    label: &str,
    rpm: Option<u32>,
    pwm: Option<u8>,
    mode: Option<u32>,
) -> FanReading {
    FanReading {
        chip: "nct6798".to_string(),
        channel,
        label: label.to_string(),
        rpm,
        pwm,
        mode,
    }
}

/// Reference board: fans 2, 3, 5, 6, all SmartFan (`pwmN_enable` = 5).
fn ref_fans() -> FanPanel {
    FanPanel {
        chip: "nct6798".to_string(),
        present: true,
        fans: vec![
            fan(2, "front1", Some(1939), Some(224), Some(5)),
            fan(3, "front2", Some(1877), Some(224), Some(5)),
            fan(5, "rear", Some(3026), Some(162), Some(5)),
            fan(6, "top", Some(1272), Some(255), Some(5)),
        ],
    }
}

/// The generating golden model with the reference fans switched on.
fn fans_model() -> TtyModel {
    let generating = load("generating-480.json");
    let mut model = sample(WatchState::Generating);
    model.in_title = title_line(&generating, "IN ");
    model.out_title = title_line(&generating, "OUT ");
    model.in_lines = region(&generating, "IN ", Some("OUT "));
    model.out_lines = region(&generating, "OUT ", None);
    model.chart = chart_story();
    model.fans = Some(ref_fans());
    model
}

/// 192x60 is the 10x18 font on a 1920x1080 screen (#73).
const FANS_GOLDENS: [(&str, u16, u16); 3] = [
    ("fans-192.json", 192, 60),
    ("fans-240.json", 240, 67),
    ("fans-286.json", 286, 60),
];

#[test]
fn fans_goldens_match_character_and_colour() {
    for (name, _, _) in FANS_GOLDENS {
        let fix = load(name);
        assert_frame(name, &fix, &fans_model());
    }
}

#[test]
#[ignore = "run with --ignored to write the T52 fans goldens"]
fn dump_fans_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, cols, rows) in FANS_GOLDENS {
        let grid = draw(&fans_model(), cols, rows);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write fans golden");
    }
}

fn fans_disabled_frames() -> Vec<(u16, u16)> {
    vec![(160, 48), (240, 67), (286, 60), (480, 135)]
}

#[test]
fn fans_off_draws_no_fans_panel() {
    for (cols, rows) in fans_disabled_frames() {
        for model in [
            sample(WatchState::Generating),
            text_off_model(WatchState::Generating),
        ] {
            let grid = draw(&model, cols, rows);
            let all: String = (0..rows).map(|row| row_string(&grid, row) + "\n").collect();
            assert!(!all.contains("FANS"), "{cols}x{rows}\n{all}");
        }
    }
}

#[test]
fn fan_gauges_at_the_fans_golden_sizes() {
    for (_, cols, rows) in FANS_GOLDENS {
        let at = format!("{cols}x{rows}");
        let grid = draw(&fans_model(), cols, rows);
        for (fan, rpm) in [
            ("front1", "1939"),
            ("front2", "1877"),
            ("rear", "3026"),
            ("top", "1272"),
        ] {
            let row = meter_row(&grid, fan);
            assert!(
                row < timeline_rule(&grid),
                "{at}: fans stay in the top band"
            );
            let text = row_string(&grid, row);
            assert!(text.contains(&format!("{rpm} rpm")), "{at}: {text}");
        }
        // `top` runs at pwm 255: a full bar.
        let row = meter_row(&grid, "top");
        let col = meter_bar_col(&grid, row, "top");
        let lit = bar_from(&grid, row, col).len();
        let after = grid.get(col + u16::try_from(lit).unwrap(), row).unwrap().ch;
        assert_eq!(after, ' ', "{at}: a full fan bar has no track");
        assert_blank_edges(&grid);
    }
}

#[test]
fn in_out_wrap_inside_the_left_part_when_fans_are_beside_them() {
    let mut model = fans_model();
    model.in_lines = vec!["I".repeat(400)];
    model.out_lines = vec!["O".repeat(400)];
    let grid = draw(&model, 240, 67);
    let mid = grid.cols() / 2;
    let title = (0..grid.rows())
        .find(|row| {
            row_string(&grid, *row).contains("IN") && row_string(&grid, *row).contains("OUT")
        })
        .expect("titles");
    let mut eyes = 0;
    let mut ohs = 0;
    for row in title..grid.rows() - 3 {
        for col in 2..mid {
            if grid.get(col, row).unwrap().ch == 'I' {
                eyes += 1;
            }
        }
        for col in mid + 1..grid.cols() - 1 {
            if grid.get(col, row).unwrap().ch == 'O' {
                ohs += 1;
            }
        }
    }
    assert!(eyes > 0 && ohs > 0, "I {eyes} O {ohs}");
}

/// #95: fan rows stack with no blank row between, so the meter keeps #52's
/// whole-cell glyph (`▇`/`▄`) instead of the half-cell `█`/`▐`/`▌`.
#[allow(dead_code)]
fn meter_cells(grid: &llama_watch::tty::grid::Grid, row: u16) -> Vec<C16> {
    (0..grid.cols())
        .filter_map(|col| {
            let cell = grid.get(col, row).unwrap();
            is_lit_meter_cell(cell.ch).then_some(cell.fg)
        })
        .collect()
}

/// #165: a fan is a gauge: its rpm, and its duty as a spectrum bar.
#[test]
fn fan_gauge_shows_rpm_and_a_duty_spectrum() {
    let grid = draw(&fans_model(), 240, 67);
    let row = meter_row(&grid, "front1");
    let text = row_string(&grid, row);
    assert!(text.contains("front1    1939 rpm"), "{text}");
    let bar = bar_from(&grid, row, meter_bar_col(&grid, row, "front1"));
    // pwm 224 of 255 is 88 % of 36 cells.
    assert_eq!(bar.len(), 32, "{bar:?}");
    assert_eq!(steps_of(&bar), STEPS, "{bar:?}");
}

#[test]
fn stalled_fan_is_dim_red_and_others_are_not() {
    let mut model = fans_model();
    let panel = model.fans.as_mut().unwrap();
    panel.fans[2].rpm = Some(0);
    panel.fans[3].rpm = Some(0);
    panel.fans[3].pwm = Some(0);
    let grid = draw(&model, 240, 67);
    let (row, col) = label_hit(&grid, "rear");
    assert_eq!(grid.get(col, row).unwrap().fg, C16::Red, "stalled label");
    let bar = meter_bar_col(&grid, row, "rear");
    assert_eq!(grid.get(bar, row).unwrap().fg, C16::Red, "stalled bar");
    let (row, col) = label_hit(&grid, "top");
    assert_ne!(
        grid.get(col, row).unwrap().fg,
        C16::Red,
        "0 pwm is not a stall"
    );
    let (row, col) = label_hit(&grid, "front1");
    assert_ne!(grid.get(col, row).unwrap().fg, C16::Red, "turning fan");
}

#[test]
fn missing_values_draw_dashes_and_absent_chip_says_so() {
    let mut model = fans_model();
    model.fans.as_mut().unwrap().fans[1] = fan(3, "front2", None, None, None);
    let grid = draw(&model, 240, 67);
    let row = meter_row(&grid, "front2");
    let text = row_string(&grid, row);
    assert!(text.contains("front2          --"), "{text}");
    let col = meter_bar_col(&grid, row, "front2");
    assert_eq!(
        grid.get(col, row).unwrap().ch,
        '·',
        "no duty, an empty track"
    );
    model.fans = Some(FanPanel {
        chip: "nct6798".to_string(),
        present: false,
        fans: Vec::new(),
    });
    let grid = draw(&model, 240, 67);
    let text = row_string(&grid, row_with(&grid, "FANS"));
    assert!(
        text.contains("FANS   no single hwmon named nct6798"),
        "{text}"
    );
    assert!(row_with(&grid, "FANS") < timeline_rule(&grid));
}

#[test]
fn fan_gauges_sit_in_the_top_band_and_fans_off_draws_none() {
    let model = fans_model();
    for (cols, rows) in [(160u16, 67u16), (160u16, 48u16)] {
        let grid = draw(&model, cols, rows);
        for fan in ["front1", "front2", "rear", "top"] {
            assert!(
                meter_row(&grid, fan) < timeline_rule(&grid),
                "{cols}x{rows} {fan}"
            );
        }
        assert_blank_edges(&grid);
    }
    let mut off = model.clone();
    off.fans = None;
    let grid = draw(&off, 160, 48);
    let all = whole(&grid);
    assert!(
        !all.contains("FANS") && !all.contains("rpm"),
        "fans off draws nothing\n{all}"
    );
}

#[test]
fn text_off_keeps_the_fan_gauges_in_the_top_band() {
    for (cols, rows, _, _) in TEXT_OFF_SIZES {
        let at = format!("{cols}x{rows}");
        let mut model = text_off_model(WatchState::Generating);
        model.fans = Some(ref_fans());
        let grid = draw(&model, cols, rows);
        assert!(meter_row(&grid, "front1") < timeline_rule(&grid), "{at}");
        assert!(!whole(&grid).contains("prompt tail"), "{at}");
        assert_blank_edges(&grid);
    }
}

// ---- T53: per-slot context sparklines ---------------------------------------

use llama_watch::tty::ctx_history::{CtxHistory, CtxPoint};

/// Six hours of one slot, sampled every 10 s, oldest step first. Each step is
/// (used context, busy).
fn ctx_story(steps: &[(u64, bool)]) -> Vec<CtxPoint> {
    let mut history = CtxHistory::new(6);
    for (used, busy) in steps {
        history.advance(10_000);
        history.sample(Some(*used), *busy);
    }
    history.points()
}

/// `n` steps growing linearly from `from` to `to`, busy.
fn grow(from: u64, to: u64, n: u64) -> Vec<(u64, bool)> {
    (0..n)
        .map(|i| (from + (to - from) * i / n.max(1), true))
        .collect()
}

/// Slot 0: an agent session grows to 180k over three hours, is compacted to
/// 40k, and grows again. Slot 1: a session to 60k, idle for 83 min (llama
/// reports 0, the cache is held), then a new session from 5k to 120k in the
/// last hour.
fn ctx_story_slots() -> Vec<Slot> {
    let mut s0 = grow(20_000, 180_000, 1_080);
    s0.extend(grow(40_000, 96_000, 1_080));
    let mut s1 = grow(8_000, 60_000, 1_300);
    s1.extend(std::iter::repeat_n((0, false), 500));
    s1.extend(grow(5_000, 120_000, 360));
    vec![
        Slot {
            id: 0,
            generating: true,
            done: 95_100,
            cached: 0,
            open: false,
            done_known: true,
            total: 95_100,
            decoded: 812,
            ctx_prompt: Some(95_100),
            n_ctx: Some(262_144),
            ctx_history: ctx_story(&s0),
        },
        Slot {
            id: 1,
            generating: true,
            done: 118_700,
            cached: 0,
            open: false,
            done_known: true,
            total: 119_400,
            decoded: 1_204,
            ctx_prompt: Some(118_700),
            n_ctx: Some(262_144),
            ctx_history: ctx_story(&s1),
        },
    ]
}

fn ctx_model() -> TtyModel {
    let generating = load("generating-480.json");
    let mut model = sample(WatchState::Generating);
    model.in_title = title_line(&generating, "IN ");
    model.out_title = title_line(&generating, "OUT ");
    model.in_lines = region(&generating, "IN ", Some("OUT "));
    model.out_lines = region(&generating, "OUT ", None);
    model.chart = chart_story();
    model.chart_glyphs = ChartGlyphs::Eighths;
    model.slots = ctx_story_slots();
    model.slots_line = "2/2 busy".to_string();
    model
}

const CTX_GOLDENS: [(&str, u16, u16); 2] = [
    ("ctx-history-240.json", 240, 67),
    ("ctx-history-286.json", 286, 60),
];

#[test]
fn ctx_history_goldens_match_character_and_colour() {
    for (name, _, _) in CTX_GOLDENS {
        let fix = load(name);
        assert_frame(name, &fix, &ctx_model());
    }
}

#[test]
#[ignore = "run with --ignored to write the T53 ctx history goldens"]
fn dump_ctx_history_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, cols, rows) in CTX_GOLDENS {
        let grid = draw(&ctx_model(), cols, rows);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write ctx golden");
    }
}

/// The sparkline's cells on `row`, from the `ctx · 6h` header column to the
/// right margin.
fn spark_cells(
    grid: &llama_watch::tty::grid::Grid,
    row: u16,
) -> (u16, Vec<llama_watch::tty::grid::Cell>) {
    let header = row_with(grid, "ctx \u{b7} 6h");
    let x = col_of(grid, header, "ctx \u{b7} 6h");
    let cells = (x..grid.cols() - 1)
        .map(|col| grid.get(col, row).expect("cell"))
        .collect();
    (x, cells)
}

#[test]
fn sparkline_is_newest_left_with_a_red_marker_at_each_reset() {
    for (cols, rows) in [(240u16, 67u16), (286, 60), (480, 135)] {
        let grid = draw(&ctx_model(), cols, rows);
        assert_blank_edges(&grid);
        let header = row_with(&grid, "ctx \u{b7} 6h");
        let s0 = row_with(&grid, "s0 gen");
        let s1 = row_with(&grid, "s1 gen");
        assert_eq!(
            s0,
            header + 1,
            "{cols}x{rows}: header sits on the SLOTS label row"
        );
        assert_eq!(s1, s0 + 1);
        for row in [s0, s1] {
            let (x, cells) = spark_cells(&grid, row);
            let text = row_string(&grid, row);
            let value = char_at(&text, "k/262k") as u16;
            assert!(value < x, "{cols}x{rows}: sparkline after the ctx value");
            assert!(cells.len() >= 12, "{cols}x{rows}: {} cols", cells.len());
            // Newest on the left: the leftmost column is today's value, not
            // an empty old one.
            assert_ne!(cells[0].ch, ' ', "{cols}x{rows} row {row}");
            let marks: Vec<usize> = cells
                .iter()
                .enumerate()
                .filter(|(_, cell)| cell.ch == 'v')
                .map(|(i, _)| i)
                .collect();
            assert_eq!(marks.len(), 1, "{cols}x{rows} row {row}: {text}");
            assert!(
                cells
                    .iter()
                    .filter(|cell| cell.ch == 'v')
                    .all(|cell| cell.fg == C16::BrightRed)
            );
            // The reset sits mid-history, with growth on both sides.
            let mark = marks[0];
            assert!(mark > 0 && mark + 1 < cells.len());
            for cell in &cells {
                assert!(
                    cell.ch == ' '
                        || cell.ch == 'v'
                        || ('\u{2581}'..='\u{2588}').contains(&cell.ch),
                    "{cols}x{rows}: {:?}",
                    cell.ch
                );
                assert!(cell.ch.is_ascii() || GLYPHS.contains(&cell.ch));
            }
        }
        // Slot 0 was compacted after three hours: its marker is near the
        // middle. Slot 1's new session is newer, so its marker is further left.
        let (_, c0) = spark_cells(&grid, s0);
        let (_, c1) = spark_cells(&grid, s1);
        let m0 = c0.iter().position(|cell| cell.ch == 'v').expect("s0 mark");
        let m1 = c1.iter().position(|cell| cell.ch == 'v').expect("s1 mark");
        assert!(
            m1 < m0,
            "{cols}x{rows}: s1 reset {m1} is newer than s0 {m0}"
        );
    }
}

#[test]
fn sparkline_columns_take_the_step_colour_of_their_fill() {
    let grid = draw(&ctx_model(), 286, 60);
    let s0 = row_with(&grid, "s0 gen");
    let (_, cells) = spark_cells(&grid, s0);
    // Right before the compaction slot 0 was near 180k of 262k (68 %): the
    // fourth step. After it, 40k to 96k (15 to 36 %) seen through 30 min
    // maxima: the first two steps.
    let mark = cells.iter().position(|cell| cell.ch == 'v').expect("mark");
    assert_eq!(
        cells[mark + 1].fg,
        C16::BrightMagenta,
        "{:?}",
        &cells[mark..]
    );
    let after = &cells[..mark];
    assert!(
        after
            .iter()
            .all(|cell| matches!(cell.fg, C16::Blue | C16::BrightBlue)),
        "{after:?}"
    );
    assert_eq!(cells[0].fg, C16::BrightBlue, "{after:?}");
}

#[test]
fn halves_sparkline_uses_only_eurlatgr_glyphs() {
    let mut model = ctx_model();
    model.chart_glyphs = ChartGlyphs::Halves;
    let grid = draw(&model, 240, 67);
    for row in [row_with(&grid, "s0 gen"), row_with(&grid, "s1 gen")] {
        let (_, cells) = spark_cells(&grid, row);
        assert!(
            cells
                .iter()
                .all(|cell| matches!(cell.ch, ' ' | 'v' | '▄' | '█')),
            "{cells:?}"
        );
        assert!(cells.iter().any(|cell| cell.ch == '▄' || cell.ch == '█'));
    }
}

#[test]
fn sparkline_hides_below_twelve_columns_and_the_panel_keeps_its_height() {
    let narrow = draw(&ctx_model(), 160, 48);
    let all: String = (0..narrow.rows())
        .map(|row| row_string(&narrow, row) + "\n")
        .collect();
    assert!(!all.contains("ctx \u{b7}"), "{all}");
    // T34's right-aligned ctx block is still there.
    let row = row_with(&narrow, "95k/262k");
    assert!(row_string(&narrow, row).contains("ctx"));

    let plain = draw(&sample(WatchState::Generating), 240, 67);
    let story = draw(&ctx_model(), 240, 67);
    // Two slots each way: same rule rows.
    let mut two = sample(WatchState::Generating);
    two.slots = ctx_story_slots();
    for slot in &mut two.slots {
        slot.ctx_history.clear();
    }
    let blank = draw(&two, 240, 67);
    let rules = |grid: &llama_watch::tty::grid::Grid| {
        (0..grid.rows())
            .filter(|row| is_rule_row(grid, *row))
            .collect::<Vec<u16>>()
    };
    assert_eq!(rules(&blank), rules(&story));
    assert_eq!(
        req_rule_row(&plain),
        req_rule_row(&draw(&sample(WatchState::Generating), 240, 67))
    );
}

/// The tty's five console steps, low to high.
/// The gauge segment (#165): a lit gauge cell with the llama-hack font.
const SEG: char = '\u{e584}';

const STEPS: [C16; 5] = [
    C16::Blue,
    C16::BrightBlue,
    C16::Magenta,
    C16::BrightMagenta,
    C16::BrightRed,
];

/// Colours of a run of cells with repeats collapsed: the steps a bar shows.
fn steps_of(fgs: &[C16]) -> Vec<C16> {
    let mut out: Vec<C16> = Vec::new();
    for fg in fgs {
        if out.last() != Some(fg) {
            out.push(*fg);
        }
    }
    out
}

/// Lit colours of the bar that starts at `col` on `row`, up to the track.
fn bar_from(grid: &llama_watch::tty::grid::Grid, row: u16, col: u16) -> Vec<C16> {
    (col..grid.cols())
        .map(|c| grid.get(c, row).expect("cell"))
        .take_while(|cell| matches!(cell.ch, '█' | '▓' | '▐' | '▌' | '▄' | '▇' | SEG))
        .map(|cell| cell.fg)
        .collect()
}

/// Row of a meter labelled `label`. The rightmost word wins, so a word on
/// the left half that repeats the name is not the meter.
fn meter_row(grid: &llama_watch::tty::grid::Grid, label: &str) -> u16 {
    label_hit(grid, label).0
}

fn meter_bar_col(grid: &llama_watch::tty::grid::Grid, row: u16, label: &str) -> u16 {
    let (found, col) = label_hit(grid, label);
    assert_eq!(found, row, "{label}");
    let _ = col;
    bar_after(grid, row, col + label.chars().count() as u16)
}

fn label_hit(grid: &llama_watch::tty::grid::Grid, label: &str) -> (u16, u16) {
    let needle: Vec<char> = label.chars().collect();
    let mut found = None;
    for row in 0..grid.rows() {
        let chars: Vec<char> = row_string(grid, row).chars().collect();
        for (at, window) in chars.windows(needle.len()).enumerate() {
            if window != needle.as_slice() {
                continue;
            }
            let before = at == 0 || chars[at - 1] == ' ';
            let after = at + needle.len();
            let boundary = after >= chars.len() || chars[after] == ' ';
            if before && boundary && bar_after_near(grid, row, after as u16).is_some() {
                found = Some((row, at as u16));
            }
        }
    }
    found.unwrap_or_else(|| panic!("missing meter {label}"))
}

fn bar_after(grid: &llama_watch::tty::grid::Grid, row: u16, from: u16) -> u16 {
    bar_after_near(grid, row, from).unwrap_or_else(|| panic!("no bar on {row}"))
}

fn bar_after_near(grid: &llama_watch::tty::grid::Grid, row: u16, from: u16) -> Option<u16> {
    (from..from.saturating_add(22).min(grid.cols())).find(|col| {
        matches!(
            grid.get(*col, row).expect("cell").ch,
            '█' | '▓' | '▐' | '▌' | '▄' | '▇' | '·' | '░' | SEG
        )
    })
}

/// Top row (and bottom row at 4K) of the CPU meter at `pct`.
fn cpu_bar(pct: f64, cols: u16, rows: u16) -> (Vec<C16>, Vec<C16>, usize) {
    let mut model = sample(WatchState::Generating);
    model.cpu_pct = Some(pct);
    let grid = draw(&model, cols, rows);
    let row = meter_row(&grid, "CPU");
    let col = meter_bar_col(&grid, row, "CPU");
    if (pct - 100.0).abs() < 0.1 && cols == 240 {
        let line: String = (0..grid.cols())
            .map(|c| grid.get(c, row).unwrap().ch)
            .collect();
        eprintln!("CPU ROW {row} bar {col}\n{line}");
    }
    let track = (col..grid.cols())
        .take_while(|c| {
            matches!(
                grid.get(*c, row).expect("cell").ch,
                '█' | '▐' | '▌' | '░' | '▄' | '▇' | '·'
            )
        })
        .count();
    (
        bar_from(&grid, row, col),
        bar_from(&grid, row + 1, col),
        track,
    )
}

/// T63: a level meter is a spectrum. Each lit cell takes the step of its own
/// position, so a full bar runs every step in order and starts blue.
#[test]
fn full_level_meter_shows_every_step_in_order() {
    for (cols, rows) in [(240u16, 67u16), (480, 135)] {
        let (top, bottom, track) = cpu_bar(100.0, cols, rows);
        assert_eq!(top.len(), track, "{cols}x{rows} a full bar has no track");
        assert_eq!(steps_of(&top), STEPS, "{cols}x{rows} {top:?}");
        // #165: one-row gauges at every size; the row under is the next gauge.
        assert!(
            steps_of(&bottom).len() < STEPS.len(),
            "{cols}x{rows} {bottom:?}"
        );
        // Each step covers about a fifth of the bar.
        for step in STEPS {
            let n = top.iter().filter(|fg| **fg == step).count();
            assert!(
                n.abs_diff(track / 5) <= 1,
                "{cols}x{rows} {step:?} covers {n} of {track}"
            );
        }
    }
}

/// T63: a 30 % bar reaches only the low steps (0-20 % and 20-40 %).
#[test]
fn thirty_percent_level_meter_holds_only_the_low_steps() {
    for (cols, rows) in [(240u16, 67u16), (480, 135)] {
        let (top, _, _) = cpu_bar(30.0, cols, rows);
        assert_eq!(
            steps_of(&top),
            [C16::Blue, C16::BrightBlue],
            "{cols}x{rows} {top:?}"
        );
        let (low, _, _) = cpu_bar(10.0, cols, rows);
        assert_eq!(steps_of(&low), [C16::Blue], "{cols}x{rows} {low:?}");
    }
}

/// T63, #165: every gauge is a spectrum, capacity (VRAM, MEM) too: each
/// lit cell takes the step of its own position.
#[test]
fn every_gauge_is_a_spectrum_capacity_too() {
    let mut model = sample(WatchState::Generating);
    model.cpu_pct = Some(90.0);
    model.gpu_pct = Some(90.0);
    model.load_pct = Some(90.0);
    model.activity_pct = Some(90.0);
    model.power_w = Some(450.0);
    model.power_limit_w = Some(500.0);
    model.vram_used_gb = Some(90.0);
    model.vram_total_gb = Some(100.0);
    model.mem_used_gb = Some(90.0);
    model.mem_total_gb = Some(100.0);
    let grid = draw(&model, 240, 67);
    for label in ["CPU", "GPU", "POWER", "LOAD", "ACT", "VRAM", "MEM"] {
        let row = meter_row(&grid, label);
        let bar = bar_from(&grid, row, meter_bar_col(&grid, row, label));
        assert_eq!(steps_of(&bar), STEPS, "{label} {bar:?}");
    }
}

/// An SGLang model (T72): no `/slots`, a backend line in SLOTS, a note in
/// IN/OUT, and RECENT rows whose rates llama-swap could not report.
fn sglang_model() -> TtyModel {
    let mut model = sample(WatchState::Generating);
    model.model_name = "flash".to_string();
    model.model_detail = "SGLang".to_string();
    let detail = ModelDetail {
        ctx: Some(204_800),
        kv_k: Some("fp8_e4m3".to_owned()),
        kv_v: Some("fp8_e4m3".to_owned()),
        quant: Some("exl3".to_owned()),
        ..ModelDetail::default()
    };
    model.setup = Some(setup_of(
        "flash",
        "flash",
        SGLANG_CMD,
        Backend::SgLang,
        Some(&detail),
        None,
    ));
    model.slots_line = "--".to_string();
    model.slots = Vec::new();
    // #79: the KV item from SGLang's token gauges.
    model.backend_lines = vec![
        "sglang  running 1/4 · queued 0 · KV 75k/204k tok 37 % · +31k cached · hit 50 %"
            .to_string(),
    ];
    model.text_note = "text needs llama.cpp /slots".to_string();
    model.in_title = "  IN".to_string();
    model.out_title = "  OUT".to_string();
    model.in_lines = Vec::new();
    model.out_lines = Vec::new();
    model.prompt_last = None;
    for req in &mut model.requests {
        req.model = "flash".to_string();
        req.prompt_tps = None;
        req.gen_tps = None;
    }
    model
}

/// An invented SGLang launch in a container (#52).
const SGLANG_CMD: &str = "podman run --rm --name flash -e SGLANG_EXL3_MOE_OFFLOAD=gpu_cache ghcr.io/example/sglang-exl3 python3 -m sglang.launch_server --model-path /models/flash --quantization exl3 --kv-cache-dtype fp8_e4m3 --context-length 204800 --mem-fraction-static 0.88 --max-running-requests 4";

const SGLANG_GOLDENS: [(&str, u16, u16); 2] =
    [("sglang-240.json", 240, 67), ("sglang-160.json", 160, 48)];

#[test]
fn sglang_goldens_match_character_and_colour() {
    for (name, _, _) in SGLANG_GOLDENS {
        let fix = load(name);
        assert_frame(name, &fix, &sglang_model());
    }
}

#[test]
#[ignore = "run with --ignored to write the T72 SGLang goldens"]
fn dump_sglang_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, cols, rows) in SGLANG_GOLDENS {
        let grid = draw(&sglang_model(), cols, rows);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write sglang golden");
    }
}

/// #31: a containerised vLLM found by its metrics: the cache facts on the
/// tuning line, the engine numbers on its backend line in SLOTS. #35: the
/// engine-measured speeds end that line, and RECENT shows them as `~`
/// rates; the row whose window saw no finished request stays `--`.
fn vllm_model() -> TtyModel {
    let mut model = sglang_model();
    model.model_name = "qwen3.8-27b".to_string();
    model.model_detail = "vLLM".to_string();
    let detail = ModelDetail {
        kv_k: Some("fp8_e4m3".to_owned()),
        kv_v: Some("fp8_e4m3".to_owned()),
        kv_block: Some(16),
        prefix_cache: Some(true),
        ..ModelDetail::default()
    };
    let info = BackendInfo {
        kind: Backend::Vllm,
        engine: EngineStats {
            spec_permille: Some(781),
            spec_len_centi: Some(290),
            ..EngineStats::default()
        },
        ..BackendInfo::default()
    };
    model.setup = Some(setup_of(
        "qwen3.8-27b",
        "qwen3.8-27b",
        "podman run --rm --name hq -e SPEC=mtp ghcr.io/example/qwen-vllm single",
        Backend::Vllm,
        Some(&detail),
        Some(&info),
    ));
    model.backend_lines = vec![
        // #79: in use is vLLM's ratio × capacity: approximate.
        "vllm  running 1 · queued 0 · KV \u{2248}77k/187k tok 41 % · hit 75 % · spec 78 % · 2.9/step · ttft 420 ms · itl 31 ms · e2e 12.5 s · preempt 3 · prefill 2,134/s · decode 41.2/s"
            .to_string(),
    ];
    let measured = [
        Some((2_134.4, 41.25)),
        Some((1_987.0, 38.6)),
        None,
        Some((123_456.0, 1_234.0)),
    ];
    for (req, speeds) in model.requests.iter_mut().zip(measured) {
        req.model = "qwen3.8-27b".to_string();
        if let Some((prompt, generated)) = speeds {
            req.prompt_tps = Some(prompt);
            req.gen_tps = Some(generated);
            req.prompt_measured = true;
            req.gen_measured = true;
        }
    }
    model
}

const VLLM_GOLDENS: [(&str, u16, u16); 2] =
    [("vllm-240.json", 240, 67), ("vllm-160.json", 160, 48)];

#[test]
fn vllm_goldens_match_character_and_colour() {
    for (name, _, _) in VLLM_GOLDENS {
        let fix = load(name);
        assert_frame(name, &fix, &vllm_model());
    }
}

#[test]
#[ignore = "run with --ignored to write the #31 vLLM goldens"]
fn dump_vllm_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, cols, rows) in VLLM_GOLDENS {
        let grid = draw(&vllm_model(), cols, rows);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write vllm golden");
    }
}

/// #33: a Strata model: one request at a time. #79: its KV is its batch
/// slots' tokens.
fn strata_model() -> TtyModel {
    let mut model = sglang_model();
    model.model_name = "bonsai-27b".to_string();
    model.model_detail = "Strata".to_string();
    let detail = ModelDetail {
        ctx: Some(262_144),
        kv_k: Some("q8".to_owned()),
        kv_v: Some("q8".to_owned()),
        ..ModelDetail::default()
    };
    model.setup = Some(setup_of(
        "bonsai-27b",
        "Bonsai 27B",
        "python serve/server.py --engine strata --config /data/strata.json",
        Backend::Strata,
        Some(&detail),
        None,
    ));
    model.backend_lines =
        vec!["strata  running 1/1 · queued 0 · KV 6k/524k tok 1 % · 2 sessions".to_string()];
    for req in &mut model.requests {
        req.model = "bonsai-27b".to_string();
    }
    model
}

/// #33: any other OpenAI-compatible server: no metrics, no detail.
fn openai_model() -> TtyModel {
    let mut model = sglang_model();
    model.model_name = "tabby".to_string();
    model.model_detail = "OpenAI-compatible".to_string();
    model.setup = Some(setup_of(
        "tabby",
        "tabby",
        "python3 main.py --config /tabby/config.yml",
        Backend::OpenAi,
        None,
        None,
    ));
    model.backend_lines = vec!["openai  running -- · queued -- · KV --".to_string()];
    for req in &mut model.requests {
        req.model = "tabby".to_string();
    }
    model
}

/// #79: a llama.cpp model's KV line, under its slot rows: every slot's
/// held tokens in the one pool they share.
fn llamacpp_kv_model() -> TtyModel {
    let mut model = sample(WatchState::Generating);
    model.backend_lines = vec!["llamacpp  KV 91k/262k tok shared 35 % · 1 session".to_string()];
    model
}

/// A golden file and the frame it holds.
type EngineGolden = (&'static str, fn() -> TtyModel);

const ENGINE_GOLDENS: [EngineGolden; 2] = [
    ("strata-240.json", strata_model),
    ("openai-240.json", openai_model),
];

const KV_GOLDENS: [(&str, u16, u16); 2] = [
    ("kv-llamacpp-240.json", 240, 67),
    ("kv-llamacpp-160x49.json", 160, 49),
];

#[test]
fn llamacpp_kv_goldens_match_character_and_colour() {
    for (name, _, _) in KV_GOLDENS {
        let fix = load(name);
        assert_frame(name, &fix, &llamacpp_kv_model());
    }
    // The line sits right under the slot rows, `≈`-free and whole.
    let grid = draw(&llamacpp_kv_model(), 160, 49);
    let slots = row_with(&grid, "SLOTS");
    assert!(
        row_string(&grid, slots + 2).ends_with("llamacpp  KV 91k/262k tok shared 35 % · 1 session"),
        "{}",
        row_string(&grid, slots + 2)
    );
}

#[test]
#[ignore = "run with --ignored to write the #79 llama.cpp KV goldens"]
fn dump_llamacpp_kv_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, cols, rows) in KV_GOLDENS {
        let grid = draw(&llamacpp_kv_model(), cols, rows);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write kv golden");
    }
}

#[test]
fn engine_goldens_match_character_and_colour() {
    for (name, make) in ENGINE_GOLDENS {
        let fix = load(name);
        assert_frame(name, &fix, &make());
    }
}

#[test]
#[ignore = "run with --ignored to write the #33 Strata and OpenAI goldens"]
fn dump_engine_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, make) in ENGINE_GOLDENS {
        let grid = draw(&make(), 240, 67);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write engine golden");
    }
}

/// #33: the header always names the engine of the shown model, in grey
/// right after the name, and says `engine --` with no model.
#[test]
fn header_always_names_the_engine() {
    let cases: [(TtyModel, &str); 7] = [
        (sample(WatchState::Generating), "Qwen 35B  llama.cpp"),
        (sample(WatchState::Ready), "Qwen 35B  llama.cpp"),
        (sglang_model(), "flash  SGLang"),
        (vllm_model(), "qwen3.8-27b  vLLM"),
        (strata_model(), "bonsai-27b  Strata"),
        (openai_model(), "tabby  OpenAI-compatible"),
        (sample(WatchState::AiDown), "model --  engine --"),
    ];
    for (model, want) in cases {
        for cols in [160, 240] {
            let grid = draw(&model, cols, 67);
            let header = row_string(&grid, 0);
            assert!(header.contains(want), "{cols}: {header}");
            let engine = want.split("  ").nth(1).expect("engine");
            let at = col_of(&grid, 0, engine);
            assert_eq!(grid.get(at, 0).unwrap().fg, C16::BrightBlack, "{want}");
        }
    }
    for state in [WatchState::Starting, WatchState::NoLlama] {
        let header = row_string(&draw(&sample(state), 240, 67), 0);
        assert!(header.contains("  engine --"), "{header}");
    }
}

#[test]
fn sglang_frame_names_the_backend_and_its_gauges() {
    let model = sglang_model();
    for (cols, rows) in [(160, 48), (240, 67), (480, 135)] {
        let grid = draw(&model, cols, rows);
        let header = row_string(&grid, 0);
        assert!(header.contains("flash  SGLang"), "{header}");
        let slots = row_with(&grid, "SLOTS");
        let line = row_string(&grid, slots + 1);
        assert!(
            line.contains("sglang  running 1/4 · queued 0 · KV 75k/204k tok 37 %"),
            "{cols}: {line}"
        );
        let in_row = (0..grid.rows())
            .find(|row| row_string(&grid, *row).trim_start().starts_with("IN"))
            .expect("IN title");
        let notes: Vec<u16> = (in_row..grid.rows())
            .filter(|row| row_string(&grid, *row).contains("text needs llama.cpp /slots"))
            .collect();
        assert!(!notes.is_empty(), "the empty panes say why: {cols}");
        let recent = row_with(&grid, "flash ");
        assert!(row_string(&grid, recent).contains("--"), "{cols}");
        for row in 0..grid.rows() {
            assert_eq!(grid.get(cols - 1, row).unwrap().ch, ' ', "last col {cols}");
        }
    }
}

#[test]
fn text_note_gives_way_to_real_text() {
    let mut model = sglang_model();
    model.in_lines = vec!["  prompt".to_string()];
    let grid = draw(&model, 240, 67);
    assert!(
        (0..grid.rows()).all(|row| !row_string(&grid, row).contains("text needs")),
        "a note only fills empty panels"
    );
}

#[test]
fn a_model_stuck_stopping_is_tagged_in_the_header() {
    let mut model = sample(WatchState::Ready);
    let grid = draw(&model, 240, 67);
    assert!(!row_string(&grid, 0).contains("stuck"));
    model.model_stuck = true;
    for cols in [160, 240] {
        let grid = draw(&model, cols, 67);
        let header = row_string(&grid, 0);
        assert!(header.contains("Qwen 35B  stopping (stuck?)"), "{header}");
        let at = col_of(&grid, 0, "stopping (stuck?)");
        assert_eq!(grid.get(at, 0).unwrap().fg, C16::Yellow);
    }
}

/// #135: STOPPING keeps the stuck tag, and a job word still wins.
#[test]
fn stopping_keeps_the_stuck_tag_and_a_job_word_wins() {
    let mut model = sample(WatchState::Ready);
    model.model_stopping = true;
    model.model_stuck = true;
    model.slots_line = "--".to_owned();
    model.swap_line = "stopping 1:01".to_owned();
    let grid = draw(&model, 240, 67);
    let header = row_string(&grid, 0);
    assert!(header.contains("STOPPING"), "{header}");
    assert!(header.contains("Qwen 35B  stopping (stuck?)"), "{header}");
    assert!(header.contains("slots --"), "{header}");
    assert!(header.contains("swap stopping 1:01"), "{header}");
    assert!(!header.contains("READY"), "{header}");
    let word = col_of(&grid, 0, "STOPPING");
    assert_eq!(grid.get(word, 0).unwrap().fg, C16::Yellow);
    assert_eq!(grid.get(word, 0).unwrap().bg, C16::Black);
    let pip = grid.get(24, 0).unwrap();
    assert_eq!((pip.ch, pip.fg, pip.bg), ('█', C16::Yellow, C16::Black));

    model.job = Some(JobView {
        job: example_job(),
        more: 0,
    });
    let header = row_string(&draw(&model, 240, 67), 0);
    assert!(header.contains("TRAINING"), "{header}");
    assert!(
        !header.contains("STOPPING") && !header.contains("LOADING"),
        "{header}"
    );
}

/// #139: a live job's word wins over GPU BUSY, and so do LOADING and STOPPING.
#[test]
fn a_job_word_beats_gpu_busy() {
    let mut model = sample(WatchState::Ready);
    model.gpu_busy = true;
    model.model_name = "outside llama-swap".to_owned();
    model.model_detail = "example-bench 22.4 GB".to_owned();
    model.slots_line = "--".to_owned();
    model.swap_line = "--".to_owned();
    model.job = Some(JobView {
        job: example_job(),
        more: 0,
    });
    let header = row_string(&draw(&model, 240, 67), 0);
    assert!(header.contains("TRAINING"), "{header}");
    assert!(!header.contains("GPU BUSY"), "{header}");

    model.job = None;
    model.model_loading = true;
    let header = row_string(&draw(&model, 240, 67), 0);
    assert!(header.contains("LOADING"), "{header}");
    assert!(!header.contains("GPU BUSY"), "{header}");

    model.model_loading = false;
    model.model_stopping = true;
    let header = row_string(&draw(&model, 240, 67), 0);
    assert!(header.contains("STOPPING"), "{header}");
    assert!(!header.contains("GPU BUSY"), "{header}");
}

/// #135: a live job's word wins over LOADING.
#[test]
fn a_job_word_beats_loading() {
    let mut model = sample(WatchState::Ready);
    model.model_loading = true;
    model.slots_line = "--".to_owned();
    model.swap_line = "loading 1:42".to_owned();
    model.job = Some(JobView {
        job: example_job(),
        more: 0,
    });
    let header = row_string(&draw(&model, 240, 67), 0);
    assert!(header.contains("TRAINING"), "{header}");
    assert!(!header.contains("LOADING"), "{header}");
}

/// #135: `loading M:SS` is one field value. It sheds whole, the same way
/// `none` does, and is never cut down to a fragment.
#[test]
fn loading_swap_sheds_as_one_field() {
    let mut loading = sample(WatchState::Ready);
    loading.model_loading = true;
    loading.model_name = "N".repeat(48);
    // A full detail puts `swap` against the clock on a real header. At 160
    // a 48-wide name alone still leaves room for `loading 1:42`, and below
    // 160 the frame is the too-small banner, which sheds nothing.
    loading.model_detail = "d".repeat(48);
    loading.slots_line = "--".to_owned();
    loading.swap_line = "loading 1:42".to_owned();
    let mut plain = loading.clone();
    plain.model_loading = false;
    plain.swap_line = "none".to_owned();
    let mut saw_whole = false;
    let mut saw_none_without_longer = false;
    for cols in 100u16..=240 {
        let header = header_line(&loading, cols);
        let shown = header.contains("swap loading 1:42");
        // Whole or nothing. A cut such as `loading` or `1:42` fails this.
        assert!(
            shown
                || (!header.contains("swap")
                    && !header.contains("loading")
                    && !header.contains("1:42")),
            "{cols}: swap was cut: {header}"
        );
        if shown {
            saw_whole = true;
            assert!(
                header.contains("slots --"),
                "{cols}: slots went with the swap field: {header}"
            );
        }
        let none = header_line(&plain, cols);
        let none_shown = none.contains("swap none");
        assert!(
            none_shown || (!none.contains("swap") && !none.contains("none")),
            "{cols}: `none` was cut: {none}"
        );
        // Widths where `none` fits and the longer value does not: nothing
        // of the longer value remains.
        if none_shown && !shown {
            saw_none_without_longer = true;
            assert!(
                !header.contains("swap") && !header.contains("loading") && !header.contains("1:42"),
                "{cols}: longer swap was cut where none still fits: {header}"
            );
        }
    }
    assert!(saw_whole, "the sweep must show the field");
    assert!(
        saw_none_without_longer,
        "the sweep must pass widths where none fits and the longer value does not"
    );
}

// #7: short screens. Below 48 rows the dashboard drops panels in order
// (IN/OUT, then FANS, then chart height) and refuses only below 160x26.

/// Rows of the chart under RECENT, from its gen ceiling label to its prompt
/// ceiling label. 0 when there is no chart.
fn chart_rows_below_recent(grid: &llama_watch::tty::grid::Grid) -> u16 {
    if !(0..grid.rows()).any(|row| row_string(grid, row).contains("TIMELINE")) {
        return 0;
    }
    chart_height_of(grid)
}

fn recent_rows_shown(grid: &llama_watch::tty::grid::Grid) -> u16 {
    let Some(header) = (0..grid.rows()).find(|row| row_string(grid, *row).contains("RECENT"))
    else {
        return 0;
    };
    let rule = (header + 1..grid.rows())
        .find(|row| is_rule_row(grid, *row))
        .unwrap_or(grid.rows());
    rule - header - 1
}

fn whole(grid: &llama_watch::tty::grid::Grid) -> String {
    (0..grid.rows())
        .map(|row| row_string(grid, row) + "\n")
        .collect()
}

fn short_model() -> TtyModel {
    let mut model = chart_model();
    model.in_lines = vec!["prompt body".to_string()];
    model.out_lines = vec!["output body".to_string()];
    model.chart = chart_story();
    model
}

fn assert_dashboard(grid: &llama_watch::tty::grid::Grid, at: &str) {
    let all = whole(grid);
    assert!(!all.contains("too small"), "{at}: refused\n{all}");
    assert!(row_string(grid, 0).contains("GENERATING"), "{at}\n{all}");
    if grid.rows() > 26 {
        assert!(meter_row(grid, "ACT") < row_with(grid, "RECENT"), "{at}");
    } else {
        assert!(whole(grid).contains("TIMELINE"), "{at}");
    }
    let health = grid.rows() - 2;
    assert!(
        row_string(grid, health).contains("snapshot"),
        "{at}: health row\n{all}"
    );
    assert!(
        is_rule_row(grid, grid.rows() - 3),
        "{at}: health rule\n{all}"
    );
    assert!(!all.contains("text off"), "{at}: text is not off\n{all}");
    assert_blank_edges(grid);
}

#[test]
fn short_160x45_keeps_the_full_chart_and_in_out() {
    let grid = draw(&short_model(), 160, 45);
    assert_dashboard(&grid, "160x45");
    let all = whole(&grid);
    assert_eq!(recent_rows_shown(&grid), 7, "{all}");
    assert_eq!(chart_rows_below_recent(&grid), 10, "{all}");
    let inn = row_with(&grid, "prompt tail");
    let out = row_with(&grid, "OUT  live");
    assert_eq!(inn, out, "IN and OUT share a row\n{all}");
    assert!(
        all.contains("prompt body") && all.contains("output body"),
        "{all}"
    );
}

#[test]
fn short_160x40_drops_in_out_first_and_keeps_recent_and_the_chart() {
    let grid = draw(&short_model(), 160, 40);
    assert_dashboard(&grid, "160x40");
    let all = whole(&grid);
    // The waterfall still fits a 6-row text band at 40 rows.
    assert!(
        all.contains("prompt body") && all.contains("output body"),
        "{all}"
    );
    assert!(recent_rows_shown(&grid) >= 4, "{all}");
    assert_eq!(chart_rows_below_recent(&grid), 9, "{all}");
}

/// #165: at 160x45 the fans fit as gauges; at the 160x26 floor they are
/// the first to go and the work gauges stay.
#[test]
fn short_screens_drop_fans_after_in_out_then_shrink_the_chart() {
    let mut model = short_model();
    model.fans = Some(ref_fans());
    let grid = draw(&model, 160, 45);
    assert!(meter_row(&grid, "front1") < timeline_rule(&grid));
    assert!(whole(&grid).contains("prompt body"), "{}", whole(&grid));
    model.temps = Some(ref_temps());
    let grid = draw(&model, 160, 26);
    assert_eq!(recent_rows_shown(&grid), 0);
    assert!(whole(&grid).contains("TIMELINE"));
    assert!(!whole(&grid).contains("rpm"), "fans went first");
    assert!(meter_row(&grid, "IO PSI") < timeline_rule(&grid));
}

#[test]
fn floor_is_160x26_and_only_smaller_is_too_small() {
    let grid = draw(&short_model(), 160, 26);
    assert_dashboard(&grid, "160x26");
    assert_eq!(
        recent_rows_shown(&grid),
        0,
        "26 rows keep the timeline, not RECENT: {}",
        whole(&grid)
    );
    assert!(whole(&grid).contains("TIMELINE"), "{}", whole(&grid));
    for (cols, rows) in [(160u16, 25u16), (159, 49), (120, 30)] {
        let grid = draw(&short_model(), cols, rows);
        assert_eq!(
            row_string(&grid, 0),
            format!("llama-watch: tty too small ({cols}x{rows}, need 160x26)")
        );
    }
    assert_eq!(llama_watch::tty::layout::MIN_COLS, 160);
    assert_eq!(llama_watch::tty::layout::MIN_ROWS, 26);
}

#[test]
fn text_off_also_draws_below_48_rows() {
    for rows in [45u16, 40, 26] {
        let grid = draw(&text_off_model(WatchState::Generating), 160, rows);
        let at = format!("160x{rows} text off");
        let all = whole(&grid);
        assert!(!all.contains("too small"), "{at}\n{all}");
        assert!(row_string(&grid, 0).contains("text off"), "{at}\n{all}");
        assert!(!all.contains("LEAKED"), "{at}");
        if rows > 26 {
            assert!(recent_rows_shown(&grid) >= 4, "{at}\n{all}");
        }
        assert_blank_edges(&grid);
    }
}

/// Many SLOTS rows on a short screen push RECENT down; nothing spills
/// into the health rows and nothing panics.
#[test]
fn many_slots_on_a_short_screen_keep_the_health_rows_clean() {
    let mut model = short_model();
    model.slots = (0..16)
        .map(|id| Slot {
            id,
            generating: true,
            done: 10,
            cached: 0,
            open: false,
            done_known: true,
            total: 20,
            decoded: 5,
            ctx_prompt: None,
            n_ctx: None,
            ctx_history: Vec::new(),
        })
        .collect();
    for rows in [26u16, 30, 40, 47] {
        let grid = draw(&model, 160, rows);
        let all = whole(&grid);
        assert!(!all.contains("too small"), "160x{rows}\n{all}");
        assert!(is_rule_row(&grid, rows - 3), "160x{rows}\n{all}");
        let health = row_string(&grid, rows - 2);
        assert!(health.contains("snapshot"), "160x{rows}: {health}");
        assert!(
            !health.contains("s1") && !health.contains("tok"),
            "160x{rows}: {health}"
        );
        assert_blank_edges(&grid);
    }
}

// ---- #9: reset reasons on the sparkline -------------------------------------

use llama_watch::resets::ResetReason;

/// [`ctx_story`], with the drop labelled `reason` a minute after it.
fn labelled_story(steps: &[(u64, bool)], reason: ResetReason) -> Vec<CtxPoint> {
    let mut history = CtxHistory::new(6);
    let mut since: Option<usize> = None;
    for (i, (used, busy)) in steps.iter().enumerate() {
        history.advance(10_000);
        if history.sample(Some(*used), *busy) {
            since = Some(i);
        }
        if since.is_some_and(|at| i == at + 6) {
            history.label(reason);
        }
    }
    history.points()
}

fn labelled_model() -> TtyModel {
    let mut model = ctx_model();
    let mut s0 = grow(20_000, 180_000, 1_080);
    s0.extend(grow(40_000, 96_000, 1_080));
    let mut s1 = grow(8_000, 60_000, 1_300);
    s1.extend(std::iter::repeat_n((0, false), 500));
    s1.extend(grow(5_000, 120_000, 360));
    model.slots[0].ctx_history = labelled_story(&s0, ResetReason::Compacted);
    model.slots[1].ctx_history = labelled_story(&s1, ResetReason::Evicted);
    model
}

#[test]
fn reset_markers_show_their_reason_the_last_one_per_slot_and_a_legend() {
    for (cols, rows) in [(240u16, 67u16), (286, 60), (480, 135)] {
        let grid = draw(&labelled_model(), cols, rows);
        assert_blank_edges(&grid);
        let header = row_with(&grid, "ctx \u{b7} 6h");
        let header_text = row_string(&grid, header);
        assert!(
            header_text.contains("c compact") && header_text.contains("e evict"),
            "{cols}x{rows}: {header_text}"
        );
        for (label, mark, fg) in [
            ("s0 gen", 'c', C16::BrightGreen),
            ("s1 gen", 'e', C16::BrightYellow),
        ] {
            let row = row_with(&grid, label);
            let (x, cells) = spark_cells(&grid, row);
            let marks: Vec<_> = cells
                .iter()
                .filter(|cell| cell.ch.is_ascii_alphabetic())
                .collect();
            assert_eq!(marks.len(), 1, "{cols}x{rows} {label}: {marks:?}");
            assert_eq!((marks[0].ch, marks[0].fg), (mark, fg));
            // The last reason sits in the gap just before the sparkline.
            let last = grid.get(x - 1, row).expect("cell");
            assert_eq!((last.ch, last.fg), (mark, fg), "{cols}x{rows} {label}");
            assert_eq!(grid.get(x - 2, row).expect("cell").ch, ' ');
        }
    }
    // No marker, no legend and no last reason.
    let mut quiet = ctx_model();
    for slot in &mut quiet.slots {
        slot.ctx_history.retain(|point| !point.reset);
    }
    let grid = draw(&quiet, 240, 67);
    let header = row_with(&grid, "ctx \u{b7} 6h");
    assert!(!row_string(&grid, header).contains("compact"));
    let row = row_with(&grid, "s0 gen");
    let (x, _) = spark_cells(&grid, row);
    assert_eq!(grid.get(x - 1, row).expect("cell").ch, ' ');
}

/// #35: RECENT's rate text. The `~` mark always fits: PROMPT is 7 wide,
/// GEN 6.
#[test]
fn engine_measured_rates_are_marked_and_fit_their_columns() {
    use llama_watch::tty::layout::{gen_rate_text, prompt_rate_text};
    assert_eq!(prompt_rate_text(Some(1193.6), false), "1,194");
    assert_eq!(prompt_rate_text(Some(2134.4), true), "~2,134");
    assert_eq!(prompt_rate_text(Some(99_999.4), true), "~99,999");
    assert_eq!(prompt_rate_text(Some(123_456.0), true), "~123k");
    assert_eq!(prompt_rate_text(Some(1_000_000.0), true), "~1,000k");
    assert_eq!(prompt_rate_text(None, true), "--");
    assert_eq!(gen_rate_text(Some(50.46), false), "50.5");
    assert_eq!(gen_rate_text(Some(41.25), true), "~41.3");
    assert_eq!(gen_rate_text(Some(999.94), true), "~999.9");
    assert_eq!(gen_rate_text(Some(999.95), true), "~1,000");
    assert_eq!(gen_rate_text(Some(9_999.4), true), "~9,999");
    assert_eq!(gen_rate_text(Some(12_345.0), true), "~12k");
    assert_eq!(gen_rate_text(None, true), "--");
    for tps in [0.0, 9.96, 999.9, 99_999.0, 100_000.0, 999_999.0] {
        assert!(
            prompt_rate_text(Some(tps), true).chars().count() <= 7,
            "{tps}"
        );
    }
    for tps in [0.0, 9.96, 999.9, 1_000.0, 9_999.0, 99_999.0, 1_000_000.0] {
        assert!(gen_rate_text(Some(tps), true).chars().count() <= 6, "{tps}");
    }
}

/// #39: a long model name and a long engine detail at 160 columns. The
/// detail drops whole trailing ` · item`s so a gap stays before the clock.
/// Since #52 the real detail is the engine alone; the drop rule stays for
/// any longer text.
#[test]
fn long_model_detail_drops_whole_items_before_the_clock() {
    let mut model = sample(WatchState::Generating);
    model.model_name = "Qwen3.8-27B on vLLM (HyperQwen, DFlash2, 240k KVarN)".to_string();
    model.model_detail = "vLLM · kv kvarn_k4v2_g128 · block 128 · prefix on".to_string();
    let grid = draw(&model, 160, 48);
    let header = row_string(&grid, 0);
    let clock = find_chars(&header, "2026-09-23").expect("clock drawn");
    let before: String = header.chars().take(clock).collect();
    assert!(
        before.ends_with("  ")
            && before
                .trim_end()
                .ends_with("vLLM · kv kvarn_k4v2_g128 · block 128"),
        "detail not cut at a whole item with a gap: {header}"
    );
    assert!(!header.contains("prefix"), "a cut item is drawn: {header}");
    // Every header field left of the clock keeps the gap, at any width.
    for cols in [160u16, 180, 200, 240, 320] {
        let grid = draw(&model, cols, 67);
        let header = row_string(&grid, 0);
        let clock = find_chars(&header, "2026-09-23").expect("clock drawn");
        let at = if cols >= 200 {
            find_chars(&header, "COOL").expect("temps drawn")
        } else {
            clock
        };
        let gap: String = header.chars().skip(at - 2).take(2).collect();
        assert_eq!(gap, "  ", "no gap before the clock at {cols}: {header}");
        assert!(
            !header.contains("prefix o") || header.contains("prefix on"),
            "detail cut mid-item at {cols}: {header}"
        );
    }
    // A short detail is left whole.
    model.model_detail = "vLLM · kv q8".to_string();
    let header = row_string(&draw(&model, 160, 48), 0);
    assert!(header.contains("KV  vLLM · kv q8"), "{header}");
    // #52: llama-watch now hands the header the engine alone (the rest is
    // in SETUP), so even this name keeps it whole and the slots field.
    model.model_detail = "vLLM".to_string();
    for cols in [160u16, 180, 200, 240, 320] {
        let header = row_string(&draw(&model, cols, 49), 0);
        assert!(header.contains("240k KV  vLLM   slots"), "{cols}: {header}");
    }
}

/// #38: a tool loop's IN, titled with what it holds. The panel is a few
/// rows, so the newest tool result is what stays visible.
#[test]
fn a_tool_loop_in_panel_shows_the_newest_result_under_its_title() {
    let mut model = sample(WatchState::Ready);
    model.in_title = "IN (last request · 3 tool results)".to_string();
    model.out_title = "OUT (last response)".to_string();
    let older = "invented older line ".repeat(60);
    model.in_lines = vec![format!(
        "[tool] {older}\n[tool] invented middle result\n[tool] INVENTED-NEWEST result"
    )];
    let grid = draw(&model, 160, 48);
    let title = row_with(&grid, "IN (last request");
    assert!(
        row_string(&grid, title).contains("3 tool results"),
        "{}",
        row_string(&grid, title)
    );
    let count = grid.rows().saturating_sub(title + 1);
    let left = left_rows(&grid, title + 1, count);
    let last = left
        .iter()
        .rev()
        .find(|row| row.contains("[tool]"))
        .expect("IN text");
    assert!(last.contains("INVENTED-NEWEST result"), "{left:#?}");
    assert!(
        left.iter()
            .any(|row| row.contains("invented middle result")),
        "{left:#?}"
    );
}

// ---- #52: SETUP under single-spaced meters ----------------------------------

/// A common shape: the 12x22 font at 160x49, eighths glyphs, two models
/// loaded and the llama.cpp one generating.
fn setup_model() -> TtyModel {
    let mut model = sample(WatchState::Generating);
    model.chart_glyphs = ChartGlyphs::Eighths;
    let mut setup = qwen_setup();
    setup.more = 1;
    model.setup = Some(setup);
    model
}

const SETUP_GOLDENS: [(&str, u16, u16); 2] = [
    ("setup-160x49.json", 160, 49),
    ("setup-320x90.json", 320, 90),
];

#[test]
fn setup_goldens_match_character_and_colour() {
    for (name, _, _) in SETUP_GOLDENS {
        let fix = load(name);
        assert_frame(name, &fix, &setup_model());
    }
}

#[test]
#[ignore = "run with --ignored to write the #52 SETUP goldens"]
fn dump_setup_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, cols, rows) in SETUP_GOLDENS {
        let grid = draw(&setup_model(), cols, rows);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write setup golden");
    }
}

// ---- #54: Strata's own report in SETUP and on the engine lines ------------

/// 160x49 with Strata generating in a container: SETUP
/// from the invented `fixtures/llama/strata-metrics.json`, the quant from
/// the llama-swap name, and the engine lines with the live phase.
fn strata_setup_model() -> TtyModel {
    let mut model = sglang_model();
    model.chart_glyphs = ChartGlyphs::Eighths;
    model.model_name = "Flash Next Q4_K_M".to_string();
    model.model_detail = "Strata".to_string();
    let body = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/llama/strata-metrics.json"),
    )
    .expect("strata fixture");
    let (_, facts) = llama_watch::metrics::parse_strata(&body);
    let detail = ModelDetail {
        ctx: facts.ctx,
        kv_k: facts.kv.clone(),
        kv_v: facts.kv.clone(),
        ..ModelDetail::default()
    };
    let info = BackendInfo {
        kind: Backend::Strata,
        engine: EngineStats {
            spec_permille: Some(700),
            expert_hit_permille: Some(874),
            pcie_share_permille: Some(92),
            ..EngineStats::default()
        },
        ..BackendInfo::default()
    };
    let rules = Rules::builtin();
    let live = LiveCtx {
        backend: Backend::Strata,
        detail: Some(&detail),
        info: Some(&info),
        engine: Some(&facts.values),
    };
    let cmd = "podman run --rm --name strata -v /models/strata:/data:ro 5e1f0c2d9a7b --config /data/configs/flash-next.json --port 8793";
    model.setup = Some(SetupView {
        id: "flash-next".to_owned(),
        name: "Flash Next Q4_K_M".to_owned(),
        more: 0,
        rows: rules.rows(&rules.extract_all(cmd, "Flash Next Q4_K_M"), &live),
    });
    model.backend_lines = vec![
        "strata  running 1/1 · queued 2 · spec 70 % · prefill 968/s · decode 30.9/s".to_string(),
        "        drafting a reply: outline · gen 40/4,096 · 31.5/s".to_string(),
    ];
    for req in &mut model.requests {
        req.model = "flash-next".to_string();
    }
    model
}

const STRATA_SETUP_GOLDEN: (&str, u16, u16) = ("setup-strata-160x49.json", 160, 49);

#[test]
fn strata_setup_golden_matches_character_and_colour() {
    let (name, _, _) = STRATA_SETUP_GOLDEN;
    assert_frame(name, &load(name), &strata_setup_model());
}

#[test]
#[ignore = "run with --ignored to write the #54 Strata SETUP golden"]
fn dump_strata_setup_golden() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    let (name, cols, rows) = STRATA_SETUP_GOLDEN;
    let grid = draw(&strata_setup_model(), cols, rows);
    std::fs::write(dir.join(name), dump_grid(&grid)).expect("write strata setup golden");
}

/// #54: the five Strata rows fit at 160x49, under the title.
#[test]
fn strata_setup_rows_at_160x49() {
    let grid = draw(&strata_setup_model(), 160, 49);
    let rows = left_rows(&grid, setup_at(&grid), 6);
    assert!(rows[0].contains("flash-next"), "{rows:#?}");
    assert!(
        whole(&grid).contains("Strata") || rows.iter().any(|row| row.contains("engine")),
        "{rows:?}"
    );
}

#[test]
fn setup_title_maps_typography_like_in_and_out() {
    let mut model = setup_model();
    let mut setup = model.setup.clone().expect("setup");
    setup.id = "qwen\u{2019}s \u{2014} caf\u{e9} \u{6027}?".to_owned();
    setup.name = String::new();
    setup.more = 0;
    model.setup = Some(setup);
    let grid = draw(&model, 160, 49);
    let rows = left_rows(&grid, setup_at(&grid), 1);
    assert_eq!(rows[0], "  SETUP  qwen\u{2019}s \u{2014} cafe \u{fffd}?");
}

/// #90: `model.backend_lines` (drawn by [`layout::layout`] with the same
/// `paint_detail`) maps the same way: typography and an accented letter
/// render as the mapped glyph, an unmapped CJK character becomes the
/// placeholder instead of `?`, and a real `?` passes through unchanged.
#[test]
fn backend_lines_map_typography_like_setup() {
    let mut model = setup_model();
    model.backend_lines = vec!["engine \u{201c}now\u{201d} caf\u{e9} \u{6027}?".to_owned()];
    let grid = draw(&model, 160, 49);
    let all = whole(&grid);
    assert!(
        all.contains("engine \u{201c}now\u{201d} cafe \u{fffd}?"),
        "{all}"
    );
}

/// #90: SETUP drops what IN/OUT drop. An escape, a zero-width joiner and a
/// soft hyphen take no cell, so the title reads exactly as `sanitize` would
/// render it; a bidi override is not dropped but drawn as the placeholder.
#[test]
fn setup_title_drops_controls_and_zero_width_like_in_and_out() {
    let mut model = setup_model();
    let mut setup = model.setup.clone().expect("setup");
    setup.id = "a\u{1b}b\u{200d}c\u{ad}\u{202e}d\te".to_owned();
    setup.name = String::new();
    setup.more = 0;
    model.setup = Some(setup);
    let grid = draw(&model, 160, 49);
    let rows = left_rows(&grid, setup_at(&grid), 1);
    assert_eq!(rows[0], "  SETUP  abc\u{fffd}de");
}

/// #90: a dropped character in the id takes no cell, so the name follows
/// the id's drawn width, not its character count.
#[test]
fn setup_title_name_follows_the_drawn_id() {
    let mut model = setup_model();
    let mut setup = model.setup.clone().expect("setup");
    setup.id = "a\u{200d}b".to_owned();
    setup.name = "NAME".to_owned();
    setup.more = 0;
    model.setup = Some(setup);
    let grid = draw(&model, 160, 49);
    let rows = left_rows(&grid, setup_at(&grid), 1);
    assert_eq!(rows[0], "  SETUP  ab \u{b7} NAME");
}

/// Rows from the label column of `grid`, `count` from `top`, trimmed.
fn setup_at(grid: &llama_watch::tty::grid::Grid) -> u16 {
    row_with(grid, "SETUP")
}

fn left_rows(grid: &llama_watch::tty::grid::Grid, top: u16, count: u16) -> Vec<String> {
    let end = grid.cols() / 2 - 2;
    (top..top + count)
        .map(|row| {
            (0..end)
                .map(|col| grid.get(col, row).expect("cell").ch)
                .collect::<String>()
                .trim_end()
                .to_owned()
        })
        .collect()
}

/// #52: SETUP under the headline: the title with `+1`, and as many rows as
/// fit (six at 160x49). Since #165 the meters are the gauge grid on the
/// right, ACT and LOAD on its first row.
#[test]
fn setup_block_sits_under_single_spaced_meters_at_160x49() {
    let grid = draw(&setup_model(), 160, 49);
    let labels: Vec<u16> = ["CPU", "GPU", "VRAM", "MEM", "POWER", "LOAD", "ACT"]
        .into_iter()
        .map(|label| meter_row(&grid, label))
        .collect();
    let timeline = timeline_rule(&grid);
    assert!(
        labels.iter().all(|row| *row < timeline),
        "meters stay in the top band: {labels:?}"
    );
    assert_eq!(labels[6], labels[5], "LOAD sits beside ACT (#165)");
    let rows = left_rows(&grid, setup_at(&grid), 6);
    assert!(
        rows[0].starts_with("  SETUP  qwen3.6-35b-a3b · Qwen 35B"),
        "{rows:#?}"
    );
    let title = setup_at(&grid);
    assert!(
        row_string(&grid, title).contains("+1"),
        "{}",
        row_string(&grid, title)
    );
    assert_eq!(grid.get(2, title).unwrap().fg, C16::White);
    assert!(!whole(&grid).contains("temp 0.6"));
}

/// #163, #165, #170: every gauge is one row of whole cells, never a
/// half-cell end. Solid `▄` by default, in the console and the grid file.
/// With `bar_segments` the segment with the llama-hack font, `▄` with
/// eurlatgr, and the grid file carries the segment either way.
#[test]
fn gauges_are_whole_segment_cells_in_every_font() {
    for (segments, glyphs, glyph, rich) in [
        (true, ChartGlyphs::Eighths, SEG, SEG),
        (true, ChartGlyphs::Halves, '▄', SEG),
        (false, ChartGlyphs::Eighths, '▄', '▄'),
        (false, ChartGlyphs::Halves, '▄', '▄'),
    ] {
        let mut model = setup_model();
        model.chart_glyphs = glyphs;
        model.bar_segments = segments;
        model.cpu_pct = Some(50.0);
        let grid = draw(&model, 160, 49);
        let row = meter_row(&grid, "CPU");
        let col = meter_bar_col(&grid, row, "CPU");
        let cells: Vec<char> = (col..grid.cols())
            .map(|c| grid.get(c, row).unwrap().ch)
            .take_while(|ch| *ch == glyph || *ch == '·')
            .collect();
        assert_eq!(cells.len(), 16, "{glyphs:?}: 16 cells at 160 columns");
        let lit = cells.iter().filter(|ch| **ch == glyph).count();
        assert_eq!(lit, 8, "{glyphs:?}: half of 16");
        assert_eq!(cells[0], glyph, "lit from the first cell");
        assert_eq!(grid.get(col, row).unwrap().fg, C16::Blue);
        assert_eq!(
            grid.get(col, row).unwrap().rich,
            rich,
            "{segments} {glyphs:?}: the file's glyph"
        );
        assert_eq!(
            grid.get(col + 8, row).unwrap().rich,
            '·',
            "the track stays a dot"
        );
        assert!(GLYPHS.contains(&glyph), "term may emit {glyph}");
    }
    // A tiny value still lights one cell.
    let mut model = setup_model();
    model.bar_segments = true;
    model.gpu_pct = Some(1.0);
    let grid = draw(&model, 160, 49);
    let gpu = meter_row(&grid, "GPU");
    let ch = grid.get(meter_bar_col(&grid, gpu, "GPU"), gpu).unwrap().ch;
    assert_eq!(ch, SEG, "a 1% bar lights a cell");
    // 4K too: one row a gauge, GPU and CPU on one row, the row under it
    // the next pair.
    let mut tall = setup_model();
    tall.bar_segments = true;
    tall.cpu_pct = Some(100.0);
    let grid = draw(&tall, 480, 135);
    let cpu = meter_row(&grid, "CPU");
    assert_eq!(meter_row(&grid, "GPU"), cpu, "GPU and CPU share a row");
    assert_eq!(meter_row(&grid, "VRAM"), cpu + 1);
    let col = meter_bar_col(&grid, cpu, "CPU");
    assert_eq!(grid.get(col, cpu).unwrap().ch, SEG);
    assert!(!matches!(grid.get(col, cpu + 1).unwrap().ch, '█' | '▓'));
}

/// #52: a wide screen from 60 rows shows up to nine SETUP rows (#114, one
/// more than the eight the block had); a short one keeps the 160x26 floor
/// (title and three rows beside SLOTS); with nothing loaded the block is
/// gone and RECENT moves back up.
#[test]
fn setup_block_height_follows_the_screen() {
    let grid = draw(&setup_model(), 240, 67);
    let all = whole(&grid);
    assert!(
        all.contains("qwen3.6-35b-a3b") || all.contains("SETUP"),
        "{all}"
    );
    let grid = draw(&setup_model(), 160, 26);
    assert_dashboard(&grid, "160x26 setup");
    assert_eq!(recent_rows_shown(&grid), 0, "{}", whole(&grid));
    assert!(whole(&grid).contains("TIMELINE"), "{}", whole(&grid));
    let mut idle = setup_model();
    idle.setup = None;
    let grid = draw(&idle, 160, 49);
    assert!(!whole(&grid).contains("SETUP"));
    assert!(whole(&grid).contains("TIMELINE"), "{}", whole(&grid));
    // Long values drop whole items from the end; a lone long one is cut.
    let mut long = setup_model();
    let setup = long.setup.as_mut().expect("setup");
    setup.id = "a-very-long-llama-swap-model-id-for-a-bakeoff-case-with-suffix-r2".to_owned();
    setup.rows[1].items[0].text = "x".repeat(90);
    let grid = draw(&long, 160, 49);
    let rows = left_rows(&grid, setup_at(&grid), 4);
    assert!(
        rows[0].contains("a-very-long-llama-swap-model-id-for-a-bakeoff-case-with-suffix-r2"),
        "{rows:#?}"
    );
    assert!(
        !rows[0].contains("Qwen 35B"),
        "the name only when it fits: {rows:#?}"
    );
    assert_blank_edges(&grid);
}

/// The commented `# >>> spec example` block, as a watch.toml would hold it.
fn spec_example_rules() -> Rules {
    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packaging/watch.example.toml"),
    )
    .expect("example");
    let block: String = text
        .split("# >>> spec example\n")
        .nth(1)
        .expect("start marker")
        .split("# <<< spec example")
        .next()
        .expect("end marker")
        .lines()
        .filter(|line| !line.starts_with("# "))
        .map(|line| line.strip_prefix('#').unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n");
    let config = Config::from_toml(&format!("[setup]\n{block}\n")).expect("spec example parses");
    // The absolute path is checked and not opened.
    config
        .validate(8)
        .expect("spec example validates without opening specs_dir");
    assert!(!config.setup.defaults);
    assert_eq!(
        config
            .setup
            .specs_dir
            .as_deref()
            .map(|path| path.to_string_lossy().into_owned()),
        Some("/etc/llama-bored/specs".to_owned())
    );
    Rules::compile(&config.setup).expect("compile")
}

/// #92, #114: the example rules over `fixtures/llama/example-spec.json`.
/// `report.label` is its own row, drawn first. The engine row keeps the
/// engine, quant and parameters. A tall screen shows every row. 160x49
/// keeps the first five data rows and drops think, serve and sample.
/// The file's paths stay off the grid.
#[test]
fn spec_example_rules_draw_the_fixture() {
    let rules = spec_example_rules();
    let body = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/llama/example-spec.json"),
    )
    .expect("fixture");
    let doc: serde_json::Value = serde_json::from_str(&body).expect("json");
    let found = rules.extract_with("", "example-35b-q6", None, Some(&doc));
    let live = LiveCtx {
        backend: Backend::LlamaCpp,
        detail: None,
        info: None,
        engine: None,
    };
    let rows = rules.rows(&found, &live);
    let drawn: Vec<(String, String)> = rows
        .iter()
        .map(|row| {
            let text = row
                .items
                .iter()
                .enumerate()
                .map(|(i, item)| {
                    if i == 0 {
                        item.text.clone()
                    } else {
                        format!("{}{}", item.sep, item.text)
                    }
                })
                .collect();
            (row.label.clone(), text)
        })
        .collect();
    assert_eq!(
        drawn,
        [
            (
                "label".to_owned(),
                "Example-35B Q6 \u{00b7} llama.cpp 0.6, 18 expert layers in RAM, 131k".to_owned()
            ),
            (
                "engine".to_owned(),
                "llama.cpp 0.6 \u{00b7} Q6_K_XL \u{00b7} params 35B / 3B \u{00b7} moe".to_owned()
            ),
            ("ctx".to_owned(), "131,072 · kv f16".to_owned()),
            (
                "experts".to_owned(),
                "18 layers in RAM · gpu 999 · mmap off".to_owned()
            ),
            ("spec".to_owned(), "none".to_owned()),
            ("think".to_owned(), "on · budget 24,000".to_owned()),
            (
                "serve".to_owned(),
                "multi-agent · store nas · role untested".to_owned()
            ),
            (
                "sample".to_owned(),
                "temp 0.6 · top-p 0.95 · top-k 20 · min-p 0 · pres 0".to_owned()
            ),
        ]
    );

    let mut model = sample(WatchState::Generating);
    model.chart_glyphs = ChartGlyphs::Eighths;
    model.setup = Some(SetupView {
        id: "example-35b-q6".to_owned(),
        name: "Example 35B".to_owned(),
        more: 0,
        rows,
    });
    let tall = draw(&model, 160, 60);
    let all = whole(&tall);
    assert!(all.contains("example-35b-q6"), "id missing");
    for leak in ["/models", "sha256", "d77d5bc", "retry wave", "8096"] {
        assert!(!all.contains(leak), "{leak}");
    }
}

fn setup_item(text: &str, text_value: bool) -> SetupItem {
    SetupItem {
        sep: " \u{00b7} ".to_owned(),
        text: text.to_owned(),
        dim: false,
        text_value,
    }
}

/// One SETUP row on the generating sample, so the block sits where it
/// always does under the meters.
fn with_setup_rows(rows: Vec<SetupRow>) -> TtyModel {
    let mut model = setup_model();
    let setup = model.setup.as_mut().expect("setup");
    setup.rows = rows;
    setup.more = 0;
    model
}

/// #114: the value is cut to the row after `max_chars`, and items after a
/// text value drop before that cut. An item before the value stays.
#[test]
fn setup_text_is_cut_to_the_row_and_later_items_drop() {
    // Value column is 13; the left half ends at column 77, 65 cells.
    // `left_rows` keeps columns 0..78, so a full cut ends on that column.
    let long = "a".repeat(80);
    let model = with_setup_rows(vec![SetupRow {
        label: "note".to_owned(),
        items: vec![setup_item(&long, true), setup_item("TAIL", false)],
    }]);
    let grid = draw(&model, 160, 49);
    let row = &left_rows(&grid, setup_at(&grid) + 1, 1)[0];
    assert!(!row.contains("TAIL"), "{row}");
    assert!(row.ends_with('\u{2026}'), "{row}");
    assert_eq!(row.chars().count(), 78, "{row}");
    assert_eq!(row.chars().filter(|ch| *ch == 'a').count(), 64);

    let kept = "b".repeat(60);
    let model = with_setup_rows(vec![SetupRow {
        label: "note".to_owned(),
        items: vec![setup_item(&kept, true), setup_item("TAIL", false)],
    }]);
    let grid = draw(&model, 160, 49);
    let row = &left_rows(&grid, setup_at(&grid) + 1, 1)[0];
    assert!(row.contains(&kept) && !row.contains("TAIL"), "{row}");
    assert!(
        !row.contains('\u{2026}'),
        "the value fits, so it is not cut: {row}"
    );

    let model = with_setup_rows(vec![SetupRow {
        label: "note".to_owned(),
        items: vec![setup_item("head", false), setup_item(&long, true)],
    }]);
    let grid = draw(&model, 160, 49);
    let row = &left_rows(&grid, setup_at(&grid) + 1, 1)[0];
    assert!(row.contains("head"), "{row}");
    assert!(
        row.ends_with('\u{2026}'),
        "the text is cut, not dropped: {row}"
    );
    assert_eq!(row.chars().count(), 78, "{row}");
}

/// #114: a head that leaves fewer than two cells drops the text value and
/// its separator. 61, 62 and 63 characters at 160 columns (65 cells) leave
/// 1, 0 and 0 after the ` · `.
#[test]
fn a_text_item_with_no_room_drops_with_its_separator() {
    for n in [61, 62, 63] {
        let head = "h".repeat(n);
        let model = with_setup_rows(vec![SetupRow {
            label: "note".to_owned(),
            items: vec![setup_item(&head, false), setup_item("LABEL", true)],
        }]);
        let grid = draw(&model, 160, 49);
        let row = &left_rows(&grid, setup_at(&grid) + 1, 1)[0];
        assert_eq!(row, &format!("    note     {head}"), "{n}: {row}");
        assert!(!row.contains('\u{2026}'), "{n}: {row}");
    }
}

/// #114: a row cut that lands on a space does not draw ` …`.
#[test]
fn setup_text_cut_drops_the_space_before_the_ellipsis() {
    let text = format!("{} zzzz", "a".repeat(63));
    let model = with_setup_rows(vec![SetupRow {
        label: "note".to_owned(),
        items: vec![setup_item(&text, true)],
    }]);
    let grid = draw(&model, 160, 49);
    let row = &left_rows(&grid, setup_at(&grid) + 1, 1)[0];
    assert!(row.ends_with(&format!("a{}", '\u{2026}')), "{row}");
    assert!(!row.contains(" \u{2026}"), "{row}");
    assert!(!row.contains('z'), "{row}");
}

/// Cell text of one terminal frame: the writer's own sequences (ESC % G
/// and CSI) are skipped. Anything else, including OSC, is left in place
/// so a control from the spec fails the check below.
fn term_cell_text(bytes: &[u8]) -> String {
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1b {
            i += 1;
            assert!(i < bytes.len(), "dangling ESC");
            match bytes[i] {
                b'[' => {
                    i += 1;
                    while i < bytes.len() && !(0x40..=0x7e).contains(&bytes[i]) {
                        assert!(
                            (0x20..=0x3f).contains(&bytes[i]),
                            "CSI param {:02x}",
                            bytes[i]
                        );
                        i += 1;
                    }
                    assert!(i < bytes.len(), "truncated CSI");
                    i += 1;
                }
                b'%' => {
                    i += 1;
                    assert!(i < bytes.len() && bytes[i] == b'G', "not ESC % G");
                    i += 1;
                }
                other => {
                    out.push(0x1b);
                    out.push(other);
                    i += 1;
                }
            }
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).expect("cell text is utf-8")
}

/// #114: a spec string is drawn like other dashboard text. `·` stays `·`.
/// The console cell is the #90 transliteration; the grid file's `rich`
/// scalar keeps the original character when it is showable. NUL, DEL, CR,
/// C1, the escape, zero-width characters and the newline never become a
/// cell. A bidi override and a combining mark become the placeholder.
#[test]
fn setup_text_keeps_the_original_character_and_drops_controls() {
    let rules = Rules::compile(&llama_watch::config::Setup {
        defaults: false,
        field: vec![llama_watch::config::SetupField {
            row: "note".to_owned(),
            source: Some("spec:report.label".to_owned()),
            kind: Some("text".to_owned()),
            order: 10,
            ..llama_watch::config::SetupField::default()
        }],
        ..llama_watch::config::Setup::default()
    })
    .expect("rule");
    let doc = serde_json::json!({
        "report": {
            "label": "Q4 \u{00b7} caf\u{e9} \u{2192} \u{4e2d} \u{0}\u{7f}\r\u{85}\u{9b}\u{9c}\u{9d}\u{1b}]9;SPEC\u{7}\n\u{202e}\u{200b}\u{200d}line\u{301}"
        }
    });
    let found = rules.extract_with("", "example", None, Some(&doc));
    assert_eq!(
        found
            .iter()
            .map(|item| item.value.as_str())
            .collect::<Vec<_>>(),
        ["Q4 \u{00b7} caf\u{e9} \u{2192} \u{4e2d} ]9;SPEC \u{202e}line\u{301}"]
    );
    let live = LiveCtx {
        backend: llama_core::backend::Backend::LlamaCpp,
        detail: None,
        info: None,
        engine: None,
    };
    let mut model = with_setup_rows(rules.rows(&found, &live));
    model.setup.as_mut().expect("setup").id = "example".to_owned();
    model.setup.as_mut().expect("setup").name.clear();
    let grid = draw(&model, 160, 49);
    let row = setup_at(&grid) + 1;
    // The text under test is the left half.
    let end = grid.cols() / 2 - 2;
    let console = left_rows(&grid, row, 1).pop().expect("row");
    let rich: String = (0..end)
        .map(|col| grid.get(col, row).expect("cell").rich)
        .collect::<String>()
        .trim_end()
        .to_owned();
    assert_eq!(
        console,
        "    note     Q4 \u{00b7} cafe > \u{fffd} ]9;SPEC \u{fffd}line\u{fffd}"
    );
    assert_eq!(
        rich,
        "    note     Q4 \u{00b7} caf\u{e9} \u{2192} \u{fffd} ]9;SPEC \u{fffd}line\u{fffd}"
    );
    assert!(!row_string(&grid, row + 1).contains("line"));
    for cell in grid.cells() {
        assert!(!cell.ch.is_control(), "console cell {:?}", cell.ch);
        assert!(!cell.rich.is_control(), "grid cell {:?}", cell.rich);
    }

    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("setup-text-{}-hostile", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("dir");
    let path = dir.join("grid");
    let mut writer =
        llama_watch::grid_out::GridWriter::new(&path, llama_core::palette::Palette::Llama);
    assert!(writer.offer(&grid, Instant::now(), 1_000).expect("write"));
    let image = llama_core::gridfile::decode(&std::fs::read(&path).expect("read")).expect("decode");
    assert_eq!((image.cols, image.rows), (grid.cols(), grid.rows()));
    let cols = usize::from(image.cols);
    let at = usize::from(row) * cols;
    let file_row: String = image.cells[at..at + usize::from(end)]
        .iter()
        .map(|cell| cell.ch)
        .collect::<String>()
        .trim_end()
        .to_owned();
    assert_eq!(file_row, rich);
    for cell in &image.cells {
        assert!(!cell.ch.is_control(), "grid file cell {:?}", cell.ch);
    }

    let mut term = Term::new(
        Vec::new(),
        || {
            Ok(Size {
                cols: 160,
                rows: 49,
            })
        },
        Duration::from_secs(60),
        Instant::now(),
    )
    .expect("term");
    term.render(&grid, Instant::now()).expect("render");
    let text = term_cell_text(term.out());
    assert!(
        text.contains("Q4 \u{00b7} cafe > \u{fffd} ]9;SPEC \u{fffd}line\u{fffd}"),
        "{text}"
    );
    assert!(
        text.chars().all(|ch| !ch.is_control()),
        "a control from the spec reached the writer: {text:?}"
    );
}

/// #114: at 60 rows the block may take one more row (nine, title included).
/// The row under the block, RECENT and the health line stay put at every
/// size the dashboard draws.
#[test]
fn setup_gains_one_row_at_60_without_pushing_the_panels_out() {
    let rows: Vec<SetupRow> = (1..=8)
        .map(|n| SetupRow {
            label: format!("u{n}"),
            items: vec![setup_item(&format!("kept-{n}"), n == 1)],
        })
        .collect();
    let model = with_setup_rows(rows);
    let sizes = [
        (160u16, 26u16),
        (160, 48),
        (160, 59),
        (160, 60),
        (192, 60),
        (240, 67),
        (286, 60),
        (320, 90),
        (480, 135),
    ];
    for (cols, rows) in sizes {
        let grid = draw(&model, cols, rows);
        let at = format!("{cols}x{rows}");
        assert_dashboard(&grid, &at);
        if recent_rows_shown(&grid) == 0 {
            assert!(whole(&grid).contains("SETUP") || rows <= 26, "{at}");
            continue;
        }
        let setup_at = row_with(&grid, "SETUP");
        let recent = row_with(&grid, "RECENT");
        let health = grid.rows() - 2;
        assert!(setup_at < recent, "{at}: SETUP overlaps RECENT");
        assert!(recent < health, "{at}: RECENT overlaps the health line");
        assert!(
            req_rule_row(&grid) < grid.rows() - 3,
            "{at}: RECENT's rule meets the health rule"
        );
        assert!(recent_rows_shown(&grid) >= 4, "{at}: RECENT lost its floor");
        let shown: Vec<bool> = (1..=8)
            .map(|n| whole(&grid).contains(&format!("kept-{n}")))
            .collect();
        // The top band keeps a few SETUP lines, not the old eight-row block.
        assert!(shown[0], "{at}: the first setup row should fit: {shown:?}");
        assert!(!row_string(&grid, health).contains("kept-"), "{at}");
        let _ = recent;
        assert_blank_edges(&grid);
    }
}

#[test]
#[ignore = "dev tool: print frames as text"]
fn print_frames() {
    for (name, model, cols, rows) in [
        (
            "generating 160x49",
            sample(WatchState::Generating),
            160u16,
            49u16,
        ),
        ("vllm 160x49", vllm_model(), 160, 49),
        ("generating 240x67", sample(WatchState::Generating), 240, 67),
        ("sglang 160x26", sglang_model(), 160, 26),
    ] {
        println!("==== {name}");
        let grid = draw(&model, cols, rows);
        for row in 0..grid.rows() {
            println!("{}", row_string(&grid, row));
        }
    }
}

#[test]
#[ignore = "dev tool: print the SETUP golden's top-left as text"]
fn print_setup_corner() {
    let grid = draw(&setup_model(), 160, 49);
    for line in left_rows(&grid, 0, 18) {
        println!("{line}");
    }
}

// ---- #74: TEMPS panel ------------------------------------------------------

use llama_watch::sources::temps::{Level, TempGroup, TempItem, TempPanel};

fn temp_item(label: &str, c: i32, level: Level) -> TempItem {
    // 80/90 °C, the anchors a reading without chip limits uses.
    temp_marked(label, c, level, Some((800, 900)))
}

fn temp_marked(label: &str, c: i32, level: Level, thresholds: Option<(i32, i32)>) -> TempItem {
    TempItem {
        label: label.to_string(),
        tenths: c * 10,
        level,
        thresholds,
    }
}

/// The reference host's TEMPS rows (invented values), one warm NVMe sensor
/// and one hot board input to show the colours.
fn ref_temps() -> TempPanel {
    use Level::{Crit, Normal, Warn};
    let group = |name: &str, items: Vec<TempItem>| TempGroup {
        name: name.to_string(),
        items,
    };
    TempPanel {
        groups: vec![
            group(
                "CPU",
                vec![
                    temp_item("Tctl", 68, Normal),
                    temp_item("CCD1", 64, Normal),
                    temp_item("CCD2", 62, Normal),
                ],
            ),
            group("GPU", vec![temp_item("", 71, Normal)]),
            group(
                "coolant",
                vec![temp_marked("", 40, Normal, Some((450, 550)))],
            ),
            // Composite only: the tty hides an NVMe's Sensor 1 and 2, and
            // the board's AUXTIN2 (#165, temps::TTY_BLOCK).
            group("NVMe0", vec![temp_item("", 82, Warn)]),
            group("NVMe1", vec![temp_item("", 66, Normal)]),
            group(
                "board",
                vec![
                    temp_item("sys", 51, Normal),
                    temp_item("cpu", 51, Normal),
                    temp_item("aux0", 27, Normal),
                    temp_item("aux1", 92, Crit),
                    temp_item("aux3", 27, Normal),
                    temp_item("aux4", 50, Normal),
                ],
            ),
            group("NIC", vec![temp_item("", 51, Normal)]),
            group("Wi-Fi", vec![temp_item("", 39, Normal)]),
        ],
    }
}

/// The fans golden model with TEMPS on, and the 10x18 font's eighths.
fn temps_model() -> TtyModel {
    let mut model = fans_model();
    model.temps = Some(ref_temps());
    model
}

/// #74 goldens: 192x60 is the 10x18 font at 1920x1080 (FANS and TEMPS
/// beside IN/OUT), 240x67 a wider screen, 160x49 the 12x22 font (TEMPS on
/// one line, FANS hidden).
const TEMPS_GOLDENS: [(&str, u16, u16); 3] = [
    ("temps-192.json", 192, 60),
    ("temps-240.json", 240, 67),
    ("temps-160x49.json", 160, 49),
];

#[test]
fn temps_goldens_match_character_and_colour() {
    for (name, _, _) in TEMPS_GOLDENS {
        let fix = load(name);
        assert_frame(name, &fix, &temps_model());
    }
}

/// #116: the grid file at 192x60 with exact colours, from the synthetic
/// temps frame and a hot one (activity past 100, generation past its ceiling).
type GridGolden = fn() -> TtyModel;

const GRID_EXACT_GOLDENS: [(&str, GridGolden); 5] = [
    ("temps-192.json", temps_model),
    ("hot-192.json", hot_model),
    ("job-192.json", job_grid_model),
    ("claims-192.json", claims_grid_model),
    ("recent-cache-192.json", cache_rows_model),
];

fn claims_grid_model() -> TtyModel {
    header_claim_model(hold_claim())
}

fn job_grid_model() -> TtyModel {
    job_model(example_job(), 0)
}

fn hot_model() -> TtyModel {
    let mut model = temps_model();
    model.activity_pct = Some(118.0);
    let ceiling = model.gen_ceiling;
    model.gen_tps = Some(ceiling * 2.0);
    for bucket in &mut model.chart {
        if bucket.gen_tps.is_some() {
            bucket.gen_tps = Some(ceiling * 2.0);
        }
    }
    model
}

/// One screen row per line. A run is `"n,RRGGBB,RRGGBB,ch"`: the repeat
/// count, foreground, background, then the character (a comma in the
/// character stays with it).
#[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
struct GridFix {
    w: u16,
    h: u16,
    rows: Vec<Vec<String>>,
}

fn grid_fix_of(model: &TtyModel) -> GridFix {
    let grid = layout_exact(model, 192, 60);
    let image = llama_watch::grid_out::image_of(&grid, llama_core::palette::Palette::Llama, 1, 0);
    let width = usize::from(image.cols);
    assert_eq!((image.cols, image.rows), (192, 60));
    let mut rows = Vec::with_capacity(usize::from(image.rows));
    for cells in image.cells.chunks(width) {
        let mut runs = Vec::new();
        let mut n = 0u32;
        let mut ch = ' ';
        let mut fg = String::new();
        let mut bg = String::new();
        for cell in cells {
            let cell_fg = hex6(cell.fg);
            let cell_bg = hex6(cell.bg);
            if n > 0 && cell.ch == ch && cell_fg == fg && cell_bg == bg {
                n += 1;
            } else {
                if n > 0 {
                    runs.push(format!("{n},{fg},{bg},{ch}"));
                }
                n = 1;
                ch = cell.ch;
                fg = cell_fg;
                bg = cell_bg;
            }
        }
        if n > 0 {
            runs.push(format!("{n},{fg},{bg},{ch}"));
        }
        let covered: u32 = runs.iter().map(|run| run_count(run)).sum();
        assert_eq!(covered, u32::from(image.cols));
        rows.push(runs);
    }
    assert_eq!(rows.len(), usize::from(image.rows));
    GridFix {
        w: image.cols,
        h: image.rows,
        rows,
    }
}

fn hex6(rgb: llama_core::color::Rgb) -> String {
    format!("{:02X}{:02X}{:02X}", rgb.r, rgb.g, rgb.b)
}

fn run_count(run: &str) -> u32 {
    run.split_once(',')
        .expect("n,fg,bg,ch")
        .0
        .parse()
        .expect("n")
}

/// Compact JSON: `{"w":N,"h":N,"rows":[` then one JSON array per screen row.
fn grid_fix_text(fix: &GridFix) -> String {
    let mut out = format!("{{\"w\":{},\"h\":{},\"rows\":[\n", fix.w, fix.h);
    for (i, row) in fix.rows.iter().enumerate() {
        if i > 0 {
            out.push_str(",\n");
        }
        out.push_str(&serde_json::to_string(row).expect("row"));
    }
    out.push_str("\n]}\n");
    out
}

fn grid_fix_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/grid")
}

#[test]
fn grid_exact_goldens_match() {
    for (name, model) in GRID_EXACT_GOLDENS {
        let path = grid_fix_dir().join(name);
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {name}: {err}"));
        let fix: GridFix =
            serde_json::from_str(&text).unwrap_or_else(|err| panic!("{name}: {err}"));
        let got = grid_fix_of(&model());
        assert_eq!(got, fix, "{name}");
        assert_eq!(text, grid_fix_text(&got), "{name}: compact form");
    }
}

#[test]
#[ignore = "run with --ignored to write the #116 grid goldens"]
fn dump_grid_exact_goldens() {
    let dir = grid_fix_dir();
    std::fs::create_dir_all(&dir).expect("fixtures/grid");
    for (name, model) in GRID_EXACT_GOLDENS {
        let text = grid_fix_text(&grid_fix_of(&model()));
        std::fs::write(dir.join(name), text).expect("write");
    }
}

/// What `llama-view --source grid --colors truecolor` writes for one busy
/// frame, palette RGB against the ramp. The counts are printed; the ramp
/// has to cost more bytes, because a run of distinct colours is a run of
/// SGR sequences.
#[test]
fn truecolor_view_writes_more_for_the_ramps() {
    let model = hot_model();
    let plain = llama_watch::grid_out::image_of(
        &layout(&model, 192, 60),
        llama_core::palette::Palette::Llama,
        1,
        0,
    );
    let ramp = llama_watch::grid_out::image_of(
        &layout_exact(&model, 192, 60),
        llama_core::palette::Palette::Llama,
        1,
        0,
    );
    let (plain_bytes, plain_colours) = truecolor_frame(&plain);
    let (ramp_bytes, ramp_colours) = truecolor_frame(&ramp);
    eprintln!(
        "llama-view truecolor 192x60 busy frame, one full first render: palette {plain_bytes} bytes, {plain_colours} colour pairs; exact {ramp_bytes} bytes, {ramp_colours} colour pairs"
    );
    assert!(ramp_bytes > plain_bytes);
    assert!(ramp_colours > plain_colours);
}

fn truecolor_frame(image: &llama_core::gridfile::GridImage) -> (usize, usize) {
    let screen = llama_view::screen_of(image, false);
    let mut renderer = llama_view::Renderer::with_colors(
        llama_view::ColorMode::Truecolor,
        llama_view::Palette::Llama,
    )
    .synchronized(true);
    let bytes = renderer
        .render(screen.cols, screen.rows, &screen.cells)
        .len();
    let mut colours = std::collections::BTreeSet::new();
    for cell in &image.cells {
        colours.insert((
            cell.fg.r, cell.fg.g, cell.fg.b, cell.bg.r, cell.bg.g, cell.bg.b,
        ));
    }
    (bytes, colours.len())
}

#[test]
#[ignore = "run with --ignored to write the #74 temps goldens"]
fn dump_temps_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, cols, rows) in TEMPS_GOLDENS {
        let grid = draw(&temps_model(), cols, rows);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write temps golden");
    }
}

#[test]
fn every_temps_row_is_one_gauge_at_240x67() {
    let grid = draw(&temps_model(), 240, 67);
    let mut rows = Vec::new();
    for name in ["GPU", "coolant", "NVMe0", "NVMe1", "board", "NIC", "Wi-Fi"] {
        rows.push(meter_row(&grid, name));
    }
    assert!(
        rows.iter().all(|row| *row < timeline_rule(&grid)),
        "{rows:?}"
    );
    // CPU is the temperature gauge under the work gauges, beside GPU.
    let gpu = meter_row(&grid, "GPU");
    let cpu_temp = row_string(&grid, gpu);
    assert!(cpu_temp.contains("CPU           68 C"), "{cpu_temp}");
    assert_blank_edges(&grid);
}

#[test]
fn temps_values_take_their_level_colours() {
    let grid = draw(&temps_model(), 240, 67);
    for (name, value, fg) in [
        ("NVMe1", "66 C", C16::BrightWhite),
        ("NVMe0", "82 C", C16::Yellow),
        ("board", "92 C", C16::BrightRed),
    ] {
        let row = meter_row(&grid, name);
        let text = row_string(&grid, row);
        assert!(text.contains(&format!("{name}{value:>13}")), "{text}");
        let col = col_of(&grid, row, value);
        assert_eq!(grid.get(col, row).unwrap().fg, fg, "{name}");
    }
    assert_blank_edges(&grid);
}

#[test]
fn at_160x49_temps_and_fans_are_gauges_in_the_top_band() {
    let grid = draw(&temps_model(), 160, 49);
    for name in ["coolant", "NVMe0", "Wi-Fi", "front1", "top"] {
        assert!(meter_row(&grid, name) < timeline_rule(&grid), "{name}");
    }
    assert!(!whole(&grid).contains("TEMPS"), "no TEMPS line");
    assert_blank_edges(&grid);
}

/// #165: the temperature gauges, a blank row, then the fan gauges.
#[test]
fn temperature_gauges_come_before_fan_gauges() {
    for (cols, rows) in [(160u16, 67u16), (240, 67)] {
        let grid = draw(&temps_model(), cols, rows);
        let temps = meter_row(&grid, "Wi-Fi");
        let fans = meter_row(&grid, "front1");
        assert_eq!(fans, temps + 2, "{cols}x{rows}: one blank row between");
        let blank: String = row_string(&grid, temps + 1)
            .chars()
            .skip(usize::from(cols / 2))
            .collect();
        assert!(blank.trim().is_empty(), "{cols}x{rows}: {blank}");
        assert!(fans < timeline_rule(&grid), "{cols}x{rows}");
        assert_blank_edges(&grid);
    }
}

#[test]
fn text_off_keeps_the_temperature_gauges_in_the_top_band() {
    let mut model = temps_model();
    model.show_text = false;
    let grid = draw(&model, 160, 49);
    for name in ["coolant", "NVMe0", "board"] {
        assert!(meter_row(&grid, name) < timeline_rule(&grid), "{name}");
    }
    assert_blank_edges(&grid);
}

#[test]
fn temps_alone_and_temps_off() {
    let mut model = temps_model();
    model.fans = None;
    let grid = draw(&model, 192, 60);
    assert!(meter_row(&grid, "Wi-Fi") < timeline_rule(&grid));
    model.temps = Some(TempPanel::default());
    let grid = draw(&model, 192, 60);
    let note = row_string(&grid, row_with(&grid, "TEMPS"));
    assert!(
        note.contains("TEMPS  no temperature inputs found"),
        "{note}"
    );
    model.temps = None;
    for (cols, rows) in [(160u16, 49u16), (192, 60), (240, 67)] {
        let grid = draw(&model, cols, rows);
        let all = whole(&grid);
        assert!(
            !all.contains("TEMPS") && !all.contains("coolant"),
            "{cols}x{rows}"
        );
    }
}

// ---- #75, #96: RECENT context bar and in-flight rows ----------------------

use llama_watch::tty::layout::{CtxCell, CtxPart, InFlight, SPIN, Spin, ctx_cells};

/// 30 cells of 6,400 tokens: an eighth is 800 tokens.
const W: usize = 30;
const SCALE: u64 = W as u64 * 8 * 800;

fn one(value: u64) -> Vec<CtxCell> {
    ctx_cells(&[(CtxPart::Cached, value)], SCALE, W, None)
}

fn lit(cells: &[CtxCell]) -> Vec<CtxCell> {
    cells
        .iter()
        .copied()
        .filter(|c| c.part != CtxPart::Track)
        .collect()
}

fn full(part: CtxPart) -> CtxCell {
    CtxCell {
        part,
        eighths: 8,
        cursor: None,
    }
}

fn rem(part: CtxPart, eighths: u8) -> CtxCell {
    CtxCell {
        part,
        eighths,
        cursor: None,
    }
}

#[test]
fn ctx_cells_are_linear_with_eighths_and_a_floor_of_two() {
    use CtxPart::Cached;
    assert_eq!(
        one(0),
        vec![
            CtxCell {
                part: CtxPart::Track,
                eighths: 1,
                cursor: None
            };
            W
        ]
    );
    // Exactly one cell: no remainder cell.
    assert_eq!(lit(&one(6_400)), [full(Cached)]);
    // One eighth short of a cell: 7 eighths, well above the floor.
    assert_eq!(lit(&one(5_600)), [rem(Cached, 7)]);
    // Three eighths: still above the floor, shown exactly.
    assert_eq!(lit(&one(2_400)), [rem(Cached, 3)]);
    // One eighth past a cell: the whole cell first, then the remainder,
    // floored to ▂ since one eighth alone is under the floor (#96).
    assert_eq!(lit(&one(7_200)), [full(Cached), rem(Cached, 2)]);
    // A non-zero segment at or under one eighth still shows at least `▂`
    // (#96): MIN_E is 2, not 1.
    assert_eq!(lit(&one(1)), [rem(Cached, 2)]);
    assert_eq!(lit(&one(799)), [rem(Cached, 2)]);
    assert_eq!(lit(&one(800)), [rem(Cached, 2)]);
    assert_eq!(lit(&one(1_600)), [rem(Cached, 2)]);
    // 13 cells and 5/8.
    let cells = lit(&one(13 * 6_400 + 5 * 800));
    assert_eq!(cells.len(), 14);
    assert!(cells[..13].iter().all(|c| *c == full(Cached)));
    assert_eq!(cells[13], rem(Cached, 5));
    // Over the scale: clipped to the width, no track left.
    assert_eq!(lit(&one(SCALE * 2)).len(), W);
}

#[test]
fn no_spin_draws_no_cursor_and_the_same_counts_give_the_same_cells() {
    let parts = [
        (CtxPart::Cached, 88_960),
        (CtxPart::New, 2_244),
        (CtxPart::Out, 612),
    ];
    let a = ctx_cells(&parts, 262_144, 24, None);
    let b = ctx_cells(&parts, 262_144, 24, None);
    assert_eq!(a, b, "a finished row draws the same bar every time");
    assert!(a.iter().all(|c| c.cursor.is_none()), "{a:?}");
}

#[test]
fn each_segment_ends_with_its_own_remainder_cell() {
    use CtxPart::{Cached, Out};
    // 2 cells + 3/8, 1 cell + 3/8.
    let cells = lit(&ctx_cells(
        &[(Cached, 2 * 6_400 + 3 * 800), (Out, 6_400 + 3 * 800)],
        SCALE,
        W,
        None,
    ));
    assert_eq!(
        cells,
        [
            full(Cached),
            full(Cached),
            rem(Cached, 3),
            full(Out),
            rem(Out, 3),
        ]
    );
}

#[test]
fn the_new_segment_steps_lower_than_cached_and_out() {
    use CtxPart::{Cached, New};
    // Same value, same scale: cached is a full `▇` (8 eighths); new, the
    // step option (#96), draws the same whole cell lower, `▄` (4 eighths).
    assert_eq!(
        lit(&ctx_cells(&[(Cached, 6_400)], SCALE, W, None)),
        [full(Cached)]
    );
    assert_eq!(
        lit(&ctx_cells(&[(New, 6_400)], SCALE, W, None)),
        [rem(New, 4)]
    );
    // A partial new cell clamps to 2..=3, rounded from its eighths height,
    // never reaching the whole cell's 4 until it is whole.
    for (value, want) in [
        (800u64, 2),
        (1_600, 2),
        (2_400, 2),
        (3_200, 2),
        (4_000, 3),
        (4_800, 3),
        (5_600, 3),
    ] {
        let cells = lit(&ctx_cells(&[(New, value)], SCALE, W, None));
        assert_eq!(cells, [rem(New, want)], "{value}");
    }
}

#[test]
fn the_head_cursor_is_the_growing_segments_last_cell_and_cycles_with_beats() {
    use CtxPart::{Cached, New, Out};
    // Prefill: new is growing. Its last cell becomes a full `▇` cursor in
    // SPIN[beats % 4], cycling beat by beat (#96).
    for beats in 0..8u32 {
        let spin = Spin {
            decoding: false,
            beats,
        };
        let cells = lit(&ctx_cells(
            &[(Cached, 6_400), (New, 6_400 + 3 * 800)],
            SCALE,
            W,
            Some(spin),
        ));
        let last = *cells.last().unwrap();
        assert_eq!(last.part, New, "{beats}");
        assert_eq!(last.eighths, 8, "a full cursor, not the stepped height");
        assert_eq!(last.cursor, Some(SPIN[(beats % 4) as usize]), "{beats}");
        assert!(
            cells[..cells.len() - 1].iter().all(|c| c.cursor.is_none()),
            "only the head cell spins: {cells:?}"
        );
    }
    // Decoding: out is growing instead.
    let cells = lit(&ctx_cells(
        &[(Cached, 6_400), (New, 800), (Out, 1_600)],
        SCALE,
        W,
        Some(Spin {
            decoding: true,
            beats: 1,
        }),
    ));
    let last = *cells.last().unwrap();
    assert_eq!(last.part, Out);
    assert_eq!(last.cursor, Some(SPIN[1]));
}

#[test]
fn an_empty_growing_segment_gets_a_one_cell_cursor() {
    use CtxPart::{Cached, New, Out};
    // Decoding has just started: `out` is growing but still empty, so a
    // single cursor cell is inserted after `new` (#96).
    let spin = Spin {
        decoding: true,
        beats: 2,
    };
    let cells = lit(&ctx_cells(
        &[(Cached, 6_400), (New, 800)],
        SCALE,
        W,
        Some(spin),
    ));
    let last = *cells.last().unwrap();
    assert_eq!(last.part, Out);
    assert_eq!(last.eighths, 8);
    assert_eq!(last.cursor, Some(SPIN[2]));
}

/// A Qwen row of known context, drawn at `cols`.
fn bar_model(rows: Vec<Activity>, glyphs: ChartGlyphs) -> TtyModel {
    let mut model = sample(WatchState::Generating);
    model.requests = rows;
    model.chart_glyphs = glyphs;
    model
}

fn bar_of(grid: &llama_watch::tty::grid::Grid, row: u16) -> Vec<(char, C16)> {
    // The bar is the run of bar cells before the status column.
    let cells: Vec<(char, C16)> = (0..grid.cols())
        .map(|col| {
            let cell = grid.get(col, row).unwrap();
            (cell.ch, cell.fg)
        })
        .collect();
    let end = cells
        .iter()
        .rposition(|(ch, _)| is_bar_char(*ch) || *ch == '!')
        .unwrap();
    let mut start = end;
    while start > 0 && (is_bar_char(cells[start - 1].0) || cells[start - 1].0 == '!') {
        start -= 1;
    }
    cells[start..=end].to_vec()
}

#[test]
fn bar_segments_use_fixed_colours_and_the_new_segment_steps_lower() {
    let row = |n_ctx| {
        let mut req = recent_model().requests[0].clone();
        (req.input_tok, req.cached_tok, req.output_tok) = (60_000, 40_000, 3_000);
        req.n_ctx = n_ctx;
        req
    };
    let grid = draw(
        &bar_model(vec![row(Some(262_144))], ChartGlyphs::Eighths),
        240,
        67,
    );
    let header = row_with(&grid, "RECENT");
    let bar = bar_of(&grid, header + 1);
    // #95: a whole cell draws `▇`, not the full-height `█`, so RECENT rows
    // stacked with no blank row between keep the same hairline gap #52
    // gave the meters.
    assert!(bar.iter().all(|(ch, _)| *ch != '█'), "{bar:?}");
    // #96: cached and out are fixed colours; new steps lower, to `▄`.
    assert!(
        bar.iter().any(|(ch, fg)| *ch == '▇' && *fg == C16::Blue),
        "cached: {bar:?}"
    );
    assert!(
        bar.iter()
            .any(|(ch, fg)| *ch == '▄' && *fg == C16::BrightCyan),
        "new, stepped lower: {bar:?}"
    );
    assert!(bar.iter().any(|(_, fg)| *fg == C16::Yellow), "out: {bar:?}");
    assert!(
        bar.iter()
            .any(|(ch, fg)| *ch == '\u{2581}' && *fg == C16::BrightBlack),
        "track"
    );
    // Halves: `▄` or `_` only; colours unchanged.
    let grid = draw(
        &bar_model(vec![row(Some(262_144))], ChartGlyphs::Halves),
        240,
        67,
    );
    let halves = bar_of(&grid, header + 1);
    assert!(
        halves.iter().all(|(ch, _)| matches!(ch, '▄' | '_')),
        "{halves:?}"
    );
    assert_eq!(halves.len(), bar.len());
    for ((_, a), (_, b)) in bar.iter().zip(&halves) {
        assert_eq!(a, b, "same colours in both glyph sets");
    }
}

#[test]
fn ninety_percent_of_the_context_is_a_warning_and_unknown_is_relative() {
    let row = |input: u64, n_ctx: Option<u64>| {
        let mut req = recent_model().requests[1].clone();
        (req.input_tok, req.cached_tok, req.output_tok) = (input, 0, 0);
        req.n_ctx = n_ctx;
        req
    };
    let warn_at = |req: Activity| {
        let grid = draw(&bar_model(vec![req], ChartGlyphs::Eighths), 240, 67);
        let header = row_with(&grid, "RECENT");
        bar_of(&grid, header + 1)
            .iter()
            // #96: out is now yellow, so the warning takes the bright step
            // of that colour to stay distinct from it.
            .any(|(ch, fg)| *ch == '!' && *fg == C16::BrightYellow)
    };
    assert!(warn_at(row(90_000, Some(100_000))));
    assert!(!warn_at(row(89_999, Some(100_000))));
    assert!(
        !warn_at(row(1_000_000, None)),
        "no context size, no warning"
    );
    // Relative: the largest unknown row fills the bar; `~` before each.
    let grid = draw(
        &bar_model(
            vec![row(40_000, None), row(10_000, None)],
            ChartGlyphs::Eighths,
        ),
        240,
        67,
    );
    let header = row_with(&grid, "RECENT");
    let big = bar_of(&grid, header + 1);
    // #96: these rows are all `new`, so a full bar steps to `▄`, never the
    // full-height `█` (#95).
    assert!(big.iter().all(|(ch, _)| *ch == '▄'), "{big:?}");
    let small = bar_of(&grid, header + 2);
    assert!(
        small.iter().any(|(_, fg)| *fg == C16::BrightBlack),
        "{small:?}"
    );
    for r in [header + 1, header + 2] {
        let line = row_string(&grid, r);
        let tilde = line.chars().position(|c| c == '~').expect("~");
        let start = line
            .chars()
            .position(|c| ('\u{2581}'..='\u{2587}').contains(&c))
            .unwrap();
        assert_eq!(tilde + 1, start, "{line}");
    }
    assert!(!whole(&grid).contains("~bar = vs largest row"));
}

#[allow(dead_code)]
fn legend_row(grid: &llama_watch::tty::grid::Grid) -> u16 {
    (0..grid.rows())
        .find(|r| row_string(grid, *r).contains("PROMPT") && row_string(grid, *r).contains(" = "))
        .expect("legend")
}

#[test]
fn header_key_sits_over_the_bar_with_fixed_colours() {
    let mut req = recent_model().requests[0].clone();
    (req.input_tok, req.cached_tok, req.output_tok) = (60_000, 40_000, 3_000);
    req.n_ctx = Some(262_144);
    for (glyphs, cols) in [
        (ChartGlyphs::Eighths, 240),
        (ChartGlyphs::Halves, 240),
        (ChartGlyphs::Eighths, 480),
    ] {
        let grid = draw(&bar_model(vec![req.clone()], glyphs), cols, 67);
        let header = row_with(&grid, "RECENT");
        let row = row_string(&grid, header);
        // The marker is `▇` with a llama-hack font, `▄` in halves mode.
        let mark = llama_watch::tty::layout::meter_glyph(glyphs);
        let start = row.chars().position(|c| c == mark).expect("segment key");
        let start_u16 = u16::try_from(start).unwrap();
        // Left-aligned over the bar: bar cell 0 is under the key's marker.
        let bar_row = row_string(&grid, header + 1);
        let bar_start = bar_row.chars().position(is_bar_char).unwrap();
        assert_eq!(start, bar_start, "{glyphs:?} {cols}: {row}");
        assert_eq!(
            grid.get(start_u16, header).unwrap().fg,
            C16::Blue,
            "{glyphs:?} {cols}: cached ▇"
        );
        assert!(row.contains("cached"), "{row}");
        assert!(row.contains("new"), "{row}");
        assert!(row.contains("out"), "{row}");
        // The header words keep their places left of the key.
        assert!(row.contains("DUR"), "{row}");
        assert!(col_of(&grid, header, "DUR") < start_u16, "{row}");
    }
}

#[test]
fn header_key_shortens_at_the_narrowest_screen() {
    // Below MIN_COLS the frame is the "too small" line; from it the bar is
    // at least BAR_FLOOR = 8 wide, narrower than the key, which then
    // shortens "cached" to "cach" (#96) and draws what fits.
    let mut req = recent_model().requests[0].clone();
    req.n_ctx = Some(262_144);
    let model = bar_model(vec![req], ChartGlyphs::Eighths);
    let min = llama_watch::tty::layout::MIN_COLS;
    for cols in min..min + 4 {
        let grid = draw(&model, cols, 49);
        let row = row_string(&grid, row_with(&grid, "RECENT"));
        assert!(row.contains('\u{2587}'), "{cols}: {row}");
    }
}

#[test]
fn legend_fits_160_columns_without_the_header_key() {
    let mut model = recent_model();
    model.chart_glyphs = ChartGlyphs::Eighths;
    let grid = draw(&model, 240, 67);
    let header = row_string(&grid, row_with(&grid, "RECENT"));
    assert!(
        header.contains("cached") && header.contains("new") && header.contains("out"),
        "{header}"
    );
    let all = whole(&grid);
    assert!(!all.contains("prompt processing speed"), "{all}");
    assert!(!all.contains("vs largest row"), "{all}");
    model.requests[0].gen_measured = true;
    model.requests[0].n_ctx = None;
    let grid = draw(&model, 160, 48);
    let header = row_string(&grid, row_with(&grid, "RECENT"));
    assert!(header.contains("GEN"), "{header}");
    assert!(header.trim_end().chars().count() <= 158, "{header}");
}

fn flight_row(processed: u64, decoded: u64, reset: Option<ResetReason>) -> Activity {
    let mut req = recent_model().requests[0].clone();
    req.id = 0;
    req.model = "Qwen 35B".to_string();
    req.time = "18:47:20".to_string();
    req.source = String::new();
    req.input_tok = 120_000;
    req.cached_tok = 0;
    req.output_tok = decoded;
    req.dur = "36.4s".to_string();
    req.live = false;
    req.n_ctx = Some(262_144);
    req.prompt_measured = true;
    req.gen_measured = decoded > 0;
    req.inflight = Some(InFlight {
        decoding: decoded > 0,
        processed,
        open: false,
        target: true,
        reset,
        ..InFlight::default()
    });
    req
}

/// #75 golden: a compaction prefill at 40 % (cached 0), a second slot
/// decoding, then the finished rows; the 10x18 font at 1080p.
fn inflight_model() -> TtyModel {
    let mut model = temps_model();
    model.chart_glyphs = ChartGlyphs::Eighths;
    let mut decoding = flight_row(31_500, 1_204, None);
    (decoding.input_tok, decoding.cached_tok) = (31_500, 29_800);
    decoding.time = "18:47:02".to_string();
    decoding.dur = "18.0s".to_string();
    let mut rows = vec![
        flight_row(48_000, 0, Some(ResetReason::Compacted)),
        decoding,
    ];
    rows.extend(model.requests.iter().cloned().map(|mut r| {
        r.live = false;
        r
    }));
    model.requests = rows;
    model
}

const INFLIGHT_GOLDENS: [(&str, u16, u16); 1] = [("recent-inflight-192.json", 192, 60)];

#[test]
fn inflight_golden_matches_character_and_colour() {
    for (name, _, _) in INFLIGHT_GOLDENS {
        assert_frame(name, &load(name), &inflight_model());
    }
}

#[test]
#[ignore = "run with --ignored to write the #75 in-flight golden"]
fn dump_inflight_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, cols, rows) in INFLIGHT_GOLDENS {
        let grid = draw(&inflight_model(), cols, rows);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write in-flight golden");
    }
}

#[test]
fn inflight_rows_lead_with_pp_or_gen_and_the_reset_letter() {
    let grid = draw(&inflight_model(), 192, 60);
    let header = row_with(&grid, "RECENT");
    let prefill = row_string(&grid, header + 1);
    let decode = row_string(&grid, header + 2);
    assert!(
        prefill.starts_with('>') && decode.starts_with('>'),
        "{prefill}\n{decode}"
    );
    assert!(prefill.trim_end().ends_with("pp c"), "{prefill}");
    assert!(decode.trim_end().ends_with("gen"), "{decode}");
    assert!(
        prefill.contains("120,000") && prefill.contains("36.4s"),
        "{prefill}"
    );
    // Prefill: new input so far (its head spinning, #96), then the pending
    // prompt as a cyan low line. This row has no cached or out tokens, so
    // every non-pending cell in the span is the new segment.
    let bar = bar_of(&grid, header + 1);
    let is_track_or_pending =
        |ch: char, fg: C16| ch == '\u{2581}' && matches!(fg, C16::BrightCyan | C16::BrightBlack);
    let pending = bar
        .iter()
        .filter(|(ch, fg)| *ch == '\u{2581}' && *fg == C16::BrightCyan)
        .count();
    let done = bar
        .iter()
        .filter(|(ch, fg)| !is_track_or_pending(*ch, *fg))
        .count();
    assert!(pending > done && done > 0, "40 % done: {bar:?}");
    assert!(
        bar.iter()
            .any(|(ch, fg)| *ch == '\u{2587}' && SPIN.contains(fg)),
        "the growing segment's head spins: {bar:?}"
    );
    // No finished row is marked `gen` while in-flight rows are shown.
    let third = row_string(&grid, header + 3);
    assert!(
        !third.starts_with('>') && !third.contains(" gen"),
        "{third}"
    );
    // At most half the rows are in flight.
    let mut model = inflight_model();
    let extra = model.requests[0].clone();
    model.requests.splice(0..0, std::iter::repeat_n(extra, 6));
    let grid = draw(&model, 192, 60);
    let header = row_with(&grid, "RECENT");
    let shown = recent_rows_shown(&grid);
    let flying = (header + 1..header + 1 + shown)
        .filter(|r| row_string(&grid, *r).starts_with('>'))
        .count();
    assert_eq!(
        flying,
        usize::from(shown / 2),
        "at most half the RECENT rows stay in flight"
    );
}

/// An in-flight llama.cpp prefill that omitted `prompt_cached` keeps the
/// done/pending split and the prefill cursor. On `integrate/kraken-dash`
/// that omitted count was drawn as cached 0, so the whole prompt is new;
/// the bars match, and there is no blue cached run. Only the CACHED cell
/// is `--`.
#[test]
fn an_inflight_prefill_without_a_cached_count_keeps_its_progress_and_cursor() {
    let mut parent = flight_row(48_000, 0, None);
    parent.cached_tok = 0;
    parent.cached_unknown = false;
    parent.prompt_tps = Some(10.0);
    parent.gen_tps = Some(10.0);
    parent.inflight.as_mut().expect("flight").beats = 2;
    let mut unknown = parent.clone();
    unknown.cached_unknown = true;

    let parent_grid = draw(&bar_model(vec![parent], ChartGlyphs::Eighths), 192, 60);
    let unknown_grid = draw(&bar_model(vec![unknown], ChartGlyphs::Eighths), 192, 60);
    let header = row_with(&parent_grid, "RECENT");
    let parent_bar = bar_of(&parent_grid, header + 1);
    let unknown_bar = bar_of(&unknown_grid, header + 1);
    assert_eq!(
        unknown_bar, parent_bar,
        "the bar matches the cached-0 drawing, cursor included"
    );
    assert!(
        unknown_bar.iter().any(|(_, fg)| *fg == SPIN[2]),
        "the prefill cursor is on the growing segment: {unknown_bar:?}"
    );
    assert!(
        unknown_bar
            .iter()
            .any(|(ch, fg)| *ch == '\u{2581}' && *fg == C16::BrightCyan),
        "the rest of the prompt stays pending: {unknown_bar:?}"
    );
    assert!(
        unknown_bar.iter().all(|(_, fg)| *fg != C16::Blue),
        "no cached split: {unknown_bar:?}"
    );
    let unknown_text = row_string(&unknown_grid, header + 1);
    let parent_text = row_string(&parent_grid, header + 1);
    assert!(unknown_text.contains("--"), "{unknown_text}");
    assert_ne!(unknown_text, parent_text);
}

/// #78: a llama.cpp prefill whose whole prompt `/slots` does not give:
/// IN is the held count with a `+`, the bar has no pending target.
#[test]
fn an_open_prefill_reads_as_a_lower_bound_with_no_target() {
    let mut model = inflight_model();
    let mut open = flight_row(48_000, 0, None);
    (open.input_tok, open.cached_tok) = (78_000, 30_000);
    if let Some(flight) = open.inflight.as_mut() {
        flight.open = true;
    }
    model.requests[0] = open;
    let grid = draw(&model, 192, 60);
    let header = row_with(&grid, "RECENT");
    let prefill = row_string(&grid, header + 1);
    assert!(prefill.contains("78,000+"), "{prefill}");
    assert!(prefill.trim_end().ends_with("pp"), "{prefill}");
    let bar = bar_of(&grid, header + 1);
    let pending = bar
        .iter()
        .filter(|(ch, fg)| *ch == '\u{2581}' && *fg == C16::BrightCyan)
        .count();
    assert_eq!(pending, 0, "no target track: {bar:?}");
    assert!(!bar.is_empty(), "{bar:?}");
}

// ---- #95: stacked bars never merge into one block -------------------------

/// A full-height block glyph: the plain full block and the half-width caps
/// used by the half-cell bars (`bar_glyph` in layout.rs). All three span the
/// whole cell height, so two of them in the same column on adjacent rows is
/// the hairline bug: the rows merge into one block with no gap between.
fn is_full_height_bar_glyph(ch: char) -> bool {
    matches!(ch, '█' | '▌' | '▐')
}

/// No column in `rows` has [`is_full_height_bar_glyph`] true on both a row
/// and the row right under it.
fn assert_no_hairline(
    grid: &llama_watch::tty::grid::Grid,
    rows: std::ops::Range<u16>,
    label: &str,
) {
    let mut row = rows.start;
    while row + 1 < rows.end {
        for col in 0..grid.cols() {
            let above = grid.get(col, row).unwrap().ch;
            let below = grid.get(col, row + 1).unwrap().ch;
            assert!(
                !(is_full_height_bar_glyph(above) && is_full_height_bar_glyph(below)),
                "{label}: rows {row}/{} col {col} both full height ({above:?} over {below:?})",
                row + 1
            );
        }
        row += 1;
    }
}

/// Fans maxed out, slots near-full, RECENT rows near their context limit:
/// every stacked bar #95 touches is lit edge to edge, which is exactly where
/// a hairline bug would show up as two adjacent full-height cells.
fn hairline_fixture(glyphs: ChartGlyphs, text_on: bool) -> TtyModel {
    let mut model = if text_on {
        fans_model()
    } else {
        let mut model = text_off_model(WatchState::Generating);
        model.fans = Some(ref_fans());
        model
    };
    model.chart_glyphs = glyphs;
    if let Some(panel) = model.fans.as_mut() {
        for fan in &mut panel.fans {
            fan.pwm = Some(255);
            fan.rpm = Some(2500);
        }
    }
    model.slots = (0..4)
        .map(|id| Slot {
            id,
            generating: true,
            done: 95,
            cached: 0,
            open: false,
            done_known: true,
            total: 100,
            decoded: 50_000,
            ctx_prompt: Some(250_000),
            n_ctx: Some(262_144),
            ctx_history: Vec::new(),
        })
        .collect();
    model.requests = (0..8)
        .map(|i| {
            activity(
                i == 0,
                4_900 - i,
                &format!("18:{:02}:{:02}", i, i),
                "192.0.2.83",
                "Qwen 35B",
                260_000,
                250_000,
                5_000,
                1200.0,
                50.0,
                "10.0s",
                false,
            )
        })
        .collect();
    model
}

#[test]
fn stacked_bars_never_merge_into_one_block() {
    for glyphs in [ChartGlyphs::Eighths, ChartGlyphs::Halves] {
        for text_on in [true, false] {
            for (cols, rows) in [(192u16, 60u16), (240u16, 67u16)] {
                let model = hairline_fixture(glyphs, text_on);
                let grid = draw(&model, cols, rows);
                let at = format!("{cols}x{rows} {glyphs:?} text_on={text_on}");

                // #165: the fan gauges, two to a row.
                let first = meter_row(&grid, "front1");
                assert_no_hairline(&grid, first..first + 2, &format!("{at} FANS"));

                let slot_rows: Vec<u16> = (0..model.slots.len())
                    .map(|id| row_with(&grid, &format!("s{id}")))
                    .collect();
                let slots_start = *slot_rows.iter().min().unwrap();
                let slots_end = *slot_rows.iter().max().unwrap() + 1;
                assert_no_hairline(&grid, slots_start..slots_end, &format!("{at} SLOTS"));

                let recent_header = row_with(&grid, "RECENT");
                let recent_end = req_rule_row(&grid);
                assert_no_hairline(
                    &grid,
                    recent_header + 1..recent_end,
                    &format!("{at} RECENT"),
                );
            }
        }
    }
}

/// #107, end to end: model text through the layout, the grid file's encoder
/// and decoder, and llama-view's renderer. The original characters reach the
/// viewer's terminal; tty11's cells for the same frame stay ASCII.
#[test]
fn the_grid_file_carries_the_original_characters_to_the_viewer() {
    use llama_core::gridfile::{GridCell, GridImage, decode, encode};
    use llama_core::palette::Palette;
    use llama_view::{ColorMode, Renderer, screen_of};

    let mut model = sample(WatchState::Generating);
    model.in_lines = vec!["caf\u{e9} \u{2192} \u{d7} \u{44f}\u{43b}\u{43a} \u{4e2d}".to_string()];
    model.out_lines = vec!["done \u{2014} \u{201c}ok\u{201d}".to_string()];
    let grid = layout(&model, 160, 48);

    // What tty11 gets: unchanged transliteration, console glyphs only.
    let console = grid_text(&grid);
    assert!(console.contains("cafe > x"), "{console}");
    assert!(!console.contains('\u{e9}') && !console.contains('\u{2192}'));

    let palette = Palette::Llama;
    let cells: Vec<GridCell> = grid
        .cells()
        .iter()
        .map(|cell| GridCell {
            ch: cell.rich,
            fg: palette.rgb(cell.fg.index()),
            bg: palette.rgb(cell.bg.index() & 7),
        })
        .collect();
    let image = GridImage {
        seq: 1,
        unix_ms: 1,
        cols: grid.cols(),
        rows: grid.rows(),
        cells,
    };
    let mut bytes = Vec::new();
    encode(&image, &mut bytes).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    let screen = screen_of(&decoded, false);
    let out = Renderer::with_colors(ColorMode::Truecolor, palette)
        .render(screen.cols, screen.rows, &screen.cells)
        .to_vec();
    let text = String::from_utf8_lossy(&out);
    assert!(text.contains("caf\u{e9} \u{2192} \u{d7} "), "{text}");
    assert!(text.contains("\u{44f}\u{43b}\u{43a}"), "{text}");
    assert!(text.contains("done \u{2014} \u{201c}ok\u{201d}"), "{text}");
    // CJK has no single-cell form: the console's placeholder is shown.
    assert!(!text.contains('\u{4e2d}'), "{text}");
    assert!(text.contains("\u{fffd}"), "{text}");
    assert!(text.contains("\u{1b}[38;2;"), "exact RGB");
}

fn grid_text(grid: &llama_watch::tty::Grid) -> String {
    grid.cells().iter().map(|cell| cell.ch).collect()
}

fn example_job() -> Job {
    Job {
        file: "example-trainer".to_owned(),
        name: "example-trainer".to_owned(),
        kind: Some("training".to_owned()),
        title: Some("Example-2B".to_owned()),
        state: JobState::Running,
        phase: Some(JobPhase {
            label: "2B".to_owned(),
            index: Some(2),
            of: Some(3),
        }),
        step: Some(921),
        total: Some(1075),
        elapsed_s: Some(15_381),
        eta_s: Some(2559),
        rate: Some(JobRate {
            value: 16.62,
            unit: Some("s/step".to_owned()),
        }),
        rates: vec![
            JobNumber {
                label: "rows/s".to_owned(),
                value: 0.96,
            },
            JobNumber {
                label: "tok/s".to_owned(),
                value: 1540.0,
            },
        ],
        values: vec![
            JobNumber {
                label: "loss".to_owned(),
                value: 0.2544,
            },
            JobNumber {
                label: "lr".to_owned(),
                value: 1.276e-5,
            },
            JobNumber {
                label: "epoch".to_owned(),
                value: 0.8465,
            },
            JobNumber {
                label: "grad".to_owned(),
                value: 0.348,
            },
        ],
        series: vec![
            JobSeries {
                label: "loss".to_owned(),
                points: vec![3.285, 0.846, 0.539, 0.254],
            },
            JobSeries {
                label: "s/step".to_owned(),
                points: vec![16.4, 16.7, 16.6],
            },
        ],
        phases: vec![
            JobPhaseRow {
                label: "synthetic test".to_owned(),
                mark: Some(PhaseMark::Done),
                elapsed_s: Some(356),
                result: Some("pass".to_owned()),
            },
            JobPhaseRow {
                label: "0.8B".to_owned(),
                mark: Some(PhaseMark::Done),
                elapsed_s: Some(10_186),
                result: Some("val 98.2 %".to_owned()),
            },
            JobPhaseRow {
                label: "2B".to_owned(),
                mark: Some(PhaseMark::Running),
                elapsed_s: Some(15_381),
                result: None,
            },
            JobPhaseRow {
                label: "4B".to_owned(),
                mark: Some(PhaseMark::Queued),
                elapsed_s: None,
                result: None,
            },
        ],
        mtime: UNIX_EPOCH + Duration::from_secs(1_700_000_000),
    }
}

fn job_model(job: Job, more: usize) -> TtyModel {
    let mut model = sample(WatchState::Ready);
    model.model_name = "--".to_owned();
    model.slots = Vec::new();
    model.requests = Vec::new();
    model.setup = None;
    model.job = Some(JobView { job, more });
    model.job_note = None;
    model
}

fn painted(model: &TtyModel, cols: u16, rows: u16) -> String {
    let grid = draw(model, cols, rows);
    (0..grid.rows())
        .map(|row| row_string(&grid, row))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn job_view_running_shows_title_progress_digits_phases() {
    let model = job_model(example_job(), 0);
    let text = painted(&model, 192, 60);
    assert!(text.contains("TRAINING"), "{text}");
    assert!(text.contains("example-trainer"), "{text}");
    assert!(text.contains("Example-2B"), "{text}");
    assert!(text.contains("2 of 3"), "{text}");
    assert!(text.contains("STEP"), "{text}");
    assert!(text.contains("921"), "{text}");
    assert!(text.contains("1,075"), "{text}");
    assert!(text.contains("left"), "{text}");
    assert!(text.contains("SECONDS / STEP"), "{text}");
    assert!(text.contains("LOSS"), "{text}");
    assert!(text.contains("VALUES"), "{text}");
    assert!(text.contains("PHASES"), "{text}");
    assert!(text.contains("JOB"), "{text}");
    assert!(text.contains("running"), "{text}");
    assert!(!text.contains("GENERATION"), "{text}");
    assert!(!text.contains("RECENT"), "{text}");
}

#[test]
fn job_view_minimal_shows_name_and_state() {
    let job = Job {
        file: "n".to_owned(),
        name: "only-name".to_owned(),
        kind: None,
        title: None,
        state: JobState::Running,
        phase: None,
        step: None,
        total: None,
        elapsed_s: None,
        eta_s: None,
        rate: None,
        rates: Vec::new(),
        values: Vec::new(),
        series: Vec::new(),
        phases: Vec::new(),
        mtime: UNIX_EPOCH,
    };
    let text = painted(&job_model(job, 0), 160, 26);
    assert!(text.contains("only-name"), "{text}");
    assert!(text.contains("JOB"), "{text}");
    assert!(text.contains("running"), "{text}");
    assert!(!text.contains("GENERATION"), "{text}");
    assert!(!text.contains("STEP"), "{text}");
}

#[test]
fn several_live_jobs_add_plus_n_after_the_name() {
    let text = painted(&job_model(example_job(), 2), 192, 60);
    assert!(text.contains("+2"), "{text}");
}

#[test]
fn a_serving_model_keeps_inference_and_a_job_note() {
    let mut model = sample(WatchState::Generating);
    model.job = None;
    model.job_note = Some("example-trainer 921/1,075 43 min left".to_owned());
    let text = painted(&model, 192, 60);
    assert!(text.contains("GEN tok/s"), "{text}");
    assert!(text.contains("RECENT"), "{text}");
    assert!(text.contains("example-trainer"), "{text}");
    assert!(text.contains("43 min left"), "{text}");
    assert!(!text.contains("PHASES"), "{text}");
}

#[test]
fn done_and_failed_notes_sit_in_the_header() {
    let mut model = sample(WatchState::Ready);
    model.job = None;
    model.job_note = Some("example-trainer done 4 h 16 min".to_owned());
    let text = painted(&model, 192, 60);
    assert!(text.contains("example-trainer done 4 h 16 min"), "{text}");
    assert!(text.contains("GEN tok/s"), "{text}");
    model.job_note = Some("example-trainer failed 12 min".to_owned());
    let text = painted(&model, 192, 60);
    assert!(text.contains("example-trainer failed 12 min"), "{text}");
}

#[test]
fn a_stale_job_does_not_replace_inference() {
    let mut model = sample(WatchState::Ready);
    model.job = None;
    model.job_note = None;
    let text = painted(&model, 192, 60);
    assert!(text.contains("GEN tok/s"), "{text}");
    assert!(!text.contains("PHASES"), "{text}");
    assert!(!text.contains("TRAINING"), "{text}");
}

fn claim_now() -> std::time::SystemTime {
    let unix = parse_rfc3339("2026-10-12T18:47:14Z").expect("rfc3339");
    UNIX_EPOCH + Duration::from_secs(u64::try_from(unix).expect("unix"))
}

fn header_claim_model(claim: HeaderClaim) -> TtyModel {
    let mut model = sample(WatchState::Generating);
    model.model_id = "qwen3.6-35b-a3b".to_owned();
    model.now = claim_now();
    model.tz_offset_s = 0;
    model.claims = vec![claim];
    model
}

fn hold_claim() -> HeaderClaim {
    HeaderClaim {
        model: "qwen3.6-35b-a3b".to_owned(),
        holder: "operator".to_owned(),
        in_use: true,
        expires_at: None,
        hold_until: Some(claim_now() + Duration::from_secs(2 * 3600 + 43 * 60)),
        hold_tz_offset_s: 0,
        hold_reason: "overnight eval, do not unload".to_owned(),
        conditional: false,
    }
}

fn header_line(model: &TtyModel, cols: u16) -> String {
    row_string(&draw(model, cols, 60), 0)
}

fn find_cells(chars: &[char], needle: &str) -> Option<usize> {
    let n: Vec<char> = needle.chars().collect();
    chars.windows(n.len()).position(|w| w == n.as_slice())
}

fn label_before_holder(left: &str, holder: &str, label: &str) -> bool {
    let Some(at) = left.find(holder) else {
        return false;
    };
    left[..at].ends_with(&format!("{label} "))
}

/// Reason text after the last separator following the holder (hold form:
/// `holder · until HH:MM · reason`).
fn hold_reason_drawn(left: &str, holder: &str) -> Option<String> {
    let at = left.find(holder)?;
    let tail = left[at..].trim_end();
    let mut parts = tail.split(" \u{00B7} ");
    let _holder = parts.next()?;
    let _status = parts.next()?;
    parts.next().map(str::to_owned)
}

#[test]
fn header_claim_four_forms_at_192() {
    let now = claim_now();
    let in_use = header_claim_model(HeaderClaim {
        model: "qwen3.6-35b-a3b".to_owned(),
        holder: "eval-runner".to_owned(),
        in_use: true,
        ..HeaderClaim::default()
    });
    let idle = header_claim_model(HeaderClaim {
        model: "qwen3.6-35b-a3b".to_owned(),
        holder: "eval-runner".to_owned(),
        expires_at: Some(now + Duration::from_secs(4 * 60 + 12)),
        ..HeaderClaim::default()
    });
    let short = header_claim_model(HeaderClaim {
        model: "qwen3.6-35b-a3b".to_owned(),
        holder: "chat-ui".to_owned(),
        expires_at: Some(now + Duration::from_secs(41)),
        ..HeaderClaim::default()
    });
    let hold = header_claim_model(hold_claim());
    let h_in_use = header_line(&in_use, 192);
    let h_idle = header_line(&idle, 192);
    let h_short = header_line(&short, 192);
    let h_hold = header_line(&hold, 192);
    assert!(
        h_in_use.contains("claim eval-runner") && h_in_use.contains("in use"),
        "{h_in_use}"
    );
    assert!(
        h_idle.contains("claim eval-runner") && h_idle.contains("4:12 left"),
        "{h_idle}"
    );
    assert!(
        h_short.contains("claim chat-ui") && h_short.contains("0:41 left"),
        "{h_short}"
    );
    // #167: `on <model>` takes room, so at 192 the reason is cut first.
    assert!(
        h_hold.contains("hold operator")
            && h_hold.contains("until 21:30")
            && h_hold.contains("overnight"),
        "{h_hold}"
    );
    let h_wide = header_line(&hold, 286);
    assert!(
        h_wide.contains(
            "hold operator on qwen3.6-35b-a3b · until 21:30 · overnight eval, do not unload"
        ),
        "{h_wide}"
    );
    let grid = draw(&in_use, 192, 60);
    let text = row_string(&grid, 0);
    let holder = char_at(&text, "eval-runner");
    assert_eq!(grid.get(holder as u16, 0).unwrap().fg, C16::BrightCyan);
    let label = char_at(&text, "claim");
    assert_eq!(grid.get(label as u16, 0).unwrap().fg, C16::BrightBlack);
    let use_at = char_at(&text, "in use");
    assert_eq!(grid.get(use_at as u16, 0).unwrap().fg, C16::BrightGreen);

    let grid = draw(&idle, 192, 60);
    let text = row_string(&grid, 0);
    let left = char_at(&text, "4:12 left");
    assert_eq!(grid.get(left as u16, 0).unwrap().fg, C16::BrightWhite);

    let grid = draw(&short, 192, 60);
    let text = row_string(&grid, 0);
    let left = char_at(&text, "0:41 left");
    assert_eq!(grid.get(left as u16, 0).unwrap().fg, C16::Yellow);

    let grid = draw(&hold, 192, 60);
    let text = row_string(&grid, 0);
    let hold_at = char_at(&text, "hold");
    assert_eq!(grid.get(hold_at as u16, 0).unwrap().fg, C16::Yellow);
    let until = char_at(&text, "until 21:30");
    // #167: a hold's end is the mock-ups' bright yellow.
    assert_eq!(grid.get(until as u16, 0).unwrap().fg, C16::BrightYellow);
    let reason = char_at(&text, "overnight");
    assert_eq!(grid.get(reason as u16, 0).unwrap().fg, C16::White);
}

#[test]
fn header_claim_hour_form_and_zero_at_expiry() {
    let now = claim_now();
    let hour = header_claim_model(HeaderClaim {
        model: "qwen3.6-35b-a3b".to_owned(),
        holder: "eval-runner".to_owned(),
        expires_at: Some(now + Duration::from_secs(3600 + 45 * 60)),
        ..HeaderClaim::default()
    });
    let minutes = header_claim_model(HeaderClaim {
        model: "qwen3.6-35b-a3b".to_owned(),
        holder: "eval-runner".to_owned(),
        expires_at: Some(now + Duration::from_secs(3600 + 5 * 60)),
        ..HeaderClaim::default()
    });
    let past = header_claim_model(HeaderClaim {
        model: "qwen3.6-35b-a3b".to_owned(),
        holder: "eval-runner".to_owned(),
        expires_at: Some(now - Duration::from_secs(1)),
        ..HeaderClaim::default()
    });
    // #167: `M:SS` up to 99:59, the hour form from 100 minutes.
    assert!(
        header_line(&hour, 192).contains("1 h 45 min left"),
        "{}",
        header_line(&hour, 192)
    );
    assert!(
        header_line(&minutes, 192).contains("65:00 left"),
        "{}",
        header_line(&minutes, 192)
    );
    assert!(
        header_line(&past, 192).contains(" 0:00 left"),
        "{}",
        header_line(&past, 192)
    );
}

#[test]
fn header_claim_hold_that_ends_tomorrow() {
    let mut claim = hold_claim();
    claim.hold_until = Some(claim_now() + Duration::from_secs(20 * 3600));
    let header = header_line(&header_claim_model(claim), 192);
    assert!(header.contains("until 10-13"), "{header}");
}

#[test]
fn header_claim_covers_160_192_213() {
    let model = header_claim_model(hold_claim());
    for cols in [160u16, 192, 213] {
        let header = header_line(&model, cols);
        assert!(
            header.contains("operator") && header.contains("until"),
            "{cols}: {header}"
        );
    }
}

/// #167: a claim on another model draws too, with `on <model>`. Before
/// #167 only the header model's claim was drawn.
#[test]
fn header_claim_for_another_model_draws_with_its_model() {
    let mut model = header_claim_model(hold_claim());
    model.model_id = "other".to_owned();
    let with = header_line(&model, 213);
    assert!(
        with.contains("hold operator on qwen3.6-35b-a3b \u{00B7} until 21:30"),
        "{with}"
    );
    // With nothing loaded too.
    model.model_id = String::new();
    assert!(header_line(&model, 213).contains("hold operator on qwen3.6-35b-a3b"));
}

#[test]
fn header_claim_does_not_overlap_the_clock() {
    let model = header_claim_model(hold_claim());
    for cols in [160u16, 192, 213] {
        let grid = draw(&model, cols, 60);
        let header = row_string(&grid, 0);
        let clock = if cols >= 200 {
            char_at(&header, "COOL")
        } else {
            char_at(&header, "2026-09-23")
        };
        let holder = char_at(&header, "operator");
        let mut end = holder;
        let mut gap = 0usize;
        for col in holder..clock {
            if grid.get(col as u16, 0).unwrap().ch == ' ' {
                gap += 1;
                if gap >= 3 {
                    break;
                }
            } else {
                gap = 0;
                end = col + 1;
            }
        }
        assert!(
            end + 2 <= clock,
            "{cols}: claim ends at {end}, clock at {clock}\n{header}"
        );
        for col in end..clock {
            assert_eq!(
                grid.get(col as u16, 0).unwrap().ch,
                ' ',
                "{cols} col {col} between claim and clock is not blank: {header}"
            );
        }
    }
}

/// Nothing claimed or reserved draws nothing (#128, #167): no claims, no
/// reservation (or one without a holder), and only displaced notes older
/// than ten minutes.
#[test]
fn no_claims_is_byte_identical_to_today() {
    let today = sample(WatchState::Generating);
    let mut empty = today.clone();
    empty.claims.clear();
    empty.model_id = "qwen3.6-35b-a3b".to_owned();
    let mut other = empty.clone();
    other.now = claim_now();
    other.gpu = Some(HeaderGpu::default());
    other.displaced = vec![HeaderDisplaced {
        model: "tiel-coder".to_owned(),
        to: "qwen-27b".to_owned(),
        by: "bakeoff".to_owned(),
        at: claim_now() - Duration::from_secs(10 * 60),
        tz_offset_s: 0,
    }];
    for cols in [160u16, 192, 213] {
        let a = draw(&today, cols, 60);
        let b = draw(&empty, cols, 60);
        let c = draw(&other, cols, 60);
        assert_eq!(console_bytes(&a), console_bytes(&b), "{cols} empty claims");
        assert_eq!(
            console_bytes(&a),
            console_bytes(&c),
            "{cols} empty reservation, old note"
        );
        let ga = layout_exact(&today, cols, 60);
        let gb = layout_exact(&empty, cols, 60);
        let gc = layout_exact(&other, cols, 60);
        assert_eq!(console_bytes(&ga), console_bytes(&gb), "{cols} grid empty");
        assert_eq!(console_bytes(&ga), console_bytes(&gc), "{cols} grid other");
    }
}

/// Shedding order, strictly: full; reason cut with the ellipsis (only
/// when at least two cells of reason text remain before it); no reason;
/// no label word; nothing. The reason is never drawn once the label is
/// dropped.
#[test]
fn header_claim_sheds_in_order_from_160_to_240() {
    let holder = "abcdefghijklmnopqrstuvwx";
    assert_eq!(holder.chars().count(), 24);
    let reason = "overnight eval, do not unload this model";
    assert_eq!(reason.chars().count(), 40);
    let now = claim_now();
    let forms: [(&str, HeaderClaim); 4] = [
        (
            "in_use",
            HeaderClaim {
                model: "qwen3.6-35b-a3b".to_owned(),
                holder: holder.to_owned(),
                in_use: true,
                ..HeaderClaim::default()
            },
        ),
        (
            "idle",
            HeaderClaim {
                model: "qwen3.6-35b-a3b".to_owned(),
                holder: holder.to_owned(),
                expires_at: Some(now + Duration::from_secs(4 * 60 + 12)),
                ..HeaderClaim::default()
            },
        ),
        (
            "short",
            HeaderClaim {
                model: "qwen3.6-35b-a3b".to_owned(),
                holder: holder.to_owned(),
                expires_at: Some(now + Duration::from_secs(41)),
                ..HeaderClaim::default()
            },
        ),
        (
            "hold",
            HeaderClaim {
                model: "qwen3.6-35b-a3b".to_owned(),
                holder: holder.to_owned(),
                hold_until: Some(now + Duration::from_secs(2 * 3600 + 43 * 60)),
                hold_reason: reason.to_owned(),
                ..HeaderClaim::default()
            },
        ),
    ];
    let extra_holds: [(&str, &str); 4] = [
        ("hold_1", "a"),
        ("hold_2", "ab"),
        ("hold_3", "abc"),
        ("hold_sp", "a b c d e f"),
    ];
    let mut all_forms: Vec<(&str, HeaderClaim, String)> = forms
        .into_iter()
        .map(|(n, c)| {
            let r = c.hold_reason.clone();
            (n, c, r)
        })
        .collect();
    for (name, reason) in extra_holds {
        all_forms.push((
            name,
            HeaderClaim {
                model: "qwen3.6-35b-a3b".to_owned(),
                holder: holder.to_owned(),
                hold_until: Some(now + Duration::from_secs(2 * 3600 + 43 * 60)),
                hold_reason: reason.to_owned(),
                ..HeaderClaim::default()
            },
            reason.to_owned(),
        ));
    }
    let names = ["Qwen 35B", &"N".repeat(40)];
    for (form_name, claim, whole_reason) in all_forms {
        let has_reason = !whole_reason.is_empty();
        let label = if claim.hold_until.is_some() {
            "hold"
        } else {
            "claim"
        };
        for name in names {
            let mut model = header_claim_model(claim.clone());
            model.model_name = name.to_owned();
            let mut blank = model.clone();
            blank.claims.clear();
            let mut rooms: Vec<(usize, u8)> = Vec::new();
            for cols in (160u16..=240).rev() {
                let grid = draw(&model, cols, 60);
                let blank_grid = draw(&blank, cols, 60);
                assert_eq!(grid.cols(), cols, "{form_name} {name} {cols} row width");
                assert_eq!(blank_grid.cols(), cols);
                let header: Vec<char> = (0..cols).map(|c| grid.get(c, 0).unwrap().ch).collect();
                let blank_h: Vec<char> = (0..cols)
                    .map(|c| blank_grid.get(c, 0).unwrap().ch)
                    .collect();
                let clock_word = if cols >= 200 { "COOL" } else { "2026-09-23" };
                let clock = find_cells(&header, clock_word).unwrap_or_else(|| {
                    panic!(
                        "{form_name} {name} {cols} no {clock_word}: {}",
                        header.iter().collect::<String>()
                    )
                });
                let blank_clock = find_cells(&blank_h, clock_word).unwrap_or_else(|| {
                    panic!(
                        "{form_name} {name} {cols} blank no {clock_word}: {}",
                        blank_h.iter().collect::<String>()
                    )
                });
                assert_eq!(clock, blank_clock, "{form_name} {name} {cols} clock column");
                assert!(
                    clock >= 2,
                    "{form_name} {name} {cols} clock has no gap: {}",
                    header.iter().collect::<String>()
                );
                assert_eq!(
                    &header[clock - 2..],
                    &blank_h[clock - 2..],
                    "{form_name} {name} {cols} right-hand items"
                );
                let left: String = header[..clock.saturating_sub(2)]
                    .iter()
                    .collect::<String>()
                    .trim_end()
                    .to_owned();
                let has_holder = left.contains(holder);
                let has_label = label_before_holder(&left, holder, label);
                let drawn_reason = hold_reason_drawn(&left, holder);
                assert!(
                    drawn_reason.is_none() || has_label,
                    "{form_name} {name} {cols}: reason without label: {left}"
                );
                if let Some(part) = drawn_reason.as_deref() {
                    let whole = part == whole_reason;
                    let cut_ok = part
                        .strip_suffix('\u{2026}')
                        .is_some_and(|head| head.chars().filter(|c| *c != ' ').count() >= 2);
                    assert!(
                        whole || cut_ok,
                        "{form_name} {name} {cols}: bad reason {part:?} in {left}"
                    );
                }
                if has_holder {
                    let last = left.chars().last().unwrap_or(' ');
                    assert_ne!(
                        last, '\u{00B7}',
                        "{form_name} {name} {cols} ends in a separator: {left}"
                    );
                }
                let has_full_reason =
                    has_reason && drawn_reason.as_deref() == Some(whole_reason.as_str());
                let has_reason_bit = drawn_reason.is_some();
                let step = if !has_holder {
                    4
                } else if !has_label {
                    3
                } else if has_reason && has_full_reason {
                    0
                } else if has_reason && has_reason_bit {
                    1
                } else {
                    2
                };
                // Order follows leftover after swap, not total width: at
                // 200 columns the COOL/CPU/GPU block takes room the clock
                // alone does not, so a narrower row can show a fuller claim.
                let leftover = find_cells(&header, &format!("swap {}", model.swap_line))
                    .map(|at| {
                        let after =
                            at + "swap".chars().count() + 1 + model.swap_line.chars().count() + 3;
                        clock.saturating_sub(2).saturating_sub(after)
                    })
                    .unwrap_or(0);
                rooms.push((leftover, step));
            }
            rooms.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            let mut seen = Vec::new();
            for &(_, step) in &rooms {
                if seen.last() != Some(&step) {
                    seen.push(step);
                }
            }
            let mut ordered = seen.clone();
            ordered.sort();
            assert_eq!(
                seen, ordered,
                "{form_name} {name}: forms {seen:?} are not in shedding order ({rooms:?})"
            );
        }
    }
}

#[test]
fn header_claim_is_not_drawn_when_swap_does_not_fit() {
    let claim = HeaderClaim {
        model: "qwen3.6-35b-a3b".to_owned(),
        holder: "eval-runner".to_owned(),
        in_use: true,
        ..HeaderClaim::default()
    };
    let mut saw_swap_dropped = 0;
    for name_len in 1usize..=48 {
        for cols in 100u16..=240 {
            let mut model = header_claim_model(claim.clone());
            model.model_name = "N".repeat(name_len);
            model.swap_line = "x".repeat(32);
            let header = header_line(&model, cols);
            if header.contains("swap ") {
                continue;
            }
            if cols >= 160 {
                saw_swap_dropped += 1;
            }
            assert!(
                !header.contains("eval-runner"),
                "claim without swap at {cols} name {name_len}: {header}"
            );
            assert!(
                !header.contains("claim eval-runner"),
                "{cols} {name_len}: {header}"
            );
        }
    }
    assert!(
        saw_swap_dropped > 0,
        "the sweep must include a frame where swap does not fit"
    );
}

/// #167: every claim draws, in order; before, only the first did.
#[test]
fn two_claims_on_the_same_model_both_draw_in_order() {
    let mut model = header_claim_model(HeaderClaim {
        model: "qwen3.6-35b-a3b".to_owned(),
        holder: "eval-runner".to_owned(),
        in_use: true,
        ..HeaderClaim::default()
    });
    model.claims.push(HeaderClaim {
        model: "qwen3.6-35b-a3b".to_owned(),
        holder: "chat-ui".to_owned(),
        in_use: false,
        ..HeaderClaim::default()
    });
    let header = header_line(&model, 213);
    let first = header.find("claim eval-runner").expect("first");
    let second = header.find("claim chat-ui").expect("second");
    assert!(first < second, "{header}");
}

/// The header model's claim comes first and matches on the full id (#128,
/// #167).
#[test]
fn header_matches_the_full_llama_swap_id() {
    let q4 = "unsloth-Qwen3.6-35B-A3B-Instruct-UD-Q4_K_XL";
    let q6 = "unsloth-Qwen3.6-35B-A3B-Instruct-UD-Q6_K_XL";
    let mut model = header_claim_model(HeaderClaim {
        model: q4.to_owned(),
        holder: "eval-runner".to_owned(),
        in_use: true,
        ..HeaderClaim::default()
    });
    model.model_id = q4.to_owned();
    model.claims.push(HeaderClaim {
        model: q6.to_owned(),
        holder: "chat-ui".to_owned(),
        in_use: true,
        ..HeaderClaim::default()
    });
    let before = |header: &str, a: &str, b: &str| {
        let first = header.find(a).unwrap_or_else(|| panic!("{a}: {header}"));
        header.find(b).is_none_or(|second| first < second)
    };
    let header = header_line(&model, 213);
    assert!(before(&header, "eval-runner", "chat-ui"), "{header}");
    model.model_id = q6.to_owned();
    let header = header_line(&model, 213);
    assert!(before(&header, "chat-ui", "eval-runner"), "{header}");
}

const CLAIMS_GOLDENS: [(&str, u16, u16); 1] = [("claims-192.json", 192, 60)];

#[test]
fn claims_golden_matches_character_and_colour() {
    for (name, cols, rows) in CLAIMS_GOLDENS {
        let fix = load(name);
        assert_eq!((fix.w, fix.h), (cols, rows), "{name}");
        assert_frame(name, &fix, &header_claim_model(hold_claim()));
    }
}

#[test]
#[ignore = "run with --ignored to write the #128 claims goldens"]
fn dump_claims_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, cols, rows) in CLAIMS_GOLDENS {
        let grid = draw(&header_claim_model(hold_claim()), cols, rows);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write claims golden");
    }
}

const JOB_GOLDENS: [(&str, u16, u16); 3] = [
    ("job-192.json", 192, 60),
    ("job-213.json", 213, 60),
    ("job-160x26.json", 160, 26),
];

#[test]
fn job_view_goldens_match_character_and_colour() {
    for (name, cols, rows) in JOB_GOLDENS {
        let fix = load(name);
        assert_eq!((fix.w, fix.h), (cols, rows), "{name}");
        assert_frame(name, &fix, &job_model(example_job(), 0));
    }
}

#[test]
fn job_view_header_shows_kind_and_job_name() {
    let model = job_model(example_job(), 0);
    let grid = draw(&model, 192, 60);
    let header = row_string(&grid, 0);
    assert!(header.contains("TRAINING"), "{header}");
    assert!(
        header.contains("job example-trainer"),
        "the word before the name is job\n{header}"
    );
    assert!(!header.contains("model"), "{header}");
    assert!(!header.contains("READY"), "{header}");
    assert!(!header.contains("Qwen"), "{header}");
}

#[test]
fn job_phase_marks_are_distinct_without_colour() {
    let text = painted(&job_model(example_job(), 0), 192, 60);
    assert!(text.contains('•'), "{text}");
    assert!(text.contains('>'), "{text}");
    assert!(text.contains('·'), "{text}");
    let mut failed = example_job();
    failed.phases[3].mark = Some(PhaseMark::Failed);
    let text = painted(&job_model(failed, 0), 192, 60);
    assert!(text.contains('x'), "{text}");
}

#[test]
fn text_off_does_not_overpaint_the_job_note() {
    let mut model = sample(WatchState::Generating);
    model.show_text = false;
    model.job = None;
    model.job_note = Some("example-trainer 921/1,075 43 min left".to_owned());
    let header = row_string(&draw(&model, 240, 60), 0);
    assert!(header.contains("text off"), "{header}");
    assert!(header.contains("example-trainer"), "{header}");
}

fn job_temps_model() -> TtyModel {
    let mut model = job_model(example_job(), 0);
    model.fans = Some(ref_fans());
    model.temps = Some(ref_temps());
    model
}

#[test]
fn job_view_keeps_fans_and_temps() {
    let grid = draw(&job_temps_model(), 192, 60);
    let all = painted(&job_temps_model(), 192, 60);
    assert!(all.contains("FANS"), "{all}");
    assert!(all.contains("TEMPS"), "{all}");
    assert!(all.contains("PHASES"), "{all}");
    assert!(all.contains("TRAINING"), "{all}");
    let fans = row_with(&grid, "FANS");
    let temps = row_with(&grid, "TEMPS");
    let health = grid.rows() - 3;
    assert!(fans < health, "FANS above the health rule");
    assert!(temps < health, "TEMPS above the health rule");
}

#[test]
fn job_charts_fill_the_room_and_pair_when_wide() {
    let model = job_model(example_job(), 0);
    let grid = draw(&model, 192, 60);
    let step = row_with(&grid, "s/step");
    let chart = row_string(&grid, step);
    assert!(
        chart.contains("loss"),
        "wide frame puts the first two series side by side\n{chart}\n{}",
        painted(&model, 192, 60)
    );
    let health = grid.rows() - 3;
    let chart_bottom = (step + 1..health)
        .rev()
        .find(|&row| {
            (0..grid.cols()).any(|col| {
                let ch = grid.get(col, row).unwrap().ch;
                ch != ' ' && ch != '|'
            })
        })
        .expect("chart ink");
    let height = chart_bottom.saturating_sub(step) + 1;
    assert!(
        height >= 5,
        "each chart is at least 5 rows, got {height} (rows {step}..={chart_bottom})"
    );
    assert!(
        chart_bottom + 1 >= health,
        "charts share every leftover row, height {height} rows {step}..={chart_bottom} health {health}"
    );
}

#[test]
fn four_job_series_use_two_rows_when_each_still_gets_five() {
    let mut job = example_job();
    job.series.push(JobSeries {
        label: "curve-a".to_owned(),
        points: vec![1.0, 2.0, 1.5],
    });
    job.series.push(JobSeries {
        label: "curve-b".to_owned(),
        points: vec![4.0, 3.0, 3.5],
    });
    let model = job_model(job, 0);
    let grid = draw(&model, 192, 60);
    let top = row_with(&grid, "s/step");
    let second = row_with(&grid, "curve-a");
    assert!(
        second > top,
        "three or four series use a second chart row\n{}",
        painted(&model, 192, 60)
    );
    let pair = row_string(&grid, second);
    assert!(
        pair.contains("curve-b"),
        "the second row pairs the last two series\n{pair}"
    );
    let health = grid.rows() - 3;
    assert!(
        second + 5 <= health,
        "each chart still gets 5 rows (second row {second}, health {health})"
    );
}

#[test]
fn a_poller_job_file_reaches_the_frame() {
    use llama_watch::config::Config;
    use llama_watch::jobs::JobState;
    use llama_watch::poller;

    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let dir = root.join(format!("tty-job-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("example-trainer.json"),
        br#"{"schema":1,"name":"example-trainer","kind":"training","title":"Example-2B","state":"running"}"#,
    )
    .unwrap();
    let cfg_path = root.join(format!("tty-job-{}.toml", std::process::id()));
    std::fs::write(
        &cfg_path,
        format!(
            r#"
[llama]
url = "http://127.0.0.1:1"
enabled = false

[jobs]
dir = "{}"
"#,
            dir.display()
        ),
    )
    .unwrap();
    let config = Config::load_validated(&cfg_path, 8).expect("valid");
    let (poller, rx) = poller::spawn(&config, llama_core::log::Stderr).expect("spawn");
    let start = Instant::now();
    let job = loop {
        match rx.try_recv() {
            Ok((_, detail)) if !detail.jobs.is_empty() => break detail.jobs[0].clone(),
            Ok(_) => {}
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                assert!(
                    start.elapsed() < Duration::from_secs(3),
                    "poller job timeout"
                );
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => panic!("poller stopped"),
        }
    };
    drop(poller);
    assert_eq!(job.state, JobState::Running);
    let text = painted(&job_model(job, 0), 192, 60);
    assert!(text.contains("TRAINING"), "{text}");
    assert!(text.contains("Example-2B"), "{text}");
    assert!(text.contains("example-trainer"), "{text}");
    assert!(!text.contains("GENERATION"), "{text}");
}

#[test]
#[ignore = "run with --ignored to write the job-view goldens"]
fn dump_job_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, cols, rows) in JOB_GOLDENS {
        let grid = draw(&job_model(example_job(), 0), cols, rows);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write job golden");
    }
}

/// A ready frame whose header says the card is busy outside llama-swap.
/// The process name is invented. The body stays the idle ready frame.
fn gpu_busy_model() -> TtyModel {
    let mut model = sample(WatchState::Ready);
    model.gpu_busy = true;
    model.model_name = "outside llama-swap".to_owned();
    model.model_detail = "example-bench 22.4 GB".to_owned();
    model.slots_line = "--".to_owned();
    model.swap_line = "--".to_owned();
    model
}

/// #139: the busy header at the default 12x24 console (160x45).
const GPU_BUSY_GOLDENS: [(&str, u16, u16); 1] = [("gpu-busy-160x45.json", 160, 45)];

#[test]
fn gpu_busy_header_golden_matches() {
    let model = gpu_busy_model();
    for (name, cols, rows) in GPU_BUSY_GOLDENS {
        let fix = load(name);
        assert_eq!((fix.w, fix.h), (cols, rows), "{name}");
        let grid = draw(&model, cols, rows);
        let header = row_string(&grid, 0);
        assert!(header.contains("GPU BUSY"), "{name}: {header}");
        assert!(header.contains("outside llama-swap"), "{name}: {header}");
        assert!(header.contains("example-bench 22.4 GB"), "{name}: {header}");
        assert!(header.contains("slots --"), "{name}: {header}");
        assert!(header.contains("swap --"), "{name}: {header}");
        assert!(!header.contains("READY"), "{name}: {header}");
        let word = col_of(&grid, 0, "GPU BUSY");
        assert_eq!(grid.get(word, 0).unwrap().fg, C16::Yellow);
        assert_eq!(grid.get(word, 0).unwrap().bg, C16::Black);
        let pip = grid.get(24, 0).unwrap();
        assert_eq!((pip.ch, pip.fg, pip.bg), ('█', C16::Yellow, C16::Black));
        let detail = col_of(&grid, 0, "example-bench");
        assert_eq!(grid.get(detail, 0).unwrap().fg, C16::BrightBlack);
        assert_frame(name, &fix, &model);
    }
}

#[test]
#[ignore = "run with --ignored to write the #139 busy-header golden"]
fn dump_gpu_busy_golden() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, cols, rows) in GPU_BUSY_GOLDENS {
        let grid = draw(&gpu_busy_model(), cols, rows);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write gpu busy golden");
    }
}

/// A ready frame whose header model is still loading. Names are invented.
fn loading_model() -> TtyModel {
    let mut model = sample(WatchState::Ready);
    model.model_name = "cedar-70b".to_owned();
    model.model_loading = true;
    model.slots_line = "--".to_owned();
    model.swap_line = "loading 1:42".to_owned();
    model.slots.clear();
    if let Some(setup) = model.setup.as_mut() {
        setup.id = "cedar-70b".to_owned();
        setup.name = "Cedar 70B".to_owned();
    }
    for row in &mut model.requests {
        row.model = if row.model == "Gemma 4B" {
            "birch-4b".to_owned()
        } else {
            "cedar-70b".to_owned()
        };
    }
    model
}

/// #135: the loading header at the default 12x24 console (160x45) and on
/// a wider grid (240x67).
const LOADING_GOLDENS: [(&str, u16, u16); 2] = [
    ("loading-160x45.json", 160, 45),
    ("loading-240.json", 240, 67),
];

#[test]
fn loading_header_goldens_match() {
    let model = loading_model();
    for (name, cols, rows) in LOADING_GOLDENS {
        let fix = load(name);
        assert_eq!((fix.w, fix.h), (cols, rows), "{name}");
        let header = row_string(&draw(&model, cols, rows), 0);
        assert!(header.contains("LOADING"), "{name}: {header}");
        assert!(header.contains("cedar-70b  llama.cpp"), "{name}: {header}");
        assert!(header.contains("slots --"), "{name}: {header}");
        assert!(header.contains("swap loading 1:42"), "{name}: {header}");
        assert!(!header.contains("READY"), "{name}: {header}");
        let word = col_of(&draw(&model, cols, rows), 0, "LOADING");
        let grid = draw(&model, cols, rows);
        assert_eq!(grid.get(word, 0).unwrap().fg, C16::Yellow);
        assert_eq!(grid.get(word, 0).unwrap().bg, C16::Black);
        assert_frame(name, &fix, &model);
    }
}

#[test]
#[ignore = "run with --ignored to write the #135 loading goldens"]
fn dump_loading_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, cols, rows) in LOADING_GOLDENS {
        let grid = draw(&loading_model(), cols, rows);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write loading golden");
    }
}

/// #163 step 1: the four console sizes and the 80x28 phone crop of 192x60.
#[test]
fn skeleton_frames_at_the_four_sizes_and_the_phone_crop() {
    let mut model = sample(WatchState::Generating);
    // Pane width at 192: cols 2..=95. "alpha beta" ends on that boundary.
    let pane = 94usize;
    let head = format!("{} beta", "a".repeat(pane - 5));
    assert_eq!(head.chars().count(), pane);
    model.in_lines = vec![format!("{head} gamma")];
    model.out_lines = vec!["Hello world from the reply".to_string()];
    model.chart = vec![llama_watch::tty::chart::ChartBucket {
        gen_tps: Some(54.2),
        prompt_tps: Some(0.0),
    }];
    let cases = [
        (213u16, 60u16, 7u16, true),
        (192, 60, 7, true),
        (160, 48, 7, true),
        (160, 42, 7, true),
        (160, 34, 7, false),
        (160, 26, 0, false),
    ];
    for (cols, rows, recent, text) in cases {
        let grid = draw(&model, cols, rows);
        let at = format!("{cols}x{rows}");
        assert_blank_edges(&grid);
        assert_eq!((grid.cols(), grid.rows()), (cols, rows), "{at}");
        // GEN "54.2" starts on column 2. PROMPT "1212" fits at pixel width 2
        // inside its 30 columns: digit 1 is `..#`, so columns 32-35 are
        // blank and the stem is column 36.
        assert_eq!(grid.get(2, 3).unwrap().ch, '█', "{at}: GEN digit");
        assert_eq!(grid.get(32, 3).unwrap().ch, ' ', "{at}: PROMPT digit 1");
        assert_eq!(grid.get(36, 3).unwrap().ch, '█', "{at}: PROMPT digit");
        let all = whole(&grid);
        assert!(all.contains("ACT ") && all.contains("LOAD"), "{at}");
        assert!(!all.contains("prompt processing (prefill)"), "{at}");
        // Ceilings sit in the left margin, right-aligned to column 7.
        let start = timeline_rule(&grid) + 1;
        assert_eq!(
            (
                grid.get(5, start).unwrap().ch,
                grid.get(6, start).unwrap().ch,
                grid.get(7, start).unwrap().ch
            ),
            ('2', '5', '0'),
            "{at}: gen ceiling"
        );
        let last = start + chart_height_of(&grid) - 1;
        assert_eq!(
            (
                grid.get(4, last).unwrap().ch,
                grid.get(5, last).unwrap().ch,
                grid.get(6, last).unwrap().ch,
                grid.get(7, last).unwrap().ch,
            ),
            ('1', '5', '0', '0'),
            "{at}: prompt ceiling"
        );
        assert!(
            (start..=last).any(|row| {
                grid.get(5, row).unwrap().ch == 'n'
                    && grid.get(6, row).unwrap().ch == 'o'
                    && grid.get(7, row).unwrap().ch == 'w'
            }),
            "{at}: now is not right-aligned in columns 5-7"
        );
        assert_eq!(recent_rows_shown(&grid), recent, "{at}");
        if recent > 0 {
            let header = row_with(&grid, "RECENT");
            let after = header + 1 + recent;
            assert!(
                is_rule_row(&grid, after),
                "{at}: unused row between RECENT and the next rule"
            );
        }
        if text {
            let mid = cols / 2;
            let title = (0..rows)
                .find(|row| grid.get(mid, *row).unwrap().ch == '|')
                .unwrap_or_else(|| panic!("{at}: no IN/OUT divider"));
            assert_eq!(grid.get(mid, title).unwrap().ch, '|', "{at}");
            assert!(title > timeline_rule(&grid), "{at}");
            let left: String = (2..mid)
                .map(|col| grid.get(col, title).unwrap().ch)
                .collect();
            let right: String = (mid + 1..cols - 1)
                .map(|col| grid.get(col, title).unwrap().ch)
                .collect();
            assert!(left.contains("IN") && !left.contains("OUT"), "{at}: {left}");
            assert!(
                right.contains("OUT") && !right.contains("IN"),
                "{at}: {right}"
            );
        }
        // #177: SETUP beside the digits from 192 columns, where the glance
        // was; under 192 that spot is empty.
        let beside: String = (64..cols / 2 - 2)
            .map(|col| grid.get(col, 2).unwrap().ch)
            .collect();
        if cols >= 192 {
            assert!(beside.starts_with("SETUP  qwen3.6"), "{at}: {beside}");
        } else {
            assert!(beside.trim().is_empty(), "{at}: {beside}");
        }
        for col in 0..cols {
            assert_eq!(
                grid.get(col, rows - 1).unwrap().ch,
                ' ',
                "{at} corner {col}"
            );
        }
    }
    let grid = draw(&model, 192, 60);
    let mid = 96u16;
    let panes: Vec<String> = (0..60)
        .map(|row| {
            (2..mid)
                .map(|col| grid.get(col, row).unwrap().ch)
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect();
    let head = format!("{} beta", "a".repeat(pane - 5));
    let beta = panes
        .iter()
        .position(|line| line == &head)
        .expect("exact-fit line was split");
    assert_eq!(
        panes[beta + 1].trim(),
        "gamma",
        "boundary space was not consumed"
    );
}

/// #163: short screens keep the rate digits, the half-height rate bars and
/// the load-matrix meters from reading as one solid mass.
#[test]
fn short_rate_and_meter_bars_do_not_bridge() {
    for (cols, rows) in [(213u16, 60u16), (192, 60)] {
        let mut model = sample(WatchState::Generating);
        // Eighths would previously draw meters as `▇`, nearly a full cell.
        model.chart_glyphs = ChartGlyphs::Eighths;
        model.gen_tps = Some(12_400.0);
        model.prompt_tps = Some(8_888.0);
        let grid = draw(&model, cols, rows);
        let at = format!("{cols}x{rows}");
        let bar = rate_bar_row(&grid, &at);
        assert_rate_gaps(&grid, bar, &at);
        assert_k_suffix(&grid, bar, &at);
        assert_digit_separation(&grid, bar, &at);
        assert_meters_do_not_touch(&grid, &at);
        assert_slot_bar_is_half(&grid, &at);
    }

    let tall = draw(&sample(WatchState::Generating), 320, 90);
    let bot = (0..tall.rows())
        .find(|row| (2..28).any(|col| tall.get(col, *row).unwrap().ch == '▓'))
        .expect("tall rate bar");
    assert!(
        (2..28).any(|col| tall.get(col, bot - 1).unwrap().ch == '█'),
        "tall rate bar stays full height"
    );
    assert!(
        (2..28).any(|col| tall.get(col, bot - 2).unwrap().ch == '█'),
        "tall digits stay against the rate bar"
    );
}

fn span_text(grid: &llama_watch::tty::grid::Grid, row: u16, x: u16, width: u16) -> String {
    (x..x + width)
        .map(|col| grid.get(col, row).unwrap().ch)
        .collect()
}

fn span_blank(grid: &llama_watch::tty::grid::Grid, row: u16, x: u16, width: u16) -> bool {
    (x..x + width).all(|col| grid.get(col, row).unwrap().ch == ' ')
}

fn rate_bar_row(grid: &llama_watch::tty::grid::Grid, at: &str) -> u16 {
    let mut found = Vec::new();
    for row in 1..grid.rows() - 1 {
        let gen_bar = span_text(grid, row, 2, 26);
        let prompt_bar = span_text(grid, row, 32, 30);
        if gen_bar.contains('▄')
            && prompt_bar.contains('▄')
            && !gen_bar.contains('█')
            && !prompt_bar.contains('█')
            && span_blank(grid, row - 1, 2, 26)
            && span_blank(grid, row - 1, 32, 30)
            && span_blank(grid, row + 1, 2, 26)
            && span_blank(grid, row + 1, 32, 30)
        {
            found.push(row);
        }
    }
    assert_eq!(found.len(), 1, "{at}: rate bar rows {found:?}");
    found[0]
}

fn assert_rate_gaps(grid: &llama_watch::tty::grid::Grid, bar: u16, at: &str) {
    for (x, width) in [(2u16, 26u16), (32, 30)] {
        let text = span_text(grid, bar, x, width);
        assert!(text.contains('▄'), "{at} {x}: {text}");
        assert!(!text.contains('█'), "{at} {x}: {text}");
        assert!(span_blank(grid, bar - 1, x, width), "{at} above {x}");
        assert!(span_blank(grid, bar + 1, x, width), "{at} below {x}");
    }
    let scale = span_text(grid, bar + 2, 2, 26);
    assert!(
        scale.chars().any(|ch| ch.is_ascii_digit()),
        "{at}: scale moved onto the blank row: {scale:?}"
    );
}

fn assert_k_suffix(grid: &llama_watch::tty::grid::Grid, bar: u16, at: &str) {
    let base = bar - 2;
    let mut cols = Vec::new();
    for col in 2..28 {
        if grid.get(col, base).unwrap().ch == 'k' {
            cols.push(col);
        }
    }
    assert_eq!(cols.len(), 1, "{at}: k columns {cols:?}");
    let col = cols[0];
    assert_eq!(grid.get(col - 1, base).unwrap().ch, ' ', "{at}");
    for above in 1..5 {
        assert_eq!(
            grid.get(col, base - above).unwrap().ch,
            ' ',
            "{at}: k is a block glyph"
        );
    }
}

fn ink_gaps(text: &str) -> Vec<usize> {
    let mut gaps = Vec::new();
    let mut seen_ink = false;
    let mut in_ink = false;
    let mut gap = 0usize;
    for ch in text.chars() {
        if ch == '█' {
            if seen_ink && !in_ink {
                gaps.push(gap);
            }
            seen_ink = true;
            in_ink = true;
            gap = 0;
        } else if seen_ink {
            in_ink = false;
            gap += 1;
        }
    }
    gaps
}

fn assert_digit_separation(grid: &llama_watch::tty::grid::Grid, bar: u16, at: &str) {
    let base = bar - 2;
    let prompt = span_text(grid, base, 32, 30);
    let gaps = ink_gaps(&prompt);
    assert!(
        gaps.len() >= 3 && gaps.iter().all(|gap| *gap >= 2),
        "{at}: prompt digit gaps {gaps:?} in {prompt:?}"
    );
    let gen_last = (2..32)
        .rev()
        .find(|col| grid.get(*col, base).unwrap().ch == '█')
        .expect("gen ink");
    let prompt_first = (32..64)
        .find(|col| grid.get(*col, base).unwrap().ch == '█')
        .expect("prompt ink");
    assert!(
        prompt_first - gen_last > 3,
        "{at}: fewer than three blank columns between gen ending {gen_last} and prompt starting {prompt_first}"
    );
}

fn right_meter_bar(grid: &llama_watch::tty::grid::Grid, label: &str) -> (u16, Vec<(u16, char)>) {
    let mut hits = Vec::new();
    for row in 0..grid.rows() {
        let text = row_string(grid, row);
        let Some(byte) = text.find(label) else {
            continue;
        };
        let col = u16::try_from(text[..byte].chars().count()).unwrap();
        if col <= grid.cols() / 2 || !text[byte + label.len()..].starts_with(' ') {
            continue;
        }
        let start = col + u16::try_from(label.chars().count()).unwrap();
        let mut run = Vec::new();
        for c in start..grid.cols() - 1 {
            let ch = grid.get(c, row).unwrap().ch;
            if matches!(ch, '▄' | '█' | '▇' | '▓' | '▐' | '▌' | '·' | SEG) {
                run.push((c, llama_core::glyph::plain(ch)));
            } else if !run.is_empty() {
                break;
            }
        }
        if run.iter().any(|(_, ch)| matches!(*ch, '▄' | '█' | '▇')) {
            hits.push((row, run));
        }
    }
    assert_eq!(hits.len(), 1, "{label} bars: {}", hits.len());
    hits.pop().unwrap()
}

fn assert_meters_do_not_touch(grid: &llama_watch::tty::grid::Grid, at: &str) {
    // #165: ACT and, under it, GPU in the first column of gauges.
    let (activity, activity_bar) = right_meter_bar(grid, "ACT");
    let (load, load_bar) = right_meter_bar(grid, "GPU");
    assert_eq!(load, activity + 1, "{at}: meters are not stacked");
    for (name, bar) in [("ACT", &activity_bar), ("GPU", &load_bar)] {
        assert!(
            bar.iter().any(|(_, ch)| *ch == '▄'),
            "{at} {name} has no half block: {bar:?}"
        );
        assert!(
            bar.iter()
                .all(|(_, ch)| !matches!(*ch, '█' | '▇' | '▓' | '▐' | '▌')),
            "{at} {name} still full height: {bar:?}"
        );
    }
    let load_at: std::collections::HashMap<u16, char> = load_bar.iter().copied().collect();
    let shared: Vec<(char, char)> = activity_bar
        .iter()
        .filter_map(|(col, ch)| load_at.get(col).map(|below| (*ch, *below)))
        .filter(|(above, below)| *above == '▄' && *below == '▄')
        .collect();
    assert!(
        !shared.is_empty(),
        "{at}: adjacent meters do not share a lit column"
    );
    assert!(
        shared
            .iter()
            .all(|(above, below)| *above == '▄' && *below == '▄'),
        "{at}: stacked meters touch: {shared:?}"
    );
}

fn assert_slot_bar_is_half(grid: &llama_watch::tty::grid::Grid, at: &str) {
    let row = row_with(grid, "s0 gen");
    let col = col_of(grid, row, "s0 gen") + u16::try_from("s0 gen".chars().count()).unwrap();
    let mut cells = Vec::new();
    for c in col..grid.cols() / 2 {
        let ch = grid.get(c, row).unwrap().ch;
        if matches!(ch, '▄' | '█' | '▇' | '·' | '▐' | '▌') {
            cells.push(ch);
        } else if !cells.is_empty() {
            break;
        }
    }
    assert!(cells.contains(&'▄'), "{at}: slot bar {cells:?}");
    assert!(!cells.contains(&'█'), "{at}: slot bar {cells:?}");
    // The ctx meter on the same row stacks with the next slot's too.
    let text = row_string(grid, row);
    assert!(
        !text.contains('▇') && !text.contains('█'),
        "{at}: a full-height bar on the slot row: {text}"
    );
}

/// The blank rows around the short rate bar are paid for by IN/OUT. A
/// screen too short to spare them keeps every SLOTS and SETUP row it had.
#[test]
fn short_screens_keep_slots_and_setup() {
    for cols in [160u16, 213] {
        for rows in 26u16..=89 {
            let grid = draw(&sample(WatchState::Generating), cols, rows);
            let text: String = (0..grid.rows())
                .map(|row| row_string(&grid, row) + "\n")
                .collect();
            let at = format!("{cols}x{rows}");
            assert!(text.contains("SLOTS"), "{at}: no SLOTS header\n{text}");
            assert!(text.contains("s0 gen"), "{at}: no s0 row\n{text}");
            if rows == 26 || rows >= 45 {
                assert!(text.contains("SETUP"), "{at}: no SETUP\n{text}");
            }
            if rows >= 45 {
                assert!(text.contains("engine"), "{at}: no engine line\n{text}");
            }
        }
    }
}

/// Two `--` placeholders on a tall screen are the same size, although
/// PROMPT has four more columns than GEN.
#[test]
fn rate_placeholders_match_in_size() {
    let grid = draw(&sample(WatchState::AiDown), 213, 135);
    // The digit rows only: row 2 is the label, the tall digits are 10 rows.
    let ink = |from: u16, to: u16| {
        (3..13u16)
            .flat_map(|row| (from..to).map(move |col| (col, row)))
            .filter(|(col, row)| grid.get(*col, *row).unwrap().ch == '█')
            .count()
    };
    let gen_ink = ink(2, 30);
    let prompt_ink = ink(32, 64);
    assert!(gen_ink > 0, "GEN placeholder drawn");
    assert_eq!(gen_ink, prompt_ink, "GEN and PROMPT `--` differ in size");
}

fn with_host_rates(zero: bool) -> TtyModel {
    let mut model = sample(WatchState::Generating);
    if zero {
        model.disk_read_bps = Some(0.0);
        model.disk_write_bps = Some(0.0);
        model.net_rx_bps = Some(0.0);
        model.net_tx_bps = Some(0.0);
        model.io_psi = Some(0.0);
        model.mem_psi = Some(0.0);
        model.swap_used_gb = Some(0.0);
        model.swap_total_gb = Some(16.0);
    } else {
        model.disk_read_bps = Some(1.8e9);
        model.disk_write_bps = Some(12e6);
        model.net_rx_bps = Some(100e6);
        model.net_tx_bps = Some(18e6);
        model.io_psi = Some(1.1);
        model.mem_psi = Some(0.0);
        model.swap_used_gb = Some(0.4);
        model.swap_total_gb = Some(16.0);
    }
    model.disk_ceiling_bps = 7e9;
    model.net_ceiling_bps = Some(125_000_000.0);
    model
}

/// #163 step 2: 12 MB/s of writes is a lit cell on a 7 GB/s ceiling, and a
/// real zero is an empty track rather than a dash.
#[test]
fn host_meters_show_non_zero_and_zero_byte_rates() {
    let grid = draw(&with_host_rates(false), 192, 60);
    let read = row_string(&grid, meter_row(&grid, "DISK R"));
    let write = row_string(&grid, meter_row(&grid, "DISK W"));
    let net = row_string(&grid, meter_row(&grid, "NET"));
    let psi = row_string(&grid, meter_row(&grid, "IO PSI"));
    let swap = row_string(&grid, meter_row(&grid, "SWAP"));
    assert!(read.contains("1.8 G/s"), "{read}");
    assert!(write.contains("12 M/s"), "{write}");
    assert!(net.contains("118 M/s"), "{net}");
    assert!(psi.contains("1.1 %"), "{psi}");
    assert!(swap.contains("0.4/16.0 G"), "{swap}");
    let write_row = meter_row(&grid, "DISK W");
    let col = meter_bar_col(&grid, write_row, "DISK W");
    assert_eq!(
        grid.get(col, write_row).unwrap().ch,
        '▄',
        "12 MB/s must light the write bar"
    );
    // Below the three-decade floor the start cell still lights.
    let mut tiny = with_host_rates(false);
    tiny.disk_read_bps = Some(100.0);
    let grid = draw(&tiny, 192, 60);
    let row = meter_row(&grid, "DISK R");
    let col = meter_bar_col(&grid, row, "DISK R");
    assert_eq!(
        grid.get(col, row).unwrap().ch,
        '▄',
        "a non-zero trickle lights"
    );

    let grid = draw(&with_host_rates(true), 192, 60);
    let write_row = meter_row(&grid, "DISK W");
    let write = row_string(&grid, write_row);
    assert!(write.contains("0 B/s"), "{write}");
    let col = meter_bar_col(&grid, write_row, "DISK W");
    assert_eq!(
        grid.get(col, write_row).unwrap().ch,
        '·',
        "zero is an empty track"
    );
    let net = row_string(&grid, meter_row(&grid, "NET"));
    assert!(net.contains("0 B/s"), "{net}");
    let psi = row_string(&grid, meter_row(&grid, "IO PSI"));
    assert!(psi.contains("0.0 %"), "{psi}");
}

/// #165: the work gauges always show. A host reading not taken yet is a
/// dash and an empty track, in its usual place.
#[test]
fn absent_host_readings_keep_their_gauges_as_dashes() {
    let grid = draw(&sample(WatchState::Generating), 192, 60);
    for label in ["SWAP", "DISK R", "DISK W", "NET", "IO PSI"] {
        let row = meter_row(&grid, label);
        let (_, col) = label_hit(&grid, label);
        let text: String = (col..col + 19)
            .map(|c| grid.get(c, row).unwrap().ch)
            .collect();
        assert!(text.trim_end().ends_with("--"), "{label}: {text}");
        let bar = meter_bar_col(&grid, row, label);
        assert_eq!(bar, col + 19, "{label}");
        assert_eq!(
            grid.get(bar, row).unwrap().ch,
            '·',
            "{label}: an empty track"
        );
    }
}

#[test]
fn memory_pressure_turns_the_swap_label_yellow() {
    let mut model = with_host_rates(false);
    model.mem_psi = Some(2.5);
    let grid = draw(&model, 192, 60);
    let (row, col) = label_hit(&grid, "SWAP");
    assert_eq!(grid.get(col, row).unwrap().fg, C16::Yellow);
    model.mem_psi = Some(0.0);
    let grid = draw(&model, 192, 60);
    let (row, col) = label_hit(&grid, "SWAP");
    assert_eq!(grid.get(col, row).unwrap().fg, C16::White);
}

/// #163, #177: the 80x28 phone crop of a 160-column frame holds the two
/// rate digits.
#[test]
fn phone_crop_holds_the_two_rate_digits() {
    for rows in [26u16, 48] {
        let grid = draw(&with_host_rates(false), 160, rows);
        let lit = |cols: std::ops::Range<u16>| {
            (3..8).any(|row| {
                cols.clone()
                    .any(|col| grid.get(col, row).unwrap().ch == '█')
            })
        };
        assert!(lit(2..28), "160x{rows}: GEN digits");
        assert!(lit(32..62), "160x{rows}: PROMPT digits");
        // LOAD keeps its gauge on the right.
        let load = meter_row(&grid, "LOAD");
        assert!(load < timeline_rule(&grid), "160x{rows}: LOAD at {load}");
        assert!(label_hit(&grid, "LOAD").1 >= 80, "160x{rows}");
    }
}

#[test]
fn health_line_shows_disk_net_and_psi() {
    let mut model = sample(WatchState::Generating);
    for (name, status) in [
        ("disk", HealthStatus::Ok),
        ("net", HealthStatus::Ok),
        ("psi", HealthStatus::Ok),
    ] {
        model.health.push(HealthSeg {
            name: name.to_owned(),
            status,
            note: String::new(),
            word_cells: 0,
            note_cells: 0,
        });
    }
    let row = row_string(&draw(&model, 240, 67), 65);
    assert!(row.contains("disk ok"), "{row}");
    assert!(row.contains("net ok"), "{row}");
    assert!(row.contains("psi ok"), "{row}");

    model.health.last_mut().unwrap().status = HealthStatus::Down;
    let row = row_string(&draw(&model, 240, 67), 65);
    assert!(row.contains("psi DOWN"), "{row}");
}

/// The new host meters must not push the frame past the 160x26 floor. They
/// appear only once a reading exists, so the step-1 floor test does not
/// cover this: draw with real disk, net, PSI and swap values present.
#[test]
fn host_meters_fit_the_160x26_floor() {
    for (cols, rows) in [(160, 26), (160, 48), (192, 60), (213, 60)] {
        let grid = draw(&with_host_rates(false), cols, rows);
        let text: String = (0..rows)
            .map(|row| {
                (0..cols)
                    .map(|col| grid.get(col, row).unwrap().ch)
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !text.contains("too small"),
            "{cols}x{rows} with host rates must draw, got:\n{text}"
        );
    }
}

/// The tail of the health line with a long snapshot number and uptime.
fn long_tail(model: &mut TtyModel) -> String {
    model.snapshot = Some(123_456_789);
    model.snapshot_age = "12.3s".to_string();
    model.errors = 12_345;
    model.uptime = "123d04h12m".to_string();
    // #167: the age keeps six cells; longer numbers take what they need.
    "snapshot #123456789  12.3s   errors 12345   up 123d04h12m".to_string()
}

fn all_down() -> Vec<HealthSeg> {
    [
        "llama-swap",
        "/running",
        "/slots",
        "metrics",
        "activity",
        "nvml",
        "hwmon",
        "proc",
        "disk",
        "net",
        "psi",
    ]
    .into_iter()
    .map(|name| HealthSeg {
        name: name.to_owned(),
        status: HealthStatus::Down,
        note: if name == "llama-swap" {
            "refused".to_owned()
        } else {
            String::new()
        },
        word_cells: 0,
        note_cells: 0,
    })
    .collect()
}

fn with_host_health(disk: HealthStatus, net: HealthStatus, psi: HealthStatus) -> TtyModel {
    let mut model = sample(WatchState::Generating);
    for (name, status) in [("disk", disk), ("net", net), ("psi", psi)] {
        model.health.push(HealthSeg {
            name: name.to_owned(),
            status,
            note: String::new(),
            word_cells: 0,
            note_cells: 0,
        });
    }
    model
}

/// #163 M2: at every width from 160 to 480 the words stop at least two
/// columns before the tail, and every DOWN source stays readable.
#[test]
fn health_line_never_overprints_its_tail_and_keeps_every_down() {
    let mut down = sample(WatchState::AiDown);
    down.health = all_down();
    let models = [
        down,
        with_host_health(HealthStatus::Ok, HealthStatus::Ok, HealthStatus::Ok),
        with_host_health(HealthStatus::Down, HealthStatus::Ok, HealthStatus::Down),
    ];
    for mut model in models {
        let tail = long_tail(&mut model);
        for cols in 160..=480u16 {
            let grid = draw(&model, cols, 48);
            let row = row_string(&grid, 46);
            let at = format!("{cols}: {row}");
            assert!(row.ends_with(&tail), "{at}");
            let words = &row[..row.len() - tail.len()];
            assert!(words.ends_with("  "), "{at}");
            let tokens: Vec<&str> = words.split_whitespace().collect();
            for seg in model
                .health
                .iter()
                .filter(|s| s.status == HealthStatus::Down)
            {
                assert!(
                    tokens.contains(&seg.name.as_str()),
                    "{} hidden at {at}",
                    seg.name
                );
            }
            assert!(
                tokens.contains(&"DOWN")
                    || !model.health.iter().any(|s| s.status == HealthStatus::Down),
                "{at}"
            );
        }
    }
}

#[test]
fn health_line_sheds_healthy_words_before_down_ones_at_160_192_and_213() {
    let mut model = with_host_health(HealthStatus::Down, HealthStatus::Ok, HealthStatus::Down);
    let row = row_string(&draw(&model, 160, 26), 24);
    assert!(
        row.contains("disk DOWN") && row.contains("psi DOWN"),
        "{row}"
    );
    assert!(
        row.starts_with("  llama-swap ok 3ms"),
        "healthy words lead: {row}"
    );

    model.health = all_down();
    let tail = long_tail(&mut model);
    for (cols, rows) in [(160u16, 26u16), (192, 60), (213, 60)] {
        let row = row_string(&draw(&model, cols, rows), rows - 2);
        let at = format!("{cols}x{rows}: {row}");
        assert!(row.ends_with(&tail), "{at}");
        if cols == 213 {
            // Room for every word once the note is dropped.
            assert!(row.contains("disk DOWN   net DOWN   psi DOWN"), "{at}");
            assert!(!row.contains("refused"), "{at}");
        } else {
            // One DOWN badge, then the names.
            assert!(
                row.contains(
                    "DOWN llama-swap /running /slots metrics activity nvml hwmon proc disk net psi"
                ),
                "{at}"
            );
        }
    }
}

/// #163 M3: at 90 rows and more the meters are two rows tall with a gap,
/// and NET and IO PSI still fit above TEMPS: SWAP sits beside POWER.
#[test]
fn host_meters_show_on_tall_screens() {
    for (cols, rows) in [(213u16, 90u16), (240, 100), (213, 60), (160, 26)] {
        let grid = draw(&with_host_rates(false), cols, rows);
        let at = format!("{cols}x{rows}");
        for label in ["SWAP", "DISK R", "DISK W", "NET", "IO PSI"] {
            let row = meter_row(&grid, label);
            assert!(row < timeline_rule(&grid), "{at}: {label}");
        }
        assert_eq!(meter_row(&grid, "SWAP"), meter_row(&grid, "POWER"), "{at}");
        assert!(meter_row(&grid, "DISK R") < meter_row(&grid, "NET"), "{at}");
    }
}

fn disk_read_cells(bps: f64) -> (usize, usize) {
    let mut model = with_host_rates(false);
    model.disk_read_bps = Some(bps);
    let grid = draw(&model, 192, 60);
    let row = meter_row(&grid, "DISK R");
    let col = meter_bar_col(&grid, row, "DISK R");
    let cells: Vec<char> = (col..grid.cols())
        .map(|c| grid.get(c, row).unwrap().ch)
        .take_while(|ch| matches!(ch, '▄' | '·'))
        .collect();
    (cells.iter().filter(|ch| **ch == '▄').count(), cells.len())
}

/// #163 M11: the disk bar spans three decades: the ceiling fills it, a
/// tenth of it fills two thirds, a hundredth one third.
#[test]
fn byte_rate_bars_span_three_decades() {
    for (bps, frac) in [(7e9, 1.0), (7e8, 2.0 / 3.0), (7e7, 1.0 / 3.0)] {
        let (lit, width) = disk_read_cells(bps);
        let want = frac * width as f64;
        assert!(
            (lit as f64 - want).abs() <= 1.0,
            "{bps} B/s lit {lit} of {width}, want {want:.1}"
        );
    }
}

/// The unit is picked after rounding: nothing reads `1000 K/s`.
#[test]
fn byte_rates_roll_over_to_the_next_unit() {
    for (bps, text) in [
        (999.6, "1.0 K/s"),
        (9_960.0, "10 K/s"),
        (999_600.0, "1.0 M/s"),
        (999.6e6, "1.0 G/s"),
        (999.4e6, "999 M/s"),
        (12.0, "12 B/s"),
    ] {
        let mut model = with_host_rates(false);
        model.disk_read_bps = Some(bps);
        let grid = draw(&model, 192, 60);
        // The gauge's value field: label and value, 18 cells.
        let (row, col) = label_hit(&grid, "DISK R");
        let field = cells_between(&grid, row, col, col + 18);
        assert!(field.ends_with(&format!(" {text}")), "{bps}: {field}");
    }
}

// ---- #165: the uniform gauge grid --------------------------------------------

/// Everything the right-hand grid shows: host rates, the reference TEMPS
/// rows and fans, and a hold on the header model (#128).
fn gauge_model() -> TtyModel {
    let claimed = header_claim_model(hold_claim());
    let mut model = with_host_rates(false);
    model.model_id = claimed.model_id;
    model.now = claimed.now;
    model.tz_offset_s = claimed.tz_offset_s;
    model.claims = claimed.claims;
    model.temps = Some(ref_temps());
    model.fans = Some(ref_fans());
    model
}

/// The right half of rows `0..rows` as text; a lit gauge cell (`▄` or the
/// segment) is `▄`.
fn right_half(grid: &llama_watch::tty::grid::Grid, rows: u16) -> String {
    let from = grid.cols() / 2 + 1;
    (0..rows)
        .map(|row| {
            let line: String = (from..grid.cols() - 1)
                .map(|col| llama_core::glyph::plain(grid.get(col, row).unwrap().ch))
                .collect();
            format!("{row:>2} {}\n", line.trim_end())
        })
        .collect()
}

#[test]
#[ignore = "prints the #165 right half at 213x60 and 160x26 for review"]
fn print_gauge_frames() {
    let model = gauge_model();
    for (cols, rows) in [
        (213u16, 60u16),
        (192, 60),
        (160, 26),
        (160, 48),
        (200, 28),
        (240, 100),
    ] {
        let grid = draw(&model, cols, rows);
        let rule = timeline_rule(&grid);
        println!("== {cols}x{rows}\n{}", right_half(&grid, rule + 1));
    }
}

/// Every console size the dashboard is tested at (#163), with the bar
/// length the right half gives it.
const GAUGE_SIZES: [(u16, u16, u16); 9] = [
    (160, 26, 16),
    (160, 45, 16),
    (160, 48, 16),
    (192, 60, 24),
    (200, 28, 26),
    (213, 60, 30),
    (213, 90, 30),
    (240, 67, 36),
    (240, 100, 36),
];

const WORK_GAUGES: [[&str; 2]; 6] = [
    ["ACT", "LOAD"],
    ["GPU", "CPU"],
    ["VRAM", "MEM"],
    ["POWER", "SWAP"],
    ["DISK R", "DISK W"],
    ["NET", "IO PSI"],
];

/// Where a gauge's label starts: the topmost row at or under `from` whose
/// gauge column `col` (first or second) begins with `label` and a blank.
fn gauge_row(grid: &llama_watch::tty::grid::Grid, label: &str, col: u16, from: u16) -> Option<u16> {
    (from..timeline_rule(grid)).find(|row| {
        let text: String = (col..col + 8)
            .map(|c| grid.get(c, *row).unwrap().ch)
            .collect();
        text.starts_with(label) && text[label.len()..].starts_with(' ')
    })
}

/// The cells of one bar: lit (`▄` or the segment), then dots.
fn gauge_cells(grid: &llama_watch::tty::grid::Grid, row: u16, at: u16) -> Vec<char> {
    (at..grid.cols())
        .map(|c| grid.get(c, row).unwrap().ch)
        .take_while(|ch| matches!(*ch, '▄' | '·' | SEG))
        .collect()
}

/// #165: at every size the work gauges are the first six rows of the right
/// half, two to a row, every bar the same length and starting at the same
/// column in its column, nothing past it but a blank.
#[test]
fn gauge_bars_line_up_at_every_size() {
    let model = gauge_model();
    for glyphs in [ChartGlyphs::Eighths, ChartGlyphs::Halves] {
        let mut model = model.clone();
        model.chart_glyphs = glyphs;
        for (cols, rows, bar) in GAUGE_SIZES {
            let grid = draw(&model, cols, rows);
            let at = format!("{cols}x{rows} {glyphs:?}");
            let x = [cols / 2 + 1, cols / 2 + 1 + 19 + bar + 4];
            for (i, pair) in WORK_GAUGES.iter().enumerate() {
                let row = 2 + u16::try_from(i).unwrap();
                for (side, label) in pair.iter().enumerate() {
                    assert_eq!(
                        gauge_row(&grid, label, x[side], 2),
                        Some(row),
                        "{at}: {label}"
                    );
                    let cells = gauge_cells(&grid, row, x[side] + 19);
                    assert_eq!(cells.len(), usize::from(bar), "{at}: {label} {cells:?}");
                    let end = x[side] + 19 + bar;
                    assert_eq!(grid.get(end, row).unwrap().ch, ' ', "{at}: {label} ends");
                    assert_eq!(
                        grid.get(x[side] + 18, row).unwrap().ch,
                        ' ',
                        "{at}: {label} gap"
                    );
                }
            }
            // The groups below keep the same columns and length.
            for (label, side) in [("coolant", 0), ("NVMe0", 1)] {
                if let Some(row) = gauge_row(&grid, label, x[side], 9) {
                    let cells = gauge_cells(&grid, row, x[side] + 19);
                    assert_eq!(cells.len(), usize::from(bar), "{at}: {label}");
                }
            }
            assert_blank_edges(&grid);
        }
    }
}

/// #165: work, a blank row, temperatures, a blank row, fans. Short of rows
/// the fans go first, then the coolest temperatures; the work gauges stay.
#[test]
fn gauge_groups_shed_fans_then_the_coolest_temperatures() {
    let model = gauge_model();
    // 213x60: all three groups, 14 rows.
    let grid = draw(&model, 213, 60);
    assert_eq!(meter_row(&grid, "coolant"), 10);
    assert_eq!(meter_row(&grid, "Wi-Fi"), 12);
    assert_eq!(meter_row(&grid, "front1"), 14);
    assert_eq!(meter_row(&grid, "top"), 15);
    for blank in [8u16, 13] {
        let text: String = (107..211).map(|c| grid.get(c, blank).unwrap().ch).collect();
        assert!(text.trim().is_empty(), "row {blank}: {text}");
    }
    assert_eq!(timeline_rule(&grid), 16);
    // 160x26: eleven rows; the fans go, every temperature stays.
    let grid = draw(&model, 160, 26);
    let all = whole(&grid);
    assert!(!all.contains("rpm"), "{all}");
    for name in ["coolant", "NVMe0", "NVMe1", "board", "NIC", "Wi-Fi"] {
        assert!(meter_row(&grid, name) < timeline_rule(&grid), "{name}");
    }
    // 200x28: ten rows; the fans, then the two coolest (Wi-Fi 39 C and
    // NIC 51 C against 80 C), go. The hotter ones keep their order.
    let grid = draw(&model, 200, 28);
    let all = whole(&grid);
    assert!(
        !all.contains("rpm") && !all.contains("Wi-Fi") && !all.contains("NIC "),
        "{all}"
    );
    assert_eq!(meter_row(&grid, "coolant"), 10);
    assert_eq!(meter_row(&grid, "board"), 11);
    assert_eq!(timeline_rule(&grid), 12);
    // The coolest by the bar, not the reading: of seven rows one goes, and
    // it is NIC (51 C of 80), not the coolant (40 C of 45).
    let mut cool = model.clone();
    cool.temps
        .as_mut()
        .unwrap()
        .groups
        .retain(|g| g.name != "Wi-Fi");
    let grid = draw(&cool, 200, 28);
    let all = whole(&grid);
    assert!(all.contains("coolant") && !all.contains("NIC "), "{all}");
    // Fans go from the end, a row at a time: eight fans need four rows,
    // three fit, the last two go.
    let mut fans = model.clone();
    fans.temps = None;
    let panel = fans.fans.as_mut().unwrap();
    for n in 7..=10u32 {
        panel
            .fans
            .push(fan(n, &format!("fan{n}"), Some(900), Some(100), Some(5)));
    }
    let grid = draw(&fans, 200, 28);
    assert_eq!(meter_row(&grid, "front1"), 9);
    assert_eq!(meter_row(&grid, "fan8"), 11);
    let all = whole(&grid);
    assert!(!all.contains("fan9") && !all.contains("fan10"), "{all}");
}

/// #128, #165: the claim and the hold in the header still draw beside the
/// gauge grid at 213x60, 192x60 and 160x26.
#[test]
fn a_claim_still_draws_with_the_gauge_grid() {
    let claimed = |in_use: bool| {
        let mut model = gauge_model();
        model.claims = vec![HeaderClaim {
            model: "qwen3.6-35b-a3b".to_owned(),
            holder: "eval-runner".to_owned(),
            in_use,
            expires_at: (!in_use).then(|| claim_now() + Duration::from_secs(4 * 60 + 5)),
            ..HeaderClaim::default()
        }];
        model
    };
    for (cols, rows) in [(213u16, 60u16), (192, 60), (160, 26)] {
        let at = format!("{cols}x{rows}");
        let hold = draw(&gauge_model(), cols, rows);
        let header = row_string(&hold, 0);
        // #167: `on <model>` while there is room, and the countdown's
        // five cells.
        let drawn = |header: &str, head: &str, status: &str| {
            [
                format!("{head} on qwen3.6-35b-a3b · {status}"),
                format!("{head} · {status}"),
            ]
            .iter()
            .any(|form| header.contains(form.as_str()))
        };
        assert!(
            drawn(&header, "hold operator", "until 21:30"),
            "{at}: {header}"
        );
        let in_use = draw(&claimed(true), cols, rows);
        let header = row_string(&in_use, 0);
        assert!(
            drawn(&header, "claim eval-runner", "in use"),
            "{at}: {header}"
        );
        let idle = draw(&claimed(false), cols, rows);
        let header = row_string(&idle, 0);
        assert!(
            drawn(&header, "claim eval-runner", " 4:05 left"),
            "{at}: {header}"
        );
        for grid in [&hold, &in_use, &idle] {
            assert_eq!(meter_row(grid, "IO PSI"), 7, "{at}");
            assert!(is_rule_row(grid, 1), "{at}");
        }
    }
}

/// #163's 80x28 phone crop of the left edge holds the headline; no gauge
/// reaches into it, at any size.
#[test]
fn no_gauge_reaches_the_phone_crop() {
    // Segments mark the gauge cells apart from the chart's blocks.
    let mut model = gauge_model();
    model.bar_segments = true;
    for (cols, rows, _) in GAUGE_SIZES {
        let grid = draw(&model, cols, rows);
        for row in 2..timeline_rule(&grid) {
            for col in 0..80u16.min(cols / 2 + 1) {
                let cell = grid.get(col, row).unwrap();
                assert_ne!(cell.rich, SEG, "{cols}x{rows}: a gauge at {col},{row}");
            }
        }
        let first = (0..cols)
            .find(|c| (2..8).any(|row| grid.get(*c, row).unwrap().rich == SEG))
            .expect("a lit gauge");
        assert!(first >= 80, "{cols}x{rows}: {first}");
    }
}

fn one_temp(c: i32, warn: i32, crit: i32) -> TtyModel {
    let mut model = gauge_model();
    model.fans = None;
    model.temps = Some(TempPanel {
        groups: vec![TempGroup {
            name: "probe".to_string(),
            items: vec![temp_marked(
                "",
                c,
                Level::Normal,
                Some((warn * 10, crit * 10)),
            )],
        }],
    });
    model
}

/// #165: a temperature gauge is empty at 20 C and full at the sensor's
/// warning point. Past it the bar stays full and its colour runs on into
/// the hot tail: red at the warning, hot orange at the critical point.
#[test]
fn temperature_gauge_runs_from_20_c_to_the_warning_point() {
    let lit = |c: i32, warn: i32, crit: i32| {
        let grid = draw(&one_temp(c, warn, crit), 213, 60);
        let row = meter_row(&grid, "probe");
        let cells = gauge_cells(&grid, row, meter_bar_col(&grid, row, "probe"));
        assert_eq!(cells.len(), 30);
        cells.iter().filter(|ch| **ch != '·').count()
    };
    assert_eq!(lit(20, 80, 90), 0);
    assert_eq!(lit(50, 80, 90), 15);
    assert_eq!(lit(80, 80, 90), 30);
    assert_eq!(lit(85, 80, 90), 30);
    assert_eq!(lit(33, 45, 50), 16, "coolant: 13 of 25 degrees");
    assert_eq!(lit(45, 45, 50), 30);
    let last = |c: i32| {
        let grid = layout_exact(&one_temp(c, 80, 90), 213, 60);
        let image =
            llama_watch::grid_out::image_of(&grid, llama_core::palette::Palette::Llama, 1, 0);
        let console = draw(&one_temp(c, 80, 90), 213, 60);
        let row = meter_row(&console, "probe");
        let end = meter_bar_col(&console, row, "probe") + 29;
        let cell = image.cells[usize::from(row) * 213 + usize::from(end)];
        (cell.fg, console.get(end, row).unwrap().fg)
    };
    let (at_warn, warn_fg) = last(80);
    let (at_crit, crit_fg) = last(90);
    assert_eq!(
        at_warn,
        llama_core::color::act_color(100.0),
        "red at the warning"
    );
    assert_eq!(
        at_crit,
        llama_core::color::act_color(120.0),
        "hot orange at critical"
    );
    assert_eq!(warn_fg, C16::BrightRed);
    assert_eq!(crit_fg, C16::BrightWhite, "the console's hot top step");
}

// ---- #165 review: sizes, shedding, fallbacks -----------------------------------

/// From a phone-sized 80x28 to 512x200 nothing panics (the shedding guard's
/// debug assertion included), no segment leaves the right half, and every
/// console that draws the dashboard shows all twelve work gauges.
#[test]
fn gauges_draw_at_every_size_and_stay_in_the_right_half() {
    let work: Vec<&str> = WORK_GAUGES.iter().flatten().copied().collect();
    let mut fails = Vec::new();
    for (cols, rows) in [
        (80u16, 28u16),
        (100, 30),
        (120, 30),
        (150, 40),
        (159, 26),
        (160, 26),
        (160, 24),
        (160, 20),
        (160, 16),
        (160, 12),
        (160, 8),
        (161, 26),
        (161, 60),
        (213, 10),
        (213, 20),
        (213, 26),
        (300, 80),
        (300, 30),
        (512, 200),
    ] {
        let mut model = gauge_model();
        model.bar_segments = true;
        let drawn =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| draw(&model, cols, rows)));
        let Ok(grid) = drawn else {
            fails.push(format!("{cols}x{rows}: panic"));
            continue;
        };
        let (gc, gr) = (grid.cols(), grid.rows());
        let segs: Vec<u16> = (0..gr)
            .flat_map(|r| (0..gc).map(move |c| (c, r)))
            .filter(|&(c, r)| {
                let cell = grid.get(c, r).unwrap();
                cell.rich == SEG || cell.ch == SEG
            })
            .map(|(c, _)| c)
            .collect();
        if gc < 160 {
            continue;
        }
        if segs.iter().any(|c| *c < gc / 2 + 1) {
            fails.push(format!("{cols}x{rows}: a segment left of the right half"));
        }
        if segs.iter().any(|c| *c > gc - 3) {
            fails.push(format!("{cols}x{rows}: a segment past the right margin"));
        }
        // Under 26 rows the console shows its too-small screen (#163).
        if gr >= 26 {
            let rh = right_half(&grid, 9);
            let missing: Vec<&str> = work.iter().copied().filter(|l| !rh.contains(l)).collect();
            if !missing.is_empty() {
                fails.push(format!("{cols}x{rows}: missing {missing:?}"));
            }
        }
    }
    assert!(fails.is_empty(), "{fails:#?}");
}

/// A fan with rpm but no duty shows its rpm over an empty track, not red.
#[test]
fn a_fan_with_rpm_but_no_duty_is_an_empty_track() {
    let mut model = gauge_model();
    model.temps = None;
    model.fans.as_mut().unwrap().fans[0] = fan(2, "front1", Some(1500), None, Some(5));
    let grid = draw(&model, 213, 60);
    let row = meter_row(&grid, "front1");
    let text = row_string(&grid, row);
    assert!(text.contains("front1    1500 rpm"), "{text}");
    let (_, col) = label_hit(&grid, "front1");
    assert_ne!(grid.get(col, row).unwrap().fg, C16::Red);
    let cells = gauge_cells(&grid, row, meter_bar_col(&grid, row, "front1"));
    assert_eq!(cells.len(), 30);
    assert!(cells.iter().all(|c| *c == '·'), "{cells:?}");
}

/// A reading without thresholds runs on the TEMPS default, full at 80 C.
#[test]
fn a_temperature_without_thresholds_is_full_at_80_c() {
    let lit = |c: i32| {
        let mut model = one_temp(c, 45, 50);
        model.temps.as_mut().unwrap().groups[0].items[0].thresholds = None;
        let grid = draw(&model, 213, 60);
        let row = meter_row(&grid, "probe");
        let cells = gauge_cells(&grid, row, meter_bar_col(&grid, row, "probe"));
        cells.iter().filter(|c| **c != '·').count()
    };
    assert_eq!(lit(50), 15, "30 of 60 degrees");
    assert_eq!(lit(80), 30);
}

/// Each percent rounds to whole cells: 30 cells at 213, 1 % is 0.3 cells.
#[test]
fn a_gauge_fill_rounds_to_whole_cells() {
    for (pct, lit) in [
        (1.0, 1usize),
        (2.0, 1),
        (50.0, 15),
        (98.0, 29),
        (99.0, 30),
        (100.0, 30),
        (130.0, 30),
    ] {
        let mut model = gauge_model();
        model.gpu_pct = Some(pct);
        let grid = draw(&model, 213, 60);
        let cells = gauge_cells(&grid, 3, 107 + 19);
        assert_eq!(cells.len(), 30);
        assert_eq!(
            cells.iter().filter(|c| **c != '·').count(),
            lit,
            "{pct}: {cells:?}"
        );
    }
}

fn temp_rows(names: &[(&str, i32)]) -> TempPanel {
    TempPanel {
        groups: names
            .iter()
            .map(|(name, c)| TempGroup {
                name: (*name).to_string(),
                items: vec![temp_marked("", *c, Level::Normal, Some((800, 900)))],
            })
            .collect(),
    }
}

/// Of two equally cool temperatures the later goes; the rest keep order.
#[test]
fn a_shed_tie_drops_the_later_and_keeps_the_order() {
    let mut model = gauge_model();
    model.fans = None;
    model.temps = Some(temp_rows(&[
        ("t1", 60),
        ("t2", 30),
        ("t3", 70),
        ("t4", 30),
        ("t5", 65),
        ("t6", 75),
        ("t7", 66),
    ]));
    let grid = draw(&model, 200, 28);
    let rh = right_half(&grid, timeline_rule(&grid));
    assert!(rh.contains("t2 ") && !rh.contains("t4 "), "{rh}");
}

/// The coolest temperature goes wherever it sits, here first: the hotter
/// ones after it stay, in order, and the fans go whole, not clipped.
#[test]
fn the_coolest_temperature_goes_even_when_it_comes_first() {
    let mut model = gauge_model();
    model.temps = Some(temp_rows(&[
        ("cold", 25),
        ("w1", 60),
        ("w2", 70),
        ("w3", 65),
        ("w4", 75),
        ("w5", 66),
        ("hot", 79),
    ]));
    let grid = draw(&model, 200, 28);
    // 200x28: ten rows, so work, a blank and three rows of temperatures.
    let x = [101u16, 101 + 19 + 26 + 4];
    for side in x {
        assert_eq!(gauge_row(&grid, "cold", side, 2), None, "{}", whole(&grid));
    }
    for (label, side, row) in [
        ("w1", 0, 9),
        ("w2", 1, 9),
        ("w3", 0, 10),
        ("w4", 1, 10),
        ("w5", 0, 11),
        ("hot", 1, 11),
    ] {
        assert_eq!(gauge_row(&grid, label, x[side], 2), Some(row), "{label}");
    }
    assert!(!whole(&grid).contains("rpm"));
    assert_eq!(timeline_rule(&grid), 12, "no second blank row, no fans");
}

/// While llama-watch starts, its own level gauges draw `--` over an empty
/// track even when readings are there; SWAP's reading stays.
#[test]
fn starting_blanks_the_level_gauges_even_with_readings() {
    let mut model = gauge_model();
    model.state = WatchState::Starting;
    let grid = draw(&model, 213, 60);
    let x = [107u16, 107 + 19 + 30 + 4];
    for (label, side, row) in [
        ("ACT", 0, 2),
        ("LOAD", 1, 2),
        ("GPU", 0, 3),
        ("CPU", 1, 3),
        ("VRAM", 0, 4),
        ("MEM", 1, 4),
        ("POWER", 0, 5),
    ] {
        assert_eq!(gauge_row(&grid, label, x[side], 2), Some(row), "{label}");
        let field: String = (x[side]..x[side] + 18)
            .map(|c| grid.get(c, row).unwrap().ch)
            .collect();
        assert!(field.trim_end().ends_with(" --"), "{label}: {field}");
        let cells = gauge_cells(&grid, row, x[side] + 19);
        assert_eq!(cells.len(), 30, "{label}");
        assert!(cells.iter().all(|c| *c == '·'), "{label}: {cells:?}");
    }
    let swap: String = (x[1]..x[1] + 18)
        .map(|c| grid.get(c, 5).unwrap().ch)
        .collect();
    assert!(swap.ends_with("0.4/16.0 G"), "{swap}");
}

/// #165: the tty hides NVMe Sensor 1 and 2 and the board's AUXTIN2
/// (`temps::TTY_BLOCK`) from its gauges and TEMPS rows, even when one is
/// the hottest reading.
#[test]
fn a_tty_blocked_sensor_never_reaches_the_screen() {
    use llama_watch::config::Temps;
    use llama_watch::sources::chips::Kind;
    use llama_watch::sources::temps::{self, TempLook, TempReading};
    let reading = |chip: &str, sensor: &str, input: &str, kind: Kind, tenths: i32| TempReading {
        chip: chip.to_owned(),
        sensor: sensor.to_owned(),
        input: input.to_owned(),
        labelled: true,
        kind,
        tenths,
        max_tenths: None,
        crit_tenths: None,
    };
    let readings = [
        reading("nvme-317k", "Composite", "temp1", Kind::Nvme, 450),
        reading("nvme-317k", "Sensor 1", "temp2", Kind::Nvme, 940),
        reading("nvme-317k", "Sensor 2", "temp3", Kind::Nvme, 964),
        reading("nct6798", "SYSTIN", "temp1", Kind::Board, 400),
        reading("nct6798", "AUXTIN2", "temp5", Kind::Board, 986),
    ];
    let screen = |look: &TempLook, cols: u16, rows: u16| {
        let mut model = gauge_model();
        model.temps = Some(temps::panel(&readings, look));
        whole(&draw(&model, cols, rows))
    };
    for (cols, rows) in [(213u16, 60u16), (160, 26), (160, 49)] {
        let all = screen(&TempLook::new(&Temps::default()), cols, rows);
        // Short labels for the TEMPS rows, values for the gauges.
        for gone in ["s1", "s2", "aux2", "94 C", "96 C", "99 C"] {
            assert!(!all.contains(gone), "{cols}x{rows}: {gone}\n{all}");
        }
        assert!(all.contains("NVMe0         45 C"), "{cols}x{rows}\n{all}");
        assert!(all.contains("board         40 C"), "{cols}x{rows}\n{all}");
        // Without the block the same readings do reach the screen.
        let open = screen(&TempLook::default(), cols, rows);
        assert!(
            open.contains("99 C") && open.contains("96 C"),
            "{cols}x{rows}\n{open}"
        );
    }
}

// ---- #167: the GPU reservation and every claim in the header ------------

/// A health line as the watcher builds it since #167: latencies in five
/// cells, `/slots` keeping the room of `idle`. `ms` is every latency.
fn fixed_health(ms: &str, slots: HealthStatus) -> Vec<HealthSeg> {
    let seg = |name: &str, status: HealthStatus, note: &str, word: usize, cells: usize| HealthSeg {
        name: name.to_owned(),
        status,
        note: note.to_owned(),
        word_cells: word,
        note_cells: cells,
    };
    vec![
        seg("llama-swap", HealthStatus::Ok, ms, 0, 5),
        seg("/running", HealthStatus::Ok, "", 0, 0),
        seg("/slots", slots, ms, 4, 5),
        seg("metrics", HealthStatus::Ok, ms, 0, 5),
        seg("activity", HealthStatus::Ok, ms, 0, 5),
        seg("nvml", HealthStatus::Ok, "", 0, 0),
        seg("hwmon", HealthStatus::Ok, "", 0, 0),
        seg("proc", HealthStatus::Ok, "", 0, 0),
        seg("disk", HealthStatus::Ok, "", 0, 0),
        seg("net", HealthStatus::Ok, "", 0, 0),
        seg("psi", HealthStatus::Ok, "", 0, 0),
    ]
}

fn at(text: &str) -> std::time::SystemTime {
    UNIX_EPOCH + Duration::from_secs(u64::try_from(parse_rfc3339(text).expect("rfc3339")).unwrap())
}

/// The header of the mock-ups: no model, idle slots, `swap none`, with the
/// #167 health line. Invented names throughout.
fn reservation_base(state: WatchState) -> TtyModel {
    let mut model = sample(state);
    model.now = claim_now();
    model.tz_offset_s = 0;
    model.model_name = "--".to_owned();
    model.model_detail = String::new();
    model.model_id = String::new();
    model.slots_line = "0/0 idle".to_owned();
    model.swap_line = "none".to_owned();
    model.setup = None;
    model.health = fixed_health("  9ms", HealthStatus::Idle);
    model.snapshot = Some(33_711);
    model.snapshot_age = "0.0s".to_owned();
    model.errors = 0;
    model.uptime = "9m59s".to_owned();
    model
}

fn serving(state: WatchState, name: &str, id: &str) -> TtyModel {
    let mut model = reservation_base(state);
    model.model_name = name.to_owned();
    model.model_id = id.to_owned();
    model.slots_line = if state == WatchState::Generating {
        "1/1 busy".to_owned()
    } else {
        "0/1 idle".to_owned()
    };
    model.health = fixed_health(" 14ms", HealthStatus::Ok);
    model
}

fn idle_claim_on(model: &str, holder: &str, secs: u64) -> HeaderClaim {
    HeaderClaim {
        model: model.to_owned(),
        holder: holder.to_owned(),
        expires_at: Some(claim_now() + Duration::from_secs(secs)),
        ..HeaderClaim::default()
    }
}

/// 1: the GPU held for a queue, with a reason, nothing loaded.
fn state_gpu_held() -> TtyModel {
    let mut model = reservation_base(WatchState::Ready);
    model.gpu = Some(HeaderGpu {
        holder: "bakeoff".to_owned(),
        hold_until: Some(at("2026-10-13T21:01:00Z")),
        hold_reason: "batch window for queued model runs".to_owned(),
        ..HeaderGpu::default()
    });
    model
}

/// 2: a claim on the model being served, `secs` left.
fn state_served_claim(secs: u64) -> TtyModel {
    let mut model = serving(WatchState::Ready, "Tiel Coder 35B-A3B", "tiel-coder");
    model.claims = vec![idle_claim_on("tiel-coder", "opencode", secs)];
    model
}

/// 3: the GPU reserved and in use, a guest claim under it on the served
/// model, a hold on a model that is not loaded.
fn state_in_use_guest_other() -> TtyModel {
    let mut model = serving(WatchState::Generating, "Qwen 27B", "qwen-27b");
    model.gpu = Some(HeaderGpu {
        holder: "bakeoff".to_owned(),
        in_use: true,
        ..HeaderGpu::default()
    });
    model.claims = vec![
        HeaderClaim {
            model: "tiel-coder".to_owned(),
            holder: "eval-runner".to_owned(),
            hold_until: Some(at("2026-10-12T23:30:00Z")),
            hold_reason: "nightly evals".to_owned(),
            ..HeaderClaim::default()
        },
        HeaderClaim {
            conditional: true,
            ..idle_claim_on("qwen-27b", "opencode", 4 * 60 + 12)
        },
    ];
    model
}

/// 4: the GPU handed to work outside llama-swap, models still stopping.
fn state_external_draining() -> TtyModel {
    let mut model = reservation_base(WatchState::Ready);
    model.gpu_busy = true;
    model.slots_line = "--".to_owned();
    model.swap_line = "--".to_owned();
    model.gpu = Some(HeaderGpu {
        holder: "bakeoff".to_owned(),
        external: true,
        draining: true,
        expires_at: Some(claim_now() + Duration::from_secs(42)),
        ..HeaderGpu::default()
    });
    model
}

/// 5: a claim displaced three minutes ago by the reservation's holder.
fn state_displaced() -> TtyModel {
    let mut model = serving(WatchState::Generating, "Qwen 27B", "qwen-27b");
    model.gpu = Some(HeaderGpu {
        holder: "bakeoff".to_owned(),
        expires_at: Some(claim_now() + Duration::from_secs(23 * 60 + 58)),
        ..HeaderGpu::default()
    });
    model.displaced = vec![HeaderDisplaced {
        model: "tiel-coder".to_owned(),
        to: "qwen-27b".to_owned(),
        by: "bakeoff".to_owned(),
        at: claim_now() - Duration::from_secs(3 * 60),
        tz_offset_s: 0,
    }];
    model
}

/// 6: nothing claimed or reserved.
fn state_nothing() -> TtyModel {
    serving(WatchState::Ready, "Tiel Coder 35B-A3B", "tiel-coder")
}

type StateFn = fn() -> TtyModel;

const GPU_STATES: [(&str, StateFn); 6] = [
    ("held", state_gpu_held),
    ("served-claim", || state_served_claim(9 * 60 + 59)),
    ("in-use-guest-other", state_in_use_guest_other),
    ("external-draining", state_external_draining),
    ("displaced", state_displaced),
    ("nothing", state_nothing),
];

const GPU_SIZES: [(u16, u16); 3] = [(213, 60), (192, 60), (160, 26)];

fn gpu_golden_name(state: &str, cols: u16, rows: u16) -> String {
    if rows == 60 {
        format!("gpu-{state}-{cols}.json")
    } else {
        format!("gpu-{state}-{cols}x{rows}.json")
    }
}

/// #167: every state at 213x60, 192x60 and 160x26, character and colour.
#[test]
fn gpu_claims_goldens_match_character_and_colour() {
    for (state, model) in GPU_STATES {
        for (cols, rows) in GPU_SIZES {
            let name = gpu_golden_name(state, cols, rows);
            let fix = load(&name);
            assert_eq!((fix.w, fix.h), (cols, rows), "{name}");
            assert_frame(&name, &fix, &model());
        }
    }
}

#[test]
#[ignore = "run with --ignored to write the #167 GPU and claims goldens"]
fn dump_gpu_claims_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (state, model) in GPU_STATES {
        for (cols, rows) in GPU_SIZES {
            let grid = draw(&model(), cols, rows);
            let name = gpu_golden_name(state, cols, rows);
            std::fs::write(dir.join(name), dump_grid(&grid)).expect("write #167 golden");
        }
    }
}

#[test]
#[ignore = "dev tool: print the #167 header and health rows as text"]
fn print_gpu_claims_rows() {
    for (state, model) in GPU_STATES {
        for (cols, rows) in GPU_SIZES {
            let grid = draw(&model(), cols, rows);
            println!("== {state} {cols}x{rows}");
            println!("{}", row_string(&grid, 0).trim_end());
            println!("{}", row_string(&grid, rows - 2).trim_end());
        }
    }
}

/// The claims area of row 0: from after `swap <value>` to the clock block.
fn claims_text(model: &TtyModel, cols: u16) -> String {
    let header = header_line(model, cols);
    let from = header
        .find(&format!("swap {}", model.swap_line))
        .map(|at| at + "swap ".len() + model.swap_line.len())
        .expect("swap");
    let clock = if cols >= 200 { "COOL" } else { "2026-09-23" };
    let to = header.find(clock).expect("clock");
    header[from..to].trim().to_owned()
}

/// #167: the order, words and forms of the mock-ups, in full at 286
/// columns, and as the shedding order leaves them at 213.
#[test]
fn gpu_claims_text_at_286_and_213() {
    for (state, full, at_213) in [
        (
            state_gpu_held(),
            "gpu bakeoff · hold until 10-13 21:01 · batch window for queued model runs",
            "gpu bakeoff · hold until 10-13 21:01 · batch window for queued model runs",
        ),
        (
            state_served_claim(599),
            "claim opencode on tiel-coder ·  9:59 left",
            "claim opencode on tiel-coder ·  9:59 left",
        ),
        (
            state_in_use_guest_other(),
            "gpu bakeoff · in use   guest opencode on qwen-27b ·  4:12 left   \
             hold eval-runner on tiel-coder · until 23:30 · nightly evals",
            "gpu bakeoff · in use   guest opencode on qwen-27b ·  4:12 left   +1 more",
        ),
        (
            state_external_draining(),
            "gpu bakeoff · external · draining ·  0:42 left",
            "gpu bakeoff · external · draining ·  0:42 left",
        ),
        (
            state_displaced(),
            "gpu bakeoff · 23:58 left   displaced tiel-coder -> qwen-27b by bakeoff 18:44",
            "bakeoff · 23:58 left   tiel-coder -> qwen-27b by bakeoff 18:44",
        ),
        (state_nothing(), "", ""),
    ] {
        assert_eq!(claims_text(&state, 286), full);
        assert_eq!(claims_text(&state, 213), at_213);
    }
}

/// #167 colours, as the mock-ups: holders bright cyan, `hold` yellow, a
/// hold's end bright yellow, in use bright green, `draining` in slot 6
/// (the palette's orange), labels and `on` grey, models and reasons white.
#[test]
fn gpu_claims_colours_follow_the_mock_ups() {
    let fg = |model: &TtyModel, needle: &str| {
        let grid = draw(model, 286, 60);
        let text = row_string(&grid, 0);
        grid.get(char_at(&text, needle) as u16, 0).unwrap().fg
    };
    let held = state_gpu_held();
    assert_eq!(fg(&held, "gpu bakeoff"), C16::BrightBlack);
    assert_eq!(fg(&held, "bakeoff"), C16::BrightCyan);
    assert_eq!(fg(&held, "hold until"), C16::BrightYellow);
    assert_eq!(fg(&held, "batch"), C16::White);
    let busy = state_in_use_guest_other();
    assert_eq!(fg(&busy, "in use"), C16::BrightGreen);
    assert_eq!(fg(&busy, "guest"), C16::BrightBlack);
    assert_eq!(fg(&busy, "opencode"), C16::BrightCyan);
    assert_eq!(fg(&busy, " on qwen-27b"), C16::BrightBlack);
    assert_eq!(fg(&busy, "qwen-27b ·"), C16::White);
    assert_eq!(fg(&busy, "4:12 left"), C16::BrightWhite);
    assert_eq!(fg(&busy, "hold eval"), C16::Yellow);
    assert_eq!(fg(&busy, "eval-runner"), C16::BrightCyan);
    assert_eq!(fg(&busy, "until 23:30"), C16::BrightYellow);
    assert_eq!(fg(&busy, "nightly"), C16::White);
    let out = state_external_draining();
    assert_eq!(fg(&out, "external"), C16::White);
    assert_eq!(fg(&out, "draining"), C16::Cyan);
    assert_eq!(fg(&out, "0:42 left"), C16::Yellow, "under a minute");
    let gone = state_displaced();
    assert_eq!(fg(&gone, "displaced"), C16::BrightBlack);
    assert_eq!(fg(&gone, "tiel-coder"), C16::White);
    assert_eq!(fg(&gone, " -> "), C16::BrightBlack);
    assert_eq!(fg(&gone, "by bakeoff"), C16::BrightBlack);
    assert_eq!(fg(&gone, "18:44"), C16::White);
}

/// #167 at every size: the area stops two cells before the clock block, no
/// holder is cut, and the claims area is the only change to the row.
#[test]
fn gpu_claims_fit_every_size_without_cutting_a_holder() {
    for (state, model) in GPU_STATES {
        let model = model();
        let mut blank = model.clone();
        blank.claims.clear();
        blank.gpu = None;
        blank.displaced.clear();
        for cols in 160u16..=240 {
            let header: Vec<char> = header_line(&model, cols).chars().collect();
            let plain: Vec<char> = header_line(&blank, cols).chars().collect();
            let clock = if cols >= 200 { "COOL" } else { "2026-09-23" };
            let end = find_cells(&header, clock).expect("clock");
            let text: String = header.iter().collect();
            assert_eq!(&header[end..], &plain[end..], "{state} {cols}");
            let left: String = header[..end]
                .iter()
                .collect::<String>()
                .trim_end()
                .to_owned();
            assert!(left.chars().count() + 2 <= end, "{state} {cols}: {text}");
            let base: String = plain[..end]
                .iter()
                .collect::<String>()
                .trim_end()
                .to_owned();
            assert!(left.starts_with(&base), "{state} {cols}: {text}");
            let area = &left[base.len()..];
            for holder in ["bakeoff", "opencode", "eval-runner"] {
                let head = &holder[..holder.len() - 1];
                assert!(
                    area.contains(holder) || !area.contains(head),
                    "{state} {cols}: {holder} cut: {text}"
                );
            }
        }
    }
}

/// Columns of each word of `text` that follows `after`, numbers left out.
fn words_after(text: &str, after: usize) -> Vec<(usize, String)> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut col = after;
    while col < chars.len() {
        if chars[col] != ' ' && (col == 0 || chars[col - 1] == ' ') {
            let word: String = chars[col..].iter().take_while(|c| **c != ' ').collect();
            if !word.starts_with(|c: char| c.is_ascii_digit()) {
                out.push((col, word));
            }
        }
        col += 1;
    }
    out
}

/// #167: nothing right of a number moves when it grows a digit: 9 to 10
/// ms and 99 to 100 ms on the health line, `9:59` to `10:00` in the header
/// (a claim's and the reservation's), and the health tail's numbers.
#[test]
fn nothing_right_of_a_number_moves() {
    let row = |model: &TtyModel, cols: u16, rows: u16, row: u16| {
        row_string(&draw(model, cols, rows), row)
    };
    for (cols, rows) in GPU_SIZES {
        let health = |ms: &str| {
            let mut model = state_nothing();
            model.health = fixed_health(ms, HealthStatus::Ok);
            row(&model, cols, rows, rows - 2)
        };
        for (a, b) in [("  9ms", " 10ms"), (" 99ms", "100ms")] {
            let (ra, rb) = (health(a), health(b));
            let at = ra.find(a).expect("number");
            assert_eq!(
                words_after(&ra, at + a.len()),
                words_after(&rb, at + b.len()),
                "{cols}x{rows} {a} -> {b}\n{ra}\n{rb}"
            );
        }
        // `/slots` keeps the room of `idle`.
        let mut ok = state_nothing();
        ok.health = fixed_health("  9ms", HealthStatus::Ok);
        let mut idle = ok.clone();
        idle.health = fixed_health("  9ms", HealthStatus::Idle);
        let (ro, ri) = (
            row(&ok, cols, rows, rows - 2),
            row(&idle, cols, rows, rows - 2),
        );
        let at = ro.find("/slots").expect("/slots") + "/slots idle".len();
        assert_eq!(
            words_after(&ro, at),
            words_after(&ri, at),
            "{cols}x{rows}\n{ro}\n{ri}"
        );

        // The tail: snapshot, age, errors, uptime.
        let tail = |seq: u64, age: &str, errors: u64, up: &str| {
            let mut model = state_nothing();
            model.snapshot = Some(seq);
            model.snapshot_age = age.to_owned();
            model.errors = errors;
            model.uptime = up.to_owned();
            row(&model, cols, rows, rows - 2)
        };
        let small = tail(9, "0.1s", 9, "9m59s");
        let big = tail(9_999_999, "1h10m", 10, "1h10m00s");
        let words = |text: &str| -> Vec<usize> {
            ["errors", "up"]
                .iter()
                .map(|w| text.rfind(&format!("{w} ")).expect("word"))
                .collect()
        };
        assert_eq!(words(&small), words(&big), "{cols}x{rows}\n{small}\n{big}");
        assert_eq!(small.find("snapshot"), big.find("snapshot"));
        assert_eq!(small.len(), big.len());

        // The header countdowns, a claim's and the reservation's.
        for (a, b) in [
            (state_served_claim(9 * 60 + 59), state_served_claim(10 * 60)),
            (
                {
                    let mut m = state_displaced();
                    m.gpu.as_mut().unwrap().expires_at =
                        Some(claim_now() + Duration::from_secs(599));
                    m
                },
                {
                    let mut m = state_displaced();
                    m.gpu.as_mut().unwrap().expires_at =
                        Some(claim_now() + Duration::from_secs(600));
                    m
                },
            ),
        ] {
            let (ha, hb) = (row(&a, cols, rows, 0), row(&b, cols, rows, 0));
            let at = ha.find("9:59 left").expect("9:59") + "9:59".len();
            assert_eq!(
                hb.find("10:00 left").map(|x| x + "10:00".len()),
                Some(at),
                "{hb}"
            );
            assert_eq!(
                words_after(&ha, at),
                words_after(&hb, at),
                "{cols}x{rows}\n{ha}\n{hb}"
            );
        }
    }
}

/// A displaced note shows for ten minutes after it happened; in the future
/// (a clock step) it shows too.
#[test]
fn a_displaced_note_shows_for_ten_minutes() {
    let mut model = state_displaced();
    model.gpu = None;
    for (ago, shown) in [(0i64, true), (599, true), (600, false), (-60, true)] {
        let at = if ago >= 0 {
            claim_now() - Duration::from_secs(ago.unsigned_abs())
        } else {
            claim_now() + Duration::from_secs(ago.unsigned_abs())
        };
        model.displaced[0].at = at;
        assert_eq!(
            claims_text(&model, 213).contains("displaced tiel-coder"),
            shown,
            "{ago}"
        );
    }
}

/// #167: with text off, the `text off` tag keeps its room: the claims area
/// stops short of it, even when a reason could fill the row.
#[test]
fn text_off_keeps_its_tag_beside_the_claims() {
    for (cols, _) in GPU_SIZES {
        let mut model = state_gpu_held();
        model.show_text = false;
        let header = header_line(&model, cols);
        let tag = header
            .find("text off")
            .unwrap_or_else(|| panic!("{cols}: {header}"));
        let holder = header.find("gpu bakeoff").expect("holder");
        assert!(holder < tag, "{cols}: {header}");
        let clock = if cols >= 200 { "COOL" } else { "2026-09-23" };
        let end = header.find(clock).expect("clock");
        assert!(tag + "text off".len() + 2 <= end, "{cols}: {header}");
        assert!(
            header[holder..tag].contains("hold until"),
            "{cols}: {header}"
        );
    }
    // At 160x26 the reason is cut to make room for the tag.
    let mut model = state_gpu_held();
    model.show_text = false;
    let header = header_line(&model, 160);
    assert!(header.contains('\u{2026}'), "{header}");
    // Text on, nothing is held back: the row is as before.
    let header = header_line(&state_gpu_held(), 160);
    assert!(
        header.contains("batch window for queued m\u{2026}"),
        "{header}"
    );
}

/// #170: by default the bars are solid everywhere: no cell of the console
/// or of the grid file is the segment, in either font, and the lit gauge
/// cells are `▄` in both. `bar_segments` brings the segment back.
#[test]
fn gauge_bars_are_solid_unless_bar_segments() {
    for glyphs in [ChartGlyphs::Eighths, ChartGlyphs::Halves] {
        let mut model = gauge_model();
        model.chart_glyphs = glyphs;
        assert!(!model.bar_segments, "solid is the default");
        let grid = layout_exact(&model, 213, 60);
        let segs = (0..60u16)
            .flat_map(|r| (0..213u16).map(move |c| (c, r)))
            .filter(|&(c, r)| {
                let cell = grid.get(c, r).unwrap();
                cell.ch == SEG || cell.rich == SEG
            })
            .count();
        assert_eq!(segs, 0, "{glyphs:?}");
        let image =
            llama_watch::grid_out::image_of(&grid, llama_core::palette::Palette::Llama, 1, 0);
        assert!(image.cells.iter().all(|c| c.ch != SEG), "{glyphs:?}");
        let row = meter_row(&grid, "GPU");
        let col = meter_bar_col(&grid, row, "GPU");
        let cell = grid.get(col, row).unwrap();
        assert_eq!((cell.ch, cell.rich), ('▄', '▄'), "{glyphs:?}");
        assert_eq!(
            image.cells[usize::from(row) * 213 + usize::from(col)].ch,
            '▄'
        );
        model.bar_segments = true;
        let grid = layout_exact(&model, 213, 60);
        let cell = grid.get(col, row).unwrap();
        let console = if glyphs == ChartGlyphs::Eighths {
            SEG
        } else {
            '▄'
        };
        assert_eq!((cell.ch, cell.rich), (console, SEG), "{glyphs:?}");
    }
}

/// #173: a renamed, non-ASCII sensor label (`aééééééx`) draws without a
/// panic at 213 columns; its gauge label keeps seven cells.
#[test]
fn a_multi_byte_sensor_label_draws_at_its_width() {
    let limits = Some((800, 900));
    let name = "a\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}x";
    let mut model = with_host_rates(false);
    model.temps = Some(TempPanel {
        groups: vec![
            TempGroup {
                name: name.to_string(),
                items: vec![temp_marked("", 92, Level::Normal, limits)],
            },
            TempGroup {
                name: "cd".to_string(),
                items: vec![temp_marked("", 25, Level::Normal, limits)],
            },
        ],
    });
    let grid = draw(&model, 213, 60);
    // The gauge label is seven cells: the console folds `é` to `e`
    // (sanitize), the grid file keeps it.
    let all = whole(&grid);
    assert!(all.contains("aeeeeee       92 C "), "{all}");
    let (row, col) = label_hit(&grid, "aeeeeee");
    let rich: String = (col..col + 8)
        .map(|c| grid.get(c, row).unwrap().rich)
        .collect();
    assert_eq!(
        rich,
        format!("{} ", name.chars().take(7).collect::<String>())
    );
}

// ---- #177: SETUP beside the rate digits in place of the glance ------------

/// Cells `from..to` of `row`, trimmed at the end.
fn cells_between(grid: &llama_watch::tty::grid::Grid, row: u16, from: u16, to: u16) -> String {
    (from..to)
        .map(|col| grid.get(col, row).unwrap().ch)
        .collect::<String>()
        .trim_end()
        .to_string()
}

fn one_item_row(label: &str, text: &str) -> SetupRow {
    SetupRow {
        label: label.to_owned(),
        items: vec![setup_item(text, false)],
    }
}

/// A row of four items that takes two lines in the 19 value cells at 192
/// columns: `aaaa · bbbbbbbb` (15), then `cccccc · dddddddddd` (19).
fn four_item_row() -> SetupRow {
    SetupRow {
        label: "wrap".to_owned(),
        items: ["aaaa", "bbbbbbbb", "cccccc", "dddddddddd"]
            .into_iter()
            .map(|text| setup_item(text, false))
            .collect(),
    }
}

/// #177: from 192 columns SETUP starts at column 64 on the headline's top
/// row, labels at 66 and values at 75, and it is not drawn under SLOTS.
#[test]
fn setup_sits_beside_the_rate_digits_from_192_columns() {
    let rows = vec![
        one_item_row("alpha", "zq-one"),
        one_item_row("beta", "zq-two"),
        one_item_row("gamma", "zq-three"),
    ];
    for cols in [192u16, 213] {
        let at = format!("{cols}x60");
        let grid = draw(&with_setup_rows(rows.clone()), cols, 60);
        let end = cols / 2 - 2;
        let title = cells_between(&grid, 2, 64, end);
        assert!(title.starts_with("SETUP  qwen3.6-35b-a3b"), "{at}: {title}");
        assert_eq!(grid.get(64, 2).unwrap().fg, C16::White, "{at}");
        assert_eq!(
            cells_between(&grid, 3, 64, end),
            "  alpha    zq-one",
            "{at}"
        );
        assert_eq!(
            cells_between(&grid, 4, 64, end),
            "  beta     zq-two",
            "{at}"
        );
        assert_eq!(
            cells_between(&grid, 5, 64, end),
            "  gamma    zq-three",
            "{at}"
        );
        assert_eq!(cells_between(&grid, 6, 64, end), "", "{at}");
        // Drawn once: not under SLOTS as well.
        let all = whole(&grid);
        assert_eq!(all.matches("SETUP").count(), 1, "{at}\n{all}");
        assert_eq!(all.matches("zq-").count(), 3, "{at}\n{all}");
        assert_blank_edges(&grid);
    }
}

/// #177: under 192 columns (160x49) the spot beside the digits stays
/// empty and SETUP stays under SLOTS: title at column 2, labels at 4,
/// values at 13.
#[test]
fn setup_stays_under_slots_at_160_columns() {
    let model = with_setup_rows(vec![one_item_row("alpha", "zq-one")]);
    let grid = draw(&model, 160, 49);
    for row in 2..12 {
        assert_eq!(cells_between(&grid, row, 64, 78), "", "row {row}");
    }
    let title = setup_at(&grid);
    assert!(title >= 12, "under SLOTS: {title}");
    assert!(cells_between(&grid, title, 2, 78).starts_with("SETUP  qwen3.6"));
    assert_eq!(cells_between(&grid, title + 1, 2, 78), "  alpha    zq-one");
    assert_eq!(whole(&grid).matches("SETUP").count(), 1);
}

/// #177: beside the digits a row that does not fit wraps whole items onto
/// a continuation line in the value column; an item wider than the room
/// has a line of its own and is cut with `…` at column 93.
#[test]
fn setup_beside_the_digits_wraps_whole_items() {
    let wide = SetupRow {
        label: "cut".to_owned(),
        items: vec![setup_item(&"e".repeat(30), false), setup_item("ff", false)],
    };
    let grid = draw(&with_setup_rows(vec![four_item_row(), wide]), 192, 60);
    assert_eq!(
        cells_between(&grid, 3, 64, 94),
        "  wrap     aaaa \u{b7} bbbbbbbb"
    );
    assert_eq!(
        cells_between(&grid, 4, 64, 94),
        format!("{}cccccc \u{b7} dddddddddd", " ".repeat(11))
    );
    assert_eq!(
        cells_between(&grid, 5, 64, 94),
        format!("  cut      {}\u{2026}", "e".repeat(18))
    );
    assert_eq!(
        cells_between(&grid, 6, 64, 94),
        format!("{}ff", " ".repeat(11))
    );
    // Under SLOTS the same rows keep one line each (#114).
    let grid = draw(&with_setup_rows(vec![four_item_row()]), 160, 49);
    let title = setup_at(&grid);
    assert_eq!(
        cells_between(&grid, title + 1, 2, 78),
        "  wrap     aaaa \u{b7} bbbbbbbb \u{b7} cccccc \u{b7} dddddddddd"
    );
}

/// #177: rows that do not fit beside the digits drop, and the title's `+N`
/// counts them with the view's own `more`, and with every row cut short:
/// trailing items left out on its last line. A row whose one item left is
/// cut with `…` is drawn and not counted.
#[test]
fn setup_beside_the_digits_counts_dropped_rows_in_its_more() {
    // The headline has ten rows at 192x60: the title and nine lines. The
    // wrapped row takes two, seven single rows the rest, three drop.
    let mut rows = vec![four_item_row()];
    rows.extend((1..=10).map(|i| one_item_row("row", &format!("zq-{i:02}"))));
    let mut model = with_setup_rows(rows);
    let grid = draw(&model, 192, 60);
    let all = whole(&grid);
    assert!(all.contains("zq-07") && !all.contains("zq-08"), "{all}");
    let title = cells_between(&grid, 2, 64, 94);
    assert!(title.ends_with("+3"), "{title}");
    assert_eq!(grid.get(93, 2).unwrap().ch, '3');
    model.setup.as_mut().expect("setup").more = 1;
    let title = cells_between(&draw(&model, 192, 60), 2, 64, 94);
    assert!(title.ends_with("+4"), "{title}");
    // Eight single rows leave one line for the wrapped row: it keeps what
    // fits on that line and leaves two items out, so it counts.
    let eight: Vec<SetupRow> = (1..=8)
        .map(|i| one_item_row("row", &format!("zq-{i:02}")))
        .collect();
    let mut rows = eight.clone();
    rows.push(four_item_row());
    let mut model = with_setup_rows(rows);
    let grid = draw(&model, 192, 60);
    assert_eq!(
        cells_between(&grid, 11, 64, 94),
        "  wrap     aaaa \u{b7} bbbbbbbb"
    );
    assert!(!whole(&grid).contains("cccccc"));
    let title = cells_between(&grid, 2, 64, 94);
    assert!(title.ends_with("+1"), "{title}");
    model.setup.as_mut().expect("setup").more = 2;
    let title = cells_between(&draw(&model, 192, 60), 2, 64, 94);
    assert!(title.ends_with("+3"), "{title}");
    // One item too wide for its line is cut with `…`: drawn, not counted.
    let mut rows = eight;
    rows.push(one_item_row("cut", &"e".repeat(30)));
    let grid = draw(&with_setup_rows(rows), 192, 60);
    assert_eq!(
        cells_between(&grid, 11, 64, 94),
        format!("  cut      {}\u{2026}", "e".repeat(18))
    );
    assert!(!cells_between(&grid, 2, 64, 94).contains('+'));
}

/// #177: under SLOTS a row is one line; one that leaves items out counts
/// in `+N`, one whose only item is cut with `…` does not.
#[test]
fn setup_under_slots_counts_rows_cut_short() {
    let cut = SetupRow {
        label: "cut".to_owned(),
        items: vec![setup_item(&"e".repeat(62), false), setup_item("ff", false)],
    };
    let grid = draw(&with_setup_rows(vec![cut]), 160, 49);
    let title = setup_at(&grid);
    assert!(
        cells_between(&grid, title, 2, 78).ends_with("+1"),
        "{}",
        cells_between(&grid, title, 2, 78)
    );
    let lone = one_item_row("cut", &"e".repeat(80));
    let grid = draw(&with_setup_rows(vec![lone]), 160, 49);
    let title = setup_at(&grid);
    assert!(!cells_between(&grid, title, 2, 78).contains('+'));
    assert!(cells_between(&grid, title + 1, 2, 78).ends_with('\u{2026}'));
}

/// #177: 191 columns draw nothing beside the digits and keep SETUP under
/// SLOTS; 192 draw it beside them.
#[test]
fn setup_moves_beside_the_digits_at_exactly_192_columns() {
    let model = with_setup_rows(vec![one_item_row("alpha", "zq-one")]);
    let grid = draw(&model, 191, 60);
    for row in 2..12 {
        assert_eq!(cells_between(&grid, row, 64, 93), "", "191: row {row}");
    }
    let title = setup_at(&grid);
    assert!(title >= 12, "191: under SLOTS at {title}");
    assert!(cells_between(&grid, title, 2, 93).starts_with("SETUP  qwen3.6"));
    assert_eq!(cells_between(&grid, title + 1, 2, 93), "  alpha    zq-one");
    let grid = draw(&model, 192, 60);
    assert!(cells_between(&grid, 2, 64, 94).starts_with("SETUP  qwen3.6"));
    assert_eq!(cells_between(&grid, 3, 64, 94), "  alpha    zq-one");
    assert_eq!(setup_at(&grid), 2);
}

/// #177: with llama-swap down, starting, or polling off, a setup view
/// draws nothing beside the digits.
#[test]
fn setup_beside_the_digits_only_while_generating_or_ready() {
    for state in [
        WatchState::AiDown,
        WatchState::Starting,
        WatchState::NoLlama,
    ] {
        let mut model = sample(state);
        model.setup = Some(qwen_setup());
        let grid = draw(&model, 213, 60);
        for row in 2..12 {
            assert_eq!(
                cells_between(&grid, row, 64, 104),
                "",
                "{state:?}: row {row}"
            );
        }
        assert!(!whole(&grid).contains("SETUP"), "{state:?}");
    }
}

/// #177: a view without rows draws its title alone beside the digits.
#[test]
fn setup_beside_the_digits_without_rows_is_the_title() {
    let grid = draw(&with_setup_rows(Vec::new()), 192, 60);
    let title = cells_between(&grid, 2, 64, 94);
    assert!(title.starts_with("SETUP  qwen3.6-35b-a3b"), "{title}");
    assert!(!title.contains('+'), "{title}");
    for row in 3..12 {
        assert_eq!(cells_between(&grid, row, 64, 94), "", "row {row}");
    }
}

/// #177: a temperature gauge shows its group's reading nearest its own
/// limit, not the hottest: 60 C of a 65 C warning point over 80 C of
/// 100 C. A tie on the gauge percent goes to the hotter reading. The label
/// stays the group's name.
#[test]
fn a_temperature_gauge_picks_the_reading_nearest_its_limit() {
    let gauge_text = |items: Vec<TempItem>| {
        let mut model = gauge_model();
        model.fans = None;
        model.temps = Some(TempPanel {
            groups: vec![TempGroup {
                name: "probe".to_string(),
                items,
            }],
        });
        let grid = draw(&model, 213, 60);
        let (row, col) = label_hit(&grid, "probe");
        cells_between(&grid, row, col, col + 18)
    };
    let near = gauge_text(vec![
        temp_marked("hot", 80, Level::Normal, Some((1000, 1100))),
        temp_marked("near", 60, Level::Normal, Some((650, 750))),
    ]);
    assert_eq!(near, "probe         60 C");
    // 50 C of 80 and 35 C of 50 are both half way from 20 C.
    let tie = gauge_text(vec![
        temp_marked("a", 35, Level::Normal, Some((500, 600))),
        temp_marked("b", 50, Level::Normal, Some((800, 900))),
    ]);
    assert_eq!(tie, "probe         50 C");
}

// ---- #180: tqLLM, one request reading and one generating ----------------------

/// An invented tqLLM launch: podman, `tqserve` its entry point.
const TQLLM_CMD: &str = "podman run --rm --name tq --device nvidia.com/gpu=all -p 127.0.0.1:8000:8000 -v /models/invented:/models:ro --entrypoint tqserve ghcr.io/example/moe-server:1 --model /models/Invented-Q4_K_M.gguf --context 65536 --np 2 -md /models/drafter.gguf --draft-max 4 --reasoning on --reasoning-budget 2048 --temperature 0.6 --top-p 0.95 --moe-host-mb 20480 --port 8000";

/// #180: tqLLM generating, from `fixtures/llama/tqllm-snapshot.json`
/// (request 4 decoding, request 5 reading, request 6 queued). SETUP comes
/// from its flags and its report, with `/health`'s fit; the engine line
/// has no KV item; the SLOTS lines, the engine numbers and the line are
/// the ones `service.rs` and `tests/poller.rs` pin for that snapshot; GEN
/// and PROMPT are the running requests' own rates; RECENT has both
/// requests in flight above the finished rows, whose cached count tqLLM
/// does not report.
fn tqllm_model() -> TtyModel {
    let mut model = sglang_model();
    model.chart_glyphs = ChartGlyphs::Eighths;
    model.model_name = "moe-tq".to_string();
    model.model_detail = "tqLLM".to_string();
    let body = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/llama/tqllm-snapshot.json"),
    )
    .expect("tqllm fixture");
    let snap = llama_watch::metrics::parse_tqllm(&body).expect("snapshot");
    let mut book = llama_watch::tqllm::TqllmBook::default();
    let now = std::time::SystemTime::now();
    book.observe("moe-tq", &snap, Instant::now(), now, now);
    book.set_fit("moe-tq", Some("2 slots x 32k fit in 22.5 GiB".to_owned()));
    let facts = book
        .observe("moe-tq", &snap, Instant::now(), now, now)
        .facts;
    let launch = llama_watch::sources::cmdline::parse_launch(TQLLM_CMD);
    assert_eq!(launch.backend, Backend::Tqllm);
    let detail = ModelDetail {
        ctx: facts.ctx,
        ..launch.detail.expect("detail")
    };
    let info = BackendInfo {
        kind: Backend::Tqllm,
        engine: EngineStats {
            spec_permille: Some(660),
            prefill_tps_tenths: Some(3099),
            decode_tps_tenths: Some(842),
            ..EngineStats::default()
        },
        ..BackendInfo::default()
    };
    let rules = Rules::builtin();
    let live = LiveCtx {
        backend: Backend::Tqllm,
        detail: Some(&detail),
        info: Some(&info),
        engine: Some(&facts.values),
    };
    model.setup = Some(SetupView {
        id: "moe-tq".to_owned(),
        name: "MoE TQ".to_owned(),
        more: 0,
        rows: rules.rows(&rules.extract(TQLLM_CMD), &live),
    });
    model.slots_line = "--".to_string();
    model.backend_lines = vec![
        "tqllm  running 2/2 · queued 1 · spec 66 % · prefill 310/s · decode 84.2/s".to_string(),
    ];
    let line = |id, done, cached, total, decoded, ctx_prompt| Slot {
        id,
        generating: true,
        done,
        cached,
        total,
        open: false,
        done_known: true,
        decoded,
        ctx_prompt: Some(ctx_prompt),
        n_ctx: Some(32_768),
        ctx_history: Vec::new(),
    };
    model.slots = vec![
        line(0, 288, 512, 800, 150, 800),
        line(1, 1_024, 1_024, 6_000, 0, 6_000),
    ];
    model.gen_tps = Some(83.5);
    model.prompt_tps = Some(1_137.777_778);
    let flight = |time: &str, prompt, cached, decoded, processed, dur: &str| Activity {
        time: time.to_string(),
        model: "moe-tq".to_string(),
        input_tok: prompt,
        cached_tok: cached,
        output_tok: decoded,
        dur: dur.to_string(),
        n_ctx: Some(32_768),
        inflight: Some(llama_watch::tty::layout::InFlight {
            decoding: decoded > 0,
            processed,
            target: true,
            ..llama_watch::tty::layout::InFlight::default()
        }),
        ..Activity::default()
    };
    let mut reading = flight("18:47:20", 6_000, 1_024, 0, 1_024, "0.9s");
    reading.prompt_tps = Some(1_137.777_778);
    reading.prompt_measured = true;
    let mut generating = flight("18:47:18", 800, 512, 150, 288, "2.5s");
    (generating.prompt_tps, generating.gen_tps) = (Some(443.076_923), Some(83.5));
    (generating.prompt_measured, generating.gen_measured) = (true, true);
    let mut rows = vec![reading, generating];
    // Finished rows: llama-swap's rates from tqLLM's timings; the cached
    // count only where one completed request matched the row.
    rows.extend(
        model
            .requests
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, mut r)| {
                r.live = false;
                r.model = "moe-tq".to_string();
                r.prompt_tps = Some(400.0);
                r.gen_tps = Some(85.4);
                if i > 0 {
                    r.cached_tok = 0;
                    r.cached_unknown = true;
                }
                r
            }),
    );
    model.requests = rows;
    model
}

const TQLLM_GOLDENS: [(&str, u16, u16); 2] = [
    ("tqllm-213x60.json", 213, 60),
    ("tqllm-160x48.json", 160, 48),
];

#[test]
fn tqllm_goldens_match_character_and_colour() {
    for (name, _, _) in TQLLM_GOLDENS {
        assert_frame(name, &load(name), &tqllm_model());
    }
}

#[test]
#[ignore = "run with --ignored to write the #180 tqLLM goldens"]
fn dump_tqllm_goldens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tty");
    for (name, cols, rows) in TQLLM_GOLDENS {
        let grid = draw(&tqllm_model(), cols, rows);
        std::fs::write(dir.join(name), dump_grid(&grid)).expect("write tqllm golden");
    }
}

/// #180: the frame's tqLLM parts at both sizes: the engine line without
/// KV, both SLOTS lines, the fit row and both in-flight rows.
#[test]
fn tqllm_frame_shows_its_lines_and_rows() {
    for (cols, rows) in [(213, 60), (160, 48)] {
        let grid = draw(&tqllm_model(), cols, rows);
        let all = whole(&grid);
        assert!(all.contains("tqllm  running 2/2 · queued 1"), "{all}");
        assert!(!all.contains("KV"), "no KV item: {all}");
        assert!(all.contains("tqLLM"), "{all}");
        let recent = row_with(&grid, "RECENT");
        let flights: Vec<String> = (recent + 1..recent + 3)
            .map(|row| row_string(&grid, row))
            .collect();
        assert!(flights[0].contains("pp"), "{flights:?}");
        assert!(flights[1].contains("gen"), "{flights:?}");
    }
}
