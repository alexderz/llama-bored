//! Text goldens for the tty11 layout. Frames are compared cell by cell.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use llama_core::backend::{Backend, BackendInfo, EngineStats};
use llama_core::detail::ModelDetail;
use llama_watch::collector::LoadSource;
use llama_watch::config::ChartGlyphs;
use llama_watch::setup_rules::{LiveCtx, Rules};
use llama_watch::tty::chart::ChartBucket;
use llama_watch::tty::grid::C16;
use llama_watch::tty::layout::{
    Activity, HealthSeg, HealthStatus, SetupView, Slot, TtyModel, WatchState, layout, replay_shown,
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
    assert_eq!(content(&grid, hello), "Hello world");
    assert_eq!(content(&grid, hello + 1), "second line");
    assert_eq!(content(&grid, hello + 2), "L".repeat(width));
    assert_eq!(content(&grid, hello + 3), "L".repeat(10));
    let out = row_with(&grid, "Xyz tail");
    assert_eq!(content(&grid, out), "Xyz tail");
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
        if !matches!(cell.ch, '█' | '▐' | '▌' | '▇' | '▄' | '░' | '▓' | '▒') {
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
    let legend = full_row(&narrow, row_with(&narrow, "prompt processing (prefill)"));
    assert!(legend.contains(LEGEND_NARROW), "160 legend: {legend}");
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
        let note = full_row(&grid, row_with(&grid, "prompt processing speed"));
        assert!(
            note.contains("prompt processing speed") && note.contains("generation speed"),
            "{cols}: {note}"
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
    assert_eq!(content(&done, 63), "Hello world█");

    let mut second = first.clone();
    second.out_lines = vec!["Hello world\nNEWCHARS".to_string()];
    second.out_shown = "Hello world".chars().count();
    second.replay_frame = Some(0);
    let held = draw(&second, 240, 67);
    assert_eq!(
        content(&held, 63),
        "Hello world█",
        "frame 0 must not blank OUT"
    );
    assert!(
        !row_string(&held, 63).contains("NEWCHARS"),
        "the new delta waits for a later frame"
    );

    second.replay_frame = Some(10);
    let revealed = draw(&second, 240, 67);
    assert!(
        (40..64).any(|row| content(&revealed, row).contains("NEWCHARS")),
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

    let tall = draw(&model, 480, 135);
    // The bar is still a spectrum: blue at its start, white-hot on red in
    // its top step.
    assert_eq!(tall.get(242, 16).unwrap().fg, C16::Blue, "tall bar start");
    let top = bar_from(&tall, 16, 242);
    let body = bar_from(&tall, 17, 242);
    assert_eq!(fg_sgr(*top.last().unwrap()), 97, "tall bar top cap");
    assert_eq!(fg_sgr(*body.last().unwrap()), 91, "tall bar body cap");
    assert_eq!(
        steps_of(&body),
        STEPS,
        "tall body runs every step, {body:?}"
    );
    assert_eq!(tall.get(241, 5).unwrap().ch, '█');
    assert_eq!(fg_sgr(tall.get(241, 5).unwrap().fg), 97, "digit top rows");
    assert_eq!(tall.get(241, 9).unwrap().ch, '█');
    assert_eq!(fg_sgr(tall.get(241, 9).unwrap().fg), 91, "digit body");

    let short = draw(&model, 240, 67);
    assert_eq!(short.get(122, 10).unwrap().fg, C16::Blue, "1-row bar start");
    assert_eq!(short.get(170, 10).unwrap().ch, '█');
    assert_eq!(
        fg_sgr(short.get(170, 10).unwrap().fg),
        97,
        "1-row bar ends hot"
    );
    let row = bar_from(&short, 10, 122);
    assert_eq!(
        steps_of(&row),
        [
            C16::Blue,
            C16::BrightBlue,
            C16::Magenta,
            C16::BrightMagenta,
            C16::BrightWhite
        ],
        "1-row capped bar: four steps then the white-hot top step, {row:?}"
    );
    assert_eq!(short.get(121, 4).unwrap().ch, '█');
    assert_eq!(fg_sgr(short.get(121, 4).unwrap().fg), 97, "short digit top");
    assert_eq!(short.get(121, 6).unwrap().ch, '█');
    assert_eq!(
        fg_sgr(short.get(121, 6).unwrap().fg),
        91,
        "short digit body"
    );
}

#[test]
fn tall_narrow_digits_stay_in_their_half() {
    let mut model = sample(WatchState::Generating);
    model.gen_tps = Some(123.0);
    model.prompt_tps = Some(2345.0);
    let grid = draw(&model, 160, 90);
    let prompt = 121u16;
    assert_eq!(
        grid.get(85, 5).unwrap().ch,
        '█',
        "shrunk GEN digits should still draw"
    );
    assert!(85 < prompt);
    assert_eq!(
        grid.get(118, 5).unwrap().ch,
        ' ',
        "GEN digits crossed into the prompt half"
    );
    assert_eq!(
        grid.get(prompt, 5).unwrap().ch,
        '█',
        "PROMPT digits missing"
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

fn chart_height_of(grid: &llama_watch::tty::grid::Grid) -> u16 {
    let start = req_rule_row(grid) + 1;
    let inn = row_with(grid, "prompt tail");
    inn.saturating_sub(start)
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
    assert_eq!(chart_height_of(&grid), 9);
    let axis = row_with(&grid, "-8m");
    assert!(
        row_string(&grid, axis).contains("now"),
        "axis missing now: {}",
        row_string(&grid, axis)
    );
    let start = req_rule_row(&grid) + 1;
    assert_eq!(axis, start + 4, "axis is the middle of 4+1+4");
    assert!(row_string(&grid, start).contains("250"), "gen ceiling");
    assert!(
        row_string(&grid, start + 8).contains("1500"),
        "prompt ceiling"
    );
    assert!(row_string(&grid, start + 3).contains("gen"), "gen label");
    assert!(
        row_string(&grid, start + 5).contains("prompt"),
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
    assert_eq!(chart_height_of(&grid), 0, "chart must hide for IN/OUT");
    let inn = row_with(&grid, "prompt tail");
    let out = row_with(&grid, "OUT  live");
    assert!(inn > row_with(&grid, "RECENT"));
    assert!(out > inn);
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
    assert_eq!(chart_height_of(&grid), 5, "chart should shrink to 5 first");
    let inn = row_with(&grid, "prompt tail");
    let out = row_with(&grid, "OUT  live");
    let health = grid.rows() - 3;
    let in_rows = out.saturating_sub(inn + 2);
    let out_rows = health.saturating_sub(out + 1);
    assert!(
        in_rows >= 3,
        "IN lost rows before the chart finished shrinking: {in_rows}"
    );
    assert!(
        out_rows >= 1,
        "OUT disappeared while the chart still had room to shrink: {out_rows}"
    );
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
    let axis = row_with(&grid, "-8m");
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
    let axis = row_with(&grid, "now");
    let axis_text = row_string(&grid, axis);
    let now_at = col_of(&grid, axis, "now");
    let age_at = col_of(&grid, axis, "-8m");
    assert_eq!(now_at, 2, "now is the left axis label, got {axis_text}");
    assert!(
        axis_text.ends_with("-8m"),
        "age label is the right axis label, got {axis_text}"
    );
    assert!(now_at < age_at, "now at {now_at}, -8m at {age_at}");

    // "now" (3) plus one blank gutter, then the newest column.
    let data_left = 6u16;
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
        gen_at > newer,
        "gen label at {gen_at} covers the newest column {newer}"
    );
    assert!(
        prompt_at > newer,
        "prompt label at {prompt_at} covers the newest column {newer}"
    );
    let start = req_rule_row(&grid) + 1;
    let tick = col_of(&grid, start, "250");
    let prompt_tick = col_of(&grid, start + 8, "1500");
    assert!(tick > newer, "ceiling covers the newest column");
    assert!(
        prompt_tick > newer,
        "prompt ceiling covers the newest column"
    );

    let wide = draw(&model, 480, 135);
    let wide_row = row_with(&wide, "now");
    let wide_axis = row_string(&wide, wide_row);
    assert_eq!(
        col_of(&wide, wide_row, "now"),
        2,
        "now stays on the left at 480, got {wide_axis}"
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
    let axis = row_with(&grid, "now");
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
    let axis = row_with(&grid, "now");
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
    let axis = row_with(&grid, "-8m");
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
    let axis = row_with(&grid, "now");
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
    // A full band puts the oldest column against the right-hand labels.
    // The newest column stays at the left edge.
    model.chart = vec![
        ChartBucket {
            gen_tps: Some(250.0),
            prompt_tps: Some(100_000.0),
        };
        400
    ];
    let grid = draw(&model, 240, 67);
    let start = req_rule_row(&grid) + 1;
    let bottom = start + 8;
    let band_right = 240u16 - 3;
    let label = "100000";
    let label_start = band_right + 1 - label.len() as u16;
    for (i, ch) in label.chars().enumerate() {
        let col = label_start + i as u16;
        let cell = grid.get(col, bottom).unwrap();
        assert_eq!(cell.ch, ch, "data overwrote ceiling col {col}");
    }
    let is_bar = |col: u16, row: u16| {
        let ch = grid.get(col, row).unwrap().ch;
        ch == '█' || ch == '▀' || ch == '▄'
    };
    let newest = (0..label_start)
        .find(|col| is_bar(*col, bottom))
        .expect("prompt bars");
    let oldest = (0..label_start)
        .rev()
        .find(|col| is_bar(*col, bottom))
        .expect("prompt bars");
    assert_eq!(
        newest, 6,
        "newest column is the left edge after the one-column gutter"
    );
    assert!(
        oldest < label_start,
        "bar at {oldest} overlaps label at {label_start}"
    );
    assert_eq!(
        oldest + 1,
        label_start,
        "oldest column should abut the ceiling label"
    );
    let above = (0..label_start)
        .rev()
        .find(|col| is_bar(*col, bottom - 1))
        .expect("unlabeled prompt row");
    assert_eq!(above, oldest, "ceiling label ate bars on the bottom row");

    let axis = row_with(&grid, "now");
    for (name, row) in [("gen", axis - 1), ("prompt", axis + 1)] {
        let name_start = band_right + 1 - name.len() as u16;
        for (i, ch) in name.chars().enumerate() {
            let col = name_start + i as u16;
            assert_eq!(
                grid.get(col, row).unwrap().ch,
                ch,
                "data overwrote {name} at col {col}"
            );
        }
        let bar = (0..name_start)
            .rev()
            .find(|col| is_bar(*col, row))
            .expect(name);
        assert!(
            bar < name_start,
            "{name} label at {name_start} covers bar at {bar}"
        );
    }
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
    let start = req_rule_row(&first) + 1;
    for row in start..start + 9 {
        assert_eq!(
            row_string(&first, row),
            row_string(&same, row),
            "row {row} changed without a level change"
        );
    }
    model.chart[0].gen_tps = Some(250.0);
    let changed = draw(&model, 240, 67);
    let band_changed =
        (start..start + 9).any(|row| row_string(&first, row) != row_string(&changed, row));
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
    let start = req_rule_row(grid) + 1;
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
    let grid = draw(&model, 240, 67);
    let axis = row_with(&grid, "now");
    let col = (0..grid.cols())
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
    let axis = row_with(&grid, "now");
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
    fix.rows
        .iter()
        .position(|row| {
            let trimmed = row.trim_start();
            trimmed.starts_with(key)
        })
        .unwrap_or_else(|| panic!("missing {key} title"))
}

fn title_line(fix: &Fix, key: &str) -> String {
    line(fix, title_row(fix, key))
}

fn region(fix: &Fix, start_key: &str, end_key: Option<&str>) -> Vec<String> {
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
        fans: None,
        temps: None,
        ctx_history_h: 6,
        setup: match state {
            WatchState::Generating | WatchState::Ready => Some(qwen_setup()),
            WatchState::AiDown | WatchState::Starting | WatchState::NoLlama => None,
        },
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
            let row = (0..grid.rows())
                .find(|row| {
                    let text = row_string(&grid, *row);
                    text.trim_start().starts_with(&format!("{label} "))
                })
                .unwrap_or_else(|| panic!("missing {label} meter"));
            let text = full_row(&grid, row);
            let value = text[..25.min(text.len())].trim_end();
            assert!(value.ends_with("--"), "{label} at {cols}x{rows}: {text}");
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

#[test]
fn activity_bar_sits_under_load_with_percent_source_and_device_watts() {
    let model = sample(WatchState::Generating);
    let grid = draw(&model, 240, 67);
    let load_row = row_with(&grid, "LOAD");
    let activity_row = row_with(&grid, "ACTIVITY");
    assert_eq!(
        activity_row,
        load_row + 1,
        "ACTIVITY is the next row after LOAD"
    );
    let load = row_string(&grid, load_row);
    let activity = row_string(&grid, activity_row);
    assert!(load.contains("70 %"), "{load}");
    assert!(!load.contains("ACTIVITY"), "{load}");
    assert!(
        activity.contains("ACTIVITY 42 % gpu 312 W"),
        "percent, source and the winning device's watts: {activity}"
    );
    let power = row_string(&grid, row_with(&grid, "POWER"));
    assert!(power.contains("312/350 W"), "{power}");
    assert!(!load.contains("gpu"), "LOAD is unchanged: {load}");
}

#[test]
fn activity_label_names_each_source_and_drops_watts_before_the_label() {
    let cases: [(f64, LoadSource, Option<f64>, &str); 5] = [
        (
            78.0,
            LoadSource::Gpu,
            Some(340.0),
            "ACTIVITY  78 % gpu 340 W",
        ),
        (100.0, LoadSource::Cpu, Some(230.0), "ACTIVITY  100 % cpu"),
        (5.0, LoadSource::Cpu, Some(31.0), "ACTIVITY  5 % cpu 31 W"),
        (12.0, LoadSource::Util, None, "ACTIVITY  12 % util"),
        (99.6, LoadSource::Gpu, Some(349.0), "ACTIVITY  100 % gpu"),
    ];
    for (pct, src, watts, want) in cases {
        let mut model = sample(WatchState::Generating);
        model.activity_pct = Some(pct);
        model.activity_src = Some(src);
        model.activity_w = watts;
        let grid = draw(&model, 240, 67);
        let row = row_string(&grid, row_with(&grid, "ACTIVITY"));
        let text: String = row.chars().take(25).collect();
        assert_eq!(
            text.split_whitespace().collect::<Vec<_>>(),
            want.split_whitespace().collect::<Vec<_>>(),
            "{row}"
        );
        assert!(
            row.chars().nth(10) == Some(' '),
            "a blank after the label: {row}"
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
        let row = row_with(&grid, "ACTIVITY");
        let text: String = row_string(&grid, row).chars().take(25).collect();
        let cells: Vec<_> = (25..grid.cols())
            .map(|col| grid.get(col, row).expect("cell"))
            .filter(|cell| matches!(cell.ch, '█' | '░' | '▉'..='▏' | '▄' | '▇'))
            .collect();
        (text, cells)
    };
    let (full_text, full) = bar(100.0);
    let (hot_text, hot) = bar(118.0);
    let (pinned_text, pinned) = bar(125.0);
    assert_eq!(
        hot_text.split_whitespace().collect::<Vec<_>>(),
        ["ACTIVITY", "118", "%", "gpu"],
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
    for (cols, rows, recent, blank) in TEXT_OFF_SIZES {
        let at = format!("{cols}x{rows}");
        let grid = draw(&text_off_model(WatchState::Generating), cols, rows);
        let dump: Vec<String> = (0..rows).map(|row| row_string(&grid, row)).collect();
        let all = dump.join("\n");
        for gone in ["prompt tail", "OUT  live", "LEAKED"] {
            assert!(!all.contains(gone), "{at}: {gone} drawn\n{all}");
        }
        let header = row_with(&grid, "RECENT");
        let req_rule = req_rule_row(&grid);
        // header, `recent` request rows, legend, rule.
        assert_eq!(req_rule, header + recent + 2, "{at}: RECENT rows\n{all}");
        for row in header + 1..=header + recent {
            assert!(
                !content(&grid, row).is_empty(),
                "{at}: RECENT row {row} empty\n{all}"
            );
        }
        let start = req_rule + 1;
        let axis = start + 6;
        assert!(row_string(&grid, axis).contains("now"), "{at}: axis\n{all}");
        assert!(
            row_string(&grid, start).contains("250"),
            "{at}: gen ceiling\n{all}"
        );
        assert!(
            row_string(&grid, start + 12).contains("1500"),
            "{at}: prompt ceiling\n{all}"
        );
        assert!(
            row_string(&grid, axis - 1).contains("gen"),
            "{at}: gen label"
        );
        assert!(
            row_string(&grid, axis + 1).contains("prompt"),
            "{at}: prompt label"
        );
        let health_rule = rows - 3;
        assert!(is_rule_row(&grid, health_rule), "{at}: health rule");
        assert_eq!(health_rule - (start + 13), blank, "{at}: blank rows\n{all}");
        for row in start + 13..health_rule {
            assert!(
                content(&grid, row).is_empty(),
                "{at}: row {row} not blank\n{all}"
            );
        }
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
    let header = row_with(&grid, "RECENT");
    let req_rule = req_rule_row(&grid);
    assert_eq!(req_rule, header + 4 + 2, "RECENT keeps its 4-row floor");
    let health_rule = grid.rows() - 3;
    let chart = health_rule - req_rule - 1;
    assert_eq!(chart, 7, "chart takes what is left, odd");
    assert!(row_string(&grid, req_rule + 4).contains("now"));
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
        let in_title = row_with(&grid, "T49-IN-TITLE");
        let out_title = row_with(&grid, "T49-OUT-TITLE");
        // One spacer row sits above the OUT title.
        let panel: Vec<u16> = (in_title + 1..out_title - 1).collect();
        assert!(panel.len() >= 3, "{cols}x{rows} panel {panel:?}");
        for row in &panel {
            assert!(
                !content(&grid, *row).trim().is_empty(),
                "{cols}x{rows} IN row {row} is blank"
            );
        }
        let last = *panel.last().expect("rows");
        assert_eq!(content(&grid, last), "the last real line", "{cols}x{rows}");
        assert_eq!(grid.get(2, last).unwrap().fg, C16::White);
        let mut labels = 0;
        for row in &panel {
            let line = content(&grid, *row);
            if line.starts_with("-- ") {
                labels += 1;
                assert_eq!(
                    grid.get(2, *row).unwrap().fg,
                    C16::BrightBlack,
                    "{cols}x{rows} label {line:?} is not dim"
                );
            }
        }
        assert!(labels > 0, "{cols}x{rows} no label in view");
    }
}

// ---- T52: FANS panel ------------------------------------------------------

use llama_watch::sources::fans::{FanPanel, FanReading};
use llama_watch::tty::layout::FANS_SIDE_COLS;

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
fn wide_screen_puts_fans_right_of_in_out() {
    for (_, cols, rows) in FANS_GOLDENS {
        let at = format!("{cols}x{rows}");
        let model = fans_model();
        let grid = draw(&model, cols, rows);
        let header = row_with(&grid, "FANS  nct6798");
        let in_row = row_with(&grid, "IN ");
        assert_eq!(header, in_row, "{at}: FANS header shares the IN title row");
        let split = u16::try_from(u32::from(cols) * 58 / 100).unwrap();
        assert_eq!(col_of(&grid, header, "FANS"), split + 2, "{at}");
        let health_rule = rows - 3;
        for row in in_row..health_rule {
            assert_eq!(grid.get(split, row).unwrap().ch, '|', "{at} row {row}");
            // IN/OUT text stops two columns before the divider.
            for col in split - 1..split {
                assert_eq!(grid.get(col, row).unwrap().ch, ' ', "{at} r{row} c{col}");
            }
        }
        let expect = [
            ("front1", "1939 rpm", "88%"),
            ("front2", "1877 rpm", "88%"),
            ("rear", "3026 rpm", "64%"),
            ("top", "1272 rpm", "100%"),
        ];
        for (i, (label, rpm, pct)) in expect.iter().enumerate() {
            let row = header + 1 + u16::try_from(i).unwrap();
            let text: String = row_string(&grid, row)
                .chars()
                .skip(usize::from(split) + 2)
                .collect();
            assert!(text.starts_with(label), "{at}: {text}");
            for part in [*rpm, *pct, "auto"] {
                assert!(text.contains(part), "{at}: {part} missing in {text}");
            }
            assert!(
                text.contains('░') || *pct == "100%",
                "{at}: no meter in {text}"
            );
        }
        assert_blank_edges(&grid);
    }
}

#[test]
fn in_out_wrap_inside_the_left_part_when_fans_are_beside_them() {
    let mut model = fans_model();
    model.in_lines = vec!["I".repeat(400)];
    model.out_lines = vec!["O".repeat(400)];
    let grid = draw(&model, 240, 67);
    let split = 240 * 58 / 100;
    let mut seen = 0;
    for row in row_with(&grid, "IN ")..64 {
        for col in split - 1..239 {
            let ch = grid.get(col, row).unwrap().ch;
            assert!(
                ch != 'I' && ch != 'O',
                "text crossed into FANS at r{row} c{col}"
            );
        }
        seen += row_string(&grid, row).matches('I').count();
    }
    assert!(seen > 0, "IN text drawn");
}

/// #95: fan rows stack with no blank row between, so the meter keeps #52's
/// whole-cell glyph (`▇`/`▄`) instead of the half-cell `█`/`▐`/`▌`.
fn meter_cells(grid: &llama_watch::tty::grid::Grid, row: u16) -> Vec<C16> {
    (0..grid.cols())
        .filter_map(|col| {
            let cell = grid.get(col, row).unwrap();
            is_lit_meter_cell(cell.ch).then_some(cell.fg)
        })
        .collect()
}

#[test]
fn fan_meter_uses_the_step_palette() {
    let grid = draw(&fans_model(), 240, 67);
    let header = row_with(&grid, "FANS  nct6798");
    let top = meter_cells(&grid, header + 4);
    assert!(top.len() > 20, "{top:?}");
    let steps = [
        C16::Blue,
        C16::BrightBlue,
        C16::Magenta,
        C16::BrightMagenta,
        C16::BrightRed,
    ];
    assert!(top.iter().all(|fg| steps.contains(fg)), "{top:?}");
    assert_eq!(top.first(), Some(&C16::Blue));
    assert_eq!(
        top.last(),
        Some(&C16::BrightRed),
        "a full meter ends on the last step"
    );
}

#[test]
fn stalled_fan_is_dim_red_and_others_are_not() {
    let mut model = fans_model();
    let panel = model.fans.as_mut().unwrap();
    panel.fans[2].rpm = Some(0); // rear: pwm 162 but not turning
    panel.fans[3].rpm = Some(0); // top: 0 rpm and pwm 0 is a fan at rest
    panel.fans[3].pwm = Some(0);
    let grid = draw(&model, 240, 67);
    let header = row_with(&grid, "FANS  nct6798");
    let label_fg = |row: u16| {
        let col = u16::try_from(240u32 * 58 / 100 + 2).unwrap();
        grid.get(col, row).unwrap().fg
    };
    assert_eq!(label_fg(header + 3), C16::Red, "stalled label");
    let stalled = meter_cells(&grid, header + 3);
    assert!(!stalled.is_empty());
    assert!(stalled.iter().all(|fg| *fg == C16::Red), "{stalled:?}");
    let rpm_row = row_string(&grid, header + 3);
    let zero = col_of(&grid, header + 3, "0 rpm");
    assert_eq!(
        grid.get(zero, header + 3).unwrap().fg,
        C16::Red,
        "{rpm_row}"
    );
    assert_eq!(label_fg(header + 1), C16::White, "turning fan");
    assert_eq!(
        label_fg(header + 4),
        C16::White,
        "0 rpm at 0 pwm is not a stall"
    );
    assert!(
        meter_cells(&grid, header + 1)
            .iter()
            .all(|fg| *fg != C16::Red),
        "healthy meter"
    );
}

#[test]
fn missing_values_draw_dashes_and_absent_chip_says_so() {
    let mut model = fans_model();
    model.fans.as_mut().unwrap().fans[1] = fan(3, "front2", None, None, None);
    let grid = draw(&model, 240, 67);
    let header = row_with(&grid, "FANS  nct6798");
    let text = row_string(&grid, header + 2);
    assert!(text.contains("-- rpm"), "{text}");
    assert!(text.ends_with("--  --"), "{text}");

    model.fans = Some(FanPanel {
        chip: "nct6798".to_string(),
        present: false,
        fans: Vec::new(),
    });
    let grid = draw(&model, 240, 67);
    let header = row_with(&grid, "FANS  nct6798");
    assert!(row_string(&grid, header + 1).contains("no single hwmon named nct6798"));
}

#[test]
fn narrow_screen_puts_fans_under_in_out_when_rows_allow() {
    let model = fans_model();
    for (cols, rows) in [(160u16, 67u16), (FANS_SIDE_COLS - 1, 80)] {
        let at = format!("{cols}x{rows}");
        let grid = draw(&model, cols, rows);
        let all: String = (0..rows).map(|row| row_string(&grid, row) + "\n").collect();
        let header = row_with(&grid, "FANS  nct6798");
        let out = row_with(&grid, "OUT ");
        assert!(header > out + 3, "{at}: OUT keeps 3 rows\n{all}");
        assert!(
            is_rule_row(&grid, header - 1),
            "{at}: rule above FANS\n{all}"
        );
        assert_eq!(
            header + 5,
            rows - 3,
            "{at}: FANS ends on the health rule\n{all}"
        );
        assert_eq!(col_of(&grid, header, "FANS"), 2, "{at}");
        assert!(row_string(&grid, header + 4).contains("top"), "{at}\n{all}");
        let in_row = row_with(&grid, "IN ");
        assert!(
            !row_string(&grid, in_row).contains("FANS"),
            "{at}: not beside IN\n{all}"
        );
        assert_blank_edges(&grid);
    }
    // 160x48 has no rows to spare under IN/OUT: FANS is hidden and IN/OUT
    // draw as without fans.
    let grid = draw(&model, 160, 48);
    let mut off = model.clone();
    off.fans = None;
    let plain = draw(&off, 160, 48);
    for row in 0..48 {
        assert_eq!(row_string(&grid, row), row_string(&plain, row), "row {row}");
    }
}

#[test]
fn text_off_gives_fans_the_bottom_of_the_freed_rows() {
    for (cols, rows, recent, _) in TEXT_OFF_SIZES {
        let at = format!("{cols}x{rows}");
        let mut model = text_off_model(WatchState::Generating);
        model.fans = Some(ref_fans());
        let grid = draw(&model, cols, rows);
        let all: String = (0..rows).map(|row| row_string(&grid, row) + "\n").collect();
        let header = row_with(&grid, "FANS  nct6798");
        let health_rule = rows - 3;
        assert_eq!(
            header + 5,
            health_rule,
            "{at}: block ends on the health rule\n{all}"
        );
        assert!(
            is_rule_row(&grid, header - 1),
            "{at}: rule above FANS\n{all}"
        );
        let (left, _) = if cols >= FANS_SIDE_COLS {
            (u16::try_from(u32::from(cols) * 58 / 100 + 2).unwrap(), 0)
        } else {
            (2, 0)
        };
        assert_eq!(col_of(&grid, header, "FANS"), left, "{at}");
        // The chart keeps its rows; RECENT gives up only what FANS needs.
        let rec_header = row_with(&grid, "RECENT");
        let req_rule = req_rule_row(&grid);
        let shown = req_rule - rec_header - 2;
        assert!(shown >= 4, "{at}: RECENT floor\n{all}");
        assert!(shown <= recent, "{at}");
        assert!(
            row_string(&grid, req_rule + 7).contains("now"),
            "{at}: chart axis\n{all}"
        );
        for gone in ["LEAKED", "prompt tail"] {
            assert!(!all.contains(gone), "{at}");
        }
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
        .take_while(|cell| matches!(cell.ch, '█' | '▓' | '▐' | '▌' | '▄' | '▇'))
        .map(|cell| cell.fg)
        .collect()
}

/// Row of the left meter labelled `label` (the label starts at column 2).
fn meter_row(grid: &llama_watch::tty::grid::Grid, label: &str) -> u16 {
    (0..grid.rows())
        .find(|row| {
            let text = row_string(grid, *row);
            text.get(2..)
                .is_some_and(|rest| rest.starts_with(label) && rest[label.len()..].starts_with(' '))
        })
        .unwrap_or_else(|| panic!("missing meter {label}"))
}

/// Top row (and bottom row at 4K) of the CPU meter at `pct`.
fn cpu_bar(pct: f64, cols: u16, rows: u16) -> (Vec<C16>, Vec<C16>, usize) {
    let mut model = sample(WatchState::Generating);
    model.cpu_pct = Some(pct);
    let grid = draw(&model, cols, rows);
    let row = meter_row(&grid, "CPU");
    let track = (27..grid.cols())
        .take_while(|c| {
            matches!(
                grid.get(*c, row).expect("cell").ch,
                '█' | '▐' | '▌' | '░' | '▄' | '▇'
            )
        })
        .count();
    (
        bar_from(&grid, row, 27),
        bar_from(&grid, row + 1, 27),
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
        if rows >= 135 {
            assert_eq!(steps_of(&bottom), STEPS, "{cols}x{rows} {bottom:?}");
        }
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

/// T63: the spectrum applies to every level meter; VRAM and MEM are capacity
/// and keep one colour for the whole bar.
#[test]
fn level_meters_are_spectra_and_capacity_meters_are_one_colour() {
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
    for label in ["CPU", "GPU", "POWER", "LOAD", "ACTIVITY"] {
        let bar = bar_from(&grid, meter_row(&grid, label), 27);
        assert_eq!(steps_of(&bar), STEPS, "{label} {bar:?}");
    }
    for label in ["VRAM", "MEM"] {
        let bar = bar_from(&grid, meter_row(&grid, label), 27);
        assert_eq!(steps_of(&bar), [C16::BrightWhite], "{label} {bar:?}");
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
            .find(|row| row_string(&grid, *row).trim_start() == "IN")
            .expect("IN title");
        let notes: Vec<u16> = (in_row..grid.rows())
            .filter(|row| row_string(&grid, *row).contains("text needs llama.cpp /slots"))
            .collect();
        assert_eq!(notes.len(), 2, "IN and OUT each say why: {cols}");
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

// #7: short screens. Below 48 rows the dashboard drops panels in order
// (IN/OUT, then FANS, then chart height) and refuses only below 160x26.

/// Rows of the chart under RECENT, from its gen ceiling label to its prompt
/// ceiling label. 0 when there is no chart.
fn chart_rows_below_recent(grid: &llama_watch::tty::grid::Grid) -> u16 {
    let from = req_rule_row(grid) + 1;
    let top = (from..grid.rows()).find(|row| row_string(grid, *row).ends_with(" 250"));
    let bottom = (from..grid.rows()).find(|row| row_string(grid, *row).ends_with(" 1500"));
    match (top, bottom) {
        (Some(top), Some(bottom)) if bottom > top => bottom - top + 1,
        _ => 0,
    }
}

fn recent_rows_shown(grid: &llama_watch::tty::grid::Grid) -> u16 {
    let header = row_with(grid, "RECENT");
    req_rule_row(grid) - header - 2
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
    assert!(
        row_with(grid, "ACTIVITY") < row_with(grid, "RECENT"),
        "{at}"
    );
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
    assert_eq!(recent_rows_shown(&grid), 4, "{all}");
    assert_eq!(chart_rows_below_recent(&grid), 9, "{all}");
    let inn = row_with(&grid, "prompt tail");
    let out = row_with(&grid, "OUT  live");
    assert!(out >= inn + 4, "IN keeps 3 rows\n{all}");
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
    for gone in ["prompt tail", "OUT  live", "prompt body", "output body"] {
        assert!(!all.contains(gone), "IN/OUT still drawn: {gone}\n{all}");
    }
    assert!(recent_rows_shown(&grid) >= 4, "{all}");
    assert_eq!(chart_rows_below_recent(&grid), 13, "{all}");
}

#[test]
fn short_screens_drop_fans_after_in_out_then_shrink_the_chart() {
    let mut model = short_model();
    model.fans = Some(ref_fans());
    // 160x45: IN/OUT gone, FANS kept at the bottom, chart full.
    let grid = draw(&model, 160, 45);
    assert_dashboard(&grid, "160x45 fans");
    let all = whole(&grid);
    assert!(!all.contains("prompt tail"), "IN/OUT go before FANS\n{all}");
    let fans = row_with(&grid, "FANS  nct6798");
    assert_eq!(fans + 5, 45 - 3, "FANS ends on the health rule\n{all}");
    assert_eq!(chart_rows_below_recent(&grid), 13, "{all}");
    // 160x40: no room for FANS under a full chart: FANS goes, chart stays.
    let grid = draw(&model, 160, 40);
    assert_dashboard(&grid, "160x40 fans");
    let all = whole(&grid);
    assert!(!all.contains("FANS"), "FANS kept over the chart\n{all}");
    assert_eq!(chart_rows_below_recent(&grid), 13, "{all}");
    // Then the chart shrinks, odd heights down to 5, then hides; RECENT
    // keeps its four rows.
    let mut last = 13;
    for rows in (26..40).rev() {
        let grid = draw(&model, 160, rows);
        let at = format!("160x{rows}");
        assert_dashboard(&grid, &at);
        assert!(recent_rows_shown(&grid) >= 4, "{at}\n{}", whole(&grid));
        let h = chart_rows_below_recent(&grid);
        assert!(h <= last, "{at}: chart grew");
        assert!(h == 0 || (h >= 5 && h % 2 == 1), "{at}: chart {h}");
        last = h;
    }
    assert_eq!(last, 0, "the chart hides at the floor");
}

#[test]
fn floor_is_160x26_and_only_smaller_is_too_small() {
    let grid = draw(&short_model(), 160, 26);
    assert_dashboard(&grid, "160x26");
    assert_eq!(recent_rows_shown(&grid), 4, "{}", whole(&grid));
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
        assert!(recent_rows_shown(&grid) >= 4, "{at}\n{all}");
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
    let title = row_with(&grid, "IN (last request · 3 tool results)");
    let out = (title + 1..grid.rows())
        .find(|row| row_string(&grid, *row).trim_start().starts_with("OUT"))
        .expect("OUT title");
    let rows: Vec<String> = (title..out).map(|row| row_string(&grid, row)).collect();
    let last = rows
        .iter()
        .rev()
        .find(|row| !row.trim().is_empty() && !row.trim().chars().all(|ch| ch == '-'))
        .expect("IN text");
    assert!(last.contains("[tool] INVENTED-NEWEST result"), "{rows:#?}");
    assert!(
        rows.iter()
            .any(|row| row.contains("[tool] invented middle result")),
        "{rows:#?}"
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
    let rows = left_rows(&grid, 10, 7);
    assert!(
        rows[1].starts_with("  SETUP  flash-next · Flash Next Q4_K_M"),
        "{rows:#?}"
    );
    assert_eq!(
        rows[2..],
        [
            "    engine   Strata 0.1.41 · Q4_K_M",
            "    ctx      262,144 · kv q8 · resident 24,576",
            "    experts  cache 14.6 GiB · 7,200 slots · hit 87 % · pcie 9 %",
            "    spec     depth 5 · mtp 3 · lookup 2 · min-p 0.40 · 70 %",
            "    serve    pcie 0.60 · arena 40.5 GiB · 12 workers · conv cache off",
        ]
    );
    let all = whole(&grid);
    assert!(
        all.contains("drafting a reply: outline · gen 40/4,096 · 31.5/s"),
        "{all}"
    );
}

/// #90: SETUP's title (`draw_setup_title`, through `paint_detail_fg`) runs
/// the model id/name through the same per-character mapping IN/OUT and
/// RECENT use (`tty::sanitize::detail_char`): a curly quote and an em
/// dash are real glyphs and pass through as themselves, an accented letter
/// becomes its base ASCII letter, an unmapped CJK character becomes the
/// placeholder rather than `?`, and a `?` the text actually had stays `?`
/// right next to it. Before this the title's own filter turned every one
/// of those but the quote into a literal `?`.
#[test]
fn setup_title_maps_typography_like_in_and_out() {
    let mut model = setup_model();
    let mut setup = model.setup.clone().expect("setup");
    setup.id = "qwen\u{2019}s \u{2014} caf\u{e9} \u{6027}?".to_owned();
    setup.name = String::new();
    setup.more = 0;
    model.setup = Some(setup);
    let grid = draw(&model, 160, 49);
    let rows = left_rows(&grid, 10, 2);
    assert_eq!(rows[1], "  SETUP  qwen\u{2019}s \u{2014} cafe \u{fffd}?");
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
    let rows = left_rows(&grid, 10, 2);
    assert_eq!(rows[1], "  SETUP  abc\u{fffd}de");
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
    let rows = left_rows(&grid, 10, 2);
    assert_eq!(rows[1], "  SETUP  ab \u{b7} NAME");
}

/// Rows from the label column of `grid`, `count` from `top`, trimmed.
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

/// #52: one meter per row, ACTIVITY right under LOAD, a blank row, then
/// SETUP: the title with `+1`, and as many rows as fit (six at 160x49).
#[test]
fn setup_block_sits_under_single_spaced_meters_at_160x49() {
    let grid = draw(&setup_model(), 160, 49);
    let labels: Vec<u16> = ["CPU", "GPU", "VRAM", "MEM", "POWER", "LOAD", "ACTIVITY"]
        .into_iter()
        .map(|label| meter_row(&grid, label))
        .collect();
    assert_eq!(labels, [3, 4, 5, 6, 7, 8, 9], "single-spaced meters");
    let rows = left_rows(&grid, 10, 7);
    assert_eq!(rows[0], "", "a blank row under ACTIVITY");
    assert!(
        rows[1].starts_with("  SETUP  qwen3.6-35b-a3b · Qwen 35B"),
        "{rows:#?}"
    );
    assert!(rows[1].ends_with("+1"), "{rows:#?}");
    assert_eq!(
        rows[2..],
        [
            "    engine   llama.cpp · UD-Q4_K_M · fa on",
            "    ctx      262,144 · kv q8_0 / q8_0",
            "    experts  16 layers in RAM",
            "    spec     draft-mtp · n-max 3",
            "    think    budget 24,000",
        ]
    );
    // `sample` does not fit under 60 rows: the rule follows the block.
    assert!(is_rule_row(&grid, 17), "{}", whole(&grid));
    assert!(!whole(&grid).contains("temp 0.6"));
    // The title's `+1` ends where the meter bars end; colours.
    let title = 11;
    let plus = col_of(&grid, title, "+1");
    assert_eq!(plus + 1, 160 / 2 - 3);
    assert_eq!(grid.get(plus, title).unwrap().fg, C16::BrightYellow);
    assert_eq!(grid.get(2, title).unwrap().fg, C16::White);
    assert_eq!(grid.get(9, title).unwrap().fg, C16::BrightWhite);
    let name = col_of(&grid, title, "Qwen 35B");
    assert_eq!(grid.get(name, title).unwrap().fg, C16::BrightBlack);
    let ctx = 13;
    assert_eq!(grid.get(4, ctx).unwrap().fg, C16::BrightBlack, "row label");
    assert_eq!(grid.get(13, ctx).unwrap().fg, C16::BrightWhite, "value");
}

/// #52: the meter bars are whole cells of `▇` with a llama-hack font
/// (eighths) and `▄` with eurlatgr (halves), so stacked bars keep a gap;
/// the colours are the spectrum as before. Tall screens keep 2-row bars.
#[test]
fn single_spaced_meters_use_a_short_block_in_whole_cells() {
    for (glyphs, glyph) in [(ChartGlyphs::Eighths, '▇'), (ChartGlyphs::Halves, '▄')] {
        let mut model = setup_model();
        model.chart_glyphs = glyphs;
        model.cpu_pct = Some(50.0);
        let grid = draw(&model, 160, 49);
        let row = meter_row(&grid, "CPU");
        let cells: Vec<char> = (27..78).map(|c| grid.get(c, row).unwrap().ch).collect();
        let lit = cells.iter().filter(|ch| **ch == glyph).count();
        assert_eq!(lit, 26, "{glyphs:?}: round(0.5 × 51) whole cells");
        assert!(
            cells.iter().all(|ch| *ch == glyph || *ch == '░'),
            "{glyphs:?}: no half-cell ends {cells:?}"
        );
        assert_eq!(cells[0], glyph, "lit from the first cell");
        assert_eq!(grid.get(27, row).unwrap().fg, C16::Blue);
        assert!(GLYPHS.contains(&glyph), "term may emit {glyph}");
    }
    // A tiny value still lights one cell.
    let mut model = setup_model();
    model.gpu_pct = Some(1.0);
    let grid = draw(&model, 160, 49);
    assert_eq!(grid.get(27, meter_row(&grid, "GPU")).unwrap().ch, '▇');
    // 4K keeps its two-row bars and blank rows.
    let grid = draw(&setup_model(), 480, 135);
    let cpu = meter_row(&grid, "CPU");
    assert_eq!(meter_row(&grid, "GPU"), cpu + 3);
    assert_eq!(grid.get(28, cpu).unwrap().ch, '█');
    assert_eq!(grid.get(28, cpu + 1).unwrap().ch, '▓');
}

/// #52: a wide screen from 60 rows shows up to eight SETUP rows; a short
/// one keeps the 160x26 floor (title and three rows beside SLOTS); with
/// nothing loaded the block is gone and RECENT moves back up.
#[test]
fn setup_block_height_follows_the_screen() {
    let grid = draw(&setup_model(), 240, 67);
    let all = whole(&grid);
    assert!(
        all.contains("    sample   temp 0.6 · top-p 0.95 · top-k 20 · min-p 0"),
        "{all}"
    );
    let grid = draw(&setup_model(), 160, 26);
    assert_dashboard(&grid, "160x26 setup");
    assert_eq!(recent_rows_shown(&grid), 4, "{}", whole(&grid));
    let rows = left_rows(&grid, 11, 4);
    assert!(rows[0].contains("SETUP"), "{rows:#?}");
    assert!(rows[3].contains("experts"), "{rows:#?}");
    assert!(is_rule_row(&grid, 15), "{}", whole(&grid));
    let mut idle = setup_model();
    idle.setup = None;
    let grid = draw(&idle, 160, 49);
    assert!(!whole(&grid).contains("SETUP"));
    assert!(is_rule_row(&grid, 15), "{}", whole(&grid));
    // Long values drop whole items from the end; a lone long one is cut.
    let mut long = setup_model();
    let setup = long.setup.as_mut().expect("setup");
    setup.id = "a-very-long-llama-swap-model-id-for-a-bakeoff-case-with-suffix-r2".to_owned();
    setup.rows[1].items[0].text = "x".repeat(90);
    let grid = draw(&long, 160, 49);
    let rows = left_rows(&grid, 11, 3);
    assert!(
        rows[0].contains("a-very-long-llama-swap-model-id-for-a-bakeoff-case-with-suffix-r2"),
        "{rows:#?}"
    );
    assert!(
        !rows[0].contains("Qwen 35B"),
        "the name only when it fits: {rows:#?}"
    );
    assert!(
        rows[2].ends_with('…') && rows[2].chars().count() == 78,
        "{rows:#?}"
    );
    assert!(!rows[2].contains("kv"), "{rows:#?}");
    assert_blank_edges(&grid);
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
    TempItem {
        label: label.to_string(),
        tenths: c * 10,
        level,
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
            group("coolant", vec![temp_item("", 40, Normal)]),
            group(
                "NVMe0",
                vec![
                    temp_item("", 50, Normal),
                    temp_item("s1", 50, Normal),
                    temp_item("s2", 82, Warn),
                ],
            ),
            group(
                "NVMe1",
                vec![
                    temp_item("", 57, Normal),
                    temp_item("s1", 57, Normal),
                    temp_item("s2", 66, Normal),
                ],
            ),
            group(
                "board",
                vec![
                    temp_item("sys", 51, Normal),
                    temp_item("cpu", 51, Normal),
                    temp_item("aux0", 27, Normal),
                    temp_item("aux1", 92, Crit),
                    temp_item("aux2", 16, Normal),
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
fn wide_screens_stack_fans_then_temps_right_of_in_out() {
    for (cols, rows) in [(192u16, 60u16), (240, 67), (286, 60)] {
        let at = format!("{cols}x{rows}");
        let grid = draw(&temps_model(), cols, rows);
        let all: String = (0..rows).map(|row| row_string(&grid, row) + "\n").collect();
        let fans = row_with(&grid, "FANS  nct6798");
        let temps = row_with(&grid, "TEMPS  warn crit");
        assert_eq!(fans, row_with(&grid, "IN "), "{at}\n{all}");
        assert_eq!(temps, fans + 6, "{at}: FANS, a blank row, TEMPS\n{all}");
        let split = u16::try_from(u32::from(cols) * 58 / 100).unwrap();
        assert_eq!(col_of(&grid, temps, "TEMPS"), split + 2, "{at}");
        for (offset, name) in [
            "CPU", "GPU", "coolant", "NVMe0", "NVMe1", "board", "NIC", "Wi-Fi",
        ]
        .iter()
        .enumerate()
        {
            let row = temps + 1 + u16::try_from(offset).unwrap();
            assert_eq!(col_of(&grid, row, name), split + 2, "{at}: {name}\n{all}");
        }
        assert!(
            row_string(&grid, temps + 6).contains("aux4 50"),
            "{at}: board whole\n{all}"
        );
        assert_blank_edges(&grid);
    }
}

#[test]
fn temps_values_take_their_level_colours() {
    let grid = draw(&temps_model(), 192, 60);
    let temps = row_with(&grid, "TEMPS  warn crit");
    let fg_at = |row: u16, needle: &str| {
        let col = col_of(&grid, row, needle);
        grid.get(col, row).expect("cell").fg
    };
    assert_eq!(fg_at(temps, "warn"), C16::Yellow);
    assert_eq!(fg_at(temps, "crit"), C16::BrightRed);
    let nvme = temps + 4;
    assert_eq!(fg_at(nvme, "82"), C16::Yellow, "NVMe0 s2 is warm");
    assert_eq!(fg_at(nvme, "s1"), C16::BrightBlack, "labels are dim");
    assert_eq!(fg_at(nvme, "50"), C16::BrightWhite);
    assert_eq!(fg_at(temps + 6, "92"), C16::BrightRed, "board aux1 is hot");
}

#[test]
fn at_160x49_temps_is_one_line_above_the_health_rule_and_fans_hide() {
    let grid = draw(&temps_model(), 160, 49);
    let all: String = (0..49).map(|row| row_string(&grid, row) + "\n").collect();
    let line = row_with(&grid, "TEMPS");
    assert_eq!(line, 49 - 4, "{all}");
    assert_eq!(
        row_string(&grid, line).trim(),
        "TEMPS  CPU 68 \u{b7} GPU 71 \u{b7} coolant 40 \u{b7} NVMe0 82 \u{b7} NVMe1 66 \u{b7} board 92 \u{b7} NIC 51 \u{b7} Wi-Fi 39"
    );
    assert!(!all.contains("FANS"), "{all}");
    // OUT keeps its three rows.
    let out = row_with(&grid, "OUT ");
    assert!(line >= out + 4, "{all}");
    assert_blank_edges(&grid);
}

#[test]
fn narrow_tall_screens_put_fans_and_temps_side_by_side_under_in_out() {
    let grid = draw(&temps_model(), 160, 67);
    let all: String = (0..67).map(|row| row_string(&grid, row) + "\n").collect();
    let fans = row_with(&grid, "FANS  nct6798");
    assert_eq!(fans, row_with(&grid, "TEMPS  warn crit"), "{all}");
    assert!(is_rule_row(&grid, fans - 1), "{all}");
    assert_eq!(
        fans + 9,
        67 - 3,
        "TEMPS (8 rows) ends on the health rule\n{all}"
    );
    assert_eq!(col_of(&grid, fans, "FANS"), 2);
    assert_eq!(col_of(&grid, fans, "TEMPS"), 90, "{all}");
    assert_blank_edges(&grid);
}

#[test]
fn text_off_short_of_rows_cuts_temps_to_the_fans_height_with_a_summary() {
    let mut model = temps_model();
    model.show_text = false;
    let grid = draw(&model, 160, 49);
    let all: String = (0..49).map(|row| row_string(&grid, row) + "\n").collect();
    let header = row_with(&grid, "TEMPS  warn crit");
    assert_eq!(header, row_with(&grid, "FANS  nct6798"), "{all}");
    assert_eq!(header + 5, 49 - 3, "{all}");
    let last = row_string(&grid, header + 4);
    assert!(
        last.contains("+ NVMe0 82 \u{b7} NVMe1 66 \u{b7} board 92 \u{b7} NIC 51 \u{b7} Wi-Fi 39"),
        "{last}"
    );
    assert_blank_edges(&grid);
}

#[test]
fn temps_alone_and_temps_off() {
    let mut model = temps_model();
    model.fans = None;
    let grid = draw(&model, 192, 60);
    assert_eq!(row_with(&grid, "TEMPS  warn crit"), row_with(&grid, "IN "));
    model.temps = Some(TempPanel::default());
    let grid = draw(&model, 192, 60);
    let header = row_with(&grid, "TEMPS");
    assert!(row_string(&grid, header + 1).contains("no temperature inputs found"));
    model.temps = None;
    for (cols, rows) in [(160u16, 49u16), (192, 60), (240, 67)] {
        let grid = draw(&model, cols, rows);
        let all: String = (0..rows).map(|row| row_string(&grid, row) + "\n").collect();
        assert!(!all.contains("TEMPS"), "{cols}x{rows}");
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
    assert!(row_string(&grid, legend_row(&grid)).contains("~bar = vs largest row"));
}

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
    let legend = row_string(&grid, legend_row(&grid));
    assert!(
        !legend.contains('\u{2587}') && !legend.contains("cached"),
        "the colour key moved to the header (#96): {legend}"
    );
    // 160 columns with every note: items drop, nothing passes the edge.
    model.requests[0].gen_measured = true;
    model.requests[0].n_ctx = None;
    let grid = draw(&model, 160, 48);
    let legend = row_string(&grid, legend_row(&grid));
    assert!(legend.trim_end().chars().count() <= 158, "{legend}");
    assert!(legend.contains("GEN"), "the base legend stays: {legend}");
    // A wider screen keeps room for both notes.
    let grid = draw(&model, 240, 67);
    let legend = row_string(&grid, legend_row(&grid));
    assert!(legend.contains("~bar = vs largest row"), "{legend}");
    assert!(legend.contains("~ = engine-measured"), "{legend}");
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
    let flying = (header + 1..header + 9)
        .filter(|r| row_string(&grid, *r).starts_with('>'))
        .count();
    assert_eq!(flying, 4);
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

                let fans_header = row_with(&grid, "FANS  nct6798");
                let fans_count = u16::try_from(model.fans.as_ref().unwrap().fans.len()).unwrap();
                assert_no_hairline(
                    &grid,
                    fans_header + 1..fans_header + 1 + fans_count,
                    &format!("{at} FANS"),
                );

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
