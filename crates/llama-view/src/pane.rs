//! Living in a tmux pane (#27): fitting tty11 into a pane of any size, the
//! frame rate while unfocused, keyboard and focus input, and the pane title.
//!
//! Signals: rustix has no way to install a signal handler and this crate
//! forbids `unsafe`, so llama-view installs none. SIGWINCH keeps its default
//! (ignored) and the window size is read every frame instead. Ctrl-C does not
//! raise SIGINT: input runs with `ISIG` off and the byte quits cleanly, as do
//! `q` and Ctrl-\. SIGTERM and SIGHUP keep their default action (exit
//! without a restore); after SIGHUP the terminal is gone anyway, and after a
//! SIGTERM the next llama-view run (or `reset`) puts the terminal back: its
//! restore re-enables canonical input and echo whatever state it started in.

use std::time::Duration;

use crate::screen::{Cell, Color, Screen};

/// `--fit`: what a pane smaller than the console shows.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Fit {
    /// The top-left corner (header, meters, RECENT), panned by `--offset-x/-y`.
    #[default]
    Crop,
    /// The middle of the console; a larger pane shows it centred.
    Center,
}

impl Fit {
    /// `crop` or `center`.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "crop" => Some(Self::Crop),
            "center" => Some(Self::Center),
            _ => None,
        }
    }
}

/// The pane's view of `screen`: `cols` x `rows` cells.
///
/// When the pane is smaller than the console on either axis, the last row is
/// a one-line hint with both sizes instead of console cells, so a cut-off
/// dashboard says so rather than looking garbled. `vt` is the console number
/// the hint names.
pub fn place(
    screen: &Screen,
    cols: u16,
    rows: u16,
    offset_x: u16,
    offset_y: u16,
    fit: Fit,
    vt: u8,
) -> Screen {
    let smaller = cols < screen.cols || rows < screen.rows;
    let body_rows = if smaller {
        rows.saturating_sub(1)
    } else {
        rows
    };
    let (base_x, base_y) = match fit {
        Fit::Crop => (0i32, 0i32),
        Fit::Center => (
            (i32::from(screen.cols) - i32::from(cols)) / 2,
            (i32::from(screen.rows) - i32::from(body_rows)) / 2,
        ),
    };
    let origin_x = base_x + i32::from(offset_x);
    let origin_y = base_y + i32::from(offset_y);
    let width = usize::from(screen.cols);
    let mut cells = Vec::with_capacity(usize::from(cols).saturating_mul(usize::from(rows)));
    for row in 0..body_rows {
        for col in 0..cols {
            let src_row = origin_y + i32::from(row);
            let src_col = origin_x + i32::from(col);
            let inside = src_row >= 0
                && src_col >= 0
                && src_row < i32::from(screen.rows)
                && src_col < i32::from(screen.cols);
            let cell = if inside {
                let index = src_row as usize * width + src_col as usize;
                screen.cells.get(index).copied().unwrap_or_else(Cell::blank)
            } else {
                Cell::blank()
            };
            cells.push(cell);
        }
    }
    if smaller && rows > 0 {
        let shown = match fit {
            Fit::Crop => "top-left shown, --fit center for the middle",
            Fit::Center => "middle shown, --fit crop for the top-left",
        };
        let text = format!(
            " tty{vt} is {}x{}, this pane {cols}x{rows}: {shown} ",
            screen.cols, screen.rows
        );
        let mut chars = text.chars();
        for _ in 0..cols {
            let ch = chars.next().unwrap_or(' ');
            cells.push(Cell::new(ch, Color::Black, Color::White));
        }
    }
    Screen {
        cols,
        rows,
        cursor_x: 0,
        cursor_y: 0,
        cells,
    }
}

/// Time between frames: `1 / fps` while focused (or when focus is unknown),
/// at most one frame a second while the pane is unfocused or the client is
/// detached. `--fps` stays the ceiling either way.
pub fn frame_period(fps: u8, focused: bool) -> Duration {
    let fps = if focused { fps.max(1) } else { 1 };
    Duration::from_nanos(1_000_000_000 / u64::from(fps))
}

/// What the keyboard and the terminal said.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputEvent {
    /// `ESC [ I` (focus reporting, `ESC [ ? 1004 h`).
    FocusIn,
    /// `ESC [ O`.
    FocusOut,
    /// `q`, `Q`, Ctrl-C or Ctrl-\: restore the terminal and exit.
    Quit,
}

/// Turns stdin bytes into [`InputEvent`]s. Escape sequences may arrive split
/// across reads; anything else (arrow keys, other CSI) is ignored.
#[derive(Clone, Debug, Default)]
pub struct InputParser {
    state: ParseState,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum ParseState {
    #[default]
    Ground,
    Esc,
    /// Inside `ESC [`; `true` once a parameter or intermediate byte was seen,
    /// so `ESC [ 1 ; 5 I` is not a focus report.
    Csi(bool),
}

impl InputParser {
    /// Feed `bytes`; push each event to `out`.
    pub fn feed(&mut self, bytes: &[u8], out: &mut Vec<InputEvent>) {
        for &byte in bytes {
            self.state = match (self.state, byte) {
                (ParseState::Ground, 0x1b) => ParseState::Esc,
                (ParseState::Ground, b'q' | b'Q' | 0x03 | 0x1c) => {
                    out.push(InputEvent::Quit);
                    ParseState::Ground
                }
                (ParseState::Ground, _) => ParseState::Ground,
                (ParseState::Esc, b'[') => ParseState::Csi(false),
                (ParseState::Esc, 0x1b) => ParseState::Esc,
                (ParseState::Esc, _) => ParseState::Ground,
                (ParseState::Csi(params), 0x20..=0x3f) => {
                    let _ = params;
                    ParseState::Csi(true)
                }
                (ParseState::Csi(false), b'I') => {
                    out.push(InputEvent::FocusIn);
                    ParseState::Ground
                }
                (ParseState::Csi(false), b'O') => {
                    out.push(InputEvent::FocusOut);
                    ParseState::Ground
                }
                (ParseState::Csi(_), 0x40..=0x7e) => ParseState::Ground,
                (ParseState::Csi(_), 0x1b) => ParseState::Esc,
                (ParseState::Csi(_), _) => ParseState::Ground,
            };
        }
    }
}

/// Longest model name put in a title.
pub const MODEL_CAP: usize = 40;
/// Longest host name put in a title.
pub const HOST_CAP: usize = 64;

/// The model on tty11's header row, as llama-watch draws it:
/// `<host>  llama-bored  █ READY  model <name>  <detail> ...`. The name runs
/// from after `  model ` to the next gap of two spaces. `--` and `...` (no
/// model, or starting) and an empty name are `None`. The result is
/// [`sanitize`]d and capped at [`MODEL_CAP`].
pub fn header_model(screen: &Screen) -> Option<String> {
    let width = usize::from(screen.cols);
    let row: String = screen
        .cells
        .get(..width)?
        .iter()
        .map(|cell| cell.ch)
        .collect();
    // The label always follows a field gap of two spaces (the state word's
    // gap, or the AI DOWN badge's own pad plus one).
    let at = row.find("  model ")?;
    let rest = row[at + "  model ".len()..].trim_start_matches(' ');
    let name = rest.split("  ").next().unwrap_or("").trim_end();
    if name.is_empty() || name == "--" || name == "..." {
        return None;
    }
    let clean = sanitize(name, MODEL_CAP);
    (!clean.is_empty()).then_some(clean)
}

/// Text that is safe inside an OSC or tmux string: printable ASCII only
/// (space through `~`), anything else becomes `?`, at most `cap` chars, no
/// leading or trailing spaces. No ESC, BEL or C1 byte can come out.
pub fn sanitize(text: &str, cap: usize) -> String {
    let out: String = text
        .chars()
        .take(cap)
        .map(|ch| if (' '..='~').contains(&ch) { ch } else { '?' })
        .collect();
    out.trim().to_string()
}

/// The pane title: `llama-view: <model> · <host>`, or `llama-view: <host>`.
pub fn title_text(model: Option<&str>, host: &str) -> String {
    match model {
        Some(model) => format!("llama-view: {model} \u{00b7} {host}"),
        None => format!("llama-view: {host}"),
    }
}

/// OSC 2 (window title, tmux `#T`), terminated by ST (`ESC \`), never BEL.
pub fn osc_title(text: &str) -> Vec<u8> {
    let mut out = b"\x1b]2;".to_vec();
    out.extend_from_slice(text.as_bytes());
    out.extend_from_slice(b"\x1b\\");
    out
}

/// tmux's window rename, `ESC k <name> ESC \` (needs `allow-rename on`).
pub fn tmux_window_name(name: &str) -> Vec<u8> {
    let mut out = b"\x1bk".to_vec();
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(b"\x1b\\");
    out
}

/// Sends the title (and, opted in, the tmux window name) when the model
/// changes, and only then.
#[derive(Clone, Debug)]
pub struct Titler {
    host: String,
    tmux_window: bool,
    last: Option<Option<String>>,
}

impl Titler {
    /// `host` is sanitised here. `tmux_window` is `--tmux-window-name` (or
    /// `LLAMA_VIEW_TMUX_WINDOW`) and `TMUX` set.
    pub fn new(host: &str, tmux_window: bool) -> Self {
        let host = sanitize(host, HOST_CAP);
        let host = if host.is_empty() {
            "localhost".to_string()
        } else {
            host
        };
        Self {
            host,
            tmux_window,
            last: None,
        }
    }

    /// The bytes to send for this frame's `model`: empty when unchanged.
    pub fn update(&mut self, model: Option<&str>) -> Vec<u8> {
        if self
            .last
            .as_ref()
            .is_some_and(|last| last.as_deref() == model)
        {
            return Vec::new();
        }
        self.last = Some(model.map(str::to_string));
        let mut out = osc_title(&title_text(model, &self.host));
        if self.tmux_window {
            out.extend_from_slice(&tmux_window_name(model.unwrap_or("llama-view")));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn screen_of(rows: &[&str]) -> Screen {
        let cols = rows.iter().map(|r| r.chars().count()).max().unwrap_or(0) as u16;
        let mut cells = Vec::new();
        for row in rows {
            let mut n = 0;
            for ch in row.chars() {
                cells.push(Cell::new(ch, Color::White, Color::Black));
                n += 1;
            }
            for _ in n..usize::from(cols) {
                cells.push(Cell::blank());
            }
        }
        Screen {
            cols,
            rows: rows.len() as u16,
            cursor_x: 0,
            cursor_y: 0,
            cells,
        }
    }

    fn text(screen: &Screen) -> Vec<String> {
        screen
            .cells
            .chunks(usize::from(screen.cols.max(1)))
            .map(|row| row.iter().map(|c| c.ch).collect())
            .collect()
    }

    #[test]
    fn a_pane_that_fits_shows_everything_and_no_hint() {
        let screen = screen_of(&["abcd", "efgh"]);
        let view = place(&screen, 6, 3, 0, 0, Fit::Crop, 11);
        assert_eq!(text(&view), ["abcd  ", "efgh  ", "      "]);
        let view = place(&screen, 4, 2, 0, 0, Fit::Crop, 11);
        assert_eq!(text(&view), ["abcd", "efgh"]);
    }

    #[test]
    fn crop_keeps_the_top_left_and_adds_a_hint_line() {
        let screen = screen_of(&["abcdef", "ghijkl", "mnopqr", "stuvwx"]);
        let view = place(&screen, 4, 3, 0, 0, Fit::Crop, 11);
        let rows = text(&view);
        assert_eq!(rows[..2], ["abcd", "ghij"]);
        assert_eq!(rows[2], " tty");
        let wide = place(&screen, 60, 3, 0, 0, Fit::Crop, 11);
        let hint = &text(&wide)[2];
        assert!(
            hint.starts_with(" tty11 is 6x4, this pane 60x3: top-left shown"),
            "{hint}"
        );
        assert!(
            wide.cells[wide.cells.len() - 1].bg == Color::White,
            "the hint is a bar"
        );
        // Offsets still pan a crop.
        let panned = place(&screen, 3, 2, 2, 1, Fit::Crop, 11);
        assert_eq!(text(&panned)[0], "ijk");
    }

    #[test]
    fn center_shows_the_middle_and_centres_a_small_console() {
        let screen = screen_of(&["abcdef", "ghijkl", "mnopqr", "stuvwx"]);
        let view = place(&screen, 2, 3, 0, 0, Fit::Center, 11);
        let rows = text(&view);
        assert_eq!(rows[..2], ["ij", "op"]);
        let wide = place(&screen, 60, 3, 0, 0, Fit::Center, 11);
        assert!(text(&wide)[2].contains("middle shown"));
        let big = place(&screen_of(&["ab"]), 4, 3, 0, 0, Fit::Center, 11);
        assert_eq!(text(&big), ["    ", " ab ", "    "]);
    }

    #[test]
    fn a_one_row_pane_is_only_the_hint() {
        let screen = screen_of(&["abc", "def"]);
        let view = place(&screen, 3, 1, 0, 0, Fit::Crop, 11);
        assert_eq!(text(&view), [" tt"]);
        let none = place(&screen, 0, 0, 0, 0, Fit::Crop, 11);
        assert!(none.cells.is_empty());
    }

    #[test]
    fn unfocused_drops_to_one_frame_a_second() {
        assert_eq!(frame_period(10, true), Duration::from_millis(100));
        assert_eq!(frame_period(10, false), Duration::from_secs(1));
        assert_eq!(frame_period(1, true), Duration::from_secs(1));
        assert_eq!(frame_period(1, false), Duration::from_secs(1));
        assert_eq!(frame_period(20, true), Duration::from_millis(50));
    }

    #[test]
    fn input_focus_and_quit_including_split_sequences() {
        let mut parser = InputParser::default();
        let mut events = Vec::new();
        parser.feed(b"\x1b[I", &mut events);
        parser.feed(b"\x1b[", &mut events);
        parser.feed(b"O", &mut events);
        assert_eq!(events, [InputEvent::FocusIn, InputEvent::FocusOut]);
        events.clear();
        // Arrow keys, modified keys and plain text do nothing.
        parser.feed(b"\x1b[A\x1b[1;5Ix\x1bOP", &mut events);
        assert!(events.is_empty(), "{events:?}");
        for quit in [&b"q"[..], b"Q", b"\x03", b"\x1c"] {
            parser.feed(quit, &mut events);
        }
        assert_eq!(events, [InputEvent::Quit; 4]);
        events.clear();
        parser.feed(b"\x1b\x1b[I", &mut events);
        assert_eq!(events, [InputEvent::FocusIn]);
    }

    #[test]
    fn header_model_parses_the_watch_header() {
        let row = "box  llama-bored    █ READY  model qwen3-coder-30b-a3b  q4_K_M · 64k   slots 1/4   swap ok       12:00";
        assert_eq!(
            header_model(&screen_of(&[row])).as_deref(),
            Some("qwen3-coder-30b-a3b")
        );
        let stuck = "box  llama-bored  █ GENERATING  model glm-4.5  stopping (stuck?)  x";
        assert_eq!(
            header_model(&screen_of(&[stuck])).as_deref(),
            Some("glm-4.5")
        );
        let end = "h  llama-bored  █ READY  model tiny";
        assert_eq!(header_model(&screen_of(&[end])).as_deref(), Some("tiny"));
    }

    #[test]
    fn header_model_absent_long_and_odd() {
        for row in [
            "box  llama-bored  █ AI DOWN  model --   slots --",
            "box  llama-bored  █ STARTING  model ...   slots --",
            "box  llama-bored  █ READY  model    ",
            "box  llama-bored  no model field here",
        ] {
            assert_eq!(header_model(&screen_of(&[row])), None, "{row}");
        }
        assert_eq!(header_model(&screen_of(&[])), None);
        let long = format!("h  llama-bored  █ READY  model {}  detail", "m".repeat(90));
        assert_eq!(
            header_model(&screen_of(&[&long])),
            Some("m".repeat(MODEL_CAP))
        );
        // The screen decoder already turned ESC into '?'; anything not
        // printable ASCII (a block glyph, a C1) becomes '?' too.
        let odd = "h  llama-bored  █ READY  model a█b\u{9b}c?\u{7}d  x";
        assert_eq!(
            header_model(&screen_of(&[odd])).as_deref(),
            Some("a?b?c??d")
        );
        // Only the first row is the header.
        let below = screen_of(&["no header", "x  model foo  y"]);
        assert_eq!(header_model(&below), None);
    }

    #[test]
    fn sanitize_never_lets_a_control_through() {
        let dirty = "a\x1b]2;x\x07\u{9c}\u{1b}\\b";
        let clean = sanitize(dirty, 64);
        assert!(
            clean.bytes().all(|b| (0x20..=0x7e).contains(&b)),
            "{clean:?}"
        );
        assert_eq!(sanitize("  box  ", 64), "box");
        assert_eq!(sanitize("abcdef", 3), "abc");
    }

    #[test]
    fn title_is_sent_on_change_only() {
        let mut titler = Titler::new("box", false);
        let first = titler.update(Some("qwen3"));
        assert_eq!(first, b"\x1b]2;llama-view: qwen3 \xc2\xb7 box\x1b\\");
        assert!(titler.update(Some("qwen3")).is_empty());
        assert_eq!(titler.update(None), b"\x1b]2;llama-view: box\x1b\\");
        assert!(titler.update(None).is_empty());
        assert!(!titler.update(Some("glm")).is_empty());
        assert!(
            !first.contains(&0x07),
            "ST, not BEL: llama-view never rings the bell"
        );
    }

    #[test]
    fn a_first_title_is_sent_even_without_a_model() {
        let mut titler = Titler::new("ti\x1btan\x07", false);
        assert_eq!(titler.update(None), b"\x1b]2;llama-view: ti?tan?\x1b\\");
        assert_eq!(
            Titler::new("", false).update(None),
            b"\x1b]2;llama-view: localhost\x1b\\"
        );
    }

    #[test]
    fn tmux_window_name_only_when_opted_in() {
        let mut off = Titler::new("box", false);
        let bytes = off.update(Some("qwen3"));
        assert!(!bytes.windows(2).any(|w| w == b"\x1bk"), "{bytes:?}");
        let mut on = Titler::new("box", true);
        let bytes = on.update(Some("qwen3"));
        assert!(bytes.ends_with(b"\x1bkqwen3\x1b\\"), "{bytes:?}");
        assert!(on.update(Some("qwen3")).is_empty());
        assert!(on.update(None).ends_with(b"\x1bkllama-view\x1b\\"));
    }

    #[test]
    fn fit_spellings() {
        assert_eq!(Fit::parse("crop"), Some(Fit::Crop));
        assert_eq!(Fit::parse("center"), Some(Fit::Center));
        assert_eq!(Fit::parse("centre"), None);
        assert_eq!(Fit::default(), Fit::Crop);
    }
}
