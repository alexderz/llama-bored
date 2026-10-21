//! One mirror session without the devices (#27): decode a vcsa read, fit it
//! to the pane, keep the title current, and track focus. `main` feeds it
//! real reads and real stdin; tests feed it fixtures.

use std::time::Duration;

use crate::cli::{ColorSettings, Options};
use crate::gridsrc::screen_of;
use crate::pane::{
    Fit, InputEvent, InputParser, Titler, frame_period, header_engine, header_model, place,
};
use crate::render::Renderer;
use crate::screen::{
    AttrLayout, Cell, Color, DecodeError, Screen, decode_screen_with, detect_layout,
};
use llama_core::gridfile::GridImage;

/// What the input asked for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Next {
    /// Keep mirroring.
    Continue,
    /// Restore the terminal and exit.
    Quit,
}

pub struct Session {
    renderer: Renderer,
    titler: Titler,
    font_glyphs: Option<AttrLayout>,
    layout: AttrLayout,
    parser: InputParser,
    events: Vec<InputEvent>,
    focused: bool,
    last_sizes: Option<((u16, u16), (u16, u16))>,
    fit: Fit,
    offset_x: u16,
    offset_y: u16,
    vt: u8,
    fps: u8,
    kiosk: bool,
    segments: bool,
}

impl Session {
    /// `host` is read once by the caller; `tmux_window` is the resolved
    /// `--tmux-window-name` (opted in and inside tmux); `kiosk` is the
    /// resolved `--kiosk` (#104): the keyboard is ignored (see [`Self::input`]).
    pub fn new(
        opts: &Options,
        colors: ColorSettings,
        host: &str,
        tmux_window: bool,
        kiosk: bool,
    ) -> Self {
        Self {
            renderer: Renderer::with_colors(colors.mode, colors.palette).synchronized(true),
            titler: Titler::new(host, tmux_window),
            font_glyphs: opts.font_glyphs,
            layout: opts.font_glyphs.unwrap_or_default(),
            parser: InputParser::default(),
            events: Vec::new(),
            focused: true,
            last_sizes: None,
            fit: opts.fit,
            offset_x: opts.offset_x,
            offset_y: opts.offset_y,
            vt: opts.vt,
            fps: opts.fps,
            kiosk,
            segments: opts.segments,
        }
    }

    /// Append this frame's bytes to `out`: the title when the model changed,
    /// then the changed cells, or a full repaint when the pane or the console
    /// changed size. Nothing when nothing changed.
    pub fn frame(
        &mut self,
        pane: (u16, u16),
        vcsa: &[u8],
        vcs: Option<&[u8]>,
        vcsu: Option<&[u8]>,
        out: &mut Vec<u8>,
    ) -> Result<(), DecodeError> {
        if self.font_glyphs.is_none()
            && let Some(found) = detect_layout(vcsa, vcsu)
        {
            self.layout = found;
        }
        let screen = decode_screen_with(vcsa, vcs, vcsu, self.layout)?;
        self.paint(screen, pane, out);
        Ok(())
    }

    /// [`Self::frame`] for the grid source (#107): the cells and exact
    /// colours of a decoded grid file, fitted, titled and diffed the same way.
    pub fn frame_grid(&mut self, pane: (u16, u16), image: &GridImage, out: &mut Vec<u8>) {
        self.paint(screen_of(image, self.segments), pane, out);
    }

    fn paint(&mut self, screen: Screen, pane: (u16, u16), out: &mut Vec<u8>) {
        let sizes = (pane, (screen.cols, screen.rows));
        if self.last_sizes.is_some_and(|last| last != sizes) {
            self.renderer.invalidate();
        }
        self.last_sizes = Some(sizes);
        out.extend_from_slice(
            &self
                .titler
                .update(header_model(&screen).as_deref(), header_engine(&screen)),
        );
        let view = place(
            &screen,
            pane.0,
            pane.1,
            self.offset_x,
            self.offset_y,
            self.fit,
            self.vt,
        );
        out.extend_from_slice(self.renderer.render(view.cols, view.rows, &view.cells));
    }

    /// `--kiosk` (#104), while the console source is down: a blank screen
    /// with `text` on its top row, through the same renderer a real frame
    /// uses. An unchanged line costs nothing (the renderer's own diff), and
    /// [`Self::invalidate`] forces a full repaint once the source returns,
    /// so the renderer does not try to diff real cells against this one.
    pub fn status_frame(&mut self, pane: (u16, u16), text: &str, out: &mut Vec<u8>) {
        let (cols, rows) = pane;
        let mut cells = vec![Cell::blank(); usize::from(cols).saturating_mul(usize::from(rows))];
        for (cell, ch) in cells.iter_mut().take(usize::from(cols)).zip(text.chars()) {
            *cell = Cell::new(ch, Color::White, Color::Black);
        }
        out.extend_from_slice(self.renderer.render(cols, rows, &cells));
    }

    /// Forget the last painted frame, so the next one (real or
    /// [`Self::status_frame`]) is a full repaint. `--kiosk` calls this when
    /// the console source comes back, so the renderer does not diff real
    /// cells against the status screen it last drew.
    pub fn invalidate(&mut self) {
        self.renderer.invalidate();
    }

    /// Bytes from stdin. `--kiosk` (#104) discards every byte: the keyboard
    /// does nothing, so `focused` never flips and the frame rate never
    /// drops (see [`Self::period`]).
    pub fn input(&mut self, bytes: &[u8]) -> Next {
        if self.kiosk {
            return Next::Continue;
        }
        self.events.clear();
        self.parser.feed(bytes, &mut self.events);
        for event in &self.events {
            match event {
                InputEvent::Quit => return Next::Quit,
                InputEvent::FocusIn => self.focused = true,
                InputEvent::FocusOut => self.focused = false,
            }
        }
        Next::Continue
    }

    /// Wait before the next frame: `1 / --fps`, or 1 s while unfocused.
    pub fn period(&self) -> Duration {
        frame_period(self.fps, self.focused)
    }

    pub fn focused(&self) -> bool {
        self.focused
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::ColorMode;
    use crate::render::{SYNC_BEGIN, SYNC_END};
    use llama_core::palette::Palette;

    const HEADER: &str = "box  llama-bored  █ READY  model qwen3-30b  q4  slots 1/1";

    /// `rows` x `cols` 512-glyph vcsa: the header on row 0, blanks below.
    fn tty(cols: u8, rows: u8, header: &str) -> (Vec<u8>, Vec<u8>) {
        let mut vcsa = vec![rows, cols, 0, 0];
        let mut vcsu = Vec::new();
        let chars: Vec<char> = header.chars().collect();
        for i in 0..usize::from(rows) * usize::from(cols) {
            let ch = if i < usize::from(cols) {
                chars.get(i).copied().unwrap_or(' ')
            } else {
                ' '
            };
            let glyph = if ch.is_ascii() { ch as u8 } else { 0xdb };
            vcsa.extend_from_slice(&[glyph, 7 << 1]);
            vcsu.extend_from_slice(&u32::from(ch).to_ne_bytes());
        }
        (vcsa, vcsu)
    }

    fn session(fit: Fit) -> Session {
        session_with(fit, false)
    }

    fn session_with(fit: Fit, kiosk: bool) -> Session {
        let opts = Options {
            fit,
            ..Options::default()
        };
        let colors = ColorSettings {
            mode: ColorMode::Truecolor,
            palette: Palette::Llama,
        };
        Session::new(&opts, colors, "box", false, kiosk)
    }

    fn has(hay: &[u8], needle: &[u8]) -> bool {
        hay.windows(needle.len()).any(|w| w == needle)
    }

    #[test]
    fn first_frame_sets_the_title_and_is_synchronized() {
        let (vcsa, vcsu) = tty(80, 4, HEADER);
        let mut s = session(Fit::Crop);
        let mut out = Vec::new();
        s.frame((100, 10), &vcsa, None, Some(&vcsu), &mut out)
            .unwrap();
        let title = "\x1b]2;llama-view: qwen3-30b \u{b7} box\x1b\\";
        assert!(
            out.starts_with(title.as_bytes()),
            "{:?}",
            String::from_utf8_lossy(&out)
        );
        assert!(has(&out, SYNC_BEGIN) && out.ends_with(SYNC_END));
        assert!(
            has(&out, b"\x1b[38;2;200;200;200m"),
            "truecolor llama white"
        );
        // Same screen again: not a byte, not even the title.
        out.clear();
        s.frame((100, 10), &vcsa, None, Some(&vcsu), &mut out)
            .unwrap();
        assert!(out.is_empty(), "{out:?}");
        // The model changes: the title follows.
        let (other, other_u) = tty(80, 4, "box  llama-bored  █ READY  model glm-4.5  x");
        s.frame((100, 10), &other, None, Some(&other_u), &mut out)
            .unwrap();
        assert!(has(&out, "llama-view: glm-4.5 \u{b7} box".as_bytes()));
    }

    #[test]
    fn a_resize_repaints_in_full_and_a_small_pane_gets_the_hint() {
        let (vcsa, vcsu) = tty(80, 4, HEADER);
        let mut s = session(Fit::Crop);
        let mut out = Vec::new();
        s.frame((100, 10), &vcsa, None, Some(&vcsu), &mut out)
            .unwrap();
        out.clear();
        s.frame((40, 3), &vcsa, None, Some(&vcsu), &mut out)
            .unwrap();
        assert!(has(&out, b"\x1b[2J"), "full repaint after a pane resize");
        assert!(
            has(&out, b" tty11 is 80x4, this pane 40x3: top"),
            "{:?}",
            String::from_utf8_lossy(&out)
        );
        // The console grows (tty11 [tty] size): full repaint again.
        out.clear();
        let (bigger, bigger_u) = tty(90, 4, HEADER);
        s.frame((40, 3), &bigger, None, Some(&bigger_u), &mut out)
            .unwrap();
        assert!(has(&out, b"\x1b[2J"), "full repaint after a console resize");
        assert!(has(&out, b"tty11 is 90x4"));
    }

    #[test]
    fn center_fit_names_itself_in_the_hint() {
        let (vcsa, vcsu) = tty(80, 4, HEADER);
        let mut s = session(Fit::Center);
        let mut out = Vec::new();
        s.frame((60, 3), &vcsa, None, Some(&vcsu), &mut out)
            .unwrap();
        assert!(has(&out, b"middle shown"));
    }

    #[test]
    fn focus_switches_the_rate_and_q_quits() {
        let mut s = session(Fit::Crop);
        assert_eq!(s.period(), Duration::from_millis(100));
        assert_eq!(s.input(b"\x1b[O"), Next::Continue);
        assert!(!s.focused());
        assert_eq!(s.period(), Duration::from_secs(1));
        assert_eq!(s.input(b"\x1b[I"), Next::Continue);
        assert_eq!(s.period(), Duration::from_millis(100));
        assert_eq!(s.input(b"x"), Next::Continue);
        assert_eq!(s.input(b"\x03"), Next::Quit, "Ctrl-C is a clean quit");
        assert_eq!(s.input(b"q"), Next::Quit);
    }

    /// #104: kiosk discards every byte, so none of the quit keys, Ctrl-C,
    /// or a focus report change anything.
    #[test]
    fn kiosk_ignores_every_input_byte() {
        let mut s = session_with(Fit::Crop, true);
        assert_eq!(s.period(), Duration::from_millis(100));
        for bytes in [&b"q"[..], b"Q", b"\x03", b"\x1c", b"\x1b[O", b"anything"] {
            assert_eq!(s.input(bytes), Next::Continue, "{bytes:?}");
        }
        // No FocusOut was ever applied: still the focused frame rate.
        assert!(s.focused());
        assert_eq!(s.period(), Duration::from_millis(100));
    }

    /// #104: a blank status screen, through the same diffing renderer a
    /// real frame uses, and `invalidate` forces the next real frame to be a
    /// full repaint rather than a diff against the status screen's cells.
    #[test]
    fn status_frame_then_recovery_forces_a_full_repaint() {
        let mut s = session_with(Fit::Crop, true);
        let mut out = Vec::new();
        s.status_frame(
            (60, 3),
            "llama-view: waiting for tty11 \u{2014} gone",
            &mut out,
        );
        assert!(!out.is_empty(), "the first status frame has to paint");
        assert!(has(&out, b"waiting for tty11"));

        // The same text again: nothing new (no flicker).
        out.clear();
        s.status_frame(
            (60, 3),
            "llama-view: waiting for tty11 \u{2014} gone",
            &mut out,
        );
        assert!(out.is_empty(), "{out:?}");

        // The source comes back: without invalidate, a same-size real frame
        // would just diff against the status cells. invalidate forces 2J.
        s.invalidate();
        out.clear();
        let (vcsa, vcsu) = tty(60, 3, HEADER);
        s.frame((60, 3), &vcsa, None, Some(&vcsu), &mut out)
            .unwrap();
        assert!(
            has(&out, b"\x1b[2J"),
            "full repaint after recovery: {out:?}"
        );
    }

    /// Longer than the pane: truncated to `cols`, never wraps into row 1.
    /// Two calls with the same overlong text truncate the same way, so the
    /// second repaints nothing — if the text had spilled into row 1 with a
    /// different split each time this would not hold.
    #[test]
    fn status_frame_truncates_to_the_pane_width() {
        let mut s = session_with(Fit::Crop, true);
        let mut out = Vec::new();
        s.status_frame((4, 2), "much too long", &mut out);
        assert!(!out.is_empty());
        out.clear();
        s.status_frame((4, 2), "much too long", &mut out);
        assert!(out.is_empty(), "{out:?}");
    }
}
