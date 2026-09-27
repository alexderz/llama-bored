//! `tty.prompt_view = "clean"`: chat-template control tokens out of a prompt
//! tail, role turns as `-- user --` label lines.
//!
//! This runs on the raw tail before the S12 sanitiser, which still runs last
//! on whatever comes out, so it adds no new way for bytes to reach the
//! console. Text with none of the markers in [`MARKERS`] and no `<|...|>`
//! special passes through unchanged.

/// What a marker does when it is found.
#[derive(Clone, Copy, Debug)]
enum Action {
    /// Remove the marker.
    Drop,
    /// Remove the marker and start a turn with this role.
    Role(&'static str),
    /// ChatML `<|im_start|>`: the role name follows on the same line.
    RoleOnLine,
    /// Llama-3 `<|start_header_id|>`: the role name runs up to this end marker.
    RoleUntil(&'static str),
}

/// Literal markers, tried in order at each `<` or `[`. Longer or more
/// specific entries come first. `<|...|>` specials not listed here are
/// dropped by [`generic_special`].
const MARKERS: &[(&str, Action)] = &[
    // ChatML (Qwen, many fine-tunes).
    ("<|im_start|>", Action::RoleOnLine),
    ("<|im_end|>", Action::Drop),
    // Llama-3.
    (
        "<|start_header_id|>",
        Action::RoleUntil("<|end_header_id|>"),
    ),
    ("<|eot_id|>", Action::Drop),
    // GLM-4: the role is the special itself.
    ("<|system|>", Action::Role("system")),
    ("<|user|>", Action::Role("user")),
    ("<|assistant|>", Action::Role("assistant")),
    ("<|observation|>", Action::Role("tool")),
    ("[gMASK]", Action::Drop),
    ("<sop>", Action::Drop),
    // Mistral.
    ("[SYSTEM_PROMPT]", Action::Role("system")),
    ("[/SYSTEM_PROMPT]", Action::Drop),
    ("[TOOL_RESULTS]", Action::Role("tool")),
    ("[/TOOL_RESULTS]", Action::Drop),
    ("[AVAILABLE_TOOLS]", Action::Drop),
    ("[/AVAILABLE_TOOLS]", Action::Drop),
    ("[TOOL_CALLS]", Action::Drop),
    ("[INST]", Action::Role("user")),
    ("[/INST]", Action::Role("assistant")),
    ("<s>", Action::Drop),
    ("</s>", Action::Drop),
    // Reasoning tags.
    ("<think>", Action::Drop),
    ("</think>", Action::Drop),
];

/// Label for a role name that is not in [`ROLES`].
const OTHER: &str = "other";

/// Role names as templates spell them, and the label each one gets.
/// A name not listed is labelled [`OTHER`].
const ROLES: &[(&str, &str)] = &[
    ("user", "user"),
    ("human", "user"),
    ("assistant", "assistant"),
    ("model", "assistant"),
    ("gpt", "assistant"),
    ("system", "system"),
    ("developer", "system"),
    ("tool", "tool"),
    ("ipython", "tool"),
    ("function", "tool"),
    ("observation", "tool"),
];

/// A turn whose content opens with one of these is labelled with the role
/// given here instead. Qwen sends tool results as a `user` turn.
const CONTENT_ROLES: &[(&str, &str)] = &[("<tool_response>", "tool")];

/// Longest `<|...|>` body the generic rule drops.
const MAX_SPECIAL: usize = 48;

/// Longest role name read after `<|im_start|>` or `<|start_header_id|>`.
const MAX_ROLE: usize = 24;

/// The label line for `role`. ASCII `--`: `─` is not in the console font.
/// The layout paints it dim like every IN line but the last, and a label is
/// never last because the trailing empty turn is dropped.
fn label_line(role: &str) -> String {
    format!("-- {role} --")
}

#[derive(Debug)]
enum Piece {
    Text(String),
    Label(String),
}

/// Remove chat-template control tokens from `text` and mark each role turn
/// with a label line. A trailing empty turn (the generation prompt) is
/// dropped, so the result ends with the last real content.
///
/// `truncated` says `text` is a tail cut from a longer prompt. When it is,
/// and markers were found, the first partial line is dropped: it may start
/// in the middle of a control token.
#[must_use]
pub fn clean(text: &str, truncated: bool) -> String {
    let mut pieces: Vec<Piece> = Vec::new();
    let mut text_buf = String::new();
    let mut found = false;
    let mut rest = text;
    while let Some(ch) = rest.chars().next() {
        if ch == '<' || ch == '[' {
            if let Some((action, len)) = match_marker(rest) {
                found = true;
                rest = &rest[len..];
                match action {
                    Action::Drop => rest = eat_line_end(&text_buf, rest),
                    Action::Role(role) => {
                        push_label(&mut pieces, &mut text_buf, role);
                        rest = skip_ws(rest);
                    }
                    Action::RoleOnLine => {
                        let (role, after) = role_on_line(rest);
                        if let Some(role) = role {
                            push_label(&mut pieces, &mut text_buf, role);
                            rest = skip_ws(after);
                        }
                    }
                    Action::RoleUntil(end) => {
                        if let Some((role, after)) = role_until(rest, end) {
                            push_label(&mut pieces, &mut text_buf, role);
                            rest = skip_ws(after);
                        }
                    }
                }
                continue;
            }
            if let Some(len) = generic_special(rest) {
                found = true;
                rest = eat_line_end(&text_buf, &rest[len..]);
                continue;
            }
        }
        text_buf.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    if !found {
        return text.to_owned();
    }
    flush(&mut pieces, &mut text_buf);
    relabel_by_content(&mut pieces);
    drop_trailing_empty_turn(&mut pieces);
    if truncated {
        drop_partial_first_line(&mut pieces);
    }
    render(&pieces)
}

fn match_marker(rest: &str) -> Option<(Action, usize)> {
    MARKERS
        .iter()
        .find(|(marker, _)| rest.starts_with(marker))
        .map(|(marker, action)| (*action, marker.len()))
}

/// `<|name|>` with a short body of no whitespace and no `<`/`>`/`|`.
fn generic_special(rest: &str) -> Option<usize> {
    let body = rest.strip_prefix("<|")?;
    let end = body.find("|>")?;
    let name = &body[..end];
    let ok = !name.is_empty()
        && name.chars().count() <= MAX_SPECIAL
        && name
            .chars()
            .all(|ch| !ch.is_whitespace() && ch != '<' && ch != '>' && ch != '|');
    ok.then_some(2 + end + 2)
}

/// A role word alone on the rest of the line, as ChatML writes it.
fn role_on_line(rest: &str) -> (Option<&'static str>, &str) {
    let line_end = rest.find('\n').unwrap_or(rest.len());
    let word = rest[..line_end].trim_end_matches([' ', '\t', '\r']);
    if !is_role_word(word) {
        return (None, rest);
    }
    let after = if line_end < rest.len() {
        &rest[line_end + 1..]
    } else {
        &rest[line_end..]
    };
    (Some(role_label(word)), after)
}

fn role_until<'a>(rest: &'a str, end: &str) -> Option<(&'static str, &'a str)> {
    let at = rest.find(end)?;
    let word = rest[..at].trim();
    is_role_word(word).then(|| (role_label(word), &rest[at + end.len()..]))
}

fn is_role_word(word: &str) -> bool {
    !word.is_empty()
        && word.len() <= MAX_ROLE
        && word
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}

/// Known spellings map through [`ROLES`]; anything else is labelled as is.
fn role_label(word: &str) -> &'static str {
    let lower = word.to_ascii_lowercase();
    ROLES
        .iter()
        .find(|(name, _)| *name == lower)
        .map_or(OTHER, |(_, label)| label)
}

/// A dropped marker that sat alone at the start of a line takes its line end
/// with it, so it leaves no blank line behind.
fn eat_line_end<'a>(text_buf: &str, rest: &'a str) -> &'a str {
    if text_buf.is_empty() || text_buf.ends_with('\n') {
        rest.strip_prefix('\n').unwrap_or(rest)
    } else {
        rest
    }
}

fn skip_ws(rest: &str) -> &str {
    rest.trim_start_matches([' ', '\t', '\r', '\n'])
}

fn push_label(pieces: &mut Vec<Piece>, text_buf: &mut String, role: &str) {
    flush(pieces, text_buf);
    pieces.push(Piece::Label(role.to_owned()));
}

fn flush(pieces: &mut Vec<Piece>, text_buf: &mut String) {
    if !text_buf.is_empty() {
        pieces.push(Piece::Text(std::mem::take(text_buf)));
    }
}

fn relabel_by_content(pieces: &mut [Piece]) {
    for i in 0..pieces.len().saturating_sub(1) {
        let Piece::Text(next) = &pieces[i + 1] else {
            continue;
        };
        let Some((_, role)) = CONTENT_ROLES
            .iter()
            .find(|(open, _)| next.trim_start().starts_with(open))
        else {
            continue;
        };
        if let Piece::Label(label) = &mut pieces[i] {
            *label = (*role).to_owned();
        }
    }
}

fn is_blank(piece: &Piece) -> bool {
    matches!(piece, Piece::Text(text) if text.trim().is_empty())
}

fn drop_trailing_empty_turn(pieces: &mut Vec<Piece>) {
    while pieces.last().is_some_and(is_blank) {
        pieces.pop();
    }
    if matches!(pieces.last(), Some(Piece::Label(_))) {
        pieces.pop();
    }
}

fn drop_partial_first_line(pieces: &mut Vec<Piece>) {
    let Some(Piece::Text(first)) = pieces.first_mut() else {
        return;
    };
    match first.find('\n') {
        Some(at) => {
            first.drain(..=at);
            if first.trim().is_empty() {
                pieces.remove(0);
            }
        }
        // One partial line that runs into the next turn: nothing of it is
        // whole, and it may hold half a token.
        None if pieces.len() > 1 => {
            pieces.remove(0);
        }
        None => {}
    }
}

fn render(pieces: &[Piece]) -> String {
    let mut out = String::new();
    let mut after_label = false;
    for piece in pieces {
        match piece {
            Piece::Label(role) => {
                let kept = out.trim_end_matches([' ', '\t', '\r', '\n']).len();
                out.truncate(kept);
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&label_line(role));
                out.push('\n');
                after_label = true;
            }
            Piece::Text(text) if after_label => {
                out.push_str(text.trim_start_matches(['\r', '\n']));
                after_label = false;
            }
            Piece::Text(text) => out.push_str(text),
        }
    }
    let kept = out.trim_end_matches([' ', '\t', '\r', '\n']).len();
    out.truncate(kept);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_turns_become_labels_and_the_generation_prompt_goes() {
        let raw = "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        assert_eq!(clean(raw, false), "-- system --\nBe brief.\n-- user --\nhi");
    }

    #[test]
    fn chatml_tool_response_turn_is_labelled_tool() {
        let raw = "<|im_start|>assistant\n<think>\nplan\n</think>\n\n<tool_call>\n<function=write>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nSuccessfully wrote to tests/test_live.py\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n";
        assert_eq!(
            clean(raw, false),
            "-- assistant --\nplan\n\n<tool_call>\n<function=write>\n</function>\n</tool_call>\n-- tool --\n<tool_response>\nSuccessfully wrote to tests/test_live.py\n</tool_response>"
        );
    }

    #[test]
    fn llama3_headers_become_labels() {
        let raw = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n\n{\"ok\":1}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        assert_eq!(
            clean(raw, false),
            "-- system --\nBe brief.\n-- user --\nhi\n-- tool --\n{\"ok\":1}"
        );
    }

    #[test]
    fn mistral_inst_blocks_become_labels() {
        let raw = "<s>[INST] hi [/INST] Hello!</s>[INST] and now? [/INST]";
        assert_eq!(
            clean(raw, false),
            "-- user --\nhi\n-- assistant --\nHello!\n-- user --\nand now?"
        );
        let tools = "<s>[SYSTEM_PROMPT]sys[/SYSTEM_PROMPT][INST]run it[/INST][TOOL_CALLS][{\"name\":\"x\"}]</s>[TOOL_RESULTS]{\"ok\":1}[/TOOL_RESULTS]";
        assert_eq!(
            clean(tools, false),
            "-- system --\nsys\n-- user --\nrun it\n-- assistant --\n[{\"name\":\"x\"}]\n-- tool --\n{\"ok\":1}"
        );
    }

    #[test]
    fn glm_role_specials_become_labels() {
        let raw = "[gMASK]<sop><|system|>\nBe brief.<|user|>\nhi<|assistant|>\n<think></think>\nHello!<|observation|>\n<tool_response>\n42\n</tool_response><|assistant|>\n<think>";
        assert_eq!(
            clean(raw, false),
            "-- system --\nBe brief.\n-- user --\nhi\n-- assistant --\nHello!\n-- tool --\n<tool_response>\n42\n</tool_response>"
        );
    }

    #[test]
    fn unknown_formats_pass_through_unchanged() {
        for raw in [
            "",
            "plain prompt\nwith lines\n",
            "### Instruction:\nhi\n### Response:\n",
            "a < b and [c] | d |> e <| f",
            "html <b>bold</b> and <|not a special|>",
        ] {
            assert_eq!(clean(raw, false), raw, "{raw:?}");
            assert_eq!(clean(raw, true), raw, "{raw:?}");
        }
    }

    #[test]
    fn unknown_specials_are_dropped_and_unknown_roles_keep_a_label() {
        assert_eq!(clean("a<|endoftext|>b", false), "ab");
        assert_eq!(
            clean("<|im_start|>critic\nno<|im_end|>", false),
            "-- other --\nno"
        );
    }

    #[test]
    fn a_cut_tail_loses_its_partial_first_line() {
        let raw = "tart|>user\nhi<|im_end|>\n<|im_start|>assistant\nyo<|im_end|>\n";
        assert_eq!(clean(raw, true), "hi\n-- assistant --\nyo");
        assert_eq!(clean(raw, false), "tart|>user\nhi\n-- assistant --\nyo");
    }
}
