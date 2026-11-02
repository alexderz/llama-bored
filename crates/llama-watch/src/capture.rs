//! llama-swap request captures, `GET /api/captures/<id>` (#5): IN and OUT
//! for a backend without `/slots` (SGLang, vLLM, other OpenAI-compatible
//! servers).
//!
//! llama-swap keeps the request and response bodies of recent requests when
//! its `captureBuffer` is on, and marks their activity rows `has_capture`. A
//! capture exists only once the request has finished, so this is the last
//! finished exchange, never text streaming in.
//!
//! The capture is JSON with base64 bodies (Go `[]byte`). Only `req_body` and
//! `resp_body` are read; the header maps are skipped by the parse, never
//! stored. From the request, IN is what is new in an OpenAI chat body (#38),
//! or a completions `prompt`:
//!
//! - the last message is `user`: that message alone (text parts joined), as
//!   before #38;
//! - otherwise (an agent's tool loop): the run of `user` and `tool` messages
//!   after the last `assistant` one, in order, newest last, one per line,
//!   tool results marked `[tool]`. With no `assistant` message at all, the
//!   run is every message but `system` / `developer` ones. [`CaptureText::
//!   input_note`] counts them for the IN title;
//! - nothing after the last `assistant` (a prefill): the last `user` message.
//!
//! Text kept while walking the messages is cut to its last
//! [`INPUT_CAP_CHARS`], the most IN ever shows. From the response,
//! OUT is the assistant text: a JSON body's `message.content`, or an SSE
//! stream's `delta.content` pieces joined; reasoning text when there is no
//! content; tool calls appended as `[call name] arguments`. The caller keeps
//! only the tails, through the console sanitiser.
//!
//! Each message of a chat request is also hashed whole, role and tool
//! calls included, so the poller can tell which of IN's messages this
//! request appended (DER-451, [`new_input_chars`]). Only the hashes and
//! roles are kept, never more text.
//!
//! The poller reads at most [`CAPTURE_CAP`] bytes: a capture of a 160k-token
//! agent prompt measured 0.84 MiB (bodies grow by a third in base64), so
//! 2 MiB covers a 200k context. A larger capture is skipped, not truncated
//! (a cut JSON document does not parse).

use std::fmt;
use std::hash::{BuildHasher, RandomState};
use std::sync::OnceLock;

use serde::de::{self, IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

/// Largest capture body the poller reads.
pub const CAPTURE_CAP: usize = 2 * 1024 * 1024;

/// IN text kept from one request: the largest `llama.input_tail_chars`.
pub const INPUT_CAP_CHARS: usize = 32_768;

/// Tool calls kept from one response.
const MAX_CALLS: usize = 16;

/// Most messages of one request hashed. A longer request keeps none, and
/// its new part is not known from the messages.
pub const MAX_MESSAGES: usize = 4096;

/// The keyed hasher of every message mark, made once per run so marks of
/// two captures compare and a hash says nothing outside this process.
fn message_hasher() -> &'static RandomState {
    static STATE: OnceLock<RandomState> = OnceLock::new();
    STATE.get_or_init(RandomState::new)
}

/// One request message as the diff sees it (DER-451): its whole JSON
/// hashed, and whether IN can show it (a `user`, `tool` or `function`
/// message).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MessageMark {
    pub hash: u64,
    pub in_role: bool,
}

/// The text of one finished exchange, before tail cutting and sanitising.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaptureText {
    /// The last user message, or the user and tool messages after the last
    /// assistant one (#38).
    pub input: String,
    /// What a tool-loop `input` holds, for the IN title: `3 tool results`,
    /// `user + 2 tool results`. Empty for a single user message or a prompt.
    pub input_note: String,
    /// The assistant's answer.
    pub output: String,
    /// Every message of a chat request, in order (DER-451). Empty for a
    /// completions prompt or past [`MAX_MESSAGES`].
    pub messages: Vec<MessageMark>,
    /// The messages [`Self::input`] is made of: each one's index in
    /// [`Self::messages`] and the character of `input` it starts at, 0
    /// when the cap cut its start.
    pub input_parts: Vec<(usize, usize)>,
}

/// Parse one capture. `None` when it is not a capture object or a body is
/// not base64.
#[must_use]
pub fn parse_capture(bytes: &[u8]) -> Option<CaptureText> {
    let capture: CaptureJson = serde_json::from_slice(bytes).ok()?;
    let request = base64_decode(&capture.req_body)?;
    let response = base64_decode(&capture.resp_body)?;
    drop(capture);
    let (input, input_note, messages, input_parts) = request_text(&request);
    Some(CaptureText {
        input,
        input_note,
        output: response_text(&response),
        messages,
        input_parts,
    })
}

/// Characters at the end of `text.input` that this request appended
/// (DER-451). `previous` are the message lists of earlier requests to the
/// same model; the one sharing the longest message prefix is this
/// conversation's last request, so other conversations in between do not
/// matter. The new part starts at the first appended `user` or `tool`
/// message: an appended assistant message is the reply OUT already showed.
///
/// `Some(input length)` when IN's first message is new (a new
/// conversation shares no prefix), `Some(0)` when nothing IN shows is new
/// (a retry of the same request). `None` when either side has no message
/// list, so the caller falls back to the uncached token count.
#[must_use]
pub fn new_input_chars<'a>(
    previous: impl IntoIterator<Item = &'a [MessageMark]>,
    text: &CaptureText,
) -> Option<usize> {
    if text.messages.is_empty() || text.input_parts.is_empty() {
        return None;
    }
    let common = previous
        .into_iter()
        .filter(|marks| !marks.is_empty())
        .map(|marks| {
            marks
                .iter()
                .zip(&text.messages)
                .take_while(|(left, right)| left.hash == right.hash)
                .count()
        })
        .max()?;
    let Some(first) = (common..text.messages.len()).find(|index| text.messages[*index].in_role)
    else {
        return Some(0);
    };
    let chars = text.input.chars().count();
    match text
        .input_parts
        .iter()
        .position(|(index, _)| *index >= first)
    {
        Some(0) => Some(chars),
        Some(at) => Some(chars.saturating_sub(text.input_parts[at].1)),
        None => Some(0),
    }
}

/// `new` of a raw text of `raw` characters, as characters of the `shown`
/// cells IN draws from its last `cap` characters, sanitised (DER-451).
/// Proportional, since sanitising can change the length; at least one
/// character when anything is new.
#[must_use]
pub fn shown_new(new: usize, raw: usize, cap: usize, shown: usize) -> usize {
    let tail = raw.min(cap);
    let new = new.min(tail);
    if new == 0 || tail == 0 {
        return 0;
    }
    if new == tail {
        return shown;
    }
    let scaled = (shown as f64 * new as f64 / tail as f64).round() as usize;
    scaled.clamp(1, shown)
}

#[derive(Deserialize)]
struct CaptureJson {
    req_body: String,
    resp_body: String,
}

/// IN of a chat request, its title note, the message marks and IN's parts;
/// or a completions prompt, with no marks.
fn request_text(body: &[u8]) -> (String, String, Vec<MessageMark>, Vec<(usize, usize)>) {
    let Ok(request) = serde_json::from_slice::<RequestJson>(body) else {
        return Default::default();
    };
    let prompt = |text: String| (cap_tail(text), String::new(), Vec::new(), Vec::new());
    match (request.messages.0, request.prompt) {
        (Some(recent), _) => recent.finish(),
        (None, Some(PromptJson::One(text))) => prompt(text),
        (None, Some(PromptJson::Many(texts))) => prompt(texts.last().cloned().unwrap_or_default()),
        (None, _) => Default::default(),
    }
}

/// The last [`INPUT_CAP_CHARS`] of `text`.
fn cap_tail(mut text: String) -> String {
    let count = text.chars().count();
    if count > INPUT_CAP_CHARS {
        let start = text
            .char_indices()
            .nth(count - INPUT_CAP_CHARS)
            .map_or(text.len(), |(at, _)| at);
        text.drain(..start);
    }
    text
}

/// Appends to a kept text, cutting its head once it is well past the cap,
/// so a long run never grows past a few caps' worth of bytes.
fn push_capped(buf: &mut String, text: &str) {
    buf.push_str(text);
    if buf.len() > 4 * INPUT_CAP_CHARS {
        *buf = cap_tail(std::mem::take(buf));
    }
}

#[derive(Deserialize)]
struct RequestJson {
    #[serde(default)]
    messages: Recent,
    #[serde(default)]
    prompt: Option<PromptJson>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PromptJson {
    One(String),
    Many(Vec<String>),
    Other(IgnoredAny),
}

/// What is new in a chat request (#38): the last `user` message, and the
/// run of `user` / `tool` messages since the last `assistant` one. Messages
/// are read one at a time and dropped, so a long history is never held
/// whole; both texts keep only about [`INPUT_CAP_CHARS`].
#[derive(Default)]
struct Recent(Option<RecentText>);

#[derive(Default)]
struct RecentText {
    /// The last `user` message.
    last_user: String,
    /// Its message index.
    last_user_at: usize,
    /// Every message's mark (DER-451); cleared past [`MAX_MESSAGES`].
    marks: Vec<MessageMark>,
    /// Messages seen, the index of the next one.
    count: usize,
    /// Each run message's index and the byte of `run` it starts at, plus
    /// [`Self::run_cut`].
    run_parts: Vec<(usize, usize)>,
    /// Bytes cut from the head of `run` since it was last cleared.
    run_cut: usize,
    /// Whether the last message seen is a `user` one.
    ends_with_user: bool,
    /// User and tool messages since the last assistant one, one per line.
    run: String,
    users: usize,
    tools: usize,
    /// Any user or tool message at all; without one, a `prompt` is IN.
    seen: bool,
}

/// Marker before each tool result in IN.
const TOOL_MARK: &str = "[tool]";

impl RecentText {
    fn add(&mut self, hash: u64, message: MessageJson) {
        let index = self.count;
        self.count += 1;
        let role = message.role.as_deref().unwrap_or_default();
        let tool = matches!(role, "tool" | "function");
        if self.count <= MAX_MESSAGES {
            self.marks.push(MessageMark {
                hash,
                in_role: matches!(role, "user" | "tool" | "function"),
            });
        } else {
            self.marks.clear();
        }
        match role {
            "assistant" => {
                self.run.clear();
                self.run_parts.clear();
                self.run_cut = 0;
                self.users = 0;
                self.tools = 0;
            }
            "user" | "tool" | "function" => {
                self.seen = true;
                let text = message.content.map(ContentJson::text).unwrap_or_default();
                if !self.run.is_empty() {
                    self.push_run("\n");
                }
                self.run_parts.push((index, self.run_cut + self.run.len()));
                if tool {
                    self.tools += 1;
                    self.push_run(TOOL_MARK);
                    if !text.is_empty() {
                        self.push_run(" ");
                    }
                } else {
                    self.users += 1;
                }
                self.push_run(&text);
                if !tool {
                    self.last_user = cap_tail(text);
                    self.last_user_at = index;
                }
            }
            // `system`, `developer`, unknown roles: not part of IN.
            _ => return,
        }
        self.ends_with_user = role == "user";
    }

    /// [`push_capped`] onto `run`, counting the bytes its cap cuts.
    fn push_run(&mut self, text: &str) {
        let before = self.run.len() + text.len();
        push_capped(&mut self.run, text);
        self.run_cut += before - self.run.len();
    }

    /// IN, its title note, the message marks and IN's parts.
    fn finish(self) -> (String, String, Vec<MessageMark>, Vec<(usize, usize)>) {
        if self.ends_with_user || self.tools == 0 {
            return (
                self.last_user,
                String::new(),
                self.marks,
                vec![(self.last_user_at, 0)],
            );
        }
        let results = if self.tools == 1 {
            "1 tool result".to_owned()
        } else {
            format!("{} tool results", self.tools)
        };
        let note = match self.users {
            0 => results,
            1 => format!("user + {results}"),
            n => format!("{n} user + {results}"),
        };
        let before = self.run.len();
        let input = cap_tail(self.run);
        let cut = self.run_cut + before - input.len();
        let parts = self
            .run_parts
            .iter()
            .map(|(index, at)| {
                let at = at.saturating_sub(cut).min(input.len());
                (*index, input[..at].chars().count())
            })
            .collect();
        (input, note, self.marks, parts)
    }
}

impl<'de> Deserialize<'de> for Recent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Walk;
        impl<'de> Visitor<'de> for Walk {
            type Value = Recent;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a message list")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Recent, A::Error> {
                let mut recent = RecentText::default();
                // Each message whole, for its hash, then as the fields
                // IN reads. One message is held at a time.
                while let Some(value) = seq.next_element::<serde_json::Value>()? {
                    let hash = message_hasher().hash_one(value.to_string());
                    let message = MessageJson::deserialize(value).map_err(de::Error::custom)?;
                    recent.add(hash, message);
                }
                Ok(Recent(recent.seen.then_some(recent)))
            }
            fn visit_unit<E: de::Error>(self) -> Result<Recent, E> {
                Ok(Recent(None))
            }
        }
        deserializer.deserialize_any(Walk)
    }
}

#[derive(Deserialize)]
struct MessageJson {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<ContentJson>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ContentJson {
    Text(String),
    Parts(Vec<PartJson>),
    Other(IgnoredAny),
}

impl ContentJson {
    /// Plain text, or the text parts joined by newlines.
    fn text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Parts(parts) => parts
                .into_iter()
                .filter_map(|part| part.text)
                .collect::<Vec<_>>()
                .join("\n"),
            Self::Other(_) => String::new(),
        }
    }
}

#[derive(Deserialize)]
struct PartJson {
    #[serde(default)]
    text: Option<String>,
}

/// Content, reasoning and tool calls gathered from a response.
#[derive(Default)]
struct Answer {
    content: String,
    reasoning: String,
    calls: Vec<(String, String)>,
}

impl Answer {
    fn add(&mut self, piece: PieceJson) {
        if let Some(text) = piece.content.or(piece.text) {
            self.content.push_str(&text);
        }
        if let Some(text) = piece.reasoning_content {
            self.reasoning.push_str(&text);
        }
        for call in piece.tool_calls.unwrap_or_default() {
            // A stream sends each call's pieces under its index; a whole
            // message lists them in order.
            let index = call.index.unwrap_or(self.calls.len()).min(MAX_CALLS - 1);
            while self.calls.len() <= index {
                self.calls.push((String::new(), String::new()));
            }
            let function = call.function.unwrap_or_default();
            let (name, args) = &mut self.calls[index];
            if let Some(part) = function.name {
                name.push_str(&part);
            }
            if let Some(part) = function.arguments {
                args.push_str(&part);
            }
        }
    }

    fn finish(self) -> String {
        let mut out = if self.content.trim().is_empty() {
            self.reasoning
        } else {
            self.content
        };
        for (name, args) in self.calls {
            if name.is_empty() && args.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&format!("[call {name}] {args}"));
        }
        out
    }
}

/// The assistant's answer from a JSON or SSE response body.
fn response_text(body: &[u8]) -> String {
    let text = String::from_utf8_lossy(body);
    let mut answer = Answer::default();
    if text.trim_start().starts_with("data:") || text.contains("\ndata:") {
        for line in text.lines() {
            let Some(data) = line.trim_start().strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            if let Ok(chunk) = serde_json::from_str::<ResponseJson>(data) {
                for choice in chunk.choices.into_iter().take(1) {
                    answer.choice(choice);
                }
            }
        }
    } else if let Ok(response) = serde_json::from_str::<ResponseJson>(&text) {
        for choice in response.choices.into_iter().take(1) {
            answer.choice(choice);
        }
    }
    answer.finish()
}

impl Answer {
    fn choice(&mut self, choice: ChoiceJson) {
        if let Some(piece) = choice.delta.or(choice.message) {
            self.add(piece);
        }
        if let Some(text) = choice.text {
            self.content.push_str(&text);
        }
    }
}

#[derive(Deserialize)]
struct ResponseJson {
    #[serde(default)]
    choices: Vec<ChoiceJson>,
}

#[derive(Deserialize)]
struct ChoiceJson {
    #[serde(default)]
    delta: Option<PieceJson>,
    #[serde(default)]
    message: Option<PieceJson>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
struct PieceJson {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallJson>>,
}

#[derive(Deserialize)]
struct ToolCallJson {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    function: Option<FunctionJson>,
}

#[derive(Default, Deserialize)]
struct FunctionJson {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Standard base64 with optional padding; whitespace is skipped. `None` for
/// any other byte, or data after padding.
fn base64_decode(text: &str) -> Option<Vec<u8>> {
    fn value(byte: u8) -> Option<u32> {
        match byte {
            b'A'..=b'Z' => Some(u32::from(byte - b'A')),
            b'a'..=b'z' => Some(u32::from(byte - b'a') + 26),
            b'0'..=b'9' => Some(u32::from(byte - b'0') + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut acc = 0u32;
    let mut bits = 0u32;
    let mut padding = false;
    for byte in text.bytes() {
        if byte.is_ascii_whitespace() {
            continue;
        }
        if byte == b'=' {
            padding = true;
            continue;
        }
        if padding {
            return None;
        }
        acc = (acc << 6) | value(byte)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// Standard base64 with padding, for tests that build captures.
#[must_use]
pub fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let word = (u32::from(chunk[0]) << 16)
            | (u32::from(chunk.get(1).copied().unwrap_or(0)) << 8)
            | u32::from(chunk.get(2).copied().unwrap_or(0));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(char::from(ALPHABET[((word >> (18 - 6 * i)) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture(req: &str, resp: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "id": 7,
            "req_path": "/v1/chat/completions",
            "req_headers": {"X-Session-Id": "INVENTED-SESSION", "Authorization": "Bearer INVENTED"},
            "req_body": base64_encode(req.as_bytes()),
            "resp_headers": {"Content-Type": "application/json"},
            "resp_body": base64_encode(resp.as_bytes()),
        }))
        .expect("json")
    }

    /// A chat request of `messages` (JSON objects), its capture parsed.
    fn chat(messages: &[serde_json::Value]) -> CaptureText {
        let req = serde_json::json!({"model": "m", "messages": messages}).to_string();
        parse_capture(&capture(&req, "{}")).expect("capture")
    }

    fn msg(role: &str, content: &str) -> serde_json::Value {
        serde_json::json!({"role": role, "content": content})
    }

    fn call(id: &str) -> serde_json::Value {
        serde_json::json!({"role": "assistant", "content": "ASSISTANT-CALL-TEXT",
            "tool_calls": [{"id": id, "type": "function",
                "function": {"name": "read", "arguments": "{}"}}]})
    }

    /// An agent's tool loop, as the captures show it (DER-451): each turn
    /// is the last one plus an assistant tool call (empty content, shown in
    /// OUT already) and its tool result. Another conversation's request in
    /// between does not hide this one's last request.
    #[test]
    fn a_tool_loop_turn_appends_only_its_tool_result() {
        let base = vec![
            msg("system", "invented system prompt"),
            msg("user", "invented task"),
            call("c1"),
            msg("tool", "first invented result"),
        ];
        let first = chat(&base);
        let other = chat(&[msg("system", "another"), msg("user", "summarise")]);
        let mut next = base.clone();
        next.extend([call("c2"), msg("tool", "second invented result")]);
        let second = chat(&next);
        assert_eq!(second.input, "[tool] second invented result");
        // The diff ends at the appended assistant call (the tool calls
        // differ only in their id), which IN does not show: the new part
        // is the tool result alone, none of the call's text.
        let common = first
            .messages
            .iter()
            .zip(&second.messages)
            .take_while(|(a, b)| a == b)
            .count();
        assert_eq!(common, base.len());
        let appended: Vec<bool> = second.messages[common..]
            .iter()
            .map(|m| m.in_role)
            .collect();
        assert_eq!(
            appended,
            [false, true],
            "an assistant call, then a tool result"
        );
        assert!(!second.input.contains("ASSISTANT-CALL-TEXT"));
        let kept = [first.messages.clone(), other.messages.clone()];
        let new = new_input_chars(kept.iter().map(Vec::as_slice), &second);
        assert_eq!(
            new,
            Some("[tool] second invented result".len()),
            "the result alone"
        );
        // Two results in one turn, then a third appended to the same run.
        let mut two = next.clone();
        two.extend([call("c3"), msg("tool", "a"), msg("tool", "b")]);
        let two_text = chat(&two);
        let mut three = two.clone();
        three.push(msg("tool", "c"));
        let three_text = chat(&three);
        assert_eq!(three_text.input, "[tool] a\n[tool] b\n[tool] c");
        let new = new_input_chars([two_text.messages.as_slice()], &three_text);
        assert_eq!(new, Some("[tool] c".len()), "only the appended result");
        // The same request again: nothing new.
        let again = new_input_chars([three_text.messages.as_slice()], &chat(&three));
        assert_eq!(again, Some(0));
    }

    /// A new conversation shares no prefix: all of IN is new. With no
    /// earlier request, or a completions prompt, the diff has no answer.
    #[test]
    fn a_new_conversation_is_all_new_and_no_history_is_unknown() {
        let earlier = chat(&[msg("system", "s"), msg("user", "old task"), call("c1")]);
        let fresh = chat(&[msg("system", "s2"), msg("user", "a brand new task")]);
        assert_eq!(
            new_input_chars([earlier.messages.as_slice()], &fresh),
            Some("a brand new task".len())
        );
        assert_eq!(new_input_chars(std::iter::empty(), &fresh), None);
        let prompt = parse_capture(&capture(r#"{"prompt":"completion text"}"#, "{}")).unwrap();
        assert!(prompt.messages.is_empty());
        assert_eq!(
            new_input_chars([earlier.messages.as_slice()], &prompt),
            None
        );
    }

    /// A run well past the cap, in multi-byte text: the run is cut as it
    /// is read (past four caps of bytes) and again at the end, and each
    /// kept message's start still points at its own `[tool]` marker,
    /// counted in characters, not bytes. Messages cut away start at 0.
    #[test]
    fn a_run_past_the_cap_keeps_its_message_starts_in_characters() {
        const RESULTS: usize = 20;
        let body = |index: usize| format!("m{index} {}", "\u{e9}\u{4e16}".repeat(2_000));
        let mut messages = vec![msg("system", "s"), msg("user", "task"), call("c1")];
        messages.extend((0..RESULTS).map(|index| msg("tool", &body(index))));
        let read: usize = (0..RESULTS).map(|index| body(index).len() + 8).sum();
        assert!(read > 4 * INPUT_CAP_CHARS, "the run is cut while read");
        let text = chat(&messages);
        assert_eq!(text.input.chars().count(), INPUT_CAP_CHARS);
        assert!(text.input.len() > INPUT_CAP_CHARS, "multi-byte");
        assert_eq!(text.input_parts.len(), RESULTS);
        let chars: Vec<char> = text.input.chars().collect();
        let kept: Vec<_> = text.input_parts.iter().filter(|(_, at)| *at > 0).collect();
        assert!(kept.len() >= 7 && kept.len() < RESULTS, "{}", kept.len());
        for (index, at) in &kept {
            let head: String = chars[*at..].iter().take(12).collect();
            let expect = format!("[tool] m{} ", index - 3);
            assert!(
                head.starts_with(&expect),
                "{head:?} at {at} is not {expect:?}"
            );
        }
        // The last result appended to the previous request's run: exactly
        // its own characters are new.
        let earlier = chat(&messages[..messages.len() - 1]);
        let new = new_input_chars([earlier.messages.as_slice()], &text);
        assert_eq!(
            new,
            Some("[tool] ".len() + body(RESULTS - 1).chars().count())
        );
    }

    #[test]
    fn shown_new_scales_onto_the_cells_drawn() {
        assert_eq!(shown_new(300, 2_000, 8_192, 2_000), 300);
        assert_eq!(shown_new(300, 20_000, 8_192, 8_192), 300);
        assert_eq!(shown_new(9_000, 20_000, 8_192, 8_000), 8_000);
        assert_eq!(shown_new(1, 1_000, 1_000, 10), 1, "anything new shows");
        assert_eq!(shown_new(0, 1_000, 1_000, 10), 0);
    }

    #[test]
    fn base64_round_trips_and_rejects_junk() {
        for text in ["", "a", "ab", "abc", "abcd", "h\u{e9}llo w\u{f6}rld\n"] {
            assert_eq!(
                base64_decode(&base64_encode(text.as_bytes())).as_deref(),
                Some(text.as_bytes())
            );
        }
        assert_eq!(base64_decode("aGk=\n").as_deref(), Some(&b"hi"[..]));
        assert_eq!(base64_decode("a$b"), None);
        assert_eq!(base64_decode("aG=k"), None);
    }

    #[test]
    fn a_json_chat_exchange_gives_the_last_user_message_and_the_answer() {
        let req = r#"{"model":"flash","messages":[
            {"role":"system","content":"You are a test fixture."},
            {"role":"user","content":"First invented question?"},
            {"role":"assistant","content":"First invented answer."},
            {"role":"tool","content":"invented tool output"},
            {"role":"user","content":[{"type":"text","text":"Second invented"},{"type":"image_url","image_url":{"url":"x"}},{"type":"text","text":"question?"}]}
        ],"stream":false}"#;
        let resp = r#"{"id":"c1","choices":[{"index":0,"message":{"role":"assistant","content":"An invented reply.","reasoning_content":"invented thinking"}}],"usage":{"prompt_tokens":9}}"#;
        let text = parse_capture(&capture(req, resp)).expect("capture");
        assert_eq!(text.input, "Second invented\nquestion?");
        assert_eq!(text.output, "An invented reply.");
        let blob = format!("{text:?}");
        assert!(
            !blob.contains("SESSION") && !blob.contains("Bearer"),
            "{blob}"
        );
    }

    #[test]
    fn an_sse_stream_is_joined_and_tool_calls_follow() {
        let req = r#"{"messages":[{"role":"user","content":"Invented streaming question"}],"stream":true}"#;
        let resp = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"invented thought\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Invented \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"streamed answer.\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"pa\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"th\\\":1}\"}}]}}]}\n\n",
            "data: not json\n\n",
            "data: [DONE]\n\n",
        );
        let text = parse_capture(&capture(req, resp)).expect("capture");
        assert_eq!(text.input, "Invented streaming question");
        assert_eq!(
            text.output,
            "Invented streamed answer.\n[call read] {\"path\":1}"
        );
    }

    #[test]
    fn reasoning_stands_in_for_empty_content_and_odd_bodies_are_empty() {
        let resp = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"only invented thinking\"}}]}\n\ndata: [DONE]\n";
        let text = parse_capture(&capture(r#"{"prompt":"invented completion prompt"}"#, resp))
            .expect("capture");
        assert_eq!(text.input, "invented completion prompt");
        assert_eq!(text.output, "only invented thinking");
        let text = parse_capture(&capture("not json", "<html>")).expect("capture");
        assert_eq!(text, CaptureText::default());
        let text = parse_capture(&capture(r#"{"messages":null}"#, "{}")).expect("capture");
        assert_eq!(text, CaptureText::default());
        // A huge tool-call index is clamped, not allocated.
        let resp = r#"{"choices":[{"message":{"tool_calls":[{"index":999999999,"function":{"name":"x","arguments":"{}"}}]}}]}"#;
        let text = parse_capture(&capture("{}", resp)).expect("capture");
        assert_eq!(text.output, "[call x] {}");
        assert_eq!(parse_capture(b"[]"), None);
        assert_eq!(parse_capture(b"{}"), None);
        assert_eq!(
            parse_capture(br#"{"req_body":"%%%","resp_body":""}"#),
            None,
            "a body that is not base64"
        );
    }

    fn input_of(req: &str) -> (String, String) {
        let text = parse_capture(&capture(req, "{}")).expect("capture");
        (text.input, text.input_note)
    }

    /// #38: an agent's tool loop never sends a new user message; IN is the
    /// tool results after the last assistant message, newest last.
    #[test]
    fn a_tool_loop_shows_the_tool_results_after_the_last_assistant_message() {
        let req = r#"{"messages":[
            {"role":"system","content":"You are an invented agent."},
            {"role":"user","content":"Invented task: tidy the fixture."},
            {"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"ls","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"a","content":"invented listing"},
            {"role":"assistant","content":"Reading three files.","tool_calls":[]},
            {"role":"tool","tool_call_id":"b","content":"invented file one"},
            {"role":"tool","tool_call_id":"c","content":[{"type":"text","text":"invented file"},{"type":"text","text":"two"}]},
            {"role":"tool","tool_call_id":"d","content":""}
        ]}"#;
        assert_eq!(
            input_of(req),
            (
                "[tool] invented file one\n[tool] invented file\ntwo\n[tool]".to_owned(),
                "3 tool results".to_owned()
            )
        );
        let one = r#"{"messages":[{"role":"user","content":"Invented task."},
            {"role":"assistant","content":"ok"},{"role":"tool","content":"invented result"}]}"#;
        assert_eq!(
            input_of(one),
            (
                "[tool] invented result".to_owned(),
                "1 tool result".to_owned()
            )
        );
    }

    #[test]
    fn user_text_and_tool_results_after_an_assistant_message_are_joined_in_order() {
        let req = r#"{"messages":[
            {"role":"user","content":"Invented first task."},
            {"role":"assistant","content":"Invented plan."},
            {"role":"user","content":[{"type":"text","text":"Invented nudge."}]},
            {"role":"tool","content":"invented result A"},
            {"role":"tool","content":"invented result B"}
        ]}"#;
        assert_eq!(
            input_of(req),
            (
                "Invented nudge.\n[tool] invented result A\n[tool] invented result B".to_owned(),
                "user + 2 tool results".to_owned()
            )
        );
    }

    /// A plain chat (last message `user`) is unchanged by #38: that message
    /// alone, no note, even after tool results.
    #[test]
    fn a_plain_chat_shows_only_the_last_user_message() {
        let req = r#"{"messages":[
            {"role":"system","content":"Invented system."},
            {"role":"user","content":"Invented hello."},
            {"role":"assistant","content":"Invented hi."},
            {"role":"user","content":"Invented follow-up?"}
        ]}"#;
        assert_eq!(
            input_of(req),
            ("Invented follow-up?".to_owned(), String::new())
        );
        // A prefill (last message assistant) keeps the last user message.
        let req = r#"{"messages":[{"role":"user","content":"Invented ask."},
            {"role":"tool","content":"invented result"},
            {"role":"assistant","content":"Invented prefix"}]}"#;
        assert_eq!(input_of(req), ("Invented ask.".to_owned(), String::new()));
        // Messages without a user or tool one fall back to the prompt.
        let req = r#"{"messages":[{"role":"system","content":"x"}],"prompt":"invented prompt"}"#;
        assert_eq!(input_of(req), ("invented prompt".to_owned(), String::new()));
    }

    /// With no assistant message, the run is every message but the system
    /// (and developer) ones.
    #[test]
    fn without_an_assistant_message_everything_after_the_system_message_is_the_run() {
        let req = r#"{"messages":[
            {"role":"system","content":"Invented system."},
            {"role":"developer","content":"Invented developer note."},
            {"role":"user","content":"Invented task."},
            {"role":"tool","content":"invented result"}
        ]}"#;
        assert_eq!(
            input_of(req),
            (
                "Invented task.\n[tool] invented result".to_owned(),
                "user + 1 tool result".to_owned()
            )
        );
    }

    /// A long run keeps its newest text, at most [`INPUT_CAP_CHARS`].
    #[test]
    fn a_long_tool_run_keeps_its_tail() {
        let big = "x".repeat(INPUT_CAP_CHARS * 3);
        let req = serde_json::json!({"messages": [
            {"role": "assistant", "content": "Invented."},
            {"role": "tool", "content": big},
            {"role": "tool", "content": big},
            {"role": "tool", "content": "INVENTED-NEWEST"},
        ]})
        .to_string();
        let (input, note) = input_of(&req);
        assert_eq!(note, "3 tool results");
        assert_eq!(input.chars().count(), INPUT_CAP_CHARS);
        assert!(
            input.ends_with("x\n[tool] INVENTED-NEWEST"),
            "{}",
            &input[input.len() - 40..]
        );
        let req = serde_json::json!({"messages": [{"role": "user", "content": big}]}).to_string();
        assert_eq!(input_of(&req).0.chars().count(), INPUT_CAP_CHARS);
    }
}
