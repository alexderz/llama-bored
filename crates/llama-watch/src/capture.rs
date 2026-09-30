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
//! stored. From the request, IN is the last `user` message of an OpenAI chat
//! body (text parts joined), or a completions `prompt`. From the response,
//! OUT is the assistant text: a JSON body's `message.content`, or an SSE
//! stream's `delta.content` pieces joined; reasoning text when there is no
//! content; tool calls appended as `[call name] arguments`. The caller keeps
//! only the tails, through the console sanitiser.
//!
//! The poller reads at most [`CAPTURE_CAP`] bytes: a capture of a 160k-token
//! agent prompt measured 0.84 MiB (bodies grow by a third in base64), so
//! 2 MiB covers a 200k context. A larger capture is skipped, not truncated
//! (a cut JSON document does not parse).

use std::fmt;

use serde::de::{self, IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

/// Largest capture body the poller reads.
pub const CAPTURE_CAP: usize = 2 * 1024 * 1024;

/// Tool calls kept from one response.
const MAX_CALLS: usize = 16;

/// The text of one finished exchange, before tail cutting and sanitising.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaptureText {
    /// The last user message.
    pub input: String,
    /// The assistant's answer.
    pub output: String,
}

/// Parse one capture. `None` when it is not a capture object or a body is
/// not base64.
#[must_use]
pub fn parse_capture(bytes: &[u8]) -> Option<CaptureText> {
    let capture: CaptureJson = serde_json::from_slice(bytes).ok()?;
    let request = base64_decode(&capture.req_body)?;
    let response = base64_decode(&capture.resp_body)?;
    drop(capture);
    Some(CaptureText {
        input: request_text(&request),
        output: response_text(&response),
    })
}

#[derive(Deserialize)]
struct CaptureJson {
    req_body: String,
    resp_body: String,
}

/// The last user message of a chat request, or a completions prompt.
fn request_text(body: &[u8]) -> String {
    let Ok(request) = serde_json::from_slice::<RequestJson>(body) else {
        return String::new();
    };
    match (request.messages.0, request.prompt) {
        (Some(text), _) | (None, Some(PromptJson::One(text))) => text,
        (None, Some(PromptJson::Many(texts))) => texts.last().cloned().unwrap_or_default(),
        (None, _) => String::new(),
    }
}

#[derive(Deserialize)]
struct RequestJson {
    #[serde(default)]
    messages: LastUser,
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

/// The text of the last `user` message. Messages are read one at a time and
/// dropped, so a long history is never held whole.
#[derive(Default)]
struct LastUser(Option<String>);

impl<'de> Deserialize<'de> for LastUser {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Walk;
        impl<'de> Visitor<'de> for Walk {
            type Value = LastUser;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a message list")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<LastUser, A::Error> {
                let mut last = None;
                while let Some(message) = seq.next_element::<MessageJson>()? {
                    if message.role.as_deref() == Some("user") {
                        last = Some(message.content.map(ContentJson::text).unwrap_or_default());
                    }
                }
                Ok(LastUser(last))
            }
            fn visit_unit<E: de::Error>(self) -> Result<LastUser, E> {
                Ok(LastUser(None))
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
}
