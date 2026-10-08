//! llama-swap `GET /api/events` (#80): the requests in flight, for every
//! engine, as llama-swap itself tracks them.
//!
//! The endpoint is Server-Sent Events: `event:message` and one
//! `data:{"type":…,"data":"<json string>"}` line per frame, a blank line
//! after each. On connect llama-swap sends a snapshot (`modelStatus`,
//! `uiConfig`, `profileChanged`, `inflight`) and, on v256, its whole log
//! history as `logData` frames (about 100 KB); after that a frame for each
//! change, and a `logData` frame for every log line.
//!
//! Only `inflight` frames are read (llama-swap's `InFlightRequestsEvent`):
//! `snapshot` with `requests[]`, `upsert` with `request`, `remove` with
//! `id`. Upserts come at most every 250 ms per request while bytes flow.
//! Of each entry only `id`, `model`, `resp_bytes` and `elapsed_ms` are
//! kept; `req_headers`, `resp_headers`, `remote_ip`, `metadata` and the
//! path are skipped unread. It carries no tokens.
//!
//! This is llama-swap's own API, like `/running`: it names no model path
//! and cannot make llama-swap load one (#70). One long-lived connection on
//! its own thread with its own agent ([`super::llamaswap::new_agent`]: no
//! proxy, no redirect), so a slow or silent stream never holds the poller,
//! which only copies [`EventsView`] out under a lock. Every line is capped
//! at [`LINE_CAP`] bytes and every frame at [`FRAME_CAP`]: a longer one is
//! read through and dropped, never kept. At most [`MAX_REQUESTS`] requests
//! are held. A session ends after [`SESSION`] (the body read's deadline)
//! and reconnects at once, the next snapshot replacing what was held; a
//! failed connect, a non-200 answer or a stream llama-swap closed waits
//! [`MIN_BACKOFF`], doubling to [`MAX_BACKOFF`] (a session that lasted
//! longer than that starts over from [`MIN_BACKOFF`]). Dropping [`Events`] stops the thread at its next read
//! or wait; it is never joined, so a stream that stays silent cannot stall
//! the poller's exit either.

use std::collections::VecDeque;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use llama_core::names::sanitize;
use serde::Deserialize;

use crate::activity::model_key;

/// Longest SSE line kept, bytes. A longer one (v256's log history) is read
/// through and dropped.
pub const LINE_CAP: usize = 256 * 1024;
/// Longest frame (its `data` lines joined) kept, bytes.
pub const FRAME_CAP: usize = 256 * 1024;
/// Most requests in flight held.
pub const MAX_REQUESTS: usize = 64;
/// Most recently ended requests remembered, for the token attribution of
/// the poll window they ended in.
pub const MAX_ENDED: usize = 64;
/// How long an ended request is remembered.
pub const ENDED_TTL: Duration = Duration::from_secs(10);
/// One connection's longest life; it reconnects at once after.
pub const SESSION: Duration = Duration::from_secs(120);
/// First wait after a failed connect.
pub const MIN_BACKOFF: Duration = Duration::from_secs(1);
/// Longest wait between connects.
pub const MAX_BACKOFF: Duration = Duration::from_secs(30);
/// Longest request id kept, characters (llama-swap's are decimal).
const ID_CHARS: usize = 24;
/// Read size.
const CHUNK: usize = 8 * 1024;

/// One request llama-swap has in flight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InflightRequest {
    /// llama-swap's in-flight id: its own counter, not the activity id.
    pub id: String,
    /// [`model_key`] of the model it names.
    pub model: String,
    /// When it started: when its frame came, less its `elapsed_ms`.
    pub started: Instant,
    /// The same, on the wall clock, for RECENT's TIME.
    pub started_wall: SystemTime,
    /// Response bytes written so far: 0 until the first byte streams.
    pub resp_bytes: u64,
}

/// A request that left llama-swap's in-flight list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EndedRequest {
    /// Its in-flight id.
    pub id: String,
    /// [`model_key`] of its model.
    pub model: String,
    /// When it started.
    pub started: Instant,
    /// When its `remove` frame came.
    pub ended: Instant,
}

/// What the stream says now.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EventsView {
    /// A session is open and its snapshot was read: the lists are whole.
    pub connected: bool,
    /// The requests in flight, oldest first.
    pub requests: Vec<InflightRequest>,
    /// Requests that ended in the last [`ENDED_TTL`], oldest first.
    pub ended: Vec<EndedRequest>,
    /// Why the last connect failed, for the poller's log; `None` while a
    /// session runs.
    pub failure: Option<&'static str>,
}

/// Splits an SSE byte stream into frames' `data`, with [`LINE_CAP`] and
/// [`FRAME_CAP`]. Field names other than `data` are ignored, as are
/// comment lines (`:`).
#[derive(Debug, Default)]
pub struct SseParser {
    line: Vec<u8>,
    /// The current line passed [`LINE_CAP`]: dropped at its end.
    line_over: bool,
    data: Vec<u8>,
    /// The current frame passed [`FRAME_CAP`] or had a dropped line.
    frame_over: bool,
    /// Data lines in the current frame.
    data_lines: usize,
}

impl SseParser {
    /// Feed bytes; `frame` gets each complete frame's data that stayed
    /// within the caps.
    pub fn feed(&mut self, bytes: &[u8], mut frame: impl FnMut(&[u8])) {
        for &byte in bytes {
            if byte == b'\n' {
                self.end_line(&mut frame);
                continue;
            }
            if self.line_over {
                continue;
            }
            if self.line.len() >= LINE_CAP {
                self.line_over = true;
                self.line.clear();
                continue;
            }
            self.line.push(byte);
        }
    }

    fn end_line(&mut self, frame: &mut impl FnMut(&[u8])) {
        let over = std::mem::take(&mut self.line_over);
        let mut line = std::mem::take(&mut self.line);
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        if over {
            // A dropped line poisons its frame: half a frame is no frame.
            self.frame_over = true;
            self.line = line;
            self.line.clear();
            return;
        }
        if line.is_empty() {
            if !self.frame_over && self.data_lines > 0 {
                frame(&self.data);
            }
            self.data.clear();
            self.frame_over = false;
            self.data_lines = 0;
        } else if let Some(rest) = line.strip_prefix(b"data:") {
            let rest = rest.strip_prefix(b" ").unwrap_or(rest);
            if self.data_lines > 0 {
                self.data.push(b'\n');
            }
            if self.data.len() + rest.len() > FRAME_CAP {
                self.frame_over = true;
                self.data.clear();
            } else if !self.frame_over {
                self.data.extend_from_slice(rest);
            }
            self.data_lines += 1;
        }
        // Keep the allocation for the next line.
        self.line = line;
        self.line.clear();
    }
}

/// The in-flight list one stream has built.
#[derive(Debug, Default)]
pub struct Inflight {
    requests: Vec<InflightRequest>,
    ended: VecDeque<EndedRequest>,
    /// An `inflight` snapshot was read this session.
    synced: bool,
}

#[derive(Deserialize)]
struct Kind {
    #[serde(rename = "type", default)]
    kind: String,
}

#[derive(Deserialize)]
struct Envelope {
    #[serde(default)]
    data: String,
}

#[derive(Deserialize)]
struct InflightEvent {
    #[serde(default)]
    operation: String,
    #[serde(default)]
    requests: Vec<Entry>,
    #[serde(default)]
    request: Option<Entry>,
    #[serde(default)]
    id: Option<String>,
}

/// One `InflightRequestEntry`; every other field is skipped unread.
#[derive(Deserialize)]
struct Entry {
    #[serde(default)]
    id: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    resp_bytes: f64,
    #[serde(default)]
    elapsed_ms: f64,
}

impl Inflight {
    /// A new session: nothing is known until its snapshot.
    pub fn new_session(&mut self) {
        self.requests.clear();
        self.synced = false;
    }

    /// Take one frame's data. Frames other than `inflight` are ignored.
    pub fn apply(&mut self, data: &[u8], now: Instant, wall: SystemTime) {
        let Ok(kind) = serde_json::from_slice::<Kind>(data) else {
            return;
        };
        if kind.kind != "inflight" {
            return;
        }
        let Ok(envelope) = serde_json::from_slice::<Envelope>(data) else {
            return;
        };
        let Ok(event) = serde_json::from_str::<InflightEvent>(&envelope.data) else {
            return;
        };
        match event.operation.as_str() {
            "snapshot" => {
                let old = std::mem::take(&mut self.requests);
                for entry in event.requests.into_iter().take(MAX_REQUESTS) {
                    let keep = old.iter().find(|req| req.id == clean_id(&entry.id));
                    if let Some(req) = request(entry, keep, now, wall) {
                        self.requests.push(req);
                    }
                }
                // Requests the snapshot no longer lists ended unseen.
                for req in old {
                    if !self.requests.iter().any(|kept| kept.id == req.id) {
                        self.end(req, now);
                    }
                }
                self.synced = true;
            }
            "upsert" => {
                let Some(entry) = event.request else {
                    return;
                };
                let id = clean_id(&entry.id);
                match self.requests.iter().position(|req| req.id == id) {
                    Some(at) => {
                        if let Some(req) = request(entry, Some(&self.requests[at]), now, wall) {
                            self.requests[at] = req;
                        }
                    }
                    None if self.requests.len() < MAX_REQUESTS => {
                        if let Some(req) = request(entry, None, now, wall) {
                            self.requests.push(req);
                        }
                    }
                    None => {}
                }
            }
            "remove" => {
                let Some(id) = event.id.as_deref().map(clean_id) else {
                    return;
                };
                if let Some(at) = self.requests.iter().position(|req| req.id == id) {
                    let req = self.requests.remove(at);
                    self.end(req, now);
                }
            }
            _ => {}
        }
    }

    fn end(&mut self, req: InflightRequest, now: Instant) {
        if self.ended.len() >= MAX_ENDED {
            self.ended.pop_front();
        }
        self.ended.push_back(EndedRequest {
            id: req.id,
            model: req.model,
            started: req.started,
            ended: now,
        });
    }

    /// What is known now. Not `connected` before the session's snapshot.
    #[must_use]
    pub fn view(&mut self, now: Instant) -> EventsView {
        self.ended
            .retain(|req| now.saturating_duration_since(req.ended) < ENDED_TTL);
        let mut requests = self.requests.clone();
        requests.sort_by_key(|req| req.started);
        EventsView {
            connected: self.synced,
            requests: if self.synced { requests } else { Vec::new() },
            ended: self.ended.iter().cloned().collect(),
            failure: None,
        }
    }
}

/// An id as kept: printable ASCII, at most [`ID_CHARS`].
fn clean_id(id: &str) -> String {
    sanitize(id, ID_CHARS)
}

/// One entry as kept. A request seen before keeps its start; a new one
/// started `elapsed_ms` before its frame came. An entry with no id or no
/// model is dropped.
fn request(
    entry: Entry,
    known: Option<&InflightRequest>,
    now: Instant,
    wall: SystemTime,
) -> Option<InflightRequest> {
    let id = clean_id(&entry.id);
    if id.is_empty() || entry.model.is_empty() {
        return None;
    }
    let bytes = if entry.resp_bytes.is_finite() && entry.resp_bytes > 0.0 {
        entry.resp_bytes.min(u64::MAX as f64) as u64
    } else {
        0
    };
    let (started, started_wall) = match known {
        Some(req) => (req.started, req.started_wall),
        None => {
            let ms = if entry.elapsed_ms.is_finite() && entry.elapsed_ms > 0.0 {
                entry.elapsed_ms.min(86_400_000.0) as u64
            } else {
                0
            };
            let ago = Duration::from_millis(ms);
            (
                now.checked_sub(ago).unwrap_or(now),
                wall.checked_sub(ago).unwrap_or(wall),
            )
        }
    };
    Some(InflightRequest {
        id,
        model: model_key(&entry.model),
        started,
        started_wall,
        resp_bytes: bytes,
    })
}

/// The stream's thread. Dropping it asks the thread to stop.
pub struct Events {
    shared: Arc<Mutex<EventsView>>,
    stop: Arc<AtomicBool>,
}

impl Drop for Events {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Events {
    /// Start reading `{url}/api/events` on a thread of its own. `timeout`
    /// bounds the connect and the response head.
    pub fn spawn(url: &str, timeout: Duration) -> std::io::Result<Self> {
        let shared = Arc::new(Mutex::new(EventsView::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let endpoint = format!("{}/api/events", url.trim_end_matches('/'));
        let thread_shared = Arc::clone(&shared);
        let thread_stop = Arc::clone(&stop);
        thread::Builder::new()
            .name("llama-events".to_owned())
            .spawn(move || run(&endpoint, timeout, &thread_shared, &thread_stop))?;
        Ok(Self { shared, stop })
    }

    /// What the stream says now: a copy, under a short lock.
    #[must_use]
    pub fn view(&self) -> EventsView {
        self.shared
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone()
    }
}

fn run(endpoint: &str, timeout: Duration, shared: &Mutex<EventsView>, stop: &AtomicBool) {
    let agent = super::llamaswap::new_agent();
    let mut backoff = MIN_BACKOFF;
    let mut inflight = Inflight::default();
    while !stop.load(Ordering::Relaxed) {
        let began = Instant::now();
        let failure = session(&agent, endpoint, timeout, shared, stop, &mut inflight);
        inflight.new_session();
        publish(shared, &mut inflight, failure);
        if stop.load(Ordering::Relaxed) {
            return;
        }
        // A session that ran to its deadline reconnects at once; any other
        // end (refused, a non-200, llama-swap closing it) waits first, so
        // a server that keeps closing the stream is not hammered.
        if failure.is_none() && began.elapsed() >= SESSION {
            backoff = MIN_BACKOFF;
            continue;
        }
        if began.elapsed() > MAX_BACKOFF {
            backoff = MIN_BACKOFF;
        }
        let until = Instant::now() + backoff;
        while Instant::now() < until {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            thread::sleep(Duration::from_millis(100));
        }
        backoff = (backoff * 2).min(MAX_BACKOFF);
    }
}

/// One connection, read until it ends. `None` when it ran until its
/// deadline or llama-swap closed it after a good start.
fn session(
    agent: &ureq::Agent,
    endpoint: &str,
    timeout: Duration,
    shared: &Mutex<EventsView>,
    stop: &AtomicBool,
    inflight: &mut Inflight,
) -> Option<&'static str> {
    let response = agent
        .get(endpoint)
        .config()
        .timeout_connect(Some(timeout))
        .timeout_recv_response(Some(timeout))
        .timeout_recv_body(Some(SESSION))
        .build()
        .call();
    let response = match response {
        Ok(response) => response,
        Err(ureq::Error::Timeout(_)) => return Some("timeout"),
        Err(ureq::Error::ConnectionFailed) => return Some("connection refused"),
        Err(_) => return Some("request failed"),
    };
    if response.status().as_u16() != 200 {
        return Some("http status");
    }
    let mut reader = response.into_body().into_reader();
    let mut parser = SseParser::default();
    let mut buf = vec![0u8; CHUNK];
    loop {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        let read = match reader.read(&mut buf) {
            Ok(0) => return None,
            Ok(read) => read,
            // The session deadline, or a dropped connection.
            Err(_) => return None,
        };
        let now = Instant::now();
        let wall = SystemTime::now();
        let mut changed = false;
        parser.feed(&buf[..read], |data| {
            inflight.apply(data, now, wall);
            changed = true;
        });
        if changed {
            publish(shared, inflight, None);
        }
    }
}

fn publish(shared: &Mutex<EventsView>, inflight: &mut Inflight, failure: Option<&'static str>) {
    let mut view = inflight.view(Instant::now());
    view.failure = failure;
    *shared.lock().unwrap_or_else(|err| err.into_inner()) = view;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An `/api/events` stream shaped like a v256 capture, invented
    /// content: connect snapshot (models, ui config, profile, in-flight),
    /// log frames, then an upsert, a second request and a remove.
    fn frame(kind: &str, data: &serde_json::Value) -> String {
        let data = serde_json::to_string(data).expect("json");
        let envelope = serde_json::json!({"type": kind, "data": data});
        format!("event:message\ndata:{envelope}\n\n")
    }

    fn entry(id: &str, model: &str, bytes: u64, elapsed: u64) -> serde_json::Value {
        serde_json::json!({
            "id": id, "timestamp": "2026-10-07T10:00:00Z", "model": model,
            "req_path": "/v1/chat/completions", "method": "POST",
            "req_headers": {"User-Agent": "INVENTED-AGENT", "Authorization": "[redacted]"},
            "remote_ip": "192.0.2.7",
            "resp_headers": {"Content-Type": "text/event-stream"},
            "resp_bytes": bytes, "elapsed_ms": elapsed,
        })
    }

    fn stream() -> String {
        let mut text = String::new();
        text += &frame(
            "logData",
            &serde_json::json!({"source": "proxy", "data": "x".repeat(LINE_CAP + 10)}),
        );
        text += &frame(
            "modelStatus",
            &serde_json::json!([{"id": "m", "state": "ready"}]),
        );
        text += &frame("uiConfig", &serde_json::json!({"activity": {}}));
        text += &frame("profileChanged", &serde_json::json!({"active": null}));
        text += &frame(
            "inflight",
            &serde_json::json!({"operation": "snapshot", "requests": [entry("30", "qwen-vllm", 0, 8_751)]}),
        );
        text += &frame(
            "logData",
            &serde_json::json!({"source": "upstream", "data": "INVENTED LOG LINE\n"}),
        );
        text += &frame(
            "inflight",
            &serde_json::json!({"operation": "upsert", "request": entry("30", "qwen-vllm", 4_096, 9_001)}),
        );
        text += &frame(
            "inflight",
            &serde_json::json!({"operation": "upsert", "request": entry("31", "qwen-sglang", 0, 0)}),
        );
        text += &frame("activity", &serde_json::json!({"id": 12}));
        text += &frame(
            "inflight",
            &serde_json::json!({"operation": "remove", "id": "30"}),
        );
        text
    }

    fn run_stream(bytes: &[u8], chunk: usize) -> (Inflight, Vec<usize>) {
        let mut parser = SseParser::default();
        let mut inflight = Inflight::default();
        let t0 = Instant::now();
        let mut sizes = Vec::new();
        for (i, piece) in bytes.chunks(chunk).enumerate() {
            parser.feed(piece, |data| {
                sizes.push(data.len());
                inflight.apply(
                    data,
                    t0 + Duration::from_millis(i as u64),
                    SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000),
                );
            });
        }
        (inflight, sizes)
    }

    #[test]
    fn the_stream_gives_snapshot_upsert_and_remove() {
        let text = stream();
        for chunk in [1, 7, 4096, text.len()] {
            let (mut inflight, sizes) = run_stream(text.as_bytes(), chunk);
            // The oversized log frame never reaches the parser's caller.
            assert!(sizes.iter().all(|n| *n <= FRAME_CAP), "{sizes:?}");
            assert_eq!(sizes.len(), 9, "every frame but the oversized one");
            let view = inflight.view(Instant::now());
            assert!(view.connected);
            assert_eq!(view.requests.len(), 1, "30 was removed");
            let req = &view.requests[0];
            assert_eq!(
                (req.id.as_str(), req.model.as_str(), req.resp_bytes),
                ("31", "qwen-sglang", 0)
            );
            assert_eq!(view.ended.len(), 1);
            assert_eq!(
                (view.ended[0].id.as_str(), view.ended[0].model.as_str()),
                ("30", "qwen-vllm")
            );
            // Nothing but the kept fields: no header, address or log text.
            let kept = format!("{view:?}");
            for secret in ["INVENTED", "192.0.2.7", "redacted", "event-stream", "chat"] {
                assert!(!kept.contains(secret), "{secret} in {kept}");
            }
        }
    }

    /// A request first seen in the snapshot started `elapsed_ms` before
    /// it; an upsert keeps that start and takes the new byte count.
    #[test]
    fn a_request_keeps_its_start_across_upserts() {
        let mut inflight = Inflight::default();
        let t0 = Instant::now() + Duration::from_secs(60);
        let wall = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
        let snap = frame(
            "inflight",
            &serde_json::json!({"operation": "snapshot", "requests": [entry("7", "m", 0, 8_751)]}),
        );
        let mut parser = SseParser::default();
        parser.feed(snap.as_bytes(), |data| inflight.apply(data, t0, wall));
        let up = frame(
            "inflight",
            &serde_json::json!({"operation": "upsert", "request": entry("7", "m", 512, 9_500)}),
        );
        parser.feed(up.as_bytes(), |data| {
            inflight.apply(data, t0 + Duration::from_secs(1), wall)
        });
        let view = inflight.view(t0);
        let req = &view.requests[0];
        assert_eq!(req.started, t0 - Duration::from_millis(8_751));
        assert_eq!(req.started_wall, wall - Duration::from_millis(8_751));
        assert_eq!(req.resp_bytes, 512);
    }

    /// Before an `inflight` snapshot nothing is listed: an upsert alone
    /// does not make the list whole. A new session starts empty again, and
    /// a snapshot that drops a request ends it.
    #[test]
    fn the_list_is_whole_only_after_a_snapshot() {
        let mut inflight = Inflight::default();
        let now = Instant::now();
        let wall = SystemTime::UNIX_EPOCH;
        let mut parser = SseParser::default();
        let up = frame(
            "inflight",
            &serde_json::json!({"operation": "upsert", "request": entry("1", "m", 0, 0)}),
        );
        parser.feed(up.as_bytes(), |data| inflight.apply(data, now, wall));
        assert!(!inflight.view(now).connected);
        assert!(inflight.view(now).requests.is_empty());
        let snap = frame(
            "inflight",
            &serde_json::json!({"operation": "snapshot", "requests": [entry("2", "m", 0, 0)]}),
        );
        parser.feed(snap.as_bytes(), |data| inflight.apply(data, now, wall));
        let view = inflight.view(now);
        assert_eq!(
            view.requests
                .iter()
                .map(|r| r.id.as_str())
                .collect::<Vec<_>>(),
            vec!["2"]
        );
        assert_eq!(
            view.ended.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            vec!["1"]
        );
        inflight.new_session();
        assert!(!inflight.view(now).connected);
        // Ended requests age out.
        assert!(inflight.view(now + ENDED_TTL).ended.is_empty());
    }

    /// Caps: a line past [`LINE_CAP`] drops its frame and only that frame;
    /// a frame of many data lines past [`FRAME_CAP`] too; at most
    /// [`MAX_REQUESTS`] are held; malformed JSON is ignored.
    #[test]
    fn caps_drop_only_the_oversized_frame() {
        let mut parser = SseParser::default();
        let mut frames = Vec::new();
        let long = "a".repeat(LINE_CAP + 1);
        parser.feed(format!("data:{long}\n\ndata:ok\n\n").as_bytes(), |d| {
            frames.push(d.to_vec())
        });
        assert_eq!(frames, vec![b"ok".to_vec()]);
        frames.clear();
        let line = "b".repeat(LINE_CAP / 2);
        parser.feed(
            format!("data:{line}\ndata:{line}\ndata:{line}\n\ndata:x\r\n: comment\r\n\r\n")
                .as_bytes(),
            |d| frames.push(d.to_vec()),
        );
        assert_eq!(frames, vec![b"x".to_vec()]);

        let mut inflight = Inflight::default();
        let many: Vec<_> = (0..MAX_REQUESTS + 10)
            .map(|i| entry(&i.to_string(), "m", 0, 0))
            .collect();
        let snap = frame(
            "inflight",
            &serde_json::json!({"operation": "snapshot", "requests": many}),
        );
        let now = Instant::now();
        parser.feed(snap.as_bytes(), |data| {
            inflight.apply(data, now, SystemTime::UNIX_EPOCH)
        });
        let extra = frame(
            "inflight",
            &serde_json::json!({"operation": "upsert", "request": entry("999", "m", 0, 0)}),
        );
        parser.feed(extra.as_bytes(), |data| {
            inflight.apply(data, now, SystemTime::UNIX_EPOCH)
        });
        assert_eq!(inflight.view(now).requests.len(), MAX_REQUESTS);
        for junk in [&b"{"[..], b"[]", br#"{"type":"inflight","data":"{not json"}"#, br#"{"type":"inflight","data":"{\"operation\":\"upsert\",\"request\":{\"id\":\"\",\"model\":\"m\"}}"}"#] {
            inflight.apply(junk, now, SystemTime::UNIX_EPOCH);
        }
        assert_eq!(inflight.view(now).requests.len(), MAX_REQUESTS);
    }
}
