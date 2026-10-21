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
//! `inflight` frames are read (llama-swap's `InFlightRequestsEvent`):
//! `snapshot` with `requests[]`, `upsert` with `request`, `remove` with
//! `id`. Upserts come at most every 250 ms per request while bytes flow.
//! Of each entry only `id`, `model`, `resp_bytes` and `elapsed_ms` are
//! kept; `req_headers`, `resp_headers`, `remote_ip`, `metadata` and the
//! path are skipped unread. It carries no tokens.
//!
//! A llama-swap build with model claims also sends `claims` frames on this
//! same stream (#128): the whole list on connect and again after every
//! change. Only the `claims` array is kept; `changes` and `at` are
//! ignored. `"claims": null` is an empty list. An entry without a usable
//! `model` or `holder` is skipped, not the list. Two claims on the same
//! model are both kept; the header draws the first. A frame whose `data`
//! does not parse leaves the previous list in place. The list is forgotten
//! when the stream reconnects, until the first `claims` frame of the new
//! session. A claim's `conditional` marks a guest under someone else's GPU
//! reservation (#167).
//!
//! The same build sends a `gpu` frame (#167): the GPU reservation after
//! each change, `null` when there is none, and once on connect right after
//! the `claims` frame. Only `gpu` is kept; `changes` and `at` are ignored.
//! A reservation without a usable `holder` is none; a `mode` other than
//! `external` and a `state` other than `draining` add nothing. A frame
//! whose `data` does not parse leaves the previous reservation in place.
//! The reservation is forgotten when the stream reconnects, as the claims
//! are.
//!
//! Displaced notes (#167) come from `claims` frames: each `displaced` change
//! with the frame's `at`, one note per former holder, newest wins, as
//! llama-swap keeps them. They outlive a reconnect, since each carries its
//! own time, and are shown for [`DISPLACED_TTL`]; a note whose former holder
//! has a claim again is not shown, as llama-swap drops it then.
//!
//! The connect snapshot carries the reservation and every claim, but not
//! the displaced notes. So each session starts with one `GET /api/gpu`
//! before the stream: its `displaced` map is merged into the notes, and its
//! `gpu` and `guests` stand in until the session's first `gpu` and `claims`
//! frames (a build that never sends them keeps those). Its other members
//! are ignored. A failed or malformed answer changes nothing and is not
//! logged: a llama-swap without reservations answers 404.
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
use crate::localtime::parse_rfc3339;
use crate::setup_rules::text_held;

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
/// Most model claims kept (first [`MAX_CLAIMS`] of a frame).
pub const MAX_CLAIMS: usize = 16;
/// Longest llama-swap model id kept on a claim, bytes.
const CLAIM_MODEL_BYTES: usize = 128;
/// Longest holder kept before text sanitising, bytes.
const CLAIM_HOLDER_BYTES: usize = 64;
/// Longest hold reason kept before text sanitising, bytes.
const CLAIM_REASON_BYTES: usize = 256;
/// Holder as drawn: SETUP's text kind, at most this many characters.
const CLAIM_HOLDER_CHARS: usize = 24;
/// Hold reason as drawn: SETUP's text kind, at most this many characters.
const CLAIM_REASON_CHARS: usize = 40;
/// How long a displaced note is shown after its time (#167).
pub const DISPLACED_TTL: Duration = Duration::from_secs(10 * 60);
/// Most displaced notes kept, newest first.
pub const MAX_DISPLACED: usize = 8;
/// Two notes of one holder with the same model, `to` and `by` this close
/// in time are one event seen twice: from a `claims` frame and from
/// `GET /api/gpu`, whose times differ by a moment.
const NOTE_SAME_WITHIN: Duration = Duration::from_secs(5);
/// Largest `GET /api/gpu` body read. ureq refuses a body of this many bytes
/// or more, as for `/running`.
pub const GPU_BODY_CAP: u64 = 65_536;
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

/// One model claim from a `claims` frame (#128).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelClaim {
    /// Sanitised llama-swap model id, at most 128 bytes. Not
    /// [`model_key`]: two ids that share their first 31 characters stay
    /// distinct.
    pub model: String,
    /// Holder as SETUP's `text` kind, at most 24 characters with the
    /// ellipsis rule. A space or a non-ASCII character is kept.
    pub holder: String,
    /// An automatic claim from use, as opposed to an explicit one.
    pub auto: bool,
    /// The claimed model is serving a request.
    pub in_use: bool,
    /// When the idle claim expires. `None` while in use, or when omitted.
    pub expires_at: Option<SystemTime>,
    /// When an explicit hold ends. `Some` makes the entry a hold.
    pub hold_until: Option<SystemTime>,
    /// Free text from a caller (SETUP's `text` kind, at most 40 characters).
    /// Empty when there is none.
    pub hold_reason: String,
    /// A guest claim: someone else holds the GPU reservation, and the claim
    /// ends when that holder switches away from the model (#167).
    pub conditional: bool,
}

/// The GPU reservation from a `gpu` frame or `GET /api/gpu` (#167).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuReservation {
    /// Holder as for a claim: SETUP's `text` kind, at most 24 characters.
    pub holder: String,
    /// Mode `external`: the card is for work llama-swap does not run.
    pub external: bool,
    /// State `draining`: an external reservation whose models still stop.
    pub draining: bool,
    /// The holder has a request running or waiting.
    pub in_use: bool,
    /// The idle deadline. `None` while in use, or when omitted.
    pub expires_at: Option<SystemTime>,
    /// When an explicit hold ends.
    pub hold_until: Option<SystemTime>,
    /// As a claim's: at most 40 characters, empty when there is none.
    pub hold_reason: String,
}

/// A claim that ended because the GPU reservation's holder stopped its
/// model (#167).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisplacedNote {
    /// The claim's former holder; not drawn, it keys the note.
    pub holder: String,
    /// The model the claim was on, at the claims cap.
    pub model: String,
    /// The model that replaced it. Empty for an unload.
    pub to: String,
    /// The reservation's holder, capped as a holder.
    pub by: String,
    /// When it happened.
    pub at: SystemTime,
}

/// What `GET /api/gpu` says (#167); the members the header does not use
/// are not kept.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GpuApi {
    pub gpu: Option<GpuReservation>,
    /// Its `guests`: the conditional claims.
    pub guests: Vec<ModelClaim>,
    /// Its `displaced` map, oldest first.
    pub displaced: Vec<DisplacedNote>,
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
    /// Latest model claims, first [`MAX_CLAIMS`]. Empty until this
    /// session's first `claims` frame, and after reconnect.
    pub claims: Vec<ModelClaim>,
    /// A `claims` frame this session did not parse. The previous list is
    /// kept. The poller logs this once per session.
    pub claims_error: Option<&'static str>,
    /// The GPU reservation (#167). `None` when there is none, and after
    /// reconnect until the session's first `gpu` frame or `GET /api/gpu`.
    pub gpu: Option<GpuReservation>,
    /// A `gpu` frame this session did not parse. The previous reservation
    /// is kept. The poller logs this once per session.
    pub gpu_error: Option<&'static str>,
    /// Displaced notes of the last [`DISPLACED_TTL`] whose former holder
    /// has no claim now, newest first (#167).
    pub displaced: Vec<DisplacedNote>,
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
            ..EventsView::default()
        }
    }
}

/// The claims list one stream has built (#128), and the displaced notes
/// (#167).
#[derive(Debug, Default)]
pub struct Claims {
    list: Vec<ModelClaim>,
    /// A `claims` frame was read this session.
    synced: bool,
    /// The list is `GET /api/gpu`'s guests, until a `claims` frame.
    stand_in: bool,
    /// A `claims` frame this session did not parse.
    bad: bool,
    /// Newest first, at most [`MAX_DISPLACED`]; kept across sessions.
    displaced: Vec<DisplacedNote>,
}

#[derive(Deserialize)]
struct ClaimsEvent {
    /// `null` is an empty list (a server may encode one that way). Missing
    /// is the same. Any other JSON type fails the frame.
    #[serde(default)]
    claims: Option<Vec<serde_json::Value>>,
    /// Read for `displaced` changes only (#167). Not an array: no notes.
    #[serde(default)]
    changes: serde_json::Value,
    /// The frame's time, the time of its `displaced` changes.
    #[serde(default)]
    at: serde_json::Value,
}

impl Claims {
    /// A new session: nothing is known until its first `claims` frame. The
    /// displaced notes stay: each carries its own time.
    pub fn new_session(&mut self) {
        self.list.clear();
        self.synced = false;
        self.stand_in = false;
        self.bad = false;
    }

    /// `GET /api/gpu`'s answer (#167): its displaced notes are merged, and
    /// its guests stand in for the list until this session's first
    /// `claims` frame.
    pub fn stand_in(&mut self, api: &GpuApi) {
        for note in &api.displaced {
            self.note(note.clone());
        }
        if !self.synced {
            self.list = api.guests.iter().take(MAX_CLAIMS).cloned().collect();
            self.stand_in = true;
        }
    }

    /// Keep `note`: one per former holder, the newest. The same event seen
    /// again (within [`NOTE_SAME_WITHIN`]) keeps the time it was first seen
    /// with; a later displacement of the same shape is a new event.
    fn note(&mut self, note: DisplacedNote) {
        if let Some(at) = self.displaced.iter().position(|n| n.holder == note.holder) {
            let old = &self.displaced[at];
            let apart = old
                .at
                .duration_since(note.at)
                .or_else(|_| note.at.duration_since(old.at))
                .unwrap_or(Duration::ZERO);
            let same = old.model == note.model
                && old.to == note.to
                && old.by == note.by
                && apart <= NOTE_SAME_WITHIN;
            if same || old.at >= note.at {
                return;
            }
            self.displaced.remove(at);
        }
        let at = self
            .displaced
            .iter()
            .position(|n| n.at < note.at)
            .unwrap_or(self.displaced.len());
        self.displaced.insert(at, note);
        self.displaced.truncate(MAX_DISPLACED);
    }

    /// Notes of the last [`DISPLACED_TTL`] before `wall` whose former holder
    /// has no claim in the list, newest first. A note from the future (a
    /// clock step) is kept.
    #[must_use]
    pub fn displaced(&self, wall: SystemTime) -> Vec<DisplacedNote> {
        let (list, _) = self.view();
        self.displaced
            .iter()
            .filter(|note| {
                wall.duration_since(note.at)
                    .map_or(true, |age| age < DISPLACED_TTL)
            })
            .filter(|note| !list.iter().any(|claim| claim.holder == note.holder))
            .cloned()
            .collect()
    }

    /// Take one frame's data. Frames other than `claims` are ignored. A
    /// `claims` frame whose `data` does not parse leaves the previous list
    /// and is remembered once for the session.
    pub fn apply(&mut self, data: &[u8]) {
        let Ok(kind) = serde_json::from_slice::<Kind>(data) else {
            return;
        };
        if kind.kind != "claims" {
            return;
        }
        let Ok(envelope) = serde_json::from_slice::<Envelope>(data) else {
            self.bad = true;
            return;
        };
        let Ok(event) = serde_json::from_str::<ClaimsEvent>(&envelope.data) else {
            self.bad = true;
            return;
        };
        self.list = event
            .claims
            .unwrap_or_default()
            .into_iter()
            .filter_map(keep_claim)
            .take(MAX_CLAIMS)
            .collect();
        self.synced = true;
        self.stand_in = false;
        let at = event.at.as_str().and_then(rfc3339_time);
        if let (Some(changes), Some(at)) = (event.changes.as_array(), at) {
            for change in changes {
                if let Some(note) = displaced_change(change, at) {
                    self.note(note);
                }
            }
        }
    }

    /// What is known now. Empty before this session's first `claims` frame
    /// (or `GET /api/gpu`'s guests standing in for it).
    #[must_use]
    pub fn view(&self) -> (Vec<ModelClaim>, Option<&'static str>) {
        (
            if self.synced || self.stand_in {
                self.list.clone()
            } else {
                Vec::new()
            },
            self.bad.then_some("malformed claims frame"),
        )
    }
}

/// The GPU reservation one stream has built (#167).
#[derive(Debug, Default)]
pub struct Gpu {
    reservation: Option<GpuReservation>,
    /// A `gpu` frame was read this session.
    synced: bool,
    /// A `gpu` frame this session did not parse.
    bad: bool,
}

#[derive(Deserialize)]
struct GpuEvent {
    /// `null` or missing: no reservation.
    #[serde(default)]
    gpu: serde_json::Value,
}

impl Gpu {
    /// A new session: no reservation until its first `gpu` frame or
    /// `GET /api/gpu`.
    pub fn new_session(&mut self) {
        self.reservation = None;
        self.synced = false;
        self.bad = false;
    }

    /// Take one frame's data. Frames other than `gpu` are ignored. A `gpu`
    /// frame whose `data` does not parse leaves the previous reservation
    /// and is remembered once for the session.
    pub fn apply(&mut self, data: &[u8]) {
        let Ok(kind) = serde_json::from_slice::<Kind>(data) else {
            return;
        };
        if kind.kind != "gpu" {
            return;
        }
        let Ok(envelope) = serde_json::from_slice::<Envelope>(data) else {
            self.bad = true;
            return;
        };
        let Ok(event) = serde_json::from_str::<GpuEvent>(&envelope.data) else {
            self.bad = true;
            return;
        };
        self.reservation = keep_gpu(&event.gpu);
        self.synced = true;
    }

    /// `GET /api/gpu`'s reservation stands in until this session's first
    /// `gpu` frame.
    pub fn stand_in(&mut self, api: &GpuApi) {
        if !self.synced {
            self.reservation = api.gpu.clone();
        }
    }

    /// What is known now, and whether a frame failed this session.
    #[must_use]
    pub fn view(&self) -> (Option<GpuReservation>, Option<&'static str>) {
        (
            self.reservation.clone(),
            self.bad.then_some("malformed gpu frame"),
        )
    }
}

/// `GET /api/gpu`'s body (#167). `None` when it is not a JSON object.
/// `gpu` is kept as a `gpu` frame's; `guests` as a `claims` frame's list,
/// entries without a usable model or holder skipped; `displaced` is a map
/// by former holder, and a note without a usable model, `by` or `at` is
/// skipped. Every other member, and a member of the wrong type, is ignored.
#[must_use]
pub fn parse_gpu_api(body: &[u8]) -> Option<GpuApi> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    let obj = value.as_object()?;
    let gpu = obj.get("gpu").and_then(keep_gpu);
    let guests = obj
        .get("guests")
        .and_then(serde_json::Value::as_array)
        .map(|list| {
            list.iter()
                .cloned()
                .filter_map(keep_claim)
                .take(MAX_CLAIMS)
                .collect()
        })
        .unwrap_or_default();
    let mut displaced: Vec<DisplacedNote> = obj
        .get("displaced")
        .and_then(serde_json::Value::as_object)
        .map(|map| {
            map.iter()
                .filter_map(|(holder, note)| {
                    let at = note
                        .get("at")
                        .and_then(serde_json::Value::as_str)
                        .and_then(rfc3339_time)?;
                    displaced_note(holder, note, at)
                })
                .collect()
        })
        .unwrap_or_default();
    displaced.sort_by_key(|note| note.at);
    Some(GpuApi {
        gpu,
        guests,
        displaced,
    })
}

/// One `GPUView` as kept: `None` for `null`, a non-object, or no usable
/// holder.
fn keep_gpu(value: &serde_json::Value) -> Option<GpuReservation> {
    let obj = value.as_object()?;
    let holder = obj.get("holder").and_then(serde_json::Value::as_str)?;
    let holder = holder_held(holder)?;
    let text = |key: &str| obj.get(key).and_then(serde_json::Value::as_str);
    let flag = |key: &str| {
        obj.get(key)
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    };
    Some(GpuReservation {
        holder,
        external: text("mode") == Some("external"),
        draining: text("state") == Some("draining"),
        in_use: flag("in_use"),
        expires_at: text("expires_at").and_then(rfc3339_time),
        hold_until: text("hold_until").and_then(rfc3339_time),
        hold_reason: text("hold_reason")
            .and_then(|raw| text_held(take_bytes(raw, CLAIM_REASON_BYTES), CLAIM_REASON_CHARS))
            .unwrap_or_default(),
    })
}

/// A `claims` frame's change, as a note when its action is `displaced`.
fn displaced_change(change: &serde_json::Value, at: SystemTime) -> Option<DisplacedNote> {
    if change.get("action").and_then(serde_json::Value::as_str) != Some("displaced") {
        return None;
    }
    let holder = change.get("holder").and_then(serde_json::Value::as_str)?;
    displaced_note(holder, change, at)
}

/// A displaced note from a `DisplacedView` or a `displaced` change: `model`,
/// `to` (optional) and `by`, with `holder` the claim's former holder.
fn displaced_note(
    holder: &str,
    value: &serde_json::Value,
    at: SystemTime,
) -> Option<DisplacedNote> {
    let text = |key: &str| value.get(key).and_then(serde_json::Value::as_str);
    Some(DisplacedNote {
        holder: holder_held(holder)?,
        model: claim_model(text("model")?)?,
        to: text("to").and_then(claim_model).unwrap_or_default(),
        by: holder_held(text("by")?)?,
        at,
    })
}

/// One `claims` array entry as kept. An entry without a usable model or
/// holder is dropped; unknown members and unusable times are skipped.
fn keep_claim(value: serde_json::Value) -> Option<ModelClaim> {
    let obj = value.as_object()?;
    let model = obj.get("model").and_then(serde_json::Value::as_str)?;
    let holder = obj.get("holder").and_then(serde_json::Value::as_str)?;
    let model = claim_model(model)?;
    let holder = holder_held(holder)?;
    let auto = obj
        .get("auto")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let in_use = obj
        .get("in_use")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let expires_at = obj
        .get("expires_at")
        .and_then(serde_json::Value::as_str)
        .and_then(rfc3339_time);
    let hold_until = obj
        .get("hold_until")
        .and_then(serde_json::Value::as_str)
        .and_then(rfc3339_time);
    let hold_reason = obj
        .get("hold_reason")
        .and_then(serde_json::Value::as_str)
        .and_then(|raw| text_held(take_bytes(raw, CLAIM_REASON_BYTES), CLAIM_REASON_CHARS))
        .unwrap_or_default();
    let conditional = obj
        .get("conditional")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    Some(ModelClaim {
        model,
        holder,
        auto,
        in_use,
        expires_at,
        hold_until,
        hold_reason,
        conditional,
    })
}

/// Sanitised llama-swap id at the claims cap. The poller stores the same
/// string as [`crate::poller::ModelSetup::swap_id`], so a header match
/// uses one function on both sides.
pub(crate) fn claim_model(raw: &str) -> Option<String> {
    let raw = take_bytes(raw, CLAIM_MODEL_BYTES);
    let cleaned = sanitize(raw, CLAIM_MODEL_BYTES);
    (!cleaned.is_empty()).then_some(cleaned)
}

/// SETUP's `text` kind, then at most [`CLAIM_HOLDER_CHARS`] with the
/// ellipsis rule. Empty after sanitising is nothing.
fn holder_held(raw: &str) -> Option<String> {
    let text = text_held(take_bytes(raw, CLAIM_HOLDER_BYTES), usize::MAX)?;
    Some(ellipsis_chars(&text, CLAIM_HOLDER_CHARS))
}

fn ellipsis_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    if max_chars == 0 {
        return String::new();
    }
    let mut out: String = text.chars().take(max_chars - 1).collect();
    let end = out.trim_end_matches(' ').len();
    out.truncate(end);
    out.push('\u{2026}');
    out
}

fn take_bytes(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn rfc3339_time(text: &str) -> Option<SystemTime> {
    let unix = parse_rfc3339(text)?;
    let unix = u64::try_from(unix).ok()?;
    Some(SystemTime::UNIX_EPOCH + Duration::from_secs(unix))
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
        let base = url.trim_end_matches('/');
        let endpoints = Endpoints {
            events: format!("{base}/api/events"),
            gpu: format!("{base}/api/gpu"),
        };
        let thread_shared = Arc::clone(&shared);
        let thread_stop = Arc::clone(&stop);
        thread::Builder::new()
            .name("llama-events".to_owned())
            .spawn(move || run(&endpoints, timeout, &thread_shared, &thread_stop))?;
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

/// What one stream builds, all reset together on a reconnect.
#[derive(Default)]
struct State {
    inflight: Inflight,
    claims: Claims,
    gpu: Gpu,
}

impl State {
    fn new_session(&mut self) {
        self.inflight.new_session();
        self.claims.new_session();
        self.gpu.new_session();
    }
}

struct Endpoints {
    events: String,
    gpu: String,
}

fn run(endpoints: &Endpoints, timeout: Duration, shared: &Mutex<EventsView>, stop: &AtomicBool) {
    let agent = super::llamaswap::new_agent();
    let mut backoff = MIN_BACKOFF;
    let mut state = State::default();
    while !stop.load(Ordering::Relaxed) {
        let began = Instant::now();
        let failure = session(&agent, endpoints, timeout, shared, stop, &mut state);
        state.new_session();
        publish(shared, &mut state, failure);
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
    endpoints: &Endpoints,
    timeout: Duration,
    shared: &Mutex<EventsView>,
    stop: &AtomicBool,
    state: &mut State,
) -> Option<&'static str> {
    // #167: the displaced notes, and a stand-in until the stream's own
    // snapshot. Read before the stream opens, so its frames win.
    if let Some(api) = fetch_gpu(agent, &endpoints.gpu, timeout) {
        state.claims.stand_in(&api);
        state.gpu.stand_in(&api);
        // Only a working stream clears the last failure: the poller would
        // log `recovered` for a GET that answered while the stream fails.
        let failure = shared.lock().unwrap_or_else(|err| err.into_inner()).failure;
        publish(shared, state, failure);
    }
    if stop.load(Ordering::Relaxed) {
        return None;
    }
    let response = agent
        .get(&endpoints.events)
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
            state.inflight.apply(data, now, wall);
            state.claims.apply(data);
            state.gpu.apply(data);
            changed = true;
        });
        if changed {
            publish(shared, state, None);
        }
    }
}

/// One `GET /api/gpu` (#167): `None` on any failure, which is not logged.
fn fetch_gpu(agent: &ureq::Agent, endpoint: &str, timeout: Duration) -> Option<GpuApi> {
    let mut response = agent
        .get(endpoint)
        .config()
        .timeout_global(Some(timeout))
        .build()
        .call()
        .ok()?;
    if response.status().as_u16() != 200 {
        return None;
    }
    let body = response
        .body_mut()
        .with_config()
        .limit(GPU_BODY_CAP)
        .read_to_vec()
        .ok()?;
    parse_gpu_api(&body)
}

fn publish(shared: &Mutex<EventsView>, state: &mut State, failure: Option<&'static str>) {
    let mut view = state.inflight.view(Instant::now());
    let (list, claims_error) = state.claims.view();
    let (gpu, gpu_error) = state.gpu.view();
    view.displaced = state.claims.displaced(SystemTime::now());
    view.claims = list;
    view.claims_error = claims_error;
    view.gpu = gpu;
    view.gpu_error = gpu_error;
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

    fn apply_claims(claims: &mut Claims, body: serde_json::Value) {
        let text = frame("claims", &body);
        let mut parser = SseParser::default();
        parser.feed(text.as_bytes(), |data| claims.apply(data));
    }

    fn claim_entry(
        model: &str,
        holder: &str,
        auto: bool,
        in_use: bool,
        expires_at: Option<&str>,
        hold_until: Option<&str>,
        hold_reason: Option<&str>,
    ) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert("model".into(), serde_json::Value::String(model.into()));
        map.insert("holder".into(), serde_json::Value::String(holder.into()));
        map.insert("auto".into(), serde_json::Value::Bool(auto));
        map.insert("in_use".into(), serde_json::Value::Bool(in_use));
        map.insert(
            "expires_at".into(),
            expires_at.map_or(serde_json::Value::Null, |s| s.into()),
        );
        map.insert(
            "hold_until".into(),
            hold_until.map_or(serde_json::Value::Null, |s| s.into()),
        );
        if let Some(reason) = hold_reason {
            map.insert("hold_reason".into(), reason.into());
        }
        map.insert("claimed_at".into(), "2026-10-12T02:36:04.718076818Z".into());
        serde_json::Value::Object(map)
    }

    #[test]
    fn claims_frames_keep_the_latest_list() {
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({"changes":[],"claims":[],"at":"2026-10-12T02:36:03.437371362Z"}),
        );
        assert!(claims.view().0.is_empty());

        apply_claims(
            &mut claims,
            serde_json::json!({
                "changes":[{"action":"claimed","model":"big","holder":"eval-runner"}],
                "claims":[claim_entry(
                    "big", "eval-runner", false, false,
                    Some("2026-10-12T02:37:04.718076818Z"), None, None,
                )],
                "at":"2026-10-12T02:36:04.718103348Z",
            }),
        );
        let (list, err) = claims.view();
        assert!(err.is_none());
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].model, "big");
        assert_eq!(list[0].holder, "eval-runner");
        assert!(!list[0].auto);
        assert!(!list[0].in_use);
        assert!(list[0].expires_at.is_some());
        assert!(list[0].hold_until.is_none());
        assert!(list[0].hold_reason.is_empty());

        apply_claims(
            &mut claims,
            serde_json::json!({
                "changes":[{"action":"claimed","model":"big","holder":"chat-ui","by":"chat-ui"}],
                "claims":[claim_entry(
                    "big", "chat-ui", true, true, None, None, None,
                )],
            }),
        );
        let list = claims.view().0;
        assert_eq!(list[0].holder, "chat-ui");
        assert!(list[0].auto && list[0].in_use);
        assert!(list[0].expires_at.is_none());

        apply_claims(
            &mut claims,
            serde_json::json!({
                "changes":[{"action":"idle","model":"big","holder":"chat-ui","by":"chat-ui"}],
                "claims":[claim_entry(
                    "big", "chat-ui", true, false,
                    Some("2026-10-12T04:47:18.26725812Z"), None, None,
                )],
            }),
        );
        let list = claims.view().0;
        assert!(!list[0].in_use);
        assert!(list[0].expires_at.is_some());

        apply_claims(
            &mut claims,
            serde_json::json!({
                "changes":[{"action":"expired","model":"big","holder":"chat-ui"}],
                "claims":[],
            }),
        );
        assert!(claims.view().0.is_empty());

        apply_claims(
            &mut claims,
            serde_json::json!({
                "changes":[{"action":"claimed","model":"big","holder":"eval-runner"}],
                "claims":[claim_entry(
                    "big", "eval-runner", false, false,
                    Some("2026-10-12T02:37:04Z"),
                    Some("2026-10-12T02:37:04Z"),
                    Some("reader check"),
                )],
            }),
        );
        let list = claims.view().0;
        assert!(list[0].hold_until.is_some());
        assert_eq!(list[0].hold_reason, "reader check");

        apply_claims(
            &mut claims,
            serde_json::json!({
                "changes":[{"action":"force-released","model":"big","holder":"eval-runner","by":"eval-runner"}],
                "claims":[],
            }),
        );
        assert!(claims.view().0.is_empty());
    }

    #[test]
    fn two_claims_on_two_models_are_kept() {
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[
                    claim_entry("big", "eval-runner", false, false, None, None, None),
                    claim_entry("small", "chat-ui", true, true, None, None, None),
                ],
            }),
        );
        let list = claims.view().0;
        assert_eq!(
            list.iter()
                .map(|c| (c.model.as_str(), c.holder.as_str()))
                .collect::<Vec<_>>(),
            vec![("big", "eval-runner"), ("small", "chat-ui")]
        );
    }

    #[test]
    fn a_malformed_claims_frame_keeps_the_previous_list() {
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry("big", "eval-runner", false, false, None, None, None)],
            }),
        );
        claims.apply(br#"{"type":"claims","data":"{not json"}"#);
        claims.apply(br#"{"type":"claims","data":"{not json"}"#);
        let (list, err) = claims.view();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].holder, "eval-runner");
        assert_eq!(err, Some("malformed claims frame"));
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry("big", "chat-ui", true, true, None, None, None)],
            }),
        );
        let (list, err) = claims.view();
        assert_eq!(list[0].holder, "chat-ui");
        assert_eq!(err, Some("malformed claims frame"));
    }

    #[test]
    fn an_entry_missing_holder_is_skipped_not_the_list() {
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[
                    {"model":"big","in_use":true},
                    claim_entry("small", "chat-ui", false, false, None, None, None),
                    {"holder":"eval-runner","in_use":false},
                ],
            }),
        );
        let list = claims.view().0;
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].model, "small");
    }

    #[test]
    fn over_long_holder_and_reason_are_capped() {
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry(
                    "big",
                    &"h".repeat(80),
                    false,
                    false,
                    None,
                    Some("2026-10-12T02:37:04Z"),
                    Some(&"r".repeat(80)),
                )],
            }),
        );
        let list = claims.view().0;
        assert_eq!(list[0].holder.chars().count(), 24);
        assert!(list[0].holder.ends_with('\u{2026}'));
        assert_eq!(list[0].hold_reason.chars().count(), 40);
    }

    #[test]
    fn a_reason_drops_escapes_and_controls() {
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry(
                    "big",
                    "operator",
                    false,
                    false,
                    None,
                    Some("2026-10-12T02:37:04Z"),
                    Some("overnight  eval,\n\u{1b}[2Jdo not unload"),
                )],
            }),
        );
        let list = claims.view().0;
        assert_eq!(list[0].hold_reason, "overnight eval, [2Jdo not unload");
        assert!(!list[0].hold_reason.contains('\u{1b}'));
    }

    #[test]
    fn more_than_sixteen_entries_keep_the_first_sixteen() {
        let mut claims = Claims::default();
        let many: Vec<_> = (0..MAX_CLAIMS + 8)
            .map(|i| claim_entry("big", &format!("h{i}"), false, false, None, None, None))
            .collect();
        apply_claims(&mut claims, serde_json::json!({"claims": many}));
        let list = claims.view().0;
        assert_eq!(list.len(), 16);
        assert_eq!(list[0].holder, "h0");
        assert_eq!(list[15].holder, "h15");
    }

    #[test]
    fn reconnect_forgets_the_claims_list() {
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry("big", "eval-runner", false, false, None, None, None)],
            }),
        );
        assert_eq!(claims.view().0.len(), 1);
        claims.new_session();
        assert!(claims.view().0.is_empty());
        assert!(claims.view().1.is_none());
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry("big", "chat-ui", true, true, None, None, None)],
            }),
        );
        assert_eq!(claims.view().0[0].holder, "chat-ui");
    }

    /// Two llama-swap ids that share their first 31 characters stay
    /// distinct: `model_key`'s 32-character cut would make both `…` and
    /// they would show each other's claim.
    #[test]
    fn two_ids_that_share_their_first_31_characters_stay_distinct() {
        let q4 = "unsloth-Qwen3.6-35B-A3B-Instruct-UD-Q4_K_XL";
        let q6 = "unsloth-Qwen3.6-35B-A3B-Instruct-UD-Q6_K_XL";
        assert_eq!(&q4[..31], &q6[..31]);
        assert_ne!(q4, q6);
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[
                    claim_entry(q4, "eval-runner", false, true, None, None, None),
                    claim_entry(q6, "chat-ui", true, false, None, None, None),
                ],
            }),
        );
        let list = claims.view().0;
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].model, q4);
        assert_eq!(list[0].holder, "eval-runner");
        assert_eq!(list[1].model, q6);
        assert_eq!(list[1].holder, "chat-ui");
        assert_ne!(list[0].model, list[1].model);
    }

    /// `"claims": null` is an empty list, not a malformed frame: a server
    /// may encode an empty list that way. It clears a previous list and
    /// logs nothing.
    #[test]
    fn a_null_claims_array_is_an_empty_list() {
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry("big", "eval-runner", false, false, None, None, None)],
            }),
        );
        assert_eq!(claims.view().0.len(), 1);
        let text = frame(
            "claims",
            &serde_json::json!({"claims": serde_json::Value::Null}),
        );
        let mut parser = SseParser::default();
        parser.feed(text.as_bytes(), |data| claims.apply(data));
        let (list, err) = claims.view();
        assert!(list.is_empty(), "{list:?}");
        assert!(err.is_none(), "{err:?}");
    }

    /// A holder is SETUP's `text` kind, then 24 characters with the
    /// ellipsis rule. A space or a non-ASCII character does not drop the
    /// claim; only an empty result after sanitising skips the entry.
    #[test]
    fn a_holder_with_a_space_or_an_accent_is_kept() {
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry(
                    "big", "Pat Example", false, true, None, None, None,
                )],
            }),
        );
        assert_eq!(claims.view().0[0].holder, "Pat Example");

        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry("big", "José", false, true, None, None, None)],
            }),
        );
        assert_eq!(claims.view().0[0].holder, "José");

        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry(
                    "big", "\u{1b}[2Jeval-runner", false, true, None, None, None,
                )],
            }),
        );
        assert_eq!(claims.view().0[0].holder, "[2Jeval-runner");
        assert!(!claims.view().0[0].holder.contains('\u{1b}'));

        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[
                    claim_entry("gone", "\u{1b}\u{07}\n\t", false, true, None, None, None),
                    claim_entry("big", "chat-ui", true, true, None, None, None),
                ],
            }),
        );
        let list = claims.view().0;
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].model, "big");
        assert_eq!(list[0].holder, "chat-ui");
    }

    /// Two claims on the same model: both are kept, in order. The header
    /// draws the first.
    #[test]
    fn two_claims_on_the_same_model_keep_the_first() {
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[
                    claim_entry("big", "eval-runner", false, true, None, None, None),
                    claim_entry("big", "chat-ui", true, false, None, None, None),
                ],
            }),
        );
        let list = claims.view().0;
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].holder, "eval-runner");
        assert_eq!(list[1].holder, "chat-ui");
        assert_eq!(list[0].model, list[1].model);
    }

    /// A holder cut at 24 characters does not leave a space before `…`.
    #[test]
    fn a_holder_cut_does_not_end_in_a_space_before_the_ellipsis() {
        let mut claims = Claims::default();
        // 22 characters, a space, then more: take(23) ends on that space.
        let raw = format!("{} extra", "1234567890123456789012");
        assert_eq!(raw.chars().nth(22), Some(' '));
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry("big", &raw, false, true, None, None, None)],
            }),
        );
        let holder = &claims.view().0[0].holder;
        // take(23) ends on the space, then the trim drops it: 22 digits + `…`.
        assert_eq!(holder.chars().count(), 23);
        assert!(holder.ends_with('\u{2026}'));
        assert!(
            !holder.contains(" \u{2026}"),
            "space before ellipsis: {holder:?}"
        );
        assert_eq!(
            holder,
            &format!("{}{}", "1234567890123456789012", '\u{2026}')
        );
    }

    /// Byte caps: a 128-byte model id is kept (then sanitised), a holder
    /// of 64 bytes is cut before text sanitising, a reason of 256 bytes
    /// too.
    #[test]
    fn claim_byte_caps_are_128_64_and_256() {
        let mut claims = Claims::default();
        let model = "m".repeat(128);
        let holder = "h".repeat(64);
        let reason = "r".repeat(256);
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry(
                    &model, &holder, false, false, None,
                    Some("2026-10-12T02:37:04Z"), Some(&reason),
                )],
            }),
        );
        let list = claims.view().0;
        assert_eq!(list[0].model.chars().count(), 128);
        assert_eq!(list[0].model, "m".repeat(128));
        assert_eq!(list[0].holder.chars().count(), 24);
        assert_eq!(list[0].holder, format!("{}{}", "h".repeat(23), '\u{2026}'));
        assert_eq!(list[0].hold_reason.chars().count(), 40);
        assert_eq!(list[0].hold_reason, "r".repeat(40));

        let too_long = "m".repeat(129);
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry(&too_long, "chat-ui", false, true, None, None, None)],
            }),
        );
        let kept = &claims.view().0[0].model;
        // 129 bytes, cut to 128, then sanitise at 128 characters: the
        // 128-byte cap binds, so this is 128 `m`s, not 127 `m`s and `…`.
        assert_eq!(kept.chars().count(), 128);
        assert_eq!(kept, &"m".repeat(128));
        assert!(!kept.contains('\u{2026}'));
    }

    /// The 64-byte holder cap and the 256-byte reason cap cut at their
    /// own lengths: ASCII longer than the later character caps, cut at
    /// the literal byte lengths.
    #[test]
    fn claim_byte_caps_cut_at_64_and_256() {
        assert_eq!(CLAIM_HOLDER_BYTES, 64);
        assert_eq!(CLAIM_REASON_BYTES, 256);
        assert_eq!(
            take_bytes(&"h".repeat(80), CLAIM_HOLDER_BYTES),
            "h".repeat(64)
        );
        assert_eq!(
            take_bytes(&"r".repeat(300), CLAIM_REASON_BYTES),
            "r".repeat(256)
        );
        let gothic_r = "\u{10348}".repeat(65);
        assert_eq!(gothic_r.len(), 260);
        let reason_bytes = take_bytes(&gothic_r, CLAIM_REASON_BYTES);
        assert_eq!(reason_bytes.len(), 256);
        assert_eq!(reason_bytes.chars().count(), 64);
        // 4-byte scalars: 16 fit in 64 bytes, 17 do not. The 24-character
        // cap is not the one that cuts.
        let gothic = "\u{10348}".repeat(17);
        assert_eq!(gothic.len(), 68);
        let held = take_bytes(&gothic, CLAIM_HOLDER_BYTES);
        assert_eq!(held.len(), 64);
        assert_eq!(held.chars().count(), 16);
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry("big", &gothic, false, true, None, None, None)],
            }),
        );
        let holder = &claims.view().0[0].holder;
        assert_eq!(holder.chars().count(), 16);
        assert_eq!(holder.as_str(), "\u{10348}".repeat(16));
        assert!(!holder.contains('\u{2026}'));
    }

    /// The 256-byte reason cap is applied before text sanitising, so
    /// control bytes in the first 256 that the sanitiser drops cannot
    /// make room for visible text after that cut.
    #[test]
    fn a_reason_cut_at_256_bytes_drops_visible_text_after_that() {
        let mut raw = String::from("seen");
        raw.push_str(&"\u{1b}".repeat(252));
        assert_eq!(raw.len(), 256);
        raw.push_str(" after the cap that a larger byte cut would keep");
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims":[claim_entry(
                    "big", "operator", false, false, None,
                    Some("2026-10-12T02:37:04Z"), Some(&raw),
                )],
            }),
        );
        assert_eq!(claims.view().0[0].hold_reason, "seen");
    }

    // ---- #167: the GPU reservation, guests and displaced notes ----------

    fn apply_gpu(gpu: &mut Gpu, body: serde_json::Value) {
        let text = frame("gpu", &body);
        let mut parser = SseParser::default();
        parser.feed(text.as_bytes(), |data| gpu.apply(data));
    }

    /// A `GPUView` as llama-swap writes it, invented names.
    fn gpu_view(holder: &str) -> serde_json::Value {
        serde_json::json!({
            "holder": holder,
            "mode": "models",
            "state": "active",
            "in_use": false,
            "expires_at": "2026-10-20T21:50:00Z",
            "claimed_at": "2026-10-20T21:40:00.123456789Z",
            "hold_until": "2026-10-21T21:01:00Z",
            "hold_reason": "invented queue",
        })
    }

    fn unix(text: &str) -> SystemTime {
        rfc3339_time(text).expect("time")
    }

    /// Every field of the reservation: holder, mode, state, in_use,
    /// expires_at, hold_until and hold_reason are kept; claimed_at,
    /// `changes` and `at` are read past.
    #[test]
    fn a_gpu_frame_keeps_every_field() {
        let mut gpu = Gpu::default();
        apply_gpu(
            &mut gpu,
            serde_json::json!({
                "changes": [{"action": "reserved", "holder": "sample-holder"}],
                "gpu": gpu_view("sample-holder"),
                "at": "2026-10-20T21:40:00Z",
            }),
        );
        let (kept, err) = gpu.view();
        assert!(err.is_none());
        let kept = kept.expect("reservation");
        assert_eq!(
            kept,
            GpuReservation {
                holder: "sample-holder".to_owned(),
                external: false,
                draining: false,
                in_use: false,
                expires_at: Some(unix("2026-10-20T21:50:00Z")),
                hold_until: Some(unix("2026-10-21T21:01:00Z")),
                hold_reason: "invented queue".to_owned(),
            }
        );

        let mut view = gpu_view("sample-holder");
        view["mode"] = "external".into();
        view["state"] = "draining".into();
        view["in_use"] = true.into();
        view["expires_at"] = serde_json::Value::Null;
        view["hold_until"] = serde_json::Value::Null;
        view.as_object_mut().unwrap().remove("hold_reason");
        apply_gpu(&mut gpu, serde_json::json!({"gpu": view}));
        let kept = gpu.view().0.expect("reservation");
        assert!(kept.external && kept.draining && kept.in_use);
        assert!(kept.expires_at.is_none() && kept.hold_until.is_none());
        assert!(kept.hold_reason.is_empty());

        // Released: `null`.
        apply_gpu(&mut gpu, serde_json::json!({"changes": [], "gpu": null}));
        assert!(gpu.view().0.is_none());
    }

    /// Unknown members are ignored; an unknown mode or state, or a field
    /// of the wrong type, adds nothing; a missing or unusable holder is no
    /// reservation, and a missing `gpu` too.
    #[test]
    fn unknown_or_missing_gpu_fields_draw_nothing_rather_than_fail() {
        let mut gpu = Gpu::default();
        let mut view = gpu_view("sample-holder");
        view["mode"] = "quantum".into();
        view["state"] = "melting".into();
        view["in_use"] = "yes".into();
        view["expires_at"] = 12.into();
        view["hold_until"] = "not a time".into();
        view["hold_reason"] = 7.into();
        view["future_field"] = serde_json::json!({"deep": [1, 2]});
        apply_gpu(&mut gpu, serde_json::json!({"gpu": view, "extra": true}));
        let kept = gpu.view().0.expect("reservation");
        assert_eq!(kept.holder, "sample-holder");
        assert!(!kept.external && !kept.draining && !kept.in_use);
        assert!(kept.expires_at.is_none() && kept.hold_until.is_none());
        assert!(kept.hold_reason.is_empty());
        assert!(gpu.view().1.is_none());

        for holderless in [
            serde_json::json!({"gpu": {"mode": "models"}}),
            serde_json::json!({"gpu": {"holder": 5}}),
            serde_json::json!({"gpu": {"holder": "\u{1b}\u{07}"}}),
            serde_json::json!({"gpu": "sample-holder"}),
            serde_json::json!({}),
        ] {
            apply_gpu(
                &mut gpu,
                serde_json::json!({"gpu": gpu_view("sample-holder")}),
            );
            apply_gpu(&mut gpu, holderless.clone());
            assert!(gpu.view().0.is_none(), "{holderless}");
            assert!(gpu.view().1.is_none(), "{holderless}");
        }
    }

    /// A frame whose `data` does not parse keeps the reservation and is
    /// remembered for the session; a new session forgets both.
    #[test]
    fn a_malformed_gpu_frame_keeps_the_reservation_until_reconnect() {
        let mut gpu = Gpu::default();
        apply_gpu(
            &mut gpu,
            serde_json::json!({"gpu": gpu_view("sample-holder")}),
        );
        gpu.apply(br#"{"type":"gpu","data":"{not json"}"#);
        gpu.apply(br#"{"type":"gpu","data":"[1,2]"}"#);
        let (kept, err) = gpu.view();
        assert_eq!(kept.expect("kept").holder, "sample-holder");
        assert_eq!(err, Some("malformed gpu frame"));
        // Other frame types are not `gpu` frames.
        gpu.apply(br#"{"type":"claims","data":"{not json"}"#);
        gpu.new_session();
        assert_eq!(gpu.view(), (None, None));
    }

    /// The holder's and the reason's caps are the claims': 64 and 256
    /// bytes, then 24 and 40 characters.
    #[test]
    fn gpu_holder_and_reason_have_the_claim_caps() {
        let mut gpu = Gpu::default();
        let mut view = gpu_view(&"h".repeat(80));
        view["hold_reason"] = "r".repeat(300).into();
        apply_gpu(&mut gpu, serde_json::json!({"gpu": view}));
        let kept = gpu.view().0.expect("reservation");
        assert_eq!(kept.holder, format!("{}{}", "h".repeat(23), '\u{2026}'));
        assert_eq!(kept.hold_reason, "r".repeat(40));
        // 64 bytes of a 4-byte scalar is 16 characters, as for a claim.
        let mut view = gpu_view(&"\u{10348}".repeat(17));
        view["hold_reason"] = format!("seen{}tail", "\u{1b}".repeat(252)).into();
        apply_gpu(&mut gpu, serde_json::json!({"gpu": view}));
        let kept = gpu.view().0.expect("reservation");
        assert_eq!(kept.holder, "\u{10348}".repeat(16));
        assert_eq!(kept.hold_reason, "seen");
    }

    #[test]
    fn a_conditional_claim_is_a_guest() {
        let mut claims = Claims::default();
        let mut guest = claim_entry("big", "guest-ui", true, false, None, None, None);
        guest["conditional"] = true.into();
        let mut odd = claim_entry("small", "other-ui", false, false, None, None, None);
        odd["conditional"] = "yes".into();
        apply_claims(
            &mut claims,
            serde_json::json!({"claims": [
                guest,
                claim_entry("big", "chat-ui", true, true, None, None, None),
                odd,
            ]}),
        );
        let list = claims.view().0;
        assert!(list[0].conditional);
        assert!(!list[1].conditional, "omitted");
        assert!(!list[2].conditional, "wrong type");
    }

    fn displaced_json(model: &str, holder: &str, to: Option<&str>, by: &str) -> serde_json::Value {
        let mut change = serde_json::json!({
            "action": "displaced", "model": model, "holder": holder, "by": by,
        });
        if let Some(to) = to {
            change["to"] = to.into();
        }
        change
    }

    /// A `displaced` change becomes a note at the frame's `at`; other
    /// actions, and changes without a model, holder or `by`, do not. One
    /// note per former holder, newest first; the notes outlive a reconnect.
    #[test]
    fn displaced_changes_become_notes_that_outlive_a_reconnect() {
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({
                "changes": [
                    displaced_json("tiel", "guest-a", Some("qwen"), "sample-holder"),
                    {"action": "claimed", "model": "qwen", "holder": "sample-holder"},
                    {"action": "displaced", "model": "x", "holder": "guest-c"},
                    {"action": "displaced", "holder": "guest-d", "by": "sample-holder"},
                ],
                "claims": [],
                "at": "2026-10-20T21:09:00Z",
            }),
        );
        apply_claims(
            &mut claims,
            serde_json::json!({
                "changes": [displaced_json("qwen", "guest-b", None, "sample-holder")],
                "claims": [],
                "at": "2026-10-20T21:10:00Z",
            }),
        );
        let wall = unix("2026-10-20T21:12:00Z");
        let notes = claims.displaced(wall);
        assert_eq!(
            notes
                .iter()
                .map(|n| (
                    n.holder.as_str(),
                    n.model.as_str(),
                    n.to.as_str(),
                    n.by.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("guest-b", "qwen", "", "sample-holder"),
                ("guest-a", "tiel", "qwen", "sample-holder"),
            ]
        );
        assert_eq!(notes[1].at, unix("2026-10-20T21:09:00Z"));
        claims.new_session();
        assert_eq!(claims.displaced(wall).len(), 2, "kept across sessions");
        // Ten minutes after it, a note is no longer shown.
        assert_eq!(claims.displaced(unix("2026-10-20T21:19:00Z")).len(), 1);
        assert!(claims.displaced(unix("2026-10-20T21:20:00Z")).is_empty());
        // A frame without `at`, or `changes` of the wrong type, adds none.
        let mut claims = Claims::default();
        apply_claims(
            &mut claims,
            serde_json::json!({
                "changes": [displaced_json("tiel", "guest-a", None, "sample-holder")],
                "claims": [],
            }),
        );
        apply_claims(
            &mut claims,
            serde_json::json!({"changes": "displaced", "claims": [], "at": "2026-10-20T21:09:00Z"}),
        );
        assert!(claims.displaced(wall).is_empty());
        assert!(claims.view().1.is_none(), "neither fails the frame");
    }

    /// The same displacement again later is a new event: frames at 21:09
    /// and 21:30, and at 21:31 the 21:30 note shows. Within five seconds
    /// it is the same event and keeps its first time.
    #[test]
    fn a_later_identical_displacement_shows_again() {
        let mut claims = Claims::default();
        for at in ["2026-10-20T21:09:00Z", "2026-10-20T21:30:00Z"] {
            apply_claims(
                &mut claims,
                serde_json::json!({
                    "changes": [displaced_json("tiel", "guest-a", Some("qwen"), "sample-holder")],
                    "claims": [],
                    "at": at,
                }),
            );
        }
        let notes = claims.displaced(unix("2026-10-20T21:31:00Z"));
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert_eq!(notes[0].at, unix("2026-10-20T21:30:00Z"));
        apply_claims(
            &mut claims,
            serde_json::json!({
                "changes": [displaced_json("tiel", "guest-a", Some("qwen"), "sample-holder")],
                "claims": [],
                "at": "2026-10-20T21:30:05Z",
            }),
        );
        let notes = claims.displaced(unix("2026-10-20T21:31:00Z"));
        assert_eq!(notes[0].at, unix("2026-10-20T21:30:00Z"), "same event");
    }

    /// A former holder with a claim again shows no note, as llama-swap
    /// drops it; a newer note replaces an older one of the same holder.
    #[test]
    fn a_note_goes_when_its_holder_claims_again_and_the_newest_wins() {
        let mut claims = Claims::default();
        let at = |t: &str| serde_json::Value::from(t);
        apply_claims(
            &mut claims,
            serde_json::json!({
                "changes": [displaced_json("tiel", "guest-a", Some("qwen"), "sample-holder")],
                "claims": [],
                "at": at("2026-10-20T21:09:00Z"),
            }),
        );
        apply_claims(
            &mut claims,
            serde_json::json!({
                "changes": [displaced_json("qwen", "guest-a", Some("tiel"), "sample-holder")],
                "claims": [],
                "at": at("2026-10-20T21:11:00Z"),
            }),
        );
        let wall = unix("2026-10-20T21:12:00Z");
        let notes = claims.displaced(wall);
        assert_eq!(notes.len(), 1);
        assert_eq!(
            (notes[0].model.as_str(), notes[0].to.as_str()),
            ("qwen", "tiel")
        );
        apply_claims(
            &mut claims,
            serde_json::json!({
                "claims": [claim_entry("tiel", "guest-a", false, true, None, None, None)],
            }),
        );
        assert!(claims.displaced(wall).is_empty());
    }

    #[test]
    fn at_most_eight_notes_are_kept() {
        let mut claims = Claims::default();
        for i in 0..MAX_DISPLACED + 4 {
            apply_claims(
                &mut claims,
                serde_json::json!({
                    "changes": [displaced_json("m", &format!("g{i}"), None, "sample-holder")],
                    "claims": [],
                    "at": format!("2026-10-20T21:{:02}:00Z", 10 + i),
                }),
            );
        }
        let notes = claims.displaced(unix("2026-10-20T21:22:00Z"));
        assert_eq!(notes.len(), MAX_DISPLACED);
        assert_eq!(notes[0].holder, "g11");
    }

    /// Every member of `GET /api/gpu`: `gpu`, `guests` and `displaced` are
    /// kept; `enabled`, `modes`, `loaded` and the three times are read
    /// past, as are unknown ones.
    #[test]
    fn the_gpu_api_body_keeps_gpu_guests_and_displaced() {
        let mut guest = claim_entry("big", "guest-ui", true, false, None, None, None);
        guest["conditional"] = true.into();
        let body = serde_json::json!({
            "gpu": gpu_view("sample-holder"),
            "enabled": true,
            "modes": ["models", "external"],
            "loaded": [{"model": "big", "state": "ready"}],
            "guests": [guest, {"model": "big"}],
            "displaced": {
                "guest-b": {"model": "qwen", "by": "sample-holder", "at": "2026-10-20T21:10:00Z"},
                "guest-a": {"model": "tiel", "to": "qwen", "by": "sample-holder", "at": "2026-10-20T21:09:00Z"},
                "guest-c": {"model": "tiel", "by": "sample-holder"},
                "guest-d": {"model": "tiel", "by": "sample-holder", "at": "never"},
            },
            "idle_timeout_seconds": 300,
            "max_hold_seconds": 86400,
            "switch_grace_seconds": 30,
            "future": {"x": 1},
        });
        let api = parse_gpu_api(body.to_string().as_bytes()).expect("parsed");
        assert_eq!(api.gpu.expect("gpu").holder, "sample-holder");
        assert_eq!(api.guests.len(), 1, "the entry without a holder is skipped");
        assert!(api.guests[0].conditional);
        assert_eq!(
            api.displaced
                .iter()
                .map(|n| (n.holder.as_str(), n.to.as_str()))
                .collect::<Vec<_>>(),
            vec![("guest-a", "qwen"), ("guest-b", "")],
            "oldest first; a note without a usable time is skipped"
        );

        // No reservation; members of the wrong type; empty maps.
        let api =
            parse_gpu_api(br#"{"gpu":null,"guests":{"a":1},"displaced":[1],"enabled":"yes"}"#)
                .expect("parsed");
        assert_eq!(api, GpuApi::default());
        let api = parse_gpu_api(br#"{}"#).expect("parsed");
        assert_eq!(api, GpuApi::default());
        // Not a JSON object: nothing.
        for junk in [&b"[]"[..], b"404 page not found", b"", b"{\"gpu\":"] {
            assert!(parse_gpu_api(junk).is_none(), "{junk:?}");
        }
    }

    /// `GET /api/gpu` stands in only until the session's own frames, and
    /// its notes merge with the stream's: the same note seen twice keeps
    /// its first time.
    #[test]
    fn the_get_stands_in_until_the_frames_and_merges_notes() {
        let api = parse_gpu_api(
            serde_json::json!({
                "gpu": gpu_view("sample-holder"),
                "guests": [claim_entry("big", "guest-ui", true, false, None, None, None)],
                "displaced": {"guest-a": {"model": "tiel", "to": "qwen", "by": "sample-holder", "at": "2026-10-20T21:09:00Z"}},
            })
            .to_string()
            .as_bytes(),
        )
        .expect("parsed");
        let mut gpu = Gpu::default();
        let mut claims = Claims::default();
        gpu.stand_in(&api);
        claims.stand_in(&api);
        assert_eq!(gpu.view().0.expect("stand-in").holder, "sample-holder");
        assert_eq!(claims.view().0[0].holder, "guest-ui");
        // The session's own frames replace it, and a later GET does not
        // replace them.
        apply_gpu(&mut gpu, serde_json::json!({"gpu": null}));
        apply_claims(
            &mut claims,
            serde_json::json!({
                "changes": [displaced_json("tiel", "guest-a", Some("qwen"), "sample-holder")],
                "claims": [],
                "at": "2026-10-20T21:09:01Z",
            }),
        );
        gpu.stand_in(&api);
        claims.stand_in(&api);
        assert!(gpu.view().0.is_none());
        assert!(claims.view().0.is_empty());
        let notes = claims.displaced(unix("2026-10-20T21:12:00Z"));
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].at, unix("2026-10-20T21:09:00Z"));
    }
}
