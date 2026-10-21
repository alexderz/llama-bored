//! #80: llama-swap's `/api/events` read on its own thread, against a fake
//! llama-swap on an ephemeral loopback port: a snapshot, a closed stream
//! and a reconnect, and a server without the endpoint. Since #167 each
//! session first asks `GET /api/gpu`; the fake answers it apart from the
//! stream's script.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use llama_watch::localtime::{Zone, format_local};
use llama_watch::sources::events::{Events, EventsView};

fn claims_frame(holder: &str) -> String {
    let data = serde_json::json!({
        "changes": [],
        "claims": [{
            "model": "m",
            "holder": holder,
            "auto": false,
            "in_use": true,
            "expires_at": null,
            "hold_until": null,
        }],
    })
    .to_string();
    let envelope = serde_json::json!({"type": "claims", "data": data});
    format!("event:message\ndata:{envelope}\n\n")
}

/// One `inflight` snapshot frame listing `ids` for model `m`, shaped like
/// a v256 capture (invented content).
fn snapshot(ids: &[&str]) -> String {
    let requests: Vec<serde_json::Value> = ids
        .iter()
        .map(|id| {
            serde_json::json!({
                "id": id, "timestamp": "2026-10-07T10:00:00Z", "model": "m",
                "req_path": "/v1/chat/completions", "method": "POST",
                "req_headers": {"User-Agent": "INVENTED"}, "remote_ip": "192.0.2.1",
                "resp_headers": {}, "resp_bytes": 0, "elapsed_ms": 100
            })
        })
        .collect();
    let data = serde_json::json!({"operation": "snapshot", "requests": requests}).to_string();
    let envelope = serde_json::json!({"type": "inflight", "data": data});
    format!("event:message\ndata:{envelope}\n\n")
}

/// What each accepted `/api/events` connection gets, in order: a status
/// and a body. A 200 body is sent chunked and the connection held `hold`
/// before it closes. Every `/api/gpu` gets `gpu` (a 200 body), or a 404.
struct Script {
    replies: Vec<(u16, String, Duration)>,
    gpu: Option<String>,
}

fn serve(listener: TcpListener, script: Script, seen: Arc<AtomicUsize>, stop: Arc<AtomicBool>) {
    listener.set_nonblocking(true).expect("nonblocking");
    let script_gpu = script.gpu;
    let mut replies = script.replies.into_iter();
    while !stop.load(Ordering::Relaxed) {
        let stream = match listener.accept() {
            Ok((stream, _)) => stream,
            Err(_) => {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        };
        stream.set_nonblocking(false).expect("blocking");
        let mut reader = BufReader::new(stream.try_clone().expect("clone"));
        let mut line = String::new();
        let mut path = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                break;
            }
            if path.is_empty() {
                path = line.split_whitespace().nth(1).unwrap_or("").to_owned();
            }
        }
        if path == "/api/gpu" {
            let (status, body) = match &script_gpu {
                Some(body) => (200, body.clone()),
                None => (404, "404 page not found".to_owned()),
            };
            let mut stream = stream;
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            continue;
        }
        assert_eq!(path, "/api/events", "the only other path asked");
        seen.fetch_add(1, Ordering::Relaxed);
        let Some((status, body, hold)) = replies.next() else {
            continue;
        };
        let mut stream = stream;
        if status == 200 {
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
            let _ = stream.write_all(head.as_bytes());
            let _ = write!(stream, "{:x}\r\n{body}\r\n", body.len());
            let _ = stream.flush();
            thread::sleep(hold);
            let _ = stream.write_all(b"0\r\n\r\n");
        } else {
            let _ = write!(
                stream,
                "HTTP/1.1 {status} Nope\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    }
}

fn start(script: Script) -> (String, Arc<AtomicUsize>, Arc<AtomicBool>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    let seen = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let (seen_t, stop_t) = (Arc::clone(&seen), Arc::clone(&stop));
    thread::spawn(move || serve(listener, script, seen_t, stop_t));
    (url, seen, stop)
}

fn wait(events: &Events, what: &str, mut pred: impl FnMut(&EventsView) -> bool) -> EventsView {
    let start = Instant::now();
    loop {
        let view = events.view();
        if pred(&view) {
            return view;
        }
        assert!(start.elapsed() < Duration::from_secs(6), "{what}: {view:?}");
        thread::sleep(Duration::from_millis(20));
    }
}

fn ids(view: &EventsView) -> Vec<String> {
    view.requests.iter().map(|req| req.id.clone()).collect()
}

/// A snapshot fills the list; the stream closing empties it (unknown, not
/// zero) and ends what it held; the reconnect's snapshot fills it again.
#[test]
fn a_closed_stream_reconnects_and_resyncs() {
    let (url, seen, stop) = start(Script {
        replies: vec![
            (200, snapshot(&["1"]), Duration::from_millis(400)),
            (200, snapshot(&["2", "3"]), Duration::from_secs(30)),
        ],
        gpu: None,
    });
    let events = Events::spawn(&url, Duration::from_secs(1)).expect("spawn");
    let first = wait(&events, "first snapshot", |view| view.connected);
    assert_eq!(ids(&first), vec!["1"]);
    let kept = format!("{first:?}");
    assert!(
        !kept.contains("INVENTED") && !kept.contains("192.0.2.1"),
        "{kept}"
    );
    wait(&events, "the stream closed", |view| !view.connected);
    let second = wait(&events, "reconnected", |view| view.connected);
    assert_eq!(ids(&second), vec!["2", "3"]);
    assert_eq!(
        seen.load(Ordering::Relaxed),
        2,
        "one reconnect, after a wait"
    );
    drop(events);
    stop.store(true, Ordering::Relaxed);
}

/// A llama-swap without the endpoint: never connected, the reason kept
/// for the poller's log, and asked again only after a wait.
#[test]
fn no_endpoint_is_a_failure_with_backoff() {
    let (url, seen, stop) = start(Script {
        replies: vec![(404, "404 page not found".to_owned(), Duration::ZERO); 8],
        gpu: None,
    });
    let events = Events::spawn(&url, Duration::from_secs(1)).expect("spawn");
    let view = wait(&events, "failure", |view| view.failure.is_some());
    assert_eq!(view.failure, Some("http status"));
    assert!(!view.connected && view.requests.is_empty());
    thread::sleep(Duration::from_millis(600));
    assert_eq!(
        seen.load(Ordering::Relaxed),
        1,
        "backs off before asking again"
    );
    drop(events);
    stop.store(true, Ordering::Relaxed);
}

/// A reconnect forgets the claims list until the new session's first
/// `claims` frame. The stream loop calls `new_session`; a unit test that
/// calls it itself would still pass if that call were removed.
#[test]
fn a_reconnect_forgets_claims_until_the_new_sessions_frame() {
    let first = snapshot(&["1"]) + &claims_frame("eval-runner");
    let second = snapshot(&["2"]);
    let (url, _, stop) = start(Script {
        replies: vec![
            (200, first, Duration::from_millis(400)),
            (200, second, Duration::from_secs(30)),
        ],
        gpu: None,
    });
    let events = Events::spawn(&url, Duration::from_secs(1)).expect("spawn");
    let first = wait(&events, "first claims", |view| {
        view.connected && view.claims.iter().any(|c| c.holder == "eval-runner")
    });
    assert_eq!(first.claims[0].holder, "eval-runner");
    wait(&events, "the stream closed", |view| !view.connected);
    let view = events.view();
    assert!(
        view.claims.is_empty(),
        "reconnect must forget the list: {:?}",
        view.claims
    );
    let second = wait(&events, "reconnected", |view| view.connected);
    assert!(
        second.claims.is_empty(),
        "no claims frame yet: {:?}",
        second.claims
    );
    drop(events);
    stop.store(true, Ordering::Relaxed);
}

/// Nothing listening: a refused connect is a failure, never a panic.
#[test]
fn a_refused_connect_is_a_failure() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    drop(listener);
    let events = Events::spawn(&url, Duration::from_millis(300)).expect("spawn");
    let view = wait(&events, "refused", |view| view.failure.is_some());
    assert!(
        matches!(
            view.failure,
            Some("connection refused" | "request failed" | "timeout")
        ),
        "{view:?}"
    );
}

fn rfc3339(at: std::time::SystemTime) -> String {
    let unix = at
        .duration_since(std::time::UNIX_EPOCH)
        .expect("after 1970")
        .as_secs();
    let unix = i64::try_from(unix).expect("unix");
    format!("{}Z", format_local(unix, &Zone::fixed(0)).replace(' ', "T"))
}

fn gpu_frame(holder: &str) -> String {
    let data = serde_json::json!({
        "changes": [{"action": "reserved", "holder": holder}],
        "gpu": {
            "holder": holder, "mode": "models", "state": "active",
            "in_use": true, "expires_at": null,
            "claimed_at": "2026-10-20T21:00:00Z", "hold_until": null,
        },
        "at": "2026-10-20T21:00:00Z",
    })
    .to_string();
    let envelope = serde_json::json!({"type": "gpu", "data": data});
    format!("event:message\ndata:{envelope}\n\n")
}

/// #167: a `gpu` frame sets the reservation; a reconnect forgets it; the
/// next session's `GET /api/gpu` stands in while that session sends no
/// `gpu` frame, and brings the displaced notes, which outlive the reconnect
/// after it.
#[test]
fn a_reconnect_forgets_the_reservation_and_the_get_stands_in() {
    let at = rfc3339(std::time::SystemTime::now() - Duration::from_secs(60));
    let gpu_body = serde_json::json!({
        "gpu": {
            "holder": "sample-holder", "mode": "external", "state": "draining",
            "in_use": false, "expires_at": null,
            "claimed_at": "2026-10-20T21:00:00Z", "hold_until": null,
        },
        "enabled": true,
        "modes": ["models", "external"],
        "loaded": [{"model": "m", "state": "stopping"}],
        "guests": [{
            "model": "m", "holder": "guest-ui", "auto": true, "in_use": false,
            "expires_at": null, "claimed_at": "2026-10-20T21:00:00Z",
            "hold_until": null, "conditional": true,
        }],
        "displaced": {"former-ui": {"model": "a", "to": "b", "by": "sample-holder", "at": at}},
        "idle_timeout_seconds": 300,
        "max_hold_seconds": 86400,
        "switch_grace_seconds": 30,
    })
    .to_string();
    let first = snapshot(&["1"]) + &gpu_frame("eval-runner");
    let second = snapshot(&["2"]);
    let (url, _, stop) = start(Script {
        replies: vec![
            (200, first, Duration::from_millis(400)),
            (200, second, Duration::from_secs(30)),
        ],
        gpu: Some(gpu_body),
    });
    let events = Events::spawn(&url, Duration::from_secs(1)).expect("spawn");
    // The stream's own frame wins over the GET read before it.
    let first = wait(&events, "first gpu frame", |view| {
        view.connected && view.gpu.as_ref().is_some_and(|g| g.holder == "eval-runner")
    });
    let gpu = first.gpu.expect("gpu");
    assert!(gpu.in_use && !gpu.external && !gpu.draining);
    assert_eq!(first.displaced.len(), 1, "{:?}", first.displaced);
    wait(&events, "the stream closed", |view| !view.connected);
    let view = events.view();
    if view.claims.is_empty() {
        // Between the sessions: no reservation, the notes kept.
        assert!(view.gpu.is_none(), "{:?}", view.gpu);
        assert_eq!(view.displaced.len(), 1, "{:?}", view.displaced);
    }
    let second = wait(&events, "reconnected", |view| view.connected);
    let gpu = second.gpu.expect("the GET stands in");
    assert_eq!(gpu.holder, "sample-holder");
    assert!(gpu.external && gpu.draining && !gpu.in_use);
    assert_eq!(second.claims.len(), 1, "the guests stand in");
    assert!(second.claims[0].conditional);
    let note = &second.displaced[0];
    assert_eq!(
        (
            note.holder.as_str(),
            note.model.as_str(),
            note.to.as_str(),
            note.by.as_str()
        ),
        ("former-ui", "a", "b", "sample-holder")
    );
    drop(events);
    stop.store(true, Ordering::Relaxed);
}

/// #167: a failed session sets the failure; the next session's working
/// stream clears it with its first frame, while the stream stays open.
/// Two failures in a row keep the failure up through both backoffs (1 s,
/// then 2 s), so a slow test thread still sees it.
#[test]
fn a_working_stream_clears_the_failure() {
    let (url, _, stop) = start(Script {
        replies: vec![
            (500, "boom".to_owned(), Duration::ZERO),
            (500, "boom".to_owned(), Duration::ZERO),
            (200, snapshot(&["1"]), Duration::from_secs(30)),
        ],
        gpu: Some(r#"{"gpu":null,"guests":[],"displaced":{}}"#.to_owned()),
    });
    let events = Events::spawn(&url, Duration::from_secs(1)).expect("spawn");
    let failed = wait(&events, "failure", |view| view.failure.is_some());
    assert_eq!(failed.failure, Some("http status"));
    let view = wait(&events, "connected", |view| view.connected);
    assert!(view.failure.is_none(), "{view:?}");
    drop(events);
    stop.store(true, Ordering::Relaxed);
}
