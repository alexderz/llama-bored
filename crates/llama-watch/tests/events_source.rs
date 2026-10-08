//! #80: llama-swap's `/api/events` read on its own thread, against a fake
//! llama-swap on an ephemeral loopback port: a snapshot, a closed stream
//! and a reconnect, and a server without the endpoint.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use llama_watch::sources::events::{Events, EventsView};

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

/// What each accepted connection gets, in order: a status and a body. A
/// 200 body is sent chunked and the connection held `hold` before it
/// closes.
struct Script {
    replies: Vec<(u16, String, Duration)>,
}

fn serve(listener: TcpListener, script: Script, seen: Arc<AtomicUsize>, stop: Arc<AtomicBool>) {
    listener.set_nonblocking(true).expect("nonblocking");
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
        assert_eq!(path, "/api/events", "the only path asked");
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
