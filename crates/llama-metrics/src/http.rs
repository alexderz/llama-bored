//! A small, strictly bounded HTTP/1.1 server for `GET /metrics`.
//!
//! - One acceptor thread polls a non-blocking listener and pets the watchdog
//!   between polls.
//! - A peer outside the allowlist is closed before a byte is read.
//! - A fixed pool of `max_conns` workers serves one request per connection
//!   (`Connection: close`). When every worker is busy, the acceptor answers
//!   `503` without reading and closes.
//! - The request head has a size cap (request line and whole head), a header
//!   count cap, and one deadline for the whole head, so a slow client
//!   (slowloris) is cut off with `408`.
//! - Only `GET /metrics` is served. Another method is `405`, another path
//!   `404`. No body is read.
//!
//! This file and `service.rs` (the one `TcpListener::bind`) are the only
//! socket code in the workspace (S16).

use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::acl::Allowlist;
use crate::expo::{CONTENT_TYPE, Rejected};

/// The one route.
pub const METRICS_PATH: &str = "/metrics";

/// Request size and time limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    /// Longest request line, bytes, without its CRLF.
    pub max_request_line: usize,
    /// Longest request head, bytes, without the final blank line.
    pub max_head: usize,
    /// Most header lines.
    pub max_headers: usize,
    /// The whole head must arrive within this time.
    pub header_timeout: Duration,
    /// Each response write must finish within this time.
    pub write_timeout: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_request_line: 1024,
            max_head: 8192,
            max_headers: 32,
            header_timeout: Duration::from_secs(2),
            write_timeout: Duration::from_secs(2),
        }
    }
}

/// A refused request, and its status line.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Refusal {
    BadRequest,
    NotFound,
    MethodNotAllowed,
    Timeout,
    UriTooLong,
    HeadersTooLarge,
    VersionNotSupported,
}

impl Refusal {
    #[must_use]
    pub fn status(self) -> &'static str {
        match self {
            Self::BadRequest => "400 Bad Request",
            Self::NotFound => "404 Not Found",
            Self::MethodNotAllowed => "405 Method Not Allowed",
            Self::Timeout => "408 Request Timeout",
            Self::UriTooLong => "414 URI Too Long",
            Self::HeadersTooLarge => "431 Request Header Fields Too Large",
            Self::VersionNotSupported => "505 HTTP Version Not Supported",
        }
    }
}

/// Validate a request head: the bytes before the terminating `\r\n\r\n`.
///
/// `Ok(())` means `GET /metrics` over HTTP/1.0 or 1.1 with well-formed
/// headers. Header values are not used.
pub fn parse_head(head: &[u8], limits: &Limits) -> Result<(), Refusal> {
    if head.len() > limits.max_head {
        return Err(Refusal::HeadersTooLarge);
    }
    let text_lines = split_crlf(head);
    // A bare CR or LF inside a line is not HTTP/1.1 framing.
    if text_lines
        .iter()
        .any(|line| line.contains(&b'\r') || line.contains(&b'\n'))
    {
        return Err(Refusal::BadRequest);
    }
    let mut lines = text_lines.into_iter();
    let line = lines.next().unwrap_or_default();
    if line.len() > limits.max_request_line {
        return Err(Refusal::UriTooLong);
    }
    let mut parts = line.split(|b| *b == b' ');
    let (Some(method), Some(target), Some(version), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(Refusal::BadRequest);
    };
    if method.is_empty() || !method.iter().all(u8::is_ascii_uppercase) {
        return Err(Refusal::BadRequest);
    }
    if target.is_empty() || !target.iter().all(|b| b.is_ascii_graphic()) {
        return Err(Refusal::BadRequest);
    }
    match version {
        b"HTTP/1.1" | b"HTTP/1.0" => {}
        v if v.starts_with(b"HTTP/") => return Err(Refusal::VersionNotSupported),
        _ => return Err(Refusal::BadRequest),
    }
    let mut count = 0_usize;
    for header in lines {
        count += 1;
        if count > limits.max_headers {
            return Err(Refusal::HeadersTooLarge);
        }
        // No obsolete line folding, no empty name, no space before the colon.
        let colon = header
            .iter()
            .position(|b| *b == b':')
            .ok_or(Refusal::BadRequest)?;
        let name = &header[..colon];
        if name.is_empty() || !name.iter().all(|b| is_tchar(*b)) {
            return Err(Refusal::BadRequest);
        }
        if header[colon + 1..]
            .iter()
            .any(|b| (b.is_ascii_control() && *b != b'\t') || *b == 0x7f)
        {
            return Err(Refusal::BadRequest);
        }
    }
    if method != b"GET" {
        return Err(Refusal::MethodNotAllowed);
    }
    if target != METRICS_PATH.as_bytes() {
        return Err(Refusal::NotFound);
    }
    Ok(())
}

/// Lines of a head joined by CRLF (the final CRLFCRLF already removed).
fn split_crlf(head: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut rest = head;
    while let Some(at) = rest.windows(2).position(|w| w == b"\r\n") {
        out.push(&rest[..at]);
        rest = &rest[at + 2..];
    }
    out.push(rest);
    out
}

fn is_tchar(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)
}

/// Read up to the blank line, within the limits. `Ok(None)` is a peer that
/// closed before sending a full head: nothing is answered.
pub fn read_head(stream: &mut TcpStream, limits: &Limits) -> Result<Option<Vec<u8>>, Refusal> {
    let deadline = Instant::now() + limits.header_timeout;
    // Room for the head, its blank line, and one byte to detect overflow.
    let cap = limits.max_head + 5;
    let mut buf = vec![0_u8; cap];
    let mut filled = 0_usize;
    loop {
        if let Some(end) = find_blank_line(&buf[..filled]) {
            buf.truncate(end);
            return Ok(Some(buf));
        }
        let first_line_open = !buf[..filled].contains(&b'\n');
        if first_line_open && filled > limits.max_request_line + 1 {
            return Err(Refusal::UriTooLong);
        }
        if filled >= cap {
            return Err(Refusal::HeadersTooLarge);
        }
        let now = Instant::now();
        if now >= deadline {
            return Err(Refusal::Timeout);
        }
        stream
            .set_read_timeout(Some(deadline - now))
            .map_err(|_| Refusal::Timeout)?;
        match stream.read(&mut buf[filled..]) {
            Ok(0) => return Ok(None),
            Ok(n) => filled += n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(err)
                if matches!(
                    err.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                return Err(Refusal::Timeout);
            }
            Err(_) => return Ok(None),
        }
    }
}

/// Offset of the head's end (before `\r\n\r\n`).
fn find_blank_line(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

/// The response body source.
pub trait Body: Send + Sync + 'static {
    /// The exposition text for one scrape.
    fn metrics(&self, rejected: Rejected) -> String;
}

/// Server settings.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub allow: Allowlist,
    /// Worker threads, and connections served at once.
    pub max_conns: usize,
    pub limits: Limits,
    /// Longest wait for a connection between watchdog pets.
    pub poll_interval: Duration,
}

/// Counters shared by the acceptor and the workers.
#[derive(Debug, Default)]
pub struct Stats {
    denied: AtomicU64,
    busy: AtomicU64,
    active: AtomicUsize,
}

impl Stats {
    #[must_use]
    pub fn rejected(&self) -> Rejected {
        Rejected {
            denied: self.denied.load(Ordering::Relaxed),
            busy: self.busy.load(Ordering::Relaxed),
        }
    }

    /// Connections admitted and not yet closed.
    #[must_use]
    pub fn active(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }
}

/// Serve until `stop` is set. `tick` runs at least every `poll_interval`
/// on the acceptor thread (the systemd watchdog in production).
pub fn serve(
    listener: TcpListener,
    cfg: ServerConfig,
    body: Arc<dyn Body>,
    stats: Arc<Stats>,
    stop: Arc<AtomicBool>,
    mut tick: impl FnMut(),
) -> io::Result<()> {
    listener.set_nonblocking(true)?;
    let max = cfg.max_conns.max(1);
    let (tx, rx) = sync_channel::<TcpStream>(max);
    let rx = Arc::new(Mutex::new(rx));
    let mut workers = Vec::with_capacity(max);
    for _ in 0..max {
        let rx = Arc::clone(&rx);
        let body = Arc::clone(&body);
        let stats = Arc::clone(&stats);
        let limits = cfg.limits;
        workers.push(thread::spawn(move || worker(&rx, &*body, &stats, &limits)));
    }
    let timeout = rustix::time::Timespec {
        tv_sec: i64::try_from(cfg.poll_interval.as_secs()).unwrap_or(1),
        tv_nsec: i64::from(cfg.poll_interval.subsec_nanos()),
    };
    while !stop.load(Ordering::SeqCst) {
        tick();
        let mut fds = [rustix::event::PollFd::new(
            &listener,
            rustix::event::PollFlags::IN,
        )];
        match rustix::event::poll(&mut fds, Some(&timeout)) {
            Ok(_) => {}
            Err(err) if err == rustix::io::Errno::INTR => continue,
            Err(err) => return Err(err.into()),
        }
        loop {
            match listener.accept() {
                Ok((stream, peer)) => {
                    if !cfg.allow.permits(peer.ip()) {
                        stats.denied.fetch_add(1, Ordering::Relaxed);
                        let _ = stream.shutdown(Shutdown::Both);
                        continue;
                    }
                    if stats.active.load(Ordering::SeqCst) >= max {
                        stats.busy.fetch_add(1, Ordering::Relaxed);
                        refuse_busy(stream);
                        continue;
                    }
                    stats.active.fetch_add(1, Ordering::SeqCst);
                    if tx.try_send(stream).is_err() {
                        stats.active.fetch_sub(1, Ordering::SeqCst);
                        stats.busy.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(err) if err.kind() == io::ErrorKind::WouldBlock => break,
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => {
                    // EMFILE, ENOBUFS, ECONNABORTED: back off, then poll again.
                    thread::sleep(Duration::from_millis(50));
                    break;
                }
            }
        }
    }
    drop(tx);
    for worker in workers {
        let _ = worker.join();
    }
    Ok(())
}

fn worker(rx: &Mutex<Receiver<TcpStream>>, body: &dyn Body, stats: &Stats, limits: &Limits) {
    loop {
        let next = {
            let guard = rx.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.recv()
        };
        let Ok(stream) = next else {
            return;
        };
        handle(stream, body, stats, limits);
        stats.active.fetch_sub(1, Ordering::SeqCst);
    }
}

fn handle(mut stream: TcpStream, body: &dyn Body, stats: &Stats, limits: &Limits) {
    if stream.set_nonblocking(false).is_err() {
        return;
    }
    let _ = stream.set_nodelay(true);
    let _ = stream.set_write_timeout(Some(limits.write_timeout));
    let outcome = match read_head(&mut stream, limits) {
        Ok(None) => return,
        Ok(Some(head)) => parse_head(&head, limits),
        Err(refusal) => Err(refusal),
    };
    let response = match outcome {
        Ok(()) => respond(
            "200 OK",
            CONTENT_TYPE,
            &body.metrics(stats.rejected()),
            false,
        ),
        Err(refusal) => respond(
            refusal.status(),
            "text/plain; charset=utf-8",
            refusal.status(),
            refusal == Refusal::MethodNotAllowed,
        ),
    };
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
    close_gently(&mut stream);
}

fn respond(status: &str, content_type: &str, body: &str, allow: bool) -> String {
    let allow = if allow { "Allow: GET\r\n" } else { "" };
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{allow}Cache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// `503` on a non-blocking socket, one write attempt, then close. The
/// acceptor never waits on a busy peer.
fn refuse_busy(mut stream: TcpStream) {
    let _ = stream.set_nonblocking(true);
    let text = respond(
        "503 Service Unavailable",
        "text/plain; charset=utf-8",
        "503 Service Unavailable",
        false,
    );
    let _ = stream.write(text.as_bytes());
    let _ = stream.shutdown(Shutdown::Both);
}

/// Half-close, then discard what the peer still sends (bounded), so the
/// response is not lost to a reset from unread request bytes.
fn close_gently(stream: &mut TcpStream) {
    let _ = stream.shutdown(Shutdown::Write);
    let _ = stream.set_read_timeout(Some(Duration::from_millis(100)));
    let deadline = Instant::now() + Duration::from_millis(250);
    let mut sink = [0_u8; 4096];
    let mut total = 0_usize;
    while Instant::now() < deadline && total < 64 * 1024 {
        match stream.read(&mut sink) {
            Ok(0) | Err(_) => break,
            Ok(n) => total += n,
        }
    }
}
