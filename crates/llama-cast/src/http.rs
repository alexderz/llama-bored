//! A small, strictly bounded HTTP/1.1 server for the DLNA MediaServer.
//!
//! - One acceptor thread polls a non-blocking listener and pets the watchdog
//!   between polls.
//! - A peer outside the allowlist is closed before a byte is read.
//! - A fixed pool of `max_clients + CONTROL_WORKERS` workers serves one
//!   request per connection (`Connection: close`). When every worker is
//!   busy the acceptor answers `503` without reading and closes. At most
//!   `max_clients` of them stream at once; one more stream request is `503`.
//! - The request head has a size cap (request line and whole head), a header
//!   count cap, and one deadline for the whole head and body (slowloris
//!   gets `408`). A POST body needs `Content-Length` and is capped; chunked
//!   bodies are refused.
//! - Routes are fixed: GET/HEAD `/desc.xml`, `/cds.xml`, `/cms.xml`,
//!   `/live.ts`; POST `/ctl/cds`, `/ctl/cms`; SUBSCRIBE `/evt/cds`,
//!   `/evt/cms`. Another method is `405`, another path `404`.
//! - SUBSCRIBE is acknowledged with a SID and never followed by an event:
//!   llama-cast opens no outbound connection.
//!
//! This file and `service.rs` (the one `TcpListener::bind`) are the only
//! TCP code in the crate (S18).

use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use llama_core::log::{self, Priority};

use crate::acl::Allowlist;
use crate::dlna::{self, Device, Service};

/// Workers beyond the stream slots, for descriptions and SOAP.
pub const CONTROL_WORKERS: usize = 4;
/// Methods served.
pub const ALLOW_METHODS: &str = "GET, HEAD, POST, SUBSCRIBE";

/// Request size and time limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    /// Longest request line, bytes, without its CRLF.
    pub max_request_line: usize,
    /// Longest request head, bytes, without the final blank line.
    pub max_head: usize,
    /// Most header lines.
    pub max_headers: usize,
    /// Largest POST body.
    pub max_body: usize,
    /// The whole head, and then the whole body, must arrive within this.
    pub header_timeout: Duration,
    /// Each response write must finish within this time.
    pub write_timeout: Duration,
    /// Each stream write must finish within this time.
    pub stream_write_timeout: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_request_line: 1024,
            max_head: 8192,
            max_headers: 32,
            max_body: 16 * 1024,
            header_timeout: Duration::from_secs(5),
            write_timeout: Duration::from_secs(5),
            stream_write_timeout: Duration::from_secs(15),
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
    LengthRequired,
    PayloadTooLarge,
    UriTooLong,
    HeadersTooLarge,
    NotImplemented,
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
            Self::LengthRequired => "411 Length Required",
            Self::PayloadTooLarge => "413 Content Too Large",
            Self::UriTooLong => "414 URI Too Long",
            Self::HeadersTooLarge => "431 Request Header Fields Too Large",
            Self::NotImplemented => "501 Not Implemented",
            Self::VersionNotSupported => "505 HTTP Version Not Supported",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Method {
    Get,
    Head,
    Post,
    Subscribe,
}

/// The fixed routes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Route {
    Description,
    CdsScpd,
    CmsScpd,
    Stream,
    Control(Service),
    Events,
}

impl Route {
    fn from_path(path: &[u8]) -> Option<Self> {
        Some(match path {
            p if p == dlna::DESC_PATH.as_bytes() => Self::Description,
            p if p == dlna::CDS_SCPD_PATH.as_bytes() => Self::CdsScpd,
            p if p == dlna::CMS_SCPD_PATH.as_bytes() => Self::CmsScpd,
            p if p == dlna::STREAM_PATH.as_bytes() => Self::Stream,
            p if p == dlna::CDS_CONTROL_PATH.as_bytes() => Self::Control(Service::ContentDirectory),
            p if p == dlna::CMS_CONTROL_PATH.as_bytes() => {
                Self::Control(Service::ConnectionManager)
            }
            p if p == dlna::CDS_EVENT_PATH.as_bytes() || p == dlna::CMS_EVENT_PATH.as_bytes() => {
                Self::Events
            }
            _ => return None,
        })
    }

    /// The methods this route answers, for `Allow:`.
    #[must_use]
    pub fn allow(self) -> &'static str {
        match self {
            Self::Description | Self::CdsScpd | Self::CmsScpd | Self::Stream => "GET, HEAD",
            Self::Control(_) => "POST",
            Self::Events => "SUBSCRIBE",
        }
    }

    fn accepts(self, method: Method) -> bool {
        matches!(
            (self, method),
            (
                Self::Description | Self::CdsScpd | Self::CmsScpd | Self::Stream,
                Method::Get | Method::Head
            ) | (Self::Control(_), Method::Post)
                | (Self::Events, Method::Subscribe)
        )
    }
}

/// A parsed, routed request head.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Request {
    pub method: Method,
    pub route: Route,
    /// Header names lowercased, values trimmed.
    pub headers: Vec<(String, String)>,
}

impl Request {
    /// The first header named `name` (lowercase).
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }
}

/// A refusal of a known route with a wrong method carries its `Allow:`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Refused {
    pub refusal: Refusal,
    pub allow: Option<&'static str>,
}

impl From<Refusal> for Refused {
    fn from(refusal: Refusal) -> Self {
        Self {
            refusal,
            allow: None,
        }
    }
}

/// Validate and route a request head: the bytes before `\r\n\r\n`.
pub fn parse_head(head: &[u8], limits: &Limits) -> Result<Request, Refused> {
    if head.len() > limits.max_head {
        return Err(Refusal::HeadersTooLarge.into());
    }
    let text_lines = split_crlf(head);
    if text_lines
        .iter()
        .any(|line| line.contains(&b'\r') || line.contains(&b'\n'))
    {
        return Err(Refusal::BadRequest.into());
    }
    let mut lines = text_lines.into_iter();
    let line = lines.next().unwrap_or_default();
    if line.len() > limits.max_request_line {
        return Err(Refusal::UriTooLong.into());
    }
    let mut parts = line.split(|b| *b == b' ');
    let (Some(method), Some(target), Some(version), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(Refusal::BadRequest.into());
    };
    if method.is_empty() || !method.iter().all(|b| b.is_ascii_uppercase() || *b == b'-') {
        return Err(Refusal::BadRequest.into());
    }
    if target.is_empty() || !target.iter().all(u8::is_ascii_graphic) {
        return Err(Refusal::BadRequest.into());
    }
    match version {
        b"HTTP/1.1" | b"HTTP/1.0" => {}
        v if v.starts_with(b"HTTP/") => return Err(Refusal::VersionNotSupported.into()),
        _ => return Err(Refusal::BadRequest.into()),
    }
    let mut headers = Vec::new();
    for header in lines {
        if headers.len() >= limits.max_headers {
            return Err(Refusal::HeadersTooLarge.into());
        }
        let colon = header
            .iter()
            .position(|b| *b == b':')
            .ok_or(Refusal::BadRequest)?;
        let name = &header[..colon];
        if name.is_empty() || !name.iter().all(|b| is_tchar(*b)) {
            return Err(Refusal::BadRequest.into());
        }
        let value = &header[colon + 1..];
        if value
            .iter()
            .any(|b| (b.is_ascii_control() && *b != b'\t') || *b == 0x7f)
        {
            return Err(Refusal::BadRequest.into());
        }
        headers.push((
            String::from_utf8_lossy(name).to_ascii_lowercase(),
            String::from_utf8_lossy(value).trim().to_owned(),
        ));
    }
    let method = match method {
        b"GET" => Method::Get,
        b"HEAD" => Method::Head,
        b"POST" => Method::Post,
        b"SUBSCRIBE" => Method::Subscribe,
        _ => {
            return Err(Refused {
                refusal: Refusal::MethodNotAllowed,
                allow: Some(ALLOW_METHODS),
            });
        }
    };
    let route = Route::from_path(target).ok_or(Refusal::NotFound)?;
    if !route.accepts(method) {
        return Err(Refused {
            refusal: Refusal::MethodNotAllowed,
            allow: Some(route.allow()),
        });
    }
    Ok(Request {
        method,
        route,
        headers,
    })
}

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

/// A head and the bytes already read past it (the start of a body).
pub struct Head {
    pub head: Vec<u8>,
    pub rest: Vec<u8>,
}

/// Read up to the blank line, within the limits. `Ok(None)` is a peer that
/// closed before sending a full head: nothing is answered.
pub fn read_head(
    stream: &mut TcpStream,
    limits: &Limits,
    deadline: Instant,
) -> Result<Option<Head>, Refusal> {
    let cap = limits.max_head + 5;
    let mut buf = vec![0_u8; cap];
    let mut filled = 0_usize;
    loop {
        if let Some(end) = buf[..filled].windows(4).position(|w| w == b"\r\n\r\n") {
            let rest = buf[end + 4..filled].to_vec();
            buf.truncate(end);
            return Ok(Some(Head { head: buf, rest }));
        }
        let first_line_open = !buf[..filled].contains(&b'\n');
        if first_line_open && filled > limits.max_request_line + 1 {
            return Err(Refusal::UriTooLong);
        }
        if filled >= cap {
            return Err(Refusal::HeadersTooLarge);
        }
        match read_some(stream, &mut buf[filled..], deadline)? {
            0 => return Ok(None),
            n => filled += n,
        }
    }
}

fn read_some(stream: &mut TcpStream, buf: &mut [u8], deadline: Instant) -> Result<usize, Refusal> {
    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err(Refusal::Timeout);
        }
        stream
            .set_read_timeout(Some(deadline - now))
            .map_err(|_| Refusal::Timeout)?;
        match stream.read(buf) {
            Ok(n) => return Ok(n),
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(err)
                if matches!(
                    err.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                return Err(Refusal::Timeout);
            }
            Err(_) => return Ok(0),
        }
    }
}

/// The body length a POST declares: `Content-Length` required, capped, and
/// no `Transfer-Encoding`.
pub fn body_length(req: &Request, limits: &Limits) -> Result<usize, Refusal> {
    if req.header("transfer-encoding").is_some() {
        return Err(Refusal::NotImplemented);
    }
    let lengths: Vec<&str> = req
        .headers
        .iter()
        .filter(|(n, _)| n == "content-length")
        .map(|(_, v)| v.as_str())
        .collect();
    let [text] = lengths.as_slice() else {
        return Err(if lengths.is_empty() {
            Refusal::LengthRequired
        } else {
            Refusal::BadRequest
        });
    };
    if text.is_empty() || text.len() > 9 || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Refusal::BadRequest);
    }
    let len: usize = text.parse().map_err(|_| Refusal::BadRequest)?;
    if len > limits.max_body {
        return Err(Refusal::PayloadTooLarge);
    }
    Ok(len)
}

/// The live stream: one per client.
pub trait Live: Send + Sync + 'static {
    /// Start a stream. Dropping it stops everything behind it.
    fn open(&self) -> io::Result<Box<dyn Read + Send>>;
}

/// What the handlers serve.
pub struct App {
    pub device: Device,
    pub live: Arc<dyn Live>,
}

/// Server settings.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub allow: Allowlist,
    /// Streams served at once.
    pub max_clients: usize,
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
    streams: AtomicUsize,
    subscriptions: AtomicU64,
}

impl Stats {
    /// Peers closed by the allowlist.
    #[must_use]
    pub fn denied(&self) -> u64 {
        self.denied.load(Ordering::Relaxed)
    }

    /// Connections or streams refused with 503.
    #[must_use]
    pub fn busy(&self) -> u64 {
        self.busy.load(Ordering::Relaxed)
    }

    /// Connections admitted and not yet closed.
    #[must_use]
    pub fn active(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }

    /// Streams running now.
    #[must_use]
    pub fn streams(&self) -> usize {
        self.streams.load(Ordering::SeqCst)
    }
}

struct Worker {
    app: Arc<App>,
    stats: Arc<Stats>,
    limits: Limits,
    max_clients: usize,
    stop: Arc<AtomicBool>,
}

/// Serve until `stop` is set. `tick` runs at least every `poll_interval`
/// on the acceptor thread (the systemd watchdog in production).
pub fn serve(
    listener: TcpListener,
    cfg: ServerConfig,
    app: Arc<App>,
    stats: Arc<Stats>,
    stop: Arc<AtomicBool>,
    mut tick: impl FnMut(),
) -> io::Result<()> {
    listener.set_nonblocking(true)?;
    let max_clients = cfg.max_clients.max(1);
    let max = max_clients + CONTROL_WORKERS;
    let (tx, rx) = sync_channel::<TcpStream>(max);
    let rx = Arc::new(Mutex::new(rx));
    let mut workers = Vec::with_capacity(max);
    for _ in 0..max {
        let rx = Arc::clone(&rx);
        let worker = Worker {
            app: Arc::clone(&app),
            stats: Arc::clone(&stats),
            limits: cfg.limits,
            max_clients,
            stop: Arc::clone(&stop),
        };
        workers.push(thread::spawn(move || worker.run(&rx)));
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

impl Worker {
    fn run(&self, rx: &Mutex<Receiver<TcpStream>>) {
        loop {
            let next = {
                let guard = rx.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                guard.recv()
            };
            let Ok(stream) = next else {
                return;
            };
            self.handle(stream);
            self.stats.active.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn handle(&self, mut stream: TcpStream) {
        if stream.set_nonblocking(false).is_err() {
            return;
        }
        let _ = stream.set_nodelay(true);
        let _ = stream.set_write_timeout(Some(self.limits.write_timeout));
        let deadline = Instant::now() + self.limits.header_timeout;
        let head = match read_head(&mut stream, &self.limits, deadline) {
            Ok(None) => return,
            Ok(Some(head)) => head,
            Err(refusal) => return self.refuse(&mut stream, refusal.into()),
        };
        let req = match parse_head(&head.head, &self.limits) {
            Ok(req) => req,
            Err(refused) => return self.refuse(&mut stream, refused),
        };
        let head_only = req.method == Method::Head;
        match req.route {
            Route::Description => {
                let body = dlna::description(&self.app.device);
                send(
                    &mut stream,
                    "200 OK",
                    dlna::XML_CONTENT_TYPE,
                    &[],
                    &body,
                    head_only,
                );
            }
            Route::CdsScpd | Route::CmsScpd => send(
                &mut stream,
                "200 OK",
                dlna::XML_CONTENT_TYPE,
                &[],
                dlna::SCPD,
                head_only,
            ),
            Route::Events => {
                let n = self.stats.subscriptions.fetch_add(1, Ordering::Relaxed) + 1;
                let sid = subscription_id(&self.app.device.udn, n);
                send(
                    &mut stream,
                    "200 OK",
                    "text/plain; charset=utf-8",
                    &[("SID", &sid), ("TIMEOUT", "Second-1800")],
                    "",
                    false,
                );
            }
            Route::Control(service) => {
                self.control(&mut stream, &req, head.rest, service, deadline)
            }
            Route::Stream if head_only => {
                let _ = stream.write_all(stream_head().as_bytes());
            }
            Route::Stream => self.stream(&mut stream),
        }
        close_gently(&mut stream);
    }

    fn refuse(&self, stream: &mut TcpStream, refused: Refused) {
        let status = refused.refusal.status();
        let allow = refused.allow.map(|a| ("Allow", a));
        let extra: Vec<(&str, &str)> = allow.into_iter().collect();
        send(
            stream,
            status,
            "text/plain; charset=utf-8",
            &extra,
            status,
            false,
        );
        close_gently(stream);
    }

    fn control(
        &self,
        stream: &mut TcpStream,
        req: &Request,
        mut body: Vec<u8>,
        service: Service,
        deadline: Instant,
    ) {
        let len = match body_length(req, &self.limits) {
            Ok(len) => len,
            Err(refusal) => return self.refuse(stream, refusal.into()),
        };
        if req
            .header("expect")
            .is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
            && body.len() < len
        {
            let _ = stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
        }
        if body.len() > len {
            return self.refuse(stream, Refusal::BadRequest.into());
        }
        let deadline = deadline.max(Instant::now()) + self.limits.header_timeout;
        let mut chunk = [0_u8; 4096];
        while body.len() < len {
            let want = (len - body.len()).min(chunk.len());
            match read_some(stream, &mut chunk[..want], deadline) {
                Ok(0) => return,
                Ok(n) => body.extend_from_slice(&chunk[..n]),
                Err(refusal) => return self.refuse(stream, refusal.into()),
            }
        }
        let Ok(text) = std::str::from_utf8(&body) else {
            return self.refuse(stream, Refusal::BadRequest.into());
        };
        let action_header = req.header("soapaction").unwrap_or("");
        let answer = dlna::parse_action(service, action_header, text)
            .and_then(|action| dlna::answer(&self.app.device, service, &action));
        match answer {
            Ok(xml) => send(stream, "200 OK", dlna::XML_CONTENT_TYPE, &[], &xml, false),
            Err(fault) => send(
                stream,
                "500 Internal Server Error",
                dlna::XML_CONTENT_TYPE,
                &[],
                &dlna::fault(fault),
                false,
            ),
        }
    }

    fn stream(&self, stream: &mut TcpStream) {
        let claimed = self
            .stats
            .streams
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                (n < self.max_clients).then_some(n + 1)
            })
            .is_ok();
        if !claimed {
            self.stats.busy.fetch_add(1, Ordering::Relaxed);
            let status = "503 Service Unavailable";
            send(
                stream,
                status,
                "text/plain; charset=utf-8",
                &[],
                status,
                false,
            );
            return;
        }
        self.copy_stream(stream);
        self.stats.streams.fetch_sub(1, Ordering::SeqCst);
    }

    fn copy_stream(&self, stream: &mut TcpStream) {
        let mut source = match self.app.live.open() {
            Ok(source) => source,
            Err(err) => {
                log::emit(
                    &mut log::Stderr,
                    Priority::Err,
                    &format!("stream: cannot start the encoder: {err}"),
                );
                let status = "500 Internal Server Error";
                send(
                    stream,
                    status,
                    "text/plain; charset=utf-8",
                    &[],
                    status,
                    false,
                );
                return;
            }
        };
        let peer = stream
            .peer_addr()
            .map_or_else(|_| "?".to_owned(), |p| p.ip().to_string());
        log::emit(
            &mut log::Stderr,
            Priority::Info,
            &format!("stream: start for {peer}"),
        );
        let _ = stream.set_write_timeout(Some(self.limits.stream_write_timeout));
        if stream.write_all(stream_head().as_bytes()).is_err() {
            return;
        }
        let mut buf = vec![0_u8; 64 * 1024];
        while !self.stop.load(Ordering::SeqCst) {
            match source.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if stream.write_all(&buf[..n]).is_err() {
                        break;
                    }
                }
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
        drop(source);
        log::emit(
            &mut log::Stderr,
            Priority::Info,
            &format!("stream: end for {peer}"),
        );
    }
}

/// The stream response head: close-delimited, no length.
#[must_use]
pub fn stream_head() -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\ntransferMode.dlna.org: Streaming\r\nServer: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        dlna::STREAM_MIME,
        crate::ssdp::server_header(),
    )
}

/// `uuid:` + the UDN's first four groups + a 12-digit subscription number.
#[must_use]
pub fn subscription_id(udn: &str, n: u64) -> String {
    let base = udn.strip_prefix("uuid:").unwrap_or(udn);
    let prefix = base.get(..24).unwrap_or("00000000-0000-0000-0000-");
    format!("uuid:{prefix}{:012x}", n & 0xffff_ffff_ffff)
}

fn send(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    extra: &[(&str, &str)],
    body: &str,
    head_only: bool,
) {
    let mut text = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nServer: {}\r\n",
        body.len(),
        crate::ssdp::server_header(),
    );
    for (name, value) in extra {
        text.push_str(name);
        text.push_str(": ");
        text.push_str(value);
        text.push_str("\r\n");
    }
    text.push_str("Cache-Control: no-store\r\nConnection: close\r\n\r\n");
    if !head_only {
        text.push_str(body);
    }
    let _ = stream.write_all(text.as_bytes());
    let _ = stream.flush();
}

/// `503` on a non-blocking socket, one write attempt, then close. The
/// acceptor never waits on a busy peer.
fn refuse_busy(mut stream: TcpStream) {
    let _ = stream.set_nonblocking(true);
    let status = "503 Service Unavailable";
    let text = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{status}",
        status.len()
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
