//! Shared test helpers: fixture paths, a device, a server on 127.0.0.1:0
//! with a fake stream, and a client.
#![allow(dead_code)]

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use llama_cast::acl::{Allowlist, Cidr};
use llama_cast::dlna::Device;
use llama_cast::http::{self, App, Limits, Live, ServerConfig, Stats};

pub const UDN: &str = "uuid:5a1e0c2d-7b3f-5e41-9c8d-2f6a4b1e0d93";
pub const BASE: &str = "http://192.168.1.20:19478";

pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

pub fn fixture(name: &str) -> PathBuf {
    workspace_root().join("fixtures/cast").join(name)
}

pub fn scratch(label: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("llama-cast-{label}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

pub fn device() -> Device {
    Device {
        name: "llama-bored".to_owned(),
        udn: UDN.to_owned(),
        base_url: BASE.to_owned(),
    }
}

pub fn allowlist(nets: &[&str]) -> Allowlist {
    Allowlist::new(
        nets.iter()
            .map(|n| n.parse::<Cidr>().expect("cidr"))
            .collect(),
    )
}

/// A stream that sends `chunk` every 10 ms until dropped, and counts the
/// open and dropped streams.
pub struct FakeLive {
    pub opened: Arc<AtomicUsize>,
    pub closed: Arc<AtomicUsize>,
    pub fail: bool,
}

impl FakeLive {
    pub fn new() -> Self {
        Self {
            opened: Arc::new(AtomicUsize::new(0)),
            closed: Arc::new(AtomicUsize::new(0)),
            fail: false,
        }
    }
}

struct FakeStream {
    closed: Arc<AtomicUsize>,
}

impl Read for FakeStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        thread::sleep(Duration::from_millis(10));
        let chunk = [0x47_u8; 188];
        let n = chunk.len().min(buf.len());
        buf[..n].copy_from_slice(&chunk[..n]);
        Ok(n)
    }
}

impl Drop for FakeStream {
    fn drop(&mut self) {
        self.closed.fetch_add(1, Ordering::SeqCst);
    }
}

impl Live for FakeLive {
    fn open(&self) -> io::Result<Box<dyn Read + Send>> {
        if self.fail {
            return Err(io::Error::other("no encoder"));
        }
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(FakeStream {
            closed: Arc::clone(&self.closed),
        }))
    }
}

pub struct Harness {
    pub addr: SocketAddr,
    pub stats: Arc<Stats>,
    pub ticks: Arc<AtomicUsize>,
    pub opened: Arc<AtomicUsize>,
    pub closed: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub fn fast_limits() -> Limits {
    Limits {
        header_timeout: Duration::from_millis(300),
        write_timeout: Duration::from_millis(500),
        stream_write_timeout: Duration::from_millis(500),
        ..Limits::default()
    }
}

pub fn spawn(allow: &[&str], max_clients: usize, limits: Limits, live: FakeLive) -> Harness {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1:0");
    let addr = listener.local_addr().expect("addr");
    let stats = Arc::new(Stats::default());
    let stop = Arc::new(AtomicBool::new(false));
    let ticks = Arc::new(AtomicUsize::new(0));
    let opened = Arc::clone(&live.opened);
    let closed = Arc::clone(&live.closed);
    let cfg = ServerConfig {
        allow: allowlist(allow),
        max_clients,
        limits,
        poll_interval: Duration::from_millis(20),
    };
    let app = Arc::new(App {
        device: device(),
        live: Arc::new(live),
    });
    let join = {
        let stats = Arc::clone(&stats);
        let stop = Arc::clone(&stop);
        let ticks = Arc::clone(&ticks);
        thread::spawn(move || {
            http::serve(listener, cfg, app, stats, stop, || {
                ticks.fetch_add(1, Ordering::SeqCst);
            })
            .expect("serve");
        })
    };
    Harness {
        addr,
        stats,
        ticks,
        opened,
        closed,
        stop,
        join: Some(join),
    }
}

/// Send `request`, then read until the server closes (at most 5 s).
pub fn exchange(addr: SocketAddr, request: &[u8]) -> String {
    let mut stream = TcpStream::connect(addr).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("timeout");
    let _ = stream.write_all(request);
    read_all(&mut stream)
}

pub fn read_all(stream: &mut TcpStream) -> String {
    let mut out = Vec::new();
    let mut buf = [0_u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => out.extend_from_slice(&buf[..n]),
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn wait_until(what: &str, budget: Duration, mut done: impl FnMut() -> bool) {
    let start = Instant::now();
    while !done() {
        assert!(start.elapsed() < budget, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(5));
    }
}

pub fn status_line(response: &str) -> &str {
    response.lines().next().unwrap_or("")
}

/// The body after the blank line.
pub fn body(response: &str) -> &str {
    response.split_once("\r\n\r\n").map_or("", |(_, b)| b)
}
