//! Shared test helpers: fixture paths, a server on 127.0.0.1:0, a client.
#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use llama_metrics::acl::{Allowlist, Cidr};
use llama_metrics::expo::Rejected;
use llama_metrics::http::{self, Body, Limits, ServerConfig, Stats};

pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

pub fn fixture(name: &str) -> PathBuf {
    workspace_root().join("fixtures/metrics").join(name)
}

pub fn scratch(label: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("llama-metrics-{label}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

pub fn allowlist(nets: &[&str]) -> Allowlist {
    Allowlist::new(
        nets.iter()
            .map(|n| n.parse::<Cidr>().expect("cidr"))
            .collect(),
    )
}

/// Body that reports the rejected counters, so tests can see them.
pub struct EchoBody;

impl Body for EchoBody {
    fn metrics(&self, rejected: Rejected) -> String {
        format!("busy {} denied {}\n", rejected.busy, rejected.denied)
    }
}

pub struct Harness {
    pub addr: SocketAddr,
    pub stats: Arc<Stats>,
    pub ticks: Arc<AtomicUsize>,
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

pub fn spawn(allow: &[&str], max_conns: usize, limits: Limits, body: Arc<dyn Body>) -> Harness {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1:0");
    let addr = listener.local_addr().expect("addr");
    let stats = Arc::new(Stats::default());
    let stop = Arc::new(AtomicBool::new(false));
    let ticks = Arc::new(AtomicUsize::new(0));
    let cfg = ServerConfig {
        allow: allowlist(allow),
        max_conns,
        limits,
        poll_interval: Duration::from_millis(20),
    };
    let join = {
        let stats = Arc::clone(&stats);
        let stop = Arc::clone(&stop);
        let ticks = Arc::clone(&ticks);
        thread::spawn(move || {
            http::serve(listener, cfg, body, stats, stop, || {
                ticks.fetch_add(1, Ordering::SeqCst);
            })
            .expect("serve");
        })
    };
    Harness {
        addr,
        stats,
        ticks,
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
