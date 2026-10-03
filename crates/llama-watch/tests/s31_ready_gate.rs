//! #31: llama-swap loads a model on any `/upstream/<id>/…` request, so the
//! watcher may request one only for a model that `/running` lists as
//! `ready`. A fake llama-swap records every path in order, and for each
//! upstream request checks the id against the `ready` set of the newest
//! `/running` body it served (the poller is one thread making one request
//! at a time, so that body is the one the poller acted on). The script runs
//! models through starting, ready, stopping, gone, a failed `/running` and
//! back, with backends that take every upstream read: the `/metrics`
//! detection probe, vLLM and llama.cpp `/metrics`, and `/slots`.

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use llama_core::log::Sink;
use llama_watch::config::{Config, ValidWatchConfig};
use llama_watch::poller;

/// A container whose image runs `vllm serve`: the command names no server.
const CONTAINER: &str = "podman run --rm --name hyperqwen ghcr.io/example/hyperqwen single";
const LLAMA: &str = "/opt/llama/llama-server -m /models/x-Q4_K_M.gguf -c 8192";

struct Fake {
    /// `/running` status and body.
    status: u16,
    running: serde_json::Value,
    /// `ready` ids of the newest `/running` the fake served; empty after a
    /// failed one (the poller then reads nothing upstream).
    ready: HashSet<String>,
    metrics: HashMap<String, Vec<u8>>,
    hits: Vec<String>,
    /// Upstream requests for a model that was not ready.
    violations: Vec<String>,
}

struct Server {
    port: u16,
    fake: Arc<Mutex<Fake>>,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl Server {
    fn start(fake: Fake) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let port = listener.local_addr().expect("addr").port();
        let fake = Arc::new(Mutex::new(fake));
        let stop = Arc::new(AtomicBool::new(false));
        let (fake_thread, stop_thread) = (Arc::clone(&fake), Arc::clone(&stop));
        let join = thread::spawn(move || {
            while !stop_thread.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => serve(stream, &fake_thread),
                    Err(_) => thread::sleep(Duration::from_millis(2)),
                }
            }
        });
        Self {
            port,
            fake,
            stop,
            join: Some(join),
        }
    }

    fn with<T>(&self, f: impl FnOnce(&mut Fake) -> T) -> T {
        f(&mut self.fake.lock().unwrap_or_else(|err| err.into_inner()))
    }

    /// Serve `running` from now on, then wait until it was served twice,
    /// so the poller has acted on it.
    fn set_running(&self, status: u16, running: serde_json::Value) {
        let before = self.with(|fake| {
            fake.status = status;
            fake.running = running;
            fake.hits.iter().filter(|p| *p == "/running").count()
        });
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(3) {
            let served = self.with(|fake| fake.hits.iter().filter(|p| *p == "/running").count());
            if served >= before + 2 {
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        panic!("the poller stopped reading /running");
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn serve(mut stream: TcpStream, fake: &Mutex<Fake>) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let mut head = Vec::new();
    let mut buf = [0u8; 1024];
    while !head.windows(4).any(|w| w == b"\r\n\r\n") && head.len() < 16 * 1024 {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => head.extend_from_slice(&buf[..n]),
        }
    }
    let head = String::from_utf8_lossy(&head);
    let path = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("")
        .to_owned();
    let (status, body) = {
        let mut fake = fake.lock().unwrap_or_else(|err| err.into_inner());
        fake.hits.push(path.clone());
        if path == "/running" {
            fake.ready = if fake.status == 200 {
                ready_ids(&fake.running)
            } else {
                HashSet::new()
            };
            (
                fake.status,
                serde_json::to_vec(&fake.running).expect("json"),
            )
        } else if let Some(rest) = path.strip_prefix("/upstream/") {
            let (id, leaf) = rest.split_once('/').unwrap_or((rest, ""));
            if !fake.ready.contains(id) {
                fake.violations.push(path.clone());
            }
            match leaf {
                "metrics" => (
                    200,
                    fake.metrics
                        .get(id)
                        .cloned()
                        .unwrap_or_else(|| b"\n".to_vec()),
                ),
                "slots" => (200, slots_body()),
                _ => (404, Vec::new()),
            }
        } else if path == "/api/metrics/activity" {
            (200, br#"{"data":[]}"#.to_vec())
        } else {
            (404, Vec::new())
        }
    };
    let reason = if status == 200 { "OK" } else { "Error" };
    let _ = stream.write_all(
        format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .as_bytes(),
    );
    let _ = stream.write_all(&body);
}

fn ready_ids(running: &serde_json::Value) -> HashSet<String> {
    running["running"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|model| model["state"] == "ready")
        .filter_map(|model| model["model"].as_str().map(str::to_owned))
        .collect()
}

fn slots_body() -> Vec<u8> {
    br#"[{"id":0,"id_task":1,"n_ctx":8192,"is_processing":true,"next_token":[{"n_decoded":5}]}]"#
        .to_vec()
}

fn running(models: &[(&str, &str, &str)]) -> serde_json::Value {
    let models: Vec<serde_json::Value> = models
        .iter()
        .map(|(id, state, cmd)| serde_json::json!({"model": id, "state": state, "cmd": cmd}))
        .collect();
    serde_json::json!({ "running": models })
}

fn vllm_body() -> Vec<u8> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/llama/vllm-metrics.txt");
    std::fs::read(path).expect("vllm fixture")
}

fn llama_body() -> Vec<u8> {
    b"llamacpp:prompt_tokens_total 10\nllamacpp:n_decode_total 5\nllamacpp:requests_processing 1\n"
        .to_vec()
}

fn config(port: u16) -> ValidWatchConfig {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("s31-{port}.toml"));
    std::fs::write(
        &path,
        format!(
            r#"
[llama]
url = "http://127.0.0.1:{port}"
running_interval_s = 0.2
running_timeout_s = 0.15
metrics_interval_s = 0.2
metrics_timeout_s = 0.15
slots_interval_s = 0.5
slots_timeout_s = 0.3
activity_interval_s = 0.35
activity_timeout_s = 0.1
"#
        ),
    )
    .expect("write config");
    Config::load_validated(&path, 8).expect("valid config")
}

struct NoLog;

impl Sink for NoLog {
    fn write_line(&mut self, _line: &str) {}
}

#[test]
fn no_upstream_request_for_a_model_that_is_not_ready() {
    let phase1 = running(&[
        ("vllm-a", "ready", CONTAINER),
        ("boot-b", "starting", CONTAINER),
        ("stop-c", "stopping", LLAMA),
        ("gone-d", "ready", LLAMA),
        ("tabby-e", "ready", "python3 main.py --port 5000"),
    ]);
    let mut metrics = HashMap::new();
    for id in ["vllm-a", "boot-b"] {
        metrics.insert(id.to_owned(), vllm_body());
    }
    for id in ["stop-c", "gone-d"] {
        metrics.insert(id.to_owned(), llama_body());
    }
    let server = Server::start(Fake {
        status: 200,
        running: phase1.clone(),
        ready: HashSet::new(),
        metrics,
        hits: Vec::new(),
        violations: Vec::new(),
    });
    let config = config(server.port);
    let (poller, _rx) = poller::spawn(&config, NoLog).expect("spawn");

    // Phase 1 runs until every ready model was read, /slots included.
    let start = Instant::now();
    let wanted = [
        "/upstream/vllm-a/metrics",
        "/upstream/gone-d/metrics",
        "/upstream/gone-d/slots",
        "/upstream/tabby-e/metrics",
    ];
    while !server.with(|fake| wanted.iter().all(|w| fake.hits.iter().any(|h| h == w))) {
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "{:?}",
            server.with(|f| f.hits.clone())
        );
        thread::sleep(Duration::from_millis(20));
    }
    thread::sleep(Duration::from_millis(300));
    // gone-d disappears, vllm-a stops, boot-b is ready (and gets probed).
    server.set_running(
        200,
        running(&[
            ("vllm-a", "stopping", CONTAINER),
            ("boot-b", "ready", CONTAINER),
            ("stop-c", "stopping", LLAMA),
        ]),
    );
    thread::sleep(Duration::from_millis(400));
    // llama-swap fails: nothing upstream until it answers again.
    server.set_running(500, phase1.clone());
    thread::sleep(Duration::from_millis(400));
    // Back with only loading and unloading models, then an empty list.
    server.set_running(
        200,
        running(&[
            ("boot-b", "starting", CONTAINER),
            ("stop-c", "stopping", LLAMA),
        ]),
    );
    thread::sleep(Duration::from_millis(400));
    server.set_running(200, running(&[]));
    thread::sleep(Duration::from_millis(300));
    drop(poller);

    let (hits, violations) = server.with(|fake| (fake.hits.clone(), fake.violations.clone()));
    assert!(violations.is_empty(), "not ready: {violations:?}\n{hits:?}");
    assert!(
        hits.iter().all(|path| !path.contains("stop-c")),
        "a stopping model is never read: {hits:?}"
    );
    assert!(
        hits.iter().any(|path| path == "/upstream/boot-b/metrics"),
        "boot-b was read once ready: {hits:?}"
    );
    // After gone-d left /running, nothing more for it.
    let gone_at = hits
        .iter()
        .rposition(|path| path.starts_with("/upstream/gone-d/"))
        .expect("gone-d was read");
    let switched = hits
        .iter()
        .position(|path| path == "/upstream/boot-b/metrics")
        .expect("boot-b read");
    assert!(gone_at < switched, "{hits:?}");
}
