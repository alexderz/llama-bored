//! #80 through the real poller: a fake llama-swap on an ephemeral
//! loopback port (one thread per connection, so `/api/events` can stay
//! open) serving `/running`, the engine's `/metrics`, SGLang's
//! `/v1/loads`, activity, and an event stream the test pushes frames to.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::TryRecvError;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use llama_core::log::Sink;
use llama_core::sample::LlamaView;
use llama_watch::config::{Config, ValidWatchConfig};
use llama_watch::poller::{self, LlamaDetail};

#[derive(Default)]
struct World {
    running: Vec<u8>,
    metrics: Vec<u8>,
    loads: Vec<u8>,
    /// Event frames, sent in order to every open `/api/events`.
    frames: Vec<String>,
    hits: Vec<String>,
}

struct Fake {
    port: u16,
    world: Arc<Mutex<World>>,
    stop: Arc<AtomicBool>,
}

impl Drop for Fake {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Fake {
    fn start(world: World) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        listener.set_nonblocking(true).expect("nonblocking");
        let world = Arc::new(Mutex::new(world));
        let stop = Arc::new(AtomicBool::new(false));
        let (w, s) = (Arc::clone(&world), Arc::clone(&stop));
        thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let (w, s) = (Arc::clone(&w), Arc::clone(&s));
                        thread::spawn(move || answer(stream, &w, &s));
                    }
                    Err(_) => thread::sleep(Duration::from_millis(5)),
                }
            }
        });
        Self { port, world, stop }
    }

    fn edit(&self, f: impl FnOnce(&mut World)) {
        f(&mut self.world.lock().unwrap_or_else(|e| e.into_inner()));
    }

    fn hits(&self) -> Vec<String> {
        self.world
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .hits
            .clone()
    }
}

fn answer(mut stream: TcpStream, world: &Mutex<World>, stop: &AtomicBool) {
    stream.set_nonblocking(false).ok();
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") && head.len() < 16_384 {
        match stream.read(&mut byte) {
            Ok(1) => head.push(byte[0]),
            _ => return,
        }
    }
    let head = String::from_utf8_lossy(&head).into_owned();
    let path = head.split_whitespace().nth(1).unwrap_or("").to_owned();
    let body = {
        let mut w = world.lock().unwrap_or_else(|e| e.into_inner());
        w.hits.push(path.clone());
        match path.as_str() {
            "/running" => Some(w.running.clone()),
            "/api/metrics/activity" => Some(br#"{"data":[]}"#.to_vec()),
            "/api/events" => None,
            p if p.ends_with("/metrics") => Some(w.metrics.clone()),
            p if p.ends_with("/v1/loads?include=core") => Some(w.loads.clone()),
            _ => Some(b"404 page not found".to_vec()),
        }
    };
    if let Some(body) = body {
        let status = if path.starts_with("/upstream/") || !body.starts_with(b"404") {
            "200 OK"
        } else {
            "404 Not Found"
        };
        let _ = write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let _ = stream.write_all(&body);
        return;
    }
    // The event stream: chunked, held open, each frame once.
    let _ = stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n",
    );
    let mut sent = 0;
    while !stop.load(Ordering::Relaxed) {
        let next: Vec<String> = {
            let w = world.lock().unwrap_or_else(|e| e.into_inner());
            w.frames[sent.min(w.frames.len())..].to_vec()
        };
        for frame in &next {
            if write!(stream, "{:x}\r\n{frame}\r\n", frame.len()).is_err() {
                return;
            }
        }
        sent += next.len();
        let _ = stream.flush();
        thread::sleep(Duration::from_millis(20));
    }
}

fn frame(kind: &str, data: &serde_json::Value) -> String {
    let envelope = serde_json::json!({"type": kind, "data": data.to_string()});
    format!("event:message\ndata:{envelope}\n\n")
}

fn inflight(op: &str, body: serde_json::Value) -> String {
    let mut data = body;
    data["operation"] = serde_json::json!(op);
    frame("inflight", &data)
}

fn entry(id: &str, model: &str, bytes: u64) -> serde_json::Value {
    serde_json::json!({
        "id": id, "timestamp": "2026-10-07T10:00:00Z", "model": model,
        "req_path": "/v1/chat/completions", "method": "POST",
        "req_headers": {"User-Agent": "INVENTED"}, "remote_ip": "192.0.2.9",
        "resp_headers": {}, "resp_bytes": bytes, "elapsed_ms": 0
    })
}

fn config(port: u16, backends: &str) -> ValidWatchConfig {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("live-{port}.toml"));
    let text = format!(
        r#"
[llama]
url = "http://127.0.0.1:{port}"
running_interval_s = 0.35
running_timeout_s = 0.25
metrics_interval_s = 0.2
metrics_timeout_s = 0.15
slots_interval_s = 0.5
slots_timeout_s = 0.3
activity_interval_s = 0.35
activity_timeout_s = 0.2

[llama.backends]
{backends}
"#
    );
    std::fs::write(&path, text).expect("write config");
    Config::load_validated(&path, 8).expect("valid config")
}

#[derive(Clone)]
struct NoLog;

impl Sink for NoLog {
    fn write_line(&mut self, _line: &str) {}
}

fn wait(
    rx: &poller::SampleRx,
    what: &str,
    mut pred: impl FnMut(&LlamaView, &LlamaDetail) -> bool,
) -> (LlamaView, LlamaDetail) {
    let start = Instant::now();
    let mut last = String::new();
    while start.elapsed() < Duration::from_secs(8) {
        match rx.try_recv() {
            Ok((view, detail)) => {
                if pred(&view, &detail) {
                    return (view, detail);
                }
                last = format!("{:?}", detail.requests);
            }
            Err(TryRecvError::Empty) => thread::sleep(Duration::from_millis(10)),
            Err(TryRecvError::Disconnected) => panic!("poller stopped"),
        }
    }
    panic!("timed out: {what}: {last}");
}

fn vllm_metrics(generated: u64, prompt: u64, cached: u64, running: u32) -> Vec<u8> {
    format!(
        "vllm:num_requests_running{{engine=\"0\",model_name=\"q\"}} {running}.0\n\
vllm:num_requests_waiting{{engine=\"0\",model_name=\"q\"}} 0.0\n\
vllm:generation_tokens_total{{engine=\"0\",model_name=\"q\"}} {generated}.0\n\
vllm:prompt_tokens_total{{engine=\"0\",model_name=\"q\"}} {prompt}.0\n\
vllm:prompt_tokens_cached_total{{engine=\"0\",model_name=\"q\"}} {cached}.0\n"
    )
    .into_bytes()
}

fn running(id: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({"running": [{"model": id, "state": "ready"}]}))
        .expect("json")
}

/// One vLLM request from start to end: listed by the event stream, its
/// tokens from the counter deltas of the metrics reads that happen
/// anyway, exact while it runs alone; the in-flight count rides the
/// series; gone once llama-swap removes it.
#[test]
fn a_vllm_request_gets_its_tokens_from_the_metrics_deltas() {
    let fake = Fake::start(World {
        running: running("qwen"),
        metrics: vllm_metrics(1_000, 50_000, 40_000, 0),
        frames: vec![inflight("snapshot", serde_json::json!({"requests": []}))],
        ..World::default()
    });
    let (_poller, rx) =
        poller::spawn(&config(fake.port, r#""qwen" = "vllm""#), NoLog).expect("spawn");
    let (_, detail) = wait(&rx, "stream up", |_, detail| {
        detail.series.first().is_some_and(|s| s.inflight == Some(0))
    });
    assert!(detail.requests.is_empty());
    fake.edit(|w| {
        w.frames.push(inflight(
            "upsert",
            serde_json::json!({"request": entry("30", "qwen", 0)}),
        ));
    });
    let (_, detail) = wait(&rx, "listed", |_, detail| !detail.requests.is_empty());
    assert_eq!(detail.requests[0].id, "30");
    assert_eq!(detail.series[0].inflight, Some(1));
    // Let a read or two credit it with nothing yet, then prefill ends
    // (9,000 prompt, 8,192 cached) and 60 tokens come out.
    thread::sleep(Duration::from_millis(500));
    fake.edit(|w| {
        w.metrics = vllm_metrics(1_060, 59_000, 48_192, 1);
        w.frames.push(inflight(
            "upsert",
            serde_json::json!({"request": entry("30", "qwen", 2_048)}),
        ));
    });
    let (_, detail) = wait(&rx, "credited", |_, detail| {
        detail
            .requests
            .first()
            .and_then(|r| r.tokens)
            .is_some_and(|t| t.output == Some(60))
    });
    let req = &detail.requests[0];
    let tokens = req.tokens.expect("tokens");
    assert_eq!(
        (tokens.prompt, tokens.cached, tokens.approx),
        (Some(9_000), Some(8_192), false)
    );
    assert_eq!(req.resp_bytes, 2_048);
    fake.edit(|w| {
        w.frames
            .push(inflight("remove", serde_json::json!({"id": "30"})))
    });
    wait(&rx, "removed", |_, detail| {
        detail.requests.is_empty() && detail.series[0].inflight == Some(0)
    });
    let hits = fake.hits();
    assert_eq!(
        hits.iter().filter(|h| *h == "/api/events").count(),
        1,
        "one connection"
    );
    assert!(
        hits.iter().all(|h| !h.contains("v1/loads")),
        "vLLM needs no loads read"
    );
}

/// SGLang: `/v1/loads` is read through the fresh gate, only while the
/// stream lists a request of the model, and its KV and rate go to it.
#[test]
fn sglang_loads_are_read_only_while_a_request_is_in_flight() {
    let loads = serde_json::json!({
        "loads": [{"dp_rank": 0, "num_running_reqs": 1, "num_used_tokens": 30000,
                   "gen_throughput": 41.5, "total_prefill_uncached_tokens": 5}]
    });
    let fake = Fake::start(World {
        running: running("sg"),
        metrics: b"sglang:num_running_reqs{model_name=\"sg\"} 1.0\nsglang:generation_tokens_total{model_name=\"sg\"} 5.0\n".to_vec(),
        loads: serde_json::to_vec(&loads).expect("json"),
        frames: vec![inflight("snapshot", serde_json::json!({"requests": []}))],
        ..World::default()
    });
    let (_poller, rx) =
        poller::spawn(&config(fake.port, r#""sg" = "sglang""#), NoLog).expect("spawn");
    wait(&rx, "stream up", |_, detail| {
        detail.series.first().is_some_and(|s| s.inflight == Some(0))
    });
    thread::sleep(Duration::from_millis(600));
    assert!(
        fake.hits().iter().all(|h| !h.contains("v1/loads")),
        "no request in flight, no loads read"
    );
    fake.edit(|w| {
        w.frames.push(inflight(
            "upsert",
            serde_json::json!({"request": entry("7", "sg", 0)}),
        ));
    });
    let (_, detail) = wait(&rx, "credited", |_, detail| {
        detail
            .requests
            .first()
            .and_then(|r| r.tokens)
            .is_some_and(|t| t.held.is_some())
    });
    let tokens = detail.requests[0].tokens.expect("tokens");
    assert_eq!(
        (tokens.held, tokens.gen_tps, tokens.approx),
        (Some(30_000), Some(41.5), false)
    );
    // Every loads read sits between two `/running` reads: the fresh gate.
    let hits: Vec<String> = fake
        .hits()
        .into_iter()
        .filter(|h| h != "/api/events" && h != "/api/metrics/activity")
        .collect();
    let mut seen = 0;
    for (i, hit) in hits.iter().enumerate() {
        if hit.ends_with("/v1/loads?include=core") {
            seen += 1;
            assert_eq!(hit, "/upstream/sg/v1/loads?include=core");
            assert_eq!(hits[i - 1], "/running", "{hits:?}");
            if let Some(next) = hits.get(i + 1) {
                assert_eq!(next, "/running", "{hits:?}");
            }
        }
    }
    assert!(seen > 0);
}
