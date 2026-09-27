//! A sentinel in llama text must not reach the log sink.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use llama_core::log::Sink;
use llama_watch::config::{Config, ValidWatchConfig};
use llama_watch::poller;

const SENTINEL: &str = "SENTINEL_LLAMA_TEXT_9f3a2c7e1b6d";

#[derive(Clone)]
struct MemLog(Arc<Mutex<Vec<String>>>);

impl Sink for MemLog {
    fn write_line(&mut self, line: &str) {
        self.0
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(line.to_owned());
    }
}

impl MemLog {
    fn lines(&self) -> Vec<String> {
        self.0.lock().unwrap_or_else(|err| err.into_inner()).clone()
    }
}

struct World {
    running_status: u16,
    running: Vec<u8>,
    metrics_status: u16,
    metrics: Vec<u8>,
    slots_status: u16,
    slots: Vec<u8>,
    activity_status: u16,
    activity: Vec<u8>,
}

struct Server {
    port: u16,
    hits: Arc<Mutex<Vec<String>>>,
    world: Arc<Mutex<World>>,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl Server {
    fn start(world: World) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let hits = Arc::new(Mutex::new(Vec::new()));
        let world = Arc::new(Mutex::new(world));
        let stop = Arc::new(AtomicBool::new(false));
        let hits_t = Arc::clone(&hits);
        let world_t = Arc::clone(&world);
        let stop_t = Arc::clone(&stop);
        let join = thread::spawn(move || serve(listener, hits_t, world_t, stop_t));
        Self {
            port,
            hits,
            world,
            stop,
            join: Some(join),
        }
    }

    fn hits(&self) -> Vec<String> {
        self.hits
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone()
    }

    fn update(&self, edit: impl FnOnce(&mut World)) {
        edit(&mut self.world.lock().unwrap_or_else(|err| err.into_inner()));
    }

    fn addr(&self) -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], self.port))
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = TcpStream::connect_timeout(&self.addr(), Duration::from_millis(200));
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn serve(
    listener: TcpListener,
    hits: Arc<Mutex<Vec<String>>>,
    world: Arc<Mutex<World>>,
    stop: Arc<AtomicBool>,
) {
    while !stop.load(Ordering::Relaxed) {
        let Some(mut stream) = accept_for(&listener, Duration::from_millis(200)) else {
            continue;
        };
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let head = read_headers(&mut stream);
        let path = request_path(&head);
        hits.lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(path.clone());
        let (status, body) = {
            let world = world.lock().unwrap_or_else(|err| err.into_inner());
            if path.ends_with("/running") {
                (world.running_status, world.running.clone())
            } else if path.contains("/api/metrics/activity") {
                (world.activity_status, world.activity.clone())
            } else if path.contains("/slots") {
                (world.slots_status, world.slots.clone())
            } else if path.contains("/metrics") {
                (world.metrics_status, world.metrics.clone())
            } else {
                (404, Vec::new())
            }
        };
        write_response(&mut stream, status, &body);
    }
}

fn accept_for(listener: &TcpListener, budget: Duration) -> Option<TcpStream> {
    listener.set_nonblocking(true).ok()?;
    let start = Instant::now();
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false).ok()?;
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
                return Some(stream);
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                if start.elapsed() >= budget {
                    return None;
                }
                thread::sleep(Duration::from_millis(5));
            }
            Err(_) => return None,
        }
    }
}

fn read_headers(stream: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0_u8; 512];
    while !buf.windows(4).any(|window| window == b"\r\n\r\n") {
        match stream.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
            Err(_) => break,
        }
        if buf.len() > 16_384 {
            break;
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

fn request_path(head: &str) -> String {
    head.lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("")
        .to_owned()
}

fn write_response(stream: &mut TcpStream, status: u16, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {status} OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
}

fn config(port: u16) -> ValidWatchConfig {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("watch-log-{port}.toml"));
    let text = format!(
        r#"
[llama]
url = "http://127.0.0.1:{port}"
running_interval_s = 0.3
running_timeout_s = 0.12
metrics_interval_s = 0.2
metrics_timeout_s = 0.08
slots_interval_s = 0.5
slots_timeout_s = 0.2
slots_max_bytes = 2048
activity_interval_s = 0.3
activity_timeout_s = 0.1
input_tail_chars = 256
output_tail_chars = 256
"#
    );
    std::fs::write(&path, text).expect("write");
    Config::load_validated(&path, 8).expect("config")
}

fn wait_hits(server: &Server, pred: impl Fn(&[String]) -> bool) {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(3) {
        if pred(&server.hits()) {
            return;
        }
        thread::sleep(Duration::from_millis(15));
    }
    panic!("paths not seen: {:?}", server.hits());
}

fn wait_log(log: &MemLog, needle: &str) {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(3) {
        if log.lines().iter().any(|line| line.contains(needle)) {
            return;
        }
        thread::sleep(Duration::from_millis(15));
    }
    panic!("log missing {needle}: {:?}", log.lines());
}

fn happy_world() -> World {
    let running = serde_json::to_vec(&serde_json::json!({
        "running": [{
            "model": "qwen3.6-35b-a3b",
            "name": "Qwen",
            "state": "ready",
            "cmd": SENTINEL
        }],
        "note": SENTINEL
    }))
    .expect("json");
    let metrics = format!(
        "# {SENTINEL}\nllamacpp:n_decode_total 10\nllamacpp:requests_processing 1\nllamacpp:prompt_tokens_total{{note=\"{SENTINEL}\"}} 3\n"
    );
    let slots = serde_json::to_vec(&serde_json::json!([{
        "id": 0,
        "id_task": 1,
        "is_processing": true,
        "n_prompt_tokens": 4,
        "n_prompt_tokens_processed": 2,
        "next_token": [{"n_decoded": 1}],
        "prompt": format!("hello {SENTINEL}"),
        "generated": format!("world {SENTINEL}")
    }]))
    .expect("json");
    let activity = serde_json::to_vec(&serde_json::json!({
        "data": [{
            "id": 1,
            "timestamp": "2026-09-24T00:00:00Z",
            "model": "qwen3.6-35b-a3b",
            "src": "ip",
            "tokens": {
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_tokens": 0,
                "prompt_per_second": 1.0,
                "tokens_per_second": 1.0
            },
            "duration_ms": 5,
            "resp_status_code": 200,
            "error_msg": SENTINEL
        }],
        "page": 1
    }))
    .expect("json");
    World {
        running_status: 200,
        running,
        metrics_status: 200,
        metrics: metrics.into_bytes(),
        slots_status: 200,
        slots,
        activity_status: 200,
        activity,
    }
}

fn oversize_slots() -> Vec<u8> {
    let cap = 2048usize;
    let prefix = format!(
        "[{{\"id\":0,\"id_task\":2,\"is_processing\":true,\"n_prompt_tokens\":8,\"n_prompt_tokens_processed\":3,\"next_token\":[{{\"n_decoded\":2}}],\"prompt\":\"{SENTINEL}"
    );
    let suffix = format!("\",\"generated\":\"{SENTINEL}\"}}]");
    let mut body = prefix.into_bytes();
    let fill = (cap + 1) - body.len() - suffix.len();
    body.extend(std::iter::repeat_n(b'Z', fill));
    body.extend_from_slice(suffix.as_bytes());
    assert!(body.len() > cap);
    assert!(
        body.windows(SENTINEL.len())
            .any(|window| window == SENTINEL.as_bytes())
    );
    body
}

#[test]
fn sentinel_is_absent_from_every_tap_error_and_oversize_log() {
    let server = Server::start(happy_world());
    let log = MemLog(Arc::new(Mutex::new(Vec::new())));
    let config = config(server.port);
    let (_poller, _rx) = poller::spawn(&config, log.clone()).expect("spawn");

    wait_hits(&server, |hits| {
        hits.iter().any(|path| path.ends_with("/running"))
            && hits
                .iter()
                .any(|path| path.contains("/upstream/") && path.ends_with("/metrics"))
            && hits.iter().any(|path| path.contains("/slots"))
            && hits
                .iter()
                .any(|path| path.contains("/api/metrics/activity"))
    });

    server.update(|world| {
        world.slots_status = 500;
        world.slots = SENTINEL.as_bytes().to_vec();
    });
    wait_log(&log, "slots:");

    server.update(|world| {
        world.slots_status = 200;
        world.slots = oversize_slots();
    });
    wait_log(&log, "body exceeds cap");

    server.update(|world| {
        world.metrics_status = 500;
        world.metrics = SENTINEL.as_bytes().to_vec();
        world.activity = SENTINEL.as_bytes().to_vec();
    });
    wait_log(&log, "metrics:");
    wait_log(&log, "activity:");

    server.update(|world| {
        world.running = SENTINEL.as_bytes().to_vec();
    });
    wait_log(&log, "llama: down");

    let lines = log.lines();
    assert!(!lines.is_empty());
    for line in &lines {
        assert!(!line.contains(SENTINEL), "sentinel leaked: {line}");
    }
}
