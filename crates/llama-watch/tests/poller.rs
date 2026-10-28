//! Fake llama-swap on an ephemeral loopback port. No live :8080.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::TryRecvError;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use llama_core::log::Sink;
use llama_core::sample::{AiState, LlamaView};
use llama_watch::config::{Config, ValidWatchConfig};
use llama_watch::poller::{self, LlamaDetail};
use llama_watch::tty::grid::Cell;

const MARKER: &str = "ENDMARKER";

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/llama")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn watch(
    port: u16,
    max_name_chars: u32,
    slots_max_bytes: u64,
    running_timeout_s: f64,
) -> ValidWatchConfig {
    watch_with(port, max_name_chars, slots_max_bytes, running_timeout_s, "")
}

fn watch_with(
    port: u16,
    max_name_chars: u32,
    slots_max_bytes: u64,
    running_timeout_s: f64,
    extra: &str,
) -> ValidWatchConfig {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let path = dir.join(format!("watch-{port}.toml"));
    let text = format!(
        r#"
[llama]
url = "http://127.0.0.1:{port}"
running_interval_s = 0.35
running_timeout_s = {running_timeout_s}
metrics_interval_s = 0.2
metrics_timeout_s = 0.1
slots_interval_s = 0.5
slots_timeout_s = 0.3
slots_max_bytes = {slots_max_bytes}
activity_interval_s = 0.35
activity_timeout_s = 0.1
input_tail_chars = 256
output_tail_chars = 256

[models]
max_name_chars = {max_name_chars}

[models.aliases]
"qwen3.6-35b-a3b" = "Qwen 35B"
{extra}
"#
    );
    std::fs::write(&path, text).expect("write watch.toml");
    Config::load_validated(&path, 8).expect("valid watch config")
}

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
    fn new() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }

    fn lines(&self) -> Vec<String> {
        self.0.lock().unwrap_or_else(|err| err.into_inner()).clone()
    }
}

struct World {
    running_status: u16,
    running: Vec<u8>,
    /// Returned for every `/running` after the first. `None` repeats `running`.
    running_after: Option<Vec<u8>>,
    running_served: u32,
    location: Option<String>,
    metrics_status: u16,
    metrics: HashMap<String, Vec<u8>>,
    slots: HashMap<String, Vec<u8>>,
    activity_status: u16,
    activity: Vec<u8>,
    /// `/api/captures/<id>` bodies (#5).
    captures: HashMap<String, Vec<u8>>,
    hang: bool,
    /// `/running` body by call number (1-based), over `running` (#70).
    script: Option<fn(u32) -> Vec<u8>>,
    /// Status and `Location` for an upstream path, over the defaults (#70).
    upstream_status: HashMap<String, (u16, Option<String>)>,
    /// Upstream paths answered this much later (#70).
    upstream_delay: HashMap<String, Duration>,
    /// When each upstream request arrived (#70).
    upstream_at: Vec<(String, Instant)>,
    /// Becomes `running_after` once an upstream request is served (#70).
    after_upstream: Option<Vec<u8>>,
    /// tqLLM's `/upstream/<id>/api/monitor/v1/snapshot` bodies (#180).
    snapshots: HashMap<String, Vec<u8>>,
    /// tqLLM's `/upstream/<id>/health` bodies (#180).
    health: HashMap<String, Vec<u8>>,
    /// `METHOD path` of every request, in order (#180).
    requests: Vec<String>,
}

impl World {
    fn running(body: Vec<u8>) -> Self {
        Self {
            running_status: 200,
            running: body,
            running_after: None,
            running_served: 0,
            location: None,
            metrics_status: 200,
            metrics: HashMap::new(),
            slots: HashMap::new(),
            activity_status: 200,
            activity: br#"{"data":[]}"#.to_vec(),
            captures: HashMap::new(),
            hang: false,
            script: None,
            upstream_status: HashMap::new(),
            upstream_delay: HashMap::new(),
            upstream_at: Vec::new(),
            after_upstream: None,
            snapshots: HashMap::new(),
            health: HashMap::new(),
            requests: Vec::new(),
        }
    }
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
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1:0");
        let port = listener.local_addr().expect("local addr").port();
        let hits = Arc::new(Mutex::new(Vec::new()));
        let world = Arc::new(Mutex::new(world));
        let stop = Arc::new(AtomicBool::new(false));
        let hits_thread = Arc::clone(&hits);
        let world_thread = Arc::clone(&world);
        let stop_thread = Arc::clone(&stop);
        let join = thread::spawn(move || serve(listener, hits_thread, world_thread, stop_thread));
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

    fn requests(&self) -> Vec<String> {
        self.world
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .requests
            .clone()
    }

    fn update(&self, edit: impl FnOnce(&mut World)) {
        let mut world = self.world.lock().unwrap_or_else(|err| err.into_inner());
        edit(&mut world);
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

impl Server {
    fn addr(&self) -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], self.port))
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
        let (action, delay) = {
            let mut world = world.lock().unwrap_or_else(|err| err.into_inner());
            let method = head.split_whitespace().next().unwrap_or("").to_owned();
            world.requests.push(format!("{method} {path}"));
            (
                reply(&mut world, &path),
                world.upstream_delay.get(&path).copied(),
            )
        };
        if let Some(delay) = delay {
            thread::sleep(delay);
        }
        match action {
            Action::Respond {
                status,
                reason,
                body,
                location,
            } => write_response(&mut stream, status, reason, &body, location.as_deref()),
            Action::Hang => {
                while !stop.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(20));
                }
            }
        }
    }
}

enum Action {
    Respond {
        status: u16,
        reason: &'static str,
        body: Vec<u8>,
        location: Option<String>,
    },
    Hang,
}

fn reply(world: &mut World, path: &str) -> Action {
    if world.hang {
        return Action::Hang;
    }
    if path.ends_with("/running") {
        world.running_served += 1;
        let body = if let Some(script) = world.script {
            script(world.running_served)
        } else if world.running_served == 1 {
            world.running.clone()
        } else {
            world
                .running_after
                .clone()
                .unwrap_or_else(|| world.running.clone())
        };
        return Action::Respond {
            status: world.running_status,
            reason: if world.running_status == 200 {
                "OK"
            } else {
                "Found"
            },
            body,
            location: world.location.clone(),
        };
    }
    if path.contains("/api/metrics/activity") {
        return Action::Respond {
            status: world.activity_status,
            reason: "OK",
            body: world.activity.clone(),
            location: None,
        };
    }
    if path == "/api/events" {
        return Action::Respond {
            status: 200,
            reason: "OK",
            body: Vec::new(),
            location: None,
        };
    }
    // #167: the events thread's one read before each stream; a build
    // without reservations.
    if path == "/api/gpu" {
        return Action::Respond {
            status: 404,
            reason: "Not Found",
            body: Vec::new(),
            location: None,
        };
    }
    if let Some(id) = path.strip_prefix("/api/captures/") {
        return match world.captures.get(id) {
            Some(body) => Action::Respond {
                status: 200,
                reason: "OK",
                body: body.clone(),
                location: None,
            },
            None => Action::Respond {
                status: 404,
                reason: "Not Found",
                body: br#"{"error":{"message":"capture not found"}}"#.to_vec(),
                location: None,
            },
        };
    }
    if path.starts_with("/upstream/") {
        world.upstream_at.push((path.to_owned(), Instant::now()));
        if let Some(body) = world.after_upstream.take() {
            world.running_after = Some(body);
        }
        if let Some((status, location)) = world.upstream_status.get(path) {
            return Action::Respond {
                status: *status,
                reason: "Status",
                body: b"model x is not loaded; path matches upstream.ignorePaths".to_vec(),
                location: location.clone(),
            };
        }
    }
    // #180: tqLLM's two leaves; a model without a body answers 404.
    let tqllm = path
        .strip_prefix("/upstream/")
        .and_then(|rest| rest.strip_suffix("/api/monitor/v1/snapshot"))
        .map(|model| world.snapshots.get(model))
        .or_else(|| upstream_model(path, "health").map(|model| world.health.get(model)));
    if let Some(body) = tqllm {
        return match body {
            Some(body) => Action::Respond {
                status: 200,
                reason: "OK",
                body: body.clone(),
                location: None,
            },
            None => Action::Respond {
                status: 404,
                reason: "Not Found",
                body: Vec::new(),
                location: None,
            },
        };
    }
    if let Some(model) = upstream_model(path, "metrics") {
        let body = world
            .metrics
            .get(model)
            .cloned()
            .unwrap_or_else(|| b"\n".to_vec());
        return Action::Respond {
            status: world.metrics_status,
            reason: "OK",
            body,
            location: None,
        };
    }
    if let Some(model) = upstream_model(path, "slots") {
        let body = world
            .slots
            .get(model)
            .cloned()
            .unwrap_or_else(|| b"[]".to_vec());
        return Action::Respond {
            status: 200,
            reason: "OK",
            body,
            location: None,
        };
    }
    Action::Respond {
        status: 404,
        reason: "Not Found",
        body: Vec::new(),
        location: None,
    }
}

fn upstream_model<'a>(path: &'a str, leaf: &str) -> Option<&'a str> {
    let rest = path.split_once("/upstream/")?.1;
    let (model, got) = rest.rsplit_once('/')?;
    (got == leaf).then_some(model)
}

fn accept_for(listener: &TcpListener, budget: Duration) -> Option<TcpStream> {
    listener.set_nonblocking(true).ok()?;
    let start = Instant::now();
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false).ok()?;
                let _ = stream.set_nodelay(true);
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

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    body: &[u8],
    location: Option<&str>,
) {
    let mut head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    if let Some(location) = location {
        head.push_str("Location: ");
        head.push_str(location);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
}

fn spawn(config: &ValidWatchConfig, log: &MemLog) -> (poller::Poller, poller::SampleRx) {
    poller::spawn(config, log.clone()).expect("spawn poller")
}

fn cells(cells: &[Cell]) -> String {
    cells.iter().map(|cell| cell.ch).collect()
}

fn wait_msg(
    rx: &poller::SampleRx,
    timeout: Duration,
    mut pred: impl FnMut(&LlamaView, &LlamaDetail) -> bool,
) -> (LlamaView, LlamaDetail) {
    let start = Instant::now();
    let mut last = String::from("no message");
    while start.elapsed() < timeout {
        match rx.try_recv() {
            Ok((view, detail)) => {
                if pred(&view, &detail) {
                    return (view, detail);
                }
                last = format!(
                    "ai={:?} decoded={:?} models={:?}",
                    view.ai, view.decoded_total, view.models
                );
            }
            Err(TryRecvError::Empty) => thread::sleep(Duration::from_millis(10)),
            Err(TryRecvError::Disconnected) => panic!("poller stopped: {last}"),
        }
    }
    panic!("timed out waiting for a poll: {last}");
}

fn metrics_body(decode: u64, processing: f64) -> Vec<u8> {
    format!(
        "llamacpp:prompt_tokens_total 1\nllamacpp:n_decode_total {decode}\nllamacpp:n_decode_total NaN\nllamacpp:requests_processing {processing}\nextra_line 4\n"
    )
    .into_bytes()
}

fn running_model(id: &str, name: &str, state: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "running": [{"model": id, "name": name, "state": state, "cmd": "llama-server not-stored"}]
    }))
    .expect("json")
}

#[test]
fn running_idle_is_measured_zero_and_skips_upstream() {
    let server = Server::start(World::running(br#"{"running":[]}"#.to_vec()));
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, detail) = wait_msg(&rx, Duration::from_secs(2), |view, detail| {
        view.ai == AiState::Idle && view.decoded_total == Some(0) && detail.prompt_tps == Some(0.0)
    });
    assert!(view.models.is_empty());
    assert!(detail.slots.is_empty());
    let hits = server.hits();
    assert!(hits.iter().any(|path| path.ends_with("/running")));
    assert!(
        hits.iter()
            .any(|path| path.contains("/api/metrics/activity"))
    );
    assert!(
        hits.iter().all(|path| !path.contains("/upstream/")),
        "{hits:?}"
    );
}

#[test]
fn running_down_on_refused_malformed_and_redirect() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);
    let log = MemLog::new();
    let config = watch(port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.ai == AiState::Down
    });
    assert_eq!(view.decoded_total, None);
    assert!(view.models.is_empty());
    drop(_poller);

    let server = Server::start(World::running(b"{".to_vec()));
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.ai == AiState::Down
    });
    assert_eq!(view.decoded_total, None);
    let lines = log.lines().join("\n");
    assert!(lines.contains("malformed json"), "{lines}");
    assert!(!lines.contains("{"), "{lines}");
    drop(_poller);

    let mut world = World::running(Vec::new());
    world.running_status = 302;
    world.location = Some("/secret".to_owned());
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.ai == AiState::Down
    });
    assert_eq!(view.decoded_total, None);
    thread::sleep(Duration::from_millis(200));
    let hits = server.hits();
    assert!(
        hits.iter().all(|path| !path.contains("/secret")),
        "{hits:?}"
    );
}

#[test]
fn ready_metrics_ignore_extra_lines_and_track_resets() {
    let mut world = World::running(fixture("running-ready.json"));
    world
        .metrics
        .insert("qwen3.6-35b-a3b".to_owned(), fixture("metrics-sample.txt"));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, detail) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.ai == AiState::Loaded && view.decoded_total == Some(0)
    });
    assert_eq!(view.prompt_total, Some(0), "first read is the baseline");
    assert_eq!(view.models[0].name, "Qwen 35B");
    assert_eq!(view.models[0].state, "ready");
    assert_eq!(detail.prompt_tps, Some(0.0));
    let hits = server.hits();
    assert!(
        hits.iter()
            .any(|path| path == "/upstream/qwen3.6-35b-a3b/metrics"),
        "{hits:?}"
    );
    assert!(hits.iter().all(|path| !path.contains("Qwen")), "{hits:?}");
    assert!(hits.iter().all(|path| !path.contains("/slots")), "{hits:?}");

    server.update(|world| {
        world
            .metrics
            .insert("qwen3.6-35b-a3b".to_owned(), metrics_body(140, 0.0));
    });
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(40)
    });
    assert_eq!(view.ai, AiState::Loaded);

    server.update(|world| {
        world
            .metrics
            .insert("qwen3.6-35b-a3b".to_owned(), metrics_body(10, 0.0));
    });
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(50)
    });

    server.update(|world| {
        world.running = serde_json::to_vec(&serde_json::json!({
            "running": [
                {"model": "qwen3.6-35b-a3b", "name": "raw", "state": "ready"},
                {"model": "other-model", "name": "Other", "state": "ready"}
            ]
        }))
        .expect("json");
        world
            .metrics
            .insert("other-model".to_owned(), metrics_body(1000, 0.0));
        world
            .metrics
            .insert("qwen3.6-35b-a3b".to_owned(), metrics_body(10, 0.0));
    });
    wait_msg(&rx, Duration::from_secs(3), |view, _| {
        view.models.len() == 2 && view.decoded_total == Some(50)
    });
    server.update(|world| {
        world
            .metrics
            .insert("other-model".to_owned(), metrics_body(1005, 0.0));
    });
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(55)
    });
    assert_eq!(view.models[1].name, "Other");
    let hits = server.hits();
    assert!(
        hits.iter()
            .any(|path| path == "/upstream/other-model/metrics"),
        "{hits:?}"
    );
}

/// #11: llama.cpp's `llamacpp:prompt_tokens_total` feeds the snapshot's
/// prompt total, on the decode counter's rules; without it there is none.
#[test]
fn llamacpp_prompt_counter_is_the_prompt_total() {
    let mut world = World::running(fixture("running-ready.json"));
    world
        .metrics
        .insert("qwen3.6-35b-a3b".to_owned(), fixture("metrics-sample.txt"));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.prompt_total == Some(0)
    });
    server.update(|world| {
        world.metrics.insert(
            "qwen3.6-35b-a3b".to_owned(),
            b"llamacpp:prompt_tokens_total 103\nllamacpp:n_decode_total 110\n".to_vec(),
        );
    });
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.prompt_total == Some(100)
    });
    assert_eq!(view.decoded_total, Some(10));
    server.update(|world| {
        world.metrics.insert(
            "qwen3.6-35b-a3b".to_owned(),
            b"llamacpp:n_decode_total 120\n".to_vec(),
        );
    });
    let (view, _) = wait_msg(&rx, Duration::from_secs(3), |view, _| {
        view.decoded_total == Some(20) && view.prompt_total.is_none()
    });
    assert_eq!(view.prompt_total, None, "unmeasured is absent, not zero");
}

#[test]
fn non_ready_model_gets_no_metrics_or_slots() {
    let server = Server::start(World::running(running_model(
        "qwen3.6-35b-a3b",
        "Qwen",
        "starting",
    )));
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.ai == AiState::Loaded
    });
    thread::sleep(Duration::from_millis(400));
    let hits = server.hits();
    assert!(
        hits.iter().all(|path| !path.contains("/upstream/")),
        "{hits:?}"
    );
    drop(_poller);

    let server = Server::start(World::running(
        serde_json::to_vec(&serde_json::json!({
            "running": [
                {"model": "qwen3.6-35b-a3b", "state": "ready"},
                {"model": "other-model", "state": "starting"}
            ]
        }))
        .expect("json"),
    ));
    server.update(|world| {
        world
            .metrics
            .insert("qwen3.6-35b-a3b".to_owned(), metrics_body(5, 1.0));
        world
            .metrics
            .insert("other-model".to_owned(), metrics_body(5, 1.0));
        world
            .slots
            .insert("qwen3.6-35b-a3b".to_owned(), fixture("slots-sample.json"));
        world
            .slots
            .insert("other-model".to_owned(), fixture("slots-sample.json"));
    });
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, _rx) = spawn(&config, &log);
    // #70: a model starting is a swap in progress, so nothing upstream is
    // read at all, not even for the ready model.
    thread::sleep(Duration::from_millis(600));
    let hits = server.hits();
    assert!(hits.iter().filter(|path| *path == "/running").count() >= 2);
    assert!(
        hits.iter().all(|path| !path.contains("/upstream/")),
        "{hits:?}"
    );
    server.update(|world| {
        world.running = running_model("qwen3.6-35b-a3b", "Qwen", "ready");
        world.running_after = None;
    });
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(2) {
        if server
            .hits()
            .iter()
            .any(|path| path.contains("/upstream/qwen3.6-35b-a3b/slots"))
        {
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let hits = server.hits();
    assert!(
        hits.iter()
            .any(|path| path == "/upstream/qwen3.6-35b-a3b/metrics"),
        "{hits:?}"
    );
    assert!(
        hits.iter().all(|path| !path.contains("other-model")),
        "{hits:?}"
    );
}

#[test]
fn slots_tails_are_truncated_and_input_is_once_per_task() {
    let first_prompt = format!("{}FIRSTPROMPT", "A".repeat(400));
    let second_prompt = format!("{}SECONDPROMPT", "B".repeat(400));
    let first = slot_body(7, &first_prompt, "gen-one\u{1b}[2J", 3, 2);
    let mut world = World::running(running_model("qwen3.6-35b-a3b", "Qwen", "ready"));
    world
        .metrics
        .insert("qwen3.6-35b-a3b".to_owned(), metrics_body(20, 1.0));
    world.slots.insert("qwen3.6-35b-a3b".to_owned(), first);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (_view, detail) = wait_msg(&rx, Duration::from_secs(2), |_view, detail| {
        detail
            .slots
            .iter()
            .any(|slot| cells(&slot.output).contains("gen-one"))
    });
    let slot = &detail.slots[0];
    let input = cells(&slot.input);
    let output = cells(&slot.output);
    assert!(input.contains("FIRSTPROMPT"), "{input}");
    assert!(!input.contains(&"A".repeat(300)), "{input}");
    assert!(!output.chars().any(|ch| ch == '\u{1b}'), "{output}");
    assert!(output.contains("gen-one"), "{output}");
    assert_eq!(slot.n_decoded, 2);

    server.update(|world| {
        world.slots.insert(
            "qwen3.6-35b-a3b".to_owned(),
            slot_body(7, &second_prompt, "gen-two", 9, 6),
        );
    });
    let (_view, detail) = wait_msg(&rx, Duration::from_secs(3), |_view, detail| {
        detail
            .slots
            .iter()
            .any(|slot| cells(&slot.output).contains("gen-two"))
    });
    let slot = &detail.slots[0];
    let input = cells(&slot.input);
    let output = cells(&slot.output);
    assert!(input.contains("FIRSTPROMPT"), "{input}");
    assert!(!input.contains("SECONDPROMPT"), "{input}");
    assert!(output.contains("gen-two"), "{output}");
    assert_eq!(slot.n_prompt_tokens_processed, 9);
    assert!(detail.prompt_tps.is_some());
}

#[test]
fn text_off_keeps_slot_numbers_and_no_prompt_or_output_text() {
    let body = serde_json::to_vec(&serde_json::json!([{
        "id": 0,
        "id_task": 7,
        "is_processing": true,
        "n_ctx": 32_768,
        "n_prompt_tokens": 9_000,
        "n_prompt_tokens_processed": 9_000,
        "next_token": [{"n_decoded": 42}],
        "prompt": "SECRET-PROMPT-T45",
        "generated": "SECRET-OUTPUT-T45"
    }]))
    .expect("json");
    let mut world = World::running(running_model("qwen3.6-35b-a3b", "Qwen", "ready"));
    world
        .metrics
        .insert("qwen3.6-35b-a3b".to_owned(), metrics_body(20, 1.0));
    world.slots.insert("qwen3.6-35b-a3b".to_owned(), body);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch_with(
        server.port,
        12,
        4_194_304,
        0.15,
        "[tty]\nshow_text = false\n",
    );
    assert!(!config.tty.show_text);
    let (_poller, rx) = spawn(&config, &log);
    let (_view, detail) = wait_msg(&rx, Duration::from_secs(2), |_view, detail| {
        detail.slots.iter().any(|slot| slot.n_decoded == 42)
    });
    let slot = &detail.slots[0];
    assert!(slot.is_processing);
    assert_eq!(slot.n_ctx, Some(32_768));
    // #78: `n_prompt_tokens` holds the output too.
    assert_eq!(slot.ctx_prompt, Some(9_000 - 42));
    assert!(slot.input.is_empty(), "{:?}", slot.input);
    assert!(slot.output.is_empty(), "{:?}", slot.output);
    let kept = format!("{detail:?}{:?}", log.lines());
    assert!(!kept.contains("SECRET"), "{kept}");
    assert!(
        server.hits().iter().any(|hit| hit.ends_with("/slots")),
        "busy state and ctx fill still come from /slots"
    );
}

#[test]
fn slots_oversize_is_not_parsed_and_keeps_previous_tails() {
    let cap = 4_096usize;
    let mut world = World::running(running_model("qwen3.6-35b-a3b", "Qwen", "ready"));
    world
        .metrics
        .insert("qwen3.6-35b-a3b".to_owned(), metrics_body(80, 1.0));
    world.slots.insert(
        "qwen3.6-35b-a3b".to_owned(),
        slot_body(3, "KEEP-TAIL", "keep-out", 6, 2),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, cap as u64, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(2), |_view, detail| {
        detail
            .slots
            .iter()
            .any(|slot| slot.n_decoded == 2 && cells(&slot.input).contains("KEEP-TAIL"))
    });
    server.update(|world| {
        world
            .slots
            .insert("qwen3.6-35b-a3b".to_owned(), oversize_slots(cap));
    });
    thread::sleep(Duration::from_millis(800));
    let (view, detail) = wait_msg(&rx, Duration::from_secs(2), |_view, detail| {
        detail.slots.iter().any(|slot| slot.n_decoded == 2)
    });
    let slot = &detail.slots[0];
    assert_eq!(
        slot.n_decoded, 2,
        "oversize body must not replace slot numbers"
    );
    assert_eq!(slot.n_prompt_tokens_processed, 6);
    let input = cells(&slot.input);
    let output = cells(&slot.output);
    assert!(input.contains("KEEP-TAIL"), "{input}");
    assert!(output.contains("keep-out"), "{output}");
    assert!(!input.contains(MARKER), "{input}");
    assert!(!output.contains("NEW-TAIL"), "{output}");
    assert_eq!(view.decoded_total, Some(0));
    assert_eq!(
        lines_count(&log, "body exceeds cap"),
        1,
        "{:?}",
        log.lines()
    );
    assert!(log.lines().iter().all(|line| !line.contains(MARKER)));
}

#[test]
fn names_never_empty_and_config_can_narrow() {
    let server = Server::start(World::running(
        serde_json::to_vec(&serde_json::json!({
            "running": [
                {"model": "三五", "name": "三五", "state": "ready"},
                {"model": "qwen-id", "name": "三五", "state": "starting"}
            ]
        }))
        .expect("json"),
    ));
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.models.len() == 2
    });
    assert_eq!(view.models[0].name, "model");
    assert_eq!(view.models[1].name, "qwen-id");
    drop(_poller);

    let server = Server::start(World::running(running_model(
        "abcdefghijklmnopqrstuvwxyz",
        "",
        "ready",
    )));
    let log = MemLog::new();
    let config = watch(server.port, 6, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        !view.models.is_empty()
    });
    assert_eq!(view.models[0].name, format!("abcde{}", '\u{2026}'));
    assert!(!view.models[0].name.is_empty());
}

#[test]
fn a_stalled_consumer_keeps_only_the_newest_publish() {
    let rx = poller::SampleRx::new();
    for n in 0..8u64 {
        rx.put((
            LlamaView {
                ai: AiState::Loaded,
                models: Vec::new(),
                decoded_total: Some(n),
                prompt_total: None,
            },
            LlamaDetail {
                slots: Vec::new(),
                activity: Vec::new(),
                gen_tps: None,
                prompt_tps: None,
                latencies: poller::PollLatencies::default(),
                prompt_cache: Vec::new(),
                capture: None,
                setup: Vec::new(),
                engine_live: Vec::new(),
                suspected_loads: Vec::new(),
                series: Vec::new(),
                requests: Vec::new(),
                jobs: Vec::new(),
                claims: Vec::new(),
                gpu: None,
                displaced: Vec::new(),
            },
        ));
    }
    let (view, _) = rx.take().expect("one sample is held");
    assert_eq!(view.decoded_total, Some(7));
    assert!(
        rx.take().is_none(),
        "the slot does not queue older publishes"
    );
}

#[test]
fn hung_endpoint_does_not_block_try_recv() {
    let mut world = World::running(br#"{"running":[]}"#.to_vec());
    world.hang = true;
    let server = Server::start(world);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let path = dir.join(format!("watch-hang-{}.toml", server.port));
    let text = format!(
        r#"
[llama]
url = "http://127.0.0.1:{port}"
running_interval_s = 2.0
running_timeout_s = 0.8
metrics_interval_s = 0.4
metrics_timeout_s = 0.2
slots_interval_s = 1.0
slots_timeout_s = 0.4
activity_interval_s = 0.4
activity_timeout_s = 0.2
"#,
        port = server.port
    );
    std::fs::write(&path, text).expect("write");
    let config = Config::load_validated(&path, 8).expect("config");
    let log = MemLog::new();
    let (_poller, rx) = spawn(&config, &log);
    let start = Instant::now();
    while server.hits().is_empty() {
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "server was not hit"
        );
        thread::sleep(Duration::from_millis(10));
    }
    let started = Instant::now();
    for _ in 0..50 {
        match rx.try_recv() {
            Ok(_) | Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => {}
        }
    }
    assert!(
        started.elapsed() < Duration::from_millis(200),
        "try_recv blocked for {:?}",
        started.elapsed()
    );
}

/// #70 (a): a model `ready` in one `/running` read but gone from the next
/// gets no upstream request at all: the gate reads `/running` itself just
/// before each upstream GET, and a model it does not list as `ready` is
/// dropped, not retried.
#[test]
fn a_model_ready_in_one_read_and_gone_in_the_next_gets_no_upstream_request() {
    let first = serde_json::to_vec(&serde_json::json!({
        "running": [{"model": "alpha", "state": "ready"}]
    }))
    .expect("json");
    let second = serde_json::to_vec(&serde_json::json!({
        "running": [{"model": "beta", "state": "ready"}]
    }))
    .expect("json");
    let mut world = World::running(first);
    world.running_after = Some(second);
    for id in ["alpha", "beta"] {
        world.metrics.insert(id.to_owned(), metrics_body(4, 1.0));
        world
            .slots
            .insert(id.to_owned(), fixture("slots-sample.json"));
    }
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, _rx) = spawn(&config, &log);
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(2) {
        if server
            .hits()
            .iter()
            .any(|path| path == "/upstream/beta/slots")
        {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    thread::sleep(Duration::from_millis(300));
    let hits = server.hits();
    assert!(
        hits.iter().any(|path| path == "/upstream/beta/metrics"),
        "{hits:?}"
    );
    assert!(
        hits.iter().all(|path| !path.contains("alpha")),
        "a model gone from the fresh read was requested: {hits:?}"
    );
    // The first read alone never led to an upstream request.
    let first_upstream = hits
        .iter()
        .position(|path| path.contains("/upstream/"))
        .expect("an upstream request");
    let running_before = hits[..first_upstream]
        .iter()
        .filter(|path| *path == "/running")
        .count();
    assert!(running_before >= 2, "{hits:?}");
}

#[test]
fn down_or_idle_clears_slot_tails() {
    let mut world = World::running(running_model("qwen3.6-35b-a3b", "Qwen", "ready"));
    world
        .metrics
        .insert("qwen3.6-35b-a3b".to_owned(), metrics_body(3, 1.0));
    world.slots.insert(
        "qwen3.6-35b-a3b".to_owned(),
        slot_body(1, "LIVE-TAIL", "gen", 1, 2),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(2), |_view, detail| {
        detail
            .slots
            .iter()
            .any(|slot| slot.is_processing && cells(&slot.input).contains("LIVE-TAIL"))
    });
    server.update(|world| {
        world.running = br#"{"running":[]}"#.to_vec();
        world.running_after = None;
    });
    let (view, detail) = wait_msg(&rx, Duration::from_secs(2), |view, detail| {
        view.ai == AiState::Idle && detail.slots.is_empty()
    });
    assert!(
        detail.slots.iter().all(|slot| !slot.is_processing),
        "{:?}",
        detail.slots
    );
    assert_eq!(view.decoded_total, Some(0));
}

fn lines_count(log: &MemLog, needle: &str) -> usize {
    log.lines()
        .iter()
        .filter(|line| line.contains(needle))
        .count()
}

fn slot_body(id_task: i64, prompt: &str, generated: &str, processed: u64, decoded: u64) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!([{
        "id": 0,
        "id_task": id_task,
        "is_processing": true,
        "n_prompt_tokens": 10,
        "n_prompt_tokens_processed": processed,
        "next_token": [{"n_decoded": decoded}],
        "prompt": prompt,
        "generated": generated
    }]))
    .expect("json")
}

fn oversize_slots(cap: usize) -> Vec<u8> {
    let prefix = format!(
        "[{{\"id\":0,\"id_task\":9,\"is_processing\":true,\"n_prompt_tokens\":99,\"n_prompt_tokens_processed\":50,\"next_token\":[{{\"n_decoded\":99}}],\"prompt\":\"{MARKER}"
    );
    let suffix = "\",\"generated\":\"NEW-TAIL\"}]";
    let target = cap + 1024;
    let fill = target - prefix.len() - suffix.len();
    let mut body = prefix.into_bytes();
    body.extend(std::iter::repeat_n(b'Q', fill));
    body.extend_from_slice(suffix.as_bytes());
    assert_eq!(body.len(), cap + 1024);
    assert_ne!(body.len(), cap + 1);
    serde_json::from_slice::<serde_json::Value>(&body).expect("oversize body is json");
    body
}

/// An SGLang launch behind a podman wrapper, names made generic.
const SGLANG_CMD: &str = "podman run --rm --name flash img python3 -m sglang.launch_server --model-path /models/x --quantization exl3 --kv-cache-dtype fp8_e4m3 --context-length 204800 --max-running-requests 4";

fn running_cmd(id: &str, cmd: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "running": [{"model": id, "state": "ready", "cmd": cmd}]
    }))
    .expect("json")
}

fn sglang_metrics(generated: u64, running: u32, queued: u32, usage: f64) -> Vec<u8> {
    format!(
        "# TYPE sglang:generation_tokens_total counter\n\
sglang:generation_tokens_total{{model_name=\"flash\"}} {generated}.0\n\
sglang:prompt_tokens_total{{model_name=\"flash\"}} 900.0\n\
sglang:num_running_reqs{{model_name=\"flash\"}} {running}.0\n\
sglang:num_queue_reqs{{model_name=\"flash\"}} {queued}.0\n\
sglang:token_usage{{model_name=\"flash\"}} 0.99\n\
sglang:full_token_usage{{model_name=\"flash\"}} {usage}\n\
sglang:cache_hit_rate{{model_name=\"flash\"}} 0.5\n"
    )
    .into_bytes()
}

fn activity_page(rows: &[(i64, &str, i64)]) -> Vec<u8> {
    let data: Vec<serde_json::Value> = rows
        .iter()
        .map(|(id, model, output)| {
            serde_json::json!({
                "id": id,
                "timestamp": format!("2026-09-29T10:00:{id:02}Z"),
                "model": model,
                "tokens": {
                    "input_tokens": 10,
                    "output_tokens": output,
                    "cache_tokens": 0,
                    "prompt_per_second": -1,
                    "tokens_per_second": -1
                },
                "duration_ms": 1000,
                "resp_status_code": 200
            })
        })
        .collect();
    serde_json::to_vec(&serde_json::json!({ "data": data })).expect("json")
}

#[test]
fn sglang_reads_its_own_metrics_and_never_slots() {
    let mut world = World::running(running_cmd("flash", SGLANG_CMD));
    let mut body = sglang_metrics(1000, 1, 2, 0.37);
    // #79: the KV token gauges; the fill becomes used / capacity.
    body.extend_from_slice(
        b"sglang:kv_used_tokens{model_name=\"flash\"} 75776.0\n\
sglang:kv_evictable_tokens{model_name=\"flash\"} 31744.0\n\
sglang:max_total_num_tokens{model_name=\"flash\"} 204800.0\n",
    );
    world.metrics.insert("flash".to_owned(), body);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(0)
            && view.models[0]
                .backend
                .is_some_and(|info| info.running.is_some())
    });
    let info = view.models[0].backend.expect("backend");
    assert_eq!(info.kind, llama_core::backend::Backend::SgLang);
    assert_eq!(info.running, Some(1));
    assert_eq!(info.queued, Some(2));
    assert_eq!(
        info.kv_permille,
        Some(370),
        "75,776 / 204,800, not token_usage"
    );
    assert_eq!(info.hit_permille, Some(500));
    assert_eq!(info.max_running, Some(4));
    let kv = info.kv.expect("#79 kv");
    assert_eq!(
        (kv.used, kv.capacity, kv.cached, kv.sessions),
        (Some(75_776), Some(204_800), Some(31_744), Some(1))
    );
    assert_eq!((kv.unified, kv.approx), (None, false));
    let detail = view.models[0].detail.as_ref().expect("detail");
    assert_eq!(detail.ctx, Some(204_800));
    assert_eq!(detail.quant.as_deref(), Some("exl3"));

    server.update(|world| {
        world
            .metrics
            .insert("flash".to_owned(), sglang_metrics(1034, 0, 0, 0.1));
    });
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(34)
    });
    thread::sleep(Duration::from_millis(600));
    let hits = server.hits();
    assert!(
        hits.iter().any(|path| path == "/upstream/flash/metrics"),
        "{hits:?}"
    );
    assert!(hits.iter().all(|path| !path.contains("/slots")), "{hits:?}");
    let lines = log.lines().join("\n");
    assert!(!lines.contains("metrics:"), "{lines}");
}

/// SGLang's real `/metrics` is ~70 KiB of latency histograms; the gauges
/// sit after them. A document past llama.cpp's 64 KiB cap still counts.
#[test]
fn sglang_reads_a_large_metrics_document() {
    let mut body = String::new();
    for i in 0..4000 {
        body.push_str(&format!(
            "sglang:e2e_request_latency_seconds_bucket{{model_name=\"flash\",le=\"{i}.0\"}} {i}.0\n"
        ));
    }
    assert!(body.len() > 128 * 1024, "{}", body.len());
    let mut bytes = body.into_bytes();
    bytes.extend(sglang_metrics(1000, 1, 0, 0.25));
    let mut world = World::running(running_cmd("flash", SGLANG_CMD));
    world.metrics.insert("flash".to_owned(), bytes);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.models[0]
            .backend
            .is_some_and(|info| info.running.is_some())
    });
    let info = view.models[0].backend.expect("backend");
    assert_eq!(info.running, Some(1));
    assert_eq!(
        info.kv_permille,
        Some(250),
        "full_token_usage without the token gauges"
    );
    assert_eq!(info.kv, None);
    let lines = log.lines().join("\n");
    assert!(!lines.contains("no /metrics"), "{lines}");
}

#[test]
fn sglang_without_metrics_counts_activity_once_and_logs_once() {
    let mut world = World::running(running_cmd("flash", SGLANG_CMD));
    world.metrics_status = 404;
    world.activity = activity_page(&[(7, "flash", 500), (6, "flash", 400)]);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    // The first activity read is only the baseline: history is not back-filled.
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(0)
    });
    assert_eq!(view.models[0].backend.expect("backend").running, None);
    server.update(|world| {
        world.activity = activity_page(&[
            (9, "flash", 25),
            (8, "other", 1000),
            (7, "flash", 500),
            (6, "flash", 400),
        ]);
    });
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(25)
    });
    // #11: the fallback's prompt tokens are the new rows' input_tokens.
    assert_eq!(view.prompt_total, Some(10));
    // The same rows read again add nothing.
    thread::sleep(Duration::from_millis(900));
    let (view, _) = wait_msg(&rx, Duration::from_secs(1), |_, _| true);
    assert_eq!(view.decoded_total, Some(25));
    assert_eq!(view.prompt_total, Some(10));
    let lines = log.lines();
    let notes: Vec<&String> = lines
        .iter()
        .filter(|line| line.contains("no /metrics"))
        .collect();
    assert_eq!(notes.len(), 1, "{lines:?}");
    assert!(
        notes[0].contains("flash: no /metrics from sglang (")
            && notes[0].contains("); using llama-swap activity"),
        "{lines:?}"
    );
    assert!(
        lines.iter().all(|line| !line.contains("metrics: ")),
        "no metrics failure flag: {lines:?}"
    );
    let hits = server.hits();
    assert!(hits.iter().all(|path| !path.contains("/slots")), "{hits:?}");

    // /metrics comes back: a new baseline, not a jump.
    server.update(|world| {
        world.metrics_status = 200;
        world
            .metrics
            .insert("flash".to_owned(), sglang_metrics(5000, 0, 0, 0.0));
    });
    thread::sleep(Duration::from_millis(600));
    server.update(|world| {
        world
            .metrics
            .insert("flash".to_owned(), sglang_metrics(5010, 0, 0, 0.0));
    });
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(35)
    });
}

#[test]
fn openai_server_gets_no_metrics_or_slots_and_counts_activity() {
    let mut world = World::running(running_cmd("tabby", "python3 main.py --port 5000"));
    world.activity = activity_page(&[(1, "tabby", 3)]);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(0)
    });
    assert_eq!(
        view.models[0].backend.expect("backend").kind,
        llama_core::backend::Backend::OpenAi
    );
    assert_eq!(view.models[0].detail, None);
    server.update(|world| {
        world.activity = activity_page(&[(2, "tabby", 40), (1, "tabby", 3)]);
    });
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(40)
    });
    // #31: one /metrics probe for this load (no known names), then one
    // tqLLM snapshot probe (#180; none: stays openai), logged once; never
    // /slots.
    let hits = server.hits();
    let upstream: Vec<&String> = hits
        .iter()
        .filter(|path| path.contains("/upstream/"))
        .collect();
    assert_eq!(
        upstream,
        [
            "/upstream/tabby/metrics",
            "/upstream/tabby/api/monitor/v1/snapshot"
        ],
        "{hits:?}"
    );
    let lines = log.lines();
    let notes: Vec<&String> = lines
        .iter()
        .filter(|line| line.contains("metric"))
        .collect();
    assert_eq!(notes.len(), 1, "{lines:?}");
    assert!(
        notes[0].contains(
            "tabby: launch command names no server; no known metric names; no tqLLM snapshot (http status); staying openai"
        ),
        "{lines:?}"
    );
}

// ---- #31: detection by metric prefix, vLLM engine numbers ----------------------

/// HyperQwen's llama-swap command: a podman wrapper whose image runs
/// `vllm serve` itself, so the command names no server (generic names).
const CONTAINER_CMD: &str = "podman run --rm --name hyperqwen --network llama --device nvidia.com/gpu=all ghcr.io/example/hyperqwen single";

/// The vLLM fixture with the spec-decode and decode counters moved on.
fn vllm_metrics(generated: u64, drafts: u64, draft_tokens: u64, accepted: u64) -> Vec<u8> {
    String::from_utf8(fixture("vllm-metrics.txt"))
        .expect("utf-8")
        .replace(
            "vllm:generation_tokens_total{engine=\"0\",model_name=\"qwen3.8-27b\"} 42000.0",
            &format!("vllm:generation_tokens_total{{engine=\"0\",model_name=\"qwen3.8-27b\"}} {generated}.0"),
        )
        .replace(
            "vllm:spec_decode_num_drafts_total{engine=\"0\",model_name=\"qwen3.8-27b\"} 20000.0",
            &format!("vllm:spec_decode_num_drafts_total{{engine=\"0\",model_name=\"qwen3.8-27b\"}} {drafts}.0"),
        )
        .replace(
            "vllm:spec_decode_num_draft_tokens_total{engine=\"0\",model_name=\"qwen3.8-27b\"} 60000.0",
            &format!("vllm:spec_decode_num_draft_tokens_total{{engine=\"0\",model_name=\"qwen3.8-27b\"}} {draft_tokens}.0"),
        )
        .replace(
            "vllm:spec_decode_num_accepted_tokens_total{engine=\"0\",model_name=\"qwen3.8-27b\"} 38000.0",
            &format!("vllm:spec_decode_num_accepted_tokens_total{{engine=\"0\",model_name=\"qwen3.8-27b\"}} {accepted}.0"),
        )
        .into_bytes()
}

#[test]
fn a_container_vllm_is_found_by_its_metrics_and_reads_engine_numbers() {
    let mut world = World::running(running_cmd("qwen3.8-27b-vllm", CONTAINER_CMD));
    world.metrics.insert(
        "qwen3.8-27b-vllm".to_owned(),
        vllm_metrics(42_000, 20_000, 60_000, 38_000),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, detail) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.models
            .first()
            .and_then(|model| model.backend)
            .is_some_and(|info| info.engine.spec_permille.is_some())
    });
    let info = view.models[0].backend.expect("backend");
    assert_eq!(info.kind, llama_core::backend::Backend::Vllm);
    assert_eq!(info.running, Some(1));
    assert_eq!(info.queued, Some(2));
    assert_eq!(info.kv_permille, Some(413));
    // The first read: the server's totals since start (38k of 60k drafted).
    assert_eq!(info.engine.spec_permille, Some(633));
    assert_eq!(info.engine.spec_len_centi, Some(290));
    assert_eq!(info.engine.sleeping, Some(false));
    assert_eq!(info.engine.preemptions, Some(0));
    assert_eq!(info.engine.ttft_us, Some(750_000));
    assert_eq!(info.engine.e2e_us, Some(25_000_000));
    assert_eq!(info.engine.itl_us, Some(24_000));
    let tuning = view.models[0]
        .detail
        .as_ref()
        .expect("detail from cache_config_info");
    assert_eq!(tuning.kv_k.as_deref(), Some("fp8_e4m3"));
    assert_eq!(tuning.kv_block, Some(16));
    assert_eq!(tuning.prefix_cache, Some(true));
    assert_eq!(tuning.ctx, None);
    // prompt_tokens_cached is the cached counter; it starts at its baseline.
    assert_eq!(
        detail
            .prompt_cache
            .iter()
            .find(|row| row.model.starts_with("qwen3.8"))
            .map(|row| (row.prompt, row.cached)),
        Some((0, Some(0)))
    );

    // 100 drafts of 3, 240 accepted: 80 % and 3.4 per step over the window.
    server.update(|world| {
        world.metrics.insert(
            "qwen3.8-27b-vllm".to_owned(),
            vllm_metrics(42_340, 20_100, 60_300, 38_240),
        );
    });
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(340)
    });
    let engine = view.models[0].backend.expect("backend").engine;
    assert_eq!(engine.spec_permille, Some(800));
    assert_eq!(engine.spec_len_centi, Some(340));
    assert_eq!(
        engine.spec_counts,
        Some(llama_core::backend::SpecCounts {
            drafts: Some(100),
            draft_tokens: 300,
            accepted: 240,
        })
    );
    thread::sleep(Duration::from_millis(500));
    let hits = server.hits();
    assert!(hits.iter().all(|path| !path.contains("/slots")), "{hits:?}");
    let lines = log.lines();
    let notes: Vec<&String> = lines
        .iter()
        .filter(|line| line.contains("names no server"))
        .collect();
    assert_eq!(notes.len(), 1, "{lines:?}");
    assert!(
        notes[0].contains("qwen3.8-27b-vllm: launch command names no server; /metrics says vllm"),
        "{lines:?}"
    );
    assert!(
        lines.iter().all(|line| !line.contains("no /metrics")),
        "{lines:?}"
    );
}

#[test]
fn a_wrapped_llama_server_is_found_by_its_metrics_and_gets_slots() {
    let mut world = World::running(running_cmd("wrapped", "/opt/bin/start-model.sh fast"));
    world
        .metrics
        .insert("wrapped".to_owned(), metrics_body(10, 1.0));
    world
        .slots
        .insert("wrapped".to_owned(), fixture("slots-sample.json"));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, detail| {
        view.models
            .first()
            .and_then(|model| model.backend)
            .is_some_and(|info| info.kind == llama_core::backend::Backend::LlamaCpp)
            && !detail.slots.is_empty()
    });
    assert_eq!(view.decoded_total, Some(0));
    let hits = server.hits();
    assert!(
        hits.iter().any(|path| path == "/upstream/wrapped/slots"),
        "{hits:?}"
    );
    assert!(
        log.lines()
            .iter()
            .any(|line| line
                .contains("wrapped: launch command names no server; /metrics says llamacpp")),
        "{:?}",
        log.lines()
    );
}

/// #67: a llama.cpp image started by digest (values invented). Its flags
/// follow the image, and are read once `/metrics` says llama.cpp, or at
/// once when `[llama.backends]` names it.
const LCPP_IMAGE_CMD: &str = "podman run --name lcpp-a --rm --network llama --device nvidia.com/gpu=all --security-opt label=disable --cap-drop ALL -v /srv/models:/models:ro ghcr.io/example/llama.cpp@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef --host 0.0.0.0 --port 5850 -fa on --metrics -m /models/invented/Invented-30B-A3B-IQ4_XS.gguf -ngl 999 -ncmoe 39 -c 262144 -ctk f16 -ctv f16 --reasoning-budget 26000";

fn lcpp_detail_is_read(view: &LlamaView) -> bool {
    view.models.first().is_some_and(|model| {
        model
            .backend
            .is_some_and(|info| info.kind == llama_core::backend::Backend::LlamaCpp)
            && model.detail.as_ref().is_some_and(|detail| {
                detail.ctx == Some(262_144)
                    && detail.ncmoe == Some(39)
                    && detail.kv_k.as_deref() == Some("f16")
                    && detail.kv_v.as_deref() == Some("f16")
                    && detail.quant.as_deref() == Some("IQ4_XS")
            })
    })
}

#[test]
fn a_llamacpp_image_gets_its_flags_once_its_metrics_say_llamacpp() {
    let mut world = World::running(running_cmd("lcpp", LCPP_IMAGE_CMD));
    world
        .metrics
        .insert("lcpp".to_owned(), metrics_body(10, 0.0));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, detail) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        lcpp_detail_is_read(view)
    });
    // The probe's answer is followed by a `/running` read at once (the
    // fresh gate's check read, #70), so the flags are read as llama.cpp's
    // in the same round. Every upstream read sits between two `/running`
    // reads: the gate's, and the check after it.
    let hits: Vec<String> = server
        .hits()
        .into_iter()
        .filter(|path| path != "/api/events" && path != "/api/gpu")
        .collect();
    assert_eq!(
        hits[..8],
        [
            "/running",
            "/running",
            "/upstream/lcpp/metrics",
            "/running",
            "/running",
            "/upstream/lcpp/metrics",
            "/running",
            "/api/metrics/activity",
        ],
        "{hits:?}"
    );
    let setup = &detail.setup[0];
    assert!(!setup.found.is_empty(), "{setup:?}");
    let kept = format!("{view:?}{detail:?}");
    for leak in ["/models", "/srv", "sha256", "5850", "lcpp-a"] {
        assert!(!kept.contains(leak), "{leak} leaked");
    }
}

#[test]
fn a_llamacpp_image_named_in_the_config_gets_its_flags_at_once() {
    let mut world = World::running(running_cmd("lcpp", LCPP_IMAGE_CMD));
    world
        .metrics
        .insert("lcpp".to_owned(), metrics_body(10, 0.0));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch_with(
        server.port,
        12,
        4_194_304,
        0.15,
        "[llama.backends]\n\"lcpp\" = \"llamacpp\"\n",
    );
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        !view.models.is_empty()
    });
    assert!(lcpp_detail_is_read(&view), "{:?}", view.models);
}

#[test]
fn a_wrapped_sglang_is_found_by_its_metrics() {
    let mut world = World::running(running_cmd("flash", "/opt/bin/start-flash.sh"));
    world
        .metrics
        .insert("flash".to_owned(), sglang_metrics(1000, 1, 2, 0.37));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.models
            .first()
            .and_then(|model| model.backend)
            .is_some_and(|info| info.running == Some(1))
    });
    assert_eq!(
        view.models[0].backend.expect("backend").kind,
        llama_core::backend::Backend::SgLang
    );
    thread::sleep(Duration::from_millis(400));
    let hits = server.hits();
    assert!(hits.iter().all(|path| !path.contains("/slots")), "{hits:?}");
}

#[test]
fn the_config_beats_detection_and_openai_there_means_no_probe() {
    let mut world = World::running(running_cmd("qwen3.8-27b-vllm", CONTAINER_CMD));
    world.metrics.insert(
        "qwen3.8-27b-vllm".to_owned(),
        vllm_metrics(42_000, 20_000, 60_000, 38_000),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch_with(
        server.port,
        12,
        4_194_304,
        0.15,
        "[llama.backends]\n\"qwen3.8-27b-vllm\" = \"openai\"\n",
    );
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(0)
    });
    assert_eq!(
        view.models[0].backend.expect("backend").kind,
        llama_core::backend::Backend::OpenAi
    );
    thread::sleep(Duration::from_millis(600));
    let hits = server.hits();
    assert!(
        hits.iter().all(|path| !path.contains("/upstream/")),
        "{hits:?}"
    );
    assert!(
        log.lines()
            .iter()
            .all(|line| !line.contains("names no server"))
    );
}

#[test]
fn each_load_is_probed_once() {
    let tabby = |state: &str| -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "running": [{"model": "tabby", "state": state, "cmd": "python3 main.py"}]
        }))
        .expect("json")
    };
    let server = Server::start(World::running(tabby("ready")));
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(0)
    });
    let probes = |server: &Server| {
        server
            .hits()
            .iter()
            .filter(|path| *path == "/upstream/tabby/metrics")
            .count()
    };
    thread::sleep(Duration::from_millis(800));
    assert_eq!(probes(&server), 1, "{:?}", server.hits());
    // Unload and load again: one more probe, and still one log line.
    server.update(|world| world.running = tabby("starting"));
    server.update(|world| world.running_after = Some(tabby("starting")));
    thread::sleep(Duration::from_millis(800));
    assert_eq!(
        probes(&server),
        1,
        "not while starting: {:?}",
        server.hits()
    );
    server.update(|world| world.running_after = Some(tabby("ready")));
    thread::sleep(Duration::from_millis(1000));
    assert_eq!(probes(&server), 2, "{:?}", server.hits());
    let notes = log
        .lines()
        .iter()
        .filter(|line| line.contains("names no server"))
        .count();
    assert_eq!(notes, 1);
}

#[test]
fn config_override_beats_the_launch_command() {
    let mut world = World::running(running_cmd("flash", "llama-server -m /m/x-Q4_K_M.gguf"));
    world
        .metrics
        .insert("flash".to_owned(), sglang_metrics(10, 1, 0, 0.2));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch_with(
        server.port,
        12,
        4_194_304,
        0.15,
        "[llama.backends]\n\"flash\" = \"sglang\"\n",
    );
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.models
            .first()
            .and_then(|model| model.backend)
            .is_some_and(|info| info.running == Some(1))
    });
    assert_eq!(
        view.models[0].backend.expect("backend").kind,
        llama_core::backend::Backend::SgLang
    );
    thread::sleep(Duration::from_millis(600));
    let hits = server.hits();
    assert!(hits.iter().all(|path| !path.contains("/slots")), "{hits:?}");
}

// ---- Strata: JSON /metrics, no slots -----------------------------------------

/// The real-shaped llama-swap Strata command (podman wrapper, generic names).
const STRATA_CMD: &str = "podman run --rm --name flash --device nvidia.com/gpu=all -v /models/strata:/data:ro localhost/strata:v0.1.27-sm86 python serve/server.py --engine strata --config /data/strata.json --port 8095";

/// `fixtures/llama/strata-metrics.json` with invented counters and state.
fn strata_metrics(output: u64, prompt: u64, reused: u64, state: &str, generated: u64) -> Vec<u8> {
    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/llama/strata-metrics.json"),
    )
    .expect("strata fixture");
    let mut doc: serde_json::Value = serde_json::from_str(&text).expect("fixture json");
    doc["totals"]["output_tokens"] = serde_json::json!(output);
    doc["totals"]["prompt_tokens"] = serde_json::json!(prompt);
    doc["totals"]["reused"] = serde_json::json!(reused);
    doc["live"]["state"] = serde_json::json!(state);
    doc["live"]["generated"] = if state == "idle" {
        serde_json::Value::Null
    } else {
        serde_json::json!(generated)
    };
    // #81: in parallel mode the busy slot carries the request too, and
    // `running` counts it.
    doc["live"]["running"] = serde_json::json!(u8::from(state != "idle"));
    doc["live"]["slots"][0] = if state == "idle" {
        serde_json::json!({"slot": 0, "state": "idle", "held_tokens": 1840})
    } else {
        serde_json::json!({"slot": 0, "state": "decoding", "prompt_tokens": 1800,
            "generated": generated, "elapsed_s": 3.2, "tok_s": 31.5})
    };
    serde_json::to_vec(&doc).expect("json")
}

#[test]
fn strata_reads_its_json_metrics_and_never_slots() {
    let mut world = World::running(running_cmd("flash", STRATA_CMD));
    world.metrics.insert(
        "flash".to_owned(),
        strata_metrics(1000, 8000, 6000, "generating", 20),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, detail) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(0)
            && view.models[0]
                .backend
                .is_some_and(|info| info.running.is_some())
    });
    let info = view.models[0].backend.expect("backend");
    assert_eq!(info.kind, llama_core::backend::Backend::Strata);
    assert_eq!(info.running, Some(1));
    assert_eq!(info.queued, Some(2));
    // #81: `live.parallel` 2, one request per batch slot.
    assert_eq!(info.max_running, Some(2));
    // #79: two batch slots holding 1,820 (1,800 + 20 generated so far)
    // and 5,120 tokens of 2 × 262,144.
    assert_eq!(info.kv_permille, Some(13));
    let kv = info.kv.expect("#79 kv");
    assert_eq!(
        (kv.used, kv.capacity, kv.sessions),
        (Some(6940), Some(524_288), Some(2))
    );
    assert_eq!(info.hit_permille, None);
    // ctx and KV come from `engine`, which the launch command lacks.
    let model_detail = view.models[0].detail.as_ref().expect("detail");
    assert_eq!(model_detail.ctx, Some(262_144));
    assert_eq!(model_detail.kv_k.as_deref(), Some("q8"));
    assert_eq!(model_detail.quant, None);
    assert_eq!(prompt_cache_of(&detail, "flash"), Some((0, Some(0))));

    // Live tokens count while the request runs; finishing moves them to
    // totals without a jump; the prompt and reused counters move at the end.
    server.update(|world| {
        world.metrics.insert(
            "flash".to_owned(),
            strata_metrics(1000, 8000, 6000, "generating", 50),
        );
    });
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(30)
    });
    server.update(|world| {
        world.metrics.insert(
            "flash".to_owned(),
            strata_metrics(1050, 9500, 7200, "idle", 0),
        );
    });
    let (view, detail) = wait_msg(&rx, Duration::from_secs(2), |view, detail| {
        view.models[0]
            .backend
            .is_some_and(|info| info.running == Some(0))
            && prompt_cache_of(detail, "flash") == Some((1500, Some(1200)))
    });
    assert_eq!(view.decoded_total, Some(30));
    assert_eq!(prompt_cache_of(&detail, "flash"), Some((1500, Some(1200))));
    // Strata restarted: its totals start again and count from zero.
    server.update(|world| {
        world
            .metrics
            .insert("flash".to_owned(), strata_metrics(7, 100, 0, "idle", 0));
    });
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(37)
    });

    thread::sleep(Duration::from_millis(600));
    let hits = server.hits();
    assert!(
        hits.iter().any(|path| path == "/upstream/flash/metrics"),
        "{hits:?}"
    );
    assert!(hits.iter().all(|path| !path.contains("/slots")), "{hits:?}");
    let lines = log.lines().join("\n");
    assert!(!lines.contains("metrics:"), "{lines}");
    assert!(!lines.contains("no /metrics"), "{lines}");
}

/// Strata started with `--api-key`: `/metrics` answers 401. llama-bored
/// keeps no secrets, so the model is counted from activity rows, logged once.
#[test]
fn strata_with_an_api_key_falls_back_to_activity_and_logs_once() {
    for status in [401, 403] {
        let mut world = World::running(running_cmd("flash", STRATA_CMD));
        world.metrics_status = status;
        world.activity = activity_page(&[(7, "flash", 500)]);
        let server = Server::start(world);
        let log = MemLog::new();
        let config = watch(server.port, 12, 4_194_304, 0.15);
        let (_poller, rx) = spawn(&config, &log);
        wait_msg(&rx, Duration::from_secs(2), |view, _| {
            view.decoded_total == Some(0)
        });
        server.update(|world| {
            world.activity = activity_page(&[(8, "flash", 25), (7, "flash", 500)]);
        });
        let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
            view.decoded_total == Some(25)
        });
        let info = view.models[0].backend.expect("backend");
        assert_eq!(info.kind, llama_core::backend::Backend::Strata);
        assert_eq!(info.running, None);
        thread::sleep(Duration::from_millis(600));
        let lines = log.lines();
        let notes: Vec<&String> = lines
            .iter()
            .filter(|line| line.contains("no /metrics"))
            .collect();
        assert_eq!(notes.len(), 1, "{lines:?}");
        assert!(
            notes[0].contains(
                "flash: no /metrics from strata (unauthorized, API key set); using llama-swap activity"
            ),
            "{lines:?}"
        );
        assert!(
            lines.iter().all(|line| !line.contains("metrics: ")),
            "no metrics failure flag: {lines:?}"
        );
        let hits = server.hits();
        assert!(hits.iter().all(|path| !path.contains("/slots")), "{hits:?}");
    }
}

/// Strata's `/metrics` carries a minute of hardware history. Past the
/// 1 MiB server cap it is dropped, not parsed, and the model falls back.
#[test]
fn strata_metrics_over_the_cap_fall_back() {
    let mut doc: serde_json::Value =
        serde_json::from_slice(&strata_metrics(1000, 10, 0, "idle", 0)).expect("json");
    let point = serde_json::json!([1_790_000_000.0, 97]);
    doc["history"]["gpu_util"] = serde_json::Value::Array(vec![point; 80_000]);
    let body = serde_json::to_vec(&doc).expect("json");
    assert!(body.len() > 1024 * 1024, "{}", body.len());
    let mut world = World::running(running_cmd("flash", STRATA_CMD));
    world.metrics.insert("flash".to_owned(), body);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(0)
    });
    assert_eq!(view.models[0].backend.expect("backend").running, None);
    let lines = log.lines().join("\n");
    assert!(
        lines.contains("flash: no /metrics from strata (oversized body)"),
        "{lines}"
    );
}

#[test]
fn config_override_can_name_strata() {
    let mut world = World::running(running_cmd("flash", "python3 app.py --port 8095"));
    world
        .metrics
        .insert("flash".to_owned(), strata_metrics(10, 10, 0, "reading", 0));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch_with(
        server.port,
        12,
        4_194_304,
        0.15,
        "[llama.backends]\n\"flash\" = \"strata\"\n",
    );
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.models
            .first()
            .and_then(|model| model.backend)
            .is_some_and(|info| info.running == Some(1))
    });
    assert_eq!(
        view.models[0].backend.expect("backend").kind,
        llama_core::backend::Backend::Strata
    );
    // No launch detail: the engine's ctx and KV still show.
    let detail = view.models[0].detail.as_ref().expect("detail from engine");
    assert_eq!(detail.ctx, Some(262_144));
    assert_eq!(detail.kv_k.as_deref(), Some("q8"));
    thread::sleep(Duration::from_millis(600));
    let hits = server.hits();
    assert!(hits.iter().all(|path| !path.contains("/slots")), "{hits:?}");
}

// ---- #10: per-model prompt and cached-prompt counters -----------------------

/// #54: Strata in a container started by image digest: the command names an
/// image digest and a config, nothing that says Strata.
const STRATA_CONTAINER_CMD: &str = "podman run --rm --name strata --network llama --device nvidia.com/gpu=all -v /models/strata:/data:ro 5e1f0c2d9a7b4e6f8c3d2a1b0e9f8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e --config /data/configs/flash-next.json --port 8793";

/// #54: the /metrics probe tells Strata by its JSON's shape, once per
/// load; the model then reads as Strata everywhere: its engine numbers,
/// its SETUP values and its live phase.
#[test]
fn a_container_strata_is_found_by_the_shape_of_its_metrics() {
    let running = serde_json::to_vec(&serde_json::json!({
        "running": [{
            "model": "flash-next",
            "name": "Flash Next Q4_K_M",
            "state": "ready",
            "cmd": STRATA_CONTAINER_CMD,
        }]
    }))
    .expect("json");
    let mut world = World::running(running);
    world.metrics.insert(
        "flash-next".to_owned(),
        strata_metrics(2500, 12_000, 9000, "generating", 40),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, detail) = wait_msg(&rx, Duration::from_secs(2), |view, detail| {
        view.models
            .first()
            .and_then(|model| model.backend)
            .is_some_and(|info| info.engine.spec_permille.is_some())
            && !detail.engine_live.is_empty()
    });
    let info = view.models[0].backend.expect("backend");
    assert_eq!(info.kind, llama_core::backend::Backend::Strata);
    assert_eq!(info.max_running, Some(2), "#81: one per batch slot");
    assert_eq!(info.running, Some(1));
    assert_eq!(info.queued, Some(2));
    // The first read: everything since Strata started.
    assert_eq!(info.engine.spec_permille, Some(700));
    assert_eq!(info.engine.spec_len_centi, None);
    assert_eq!(info.engine.prefill_tps_tenths, Some(9677));
    assert_eq!(info.engine.decode_tps_tenths, Some(309));
    assert_eq!(info.engine.expert_hit_permille, Some(874));
    assert_eq!(info.engine.pcie_share_permille, Some(92));
    let tuning = view.models[0].detail.as_ref().expect("detail from engine");
    assert_eq!(tuning.ctx, Some(262_144));
    assert_eq!(tuning.kv_k.as_deref(), Some("q8"));
    // SETUP gets the engine's settings and the quant from the name.
    let setup = &detail.setup[0];
    assert_eq!(
        setup.engine.get("expert_cache_mib").map(String::as_str),
        Some("14950")
    );
    assert_eq!(
        setup.engine.get("version").map(String::as_str),
        Some("0.1.41")
    );
    assert!(
        setup.found.iter().any(|found| found.value == "Q4_K_M"),
        "{setup:?}"
    );
    // The live phase, sanitised, for tty11 only.
    let live = &detail.engine_live[0];
    assert_eq!(live.model, view.models[0].name, "keyed like the snapshot");
    assert_eq!(
        live.live.phase.as_deref(),
        Some("drafting a reply: outline")
    );
    assert_eq!(live.live.generated, Some(40));

    // The next window: one more request.
    server.update(|world| {
        let mut doc: serde_json::Value =
            serde_json::from_slice(&strata_metrics(2800, 13_000, 9800, "idle", 0)).expect("json");
        doc["totals"]["requests"] = serde_json::json!(8);
        doc["totals"]["prompt_ms"] = serde_json::json!(3500.0);
        doc["totals"]["decode_ms"] = serde_json::json!(91_000.0);
        doc["totals"]["drafts_offered"] = serde_json::json!(2680);
        doc["totals"]["drafts_accepted"] = serde_json::json!(1890);
        world.metrics.insert(
            "flash-next".to_owned(),
            serde_json::to_vec(&doc).expect("json"),
        );
    });
    let (view, detail) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.models[0]
            .backend
            .is_some_and(|info| info.engine.spec_permille == Some(750))
    });
    let engine = view.models[0].backend.expect("backend").engine;
    assert_eq!(engine.prefill_tps_tenths, Some(5000));
    assert_eq!(engine.decode_tps_tenths, Some(300));
    assert_eq!(
        engine.spec_counts,
        Some(llama_core::backend::SpecCounts {
            drafts: None,
            draft_tokens: 280,
            accepted: 210,
        })
    );
    // Idle: the live report says so, and the line has nothing to show.
    assert_eq!(
        detail
            .engine_live
            .first()
            .and_then(|entry| entry.live.state.as_deref()),
        Some("idle")
    );

    thread::sleep(Duration::from_millis(600));
    let hits = server.hits();
    assert!(hits.iter().all(|path| !path.contains("/slots")), "{hits:?}");
    let lines = log.lines();
    let notes: Vec<&String> = lines
        .iter()
        .filter(|line| line.contains("names no server"))
        .collect();
    assert_eq!(notes.len(), 1, "probed once: {lines:?}");
    assert!(
        notes[0].contains("flash-next: launch command names no server; /metrics says strata"),
        "{lines:?}"
    );
    assert!(
        lines.iter().all(|line| !line.contains("no /metrics")),
        "{lines:?}"
    );
}

fn cache_row(id: i64, model: &str, input: i64, cache: i64) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "timestamp": format!("2026-09-29T11:00:{id:02}Z"),
        "model": model,
        "tokens": {"input_tokens": input, "output_tokens": 5, "cache_tokens": cache,
                   "prompt_per_second": -1, "tokens_per_second": -1},
        "duration_ms": 100,
        "resp_status_code": 200
    })
}

fn cache_page(rows: &[serde_json::Value]) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({ "data": rows })).expect("json")
}

fn sglang_cached_metrics(prompt: u64, cached: u64) -> Vec<u8> {
    format!(
        "sglang:generation_tokens_total{{model_name=\"flash\"}} 10.0\n\
sglang:prompt_tokens_total{{model_name=\"flash\"}} {prompt}.0\n\
sglang:cached_tokens_total{{cache_source=\"device\",model_name=\"flash\"}} {cached}.0\n\
sglang:num_running_reqs{{model_name=\"flash\"}} 0.0\n"
    )
    .into_bytes()
}

fn prompt_cache_of(detail: &poller::LlamaDetail, model: &str) -> Option<(u64, Option<u64>)> {
    detail
        .prompt_cache
        .iter()
        .find(|entry| entry.model == model)
        .map(|entry| (entry.prompt, entry.cached))
}

#[test]
fn prompt_cache_counters_come_from_activity_rows_and_sglang_metrics() {
    let running = serde_json::to_vec(&serde_json::json!({
        "running": [
            {"model": "qwen", "name": "Qwen", "state": "ready", "cmd": "llama-server -m x.gguf"},
            {"model": "flash", "state": "ready", "cmd": SGLANG_CMD}
        ]
    }))
    .expect("json");
    let mut world = World::running(running);
    world
        .metrics
        .insert("qwen".to_owned(), metrics_body(100, 0.0));
    world
        .metrics
        .insert("flash".to_owned(), sglang_cached_metrics(10_000, 9_000));
    // History before the watcher started is only the baseline.
    world.activity = cache_page(&[cache_row(1, "qwen", 5_000, 0)]);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    // A model with no finished request yet reads 0; llama.cpp's cached
    // counter starts with it.
    wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        prompt_cache_of(detail, "Qwen") == Some((0, Some(0)))
            && prompt_cache_of(detail, "flash") == Some((0, Some(0)))
    });
    server.update(|world| {
        world.activity = cache_page(&[
            // llama.cpp timings: 69 processed plus 553 reused.
            cache_row(4, "qwen", 69, 553),
            // An SGLang row: /metrics counts flash, so this is skipped.
            cache_row(3, "flash", 777, -1),
            cache_row(2, "qwen", 1_000, 0),
            cache_row(1, "qwen", 5_000, 0),
        ]);
        world
            .metrics
            .insert("flash".to_owned(), sglang_cached_metrics(10_500, 9_400));
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        prompt_cache_of(detail, "Qwen") == Some((1_622, Some(553)))
            && prompt_cache_of(detail, "flash") == Some((500, Some(400)))
    });
    // Read again: nothing is counted twice.
    thread::sleep(Duration::from_millis(900));
    let (_, again) = wait_msg(&rx, Duration::from_secs(1), |_, _| true);
    assert_eq!(prompt_cache_of(&again, "Qwen"), Some((1_622, Some(553))));
    assert_eq!(prompt_cache_of(&again, "flash"), Some((500, Some(400))));
    assert_eq!(detail.prompt_cache.len(), 2);
}

// ---- #9: reset reasons through the poller -----------------------------------

/// A slot decoding its first token after a `prompt`-token prefill: the
/// whole prompt is known (#78).
fn ctx_slot(id_task: i64, prompt: u64) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!([{
        "id": 0,
        "id_task": id_task,
        "is_processing": true,
        "n_ctx": 262_144,
        "n_prompt_tokens": prompt,
        "n_prompt_tokens_processed": prompt,
        "next_token": [{"n_decoded": 1}],
        "prompt": "INVENTED-SECRET-PROMPT",
        "generated": "INVENTED-SECRET-OUTPUT"
    }]))
    .expect("json")
}

/// A compaction seen on `/slots`, decided by the request's activity row,
/// with `show_text = false`: numbers only.
#[test]
fn a_slot_drop_is_classified_from_activity_with_text_off() {
    let id = "qwen3.6-35b-a3b";
    let mut world = World::running(running_model(id, "Qwen", "ready"));
    world.metrics.insert(id.to_owned(), metrics_body(20, 1.0));
    world.slots.insert(id.to_owned(), ctx_slot(1, 80_000));
    world.activity = cache_page(&[cache_row(1, id, 100, 0)]);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch_with(
        server.port,
        12,
        4_194_304,
        0.15,
        "[tty]\nshow_text = false\n",
    );
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        detail
            .slots
            .iter()
            .any(|slot| slot.ctx_used == Some(80_000))
    });
    server.update(|world| {
        world.slots.insert(id.to_owned(), ctx_slot(2, 20_000));
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        detail
            .slots
            .iter()
            .any(|slot| slot.ctx_used == Some(20_000))
    });
    assert_eq!(detail.slots[0].resets.total(), 0, "waits for its row");
    server.update(|world| {
        world.activity = cache_page(&[cache_row(2, id, 6_000, 14_000), cache_row(1, id, 100, 0)]);
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        detail.slots.iter().any(|slot| slot.resets.total() == 1)
    });
    let slot = &detail.slots[0];
    assert_eq!(slot.resets.compacted, 1);
    assert_eq!(
        slot.last_reset,
        Some(llama_watch::resets::ResetReason::Compacted)
    );
    let kept = format!("{detail:?}{:?}", log.lines());
    assert!(!kept.contains("SECRET"), "{kept}");
}

// ---- #5: IN/OUT from llama-swap captures ------------------------------------

/// Every path the watcher may GET from llama-swap (the S17 fence, #5;
/// tqLLM's snapshot and health, #180).
fn allowed_path(path: &str) -> bool {
    let upstream = path
        .strip_prefix("/upstream/")
        .and_then(|rest| rest.split_once('/'))
        .is_some_and(|(model, leaf)| {
            !model.is_empty()
                && !model.contains("..")
                && matches!(
                    leaf,
                    "metrics"
                        | "slots"
                        | "v1/loads?include=core"
                        | "api/monitor/v1/snapshot"
                        | "health"
                )
        });
    let capture = path
        .strip_prefix("/api/captures/")
        .is_some_and(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()));
    // #80, #128: llama-swap's event stream; #167: its `GET /api/gpu`, both
    // while llama-swap answers.
    path == "/running"
        || path == "/api/metrics/activity"
        || path == "/api/events"
        || path == "/api/gpu"
        || upstream
        || capture
}

fn capture_row(id: i64, model: &str, has_capture: bool) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "timestamp": format!("2026-09-29T12:00:{id:02}Z"),
        "model": model,
        "tokens": {"input_tokens": 12, "output_tokens": 5, "cache_tokens": -1,
                   "prompt_per_second": -1, "tokens_per_second": -1},
        "duration_ms": 100,
        "resp_status_code": 200,
        "has_capture": has_capture
    })
}

fn capture_body(user: &str, response: &str) -> Vec<u8> {
    let request = serde_json::json!({
        "model": "flash",
        "messages": [
            {"role": "system", "content": "Invented system prompt."},
            {"role": "user", "content": "An older invented question."},
            {"role": "assistant", "content": "An older invented answer."},
            {"role": "user", "content": user}
        ]
    });
    serde_json::to_vec(&serde_json::json!({
        "id": 1,
        "req_path": "/v1/chat/completions",
        "req_headers": {"X-Session-Id": "INVENTED-SESSION-HEADER"},
        "req_body": llama_watch::capture::base64_encode(request.to_string().as_bytes()),
        "resp_headers": {"Content-Type": "text/event-stream"},
        "resp_body": llama_watch::capture::base64_encode(response.as_bytes()),
    }))
    .expect("json")
}

fn cells_text(cells: &[Cell]) -> String {
    cells.iter().map(|cell| cell.ch).collect()
}

fn capture_hits(server: &Server) -> Vec<String> {
    server
        .hits()
        .into_iter()
        .filter(|path| path.starts_with("/api/captures/"))
        .collect()
}

#[test]
fn a_backend_without_slots_gets_in_and_out_from_its_newest_capture_once() {
    let mut world = World::running(running_cmd("flash", SGLANG_CMD));
    world
        .metrics
        .insert("flash".to_owned(), sglang_metrics(10, 0, 0, 0.1));
    world.activity = cache_page(&[capture_row(3, "flash", true), capture_row(2, "flash", true)]);
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Invented \"}}]}\n\n\
               data: {\"choices\":[{\"delta\":{\"content\":\"streamed reply\\u001b[2J.\"}}]}\n\n\
               data: [DONE]\n\n";
    world.captures.insert(
        "3".to_owned(),
        capture_body("Newest invented question?", sse),
    );
    world.captures.insert(
        "2".to_owned(),
        capture_body("Older invented question?", "{}"),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (_, detail) = wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        detail.capture.is_some()
    });
    let capture = detail.capture.expect("capture");
    assert_eq!(capture.model, "flash");
    assert_eq!(capture.id, 3);
    assert_eq!(cells_text(&capture.input), "Newest invented question?");
    // The sanitiser ran: no escape byte reaches a cell.
    assert_eq!(cells_text(&capture.output), "Invented streamed reply[2J.");
    // The same row read again is not fetched again.
    thread::sleep(Duration::from_millis(900));
    assert_eq!(capture_hits(&server), vec!["/api/captures/3".to_owned()]);

    // A newer row: one more GET, with a plain JSON response.
    server.update(|world| {
        world.activity =
            cache_page(&[capture_row(4, "flash", true), capture_row(3, "flash", true)]);
        world.captures.insert(
            "4".to_owned(),
            capture_body(
                "Third invented question?",
                r#"{"choices":[{"message":{"content":"A plain invented reply."}}]}"#,
            ),
        );
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        detail
            .capture
            .as_ref()
            .is_some_and(|capture| capture.id == 4)
    });
    let capture = detail.capture.clone().expect("capture");
    assert_eq!(cells_text(&capture.output), "A plain invented reply.");
    thread::sleep(Duration::from_millis(500));
    assert_eq!(
        capture_hits(&server),
        vec!["/api/captures/3".to_owned(), "/api/captures/4".to_owned()]
    );
    let hits = server.hits();
    assert!(hits.iter().all(|path| allowed_path(path)), "{hits:?}");
    let kept = format!("{detail:?}{:?}", log.lines());
    assert!(!kept.contains("INVENTED-SESSION"), "headers are never kept");
}

#[test]
fn captures_off_oversize_and_text_off_fetch_nothing_or_keep_nothing() {
    // No `has_capture`: llama-swap has captures off; nothing is fetched.
    let mut world = World::running(running_cmd("flash", SGLANG_CMD));
    world
        .metrics
        .insert("flash".to_owned(), sglang_metrics(10, 0, 0, 0.1));
    world.activity = cache_page(&[capture_row(5, "flash", false)]);
    world.captures.insert(
        "5".to_owned(),
        capture_body("INVENTED-SHOULD-NOT-SHOW", "{}"),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    thread::sleep(Duration::from_millis(900));
    let (_, detail) = wait_msg(&rx, Duration::from_secs(1), |_, _| true);
    assert_eq!(detail.capture, None);
    assert!(capture_hits(&server).is_empty());

    // Oversize: read up to the cap, not parsed, logged once; nothing shown.
    let big = format!(
        "{{\"req_body\":\"{}\",\"resp_body\":\"\"}}",
        "A".repeat(llama_watch::capture::CAPTURE_CAP)
    );
    server.update(|world| {
        world.activity = cache_page(&[capture_row(6, "flash", true)]);
        world.captures.insert("6".to_owned(), big.into_bytes());
    });
    thread::sleep(Duration::from_millis(1_200));
    let (_, detail) = wait_msg(&rx, Duration::from_secs(1), |_, _| true);
    assert_eq!(detail.capture, None);
    assert_eq!(capture_hits(&server), vec!["/api/captures/6".to_owned()]);
    let lines = log.lines();
    assert_eq!(
        lines
            .iter()
            .filter(|line| line.contains("captures:"))
            .count(),
        1,
        "{lines:?}"
    );

    // Text off: never a capture GET, even with a fresh captured row.
    let mut world = World::running(running_cmd("flash", SGLANG_CMD));
    world
        .metrics
        .insert("flash".to_owned(), sglang_metrics(10, 0, 0, 0.1));
    world.activity = cache_page(&[capture_row(7, "flash", true)]);
    world
        .captures
        .insert("7".to_owned(), capture_body("INVENTED-SECRET-IN", "{}"));
    let quiet = Server::start(world);
    let config = watch_with(
        quiet.port,
        12,
        4_194_304,
        0.15,
        "[tty]\nshow_text = false\n",
    );
    let (_poller2, rx2) = spawn(&config, &log);
    thread::sleep(Duration::from_millis(900));
    let (_, detail) = wait_msg(&rx2, Duration::from_secs(1), |_, _| true);
    assert_eq!(detail.capture, None);
    assert!(capture_hits(&quiet).is_empty(), "{:?}", quiet.hits());
}

/// A llama.cpp model has `/slots`: its rows never trigger a capture GET.
#[test]
fn a_llamacpp_model_never_fetches_captures() {
    let id = "qwen3.6-35b-a3b";
    let mut world = World::running(running_model(id, "Qwen", "ready"));
    world.metrics.insert(id.to_owned(), metrics_body(20, 0.0));
    world.activity = cache_page(&[capture_row(9, id, true)]);
    world
        .captures
        .insert("9".to_owned(), capture_body("INVENTED", "{}"));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, _rx) = spawn(&config, &log);
    thread::sleep(Duration::from_millis(900));
    assert!(capture_hits(&server).is_empty());
    let hits = server.hits();
    assert!(hits.iter().all(|path| allowed_path(path)), "{hits:?}");
}

/// #66: current llama-server's `/slots` without SLOTS_DEBUG, values
/// invented: numbers, no `prompt` or `generated`.
fn textless_slots(id_task: i64) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!([{
        "id": 0, "n_ctx": 262_144, "speculative": false, "is_processing": true,
        "id_task": id_task, "n_prompt_tokens": 46, "n_prompt_tokens_processed": 30,
        "n_prompt_tokens_cache": 10, "params": {"n_predict": -1},
        "next_token": [{"has_next_token": true, "has_new_line": false, "n_remain": -1, "n_decoded": 7}]
    }]))
    .expect("json")
}

const TEXTLESS_LOG: &str = "lcpp: /slots has no prompt text; start llama-server with LLAMA_SERVER_SLOTS_DEBUG=1 for live IN/OUT";

/// #66: a llama.cpp model whose `/slots` has no text gets IN and OUT from
/// its captures, says why once, and goes back to live text when `/slots`
/// has it again. `/slots` still gives the numbers.
#[test]
fn a_llamacpp_model_without_slot_text_uses_captures_until_text_appears() {
    let id = "lcpp";
    let mut world = World::running(running_model(id, "Lcpp", "ready"));
    world.metrics.insert(id.to_owned(), metrics_body(20, 1.0));
    world.slots.insert(id.to_owned(), textless_slots(3));
    world.activity = cache_page(&[capture_row(4, id, true), capture_row(3, id, true)]);
    world.captures.insert(
        "4".to_owned(),
        capture_body(
            "Newest invented question?",
            r#"{"choices":[{"message":{"content":"An invented reply."}}]}"#,
        ),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (_, detail) = wait_msg(&rx, Duration::from_secs(3), |_, detail| {
        detail.capture.is_some()
    });
    let capture = detail.capture.clone().expect("capture");
    assert!(capture.over_slots);
    assert_eq!(capture.model, "Lcpp");
    assert_eq!(capture.id, 4);
    assert_eq!(cells_text(&capture.input), "Newest invented question?");
    assert_eq!(cells_text(&capture.output), "An invented reply.");
    // The slot numbers still come from `/slots`.
    let slot = detail.slots.first().expect("slot");
    assert_eq!(
        (slot.n_ctx, slot.ctx_used, slot.n_decoded),
        (Some(262_144), Some(46), 7)
    );
    // Several more `/slots` reads: logged once, fetched once.
    thread::sleep(Duration::from_millis(1_200));
    assert_eq!(lines_count(&log, TEXTLESS_LOG), 1, "{:?}", log.lines());
    assert_eq!(capture_hits(&server), vec!["/api/captures/4".to_owned()]);

    // The server comes back with SLOTS_DEBUG: live text, no capture.
    server.update(|world| {
        world.slots.insert(
            id.to_owned(),
            slot_body(5, "Live invented prompt", "Live invented output", 3, 2),
        );
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(3), |_, detail| {
        detail.capture.is_none()
            && detail
                .slots
                .first()
                .is_some_and(|slot| cells(&slot.output) == "Live invented output")
    });
    assert!(detail.capture.is_none());
    // A newer row is no longer fetched for it.
    server.update(|world| {
        world.activity = cache_page(&[capture_row(6, id, true), capture_row(4, id, true)]);
        world
            .captures
            .insert("6".to_owned(), capture_body("INVENTED-NOT-FETCHED", "{}"));
    });
    thread::sleep(Duration::from_millis(900));
    assert_eq!(capture_hits(&server), vec!["/api/captures/4".to_owned()]);
    assert_eq!(lines_count(&log, TEXTLESS_LOG), 1);
    let hits = server.hits();
    assert!(hits.iter().all(|path| allowed_path(path)), "{hits:?}");
}

// ---- #35: engine-measured speeds on RECENT rows ---------------------------------

/// The vLLM fixture with `extra` requests finished past its 42, each of
/// `prefill_s` / `prompt` computed tokens and `decode_s` / `generated`.
fn vllm_finished(
    extra: u32,
    prefill_s: f64,
    prompt: u64,
    decode_s: f64,
    generated: u64,
) -> Vec<u8> {
    let count = 42 + u64::from(extra);
    let mut text = String::from_utf8(fixture("vllm-metrics.txt")).expect("utf-8");
    for (name, sum) in [
        ("vllm:request_prefill_time_seconds", 50.0 + prefill_s),
        (
            "vllm:request_prefill_kv_computed_tokens",
            125_000.0 + prompt as f64,
        ),
        ("vllm:request_decode_time_seconds", 1000.0 + decode_s),
        (
            "vllm:request_generation_tokens",
            42_000.0 + generated as f64,
        ),
    ] {
        let labels = "{engine=\"0\",model_name=\"qwen3.8-27b\"}";
        let old_sum = text
            .lines()
            .find(|line| line.starts_with(&format!("{name}_sum{labels}")))
            .expect("sum line")
            .to_owned();
        text = text
            .replace(&old_sum, &format!("{name}_sum{labels} {sum:.1}"))
            .replace(
                &format!("{name}_count{labels} 42.0"),
                &format!("{name}_count{labels} {count}.0"),
            );
    }
    text.into_bytes()
}

/// `(id, model, prompt tok/s, gen tok/s)`; `-1` is llama-swap's unknown.
fn speed_page(rows: &[(i64, &str, f64, f64)]) -> Vec<u8> {
    let data: Vec<serde_json::Value> = rows
        .iter()
        .map(|(id, model, prompt, gen_tps)| {
            serde_json::json!({
                "id": id,
                "timestamp": format!("2026-10-03T10:00:{id:02}Z"),
                "model": model,
                "tokens": {
                    "input_tokens": 9000,
                    "output_tokens": 401,
                    "cache_tokens": 1000,
                    "prompt_per_second": prompt,
                    "tokens_per_second": gen_tps
                },
                "duration_ms": 10_000,
                "resp_status_code": 200
            })
        })
        .collect();
    serde_json::to_vec(&serde_json::json!({ "data": data })).expect("json")
}

fn with_prompt_total(body: Vec<u8>, total: u64) -> Vec<u8> {
    let mut text = String::from_utf8(body).expect("utf8");
    let old = text
        .lines()
        .find(|line| line.starts_with("vllm:prompt_tokens_total{"))
        .expect("prompt counter")
        .to_owned();
    text = text.replace(
        &old,
        &format!("vllm:prompt_tokens_total{{engine=\"0\",model_name=\"qwen3.8-27b\"}} {total}.0"),
    );
    text.into_bytes()
}

/// #188: 20,000 prompt tokens in 10 s of prefill is 2,000 on the big
/// number, and it is still 2,000 a second later. The wall-clock counter
/// would read about 20,000 on the sample that lands the tokens and 0 once
/// that second has passed.
#[test]
fn vllm_prefill_speed_holds_the_big_prompt_number() {
    let mut world = World::running(running_cmd("qwen3.8-27b-vllm", CONTAINER_CMD));
    world.metrics.insert(
        "qwen3.8-27b-vllm".to_owned(),
        with_prompt_total(vllm_finished(0, 0.0, 0, 0.0, 0), 500_000),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let near = |got: Option<f64>, want: f64| got.is_some_and(|got| (got - want).abs() < 1.0);
    wait_msg(&rx, Duration::from_secs(3), |_, detail| {
        near(detail.prompt_tps, 2_500.0)
    });
    server.update(|world| {
        world.metrics.insert(
            "qwen3.8-27b-vllm".to_owned(),
            with_prompt_total(vllm_finished(1, 10.0, 20_000, 0.0, 0), 520_000),
        );
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(3), |_, detail| {
        near(detail.prompt_tps, 2_000.0)
    });
    assert!(
        near(detail.prompt_tps, 2_000.0),
        "not the 20,000 wall-clock spike: {:?}",
        detail.prompt_tps
    );
    thread::sleep(Duration::from_millis(1_200));
    let (_, detail) = wait_msg(&rx, Duration::from_secs(2), |_, _| true);
    assert!(
        near(detail.prompt_tps, 2_000.0),
        "a second later, not 0 and not 20,000: {:?}",
        detail.prompt_tps
    );
}

/// #188: llama.cpp's live slot rate stays the big number even when the
/// prompt counter jumps by far more in the same second.
#[test]
fn llamacpp_live_slot_rate_stays_the_big_prompt_number() {
    let mut world = World::running(running_model("qwen3.6-35b-a3b", "Qwen", "ready"));
    world.metrics.insert(
        "qwen3.6-35b-a3b".to_owned(),
        b"llamacpp:prompt_tokens_total 100\nllamacpp:n_decode_total 20\nllamacpp:requests_processing 1\n".to_vec(),
    );
    world.slots.insert(
        "qwen3.6-35b-a3b".to_owned(),
        slot_body(7, "prompt", "", 100, 0),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(3), |_, detail| {
        !detail.slots.is_empty()
    });
    server.update(|world| {
        world.metrics.insert(
            "qwen3.6-35b-a3b".to_owned(),
            b"llamacpp:prompt_tokens_total 100100\nllamacpp:n_decode_total 20\nllamacpp:requests_processing 1\n".to_vec(),
        );
        world.slots.insert(
            "qwen3.6-35b-a3b".to_owned(),
            slot_body(7, "prompt", "", 2_100, 0),
        );
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(3), |_, detail| {
        detail.prompt_tps.is_some_and(|rate| rate > 0.0)
    });
    let rate = detail.prompt_tps.expect("slot rate");
    assert!(
        rate < 20_000.0,
        "the live slot rate, not the 100,000-token wall-clock spike: {rate}"
    );
}

#[test]
fn vllm_rows_get_the_engine_speeds_of_the_window_they_finished_in() {
    let running = serde_json::to_vec(&serde_json::json!({
        "running": [
            {"model": "qwen3.8-27b-vllm", "state": "ready", "cmd": CONTAINER_CMD},
            {"model": "fast", "state": "ready", "cmd": "llama-server --port 1"}
        ]
    }))
    .expect("json");
    let mut world = World::running(running);
    world.metrics.insert(
        "qwen3.8-27b-vllm".to_owned(),
        vllm_finished(0, 0.0, 0, 0.0, 0),
    );
    world
        .metrics
        .insert("fast".to_owned(), metrics_body(10, 0.0));
    world.activity = speed_page(&[(1, "qwen3.8-27b-vllm", -1.0, -1.0)]);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let vllm_engine = |view: &LlamaView| {
        view.models
            .iter()
            .find_map(|model| {
                model
                    .backend
                    .filter(|b| b.kind == llama_core::backend::Backend::Vllm)
            })
            .map(|info| info.engine)
    };
    // The baseline: the exported window is the server's lifetime so far.
    let (view, _) = wait_msg(&rx, Duration::from_secs(3), |view, detail| {
        detail.activity.iter().any(|row| row.id == 1)
            && vllm_engine(view).is_some_and(|e| e.prefill_tps_tenths.is_some())
    });
    let engine = vllm_engine(&view).expect("vllm");
    assert_eq!(engine.prefill_tps_tenths, Some(25_000), "125,000 in 50 s");
    assert_eq!(engine.decode_tps_tenths, Some(420), "41,958 in 1,000 s");
    thread::sleep(Duration::from_millis(500));

    // One request finishes (8,000 computed tokens in 2 s; 401 generated,
    // 400 decoded in 8 s) and its row appears, with a llama.cpp row that
    // has llama-swap's own timings.
    server.update(|world| {
        world.metrics.insert(
            "qwen3.8-27b-vllm".to_owned(),
            vllm_finished(1, 2.0, 8_000, 8.0, 401),
        );
        world.activity = speed_page(&[
            (3, "fast", 1193.6, 50.5),
            (2, "qwen3.8-27b-vllm", -1.0, -1.0),
            (1, "qwen3.8-27b-vllm", -1.0, -1.0),
        ]);
    });
    let (view, detail) = wait_msg(&rx, Duration::from_secs(3), |_, detail| {
        detail
            .activity
            .iter()
            .any(|row| row.id == 2 && row.engine_prompt_tps.is_some())
    });
    let row = |id: i64| {
        detail
            .activity
            .iter()
            .find(|row| row.id == id)
            .expect("row")
    };
    let measured = row(2);
    assert!((measured.engine_prompt_tps.unwrap() - 4_000.0).abs() < 1e-6);
    assert!((measured.engine_gen_tps.unwrap() - 50.0).abs() < 1e-6);
    assert_eq!((measured.prompt_tps, measured.gen_tps), (None, None));
    let llama = row(3);
    assert_eq!(llama.prompt_tps, Some(1193.6), "llama.cpp keeps its own");
    assert_eq!(llama.gen_tps, Some(50.5));
    assert_eq!(
        (llama.engine_prompt_tps, llama.engine_gen_tps),
        (None, None)
    );
    let history = row(1);
    assert_eq!(
        (history.engine_prompt_tps, history.engine_gen_tps),
        (None, None),
        "a row from before the watcher started has no window"
    );
    let engine = vllm_engine(&view).expect("vllm");
    assert_eq!(engine.prefill_tps_tenths, Some(40_000));
    assert_eq!(engine.decode_tps_tenths, Some(500));

    // Two requests finish together: both rows get the window's average.
    thread::sleep(Duration::from_millis(500));
    server.update(|world| {
        world.metrics.insert(
            "qwen3.8-27b-vllm".to_owned(),
            vllm_finished(3, 4.0, 12_000, 18.0, 901),
        );
        world.activity = speed_page(&[
            (5, "qwen3.8-27b-vllm", -1.0, -1.0),
            (4, "qwen3.8-27b-vllm", -1.0, -1.0),
            (3, "fast", 1193.6, 50.5),
            (2, "qwen3.8-27b-vllm", -1.0, -1.0),
        ]);
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(3), |_, detail| {
        detail
            .activity
            .iter()
            .filter(|row| row.engine_gen_tps.is_some())
            .count()
            == 3
    });
    for id in [4, 5] {
        let row = detail
            .activity
            .iter()
            .find(|row| row.id == id)
            .expect("row");
        // 4,000 tokens in 2 s; 500 generated, 498 decoded in 10 s.
        assert!((row.engine_prompt_tps.unwrap() - 2_000.0).abs() < 1e-6);
        assert!((row.engine_gen_tps.unwrap() - 49.8).abs() < 1e-6);
    }
    let kept = detail.activity.iter().find(|row| row.id == 2).expect("row");
    assert!((kept.engine_gen_tps.unwrap() - 50.0).abs() < 1e-6, "kept");

    // A row whose window saw no finished request stays without.
    thread::sleep(Duration::from_millis(500));
    server.update(|world| {
        world.activity = speed_page(&[
            (6, "qwen3.8-27b-vllm", -1.0, -1.0),
            (5, "qwen3.8-27b-vllm", -1.0, -1.0),
        ]);
    });
    wait_msg(&rx, Duration::from_secs(3), |_, detail| {
        detail.activity.iter().any(|row| row.id == 6)
    });
    thread::sleep(Duration::from_millis(800));
    let (_, detail) = wait_msg(&rx, Duration::from_secs(3), |_, _| true);
    let unmatched = detail.activity.iter().find(|row| row.id == 6).expect("row");
    assert_eq!(
        (unmatched.engine_prompt_tps, unmatched.engine_gen_tps),
        (None, None)
    );
    let hits = server.hits();
    assert!(
        hits.iter()
            .all(|path| !path.contains("/upstream/qwen3.8-27b-vllm/slots")),
        "{hits:?}"
    );
}

/// Cumulative SGLang counters. Both `is_streaming` series and both
/// `cache_source` series are present so a last-wins parse cannot match.
#[allow(clippy::too_many_arguments)]
fn sglang_window(
    requests: (u64, u64),
    prompt: u64,
    cached: (u64, u64),
    generated: u64,
    e2e: (f64, f64),
    ttft: (f64, f64),
    running: u32,
) -> Vec<u8> {
    format!(
        "sglang:num_requests_total{{model_name=\"flash\",is_streaming=\"true\"}} {true_n}.0\n\
sglang:num_requests_total{{model_name=\"flash\",is_streaming=\"false\"}} {false_n}.0\n\
sglang:prompt_tokens_total{{model_name=\"flash\"}} {prompt}.0\n\
sglang:generation_tokens_total{{model_name=\"flash\"}} {generated}.0\n\
sglang:cached_tokens_total{{cache_source=\"device\",model_name=\"flash\"}} {device}.0\n\
sglang:cached_tokens_total{{cache_source=\"host\",model_name=\"flash\"}} {host}.0\n\
sglang:e2e_request_latency_seconds_sum{{model_name=\"flash\"}} {e2e_sum}\n\
sglang:e2e_request_latency_seconds_count{{model_name=\"flash\"}} {e2e_count}\n\
sglang:time_to_first_token_seconds_sum{{model_name=\"flash\",is_streaming=\"true\"}} {ttft_sum}\n\
sglang:time_to_first_token_seconds_count{{model_name=\"flash\",is_streaming=\"true\"}} {ttft_count}\n\
sglang:num_running_reqs{{model_name=\"flash\"}} {running}.0\n",
        true_n = requests.0,
        false_n = requests.1,
        device = cached.0,
        host = cached.1,
        e2e_sum = e2e.0,
        e2e_count = e2e.1,
        ttft_sum = ttft.0,
        ttft_count = ttft.1,
    )
    .into_bytes()
}

/// `sec`, as llama-swap writes an activity timestamp.
fn sglang_stamp_at(sec: i64) -> String {
    format!(
        "{}Z",
        llama_watch::localtime::format_local(sec, llama_watch::localtime::Zone::utc())
    )
}

/// The unix second once the clock is 50 ms past it, and the second is at
/// least `earliest`. Row stamps are whole seconds. Posting in the last
/// fraction of a second lets the scrape that sees the finish land in the
/// next second and overlap the following row's window.
fn wait_row_second(earliest: i64) -> i64 {
    let phase_ms = 50u64;
    let slack_ms = 100u64;
    loop {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock");
        let sec = now.as_secs() as i64;
        let frac = now.subsec_millis() as u64;
        if sec >= earliest && (phase_ms..phase_ms + slack_ms).contains(&frac) {
            return sec;
        }
        let target_sec = if sec < earliest {
            earliest
        } else if frac < phase_ms {
            sec
        } else {
            sec + 1
        };
        let now_ms = now.as_secs() * 1_000 + frac;
        let target_ms = (target_sec as u64) * 1_000 + phase_ms;
        thread::sleep(Duration::from_millis(
            target_ms.saturating_sub(now_ms).max(1),
        ));
    }
}

/// One activity page. `cache`, `prompt` and `gen_rate` are llama-swap's
/// numbers; `-1` is its "not reported". `output` and `duration_ms` have to
/// match the window or the row is not that request. `stamps` is each row's
/// own end time, in the same order: llama-swap does not rewrite it when
/// the page is read again, and the ring treats a new timestamp as a new row.
fn sglang_activity(rows: &[(i64, i64, i64, i64, f64, f64, u64)], stamps: &[String]) -> Vec<u8> {
    let data: Vec<serde_json::Value> = rows
        .iter()
        .zip(stamps)
        .map(
            |((id, input, output, cache, prompt, gen_rate, duration_ms), stamp)| {
                serde_json::json!({
                    "id": id,
                    "timestamp": stamp,
                    "model": "flash",
                    "tokens": {
                        "input_tokens": input,
                        "output_tokens": output,
                        "cache_tokens": cache,
                        "prompt_per_second": prompt,
                        "tokens_per_second": gen_rate
                    },
                    "duration_ms": duration_ms,
                    "resp_status_code": 200
                })
            },
        )
        .collect();
    serde_json::to_vec(&serde_json::json!({ "data": data })).expect("json")
}

/// #133: one SGLang request's counters fill its row; a row llama-swap
/// already measured is left alone.
#[test]
fn sglang_one_request_fills_its_row_and_a_reported_row_is_left_alone() {
    let mut world = World::running(running_cmd("flash", SGLANG_CMD));
    world.metrics.insert(
        "flash".to_owned(),
        sglang_window((0, 0), 0, (0, 0), 0, (0.0, 0.0), (0.0, 0.0), 0),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(3), |view, _| {
        view.models.first().is_some_and(|model| {
            model.backend.is_some_and(|info| {
                info.kind == llama_core::backend::Backend::SgLang && info.running == Some(0)
            })
        })
    });
    // Idle scrapes, so the watcher's start is outside the first row's
    // isolation window. A row is filled only after a scrape stamped
    // more than six seconds past its end. Each request starts just after
    // a second boundary. The next row is eight seconds later, so its
    // window opens at the previous second plus three: a finish scrape
    // that slips into the following second stays a second clear of it.
    thread::sleep(Duration::from_secs(6));
    let sec1 = wait_row_second(0);

    // Prompt 1,000 (600 cached), TTFT 2 s, 401 generated, e2e 10 s.
    let stamp1 = sglang_stamp_at(sec1);
    server.update(|world| {
        world.metrics.insert(
            "flash".to_owned(),
            sglang_window((1, 0), 1_000, (400, 200), 401, (10.0, 1.0), (2.0, 1.0), 0),
        );
        world.activity = sglang_activity(
            &[(1, 1_000, 401, -1, -1.0, -1.0, 10_000)],
            std::slice::from_ref(&stamp1),
        );
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(12), |_, detail| {
        detail
            .activity
            .iter()
            .any(|row| row.id == 1 && row.cached_tokens.is_some())
    });
    let row = detail.activity.iter().find(|row| row.id == 1).expect("row");
    assert_eq!(row.cached_tokens, Some(600));
    assert!((row.engine_prompt_tps.expect("prompt") - 200.0).abs() < 1e-6);
    assert!((row.engine_gen_tps.expect("gen") - 50.0).abs() < 1e-6);
    assert_eq!((row.prompt_tps, row.gen_tps), (None, None));

    // A second request, only after the first row is decided and a second
    // clear of that finish. The row already has a cached count and a
    // prompt rate. The window is clean (prompt 500, cached 300, TTFT 2 s,
    // 100 generated, e2e 4 s). Those reported values stay. The generation
    // rate was not reported, so it is filled: (100 − 1) / (4 − 2) = 49.5.
    let sec2 = wait_row_second(sec1 + 8);
    let stamp2 = sglang_stamp_at(sec2);
    server.update(|world| {
        world.metrics.insert(
            "flash".to_owned(),
            sglang_window((2, 0), 1_500, (700, 200), 501, (14.0, 2.0), (4.0, 2.0), 0),
        );
        world.activity = sglang_activity(
            &[
                (2, 500, 100, 0, 12.5, -1.0, 4_000),
                (1, 1_000, 401, -1, -1.0, -1.0, 10_000),
            ],
            &[stamp2.clone(), stamp1.clone()],
        );
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(12), |_, detail| {
        detail
            .activity
            .iter()
            .any(|row| row.id == 2 && row.engine_gen_tps.is_some())
    });
    let reported = detail.activity.iter().find(|row| row.id == 2).expect("row");
    assert_eq!(reported.cached_tokens, Some(0));
    assert_eq!(reported.prompt_tps, Some(12.5));
    assert_eq!(reported.gen_tps, None);
    assert_eq!(reported.engine_prompt_tps, None);
    assert!((reported.engine_gen_tps.expect("unreported gen") - 49.5).abs() < 1e-6);
    let kept = detail.activity.iter().find(|row| row.id == 1).expect("row");
    assert_eq!(kept.cached_tokens, Some(600));
    assert!((kept.engine_prompt_tps.expect("kept") - 200.0).abs() < 1e-6);
    assert!((kept.engine_gen_tps.expect("kept") - 50.0).abs() < 1e-6);

    // A rate was reported and the cached count was not, after the second
    // row is decided. The count is filled (100); the prompt rate stays 8.
    // The stamp is chosen before the update: sleeping while that closure
    // holds the server lock stalls a metrics read, and a read over two
    // seconds is treated as a model load and not scraped again.
    let stamp3 = sglang_stamp_at(wait_row_second(sec2 + 8));
    server.update(|world| {
        world.metrics.insert(
            "flash".to_owned(),
            sglang_window((3, 0), 2_500, (800, 200), 902, (24.0, 3.0), (6.0, 3.0), 0),
        );
        world.activity = sglang_activity(
            &[
                (3, 1_000, 401, -1, 8.0, -1.0, 10_000),
                (2, 500, 100, 0, 12.5, -1.0, 4_000),
                (1, 1_000, 401, -1, -1.0, -1.0, 10_000),
            ],
            &[stamp3, stamp2, stamp1],
        );
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(12), |_, detail| {
        detail
            .activity
            .iter()
            .any(|row| row.id == 3 && row.cached_tokens.is_some())
    });
    let partial = detail.activity.iter().find(|row| row.id == 3).expect("row");
    assert_eq!(partial.cached_tokens, Some(100));
    assert_eq!(partial.prompt_tps, Some(8.0));
    assert_eq!(partial.engine_prompt_tps, None);
    assert!((partial.engine_gen_tps.expect("gen") - 50.0).abs() < 1e-6);
}

// ---- #46: engine and counter state across restarts and engine changes ------

/// A failed `/running` read forgets the engines' windows: the old
/// process's acceptance does not outlive it.
#[test]
fn engine_windows_are_forgotten_across_a_down_and_up() {
    let mut world = World::running(running_cmd("qwen3.8-27b-vllm", CONTAINER_CMD));
    world.metrics.insert(
        "qwen3.8-27b-vllm".to_owned(),
        vllm_metrics(42_000, 20_000, 60_000, 38_000),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let spec = |view: &LlamaView| {
        view.models
            .first()
            .and_then(|model| model.backend)
            .and_then(|info| info.engine.spec_permille)
    };
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        spec(view) == Some(633)
    });
    server.update(|world| {
        world.metrics.insert(
            "qwen3.8-27b-vllm".to_owned(),
            vllm_metrics(42_340, 20_100, 60_300, 38_240),
        );
    });
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        spec(view) == Some(800)
    });
    server.update(|world| world.running_status = 500);
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.ai == AiState::Down
    });
    server.update(|world| world.running_status = 200);
    // The first read after the gap starts over: the server's totals since
    // start, not the old window's 80 % kept because nothing moved.
    let (view, _) = wait_msg(&rx, Duration::from_secs(3), |view, _| spec(view).is_some());
    assert_eq!(spec(&view), Some(634));
}

/// An unloaded model's decode counter starts from 0 when it is loaded
/// again, so a new process already past the old value is counted in full.
#[test]
fn a_reloaded_process_is_counted_from_zero() {
    let id = "qwen3.6-35b-a3b";
    let mut world = World::running(running_model(id, "Qwen", "ready"));
    world
        .metrics
        .insert(id.to_owned(), metrics_body(1_000, 0.0));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(0)
    });
    server.update(|world| {
        world
            .metrics
            .insert(id.to_owned(), metrics_body(1_200, 0.0));
    });
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(200)
    });
    server.update(|world| world.running = running_model(id, "Qwen", "starting"));
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.models
            .first()
            .is_some_and(|model| model.state == "starting")
    });
    server.update(|world| {
        world.running = running_model(id, "Qwen", "ready");
        world
            .metrics
            .insert(id.to_owned(), metrics_body(1_500, 0.0));
    });
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.models
            .first()
            .is_some_and(|model| model.state == "ready")
            && view.decoded_total.is_some_and(|total| total != 200)
    });
    assert_eq!(view.decoded_total, Some(1_700), "200 + the new 1,500");
}

/// One llama-swap id that moves from SGLang to llama.cpp goes back to
/// counting its prompt tokens from activity rows.
#[test]
fn prompt_cache_follows_an_engine_change() {
    let mut world = World::running(running_cmd("flash", SGLANG_CMD));
    world
        .metrics
        .insert("flash".to_owned(), sglang_cached_metrics(10_000, 9_000));
    world.activity = cache_page(&[cache_row(1, "flash", 5_000, 0)]);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        prompt_cache_of(detail, "flash") == Some((0, Some(0)))
    });
    server.update(|world| {
        world
            .metrics
            .insert("flash".to_owned(), sglang_cached_metrics(10_500, 9_400));
    });
    wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        prompt_cache_of(detail, "flash") == Some((500, Some(400)))
    });
    server.update(|world| {
        world.running = running_cmd("flash", "llama-server -m x.gguf");
        world
            .metrics
            .insert("flash".to_owned(), metrics_body(10, 0.0));
    });
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.models
            .first()
            .and_then(|model| model.backend)
            .is_some_and(|info| info.kind == llama_core::backend::Backend::LlamaCpp)
    });
    server.update(|world| {
        world.activity = cache_page(&[
            cache_row(2, "flash", 69, 553),
            cache_row(1, "flash", 5_000, 0),
        ]);
    });
    wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        prompt_cache_of(detail, "flash") == Some((1_122, Some(953)))
    });
}

// ---- #44: RECENT's own rows across llama-swap restarts ----------------------

/// One activity row of llama-swap generation `hour` (its timestamp hour),
/// so a reused id has a different fingerprint.
fn gen_row(id: i64, model: &str, hour: u32, output: i64) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "timestamp": format!("2026-09-29T{hour:02}:00:{id:02}Z"),
        "model": model,
        "tokens": {"input_tokens": 10, "output_tokens": output, "cache_tokens": -1,
                   "prompt_per_second": -1, "tokens_per_second": -1},
        "duration_ms": 1000,
        "resp_status_code": 200
    })
}

fn recent_of(detail: &LlamaDetail) -> Vec<(i64, Option<u64>)> {
    detail
        .activity
        .iter()
        .map(|row| (row.id, row.output_tokens))
        .collect()
}

/// llama-swap restarts and numbers its rows from 0 again, once through an
/// empty list and once straight to a top id at or above the old one: the
/// old rows stay in RECENT, the new ones go on top, and every new row is
/// counted once.
#[test]
fn a_llama_swap_restart_keeps_recent_and_counts_every_reused_id() {
    let mut world = World::running(running_cmd("tabby", "python3 main.py --port 5000"));
    world.activity = cache_page(&[gen_row(0, "tabby", 10, 3)]);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(0)
    });
    server.update(|world| {
        world.activity = cache_page(&[gen_row(1, "tabby", 10, 40), gen_row(0, "tabby", 10, 3)]);
    });
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(40)
    });

    // Restart: the list comes back empty. RECENT keeps its rows.
    server.update(|world| world.activity = br#"{"data":[]}"#.to_vec());
    thread::sleep(Duration::from_millis(900));
    let (_, detail) = wait_msg(&rx, Duration::from_secs(1), |_, _| true);
    assert_eq!(recent_of(&detail), vec![(1, Some(40)), (0, Some(3))]);
    // The new llama-swap's first rows reuse ids 0 and 1.
    server.update(|world| {
        world.activity = cache_page(&[gen_row(1, "tabby", 11, 7), gen_row(0, "tabby", 11, 5)]);
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(52)
    });
    assert_eq!(
        recent_of(&detail),
        vec![(1, Some(7)), (0, Some(5)), (1, Some(40)), (0, Some(3))]
    );

    // Restart again, straight to a page whose top id (2) is above the
    // newest seen (1): told apart by the rows' fingerprints, not skipped.
    server.update(|world| {
        world.activity = cache_page(&[
            gen_row(2, "tabby", 12, 11),
            gen_row(1, "tabby", 12, 13),
            gen_row(0, "tabby", 12, 17),
        ]);
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(93)
    });
    assert_eq!(
        recent_of(&detail),
        vec![
            (2, Some(11)),
            (1, Some(13)),
            (0, Some(17)),
            (1, Some(7)),
            (0, Some(5)),
            (1, Some(40)),
            (0, Some(3)),
        ]
    );
    // Read again: nothing is counted twice.
    thread::sleep(Duration::from_millis(900));
    let (view, _) = wait_msg(&rx, Duration::from_secs(1), |view, _| {
        view.decoded_total.is_some()
    });
    assert_eq!(view.decoded_total, Some(93));
    let restarts = log
        .lines()
        .iter()
        .filter(|line| line.contains("llama-swap restarted"))
        .count();
    assert_eq!(restarts, 2, "{:?}", log.lines());
}

/// A new row that reuses an old row's id gets no speeds from the old one,
/// and the old row keeps its own.
#[test]
fn a_reused_id_gets_no_stale_engine_speeds() {
    let vllm = "qwen3.8-27b-vllm";
    let mut world = World::running(running_cmd(vllm, CONTAINER_CMD));
    world
        .metrics
        .insert(vllm.to_owned(), vllm_finished(0, 0.0, 0, 0.0, 0));
    world.activity = cache_page(&[gen_row(1, vllm, 10, 401)]);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(3), |view, detail| {
        !detail.activity.is_empty()
            && view
                .models
                .first()
                .and_then(|model| model.backend)
                .is_some_and(|info| info.kind == llama_core::backend::Backend::Vllm)
    });
    thread::sleep(Duration::from_millis(500));
    // One request finishes and its row, id 2, appears.
    server.update(|world| {
        world
            .metrics
            .insert(vllm.to_owned(), vllm_finished(1, 2.0, 8_000, 8.0, 401));
        world.activity = cache_page(&[gen_row(2, vllm, 10, 401), gen_row(1, vllm, 10, 401)]);
    });
    wait_msg(&rx, Duration::from_secs(3), |_, detail| {
        detail
            .activity
            .iter()
            .any(|row| row.id == 2 && row.engine_prompt_tps.is_some())
    });
    // llama-swap restarts; its new rows reuse ids 0..=2 and no request
    // finished on the engine in their window.
    thread::sleep(Duration::from_millis(500));
    server.update(|world| {
        world.activity = cache_page(&[
            gen_row(2, vllm, 11, 9),
            gen_row(1, vllm, 11, 9),
            gen_row(0, vllm, 11, 9),
        ]);
    });
    wait_msg(&rx, Duration::from_secs(3), |_, detail| {
        detail.activity.len() == 5
    });
    thread::sleep(Duration::from_millis(800));
    let (_, detail) = wait_msg(&rx, Duration::from_secs(3), |_, _| true);
    let shown: Vec<(i64, Option<u64>, Option<f64>)> = detail
        .activity
        .iter()
        .map(|row| (row.id, row.output_tokens, row.engine_prompt_tps))
        .collect();
    assert_eq!(
        shown.iter().map(|row| (row.0, row.1)).collect::<Vec<_>>(),
        vec![
            (2, Some(9)),
            (1, Some(9)),
            (0, Some(9)),
            (2, Some(401)),
            (1, Some(401))
        ]
    );
    for row in &detail.activity[..3] {
        assert_eq!(
            (row.engine_prompt_tps, row.engine_gen_tps),
            (None, None),
            "new id {} has no stale speeds",
            row.id
        );
    }
    let old = &detail.activity[3];
    assert!(
        (old.engine_prompt_tps.unwrap() - 4_000.0).abs() < 1e-6,
        "kept"
    );
    assert!((old.engine_gen_tps.unwrap() - 50.0).abs() < 1e-6, "kept");
}

fn restamp(mut row: serde_json::Value, hour: u32) -> serde_json::Value {
    let id = row["id"].as_i64().expect("id");
    row["timestamp"] = format!("2026-09-29T{hour:02}:00:{id:02}Z").into();
    row
}

/// After a restart, RECENT still shows the old rows, but no capture is
/// fetched for them: their ids may now name another request. A reused id
/// is fetched once for its new request.
#[test]
fn no_capture_is_fetched_for_an_old_generation_row() {
    let mut world = World::running(running_cmd("flash", SGLANG_CMD));
    world
        .metrics
        .insert("flash".to_owned(), sglang_metrics(10, 0, 0, 0.1));
    world.activity = cache_page(&[capture_row(5, "flash", true), capture_row(4, "flash", true)]);
    world
        .captures
        .insert("5".to_owned(), capture_body("Old fifth question?", "{}"));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        detail
            .capture
            .as_ref()
            .is_some_and(|capture| capture.id == 5)
    });

    // Restart: ids 1 and 0 of the new llama-swap. Its capture 5 is gone.
    server.update(|world| {
        world.activity = cache_page(&[
            restamp(capture_row(1, "flash", true), 13),
            restamp(capture_row(0, "flash", true), 13),
        ]);
        world.captures.clear();
        world
            .captures
            .insert("1".to_owned(), capture_body("New first question?", "{}"));
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        detail
            .capture
            .as_ref()
            .is_some_and(|capture| capture.id == 1)
    });
    assert_eq!(
        cells_text(&detail.capture.expect("capture").input),
        "New first question?"
    );
    let ids: Vec<i64> = detail.activity.iter().map(|row| row.id).collect();
    assert_eq!(ids, vec![1, 0, 5, 4], "old rows stay in RECENT");
    thread::sleep(Duration::from_millis(900));
    assert_eq!(
        capture_hits(&server),
        vec!["/api/captures/5".to_owned(), "/api/captures/1".to_owned()],
        "nothing fetched for the old rows 5 and 4"
    );

    // The new llama-swap reaches id 5: that is a new request, fetched once.
    server.update(|world| {
        world.activity = cache_page(&[
            restamp(capture_row(5, "flash", true), 13),
            restamp(capture_row(4, "flash", true), 13),
            restamp(capture_row(3, "flash", true), 13),
            restamp(capture_row(2, "flash", true), 13),
            restamp(capture_row(1, "flash", true), 13),
            restamp(capture_row(0, "flash", true), 13),
        ]);
        world
            .captures
            .insert("5".to_owned(), capture_body("New fifth question?", "{}"));
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        detail
            .capture
            .as_ref()
            .is_some_and(|capture| cells_text(&capture.input) == "New fifth question?")
    });
    let ids: Vec<i64> = detail.activity.iter().map(|row| row.id).collect();
    assert_eq!(ids, vec![5, 4, 3, 2, 1, 0, 5, 4], "eight rows with text on");
    thread::sleep(Duration::from_millis(900));
    assert_eq!(
        capture_hits(&server),
        vec![
            "/api/captures/5".to_owned(),
            "/api/captures/1".to_owned(),
            "/api/captures/5".to_owned()
        ]
    );
}

// ---- #70: polling never makes llama-swap load a model ----

const VLLM_CMD: &str = "vllm serve /models/q --port 5000";

/// Wait until `pred` holds for a sample, asserting `each` on every sample.
fn wait_each(
    rx: &poller::SampleRx,
    timeout: Duration,
    mut each: impl FnMut(&LlamaView, &LlamaDetail),
    mut pred: impl FnMut(&LlamaView, &LlamaDetail) -> bool,
) -> (LlamaView, LlamaDetail) {
    wait_msg(rx, timeout, |view, detail| {
        each(view, detail);
        pred(view, detail)
    })
}

fn engine_numbers(view: &LlamaView) -> bool {
    view.models
        .first()
        .and_then(|model| model.backend)
        .is_some_and(|info| info.running.is_some() && info.engine.spec_permille.is_some())
}

/// Every upstream request comes right after a `/running` read and is
/// followed by one: the fresh gate and its self-check (#70).
fn assert_gated(hits: &[String]) {
    // #80: `/api/events` is another thread's connection, not an upstream
    // read; it may land anywhere in the order. So is `/api/gpu` (#167).
    let hits: Vec<String> = hits
        .iter()
        .filter(|path| *path != "/api/events" && *path != "/api/gpu")
        .cloned()
        .collect();
    let hits = hits.as_slice();
    for (i, path) in hits.iter().enumerate() {
        if path.contains("/upstream/") {
            assert_eq!(
                i.checked_sub(1).map(|j| hits[j].as_str()),
                Some("/running"),
                "no fresh /running before {path}: {hits:?}"
            );
            if i + 1 < hits.len() {
                assert_eq!(
                    hits[i + 1],
                    "/running",
                    "no check read after {path}: {hits:?}"
                );
            }
        }
    }
}

/// #70 (b): a 409 from `/upstream/<id>/metrics` (llama-swap's
/// `upstream.ignorePaths` answer) means "not loaded": the engine numbers
/// go, one log line per change, llama-swap stays up and nothing is a tap
/// failure, and there is no second read in the same round.
#[test]
fn a_409_means_not_loaded_quietly_and_is_not_retried_in_the_round() {
    let mut world = World::running(running_cmd("v", VLLM_CMD));
    world
        .metrics
        .insert("v".to_owned(), vllm_metrics(42_000, 20_000, 60_000, 38_000));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let up = |view: &LlamaView, _: &LlamaDetail| assert_eq!(view.ai, AiState::Loaded);
    wait_each(&rx, Duration::from_secs(2), up, |view, _| {
        engine_numbers(view)
    });

    let refuse = |world: &mut World| {
        world
            .upstream_status
            .insert("/upstream/v/metrics".to_owned(), (409, None));
    };
    server.update(refuse);
    let refused_at = Instant::now();
    let (view, _) = wait_each(&rx, Duration::from_secs(2), up, |view, _| {
        !engine_numbers(view)
    });
    // The model is still listed (llama-swap says ready), its numbers absent.
    assert_eq!(view.models.len(), 1);
    let info = view.models[0].backend.expect("backend");
    assert_eq!(
        (info.running, info.queued, info.kv_permille),
        (None, None, None)
    );
    assert_eq!(info.engine, llama_core::backend::EngineStats::default());
    // Several more rounds of 409s: still up, still absent, still one line.
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(800) {
        if let Ok((view, _)) = rx.try_recv() {
            assert_eq!(view.ai, AiState::Loaded);
            assert!(!engine_numbers(&view));
        }
        thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(
        lines_count(&log, "llama-swap says not loaded"),
        1,
        "{:?}",
        log.lines()
    );
    assert_eq!(
        lines_count(&log, "v: llama-swap says not loaded; skipping until ready"),
        1
    );
    for quiet in ["metrics:", "no /metrics", "llama: down", "suspected"] {
        assert_eq!(lines_count(&log, quiet), 0, "{quiet}: {:?}", log.lines());
    }
    // No retry within a round: refused reads are a metrics interval apart.
    let refused: Vec<Instant> = {
        let world = server.world.lock().unwrap_or_else(|err| err.into_inner());
        world
            .upstream_at
            .iter()
            .filter(|(path, at)| path == "/upstream/v/metrics" && *at >= refused_at)
            .map(|(_, at)| *at)
            .collect()
    };
    assert!(refused.len() >= 3, "{refused:?}");
    for pair in refused.windows(2) {
        assert!(
            pair[1].duration_since(pair[0]) >= Duration::from_millis(150),
            "retried within a round: {refused:?}"
        );
    }
    assert_gated(&server.hits());

    // Loaded again: the numbers come back; a new 409 is a new line.
    server.update(|world| {
        world.upstream_status.clear();
    });
    wait_each(&rx, Duration::from_secs(2), up, |view, _| {
        engine_numbers(view)
    });
    server.update(refuse);
    wait_each(&rx, Duration::from_secs(2), up, |view, _| {
        !engine_numbers(view)
    });
    thread::sleep(Duration::from_millis(300));
    assert_eq!(
        lines_count(&log, "llama-swap says not loaded"),
        2,
        "{:?}",
        log.lines()
    );
}

/// #70: any other non-2xx answer is "not available now" too: no second
/// read of that model in the round (`/slots` after a failed `/metrics`).
#[test]
fn a_non_2xx_upstream_answer_skips_the_model_for_the_round() {
    let mut world = World::running(running_model("l", "L", "ready"));
    world.metrics.insert("l".to_owned(), metrics_body(5, 1.0));
    world
        .slots
        .insert("l".to_owned(), fixture("slots-sample.json"));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, _rx) = spawn(&config, &log);
    let start = Instant::now();
    while !server.hits().iter().any(|path| path == "/upstream/l/slots") {
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "{:?}",
            server.hits()
        );
        thread::sleep(Duration::from_millis(10));
    }
    server.update(|world| {
        world
            .upstream_status
            .insert("/upstream/l/metrics".to_owned(), (503, None));
    });
    let failed_at = Instant::now();
    thread::sleep(Duration::from_millis(1200));
    let world = server.world.lock().unwrap_or_else(|err| err.into_inner());
    let after: Vec<&(String, Instant)> = world
        .upstream_at
        .iter()
        .filter(|(_, at)| *at >= failed_at)
        .collect();
    // A `/slots` read never follows a failed `/metrics` read of the same
    // round: within a round they are a few milliseconds apart (two `/running`
    // reads between them). Separate rounds can land under 100 ms apart on a
    // loaded host, so the bound is 30 ms.
    for pair in after.windows(2) {
        if pair[0].0 == "/upstream/l/metrics" && pair[1].0 == "/upstream/l/slots" {
            assert!(
                pair[1].1.duration_since(pair[0].1) >= Duration::from_millis(30),
                "{after:?}"
            );
        }
    }
    drop(world);
    assert_eq!(
        lines_count(&log, "metrics: http status"),
        1,
        "{:?}",
        log.lines()
    );
    assert_gated(&server.hits());
}

/// #70 (c): in every round the order is `/running`, then the upstream
/// read, then `/running` again, for every backend and every upstream path.
#[test]
fn every_upstream_read_is_between_two_running_reads() {
    let body = serde_json::to_vec(&serde_json::json!({
        "running": [
            {"model": "l", "state": "ready", "cmd": "llama-server -m /m/x.gguf"},
            {"model": "v", "state": "ready", "cmd": VLLM_CMD},
            {"model": "p", "state": "ready", "cmd": CONTAINER_CMD},
        ]
    }))
    .expect("json");
    let mut world = World::running(body);
    world.metrics.insert("l".to_owned(), metrics_body(5, 1.0));
    world
        .slots
        .insert("l".to_owned(), fixture("slots-sample.json"));
    world
        .metrics
        .insert("v".to_owned(), vllm_metrics(1, 1, 1, 1));
    world
        .metrics
        .insert("p".to_owned(), vllm_metrics(1, 1, 1, 1));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, _rx) = spawn(&config, &log);
    let wanted = [
        "/upstream/l/metrics",
        "/upstream/l/slots",
        "/upstream/v/metrics",
        "/upstream/p/metrics",
    ];
    let start = Instant::now();
    while !wanted
        .iter()
        .all(|want| server.hits().iter().filter(|hit| hit == want).count() >= 2)
    {
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "{:?}",
            server.hits()
        );
        thread::sleep(Duration::from_millis(10));
    }
    assert_gated(&server.hits());
    assert_eq!(lines_count(&log, "suspected"), 0, "{:?}", log.lines());
}

/// `/running` for #70 (d): `p` (probed) is ready only in read 1 and from
/// read 12 on; `l` (llama.cpp, busy) is gone from every third read. No
/// model is ever in another state, so no read is a swap.
fn flapping(n: u32) -> Vec<u8> {
    let mut models = Vec::new();
    if n == 1 || n >= 12 {
        models.push(serde_json::json!({"model": "p", "state": "ready", "cmd": CONTAINER_CMD}));
    }
    if !n.is_multiple_of(3) {
        models.push(
            serde_json::json!({"model": "l", "state": "ready", "cmd": "llama-server -m /m/x.gguf"}),
        );
    }
    serde_json::to_vec(&serde_json::json!({ "running": models })).expect("json")
}

/// #70 (d): the backend probe and `/slots` obey the same gate as
/// `/metrics`: each is sent only right after a `/running` read that lists
/// its model as `ready`.
#[test]
fn the_probe_and_slots_obey_the_fresh_gate() {
    let mut world = World::running(Vec::new());
    world.script = Some(flapping);
    world.metrics.insert("l".to_owned(), metrics_body(5, 1.0));
    world
        .slots
        .insert("l".to_owned(), fixture("slots-sample.json"));
    world
        .metrics
        .insert("p".to_owned(), vllm_metrics(1, 1, 1, 1));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, _rx) = spawn(&config, &log);
    let start = Instant::now();
    while !(server.hits().iter().any(|hit| hit == "/upstream/p/metrics")
        && server
            .hits()
            .iter()
            .filter(|hit| *hit == "/upstream/l/slots")
            .count()
            >= 2)
    {
        assert!(
            start.elapsed() < Duration::from_secs(6),
            "{:?}",
            server.hits()
        );
        thread::sleep(Duration::from_millis(10));
    }
    let hits = server.hits();
    assert_gated(&hits);
    // Replay the script: each upstream request's `/running` listed it.
    let mut served = 0;
    for (i, path) in hits.iter().enumerate() {
        if path == "/running" {
            served += 1;
            continue;
        }
        let Some(rest) = path.strip_prefix("/upstream/") else {
            continue;
        };
        let id = rest.split('/').next().expect("id");
        let listed: serde_json::Value = serde_json::from_slice(&flapping(served)).expect("json");
        let ready = listed["running"]
            .as_array()
            .expect("array")
            .iter()
            .any(|model| model["model"] == id && model["state"] == "ready");
        assert!(ready, "{path} after /running #{served} (hit {i}): {hits:?}");
    }
    // `p` was ready in the first read only, then gone until read 12: no
    // probe before that.
    let probe_at = hits
        .iter()
        .position(|hit| hit == "/upstream/p/metrics")
        .expect("probe");
    let running_before = hits[..probe_at]
        .iter()
        .filter(|hit| *hit == "/running")
        .count();
    assert!(running_before >= 12, "{hits:?}");
}

/// #70: a model starting right after one of our upstream reads is a
/// suspected load: a warning naming the model and path, a 5-minute stop
/// on that model's upstream reads, and a count on the detail.
#[test]
fn a_model_starting_after_our_read_is_a_suspected_load() {
    let mut world = World::running(running_model("l", "L", "ready"));
    world.metrics.insert("l".to_owned(), metrics_body(5, 0.0));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let start = Instant::now();
    while !server.hits().iter().any(|hit| hit == "/upstream/l/metrics") {
        assert!(start.elapsed() < Duration::from_secs(2));
        thread::sleep(Duration::from_millis(10));
    }
    // Once our next upstream read is served, `x` is starting: as if that
    // read had loaded it.
    server.update(|world| {
        world.after_upstream = Some(
            serde_json::to_vec(&serde_json::json!({
                "running": [
                    {"model": "l", "state": "ready", "cmd": "llama-server"},
                    {"model": "x", "state": "starting"}
                ]
            }))
            .expect("json"),
        );
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        !detail.suspected_loads.is_empty()
    });
    assert_eq!(detail.suspected_loads, vec![("l".to_owned(), 1)]);
    let warned = log
        .lines()
        .into_iter()
        .filter(|line| line.contains("suspected model load"))
        .collect::<Vec<_>>();
    assert_eq!(warned.len(), 1, "{:?}", log.lines());
    assert!(
        warned[0].contains("l: suspected model load after GET /upstream/l/metrics"),
        "{warned:?}"
    );
    assert!(warned[0].contains("300 s"), "{warned:?}");
    // The swap ends, but `l` stays backed off.
    server.update(|world| world.running_after = None);
    let quiet_from = server.hits().len();
    thread::sleep(Duration::from_millis(800));
    let hits = server.hits();
    assert!(
        hits[quiet_from..]
            .iter()
            .all(|hit| !hit.contains("/upstream/")),
        "{:?}",
        &hits[quiet_from..]
    );
    assert!(hits[quiet_from..].iter().any(|hit| hit == "/running"));
}

/// #70: an upstream read slower than 2 s may have waited for a load: a
/// suspected load, and that model is left alone.
#[test]
fn a_slow_upstream_read_is_a_suspected_load() {
    let mut world = World::running(running_model("l", "L", "ready"));
    world.metrics.insert("l".to_owned(), metrics_body(5, 0.0));
    world.upstream_delay.insert(
        "/upstream/l/metrics".to_owned(),
        Duration::from_millis(2100),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch_with(server.port, 12, 4_194_304, 0.15, "");
    // A timeout over 2 s needs a longer metrics interval.
    let path =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("watch-{}.toml", server.port));
    let text = std::fs::read_to_string(&path)
        .expect("watch.toml")
        .replace("metrics_interval_s = 0.2", "metrics_interval_s = 2.5")
        .replace("metrics_timeout_s = 0.1", "metrics_timeout_s = 2.4");
    std::fs::write(&path, text).expect("write");
    drop(config);
    let config = Config::load_validated(&path, 8).expect("valid");
    let (_poller, rx) = spawn(&config, &log);
    let (_, detail) = wait_msg(&rx, Duration::from_secs(5), |_, detail| {
        !detail.suspected_loads.is_empty()
    });
    assert_eq!(detail.suspected_loads, vec![("l".to_owned(), 1)]);
    assert_eq!(
        lines_count(
            &log,
            "l: suspected model load after GET /upstream/l/metrics (it took 2."
        ),
        1,
        "{:?}",
        log.lines()
    );
    let reads = server
        .hits()
        .iter()
        .filter(|hit| *hit == "/upstream/l/metrics")
        .count();
    thread::sleep(Duration::from_millis(3000));
    let later = server
        .hits()
        .iter()
        .filter(|hit| *hit == "/upstream/l/metrics")
        .count();
    assert_eq!(reads, later, "backed off: {:?}", server.hits());
}

/// #70: an upstream redirect is not followed, so it cannot reach another
/// model's path.
#[test]
fn an_upstream_redirect_is_not_followed() {
    let mut world = World::running(running_model("l", "L", "ready"));
    world.upstream_status.insert(
        "/upstream/l/metrics".to_owned(),
        (302, Some("/upstream/other/metrics".to_owned())),
    );
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, _rx) = spawn(&config, &log);
    let start = Instant::now();
    while server
        .hits()
        .iter()
        .filter(|hit| *hit == "/upstream/l/metrics")
        .count()
        < 3
    {
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "{:?}",
            server.hits()
        );
        thread::sleep(Duration::from_millis(10));
    }
    let hits = server.hits();
    assert!(hits.iter().all(|hit| !hit.contains("other")), "{hits:?}");
    assert_gated(&hits);
}

/// One activity row with a status, a duration and llama.cpp draft fields.
fn activity_row(
    id: i64,
    model: &str,
    status: u16,
    ms: u64,
    drafts: (i64, i64),
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "timestamp": format!("2026-10-07T10:00:{id:02}Z"),
        "model": model,
        "tokens": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_tokens": 2,
            "draft_tokens": drafts.0,
            "draft_acc_tokens": drafts.1,
            "prompt_per_second": 100.0,
            "tokens_per_second": 50.0
        },
        "duration_ms": ms,
        "resp_status_code": status
    })
}

/// `fixtures/llama/llamacpp-metrics.txt` with the request-end counters moved.
fn llamacpp_metrics(predicted: u64, prompt_s: f64, predicted_s: f64) -> Vec<u8> {
    String::from_utf8(fixture("llamacpp-metrics.txt"))
        .expect("utf-8")
        .replace(
            "llamacpp:tokens_predicted_total 2048",
            &format!("llamacpp:tokens_predicted_total {predicted}"),
        )
        .replace(
            "llamacpp:prompt_seconds_total 4.312",
            &format!("llamacpp:prompt_seconds_total {prompt_s}"),
        )
        .replace(
            "llamacpp:tokens_predicted_seconds_total 40.96",
            &format!("llamacpp:tokens_predicted_seconds_total {predicted_s}"),
        )
        .into_bytes()
}

/// #71: a llama.cpp model's cumulative numbers come from its request-end
/// counters and llama-swap's activity rows, by llama-swap id.
#[test]
fn llamacpp_series_are_cumulative_and_count_activity_rows() {
    let id = "qwen3.6-35b-a3b";
    let mut world = World::running(running_model(id, "Qwen 35B", "ready"));
    world
        .metrics
        .insert(id.to_owned(), llamacpp_metrics(2048, 4.312, 40.96));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (_, detail) = wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        detail
            .series
            .first()
            .is_some_and(|s| s.generation_tokens == Some(0) && s.requests_ok == Some(0))
    });
    let series = &detail.series[0];
    assert_eq!(series.id, id);
    assert_eq!(series.model, "Qwen 35B");
    assert_eq!(series.running, Some(1), "requests_processing");
    assert_eq!(series.waiting, Some(2), "requests_deferred");
    assert_eq!(series.kv_permille, None, "#79: b11429 has no KV gauge");
    assert_eq!(
        series.prefill_seconds,
        Some(0.0),
        "first read is a baseline"
    );
    assert_eq!(series.ttft, None, "llama.cpp reports no TTFT");

    server.update(|world| {
        world
            .metrics
            .insert(id.to_owned(), llamacpp_metrics(2148, 5.312, 42.96));
        world.activity = serde_json::to_vec(&serde_json::json!({ "data": [
            activity_row(1, id, 200, 1000, (10, 7)),
            activity_row(2, id, 500, 500, (-1, -1)),
        ]}))
        .expect("json");
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(3), |_, detail| {
        detail
            .series
            .first()
            .is_some_and(|s| s.generation_tokens == Some(100) && s.requests_error == Some(1))
    });
    let series = &detail.series[0];
    assert_eq!(series.requests_ok, Some(1));
    let close = |got: Option<f64>, want: f64| got.is_some_and(|v| (v - want).abs() < 1e-9);
    assert!(close(series.prefill_seconds, 1.0), "{series:?}");
    assert!(close(series.decode_seconds, 2.0), "{series:?}");
    assert_eq!(series.spec_draft_tokens, Some(10));
    assert_eq!(series.spec_accepted_tokens, Some(7));
    let e2e = series.e2e.expect("activity durations");
    assert_eq!((e2e.sum, e2e.count), (1.5, 2.0));
}

/// #71: vLLM's numbers fill the same fields from its own counters and
/// histograms; its live gauges stay on `BackendInfo`, not the series.
#[test]
fn vllm_series_fill_the_same_fields() {
    let id = "qwen3.8-27b-vllm";
    let mut world = World::running(running_cmd(id, VLLM_CMD));
    world
        .metrics
        .insert(id.to_owned(), vllm_metrics(42_000, 20_000, 60_000, 38_000));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        detail
            .series
            .first()
            .is_some_and(|s| s.generation_tokens == Some(0))
    });
    server.update(|world| {
        world
            .metrics
            .insert(id.to_owned(), vllm_metrics(42_340, 20_100, 60_300, 38_240));
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(2), |_, detail| {
        detail
            .series
            .first()
            .is_some_and(|s| s.generation_tokens == Some(340))
    });
    let series = &detail.series[0];
    assert_eq!(series.running, None);
    assert_eq!(series.prefill_seconds, Some(0.0));
    assert_eq!(series.decode_seconds, Some(0.0));
    for hist in [series.ttft, series.itl, series.e2e] {
        let hist = hist.expect("engine histogram");
        assert_eq!((hist.sum, hist.count), (0.0, 0.0), "baseline, unchanged");
    }
    assert_eq!(series.spec_draft_tokens, None, "vLLM's are engine counters");
}

/// #79: a llama.cpp model's KV is its slots' held tokens, idle ones
/// included. Without `-np` the cache is unified (one 65,536 pool all three
/// slots report); with `-np 3` each slot has its own 65,536.
#[test]
fn llamacpp_kv_is_the_slots_held_tokens_shared_or_per_slot() {
    for (cmd, capacity, unified) in [
        ("llama-server -m /m/x-Q4_K_M.gguf -c 65536", 65_536, true),
        (
            "llama-server -m /m/x-Q4_K_M.gguf -c 196608 -np 3",
            196_608,
            false,
        ),
        (
            "llama-server -m /m/x-Q4_K_M.gguf -np 3 --kv-unified",
            65_536,
            true,
        ),
    ] {
        let mut world = World::running(running_cmd("l", cmd));
        world.metrics.insert("l".to_owned(), metrics_body(5, 1.0));
        world.slots.insert("l".to_owned(), fixture("slots-kv.json"));
        let server = Server::start(world);
        let log = MemLog::new();
        let config = watch(server.port, 12, 4_194_304, 0.15);
        let (_poller, rx) = spawn(&config, &log);
        let (view, _) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
            view.models
                .first()
                .and_then(|model| model.backend)
                .is_some_and(|info| info.kv.is_some())
        });
        let kv = view.models[0].backend.and_then(|info| info.kv).expect("kv");
        assert_eq!(kv.used, Some(50_000), "{cmd}: 41,000 busy + 9,000 idle");
        assert_eq!(kv.capacity, Some(capacity), "{cmd}");
        assert_eq!(kv.sessions, Some(2), "{cmd}: the empty slot holds none");
        assert_eq!(kv.unified, Some(unified), "{cmd}");
        assert!(!kv.unified_assumed && !kv.approx, "{cmd}");
        assert_eq!(kv.cached, None, "{cmd}");
    }
}

/// #92: a spec file's number lands beside the flag source. The other model
/// has no file, so its spec field stays blank and its flag still shows.
#[test]
fn a_spec_file_fills_setup_and_a_missing_file_leaves_the_flags() {
    let body = serde_json::to_vec(&serde_json::json!({
        "running": [
            {
                "model": "example-35b-q6",
                "name": "Example 35B",
                "state": "ready",
                "cmd": "llama-server -c 4096"
            },
            {
                "model": "other-model",
                "state": "ready",
                "cmd": "llama-server -c 4096"
            }
        ]
    }))
    .expect("json");
    let server = Server::start(World::running(body));
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("specs-{}", server.port));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("specs dir");
    std::fs::write(
        dir.join("example-35b-q6.json"),
        br#"{"served_context":131072}"#,
    )
    .expect("spec");
    let spec = dir.display().to_string();
    assert!(dir.is_absolute(), "{spec}");
    assert!(!spec.contains(['"', '\\']), "{spec}");
    let path =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("watch-spec-{}.toml", server.port));
    let text = format!(
        r#"
[llama]
url = "http://127.0.0.1:{port}"
running_interval_s = 0.35
running_timeout_s = 0.15
metrics_interval_s = 0.2
metrics_timeout_s = 0.1
slots_interval_s = 0.5
slots_timeout_s = 0.3
slots_max_bytes = 4194304
activity_interval_s = 0.35
activity_timeout_s = 0.1
input_tail_chars = 256
output_tail_chars = 256

[models]
max_name_chars = 12

[setup]
defaults = false
specs_dir = "{spec}"

[[setup.field]]
row = "ctx"
source = "spec:served_context"
kind = "number"
order = 20

[[setup.field]]
row = "ctx"
source = "flag:-c"
kind = "number"
order = 21
"#,
        port = server.port
    );
    std::fs::write(&path, text).expect("write watch.toml");
    let config = Config::load_validated(&path, 8).expect("valid watch config");
    let log = MemLog::new();
    let (_poller, rx) = spawn(&config, &log);
    let (_, detail) = wait_msg(&rx, Duration::from_secs(3), |_, detail| {
        detail.setup.iter().any(|setup| {
            setup.id == "example-35b-q6"
                && setup.found.iter().any(|found| found.value == "131072")
                && setup.found.iter().any(|found| found.value == "4096")
        })
    });
    let values = |id: &str| -> Vec<String> {
        detail
            .setup
            .iter()
            .find(|setup| setup.id == id)
            .unwrap_or_else(|| panic!("no setup for {id}"))
            .found
            .iter()
            .map(|found| found.value.clone())
            .collect()
    };
    let example = values("example-35b-q6");
    assert!(example.iter().any(|value| value == "131072"), "{example:?}");
    assert!(example.iter().any(|value| value == "4096"), "{example:?}");
    let other = values("other-model");
    assert!(other.iter().any(|value| value == "4096"), "{other:?}");
    assert!(!other.iter().any(|value| value == "131072"), "{other:?}");
    assert!(
        log.lines().iter().all(|line| !line.contains("setup spec:")),
        "{:?}",
        log.lines()
    );
}

fn jobs_scratch(name: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let dir = root.join(format!("poller-job-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("jobs dir");
    dir
}

fn jobs_watch(dir: &std::path::Path, extra: &str) -> ValidWatchConfig {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("watch-jobs-{}.toml", std::process::id()));
    let text = format!(
        r#"
[llama]
url = "http://127.0.0.1:1"
enabled = false

[jobs]
dir = "{}"
stale_s = 180
{extra}
"#,
        dir.display()
    );
    std::fs::write(&path, text).expect("write watch.toml");
    Config::load_validated(&path, 8).expect("valid watch config")
}

#[test]
fn jobs_dir_with_llama_off_is_read() {
    let dir = jobs_scratch("off");
    std::fs::write(
        dir.join("example-trainer.json"),
        br#"{"schema":1,"name":"example-trainer","kind":"training","state":"running","title":"Example-2B"}"#,
    )
    .unwrap();
    let config = jobs_watch(&dir, "");
    assert!(!config.llama.enabled);
    assert_eq!(config.jobs.dir.as_deref(), Some(dir.as_path()));
    let log = MemLog::new();
    let (_poller, rx) = spawn(&config, &log);
    let (_view, detail) = wait_msg(&rx, Duration::from_secs(3), |_, detail| {
        !detail.jobs.is_empty()
    });
    assert_eq!(detail.jobs[0].name, "example-trainer");
    assert_eq!(detail.jobs[0].kind.as_deref(), Some("training"));
    assert_eq!(detail.jobs[0].state, llama_watch::jobs::JobState::Running);
}

// ---- #180: tqLLM's monitor snapshot ---------------------------------------------

/// tqLLM in podman, told by its entry point. Invented paths and image.
const TQLLM_CMD: &str = "podman run --rm --name tq --device nvidia.com/gpu=all -p 127.0.0.1:${PORT}:8000 -v /models/invented:/models:ro --entrypoint tqserve ghcr.io/example/moe-server:1 --model /models/invented.gguf --context 65536 --np 2 --port 8000";

/// `fixtures/llama/tqllm-snapshot.json` under another `instance_id`, its
/// requests edited.
fn tq_snapshot(instance: &str, edit: impl FnOnce(&mut Vec<serde_json::Value>)) -> Vec<u8> {
    let mut doc: serde_json::Value =
        serde_json::from_slice(&fixture("tqllm-snapshot.json")).expect("tqllm fixture");
    doc["instance_id"] = serde_json::json!(instance);
    edit(doc["requests"].as_array_mut().expect("requests"));
    serde_json::to_vec(&doc).expect("json")
}

/// Request `id`'s member `key` set to `value`.
fn tq_set(requests: &mut [serde_json::Value], id: u64, key: &str, value: serde_json::Value) {
    let req = requests
        .iter_mut()
        .find(|req| req["id"] == id)
        .expect("request");
    req[key] = value;
}

const TQ_HEALTH: &[u8] =
    br#"{"status":"ok","engine":"0a1b2c3","fit":"2 slots x 32k fit in 22.5 GiB"}"#;

fn tq_hits(server: &Server, leaf: &str) -> usize {
    server
        .hits()
        .iter()
        .filter(|path| path.as_str() == format!("/upstream/tq/{leaf}"))
        .count()
}

/// Every request the watcher made is a GET on an allowed path; nothing
/// went to tqLLM's slot admin or its chat bridge (#180).
fn assert_get_only(server: &Server) {
    let requests = server.requests();
    assert!(!requests.is_empty());
    for request in &requests {
        let (method, path) = request.split_once(' ').expect("method and path");
        assert_eq!(method, "GET", "{requests:?}");
        assert!(allowed_path(path), "{requests:?}");
        for never in ["admin", "chat", "completions", "/slots"] {
            assert!(!path.contains(never), "{never}: {requests:?}");
        }
    }
}

/// #180: a server the command does not name and whose `/metrics` says
/// nothing is tqLLM when its snapshot answers, once per load and logged
/// once; from then on the snapshot is read in place of `/metrics`.
#[test]
fn a_container_tqllm_is_found_by_its_snapshot() {
    let mut world = World::running(running_cmd("tq", CONTAINER_CMD));
    world
        .snapshots
        .insert("tq".to_owned(), tq_snapshot("4242000000000000007", |_| ()));
    world.health.insert("tq".to_owned(), TQ_HEALTH.to_vec());
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, _) = wait_msg(&rx, Duration::from_secs(3), |view, _| {
        view.models
            .first()
            .and_then(|model| model.backend)
            .is_some_and(|info| {
                info.kind == llama_core::backend::Backend::Tqllm && info.running.is_some()
            })
    });
    let info = view.models[0].backend.expect("backend");
    assert_eq!(info.kv_permille, None, "no KV gauge");
    thread::sleep(Duration::from_millis(600));
    assert_eq!(tq_hits(&server, "metrics"), 1, "{:?}", server.hits());
    assert!(tq_hits(&server, "api/monitor/v1/snapshot") >= 3);
    assert_eq!(tq_hits(&server, "health"), 1);
    assert_get_only(&server);
    let lines = log.lines();
    let notes: Vec<&String> = lines
        .iter()
        .filter(|line| line.contains("names no server"))
        .collect();
    assert_eq!(notes.len(), 1, "{lines:?}");
    assert!(
        notes[0].contains(
            "tq: launch command names no server; no known metric names; /api/monitor/v1/snapshot says tqllm"
        ),
        "{lines:?}"
    );
}

/// #180: tqLLM's counters come from its requests, counted once per
/// `instance_id`: a running request's output as it grows, a request that
/// appears and finishes between two reads in full. Its gauges are the
/// running, queued and slot counts; `/health`'s `fit` is read once per
/// instance; a restart starts the counters again. Never `/metrics`.
#[test]
fn tqllm_counts_its_snapshot_and_reads_health_once_per_instance() {
    let mut world = World::running(running_cmd("tq", TQLLM_CMD));
    world
        .snapshots
        .insert("tq".to_owned(), tq_snapshot("4242000000000000007", |_| ()));
    world.health.insert("tq".to_owned(), TQ_HEALTH.to_vec());
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, detail) = wait_msg(&rx, Duration::from_secs(3), |view, detail| {
        view.decoded_total == Some(0)
            && view.models[0]
                .backend
                .is_some_and(|info| info.running.is_some())
            && detail
                .setup
                .first()
                .is_some_and(|setup| setup.engine.contains_key("fit"))
    });
    let info = view.models[0].backend.expect("backend");
    assert_eq!(info.kind, llama_core::backend::Backend::Tqllm);
    assert_eq!((info.running, info.queued), (Some(2), Some(1)));
    assert_eq!(info.max_running, Some(2), "engine.slots");
    assert_eq!(info.kv_permille, None);
    // Engine numbers from the finished requests: 248 of 376 drafted
    // tokens accepted; 220 uncached prompt tokens in 0.71 s; 310 output
    // tokens in 3.68 s (each request's first comes from its prefill).
    assert_eq!(info.engine.spec_permille, Some(660));
    assert_eq!(info.engine.prefill_tps_tenths, Some(3099));
    assert_eq!(info.engine.decode_tps_tenths, Some(842));
    // GEN and PROMPT: the sums of the running requests' own rates.
    assert_eq!(detail.gen_tps, Some(83.5));
    assert!(
        detail
            .prompt_tps
            .is_some_and(|tps| (tps - 1_137.777_778).abs() < 1e-6),
        "{:?}",
        detail.prompt_tps
    );
    let setup = &detail.setup[0];
    assert_eq!(
        setup.engine.get("fit").map(String::as_str),
        Some("2 slots x 32k fit in 22.5 GiB")
    );
    assert_eq!(
        setup.engine.get("version").map(String::as_str),
        Some("0a1b2c3")
    );
    // The snapshot's per-slot context beats the flag's 65,536 / 2.
    let tuning = view.models[0].detail.as_ref().expect("detail");
    assert_eq!(tuning.ctx, Some(32_768));
    let live = &detail.engine_live[0].live;
    assert_eq!(live.instance.as_deref(), Some("4242000000000000007"));
    assert_eq!(live.requests.len(), 3, "decode, prefill, queued");

    // Request 4 decodes 40 more; request 7 came and finished in between.
    server.update(|world| {
        world.snapshots.insert(
            "tq".to_owned(),
            tq_snapshot("4242000000000000007", |requests| {
                tq_set(requests, 4, "output_tokens", serde_json::json!(190));
                let mut seven = requests[0].clone();
                seven["id"] = serde_json::json!(7);
                seven["output_tokens"] = serde_json::json!(25);
                requests.push(seven);
            }),
        );
    });
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(65)
    });
    thread::sleep(Duration::from_millis(500));
    let (view, _) = wait_msg(&rx, Duration::from_secs(2), |_, _| true);
    assert_eq!(view.decoded_total, Some(65), "each id counted once");
    assert_eq!(tq_hits(&server, "health"), 1);

    // A new instance: its requests count from zero, and `/health` is read
    // again.
    server.update(|world| {
        world.snapshots.insert(
            "tq".to_owned(),
            tq_snapshot("4242000000000000099", |requests| {
                requests.retain(|req| req["id"] == 1);
            }),
        );
    });
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(77)
    });
    thread::sleep(Duration::from_millis(400));
    assert_eq!(tq_hits(&server, "health"), 2);
    assert_eq!(tq_hits(&server, "metrics"), 0);
    assert_get_only(&server);
    let lines = log.lines();
    assert_eq!(
        lines_count(&log, "tq: tqLLM restarted (new instance_id)"),
        1,
        "{lines:?}"
    );
    assert!(
        lines.iter().all(|line| !line.contains("no /metrics")),
        "{lines:?}"
    );
}

/// #180: a snapshot of another `schema_version` is no metrics: the model
/// is counted from activity rows, logged once for the load.
#[test]
fn tqllm_another_schema_is_no_metrics_and_logged_once() {
    let mut body: serde_json::Value =
        serde_json::from_slice(&tq_snapshot("4242000000000000007", |_| ())).expect("json");
    body["schema_version"] = serde_json::json!(2);
    let mut world = World::running(running_cmd("tq", TQLLM_CMD));
    world
        .snapshots
        .insert("tq".to_owned(), serde_json::to_vec(&body).expect("json"));
    world.activity = activity_page(&[(1, "tq", 30)]);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(0)
    });
    server.update(|world| world.activity = activity_page(&[(2, "tq", 12), (1, "tq", 30)]));
    wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(12)
    });
    thread::sleep(Duration::from_millis(600));
    assert!(tq_hits(&server, "api/monitor/v1/snapshot") >= 3);
    assert_eq!(tq_hits(&server, "health"), 0, "no snapshot, no health");
    let lines = log.lines();
    assert_eq!(
        lines_count(
            &log,
            "tq: tqLLM snapshot schema_version 2 is not 1; using llama-swap activity"
        ),
        1,
        "{lines:?}"
    );
    assert_get_only(&server);
}

/// #180: a snapshot over the 256 KiB cap is not parsed; the model is
/// counted from activity rows, logged once.
#[test]
fn tqllm_snapshot_over_the_cap_falls_back() {
    let mut body: serde_json::Value =
        serde_json::from_slice(&tq_snapshot("4242000000000000007", |_| ())).expect("json");
    body["a_future_member"] = serde_json::json!("x".repeat(poller::TQLLM_CAP));
    let mut world = World::running(running_cmd("tq", TQLLM_CMD));
    world
        .snapshots
        .insert("tq".to_owned(), serde_json::to_vec(&body).expect("json"));
    world.activity = activity_page(&[(1, "tq", 30)]);
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, detail) = wait_msg(&rx, Duration::from_secs(2), |view, _| {
        view.decoded_total == Some(0)
    });
    assert_eq!(
        view.models[0].backend.and_then(|info| info.running),
        None,
        "no gauges"
    );
    assert!(detail.engine_live.is_empty());
    thread::sleep(Duration::from_millis(600));
    assert_eq!(
        lines_count(&log, "tq: no /metrics from tqllm (oversized body)"),
        1,
        "{:?}",
        log.lines()
    );
    assert_get_only(&server);
}

/// One tqLLM activity row, as llama-swap writes it.
fn tq_activity(
    id: i64,
    stamp: &str,
    input: i64,
    output: i64,
    prompt: f64,
    gen_rate: f64,
) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({ "data": [{
        "id": id,
        "timestamp": stamp,
        "model": "tq",
        "tokens": {
            "input_tokens": input,
            "output_tokens": output,
            "cache_tokens": -1,
            "prompt_per_second": prompt,
            "tokens_per_second": gen_rate
        },
        "duration_ms": 3_500,
        "resp_status_code": 200
    }]}))
    .expect("json")
}

/// #180: a tqLLM row matched to its one finished request shows that
/// request's own prompt rate, without the cache, in place of llama-swap's,
/// which counts the cache (6,000 tokens, 5,000 of them cached, read in
/// 1 s: llama-swap says 6,000/s, the request 1,000/s). CACHED comes from
/// the request; llama-swap's IN and GEN stay.
#[test]
fn a_tqllm_row_takes_its_requests_own_prompt_rate() {
    let only_four = |requests: &mut Vec<serde_json::Value>| requests.retain(|req| req["id"] == 4);
    let mut world = World::running(running_cmd("tq", TQLLM_CMD));
    world.snapshots.insert(
        "tq".to_owned(),
        tq_snapshot("4242000000000000007", only_four),
    );
    world.health.insert("tq".to_owned(), TQ_HEALTH.to_vec());
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(3), |view, _| {
        view.models[0]
            .backend
            .is_some_and(|info| info.running == Some(1))
    });
    thread::sleep(Duration::from_millis(500));
    let now = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock");
    let stamp = sglang_stamp_at(now.as_secs() as i64);
    server.update(|world| {
        world.snapshots.insert(
            "tq".to_owned(),
            tq_snapshot("4242000000000000007", |requests| {
                only_four(requests);
                for (key, value) in [
                    ("phase", serde_json::json!("complete")),
                    ("elapsed_ms", serde_json::json!(3_500.0)),
                    ("prompt_tokens", serde_json::json!(6_000)),
                    ("prefill_processed_tokens", serde_json::json!(6_000)),
                    ("prefill_cached_tokens", serde_json::json!(5_000)),
                    ("prefill_ms", serde_json::json!(1_000.0)),
                    ("prefill_tokens_per_second", serde_json::json!(1_000.0)),
                    ("output_tokens", serde_json::json!(40)),
                    ("decode_ms", serde_json::json!(1_000.0)),
                    ("decode_tokens_per_second", serde_json::json!(39.0)),
                ] {
                    tq_set(requests, 4, key, value);
                }
            }),
        );
        world.activity = tq_activity(1, &stamp, 6_000, 40, 6_000.0, 40.0);
    });
    let (_, detail) = wait_msg(&rx, Duration::from_secs(10), |_, detail| {
        detail
            .activity
            .iter()
            .any(|row| row.id == 1 && row.cached_tokens.is_some())
    });
    let row = detail.activity.iter().find(|row| row.id == 1).expect("row");
    assert_eq!(row.cached_tokens, Some(5_000));
    assert_eq!(row.prompt_tps, Some(1_000.0), "the request's own rate");
    assert_eq!(row.gen_tps, Some(40.0), "llama-swap's");
    assert_eq!(row.input_tokens, Some(6_000), "llama-swap's");
    assert_get_only(&server);
}

/// #180: a `/health` that fails (404 here) loses only the `fit`, logged
/// once: the gauges stay through it, and the fit of the old instance goes.
#[test]
fn a_failed_tqllm_health_drops_only_the_fit() {
    let mut world = World::running(running_cmd("tq", TQLLM_CMD));
    world
        .snapshots
        .insert("tq".to_owned(), tq_snapshot("4242000000000000007", |_| ()));
    world.health.insert("tq".to_owned(), TQ_HEALTH.to_vec());
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    wait_msg(&rx, Duration::from_secs(3), |view, detail| {
        view.models[0]
            .backend
            .is_some_and(|info| info.running.is_some())
            && detail
                .setup
                .first()
                .is_some_and(|setup| setup.engine.contains_key("fit"))
    });
    // A new instance, and `/health` now fails.
    server.update(|world| {
        world.health.clear();
        world
            .snapshots
            .insert("tq".to_owned(), tq_snapshot("4242000000000000099", |_| ()));
    });
    let start = Instant::now();
    let mut fit_gone = false;
    while start.elapsed() < Duration::from_millis(1_500) {
        match rx.try_recv() {
            Ok((view, detail)) => {
                let info = view.models[0].backend.expect("backend");
                assert_eq!(info.running, Some(2), "the gauges stay");
                assert_eq!(
                    detail.setup[0].engine.get("version").map(String::as_str),
                    Some("0a1b2c3"),
                    "the other engine values stay"
                );
                fit_gone = !detail.setup[0].engine.contains_key("fit");
            }
            Err(_) => thread::sleep(Duration::from_millis(10)),
        }
    }
    assert!(fit_gone, "the old fit went");
    assert_eq!(tq_hits(&server, "health"), 2, "once per instance");
    assert_eq!(
        lines_count(&log, "tq: no tqLLM /health (http status); no fit shown"),
        1,
        "{:?}",
        log.lines()
    );
    assert_get_only(&server);
}

/// #180: what a tqLLM command says about its KV. No `--kv`, `-ctk` or
/// `-ctv`: bf16 on the wire, and SETUP's default (dim, no `kv_asked`). A
/// value that is not a token: unknown, and `kv_asked` is `--`. A valid one:
/// that value.
#[test]
fn a_tqllm_kv_is_its_default_unknown_or_what_was_asked() {
    let cmds = [
        ("tq-default", "tqserve --model /m/a.gguf --port 8086"),
        ("tq-bad", "tqserve --model /m/b.gguf --kv q8;x --port 8086"),
        (
            "tq-named",
            "tqserve --model /m/c.gguf --kv q8/tcq2 --port 8086",
        ),
    ];
    let running: Vec<serde_json::Value> = cmds
        .iter()
        .map(|(id, cmd)| serde_json::json!({"model": id, "name": id, "state": "ready", "cmd": cmd}))
        .collect();
    let mut world = World::running(
        serde_json::to_vec(&serde_json::json!({ "running": running })).expect("json"),
    );
    for (id, _) in cmds {
        world
            .snapshots
            .insert(id.to_owned(), tq_snapshot("4242000000000000007", |_| ()));
    }
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, rx) = spawn(&config, &log);
    let (view, detail) = wait_msg(&rx, Duration::from_secs(3), |view, detail| {
        view.models.len() == 3
            && view
                .models
                .iter()
                .all(|model| model.backend.is_some_and(|info| info.running.is_some()))
            && detail.setup.len() == 3
    });
    let kv = |name: &str| {
        let model = view
            .models
            .iter()
            .find(|model| model.name == name)
            .expect("model");
        let detail = model.detail.as_ref().expect("detail");
        (detail.kv_k.clone(), detail.kv_v.clone())
    };
    let asked = |id: &str| {
        detail
            .setup
            .iter()
            .find(|setup| setup.id == id)
            .expect("setup")
            .engine
            .get("kv_asked")
            .cloned()
    };
    let bf16 = Some("bf16".to_owned());
    assert_eq!(kv("tq-default"), (bf16.clone(), bf16));
    assert_eq!(asked("tq-default"), None, "SETUP draws its default dim");
    assert_eq!(kv("tq-bad"), (None, None), "unknown");
    assert_eq!(asked("tq-bad").as_deref(), Some("--"));
    assert_eq!(
        kv("tq-named"),
        (Some("q8".to_owned()), Some("tcq2".to_owned()))
    );
    assert_eq!(asked("tq-named").as_deref(), Some("q8/tcq2"));
}
