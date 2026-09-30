//! Fake llama-swap on an ephemeral loopback port. No live :8080.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::TryRecvError;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

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
        let action = {
            let mut world = world.lock().unwrap_or_else(|err| err.into_inner());
            reply(&mut world, &path)
        };
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
        let body = if world.running_served == 1 {
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
    assert_eq!(slot.ctx_prompt, Some(9_000));
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

#[test]
fn one_running_read_cannot_request_a_model_the_other_response_marks_ready() {
    let first = serde_json::to_vec(&serde_json::json!({
        "running": [
            {"model": "alpha", "state": "ready"},
            {"model": "beta", "state": "starting"}
        ]
    }))
    .expect("json");
    let second = serde_json::to_vec(&serde_json::json!({
        "running": [
            {"model": "beta", "state": "ready"},
            {"model": "alpha", "state": "starting"}
        ]
    }))
    .expect("json");
    let mut world = World::running(first);
    world.running_after = Some(second);
    world
        .metrics
        .insert("alpha".to_owned(), metrics_body(4, 1.0));
    world
        .metrics
        .insert("beta".to_owned(), metrics_body(4, 1.0));
    world
        .slots
        .insert("alpha".to_owned(), fixture("slots-sample.json"));
    world
        .slots
        .insert("beta".to_owned(), fixture("slots-sample.json"));
    let server = Server::start(world);
    let log = MemLog::new();
    let config = watch(server.port, 12, 4_194_304, 0.15);
    let (_poller, _rx) = spawn(&config, &log);
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(2) {
        if server
            .hits()
            .iter()
            .any(|path| path.contains("/upstream/") && path.ends_with("/metrics"))
        {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    let hits = server.hits();
    let metrics_at = hits
        .iter()
        .position(|path| path.contains("/upstream/") && path.ends_with("/metrics"))
        .expect("a metrics request");
    let early = &hits[..=metrics_at];
    let running_before = early
        .iter()
        .filter(|path| path.ends_with("/running"))
        .count();
    assert_eq!(
        running_before, 1,
        "one /running in the first cycle: {early:?}"
    );
    assert!(
        early.iter().any(|path| path == "/upstream/alpha/metrics"),
        "{early:?}"
    );
    assert!(
        early.iter().all(|path| !path.contains("beta")),
        "non-ready id was requested: {early:?}"
    );
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
sglang:token_usage{{model_name=\"flash\"}} {usage}\n\
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
    world
        .metrics
        .insert("flash".to_owned(), sglang_metrics(1000, 1, 2, 0.37));
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
    assert_eq!(info.kv_permille, Some(370));
    assert_eq!(info.hit_permille, Some(500));
    assert_eq!(info.max_running, Some(4));
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
    assert_eq!(info.kv_permille, Some(250));
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
    let hits = server.hits();
    assert!(
        hits.iter().all(|path| !path.contains("/upstream/")),
        "{hits:?}"
    );
    assert!(log.lines().iter().all(|line| !line.contains("metrics")));
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
    assert_eq!(info.max_running, Some(1));
    assert_eq!(info.kv_permille, None);
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

fn ctx_slot(id_task: i64, prompt: u64) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!([{
        "id": 0,
        "id_task": id_task,
        "is_processing": true,
        "n_ctx": 262_144,
        "n_prompt_tokens": prompt,
        "n_prompt_tokens_processed": prompt,
        "next_token": [{"n_decoded": 0}],
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

/// Every path the watcher may GET from llama-swap (the S17 fence, #5).
fn allowed_path(path: &str) -> bool {
    let upstream = path
        .strip_prefix("/upstream/")
        .and_then(|rest| rest.split_once('/'))
        .is_some_and(|(model, leaf)| {
            !model.is_empty() && !model.contains("..") && matches!(leaf, "metrics" | "slots")
        });
    let capture = path
        .strip_prefix("/api/captures/")
        .is_some_and(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()));
    path == "/running" || path == "/api/metrics/activity" || upstream || capture
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
