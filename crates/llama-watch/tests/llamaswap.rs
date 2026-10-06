//! Fake llama-swap on an ephemeral loopback port. No outbound network.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use llama_watch::sources::llamaswap::{Reading, RunningStatus, new_agent, read_with};

const CANARY_CMD: &str = "CANARY_CMD_9f3a2c7e";
const CANARY_PROXY: &str = "CANARY_PROXY_1b6d4e8a";

static HTTP_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    HTTP_LOCK.lock().unwrap_or_else(|err| err.into_inner())
}

fn assert_canary_absent(reading: &Reading) {
    let blob = format!("{reading:?}");
    assert!(!blob.contains(CANARY_CMD), "cmd canary leaked:\n{blob}");
    assert!(!blob.contains(CANARY_PROXY), "proxy canary leaked:\n{blob}");
}

fn assert_down(reading: &Reading, reason: &'static str) {
    assert_eq!(reading.ai, RunningStatus::Down(reason));
    assert!(reading.models.is_empty());
    assert_canary_absent(reading);
}

fn fetch(url: &str, timeout: Duration, aliases: &HashMap<String, String>) -> Reading {
    read_with(
        &new_agent(),
        url,
        timeout,
        aliases,
        &llama_watch::setup_rules::Rules::builtin(),
        &HashMap::new(),
    )
}

fn fetch_default(port: u16) -> Reading {
    fetch(
        &format!("http://127.0.0.1:{port}"),
        Duration::from_secs(2),
        &HashMap::new(),
    )
}

struct Server {
    port: u16,
    paths: std::sync::Arc<Mutex<Vec<String>>>,
    handle: Option<JoinHandle<()>>,
}

impl Server {
    fn addr(&self) -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], self.port))
    }

    fn paths(&self) -> Vec<String> {
        self.paths
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone()
    }

    fn finish(mut self) {
        if let Some(handle) = self.handle.take() {
            handle.join().expect("fake server panicked");
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if self.handle.is_some() {
            let _ = TcpStream::connect_timeout(&self.addr(), Duration::from_millis(200));
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

fn serve(
    accepts: usize,
    later_budget: Duration,
    mut handler: impl FnMut(&mut TcpStream, &str) + Send + 'static,
) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1:0");
    let port = listener.local_addr().expect("local addr").port();
    let paths = std::sync::Arc::new(Mutex::new(Vec::new()));
    let paths_thread = std::sync::Arc::clone(&paths);
    let handle = thread::spawn(move || {
        for index in 0..accepts {
            let budget = if index == 0 {
                Duration::from_secs(2)
            } else {
                later_budget
            };
            let Some(mut stream) = accept_for(&listener, budget) else {
                break;
            };
            let head = read_headers(&mut stream);
            let path = request_path(&head);
            paths_thread
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .push(path.clone());
            handler(&mut stream, &path);
        }
    });
    Server {
        port,
        paths,
        handle: Some(handle),
    }
}

fn serve_body(status: u16, reason: &'static str, body: Vec<u8>) -> Server {
    serve(1, Duration::ZERO, move |stream, _path| {
        write_response(stream, status, reason, &body, &[]);
    })
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
    headers: &[(&str, &str)],
) {
    let mut head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (name, value) in headers {
        head.push_str(name);
        head.push_str(": ");
        head.push_str(value);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
}

fn write_drip(stream: &mut TcpStream, body: &[u8], gap: Duration) {
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }
    for &byte in body {
        if stream.write_all(&[byte]).is_err() {
            return;
        }
        thread::sleep(gap);
    }
}

fn json_body(value: serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(&value).expect("json")
}

fn running_with(entries: Vec<serde_json::Value>) -> Vec<u8> {
    json_body(serde_json::json!({
        "running": entries,
        "cmd": CANARY_CMD,
        "proxy": CANARY_PROXY,
    }))
}

fn entry(model: Option<&str>, name: Option<&str>, state: Option<&str>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    if let Some(model) = model {
        map.insert(
            "model".to_owned(),
            serde_json::Value::String(model.to_owned()),
        );
    }
    if let Some(name) = name {
        map.insert(
            "name".to_owned(),
            serde_json::Value::String(name.to_owned()),
        );
    }
    if let Some(state) = state {
        map.insert(
            "state".to_owned(),
            serde_json::Value::String(state.to_owned()),
        );
    }
    map.insert(
        "cmd".to_owned(),
        serde_json::Value::String(CANARY_CMD.to_owned()),
    );
    map.insert(
        "proxy".to_owned(),
        serde_json::Value::String(CANARY_PROXY.to_owned()),
    );
    serde_json::Value::Object(map)
}

fn run_proxy_child(port: u16, proxy_port: u16) -> std::process::Output {
    let exe = std::env::current_exe().expect("test binary");
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("env_proxy_is_ignored")
        .arg("--exact")
        .arg("--test-threads=1")
        .env("LLAMA_WATCH_PROXY_CHILD", "1")
        .env("LLAMA_WATCH_PORT", port.to_string())
        .env("http_proxy", format!("http://127.0.0.1:{proxy_port}"))
        .env_remove("HTTP_PROXY")
        .env_remove("https_proxy")
        .env_remove("HTTPS_PROXY")
        .env_remove("all_proxy")
        .env_remove("ALL_PROXY")
        .env_remove("no_proxy")
        .env_remove("NO_PROXY");
    cmd.output().expect("spawn proxy child")
}

#[test]
fn empty_running_is_idle() {
    let _guard = lock();
    let body = running_with(vec![]);
    let server = serve_body(200, "OK", body);
    let reading = fetch(
        &format!("http://127.0.0.1:{}/", server.port),
        Duration::from_secs(2),
        &HashMap::new(),
    );
    assert_eq!(server.paths(), vec!["/running".to_owned()]);
    server.finish();
    assert_eq!(reading.ai, RunningStatus::Idle);
    assert!(reading.models.is_empty());
    assert_canary_absent(&reading);
}

#[test]
fn one_model_uses_name_and_keeps_state() {
    let _guard = lock();
    let body = running_with(vec![entry(
        Some("model-id"),
        Some("DeepSeek"),
        Some("phase-7f3c"),
    )]);
    let server = serve_body(200, "OK", body);
    let reading = fetch_default(server.port);
    assert_eq!(server.paths(), vec!["/running".to_owned()]);
    server.finish();
    assert_eq!(reading.ai, RunningStatus::Loaded);
    assert_eq!(reading.models.len(), 1);
    assert_eq!(reading.models[0].name, "DeepSeek");
    assert_eq!(reading.models[0].state, "phase-7f3c");
    assert_canary_absent(&reading);
}

#[test]
fn three_models_preserve_order() {
    let _guard = lock();
    let mut aliases = HashMap::new();
    aliases.insert("qwen3.6-35b-a3b".to_owned(), "Qwen 35B".to_owned());
    let body = running_with(vec![
        entry(Some("qwen3.6-35b-a3b"), Some("ignored"), Some("ready")),
        entry(Some("other-id"), Some("DeepSeek"), Some("starting")),
        entry(Some("plain-model"), None, Some("stopping")),
    ]);
    let server = serve_body(200, "OK", body);
    let reading = fetch(
        &format!("http://127.0.0.1:{}", server.port),
        Duration::from_secs(2),
        &aliases,
    );
    server.finish();
    assert_eq!(reading.ai, RunningStatus::Loaded);
    assert_eq!(
        reading
            .models
            .iter()
            .map(|model| model.name.as_str())
            .collect::<Vec<_>>(),
        vec!["Qwen 35B", "DeepSeek", "plain-model"]
    );
    assert_eq!(
        reading
            .models
            .iter()
            .map(|model| model.state.as_str())
            .collect::<Vec<_>>(),
        vec!["ready", "starting", "stopping"]
    );
    assert_canary_absent(&reading);
}

#[test]
fn alias_wins_and_is_sanitised() {
    let _guard = lock();
    let mut aliases = HashMap::new();
    aliases.insert("qwen3.6-35b-a3b".to_owned(), "Qwen   35B 🔥".to_owned());
    let body = running_with(vec![entry(
        Some("qwen3.6-35b-a3b"),
        Some("raw-internal-name"),
        Some("ready"),
    )]);
    let server = serve_body(200, "OK", body);
    let reading = fetch(
        &format!("http://127.0.0.1:{}", server.port),
        Duration::from_secs(2),
        &aliases,
    );
    server.finish();
    assert_eq!(reading.models[0].id, "qwen3.6-35b-a3b");
    assert_eq!(reading.models[0].name, "Qwen 35B");
    let blob = format!("{reading:?}");
    assert!(!blob.contains("raw-internal-name"), "{blob}");
    assert_canary_absent(&reading);
}

#[test]
fn long_name_is_truncated_with_ellipsis() {
    let _guard = lock();
    let body = running_with(vec![entry(
        Some("abcdefghijklmnopqrstuvwxyz"),
        None,
        Some("ready"),
    )]);
    let server = serve_body(200, "OK", body);
    let reading = fetch_default(server.port);
    server.finish();
    assert_eq!(reading.ai, RunningStatus::Loaded);
    assert_eq!(reading.models[0].name, "abcdefghijk…");
    assert_eq!(reading.models[0].name.chars().count(), 12);
    assert!(reading.models[0].name.ends_with('\u{2026}'));
    assert_canary_absent(&reading);
}

#[test]
fn unicode_name_keeps_printable_ascii() {
    let _guard = lock();
    let body = running_with(vec![entry(
        Some("unused"),
        Some("Qwen 三五B"),
        Some("ready"),
    )]);
    let server = serve_body(200, "OK", body);
    let reading = fetch_default(server.port);
    server.finish();
    assert_eq!(reading.models[0].name, "Qwen B");
    let blob = format!("{reading:?}");
    assert!(!blob.contains("三五"), "{blob}");
    assert_canary_absent(&reading);
}

#[test]
fn status_500_is_down_even_when_the_body_would_be_idle() {
    let _guard = lock();
    let body = running_with(vec![]);
    let server = serve_body(500, "Internal Server Error", body);
    let reading = fetch_default(server.port);
    server.finish();
    assert_down(&reading, "http status");
}

#[test]
fn redirect_is_not_followed_and_is_down() {
    let _guard = lock();
    // Relative Location stays on this listener. Following it would GET /secret
    // and come back Loaded; a 302 that is not followed stays Down.
    let from_body = running_with(vec![entry(Some("from-body"), None, Some("ready"))]);
    let from_redirect = running_with(vec![entry(Some("from-redirect"), None, Some("ready"))]);
    let server = serve(2, Duration::from_millis(400), move |stream, path| {
        if path.starts_with("/secret") {
            write_response(stream, 200, "OK", &from_redirect, &[]);
        } else {
            write_response(stream, 302, "Found", &from_body, &[("Location", "/secret")]);
        }
    });
    let reading = fetch_default(server.port);
    let paths = server.paths();
    server.finish();
    assert_down(&reading, "http status");
    assert_eq!(paths, vec!["/running".to_owned()], "followed {paths:?}");
    let blob = format!("{reading:?}");
    assert!(
        !blob.contains("from-redirect"),
        "followed the redirect: {blob}"
    );
    assert!(!blob.contains("from-body"), "parsed the 302 body: {blob}");
}

#[test]
fn slow_drip_past_the_deadline_is_down() {
    let _guard = lock();
    let body = json_body(serde_json::json!({
        "cmd": CANARY_CMD,
        "proxy": CANARY_PROXY,
        "running": [],
        "pad": "0123456789abcdef0123456789abcdef",
    }));
    let server = serve(1, Duration::ZERO, move |stream, _path| {
        write_drip(stream, &body, Duration::from_millis(40));
    });
    let started = Instant::now();
    let reading = fetch(
        &format!("http://127.0.0.1:{}", server.port),
        Duration::from_millis(300),
        &HashMap::new(),
    );
    let elapsed = started.elapsed();
    server.finish();
    assert_down(&reading, "timeout");
    assert!(
        elapsed > Duration::from_millis(150),
        "returned too fast: {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_millis(900),
        "waited for the drip instead of the deadline: {elapsed:?}"
    );
    assert_canary_absent(&reading);
}

#[test]
fn refused_connection_is_down() {
    let _guard = lock();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);
    let reading = fetch_default(port);
    assert_down(&reading, "connection refused");
}

fn padded_idle(total: usize) -> Vec<u8> {
    let prefix = br#"{"running":[],"pad":""#;
    let suffix = br#""}"#;
    assert!(total > prefix.len() + suffix.len());
    let mut body = prefix.to_vec();
    body.extend(std::iter::repeat_n(
        b'A',
        total - prefix.len() - suffix.len(),
    ));
    body.extend_from_slice(suffix);
    assert_eq!(body.len(), total);
    body
}

#[test]
fn body_just_under_the_limit_is_idle() {
    let _guard = lock();
    // ureq's limit reader errors once `limit` bytes have been consumed, so
    // 65_535 is the largest body that still parses under `.limit(65_536)`.
    let server = serve_body(200, "OK", padded_idle(65_535));
    let reading = fetch_default(server.port);
    server.finish();
    assert_eq!(reading.ai, RunningStatus::Idle);
}

#[test]
fn oversized_body_is_down() {
    let _guard = lock();
    let mut body =
        format!(r#"{{"cmd":"{CANARY_CMD}","proxy":"{CANARY_PROXY}","running":[],"pad":""#)
            .into_bytes();
    body.extend(std::iter::repeat_n(b'A', 70_000));
    body.extend_from_slice(br#""}"#);
    assert!(body.len() > 65_536);
    let server = serve_body(200, "OK", body);
    let reading = fetch_default(server.port);
    server.finish();
    assert_down(&reading, "oversized body");
}

#[test]
fn malformed_json_is_down() {
    let _guard = lock();
    let body = format!(r#"{{"running":[{CANARY_CMD}"#).into_bytes();
    let server = serve_body(200, "OK", body);
    let reading = fetch_default(server.port);
    server.finish();
    assert_down(&reading, "malformed json");
}

#[test]
fn running_missing_is_down() {
    let _guard = lock();
    let body = json_body(serde_json::json!({
        "cmd": CANARY_CMD,
        "proxy": CANARY_PROXY,
        "models": [],
    }));
    let server = serve_body(200, "OK", body);
    let reading = fetch_default(server.port);
    server.finish();
    assert_down(&reading, "malformed json");
}

#[test]
fn running_wrong_type_is_down() {
    let _guard = lock();
    let bodies = [
        format!(r#"{{"running":{{"cmd":"{CANARY_CMD}","proxy":"{CANARY_PROXY}"}}}}"#),
        format!(r#"{{"running":"{CANARY_CMD}"}}"#),
        format!(r#"{{"running":null,"cmd":"{CANARY_CMD}"}}"#),
    ];
    for body in bodies {
        let server = serve_body(200, "OK", body.into_bytes());
        let reading = fetch_default(server.port);
        server.finish();
        assert_down(&reading, "malformed json");
    }
}

#[test]
fn env_proxy_is_ignored() {
    if std::env::var_os("LLAMA_WATCH_PROXY_CHILD").is_some() {
        let port = std::env::var("LLAMA_WATCH_PORT")
            .expect("port")
            .parse()
            .expect("port number");
        let reading = fetch_default(port);
        assert_eq!(reading.ai, RunningStatus::Idle);
        assert_canary_absent(&reading);
        return;
    }

    let _guard = lock();
    let proxy_listener = TcpListener::bind("127.0.0.1:0").expect("bind closed proxy port");
    let proxy_port = proxy_listener.local_addr().expect("addr").port();
    drop(proxy_listener);
    let server = serve_body(200, "OK", running_with(vec![]));
    let output = run_proxy_child(server.port, proxy_port);
    server.finish();
    assert!(
        output.status.success(),
        "proxy child failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
