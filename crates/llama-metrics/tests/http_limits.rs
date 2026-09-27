//! Request limits and routing over a real socket on 127.0.0.1:0, plus the
//! head parser on its own.

mod common;

use std::io::Write;
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

use common::{EchoBody, exchange, spawn, status_line, wait_until};
use llama_metrics::expo::CONTENT_TYPE;
use llama_metrics::http::{Limits, Refusal, parse_head};

const LOCAL: &[&str] = &["127.0.0.1/32"];

fn fast_limits() -> Limits {
    Limits {
        header_timeout: Duration::from_millis(300),
        write_timeout: Duration::from_millis(500),
        ..Limits::default()
    }
}

#[test]
fn get_metrics_is_served_with_the_exposition_content_type() {
    let h = spawn(LOCAL, 4, fast_limits(), Arc::new(EchoBody));
    let resp = exchange(
        h.addr,
        b"GET /metrics HTTP/1.1\r\nHost: llama-host\r\nUser-Agent: Prometheus/3\r\nAccept: */*\r\n\r\n",
    );
    assert_eq!(status_line(&resp), "HTTP/1.1 200 OK", "{resp}");
    assert!(resp.contains(&format!("Content-Type: {CONTENT_TYPE}\r\n")));
    assert!(resp.contains("Connection: close\r\n"));
    assert!(resp.ends_with("busy 0 denied 0\n"), "{resp}");
    // HTTP/1.0 is fine too.
    let resp = exchange(h.addr, b"GET /metrics HTTP/1.0\r\n\r\n");
    assert_eq!(status_line(&resp), "HTTP/1.1 200 OK");
}

#[test]
fn bad_method_is_405_with_allow() {
    let h = spawn(LOCAL, 4, fast_limits(), Arc::new(EchoBody));
    for method in [
        "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT",
    ] {
        let req = format!("{method} /metrics HTTP/1.1\r\nHost: x\r\n\r\n");
        let resp = exchange(h.addr, req.as_bytes());
        assert_eq!(
            status_line(&resp),
            "HTTP/1.1 405 Method Not Allowed",
            "{method}"
        );
        assert!(resp.contains("Allow: GET\r\n"), "{resp}");
        assert!(!resp.contains("busy"), "{method} reached the body");
    }
}

#[test]
fn bad_path_is_404() {
    let h = spawn(LOCAL, 4, fast_limits(), Arc::new(EchoBody));
    for path in [
        "/",
        "/metrics/",
        "/metrics?x=1",
        "/Metrics",
        "/../metrics",
        "//metrics",
        "/run/llama-watch/snapshot.json",
        "http://llama-host/metrics",
        "*",
    ] {
        let req = format!("GET {path} HTTP/1.1\r\n\r\n");
        let resp = exchange(h.addr, req.as_bytes());
        assert_eq!(status_line(&resp), "HTTP/1.1 404 Not Found", "{path}");
        assert!(!resp.contains("busy"));
    }
}

#[test]
fn malformed_requests_are_400_or_505() {
    let h = spawn(LOCAL, 4, fast_limits(), Arc::new(EchoBody));
    let cases: &[(&[u8], &str)] = &[
        (b"GET /metrics\r\n\r\n", "400"),
        (b"GET  /metrics HTTP/1.1\r\n\r\n", "400"),
        (b"get /metrics HTTP/1.1\r\n\r\n", "400"),
        (b"GET /metrics HTTP/2.0\r\n\r\n", "505"),
        (b"GET /metrics FTP/1.0\r\n\r\n", "400"),
        (b"GET /metrics HTTP/1.1\r\nno-colon\r\n\r\n", "400"),
        (b"GET /metrics HTTP/1.1\r\nBad Name: x\r\n\r\n", "400"),
        (b"GET /metrics HTTP/1.1\r\n folded: x\r\n\r\n", "400"),
        (b"GET /metrics HTTP/1.1\nHost: x\r\n\r\n", "400"),
        (b"GET /metrics HTTP/1.1\r\nX: a\x01b\r\n\r\n", "400"),
        (
            b"\x16\x03\x01\x02\x00\x01\x00\x01\xfc\x03\x03\r\n\r\n",
            "400",
        ),
    ];
    for (req, want) in cases {
        let resp = exchange(h.addr, req);
        assert!(
            status_line(&resp).starts_with(&format!("HTTP/1.1 {want} ")),
            "{:?} -> {resp:?}",
            String::from_utf8_lossy(req)
        );
    }
}

#[test]
fn oversized_request_line_is_414_and_oversized_head_is_431() {
    let h = spawn(LOCAL, 4, fast_limits(), Arc::new(EchoBody));
    let long_path = format!("GET /{} HTTP/1.1\r\n\r\n", "a".repeat(4000));
    let resp = exchange(h.addr, long_path.as_bytes());
    assert_eq!(status_line(&resp), "HTTP/1.1 414 URI Too Long", "{resp}");

    // No CRLF at all: refused once the line cap is passed, not at EOF.
    let resp = exchange(h.addr, "G".repeat(2000).as_bytes());
    assert_eq!(status_line(&resp), "HTTP/1.1 414 URI Too Long");

    let big = format!(
        "GET /metrics HTTP/1.1\r\nX-Pad: {}\r\n\r\n",
        "b".repeat(9000)
    );
    let resp = exchange(h.addr, big.as_bytes());
    assert_eq!(
        status_line(&resp),
        "HTTP/1.1 431 Request Header Fields Too Large",
        "{resp}"
    );

    let mut many = String::from("GET /metrics HTTP/1.1\r\n");
    for i in 0..33 {
        many.push_str(&format!("X-{i}: y\r\n"));
    }
    many.push_str("\r\n");
    let resp = exchange(h.addr, many.as_bytes());
    assert_eq!(
        status_line(&resp),
        "HTTP/1.1 431 Request Header Fields Too Large"
    );

    // 32 headers is still fine.
    let mut ok = String::from("GET /metrics HTTP/1.1\r\n");
    for i in 0..32 {
        ok.push_str(&format!("X-{i}: y\r\n"));
    }
    ok.push_str("\r\n");
    assert_eq!(
        status_line(&exchange(h.addr, ok.as_bytes())),
        "HTTP/1.1 200 OK"
    );
}

#[test]
fn slowloris_is_cut_off_by_one_deadline_for_the_whole_head() {
    let h = spawn(LOCAL, 4, fast_limits(), Arc::new(EchoBody));
    let mut stream = TcpStream::connect(h.addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut writer = stream.try_clone().unwrap();
    let start = Instant::now();
    // One byte every 40 ms keeps each read short of a per-read timeout.
    let trickle = thread::spawn(move || {
        for byte in b"GET /metrics HTTP/1.1\r\nX-Slow: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        {
            if writer.write_all(&[*byte]).is_err() {
                return;
            }
            thread::sleep(Duration::from_millis(40));
        }
    });
    let resp = common::read_all(&mut stream);
    let elapsed = start.elapsed();
    assert_eq!(
        status_line(&resp),
        "HTTP/1.1 408 Request Timeout",
        "{resp:?}"
    );
    assert!(
        elapsed < Duration::from_millis(1500),
        "the head deadline did not cut the trickle: {elapsed:?}"
    );
    let _ = trickle.join();

    // An idle connection gets the same answer, and frees its worker.
    let resp = exchange(h.addr, b"GET /metr");
    assert_eq!(status_line(&resp), "HTTP/1.1 408 Request Timeout");
    wait_until("workers idle", Duration::from_secs(3), || {
        h.stats.active() == 0
    });
}

#[test]
fn connections_beyond_max_conns_get_503_and_the_pool_recovers() {
    let limits = Limits {
        header_timeout: Duration::from_secs(10),
        ..fast_limits()
    };
    let h = spawn(LOCAL, 2, limits, Arc::new(EchoBody));
    let hold_a = TcpStream::connect(h.addr).unwrap();
    let hold_b = TcpStream::connect(h.addr).unwrap();
    wait_until("two held workers", Duration::from_secs(3), || {
        h.stats.active() == 2
    });
    let resp = exchange(h.addr, b"GET /metrics HTTP/1.1\r\n\r\n");
    assert_eq!(
        status_line(&resp),
        "HTTP/1.1 503 Service Unavailable",
        "{resp:?}"
    );
    let resp = exchange(h.addr, b"GET /metrics HTTP/1.1\r\n\r\n");
    assert_eq!(status_line(&resp), "HTTP/1.1 503 Service Unavailable");
    assert_eq!(h.stats.rejected().busy, 2);
    assert_eq!(h.stats.active(), 2, "a refused connection took a slot");

    drop(hold_a);
    drop(hold_b);
    wait_until("slots released", Duration::from_secs(3), || {
        h.stats.active() == 0
    });
    let resp = exchange(h.addr, b"GET /metrics HTTP/1.1\r\n\r\n");
    assert_eq!(status_line(&resp), "HTTP/1.1 200 OK");
    assert!(resp.ends_with("busy 2 denied 0\n"), "{resp}");
}

#[test]
fn many_parallel_scrapes_all_complete() {
    let h = spawn(LOCAL, 8, fast_limits(), Arc::new(EchoBody));
    let addr = h.addr;
    let threads: Vec<_> = (0..8)
        .map(|_| thread::spawn(move || exchange(addr, b"GET /metrics HTTP/1.1\r\n\r\n")))
        .collect();
    let mut ok = 0;
    for t in threads {
        let resp = t.join().unwrap();
        let status = status_line(&resp).to_owned();
        assert!(
            status == "HTTP/1.1 200 OK" || status == "HTTP/1.1 503 Service Unavailable",
            "{status}"
        );
        ok += usize::from(status == "HTTP/1.1 200 OK");
    }
    assert!(ok >= 1);
}

#[test]
fn the_acceptor_pets_the_watchdog_while_idle() {
    let h = spawn(LOCAL, 1, fast_limits(), Arc::new(EchoBody));
    let before = h.ticks.load(Ordering::SeqCst);
    wait_until("watchdog ticks", Duration::from_secs(2), || {
        h.ticks.load(Ordering::SeqCst) >= before + 5
    });
}

#[test]
fn parse_head_rules() {
    let l = Limits::default();
    assert_eq!(parse_head(b"GET /metrics HTTP/1.1\r\nHost: a", &l), Ok(()));
    assert_eq!(parse_head(b"GET /metrics HTTP/1.1", &l), Ok(()));
    assert_eq!(
        parse_head(b"GET /metrics HTTP/1.1\nHost: a", &l),
        Err(Refusal::BadRequest)
    );
    assert_eq!(
        parse_head(b"GET /metrics HTTP/1.1\r", &l),
        Err(Refusal::BadRequest)
    );
    assert_eq!(
        parse_head(b"POST / HTTP/1.1", &l),
        Err(Refusal::MethodNotAllowed)
    );
    assert_eq!(parse_head(b"GET / HTTP/1.1", &l), Err(Refusal::NotFound));
    assert_eq!(
        parse_head(b"GET /metrics HTTP/3", &l),
        Err(Refusal::VersionNotSupported)
    );
    let line = format!("GET /{} HTTP/1.1", "x".repeat(l.max_request_line));
    assert_eq!(parse_head(line.as_bytes(), &l), Err(Refusal::UriTooLong));
    let head = vec![b'a'; l.max_head + 1];
    assert_eq!(parse_head(&head, &l), Err(Refusal::HeadersTooLarge));
}
