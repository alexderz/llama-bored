//! Routes, request limits, the allowlist and the stream cap, over a real
//! socket on 127.0.0.1:0 with a fake stream; and the head parser alone.

mod common;

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

use common::{FakeLive, body, exchange, fast_limits, fixture, spawn, status_line, wait_until};
use llama_cast::dlna::PROTOCOL_INFO;
use llama_cast::http::{Limits, Method, Refusal, Route, parse_head, subscription_id};

const LOCAL: &[&str] = &["127.0.0.1/32"];

fn soap(path: &str, action: &str, payload: &str) -> Vec<u8> {
    format!(
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: text/xml; charset=\"utf-8\"\r\nSOAPACTION: \"{action}\"\r\nContent-Length: {}\r\n\r\n{payload}",
        payload.len()
    )
    .into_bytes()
}

const BROWSE: &str = "<?xml version=\"1.0\"?><s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\"><s:Body><u:Browse xmlns:u=\"urn:schemas-upnp-org:service:ContentDirectory:1\"><ObjectID>0</ObjectID><BrowseFlag>BrowseDirectChildren</BrowseFlag><Filter>*</Filter><StartingIndex>0</StartingIndex><RequestedCount>0</RequestedCount><SortCriteria></SortCriteria></u:Browse></s:Body></s:Envelope>";

#[test]
fn description_and_scpds_are_served() {
    let h = spawn(LOCAL, 2, fast_limits(), FakeLive::new());
    let resp = exchange(h.addr, b"GET /desc.xml HTTP/1.1\r\nHost: x\r\n\r\n");
    assert_eq!(status_line(&resp), "HTTP/1.1 200 OK", "{resp}");
    assert!(resp.contains("Content-Type: text/xml; charset=\"utf-8\"\r\n"));
    assert!(resp.contains("Connection: close\r\n"));
    assert_eq!(
        body(&resp),
        std::fs::read_to_string(fixture("desc.xml")).unwrap()
    );
    for path in ["/cds.xml", "/cms.xml"] {
        let resp = exchange(h.addr, format!("GET {path} HTTP/1.0\r\n\r\n").as_bytes());
        assert_eq!(status_line(&resp), "HTTP/1.1 200 OK", "{path}");
        assert!(body(&resp).contains("<scpd xmlns=\"urn:schemas-upnp-org:service-1-0\">"));
    }
    // HEAD: same head, no body.
    let resp = exchange(h.addr, b"HEAD /desc.xml HTTP/1.1\r\n\r\n");
    assert_eq!(status_line(&resp), "HTTP/1.1 200 OK");
    assert!(body(&resp).is_empty(), "{resp}");
}

#[test]
fn browse_over_http() {
    let h = spawn(LOCAL, 2, fast_limits(), FakeLive::new());
    let resp = exchange(
        h.addr,
        &soap(
            "/ctl/cds",
            "urn:schemas-upnp-org:service:ContentDirectory:1#Browse",
            BROWSE,
        ),
    );
    assert_eq!(status_line(&resp), "HTTP/1.1 200 OK", "{resp}");
    assert_eq!(
        body(&resp),
        std::fs::read_to_string(fixture("browse-children.xml")).unwrap()
    );
    let resp = exchange(
        h.addr,
        &soap(
            "/ctl/cms",
            "urn:schemas-upnp-org:service:ConnectionManager:1#GetProtocolInfo",
            "",
        ),
    );
    assert_eq!(status_line(&resp), "HTTP/1.1 200 OK", "{resp}");
    assert!(body(&resp).contains(PROTOCOL_INFO));
    // Wrong service: a UPnP fault.
    let resp = exchange(
        h.addr,
        &soap(
            "/ctl/cms",
            "urn:schemas-upnp-org:service:ContentDirectory:1#Browse",
            BROWSE,
        ),
    );
    assert_eq!(status_line(&resp), "HTTP/1.1 500 Internal Server Error");
    assert!(body(&resp).contains("<errorCode>401</errorCode>"));
}

#[test]
fn a_body_split_across_writes_and_expect_continue() {
    let h = spawn(LOCAL, 2, fast_limits(), FakeLive::new());
    let mut stream = TcpStream::connect(h.addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let head = format!(
        "POST /ctl/cds HTTP/1.1\r\nSOAPACTION: \"urn:schemas-upnp-org:service:ContentDirectory:1#Browse\"\r\nExpect: 100-continue\r\nContent-Length: {}\r\n\r\n",
        BROWSE.len()
    );
    stream.write_all(head.as_bytes()).unwrap();
    let mut cont = [0_u8; 25];
    stream.read_exact(&mut cont).unwrap();
    assert_eq!(&cont, b"HTTP/1.1 100 Continue\r\n\r\n");
    let (a, b) = BROWSE.split_at(40);
    stream.write_all(a.as_bytes()).unwrap();
    thread::sleep(Duration::from_millis(50));
    stream.write_all(b.as_bytes()).unwrap();
    let resp = common::read_all(&mut stream);
    assert_eq!(status_line(&resp), "HTTP/1.1 200 OK", "{resp}");
}

#[test]
fn post_body_rules() {
    let h = spawn(LOCAL, 2, fast_limits(), FakeLive::new());
    let action = "SOAPACTION: \"urn:schemas-upnp-org:service:ContentDirectory:1#Browse\"\r\n";
    for (req, want) in [
        (
            format!("POST /ctl/cds HTTP/1.1\r\n{action}\r\n"),
            "411 Length Required",
        ),
        (
            format!("POST /ctl/cds HTTP/1.1\r\n{action}Content-Length: 99999\r\n\r\n"),
            "413 Content Too Large",
        ),
        (
            format!("POST /ctl/cds HTTP/1.1\r\n{action}Content-Length: -1\r\n\r\n"),
            "400 Bad Request",
        ),
        (
            format!(
                "POST /ctl/cds HTTP/1.1\r\n{action}Content-Length: 3\r\nContent-Length: 3\r\n\r\nabc"
            ),
            "400 Bad Request",
        ),
        (
            format!("POST /ctl/cds HTTP/1.1\r\n{action}Transfer-Encoding: chunked\r\n\r\n"),
            "501 Not Implemented",
        ),
    ] {
        let resp = exchange(h.addr, req.as_bytes());
        assert_eq!(status_line(&resp), format!("HTTP/1.1 {want}"), "{req:?}");
    }
    // A body that is not UTF-8.
    let mut req =
        format!("POST /ctl/cds HTTP/1.1\r\n{action}Content-Length: 2\r\n\r\n").into_bytes();
    req.extend_from_slice(&[0xff, 0xfe]);
    assert_eq!(
        status_line(&exchange(h.addr, &req)),
        "HTTP/1.1 400 Bad Request"
    );
    // A body that never arrives times out.
    let started = Instant::now();
    let resp = exchange(
        h.addr,
        format!("POST /ctl/cds HTTP/1.1\r\n{action}Content-Length: 100\r\n\r\nabc").as_bytes(),
    );
    assert_eq!(status_line(&resp), "HTTP/1.1 408 Request Timeout");
    assert!(started.elapsed() < Duration::from_secs(3));
}

#[test]
fn subscribe_gets_a_sid_and_no_callback() {
    let h = spawn(LOCAL, 2, fast_limits(), FakeLive::new());
    let resp = exchange(
        h.addr,
        b"SUBSCRIBE /evt/cds HTTP/1.1\r\nCALLBACK: <http://127.0.0.1:9/>\r\nNT: upnp:event\r\nTIMEOUT: Second-300\r\n\r\n",
    );
    assert_eq!(status_line(&resp), "HTTP/1.1 200 OK", "{resp}");
    assert!(
        resp.contains("\r\nSID: uuid:5a1e0c2d-7b3f-5e41-9c8d-000000000001\r\n"),
        "{resp}"
    );
    assert!(resp.contains("\r\nTIMEOUT: Second-1800\r\n"));
    assert!(resp.contains("\r\nContent-Length: 0\r\n"));
    let second = exchange(h.addr, b"SUBSCRIBE /evt/cms HTTP/1.1\r\n\r\n");
    assert!(second.contains("SID: uuid:5a1e0c2d-7b3f-5e41-9c8d-000000000002\r\n"));
    assert_eq!(
        subscription_id("uuid:5a1e0c2d-7b3f-5e41-9c8d-2f6a4b1e0d93", 255),
        "uuid:5a1e0c2d-7b3f-5e41-9c8d-0000000000ff"
    );
}

#[test]
fn stream_head_and_body() {
    let h = spawn(LOCAL, 2, fast_limits(), FakeLive::new());
    let resp = exchange(h.addr, b"HEAD /live.ts HTTP/1.1\r\n\r\n");
    assert_eq!(status_line(&resp), "HTTP/1.1 200 OK");
    assert!(resp.contains("\r\nContent-Type: video/mpeg\r\n"), "{resp}");
    assert!(resp.contains("\r\ntransferMode.dlna.org: Streaming\r\n"));
    assert!(!resp.contains("Content-Length"));
    assert!(body(&resp).is_empty());
    assert_eq!(
        h.opened.load(Ordering::SeqCst),
        0,
        "HEAD started an encoder"
    );

    let mut stream = TcpStream::connect(h.addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    stream
        .write_all(b"GET /live.ts HTTP/1.1\r\nHost: x\r\n\r\n")
        .unwrap();
    let mut got = Vec::new();
    let mut buf = [0_u8; 4096];
    while got.len() < 2000 {
        let n = stream.read(&mut buf).unwrap();
        assert!(n > 0, "stream closed early");
        got.extend_from_slice(&buf[..n]);
    }
    let text = String::from_utf8_lossy(&got);
    assert!(
        text.starts_with(
            "HTTP/1.1 200 OK\r\nContent-Type: video/mpeg\r\ntransferMode.dlna.org: Streaming\r\n"
        ),
        "{text}"
    );
    let at = got.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    assert!(got[at..].iter().all(|b| *b == 0x47));
    assert_eq!(h.stats.streams(), 1);
    // The client goes away: the stream (the encoder) is dropped.
    drop(stream);
    wait_until("stream dropped", Duration::from_secs(5), || {
        h.closed.load(Ordering::SeqCst) == 1 && h.stats.streams() == 0
    });
}

#[test]
fn streams_are_capped_at_max_clients() {
    let h = spawn(LOCAL, 2, fast_limits(), FakeLive::new());
    let open = || {
        let mut s = TcpStream::connect(h.addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        s.write_all(b"GET /live.ts HTTP/1.1\r\n\r\n").unwrap();
        let mut buf = [0_u8; 64];
        let n = s.read(&mut buf).unwrap();
        (s, String::from_utf8_lossy(&buf[..n]).into_owned())
    };
    let (a, ra) = open();
    let (b, rb) = open();
    assert!(ra.starts_with("HTTP/1.1 200 OK"), "{ra}");
    assert!(rb.starts_with("HTTP/1.1 200 OK"), "{rb}");
    wait_until("two streams", Duration::from_secs(5), || {
        h.stats.streams() == 2
    });
    let third = exchange(h.addr, b"GET /live.ts HTTP/1.1\r\n\r\n");
    assert_eq!(status_line(&third), "HTTP/1.1 503 Service Unavailable");
    assert_eq!(
        h.opened.load(Ordering::SeqCst),
        2,
        "a third encoder started"
    );
    // Descriptions still work while both streams run.
    let desc = exchange(h.addr, b"GET /desc.xml HTTP/1.1\r\n\r\n");
    assert_eq!(status_line(&desc), "HTTP/1.1 200 OK");
    drop(a);
    wait_until("slot freed", Duration::from_secs(5), || {
        h.stats.streams() == 1
    });
    let (_c, rc) = open();
    assert!(rc.starts_with("HTTP/1.1 200 OK"), "{rc}");
    drop(b);
}

#[test]
fn encoder_failure_is_500() {
    let mut live = FakeLive::new();
    live.fail = true;
    let h = spawn(LOCAL, 2, fast_limits(), live);
    let resp = exchange(h.addr, b"GET /live.ts HTTP/1.1\r\n\r\n");
    assert_eq!(status_line(&resp), "HTTP/1.1 500 Internal Server Error");
    assert_eq!(h.stats.streams(), 0);
}

#[test]
fn a_peer_outside_the_allowlist_is_closed_unread() {
    let h = spawn(&["10.0.0.0/8"], 2, fast_limits(), FakeLive::new());
    let resp = exchange(h.addr, b"GET /desc.xml HTTP/1.1\r\n\r\n");
    assert!(resp.is_empty(), "{resp}");
    wait_until("denied counted", Duration::from_secs(2), || {
        h.stats.denied() == 1
    });
    let resp = exchange(h.addr, b"GET /live.ts HTTP/1.1\r\n\r\n");
    assert!(resp.is_empty());
    assert_eq!(h.opened.load(Ordering::SeqCst), 0);
}

#[test]
fn methods_and_paths() {
    let h = spawn(LOCAL, 2, fast_limits(), FakeLive::new());
    for (req, want, allow) in [
        (
            "PUT /desc.xml",
            "405 Method Not Allowed",
            Some("GET, HEAD, POST, SUBSCRIBE"),
        ),
        (
            "DELETE /live.ts",
            "405 Method Not Allowed",
            Some("GET, HEAD, POST, SUBSCRIBE"),
        ),
        (
            "UNSUBSCRIBE /evt/cds",
            "405 Method Not Allowed",
            Some("GET, HEAD, POST, SUBSCRIBE"),
        ),
        (
            "POST /desc.xml",
            "405 Method Not Allowed",
            Some("GET, HEAD"),
        ),
        ("GET /ctl/cds", "405 Method Not Allowed", Some("POST")),
        ("GET /evt/cds", "405 Method Not Allowed", Some("SUBSCRIBE")),
        (
            "SUBSCRIBE /live.ts",
            "405 Method Not Allowed",
            Some("GET, HEAD"),
        ),
        ("GET /", "404 Not Found", None),
        ("GET /live1.ts", "404 Not Found", None),
        ("GET /desc.xml?x=1", "404 Not Found", None),
        ("GET /../etc/passwd", "404 Not Found", None),
        ("GET /metrics", "404 Not Found", None),
    ] {
        let resp = exchange(h.addr, format!("{req} HTTP/1.1\r\n\r\n").as_bytes());
        assert_eq!(status_line(&resp), format!("HTTP/1.1 {want}"), "{req}");
        match allow {
            Some(allow) => assert!(
                resp.contains(&format!("\r\nAllow: {allow}\r\n")),
                "{req}: {resp}"
            ),
            None => assert!(!resp.contains("Allow:"), "{req}"),
        }
    }
    assert_eq!(h.opened.load(Ordering::SeqCst), 0);
}

#[test]
fn head_limits() {
    let h = spawn(LOCAL, 2, fast_limits(), FakeLive::new());
    let long = format!("GET /{} HTTP/1.1\r\n\r\n", "a".repeat(2000));
    assert_eq!(
        status_line(&exchange(h.addr, long.as_bytes())),
        "HTTP/1.1 414 URI Too Long"
    );
    let many: String = (0..40).map(|i| format!("X-{i}: y\r\n")).collect();
    assert_eq!(
        status_line(&exchange(
            h.addr,
            format!("GET /desc.xml HTTP/1.1\r\n{many}\r\n").as_bytes()
        )),
        "HTTP/1.1 431 Request Header Fields Too Large"
    );
    let big = format!("GET /desc.xml HTTP/1.1\r\nX: {}\r\n\r\n", "b".repeat(9000));
    assert_eq!(
        status_line(&exchange(h.addr, big.as_bytes())),
        "HTTP/1.1 431 Request Header Fields Too Large"
    );
    assert_eq!(
        status_line(&exchange(h.addr, b"GET /desc.xml HTTP/2.0\r\n\r\n")),
        "HTTP/1.1 505 HTTP Version Not Supported"
    );
    // Slowloris: a head that never ends.
    let started = Instant::now();
    let resp = exchange(h.addr, b"GET /desc.xml HTTP/1.1\r\nX: ");
    assert_eq!(status_line(&resp), "HTTP/1.1 408 Request Timeout");
    assert!(started.elapsed() < Duration::from_secs(3));
}

#[test]
fn workers_are_capped_and_the_watchdog_ticks() {
    // 1 stream slot + 4 control workers; hold them all with idle peers.
    let h = spawn(
        LOCAL,
        1,
        Limits {
            header_timeout: Duration::from_secs(3),
            ..fast_limits()
        },
        FakeLive::new(),
    );
    let idle: Vec<TcpStream> = (0..5)
        .map(|_| TcpStream::connect(h.addr).unwrap())
        .collect();
    wait_until("five active", Duration::from_secs(2), || {
        h.stats.active() == 5
    });
    let resp = exchange(h.addr, b"GET /desc.xml HTTP/1.1\r\n\r\n");
    assert_eq!(status_line(&resp), "HTTP/1.1 503 Service Unavailable");
    assert!(h.stats.busy() >= 1);
    drop(idle);
    let ticks = h.ticks.load(Ordering::SeqCst);
    wait_until("watchdog ticks", Duration::from_secs(2), || {
        h.ticks.load(Ordering::SeqCst) > ticks
    });
}

#[test]
fn parse_head_unit() {
    let l = Limits::default();
    let req = parse_head(b"GET /live.ts HTTP/1.1\r\nHost: a\r\nUser-Agent: Roku", &l).unwrap();
    assert_eq!(req.method, Method::Get);
    assert_eq!(req.route, Route::Stream);
    assert_eq!(req.header("user-agent"), Some("Roku"));
    for (head, want) in [
        (&b"GET  /live.ts HTTP/1.1"[..], Refusal::BadRequest),
        (b"get /live.ts HTTP/1.1", Refusal::BadRequest),
        (b"GET /live.ts HTTP/1.1\r\nBad Name: x", Refusal::BadRequest),
        (b"GET /live.ts HTTP/1.1\r\n folded", Refusal::BadRequest),
        (b"GET /live.ts HTTP/1.1\r\nX: a\rb", Refusal::BadRequest),
        (b"GET /live.ts FTP/1.0", Refusal::BadRequest),
    ] {
        assert_eq!(
            parse_head(head, &l).map_err(|r| r.refusal),
            Err(want),
            "{:?}",
            String::from_utf8_lossy(head)
        );
    }
}
