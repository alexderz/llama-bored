//! SSDP: M-SEARCH parsing, answers and NOTIFY text (goldens), the UDN, and
//! the discovery loop's allowlist and rate cap.

mod common;

use std::net::{SocketAddr, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use common::{BASE, UDN, allowlist};
use llama_cast::discovery::{self, Advert, MAX_REPLIES_PER_S, ReplyBudget};
use llama_cast::ssdp;

const V: &str = env!("CARGO_PKG_VERSION");

fn msearch(st: &str) -> Vec<u8> {
    format!(
        "M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\nMX: 2\r\nST: {st}\r\nUSER-AGENT: Roku/DVP-14.0 UPnP/1.0\r\n\r\n"
    )
    .into_bytes()
}

fn advert(allow: &[&str]) -> Advert {
    Advert {
        udn: UDN.to_owned(),
        base_url: BASE.to_owned(),
        allow: allowlist(allow),
    }
}

fn peer(text: &str) -> SocketAddr {
    text.parse().unwrap()
}

#[test]
fn msearch_parsing() {
    assert_eq!(
        ssdp::parse_msearch(&msearch("ssdp:all")).as_deref(),
        Some("ssdp:all")
    );
    // Header names are case-insensitive; MAN may be unquoted.
    assert_eq!(
        ssdp::parse_msearch(
            b"M-SEARCH * HTTP/1.1\r\nman: ssdp:discover\r\nst: upnp:rootdevice\r\n\r\n"
        )
        .as_deref(),
        Some("upnp:rootdevice")
    );
    for bad in [
        &b"NOTIFY * HTTP/1.1\r\nNT: upnp:rootdevice\r\nNTS: ssdp:alive\r\n\r\n"[..],
        b"M-SEARCH * HTTP/1.1\r\nST: ssdp:all\r\n\r\n",
        b"M-SEARCH * HTTP/1.1\r\nMAN: \"ssdp:discover\"\r\n\r\n",
        b"M-SEARCH * HTTP/1.1\r\nMAN: \"ssdp:discover\"\r\nST: \r\n\r\n",
        b"M-SEARCH / HTTP/1.1\r\nMAN: \"ssdp:discover\"\r\nST: ssdp:all\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nST: ssdp:all\r\n\r\n",
        b"M-SEARCH * HTTP/1.1\r\nMAN \"ssdp:discover\"\r\nST: ssdp:all\r\n\r\n",
        b"\xff\xfe",
        b"",
    ] {
        assert_eq!(
            ssdp::parse_msearch(bad),
            None,
            "{:?}",
            String::from_utf8_lossy(bad)
        );
    }
    let mut big = msearch("ssdp:all");
    big.extend(std::iter::repeat_n(b'x', ssdp::MAX_DATAGRAM));
    assert_eq!(ssdp::parse_msearch(&big), None);
}

#[test]
fn search_targets() {
    let all = ssdp::answer_targets("ssdp:all", UDN);
    assert_eq!(
        all,
        [
            "upnp:rootdevice",
            UDN,
            "urn:schemas-upnp-org:device:MediaServer:1",
            "urn:schemas-upnp-org:service:ContentDirectory:1",
        ]
    );
    for one in &all {
        assert_eq!(ssdp::answer_targets(one, UDN), std::slice::from_ref(one));
    }
    assert!(ssdp::answer_targets("urn:schemas-upnp-org:device:MediaRenderer:1", UDN).is_empty());
    assert!(ssdp::answer_targets("uuid:someone-else", UDN).is_empty());
}

#[test]
fn search_reply_golden() {
    assert_eq!(
        ssdp::search_reply(BASE, UDN, "urn:schemas-upnp-org:device:MediaServer:1"),
        format!(
            "HTTP/1.1 200 OK\r\n\
             CACHE-CONTROL: max-age=1800\r\n\
             EXT:\r\n\
             LOCATION: http://192.168.1.20:19478/desc.xml\r\n\
             SERVER: Linux/1 UPnP/1.0 llama-bored/{V}\r\n\
             ST: urn:schemas-upnp-org:device:MediaServer:1\r\n\
             USN: uuid:5a1e0c2d-7b3f-5e41-9c8d-2f6a4b1e0d93::urn:schemas-upnp-org:device:MediaServer:1\r\n\
             \r\n"
        )
    );
    // The UDN target's USN is the bare UDN.
    assert!(ssdp::search_reply(BASE, UDN, UDN).contains(&format!("\r\nUSN: {UDN}\r\n")));
}

#[test]
fn notify_goldens() {
    assert_eq!(
        ssdp::notify_alive(BASE, UDN, "upnp:rootdevice"),
        format!(
            "NOTIFY * HTTP/1.1\r\n\
             HOST: 239.255.255.250:1900\r\n\
             CACHE-CONTROL: max-age=1800\r\n\
             LOCATION: http://192.168.1.20:19478/desc.xml\r\n\
             NT: upnp:rootdevice\r\n\
             NTS: ssdp:alive\r\n\
             SERVER: Linux/1 UPnP/1.0 llama-bored/{V}\r\n\
             USN: uuid:5a1e0c2d-7b3f-5e41-9c8d-2f6a4b1e0d93::upnp:rootdevice\r\n\
             \r\n"
        )
    );
    assert_eq!(
        ssdp::notify_byebye(UDN, UDN),
        "NOTIFY * HTTP/1.1\r\n\
         HOST: 239.255.255.250:1900\r\n\
         NT: uuid:5a1e0c2d-7b3f-5e41-9c8d-2f6a4b1e0d93\r\n\
         NTS: ssdp:byebye\r\n\
         USN: uuid:5a1e0c2d-7b3f-5e41-9c8d-2f6a4b1e0d93\r\n\
         \r\n"
    );
}

#[test]
fn udn_is_a_stable_hash_not_the_machine_id() {
    let id = "0123456789abcdef0123456789abcdef";
    let udn = ssdp::udn_from_machine_id(&format!("{id}\n"));
    assert_eq!(udn, ssdp::udn_from_machine_id(id));
    assert_eq!(udn.len(), "uuid:".len() + 36);
    assert!(udn.starts_with("uuid:"));
    assert!(!udn.replace('-', "").contains(&id[..8]));
    // Version 5 and RFC 4122 variant bits.
    let bytes = udn.as_bytes();
    assert_eq!(bytes[5 + 14], b'5');
    assert!(matches!(bytes[5 + 19], b'8' | b'9' | b'a' | b'b'));
    assert_ne!(
        udn,
        ssdp::udn_from_machine_id("fedcba9876543210fedcba9876543210")
    );
    // Pinned, so an accidental change does not re-identify every TV's entry.
    assert_eq!(udn, "uuid:52d61133-a0c2-526c-a08c-368b1bb40845");
    assert!(ssdp::valid_machine_id(&format!("{id}\n")));
    for bad in [
        "",
        "0123",
        &id.to_uppercase(),
        &format!("{id}0"),
        "../../etc/shadow",
    ] {
        assert!(!ssdp::valid_machine_id(bad), "{bad}");
    }
}

#[test]
fn sha256_known_vectors() {
    use llama_cast::sha256::{digest, hex};
    assert_eq!(
        hex(&digest(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(&digest(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        hex(&digest(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    assert_eq!(
        hex(&digest(&[b'a'; 1000])),
        "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
    );
}

#[test]
fn answers_follow_the_allowlist() {
    let adv = advert(&["192.168.1.0/24"]);
    let got = discovery::answers(&adv, &msearch("ssdp:all"), peer("192.168.1.50:50000"));
    assert_eq!(got.len(), 4);
    assert!(got[0].contains("ST: upnp:rootdevice\r\n"));
    // Outside the allowlist: nothing.
    assert!(discovery::answers(&adv, &msearch("ssdp:all"), peer("10.0.0.5:50000")).is_empty());
    // Port 0 cannot be answered.
    assert!(discovery::answers(&adv, &msearch("ssdp:all"), peer("192.168.1.50:0")).is_empty());
    // A NOTIFY from a neighbour is not answered.
    let notify = ssdp::notify_alive("http://192.168.1.9:80", "uuid:x", "upnp:rootdevice");
    assert!(discovery::answers(&adv, notify.as_bytes(), peer("192.168.1.9:1900")).is_empty());
}

#[test]
fn reply_budget_caps_a_second() {
    let start = Instant::now();
    let mut budget = ReplyBudget::new(start);
    for _ in 0..MAX_REPLIES_PER_S {
        assert!(budget.take(start));
    }
    assert!(!budget.take(start + Duration::from_millis(900)));
    assert!(budget.take(start + Duration::from_millis(1000)));
}

#[test]
fn discovery_loop_answers_allowed_peers_on_loopback() {
    let server = UdpSocket::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let adv = advert(&["127.0.0.1/32"]);
    let join = {
        let stop = Arc::clone(&stop);
        thread::spawn(move || discovery::run(&server, &adv, &stop))
    };
    let client = UdpSocket::bind("127.0.0.1:0").unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    client
        .send_to(
            &msearch("urn:schemas-upnp-org:service:ContentDirectory:1"),
            addr,
        )
        .unwrap();
    let mut buf = [0_u8; 2048];
    let (n, from) = client.recv_from(&mut buf).expect("an answer");
    assert_eq!(from, addr);
    let text = String::from_utf8_lossy(&buf[..n]);
    assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
    assert!(text.contains("ST: urn:schemas-upnp-org:service:ContentDirectory:1\r\n"));
    // Junk is dropped, and the loop keeps answering.
    client.send_to(b"GET / HTTP/1.1\r\n\r\n", addr).unwrap();
    client.send_to(&msearch("upnp:rootdevice"), addr).unwrap();
    let (n, _) = client.recv_from(&mut buf).expect("a second answer");
    assert!(String::from_utf8_lossy(&buf[..n]).contains("ST: upnp:rootdevice\r\n"));
    stop.store(true, Ordering::SeqCst);
    join.join().unwrap().expect("loop ends cleanly");
}

#[test]
fn discovery_loop_ignores_peers_outside_the_allowlist() {
    let server = UdpSocket::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let adv = advert(&["192.168.1.0/24"]);
    let join = {
        let stop = Arc::clone(&stop);
        thread::spawn(move || discovery::run(&server, &adv, &stop))
    };
    let client = UdpSocket::bind("127.0.0.1:0").unwrap();
    client
        .set_read_timeout(Some(Duration::from_millis(400)))
        .unwrap();
    client.send_to(&msearch("ssdp:all"), addr).unwrap();
    let mut buf = [0_u8; 2048];
    assert!(
        client.recv_from(&mut buf).is_err(),
        "answered a denied peer"
    );
    stop.store(true, Ordering::SeqCst);
    join.join().unwrap().unwrap();
}

/// #21: byebye on stop is repeated, as datagrams get lost.
#[test]
fn byebye_is_sent_more_than_once() {
    const { assert!(discovery::BYEBYE_ROUNDS >= 2) };
    assert!(
        discovery::BYEBYE_GAP < Duration::from_secs(1),
        "fits TimeoutStopSec"
    );
}
