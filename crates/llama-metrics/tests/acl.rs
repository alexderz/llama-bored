//! The CIDR allowlist: parsing, matching, and a denied peer over a real socket.

mod common;

use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use common::{EchoBody, allowlist, exchange, spawn, status_line};
use llama_metrics::acl::{Cidr, CidrError};
use llama_metrics::http::Limits;

fn ip(text: &str) -> IpAddr {
    text.parse().unwrap()
}

#[test]
fn lan_and_loopback_are_allowed_and_everything_else_is_not() {
    let list = allowlist(&["192.168.0.0/22", "127.0.0.1/32"]);
    for peer in [
        "192.168.0.0",
        "192.168.0.1",
        "192.168.2.40",
        "192.168.3.255",
        "127.0.0.1",
    ] {
        assert!(list.permits(ip(peer)), "{peer} should be allowed");
    }
    for peer in [
        "192.168.4.0",
        "192.167.255.255",
        "192.169.0.1",
        "127.0.0.2",
        "172.16.1.10",
        "0.0.0.0",
        "255.255.255.255",
        "::1",
        "fe80::1",
        "2001:db8::1",
    ] {
        assert!(!list.permits(ip(peer)), "{peer} should be denied");
    }
}

#[test]
fn v4_mapped_peers_on_a_dual_stack_socket_match_as_ipv4() {
    let list = allowlist(&["192.168.0.0/22"]);
    assert!(list.permits(ip("::ffff:192.168.1.2")));
    assert!(!list.permits(ip("::ffff:192.168.4.2")));
}

#[test]
fn ipv6_networks_match_by_prefix() {
    let list = allowlist(&["fd00:db8::/64", "::1/128"]);
    assert!(list.permits(ip("fd00:db8::1")));
    assert!(list.permits(ip("fd00:db8::ffff:1")));
    assert!(list.permits(ip("::1")));
    assert!(!list.permits(ip("fd00:db8:0:1::1")));
    assert!(!list.permits(ip("127.0.0.1")));
}

#[test]
fn empty_list_allows_nobody() {
    assert!(!allowlist(&[]).permits(ip("127.0.0.1")));
}

#[test]
fn cidr_parsing_is_strict() {
    assert!("192.168.0.0/22".parse::<Cidr>().is_ok());
    assert!("127.0.0.1/32".parse::<Cidr>().is_ok());
    assert!("fd00::/8".parse::<Cidr>().is_ok());
    for bad in [
        "192.168.0.0",
        "192.168.0.0/",
        "/22",
        "192.168.0.0/x",
        "192.168.0.0/+2",
        "192.168.0.0/0022",
        "host/24",
        "192.168.0.0/22 ",
    ] {
        assert!(
            matches!(bad.parse::<Cidr>(), Err(CidrError::Syntax(_))),
            "{bad} parsed"
        );
    }
    assert!(matches!(
        "::ffff:10.0.0.0/104".parse::<Cidr>(),
        Err(CidrError::Syntax(_))
    ));
    for bad in ["0.0.0.0/0", "::/0", "10.0.0.0/33", "fd00::/129"] {
        assert!(
            matches!(bad.parse::<Cidr>(), Err(CidrError::Prefix(_))),
            "{bad} parsed"
        );
    }
    for bad in ["192.168.0.5/22", "127.0.0.1/8", "fd00::1/64"] {
        assert!(
            matches!(bad.parse::<Cidr>(), Err(CidrError::HostBits(_))),
            "{bad} parsed"
        );
    }
}

fn limits() -> Limits {
    Limits {
        header_timeout: Duration::from_millis(500),
        ..Limits::default()
    }
}

#[test]
fn allowed_peer_is_served() {
    let h = spawn(
        &["192.168.0.0/22", "127.0.0.1/32"],
        2,
        limits(),
        Arc::new(EchoBody),
    );
    let resp = exchange(h.addr, b"GET /metrics HTTP/1.1\r\n\r\n");
    assert_eq!(status_line(&resp), "HTTP/1.1 200 OK");
    assert_eq!(h.stats.rejected().denied, 0);
}

/// Loopback is not in this list, so the test client is a foreign peer: the
/// server closes without a status line and counts the denial.
#[test]
fn denied_peer_gets_no_response_at_all() {
    let h = spawn(&["192.168.0.0/22"], 2, limits(), Arc::new(EchoBody));
    for _ in 0..3 {
        let resp = exchange(h.addr, b"GET /metrics HTTP/1.1\r\n\r\n");
        assert_eq!(resp, "", "a denied peer got bytes back");
    }
    common::wait_until("denials counted", Duration::from_secs(2), || {
        h.stats.rejected().denied == 3
    });
    assert_eq!(h.stats.active(), 0, "a denied peer took a worker");
}
