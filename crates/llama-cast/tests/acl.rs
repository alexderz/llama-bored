//! The in-process allowlist (same rules as llama-metrics).

use std::net::IpAddr;

use llama_cast::acl::{Allowlist, Cidr, CidrError};

fn ip(text: &str) -> IpAddr {
    text.parse().unwrap()
}

#[test]
fn contains_and_permits() {
    let lan: Cidr = "192.168.1.0/24".parse().unwrap();
    assert!(lan.contains(ip("192.168.1.77")));
    assert!(!lan.contains(ip("192.168.2.1")));
    // A v4-mapped peer from a dual-stack socket is compared as IPv4.
    assert!(lan.contains(ip("::ffff:192.168.1.9")));
    let list = Allowlist::new(vec![lan, "127.0.0.1/32".parse().unwrap()]);
    assert!(list.permits(ip("127.0.0.1")));
    assert!(!list.permits(ip("127.0.0.2")));
    assert!(!list.permits(ip("10.0.0.1")));
    assert!(!Allowlist::default().permits(ip("127.0.0.1")));
}

#[test]
fn the_whole_internet_is_refused() {
    assert_eq!(
        "0.0.0.0/0".parse::<Cidr>(),
        Err(CidrError::Prefix("0.0.0.0/0".into()))
    );
    assert_eq!(
        "::/0".parse::<Cidr>(),
        Err(CidrError::Prefix("::/0".into()))
    );
    assert!(matches!(
        "::ffff:10.0.0.0/104".parse::<Cidr>(),
        Err(CidrError::Syntax(_))
    ));
    assert!(matches!(
        "10.0.0.0/+8".parse::<Cidr>(),
        Err(CidrError::Syntax(_))
    ));
}
