//! SSDP message text: M-SEARCH parsing, the unicast answers, and the
//! multicast NOTIFY alive/byebye announcements. No I/O here; the socket
//! loop is `discovery.rs`.

use std::net::Ipv4Addr;

use crate::dlna::{CDS_TYPE, DESC_PATH, DEVICE_TYPE};

/// The SSDP multicast group and port.
pub const GROUP: Ipv4Addr = Ipv4Addr::new(239, 255, 255, 250);
pub const PORT: u16 = 1900;
/// Seconds a control point may cache an announcement.
pub const MAX_AGE_S: u32 = 1800;
/// Seconds between NOTIFY ssdp:alive rounds.
pub const NOTIFY_INTERVAL_S: u64 = 30;
/// Largest datagram looked at; a longer one is dropped.
pub const MAX_DATAGRAM: usize = 2048;

/// The `SERVER:` header.
#[must_use]
pub fn server_header() -> String {
    format!("Linux/1 UPnP/1.0 llama-bored/{}", env!("CARGO_PKG_VERSION"))
}

/// Advertised targets, in announcement order: the root device, the UDN,
/// the device type, the ContentDirectory service.
#[must_use]
pub fn targets(udn: &str) -> Vec<String> {
    vec![
        "upnp:rootdevice".to_owned(),
        udn.to_owned(),
        DEVICE_TYPE.to_owned(),
        CDS_TYPE.to_owned(),
    ]
}

/// USN for a target: the bare UDN for the UDN itself, else `UDN::target`.
#[must_use]
pub fn usn(udn: &str, target: &str) -> String {
    if target == udn {
        udn.to_owned()
    } else {
        format!("{udn}::{target}")
    }
}

/// The search targets an M-SEARCH asked for, answered in order.
/// `ssdp:all` is every target; an unknown target is nothing.
#[must_use]
pub fn answer_targets(st: &str, udn: &str) -> Vec<String> {
    let all = targets(udn);
    if st == "ssdp:all" {
        return all;
    }
    all.into_iter().filter(|t| t == st).collect()
}

/// Parse an M-SEARCH datagram. Returns its `ST` when it is
/// `M-SEARCH * HTTP/1.1` with `MAN: "ssdp:discover"` (quoted or not);
/// anything else (NOTIFY, answers, junk, a missing ST) is `None`.
#[must_use]
pub fn parse_msearch(datagram: &[u8]) -> Option<String> {
    if datagram.len() > MAX_DATAGRAM {
        return None;
    }
    let text = std::str::from_utf8(datagram).ok()?;
    let mut lines = text.split("\r\n");
    if lines.next()? != "M-SEARCH * HTTP/1.1" {
        return None;
    }
    let mut st = None;
    let mut man = false;
    for line in lines {
        if line.is_empty() {
            break;
        }
        let (name, value) = line.split_once(':')?;
        let value = value.trim();
        match name.trim().to_ascii_uppercase().as_str() {
            "ST" => st = Some(value.to_owned()),
            "MAN" => man = value.trim_matches('"') == "ssdp:discover",
            _ => {}
        }
    }
    let st = st?;
    if !man || st.is_empty() || st.len() > 256 || st.bytes().any(|b| b.is_ascii_control()) {
        return None;
    }
    Some(st)
}

/// The unicast answer to an M-SEARCH, for one target.
#[must_use]
pub fn search_reply(base_url: &str, udn: &str, target: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nCACHE-CONTROL: max-age={MAX_AGE_S}\r\nEXT:\r\nLOCATION: {base_url}{DESC_PATH}\r\nSERVER: {}\r\nST: {target}\r\nUSN: {}\r\n\r\n",
        server_header(),
        usn(udn, target),
    )
}

/// NOTIFY ssdp:alive for one target.
#[must_use]
pub fn notify_alive(base_url: &str, udn: &str, target: &str) -> String {
    format!(
        "NOTIFY * HTTP/1.1\r\nHOST: {GROUP}:{PORT}\r\nCACHE-CONTROL: max-age={MAX_AGE_S}\r\nLOCATION: {base_url}{DESC_PATH}\r\nNT: {target}\r\nNTS: ssdp:alive\r\nSERVER: {}\r\nUSN: {}\r\n\r\n",
        server_header(),
        usn(udn, target),
    )
}

/// NOTIFY ssdp:byebye for one target.
#[must_use]
pub fn notify_byebye(udn: &str, target: &str) -> String {
    format!(
        "NOTIFY * HTTP/1.1\r\nHOST: {GROUP}:{PORT}\r\nNT: {target}\r\nNTS: ssdp:byebye\r\nUSN: {}\r\n\r\n",
        usn(udn, target),
    )
}

/// `uuid:` + a version-5-shaped UUID from the SHA-256 of a fixed label and
/// the machine id. Stable per host, and not the machine id itself.
#[must_use]
pub fn udn_from_machine_id(machine_id: &str) -> String {
    let mut input = b"llama-cast UDN v1\n".to_vec();
    input.extend_from_slice(machine_id.trim().as_bytes());
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&crate::sha256::digest(&input)[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex = crate::sha256::hex(&bytes);
    format!(
        "uuid:{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// A machine id is 32 lowercase hex digits (`/etc/machine-id`).
#[must_use]
pub fn valid_machine_id(text: &str) -> bool {
    let id = text.trim_end_matches('\n');
    id.len() == 32
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
