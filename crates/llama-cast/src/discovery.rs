//! The SSDP loop on the one UDP socket (bound to `0.0.0.0:1900` by
//! `service.rs`, joined to 239.255.255.250 on `interface_addr`).
//!
//! - M-SEARCH from a peer in the allowlist is answered by unicast to that
//!   peer; anything else (another sender, NOTIFY, junk, an oversize
//!   datagram) is dropped unanswered.
//! - NOTIFY ssdp:alive goes to the group at start and every 30 s, and
//!   ssdp:byebye when `stop` is set.
//! - Answers are rate-capped, so an allowed host cannot use the server as
//!   a datagram amplifier.
//!
//! The only datagrams sent are to the multicast group and, as answers, to
//! the M-SEARCH sender. There is no other outbound traffic.

use std::io;
use std::net::{IpAddr, SocketAddr, SocketAddrV4, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::acl::Allowlist;
use crate::ssdp::{self, GROUP, MAX_DATAGRAM, NOTIFY_INTERVAL_S, PORT};

/// Most answer datagrams sent in one second.
pub const MAX_REPLIES_PER_S: u32 = 32;

/// What discovery advertises.
#[derive(Clone, Debug)]
pub struct Advert {
    pub udn: String,
    /// `http://ADDR:PORT`.
    pub base_url: String,
    pub allow: Allowlist,
}

/// A one-second reply budget.
#[derive(Debug)]
pub struct ReplyBudget {
    window: Instant,
    used: u32,
}

impl ReplyBudget {
    #[must_use]
    pub fn new(now: Instant) -> Self {
        Self {
            window: now,
            used: 0,
        }
    }

    /// True when one more reply fits the current second.
    pub fn take(&mut self, now: Instant) -> bool {
        if now.duration_since(self.window) >= Duration::from_secs(1) {
            self.window = now;
            self.used = 0;
        }
        if self.used >= MAX_REPLIES_PER_S {
            return false;
        }
        self.used += 1;
        true
    }
}

/// The datagrams to send for one received datagram, or none.
#[must_use]
pub fn answers(advert: &Advert, datagram: &[u8], peer: SocketAddr) -> Vec<String> {
    if !advert.allow.permits(peer.ip()) || peer.port() == 0 {
        return Vec::new();
    }
    let Some(st) = ssdp::parse_msearch(datagram) else {
        return Vec::new();
    };
    ssdp::answer_targets(&st, &advert.udn)
        .iter()
        .map(|target| ssdp::search_reply(&advert.base_url, &advert.udn, target))
        .collect()
}

fn group() -> SocketAddr {
    SocketAddr::V4(SocketAddrV4::new(GROUP, PORT))
}

/// Send one NOTIFY ssdp:alive round.
pub fn announce(socket: &UdpSocket, advert: &Advert) {
    for target in ssdp::targets(&advert.udn) {
        let text = ssdp::notify_alive(&advert.base_url, &advert.udn, &target);
        let _ = socket.send_to(text.as_bytes(), group());
    }
}

/// Send one NOTIFY ssdp:byebye round.
pub fn byebye(socket: &UdpSocket, udn: &str) {
    for target in ssdp::targets(udn) {
        let text = ssdp::notify_byebye(udn, &target);
        let _ = socket.send_to(text.as_bytes(), group());
    }
}

/// Answer and announce until `stop` is set, then say byebye.
pub fn run(socket: &UdpSocket, advert: &Advert, stop: &Arc<AtomicBool>) -> io::Result<()> {
    socket.set_read_timeout(Some(Duration::from_millis(500)))?;
    let interval = Duration::from_secs(NOTIFY_INTERVAL_S);
    announce(socket, advert);
    let mut last = Instant::now();
    let mut budget = ReplyBudget::new(last);
    let mut buf = vec![0_u8; MAX_DATAGRAM + 1];
    while !stop.load(Ordering::SeqCst) {
        if last.elapsed() >= interval {
            announce(socket, advert);
            last = Instant::now();
        }
        let (len, peer) = match socket.recv_from(&mut buf) {
            Ok(got) => got,
            Err(err)
                if matches!(
                    err.kind(),
                    io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                        | io::ErrorKind::Interrupted
                ) =>
            {
                continue;
            }
            Err(err) => return Err(err),
        };
        if len > MAX_DATAGRAM || matches!(peer.ip(), IpAddr::V6(_)) {
            continue;
        }
        for reply in answers(advert, &buf[..len], peer) {
            if !budget.take(Instant::now()) {
                break;
            }
            let _ = socket.send_to(reply.as_bytes(), peer);
        }
    }
    byebye(socket, &advert.udn);
    Ok(())
}
