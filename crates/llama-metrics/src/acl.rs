//! In-process CIDR allowlist. A peer not in the list is closed before any
//! byte of its request is read.
//!
//! The unit's `IPAddressAllow=` is the kernel copy of the same list; the
//! packaging test keeps the two equal for the shipped example.

use std::fmt;
use std::net::IpAddr;
use std::str::FromStr;

use thiserror::Error;

/// Why a CIDR string was refused.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CidrError {
    /// Not `ADDR/PREFIX`.
    #[error("CIDR {0:?} is not ADDR/PREFIX")]
    Syntax(String),
    /// The prefix is 0 or longer than the address family allows.
    #[error("CIDR {0:?} has a prefix outside 1..=32 (IPv4) or 1..=128 (IPv6)")]
    Prefix(String),
    /// Bits after the prefix are set, such as `192.168.0.5/22`.
    #[error("CIDR {0:?} has host bits set")]
    HostBits(String),
}

/// One network: an address with every host bit clear, and a prefix length.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cidr {
    net: IpAddr,
    prefix: u8,
}

impl Cidr {
    /// Network address.
    #[must_use]
    pub fn network(&self) -> IpAddr {
        self.net
    }

    /// Prefix length.
    #[must_use]
    pub fn prefix(&self) -> u8 {
        self.prefix
    }

    /// True when `addr` is inside this network. An IPv4-mapped IPv6 peer
    /// (`::ffff:a.b.c.d`, from a dual-stack `[::]` socket) is compared as IPv4.
    #[must_use]
    pub fn contains(&self, addr: IpAddr) -> bool {
        match (self.net, addr.to_canonical()) {
            (IpAddr::V4(net), IpAddr::V4(peer)) => {
                let mask = mask32(self.prefix);
                u32::from(peer) & mask == u32::from(net)
            }
            (IpAddr::V6(net), IpAddr::V6(peer)) => {
                let mask = mask128(self.prefix);
                u128::from(peer) & mask == u128::from(net)
            }
            _ => false,
        }
    }
}

fn mask32(prefix: u8) -> u32 {
    if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - u32::from(prefix))
    }
}

fn mask128(prefix: u8) -> u128 {
    if prefix == 0 {
        0
    } else {
        u128::MAX << (128 - u32::from(prefix))
    }
}

impl FromStr for Cidr {
    type Err = CidrError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let (addr, prefix) = text
            .split_once('/')
            .ok_or_else(|| CidrError::Syntax(text.to_owned()))?;
        let net: IpAddr = addr
            .parse()
            .map_err(|_| CidrError::Syntax(text.to_owned()))?;
        // Peers are compared in canonical form, so a v4-mapped v6 network
        // would never match. Write it as IPv4.
        if net.is_ipv6() && net.to_canonical().is_ipv4() {
            return Err(CidrError::Syntax(text.to_owned()));
        }
        if prefix.is_empty() || !prefix.bytes().all(|b| b.is_ascii_digit()) || prefix.len() > 3 {
            return Err(CidrError::Syntax(text.to_owned()));
        }
        let prefix: u8 = prefix
            .parse()
            .map_err(|_| CidrError::Syntax(text.to_owned()))?;
        let max = if net.is_ipv4() { 32 } else { 128 };
        // /0 is the whole internet. The exporter is LAN-only by design.
        if prefix == 0 || prefix > max {
            return Err(CidrError::Prefix(text.to_owned()));
        }
        let clean = match net {
            IpAddr::V4(v4) => u32::from(v4) & !mask32(prefix) == 0,
            IpAddr::V6(v6) => u128::from(v6) & !mask128(prefix) == 0,
        };
        if !clean {
            return Err(CidrError::HostBits(text.to_owned()));
        }
        Ok(Self { net, prefix })
    }
}

impl fmt::Display for Cidr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.net, self.prefix)
    }
}

/// The peers allowed to connect. Empty allows nobody.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Allowlist {
    nets: Vec<Cidr>,
}

impl Allowlist {
    #[must_use]
    pub fn new(nets: Vec<Cidr>) -> Self {
        Self { nets }
    }

    /// True when some network in the list contains `peer`.
    #[must_use]
    pub fn permits(&self, peer: IpAddr) -> bool {
        self.nets.iter().any(|net| net.contains(peer))
    }

    /// The networks, in config order.
    #[must_use]
    pub fn nets(&self) -> &[Cidr] {
        &self.nets
    }
}
