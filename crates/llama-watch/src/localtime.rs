//! Host local time for the tty11 header clock and RECENT's TIME column (#48).
//!
//! No zone crate and no libc (this crate forbids `unsafe`): the host zone is
//! read from the TZif file `TZ` names, or `/etc/localtime` when `TZ` is
//! unset, and the offset for an instant comes from its transitions or, past
//! the last one, from the footer's POSIX rule. Both live under `/etc` and
//! `/usr`, which `ProtectSystem=strict` leaves readable. Anything that fails
//! to load is UTC, the clock the header showed before.
//!
//! [`HostZone`] rereads the file at most once a minute, so a zone change
//! (`timedatectl set-timezone`) shows up without a restart; DST needs no
//! reread, since the rules cover it.

use std::path::Path;
use std::time::{Duration, Instant};

/// Seconds between rereads of the host zone.
pub const ZONE_REFRESH: Duration = Duration::from_secs(60);

/// Where `TZ=Name` is looked up.
const ZONEINFO: &str = "/usr/share/zoneinfo";
/// The host zone when `TZ` is unset.
const LOCALTIME: &str = "/etc/localtime";
/// Largest zone file read. Real ones are a few KiB.
const MAX_TZIF: u64 = 256 * 1024;

/// A time zone: offsets from UTC by instant.
#[derive(Clone, Debug, PartialEq)]
pub struct Zone {
    /// Transition instants (Unix seconds, ascending) and the index into
    /// `offsets` that applies from each one on.
    transitions: Vec<(i64, usize)>,
    /// UTC offsets in seconds, east positive. Empty means UTC. Index 0 is
    /// the offset before the first transition.
    offsets: Vec<i32>,
    /// The footer rule, for instants after the last transition.
    rule: Option<PosixRule>,
}

static UTC: Zone = Zone::UTC;

impl Zone {
    /// UTC.
    pub const UTC: Self = Self {
        transitions: Vec::new(),
        offsets: Vec::new(),
        rule: None,
    };

    /// A shared UTC zone.
    #[must_use]
    pub fn utc() -> &'static Self {
        &UTC
    }

    /// A zone that is always `offset_s` seconds east of UTC.
    #[must_use]
    pub fn fixed(offset_s: i32) -> Self {
        Self {
            transitions: Vec::new(),
            offsets: vec![offset_s],
            rule: None,
        }
    }

    /// Parse a TZif file (RFC 8536, versions 1 to 4). The 64-bit block and
    /// the footer rule are used when present. Leap-second records are
    /// skipped. `None` when the data is not TZif.
    #[must_use]
    pub fn from_tzif(data: &[u8]) -> Option<Self> {
        let (v1, after_v1) = tzif_block(data, 4)?;
        if data[4] == 0 {
            return Some(v1);
        }
        let rest = data.get(after_v1..)?;
        let (mut v2, after_v2) = tzif_block(rest, 8)?;
        let footer = rest.get(after_v2..).unwrap_or_default();
        let footer = footer.strip_prefix(b"\n").unwrap_or_default();
        let end = footer.iter().position(|&b| b == b'\n').unwrap_or(0);
        let text = std::str::from_utf8(&footer[..end]).ok().unwrap_or("");
        if !text.is_empty() {
            v2.rule = PosixRule::parse(text);
        }
        Some(v2)
    }

    /// A zone from a POSIX `TZ` rule such as `EST5EDT,M3.2.0,M11.1.0`.
    #[must_use]
    pub fn from_posix(text: &str) -> Option<Self> {
        Some(Self {
            transitions: Vec::new(),
            offsets: Vec::new(),
            rule: Some(PosixRule::parse(text)?),
        })
    }

    /// UTC offset in seconds (east positive) at Unix second `unix`.
    #[must_use]
    pub fn offset_at(&self, unix: i64) -> i32 {
        let after_last = self.transitions.last().is_none_or(|&(at, _)| unix >= at);
        if after_last && let Some(rule) = &self.rule {
            return rule.offset_at(unix);
        }
        let index = match self.transitions.partition_point(|&(at, _)| at <= unix) {
            0 => 0,
            n => self.transitions[n - 1].1,
        };
        self.offsets.get(index).copied().unwrap_or(0)
    }
}

/// One TZif header and data block. `time_size` is 4 (v1) or 8 (v2+).
/// Returns the zone and the bytes the block used.
fn tzif_block(data: &[u8], time_size: usize) -> Option<(Zone, usize)> {
    if data.get(..4)? != b"TZif" {
        return None;
    }
    let count = |at: usize| -> Option<usize> {
        let bytes: [u8; 4] = data.get(at..at + 4)?.try_into().ok()?;
        usize::try_from(u32::from_be_bytes(bytes)).ok()
    };
    let isutcnt = count(20)?;
    let isstdcnt = count(24)?;
    let leapcnt = count(28)?;
    let timecnt = count(32)?;
    let typecnt = count(36)?;
    let charcnt = count(40)?;
    if typecnt == 0 {
        return None;
    }
    let times_at = 44usize;
    let index_at = times_at.checked_add(timecnt.checked_mul(time_size)?)?;
    let types_at = index_at.checked_add(timecnt)?;
    let chars_at = types_at.checked_add(typecnt.checked_mul(6)?)?;
    let end = chars_at
        .checked_add(charcnt)?
        .checked_add(leapcnt.checked_mul(time_size + 4)?)?
        .checked_add(isstdcnt)?
        .checked_add(isutcnt)?;
    if data.len() < end {
        return None;
    }
    let mut offsets = Vec::with_capacity(typecnt);
    for i in 0..typecnt {
        let at = types_at + i * 6;
        let bytes: [u8; 4] = data[at..at + 4].try_into().ok()?;
        offsets.push(i32::from_be_bytes(bytes));
    }
    let mut transitions = Vec::with_capacity(timecnt);
    for i in 0..timecnt {
        let at = times_at + i * time_size;
        let time = if time_size == 8 {
            i64::from_be_bytes(data[at..at + 8].try_into().ok()?)
        } else {
            i64::from(i32::from_be_bytes(data[at..at + 4].try_into().ok()?))
        };
        let index = usize::from(data[index_at + i]);
        if index >= typecnt {
            return None;
        }
        if transitions.last().is_some_and(|&(prev, _)| time <= prev) {
            return None;
        }
        transitions.push((time, index));
    }
    Some((
        Zone {
            transitions,
            offsets,
            rule: None,
        },
        end,
    ))
}

/// A POSIX `TZ` rule: a standard offset and, optionally, DST with its
/// start and end.
#[derive(Clone, Debug, PartialEq)]
struct PosixRule {
    /// Standard UTC offset, east positive.
    std: i32,
    /// DST offset and its start and end.
    dst: Option<(i32, Change, Change)>,
}

/// When DST starts or ends: a day of the year and seconds after local
/// midnight (may be negative or past 24 h).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Change {
    day: Day,
    secs: i64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Day {
    /// `Jn`: 1..=365, never counting February 29.
    Julian(u16),
    /// `n`: 0..=365, counting February 29.
    Zero(u16),
    /// `Mm.w.d`: month, week 1..=5 (5 is the last), weekday 0 (Sunday)..=6.
    Month(u8, u8, u8),
}

impl PosixRule {
    fn parse(text: &str) -> Option<Self> {
        let mut rest = text.as_bytes();
        name(&mut rest)?;
        // POSIX offsets are west positive.
        let std = -hms(&mut rest, true)?;
        if rest.is_empty() {
            return Some(Self { std, dst: None });
        }
        name(&mut rest)?;
        let dst_off = if rest.first().is_some_and(|&b| b != b',') {
            -hms(&mut rest, true)?
        } else {
            std + 3600
        };
        let (start, end) = if rest.is_empty() {
            // The US rule, as glibc assumes when none is given.
            (
                Change {
                    day: Day::Month(3, 2, 0),
                    secs: 7200,
                },
                Change {
                    day: Day::Month(11, 1, 0),
                    secs: 7200,
                },
            )
        } else {
            rest = rest.strip_prefix(b",")?;
            let start = change(&mut rest)?;
            rest = rest.strip_prefix(b",")?;
            let end = change(&mut rest)?;
            (start, end)
        };
        if !rest.is_empty() {
            return None;
        }
        Some(Self {
            std,
            dst: Some((dst_off, start, end)),
        })
    }

    fn offset_at(&self, unix: i64) -> i32 {
        let Some((dst, start, end)) = self.dst else {
            return self.std;
        };
        let year = civil_from_days((unix + i64::from(self.std)).div_euclid(86_400)).0;
        // Start is given in standard time, end in DST.
        let begins = start.local_secs(year) - i64::from(self.std);
        let ends = end.local_secs(year) - i64::from(dst);
        let in_dst = if begins < ends {
            begins <= unix && unix < ends
        } else {
            // Southern hemisphere: DST spans the new year.
            !(ends <= unix && unix < begins)
        };
        if in_dst { dst } else { self.std }
    }
}

impl Change {
    /// The change in `year`, as seconds since the epoch in local (wall) terms.
    fn local_secs(self, year: i64) -> i64 {
        let jan1 = days_from_civil(year, 1, 1);
        let leap = is_leap(year);
        let day = match self.day {
            Day::Julian(n) => {
                let n = i64::from(n);
                jan1 + n - 1 + i64::from(leap && n >= 60)
            }
            Day::Zero(n) => jan1 + i64::from(n),
            Day::Month(month, week, weekday) => {
                let first = days_from_civil(year, u32::from(month), 1);
                // 1970-01-01 was a Thursday (4).
                let first_wd = (first + 4).rem_euclid(7);
                let mut day = first + (i64::from(weekday) - first_wd).rem_euclid(7);
                day += 7 * (i64::from(week) - 1);
                let len = month_len(year, u32::from(month));
                while day >= first + len {
                    day -= 7;
                }
                day
            }
        };
        day * 86_400 + self.secs
    }
}

/// Skip a zone abbreviation: three or more letters, or `<...>`.
fn name(rest: &mut &[u8]) -> Option<()> {
    if let Some(quoted) = rest.strip_prefix(b"<") {
        let close = quoted.iter().position(|&b| b == b'>')?;
        *rest = &quoted[close + 1..];
        return Some(());
    }
    let len = rest.iter().take_while(|b| b.is_ascii_alphabetic()).count();
    if len < 3 {
        return None;
    }
    *rest = &rest[len..];
    Some(())
}

/// `[+-]hh[:mm[:ss]]` in seconds. `signed` allows the sign.
fn hms(rest: &mut &[u8], signed: bool) -> Option<i32> {
    let mut sign = 1;
    if signed && let Some((&first, tail)) = rest.split_first() {
        if first == b'-' {
            sign = -1;
            *rest = tail;
        } else if first == b'+' {
            *rest = tail;
        }
    }
    let hours = number(rest, 167)?;
    let mut secs = hours * 3600;
    for scale in [60, 1] {
        let Some(tail) = rest.strip_prefix(b":") else {
            break;
        };
        *rest = tail;
        secs += number(rest, 59)? * scale;
    }
    Some(sign * secs)
}

fn number(rest: &mut &[u8], max: i32) -> Option<i32> {
    let len = rest.iter().take_while(|b| b.is_ascii_digit()).count();
    if len == 0 || len > 3 {
        return None;
    }
    let value: i32 = std::str::from_utf8(&rest[..len]).ok()?.parse().ok()?;
    *rest = &rest[len..];
    (value <= max).then_some(value)
}

fn change(rest: &mut &[u8]) -> Option<Change> {
    let day = if let Some(tail) = rest.strip_prefix(b"J") {
        *rest = tail;
        let n = number(rest, 365)?;
        Day::Julian(u16::try_from(n).ok().filter(|&n| n >= 1)?)
    } else if let Some(tail) = rest.strip_prefix(b"M") {
        *rest = tail;
        let month = number(rest, 12)?;
        *rest = rest.strip_prefix(b".")?;
        let week = number(rest, 5)?;
        *rest = rest.strip_prefix(b".")?;
        let weekday = number(rest, 6)?;
        if month < 1 || week < 1 {
            return None;
        }
        Day::Month(
            u8::try_from(month).ok()?,
            u8::try_from(week).ok()?,
            u8::try_from(weekday).ok()?,
        )
    } else {
        Day::Zero(u16::try_from(number(rest, 365)?).ok()?)
    };
    let secs = if let Some(tail) = rest.strip_prefix(b"/") {
        *rest = tail;
        i64::from(hms(rest, true)?)
    } else {
        7200
    };
    Some(Change { day, secs })
}

/// The host zone: `TZ`, else `/etc/localtime`, else UTC.
#[must_use]
pub fn load_host_zone() -> Zone {
    match std::env::var("TZ") {
        Ok(tz) => zone_from_tz(&tz, Path::new(ZONEINFO)),
        Err(std::env::VarError::NotPresent) => read_tzif(Path::new(LOCALTIME)).unwrap_or_default(),
        Err(std::env::VarError::NotUnicode(_)) => Zone::UTC,
    }
}

/// The zone a `TZ` value names, as glibc reads it: empty is UTC; `:file`
/// or `file` is a TZif file (relative to `zoneinfo`); otherwise a POSIX
/// rule. A name with a `..` part is not looked up.
fn zone_from_tz(tz: &str, zoneinfo: &Path) -> Zone {
    let spec = tz.strip_prefix(':').unwrap_or(tz);
    if spec.is_empty() {
        return Zone::UTC;
    }
    let path = Path::new(spec);
    let file = if path.is_absolute() {
        read_tzif(path)
    } else if path
        .components()
        .all(|part| matches!(part, std::path::Component::Normal(_)))
    {
        read_tzif(&zoneinfo.join(path))
    } else {
        None
    };
    file.or_else(|| Zone::from_posix(spec)).unwrap_or_default()
}

fn read_tzif(path: &Path) -> Option<Zone> {
    use std::io::Read;
    let file = std::fs::File::open(path).ok()?;
    let mut data = Vec::new();
    file.take(MAX_TZIF).read_to_end(&mut data).ok()?;
    Zone::from_tzif(&data)
}

impl Default for Zone {
    fn default() -> Self {
        Self::UTC
    }
}

/// The host zone, reread at most every [`ZONE_REFRESH`].
pub struct HostZone {
    zone: Zone,
    loaded: Option<Instant>,
    load: fn() -> Zone,
}

impl HostZone {
    /// Reads the real host zone on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::with_loader(load_host_zone)
    }

    /// A host zone read by `load` (tests).
    #[must_use]
    pub fn with_loader(load: fn() -> Zone) -> Self {
        Self {
            zone: Zone::UTC,
            loaded: None,
            load,
        }
    }

    /// The zone at monotonic time `mono`, rereading it when the last read
    /// is [`ZONE_REFRESH`] old.
    pub fn get(&mut self, mono: Instant) -> &Zone {
        let stale = self
            .loaded
            .is_none_or(|at| mono.saturating_duration_since(at) >= ZONE_REFRESH);
        if stale {
            self.zone = (self.load)();
            self.loaded = Some(mono);
        }
        &self.zone
    }
}

impl Default for HostZone {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse an RFC 3339 date-time (`2026-10-04T05:52:17Z`, an optional
/// fraction, `Z` or `±hh:mm`; `t`, `z` and a space separator are accepted)
/// to Unix seconds. The fraction is dropped. `None` for anything else.
#[must_use]
pub fn parse_rfc3339(text: &str) -> Option<i64> {
    let b = text.as_bytes();
    let digits = |from: usize, len: usize| -> Option<i64> {
        let part = b.get(from..from + len)?;
        if !part.iter().all(u8::is_ascii_digit) {
            return None;
        }
        std::str::from_utf8(part).ok()?.parse().ok()
    };
    let year = digits(0, 4)?;
    let month = digits(5, 2)?;
    let day = digits(8, 2)?;
    let hour = digits(11, 2)?;
    let min = digits(14, 2)?;
    let sec = digits(17, 2)?;
    if b.get(4) != Some(&b'-')
        || b.get(7) != Some(&b'-')
        || !matches!(b.get(10), Some(b'T' | b't' | b' '))
        || b.get(13) != Some(&b':')
        || b.get(16) != Some(&b':')
    {
        return None;
    }
    let month_u = u32::try_from(month).ok()?;
    if !(1..=12).contains(&month_u)
        || day < 1
        || day > month_len(year, month_u)
        || hour > 23
        || min > 59
        || sec > 60
    {
        return None;
    }
    let mut at = 19;
    if b.get(at) == Some(&b'.') {
        at += 1;
        let len = b[at..].iter().take_while(|c| c.is_ascii_digit()).count();
        if len == 0 {
            return None;
        }
        at += len;
    }
    let offset = match b.get(at..)? {
        b"Z" | b"z" => 0,
        [sign @ (b'+' | b'-'), _, _, b':', _, _] => {
            let hh = digits(at + 1, 2)?;
            let mm = digits(at + 4, 2)?;
            if hh > 23 || mm > 59 {
                return None;
            }
            let secs = hh * 3600 + mm * 60;
            if *sign == b'-' { -secs } else { secs }
        }
        _ => return None,
    };
    // A leap second (:60) counts as the next second.
    let days = days_from_civil(year, month_u, u32::try_from(day).ok()?);
    Some(days * 86_400 + hour * 3600 + min * 60 + sec - offset)
}

/// `YYYY-MM-DD HH:MM:SS` for Unix second `unix` in `zone`.
#[must_use]
pub fn format_local(unix: i64, zone: &Zone) -> String {
    let local = unix + i64::from(zone.offset_at(unix));
    let days = local.div_euclid(86_400);
    let tod = local.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02} {hour:02}:{min:02}:{sec:02}",
        hour = tod / 3600,
        min = (tod % 3600) / 60,
        sec = tod % 60
    )
}

/// An activity timestamp as RECENT shows it: in `zone` when it is RFC 3339,
/// else `raw` (already sanitised) unchanged.
#[must_use]
pub fn local_request_time(raw: &str, zone: &Zone) -> String {
    match parse_rfc3339(raw) {
        Some(unix) => format_local(unix, zone),
        None => raw.to_owned(),
    }
}

fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn month_len(year: i64, month: u32) -> i64 {
    match month {
        2 if is_leap(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Howard Hinnant's `days_from_civil`: days since 1970-01-01.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let yoe = year - era * 400;
    let month = i64::from(month);
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Howard Hinnant's `civil_from_days`. `days` is days since 1970-01-01.
#[must_use]
pub fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (
        year,
        u32::try_from(month).unwrap_or(1),
        u32::try_from(day).unwrap_or(1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// 2026-10-04T05:52:17Z, the issue's example.
    const ISSUE: i64 = 1_791_093_137;

    fn new_york() -> Zone {
        let data = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/tz/America-New_York.tzif"
        ));
        Zone::from_tzif(data).expect("fixture TZif")
    }

    /// A TZif file: a v1 block, and for `version > 0` a v2 block and footer.
    fn tzif(version: u8, transitions: &[(i64, u8)], offsets: &[i32], footer: &str) -> Vec<u8> {
        let block = |time_size: usize| {
            let mut out = Vec::new();
            out.extend_from_slice(b"TZif");
            out.push(version);
            out.extend_from_slice(&[0; 15]);
            for count in [0, 0, 0, transitions.len(), offsets.len(), 4] {
                out.extend_from_slice(&u32::try_from(count).unwrap().to_be_bytes());
            }
            for &(at, _) in transitions {
                if time_size == 8 {
                    out.extend_from_slice(&at.to_be_bytes());
                } else {
                    out.extend_from_slice(&i32::try_from(at).unwrap().to_be_bytes());
                }
            }
            out.extend(transitions.iter().map(|&(_, index)| index));
            for &offset in offsets {
                out.extend_from_slice(&offset.to_be_bytes());
                out.extend_from_slice(&[0, 0]);
            }
            out.extend_from_slice(b"ABC\0");
            out
        };
        let mut out = block(4);
        if version != 0 {
            out.extend(block(8));
            out.push(b'\n');
            out.extend_from_slice(footer.as_bytes());
            out.push(b'\n');
        }
        out
    }

    #[test]
    fn parses_z_offsets_and_fractions() {
        assert_eq!(parse_rfc3339("2026-10-04T05:52:17Z"), Some(ISSUE));
        assert_eq!(parse_rfc3339("2026-10-04t05:52:17z"), Some(ISSUE));
        assert_eq!(parse_rfc3339("2026-10-04 05:52:17Z"), Some(ISSUE));
        assert_eq!(parse_rfc3339("2026-10-04T05:52:17.5Z"), Some(ISSUE));
        assert_eq!(parse_rfc3339("2026-10-04T05:52:17.123456789Z"), Some(ISSUE));
        assert_eq!(parse_rfc3339("2026-10-04T00:52:17-05:00"), Some(ISSUE));
        assert_eq!(parse_rfc3339("2026-10-04T11:22:17.04+05:30"), Some(ISSUE));
        assert_eq!(parse_rfc3339("2026-10-04T05:52:17+00:00"), Some(ISSUE));
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_rfc3339("2024-02-29T00:00:00Z"), Some(1_709_164_800));
    }

    #[test]
    fn rejects_junk() {
        for junk in [
            "",
            "18:47:01",
            "2026-10-04",
            "2026-10-04T05:52:17",
            "2026-10-04 05:52:17",
            "2026-10-04T05:52:17+0500",
            "2026-10-04T05:52:17+05",
            "2026-10-04T05:52:17.Z",
            "2026-10-04T05:52:17Zjunk",
            "2026-10-04T05:52:17Z[2J",
            "2026-13-04T05:52:17Z",
            "2026-02-30T05:52:17Z",
            "2025-02-29T05:52:17Z",
            "2026-10-04T24:00:00Z",
            "2026-10-04T05:60:00Z",
            "2026-10-04T05:52:17+24:00",
            "2026/10/04T05:52:17Z",
            "+026-10-04T05:52:17Z",
            "2026-10-04T05:52:1٧Z",
            "not a time",
        ] {
            assert_eq!(parse_rfc3339(junk), None, "{junk:?}");
        }
    }

    #[test]
    fn converts_with_a_fixed_zone() {
        assert_eq!(format_local(ISSUE, Zone::utc()), "2026-10-04 05:52:17");
        assert_eq!(
            format_local(ISSUE, &Zone::fixed(-5 * 3600)),
            "2026-10-04 00:52:17"
        );
        assert_eq!(
            format_local(ISSUE, &Zone::fixed(-6 * 3600)),
            "2026-10-03 23:52:17"
        );
        assert_eq!(
            format_local(ISSUE, &Zone::fixed(5 * 3600 + 1800)),
            "2026-10-04 11:22:17"
        );
        assert_eq!(format_local(0, &Zone::fixed(-1)), "1969-12-31 23:59:59");
    }

    #[test]
    fn the_fixture_zone_follows_dst() {
        let zone = new_york();
        // Summer (EDT, -4) and winter (EST, -5).
        assert_eq!(format_local(ISSUE, &zone), "2026-10-04 01:52:17");
        let winter = parse_rfc3339("2026-01-15T12:00:00Z").unwrap();
        assert_eq!(format_local(winter, &zone), "2026-01-15 07:00:00");
        // 2026-03-08 02:00 EST jumps to 03:00 EDT; 2026-11-01 02:00 EDT
        // falls back to 01:00 EST.
        let spring = parse_rfc3339("2026-03-08T07:00:00Z").unwrap();
        assert_eq!(zone.offset_at(spring - 1), -5 * 3600);
        assert_eq!(zone.offset_at(spring), -4 * 3600);
        let fall = parse_rfc3339("2026-11-01T06:00:00Z").unwrap();
        assert_eq!(zone.offset_at(fall - 1), -4 * 3600);
        assert_eq!(zone.offset_at(fall), -5 * 3600);
        // Far past the table: the footer rule decides.
        let later = parse_rfc3339("2100-07-01T12:00:00Z").unwrap();
        assert_eq!(zone.offset_at(later), -4 * 3600);
        let later = parse_rfc3339("2100-12-01T12:00:00Z").unwrap();
        assert_eq!(zone.offset_at(later), -5 * 3600);
    }

    #[test]
    fn a_slim_v2_file_runs_on_its_footer() {
        // One historic transition, then the POSIX rule.
        let data = tzif(b'2', &[(-100, 0)], &[-18_000], "EST5EDT,M3.2.0,M11.1.0");
        let zone = Zone::from_tzif(&data).expect("slim");
        assert_eq!(zone, Zone::from_tzif(&data).unwrap());
        assert_eq!(zone.offset_at(ISSUE), -4 * 3600);
        assert_eq!(
            zone.offset_at(parse_rfc3339("2026-01-15T12:00:00Z").unwrap()),
            -5 * 3600
        );
        // Before the first transition: type 0.
        assert_eq!(zone.offset_at(-1_000), -18_000);
    }

    #[test]
    fn a_v1_file_uses_its_transitions() {
        let data = tzif(0, &[(1_000, 1), (2_000, 0)], &[3600, 7200], "");
        let zone = Zone::from_tzif(&data).expect("v1");
        assert_eq!(zone.offset_at(0), 3600);
        assert_eq!(zone.offset_at(1_000), 7200);
        assert_eq!(zone.offset_at(1_999), 7200);
        assert_eq!(zone.offset_at(5_000), 3600);
        // An empty footer keeps the last type.
        let data = tzif(b'3', &[(1_000, 1)], &[3600, 7200], "");
        assert_eq!(Zone::from_tzif(&data).unwrap().offset_at(9_999_999), 7200);
    }

    #[test]
    fn bad_tzif_is_refused() {
        assert_eq!(Zone::from_tzif(b""), None);
        assert_eq!(Zone::from_tzif(b"TZif2 not really a zone file"), None);
        let mut data = tzif(0, &[(1_000, 1)], &[3600, 7200], "");
        data.truncate(data.len() - 3);
        assert_eq!(Zone::from_tzif(&data), None);
        // A type index past the table.
        let data = tzif(0, &[(1_000, 5)], &[3600], "");
        assert_eq!(Zone::from_tzif(&data), None);
    }

    #[test]
    fn posix_rules_cover_both_hemispheres() {
        let zone = Zone::from_posix("<+0530>-5:30").expect("fixed");
        assert_eq!(zone.offset_at(ISSUE), 19_800);
        let zone = Zone::from_posix("UTC0").expect("utc");
        assert_eq!(zone.offset_at(ISSUE), 0);
        // Australian east coast: DST from October to April.
        let zone = Zone::from_posix("AEST-10AEDT,M10.1.0,M4.1.0/3").expect("south");
        let jan = parse_rfc3339("2026-01-15T00:00:00Z").unwrap();
        let jul = parse_rfc3339("2026-07-15T00:00:00Z").unwrap();
        assert_eq!(zone.offset_at(jan), 11 * 3600);
        assert_eq!(zone.offset_at(jul), 10 * 3600);
        // 2026-10-04 02:00 AEST (16:00Z on the 3rd) starts DST.
        let start = parse_rfc3339("2026-10-03T16:00:00Z").unwrap();
        assert_eq!(zone.offset_at(start - 1), 10 * 3600);
        assert_eq!(zone.offset_at(start), 11 * 3600);
        // Julian and zero-based days.
        let zone = Zone::from_posix("XST3XDT,J60/0,300").expect("days");
        let mar1 = parse_rfc3339("2026-03-01T03:00:00Z").unwrap();
        assert_eq!(zone.offset_at(mar1 - 1), -3 * 3600);
        assert_eq!(zone.offset_at(mar1), -2 * 3600);
        for junk in [
            "",
            "X5",
            "EST",
            "EST5EDT,M3.2",
            "EST5EDT,M13.1.0,M11.1.0",
            "<EST5",
        ] {
            assert_eq!(Zone::from_posix(junk), None, "{junk:?}");
        }
    }

    #[test]
    fn tz_names_files_rules_or_utc() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tz");
        let ny = new_york();
        assert_eq!(zone_from_tz("America-New_York.tzif", &dir), ny);
        assert_eq!(zone_from_tz(":America-New_York.tzif", &dir), ny);
        let abs = dir.join("America-New_York.tzif");
        assert_eq!(
            zone_from_tz(abs.to_str().unwrap(), Path::new("/nowhere")),
            ny
        );
        assert_eq!(zone_from_tz("", &dir), Zone::UTC);
        assert_eq!(zone_from_tz(":", &dir), Zone::UTC);
        assert_eq!(
            zone_from_tz("EST5EDT,M3.2.0,M11.1.0", &dir).offset_at(ISSUE),
            -4 * 3600
        );
        // `..` is never followed, and an unknown name is UTC.
        assert_eq!(zone_from_tz("../tz/America-New_York.tzif", &dir), Zone::UTC);
        assert_eq!(zone_from_tz("Nowhere/Zone", &dir), Zone::UTC);
    }

    #[test]
    fn the_host_zone_is_reread_at_most_once_a_minute() {
        use std::sync::atomic::{AtomicI32, Ordering};
        static OFFSET: AtomicI32 = AtomicI32::new(3600);
        fn load() -> Zone {
            Zone::fixed(OFFSET.load(Ordering::SeqCst))
        }
        let mut host = HostZone::with_loader(load);
        let t0 = Instant::now();
        assert_eq!(host.get(t0).offset_at(0), 3600);
        OFFSET.store(7200, Ordering::SeqCst);
        assert_eq!(host.get(t0 + Duration::from_secs(59)).offset_at(0), 3600);
        assert_eq!(host.get(t0 + ZONE_REFRESH).offset_at(0), 7200);
    }

    #[test]
    fn request_time_converts_or_falls_back() {
        let zone = Zone::fixed(-5 * 3600);
        assert_eq!(
            local_request_time("2026-10-04T05:52:17Z", &zone),
            "2026-10-04 00:52:17"
        );
        assert_eq!(
            local_request_time("2026-10-04T05:52:17.25+02:00", &zone),
            "2026-10-03 22:52:17"
        );
        for raw in [
            "18:47:01",
            "2026-10-04 05:52:17",
            "2026-10-04T05:52:17Z[2J",
            "",
        ] {
            assert_eq!(local_request_time(raw, &zone), raw);
        }
    }

    #[test]
    fn civil_round_trips() {
        for days in [-719_468, -1, 0, 1, 10_957, 20_729, 2_932_896] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days, "{days}");
        }
    }
}
