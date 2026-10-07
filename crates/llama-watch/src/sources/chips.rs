//! Every hwmon chip, for the TEMPS and FANS discovery (#74). Read only.
//!
//! [`discover`] lists each `/sys/class/hwmon/hwmonN` that resolves inside
//! the sys root and has a readable `name`, and gives it a stable chip name
//! for patterns, renames and the export:
//!
//! - the hwmon `name`, lowercased, with every character outside
//!   `[a-z0-9_.-]` made `_` (`r8169_0_500:00` is `r8169_0_500_00`);
//! - when two or more chips share that name, a suffix from the device, the
//!   first of these that tells every one of them apart: the last four
//!   letters or digits of `device/serial` (NVMe: `nvme-345a`), the PCI
//!   address of the device as `bus.dev` (`r8169-05.00`), the device's own
//!   name (`nvme0`), and last the `hwmonN` number.
//!
//! Chips are ordered by kind (CPU, GPU, coolant, NVMe, disk, board, memory,
//! NIC, Wi-Fi, ACPI, other), then by device, then by name: the order of the
//! TEMPS rows, the wire rows and the trimming of an oversize snapshot.
//!
//! [`Pattern`] is the glob `[temps]` and `[fans]` share: `chip` or
//! `chip:sensor`, `*` for any run and `?` for one character, ASCII case
//! ignored. A sensor has two names, its label (`tempN_label`,
//! `fanN_label`) and its input (`tempN`, `fanN`); a pattern matches either.

use std::path::{Path, PathBuf};

use crate::sources::hwmon::{Located, locate, read_sensor_name, sys_root};
use crate::sources::{Roots, SourceId};

/// Longest chip name, as the wire allows.
pub const MAX_CHIP: usize = llama_core::wire::MAX_TEMP_CHIP_CHARS;
/// Most chips one discovery keeps.
pub const MAX_CHIPS: usize = 64;

/// What a chip measures, from its hwmon `name`. Orders the TEMPS rows.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Kind {
    /// `k10temp`, `coretemp`, `zenpower`.
    Cpu,
    /// The GPU: NVML (chip `gpu`), or an `amdgpu`, `nouveau`, `radeon`,
    /// `i915` or `xe` hwmon.
    Gpu,
    /// A liquid cooler: `z53`, `x53`, `kraken*`, `d5next`.
    Coolant,
    /// `nvme`.
    Nvme,
    /// `drivetemp`.
    Disk,
    /// A Super-I/O or board controller: `nct*`, `it87`, `w83*`, `f71*`,
    /// `asus*`.
    Board,
    /// `spd5118`, `jc42`.
    Memory,
    /// A network adapter driver (`r8169`, `igc`, `ixgbe`, ...).
    Nic,
    /// A Wi-Fi driver (`iwlwifi`, `mt79*`, `ath1*`, ...).
    Wifi,
    /// `acpitz`.
    Acpi,
    /// Anything else, under its own name.
    Other,
}

const NIC_DRIVERS: [&str; 14] = [
    "r8169", "r8125", "r8126", "igb", "igc", "ixgbe", "i40e", "ice", "e1000e", "atlantic", "bnxt",
    "mlx5", "tg3", "aq",
];
const WIFI_DRIVERS: [&str; 7] = ["iwlwifi", "mt79", "mt76", "ath1", "ath9k", "rtw", "brcmf"];

impl Kind {
    /// The kind of a raw hwmon `name`.
    #[must_use]
    pub fn of(name: &str) -> Self {
        let name = name.to_ascii_lowercase();
        let starts = |prefixes: &[&str]| prefixes.iter().any(|p| name.starts_with(p));
        match name.as_str() {
            "k10temp" | "coretemp" | "zenpower" => Self::Cpu,
            "gpu" | "amdgpu" | "nouveau" | "radeon" | "i915" | "xe" => Self::Gpu,
            "z53" | "x53" | "d5next" => Self::Coolant,
            "nvme" => Self::Nvme,
            "drivetemp" => Self::Disk,
            "spd5118" | "jc42" => Self::Memory,
            "acpitz" => Self::Acpi,
            _ if starts(&["kraken"]) => Self::Coolant,
            _ if starts(&["nct", "it87", "it86", "w83", "f71", "asus"]) => Self::Board,
            _ if starts(&NIC_DRIVERS) => Self::Nic,
            _ if starts(&WIFI_DRIVERS) => Self::Wifi,
            _ => Self::Other,
        }
    }

    /// TEMPS row name: `CPU`, `GPU`, `coolant`, `NVMe`, `disk`, `board`,
    /// `DIMM`, `NIC`, `Wi-Fi`, `ACPI`; `None` for [`Kind::Other`], which
    /// uses the chip name.
    #[must_use]
    pub fn word(self) -> Option<&'static str> {
        Some(match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
            Self::Coolant => "coolant",
            Self::Nvme => "NVMe",
            Self::Disk => "disk",
            Self::Board => "board",
            Self::Memory => "DIMM",
            Self::Nic => "NIC",
            Self::Wifi => "Wi-Fi",
            Self::Acpi => "ACPI",
            Self::Other => return None,
        })
    }

    /// Kinds that come several to a host number every row (`NVMe0`).
    #[must_use]
    pub fn always_numbered(self) -> bool {
        matches!(self, Self::Nvme | Self::Disk | Self::Memory)
    }
}

/// One discovered hwmon chip.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Chip {
    /// Canonical `hwmonN` directory, inside the sys root.
    pub dir: PathBuf,
    /// The `name` file, trimmed.
    pub name: String,
    /// Stable chip name (see the module docs).
    pub chip: String,
    /// The device's own name (`nvme0`, `0000:05:00.0`), or empty.
    pub device: String,
    /// What it measures.
    pub kind: Kind,
}

/// The hwmon `name` as a chip name: lowercase `[a-z0-9_.-]`, at most
/// [`MAX_CHIP`]; `hwmon` for an empty name.
#[must_use]
pub fn chip_name(name: &str) -> String {
    let out: String = name
        .trim()
        .chars()
        .take(MAX_CHIP)
        .map(|ch| {
            let ch = ch.to_ascii_lowercase();
            if ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '_' | '.' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() {
        "hwmon".to_owned()
    } else {
        out
    }
}

/// Every hwmon chip under `roots.sys`, named and ordered. `Err` only when
/// the sys root or `class/hwmon` cannot be read.
pub fn discover(roots: &Roots) -> Result<Vec<Chip>, String> {
    let id = SourceId::HwmonCpu;
    let root = sys_root(roots, id).map_err(|err| err.message)?;
    let class = roots.sys.join("class/hwmon");
    let entries =
        std::fs::read_dir(&class).map_err(|err| format!("read {}: {err}", class.display()))?;
    let mut found: Vec<(u32, PathBuf, String)> = Vec::new();
    for entry in entries.flatten() {
        let Some(file) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(index) = file
            .strip_prefix("hwmon")
            .filter(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|rest| rest.parse::<u32>().ok())
        else {
            continue;
        };
        let Ok(canon) = entry.path().canonicalize() else {
            continue;
        };
        if !canon.starts_with(&root) || !canon.is_dir() {
            continue;
        }
        let Some(name) = read_sensor_name(id, &root, &canon) else {
            continue;
        };
        if found.iter().any(|(_, dir, _)| *dir == canon) {
            continue;
        }
        found.push((index, canon, name));
    }
    found.sort_by_key(|(index, _, _)| *index);
    let found: Vec<(u32, PathBuf, String)> = found.into_iter().take(MAX_CHIPS).collect();
    let mut chips: Vec<Chip> = found
        .iter()
        .map(|(_, dir, name)| Chip {
            dir: dir.clone(),
            name: name.clone(),
            chip: chip_name(name),
            device: device_path(&root, dir)
                .and_then(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_owned))
                .unwrap_or_default(),
            kind: Kind::of(name),
        })
        .collect();
    let indexes: Vec<u32> = found.iter().map(|(index, _, _)| *index).collect();
    disambiguate(&root, &mut chips, &indexes);
    chips.sort_by(|a, b| (a.kind, &a.device, &a.chip).cmp(&(b.kind, &b.device, &b.chip)));
    Ok(chips)
}

/// One way to tell same-named chips apart, from the chip and its `hwmonN`.
type Suffix<'a> = dyn Fn(&Chip, u32) -> Option<String> + 'a;

/// Give chips that share a name a device suffix (module docs).
fn disambiguate(root: &Path, chips: &mut [Chip], hwmon: &[u32]) {
    let mut names: Vec<String> = chips.iter().map(|c| c.chip.clone()).collect();
    names.sort();
    names.dedup();
    for base in names {
        let group: Vec<usize> = (0..chips.len())
            .filter(|i| chips[*i].chip == base)
            .collect();
        if group.len() < 2 {
            continue;
        }
        let methods: [&Suffix<'_>; 4] = [
            &|chip, _| serial_suffix(root, &chip.dir),
            &|chip, _| pci_suffix(root, &chip.dir),
            &|chip, _| {
                Some(chip_name(&chip.device)).filter(|d| !chip.device.is_empty() && d != "hwmon")
            },
            &|_, n| Some(format!("hwmon{n}")),
        ];
        for method in methods {
            let suffixes: Vec<Option<String>> = group
                .iter()
                .map(|i| method(&chips[*i], hwmon[*i]))
                .collect();
            let mut seen: Vec<&String> = suffixes.iter().flatten().collect();
            seen.sort();
            seen.dedup();
            if seen.len() != group.len() {
                continue;
            }
            for (i, suffix) in group.iter().zip(suffixes) {
                let suffix = suffix.unwrap_or_default();
                let full = if suffix.starts_with(&base) {
                    suffix
                } else {
                    let keep = MAX_CHIP.saturating_sub(suffix.len() + 1).max(1);
                    format!("{}-{suffix}", &base[..base.len().min(keep)])
                };
                chips[*i].chip = full.chars().take(MAX_CHIP).collect();
            }
            break;
        }
    }
}

/// The chip's `device` link, resolved inside the sys root.
fn device_path(root: &Path, dir: &Path) -> Option<PathBuf> {
    match locate(SourceId::HwmonCpu, root, &dir.join("device")).ok()? {
        Located::Inside(path) => Some(path),
        Located::Outside => None,
    }
}

/// Last four letters or digits of `device/serial`, lowercased.
fn serial_suffix(root: &Path, dir: &Path) -> Option<String> {
    let path = device_path(root, dir)?.join("serial");
    let Ok(Located::Inside(path)) = locate(SourceId::HwmonCpu, root, &path) else {
        return None;
    };
    let text = std::fs::read_to_string(path).ok()?;
    let alnum: Vec<char> = text
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect();
    (alnum.len() >= 4).then(|| alnum[alnum.len() - 4..].iter().collect())
}

/// The nearest PCI address on the device path, as `bus.dev` (`.fn` when
/// the function is not 0).
fn pci_suffix(root: &Path, dir: &Path) -> Option<String> {
    let device = device_path(root, dir)?;
    device
        .ancestors()
        .filter_map(|p| p.file_name()?.to_str())
        .find_map(pci_short)
}

/// `0000:05:00.0` is `05.00`; `0000:05:00.1` is `05.00.1`.
fn pci_short(text: &str) -> Option<String> {
    let b = text.as_bytes();
    let hex = |range: std::ops::Range<usize>| b[range].iter().all(u8::is_ascii_hexdigit);
    if b.len() != 12 || b[4] != b':' || b[7] != b':' || b[10] != b'.' {
        return None;
    }
    if !hex(0..4) || !hex(5..7) || !hex(8..10) || !b[11].is_ascii_digit() {
        return None;
    }
    let short = format!("{}.{}", &text[5..7], &text[8..10]).to_ascii_lowercase();
    Some(if b[11] == b'0' {
        short
    } else {
        format!("{short}.{}", b[11] as char)
    })
}

/// A sensor index file such as `temp3_input` or `fan2_input`: `3`, `2`.
#[must_use]
pub fn input_index(file: &str, prefix: &str) -> Option<u32> {
    let rest = file.strip_prefix(prefix)?.strip_suffix("_input")?;
    if rest.is_empty() || rest.len() > 3 || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok().filter(|n| *n > 0)
}

/// Contents of `file` under `dir`, trimmed, when it resolves inside the
/// sys root and reads as UTF-8. `Err` carries the reason for the log.
pub fn read_text(root: &Path, dir: &Path, file: &str) -> Result<String, String> {
    match locate(SourceId::HwmonCpu, root, &dir.join(file)) {
        Ok(Located::Inside(path)) => std::fs::read_to_string(&path)
            .map(|text| text.trim().to_owned())
            .map_err(|err| short_error(&err)),
        Ok(Located::Outside) => Err("escapes the sys root".to_owned()),
        Err(err) => Err(err.message),
    }
}

/// `EIO`, `ENODATA` and the like, without the path.
fn short_error(err: &std::io::Error) -> String {
    match err.raw_os_error() {
        Some(5) => "EIO".to_owned(),
        Some(61) => "ENODATA".to_owned(),
        Some(code) => format!("errno {code}"),
        None => err.kind().to_string(),
    }
}

/// Printable ASCII, at most `cap` characters; other characters are `?`.
#[must_use]
pub fn printable(text: &str, cap: usize) -> String {
    text.trim()
        .chars()
        .take(cap)
        .map(|ch| {
            if ch == ' ' || ch.is_ascii_graphic() {
                ch
            } else {
                '?'
            }
        })
        .collect()
}

/// A `[temps]` / `[fans]` glob: `chip` or `chip:sensor` (module docs).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Pattern {
    chip: String,
    sensor: Option<String>,
}

impl Pattern {
    /// `None` for an invalid glob: empty, an empty side of the `:`, more
    /// than one `:`, a character that is not printable ASCII, or one of
    /// `/ [ ] { } \`, which these globs do not support.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        if text.is_empty()
            || text.chars().count() > 64
            || !text.chars().all(|ch| ch == ' ' || ch.is_ascii_graphic())
            || text.contains(['/', '[', ']', '{', '}', '\\'])
        {
            return None;
        }
        let (chip, sensor) = match text.split_once(':') {
            Some((chip, sensor)) => {
                if sensor.contains(':') || sensor.trim().is_empty() {
                    return None;
                }
                (chip, Some(sensor.to_ascii_lowercase()))
            }
            None => (text, None),
        };
        if chip.trim().is_empty() {
            return None;
        }
        Some(Self {
            chip: chip.to_ascii_lowercase(),
            sensor,
        })
    }

    /// Whether this names `chip` and, with a sensor part, one of the
    /// sensor's names.
    #[must_use]
    pub fn matches(&self, chip: &str, names: &[&str]) -> bool {
        if !glob(&self.chip, &chip.to_ascii_lowercase()) {
            return false;
        }
        match &self.sensor {
            None => true,
            Some(sensor) => names
                .iter()
                .any(|name| glob(sensor, &name.to_ascii_lowercase())),
        }
    }

    /// `true` for a `chip:sensor` pattern, more specific than a chip.
    #[must_use]
    pub fn has_sensor(&self) -> bool {
        self.sensor.is_some()
    }
}

/// `*` matches any run, `?` one character; everything else itself.
#[must_use]
pub fn glob(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let mut star: Option<(usize, usize)> = None;
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some((pi, ti));
            pi += 1;
        } else if let Some((sp, st)) = star {
            pi = sp + 1;
            ti = st + 1;
            star = Some((sp, st + 1));
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|c| *c == '*')
}

/// The `allow` / `block` verdict for one sensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Verdict {
    /// A `block` pattern matches: hidden, whatever else matches.
    Blocked,
    /// `allow` is not empty and none of it matches: hidden.
    NotAllowed,
    /// An `allow` pattern matches: shown past the built-in junk rules.
    Allowed,
    /// `allow` is empty and nothing blocks it: the built-in rules decide.
    Open,
}

/// Compiled `allow` and `block` lists.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Filter {
    allow: Vec<Pattern>,
    block: Vec<Pattern>,
}

impl Filter {
    /// Invalid patterns were refused by config validation; any left are
    /// skipped here.
    #[must_use]
    pub fn new(allow: &[String], block: &[String]) -> Self {
        Self {
            allow: allow.iter().filter_map(|p| Pattern::parse(p)).collect(),
            block: block.iter().filter_map(|p| Pattern::parse(p)).collect(),
        }
    }

    /// `block` wins; then a non-empty `allow` must match.
    #[must_use]
    pub fn verdict(&self, chip: &str, names: &[&str]) -> Verdict {
        if self.block.iter().any(|p| p.matches(chip, names)) {
            Verdict::Blocked
        } else if self.allow.is_empty() {
            Verdict::Open
        } else if self.allow.iter().any(|p| p.matches(chip, names)) {
            Verdict::Allowed
        } else {
            Verdict::NotAllowed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs_match_runs_and_single_characters() {
        assert!(glob("nct*", "nct6798"));
        assert!(glob("*", ""));
        assert!(glob("aux?in*", "auxtin3"));
        assert!(glob("*tin", "systin"));
        assert!(!glob("nct", "nct6798"));
        assert!(!glob("pch_*", "systin"));
        assert!(glob("a*b*c", "axxbyyc"));
        assert!(!glob("a*b*c", "axxbyy"));
    }

    #[test]
    fn patterns_parse_strictly() {
        for bad in [
            "", ":x", "x:", "a:b:c", "nct[0-9]", "a/b", "{a,b}", "x\\y", "é",
        ] {
            assert!(Pattern::parse(bad).is_none(), "{bad:?}");
        }
        let p = Pattern::parse("NCT6798:systin").unwrap();
        assert!(p.matches("nct6798", &["SYSTIN", "temp1"]));
        assert!(!p.matches("nct6798", &["CPUTIN", "temp2"]));
        let p = Pattern::parse("nct6798:temp1").unwrap();
        assert!(p.matches("nct6798", &["SYSTIN", "temp1"]));
        assert!(
            Pattern::parse("z53")
                .unwrap()
                .matches("z53", &["Fan speed", "fan2"])
        );
    }

    #[test]
    fn block_wins_and_a_non_empty_allow_limits() {
        let open = Filter::new(&[], &[]);
        assert_eq!(open.verdict("nvme-345a", &["Composite"]), Verdict::Open);
        let f = Filter::new(&["nct6798".into()], &["nct6798:AUXTIN*".into()]);
        assert_eq!(f.verdict("nct6798", &["SYSTIN"]), Verdict::Allowed);
        assert_eq!(f.verdict("nct6798", &["AUXTIN3"]), Verdict::Blocked);
        assert_eq!(f.verdict("k10temp", &["Tctl"]), Verdict::NotAllowed);
        let block_only = Filter::new(&[], &["nvme*".into()]);
        assert_eq!(
            block_only.verdict("nvme-345a", &["Composite"]),
            Verdict::Blocked
        );
        assert_eq!(block_only.verdict("k10temp", &["Tctl"]), Verdict::Open);
    }

    #[test]
    fn names_are_wire_safe() {
        assert_eq!(chip_name("r8169_0_500:00"), "r8169_0_500_00");
        assert_eq!(chip_name("NCT6798"), "nct6798");
        assert_eq!(chip_name(""), "hwmon");
        assert_eq!(chip_name(&"x".repeat(40)).len(), MAX_CHIP);
    }

    #[test]
    fn pci_addresses_shorten() {
        assert_eq!(pci_short("0000:05:00.0").as_deref(), Some("05.00"));
        assert_eq!(pci_short("0000:0a:1f.3").as_deref(), Some("0a.1f.3"));
        assert_eq!(pci_short("nvme0"), None);
        assert_eq!(pci_short("pci0000:00"), None);
    }

    #[test]
    fn kinds_follow_the_driver_name() {
        assert_eq!(Kind::of("k10temp"), Kind::Cpu);
        assert_eq!(Kind::of("nct6798"), Kind::Board);
        assert_eq!(Kind::of("r8169_0_500:00"), Kind::Nic);
        assert_eq!(Kind::of("iwlwifi_1"), Kind::Wifi);
        assert_eq!(Kind::of("z53"), Kind::Coolant);
        assert_eq!(Kind::of("nvme"), Kind::Nvme);
        assert_eq!(Kind::of("mystery"), Kind::Other);
    }
}
