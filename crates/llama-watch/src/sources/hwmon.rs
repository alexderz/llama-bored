//! Read-only hwmon temperatures, rooted at [`Roots`](crate::sources::Roots).
//!
//! Coolant is the `hwmonN` directory whose `name` is `z53`, `temp1_input` / 1000.
//! CPU temperature is the `k10temp` (AMD) directory's `tempN_input` whose
//! `tempN_label` is `Tctl`. Without `k10temp` it is the `coretemp` (Intel)
//! input labelled `Package id 0`. Values are millidegrees in the file.
//!
//! The directory is resolved on every call. A `hwmonN` whose `name` cannot be
//! read is skipped, so one bad directory does not hide the sensor, and a
//! renumber between calls is found without a cached index.
//!
//! File contents are read with [`std::fs::read_to_string`] only.

use std::path::{Path, PathBuf};

use crate::sources::{Roots, SourceError, SourceId};

/// Coolant temperature in °C from the `z53` hwmon `temp1_input`.
pub fn read_coolant(roots: &Roots) -> Result<f32, SourceError> {
    let id = SourceId::HwmonCoolant;
    let root = sys_root(roots, id)?;
    let dir = find_named(roots, &root, id, "z53")?;
    read_millidegree(id, &root, &dir.join("temp1_input"))
}

/// Socket energy counters from the `zenergy` hwmon, in microjoules.
///
/// The directory is the `hwmonN` whose `name` is `zenergy`. Each
/// `energyN_input` whose `energyN_label` begins with `Esocket` is returned,
/// in ascending `N`. `energy_uj` is never opened.
pub(crate) fn read_socket_energy_uj(roots: &Roots) -> Result<Vec<u64>, String> {
    // Path checks reuse the hwmon helpers. The caller does not record this id:
    // a missing zenergy is a load fallback, not a coolant or CPU-temp failure.
    let id = SourceId::HwmonCpu;
    let root = sys_root(roots, id).map_err(|err| err.message)?;
    let dir = find_named(roots, &root, id, "zenergy").map_err(|err| err.message)?;
    let entries =
        std::fs::read_dir(&dir).map_err(|err| format!("read {}: {err}", dir.display()))?;
    let mut indexes = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| format!("read {}: {err}", dir.display()))?;
        let Some(file_name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let Some(index) = energy_label_index(&file_name) else {
            continue;
        };
        let Ok(Located::Inside(path)) = locate(id, &root, &entry.path()) else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if text.trim().starts_with("Esocket") {
            indexes.push(index);
        }
    }
    if indexes.is_empty() {
        return Err(format!("no Esocket energy under {}", dir.display()));
    }
    indexes.sort_unstable();
    indexes.dedup();
    let mut counters = Vec::with_capacity(indexes.len());
    for index in indexes {
        let path = dir.join(format!("energy{index}_input"));
        counters.push(read_microjoules(id, &root, &path).map_err(|err| err.message)?);
    }
    Ok(counters)
}

fn energy_label_index(file_name: &str) -> Option<u32> {
    let rest = file_name.strip_prefix("energy")?.strip_suffix("_label")?;
    if rest.is_empty() || !rest.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

fn read_microjoules(id: SourceId, root: &Path, path: &Path) -> Result<u64, SourceError> {
    let Located::Inside(path) = locate(id, root, path)? else {
        return Err(SourceError::new(
            id,
            format!("{} escapes {}", path.display(), root.display()),
        ));
    };
    let text = std::fs::read_to_string(&path)
        .map_err(|err| SourceError::new(id, format!("read {}: {err}", path.display())))?;
    text.trim()
        .parse()
        .map_err(|_| SourceError::new(id, format!("bad energy in {}", path.display())))
}

/// CPU temperature in °C: `k10temp` `Tctl`, else `coretemp` `Package id 0`.
///
/// `coretemp` is read only when no `k10temp` directory exists. A `k10temp`
/// without `Tctl` is an error, not a reason to look elsewhere.
pub fn read_cpu_temp(roots: &Roots) -> Result<f32, SourceError> {
    let id = SourceId::HwmonCpu;
    let root = sys_root(roots, id)?;
    let (dir, label) = match find_optional(roots, &root, id, "k10temp")? {
        Some(dir) => (dir, "Tctl"),
        None => match find_optional(roots, &root, id, "coretemp")? {
            Some(dir) => (dir, "Package id 0"),
            None => {
                return Err(SourceError::new(id, "no hwmon named k10temp or coretemp"));
            }
        },
    };
    let index = label_index(&root, &dir, label)?;
    read_millidegree(id, &root, &dir.join(format!("temp{index}_input")))
}

pub(crate) fn sys_root(roots: &Roots, id: SourceId) -> Result<PathBuf, SourceError> {
    roots
        .sys
        .canonicalize()
        .map_err(|err| SourceError::new(id, format!("read {}: {err}", roots.sys.display())))
}

fn find_named(
    roots: &Roots,
    root: &Path,
    id: SourceId,
    sensor: &str,
) -> Result<PathBuf, SourceError> {
    find_optional(roots, root, id, sensor)?
        .ok_or_else(|| SourceError::new(id, format!("no hwmon named {sensor}")))
}

/// `Ok(None)` when no `hwmonN` is named `sensor`. Two or more is an error.
pub(crate) fn find_optional(
    roots: &Roots,
    root: &Path,
    id: SourceId,
    sensor: &str,
) -> Result<Option<PathBuf>, SourceError> {
    let class = roots.sys.join("class/hwmon");
    let entries = std::fs::read_dir(&class)
        .map_err(|err| SourceError::new(id, format!("read {}: {err}", class.display())))?;
    let mut found = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|err| SourceError::new(id, format!("read {}: {err}", class.display())))?;
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if !is_hwmon_index(&name) {
            continue;
        }
        // `hwmonN` is a symlink on sysfs. Follow it, then keep the directory
        // only when it still sits under the sys root. A broken link, a
        // non-directory, or a name that cannot be read is skipped.
        let Some(dir) = canonical_dir_inside(root, &entry.path()) else {
            continue;
        };
        if read_sensor_name(id, root, &dir).as_deref() == Some(sensor) {
            found.push(dir);
        }
    }
    match found.len() {
        0 => Ok(None),
        1 => Ok(Some(found.remove(0))),
        _ => Err(SourceError::new(
            id,
            format!("multiple hwmon dirs named {sensor}"),
        )),
    }
}

fn canonical_dir_inside(root: &Path, dir: &Path) -> Option<PathBuf> {
    let canon = dir.canonicalize().ok()?;
    if canon.starts_with(root) {
        Some(canon)
    } else {
        None
    }
}

fn is_hwmon_index(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("hwmon") else {
        return false;
    };
    !rest.is_empty() && rest.bytes().all(|byte| byte.is_ascii_digit())
}

pub(crate) fn read_sensor_name(id: SourceId, root: &Path, dir: &Path) -> Option<String> {
    let Located::Inside(path) = locate(id, root, &dir.join("name")).ok()? else {
        return None;
    };
    let text = std::fs::read_to_string(path).ok()?;
    let name = text.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

fn label_index(root: &Path, dir: &Path, wanted: &str) -> Result<u32, SourceError> {
    let id = SourceId::HwmonCpu;
    let entries = std::fs::read_dir(dir)
        .map_err(|err| SourceError::new(id, format!("read {}: {err}", dir.display())))?;
    let mut indexes = Vec::new();
    for entry in entries {
        let entry =
            entry.map_err(|err| SourceError::new(id, format!("read {}: {err}", dir.display())))?;
        let Some(file_name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let Some(index) = temp_label_index(&file_name) else {
            continue;
        };
        let path = entry.path();
        // A label that cannot be read, is not UTF-8, or points outside the
        // sys root is skipped. One bad file must not hide the wanted label.
        let Ok(Located::Inside(path)) = locate(id, root, &path) else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if text.trim() == wanted {
            indexes.push(index);
        }
    }
    match indexes.as_slice() {
        [index] => Ok(*index),
        [] => Err(SourceError::new(
            id,
            format!("no {wanted} label under {}", dir.display()),
        )),
        _ => Err(SourceError::new(
            id,
            format!("multiple {wanted} labels under {}", dir.display()),
        )),
    }
}

fn temp_label_index(file_name: &str) -> Option<u32> {
    let rest = file_name.strip_prefix("temp")?.strip_suffix("_label")?;
    if rest.is_empty() || !rest.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

pub(crate) enum Located {
    Inside(PathBuf),
    Outside,
}

pub(crate) fn locate(id: SourceId, root: &Path, path: &Path) -> Result<Located, SourceError> {
    let canon = path
        .canonicalize()
        .map_err(|err| SourceError::new(id, format!("read {}: {err}", path.display())))?;
    if canon.starts_with(root) {
        Ok(Located::Inside(canon))
    } else {
        Ok(Located::Outside)
    }
}

fn read_millidegree(id: SourceId, root: &Path, path: &Path) -> Result<f32, SourceError> {
    let Located::Inside(path) = locate(id, root, path)? else {
        return Err(SourceError::new(
            id,
            format!("{} escapes {}", path.display(), root.display()),
        ));
    };
    let text = std::fs::read_to_string(&path)
        .map_err(|err| SourceError::new(id, format!("read {}: {err}", path.display())))?;
    let raw: i64 = text
        .trim()
        .parse()
        .map_err(|_| SourceError::new(id, format!("bad temperature in {}", path.display())))?;
    Ok(raw as f32 / 1000.0)
}

#[cfg(test)]
mod tests {
    use super::{read_coolant, read_cpu_temp};
    use crate::sources::{Roots, SourceId};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "kraken-lcd-hwmon-unit-{label}-{}-{n}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).expect("scratch dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn roots(&self) -> Roots {
            Roots {
                proc: self.path().to_path_buf(),
                sys: self.path().to_path_buf(),
            }
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn sensor(dir: &Path, name: &str, temp1: &str) {
        std::fs::create_dir_all(dir).expect("sensor dir");
        std::fs::write(dir.join("name"), name).expect("name");
        std::fs::write(dir.join("temp1_input"), temp1).expect("temp");
    }

    #[test]
    fn hwmon_symlink_is_resolved_like_a_directory() {
        let scratch = Scratch::new("symlink");
        let class = scratch.path().join("class/hwmon");
        let target = scratch.path().join("device-z53");
        sensor(&target, "z53\n", "31000\n");
        std::fs::create_dir_all(&class).expect("class");
        std::os::unix::fs::symlink(&target, class.join("hwmon5")).expect("symlink");
        assert_eq!(read_coolant(&scratch.roots()).expect("linked z53"), 31.0);
    }

    #[test]
    fn temp_input_symlink_outside_the_sys_root_is_not_read() {
        let scratch = Scratch::new("temp-escape");
        let outside = Scratch::new("temp-outside");
        std::fs::write(outside.path().join("secret"), "99000\n").expect("secret");
        let dir = scratch.path().join("class/hwmon/hwmon5");
        std::fs::create_dir_all(&dir).expect("hwmon5");
        std::fs::write(dir.join("name"), "z53\n").expect("name");
        std::os::unix::fs::symlink(outside.path().join("secret"), dir.join("temp1_input"))
            .expect("temp symlink");
        let err = read_coolant(&scratch.roots()).expect_err("escaped temp");
        assert_eq!(err.id, SourceId::HwmonCoolant);
        assert!(err.message.contains("escapes"), "{}", err.message);
    }

    #[test]
    fn hwmon_symlink_outside_the_sys_root_is_ignored() {
        let scratch = Scratch::new("escape");
        let outside = Scratch::new("outside");
        let class = scratch.path().join("class/hwmon");
        sensor(&outside.path().join("secret"), "z53\n", "99000\n");
        std::fs::create_dir_all(&class).expect("class");
        std::os::unix::fs::symlink(outside.path().join("secret"), class.join("hwmon5"))
            .expect("symlink");
        let err = read_coolant(&scratch.roots()).expect_err("outside root");
        assert_eq!(err.id, SourceId::HwmonCoolant);
        assert!(err.message.contains("z53"), "{}", err.message);
    }

    #[test]
    fn scan_skips_a_hwmon_dir_whose_name_cannot_be_read() {
        let scratch = Scratch::new("skip");
        let class = scratch.path().join("class/hwmon");
        std::fs::create_dir_all(class.join("hwmon5")).expect("broken dir");
        sensor(&class.join("hwmon6"), "z53\n", "20000\n");
        assert_eq!(read_coolant(&scratch.roots()).expect("z53"), 20.0);
    }

    #[test]
    fn duplicate_z53_is_a_coolant_error() {
        let scratch = Scratch::new("dup");
        let class = scratch.path().join("class/hwmon");
        sensor(&class.join("hwmon2"), "z53\n", "10000\n");
        sensor(&class.join("hwmon4"), "z53\n", "20000\n");
        let err = read_coolant(&scratch.roots()).expect_err("duplicate");
        assert_eq!(err.id, SourceId::HwmonCoolant);
        assert!(err.message.contains("multiple"), "{}", err.message);
    }

    #[test]
    fn missing_sensors_use_their_source_ids() {
        let scratch = Scratch::new("missing");
        std::fs::create_dir_all(scratch.path().join("class/hwmon")).expect("class");
        let coolant = read_coolant(&scratch.roots()).expect_err("no z53");
        assert_eq!(coolant.id, SourceId::HwmonCoolant);
        assert!(coolant.message.contains("z53"), "{}", coolant.message);
        let cpu = read_cpu_temp(&scratch.roots()).expect_err("no k10temp");
        assert_eq!(cpu.id, SourceId::HwmonCpu);
        assert!(cpu.message.contains("k10temp"), "{}", cpu.message);
        assert!(cpu.message.contains("coretemp"), "{}", cpu.message);
    }

    fn labelled(dir: &Path, name: &str, temps: &[(u32, &str, &str)]) {
        std::fs::create_dir_all(dir).expect("sensor dir");
        std::fs::write(dir.join("name"), name).expect("name");
        for (index, label, milli) in temps {
            std::fs::write(dir.join(format!("temp{index}_label")), label).expect("label");
            std::fs::write(dir.join(format!("temp{index}_input")), milli).expect("input");
        }
    }

    #[test]
    fn intel_coretemp_package_is_the_cpu_temp_without_k10temp() {
        let scratch = Scratch::new("coretemp");
        labelled(
            &scratch.path().join("class/hwmon/hwmon4"),
            "coretemp\n",
            &[(2, "Core 0\n", "40000\n"), (1, "Package id 0\n", "52000\n")],
        );
        assert_eq!(read_cpu_temp(&scratch.roots()).expect("package"), 52.0);
    }

    #[test]
    fn k10temp_wins_over_coretemp() {
        let scratch = Scratch::new("both");
        labelled(
            &scratch.path().join("class/hwmon/hwmon3"),
            "k10temp\n",
            &[(1, "Tctl\n", "61000\n")],
        );
        labelled(
            &scratch.path().join("class/hwmon/hwmon4"),
            "coretemp\n",
            &[(1, "Package id 0\n", "52000\n")],
        );
        assert_eq!(read_cpu_temp(&scratch.roots()).expect("tctl"), 61.0);
    }

    #[test]
    fn coretemp_without_a_package_label_is_an_error() {
        let scratch = Scratch::new("core-only");
        labelled(
            &scratch.path().join("class/hwmon/hwmon4"),
            "coretemp\n",
            &[(2, "Core 0\n", "40000\n")],
        );
        let err = read_cpu_temp(&scratch.roots()).expect_err("no package");
        assert_eq!(err.id, SourceId::HwmonCpu);
        assert!(err.message.contains("Package id 0"), "{}", err.message);
    }

    #[test]
    fn unreadable_or_malformed_label_does_not_hide_tctl() {
        let scratch = Scratch::new("bad-label");
        let dir = scratch.path().join("class/hwmon/hwmon3");
        std::fs::create_dir_all(&dir).expect("hwmon3");
        std::fs::write(dir.join("name"), "k10temp\n").expect("name");
        // temp1 is not Tctl. A scan that aborts on a bad label, or that
        // reads the first input, would miss 77.5°C.
        std::fs::write(dir.join("temp1_input"), "10000\n").expect("temp1");
        std::fs::write(dir.join("temp1_label"), "Tccd1\n").expect("label1");
        std::fs::create_dir(dir.join("temp2_label")).expect("unreadable label");
        std::fs::write(dir.join("temp3_label"), [0xff, 0xfe]).expect("malformed label");
        std::fs::write(dir.join("temp4_input"), "77500\n").expect("temp4");
        std::fs::write(dir.join("temp4_label"), "Tctl\n").expect("tctl");
        assert_eq!(read_cpu_temp(&scratch.roots()).expect("tctl"), 77.5);
    }

    #[test]
    fn temperatures_are_signed_millidegrees() {
        let scratch = Scratch::new("signed");
        sensor(
            &scratch.path().join("class/hwmon/hwmon0"),
            "z53\n",
            "-2500\n",
        );
        assert_eq!(read_coolant(&scratch.roots()).expect("negative"), -2.5);
    }

    #[test]
    fn bad_or_missing_temp_input_is_an_error() {
        let scratch = Scratch::new("bad-temp");
        let dir = scratch.path().join("class/hwmon/hwmon5");
        sensor(&dir, "z53\n", "nope\n");
        let err = read_coolant(&scratch.roots()).expect_err("bad text");
        assert_eq!(err.id, SourceId::HwmonCoolant);
        assert!(err.message.contains("bad"), "{}", err.message);

        std::fs::remove_file(dir.join("temp1_input")).expect("drop temp");
        let err = read_coolant(&scratch.roots()).expect_err("missing temp");
        assert_eq!(err.id, SourceId::HwmonCoolant);
    }
}
