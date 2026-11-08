//! Optional per-GPU blower rpm. Read only.
//!
//! `[gpu.fan]` maps a device selector to `"<hwmon name>/fanN"`. The chip is
//! the hwmon directory whose `name` file matches, never `hwmonN` itself.
//! Only `fanN_input` is opened.

use std::collections::BTreeMap;
use std::path::Path;

use crate::collector::GpuDeviceTelemetry;
use crate::config::{self, GpuConfig};
use crate::sources::hwmon::{self, Located};
use crate::sources::{Roots, SourceId};

/// Fill [`GpuDeviceTelemetry::fan_rpm`] from `[gpu.fan]`.
///
/// A missing chip, a duplicate name, or an unreadable file leaves `None`.
/// Nothing here writes sysfs.
pub(crate) fn apply(roots: &Roots, config: &GpuConfig, devices: &mut [GpuDeviceTelemetry]) {
    if config.fan.is_empty() || devices.is_empty() {
        return;
    }
    let id = SourceId::Gpu;
    let Ok(root) = hwmon::sys_root(roots, id) else {
        return;
    };
    let mut dirs: BTreeMap<String, Option<std::path::PathBuf>> = BTreeMap::new();
    for device in devices.iter_mut() {
        let Some(spec) = config.fan_spec(&device.identity) else {
            continue;
        };
        let Some((chip, channel)) = config::parse_gpu_fan(spec) else {
            continue;
        };
        let dir = dirs
            .entry(chip.clone())
            .or_insert_with(|| hwmon::find_optional(roots, &root, id, &chip).ok().flatten());
        let Some(dir) = dir.clone() else {
            continue;
        };
        device.fan_rpm = read_input(&root, &dir, channel);
    }
}

/// `fan{channel}_input` as rpm. `None` when the file is missing or not a
/// non-negative integer. `pwm*` is never opened.
fn read_input(root: &Path, dir: &Path, channel: u32) -> Option<u32> {
    let path = dir.join(format!("fan{channel}_input"));
    let Ok(Located::Inside(path)) = hwmon::locate(SourceId::Gpu, root, &path) else {
        return None;
    };
    let text = std::fs::read_to_string(path).ok()?;
    let rpm: u64 = text.trim().parse().ok()?;
    u32::try_from(rpm).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::GpuExtra;
    use crate::sources::gpu::GpuDeviceIdentity;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new(label: &str) -> Self {
            let n = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = std::env::temp_dir().join(format!("der674-fan-{label}-{n}"));
            std::fs::create_dir_all(&path).expect("scratch");
            Self(path)
        }

        fn roots(&self) -> Roots {
            Roots {
                proc: self.0.clone(),
                sys: self.0.clone(),
            }
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn chip(scratch: &Scratch, index: u32, name: &str, fan: &str, rpm: &str) {
        let dir = scratch.0.join(format!("class/hwmon/hwmon{index}"));
        std::fs::create_dir_all(&dir).expect("hwmon");
        std::fs::write(dir.join("name"), name).expect("name");
        std::fs::write(dir.join(fan), rpm).expect("fan");
    }

    fn device(index: u32, pci: &str) -> GpuDeviceTelemetry {
        GpuDeviceTelemetry {
            identity: GpuDeviceIdentity {
                index,
                uuid: None,
                pci_bus_id: Some(pci.to_owned()),
                name: None,
            },
            label: pci.to_owned(),
            util_pct: None,
            temp_c: None,
            extra: GpuExtra {
                vram_used: None,
                vram_total: None,
                power_mw: None,
                power_limit_mw: None,
            },
            fan_rpm: None,
        }
    }

    #[test]
    fn fan_rpm_follows_the_chip_name_not_the_hwmon_number() {
        let scratch = Scratch::new("by-name");
        chip(&scratch, 1, "nct6775\n", "fan1_input", "1\n");
        chip(&scratch, 9, "nct6798\n", "fan1_input", "900\n");
        std::fs::write(scratch.0.join("class/hwmon/hwmon9/pwm1"), "128\n").expect("pwm");
        let mut cfg = GpuConfig::default();
        cfg.fan.insert("0000:05:00.0".into(), "nct6798/fan1".into());
        let mut devices = vec![device(0, "0000:05:00.0"), device(1, "0000:0b:00.0")];
        apply(&scratch.roots(), &cfg, &mut devices);
        assert_eq!(devices[0].fan_rpm, Some(900));
        assert_eq!(devices[1].fan_rpm, None);
        let pwm = std::fs::read_to_string(scratch.0.join("class/hwmon/hwmon9/pwm1")).unwrap();
        assert_eq!(pwm, "128\n", "pwm is not written");
    }

    #[test]
    fn two_chips_with_the_same_name_leave_the_fan_unset() {
        let scratch = Scratch::new("dup");
        chip(&scratch, 2, "nct6798\n", "fan1_input", "100\n");
        chip(&scratch, 3, "nct6798\n", "fan1_input", "200\n");
        let mut cfg = GpuConfig::default();
        cfg.fan.insert("0".into(), "nct6798/fan1".into());
        let mut devices = vec![device(0, "0000:05:00.0")];
        devices[0].identity.pci_bus_id = None;
        apply(&scratch.roots(), &cfg, &mut devices);
        assert_eq!(devices[0].fan_rpm, None);
    }

    #[test]
    fn a_missing_chip_or_input_leaves_the_fan_unset() {
        let scratch = Scratch::new("missing");
        chip(&scratch, 4, "nct6798\n", "fan2_input", "700\n");
        let mut cfg = GpuConfig::default();
        cfg.fan.insert("0".into(), "nct6798/fan1".into());
        let mut devices = vec![device(0, "0000:05:00.0")];
        devices[0].identity.pci_bus_id = None;
        apply(&scratch.roots(), &cfg, &mut devices);
        assert_eq!(devices[0].fan_rpm, None);
        cfg.fan.insert("0".into(), "no-such/fan1".into());
        apply(&scratch.roots(), &cfg, &mut devices);
        assert_eq!(devices[0].fan_rpm, None);
    }
}
