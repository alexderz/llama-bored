//! T52: the read-only `[fans]` hwmon source, on fake sys roots only.
//!
//! Fixtures live under `CARGO_TARGET_TMPDIR`. Nothing here opens the real
//! `/sys`, and the source under test never writes.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use llama_core::sample::{AiState, LlamaView};
use llama_watch::collector::WatchCollector;
use llama_watch::config::{Config, Fans, ValidWatchConfig};
use llama_watch::sources::Roots;
use llama_watch::sources::fans::{FanPanel, FanReading, FanSource, RESCAN, mode_word};
use llama_watch::sources::gpu::{GpuBackend, GpuError};

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "t52-fans-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(path.join("class/hwmon")).expect("class dir");
        Self(path)
    }

    fn roots(&self) -> Roots {
        Roots {
            proc: self.0.clone(),
            sys: self.0.clone(),
        }
    }

    fn hwmon(&self, n: u32) -> PathBuf {
        self.0.join(format!("class/hwmon/hwmon{n}"))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = Command::new("chmod")
            .arg("-R")
            .arg("u+w")
            .arg(&self.0)
            .status();
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Reference board: fans 2, 3, 5, 6 on the nct6798, all SmartFan (5).
const REF_BOARD: [(u32, &str, &str); 4] = [
    (2, "1939", "224"),
    (3, "1877", "224"),
    (5, "3026", "162"),
    (6, "1272", "255"),
];

fn chip(dir: &Path, name: &str, fans: &[(u32, &str, &str)]) {
    std::fs::create_dir_all(dir).expect("hwmon dir");
    std::fs::write(dir.join("name"), format!("{name}\n")).expect("name");
    for (n, rpm, pwm) in fans {
        std::fs::write(dir.join(format!("fan{n}_input")), format!("{rpm}\n")).expect("fan");
        std::fs::write(dir.join(format!("pwm{n}")), format!("{pwm}\n")).expect("pwm");
        std::fs::write(dir.join(format!("pwm{n}_enable")), "5\n").expect("enable");
    }
}

fn fans_config(hwmon: &str, channels: &[u32], labels: Option<&[&str]>) -> Fans {
    Fans {
        enabled: true,
        hwmon: hwmon.to_owned(),
        channels: channels.to_vec(),
        labels: labels.map(|l| l.iter().map(|s| (*s).to_owned()).collect()),
        ..Fans::default()
    }
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<String>>>);

impl Capture {
    fn lines(&self) -> Vec<String> {
        self.0.lock().unwrap_or_else(|err| err.into_inner()).clone()
    }
}

impl llama_core::log::Sink for Capture {
    fn write_line(&mut self, line: &str) {
        self.0
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(line.to_owned());
    }
}

fn ref_panel(labels: [&str; 4]) -> FanPanel {
    FanPanel {
        chip: "nct6798".to_owned(),
        present: true,
        fans: REF_BOARD
            .iter()
            .zip(labels)
            .map(|((n, rpm, pwm), label)| FanReading {
                chip: "nct6798".to_owned(),
                channel: *n,
                label: label.to_owned(),
                rpm: Some(rpm.parse().unwrap()),
                pwm: Some(pwm.parse().unwrap()),
                mode: Some(5),
            })
            .collect(),
    }
}

#[test]
fn disabled_fans_build_no_source() {
    assert!(FanSource::new(&Fans::default()).is_none());
}

#[test]
fn reads_rpm_pwm_and_mode_for_each_configured_channel() {
    let scratch = Scratch::new("refboard");
    chip(&scratch.hwmon(8), "nct6798", &REF_BOARD);
    let mut source = FanSource::new(&fans_config("nct6798", &[2, 3, 5, 6], None)).unwrap();
    let mut log = Capture::default();
    let panel = source.read(&scratch.roots(), Instant::now(), &mut log);
    assert_eq!(panel, ref_panel(["fan2", "fan3", "fan5", "fan6"]));
}

#[test]
fn device_is_found_by_name_whatever_its_hwmon_number() {
    // Other chips sit on the numbers nct6798 had on other boots.
    for n in [0u32, 3, 8, 11] {
        let scratch = Scratch::new(&format!("number-{n}"));
        for other in [0u32, 3, 8, 11].into_iter().filter(|o| *o != n) {
            chip(&scratch.hwmon(other), "decoy", &[(2, "1", "1")]);
        }
        chip(&scratch.hwmon(n), "nct6798", &REF_BOARD);
        let mut source = FanSource::new(&fans_config("nct6798", &[2, 3, 5, 6], None)).unwrap();
        let panel = source.read(&scratch.roots(), Instant::now(), &mut Capture::default());
        assert!(panel.present, "nct6798 at hwmon{n} not found");
        assert_eq!(panel.fans[0].rpm, Some(1939), "hwmon{n}: {panel:?}");
    }
}

#[test]
fn a_renumber_between_reads_is_followed_by_name() {
    let scratch = Scratch::new("renumber");
    chip(&scratch.hwmon(8), "nct6798", &REF_BOARD);
    let mut source = FanSource::new(&fans_config("nct6798", &[2], None)).unwrap();
    let t0 = Instant::now();
    let mut log = Capture::default();
    assert_eq!(
        source.read(&scratch.roots(), t0, &mut log).fans[0].rpm,
        Some(1939)
    );
    // "Reboot": hwmon8 is now another chip and nct6798 moved to hwmon9.
    std::fs::remove_dir_all(scratch.hwmon(8)).unwrap();
    chip(&scratch.hwmon(8), "asus", &[(2, "7", "7")]);
    chip(&scratch.hwmon(9), "nct6798", &[(2, "1500", "100")]);
    let panel = source.read(&scratch.roots(), t0 + Duration::from_millis(100), &mut log);
    assert!(panel.present);
    assert_eq!(panel.fans[0].rpm, Some(1500), "{panel:?}");
}

#[test]
fn no_match_is_absent_with_one_log_line_and_a_slow_rescan() {
    let scratch = Scratch::new("absent");
    chip(&scratch.hwmon(2), "k10temp", &[]);
    let mut source = FanSource::new(&fans_config("nct6798", &[2, 3], None)).unwrap();
    let log = Capture::default();
    let mut sink = log.clone();
    let t0 = Instant::now();
    for tick in 0..20u64 {
        let panel = source.read(
            &scratch.roots(),
            t0 + Duration::from_millis(100 * tick),
            &mut sink,
        );
        assert!(!panel.present);
        assert!(panel.fans.is_empty());
        assert_eq!(panel.chip, "nct6798");
    }
    let lines = log.lines();
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].contains("no hwmon named nct6798"), "{lines:?}");
    // The chip appears (module loaded late). Found on the next rescan.
    chip(&scratch.hwmon(8), "nct6798", &REF_BOARD);
    let early = source.read(&scratch.roots(), t0 + Duration::from_secs(3), &mut sink);
    assert!(!early.present, "no scan before RESCAN");
    let later = source.read(&scratch.roots(), t0 + RESCAN, &mut sink);
    assert!(later.present);
    let lines = log.lines();
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(lines[1].contains("found"), "{lines:?}");
}

#[test]
fn two_matches_are_absent_not_a_guess() {
    let scratch = Scratch::new("dup");
    chip(&scratch.hwmon(4), "nct6798", &REF_BOARD);
    chip(&scratch.hwmon(8), "nct6798", &REF_BOARD);
    let mut source = FanSource::new(&fans_config("nct6798", &[2], None)).unwrap();
    let log = Capture::default();
    let mut sink = log.clone();
    let panel = source.read(&scratch.roots(), Instant::now(), &mut sink);
    assert!(!panel.present);
    let lines = log.lines();
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].contains("multiple"), "{lines:?}");
}

#[test]
fn a_missing_or_bad_file_blanks_only_that_value() {
    let scratch = Scratch::new("partial");
    let dir = scratch.hwmon(8);
    chip(&dir, "nct6798", &REF_BOARD);
    std::fs::remove_file(dir.join("pwm3")).unwrap();
    std::fs::write(dir.join("fan5_input"), "garbage\n").unwrap();
    let mut source = FanSource::new(&fans_config("nct6798", &[2, 3, 5, 7], None)).unwrap();
    let panel = source.read(&scratch.roots(), Instant::now(), &mut Capture::default());
    assert!(panel.present);
    assert_eq!(panel.fans[1].pwm, None);
    assert_eq!(panel.fans[1].rpm, Some(1877));
    assert_eq!(panel.fans[2].rpm, None);
    assert_eq!(panel.fans[2].pwm, Some(162));
    // Channel 7 is not wired: every value is absent, the row stays.
    assert_eq!(panel.fans[3].channel, 7);
    assert_eq!(
        (panel.fans[3].rpm, panel.fans[3].pwm, panel.fans[3].mode),
        (None, None, None)
    );
}

#[test]
fn labels_come_from_config_and_default_to_fan_n() {
    let scratch = Scratch::new("labels");
    chip(&scratch.hwmon(8), "nct6798", &REF_BOARD);
    let config = fans_config(
        "nct6798",
        &[2, 3, 5, 6],
        Some(&["front1", "", "re\x1bar", "top"]),
    );
    let mut source = FanSource::new(&config).unwrap();
    let panel = source.read(&scratch.roots(), Instant::now(), &mut Capture::default());
    let labels: Vec<&str> = panel.fans.iter().map(|f| f.label.as_str()).collect();
    assert_eq!(labels, ["front1", "fan3", "re?ar", "top"]);
}

#[test]
fn mode_words_are_generic() {
    assert_eq!(mode_word(Some(0)), "full");
    assert_eq!(mode_word(Some(1)), "manual");
    assert_eq!(mode_word(Some(2)), "auto");
    assert_eq!(mode_word(Some(5)), "auto");
    assert_eq!(mode_word(None), "--");
}

fn euid_is_root() -> bool {
    Command::new("id")
        .arg("-u")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .is_some_and(|text| text.trim() == "0")
}

fn tree_state(dir: &Path) -> Vec<(PathBuf, Vec<u8>, SystemTime)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let meta = std::fs::metadata(&path).unwrap();
        out.push((
            path.clone(),
            std::fs::read(&path).unwrap(),
            meta.modified().unwrap(),
        ));
    }
    out.sort();
    out
}

/// The fan files are writable on real sysfs. On a read-only fake the source
/// still works, and after many reads no file content or mtime has changed.
#[test]
fn read_only_fixture_reads_and_nothing_changes() {
    if euid_is_root() {
        eprintln!("skipping: euid 0 ignores chmod a-w");
        return;
    }
    let scratch = Scratch::new("readonly");
    let dir = scratch.hwmon(8);
    chip(&dir, "nct6798", &REF_BOARD);
    let before = tree_state(&dir);
    let status = Command::new("chmod")
        .arg("-R")
        .arg("a-w")
        .arg(&scratch.0)
        .status()
        .unwrap();
    assert!(status.success());
    let mut source = FanSource::new(&fans_config("nct6798", &[2, 3, 5, 6], None)).unwrap();
    let t0 = Instant::now();
    for tick in 0..30u64 {
        let panel = source.read(
            &scratch.roots(),
            t0 + Duration::from_millis(100 * tick),
            &mut Capture::default(),
        );
        assert_eq!(panel, ref_panel(["fan2", "fan3", "fan5", "fan6"]));
    }
    assert_eq!(tree_state(&dir), before);
}

struct SteadyGpu;

impl GpuBackend for SteadyGpu {
    fn init(&mut self) -> Result<(), GpuError> {
        Ok(())
    }
    fn util_pct(&mut self) -> Result<f32, GpuError> {
        Ok(10.0)
    }
    fn temp_c(&mut self) -> Result<f32, GpuError> {
        Ok(40.0)
    }
    fn disconnect(&mut self) {}
}

fn valid(toml: &str) -> ValidWatchConfig {
    let dir = Scratch::new("cfg");
    let path = dir.0.join("watch.toml");
    std::fs::write(&path, toml).unwrap();
    Config::load_validated(&path, 32).expect("valid config")
}

#[test]
fn collector_carries_fans_beside_the_snapshot_only_when_enabled() {
    let scratch = Scratch::new("collector");
    chip(&scratch.hwmon(8), "nct6798", &REF_BOARD);
    let view = LlamaView {
        ai: AiState::Idle,
        models: Vec::new(),
        decoded_total: None,
        prompt_total: None,
    };
    let on = valid(
        "[fans]\nenabled = true\nhwmon = \"nct6798\"\nchannels = [2, 3, 5, 6]\n\
         labels = [\"front1\", \"front2\", \"rear\", \"top\"]\n",
    );
    let mut collector = WatchCollector::new(scratch.roots(), SteadyGpu, &on, Capture::default());
    let sample = collector.sample(Instant::now(), SystemTime::UNIX_EPOCH, &view);
    assert_eq!(
        sample.fans,
        Some(ref_panel(["front1", "front2", "rear", "top"]))
    );

    let off = valid("");
    let mut collector = WatchCollector::new(scratch.roots(), SteadyGpu, &off, Capture::default());
    let sample = collector.sample(Instant::now(), SystemTime::UNIX_EPOCH, &view);
    assert_eq!(sample.fans, None);
}
