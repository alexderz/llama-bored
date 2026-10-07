//! #74: TEMPS discovery and the FANS discovery, on a fake sysfs tree.
//!
//! `fixtures/sys-hwmon` is a reference host with invented values: two NVMe
//! drives of the same name, a Super-I/O chip with unwired and duplicate
//! inputs, a CPU, a NIC, Wi-Fi and an AIO cooler. Each test copies it under
//! `CARGO_TARGET_TMPDIR` (symlinks kept) and changes the copy. Nothing here
//! opens the real `/sys`, and the sources under test never write.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use llama_watch::config::{Config, Fans, Temps};
use llama_watch::sources::Roots;
use llama_watch::sources::chips::Kind;
use llama_watch::sources::fans::FanSource;
use llama_watch::sources::temps::{
    self, Level, READ_EVERY, REDISCOVER, STUCK_AFTER, TempLook, TempReading, TempSource,
};

struct Scratch(PathBuf);

impl Scratch {
    /// A copy of `fixtures/sys-hwmon`.
    fn reference(label: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "t74-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/sys-hwmon");
        copy_tree(&fixture, &path);
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

    fn set(&self, n: u32, file: &str, text: &str) {
        std::fs::write(self.hwmon(n).join(file), format!("{text}\n")).expect("set");
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("dir");
    for entry in std::fs::read_dir(from).expect("read fixture") {
        let entry = entry.expect("entry");
        let target = to.join(entry.file_name());
        let kind = entry.file_type().expect("type");
        if kind.is_symlink() {
            let link = std::fs::read_link(entry.path()).expect("link");
            std::os::unix::fs::symlink(link, &target).expect("symlink");
        } else if kind.is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).expect("copy");
        }
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

fn temps_from(toml: &str) -> Temps {
    let cfg = Config::from_toml(toml).expect("toml");
    cfg.validate(32).expect("valid");
    cfg.temps
}

fn shown(readings: &[TempReading]) -> Vec<(String, String, i32)> {
    readings
        .iter()
        .map(|r| (r.chip.clone(), r.sensor.clone(), r.tenths))
        .collect()
}

fn names(readings: &[TempReading]) -> Vec<String> {
    readings
        .iter()
        .map(|r| format!("{}:{}", r.chip, r.sensor))
        .collect()
}

const GPU_C: Option<f32> = Some(71.2);

fn read_once(scratch: &Scratch, temps: &Temps) -> Vec<TempReading> {
    let mut source = TempSource::new(temps).expect("enabled");
    source.read(
        &scratch.roots(),
        Instant::now(),
        GPU_C,
        &mut Capture::default(),
    )
}

#[test]
fn reference_host_is_discovered_named_ordered_and_filtered() {
    let scratch = Scratch::reference("reference");
    let got = read_once(&scratch, &Temps::default());
    let want: Vec<(String, String, i32)> = [
        ("k10temp", "Tctl", 683),
        ("k10temp", "Tccd1", 638),
        ("k10temp", "Tccd2", 620),
        ("gpu", "gpu", 712),
        ("z53", "Coolant temp", 402),
        ("nvme-317k", "Composite", 499),
        ("nvme-317k", "Sensor 1", 499),
        ("nvme-317k", "Sensor 2", 669),
        ("nvme-842p", "Composite", 569),
        ("nvme-842p", "Sensor 1", 569),
        ("nvme-842p", "Sensor 2", 659),
        ("nct6798", "SYSTIN", 505),
        ("nct6798", "CPUTIN", 505),
        ("nct6798", "AUXTIN0", 270),
        ("nct6798", "AUXTIN1", 715),
        ("nct6798", "AUXTIN2", 155),
        ("nct6798", "AUXTIN3", 270),
        ("nct6798", "AUXTIN4", 500),
        ("r8169_0_500_00", "temp1", 510),
        ("iwlwifi_1", "temp1", 390),
    ]
    .iter()
    .map(|(c, s, t)| ((*c).to_owned(), (*s).to_owned(), *t))
    .collect();
    assert_eq!(shown(&got), want);
    let kinds: Vec<Kind> = got.iter().map(|r| r.kind).collect();
    assert!(kinds.windows(2).all(|w| w[0] <= w[1]), "{kinds:?}");
    // The chip's own limits ride along for the colours.
    let composite = &got[5];
    assert_eq!(
        (composite.max_tenths, composite.crit_tenths),
        (Some(849), Some(849))
    );
    assert_eq!(got[18].crit_tenths, Some(1200));
}

#[test]
fn super_io_defaults_drop_pch_and_the_cpu_duplicates() {
    let scratch = Scratch::reference("defaults");
    let on = names(&read_once(&scratch, &Temps::default()));
    for gone in ["PECI", "SMBUSMASTER", "TSI0", "TSI1", "PCH_"] {
        assert!(!on.iter().any(|n| n.contains(gone)), "{gone} shown: {on:?}");
    }
    // Without the defaults the duplicates show; PCH reads 0 and stays out.
    let off = names(&read_once(
        &scratch,
        &temps_from("[temps]\ndefaults = false\n"),
    ));
    for back in [
        "nct6798:PECI Agent 0 Calibration",
        "nct6798:SMBUSMASTER 1",
        "nct6798:TSI0_TEMP",
        "nct6798:TSI1_TEMP",
    ] {
        assert!(off.iter().any(|n| n == back), "{back} missing: {off:?}");
    }
    assert!(!off.iter().any(|n| n.contains("PCH_")), "{off:?}");
}

#[test]
fn bogus_values_are_dropped_whatever_allow_says() {
    let scratch = Scratch::reference("bogus");
    scratch.set(2, "temp3_input", "-62000");
    scratch.set(2, "temp4_input", "127000");
    scratch.set(2, "temp5_input", "4999");
    scratch.set(2, "temp6_input", "5000");
    scratch.set(2, "temp7_input", "126999");
    let allow_all = temps_from("[temps]\nallow = [\"nct6798\"]\n");
    for temps in [Temps::default(), allow_all] {
        let got = names(&read_once(&scratch, &temps));
        for gone in ["AUXTIN0", "AUXTIN1", "AUXTIN2", "PCH_CPU_TEMP"] {
            assert!(!got.iter().any(|n| n.ends_with(gone)), "{gone}: {got:?}");
        }
        for kept in ["nct6798:AUXTIN3", "nct6798:AUXTIN4"] {
            assert!(got.iter().any(|n| n == kept), "{kept}: {got:?}");
        }
    }
}

#[test]
fn allow_limits_and_forces_past_the_defaults_and_block_wins() {
    let scratch = Scratch::reference("allow");
    let got = names(&read_once(
        &scratch,
        &temps_from(
            "[temps]\nallow = [\"k10temp:Tctl\", \"nct6798:PECI*\", \"NVME*:composite\"]\n",
        ),
    ));
    assert_eq!(
        got,
        [
            "k10temp:Tctl",
            "nvme-317k:Composite",
            "nvme-842p:Composite",
            "nct6798:PECI Agent 0 Calibration"
        ]
    );
    // block beats allow, and the input name matches like the label.
    let got = names(&read_once(
        &scratch,
        &temps_from(
            "[temps]\nallow = [\"nvme*\", \"gpu\"]\nblock = [\"nvme-842p\", \"nvme-317k:temp3\"]\n",
        ),
    ));
    assert_eq!(
        got,
        ["gpu:gpu", "nvme-317k:Composite", "nvme-317k:Sensor 1"]
    );
    // Block alone keeps everything else.
    let got = names(&read_once(
        &scratch,
        &temps_from("[temps]\nblock = [\"nct*\", \"gpu\", \"*:Sensor ?\"]\n"),
    ));
    assert_eq!(got.len(), 8, "{got:?}");
    assert!(
        got.iter()
            .all(|n| !n.starts_with("nct") && !n.contains("Sensor"))
    );
}

#[test]
fn same_named_chips_fall_back_from_serial_to_pci_address() {
    let scratch = Scratch::reference("dups");
    for n in [0, 1] {
        std::fs::remove_file(scratch.hwmon(n).join("device/serial")).expect("serial");
    }
    let got = shown(&read_once(&scratch, &Temps::default()));
    let chips: Vec<&str> = got
        .iter()
        .filter(|(c, _, _)| c.starts_with("nvme"))
        .map(|(c, _, _)| c.as_str())
        .collect();
    assert_eq!(
        chips,
        ["nvme-01.00"; 3]
            .iter()
            .chain(&["nvme-02.00"; 3])
            .copied()
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_read_error_skips_the_input_with_one_line_per_transition() {
    let scratch = Scratch::reference("eio");
    let mut source = TempSource::new(&Temps::default()).unwrap();
    let log = Capture::default();
    let mut sink = log.clone();
    let t0 = Instant::now();
    let file = scratch.hwmon(0).join("temp3_input");
    // A directory where the file was: the read fails, as EIO would.
    std::fs::remove_file(&file).unwrap();
    std::fs::create_dir(&file).unwrap();
    for tick in 0..5u32 {
        let got = names(&source.read(&scratch.roots(), t0 + READ_EVERY * tick, None, &mut sink));
        assert!(!got.contains(&"nvme-317k:Sensor 2".to_owned()), "{got:?}");
        assert!(got.contains(&"nvme-317k:Sensor 1".to_owned()));
    }
    let warnings: Vec<String> = log
        .lines()
        .into_iter()
        .filter(|l| l.contains("unreadable"))
        .collect();
    assert_eq!(warnings.len(), 1, "{:?}", log.lines());
    assert!(warnings[0].contains("nvme-317k temp3"), "{warnings:?}");
    std::fs::remove_dir(&file).unwrap();
    std::fs::write(&file, "66000\n").unwrap();
    let got = source.read(&scratch.roots(), t0 + READ_EVERY * 6, None, &mut sink);
    assert!(names(&got).contains(&"nvme-317k:Sensor 2".to_owned()));
    assert!(
        log.lines().iter().any(|l| l.contains("readable again")),
        "{:?}",
        log.lines()
    );
}

#[test]
fn values_are_read_at_most_once_a_second_and_chips_found_every_30_s() {
    let scratch = Scratch::reference("rate");
    let mut source = TempSource::new(&Temps::default()).unwrap();
    let mut sink = Capture::default();
    let t0 = Instant::now();
    let tctl = |got: &[TempReading]| got.iter().find(|r| r.sensor == "Tctl").map(|r| r.tenths);
    assert_eq!(
        tctl(&source.read(&scratch.roots(), t0, None, &mut sink)),
        Some(683)
    );
    scratch.set(3, "temp1_input", "70000");
    let early = source.read(
        &scratch.roots(),
        t0 + Duration::from_millis(900),
        None,
        &mut sink,
    );
    assert_eq!(tctl(&early), Some(683), "re-read before a second");
    assert_eq!(
        tctl(&source.read(&scratch.roots(), t0 + READ_EVERY, None, &mut sink)),
        Some(700)
    );
    // Hotplug: Wi-Fi goes away; it stays until the next discovery (its
    // reads now fail and skip it), and comes back on the one after.
    let link = scratch.hwmon(6);
    let target = std::fs::read_link(&link).unwrap();
    std::fs::remove_file(&link).unwrap();
    let gone = source.read(&scratch.roots(), t0 + REDISCOVER, None, &mut sink);
    assert!(!names(&gone).iter().any(|n| n.starts_with("iwlwifi")));
    std::os::unix::fs::symlink(target, &link).unwrap();
    let back = source.read(&scratch.roots(), t0 + REDISCOVER * 2, None, &mut sink);
    assert!(names(&back).iter().any(|n| n.starts_with("iwlwifi")));
}

#[test]
fn an_input_that_never_moves_while_its_chip_does_is_hidden_after_ten_minutes() {
    let scratch = Scratch::reference("stuck");
    let mut sink = Capture::default();
    let t0 = Instant::now();
    let mut plain = TempSource::new(&Temps::default()).unwrap();
    let allowed = temps_from("[temps]\nallow = [\"nct6798\", \"k10temp\"]\n");
    let mut forced = TempSource::new(&allowed).unwrap();
    let mut at = |source: &mut TempSource, t: Duration| {
        names(&source.read(&scratch.roots(), t0 + t, None, &mut sink))
    };
    assert!(at(&mut plain, Duration::ZERO).contains(&"nct6798:AUXTIN0".to_owned()));
    at(&mut forced, Duration::ZERO);
    // SYSTIN moves; AUXTIN0 and AUXTIN3 hold 27.0.
    scratch.set(2, "temp1_input", "51000");
    let before = at(&mut plain, STUCK_AFTER - Duration::from_secs(1));
    assert!(
        before.contains(&"nct6798:AUXTIN0".to_owned()),
        "not before ten minutes"
    );
    scratch.set(2, "temp1_input", "51500");
    let after = at(&mut plain, STUCK_AFTER + Duration::from_secs(1));
    for gone in ["nct6798:AUXTIN0", "nct6798:AUXTIN3"] {
        assert!(!after.contains(&gone.to_owned()), "{gone}: {after:?}");
    }
    assert!(after.contains(&"nct6798:SYSTIN".to_owned()));
    // k10temp did not move at all: nothing on it is "stuck".
    assert!(after.contains(&"k10temp:Tctl".to_owned()));
    // allow names it: shown anyway.
    at(&mut forced, STUCK_AFTER - Duration::from_secs(1));
    let kept = at(&mut forced, STUCK_AFTER + Duration::from_secs(1));
    assert!(kept.contains(&"nct6798:AUXTIN0".to_owned()), "{kept:?}");
    // It moves: back at once.
    scratch.set(2, "temp3_input", "27500");
    let moved = at(&mut plain, STUCK_AFTER + Duration::from_secs(3));
    assert!(moved.contains(&"nct6798:AUXTIN0".to_owned()), "{moved:?}");
}

#[test]
fn disabled_temps_build_no_source() {
    assert!(TempSource::new(&temps_from("[temps]\nenabled = false\n")).is_none());
    assert!(
        TempSource::new(&Temps::default()).is_some(),
        "on by default"
    );
}

fn panel_rows(temps: &Temps, scratch: &Scratch) -> Vec<String> {
    let readings = read_once(scratch, temps);
    let panel = temps::panel(&readings, &TempLook::new(temps));
    panel
        .groups
        .iter()
        .map(|g| {
            let items: Vec<String> = g
                .items
                .iter()
                .map(|i| {
                    let value = format!("{}", (f64::from(i.tenths) / 10.0).round());
                    if i.label.is_empty() {
                        value
                    } else {
                        format!("{} {value}", i.label)
                    }
                })
                .collect();
            format!("{} | {}", g.name, items.join(" · "))
        })
        .collect()
}

#[test]
fn panel_groups_by_device_with_short_labels() {
    let scratch = Scratch::reference("panel");
    assert_eq!(
        panel_rows(&Temps::default(), &scratch),
        [
            "CPU | Tctl 68 · CCD1 64 · CCD2 62",
            "GPU | 71",
            "coolant | 40",
            "NVMe0 | 50 · s1 50 · s2 67",
            "NVMe1 | 57 · s1 57 · s2 66",
            "board | sys 51 · cpu 51 · aux0 27 · aux1 72 · aux2 16 · aux3 27 · aux4 50",
            "NIC | 51",
            "Wi-Fi | 39",
        ]
    );
    let renamed = temps_from(
        "[temps]\nrename = { \"nct6798:SYSTIN\" = \"board\", \"nct6798:CPUTIN\" = \"socket\", \
         \"nvme-842p\" = \"scratch\", \"iwlwifi_1\" = \"radio\" }\n",
    );
    let rows = panel_rows(&renamed, &scratch);
    assert!(
        rows.contains(&"NVMe0 | 50 · s1 50 · s2 67".to_owned()),
        "{rows:?}"
    );
    assert!(
        rows.contains(&"scratch | 57 · s1 57 · s2 66".to_owned()),
        "{rows:?}"
    );
    assert!(
        rows.iter()
            .any(|r| r.starts_with("board | 51 · socket 51 · aux0")),
        "{rows:?}"
    );
    assert!(rows.contains(&"radio | 39".to_owned()), "{rows:?}");
}

#[test]
fn colours_come_from_config_then_the_chip_then_defaults() {
    let scratch = Scratch::reference("levels");
    scratch.set(3, "temp1_input", "91000"); // Tctl: default crit 90
    scratch.set(3, "temp3_input", "80000"); // Tccd1: default warn 80
    scratch.set(2, "temp1_input", "80000"); // SYSTIN: its max 80
    scratch.set(0, "temp1_input", "84850"); // Composite: its crit 84.85
    scratch.set(0, "temp3_input", "81000"); // Sensor 2: no limits, warn 80
    scratch.set(5, "temp1_input", "46000"); // coolant: warn 45
    let level = |temps: &Temps, chip: &str, sensor: &str| {
        let readings = read_once(&scratch, temps);
        let look = TempLook::new(temps);
        let one: Vec<TempReading> = readings
            .into_iter()
            .filter(|r| r.chip == chip && r.sensor == sensor)
            .collect();
        temps::panel(&one, &look).groups[0].items[0].level
    };
    let d = Temps::default();
    assert_eq!(level(&d, "k10temp", "Tctl"), Level::Crit);
    assert_eq!(level(&d, "k10temp", "Tccd1"), Level::Warn);
    assert_eq!(level(&d, "k10temp", "Tccd2"), Level::Normal);
    assert_eq!(level(&d, "nct6798", "SYSTIN"), Level::Warn);
    assert_eq!(level(&d, "nvme-317k", "Composite"), Level::Crit);
    assert_eq!(level(&d, "nvme-317k", "Sensor 2"), Level::Warn);
    assert_eq!(level(&d, "z53", "Coolant temp"), Level::Warn);
    // Overrides: the chip:sensor pattern beats the chip pattern.
    let o = temps_from(
        "[temps]\nwarn = { \"z53\" = 50, \"k10temp\" = 85, \"k10temp:Tccd*\" = 75 }\n\
         crit = { \"k10temp\" = 95 }\n",
    );
    assert_eq!(level(&o, "z53", "Coolant temp"), Level::Normal);
    assert_eq!(level(&o, "k10temp", "Tctl"), Level::Warn);
    assert_eq!(level(&o, "k10temp", "Tccd1"), Level::Warn);
}

// ---- #74: FANS discovery ---------------------------------------------------

fn fans_from(toml: &str) -> Fans {
    let cfg = Config::from_toml(toml).expect("toml");
    cfg.validate(32).expect("valid");
    cfg.fans
}

fn fan_rows(scratch: &Scratch, fans: &Fans) -> Vec<String> {
    let mut source = FanSource::new(fans).expect("enabled");
    let panel = source.read(&scratch.roots(), Instant::now(), &mut Capture::default());
    panel
        .fans
        .iter()
        .map(|f| format!("{}:fan{} {} {:?}", f.chip, f.channel, f.label, f.rpm))
        .collect()
}

#[test]
fn fans_are_discovered_on_every_chip_and_empty_headers_hide() {
    let scratch = Scratch::reference("fans");
    let mut source = FanSource::new(&fans_from("[fans]\nenabled = true\n")).unwrap();
    let panel = source.read(&scratch.roots(), Instant::now(), &mut Capture::default());
    assert!(panel.present);
    assert_eq!(panel.chip, "z53 \u{b7} nct6798");
    assert_eq!(
        fan_rows(&scratch, &fans_from("[fans]\nenabled = true\n")),
        [
            "z53:fan1 Pump Some(2810)",
            "z53:fan2 Fan Some(1210)",
            "nct6798:fan2 fan2 Some(1940)",
            "nct6798:fan3 fan3 Some(1880)",
            "nct6798:fan5 fan5 Some(3020)",
            "nct6798:fan6 fan6 Some(1270)",
        ]
    );
}

#[test]
fn a_fan_that_spun_once_stays_so_a_stall_shows() {
    let scratch = Scratch::reference("spun");
    let mut source = FanSource::new(&fans_from("[fans]\nenabled = true\n")).unwrap();
    let t0 = Instant::now();
    let mut sink = Capture::default();
    source.read(&scratch.roots(), t0, &mut sink);
    scratch.set(2, "fan2_input", "0");
    let panel = source.read(&scratch.roots(), t0 + Duration::from_secs(1), &mut sink);
    let fan2 = panel
        .fans
        .iter()
        .find(|f| f.chip == "nct6798" && f.channel == 2);
    assert_eq!(fan2.map(|f| f.rpm), Some(Some(0)));
    // An empty header that starts spinning appears.
    scratch.set(2, "fan4_input", "900");
    let panel = source.read(&scratch.roots(), t0 + Duration::from_secs(2), &mut sink);
    assert!(
        panel
            .fans
            .iter()
            .any(|f| f.chip == "nct6798" && f.channel == 4)
    );
}

#[test]
fn fan_allow_block_and_rename_follow_the_temps_rules() {
    let scratch = Scratch::reference("fan-rules");
    // allow limits and forces in a header that never spun; block wins;
    // a label pattern matches like fanN; rename wins over the label.
    let fans = fans_from(
        "[fans]\nenabled = true\nallow = [\"nct6798:fan?\", \"z53:Pump*\"]\n\
         block = [\"nct6798:fan5\", \"nct6798:fan6\"]\n\
         rename = { \"nct6798:fan2\" = \"rad top\", \"z53:fan1\" = \"pump\" }\n",
    );
    assert_eq!(
        fan_rows(&scratch, &fans),
        [
            "z53:fan1 pump Some(2810)",
            "nct6798:fan1 fan1 Some(0)",
            "nct6798:fan2 rad top Some(1940)",
            "nct6798:fan3 fan3 Some(1880)",
            "nct6798:fan4 fan4 Some(0)",
            "nct6798:fan7 fan7 Some(0)",
        ]
    );
    // Without the built-in rule the empty headers show too.
    let all = fan_rows(
        &scratch,
        &fans_from("[fans]\nenabled = true\ndefaults = false\n"),
    );
    assert_eq!(all.len(), 9, "{all:?}");
}

#[test]
fn the_older_hwmon_and_channels_keys_still_work_with_block_and_rename() {
    let scratch = Scratch::reference("shorthand");
    let fans = fans_from(
        "[fans]\nenabled = true\nhwmon = \"nct6798\"\nchannels = [2, 3, 5, 7]\n\
         labels = [\"front1\", \"front2\", \"rear\", \"spare\"]\n\
         block = [\"nct6798:fan5\"]\nrename = { \"nct6798:fan2\" = \"rad top\" }\n",
    );
    assert_eq!(
        fan_rows(&scratch, &fans),
        [
            "nct6798:fan2 rad top Some(1940)",
            "nct6798:fan3 front2 Some(1880)",
            "nct6798:fan7 spare Some(0)",
        ]
    );
}

#[test]
fn shorthand_and_allow_together_and_unknown_keys_are_refused() {
    let both = Config::from_toml(
        "[fans]\nenabled = true\nhwmon = \"nct6798\"\nchannels = [2]\nallow = [\"z53\"]\n",
    )
    .expect("parses");
    let err = both.validate(32).expect_err("conflict").to_string();
    assert!(err.contains("cannot both be set"), "{err}");
    for bad in [
        "[temps]\ninclude = [\"k10temp\"]\n",
        "[temps]\nexclude = [\"k10temp\"]\n",
        "[fans]\ninclude = [\"z53\"]\n",
        "[temps]\nrename = { \"k10temp\" = 5 }\n",
    ] {
        assert!(Config::from_toml(bad).is_err(), "accepted {bad}");
    }
    let glob = Config::from_toml("[temps]\nallow = [\"nct{1,2}\"]\n").expect("parses");
    let err = glob.validate(32).expect_err("bad glob").to_string();
    assert!(err.contains("not a valid glob"), "{err}");
}
