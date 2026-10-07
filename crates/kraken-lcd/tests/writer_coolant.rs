//! The LCD coolant is the writer's own z53 read, including when the snapshot disagrees.

use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

use kraken_lcd::log::Sink;
use kraken_lcd::present::present;
use kraken_lcd::service::Sampler;
use kraken_lcd::snapshot_reader::{ManualMono, SnapshotReader};
use llama_core::sample::AiState;
use llama_core::wire::{self, Ai, AiWire, Host, Tokens, WireSnapshot};

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("t23-cool-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(path.join("sys/class/hwmon")).expect("scratch");
        Self(path)
    }

    fn sys(&self) -> PathBuf {
        self.0.join("sys")
    }

    fn snap(&self) -> PathBuf {
        self.0.join("snapshot.json")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Quiet;

impl Sink for Quiet {
    fn write_line(&mut self, _line: &str) {}
}

fn publish(path: &std::path::Path, t_mono_ns: u64) {
    let snapshot = WireSnapshot {
        schema: wire::SCHEMA,
        run_id: 1,
        seq: 1,
        t_mono_ns,
        t_wall_ms: 1,
        host: Host {
            load_pct: Some(10.0),
            activity_pct: None,
            cpu_pct: Some(10.0),
            cpu_topk_pct: Some(10.0),
            gpu_pct: Some(10.0),
            mem_pct: Some(10.0),
            coolant_c: Some(99.0),
            cpu_c: Some(40.0),
            gpu_c: Some(40.0),
            gpu_w: None,
            gpu_limit_w: None,
            cpu_w: None,
            vram_used_bytes: None,
            vram_total_bytes: None,
            mem_used_bytes: None,
            mem_total_bytes: None,
        },
        ai: Ai {
            state: AiWire::Idle,
            models: Vec::new(),
        },
        tokens: Tokens {
            decoded_total: Some(1),
            prompt_total: None,
        },
        fans: Vec::new(),
        sources: None,
        suspected_loads: Vec::new(),
    };
    let bytes = wire::to_json(&snapshot).expect("json");
    std::fs::write(path, bytes).expect("write");
}

fn z53(sys: &std::path::Path, milli: &str) {
    let dir = sys.join("class/hwmon/hwmon2");
    std::fs::create_dir_all(&dir).expect("hwmon");
    std::fs::write(dir.join("name"), "z53\n").expect("name");
    std::fs::write(dir.join("temp1_input"), milli).expect("temp");
}

#[test]
fn lcd_coolant_follows_z53_when_the_snapshot_disagrees_or_is_stale() {
    let scratch = Scratch::new("disagree");
    z53(&scratch.sys(), "34100\n");
    let now = 5_000_000_000_u64;
    publish(&scratch.snap(), now);
    let mut reader = SnapshotReader::new(
        scratch.snap(),
        Duration::from_secs(1),
        scratch.sys(),
        Quiet,
        ManualMono::new(now),
    );
    let fresh = reader.sample(Instant::now(), SystemTime::UNIX_EPOCH);
    assert_eq!(fresh.coolant_c, Some(34_100.0 / 1_000.0));
    assert_ne!(fresh.coolant_c, Some(99.0));
    let view = present(
        &fresh,
        &kraken_lcd::history::History::new(fresh.t_mono),
        None,
        &kraken_lcd::config::Config::default(),
    );
    assert_eq!(view.coolant_c, Some(34), "the LCD shows the z53 reading");

    reader.set_now(now + 5_000_000_000);
    let stale = reader.sample(fresh.t_mono, SystemTime::UNIX_EPOCH);
    assert_eq!(stale.ai, AiState::NoData);
    assert_eq!(stale.coolant_c, Some(34_100.0 / 1_000.0));
    let stale_view = present(
        &stale,
        &kraken_lcd::history::History::new(stale.t_mono),
        Some(&view),
        &kraken_lcd::config::Config::default(),
    );
    assert_eq!(stale_view.coolant_c, Some(34));
}

#[test]
fn missing_z53_is_an_em_dash_on_the_lcd() {
    let scratch = Scratch::new("missing");
    let now = 5_000_000_000_u64;
    publish(&scratch.snap(), now);
    let mut reader = SnapshotReader::new(
        scratch.snap(),
        Duration::from_secs(1),
        scratch.sys(),
        Quiet,
        ManualMono::new(now),
    );
    let snap = reader.sample(Instant::now(), SystemTime::UNIX_EPOCH);
    assert_eq!(snap.coolant_c, None);
    let view = present(
        &snap,
        &kraken_lcd::history::History::new(snap.t_mono),
        None,
        &kraken_lcd::config::Config::default(),
    );
    assert_eq!(view.coolant_c, None, "a missing z53 is drawn as —");
}

#[test]
fn coolant_outside_minus_20_to_150_is_an_em_dash() {
    let now = 5_000_000_000_u64;
    let cases = [
        ("200c", "200000\n", None),
        ("just-over", "150001\n", None),
        ("just-under", "-20001\n", None),
        ("nan", "nan\n", None),
        ("inf", "inf\n", None),
        ("top", "150000\n", Some(150.0)),
        ("bottom", "-20000\n", Some(-20.0)),
    ];
    for (label, milli, expect) in cases {
        let scratch = Scratch::new(label);
        z53(&scratch.sys(), milli);
        publish(&scratch.snap(), now);
        let mut reader = SnapshotReader::new(
            scratch.snap(),
            Duration::from_secs(1),
            scratch.sys(),
            Quiet,
            ManualMono::new(now),
        );
        let snap = reader.sample(Instant::now(), SystemTime::UNIX_EPOCH);
        assert_eq!(snap.coolant_c, expect, "{label}");
        let view = present(
            &snap,
            &kraken_lcd::history::History::new(snap.t_mono),
            None,
            &kraken_lcd::config::Config::default(),
        );
        let shown = expect.map(|celsius| celsius as i16);
        assert_eq!(view.coolant_c, shown, "{label} on the LCD");
    }
}
