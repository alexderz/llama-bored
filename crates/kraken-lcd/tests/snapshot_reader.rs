//! Snapshot reader: open flags, size cap, cross-read checks, freshness, logs.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use kraken_lcd::log::Sink;
use kraken_lcd::service::Sampler;
use kraken_lcd::snapshot_reader::{ManualMono, SnapshotReader};
use llama_core::sample::{AiState, TokenReading};
use llama_core::wire::{self, Ai, AiWire, Host, ModelState, ModelWire, Tokens, WireSnapshot};

const NOW_NS: u64 = 10_000_000_000;
const STALE: Duration = Duration::from_secs(1);

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("t23-reader-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(path.join("sys/class/hwmon")).expect("scratch");
        Self(path)
    }

    fn snap(&self) -> PathBuf {
        self.0.join("snapshot.json")
    }

    fn sys(&self) -> PathBuf {
        self.0.join("sys")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone, Default)]
struct MemLog(Arc<Mutex<Vec<String>>>);

impl Sink for MemLog {
    fn write_line(&mut self, line: &str) {
        self.0
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(line.to_owned());
    }
}

impl MemLog {
    fn labels(&self) -> Vec<String> {
        self.0
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .iter()
            .filter_map(|line| line.split_once('>').map(|(_, rest)| rest.to_owned()))
            .collect()
    }

    fn count(&self, label: &str) -> usize {
        self.labels()
            .iter()
            .filter(|line| line.as_str() == label)
            .count()
    }
}

fn wire(run_id: u64, seq: u64, t_mono_ns: u64) -> WireSnapshot {
    WireSnapshot {
        schema: wire::SCHEMA,
        run_id,
        seq,
        t_mono_ns,
        t_wall_ms: 1_700_000_000_000,
        host: Host {
            load_pct: Some(12.5),
            activity_pct: None,
            cpu_pct: Some(10.0),
            cpu_topk_pct: Some(41.0),
            gpu_pct: Some(8.0),
            mem_pct: Some(40.0),
            coolant_c: Some(99.0),
            cpu_c: Some(70.0),
            gpu_c: Some(60.0),
            gpu_w: None,
            gpu_limit_w: None,
            cpu_w: None,
            vram_used_bytes: None,
            vram_total_bytes: None,
            mem_used_bytes: None,
            mem_total_bytes: None,
        },
        ai: Ai {
            state: AiWire::Loaded,
            models: vec![ModelWire {
                backend: None,
                running: None,
                queued: None,
                kv_fill: None,
                name: "Qwen 35B".to_owned(),
                state: ModelState::Ready,
                full_name: None,
                detail: None,
                cache_hit: None,
                slots_busy: None,
                slots_total: None,
                prompt_tokens: None,
                prompt_cached_tokens: None,
                slot_ctx: Vec::new(),
                engine: None,
            }],
        },
        tokens: Tokens {
            decoded_total: Some(1_000 + seq),
            prompt_total: None,
        },
        fans: Vec::new(),
        sources: None,
    }
}

fn write_snap(path: &Path, snapshot: &WireSnapshot) {
    let bytes = wire::to_json(snapshot).expect("encode");
    let mut file = std::fs::File::create(path).expect("create");
    file.write_all(&bytes).expect("write");
}

fn reader(scratch: &Scratch, log: MemLog) -> SnapshotReader<MemLog, ManualMono> {
    let mut reader = SnapshotReader::new(
        scratch.snap(),
        STALE,
        scratch.sys(),
        log,
        ManualMono::new(NOW_NS),
    );
    reader.set_now(NOW_NS);
    reader
}

fn sample(reader: &mut SnapshotReader<MemLog, ManualMono>) -> llama_core::sample::Snapshot {
    reader.sample(Instant::now(), SystemTime::UNIX_EPOCH)
}

#[test]
fn activity_pct_is_copied_and_load_pct_stays_the_linux_metric() {
    let scratch = Scratch::new("activity");
    let mut snapshot = wire(7, 3, NOW_NS);
    snapshot.host.activity_pct = Some(80.0);
    snapshot.host.load_pct = Some(12.5);
    write_snap(&scratch.snap(), &snapshot);
    let snap = sample(&mut reader(&scratch, MemLog::default()));
    assert_eq!(snap.load, Some(12.5));
    assert_eq!(snap.activity, Some(80.0));
}

/// T54: the writer reads activity up to the 125 peg and rejects above it.
#[test]
fn activity_over_100_is_read_and_above_125_is_rejected() {
    let scratch = Scratch::new("activity-redline");
    let mut snapshot = wire(7, 3, NOW_NS);
    snapshot.host.activity_pct = Some(118.0);
    write_snap(&scratch.snap(), &snapshot);
    let snap = sample(&mut reader(&scratch, MemLog::default()));
    assert_eq!(snap.activity, Some(118.0));
    assert_ne!(snap.ai, AiState::NoData, "fresh, not no data");

    let over = Scratch::new("activity-over");
    let mut snapshot = wire(7, 3, NOW_NS);
    snapshot.host.activity_pct = Some(126.0);
    write_snap(&over.snap(), &snapshot);
    let snap = sample(&mut reader(&over, MemLog::default()));
    assert_no_data(&snap);
}

fn assert_no_data(snap: &llama_core::sample::Snapshot) {
    assert_eq!(snap.ai, AiState::NoData);
    assert!(snap.models.is_empty());
    assert_eq!(snap.tokens, None);
    assert_eq!(snap.load, None);
    assert_eq!(snap.cpu_pct, None);
    assert_eq!(snap.cpu_topk_pct, None);
    assert_eq!(snap.gpu_pct, None);
    assert_eq!(snap.mem_pct, None);
    assert_eq!(snap.cpu_c, None);
    assert_eq!(snap.gpu_c, None);
    assert!(snap.errors.is_empty());
}

#[test]
fn missing_file_is_no_data_and_not_a_host_sample() {
    let scratch = Scratch::new("missing");
    let log = MemLog::default();
    let mut reader = reader(&scratch, log.clone());
    let snap = sample(&mut reader);
    assert_no_data(&snap);
    assert_eq!(snap.coolant_c, None);
    assert_eq!(log.count("snapshot missing"), 1);
}

#[test]
fn fresh_snapshot_keeps_host_ai_models_and_tokens() {
    let scratch = Scratch::new("fresh");
    write_snap(&scratch.snap(), &wire(7, 3, NOW_NS));
    let log = MemLog::default();
    let mut reader = reader(&scratch, log);
    let snap = sample(&mut reader);
    assert_eq!(snap.load, Some(12.5));
    assert_eq!(snap.activity, None, "an older snapshot omits activity_pct");
    assert_eq!(snap.cpu_pct, Some(10.0));
    assert_eq!(snap.cpu_topk_pct, Some(41.0));
    assert_eq!(snap.gpu_pct, Some(8.0));
    assert_eq!(snap.mem_pct, Some(40.0));
    assert_eq!(snap.cpu_c, Some(70.0));
    assert_eq!(snap.gpu_c, Some(60.0));
    assert_eq!(snap.coolant_c, None, "JSON coolant is not the LCD coolant");
    assert_eq!(snap.ai, AiState::Loaded);
    assert_eq!(snap.models.len(), 1);
    assert_eq!(snap.models[0].name, "Qwen 35B");
    assert_eq!(snap.models[0].state, "ready");
    assert_eq!(
        snap.tokens,
        Some(TokenReading {
            run_id: 7,
            seq: 3,
            t_mono_ns: NOW_NS,
            decoded_total: Some(1_003),
        })
    );
}

#[test]
fn empty_file_is_invalid() {
    let scratch = Scratch::new("empty");
    std::fs::write(scratch.snap(), b"").expect("empty");
    let log = MemLog::default();
    let mut reader = reader(&scratch, log.clone());
    assert_no_data(&sample(&mut reader));
    assert_eq!(log.count("snapshot invalid"), 1);
    assert_eq!(log.count("snapshot missing"), 0);
}

#[test]
fn one_past_max_bytes_is_too_large() {
    let scratch = Scratch::new("big");
    std::fs::write(scratch.snap(), vec![b'x'; wire::MAX_BYTES + 1]).expect("big");
    let log = MemLog::default();
    let mut reader = reader(&scratch, log.clone());
    assert_no_data(&sample(&mut reader));
    assert_eq!(log.count("snapshot too large"), 1);
}

#[test]
fn symlink_is_not_followed() {
    let scratch = Scratch::new("link");
    let real = scratch.0.join("real.json");
    write_snap(&real, &wire(1, 1, NOW_NS));
    std::os::unix::fs::symlink(&real, scratch.snap()).expect("symlink");
    let log = MemLog::default();
    let mut reader = reader(&scratch, log.clone());
    let snap = sample(&mut reader);
    assert_no_data(&snap);
    assert_eq!(snap.tokens, None, "the target must not be parsed");
    assert_eq!(log.count("snapshot fresh"), 0);
    assert_eq!(log.count("snapshot missing"), 0);
    assert_eq!(log.count("snapshot invalid"), 1);
}

#[test]
fn fifo_does_not_block() {
    let scratch = Scratch::new("fifo");
    let path = scratch.snap();
    let status = std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .expect("mkfifo");
    assert!(status.success(), "mkfifo");
    let log = MemLog::default();
    let mut reader = reader(&scratch, log.clone());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let snap = sample(&mut reader);
        let _ = tx.send(snap.ai);
    });
    let ai = rx
        .recv_timeout(Duration::from_millis(500))
        .expect("opening a fifo blocked the reader");
    assert_eq!(ai, AiState::NoData);
    assert_eq!(log.count("snapshot not a regular file"), 1);
}

#[test]
fn directory_is_not_a_snapshot() {
    let scratch = Scratch::new("dir");
    std::fs::create_dir(scratch.snap()).expect("dir");
    let log = MemLog::default();
    let mut reader = reader(&scratch, log.clone());
    assert_no_data(&sample(&mut reader));
    assert_eq!(log.count("snapshot not a regular file"), 1);
}

#[test]
fn seq_regression_is_invalid_and_does_not_move_the_accepted_seq() {
    let scratch = Scratch::new("regress");
    let log = MemLog::default();
    let mut reader = reader(&scratch, log.clone());
    write_snap(&scratch.snap(), &wire(4, 5, NOW_NS));
    let first = sample(&mut reader);
    assert_eq!(first.tokens.map(|t| t.seq), Some(5));
    write_snap(&scratch.snap(), &wire(4, 4, NOW_NS));
    assert_no_data(&sample(&mut reader));
    assert_no_data(&sample(&mut reader));
    assert_eq!(log.count("snapshot seq regression"), 1);
    write_snap(&scratch.snap(), &wire(4, 6, NOW_NS));
    let recovered = sample(&mut reader);
    assert_eq!(recovered.tokens.map(|t| t.seq), Some(6));
}

#[test]
fn same_seq_is_a_reread_with_no_new_token_sample() {
    let scratch = Scratch::new("reread");
    write_snap(&scratch.snap(), &wire(4, 5, NOW_NS));
    let log = MemLog::default();
    let mut reader = reader(&scratch, log.clone());
    let first = sample(&mut reader);
    let second = sample(&mut reader);
    assert_eq!(first.tokens, second.tokens);
    assert_eq!(second.tokens.map(|t| (t.run_id, t.seq)), Some((4, 5)));
    assert_eq!(second.ai, AiState::Loaded);
    assert_eq!(log.count("snapshot fresh"), 1);
    assert_eq!(log.count("snapshot seq regression"), 0);
}

#[test]
fn future_timestamp_past_the_slack_is_invalid() {
    let scratch = Scratch::new("future");
    let slack = 50 * 1_000_000;
    write_snap(&scratch.snap(), &wire(1, 1, NOW_NS + slack + 1));
    let log = MemLog::default();
    let mut reader = reader(&scratch, log.clone());
    assert_no_data(&sample(&mut reader));
    assert_eq!(log.count("snapshot from the future"), 1);

    write_snap(&scratch.snap(), &wire(1, 1, NOW_NS + slack));
    let on_the_line = sample(&mut reader);
    assert_eq!(on_the_line.ai, AiState::Loaded);
    assert!(on_the_line.tokens.is_some());
}

#[test]
fn stale_boundary_is_stale_after_plus_or_minus_one_millisecond() {
    let scratch = Scratch::new("stale");
    let log = MemLog::default();
    let mut reader = reader(&scratch, log.clone());
    let ms = 1_000_000_u64;
    write_snap(&scratch.snap(), &wire(1, 1, NOW_NS - 1_000_000_000 + ms));
    let inside = sample(&mut reader);
    assert_eq!(
        inside.ai,
        AiState::Loaded,
        "1 ms inside the window is fresh"
    );
    assert!(inside.tokens.is_some());

    write_snap(&scratch.snap(), &wire(1, 2, NOW_NS - 1_000_000_000));
    let exact = sample(&mut reader);
    assert_eq!(exact.ai, AiState::Loaded, "age == stale_after_s is fresh");

    write_snap(&scratch.snap(), &wire(1, 3, NOW_NS - 1_000_000_000 - ms));
    let outside = sample(&mut reader);
    assert_no_data(&outside);
    assert_eq!(log.count("snapshot stale"), 1);
}

#[test]
fn stale_then_fresh_again_is_a_normal_sample() {
    let scratch = Scratch::new("again");
    let log = MemLog::default();
    let mut reader = reader(&scratch, log.clone());
    write_snap(&scratch.snap(), &wire(1, 1, NOW_NS - 2_000_000_000));
    assert_no_data(&sample(&mut reader));
    write_snap(&scratch.snap(), &wire(1, 2, NOW_NS));
    let again = sample(&mut reader);
    assert_eq!(again.ai, AiState::Loaded);
    assert_eq!(again.load, Some(12.5));
    assert_eq!(again.cpu_pct, Some(10.0));
    assert_eq!(again.models.len(), 1);
    assert_eq!(again.models[0].name, "Qwen 35B");
    assert!(again.tokens.is_some());
    assert_eq!(log.count("snapshot stale"), 1);
    assert_eq!(log.count("snapshot fresh"), 1);
}

#[test]
fn run_id_change_accepts_a_lower_seq() {
    let scratch = Scratch::new("run");
    let log = MemLog::default();
    let mut reader = reader(&scratch, log);
    write_snap(&scratch.snap(), &wire(1, 50, NOW_NS));
    assert_eq!(sample(&mut reader).tokens.map(|t| t.seq), Some(50));
    write_snap(&scratch.snap(), &wire(2, 1, NOW_NS));
    let next = sample(&mut reader);
    assert_eq!(
        next.tokens.map(|t| (t.run_id, t.seq)),
        Some((2, 1)),
        "a new run_id is not a seq regression"
    );
    assert_eq!(next.ai, AiState::Loaded);
}

#[test]
fn non_canonical_name_is_rejected() {
    let scratch = Scratch::new("name");
    let mut snapshot = wire(1, 1, NOW_NS);
    snapshot.ai.models[0].name = "1234567890123".to_owned();
    write_snap(&scratch.snap(), &snapshot);
    let log = MemLog::default();
    let mut reader = reader(&scratch, log.clone());
    assert_no_data(&sample(&mut reader));
    assert_eq!(log.count("snapshot invalid"), 1);
    assert_eq!(log.count("snapshot fresh"), 0);
}

#[test]
fn each_transition_is_logged_once() {
    let scratch = Scratch::new("logs");
    let log = MemLog::default();
    let mut reader = reader(&scratch, log.clone());
    let _ = sample(&mut reader);
    let _ = sample(&mut reader);
    let _ = sample(&mut reader);
    assert_eq!(log.count("snapshot missing"), 1);

    write_snap(&scratch.snap(), &wire(1, 1, NOW_NS));
    let _ = sample(&mut reader);
    let _ = sample(&mut reader);
    assert_eq!(log.count("snapshot fresh"), 1);

    reader.set_now(NOW_NS + 2_000_000_000);
    let _ = sample(&mut reader);
    let _ = sample(&mut reader);
    assert_eq!(log.count("snapshot stale"), 1);

    std::fs::write(scratch.snap(), b"{").expect("garbage");
    let _ = sample(&mut reader);
    let _ = sample(&mut reader);
    assert_eq!(log.count("snapshot invalid"), 1);

    std::fs::remove_file(scratch.snap()).expect("unlink");
    let _ = sample(&mut reader);
    let _ = sample(&mut reader);
    assert_eq!(log.count("snapshot missing"), 2);
}

#[test]
fn own_coolant_read_wins_over_the_snapshot() {
    let scratch = Scratch::new("coolant");
    let hwmon = scratch.sys().join("class/hwmon/hwmon0");
    std::fs::create_dir_all(&hwmon).expect("hwmon");
    std::fs::write(hwmon.join("name"), "z53\n").expect("name");
    std::fs::write(hwmon.join("temp1_input"), "34100\n").expect("temp");
    write_snap(&scratch.snap(), &wire(1, 1, NOW_NS));
    let mut reader = reader(&scratch, MemLog::default());
    let fresh = sample(&mut reader);
    let coolant = 34_100.0_f32 / 1_000.0;
    assert_eq!(fresh.coolant_c, Some(coolant));
    reader.set_now(NOW_NS + 5_000_000_000);
    let stale = sample(&mut reader);
    assert_eq!(stale.ai, AiState::NoData);
    assert_eq!(stale.coolant_c, Some(coolant));
}

#[test]
fn full_name_and_detail_are_copied_and_absent_from_an_older_watcher() {
    let scratch = Scratch::new("detail");
    let mut snapshot = wire(7, 3, NOW_NS);
    snapshot.ai.models[0].full_name = Some("Ternary Bonsai 2 27B".to_owned());
    snapshot.ai.models[0].detail = Some(wire::ModelDetail {
        ctx: Some(262_144),
        quant: Some("PTQ1_0".to_owned()),
        ..wire::ModelDetail::default()
    });
    write_snap(&scratch.snap(), &snapshot);
    let snap = sample(&mut reader(&scratch, MemLog::default()));
    assert_eq!(snap.ai, AiState::Loaded);
    assert_eq!(
        snap.models[0].full_name.as_deref(),
        Some("Ternary Bonsai 2 27B")
    );
    assert_eq!(snap.models[0].detail, snapshot.ai.models[0].detail);

    let old = Scratch::new("detail-old");
    write_snap(&old.snap(), &wire(7, 3, NOW_NS));
    let snap = sample(&mut reader(&old, MemLog::default()));
    assert_eq!(snap.models[0].name, "Qwen 35B");
    assert_eq!(snap.models[0].full_name, None);
    assert_eq!(snap.models[0].detail, None);
}

/// #31: the LCD takes only the speculative acceptance from a model's
/// engine numbers; everything else in them stays unread.
#[test]
fn spec_acceptance_is_the_only_engine_number_read() {
    let scratch = Scratch::new("engine");
    let mut snapshot = wire(7, 3, NOW_NS);
    snapshot.ai.models[0].backend = Some(wire::Backend::Vllm);
    snapshot.ai.models[0].engine = Some(wire::EngineWire {
        spec_accept: Some(0.781),
        spec_len: Some(2.9),
        preemptions: Some(3),
        ttft_s: Some(0.4),
        ..wire::EngineWire::default()
    });
    write_snap(&scratch.snap(), &snapshot);
    let snap = sample(&mut reader(&scratch, MemLog::default()));
    let info = snap.models[0].backend.expect("spec carried");
    assert_eq!(info.kind, wire::Backend::Vllm);
    assert_eq!(
        info.engine,
        llama_core::backend::EngineStats {
            spec_permille: Some(781),
            ..llama_core::backend::EngineStats::default()
        }
    );
    assert_eq!(info.running, None);

    let none = Scratch::new("engine-none");
    snapshot.ai.models[0].engine = Some(wire::EngineWire {
        sleeping: Some(true),
        ..wire::EngineWire::default()
    });
    write_snap(&none.snap(), &snapshot);
    let snap = sample(&mut reader(&none, MemLog::default()));
    let info = snap.models[0]
        .backend
        .expect("#33: the engine is always carried");
    assert_eq!(info.kind, wire::Backend::Vllm);
    assert!(info.engine.is_empty(), "no spec, no engine numbers");

    // #33: an absent backend reads as llama.cpp.
    let old = Scratch::new("engine-absent");
    snapshot.ai.models[0].backend = None;
    snapshot.ai.models[0].engine = None;
    write_snap(&old.snap(), &snapshot);
    let snap = sample(&mut reader(&old, MemLog::default()));
    assert_eq!(
        snap.models[0].backend.map(|info| info.kind),
        Some(wire::Backend::LlamaCpp)
    );
}
