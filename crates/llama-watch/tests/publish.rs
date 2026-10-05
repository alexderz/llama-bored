//! Atomic snapshot publish. Readers see a previous file or a complete new one.
//!
//! Every directory is under `CARGO_TARGET_TMPDIR`. Nothing opens `/run/llama-watch`.

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use llama_core::sample::{AiState, LlamaView, ModelInfo, Snapshot};
use llama_core::wire::{self, AiWire, ModelState};
use llama_watch::publish::{Extras, PublishError, Publisher, SlotCtx};
use llama_watch::resets::ResetCounts;

fn scratch(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("publish-{label}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&path).expect("scratch dir");
    path
}

#[derive(Clone, Default)]
struct Capture {
    lines: Arc<Mutex<Vec<String>>>,
}

impl Capture {
    fn lines(&self) -> Vec<String> {
        self.lines
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone()
    }
}

impl llama_core::log::Sink for Capture {
    fn write_line(&mut self, line: &str) {
        self.lines
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push(line.to_owned());
    }
}

fn snapshot(wall: SystemTime, load: Option<f32>, ai: AiState, models: Vec<ModelInfo>) -> Snapshot {
    Snapshot {
        t_mono: Instant::now(),
        t_wall: wall,
        load,
        activity: None,
        cpu_pct: None,
        cpu_topk_pct: None,
        gpu_pct: None,
        mem_pct: None,
        coolant_c: None,
        cpu_c: None,
        gpu_c: None,
        ai,
        models,
        tokens: None,
        errors: std::collections::BTreeSet::new(),
    }
}

fn idle(wall: SystemTime) -> Snapshot {
    snapshot(wall, None, AiState::Idle, Vec::new())
}

fn view(ai: AiState, models: Vec<ModelInfo>, decoded: Option<u64>) -> LlamaView {
    LlamaView {
        ai,
        models,
        decoded_total: decoded,
        prompt_total: None,
    }
}

fn idle_view() -> LlamaView {
    view(AiState::Idle, Vec::new(), None)
}

fn mode_bits(path: &std::path::Path) -> u32 {
    std::fs::metadata(path).expect("stat").permissions().mode() & 0o777
}

#[test]
fn production_passes_the_snapshot_directory_constant() {
    assert_eq!(wire::SNAPSHOT_DIR, "/run/llama-watch");
    assert!(wire::SNAPSHOT_PATH.starts_with(wire::SNAPSHOT_DIR));
}

#[test]
fn reader_never_sees_an_invalid_snapshot_and_seq_is_monotonic() {
    let dir = scratch("hammer");
    let log = Capture::default();
    let mut publisher = Publisher::open(&dir, log).expect("open tmp dir");
    let path = dir.join("snapshot.json");
    let stop = Arc::new(AtomicBool::new(false));
    let stop_reader = Arc::clone(&stop);
    let read_path = path.clone();
    let reader = std::thread::spawn(move || {
        let mut seen = 0u64;
        let mut last_seq = 0u64;
        let mut bad = None;
        while !stop_reader.load(Ordering::Relaxed) || seen == 0 {
            match std::fs::read(&read_path) {
                Ok(bytes) if bytes.is_empty() => {
                    bad = Some("empty snapshot".to_owned());
                    break;
                }
                Ok(bytes) => match wire::parse_validated(&bytes) {
                    Ok(parsed) => {
                        if parsed.seq < last_seq {
                            bad = Some(format!("seq went {} then {}", last_seq, parsed.seq));
                            break;
                        }
                        last_seq = parsed.seq;
                        seen += 1;
                    }
                    Err(err) => {
                        bad = Some(err.to_string());
                        break;
                    }
                },
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => {
                    bad = Some(err.to_string());
                    break;
                }
            }
            if stop_reader.load(Ordering::Relaxed) && seen > 0 {
                break;
            }
        }
        (seen, last_seq, bad)
    });

    let wall = SystemTime::UNIX_EPOCH + Duration::from_millis(1_700_000_000_000);
    for _ in 0..10_000 {
        publisher
            .publish(&idle(wall), &idle_view())
            .expect("publish");
    }
    stop.store(true, Ordering::Relaxed);
    let (seen, last_seq, bad) = reader.join().expect("reader");
    assert!(bad.is_none(), "{bad:?}");
    assert!(seen > 0, "reader never observed a snapshot");
    assert!(last_seq <= 10_000, "seq {last_seq} above the publish count");
    assert!(last_seq > 0);

    let final_bytes = std::fs::read(&path).expect("final snapshot");
    let final_snap = wire::parse_validated(&final_bytes).expect("final parse");
    assert_eq!(final_snap.seq, 10_000);
    assert_eq!(final_snap.schema, wire::SCHEMA);
    assert!(final_bytes.len() <= wire::MAX_BYTES);
}

fn assert_published_regular(dir: &std::path::Path) {
    let path = dir.join("snapshot.json");
    let meta = std::fs::metadata(&path).expect("snapshot metadata");
    assert!(
        meta.file_type().is_file(),
        "snapshot.json is a regular file"
    );
    assert_eq!(mode_bits(&path), 0o640);
    wire::parse_validated(&std::fs::read(&path).expect("snapshot bytes")).expect("valid snapshot");
    assert!(
        !dir.join("snapshot.json.tmp").exists(),
        "temp name is consumed by the rename"
    );
}

#[test]
fn wire_fields_come_from_the_snapshot_and_the_llama_view() {
    let dir = scratch("wire");
    let mut publisher = Publisher::open(&dir, Capture::default()).expect("open");
    let wall = SystemTime::UNIX_EPOCH + Duration::from_millis(1_790_200_000_123);
    let snap = snapshot(
        wall,
        Some(12.5),
        AiState::Loaded,
        vec![ModelInfo {
            backend: None,
            name: "hello\nworld".to_owned(),
            state: "starting".to_owned(),
            full_name: None,
            detail: None,
        }],
    );
    let llama = view(
        AiState::Loaded,
        vec![ModelInfo {
            backend: None,
            name: "hello\nworld".to_owned(),
            state: "starting".to_owned(),
            full_name: None,
            detail: None,
        }],
        Some(99),
    );
    publisher.publish(&snap, &llama).expect("first");
    publisher.publish(&snap, &llama).expect("second");
    let bytes = std::fs::read(dir.join("snapshot.json")).expect("read");
    let text = std::str::from_utf8(&bytes).expect("utf8");
    assert!(text.contains("\"cpu_pct\":null"), "{text}");
    assert!(text.contains("\"load_pct\":12.5"), "{text}");
    let parsed = wire::parse_validated(&bytes).expect("parse");
    assert_eq!(parsed.schema, 1);
    assert_eq!(parsed.seq, 2);
    assert_ne!(parsed.run_id, 0);
    assert!(parsed.t_mono_ns > 0);
    assert_eq!(parsed.t_wall_ms, 1_790_200_000_123);
    assert_eq!(parsed.host.load_pct, Some(12.5));
    assert_eq!(parsed.host.cpu_pct, None);
    assert_eq!(parsed.ai.state, AiWire::Loaded);
    assert_eq!(parsed.ai.models.len(), 1);
    assert_eq!(parsed.ai.models[0].name, "hello world");
    assert_eq!(parsed.ai.models[0].state, ModelState::Starting);
    assert_eq!(parsed.tokens.decoded_total, Some(99));

    let first_run = parsed.run_id;
    let first_mono = parsed.t_mono_ns;
    publisher.publish(&snap, &llama).expect("third");
    let again = wire::parse_validated(&std::fs::read(dir.join("snapshot.json")).unwrap()).unwrap();
    assert_eq!(again.run_id, first_run);
    assert_eq!(again.seq, 3);
    assert!(again.t_mono_ns >= first_mono);
}

fn sglang(hit_permille: Option<u16>) -> Option<llama_core::backend::BackendInfo> {
    Some(llama_core::backend::BackendInfo {
        kind: llama_core::backend::Backend::SgLang,
        running: Some(2),
        queued: Some(1),
        kv_permille: Some(370),
        hit_permille,
        ..Default::default()
    })
}

fn model(name: &str, backend: Option<llama_core::backend::BackendInfo>) -> ModelInfo {
    ModelInfo {
        backend,
        name: name.to_owned(),
        state: "ready".to_owned(),
        full_name: None,
        detail: None,
    }
}

#[test]
fn metrics_extras_reach_the_wire() {
    let dir = scratch("extras");
    let mut publisher = Publisher::open(&dir, Capture::default()).expect("open");
    let mut snap = snapshot(
        SystemTime::now(),
        None,
        AiState::Loaded,
        vec![
            model("flash", sglang(Some(800))),
            model(
                "bonsai",
                Some(llama_core::backend::BackendInfo {
                    kind: llama_core::backend::Backend::LlamaCpp,
                    // llama.cpp keeps its slot view; no gauges on the wire.
                    hit_permille: Some(500),
                    ..Default::default()
                }),
            ),
        ],
    );
    snap.mem_pct = Some(50.0);
    let mut llama = view(AiState::Loaded, snap.models.clone(), Some(10));
    llama.prompt_total = Some(4_000);
    let extras = Extras {
        gpu_w: Some(312.5),
        gpu_limit_w: Some(600.0),
        cpu_w: Some(88.25),
        vram_used: Some(20 << 30),
        vram_total: Some(32 << 30),
        mem_used: Some(64 << 30),
        mem_total: Some(128 << 30),
        slots: vec![("bonsai".to_owned(), 1, 4), ("gone".to_owned(), 1, 1)],
        prompt_cache: vec![
            ("flash".to_owned(), 9_000, Some(8_000)),
            ("bonsai".to_owned(), 700, None),
        ],
        slot_ctx: vec![
            SlotCtx {
                model: "bonsai".to_owned(),
                slot: 1,
                used: Some(91_000),
                resets: new_resets(3),
            },
            SlotCtx {
                model: "bonsai".to_owned(),
                slot: 0,
                used: Some(5),
                resets: new_resets(0),
            },
            SlotCtx {
                model: "gone".to_owned(),
                slot: 0,
                used: Some(5),
                resets: new_resets(0),
            },
        ],
        fans: vec![
            (2, "CPU".to_owned(), Some(1200), Some(255)),
            (3, "fan3".to_owned(), None, Some(0)),
        ],
        sources: Some(wire::Sources {
            llama_swap: Some(wire::SourceWire {
                up: true,
                latency_s: Some(0.003),
            }),
            ..Default::default()
        }),
    };
    publisher
        .publish_with(&snap, &llama, &extras)
        .expect("publish");
    let wire = wire::parse_validated(&std::fs::read(dir.join("snapshot.json")).unwrap())
        .expect("validates");
    assert_eq!(wire.host.gpu_w, Some(312.5));
    assert_eq!(wire.host.gpu_limit_w, Some(600.0));
    assert_eq!(wire.host.cpu_w, Some(88.25));
    assert_eq!(wire.host.vram_used_bytes, Some(20 << 30));
    assert_eq!(wire.host.vram_total_bytes, Some(32 << 30));
    assert_eq!(wire.host.mem_used_bytes, Some(64 << 30));
    assert_eq!(wire.host.mem_total_bytes, Some(128 << 30));
    assert_eq!(wire.tokens.prompt_total, Some(4_000));
    let flash = &wire.ai.models[0];
    assert_eq!(
        (flash.running, flash.queued, flash.kv_fill, flash.cache_hit),
        (Some(2), Some(1), Some(0.37), Some(0.8))
    );
    assert_eq!((flash.slots_busy, flash.slots_total), (None, None));
    let bonsai = &wire.ai.models[1];
    assert_eq!(bonsai.cache_hit, None, "llama.cpp gauges stay off the wire");
    assert_eq!((bonsai.slots_busy, bonsai.slots_total), (Some(1), Some(4)));
    // #10: prompt counters by display name, slots lowest id first.
    assert_eq!(
        (flash.prompt_tokens, flash.prompt_cached_tokens),
        (Some(9_000), Some(8_000))
    );
    assert!(flash.slot_ctx.is_empty());
    assert_eq!(
        (bonsai.prompt_tokens, bonsai.prompt_cached_tokens),
        (Some(700), None)
    );
    assert_eq!(
        bonsai.slot_ctx,
        vec![
            wire::SlotCtxWire {
                slot: 0,
                used: 5,
                resets: wire_resets(0)
            },
            wire::SlotCtxWire {
                slot: 1,
                used: 91_000,
                resets: wire_resets(3)
            },
        ]
    );
    assert_eq!(wire.fans.len(), 2);
    assert_eq!(wire.fans[0].channel, 2);
    assert_eq!(wire.fans[0].label, "CPU");
    assert_eq!(wire.fans[0].rpm, Some(1200));
    assert_eq!(wire.fans[0].pwm, Some(1.0));
    assert_eq!(wire.fans[1].rpm, None);
    assert_eq!(wire.fans[1].pwm, Some(0.0));
    let sources = wire.sources.expect("sources");
    assert_eq!(
        sources.llama_swap,
        Some(wire::SourceWire {
            up: true,
            latency_s: Some(0.003)
        })
    );
    assert_eq!(sources.proc, None);
}

#[test]
fn out_of_range_extras_are_left_out_not_fatal() {
    let dir = scratch("extras-bad");
    let lines = Capture::default();
    let mut publisher = Publisher::open(&dir, lines.clone()).expect("open");
    let extras = Extras {
        gpu_w: Some(f64::NAN),
        gpu_limit_w: Some(1e9),
        cpu_w: Some(-1.0),
        vram_used: Some(u64::MAX),
        slots: vec![("x".to_owned(), 9, 4)],
        // Cached above the prompt is cut to it.
        prompt_cache: vec![("x".to_owned(), 10, Some(50))],
        // A negative or too-high slot id, a slot with no count, a repeat,
        // and a context past the wire top.
        slot_ctx: [
            (-1, Some(1)),
            (i64::from(wire::MAX_SLOTS), Some(1)),
            (3, None),
            (4, Some(u64::MAX)),
            (4, Some(2)),
        ]
        .into_iter()
        .map(|(slot, used)| SlotCtx {
            model: "x".to_owned(),
            slot,
            used,
            resets: new_resets(1),
        })
        .collect(),
        fans: vec![
            (0, "zero".to_owned(), None, None),
            (17, "high".to_owned(), None, None),
            (1, "\n\t".to_owned(), None, None),
            (2, "fast".to_owned(), Some(u32::MAX), None),
            (2, "again".to_owned(), None, None),
        ],
        sources: Some(wire::Sources {
            metrics: Some(wire::SourceWire {
                up: false,
                latency_s: Some(1e6),
            }),
            ..Default::default()
        }),
        ..Extras::default()
    };
    let snap = snapshot(
        SystemTime::now(),
        None,
        AiState::Loaded,
        vec![model("x", None)],
    );
    let llama = view(AiState::Loaded, snap.models.clone(), None);
    publisher
        .publish_with(&snap, &llama, &extras)
        .expect("still publishes");
    let wire = wire::parse_validated(&std::fs::read(dir.join("snapshot.json")).unwrap())
        .expect("validates");
    assert_eq!(wire.host.gpu_w, None);
    assert_eq!(wire.host.gpu_limit_w, None);
    assert_eq!(wire.host.cpu_w, None);
    assert_eq!(wire.host.vram_used_bytes, None);
    assert_eq!(wire.ai.models[0].slots_busy, None);
    assert_eq!(wire.ai.models[0].slots_total, None);
    assert_eq!(wire.ai.models[0].prompt_cached_tokens, Some(10));
    assert_eq!(
        wire.ai.models[0].slot_ctx,
        vec![wire::SlotCtxWire {
            slot: 4,
            used: wire::MAX_CTX_TOKENS,
            resets: wire_resets(1)
        }]
    );
    assert_eq!(wire.fans.len(), 1, "{:?}", wire.fans);
    assert_eq!((wire.fans[0].channel, wire.fans[0].rpm), (2, None));
    assert_eq!(
        wire.sources.and_then(|s| s.metrics),
        Some(wire::SourceWire {
            up: false,
            latency_s: None
        })
    );
    assert!(lines.lines().is_empty(), "{:?}", lines.lines());
}

fn new_resets(n: u64) -> ResetCounts {
    ResetCounts {
        new: n,
        ..ResetCounts::default()
    }
}

fn wire_resets(n: u64) -> wire::SlotResetsWire {
    wire::SlotResetsWire {
        new: n,
        ..wire::SlotResetsWire::default()
    }
}

/// #10: more slots than the wire carries keeps the first models' lowest
/// slots, and the snapshot still publishes.
#[test]
fn slot_rows_past_the_wire_cap_are_dropped_not_fatal() {
    let dir = scratch("slot-cap");
    let mut publisher = Publisher::open(&dir, Capture::default()).expect("open");
    let names = ["a", "b", "c"];
    let snap = snapshot(
        SystemTime::now(),
        None,
        AiState::Loaded,
        names.iter().map(|name| model(name, None)).collect(),
    );
    let llama = view(AiState::Loaded, snap.models.clone(), None);
    let extras = Extras {
        slot_ctx: names
            .iter()
            .flat_map(|name| {
                (0..20).rev().map(|slot| SlotCtx {
                    model: (*name).to_owned(),
                    slot,
                    used: Some(1_000),
                    resets: new_resets(0),
                })
            })
            .collect(),
        ..Extras::default()
    };
    publisher
        .publish_with(&snap, &llama, &extras)
        .expect("publishes");
    let wire = wire::parse_validated(&std::fs::read(dir.join("snapshot.json")).unwrap())
        .expect("validates");
    let counts: Vec<usize> = wire.ai.models.iter().map(|m| m.slot_ctx.len()).collect();
    assert_eq!(counts, vec![20, wire::MAX_SLOT_CTX - 20, 0]);
    let b: Vec<u16> = wire.ai.models[1].slot_ctx.iter().map(|r| r.slot).collect();
    assert_eq!(b, (0..12).collect::<Vec<u16>>());
}

#[test]
fn oversize_snapshot_is_not_published_and_logged_once() {
    let dir = scratch("oversize");
    let log = Capture::default();
    let mut publisher = Publisher::open(&dir, log.clone()).expect("open");
    let wall = SystemTime::UNIX_EPOCH;
    publisher.publish(&idle(wall), &idle_view()).expect("first");
    let path = dir.join("snapshot.json");
    let kept = std::fs::read(&path).expect("first bytes");
    assert!(kept.len() <= wire::MAX_BYTES);
    assert!(kept.len() > 1);

    // A valid snapshot cannot reach MAX_BYTES. A lower cap takes the same branch.
    publisher.set_max_bytes(kept.len() - 1);
    let rejected = publisher.publish(&idle(wall), &idle_view());
    assert!(
        matches!(rejected, Err(PublishError::TooLarge)),
        "{rejected:?}"
    );
    let still = wire::parse_validated(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(still.seq, 1, "rejected publish keeps the previous file");
    let logged: Vec<_> = log
        .lines()
        .into_iter()
        .filter(|line| line.contains("maximum"))
        .collect();
    assert_eq!(logged.len(), 1, "{logged:?}");

    let rejected = publisher.publish(&idle(wall), &idle_view());
    assert!(matches!(rejected, Err(PublishError::TooLarge)));
    let still = wire::parse_validated(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(still.seq, 1);
    let logged: Vec<_> = log
        .lines()
        .into_iter()
        .filter(|line| line.contains("maximum"))
        .collect();
    assert_eq!(logged.len(), 1, "oversize is logged once: {logged:?}");

    publisher.set_max_bytes(wire::MAX_BYTES);
    publisher
        .publish(&idle(wall), &idle_view())
        .expect("a later snapshot publishes");
    publisher.set_max_bytes(kept.len() - 1);
    let rejected = publisher.publish(&idle(wall), &idle_view());
    assert!(matches!(rejected, Err(PublishError::TooLarge)));
    let logged: Vec<_> = log
        .lines()
        .into_iter()
        .filter(|line| line.contains("maximum"))
        .collect();
    assert_eq!(
        logged.len(),
        2,
        "a success clears the oversize log: {logged:?}"
    );
}

#[test]
fn invalid_snapshot_is_not_written() {
    let dir = scratch("invalid");
    let mut publisher = Publisher::open(&dir, Capture::default()).expect("open");
    let wall = SystemTime::UNIX_EPOCH;
    publisher.publish(&idle(wall), &idle_view()).expect("good");
    let kept = std::fs::read(dir.join("snapshot.json")).unwrap();
    let mut bad = snapshot(wall, None, AiState::Idle, Vec::new());
    bad.gpu_c = Some(200.0);
    let err = publisher.publish(&bad, &idle_view());
    assert!(err.is_err(), "an out-of-range temperature must not publish");
    assert_eq!(std::fs::read(dir.join("snapshot.json")).unwrap(), kept);
}

#[test]
fn twelve_models_and_mem_pct_101_still_publish() {
    let dir = scratch("clamp");
    let mut publisher = Publisher::open(&dir, Capture::default()).expect("open");
    let models: Vec<_> = (0..12)
        .map(|index| ModelInfo {
            backend: None,
            name: format!("m{index}"),
            state: "ready".to_owned(),
            full_name: None,
            detail: None,
        })
        .collect();
    let mut snap = snapshot(SystemTime::UNIX_EPOCH, Some(140.0), AiState::Loaded, models);
    snap.mem_pct = Some(101.0);
    snap.cpu_pct = Some(-3.0);
    snap.cpu_topk_pct = Some(250.0);
    snap.gpu_pct = Some(140.0);
    publisher
        .publish(&snap, &view(AiState::Loaded, Vec::new(), None))
        .expect("one bad field does not block the publish");
    let parsed = wire::parse_validated(&std::fs::read(dir.join("snapshot.json")).unwrap())
        .expect("published snapshot validates");
    assert_eq!(parsed.host.mem_pct, Some(100.0));
    assert_eq!(parsed.host.load_pct, Some(100.0));
    assert_eq!(parsed.host.cpu_pct, Some(0.0));
    assert_eq!(parsed.host.cpu_topk_pct, Some(100.0));
    assert_eq!(parsed.host.gpu_pct, Some(100.0));
    assert_eq!(parsed.ai.models.len(), wire::MAX_MODELS);
    assert_eq!(parsed.ai.models[0].name, "m0");
    assert_eq!(parsed.ai.models[wire::MAX_MODELS - 1].name, "m7");
}

/// T54: activity keeps its redline across the wire. 118 stays 118, anything
/// above the 125 peg is clamped to it, and the writer's parse accepts both.
#[test]
fn activity_over_100_publishes_and_validates_up_to_the_125_peg() {
    for (raw, wire_value) in [(118.0, 118.0), (125.0, 125.0), (140.0, 125.0), (-2.0, 0.0)] {
        let dir = scratch("activity");
        let mut publisher = Publisher::open(&dir, Capture::default()).expect("open");
        let mut snap = snapshot(
            SystemTime::UNIX_EPOCH,
            Some(90.0),
            AiState::Idle,
            Vec::new(),
        );
        snap.activity = Some(raw);
        publisher.publish(&snap, &idle_view()).expect("publish");
        let bytes = std::fs::read(dir.join("snapshot.json")).expect("published");
        let parsed = wire::parse_validated(&bytes).expect("the writer accepts it");
        assert_eq!(parsed.host.activity_pct, Some(wire_value), "raw {raw}");
        assert_eq!(parsed.host.load_pct, Some(90.0));
    }
}

#[test]
fn symlink_at_the_temp_name_does_not_redirect_the_write() {
    let dir = scratch("symlink");
    let secret = scratch("secret-symlink");
    let secret_file = secret.join("payload");
    std::fs::write(&secret_file, b"SENTINEL").unwrap();
    std::os::unix::fs::symlink(&secret_file, dir.join("snapshot.json.tmp")).unwrap();
    let mut publisher = Publisher::open(&dir, Capture::default()).expect("open");
    publisher
        .publish(&idle(SystemTime::UNIX_EPOCH), &idle_view())
        .expect("a planted symlink does not block publish");
    assert_eq!(std::fs::read(&secret_file).unwrap(), b"SENTINEL");
    assert_published_regular(&dir);
}

#[test]
fn fifo_at_the_temp_name_is_not_written() {
    let dir = scratch("fifo");
    let tmp = dir.join("snapshot.json.tmp");
    let dirfd = rustix::fs::open(
        &dir,
        rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::DIRECTORY | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .expect("open dir");
    rustix::fs::mkfifoat(
        &dirfd,
        "snapshot.json.tmp",
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
    )
    .expect("mkfifo");
    let held = rustix::fs::open(
        &tmp,
        rustix::fs::OFlags::RDWR | rustix::fs::OFlags::NONBLOCK | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .expect("hold the fifo open");
    let mut publisher = Publisher::open(&dir, Capture::default()).expect("open");
    publisher
        .publish(&idle(SystemTime::UNIX_EPOCH), &idle_view())
        .expect("a planted fifo does not block publish");
    let mut buf = [0u8; 64];
    match rustix::io::read(&held, &mut buf) {
        Ok(0) => {}
        Err(rustix::io::Errno::AGAIN) => {}
        Ok(n) => panic!("snapshot bytes were written into the fifo: {n}"),
        Err(err) => panic!("fifo read failed: {err}"),
    }
    assert_published_regular(&dir);
}

#[test]
fn hard_link_at_the_temp_name_does_not_truncate_the_sentinel() {
    let dir = scratch("hardlink");
    let sentinel = dir.join("sentinel");
    std::fs::write(&sentinel, b"SENTINEL").unwrap();
    std::fs::hard_link(&sentinel, dir.join("snapshot.json.tmp")).unwrap();
    let mut publisher = Publisher::open(&dir, Capture::default()).expect("open");
    publisher
        .publish(&idle(SystemTime::UNIX_EPOCH), &idle_view())
        .expect("a planted hard link does not block publish");
    assert_eq!(std::fs::read(&sentinel).unwrap(), b"SENTINEL");
    assert_published_regular(&dir);
    let snap_ino =
        std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(dir.join("snapshot.json")).unwrap());
    let sentinel_ino = std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(&sentinel).unwrap());
    assert_ne!(snap_ino, sentinel_ino);
}

#[test]
fn directory_symlink_is_not_opened() {
    let real = scratch("real-dir");
    let link_parent = scratch("link-parent");
    let link = link_parent.join("linked");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let opened = Publisher::open(&link, Capture::default());
    assert!(
        opened.is_err(),
        "O_NOFOLLOW must reject a symlink directory"
    );
    assert!(!real.join("snapshot.json").exists());
    assert!(!real.join("snapshot.json.tmp").exists());
}

#[test]
fn full_name_and_detail_reach_the_wire_and_bad_ones_do_not() {
    use llama_core::detail::ModelDetail;

    let dir = scratch("detail");
    let mut publisher = Publisher::open(&dir, Capture::default()).expect("open");
    let wall = SystemTime::UNIX_EPOCH + Duration::from_millis(1);
    let detail = ModelDetail {
        ctx: Some(262_144),
        ncmoe: None,
        kv_k: Some("q8_0".to_owned()),
        kv_v: Some("q8_0".to_owned()),
        quant: Some("PTQ1_0".to_owned()),
        fa: Some(true),
        kv_block: None,
        prefix_cache: None,
    };
    let models = vec![
        ModelInfo {
            backend: None,
            name: "Ternary Bon…".to_owned(),
            state: "ready".to_owned(),
            full_name: Some("Ternary Bonsai 2 27B".to_owned()),
            detail: Some(detail.clone()),
        },
        ModelInfo {
            backend: None,
            name: "Qwen 35B".to_owned(),
            state: "ready".to_owned(),
            full_name: Some("Qwen 35B".to_owned()),
            detail: Some(ModelDetail {
                quant: Some("/models/secret.gguf".to_owned()),
                ..ModelDetail::default()
            }),
        },
    ];
    let snap = snapshot(wall, Some(1.0), AiState::Loaded, models.clone());
    let llama = view(AiState::Loaded, models, None);
    publisher.publish(&snap, &llama).expect("publish");
    let bytes = std::fs::read(dir.join("snapshot.json")).expect("read");
    let text = std::str::from_utf8(&bytes).expect("utf8");
    assert!(!text.contains("/models"), "{text}");
    let parsed = wire::parse_validated(&bytes).expect("parse");
    assert_eq!(parsed.ai.models[0].name, "Ternary Bon…");
    assert_eq!(
        parsed.ai.models[0].full_name.as_deref(),
        Some("Ternary Bonsai 2 27B")
    );
    assert_eq!(parsed.ai.models[0].detail, Some(detail));
    // Same as the canonical name: omitted. Not allowlisted: dropped.
    assert_eq!(parsed.ai.models[1].full_name, None);
    assert_eq!(parsed.ai.models[1].detail, None);
}

#[test]
fn backend_and_its_gauges_reach_the_wire() {
    use llama_core::backend::{Backend, BackendInfo};

    let dir = scratch("backend");
    let mut publisher = Publisher::open(&dir, Capture::default()).expect("open");
    let wall = SystemTime::UNIX_EPOCH + Duration::from_millis(1);
    let model = |name: &str, backend: Option<BackendInfo>| ModelInfo {
        name: name.to_owned(),
        state: "ready".to_owned(),
        full_name: None,
        detail: None,
        backend,
    };
    let gauges = BackendInfo {
        kind: Backend::SgLang,
        max_running: Some(4),
        running: Some(1),
        queued: Some(0),
        kv_permille: Some(370),
        hit_permille: Some(800),
        engine: Default::default(),
    };
    let models = vec![
        model("flash", Some(gauges)),
        // llama.cpp has /slots: its gauges stay off the wire.
        model(
            "qwen",
            Some(BackendInfo {
                kind: Backend::LlamaCpp,
                ..gauges
            }),
        ),
        model(
            "tabby",
            Some(BackendInfo {
                kind: Backend::OpenAi,
                ..BackendInfo::default()
            }),
        ),
        model("old", None),
    ];
    let snap = snapshot(wall, Some(1.0), AiState::Loaded, models.clone());
    let llama = view(AiState::Loaded, models, None);
    publisher.publish(&snap, &llama).expect("publish");
    let bytes = std::fs::read(dir.join("snapshot.json")).expect("read");
    let parsed = wire::parse_validated(&bytes).expect("parse");
    let m = &parsed.ai.models;
    assert_eq!(m[0].backend, Some(Backend::SgLang));
    assert_eq!((m[0].running, m[0].queued), (Some(1), Some(0)));
    assert_eq!(m[0].kv_fill, Some(0.37));
    assert_eq!(m[1].backend, Some(Backend::LlamaCpp));
    assert_eq!(
        (m[1].running, m[1].queued, m[1].kv_fill),
        (None, None, None)
    );
    assert_eq!(m[2].backend, Some(Backend::OpenAi));
    assert_eq!(
        (m[2].running, m[2].queued, m[2].kv_fill),
        (None, None, None)
    );
    assert_eq!(m[3].backend, None);
    let text = std::str::from_utf8(&bytes).expect("utf8");
    // Since #11 the hit rate is on the wire as `cache_hit`, for llama-metrics.
    assert_eq!(m[0].cache_hit, Some(0.8));
    assert_eq!(m[1].cache_hit, None);
    assert!(!text.contains("max_running"), "{text}");
}

/// #31: a vLLM model's engine numbers reach the wire as seconds, ratios,
/// counters and (#35) tok/s; an empty set and llama.cpp's are left off.
#[test]
fn engine_numbers_reach_the_wire() {
    use llama_core::backend::{Backend, BackendInfo, EngineStats, SpecCounts};

    let dir = scratch("engine");
    let mut publisher = Publisher::open(&dir, Capture::default()).expect("open");
    let wall = SystemTime::UNIX_EPOCH + Duration::from_millis(1);
    let engine = EngineStats {
        spec_permille: Some(780),
        spec_len_centi: Some(290),
        spec_counts: Some(SpecCounts {
            drafts: Some(100),
            draft_tokens: 300,
            accepted: 234,
        }),
        preemptions: Some(3),
        sleeping: Some(false),
        ttft_us: Some(420_000),
        itl_us: Some(31_000),
        e2e_us: Some(12_500_000),
        prefill_tps_tenths: Some(21_345),
        decode_tps_tenths: Some(412),
        expert_hit_permille: Some(856),
        pcie_share_permille: Some(106),
    };
    // #54: Strata counts draft tokens but no rounds.
    let strata = EngineStats {
        spec_permille: Some(700),
        spec_counts: Some(SpecCounts {
            drafts: None,
            draft_tokens: 391,
            accepted: 274,
        }),
        expert_hit_permille: Some(1000),
        pcie_share_permille: Some(0),
        ..EngineStats::default()
    };
    let model = |name: &str, kind: Backend, engine: EngineStats| ModelInfo {
        name: name.to_owned(),
        state: "ready".to_owned(),
        full_name: None,
        detail: None,
        backend: Some(BackendInfo {
            kind,
            engine,
            ..BackendInfo::default()
        }),
    };
    let models = vec![
        model("vllm", Backend::Vllm, engine),
        model("idle", Backend::Vllm, EngineStats::default()),
        model("llama", Backend::LlamaCpp, engine),
        model("strata", Backend::Strata, strata),
    ];
    let snap = snapshot(wall, Some(1.0), AiState::Loaded, models.clone());
    publisher
        .publish(&snap, &view(AiState::Loaded, models, None))
        .expect("publish");
    let bytes = std::fs::read(dir.join("snapshot.json")).expect("read");
    let parsed = wire::parse_validated(&bytes).expect("parse");
    let got = parsed.ai.models[0].engine.clone().expect("engine");
    assert_eq!(
        got,
        wire::EngineWire {
            spec_accept: Some(0.78),
            spec_len: Some(2.9),
            spec_drafts: Some(100),
            spec_draft_tokens: Some(300),
            spec_accepted_tokens: Some(234),
            preemptions: Some(3),
            sleeping: Some(false),
            ttft_s: Some(0.42),
            itl_s: Some(0.031),
            e2e_s: Some(12.5),
            prefill_tps: Some(2134.5),
            decode_tps: Some(41.2),
            expert_hit: Some(0.856),
            pcie_share: Some(0.106),
        }
    );
    assert_eq!(
        parsed.ai.models[3].engine,
        Some(wire::EngineWire {
            spec_accept: Some(0.7),
            spec_draft_tokens: Some(391),
            spec_accepted_tokens: Some(274),
            expert_hit: Some(1.0),
            pcie_share: Some(0.0),
            ..wire::EngineWire::default()
        }),
        "no rounds, no spec_drafts"
    );
    assert_eq!(parsed.ai.models[1].engine, None);
    assert_eq!(
        parsed.ai.models[2].engine, None,
        "llama.cpp keeps its slot view"
    );
}
