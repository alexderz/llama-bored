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
use llama_watch::publish::{PublishError, Publisher};

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
            name: "hello\nworld".to_owned(),
            state: "starting".to_owned(),
            full_name: None,
            detail: None,
        }],
    );
    let llama = view(
        AiState::Loaded,
        vec![ModelInfo {
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
    };
    let models = vec![
        ModelInfo {
            name: "Ternary Bon…".to_owned(),
            state: "ready".to_owned(),
            full_name: Some("Ternary Bonsai 2 27B".to_owned()),
            detail: Some(detail.clone()),
        },
        ModelInfo {
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
