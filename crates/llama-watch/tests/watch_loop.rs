//! The 10 Hz loop against fakes: no live llama-swap, no `/proc`, no device.

use std::collections::VecDeque;
use std::io::{self, Read as _, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime};

use llama_core::log::Sink;
use llama_core::sample::{AiState, LlamaView, ModelInfo, Snapshot};
use llama_core::wire;
use llama_watch::collector::{GpuExtra, WatchCollector, WatchSample};
use llama_watch::config::{Config, ValidWatchConfig};
use llama_watch::poller::{LlamaDetail, PollLatencies};
use llama_watch::publish::{PublishError, Publisher};
use llama_watch::service::{self, Clock, LlamaFeed, LoopInput, Notifier, Stop};
use llama_watch::slots::SlotView;
use llama_watch::sources::Roots;
use llama_watch::sources::gpu::{GpuBackend, GpuError};
use llama_watch::tty::chart::ChartBucket;
use llama_watch::tty::grid::{C16, Cell};
use llama_watch::tty::layout::TtyModel;
use llama_watch::tty::term::{Size, Term};

#[test]
fn seq_advances_once_per_tick_and_each_snapshot_validates() {
    let scratch = Scratch::new("seq");
    let proc = ProcDir::new(&scratch);
    let snap = scratch.path().join("snap");
    std::fs::create_dir_all(&snap).expect("snap dir");
    let cfg_path = scratch.path().join("watch.toml");
    std::fs::write(&cfg_path, config_text(9, 256)).expect("toml");
    let config = Config::load_validated(&cfg_path, 8).expect("config");
    let roots = proc.roots();
    let collector = WatchCollector::new(roots.clone(), FakeGpu::steady(), &config, MemLog::new());
    let seqs = Arc::new(Mutex::new(Vec::new()));
    let publisher = SeqLog {
        inner: Publisher::open(&snap, MemLog::new()).expect("publisher"),
        path: snap.join("snapshot.json"),
        seqs: Arc::clone(&seqs),
    };
    let exit = service::run_loop(LoopInput {
        config: &config,
        feed: Queue::default(),
        sampler: collector,
        publisher,
        render: NoopRender,
        clock: FakeClock {
            mono: Instant::now(),
            wall: SystemTime::UNIX_EPOCH + Duration::from_secs(1_800_000_000),
        },
        notify: RecNotify::default(),
        stop: StopAfter::new(3),
        log: MemLog::new(),
        roots: &roots,
        started: Instant::now(),
    });
    assert_eq!(exit, service::LoopExit::Stopped);
    assert_eq!(
        seqs.lock().expect("seqs").clone(),
        vec![1, 2, 3],
        "each tick publishes the next seq"
    );
    let bytes = std::fs::read(snap.join("snapshot.json")).unwrap_or_default();
    assert!(
        wire::parse_validated(&bytes).is_ok(),
        "published bytes must validate"
    );
}

#[test]
fn three_ticks_send_ready_then_a_watchdog_each_and_stopping() {
    let fx = Fixture::new("notify");
    let notify = RecNotify::default();
    let events = Arc::clone(&notify.events);
    let exit = fx.drive(
        Queue::default(),
        fx.collector(),
        OkPublish,
        NoopRender,
        notify,
        StopAfter::new(3),
        MemLog::new(),
    );
    assert_eq!(exit, service::LoopExit::Stopped);
    assert_eq!(
        events.lock().expect("events").clone(),
        vec!["READY", "WATCHDOG", "WATCHDOG", "WATCHDOG", "STOPPING"]
    );
}

#[test]
fn a_panicking_step_skips_the_watchdog() {
    let fx = Fixture::new("panic");
    let notify = RecNotify::default();
    let events = Arc::clone(&notify.events);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fx.drive(
            Queue::default(),
            Boom,
            OkPublish,
            NoopRender,
            notify,
            StopAfter::new(1),
            MemLog::new(),
        )
    }));
    assert!(result.is_err(), "the injected panic must escape the tick");
    assert_eq!(events.lock().expect("events").clone(), vec!["READY"]);
}

#[test]
fn a_hung_step_skips_the_watchdog_until_it_returns() {
    let fx = Fixture::new("hang");
    let notify = RecNotify::default();
    let events = Arc::clone(&notify.events);
    let entered = Arc::new(AtomicBool::new(false));
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let entered_tick = Arc::clone(&entered);
    let release_tick = Arc::clone(&release);
    std::thread::scope(|scope| {
        let handle = scope.spawn(|| {
            fx.drive(
                Queue::default(),
                Hang {
                    entered: entered_tick,
                    release: release_tick,
                },
                OkPublish,
                NoopRender,
                notify,
                FlagStop(Arc::clone(&stop)),
                MemLog::new(),
            )
        });
        let start = Instant::now();
        while !entered.load(Ordering::SeqCst) {
            assert!(
                start.elapsed() < Duration::from_secs(2),
                "tick did not block"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            events.lock().expect("events").clone(),
            vec!["READY"],
            "watchdog waits until the hung step returns"
        );
        stop.store(true, Ordering::SeqCst);
        {
            let (lock, cv) = &*release;
            *lock.lock().expect("gate") = true;
            cv.notify_one();
        }
        handle.join().expect("loop thread");
    });
    let events = events.lock().expect("events").clone();
    assert_eq!(*events.last().expect("stopping"), "STOPPING");
    assert!(events.contains(&"WATCHDOG"));
}

#[test]
fn too_small_window_draws_only_the_banner() {
    let fx = Fixture::new("tiny");
    let calls = Arc::new(Mutex::new(Vec::new()));
    let term = term_at(80, 24, Arc::clone(&calls));
    fx.drive(
        Queue::default(),
        fx.collector(),
        OkPublish,
        term,
        RecNotify::default(),
        StopAfter::new(1),
        MemLog::new(),
    );
    let bytes = calls.lock().expect("writes").iter().sum::<usize>();
    assert!(bytes > 0, "the too-small path still paints its one line");
}

#[test]
fn an_unchanged_full_frame_writes_zero_bytes() {
    let fx = Fixture::new("frame");
    let calls = Arc::new(Mutex::new(Vec::new()));
    let term = term_at(160, 48, Arc::clone(&calls));
    fx.drive(
        Queue::default(),
        fx.collector(),
        Flaky { left: 8 },
        term,
        RecNotify::default(),
        StopAfter::new(2),
        MemLog::new(),
    );
    let calls = calls.lock().expect("writes").clone();
    assert_eq!(
        calls.len(),
        1,
        "two ticks in the same second with no new snapshot must not redraw: {calls:?}"
    );
    assert!(calls[0] > 0);
}

#[test]
fn the_newest_queued_sample_wins() {
    let fx = Fixture::new("latest");
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut feed = Queue::default();
    feed.items.push_back((view_named("alpha"), empty_detail()));
    feed.items.push_back((view_named("beta"), empty_detail()));
    fx.drive(
        feed,
        fx.collector(),
        SeenPublish {
            names: Arc::clone(&seen),
        },
        NoopRender,
        RecNotify::default(),
        StopAfter::new(1),
        MemLog::new(),
    );
    assert_eq!(seen.lock().expect("names").clone(), vec!["beta".to_owned()]);
}

#[test]
fn a_stalled_poller_does_not_stop_ticks() {
    let server = HangServer::start();
    let fx = Fixture::with_port("stall", server.port, 256);
    let log = MemLog::new();
    let (poller, rx) = llama_watch::poller::spawn(fx.config(), log).expect("spawn once");
    let start = Instant::now();
    while server.hits.load(Ordering::Relaxed) == 0 {
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "poller never connected"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let seqs = Arc::new(Mutex::new(Vec::new()));
    let snap = fx.scratch.path().join("snap");
    std::fs::create_dir_all(&snap).expect("snap");
    let publisher = SeqLog {
        inner: Publisher::open(&snap, MemLog::new()).expect("publisher"),
        path: snap.join("snapshot.json"),
        seqs: Arc::clone(&seqs),
    };
    fx.drive(
        service::SlotFeed::new(rx),
        fx.collector(),
        publisher,
        NoopRender,
        RecNotify::default(),
        StopAfter::new(5),
        MemLog::new(),
    );
    // The hung server would block a waiting recv. Seq still advances because
    // the loop only take()s a sample and does not wait for the poller.
    assert_eq!(seqs.lock().expect("seqs").clone(), vec![1, 2, 3, 4, 5]);
    drop(poller);
    drop(server);
}

#[test]
fn write_errors_are_logged_once_per_transition() {
    let fx = Fixture::new("write");
    let log = MemLog::new();
    let lines = log.lines_handle();
    fx.drive(
        Queue::default(),
        fx.collector(),
        Flaky { left: 3 },
        NoopRender,
        RecNotify::default(),
        StopAfter::new(5),
        log,
    );
    let lines = lines.lock().expect("log").clone();
    let fails = lines
        .iter()
        .filter(|line| line.contains("snapshot write failed"))
        .count();
    let recoveries = lines
        .iter()
        .filter(|line| line.contains("snapshot write recovered"))
        .count();
    assert_eq!(fails, 1, "{lines:?}");
    assert_eq!(recoveries, 1, "{lines:?}");
}

#[test]
fn the_layout_receives_a_capped_tail() {
    let fx = Fixture::with_port("cap", 9, 256);
    let models = Arc::new(Mutex::new(Vec::new()));
    let mut feed = Queue::default();
    feed.items
        .push_back((view_named("beta"), detail_output(&"x".repeat(300))));
    fx.drive(
        feed,
        fx.collector(),
        OkPublish,
        RecRender {
            models: Arc::clone(&models),
        },
        RecNotify::default(),
        StopAfter::new(1),
        MemLog::new(),
    );
    let models = models.lock().expect("models").clone();
    let tail = models[0].out_lines.concat();
    assert_eq!(tail.chars().count(), 256);
    assert!(tail.chars().all(|ch| ch == 'x'));
}

#[test]
fn tick_loop_keeps_a_chart_ring_from_the_live_rates() {
    let fx = Fixture::new("chart-ring");
    let models = Arc::new(Mutex::new(Vec::new()));
    let mut feed = Queue::default();
    let mut detail = empty_detail();
    detail.gen_tps = Some(20.0);
    detail.prompt_tps = Some(400.0);
    feed.items.push_back((view_named("beta"), detail));
    fx.drive(
        feed,
        fx.collector(),
        OkPublish,
        RecRender {
            models: Arc::clone(&models),
        },
        RecNotify::default(),
        StopAfter::new(25),
        MemLog::new(),
    );
    let models = models.lock().expect("models").clone();
    assert_eq!(models.len(), 25);
    assert_eq!(models[0].chart_bucket_s, 2);
    assert_eq!(
        models[0].chart,
        vec![ChartBucket {
            gen_tps: Some(20.0),
            prompt_tps: Some(400.0),
        }]
    );
    let last = &models[24].chart;
    assert_eq!(last.len(), 2, "{last:?}");
    assert!(
        last.iter()
            .all(|b| b.gen_tps == Some(20.0) && b.prompt_tps == Some(400.0)),
        "{last:?}"
    );
}

#[test]
fn the_layout_receives_the_configured_rate_ceilings() {
    let fx = Fixture::new("ceilings");
    let models = Arc::new(Mutex::new(Vec::new()));
    let mut feed = Queue::default();
    feed.items.push_back((view_named("beta"), empty_detail()));
    fx.drive(
        feed,
        fx.collector(),
        OkPublish,
        RecRender {
            models: Arc::clone(&models),
        },
        RecNotify::default(),
        StopAfter::new(1),
        MemLog::new(),
    );
    let models = models.lock().expect("models").clone();
    assert_eq!(models[0].gen_ceiling, 227.0, "tty.gen_ceiling_tps");
    assert_eq!(models[0].prompt_ceiling, 915.0, "tty.prompt_ceiling_tps");
}

#[test]
fn trimming_the_tail_head_rewinds_out_shown() {
    let fx = Fixture::with_port("rewind", 9, 256);
    let models = Arc::new(Mutex::new(Vec::new()));
    let mut feed = Paced::default();
    let same = detail_output(&"a".repeat(256));
    for _ in 0..11 {
        feed.items.push_back((view_named("beta"), same.clone()));
    }
    feed.items.push_back((
        view_named("beta"),
        detail_output(&format!("{}{}", "a".repeat(256), "b".repeat(10))),
    ));
    fx.drive(
        feed,
        fx.collector(),
        OkPublish,
        RecRender {
            models: Arc::clone(&models),
        },
        RecNotify::default(),
        StopAfter::new(12),
        MemLog::new(),
    );
    let models = models.lock().expect("models").clone();
    assert_eq!(models.len(), 12);
    let last = models.last().expect("last frame");
    let tail = last.out_lines.concat();
    assert_eq!(tail.chars().count(), 256);
    assert!(tail.ends_with("bbbbbbbbbb"), "{tail}");
    assert_eq!(last.out_shown, 246);
    assert_eq!(last.replay_frame, Some(0));
}

#[test]
fn a_non_uniform_tail_rewinds_shown_by_the_dropped_prefix() {
    let fx = Fixture::with_port("prose", 9, 256);
    let models = Arc::new(Mutex::new(Vec::new()));
    let base = prose(256);
    let grown = format!("{base}XYZ");
    let mut feed = Paced::default();
    for _ in 0..11 {
        feed.items
            .push_back((view_named("beta"), detail_output(&base)));
    }
    feed.items
        .push_back((view_named("beta"), detail_output(&grown)));
    fx.drive(
        feed,
        fx.collector(),
        OkPublish,
        RecRender {
            models: Arc::clone(&models),
        },
        RecNotify::default(),
        StopAfter::new(12),
        MemLog::new(),
    );
    let models = models.lock().expect("models").clone();
    let last = models.last().expect("last");
    let tail = last.out_lines.concat();
    assert_eq!(tail.chars().count(), 256);
    assert!(tail.ends_with("XYZ"), "{tail}");
    assert_eq!(last.out_shown, 253, "dropped 3 chars from the head");
    assert_ne!(last.out_shown, 0);
}

#[test]
fn a_growing_busy_slot_does_not_reset_shown_when_another_slot_is_stale() {
    let fx = Fixture::with_port("slots", 9, 256);
    let models = Arc::new(Mutex::new(Vec::new()));
    let mut feed = Paced::default();
    let stale = "STALE-TAIL";
    let first = detail_two_slots("hello", stale);
    for _ in 0..11 {
        feed.items.push_back((view_named("beta"), first.clone()));
    }
    feed.items
        .push_back((view_named("beta"), detail_two_slots("hello!", stale)));
    fx.drive(
        feed,
        fx.collector(),
        OkPublish,
        RecRender {
            models: Arc::clone(&models),
        },
        RecNotify::default(),
        StopAfter::new(12),
        MemLog::new(),
    );
    let models = models.lock().expect("models").clone();
    let before = models[10].out_lines.concat();
    let last = models.last().expect("grown");
    let tail = last.out_lines.concat();
    assert!(
        tail.ends_with("hello!"),
        "the busy slot's text is the end of OUT: {tail}"
    );
    assert_eq!(
        last.out_shown,
        before.chars().count(),
        "shown keeps the already-revealed prefix, got tail {tail}"
    );
    assert_ne!(last.out_shown, 0);
    assert_eq!(last.replay_frame, Some(0));
}

#[test]
fn two_busy_slots_growing_together_do_not_blank_the_out_panel() {
    let fx = Fixture::with_port("both-busy", 9, 256);
    let models = Arc::new(Mutex::new(Vec::new()));
    let mut feed = Paced::default();
    let first = detail_both_busy("AAAA", "BBBB");
    for _ in 0..11 {
        feed.items.push_back((view_named("beta"), first.clone()));
    }
    feed.items
        .push_back((view_named("beta"), detail_both_busy("AAAAX", "BBBBX")));
    feed.items
        .push_back((view_named("beta"), detail_both_busy("AAAAXY", "BBBBXY")));
    fx.drive(
        feed,
        fx.collector(),
        OkPublish,
        RecRender {
            models: Arc::clone(&models),
        },
        RecNotify::default(),
        StopAfter::new(13),
        MemLog::new(),
    );
    let models = models.lock().expect("models").clone();
    let grown = &models[11];
    assert_eq!(grown.replay_frame, Some(0));
    assert!(
        grown.out_shown > 0,
        "frame 0 keeps the shared prefix, tail {}",
        grown.out_lines.concat()
    );
    let again = models.last().expect("second growth");
    assert!(
        again.out_shown > 0,
        "a second growth still does not blank, tail {}",
        again.out_lines.concat()
    );
}

#[test]
fn a_new_request_with_no_shared_prefix_resets_shown() {
    let fx = Fixture::with_port("fresh", 9, 256);
    let models = Arc::new(Mutex::new(Vec::new()));
    let mut feed = Paced::default();
    let first = detail_output("hello world");
    for _ in 0..11 {
        feed.items.push_back((view_named("beta"), first.clone()));
    }
    feed.items
        .push_back((view_named("beta"), detail_output("ZZZZZZZZZZ")));
    fx.drive(
        feed,
        fx.collector(),
        OkPublish,
        RecRender {
            models: Arc::clone(&models),
        },
        RecNotify::default(),
        StopAfter::new(12),
        MemLog::new(),
    );
    let models = models.lock().expect("models").clone();
    let last = models.last().expect("new request");
    assert_eq!(last.out_lines.concat(), "ZZZZZZZZZZ");
    assert_eq!(last.out_shown, 0);
    assert_eq!(last.replay_frame, Some(0));
}

#[test]
fn snapshot_age_tracks_the_last_successful_publish() {
    let fx = Fixture::new("age");
    let models = Arc::new(Mutex::new(Vec::new()));
    fx.drive(
        Queue::default(),
        fx.collector(),
        FailAfterOk { left: 1 },
        RecRender {
            models: Arc::clone(&models),
        },
        RecNotify::default(),
        StopAfter::new(31),
        MemLog::new(),
    );
    let models = models.lock().expect("models").clone();
    let last = models.last().expect("last tick");
    assert_eq!(
        last.snapshot,
        Some(1),
        "failed publishes must not advance #N"
    );
    assert_eq!(last.snapshot_age, "3.0s", "age since the last good publish");
}

#[test]
fn draw_errors_are_logged_once_per_transition() {
    let fx = Fixture::new("draw");
    let log = MemLog::new();
    let lines = log.lines_handle();
    fx.drive(
        Queue::default(),
        fx.collector(),
        OkPublish,
        FlakyDraw { left: 3 },
        RecNotify::default(),
        StopAfter::new(5),
        log,
    );
    let lines = lines.lock().expect("log").clone();
    let fails = lines
        .iter()
        .filter(|line| line.contains("console draw failed"))
        .count();
    let recoveries = lines
        .iter()
        .filter(|line| line.contains("console draw recovered"))
        .count();
    assert_eq!(fails, 1, "{lines:?}");
    assert_eq!(recoveries, 1, "{lines:?}");
}

#[test]
fn memory_comes_from_the_sample_and_host_facts_are_read_once() {
    let fx = Fixture::new("facts");
    write_hostname(&fx.roots.proc, "alpha");
    let models = Arc::new(Mutex::new(Vec::new()));
    let sampler = EditOnSecond {
        inner: fx.collector(),
        proc_root: fx.roots.proc.clone(),
        ticks: 0,
    };
    fx.drive(
        Queue::default(),
        sampler,
        OkPublish,
        RecRender {
            models: Arc::clone(&models),
        },
        RecNotify::default(),
        StopAfter::new(2),
        MemLog::new(),
    );
    let models = models.lock().expect("models").clone();
    assert_eq!(models[0].host, "ALPHA");
    assert_eq!(models[1].host, "ALPHA", "hostname is fixed at startup");
    assert_eq!(models[0].cpu_cores, Some(1));
    assert_eq!(
        models[1].cpu_cores,
        Some(1),
        "core count is fixed at startup"
    );

    let mem_models = Arc::new(Mutex::new(Vec::new()));
    fx.drive(
        Queue::default(),
        MemPct(0.0),
        OkPublish,
        RecRender {
            models: Arc::clone(&mem_models),
        },
        RecNotify::default(),
        StopAfter::new(1),
        MemLog::new(),
    );
    let frame = &mem_models.lock().expect("mem")[0];
    let used = frame.mem_used_gb.expect("used");
    let total = frame.mem_total_gb.expect("total");
    assert!(
        used < 0.05,
        "used GB follows the sample percent, got {used} of {total}"
    );
    assert!(
        total > 20.0,
        "total comes from the startup read, got {total}"
    );
}

#[test]
fn an_overrun_starts_the_next_tick_at_once_and_the_following_tick_waits() {
    let fx = Fixture::new("deadline");
    let mono = Arc::new(Mutex::new(Instant::now()));
    let slept = Arc::new(Mutex::new(Vec::new()));
    let clock = SharedClock {
        mono: Arc::clone(&mono),
        wall: SystemTime::UNIX_EPOCH + Duration::from_secs(1_800_000_000),
        slept: Arc::clone(&slept),
    };
    service::run_loop(service::LoopInput {
        config: fx.config(),
        feed: Queue::default(),
        sampler: JumpClock {
            mono: Arc::clone(&mono),
            jumped: false,
        },
        publisher: OkPublish,
        render: NoopRender,
        clock,
        notify: RecNotify::default(),
        stop: StopAfter::new(2),
        log: MemLog::new(),
        roots: &fx.roots,
        started: Instant::now(),
    });
    let slept = slept.lock().expect("sleeps").clone();
    let waits: Vec<_> = slept.into_iter().filter(|d| !d.is_zero()).collect();
    assert_eq!(
        waits,
        vec![Duration::from_millis(100)],
        "a 150 ms tick does not sleep; the next tick waits one period"
    );
}

#[test]
fn only_run_config_is_a_command_and_root_exits_2() {
    assert!(service::parse_args(["run", "--config", "watch.toml"]).is_ok());
    assert!(service::parse_args(["restore-stock"]).is_err());
    assert!(service::parse_args(["run"]).is_err());
    assert!(service::parse_args(std::iter::empty::<&str>()).is_err());
    assert_eq!(service::root_exit(true), Some(2));
    assert_eq!(service::root_exit(false), None);
}

#[test]
fn run_accepts_no_text_in_either_order() {
    let plain = service::parse_args(["run", "--config", "watch.toml"]).expect("plain run");
    assert_eq!(plain.config, std::path::PathBuf::from("watch.toml"));
    assert!(!plain.no_text);
    for args in [
        ["run", "--config", "watch.toml", "--no-text"],
        ["run", "--no-text", "--config", "watch.toml"],
    ] {
        let parsed = service::parse_args(args).expect("run --no-text");
        assert_eq!(parsed.config, std::path::PathBuf::from("watch.toml"));
        assert!(parsed.no_text, "{args:?}");
    }
    assert!(
        service::parse_args(["run", "--config", "watch.toml", "--no-text", "--no-text"]).is_err()
    );
    assert!(service::parse_args(["run", "--no-text"]).is_err());
    assert!(service::parse_args(["run", "--config", "watch.toml", "--text"]).is_err());
    assert!(service::parse_args(["run", "--config", "--no-text"]).is_err());
}

#[test]
fn a_missing_snapshot_directory_is_fatal_and_does_not_start_a_poller() {
    let scratch = Scratch::new("boot");
    let cfg = scratch.path().join("watch.toml");
    std::fs::write(&cfg, config_text(9, 256)).expect("toml");
    match service::prepare(&cfg, 8, &scratch.path().join("missing")) {
        Err(service::PrepareError::Snapshot(_)) => {}
        Err(err) => panic!("expected a snapshot error, got {err}"),
        Ok(_) => panic!("a missing snapshot directory opened"),
    }
}

struct SeqLog<P> {
    inner: P,
    path: PathBuf,
    seqs: Arc<Mutex<Vec<u64>>>,
}

impl<P: service::PublishStep> service::PublishStep for SeqLog<P> {
    fn publish(
        &mut self,
        snapshot: &llama_core::sample::Snapshot,
        llama: &LlamaView,
    ) -> Result<(), llama_watch::publish::PublishError> {
        self.inner.publish(snapshot, llama)?;
        let bytes = std::fs::read(&self.path).unwrap_or_default();
        let wire = wire::parse_validated(&bytes).expect("tick snapshot validates");
        self.seqs.lock().expect("seqs").push(wire.seq);
        Ok(())
    }
}

struct NoopRender;

impl service::RenderStep for NoopRender {
    fn draw(
        &mut self,
        _model: &llama_watch::tty::layout::TtyModel,
        _now: Instant,
    ) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Default)]
struct Queue {
    items: std::collections::VecDeque<(LlamaView, llama_watch::poller::LlamaDetail)>,
}

impl LlamaFeed for Queue {
    fn poll(&mut self) -> Option<(LlamaView, llama_watch::poller::LlamaDetail)> {
        self.items.pop_front()
    }
}

struct FakeClock {
    mono: Instant,
    wall: SystemTime,
}

impl Clock for FakeClock {
    fn mono(&self) -> Instant {
        self.mono
    }

    fn wall(&self) -> SystemTime {
        self.wall
    }

    fn sleep(&mut self, d: Duration) {
        self.mono += d;
        self.wall += d;
    }
}

struct StopAfter {
    seen: AtomicU64,
    limit: u64,
}

impl StopAfter {
    fn new(ticks: u64) -> Self {
        Self {
            seen: AtomicU64::new(0),
            limit: ticks,
        }
    }
}

impl Stop for StopAfter {
    fn requested(&self) -> bool {
        let prev = self.seen.fetch_add(1, Ordering::Relaxed);
        prev >= self.limit
    }
}

#[derive(Clone, Default)]
struct RecNotify {
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl Notifier for RecNotify {
    fn ready(&mut self) {
        self.push("READY");
    }

    fn watchdog(&mut self) {
        self.push("WATCHDOG");
    }

    fn stopping(&mut self) {
        self.push("STOPPING");
    }
}

impl RecNotify {
    fn push(&self, event: &'static str) {
        self.events.lock().expect("events").push(event);
    }
}

#[derive(Clone)]
struct MemLog(Arc<Mutex<Vec<String>>>);

impl MemLog {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }

    fn lines_handle(&self) -> Arc<Mutex<Vec<String>>> {
        Arc::clone(&self.0)
    }
}

impl Sink for MemLog {
    fn write_line(&mut self, line: &str) {
        self.0.lock().expect("log").push(line.to_owned());
    }
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("watch-loop-{label}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path).expect("scratch");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct ProcDir {
    dir: PathBuf,
}

impl ProcDir {
    fn new(scratch: &Scratch) -> Self {
        let dir = scratch.path().join("proc");
        std::fs::create_dir_all(&dir).expect("proc");
        let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        std::fs::copy(fixtures.join("proc/mem/meminfo"), dir.join("meminfo")).expect("meminfo");
        std::fs::write(
            dir.join("stat"),
            "cpu  0 0 0 0 0 0 0 0\ncpu0 100 0 0 100 0 0 0 0\n",
        )
        .expect("stat");
        Self { dir }
    }

    fn roots(&self) -> Roots {
        Roots {
            proc: self.dir.clone(),
            sys: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/sys"),
        }
    }
}

struct FakeGpu;

impl FakeGpu {
    fn steady() -> Self {
        Self
    }
}

impl GpuBackend for FakeGpu {
    fn init(&mut self) -> Result<(), GpuError> {
        Ok(())
    }

    fn util_pct(&mut self) -> Result<f32, GpuError> {
        Ok(10.0)
    }

    fn temp_c(&mut self) -> Result<f32, GpuError> {
        Ok(40.0)
    }

    fn memory_info(&mut self) -> Result<(u64, u64), GpuError> {
        Ok((1_000, 2_000))
    }

    fn power_usage(&mut self) -> Result<u32, GpuError> {
        Ok(30_000)
    }

    fn enforced_power_limit(&mut self) -> Result<u32, GpuError> {
        Ok(100_000)
    }

    fn disconnect(&mut self) {}
}

fn config_text(port: u16, output_tail: u32) -> String {
    format!(
        r#"
[collector]
tick_s = 0.1
cpu_window_s = 1.0
cpu_top_k = 1

[llama]
url = "http://127.0.0.1:{port}"
running_interval_s = 0.35
running_timeout_s = 0.15
metrics_interval_s = 0.2
metrics_timeout_s = 0.1
slots_interval_s = 0.5
slots_timeout_s = 0.3
slots_max_bytes = 4194304
activity_interval_s = 0.35
activity_timeout_s = 0.1
input_tail_chars = 256
output_tail_chars = {output_tail}

[models]
max_name_chars = 12

[tty]
fps = 10
full_redraw_s = 5
gen_ceiling_tps = 227
prompt_ceiling_tps = 915
"#
    )
}

struct Fixture {
    scratch: Scratch,
    config: ValidWatchConfig,
    roots: Roots,
}

impl Fixture {
    fn new(label: &str) -> Self {
        Self::with_port(label, 9, 256)
    }

    fn with_port(label: &str, port: u16, tail: u32) -> Self {
        let scratch = Scratch::new(label);
        let proc_dir = ProcDir::new(&scratch);
        let roots = proc_dir.roots();
        let cfg = scratch.path().join("watch.toml");
        std::fs::write(&cfg, config_text(port, tail)).expect("toml");
        let config = Config::load_validated(&cfg, 8).expect("config");
        Self {
            scratch,
            config,
            roots,
        }
    }

    fn config(&self) -> &ValidWatchConfig {
        &self.config
    }

    fn collector(&self) -> WatchCollector<'_, FakeGpu, MemLog> {
        WatchCollector::new(
            self.roots.clone(),
            FakeGpu::steady(),
            &self.config,
            MemLog::new(),
        )
    }

    // One parameter per loop input. Folding them would hide the harness.
    #[allow(clippy::too_many_arguments)]
    fn drive<Feed, Samp, Pub, Rend, Ntf, Stp, Lg>(
        &self,
        feed: Feed,
        sampler: Samp,
        publisher: Pub,
        render: Rend,
        notify: Ntf,
        stop: Stp,
        log: Lg,
    ) -> service::LoopExit
    where
        Feed: LlamaFeed,
        Samp: service::SampleStep,
        Pub: service::PublishStep,
        Rend: service::RenderStep,
        Ntf: Notifier,
        Stp: Stop,
        Lg: Sink,
    {
        service::run_loop(LoopInput {
            config: &self.config,
            feed,
            sampler,
            publisher,
            render,
            clock: FakeClock {
                mono: Instant::now(),
                wall: SystemTime::UNIX_EPOCH + Duration::from_secs(1_800_000_000),
            },
            notify,
            stop,
            log,
            roots: &self.roots,
            started: Instant::now(),
        })
    }
}

struct OkPublish;

impl service::PublishStep for OkPublish {
    fn publish(&mut self, _snapshot: &Snapshot, _llama: &LlamaView) -> Result<(), PublishError> {
        Ok(())
    }
}

struct SeenPublish {
    names: Arc<Mutex<Vec<String>>>,
}

impl service::PublishStep for SeenPublish {
    fn publish(&mut self, _snapshot: &Snapshot, llama: &LlamaView) -> Result<(), PublishError> {
        let name = llama
            .models
            .first()
            .map(|model| model.name.clone())
            .unwrap_or_default();
        self.names.lock().expect("names").push(name);
        Ok(())
    }
}

struct Flaky {
    left: usize,
}

impl service::PublishStep for Flaky {
    fn publish(&mut self, _snapshot: &Snapshot, _llama: &LlamaView) -> Result<(), PublishError> {
        if self.left > 0 {
            self.left -= 1;
            return Err(PublishError::Write(io::Error::other("disk")));
        }
        Ok(())
    }
}

struct Boom;

impl service::SampleStep for Boom {
    fn sample(&mut self, _mono: Instant, _wall: SystemTime, _llama: &LlamaView) -> WatchSample {
        panic!("injected sample panic");
    }
}

struct Hang {
    entered: Arc<AtomicBool>,
    release: Arc<(Mutex<bool>, Condvar)>,
}

impl service::SampleStep for Hang {
    fn sample(&mut self, mono: Instant, wall: SystemTime, _llama: &LlamaView) -> WatchSample {
        self.entered.store(true, Ordering::SeqCst);
        let (lock, cv) = &*self.release;
        let mut go = lock.lock().expect("gate");
        while !*go {
            go = cv.wait(go).expect("gate wait");
        }
        blank_sample(mono, wall)
    }
}

struct FlagStop(Arc<AtomicBool>);

impl Stop for FlagStop {
    fn requested(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

struct LogWrite {
    calls: Arc<Mutex<Vec<usize>>>,
}

impl Write for LogWrite {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.calls.lock().expect("writes").push(buf.len());
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct RecRender {
    models: Arc<Mutex<Vec<TtyModel>>>,
}

impl service::RenderStep for RecRender {
    fn draw(&mut self, model: &TtyModel, _now: Instant) -> io::Result<()> {
        self.models.lock().expect("models").push(model.clone());
        Ok(())
    }
}

#[derive(Default)]
struct Paced {
    items: VecDeque<(LlamaView, LlamaDetail)>,
    hold: bool,
}

impl LlamaFeed for Paced {
    fn poll(&mut self) -> Option<(LlamaView, LlamaDetail)> {
        if self.hold {
            self.hold = false;
            return None;
        }
        let item = self.items.pop_front()?;
        self.hold = true;
        Some(item)
    }
}

struct HangServer {
    port: u16,
    hits: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl HangServer {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let port = listener.local_addr().expect("addr").port();
        let hits = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let hits_thread = Arc::clone(&hits);
        let stop_thread = Arc::clone(&stop);
        let join = std::thread::spawn(move || {
            while !stop_thread.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        hits_thread.fetch_add(1, Ordering::Relaxed);
                        let _ = stream.set_read_timeout(Some(Duration::from_millis(200)));
                        let mut buf = [0u8; 2048];
                        let _ = stream.read(&mut buf);
                        while !stop_thread.load(Ordering::Relaxed) {
                            std::thread::sleep(Duration::from_millis(10));
                        }
                    }
                    Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            port,
            hits,
            stop,
            join: Some(join),
        }
    }
}

impl Drop for HangServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn view_named(name: &str) -> LlamaView {
    LlamaView {
        ai: AiState::Loaded,
        models: vec![ModelInfo {
            name: name.to_owned(),
            state: "ready".to_owned(),
            full_name: None,
            detail: None,
        }],
        decoded_total: Some(1),
    }
}

fn empty_detail() -> LlamaDetail {
    LlamaDetail {
        slots: Vec::new(),
        activity: Vec::new(),
        gen_tps: None,
        prompt_tps: None,
        latencies: PollLatencies::default(),
    }
}

fn detail_output(text: &str) -> LlamaDetail {
    let mut detail = empty_detail();
    detail.slots.push(SlotView {
        model: "m".to_owned(),
        id: 0,
        id_task: 1,
        is_processing: false,
        n_prompt_tokens: 4,
        n_prompt_tokens_processed: 4,
        n_decoded: 1,
        n_ctx: None,
        ctx_prompt: None,
        input: Vec::new(),
        output: text
            .chars()
            .map(|ch| Cell::new(ch, C16::White, C16::Black))
            .collect(),
    });
    detail
}

fn detail_both_busy(left: &str, right: &str) -> LlamaDetail {
    let mut detail = empty_detail();
    detail.slots.push(slot_text(0, left, true));
    detail.slots.push(slot_text(1, right, true));
    detail
}

fn detail_two_slots(busy: &str, stale: &str) -> LlamaDetail {
    let mut detail = empty_detail();
    detail.slots.push(slot_text(0, busy, true));
    detail.slots.push(slot_text(1, stale, false));
    detail
}

fn slot_text(id: i64, text: &str, busy: bool) -> SlotView {
    SlotView {
        model: "m".to_owned(),
        id,
        id_task: id,
        is_processing: busy,
        n_prompt_tokens: 4,
        n_prompt_tokens_processed: 4,
        n_decoded: 1,
        n_ctx: None,
        ctx_prompt: None,
        input: Vec::new(),
        output: text
            .chars()
            .map(|ch| Cell::new(ch, C16::White, C16::Black))
            .collect(),
    }
}

fn prose(len: usize) -> String {
    let sentence = "The quick brown fox jumps over the lazy dog. ";
    let mut out = String::new();
    while out.chars().count() < len {
        out.push_str(sentence);
    }
    out.chars().take(len).collect()
}

fn term_at(cols: u16, rows: u16, calls: Arc<Mutex<Vec<usize>>>) -> Term<LogWrite> {
    Term::new(
        LogWrite { calls },
        move || Ok(Size { cols, rows }),
        Duration::from_secs(5),
        Instant::now(),
    )
    .expect("term")
}

fn write_hostname(proc_root: &Path, name: &str) {
    let dir = proc_root.join("sys/kernel");
    std::fs::create_dir_all(&dir).expect("hostname dir");
    std::fs::write(dir.join("hostname"), format!("{name}\n")).expect("hostname");
}

struct FailAfterOk {
    left: usize,
}

impl service::PublishStep for FailAfterOk {
    fn publish(&mut self, _snapshot: &Snapshot, _llama: &LlamaView) -> Result<(), PublishError> {
        if self.left > 0 {
            self.left -= 1;
            Ok(())
        } else {
            Err(PublishError::Write(io::Error::other("disk")))
        }
    }
}

struct FlakyDraw {
    left: usize,
}

impl service::RenderStep for FlakyDraw {
    fn draw(&mut self, _model: &TtyModel, _now: Instant) -> io::Result<()> {
        if self.left > 0 {
            self.left -= 1;
            Err(io::Error::other("tty"))
        } else {
            Ok(())
        }
    }
}

struct EditOnSecond<S> {
    inner: S,
    proc_root: PathBuf,
    ticks: u32,
}

impl<S: service::SampleStep> service::SampleStep for EditOnSecond<S> {
    fn sample(&mut self, mono: Instant, wall: SystemTime, llama: &LlamaView) -> WatchSample {
        self.ticks += 1;
        if self.ticks == 2 {
            write_hostname(&self.proc_root, "beta");
            std::fs::write(
                self.proc_root.join("stat"),
                "cpu  0 0 0 0 0 0 0 0\n\
                 cpu0 1 0 0 1 0 0 0 0\n\
                 cpu1 1 0 0 1 0 0 0 0\n\
                 cpu2 1 0 0 1 0 0 0 0\n\
                 cpu3 1 0 0 1 0 0 0 0\n",
            )
            .expect("stat");
        }
        self.inner.sample(mono, wall, llama)
    }
}

struct MemPct(f32);

impl service::SampleStep for MemPct {
    fn sample(&mut self, mono: Instant, wall: SystemTime, _llama: &LlamaView) -> WatchSample {
        let mut sample = blank_sample(mono, wall);
        sample.snapshot.mem_pct = Some(self.0);
        sample
    }
}

struct SharedClock {
    mono: Arc<Mutex<Instant>>,
    wall: SystemTime,
    slept: Arc<Mutex<Vec<Duration>>>,
}

impl Clock for SharedClock {
    fn mono(&self) -> Instant {
        *self.mono.lock().expect("mono")
    }

    fn wall(&self) -> SystemTime {
        self.wall
    }

    fn sleep(&mut self, d: Duration) {
        self.slept.lock().expect("slept").push(d);
        *self.mono.lock().expect("mono") += d;
    }
}

struct JumpClock {
    mono: Arc<Mutex<Instant>>,
    jumped: bool,
}

impl service::SampleStep for JumpClock {
    fn sample(&mut self, mono: Instant, wall: SystemTime, _llama: &LlamaView) -> WatchSample {
        if !self.jumped {
            *self.mono.lock().expect("mono") += Duration::from_millis(150);
            self.jumped = true;
        }
        blank_sample(mono, wall)
    }
}

fn blank_sample(mono: Instant, wall: SystemTime) -> WatchSample {
    WatchSample {
        snapshot: Snapshot {
            t_mono: mono,
            t_wall: wall,
            load: None,
            activity: None,
            cpu_pct: None,
            cpu_topk_pct: None,
            gpu_pct: None,
            mem_pct: None,
            coolant_c: None,
            cpu_c: None,
            gpu_c: None,
            ai: AiState::Idle,
            models: Vec::new(),
            tokens: None,
            errors: std::collections::BTreeSet::new(),
        },
        gpu: GpuExtra::default(),
        cpu_w: None,
        activity_w: None,
        load_source: llama_watch::collector::LoadSource::Util,
        fans: None,
    }
}
