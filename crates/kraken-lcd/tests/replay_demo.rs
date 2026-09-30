//! Dev tool, not a check: replays recorded snapshots through the real
//! writer loop (stream mode, fake LCD) and dumps every uploaded frame as
//! raw RGBA, for the README demo. Ignored unless run by name:
//!
//! REPLAY_IN=dir REPLAY_OUT=dir [REPLAY_FROM_S=..] [REPLAY_TO_S=..] \
//!   cargo test -p kraken-lcd --test replay_demo -- --ignored --nocapture
//!
//! `REPLAY_IN` holds `index.tsv` (wall seconds, file name) and the
//! snapshot JSON files. Frames go to `REPLAY_OUT/NNNNNN.rgba` (320x320).

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant, SystemTime};

use kraken_lcd::config::{Config, ValidConfig};
use kraken_lcd::device::{LcdSink, SinkError};
use kraken_lcd::log;
use kraken_lcd::render::Frame;
use kraken_lcd::service::{self, Clock, LoopInput, Notifier, Sampler, Stop};
use kraken_lcd::snapshot_reader::{ManualMono, SnapshotReader};
use llama_core::sample::Snapshot;

struct FakeClock {
    mono: Rc<Cell<Instant>>,
    wall: Rc<Cell<SystemTime>>,
}

impl Clock for FakeClock {
    fn mono(&self) -> Instant {
        self.mono.get()
    }
    fn wall(&self) -> SystemTime {
        self.wall.get()
    }
    fn sleep(&mut self, d: Duration) {
        self.mono.set(self.mono.get() + d);
        self.wall.set(self.wall.get() + d);
    }
}

struct Quiet;
impl Notifier for Quiet {
    fn ready(&mut self) {}
    fn watchdog(&mut self) {}
    fn stopping(&mut self) {}
}

struct NoLog;
impl log::Sink for NoLog {
    fn write_line(&mut self, _line: &str) {}
}

struct Done(Rc<Cell<bool>>);
impl Stop for Done {
    fn requested(&self) -> bool {
        self.0.get()
    }
}

/// Feeds the recorded snapshot whose wall time matches the loop's clock.
struct Replay {
    items: Vec<(f64, PathBuf)>,
    origin: Instant,
    t0: f64,
    live: PathBuf,
    sys: PathBuf,
    reader: SnapshotReader<NoLog, ManualMono>,
    at: Rc<Cell<f64>>,
    done: Rc<Cell<bool>>,
}

impl Sampler for Replay {
    fn sample(&mut self, mono: Instant, wall: SystemTime) -> Snapshot {
        let t = self.t0 + mono.saturating_duration_since(self.origin).as_secs_f64();
        self.at.set(t - self.t0);
        let i = self
            .items
            .partition_point(|(ts, _)| *ts <= t)
            .saturating_sub(1);
        if i + 1 >= self.items.len() {
            self.done.set(true);
        }
        let bytes = std::fs::read(&self.items[i].1).expect("snapshot");
        let wire = llama_core::wire::parse_validated(&bytes).expect("valid snapshot");
        if let Some(c) = wire.host.coolant_c {
            let milli = (c * 1000.0).round() as i64;
            std::fs::write(
                self.sys.join("class/hwmon/hwmon0/temp1_input"),
                format!("{milli}\n"),
            )
            .expect("coolant");
        }
        std::fs::write(&self.live, &bytes).expect("live");
        self.reader.set_now(wire.t_mono_ns + 50_000_000);
        self.reader.sample(mono, wall)
    }
}

#[derive(Clone)]
struct Dump {
    out: PathBuf,
    n: Rc<RefCell<u64>>,
    at: Rc<Cell<f64>>,
    window: (f64, f64),
}

impl Dump {
    fn save(&self, frame: &Frame) {
        let t = self.at.get();
        if t < self.window.0 || t > self.window.1 {
            return;
        }
        let mut n = self.n.borrow_mut();
        std::fs::write(self.out.join(format!("{:06}.rgba", *n)), frame.0.data()).expect("frame");
        *n += 1;
    }
}

impl LcdSink for Dump {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        self.save(frame);
        Ok(())
    }
    fn show_slot(&mut self, _slot: u8, frame: &Frame) -> Result<(), SinkError> {
        self.save(frame);
        Ok(())
    }
    fn restore_stock(&mut self) {}
    fn needs_reupload(&mut self) {}
    fn tick(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
    fn blocked(&mut self) -> bool {
        false
    }
}

fn env_f(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "dev tool: README demo replay"]
fn replay_recorded_snapshots() {
    let input = PathBuf::from(std::env::var("REPLAY_IN").expect("REPLAY_IN"));
    let out = PathBuf::from(std::env::var("REPLAY_OUT").expect("REPLAY_OUT"));
    std::fs::create_dir_all(&out).expect("out");
    let index = std::fs::read_to_string(input.join("index.tsv")).expect("index.tsv");
    let items: Vec<(f64, PathBuf)> = index
        .lines()
        .filter_map(|l| {
            let (ts, name) = l.split_once('\t')?;
            Some((ts.parse().ok()?, input.join(name)))
        })
        .collect();
    assert!(!items.is_empty(), "no snapshots");

    let scratch = out.join(".replay");
    let sys = scratch.join("sys");
    std::fs::create_dir_all(sys.join("class/hwmon/hwmon0")).expect("sys");
    std::fs::write(sys.join("class/hwmon/hwmon0/name"), "z53\n").expect("name");
    let config_path = scratch.join("config.toml");
    std::fs::write(
        &config_path,
        "[writer]\ntick_s = 0.5\n[snapshot]\nstale_after_s = 1.0\n\
         [dial]\ntiers = [[0.5, 10], [5, 2], [15, 3], [60, 4], [300, 5]]\nceiling_tps = 150.0\n\
         [upload]\nmode = \"stream\"\nstream_fps = 10\nmin_interval_s = 60\nfail_limit = 3\n",
    )
    .expect("config");
    let config: ValidConfig = Config::load_validated(&config_path).expect("valid config");

    let origin = Instant::now();
    let t0 = items[0].0;
    let at = Rc::new(Cell::new(0.0));
    let done = Rc::new(Cell::new(false));
    let live = scratch.join("snapshot.json");
    let reader = SnapshotReader::new(
        &live,
        Duration::from_secs(1),
        &sys,
        NoLog,
        ManualMono::new(0),
    );
    let sampler = Replay {
        items,
        origin,
        t0,
        live,
        sys,
        reader,
        at: at.clone(),
        done: done.clone(),
    };
    let wall0 = SystemTime::UNIX_EPOCH + Duration::from_secs_f64(t0);
    let mut clock = FakeClock {
        mono: Rc::new(Cell::new(origin)),
        wall: Rc::new(Cell::new(wall0)),
    };
    let dump = Dump {
        out: out.clone(),
        n: Rc::new(RefCell::new(0)),
        at,
        window: (env_f("REPLAY_FROM_S", 0.0), env_f("REPLAY_TO_S", f64::MAX)),
    };
    let count = dump.n.clone();
    let mut assets = kraken_lcd::render::Assets::load().expect("assets");
    service::run_loop(LoopInput {
        config: &config,
        sampler,
        clock: &mut clock,
        notify: &mut Quiet,
        stop: Done(done),
        log: NoLog,
        open: move || Ok(dump.clone()),
        assets: &mut assets,
        latch_at_start: false,
    });
    println!("frames: {}", count.borrow());
}
