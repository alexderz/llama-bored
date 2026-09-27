//! S3 render half. The collect-on-read-only-roots half lives in llama-watch.
//!
//! Nothing here opens a device node or the real `/proc` or `/sys`.

use std::collections::BTreeSet;
use std::time::{Instant, SystemTime};

use kraken_lcd::config::DisplayCfg;
use kraken_lcd::history::History;
use kraken_lcd::present::present;
use kraken_lcd::render::{self, Assets};
use llama_core::sample::{AiState, Snapshot};

#[test]
fn render_half_draws_a_frame() {
    let t_mono = Instant::now();
    let snapshot = Snapshot {
        t_mono,
        t_wall: SystemTime::UNIX_EPOCH,
        load: Some(40.0),
        activity: None,
        cpu_pct: Some(10.0),
        cpu_topk_pct: Some(40.0),
        gpu_pct: Some(8.0),
        mem_pct: Some(30.0),
        coolant_c: Some(36.0),
        cpu_c: Some(42.0),
        gpu_c: Some(51.0),
        ai: AiState::Idle,
        models: Vec::new(),
        tokens: None,
        errors: BTreeSet::new(),
    };
    let mut history = History::new(snapshot.t_mono);
    history.add(snapshot.t_mono, snapshot.load);
    let view = present(
        &snapshot,
        &history,
        None,
        &kraken_lcd::config::Config::default(),
    );
    let mut assets = Assets::load().expect("assets");
    let frame = render::render(&view, &DisplayCfg::default(), &mut assets);
    assert_eq!((frame.0.width(), frame.0.height()), (320, 320));
    assert!(
        frame.0.pixels().iter().any(|pixel| pixel.red() > 0),
        "the rendered frame should contain the ring or layout"
    );
}
