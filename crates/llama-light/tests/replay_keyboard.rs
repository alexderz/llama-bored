//! Dev tool, not a check: replays recorded snapshots through the real
//! llama-light renderer and prints each keyboard frame, for the README demo.
//! Ignored unless run by name:
//!
//! REPLAY_IN=dir REPLAY_CONFIG=light.toml REPLAY_OUT=frames.tsv \
//!   [REPLAY_FROM_S=..] [REPLAY_TO_S=..] \
//!   cargo test -p llama-light --test replay_keyboard -- --ignored --nocapture
//!
//! `REPLAY_IN` holds `index.tsv` (wall seconds, file name) and the snapshot
//! JSON files. Each output line is the time in seconds, then one
//! `name=#rrggbb` per key.

use std::fmt::Write as _;
use std::path::PathBuf;

use llama_light::config::parse;
use llama_light::keyboard::keymap::KEYS;
use llama_light::mapping::Renderer;

fn env_f(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "dev tool: README demo replay"]
fn replay_keyboard_frames() {
    let input = PathBuf::from(std::env::var("REPLAY_IN").expect("REPLAY_IN"));
    let config = std::fs::read_to_string(std::env::var("REPLAY_CONFIG").expect("REPLAY_CONFIG"))
        .expect("config");
    let out = std::env::var("REPLAY_OUT").expect("REPLAY_OUT");
    let (from, to) = (env_f("REPLAY_FROM_S", 0.0), env_f("REPLAY_TO_S", f64::MAX));
    let index = std::fs::read_to_string(input.join("index.tsv")).expect("index.tsv");
    let items: Vec<(f64, PathBuf)> = index
        .lines()
        .filter_map(|l| {
            let (ts, name) = l.split_once('\t')?;
            Some((ts.parse().ok()?, input.join(name)))
        })
        .collect();
    let t0 = items.first().expect("snapshots").0;
    let mut renderer = Renderer::new(parse(&config).expect("valid config"));
    let mut text = String::new();
    let mut last = t0;
    for (ts, path) in &items {
        let bytes = std::fs::read(path).expect("snapshot");
        let snap = llama_core::wire::parse_validated(&bytes).expect("valid snapshot");
        let frame = renderer.frames(&snap, (ts - last) as f32).keyboard;
        last = *ts;
        let t = ts - t0;
        if t < from || t > to {
            continue;
        }
        let _ = write!(text, "{t:.2}");
        for (key, c) in KEYS.iter().zip(&frame) {
            let _ = write!(text, "\t{}=#{:02x}{:02x}{:02x}", key.name, c.r, c.g, c.b);
        }
        text.push('\n');
    }
    std::fs::write(out, text).expect("out");
}
