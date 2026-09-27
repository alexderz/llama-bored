//! light.toml: defaults, bounds, every validation error, and the shipped example.

use llama_core::color::hex;
use llama_light::config::{Layout, Style, Target, parse};
use llama_light::metric::Metric;
use llama_light::palette::{Palette, Scale};

fn error(text: &str) -> String {
    match parse(text) {
        Ok(config) => panic!("accepted:\n{text}\n{config:?}"),
        Err(err) => err.0,
    }
}

#[test]
fn an_empty_file_is_mirrored_fans_showing_activity_on_act() {
    let config = parse("").expect("empty");
    assert!(config.aura.enabled);
    assert_eq!(config.aura.leds_per_fan, 6);
    assert_eq!(config.aura.layout, Layout::Mirrored);
    assert_eq!(config.aura.brightness_max, 80);
    assert_eq!(config.aura.fps, 10);
    assert!(!config.keyboard_enabled);
    assert_eq!(config.layers.len(), 1);
    let layer = &config.layers[0];
    assert_eq!(layer.target, Target::AuraFans);
    assert_eq!(layer.metric, Metric::Activity);
    assert_eq!(layer.style, Style::Solid);
    assert_eq!(layer.palette, Palette::Act);
    assert_eq!(layer.range, (0.0, 100.0));
}

#[test]
fn aura_metric_and_style_set_the_default_layer() {
    let config = parse("[aura]\nmetric = \"gpu\"\nstyle = \"ring\"\n").expect("parse");
    assert_eq!(config.layers[0].metric, Metric::Gpu);
    assert_eq!(config.layers[0].style, Style::Gauge);
}

#[test]
fn bounds_are_enforced_with_the_key_and_the_allowed_range() {
    for (text, key) in [
        (
            "[aura]\nbrightness_max = 101\n",
            "aura.brightness_max = 101",
        ),
        ("[aura]\nbrightness_max = -1\n", "aura.brightness_max = -1"),
        ("[aura]\nfps = 0\n", "aura.fps = 0"),
        ("[aura]\nfps = 21\n", "aura.fps = 21"),
        ("[aura]\nleds_per_fan = 0\n", "aura.leds_per_fan = 0"),
        ("[aura]\nleds_per_fan = 21\n", "aura.leds_per_fan = 21"),
        (
            "[aura]\nfans = \"chain\"\nchain_len = 9\n",
            "aura.chain_len = 9",
        ),
        ("[[light]]\nbrightness = 101\n", "light[0].brightness = 101"),
        ("[[light]]\nsmooth_s = 61.0\n", "light[0].smooth_s = 61"),
    ] {
        let err = error(text);
        assert!(err.contains(key), "{err}");
        assert!(err.contains("out of range"), "{err}");
    }
    for ok in [
        "[aura]\nbrightness_max = 0\nfps = 1\nleds_per_fan = 1\n",
        "[aura]\nbrightness_max = 100\nfps = 20\nleds_per_fan = 20\n",
    ] {
        parse(ok).expect(ok);
    }
}

#[test]
fn unknown_keys_and_bad_types_are_named() {
    let err = error("[aura]\nbrightnes_max = 50\n");
    assert!(err.contains("brightnes_max"), "{err}");
    let err = error("[aura]\nfps = \"fast\"\n");
    assert!(err.contains("fps") || err.contains("integer"), "{err}");
    let err = error("[[light]]\ncolour = \"red\"\n");
    assert!(err.contains("colour"), "{err}");
}

#[test]
fn chain_layout_and_targets() {
    let config = parse(
        "[aura]\nfans = \"chain\"\nchain_len = 4\n\n\
         [[light]]\ntarget = \"aura.chain[0]\"\nmetric = \"gpu\"\n\n\
         [[light]]\ntarget = \"aura.chain[1..3]\"\nmetric = \"cpu\"\n\n\
         [[light]]\ntarget = \"aura.chain[2..=3]\"\nmetric = \"load\"\n",
    )
    .expect("chain");
    assert_eq!(config.aura.layout, Layout::Chain(4));
    assert_eq!(config.aura.frame_len(), 24);
    assert_eq!(
        config.layers[0].target,
        Target::AuraChain { start: 0, end: 1 }
    );
    assert_eq!(
        config.layers[1].target,
        Target::AuraChain { start: 1, end: 3 }
    );
    assert_eq!(
        config.layers[2].target,
        Target::AuraChain { start: 2, end: 4 }
    );
}

#[test]
fn the_chain_shorthand_is_one_entry_per_fan() {
    let config = parse(
        "[aura]\nchain = [ { metric = \"gpu\" }, { metric = \"cpu\", style = \"ring\" }, { metric = \"tokens_rate\" } ]\n",
    )
    .expect("shorthand");
    assert_eq!(config.aura.layout, Layout::Chain(3));
    assert_eq!(config.layers.len(), 3);
    assert_eq!(
        config.layers[1].target,
        Target::AuraChain { start: 1, end: 2 }
    );
    assert_eq!(config.layers[1].style, Style::Gauge);
    assert_eq!(config.layers[2].metric, Metric::TokensRate);
}

#[test]
fn targets_must_exist() {
    let err = error("[[light]]\ntarget = \"aura.chain[0]\"\n");
    assert!(err.contains("needs aura.fans = \"chain\""), "{err}");
    let err =
        error("[aura]\nfans = \"chain\"\nchain_len = 4\n[[light]]\ntarget = \"aura.chain[4]\"\n");
    assert!(err.contains("fan 4 does not exist"), "{err}");
    let err = error(
        "[aura]\nfans = \"chain\"\nchain_len = 4\n[[light]]\ntarget = \"aura.chain[2..2]\"\n",
    );
    assert!(err.contains("empty"), "{err}");
    let err = error("[[light]]\ntarget = \"aura.ring\"\n");
    assert!(err.contains("is not a target"), "{err}");
    let err = error("[[light]]\ntarget = 'keyboard.keys[\"F1\"..\"F12\"]'\n");
    assert!(err.contains("[keyboard] enabled = true"), "{err}");
    let err = error("[keyboard]\nenabled = true\n[[light]]\ntarget = 'keyboard.keys[\"F13\"]'\n");
    assert!(err.contains("\"F13\" is not a known key"), "{err}");
    let err =
        error("[keyboard]\nenabled = true\n[[light]]\ntarget = 'keyboard.keys[\"F12\"..\"F1\"]'\n");
    assert!(err.contains("first key comes after the last"), "{err}");
    let config = parse(
        "[keyboard]\nenabled = true\n[[light]]\ntarget = 'keyboard.keys[\"F1\"..\"F12\"]'\nstyle = \"bar\"\n",
    )
    .expect("keyboard gauge");
    assert_eq!(
        config.layers[0].target,
        Target::KeyboardKeys { start: 1, end: 12 }
    );
}

#[test]
fn layout_conflicts_are_explained() {
    let err = error("[aura]\nchain_len = 3\n");
    assert!(err.contains("aura.fans is not \"chain\""), "{err}");
    let err = error("[aura]\nfans = \"chain\"\n");
    assert!(err.contains("needs aura.chain_len"), "{err}");
    let err = error("[aura]\nfans = \"mirrored\"\nchain = [ { metric = \"gpu\" } ]\n");
    assert!(err.contains("splitter"), "{err}");
    let err = error("[aura]\nchain_len = 3\nchain = [ { metric = \"gpu\" } ]\n");
    assert!(
        err.contains("chain_len = 3 but aura.chain lists 1"),
        "{err}"
    );
    let err = error("[aura]\nfans = \"daisy\"\n");
    assert!(err.contains("\"daisy\""), "{err}");
    let err = error("[aura]\nchain = [ { target = \"aura.fans\" } ]\n");
    assert!(err.contains("remove target"), "{err}");
}

#[test]
fn metrics_styles_scales_and_palettes_are_closed_sets() {
    let err = error("[[light]]\nmetric = \"fan_rpm\"\n");
    assert!(err.contains("\"fan_rpm\" is not a metric"), "{err}");
    assert!(err.contains("tokens_rate"), "{err}");
    let err = error("[[light]]\nmetric = \"ctx\"\n");
    assert!(err.contains("not in snapshot v1"), "{err}");
    let err = error("[[light]]\nstyle = \"rainbow\"\n");
    assert!(err.contains("\"rainbow\" is not a style"), "{err}");
    let err = error("[[light]]\nscale = \"exp\"\n");
    assert!(err.contains("\"exp\""), "{err}");
    let err = error("[[light]]\npalette = \"neon\"\n");
    assert!(err.contains("\"neon\" is not a palette"), "{err}");
    let err =
        error("[[light]]\npalette = \"act\"\nstops = [[0, \"#000000\"], [100, \"#FFFFFF\"]]\n");
    assert!(err.contains("palette or stops, not both"), "{err}");
    for name in [
        "activity",
        "gpu",
        "cpu",
        "load",
        "tokens_rate",
        "coolant",
        "gpu_temp",
        "cpu_temp",
        "mem",
    ] {
        parse(&format!("[[light]]\nmetric = \"{name}\"\n")).expect(name);
    }
    for style in ["solid", "ring", "bar", "pulse"] {
        parse(&format!("[[light]]\nstyle = \"{style}\"\n")).expect(style);
    }
    for palette in ["act", "thermal", "mono"] {
        parse(&format!("[[light]]\npalette = \"{palette}\"\n")).expect(palette);
    }
}

#[test]
fn ranges_must_be_finite_with_min_below_max() {
    let err = error("[[light]]\nrange = [50, 20]\n");
    assert!(err.contains("min 50 must be below max 20"), "{err}");
    let err = error("[[light]]\nrange = [5, 5]\n");
    assert!(err.contains("must be below"), "{err}");
    let err = error("[[light]]\nrange = [0, nan]\n");
    assert!(err.contains("not a finite number"), "{err}");
    let err = error("[[light]]\nrange = [0, inf]\n");
    assert!(err.contains("not a finite number"), "{err}");
    let err = error("[[light]]\nrange = [0]\n");
    assert!(err.contains("[min, max]"), "{err}");
    let err = error("[[light]]\nrange = [0, 100]\nscale = \"log\"\n");
    assert!(err.contains("log") && err.contains("above 0"), "{err}");
    let config = parse("[[light]]\nrange = [1, 200]\nscale = \"log\"\n").expect("log");
    assert_eq!(config.layers[0].scale, Scale::Log);
}

#[test]
fn stops_are_parsed_in_metric_units_or_percent_and_must_ascend() {
    let config = parse(
        "[[light]]\nmetric = \"coolant\"\nrange = [20, 40]\n\
         stops = [[20, \"#4A55C8\"], [\"50%\", \"#D044A8\"], [40, \"#FF3A22\"], [45, \"#FFFFFF\"]]\n",
    )
    .expect("stops");
    let Palette::Stops(stops) = &config.layers[0].palette else {
        panic!("not stops");
    };
    assert_eq!(
        stops,
        &vec![
            (0.0, hex(0x4A55C8)),
            (50.0, hex(0xD044A8)),
            (100.0, hex(0xFF3A22)),
            (125.0, hex(0xFFFFFF)),
        ]
    );

    let err = error("[[light]]\nstops = [[60, \"#000000\"], [40, \"#FFFFFF\"]]\n");
    assert!(err.contains("stops must ascend"), "{err}");
    assert!(err.contains("light[0].stops[1]"), "{err}");
    let err = error("[[light]]\nstops = [[0, \"#000000\"], [0, \"#FFFFFF\"]]\n");
    assert!(err.contains("stops must ascend"), "{err}");
    let err = error("[[light]]\nstops = [[0, \"#000000\"]]\n");
    assert!(err.contains("at least 2 stops"), "{err}");
    let err = error("[[light]]\nstops = [[0, \"red\"], [100, \"#FFFFFF\"]]\n");
    assert!(err.contains("\"red\" is not a #RRGGBB colour"), "{err}");
    let err = error("[[light]]\nstops = [[0, \"#000000\"], [\"half\", \"#FFFFFF\"]]\n");
    assert!(err.contains("\"N%\""), "{err}");
    let err = error("[[light]]\nstops = [[0, \"#000000\"], [200, \"#FFFFFF\"]]\n");
    assert!(err.contains("between 0% and 125%"), "{err}");
    let err = error("[[light]]\nidle_color = \"#12345\"\n");
    assert!(err.contains("light[0].idle_color"), "{err}");
}

#[test]
fn the_shipped_example_is_valid() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../packaging/light.example.toml"
    ))
    .expect("example");
    let config = parse(&text).expect("light.example.toml");
    assert_eq!(config.aura.layout, Layout::Mirrored);
    assert_eq!(config.aura.leds_per_fan, 6);
    assert!(!config.keyboard_enabled);
}

/// Every commented example block in light.example.toml parses once uncommented.
#[test]
fn every_commented_example_in_the_shipped_file_is_valid() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../packaging/light.example.toml"
    ))
    .expect("example");
    let mut blocks = Vec::new();
    let mut current: Option<(String, String)> = None;
    for line in text.lines() {
        if let Some(name) = line.strip_prefix("# --- example: ") {
            if let Some(done) = current.take() {
                blocks.push(done);
            }
            current = Some((name.trim().to_owned(), String::new()));
        } else if line.starts_with("# --- end") {
            if let Some(done) = current.take() {
                blocks.push(done);
            }
        } else if let Some((_, body)) = current.as_mut() {
            let uncommented = line
                .strip_prefix("# ")
                .or_else(|| line.strip_prefix('#'))
                .unwrap_or(line);
            body.push_str(uncommented);
            body.push('\n');
        }
    }
    assert!(
        blocks.len() >= 3,
        "expected at least three example blocks, got {}",
        blocks.len()
    );
    for (name, body) in blocks {
        if let Err(err) = parse(&body) {
            panic!("example {name} does not parse: {err}\n{body}");
        }
    }
}
