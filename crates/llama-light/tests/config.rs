//! light.toml: defaults, bounds, every validation error, and the shipped example.

use llama_core::color::hex;
use llama_light::config::{Gradient, Layout, Style, Target, parse};
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
    assert!(!config.keyboard.enabled);
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
        Target::AuraChain {
            start: 0,
            end: 1,
            reversed: false
        }
    );
    assert_eq!(
        config.layers[1].target,
        Target::AuraChain {
            start: 1,
            end: 3,
            reversed: false
        }
    );
    assert_eq!(
        config.layers[2].target,
        Target::AuraChain {
            start: 2,
            end: 4,
            reversed: false
        }
    );
}

#[test]
fn a_descending_chain_range_lists_the_same_fans_from_the_far_led() {
    let forward = chain_target(4, "aura.chain[2..4]").expect("forward");
    let reversed = chain_target(4, "aura.chain[4..2]").expect("reversed exclusive");
    assert_eq!(
        forward,
        Target::AuraChain {
            start: 2,
            end: 4,
            reversed: false
        }
    );
    assert_eq!(
        reversed,
        Target::AuraChain {
            start: 2,
            end: 4,
            reversed: true
        }
    );
    assert_eq!(
        chain_target(5, "aura.chain[2..=4]").expect("inclusive forward"),
        Target::AuraChain {
            start: 2,
            end: 5,
            reversed: false
        }
    );
    assert_eq!(
        chain_target(5, "aura.chain[4..=2]").expect("inclusive reversed"),
        Target::AuraChain {
            start: 2,
            end: 5,
            reversed: true
        }
    );
    let one = Target::AuraChain {
        start: 2,
        end: 3,
        reversed: false,
    };
    assert_eq!(chain_target(4, "aura.chain[2]").expect("one fan"), one);
    assert_eq!(
        chain_target(4, "aura.chain[2..=2]").expect("one inclusive"),
        one
    );

    let empty = chain_target(4, "aura.chain[2..2]").expect_err("empty");
    assert!(empty.contains("the range is empty"), "{empty}");
    assert_eq!(
        chain_reason(4, "aura.chain[2..5]"),
        chain_reason(4, "aura.chain[5..2]"),
        "a reversed exclusive range names the same missing fan"
    );
    assert!(
        chain_reason(4, "aura.chain[5..2]").contains("fan 4 does not exist"),
        "{}",
        chain_reason(4, "aura.chain[5..2]")
    );
    assert_eq!(
        chain_reason(4, "aura.chain[2..=5]"),
        chain_reason(4, "aura.chain[5..=2]")
    );
    assert!(
        chain_reason(4, "aura.chain[5..=2]").contains("fan 5 does not exist"),
        "{}",
        chain_reason(4, "aura.chain[5..=2]")
    );
    assert_eq!(
        chain_reason(4, "aura.chain[2..=4]"),
        chain_reason(4, "aura.chain[4..=2]")
    );
    assert!(chain_reason(4, "aura.chain[4..=2]").contains("fan 4 does not exist"));
}

#[test]
fn an_inclusive_bound_of_usize_max_names_the_missing_fan() {
    let fan = usize::MAX;
    for target in [
        format!("aura.chain[2..={fan}]"),
        format!("aura.chain[{fan}..=2]"),
        format!("aura.chain[{fan}]"),
    ] {
        let err = chain_target(4, &target).expect_err(&target);
        assert!(
            err.contains(&format!("fan {fan} does not exist")),
            "{target}: {err}"
        );
        assert!(err.contains("the chain has 4 fans"), "{target}: {err}");
    }
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
        Target::AuraChain {
            start: 1,
            end: 2,
            reversed: false
        }
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
    assert!(err.contains("aura.chain[B..A]"), "{err}");
    assert!(err.contains("aura.chain[B..=A]"), "{err}");
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
        Target::KeyboardKeys((1..=12).collect())
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
    assert!(!config.keyboard.enabled);
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

fn chain_target(chain_len: usize, target: &str) -> Result<Target, String> {
    let text = format!(
        "[aura]\nfans = \"chain\"\nchain_len = {chain_len}\n[[light]]\ntarget = \"{target}\"\n"
    );
    parse(&text)
        .map(|config| config.layers[0].target.clone())
        .map_err(|err| err.0)
}

/// The explanation after the quoted target, so a forward range and its
/// reverse can be compared without the spelling of the target.
fn chain_reason(chain_len: usize, target: &str) -> String {
    let err = chain_target(chain_len, target).expect_err(target);
    err.split_once(": ")
        .map(|(_, reason)| reason.to_owned())
        .unwrap_or(err)
}

fn kb_target(target: &str) -> Result<Target, String> {
    let text = format!("[keyboard]\nenabled = true\n[[light]]\ntarget = '{target}'\n");
    parse(&text)
        .map(|c| c.layers[0].target.clone())
        .map_err(|e| e.0)
}

fn names(target: &Target) -> Vec<&'static str> {
    use llama_light::keyboard::keymap::KEYS;
    match target {
        Target::KeyboardKeys(keys) => keys.iter().map(|i| KEYS[*i].name).collect(),
        other => panic!("not a key list: {other:?}"),
    }
}

#[test]
fn keyboard_targets_take_ranges_lists_and_all() {
    let t = kb_target(r#"keyboard.keys["F1".."F12"]"#).expect("f-row");
    assert_eq!(
        names(&t),
        [
            "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"
        ]
    );
    let t = kb_target(r#"keyboard.keys["F1"..="F4"]"#).expect("inclusive");
    assert_eq!(names(&t), ["F1", "F2", "F3", "F4"]);
    let t = kb_target(r#"keyboard.keys["W","A","S","D"]"#).expect("wasd");
    assert_eq!(names(&t), ["W", "A", "S", "D"]);
    let t = kb_target(r#"keyboard.keys[ "1" .. "0" , "Escape" ]"#).expect("spaces");
    assert_eq!(
        names(&t),
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "Escape"]
    );
    // Names with punctuation are fine inside quotes.
    let t = kb_target(r#"keyboard.keys[",", ".", "/", "\ (ANSI)", "[", "]"]"#).expect("punct");
    assert_eq!(names(&t), [",", ".", "/", "\\ (ANSI)", "[", "]"]);
    let t = kb_target(r#"keyboard.keys["Number Pad 7"]"#).expect("numpad");
    assert_eq!(names(&t), ["Number Pad 7"]);
    assert_eq!(kb_target("keyboard.all").expect("all"), Target::KeyboardAll);
}

#[test]
fn keyboard_target_errors_are_clear() {
    let err = kb_target(r#"keyboard.keys["f1"]"#).expect_err("case");
    assert!(
        err.contains("\"f1\" is not a known key; did you mean \"F1\"?"),
        "{err}"
    );
    let err = kb_target(r#"keyboard.keys["Esc"]"#).expect_err("alias");
    assert!(err.contains("\"Esc\" is not a known key"), "{err}");
    assert!(err.contains("\"Escape\""), "{err}");
    let err = kb_target(r#"keyboard.keys[F1]"#).expect_err("unquoted");
    assert!(err.contains("key names are quoted"), "{err}");
    let err = kb_target(r#"keyboard.keys["W" "A"]"#).expect_err("no comma");
    assert!(err.contains("expected , or .."), "{err}");
    let err = kb_target(r#"keyboard.keys["W","W"]"#).expect_err("twice");
    assert!(err.contains("\"W\" is listed twice"), "{err}");
    let err = kb_target(r#"keyboard.keys["F1".."F4","F3"]"#).expect_err("overlap");
    assert!(err.contains("listed twice"), "{err}");
    let err = kb_target(r#"keyboard.keys[]"#).expect_err("empty");
    assert!(err.contains("quoted"), "{err}");
    let err = kb_target("keyboard.leds").expect_err("unknown");
    assert!(err.contains("is not a target"), "{err}");
    let err = error("[[light]]\ntarget = \"keyboard.all\"\n");
    assert!(err.contains("[keyboard] enabled = true"), "{err}");
}

#[test]
fn keyboard_brightness_cap_and_fixed_colour() {
    let config = parse(
        "[keyboard]\nenabled = true\nbrightness_max = 40\n[[light]]\ntarget = \"keyboard.all\"\ncolor = \"#102030\"\nbrightness = 20\n",
    )
    .expect("parse");
    assert_eq!(config.keyboard.brightness_max, 40);
    assert_eq!(config.layers[0].fixed, Some(hex(0x102030)));
    assert_eq!(config.layers[0].brightness, 20.0);
    // Every entry is on the keyboard: the fans keep their default entry.
    assert_eq!(config.layers.len(), 2);
    assert_eq!(config.layers[1].target, Target::AuraFans);
    let err = error("[keyboard]\nbrightness_max = 101\n");
    assert!(err.contains("keyboard.brightness_max = 101"), "{err}");
    let err = error("[[light]]\ncolor = \"#102030\"\nmetric = \"gpu\"\n");
    assert!(
        err.contains("color is a fixed colour; remove metric"),
        "{err}"
    );
    let err = error("[[light]]\ncolor = \"blue\"\n");
    assert!(err.contains("light[0].color = \"blue\""), "{err}");
    let err = error("[keyboard]\nenabled = true\nleds = 3\n");
    assert!(err.contains("unknown field"), "{err}");
}

#[test]
fn heat_parses_its_keys_and_the_defaults_fit_a_six_led_fan() {
    let config = parse(
        "[aura]\nfans = \"chain\"\nchain_len = 2\nleds_per_fan = 14\n\
         [[light]]\n\
         target = \"aura.chain[0]\"\n\
         style = \"heat\"\n\
         transit_s = 30\n\
         foot_leds = 3\n\
         conduction_s = 5000\n\
         attack_s = 0.7\n\
         smooth_s = 0.4\n\
         gradient = \"value\"\n\
         follow = \"aura.chain[1]\"\n",
    )
    .expect("heat");
    let layer = &config.layers[0];
    assert_eq!(layer.style, Style::Heat);
    assert_eq!(layer.transit_s, 30.0);
    assert_eq!(layer.foot_leds, 3);
    assert_eq!(layer.conduction_s, 5_000.0);
    assert_eq!(layer.attack_s, 0.7);
    assert_eq!(layer.smooth_s, 0.4);
    assert_eq!(layer.gradient, Gradient::Value);
    assert_eq!(
        layer.follow,
        Some(Target::AuraChain {
            start: 1,
            end: 2,
            reversed: false
        })
    );

    let defaults = parse("[[light]]\nstyle = \"heat\"\n").expect("defaults");
    let layer = &defaults.layers[0];
    assert_eq!(layer.style, Style::Heat);
    assert_eq!(layer.transit_s, 60.0);
    assert_eq!(layer.foot_leds, 5);
    assert_eq!(layer.conduction_s, 20_000.0);
    assert_eq!(layer.attack_s, 0.0);
    assert_eq!(layer.smooth_s, 0.0);
    assert_eq!(layer.gradient, Gradient::Value);
    assert_eq!(layer.follow, None);

    let edges = parse(
        "[aura]\nleds_per_fan = 14\n\
         [[light]]\nstyle = \"heat\"\ntransit_s = 3600\n\
         conduction_s = 1000000\nfoot_leds = 13\n",
    )
    .expect("upper bounds");
    assert_eq!(edges.layers[0].transit_s, 3_600.0);
    assert_eq!(edges.layers[0].conduction_s, 1_000_000.0);
    assert_eq!(edges.layers[0].foot_leds, 13);

    let two = parse("[aura]\nleds_per_fan = 2\n[[light]]\nstyle = \"heat\"\nfoot_leds = 1\n")
        .expect("two LEDs, foot of 1");
    assert_eq!(two.layers[0].foot_leds, 1);

    let floor = parse("[[light]]\nstyle = \"heat\"\nconduction_s = 1\n").expect("floor");
    assert_eq!(floor.layers[0].conduction_s, 1.0);
}

#[test]
fn heat_rejects_keys_that_do_not_apply_and_a_foot_the_span_cannot_hold() {
    let cases = [
        (
            "[[light]]\nstyle = \"heat\"\nedge = \"fractional\"\n",
            "light[0].edge is for style = \"bar\" or \"ladder\" (this entry is \"heat\")",
        ),
        (
            "[[light]]\nstyle = \"heat\"\nfill_to = 125\n",
            "light[0].fill_to is for style = \"bar\" (this entry is \"heat\")",
        ),
        (
            "[[light]]\nstyle = \"heat\"\nthresholds = [50, 100]\n",
            "light[0].thresholds is for style = \"ladder\" (this entry is \"heat\")",
        ),
        (
            "[[light]]\nstyle = \"heat\"\ngradient = \"position\"\n",
            "light[0].gradient = \"position\" does not apply to style = \"heat\"; each LED takes the colour of its own value",
        ),
        (
            "[[light]]\nstyle = \"heat\"\ngradient = \"rainbow\"\n",
            "light[0].gradient = \"rainbow\" is not \"value\" or \"position\"",
        ),
        (
            "[[light]]\nstyle = \"bar\"\ntransit_s = 30\n",
            "light[0].transit_s is for style = \"heat\" (this entry is \"bar\")",
        ),
        (
            "[[light]]\nstyle = \"bar\"\nfoot_leds = 2\n",
            "light[0].foot_leds is for style = \"heat\" (this entry is \"bar\")",
        ),
        (
            "[[light]]\nstyle = \"bar\"\nconduction_s = 1000\n",
            "light[0].conduction_s is for style = \"heat\" (this entry is \"bar\")",
        ),
        (
            "[[light]]\nstyle = \"bar\"\nfollow = \"aura.fans\"\n",
            "light[0].follow is for style = \"heat\" (this entry is \"bar\")",
        ),
        (
            "[[light]]\nstyle = \"heat\"\ntransit_s = 0\n",
            "light[0].transit_s = 0 is out of range; allowed above 0 up to 3600",
        ),
        (
            "[[light]]\nstyle = \"heat\"\ntransit_s = -1\n",
            "out of range; allowed above 0",
        ),
        (
            "[[light]]\nstyle = \"heat\"\ntransit_s = 3601\n",
            "light[0].transit_s = 3601 is out of range; allowed above 0 up to 3600",
        ),
        (
            "[[light]]\nstyle = \"heat\"\nconduction_s = 0\n",
            "light[0].conduction_s = 0 is out of range; allowed 1 up to 1000000",
        ),
        (
            "[[light]]\nstyle = \"heat\"\nconduction_s = 0.5\n",
            "light[0].conduction_s = 0.5 is out of range; allowed 1 up to 1000000",
        ),
        (
            "[[light]]\nstyle = \"heat\"\nconduction_s = 1000001\n",
            "out of range; allowed 1 up to 1000000",
        ),
        (
            "[aura]\nfans = \"chain\"\nchain_len = 2\nleds_per_fan = 8\n\
             [[light]]\ntarget = \"aura.chain[0]\"\nstyle = \"heat\"\n\
             foot_leds = 1\nfollow = \"aura.chain[0]\"\n",
            "light[0].follow = \"aura.chain[0]\" shares LEDs with this entry; heat follow must share no LED with the heat span",
        ),
        (
            "[aura]\nfans = \"chain\"\nchain_len = 3\nleds_per_fan = 8\n\
             [[light]]\ntarget = \"aura.chain[0..2]\"\nstyle = \"heat\"\n\
             follow = \"aura.chain[1]\"\n",
            "shares LEDs with this entry",
        ),
        (
            "[aura]\nfans = \"chain\"\nchain_len = 1\nleds_per_fan = 8\n\
             [[light]]\ntarget = \"aura.chain[1..0]\"\nstyle = \"heat\"\n\
             foot_leds = 1\nfollow = \"aura.chain[0]\"\n",
            "share no LED",
        ),
        (
            "[aura]\nfans = \"chain\"\nchain_len = 2\nleds_per_fan = 8\n\
             [[light]]\ntarget = \"aura.chain[1]\"\nstyle = \"heat\"\n\
             foot_leds = 1\nfollow = \"aura.fans\"\n",
            "share no LED",
        ),
        (
            "[[light]]\nstyle = \"heat\"\nfoot_leds = 0\n",
            "light[0].foot_leds = 0 must be a whole number of LEDs, at least 1 and fewer than the span",
        ),
        (
            "[[light]]\nstyle = \"heat\"\nfoot_leds = -3\n",
            "whole number of LEDs, at least 1",
        ),
        (
            "[aura]\nleds_per_fan = 6\n[[light]]\nstyle = \"heat\"\nfoot_leds = 6\n",
            "light[0].foot_leds = 6 must be at least 1 and fewer than the span's 6 LEDs",
        ),
        (
            "[aura]\nleds_per_fan = 2\n[[light]]\nstyle = \"heat\"\n",
            "light[0].foot_leds = 5 must be at least 1 and fewer than the span's 2 LEDs",
        ),
        (
            "[aura]\nleds_per_fan = 1\n[[light]]\nstyle = \"heat\"\n",
            "light[0]: style = \"heat\" needs at least 2 LEDs so the foot can be fewer than the span (this span has 1)",
        ),
        (
            "[keyboard]\nenabled = true\n[[light]]\ntarget = \"keyboard.all\"\nstyle = \"heat\"\n",
            "light[0]: style = \"heat\" is for an Aura span",
        ),
        (
            "[keyboard]\nenabled = true\n[[light]]\nstyle = \"heat\"\nfollow = \"keyboard.all\"\n",
            "light[0].follow = \"keyboard.all\" is on the keyboard; heat follow is an Aura target",
        ),
        (
            "[[light]]\nstyle = \"heat\"\nfollow = \"not-a-target\"\n",
            "light[0].follow = \"not-a-target\" is not a target",
        ),
        (
            "[aura]\nstyle = \"heat\"\n",
            "aura.style = \"heat\": the default entry is solid, ring, bar or pulse; write a [[light]] entry for style = \"heat\"",
        ),
    ];
    for (text, needle) in cases {
        let err = error(text);
        assert!(
            err.contains(needle),
            "{text}\nwanted: {needle}\ngot:    {err}"
        );
    }
}
