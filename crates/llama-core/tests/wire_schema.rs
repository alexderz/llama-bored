//! Schema limits for snapshot v1.

use llama_core::wire::{
    self, Ai, AiWire, Host, ModelState, ModelWire, Tokens, WireError, WireSnapshot,
    parse_validated, to_json, validate,
};

fn host_ok() -> Host {
    Host {
        load_pct: Some(50.0),
        activity_pct: Some(33.0),
        cpu_pct: Some(10.0),
        cpu_topk_pct: Some(40.0),
        gpu_pct: Some(50.0),
        mem_pct: Some(11.0),
        coolant_c: Some(34.0),
        cpu_c: Some(70.0),
        gpu_c: Some(60.0),
    }
}

fn valid() -> WireSnapshot {
    WireSnapshot {
        schema: wire::SCHEMA,
        run_id: 17_461_234_567_890_123,
        seq: 48_213,
        t_mono_ns: 912_345_678_901_234,
        t_wall_ms: 1_790_200_000_123,
        host: host_ok(),
        ai: Ai {
            state: AiWire::Loaded,
            models: vec![
                ModelWire {
                    backend: None,
                    running: None,
                    queued: None,
                    kv_fill: None,
                    name: "Qwen 35B".to_owned(),
                    state: ModelState::Ready,
                    full_name: None,
                    detail: None,
                },
                ModelWire {
                    backend: None,
                    running: None,
                    queued: None,
                    kv_fill: None,
                    name: "DeepSeek".to_owned(),
                    state: ModelState::Starting,
                    full_name: None,
                    detail: None,
                },
            ],
        },
        tokens: Tokens {
            decoded_total: Some(1_234_567),
        },
    }
}

fn base_json() -> String {
    r#"{"schema":1,"run_id":1,"seq":1,"t_mono_ns":1,"t_wall_ms":1,"host":{"load_pct":1.0,"cpu_pct":1.0,"cpu_topk_pct":1.0,"gpu_pct":1.0,"mem_pct":1.0,"coolant_c":1.0,"cpu_c":1.0,"gpu_c":1.0},"ai":{"state":"loaded","models":[{"name":"Qwen 35B","state":"ready"}]},"tokens":{"decoded_total":1}}"#
        .to_owned()
}

fn insert_after(json: &str, needle: &str, extra: &str) -> Vec<u8> {
    let pos = json.find(needle).expect(needle);
    let at = pos + needle.len();
    let mut out = String::with_capacity(json.len() + extra.len());
    out.push_str(&json[..at]);
    out.push_str(extra);
    out.push_str(&json[at..]);
    out.into_bytes()
}

#[test]
fn sanitize_wire_names_pass_validate() {
    let raws = [
        "Qwen 35B",
        "abcdefghijklmnopqrstuvwxyz",
        "Qwen  35B",
        "hello   world",
        "Qwen 三五B",
        "héllo…",
        "a…b",
        "1234567890123",
        "  spaced  ",
        "",
        "🔥",
    ];
    for raw in raws {
        let name = llama_core::names::sanitize_wire(raw);
        if name.is_empty() {
            continue;
        }
        let mut snap = valid();
        snap.ai.models = vec![ModelWire {
            backend: None,
            running: None,
            queued: None,
            kv_fill: None,
            name: name.clone(),
            state: ModelState::Ready,
            full_name: None,
            detail: None,
        }];
        assert_eq!(validate(&snap), Ok(()), "raw={raw:?} name={name:?}");
        assert!(
            name.chars().count() <= wire::CANONICAL_NAME_CHARS,
            "raw={raw:?} name={name:?}"
        );
    }
}

#[test]
fn constants_match_the_v1_contract() {
    assert_eq!(wire::SCHEMA, 1);
    assert_eq!(wire::MAX_BYTES, 16 * 1024);
    assert_eq!(wire::MAX_MODELS, 8);
    assert_eq!(wire::MAX_NAME_CHARS, 13);
    assert_eq!(wire::CANONICAL_NAME_CHARS, 12);
    assert_eq!(wire::SNAPSHOT_DIR, "/run/llama-watch");
    assert_eq!(wire::SNAPSHOT_PATH, "/run/llama-watch/snapshot.json");
    assert!(wire::SNAPSHOT_PATH.starts_with(wire::SNAPSHOT_DIR));
}

#[test]
fn valid_snapshot_round_trips() {
    let snap = valid();
    let bytes = to_json(&snap).expect("encode");
    assert!(!bytes.is_empty());
    assert!(bytes.len() <= wire::MAX_BYTES);
    let text = String::from_utf8(bytes.clone()).expect("json is utf-8");
    assert!(text.contains("\"loaded\""), "{text}");
    assert!(text.contains("\"ready\""), "{text}");
    assert!(text.contains("\"starting\""), "{text}");
    assert!(!text.contains("\"Loaded\""), "{text}");
    let back = parse_validated(&bytes).expect("validated parse");
    assert_eq!(back, snap);

    let mut empty = valid();
    empty.ai = Ai {
        state: AiWire::Down,
        models: Vec::new(),
    };
    empty.host = Host {
        load_pct: None,
        activity_pct: None,
        cpu_pct: None,
        cpu_topk_pct: None,
        gpu_pct: None,
        mem_pct: None,
        coolant_c: None,
        cpu_c: None,
        gpu_c: None,
    };
    empty.tokens.decoded_total = None;
    let bytes = to_json(&empty).expect("encode");
    assert_eq!(parse_validated(&bytes).expect("validated parse"), empty);
    assert_eq!(validate(&empty), Ok(()));

    empty.ai.state = AiWire::Idle;
    assert_eq!(validate(&empty), Ok(()));
    empty.ai.state = AiWire::Loaded;
    empty.ai.models.clear();
    assert_eq!(validate(&empty), Ok(()));
    empty.tokens.decoded_total = Some(u64::MAX);
    assert_eq!(validate(&empty), Ok(()));
}

#[test]
fn unknown_field_at_each_level_is_rejected() {
    let base = base_json();
    let parsed = parse_validated(base.as_bytes()).expect("base json");
    assert_eq!(parsed.schema, wire::SCHEMA);
    let cases = [
        ("root", insert_after(&base, "{", r#""nope":1,"#)),
        ("host", insert_after(&base, r#""host":{"#, r#""nope":1,"#)),
        ("ai", insert_after(&base, r#""ai":{"#, r#""nope":1,"#)),
        (
            "model",
            insert_after(&base, r#""models":[{"#, r#""nope":1,"#),
        ),
        (
            "tokens",
            insert_after(&base, r#""tokens":{"#, r#""nope":1,"#),
        ),
    ];
    for (level, bytes) in cases {
        assert!(
            matches!(parse_validated(&bytes), Err(WireError::Parse)),
            "{level}"
        );
    }
    let upper = base.replace("\"loaded\"", "\"Loaded\"");
    assert!(matches!(
        parse_validated(upper.as_bytes()),
        Err(WireError::Parse)
    ));
    let nodata = base.replace("\"loaded\"", "\"nodata\"");
    assert!(matches!(
        parse_validated(nodata.as_bytes()),
        Err(WireError::Parse)
    ));
}

#[test]
fn length_is_checked_before_the_parse() {
    let mut padded = to_json(&valid()).expect("encode");
    padded.resize(wire::MAX_BYTES + 1, b' ');
    assert_eq!(
        parse_validated(&padded),
        Err(WireError::TooLong {
            len: wire::MAX_BYTES + 1
        })
    );
    let spaces = vec![b' '; wire::MAX_BYTES];
    assert_eq!(parse_validated(&spaces), Err(WireError::Parse));
}

#[test]
fn activity_pct_is_additive_on_schema_v1() {
    let base = base_json();
    let old = parse_validated(base.as_bytes()).expect("snapshot without activity_pct");
    assert_eq!(old.schema, 1);
    assert_eq!(old.host.load_pct, Some(1.0));
    assert_eq!(old.host.activity_pct, None);

    let bytes = insert_after(&base, r#""host":{"#, r#""activity_pct":42.0,"#);
    let new = parse_validated(&bytes).expect("snapshot with activity_pct");
    assert_eq!(new.schema, 1);
    assert_eq!(new.host.load_pct, Some(1.0));
    assert_eq!(new.host.activity_pct, Some(42.0));

    let mut snap = valid();
    snap.host.activity_pct = None;
    let text = String::from_utf8(to_json(&snap).expect("encode")).expect("utf8");
    assert!(
        !text.contains("activity_pct"),
        "absent activity is omitted so an older file shape stays valid: {text}"
    );
}

#[test]
fn schema_must_be_one() {
    let mut snap = valid();
    snap.schema = 2;
    assert_eq!(validate(&snap), Err(WireError::Schema));
    let bytes = to_json(&snap).expect("encode");
    assert_eq!(parse_validated(&bytes), Err(WireError::Schema));
    snap.schema = 0;
    let bytes = to_json(&snap).expect("encode");
    assert_eq!(parse_validated(&bytes), Err(WireError::Schema));
}

#[test]
fn non_finite_host_numbers_serialise_as_null() {
    let mut snap = valid();
    snap.host.load_pct = Some(f32::NAN);
    snap.host.cpu_c = Some(f32::INFINITY);
    snap.host.gpu_c = Some(f32::NEG_INFINITY);
    let bytes = to_json(&snap).expect("encode does not use an empty buffer as an error");
    assert!(!bytes.is_empty());
    let text = String::from_utf8(bytes.clone()).expect("json");
    assert!(text.contains("\"load_pct\":null"), "{text}");
    assert!(text.contains("\"cpu_c\":null"), "{text}");
    assert!(text.contains("\"gpu_c\":null"), "{text}");
    let back = parse_validated(&bytes).expect("null is in range");
    assert_eq!(back.host.load_pct, None);
    assert_eq!(back.host.cpu_c, None);
    assert_eq!(back.host.gpu_c, None);
    assert_eq!(back.host.cpu_pct, snap.host.cpu_pct);
}

#[test]
fn percent_and_temperature_bounds() {
    type SetHost = fn(&mut Host, Option<f32>);
    let pcts: &[(&str, SetHost)] = &[
        ("load_pct", |host, value| host.load_pct = value),
        ("cpu_pct", |host, value| host.cpu_pct = value),
        ("cpu_topk_pct", |host, value| host.cpu_topk_pct = value),
        ("gpu_pct", |host, value| host.gpu_pct = value),
        ("mem_pct", |host, value| host.mem_pct = value),
    ];
    for (field, set) in pcts {
        for value in [Some(0.0), Some(100.0), None] {
            let mut snap = valid();
            set(&mut snap.host, value);
            assert_eq!(validate(&snap), Ok(()), "{field} {value:?}");
        }
        for value in [Some(-1.0), Some(101.0), Some(f32::NAN), Some(f32::INFINITY)] {
            let mut snap = valid();
            set(&mut snap.host, value);
            assert_eq!(
                validate(&snap),
                Err(WireError::OutOfRange { field }),
                "{field} {value:?}"
            );
        }
    }

    let temps: &[(&str, SetHost)] = &[
        ("coolant_c", |host, value| host.coolant_c = value),
        ("cpu_c", |host, value| host.cpu_c = value),
        ("gpu_c", |host, value| host.gpu_c = value),
    ];
    for (field, set) in temps {
        for value in [Some(-20.0), Some(150.0), None] {
            let mut snap = valid();
            set(&mut snap.host, value);
            assert_eq!(validate(&snap), Ok(()), "{field} {value:?}");
        }
        for value in [Some(-21.0), Some(151.0), Some(f32::NEG_INFINITY)] {
            let mut snap = valid();
            set(&mut snap.host, value);
            assert_eq!(
                validate(&snap),
                Err(WireError::OutOfRange { field }),
                "{field} {value:?}"
            );
        }
    }
}

/// T54: activity is nominal-relative, so a spike reads above 100. The wire
/// accepts 0..=125 for `activity_pct` and still 0..=100 for every other percent.
#[test]
fn activity_pct_accepts_the_redline_up_to_125() {
    for value in [Some(0.0), Some(100.0), Some(112.5), Some(125.0), None] {
        let mut snap = valid();
        snap.host.activity_pct = value;
        assert_eq!(validate(&snap), Ok(()), "activity_pct {value:?}");
        let bytes = serde_json::to_vec(&snap).expect("encode");
        let back = parse_validated(&bytes).expect("a redline activity survives the wire");
        assert_eq!(back.host.activity_pct, value);
    }
    for value in [Some(-1.0), Some(125.5), Some(f32::NAN), Some(f32::INFINITY)] {
        let mut snap = valid();
        snap.host.activity_pct = value;
        assert_eq!(
            validate(&snap),
            Err(WireError::OutOfRange {
                field: "activity_pct"
            }),
            "activity_pct {value:?}"
        );
    }
    let mut load = valid();
    load.host.load_pct = Some(112.0);
    assert_eq!(
        validate(&load),
        Err(WireError::OutOfRange { field: "load_pct" }),
        "only activity has the redline"
    );
}

#[test]
fn model_count_and_loaded_state() {
    let mut snap = valid();
    snap.ai.models = (0..8)
        .map(|index| ModelWire {
            backend: None,
            running: None,
            queued: None,
            kv_fill: None,
            name: format!("m{index}"),
            state: ModelState::Other,
            full_name: None,
            detail: None,
        })
        .collect();
    assert_eq!(validate(&snap), Ok(()));

    snap.ai.models.push(ModelWire {
        backend: None,
        running: None,
        queued: None,
        kv_fill: None,
        name: "m8".to_owned(),
        state: ModelState::Stopping,
        full_name: None,
        detail: None,
    });
    assert_eq!(validate(&snap), Err(WireError::TooManyModels));

    snap.ai.models.truncate(1);
    snap.ai.state = AiWire::Idle;
    assert_eq!(validate(&snap), Err(WireError::ModelsNotLoaded));
    snap.ai.state = AiWire::Down;
    assert_eq!(validate(&snap), Err(WireError::ModelsNotLoaded));
}

#[test]
fn name_length_and_canonical_form() {
    let mut snap = valid();
    snap.ai.models = vec![ModelWire {
        backend: None,
        running: None,
        queued: None,
        kv_fill: None,
        name: "123456789012".to_owned(),
        state: ModelState::Ready,
        full_name: None,
        detail: None,
    }];
    assert_eq!(validate(&snap), Ok(()));

    snap.ai.models[0].name = "12345678901…".to_owned();
    assert_eq!(snap.ai.models[0].name.chars().count(), 12);
    assert_eq!(validate(&snap), Ok(()));

    snap.ai.models[0].name = "1234567890123".to_owned();
    assert_eq!(snap.ai.models[0].name.chars().count(), 13);
    assert_eq!(validate(&snap), Err(WireError::Name));

    snap.ai.models[0].name = "12345678901234".to_owned();
    assert_eq!(snap.ai.models[0].name.chars().count(), 14);
    assert_eq!(validate(&snap), Err(WireError::Name));

    snap.ai.models[0].name = "Qwen  35B".to_owned();
    assert_eq!(validate(&snap), Err(WireError::Name));

    snap.ai.models[0].name = String::new();
    assert_eq!(validate(&snap), Err(WireError::Name));
}

fn bonsai_detail() -> wire::ModelDetail {
    wire::ModelDetail {
        ctx: Some(262_144),
        ncmoe: None,
        kv_k: Some("q8_0".to_owned()),
        kv_v: Some("q8_0".to_owned()),
        quant: Some("PTQ1_0".to_owned()),
        fa: Some(true),
    }
}

/// The v1 model entry before T51, frozen here: what an older writer parses.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct PreT51Model {
    name: String,
    state: ModelState,
}

#[test]
fn model_detail_is_additive_on_schema_v1() {
    // New writer, old snapshot: no full_name, no detail.
    let base = base_json();
    let old = parse_validated(base.as_bytes()).expect("snapshot without model detail");
    assert_eq!(old.ai.models[0].name, "Qwen 35B");
    assert_eq!(old.ai.models[0].full_name, None);
    assert_eq!(old.ai.models[0].detail, None);

    // New writer, new snapshot.
    let bytes = insert_after(
        &base,
        r#""state":"ready""#,
        r#","full_name":"Ternary Bonsai 2 27B","detail":{"ctx":262144,"kv_k":"q8_0","kv_v":"q8_0","quant":"PTQ1_0","fa":true}"#,
    );
    let new = parse_validated(&bytes).expect("snapshot with model detail");
    assert_eq!(
        new.ai.models[0].full_name.as_deref(),
        Some("Ternary Bonsai 2 27B")
    );
    assert_eq!(new.ai.models[0].detail, Some(bonsai_detail()));

    // A new watcher with nothing to add writes the old shape byte for byte,
    // which the pre-T51 model entry still parses.
    let mut snap = valid();
    for model in &mut snap.ai.models {
        model.full_name = None;
        model.detail = None;
    }
    let text = String::from_utf8(to_json(&snap).expect("encode")).expect("utf8");
    assert!(!text.contains("full_name"), "{text}");
    assert!(!text.contains("detail"), "{text}");
    let value: serde_json::Value = serde_json::from_str(&text).expect("json");
    for model in value["ai"]["models"].as_array().expect("models") {
        serde_json::from_value::<PreT51Model>(model.clone()).expect("old writer shape");
    }

    // Round trip with every field set.
    let mut full = valid();
    full.ai.models[0].full_name = Some("Ternary Bonsai 2 27B".to_owned());
    full.ai.models[0].detail = Some(wire::ModelDetail {
        ncmoe: Some(16),
        ..bonsai_detail()
    });
    let bytes = to_json(&full).expect("encode");
    assert_eq!(parse_validated(&bytes).expect("round trip"), full);
}

#[test]
fn model_detail_is_allowlisted() {
    let bad_tokens = [
        "",
        "/models/llm/x.gguf",
        "q8 0",
        "q8_0;rm",
        "é",
        "AAAAAAAAAAAAAAAAA",
    ];
    for token in bad_tokens {
        for field in 0..3 {
            let mut detail = bonsai_detail();
            let slot = match field {
                0 => &mut detail.kv_k,
                1 => &mut detail.kv_v,
                _ => &mut detail.quant,
            };
            *slot = Some(token.to_owned());
            let mut snap = valid();
            snap.ai.models[0].detail = Some(detail);
            assert_eq!(validate(&snap), Err(WireError::Detail), "{field} {token:?}");
        }
    }
    let long = "x".repeat(wire::MAX_FULL_NAME_CHARS + 1);
    for name in ["", " lead", "two  spaces", "tab\tname", long.as_str()] {
        let mut snap = valid();
        snap.ai.models[0].full_name = Some(name.to_owned());
        assert_eq!(validate(&snap), Err(WireError::Detail), "{name:?}");
    }
    let unknown = insert_after(
        &base_json(),
        r#""state":"ready""#,
        r#","detail":{"cmd":"/models/prism/llama-server"}"#,
    );
    assert!(matches!(parse_validated(&unknown), Err(WireError::Parse)));
}

#[test]
fn backend_gauges_are_additive_on_schema_v1() {
    // An older watcher's model has no backend fields and still validates.
    let old = parse_validated(base_json().as_bytes()).expect("pre-T72 snapshot");
    assert_eq!(old.ai.models[0].backend, None);
    assert_eq!(old.ai.models[0].running, None);
    assert_eq!(old.ai.models[0].kv_fill, None);

    let mut snap = valid();
    snap.ai.models[0].backend = Some(wire::Backend::SgLang);
    snap.ai.models[0].running = Some(1);
    snap.ai.models[0].queued = Some(0);
    snap.ai.models[0].kv_fill = Some(0.37);
    snap.ai.models[1].backend = Some(wire::Backend::LlamaCpp);
    let bytes = to_json(&snap).expect("encode");
    let text = std::str::from_utf8(&bytes).expect("utf-8");
    assert!(
        text.contains(r#""backend":"sglang","running":1,"queued":0,"kv_fill":0.37"#),
        "{text}"
    );
    assert!(text.contains(r#""backend":"llamacpp""#), "{text}");
    assert_eq!(parse_validated(&bytes).expect("round trip"), snap);

    for word in ["vllm", "openai", "llamacpp", "sglang"] {
        let json = insert_after(
            &base_json(),
            r#""state":"ready""#,
            &format!(r#","backend":"{word}""#),
        );
        assert!(parse_validated(&json).is_ok(), "{word}");
    }
    let json = insert_after(&base_json(), r#""state":"ready""#, r#","backend":"tabby""#);
    assert_eq!(parse_validated(&json), Err(WireError::Parse));
    let json = insert_after(&base_json(), r#""state":"ready""#, r#","running":-1"#);
    assert_eq!(parse_validated(&json), Err(WireError::Parse));
}

#[test]
fn backend_gauges_are_bounded() {
    let mut snap = valid();
    snap.ai.models[0].running = Some(wire::MAX_REQS);
    snap.ai.models[0].queued = Some(wire::MAX_REQS);
    snap.ai.models[0].kv_fill = Some(1.0);
    assert_eq!(validate(&snap), Ok(()));
    snap.ai.models[0].kv_fill = Some(0.0);
    assert_eq!(validate(&snap), Ok(()));

    let mut snap = valid();
    snap.ai.models[0].running = Some(wire::MAX_REQS + 1);
    assert_eq!(validate(&snap), Err(WireError::Gauge));
    let mut snap = valid();
    snap.ai.models[0].queued = Some(u16::MAX);
    assert_eq!(validate(&snap), Err(WireError::Gauge));
    for fill in [1.01, -0.01, f32::NAN, f32::INFINITY] {
        let mut snap = valid();
        snap.ai.models[0].kv_fill = Some(fill);
        assert_eq!(validate(&snap), Err(WireError::Gauge), "{fill}");
    }
    let json = insert_after(&base_json(), r#""state":"ready""#, r#","kv_fill":1.5"#);
    assert_eq!(parse_validated(&json), Err(WireError::Gauge));
    let json = insert_after(&base_json(), r#""state":"ready""#, r#","running":70000"#);
    assert_eq!(parse_validated(&json), Err(WireError::Parse));
}
