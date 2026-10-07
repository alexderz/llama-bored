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
        gpu_w: None,
        gpu_limit_w: None,
        cpu_w: None,
        vram_used_bytes: None,
        vram_total_bytes: None,
        mem_used_bytes: None,
        mem_total_bytes: None,
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
                    cache_hit: None,
                    slots_busy: None,
                    slots_total: None,
                    prompt_tokens: None,
                    prompt_cached_tokens: None,
                    slot_ctx: Vec::new(),
                    engine: None,
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
                    cache_hit: None,
                    slots_busy: None,
                    slots_total: None,
                    prompt_tokens: None,
                    prompt_cached_tokens: None,
                    slot_ctx: Vec::new(),
                    engine: None,
                },
            ],
        },
        tokens: Tokens {
            decoded_total: Some(1_234_567),
            prompt_total: None,
        },
        fans: Vec::new(),
        sources: None,
        suspected_loads: Vec::new(),
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
            cache_hit: None,
            slots_busy: None,
            slots_total: None,
            prompt_tokens: None,
            prompt_cached_tokens: None,
            slot_ctx: Vec::new(),
            engine: None,
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
        gpu_w: None,
        gpu_limit_w: None,
        cpu_w: None,
        vram_used_bytes: None,
        vram_total_bytes: None,
        mem_used_bytes: None,
        mem_total_bytes: None,
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

/// Compatibility rule (#12): within one `schema` number a reader ignores
/// fields it does not know, so a newer watcher's additive field does not take
/// an older reader down. This replaces the pre-0.2 rule that rejected every
/// unknown field. It is safe because a reader only ever sees typed, known
/// fields, and those keep every check: the schema number, the size cap, and
/// the range and allowlist rules below. An unknown field's value is dropped
/// at the parse and can reach no display or export.
#[test]
fn unknown_fields_are_ignored_within_schema_v1() {
    let base = base_json();
    let parsed = parse_validated(base.as_bytes()).expect("base json");
    assert_eq!(parsed.schema, wire::SCHEMA);
    let cases = [
        ("root", insert_after(&base, "{", r#""nope":1,"#)),
        (
            "root object",
            insert_after(&base, "{", r#""future":{"a":[1,2,{"b":"x"}]},"#),
        ),
        ("host", insert_after(&base, r#""host":{"#, r#""nope":1,"#)),
        ("ai", insert_after(&base, r#""ai":{"#, r#""nope":1,"#)),
        (
            "model",
            insert_after(&base, r#""models":[{"#, r#""nope":"text","#),
        ),
        (
            "tokens",
            insert_after(&base, r#""tokens":{"#, r#""nope":1,"#),
        ),
    ];
    for (level, bytes) in cases {
        let snap = parse_validated(&bytes).unwrap_or_else(|err| panic!("{level}: {err}"));
        assert_eq!(snap, parsed, "{level}: the unknown field changes nothing");
    }

    // Known fields stay strict next to an unknown one.
    let extra_and_bad = insert_after(
        &base.replace(r#""gpu_pct":1.0"#, r#""gpu_pct":101.0"#),
        r#""host":{"#,
        r#""nope":1,"#,
    );
    assert_eq!(
        parse_validated(&extra_and_bad),
        Err(WireError::OutOfRange { field: "gpu_pct" })
    );
    let bad_type = insert_after(&base, r#""host":{"#, r#""nope":1,"#);
    let bad_type = String::from_utf8(bad_type)
        .expect("utf8")
        .replace(r#""cpu_c":1.0"#, r#""cpu_c":"hot""#);
    assert!(matches!(
        parse_validated(bad_type.as_bytes()),
        Err(WireError::Parse)
    ));
    // A wrong schema number is still refused, unknown fields or not.
    let v2 = insert_after(
        &base.replace(r#""schema":1"#, r#""schema":2"#),
        "{",
        r#""nope":1,"#,
    );
    assert_eq!(parse_validated(&v2), Err(WireError::Schema));
    // And the size cap still comes first.
    let mut big = insert_after(&base, "{", r#""nope":1,"#);
    big.resize(wire::MAX_BYTES + 1, b' ');
    assert_eq!(
        parse_validated(&big),
        Err(WireError::TooLong {
            len: wire::MAX_BYTES + 1
        })
    );

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
            cache_hit: None,
            slots_busy: None,
            slots_total: None,
            prompt_tokens: None,
            prompt_cached_tokens: None,
            slot_ctx: Vec::new(),
            engine: None,
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
        cache_hit: None,
        slots_busy: None,
        slots_total: None,
        prompt_tokens: None,
        prompt_cached_tokens: None,
        slot_ctx: Vec::new(),
        engine: None,
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
        cache_hit: None,
        slots_busy: None,
        slots_total: None,
        prompt_tokens: None,
        prompt_cached_tokens: None,
        slot_ctx: Vec::new(),
        engine: None,
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
        kv_block: None,
        prefix_cache: None,
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
    // An unknown detail key is ignored under the #12 rule: the path is
    // dropped at the parse, so no reader can draw or export it.
    let unknown = insert_after(
        &base_json(),
        r#""state":"ready""#,
        r#","detail":{"cmd":"/models/prism/llama-server"}"#,
    );
    let parsed = parse_validated(&unknown).expect("unknown detail key is ignored");
    assert_eq!(
        parsed.ai.models[0].detail,
        Some(wire::ModelDetail::default())
    );
    let debug = format!("{parsed:?}");
    assert!(!debug.contains("/models"), "{debug}");
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

    let mut strata = valid();
    strata.ai.models[0].backend = Some(wire::Backend::Strata);
    strata.ai.models[0].running = Some(1);
    strata.ai.models[0].queued = Some(0);
    let bytes = to_json(&strata).expect("encode");
    let text = std::str::from_utf8(&bytes).expect("utf-8");
    assert!(
        text.contains(r#""backend":"strata","running":1,"queued":0"#),
        "{text}"
    );
    assert_eq!(parse_validated(&bytes).expect("strata round trip"), strata);

    for word in ["vllm", "openai", "llamacpp", "sglang", "strata"] {
        let json = insert_after(
            &base_json(),
            r#""state":"ready""#,
            &format!(r#","backend":"{word}""#),
        );
        let snap = parse_validated(&json).expect(word);
        assert_eq!(
            snap.ai.models[0].backend.map(wire::Backend::as_str),
            Some(word)
        );
    }
    // A newer watcher's backend word reads as openai (no gauges this reader
    // knows) instead of rejecting the whole snapshot. A non-string still fails.
    let json = insert_after(&base_json(), r#""state":"ready""#, r#","backend":"tabby""#);
    let snap = parse_validated(&json).expect("unknown backend word");
    assert_eq!(snap.ai.models[0].backend, Some(wire::Backend::OpenAi));
    let json = insert_after(&base_json(), r#""state":"ready""#, r#","backend":1"#);
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

/// Every #11 field set, at its top value where there is one.
fn full() -> WireSnapshot {
    let mut snap = valid();
    snap.host.gpu_w = Some(wire::MAX_WATTS);
    snap.host.gpu_limit_w = Some(600.0);
    snap.host.cpu_w = Some(0.0);
    snap.host.vram_used_bytes = Some(wire::MAX_MEM_BYTES);
    snap.host.vram_total_bytes = Some(34_190_917_632);
    snap.host.mem_used_bytes = Some(0);
    snap.host.mem_total_bytes = Some(134_217_728_000);
    snap.tokens.prompt_total = Some(u64::MAX);
    snap.ai.models[0].cache_hit = Some(1.0);
    snap.ai.models[0].slots_busy = Some(wire::MAX_SLOTS);
    snap.ai.models[0].slots_total = Some(wire::MAX_SLOTS);
    snap.ai.models[1].slots_busy = Some(0);
    snap.ai.models[1].slots_total = Some(4);
    snap.fans = (1..=8)
        .map(|channel| wire::FanWire {
            channel: channel * 2,
            label: "CPU fan ~9".to_owned(),
            rpm: Some(wire::MAX_FAN_RPM),
            pwm: Some(1.0),
        })
        .collect();
    let up = |latency: f32| {
        Some(wire::SourceWire {
            up: true,
            latency_s: Some(latency),
        })
    };
    snap.sources = Some(wire::Sources {
        llama_swap: up(wire::MAX_LATENCY_S),
        running: up(0.0),
        slots: Some(wire::SourceWire {
            up: false,
            latency_s: None,
        }),
        metrics: up(0.004),
        activity: up(0.01),
        gpu: up(0.0),
        hwmon: up(0.0),
        proc: up(0.0),
    });
    snap.suspected_loads = (0..wire::MAX_SUSPECTED_LOADS)
        .map(|n| wire::SuspectedLoadWire {
            model: format!("{n}{}", "x".repeat(wire::MAX_FULL_NAME_CHARS - 1)),
            count: u64::MAX,
        })
        .collect();
    snap
}

#[test]
fn suspected_loads_are_additive_and_bounded() {
    // #70: an older watcher sends none; nothing writes the key when empty.
    let old = parse_validated(base_json().as_bytes()).expect("pre-#70 snapshot");
    assert!(old.suspected_loads.is_empty());
    let text = String::from_utf8(to_json(&valid()).expect("encode")).expect("utf8");
    assert!(!text.contains("suspected"), "{text}");
    let json = insert_after(
        &base_json(),
        "{",
        r#""suspected_loads":[{"model":"qwen3.6-35b-a3b","count":3}],"#,
    );
    let snap = parse_validated(&json).expect("one row");
    assert_eq!(snap.suspected_loads[0].count, 3);

    let gauge_cases: [Mutate; 3] = [
        |s| s.suspected_loads[0].count = 0,
        |s| s.suspected_loads[1].model = s.suspected_loads[0].model.clone(),
        |s| {
            s.suspected_loads.push(wire::SuspectedLoadWire {
                model: "one-more".to_owned(),
                count: 1,
            })
        },
    ];
    for (i, bad) in gauge_cases.into_iter().enumerate() {
        let mut snap = full();
        bad(&mut snap);
        assert_eq!(validate(&snap), Err(WireError::Gauge), "case {i}");
    }
    let detail_cases: [Mutate; 3] = [
        |s| s.suspected_loads[0].model = String::new(),
        |s| s.suspected_loads[0].model = "a\nb".to_owned(),
        |s| s.suspected_loads[0].model = "x".repeat(wire::MAX_FULL_NAME_CHARS + 1),
    ];
    for (i, bad) in detail_cases.into_iter().enumerate() {
        let mut snap = full();
        bad(&mut snap);
        assert_eq!(validate(&snap), Err(WireError::Detail), "case {i}");
    }
    let json = insert_after(
        &base_json(),
        "{",
        r#""suspected_loads":[{"model":"m","count":-1}],"#,
    );
    assert_eq!(parse_validated(&json), Err(WireError::Parse));
}

#[test]
fn metrics_fields_are_additive_on_schema_v1() {
    // An older watcher's snapshot has none of them and still validates.
    let old = parse_validated(base_json().as_bytes()).expect("pre-#11 snapshot");
    assert_eq!(old.host.gpu_w, None);
    assert_eq!(old.tokens.prompt_total, None);
    assert_eq!(old.ai.models[0].slots_total, None);
    assert!(old.fans.is_empty());
    assert_eq!(old.sources, None);

    // Nothing known writes the old shape: no new keys at all.
    let text = String::from_utf8(to_json(&valid()).expect("encode")).expect("utf8");
    for key in [
        "gpu_w",
        "cpu_w",
        "_bytes",
        "prompt_total",
        "cache_hit",
        "slots_",
        "fans",
        "sources",
    ] {
        assert!(!text.contains(key), "{key}: {text}");
    }

    let snap = full();
    let bytes = to_json(&snap).expect("encode");
    assert_eq!(parse_validated(&bytes).expect("round trip"), snap);
    let text = std::str::from_utf8(&bytes).expect("utf8");
    assert!(text.contains(r#""llama-swap":{"up":true"#), "{text}");
}

#[test]
fn a_full_snapshot_with_eight_long_models_fits_the_cap() {
    let mut snap = full();
    let model = ModelWire {
        name: "Qwen3-Coder…".to_owned(),
        state: ModelState::Ready,
        full_name: Some("x".repeat(wire::MAX_FULL_NAME_CHARS)),
        detail: Some(wire::ModelDetail {
            ctx: Some(u32::MAX),
            ncmoe: Some(u16::MAX),
            kv_k: Some("a".repeat(16)),
            kv_v: Some("b".repeat(16)),
            quant: Some("c".repeat(16)),
            fa: Some(true),
            kv_block: Some(llama_core::detail::MAX_KV_BLOCK),
            prefix_cache: Some(false),
        }),
        backend: Some(wire::Backend::LlamaCpp),
        running: Some(wire::MAX_REQS),
        queued: Some(wire::MAX_REQS),
        kv_fill: Some(0.123_456_7),
        cache_hit: Some(0.123_456_7),
        slots_busy: Some(wire::MAX_SLOTS),
        slots_total: Some(wire::MAX_SLOTS),
        prompt_tokens: None,
        prompt_cached_tokens: None,
        slot_ctx: Vec::new(),
        engine: None,
    };
    snap.ai.models = vec![model; wire::MAX_MODELS];
    snap.host.load_pct = Some(12.345_678);
    let bytes = to_json(&snap).expect("encode");
    assert!(
        bytes.len() < wire::MAX_BYTES / 2,
        "worst case is {} bytes",
        bytes.len()
    );
    assert!(parse_validated(&bytes).is_ok());
}

type Mutate = fn(&mut WireSnapshot);

#[test]
fn metrics_fields_are_bounded() {
    assert_eq!(validate(&full()), Ok(()));
    let host_cases: [(&str, Mutate); 9] = [
        ("gpu_w", |s| s.host.gpu_w = Some(-0.1)),
        ("gpu_w", |s| s.host.gpu_w = Some(wire::MAX_WATTS + 1.0)),
        ("gpu_w", |s| s.host.gpu_w = Some(f32::NAN)),
        ("gpu_limit_w", |s| s.host.gpu_limit_w = Some(f32::INFINITY)),
        ("cpu_w", |s| s.host.cpu_w = Some(-5.0)),
        ("vram_used_bytes", |s| {
            s.host.vram_used_bytes = Some(wire::MAX_MEM_BYTES + 1)
        }),
        ("vram_total_bytes", |s| {
            s.host.vram_total_bytes = Some(u64::MAX)
        }),
        ("mem_used_bytes", |s| s.host.mem_used_bytes = Some(u64::MAX)),
        ("mem_total_bytes", |s| {
            s.host.mem_total_bytes = Some(u64::MAX)
        }),
    ];
    for (field, bad) in host_cases {
        let mut snap = full();
        bad(&mut snap);
        assert_eq!(validate(&snap), Err(WireError::OutOfRange { field }));
    }

    let gauge_cases: [Mutate; 6] = [
        |s| s.ai.models[0].cache_hit = Some(1.01),
        |s| s.ai.models[0].cache_hit = Some(f32::NAN),
        |s| s.ai.models[1].slots_busy = Some(5),
        |s| s.ai.models[1].slots_total = None,
        |s| s.ai.models[0].slots_total = Some(wire::MAX_SLOTS + 1),
        |s| {
            s.ai.models[0].slots_busy = None;
            s.ai.models[0].slots_total = Some(wire::MAX_SLOTS + 1);
        },
    ];
    for (i, bad) in gauge_cases.into_iter().enumerate() {
        let mut snap = full();
        bad(&mut snap);
        assert_eq!(validate(&snap), Err(WireError::Gauge), "case {i}");
    }

    let fan_cases: [Mutate; 10] = [
        |s| s.fans.push(s.fans[0].clone()),
        |s| s.fans[1].channel = s.fans[0].channel,
        |s| s.fans[0].channel = 0,
        |s| s.fans[0].channel = wire::MAX_FAN_CHANNEL + 1,
        |s| s.fans[0].label = String::new(),
        |s| s.fans[0].label = "x".repeat(wire::MAX_FAN_LABEL_CHARS + 1),
        |s| s.fans[0].label = "fan\n1".to_owned(),
        |s| s.fans[0].label = "fän".to_owned(),
        |s| s.fans[0].rpm = Some(wire::MAX_FAN_RPM + 1),
        |s| s.fans[0].pwm = Some(1.5),
    ];
    for (i, bad) in fan_cases.into_iter().enumerate() {
        let mut snap = full();
        bad(&mut snap);
        assert_eq!(validate(&snap), Err(WireError::Fan), "case {i}");
    }

    for latency in [-0.001, wire::MAX_LATENCY_S + 1.0, f32::NAN] {
        let mut snap = full();
        if let Some(sources) = snap.sources.as_mut() {
            sources.metrics = Some(wire::SourceWire {
                up: true,
                latency_s: Some(latency),
            });
        }
        assert_eq!(
            validate(&snap),
            Err(WireError::OutOfRange { field: "latency_s" }),
            "{latency}"
        );
    }

    // Types stay strict on the wire: a negative count or a string is a parse error.
    for extra in [r#""vram_used_bytes":-1,"#, r#""gpu_w":"hot","#] {
        let json = insert_after(&base_json(), r#""host":{"#, extra);
        assert_eq!(parse_validated(&json), Err(WireError::Parse), "{extra}");
    }
    let json = insert_after(&base_json(), "{", r#""sources":{"proc":{"up":"yes"}},"#);
    assert_eq!(parse_validated(&json), Err(WireError::Parse));
    // A source this reader does not know is ignored (#12).
    let json = insert_after(&base_json(), "{", r#""sources":{"tpu":{"up":true}},"#);
    let snap = parse_validated(&json).expect("unknown source is ignored");
    assert_eq!(snap.sources, Some(wire::Sources::default()));
}

// ---- #10: per-slot context and prompt cache counters ------------------------

fn slot_row(slot: u16, used: u64, resets: u64) -> wire::SlotCtxWire {
    wire::SlotCtxWire {
        slot,
        used,
        resets: wire::SlotResetsWire {
            new: resets,
            ..Default::default()
        },
    }
}

#[test]
fn slot_ctx_and_prompt_counters_are_additive_on_schema_v1() {
    // An older watcher's model has none of them.
    let old = parse_validated(base_json().as_bytes()).expect("pre-#10 snapshot");
    assert_eq!(old.ai.models[0].prompt_tokens, None);
    assert_eq!(old.ai.models[0].prompt_cached_tokens, None);
    assert!(old.ai.models[0].slot_ctx.is_empty());
    let text = String::from_utf8(to_json(&valid()).expect("encode")).expect("utf8");
    for key in ["prompt_tokens", "prompt_cached", "slot_ctx"] {
        assert!(!text.contains(key), "{key}: {text}");
    }

    let mut snap = valid();
    snap.ai.models[0].prompt_tokens = Some(u64::MAX);
    snap.ai.models[0].prompt_cached_tokens = Some(u64::MAX);
    snap.ai.models[0].slot_ctx = vec![
        slot_row(0, wire::MAX_CTX_TOKENS, u64::MAX),
        slot_row(wire::MAX_SLOTS - 1, 0, 0),
    ];
    let bytes = to_json(&snap).expect("encode");
    assert_eq!(parse_validated(&bytes).expect("round trip"), snap);

    // A reader that predates `resets` inside a row still reads the row.
    let json = insert_after(
        &base_json(),
        r#""state":"ready""#,
        r#","prompt_tokens":10,"prompt_cached_tokens":4,"slot_ctx":[{"slot":1,"used":5}]"#,
    );
    let snap = parse_validated(&json).expect("resets defaults to 0");
    assert_eq!(snap.ai.models[0].slot_ctx, vec![slot_row(1, 5, 0)]);
    assert_eq!(snap.ai.models[0].prompt_cached_tokens, Some(4));

    // #9: drops by reason; a zero is omitted, and a reason this reader does
    // not know is ignored like any unknown field.
    let mut two = valid();
    two.ai.models[0].slot_ctx = vec![slot_row(0, 9, 2)];
    let text = String::from_utf8(to_json(&two).expect("encode")).expect("utf8");
    assert!(text.contains(r#""resets":{"new":2}"#), "{text}");
    let json = insert_after(
        &base_json(),
        r#""state":"ready""#,
        r#","slot_ctx":[{"slot":0,"used":9,"resets":{"compacted":1,"evicted":3,"unknown":4,"merged":7}}]"#,
    );
    let snap = parse_validated(&json).expect("unknown reason ignored");
    assert_eq!(
        snap.ai.models[0].slot_ctx[0].resets,
        wire::SlotResetsWire {
            compacted: 1,
            new: 0,
            evicted: 3,
            unknown: 4
        }
    );
    let json = insert_after(
        &base_json(),
        r#""state":"ready""#,
        r#","slot_ctx":[{"slot":0,"used":9,"resets":{"new":-1}}]"#,
    );
    assert_eq!(parse_validated(&json), Err(WireError::Parse));
}

#[test]
fn slot_ctx_and_prompt_counters_are_bounded() {
    let cases: [Mutate; 7] = [
        |s| s.ai.models[0].prompt_cached_tokens = Some(1),
        |s| {
            s.ai.models[0].prompt_tokens = Some(3);
            s.ai.models[0].prompt_cached_tokens = Some(4);
        },
        |s| s.ai.models[0].slot_ctx = vec![slot_row(wire::MAX_SLOTS, 1, 0)],
        |s| s.ai.models[0].slot_ctx = vec![slot_row(0, wire::MAX_CTX_TOKENS + 1, 0)],
        |s| s.ai.models[0].slot_ctx = vec![slot_row(2, 1, 0), slot_row(2, 3, 0)],
        |s| {
            s.ai.models[0].slot_ctx = (0..=wire::MAX_SLOT_CTX as u16)
                .map(|id| slot_row(id, 1, 0))
                .collect();
        },
        // The cap counts every model's rows together.
        |s| {
            s.ai.models[0].slot_ctx = (0..20).map(|id| slot_row(id, 1, 0)).collect();
            s.ai.models[1].slot_ctx = (0..13).map(|id| slot_row(id, 1, 0)).collect();
        },
    ];
    for (i, bad) in cases.into_iter().enumerate() {
        let mut snap = valid();
        bad(&mut snap);
        assert_eq!(validate(&snap), Err(WireError::Gauge), "case {i}");
    }
    // The same slot id in two models is fine; so is exactly the cap.
    let mut snap = valid();
    snap.ai.models[0].slot_ctx = (0..16).map(|id| slot_row(id, 1, 0)).collect();
    snap.ai.models[1].slot_ctx = (0..16).map(|id| slot_row(id, 1, 0)).collect();
    assert_eq!(validate(&snap), Ok(()));
    // Negative or text values are parse errors.
    for extra in [
        r#","slot_ctx":[{"slot":-1,"used":1}]"#,
        r#","slot_ctx":[{"slot":0,"used":"lots"}]"#,
        r#","prompt_tokens":-5"#,
    ] {
        let json = insert_after(&base_json(), r#""state":"ready""#, extra);
        assert_eq!(parse_validated(&json), Err(WireError::Parse), "{extra}");
    }
}

#[test]
fn a_worst_case_snapshot_with_every_slot_row_fits_the_cap() {
    let mut snap = full();
    let model = ModelWire {
        name: "Qwen3-Coder…".to_owned(),
        state: ModelState::Ready,
        full_name: Some("x".repeat(wire::MAX_FULL_NAME_CHARS)),
        detail: Some(wire::ModelDetail {
            ctx: Some(u32::MAX),
            ncmoe: Some(u16::MAX),
            kv_k: Some("a".repeat(16)),
            kv_v: Some("b".repeat(16)),
            quant: Some("c".repeat(16)),
            fa: Some(true),
            kv_block: Some(llama_core::detail::MAX_KV_BLOCK),
            prefix_cache: Some(false),
        }),
        backend: Some(wire::Backend::LlamaCpp),
        running: Some(wire::MAX_REQS),
        queued: Some(wire::MAX_REQS),
        kv_fill: Some(0.123_456_7),
        cache_hit: Some(0.123_456_7),
        slots_busy: Some(wire::MAX_SLOTS),
        slots_total: Some(wire::MAX_SLOTS),
        prompt_tokens: Some(u64::MAX),
        prompt_cached_tokens: Some(u64::MAX),
        slot_ctx: Vec::new(),
        // #31: every engine number, at its widest.
        engine: Some(wire::EngineWire {
            spec_accept: Some(0.123_456_7),
            spec_len: Some(12.345_678),
            spec_drafts: Some(u64::MAX),
            spec_draft_tokens: Some(u64::MAX),
            spec_accepted_tokens: Some(u64::MAX),
            preemptions: Some(u64::MAX),
            sleeping: Some(false),
            ttft_s: Some(1_234.567_8),
            itl_s: Some(0.012_345_67),
            e2e_s: Some(3_599.123_4),
            // #35: both speeds at their widest.
            prefill_tps: Some(987_654.3),
            decode_tps: Some(123_456.79),
            // #54: Strata's expert cache ratios.
            expert_hit: Some(0.123_456_7),
            pcie_share: Some(0.123_456_7),
        }),
    };
    snap.ai.models = vec![model; wire::MAX_MODELS];
    snap.ai.models[0].slot_ctx = (0..wire::MAX_SLOT_CTX as u16)
        .map(|id| wire::SlotCtxWire {
            slot: wire::MAX_SLOTS - 1 - id,
            used: wire::MAX_CTX_TOKENS,
            resets: wire::SlotResetsWire {
                compacted: u64::MAX,
                new: u64::MAX,
                evicted: u64::MAX,
                unknown: u64::MAX,
            },
        })
        .collect();
    snap.host.load_pct = Some(12.345_678);
    let bytes = to_json(&snap).expect("encode");
    assert!(
        bytes.len() < wire::MAX_BYTES - 1024,
        "worst case is {} bytes",
        bytes.len()
    );
    assert!(parse_validated(&bytes).is_ok());
}

/// #31: engine numbers are optional, bounded, and an older reader's
/// unknown key.
#[test]
fn engine_numbers_are_optional_and_bounded() {
    let engine = || wire::EngineWire {
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
        decode_tps: Some(41.25),
        expert_hit: Some(0.856),
        pcie_share: Some(0.106),
    };
    let mut snap = valid();
    snap.ai.models[0].engine = Some(engine());
    let bytes = to_json(&snap).expect("encode");
    let text = std::str::from_utf8(&bytes).expect("utf8");
    assert!(text.contains(r#""engine":{"spec_accept":0.78"#), "{text}");
    assert!(
        text.contains(
            r#""prefill_tps":2134.5,"decode_tps":41.25,"expert_hit":0.856,"pcie_share":0.106}"#
        ),
        "{text}"
    );
    assert_eq!(parse_validated(&bytes).expect("valid"), snap);
    // Absent and empty both read as no numbers; an unknown inner key is ignored.
    let json = insert_after(
        &base_json(),
        r#""state":"ready""#,
        r#","engine":{"sleeping":true,"spec_tree":[1,2]}"#,
    );
    let read = parse_validated(&json).expect("unknown engine key ignored");
    assert_eq!(
        read.ai.models[0].engine.as_ref().and_then(|e| e.sleeping),
        Some(true)
    );
    let cases: [Mutate; 18] = [
        |s| s.ai.models[0].engine.as_mut().unwrap().spec_accept = Some(1.01),
        |s| s.ai.models[0].engine.as_mut().unwrap().spec_accept = Some(f32::NAN),
        |s| s.ai.models[0].engine.as_mut().unwrap().spec_len = Some(0.5),
        |s| s.ai.models[0].engine.as_mut().unwrap().spec_len = Some(65.0),
        |s| s.ai.models[0].engine.as_mut().unwrap().spec_accepted_tokens = Some(301),
        |s| s.ai.models[0].engine.as_mut().unwrap().spec_draft_tokens = None,
        |s| s.ai.models[0].engine.as_mut().unwrap().ttft_s = Some(-0.1),
        |s| s.ai.models[0].engine.as_mut().unwrap().e2e_s = Some(3600.5),
        // #35: speeds are 0..=MAX_ENGINE_TPS and finite.
        |s| s.ai.models[0].engine.as_mut().unwrap().prefill_tps = Some(-1.0),
        |s| s.ai.models[0].engine.as_mut().unwrap().prefill_tps = Some(1_000_001.0),
        |s| s.ai.models[0].engine.as_mut().unwrap().decode_tps = Some(f32::INFINITY),
        |s| s.ai.models[0].engine.as_mut().unwrap().decode_tps = Some(-0.5),
        // #54: the expert cache ratios are 0..=1 and finite.
        |s| s.ai.models[0].engine.as_mut().unwrap().expert_hit = Some(1.01),
        |s| s.ai.models[0].engine.as_mut().unwrap().expert_hit = Some(-0.01),
        |s| s.ai.models[0].engine.as_mut().unwrap().expert_hit = Some(f32::NAN),
        |s| s.ai.models[0].engine.as_mut().unwrap().pcie_share = Some(1.5),
        |s| s.ai.models[0].engine.as_mut().unwrap().pcie_share = Some(-1.0),
        |s| s.ai.models[0].engine.as_mut().unwrap().pcie_share = Some(f32::INFINITY),
    ];
    for (i, bad) in cases.into_iter().enumerate() {
        let mut snap = valid();
        snap.ai.models[0].engine = Some(engine());
        bad(&mut snap);
        assert_eq!(validate(&snap), Err(WireError::Gauge), "case {i}");
    }
    // KV block size and prefix caching ride in the detail.
    let mut snap = valid();
    snap.ai.models[0].detail = Some(wire::ModelDetail {
        kv_block: Some(16),
        prefix_cache: Some(true),
        ..wire::ModelDetail::default()
    });
    assert_eq!(validate(&snap), Ok(()));
    snap.ai.models[0].detail.as_mut().unwrap().kv_block = Some(0);
    assert_eq!(validate(&snap), Err(WireError::Detail));
}
