//! Fixed-seed mutations of a v1 snapshot. Nothing here may panic.
//! An accepted buffer round-trips through [`parse_validated`](llama_core::wire::parse_validated).

use llama_core::wire::{
    Ai, AiWire, Host, ModelDetail, ModelState, ModelWire, Tokens, WireSnapshot, parse_validated,
    to_json,
};

const ITERATIONS: usize = 100_000;
const SEED: u64 = 0x17C0_FFEE_5A17_0017;

fn sample() -> WireSnapshot {
    WireSnapshot {
        schema: 1,
        run_id: 9,
        seq: 4,
        t_mono_ns: 100,
        t_wall_ms: 200,
        host: Host {
            load_pct: Some(1.0),
            activity_pct: Some(1.5),
            cpu_pct: Some(2.0),
            cpu_topk_pct: Some(3.0),
            gpu_pct: Some(4.0),
            mem_pct: Some(5.0),
            coolant_c: Some(6.0),
            cpu_c: Some(7.0),
            gpu_c: Some(8.0),
            gpu_w: None,
            gpu_limit_w: None,
            cpu_w: None,
            vram_used_bytes: None,
            vram_total_bytes: None,
            mem_used_bytes: None,
            mem_total_bytes: None,
        },
        ai: Ai {
            state: AiWire::Loaded,
            models: vec![ModelWire {
                backend: None,
                running: None,
                queued: None,
                kv_fill: None,
                name: "Qwen 35B".to_owned(),
                state: ModelState::Ready,
                full_name: Some("Qwen3.6 35B-A3B".to_owned()),
                detail: Some(ModelDetail {
                    ctx: Some(131_072),
                    ncmoe: Some(16),
                    kv_k: Some("q8_0".to_owned()),
                    kv_v: Some("q4_0".to_owned()),
                    quant: Some("UD-Q4_K_M".to_owned()),
                    fa: Some(true),
                    kv_block: None,
                    prefix_cache: None,
                }),
                cache_hit: None,
                slots_busy: None,
                slots_total: None,
                prompt_tokens: None,
                prompt_cached_tokens: None,
                slot_ctx: Vec::new(),
                engine: None,
            }],
        },
        tokens: Tokens {
            decoded_total: Some(10),
            prompt_total: None,
        },
        fans: Vec::new(),
        sources: None,
    }
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn exercise(bytes: &[u8]) {
    if let Ok(snap) = parse_validated(bytes) {
        let encoded = to_json(&snap).expect("validated snapshot encodes");
        assert_eq!(parse_validated(&encoded).expect("round trip"), snap);
    }
}

/// JSON number tokens outside strings. Digits inside names are not numbers.
fn number_spans(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut index = 0;
    let mut in_string = false;
    let mut escape = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            if escape {
                escape = false;
            } else if byte == b'\\' {
                escape = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if byte == b'"' {
            in_string = true;
            index += 1;
            continue;
        }
        if byte == b'-' || byte.is_ascii_digit() {
            let start = index;
            if byte == b'-' {
                index += 1;
            }
            let digits = index;
            while index < bytes.len() && bytes[index].is_ascii_digit() {
                index += 1;
            }
            if index == digits {
                continue;
            }
            if index < bytes.len() && bytes[index] == b'.' {
                index += 1;
                while index < bytes.len() && bytes[index].is_ascii_digit() {
                    index += 1;
                }
            }
            if index < bytes.len() && (bytes[index] == b'e' || bytes[index] == b'E') {
                let exponent = index;
                index += 1;
                if index < bytes.len() && (bytes[index] == b'+' || bytes[index] == b'-') {
                    index += 1;
                }
                let exp_digits = index;
                while index < bytes.len() && bytes[index].is_ascii_digit() {
                    index += 1;
                }
                if index == exp_digits {
                    index = exponent;
                }
            }
            spans.push((start, index));
            continue;
        }
        index += 1;
    }
    spans
}

fn replace_span(base: &[u8], start: usize, end: usize, literal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(base.len() + literal.len());
    out.extend_from_slice(&base[..start]);
    out.extend_from_slice(literal);
    out.extend_from_slice(&base[end..]);
    out
}

fn replace_random_number(base: &[u8], literal: &[u8], state: &mut u64) -> Vec<u8> {
    let spans = number_spans(base);
    if spans.is_empty() {
        return base.to_vec();
    }
    let (start, end) = spans[(xorshift(state) as usize) % spans.len()];
    replace_span(base, start, end, literal)
}

fn replace_field_number(base: &[u8], field: &str, literal: &[u8]) -> Vec<u8> {
    let key = format!("\"{field}\":");
    let text = std::str::from_utf8(base).expect("snapshot json");
    let key_at = text.find(&key).unwrap_or_else(|| panic!("missing {field}"));
    let start = key_at + key.len();
    let spans = number_spans(base);
    let (num_start, num_end) = spans
        .into_iter()
        .find(|(span_start, _)| *span_start >= start)
        .unwrap_or_else(|| panic!("{field} has no number"));
    replace_span(base, num_start, num_end, literal)
}

fn duplicate_key(base: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(base.len() + 12);
    if let Some(pos) = base.iter().position(|byte| *byte == b'{') {
        out.extend_from_slice(&base[..=pos]);
        out.extend_from_slice(br#""schema":2,"#);
        out.extend_from_slice(&base[pos + 1..]);
    } else {
        out.extend_from_slice(base);
    }
    out
}

fn deep_array(depth: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(depth * 2 + 1);
    out.extend(std::iter::repeat_n(b'[', depth));
    out.push(b'0');
    out.extend(std::iter::repeat_n(b']', depth));
    out
}

fn flip_byte(base: &[u8], state: &mut u64) -> Vec<u8> {
    if base.is_empty() {
        return vec![(xorshift(state) & 0xff) as u8];
    }
    let mut out = base.to_vec();
    let index = (xorshift(state) as usize) % out.len();
    let bit = (xorshift(state) % 8) as u8;
    out[index] ^= 1 << bit;
    out
}

fn mutate(base: &[u8], state: &mut u64) -> Vec<u8> {
    match xorshift(state) % 6 {
        0 => flip_byte(base, state),
        1 => {
            let len = (xorshift(state) as usize) % (base.len() + 1);
            base[..len].to_vec()
        }
        2 => duplicate_key(base),
        3 => replace_random_number(base, &vec![b'9'; 400], state),
        4 => {
            let literal: &[u8] = if xorshift(state).is_multiple_of(2) {
                b"1e39"
            } else {
                b"1e999"
            };
            replace_random_number(base, literal, state)
        }
        _ => deep_array(200 + (xorshift(state) as usize % 50)),
    }
}

#[test]
fn hostile_inputs_do_not_panic() {
    let base = to_json(&sample()).expect("encode");
    exercise(b"1e999");
    exercise(&deep_array(256));
    exercise(&replace_field_number(&base, "load_pct", &vec![b'9'; 400]));
    exercise(&replace_field_number(&base, "cpu_c", b"1e39"));
    exercise(&replace_field_number(&base, "gpu_pct", b"1e999"));
    exercise(&replace_field_number(&base, "decoded_total", b"1e999"));
    exercise(&duplicate_key(&base));
    let mut seed = SEED;
    exercise(&flip_byte(&base, &mut seed));
}

#[test]
fn fixed_seed_mutation_loop_accepts_only_valid_snapshots() {
    let base = to_json(&sample()).expect("encode");
    assert!(base.len() < ITERATIONS, "base json is {} bytes", base.len());
    let original = parse_validated(&base).expect("unmutated snapshot");
    assert_eq!(
        parse_validated(&to_json(&original).expect("encode")).expect("round trip"),
        original
    );

    let mut padded = base.clone();
    padded.resize(llama_core::wire::MAX_BYTES + 1, b' ');
    assert!(
        parse_validated(&padded).is_err(),
        "oversize must be rejected"
    );

    let mut exercised = 0usize;
    for len in 0..=base.len() {
        exercise(&base[..len]);
        exercised += 1;
    }
    let mut state = SEED;
    while exercised < ITERATIONS {
        exercise(&mutate(&base, &mut state));
        exercised += 1;
    }
    assert_eq!(exercised, ITERATIONS);
    assert_ne!(state, 0, "xorshift collapsed");
}
