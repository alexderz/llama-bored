//! Per-model tuning detail: context size, CPU MoE layers, KV precision,
//! quant tag, and flash attention.
//!
//! The watcher derives it from the llama-swap launch command
//! (`llama-watch` `sources::cmdline`). Only numbers and short allowlisted
//! tokens are kept, never a path or a raw argument. The tty draws [`line`];
//! the LCD draws [`fitted`] to its chord.

use serde::{Deserialize, Serialize};

/// Longest token kept for a KV type or a quant tag.
pub const MAX_TOKEN_CHARS: usize = 16;
/// Longest full display name carried beside the canonical wire name.
pub const MAX_FULL_NAME_CHARS: usize = 48;
/// `ncmoe` value meaning every MoE layer stays on the CPU (`--cpu-moe`).
pub const NCMOE_ALL: u16 = u16::MAX;
/// KV cache type llama.cpp uses when no `-ctk` / `-ctv` is given.
pub const KV_DEFAULT: &str = "f16";
/// Separator between detail items on the LCD and the tty.
pub const SEPARATOR: &str = " \u{00B7} ";
/// Largest KV cache block size kept, tokens.
pub const MAX_KV_BLOCK: u32 = 1 << 20;

/// Tuning numbers and tokens for one model. Every field is optional.
///
/// Part of the snapshot wire: an unknown key is ignored, like every wire
/// struct (see [`crate::wire`], Compatibility).
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ModelDetail {
    /// Context size in tokens (`-c`). Absent or zero is the model default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx: Option<u32>,
    /// MoE layers kept on the CPU (`-ncmoe`). [`NCMOE_ALL`] is `--cpu-moe`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ncmoe: Option<u16>,
    /// K cache type (`-ctk`). Absent is [`KV_DEFAULT`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_k: Option<String>,
    /// V cache type (`-ctv`). Absent is [`KV_DEFAULT`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_v: Option<String>,
    /// GGUF quant tag from the model file name, such as `Q4_K_M`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    /// Flash attention on or off. Absent is `auto` or unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fa: Option<bool>,
    /// KV cache block size in tokens, 1..=[`MAX_KV_BLOCK`] (vLLM's
    /// `cache_config_info` `block_size`, #31). The tty draws it; the LCD does not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_block: Option<u32>,
    /// Prefix caching on or off (vLLM's `enable_prefix_caching`, #31).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_cache: Option<bool>,
}

/// True for a token made only of `[A-Za-z0-9_.+-]`, 1..=[`MAX_TOKEN_CHARS`] long.
#[must_use]
pub fn is_token(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= MAX_TOKEN_CHARS
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'+' | b'-'))
}

/// True when every token field passes [`is_token`] and the KV block size
/// is within 1..=[`MAX_KV_BLOCK`].
#[must_use]
pub fn is_valid(detail: &ModelDetail) -> bool {
    [&detail.kv_k, &detail.kv_v, &detail.quant]
        .into_iter()
        .flatten()
        .all(|token| is_token(token))
        && detail
            .kv_block
            .is_none_or(|block| (1..=MAX_KV_BLOCK).contains(&block))
}

/// The engine's own cache facts for the tty (#31): `block 16`, `prefix on`.
/// Empty when the server reported none. Not part of [`line`] or the LCD.
#[must_use]
pub fn engine_items(detail: &ModelDetail) -> Vec<String> {
    let mut items = Vec::new();
    if let Some(block) = detail
        .kv_block
        .filter(|block| (1..=MAX_KV_BLOCK).contains(block))
    {
        items.push(format!("block {block}"));
    }
    if let Some(on) = detail.prefix_cache {
        items.push(if on { "prefix on" } else { "prefix off" }.to_owned());
    }
    items
}

/// [`fitted`] with `tail` (the LCD's `spec 78 %`, #31) kept at the end.
///
/// The detail items drop in [`fitted`]'s order while `tail` stays; when
/// even the shortest detail with `tail` is too wide, `tail` goes and the
/// detail is fitted alone. No detail at all draws `tail` alone, cut with
/// `…` if it must be.
#[must_use]
pub fn fitted_with(
    detail: Option<&ModelDetail>,
    tail: Option<&str>,
    fits: impl Fn(&str) -> bool,
) -> String {
    let Some(tail) = tail.filter(|tail| !tail.is_empty()) else {
        return detail
            .map(|detail| fitted(detail, &fits))
            .unwrap_or_default();
    };
    let Some(detail) = detail else {
        return cut(tail, &fits);
    };
    let quant = detail.quant.as_deref().filter(|q| is_token(q));
    let stripped = quant.map(|q| q.strip_prefix("UD-").unwrap_or(q));
    let steps = [
        build(detail, quant, true),
        build(detail, stripped, true),
        build(detail, None, true),
        build(detail, None, false),
    ];
    for text in &steps {
        let text = if text.is_empty() {
            tail.to_owned()
        } else {
            format!("{text}{SEPARATOR}{tail}")
        };
        if fits(&text) {
            return text;
        }
    }
    fitted(detail, fits)
}

/// `text`, or its longest prefix plus `…` that `fits` accepts.
fn cut(text: &str, fits: &impl Fn(&str) -> bool) -> String {
    if fits(text) {
        return text.to_owned();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut keep = chars.len();
    loop {
        let text = format!("{}\u{2026}", chars[..keep].iter().collect::<String>());
        if keep <= 1 || fits(&text) {
            return text;
        }
        keep -= 1;
    }
}

/// The whole detail line: `{ctx} · kv {kv} · {quant} · moe {n}`.
///
/// For example `256k · kv q8 · UD-Q4_K_M · moe 16`. An absent ctx or quant
/// is left out, as is moe when absent or zero. KV is always there: absent
/// flags mean `f16`. Flash attention is carried on the wire but not drawn.
#[must_use]
pub fn line(detail: &ModelDetail) -> String {
    build(
        detail,
        detail.quant.as_deref().filter(|q| is_token(q)),
        true,
    )
}

/// [`line`] shortened until `fits` accepts it (T51 V1 drop order).
///
/// First the `UD-` vendor prefix comes off the quant, then the quant goes,
/// then the kv token. ctx and moe are the knobs that get tuned, so they
/// stay. If even that is too wide, the end is cut with `…`.
#[must_use]
pub fn fitted(detail: &ModelDetail, fits: impl Fn(&str) -> bool) -> String {
    let quant = detail.quant.as_deref().filter(|q| is_token(q));
    let stripped = quant.map(|q| q.strip_prefix("UD-").unwrap_or(q));
    let steps = [
        build(detail, quant, true),
        build(detail, stripped, true),
        build(detail, None, true),
        build(detail, None, false),
    ];
    for text in &steps {
        if fits(text) {
            return text.clone();
        }
    }
    let last: Vec<char> = steps[3].chars().collect();
    let mut keep = last.len();
    loop {
        let text = format!("{}\u{2026}", last[..keep].iter().collect::<String>());
        if keep <= 1 || fits(&text) {
            return text;
        }
        keep -= 1;
    }
}

fn build(detail: &ModelDetail, quant: Option<&str>, kv: bool) -> String {
    let mut parts = Vec::new();
    if let Some(ctx) = detail.ctx.filter(|ctx| *ctx > 0) {
        parts.push(ctx_text(ctx));
    }
    if kv {
        parts.push(kv_text(detail));
    }
    if let Some(quant) = quant {
        parts.push(quant.to_owned());
    }
    match detail.ncmoe {
        Some(0) | None => {}
        Some(NCMOE_ALL) => parts.push("moe all".to_owned()),
        Some(n) => parts.push(format!("moe {n}")),
    }
    parts.join(SEPARATOR)
}

/// Binary thousands: `262144` is `256k`, `131072` is `128k`; from 1 Mi on,
/// whole `M`. Rounded to nearest.
fn ctx_text(ctx: u32) -> String {
    let ctx = u64::from(ctx);
    if ctx >= 1 << 20 {
        format!("{}M", (ctx + (1 << 19)) >> 20)
    } else if ctx >= 1 << 10 {
        format!("{}k", (ctx + (1 << 9)) >> 10)
    } else {
        ctx.to_string()
    }
}

/// `kv q8` when K and V match, else `kv q8/q4`. A trailing `_0` is dropped.
fn kv_text(detail: &ModelDetail) -> String {
    let token = |value: &Option<String>| -> String {
        let token = value
            .as_deref()
            .filter(|t| is_token(t))
            .unwrap_or(KV_DEFAULT);
        token.strip_suffix("_0").unwrap_or(token).to_owned()
    };
    let k = token(&detail.kv_k);
    let v = token(&detail.kv_v);
    if k == v {
        format!("kv {k}")
    } else {
        format!("kv {k}/{v}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bonsai() -> ModelDetail {
        ModelDetail {
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

    fn qwen() -> ModelDetail {
        ModelDetail {
            ncmoe: Some(16),
            quant: Some("UD-Q4_K_M".to_owned()),
            ..bonsai()
        }
    }

    #[test]
    fn v1_line_format() {
        assert_eq!(line(&bonsai()), "256k · kv q8 · PTQ1_0");
        assert_eq!(line(&qwen()), "256k · kv q8 · UD-Q4_K_M · moe 16");
    }

    #[test]
    fn kv_defaults_and_mixed() {
        let mut detail = ModelDetail::default();
        assert_eq!(line(&detail), "kv f16");
        detail.kv_k = Some("q8_0".to_owned());
        detail.kv_v = Some("q4_0".to_owned());
        assert_eq!(line(&detail), "kv q8/q4");
        detail.kv_v = None;
        assert_eq!(line(&detail), "kv q8/f16");
    }

    #[test]
    fn ctx_and_moe_text() {
        assert_eq!(ctx_text(262_144), "256k");
        assert_eq!(ctx_text(131_072), "128k");
        assert_eq!(ctx_text(32_768), "32k");
        assert_eq!(ctx_text(1_048_576), "1M");
        assert_eq!(ctx_text(512), "512");
        let detail = ModelDetail {
            ctx: Some(0),
            ncmoe: Some(NCMOE_ALL),
            ..ModelDetail::default()
        };
        assert_eq!(line(&detail), "kv f16 · moe all");
        let detail = ModelDetail {
            ncmoe: Some(0),
            fa: Some(true),
            ..ModelDetail::default()
        };
        assert_eq!(line(&detail), "kv f16");
    }

    #[test]
    fn drop_order_is_ud_then_quant_then_kv_then_ellipsis() {
        let detail = qwen();
        let full = line(&detail);
        let by_len = |max: usize| fitted(&detail, |text: &str| text.chars().count() <= max);
        assert_eq!(by_len(99), full);
        assert_eq!(by_len(30), "256k · kv q8 · Q4_K_M · moe 16");
        assert_eq!(by_len(25), "256k · kv q8 · moe 16");
        assert_eq!(by_len(15), "256k · moe 16");
        assert_eq!(by_len(8), "256k · …");
        assert_eq!(by_len(0), "2…");
    }

    #[test]
    fn a_spec_tail_stays_while_the_detail_drops() {
        let detail = qwen();
        let tail = Some("spec 78 %");
        let by_len = |max: usize| {
            fitted_with(Some(&detail), tail, |text: &str| {
                text.chars().count() <= max
            })
        };
        assert_eq!(by_len(99), "256k · kv q8 · UD-Q4_K_M · moe 16 · spec 78 %");
        assert_eq!(by_len(42), "256k · kv q8 · Q4_K_M · moe 16 · spec 78 %");
        assert_eq!(by_len(33), "256k · kv q8 · moe 16 · spec 78 %");
        assert_eq!(by_len(25), "256k · moe 16 · spec 78 %");
        // Too narrow for the tail: the detail alone, as before.
        assert_eq!(by_len(24), "256k · kv q8 · moe 16");
        assert_eq!(
            fitted_with(Some(&detail), None, |t: &str| t.chars().count() <= 99),
            line(&detail)
        );
        // No detail: the tail alone, cut if it must be.
        assert_eq!(fitted_with(None, tail, |_: &str| true), "spec 78 %");
        assert_eq!(
            fitted_with(None, tail, |t: &str| t.chars().count() <= 5),
            "spec…"
        );
        assert_eq!(fitted_with(None, None, |_: &str| true), "");
        // vLLM with only a KV dtype from cache_config_info.
        let vllm = ModelDetail {
            kv_k: Some("fp8_e4m3".to_owned()),
            kv_v: Some("fp8_e4m3".to_owned()),
            ..ModelDetail::default()
        };
        assert_eq!(
            fitted_with(Some(&vllm), tail, |_: &str| true),
            "kv fp8_e4m3 · spec 78 %"
        );
    }

    #[test]
    fn engine_items_are_block_and_prefix_only() {
        let mut detail = ModelDetail::default();
        assert!(engine_items(&detail).is_empty());
        detail.kv_block = Some(16);
        detail.prefix_cache = Some(true);
        assert_eq!(engine_items(&detail), ["block 16", "prefix on"]);
        assert_eq!(line(&detail), "kv f16", "not on the shared line");
        detail.prefix_cache = Some(false);
        assert_eq!(engine_items(&detail), ["block 16", "prefix off"]);
        assert!(is_valid(&detail));
        detail.kv_block = Some(0);
        assert!(!is_valid(&detail));
        assert_eq!(engine_items(&detail), ["prefix off"]);
        detail.kv_block = Some(MAX_KV_BLOCK + 1);
        assert!(!is_valid(&detail));
    }

    #[test]
    fn token_allowlist() {
        for good in ["Q4_K_M", "UD-Q4_K_M", "q8_0", "bf16", "a.b+c"] {
            assert!(is_token(good), "{good}");
        }
        for bad in ["", "a b", "a/b", "é", "../x", "x\n", "12345678901234567"] {
            assert!(!is_token(bad), "{bad:?}");
        }
        let mut detail = bonsai();
        assert!(is_valid(&detail));
        detail.quant = Some("/models/x".to_owned());
        assert!(!is_valid(&detail));
        assert!(!line(&detail).contains('/'));
    }
}
