//! llama-server launch command to a [`ModelDetail`].
//!
//! The command is untrusted text from llama-swap `/running`. It is read once,
//! inside the `/running` deserialiser, and dropped. Only numbers and tokens
//! that pass [`llama_core::detail::is_token`] leave this module. The model
//! path is reduced to a GGUF quant tag by a strict matcher; a file name that
//! carries no tag gives no quant, and no path segment is ever kept.

use llama_core::detail::{MAX_TOKEN_CHARS, ModelDetail, NCMOE_ALL, is_token};

/// Parse the flags this display cares about. Later flags win, as in llama.cpp.
#[must_use]
pub fn parse(cmd: &str) -> ModelDetail {
    let mut detail = ModelDetail::default();
    let mut args = cmd.split_whitespace().peekable();
    while let Some(arg) = args.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((flag, value)) if flag.starts_with("--") => (flag, Some(value)),
            _ => (arg, None),
        };
        match flag {
            "-c" | "--ctx-size" => {
                if let Some(value) = inline.or_else(|| args.next()) {
                    detail.ctx = value.parse::<u32>().ok().filter(|ctx| *ctx > 0);
                }
            }
            "-ncmoe" | "--n-cpu-moe" => {
                if let Some(value) = inline.or_else(|| args.next()) {
                    detail.ncmoe = value
                        .parse::<u32>()
                        .ok()
                        .map(|n| u16::try_from(n).unwrap_or(NCMOE_ALL));
                }
            }
            "-cmoe" | "--cpu-moe" => detail.ncmoe = Some(NCMOE_ALL),
            "-ctk" | "--cache-type-k" => {
                if let Some(value) = inline.or_else(|| args.next()) {
                    detail.kv_k = kv_token(value);
                }
            }
            "-ctv" | "--cache-type-v" => {
                if let Some(value) = inline.or_else(|| args.next()) {
                    detail.kv_v = kv_token(value);
                }
            }
            "-m" | "--model" => {
                if let Some(value) = inline.or_else(|| args.next()) {
                    detail.quant = quant_tag(value);
                }
            }
            "-fa" | "--flash-attn" => {
                let value = match inline {
                    Some(value) => Some(value),
                    None => match args.peek().copied().and_then(fa_word) {
                        Some(_) => args.next(),
                        None => None,
                    },
                };
                detail.fa = match value {
                    Some(word) => fa_word(word).flatten(),
                    None => Some(true),
                };
            }
            _ => {}
        }
    }
    detail
}

/// `Some(Some(on))` for an on/off word, `Some(None)` for `auto`, else `None`.
fn fa_word(word: &str) -> Option<Option<bool>> {
    match word.to_ascii_lowercase().as_str() {
        "on" | "true" | "1" | "enabled" => Some(Some(true)),
        "off" | "false" | "0" | "disabled" => Some(Some(false)),
        "auto" => Some(None),
        _ => None,
    }
}

fn kv_token(value: &str) -> Option<String> {
    let lower = value.to_ascii_lowercase();
    is_token(&lower).then_some(lower)
}

/// GGUF quant tag from a model path, e.g. `Q4_K_M`, `UD-Q4_K_M`, `PTQ1_0`.
///
/// Only the file name is read: `.gguf` and a `-00001-of-00003` shard suffix
/// are removed, the stem is split on `-` and `.`, and the last piece that is
/// a quant tag wins. An `UD` piece right before it is kept as `UD-`.
#[must_use]
pub fn quant_tag(path: &str) -> Option<String> {
    let file = path.rsplit('/').next().unwrap_or(path);
    let lower = file.to_ascii_lowercase();
    let stem = &file[..lower.strip_suffix(".gguf")?.len()];
    let stem = strip_shard(stem);
    let pieces: Vec<&str> = stem.split(['-', '.']).collect();
    for index in (0..pieces.len()).rev() {
        let piece = pieces[index].to_ascii_uppercase();
        if !is_quant(&piece) {
            continue;
        }
        let tag = if index > 0 && pieces[index - 1].eq_ignore_ascii_case("ud") {
            format!("UD-{piece}")
        } else {
            piece
        };
        return (tag.len() <= MAX_TOKEN_CHARS && is_token(&tag)).then_some(tag);
    }
    None
}

/// Drop a trailing `-NNNNN-of-NNNNN`.
fn strip_shard(stem: &str) -> &str {
    let bytes = stem.as_bytes();
    // "-00001-of-00003" is 15 bytes.
    if bytes.len() > 15 {
        let tail = &stem[stem.len() - 15..];
        let t = tail.as_bytes();
        let digits = |range: std::ops::Range<usize>| t[range].iter().all(u8::is_ascii_digit);
        if t[0] == b'-' && digits(1..6) && &tail[6..10] == "-of-" && digits(10..15) {
            return &stem[..stem.len() - 15];
        }
    }
    stem
}

/// `F16`, `BF16`, `F32`, `MXFP4`, or `[I|T|P|PT]Q<1-8>_<suffix>`.
fn is_quant(piece: &str) -> bool {
    if matches!(piece, "F16" | "BF16" | "F32" | "MXFP4" | "MXFP4_MOE") {
        return true;
    }
    let rest = ["PT", "I", "T", "P", ""]
        .iter()
        .find_map(|prefix| piece.strip_prefix(prefix)?.strip_prefix('Q'));
    let Some(rest) = rest else {
        return false;
    };
    let mut chars = rest.chars();
    if !matches!(chars.next(), Some('1'..='8')) {
        return false;
    }
    let Some(suffix) = chars.as_str().strip_prefix('_') else {
        return false;
    };
    matches!(
        suffix,
        "0" | "1"
            | "K"
            | "K_S"
            | "K_M"
            | "K_L"
            | "K_XL"
            | "XXS"
            | "XS"
            | "S"
            | "M"
            | "L"
            | "XL"
            | "NL"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const BONSAI: &str = "/models/prism/llama-server --host 127.0.0.1 --port 5800 -fa on --metrics\n-ctk q8_0 -ctv q8_0\n-m /models/llm/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf\n-ngl 999 -c 262144\n";

    #[test]
    fn real_bonsai_command() {
        let detail = parse(BONSAI);
        assert_eq!(
            detail,
            ModelDetail {
                ctx: Some(262_144),
                ncmoe: None,
                kv_k: Some("q8_0".to_owned()),
                kv_v: Some("q8_0".to_owned()),
                quant: Some("PTQ1_0".to_owned()),
                fa: Some(true),
            }
        );
        let debug = format!("{detail:?}");
        assert!(!debug.contains("/models"), "{debug}");
        assert!(!debug.contains("5800"), "{debug}");
    }

    #[test]
    fn long_flags_moe_and_defaults() {
        let detail = parse(
            "llama-server --ctx-size=32768 --n-cpu-moe 16 --model /m/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --flash-attn",
        );
        assert_eq!(detail.ctx, Some(32_768));
        assert_eq!(detail.ncmoe, Some(16));
        assert_eq!(detail.quant.as_deref(), Some("UD-Q4_K_M"));
        assert_eq!(detail.fa, Some(true));
        assert_eq!(detail.kv_k, None);
        assert_eq!(parse("x --cpu-moe").ncmoe, Some(NCMOE_ALL));
        assert_eq!(parse("x -ncmoe 99999").ncmoe, Some(NCMOE_ALL));
        assert_eq!(parse("x -fa off").fa, Some(false));
        assert_eq!(parse("x -fa auto").fa, None);
        assert_eq!(parse("x -c 0").ctx, None);
        assert_eq!(parse("x -c 1 -c 2").ctx, Some(2));
        assert_eq!(parse("x -c").ctx, None);
        assert_eq!(parse("").ctx, None);
    }

    #[test]
    fn hostile_values_are_dropped() {
        let detail = parse("x -ctk ../../etc -ctv q4_0;rm -c -5 -m /a/b/evil.bin");
        assert_eq!(detail.kv_k, None);
        assert_eq!(detail.kv_v, None);
        assert_eq!(detail.ctx, None);
        assert_eq!(detail.quant, None);
        assert_eq!(parse("x -ctk AAAAAAAAAAAAAAAAAAAAAAAA").kv_k, None);
    }

    #[test]
    fn quant_tags() {
        let cases = [
            ("/m/Ternary-Bonsai-2-27B-PTQ1_0.gguf", Some("PTQ1_0")),
            ("/m/Qwen3-8B-Q4_K_M.gguf", Some("Q4_K_M")),
            ("/m/qwen3-8b-q8_0.gguf", Some("Q8_0")),
            ("/m/model.IQ4_XS.gguf", Some("IQ4_XS")),
            ("/m/big-UD-Q2_K_XL-00001-of-00003.gguf", Some("UD-Q2_K_XL")),
            ("/m/gpt-oss-20b-MXFP4.gguf", Some("MXFP4")),
            ("/m/x-BF16.gguf", Some("BF16")),
            ("/m/x-TQ2_0.gguf", Some("TQ2_0")),
            ("/Q4_K_M/model.gguf", None),
            ("/m/x-Q4.gguf", None),
            ("/m/x-Q9_0.gguf", None),
            ("/m/x-Q4_K_M.bin", None),
            ("/m/x-Q4_K_MM.gguf", None),
            ("/m/secret-name.gguf", None),
        ];
        for (path, expect) in cases {
            assert_eq!(quant_tag(path).as_deref(), expect, "{path}");
        }
    }
}
