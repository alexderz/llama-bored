//! Launch command to a [`Backend`] and a [`ModelDetail`].
//!
//! The command is untrusted text from llama-swap `/running`. It is read once,
//! inside the `/running` deserialiser, and dropped. Only numbers and tokens
//! that pass [`llama_core::detail::is_token`] leave this module. The model
//! path is reduced to a GGUF quant tag by a strict matcher; a file name that
//! carries no tag gives no quant, and no path segment is ever kept.
//!
//! The server is told by its entry point (T72): a `llama-server` binary under
//! any path, `sglang.launch_server` / `sglang serve`, `vllm serve` /
//! `vllm.entrypoints`, Strata's `serve/server.py` (see [`strata_at`]), else
//! any OpenAI-compatible server. SGLang and vLLM flags are read only after
//! the entry point, so a `podman run ...` wrapper in front cannot lend its
//! own flags.

use llama_core::backend::Backend;
use llama_core::detail::{MAX_TOKEN_CHARS, ModelDetail, NCMOE_ALL, is_token};

/// KV type shown for SGLang and vLLM when no `--kv-cache-dtype` is given.
const KV_AUTO: &str = "auto";

/// What one launch command says about its server.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Launch {
    /// Server kind from the entry point.
    pub backend: Backend,
    /// Tuning detail. `None` for a server whose flags are not read.
    pub detail: Option<ModelDetail>,
    /// `--max-running-requests` (SGLang) or `--max-num-seqs` (vLLM); 1 for
    /// Strata, which serves one request at a time.
    pub max_running: Option<u16>,
}

/// Detect the server, then read its flags.
#[must_use]
pub fn parse_launch(cmd: &str) -> Launch {
    let args: Vec<&str> = cmd.split_whitespace().collect();
    match detect(&args) {
        (Backend::LlamaCpp, _) => Launch {
            backend: Backend::LlamaCpp,
            detail: Some(parse(cmd)),
            max_running: None,
        },
        // Strata's flags name only a JSON config: ctx and KV come from its
        // `/metrics` (`engine.max_context`, `engine.kv`) once it answers.
        (Backend::Strata, _) => Launch {
            backend: Backend::Strata,
            detail: Some(ModelDetail {
                kv_k: Some(KV_AUTO.to_owned()),
                kv_v: Some(KV_AUTO.to_owned()),
                ..ModelDetail::default()
            }),
            max_running: Some(1),
        },
        (Backend::OpenAi, _) => Launch {
            backend: Backend::OpenAi,
            detail: None,
            max_running: None,
        },
        (backend, start) => server_flags(backend, &args[start..]),
    }
}

/// The index of the first argument after the server's entry point, or
/// `None` when the command names no server this module knows (#52: the
/// SETUP rules read server flags after it and a wrapper's env before it).
#[must_use]
pub fn entry_point(args: &[&str]) -> Option<usize> {
    match detect(args) {
        (Backend::OpenAi, _) => None,
        (_, start) => Some(start),
    }
}

/// The server kind and the index of the first argument after its entry point.
fn detect(args: &[&str]) -> (Backend, usize) {
    for (index, arg) in args.iter().enumerate() {
        let base = arg.rsplit('/').next().unwrap_or(arg);
        let next = args.get(index + 1).copied();
        if base.starts_with("llama-server") {
            return (Backend::LlamaCpp, index + 1);
        }
        if *arg == "sglang.launch_server" {
            return (Backend::SgLang, index + 1);
        }
        if base == "sglang" && next == Some("serve") {
            return (Backend::SgLang, index + 2);
        }
        if base == "vllm" && next == Some("serve") {
            return (Backend::Vllm, index + 2);
        }
        if arg.starts_with("vllm.entrypoints") {
            return (Backend::Vllm, index + 1);
        }
        if strata_at(args, index) {
            return (Backend::Strata, index + 1);
        }
    }
    (Backend::OpenAi, args.len())
}

/// Strata's server script: `serve/server.py`, bare or under a directory.
fn is_strata_script(arg: &str) -> bool {
    arg == "serve/server.py" || arg.ends_with("/serve/server.py")
}

/// `args[index]` is Strata's `serve/server.py` and either
/// - a later `--engine strata` (or `--engine=strata`) picks its engine, or
/// - it runs as `<image> python… serve/server.py` where the image's last
///   path segment starts with `strata` (`localhost/strata:v0.1.27-sm86`).
///
/// A `serve/server.py` of some other project, or `strata` elsewhere in a
/// path, is not enough.
fn strata_at(args: &[&str], index: usize) -> bool {
    if !args.get(index).is_some_and(|arg| is_strata_script(arg)) {
        return false;
    }
    let rest = &args[index + 1..];
    let engine = rest.iter().enumerate().any(|(at, arg)| {
        *arg == "--engine=strata" || (*arg == "--engine" && rest.get(at + 1) == Some(&"strata"))
    });
    if engine {
        return true;
    }
    let Some(before) = index.checked_sub(2) else {
        return false;
    };
    let python = args[index - 1]
        .rsplit('/')
        .next()
        .is_some_and(|base| base.starts_with("python"));
    let image = args[before].rsplit('/').next().unwrap_or_default();
    python && !args[before].starts_with('-') && image.starts_with("strata")
}

/// SGLang or vLLM flags. KV is `auto` unless a dtype is given.
fn server_flags(backend: Backend, args: &[&str]) -> Launch {
    let (ctx_flag, running_flag) = if backend == Backend::Vllm {
        ("--max-model-len", "--max-num-seqs")
    } else {
        ("--context-length", "--max-running-requests")
    };
    let mut detail = ModelDetail::default();
    let mut kv = None;
    let mut max_running = None;
    let mut args = args.iter().copied();
    while let Some(arg) = args.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((flag, value)) if flag.starts_with("--") => (flag, Some(value)),
            _ => (arg, None),
        };
        let wanted = flag == ctx_flag
            || flag == running_flag
            || flag == "--kv-cache-dtype"
            || flag == "--quantization";
        if !wanted {
            continue;
        }
        let Some(value) = inline.or_else(|| args.next()) else {
            break;
        };
        if flag == ctx_flag {
            detail.ctx = value.parse::<u32>().ok().filter(|ctx| *ctx > 0);
        } else if flag == running_flag {
            max_running = value.parse::<u16>().ok().filter(|n| *n > 0);
        } else if flag == "--kv-cache-dtype" {
            kv = kv_token(value);
        } else {
            detail.quant = is_token(value).then(|| value.to_owned());
        }
    }
    let kv = kv.unwrap_or_else(|| KV_AUTO.to_owned());
    detail.kv_k = Some(kv.clone());
    detail.kv_v = Some(kv);
    Launch {
        backend,
        detail: Some(detail),
        max_running,
    }
}

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
                kv_block: None,
                prefix_cache: None,
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

    /// A podman-wrapped SGLang launch, names made generic.
    const SGLANG_PODMAN: &str = "podman run --name flash --rm --network llama --shm-size 64g --device nvidia.com/gpu=all -v /models/x:/models/x:ro --entrypoint /opt/entrypoint.sh -e HF_HUB_OFFLINE=1 ghcr.io/example/sglang-exl3@sha256:abc python3 -m sglang.launch_server --model-path /models/x --quantization exl3 --trust-remote-code --host 0.0.0.0 --port 30100 --served-model-name flash --kv-cache-dtype fp8_e4m3 --context-length 204800 --mem-fraction-static 0.88 --max-running-requests 4 --max-total-tokens 210000";

    #[test]
    fn backend_from_the_entry_point() {
        let kind = |cmd: &str| parse_launch(cmd).backend;
        assert_eq!(kind(BONSAI), Backend::LlamaCpp);
        assert_eq!(
            kind("/opt/ik_llama/bin/llama-server -m /m/x-Q4_K_M.gguf"),
            Backend::LlamaCpp
        );
        assert_eq!(kind("llama-server --port 1"), Backend::LlamaCpp);
        assert_eq!(
            kind("/opt/bin/llama-server-cuda --port 1"),
            Backend::LlamaCpp
        );
        assert_eq!(kind(SGLANG_PODMAN), Backend::SgLang);
        assert_eq!(kind("sglang serve --model-path /m/x"), Backend::SgLang);
        assert_eq!(kind("/venv/bin/vllm serve /m/x --port 1"), Backend::Vllm);
        assert_eq!(
            kind("python -m vllm.entrypoints.openai.api_server --model /m/x"),
            Backend::Vllm
        );
        assert_eq!(
            kind("python3 main.py --config /tabby/config.yml"),
            Backend::OpenAi
        );
        assert_eq!(kind("sglang-exl3 --port 1"), Backend::OpenAi);
        assert_eq!(kind(""), Backend::OpenAi);
        assert_eq!(parse_launch("python3 main.py").detail, None);
    }

    /// The real-shaped llama-swap Strata command (podman wrapper, generic names).
    const STRATA_PODMAN: &str = "podman run --rm --name flash-strata --network llama --device nvidia.com/gpu=all -v /models/strata:/data:ro localhost/strata:v0.1.27-sm86 python serve/server.py --engine strata --config /data/strata.json --port 8095";

    #[test]
    fn strata_is_told_by_its_script_and_engine_or_image() {
        let kind = |cmd: &str| parse_launch(cmd).backend;
        let launch = parse_launch(STRATA_PODMAN);
        assert_eq!(launch.backend, Backend::Strata);
        assert_eq!(launch.max_running, Some(1));
        let detail = launch.detail.expect("strata detail");
        assert_eq!(detail.ctx, None, "ctx comes from /metrics");
        assert_eq!(detail.kv_k.as_deref(), Some(KV_AUTO));
        assert_eq!(detail.quant, None);
        let debug = format!("{detail:?}");
        assert!(
            !debug.contains("/data") && !debug.contains("8095"),
            "{debug}"
        );

        for cmd in [
            "python serve/server.py --engine strata",
            "python3 /opt/strata/serve/server.py --config c.json --engine=strata",
            "/venv/bin/python serve/server.py --port 1 --engine strata --config x",
            // The image alone is enough when it runs python serve/server.py.
            "podman run --rm localhost/strata:v0.1.27-sm86 python serve/server.py --config /data/strata.json",
            "docker run ghcr.io/example/strata-sm86@sha256:abc python3 serve/server.py",
        ] {
            assert_eq!(kind(cmd), Backend::Strata, "{cmd}");
        }
    }

    #[test]
    fn strata_in_a_path_is_not_strata() {
        let kind = |cmd: &str| parse_launch(cmd).backend;
        // A llama-server whose paths say strata stays llama.cpp.
        assert_eq!(
            kind("/opt/strata/llama-server -m /models/strata/x-Q4_K_M.gguf --port 8095"),
            Backend::LlamaCpp
        );
        assert_eq!(
            kind("podman run -v /models/strata:/m img llama-server -m /m/x.gguf"),
            Backend::LlamaCpp
        );
        for cmd in [
            // Another project's serve/server.py.
            "python serve/server.py --port 8095",
            "python serve/server.py --engine mock",
            // strata only in a volume or a config path.
            "podman run -v /data/strata:/data img python serve/server.py --config /data/strata.json",
            "python /srv/strata/app.py --engine strata",
            "python serve/server.py.bak --engine strata",
            "python notserve/server.py --engine strata",
            // --engine strata before the script belongs to the wrapper.
            "run --engine strata --rm python serve/server.py",
            // The image must sit right before python.
            "podman run localhost/strata:v1 sh -c python serve/server.py",
            "strata",
        ] {
            assert_eq!(kind(cmd), Backend::OpenAi, "{cmd}");
        }
    }

    #[test]
    fn sglang_flags_after_a_podman_wrapper() {
        let launch = parse_launch(SGLANG_PODMAN);
        assert_eq!(launch.max_running, Some(4));
        let detail = launch.detail.expect("sglang detail");
        assert_eq!(
            detail,
            ModelDetail {
                ctx: Some(204_800),
                ncmoe: None,
                kv_k: Some("fp8_e4m3".to_owned()),
                kv_v: Some("fp8_e4m3".to_owned()),
                quant: Some("exl3".to_owned()),
                fa: None,
                kv_block: None,
                prefix_cache: None,
            }
        );
        let debug = format!("{detail:?}");
        assert!(!debug.contains("/models"), "{debug}");
        assert!(!debug.contains("30100"), "{debug}");
        // A wrapper flag before the entry point is not read as a server flag.
        let wrapped = parse_launch(
            "podman run --context-length 7 img python3 -m sglang.launch_server --model-path /m",
        );
        assert_eq!(wrapped.detail.expect("detail").ctx, None);
    }

    #[test]
    fn vllm_flags_and_defaults() {
        let launch = parse_launch(
            "vllm serve /m/x --max-model-len=32768 --quantization awq --max-num-seqs 8 --context-length 5",
        );
        assert_eq!(launch.backend, Backend::Vllm);
        assert_eq!(launch.max_running, Some(8));
        let detail = launch.detail.expect("detail");
        assert_eq!(detail.ctx, Some(32_768));
        assert_eq!(detail.quant.as_deref(), Some("awq"));
        assert_eq!(detail.kv_k.as_deref(), Some("auto"));
        assert_eq!(detail.kv_v.as_deref(), Some("auto"));
        let bad = parse_launch(
            "vllm serve /m --max-model-len -1 --quantization ../x --max-num-seqs 0 --kv-cache-dtype a/b",
        );
        let detail = bad.detail.expect("detail");
        assert_eq!(detail.ctx, None);
        assert_eq!(detail.quant, None);
        assert_eq!(detail.kv_k.as_deref(), Some("auto"));
        assert_eq!(bad.max_running, None);
        assert_eq!(
            parse_launch("vllm serve /m --max-model-len")
                .detail
                .expect("d")
                .ctx,
            None
        );
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
