//! #52: the SETUP rows each engine shape gets from the built-in rules plus
//! the commented example block in `packaging/watch.example.toml`, over the
//! invented launch commands in `fixtures/llama/running-setup.json`.

use std::path::PathBuf;

use llama_core::backend::{Backend, BackendInfo, EngineStats};
use llama_core::detail::ModelDetail;
use llama_watch::config::{Config, Setup};
use llama_watch::setup_rules::{LiveCtx, Rules};
use llama_watch::sources::cmdline::parse_launch;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `(id, cmd)` of every fixture entry.
fn fixture() -> Vec<(String, String)> {
    let text =
        std::fs::read_to_string(root().join("fixtures/llama/running-setup.json")).expect("fixture");
    let body: serde_json::Value = serde_json::from_str(&text).expect("json");
    body["running"]
        .as_array()
        .expect("running")
        .iter()
        .map(|entry| {
            (
                entry["model"].as_str().expect("model").to_owned(),
                entry["cmd"].as_str().expect("cmd").to_owned(),
            )
        })
        .collect()
}

fn cmd_of(id: &str) -> String {
    fixture()
        .into_iter()
        .find(|(model, _)| model == id)
        .map(|(_, cmd)| cmd)
        .unwrap_or_else(|| panic!("no fixture {id}"))
}

/// The example block, uncommented, as a watch.toml would hold it.
fn example_setup() -> Setup {
    let text =
        std::fs::read_to_string(root().join("packaging/watch.example.toml")).expect("example");
    let block: String = text
        .split("# >>> setup example\n")
        .nth(1)
        .expect("start marker")
        .split("# <<< setup example")
        .next()
        .expect("end marker")
        .lines()
        .filter(|line| !line.starts_with("# "))
        .map(|line| line.strip_prefix('#').unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n");
    let config = Config::from_toml(&format!("[setup]\n{block}\n")).expect("example block parses");
    assert!(config.validate(8).is_ok(), "example block validates");
    config.setup
}

fn with_example() -> Rules {
    Rules::compile(&example_setup()).expect("compile")
}

fn rows(
    rules: &Rules,
    cmd: &str,
    backend: Backend,
    detail: Option<&ModelDetail>,
    info: Option<&BackendInfo>,
) -> Vec<String> {
    let live = LiveCtx {
        backend,
        detail,
        info,
    };
    rules
        .rows(&rules.extract(cmd), &live)
        .into_iter()
        .map(|row| {
            let mut text = format!("{}:", row.label);
            for (i, item) in row.items.iter().enumerate() {
                if i > 0 {
                    text.push_str(&item.sep);
                } else {
                    text.push(' ');
                }
                text.push_str(&item.text);
                if item.dim {
                    text.push('*');
                }
            }
            text
        })
        .collect()
}

fn engine(spec_len_centi: Option<u16>, spec_permille: Option<u16>) -> BackendInfo {
    BackendInfo {
        engine: EngineStats {
            spec_len_centi,
            spec_permille,
            ..EngineStats::default()
        },
        ..BackendInfo::default()
    }
}

#[test]
fn llama_cpp_moe_with_mtp() {
    let cmd = cmd_of("qwen3.8-35b-a3b-mtp");
    let detail = parse_launch(&cmd).detail;
    assert_eq!(
        rows(
            &with_example(),
            &cmd,
            Backend::LlamaCpp,
            detail.as_ref(),
            None
        ),
        [
            "engine: llama.cpp · UD-Q4_K_M · fa on",
            "ctx: 262,144 · kv q8_0 / q8_0",
            "experts: 24 layers in RAM",
            "spec: draft-mtp · n-max 4",
            "think: budget 24,000 · on",
            "sample: temp 0.6 · top-p 0.95 · top-k 20 · min-p 0",
        ]
    );
}

#[test]
fn llama_cpp_defaults_show_dim() {
    let cmd = "/opt/llama/llama-server -m /m/Dense-27B-Q8_0.gguf -c 32768";
    assert_eq!(
        rows(&Rules::builtin(), cmd, Backend::LlamaCpp, None, None),
        [
            "engine: llama.cpp · Q8_0",
            "ctx: 32,768 · kv f16* / f16*",
            "experts: full GPU*",
            "spec: none*",
        ]
    );
    let cmd = "llama-server --cpu-moe -ncmoe 7 -md /m/draft-Q4_0.gguf --reasoning-budget 0";
    assert_eq!(
        rows(&Rules::builtin(), cmd, Backend::LlamaCpp, None, None)[2..],
        [
            "experts: all layers in RAM",
            "spec: draft model",
            "think: off",
        ]
    );
}

#[test]
fn vllm_serve_flags_and_live_acceptance() {
    let cmd = cmd_of("gemma-4-31b-vllm");
    let launch = parse_launch(&cmd);
    assert_eq!(launch.backend, Backend::Vllm);
    let info = engine(Some(358), Some(648));
    assert_eq!(
        rows(
            &with_example(),
            &cmd,
            Backend::Vllm,
            launch.detail.as_ref(),
            Some(&info)
        ),
        [
            "engine: vLLM",
            "ctx: 73,728 · kv fp8",
            "spec: draft_model · n 4 · acc 3.6/step · 65 %",
            "serve: mem 0.92 · seqs 8 · batch 4,096",
        ]
    );
}

/// The container names no server: until its /metrics says vLLM it is an
/// OpenAI-compatible server, and only the engine row shows.
#[test]
fn env_configured_container_with_the_example_rules() {
    let cmd = cmd_of("qwen3.8-27b-vllm-64k");
    assert_eq!(parse_launch(&cmd).backend, Backend::OpenAi);
    let rules = with_example();
    assert_eq!(
        rows(&rules, &cmd, Backend::OpenAi, None, None),
        ["engine: OpenAI-compatible"]
    );
    // After detection, with cache_config_info's facts and live acceptance.
    let facts = ModelDetail {
        kv_k: Some("fp8_e4m3".to_owned()),
        kv_v: Some("fp8_e4m3".to_owned()),
        kv_block: Some(16),
        prefix_cache: Some(true),
        ..ModelDetail::default()
    };
    let info = engine(Some(412), None);
    assert_eq!(
        rows(&rules, &cmd, Backend::Vllm, Some(&facts), Some(&info)),
        [
            "engine: vLLM · hq-uncensored",
            "ctx: fast · kv fp8_e4m3 · block 16 · prefix on",
            "spec: dflash2 · n 15 · acc 4.1/step",
        ]
    );
    // No cache facts yet: PREFIX_CACHE stands in for the engine's report.
    assert_eq!(
        rows(&rules, &cmd, Backend::Vllm, None, None),
        [
            "engine: vLLM · hq-uncensored",
            "ctx: fast · prefix on",
            "spec: dflash2 · n 15",
        ]
    );
    // The built-in rules alone know nothing of its env.
    assert_eq!(
        rows(&Rules::builtin(), &cmd, Backend::Vllm, Some(&facts), None),
        [
            "engine: vLLM",
            "ctx: kv fp8_e4m3 · block 16 · prefix on",
            "spec: none*",
        ]
    );
    // The env rules are `match`ed to the image: another container's env
    // is not read.
    let other = cmd.replace("hyperqwen", "otherimage");
    assert_eq!(
        rows(&rules, &other, Backend::Vllm, None, None),
        ["engine: vLLM", "spec: none*"]
    );
}

#[test]
fn sglang_in_podman_with_wrapper_env() {
    let cmd = cmd_of("flash-sglang");
    let launch = parse_launch(&cmd);
    assert_eq!(launch.backend, Backend::SgLang);
    assert_eq!(
        rows(
            &with_example(),
            &cmd,
            Backend::SgLang,
            launch.detail.as_ref(),
            None
        ),
        [
            "engine: SGLang · exl3",
            "ctx: 204,800 · kv fp8_e4m3",
            "experts: offload gpu_cache",
            "spec: none*",
            "serve: mem 0.88 · reqs 2",
        ]
    );
}

#[test]
fn strata_reports_its_own_ctx_and_kv() {
    let cmd = cmd_of("bonsai-strata");
    assert_eq!(parse_launch(&cmd).backend, Backend::Strata);
    let facts = ModelDetail {
        ctx: Some(262_144),
        kv_k: Some("q8".to_owned()),
        kv_v: Some("q8".to_owned()),
        ..ModelDetail::default()
    };
    assert_eq!(
        rows(&with_example(), &cmd, Backend::Strata, Some(&facts), None),
        ["engine: Strata", "ctx: 262,144 · kv q8"]
    );
}

/// Whatever a rule names, a path, a long value or a shell fragment never
/// comes out: only numbers, short tokens and file stems.
#[test]
fn hostile_commands_give_only_safe_values() {
    let toml = r#"
[setup]
defaults = false
[[setup.field]]
row = "x"
source = "flag:--a"
kind = "token"
[[setup.field]]
row = "x"
source = "env:SECRET"
kind = "token"
[[setup.field]]
row = "x"
source = "json:--j:k"
kind = "token"
[[setup.field]]
row = "x"
source = "flag:--n"
kind = "number"
"#;
    let config = Config::from_toml(toml).expect("toml");
    let rules = Rules::compile(&config.setup).expect("rules");
    for cmd in [
        "x --a /home/alex/.ssh/id_ed25519-very-long-name --n 1e9",
        "x --a $(rm -rf /) --n 12345678901234567",
        "SECRET=hunter2;curl x --a 'a b' --j '{\"k\": \"/etc/shadow-and-more-text\"}'",
        "x --j '{\"k\": {\"deep\": 1}}' --a",
    ] {
        for found in rules.extract(cmd) {
            assert!(
                !found.value.contains('/') && found.value.len() <= 16,
                "{cmd}: {found:?}"
            );
            assert!(
                found
                    .value
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_.+-".contains(&b)),
                "{cmd}: {found:?}"
            );
        }
    }
    let values: Vec<String> = rules
        .extract("x --a /m/dir/id_ed25519.pub")
        .into_iter()
        .map(|found| found.value)
        .collect();
    assert_eq!(values, ["id_ed25519"], "a path gives its file stem only");
}

#[test]
fn defaults_false_keeps_only_the_listed_rules() {
    let config = Config::from_toml(
        "[setup]\ndefaults = false\n[[setup.field]]\nrow = \"ctx\"\nsource = \"flag:-c\"\nkind = \"number\"\n",
    )
    .expect("toml");
    let rules = Rules::compile(&config.setup).expect("rules");
    assert_eq!(
        rows(
            &rules,
            "llama-server -c 8192 -fa on",
            Backend::LlamaCpp,
            None,
            None
        ),
        ["ctx: 8,192"]
    );
    let none = Config::from_toml("[setup]\ndefaults = false\n").expect("toml");
    assert!(Rules::compile(&none.setup).expect("rules").is_empty());
}

#[test]
fn bad_rules_fail_validation_with_the_field_index() {
    for (toml, want) in [
        (
            "[[setup.field]]\nrow = \"ctx\"\nsource = \"flag:-c\"\nkind = \"integer\"\n",
            "setup.field[0]: unknown kind \"integer\"",
        ),
        (
            "[[setup.field]]\nrow = \"ctx\"\nsource = \"flag:-c\"\nkind = \"number\"\n[[setup.field]]\nrow = \"ctx\"\nsource = \"live:max_model_len\"\n",
            "setup.field[1]: unknown live source",
        ),
        (
            "[[setup.field]]\nrow = \"ctx\"\nsource = \"argv:3\"\nkind = \"number\"\n",
            "setup.field[0]: source",
        ),
    ] {
        let config = Config::from_toml(toml).expect("parses");
        let error = config.validate(8).expect_err("invalid").to_string();
        assert!(error.starts_with(want), "{error}");
    }
    // An unknown key or engine word is a parse error, as everywhere else.
    assert!(Config::from_toml("[setup]\nextend = true\n").is_err());
    assert!(Config::from_toml("[[setup.field]]\nrow = \"x\"\nengines = [\"tabby\"]\n").is_err());
}
