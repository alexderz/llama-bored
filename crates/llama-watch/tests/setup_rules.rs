//! #52: the SETUP rows each engine shape gets from the built-in rules plus
//! the commented example block in `packaging/watch.example.toml`, over the
//! invented launch commands in `fixtures/llama/running-setup.json`.

use std::path::PathBuf;

use llama_core::backend::{Backend, BackendInfo, EngineStats};
use llama_core::detail::ModelDetail;
use llama_watch::config::{Config, Setup};
use llama_watch::metrics::{EngineFacts, parse_strata};
use llama_watch::setup_rules::{Found, LiveCtx, Rules};
use llama_watch::sources::cmdline::{parse_launch, parse_launch_as};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `name` of a fixture entry.
fn name_of(id: &str) -> String {
    let text =
        std::fs::read_to_string(root().join("fixtures/llama/running-setup.json")).expect("fixture");
    let body: serde_json::Value = serde_json::from_str(&text).expect("json");
    body["running"]
        .as_array()
        .expect("running")
        .iter()
        .find(|entry| entry["model"] == id)
        .and_then(|entry| entry["name"].as_str())
        .unwrap_or_else(|| panic!("no fixture {id}"))
        .to_owned()
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
    rows_of(rules, &rules.extract(cmd), backend, detail, info)
}

fn rows_of(
    rules: &Rules,
    found: &[Found],
    backend: Backend,
    detail: Option<&ModelDetail>,
    info: Option<&BackendInfo>,
) -> Vec<String> {
    let live = LiveCtx {
        backend,
        detail,
        info,
        engine: None,
    };
    rules
        .rows(found, &live)
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

/// #67: a llama.cpp image started by digest names no server. Once the
/// config or the `/metrics` probe says llama.cpp, the flags after the
/// image fill the detail and the SETUP rows; the podman options before it
/// lend nothing.
#[test]
fn llama_cpp_container_flags_after_the_image() {
    let id = "invented-lcpp-container";
    let cmd = cmd_of(id);
    assert_eq!(parse_launch(&cmd).backend, Backend::OpenAi);
    let rules = with_example();
    // Not yet known: an OpenAI-compatible server, the engine row only.
    let found = rules.extract_all(&cmd, &name_of(id));
    assert_eq!(
        rows_of(&rules, &found, Backend::OpenAi, None, None),
        ["engine: OpenAI-compatible"]
    );
    let detail = parse_launch_as(&cmd, Some(Backend::LlamaCpp)).detail;
    let found = rules.extract_as(&cmd, &name_of(id), Some(Backend::LlamaCpp));
    assert_eq!(
        rows_of(&rules, &found, Backend::LlamaCpp, detail.as_ref(), None),
        [
            "engine: llama.cpp · IQ4_XS · fa on",
            "ctx: 262,144 · kv f16 / f16",
            "experts: 39 layers in RAM",
            "spec: none*",
            "think: budget 26,000 · on",
            "sample: temp 1.0 · top-p 0.95 · top-k 20 · min-p 0",
        ]
    );
    for value in &found {
        assert!(
            !value.value.contains('/') && !value.value.contains("sha256"),
            "{value:?}"
        );
    }
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
    // Known as vLLM (#67): the env before the image reads the same.
    let known = rules.extract_as(&cmd, "", Some(Backend::Vllm));
    assert_eq!(known, rules.extract(&cmd));
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

/// #54: Strata in a container names nothing in its command; what it
/// reports in `/metrics` (the invented fixture) fills its SETUP rows, and
/// the quant comes from the llama-swap model name.
#[test]
fn strata_in_a_container_shows_what_it_reports() {
    let id = "flash-next-strata";
    let cmd = cmd_of(id);
    assert_eq!(
        parse_launch(&cmd).backend,
        Backend::OpenAi,
        "told by /metrics"
    );
    let body = std::fs::read_to_string(root().join("fixtures/llama/strata-metrics.json"))
        .expect("strata fixture");
    let (_, facts) = parse_strata(&body);
    let EngineFacts {
        ctx, kv, values, ..
    } = facts;
    let detail = ModelDetail {
        ctx,
        kv_k: kv.clone(),
        kv_v: kv,
        ..ModelDetail::default()
    };
    let info = BackendInfo {
        engine: EngineStats {
            spec_permille: Some(700),
            expert_hit_permille: Some(874),
            pcie_share_permille: Some(92),
            ..EngineStats::default()
        },
        ..BackendInfo::default()
    };
    let rules = with_example();
    let live = LiveCtx {
        backend: Backend::Strata,
        detail: Some(&detail),
        info: Some(&info),
        engine: Some(&values),
    };
    let found = rules.extract_all(&cmd, &name_of(id));
    let text: Vec<String> = rules
        .rows(&found, &live)
        .into_iter()
        .map(|row| {
            let items: Vec<String> = row
                .items
                .iter()
                .map(|item| format!("{}{}", item.sep, item.text))
                .collect();
            format!("{}:{}", row.label, items.concat())
        })
        .collect();
    assert_eq!(
        text,
        [
            "engine: · Strata 0.1.41 · Q4_K_M",
            "ctx: · 262,144 · kv q8 · resident 24,576",
            "experts: · cache 14.6 GiB · 7,200 slots · hit 87 % · pcie 9 %",
            "spec: · depth 5 · mtp 3 · lookup 2 · min-p 0.40 · 70 %",
            "serve: · pcie 0.60 · arena 40.5 GiB · 12 workers · conv cache off",
        ]
    );
    // The conversation cache's budget stands in for `off` while it is on.
    let mut on = values.clone();
    on.insert("conversation_cache", "on".to_owned());
    on.insert("conversation_cache_mib", "2560".to_owned());
    let live = LiveCtx {
        engine: Some(&on),
        ..live
    };
    let serve = rules
        .rows(&found, &live)
        .into_iter()
        .find(|row| row.label == "serve")
        .expect("serve");
    let items: Vec<&str> = serve.items.iter().map(|item| item.text.as_str()).collect();
    assert_eq!(
        items,
        [
            "pcie 0.60",
            "arena 40.5 GiB",
            "12 workers",
            "conv cache 2.5 GiB"
        ]
    );
    // Without its report, Strata's rows are what the command gives: none
    // but the engine and the quant from the name.
    let bare = LiveCtx {
        backend: Backend::Strata,
        detail: None,
        info: None,
        engine: None,
    };
    let rows = rules.rows(&found, &bare);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0].items.len(), 2);
}

/// #54: `engine:<key>` names a key of the fixed list, `name` needs kind
/// quant, and `mib` draws GiB.
#[test]
fn engine_and_name_sources_are_checked() {
    for (toml, want) in [
        (
            "[[setup.field]]\nrow = \"x\"\nsource = \"engine:history\"\n",
            "setup.field[0]: unknown engine key \"history\"",
        ),
        (
            "[[setup.field]]\nrow = \"x\"\nsource = \"engine:kv\"\nkind = \"present\"\n",
            "setup.field[0]: kind present needs a flag source",
        ),
        (
            "[[setup.field]]\nrow = \"x\"\nsource = \"engine:kv\"\nkind = \"quant\"\n",
            "setup.field[0]: an engine source takes",
        ),
        (
            "[[setup.field]]\nrow = \"x\"\nsource = \"name\"\n",
            "setup.field[0]: a name source needs kind quant",
        ),
        (
            "[[setup.field]]\nrow = \"x\"\nsource = \"name\"\nkind = \"token\"\n",
            "setup.field[0]: a name source needs kind quant",
        ),
    ] {
        let config = Config::from_toml(toml).expect("parses");
        let error = config.validate(8).expect_err("invalid").to_string();
        assert!(error.starts_with(want), "{error}");
    }
    let config = Config::from_toml(
        "[setup]\ndefaults = false\n[[setup.field]]\nrow = \"mem\"\nsource = \"engine:arena_mib\"\nkind = \"mib\"\n[[setup.field]]\nrow = \"mem\"\nsource = \"engine:expert_cache_mib\"\nkind = \"mib\"\n[[setup.field]]\nrow = \"mem\"\nsource = \"engine:kv\"\nkind = \"number\"\n",
    )
    .expect("toml");
    assert!(config.validate(8).is_ok());
    let rules = Rules::compile(&config.setup).expect("rules");
    let values = [
        ("arena_mib", "512"),
        ("expert_cache_mib", "1048576"),
        ("kv", "q8"),
    ]
    .into_iter()
    .map(|(key, value)| (key, value.to_owned()))
    .collect();
    let live = LiveCtx {
        backend: Backend::Strata,
        detail: None,
        info: None,
        engine: Some(&values),
    };
    let rows = rules.rows(&[], &live);
    let items: Vec<&str> = rows[0]
        .items
        .iter()
        .map(|item| item.text.as_str())
        .collect();
    assert_eq!(items, ["512 MiB", "1,024.0 GiB"], "a token is not a number");
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
