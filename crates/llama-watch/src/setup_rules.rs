//! tty11's SETUP block (#52): `[setup]` rules pick settings out of a
//! model's launch command and its engine's own reports.
//!
//! Which flags matter is configuration: the built-in rules are
//! `setup_defaults.toml` (upstream llama.cpp, vLLM and SGLang flags, and
//! Strata's own report), and
//! `[[setup.field]]` in watch.toml adds more. This module is the plumbing,
//! and it holds the limits whatever a rule says:
//!
//! - The command is read once, while llama-swap's `/running` body is
//!   decoded, and dropped. [`Rules::extract`] keeps only a number
//!   ([`MAX_NUMBER_CHARS`]), a short token ([`is_token`]: 16 characters of
//!   `[A-Za-z0-9_.+-]`, the file stem of a path-like value), a GGUF quant
//!   tag, or the word `on` for a present flag. Never a path or a raw
//!   argument.
//! - Rule text (row, label, suffix, map, default) is checked at load:
//!   printable ASCII or `·`, short caps.
//! - A JSON-valued flag is read one level deep and only up to
//!   [`MAX_JSON_BYTES`].
//!
//! - `engine:<key>` (#54) reads a setting the engine reports about
//!   itself. The keys are [`ENGINE_KEYS`], a fixed list the engine parser
//!   fills with numbers and short tokens; no other part of its JSON is
//!   reachable.
//! - `name` (#54) scans the llama-swap model name for a GGUF quant tag
//!   ([`cmdline::name_quant`]), for an engine whose command names no
//!   weights file.
//! - `spec:<dotted.path>` (#92) reads one value from the model's entry
//!   spec, a JSON file the poller already loaded. A missing path, null,
//!   object or array leaves the field blank. The file itself is not opened
//!   here.
//! - `kind = "text"` (#114) is valid only with a `spec:` source. It shows
//!   a JSON string: whitespace runs collapse to one space, controls are
//!   dropped, and the value is cut to `max_chars` (default
//!   [`DEFAULT_TEXT_CHARS`], at most [`MAX_TEXT_VALUE_CHARS`]). A number,
//!   boolean, null, object or array shows nothing. The launch command, the
//!   environment and the engine stay limited to numbers and short tokens,
//!   so a path or a secret on a command line still cannot reach the screen.
//!
//! [`Rules::rows`] turns the extracted values plus the live engine numbers
//! into the rows the layout draws.

use std::collections::BTreeMap;

use llama_core::backend::{Backend, BackendInfo};
use llama_core::detail::{ModelDetail, is_token};

use crate::config::{Setup, SetupField};
use crate::metrics::{ENGINE_KEYS, EngineValues};
use crate::sources::cmdline;
use crate::tty::layout::{SetupItem, SetupRow};

/// The built-in rules, in `[[setup.field]]` form.
pub const DEFAULTS: &str = include_str!("setup_defaults.toml");

/// Most rules, built-in and configured together.
pub const MAX_FIELDS: usize = 128;
/// Longest row name.
pub const MAX_ROW_CHARS: usize = 8;
/// Longest label, suffix, map value or default.
pub const MAX_TEXT_CHARS: usize = 32;
/// Longest separator.
pub const MAX_SEP_CHARS: usize = 4;
/// Longest `match` text.
pub const MAX_MATCH_CHARS: usize = 64;
/// Longest flag, env or JSON key name.
pub const MAX_NAME_CHARS: usize = 48;
/// Most flag names in one source.
pub const MAX_FLAGS: usize = 8;
/// Most `map` entries in one rule.
pub const MAX_MAP: usize = 16;
/// Longest JSON object read after a JSON-valued flag.
pub const MAX_JSON_BYTES: usize = 4096;
/// Longest number kept from a command, sign and point included.
pub const MAX_NUMBER_CHARS: usize = 16;
/// Most dot-separated segments in a `spec:` path.
pub const MAX_SPEC_SEGMENTS: usize = 8;
/// Longest one segment of a `spec:` path.
pub const MAX_SPEC_SEGMENT: usize = 32;
/// Longest spec string handed to [`clean`]. A longer string yields nothing,
/// rather than a file stem taken from the tail of an untrusted value. A
/// `text` label of about 130 characters is under this cap and is cut to
/// [`MAX_TEXT_VALUE_CHARS`], not dropped.
pub const MAX_SPEC_VALUE_CHARS: usize = 1024;
/// Characters kept from a `text` value when the rule sets no `max_chars` (#114).
pub const DEFAULT_TEXT_CHARS: usize = 64;
/// Most characters a `text` rule may keep. Past this the value is cut.
pub const MAX_TEXT_VALUE_CHARS: usize = 96;

/// Separator between items on one row unless a rule names its own.
const SEP: &str = " \u{00B7} ";

/// How a command value is read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    /// `-?digits(.digits)?`, drawn with thousands separators.
    Number,
    /// A short allowlisted token; a path-like value gives its file stem.
    Token,
    /// A GGUF quant tag from a model file name (`Q4_K_M`).
    Quant,
    /// The flag is there: the value is `on`.
    Present,
    /// A number of MiB, drawn as `16.4 GiB` (or `512 MiB` below 1 GiB).
    Mib,
    /// A JSON string from a `spec:` source (#114), cut to `max_chars`.
    Text,
}

/// A number the watcher already has from the engine or the launch command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Live {
    /// The engine's display name (#33).
    Engine,
    /// Context size: the launch command's, else the engine's own report.
    Ctx,
    /// KV cache dtype: the command's, else the engine's report.
    KvDtype,
    /// vLLM KV block size.
    KvBlock,
    /// vLLM prefix caching, `on` or `off`.
    PrefixCache,
    /// Speculative acceptance, `78 %`.
    SpecAccept,
    /// Mean tokens per speculative step, `3.6/step`.
    SpecLen,
    /// Expert cache hit rate of the newest request, `87 %` (#54).
    ExpertHit,
    /// PCIe share of that request's expert reads, `9 %` (#54).
    PcieShare,
}

impl Live {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "engine" => Self::Engine,
            "ctx" => Self::Ctx,
            "kv_dtype" => Self::KvDtype,
            "kv_block" => Self::KvBlock,
            "prefix_cache" => Self::PrefixCache,
            "spec_accept" => Self::SpecAccept,
            "spec_len" => Self::SpecLen,
            "expert_hit" => Self::ExpertHit,
            "pcie_share" => Self::PcieShare,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Source {
    /// No source: only the rule's `default` is ever drawn.
    Fixed,
    /// `--flag V`, `--flag=V`, or a short flag.
    Flag(Vec<String>),
    /// `-e NAME=V`, `--env NAME=V`, `--env=NAME=V` or `NAME=V`.
    Env(String),
    /// A key, one level deep, of a JSON object given to a flag.
    Json { flags: Vec<String>, key: String },
    /// A number the watcher already has.
    Live(Live),
    /// A setting the engine reports, one of [`ENGINE_KEYS`] (#54).
    Engine(&'static str),
    /// A quant tag in the llama-swap model name (#54).
    Name,
    /// A dotted path into the model's entry spec (#92).
    Spec(Vec<String>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Rule {
    row: String,
    engines: Vec<Backend>,
    /// Lowercase `match` text.
    matches: Option<String>,
    source: Source,
    kind: Kind,
    /// Characters kept from a [`Kind::Text`] value. Unused for every other kind.
    max_chars: usize,
    label: String,
    suffix: String,
    sep: String,
    map: BTreeMap<String, String>,
    default: Option<String>,
    fallback: bool,
    group: Option<String>,
    order: u16,
}

/// One value a rule took from a launch command: the rule's index in
/// [`Rules`] and the cleaned value text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Found {
    /// Index of the rule in its [`Rules`].
    pub rule: u16,
    /// A number, a token, a quant tag, `on`, or a sanitised `text` string.
    /// Never a path. A `text` string is not yet cut to `max_chars`: [`Rules::rows`]
    /// does that after `map` has had a chance to match the whole value.
    pub value: String,
}

/// What [`Rules::rows`] reads besides the command values.
#[derive(Clone, Copy, Debug)]
pub struct LiveCtx<'a> {
    /// The model's engine, after detection.
    pub backend: Backend,
    /// The model's tuning detail, with what the engine reported.
    pub detail: Option<&'a ModelDetail>,
    /// The engine's live gauges.
    pub info: Option<&'a BackendInfo>,
    /// Settings the engine reports about itself, for `engine:<key>` (#54).
    pub engine: Option<&'a EngineValues>,
}

/// Compiled `[setup]` rules, sorted by order (stable: built-in rules first,
/// then watch.toml's in file order).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Rules {
    rules: Vec<Rule>,
}

impl Rules {
    /// The built-in rules alone.
    #[must_use]
    pub fn builtin() -> Self {
        Self::compile(&Setup::default()).unwrap_or_default()
    }

    /// Check and compile `[setup]`. `Err` is the index of the bad
    /// `[[setup.field]]` and why it is bad.
    pub fn compile(setup: &Setup) -> Result<Self, (usize, String)> {
        let mut rules = Vec::new();
        if setup.defaults {
            let builtin: Setup = toml::from_str(DEFAULTS)
                .map_err(|_| (0, "built-in rules do not parse".to_owned()))?;
            for (index, field) in builtin.field.iter().enumerate() {
                rules.push(
                    compile_field(field)
                        .map_err(|reason| (index, format!("built-in rule: {reason}")))?,
                );
            }
        }
        for (index, field) in setup.field.iter().enumerate() {
            rules.push(compile_field(field).map_err(|reason| (index, reason))?);
        }
        if rules.len() > MAX_FIELDS {
            return Err((
                setup.field.len().saturating_sub(1),
                format!("more than {MAX_FIELDS} rules with the built-in ones"),
            ));
        }
        rules.sort_by_key(|rule| rule.order);
        Ok(Self { rules })
    }

    /// True when there are no rules at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Every command value the rules name. `cmd` is the raw launch command;
    /// nothing of it is kept but the cleaned values.
    #[must_use]
    pub fn extract(&self, cmd: &str) -> Vec<Found> {
        self.extract_all(cmd, "")
    }

    /// [`Self::extract`], plus the `name` rules' quant tag from the
    /// llama-swap model `name` (#54). Of the name only the tag is kept.
    #[must_use]
    pub fn extract_all(&self, cmd: &str, name: &str) -> Vec<Found> {
        self.extract_as(cmd, name, None)
    }

    /// [`Self::extract_all`] with the server `known` from `[llama.backends]`
    /// or the `/metrics` probe (#67): a container that names no server has
    /// its server flags after the image and its env before it.
    #[must_use]
    pub fn extract_as(&self, cmd: &str, name: &str, known: Option<Backend>) -> Vec<Found> {
        self.extract_with(cmd, name, known, None)
    }

    /// [`Self::extract_as`] plus the model's entry spec (#92). `None`, and
    /// a path that misses, leaves those rules with no value. The document
    /// is not kept.
    #[must_use]
    pub fn extract_with(
        &self,
        cmd: &str,
        name: &str,
        known: Option<Backend>,
        spec: Option<&serde_json::Value>,
    ) -> Vec<Found> {
        let tokens = tokenize(cmd);
        let words: Vec<&str> = tokens.iter().map(|(_, word)| *word).collect();
        // A server's flags come after its entry point (or its container
        // image, #67) and a wrapper's env before it, so `podman run -e`
        // cannot lend a server flag and a llama-server `-e` (escapes) is
        // never read as env. With neither the whole command is both.
        let (flags_from, env_to) = match cmdline::flags_start(&words, known) {
            Some(at) => (at, at),
            None => (0, words.len()),
        };
        let lower = cmd.to_ascii_lowercase();
        let mut out = Vec::new();
        for (index, rule) in self.rules.iter().enumerate() {
            if rule
                .matches
                .as_deref()
                .is_some_and(|text| !lower.contains(text))
            {
                continue;
            }
            let value = match &rule.source {
                Source::Flag(names) => flag_value(&words[flags_from..], names, rule.kind),
                Source::Env(name) => env_value(&words[..env_to], name),
                Source::Json { flags, key } => json_value(cmd, &tokens[flags_from..], flags, key),
                Source::Spec(path) => spec.and_then(|doc| spec_at(doc, path, rule.kind)),
                // Already a checked tag, not a file name to read one from.
                Source::Name => {
                    if let Some(tag) = cmdline::name_quant(name) {
                        push(&mut out, index, tag);
                    }
                    continue;
                }
                Source::Fixed | Source::Live(_) | Source::Engine(_) => continue,
            }
            .and_then(|raw| clean(rule.kind, &raw));
            if let Some(value) = value {
                push(&mut out, index, value);
            }
        }
        out
    }

    /// The SETUP rows for one model, in order, without empty rows.
    #[must_use]
    pub fn rows(&self, found: &[Found], live: &LiveCtx<'_>) -> Vec<SetupRow> {
        let mut rows: Vec<SetupRow> = Vec::new();
        // The group of each item drawn so far, by row.
        let mut groups: Vec<Vec<Option<&str>>> = Vec::new();
        for (index, rule) in self.rules.iter().enumerate() {
            let at = match rows.iter().position(|row| row.label == rule.row) {
                Some(at) => at,
                None => {
                    rows.push(SetupRow {
                        label: rule.row.clone(),
                        items: Vec::new(),
                    });
                    groups.push(Vec::new());
                    rows.len() - 1
                }
            };
            if !rule.engines.is_empty() && !rule.engines.contains(&live.backend) {
                continue;
            }
            let group = rule.group.as_deref();
            let covered = match group {
                Some(group) => groups[at].contains(&Some(group)),
                None => !groups[at].is_empty(),
            };
            if rule.fallback && covered {
                continue;
            }
            let value = match &rule.source {
                Source::Live(which) => live_value(*which, live),
                Source::Engine(key) => live
                    .engine
                    .and_then(|values| values.get(key))
                    .and_then(|value| engine_value(rule.kind, value)),
                Source::Fixed => None,
                _ => found
                    .iter()
                    .rev()
                    .find(|found| usize::from(found.rule) == index)
                    .map(|found| found.value.clone()),
            };
            // `map` matches the sanitised value before the `max_chars` cut,
            // so a short label still maps when a longer one would be cut.
            // `label` and `suffix` wrap the cut value, as they do for a token.
            let (text, dim) = match value {
                Some(value) => match rule.map.get(&value) {
                    Some(text) => (text.clone(), false),
                    None => {
                        let shown = match rule.kind {
                            Kind::Number if is_number(&value) => number_text(&value),
                            Kind::Mib if is_number(&value) => mib_text(&value),
                            Kind::Text => crate::tty::layout::cut_text(&value, rule.max_chars),
                            _ => value,
                        };
                        let text = if rule.label.is_empty() {
                            format!("{shown}{}", rule.suffix)
                        } else {
                            format!("{} {shown}{}", rule.label, rule.suffix)
                        };
                        (text, false)
                    }
                },
                None => match &rule.default {
                    Some(text) => (text.clone(), true),
                    None => continue,
                },
            };
            rows[at].items.push(SetupItem {
                sep: rule.sep.clone(),
                text,
                dim,
                // A dim default is a short configured string, not the value.
                text_value: rule.kind == Kind::Text && !dim,
            });
            groups[at].push(group);
        }
        rows.retain(|row| !row.items.is_empty());
        rows
    }
}

/// Record rule `index`'s value. More than `u16::MAX` rules never compile.
fn push(out: &mut Vec<Found>, index: usize, value: String) {
    if let Ok(rule) = u16::try_from(index) {
        out.push(Found { rule, value });
    }
}

fn compile_field(field: &SetupField) -> Result<Rule, String> {
    let row = &field.row;
    if row.is_empty()
        || row.chars().count() > MAX_ROW_CHARS
        || !row
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
    {
        return Err(format!(
            "row must be 1..={MAX_ROW_CHARS} characters of A-Z a-z 0-9 _ - ."
        ));
    }
    let matches = match &field.matches {
        Some(text) => {
            text_ok("match", text, MAX_MATCH_CHARS)?;
            if text.is_empty() {
                return Err("match must not be empty".to_owned());
            }
            Some(text.to_ascii_lowercase())
        }
        None => None,
    };
    let source = parse_source(field.source.as_deref())?;
    // Before the source-specific kind errors, so every non-spec source
    // names this rule the same way.
    if field.kind.as_deref() == Some("text") && !matches!(source, Source::Spec(_)) {
        return Err(format!("rule {row:?}: kind text needs a spec: source"));
    }
    let kind = match (&source, field.kind.as_deref()) {
        (Source::Fixed | Source::Live(_) | Source::Engine(_), None) => Kind::Token,
        (Source::Name, Some("quant")) => Kind::Quant,
        (Source::Name, _) => return Err("a name source needs kind quant".to_owned()),
        (Source::Fixed, Some(_)) => return Err("kind needs a source".to_owned()),
        (Source::Live(_), Some(_)) => {
            return Err("kind is not allowed with a live source".to_owned());
        }
        (_, None) => return Err("kind is required with this source".to_owned()),
        (_, Some(word)) => match word {
            "number" => Kind::Number,
            "token" => Kind::Token,
            "quant" => Kind::Quant,
            "present" => Kind::Present,
            "mib" => Kind::Mib,
            "text" => Kind::Text,
            other => {
                return Err(format!(
                    "unknown kind {:?} (number, token, quant, present, mib, text)",
                    printable(other)
                ));
            }
        },
    };
    if kind == Kind::Present && !matches!(source, Source::Flag(_)) {
        return Err("kind present needs a flag source".to_owned());
    }
    if matches!(source, Source::Engine(_)) && matches!(kind, Kind::Quant | Kind::Present) {
        return Err("an engine source takes kind number, token or mib".to_owned());
    }
    if field.max_chars.is_some() && kind != Kind::Text {
        return Err(format!(
            "rule {row:?}: max_chars is only valid with kind text"
        ));
    }
    let max_chars = if kind == Kind::Text {
        match field.max_chars {
            None => DEFAULT_TEXT_CHARS,
            Some(n) => match usize::try_from(n) {
                Ok(n) if (1..=MAX_TEXT_VALUE_CHARS).contains(&n) => n,
                _ => {
                    return Err(format!(
                        "rule {row:?}: max_chars {n} is outside 1..={MAX_TEXT_VALUE_CHARS}"
                    ));
                }
            },
        }
    } else {
        0
    };
    if source == Source::Fixed && field.default.is_none() {
        return Err("a field with no source needs a default".to_owned());
    }
    let label = field.label.clone().unwrap_or_default();
    text_ok("label", &label, MAX_TEXT_CHARS)?;
    let suffix = field.suffix.clone().unwrap_or_default();
    text_ok("suffix", &suffix, MAX_TEXT_CHARS)?;
    let sep = field.sep.clone().unwrap_or_else(|| SEP.to_owned());
    text_ok("sep", &sep, MAX_SEP_CHARS)?;
    if field.map.len() > MAX_MAP {
        return Err(format!("map has more than {MAX_MAP} entries"));
    }
    for (key, value) in &field.map {
        text_ok("map key", key, MAX_TEXT_CHARS)?;
        text_ok("map value", value, MAX_TEXT_CHARS)?;
        if value.is_empty() {
            return Err("map value must not be empty".to_owned());
        }
    }
    if let Some(group) = &field.group
        && (group.is_empty() || !is_token(group))
    {
        return Err("group must be 1..=16 characters of A-Z a-z 0-9 _ . + -".to_owned());
    }
    if let Some(default) = &field.default {
        text_ok("default", default, MAX_TEXT_CHARS)?;
        if default.is_empty() {
            return Err("default must not be empty".to_owned());
        }
    }
    Ok(Rule {
        row: row.clone(),
        engines: field.engines.clone(),
        matches,
        source,
        kind,
        max_chars,
        label,
        suffix,
        sep,
        map: field.map.clone(),
        default: field.default.clone(),
        fallback: field.fallback,
        group: field.group.clone(),
        order: field.order,
    })
}

fn parse_source(source: Option<&str>) -> Result<Source, String> {
    let Some(source) = source else {
        return Ok(Source::Fixed);
    };
    if source == "name" {
        return Ok(Source::Name);
    }
    let bad = || {
        format!(
            "source {:?} is not flag:NAMES, env:NAME, json:NAMES:KEY, live:NAME, engine:KEY, spec:PATH or name",
            printable(source)
        )
    };
    let (scheme, rest) = source.split_once(':').ok_or_else(bad)?;
    match scheme {
        "flag" => Ok(Source::Flag(flag_names(rest)?)),
        "env" => {
            let ok = !rest.is_empty()
                && rest.len() <= MAX_NAME_CHARS
                && rest.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                && !rest.as_bytes()[0].is_ascii_digit();
            if ok {
                Ok(Source::Env(rest.to_owned()))
            } else {
                Err("env name must be letters, digits and _".to_owned())
            }
        }
        "json" => {
            let (flags, key) = rest.rsplit_once(':').ok_or_else(bad)?;
            let ok = !key.is_empty()
                && key.len() <= MAX_NAME_CHARS
                && key
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'));
            if !ok {
                return Err("json key must be letters, digits, _ - .".to_owned());
            }
            Ok(Source::Json {
                flags: flag_names(flags)?,
                key: key.to_owned(),
            })
        }
        "live" => Live::from_name(rest).map(Source::Live).ok_or_else(|| {
            format!(
                "unknown live source {:?} (engine, ctx, kv_dtype, kv_block, prefix_cache, spec_accept, spec_len, expert_hit, pcie_share)",
                printable(rest)
            )
        }),
        "engine" => ENGINE_KEYS
            .iter()
            .find(|key| **key == rest)
            .map(|key| Source::Engine(key))
            .ok_or_else(|| {
                format!(
                    "unknown engine key {:?} (see README, SETUP rules)",
                    printable(rest)
                )
            }),
        "spec" => spec_path(rest).map(Source::Spec),
        _ => Err(bad()),
    }
}

/// `a.b.c`: 1..=[`MAX_SPEC_SEGMENTS`] pieces of `[A-Za-z0-9_]`, each at
/// most [`MAX_SPEC_SEGMENT`] bytes. Anything else is a bad rule.
fn spec_path(path: &str) -> Result<Vec<String>, String> {
    let bad = || {
        format!(
            "spec path must be 1..={MAX_SPEC_SEGMENTS} dot-separated segments of letters, digits and _"
        )
    };
    if path.is_empty() || path.starts_with('.') || path.ends_with('.') || path.contains("..") {
        return Err(bad());
    }
    let parts: Vec<&str> = path.split('.').collect();
    if parts.is_empty() || parts.len() > MAX_SPEC_SEGMENTS {
        return Err(bad());
    }
    for part in &parts {
        let ok = !part.is_empty()
            && part.len() <= MAX_SPEC_SEGMENT
            && part.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
        if !ok {
            return Err(bad());
        }
    }
    Ok(parts.iter().map(|part| (*part).to_owned()).collect())
}

fn flag_names(list: &str) -> Result<Vec<String>, String> {
    let names: Vec<String> = list.split(',').map(str::to_owned).collect();
    if names.len() > MAX_FLAGS {
        return Err(format!("more than {MAX_FLAGS} flag names"));
    }
    for name in &names {
        let ok = name.len() >= 2
            && name.len() <= MAX_NAME_CHARS
            && name.starts_with('-')
            && name != "--"
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'));
        if !ok {
            return Err(format!(
                "flag {:?} must start with - and be letters, digits, _ - .",
                printable(name)
            ));
        }
    }
    Ok(names)
}

/// Rule text: printable ASCII or `·`, at most `cap` characters.
fn text_ok(what: &str, text: &str, cap: usize) -> Result<(), String> {
    if text.chars().count() > cap {
        return Err(format!("{what} is longer than {cap} characters"));
    }
    if !text
        .chars()
        .all(|ch| ('\u{20}'..='\u{7e}').contains(&ch) || ch == '\u{00B7}')
    {
        return Err(format!("{what} must be printable ASCII"));
    }
    Ok(())
}

/// Config text quoted back in an error: printable ASCII, capped.
fn printable(text: &str) -> String {
    text.chars()
        .take(MAX_NAME_CHARS)
        .map(|ch| if ch.is_ascii_graphic() { ch } else { '?' })
        .collect()
}

/// Whitespace-separated words and their byte offsets.
fn tokenize(cmd: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut start = None;
    for (at, ch) in cmd.char_indices() {
        if ch.is_whitespace() {
            if let Some(from) = start.take() {
                out.push((from, &cmd[from..at]));
            }
        } else if start.is_none() {
            start = Some(at);
        }
    }
    if let Some(from) = start {
        out.push((from, &cmd[from..]));
    }
    out
}

/// The value of the last of `names` in `words`. A following word that is
/// itself a flag is not a value (a negative number is).
fn flag_value(words: &[&str], names: &[String], kind: Kind) -> Option<String> {
    let mut out = None;
    let mut index = 0;
    while index < words.len() {
        let word = words[index];
        for name in names {
            if word == name {
                if kind == Kind::Present {
                    out = Some("on".to_owned());
                } else if let Some(next) = words.get(index + 1)
                    && (!next.starts_with('-') || is_number(next))
                {
                    out = Some((*next).to_owned());
                    index += 1;
                }
                break;
            }
            if let Some(value) = word
                .strip_prefix(name.as_str())
                .and_then(|rest| rest.strip_prefix('='))
            {
                out = Some(if kind == Kind::Present {
                    "on".to_owned()
                } else {
                    value.to_owned()
                });
                break;
            }
        }
        index += 1;
    }
    out
}

/// The value of the last `NAME=V` env assignment in `words`.
fn env_value(words: &[&str], name: &str) -> Option<String> {
    let mut out = None;
    let mut index = 0;
    while index < words.len() {
        let word = words[index];
        let assignment = if word == "-e" || word == "--env" {
            index += 1;
            words.get(index).copied()
        } else if let Some(rest) = word.strip_prefix("--env=") {
            Some(rest)
        } else if word.starts_with(|ch: char| ch.is_ascii_alphabetic() || ch == '_') {
            Some(word)
        } else {
            None
        };
        if let Some((key, value)) = assignment.map(strip_quotes).and_then(|a| a.split_once('='))
            && key == name
        {
            out = Some(value.to_owned());
        }
        index += 1;
    }
    out
}

/// `key` of the JSON object after the last of `flags` (one level deep).
fn json_value(cmd: &str, tokens: &[(usize, &str)], flags: &[String], key: &str) -> Option<String> {
    let mut out = None;
    for (at, word) in tokens {
        for flag in flags {
            let start = if *word == flag {
                at + flag.len()
            } else if word
                .strip_prefix(flag.as_str())
                .is_some_and(|rest| rest.starts_with('='))
            {
                at + flag.len() + 1
            } else {
                continue;
            };
            // As written, else with shell-escaped quotes (`{\"a\":1}`).
            let mut end = cmd.len().min(start + 2 * MAX_JSON_BYTES);
            while !cmd.is_char_boundary(end) {
                end -= 1;
            }
            let rest = &cmd[start..end];
            let value = json_object(rest)
                .and_then(|object| json_key(object, key))
                .or_else(|| {
                    let unescaped = rest.replace("\\\"", "\"");
                    json_object(&unescaped).and_then(|object| json_key(object, key))
                });
            if value.is_some() {
                out = value;
            }
        }
    }
    out
}

/// The balanced `{…}` at the start of `text` (after blanks and one quote),
/// at most [`MAX_JSON_BYTES`].
fn json_object(text: &str) -> Option<&str> {
    let text = text.trim_start();
    let text = text.strip_prefix(['\'', '"']).unwrap_or(text);
    if !text.starts_with('{') {
        return None;
    }
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (at, byte) in text.bytes().enumerate() {
        if at >= MAX_JSON_BYTES {
            return None;
        }
        if in_string {
            match byte {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[..=at]);
                }
            }
            _ => {}
        }
    }
    None
}

/// A string, number or bool at `key` of a JSON object, as text.
fn json_key(object: &str, key: &str) -> Option<String> {
    let parsed = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(object).ok()?;
    match parsed.get(key)? {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Number(number) => Some(number.to_string()),
        serde_json::Value::Bool(on) => Some(on.to_string()),
        _ => None,
    }
}

/// The scalar at `path` of a spec document. An object, array, null or a
/// missing key is nothing. A string longer than [`MAX_SPEC_VALUE_CHARS`]
/// is nothing, so a huge path cannot become a file stem in [`clean`].
/// A [`Kind::Text`] rule takes a string only: a number or a bool shows
/// nothing, and the string is cut later, not dropped here.
fn spec_at(doc: &serde_json::Value, path: &[String], kind: Kind) -> Option<String> {
    let mut cur = doc;
    for segment in path {
        cur = cur.as_object()?.get(segment)?;
    }
    if kind == Kind::Text {
        return match cur {
            serde_json::Value::String(text) if text.chars().count() <= MAX_SPEC_VALUE_CHARS => {
                Some(text.clone())
            }
            _ => None,
        };
    }
    match cur {
        serde_json::Value::Null | serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
            None
        }
        serde_json::Value::Bool(on) => Some(on.to_string()),
        serde_json::Value::String(text) => {
            (text.chars().count() <= MAX_SPEC_VALUE_CHARS).then(|| text.clone())
        }
        serde_json::Value::Number(number) => Some(spec_number(number)),
    }
}

/// Integral floats such as `20.0` print as `20`. A fraction stays as the
/// JSON number wrote it (`0.95`). `json:` values are not rewritten here.
fn spec_number(number: &serde_json::Number) -> String {
    if let Some(n) = number.as_i64() {
        return n.to_string();
    }
    if let Some(n) = number.as_u64() {
        return n.to_string();
    }
    // 1e15 is inside the range where every integer is an exact f64, so the
    // cast does not round. -0.0 passes fract() and prints as 0.
    if let Some(n) = number.as_f64()
        && n.is_finite()
        && n.fract() == 0.0
        && n.abs() < 1e15
    {
        return (n as i64).to_string();
    }
    number.to_string()
}

fn strip_quotes(text: &str) -> &str {
    for quote in ['\'', '"'] {
        if let Some(inner) = text
            .strip_prefix(quote)
            .and_then(|rest| rest.strip_suffix(quote))
        {
            return inner;
        }
    }
    text
}

/// The display-safe value, or `None` when `raw` does not pass for `kind`.
fn clean(kind: Kind, raw: &str) -> Option<String> {
    // A spec string is already decoded. Shell quotes are a command-line
    // concern; stripping them would eat a label that starts and ends with `"`.
    if kind == Kind::Text {
        return clean_text(raw);
    }
    let raw = strip_quotes(raw);
    match kind {
        Kind::Number => is_number(raw).then(|| raw.to_owned()),
        Kind::Token => {
            let value = if raw.contains('/') {
                let file = raw.rsplit('/').next().unwrap_or(raw);
                match file.rfind('.') {
                    Some(dot) if dot > 0 => &file[..dot],
                    _ => file,
                }
            } else {
                raw
            };
            is_token(value).then(|| value.to_owned())
        }
        Kind::Quant => cmdline::quant_tag(raw),
        Kind::Present => Some("on".to_owned()),
        Kind::Mib => (is_number(raw) && !raw.starts_with('-')).then(|| raw.to_owned()),
        Kind::Text => None,
    }
}

/// A `text` value. Whitespace runs, including newlines and tabs, become
/// one space; leading and trailing whitespace go. A control or a
/// zero-width character is dropped, so an escape sequence or a newline
/// cannot reach the console or the grid file. Everything else is kept,
/// and the console font's mapping plus the grid file's `rich` scalar
/// happen when the row is drawn. Empty is nothing.
fn clean_text(raw: &str) -> Option<String> {
    let mut out = String::new();
    let mut spaced = false;
    for ch in raw.chars() {
        if ch.is_whitespace() {
            if !out.is_empty() {
                spaced = true;
            }
            continue;
        }
        if text_skipped(ch) {
            continue;
        }
        if spaced {
            out.push(' ');
            spaced = false;
        }
        out.push(ch);
    }
    (!out.is_empty()).then_some(out)
}

/// Controls, and the zero-width characters [`crate::tty::sanitize`] gives
/// no cell. Whitespace is not here: [`clean_text`] turns it into spaces.
fn text_skipped(ch: char) -> bool {
    matches!(
        ch,
        '\u{00ad}' | '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{feff}'
    ) || ch.is_control()
}

/// An engine-reported value for a rule of `kind`: a number for `number`
/// and `mib`, else a number or a short token.
fn engine_value(kind: Kind, value: &str) -> Option<String> {
    let ok = match kind {
        Kind::Number | Kind::Mib => is_number(value),
        Kind::Text => false,
        _ => is_number(value) || is_token(value),
    };
    ok.then(|| value.to_owned())
}

/// MiB as `16.4 GiB`, or `512 MiB` below 1 GiB.
fn mib_text(text: &str) -> String {
    let Ok(mib) = text.parse::<f64>() else {
        return text.to_owned();
    };
    if mib < 1024.0 {
        format!("{} MiB", crate::tty::layout::commas(mib.round() as u64))
    } else {
        let tenths = (mib / 1024.0 * 10.0).round() as u64;
        format!(
            "{}.{} GiB",
            crate::tty::layout::commas(tenths / 10),
            tenths % 10
        )
    }
}

/// `-?digits(.digits)?`, at most [`MAX_NUMBER_CHARS`].
fn is_number(text: &str) -> bool {
    if text.is_empty() || text.len() > MAX_NUMBER_CHARS {
        return false;
    }
    let body = text.strip_prefix('-').unwrap_or(text);
    let (whole, fraction) = match body.split_once('.') {
        Some((whole, fraction)) => (whole, Some(fraction)),
        None => (body, None),
    };
    let digits = |part: &str| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit());
    digits(whole) && fraction.is_none_or(digits)
}

/// `262144` is `262,144`; a fraction and a sign stay as written.
fn number_text(text: &str) -> String {
    let (sign, body) = match text.strip_prefix('-') {
        Some(body) => ("-", body),
        None => ("", text),
    };
    let (whole, fraction) = match body.split_once('.') {
        Some((whole, fraction)) => (whole, Some(fraction)),
        None => (body, None),
    };
    let grouped = match whole.parse::<u64>() {
        Ok(n) => crate::tty::layout::commas(n),
        Err(_) => whole.to_owned(),
    };
    match fraction {
        Some(fraction) => format!("{sign}{grouped}.{fraction}"),
        None => format!("{sign}{grouped}"),
    }
}

/// Per mille as a whole percent, `87 %`.
fn percent(permille: u16) -> String {
    format!("{} %", (u32::from(permille.min(1000)) + 5) / 10)
}

fn live_value(which: Live, live: &LiveCtx<'_>) -> Option<String> {
    let detail = live.detail;
    let engine = live.info.map(|info| &info.engine);
    match which {
        Live::Engine => Some(live.backend.display_name().to_owned()),
        Live::Ctx => detail
            .and_then(|d| d.ctx)
            .filter(|ctx| *ctx > 0)
            .map(|ctx| crate::tty::layout::commas(u64::from(ctx))),
        Live::KvDtype => {
            let detail = detail?;
            let k = detail.kv_k.as_deref().filter(|t| is_token(t))?;
            match detail.kv_v.as_deref().filter(|t| is_token(t)) {
                Some(v) if v != k => Some(format!("{k}/{v}")),
                _ => Some(k.to_owned()),
            }
        }
        Live::KvBlock => detail
            .and_then(|d| d.kv_block)
            .filter(|block| (1..=llama_core::detail::MAX_KV_BLOCK).contains(block))
            .map(|block| crate::tty::layout::commas(u64::from(block))),
        Live::PrefixCache => detail
            .and_then(|d| d.prefix_cache)
            .map(|on| if on { "on" } else { "off" }.to_owned()),
        Live::SpecAccept => engine.and_then(|e| e.spec_permille).map(percent),
        Live::ExpertHit => engine.and_then(|e| e.expert_hit_permille).map(percent),
        Live::PcieShare => engine.and_then(|e| e.pcie_share_permille).map(percent),
        Live::SpecLen => engine.and_then(|e| e.spec_len_centi).map(|centi| {
            let tenths = (u32::from(centi) + 5) / 10;
            format!("{}.{}/step", tenths / 10, tenths % 10)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(row: &str, source: Option<&str>, kind: Option<&str>) -> SetupField {
        SetupField {
            row: row.to_owned(),
            source: source.map(str::to_owned),
            kind: kind.map(str::to_owned),
            order: 1000,
            ..SetupField::default()
        }
    }

    fn only(fields: Vec<SetupField>) -> Rules {
        Rules::compile(&Setup {
            defaults: false,
            field: fields,
            ..Setup::default()
        })
        .expect("compile")
    }

    #[test]
    fn builtin_rules_compile() {
        let rules = Rules::builtin();
        assert!(rules.rules.len() > 30, "{}", rules.rules.len());
        assert!(rules.rules.windows(2).all(|w| w[0].order <= w[1].order));
        assert!(
            rules
                .rules
                .iter()
                .all(|rule| !matches!(rule.source, Source::Spec(_))),
            "built-in rules do not read spec files"
        );
    }

    #[test]
    fn tokens_keep_offsets() {
        assert_eq!(
            tokenize(" a  bc\n-d=1 "),
            vec![(1, "a"), (4, "bc"), (7, "-d=1")]
        );
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn flag_forms_and_later_wins() {
        let words = ["x", "-c", "1", "--ctx-size=2", "-c", "--other", "-n", "-1"];
        let names = vec!["-c".to_owned(), "--ctx-size".to_owned()];
        assert_eq!(
            flag_value(&words, &names, Kind::Number).as_deref(),
            Some("2")
        );
        let n = vec!["-n".to_owned()];
        assert_eq!(flag_value(&words, &n, Kind::Number).as_deref(), Some("-1"));
        let other = vec!["--other".to_owned()];
        assert_eq!(
            flag_value(&words, &other, Kind::Present).as_deref(),
            Some("on")
        );
        assert_eq!(flag_value(&words, &other, Kind::Token), None);
        // A prefix of a longer flag is not the flag.
        let short = vec!["--ctx".to_owned()];
        assert_eq!(flag_value(&words, &short, Kind::Number), None);
    }

    #[test]
    fn env_forms() {
        let words = [
            "podman",
            "run",
            "-e",
            "SPEC=mtp",
            "--env",
            "'CTX=long'",
            "--env=A=1",
            "B=2",
            "-e",
            "SPEC=dflash2",
        ];
        assert_eq!(env_value(&words, "SPEC").as_deref(), Some("dflash2"));
        assert_eq!(env_value(&words, "CTX").as_deref(), Some("long"));
        assert_eq!(env_value(&words, "A").as_deref(), Some("1"));
        assert_eq!(env_value(&words, "B").as_deref(), Some("2"));
        assert_eq!(env_value(&words, "C"), None);
    }

    #[test]
    fn json_objects_one_level_deep() {
        let cmd = r#"vllm serve /m --speculative-config '{"method": "mtp", "num_speculative_tokens": 4, "model": "/models/drafter", "deep": {"x": 1}}' --port 1"#;
        let tokens = tokenize(cmd);
        let flags = vec!["--speculative-config".to_owned()];
        let get = |key: &str| json_value(cmd, &tokens, &flags, key);
        assert_eq!(get("method").as_deref(), Some("mtp"));
        assert_eq!(get("num_speculative_tokens").as_deref(), Some("4"));
        assert_eq!(get("deep"), None);
        assert_eq!(get("absent"), None);
        // `=`-joined, escaped quotes, and a bool.
        let cmd = r#"x --chat-template-kwargs={\"enable_thinking\":false}"#;
        let flags = vec!["--chat-template-kwargs".to_owned()];
        assert_eq!(
            json_value(cmd, &tokenize(cmd), &flags, "enable_thinking").as_deref(),
            Some("false")
        );
        // Unbalanced or oversized objects read nothing.
        assert_eq!(json_object("{\"a\": 1"), None);
        let huge = format!("{{\"a\":\"{}\"}}", "x".repeat(MAX_JSON_BYTES));
        assert_eq!(json_object(&huge), None);
        assert_eq!(json_object(r#" '{"a":"}"}' x"#), Some(r#"{"a":"}"}"#));
    }

    #[test]
    fn values_are_numbers_tokens_or_stems_never_paths() {
        assert_eq!(clean(Kind::Number, "262144").as_deref(), Some("262144"));
        assert_eq!(clean(Kind::Number, "'0.92'").as_deref(), Some("0.92"));
        assert_eq!(clean(Kind::Number, "-1").as_deref(), Some("-1"));
        for bad in [
            "1e5",
            "",
            "-",
            "1.",
            ".5",
            "12345678901234567",
            "1,000",
            "0x10",
        ] {
            assert_eq!(clean(Kind::Number, bad), None, "{bad}");
        }
        assert_eq!(clean(Kind::Token, "q8_0").as_deref(), Some("q8_0"));
        assert_eq!(
            clean(Kind::Token, "/models/hq/uncensored.safetensors").as_deref(),
            Some("uncensored")
        );
        assert_eq!(
            clean(Kind::Token, "/models/a-very-long-file-name.gguf"),
            None
        );
        assert_eq!(clean(Kind::Token, "a b"), None);
        // A path-like value is only ever its file stem.
        assert_eq!(clean(Kind::Token, "../../etc/x.conf").as_deref(), Some("x"));
        assert_eq!(clean(Kind::Token, "x;rm"), None);
        assert_eq!(
            clean(Kind::Quant, "/m/Qwen3-27B-UD-Q6_K_XL.gguf").as_deref(),
            Some("UD-Q6_K_XL")
        );
        assert_eq!(clean(Kind::Quant, "/m/secret-name.gguf"), None);
        assert_eq!(number_text("262144"), "262,144");
        assert_eq!(number_text("-24000"), "-24,000");
        assert_eq!(number_text("4096.25"), "4,096.25");
        assert_eq!(number_text("0.92"), "0.92");
    }

    #[test]
    fn bad_rules_are_clear_errors() {
        let error = |field: SetupField| {
            Rules::compile(&Setup {
                defaults: true,
                field: vec![SetupField::default(), field],
                ..Setup::default()
            })
            .expect_err("bad rule")
        };
        let ok = field("ctx", Some("flag:-c"), Some("number"));
        assert!(only(vec![ok.clone()]).rules.len() == 1);
        // Index 0 is the empty row before it.
        assert_eq!(error(ok.clone()).0, 0);
        let (index, reason) = Rules::compile(&Setup {
            defaults: true,
            field: vec![ok.clone(), field("ctx", Some("flag:-c"), Some("nubmer"))],
            ..Setup::default()
        })
        .expect_err("kind");
        assert_eq!(index, 1);
        assert!(reason.contains("unknown kind \"nubmer\""), "{reason}");
        let cases = [
            (field("", Some("flag:-c"), Some("number")), "row"),
            (
                field("way-too-long", Some("flag:-c"), Some("number")),
                "row",
            ),
            (
                field("ctx", Some("flag:c"), Some("number")),
                "must start with -",
            ),
            (field("ctx", Some("flag:-c"), None), "kind is required"),
            (
                field("ctx", Some("live:ctx"), Some("number")),
                "not allowed",
            ),
            (field("ctx", Some("live:nope"), None), "unknown live source"),
            (field("ctx", Some("env:9X"), Some("token")), "env name"),
            (field("ctx", Some("json:--x"), Some("token")), "source"),
            (
                field("ctx", Some("json:--x:a b"), Some("token")),
                "json key",
            ),
            (field("ctx", Some("env:X"), Some("present")), "present"),
            (field("ctx", Some("http:x"), Some("token")), "source"),
            (field("ctx", Some("spec:"), Some("number")), "spec path"),
            (field("ctx", Some("spec:a-b"), Some("number")), "spec path"),
            (field("ctx", Some("spec:a..b"), Some("number")), "spec path"),
            (field("ctx", Some("spec:.a"), Some("number")), "spec path"),
            (field("ctx", Some("spec:a."), Some("number")), "spec path"),
            (
                field("ctx", Some("spec:a.b.c.d.e.f.g.h.i"), Some("number")),
                "spec path",
            ),
            (field("ctx", Some("spec:x"), None), "kind is required"),
            (field("ctx", Some("spec:x"), Some("present")), "present"),
            (field("ctx", None, None), "needs a default"),
        ];
        for (rule, want) in cases {
            let (index, reason) = Rules::compile(&Setup {
                defaults: false,
                field: vec![rule.clone()],
                ..Setup::default()
            })
            .expect_err(want);
            assert_eq!(index, 0);
            assert!(reason.contains(want), "{want}: {reason}");
        }
        let mut text = field("ctx", Some("flag:-c"), Some("number"));
        text.label = Some("bad\u{1b}[31m".to_owned());
        assert!(only_err(text).contains("label must be printable"));
        let mut long = field("ctx", Some("flag:-c"), Some("number"));
        long.default = Some("x".repeat(MAX_TEXT_CHARS + 1));
        assert!(only_err(long).contains("default is longer"));
    }

    fn only_err(field: SetupField) -> String {
        Rules::compile(&Setup {
            defaults: false,
            field: vec![field],
            ..Setup::default()
        })
        .expect_err("bad")
        .1
    }

    #[test]
    fn rows_apply_engines_map_default_and_fallback() {
        let mut ncmoe = field("experts", Some("flag:-ncmoe"), Some("number"));
        ncmoe.suffix = Some(" layers in RAM".to_owned());
        ncmoe.map.insert("0".to_owned(), "full GPU".to_owned());
        ncmoe.default = Some("full GPU".to_owned());
        ncmoe.fallback = true;
        ncmoe.order = 31;
        let mut cmoe = field("experts", Some("flag:--cpu-moe"), Some("present"));
        cmoe.map
            .insert("on".to_owned(), "all layers in RAM".to_owned());
        cmoe.order = 30;
        let mut vllm_only = field("ctx", Some("flag:-c"), Some("number"));
        vllm_only.engines = vec![Backend::Vllm];
        vllm_only.order = 20;
        let rules = only(vec![ncmoe, cmoe, vllm_only]);
        let live = LiveCtx {
            backend: Backend::LlamaCpp,
            detail: None,
            info: None,
            engine: None,
        };
        let rows = |cmd: &str| -> Vec<(String, Vec<(String, bool)>)> {
            rules
                .rows(&rules.extract(cmd), &live)
                .into_iter()
                .map(|row| {
                    (
                        row.label,
                        row.items.into_iter().map(|i| (i.text, i.dim)).collect(),
                    )
                })
                .collect()
        };
        let want =
            |text: &str, dim: bool| vec![("experts".to_owned(), vec![(text.to_owned(), dim)])];
        assert_eq!(rows("llama-server -c 4096"), want("full GPU", true));
        assert_eq!(
            rows("llama-server -ncmoe 24"),
            want("24 layers in RAM", false)
        );
        assert_eq!(rows("llama-server -ncmoe 0"), want("full GPU", false));
        assert_eq!(
            rows("llama-server -ncmoe 24 --cpu-moe"),
            want("all layers in RAM", false)
        );
    }

    #[test]
    fn match_limits_a_rule_to_matching_commands() {
        let mut spec = field("spec", Some("env:SPEC"), Some("token"));
        spec.matches = Some("QwenBox".to_owned());
        let rules = only(vec![spec]);
        assert_eq!(
            rules.extract("podman run -e SPEC=mtp ghcr.io/x/qwenbox single"),
            vec![Found {
                rule: 0,
                value: "mtp".to_owned()
            }]
        );
        assert!(
            rules
                .extract("podman run -e SPEC=mtp ghcr.io/x/other single")
                .is_empty()
        );
    }

    #[test]
    fn spec_paths_read_scalars_like_json_values() {
        let eight = (0..MAX_SPEC_SEGMENTS)
            .map(|i| format!("a{i}"))
            .collect::<Vec<_>>()
            .join(".");
        assert_eq!(
            only(vec![field(
                "ctx",
                Some(&format!("spec:{eight}")),
                Some("number")
            )])
            .rules
            .len(),
            1
        );
        let wide = "a".repeat(MAX_SPEC_SEGMENT);
        assert_eq!(
            only(vec![field(
                "ctx",
                Some(&format!("spec:{wide}")),
                Some("token")
            )])
            .rules
            .len(),
            1
        );
        let wider = "a".repeat(MAX_SPEC_SEGMENT + 1);
        assert!(
            only_err(field("ctx", Some(&format!("spec:{wider}")), Some("number")))
                .contains("spec path")
        );

        let doc: serde_json::Value = serde_json::from_str(
            r#"{"served_context":131072,"on":true,"off":false,"quant":"/models/Example-Q6_K_XL.gguf","stem":"/models/example.gguf","mib":2048,"top_k":20.0,"min_p":0.0,"top_p":0.95,"neg":-0.0,"thinking":{"server_budget":24000},"gone":null,"obj":{"a":1},"arr":[1],"text":"f16","blank":"","esc":"a\u001bb"}"#,
        )
        .expect("doc");
        let mut huge = doc.clone();
        huge.as_object_mut().unwrap().insert(
            "huge".to_owned(),
            serde_json::Value::String("x".repeat(2000)),
        );
        // A long path whose tail would be a token, if clean() saw it.
        huge.as_object_mut().unwrap().insert(
            "long_path".to_owned(),
            serde_json::Value::String(format!("/models/{}/ok.gguf", "a".repeat(2000))),
        );

        let mut ctx = field("ctx", Some("spec:served_context"), Some("number"));
        ctx.order = 20;
        let mut flag = field("think", Some("spec:on"), Some("token"));
        flag.map.insert("true".to_owned(), "on".to_owned());
        flag.map.insert("false".to_owned(), "off".to_owned());
        flag.order = 50;
        let mut off = field("think", Some("spec:off"), Some("token"));
        off.map.insert("true".to_owned(), "on".to_owned());
        off.map.insert("false".to_owned(), "off".to_owned());
        off.order = 51;
        let mut quant = field("engine", Some("spec:quant"), Some("quant"));
        quant.order = 13;
        let mut stem = field("engine", Some("spec:stem"), Some("token"));
        stem.order = 14;
        let mut mib = field("experts", Some("spec:mib"), Some("mib"));
        mib.order = 30;
        let mut top_k = field("sample", Some("spec:top_k"), Some("number"));
        top_k.label = Some("top-k".to_owned());
        top_k.order = 72;
        let mut min_p = field("sample", Some("spec:min_p"), Some("number"));
        min_p.label = Some("min-p".to_owned());
        min_p.sep = Some(" ".to_owned());
        min_p.order = 73;
        let mut top_p = field("sample", Some("spec:top_p"), Some("number"));
        top_p.label = Some("top-p".to_owned());
        top_p.order = 71;
        let mut neg = field("sample", Some("spec:neg"), Some("number"));
        neg.order = 74;
        let mut budget = field("think", Some("spec:thinking.server_budget"), Some("number"));
        budget.label = Some("budget".to_owned());
        budget.order = 52;
        let mut text = field("ctx", Some("spec:text"), Some("token"));
        text.label = Some("kv".to_owned());
        text.order = 22;
        let vllm = {
            let mut rule = field("ctx", Some("spec:served_context"), Some("number"));
            rule.engines = vec![Backend::Vllm];
            rule.order = 21;
            rule
        };
        let mut matched = field("ctx", Some("spec:text"), Some("token"));
        matched.matches = Some("only-this".to_owned());
        matched.order = 23;
        let rules = only(vec![
            ctx,
            flag,
            off,
            quant,
            stem,
            mib,
            top_k,
            min_p,
            top_p,
            neg,
            budget,
            text,
            vllm,
            matched,
            field("ctx", Some("spec:gone"), Some("token")),
            field("ctx", Some("spec:obj"), Some("token")),
            field("ctx", Some("spec:arr"), Some("token")),
            field("ctx", Some("spec:missing"), Some("token")),
            field("ctx", Some("spec:obj.a"), Some("number")),
            field("ctx", Some("spec:blank"), Some("token")),
            field("ctx", Some("spec:esc"), Some("token")),
            field("ctx", Some("spec:huge"), Some("token")),
            field("ctx", Some("spec:long_path"), Some("token")),
            field("think", Some("spec:on"), Some("number")),
        ]);
        let found = rules.extract_with("llama-server", "example", None, Some(&huge));
        let values: Vec<&str> = found.iter().map(|item| item.value.as_str()).collect();
        assert!(values.contains(&"131072"), "{values:?}");
        assert!(values.contains(&"true"), "{values:?}");
        assert!(values.contains(&"false"), "{values:?}");
        assert!(values.contains(&"Q6_K_XL"), "{values:?}");
        assert!(values.contains(&"example"), "{values:?}");
        assert!(values.contains(&"2048"), "{values:?}");
        assert!(values.contains(&"20"), "{values:?}");
        assert!(values.contains(&"0"), "{values:?}");
        assert!(values.contains(&"0.95"), "{values:?}");
        assert!(values.contains(&"24000"), "{values:?}");
        assert!(values.contains(&"f16"), "{values:?}");
        assert!(
            !values
                .iter()
                .any(|v| v.contains('\u{1b}') || v.contains('/')),
            "{values:?}"
        );
        // A bool is not a number, and the hostile strings are not tokens.
        assert_eq!(
            values.iter().filter(|v| **v == "true").count(),
            1,
            "{values:?}"
        );

        let live = |backend| LiveCtx {
            backend,
            detail: None,
            info: None,
            engine: None,
        };
        let row_text = |backend| -> Vec<String> {
            rules
                .rows(&found, &live(backend))
                .into_iter()
                .map(|row| {
                    row.items
                        .iter()
                        .enumerate()
                        .map(|(i, item)| {
                            if i == 0 {
                                item.text.clone()
                            } else {
                                format!("{}{}", item.sep, item.text)
                            }
                        })
                        .collect::<String>()
                })
                .collect()
        };
        let llama = row_text(Backend::LlamaCpp);
        // `kv` is the label. `1` is the nested `obj.a` number on the same row.
        assert!(
            llama.iter().any(|row| row == "131,072 · kv f16 · 1"),
            "{llama:?}"
        );
        assert!(llama.iter().any(|row| row.contains("on")), "{llama:?}");
        assert!(llama.iter().any(|row| row.contains("off")), "{llama:?}");
        assert!(llama.iter().any(|row| row.contains("2.0 GiB")), "{llama:?}");
        assert!(
            llama.iter().any(|row| row.contains("top-p 0.95")
                && row.contains("top-k 20")
                && row.contains(" min-p 0")),
            "{llama:?}"
        );
        assert!(
            llama.iter().any(|row| row.contains("budget 24,000")),
            "{llama:?}"
        );
        // `match` did not see "only-this", so the second f16 is absent.
        assert_eq!(
            llama.iter().filter(|row| row.contains("f16")).count(),
            1,
            "{llama:?}"
        );
        let vllm_rows = row_text(Backend::Vllm);
        assert!(
            vllm_rows
                .iter()
                .any(|row| row.contains("131,072 · 131,072")),
            "engines filter adds the vllm copy: {vllm_rows:?}"
        );
        assert!(
            rules
                .extract_with("llama-server", "example", None, None)
                .is_empty()
        );
    }

    /// #114: a `text` rule shows a spec string, cut to `max_chars`, and
    /// nothing else. `map` matches the sanitised value before the cut;
    /// `label` and `suffix` wrap the cut value.
    #[test]
    fn text_rules_show_a_sanitised_spec_string() {
        let mut label = field("note", Some("spec:report.label"), Some("text"));
        label.order = 10;
        let mut capped = field("note", Some("spec:report.long"), Some("text"));
        capped.max_chars = Some(i64::try_from(MAX_TEXT_VALUE_CHARS).unwrap());
        capped.order = 11;
        let mut wide = field("note", Some("spec:report.huge"), Some("text"));
        wide.order = 12;
        let mut over = field("note", Some("spec:report.over"), Some("text"));
        over.order = 13;
        let mut short = field("note", Some("spec:report.short"), Some("text"));
        short.max_chars = Some(4);
        short.label = Some("is".to_owned());
        short.suffix = Some("!".to_owned());
        short.order = 14;
        let mut mapped = field("note", Some("spec:report.nick"), Some("text"));
        mapped.map.insert("hi there".to_owned(), "hello".to_owned());
        mapped.label = Some("ignored".to_owned());
        mapped.max_chars = Some(4);
        mapped.order = 15;
        let mut blank = field("note", Some("spec:report.blank"), Some("text"));
        blank.default = Some("none".to_owned());
        blank.order = 16;
        let scalars = ["num", "flag", "gone", "obj", "arr"];
        let mut fields = vec![label, capped, wide, over, short, mapped, blank];
        for (i, name) in scalars.iter().enumerate() {
            let mut rule = field("skip", Some(&format!("spec:report.{name}")), Some("text"));
            rule.order = 20 + u16::try_from(i).unwrap();
            fields.push(rule);
        }
        let rules = only(fields);

        let long = "x".repeat(130);
        let exactly = "y".repeat(MAX_SPEC_VALUE_CHARS);
        let too_long = "z".repeat(MAX_SPEC_VALUE_CHARS + 1);
        // NUL, DEL, CR, C1 (U+0085 is also a line break), bidi, zero-width
        // and a combining mark. Controls and zero-width characters go;
        // CR and U+0085 collapse with the space already there.
        let hostile = "  a\u{0}\u{1b}]9;SPEC\u{7}\u{7f}b\nb\tc \r\u{85}\u{9b}\u{9c}\u{9d}\u{202e}x\u{200b}\u{200c}\u{200d}\u{feff}\u{ad}y\u{301}  ";
        let doc = serde_json::json!({
            "report": {
                "label": hostile,
                "long": long,
                "huge": exactly,
                "over": too_long,
                "short": "hello",
                "nick": "  hi \n there  ",
                "blank": " \n\t ",
                "num": 35,
                "flag": true,
                "gone": null,
                "obj": {"a": 1},
                "arr": [1],
            }
        });
        let found = rules.extract_with("", "example", None, Some(&doc));
        let values: Vec<&str> = found.iter().map(|item| item.value.as_str()).collect();
        // Controls and the escape are gone; whitespace has collapsed.
        // The bidi mark and the combining mark stay for the row painter.
        let cleaned = "a]9;SPECb b c \u{202e}xy\u{301}";
        assert!(values.contains(&cleaned), "{values:?}");
        assert!(
            values
                .iter()
                .all(|value| !value.chars().any(char::is_control)),
            "{values:?}"
        );
        for ch in [
            '\0', '\u{1b}', '\u{7}', '\u{7f}', '\r', '\u{85}', '\u{9b}', '\u{9c}', '\u{9d}',
            '\u{200b}', '\u{200c}', '\u{200d}', '\u{feff}', '\u{ad}',
        ] {
            assert!(
                values.iter().all(|value| !value.contains(ch)),
                "{ch:?} in {values:?}"
            );
        }
        // 130 and 1024 survive loading. 1025 is dropped, not cut.
        assert!(values.contains(&long.as_str()), "{values:?}");
        assert!(values.contains(&exactly.as_str()), "{values:?}");
        assert!(
            !values.iter().any(|value| value.contains('z')),
            "{values:?}"
        );
        assert_eq!(
            values.len(),
            5,
            "non-strings and the blank show nothing: {values:?}"
        );

        let live = LiveCtx {
            backend: Backend::LlamaCpp,
            detail: None,
            info: None,
            engine: None,
        };
        let rows = rules.rows(&found, &live);
        let note = rows.iter().find(|row| row.label == "note").expect("note");
        let texts: Vec<&str> = note.items.iter().map(|item| item.text.as_str()).collect();
        let cut96: String = "x"
            .repeat(MAX_TEXT_VALUE_CHARS - 1)
            .chars()
            .chain(['\u{2026}'])
            .collect();
        let cut64: String = "y"
            .repeat(DEFAULT_TEXT_CHARS - 1)
            .chars()
            .chain(['\u{2026}'])
            .collect();
        assert_eq!(
            texts,
            [
                "a]9;SPECb b c \u{202e}xy\u{301}",
                cut96.as_str(),
                cut64.as_str(),
                "is hel\u{2026}!",
                "hello",
                "none",
            ]
        );
        assert!(note.items[5].dim, "the default is dim");
        assert!(!note.items[5].text_value, "a default is not the text value");
        assert!(note.items[0].text_value);
        assert!(note.items[3].text_value);
        assert!(
            note.items[4].text_value,
            "a mapped text value still keeps the row"
        );
        assert!(rows.iter().all(|row| row.label != "skip"), "{rows:?}");
    }

    #[test]
    fn text_kind_and_max_chars_are_load_errors() {
        let text_on = |source: &str| field("ctx", Some(source), Some("text"));
        for source in [
            "flag:-c",
            "env:SPEC",
            "json:--speculative-config:method",
            "live:ctx",
            "engine:kv",
            "name",
        ] {
            let reason = only_err(text_on(source));
            assert!(
                reason.contains("rule \"ctx\"")
                    && reason.contains("kind text needs a spec: source"),
                "{source}: {reason}"
            );
        }
        let mut no_source = field("ctx", None, Some("text"));
        no_source.default = Some("x".to_owned());
        let reason = only_err(no_source);
        assert!(reason.contains("rule \"ctx\""), "{reason}");
        assert!(reason.contains("spec:"), "{reason}");

        let mut numbered = field("ctx", Some("flag:-c"), Some("number"));
        numbered.max_chars = Some(64);
        let reason = only_err(numbered);
        assert!(
            reason.contains("rule \"ctx\"") && reason.contains("max_chars is only valid"),
            "{reason}"
        );

        for bad in [0, 97, -1, 10_000] {
            let mut rule = field("note", Some("spec:report.label"), Some("text"));
            rule.max_chars = Some(bad);
            let reason = only_err(rule);
            assert!(
                reason.contains("rule \"note\"") && reason.contains("outside 1..=96"),
                "{bad}: {reason}"
            );
        }
        let mut edge = field("note", Some("spec:report.label"), Some("text"));
        edge.max_chars = Some(1);
        assert_eq!(only(vec![edge]).rules[0].max_chars, 1);
        let mut top = field("note", Some("spec:report.label"), Some("text"));
        top.max_chars = Some(i64::try_from(MAX_TEXT_VALUE_CHARS).unwrap());
        assert_eq!(only(vec![top]).rules[0].max_chars, MAX_TEXT_VALUE_CHARS);
        let plain = field("note", Some("spec:report.label"), Some("text"));
        assert_eq!(only(vec![plain]).rules[0].max_chars, DEFAULT_TEXT_CHARS);
    }

    /// #114: `max_chars` uses the same cut as the row, so a space at the
    /// cut does not stay in front of `…`.
    #[test]
    fn text_max_chars_drops_the_space_before_the_ellipsis() {
        let mut rule = field("note", Some("spec:report.label"), Some("text"));
        rule.max_chars = Some(7);
        let rules = only(vec![rule]);
        let doc = serde_json::json!({ "report": { "label": "hello world" } });
        let found = rules.extract_with("", "example", None, Some(&doc));
        let live = LiveCtx {
            backend: Backend::LlamaCpp,
            detail: None,
            info: None,
            engine: None,
        };
        let rows = rules.rows(&found, &live);
        assert_eq!(rows[0].items[0].text, "hello\u{2026}");
    }
}
