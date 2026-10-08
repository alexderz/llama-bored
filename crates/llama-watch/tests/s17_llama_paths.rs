//! S17: the watcher's llama-swap reads are a fixed set of GETs on the
//! configured loopback URL (`llama.url` must be a loopback IP literal):
//! `/running`, `/upstream/<model>/metrics`, `/upstream/<model>/slots`,
//! `/api/metrics/activity` and, since #5, `/api/captures/<id>`. A new
//! endpoint fails this scan until it is added here on purpose. #31's
//! backend detection reuses `/upstream/<model>/metrics`: no new path.
//! `tests/s31_ready_gate.rs` pins at runtime that no upstream path is
//! requested for a model `/running` does not list as `ready`, in the
//! `/running` read just before it (#70). The scan below pins that every
//! upstream GET goes through that one fresh gate, `upstream_read`, and that
//! the agent follows no redirect.
//!
//! #80 adds two on purpose. `GET /api/events`, llama-swap's own SSE stream
//! (like `/running`, it names no model path and cannot load one), read on
//! its own thread by `src/sources/events.rs` with an agent from
//! `new_agent`. And the upstream leaf `v1/loads?include=core` (SGLang's
//! live load report), through the same fresh gate.

use std::path::PathBuf;

fn source(rel: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{rel}: {err}"));
    // Production code only: the unit tests at the end build other URLs.
    let code = text.split("#[cfg(test)]").next().unwrap_or("");
    code.lines()
        .map(|line| match line.find("//") {
            Some(at) if !line[..at].contains('"') => &line[..at],
            _ => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The string literals inside each call to `name(`.
fn call_literals(code: &str, name: &str) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    let needle = format!("{name}(");
    let mut rest = code;
    while let Some(at) = rest.find(&needle) {
        let before = rest[..at].chars().last();
        let args_start = at + needle.len();
        rest = &rest[args_start..];
        if before.is_some_and(|ch| ch.is_alphanumeric() || ch == '_') {
            continue;
        }
        let mut depth = 1usize;
        let mut end = 0;
        for (i, ch) in rest.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let args = &rest[..end];
        let literals: Vec<String> = args
            .split('"')
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, lit)| lit.to_owned())
            .collect();
        out.push(literals);
    }
    out
}

#[test]
fn the_poller_gets_only_the_allowed_llama_swap_paths() {
    let poller = source("src/poller.rs");
    let mut joined: Vec<String> = call_literals(&poller, "join_url")
        .into_iter()
        .flatten()
        .collect();
    joined.sort();
    joined.dedup();
    // `upstream` builds its path through `join_url`; its leaves are below.
    joined.retain(|lit| lit != "upstream/{model}/{leaf}");
    assert_eq!(
        joined,
        vec![
            "api/captures/{}".to_owned(),
            "api/metrics/activity".to_owned()
        ],
        "a new llama-swap path needs S17 updated on purpose"
    );
    // #70: the leaves are named only at the fresh gate's call sites.
    let mut leaves: Vec<String> = call_literals(&poller, "upstream_read")
        .into_iter()
        .flatten()
        .collect();
    leaves.sort();
    leaves.dedup();
    assert_eq!(
        leaves,
        vec![
            "metrics".to_owned(),
            "slots".to_owned(),
            "v1/loads?include=core".to_owned()
        ]
    );
    // The gate's own name for that leaf is the same text (#80).
    assert!(poller.contains(r#"const LOADS_LEAF: &str = "v1/loads?include=core";"#));

    // #80: the event stream is one fixed path, one GET site, on an agent
    // from `new_agent`.
    let events = source("src/sources/events.rs");
    assert!(events.contains(r#"format!("{}/api/events", url.trim_end_matches('/'))"#));
    assert_eq!(events.matches("/api/").count(), 1, "one llama-swap path");
    assert_eq!(
        events.matches(".get(endpoint)").count(),
        1,
        "events GET sites"
    );
    assert!(events.contains("super::llamaswap::new_agent()"));
    assert_eq!(
        events.matches("Agent::").count(),
        0,
        "one agent, from new_agent"
    );

    let running = source("src/sources/llamaswap.rs");
    assert!(running.contains(r#"format!("{}/running""#));

    // One GET call site each, and no other HTTP verb anywhere.
    assert_eq!(poller.matches(".get(url)").count(), 1, "poller GET sites");
    assert_eq!(running.matches(".get(&endpoint)").count(), 1);
    for rel in [
        "src/poller.rs",
        "src/sources/llamaswap.rs",
        "src/sources/events.rs",
        "src/service.rs",
    ] {
        let code = source(rel);
        // (`SampleRx::put` is the one-deep sample slot, not HTTP.)
        for verb in [
            ".post(", ".put(url", ".put(&", ".delete(", ".patch(", ".head(", "unload",
        ] {
            assert!(!code.contains(verb), "{rel} uses {verb}");
        }
    }
}

#[test]
fn a_capture_is_fetched_only_with_text_on_and_for_a_captured_row() {
    let poller = source("src/poller.rs");
    let body = poller
        .split("fn poll_capture")
        .nth(1)
        .expect("poll_capture")
        .split("\n    fn ")
        .next()
        .expect("body");
    let text_gate = body.find("if !self.limits.show_text").expect("text gate");
    let captured_gate = body.find("!row.captured").expect("has_capture gate");
    let seen_gate = body.find("self.capture_seen == Some(key)").expect("dedupe");
    // #44: a row is (llama-swap generation, id), and the rows offered are
    // the page just read, never RECENT's older rows.
    let key = body
        .find("let key = (self.recent.generation(), row.id);")
        .expect("generation key");
    assert!(
        body.contains("let Some((row, model)) = rows"),
        "the page's rows"
    );
    let get = body.find("get_limited(").expect("GET");
    assert!(text_gate < get && captured_gate < get && seen_gate < get && key < seen_gate);
    assert!(body.contains("CAPTURE_CAP"), "the read is capped");
}

#[test]
fn the_scanner_finds_literals_in_calls() {
    let code = r#"let a = join_url(&base, "api/x"); let b = upstream(&u, &m, "slots");"#;
    assert_eq!(
        call_literals(code, "join_url"),
        vec![vec!["api/x".to_owned()]]
    );
    assert_eq!(
        call_literals(code, "upstream"),
        vec![vec!["slots".to_owned()]]
    );
}

/// The body of `fn name` in `code`, up to the next method.
fn body(code: &str, name: &str) -> String {
    code.split(&format!("fn {name}("))
        .nth(1)
        .unwrap_or_else(|| panic!("{name}"))
        .split("\n    fn ")
        .next()
        .expect("body")
        .to_owned()
}

/// Byte offset of `needle` in `text`, which must hold it.
fn at(text: &str, needle: &str, what: &str) -> usize {
    text.find(needle)
        .unwrap_or_else(|| panic!("{what}: no {needle:?}"))
}

#[test]
fn upstream_reads_come_from_the_ready_list_only() {
    let poller = source("src/poller.rs");
    // Only `ready` entries of a good `/running` read join the list.
    let running = body(&poller, "poll_running");
    assert!(running.contains(r#"if info.state != "ready" {"#));
    assert!(running.contains("self.ready.clear();"));
    let fill = running.find("self.ready.push(").expect("push");
    let gate = running
        .find("if self.running_up {")
        .expect("running_up gate");
    assert!(gate < fill);
    // #70: any model not `ready` is a swap in progress.
    assert!(running.contains(r#".any(|info| info.state != "ready")"#));
}

/// #70: one function makes every `/upstream/<id>/…` GET, and it reads
/// `/running` itself just before, checks the swap and the `ready` list,
/// and reads `/running` again just after for the self-check.
#[test]
fn every_upstream_get_goes_through_the_fresh_gate() {
    let poller = source("src/poller.rs");
    // The URL builder is called once, in the gate.
    assert_eq!(
        poller.matches("upstream(&").count(),
        1,
        "one upstream URL site"
    );
    assert_eq!(
        poller.matches("\"upstream/").count(),
        1,
        "one upstream path"
    );
    let gate = body(&poller, "upstream_read");
    let blocked = at(&gate, "if self.round_blocked", "gate");
    let backed = at(&gate, "self.backed_off(id)", "gate");
    let fresh = at(&gate, "self.poll_running();", "gate");
    let swap = at(&gate, "if self.swapping {", "gate");
    let ready = at(
        &gate,
        "self.ready.iter().find(|model| model.id == id)",
        "gate",
    );
    let url = at(&gate, "upstream(&self.limits.url, &model.id, leaf)", "gate");
    let get = at(&gate, "get_limited(&self.agent, &url", "gate");
    assert!(
        blocked < fresh && backed < fresh,
        "skips come before the read"
    );
    assert!(fresh < swap && swap < ready && ready < url && url < get);
    // Nothing but the `/running` read sits between the gate's check and
    // the GET: no other request.
    assert_eq!(gate[fresh..get].matches("get_").count(), 0);
    assert_eq!(gate[fresh..get].matches("self.poll_").count(), 1);
    // The check read, then the self-check, after the GET.
    let check = gate[get..].find("self.poll_running();").expect("check") + get;
    let suspect = at(&gate, "self.suspect_load(", "gate");
    assert!(check < suspect);
    assert!(gate.contains("SUSPECT_SLOW"));
    // Every upstream reader goes through the gate and makes no GET itself.
    for name in ["probe_backends", "poll_metrics", "poll_slots", "poll_loads"] {
        let text = body(&poller, name);
        assert!(text.contains("self.upstream_read("), "{name}");
        assert!(!text.contains("get_"), "{name} GETs itself");
        assert!(!text.contains("upstream(&"), "{name} builds a URL");
    }
    let suspect = body(&poller, "suspect_load");
    assert!(suspect.contains("SUSPECT_BACKOFF"));
    assert!(suspect.contains("Priority::Warning"));
}

/// #70: an upstream GET follows no redirect, so llama-swap cannot send it
/// on to another model's path.
#[test]
fn the_llama_swap_agent_follows_no_redirect() {
    let agent = body(&source("src/sources/llamaswap.rs"), "new_agent");
    assert!(agent.contains(".max_redirects(0)"), "{agent}");
    let poller = source("src/poller.rs");
    assert!(poller.contains("let agent = llamaswap::new_agent();"));
    assert_eq!(
        poller.matches("Agent::").count(),
        0,
        "one agent, from new_agent"
    );
}
