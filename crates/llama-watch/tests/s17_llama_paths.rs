//! S17: the watcher's llama-swap reads are a fixed set of GETs on the
//! configured loopback URL (`llama.url` must be a loopback IP literal):
//! `/running`, `/upstream/<model>/metrics`, `/upstream/<model>/slots`,
//! `/api/metrics/activity` and, since #5, `/api/captures/<id>`. A new
//! endpoint fails this scan until it is added here on purpose. #31's
//! backend detection reuses `/upstream/<model>/metrics`: no new path.
//! `tests/s31_ready_gate.rs` pins at runtime that no upstream path is
//! requested for a model `/running` does not list as `ready`.

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
    let mut leaves: Vec<String> = call_literals(&poller, "upstream")
        .into_iter()
        .flatten()
        .collect();
    leaves.sort();
    leaves.dedup();
    assert_eq!(leaves, vec!["metrics".to_owned(), "slots".to_owned()]);

    let running = source("src/sources/llamaswap.rs");
    assert!(running.contains(r#"format!("{}/running""#));

    // One GET call site each, and no other HTTP verb anywhere.
    assert_eq!(poller.matches(".get(url)").count(), 1, "poller GET sites");
    assert_eq!(running.matches(".get(&endpoint)").count(), 1);
    for rel in [
        "src/poller.rs",
        "src/sources/llamaswap.rs",
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

#[test]
fn upstream_reads_come_from_the_ready_list_only() {
    let poller = source("src/poller.rs");
    let body = |name: &str| -> String {
        poller
            .split(&format!("fn {name}"))
            .nth(1)
            .unwrap_or_else(|| panic!("{name}"))
            .split("\n    fn ")
            .next()
            .expect("body")
            .to_owned()
    };
    // Only `ready` entries of a good `/running` read join the list.
    let running = body("poll_running");
    assert!(running.contains(r#"if info.state != "ready" {"#));
    assert!(running.contains("self.ready.clear();"));
    let fill = running.find("self.ready.push(").expect("push");
    let gate = running
        .find("if self.running_up {")
        .expect("running_up gate");
    assert!(gate < fill);
    // #31: the probe walks that list before its one GET, at the server cap.
    let probe = body("probe_backends");
    let from_ready = probe
        .find("self\n            .ready")
        .expect("probe reads self.ready");
    let get = probe.find("get_exact(").expect("GET");
    assert!(from_ready < get);
    assert!(probe.contains("SERVER_METRICS_CAP"));
    assert!(probe.contains("upstream(&self.limits.url, &id, \"metrics\")"));
    // Metrics and slots GETs, likewise, start from `self.ready`.
    for name in ["poll_metrics", "poll_slots"] {
        let text = body(name);
        let from_ready = text.find(".ready").unwrap_or(usize::MAX);
        let get = text.find("get_").expect("GET");
        assert!(from_ready < get, "{name}");
    }
}
