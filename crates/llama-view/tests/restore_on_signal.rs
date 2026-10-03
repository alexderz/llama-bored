//! The binary restores the terminal when `--once` returns.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use llama_view::{ENTER, FOCUS_ON, RESTORE, SYNC_BEGIN, SYNC_END};

#[test]
fn once_restores_the_terminal_on_exit() {
    let dir = std::env::temp_dir().join(format!("llama-view-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let vcsa = dir.join("vcsa11");
    // ESC, C1, DEL in the glyph bytes. The child must not copy them out.
    let mut bytes = vec![1u8, 3, 0, 0];
    for ch in [0x1Bu8, 0x9B, 0x7F] {
        bytes.push(ch);
        bytes.push(0x07);
    }
    std::fs::write(&vcsa, &bytes).expect("fixture");

    let mut child = Command::new(env!("CARGO_BIN_EXE_llama-view"))
        .args(["--device", vcsa.to_str().expect("utf-8 path"), "--once"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");

    let status = wait_deadline(&mut child, Duration::from_secs(2));
    let mut out = Vec::new();
    child
        .stdout
        .take()
        .expect("stdout")
        .read_to_end(&mut out)
        .expect("read stdout");
    let mut err = String::new();
    child
        .stderr
        .take()
        .expect("stderr")
        .read_to_string(&mut err)
        .expect("read stderr");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(status.success(), "status {status}, stderr {err}");
    assert!(
        out.windows(ENTER.len()).any(|w| w == ENTER),
        "missing enter: {out:?}"
    );
    assert!(
        out.windows(RESTORE.len()).any(|w| w == RESTORE),
        "missing restore: {out:?}"
    );
    let enter_at = find(&out, ENTER);
    let restore_at = find_last(&out, RESTORE);
    assert!(enter_at < restore_at, "restore was not after enter");
    assert!(!out.contains(&0x7F), "DEL reached stdout");
    // #27: the frame is synchronized, the pane gets a title (ST, no BEL),
    // and focus reporting stays off when stdin is not a terminal.
    assert!(find(&out, SYNC_BEGIN) < find_last(&out, SYNC_END));
    assert!(
        out.windows(9).any(|w| w == b"\x1b]2;llama"),
        "title: {out:?}"
    );
    assert!(!out.contains(&0x07), "no bell");
    assert!(!out.windows(FOCUS_ON.len()).any(|w| w == FOCUS_ON));
    assert!(
        !out.windows(2).any(|pair| pair == [0xC2, 0x9B]),
        "C1 CSI encoding reached stdout: {out:?}"
    );
}

fn wait_deadline(child: &mut std::process::Child, limit: Duration) -> std::process::ExitStatus {
    let deadline = Instant::now() + limit;
    loop {
        if let Some(status) = child.try_wait().expect("wait") {
            return status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("llama-view did not exit after SIGTERM");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn find(hay: &[u8], needle: &[u8]) -> usize {
    hay.windows(needle.len())
        .position(|w| w == needle)
        .expect("needle")
}

fn find_last(hay: &[u8], needle: &[u8]) -> usize {
    hay.windows(needle.len())
        .rposition(|w| w == needle)
        .expect("needle")
}
