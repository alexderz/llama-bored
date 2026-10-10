//! `--kiosk` (#104): on a dedicated console the vcsa device can be missing
//! for a moment (an llama-watch restart deallocates tty11 briefly), and
//! llama-view must not exit for that — it shows a status line and keeps
//! running so kmscon never has to respawn it.

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[test]
fn kiosk_waits_for_a_missing_source_instead_of_exiting() {
    let dir = std::env::temp_dir().join(format!("llama-view-kiosk-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    // Never created: opening it fails with ENOENT, the condition #104 is for.
    let missing = dir.join("vcsa11");

    let mut child = Command::new(env!("CARGO_BIN_EXE_llama-view"))
        .args(["--kiosk", "--device", missing.to_str().expect("utf-8 path")])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");

    // Collect stdout on a thread: the child runs forever (that is the
    // point), so the test reads what it has written so far rather than
    // waiting for EOF.
    let mut stdout = child.stdout.take().expect("stdout");
    let seen = Arc::new(Mutex::new(Vec::new()));
    let writer = Arc::clone(&seen);
    let reader = std::thread::spawn(move || {
        let mut chunk = [0u8; 4096];
        loop {
            match stdout.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => writer.lock().expect("lock").extend_from_slice(&chunk[..n]),
            }
        }
    });

    std::thread::sleep(Duration::from_millis(600));
    assert!(
        child.try_wait().expect("try_wait").is_none(),
        "llama-view exited instead of waiting for the missing device"
    );

    let out = seen.lock().expect("lock").clone();
    assert!(
        out.windows(b"waiting for".len())
            .any(|w| w == b"waiting for"),
        "no waiting line in {:?}",
        String::from_utf8_lossy(&out)
    );
    let path = missing.to_str().expect("utf-8 path").as_bytes();
    assert!(
        out.windows(path.len()).any(|w| w == path),
        "the device path is missing from the status line: {:?}",
        String::from_utf8_lossy(&out)
    );

    child.kill().expect("kill");
    let _ = child.wait();
    let _ = reader.join();
    let _ = std::fs::remove_dir_all(&dir);
}
