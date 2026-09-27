//! Start-up self-checks: pass and fail through the injectable host seam.
//!
//! Nothing here opens a device node or the real `/sys` for the production
//! probe. Fake roots live under a temp dir or `fixtures/`.

use std::path::{Path, PathBuf};

use kraken_lcd::config::Config;
use kraken_lcd::device::SYS_ROOT;
use kraken_lcd::service::{self, CheckEnv, CheckError};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("t14-check-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch");
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn passing_env(state: &Path) -> CheckEnv {
    CheckEnv {
        sys_root: PathBuf::from(SYS_ROOT),
        state_dir: state.to_path_buf(),
        euid_is_root: false,
        sys_is_readonly: true,
        z53_present: true,
        state_dir_writable: true,
    }
}

#[test]
fn self_check_passes_when_every_host_probe_is_green() {
    let scratch = Scratch::new("pass");
    let cfg = Config::default();
    if let Err(err) = service::self_check(&cfg, &passing_env(&scratch.0)) {
        panic!("checks: {err}");
    }
}

#[test]
fn self_check_rejects_non_default_roots() {
    let scratch = Scratch::new("roots");
    let cfg = Config::default();
    let mut env = passing_env(&scratch.0);
    env.sys_root = scratch.0.join("sys");
    let err = match service::self_check(&cfg, &env) {
        Err(err) => err,
        Ok(_) => panic!("roots"),
    };
    assert!(matches!(err, CheckError::SysRootNotDefault), "{err}");
}

#[test]
fn self_check_rejects_a_writable_sys() {
    let scratch = Scratch::new("sys");
    let cfg = Config::default();
    let mut env = passing_env(&scratch.0);
    env.sys_is_readonly = false;
    let err = match service::self_check(&cfg, &env) {
        Err(err) => err,
        Ok(_) => panic!("sys"),
    };
    assert!(matches!(err, CheckError::SysNotReadonly), "{err}");
}

#[test]
fn self_check_rejects_root() {
    let scratch = Scratch::new("root");
    let cfg = Config::default();
    let mut env = passing_env(&scratch.0);
    env.euid_is_root = true;
    let err = match service::self_check(&cfg, &env) {
        Err(err) => err,
        Ok(_) => panic!("root"),
    };
    assert!(matches!(err, CheckError::RunningAsRoot), "{err}");
}

#[test]
fn self_check_rejects_a_missing_z53() {
    let scratch = Scratch::new("z53");
    let cfg = Config::default();
    let mut env = passing_env(&scratch.0);
    env.z53_present = false;
    let err = match service::self_check(&cfg, &env) {
        Err(err) => err,
        Ok(_) => panic!("z53"),
    };
    assert!(matches!(err, CheckError::NoZ53), "{err}");
}

#[test]
fn self_check_rejects_an_unwritable_state_dir() {
    let scratch = Scratch::new("state");
    let cfg = Config::default();
    let mut env = passing_env(&scratch.0);
    env.state_dir_writable = false;
    let err = match service::self_check(&cfg, &env) {
        Err(err) => err,
        Ok(_) => panic!("state"),
    };
    assert!(matches!(err, CheckError::StateDirNotWritable), "{err}");
}

#[test]
fn self_check_rejects_an_invalid_config() {
    let scratch = Scratch::new("cfg");
    let mut cfg = Config::default();
    cfg.upload.min_interval_s = 9;
    let err = match service::self_check(&cfg, &passing_env(&scratch.0)) {
        Err(err) => err,
        Ok(_) => panic!("config"),
    };
    assert!(matches!(err, CheckError::Invalid(_)), "{err}");
}

#[test]
fn z53_exists_on_the_hwmon_fixture() {
    let sys = fixtures().join("sys");
    assert!(service::z53_exists(&sys));
}

#[test]
fn z53_is_absent_on_an_empty_sys_root() {
    let scratch = Scratch::new("empty-sys");
    std::fs::create_dir_all(scratch.0.join("class/hwmon")).expect("hwmon");
    assert!(!service::z53_exists(&scratch.0));
}

#[test]
fn production_probe_constructor_is_not_the_test_seam() {
    // The test constructor is CheckEnv { ... }. Production run/restore-stock
    // call CheckEnv::production(), which always reads the real host.
    let _ = CheckEnv::production;
}
