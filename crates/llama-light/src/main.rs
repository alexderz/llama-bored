use std::path::PathBuf;
use std::process::ExitCode;

use llama_core::log::{self, Priority};
use llama_light::aura::AuraBackend;
use llama_light::aura::device::HidrawOpener;
use llama_light::config::{self, ConfigFile};
use llama_light::keyboard::KeyboardStub;
use llama_light::service::{self, HostClock, Light, Parts, SdNotify};
use llama_light::snapshot::SnapshotFile;

const USAGE: &str = "usage: llama-light <run|restore|check> --config PATH";
/// Device and sysfs roots. The unit allows only /dev/llama-light/aura.
const DEV_ROOT: &str = "/dev";
const SYS_ROOT: &str = "/sys";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (command, path) = match args.as_slice() {
        [command, flag, path] if flag == "--config" => (command.as_str(), PathBuf::from(path)),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match command {
        "run" => run(path),
        "restore" => restore(path),
        "check" => check(path),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn run(path: PathBuf) -> ExitCode {
    let mut sink = log::Stderr;
    let config = match config::load(&path) {
        Ok(config) => config,
        Err(err) => {
            log::emit(&mut sink, Priority::Err, &format!("config: {err}"));
            return ExitCode::from(2);
        }
    };
    let keyboard: Option<Box<dyn llama_light::backend::Backend>> = if config.keyboard_enabled {
        log::emit(
            &mut sink,
            Priority::Info,
            "keyboard: detection only; the keyboard protocol is not implemented, nothing is written",
        );
        Some(Box::new(KeyboardStub::new(SYS_ROOT)))
    } else {
        None
    };
    let mut light = Light::new(Parts {
        clock: HostClock,
        snapshots: SnapshotFile::published(),
        config_source: ConfigFile::new(path),
        notify: SdNotify,
        sink,
        config,
        aura: Some(Box::new(AuraBackend::new(HidrawOpener::new(
            DEV_ROOT, SYS_ROOT,
        )))),
        keyboard,
    });
    light.run(None);
    ExitCode::SUCCESS
}

fn restore(path: PathBuf) -> ExitCode {
    let mut sink = log::Stderr;
    let config = match config::load(&path) {
        Ok(config) => config,
        Err(err) => {
            log::emit(
                &mut sink,
                Priority::Warning,
                &format!("restore: config unusable ({err}); using defaults"),
            );
            config::LightConfig::default()
        }
    };
    let mut aura = AuraBackend::new(HidrawOpener::new(DEV_ROOT, SYS_ROOT));
    ExitCode::from(service::restore(&config, &mut aura, &mut sink))
}

/// Validate without touching any device.
fn check(path: PathBuf) -> ExitCode {
    match config::load(&path) {
        Ok(config) => {
            eprintln!(
                "{}: ok ({} light entr{}, {} LEDs per frame, {} fps, brightness cap {} %)",
                path.display(),
                config.layers.len(),
                if config.layers.len() == 1 { "y" } else { "ies" },
                config.aura.frame_len(),
                config.aura.fps,
                config.aura.brightness_max
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}
