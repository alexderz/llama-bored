use std::path::PathBuf;
use std::process::ExitCode;

use llama_core::log::{self, Priority};
use llama_light::aura::AuraBackend;
use llama_light::aura::device::HidrawOpener;
use llama_light::backend::Backend;
use llama_light::config::{self, ConfigFile};
use llama_light::keyboard::KeyboardBackend;
use llama_light::keyboard::device::KeyboardOpenerHidraw;
use llama_light::service::{self, HostClock, Light, Parts, RunEnd, SdNotify};
use llama_light::snapshot::SnapshotFile;

const USAGE: &str = "usage: llama-light <run|restore|check> --config PATH";
/// Device and sysfs roots. The unit allows only /dev/llama-light/aura and
/// /dev/llama-light/keyboard.
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
    // Always built, so a reload can turn the keyboard on. The opener notes
    // which node the pin names now, when the unit's device list was made.
    let keyboard: Option<Box<dyn Backend>> = Some(Box::new(KeyboardBackend::new(
        KeyboardOpenerHidraw::new(DEV_ROOT, SYS_ROOT),
    )));
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
    match light.run(None) {
        // Restart=on-failure brings the unit back with the new node allowed.
        RunEnd::Restart => ExitCode::from(3),
        RunEnd::Ticks => ExitCode::SUCCESS,
    }
}

fn restore(path: PathBuf) -> ExitCode {
    let mut sink = log::Stderr;
    // An unusable config still hands the keyboard back: it may have been
    // on under the config that was running.
    let (config, keyboard_on) = match config::load(&path) {
        Ok(config) => {
            let on = config.keyboard.enabled;
            (config, on)
        }
        Err(err) => {
            log::emit(
                &mut sink,
                Priority::Warning,
                &format!("restore: config unusable ({err}); using defaults"),
            );
            (config::LightConfig::default(), true)
        }
    };
    let mut aura = AuraBackend::new(HidrawOpener::new(DEV_ROOT, SYS_ROOT));
    let mut keyboard = KeyboardBackend::new(KeyboardOpenerHidraw::new(DEV_ROOT, SYS_ROOT));
    let keyboard: Option<&mut dyn Backend> = if keyboard_on {
        Some(&mut keyboard)
    } else {
        None
    };
    ExitCode::from(service::restore(&config, &mut aura, keyboard, &mut sink))
}

/// Validate without touching any device.
fn check(path: PathBuf) -> ExitCode {
    match config::load(&path) {
        Ok(config) => {
            eprintln!(
                "{}: ok ({} light entr{}, {} LEDs per frame, {} fps, brightness cap {} %; keyboard {})",
                path.display(),
                config.layers.len(),
                if config.layers.len() == 1 { "y" } else { "ies" },
                config.aura.frame_len(),
                config.aura.fps,
                config.aura.brightness_max,
                if config.keyboard.enabled {
                    format!("on, brightness cap {} %", config.keyboard.brightness_max)
                } else {
                    "off".to_owned()
                }
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}
