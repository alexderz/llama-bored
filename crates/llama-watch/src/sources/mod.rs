pub mod cmdline;
pub mod fans;
pub mod gpu;
pub mod hwmon;
pub mod llamaswap;
pub mod proc;

use std::path::PathBuf;

pub use llama_core::sample::{SourceError, SourceId};

/// Proc and sys roots for the watcher. The writer does not use this type:
/// its default points at `/proc`, which the writer must not read.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Roots {
    /// `/proc` in production. Tests pass a fixture directory.
    pub proc: PathBuf,
    /// `/sys` in production. Tests pass a fixture directory.
    pub sys: PathBuf,
}

impl Default for Roots {
    fn default() -> Self {
        Self {
            proc: PathBuf::from("/proc"),
            sys: PathBuf::from("/sys"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn source_ids_display_as_lld_names() {
        let cases = [
            (SourceId::ProcCpu, "proc.cpu"),
            (SourceId::ProcMem, "proc.mem"),
            (SourceId::HwmonCoolant, "hwmon.coolant"),
            (SourceId::HwmonCpu, "hwmon.cpu"),
            (SourceId::Gpu, "gpu"),
            (SourceId::Llama, "llama"),
        ];
        for (id, name) in cases {
            assert_eq!(id.as_str(), name);
            assert_eq!(id.to_string(), name);
        }
    }

    #[test]
    fn default_roots_are_proc_and_sys() {
        let roots = Roots::default();
        assert_eq!(roots.proc.as_path(), Path::new("/proc"));
        assert_eq!(roots.sys.as_path(), Path::new("/sys"));
    }

    #[test]
    fn source_error_is_tagged_with_source_id() {
        let err = SourceError::new(SourceId::ProcMem, "missing MemTotal");
        assert_eq!(err.id, SourceId::ProcMem);
        assert_eq!(err.to_string(), "proc.mem: missing MemTotal");
    }
}
