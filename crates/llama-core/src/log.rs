//! stderr lines with sd-daemon `<N>` priority prefixes.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Priority {
    Crit,
    Err,
    Warning,
    Info,
    Debug,
}

pub fn prefix(priority: Priority) -> &'static str {
    match priority {
        Priority::Crit => "<2>",
        Priority::Err => "<3>",
        Priority::Warning => "<4>",
        Priority::Info => "<6>",
        Priority::Debug => "<7>",
    }
}

pub fn format_line(priority: Priority, message: &str) -> String {
    format!("{}{message}", prefix(priority))
}

pub trait Sink {
    fn write_line(&mut self, line: &str);
}

pub struct Stderr;

impl Sink for Stderr {
    fn write_line(&mut self, line: &str) {
        eprintln!("{line}");
    }
}

pub fn emit(sink: &mut impl Sink, priority: Priority, message: &str) {
    sink.write_line(&format_line(priority, message));
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Capture(Vec<String>);

    impl Sink for Capture {
        fn write_line(&mut self, line: &str) {
            self.0.push(line.to_owned());
        }
    }

    #[test]
    fn prefixes_match_sd_daemon_priorities() {
        let cases = [
            (Priority::Crit, "<2>"),
            (Priority::Err, "<3>"),
            (Priority::Warning, "<4>"),
            (Priority::Info, "<6>"),
            (Priority::Debug, "<7>"),
        ];
        for (priority, expected) in cases {
            assert_eq!(prefix(priority), expected, "{priority:?}");
        }
    }

    #[test]
    fn format_line_puts_prefix_before_message() {
        assert_eq!(format_line(Priority::Err, "boom"), "<3>boom");
    }

    #[test]
    fn emit_writes_formatted_line_to_sink() {
        let mut sink = Capture(Vec::new());
        emit(&mut sink, Priority::Info, "ready");
        assert_eq!(sink.0, ["<6>ready"]);
    }
}
