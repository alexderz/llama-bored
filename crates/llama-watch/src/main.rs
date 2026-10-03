#![forbid(unsafe_code)]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // The unit's root pre-step (#7): font and size of tty11, then exit.
    if let Some(("tty-setup", rest)) = args.split_first().map(|(cmd, rest)| (cmd.as_str(), rest)) {
        std::process::exit(llama_watch::tty::setup::main(rest));
    }
    // The unit's `ExecStopPost=` (#26): `ESC ] R` on tty11, reads nothing.
    if args.first().map(String::as_str) == Some("tty-reset") {
        if args.len() != 1 {
            eprintln!("usage: llama-watch tty-reset");
            std::process::exit(2);
        }
        let code = match llama_watch::tty::term::reset_palette_on_stdout() {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("tty-reset: {err}");
                1
            }
        };
        std::process::exit(code);
    }
    let code = match llama_watch::service::parse_args(args) {
        Ok(args) => llama_watch::service::run(&args),
        Err(err) => {
            eprintln!("{err}");
            2
        }
    };
    std::process::exit(code);
}
