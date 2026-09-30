#![forbid(unsafe_code)]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // The unit's root pre-step (#7): font and size of tty11, then exit.
    if let Some(("tty-setup", rest)) = args.split_first().map(|(cmd, rest)| (cmd.as_str(), rest)) {
        std::process::exit(llama_watch::tty::setup::main(rest));
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
