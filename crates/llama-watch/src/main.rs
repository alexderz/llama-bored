#![forbid(unsafe_code)]

fn main() {
    let code = match llama_watch::service::parse_args(std::env::args().skip(1)) {
        Ok(args) => llama_watch::service::run(&args),
        Err(err) => {
            eprintln!("{err}");
            2
        }
    };
    std::process::exit(code);
}
