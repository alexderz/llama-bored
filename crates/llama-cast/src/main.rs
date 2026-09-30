#![forbid(unsafe_code)]

fn main() {
    let code = match llama_cast::service::parse_args(std::env::args().skip(1)) {
        Ok(command) => llama_cast::service::execute(&command),
        Err(usage) => {
            eprintln!("{usage}");
            llama_cast::service::EXIT_CONFIG
        }
    };
    std::process::exit(code);
}
