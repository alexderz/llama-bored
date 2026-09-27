#![forbid(unsafe_code)]

fn main() {
    let code = match llama_metrics::service::parse_args(std::env::args().skip(1)) {
        Ok(command) => llama_metrics::service::execute(&command),
        Err(usage) => {
            eprintln!("{usage}");
            llama_metrics::service::EXIT_CONFIG
        }
    };
    std::process::exit(code);
}
