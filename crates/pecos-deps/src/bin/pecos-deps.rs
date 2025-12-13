//! PECOS dependency management CLI

fn main() {
    env_logger::init();

    if let Err(e) = pecos_deps::cli::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
