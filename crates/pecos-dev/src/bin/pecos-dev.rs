//! PECOS developer tools command-line interface

fn main() {
    env_logger::init();

    if let Err(e) = pecos_dev::cli::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
