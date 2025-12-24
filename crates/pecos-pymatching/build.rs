//! Build script for pecos-pymatching

mod build_pymatching;
mod build_stim;

fn main() {
    // Initialize logger for build script
    env_logger::init();

    // Build PyMatching (download handled inside build_pymatching)
    build_pymatching::build().expect("PyMatching build failed");
}
