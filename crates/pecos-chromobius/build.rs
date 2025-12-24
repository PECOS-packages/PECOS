//! Build script for pecos-chromobius

mod build_chromobius;
mod build_stim;
mod chromobius_patch;

fn main() {
    // Initialize logger for build script
    env_logger::init();

    // Build Chromobius (download handled inside build_chromobius)
    build_chromobius::build().expect("Chromobius build failed");
}
