//! Build script for pecos-quest

mod build_quest;

fn main() {
    // Initialize logger for build script
    env_logger::init();

    // Build QuEST (download handled inside build_quest)
    build_quest::build().expect("QuEST build failed");
}
