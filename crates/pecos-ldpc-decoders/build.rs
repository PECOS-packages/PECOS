//! Build script for pecos-ldpc-decoders

mod build_ldpc;

fn main() {
    // Initialize logger for build script
    env_logger::init();

    // Build LDPC (download handled inside build_ldpc)
    build_ldpc::build().expect("LDPC build failed");
}
