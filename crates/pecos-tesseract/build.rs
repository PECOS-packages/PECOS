//! Build script for pecos-tesseract

mod build_stim;
mod build_tesseract;

fn main() {
    // Initialize logger for build script
    env_logger::init();

    // Build Tesseract (download handled inside build_tesseract)
    build_tesseract::build().expect("Tesseract build failed");
}
