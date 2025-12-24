//! Build script for pecos-tesseract

mod build_stim;
mod build_tesseract;

fn main() {
    // Download dependencies using shared utilities
    let mut downloads = Vec::new();

    // Stim dependency (Tesseract-specific version)
    downloads.push(pecos_build_utils::stim_download_info("tesseract"));

    // Tesseract dependency
    downloads.push(pecos_build_utils::tesseract_download_info());

    // Download if needed
    if let Err(e) = pecos_build_utils::download_all_cached(downloads) {
        if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
            println!("cargo:warning=Download failed: {e}, continuing with build");
        }
    }

    // Build Tesseract
    build_tesseract::build().expect("Tesseract build failed");
}
