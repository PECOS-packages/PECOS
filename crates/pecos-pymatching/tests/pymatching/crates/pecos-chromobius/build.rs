//! Build script for pecos-chromobius

mod build_chromobius;
mod build_stim;
mod chromobius_patch;

fn main() {
    // Download dependencies using shared utilities
    let mut downloads = Vec::new();

    // Stim dependency
    downloads.push(pecos_build_utils::stim_download_info("chromobius"));

    // Chromobius dependency
    downloads.push(pecos_build_utils::chromobius_download_info());

    // PyMatching dependency (shared with Chromobius)
    downloads.push(pecos_build_utils::pymatching_download_info());

    // Download if needed
    if let Err(e) = pecos_build_utils::download_all_cached(downloads) {
        if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
            println!("cargo:warning=Download failed: {e}, continuing with build");
        }
    }

    // Build Chromobius
    build_chromobius::build().expect("Chromobius build failed");
}
