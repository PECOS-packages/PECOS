//! Build script for Tesseract decoder integration

use pecos_build_utils::{
    Result, download_cached, extract_archive, report_cache_config, tesseract_download_info,
};
use std::env;
use std::path::{Path, PathBuf};

// Use the shared modules from the parent
use crate::build_stim;

/// Main build function for Tesseract
pub fn build() -> Result<()> {
    println!("cargo:rerun-if-changed=build_tesseract.rs");
    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    println!("cargo:rerun-if-changed=include/tesseract_bridge.h");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let tesseract_dir = out_dir.join("tesseract-decoder");

    // Always emit link directives - Cargo will cache these
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=tesseract-bridge");

    // Link C++ standard library
    if cfg!(target_env = "msvc") {
        // MSVC automatically links the C++ runtime
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Check if the compiled library already exists
    let lib_path = out_dir.join("libtesseract-bridge.a");
    if lib_path.exists() && tesseract_dir.exists() {
        if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
            println!("cargo:warning=Tesseract library already built, skipping compilation");
        }
        return Ok(());
    }

    // Use shared Stim directory
    let stim_dir = build_stim::ensure_stim(&out_dir)?;

    // Download and extract Tesseract source if not already present
    if !tesseract_dir.exists() {
        download_and_extract_tesseract(&out_dir)?;
    }

    // Build using cxx
    build_cxx_bridge(&tesseract_dir, &stim_dir)?;

    Ok(())
}

fn download_and_extract_tesseract(out_dir: &Path) -> Result<()> {
    let info = tesseract_download_info();

    let tar_gz = download_cached(&info)?;
    extract_archive(&tar_gz, out_dir, Some("tesseract-decoder"))?;

    if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
        println!("cargo:warning=Tesseract source ready");
    }
    Ok(())
}

fn build_cxx_bridge(tesseract_dir: &Path, stim_dir: &Path) -> Result<()> {
    let tesseract_src_dir = tesseract_dir.join("src");
    let stim_src_dir = stim_dir.join("src");

    // Find essential Stim source files for DEM functionality
    let stim_files = collect_minimal_stim_sources(&stim_src_dir)?;

    // Build everything together
    let mut build = cxx_build::bridge("src/bridge.rs");

    // Add our bridge implementation
    build.file("src/bridge.cpp");

    // Add Tesseract core files
    build
        .file(tesseract_src_dir.join("common.cc"))
        .file(tesseract_src_dir.join("utils.cc"))
        .file(tesseract_src_dir.join("tesseract.cc"));

    // Configure build
    build
        .std("c++20")
        .include(&tesseract_src_dir)
        .include(&stim_src_dir)
        .include("include")
        .include("src")
        .define("TESSERACT_BRIDGE_EXPORTS", None); // Define export macro

    // Report ccache/sccache configuration
    report_cache_config();

    // Use different optimization levels for debug vs release builds
    if cfg!(debug_assertions) {
        build.flag_if_supported("-O0"); // No optimization for faster compilation
        build.flag_if_supported("-g"); // Include debug symbols
    } else {
        build.flag_if_supported("-O3"); // Full optimization for release
    }

    // Add Stim files to the build
    for file in stim_files {
        build.file(file);
    }

    // Hide all symbols by default
    if cfg!(not(target_env = "msvc")) {
        build.flag("-fvisibility=hidden");
        build.flag("-fvisibility-inlines-hidden");
    }

    // Only use -march=native if not cross-compiling
    if env::var("CARGO_CFG_TARGET_ARCH").ok() == env::var("HOST_ARCH").ok()
        && env::var("DECODER_DISABLE_NATIVE_ARCH").is_err()
    {
        build.flag_if_supported("-march=native");
    }

    // Platform-specific configurations
    if cfg!(not(target_env = "msvc")) {
        build
            .flag("-w") // Suppress all warnings from external code
            .flag_if_supported("-fopenmp") // Enable OpenMP if available
            .flag("-fPIC"); // Position independent code
    } else {
        build
            .flag("/W0") // Warning level 0 (no warnings)
            .flag_if_supported("/openmp"); // Enable OpenMP if available
    }

    // Build everything together
    build.compile("tesseract-bridge");

    Ok(())
}

fn collect_minimal_stim_sources(stim_src_dir: &Path) -> Result<Vec<PathBuf>> {
    // Use Tesseract-specific minimal Stim sources
    build_stim::collect_stim_sources_tesseract(stim_src_dir)
}
