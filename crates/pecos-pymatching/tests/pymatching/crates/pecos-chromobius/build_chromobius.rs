//! Build script for Chromobius decoder integration

use pecos_build_utils::{
    Result, chromobius_download_info, download_cached, extract_archive, report_cache_config,
};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

// Use the shared modules from the parent
use crate::build_stim;
use crate::chromobius_patch;

/// Main build function for Chromobius
pub fn build() -> Result<()> {
    println!("cargo:rerun-if-changed=build_chromobius.rs");
    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    println!("cargo:rerun-if-changed=include/chromobius_bridge.h");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let chromobius_dir = out_dir.join("chromobius");

    // Always emit link directives - Cargo will cache these
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=chromobius-bridge");

    // Link C++ standard library - needed when Chromobius is used without PyMatching
    if cfg!(target_env = "msvc") {
        // MSVC automatically links the C++ runtime
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Check if the compiled library already exists
    let lib_path = out_dir.join("libchromobius-bridge.a");
    if lib_path.exists() && chromobius_dir.exists() {
        if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
            println!("cargo:warning=Chromobius library already built, skipping compilation");
        }
        return Ok(());
    }

    // Use shared Stim directory
    let stim_dir = build_stim::ensure_stim(&out_dir)?;
    let pymatching_dir = out_dir.join("PyMatching");

    // Download and extract Chromobius source if not already present
    if !chromobius_dir.exists() {
        download_and_extract_chromobius(&out_dir)?;
    }

    // Apply compatibility patches for newer Stim version
    chromobius_patch::patch_chromobius_for_newer_stim(&chromobius_dir)?;

    // Download and extract PyMatching source if not already present
    if !pymatching_dir.exists() {
        download_and_extract_pymatching(&out_dir)?;
    }

    // Build using cxx
    build_cxx_bridge(&chromobius_dir, &stim_dir, &pymatching_dir)?;

    Ok(())
}

fn download_and_extract_chromobius(out_dir: &Path) -> Result<()> {
    let info = chromobius_download_info();
    let tar_gz = download_cached(&info)?;
    extract_archive(&tar_gz, out_dir, Some("chromobius"))?;

    if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
        println!("cargo:warning=Chromobius source downloaded and extracted");
    }
    Ok(())
}

fn download_and_extract_pymatching(out_dir: &Path) -> Result<()> {
    let info = pecos_build_utils::pymatching_download_info();
    let tar_gz = download_cached(&info)?;
    extract_archive(&tar_gz, out_dir, Some("PyMatching"))?;

    if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
        println!("cargo:warning=PyMatching source downloaded and extracted");
    }
    Ok(())
}

fn build_cxx_bridge(chromobius_dir: &Path, stim_dir: &Path, pymatching_dir: &Path) -> Result<()> {
    let chromobius_src_dir = chromobius_dir.join("src");
    let stim_src_dir = stim_dir.join("src");
    let pymatching_src_dir = pymatching_dir.join("src");

    // Find essential source files
    let chromobius_files = collect_chromobius_sources(&chromobius_src_dir)?;
    let stim_files = collect_stim_sources(&stim_src_dir)?;
    let pymatching_files = collect_pymatching_sources(&pymatching_src_dir)?;

    // Build the cxx bridge first to generate headers
    let mut build = cxx_build::bridge("src/bridge.rs");

    // Add our bridge implementation
    build.file("src/bridge.cpp");

    // Add Chromobius core files
    for file in chromobius_files {
        build.file(file);
    }

    // Add PyMatching files
    for file in pymatching_files {
        build.file(file);
    }

    // Configure build
    build
        .std("c++20")
        .include(chromobius_src_dir)
        .include(stim_src_dir)
        .include(stim_dir) // For amalgamated stim.h
        .include(pymatching_src_dir)
        .include("include")
        .include("src")
        .define("CHROMOBIUS_BRIDGE_EXPORTS", None); // Define export macro

    // Report ccache/sccache configuration
    report_cache_config();

    // Use different optimization levels for debug vs release builds
    if cfg!(debug_assertions) {
        build.flag_if_supported("-O0"); // No optimization for faster compilation
        build.flag_if_supported("-g"); // Include debug symbols
    } else {
        build.flag_if_supported("-O3"); // Full optimization for release
    }

    // Hide all symbols by default
    if cfg!(not(target_env = "msvc")) {
        build.flag("-fvisibility=hidden");
        build.flag("-fvisibility-inlines-hidden");
    }

    // Only use -march=native if not cross-compiling and not explicitly disabled
    if env::var("CARGO_CFG_TARGET_ARCH").ok() == env::var("HOST_ARCH").ok()
        && env::var("DECODER_DISABLE_NATIVE_ARCH").is_err()
    {
        build.flag_if_supported("-march=native");
    }

    // Platform-specific configurations
    if cfg!(not(target_env = "msvc")) {
        // For GCC/Clang
        build
            .flag("-w") // Suppress all warnings from external code
            .flag_if_supported("-fopenmp") // Enable OpenMP if available
            .flag("-fPIC"); // Position independent code for shared library
    } else {
        // For MSVC
        build
            .flag("/W0") // Warning level 0 (no warnings)
            .flag_if_supported("/openmp"); // Enable OpenMP if available
    }

    // Add Stim files to the main build
    for file in &stim_files {
        build.file(file);
    }

    // Build everything together
    build.compile("chromobius-bridge");

    Ok(())
}

fn collect_chromobius_sources(chromobius_src_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    // Collect all non-test, non-perf, non-pybind .cc files
    collect_cc_files_filtered(chromobius_src_dir, &mut files)?;

    if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
        println!(
            "cargo:warning=Found {} Chromobius source files",
            files.len()
        );
    }
    Ok(files)
}

fn collect_stim_sources(stim_src_dir: &Path) -> Result<Vec<PathBuf>> {
    // Use Chromobius-specific Stim sources
    build_stim::collect_stim_sources_chromobius(stim_src_dir)
}

fn collect_pymatching_sources(pymatching_src_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    // PyMatching sparse_blossom implementation files
    let sparse_blossom_dir = pymatching_src_dir.join("pymatching/sparse_blossom");
    if sparse_blossom_dir.exists() {
        collect_cc_files_filtered(&sparse_blossom_dir, &mut files)?;
    }

    if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
        println!(
            "cargo:warning=Found {} PyMatching source files",
            files.len()
        );
    }
    Ok(files)
}

fn collect_cc_files_filtered(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Skip test directories
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name == "test" || name == "tests" {
                    continue;
                }
            }
            collect_cc_files_filtered(&path, files)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("cc") {
            let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Skip test, perf, pybind, and main files
            if filename.contains(".test.")
                || filename.contains(".perf.")
                || filename.contains(".pybind.")
                || filename == "main.cc"
            {
                continue;
            }
            if !files.contains(&path) {
                files.push(path);
            }
        }
    }

    Ok(())
}
