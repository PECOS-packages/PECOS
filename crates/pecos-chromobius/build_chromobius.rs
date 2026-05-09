//! Build script for Chromobius decoder integration

use log::info;
use pecos_build::{Manifest, Result, check_cxx20_toolchain, ensure_dep_ready, report_cache_config};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

// Use the shared modules from the parent
use crate::build_stim;
use crate::chromobius_patch;

/// Get the build profile from Cargo's environment
fn get_build_profile() -> String {
    if let Ok(out_dir) = env::var("OUT_DIR") {
        let parts: Vec<&str> = out_dir.split(std::path::MAIN_SEPARATOR).collect();
        if let Some(target_idx) = parts.iter().position(|&p| p == "target")
            && let Some(profile_name) = parts.get(target_idx + 1)
        {
            return match *profile_name {
                "native" => "native",
                "release" => "release",
                "debug" => "debug",
                _ => {
                    if env::var("PROFILE").as_deref() == Ok("release") {
                        "release"
                    } else {
                        "debug"
                    }
                }
            }
            .to_string();
        }
    }

    match env::var("PROFILE").as_deref() {
        Ok("release") => "release".to_string(),
        _ => "debug".to_string(),
    }
}

/// Main build function for Chromobius
pub fn build() -> Result<()> {
    check_cxx20_toolchain();

    println!("cargo:rerun-if-changed=build_chromobius.rs");
    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    println!("cargo:rerun-if-changed=include/chromobius_bridge.h");
    println!("cargo:rerun-if-env-changed=FORCE_REBUILD");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    // Always emit link directives - Cargo will cache these
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=chromobius-bridge");

    // Get dependencies (downloads to ~/.pecos/cache/, extracts to ~/.pecos/deps/)
    let manifest = Manifest::find_and_load_validated()?;
    let chromobius_dir = ensure_dep_ready("chromobius", &manifest)?;
    let stim_dir = ensure_dep_ready("stim", &manifest)?;
    // PyMatching headers and compiled objects come from pecos-pymatching via cargo metadata.
    // This avoids compiling a second copy of PyMatching sources which would cause
    // duplicate symbol errors at link time.
    let pymatching_include = env::var("DEP_PYMATCHING_PECOS_PYMATCHING_INCLUDE").map_or_else(
        |_| {
            // Fallback: download and use directly (for standalone builds)
            ensure_dep_ready("pymatching", &manifest)
                .expect("pymatching dependency")
                .join("src")
        },
        PathBuf::from,
    );
    let stim_include = env::var("DEP_PYMATCHING_PECOS_STIM_INCLUDE").map_or_else(
        |_| {
            ensure_dep_ready("stim", &manifest)
                .expect("stim dependency")
                .join("src")
        },
        PathBuf::from,
    );
    let stim_dir_for_header = env::var("DEP_PYMATCHING_PECOS_STIM_DIR").map_or_else(
        |_| ensure_dep_ready("stim", &manifest).expect("stim dependency"),
        PathBuf::from,
    );
    let pymatching_lib_dir = env::var("DEP_PYMATCHING_PECOS_LIB_DIR")
        .ok()
        .map(PathBuf::from);

    // Apply compatibility patches for newer Stim version
    chromobius_patch::patch_chromobius_for_newer_stim(&chromobius_dir)?;

    // Generate amalgamated stim.h if needed
    build_stim::generate_amalgamated_header(&stim_dir)?;

    // Build using cxx -- only chromobius and stim sources, NOT pymatching
    build_cxx_bridge(
        &chromobius_dir,
        &stim_dir,
        &pymatching_include,
        &stim_include,
        &stim_dir_for_header,
        pymatching_lib_dir.as_deref(),
    )?;

    Ok(())
}

fn build_cxx_bridge(
    chromobius_dir: &Path,
    stim_dir: &Path,
    pymatching_include: &Path,
    stim_include: &Path,
    stim_dir_for_header: &Path,
    pymatching_lib_dir: Option<&Path>,
) -> Result<()> {
    let chromobius_src_dir = chromobius_dir.join("src");
    let stim_src_dir = stim_dir.join("src");

    // Find essential source files -- only chromobius and stim, NOT pymatching.
    // PyMatching objects come from pecos-pymatching (linked, not compiled here).
    let chromobius_files = collect_chromobius_sources(&chromobius_src_dir)?;
    let stim_files = build_stim::collect_stim_sources(&stim_src_dir);

    // Build the cxx bridge first to generate headers
    let mut build = cxx_build::bridge("src/bridge.rs");

    let target = env::var("TARGET").unwrap_or_default();

    // On macOS, explicitly use system clang to ensure SDK paths are correct.
    if target.contains("darwin") && env::var("CXX").is_err() && env::var("CC").is_err() {
        build.compiler("/usr/bin/clang++");
    }

    // Add our bridge implementation
    build.file("src/bridge.cpp");

    // Add Chromobius core files
    for file in chromobius_files {
        build.file(file);
    }

    // PyMatching objects are provided by pecos-pymatching -- NOT compiled here.
    // We only need headers for compilation, not source files.

    // Configure build
    build
        .std("c++20")
        .include(&chromobius_src_dir)
        .include(&stim_src_dir)
        .include(stim_dir_for_header) // For amalgamated stim.h
        .include(pymatching_include) // PyMatching headers (from pecos-pymatching)
        .include(stim_include) // Stim headers
        .include("include")
        .include("src")
        .define("CHROMOBIUS_BRIDGE_EXPORTS", None);

    // Report ccache/sccache configuration
    report_cache_config();

    // Use build profile for optimization settings
    let profile = get_build_profile();
    match profile.as_str() {
        "native" => {
            build.flag_if_supported("-O3");
            if env::var("CARGO_CFG_TARGET_ARCH").ok() == env::var("HOST_ARCH").ok() {
                build.flag_if_supported("-march=native");
            }
        }
        "release" => {
            build.flag_if_supported("-O3");
        }
        _ => {
            build.flag_if_supported("-O0");
            build.flag_if_supported("-g");
        }
    }

    // Add Stim files to the main build
    for file in &stim_files {
        build.file(file);
    }

    // Platform-specific configurations
    if cfg!(not(target_env = "msvc")) {
        build
            .flag("-fvisibility=hidden")
            .flag("-fvisibility-inlines-hidden")
            .flag("-w")
            .flag_if_supported("-fopenmp")
            .flag("-fPIC");

        if target.contains("darwin") {
            build.flag("-stdlib=libc++");
            build.flag("-L/usr/lib");
            build.flag("-Wl,-search_paths_first");
        }
    } else {
        build
            .flag("/W0")
            .flag("/MD")
            .flag("/EHsc") // Enable C++ exception handling
            .flag_if_supported("/permissive-")
            .flag_if_supported("/Zc:__cplusplus");

        // Force include standard headers that external libraries assume are available
        // MSVC is stricter than GCC/Clang about transitive includes
        build.flag("/FI").flag("array"); // For std::array
        build.flag("/FI").flag("numeric"); // For std::iota (used by PyMatching)
    }

    build.compile("chromobius-bridge");

    // Link against pecos-pymatching's compiled PyMatching objects
    if let Some(lib_dir) = pymatching_lib_dir {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=static=pymatching-bridge");
    }

    // On macOS, link against the system C++ library
    if target.contains("darwin") {
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-arg=-Wl,-search_paths_first");
    }

    Ok(())
}

fn collect_chromobius_sources(chromobius_src_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    // Collect all non-test, non-perf, non-pybind .cc files
    collect_cc_files_filtered(chromobius_src_dir, &mut files)?;

    info!("Found {} Chromobius source files", files.len());
    Ok(files)
}

fn collect_cc_files_filtered(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Skip test directories
            if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && (name == "test" || name == "tests")
            {
                continue;
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
