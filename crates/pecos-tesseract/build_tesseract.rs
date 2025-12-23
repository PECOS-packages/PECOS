//! Build script for Tesseract decoder integration

use pecos_build::{Manifest, Result, ensure_dep_ready, report_cache_config};
use std::env;
use std::path::{Path, PathBuf};

// Use the shared modules from the parent
use crate::build_stim;

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

/// Main build function for Tesseract
pub fn build() -> Result<()> {
    println!("cargo:rerun-if-changed=build_tesseract.rs");
    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    println!("cargo:rerun-if-changed=include/tesseract_bridge.h");
    println!("cargo:rerun-if-env-changed=FORCE_REBUILD");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    // Always emit link directives - Cargo will cache these
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=tesseract-bridge");

    // Get Tesseract and Stim sources (downloads to ~/.pecos/cache/, extracts to ~/.pecos/deps/)
    let manifest = Manifest::find_and_load_validated()?;
    let tesseract_dir = ensure_dep_ready("tesseract", &manifest)?;
    let stim_dir = ensure_dep_ready("stim", &manifest)?;

    // Build using cxx
    build_cxx_bridge(&tesseract_dir, &stim_dir);

    Ok(())
}

fn build_cxx_bridge(tesseract_dir: &Path, stim_dir: &Path) {
    let tesseract_src_dir = tesseract_dir.join("src");
    let stim_src_dir = stim_dir.join("src");

    // Find essential Stim source files for DEM functionality
    let stim_files = build_stim::collect_stim_sources(&stim_src_dir);

    // Build everything together
    let mut build = cxx_build::bridge("src/bridge.rs");

    let target = env::var("TARGET").unwrap_or_default();

    // On macOS, explicitly use system clang to ensure SDK paths are correct.
    if target.contains("darwin") && env::var("CXX").is_err() && env::var("CC").is_err() {
        build.compiler("/usr/bin/clang++");
    }

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
        .define("TESSERACT_BRIDGE_EXPORTS", None);

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

    // Add Stim files to the build
    for file in stim_files {
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
        build.flag("/FI").flag("numeric"); // For std::iota
    }

    build.compile("tesseract-bridge");

    // On macOS, link against the system C++ library
    if target.contains("darwin") {
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-arg=-Wl,-search_paths_first");
    }
}
