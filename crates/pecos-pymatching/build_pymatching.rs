//! Build script for `PyMatching` decoder integration

use log::info;
use pecos_build::{Manifest, Result, check_cxx20_toolchain, ensure_dep_ready, report_cache_config};
use std::env;
use std::path::{Path, PathBuf};

// Use the shared modules from the parent
use crate::build_stim;

/// Get the build profile from Cargo's environment
/// Returns "debug", "release", or "native"
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

/// Main build function for `PyMatching`
pub fn build() -> Result<()> {
    check_cxx20_toolchain();

    // Tell Cargo when to rerun this build script
    println!("cargo:rerun-if-changed=build_pymatching.rs");
    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    println!("cargo:rerun-if-changed=include/pymatching_bridge.h");
    println!("cargo:rerun-if-env-changed=FORCE_REBUILD");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    // Always emit link directives - these are cached by Cargo
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=pymatching-bridge");

    // Get PyMatching and Stim sources (downloads to ~/.pecos/cache/, extracts to ~/.pecos/deps/)
    let manifest = Manifest::find_and_load_validated()?;
    let pymatching_dir = ensure_dep_ready("pymatching", &manifest)?;
    let stim_dir = ensure_dep_ready("stim", &manifest)?;

    // Build using cxx
    build_cxx_bridge(&pymatching_dir, &stim_dir)?;

    Ok(())
}

fn build_cxx_bridge(pymatching_dir: &Path, stim_dir: &Path) -> Result<()> {
    let pymatching_src_dir = pymatching_dir.join("src");
    let stim_src_dir = stim_dir.join("src");

    // Find essential Stim source files for DEM functionality
    let stim_files = build_stim::collect_stim_sources(&stim_src_dir);

    // Collect PyMatching source files
    let pymatching_files = collect_pymatching_sources(&pymatching_src_dir)?;

    // Build the CXX bridge
    let mut build = cxx_build::bridge("src/bridge.rs");

    let target = env::var("TARGET").unwrap_or_default();

    // On macOS, explicitly use system clang to ensure SDK paths are correct.
    if target.contains("darwin") && env::var("CXX").is_err() && env::var("CC").is_err() {
        build.compiler("/usr/bin/clang++");
    }

    // Add our bridge implementation
    build.file("src/bridge.cpp");

    // Configure build
    build
        .std("c++20")
        .include(&pymatching_src_dir)
        .include(&stim_src_dir)
        .include("include")
        .include("src")
        .define("PYMATCHING_BRIDGE_EXPORTS", None);

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

    // Add PyMatching files to the build
    for file in &pymatching_files {
        build.file(file);
    }

    // Add Stim files to the build
    for file in &stim_files {
        build.file(file);
    }

    // Platform-specific configurations
    if cfg!(not(target_env = "msvc")) {
        build
            .flag("-fvisibility=hidden")
            .flag("-fvisibility-inlines-hidden")
            .flag("-w") // Suppress all warnings from external code
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

    build.compile("pymatching-bridge");

    // On macOS, link against the system C++ library
    if target.contains("darwin") {
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-arg=-Wl,-search_paths_first");
    }

    Ok(())
}

fn collect_pymatching_sources(pymatching_src_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut sources = Vec::new();

    // Core PyMatching sparse blossom implementation files
    let sparse_blossom_dir = pymatching_src_dir.join("pymatching/sparse_blossom");

    // Driver files
    let driver_dir = sparse_blossom_dir.join("driver");
    sources.extend([
        driver_dir.join("user_graph.cc"),
        driver_dir.join("mwpm_decoding.cc"),
        driver_dir.join("io.cc"),
    ]);

    // Matcher files
    let matcher_dir = sparse_blossom_dir.join("matcher");
    sources.extend([
        matcher_dir.join("mwpm.cc"),
        matcher_dir.join("alternating_tree.cc"),
    ]);

    // Flooder files
    let flooder_dir = sparse_blossom_dir.join("flooder");
    sources.extend([
        flooder_dir.join("graph_flooder.cc"),
        flooder_dir.join("graph.cc"),
        flooder_dir.join("detector_node.cc"),
        flooder_dir.join("match.cc"),
        flooder_dir.join("graph_fill_region.cc"),
    ]);

    // Tracker files
    let tracker_dir = sparse_blossom_dir.join("tracker");
    sources.push(tracker_dir.join("flood_check_event.cc"));

    // Search files
    let search_dir = sparse_blossom_dir.join("search");
    sources.extend([
        search_dir.join("search_graph.cc"),
        search_dir.join("search_flooder.cc"),
        search_dir.join("search_detector_node.cc"),
    ]);

    // Flooder matcher interop files
    let interop_dir = sparse_blossom_dir.join("flooder_matcher_interop");
    sources.extend([
        interop_dir.join("compressed_edge.cc"),
        interop_dir.join("region_edge.cc"),
        interop_dir.join("mwpm_event.cc"),
    ]);

    // Random number generation files (needed for add_noise)
    let rand_dir = pymatching_src_dir.join("pymatching/rand");
    sources.push(rand_dir.join("rand_gen.cc"));

    // Filter to only include files that exist
    let existing_sources: Vec<PathBuf> = sources
        .into_iter()
        .filter(|path| {
            let exists = path.exists();
            if !exists {
                info!("PyMatching source file not found: {}", path.display());
            }
            exists
        })
        .collect();

    if existing_sources.is_empty() {
        return Err(pecos_build::Error::Config(
            "No PyMatching source files found".to_string(),
        ));
    }

    info!("Found {} PyMatching source files", existing_sources.len());

    Ok(existing_sources)
}
