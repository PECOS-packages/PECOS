//! Build script for `QuEST` integration

use pecos_build_utils::{
    Result, download_cached, extract_archive, quest_download_info, report_cache_config,
};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Main build function for `QuEST`
pub fn build() -> Result<()> {
    // Tell Cargo when to rerun this build script
    println!("cargo:rerun-if-changed=build_quest.rs");
    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    println!("cargo:rerun-if-changed=src/gpu_stubs.cpp");
    println!("cargo:rerun-if-changed=include/quest_ffi.h");

    // Also rerun if the user forces a rebuild
    println!("cargo:rerun-if-env-changed=FORCE_REBUILD");

    // Check for GPU feature
    println!("cargo:rerun-if-env-changed=QUEST_ENABLE_GPU");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDACXX");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let quest_dir = out_dir.join("quest");

    // Always emit link directives - these are cached by Cargo
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=quest-bridge");

    // Download and extract QuEST source if not already present
    if !quest_dir.exists() {
        download_and_extract_quest(&out_dir)?;
    }

    // Build using cxx
    build_cxx_bridge(&quest_dir);

    Ok(())
}

fn download_and_extract_quest(out_dir: &Path) -> Result<()> {
    let info = quest_download_info();
    let tar_gz = download_cached(&info)?;

    // Extract archive to "extracted" subdirectory
    let extracted_dir = out_dir.join("extracted");
    extract_archive(&tar_gz, &extracted_dir, None)?;

    // The archive extracts with an additional "extracted" directory level
    // The quest source is inside extracted/extracted/quest/
    let quest_source_dir = extracted_dir.join("extracted").join("quest");
    let quest_dir = out_dir.join("quest");

    if quest_source_dir.exists() && !quest_dir.exists() {
        // Use copy-recursive instead of rename to handle cross-filesystem moves
        copy_dir_recursive(&quest_source_dir, &quest_dir)?;
    }

    if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
        println!("cargo:warning=QuEST source downloaded and extracted");
    }
    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let entry_path = entry.path();
        let file_name = entry.file_name();
        let dst_path = dst.join(file_name);

        if entry_path.is_dir() {
            copy_dir_recursive(&entry_path, &dst_path)?;
        } else {
            fs::copy(&entry_path, &dst_path)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn build_cxx_bridge(quest_dir: &Path) {
    let quest_src_dir = quest_dir.join("src");
    let quest_include_dir = quest_dir.join("include");

    // Build the cxx bridge first to generate headers
    let mut build = cxx_build::bridge("src/bridge.rs");

    // Determine if we're building with GPU support
    // Check if the gpu feature is enabled via CARGO_FEATURE_GPU env var
    let gpu_feature_enabled = env::var("CARGO_FEATURE_GPU").is_ok();

    // Check if CUDA is actually available
    let cuda_available = env::var("CUDA_PATH").is_ok() || env::var("CUDACXX").is_ok();

    // Only enable GPU if both the feature is enabled AND CUDA is available
    let gpu_enabled = gpu_feature_enabled && cuda_available;

    // Warn if GPU feature was requested but CUDA is not available
    if gpu_feature_enabled && !cuda_available {
        println!(
            "cargo:warning=GPU feature requested but CUDA not found. Building CPU-only version."
        );
        println!(
            "cargo:warning=Set CUDA_PATH or CUDACXX environment variable to enable GPU support."
        );
    }

    // Add QuEST source files
    let api_dir = quest_src_dir.join("api");
    let core_dir = quest_src_dir.join("core");
    let cpu_dir = quest_src_dir.join("cpu");
    let comm_dir = quest_src_dir.join("comm");

    // Add all necessary QuEST source files
    // For CPU-only builds or when CUDA is not available, include GPU stubs
    if !gpu_enabled {
        build.file("src/gpu_stubs.cpp");
    }

    build
        .file("src/bridge.cpp")
        // API layer
        .file(api_dir.join("calculations.cpp"))
        .file(api_dir.join("channels.cpp"))
        .file(api_dir.join("debug.cpp"))
        .file(api_dir.join("decoherence.cpp"))
        .file(api_dir.join("environment.cpp"))
        .file(api_dir.join("initialisations.cpp"))
        .file(api_dir.join("matrices.cpp"))
        .file(api_dir.join("modes.cpp"))
        .file(api_dir.join("operations.cpp"))
        .file(api_dir.join("paulis.cpp"))
        .file(api_dir.join("qureg.cpp"))
        .file(api_dir.join("types.cpp"))
        // Core utilities
        .file(core_dir.join("errors.cpp"))
        .file(core_dir.join("utilities.cpp"))
        .file(core_dir.join("validation.cpp"))
        .file(core_dir.join("memory.cpp"))
        .file(core_dir.join("printer.cpp"))
        .file(core_dir.join("randomiser.cpp"))
        .file(core_dir.join("parser.cpp"))
        .file(core_dir.join("localiser.cpp"))
        .file(core_dir.join("autodeployer.cpp"))
        // Accelerator.cpp contains dispatch logic for both CPU and GPU
        .file(core_dir.join("accelerator.cpp"));

    // Add GPU-specific files only if GPU is enabled
    if gpu_enabled {
        // Add GPU source files
        let gpu_dir = quest_src_dir.join("gpu");
        if gpu_dir.exists() {
            build
                .file(gpu_dir.join("gpu_config.cpp"))
                .file(gpu_dir.join("gpu_subroutines.cpp"));
        }
    }

    // CPU backend
    build
        .file(cpu_dir.join("cpu_config.cpp"))
        .file(cpu_dir.join("cpu_subroutines.cpp"))
        // Communication (even for single-node)
        .file(comm_dir.join("comm_config.cpp"))
        .file(comm_dir.join("comm_routines.cpp"));

    // Include directories
    build
        .include(&quest_include_dir)
        .include(&quest_src_dir)
        .include(quest_dir.parent().unwrap()) // Add out_dir so "quest/include/..." resolves correctly
        .include("include");

    // Define preprocessor flags based on features
    build
        .define("COMPILE_CPU", "1")
        .define("COMPILE_OPENMP", "0") // Disable OpenMP for simplicity initially
        .define("COMPILE_MPI", "0") // Disable MPI for simplicity initially
        .define("FLOAT_PRECISION", "2"); // Double precision by default

    if gpu_enabled {
        build.define("COMPILE_CUDA", "1").define("COMPILE_GPU", "1");

        // Check for cuQuantum support
        if env::var("QUEST_ENABLE_CUQUANTUM").is_ok() {
            build.define("COMPILE_CUQUANTUM", "1");
        } else {
            build.define("COMPILE_CUQUANTUM", "0");
        }

        // Add CUDA include/lib paths if available
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            build.include(Path::new(&cuda_path).join("include"));
            println!("cargo:rustc-link-search=native={cuda_path}/lib64");
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cublas");
        }
    } else {
        build
            .define("COMPILE_CUDA", "0")
            .define("COMPILE_GPU", "0")
            .define("COMPILE_CUQUANTUM", "0");
    }

    // Use C++20 standard (QuEST v4 uses designated initializers which require C++20)
    build.std("c++20");

    // Report ccache/sccache configuration
    report_cache_config();

    // Disable warnings for external QuEST code
    // This properly handles warning flags without conflicts
    build.warnings(false);

    // Use different optimization levels for debug vs release builds
    if cfg!(debug_assertions) {
        build.flag_if_supported("-O0"); // No optimization for faster compilation
        build.flag_if_supported("-g"); // Include debug symbols
    } else {
        build.flag_if_supported("-O3"); // Full optimization for release
    }

    // Platform-specific flags
    if cfg!(not(target_env = "msvc")) {
        // For GCC/Clang
        build.flag_if_supported("-fPIC"); // Position-independent code
    } else {
        // For MSVC
        build
            .flag_if_supported("/permissive-") // Enable standards-compliant C++ parsing
            .flag_if_supported("/Zc:__cplusplus"); // Report correct __cplusplus macro value
    }

    build.compile("quest-bridge");
}
