//! Build script for `QuEST` integration

use log::{debug, info};
use pecos_build_utils::{
    Result, download_cached, extract_archive, quest_download_info, report_cache_config,
};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Detect CUDA installation using nvcc command
/// Returns the CUDA installation path if found
fn detect_cuda_path() -> Option<String> {
    // First check environment variables
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        info!("Found CUDA via CUDA_PATH: {cuda_path}");
        return Some(cuda_path);
    }

    // Try to find nvcc in PATH
    if let Ok(nvcc_output) = Command::new("nvcc").arg("--version").output()
        && nvcc_output.status.success()
    {
        // Try to get CUDA path from nvcc location using 'which nvcc'
        if let Ok(which_output) = Command::new("which").arg("nvcc").output()
            && which_output.status.success()
        {
            let nvcc_path = String::from_utf8_lossy(&which_output.stdout)
                .trim()
                .to_string();
            // nvcc is typically at /usr/local/cuda[-version]/bin/nvcc
            // We want /usr/local/cuda[-version]
            let path = Path::new(&nvcc_path);
            if let Some(bin_dir) = path.parent()
                && let Some(cuda_root) = bin_dir.parent()
            {
                info!("Found CUDA via nvcc in PATH: {}", cuda_root.display());
                return Some(cuda_root.to_string_lossy().to_string());
            }
        }
    }

    // Fallback to checking standard installation paths
    // Check symlinks first, then specific versions
    for path in &[
        "/usr/local/cuda",      // Common symlink
        "/usr/local/cuda-13",   // Version symlink
        "/usr/local/cuda-13.0", // Specific CUDA 13.0
        "/usr/local/cuda-13.1", // Specific CUDA 13.1
        "/usr/local/cuda-12",   // Version symlink
        "/usr/local/cuda-12.0", // Specific CUDA 12.0
        "/usr/local/cuda-11",   // Version symlink
        "/usr/local/cuda-11.0", // Specific CUDA 11.0
    ] {
        if Path::new(path).exists() {
            info!("Found CUDA at standard path: {path}");
            return Some((*path).to_string());
        }
    }

    None
}

/// Compile CUDA source files with nvcc
/// Returns None if compilation fails
fn compile_cuda_files(
    cuda_path: &str,
    gpu_files: &[PathBuf],
    quest_dir: &Path,
    out_dir: &Path,
) -> Option<Vec<PathBuf>> {
    let mut object_files = Vec::new();

    // Construct path to nvcc using the detected CUDA installation
    let nvcc_path = Path::new(cuda_path).join("bin").join("nvcc");

    info!("Compiling GPU files with nvcc at: {}", nvcc_path.display());

    for gpu_file in gpu_files {
        let file_stem = gpu_file.file_stem()?.to_str()?;
        let obj_file = out_dir.join(format!("{file_stem}.o"));

        let quest_include_dir = quest_dir.join("include");
        let quest_src_dir = quest_dir.join("src");

        // Compile with nvcc
        debug!("Compiling: {}", gpu_file.file_name()?.to_str()?);
        let output = Command::new(&nvcc_path)
            .arg("-c")
            .arg(gpu_file)
            .arg("-o")
            .arg(&obj_file)
            .arg("-x")
            .arg("cu") // Treat .cpp files as CUDA source
            .arg("-I")
            .arg(&quest_include_dir)
            .arg("-I")
            .arg(&quest_src_dir)
            .arg("-I")
            .arg(quest_dir.parent()?)
            .arg("--std=c++20")
            .arg("-DCOMPILE_GPU=1")
            .arg("-DCOMPILE_CUDA=1")
            .arg("-DCOMPILE_CPU=1")
            .arg("-DCOMPILE_OPENMP=0")
            .arg("-DCOMPILE_MPI=0")
            .arg("-DCOMPILE_CUQUANTUM=0")
            .arg("-DFLOAT_PRECISION=2")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .output()
            .ok()?;

        if !output.status.success() {
            let stderr_str = String::from_utf8_lossy(&output.stderr);

            // Check if this is the known CUDA 13 incompatibility
            if stderr_str.contains("thrust::unary_function")
                || stderr_str.contains("thrust::binary_function")
            {
                println!(
                    "cargo:warning=GPU compilation failed: QuEST is incompatible with CUDA 13+"
                );
                println!("cargo:warning=The QuEST library requires CUDA 11 or 12 for GPU support");
                println!("cargo:warning=Consider using CUDA 12 or building without GPU feature");
            } else {
                println!(
                    "cargo:warning=nvcc compilation failed for {}",
                    gpu_file.file_name().unwrap().to_str().unwrap()
                );
            }

            // Write full error to a temp file for debugging
            let error_file = "/tmp/nvcc_error.log";
            if let Err(e) = fs::write(error_file, stderr_str.as_bytes()) {
                debug!("Failed to write error log: {e}");
            } else {
                debug!("Full error written to {error_file}");
            }

            return None;
        }

        debug!("Successfully compiled {}", gpu_file.file_name()?.to_str()?);
        object_files.push(obj_file);
    }

    info!("Successfully compiled all GPU files");
    Some(object_files)
}

/// Patch `QuEST` GPU code for CUDA 13 compatibility
///
/// Removes `thrust::unary_function` and `thrust::binary_function` inheritance
/// which were deprecated and removed in modern CUDA/Thrust versions.
/// With C++20, these base classes are no longer needed.
fn patch_quest_for_cuda13(quest_dir: &Path) -> Result<()> {
    let thrust_file = quest_dir.join("src/gpu/gpu_thrust.cuh");

    if !thrust_file.exists() {
        // GPU files don't exist, nothing to patch
        return Ok(());
    }

    info!("Patching QuEST for CUDA 13 compatibility...");

    let content = fs::read_to_string(&thrust_file)?;

    // Use regex to remove thrust::unary_function and thrust::binary_function inheritance
    // Pattern: "struct NAME : public thrust::(unary|binary)_function<...>"
    // Replace with: "struct NAME"

    // First, handle single-line patterns (with opening brace)
    let patched = content
        .replace(": public thrust::unary_function<cu_qcomp,cu_qcomp> {", " {")
        .replace(": public thrust::unary_function<cu_qcomp,qreal> {", " {")
        .replace(": public thrust::unary_function<qindex,cu_qcomp> {", " {")
        .replace(": public thrust::unary_function<qindex,qindex> {", " {")
        .replace(
            ": public thrust::binary_function<cu_qcomp,cu_qcomp,cu_qcomp> {",
            " {",
        )
        .replace(
            ": public thrust::binary_function<cu_qcomp,cu_qcomp,qreal> {",
            " {",
        )
        .replace(
            ": public thrust::binary_function<qindex,cu_qcomp,qreal> {",
            " {",
        )
        .replace(
            ": public thrust::binary_function<qindex,cu_qcomp,cu_qcomp> {",
            " {",
        )
        // Handle multi-line patterns (no opening brace on same line)
        .replace(": public thrust::unary_function<cu_qcomp,cu_qcomp>", "")
        .replace(": public thrust::unary_function<cu_qcomp,qreal>", "")
        .replace(": public thrust::unary_function<qindex,cu_qcomp>", "")
        .replace(": public thrust::unary_function<qindex,qindex>", "")
        .replace(
            ": public thrust::binary_function<cu_qcomp,cu_qcomp,cu_qcomp>",
            "",
        )
        .replace(
            ": public thrust::binary_function<cu_qcomp,cu_qcomp,qreal>",
            "",
        )
        .replace(
            ": public thrust::binary_function<qindex,cu_qcomp,qreal>",
            "",
        )
        .replace(
            ": public thrust::binary_function<qindex,cu_qcomp,cu_qcomp>",
            "",
        );

    fs::write(&thrust_file, patched)?;

    info!("Successfully patched gpu_thrust.cuh for CUDA 13");

    Ok(())
}

/// Generate quest.h from quest.h.in template (`QuEST` v4.1.0+)
fn generate_quest_header(quest_dir: &Path) -> Result<()> {
    let template_file = quest_dir.join("include/quest.h.in");
    let output_file = quest_dir.join("include/quest.h");

    if !template_file.exists() {
        // quest.h already exists or not using template-based build
        return Ok(());
    }

    info!("Generating quest.h from template...");

    let template = fs::read_to_string(&template_file)?;

    // Since MULTI_LIB_HEADERS=0, we want the #if !0 block to be active
    // which means we need to process the #cmakedefine directives
    let is_gpu = env::var("CARGO_FEATURE_GPU").is_ok();

    // Process the template line by line to handle conditional blocks
    let mut in_multi_lib_block = false;
    let mut found_cmakedefine = false;
    let quest_h = template
        .lines()
        .filter_map(|line| {
            // Track when we're in the MULTI_LIB_HEADERS conditional
            if line.contains("#if !@MULTI_LIB_HEADERS@") {
                in_multi_lib_block = true;
                return None; // Remove this line
            }

            // Process #cmakedefine directives (these are inside the block we're removing the conditional from)
            if line.contains("#cmakedefine") {
                found_cmakedefine = true;
                if line.contains("#cmakedefine FLOAT_PRECISION @FLOAT_PRECISION@") {
                    return Some("#define FLOAT_PRECISION 2".to_string());
                }
                if line.contains("#cmakedefine01 COMPILE_MPI") {
                    return Some("#define COMPILE_MPI 0".to_string());
                }
                if line.contains("#cmakedefine01 COMPILE_OPENMP") {
                    return Some("#define COMPILE_OPENMP 0".to_string());
                }
                if line.contains("#cmakedefine01 COMPILE_CUDA") {
                    return Some(format!("#define COMPILE_CUDA {}", i32::from(is_gpu)));
                }
                if line.contains("#cmakedefine01 COMPILE_CUQUANTUM") {
                    return Some("#define COMPILE_CUQUANTUM 0".to_string());
                }
            }

            // Remove the #endif that closes the MULTI_LIB_HEADERS block
            if line.contains("#endif") && in_multi_lib_block && found_cmakedefine {
                in_multi_lib_block = false;
                found_cmakedefine = false;
                return None; // Remove this specific #endif
            }

            Some(line.to_string())
        })
        .collect::<Vec<_>>()
        .join("\n");

    fs::write(&output_file, quest_h)?;

    info!("Successfully generated quest.h");

    Ok(())
}

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
    build_cxx_bridge(&quest_dir, &out_dir);

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

        // Apply CUDA 13 compatibility patches
        patch_quest_for_cuda13(&quest_dir)?;

        // Generate quest.h from quest.h.in (QuEST v4.1.0 requirement)
        generate_quest_header(&quest_dir)?;
    }

    info!("QuEST source downloaded and extracted");
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
fn build_cxx_bridge(quest_dir: &Path, out_dir: &Path) {
    let quest_src_dir = quest_dir.join("src");
    let quest_include_dir = quest_dir.join("include");

    // Build the cxx bridge first to generate headers
    let mut build = cxx_build::bridge("src/bridge.rs");

    // Determine if we're building with GPU support
    // Check if the gpu feature is enabled via CARGO_FEATURE_GPU env var
    let gpu_feature_enabled = env::var("CARGO_FEATURE_GPU").is_ok();

    // Detect CUDA installation
    let cuda_path = detect_cuda_path();
    let cuda_available = cuda_path.is_some();

    // Only enable GPU if both the feature is enabled AND CUDA is available
    let gpu_enabled = gpu_feature_enabled && cuda_available;

    // Error if GPU feature was requested but CUDA is not available
    if gpu_feature_enabled && !cuda_available {
        eprintln!("ERROR: GPU feature enabled but CUDA not found");
        eprintln!("  CUDA Toolkit must be installed to build with GPU support");
        eprintln!("  Solutions:");
        eprintln!("    1. Install CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)");
        eprintln!("    2. Ensure nvcc is in PATH or set CUDA_PATH environment variable");
        eprintln!("    3. Build without GPU feature: cargo build -p pecos-quest");
        std::process::exit(1);
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

    // GPU files will be compiled separately with nvcc
    // Don't add them to cxx_build
    let gpu_object_files = if gpu_enabled {
        let gpu_dir = quest_src_dir.join("gpu");
        if !gpu_dir.exists() {
            eprintln!("\nERROR: GPU feature enabled but QuEST GPU source not found");
            eprintln!("  Expected directory: {}", gpu_dir.display());
            eprintln!("  This may indicate an incomplete QuEST download");
            std::process::exit(1);
        }

        let gpu_files = vec![
            gpu_dir.join("gpu_config.cpp"),
            gpu_dir.join("gpu_subroutines.cpp"),
        ];

        // Compile GPU files with nvcc
        if let Some(obj_files) =
            compile_cuda_files(cuda_path.as_ref().unwrap(), &gpu_files, quest_dir, out_dir)
        {
            info!("GPU compilation successful - QuEST built with CUDA support");
            Some(obj_files)
        } else {
            eprintln!("\nERROR: GPU feature enabled but GPU compilation failed");
            eprintln!("  See warnings above for compilation errors");
            eprintln!("  Solutions:");
            eprintln!("    1. Use CUDA 11 or 12 instead of CUDA 13 (QuEST incompatibility)");
            eprintln!("    2. Build without GPU feature: cargo build -p pecos-quest");
            eprintln!("    3. Use Python GPU simulators (CuStateVec/MPS) which work with CUDA 13");
            std::process::exit(1);
        }
    } else {
        None
    };

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
        if let Some(ref cuda_path) = cuda_path {
            build.include(Path::new(&cuda_path).join("include"));
            println!("cargo:rustc-link-search=native={cuda_path}/lib64");
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cublas");

            info!("Using CUDA from: {cuda_path}");
        }
    } else {
        build
            .define("COMPILE_CUDA", "0")
            .define("COMPILE_GPU", "0")
            .define("COMPILE_CUQUANTUM", "0");
    }

    // Use C++20 standard (QuEST v4 uses designated initializers which require C++20)
    // However, on macOS there's a known issue with C++20 and cxx crate's pointer_traits
    // specializations, so we use C++17 there (designated initializers are a GNU extension
    // that works in C++17 with Clang)
    if std::env::var("TARGET")
        .unwrap_or_default()
        .contains("darwin")
    {
        build.std("c++17");
        // Enable GNU extensions to support designated initializers in C++17
        build.flag_if_supported("-Wno-c++20-designator");
    } else {
        build.std("c++20");
    }

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
            .flag_if_supported("/Zc:__cplusplus") // Report correct __cplusplus macro value
            .flag("/Z7"); // Embed debug info in .obj files (no PDB) - required for parallel builds
    }

    // Platform-specific C++ library linking configuration
    if cfg!(not(target_env = "msvc")) {
        // On macOS, use the -stdlib=libc++ flag to ensure proper C++ standard library linkage
        // This tells the linker to use the system libc++ from the dyld shared cache
        // without creating problematic @rpath references
        if std::env::var("TARGET")
            .unwrap_or_default()
            .contains("darwin")
        {
            build.flag("-stdlib=libc++");
            // Note: Linker-specific flags are passed via cargo:rustc-link-arg below, not here
        }
    }

    build.compile("quest-bridge");

    // Link GPU object files if they were compiled
    if let Some(gpu_objs) = gpu_object_files {
        for obj in &gpu_objs {
            println!("cargo:rustc-link-arg={}", obj.display());
        }
    }

    // On macOS, ensure the C++ standard library is linked correctly
    // Use the system libc++ which is in the dyld shared cache (macOS Big Sur+)
    // We rely on the compiler's default behavior rather than explicit cargo directives
    // which can create problematic @rpath references
    if std::env::var("TARGET")
        .unwrap_or_default()
        .contains("darwin")
    {
        // Link against the system C++ library
        // Use -L flag to prioritize system library paths over Homebrew
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-lib=c++");

        // Prevent Homebrew's libunwind from being opportunistically linked
        // by ensuring system paths are searched first
        println!("cargo:rustc-link-arg=-Wl,-search_paths_first");
    }
}
