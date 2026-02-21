//! Build script for `QuEST` integration
//!
//! This build script produces:
//! 1. A static library (libquest-bridge.a) for CPU-only `QuEST` operations
//! 2. Optionally, a shared library (`libpecos_quest_cuda.so`) for CUDA operations (when cuda feature enabled)
//!
//! The CUDA library is loaded at runtime via dlopen, allowing a single binary to work
//! on systems with and without CUDA installed.

use log::{debug, info};
use pecos_build::{Manifest, Result, ensure_dep_ready, report_cache_config};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Detect CUDA installation using nvcc command
/// Returns the CUDA installation path if found
///
/// Search order:
/// 1. `~/.pecos/cuda/` (local installation via pecos install cuda)
/// 2. `CUDA_PATH` environment variable
/// 3. `nvcc` in PATH
/// 4. Standard system paths
fn detect_cuda_path() -> Option<String> {
    // 1. Check ~/.pecos/cuda/ first (local installation via pecos)
    if let Some(home) = dirs::home_dir() {
        let pecos_cuda = home.join(".pecos").join("cuda");
        let nvcc_path = pecos_cuda.join("bin").join("nvcc");
        if nvcc_path.exists() {
            info!("Found CUDA in ~/.pecos/cuda/ (installed via pecos)");
            return Some(pecos_cuda.to_string_lossy().to_string());
        }
    }

    // 2. Check environment variables
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        info!("Found CUDA via CUDA_PATH: {cuda_path}");
        return Some(cuda_path);
    }

    // 3. Try to find nvcc in PATH
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

    // 4. Fallback to checking standard installation paths
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

/// Build the GPU shared library (`libpecos_quest_cuda.so`)
///
/// This library contains the GPU-accelerated `QuEST` implementation and is loaded
/// at runtime via dlopen. This allows the main library to work on systems without CUDA.
#[allow(clippy::too_many_lines)]
fn build_gpu_shared_library(cuda_path: &str, quest_dir: &Path, out_dir: &Path) -> Option<PathBuf> {
    info!("Building GPU shared library (libpecos_quest_cuda.so)...");

    // nvcc executable name differs by platform
    let nvcc_name = if cfg!(target_os = "windows") {
        "nvcc.exe"
    } else {
        "nvcc"
    };
    let nvcc_path = Path::new(cuda_path).join("bin").join(nvcc_name);
    info!("Using nvcc at: {}", nvcc_path.display());
    let quest_include_dir = quest_dir.join("include");
    let quest_src_dir = quest_dir.join("src");
    let gpu_dir = quest_src_dir.join("gpu");

    // Source files for the GPU library
    let bridge_gpu = PathBuf::from("src/bridge_cuda.cpp");
    let gpu_config = gpu_dir.join("gpu_config.cpp");
    let gpu_subroutines = gpu_dir.join("gpu_subroutines.cpp");

    // QuEST core files needed by the GPU library
    let api_dir = quest_src_dir.join("api");
    let core_dir = quest_src_dir.join("core");
    let cpu_dir = quest_src_dir.join("cpu");
    let comm_dir = quest_src_dir.join("comm");

    // Collect all source files
    let source_files = vec![
        bridge_gpu,
        gpu_config,
        gpu_subroutines,
        // API layer
        api_dir.join("calculations.cpp"),
        api_dir.join("channels.cpp"),
        api_dir.join("debug.cpp"),
        api_dir.join("decoherence.cpp"),
        api_dir.join("environment.cpp"),
        api_dir.join("initialisations.cpp"),
        api_dir.join("matrices.cpp"),
        api_dir.join("modes.cpp"),
        api_dir.join("operations.cpp"),
        api_dir.join("paulis.cpp"),
        api_dir.join("qureg.cpp"),
        api_dir.join("types.cpp"),
        // Core utilities
        core_dir.join("errors.cpp"),
        core_dir.join("utilities.cpp"),
        core_dir.join("validation.cpp"),
        core_dir.join("memory.cpp"),
        core_dir.join("printer.cpp"),
        core_dir.join("randomiser.cpp"),
        core_dir.join("parser.cpp"),
        core_dir.join("localiser.cpp"),
        core_dir.join("autodeployer.cpp"),
        core_dir.join("accelerator.cpp"),
        // CPU backend (still needed for some operations)
        cpu_dir.join("cpu_config.cpp"),
        cpu_dir.join("cpu_subroutines.cpp"),
        // Communication
        comm_dir.join("comm_config.cpp"),
        comm_dir.join("comm_routines.cpp"),
    ];

    // Compile all source files to object files
    let mut object_files = Vec::new();
    for src_file in &source_files {
        let file_stem = src_file.file_stem()?.to_str()?;
        // Windows uses .obj extension, Unix uses .o
        let obj_ext = if cfg!(target_os = "windows") {
            "obj"
        } else {
            "o"
        };
        let obj_file = out_dir.join(format!("gpu_{file_stem}.{obj_ext}"));

        debug!("Compiling for GPU lib: {}", src_file.display());
        let mut compile_cmd = Command::new(&nvcc_path);
        compile_cmd
            .arg("-c")
            .arg(src_file)
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
            .arg("-I")
            .arg("include") // For quest_ffi.h
            .arg("--std=c++20")
            .arg("-DCOMPILE_GPU=1")
            .arg("-DCOMPILE_CUDA=1")
            .arg("-DCOMPILE_CPU=1")
            .arg("-DCOMPILE_OPENMP=0")
            .arg("-DCOMPILE_MPI=0")
            .arg("-DCOMPILE_CUQUANTUM=0")
            .arg("-DFLOAT_PRECISION=2")
            // Target compute capability 7.5 (Turing) which supports atomicAdd(double*, double)
            // sm_75 is the minimum supported by both CUDA 12.x and 13.x
            .arg("-arch=sm_75")
            // Allow newer GCC versions (e.g., GCC 14 in manylinux_2_28)
            .arg("-allow-unsupported-compiler");

        // Platform-specific compiler flags
        if cfg!(target_os = "windows") {
            // Windows/MSVC: no -fPIC needed (not applicable)
            // Use /EHsc for C++ exception handling
            compile_cmd.arg("-Xcompiler").arg("/EHsc");
        } else {
            // Unix: position-independent code for shared libraries
            compile_cmd.arg("-Xcompiler").arg("-fPIC");
        }

        let output = compile_cmd.output().ok()?;

        if !output.status.success() {
            let stdout_str = String::from_utf8_lossy(&output.stdout);
            let stderr_str = String::from_utf8_lossy(&output.stderr);
            eprintln!(
                "ERROR: Failed to compile {} for GPU library",
                src_file.display()
            );
            eprintln!("Exit status: {:?}", output.status);
            if !stdout_str.is_empty() {
                eprintln!("stdout:\n{stdout_str}");
            }
            if !stderr_str.is_empty() {
                eprintln!("stderr:\n{stderr_str}");
            }
            return None;
        }

        object_files.push(obj_file);
    }

    // Link into a shared library
    let lib_name = if cfg!(target_os = "macos") {
        "libpecos_quest_cuda.dylib"
    } else if cfg!(target_os = "windows") {
        "pecos_quest_cuda.dll"
    } else {
        "libpecos_quest_cuda.so"
    };

    let gpu_lib_path = out_dir.join(lib_name);

    info!("Linking GPU shared library: {}", gpu_lib_path.display());

    let mut link_cmd = Command::new(&nvcc_path);
    link_cmd
        .arg("-shared")
        .arg("-o")
        .arg(&gpu_lib_path)
        .args(&object_files);

    // Platform-specific library paths and linking
    if cfg!(target_os = "windows") {
        // Windows: CUDA libraries are in lib\x64
        link_cmd
            .arg(format!("-L{cuda_path}/lib/x64"))
            .arg("-lcudart")
            .arg("-lcublas");
        // Windows uses MSVC runtime, no need to explicitly link C++ stdlib
    } else {
        // Unix: CUDA libraries are in lib64
        link_cmd
            .arg(format!("-L{cuda_path}/lib64"))
            .arg("-lcudart")
            .arg("-lcublas");
        // Add C++ standard library
        if cfg!(target_os = "macos") {
            link_cmd.arg("-lc++");
        } else {
            link_cmd.arg("-lstdc++");
        }
    }

    let output = link_cmd.output().ok()?;

    if !output.status.success() {
        let stderr_str = String::from_utf8_lossy(&output.stderr);
        eprintln!("ERROR: Failed to link GPU shared library");
        eprintln!("{stderr_str}");
        return None;
    }

    info!(
        "Successfully built GPU shared library: {}",
        gpu_lib_path.display()
    );

    // Also copy to target directory for easier discovery
    // Try CARGO_TARGET_DIR first, then derive from OUT_DIR
    let target_lib_dir = if let Ok(target_dir) = env::var("CARGO_TARGET_DIR") {
        let profile = get_build_profile();
        Some(Path::new(&target_dir).join(&profile))
    } else {
        // OUT_DIR is something like: target/release/build/pecos-quest-xxx/out
        // We want: target/release/
        out_dir
            .parent() // build/pecos-quest-xxx
            .and_then(|p| p.parent()) // build
            .and_then(|p| p.parent()) // release or debug
            .map(std::path::Path::to_path_buf)
    };

    if let Some(target_dir) = target_lib_dir {
        let target_lib_path = target_dir.join(lib_name);
        if let Some(parent) = target_lib_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Err(e) = fs::copy(&gpu_lib_path, &target_lib_path) {
            debug!("Could not copy CUDA lib to target dir: {e}");
        } else {
            info!("Copied CUDA lib to: {}", target_lib_path.display());
        }
    }

    Some(gpu_lib_path)
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
    //
    // IMPORTANT: The main library is ALWAYS CPU-only (COMPILE_CUDA=0).
    // GPU support is provided via a separate shared library (libpecos_quest_cuda.so)
    // which is compiled with nvcc and has its own COMPILE_CUDA=1 flag.
    // This generated quest.h is only used by the main library.

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
                    // Main library is always CPU-only; GPU library is separate
                    return Some("#define COMPILE_CUDA 0".to_string());
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

/// Get the build profile from Cargo's environment
/// Returns "debug", "release", or "native"
///
/// Note: Cargo's PROFILE env var only reports "debug" or "release" even for custom profiles
/// (due to backward compatibility - see RFC 2678). Custom profiles inherit from these base
/// profiles, so PROFILE reflects the parent. To detect custom profiles like "native", we
/// check the `OUT_DIR` path which contains the actual profile directory name.
///
/// Profile behavior:
/// - "debug" -> no C++ optimization, fast compile
/// - "release" -> full optimization (-O3)
/// - "native" -> full optimization + CPU-specific (-O3 -march=native)
fn get_build_profile() -> String {
    // First check OUT_DIR for custom profile name (e.g., target/native/build/...)
    // Custom profiles get their own directory under target/
    if let Ok(out_dir) = env::var("OUT_DIR") {
        // OUT_DIR looks like: .../target/<profile>/build/<crate>-<hash>/out
        // We want to extract <profile>
        let parts: Vec<&str> = out_dir.split(std::path::MAIN_SEPARATOR).collect();
        if let Some(target_idx) = parts.iter().position(|&p| p == "target")
            && let Some(profile_name) = parts.get(target_idx + 1)
        {
            return match *profile_name {
                "native" => "native",
                "release" => "release",
                "debug" => "debug",
                _ => {
                    // Unknown profile, fall back to PROFILE env var
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

    // Fallback to PROFILE env var (will be "debug" or "release")
    match env::var("PROFILE").as_deref() {
        Ok("release") => "release".to_string(),
        _ => "debug".to_string(),
    }
}

/// Main build function for `QuEST`
pub fn build() -> Result<()> {
    // Tell Cargo when to rerun this build script
    println!("cargo:rerun-if-changed=build_quest.rs");
    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    println!("cargo:rerun-if-changed=src/bridge_cuda.cpp");
    println!("cargo:rerun-if-changed=src/gpu_stubs.cpp");
    println!("cargo:rerun-if-changed=src/cuda_loader.rs");
    println!("cargo:rerun-if-changed=include/quest_ffi.h");

    // Also rerun if the user forces a rebuild
    println!("cargo:rerun-if-env-changed=FORCE_REBUILD");

    // Check for GPU feature
    println!("cargo:rerun-if-env-changed=QUEST_ENABLE_GPU");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDACXX");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    // Always emit link directives - these are cached by Cargo
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=quest-bridge");

    // Get QuEST source from ~/.pecos/deps/ (persists across cargo clean)
    let quest_dir = get_quest_source()?;

    // Build using cxx
    build_cxx_bridge(&quest_dir, &out_dir);

    Ok(())
}

/// Get `QuEST` source directory, downloading and extracting if needed
///
/// Returns the path to the `quest/` subdirectory within the extracted archive.
/// Also applies patches for CUDA 13 compatibility and generates quest.h header.
fn get_quest_source() -> Result<PathBuf> {
    // Load manifest and get QuEST dependency
    let manifest = Manifest::find_and_load_validated()?;

    // ensure_dep_ready downloads to ~/.pecos/cache/ and extracts to ~/.pecos/deps/
    let deps_path = ensure_dep_ready("quest", &manifest)?;

    // The QuEST archive extracts as: deps/quest-<version>/quest/
    // (contains quest/ subdirectory with actual source)
    let quest_dir = deps_path.join("quest");

    if !quest_dir.exists() {
        return Err(pecos_build::Error::Archive(format!(
            "QuEST source directory not found at: {}",
            quest_dir.display()
        )));
    }

    // Apply CUDA 13 compatibility patches (idempotent)
    patch_quest_for_cuda13(&quest_dir)?;

    // Generate quest.h from quest.h.in (idempotent - only runs if template exists)
    generate_quest_header(&quest_dir)?;

    info!("Using QuEST source from {}", quest_dir.display());
    Ok(quest_dir)
}

#[allow(clippy::too_many_lines)]
fn build_cxx_bridge(quest_dir: &Path, out_dir: &Path) {
    let quest_src_dir = quest_dir.join("src");
    let quest_include_dir = quest_dir.join("include");

    // Build the cxx bridge first to generate headers
    let mut build = cxx_build::bridge("src/bridge.rs");

    // On macOS, explicitly use system clang to ensure SDK paths are correct.
    // The PECOS LLVM clang may be in PATH but doesn't have SDK headers configured,
    // causing "math.h file not found" errors during compilation.
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("darwin") && env::var("CXX").is_err() && env::var("CC").is_err() {
        build.compiler("/usr/bin/clang++");
    }

    // Determine if we're building with GPU support
    // Check if the gpu feature is enabled via CARGO_FEATURE_CUDA env var
    let gpu_feature_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();

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

    // IMPORTANT: The main library ALWAYS uses gpu_stubs.cpp (CPU only).
    // GPU support is provided by a separate shared library (libpecos_quest_cuda.so)
    // that is loaded at runtime via dlopen. This allows a single binary to work
    // on systems with and without CUDA installed.
    build.file("src/gpu_stubs.cpp");

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

    // Build the separate GPU shared library if GPU feature is enabled
    // This library will be loaded at runtime via dlopen
    if gpu_enabled {
        let gpu_dir = quest_src_dir.join("gpu");
        if !gpu_dir.exists() {
            eprintln!("\nERROR: GPU feature enabled but QuEST GPU source not found");
            eprintln!("  Expected directory: {}", gpu_dir.display());
            eprintln!("  This may indicate an incomplete QuEST download");
            std::process::exit(1);
        }

        // Build the separate GPU shared library
        if let Some(gpu_lib_path) =
            build_gpu_shared_library(cuda_path.as_ref().unwrap(), quest_dir, out_dir)
        {
            info!(
                "GPU shared library built successfully: {}",
                gpu_lib_path.display()
            );
            // Emit the GPU library path so downstream crates can find it
            println!(
                "cargo:rustc-env=PECOS_QUEST_CUDA_LIB={}",
                gpu_lib_path.display()
            );
        } else {
            eprintln!("\nERROR: GPU feature enabled but GPU library build failed");
            eprintln!("  See warnings above for compilation errors");
            eprintln!("  Solutions:");
            eprintln!("    1. Use CUDA 11 or 12 instead of CUDA 13 (QuEST incompatibility)");
            eprintln!("    2. Build without GPU feature: cargo build -p pecos-quest");
            eprintln!("    3. Use Python GPU simulators (CuStateVec/MPS) which work with CUDA 13");
            std::process::exit(1);
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
    // IMPORTANT: The main library is ALWAYS CPU-only. GPU support is provided via
    // a separate shared library (libpecos_quest_cuda.so) loaded at runtime via dlopen.
    // This allows a single binary to work on systems with and without CUDA.
    build
        .define("COMPILE_CPU", "1")
        .define("COMPILE_OPENMP", "0") // Disable OpenMP for simplicity initially
        .define("COMPILE_MPI", "0") // Disable MPI for simplicity initially
        .define("FLOAT_PRECISION", "2") // Double precision by default
        .define("COMPILE_CUDA", "0") // Main library never uses CUDA directly
        .define("COMPILE_GPU", "0") // GPU ops are in the separate GPU library
        .define("COMPILE_CUQUANTUM", "0");

    // Note: We do NOT link cudart/cublas here. The GPU library handles CUDA linking
    // and is loaded at runtime only when GPU is requested.

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

    // Use build profile for optimization settings
    let profile = get_build_profile();
    match profile.as_str() {
        "native" => {
            // Native profile: release optimizations + CPU-specific optimizations
            build.flag_if_supported("-O3");
            build.flag_if_supported("-march=native");
        }
        "release" => {
            // Release profile: full optimization
            build.flag_if_supported("-O3");
        }
        _ => {
            // Dev profile: no optimization for faster compilation
            build.flag_if_supported("-O0");
            build.flag_if_supported("-g"); // Include debug symbols
        }
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

    // Note: GPU object files are now compiled into a separate shared library
    // (libpecos_quest_cuda.so) which is built by build_gpu_shared_library()
    // and loaded at runtime via dlopen.

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
