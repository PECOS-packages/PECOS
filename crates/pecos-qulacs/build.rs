use log::warn;
use pecos_build::{Manifest, ensure_dep_ready};
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    // Initialize logger for build script
    env_logger::init();

    setup_rerun_conditions();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap_or_default();
    let is_windows = target.contains("windows");

    // Ensure dependencies are downloaded and extracted to ~/.pecos/deps/
    let (qulacs_path, eigen_path, boost_path) = download_and_extract_dependencies();

    // Build our wrapper with actual Qulacs
    let mut build = cxx_build::bridge("src/bridge.rs");

    // Add our wrapper
    build.file("src/qulacs_wrapper.cpp");

    // Add essential Qulacs source files
    let qulacs_src = qulacs_path.join("src");
    add_qulacs_source_files(&mut build, &qulacs_src);

    // Configure includes and compiler flags
    configure_build(
        &mut build,
        &eigen_path,
        &boost_path,
        &qulacs_src,
        &out_dir,
        is_windows,
        &target,
    );

    // Compile everything
    build.compile("qulacs_wrapper");

    // Add Windows-specific boost exception stub if needed
    if is_windows {
        create_windows_boost_stub(&out_dir);
    }

    // On macOS, link against the system C++ library from dyld shared cache
    if target.contains("darwin") {
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-arg=-Wl,-search_paths_first");
    }
}

fn setup_rerun_conditions() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=src/qulacs_wrapper.cpp");
    println!("cargo:rerun-if-changed=src/qulacs_wrapper.h");
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

fn download_and_extract_dependencies() -> (PathBuf, PathBuf, PathBuf) {
    // Load manifest (crate-local or workspace-level, with validation)
    let manifest =
        Manifest::find_and_load_validated().expect("pecos.toml not found or validation failed");

    // Ensure dependencies are downloaded and extracted to ~/.pecos/deps/
    // This persists across `cargo clean` for faster rebuilds
    let qulacs_path = ensure_dep_ready("qulacs", &manifest).expect("Failed to get Qulacs");
    let eigen_path = ensure_dep_ready("eigen", &manifest).expect("Failed to get Eigen");
    let boost_path = ensure_dep_ready("boost", &manifest).expect("Failed to get Boost");

    (qulacs_path, eigen_path, boost_path)
}

fn add_qulacs_source_files(build: &mut cc::Build, qulacs_src: &Path) {
    // Core cppsim files - only add files that exist
    let cppsim_files = vec![
        "state.cpp",
        "state_dm.cpp", // Added: contains state::from_ptree implementation
        "gate.cpp",
        "gate_factory.cpp",
        "gate_matrix.cpp",
        "gate_named_one.cpp",
        "utility.cpp",
        "circuit.cpp",
        "qubit_info.cpp",
        "gate_matrix_sparse.cpp",
        "gate_matrix_diagonal.cpp",
        "gate_merge.cpp",
        "pauli_operator.cpp",
        "general_quantum_operator.cpp",
        "observable.cpp",
        "gate_noisy_evolution.cpp",
    ];

    for file in &cppsim_files {
        let path = qulacs_src.join("cppsim").join(file);
        if path.exists() {
            build.file(path);
        } else {
            warn!("Skipping missing file: cppsim/{file}");
        }
    }

    // Core csim files - these are the actual files present in Qulacs 0.6.12
    let csim_files = vec![
        "memory_ops.cpp",
        "stat_ops.cpp",
        "update_ops_named.cpp",
        "update_ops_named_X.cpp",
        "update_ops_named_Y.cpp",
        "update_ops_named_Z.cpp",
        "update_ops_named_H.cpp",
        "update_ops_named_CNOT.cpp",
        "update_ops_named_CZ.cpp",
        "update_ops_named_SWAP.cpp",
        "update_ops_named_state.cpp",
        "update_ops_matrix_dense_single.cpp",
        "update_ops_pauli_single.cpp",
        "stat_ops_probability.cpp",
        "utility.cpp",
        "init_ops_fill.cpp",
        "init_ops_random.cpp",
        "update_ops_matrix_dense_double.cpp",
        "update_ops_matrix_diagonal_single.cpp",
        "update_ops_matrix_phase_single.cpp",
        "update_ops_matrix_dense_multi.cpp",
        "update_ops_matrix_diagonal_multi.cpp",
        "update_ops_pauli_multi.cpp",
        "stat_ops_expectation_value.cpp",
        "stat_ops_transition_amplitude.cpp",
        "update_ops_dm.cpp",
        "memory_ops_dm.cpp",
        "stat_ops_dm.cpp",
        "constant.cpp",
        // Files that were missing but actually exist in Qulacs 0.6.12
        "update_ops_control_single_target_single.cpp",
        "update_ops_control_single_target_multi.cpp",
        "update_ops_control_multi_target_single.cpp",
        "update_ops_control_multi_target_multi.cpp",
        "update_ops_named_FusedSWAP.cpp",
        "update_ops_reflection.cpp",
        "update_ops_reversible_boolean.cpp",
        "update_ops_qft.cpp",
        "update_ops_named_projection.cpp",
        "update_ops_matrix_dense_double_eigen.cpp",
        "update_ops_matrix_dense_multi_eigen.cpp",
    ];

    for file in &csim_files {
        let path = qulacs_src.join("csim").join(file);
        if path.exists() {
            build.file(path);
        } else {
            warn!("Skipping missing file: csim/{file}");
        }
    }
}

fn configure_build(
    build: &mut cc::Build,
    eigen_path: &Path,
    boost_path: &Path,
    qulacs_src: &Path,
    out_dir: &Path,
    is_windows: bool,
    target: &str,
) {
    // Include directories
    build.include(eigen_path);
    build.include(boost_path);
    build.include(qulacs_src);
    build.include(qulacs_src.join("cppsim"));
    build.include(qulacs_src.join("csim"));
    build.include("src");
    build.include(out_dir);

    // Configure the C++ compiler based on platform.
    // - macOS: MUST use system clang (/usr/bin/clang++) which has proper SDK paths.
    //   PECOS's bundled clang doesn't have macOS SDK headers configured (missing math.h, etc.)
    //   and the cc crate will find PECOS clang first if it's in PATH.
    // - Windows: Use MSVC (default). PECOS's bundled clang-cl is LLVM 14, but MSVC 2022's STL
    //   requires Clang 19.0.0+ when using clang-cl, causing "STL1000: Unexpected compiler version".
    // - Linux: Use system GCC (PECOS clang can't find system GCC headers for libstdc++)
    // Only override if CXX/CC env vars are not already set (allow user override).
    if env::var("CXX").is_err() && env::var("CC").is_err() && target.contains("darwin") {
        // On macOS, explicitly use system clang to ensure SDK paths are correct.
        // The PECOS LLVM clang may be in PATH but doesn't have SDK headers.
        build.compiler("/usr/bin/clang++");
    }
    // On Windows and Linux, use the default compiler (MSVC on Windows, GCC on Linux)

    // Get the build profile for optimization decisions
    let profile = get_build_profile();
    let is_release = profile == "release" || profile == "native";

    // Set compiler flags based on platform and compiler
    if is_windows {
        // MSVC-specific settings
        build.std("c++14");
        // Define Boost exception handling for Windows
        build.define("BOOST_NO_EXCEPTIONS", None);
        build.define("_USE_MATH_DEFINES", None);
        // Windows needs these for proper linking
        build.define("_WINDOWS", None);
        build.define("NOMINMAX", None);

        // Fix MSVC compiler crash with Eigen templates
        build.flag("/bigobj"); // Allow larger object files
        build.flag("/EHsc"); // Enable exception handling
        build.flag("/Z7"); // Embed debug info in .obj files (no PDB) - required for parallel builds

        // Suppress warnings from external headers (Eigen, Boost, Qulacs)
        build.flag_if_supported("/external:anglebrackets"); // Treat angle-bracket includes as external
        build.flag_if_supported("/external:W0"); // Disable warnings for external headers

        // Use optimization level based on Cargo profile
        if is_release {
            build.opt_level(2); // Maximize speed optimization (/O2)
        } else {
            build.opt_level(0); // No optimization for debug builds
        }
    } else {
        build.flag_if_supported("-std=c++14");

        // Use profile-based optimization settings
        match profile.as_str() {
            "native" => {
                // Native profile: release optimizations + CPU-specific optimizations
                build.flag_if_supported("-O3");
                build.flag_if_supported("-march=native"); // CPU-specific optimizations
            }
            "release" => {
                // Release profile: optimized build
                build.flag_if_supported("-O3");
            }
            _ => {
                // Dev profile: no optimization flags for fastest compile times
            }
        }
        // Debug builds use cc crate's default (no optimization flags)

        // Safe math optimizations (don't cause ICEs, provide modest speedup)
        // Applied to all profiles
        build.flag_if_supported("-fno-math-errno");
        build.flag_if_supported("-fno-trapping-math");

        // Suppress all warnings from third-party C++ code (Qulacs, Eigen, Boost)
        build.warnings(false);

        // On macOS, use libc++ (the system default and what PECOS clang expects)
        if target.contains("darwin") {
            build.flag("-stdlib=libc++");
            // Note: Linker flags are passed via cargo:rustc-link-arg below, not here
        }
        // On Linux, use system default (libstdc++) - no flag needed
    }

    // Define preprocessor macros - only disable Eigen debug checks in release mode
    if is_release {
        build.define("EIGEN_NO_DEBUG", None);
    }

    // Enable SIMD-optimized gate kernels in Qulacs (matches Qulacs CMake USE_SIMD=Yes).
    // _USE_SIMD activates hand-written SIMD intrinsics for gates like H, X, CNOT, RZ, etc.
    // On x86/x86_64, Qulacs's type.hpp will #undef _USE_SIMD if the compiler doesn't define
    // __AVX2__, so this is safe even when -march=native isn't used.
    if target.contains("x86_64")
        || target.contains("x86")
        || target.contains("i686")
        || target.contains("aarch64")
    {
        build.define("_USE_SIMD", None);
    }
}

fn create_windows_boost_stub(out_dir: &Path) {
    println!("cargo:rustc-link-lib=static=qulacs_wrapper");
    // Create a simple boost exception handler stub
    std::fs::write(
        out_dir.join("boost_exception_stub.cpp"),
        r#"
        #include <exception>
        namespace boost {
            struct source_location {
                const char* file_name() const { return ""; }
                const char* function_name() const { return ""; }
                int line() const { return 0; }
            };
            void throw_exception(std::exception const& e, source_location const&) {
                throw e;
            }
        }
        "#,
    )
    .expect("Failed to write boost exception stub");

    // Compile the stub
    cc::Build::new()
        .cpp(true)
        .file(out_dir.join("boost_exception_stub.cpp"))
        .std("c++14")
        .compile("boost_exception_stub");
}
