use log::warn;
use pecos_build_utils::{
    boost_download_info, download_cached, eigen_download_info, extract_archive,
    qulacs_download_info,
};
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    // Initialize logger for build script
    env_logger::init();

    setup_rerun_conditions();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap_or_default();
    let is_windows = target.contains("windows");

    // Download and extract dependencies
    let (qulacs_path, eigen_path, boost_path) = download_and_extract_dependencies(&out_dir);

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

fn download_and_extract_dependencies(out_dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    // Download all dependencies
    let qulacs_data = download_cached(&qulacs_download_info()).expect("Failed to download Qulacs");
    let eigen_data = download_cached(&eigen_download_info()).expect("Failed to download Eigen");
    let boost_data = download_cached(&boost_download_info()).expect("Failed to download Boost");

    // Extract archives
    let qulacs_path =
        extract_archive(&qulacs_data, out_dir, Some("qulacs")).expect("Failed to extract Qulacs");
    let eigen_path =
        extract_archive(&eigen_data, out_dir, Some("eigen")).expect("Failed to extract Eigen");
    let boost_path =
        extract_archive(&boost_data, out_dir, Some("boost")).expect("Failed to extract Boost");

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

    // Set compiler flags
    if is_windows {
        // Windows-specific settings
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

        // Use standard optimization level - /bigobj should prevent compiler crashes
        build.opt_level(2); // Maximize speed optimization (/O2)
    } else {
        build.flag_if_supported("-std=c++14");
        build.flag_if_supported("-O3");

        // On macOS with ARM (Apple Silicon), -ffast-math causes issues with Eigen's NEON code
        // which uses infinity constants. Use a more targeted optimization instead.
        if target.contains("darwin") && target.contains("aarch64") {
            // Enable fast math but allow infinity and NaN
            build.flag_if_supported("-fno-math-errno");
            build.flag_if_supported("-fno-trapping-math");
        } else {
            build.flag_if_supported("-ffast-math");
        }

        // Silence OpenMP pragma warnings since we intentionally don't use OpenMP
        // PECOS uses thread-level parallelism instead of OpenMP's internal parallelism
        build.flag_if_supported("-Wno-unknown-pragmas");

        // Suppress specific warnings from third-party libraries (Eigen, Boost, Qulacs)
        build.flag_if_supported("-Wno-unused-but-set-variable"); // Eigen SparseLU warnings
        build.flag_if_supported("-Wno-deprecated-copy-with-user-provided-copy"); // Boost warnings
        build.flag_if_supported("-Wno-unqualified-std-cast-call"); // Qulacs move() warnings
        build.flag_if_supported("-Wno-inconsistent-missing-override"); // Qulacs override warnings

        // On macOS, use the -stdlib=libc++ flag to ensure proper C++ standard library linkage
        if target.contains("darwin") {
            build.flag("-stdlib=libc++");
            // Note: Linker flags are passed via cargo:rustc-link-arg below, not here
        }
    }

    // Define preprocessor macros
    build.define("EIGEN_NO_DEBUG", None);
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
