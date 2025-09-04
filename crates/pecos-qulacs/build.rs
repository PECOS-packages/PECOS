use pecos_build_utils::{
    boost_download_info, download_cached, eigen_download_info, extract_archive,
    qulacs_download_info,
};
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=src/qulacs_wrapper.cpp");
    println!("cargo:rerun-if-changed=src/qulacs_wrapper.h");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap_or_default();
    let is_windows = target.contains("windows");

    // Download all dependencies
    let qulacs_data = download_cached(&qulacs_download_info()).expect("Failed to download Qulacs");
    let eigen_data = download_cached(&eigen_download_info()).expect("Failed to download Eigen");
    let boost_data = download_cached(&boost_download_info()).expect("Failed to download Boost");

    // Extract archives
    let qulacs_path =
        extract_archive(&qulacs_data, &out_dir, Some("qulacs")).expect("Failed to extract Qulacs");
    let eigen_path =
        extract_archive(&eigen_data, &out_dir, Some("eigen")).expect("Failed to extract Eigen");
    let boost_path =
        extract_archive(&boost_data, &out_dir, Some("boost")).expect("Failed to extract Boost");

    // Build our wrapper with actual Qulacs
    let mut build = cxx_build::bridge("src/bridge.rs");

    // Add our wrapper
    build.file("src/qulacs_wrapper.cpp");

    // Add essential Qulacs source files
    let qulacs_src = qulacs_path.join("src");

    // Core cppsim files
    build.file(qulacs_src.join("cppsim/state.cpp"));
    build.file(qulacs_src.join("cppsim/gate.cpp"));
    build.file(qulacs_src.join("cppsim/gate_factory.cpp"));
    build.file(qulacs_src.join("cppsim/gate_matrix.cpp"));
    build.file(qulacs_src.join("cppsim/gate_named_one.cpp"));
    build.file(qulacs_src.join("cppsim/utility.cpp"));
    build.file(qulacs_src.join("cppsim/circuit.cpp"));
    build.file(qulacs_src.join("cppsim/qubit_info.cpp"));
    
    // Add missing gate implementation files for Windows
    build.file(qulacs_src.join("cppsim/gate_matrix_sparse.cpp"));
    build.file(qulacs_src.join("cppsim/gate_matrix_diagonal.cpp"));
    build.file(qulacs_src.join("cppsim/gate_named_two.cpp"));
    build.file(qulacs_src.join("cppsim/gate_named_pauli.cpp"));
    build.file(qulacs_src.join("cppsim/gate_merge.cpp"));
    build.file(qulacs_src.join("cppsim/gate_reversible.cpp"));
    build.file(qulacs_src.join("cppsim/gate_reflect.cpp"));
    
    // Add quantum operator files
    build.file(qulacs_src.join("cppsim/pauli_operator.cpp"));
    build.file(qulacs_src.join("cppsim/general_quantum_operator.cpp"));
    build.file(qulacs_src.join("cppsim/hermitian_quantum_operator.cpp"));
    build.file(qulacs_src.join("cppsim/observable.cpp"));
    
    // Add noisy evolution files  
    build.file(qulacs_src.join("cppsim/gate_noisy_evolution.cpp"));

    // Core csim files
    build.file(qulacs_src.join("csim/memory_ops.cpp"));
    build.file(qulacs_src.join("csim/stat_ops.cpp"));
    build.file(qulacs_src.join("csim/update_ops_named.cpp"));
    build.file(qulacs_src.join("csim/update_ops_named_X.cpp"));
    build.file(qulacs_src.join("csim/update_ops_named_Y.cpp"));
    build.file(qulacs_src.join("csim/update_ops_named_Z.cpp"));
    build.file(qulacs_src.join("csim/update_ops_named_H.cpp"));
    build.file(qulacs_src.join("csim/update_ops_named_CNOT.cpp"));
    build.file(qulacs_src.join("csim/update_ops_named_CZ.cpp"));
    build.file(qulacs_src.join("csim/update_ops_named_SWAP.cpp"));
    build.file(qulacs_src.join("csim/update_ops_named_state.cpp"));
    build.file(qulacs_src.join("csim/update_ops_matrix_dense_single.cpp"));
    build.file(qulacs_src.join("csim/update_ops_pauli_single.cpp"));
    build.file(qulacs_src.join("csim/stat_ops_probability.cpp"));

    // Additional missing utility files
    build.file(qulacs_src.join("csim/utility.cpp"));
    build.file(qulacs_src.join("csim/init_ops_fill.cpp"));
    build.file(qulacs_src.join("csim/init_ops_random.cpp"));

    // Matrix operations that might be needed for gates
    build.file(qulacs_src.join("csim/update_ops_matrix_dense_double.cpp"));
    build.file(qulacs_src.join("csim/update_ops_matrix_diagonal_single.cpp"));
    build.file(qulacs_src.join("csim/update_ops_matrix_phase_single.cpp"));
    build.file(qulacs_src.join("csim/update_ops_control_single_target.cpp"));
    build.file(qulacs_src.join("csim/update_ops_control_multi_target.cpp"));
    build.file(qulacs_src.join("csim/update_ops_matrix_dense_multi.cpp"));
    build.file(qulacs_src.join("csim/update_ops_matrix_sparse.cpp"));
    build.file(qulacs_src.join("csim/update_ops_matrix_diagonal_multi.cpp"));
    build.file(qulacs_src.join("csim/update_ops_matrix_diagonal_double.cpp"));
    
    // Pauli operations needed for quantum operators
    build.file(qulacs_src.join("csim/update_ops_pauli_multi.cpp"));
    build.file(qulacs_src.join("csim/stat_ops_expectation_value.cpp"));
    build.file(qulacs_src.join("csim/stat_ops_transition_amplitude.cpp"));

    // Density matrix operations (apparently needed by gate factory)
    build.file(qulacs_src.join("csim/update_ops_dm.cpp"));
    build.file(qulacs_src.join("csim/memory_ops_dm.cpp"));
    build.file(qulacs_src.join("csim/stat_ops_dm.cpp"));

    // Constants needed by operations
    build.file(qulacs_src.join("csim/constant.cpp"));
    
    // Special gate operations referenced in errors
    build.file(qulacs_src.join("csim/update_ops_P0_P1.cpp"));
    build.file(qulacs_src.join("csim/update_ops_rotate.cpp"));
    build.file(qulacs_src.join("csim/update_ops_FusedSWAP.cpp"));
    build.file(qulacs_src.join("csim/update_ops_reflection.cpp"));
    build.file(qulacs_src.join("csim/update_ops_reversible.cpp"));

    // Include directories
    build.include(&eigen_path);
    build.include(&boost_path);
    build.include(&qulacs_src);
    build.include(qulacs_src.join("cppsim"));
    build.include(qulacs_src.join("csim"));
    build.include("src");
    build.include(&out_dir);

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
    } else {
        build.flag_if_supported("-std=c++14");
        build.flag_if_supported("-O3");
        build.flag_if_supported("-ffast-math");
        // Silence OpenMP pragma warnings since we intentionally don't use OpenMP
        // PECOS uses thread-level parallelism instead of OpenMP's internal parallelism
        build.flag_if_supported("-Wno-unknown-pragmas");
    }

    // Define preprocessor macros
    build.define("EIGEN_NO_DEBUG", None);

    // Compile everything
    build.compile("qulacs_wrapper");
    
    // Add a stub implementation for boost::throw_exception for Windows
    if is_windows {
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
        ).expect("Failed to write boost exception stub");
        
        // Compile the stub
        cc::Build::new()
            .cpp(true)
            .file(out_dir.join("boost_exception_stub.cpp"))
            .std("c++14")
            .compile("boost_exception_stub");
    }
}
