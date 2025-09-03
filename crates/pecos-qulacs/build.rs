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

    // Core cppsim files (only ones that exist)
    build.file(qulacs_src.join("cppsim/state.cpp"));
    build.file(qulacs_src.join("cppsim/gate.cpp"));
    build.file(qulacs_src.join("cppsim/gate_factory.cpp"));
    build.file(qulacs_src.join("cppsim/gate_matrix.cpp"));
    build.file(qulacs_src.join("cppsim/gate_named_one.cpp"));
    build.file(qulacs_src.join("cppsim/utility.cpp"));
    build.file(qulacs_src.join("cppsim/circuit.cpp"));
    build.file(qulacs_src.join("cppsim/qubit_info.cpp"));

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

    // Density matrix operations (apparently needed by gate factory)
    build.file(qulacs_src.join("csim/update_ops_dm.cpp"));
    build.file(qulacs_src.join("csim/memory_ops_dm.cpp"));
    build.file(qulacs_src.join("csim/stat_ops_dm.cpp"));

    // Constants needed by operations
    build.file(qulacs_src.join("csim/constant.cpp"));

    // Include directories
    build.include(&eigen_path);
    build.include(&boost_path);
    build.include(&qulacs_src);
    build.include(qulacs_src.join("cppsim"));
    build.include(qulacs_src.join("csim"));
    build.include("src");
    build.include(&out_dir);

    // Set compiler flags
    build.flag_if_supported("-std=c++14");
    build.flag_if_supported("-O3");
    build.flag_if_supported("-ffast-math");

    // Silence OpenMP pragma warnings since we intentionally don't use OpenMP
    // PECOS uses thread-level parallelism instead of OpenMP's internal parallelism
    build.flag_if_supported("-Wno-unknown-pragmas");

    // Define preprocessor macros
    // Note: _USE_MATH_DEFINES is already defined in Qulacs source files
    // to avoid redefinition warnings, we let Qulacs handle this internally
    build.define("EIGEN_NO_DEBUG", None);

    // Compile everything
    build.compile("qulacs_wrapper");
}
