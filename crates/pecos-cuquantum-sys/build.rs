//! Build script for pecos-cuquantum-sys
//!
//! Generates FFI bindings to cuQuantum using bindgen.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CUQUANTUM_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // Find cuQuantum installation
    let cuquantum_path = match pecos_build::cuquantum::find_cuquantum() {
        Some(path) => path,
        None => {
            // If cuQuantum is not found, generate stub bindings
            eprintln!("Warning: cuQuantum not found. Generating stub bindings.");
            eprintln!("To use cuQuantum, either:");
            eprintln!("  1. Set CUQUANTUM_ROOT environment variable");
            eprintln!("  2. Install cuQuantum to ~/.pecos/cuquantum/");
            eprintln!("  3. Install cuQuantum system-wide");

            generate_stub_bindings();
            return;
        }
    };

    // Find CUDA installation (required for cuComplex.h etc.)
    let cuda_path = match pecos_build::cuda::find_cuda() {
        Some(path) => path,
        None => {
            eprintln!("Warning: CUDA not found. cuQuantum requires CUDA.");
            generate_stub_bindings();
            return;
        }
    };

    println!(
        "cargo:warning=Using cuQuantum from: {}",
        cuquantum_path.display()
    );
    println!("cargo:warning=Using CUDA from: {}", cuda_path.display());

    // Get library directory
    let lib_dir = pecos_build::cuquantum::get_lib_dir(&cuquantum_path)
        .expect("Could not find cuQuantum lib directory");

    // Set up link paths
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=custatevec");
    println!("cargo:rustc-link-lib=custabilizer");
    println!("cargo:rustc-link-lib=cutensornet");
    println!("cargo:rustc-link-lib=cudensitymat");

    // Emit metadata so downstream build scripts can read library paths
    // via DEP_PECOS_CUQUANTUM_SYS_CUQUANTUM_LIB_DIR
    println!("cargo:cuquantum_lib_dir={}", lib_dir.display());

    // cuTensor is required by cuTensorNet at runtime.
    // Find or install it to ~/.pecos/deps/cutensor-<version>/
    match pecos_build::cutensor::ensure_cutensor() {
        Ok(cutensor_path) => {
            if let Some(cutensor_lib) = pecos_build::cutensor::get_lib_dir(&cutensor_path) {
                println!(
                    "cargo:warning=Using cuTensor from: {}",
                    cutensor_path.display()
                );
                println!("cargo:rustc-link-search=native={}", cutensor_lib.display());
                println!("cargo:cutensor_lib_dir={}", cutensor_lib.display());
            }
        }
        Err(e) => {
            eprintln!("Warning: cuTensor not found: {e}");
            eprintln!("cuTensorNet may fail to load at runtime without libcutensor.");
        }
    }

    // Also need CUDA runtime
    if let Some(cuda_lib) = get_cuda_lib_dir(&cuda_path) {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        println!("cargo:cuda_lib_dir={}", cuda_lib.display());
    }
    println!("cargo:rustc-link-lib=cudart");

    // Generate bindings
    let cuquantum_include = pecos_build::cuquantum::get_include_dir(&cuquantum_path);
    let cuda_include = cuda_path.join("include");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", cuquantum_include.display()))
        .clang_arg(format!("-I{}", cuda_include.display()))
        // cuStateVec API
        .allowlist_function("custatevec.*")
        .allowlist_type("custatevec.*")
        .allowlist_var("CUSTATEVEC_.*")
        // cuStabilizer API
        .allowlist_function("custabilizer.*")
        .allowlist_type("custabilizer.*")
        .allowlist_var("CUSTABILIZER_.*")
        // cuTensorNet API
        .allowlist_function("cutensornet.*")
        .allowlist_type("cutensornet.*")
        .allowlist_var("CUTENSORNET_.*")
        // cuDensityMat API
        .allowlist_function("cudensitymat.*")
        .allowlist_type("cudensitymat.*")
        .allowlist_var("CUDENSITYMAT_.*")
        // CUDA types we need
        .allowlist_type("cudaStream_t")
        .allowlist_type("cuComplex")
        .allowlist_type("cuDoubleComplex")
        .allowlist_type("cudaDataType_t")
        .allowlist_type("cudaDataType")
        .allowlist_type("cudaMemcpyKind")
        // CUDA runtime functions we need for memory management
        .allowlist_function("cudaMalloc")
        .allowlist_function("cudaFree")
        .allowlist_function("cudaMemcpy")
        .allowlist_function("cudaMemset")
        .allowlist_function("cudaDeviceSynchronize")
        .allowlist_var("cudaMemcpyHostToDevice")
        .allowlist_var("cudaMemcpyDeviceToHost")
        .allowlist_var("cudaMemcpyDeviceToDevice")
        // Derive traits
        .derive_debug(true)
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        // Use core instead of std where possible
        .use_core()
        // Generate rustified enums
        .rustified_enum("custatevec.*")
        .rustified_enum("custabilizer.*")
        .rustified_enum("cutensornet.*")
        .rustified_enum("cudensitymat.*")
        .rustified_enum("cudaDataType.*")
        // Block system headers we don't need
        .blocklist_file(".*/bits/.*")
        .blocklist_file(".*/sys/.*")
        // Block logger functions that use FILE* type
        .blocklist_function(".*LoggerSetFile.*")
        .blocklist_function(".*LoggerOpenFile.*")
        // Disable doc comment generation to avoid doctest issues
        .generate_comments(false)
        // Parse callbacks
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Failed to generate bindings");

    // Write bindings to OUT_DIR
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings");
}

/// Get CUDA library directory
fn get_cuda_lib_dir(cuda_path: &std::path::Path) -> Option<PathBuf> {
    let lib64 = cuda_path.join("lib64");
    if lib64.exists() {
        return Some(lib64);
    }

    let lib = cuda_path.join("lib");
    if lib.exists() {
        return Some(lib);
    }

    // On Windows, might be lib/x64
    let lib_x64 = cuda_path.join("lib").join("x64");
    if lib_x64.exists() {
        return Some(lib_x64);
    }

    None
}

/// Generate stub bindings when cuQuantum is not available
fn generate_stub_bindings() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Note: No inner doc comments or inner attributes since this is included via include!()
    // Also use `unsafe extern "C"` for Rust 2024 edition
    let stub_content = r#"
// Stub bindings - cuQuantum not available at build time
// These stubs allow the crate to compile without cuQuantum installed,
// but any attempt to use the functions will fail at link time.

use core::ffi::c_void;

/// Opaque handle type for cuStateVec
pub type custatevecHandle_t = *mut c_void;

/// CUDA stream type
pub type cudaStream_t = *mut c_void;

/// cuStateVec status codes
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum custatevecStatus_t {
    CUSTATEVEC_STATUS_SUCCESS = 0,
    CUSTATEVEC_STATUS_NOT_INITIALIZED = 1,
    CUSTATEVEC_STATUS_ALLOC_FAILED = 2,
    CUSTATEVEC_STATUS_INVALID_VALUE = 3,
    CUSTATEVEC_STATUS_ARCH_MISMATCH = 4,
    CUSTATEVEC_STATUS_EXECUTION_FAILED = 5,
    CUSTATEVEC_STATUS_INTERNAL_ERROR = 6,
    CUSTATEVEC_STATUS_NOT_SUPPORTED = 7,
    CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE = 8,
    CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED = 9,
    CUSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR = 10,
    CUSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR = 11,
    CUSTATEVEC_STATUS_COMMUNICATOR_ERROR = 12,
    CUSTATEVEC_STATUS_LOADING_LIBRARY_FAILED = 13,
    CUSTATEVEC_STATUS_MAX_VALUE = 14,
}

/// CUDA data types
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cudaDataType_t {
    CUDA_R_32F = 0,
    CUDA_R_64F = 1,
    CUDA_C_32F = 4,
    CUDA_C_64F = 5,
}

/// Complex float (stub)
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct cuComplex {
    pub x: f32,
    pub y: f32,
}

/// Complex double (stub)
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct cuDoubleComplex {
    pub x: f64,
    pub y: f64,
}

// =============================================================================
// cuStabilizer types
// =============================================================================

/// Opaque handle type for cuStabilizer
pub type custabilizerHandle_t = *mut c_void;

/// Opaque state type for cuStabilizer
pub type custabilizerState_t = *mut c_void;

/// cuStabilizer status codes
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum custabilizerStatus_t {
    CUSTABILIZER_STATUS_SUCCESS = 0,
    CUSTABILIZER_STATUS_NOT_INITIALIZED = 1,
    CUSTABILIZER_STATUS_ALLOC_FAILED = 2,
    CUSTABILIZER_STATUS_INVALID_VALUE = 3,
    CUSTABILIZER_STATUS_ARCH_MISMATCH = 4,
    CUSTABILIZER_STATUS_EXECUTION_FAILED = 5,
    CUSTABILIZER_STATUS_INTERNAL_ERROR = 6,
    CUSTABILIZER_STATUS_NOT_SUPPORTED = 7,
    CUSTABILIZER_STATUS_INSUFFICIENT_WORKSPACE = 8,
    CUSTABILIZER_STATUS_MAX_VALUE = 9,
}

/// Pauli operator types for cuStabilizer
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum custabilizerPauli_t {
    CUSTABILIZER_PAULI_I = 0,
    CUSTABILIZER_PAULI_X = 1,
    CUSTABILIZER_PAULI_Y = 2,
    CUSTABILIZER_PAULI_Z = 3,
}

// =============================================================================
// Stub function declarations - these will fail at link time if called
// =============================================================================

/// Matrix layout for cuStateVec
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum custatevecMatrixLayout_t {
    CUSTATEVEC_MATRIX_LAYOUT_COL = 0,
    CUSTATEVEC_MATRIX_LAYOUT_ROW = 1,
}

/// Compute type for cuStateVec operations
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum custatevecComputeType_t {
    CUSTATEVEC_COMPUTE_32F = 4,
    CUSTATEVEC_COMPUTE_64F = 5,
    CUSTATEVEC_COMPUTE_TF32 = 12,
}

/// Collapse operation for measurement
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum custatevecCollapseOp_t {
    CUSTATEVEC_COLLAPSE_NONE = 0,
    CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO = 1,
}

/// Sampler descriptor type (opaque)
pub type custatevecSamplerDescriptor_t = *mut c_void;

// cuStateVec functions
unsafe extern "C" {
    pub fn custatevecCreate(handle: *mut custatevecHandle_t) -> custatevecStatus_t;
    pub fn custatevecDestroy(handle: custatevecHandle_t) -> custatevecStatus_t;
    pub fn custatevecGetProperty(
        type_: i32,
        value: *mut i32,
    ) -> custatevecStatus_t;

    // State initialization
    pub fn custatevecInitializeStateVector(
        handle: custatevecHandle_t,
        sv: *mut c_void,
        sv_data_type: cudaDataType_t,
        n_index_bits: u32,
        sv_type: i32,  // custatevecStateVectorType_t
    ) -> custatevecStatus_t;

    // Matrix application
    pub fn custatevecApplyMatrixGetWorkspaceSize(
        handle: custatevecHandle_t,
        sv_data_type: cudaDataType_t,
        n_index_bits: u32,
        matrix: *const c_void,
        matrix_data_type: cudaDataType_t,
        layout: custatevecMatrixLayout_t,
        adjoint: i32,
        n_targets: u32,
        n_controls: u32,
        compute_type: custatevecComputeType_t,
        extra_workspace_size_in_bytes: *mut usize,
    ) -> custatevecStatus_t;

    pub fn custatevecApplyMatrix(
        handle: custatevecHandle_t,
        sv: *mut c_void,
        sv_data_type: cudaDataType_t,
        n_index_bits: u32,
        matrix: *const c_void,
        matrix_data_type: cudaDataType_t,
        layout: custatevecMatrixLayout_t,
        adjoint: i32,
        targets: *const i32,
        n_targets: u32,
        controls: *const i32,
        control_bit_values: *const i32,
        n_controls: u32,
        compute_type: custatevecComputeType_t,
        extra_workspace: *mut c_void,
        extra_workspace_size_in_bytes: usize,
    ) -> custatevecStatus_t;

    // Measurement on Z basis
    pub fn custatevecMeasureOnZBasis(
        handle: custatevecHandle_t,
        sv: *mut c_void,
        sv_data_type: cudaDataType_t,
        n_index_bits: u32,
        parity: *mut i32,
        basis_bits: *const i32,
        n_basis_bits: u32,
        rand_num: f64,
        collapse: custatevecCollapseOp_t,
    ) -> custatevecStatus_t;

    // Batch measurement
    pub fn custatevecBatchMeasure(
        handle: custatevecHandle_t,
        sv: *mut c_void,
        sv_data_type: cudaDataType_t,
        n_index_bits: u32,
        bit_string: *mut i32,
        bit_ordering: *const i32,
        bit_string_len: u32,
        rand_num: f64,
        collapse: custatevecCollapseOp_t,
    ) -> custatevecStatus_t;

    // Sampling
    pub fn custatevecSamplerCreate(
        handle: custatevecHandle_t,
        sv: *const c_void,
        sv_data_type: cudaDataType_t,
        n_index_bits: u32,
        sampler: *mut custatevecSamplerDescriptor_t,
        n_max_shots: u32,
        extra_workspace_size_in_bytes: *mut usize,
    ) -> custatevecStatus_t;

    pub fn custatevecSamplerDestroy(
        sampler: custatevecSamplerDescriptor_t,
    ) -> custatevecStatus_t;

    pub fn custatevecSamplerPreprocess(
        handle: custatevecHandle_t,
        sampler: custatevecSamplerDescriptor_t,
        extra_workspace: *mut c_void,
        extra_workspace_size_in_bytes: usize,
    ) -> custatevecStatus_t;

    pub fn custatevecSamplerSample(
        handle: custatevecHandle_t,
        sampler: custatevecSamplerDescriptor_t,
        bit_strings: *mut i64,
        bit_ordering: *const i32,
        bit_string_len: u32,
        rand_nums: *const f64,
        n_shots: u32,
        output: i32,  // custatevecSamplerOutput_t
    ) -> custatevecStatus_t;
}

// CUDA runtime functions for memory management
unsafe extern "C" {
    pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(dev_ptr: *mut c_void) -> i32;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    pub fn cudaMemset(dev_ptr: *mut c_void, value: i32, count: usize) -> i32;
    pub fn cudaDeviceSynchronize() -> i32;
}

/// CUDA memory copy kinds
pub const CUDA_MEMCPY_HOST_TO_HOST: i32 = 0;
pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

// cuStabilizer functions
unsafe extern "C" {
    pub fn custabilizerCreate(handle: *mut custabilizerHandle_t) -> custabilizerStatus_t;
    pub fn custabilizerDestroy(handle: custabilizerHandle_t) -> custabilizerStatus_t;
    pub fn custabilizerStateCreate(
        handle: custabilizerHandle_t,
        state: *mut custabilizerState_t,
        num_qubits: u32,
    ) -> custabilizerStatus_t;
    pub fn custabilizerStateDestroy(
        handle: custabilizerHandle_t,
        state: custabilizerState_t,
    ) -> custabilizerStatus_t;
    pub fn custabilizerApplyPauli(
        handle: custabilizerHandle_t,
        state: custabilizerState_t,
        pauli: custabilizerPauli_t,
        qubit: u32,
    ) -> custabilizerStatus_t;
    pub fn custabilizerApplyH(
        handle: custabilizerHandle_t,
        state: custabilizerState_t,
        qubit: u32,
    ) -> custabilizerStatus_t;
    pub fn custabilizerApplyS(
        handle: custabilizerHandle_t,
        state: custabilizerState_t,
        qubit: u32,
    ) -> custabilizerStatus_t;
    pub fn custabilizerApplySdg(
        handle: custabilizerHandle_t,
        state: custabilizerState_t,
        qubit: u32,
    ) -> custabilizerStatus_t;
    pub fn custabilizerApplyCNOT(
        handle: custabilizerHandle_t,
        state: custabilizerState_t,
        control: u32,
        target: u32,
    ) -> custabilizerStatus_t;
    pub fn custabilizerApplyCZ(
        handle: custabilizerHandle_t,
        state: custabilizerState_t,
        qubit_a: u32,
        qubit_b: u32,
    ) -> custabilizerStatus_t;
    pub fn custabilizerMeasure(
        handle: custabilizerHandle_t,
        state: custabilizerState_t,
        qubit: u32,
        outcome: *mut i32,
    ) -> custabilizerStatus_t;
    pub fn custabilizerReset(
        handle: custabilizerHandle_t,
        state: custabilizerState_t,
    ) -> custabilizerStatus_t;
}

// =============================================================================
// cuTensorNet types
// =============================================================================

/// Opaque handle type for cuTensorNet
pub type cutensornetHandle_t = *mut c_void;

/// Opaque network descriptor type
pub type cutensornetNetworkDescriptor_t = *mut c_void;

/// Opaque contraction optimizer config type
pub type cutensornetContractionOptimizerConfig_t = *mut c_void;

/// Opaque contraction optimizer info type
pub type cutensornetContractionOptimizerInfo_t = *mut c_void;

/// Opaque contraction plan type
pub type cutensornetContractionPlan_t = *mut c_void;

/// Opaque workspace descriptor type
pub type cutensornetWorkspaceDescriptor_t = *mut c_void;

/// cuTensorNet status codes
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cutensornetStatus_t {
    CUTENSORNET_STATUS_SUCCESS = 0,
    CUTENSORNET_STATUS_NOT_INITIALIZED = 1,
    CUTENSORNET_STATUS_ALLOC_FAILED = 2,
    CUTENSORNET_STATUS_INVALID_VALUE = 3,
    CUTENSORNET_STATUS_ARCH_MISMATCH = 4,
    CUTENSORNET_STATUS_MAPPING_ERROR = 5,
    CUTENSORNET_STATUS_EXECUTION_FAILED = 6,
    CUTENSORNET_STATUS_INTERNAL_ERROR = 7,
    CUTENSORNET_STATUS_NOT_SUPPORTED = 8,
    CUTENSORNET_STATUS_LICENSE_ERROR = 9,
    CUTENSORNET_STATUS_CUBLAS_ERROR = 10,
    CUTENSORNET_STATUS_CUDA_ERROR = 11,
    CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE = 12,
    CUTENSORNET_STATUS_INSUFFICIENT_DRIVER = 13,
    CUTENSORNET_STATUS_IO_ERROR = 14,
    CUTENSORNET_STATUS_CUTENSOR_ERROR = 15,
    CUTENSORNET_STATUS_MAX_VALUE = 16,
}

/// Compute type for cuTensorNet operations
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cutensornetComputeType_t {
    CUTENSORNET_COMPUTE_32F = 4,
    CUTENSORNET_COMPUTE_64F = 5,
    CUTENSORNET_COMPUTE_TF32 = 12,
    CUTENSORNET_COMPUTE_16BF = 14,
}

// cuTensorNet functions
unsafe extern "C" {
    pub fn cutensornetCreate(handle: *mut cutensornetHandle_t) -> cutensornetStatus_t;
    pub fn cutensornetDestroy(handle: cutensornetHandle_t) -> cutensornetStatus_t;
    pub fn cutensornetGetVersion() -> usize;
    pub fn cutensornetCreateNetworkDescriptor(
        handle: cutensornetHandle_t,
        num_inputs: i32,
        num_modes_in: *const i32,
        extents_in: *const *const i64,
        strides_in: *const *const i64,
        modes_in: *const *const i32,
        qualifiers_in: *const u32,
        num_modes_out: i32,
        extents_out: *const i64,
        strides_out: *const i64,
        modes_out: *const i32,
        data_type: cudaDataType_t,
        compute_type: cutensornetComputeType_t,
        desc_net: *mut cutensornetNetworkDescriptor_t,
    ) -> cutensornetStatus_t;
    pub fn cutensornetDestroyNetworkDescriptor(
        desc_net: cutensornetNetworkDescriptor_t,
    ) -> cutensornetStatus_t;
    pub fn cutensornetCreateWorkspaceDescriptor(
        handle: cutensornetHandle_t,
        workspace_desc: *mut cutensornetWorkspaceDescriptor_t,
    ) -> cutensornetStatus_t;
    pub fn cutensornetDestroyWorkspaceDescriptor(
        workspace_desc: cutensornetWorkspaceDescriptor_t,
    ) -> cutensornetStatus_t;
}

// =============================================================================
// cuDensityMat types
// =============================================================================

/// Opaque handle type for cuDensityMat
pub type cudensitymatHandle_t = *mut c_void;

/// Opaque state type for cuDensityMat
pub type cudensitymatState_t = *mut c_void;

/// Opaque operator type for cuDensityMat
pub type cudensitymatOperator_t = *mut c_void;

/// cuDensityMat status codes
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cudensitymatStatus_t {
    CUDENSITYMAT_STATUS_SUCCESS = 0,
    CUDENSITYMAT_STATUS_NOT_INITIALIZED = 1,
    CUDENSITYMAT_STATUS_ALLOC_FAILED = 2,
    CUDENSITYMAT_STATUS_INVALID_VALUE = 3,
    CUDENSITYMAT_STATUS_ARCH_MISMATCH = 4,
    CUDENSITYMAT_STATUS_EXECUTION_FAILED = 5,
    CUDENSITYMAT_STATUS_INTERNAL_ERROR = 6,
    CUDENSITYMAT_STATUS_NOT_SUPPORTED = 7,
    CUDENSITYMAT_STATUS_CUBLAS_ERROR = 8,
    CUDENSITYMAT_STATUS_CUDA_ERROR = 9,
    CUDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE = 10,
    CUDENSITYMAT_STATUS_MAX_VALUE = 11,
}

/// State purity type
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cudensitymatStatePurity_t {
    CUDENSITYMAT_STATE_PURITY_PURE = 0,
    CUDENSITYMAT_STATE_PURITY_MIXED = 1,
}

// cuDensityMat functions
unsafe extern "C" {
    pub fn cudensitymatCreate(handle: *mut cudensitymatHandle_t) -> cudensitymatStatus_t;
    pub fn cudensitymatDestroy(handle: cudensitymatHandle_t) -> cudensitymatStatus_t;
    pub fn cudensitymatGetVersion() -> usize;
    pub fn cudensitymatCreateState(
        handle: cudensitymatHandle_t,
        purity: cudensitymatStatePurity_t,
        num_qubits: i32,
        data_type: cudaDataType_t,
        state: *mut cudensitymatState_t,
    ) -> cudensitymatStatus_t;
    pub fn cudensitymatDestroyState(
        state: cudensitymatState_t,
    ) -> cudensitymatStatus_t;
    pub fn cudensitymatCreateOperator(
        handle: cudensitymatHandle_t,
        num_qubits: i32,
        data_type: cudaDataType_t,
        op: *mut cudensitymatOperator_t,
    ) -> cudensitymatStatus_t;
    pub fn cudensitymatDestroyOperator(
        op: cudensitymatOperator_t,
    ) -> cudensitymatStatus_t;
}
"#;

    std::fs::write(out_path.join("bindings.rs"), stub_content)
        .expect("Failed to write stub bindings");
}
