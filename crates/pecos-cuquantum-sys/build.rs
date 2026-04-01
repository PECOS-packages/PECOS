//! Build script for pecos-cuquantum-sys
//!
//! Generates FFI bindings to cuQuantum using bindgen.

use std::env;
use std::path::PathBuf;

fn main() {
    env_logger::init();
    println!("cargo:rerun-if-env-changed=CUQUANTUM_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo::rustc-check-cfg=cfg(cuquantum_stub)");

    // Find cuQuantum installation
    let cuquantum_path = match pecos_build::cuquantum::find_cuquantum() {
        Some(path) => path,
        None => {
            // If CUDA is available, try auto-installing cuQuantum
            if pecos_build::cuda::find_cuda().is_some() {
                match pecos_build::cuquantum::ensure_cuquantum() {
                    Ok(path) => {
                        println!(
                            "cargo:warning=Auto-installed cuQuantum to: {}",
                            path.display()
                        );
                        path
                    }
                    Err(e) => {
                        log::warn!("Failed to auto-install cuQuantum: {e}");
                        generate_stub_bindings();
                        return;
                    }
                }
            } else {
                log::info!("cuQuantum not found. Generating stub bindings.");
                log::info!("To use cuQuantum, either:");
                log::info!("  1. Set CUQUANTUM_ROOT environment variable");
                log::info!("  2. Install cuQuantum via: pecos install cuquantum");
                log::info!("  3. Install cuQuantum system-wide");

                generate_stub_bindings();
                return;
            }
        }
    };

    // Find CUDA installation (required for cuComplex.h etc.)
    let cuda_path = match pecos_build::cuda::find_cuda() {
        Some(path) => path,
        None => {
            log::info!("CUDA not found. cuQuantum requires CUDA.");
            generate_stub_bindings();
            return;
        }
    };

    log::info!("Using cuQuantum from: {}", cuquantum_path.display());
    log::info!("Using CUDA from: {}", cuda_path.display());

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
                log::info!("Using cuTensor from: {}", cutensor_path.display());
                println!("cargo:rustc-link-search=native={}", cutensor_lib.display());
                println!("cargo:cutensor_lib_dir={}", cutensor_lib.display());
            }
        }
        Err(e) => {
            log::warn!("cuTensor not found: {e}");
            log::warn!("cuTensorNet may fail to load at runtime without libcutensor.");
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
    println!("cargo::rustc-cfg=cuquantum_stub");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Stub bindings provide actual function implementations (not just declarations)
    // so the crate compiles AND links without the cuQuantum SDK. Constructor guards
    // in pecos-cuquantum prevent these stubs from being called at runtime.
    let stub_content = r#"
// Stub bindings - cuQuantum not available at build time
// These stubs provide function implementations that return error codes,
// allowing compilation and linking without the cuQuantum SDK installed.
// Constructor guards in pecos-cuquantum prevent these from being called at runtime.

use core::ffi::c_void;
use core::ffi::c_char;

// =============================================================================
// CUDA types
// =============================================================================

pub type cudaStream_t = *mut c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cudaDataType_t {
    CUDA_R_32F = 0,
    CUDA_R_64F = 1,
    CUDA_C_32F = 4,
    CUDA_C_64F = 5,
}

pub type cudaMemcpyKind = u32;
pub const cudaMemcpyKind_cudaMemcpyHostToHost: cudaMemcpyKind = 0;
pub const cudaMemcpyKind_cudaMemcpyHostToDevice: cudaMemcpyKind = 1;
pub const cudaMemcpyKind_cudaMemcpyDeviceToHost: cudaMemcpyKind = 2;
pub const cudaMemcpyKind_cudaMemcpyDeviceToDevice: cudaMemcpyKind = 3;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct cuComplex {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct cuDoubleComplex {
    pub x: f64,
    pub y: f64,
}

// =============================================================================
// CUDA runtime function stubs
// =============================================================================

/// # Safety
/// Stub: no-op, returns error. Pointers are not dereferenced.
pub unsafe extern "C" fn cudaMalloc(_dev_ptr: *mut *mut c_void, _size: usize) -> i32 { 1 }
/// # Safety
/// Stub: no-op. Pointers are not dereferenced.
pub unsafe extern "C" fn cudaFree(_dev_ptr: *mut c_void) -> i32 { 0 }
/// # Safety
/// Stub: no-op, returns error. Pointers are not dereferenced.
pub unsafe extern "C" fn cudaMemcpy(
    _dst: *mut c_void,
    _src: *const c_void,
    _count: usize,
    _kind: cudaMemcpyKind,
) -> i32 { 1 }
/// # Safety
/// Stub: no-op, returns error. Pointers are not dereferenced.
pub unsafe extern "C" fn cudaMemset(_dev_ptr: *mut c_void, _value: i32, _count: usize) -> i32 { 1 }
/// # Safety
/// Stub: no-op, returns error.
pub unsafe extern "C" fn cudaDeviceSynchronize() -> i32 { 1 }

// =============================================================================
// cuStateVec types
// =============================================================================

pub type custatevecHandle_t = *mut c_void;
pub type custatevecSamplerDescriptor_t = *mut c_void;

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

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum custatevecMatrixLayout_t {
    CUSTATEVEC_MATRIX_LAYOUT_COL = 0,
    CUSTATEVEC_MATRIX_LAYOUT_ROW = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum custatevecComputeType_t {
    CUSTATEVEC_COMPUTE_32F = 4,
    CUSTATEVEC_COMPUTE_64F = 5,
    CUSTATEVEC_COMPUTE_TF32 = 12,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum custatevecCollapseOp_t {
    CUSTATEVEC_COLLAPSE_NONE = 0,
    CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO = 1,
}

// =============================================================================
// cuStateVec function stubs
// =============================================================================

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custatevecCreate(
    _handle: *mut custatevecHandle_t,
) -> custatevecStatus_t {
    custatevecStatus_t::CUSTATEVEC_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: no-op. Pointers are not dereferenced.
pub unsafe extern "C" fn custatevecDestroy(
    _handle: custatevecHandle_t,
) -> custatevecStatus_t {
    custatevecStatus_t::CUSTATEVEC_STATUS_SUCCESS
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custatevecGetProperty(
    _type_: i32,
    _value: *mut i32,
) -> custatevecStatus_t {
    custatevecStatus_t::CUSTATEVEC_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custatevecInitializeStateVector(
    _handle: custatevecHandle_t,
    _sv: *mut c_void,
    _sv_data_type: cudaDataType_t,
    _n_index_bits: u32,
    _sv_type: i32,
) -> custatevecStatus_t {
    custatevecStatus_t::CUSTATEVEC_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custatevecApplyMatrixGetWorkspaceSize(
    _handle: custatevecHandle_t,
    _sv_data_type: cudaDataType_t,
    _n_index_bits: u32,
    _matrix: *const c_void,
    _matrix_data_type: cudaDataType_t,
    _layout: custatevecMatrixLayout_t,
    _adjoint: i32,
    _n_targets: u32,
    _n_controls: u32,
    _compute_type: custatevecComputeType_t,
    _extra_workspace_size_in_bytes: *mut usize,
) -> custatevecStatus_t {
    custatevecStatus_t::CUSTATEVEC_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custatevecApplyMatrix(
    _handle: custatevecHandle_t,
    _sv: *mut c_void,
    _sv_data_type: cudaDataType_t,
    _n_index_bits: u32,
    _matrix: *const c_void,
    _matrix_data_type: cudaDataType_t,
    _layout: custatevecMatrixLayout_t,
    _adjoint: i32,
    _targets: *const i32,
    _n_targets: u32,
    _controls: *const i32,
    _control_bit_values: *const i32,
    _n_controls: u32,
    _compute_type: custatevecComputeType_t,
    _extra_workspace: *mut c_void,
    _extra_workspace_size_in_bytes: usize,
) -> custatevecStatus_t {
    custatevecStatus_t::CUSTATEVEC_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custatevecMeasureOnZBasis(
    _handle: custatevecHandle_t,
    _sv: *mut c_void,
    _sv_data_type: cudaDataType_t,
    _n_index_bits: u32,
    _parity: *mut i32,
    _basis_bits: *const i32,
    _n_basis_bits: u32,
    _rand_num: f64,
    _collapse: custatevecCollapseOp_t,
) -> custatevecStatus_t {
    custatevecStatus_t::CUSTATEVEC_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custatevecBatchMeasure(
    _handle: custatevecHandle_t,
    _sv: *mut c_void,
    _sv_data_type: cudaDataType_t,
    _n_index_bits: u32,
    _bit_string: *mut i32,
    _bit_ordering: *const i32,
    _bit_string_len: u32,
    _rand_num: f64,
    _collapse: custatevecCollapseOp_t,
) -> custatevecStatus_t {
    custatevecStatus_t::CUSTATEVEC_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custatevecSamplerCreate(
    _handle: custatevecHandle_t,
    _sv: *const c_void,
    _sv_data_type: cudaDataType_t,
    _n_index_bits: u32,
    _sampler: *mut custatevecSamplerDescriptor_t,
    _n_max_shots: u32,
    _extra_workspace_size_in_bytes: *mut usize,
) -> custatevecStatus_t {
    custatevecStatus_t::CUSTATEVEC_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: no-op. Pointers are not dereferenced.
pub unsafe extern "C" fn custatevecSamplerDestroy(
    _sampler: custatevecSamplerDescriptor_t,
) -> custatevecStatus_t {
    custatevecStatus_t::CUSTATEVEC_STATUS_SUCCESS
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custatevecSamplerPreprocess(
    _handle: custatevecHandle_t,
    _sampler: custatevecSamplerDescriptor_t,
    _extra_workspace: *mut c_void,
    _extra_workspace_size_in_bytes: usize,
) -> custatevecStatus_t {
    custatevecStatus_t::CUSTATEVEC_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custatevecSamplerSample(
    _handle: custatevecHandle_t,
    _sampler: custatevecSamplerDescriptor_t,
    _bit_strings: *mut i64,
    _bit_ordering: *const i32,
    _bit_string_len: u32,
    _rand_nums: *const f64,
    _n_shots: u32,
    _output: i32,
) -> custatevecStatus_t {
    custatevecStatus_t::CUSTATEVEC_STATUS_NOT_INITIALIZED
}

// =============================================================================
// cuStabilizer types
// =============================================================================

pub type custabilizerHandle_t = *mut c_void;
pub type custabilizerCircuit_t = *mut c_void;
pub type custabilizerFrameSimulator_t = *mut c_void;

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

// =============================================================================
// cuStabilizer function stubs
// =============================================================================

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custabilizerCreate(
    _handle: *mut custabilizerHandle_t,
) -> custabilizerStatus_t {
    custabilizerStatus_t::CUSTABILIZER_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: no-op. Pointers are not dereferenced.
pub unsafe extern "C" fn custabilizerDestroy(
    _handle: custabilizerHandle_t,
) -> custabilizerStatus_t {
    custabilizerStatus_t::CUSTABILIZER_STATUS_SUCCESS
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custabilizerCircuitSizeFromString(
    _handle: custabilizerHandle_t,
    _str: *const c_char,
    _size: *mut i64,
) -> custabilizerStatus_t {
    custabilizerStatus_t::CUSTABILIZER_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custabilizerCreateCircuitFromString(
    _handle: custabilizerHandle_t,
    _str: *const c_char,
    _buffer: *mut c_void,
    _buffer_size: i64,
    _circuit: *mut custabilizerCircuit_t,
) -> custabilizerStatus_t {
    custabilizerStatus_t::CUSTABILIZER_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: no-op. Pointers are not dereferenced.
pub unsafe extern "C" fn custabilizerDestroyCircuit(
    _circuit: custabilizerCircuit_t,
) -> custabilizerStatus_t {
    custabilizerStatus_t::CUSTABILIZER_STATUS_SUCCESS
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custabilizerCreateFrameSimulator(
    _handle: custabilizerHandle_t,
    _num_qubits: i64,
    _num_shots: i64,
    _max_measurements: i64,
    _table_stride: i64,
    _frame_sim: *mut custabilizerFrameSimulator_t,
) -> custabilizerStatus_t {
    custabilizerStatus_t::CUSTABILIZER_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: no-op. Pointers are not dereferenced.
pub unsafe extern "C" fn custabilizerDestroyFrameSimulator(
    _frame_sim: custabilizerFrameSimulator_t,
) -> custabilizerStatus_t {
    custabilizerStatus_t::CUSTABILIZER_STATUS_SUCCESS
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn custabilizerFrameSimulatorApplyCircuit(
    _handle: custabilizerHandle_t,
    _frame_sim: custabilizerFrameSimulator_t,
    _circuit: custabilizerCircuit_t,
    _randomize: i32,
    _seed: u64,
    _x_table: *mut u32,
    _z_table: *mut u32,
    _m_table: *mut u32,
    _stream: cudaStream_t,
) -> custabilizerStatus_t {
    custabilizerStatus_t::CUSTABILIZER_STATUS_NOT_INITIALIZED
}

// =============================================================================
// cuTensorNet types
// =============================================================================

pub type cutensornetHandle_t = *mut c_void;
pub type cutensornetNetworkDescriptor_t = *mut c_void;
pub type cutensornetContractionOptimizerConfig_t = *mut c_void;
pub type cutensornetContractionOptimizerInfo_t = *mut c_void;
pub type cutensornetContractionPlan_t = *mut c_void;
pub type cutensornetWorkspaceDescriptor_t = *mut c_void;

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

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cutensornetComputeType_t {
    CUTENSORNET_COMPUTE_32F = 4,
    CUTENSORNET_COMPUTE_64F = 5,
    CUTENSORNET_COMPUTE_TF32 = 12,
    CUTENSORNET_COMPUTE_16BF = 14,
}

// =============================================================================
// cuTensorNet function stubs
// =============================================================================

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn cutensornetCreate(
    _handle: *mut cutensornetHandle_t,
) -> cutensornetStatus_t {
    cutensornetStatus_t::CUTENSORNET_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: no-op. Pointers are not dereferenced.
pub unsafe extern "C" fn cutensornetDestroy(
    _handle: cutensornetHandle_t,
) -> cutensornetStatus_t {
    cutensornetStatus_t::CUTENSORNET_STATUS_SUCCESS
}

/// # Safety
/// Stub: returns 0.
pub unsafe extern "C" fn cutensornetGetVersion() -> usize { 0 }

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn cutensornetCreateNetworkDescriptor(
    _handle: cutensornetHandle_t,
    _num_inputs: i32,
    _num_modes_in: *const i32,
    _extents_in: *const *const i64,
    _strides_in: *const *const i64,
    _modes_in: *const *const i32,
    _qualifiers_in: *const u32,
    _num_modes_out: i32,
    _extents_out: *const i64,
    _strides_out: *const i64,
    _modes_out: *const i32,
    _data_type: cudaDataType_t,
    _compute_type: cutensornetComputeType_t,
    _desc_net: *mut cutensornetNetworkDescriptor_t,
) -> cutensornetStatus_t {
    cutensornetStatus_t::CUTENSORNET_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: no-op. Pointers are not dereferenced.
pub unsafe extern "C" fn cutensornetDestroyNetworkDescriptor(
    _desc_net: cutensornetNetworkDescriptor_t,
) -> cutensornetStatus_t {
    cutensornetStatus_t::CUTENSORNET_STATUS_SUCCESS
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn cutensornetCreateWorkspaceDescriptor(
    _handle: cutensornetHandle_t,
    _workspace_desc: *mut cutensornetWorkspaceDescriptor_t,
) -> cutensornetStatus_t {
    cutensornetStatus_t::CUTENSORNET_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: no-op. Pointers are not dereferenced.
pub unsafe extern "C" fn cutensornetDestroyWorkspaceDescriptor(
    _workspace_desc: cutensornetWorkspaceDescriptor_t,
) -> cutensornetStatus_t {
    cutensornetStatus_t::CUTENSORNET_STATUS_SUCCESS
}

// =============================================================================
// cuDensityMat types
// =============================================================================

pub type cudensitymatHandle_t = *mut c_void;
pub type cudensitymatState_t = *mut c_void;
pub type cudensitymatOperator_t = *mut c_void;

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

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cudensitymatStatePurity_t {
    CUDENSITYMAT_STATE_PURITY_PURE = 0,
    CUDENSITYMAT_STATE_PURITY_MIXED = 1,
}

// =============================================================================
// cuDensityMat function stubs
// =============================================================================

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn cudensitymatCreate(
    _handle: *mut cudensitymatHandle_t,
) -> cudensitymatStatus_t {
    cudensitymatStatus_t::CUDENSITYMAT_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: no-op. Pointers are not dereferenced.
pub unsafe extern "C" fn cudensitymatDestroy(
    _handle: cudensitymatHandle_t,
) -> cudensitymatStatus_t {
    cudensitymatStatus_t::CUDENSITYMAT_STATUS_SUCCESS
}

/// # Safety
/// Stub: returns 0.
pub unsafe extern "C" fn cudensitymatGetVersion() -> usize { 0 }

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn cudensitymatCreateState(
    _handle: cudensitymatHandle_t,
    _purity: cudensitymatStatePurity_t,
    _num_space_modes: i32,
    _space_mode_extents: *const i64,
    _batch_size: i64,
    _data_type: cudaDataType_t,
    _state: *mut cudensitymatState_t,
) -> cudensitymatStatus_t {
    cudensitymatStatus_t::CUDENSITYMAT_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: no-op. Pointers are not dereferenced.
pub unsafe extern "C" fn cudensitymatDestroyState(
    _state: cudensitymatState_t,
) -> cudensitymatStatus_t {
    cudensitymatStatus_t::CUDENSITYMAT_STATUS_SUCCESS
}

/// # Safety
/// Stub: returns NOT_INITIALIZED. Pointers are not dereferenced.
pub unsafe extern "C" fn cudensitymatCreateOperator(
    _handle: cudensitymatHandle_t,
    _num_qubits: i32,
    _data_type: cudaDataType_t,
    _op: *mut cudensitymatOperator_t,
) -> cudensitymatStatus_t {
    cudensitymatStatus_t::CUDENSITYMAT_STATUS_NOT_INITIALIZED
}

/// # Safety
/// Stub: no-op. Pointers are not dereferenced.
pub unsafe extern "C" fn cudensitymatDestroyOperator(
    _op: cudensitymatOperator_t,
) -> cudensitymatStatus_t {
    cudensitymatStatus_t::CUDENSITYMAT_STATUS_SUCCESS
}
"#;

    std::fs::write(out_path.join("bindings.rs"), stub_content)
        .expect("Failed to write stub bindings");
}
