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

    // Find cuQuantum installation (no auto-install -- use `pecos install cuquantum`)
    let cuquantum_path = match pecos_build::cuquantum::find_cuquantum() {
        Some(path) => path,
        None => {
            log::info!("cuQuantum not found. Generating stub bindings.");
            log::info!("To install cuQuantum, run: pecos setup");
            generate_stub_bindings();
            return;
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

    // No static linking -- libraries are loaded at runtime via libloading.
    // Emit metadata so downstream build scripts can find library paths for rpath hints.

    // Emit metadata so downstream build scripts can read library paths
    // via DEP_PECOS_CUQUANTUM_SYS_CUQUANTUM_LIB_DIR
    println!("cargo:cuquantum_lib_dir={}", lib_dir.display());

    // cuTensor is required by cuTensorNet at runtime.
    // It's found by the runtime loader (not linked at build time).
    if let Some(cutensor_path) = pecos_build::cutensor::find_cutensor() {
        if let Some(cutensor_lib) = pecos_build::cutensor::get_lib_dir(&cutensor_path) {
            log::info!("Using cuTensor from: {}", cutensor_path.display());
            println!("cargo:cutensor_lib_dir={}", cutensor_lib.display());
        }
    } else {
        log::info!("cuTensor not found. Run: pecos setup");
    }

    // Emit CUDA lib dir metadata (for rpath hints in downstream crates)
    if let Some(cuda_lib) = get_cuda_lib_dir(&cuda_path) {
        println!("cargo:cuda_lib_dir={}", cuda_lib.display());
    }

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
        // Structs with function pointer fields can't meaningfully derive Eq/Hash
        .no_partialeq("custatevecDeviceMemHandler_t")
        .no_partialeq("cutensornetDeviceMemHandler_t")
        .no_partialeq("cutensornetDistributedInterface_t")
        .no_partialeq("cudensitymatDistributedInterface_t")
        .no_partialeq("cudensitymatWrappedScalarCallback_t")
        .no_partialeq("cudensitymatWrappedTensorCallback_t")
        .no_partialeq("cudensitymatWrappedScalarGradientCallback_t")
        .no_partialeq("cudensitymatWrappedTensorGradientCallback_t")
        .no_hash("custatevecDeviceMemHandler_t")
        .no_hash("cutensornetDeviceMemHandler_t")
        .no_hash("cutensornetDistributedInterface_t")
        .no_hash("cudensitymatDistributedInterface_t")
        .no_hash("cudensitymatWrappedScalarCallback_t")
        .no_hash("cudensitymatWrappedTensorCallback_t")
        .no_hash("cudensitymatWrappedScalarGradientCallback_t")
        .no_hash("cudensitymatWrappedTensorGradientCallback_t")
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

    // Stub bindings: type definitions only. Functions are loaded at runtime
    // via libloading in the loader module.
    let stub_content = r#"
// Stub type definitions - cuQuantum not available at build time.
// Functions are loaded at runtime via the loader module.

use core::ffi::c_void;

// --- CUDA types ---

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

// --- cuStateVec types ---

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

// --- cuStabilizer types ---

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

// --- cuTensorNet types ---

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

// --- cuDensityMat types ---

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
"#;

    std::fs::write(out_path.join("bindings.rs"), stub_content)
        .expect("Failed to write stub bindings");
}
