//! Runtime loader for cuQuantum libraries.
//!
//! Loads cuQuantum shared libraries at runtime via `libloading`,
//! enabling a single binary to work on both systems with and without
//! NVIDIA cuQuantum installed.

use crate::*;
use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::OnceLock;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum CuQuantumLoadError {
    #[error("cuQuantum libraries not found. Searched: {searched_paths}")]
    LibraryNotFound { searched_paths: String },
    #[error("Failed to load {lib_name}: {reason}")]
    LoadFailed { lib_name: String, reason: String },
    #[error("Missing symbol {symbol} in {lib_name}: {reason}")]
    MissingSymbol {
        lib_name: String,
        symbol: String,
        reason: String,
    },
}

type LoadResult<T> = std::result::Result<T, CuQuantumLoadError>;

macro_rules! load_sym {
    ($lib:expr, $lib_name:expr, $name:literal) => {{
        let sym: Symbol<_> = $lib
            .get($name)
            .map_err(|e| CuQuantumLoadError::MissingSymbol {
                lib_name: $lib_name.into(),
                symbol: String::from_utf8_lossy($name)
                    .trim_end_matches('\0')
                    .to_string(),
                reason: e.to_string(),
            })?;
        *sym
    }};
}

/// Loaded cuQuantum function pointers.
#[allow(non_snake_case)]
pub struct CuQuantumBackend {
    // Keep libraries alive
    _cuda_rt: Library,
    _custatevec: Library,
    _custabilizer: Library,
    _cutensornet: Library,
    _cudensitymat: Library,

    // --- CUDA runtime ---
    pub cudaMalloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> i32,
    pub cudaFree: unsafe extern "C" fn(*mut c_void) -> i32,
    pub cudaMemcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, cudaMemcpyKind) -> i32,
    pub cudaMemset: unsafe extern "C" fn(*mut c_void, i32, usize) -> i32,
    pub cudaDeviceSynchronize: unsafe extern "C" fn() -> i32,

    // --- cuStateVec ---
    pub custatevecCreate: unsafe extern "C" fn(*mut custatevecHandle_t) -> custatevecStatus_t,
    pub custatevecDestroy: unsafe extern "C" fn(custatevecHandle_t) -> custatevecStatus_t,
    pub custatevecApplyMatrix: unsafe extern "C" fn(
        custatevecHandle_t,
        *mut c_void,
        cudaDataType_t,
        u32,
        *const c_void,
        cudaDataType_t,
        custatevecMatrixLayout_t,
        i32,
        *const i32,
        u32,
        *const i32,
        *const i32,
        u32,
        custatevecComputeType_t,
        *mut c_void,
        usize,
    ) -> custatevecStatus_t,
    pub custatevecMeasureOnZBasis: unsafe extern "C" fn(
        custatevecHandle_t,
        *mut c_void,
        cudaDataType_t,
        u32,
        *mut i32,
        *const i32,
        u32,
        f64,
        custatevecCollapseOp_t,
    ) -> custatevecStatus_t,
    pub custatevecBatchMeasure: unsafe extern "C" fn(
        custatevecHandle_t,
        *mut c_void,
        cudaDataType_t,
        u32,
        *mut i32,
        *const i32,
        u32,
        f64,
        custatevecCollapseOp_t,
    ) -> custatevecStatus_t,

    // --- cuStabilizer ---
    pub custabilizerCreate: unsafe extern "C" fn(*mut custabilizerHandle_t) -> custabilizerStatus_t,
    pub custabilizerDestroy: unsafe extern "C" fn(custabilizerHandle_t) -> custabilizerStatus_t,
    pub custabilizerCircuitSizeFromString: unsafe extern "C" fn(
        custabilizerHandle_t,
        *const std::ffi::c_char,
        *mut i64,
    ) -> custabilizerStatus_t,
    pub custabilizerCreateCircuitFromString: unsafe extern "C" fn(
        custabilizerHandle_t,
        *const std::ffi::c_char,
        *mut c_void,
        i64,
        *mut custabilizerCircuit_t,
    ) -> custabilizerStatus_t,
    pub custabilizerDestroyCircuit:
        unsafe extern "C" fn(custabilizerCircuit_t) -> custabilizerStatus_t,
    pub custabilizerCreateFrameSimulator: unsafe extern "C" fn(
        custabilizerHandle_t,
        i64,
        i64,
        i64,
        i64,
        *mut custabilizerFrameSimulator_t,
    ) -> custabilizerStatus_t,
    pub custabilizerDestroyFrameSimulator:
        unsafe extern "C" fn(custabilizerFrameSimulator_t) -> custabilizerStatus_t,
    pub custabilizerFrameSimulatorApplyCircuit: unsafe extern "C" fn(
        custabilizerHandle_t,
        custabilizerFrameSimulator_t,
        custabilizerCircuit_t,
        i32,
        u64,
        *mut u32,
        *mut u32,
        *mut u32,
        cudaStream_t,
    ) -> custabilizerStatus_t,

    // --- cuTensorNet ---
    pub cutensornetCreate: unsafe extern "C" fn(*mut cutensornetHandle_t) -> cutensornetStatus_t,
    pub cutensornetDestroy: unsafe extern "C" fn(cutensornetHandle_t) -> cutensornetStatus_t,
    pub cutensornetGetVersion: unsafe extern "C" fn() -> usize,

    // --- cuDensityMat ---
    pub cudensitymatCreate: unsafe extern "C" fn(*mut cudensitymatHandle_t) -> cudensitymatStatus_t,
    pub cudensitymatDestroy: unsafe extern "C" fn(cudensitymatHandle_t) -> cudensitymatStatus_t,
    pub cudensitymatGetVersion: unsafe extern "C" fn() -> usize,
    pub cudensitymatCreateState: unsafe extern "C" fn(
        cudensitymatHandle_t,
        cudensitymatStatePurity_t,
        i32,
        *const i64,
        i64,
        cudaDataType_t,
        *mut cudensitymatState_t,
    ) -> cudensitymatStatus_t,
    pub cudensitymatDestroyState: unsafe extern "C" fn(cudensitymatState_t) -> cudensitymatStatus_t,
}

// SAFETY: Function pointers are Copy and the Library handles just need to stay alive.
// The CUDA/cuQuantum functions themselves handle thread safety internally.
unsafe impl Send for CuQuantumBackend {}
unsafe impl Sync for CuQuantumBackend {}

static BACKEND: OnceLock<Result<CuQuantumBackend, CuQuantumLoadError>> = OnceLock::new();

/// Load cuQuantum libraries. Thread-safe, loads only once.
pub fn try_load() -> Result<&'static CuQuantumBackend, &'static CuQuantumLoadError> {
    BACKEND.get_or_init(load_all).as_ref()
}

/// Check if cuQuantum is available at runtime.
pub fn is_available() -> bool {
    try_load().is_ok()
}

fn cuda_search_paths() -> Vec<PathBuf> {
    let mut paths = vec![];
    if let Ok(p) = std::env::var("CUDA_PATH") {
        paths.push(PathBuf::from(&p).join("lib64"));
        paths.push(PathBuf::from(&p).join("lib"));
    }
    if let Some(home) = dirs::home_dir() {
        paths.push(home.join(".pecos/deps/cuda/lib64"));
    }
    paths.push(PathBuf::from("/usr/local/cuda/lib64"));
    paths
}

fn cuquantum_search_paths() -> Vec<PathBuf> {
    let mut paths = vec![];
    if let Ok(p) = std::env::var("CUQUANTUM_ROOT") {
        paths.push(PathBuf::from(&p).join("lib64"));
        paths.push(PathBuf::from(&p).join("lib"));
    }
    if let Some(home) = dirs::home_dir() {
        paths.push(home.join(".pecos/deps/cuquantum/lib64"));
        paths.push(home.join(".pecos/deps/cuquantum/lib"));
        paths.push(home.join(".pecos/cuquantum/lib64"));
        paths.push(home.join(".pecos/cuquantum/lib"));
    }
    paths.push(PathBuf::from("/usr/local/cuquantum/lib64"));
    paths.push(PathBuf::from("/opt/nvidia/cuquantum/lib64"));
    paths
}

fn cutensor_search_paths() -> Vec<PathBuf> {
    let mut paths = vec![];
    if let Some(home) = dirs::home_dir() {
        paths.push(home.join(".pecos/deps/cutensor/lib"));
        paths.push(home.join(".pecos/deps/cutensor/lib64"));
    }
    paths.push(PathBuf::from("/usr/local/cutensor/lib"));
    paths
}

fn try_load_lib(name: &str, search_dirs: &[PathBuf]) -> LoadResult<Library> {
    // Try each search directory with the library name appended
    for dir in search_dirs {
        let path = dir.join(name);
        log::debug!("Trying to load {name} from: {}", path.display());
        match unsafe { Library::new(&path) } {
            Ok(lib) => {
                log::info!("Loaded {name} from: {}", path.display());
                return Ok(lib);
            }
            Err(e) => log::debug!("  Failed: {e}"),
        }
    }
    // Fall back to bare name (system linker search)
    log::debug!("Trying system path for {name}");
    unsafe { Library::new(name) }.map_err(|_| {
        let searched = search_dirs
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        CuQuantumLoadError::LibraryNotFound {
            searched_paths: format!("{name} not found in: {searched}, or system paths"),
        }
    })
}

fn load_all() -> Result<CuQuantumBackend, CuQuantumLoadError> {
    let cuda_paths = cuda_search_paths();
    let cq_paths = cuquantum_search_paths();
    let ct_paths = cutensor_search_paths();

    // Load CUDA runtime first (transitive dependency for everything else)
    let cuda_rt = try_load_lib("libcudart.so", &cuda_paths)?;

    // Load cuTensor before cuTensorNet (transitive dependency)
    // cuTensor may not be needed if we're not using cuTensorNet, but load it
    // to ensure cuTensorNet can resolve its symbols.
    let _cutensor = try_load_lib("libcutensor.so", &ct_paths).ok();

    let custatevec = try_load_lib("libcustatevec.so", &cq_paths)?;
    let custabilizer = try_load_lib("libcustabilizer.so", &cq_paths)?;
    let cutensornet = try_load_lib("libcutensornet.so", &cq_paths)?;
    let cudensitymat = try_load_lib("libcudensitymat.so", &cq_paths)?;

    unsafe {
        Ok(CuQuantumBackend {
            // CUDA runtime
            cudaMalloc: load_sym!(cuda_rt, "libcudart", b"cudaMalloc\0"),
            cudaFree: load_sym!(cuda_rt, "libcudart", b"cudaFree\0"),
            cudaMemcpy: load_sym!(cuda_rt, "libcudart", b"cudaMemcpy\0"),
            cudaMemset: load_sym!(cuda_rt, "libcudart", b"cudaMemset\0"),
            cudaDeviceSynchronize: load_sym!(cuda_rt, "libcudart", b"cudaDeviceSynchronize\0"),

            // cuStateVec
            custatevecCreate: load_sym!(custatevec, "libcustatevec", b"custatevecCreate\0"),
            custatevecDestroy: load_sym!(custatevec, "libcustatevec", b"custatevecDestroy\0"),
            custatevecApplyMatrix: load_sym!(
                custatevec,
                "libcustatevec",
                b"custatevecApplyMatrix\0"
            ),
            custatevecMeasureOnZBasis: load_sym!(
                custatevec,
                "libcustatevec",
                b"custatevecMeasureOnZBasis\0"
            ),
            custatevecBatchMeasure: load_sym!(
                custatevec,
                "libcustatevec",
                b"custatevecBatchMeasure\0"
            ),

            // cuStabilizer
            custabilizerCreate: load_sym!(custabilizer, "libcustabilizer", b"custabilizerCreate\0"),
            custabilizerDestroy: load_sym!(
                custabilizer,
                "libcustabilizer",
                b"custabilizerDestroy\0"
            ),
            custabilizerCircuitSizeFromString: load_sym!(
                custabilizer,
                "libcustabilizer",
                b"custabilizerCircuitSizeFromString\0"
            ),
            custabilizerCreateCircuitFromString: load_sym!(
                custabilizer,
                "libcustabilizer",
                b"custabilizerCreateCircuitFromString\0"
            ),
            custabilizerDestroyCircuit: load_sym!(
                custabilizer,
                "libcustabilizer",
                b"custabilizerDestroyCircuit\0"
            ),
            custabilizerCreateFrameSimulator: load_sym!(
                custabilizer,
                "libcustabilizer",
                b"custabilizerCreateFrameSimulator\0"
            ),
            custabilizerDestroyFrameSimulator: load_sym!(
                custabilizer,
                "libcustabilizer",
                b"custabilizerDestroyFrameSimulator\0"
            ),
            custabilizerFrameSimulatorApplyCircuit: load_sym!(
                custabilizer,
                "libcustabilizer",
                b"custabilizerFrameSimulatorApplyCircuit\0"
            ),

            // cuTensorNet
            cutensornetCreate: load_sym!(cutensornet, "libcutensornet", b"cutensornetCreate\0"),
            cutensornetDestroy: load_sym!(cutensornet, "libcutensornet", b"cutensornetDestroy\0"),
            cutensornetGetVersion: load_sym!(
                cutensornet,
                "libcutensornet",
                b"cutensornetGetVersion\0"
            ),

            // cuDensityMat
            cudensitymatCreate: load_sym!(cudensitymat, "libcudensitymat", b"cudensitymatCreate\0"),
            cudensitymatDestroy: load_sym!(
                cudensitymat,
                "libcudensitymat",
                b"cudensitymatDestroy\0"
            ),
            cudensitymatGetVersion: load_sym!(
                cudensitymat,
                "libcudensitymat",
                b"cudensitymatGetVersion\0"
            ),
            cudensitymatCreateState: load_sym!(
                cudensitymat,
                "libcudensitymat",
                b"cudensitymatCreateState\0"
            ),
            cudensitymatDestroyState: load_sym!(
                cudensitymat,
                "libcudensitymat",
                b"cudensitymatDestroyState\0"
            ),

            _cuda_rt: cuda_rt,
            _custatevec: custatevec,
            _custabilizer: custabilizer,
            _cutensornet: cutensornet,
            _cudensitymat: cudensitymat,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_load_returns_result() {
        // Should not panic regardless of whether cuQuantum is installed
        let _ = try_load();
    }

    #[test]
    fn is_available_returns_bool() {
        let _ = is_available();
    }

    #[test]
    fn search_paths_not_empty() {
        assert!(!cuda_search_paths().is_empty());
        assert!(!cuquantum_search_paths().is_empty());
    }
}
