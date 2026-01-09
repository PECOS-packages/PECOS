//! Runtime loader for the CUDA-accelerated `QuEST` backend
//!
//! This module provides functionality to dynamically load the PECOS `QuEST` CUDA
//! backend library at runtime, enabling a single binary to work on both systems
//! with and without NVIDIA CUDA installed.

use libloading::{Library, Symbol};
use std::path::PathBuf;
use std::sync::OnceLock;
use thiserror::Error;

/// Errors that can occur when loading the `QuEST` CUDA backend
#[derive(Error, Debug, Clone)]
pub enum CudaLoadError {
    #[error("QuEST CUDA backend not found. Searched paths: {searched_paths}")]
    LibraryNotFound { searched_paths: String },

    #[error("Failed to load QuEST CUDA backend: {0}")]
    LoadFailed(String),

    #[error("Missing symbol in QuEST CUDA backend: {0}")]
    MissingSymbol(String),

    #[error("NVIDIA CUDA runtime not available: {0}")]
    CudaUnavailable(String),
}

/// Result type for CUDA loading operations
pub type CudaResult<T> = std::result::Result<T, CudaLoadError>;

/// `QuEST` CUDA backend that holds the loaded library and function pointers
pub struct CudaBackend {
    /// Keep the backend library loaded for the lifetime of this struct
    _library: Library,

    // Function pointers for QuEST CUDA backend operations
    // Environment management
    pub create_env: unsafe extern "C" fn() -> *mut u8,
    pub destroy_env: unsafe extern "C" fn(*mut u8),
    pub get_env_info: unsafe extern "C" fn(*mut u8) -> CudaEnvInfo,

    // Qureg management
    pub create_qureg: unsafe extern "C" fn(*mut u8, i32) -> *mut u8,
    pub create_density_qureg: unsafe extern "C" fn(*mut u8, i32) -> *mut u8,
    pub destroy_qureg: unsafe extern "C" fn(*mut u8),

    // State initialization
    pub init_zero_state: unsafe extern "C" fn(*mut u8),
    pub init_plus_state: unsafe extern "C" fn(*mut u8),
    pub init_classical_state: unsafe extern "C" fn(*mut u8, i64),

    // Single-qubit gates
    pub apply_pauli_x: unsafe extern "C" fn(*mut u8, i32),
    pub apply_pauli_y: unsafe extern "C" fn(*mut u8, i32),
    pub apply_pauli_z: unsafe extern "C" fn(*mut u8, i32),
    pub apply_hadamard: unsafe extern "C" fn(*mut u8, i32),
    pub apply_s_gate: unsafe extern "C" fn(*mut u8, i32),
    pub apply_t_gate: unsafe extern "C" fn(*mut u8, i32),
    pub apply_phase_shift: unsafe extern "C" fn(*mut u8, i32, f64),

    // Rotation gates
    pub apply_rotation_x: unsafe extern "C" fn(*mut u8, i32, f64),
    pub apply_rotation_y: unsafe extern "C" fn(*mut u8, i32, f64),
    pub apply_rotation_z: unsafe extern "C" fn(*mut u8, i32, f64),

    // Two-qubit gates
    pub apply_cnot: unsafe extern "C" fn(*mut u8, i32, i32),
    pub apply_cz: unsafe extern "C" fn(*mut u8, i32, i32),
    pub apply_swap: unsafe extern "C" fn(*mut u8, i32, i32),
    pub apply_controlled_phase_shift: unsafe extern "C" fn(*mut u8, i32, i32, f64),

    // Measurement
    pub measure: unsafe extern "C" fn(*mut u8, i32) -> i32,
    pub calc_prob_of_outcome: unsafe extern "C" fn(*mut u8, i32, i32) -> f64,
    pub apply_forced_measurement: unsafe extern "C" fn(*mut u8, i32, i32) -> f64,

    // Amplitude access
    pub get_real_amp: unsafe extern "C" fn(*mut u8, i64) -> f64,
    pub get_imag_amp: unsafe extern "C" fn(*mut u8, i64) -> f64,
    pub get_prob_amp: unsafe extern "C" fn(*mut u8, i64) -> f64,
    pub calc_total_prob: unsafe extern "C" fn(*mut u8) -> f64,
    pub calc_purity: unsafe extern "C" fn(*mut u8) -> f64,

    // Info
    pub get_num_amps: unsafe extern "C" fn(*mut u8) -> i64,
    pub get_num_qubits: unsafe extern "C" fn(*mut u8) -> i32,
    pub is_density_matrix: unsafe extern "C" fn(*mut u8) -> bool,
}

/// CUDA environment info returned by the CUDA library
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CudaEnvInfo {
    pub is_multithreaded: bool,
    pub is_gpu_accelerated: bool,
    pub is_distributed: bool,
    pub rank: i32,
    pub num_nodes: i32,
}

/// Global CUDA backend instance (lazily initialized)
static CUDA_BACKEND: OnceLock<Result<CudaBackend, CudaLoadError>> = OnceLock::new();

/// Wrapper for the CUDA environment handle that can be shared across threads
///
/// # Safety
/// The CUDA environment handle is thread-safe through `QuEST`'s internal synchronization.
/// All operations on the environment go through the CUDA backend functions which handle
/// synchronization appropriately.
struct SharedEnvHandle(*mut u8);

// SAFETY: The CUDA environment handle is thread-safe through QuEST's internal synchronization
// and is only accessed through the loaded CUDA backend functions which are also thread-safe.
unsafe impl Send for SharedEnvHandle {}
unsafe impl Sync for SharedEnvHandle {}

/// Shared CUDA environment handle (lazily initialized, never destroyed)
///
/// `QuEST`'s CUDA environment has issues with destruction and recreation - once destroyed,
/// subsequent attempts to create a new environment often fail. This static environment
/// is shared across all `QuestCudaStateVecEngine` instances and persists for the lifetime
/// of the process. Only the quantum registers (quregs) are created/destroyed per engine.
static SHARED_CUDA_ENV: OnceLock<SharedEnvHandle> = OnceLock::new();

/// Get or create the shared CUDA environment
///
/// This function returns the shared CUDA environment handle, creating it if necessary.
/// The environment is never destroyed, avoiding `QuEST` CUDA recreation issues.
///
/// # Errors
/// Returns `CudaLoadError` if:
/// - The CUDA backend cannot be loaded
/// - The environment cannot be created
pub fn get_shared_cuda_env() -> Result<(*mut u8, &'static CudaBackend), CudaLoadError> {
    let backend = try_load_cuda().map_err(std::clone::Clone::clone)?;

    let env = SHARED_CUDA_ENV.get_or_init(|| {
        let env_handle = unsafe { (backend.create_env)() };
        if env_handle.is_null() {
            log::error!("Failed to create shared CUDA QuEST environment");
            SharedEnvHandle(std::ptr::null_mut())
        } else {
            log::info!("Created shared CUDA QuEST environment");
            SharedEnvHandle(env_handle)
        }
    });

    if env.0.is_null() {
        return Err(CudaLoadError::CudaUnavailable(
            "Failed to create shared CUDA environment".to_string(),
        ));
    }

    Ok((env.0, backend))
}

/// Library name varies by platform
#[cfg(target_os = "linux")]
const CUDA_LIB_NAME: &str = "libpecos_quest_cuda.so";
#[cfg(target_os = "macos")]
const CUDA_LIB_NAME: &str = "libpecos_quest_cuda.dylib";
#[cfg(target_os = "windows")]
const CUDA_LIB_NAME: &str = "pecos_quest_cuda.dll";

/// Attempt to load the CUDA backend library.
///
/// This function is thread-safe and will only attempt to load the library once.
/// Subsequent calls return the cached result.
///
/// # Returns
/// - `Ok(&CudaBackend)` if the CUDA library was loaded successfully
/// - `Err(&CudaLoadError)` if loading failed (CUDA not available, library not found, etc.)
///
/// # Errors
/// Returns a `CudaLoadError` if:
/// - The CUDA library cannot be found in any of the search paths (`LibraryNotFound`)
/// - The library exists but cannot be loaded (`LoadFailed`)
/// - Required symbols are missing from the library (`MissingSymbol`)
pub fn try_load_cuda() -> Result<&'static CudaBackend, &'static CudaLoadError> {
    CUDA_BACKEND.get_or_init(load_cuda_library).as_ref()
}

/// Check if CUDA acceleration is available without fully initializing it
#[must_use]
pub fn is_cuda_available() -> bool {
    try_load_cuda().is_ok()
}

/// Get the search paths for the CUDA library
fn get_cuda_library_search_paths() -> Vec<PathBuf> {
    let mut paths = vec![];

    // 1. Environment variable set by Python package (highest priority)
    if let Ok(pkg_path) = std::env::var("PECOS_QUEST_CUDA_LIB") {
        paths.push(PathBuf::from(pkg_path));
    }

    // 2. Same directory as the current executable
    if let Ok(exe_path) = std::env::current_exe()
        && let Some(dir) = exe_path.parent()
    {
        paths.push(dir.join(CUDA_LIB_NAME));
    }

    // 3. PECOS home directory (~/.pecos/lib/)
    if let Some(home) = dirs::home_dir() {
        paths.push(home.join(".pecos").join("lib").join(CUDA_LIB_NAME));
    }

    // 4. Cargo target directory (for development)
    // Check both debug and release directories relative to current dir
    let cargo_target_paths = [
        PathBuf::from("target/release").join(CUDA_LIB_NAME),
        PathBuf::from("target/debug").join(CUDA_LIB_NAME),
    ];
    paths.extend(cargo_target_paths);

    // 5. System library path (let the dynamic linker search)
    paths.push(PathBuf::from(CUDA_LIB_NAME));

    paths
}

/// Load the CUDA library from one of the search paths
fn load_cuda_library() -> Result<CudaBackend, CudaLoadError> {
    let search_paths = get_cuda_library_search_paths();

    for path in &search_paths {
        log::debug!("Trying to load CUDA library from: {}", path.display());

        match unsafe { Library::new(path) } {
            Ok(lib) => {
                log::info!("Loaded CUDA library from: {}", path.display());
                return load_symbols(lib);
            }
            Err(e) => {
                log::debug!("Failed to load from {}: {e}", path.display());
            }
        }
    }

    let searched = search_paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");

    Err(CudaLoadError::LibraryNotFound {
        searched_paths: searched,
    })
}

/// Helper macro to load a symbol from the library
macro_rules! load_symbol {
    ($lib:expr, $name:expr, $type:ty) => {{
        let symbol: Symbol<$type> = $lib
            .get(concat!("pecos_quest_cuda_", $name, "\0").as_bytes())
            .map_err(|e| CudaLoadError::MissingSymbol(format!("{}: {e}", $name)))?;
        *symbol
    }};
}

/// Load all required symbols from the CUDA library
#[allow(clippy::too_many_lines)]
fn load_symbols(lib: Library) -> Result<CudaBackend, CudaLoadError> {
    // Load all symbols and extract function pointers
    // We use a macro to reduce boilerplate
    let backend = unsafe {
        CudaBackend {
            // Environment management
            create_env: load_symbol!(lib, "create_env", unsafe extern "C" fn() -> *mut u8),
            destroy_env: load_symbol!(lib, "destroy_env", unsafe extern "C" fn(*mut u8)),
            get_env_info: load_symbol!(
                lib,
                "get_env_info",
                unsafe extern "C" fn(*mut u8) -> CudaEnvInfo
            ),

            // Qureg management
            create_qureg: load_symbol!(
                lib,
                "create_qureg",
                unsafe extern "C" fn(*mut u8, i32) -> *mut u8
            ),
            create_density_qureg: load_symbol!(
                lib,
                "create_density_qureg",
                unsafe extern "C" fn(*mut u8, i32) -> *mut u8
            ),
            destroy_qureg: load_symbol!(lib, "destroy_qureg", unsafe extern "C" fn(*mut u8)),

            // State initialization
            init_zero_state: load_symbol!(lib, "init_zero_state", unsafe extern "C" fn(*mut u8)),
            init_plus_state: load_symbol!(lib, "init_plus_state", unsafe extern "C" fn(*mut u8)),
            init_classical_state: load_symbol!(
                lib,
                "init_classical_state",
                unsafe extern "C" fn(*mut u8, i64)
            ),

            // Single-qubit gates
            apply_pauli_x: load_symbol!(lib, "apply_pauli_x", unsafe extern "C" fn(*mut u8, i32)),
            apply_pauli_y: load_symbol!(lib, "apply_pauli_y", unsafe extern "C" fn(*mut u8, i32)),
            apply_pauli_z: load_symbol!(lib, "apply_pauli_z", unsafe extern "C" fn(*mut u8, i32)),
            apply_hadamard: load_symbol!(lib, "apply_hadamard", unsafe extern "C" fn(*mut u8, i32)),
            apply_s_gate: load_symbol!(lib, "apply_s_gate", unsafe extern "C" fn(*mut u8, i32)),
            apply_t_gate: load_symbol!(lib, "apply_t_gate", unsafe extern "C" fn(*mut u8, i32)),
            apply_phase_shift: load_symbol!(
                lib,
                "apply_phase_shift",
                unsafe extern "C" fn(*mut u8, i32, f64)
            ),

            // Rotation gates
            apply_rotation_x: load_symbol!(
                lib,
                "apply_rotation_x",
                unsafe extern "C" fn(*mut u8, i32, f64)
            ),
            apply_rotation_y: load_symbol!(
                lib,
                "apply_rotation_y",
                unsafe extern "C" fn(*mut u8, i32, f64)
            ),
            apply_rotation_z: load_symbol!(
                lib,
                "apply_rotation_z",
                unsafe extern "C" fn(*mut u8, i32, f64)
            ),

            // Two-qubit gates
            apply_cnot: load_symbol!(lib, "apply_cnot", unsafe extern "C" fn(*mut u8, i32, i32)),
            apply_cz: load_symbol!(lib, "apply_cz", unsafe extern "C" fn(*mut u8, i32, i32)),
            apply_swap: load_symbol!(lib, "apply_swap", unsafe extern "C" fn(*mut u8, i32, i32)),
            apply_controlled_phase_shift: load_symbol!(
                lib,
                "apply_controlled_phase_shift",
                unsafe extern "C" fn(*mut u8, i32, i32, f64)
            ),

            // Measurement
            measure: load_symbol!(lib, "measure", unsafe extern "C" fn(*mut u8, i32) -> i32),
            calc_prob_of_outcome: load_symbol!(
                lib,
                "calc_prob_of_outcome",
                unsafe extern "C" fn(*mut u8, i32, i32) -> f64
            ),
            apply_forced_measurement: load_symbol!(
                lib,
                "apply_forced_measurement",
                unsafe extern "C" fn(*mut u8, i32, i32) -> f64
            ),

            // Amplitude access
            get_real_amp: load_symbol!(
                lib,
                "get_real_amp",
                unsafe extern "C" fn(*mut u8, i64) -> f64
            ),
            get_imag_amp: load_symbol!(
                lib,
                "get_imag_amp",
                unsafe extern "C" fn(*mut u8, i64) -> f64
            ),
            get_prob_amp: load_symbol!(
                lib,
                "get_prob_amp",
                unsafe extern "C" fn(*mut u8, i64) -> f64
            ),
            calc_total_prob: load_symbol!(
                lib,
                "calc_total_prob",
                unsafe extern "C" fn(*mut u8) -> f64
            ),
            calc_purity: load_symbol!(lib, "calc_purity", unsafe extern "C" fn(*mut u8) -> f64),

            // Info
            get_num_amps: load_symbol!(lib, "get_num_amps", unsafe extern "C" fn(*mut u8) -> i64),
            get_num_qubits: load_symbol!(
                lib,
                "get_num_qubits",
                unsafe extern "C" fn(*mut u8) -> i32
            ),
            is_density_matrix: load_symbol!(
                lib,
                "is_density_matrix",
                unsafe extern "C" fn(*mut u8) -> bool
            ),

            // Keep library loaded
            _library: lib,
        }
    };

    Ok(backend)
}

/// Get a detailed error message for when CUDA acceleration is requested but unavailable
#[must_use]
pub fn cuda_unavailable_error_message() -> String {
    let search_paths = get_cuda_library_search_paths();
    let paths_str = search_paths
        .iter()
        .map(|p| format!("  - {}", p.display()))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r"CUDA acceleration requested but not available.

Possible causes:
  - NVIDIA CUDA runtime (libcudart.so, libcublas.so) is not installed
  - No NVIDIA GPU driver is installed
  - The QuEST CUDA backend ({CUDA_LIB_NAME}) was not found

Searched locations:
{paths_str}

Solutions:
  - Install NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
  - Verify GPU availability: nvidia-smi
  - Set PECOS_QUEST_CUDA_LIB environment variable to the backend library path
  - Use CPU mode by setting use_cuda=False"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_paths_not_empty() {
        let paths = get_cuda_library_search_paths();
        assert!(!paths.is_empty(), "Should have at least one search path");
    }

    #[test]
    fn test_cuda_load_returns_result() {
        // This test just verifies the function doesn't panic
        // On systems without CUDA, it should return an error
        let result = try_load_cuda();
        // Either success or error is fine, we just verify it works
        match result {
            Ok(_) => println!("CUDA library loaded successfully"),
            Err(e) => println!("CUDA library not available: {e}"),
        }
    }

    #[test]
    fn test_error_message_is_helpful() {
        let msg = cuda_unavailable_error_message();
        assert!(msg.contains("CUDA acceleration requested"));
        assert!(msg.contains("CUDA"));
        assert!(msg.contains("nvidia-smi"));
    }
}
