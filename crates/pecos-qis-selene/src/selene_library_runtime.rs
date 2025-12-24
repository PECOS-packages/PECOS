//! Flexible Selene Runtime Wrappers
//!
//! This module provides flexible wrappers for Selene runtime shared libraries,
//! supporting both auto-built runtimes and user-provided .so file paths.

use libloading::{Library, Symbol};
use log::{debug, info};
use pecos_qis_core::runtime::{ClassicalState, QisRuntime, Result, RuntimeError, Shot};
use pecos_qis_ffi_types::OperationCollector;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Configuration for Selene runtime libraries
#[derive(Debug, Clone)]
pub struct SeleneRuntimeConfig {
    /// Path to the runtime .so file (if None, auto-build is attempted)
    pub library_path: Option<PathBuf>,
    /// Runtime type identifier (e.g., "simple", "`soft_rz`", "custom")
    pub runtime_type: String,
    /// Additional runtime-specific configuration
    pub runtime_options: BTreeMap<String, String>,
    /// Whether to auto-build if `library_path` is not provided
    pub auto_build: bool,
    /// Base directory for auto-built runtimes
    pub build_dir: Option<PathBuf>,
}

impl Default for SeleneRuntimeConfig {
    fn default() -> Self {
        Self {
            library_path: None,
            runtime_type: "simple".to_string(),
            runtime_options: BTreeMap::new(),
            auto_build: true,
            build_dir: None,
        }
    }
}

/// Thread-safe wrapper for Selene runtime handle
#[derive(Debug)]
struct SeleneRuntimeHandle {
    handle: *mut std::ffi::c_void,
}

// Safety: The Selene runtime is designed to be thread-safe
unsafe impl Send for SeleneRuntimeHandle {}
unsafe impl Sync for SeleneRuntimeHandle {}

impl Clone for SeleneRuntimeHandle {
    fn clone(&self) -> Self {
        // Note: This creates a copy of the pointer, not the underlying runtime
        // For true cloning, we'd need to call the runtime's clone function
        Self {
            handle: self.handle,
        }
    }
}

/// Generic wrapper for any Selene-compatible runtime shared library
///
/// This provides a unified interface to interact with Selene runtime .so files,
/// automatically handling FFI calls and lifecycle management.
#[derive(Debug, Clone)]
pub struct QisSeleneLibraryRuntime {
    /// Loaded shared library handle
    library: Arc<Library>,
    /// Runtime configuration
    config: SeleneRuntimeConfig,
    /// Classical state maintained by this runtime
    state: ClassicalState,
    /// Loaded QIS interface (program)
    interface: Option<OperationCollector>,
    /// Runtime handle from the .so file
    runtime_handle: Option<SeleneRuntimeHandle>,
}

/// Specific implementation for Selene Simple Runtime
///
/// This is a convenience wrapper that automatically builds or locates
/// the `selene_simple_runtime.so` file.
#[derive(Debug, Clone)]
pub struct QisSeleneSimpleRuntime {
    inner: QisSeleneLibraryRuntime,
}

/// FFI function signatures for Selene runtime interface
type CreateRuntimeFn = unsafe extern "C" fn() -> *mut std::ffi::c_void;
type DestroyRuntimeFn = unsafe extern "C" fn(*mut std::ffi::c_void);
type LoadInterfaceFn = unsafe extern "C" fn(*mut std::ffi::c_void, *const u8, usize) -> i32;
type ExecuteUntilQuantumFn = unsafe extern "C" fn(*mut std::ffi::c_void) -> i32;
type ProvideResultsFn = unsafe extern "C" fn(*mut std::ffi::c_void, *const u8, usize) -> i32;
type ShotStartFn = unsafe extern "C" fn(*mut std::ffi::c_void, u64, u64) -> i32;
type ShotEndFn = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u8, *mut usize) -> i32;

impl QisSeleneLibraryRuntime {
    /// Create a new Selene library runtime with configuration
    ///
    /// # Errors
    /// Returns an error if the library cannot be found or loaded, or if auto-build fails.
    pub fn new(config: SeleneRuntimeConfig) -> Result<Self> {
        let library_path = match &config.library_path {
            Some(path) => path.clone(),
            None if config.auto_build => {
                // Attempt to auto-build the runtime
                Self::auto_build_runtime(&config)?
            }
            None => {
                return Err(RuntimeError::FfiError(
                    "No library path provided and auto_build is disabled".to_string(),
                ));
            }
        };

        info!("Loading Selene runtime library: {}", library_path.display());

        // Load the shared library
        let library = unsafe {
            Library::new(&library_path)
                .map_err(|e| RuntimeError::FfiError(format!("Failed to load library: {e}")))?
        };

        // Verify required symbols exist
        Self::verify_library_symbols(&library)?;

        let runtime = Self {
            library: Arc::new(library),
            config,
            state: ClassicalState::default(),
            interface: None,
            runtime_handle: None,
        };

        info!("Selene runtime library loaded successfully");
        Ok(runtime)
    }

    /// Create a new runtime from a specific .so file path
    ///
    /// # Errors
    /// Returns an error if the library at the specified path cannot be loaded.
    pub fn from_library_path<P: AsRef<Path>>(path: P, runtime_type: &str) -> Result<Self> {
        let config = SeleneRuntimeConfig {
            library_path: Some(path.as_ref().to_path_buf()),
            runtime_type: runtime_type.to_string(),
            auto_build: false,
            ..Default::default()
        };

        Self::new(config)
    }

    /// Auto-build a Selene runtime library
    fn auto_build_runtime(config: &SeleneRuntimeConfig) -> Result<PathBuf> {
        info!("Auto-building Selene runtime: {}", config.runtime_type);

        let build_dir = config
            .build_dir
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("pecos_selene_runtimes"));

        // Create build directory
        std::fs::create_dir_all(&build_dir)
            .map_err(|e| RuntimeError::FfiError(format!("Failed to create build dir: {e}")))?;

        let lib_ext = if cfg!(target_os = "macos") {
            "dylib"
        } else if cfg!(target_os = "windows") {
            "dll"
        } else {
            "so"
        };
        let so_path = build_dir.join(format!(
            "selene_{}_runtime.{}",
            config.runtime_type, lib_ext
        ));

        // Check if already built
        if so_path.exists() {
            info!("Using existing auto-built runtime: {}", so_path.display());
            return Ok(so_path);
        }

        // Attempt to build using Selene's build system
        let selene_path = std::env::var("SELENE_PATH")
            .unwrap_or_else(|_| "/home/ciaranra/Repos/cl_projects/gup/selene".to_string());

        let build_script = format!(
            r#"
cd "{selene_path}/selene-runtimes/{runtime_type}"
make clean && make
cp selene_{runtime_type}_runtime.{lib_ext} "{so_path}"
"#,
            selene_path = selene_path,
            runtime_type = config.runtime_type,
            lib_ext = lib_ext,
            so_path = so_path.display()
        );

        let output = std::process::Command::new("bash")
            .arg("-c")
            .arg(&build_script)
            .output()
            .map_err(|e| RuntimeError::FfiError(format!("Build command failed: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(RuntimeError::FfiError(format!("Build failed: {stderr}")));
        }

        if !so_path.exists() {
            return Err(RuntimeError::FfiError(
                "Build completed but .so file not found".to_string(),
            ));
        }

        info!("Auto-built Selene runtime: {}", so_path.display());
        Ok(so_path)
    }

    /// Verify that the loaded library has required Selene runtime symbols
    fn verify_library_symbols(library: &Library) -> Result<()> {
        let required_symbols = [
            "selene_runtime_create",
            "selene_runtime_destroy",
            "selene_runtime_load_interface",
            "selene_runtime_execute_until_quantum",
            "selene_runtime_provide_results",
            "selene_runtime_shot_start",
            "selene_runtime_shot_end",
        ];

        for symbol_name in &required_symbols {
            unsafe {
                library
                    .get::<*mut std::ffi::c_void>(symbol_name.as_bytes())
                    .map_err(|_| {
                        RuntimeError::FfiError(format!(
                            "Required symbol '{symbol_name}' not found in library"
                        ))
                    })?;
            }
        }

        debug!("All required symbols found in Selene runtime library");
        Ok(())
    }

    /// Initialize the runtime handle
    fn initialize_runtime(&mut self) -> Result<()> {
        if self.runtime_handle.is_some() {
            return Ok(()); // Already initialized
        }

        let create_fn: Symbol<CreateRuntimeFn> = unsafe {
            self.library.get(b"selene_runtime_create").map_err(|e| {
                RuntimeError::FfiError(format!("Failed to get create function: {e}"))
            })?
        };

        let handle = unsafe { create_fn() };
        if handle.is_null() {
            return Err(RuntimeError::FfiError(
                "Failed to create runtime handle".to_string(),
            ));
        }

        self.runtime_handle = Some(SeleneRuntimeHandle { handle });
        debug!("Selene runtime handle initialized");
        Ok(())
    }

    /// Get the runtime type identifier
    #[must_use]
    pub fn runtime_type(&self) -> &str {
        &self.config.runtime_type
    }

    /// Get the library path
    #[must_use]
    pub fn library_path(&self) -> Option<&Path> {
        self.config.library_path.as_deref()
    }
}

impl QisRuntime for QisSeleneLibraryRuntime {
    fn load_interface(&mut self, interface: OperationCollector) -> Result<()> {
        self.initialize_runtime()?;

        let handle = self
            .runtime_handle
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Runtime not initialized".to_string()))?
            .handle;

        // Serialize interface using bincode for efficient FFI transfer
        let interface_bytes = bincode::encode_to_vec(&interface, bincode::config::standard())
            .map_err(|e| RuntimeError::FfiError(format!("Failed to serialize interface: {e}")))?;

        let load_fn: Symbol<LoadInterfaceFn> = unsafe {
            self.library
                .get(b"selene_runtime_load_interface")
                .map_err(|e| RuntimeError::FfiError(format!("Failed to get load function: {e}")))?
        };

        let result = unsafe { load_fn(handle, interface_bytes.as_ptr(), interface_bytes.len()) };

        if result != 0 {
            return Err(RuntimeError::FfiError(format!(
                "Load interface failed with code: {result}"
            )));
        }

        self.interface = Some(interface);
        info!("Interface loaded into Selene runtime");
        Ok(())
    }

    fn execute_until_quantum(&mut self) -> Result<Option<Vec<pecos_qis_ffi_types::QuantumOp>>> {
        let handle = self
            .runtime_handle
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Runtime not initialized".to_string()))?
            .handle;

        let execute_fn: Symbol<ExecuteUntilQuantumFn> = unsafe {
            self.library
                .get(b"selene_runtime_execute_until_quantum")
                .map_err(|e| {
                    RuntimeError::FfiError(format!("Failed to get execute function: {e}"))
                })?
        };

        let result = unsafe { execute_fn(handle) };

        match result {
            0 => Ok(None), // Program complete
            1 => {
                // TODO: Get quantum operations from runtime
                // This would require additional FFI to retrieve the operations
                Ok(Some(Vec::new()))
            }
            _ => Err(RuntimeError::ExecutionError(format!(
                "Execute failed with code: {result}"
            ))),
        }
    }

    fn provide_measurements(&mut self, measurements: BTreeMap<usize, bool>) -> Result<()> {
        let handle = self
            .runtime_handle
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Runtime not initialized".to_string()))?
            .handle;

        // Serialize measurements using bincode for efficient FFI transfer
        let measurements_bytes = bincode::encode_to_vec(&measurements, bincode::config::standard())
            .map_err(|e| {
                RuntimeError::FfiError(format!("Failed to serialize measurements: {e}"))
            })?;

        let provide_fn: Symbol<ProvideResultsFn> = unsafe {
            self.library
                .get(b"selene_runtime_provide_results")
                .map_err(|e| {
                    RuntimeError::FfiError(format!("Failed to get provide function: {e}"))
                })?
        };

        let result = unsafe {
            provide_fn(
                handle,
                measurements_bytes.as_ptr(),
                measurements_bytes.len(),
            )
        };

        if result != 0 {
            return Err(RuntimeError::FfiError(format!(
                "Provide measurements failed with code: {result}"
            )));
        }

        // Update local state
        self.state.measurements.extend(measurements);
        Ok(())
    }

    fn get_classical_state(&self) -> &ClassicalState {
        &self.state
    }

    fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
        &mut self.state
    }

    fn shot_start(&mut self, shot_id: u64, seed: Option<u64>) -> Result<()> {
        let handle = self
            .runtime_handle
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Runtime not initialized".to_string()))?
            .handle;

        let shot_start_fn: Symbol<ShotStartFn> = unsafe {
            self.library
                .get(b"selene_runtime_shot_start")
                .map_err(|e| {
                    RuntimeError::FfiError(format!("Failed to get shot_start function: {e}"))
                })?
        };

        let result = unsafe { shot_start_fn(handle, shot_id, seed.unwrap_or(0)) };

        if result != 0 {
            return Err(RuntimeError::FfiError(format!(
                "Shot start failed with code: {result}"
            )));
        }

        // Update local state
        self.state.shot_id = Some(shot_id);
        self.state.pc = 0;
        self.state.call_stack.clear();
        self.state.measurements.clear();
        self.state.variables.clear();

        Ok(())
    }

    fn shot_end(&mut self) -> Result<Shot> {
        let handle = self
            .runtime_handle
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Runtime not initialized".to_string()))?
            .handle;

        let shot_end_fn: Symbol<ShotEndFn> = unsafe {
            self.library.get(b"selene_runtime_shot_end").map_err(|e| {
                RuntimeError::FfiError(format!("Failed to get shot_end function: {e}"))
            })?
        };

        // TODO: Implement proper result retrieval from runtime
        let mut buffer = vec![0u8; 1024];
        let mut size = buffer.len();

        let result = unsafe { shot_end_fn(handle, buffer.as_mut_ptr(), &raw mut size) };

        if result != 0 {
            return Err(RuntimeError::FfiError(format!(
                "Shot end failed with code: {result}"
            )));
        }

        Ok(Shot {
            measurements: self.state.measurements.clone(),
            registers: self.state.registers.clone(),
            metadata: std::collections::BTreeMap::new(),
        })
    }

    fn is_complete(&self) -> bool {
        // TODO: Query runtime for completion status
        false
    }

    fn num_qubits(&self) -> usize {
        self.interface
            .as_ref()
            .map_or(0, |i| i.allocated_qubits.len())
    }
}

impl Drop for QisSeleneLibraryRuntime {
    fn drop(&mut self) {
        if let Some(runtime_handle) = self.runtime_handle.take()
            && let Ok(destroy_fn) = unsafe {
                self.library
                    .get::<Symbol<DestroyRuntimeFn>>(b"selene_runtime_destroy")
            }
        {
            unsafe { destroy_fn(runtime_handle.handle) };
            debug!("Selene runtime handle destroyed");
        }
    }
}

// Convenience implementation for Selene Simple Runtime
impl QisSeleneSimpleRuntime {
    /// Create a new Selene Simple Runtime with auto-build
    ///
    /// # Errors
    /// Returns an error if the Selene simple runtime library cannot be built or loaded.
    pub fn new() -> Result<Self> {
        let config = SeleneRuntimeConfig {
            runtime_type: "simple".to_string(),
            auto_build: true,
            ..Default::default()
        };

        let inner = QisSeleneLibraryRuntime::new(config)?;
        Ok(Self { inner })
    }

    /// Create a new Selene Simple Runtime from a specific .so path
    ///
    /// # Errors
    /// Returns an error if the library at the specified path cannot be loaded.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = QisSeleneLibraryRuntime::from_library_path(path, "simple")?;
        Ok(Self { inner })
    }
}

impl QisRuntime for QisSeleneSimpleRuntime {
    fn load_interface(&mut self, interface: OperationCollector) -> Result<()> {
        self.inner.load_interface(interface)
    }

    fn execute_until_quantum(&mut self) -> Result<Option<Vec<pecos_qis_ffi_types::QuantumOp>>> {
        self.inner.execute_until_quantum()
    }

    fn provide_measurements(&mut self, measurements: BTreeMap<usize, bool>) -> Result<()> {
        self.inner.provide_measurements(measurements)
    }

    fn get_classical_state(&self) -> &ClassicalState {
        self.inner.get_classical_state()
    }

    fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
        self.inner.get_classical_state_mut()
    }

    fn shot_start(&mut self, shot_id: u64, seed: Option<u64>) -> Result<()> {
        self.inner.shot_start(shot_id, seed)
    }

    fn shot_end(&mut self) -> Result<Shot> {
        self.inner.shot_end()
    }

    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }
}

/// Convenience function to create a Selene Simple Runtime with auto-build
///
/// # Errors
/// Returns an error if the Selene simple runtime library cannot be built or loaded.
pub fn selene_simple_runtime() -> Result<QisSeleneSimpleRuntime> {
    QisSeleneSimpleRuntime::new()
}

/// Convenience function to create a Selene Simple Runtime from a path
///
/// # Errors
/// Returns an error if the library at the specified path cannot be loaded.
pub fn selene_simple_runtime_from_path<P: AsRef<Path>>(path: P) -> Result<QisSeleneSimpleRuntime> {
    QisSeleneSimpleRuntime::from_path(path)
}

/// Create a generic Selene runtime wrapper for any compatible .so file
///
/// # Errors
/// Returns an error if the library at the specified path cannot be loaded.
pub fn selene_library_runtime<P: AsRef<Path>>(
    path: P,
    runtime_type: &str,
) -> Result<QisSeleneLibraryRuntime> {
    QisSeleneLibraryRuntime::from_library_path(path, runtime_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selene_runtime_config() {
        let config = SeleneRuntimeConfig::default();
        assert_eq!(config.runtime_type, "simple");
        assert!(config.auto_build);
        assert!(config.library_path.is_none());
    }

    #[test]
    fn test_runtime_creation_without_library() {
        // This should attempt auto-build (may fail if Selene not available)
        match QisSeleneSimpleRuntime::new() {
            Ok(_runtime) => {
                println!("Selene simple runtime created successfully");
            }
            Err(e) => {
                println!(
                    "WARNING: Selene simple runtime creation failed (expected if Selene not available): {e}"
                );
            }
        }
    }
}
