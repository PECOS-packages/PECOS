//! Trait for QIS program execution interfaces
//!
//! This module defines the `QisInterface` trait that different implementations
//! (JIT, Helios, etc.) must implement to execute quantum programs and collect operations.

use pecos_qis_ffi_types::OperationCollector;
use std::collections::BTreeMap;

/// Program format for loading
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramFormat {
    /// LLVM IR text
    LlvmIrText,
    /// LLVM bitcode
    LlvmBitcode,
    /// HUGR bytes
    HugrBytes,
    /// QIS bitcode (Selene format)
    QisBitcode,
}

/// Error type for interface operations
///
/// This is kept minimal to avoid circular dependencies with pecos-core.
/// Implementations can convert to `PecosError` as needed.
#[derive(Debug, Clone)]
pub enum InterfaceError {
    /// Program loading error
    LoadError(String),
    /// Execution error
    ExecutionError(String),
    /// Invalid program format
    InvalidFormat(String),
    /// Other error
    Other(String),
}

impl std::fmt::Display for InterfaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoadError(msg) => write!(f, "Load error: {msg}"),
            Self::ExecutionError(msg) => write!(f, "Execution error: {msg}"),
            Self::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for InterfaceError {}

/// Trait for QIS interface implementations
///
/// A `QisInterface` implementation is responsible for executing a quantum program and
/// collecting the quantum operations that need to be performed.
///
/// The primary implementation is:
/// - `QisHeliosInterface` - Links with Selene's Helios compiler
///
/// All implementations must support dynamic execution mode for proper handling of
/// measurement-dependent conditionals.
pub trait QisInterface: Send + Sync {
    /// Load a program into the interface
    ///
    /// The format depends on the implementation:
    /// - JIT: LLVM IR text or bitcode
    /// - Helios: QIS bitcode or HUGR bytes
    ///
    /// # Errors
    /// Returns an error if the program cannot be loaded or parsed.
    fn load_program(
        &mut self,
        program_bytes: &[u8],
        format: ProgramFormat,
    ) -> Result<(), InterfaceError>;

    /// Execute the program to collect operations
    ///
    /// This runs the program in "collection mode" to discover all quantum
    /// operations without actually performing quantum simulation.
    ///
    /// # Errors
    /// Returns an error if the program execution fails.
    fn collect_operations(&mut self) -> Result<OperationCollector, InterfaceError>;

    /// Execute with measurement results
    ///
    /// This runs the program with specific measurement results to handle
    /// conditional execution paths correctly.
    ///
    /// # Errors
    /// Returns an error if the program execution fails.
    fn execute_with_measurements(
        &mut self,
        measurements: BTreeMap<usize, bool>,
    ) -> Result<OperationCollector, InterfaceError>;

    /// Get metadata about the implementation
    fn metadata(&self) -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    /// Get the name of this implementation
    fn name(&self) -> &'static str;

    /// Reset the interface for a new execution
    ///
    /// # Errors
    /// Returns an error if the reset operation fails.
    fn reset(&mut self) -> Result<(), InterfaceError>;

    // ========================================================================
    // Dynamic execution methods (for circuits with mid-circuit measurement)
    // ========================================================================

    /// Check if this interface supports dynamic execution
    ///
    /// Dynamic execution allows conditionals that depend on measurement results
    /// to work correctly by blocking at measurement points and coordinating
    /// with the main thread.
    fn supports_dynamic(&self) -> bool {
        false
    }

    /// Enable dynamic execution mode
    ///
    /// This should be called before starting dynamic execution. It enables
    /// the synchronization primitives used for coordination.
    ///
    /// # Errors
    /// Returns an error if dynamic execution is not supported by this interface.
    fn enable_dynamic_mode(&mut self) -> Result<(), InterfaceError> {
        Err(InterfaceError::Other(
            "Dynamic execution not supported by this interface".to_string(),
        ))
    }

    /// Disable dynamic execution mode
    ///
    /// # Errors
    /// Returns an error if dynamic execution is not supported by this interface.
    fn disable_dynamic_mode(&mut self) -> Result<(), InterfaceError> {
        Ok(())
    }

    /// Wait for the running program to need a measurement result
    ///
    /// This blocks until the program calls `___read_future_bool` and needs
    /// a result that isn't available. Returns the result ID that is needed,
    /// or None on timeout.
    fn wait_for_result_needed(&self, _timeout_ms: u64) -> Option<u64> {
        None
    }

    /// Set a measurement result for the running program
    ///
    /// This provides the result that the program is waiting for in `___read_future_bool`.
    ///
    /// # Errors
    /// Returns an error if dynamic execution is not supported by this interface.
    fn set_measurement_result(
        &mut self,
        _result_id: u64,
        _value: bool,
    ) -> Result<(), InterfaceError> {
        Err(InterfaceError::Other(
            "Dynamic execution not supported by this interface".to_string(),
        ))
    }

    /// Signal that the measurement result is ready
    ///
    /// This wakes up the blocked program to continue execution.
    ///
    /// # Errors
    /// Returns an error if dynamic execution is not supported by this interface.
    fn signal_result_ready(&mut self) -> Result<(), InterfaceError> {
        Err(InterfaceError::Other(
            "Dynamic execution not supported by this interface".to_string(),
        ))
    }

    /// Get the pending operations collected so far
    ///
    /// This returns the operations that have been collected since the last
    /// call, without waiting for the program to complete.
    ///
    /// # Errors
    /// Returns an error if dynamic execution is not supported by this interface.
    fn get_pending_operations(
        &self,
    ) -> Result<Vec<pecos_qis_ffi_types::Operation>, InterfaceError> {
        Err(InterfaceError::Other(
            "Dynamic execution not supported by this interface".to_string(),
        ))
    }

    /// Get the path to the QIS FFI library for dynamic execution
    ///
    /// This is used by the engine to load the library separately for main thread FFI calls.
    fn get_qis_ffi_lib_path(&self) -> Option<std::path::PathBuf> {
        None
    }

    /// Get the execution context pointer for dynamic execution
    ///
    /// This returns a raw pointer to the execution context, which can be used
    /// to register the context on other library handles for cross-thread communication.
    /// The pointer is opaque - it should only be passed to FFI registration functions.
    ///
    /// Returns None if dynamic execution is not supported or not enabled.
    fn get_execution_context_ptr(&self) -> Option<*mut std::ffi::c_void> {
        None
    }

    /// Get a synchronization handle for the main thread
    ///
    /// This returns a handle that can be used by the main thread to call FFI functions
    /// for synchronization while the interface is running on a worker thread.
    ///
    /// The handle uses the same library instance (singleton) as the worker thread,
    /// ensuring TLS is consistent across threads (important on macOS).
    ///
    /// Returns None if dynamic execution is not supported.
    fn get_sync_handle(&self) -> Option<Box<dyn DynamicSyncHandle>> {
        None
    }
}

/// Handle for main thread synchronization with a dynamic worker thread
///
/// This trait provides methods for the main thread to coordinate with a worker
/// thread running an LLVM program. All methods access the FFI library through
/// the same singleton instance used by the worker thread.
#[allow(clippy::module_name_repetitions)]
pub trait DynamicSyncHandle: Send + Sync {
    /// Wait for the worker to need a measurement result
    ///
    /// Returns `Some(result_id)` if worker needs a result, None on timeout or completion.
    fn wait_for_need_result(&self, timeout_ms: u64) -> Option<u64>;

    /// Set a measurement result for the running program
    ///
    /// # Errors
    /// Returns an error if the FFI call fails or no execution context is registered.
    fn set_measurement_result(&self, result_id: u64, value: bool) -> Result<(), InterfaceError>;

    /// Signal that the measurement result is ready
    ///
    /// # Errors
    /// Returns an error if the FFI call fails or no execution context is registered.
    fn signal_result_ready(&self) -> Result<(), InterfaceError>;

    /// Get the pending operations collected so far
    ///
    /// # Errors
    /// Returns an error if the FFI call fails or no execution context is registered.
    fn get_pending_operations(&self)
    -> Result<Vec<pecos_qis_ffi_types::Operation>, InterfaceError>;

    /// Abort the dynamic execution
    ///
    /// # Errors
    /// Returns an error if the FFI call fails.
    fn abort_execution(&self) -> Result<(), InterfaceError>;

    /// Get named results from the execution context
    ///
    /// Returns a map of result names to their boolean values.
    /// Named results are stored by `print_bool` and `print_bool_arr` FFI calls.
    ///
    /// # Errors
    /// Returns an error if the FFI call fails or JSON parsing fails.
    fn get_named_results(
        &self,
    ) -> Result<std::collections::BTreeMap<String, Vec<bool>>, InterfaceError>;
}

/// Box type for interface implementations
pub type BoxedInterface = Box<dyn QisInterface>;
