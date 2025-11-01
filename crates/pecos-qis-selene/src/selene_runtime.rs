//! Selene Runtime implementation of `QisRuntime`
//!
//! This wraps a Selene .so runtime plugin and implements the `QisRuntime` trait
//! to provide a Selene-based classical interpreter for QIS programs.

use log::{debug, trace};
use pecos_qis_core::runtime::{ClassicalState, QisRuntime, Result, RuntimeError, Shot};
use pecos_qis_ffi_types::{Operation, OperationCollector, QuantumOp};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

/// Selene runtime implementation
pub struct SeleneRuntime {
    /// Path to the Selene .so file
    plugin_path: String,

    /// Loaded library (if any)
    #[allow(dead_code)]
    library: Option<Arc<libloading::Library>>,

    /// Runtime instance pointer
    #[allow(dead_code)]
    instance: Option<*mut c_void>,

    /// Current classical state
    state: ClassicalState,

    /// Operations buffer for batching
    operations_buffer: Vec<QuantumOp>,

    /// Maximum batch size for operations
    batch_size: usize,

    /// Number of qubits
    num_qubits: usize,

    /// Number of allocated result slots
    num_results: usize,

    /// Loaded QIS interface
    interface: Option<OperationCollector>,

    /// Current operation index
    current_op_index: usize,
}

// Safety: The Selene runtime is designed to be thread-safe
unsafe impl Send for SeleneRuntime {}
unsafe impl Sync for SeleneRuntime {}

impl SeleneRuntime {
    /// Create a new Selene runtime with the given plugin path
    pub fn new(plugin_path: impl AsRef<Path>) -> Self {
        Self {
            plugin_path: plugin_path.as_ref().to_string_lossy().to_string(),
            library: None,
            instance: None,
            state: ClassicalState::default(),
            operations_buffer: Vec::new(),
            batch_size: 100,
            num_qubits: 0,
            num_results: 0,
            interface: None,
            current_op_index: 0,
        }
    }

    /// Load the Selene plugin
    fn load_plugin(&mut self) -> Result<()> {
        if self.library.is_some() {
            return Ok(());
        }

        debug!(
            "Loading Selene plugin from {} with {} qubits and {} results",
            self.plugin_path, self.num_qubits, self.num_results
        );

        unsafe {
            let lib = Arc::new(
                libloading::Library::new(&self.plugin_path)
                    .map_err(|e| RuntimeError::FfiError(format!("Failed to load plugin: {e}")))?,
            );

            // Initialize runtime instance
            let init_fn: libloading::Symbol<
                unsafe extern "C" fn(*mut *mut c_void, u64, u64, u32, *const *const i8) -> i32,
            > = lib
                .get(b"selene_runtime_init")
                .map_err(|e| RuntimeError::FfiError(format!("Missing init function: {e}")))?;

            let mut instance: *mut c_void = std::ptr::null_mut();
            let errno = init_fn(
                &raw mut instance,
                self.num_qubits as u64,
                0,                // start time
                0,                // argc
                std::ptr::null(), // argv
            );

            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "Init failed with errno {errno}"
                )));
            }

            self.library = Some(lib);
            self.instance = Some(instance);
        }

        Ok(())
    }

    /// Process operations from the interface sequentially
    fn process_interface_ops(&mut self) -> Result<Option<Vec<QuantumOp>>> {
        let interface = self
            .interface
            .as_ref()
            .ok_or(RuntimeError::NoProgramLoaded)?;

        self.operations_buffer.clear();

        // For quantum programs, process ALL quantum operations in a single batch
        // to maintain quantum coherence and entanglement
        while self.current_op_index < interface.operations.len() {
            let op = &interface.operations[self.current_op_index];

            match op {
                Operation::Quantum(qop) => {
                    trace!("Processing quantum operation: {qop:?}");
                    self.operations_buffer.push(qop.clone());
                    self.current_op_index += 1;
                }
                Operation::AllocateQubit { id } => {
                    trace!("Allocating qubit {id}");
                    self.num_qubits = self.num_qubits.max(id + 1);
                    self.current_op_index += 1;
                }
                Operation::AllocateResult { id } => {
                    trace!("Allocating result {id}");
                    self.num_results = self.num_results.max(id + 1);
                    self.current_op_index += 1;
                }
                Operation::ReleaseQubit { id } => {
                    trace!("Releasing qubit {id}");
                    let _ = id; // Just track it
                    self.current_op_index += 1;
                }
                Operation::RecordOutput {
                    result_id,
                    register_name,
                } => {
                    trace!(
                        "Recording output: result_id={result_id}, register_name={register_name}"
                    );
                    // Metadata operation - just advance the index
                    // The actual result mapping is handled by the runtime's results collection
                    self.current_op_index += 1;
                }
                Operation::Barrier => {
                    trace!("Barrier encountered");
                    // Barriers don't produce quantum ops but can break batches
                    self.current_op_index += 1;
                    if !self.operations_buffer.is_empty() {
                        // End current batch at barrier
                        break;
                    }
                }
            }
        }

        if self.operations_buffer.is_empty() {
            Ok(None)
        } else {
            trace!(
                "Returning batch of {} quantum operations",
                self.operations_buffer.len()
            );
            Ok(Some(self.operations_buffer.clone()))
        }
    }
}

impl Clone for SeleneRuntime {
    fn clone(&self) -> Self {
        // For now, create a new instance with the same plugin path
        // The library itself can't be cloned, so we'll reload if needed
        Self {
            plugin_path: self.plugin_path.clone(),
            library: None,  // Will be reloaded on demand
            instance: None, // Will be recreated on demand
            state: self.state.clone(),
            operations_buffer: self.operations_buffer.clone(),
            batch_size: self.batch_size,
            num_qubits: self.num_qubits,
            num_results: self.num_results,
            interface: self.interface.clone(),
            current_op_index: self.current_op_index,
        }
    }
}

impl QisRuntime for SeleneRuntime {
    fn load_interface(&mut self, interface: OperationCollector) -> Result<()> {
        debug!(
            "Loading QIS interface with {} operations",
            interface.operations.len()
        );

        // Count qubits and results
        self.num_qubits = interface
            .allocated_qubits
            .iter()
            .max()
            .map_or(0, |&q| q + 1);
        self.num_results = interface
            .allocated_results
            .iter()
            .max()
            .map_or(0, |&r| r + 1);

        debug!(
            "Interface has {} qubits and {} result slots",
            self.num_qubits, self.num_results
        );

        self.interface = Some(interface);
        self.current_op_index = 0;

        // Don't load the plugin yet - defer until actually needed
        // This allows creating and testing the runtime without a real .so file

        Ok(())
    }

    fn execute_until_quantum(&mut self) -> Result<Option<Vec<QuantumOp>>> {
        // For now, we'll use the simple approach of processing from the interface
        // In a full implementation, we'd call into the Selene runtime's
        // get_next_operations function
        self.process_interface_ops()
    }

    fn provide_measurements(&mut self, measurements: BTreeMap<usize, bool>) -> Result<()> {
        debug!(
            "Received {} measurement results, num_results={}, allocated_results={:?}",
            measurements.len(),
            self.num_results,
            self.interface.as_ref().map(|i| &i.allocated_results)
        );

        // Store measurements in classical state
        for (result_id, value) in measurements {
            trace!(
                "Measurement result {} = {} (num_results={})",
                result_id, value, self.num_results
            );
            self.state.measurements.insert(result_id, value);

            // For Selene runtime: Only pass measurements that were explicitly allocated
            // The Selene runtime doesn't support dynamic result allocation, so we must
            // check if this result was known at compile time
            if let Some(interface) = &mut self.interface {
                if interface.allocated_results.contains(&result_id) {
                    // This result was explicitly allocated, try to pass to Selene runtime
                    if let Some(lib) = &self.library
                        && let Some(instance) = self.instance
                    {
                        unsafe {
                            if let Ok(set_result_fn) =
                                lib.get::<unsafe extern "C" fn(*mut c_void, u64, bool) -> i32>(
                                    b"selene_runtime_set_bool_result",
                                )
                            {
                                let errno = set_result_fn(instance, result_id as u64, value);
                                if errno != 0 {
                                    // Unexpected error - log it at trace level since this is normal
                                    // for programs that don't explicitly allocate all result slots
                                    log::trace!(
                                        "Selene runtime returned error {errno} for result {result_id}"
                                    );
                                }
                            }
                        }
                    }
                } else {
                    // Result wasn't explicitly allocated - this is normal for LLVM programs
                    // that use implicit result IDs in measurements
                    log::trace!(
                        "Measurement result {result_id} was not explicitly allocated, storing locally only"
                    );
                }

                // Update the interface with the measurement result
                interface.store_result(result_id, value);
            } else {
                // No interface loaded - just store locally
                log::trace!("No interface loaded, storing measurement {result_id} locally");
            }
        }

        Ok(())
    }

    fn get_classical_state(&self) -> &ClassicalState {
        &self.state
    }

    fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
        &mut self.state
    }

    fn is_complete(&self) -> bool {
        self.interface
            .as_ref()
            .is_none_or(|i| self.current_op_index >= i.operations.len())
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn set_batch_size(&mut self, size: usize) {
        self.batch_size = size;
    }

    fn shot_start(&mut self, shot_id: u64, seed: Option<u64>) -> Result<()> {
        // Try to load the plugin if not already loaded
        if self.library.is_none() && std::path::Path::new(&self.plugin_path).exists() {
            self.load_plugin()?;
        }

        if let Some(lib) = &self.library
            && let Some(instance) = self.instance
        {
            unsafe {
                if let Ok(shot_start_fn) = lib
                    .get::<unsafe extern "C" fn(*mut c_void, u64, u64) -> i32>(
                        b"selene_runtime_shot_start",
                    )
                {
                    let errno = shot_start_fn(instance, shot_id, seed.unwrap_or(0));
                    if errno != 0 {
                        return Err(RuntimeError::ExecutionError(format!(
                            "Shot start failed with errno {errno}"
                        )));
                    }
                }
            }
        }

        // Reset state for new shot
        self.state = ClassicalState::default();
        self.current_op_index = 0;

        Ok(())
    }

    fn shot_end(&mut self) -> Result<Shot> {
        if let Some(lib) = &self.library
            && let Some(instance) = self.instance
        {
            unsafe {
                if let Ok(shot_end_fn) = lib
                    .get::<unsafe extern "C" fn(*mut c_void, u64, u64) -> i32>(
                        b"selene_runtime_shot_end",
                    )
                {
                    let _ = shot_end_fn(instance, 0, 0);
                }
            }
        }

        // Return the shot with measurements and registers
        let shot = Shot {
            measurements: self.state.measurements.clone(),
            registers: self.state.registers.clone(),
            ..Default::default()
        };

        Ok(shot)
    }

    fn reset(&mut self) -> Result<()> {
        // Clean up the runtime instance
        if let Some(lib) = &self.library
            && let Some(instance) = self.instance
        {
            unsafe {
                if let Ok(exit_fn) =
                    lib.get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"selene_runtime_exit")
                {
                    let _ = exit_fn(instance);
                }
            }
        }

        self.instance = None;
        self.library = None;
        self.state = ClassicalState::default();
        self.current_op_index = 0;

        Ok(())
    }
}

impl Drop for SeleneRuntime {
    fn drop(&mut self) {
        let _ = self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selene_runtime_creation() {
        let runtime = SeleneRuntime::new("/path/to/selene.so");
        assert_eq!(runtime.num_qubits(), 0);
        assert!(runtime.is_complete());
    }
}
