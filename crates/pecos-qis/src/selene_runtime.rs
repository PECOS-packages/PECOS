//! Selene Runtime implementation of `QisRuntime`
//!
//! This wraps a Selene .so runtime plugin and implements the `QisRuntime` trait
//! to provide a Selene-based classical interpreter for QIS programs.

use crate::runtime::{ClassicalState, QisRuntime, Result, RuntimeError, Shot};
use log::{debug, trace};
use pecos_qis_ffi_types::{Operation, OperationCollector, QuantumOp};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::Arc;

/// Selene runtime implementation
///
/// The `library` field is wrapped in `ManuallyDrop` to prevent calling `dlclose()`
/// during process exit. Calling `dlclose()` during shutdown can cause hangs because
/// thread-local storage may already be partially torn down, or other static
/// destructors may be running concurrently.
pub struct SeleneRuntime {
    /// Path to the Selene .so file
    plugin_path: String,

    /// Loaded library (if any)
    /// Wrapped in `ManuallyDrop` to prevent `dlclose()` during process exit.
    #[allow(dead_code)]
    library: Option<ManuallyDrop<Arc<libloading::Library>>>,

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

    /// Flag indicating we need to re-execute with known measurements
    /// Set to true after measurements are provided for dynamic circuits
    needs_reexecution: bool,

    /// Track measurement result IDs that have been seen but not yet resolved
    pending_measurements: Vec<usize>,
}

// SAFETY: SeleneRuntime owns its instance pointer exclusively.
// WARNING: The Selene FFI runtime may not be thread-safe for concurrent access.
// Sync is required by the QisRuntime/Engine trait but callers must ensure
// single-threaded access to any given instance.
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
            needs_reexecution: false,
            pending_measurements: Vec::new(),
        }
    }

    /// Check if this runtime needs re-execution with known measurements
    ///
    /// This is set to true after measurements are provided for programs
    /// that may have conditional logic depending on measurement results.
    #[must_use]
    pub fn needs_reexecution(&self) -> bool {
        self.needs_reexecution
    }

    /// Clear the re-execution flag after operations have been reloaded
    pub fn clear_reexecution_flag(&mut self) {
        self.needs_reexecution = false;
    }

    /// Reload operations from a new execution (used for dynamic circuits)
    pub fn reload_operations(&mut self, operations: OperationCollector) {
        debug!(
            "Reloading operations with {} ops (previous: {} ops)",
            operations.operations.len(),
            self.interface.as_ref().map_or(0, |i| i.operations.len())
        );

        // Update qubit and result counts from new execution
        self.num_qubits = operations
            .allocated_qubits
            .iter()
            .max()
            .map_or(0, |&q| q + 1);
        self.num_results = operations
            .allocated_results
            .iter()
            .max()
            .map_or(0, |&r| r + 1);

        self.interface = Some(operations);
        self.current_op_index = 0;
        self.needs_reexecution = false;
        self.pending_measurements.clear();
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

            self.library = Some(ManuallyDrop::new(lib));
            self.instance = Some(instance);
        }

        Ok(())
    }

    /// Process operations from the interface sequentially
    ///
    /// This method now breaks at measurement operations to allow the quantum
    /// simulator to execute measurements before continuing. This is essential
    /// for dynamic circuits where conditionals depend on measurement results.
    fn process_interface_ops(&mut self) -> Result<Option<Vec<QuantumOp>>> {
        let interface = self
            .interface
            .as_ref()
            .ok_or(RuntimeError::NoProgramLoaded)?;

        self.operations_buffer.clear();
        self.pending_measurements.clear();

        while self.current_op_index < interface.operations.len() {
            let op = &interface.operations[self.current_op_index];

            match op {
                Operation::Quantum(qop) => {
                    trace!("Processing quantum operation: {qop:?}");
                    self.operations_buffer.push(qop.clone());
                    self.current_op_index += 1;

                    // Check if this is a measurement operation
                    if let QuantumOp::Measure(_, result_id) = qop {
                        self.pending_measurements.push(*result_id);
                        debug!(
                            "Breaking batch after measurement (result_id={result_id}) to wait for results"
                        );
                        // Break the batch after measurements to get results
                        // This enables dynamic circuits with conditionals
                        break;
                    }

                    // Also break if we've reached the batch size limit
                    if self.operations_buffer.len() >= self.batch_size {
                        debug!("Breaking batch at size limit ({})", self.batch_size);
                        break;
                    }
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
            needs_reexecution: self.needs_reexecution,
            pending_measurements: self.pending_measurements.clone(),
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
        self.needs_reexecution = false;
        self.pending_measurements.clear();

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
        for (result_id, value) in &measurements {
            trace!(
                "Measurement result {} = {} (num_results={})",
                result_id, value, self.num_results
            );
            self.state.measurements.insert(*result_id, *value);

            // For Selene runtime: Only pass measurements that were explicitly allocated
            // The Selene runtime doesn't support dynamic result allocation, so we must
            // check if this result was known at compile time
            if let Some(interface) = &mut self.interface {
                if interface.allocated_results.contains(result_id) {
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
                                let errno = set_result_fn(instance, *result_id as u64, *value);
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
                interface.store_result(*result_id, *value);
            } else {
                // No interface loaded - just store locally
                log::trace!("No interface loaded, storing measurement {result_id} locally");
            }
        }

        // Check if there are remaining operations that might depend on these measurements
        // If so, we need to re-execute the program with the known measurement values
        // so that conditionals can evaluate correctly
        if let Some(interface) = &self.interface {
            let remaining_ops = interface
                .operations
                .len()
                .saturating_sub(self.current_op_index);
            if remaining_ops > 0 && !measurements.is_empty() {
                debug!(
                    "Setting needs_reexecution=true: {} ops remaining after {} measurements",
                    remaining_ops,
                    measurements.len()
                );
                self.needs_reexecution = true;
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

    fn needs_reexecution(&self) -> bool {
        self.needs_reexecution
    }

    fn clear_reexecution_flag(&mut self) {
        self.needs_reexecution = false;
    }

    fn reload_operations(&mut self, operations: OperationCollector) {
        SeleneRuntime::reload_operations(self, operations);
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
        self.needs_reexecution = false;
        self.pending_measurements.clear();

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
        // Intentionally skip cleanup during drop.
        //
        // IMPORTANT: The FFI call to selene_runtime_exit in reset() can hang
        // during process shutdown because:
        // 1. Thread-local storage may already be partially torn down
        // 2. Other static destructors may be running concurrently
        // 3. The library's internal state may be inconsistent
        //
        // Since drop() is typically called during process exit, it's safe to skip
        // the cleanup and let the OS reclaim all resources. This avoids the
        // intermittent hang that was occurring ~15-20% of the time when running
        // tests in parallel.
        //
        // During normal operation (not process exit), call reset() explicitly
        // before dropping if cleanup is needed.

        // Just clear our local state without making FFI calls
        self.instance = None;
        // Note: We intentionally don't set self.library = None here because
        // the Arc<Library> might be shared, and we don't want to trigger
        // dlclose() during process exit.
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
