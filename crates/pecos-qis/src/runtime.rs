//! QIS Runtime Trait
//!
//! This module defines the trait for classical interpreters that process QIS programs.
//! A `QisRuntime` is responsible for:
//! - Managing control flow (loops, conditionals, function calls)
//! - Maintaining classical state (registers, variables)
//! - Emitting quantum operations for external execution
//! - Handling measurement results for classical control
//!
//! This is analogous to Selene's runtime concept - a classical interpreter that
//! doesn't perform quantum simulation but manages program execution flow.

use log::trace;
use pecos_qis_ffi_types::{LoweredQuantumOp, Operation, OperationCollector, QuantumOp};
use std::collections::BTreeMap;

/// Result type for runtime operations
pub type Result<T> = std::result::Result<T, RuntimeError>;

/// Errors that can occur during runtime execution
#[derive(Debug, Clone)]
pub enum RuntimeError {
    /// Program has not been loaded
    NoProgramLoaded,

    /// Invalid qubit ID
    InvalidQubit(usize),

    /// Invalid result ID
    InvalidResult(usize),

    /// Control flow error (e.g., stack overflow)
    ControlFlowError(String),

    /// Program execution error
    ExecutionError(String),

    /// FFI error when using external runtime
    FfiError(String),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoProgramLoaded => write!(f, "No program has been loaded"),
            Self::InvalidQubit(id) => write!(f, "Invalid qubit ID: {id}"),
            Self::InvalidResult(id) => write!(f, "Invalid result ID: {id}"),
            Self::ControlFlowError(msg) => write!(f, "Control flow error: {msg}"),
            Self::ExecutionError(msg) => write!(f, "Execution error: {msg}"),
            Self::FfiError(msg) => write!(f, "FFI error: {msg}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

/// Classical state maintained by the runtime
#[derive(Debug, Clone, Default)]
pub struct ClassicalState {
    /// Program counter (current instruction)
    pub pc: usize,

    /// Call stack for function calls
    pub call_stack: Vec<CallFrame>,

    /// Classical registers (name -> bits) - `BTreeMap` for deterministic ordering
    pub registers: BTreeMap<String, Vec<bool>>,

    /// Measurement results received - `BTreeMap` for deterministic ordering
    pub measurements: BTreeMap<usize, bool>,

    /// Local variables (name -> value) - `BTreeMap` for deterministic ordering
    pub variables: BTreeMap<String, Value>,

    /// Shot ID for current execution
    pub shot_id: Option<u64>,
}

/// Stack frame for function calls
#[derive(Debug, Clone)]
pub struct CallFrame {
    /// Return address (instruction to return to)
    pub return_address: usize,

    /// Function name
    pub function_name: String,

    /// Local variables for this frame - `BTreeMap` for deterministic ordering
    pub locals: BTreeMap<String, Value>,
}

// Enable cloning of trait objects
dyn_clone::clone_trait_object!(QisRuntime);

/// Classical values that can be stored in variables
#[derive(Debug, Clone)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    BitVec(Vec<bool>),
}

/// Shot result after execution completes
#[derive(Debug, Clone, Default)]
pub struct Shot {
    /// Measurement results by result ID - `BTreeMap` for deterministic ordering
    pub measurements: BTreeMap<usize, bool>,

    /// Classical register values - `BTreeMap` for deterministic ordering
    pub registers: BTreeMap<String, Vec<bool>>,

    /// Additional metadata - `HashMap` is OK here since it's just metadata
    pub metadata: BTreeMap<String, String>,
}

/// Trait for classical interpreters that process QIS programs
///
/// This trait is inspired by Selene's `RuntimeInterface` but adapted for PECOS.
/// Implementations can wrap external runtimes (like Selene .so) via FFI or
/// provide native Rust interpretation.
pub trait QisRuntime: Send + Sync + dyn_clone::DynClone {
    /// Load a QIS program for execution
    ///
    /// This takes the linked QIS interface (program + Rust functions)
    /// and prepares it for execution.
    ///
    /// # Errors
    /// Returns an error if the interface cannot be loaded.
    fn load_interface(&mut self, interface: OperationCollector) -> Result<()>;

    /// Start or continue program execution until quantum operations are needed
    ///
    /// This is analogous to Selene's `get_next_operations()`.
    /// Returns quantum operations to be executed or None if program is complete.
    ///
    /// # Errors
    /// Returns an error if program execution fails.
    fn execute_until_quantum(&mut self) -> Result<Option<Vec<QuantumOp>>>;

    /// Provide measurement results back to the runtime
    ///
    /// The runtime uses these results for classical control flow decisions.
    ///
    /// # Errors
    /// Returns an error if the measurements cannot be provided.
    fn provide_measurements(&mut self, measurements: BTreeMap<usize, bool>) -> Result<()>;

    /// Provide integer-valued measurement outcomes back to the runtime.
    ///
    /// The default keeps existing Boolean runtimes compatible and rejects a
    /// leakage outcome instead of silently converting 2 to true.
    ///
    /// # Errors
    /// Returns an error if any outcome is not 0 or 1, since a Boolean runtime cannot
    /// represent a leakage outcome, or if providing the measurements themselves fails.
    fn provide_measurement_outcomes(&mut self, outcomes: BTreeMap<usize, u32>) -> Result<()> {
        let measurements = outcomes
            .into_iter()
            .map(|(result_id, value)| match value {
                0 => Ok((result_id, false)),
                1 => Ok((result_id, true)),
                _ => Err(RuntimeError::ExecutionError(format!(
                    "runtime does not support leakage outcome {value} for result {result_id}"
                ))),
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        self.provide_measurements(measurements)
    }

    /// Get the current classical state (for debugging/inspection)
    fn get_classical_state(&self) -> &ClassicalState;

    /// Get mutable access to classical state
    fn get_classical_state_mut(&mut self) -> &mut ClassicalState;

    /// Start a new shot
    ///
    /// This resets the runtime state for a new execution of the program.
    /// Inspired by Selene's `shot_start()`.
    ///
    /// # Errors
    /// Returns an error if the shot cannot be started.
    fn shot_start(&mut self, shot_id: u64, seed: Option<u64>) -> Result<()> {
        trace!("Starting shot {shot_id} with seed {seed:?}");
        let state = self.get_classical_state_mut();
        state.pc = 0;
        state.call_stack.clear();
        state.measurements.clear();
        state.variables.clear();
        state.shot_id = Some(shot_id);
        Ok(())
    }

    /// End the current shot and return results
    ///
    /// This finalizes the shot and returns the collected results.
    /// Inspired by Selene's `shot_end()`.
    ///
    /// # Errors
    /// Returns an error if the shot cannot be finalized.
    fn shot_end(&mut self) -> Result<Shot> {
        trace!("Ending shot");
        let state = self.get_classical_state();
        Ok(Shot {
            measurements: state.measurements.clone(),
            registers: state.registers.clone(),
            metadata: BTreeMap::new(),
        })
    }

    /// Check if program execution is complete
    fn is_complete(&self) -> bool;

    /// Reset the runtime for a new execution
    ///
    /// A reset clears execution state; it does NOT start a shot. Shot
    /// boundaries are delivered exactly once per shot via `shot_start` /
    /// `shot_end`, so a runtime keying on shot identity never sees phantom
    /// shot-0 starts from resets (the engine may reset more than once
    /// between shots).
    ///
    /// # Errors
    /// Returns an error if the runtime cannot be reset.
    fn reset(&mut self) -> Result<()> {
        let state = self.get_classical_state_mut();
        state.pc = 0;
        state.call_stack.clear();
        state.measurements.clear();
        state.variables.clear();
        state.shot_id = None;
        Ok(())
    }

    /// Get the number of qubits used by the program
    fn num_qubits(&self) -> usize;

    /// Provide an external qubit-count hint before runtime initialization.
    ///
    /// Dynamic programs discover qubits while the program runs, but some
    /// runtime plugins need the total device size during initialization.
    fn set_num_qubits(&mut self, _num_qubits: usize) {
        // Default implementation does nothing.
    }

    /// Set the maximum number of operations to batch
    ///
    /// This allows tuning the trade-off between runtime overhead and
    /// quantum simulator efficiency.
    fn set_batch_size(&mut self, size: usize) {
        // Default implementation does nothing
        let _ = size;
    }

    /// Whether this runtime can lower freshly collected program operations.
    ///
    /// Dynamic QIS execution yields PECOS `Operation`s from the running program.
    /// Most runtimes leave those operations for the engine to lower directly.
    /// Selene runtime plugins opt in here so the operation stream flows through
    /// the runtime scheduler before it reaches the PECOS quantum/noise stack.
    fn supports_operation_lowering(&self) -> bool {
        false
    }

    /// Lower freshly collected program operations through the runtime.
    ///
    /// Implementations that return `true` from `supports_operation_lowering`
    /// should accept the given operations, drain any runtime-ready scheduled
    /// operations, and return them as PECOS quantum operations.
    ///
    /// # Errors
    /// Returns an error if the runtime cannot accept or lower the operations.
    fn lower_operations(&mut self, _operations: &[Operation]) -> Result<Vec<QuantumOp>> {
        Err(RuntimeError::ExecutionError(
            "runtime does not support operation lowering".to_string(),
        ))
    }

    /// Lower freshly collected program operations through the runtime with provenance.
    ///
    /// Runtimes that can preserve source/scheduler metadata should override this
    /// method. The default preserves the existing `lower_operations` behavior and
    /// attaches empty metadata to every lowered operation.
    ///
    /// # Errors
    /// Returns an error if the runtime cannot accept or lower the operations.
    fn lower_operations_with_metadata(
        &mut self,
        operations: &[Operation],
    ) -> Result<Vec<LoweredQuantumOp>> {
        self.lower_operations(operations)
            .map(|ops| ops.into_iter().map(LoweredQuantumOp::from).collect())
    }

    /// Flush and drain any operations the runtime scheduler is still holding.
    ///
    /// A scheduling runtime may defer operations across lowering batches.
    /// The engine calls this at shot completion, before certifying the trace
    /// complete: a non-empty result means lowered operations exist that no
    /// batch will ever carry, so the shot must fail rather than certify.
    /// Implementations must force their scheduler to release held work (e.g.
    /// a terminal barrier) before collecting, not merely poll for ready
    /// operations.
    ///
    /// The default fails closed: the engine only calls this for runtimes
    /// returning `true` from `supports_operation_lowering`, and a lowering
    /// runtime that has not implemented the drain protocol cannot be
    /// certified drained. Non-lowering runtimes are never asked.
    ///
    /// # Errors
    /// Returns an error if the runtime cannot prove its scheduler is drained.
    fn drain_pending_operations(&mut self) -> Result<Vec<QuantumOp>> {
        Err(RuntimeError::ExecutionError(
            "lowering runtime does not implement drain_pending_operations; \
             cannot verify the scheduler is drained"
                .to_string(),
        ))
    }

    /// Check if the runtime needs to re-execute with known measurements
    ///
    /// This is set to true after measurements are provided for programs
    /// that may have conditional logic depending on measurement results.
    /// The engine should call `reload_operations()` with new operations
    /// from the interface's `execute_with_measurements()`.
    fn needs_reexecution(&self) -> bool {
        // Default: no re-execution needed (for static circuits)
        false
    }

    /// Clear the re-execution flag after operations have been reloaded
    fn clear_reexecution_flag(&mut self) {
        // Default: no-op
    }

    /// Reload operations from a new execution (used for dynamic circuits)
    ///
    /// This is called when re-executing with known measurement values
    /// allows conditionals in the program to take the correct branches.
    fn reload_operations(&mut self, operations: OperationCollector) {
        // Default: just reload via load_interface
        let _ = self.load_interface(operations);
    }
}
