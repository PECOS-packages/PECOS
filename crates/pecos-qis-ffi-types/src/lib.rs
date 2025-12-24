//! QIS Interface Data Types
//!
//! This crate provides the data structures for quantum instruction set (QIS) FFI operations.
//! These types can be safely linked into any Rust binary without exporting FFI symbols.
//!
//! The actual FFI implementation (with `#[no_mangle]` functions) is in `pecos-qis-ffi`.

use std::collections::BTreeMap;

mod operations;

pub use operations::{Operation, QuantumOp};

/// Collection of quantum operations from program execution
///
/// This struct is used to collect quantum operations during FFI execution.
/// It's referenced through thread-local storage by the FFI functions.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct OperationCollector {
    /// Collected quantum operations in order
    pub operations: Vec<Operation>,

    /// Mapping of measurement result IDs to their values (when known)
    pub measurements: BTreeMap<usize, Option<bool>>,

    /// Allocated qubit IDs
    pub allocated_qubits: Vec<usize>,

    /// Allocated result IDs
    pub allocated_results: Vec<usize>,

    /// Next available qubit ID
    next_qubit_id: usize,

    /// Next available result ID
    next_result_id: usize,
}

// Type alias for backward compatibility during transition
pub type OperationList = OperationCollector;

impl OperationCollector {
    /// Create a new operation collector
    #[must_use]
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            measurements: BTreeMap::new(),
            allocated_qubits: Vec::new(),
            allocated_results: Vec::new(),
            next_qubit_id: 0,
            next_result_id: 0,
        }
    }

    /// Queue an operation for later execution
    pub fn queue_operation(&mut self, op: Operation) {
        self.operations.push(op);
    }

    /// Allocate a new qubit and return its ID
    pub fn allocate_qubit(&mut self) -> usize {
        let id = self.next_qubit_id;
        self.next_qubit_id += 1;
        self.allocated_qubits.push(id);
        id
    }

    /// Allocate a new result slot and return its ID
    pub fn allocate_result(&mut self) -> usize {
        let id = self.next_result_id;
        self.next_result_id += 1;
        self.allocated_results.push(id);
        self.measurements.insert(id, None);
        id
    }

    /// Store a measurement result (used by runtime when results are available)
    pub fn store_result(&mut self, result_id: usize, value: bool) {
        self.measurements.insert(result_id, Some(value));
    }

    /// Get a measurement result (blocks until available in actual runtime)
    #[must_use]
    pub fn get_result(&self, result_id: usize) -> Option<bool> {
        self.measurements.get(&result_id).and_then(|v| *v)
    }

    /// Pre-populate measurement results (for conditional execution)
    /// This allows setting measurement outcomes before program execution
    pub fn set_measurement_results(&mut self, results: impl IntoIterator<Item = (usize, bool)>) {
        for (result_id, value) in results {
            self.measurements.insert(result_id, Some(value));
        }
    }

    /// Reset the interface for a new shot
    pub fn reset(&mut self) {
        self.operations.clear();
        self.measurements.clear();
        self.allocated_qubits.clear();
        self.allocated_results.clear();
        self.next_qubit_id = 0;
        self.next_result_id = 0;
    }

    /// Extract the collected operations (consumes them)
    pub fn take_operations(&mut self) -> Vec<Operation> {
        std::mem::take(&mut self.operations)
    }
}
