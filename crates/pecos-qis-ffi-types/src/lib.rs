//! QIS Interface Data Types
//!
//! Data structures for quantum instruction set (QIS) FFI operations.
//! These types can be safely linked into any Rust binary without exporting FFI symbols.
//!
//! The actual FFI implementation (with `#[no_mangle]` functions) is in `pecos-qis-ffi`.

mod operations;

pub use operations::{Operation, QuantumOp};

const DEFAULT_OPERATION_CAPACITY: usize = 1024;
const DEFAULT_MEASUREMENT_CAPACITY: usize = 256;
const DEFAULT_ID_CAPACITY: usize = 128;

/// Collection of quantum operations from program execution
///
/// This struct is used to collect quantum operations during FFI execution.
/// It's referenced through thread-local storage by the FFI functions.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct OperationCollector {
    /// Collected quantum operations in order
    pub operations: Vec<Operation>,

    /// Mapping of measurement result IDs to their values (when known)
    pub measurements: Vec<Option<bool>>,

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
            operations: Vec::with_capacity(DEFAULT_OPERATION_CAPACITY),
            measurements: Vec::with_capacity(DEFAULT_MEASUREMENT_CAPACITY),
            allocated_qubits: Vec::with_capacity(DEFAULT_ID_CAPACITY),
            allocated_results: Vec::with_capacity(DEFAULT_ID_CAPACITY),
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
        match self.measurements.len().cmp(&id) {
            std::cmp::Ordering::Equal => self.measurements.push(None),
            std::cmp::Ordering::Less => {
                self.measurements.resize(id, None);
                self.measurements.push(None);
            }
            std::cmp::Ordering::Greater => {
                self.measurements[id] = None;
            }
        }
        id
    }

    /// Store a measurement result (used by runtime when results are available)
    pub fn store_result(&mut self, result_id: usize, value: bool) {
        if self.measurements.len() <= result_id {
            self.measurements.resize(result_id + 1, None);
        }
        self.measurements[result_id] = Some(value);
    }

    /// Get a measurement result (blocks until available in actual runtime)
    #[must_use]
    pub fn get_result(&self, result_id: usize) -> Option<bool> {
        self.measurements.get(result_id).copied().flatten()
    }

    /// Pre-populate measurement results (for conditional execution)
    /// This allows setting measurement outcomes before program execution
    pub fn set_measurement_results(&mut self, results: impl IntoIterator<Item = (usize, bool)>) {
        for (result_id, value) in results {
            if self.measurements.len() <= result_id {
                self.measurements.resize(result_id + 1, None);
            }
            self.measurements[result_id] = Some(value);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_collector() {
        let collector = OperationCollector::new();
        assert!(collector.operations.is_empty());
        assert!(collector.measurements.is_empty());
        assert!(collector.allocated_qubits.is_empty());
        assert!(collector.allocated_results.is_empty());
    }

    #[test]
    fn test_default_collector() {
        let collector = OperationCollector::default();
        assert!(collector.operations.is_empty());
    }

    #[test]
    fn test_queue_operation() {
        let mut collector = OperationCollector::new();
        collector.queue_operation(Operation::AllocateQubit { id: 0 });
        collector.queue_operation(QuantumOp::H(0).into());

        assert_eq!(collector.operations.len(), 2);
        assert_eq!(collector.operations[0], Operation::AllocateQubit { id: 0 });
        assert_eq!(collector.operations[1], Operation::Quantum(QuantumOp::H(0)));
    }

    #[test]
    fn test_allocate_qubit() {
        let mut collector = OperationCollector::new();

        let q0 = collector.allocate_qubit();
        let q1 = collector.allocate_qubit();
        let q2 = collector.allocate_qubit();

        assert_eq!(q0, 0);
        assert_eq!(q1, 1);
        assert_eq!(q2, 2);
        assert_eq!(collector.allocated_qubits, vec![0, 1, 2]);
    }

    #[test]
    fn test_allocate_result() {
        let mut collector = OperationCollector::new();

        let r0 = collector.allocate_result();
        let r1 = collector.allocate_result();

        assert_eq!(r0, 0);
        assert_eq!(r1, 1);
        assert_eq!(collector.allocated_results, vec![0, 1]);
        // Results should be initialized to None
        assert_eq!(collector.measurements.first(), Some(&None));
        assert_eq!(collector.measurements.get(1), Some(&None));
    }

    #[test]
    fn test_store_and_get_result() {
        let mut collector = OperationCollector::new();
        let r0 = collector.allocate_result();

        // Initially None
        assert_eq!(collector.get_result(r0), None);

        // Store a result
        collector.store_result(r0, true);
        assert_eq!(collector.get_result(r0), Some(true));

        // Store another result
        collector.store_result(r0, false);
        assert_eq!(collector.get_result(r0), Some(false));
    }

    #[test]
    fn test_get_result_nonexistent() {
        let collector = OperationCollector::new();
        assert_eq!(collector.get_result(999), None);
    }

    #[test]
    fn test_set_measurement_results() {
        let mut collector = OperationCollector::new();

        collector.set_measurement_results([(0, true), (1, false), (2, true)]);

        assert_eq!(collector.get_result(0), Some(true));
        assert_eq!(collector.get_result(1), Some(false));
        assert_eq!(collector.get_result(2), Some(true));
    }

    #[test]
    fn test_reset() {
        let mut collector = OperationCollector::new();

        // Add some state
        collector.allocate_qubit();
        collector.allocate_qubit();
        collector.allocate_result();
        collector.queue_operation(QuantumOp::H(0).into());
        collector.store_result(0, true);

        // Verify state exists
        assert!(!collector.operations.is_empty());
        assert!(!collector.allocated_qubits.is_empty());

        // Reset
        collector.reset();

        // Verify all cleared
        assert!(collector.operations.is_empty());
        assert!(collector.measurements.is_empty());
        assert!(collector.allocated_qubits.is_empty());
        assert!(collector.allocated_results.is_empty());

        // Verify IDs reset
        let q = collector.allocate_qubit();
        let r = collector.allocate_result();
        assert_eq!(q, 0);
        assert_eq!(r, 0);
    }

    #[test]
    fn test_take_operations() {
        let mut collector = OperationCollector::new();
        collector.queue_operation(QuantumOp::H(0).into());
        collector.queue_operation(QuantumOp::X(1).into());

        let ops = collector.take_operations();

        assert_eq!(ops.len(), 2);
        assert!(collector.operations.is_empty());
    }

    #[test]
    fn test_clone() {
        let mut collector = OperationCollector::new();
        collector.allocate_qubit();
        collector.queue_operation(QuantumOp::H(0).into());

        let cloned = collector.clone();

        assert_eq!(cloned.operations.len(), 1);
        assert_eq!(cloned.allocated_qubits.len(), 1);
    }

    #[test]
    fn test_debug_format() {
        let collector = OperationCollector::new();
        let debug_str = format!("{collector:?}");
        assert!(debug_str.contains("OperationCollector"));
    }

    #[test]
    fn test_operation_from_quantum_op() {
        let quantum_op = QuantumOp::CX(0, 1);
        let operation: Operation = quantum_op.clone().into();

        assert_eq!(operation, Operation::Quantum(quantum_op));
    }

    #[test]
    fn test_all_quantum_ops() {
        let mut collector = OperationCollector::new();

        // Single-qubit gates
        collector.queue_operation(QuantumOp::H(0).into());
        collector.queue_operation(QuantumOp::X(0).into());
        collector.queue_operation(QuantumOp::Y(0).into());
        collector.queue_operation(QuantumOp::Z(0).into());
        collector.queue_operation(QuantumOp::S(0).into());
        collector.queue_operation(QuantumOp::Sdg(0).into());
        collector.queue_operation(QuantumOp::T(0).into());
        collector.queue_operation(QuantumOp::Tdg(0).into());

        // Rotation gates
        collector.queue_operation(QuantumOp::RX(1.57, 0).into());
        collector.queue_operation(QuantumOp::RY(1.57, 0).into());
        collector.queue_operation(QuantumOp::RZ(1.57, 0).into());
        collector.queue_operation(QuantumOp::RXY(1.57, 0.78, 0).into());

        // Two-qubit gates
        collector.queue_operation(QuantumOp::CX(0, 1).into());
        collector.queue_operation(QuantumOp::CY(0, 1).into());
        collector.queue_operation(QuantumOp::CZ(0, 1).into());
        collector.queue_operation(QuantumOp::CH(0, 1).into());
        collector.queue_operation(QuantumOp::CRZ(1.57, 0, 1).into());
        collector.queue_operation(QuantumOp::ZZ(0, 1).into());
        collector.queue_operation(QuantumOp::RZZ(1.57, 0, 1).into());

        // Three-qubit gates
        collector.queue_operation(QuantumOp::CCX(0, 1, 2).into());

        // Measurement and reset
        collector.queue_operation(QuantumOp::Measure(0, 0).into());
        collector.queue_operation(QuantumOp::Reset(0).into());

        assert_eq!(collector.operations.len(), 22);
    }

    #[test]
    fn test_all_operation_types() {
        let mut collector = OperationCollector::new();

        collector.queue_operation(Operation::AllocateQubit { id: 0 });
        collector.queue_operation(Operation::AllocateResult { id: 0 });
        collector.queue_operation(Operation::ReleaseQubit { id: 0 });
        collector.queue_operation(Operation::RecordOutput {
            result_id: 0,
            register_name: "c0".to_string(),
        });
        collector.queue_operation(Operation::Barrier);
        collector.queue_operation(Operation::Quantum(QuantumOp::H(0)));

        assert_eq!(collector.operations.len(), 6);
    }

    #[test]
    fn test_operation_equality() {
        let op1 = Operation::AllocateQubit { id: 5 };
        let op2 = Operation::AllocateQubit { id: 5 };
        let op3 = Operation::AllocateQubit { id: 6 };

        assert_eq!(op1, op2);
        assert_ne!(op1, op3);
    }

    #[test]
    fn test_quantum_op_equality() {
        let op1 = QuantumOp::RX(1.5, 0);
        let op2 = QuantumOp::RX(1.5, 0);
        let op3 = QuantumOp::RX(1.6, 0);

        assert_eq!(op1, op2);
        assert_ne!(op1, op3);
    }
}
