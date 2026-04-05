//! Batched circuit execution for DOD-optimized performance.
//!
//! This module provides:
//! - `BatchedCircuit`: Groups operations by type for cache-efficient execution
//! - `BatchExecutor`: Trait for simulators to implement batched execution
//! - `SimpleBatchExecutor`: Sequential fallback for simulators without batching

use super::{GateId, ResolvedCircuit, ResolvedOp};
use pecos_core::{Angle64, QubitId};
use smallvec::SmallVec;

/// A batch of operations of the same type.
///
/// Batches enable:
/// - Single virtual dispatch per gate type (not per gate)
/// - SIMD-friendly contiguous data
/// - Better cache utilization
#[derive(Clone, Debug)]
#[allow(clippy::type_complexity)] // MultiAngle intentionally uses nested generics
pub enum Batch {
    /// Single-qubit gates with no angles.
    SingleQubit {
        gate_id: GateId,
        qubits: Vec<QubitId>,
    },

    /// Two-qubit gates with no angles.
    TwoQubit {
        gate_id: GateId,
        pairs: Vec<(QubitId, QubitId)>,
    },

    /// Rotation gates (single angle).
    Rotation {
        gate_id: GateId,
        ops: Vec<(QubitId, Angle64)>,
    },

    /// Multi-angle gates (rare, but supported).
    MultiAngle {
        gate_id: GateId,
        ops: Vec<(SmallVec<[QubitId; 4]>, SmallVec<[Angle64; 3]>)>,
    },

    /// Preparation operations.
    Prep {
        basis: super::PrepBasis,
        qubits: Vec<QubitId>,
    },

    /// Measurement operations.
    Measure {
        basis: super::MeasBasis,
        ops: Vec<(QubitId, super::ResultId)>,
    },

    /// Conditional operations (cannot be batched, executed individually).
    Conditional {
        condition: super::ResultId,
        if_one: Vec<ResolvedOp>,
        if_zero: Vec<ResolvedOp>,
    },

    /// Classical XOR operations.
    XorResult {
        ops: Vec<(super::ResultId, super::ResultId)>,
    },

    /// Output result markers.
    OutputResult { results: Vec<super::ResultId> },
}

impl Batch {
    /// Get the number of operations in this batch.
    #[must_use]
    #[allow(clippy::match_same_arms)] // Different fields, intentionally separate arms
    pub fn len(&self) -> usize {
        match self {
            Self::SingleQubit { qubits, .. } => qubits.len(),
            Self::TwoQubit { pairs, .. } => pairs.len(),
            Self::Rotation { ops, .. } => ops.len(),
            Self::MultiAngle { ops, .. } => ops.len(),
            Self::Prep { qubits, .. } => qubits.len(),
            Self::Measure { ops, .. } => ops.len(),
            Self::Conditional { .. } => 1,
            Self::XorResult { ops } => ops.len(),
            Self::OutputResult { results } => results.len(),
        }
    }

    /// Check if this batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A circuit organized into batches for efficient execution.
///
/// Operations are grouped by type and executed in sequence.
/// Within each batch, operations can potentially be parallelized.
#[derive(Clone, Debug, Default)]
pub struct BatchedCircuit {
    /// Batches of operations in execution order.
    pub batches: Vec<Batch>,
    /// Number of result slots used.
    pub result_count: usize,
}

impl BatchedCircuit {
    /// Create a new empty batched circuit.
    #[must_use]
    pub fn new() -> Self {
        Self {
            batches: Vec::new(),
            result_count: 0,
        }
    }

    /// Create a batched circuit from a resolved circuit.
    ///
    /// Groups consecutive operations of the same type into batches.
    ///
    /// # Panics
    /// Panics if `current_batch` is unexpectedly `None` during flush.
    #[must_use]
    pub fn from_resolved(resolved: &ResolvedCircuit) -> Self {
        let mut batched = Self::new();
        batched.result_count = resolved.result_count;

        let mut current_batch: Option<Batch> = None;

        for op in &resolved.ops {
            // Check if we can extend the current batch
            if let Some(ref mut batch) = current_batch {
                if Self::can_extend(batch, op) {
                    Self::extend_batch(batch, op);
                    continue;
                }
                // Flush current batch and start new one
                batched.batches.push(
                    current_batch
                        .take()
                        .expect("current_batch is Some in this branch"),
                );
            }

            // Start a new batch
            current_batch = Some(Self::start_batch(op));
        }

        // Flush final batch
        if let Some(batch) = current_batch {
            batched.batches.push(batch);
        }

        batched
    }

    /// Check if an operation can extend the current batch.
    fn can_extend(batch: &Batch, op: &ResolvedOp) -> bool {
        match (batch, op) {
            (
                Batch::SingleQubit { gate_id: g1, .. },
                ResolvedOp::Gate {
                    gate_id: g2,
                    qubits,
                    angles,
                },
            ) if qubits.len() == 1 && angles.is_empty() => *g1 == *g2,
            (
                Batch::TwoQubit { gate_id: g1, .. },
                ResolvedOp::Gate {
                    gate_id: g2,
                    qubits,
                    angles,
                },
            ) if qubits.len() == 2 && angles.is_empty() => *g1 == *g2,
            (
                Batch::Rotation { gate_id: g1, .. },
                ResolvedOp::Gate {
                    gate_id: g2,
                    qubits,
                    angles,
                },
            ) if qubits.len() == 1 && angles.len() == 1 => *g1 == *g2,
            (Batch::Prep { basis: b1, .. }, ResolvedOp::Prep { basis: b2, .. }) => *b1 == *b2,
            (Batch::Measure { basis: b1, .. }, ResolvedOp::Measure { basis: b2, .. }) => *b1 == *b2,
            (Batch::XorResult { .. }, ResolvedOp::XorResult { .. })
            | (Batch::OutputResult { .. }, ResolvedOp::OutputResult { .. }) => true,
            _ => false,
        }
    }

    /// Start a new batch from an operation.
    fn start_batch(op: &ResolvedOp) -> Batch {
        match op {
            ResolvedOp::Gate {
                gate_id,
                qubits,
                angles,
            } => {
                if angles.is_empty() {
                    if qubits.len() == 1 {
                        Batch::SingleQubit {
                            gate_id: *gate_id,
                            qubits: vec![qubits[0]],
                        }
                    } else if qubits.len() == 2 {
                        Batch::TwoQubit {
                            gate_id: *gate_id,
                            pairs: vec![(qubits[0], qubits[1])],
                        }
                    } else {
                        Batch::MultiAngle {
                            gate_id: *gate_id,
                            ops: vec![(qubits.clone(), angles.clone())],
                        }
                    }
                } else if angles.len() == 1 && qubits.len() == 1 {
                    Batch::Rotation {
                        gate_id: *gate_id,
                        ops: vec![(qubits[0], angles[0])],
                    }
                } else {
                    Batch::MultiAngle {
                        gate_id: *gate_id,
                        ops: vec![(qubits.clone(), angles.clone())],
                    }
                }
            }
            ResolvedOp::Prep { qubit, basis } => Batch::Prep {
                basis: *basis,
                qubits: vec![*qubit],
            },
            ResolvedOp::Measure {
                qubit,
                basis,
                result,
            } => Batch::Measure {
                basis: *basis,
                ops: vec![(*qubit, *result)],
            },
            ResolvedOp::Conditional {
                condition,
                if_one,
                if_zero,
            } => Batch::Conditional {
                condition: *condition,
                if_one: if_one.clone(),
                if_zero: if_zero.clone(),
            },
            ResolvedOp::XorResult { target, source } => Batch::XorResult {
                ops: vec![(*target, *source)],
            },
            ResolvedOp::OutputResult { result } => Batch::OutputResult {
                results: vec![*result],
            },
        }
    }

    /// Extend a batch with an operation.
    fn extend_batch(batch: &mut Batch, op: &ResolvedOp) {
        match (batch, op) {
            (Batch::SingleQubit { qubits, .. }, ResolvedOp::Gate { qubits: q, .. }) => {
                qubits.push(q[0]);
            }
            (Batch::TwoQubit { pairs, .. }, ResolvedOp::Gate { qubits: q, .. }) => {
                pairs.push((q[0], q[1]));
            }
            (
                Batch::Rotation { ops, .. },
                ResolvedOp::Gate {
                    qubits: q,
                    angles: a,
                    ..
                },
            ) => {
                ops.push((q[0], a[0]));
            }
            (Batch::Prep { qubits, .. }, ResolvedOp::Prep { qubit, .. }) => {
                qubits.push(*qubit);
            }
            (Batch::Measure { ops, .. }, ResolvedOp::Measure { qubit, result, .. }) => {
                ops.push((*qubit, *result));
            }
            (Batch::XorResult { ops }, ResolvedOp::XorResult { target, source }) => {
                ops.push((*target, *source));
            }
            (Batch::OutputResult { results }, ResolvedOp::OutputResult { result }) => {
                results.push(*result);
            }
            _ => {} // Should not happen if can_extend is correct
        }
    }

    /// Get the total number of operations across all batches.
    #[must_use]
    pub fn op_count(&self) -> usize {
        self.batches.iter().map(Batch::len).sum()
    }

    /// Get the number of batches.
    #[must_use]
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }
}

/// Trait for executing batched circuits.
///
/// Simulators can implement this trait to take advantage of batched execution.
/// Default implementations are provided that execute operations sequentially.
pub trait BatchExecutor {
    /// The result type for measurements.
    type MeasurementResults;

    /// Execute a single-qubit gate batch.
    fn execute_single_qubit(&mut self, gate_id: GateId, qubits: &[QubitId]);

    /// Execute a two-qubit gate batch.
    fn execute_two_qubit(&mut self, gate_id: GateId, pairs: &[(QubitId, QubitId)]);

    /// Execute a rotation gate batch.
    fn execute_rotation(&mut self, gate_id: GateId, ops: &[(QubitId, Angle64)]);

    /// Execute preparations.
    fn execute_prep(&mut self, basis: super::PrepBasis, qubits: &[QubitId]);

    /// Execute measurements.
    fn execute_measure(
        &mut self,
        basis: super::MeasBasis,
        ops: &[(QubitId, super::ResultId)],
        results: &mut Self::MeasurementResults,
    );

    /// Execute a conditional block.
    fn execute_conditional(
        &mut self,
        condition: super::ResultId,
        if_one: &[ResolvedOp],
        if_zero: &[ResolvedOp],
        results: &mut Self::MeasurementResults,
    );

    /// Execute the full batched circuit.
    fn execute_batched(&mut self, circuit: &BatchedCircuit) -> Self::MeasurementResults
    where
        Self::MeasurementResults: Default,
    {
        let mut results = Self::MeasurementResults::default();

        for batch in &circuit.batches {
            match batch {
                Batch::SingleQubit { gate_id, qubits } => {
                    self.execute_single_qubit(*gate_id, qubits);
                }
                Batch::TwoQubit { gate_id, pairs } => {
                    self.execute_two_qubit(*gate_id, pairs);
                }
                Batch::Rotation { gate_id, ops } => {
                    self.execute_rotation(*gate_id, ops);
                }
                Batch::MultiAngle { gate_id, ops } => {
                    // Fall back to individual execution for multi-angle gates
                    for (qubits, angles) in ops {
                        if qubits.len() == 1 && angles.len() == 1 {
                            self.execute_rotation(*gate_id, &[(qubits[0], angles[0])]);
                        } else if qubits.len() == 1 && angles.is_empty() {
                            self.execute_single_qubit(*gate_id, &[qubits[0]]);
                        } else if qubits.len() == 2 && angles.is_empty() {
                            self.execute_two_qubit(*gate_id, &[(qubits[0], qubits[1])]);
                        }
                        // Other cases would need more specific handling
                    }
                }
                Batch::Prep { basis, qubits } => {
                    self.execute_prep(*basis, qubits);
                }
                Batch::Measure { basis, ops } => {
                    self.execute_measure(*basis, ops, &mut results);
                }
                Batch::Conditional {
                    condition,
                    if_one,
                    if_zero,
                } => {
                    self.execute_conditional(*condition, if_one, if_zero, &mut results);
                }
                Batch::XorResult { .. } | Batch::OutputResult { .. } => {
                    // Classical operations handled by results tracking
                }
            }
        }

        results
    }
}

/// Simple executor that processes operations one at a time.
///
/// This trait provides a simpler interface for simulators that don't
/// need batched execution. A blanket implementation of `BatchExecutor`
/// is provided for any type implementing this trait.
pub trait SimpleExecutor {
    /// The result type for measurements.
    type MeasurementResults: Default;

    /// Execute a single gate operation.
    fn execute_gate(&mut self, gate_id: GateId, qubits: &[QubitId], angles: &[Angle64]);

    /// Execute a preparation.
    fn execute_prep(&mut self, basis: super::PrepBasis, qubit: QubitId);

    /// Execute a measurement.
    fn execute_measure(
        &mut self,
        basis: super::MeasBasis,
        qubit: QubitId,
        result: super::ResultId,
        results: &mut Self::MeasurementResults,
    );

    /// Get a measurement result for conditional execution.
    fn get_result(&self, result: super::ResultId, results: &Self::MeasurementResults) -> bool;
}

/// Blanket implementation: any `SimpleExecutor` is also a `BatchExecutor`.
impl<T: SimpleExecutor> BatchExecutor for T {
    type MeasurementResults = T::MeasurementResults;

    fn execute_single_qubit(&mut self, gate_id: GateId, qubits: &[QubitId]) {
        for &qubit in qubits {
            self.execute_gate(gate_id, &[qubit], &[]);
        }
    }

    fn execute_two_qubit(&mut self, gate_id: GateId, pairs: &[(QubitId, QubitId)]) {
        for &(q0, q1) in pairs {
            self.execute_gate(gate_id, &[q0, q1], &[]);
        }
    }

    fn execute_rotation(&mut self, gate_id: GateId, ops: &[(QubitId, Angle64)]) {
        for &(qubit, angle) in ops {
            self.execute_gate(gate_id, &[qubit], &[angle]);
        }
    }

    fn execute_prep(&mut self, basis: super::PrepBasis, qubits: &[QubitId]) {
        for &qubit in qubits {
            SimpleExecutor::execute_prep(self, basis, qubit);
        }
    }

    fn execute_measure(
        &mut self,
        basis: super::MeasBasis,
        ops: &[(QubitId, super::ResultId)],
        results: &mut Self::MeasurementResults,
    ) {
        for &(qubit, result) in ops {
            SimpleExecutor::execute_measure(self, basis, qubit, result, results);
        }
    }

    fn execute_conditional(
        &mut self,
        condition: super::ResultId,
        if_one: &[ResolvedOp],
        if_zero: &[ResolvedOp],
        results: &mut Self::MeasurementResults,
    ) {
        let cond_value = self.get_result(condition, results);
        let ops = if cond_value { if_one } else { if_zero };

        for op in ops {
            match op {
                ResolvedOp::Gate {
                    gate_id,
                    qubits,
                    angles,
                } => {
                    self.execute_gate(*gate_id, qubits, angles);
                }
                ResolvedOp::Prep { qubit, basis } => {
                    SimpleExecutor::execute_prep(self, *basis, *qubit);
                }
                ResolvedOp::Measure {
                    qubit,
                    basis,
                    result,
                } => {
                    SimpleExecutor::execute_measure(self, *basis, *qubit, *result, results);
                }
                ResolvedOp::Conditional {
                    condition,
                    if_one,
                    if_zero,
                } => {
                    self.execute_conditional(*condition, if_one, if_zero, results);
                }
                ResolvedOp::XorResult { .. } | ResolvedOp::OutputResult { .. } => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::gates;
    use super::*;

    #[test]
    fn test_batch_from_resolved_groups_same_gates() {
        let resolved = ResolvedCircuit::new(vec![
            ResolvedOp::gate1(gates::H, QubitId(0)),
            ResolvedOp::gate1(gates::H, QubitId(1)),
            ResolvedOp::gate1(gates::H, QubitId(2)),
            ResolvedOp::gate2(gates::CX, QubitId(0), QubitId(1)),
            ResolvedOp::gate2(gates::CX, QubitId(1), QubitId(2)),
        ]);

        let batched = BatchedCircuit::from_resolved(&resolved);

        // Should have 2 batches: 3 H gates, then 2 CX gates
        assert_eq!(batched.batch_count(), 2);
        assert_eq!(batched.op_count(), 5);

        match &batched.batches[0] {
            Batch::SingleQubit { gate_id, qubits } => {
                assert_eq!(*gate_id, gates::H);
                assert_eq!(qubits.len(), 3);
            }
            _ => panic!("Expected SingleQubit batch"),
        }

        match &batched.batches[1] {
            Batch::TwoQubit { gate_id, pairs } => {
                assert_eq!(*gate_id, gates::CX);
                assert_eq!(pairs.len(), 2);
            }
            _ => panic!("Expected TwoQubit batch"),
        }
    }

    #[test]
    fn test_batch_separates_different_gates() {
        let resolved = ResolvedCircuit::new(vec![
            ResolvedOp::gate1(gates::H, QubitId(0)),
            ResolvedOp::gate1(gates::X, QubitId(0)),
            ResolvedOp::gate1(gates::H, QubitId(1)),
        ]);

        let batched = BatchedCircuit::from_resolved(&resolved);

        // Should have 3 batches: H, X, H (different gate types don't merge)
        assert_eq!(batched.batch_count(), 3);
    }

    #[test]
    fn test_batch_rotations() {
        let resolved = ResolvedCircuit::new(vec![
            ResolvedOp::rotation(gates::RZ, QubitId(0), Angle64::QUARTER_TURN),
            ResolvedOp::rotation(gates::RZ, QubitId(1), Angle64::HALF_TURN),
        ]);

        let batched = BatchedCircuit::from_resolved(&resolved);

        assert_eq!(batched.batch_count(), 1);

        match &batched.batches[0] {
            Batch::Rotation { gate_id, ops } => {
                assert_eq!(*gate_id, gates::RZ);
                assert_eq!(ops.len(), 2);
                assert_eq!(ops[0], (QubitId(0), Angle64::QUARTER_TURN));
                assert_eq!(ops[1], (QubitId(1), Angle64::HALF_TURN));
            }
            _ => panic!("Expected Rotation batch"),
        }
    }

    #[test]
    fn test_batch_prep_measure() {
        use super::super::{MeasBasis, PrepBasis, ResultId};

        let resolved = ResolvedCircuit::new(vec![
            ResolvedOp::Prep {
                qubit: QubitId(0),
                basis: PrepBasis::Z,
            },
            ResolvedOp::Prep {
                qubit: QubitId(1),
                basis: PrepBasis::Z,
            },
            ResolvedOp::gate1(gates::H, QubitId(0)),
            ResolvedOp::Measure {
                qubit: QubitId(0),
                basis: MeasBasis::Z,
                result: ResultId(0),
            },
            ResolvedOp::Measure {
                qubit: QubitId(1),
                basis: MeasBasis::Z,
                result: ResultId(1),
            },
        ]);

        let batched = BatchedCircuit::from_resolved(&resolved);

        // Prep(Z) batch, H batch, Measure(Z) batch
        assert_eq!(batched.batch_count(), 3);

        match &batched.batches[0] {
            Batch::Prep { basis, qubits } => {
                assert_eq!(*basis, PrepBasis::Z);
                assert_eq!(qubits.len(), 2);
            }
            _ => panic!("Expected Prep batch"),
        }

        match &batched.batches[2] {
            Batch::Measure { basis, ops } => {
                assert_eq!(*basis, MeasBasis::Z);
                assert_eq!(ops.len(), 2);
            }
            _ => panic!("Expected Measure batch"),
        }
    }

    #[test]
    fn test_batch_conditional_not_merged() {
        use super::super::ResultId;

        let resolved = ResolvedCircuit::new(vec![
            ResolvedOp::gate1(gates::H, QubitId(0)),
            ResolvedOp::Conditional {
                condition: ResultId(0),
                if_one: vec![ResolvedOp::gate1(gates::X, QubitId(0))],
                if_zero: vec![],
            },
            ResolvedOp::gate1(gates::H, QubitId(0)),
        ]);

        let batched = BatchedCircuit::from_resolved(&resolved);

        // Conditionals should not be merged
        assert_eq!(batched.batch_count(), 3);
        assert!(matches!(batched.batches[1], Batch::Conditional { .. }));
    }

    #[test]
    fn test_empty_batch() {
        let resolved = ResolvedCircuit::new(vec![]);
        let batched = BatchedCircuit::from_resolved(&resolved);

        assert_eq!(batched.batch_count(), 0);
        assert_eq!(batched.op_count(), 0);
    }

    #[test]
    fn test_batch_len() {
        let batch = Batch::SingleQubit {
            gate_id: gates::H,
            qubits: vec![QubitId(0), QubitId(1), QubitId(2)],
        };
        assert_eq!(batch.len(), 3);
        assert!(!batch.is_empty());

        let empty = Batch::SingleQubit {
            gate_id: gates::H,
            qubits: vec![],
        };
        assert!(empty.is_empty());
    }
}
