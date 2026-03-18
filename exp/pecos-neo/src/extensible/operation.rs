//! Extended operations for adaptors with measurement and control flow.
//!
//! This module extends the basic gate adaptor concept to support:
//! - Measurements with result tracking
//! - Preparations in arbitrary bases
//! - Conditional operations based on measurement outcomes
//! - Ancilla qubit management

use super::{GateId, gates};
use pecos_core::{Angle64, QubitId};
use smallvec::SmallVec;

/// Identifier for a measurement result within an adapted operation sequence.
///
/// Results are scoped to a single adaptor expansion and can be referenced
/// by conditional operations within that expansion.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ResultId(pub u16);

/// Basis for preparation operations.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub enum PrepBasis {
    /// Prepare |0⟩ (Z basis, +1 eigenstate)
    #[default]
    Z,
    /// Prepare |+⟩ (X basis, +1 eigenstate)
    X,
    /// Prepare |+i⟩ (Y basis, +1 eigenstate)
    Y,
}

/// Basis for measurement operations.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub enum MeasBasis {
    /// Measure in Z basis
    #[default]
    Z,
    /// Measure in X basis
    X,
    /// Measure in Y basis
    Y,
}

/// An operation in an adapted sequence.
///
/// This extends `AdaptedGate` to support the full range of operations
/// needed for stabilizer measurements and other composite operations.
#[derive(Clone, Debug)]
pub enum AdaptedOp {
    /// A quantum gate.
    Gate {
        gate_id: GateId,
        qubits: SmallVec<[QubitId; 4]>,
        angles: SmallVec<[Angle64; 3]>,
    },

    /// Prepare a qubit in a specific basis.
    Prep { qubit: QubitId, basis: PrepBasis },

    /// Measure a qubit, storing the result.
    Measure {
        qubit: QubitId,
        basis: MeasBasis,
        /// Where to store the result for later reference.
        result: ResultId,
    },

    /// Operations conditioned on a measurement result.
    Conditional {
        /// The measurement result to check.
        condition: ResultId,
        /// Operations to execute if result is 1 (true).
        if_one: Vec<AdaptedOp>,
        /// Operations to execute if result is 0 (false).
        if_zero: Vec<AdaptedOp>,
    },

    /// XOR a measurement result into another (classical operation).
    ///
    /// `target = target XOR source`
    XorResult { target: ResultId, source: ResultId },

    /// Output a result as the operation's return value.
    ///
    /// For operations like MZX that return a measurement outcome,
    /// this specifies which internal result to expose.
    OutputResult { result: ResultId },
}

impl AdaptedOp {
    /// Create a single-qubit gate with no angles.
    #[must_use]
    pub fn gate1(gate_id: GateId, qubit: QubitId) -> Self {
        Self::Gate {
            gate_id,
            qubits: smallvec::smallvec![qubit],
            angles: SmallVec::new(),
        }
    }

    /// Create a single-qubit rotation gate.
    #[must_use]
    pub fn rotation(gate_id: GateId, qubit: QubitId, angle: Angle64) -> Self {
        Self::Gate {
            gate_id,
            qubits: smallvec::smallvec![qubit],
            angles: smallvec::smallvec![angle],
        }
    }

    /// Create a two-qubit gate with no angles.
    #[must_use]
    pub fn gate2(gate_id: GateId, q0: QubitId, q1: QubitId) -> Self {
        Self::Gate {
            gate_id,
            qubits: smallvec::smallvec![q0, q1],
            angles: SmallVec::new(),
        }
    }

    /// Create a Z-basis preparation.
    #[must_use]
    pub fn pz(qubit: QubitId) -> Self {
        Self::Prep {
            qubit,
            basis: PrepBasis::Z,
        }
    }

    /// Create an X-basis preparation.
    #[must_use]
    pub fn px(qubit: QubitId) -> Self {
        Self::Prep {
            qubit,
            basis: PrepBasis::X,
        }
    }

    /// Create a Z-basis measurement.
    #[must_use]
    pub fn mz(qubit: QubitId, result: ResultId) -> Self {
        Self::Measure {
            qubit,
            basis: MeasBasis::Z,
            result,
        }
    }

    /// Create an X-basis measurement.
    #[must_use]
    pub fn mx(qubit: QubitId, result: ResultId) -> Self {
        Self::Measure {
            qubit,
            basis: MeasBasis::X,
            result,
        }
    }

    /// Create a conditional X gate (correction based on measurement).
    #[must_use]
    pub fn conditional_x(condition: ResultId, qubit: QubitId) -> Self {
        Self::Conditional {
            condition,
            if_one: vec![Self::gate1(gates::X, qubit)],
            if_zero: vec![],
        }
    }

    /// Create a conditional Z gate (correction based on measurement).
    #[must_use]
    pub fn conditional_z(condition: ResultId, qubit: QubitId) -> Self {
        Self::Conditional {
            condition,
            if_one: vec![Self::gate1(gates::Z, qubit)],
            if_zero: vec![],
        }
    }
}

/// Requirements for ancilla qubits in an adapted operation.
#[derive(Clone, Debug, Default)]
pub struct AncillaRequirements {
    /// Number of ancilla qubits needed.
    pub count: usize,
    /// Whether ancillas must start in |0⟩ state (true) or can be dirty (false).
    pub clean: bool,
}

impl AncillaRequirements {
    /// No ancillas required.
    #[must_use]
    pub fn none() -> Self {
        Self {
            count: 0,
            clean: false,
        }
    }

    /// Require N clean ancillas (initialized to |0⟩).
    #[must_use]
    pub fn clean(count: usize) -> Self {
        Self { count, clean: true }
    }

    /// Require N dirty ancillas (any state acceptable).
    #[must_use]
    pub fn dirty(count: usize) -> Self {
        Self {
            count,
            clean: false,
        }
    }
}

/// Result of adapting an operation.
#[derive(Clone, Debug)]
pub struct AdaptedSequence {
    /// The sequence of operations to execute.
    pub ops: Vec<AdaptedOp>,
    /// Number of result slots used (for result tracking).
    pub result_count: usize,
}

impl AdaptedSequence {
    /// Create a new adapted sequence.
    #[must_use]
    pub fn new(ops: Vec<AdaptedOp>) -> Self {
        // Count the maximum result ID used
        let result_count = Self::count_results(&ops);
        Self { ops, result_count }
    }

    fn count_results(ops: &[AdaptedOp]) -> usize {
        let mut max_id = 0usize;
        for op in ops {
            match op {
                AdaptedOp::Measure { result, .. } | AdaptedOp::OutputResult { result } => {
                    max_id = max_id.max(result.0 as usize + 1);
                }
                AdaptedOp::Conditional {
                    if_one, if_zero, ..
                } => {
                    max_id = max_id.max(Self::count_results(if_one));
                    max_id = max_id.max(Self::count_results(if_zero));
                }
                AdaptedOp::XorResult { target, source } => {
                    max_id = max_id.max(target.0 as usize + 1);
                    max_id = max_id.max(source.0 as usize + 1);
                }
                _ => {}
            }
        }
        max_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapted_op_gate_constructors() {
        let h = AdaptedOp::gate1(gates::H, QubitId(0));
        match h {
            AdaptedOp::Gate {
                gate_id,
                qubits,
                angles,
            } => {
                assert_eq!(gate_id, gates::H);
                assert_eq!(qubits.as_slice(), &[QubitId(0)]);
                assert!(angles.is_empty());
            }
            _ => panic!("Expected Gate"),
        }

        let cx = AdaptedOp::gate2(gates::CX, QubitId(0), QubitId(1));
        match cx {
            AdaptedOp::Gate {
                gate_id, qubits, ..
            } => {
                assert_eq!(gate_id, gates::CX);
                assert_eq!(qubits.as_slice(), &[QubitId(0), QubitId(1)]);
            }
            _ => panic!("Expected Gate"),
        }
    }

    #[test]
    fn test_adapted_op_prep_meas() {
        let prep = AdaptedOp::pz(QubitId(0));
        match prep {
            AdaptedOp::Prep { qubit, basis } => {
                assert_eq!(qubit, QubitId(0));
                assert_eq!(basis, PrepBasis::Z);
            }
            _ => panic!("Expected Prep"),
        }

        let meas = AdaptedOp::mz(QubitId(0), ResultId(0));
        match meas {
            AdaptedOp::Measure {
                qubit,
                basis,
                result,
            } => {
                assert_eq!(qubit, QubitId(0));
                assert_eq!(basis, MeasBasis::Z);
                assert_eq!(result, ResultId(0));
            }
            _ => panic!("Expected Measure"),
        }
    }

    #[test]
    fn test_ancilla_requirements() {
        let none = AncillaRequirements::none();
        assert_eq!(none.count, 0);

        let clean = AncillaRequirements::clean(2);
        assert_eq!(clean.count, 2);
        assert!(clean.clean);

        let dirty = AncillaRequirements::dirty(1);
        assert_eq!(dirty.count, 1);
        assert!(!dirty.clean);
    }

    #[test]
    fn test_adapted_sequence_result_count() {
        let ops = vec![
            AdaptedOp::pz(QubitId(0)),
            AdaptedOp::mz(QubitId(0), ResultId(0)),
            AdaptedOp::mz(QubitId(1), ResultId(1)),
        ];
        let seq = AdaptedSequence::new(ops);
        assert_eq!(seq.result_count, 2);
    }

    #[test]
    fn test_conditional_x() {
        let cond_x = AdaptedOp::conditional_x(ResultId(0), QubitId(1));
        match cond_x {
            AdaptedOp::Conditional {
                condition,
                if_one,
                if_zero,
            } => {
                assert_eq!(condition, ResultId(0));
                assert_eq!(if_one.len(), 1);
                assert!(if_zero.is_empty());
            }
            _ => panic!("Expected Conditional"),
        }
    }
}
