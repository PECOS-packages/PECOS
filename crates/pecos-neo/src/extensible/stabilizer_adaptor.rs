//! Stabilizer measurement and preparation adaptors.
//!
//! This module provides adaptors for joint stabilizer operations like:
//! - `MZX`: Measure the Z⊗X stabilizer
//! - `PZX`: Prepare a +1 eigenstate of Z⊗X
//!
//! These operations use ancilla qubits and decompose into primitive operations.

use super::operation::{AdaptedOp, AdaptedSequence, AncillaRequirements, PrepBasis, ResultId};
use super::{GateId, GateSupportSet, gates};
use pecos_core::{Angle64, QubitId};

/// Trait for extended adaptors that can produce operation sequences
/// including measurements, preparations, and conditionals.
pub trait ExtendedAdaptor: Send + Sync {
    /// Check if this adaptor can handle the given gate.
    fn can_adapt(&self, gate_id: GateId) -> bool;

    /// Get ancilla requirements for adapting this gate.
    fn ancilla_requirements(&self, gate_id: GateId) -> AncillaRequirements;

    /// Adapt a gate into an operation sequence.
    ///
    /// # Arguments
    /// * `gate_id` - The gate to adapt
    /// * `qubits` - Target qubits for the operation
    /// * `angles` - Angle parameters (if any)
    /// * `ancillas` - Ancilla qubits provided by the caller
    ///
    /// # Returns
    /// A sequence of operations equivalent to the original gate.
    fn adapt(
        &self,
        gate_id: GateId,
        qubits: &[QubitId],
        angles: &[Angle64],
        ancillas: &[QubitId],
    ) -> AdaptedSequence;

    /// Get the set of gates this adaptor can handle.
    fn adaptable_gates(&self) -> GateSupportSet;
}

/// Extended gate IDs for stabilizer operations.
///
/// These are in the user-defined range (>= 256).
#[allow(non_upper_case_globals)]
pub mod stabilizer_gates {
    use super::GateId;

    // Joint measurements (2-qubit stabilizers)
    pub const MZZ: GateId = GateId(256);
    pub const MXX: GateId = GateId(257);
    pub const MYY: GateId = GateId(258);
    pub const MZX: GateId = GateId(259);
    pub const MXZ: GateId = GateId(260);
    pub const MZY: GateId = GateId(261);
    pub const MYZ: GateId = GateId(262);
    pub const MXY: GateId = GateId(263);
    pub const MYX: GateId = GateId(264);

    // Joint preparations (2-qubit stabilizer eigenstates)
    pub const PZZ: GateId = GateId(280);
    pub const PXX: GateId = GateId(281);
    pub const PYY: GateId = GateId(282);
    pub const PZX: GateId = GateId(283);
    pub const PXZ: GateId = GateId(284);
    pub const PZY: GateId = GateId(285);
    pub const PYZ: GateId = GateId(286);
    pub const PXY: GateId = GateId(287);
    pub const PYX: GateId = GateId(288);
}

/// Adaptor for 2-qubit stabilizer measurements.
///
/// Decomposes joint measurements like MZX into:
/// - Ancilla preparation
/// - Entangling gates
/// - Ancilla measurement
pub struct StabilizerMeasurementAdaptor {
    adaptable: GateSupportSet,
}

impl Default for StabilizerMeasurementAdaptor {
    fn default() -> Self {
        Self::new()
    }
}

impl StabilizerMeasurementAdaptor {
    /// Create a new stabilizer measurement adaptor.
    #[must_use]
    pub fn new() -> Self {
        use stabilizer_gates::{MXX, MXY, MXZ, MYX, MYY, MYZ, MZX, MZY, MZZ};

        let mut adaptable = GateSupportSet::new();
        adaptable.insert(MZZ);
        adaptable.insert(MXX);
        adaptable.insert(MYY);
        adaptable.insert(MZX);
        adaptable.insert(MXZ);
        adaptable.insert(MZY);
        adaptable.insert(MYZ);
        adaptable.insert(MXY);
        adaptable.insert(MYX);

        Self { adaptable }
    }

    /// Decompose a single-Pauli measurement coupling.
    ///
    /// Returns the gates needed to couple a data qubit to an ancilla
    /// for measuring in the given basis.
    fn coupling_for_basis(basis: char, data: QubitId, ancilla: QubitId) -> Vec<AdaptedOp> {
        match basis {
            'Z' => {
                // Z measurement: CX from data to ancilla
                vec![AdaptedOp::gate2(gates::CX, data, ancilla)]
            }
            'X' => {
                // X measurement: H, CX, H (or equivalently CZ after H on data)
                vec![
                    AdaptedOp::gate1(gates::H, data),
                    AdaptedOp::gate2(gates::CX, data, ancilla),
                    AdaptedOp::gate1(gates::H, data),
                ]
            }
            'Y' => {
                // Y measurement: SXdg, CX, SX
                vec![
                    AdaptedOp::gate1(gates::SXdg, data),
                    AdaptedOp::gate2(gates::CX, data, ancilla),
                    AdaptedOp::gate1(gates::SX, data),
                ]
            }
            _ => panic!("Invalid basis: {basis}"),
        }
    }

    /// Get the two Pauli bases for a stabilizer gate ID.
    fn bases_for_gate(gate_id: GateId) -> (char, char) {
        use stabilizer_gates::{MXX, MXY, MXZ, MYX, MYY, MYZ, MZX, MZY, MZZ};
        match gate_id {
            id if id == MZZ => ('Z', 'Z'),
            id if id == MXX => ('X', 'X'),
            id if id == MYY => ('Y', 'Y'),
            id if id == MZX => ('Z', 'X'),
            id if id == MXZ => ('X', 'Z'),
            id if id == MZY => ('Z', 'Y'),
            id if id == MYZ => ('Y', 'Z'),
            id if id == MXY => ('X', 'Y'),
            id if id == MYX => ('Y', 'X'),
            _ => panic!("Unknown stabilizer measurement gate: {gate_id:?}"),
        }
    }
}

impl ExtendedAdaptor for StabilizerMeasurementAdaptor {
    fn can_adapt(&self, gate_id: GateId) -> bool {
        self.adaptable.contains(gate_id)
    }

    fn ancilla_requirements(&self, gate_id: GateId) -> AncillaRequirements {
        if self.adaptable.contains(gate_id) {
            AncillaRequirements::clean(1)
        } else {
            AncillaRequirements::none()
        }
    }

    fn adaptable_gates(&self) -> GateSupportSet {
        self.adaptable.clone()
    }

    fn adapt(
        &self,
        gate_id: GateId,
        qubits: &[QubitId],
        _angles: &[Angle64],
        ancillas: &[QubitId],
    ) -> AdaptedSequence {
        assert!(
            qubits.len() >= 2,
            "Stabilizer measurement requires 2 data qubits"
        );
        assert!(
            !ancillas.is_empty(),
            "Stabilizer measurement requires 1 ancilla"
        );

        let (q0, q1) = (qubits[0], qubits[1]);
        let ancilla = ancillas[0];
        let (basis0, basis1) = Self::bases_for_gate(gate_id);

        let mut ops = Vec::new();

        // 1. Prepare ancilla in |0⟩
        ops.push(AdaptedOp::pz(ancilla));

        // 2. Couple first data qubit to ancilla
        ops.extend(Self::coupling_for_basis(basis0, q0, ancilla));

        // 3. Couple second data qubit to ancilla
        ops.extend(Self::coupling_for_basis(basis1, q1, ancilla));

        // 4. Measure ancilla
        ops.push(AdaptedOp::mz(ancilla, ResultId(0)));

        // 5. Output the result
        ops.push(AdaptedOp::OutputResult {
            result: ResultId(0),
        });

        AdaptedSequence::new(ops)
    }
}

/// Adaptor for 2-qubit stabilizer preparations.
///
/// Prepares a +1 eigenstate of the given 2-qubit Pauli operator.
pub struct StabilizerPreparationAdaptor {
    adaptable: GateSupportSet,
}

impl Default for StabilizerPreparationAdaptor {
    fn default() -> Self {
        Self::new()
    }
}

impl StabilizerPreparationAdaptor {
    /// Create a new stabilizer preparation adaptor.
    #[must_use]
    pub fn new() -> Self {
        use stabilizer_gates::{PXX, PXY, PXZ, PYX, PYY, PYZ, PZX, PZY, PZZ};

        let mut adaptable = GateSupportSet::new();
        adaptable.insert(PZZ);
        adaptable.insert(PXX);
        adaptable.insert(PYY);
        adaptable.insert(PZX);
        adaptable.insert(PXZ);
        adaptable.insert(PZY);
        adaptable.insert(PYZ);
        adaptable.insert(PXY);
        adaptable.insert(PYX);

        Self { adaptable }
    }

    /// Get the two Pauli bases for a stabilizer gate ID.
    fn bases_for_gate(gate_id: GateId) -> (char, char) {
        use stabilizer_gates::{PXX, PXY, PXZ, PYX, PYY, PYZ, PZX, PZY, PZZ};
        match gate_id {
            id if id == PZZ => ('Z', 'Z'),
            id if id == PXX => ('X', 'X'),
            id if id == PYY => ('Y', 'Y'),
            id if id == PZX => ('Z', 'X'),
            id if id == PXZ => ('X', 'Z'),
            id if id == PZY => ('Z', 'Y'),
            id if id == PYZ => ('Y', 'Z'),
            id if id == PXY => ('X', 'Y'),
            id if id == PYX => ('Y', 'X'),
            _ => panic!("Unknown stabilizer preparation gate: {gate_id:?}"),
        }
    }

    /// Get the preparation basis for a single Pauli.
    fn prep_basis_for_pauli(pauli: char) -> PrepBasis {
        match pauli {
            'Z' => PrepBasis::Z, // |0⟩ is +1 eigenstate of Z
            'X' => PrepBasis::X, // |+⟩ is +1 eigenstate of X
            'Y' => PrepBasis::Y, // |+i⟩ is +1 eigenstate of Y
            _ => panic!("Invalid Pauli: {pauli}"),
        }
    }
}

impl ExtendedAdaptor for StabilizerPreparationAdaptor {
    fn can_adapt(&self, gate_id: GateId) -> bool {
        self.adaptable.contains(gate_id)
    }

    fn ancilla_requirements(&self, _gate_id: GateId) -> AncillaRequirements {
        // Preparations don't need ancillas
        AncillaRequirements::none()
    }

    fn adaptable_gates(&self) -> GateSupportSet {
        self.adaptable.clone()
    }

    fn adapt(
        &self,
        gate_id: GateId,
        qubits: &[QubitId],
        _angles: &[Angle64],
        _ancillas: &[QubitId],
    ) -> AdaptedSequence {
        assert!(
            qubits.len() >= 2,
            "Stabilizer preparation requires 2 qubits"
        );

        let (q0, q1) = (qubits[0], qubits[1]);
        let (basis0, basis1) = Self::bases_for_gate(gate_id);

        // Prepare each qubit in its respective +1 eigenstate.
        // This gives a +1 eigenstate of the joint operator.
        // For example, PZX prepares |0⟩|+⟩ which is +1 eigenstate of Z⊗X.
        let ops = vec![
            AdaptedOp::Prep {
                qubit: q0,
                basis: Self::prep_basis_for_pauli(basis0),
            },
            AdaptedOp::Prep {
                qubit: q1,
                basis: Self::prep_basis_for_pauli(basis1),
            },
        ];

        AdaptedSequence::new(ops)
    }
}

/// Combined adaptor for all stabilizer operations.
pub struct StabilizerAdaptor {
    measurement: StabilizerMeasurementAdaptor,
    preparation: StabilizerPreparationAdaptor,
}

impl Default for StabilizerAdaptor {
    fn default() -> Self {
        Self::new()
    }
}

impl StabilizerAdaptor {
    /// Create a new combined stabilizer adaptor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            measurement: StabilizerMeasurementAdaptor::new(),
            preparation: StabilizerPreparationAdaptor::new(),
        }
    }
}

impl ExtendedAdaptor for StabilizerAdaptor {
    fn can_adapt(&self, gate_id: GateId) -> bool {
        self.measurement.can_adapt(gate_id) || self.preparation.can_adapt(gate_id)
    }

    fn ancilla_requirements(&self, gate_id: GateId) -> AncillaRequirements {
        if self.measurement.can_adapt(gate_id) {
            self.measurement.ancilla_requirements(gate_id)
        } else if self.preparation.can_adapt(gate_id) {
            self.preparation.ancilla_requirements(gate_id)
        } else {
            AncillaRequirements::none()
        }
    }

    fn adaptable_gates(&self) -> GateSupportSet {
        let mut gates = self.measurement.adaptable_gates();
        gates.union_with(&self.preparation.adaptable_gates());
        gates
    }

    fn adapt(
        &self,
        gate_id: GateId,
        qubits: &[QubitId],
        angles: &[Angle64],
        ancillas: &[QubitId],
    ) -> AdaptedSequence {
        if self.measurement.can_adapt(gate_id) {
            self.measurement.adapt(gate_id, qubits, angles, ancillas)
        } else if self.preparation.can_adapt(gate_id) {
            self.preparation.adapt(gate_id, qubits, angles, ancillas)
        } else {
            panic!("StabilizerAdaptor cannot adapt gate {gate_id:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mzx_decomposition() {
        let adaptor = StabilizerMeasurementAdaptor::new();

        assert!(adaptor.can_adapt(stabilizer_gates::MZX));

        let reqs = adaptor.ancilla_requirements(stabilizer_gates::MZX);
        assert_eq!(reqs.count, 1);
        assert!(reqs.clean);

        let seq = adaptor.adapt(
            stabilizer_gates::MZX,
            &[QubitId(0), QubitId(1)],
            &[],
            &[QubitId(2)], // ancilla
        );

        // Should have: prep, CX (Z coupling), H, CX, H (X coupling), measure, output
        assert!(seq.ops.len() >= 6);
        assert_eq!(seq.result_count, 1);

        // First op should be prep
        assert!(matches!(
            seq.ops[0],
            AdaptedOp::Prep {
                qubit: QubitId(2),
                basis: PrepBasis::Z
            }
        ));

        // Last op should be output
        assert!(matches!(
            seq.ops.last(),
            Some(AdaptedOp::OutputResult {
                result: ResultId(0)
            })
        ));
    }

    #[test]
    fn test_pzx_decomposition() {
        let adaptor = StabilizerPreparationAdaptor::new();

        assert!(adaptor.can_adapt(stabilizer_gates::PZX));

        let reqs = adaptor.ancilla_requirements(stabilizer_gates::PZX);
        assert_eq!(reqs.count, 0);

        let seq = adaptor.adapt(stabilizer_gates::PZX, &[QubitId(0), QubitId(1)], &[], &[]);

        // Should have 2 preparations
        assert_eq!(seq.ops.len(), 2);

        // First qubit prepared in Z basis (|0⟩)
        assert!(matches!(
            seq.ops[0],
            AdaptedOp::Prep {
                qubit: QubitId(0),
                basis: PrepBasis::Z
            }
        ));

        // Second qubit prepared in X basis (|+⟩)
        assert!(matches!(
            seq.ops[1],
            AdaptedOp::Prep {
                qubit: QubitId(1),
                basis: PrepBasis::X
            }
        ));
    }

    #[test]
    fn test_mzz_decomposition() {
        let adaptor = StabilizerMeasurementAdaptor::new();

        let seq = adaptor.adapt(
            stabilizer_gates::MZZ,
            &[QubitId(0), QubitId(1)],
            &[],
            &[QubitId(2)],
        );

        // MZZ should have: prep, CX, CX, measure, output (simpler than MZX)
        assert!(seq.ops.len() >= 4);
    }

    #[test]
    fn test_combined_adaptor() {
        let adaptor = StabilizerAdaptor::new();

        assert!(adaptor.can_adapt(stabilizer_gates::MZX));
        assert!(adaptor.can_adapt(stabilizer_gates::PZX));
        assert!(!adaptor.can_adapt(gates::H)); // Core gates not handled
    }

    #[test]
    fn test_all_stabilizer_measurements() {
        use stabilizer_gates::*;
        let adaptor = StabilizerMeasurementAdaptor::new();

        for gate in [MZZ, MXX, MYY, MZX, MXZ, MZY, MYZ, MXY, MYX] {
            assert!(adaptor.can_adapt(gate), "Should handle {gate:?}");

            let seq = adaptor.adapt(gate, &[QubitId(0), QubitId(1)], &[], &[QubitId(2)]);

            assert!(!seq.ops.is_empty());
            assert!(seq.result_count >= 1);
        }
    }

    #[test]
    fn test_all_stabilizer_preparations() {
        use stabilizer_gates::*;
        let adaptor = StabilizerPreparationAdaptor::new();

        for gate in [PZZ, PXX, PYY, PZX, PXZ, PZY, PYZ, PXY, PYX] {
            assert!(adaptor.can_adapt(gate), "Should handle {gate:?}");

            let seq = adaptor.adapt(gate, &[QubitId(0), QubitId(1)], &[], &[]);

            assert_eq!(seq.ops.len(), 2); // Two preparations
        }
    }
}
