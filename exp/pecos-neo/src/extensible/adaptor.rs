//! Gate adaptors for decomposing unsupported gates into supported ones.
//!
//! When a simulator doesn't natively support a gate, an adaptor can
//! decompose it into a sequence of gates that the simulator does support.

use super::{GateId, GateSupportSet, gates};
use pecos_core::{Angle64, QubitId};
use smallvec::SmallVec;

/// A gate instance for adaptor output.
#[derive(Clone, Debug)]
pub struct AdaptedGate {
    /// The gate type
    pub gate_id: GateId,
    /// Target qubits
    pub qubits: SmallVec<[QubitId; 4]>,
    /// Angle parameters
    pub angles: SmallVec<[Angle64; 3]>,
}

impl AdaptedGate {
    /// Create a new adapted gate.
    #[must_use]
    pub fn new(gate_id: GateId, qubits: &[QubitId], angles: &[Angle64]) -> Self {
        Self {
            gate_id,
            qubits: qubits.iter().copied().collect(),
            angles: angles.iter().copied().collect(),
        }
    }

    /// Create a single-qubit gate with no angles.
    #[must_use]
    pub fn single(gate_id: GateId, qubit: QubitId) -> Self {
        Self::new(gate_id, &[qubit], &[])
    }

    /// Create a single-qubit rotation gate.
    #[must_use]
    pub fn rotation(gate_id: GateId, qubit: QubitId, angle: Angle64) -> Self {
        Self::new(gate_id, &[qubit], &[angle])
    }

    /// Create a two-qubit gate with no angles.
    #[must_use]
    pub fn two_qubit(gate_id: GateId, q0: QubitId, q1: QubitId) -> Self {
        Self::new(gate_id, &[q0, q1], &[])
    }
}

/// Trait for gate adaptors that decompose gates.
pub trait GateAdaptor: Send + Sync {
    /// Check if this adaptor can decompose the given gate.
    fn can_adapt(&self, gate_id: GateId) -> bool;

    /// Decompose a gate into a sequence of other gates.
    ///
    /// The adaptor receives the gate ID, qubits, and angles, and returns
    /// a sequence of gates that are equivalent to the original.
    fn adapt(&self, gate_id: GateId, qubits: &[QubitId], angles: &[Angle64]) -> Vec<AdaptedGate>;

    /// Get the set of gates this adaptor can decompose.
    fn adaptable_gates(&self) -> GateSupportSet;
}

/// Standard adaptor with common gate decompositions.
///
/// Decomposes gates into Clifford+RZ gate set.
pub struct StandardAdaptor {
    /// Gates that can be adapted
    can_adapt_bits: GateSupportSet,
}

impl Default for StandardAdaptor {
    fn default() -> Self {
        Self::stab_vec()
    }
}

impl StandardAdaptor {
    /// Create an adaptor targeting Clifford+RZ gate set.
    #[must_use]
    pub fn stab_vec() -> Self {
        let mut bits = GateSupportSet::new();

        // Gates we can decompose into Clifford+RZ
        bits.insert(gates::T);
        bits.insert(gates::Tdg);
        bits.insert(gates::RX);
        bits.insert(gates::RY);
        bits.insert(gates::SWAP);
        bits.insert(gates::RZZ);
        bits.insert(gates::RXX);
        bits.insert(gates::RYY);
        bits.insert(gates::CCX);

        Self {
            can_adapt_bits: bits,
        }
    }

    /// Create an empty adaptor (for adding custom decompositions).
    #[must_use]
    pub fn new() -> Self {
        Self {
            can_adapt_bits: GateSupportSet::new(),
        }
    }
}

impl GateAdaptor for StandardAdaptor {
    fn can_adapt(&self, gate_id: GateId) -> bool {
        self.can_adapt_bits.contains(gate_id)
    }

    fn adaptable_gates(&self) -> GateSupportSet {
        self.can_adapt_bits.clone()
    }

    fn adapt(&self, gate_id: GateId, qubits: &[QubitId], angles: &[Angle64]) -> Vec<AdaptedGate> {
        match gate_id {
            id if id == gates::T => {
                // T = RZ(π/4)
                let angle = Angle64::HALF_TURN / 4;
                vec![AdaptedGate::rotation(gates::RZ, qubits[0], angle)]
            }

            id if id == gates::Tdg => {
                // Tdg = RZ(-π/4)
                let angle = Angle64::ZERO - Angle64::HALF_TURN / 4;
                vec![AdaptedGate::rotation(gates::RZ, qubits[0], angle)]
            }

            id if id == gates::RX => {
                // RX(θ) = H RZ(θ) H
                let theta = angles[0];
                let q = qubits[0];
                vec![
                    AdaptedGate::single(gates::H, q),
                    AdaptedGate::rotation(gates::RZ, q, theta),
                    AdaptedGate::single(gates::H, q),
                ]
            }

            id if id == gates::RY => {
                // RY(θ) = SXdg RZ(θ) SX
                // Or equivalently: RZ(-π/2) RX(θ) RZ(π/2)
                let theta = angles[0];
                let q = qubits[0];
                let half_pi = Angle64::QUARTER_TURN;
                let neg_half_pi = Angle64::ZERO - Angle64::QUARTER_TURN;
                vec![
                    AdaptedGate::rotation(gates::RZ, q, neg_half_pi),
                    AdaptedGate::single(gates::H, q),
                    AdaptedGate::rotation(gates::RZ, q, theta),
                    AdaptedGate::single(gates::H, q),
                    AdaptedGate::rotation(gates::RZ, q, half_pi),
                ]
            }

            id if id == gates::SWAP => {
                // SWAP = CX(0,1) CX(1,0) CX(0,1)
                let (q0, q1) = (qubits[0], qubits[1]);
                vec![
                    AdaptedGate::two_qubit(gates::CX, q0, q1),
                    AdaptedGate::two_qubit(gates::CX, q1, q0),
                    AdaptedGate::two_qubit(gates::CX, q0, q1),
                ]
            }

            id if id == gates::RZZ => {
                // RZZ(θ) = CX(0,1) RZ(θ,1) CX(0,1)
                let theta = angles[0];
                let (q0, q1) = (qubits[0], qubits[1]);
                vec![
                    AdaptedGate::two_qubit(gates::CX, q0, q1),
                    AdaptedGate::rotation(gates::RZ, q1, theta),
                    AdaptedGate::two_qubit(gates::CX, q0, q1),
                ]
            }

            id if id == gates::RXX => {
                // RXX(θ) = H⊗H RZZ(θ) H⊗H
                let theta = angles[0];
                let (q0, q1) = (qubits[0], qubits[1]);
                vec![
                    AdaptedGate::single(gates::H, q0),
                    AdaptedGate::single(gates::H, q1),
                    AdaptedGate::two_qubit(gates::CX, q0, q1),
                    AdaptedGate::rotation(gates::RZ, q1, theta),
                    AdaptedGate::two_qubit(gates::CX, q0, q1),
                    AdaptedGate::single(gates::H, q0),
                    AdaptedGate::single(gates::H, q1),
                ]
            }

            id if id == gates::RYY => {
                // RYY(θ) = SX⊗SX RZZ(θ) SXdg⊗SXdg
                let theta = angles[0];
                let (q0, q1) = (qubits[0], qubits[1]);
                vec![
                    AdaptedGate::single(gates::SX, q0),
                    AdaptedGate::single(gates::SX, q1),
                    AdaptedGate::two_qubit(gates::CX, q0, q1),
                    AdaptedGate::rotation(gates::RZ, q1, theta),
                    AdaptedGate::two_qubit(gates::CX, q0, q1),
                    AdaptedGate::single(gates::SXdg, q0),
                    AdaptedGate::single(gates::SXdg, q1),
                ]
            }

            id if id == gates::CCX => {
                // CCX (Toffoli) decomposition into Clifford+T
                // This is a standard decomposition
                let (q0, q1, q2) = (qubits[0], qubits[1], qubits[2]);
                let t_angle = Angle64::HALF_TURN / 4;
                let tdg_angle = Angle64::ZERO - t_angle;

                vec![
                    AdaptedGate::single(gates::H, q2),
                    AdaptedGate::two_qubit(gates::CX, q1, q2),
                    AdaptedGate::rotation(gates::RZ, q2, tdg_angle),
                    AdaptedGate::two_qubit(gates::CX, q0, q2),
                    AdaptedGate::rotation(gates::RZ, q2, t_angle),
                    AdaptedGate::two_qubit(gates::CX, q1, q2),
                    AdaptedGate::rotation(gates::RZ, q2, tdg_angle),
                    AdaptedGate::two_qubit(gates::CX, q0, q2),
                    AdaptedGate::rotation(gates::RZ, q1, t_angle),
                    AdaptedGate::rotation(gates::RZ, q2, t_angle),
                    AdaptedGate::single(gates::H, q2),
                    AdaptedGate::two_qubit(gates::CX, q0, q1),
                    AdaptedGate::rotation(gates::RZ, q0, t_angle),
                    AdaptedGate::rotation(gates::RZ, q1, tdg_angle),
                    AdaptedGate::two_qubit(gates::CX, q0, q1),
                ]
            }

            _ => {
                panic!("StandardAdaptor cannot adapt gate {gate_id:?}");
            }
        }
    }
}

/// Composite adaptor that chains multiple adaptors.
pub struct CompositeAdaptor {
    adaptors: Vec<Box<dyn GateAdaptor>>,
}

impl CompositeAdaptor {
    /// Create a new composite adaptor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            adaptors: Vec::new(),
        }
    }

    /// Add an adaptor to the chain.
    pub fn add<A: GateAdaptor + 'static>(&mut self, adaptor: A) {
        self.adaptors.push(Box::new(adaptor));
    }

    /// Builder pattern for adding adaptors.
    #[must_use]
    pub fn with<A: GateAdaptor + 'static>(mut self, adaptor: A) -> Self {
        self.add(adaptor);
        self
    }
}

impl Default for CompositeAdaptor {
    fn default() -> Self {
        Self::new()
    }
}

impl GateAdaptor for CompositeAdaptor {
    fn can_adapt(&self, gate_id: GateId) -> bool {
        self.adaptors.iter().any(|a| a.can_adapt(gate_id))
    }

    fn adaptable_gates(&self) -> GateSupportSet {
        let mut result = GateSupportSet::new();
        for adaptor in &self.adaptors {
            result.union_with(&adaptor.adaptable_gates());
        }
        result
    }

    fn adapt(&self, gate_id: GateId, qubits: &[QubitId], angles: &[Angle64]) -> Vec<AdaptedGate> {
        for adaptor in &self.adaptors {
            if adaptor.can_adapt(gate_id) {
                return adaptor.adapt(gate_id, qubits, angles);
            }
        }
        panic!("CompositeAdaptor cannot adapt gate {gate_id:?}");
    }
}

/// Decomposition function type for custom adaptors.
type DecomposeFn = Box<dyn Fn(&[QubitId], &[Angle64]) -> Vec<AdaptedGate> + Send + Sync>;

/// Custom adaptor for user-defined decompositions.
pub struct CustomAdaptor {
    /// Gate ID this adaptor handles
    gate_id: GateId,
    /// Decomposition function
    decompose: DecomposeFn,
}

impl CustomAdaptor {
    /// Create a new custom adaptor.
    pub fn new<F>(gate_id: GateId, decompose: F) -> Self
    where
        F: Fn(&[QubitId], &[Angle64]) -> Vec<AdaptedGate> + Send + Sync + 'static,
    {
        Self {
            gate_id,
            decompose: Box::new(decompose),
        }
    }
}

impl GateAdaptor for CustomAdaptor {
    fn can_adapt(&self, gate_id: GateId) -> bool {
        gate_id == self.gate_id
    }

    fn adaptable_gates(&self) -> GateSupportSet {
        let mut set = GateSupportSet::new();
        set.insert(self.gate_id);
        set
    }

    fn adapt(&self, _gate_id: GateId, qubits: &[QubitId], angles: &[Angle64]) -> Vec<AdaptedGate> {
        (self.decompose)(qubits, angles)
    }
}

// ============================================================================
// Bridge to Extended Adaptor System
// ============================================================================

use super::operation::{AdaptedOp, AdaptedSequence, AncillaRequirements};
use super::stabilizer_adaptor::ExtendedAdaptor;

impl AdaptedGate {
    /// Convert to an `AdaptedOp::Gate`.
    #[must_use]
    pub fn to_op(&self) -> AdaptedOp {
        AdaptedOp::Gate {
            gate_id: self.gate_id,
            qubits: self.qubits.clone(),
            angles: self.angles.clone(),
        }
    }
}

impl From<AdaptedGate> for AdaptedOp {
    fn from(gate: AdaptedGate) -> Self {
        gate.to_op()
    }
}

/// Wrapper that lifts a `GateAdaptor` to an `ExtendedAdaptor`.
///
/// This allows using existing gate decomposition adaptors (like `StandardAdaptor`)
/// within the extended adaptor framework that supports measurements and conditionals.
pub struct LiftedAdaptor<A: GateAdaptor> {
    inner: A,
}

impl<A: GateAdaptor> LiftedAdaptor<A> {
    /// Wrap a gate adaptor as an extended adaptor.
    #[must_use]
    pub fn new(adaptor: A) -> Self {
        Self { inner: adaptor }
    }

    /// Get a reference to the inner adaptor.
    #[must_use]
    pub fn inner(&self) -> &A {
        &self.inner
    }
}

impl<A: GateAdaptor + 'static> ExtendedAdaptor for LiftedAdaptor<A> {
    fn can_adapt(&self, gate_id: GateId) -> bool {
        self.inner.can_adapt(gate_id)
    }

    fn ancilla_requirements(&self, _gate_id: GateId) -> AncillaRequirements {
        // Pure unitary decompositions don't need ancillas
        AncillaRequirements::none()
    }

    fn adaptable_gates(&self) -> GateSupportSet {
        self.inner.adaptable_gates()
    }

    fn adapt(
        &self,
        gate_id: GateId,
        qubits: &[QubitId],
        angles: &[Angle64],
        _ancillas: &[QubitId],
    ) -> AdaptedSequence {
        let gates = self.inner.adapt(gate_id, qubits, angles);
        let ops: Vec<AdaptedOp> = gates.into_iter().map(Into::into).collect();
        AdaptedSequence::new(ops)
    }
}

/// Composite extended adaptor that chains multiple extended adaptors.
///
/// This can combine gate adaptors (via `LiftedAdaptor`) with stabilizer
/// adaptors and other extended adaptors.
pub struct CompositeExtendedAdaptor {
    adaptors: Vec<Box<dyn ExtendedAdaptor>>,
}

impl Default for CompositeExtendedAdaptor {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositeExtendedAdaptor {
    /// Create a new empty composite adaptor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            adaptors: Vec::new(),
        }
    }

    /// Add an extended adaptor.
    pub fn add<A: ExtendedAdaptor + 'static>(&mut self, adaptor: A) {
        self.adaptors.push(Box::new(adaptor));
    }

    /// Add a gate adaptor (automatically lifted).
    pub fn add_gate_adaptor<A: GateAdaptor + 'static>(&mut self, adaptor: A) {
        self.adaptors.push(Box::new(LiftedAdaptor::new(adaptor)));
    }

    /// Builder pattern: add an extended adaptor.
    #[must_use]
    pub fn with<A: ExtendedAdaptor + 'static>(mut self, adaptor: A) -> Self {
        self.add(adaptor);
        self
    }

    /// Builder pattern: add a gate adaptor (automatically lifted).
    #[must_use]
    pub fn with_gate_adaptor<A: GateAdaptor + 'static>(mut self, adaptor: A) -> Self {
        self.add_gate_adaptor(adaptor);
        self
    }

    /// Create a standard composite with common adaptors.
    ///
    /// Includes:
    /// - `StandardAdaptor` for gate decompositions (T, SWAP, CCX, etc.)
    /// - `StabilizerAdaptor` for joint measurements/preparations
    #[must_use]
    pub fn standard() -> Self {
        use super::stabilizer_adaptor::StabilizerAdaptor;

        Self::new()
            .with_gate_adaptor(StandardAdaptor::stab_vec())
            .with(StabilizerAdaptor::new())
    }
}

impl ExtendedAdaptor for CompositeExtendedAdaptor {
    fn can_adapt(&self, gate_id: GateId) -> bool {
        self.adaptors.iter().any(|a| a.can_adapt(gate_id))
    }

    fn ancilla_requirements(&self, gate_id: GateId) -> AncillaRequirements {
        for adaptor in &self.adaptors {
            if adaptor.can_adapt(gate_id) {
                return adaptor.ancilla_requirements(gate_id);
            }
        }
        AncillaRequirements::none()
    }

    fn adaptable_gates(&self) -> GateSupportSet {
        let mut result = GateSupportSet::new();
        for adaptor in &self.adaptors {
            result.union_with(&adaptor.adaptable_gates());
        }
        result
    }

    fn adapt(
        &self,
        gate_id: GateId,
        qubits: &[QubitId],
        angles: &[Angle64],
        ancillas: &[QubitId],
    ) -> AdaptedSequence {
        for adaptor in &self.adaptors {
            if adaptor.can_adapt(gate_id) {
                return adaptor.adapt(gate_id, qubits, angles, ancillas);
            }
        }
        panic!("CompositeExtendedAdaptor cannot adapt gate {gate_id:?}");
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;
    use crate::extensible::stabilizer_adaptor::stabilizer_gates;

    #[test]
    fn test_adapted_gate_to_op() {
        let gate = AdaptedGate::single(gates::H, QubitId(0));
        let op = gate.to_op();

        match op {
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
    }

    #[test]
    fn test_lifted_adaptor() {
        let standard = StandardAdaptor::stab_vec();
        let lifted = LiftedAdaptor::new(standard);

        assert!(lifted.can_adapt(gates::T));
        assert!(lifted.can_adapt(gates::SWAP));

        let reqs = lifted.ancilla_requirements(gates::T);
        assert_eq!(reqs.count, 0); // Pure unitary, no ancillas

        let seq = lifted.adapt(gates::T, &[QubitId(0)], &[], &[]);
        assert!(!seq.ops.is_empty());

        // T should decompose to RZ(pi/4)
        match &seq.ops[0] {
            AdaptedOp::Gate { gate_id, .. } => {
                assert_eq!(*gate_id, gates::RZ);
            }
            _ => panic!("Expected Gate"),
        }
    }

    #[test]
    fn test_composite_extended_adaptor() {
        let composite = CompositeExtendedAdaptor::standard();

        // Should handle gate decompositions
        assert!(composite.can_adapt(gates::T));
        assert!(composite.can_adapt(gates::SWAP));
        assert!(composite.can_adapt(gates::CCX));

        // Should handle stabilizer operations
        assert!(composite.can_adapt(stabilizer_gates::MZX));
        assert!(composite.can_adapt(stabilizer_gates::PZX));

        // Gate decomposition (no ancillas)
        let t_reqs = composite.ancilla_requirements(gates::T);
        assert_eq!(t_reqs.count, 0);

        // Stabilizer measurement (needs ancilla)
        let mzx_reqs = composite.ancilla_requirements(stabilizer_gates::MZX);
        assert_eq!(mzx_reqs.count, 1);
    }

    #[test]
    fn test_swap_decomposition_via_lifted() {
        let lifted = LiftedAdaptor::new(StandardAdaptor::stab_vec());

        let seq = lifted.adapt(gates::SWAP, &[QubitId(0), QubitId(1)], &[], &[]);

        // SWAP = CX CX CX (3 gates)
        assert_eq!(seq.ops.len(), 3);

        for op in &seq.ops {
            match op {
                AdaptedOp::Gate { gate_id, .. } => {
                    assert_eq!(*gate_id, gates::CX);
                }
                _ => panic!("Expected Gate"),
            }
        }
    }
}
