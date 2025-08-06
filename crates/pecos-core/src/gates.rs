//! Gate command representation for quantum operations
//!
//! This module provides the `GateCommand` struct which represents a quantum
//! gate operation with its type, qubits, and parameters.

use crate::QubitId;
use crate::gate_type::GateType;

/// Flat gate command representation for quantum operations
///
/// This struct provides a clean, flat representation of quantum gate commands
/// without unnecessary nesting. It serves as the primary interface for gate
/// operations in the `ByteMessage` system.
///
/// # Design
/// - Uses `QubitId` for type-safe qubit representation
/// - Flat structure for easy access to gate data
/// - Compatible with binary protocol serialization
#[derive(Debug, Clone, PartialEq)]
pub struct Gate {
    /// The type of the gate
    pub gate_type: GateType,
    /// Optional parameters for parameterized gates
    pub params: Vec<f64>,
    /// The qubits the gate acts on
    pub qubits: Vec<QubitId>,
}

/// Legacy quantum gate representation for `ByteMessageBuilder` compatibility
///
/// This struct is designed to replace `QuantumCommand` with a more FFI-friendly
/// representation. It contains all the information needed to represent a quantum
/// gate operation.
///
impl Gate {
    /// Create a new gate command
    #[must_use]
    pub fn new(gate_type: GateType, params: Vec<f64>, qubits: Vec<impl Into<QubitId>>) -> Self {
        Self {
            gate_type,
            params,
            qubits: qubits.into_iter().map(Into::into).collect(),
        }
    }

    /// Total number of qubits being gated
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// The number of individual gates represented by this `Gate`
    #[inline]
    #[must_use]
    pub fn num_gates(&self) -> usize {
        self.num_qubits() / self.quantum_arity()
    }

    /// Helper function to flatten qubit pairs into a vector of `QubitId`s
    fn flatten_qubit_pairs(
        qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> Vec<QubitId> {
        qubit_pairs
            .iter()
            .flat_map(|&(q1, q2)| [q1.into(), q2.into()])
            .collect()
    }

    /// Create X gate on multiple qubits
    #[must_use]
    pub fn x(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::new(
            GateType::X,
            vec![],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create Y gate on multiple qubits
    #[must_use]
    pub fn y(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::new(
            GateType::Y,
            vec![],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create Z gate on multiple qubits
    #[must_use]
    pub fn z(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::new(
            GateType::Z,
            vec![],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create H gate on multiple qubits
    #[must_use]
    pub fn h(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::new(
            GateType::H,
            vec![],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create CX gate from flat qubit list (control1, target1, control2, target2, ...)
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even, as `CX` gates require pairs of qubits.
    #[must_use]
    pub fn cx_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len() % 2 == 0,
            "CX gate requires an even number of qubits"
        );
        Self::new(
            GateType::CX,
            vec![],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create CX gate on multiple qubit pairs
    #[must_use]
    pub fn cx(qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::cx_vec(&flat_qubits)
    }

    /// Create SZZ gate from flat qubit list (`qubit1_1`, `qubit2_1`, `qubit1_2`, `qubit2_2`, ...)
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even, as `SZZ` gates require pairs of qubits.
    #[must_use]
    pub fn szz_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len() % 2 == 0,
            "SZZ gate requires an even number of qubits"
        );
        Self::new(
            GateType::SZZ,
            vec![],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create SZZ gate on multiple qubit pairs
    #[must_use]
    pub fn szz(qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::szz_vec(&flat_qubits)
    }

    /// Create `SZZdg` gate from flat qubit list (`qubit1_1`, `qubit2_1`, `qubit1_2`, `qubit2_2`, ...)
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even, as `SZZdg` gates require pairs of qubits.
    #[must_use]
    pub fn szzdg_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len() % 2 == 0,
            "SZZdg gate requires an even number of qubits"
        );
        Self::new(
            GateType::SZZdg,
            vec![],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create `SZZdg` gate on multiple qubit pairs
    #[must_use]
    pub fn szzdg(qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::szzdg_vec(&flat_qubits)
    }

    /// Create RZZ gate from flat qubit list (`qubit1_1`, `qubit2_1`, `qubit1_2`, `qubit2_2`, ...)
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even, as `RZZ` gates require pairs of qubits.
    #[must_use]
    pub fn rzz_vec(theta: f64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len() % 2 == 0,
            "RZZ gate requires an even number of qubits"
        );
        Self::new(
            GateType::RZZ,
            vec![theta],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create RZZ gate on multiple qubit pairs
    #[must_use]
    pub fn rzz(
        theta: f64,
        qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::rzz_vec(theta, &flat_qubits)
    }

    /// Create RZ gate on multiple qubits
    #[must_use]
    pub fn rz(theta: f64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::new(
            GateType::RZ,
            vec![theta],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create R1XY gate on multiple qubits
    #[must_use]
    pub fn r1xy(theta: f64, phi: f64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::new(
            GateType::R1XY,
            vec![theta, phi],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create U gate on multiple qubits
    #[must_use]
    pub fn u(theta: f64, phi: f64, lambda: f64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::new(
            GateType::U,
            vec![theta, phi, lambda],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create Measure gate on multiple qubits
    #[must_use]
    pub fn measure(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::new(
            GateType::Measure,
            vec![],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create `MeasureLeaked` gate on multiple qubits
    #[must_use]
    pub fn measure_leaked(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::new(
            GateType::MeasureLeaked,
            vec![],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create Prep gate on multiple qubits
    #[must_use]
    pub fn prep(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::new(
            GateType::Prep,
            vec![],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create a new Idle gate for qubits idling for a specific duration
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the idle period in seconds
    /// * `qubits` - The qubits that are idling
    ///
    /// # Returns
    ///
    /// A new Idle gate with the specified parameters
    #[must_use]
    pub fn idle(duration: f64, qubits: Vec<QubitId>) -> Self {
        Self::new(GateType::Idle, vec![duration], qubits)
    }

    /// Returns the duration of an idle gate, or 0.0 if not an idle gate
    #[must_use]
    pub fn idle_duration(&self) -> f64 {
        if self.gate_type == GateType::Idle && !self.params.is_empty() {
            self.params[0]
        } else {
            0.0
        }
    }

    /// Returns the number of angle parameters this gate requires
    ///
    /// # Returns
    ///
    /// The number of floating-point angle parameters needed for this gate type
    #[inline]
    #[must_use]
    pub fn classical_arity(&self) -> usize {
        self.gate_type.classical_arity()
    }

    /// Returns the number of qubits this gate operates on
    ///
    /// # Returns
    ///
    /// The number of qubits this gate type requires (1 or 2)
    #[inline]
    #[must_use]
    pub fn quantum_arity(&self) -> usize {
        self.gate_type.quantum_arity()
    }

    /// Returns whether this gate requires angle parameters
    #[inline]
    #[must_use]
    pub fn is_parameterized(&self) -> bool {
        self.gate_type.is_parameterized()
    }

    /// Returns whether this gate operates on a single qubit
    #[inline]
    #[must_use]
    pub fn is_single_qubit(&self) -> bool {
        self.gate_type.is_single_qubit()
    }

    /// Returns whether this gate operates on two qubits
    #[inline]
    #[must_use]
    pub fn is_two_qubit(&self) -> bool {
        self.gate_type.is_two_qubit()
    }

    /// Validates that this gate has the correct number of parameters and qubits
    ///
    /// # Returns
    ///
    /// `Ok(())` if the gate is valid, or an error message describing the issue
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The number of parameters doesn't match the gate's classical arity
    /// - The number of qubits is not a multiple of the gate's quantum arity
    pub fn validate(&self) -> Result<(), String> {
        if self.params.len() != self.classical_arity() {
            return Err(format!(
                "Gate {:?} expected {} parameters, got {}",
                self.gate_type,
                self.classical_arity(),
                self.params.len()
            ));
        }
        if self.qubits.len() % self.quantum_arity() != 0 {
            return Err(format!(
                "Gate {:?} requires a multiple of {} qubits, got {}",
                self.gate_type,
                self.quantum_arity(),
                self.qubits.len()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_command_creation() {
        // Single qubit gates
        let x_gate = Gate::x(&[0, 1, 2]);
        assert_eq!(x_gate.gate_type, GateType::X);
        assert_eq!(
            x_gate.qubits,
            vec![QubitId::from(0), QubitId::from(1), QubitId::from(2)]
        );
        assert!(x_gate.params.is_empty());

        // Parameterized single qubit gates
        let rz_gate = Gate::rz(0.5, &[1, 2]);
        assert_eq!(rz_gate.gate_type, GateType::RZ);
        assert_eq!(rz_gate.qubits, vec![QubitId::from(1), QubitId::from(2)]);
        assert_eq!(rz_gate.params, vec![0.5]);

        // Two qubit gates
        let cx_gate = Gate::cx(&[(0, 1), (2, 3)]);
        assert_eq!(cx_gate.gate_type, GateType::CX);
        assert_eq!(
            cx_gate.qubits,
            vec![
                QubitId::from(0),
                QubitId::from(1),
                QubitId::from(2),
                QubitId::from(3)
            ]
        );
        assert!(cx_gate.params.is_empty());

        // Measure gates
        let measure_gate = Gate::measure(&[2, 3]);
        assert_eq!(measure_gate.gate_type, GateType::Measure);
        assert_eq!(
            measure_gate.qubits,
            vec![QubitId::from(2), QubitId::from(3)]
        );
        assert!(measure_gate.params.is_empty());
    }

    #[test]
    fn test_two_qubit_gate_vec_variants() {
        // Test CX with _vec variant - much more convenient when you have a flat list
        let cx_pairs = Gate::cx(&[(0, 1), (2, 3)]);
        let cx_vec = Gate::cx_vec(&[0, 1, 2, 3]);
        assert_eq!(cx_pairs.gate_type, cx_vec.gate_type);
        assert_eq!(cx_pairs.qubits, cx_vec.qubits);
        assert_eq!(cx_pairs.params, cx_vec.params);

        // Test SZZ with _vec variant
        let szz_pairs = Gate::szz(&[(1, 2), (3, 4)]);
        let szz_vec = Gate::szz_vec(&[1, 2, 3, 4]);
        assert_eq!(szz_pairs.gate_type, szz_vec.gate_type);
        assert_eq!(szz_pairs.qubits, szz_vec.qubits);
        assert_eq!(szz_pairs.params, szz_vec.params);

        // Test SZZdg with _vec variant
        let szzdg_pairs = Gate::szzdg(&[(0, 2), (1, 3)]);
        let szzdg_vec = Gate::szzdg_vec(&[0, 2, 1, 3]);
        assert_eq!(szzdg_pairs.gate_type, szzdg_vec.gate_type);
        assert_eq!(szzdg_pairs.qubits, szzdg_vec.qubits);
        assert_eq!(szzdg_pairs.params, szzdg_vec.params);

        // Test RZZ with _vec variant
        let rzz_pairs = Gate::rzz(0.25, &[(0, 1), (2, 3)]);
        let rzz_vec = Gate::rzz_vec(0.25, &[0, 1, 2, 3]);
        assert_eq!(rzz_pairs.gate_type, rzz_vec.gate_type);
        assert_eq!(rzz_pairs.qubits, rzz_vec.qubits);
        assert_eq!(rzz_pairs.params, rzz_vec.params);
    }

    #[test]
    #[should_panic(expected = "CX gate requires an even number of qubits")]
    fn test_cx_vec_odd_qubits() {
        let _ = Gate::cx_vec(&[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "SZZ gate requires an even number of qubits")]
    fn test_szz_vec_odd_qubits() {
        let _ = Gate::szz_vec(&[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "SZZdg gate requires an even number of qubits")]
    fn test_szzdg_vec_odd_qubits() {
        let _ = Gate::szzdg_vec(&[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "RZZ gate requires an even number of qubits")]
    fn test_rzz_vec_odd_qubits() {
        let _ = Gate::rzz_vec(0.5, &[0, 1, 2]);
    }

    #[test]
    fn test_flatten_qubit_pairs_helper() {
        // Test the helper function directly
        let pairs = [(0usize, 1usize), (2usize, 3usize), (4usize, 5usize)];
        let flattened = Gate::flatten_qubit_pairs(&pairs);
        let expected: Vec<QubitId> = vec![0, 1, 2, 3, 4, 5]
            .into_iter()
            .map(QubitId::from)
            .collect();
        assert_eq!(flattened, expected);

        // Test empty case
        let empty_pairs: &[(usize, usize)] = &[];
        let flattened_empty = Gate::flatten_qubit_pairs(empty_pairs);
        assert!(flattened_empty.is_empty());
    }

    #[test]
    fn test_gate_arity_methods() {
        // Test single-qubit gates
        let x_gate = Gate::x(&[0]);
        assert_eq!(x_gate.classical_arity(), 0);
        assert_eq!(x_gate.quantum_arity(), 1);
        assert!(!x_gate.is_parameterized());
        assert!(x_gate.is_single_qubit());
        assert!(!x_gate.is_two_qubit());

        // Test parameterized single-qubit gates
        let rz_gate = Gate::rz(1.5, &[0]);
        assert_eq!(rz_gate.classical_arity(), 1);
        assert_eq!(rz_gate.quantum_arity(), 1);
        assert!(rz_gate.is_parameterized());
        assert!(rz_gate.is_single_qubit());
        assert!(!rz_gate.is_two_qubit());

        // Test two-parameter single-qubit gates
        let r1xy_gate = Gate::r1xy(0.5, 1.0, &[1]);
        assert_eq!(r1xy_gate.classical_arity(), 2);
        assert_eq!(r1xy_gate.quantum_arity(), 1);
        assert!(r1xy_gate.is_parameterized());
        assert!(r1xy_gate.is_single_qubit());
        assert!(!r1xy_gate.is_two_qubit());

        // Test three-parameter single-qubit gates
        let u_gate = Gate::u(0.5, 1.0, 1.5, &[2]);
        assert_eq!(u_gate.classical_arity(), 3);
        assert_eq!(u_gate.quantum_arity(), 1);
        assert!(u_gate.is_parameterized());
        assert!(u_gate.is_single_qubit());
        assert!(!u_gate.is_two_qubit());

        // Test two-qubit gates
        let cx_gate = Gate::cx(&[(0, 1)]);
        assert_eq!(cx_gate.classical_arity(), 0);
        assert_eq!(cx_gate.quantum_arity(), 2);
        assert!(!cx_gate.is_parameterized());
        assert!(!cx_gate.is_single_qubit());
        assert!(cx_gate.is_two_qubit());

        // Test parameterized two-qubit gates
        let rzz_two_qubit = Gate::rzz(0.25, &[(0, 1)]);
        assert_eq!(rzz_two_qubit.classical_arity(), 1);
        assert_eq!(rzz_two_qubit.quantum_arity(), 2);
        assert!(rzz_two_qubit.is_parameterized());
        assert!(!rzz_two_qubit.is_single_qubit());
        assert!(rzz_two_qubit.is_two_qubit());

        // Test idle gate (single-qubit, parameterized)
        let idle_gate = Gate::idle(1.0, vec![QubitId::from(0)]);
        assert_eq!(idle_gate.classical_arity(), 1);
        assert_eq!(idle_gate.quantum_arity(), 1);
        assert!(idle_gate.is_parameterized());
        assert!(idle_gate.is_single_qubit());
        assert!(!idle_gate.is_two_qubit());
    }

    #[test]
    fn test_gate_validation() {
        // Test valid gates
        let valid_x = Gate::x(&[0]);
        assert!(valid_x.validate().is_ok());

        let valid_rz = Gate::rz(1.5, &[1]);
        assert!(valid_rz.validate().is_ok());

        let valid_r1xy = Gate::r1xy(0.5, 1.0, &[2]);
        assert!(valid_r1xy.validate().is_ok());

        let valid_u = Gate::u(0.5, 1.0, 1.5, &[3]);
        assert!(valid_u.validate().is_ok());

        let valid_cx_gate = Gate::cx(&[(0, 1)]);
        assert!(valid_cx_gate.validate().is_ok());

        let valid_rzz = Gate::rzz(0.25, &[(2, 3)]);
        assert!(valid_rzz.validate().is_ok());

        // Test invalid gates - wrong parameter count
        let invalid_params = Gate::new(GateType::RZ, vec![1.0, 2.0], vec![QubitId::from(0)]);
        assert!(invalid_params.validate().is_err());
        assert!(
            invalid_params
                .validate()
                .unwrap_err()
                .contains("expected 1 parameters, got 2")
        );

        let missing_params = Gate::new(GateType::U, vec![1.0], vec![QubitId::from(0)]);
        assert!(missing_params.validate().is_err());
        assert!(
            missing_params
                .validate()
                .unwrap_err()
                .contains("expected 3 parameters, got 1")
        );

        // Test invalid gates - wrong qubit count (not a multiple of quantum arity)
        let invalid_qubits = Gate::new(GateType::CX, vec![], vec![QubitId::from(0)]);
        assert!(invalid_qubits.validate().is_err());
        assert!(
            invalid_qubits
                .validate()
                .unwrap_err()
                .contains("requires a multiple of 2 qubits, got 1")
        );

        let odd_cx_qubits = Gate::new(
            GateType::CX,
            vec![],
            vec![QubitId::from(0), QubitId::from(1), QubitId::from(2)],
        );
        assert!(odd_cx_qubits.validate().is_err());
        assert!(
            odd_cx_qubits
                .validate()
                .unwrap_err()
                .contains("requires a multiple of 2 qubits, got 3")
        );

        // Test valid multi-qubit gates
        let multi_x = Gate::new(
            GateType::X,
            vec![],
            vec![QubitId::from(0), QubitId::from(1), QubitId::from(2)],
        );
        assert!(multi_x.validate().is_ok()); // Multiple X gates on different qubits

        let multi_cx_gates = Gate::new(
            GateType::CX,
            vec![],
            vec![
                QubitId::from(0),
                QubitId::from(1),
                QubitId::from(2),
                QubitId::from(3),
            ],
        );
        assert!(multi_cx_gates.validate().is_ok()); // Multiple CX gates
    }
}
