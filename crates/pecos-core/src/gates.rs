//! Gate command representation for quantum operations
//!
//! This module provides the `GateCommand` struct which represents a quantum
//! gate operation with its type, qubits, and parameters.

use crate::Angle64;
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
/// - Uses `Angle64` for rotation angles (in full turns)
/// - Flat structure for easy access to gate data
/// - Compatible with binary protocol serialization
#[derive(Debug, Clone, PartialEq)]
pub struct Gate {
    /// The type of the gate
    pub gate_type: GateType,
    /// Rotation angles for parameterized gates (in full turns).
    /// Use `Angle64::from_turns()` or `Angle64::from_radians()` to create.
    pub angles: Vec<Angle64>,
    /// Other non-angle parameters (e.g., duration for Idle gate)
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
    /// Create a new gate command with angles and params
    #[must_use]
    pub fn new(
        gate_type: GateType,
        angles: Vec<Angle64>,
        params: Vec<f64>,
        qubits: Vec<impl Into<QubitId>>,
    ) -> Self {
        Self {
            gate_type,
            angles,
            params,
            qubits: qubits.into_iter().map(Into::into).collect(),
        }
    }

    /// Create a new gate command with angles only (no other params)
    #[must_use]
    pub fn with_angles(
        gate_type: GateType,
        angles: Vec<Angle64>,
        qubits: Vec<impl Into<QubitId>>,
    ) -> Self {
        Self::new(gate_type, angles, vec![], qubits)
    }

    /// Create a new gate command with no angles or params
    #[must_use]
    pub fn simple(gate_type: GateType, qubits: Vec<impl Into<QubitId>>) -> Self {
        Self::new(gate_type, vec![], vec![], qubits)
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
        Self::simple(GateType::X, qubits.iter().map(|&q| q.into()).collect())
    }

    /// Create Y gate on multiple qubits
    #[must_use]
    pub fn y(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(GateType::Y, qubits.iter().map(|&q| q.into()).collect())
    }

    /// Create Z gate on multiple qubits
    #[must_use]
    pub fn z(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(GateType::Z, qubits.iter().map(|&q| q.into()).collect())
    }

    /// Create H gate on multiple qubits
    #[must_use]
    pub fn h(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(GateType::H, qubits.iter().map(|&q| q.into()).collect())
    }

    /// Create CX gate from flat qubit list (control1, target1, control2, target2, ...)
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even, as `CX` gates require pairs of qubits.
    #[must_use]
    pub fn cx_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "CX gate requires an even number of qubits"
        );
        Self::simple(GateType::CX, qubits.iter().map(|&q| q.into()).collect())
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
            qubits.len().is_multiple_of(2),
            "SZZ gate requires an even number of qubits"
        );
        Self::simple(GateType::SZZ, qubits.iter().map(|&q| q.into()).collect())
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
            qubits.len().is_multiple_of(2),
            "SZZdg gate requires an even number of qubits"
        );
        Self::simple(GateType::SZZdg, qubits.iter().map(|&q| q.into()).collect())
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
    pub fn rzz_vec(theta: Angle64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "RZZ gate requires an even number of qubits"
        );
        Self::with_angles(
            GateType::RZZ,
            vec![theta],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create RZZ gate on multiple qubit pairs
    #[must_use]
    pub fn rzz(
        theta: Angle64,
        qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::rzz_vec(theta, &flat_qubits)
    }

    /// Create RX gate on multiple qubits
    #[must_use]
    pub fn rx(theta: Angle64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::with_angles(
            GateType::RX,
            vec![theta],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create RY gate on multiple qubits
    #[must_use]
    pub fn ry(theta: Angle64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::with_angles(
            GateType::RY,
            vec![theta],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create RZ gate on multiple qubits
    #[must_use]
    pub fn rz(theta: Angle64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::with_angles(
            GateType::RZ,
            vec![theta],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create R1XY gate on multiple qubits
    #[must_use]
    pub fn r1xy(theta: Angle64, phi: Angle64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::with_angles(
            GateType::R1XY,
            vec![theta, phi],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create U gate on multiple qubits
    #[must_use]
    pub fn u(
        theta: Angle64,
        phi: Angle64,
        lambda: Angle64,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> Self {
        Self::with_angles(
            GateType::U,
            vec![theta, phi, lambda],
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create Measure gate on multiple qubits
    #[must_use]
    pub fn measure(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::Measure,
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create `MeasureLeaked` gate on multiple qubits
    #[must_use]
    pub fn measure_leaked(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::MeasureLeaked,
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create Prep gate on multiple qubits
    #[must_use]
    pub fn prep(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(GateType::Prep, qubits.iter().map(|&q| q.into()).collect())
    }

    /// Create `QAlloc` gate to allocate qubits in the |0⟩ state
    #[must_use]
    pub fn qalloc(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(GateType::QAlloc, qubits.iter().map(|&q| q.into()).collect())
    }

    /// Create `QFree` gate to deallocate qubits
    #[must_use]
    pub fn qfree(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(GateType::QFree, qubits.iter().map(|&q| q.into()).collect())
    }

    /// Create `MeasureFree` gate (measure and deallocate) on multiple qubits
    #[must_use]
    pub fn measure_free(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::MeasureFree,
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
        Self::new(GateType::Idle, vec![], vec![duration], qubits)
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

    /// Create a new `MeasCrosstalkGlobalPayload` with the data from runtime.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The qubits that are guaranteed *not* to be affected by the
    ///   global crosstalk event.
    ///
    /// NOTE: it seems unintuitive to give the complement of the list of victim qubits.
    /// It fits better with the previous version of crosstalk, but we might want to
    /// refactor this.
    ///
    /// # Returns
    ///
    /// A new `MeasCrosstalkGlobalPayload` gate with the specified parameters
    #[must_use]
    pub fn meas_crosstalk_global_payload(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::MeasCrosstalkGlobalPayload,
            qubits.iter().map(|&q| q.into()).collect(),
        )
    }

    /// Create a new `MeasCrosstalkLocalPayload` with the data from runtime.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The qubits that are potential victims of the local crosstalk event.
    ///
    /// # Returns
    ///
    /// A new `MeasCrosstalkLocalPayload` gate with the specified parameters
    #[must_use]
    pub fn meas_crosstalk_local_payload(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::MeasCrosstalkLocalPayload,
            qubits.iter().map(|&q| q.into()).collect(),
        )
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

    /// Returns the number of angle parameters this gate requires
    #[inline]
    #[must_use]
    pub fn angle_arity(&self) -> usize {
        self.gate_type.angle_arity()
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
    /// - The number of angles doesn't match the gate's angle arity
    /// - The number of qubits is not a multiple of the gate's quantum arity
    pub fn validate(&self) -> Result<(), String> {
        // Check angle parameters
        if self.angles.len() != self.angle_arity() {
            return Err(format!(
                "Gate {:?} expected {} angle parameters, got {}",
                self.gate_type,
                self.angle_arity(),
                self.angles.len()
            ));
        }
        // Check qubit count
        if !self.qubits.len().is_multiple_of(self.quantum_arity()) {
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
        assert!(x_gate.angles.is_empty());

        // Parameterized single qubit gates
        let rz_gate = Gate::rz(Angle64::from_turns(0.5), &[1, 2]);
        assert_eq!(rz_gate.gate_type, GateType::RZ);
        assert_eq!(rz_gate.qubits, vec![QubitId::from(1), QubitId::from(2)]);
        assert_eq!(rz_gate.angles, vec![Angle64::from_turns(0.5)]);

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
        assert!(cx_gate.angles.is_empty());

        // Measure gates
        let measure_gate = Gate::measure(&[2, 3]);
        assert_eq!(measure_gate.gate_type, GateType::Measure);
        assert_eq!(
            measure_gate.qubits,
            vec![QubitId::from(2), QubitId::from(3)]
        );
        assert!(measure_gate.angles.is_empty());
    }

    #[test]
    fn test_two_qubit_gate_vec_variants() {
        // Test CX with _vec variant - much more convenient when you have a flat list
        let cx_pairs = Gate::cx(&[(0, 1), (2, 3)]);
        let cx_vec = Gate::cx_vec(&[0, 1, 2, 3]);
        assert_eq!(cx_pairs.gate_type, cx_vec.gate_type);
        assert_eq!(cx_pairs.qubits, cx_vec.qubits);
        assert_eq!(cx_pairs.angles, cx_vec.angles);

        // Test SZZ with _vec variant
        let szz_pairs = Gate::szz(&[(1, 2), (3, 4)]);
        let szz_vec = Gate::szz_vec(&[1, 2, 3, 4]);
        assert_eq!(szz_pairs.gate_type, szz_vec.gate_type);
        assert_eq!(szz_pairs.qubits, szz_vec.qubits);
        assert_eq!(szz_pairs.angles, szz_vec.angles);

        // Test SZZdg with _vec variant
        let szzdg_pairs = Gate::szzdg(&[(0, 2), (1, 3)]);
        let szzdg_vec = Gate::szzdg_vec(&[0, 2, 1, 3]);
        assert_eq!(szzdg_pairs.gate_type, szzdg_vec.gate_type);
        assert_eq!(szzdg_pairs.qubits, szzdg_vec.qubits);
        assert_eq!(szzdg_pairs.angles, szzdg_vec.angles);

        // Test RZZ with _vec variant
        let angle = Angle64::from_turns(0.25);
        let rzz_pairs = Gate::rzz(angle, &[(0, 1), (2, 3)]);
        let rzz_vec = Gate::rzz_vec(angle, &[0, 1, 2, 3]);
        assert_eq!(rzz_pairs.gate_type, rzz_vec.gate_type);
        assert_eq!(rzz_pairs.qubits, rzz_vec.qubits);
        assert_eq!(rzz_pairs.angles, rzz_vec.angles);
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
        let _ = Gate::rzz_vec(Angle64::from_turns(0.5), &[0, 1, 2]);
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
        assert_eq!(x_gate.angle_arity(), 0);
        assert_eq!(x_gate.quantum_arity(), 1);
        assert!(!x_gate.is_parameterized());
        assert!(x_gate.is_single_qubit());
        assert!(!x_gate.is_two_qubit());

        // Test parameterized single-qubit gates
        let rz_gate = Gate::rz(Angle64::from_turns(0.25), &[0]);
        assert_eq!(rz_gate.classical_arity(), 1);
        assert_eq!(rz_gate.angle_arity(), 1);
        assert_eq!(rz_gate.quantum_arity(), 1);
        assert!(rz_gate.is_parameterized());
        assert!(rz_gate.is_single_qubit());
        assert!(!rz_gate.is_two_qubit());

        // Test two-parameter single-qubit gates
        let r1xy_gate = Gate::r1xy(Angle64::from_turns(0.5), Angle64::from_turns(0.25), &[1]);
        assert_eq!(r1xy_gate.classical_arity(), 2);
        assert_eq!(r1xy_gate.angle_arity(), 2);
        assert_eq!(r1xy_gate.quantum_arity(), 1);
        assert!(r1xy_gate.is_parameterized());
        assert!(r1xy_gate.is_single_qubit());
        assert!(!r1xy_gate.is_two_qubit());

        // Test three-parameter single-qubit gates
        let u_gate = Gate::u(
            Angle64::from_turns(0.5),
            Angle64::from_turns(0.25),
            Angle64::from_turns(0.125),
            &[2],
        );
        assert_eq!(u_gate.classical_arity(), 3);
        assert_eq!(u_gate.angle_arity(), 3);
        assert_eq!(u_gate.quantum_arity(), 1);
        assert!(u_gate.is_parameterized());
        assert!(u_gate.is_single_qubit());
        assert!(!u_gate.is_two_qubit());

        // Test two-qubit gates
        let cx_gate = Gate::cx(&[(0, 1)]);
        assert_eq!(cx_gate.classical_arity(), 0);
        assert_eq!(cx_gate.angle_arity(), 0);
        assert_eq!(cx_gate.quantum_arity(), 2);
        assert!(!cx_gate.is_parameterized());
        assert!(!cx_gate.is_single_qubit());
        assert!(cx_gate.is_two_qubit());

        // Test parameterized two-qubit gates
        let rzz_two_qubit = Gate::rzz(Angle64::from_turns(0.25), &[(0, 1)]);
        assert_eq!(rzz_two_qubit.classical_arity(), 1);
        assert_eq!(rzz_two_qubit.angle_arity(), 1);
        assert_eq!(rzz_two_qubit.quantum_arity(), 2);
        assert!(rzz_two_qubit.is_parameterized());
        assert!(!rzz_two_qubit.is_single_qubit());
        assert!(rzz_two_qubit.is_two_qubit());

        // Test idle gate (single-qubit, parameterized but not with angles)
        let idle_gate = Gate::idle(1.0, vec![QubitId::from(0)]);
        assert_eq!(idle_gate.classical_arity(), 1);
        assert_eq!(idle_gate.angle_arity(), 0); // Idle uses params, not angles
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

        let valid_rz = Gate::rz(Angle64::from_turns(0.25), &[1]);
        assert!(valid_rz.validate().is_ok());

        let valid_r1xy = Gate::r1xy(Angle64::from_turns(0.5), Angle64::from_turns(0.25), &[2]);
        assert!(valid_r1xy.validate().is_ok());

        let valid_u = Gate::u(
            Angle64::from_turns(0.5),
            Angle64::from_turns(0.25),
            Angle64::from_turns(0.125),
            &[3],
        );
        assert!(valid_u.validate().is_ok());

        let valid_cx_gate = Gate::cx(&[(0, 1)]);
        assert!(valid_cx_gate.validate().is_ok());

        let valid_rzz = Gate::rzz(Angle64::from_turns(0.25), &[(2, 3)]);
        assert!(valid_rzz.validate().is_ok());

        // Test invalid gates - wrong angle count
        let invalid_angles = Gate::new(
            GateType::RZ,
            vec![Angle64::from_turns(0.25), Angle64::from_turns(0.5)],
            vec![],
            vec![QubitId::from(0)],
        );
        assert!(invalid_angles.validate().is_err());
        assert!(
            invalid_angles
                .validate()
                .unwrap_err()
                .contains("expected 1 angle parameters, got 2")
        );

        let missing_angles = Gate::new(
            GateType::U,
            vec![Angle64::from_turns(0.25)],
            vec![],
            vec![QubitId::from(0)],
        );
        assert!(missing_angles.validate().is_err());
        assert!(
            missing_angles
                .validate()
                .unwrap_err()
                .contains("expected 3 angle parameters, got 1")
        );

        // Test invalid gates - wrong qubit count (not a multiple of quantum arity)
        let invalid_qubits = Gate::new(GateType::CX, vec![], vec![], vec![QubitId::from(0)]);
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
            vec![],
            vec![QubitId::from(0), QubitId::from(1), QubitId::from(2)],
        );
        assert!(multi_x.validate().is_ok()); // Multiple X gates on different qubits

        let multi_cx_gates = Gate::new(
            GateType::CX,
            vec![],
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
