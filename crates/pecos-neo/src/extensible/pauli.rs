//! Pauli operators and stabilizer strings.
//!
//! This module provides types for representing arbitrary Pauli strings
//! and generating their measurement/preparation decompositions.

use super::gates;
use super::operation::{AdaptedOp, AdaptedSequence, AncillaRequirements, PrepBasis, ResultId};
use pecos_core::QubitId;
use smallvec::SmallVec;

/// A single-qubit Pauli operator.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub enum Pauli {
    /// Identity (no operation)
    #[default]
    I,
    /// Pauli X
    X,
    /// Pauli Y
    Y,
    /// Pauli Z
    Z,
}

impl Pauli {
    /// Parse a Pauli from a character.
    #[must_use]
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'I' | 'i' | '_' => Some(Self::I),
            'X' | 'x' => Some(Self::X),
            'Y' | 'y' => Some(Self::Y),
            'Z' | 'z' => Some(Self::Z),
            _ => None,
        }
    }

    /// Convert to a character.
    #[must_use]
    pub const fn to_char(self) -> char {
        match self {
            Self::I => 'I',
            Self::X => 'X',
            Self::Y => 'Y',
            Self::Z => 'Z',
        }
    }

    /// Check if this is the identity.
    #[must_use]
    pub const fn is_identity(self) -> bool {
        matches!(self, Self::I)
    }

    /// Get the preparation basis for +1 eigenstate.
    #[must_use]
    pub const fn prep_basis(self) -> PrepBasis {
        match self {
            Self::I | Self::Z => PrepBasis::Z, // |0⟩
            Self::X => PrepBasis::X,           // |+⟩
            Self::Y => PrepBasis::Y,           // |+i⟩
        }
    }
}

/// A Pauli string (tensor product of single-qubit Paulis).
///
/// Represents operators like Z⊗X⊗I⊗Z.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PauliString {
    /// The Pauli operators, one per qubit.
    paulis: SmallVec<[Pauli; 8]>,
    /// Sign: true = +1, false = -1.
    positive: bool,
}

impl PauliString {
    /// Create a new Pauli string from a slice of Paulis.
    #[must_use]
    pub fn new(paulis: &[Pauli]) -> Self {
        Self {
            paulis: paulis.iter().copied().collect(),
            positive: true,
        }
    }

    /// Create from a string like "ZXI" or "XZZY".
    ///
    /// Returns `None` if the string contains invalid characters.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        // Handle optional sign prefix
        let (positive, chars) = if let Some(rest) = s.strip_prefix('-') {
            (false, rest)
        } else if let Some(rest) = s.strip_prefix('+') {
            (true, rest)
        } else {
            (true, s)
        };

        let paulis: Option<SmallVec<[Pauli; 8]>> = chars.chars().map(Pauli::from_char).collect();

        paulis.map(|p| Self {
            paulis: p,
            positive,
        })
    }

    /// Create an n-qubit identity string.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self {
            paulis: std::iter::repeat_n(Pauli::I, n).collect(),
            positive: true,
        }
    }

    /// Create a single Z on qubit i of n qubits.
    #[must_use]
    pub fn single_z(i: usize, n: usize) -> Self {
        let mut paulis: SmallVec<[Pauli; 8]> = std::iter::repeat_n(Pauli::I, n).collect();
        if i < n {
            paulis[i] = Pauli::Z;
        }
        Self {
            paulis,
            positive: true,
        }
    }

    /// Create a single X on qubit i of n qubits.
    #[must_use]
    pub fn single_x(i: usize, n: usize) -> Self {
        let mut paulis: SmallVec<[Pauli; 8]> = std::iter::repeat_n(Pauli::I, n).collect();
        if i < n {
            paulis[i] = Pauli::X;
        }
        Self {
            paulis,
            positive: true,
        }
    }

    /// Get the number of qubits.
    #[must_use]
    pub fn len(&self) -> usize {
        self.paulis.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.paulis.is_empty()
    }

    /// Get the Pauli at index i.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<Pauli> {
        self.paulis.get(i).copied()
    }

    /// Get the sign (+1 or -1).
    #[must_use]
    pub const fn is_positive(&self) -> bool {
        self.positive
    }

    /// Set the sign.
    #[must_use]
    pub const fn with_sign(mut self, positive: bool) -> Self {
        self.positive = positive;
        self
    }

    /// Negate the sign.
    #[must_use]
    pub const fn negated(mut self) -> Self {
        self.positive = !self.positive;
        self
    }

    /// Get the weight (number of non-identity Paulis).
    #[must_use]
    pub fn weight(&self) -> usize {
        self.paulis.iter().filter(|p| !p.is_identity()).count()
    }

    /// Check if this is the identity (all I).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.paulis.iter().all(|p| p.is_identity())
    }

    /// Iterate over (index, pauli) pairs for non-identity terms.
    pub fn non_identity_terms(&self) -> impl Iterator<Item = (usize, Pauli)> + '_ {
        self.paulis
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_identity())
            .map(|(i, p)| (i, *p))
    }

    /// Convert to string representation.
    #[must_use]
    pub fn to_string_repr(&self) -> String {
        let sign = if self.positive { "" } else { "-" };
        let paulis: String = self.paulis.iter().map(|p| p.to_char()).collect();
        format!("{sign}{paulis}")
    }
}

impl std::fmt::Display for PauliString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string_repr())
    }
}

/// Builder for stabilizer measurements.
///
/// Generates the gate sequence to measure an arbitrary Pauli string.
#[derive(Clone, Debug)]
pub struct StabilizerMeasurement {
    /// The Pauli string to measure.
    pauli: PauliString,
}

impl StabilizerMeasurement {
    /// Create a new stabilizer measurement.
    #[must_use]
    pub fn new(pauli: PauliString) -> Self {
        Self { pauli }
    }

    /// Create from a string like "ZXZ" or "-XZZY".
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        PauliString::from_str(s).map(Self::new)
    }

    /// Get ancilla requirements (1 clean ancilla for non-trivial measurements).
    #[must_use]
    pub fn ancilla_requirements(&self) -> AncillaRequirements {
        if self.pauli.weight() == 0 {
            AncillaRequirements::none()
        } else {
            AncillaRequirements::clean(1)
        }
    }

    /// Generate the measurement sequence.
    ///
    /// # Arguments
    /// * `qubits` - The data qubits (must match pauli string length)
    /// * `ancilla` - The ancilla qubit for the measurement
    ///
    /// # Returns
    /// An `AdaptedSequence` that measures the stabilizer and outputs the result.
    #[must_use]
    pub fn decompose(&self, qubits: &[QubitId], ancilla: QubitId) -> AdaptedSequence {
        assert_eq!(
            qubits.len(),
            self.pauli.len(),
            "Qubit count must match Pauli string length"
        );

        let mut ops = Vec::new();

        // 1. Prepare ancilla in |0⟩
        ops.push(AdaptedOp::pz(ancilla));

        // 2. For each non-identity Pauli, couple to ancilla
        for (i, pauli) in self.pauli.non_identity_terms() {
            let q = qubits[i];
            match pauli {
                Pauli::Z => {
                    // Z measurement: CX from data to ancilla
                    ops.push(AdaptedOp::gate2(gates::CX, q, ancilla));
                }
                Pauli::X => {
                    // X measurement: H, CX, H
                    ops.push(AdaptedOp::gate1(gates::H, q));
                    ops.push(AdaptedOp::gate2(gates::CX, q, ancilla));
                    ops.push(AdaptedOp::gate1(gates::H, q));
                }
                Pauli::Y => {
                    // Y measurement: SXdg, CX, SX
                    ops.push(AdaptedOp::gate1(gates::SXdg, q));
                    ops.push(AdaptedOp::gate2(gates::CX, q, ancilla));
                    ops.push(AdaptedOp::gate1(gates::SX, q));
                }
                Pauli::I => unreachable!(),
            }
        }

        // 3. If negative sign, flip the ancilla before measurement
        if !self.pauli.is_positive() {
            ops.push(AdaptedOp::gate1(gates::X, ancilla));
        }

        // 4. Measure ancilla
        ops.push(AdaptedOp::mz(ancilla, ResultId(0)));

        // 5. Output result
        ops.push(AdaptedOp::OutputResult {
            result: ResultId(0),
        });

        AdaptedSequence::new(ops)
    }
}

/// Builder for stabilizer preparations.
///
/// Generates the gate sequence to prepare a +1 eigenstate of a Pauli string.
#[derive(Clone, Debug)]
pub struct StabilizerPreparation {
    /// The Pauli string whose eigenstate to prepare.
    pauli: PauliString,
}

impl StabilizerPreparation {
    /// Create a new stabilizer preparation.
    #[must_use]
    pub fn new(pauli: PauliString) -> Self {
        Self { pauli }
    }

    /// Create from a string like "ZX" or "-XZ".
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        PauliString::from_str(s).map(Self::new)
    }

    /// Get ancilla requirements (none for preparation).
    #[must_use]
    pub fn ancilla_requirements(&self) -> AncillaRequirements {
        AncillaRequirements::none()
    }

    /// Generate the preparation sequence.
    ///
    /// Prepares each qubit in the +1 eigenstate of its Pauli term.
    /// For negative sign, flips the first non-identity qubit.
    #[must_use]
    pub fn decompose(&self, qubits: &[QubitId]) -> AdaptedSequence {
        assert_eq!(
            qubits.len(),
            self.pauli.len(),
            "Qubit count must match Pauli string length"
        );

        let mut ops = Vec::new();
        let mut first_nontrivial = None;

        // Prepare each qubit in the appropriate basis
        for (i, pauli) in self.pauli.paulis.iter().enumerate() {
            let q = qubits[i];
            ops.push(AdaptedOp::Prep {
                qubit: q,
                basis: pauli.prep_basis(),
            });

            if first_nontrivial.is_none() && !pauli.is_identity() {
                first_nontrivial = Some(i);
            }
        }

        // For negative eigenstate, flip the first non-trivial qubit
        // This changes the eigenvalue from +1 to -1
        if !self.pauli.is_positive()
            && let Some(i) = first_nontrivial
        {
            let q = qubits[i];
            // Apply the Pauli itself to flip the eigenvalue
            match self.pauli.paulis[i] {
                Pauli::X | Pauli::Y => ops.push(AdaptedOp::gate1(gates::Z, q)),
                Pauli::Z => ops.push(AdaptedOp::gate1(gates::X, q)),
                Pauli::I => {}
            }
        }

        AdaptedSequence::new(ops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_from_char() {
        assert_eq!(Pauli::from_char('X'), Some(Pauli::X));
        assert_eq!(Pauli::from_char('Y'), Some(Pauli::Y));
        assert_eq!(Pauli::from_char('Z'), Some(Pauli::Z));
        assert_eq!(Pauli::from_char('I'), Some(Pauli::I));
        assert_eq!(Pauli::from_char('x'), Some(Pauli::X));
        assert_eq!(Pauli::from_char('_'), Some(Pauli::I));
        assert_eq!(Pauli::from_char('Q'), None);
    }

    #[test]
    fn test_pauli_string_from_str() {
        let ps = PauliString::from_str("ZXI").unwrap();
        assert_eq!(ps.len(), 3);
        assert_eq!(ps.get(0), Some(Pauli::Z));
        assert_eq!(ps.get(1), Some(Pauli::X));
        assert_eq!(ps.get(2), Some(Pauli::I));
        assert!(ps.is_positive());

        let neg = PauliString::from_str("-XZ").unwrap();
        assert!(!neg.is_positive());
        assert_eq!(neg.len(), 2);
    }

    #[test]
    fn test_pauli_string_weight() {
        assert_eq!(PauliString::from_str("III").unwrap().weight(), 0);
        assert_eq!(PauliString::from_str("ZII").unwrap().weight(), 1);
        assert_eq!(PauliString::from_str("ZXI").unwrap().weight(), 2);
        assert_eq!(PauliString::from_str("ZXY").unwrap().weight(), 3);
    }

    #[test]
    fn test_pauli_string_display() {
        assert_eq!(PauliString::from_str("ZXI").unwrap().to_string(), "ZXI");
        assert_eq!(PauliString::from_str("-XZ").unwrap().to_string(), "-XZ");
    }

    #[test]
    fn test_stabilizer_measurement_zz() {
        let meas = StabilizerMeasurement::from_str("ZZ").unwrap();
        let seq = meas.decompose(&[QubitId(0), QubitId(1)], QubitId(2));

        // Should have: prep, CX, CX, meas, output
        assert!(seq.ops.len() >= 4);
        assert_eq!(seq.result_count, 1);
    }

    #[test]
    fn test_stabilizer_measurement_zxiz() {
        let meas = StabilizerMeasurement::from_str("ZXIZ").unwrap();
        let qubits = [QubitId(0), QubitId(1), QubitId(2), QubitId(3)];
        let seq = meas.decompose(&qubits, QubitId(4));

        // Weight is 3 (Z, X, Z), so we need couplings for each
        // Z: 1 gate (CX), X: 3 gates (H, CX, H), Z: 1 gate (CX)
        // Plus prep, measure, output
        assert!(seq.ops.len() >= 8);
    }

    #[test]
    fn test_stabilizer_preparation_zx() {
        let prep = StabilizerPreparation::from_str("ZX").unwrap();
        let seq = prep.decompose(&[QubitId(0), QubitId(1)]);

        // Should prep q0 in Z (|0⟩) and q1 in X (|+⟩)
        assert_eq!(seq.ops.len(), 2);

        match &seq.ops[0] {
            AdaptedOp::Prep { qubit, basis } => {
                assert_eq!(*qubit, QubitId(0));
                assert_eq!(*basis, PrepBasis::Z);
            }
            _ => panic!("Expected Prep"),
        }

        match &seq.ops[1] {
            AdaptedOp::Prep { qubit, basis } => {
                assert_eq!(*qubit, QubitId(1));
                assert_eq!(*basis, PrepBasis::X);
            }
            _ => panic!("Expected Prep"),
        }
    }

    #[test]
    fn test_stabilizer_preparation_negative() {
        let prep = StabilizerPreparation::from_str("-ZX").unwrap();
        let seq = prep.decompose(&[QubitId(0), QubitId(1)]);

        // Should prep then apply correction for -1 eigenvalue
        assert!(seq.ops.len() >= 2);
    }

    #[test]
    fn test_four_qubit_parity_check() {
        // Common in QEC: measure ZZZZ parity
        let meas = StabilizerMeasurement::from_str("ZZZZ").unwrap();
        let qubits = [QubitId(0), QubitId(1), QubitId(2), QubitId(3)];
        let seq = meas.decompose(&qubits, QubitId(4));

        // prep + 4 CX gates + meas + output
        assert!(seq.ops.len() >= 6);
    }

    #[test]
    fn test_surface_code_stabilizer() {
        // X-type stabilizer in surface code: XXXX
        let meas = StabilizerMeasurement::from_str("XXXX").unwrap();
        let qubits = [QubitId(0), QubitId(1), QubitId(2), QubitId(3)];
        let seq = meas.decompose(&qubits, QubitId(4));

        // Each X needs H-CX-H (3 gates), plus prep/meas/output
        assert!(seq.ops.len() >= 14);
    }
}
