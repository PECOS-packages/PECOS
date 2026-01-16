// Copyright 2024 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use crate::{Pauli, PauliBitmap, PauliOperator, PauliSparse, QuarterPhase, QubitId, VecSet};
use std::fmt;
use std::str::FromStr;

/// A string of Pauli operators acting on multiple qubits.
///
/// `PauliString` is the primary user-facing type for working with Pauli operators.
/// It stores individual Pauli operators (I, X, Y, Z) for each qubit along with a
/// global phase (+1, -1, +i, -i).
///
/// # Examples
///
/// ```
/// use pecos_core::{PauliString, Pauli, QuarterPhase, PauliOperator};
///
/// // Create from individual operators
/// let p = PauliString::from_paulis(&[Pauli::X, Pauli::X, Pauli::Z, Pauli::I]);
///
/// // Create single-qubit operators
/// let x0 = PauliString::x(0);
/// let z1 = PauliString::z(1);
///
/// // Check properties
/// assert_eq!(p.weight(), 3);  // X, X, Z (not counting I)
/// ```
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PauliString {
    phase: QuarterPhase,
    paulis: Vec<(Pauli, QubitId)>,
}

impl Default for PauliString {
    fn default() -> Self {
        Self::new()
    }
}

impl PauliString {
    /// Creates a new empty `PauliString` (identity operator with phase +1).
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            phase: QuarterPhase::PlusOne,
            paulis: Vec::new(),
        }
    }

    /// Creates a `PauliString` with the given phase and paulis.
    #[inline]
    #[must_use]
    pub fn with_phase_and_paulis(phase: QuarterPhase, paulis: Vec<(Pauli, QubitId)>) -> Self {
        Self { phase, paulis }
    }

    /// Creates a `PauliString` from a slice of Pauli operators on consecutive qubits.
    ///
    /// Qubits are numbered 0, 1, 2, ... in order. Identity operators are not stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_core::{PauliString, Pauli};
    ///
    /// let p = PauliString::from_paulis(&[Pauli::X, Pauli::Y, Pauli::Z]);
    /// // Creates X on qubit 0, Y on qubit 1, Z on qubit 2
    /// ```
    #[must_use]
    pub fn from_paulis(paulis: &[Pauli]) -> Self {
        Self::from_paulis_with_phase(QuarterPhase::PlusOne, paulis)
    }

    /// Creates a `PauliString` from a slice of Pauli operators with a specified phase.
    #[must_use]
    pub fn from_paulis_with_phase(phase: QuarterPhase, paulis: &[Pauli]) -> Self {
        let paulis = paulis
            .iter()
            .enumerate()
            .filter(|(_, p)| **p != Pauli::I)
            .map(|(i, p)| (*p, QubitId::new(i)))
            .collect();
        Self { phase, paulis }
    }

    /// Creates a single-qubit X operator on the given qubit.
    #[inline]
    #[must_use]
    pub fn x(qubit: usize) -> Self {
        Self {
            phase: QuarterPhase::PlusOne,
            paulis: vec![(Pauli::X, QubitId::new(qubit))],
        }
    }

    /// Creates a single-qubit Y operator on the given qubit.
    #[inline]
    #[must_use]
    pub fn y(qubit: usize) -> Self {
        Self {
            phase: QuarterPhase::PlusOne,
            paulis: vec![(Pauli::Y, QubitId::new(qubit))],
        }
    }

    /// Creates a single-qubit Z operator on the given qubit.
    #[inline]
    #[must_use]
    pub fn z(qubit: usize) -> Self {
        Self {
            phase: QuarterPhase::PlusOne,
            paulis: vec![(Pauli::Z, QubitId::new(qubit))],
        }
    }

    /// Creates an identity operator (empty `PauliString` with phase +1).
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        Self::new()
    }

    /// Creates a `PauliString` from non-overlapping X, Y, Z qubit sets.
    ///
    /// This is the inverse of `decompose()`. The qubit sets should be disjoint
    /// (no qubit should appear in more than one set).
    ///
    /// # Arguments
    ///
    /// * `phase` - The phase of the Pauli string
    /// * `x_qubits` - Qubits with pure X operator
    /// * `y_qubits` - Qubits with Y operator
    /// * `z_qubits` - Qubits with pure Z operator
    #[must_use]
    pub fn from_decomposed<I1, I2, I3>(
        phase: QuarterPhase,
        x_qubits: I1,
        y_qubits: I2,
        z_qubits: I3,
    ) -> Self
    where
        I1: IntoIterator<Item = usize>,
        I2: IntoIterator<Item = usize>,
        I3: IntoIterator<Item = usize>,
    {
        let mut paulis: Vec<(Pauli, QubitId)> = Vec::new();

        for q in x_qubits {
            paulis.push((Pauli::X, QubitId::new(q)));
        }
        for q in y_qubits {
            paulis.push((Pauli::Y, QubitId::new(q)));
        }
        for q in z_qubits {
            paulis.push((Pauli::Z, QubitId::new(q)));
        }

        // Sort by qubit index for consistent ordering
        paulis.sort_by_key(|(_, q)| q.index());

        Self { phase, paulis }
    }

    /// Returns the phase of this `PauliString`.
    #[inline]
    #[must_use]
    pub fn phase(&self) -> QuarterPhase {
        self.phase
    }

    /// Returns the phase (legacy name, prefer `phase()`).
    #[inline]
    #[must_use]
    pub fn get_phase(&self) -> QuarterPhase {
        self.phase
    }

    /// Returns a reference to the underlying Pauli operators.
    #[inline]
    #[must_use]
    pub fn paulis(&self) -> &[(Pauli, QubitId)] {
        &self.paulis
    }

    /// Returns a reference to the underlying Pauli operators (legacy name).
    #[inline]
    #[must_use]
    pub fn get_paulis(&self) -> &Vec<(Pauli, QubitId)> {
        &self.paulis
    }

    /// Iterates over (Pauli, `QubitId`) pairs.
    #[inline]
    pub fn iter_pairs(&self) -> impl Iterator<Item = (Pauli, QubitId)> + '_ {
        self.paulis.iter().copied()
    }

    /// Sets the phase of this `PauliString`.
    #[inline]
    pub fn set_phase(&mut self, phase: QuarterPhase) {
        self.phase = phase;
    }

    /// Returns the Pauli operator at the given qubit, or `Pauli::I` if not present.
    #[must_use]
    pub fn get(&self, qubit: usize) -> Pauli {
        self.paulis
            .iter()
            .find(|(_, q)| q.index() == qubit)
            .map_or(Pauli::I, |(p, _)| *p)
    }

    /// Returns the set of qubits this operator acts on non-trivially.
    #[must_use]
    pub fn qubits(&self) -> Vec<usize> {
        self.paulis.iter().map(|(_, q)| q.index()).collect()
    }

    /// Returns true if this is the identity operator (no non-trivial Paulis).
    #[inline]
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.paulis.is_empty()
    }

    // ========================================================================
    // Decomposition methods - non-overlapping X-only, Y, Z-only sets
    // ========================================================================

    /// Returns qubit IDs where the Pauli is exactly X (not Y).
    ///
    /// This differs from `x_positions()` which returns positions where the X-bit
    /// is set (including Y positions). This method returns only pure X positions.
    #[must_use]
    pub fn x_only_qubits(&self) -> Vec<QubitId> {
        self.paulis
            .iter()
            .filter(|(p, _)| *p == Pauli::X)
            .map(|(_, q)| *q)
            .collect()
    }

    /// Returns qubit IDs where the Pauli is exactly Y.
    #[must_use]
    pub fn y_qubits(&self) -> Vec<QubitId> {
        self.paulis
            .iter()
            .filter(|(p, _)| *p == Pauli::Y)
            .map(|(_, q)| *q)
            .collect()
    }

    /// Returns qubit IDs where the Pauli is exactly Z (not Y).
    ///
    /// This differs from `z_positions()` which returns positions where the Z-bit
    /// is set (including Y positions). This method returns only pure Z positions.
    #[must_use]
    pub fn z_only_qubits(&self) -> Vec<QubitId> {
        self.paulis
            .iter()
            .filter(|(p, _)| *p == Pauli::Z)
            .map(|(_, q)| *q)
            .collect()
    }

    /// Decomposes this `PauliString` into its phase and non-overlapping X, Y, Z qubit sets.
    ///
    /// This is useful for user-facing representation and serialization.
    /// The three qubit sets are disjoint - each qubit appears in at most one set.
    ///
    /// # Returns
    ///
    /// A tuple of `(phase, x_only_qubits, y_qubits, z_only_qubits)`
    #[must_use]
    pub fn decompose(&self) -> (QuarterPhase, Vec<QubitId>, Vec<QubitId>, Vec<QubitId>) {
        let mut x_only = Vec::new();
        let mut y = Vec::new();
        let mut z_only = Vec::new();

        for (pauli, qubit) in &self.paulis {
            match pauli {
                Pauli::X => x_only.push(*qubit),
                Pauli::Y => y.push(*qubit),
                Pauli::Z => z_only.push(*qubit),
                Pauli::I => {}
            }
        }

        (self.phase, x_only, y, z_only)
    }

    /// Returns a string representation like "XXZI" (without phase).
    ///
    /// If `num_qubits` is provided, pads with 'I' for missing qubits.
    /// Otherwise, only shows non-identity operators.
    #[must_use]
    pub fn pauli_str(&self, num_qubits: Option<usize>) -> String {
        if let Some(n) = num_qubits {
            let mut chars: Vec<char> = vec!['I'; n];
            for (pauli, qubit) in &self.paulis {
                if qubit.index() < n {
                    chars[qubit.index()] = match pauli {
                        Pauli::I => 'I',
                        Pauli::X => 'X',
                        Pauli::Y => 'Y',
                        Pauli::Z => 'Z',
                    };
                }
            }
            chars.into_iter().collect()
        } else {
            if self.paulis.is_empty() {
                return "I".to_string();
            }
            let max_qubit = self
                .paulis
                .iter()
                .map(|(_, q)| q.index())
                .max()
                .unwrap_or(0);
            self.pauli_str(Some(max_qubit + 1))
        }
    }

    // Conversion to efficient representations
    /// # Errors
    ///
    /// Results in an error if failed to create a valid `PauliSparse`
    pub fn into_pauli_sparse(self) -> Result<PauliSparse<VecSet<usize>>, String> {
        // Convert to SetPauli representation
        let mut x_positions = Vec::new();
        let mut y_positions = Vec::new();
        let mut z_positions = Vec::new();

        for (pauli, qubit) in self.paulis {
            let idx = qubit.index();
            match pauli {
                Pauli::X => x_positions.push(idx),
                Pauli::Z => z_positions.push(idx),
                Pauli::Y => y_positions.push(idx),
                Pauli::I => {}
            }
        }

        PauliSparse::with_operators(self.phase, &x_positions, &y_positions, &z_positions)
    }

    /// # Errors
    ///
    /// Results in an error if `QubitId`s are larger than 64 bits or if failed to create a valid `PauliBitmap`
    pub fn into_pauli_bitmap(self) -> Result<PauliBitmap, String> {
        // Convert to BitSetPauli if all qubits are < 64
        if self.paulis.iter().any(|(_, q)| q.index() >= 64) {
            return Err("QubitId larger than 64 bits".to_string());
        }

        let mut x_positions = Vec::new();
        let mut y_positions = Vec::new();
        let mut z_positions = Vec::new();

        for (pauli, qubit) in self.paulis {
            let idx = qubit.index() as u64;
            match pauli {
                Pauli::X => x_positions.push(idx),
                Pauli::Z => z_positions.push(idx),
                Pauli::Y => y_positions.push(idx),
                Pauli::I => {}
            }
        }

        PauliBitmap::with_operators(self.phase, &x_positions, &y_positions, &z_positions)
    }
}

impl From<PauliSparse<VecSet<usize>>> for PauliString {
    fn from(pauli_sparse: PauliSparse<VecSet<usize>>) -> Self {
        let mut paulis = Vec::new();

        // Collect all qubit positions
        let mut all_positions: Vec<_> = pauli_sparse
            .x_positions()
            .iter()
            .chain(pauli_sparse.z_positions().iter())
            .copied()
            .collect();
        all_positions.sort_unstable();
        all_positions.dedup();

        // Determine Pauli operator for each position
        for pos in all_positions {
            let qubit = QubitId::new(pos);
            let pauli = match (
                pauli_sparse.x_positions().contains(&pos),
                pauli_sparse.z_positions().contains(&pos),
            ) {
                (true, false) => Pauli::X,
                (false, true) => Pauli::Z,
                (true, true) => Pauli::Y,
                (false, false) => continue,
            };
            paulis.push((pauli, qubit));
        }

        Self {
            phase: pauli_sparse.phase(),
            paulis,
        }
    }
}

impl TryFrom<PauliBitmap> for PauliString {
    type Error = &'static str;

    fn try_from(pauli_bit: PauliBitmap) -> Result<Self, Self::Error> {
        let mut paulis = Vec::new();

        // Iterate through set bits in both x_bits and z_bits
        for i in 0..64 {
            let x_set = (pauli_bit.get_x_bits() >> i) & 1 == 1;
            let z_set = (pauli_bit.get_z_bits() >> i) & 1 == 1;

            let pauli = match (x_set, z_set) {
                (true, false) => Pauli::X,
                (false, true) => Pauli::Z,
                (true, true) => Pauli::Y,
                (false, false) => continue,
            };

            paulis.push((pauli, QubitId::new(i)));
        }

        Ok(Self {
            phase: pauli_bit.phase(),
            paulis,
        })
    }
}

// ============================================================================
// PauliOperator trait implementation
// ============================================================================

impl PauliOperator for PauliString {
    fn phase(&self) -> QuarterPhase {
        self.phase
    }

    fn x_positions(&self) -> Vec<usize> {
        self.paulis
            .iter()
            .filter(|(p, _)| matches!(p, Pauli::X | Pauli::Y))
            .map(|(_, q)| q.index())
            .collect()
    }

    fn z_positions(&self) -> Vec<usize> {
        self.paulis
            .iter()
            .filter(|(p, _)| matches!(p, Pauli::Z | Pauli::Y))
            .map(|(_, q)| q.index())
            .collect()
    }

    fn multiply(&self, other: &Self) -> Self {
        // Convert to sparse representation, multiply, convert back
        // This is simpler and avoids duplicating the multiplication logic
        let sparse_self = self.clone().into_pauli_sparse().expect("valid pauli");
        let sparse_other = other.clone().into_pauli_sparse().expect("valid pauli");
        let result = sparse_self.multiply(&sparse_other);
        PauliString::from(result)
    }

    fn weight(&self) -> usize {
        self.paulis.len()
    }

    fn commutes_with(&self, other: &Self) -> bool {
        // Count anticommuting pairs: X-Z and Z-X overlaps
        let mut anticommute_count = 0;

        for (p1, q1) in &self.paulis {
            for (p2, q2) in &other.paulis {
                if q1 == q2 {
                    // Check if these Paulis anticommute
                    let anti = matches!(
                        (p1, p2),
                        (Pauli::X | Pauli::Z, Pauli::Y)
                            | (Pauli::X | Pauli::Y, Pauli::Z)
                            | (Pauli::Y | Pauli::Z, Pauli::X)
                    );
                    if anti {
                        anticommute_count += 1;
                    }
                }
            }
        }

        anticommute_count % 2 == 0
    }

    fn from_single(qubit: usize, pauli: Pauli) -> Self {
        if pauli == Pauli::I {
            Self::new()
        } else {
            Self {
                phase: QuarterPhase::PlusOne,
                paulis: vec![(pauli, QubitId::new(qubit))],
            }
        }
    }
}

// ============================================================================
// Display implementation
// ============================================================================

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format phase
        let phase_str = match self.phase {
            QuarterPhase::PlusOne => "+",
            QuarterPhase::MinusOne => "-",
            QuarterPhase::PlusI => "+i",
            QuarterPhase::MinusI => "-i",
        };
        write!(f, "{}{}", phase_str, self.pauli_str(None))
    }
}

// ============================================================================
// FromStr implementation
// ============================================================================

/// Error type for parsing `PauliString` from a string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsePauliStringError {
    pub message: String,
}

impl fmt::Display for ParsePauliStringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ParsePauliStringError {}

impl FromStr for PauliString {
    type Err = ParsePauliStringError;

    /// Parses a `PauliString` from a string like "+iXXZI" or "XXZI".
    ///
    /// # Format
    /// - Optional phase prefix: `+`, `-`, `+i`, `-i`, `i`
    /// - Followed by Pauli operators: `I`, `X`, `Y`, `Z`
    ///
    /// # Examples
    /// ```
    /// use pecos_core::{PauliString, QuarterPhase};
    /// use std::str::FromStr;
    ///
    /// let p = PauliString::from_str("+iXXZI").unwrap();
    /// assert_eq!(p.phase(), QuarterPhase::PlusI);
    ///
    /// let p = PauliString::from_str("XYZ").unwrap();
    /// assert_eq!(p.phase(), QuarterPhase::PlusOne);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(Self::new());
        }

        let mut chars = s.chars().peekable();

        // Parse phase prefix
        let phase = match chars.peek() {
            Some('+') => {
                chars.next();
                if chars.peek() == Some(&'i') {
                    chars.next();
                    QuarterPhase::PlusI
                } else {
                    QuarterPhase::PlusOne
                }
            }
            Some('-') => {
                chars.next();
                if chars.peek() == Some(&'i') {
                    chars.next();
                    QuarterPhase::MinusI
                } else {
                    QuarterPhase::MinusOne
                }
            }
            Some('i') => {
                chars.next();
                QuarterPhase::PlusI
            }
            _ => QuarterPhase::PlusOne,
        };

        // Parse Pauli operators
        let mut paulis = Vec::new();
        for (idx, c) in chars.enumerate() {
            let pauli = match c {
                'I' | '1' => continue, // Skip identity
                'X' | 'x' => Pauli::X,
                'Y' | 'y' => Pauli::Y,
                'Z' | 'z' => Pauli::Z,
                c => {
                    return Err(ParsePauliStringError {
                        message: format!("Invalid Pauli character: '{c}'"),
                    });
                }
            };
            paulis.push((pauli, QubitId::new(idx)));
        }

        Ok(Self { phase, paulis })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x_only_qubits() {
        // XYZI - X on 0, Y on 1, Z on 2, I on 3
        let p = PauliString::from_paulis(&[Pauli::X, Pauli::Y, Pauli::Z, Pauli::I]);
        let x_only = p.x_only_qubits();
        assert_eq!(x_only.len(), 1);
        assert_eq!(x_only[0].index(), 0);
    }

    #[test]
    fn test_y_qubits() {
        let p = PauliString::from_paulis(&[Pauli::X, Pauli::Y, Pauli::Z, Pauli::Y]);
        let y = p.y_qubits();
        assert_eq!(y.len(), 2);
        assert_eq!(y[0].index(), 1);
        assert_eq!(y[1].index(), 3);
    }

    #[test]
    fn test_z_only_qubits() {
        let p = PauliString::from_paulis(&[Pauli::X, Pauli::Y, Pauli::Z, Pauli::I]);
        let z_only = p.z_only_qubits();
        assert_eq!(z_only.len(), 1);
        assert_eq!(z_only[0].index(), 2);
    }

    #[test]
    fn test_decompose() {
        // +iXYZI
        let p = PauliString::from_paulis_with_phase(
            QuarterPhase::PlusI,
            &[Pauli::X, Pauli::Y, Pauli::Z, Pauli::I],
        );

        let (phase, x_only, y, z_only) = p.decompose();

        assert_eq!(phase, QuarterPhase::PlusI);
        assert_eq!(x_only.len(), 1);
        assert_eq!(x_only[0].index(), 0);
        assert_eq!(y.len(), 1);
        assert_eq!(y[0].index(), 1);
        assert_eq!(z_only.len(), 1);
        assert_eq!(z_only[0].index(), 2);
    }

    #[test]
    fn test_decompose_disjoint() {
        // Verify that decomposition produces disjoint sets
        let p = PauliString::from_paulis(&[Pauli::X, Pauli::Y, Pauli::Z, Pauli::Y, Pauli::X]);

        let (_, x_only, y, z_only) = p.decompose();

        // Total should equal weight
        assert_eq!(x_only.len() + y.len() + z_only.len(), p.weight());

        // Verify no overlap
        let x_set: std::collections::HashSet<_> =
            x_only.iter().map(crate::qubit_id::QubitId::index).collect();
        let y_set: std::collections::HashSet<_> =
            y.iter().map(crate::qubit_id::QubitId::index).collect();
        let z_set: std::collections::HashSet<_> =
            z_only.iter().map(crate::qubit_id::QubitId::index).collect();

        assert!(x_set.is_disjoint(&y_set));
        assert!(x_set.is_disjoint(&z_set));
        assert!(y_set.is_disjoint(&z_set));
    }

    #[test]
    fn test_decompose_vs_positions() {
        // Verify that x_positions includes Y, but x_only_qubits doesn't
        let p = PauliString::from_paulis(&[Pauli::X, Pauli::Y, Pauli::Z]);

        // x_positions should return [0, 1] (X and Y)
        let x_pos = p.x_positions();
        assert_eq!(x_pos.len(), 2);
        assert!(x_pos.contains(&0));
        assert!(x_pos.contains(&1));

        // x_only_qubits should return just [0] (only pure X)
        let x_only = p.x_only_qubits();
        assert_eq!(x_only.len(), 1);
        assert_eq!(x_only[0].index(), 0);
    }

    #[test]
    fn test_from_decomposed() {
        // Create from decomposed form
        let p = PauliString::from_decomposed(
            QuarterPhase::MinusOne,
            [0, 3], // X on qubits 0, 3
            [1],    // Y on qubit 1
            [2, 4], // Z on qubits 2, 4
        );

        assert_eq!(p.phase(), QuarterPhase::MinusOne);
        assert_eq!(p.weight(), 5);
        assert_eq!(p.get(0), Pauli::X);
        assert_eq!(p.get(1), Pauli::Y);
        assert_eq!(p.get(2), Pauli::Z);
        assert_eq!(p.get(3), Pauli::X);
        assert_eq!(p.get(4), Pauli::Z);
    }

    #[test]
    fn test_decompose_roundtrip() {
        // Create a PauliString, decompose it, recreate it
        let original = PauliString::from_paulis_with_phase(
            QuarterPhase::MinusI,
            &[Pauli::X, Pauli::Y, Pauli::Z, Pauli::Y, Pauli::X],
        );

        let (phase, x_only, y, z_only) = original.decompose();
        let reconstructed = PauliString::from_decomposed(
            phase,
            x_only.iter().map(crate::qubit_id::QubitId::index),
            y.iter().map(crate::qubit_id::QubitId::index),
            z_only.iter().map(crate::qubit_id::QubitId::index),
        );

        assert_eq!(original.phase(), reconstructed.phase());
        assert_eq!(original.weight(), reconstructed.weight());
        for q in 0..5 {
            assert_eq!(original.get(q), reconstructed.get(q));
        }
    }

    // ========================================================================
    // pauli_str tests
    // ========================================================================

    #[test]
    fn test_pauli_str_basic() {
        // pauli_str returns just the Pauli characters, no phase prefix
        let p = PauliString::from_paulis(&[Pauli::X, Pauli::Y, Pauli::Z]);
        let s = p.pauli_str(None);
        assert_eq!(s, "XYZ");
    }

    #[test]
    fn test_pauli_str_with_phase() {
        // Phase is stored separately, not included in pauli_str output
        let p = PauliString::from_paulis_with_phase(QuarterPhase::PlusI, &[Pauli::X, Pauli::Y]);
        let s = p.pauli_str(None);
        assert_eq!(s, "XY");
        // Phase is accessible separately
        assert_eq!(p.phase(), QuarterPhase::PlusI);
    }

    #[test]
    fn test_pauli_str_minus_phase() {
        // Phase is stored separately
        let p = PauliString::from_paulis_with_phase(QuarterPhase::MinusOne, &[Pauli::Z]);
        let s = p.pauli_str(None);
        assert_eq!(s, "Z");
        assert_eq!(p.phase(), QuarterPhase::MinusOne);
    }

    #[test]
    fn test_pauli_str_minus_i_phase() {
        let p = PauliString::from_paulis_with_phase(QuarterPhase::MinusI, &[Pauli::X]);
        let s = p.pauli_str(None);
        assert_eq!(s, "X");
        assert_eq!(p.phase(), QuarterPhase::MinusI);
    }

    #[test]
    fn test_pauli_str_with_num_qubits() {
        let p = PauliString::from_paulis(&[Pauli::X]);
        // Request 3 qubits, should pad with I
        let s = p.pauli_str(Some(3));
        assert_eq!(s, "XII");
    }

    #[test]
    fn test_pauli_str_identity() {
        let p = PauliString::identity();
        let s = p.pauli_str(Some(2));
        assert_eq!(s, "II");
    }

    // ========================================================================
    // into_pauli_sparse tests
    // ========================================================================

    #[test]
    fn test_into_pauli_sparse_basic() {
        let p = PauliString::from_paulis(&[Pauli::X, Pauli::Y, Pauli::Z]);
        let sparse = p.into_pauli_sparse();
        assert!(sparse.is_ok());
        let sparse = sparse.unwrap();
        assert_eq!(sparse.weight(), 3);
    }

    #[test]
    fn test_into_pauli_sparse_preserves_phase() {
        let p = PauliString::from_paulis_with_phase(QuarterPhase::MinusI, &[Pauli::X, Pauli::Z]);
        let sparse = p.into_pauli_sparse();
        assert!(sparse.is_ok());
        let sparse = sparse.unwrap();
        assert_eq!(sparse.phase(), QuarterPhase::MinusI);
    }

    #[test]
    fn test_into_pauli_sparse_identity() {
        let p = PauliString::identity();
        let sparse = p.into_pauli_sparse();
        assert!(sparse.is_ok());
        let sparse = sparse.unwrap();
        assert_eq!(sparse.weight(), 0);
    }

    // ========================================================================
    // into_pauli_bitmap tests
    // ========================================================================

    #[test]
    fn test_into_pauli_bitmap_basic() {
        let p = PauliString::from_paulis(&[Pauli::X, Pauli::Y, Pauli::Z]);
        let bitmap = p.into_pauli_bitmap();
        assert!(bitmap.is_ok());
        let bitmap = bitmap.unwrap();
        assert_eq!(bitmap.weight(), 3);
    }

    #[test]
    fn test_into_pauli_bitmap_preserves_phase() {
        let p = PauliString::from_paulis_with_phase(QuarterPhase::PlusI, &[Pauli::Y]);
        let bitmap = p.into_pauli_bitmap();
        assert!(bitmap.is_ok());
        let bitmap = bitmap.unwrap();
        assert_eq!(bitmap.phase(), QuarterPhase::PlusI);
    }

    #[test]
    fn test_into_pauli_bitmap_weight() {
        // Create a PauliString with mixed Paulis, verify weight
        let p = PauliString::from_paulis(&[Pauli::X, Pauli::I, Pauli::Z, Pauli::Y]);
        let bitmap = p.into_pauli_bitmap().unwrap();

        // Weight should be 3 (X, Z, Y - not counting I)
        assert_eq!(bitmap.weight(), 3);
    }
}
