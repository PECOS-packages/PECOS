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

use crate::{Pauli, PauliBitmap, PauliOperator, PauliSparse, Phase, QuarterPhase, QubitId, VecSet};
use std::fmt;
use std::str::FromStr;

/// A string of Pauli operators acting on multiple qubits.
///
/// `PauliString` is the primary user-facing type for working with Pauli operators.
/// It stores individual Pauli operators (I, X, Y, Z) for each qubit along with a
/// global phase (+1, -1, +i, -i).
///
/// # `Hash` and `Eq` caveats
///
/// `Hash` and `Eq` compare the internal `(Pauli, QubitId)` pairs in storage order.
/// Two `PauliString`s that represent the same operator but were constructed with
/// different qubit ordering may compare as unequal and hash differently.
/// If you need order-independent equality and hashing (e.g., for a `HashSet`),
/// use [`PauliSet`] which normalizes to a canonical sorted form.
///
/// [`PauliSet`]: https://docs.rs/pecos-quantum/latest/pecos_quantum/struct.PauliSet.html
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

    /// Creates a multi-qubit X operator: X on each of the given qubits.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_core::{PauliString, Pauli, PauliOperator};
    ///
    /// let p = PauliString::xs(&[0, 2, 5]);
    /// assert_eq!(p.get(0), Pauli::X);
    /// assert_eq!(p.get(1), Pauli::I);
    /// assert_eq!(p.get(2), Pauli::X);
    /// assert_eq!(p.get(5), Pauli::X);
    /// assert_eq!(p.weight(), 3);
    /// ```
    #[must_use]
    pub fn xs(qubits: &[usize]) -> Self {
        Self {
            phase: QuarterPhase::PlusOne,
            paulis: qubits
                .iter()
                .map(|&q| (Pauli::X, QubitId::new(q)))
                .collect(),
        }
    }

    /// Creates a multi-qubit Y operator: Y on each of the given qubits.
    #[must_use]
    pub fn ys(qubits: &[usize]) -> Self {
        Self {
            phase: QuarterPhase::PlusOne,
            paulis: qubits
                .iter()
                .map(|&q| (Pauli::Y, QubitId::new(q)))
                .collect(),
        }
    }

    /// Creates a multi-qubit Z operator: Z on each of the given qubits.
    #[must_use]
    pub fn zs(qubits: &[usize]) -> Self {
        Self {
            phase: QuarterPhase::PlusOne,
            paulis: qubits
                .iter()
                .map(|&q| (Pauli::Z, QubitId::new(q)))
                .collect(),
        }
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

    // --- Decomposition methods - non-overlapping X-only, Y, Z-only sets ---

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

    /// Returns the dense string representation with phase prefix.
    ///
    /// Every qubit from 0 to the highest non-identity qubit gets a character.
    /// If `num_qubits` is given, pads with `I` to that width.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_core::{PauliString, QuarterPhase};
    /// use std::str::FromStr;
    ///
    /// let p: PauliString = "X0 Z2".parse().unwrap();
    /// assert_eq!(p.to_dense_str(None), "+XIZ");
    /// assert_eq!(p.to_dense_str(Some(5)), "+XIZII");
    /// ```
    #[must_use]
    pub fn to_dense_str(&self, num_qubits: Option<usize>) -> String {
        let phase_str = match self.phase {
            QuarterPhase::PlusOne => "+",
            QuarterPhase::MinusOne => "-",
            QuarterPhase::PlusI => "+i",
            QuarterPhase::MinusI => "-i",
        };
        format!("{phase_str}{}", self.pauli_str(num_qubits))
    }

    /// Returns the sparse string representation with phase prefix.
    ///
    /// Only non-identity entries are included, as `P<qubit>` tokens separated
    /// by spaces. Qubit indices are sorted.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_core::{PauliString, QuarterPhase};
    /// use std::str::FromStr;
    ///
    /// let p: PauliString = "X0 Z2".parse().unwrap();
    /// assert_eq!(p.to_sparse_str(), "+X0 Z2");
    ///
    /// let p: PauliString = "-i X0 Z2".parse().unwrap();
    /// assert_eq!(p.to_sparse_str(), "-iX0 Z2");
    /// ```
    #[must_use]
    pub fn to_sparse_str(&self) -> String {
        let phase_str = match self.phase {
            QuarterPhase::PlusOne => "+",
            QuarterPhase::MinusOne => "-",
            QuarterPhase::PlusI => "+i",
            QuarterPhase::MinusI => "-i",
        };
        let mut parts = Vec::new();
        let mut sorted_paulis: Vec<_> =
            self.paulis.iter().filter(|(p, _)| *p != Pauli::I).collect();
        sorted_paulis.sort_by_key(|(_, q)| q.index());
        for (pauli, qubit) in &sorted_paulis {
            let c = match pauli {
                Pauli::X => 'X',
                Pauli::Y => 'Y',
                Pauli::Z => 'Z',
                Pauli::I => unreachable!(),
            };
            parts.push(format!("{c}{}", qubit.index()));
        }
        if parts.is_empty() {
            format!("{phase_str}I")
        } else {
            format!("{phase_str}{}", parts.join(" "))
        }
    }

    /// Returns the 2^n x 2^n complex matrix representation of this Pauli string.
    ///
    /// The matrix is the tensor product of single-qubit Pauli matrices (I, X, Y, Z)
    /// multiplied by the global phase. Qubits are ordered from 0 to `num_qubits - 1`.
    ///
    /// Returns a row-major flat vector of length `4^num_qubits` and the dimension `2^num_qubits`.
    ///
    /// This is a lightweight version that returns a flat `Vec` (no nalgebra dependency).
    /// For the `DMatrix` version, use `ToMatrix::to_matrix()` from pecos-quantum.
    ///
    /// # Panics
    ///
    /// Panics if `num_qubits > 12` (matrix would be 4096 x 4096 = 16M entries).
    #[must_use]
    pub fn to_flat_matrix(&self, num_qubits: usize) -> (Vec<num_complex::Complex64>, usize) {
        assert!(
            num_qubits <= 12,
            "to_matrix supports at most 12 qubits, got {num_qubits}"
        );

        let dim = 1usize << num_qubits;
        let phase = self.phase.to_complex();

        // Build the single-qubit matrices for each qubit position
        let single_qubit_matrices: Vec<[num_complex::Complex64; 4]> = (0..num_qubits)
            .map(|q| {
                let c0 = num_complex::Complex64::new(0.0, 0.0);
                let c1 = num_complex::Complex64::new(1.0, 0.0);
                let cm1 = num_complex::Complex64::new(-1.0, 0.0);
                let ci = num_complex::Complex64::new(0.0, 1.0);
                let cmi = num_complex::Complex64::new(0.0, -1.0);

                match self.get(q) {
                    Pauli::I => [c1, c0, c0, c1],  // [[1,0],[0,1]]
                    Pauli::X => [c0, c1, c1, c0],  // [[0,1],[1,0]]
                    Pauli::Y => [c0, cmi, ci, c0], // [[0,-i],[i,0]]
                    Pauli::Z => [c1, c0, c0, cm1], // [[1,0],[0,-1]]
                }
            })
            .collect();

        // Compute the tensor product via index decomposition:
        // For row r and col c, decompose into per-qubit bits and multiply.
        let mut matrix = vec![num_complex::Complex64::new(0.0, 0.0); dim * dim];
        for r in 0..dim {
            for c in 0..dim {
                let mut val = phase;
                for (q, sq_mat) in single_qubit_matrices.iter().enumerate() {
                    let rbit = (r >> (num_qubits - 1 - q)) & 1;
                    let cbit = (c >> (num_qubits - 1 - q)) & 1;
                    val *= sq_mat[rbit * 2 + cbit];
                }
                matrix[r * dim + c] = val;
            }
        }

        (matrix, dim)
    }

    /// Tensor product with another `PauliString` on disjoint qubits.
    ///
    /// Returns `Err` if the two strings share any qubit. For the unchecked
    /// version, use the `&` operator.
    ///
    /// # Errors
    ///
    /// Returns an error message if the qubit sets overlap.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_core::{PauliString, PauliOperator};
    ///
    /// let a = PauliString::x(0) & PauliString::y(1);
    /// let b = PauliString::z(2);
    /// let ab = a.tensor(&b).unwrap();
    /// assert_eq!(ab.weight(), 3);
    /// ```
    pub fn tensor(&self, other: &PauliString) -> Result<PauliString, String> {
        let my_qubits: std::collections::HashSet<usize> = self.qubits().into_iter().collect();
        for q in other.qubits() {
            if my_qubits.contains(&q) {
                return Err(format!("qubit {q} appears in both operands"));
            }
        }
        Ok(self & other)
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

// --- PauliOperator trait implementation ---

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

// --- Display implementation ---

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

// --- FromStr implementation ---

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

/// Parses a phase prefix from a char iterator, returning the phase and
/// consuming the prefix characters.
fn parse_phase_prefix(chars: &mut std::iter::Peekable<impl Iterator<Item = char>>) -> QuarterPhase {
    match chars.peek() {
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
    }
}

impl PauliString {
    /// Parses from dense format where character position = qubit index.
    ///
    /// Format: `[phase]<I|X|Y|Z>...`
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_core::{PauliString, Pauli, QuarterPhase, PauliOperator};
    ///
    /// let p = PauliString::from_dense_str("+iXIZI").unwrap();
    /// assert_eq!(p.phase(), QuarterPhase::PlusI);
    /// assert_eq!(p.get(0), Pauli::X);
    /// assert_eq!(p.get(1), Pauli::I);
    /// assert_eq!(p.get(2), Pauli::Z);
    ///
    /// let p = PauliString::from_dense_str("XYZ").unwrap();
    /// assert_eq!(p.weight(), 3);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if any character after the phase prefix is not a valid
    /// Pauli operator (`I`, `X`, `Y`, `Z`).
    pub fn from_dense_str(s: &str) -> Result<Self, ParsePauliStringError> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(Self::new());
        }

        let mut chars = s.chars().peekable();
        let phase = parse_phase_prefix(&mut chars);

        let mut paulis = Vec::new();
        for (idx, c) in chars.enumerate() {
            let pauli = match c {
                'I' | 'i' | '1' => continue,
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

    /// Parses from sparse format where each token is a Pauli operator followed
    /// by a qubit index.
    ///
    /// Format: `[phase] <P><qubit> [<P><qubit> ...]`
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_core::{PauliString, Pauli, QuarterPhase, PauliOperator};
    ///
    /// let p = PauliString::from_sparse_str("-i X2 Z4 Y7").unwrap();
    /// assert_eq!(p.phase(), QuarterPhase::MinusI);
    /// assert_eq!(p.get(2), Pauli::X);
    /// assert_eq!(p.get(4), Pauli::Z);
    /// assert_eq!(p.get(7), Pauli::Y);
    /// assert_eq!(p.weight(), 3);
    ///
    /// let p = PauliString::from_sparse_str("X0 Z1").unwrap();
    /// assert_eq!(p.get(0), Pauli::X);
    /// assert_eq!(p.get(1), Pauli::Z);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if a token doesn't start with a valid Pauli letter
    /// or doesn't have a valid qubit index.
    pub fn from_sparse_str(s: &str) -> Result<Self, ParsePauliStringError> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(Self::new());
        }

        let mut chars = s.chars().peekable();
        let phase = parse_phase_prefix(&mut chars);

        let remainder: String = chars.collect();
        let remainder = remainder.trim();
        if remainder.is_empty() {
            return Ok(Self::with_phase_and_paulis(phase, Vec::new()));
        }

        let mut paulis = Vec::new();
        let tokens: Vec<&str> = remainder.split_whitespace().collect();
        let mut i = 0;
        while i < tokens.len() {
            let token = tokens[i];
            let mut token_chars = token.chars();
            let pauli_char = token_chars.next().ok_or_else(|| ParsePauliStringError {
                message: "Empty token".to_string(),
            })?;

            let pauli = match pauli_char {
                'X' | 'x' => Pauli::X,
                'Y' | 'y' => Pauli::Y,
                'Z' | 'z' => Pauli::Z,
                c => {
                    return Err(ParsePauliStringError {
                        message: format!("Invalid Pauli character: '{c}'"),
                    });
                }
            };

            let qubit_str: String = token_chars.collect();
            let qubit: usize = if qubit_str.is_empty() {
                // Qubit index may be the next token (e.g., "X 0" instead of "X0")
                i += 1;
                let next = tokens.get(i).ok_or_else(|| ParsePauliStringError {
                    message: format!("Missing qubit index after '{pauli_char}'"),
                })?;
                next.parse().map_err(|_| ParsePauliStringError {
                    message: format!("Invalid qubit index: '{next}'"),
                })?
            } else {
                qubit_str.parse().map_err(|_| ParsePauliStringError {
                    message: format!("Invalid qubit index in token: '{token}'"),
                })?
            };

            paulis.push((pauli, QubitId::new(qubit)));
            i += 1;
        }

        paulis.sort_by_key(|(_, q)| *q);

        Ok(Self { phase, paulis })
    }
}

impl FromStr for PauliString {
    type Err = ParsePauliStringError;

    /// Parses a `PauliString`, auto-detecting the format:
    ///
    /// - **Sparse** (if the body contains digits): `"X0 Z4 Y7"`, `"-i X2 Z4"`
    /// - **Dense** (if the body is all Pauli letters): `"XIZIY"`, `"+iXXZI"`
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_core::{PauliString, Pauli, QuarterPhase, PauliOperator};
    /// use std::str::FromStr;
    ///
    /// // Dense format
    /// let p: PauliString = "XYZ".parse().unwrap();
    /// assert_eq!(p.weight(), 3);
    ///
    /// // Sparse format
    /// let p: PauliString = "-i X2 Z4 Y7".parse().unwrap();
    /// assert_eq!(p.phase(), QuarterPhase::MinusI);
    /// assert_eq!(p.get(2), Pauli::X);
    /// assert_eq!(p.get(4), Pauli::Z);
    /// assert_eq!(p.get(7), Pauli::Y);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(Self::new());
        }

        // After skipping a possible phase prefix, check if the body contains digits.
        // If so, it's sparse format; otherwise, dense.
        let body = s.trim_start_matches(['+', '-']);
        let body = body.strip_prefix('i').unwrap_or(body);
        let body = body.trim();

        if body.chars().any(|c| c.is_ascii_digit()) {
            Self::from_sparse_str(s)
        } else {
            Self::from_dense_str(s)
        }
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

    // --- pauli_str tests ---

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

    // --- to_dense_str / to_sparse_str tests ---

    #[test]
    fn test_to_dense_str_basic() {
        let p: PauliString = "X0 Z2".parse().unwrap();
        assert_eq!(p.to_dense_str(None), "+XIZ");
    }

    #[test]
    fn test_to_dense_str_with_num_qubits() {
        let p: PauliString = "X0 Z2".parse().unwrap();
        assert_eq!(p.to_dense_str(Some(5)), "+XIZII");
    }

    #[test]
    fn test_to_dense_str_with_phase() {
        let p: PauliString = "-i X0 Y1".parse().unwrap();
        assert_eq!(p.to_dense_str(None), "-iXY");
    }

    #[test]
    fn test_to_dense_str_identity() {
        let p = PauliString::identity();
        assert_eq!(p.to_dense_str(None), "+I");
    }

    #[test]
    fn test_to_sparse_str_basic() {
        let p: PauliString = "X0 Z2".parse().unwrap();
        assert_eq!(p.to_sparse_str(), "+X0 Z2");
    }

    #[test]
    fn test_to_sparse_str_with_phase() {
        let p: PauliString = "-i X0 Z2".parse().unwrap();
        assert_eq!(p.to_sparse_str(), "-iX0 Z2");
    }

    #[test]
    fn test_to_sparse_str_identity() {
        let p = PauliString::identity();
        assert_eq!(p.to_sparse_str(), "+I");
    }

    #[test]
    fn test_to_sparse_str_high_qubit() {
        let p = PauliString::x(10000);
        assert_eq!(p.to_sparse_str(), "+X10000");
    }

    #[test]
    fn test_roundtrip_sparse() {
        let original: PauliString = "-i X2 Z4 Y7".parse().unwrap();
        let s = original.to_sparse_str();
        let roundtripped: PauliString = s.parse().unwrap();
        assert_eq!(original.phase(), roundtripped.phase());
        assert_eq!(original.get(2), roundtripped.get(2));
        assert_eq!(original.get(4), roundtripped.get(4));
        assert_eq!(original.get(7), roundtripped.get(7));
    }

    #[test]
    fn test_roundtrip_dense() {
        let original: PauliString = "+iXYZI".parse().unwrap();
        let s = original.to_dense_str(None);
        let roundtripped = PauliString::from_dense_str(&s).unwrap();
        assert_eq!(original.phase(), roundtripped.phase());
        for q in 0..4 {
            assert_eq!(original.get(q), roundtripped.get(q));
        }
    }

    // --- from_str / from_sparse_str / from_dense_str tests ---

    #[test]
    fn test_from_str_dense_auto() {
        // No digits -> auto-detects as dense
        let p: PauliString = "XYZ".parse().unwrap();
        assert_eq!(p.get(0), Pauli::X);
        assert_eq!(p.get(1), Pauli::Y);
        assert_eq!(p.get(2), Pauli::Z);
        assert_eq!(p.phase(), QuarterPhase::PlusOne);
    }

    #[test]
    fn test_from_str_dense_with_phase() {
        let p: PauliString = "+iXXZI".parse().unwrap();
        assert_eq!(p.phase(), QuarterPhase::PlusI);
        assert_eq!(p.get(0), Pauli::X);
        assert_eq!(p.get(2), Pauli::Z);
    }

    #[test]
    fn test_from_str_sparse_auto() {
        // Has digits -> auto-detects as sparse
        let p: PauliString = "X0 Z4 Y7".parse().unwrap();
        assert_eq!(p.get(0), Pauli::X);
        assert_eq!(p.get(4), Pauli::Z);
        assert_eq!(p.get(7), Pauli::Y);
        assert_eq!(p.weight(), 3);
    }

    #[test]
    fn test_from_str_sparse_with_phase() {
        let p: PauliString = "-i X2 Z4 Y7".parse().unwrap();
        assert_eq!(p.phase(), QuarterPhase::MinusI);
        assert_eq!(p.get(2), Pauli::X);
        assert_eq!(p.get(4), Pauli::Z);
        assert_eq!(p.get(7), Pauli::Y);
    }

    #[test]
    fn test_from_sparse_str_single() {
        let p = PauliString::from_sparse_str("X0").unwrap();
        assert_eq!(p.get(0), Pauli::X);
        assert_eq!(p.weight(), 1);
    }

    #[test]
    fn test_from_sparse_str_high_qubit() {
        let p = PauliString::from_sparse_str("X10000").unwrap();
        assert_eq!(p.get(10000), Pauli::X);
        assert_eq!(p.weight(), 1);
    }

    #[test]
    fn test_from_sparse_str_negative_phase() {
        let p = PauliString::from_sparse_str("-X0 Z1").unwrap();
        assert_eq!(p.phase(), QuarterPhase::MinusOne);
        assert_eq!(p.get(0), Pauli::X);
        assert_eq!(p.get(1), Pauli::Z);
    }

    #[test]
    fn test_from_sparse_str_sorted() {
        // Tokens out of order should still produce sorted output
        let p = PauliString::from_sparse_str("Z5 X0 Y2").unwrap();
        let qubits = p.qubits();
        assert_eq!(qubits, vec![0, 2, 5]);
    }

    #[test]
    fn test_from_sparse_str_empty() {
        let p = PauliString::from_sparse_str("").unwrap();
        assert!(p.is_identity());
    }

    #[test]
    fn test_from_sparse_str_phase_only() {
        let p = PauliString::from_sparse_str("-i").unwrap();
        assert_eq!(p.phase(), QuarterPhase::MinusI);
        assert!(p.is_identity());
    }

    #[test]
    fn test_from_dense_str_explicit() {
        let p = PauliString::from_dense_str("ZZI").unwrap();
        assert_eq!(p.get(0), Pauli::Z);
        assert_eq!(p.get(1), Pauli::Z);
        assert_eq!(p.get(2), Pauli::I);
    }

    #[test]
    fn test_from_str_empty() {
        let p: PauliString = "".parse().unwrap();
        assert!(p.is_identity());
    }

    #[test]
    fn test_from_sparse_str_invalid_pauli() {
        assert!(PauliString::from_sparse_str("Q0").is_err());
    }

    #[test]
    fn test_from_sparse_str_invalid_qubit() {
        assert!(PauliString::from_sparse_str("Xabc").is_err());
    }

    #[test]
    fn test_from_sparse_str_spaced() {
        // "X 0" should parse the same as "X0"
        let p1 = PauliString::from_sparse_str("X 0").unwrap();
        let p2 = PauliString::from_sparse_str("X0").unwrap();
        assert_eq!(p1.get(0), p2.get(0));
        assert_eq!(p1.phase(), p2.phase());
    }

    #[test]
    fn test_from_sparse_str_multi_spaced() {
        // "X 0 Z 1" should work
        let p = PauliString::from_sparse_str("X 0 Z 1").unwrap();
        assert_eq!(p.get(0), Pauli::X);
        assert_eq!(p.get(1), Pauli::Z);
    }

    #[test]
    fn test_from_sparse_str_mixed_spaced() {
        // Mix of spaced and compact: "X0 Z 1"
        let p = PauliString::from_sparse_str("X0 Z 1").unwrap();
        assert_eq!(p.get(0), Pauli::X);
        assert_eq!(p.get(1), Pauli::Z);
    }

    #[test]
    fn test_from_sparse_str_trailing_pauli_error() {
        // "X" alone with no qubit should error
        assert!(PauliString::from_sparse_str("X").is_err());
    }

    #[test]
    fn test_from_sparse_str_spaced_with_phase() {
        let p = PauliString::from_sparse_str("-i X 2 Z 4").unwrap();
        assert_eq!(p.phase(), QuarterPhase::MinusI);
        assert_eq!(p.get(2), Pauli::X);
        assert_eq!(p.get(4), Pauli::Z);
    }

    // --- into_pauli_sparse tests ---

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

    // --- into_pauli_bitmap tests ---

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

    // --- set_phase tests ---

    #[test]
    fn test_set_phase() {
        let mut p = PauliString::x(0);
        assert_eq!(p.phase(), QuarterPhase::PlusOne);
        p.set_phase(QuarterPhase::MinusI);
        assert_eq!(p.phase(), QuarterPhase::MinusI);
        // Paulis unchanged
        assert_eq!(p.get(0), Pauli::X);
    }

    #[test]
    fn test_set_phase_all_variants() {
        let mut p = PauliString::z(2);
        for phase in [
            QuarterPhase::PlusOne,
            QuarterPhase::MinusOne,
            QuarterPhase::PlusI,
            QuarterPhase::MinusI,
        ] {
            p.set_phase(phase);
            assert_eq!(p.phase(), phase);
        }
    }

    // --- get edge case tests ---

    #[test]
    fn test_get_returns_identity_for_missing_qubit() {
        let p = PauliString::x(5);
        assert_eq!(p.get(0), Pauli::I);
        assert_eq!(p.get(3), Pauli::I);
        assert_eq!(p.get(100), Pauli::I);
    }

    #[test]
    fn test_get_identity_operator() {
        let p = PauliString::identity();
        assert_eq!(p.get(0), Pauli::I);
        assert_eq!(p.get(999), Pauli::I);
    }

    // --- PauliOperator trait method tests (commutes_with, x_positions, z_positions) ---

    #[test]
    fn test_commutes_with_same_type() {
        // X commutes with X on same qubit
        let x0 = PauliString::x(0);
        assert!(x0.commutes_with(&x0));

        let z0 = PauliString::z(0);
        assert!(z0.commutes_with(&z0));
    }

    #[test]
    fn test_commutes_with_different_qubits() {
        // Any Paulis on different qubits commute
        let x0 = PauliString::x(0);
        let z1 = PauliString::z(1);
        assert!(x0.commutes_with(&z1));
    }

    #[test]
    fn test_anticommutes_with_xz_same_qubit() {
        let x0 = PauliString::x(0);
        let z0 = PauliString::z(0);
        assert!(!x0.commutes_with(&z0));
        assert!(x0.anticommutes_with(&z0));
    }

    #[test]
    fn test_commutes_with_multi_qubit_even_anticommuting() {
        // XZ and ZX: anticommute on q0 (X-Z) and q1 (Z-X) -> even count -> commute
        let xz = PauliString::from_paulis(&[Pauli::X, Pauli::Z]);
        let zx = PauliString::from_paulis(&[Pauli::Z, Pauli::X]);
        assert!(xz.commutes_with(&zx));
    }

    #[test]
    fn test_commutes_with_multi_qubit_odd_anticommuting() {
        // XX and ZI: anticommute only on q0 (X-Z) -> odd count -> anticommute
        let xx = PauliString::from_paulis(&[Pauli::X, Pauli::X]);
        let zi = PauliString::from_paulis(&[Pauli::Z, Pauli::I]);
        assert!(!xx.commutes_with(&zi));
    }

    #[test]
    fn test_commutes_with_identity() {
        // Identity commutes with everything
        let id = PauliString::identity();
        let x0 = PauliString::x(0);
        assert!(id.commutes_with(&x0));
        assert!(x0.commutes_with(&id));
    }

    #[test]
    fn test_x_positions_includes_y() {
        // x_positions returns positions where x-bit is set (X and Y)
        let p = PauliString::from_paulis(&[Pauli::X, Pauli::Y, Pauli::Z, Pauli::I]);
        let x_pos = p.x_positions();
        assert!(x_pos.contains(&0)); // X
        assert!(x_pos.contains(&1)); // Y
        assert!(!x_pos.contains(&2)); // Z - no x-bit
        assert!(!x_pos.contains(&3)); // I
    }

    #[test]
    fn test_z_positions_includes_y() {
        // z_positions returns positions where z-bit is set (Z and Y)
        let p = PauliString::from_paulis(&[Pauli::X, Pauli::Y, Pauli::Z, Pauli::I]);
        let z_pos = p.z_positions();
        assert!(!z_pos.contains(&0)); // X - no z-bit
        assert!(z_pos.contains(&1)); // Y
        assert!(z_pos.contains(&2)); // Z
        assert!(!z_pos.contains(&3)); // I
    }

    #[test]
    fn test_x_positions_empty_for_identity() {
        let p = PauliString::identity();
        assert!(p.x_positions().is_empty());
    }

    #[test]
    fn test_z_positions_empty_for_identity() {
        let p = PauliString::identity();
        assert!(p.z_positions().is_empty());
    }

    // --- Algebraic property tests ---

    #[test]
    fn test_multiply_associativity() {
        // (X * Y) * Z == X * (Y * Z) on same qubit
        let x = PauliString::x(0);
        let y = PauliString::y(0);
        let z = PauliString::z(0);

        let lhs = x.multiply(&y).multiply(&z);
        let rhs = x.multiply(&y.multiply(&z));
        assert_eq!(lhs.phase(), rhs.phase());
        assert_eq!(lhs.get(0), rhs.get(0));
    }

    #[test]
    fn test_multiply_identity_is_neutral() {
        let x = PauliString::x(0);
        let id = PauliString::identity();
        let result = x.multiply(&id);
        assert_eq!(result.phase(), QuarterPhase::PlusOne);
        assert_eq!(result.get(0), Pauli::X);
    }

    #[test]
    fn test_multiply_self_inverse() {
        // P * P = I for any single-qubit Pauli
        for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
            let p = PauliString::from_single(0, pauli);
            let result = p.multiply(&p);
            assert_eq!(
                result.weight(),
                0,
                "{pauli:?} * {pauli:?} should be identity"
            );
            assert_eq!(result.phase(), QuarterPhase::PlusOne);
        }
    }

    #[test]
    fn test_multiply_xx_is_identity() {
        // X * X = I (self-inverse, no Y involvement)
        let x = PauliString::x(0);
        let result = x.multiply(&x);
        assert_eq!(result.weight(), 0);
        assert_eq!(result.phase(), QuarterPhase::PlusOne);
    }

    #[test]
    fn test_multiply_zz_is_identity() {
        // Z * Z = I
        let z = PauliString::z(0);
        let result = z.multiply(&z);
        assert_eq!(result.weight(), 0);
        assert_eq!(result.phase(), QuarterPhase::PlusOne);
    }

    #[test]
    fn test_multiply_xz_result_type() {
        // X * Z on same qubit should give Y (result Pauli is correct even if phase differs)
        let x = PauliString::x(0);
        let z = PauliString::z(0);
        let result = x.multiply(&z);
        assert_eq!(result.get(0), Pauli::Y);
    }

    #[test]
    fn test_multiply_different_qubits() {
        // X(0) * Z(1) should give X(0)Z(1) - no phase change
        let x = PauliString::x(0);
        let z = PauliString::z(1);
        let result = x.multiply(&z);
        assert_eq!(result.get(0), Pauli::X);
        assert_eq!(result.get(1), Pauli::Z);
        assert_eq!(result.phase(), QuarterPhase::PlusOne);
    }

    // --- from_decomposed edge cases ---

    #[test]
    fn test_from_decomposed_empty_is_identity() {
        let p = PauliString::from_decomposed(
            QuarterPhase::PlusOne,
            std::iter::empty::<usize>(),
            std::iter::empty::<usize>(),
            std::iter::empty::<usize>(),
        );
        assert!(p.is_identity());
    }

    // --- into_pauli_bitmap edge case ---

    #[test]
    fn test_into_pauli_bitmap_too_large() {
        let p = PauliString::x(64);
        assert!(p.into_pauli_bitmap().is_err());
    }

    // --- Cross-consistency: PauliOperator::multiply vs algebra * operator ---

    #[test]
    fn test_multiply_vs_algebra_no_y_inputs() {
        // X * Z = -iY (no Y in inputs, both paths should agree)

        let x = PauliString::x(0);
        let z = PauliString::z(0);

        let algebra_result = x.clone() * z.clone();
        let trait_result = x.multiply(&z);

        assert_eq!(
            algebra_result.get(0),
            trait_result.get(0),
            "X*Z Pauli result should agree"
        );
        assert_eq!(
            algebra_result.phase(),
            trait_result.phase(),
            "X*Z phase should agree"
        );
    }

    #[test]
    fn test_multiply_vs_algebra_y_input() {
        // X * Y = iZ (Y in input: algebra * is correct, trait multiply may differ)
        // This test documents whether the two paths are consistent.

        let x = PauliString::x(0);
        let y = PauliString::y(0);

        let algebra_result = x.clone() * y.clone();
        let trait_result = x.multiply(&y);

        assert_eq!(
            algebra_result.get(0),
            trait_result.get(0),
            "X*Y Pauli type should agree between algebra and trait"
        );
        assert_eq!(
            algebra_result.phase(),
            trait_result.phase(),
            "X*Y phase should agree between algebra and trait multiply"
        );
    }

    #[test]
    fn test_multiply_vs_algebra_all_single_qubit_products() {
        // Exhaustive check: every pair of single-qubit Paulis

        for p1 in [Pauli::X, Pauli::Y, Pauli::Z] {
            for p2 in [Pauli::X, Pauli::Y, Pauli::Z] {
                let a = PauliString::from_single(0, p1);
                let b = PauliString::from_single(0, p2);

                let algebra_result = a.clone() * b.clone();
                let trait_result = a.multiply(&b);

                assert_eq!(
                    algebra_result.get(0),
                    trait_result.get(0),
                    "{p1:?}*{p2:?}: Pauli type mismatch"
                );
                assert_eq!(
                    algebra_result.phase(),
                    trait_result.phase(),
                    "{p1:?}*{p2:?}: phase mismatch (algebra={:?}, trait={:?})",
                    algebra_result.phase(),
                    trait_result.phase()
                );
            }
        }
    }

    // --- Roundtrip conversion tests with Y operators ---

    #[test]
    fn test_roundtrip_pauli_sparse_with_y() {
        let original = PauliString::from_paulis_with_phase(
            QuarterPhase::MinusI,
            &[Pauli::Y, Pauli::X, Pauli::Z],
        );
        let sparse = original.clone().into_pauli_sparse().unwrap();
        let roundtripped = PauliString::from(sparse);
        assert_eq!(original.phase(), roundtripped.phase());
        for q in 0..3 {
            assert_eq!(original.get(q), roundtripped.get(q), "qubit {q} mismatch");
        }
    }

    #[test]
    fn test_roundtrip_pauli_bitmap_with_y() {
        let original = PauliString::from_paulis_with_phase(
            QuarterPhase::PlusI,
            &[Pauli::Y, Pauli::Y, Pauli::Z],
        );
        let bitmap = original.clone().into_pauli_bitmap().unwrap();
        let roundtripped = PauliString::try_from(bitmap).unwrap();
        assert_eq!(original.phase(), roundtripped.phase());
        for q in 0..3 {
            assert_eq!(original.get(q), roundtripped.get(q), "qubit {q} mismatch");
        }
    }

    #[test]
    fn test_roundtrip_multiply_with_y_via_sparse() {
        // Verify that roundtripping through PauliSparse gives correct multiply
        let x = PauliString::x(0);
        let y = PauliString::y(0);
        // Use the trait multiply (goes through PauliSparse)
        let result = x.multiply(&y);
        // X * Y = iZ
        assert_eq!(result.get(0), Pauli::Z);
        assert_eq!(result.phase(), QuarterPhase::PlusI);
    }

    // --- into_pauli_bitmap boundary qubit ---

    #[test]
    fn test_into_pauli_bitmap_qubit_63() {
        let p = PauliString::x(63);
        let bitmap = p.into_pauli_bitmap().unwrap();
        assert_eq!(bitmap.weight(), 1);
        assert!(bitmap.x_positions().contains(&63));
    }

    // --- Y in parse/display ---

    #[test]
    fn test_parse_sparse_y_operators() {
        let p: PauliString = "Y0 Y1".parse().unwrap();
        assert_eq!(p.get(0), Pauli::Y);
        assert_eq!(p.get(1), Pauli::Y);
        assert_eq!(p.weight(), 2);
    }

    #[test]
    fn test_parse_dense_y_operators() {
        let p: PauliString = "YYZ".parse().unwrap();
        assert_eq!(p.get(0), Pauli::Y);
        assert_eq!(p.get(1), Pauli::Y);
        assert_eq!(p.get(2), Pauli::Z);
    }

    #[test]
    fn test_roundtrip_sparse_y_with_phase() {
        let original: PauliString = "-Y0 Y1".parse().unwrap();
        let s = original.to_sparse_str();
        let roundtripped: PauliString = s.parse().unwrap();
        assert_eq!(original.phase(), roundtripped.phase());
        assert_eq!(original.get(0), roundtripped.get(0));
        assert_eq!(original.get(1), roundtripped.get(1));
    }

    #[test]
    fn test_roundtrip_dense_y_with_phase() {
        let original =
            PauliString::from_paulis_with_phase(QuarterPhase::MinusI, &[Pauli::Y, Pauli::X]);
        let s = original.to_dense_str(None);
        let roundtripped = PauliString::from_dense_str(&s).unwrap();
        assert_eq!(original.phase(), roundtripped.phase());
        assert_eq!(original.get(0), roundtripped.get(0));
        assert_eq!(original.get(1), roundtripped.get(1));
    }

    // --- to_matrix tests ---

    fn c(re: f64, im: f64) -> num_complex::Complex64 {
        num_complex::Complex64::new(re, im)
    }

    #[test]
    fn test_to_flat_matrix_identity() {
        let id = PauliString::identity();
        let (mat, dim) = id.to_flat_matrix(1);
        assert_eq!(dim, 2);
        assert_eq!(
            mat,
            vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)]
        );
    }

    #[test]
    fn test_to_flat_matrix_x() {
        let x = PauliString::x(0);
        let (mat, dim) = x.to_flat_matrix(1);
        assert_eq!(dim, 2);
        // X = [[0,1],[1,0]]
        assert_eq!(
            mat,
            vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)]
        );
    }

    #[test]
    fn test_to_flat_matrix_y() {
        let y = PauliString::y(0);
        let (mat, dim) = y.to_flat_matrix(1);
        assert_eq!(dim, 2);
        // Y = [[0,-i],[i,0]]
        assert_eq!(
            mat,
            vec![c(0.0, 0.0), c(0.0, -1.0), c(0.0, 1.0), c(0.0, 0.0)]
        );
    }

    #[test]
    fn test_to_flat_matrix_z() {
        let z = PauliString::z(0);
        let (mat, dim) = z.to_flat_matrix(1);
        assert_eq!(dim, 2);
        // Z = [[1,0],[0,-1]]
        assert_eq!(
            mat,
            vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 0.0)]
        );
    }

    #[test]
    fn test_to_flat_matrix_xz_tensor_product() {
        // X tensor Z on 2 qubits
        let xz = PauliString::x(0) & PauliString::z(1);
        let (mat, dim) = xz.to_flat_matrix(2);
        assert_eq!(dim, 4);
        // X (x) Z = [[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]]
        let expected = vec![
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(-1.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(-1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
        ];
        assert_eq!(mat, expected);
    }

    #[test]
    fn test_to_flat_matrix_with_phase() {
        // -X should be -1 times X
        let mx = PauliString::from_paulis_with_phase(QuarterPhase::MinusOne, &[Pauli::X]);
        let (mat, _) = mx.to_flat_matrix(1);
        assert_eq!(
            mat,
            vec![c(0.0, 0.0), c(-1.0, 0.0), c(-1.0, 0.0), c(0.0, 0.0)]
        );
    }

    #[test]
    fn test_to_flat_matrix_hermitian() {
        // All real-phase Pauli strings are Hermitian: M = M†
        for p in [PauliString::x(0), PauliString::y(0), PauliString::z(0)] {
            let (mat, dim) = p.to_flat_matrix(1);
            for r in 0..dim {
                for col in 0..dim {
                    let m_rc = mat[r * dim + col];
                    let m_cr = mat[col * dim + r].conj();
                    assert!((m_rc - m_cr).norm() < 1e-14, "Not Hermitian at ({r},{col})");
                }
            }
        }
    }

    #[test]
    fn test_to_flat_matrix_unitary() {
        // All Pauli matrices square to identity: P^2 = I
        let xz = PauliString::x(0) & PauliString::z(1);
        let (mat, dim) = xz.to_flat_matrix(2);

        // Compute M*M
        let mut product = vec![c(0.0, 0.0); dim * dim];
        for r in 0..dim {
            for col in 0..dim {
                let mut sum = c(0.0, 0.0);
                for k in 0..dim {
                    sum += mat[r * dim + k] * mat[k * dim + col];
                }
                product[r * dim + col] = sum;
            }
        }

        // Should be identity
        for r in 0..dim {
            for col in 0..dim {
                let expected = if r == col { c(1.0, 0.0) } else { c(0.0, 0.0) };
                assert!(
                    (product[r * dim + col] - expected).norm() < 1e-14,
                    "P^2 != I at ({r},{col})"
                );
            }
        }
    }
}
