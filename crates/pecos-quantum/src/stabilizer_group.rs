// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! A Pauli stabilizer group: commuting Pauli strings with [`Sign`] phases.
//!
//! A [`PauliStabilizerGroup`] wraps [`PauliSequence`] with the additional constraints
//! that all generators mutually commute and have [`Sign`] phases (`{+1, -1}`).
//! These constraints are validated at construction time.
//!
//! While [`PauliString`]s carry [`QuarterPhase`] (`{+1, -1, +i, -i}`), stabilizer
//! generators are restricted to the [`Sign`] subset (`{+1, -1}`). A generator with
//! phase +i would violate the stabilizer condition since `(iP)(iP) = -I`, which
//! stabilizes no quantum state.
//!
//! [`PauliString`]: pecos_core::PauliString
//! [`QuarterPhase`]: pecos_core::QuarterPhase
//! [`Sign`]: pecos_core::Sign
//!
//! # Examples
//!
//! ```
//! use pecos_quantum::PauliStabilizerGroup;
//! use pecos_core::pauli::constructors::*;
//!
//! // Repetition code stabilizers
//! let stab = PauliStabilizerGroup::new(vec![
//!     Zs(&[0, 1]),
//!     Zs(&[1, 2]),
//! ], 3).unwrap();
//!
//! assert_eq!(stab.rank(), 2);
//! assert_eq!(stab.num_logical_qubits(), 1);
//! assert!(stab.contains(&Zs(&[0, 2])));
//! ```

use crate::pauli_sequence::{F2Matrix, PauliSequence};
use pecos_core::{Pauli, PauliOperator, PauliString, QuarterPhase, QubitId};
use std::fmt;
use std::str::FromStr;

/// Converts a binary symplectic vector `(x_0..x_{n-1} | z_0..z_{n-1})` to a `PauliString`.
fn symplectic_vec_to_pauli(vec: &[u8], n: usize) -> PauliString {
    let mut paulis = Vec::new();
    for q in 0..n {
        let x = vec[q];
        let z = vec[n + q];
        let pauli = match (x, z) {
            (1, 0) => Pauli::X,
            (0, 1) => Pauli::Z,
            (1, 1) => Pauli::Y,
            _ => continue,
        };
        paulis.push((pauli, QubitId::new(q)));
    }
    PauliString::with_phase_and_paulis(QuarterPhase::PlusOne, paulis)
}

/// Errors that can occur when constructing a [`PauliStabilizerGroup`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PauliStabilizerGroupError {
    /// Generators at indices (i, j) anticommute.
    NonCommuting(usize, usize),
    /// Generator at index i has non-real phase (not +1 or -1).
    NonRealPhase(usize),
}

impl fmt::Display for PauliStabilizerGroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonCommuting(i, j) => {
                write!(f, "generators {i} and {j} anticommute")
            }
            Self::NonRealPhase(i) => {
                write!(
                    f,
                    "generator {i} has non-real phase (stabilizers must have phase +1 or -1)"
                )
            }
        }
    }
}

impl std::error::Error for PauliStabilizerGroupError {}

/// A Pauli stabilizer group: commuting Pauli generators with [`Sign`] phases.
///
/// This is a validated wrapper around [`PauliSequence`] enforcing:
/// - All generators mutually commute (abelian)
/// - All generators have [`Sign`] phase (`{+1, -1}`)
///
/// Each [`PauliString`] carries a [`QuarterPhase`], but this type validates that
/// only the [`Sign`] subset is used. These are the standard requirements for a
/// stabilizer group in QEC: each stabilizer must square to +I (which requires
/// real phase), and all stabilizers must commute to define a consistent code space.
///
/// [`PauliString`]: pecos_core::PauliString
/// [`QuarterPhase`]: pecos_core::QuarterPhase
/// [`Sign`]: pecos_core::Sign
///
/// # Examples
///
/// ```
/// use pecos_quantum::PauliStabilizerGroup;
/// use pecos_core::pauli::constructors::*;
///
/// // 5-qubit code stabilizers: XZZXI, IXZZX, XIXZZ, ZXIXZ
/// let stab = PauliStabilizerGroup::new(vec![
///     X(0) & Z(1) & Z(2) & X(3),   // XZZXI
///     X(1) & Z(2) & Z(3) & X(4),   // IXZZX
///     X(0) & X(2) & Z(3) & Z(4),   // XIXZZ
///     Z(0) & X(1) & X(3) & Z(4),   // ZXIXZ
/// ], 5).unwrap();
///
/// assert_eq!(stab.rank(), 4);
/// assert_eq!(stab.num_logical_qubits(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct PauliStabilizerGroup {
    inner: PauliSequence,
}

impl PauliStabilizerGroup {
    /// Creates a new `PauliStabilizerGroup`, validating that all generators commute
    /// and have real phases.
    ///
    /// # Errors
    ///
    /// Returns [`PauliStabilizerGroupError::NonRealPhase`] if any generator has phase +i or -i.
    /// Returns [`PauliStabilizerGroupError::NonCommuting`] if any pair of generators anticommute.
    pub fn new(
        generators: Vec<PauliString>,
        num_qubits: usize,
    ) -> Result<Self, PauliStabilizerGroupError> {
        // Validate real phases
        for (i, generator) in generators.iter().enumerate() {
            match generator.phase() {
                QuarterPhase::PlusOne | QuarterPhase::MinusOne => {}
                _ => return Err(PauliStabilizerGroupError::NonRealPhase(i)),
            }
        }

        // Validate mutual commutativity
        for i in 0..generators.len() {
            for j in (i + 1)..generators.len() {
                if !generators[i].commutes_with(&generators[j]) {
                    return Err(PauliStabilizerGroupError::NonCommuting(i, j));
                }
            }
        }

        let inner = PauliSequence::new(generators, num_qubits);
        Ok(Self { inner })
    }

    /// Creates a `PauliStabilizerGroup` from string representations.
    ///
    /// # Errors
    ///
    /// Returns an error if any string cannot be parsed, or if the resulting
    /// generators don't form a valid stabilizer group.
    pub fn from_strs(strings: &[&str]) -> Result<Self, Box<dyn std::error::Error>> {
        let coll = PauliSequence::from_strs(strings)?;
        let num_qubits = coll.num_qubits();
        let generators = coll.paulis().to_vec();
        Ok(Self::new(generators, num_qubits)?)
    }

    /// Returns a reference to the underlying [`PauliSequence`].
    #[must_use]
    pub fn as_collection(&self) -> &PauliSequence {
        &self.inner
    }

    /// Returns a reference to the stabilizer generators.
    #[must_use]
    pub fn stabilizers(&self) -> &[PauliString] {
        self.inner.paulis()
    }

    /// Returns the number of generators.
    #[must_use]
    pub fn num_generators(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of physical qubits.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Computes the rank (number of independent generators).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Returns the number of logical qubits: `n - rank`.
    #[must_use]
    pub fn num_logical_qubits(&self) -> usize {
        self.num_qubits().saturating_sub(self.rank())
    }

    /// Returns the code parameters as `[[n, k]]` where n is physical qubits and k is logical qubits.
    #[must_use]
    pub fn code_parameters(&self) -> String {
        let n = self.num_qubits();
        let k = self.num_logical_qubits();
        format!("[[{n}, {k}]]")
    }

    /// Returns `true` if all generators are linearly independent.
    ///
    /// Equivalent to checking `rank() == num_generators()`.
    #[must_use]
    pub fn is_independent(&self) -> bool {
        self.rank() == self.num_generators()
    }

    /// Checks if a Pauli string is in the stabilizer group (ignoring phase).
    #[must_use]
    pub fn contains(&self, pauli: &PauliString) -> bool {
        self.inner.contains(pauli)
    }

    /// Checks if a Pauli string is in the stabilizer group (including phase).
    #[must_use]
    pub fn contains_with_phase(&self, pauli: &PauliString) -> bool {
        self.inner.contains_with_phase(pauli)
    }

    /// Returns the group element formed by multiplying the selected generators.
    ///
    /// Each bit in `mask` selects a generator: bit 0 = generator 0, bit 1 = generator 1, etc.
    /// `mask = 0` returns the identity (product of zero generators).
    ///
    /// # Panics
    ///
    /// Panics if `mask` references a generator index >= `num_generators()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliStabilizerGroup;
    /// use pecos_core::pauli::constructors::*;
    /// use pecos_core::PauliOperator;
    ///
    /// let stab = PauliStabilizerGroup::new(vec![Zs(&[0, 1]), Zs(&[1, 2])], 3).unwrap();
    ///
    /// // Identity (no generators selected)
    /// assert_eq!(stab.element(0b00).weight(), 0);
    ///
    /// // First generator: ZZI
    /// assert_eq!(stab.element(0b01), Zs(&[0, 1]));
    ///
    /// // Both generators: ZZI * IZZ = ZIZ
    /// assert_eq!(stab.element(0b11), Zs(&[0, 2]));
    /// ```
    #[must_use]
    pub fn element(&self, mask: u64) -> PauliString {
        let gens = self.inner.paulis();
        assert!(
            mask < (1u64 << gens.len()),
            "mask {mask} exceeds number of generators ({})",
            gens.len()
        );
        let mut result = PauliString::identity();
        for (i, g) in gens.iter().enumerate() {
            if mask & (1u64 << i) != 0 {
                result = result * g.clone();
            }
        }
        result
    }

    /// Multiplies a Pauli string by the generator at the given index.
    ///
    /// Returns `generators[index] * pauli`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= num_generators()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliStabilizerGroup;
    /// use pecos_core::pauli::constructors::*;
    /// use pecos_core::PauliOperator;
    ///
    /// let stab = PauliStabilizerGroup::new(vec![Zs(&[0, 1]), Zs(&[1, 2])], 3).unwrap();
    ///
    /// // Multiply Z(0) by generator 0 (ZZI): ZII * ZZI = IZI
    /// let result = stab.multiply_by(0, &Z(0));
    /// assert_eq!(result.weight(), 1);
    /// ```
    #[must_use]
    pub fn multiply_by(&self, index: usize, pauli: &PauliString) -> PauliString {
        let gens = self.inner.paulis();
        assert!(
            index < gens.len(),
            "index {index} exceeds number of generators ({})",
            gens.len()
        );
        gens[index].clone() * pauli.clone()
    }

    /// Returns an iterator over all elements of the stabilizer group.
    ///
    /// For `r` generators, this yields `2^r` elements (every product of a subset
    /// of generators, including the identity for the empty subset).
    ///
    /// **Warning**: The group size is exponential in the number of generators.
    /// For large groups, prefer [`contains`](Self::contains) or
    /// [`contains_with_phase`](Self::contains_with_phase) for membership testing.
    ///
    /// # Panics
    ///
    /// Panics if the group has more than 30 generators (2^30 > 10^9 elements).
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliStabilizerGroup;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// let stab = PauliStabilizerGroup::new(vec![Zs(&[0, 1]), Zs(&[1, 2])], 3).unwrap();
    /// let elements: Vec<_> = stab.elements().collect();
    /// // 2 generators -> 2^2 = 4 elements: I, ZZI, IZZ, ZIZ
    /// assert_eq!(elements.len(), 4);
    /// ```
    pub fn elements(&self) -> impl Iterator<Item = PauliString> + '_ {
        let r = self.inner.len();
        assert!(
            r <= 30,
            "elements() would yield 2^{r} items; use contains() for membership testing instead"
        );
        (0u64..(1u64 << r)).map(move |mask| self.element(mask))
    }

    /// Returns the binary symplectic matrix representation.
    #[must_use]
    pub fn to_symplectic_matrix(&self) -> F2Matrix {
        self.inner.to_symplectic_matrix()
    }

    /// Returns the commutation matrix (always all-true for a valid stabilizer group).
    #[must_use]
    pub fn commutation_matrix(&self) -> Vec<Vec<bool>> {
        self.inner.commutation_matrix()
    }

    /// Returns the generators in row-reduced form, removing redundant generators.
    #[must_use]
    pub fn row_reduce(&self) -> PauliSequence {
        self.inner.row_reduce()
    }

    /// Iterates over the stabilizer generators.
    pub fn iter(&self) -> impl Iterator<Item = &PauliString> {
        self.inner.iter()
    }

    /// Returns a basis for the logical operators of the stabilizer code.
    ///
    /// These are Pauli strings that commute with all stabilizers but are not
    /// in the stabilizer group (i.e., they act non-trivially on the code space).
    /// The returned vectors are in binary symplectic form (length `2n`).
    ///
    /// For an `[[n, k]]` code, the logical subspace has dimension `2k`:
    /// `k` logical X operators and `k` logical Z operators.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliStabilizerGroup;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// // Repetition code [[3,1]]: logicals are X_L = XXX, Z_L = Z on any qubit
    /// let stab = PauliStabilizerGroup::new(vec![Zs(&[0, 1]), Zs(&[1, 2])], 3).unwrap();
    /// let logicals = stab.logical_operators();
    /// // 2k = 2 independent logical directions (X_L and Z_L)
    /// assert_eq!(logicals.len(), 2);
    /// ```
    #[must_use]
    pub fn logical_operators(&self) -> Vec<PauliString> {
        let n = self.num_qubits();
        let centralizer_basis = self.inner.centralizer();

        // Row-reduce the stabilizer matrix to get RREF with pivot positions
        let stab_mat = self.inner.to_symplectic_matrix();
        let (stab_rref, stab_pivots) = stab_mat.row_reduce();

        // For each centralizer basis vector, reduce it modulo the stabilizer RREF.
        // If the residual is non-zero, it's a genuine logical operator.
        // Then reduce logicals among themselves to get an independent set.
        let mut logical_vecs: Vec<Vec<u8>> = Vec::new();

        for cvec in &centralizer_basis {
            let mut v = cvec.clone();

            // Reduce using stabilizer RREF
            for (row_idx, &pivot_col) in stab_pivots.iter().enumerate() {
                if v[pivot_col] == 1 {
                    for (col, vi) in v.iter_mut().enumerate() {
                        *vi ^= stab_rref.row(row_idx)[col];
                    }
                }
            }

            // If residual is non-zero, this is a logical direction
            if v.iter().any(|&b| b != 0) {
                logical_vecs.push(v);
            }
        }

        // Row-reduce the logical vectors to get an independent set
        if logical_vecs.len() > 1 {
            let mut log_mat = F2Matrix::zeros(logical_vecs.len(), 2 * n);
            for (i, v) in logical_vecs.iter().enumerate() {
                log_mat.rows[i].clone_from(v);
            }
            let (reduced, _) = log_mat.row_reduce();
            logical_vecs = (0..reduced.num_rows())
                .map(|i| reduced.row(i).to_vec())
                .filter(|r| r.iter().any(|&b| b != 0))
                .collect();
        }

        logical_vecs
            .iter()
            .map(|v| symplectic_vec_to_pauli(v, n))
            .collect()
    }

    /// Computes the code distance for small codes.
    ///
    /// The distance is the minimum weight of a non-trivial logical operator
    /// (a Pauli that commutes with all stabilizers but is not in the stabilizer group).
    ///
    /// Returns `None` if there are no logical qubits (k = 0).
    ///
    /// **Complexity**: O(2^k * 2^r) where k = number of logical operators and
    /// r = rank. Only suitable for small codes.
    ///
    /// # Panics
    ///
    /// Panics if `k + rank > 30` to prevent accidental exponential blowup.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliStabilizerGroup;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// // Repetition code [[3,1,3]]: distance 3 (logical X = XXX)
    /// let stab = PauliStabilizerGroup::new(vec![Zs(&[0, 1]), Zs(&[1, 2])], 3).unwrap();
    /// assert_eq!(stab.distance(), Some(1)); // logical Z = Z on any single qubit
    /// ```
    #[must_use]
    pub fn distance(&self) -> Option<usize> {
        let logicals = self.logical_operators();
        if logicals.is_empty() {
            return None;
        }

        let n = self.num_qubits();
        let k = logicals.len();

        // Get the stabilizer generators in reduced form for coset optimization
        let reduced = self.row_reduce();
        let stab_paulis: Vec<&PauliString> = reduced.paulis().iter().collect();
        let r = stab_paulis.len();

        assert!(
            k + r <= 30,
            "distance() is O(2^(k+r)) and would enumerate 2^{} combinations; \
             use a different algorithm for large codes",
            k + r,
        );

        let mut min_weight = n + 1; // upper bound

        // For each non-zero combination of logical operators...
        for logical_mask in 1u64..(1u64 << k) {
            // Build the logical operator from combination of basis logicals
            let mut logical = PauliString::identity();
            for (i, log) in logicals.iter().enumerate() {
                if logical_mask & (1u64 << i) != 0 {
                    logical = logical * log.clone();
                }
            }

            // Try all combinations of stabilizers to minimize weight
            // (multiply by stabilizer elements to find minimum weight representative)
            for stab_mask in 0u64..(1u64 << r) {
                let mut candidate = logical.clone();
                for (i, stab) in stab_paulis.iter().enumerate() {
                    if stab_mask & (1u64 << i) != 0 {
                        candidate = candidate * (*stab).clone();
                    }
                }
                let w = candidate.weight();
                if w < min_weight {
                    min_weight = w;
                }
            }
        }

        Some(min_weight)
    }

    /// Computes the syndrome of an error Pauli against the stabilizer generators.
    ///
    /// Returns a binary vector of length `num_generators()` where entry `i` is `true`
    /// if the error anticommutes with generator `i` (i.e., would trigger that detector).
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliStabilizerGroup;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// // Repetition code: ZZI, IZZ on 3 qubits
    /// let stab = PauliStabilizerGroup::new(vec![Zs(&[0, 1]), Zs(&[1, 2])], 3).unwrap();
    ///
    /// // X error on qubit 0 triggers first stabilizer only
    /// assert_eq!(stab.syndrome(&X(0)), vec![true, false]);
    ///
    /// // X error on qubit 1 triggers both stabilizers
    /// assert_eq!(stab.syndrome(&X(1)), vec![true, true]);
    ///
    /// // Z error commutes with all Z-stabilizers
    /// assert_eq!(stab.syndrome(&Z(0)), vec![false, false]);
    /// ```
    #[must_use]
    pub fn syndrome(&self, error: &PauliString) -> Vec<bool> {
        self.inner
            .paulis()
            .iter()
            .map(|stab| !stab.commutes_with(error))
            .collect()
    }

    /// Returns the dense string representation, one stabilizer per line.
    ///
    /// Delegates to [`PauliSequence::to_dense_str`].
    #[must_use]
    pub fn to_dense_str(&self) -> String {
        self.inner.to_dense_str()
    }

    /// Returns the sparse string representation, one stabilizer per line.
    ///
    /// Delegates to [`PauliSequence::to_sparse_str`].
    #[must_use]
    pub fn to_sparse_str(&self) -> String {
        self.inner.to_sparse_str()
    }
}

impl FromStr for PauliStabilizerGroup {
    type Err = Box<dyn std::error::Error>;

    /// Parses a `PauliStabilizerGroup` from newline-delimited Pauli strings.
    ///
    /// Each line is parsed via [`PauliString::from_str`]. The resulting generators
    /// are validated for mutual commutativity and real phases.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliStabilizerGroup;
    /// use std::str::FromStr;
    ///
    /// let stab: PauliStabilizerGroup = "ZZI\nIZZ".parse().unwrap();
    /// assert_eq!(stab.rank(), 2);
    /// assert_eq!(stab.num_logical_qubits(), 1);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let seq: PauliSequence = s.parse()?;
        let num_qubits = seq.num_qubits();
        let generators = seq.paulis().to_vec();
        Ok(Self::new(generators, num_qubits)?)
    }
}

impl fmt::Display for PauliStabilizerGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::pauli::constructors::*;

    #[test]
    fn test_repetition_code() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        assert_eq!(stab.rank(), 2);
        assert_eq!(stab.num_logical_qubits(), 1);
        assert_eq!(stab.code_parameters(), "[[3, 1]]");
    }

    #[test]
    fn test_steane_code() {
        let stab = PauliStabilizerGroup::new(
            vec![
                Xs([0, 2, 4, 6]),
                Xs([1, 2, 5, 6]),
                Xs([3, 4, 5, 6]),
                Zs([0, 2, 4, 6]),
                Zs([1, 2, 5, 6]),
                Zs([3, 4, 5, 6]),
            ],
            7,
        )
        .unwrap();
        assert_eq!(stab.rank(), 6);
        assert_eq!(stab.num_logical_qubits(), 1);
        assert_eq!(stab.code_parameters(), "[[7, 1]]");
    }

    #[test]
    fn test_rejects_non_commuting() {
        let result = PauliStabilizerGroup::new(vec![X(0), Z(0)], 1);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            PauliStabilizerGroupError::NonCommuting(0, 1)
        );
    }

    #[test]
    fn test_rejects_imaginary_phase() {
        use pecos_core::pauli::algebra::i;
        let result = PauliStabilizerGroup::new(vec![i * X(0)], 1);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            PauliStabilizerGroupError::NonRealPhase(0)
        );
    }

    #[test]
    fn test_accepts_negative_phase() {
        // -ZZ is a valid stabilizer (phase is -1, which is real)
        let stab = PauliStabilizerGroup::new(vec![-Zs([0, 1])], 2);
        assert!(stab.is_ok());
    }

    #[test]
    fn test_contains() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        assert!(stab.contains(&Zs([0, 2])));
        assert!(!stab.contains(&X(0)));
    }

    #[test]
    fn test_from_strs() {
        let stab = PauliStabilizerGroup::from_strs(&["ZZI", "IZZ"]).unwrap();
        assert_eq!(stab.rank(), 2);
        assert_eq!(stab.num_logical_qubits(), 1);
    }

    #[test]
    fn test_display() {
        let stab = PauliStabilizerGroup::from_strs(&["ZZI", "IZZ"]).unwrap();
        let s = format!("{stab}");
        assert_eq!(s, "ZZI\nIZZ");
    }

    #[test]
    fn test_five_qubit_code() {
        // [[5,1,3]] code: XZZXI, IXZZX, XIXZZ, ZXIXZ
        let stab = PauliStabilizerGroup::new(
            vec![
                X(0) & Z(1) & Z(2) & X(3), // XZZXI
                X(1) & Z(2) & Z(3) & X(4), // IXZZX
                X(0) & X(2) & Z(3) & Z(4), // XIXZZ
                Z(0) & X(1) & X(3) & Z(4), // ZXIXZ
            ],
            5,
        )
        .unwrap();
        assert_eq!(stab.rank(), 4);
        assert_eq!(stab.num_logical_qubits(), 1);
        assert_eq!(stab.code_parameters(), "[[5, 1]]");
    }

    // ========================================================================
    // FromStr / to_dense_str / to_sparse_str tests
    // ========================================================================

    #[test]
    fn test_from_str_dense() {
        let stab: PauliStabilizerGroup = "ZZI\nIZZ".parse().unwrap();
        assert_eq!(stab.rank(), 2);
        assert_eq!(stab.num_logical_qubits(), 1);
    }

    #[test]
    fn test_from_str_sparse() {
        let stab: PauliStabilizerGroup = "Z0 Z1\nZ1 Z2".parse().unwrap();
        assert_eq!(stab.rank(), 2);
    }

    #[test]
    fn test_from_str_rejects_non_commuting() {
        let result: Result<PauliStabilizerGroup, _> = "X0\nZ0".parse();
        assert!(result.is_err());
    }

    #[test]
    fn test_to_dense_str() {
        let stab = PauliStabilizerGroup::from_strs(&["ZZI", "IZZ"]).unwrap();
        assert_eq!(stab.to_dense_str(), "ZZI\nIZZ");
    }

    #[test]
    fn test_to_sparse_str() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        assert_eq!(stab.to_sparse_str(), "+Z0 Z1\n+Z1 Z2");
    }

    #[test]
    fn test_roundtrip() {
        let original = PauliStabilizerGroup::from_strs(&["ZZI", "IZZ"]).unwrap();
        let s = original.to_dense_str();
        let roundtripped: PauliStabilizerGroup = s.parse().unwrap();
        assert_eq!(roundtripped.rank(), original.rank());
        assert_eq!(roundtripped.num_qubits(), original.num_qubits());
    }

    // ========================================================================
    // New feature tests
    // ========================================================================

    #[test]
    fn test_is_independent() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        assert!(stab.is_independent());

        // Add a redundant generator: ZIZ = ZZI * IZZ
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2]), Zs([0, 2])], 3).unwrap();
        assert!(!stab.is_independent());
    }

    #[test]
    fn test_syndrome_repetition_code() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();

        // X error on qubit 0: anticommutes with ZZI only
        assert_eq!(stab.syndrome(&X(0)), vec![true, false]);

        // X error on qubit 1: anticommutes with both
        assert_eq!(stab.syndrome(&X(1)), vec![true, true]);

        // X error on qubit 2: anticommutes with IZZ only
        assert_eq!(stab.syndrome(&X(2)), vec![false, true]);

        // Z error: commutes with all Z-stabilizers
        assert_eq!(stab.syndrome(&Z(0)), vec![false, false]);
        assert_eq!(stab.syndrome(&Z(1)), vec![false, false]);
    }

    #[test]
    fn test_logical_operators_repetition_code() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        let logicals = stab.logical_operators();
        // [[3,1]] code: 2k = 2 independent logical operators
        assert_eq!(logicals.len(), 2);
    }

    #[test]
    fn test_logical_operators_steane_code() {
        let stab = PauliStabilizerGroup::new(
            vec![
                Xs([0, 2, 4, 6]),
                Xs([1, 2, 5, 6]),
                Xs([3, 4, 5, 6]),
                Zs([0, 2, 4, 6]),
                Zs([1, 2, 5, 6]),
                Zs([3, 4, 5, 6]),
            ],
            7,
        )
        .unwrap();
        let logicals = stab.logical_operators();
        // [[7,1]] code: 2k = 2 logical operators
        assert_eq!(logicals.len(), 2);
    }

    #[test]
    fn test_distance_repetition_code() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        // Minimum weight logical: single Z (weight 1)
        assert_eq!(stab.distance(), Some(1));
    }

    #[test]
    fn test_distance_steane_code() {
        let stab = PauliStabilizerGroup::new(
            vec![
                Xs([0, 2, 4, 6]),
                Xs([1, 2, 5, 6]),
                Xs([3, 4, 5, 6]),
                Zs([0, 2, 4, 6]),
                Zs([1, 2, 5, 6]),
                Zs([3, 4, 5, 6]),
            ],
            7,
        )
        .unwrap();
        // [[7,1,3]] Steane code: distance 3
        assert_eq!(stab.distance(), Some(3));
    }

    #[test]
    fn test_distance_no_logicals() {
        // n qubits, n stabilizers, k=0 -> no distance
        let stab = PauliStabilizerGroup::new(vec![Z(0), Z(1)], 2).unwrap();
        assert_eq!(stab.distance(), None);
    }

    #[test]
    fn test_distance_five_qubit_code() {
        let stab = PauliStabilizerGroup::new(
            vec![
                X(0) & Z(1) & Z(2) & X(3),
                X(1) & Z(2) & Z(3) & X(4),
                X(0) & X(2) & Z(3) & Z(4),
                Z(0) & X(1) & X(3) & Z(4),
            ],
            5,
        )
        .unwrap();
        // [[5,1,3]] code: distance 3
        assert_eq!(stab.distance(), Some(3));
    }

    // ========================================================================
    // Group element tests
    // ========================================================================

    #[test]
    fn test_element_identity() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        let id = stab.element(0b00);
        assert!(id.is_identity());
    }

    #[test]
    fn test_element_single_generator() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        assert_eq!(stab.element(0b01), Zs([0, 1]));
        assert_eq!(stab.element(0b10), Zs([1, 2]));
    }

    #[test]
    fn test_element_product() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        // ZZI * IZZ = ZIZ
        let product = stab.element(0b11);
        assert_eq!(product.get(0), Pauli::Z);
        assert_eq!(product.get(1), Pauli::I);
        assert_eq!(product.get(2), Pauli::Z);
    }

    #[test]
    fn test_multiply_by() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        // generator[0] (ZZI) * Z(0) = IZI
        let result = stab.multiply_by(0, &Z(0));
        assert_eq!(result.get(0), Pauli::I);
        assert_eq!(result.get(1), Pauli::Z);
        assert_eq!(result.weight(), 1);
    }

    #[test]
    fn test_multiply_by_identity() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1])], 2).unwrap();
        let id = PauliString::identity();
        let result = stab.multiply_by(0, &id);
        assert_eq!(result, Zs([0, 1]));
    }

    #[test]
    fn test_elements_count() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        let elements: Vec<_> = stab.elements().collect();
        assert_eq!(elements.len(), 4); // 2^2
    }

    #[test]
    fn test_elements_all_in_group() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        for elem in stab.elements() {
            assert!(stab.contains_with_phase(&elem));
        }
    }

    #[test]
    fn test_elements_contains_identity() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1])], 2).unwrap();
        let elements: Vec<_> = stab.elements().collect();
        assert!(elements.iter().any(pecos_core::PauliString::is_identity));
    }

    #[test]
    fn test_elements_single_generator() {
        let stab = PauliStabilizerGroup::new(vec![Z(0)], 1).unwrap();
        let elements: Vec<_> = stab.elements().collect();
        assert_eq!(elements.len(), 2); // I, Z
        assert!(elements.iter().any(pecos_core::PauliString::is_identity));
        assert!(elements.iter().any(|e| *e == Z(0)));
    }

    #[test]
    fn test_element_with_negative_phase() {
        // -ZZ is a valid stabilizer generator
        let stab = PauliStabilizerGroup::new(vec![-Zs([0, 1])], 2).unwrap();
        let elem = stab.element(0b1);
        assert_eq!(elem.phase(), QuarterPhase::MinusOne);
        assert_eq!(elem.weight(), 2);

        // Product with itself: (-ZZ)(-ZZ) = +II = identity
        let elements: Vec<_> = stab.elements().collect();
        assert_eq!(elements.len(), 2);
        assert!(elements.iter().any(pecos_core::PauliString::is_identity));
    }

    #[test]
    fn test_syndrome_steane_code() {
        let stab = PauliStabilizerGroup::new(
            vec![
                Xs([0, 2, 4, 6]),
                Xs([1, 2, 5, 6]),
                Xs([3, 4, 5, 6]),
                Zs([0, 2, 4, 6]),
                Zs([1, 2, 5, 6]),
                Zs([3, 4, 5, 6]),
            ],
            7,
        )
        .unwrap();

        // Z error on qubit 0: anticommutes with X stabilizers that include qubit 0
        let syn = stab.syndrome(&Z(0));
        assert!(syn[0]); // X on {0,2,4,6} -- includes qubit 0
        assert!(!syn[1]); // X on {1,2,5,6} -- doesn't include qubit 0
        assert!(!syn[2]); // X on {3,4,5,6} -- doesn't include qubit 0
        // Z stabilizers commute with Z errors
        assert!(!syn[3]);
        assert!(!syn[4]);
        assert!(!syn[5]);
    }

    #[test]
    fn test_syndrome_y_error() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        // Y error on qubit 1: Y anticommutes with Z, so triggers both Z-stabilizers
        let syn = stab.syndrome(&Y(1));
        assert_eq!(syn, vec![true, true]);
    }

    #[test]
    fn test_syndrome_multi_qubit_error() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        // X error on qubits 0 and 2: triggers both stabilizers
        let error = X(0) & X(2);
        let syn = stab.syndrome(&error);
        assert_eq!(syn, vec![true, true]);
    }

    #[test]
    fn test_syndrome_stabilizer_element() {
        // Applying a stabilizer element should give trivial syndrome
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        let syn = stab.syndrome(&Zs([0, 1]));
        assert_eq!(syn, vec![false, false]);
    }

    #[test]
    fn test_logical_operators_five_qubit_code() {
        let stab = PauliStabilizerGroup::new(
            vec![
                X(0) & Z(1) & Z(2) & X(3),
                X(1) & Z(2) & Z(3) & X(4),
                X(0) & X(2) & Z(3) & Z(4),
                Z(0) & X(1) & X(3) & Z(4),
            ],
            5,
        )
        .unwrap();
        let logicals = stab.logical_operators();
        // [[5,1]]: 2k = 2 logical operators
        assert_eq!(logicals.len(), 2);
        // Each logical should commute with all stabilizers
        for l in &logicals {
            for s in stab.iter() {
                assert!(l.commutes_with(s));
            }
        }
        // Logicals should NOT be in the stabilizer group
        for l in &logicals {
            assert!(!stab.contains(l));
        }
    }

    #[test]
    fn test_logical_operators_commute_with_stabilizers() {
        // General property: every logical commutes with every stabilizer
        let stab = PauliStabilizerGroup::new(
            vec![
                Xs([0, 2, 4, 6]),
                Xs([1, 2, 5, 6]),
                Xs([3, 4, 5, 6]),
                Zs([0, 2, 4, 6]),
                Zs([1, 2, 5, 6]),
                Zs([3, 4, 5, 6]),
            ],
            7,
        )
        .unwrap();
        for l in stab.logical_operators() {
            for s in stab.iter() {
                assert!(
                    l.commutes_with(s),
                    "logical {} anticommutes with stabilizer {}",
                    l.to_sparse_str(),
                    s.to_sparse_str()
                );
            }
        }
    }

    #[test]
    fn test_distance_with_redundant_generators() {
        // Same code, but with a redundant generator
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2]), Zs([0, 2])], 3).unwrap();
        assert!(!stab.is_independent());
        assert_eq!(stab.distance(), Some(1));
    }

    #[test]
    fn test_elements_closure() {
        // For a valid stabilizer group, the product of any two elements
        // should also be an element
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        let elements: Vec<_> = stab.elements().collect();
        for a in &elements {
            for b in &elements {
                let product = a.clone() * b.clone();
                assert!(
                    stab.contains_with_phase(&product),
                    "{} * {} = {} not in group",
                    a.to_sparse_str(),
                    b.to_sparse_str(),
                    product.to_sparse_str()
                );
            }
        }
    }

    #[test]
    fn test_elements_self_inverse() {
        // Every stabilizer element squares to identity
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        for elem in stab.elements() {
            let squared = elem.clone() * elem;
            assert!(
                squared.is_identity(),
                "{} does not square to I",
                squared.to_sparse_str()
            );
        }
    }

    #[test]
    fn test_multiply_by_non_group_element() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1])], 2).unwrap();
        // Multiply X(0) by generator ZZ: ZZ * X(0) = -Y(0)Z(1) (different from stabilizer)
        let result = stab.multiply_by(0, &X(0));
        assert!(!stab.contains(&result));
    }

    #[test]
    fn test_empty_stabilizer_group() {
        let stab = PauliStabilizerGroup::new(vec![], 3).unwrap();
        assert_eq!(stab.rank(), 0);
        assert_eq!(stab.num_logical_qubits(), 3);
        assert_eq!(stab.elements().count(), 1); // just identity
    }

    #[test]
    #[should_panic(expected = "exceeds number of generators")]
    fn test_element_out_of_range_panics() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1])], 2).unwrap();
        // mask 0b11 = 3 but only 1 generator, so bit 1 is out of range
        let _ = stab.element(0b11);
    }

    #[test]
    fn test_distance_full_rank() {
        // [[2,0]] code: 2 generators on 2 qubits, k=0, no logicals
        let stab = PauliStabilizerGroup::new(vec![Z(0), Z(1)], 2).unwrap();
        assert_eq!(stab.num_logical_qubits(), 0);
        assert_eq!(stab.distance(), None);
    }

    #[test]
    fn test_syndrome_identity_error() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        // Identity error has zero syndrome
        let id = PauliString::identity();
        let s = stab.syndrome(&id);
        assert!(s.iter().all(|&b| !b), "identity should have zero syndrome");
    }

    #[test]
    fn test_logical_operators_anticommute_with_each_other() {
        // For [[3,1]] repetition code, logical X and Z should anticommute
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])], 3).unwrap();
        let logicals = stab.logical_operators();
        // Should have 2 logicals (X_L and Z_L)
        assert_eq!(logicals.len(), 2);
        // At least one pair should anticommute (the X_L and Z_L)
        let mut found_anticommuting = false;
        for i in 0..logicals.len() {
            for j in (i + 1)..logicals.len() {
                if logicals[i].anticommutes_with(&logicals[j]) {
                    found_anticommuting = true;
                }
            }
        }
        assert!(found_anticommuting, "logical X and Z should anticommute");
    }
}
