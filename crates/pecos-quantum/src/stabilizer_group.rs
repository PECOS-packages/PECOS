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
//! A [`PauliStabilizerGroup`] wraps [`PauliGroup`] with the additional constraint
//! that all generators have [`Sign`] phases (`{+1, -1}`).
//! The commutativity constraint is inherited from [`PauliGroup`].
//!
//! [`PauliGroup`]: crate::PauliGroup
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
//! ]).unwrap();
//!
//! assert_eq!(stab.rank(), 2);
//! assert!(stab.contains(&Zs(&[0, 2])));
//! ```

use crate::pauli_group::{PauliGroup, PauliGroupError};
use crate::pauli_sequence::{F2Matrix, PauliSequence};
use crate::pauli_set::PauliSet;
use pecos_core::{PauliString, QuarterPhase};
use std::fmt;
use std::str::FromStr;

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
/// This is a validated wrapper around [`PauliGroup`] adding the constraint that
/// all generators have [`Sign`] phase (`{+1, -1}`). The commutativity constraint
/// is inherited from [`PauliGroup`].
///
/// These are the standard requirements for a stabilizer group in QEC: each
/// stabilizer must square to `+I` (which requires real phase), and all
/// stabilizers must commute to define a consistent code space.
///
/// [`PauliGroup`]: crate::PauliGroup
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
/// ]).unwrap();
///
/// assert_eq!(stab.rank(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct PauliStabilizerGroup {
    pub(crate) inner: PauliGroup,
}

impl PauliStabilizerGroup {
    /// Creates a new `PauliStabilizerGroup`, validating that all generators commute
    /// and have real phases.
    ///
    /// # Errors
    ///
    /// Returns [`PauliStabilizerGroupError::NonRealPhase`] if any generator has phase +i or -i.
    /// Returns [`PauliStabilizerGroupError::NonCommuting`] if any pair of generators anticommute.
    pub fn new(generators: Vec<PauliString>) -> Result<Self, PauliStabilizerGroupError> {
        // Validate real phases
        for (i, generator) in generators.iter().enumerate() {
            match generator.phase() {
                QuarterPhase::PlusOne | QuarterPhase::MinusOne => {}
                _ => return Err(PauliStabilizerGroupError::NonRealPhase(i)),
            }
        }

        // PauliGroup::new validates mutual commutativity
        let inner = PauliGroup::new(generators).map_err(|e| match e {
            PauliGroupError::NonCommuting(a, b) => PauliStabilizerGroupError::NonCommuting(a, b),
        })?;
        Ok(Self { inner })
    }

    /// Creates a `PauliStabilizerGroup` without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the generators mutually commute and have
    /// real phases. This is intended for internal use where the generators
    /// are known to be valid (e.g., extracted from a simulator tableau).
    #[must_use]
    pub fn from_generators_unchecked(generators: Vec<PauliString>) -> Self {
        Self {
            inner: PauliGroup::from_generators_unchecked(generators),
        }
    }

    /// Creates a `PauliStabilizerGroup` from string representations.
    ///
    /// # Errors
    ///
    /// Returns an error if any string cannot be parsed, or if the resulting
    /// generators don't form a valid stabilizer group.
    pub fn from_strs(strings: &[&str]) -> Result<Self, Box<dyn std::error::Error>> {
        let coll = PauliSequence::from_strs(strings)?;
        let generators = coll.paulis().to_vec();
        Ok(Self::new(generators)?)
    }

    /// Creates a `PauliStabilizerGroup` from a [`PauliSet`], row-reducing to
    /// independent generators.
    ///
    /// This validates commutativity and real phases, then reduces to a minimal
    /// generating set.
    ///
    /// # Errors
    ///
    /// Returns [`PauliStabilizerGroupError::NonCommuting`] if any pair anticommute.
    /// Returns [`PauliStabilizerGroupError::NonRealPhase`] if any reduced generator
    /// has a non-real phase.
    pub fn try_from_set(set: &PauliSet) -> Result<Self, PauliStabilizerGroupError> {
        Self::try_from(set.to_sequence())
    }

    /// Returns a reference to the underlying [`PauliGroup`].
    #[must_use]
    pub fn as_group(&self) -> &PauliGroup {
        &self.inner
    }

    /// Returns a reference to the underlying [`PauliSequence`].
    #[must_use]
    pub fn as_collection(&self) -> &PauliSequence {
        self.inner.as_sequence()
    }

    /// Returns a reference to the stabilizer generators.
    #[must_use]
    pub fn stabilizers(&self) -> &[PauliString] {
        self.inner.generators()
    }

    /// Returns the number of generators.
    #[must_use]
    pub fn num_generators(&self) -> usize {
        self.inner.num_generators()
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
    /// let stab = PauliStabilizerGroup::new(vec![Zs(&[0, 1]), Zs(&[1, 2])]).unwrap();
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
        let n = self.num_generators();
        assert!(
            mask < (1u64 << n),
            "mask {mask} exceeds number of generators ({n})"
        );
        let exponents: Vec<u32> = (0..n)
            .map(|idx| u32::from(mask & (1u64 << idx) != 0))
            .collect();
        self.inner.element(&exponents)
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
    /// let stab = PauliStabilizerGroup::new(vec![Zs(&[0, 1]), Zs(&[1, 2])]).unwrap();
    ///
    /// // Multiply Z(0) by generator 0 (ZZI): ZII * ZZI = IZI
    /// let result = stab.multiply_by(0, &Z(0));
    /// assert_eq!(result.weight(), 1);
    /// ```
    #[must_use]
    pub fn multiply_by(&self, index: usize, pauli: &PauliString) -> PauliString {
        self.inner.multiply_by(index, pauli)
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
    /// let stab = PauliStabilizerGroup::new(vec![Zs(&[0, 1]), Zs(&[1, 2])]).unwrap();
    /// let elements: Vec<_> = stab.elements().collect();
    /// // 2 generators -> 2^2 = 4 elements: I, ZZI, IZZ, ZIZ
    /// assert_eq!(elements.len(), 4);
    /// ```
    pub fn elements(&self) -> impl Iterator<Item = PauliString> + '_ {
        self.inner.elements()
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

    /// Transforms all generators by a Clifford gate: each `g_i` -> `C g_i C†`.
    ///
    /// Returns a new `PauliStabilizerGroup` with the transformed generators.
    /// Clifford gates preserve commutation relations and real phases, so the
    /// result is always a valid stabilizer group.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliStabilizerGroup;
    /// use pecos_core::pauli::constructors::*;
    /// use pecos_core::clifford_rep::CliffordRep;
    ///
    /// // Repetition code stabilizers: ZZ_, _ZZ
    /// let stab = PauliStabilizerGroup::new(vec![
    ///     Zs([0, 1]),
    ///     Zs([1, 2]),
    /// ]).unwrap();
    ///
    /// // Apply Hadamard to all qubits: Z -> X
    /// let h_all = CliffordRep::h(0)
    ///     .compose(&CliffordRep::h(1))
    ///     .compose(&CliffordRep::h(2));
    /// let transformed = stab.apply_clifford(&h_all);
    ///
    /// // Now we should have XX_, _XX stabilizers
    /// assert!(transformed.contains(&Xs([0, 1])));
    /// assert!(transformed.contains(&Xs([1, 2])));
    /// ```
    #[must_use]
    pub fn apply_clifford(
        &self,
        clifford: &pecos_core::clifford_rep::CliffordRep,
    ) -> PauliStabilizerGroup {
        // Clifford conjugation preserves commutation and real phases.
        PauliStabilizerGroup {
            inner: self.inner.apply_clifford(clifford),
        }
    }

    // ========================================================================
    // Mutation methods
    // ========================================================================

    /// Adds a generator to the stabilizer group.
    ///
    /// The new generator must commute with all existing generators and have
    /// a real phase (+1 or -1).
    ///
    /// # Errors
    ///
    /// Returns an error if the generator has non-real phase or anticommutes
    /// with any existing generator.
    pub fn add_generator(
        &mut self,
        generator: PauliString,
    ) -> Result<(), PauliStabilizerGroupError> {
        match generator.phase() {
            QuarterPhase::PlusOne | QuarterPhase::MinusOne => {}
            _ => {
                return Err(PauliStabilizerGroupError::NonRealPhase(
                    self.num_generators(),
                ));
            }
        }

        self.inner.add_generator(generator).map_err(|e| match e {
            PauliGroupError::NonCommuting(a, b) => PauliStabilizerGroupError::NonCommuting(a, b),
        })
    }

    /// Removes the generator at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= num_generators()`.
    pub fn remove_generator(&mut self, index: usize) -> PauliString {
        self.inner.remove_generator(index)
    }

    /// Merges another stabilizer group into this one.
    ///
    /// All generators from `other` must commute with all generators in `self`.
    /// The resulting group acts on `max(self.num_qubits(), other.num_qubits())` qubits.
    ///
    /// This is useful for lattice surgery: merging two code blocks by adding
    /// joint stabilizers.
    ///
    /// # Errors
    ///
    /// Returns an error if any generator from `other` anticommutes with a
    /// generator from `self`.
    pub fn merge(&mut self, other: &PauliStabilizerGroup) -> Result<(), PauliStabilizerGroupError> {
        self.inner.merge(&other.inner).map_err(|e| match e {
            PauliGroupError::NonCommuting(a, b) => PauliStabilizerGroupError::NonCommuting(a, b),
        })
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
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let seq: PauliSequence = s.parse()?;
        let generators = seq.paulis().to_vec();
        Ok(Self::new(generators)?)
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
    use pecos_core::{Pauli, PauliOperator};

    #[test]
    fn test_repetition_code() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        assert_eq!(stab.rank(), 2);
    }

    #[test]
    fn test_steane_code() {
        let stab = PauliStabilizerGroup::new(vec![
            Xs([0, 2, 4, 6]),
            Xs([1, 2, 5, 6]),
            Xs([3, 4, 5, 6]),
            Zs([0, 2, 4, 6]),
            Zs([1, 2, 5, 6]),
            Zs([3, 4, 5, 6]),
        ])
        .unwrap();
        assert_eq!(stab.rank(), 6);
    }

    #[test]
    fn test_rejects_non_commuting() {
        let result = PauliStabilizerGroup::new(vec![X(0), Z(0)]);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            PauliStabilizerGroupError::NonCommuting(0, 1)
        );
    }

    #[test]
    fn test_rejects_imaginary_phase() {
        use pecos_core::pauli::algebra::i;
        let result = PauliStabilizerGroup::new(vec![i * X(0)]);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            PauliStabilizerGroupError::NonRealPhase(0)
        );
    }

    #[test]
    fn test_accepts_negative_phase() {
        // -ZZ is a valid stabilizer (phase is -1, which is real)
        let stab = PauliStabilizerGroup::new(vec![-Zs([0, 1])]);
        assert!(stab.is_ok());
    }

    #[test]
    fn test_contains() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        assert!(stab.contains(&Zs([0, 2])));
        assert!(!stab.contains(&X(0)));
    }

    #[test]
    fn test_from_strs() {
        let stab = PauliStabilizerGroup::from_strs(&["ZZI", "IZZ"]).unwrap();
        assert_eq!(stab.rank(), 2);
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
        let stab = PauliStabilizerGroup::new(vec![
            X(0) & Z(1) & Z(2) & X(3), // XZZXI
            X(1) & Z(2) & Z(3) & X(4), // IXZZX
            X(0) & X(2) & Z(3) & Z(4), // XIXZZ
            Z(0) & X(1) & X(3) & Z(4), // ZXIXZ
        ])
        .unwrap();
        assert_eq!(stab.rank(), 4);
    }

    // ========================================================================
    // FromStr / to_dense_str / to_sparse_str tests
    // ========================================================================

    #[test]
    fn test_from_str_dense() {
        let stab: PauliStabilizerGroup = "ZZI\nIZZ".parse().unwrap();
        assert_eq!(stab.rank(), 2);
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
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
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
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        assert!(stab.is_independent());

        // Add a redundant generator: ZIZ = ZZI * IZZ
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2]), Zs([0, 2])]).unwrap();
        assert!(!stab.is_independent());
    }

    // ========================================================================
    // Group element tests
    // ========================================================================

    #[test]
    fn test_element_identity() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        let id = stab.element(0b00);
        assert!(id.is_identity());
    }

    #[test]
    fn test_element_single_generator() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        assert_eq!(stab.element(0b01), Zs([0, 1]));
        assert_eq!(stab.element(0b10), Zs([1, 2]));
    }

    #[test]
    fn test_element_product() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        // ZZI * IZZ = ZIZ
        let product = stab.element(0b11);
        assert_eq!(product.get(0), Pauli::Z);
        assert_eq!(product.get(1), Pauli::I);
        assert_eq!(product.get(2), Pauli::Z);
    }

    #[test]
    fn test_multiply_by() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        // generator[0] (ZZI) * Z(0) = IZI
        let result = stab.multiply_by(0, &Z(0));
        assert_eq!(result.get(0), Pauli::I);
        assert_eq!(result.get(1), Pauli::Z);
        assert_eq!(result.weight(), 1);
    }

    #[test]
    fn test_multiply_by_identity() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1])]).unwrap();
        let id = PauliString::identity();
        let result = stab.multiply_by(0, &id);
        assert_eq!(result, Zs([0, 1]));
    }

    #[test]
    fn test_elements_count() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        let elements: Vec<_> = stab.elements().collect();
        assert_eq!(elements.len(), 4); // 2^2
    }

    #[test]
    fn test_elements_all_in_group() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        for elem in stab.elements() {
            assert!(stab.contains_with_phase(&elem));
        }
    }

    #[test]
    fn test_elements_contains_identity() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1])]).unwrap();
        let elements: Vec<_> = stab.elements().collect();
        assert!(elements.iter().any(pecos_core::PauliString::is_identity));
    }

    #[test]
    fn test_elements_single_generator() {
        let stab = PauliStabilizerGroup::new(vec![Z(0)]).unwrap();
        let elements: Vec<_> = stab.elements().collect();
        assert_eq!(elements.len(), 2); // I, Z
        assert!(elements.iter().any(pecos_core::PauliString::is_identity));
        assert!(elements.iter().any(|e| *e == Z(0)));
    }

    #[test]
    fn test_element_with_negative_phase() {
        // -ZZ is a valid stabilizer generator
        let stab = PauliStabilizerGroup::new(vec![-Zs([0, 1])]).unwrap();
        let elem = stab.element(0b1);
        assert_eq!(elem.phase(), QuarterPhase::MinusOne);
        assert_eq!(elem.weight(), 2);

        // Product with itself: (-ZZ)(-ZZ) = +II = identity
        let elements: Vec<_> = stab.elements().collect();
        assert_eq!(elements.len(), 2);
        assert!(elements.iter().any(pecos_core::PauliString::is_identity));
    }

    #[test]
    fn test_elements_closure() {
        // For a valid stabilizer group, the product of any two elements
        // should also be an element
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
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
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
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
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1])]).unwrap();
        // Multiply X(0) by generator ZZ: ZZ * X(0) = -Y(0)Z(1) (different from stabilizer)
        let result = stab.multiply_by(0, &X(0));
        assert!(!stab.contains(&result));
    }

    #[test]
    fn test_empty_stabilizer_group() {
        let stab = PauliStabilizerGroup::new(vec![]).unwrap();
        assert_eq!(stab.rank(), 0);
        assert_eq!(stab.num_qubits(), 0);
        assert_eq!(stab.elements().count(), 1); // just identity
    }

    #[test]
    #[should_panic(expected = "exceeds number of generators")]
    fn test_element_out_of_range_panics() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1])]).unwrap();
        // mask 0b11 = 3 but only 1 generator, so bit 1 is out of range
        let _ = stab.element(0b11);
    }

    // ========================================================================
    // apply_clifford tests
    // ========================================================================

    #[test]
    fn test_apply_clifford_hadamard_all() {
        use pecos_core::clifford_rep::CliffordRep;

        // Repetition code: +ZZ_, +_ZZ
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();

        // Apply H to all qubits: Z -> X (phase preserved)
        let h_all = CliffordRep::h(0)
            .compose(&CliffordRep::h(1))
            .compose(&CliffordRep::h(2));
        let transformed = stab.apply_clifford(&h_all);

        // Verify body AND phase
        assert!(transformed.contains_with_phase(&Xs([0, 1])));
        assert!(transformed.contains_with_phase(&Xs([1, 2])));
        assert_eq!(transformed.rank(), 2);
    }

    #[test]
    fn test_apply_clifford_identity() {
        use pecos_core::clifford_rep::CliffordRep;

        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        let id = CliffordRep::identity(3);
        let transformed = stab.apply_clifford(&id);

        assert!(transformed.contains_with_phase(&Zs([0, 1])));
        assert!(transformed.contains_with_phase(&Zs([1, 2])));
    }

    #[test]
    fn test_apply_clifford_cx() {
        use pecos_core::clifford_rep::CliffordRep;

        // Single stabilizer: +ZZ
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1])]).unwrap();

        // CX(0,1): Z_0 stays, Z_1 -> Z_0 Z_1
        // So ZZ = Z_0 * Z_1 -> Z_0 * (Z_0 * Z_1) = Z_1
        let cx = CliffordRep::cx(0, 1);
        let transformed = stab.apply_clifford(&cx);

        assert!(transformed.contains_with_phase(&Z(1)));
    }

    #[test]
    fn test_apply_clifford_z_gate_flips_x_phase() {
        use pecos_core::clifford_rep::CliffordRep;

        // Stabilizer: +XX
        let stab = PauliStabilizerGroup::new(vec![Xs([0, 1])]).unwrap();

        // Z on qubit 0: X -> -X, so XX -> -XX (phase flip)
        let z0 = CliffordRep::z(0).extended_to(2);
        let transformed = stab.apply_clifford(&z0);

        // Phase should be -1 now
        assert!(transformed.contains_with_phase(&(-Xs([0, 1]))));
    }

    #[test]
    fn test_apply_clifford_s_gate() {
        use pecos_core::clifford_rep::CliffordRep;

        // Stabilizer: +XZ (on qubits 0,1)
        let stab = PauliStabilizerGroup::new(vec![X(0) & Z(1)]).unwrap();

        // SZ on qubit 0: X -> Y, Z -> Z
        let s0 = CliffordRep::sz(0).extended_to(2);
        let transformed = stab.apply_clifford(&s0);

        // XZ -> YZ with phase +1
        assert!(transformed.contains_with_phase(&(Y(0) & Z(1))));
    }

    #[test]
    fn test_apply_clifford_swap() {
        use pecos_core::clifford_rep::CliffordRep;

        // Stabilizer: +XZ (X on qubit 0, Z on qubit 1)
        let stab = PauliStabilizerGroup::new(vec![X(0) & Z(1)]).unwrap();

        let swap = CliffordRep::swap(0, 1);
        let transformed = stab.apply_clifford(&swap);

        // SWAP exchanges qubits: XZ -> ZX
        assert!(transformed.contains_with_phase(&(Z(0) & X(1))));
    }

    #[test]
    fn test_apply_clifford_cz() {
        use pecos_core::clifford_rep::CliffordRep;

        // Stabilizer: +XI (X on qubit 0 only)
        let stab = PauliStabilizerGroup::new(vec![X(0)]).unwrap();

        // CZ: X_0 -> X_0 Z_1
        let cz = CliffordRep::cz(0, 1);
        let transformed = stab.apply_clifford(&cz);

        assert!(transformed.contains_with_phase(&(X(0) & Z(1))));
    }

    // ========================================================================
    // Mutation method tests
    // ========================================================================

    #[test]
    fn test_add_generator() {
        let mut group = PauliStabilizerGroup::new(vec![Zs([0, 1])]).unwrap();
        assert_eq!(group.num_generators(), 1);

        // Add a commuting generator
        group.add_generator(Zs([1, 2])).unwrap();
        assert_eq!(group.num_generators(), 2);
        assert_eq!(group.rank(), 2);
    }

    #[test]
    fn test_add_generator_rejects_anticommuting() {
        let mut group = PauliStabilizerGroup::new(vec![Zs([0, 1])]).unwrap();
        let result = group.add_generator(X(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_add_generator_rejects_imaginary_phase() {
        let mut group = PauliStabilizerGroup::new(vec![Zs([0, 1])]).unwrap();
        let bad = PauliString::from_paulis_with_phase(QuarterPhase::PlusI, &[Pauli::Z]);
        let result = group.add_generator(bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_generator() {
        let mut group = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        assert_eq!(group.num_generators(), 2);

        let removed = group.remove_generator(0);
        assert_eq!(group.num_generators(), 1);
        assert_eq!(removed.weight(), 2);
    }

    #[test]
    fn test_merge_compatible_groups() {
        // Two groups on disjoint qubits
        let mut group_a = PauliStabilizerGroup::new(vec![Zs([0, 1])]).unwrap();
        let group_b = PauliStabilizerGroup::new(vec![Zs([2, 3])]).unwrap();

        group_a.merge(&group_b).unwrap();
        assert_eq!(group_a.num_generators(), 2);
        assert_eq!(group_a.rank(), 2);
    }

    #[test]
    fn test_merge_rejects_anticommuting() {
        let mut group_a = PauliStabilizerGroup::new(vec![Zs([0, 1])]).unwrap();
        // X(0) anticommutes with Z(0)Z(1) (odd overlap on qubit 0)
        let group_b = PauliStabilizerGroup::new(vec![X(0)]).unwrap();

        let result = group_a.merge(&group_b);
        assert!(result.is_err());
    }

    // ========================================================================
    // Refactoring: element() API consistency with PauliGroup
    // ========================================================================

    #[test]
    fn element_mask_matches_group_exponents() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        let group = stab.as_group();

        // mask 0b01 = generator 0 only
        assert_eq!(stab.element(0b01), group.element(&[1, 0]));
        // mask 0b10 = generator 1 only
        assert_eq!(stab.element(0b10), group.element(&[0, 1]));
        // mask 0b11 = both generators
        assert_eq!(stab.element(0b11), group.element(&[1, 1]));
        // mask 0b00 = identity
        assert_eq!(stab.element(0b00), group.element(&[0, 0]));
    }

    #[test]
    fn elements_match_group_elements() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        let stab_elems: Vec<_> = stab.elements().collect();
        let group_elems: Vec<_> = stab.as_group().elements().collect();
        assert_eq!(stab_elems.len(), group_elems.len());
        // Every stabilizer element should be in the group
        for elem in &stab_elems {
            assert!(
                stab.as_group().contains_with_phase(elem),
                "stab element {} not found in group",
                elem.to_sparse_str()
            );
        }
    }

    // ========================================================================
    // Merge with different num_qubits
    // ========================================================================

    #[test]
    fn merge_different_num_qubits() {
        let mut g1 = PauliStabilizerGroup::new(vec![Z(0)]).unwrap();
        let g2 = PauliStabilizerGroup::new(vec![Z(2)]).unwrap();
        assert!(g1.merge(&g2).is_ok());
        assert_eq!(g1.num_generators(), 2);
    }

    // ========================================================================
    // Add, remove, add mutation sequence
    // ========================================================================

    #[test]
    fn add_remove_add_sequence() {
        let mut stab = PauliStabilizerGroup::new(vec![Z(0)]).unwrap();
        assert_eq!(stab.num_generators(), 1);

        stab.add_generator(Z(1)).unwrap();
        assert_eq!(stab.num_generators(), 2);

        let removed = stab.remove_generator(0);
        assert_eq!(removed, Z(0));
        assert_eq!(stab.num_generators(), 1);

        // Re-add Z(0) — should still commute with Z(1)
        stab.add_generator(Z(0)).unwrap();
        assert_eq!(stab.num_generators(), 2);
        assert!(stab.contains(&Zs([0, 1])));
    }

    // ========================================================================
    // new() vs from_generators_unchecked() consistency
    // ========================================================================

    #[test]
    fn new_matches_unchecked() {
        let gens = vec![Zs([0, 1]), Zs([1, 2])];
        let checked = PauliStabilizerGroup::new(gens.clone()).unwrap();
        let unchecked = PauliStabilizerGroup::from_generators_unchecked(gens);
        assert_eq!(checked.rank(), unchecked.rank());
        assert_eq!(checked.num_qubits(), unchecked.num_qubits());
        assert_eq!(checked.stabilizers().len(), unchecked.stabilizers().len());
        for s in checked.stabilizers() {
            assert!(unchecked.contains_with_phase(s));
        }
    }

    // ========================================================================
    // as_group() accessor
    // ========================================================================

    #[test]
    fn as_group_has_real_phases() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1])]).unwrap();
        let group = stab.as_group();
        assert!(group.has_real_phases());
        assert!(!group.contains_minus_identity());
    }

    // ========================================================================
    // try_from_set
    // ========================================================================

    #[test]
    fn try_from_set_with_redundant_elements() {
        let set = PauliSet::from_iter([Zs([0, 1]), Zs([1, 2]), Zs([0, 2])]);
        let stab = PauliStabilizerGroup::try_from_set(&set).unwrap();
        assert_eq!(stab.rank(), 2);
        assert!(stab.is_independent());
        // All original elements should be in the group
        assert!(stab.contains_with_phase(&Zs([0, 1])));
        assert!(stab.contains_with_phase(&Zs([1, 2])));
        assert!(stab.contains_with_phase(&Zs([0, 2])));
    }

    #[test]
    fn try_from_set_non_commuting_fails() {
        let set = PauliSet::from_iter([X(0), Z(0)]);
        let result = PauliStabilizerGroup::try_from_set(&set);
        assert!(matches!(
            result.unwrap_err(),
            PauliStabilizerGroupError::NonCommuting(_, _)
        ));
    }
}
