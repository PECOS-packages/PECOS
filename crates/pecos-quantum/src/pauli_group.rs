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

//! An abelian subgroup of the Pauli group with [`QuarterPhase`] generators.
//!
//! A [`PauliGroup`] wraps [`PauliSequence`] with the constraint that all generators
//! mutually commute. Unlike [`PauliStabilizerGroup`], generators may carry any
//! [`QuarterPhase`] (`{+1, -1, +i, -i}`), not just [`Sign`] (`{+1, -1}`).
//!
//! # Stabilizer vs Pauli groups
//!
//! Stabilizer generators have real phases, so every element squares to `+I`:
//! the group has exponent 2 and lives naturally over GF(2).
//!
//! When a generator has phase `+i` or `-i`, it has order 4 instead of 2
//! (since `(iP)^2 = -I ≠ I` but `(iP)^4 = I`). The group can therefore
//! contain `-I`, meaning it cannot stabilize any quantum state.
//!
//! # Conversion
//!
//! - Every [`PauliStabilizerGroup`] is a valid [`PauliGroup`] (via [`From`]).
//! - A [`PauliGroup`] can be converted to a [`PauliStabilizerGroup`] only if
//!   all generators have real phases (via [`TryFrom`]).
//!
//! [`QuarterPhase`]: pecos_core::QuarterPhase
//! [`Sign`]: pecos_core::Sign
//! [`PauliStabilizerGroup`]: crate::PauliStabilizerGroup
//!
//! # Examples
//!
//! ```
//! use pecos_quantum::PauliGroup;
//! use pecos_core::pauli::constructors::*;
//! use pecos_core::pauli::algebra::i;
//!
//! // Generators with imaginary phases are allowed
//! let group = PauliGroup::new(vec![
//!     i * X(0) & Y(1),
//!     Z(2),
//! ]).unwrap();
//!
//! assert_eq!(group.rank(), 2);
//! assert!(group.contains_minus_identity());
//! ```

use crate::pauli_sequence::{F2Matrix, PauliSequence};
use crate::pauli_set::PauliSet;
use crate::stabilizer_group::{PauliStabilizerGroup, PauliStabilizerGroupError};
use pecos_core::{PauliOperator, PauliString, Phase, QuarterPhase};
use std::fmt;
use std::str::FromStr;

/// Errors that can occur when constructing a [`PauliGroup`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PauliGroupError {
    /// Generators at indices (i, j) anticommute.
    NonCommuting(usize, usize),
}

impl fmt::Display for PauliGroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonCommuting(idx_a, idx_b) => {
                write!(f, "generators {idx_a} and {idx_b} anticommute")
            }
        }
    }
}

impl std::error::Error for PauliGroupError {}

/// Returns the order of a Pauli group element.
///
/// Real-phase generators (±1) have order 2: `g^2 = I`.
/// Imaginary-phase generators (±i) have order 4: `g^4 = I`.
fn generator_order(phase: QuarterPhase) -> u32 {
    match phase {
        QuarterPhase::PlusOne | QuarterPhase::MinusOne => 2,
        QuarterPhase::PlusI | QuarterPhase::MinusI => 4,
    }
}

/// An abelian subgroup of the Pauli group with [`QuarterPhase`] generators.
///
/// This is a validated wrapper around [`PauliSequence`] enforcing:
/// - All generators mutually commute (abelian)
///
/// Generators may carry any [`QuarterPhase`] phase (`{+1, -1, +i, -i}`).
/// For the more restrictive stabilizer group (real phases only), see
/// [`PauliStabilizerGroup`].
///
/// [`QuarterPhase`]: pecos_core::QuarterPhase
/// [`PauliStabilizerGroup`]: crate::PauliStabilizerGroup
///
/// # Examples
///
/// ```
/// use pecos_quantum::PauliGroup;
/// use pecos_core::pauli::constructors::*;
/// use pecos_core::pauli::algebra::i;
///
/// // A group with an imaginary-phase generator
/// let group = PauliGroup::new(vec![i * X(0), Z(1)]).unwrap();
/// assert_eq!(group.num_generators(), 2);
///
/// // The generator iX has order 4 (not 2), so the group contains -I
/// assert!(group.contains_minus_identity());
///
/// // Cannot convert to a stabilizer group because of the imaginary phase
/// use pecos_quantum::PauliStabilizerGroup;
/// assert!(PauliStabilizerGroup::try_from(group).is_err());
/// ```
#[derive(Debug, Clone)]
pub struct PauliGroup {
    inner: PauliSequence,
}

impl PauliGroup {
    /// Creates a new `PauliGroup`, validating that all generators commute.
    ///
    /// # Errors
    ///
    /// Returns [`PauliGroupError::NonCommuting`] if any pair of generators anticommute.
    pub fn new(generators: Vec<PauliString>) -> Result<Self, PauliGroupError> {
        for idx_a in 0..generators.len() {
            for idx_b in (idx_a + 1)..generators.len() {
                if !generators[idx_a].commutes_with(&generators[idx_b]) {
                    return Err(PauliGroupError::NonCommuting(idx_a, idx_b));
                }
            }
        }

        let inner = PauliSequence::new(generators);
        Ok(Self { inner })
    }

    /// Creates a `PauliGroup` without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the generators mutually commute.
    #[must_use]
    pub fn from_generators_unchecked(generators: Vec<PauliString>) -> Self {
        Self {
            inner: PauliSequence::new(generators),
        }
    }

    /// Creates a `PauliGroup` from a [`PauliSet`], row-reducing to independent generators.
    ///
    /// This validates commutativity and reduces to a minimal generating set.
    ///
    /// # Errors
    ///
    /// Returns [`PauliGroupError::NonCommuting`] if any pair of elements anticommute.
    pub fn try_from_set(set: &PauliSet) -> Result<Self, PauliGroupError> {
        Self::try_from(set.to_sequence())
    }

    /// Creates a `PauliGroup` from string representations.
    ///
    /// # Errors
    ///
    /// Returns an error if any string cannot be parsed, or if the resulting
    /// generators don't form a valid abelian group.
    pub fn from_strs(strings: &[&str]) -> Result<Self, Box<dyn std::error::Error>> {
        let coll = PauliSequence::from_strs(strings)?;
        let generators = coll.paulis().to_vec();
        Ok(Self::new(generators)?)
    }

    /// Returns a reference to the underlying [`PauliSequence`].
    #[must_use]
    pub fn as_sequence(&self) -> &PauliSequence {
        &self.inner
    }

    /// Returns a reference to the generators.
    #[must_use]
    pub fn generators(&self) -> &[PauliString] {
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

    /// Computes the rank (number of linearly independent generators over GF(2)).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Returns `true` if all generators are linearly independent (over GF(2)).
    #[must_use]
    pub fn is_independent(&self) -> bool {
        self.rank() == self.num_generators()
    }

    /// Returns the order of the `idx`-th generator.
    ///
    /// Real-phase generators (±1) have order 2. Imaginary-phase generators (±i)
    /// have order 4.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= num_generators()`.
    #[must_use]
    pub fn generator_order(&self, idx: usize) -> u32 {
        generator_order(self.inner.paulis()[idx].phase())
    }

    /// Returns the total number of elements in the group.
    ///
    /// Each generator contributes a factor equal to its order (2 or 4).
    #[must_use]
    pub fn group_order(&self) -> u64 {
        self.inner
            .paulis()
            .iter()
            .map(|g| u64::from(generator_order(g.phase())))
            .product()
    }

    /// Checks if a Pauli string's body is in the group (ignoring phase).
    ///
    /// Uses the GF(2) symplectic representation.
    #[must_use]
    pub fn contains(&self, pauli: &PauliString) -> bool {
        self.inner.contains(pauli)
    }

    /// Checks if a Pauli string is in the group (including phase).
    ///
    /// This correctly handles generators with imaginary phases (order 4).
    /// The GF(2) body decomposition determines which generators contribute
    /// with odd exponents; generators with imaginary phases can additionally
    /// contribute a sign flip via their even-exponent powers (`g^2 = -I`).
    #[must_use]
    pub fn contains_with_phase(&self, pauli: &PauliString) -> bool {
        let n = self.inner.num_qubits();
        let k = self.inner.paulis().len();

        // Build augmented matrix [symplectic | identity] to track which generators are used
        let aug_cols = 2 * n + k;
        let mut mat = F2Matrix::zeros(k, aug_cols);

        for (row_idx, generator) in self.inner.paulis().iter().enumerate() {
            for q in generator.x_positions() {
                if q < n {
                    mat.rows[row_idx][q] = 1;
                }
            }
            for q in generator.z_positions() {
                if q < n {
                    mat.rows[row_idx][n + q] = 1;
                }
            }
            mat.rows[row_idx][2 * n + row_idx] = 1;
        }

        let (reduced, pivots) = mat.row_reduce();

        // Build target symplectic vector
        let mut target = vec![0u8; aug_cols];
        for q in pauli.x_positions() {
            if q < n {
                target[q] = 1;
            }
        }
        for q in pauli.z_positions() {
            if q < n {
                target[n + q] = 1;
            }
        }

        // Eliminate using reduced rows
        for (row_idx, &pivot_col) in pivots.iter().enumerate() {
            if target[pivot_col] == 1 {
                for (col, t) in target.iter_mut().enumerate() {
                    *t ^= reduced.rows[row_idx][col];
                }
            }
        }

        // Body must be zero
        if !target[..2 * n].iter().all(|&b| b == 0) {
            return false;
        }

        // Compute base phase from the GF(2) decomposition (each used generator once)
        let mut base_product = PauliString::identity();
        for (idx, generator) in self.inner.paulis().iter().enumerate() {
            if target[2 * n + idx] == 1 {
                base_product = base_product * generator;
            }
        }

        let base_phase = base_product.phase();
        let target_phase = pauli.phase();

        if target_phase == base_phase {
            return true;
        }

        // Check if we can reach the target by flipping the sign via imaginary generators.
        // Each generator with ±i phase has g^2 = -I, contributing a factor of -1.
        // We can apply any subset of these, so if there's at least one imaginary-phase
        // generator, we can flip the overall sign.
        let has_imaginary = self
            .inner
            .paulis()
            .iter()
            .any(|g| matches!(g.phase(), QuarterPhase::PlusI | QuarterPhase::MinusI));

        if has_imaginary {
            let neg_base = base_phase.multiply(&QuarterPhase::MinusOne);
            target_phase == neg_base
        } else {
            false
        }
    }

    /// Returns `true` if `-I` is an element of this group.
    ///
    /// A group contains `-I` if and only if at least one generator has an
    /// imaginary phase (`+i` or `-i`), since `(iP)^2 = -I`.
    ///
    /// Stabilizer groups never contain `-I` (that is precisely the stabilizer
    /// condition: every element must square to `+I`).
    #[must_use]
    pub fn contains_minus_identity(&self) -> bool {
        self.inner
            .paulis()
            .iter()
            .any(|g| matches!(g.phase(), QuarterPhase::PlusI | QuarterPhase::MinusI))
    }

    /// Returns `true` if all generators have real phases (`+1` or `-1`).
    ///
    /// When this returns `true`, the group can be converted to a
    /// [`PauliStabilizerGroup`] via [`TryFrom`].
    #[must_use]
    pub fn has_real_phases(&self) -> bool {
        self.inner
            .paulis()
            .iter()
            .all(|g| matches!(g.phase(), QuarterPhase::PlusOne | QuarterPhase::MinusOne))
    }

    /// Returns the group element for the given exponent tuple.
    ///
    /// Each entry `exponents[i]` is the power of the `i`-th generator.
    /// Real-phase generators are taken mod 2; imaginary-phase generators mod 4.
    ///
    /// # Panics
    ///
    /// Panics if `exponents.len() != num_generators()`.
    #[must_use]
    pub fn element(&self, exponents: &[u32]) -> PauliString {
        let gens = self.inner.paulis();
        assert_eq!(
            exponents.len(),
            gens.len(),
            "expected {} exponents, got {}",
            gens.len(),
            exponents.len()
        );

        let mut result = PauliString::identity();
        for (g, &exp) in gens.iter().zip(exponents) {
            let order = generator_order(g.phase());
            let exp = exp % order;
            for _ in 0..exp {
                result = result * g.clone();
            }
        }
        result
    }

    /// Returns an iterator over all elements of the group.
    ///
    /// Each generator of order `o` contributes `o` choices (exponents `0..o`),
    /// so the total element count is `group_order()`.
    ///
    /// **Warning**: The group size can be exponential in the number of generators.
    ///
    /// # Panics
    ///
    /// Panics if the group order exceeds `2^30`.
    pub fn elements(&self) -> impl Iterator<Item = PauliString> + '_ {
        let orders: Vec<u32> = self
            .inner
            .paulis()
            .iter()
            .map(|g| generator_order(g.phase()))
            .collect();
        let total = self.group_order();
        assert!(
            total <= 1 << 30,
            "elements() would yield {total} items; use contains() for membership testing"
        );

        (0..total).map(move |mut idx| {
            let mut exponents = Vec::with_capacity(orders.len());
            for &order in &orders {
                exponents.push((idx % u64::from(order)) as u32);
                idx /= u64::from(order);
            }
            self.element(&exponents)
        })
    }

    /// Multiplies a Pauli string by the generator at the given index.
    ///
    /// Returns `generators[index] * pauli`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= num_generators()`.
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

    /// Returns the binary symplectic matrix representation.
    #[must_use]
    pub fn to_symplectic_matrix(&self) -> F2Matrix {
        self.inner.to_symplectic_matrix()
    }

    /// Returns the commutation matrix (always all-true for a valid group).
    #[must_use]
    pub fn commutation_matrix(&self) -> Vec<Vec<bool>> {
        self.inner.commutation_matrix()
    }

    /// Returns the generators in row-reduced form, removing redundant generators.
    #[must_use]
    pub fn row_reduce(&self) -> PauliSequence {
        self.inner.row_reduce()
    }

    /// Iterates over the generators.
    pub fn iter(&self) -> impl Iterator<Item = &PauliString> {
        self.inner.iter()
    }

    /// Returns the dense string representation, one generator per line.
    #[must_use]
    pub fn to_dense_str(&self) -> String {
        self.inner.to_dense_str()
    }

    /// Returns the sparse string representation, one generator per line.
    #[must_use]
    pub fn to_sparse_str(&self) -> String {
        self.inner.to_sparse_str()
    }

    /// Transforms all generators by a Clifford gate: each `g_i` -> `C g_i C†`.
    ///
    /// Clifford conjugation preserves commutativity and maps quarter-turn phases
    /// to quarter-turn phases, so the result is always a valid `PauliGroup`.
    #[must_use]
    pub fn apply_clifford(&self, clifford: &pecos_core::clifford_rep::CliffordRep) -> PauliGroup {
        let transformed: Vec<PauliString> = self
            .inner
            .paulis()
            .iter()
            .map(|g| clifford.apply(g))
            .collect();

        PauliGroup {
            inner: PauliSequence::new(transformed),
        }
    }

    // ========================================================================
    // Mutation methods
    // ========================================================================

    /// Adds a generator to the group.
    ///
    /// The new generator must commute with all existing generators.
    ///
    /// # Errors
    ///
    /// Returns an error if the generator anticommutes with any existing generator.
    pub fn add_generator(&mut self, generator: PauliString) -> Result<(), PauliGroupError> {
        for (idx, existing) in self.inner.paulis().iter().enumerate() {
            if !generator.commutes_with(existing) {
                return Err(PauliGroupError::NonCommuting(self.num_generators(), idx));
            }
        }

        self.inner.push(generator);
        Ok(())
    }

    /// Removes the generator at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= num_generators()`.
    pub fn remove_generator(&mut self, index: usize) -> PauliString {
        assert!(
            index < self.num_generators(),
            "index {index} out of range for {} generators",
            self.num_generators()
        );
        self.inner.remove(index)
    }

    /// Merges another group into this one.
    ///
    /// All generators from `other` must commute with all generators in `self`.
    ///
    /// # Errors
    ///
    /// Returns an error if any generator from `other` anticommutes with a
    /// generator from `self`.
    pub fn merge(&mut self, other: &PauliGroup) -> Result<(), PauliGroupError> {
        let base_len = self.num_generators();
        for (new_idx, new_gen) in other.generators().iter().enumerate() {
            for (old_idx, old_gen) in self.inner.paulis().iter().enumerate() {
                if !new_gen.commutes_with(old_gen) {
                    return Err(PauliGroupError::NonCommuting(base_len + new_idx, old_idx));
                }
            }
        }

        self.inner.extend(other.generators().iter().cloned());
        Ok(())
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl From<PauliStabilizerGroup> for PauliGroup {
    /// Extracts the underlying `PauliGroup` from a stabilizer group.
    fn from(stab: PauliStabilizerGroup) -> Self {
        stab.inner
    }
}

impl TryFrom<PauliSequence> for PauliGroup {
    type Error = PauliGroupError;

    /// Converts a [`PauliSequence`] into a [`PauliGroup`] by validating
    /// commutativity and row-reducing to independent generators.
    ///
    /// # Errors
    ///
    /// Returns [`PauliGroupError::NonCommuting`] if any pair of elements anticommute.
    fn try_from(seq: PauliSequence) -> Result<Self, Self::Error> {
        let paulis = seq.paulis();
        for i in 0..paulis.len() {
            for j in (i + 1)..paulis.len() {
                if !paulis[i].commutes_with(&paulis[j]) {
                    return Err(PauliGroupError::NonCommuting(i, j));
                }
            }
        }

        let reduced = seq.row_reduce();
        Ok(Self { inner: reduced })
    }
}

impl TryFrom<PauliGroup> for PauliStabilizerGroup {
    type Error = PauliStabilizerGroupError;

    /// Converts a `PauliGroup` to a `PauliStabilizerGroup` if all generators
    /// have real phases (`+1` or `-1`).
    ///
    /// # Errors
    ///
    /// Returns [`PauliStabilizerGroupError::NonRealPhase`] if any generator
    /// has phase `+i` or `-i`.
    fn try_from(group: PauliGroup) -> Result<Self, Self::Error> {
        for (idx, generator) in group.generators().iter().enumerate() {
            match generator.phase() {
                QuarterPhase::PlusOne | QuarterPhase::MinusOne => {}
                _ => return Err(PauliStabilizerGroupError::NonRealPhase(idx)),
            }
        }

        Ok(PauliStabilizerGroup { inner: group })
    }
}

impl TryFrom<PauliSequence> for PauliStabilizerGroup {
    type Error = PauliStabilizerGroupError;

    /// Converts a [`PauliSequence`] into a [`PauliStabilizerGroup`] by validating
    /// commutativity and real phases, then row-reducing to independent generators.
    ///
    /// # Errors
    ///
    /// Returns [`PauliStabilizerGroupError::NonCommuting`] if any pair anticommute.
    /// Returns [`PauliStabilizerGroupError::NonRealPhase`] if any reduced generator
    /// has phase `+i` or `-i`.
    fn try_from(seq: PauliSequence) -> Result<Self, Self::Error> {
        let group = PauliGroup::try_from(seq).map_err(|e| match e {
            PauliGroupError::NonCommuting(a, b) => PauliStabilizerGroupError::NonCommuting(a, b),
        })?;
        Self::try_from(group)
    }
}

impl From<PauliGroup> for PauliSequence {
    /// Extracts the generators as a [`PauliSequence`].
    fn from(group: PauliGroup) -> Self {
        group.inner
    }
}

impl From<PauliStabilizerGroup> for PauliSequence {
    /// Extracts the generators as a [`PauliSequence`].
    fn from(stab: PauliStabilizerGroup) -> Self {
        stab.inner.inner
    }
}

impl From<PauliGroup> for PauliSet {
    /// Collects the generators into a [`PauliSet`].
    fn from(group: PauliGroup) -> Self {
        group.generators().iter().cloned().collect()
    }
}

impl From<PauliStabilizerGroup> for PauliSet {
    /// Collects the generators into a [`PauliSet`].
    fn from(stab: PauliStabilizerGroup) -> Self {
        PauliSet::from(PauliGroup::from(stab))
    }
}

// ============================================================================
// Display and FromStr
// ============================================================================

impl FromStr for PauliGroup {
    type Err = Box<dyn std::error::Error>;

    /// Parses a `PauliGroup` from newline-delimited Pauli strings.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let seq: PauliSequence = s.parse()?;
        let generators = seq.paulis().to_vec();
        Ok(Self::new(generators)?)
    }
}

impl fmt::Display for PauliGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::pauli::algebra::i;
    use pecos_core::pauli::constructors::*;

    // ========================================================================
    // Construction and basic properties
    // ========================================================================

    #[test]
    fn real_phase_generators() {
        let group = PauliGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        assert_eq!(group.rank(), 2);
        assert!(group.has_real_phases());
        assert!(!group.contains_minus_identity());
    }

    #[test]
    fn imaginary_phase_generator() {
        let group = PauliGroup::new(vec![i * X(0), Z(1)]).unwrap();
        assert_eq!(group.num_generators(), 2);
        assert!(!group.has_real_phases());
    }

    #[test]
    fn rejects_non_commuting() {
        let result = PauliGroup::new(vec![X(0), Z(0)]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PauliGroupError::NonCommuting(0, 1));
    }

    #[test]
    fn accepts_imaginary_phase() {
        let result = PauliGroup::new(vec![i * X(0)]);
        assert!(result.is_ok());
    }

    #[test]
    fn generator_orders() {
        let group = PauliGroup::new(vec![X(0), i * Z(1), -Z(2), -i * X(3)]).unwrap();
        assert_eq!(group.generator_order(0), 2); // +X: order 2
        assert_eq!(group.generator_order(1), 4); // iZ: order 4
        assert_eq!(group.generator_order(2), 2); // -Z: order 2
        assert_eq!(group.generator_order(3), 4); // -iX: order 4
    }

    #[test]
    fn group_order_all_real() {
        let group = PauliGroup::new(vec![X(0), Z(1)]).unwrap();
        assert_eq!(group.group_order(), 4); // 2 * 2
    }

    #[test]
    fn group_order_mixed() {
        let group = PauliGroup::new(vec![i * X(0), Z(1)]).unwrap();
        assert_eq!(group.group_order(), 8); // 4 * 2
    }

    // ========================================================================
    // contains_minus_identity
    // ========================================================================

    #[test]
    fn contains_minus_identity_with_imaginary() {
        let group = PauliGroup::new(vec![i * X(0)]).unwrap();
        assert!(group.contains_minus_identity());
    }

    #[test]
    fn no_minus_identity_with_real() {
        let group = PauliGroup::new(vec![X(0)]).unwrap();
        assert!(!group.contains_minus_identity());
    }

    #[test]
    fn no_minus_identity_with_neg_real() {
        // (-X)^2 = X^2 = +I
        let group = PauliGroup::new(vec![-X(0)]).unwrap();
        assert!(!group.contains_minus_identity());
    }

    #[test]
    fn neg_imaginary_also_gives_minus_identity() {
        // (-iX)^2 = (-i)^2 * X^2 = -I
        let group = PauliGroup::new(vec![-i * X(0)]).unwrap();
        assert!(group.contains_minus_identity());
    }

    // ========================================================================
    // Element enumeration
    // ========================================================================

    #[test]
    fn elements_real_generators() {
        let group = PauliGroup::new(vec![Z(0), Z(1)]).unwrap();
        let elements: Vec<_> = group.elements().collect();
        assert_eq!(elements.len(), 4); // 2 * 2
    }

    #[test]
    fn elements_imaginary_generator() {
        // iX has order 4: {I, iX, -I, -iX}
        let group = PauliGroup::new(vec![i * X(0)]).unwrap();
        let elements: Vec<_> = group.elements().collect();
        assert_eq!(elements.len(), 4);

        // Check the four elements
        assert!(elements[0].is_identity()); // (iX)^0 = I
        assert_eq!(elements[1].phase(), QuarterPhase::PlusI); // (iX)^1 = iX
        assert_eq!(elements[2].phase(), QuarterPhase::MinusOne); // (iX)^2 = -I
        assert_eq!(elements[3].phase(), QuarterPhase::MinusI); // (iX)^3 = -iX
    }

    #[test]
    fn elements_mixed_orders() {
        let group = PauliGroup::new(vec![i * X(0), Z(1)]).unwrap();
        let elements: Vec<_> = group.elements().collect();
        assert_eq!(elements.len(), 8); // 4 * 2
    }

    #[test]
    fn elements_closure() {
        // Product of any two elements should be in the group
        let group = PauliGroup::new(vec![i * X(0), Z(1)]).unwrap();
        let elements: Vec<_> = group.elements().collect();
        for a in &elements {
            for b in &elements {
                let product = a.clone() * b.clone();
                assert!(
                    group.contains_with_phase(&product),
                    "{} * {} = {} not in group",
                    a.to_sparse_str(),
                    b.to_sparse_str(),
                    product.to_sparse_str()
                );
            }
        }
    }

    #[test]
    fn element_by_exponents() {
        let group = PauliGroup::new(vec![i * X(0)]).unwrap();
        let e0 = group.element(&[0]);
        let e1 = group.element(&[1]);
        let e2 = group.element(&[2]);
        let e3 = group.element(&[3]);

        assert!(e0.is_identity());
        assert_eq!(e1.phase(), QuarterPhase::PlusI);
        assert_eq!(e2.phase(), QuarterPhase::MinusOne);
        assert_eq!(e2.weight(), 0); // -I has no body
        assert_eq!(e3.phase(), QuarterPhase::MinusI);
    }

    #[test]
    fn element_exponents_wrap() {
        let group = PauliGroup::new(vec![i * X(0)]).unwrap();
        // exponent 4 wraps to 0 (identity)
        assert_eq!(group.element(&[4]), group.element(&[0]));
        // exponent 5 wraps to 1
        assert_eq!(group.element(&[5]), group.element(&[1]));
    }

    // ========================================================================
    // contains / contains_with_phase
    // ========================================================================

    #[test]
    fn contains_body() {
        let group = PauliGroup::new(vec![(i * X(0)) & Y(1), Z(2)]).unwrap();
        assert!(group.contains(&(X(0) & Y(1))));
        assert!(group.contains(&Z(2)));
        assert!(!group.contains(&X(2)));
    }

    #[test]
    fn contains_with_phase_real_group() {
        let group = PauliGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        assert!(group.contains_with_phase(&Zs([0, 1])));
        assert!(group.contains_with_phase(&Zs([0, 2]))); // product of generators
        assert!(!group.contains_with_phase(&(-Zs([0, 1])))); // wrong phase
    }

    #[test]
    fn contains_with_phase_imaginary_group() {
        let group = PauliGroup::new(vec![i * X(0)]).unwrap();
        // All four elements should be found
        assert!(group.contains_with_phase(&PauliString::identity()));
        assert!(group.contains_with_phase(&(i * X(0))));
        let minus_id = PauliString::with_phase_and_paulis(QuarterPhase::MinusOne, vec![]);
        assert!(group.contains_with_phase(&minus_id));
        assert!(group.contains_with_phase(&(-i * X(0))));
    }

    #[test]
    fn contains_with_phase_minus_identity() {
        let group = PauliGroup::new(vec![i * X(0), Z(1)]).unwrap();
        let minus_id = PauliString::with_phase_and_paulis(QuarterPhase::MinusOne, vec![]);
        assert!(group.contains_with_phase(&minus_id));
    }

    #[test]
    fn contains_with_phase_not_in_group() {
        let group = PauliGroup::new(vec![Z(0)]).unwrap();
        assert!(!group.contains_with_phase(&(i * Z(0)))); // iZ not in group of {I, Z}
    }

    // ========================================================================
    // Conversions
    // ========================================================================

    #[test]
    fn from_stabilizer_group() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        let group = PauliGroup::from(stab);
        assert_eq!(group.rank(), 2);
        assert!(group.has_real_phases());
    }

    #[test]
    fn try_to_stabilizer_real_phases() {
        let group = PauliGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        let stab = PauliStabilizerGroup::try_from(group);
        assert!(stab.is_ok());
        let stab = stab.unwrap();
        assert_eq!(stab.rank(), 2);
    }

    #[test]
    fn try_to_stabilizer_imaginary_fails() {
        let group = PauliGroup::new(vec![i * X(0), Z(1)]).unwrap();
        let result = PauliStabilizerGroup::try_from(group);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            PauliStabilizerGroupError::NonRealPhase(0)
        );
    }

    #[test]
    fn roundtrip_stabilizer_to_group_to_stabilizer() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        let group = PauliGroup::from(stab.clone());
        let stab2 = PauliStabilizerGroup::try_from(group).unwrap();
        assert_eq!(stab.rank(), stab2.rank());
        assert_eq!(stab.num_qubits(), stab2.num_qubits());
    }

    // ========================================================================
    // Mutation
    // ========================================================================

    #[test]
    fn add_generator_commuting() {
        let mut group = PauliGroup::new(vec![Z(0)]).unwrap();
        assert!(group.add_generator(Z(1)).is_ok());
        assert_eq!(group.num_generators(), 2);
    }

    #[test]
    fn add_generator_anticommuting() {
        let mut group = PauliGroup::new(vec![Z(0)]).unwrap();
        assert!(group.add_generator(X(0)).is_err());
    }

    #[test]
    fn add_imaginary_generator() {
        let mut group = PauliGroup::new(vec![Z(0)]).unwrap();
        assert!(!group.contains_minus_identity());
        assert!(group.add_generator(i * Z(1)).is_ok());
        assert!(group.contains_minus_identity());
    }

    #[test]
    fn remove_generator() {
        let mut group = PauliGroup::new(vec![Z(0), Z(1)]).unwrap();
        let removed = group.remove_generator(0);
        assert_eq!(removed, Z(0));
        assert_eq!(group.num_generators(), 1);
    }

    #[test]
    fn merge_groups() {
        let mut g1 = PauliGroup::new(vec![Z(0)]).unwrap();
        let g2 = PauliGroup::new(vec![Z(1)]).unwrap();
        assert!(g1.merge(&g2).is_ok());
        assert_eq!(g1.num_generators(), 2);
    }

    #[test]
    fn merge_anticommuting_fails() {
        let mut g1 = PauliGroup::new(vec![Z(0)]).unwrap();
        let g2 = PauliGroup::new(vec![X(0)]).unwrap();
        assert!(g1.merge(&g2).is_err());
    }

    // ========================================================================
    // Display, FromStr, apply_clifford
    // ========================================================================

    #[test]
    fn display() {
        let group = PauliGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        let s = format!("{group}");
        assert_eq!(s, "ZZI\nIZZ");
    }

    #[test]
    fn from_str() {
        let group: PauliGroup = "ZZI\nIZZ".parse().unwrap();
        assert_eq!(group.rank(), 2);
    }

    #[test]
    fn apply_clifford_preserves_group() {
        use pecos_core::clifford_rep::CliffordRep;

        let group = PauliGroup::new(vec![i * X(0), Z(1)]).unwrap();

        // H on qubit 0: X -> Z, Z -> X
        let h0 = CliffordRep::h(0).extended_to(2);
        let transformed = group.apply_clifford(&h0);

        assert_eq!(transformed.num_generators(), 2);
        assert!(transformed.contains_minus_identity());
    }

    #[test]
    fn empty_group() {
        let group = PauliGroup::new(vec![]).unwrap();
        assert_eq!(group.rank(), 0);
        assert_eq!(group.num_generators(), 0);
        assert_eq!(group.group_order(), 1);
        assert_eq!(group.elements().count(), 1);
        assert!(!group.contains_minus_identity());
    }

    #[test]
    fn multi_qubit_imaginary_generator() {
        let generator = (i * X(0)) & Y(1) & Z(2);
        let group = PauliGroup::new(vec![generator]).unwrap();
        assert_eq!(group.rank(), 1);
        assert!(group.contains_minus_identity());
        assert!(PauliStabilizerGroup::try_from(group).is_err());
    }

    #[test]
    fn multiply_by() {
        let group = PauliGroup::new(vec![i * X(0), Z(1)]).unwrap();
        let result = group.multiply_by(0, &PauliString::identity());
        assert_eq!(result.phase(), QuarterPhase::PlusI);
    }

    #[test]
    fn independent_check() {
        let group = PauliGroup::new(vec![Z(0), Z(1)]).unwrap();
        assert!(group.is_independent());

        let group = PauliGroup::new(vec![Z(0), Z(1), Zs([0, 1])]).unwrap();
        assert!(!group.is_independent());
    }

    #[test]
    fn has_real_phases_mixed() {
        let group = PauliGroup::new(vec![Z(0), i * Z(1)]).unwrap();
        assert!(!group.has_real_phases());
    }

    // ========================================================================
    // contains_with_phase: multiple imaginary generators
    // ========================================================================

    #[test]
    fn contains_with_phase_two_imaginary_generators() {
        // Both generators have imaginary phase
        let group = PauliGroup::new(vec![i * X(0), i * Z(1)]).unwrap();
        let elements: Vec<_> = group.elements().collect();
        // 4 * 4 = 16 elements
        assert_eq!(elements.len(), 16);

        // Every element must be found by contains_with_phase
        for elem in &elements {
            assert!(
                group.contains_with_phase(elem),
                "element {} not found by contains_with_phase",
                elem.to_sparse_str()
            );
        }
    }

    #[test]
    fn contains_with_phase_mixed_imaginary_signs() {
        // -i * X(0) and i * Z(1)
        let group = PauliGroup::new(vec![-i * X(0), i * Z(1)]).unwrap();
        let elements: Vec<_> = group.elements().collect();
        assert_eq!(elements.len(), 16);

        for elem in &elements {
            assert!(
                group.contains_with_phase(elem),
                "element {} not found",
                elem.to_sparse_str()
            );
        }
    }

    #[test]
    fn closure_two_imaginary_generators() {
        let group = PauliGroup::new(vec![i * X(0), i * Z(1)]).unwrap();
        let elements: Vec<_> = group.elements().collect();
        for a in &elements {
            for b in &elements {
                let product = a.clone() * b.clone();
                assert!(
                    group.contains_with_phase(&product),
                    "{} * {} = {} not in group",
                    a.to_sparse_str(),
                    b.to_sparse_str(),
                    product.to_sparse_str()
                );
            }
        }
    }

    // ========================================================================
    // contains_with_phase: consistency with contains()
    // ========================================================================

    #[test]
    fn contains_with_phase_implies_contains() {
        let group = PauliGroup::new(vec![i * X(0), Z(1)]).unwrap();
        let elements: Vec<_> = group.elements().collect();
        for elem in &elements {
            if group.contains_with_phase(elem) {
                assert!(
                    group.contains(elem),
                    "contains_with_phase true but contains false for {}",
                    elem.to_sparse_str()
                );
            }
        }
    }

    // ========================================================================
    // element() panic conditions
    // ========================================================================

    #[test]
    #[should_panic(expected = "expected 1 exponents, got 2")]
    fn element_wrong_exponent_count_panics() {
        let group = PauliGroup::new(vec![X(0)]).unwrap();
        let _ = group.element(&[0, 1]);
    }

    // ========================================================================
    // String representations
    // ========================================================================

    #[test]
    fn to_dense_str_format() {
        let group = PauliGroup::new(vec![X(0) & Z(1), Y(2)]).unwrap();
        let dense = group.to_dense_str();
        assert!(dense.contains('X'));
        assert!(dense.contains('Z'));
        assert!(dense.contains('Y'));
    }

    #[test]
    fn to_sparse_str_format() {
        let group = PauliGroup::new(vec![Zs([0, 1])]).unwrap();
        let sparse = group.to_sparse_str();
        assert!(sparse.contains("Z0"));
        assert!(sparse.contains("Z1"));
    }

    // ========================================================================
    // Dependent generators with imaginary phases
    // ========================================================================

    #[test]
    fn dependent_generators_imaginary() {
        // Z(0), i*Z(0) are linearly dependent (same body, different phase)
        // But they commute, so the group is valid
        let group = PauliGroup::new(vec![Z(0), i * Z(0)]).unwrap();
        assert!(!group.is_independent()); // bodies are linearly dependent over GF(2)
        assert!(group.contains_minus_identity());
    }

    // ========================================================================
    // Generator order doesn't change group
    // ========================================================================

    #[test]
    fn generator_order_invariant() {
        let g1 = PauliGroup::new(vec![i * X(0), Z(1)]).unwrap();
        let g2 = PauliGroup::new(vec![Z(1), i * X(0)]).unwrap();

        // Same elements in both
        let e1: Vec<_> = g1.elements().collect();
        for elem in &e1 {
            assert!(
                g2.contains_with_phase(elem),
                "{} in g1 but not g2",
                elem.to_sparse_str()
            );
        }
    }

    // ========================================================================
    // commutation_matrix for valid group
    // ========================================================================

    #[test]
    fn commutation_matrix_all_true() {
        let group = PauliGroup::new(vec![X(0), Z(1), X(2)]).unwrap();
        let mat = group.commutation_matrix();
        for row in &mat {
            for &val in row {
                assert!(
                    val,
                    "commutation matrix should be all-true for abelian group"
                );
            }
        }
    }

    // ========================================================================
    // TryFrom<PauliSequence> and try_from_set
    // ========================================================================

    #[test]
    fn try_from_sequence_independent() {
        let seq = PauliSequence::new(vec![Z(0), Z(1)]);
        let group = PauliGroup::try_from(seq).unwrap();
        assert_eq!(group.num_generators(), 2);
        assert!(group.is_independent());
    }

    #[test]
    fn try_from_sequence_reduces_redundant() {
        // Z(0)*Z(1) is a product of Z(0) and Z(1), so it's redundant
        let seq = PauliSequence::new(vec![Z(0), Z(1), Zs([0, 1])]);
        let group = PauliGroup::try_from(seq).unwrap();
        assert_eq!(group.num_generators(), 2);
        assert!(group.is_independent());
    }

    #[test]
    fn try_from_sequence_non_commuting_fails() {
        let seq = PauliSequence::new(vec![X(0), Z(0)]);
        let result = PauliGroup::try_from(seq);
        assert_eq!(result.unwrap_err(), PauliGroupError::NonCommuting(0, 1));
    }

    #[test]
    fn try_from_sequence_imaginary_phases() {
        let seq = PauliSequence::new(vec![i * X(0), Z(1)]);
        let group = PauliGroup::try_from(seq).unwrap();
        assert_eq!(group.num_generators(), 2);
        assert!(!group.has_real_phases());
    }

    #[test]
    fn try_from_sequence_empty() {
        let seq = PauliSequence::new(vec![]);
        let group = PauliGroup::try_from(seq).unwrap();
        assert_eq!(group.num_generators(), 0);
        assert_eq!(group.num_qubits(), 0);
    }

    #[test]
    fn try_from_sequence_generates_same_group() {
        // Three generators where one is redundant
        let seq = PauliSequence::new(vec![Z(0), Z(1), Zs([0, 1])]);
        let group = PauliGroup::try_from(seq).unwrap();

        // The reduced group should contain all original elements
        assert!(group.contains_with_phase(&Z(0)));
        assert!(group.contains_with_phase(&Z(1)));
        assert!(group.contains_with_phase(&Zs([0, 1])));
    }

    #[test]
    fn try_from_set() {
        let set = PauliSet::from_iter([Z(0), Z(1), Zs([0, 1])]);
        let group = PauliGroup::try_from_set(&set).unwrap();
        assert_eq!(group.num_generators(), 2);
        assert!(group.contains_with_phase(&Z(0)));
        assert!(group.contains_with_phase(&Z(1)));
        assert!(group.contains_with_phase(&Zs([0, 1])));
    }

    #[test]
    fn try_from_set_non_commuting_fails() {
        let set = PauliSet::from_iter([X(0), Z(0)]);
        let result = PauliGroup::try_from_set(&set);
        assert!(result.is_err());
    }

    #[test]
    fn try_from_sequence_to_stabilizer() {
        let seq = PauliSequence::new(vec![Z(0), Z(1), Zs([0, 1])]);
        let stab = PauliStabilizerGroup::try_from(seq).unwrap();
        assert_eq!(stab.rank(), 2);
    }

    #[test]
    fn try_from_sequence_to_stabilizer_imaginary_fails() {
        let seq = PauliSequence::new(vec![i * X(0), Z(1)]);
        let result = PauliStabilizerGroup::try_from(seq);
        assert!(result.is_err());
    }

    #[test]
    fn try_from_sequence_to_stabilizer_non_commuting_fails() {
        let seq = PauliSequence::new(vec![X(0), Z(0)]);
        let result = PauliStabilizerGroup::try_from(seq);
        assert!(matches!(
            result.unwrap_err(),
            PauliStabilizerGroupError::NonCommuting(0, 1)
        ));
    }

    // ========================================================================
    // Into PauliSequence / PauliSet
    // ========================================================================

    #[test]
    fn group_into_sequence() {
        let group = PauliGroup::new(vec![Z(0), Z(1)]).unwrap();
        let seq = PauliSequence::from(group);
        assert_eq!(seq.len(), 2);
        assert_eq!(seq.num_qubits(), 2);
    }

    #[test]
    fn stabilizer_into_sequence() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        let seq = PauliSequence::from(stab);
        assert_eq!(seq.len(), 2);
        assert_eq!(seq.num_qubits(), 3);
    }

    #[test]
    fn group_into_set() {
        let group = PauliGroup::new(vec![Z(0), Z(1)]).unwrap();
        let set = PauliSet::from(group);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&Z(0)));
        assert!(set.contains(&Z(1)));
    }

    #[test]
    fn stabilizer_into_set() {
        let stab = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        let set = PauliSet::from(stab);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&Zs([0, 1])));
        assert!(set.contains(&Zs([1, 2])));
    }

    #[test]
    fn roundtrip_sequence_to_group_to_sequence() {
        let original = PauliSequence::new(vec![Z(0), Z(1)]);
        let group = PauliGroup::try_from(original.clone()).unwrap();
        let recovered = PauliSequence::from(group);
        assert_eq!(recovered.len(), original.len());
        assert_eq!(recovered.num_qubits(), original.num_qubits());
    }

    #[test]
    fn roundtrip_set_to_group_to_set() {
        let original = PauliSet::from_iter([Z(0), Z(1)]);
        let group = PauliGroup::try_from_set(&original).unwrap();
        let recovered = PauliSet::from(group);
        assert_eq!(recovered.len(), original.len());
        assert!(recovered.contains(&Z(0)));
        assert!(recovered.contains(&Z(1)));
    }
}
