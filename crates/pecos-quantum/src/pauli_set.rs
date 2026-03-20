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

//! An unordered set of unique Pauli strings.
//!
//! [`PauliSet`] stores a set of distinct [`PauliString`]s with fast membership
//! testing and standard set operations (union, intersection, difference).
//!
//! Two [`PauliString`]s are considered equal (and thus deduplicated) when they
//! have the same Pauli operators on the same qubits **and** the same phase.
//! That is, `+XZ` and `-XZ` are distinct elements.
//!
//! Internally, each Pauli string is stored in a canonical form sorted by qubit
//! index, backed by a [`BTreeSet`] for deterministic iteration and efficient
//! comparison without hashing. This handles sparse Paulis on arbitrary qubit
//! indices efficiently -- only non-identity entries are stored.
//!
//! For an ordered sequence with symplectic analysis tools, see [`PauliSequence`].
//!
//! [`PauliString`]: pecos_core::PauliString
//! [`PauliSequence`]: crate::PauliSequence
//! [`BTreeSet`]: std::collections::BTreeSet
//!
//! # Examples
//!
//! ```
//! use pecos_quantum::PauliSet;
//! use pecos_core::pauli::constructors::*;
//!
//! let mut set = PauliSet::new();
//! set.insert(X(0));
//! set.insert(Z(1));
//! set.insert(X(0)); // duplicate, ignored
//!
//! assert_eq!(set.len(), 2);
//! assert!(set.contains(&X(0)));
//! assert!(!set.contains(&Y(0)));
//! ```

use crate::PauliSequence;
use pecos_core::{ParsePauliStringError, Pauli, PauliOperator, PauliString, QuarterPhase, QubitId};
use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

/// Canonical representation of a Pauli string for efficient set operations.
///
/// Guarantees sorted order by qubit index (no duplicates, no identity entries),
/// which gives consistent `Ord`/`Eq` regardless of how the original
/// `PauliString` was constructed.
///
/// This is sparse: only non-identity entries are stored, so a single X on
/// qubit 10000 takes the same space as X on qubit 0.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct PauliKey {
    phase: QuarterPhase,
    /// Sorted by `QubitId`, no duplicates, no identity entries.
    ops: Vec<(QubitId, Pauli)>,
}

impl PauliKey {
    fn from_pauli_string(ps: &PauliString) -> Self {
        let mut ops: Vec<(QubitId, Pauli)> = ps
            .paulis()
            .iter()
            .filter(|(p, _)| *p != Pauli::I)
            .map(|(p, q)| (*q, *p))
            .collect();
        ops.sort_by_key(|(q, _)| *q);
        let len_before = ops.len();
        ops.dedup_by_key(|(q, _)| *q);
        debug_assert_eq!(
            len_before,
            ops.len(),
            "PauliString has duplicate qubit entries; this is a bug in the PauliString constructor"
        );
        Self {
            phase: ps.phase(),
            ops,
        }
    }

    fn to_pauli_string(&self) -> PauliString {
        let paulis: Vec<(Pauli, QubitId)> = self.ops.iter().map(|(q, p)| (*p, *q)).collect();
        PauliString::with_phase_and_paulis(self.phase, paulis)
    }
}

/// An unordered set of unique [`PauliString`]s.
///
/// Backed by a [`BTreeSet`] with a canonical sorted key, providing:
/// - Deterministic iteration order
/// - Efficient comparison without hashing (sparse, only non-identity entries stored)
/// - No qubit count limit
///
/// Two Pauli strings are considered equal when they have the same operators
/// and the same phase.
///
/// [`PauliString`]: pecos_core::PauliString
/// [`BTreeSet`]: std::collections::BTreeSet
///
/// # Examples
///
/// ```
/// use pecos_quantum::PauliSet;
/// use pecos_core::pauli::constructors::*;
///
/// let a = PauliSet::from_iter([X(0), Z(1), X(0) & Z(1)]);
/// let b = PauliSet::from_iter([Z(1), Y(2)]);
///
/// let union = &a | &b;
/// assert_eq!(union.len(), 4);
///
/// let intersection = &a & &b;
/// assert_eq!(intersection.len(), 1);
/// assert!(intersection.contains(&Z(1)));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PauliSet {
    inner: BTreeSet<PauliKey>,
}

impl PauliSet {
    /// Creates an empty `PauliSet`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: BTreeSet::new(),
        }
    }

    /// Inserts a Pauli string. Returns `true` if it was not already present.
    pub fn insert(&mut self, pauli: PauliString) -> bool {
        self.inner.insert(PauliKey::from_pauli_string(&pauli))
    }

    /// Removes a Pauli string. Returns `true` if it was present.
    pub fn remove(&mut self, pauli: &PauliString) -> bool {
        self.inner.remove(&PauliKey::from_pauli_string(pauli))
    }

    /// Returns `true` if the set contains this Pauli string (exact match including phase).
    #[must_use]
    pub fn contains(&self, pauli: &PauliString) -> bool {
        self.inner.contains(&PauliKey::from_pauli_string(pauli))
    }

    /// Returns the number of Pauli strings in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterates over the Pauli strings in a deterministic order.
    pub fn iter(&self) -> impl Iterator<Item = PauliString> + '_ {
        self.inner.iter().map(PauliKey::to_pauli_string)
    }

    /// Returns the union of two sets.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.union(&other.inner).cloned().collect(),
        }
    }

    /// Returns the intersection of two sets.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.intersection(&other.inner).cloned().collect(),
        }
    }

    /// Returns elements in `self` but not in `other`.
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.difference(&other.inner).cloned().collect(),
        }
    }

    /// Returns elements in either set but not both.
    #[must_use]
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        Self {
            inner: self
                .inner
                .symmetric_difference(&other.inner)
                .cloned()
                .collect(),
        }
    }

    /// Returns `true` if `self` is a subset of `other`.
    #[must_use]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.inner.is_subset(&other.inner)
    }

    /// Returns `true` if the two sets have no elements in common.
    #[must_use]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.inner.is_disjoint(&other.inner)
    }

    /// Converts to a [`PauliSequence`] for symplectic analysis.
    ///
    /// The resulting sequence's order matches this set's deterministic iteration order.
    #[must_use]
    pub fn to_sequence(&self) -> PauliSequence {
        PauliSequence::new(self.iter().collect())
    }

    /// Returns `true` if all elements mutually commute.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliSet;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// let commuting = PauliSet::from_iter([Zs(&[0, 1]), Zs(&[1, 2])]);
    /// assert!(commuting.is_abelian());
    ///
    /// let non_commuting = PauliSet::from_iter([X(0), Z(0)]);
    /// assert!(!non_commuting.is_abelian());
    /// ```
    #[must_use]
    pub fn is_abelian(&self) -> bool {
        let paulis: Vec<PauliString> = self.iter().collect();
        for i in 0..paulis.len() {
            for j in (i + 1)..paulis.len() {
                if !paulis[i].commutes_with(&paulis[j]) {
                    return false;
                }
            }
        }
        true
    }

    /// Returns the sparse string representation in set notation.
    ///
    /// Format: `{+X0 Z2, +Z1, -Y3}` -- each element uses sparse format.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliSet;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// let set = PauliSet::from_iter([X(0), Z(1)]);
    /// let s = set.to_sparse_str();
    /// assert!(s.starts_with('{') && s.ends_with('}'));
    /// ```
    #[must_use]
    pub fn to_sparse_str(&self) -> String {
        let elements: Vec<String> = self.iter().map(|p| p.to_sparse_str()).collect();
        format!("{{{}}}", elements.join(", "))
    }

    /// Returns the dense string representation in set notation.
    ///
    /// Format: `{+XZ, +ZI, -YI}` -- each element uses dense format
    /// with phase prefix.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliSet;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// let set = PauliSet::from_iter([X(0), Z(1)]);
    /// let s = set.to_dense_str();
    /// assert!(s.starts_with('{') && s.ends_with('}'));
    /// ```
    #[must_use]
    pub fn to_dense_str(&self) -> String {
        let elements: Vec<String> = self.iter().map(|p| p.to_dense_str(None)).collect();
        format!("{{{}}}", elements.join(", "))
    }
}

impl FromStr for PauliSet {
    type Err = ParsePauliStringError;

    /// Parses a `PauliSet` from a comma-separated string, optionally wrapped in braces.
    ///
    /// Accepts: `"{X0, Z1, Y2}"`, `"X0, Z1, Y2"`, `"{XZI, IZZ}"`.
    /// Each element is parsed via [`PauliString::from_str`].
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliSet;
    /// use pecos_core::pauli::constructors::*;
    /// use std::str::FromStr;
    ///
    /// let set: PauliSet = "{X0, Z1}".parse().unwrap();
    /// assert_eq!(set.len(), 2);
    /// assert!(set.contains(&X(0)));
    /// assert!(set.contains(&Z(1)));
    ///
    /// // Braces are optional
    /// let set: PauliSet = "X0, Z1".parse().unwrap();
    /// assert_eq!(set.len(), 2);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        // Strip optional braces
        let inner = s
            .strip_prefix('{')
            .and_then(|s| s.strip_suffix('}'))
            .unwrap_or(s)
            .trim();

        if inner.is_empty() {
            return Ok(Self::new());
        }

        let paulis: Result<Vec<PauliString>, _> =
            inner.split(',').map(|elem| elem.trim().parse()).collect();
        Ok(paulis?.into_iter().collect())
    }
}

impl Default for PauliSet {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<PauliString> for PauliSet {
    fn from_iter<T: IntoIterator<Item = PauliString>>(iter: T) -> Self {
        Self {
            inner: iter
                .into_iter()
                .map(|ps| PauliKey::from_pauli_string(&ps))
                .collect(),
        }
    }
}

impl From<PauliSequence> for PauliSet {
    /// Collects all elements of the sequence into a set, deduplicating.
    fn from(seq: PauliSequence) -> Self {
        seq.paulis().iter().cloned().collect()
    }
}

impl IntoIterator for PauliSet {
    type Item = PauliString;
    type IntoIter = std::vec::IntoIter<PauliString>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner
            .into_iter()
            .map(|k| k.to_pauli_string())
            .collect::<Vec<_>>()
            .into_iter()
    }
}

// UnitaryRep overloads for set operations

impl std::ops::BitOr for &PauliSet {
    type Output = PauliSet;
    fn bitor(self, rhs: Self) -> PauliSet {
        self.union(rhs)
    }
}

impl std::ops::BitAnd for &PauliSet {
    type Output = PauliSet;
    fn bitand(self, rhs: Self) -> PauliSet {
        self.intersection(rhs)
    }
}

impl std::ops::Sub for &PauliSet {
    type Output = PauliSet;
    fn sub(self, rhs: Self) -> PauliSet {
        self.difference(rhs)
    }
}

impl std::ops::BitXor for &PauliSet {
    type Output = PauliSet;
    fn bitxor(self, rhs: Self) -> PauliSet {
        self.symmetric_difference(rhs)
    }
}

impl fmt::Display for PauliSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_sparse_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::pauli::constructors::*;

    #[test]
    fn test_new_empty() {
        let set = PauliSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn test_insert_and_contains() {
        let mut set = PauliSet::new();
        assert!(set.insert(X(0)));
        assert!(set.insert(Z(1)));
        assert!(!set.insert(X(0))); // duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&X(0)));
        assert!(set.contains(&Z(1)));
        assert!(!set.contains(&Y(0)));
    }

    #[test]
    fn test_remove() {
        let mut set = PauliSet::from_iter([X(0), Z(1)]);
        assert!(set.remove(&X(0)));
        assert!(!set.remove(&X(0))); // already removed
        assert_eq!(set.len(), 1);
        assert!(!set.contains(&X(0)));
    }

    #[test]
    fn test_phase_distinguishes() {
        // +X and -X are distinct elements
        let mut set = PauliSet::new();
        set.insert(X(0));
        set.insert(-X(0));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_from_iter() {
        let set = PauliSet::from_iter([X(0), Z(1), X(0)]);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_union() {
        let a = PauliSet::from_iter([X(0), Z(1)]);
        let b = PauliSet::from_iter([Z(1), Y(2)]);
        let u = &a | &b;
        assert_eq!(u.len(), 3);
        assert!(u.contains(&X(0)));
        assert!(u.contains(&Z(1)));
        assert!(u.contains(&Y(2)));
    }

    #[test]
    fn test_intersection() {
        let a = PauliSet::from_iter([X(0), Z(1)]);
        let b = PauliSet::from_iter([Z(1), Y(2)]);
        let i = &a & &b;
        assert_eq!(i.len(), 1);
        assert!(i.contains(&Z(1)));
    }

    #[test]
    fn test_difference() {
        let a = PauliSet::from_iter([X(0), Z(1)]);
        let b = PauliSet::from_iter([Z(1), Y(2)]);
        let d = &a - &b;
        assert_eq!(d.len(), 1);
        assert!(d.contains(&X(0)));
    }

    #[test]
    fn test_symmetric_difference() {
        let a = PauliSet::from_iter([X(0), Z(1)]);
        let b = PauliSet::from_iter([Z(1), Y(2)]);
        let sd = &a ^ &b;
        assert_eq!(sd.len(), 2);
        assert!(sd.contains(&X(0)));
        assert!(sd.contains(&Y(2)));
    }

    #[test]
    fn test_is_subset() {
        let a = PauliSet::from_iter([X(0)]);
        let b = PauliSet::from_iter([X(0), Z(1)]);
        assert!(a.is_subset(&b));
        assert!(!b.is_subset(&a));
    }

    #[test]
    fn test_is_disjoint() {
        let a = PauliSet::from_iter([X(0)]);
        let b = PauliSet::from_iter([Z(1)]);
        let c = PauliSet::from_iter([X(0), Y(2)]);
        assert!(a.is_disjoint(&b));
        assert!(!a.is_disjoint(&c));
    }

    #[test]
    fn test_to_sequence() {
        let set = PauliSet::from_iter([Zs([0, 1]), Zs([1, 2])]);
        let coll = set.to_sequence();
        assert_eq!(coll.len(), 2);
        assert_eq!(coll.rank(), 2);
    }

    #[test]
    fn test_equality() {
        let a = PauliSet::from_iter([X(0), Z(1)]);
        let b = PauliSet::from_iter([Z(1), X(0)]);
        assert_eq!(a, b); // order doesn't matter
    }

    #[test]
    fn test_into_iter() {
        let set = PauliSet::from_iter([X(0), Z(1)]);
        let collected: PauliSet = set.into_iter().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_default() {
        let set = PauliSet::default();
        assert!(set.is_empty());
    }

    #[test]
    fn test_multi_qubit() {
        let mut set = PauliSet::new();
        set.insert(X(0) & Z(1));
        set.insert(Zs([0, 1, 2]));
        set.insert(X(0) & Z(1)); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_canonical_dedup() {
        // PauliStrings built in different orders should be deduplicated
        // if they represent the same operator
        let ps1 = PauliString::with_phase_and_paulis(
            QuarterPhase::PlusOne,
            vec![(Pauli::X, QubitId::new(0)), (Pauli::Z, QubitId::new(1))],
        );
        let ps2 = PauliString::with_phase_and_paulis(
            QuarterPhase::PlusOne,
            vec![(Pauli::Z, QubitId::new(1)), (Pauli::X, QubitId::new(0))],
        );
        let set = PauliSet::from_iter([ps1, ps2]);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_sparse_high_qubit() {
        // Sparse Pauli on a high qubit index -- should work fine
        let mut set = PauliSet::new();
        set.insert(PauliString::x(10000));
        set.insert(PauliString::z(50000));
        assert_eq!(set.len(), 2);
        assert!(set.contains(&PauliString::x(10000)));
    }

    // ========================================================================
    // FromStr / to_dense_str / to_sparse_str tests
    // ========================================================================

    #[test]
    fn test_from_str_with_braces() {
        let set: PauliSet = "{X0, Z1}".parse().unwrap();
        assert_eq!(set.len(), 2);
        assert!(set.contains(&X(0)));
        assert!(set.contains(&Z(1)));
    }

    #[test]
    fn test_from_str_without_braces() {
        let set: PauliSet = "X0, Z1".parse().unwrap();
        assert_eq!(set.len(), 2);
        assert!(set.contains(&X(0)));
    }

    #[test]
    fn test_from_str_dense() {
        let set: PauliSet = "{XZI, IZZ}".parse().unwrap();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_from_str_empty() {
        let set: PauliSet = "{}".parse().unwrap();
        assert!(set.is_empty());

        let set: PauliSet = "".parse().unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn test_from_str_deduplicates() {
        let set: PauliSet = "{X0, X0, Z1}".parse().unwrap();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_to_sparse_str() {
        let set = PauliSet::from_iter([X(0), Z(1)]);
        let s = set.to_sparse_str();
        assert!(s.starts_with('{'));
        assert!(s.ends_with('}'));
        assert!(s.contains("X0"));
        assert!(s.contains("Z1"));
    }

    #[test]
    fn test_to_dense_str() {
        let set = PauliSet::from_iter([X(0), Z(1)]);
        let s = set.to_dense_str();
        assert!(s.starts_with('{'));
        assert!(s.ends_with('}'));
    }

    #[test]
    fn test_roundtrip_sparse() {
        let original = PauliSet::from_iter([X(0), Z(1), Y(2)]);
        let s = original.to_sparse_str();
        let roundtripped: PauliSet = s.parse().unwrap();
        assert_eq!(roundtripped.len(), original.len());
        assert!(roundtripped.contains(&X(0)));
        assert!(roundtripped.contains(&Z(1)));
        assert!(roundtripped.contains(&Y(2)));
    }

    #[test]
    fn test_is_abelian_commuting() {
        let set = PauliSet::from_iter([Zs([0, 1]), Zs([1, 2])]);
        assert!(set.is_abelian());
    }

    #[test]
    fn test_is_abelian_non_commuting() {
        let set = PauliSet::from_iter([X(0), Z(0)]);
        assert!(!set.is_abelian());
    }

    #[test]
    fn test_is_abelian_empty() {
        let set = PauliSet::new();
        assert!(set.is_abelian());
    }

    #[test]
    fn test_is_abelian_single() {
        let set = PauliSet::from_iter([X(0)]);
        assert!(set.is_abelian());
    }

    #[test]
    fn test_is_abelian_y_commuting() {
        // Y on different qubits commute
        let set = PauliSet::from_iter([Y(0), Y(1)]);
        assert!(set.is_abelian());
    }

    #[test]
    fn test_is_abelian_y_anticommuting() {
        // Y and Z on same qubit anticommute
        let set = PauliSet::from_iter([Y(0), Z(0)]);
        assert!(!set.is_abelian());
    }

    #[test]
    fn test_deduplication_y() {
        // Same Y added twice should deduplicate
        let set = PauliSet::from_iter([Y(0), Y(0)]);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_deduplication_phase_distinguishes() {
        // +Y(0) and -Y(0) are different elements
        let set = PauliSet::from_iter([Y(0), -Y(0)]);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_roundtrip_negative_y() {
        let original = PauliSet::from_iter([-Y(0), X(1)]);
        let s = original.to_sparse_str();
        let roundtripped: PauliSet = s.parse().unwrap();
        assert_eq!(roundtripped.len(), 2);
        assert!(roundtripped.contains(&(-Y(0))));
        assert!(roundtripped.contains(&X(1)));
    }

    #[test]
    fn test_from_sequence() {
        let seq = PauliSequence::new(vec![Z(0), Z(1), Z(0)]);
        let set = PauliSet::from(seq);
        // Z(0) appears twice in the sequence but is deduplicated in the set
        assert_eq!(set.len(), 2);
        assert!(set.contains(&Z(0)));
        assert!(set.contains(&Z(1)));
    }

    #[test]
    fn test_roundtrip_set_to_sequence_to_set() {
        let original = PauliSet::from_iter([Zs([0, 1]), Zs([1, 2])]);
        let seq = original.to_sequence();
        let recovered = PauliSet::from(seq);
        assert_eq!(recovered, original);
    }
}
