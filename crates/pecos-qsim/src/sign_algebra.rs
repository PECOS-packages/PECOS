// Copyright 2025 The PECOS Developers
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

//! Sign algebra abstractions for stabilizer simulators.
//!
//! This module provides traits and types for representing stabilizer signs in different ways:
//! - [`PhaseSign`]: Traditional phase representation (+1, -1, +i, -i) for standard simulation
//! - [`SymbolicSign`]: Measurement index sets for symbolic/deferred measurement simulation
//!
//! The [`SignAlgebra`] trait abstracts over these representations, allowing the stabilizer
//! simulator to be generic over the sign type.

use core::fmt::Debug;
use pecos_core::BitSet;

/// Trait for sign algebras used in stabilizer simulation.
///
/// This trait abstracts over different representations of stabilizer signs:
/// - Traditional phases (+1, -1, +i, -i) that collapse to concrete measurement outcomes
/// - Symbolic signs that track which measurements contribute to an outcome
///
/// # Type Parameters
/// - `Outcome`: The type returned when reading a measurement result
pub trait SignAlgebra: Clone + Debug + Default + Send + Sync {
    /// The type returned when reading a measurement outcome.
    /// For traditional signs, this is `bool` (0 or 1).
    /// For symbolic signs, this is a set of measurement indices.
    type Outcome: Clone + Debug;

    /// Returns the identity element (corresponds to +1 or empty set).
    fn identity() -> Self;

    /// Multiply two signs together.
    /// For traditional signs: phase multiplication.
    /// For symbolic signs: XOR / symmetric difference of index sets.
    #[must_use]
    fn multiply(&self, other: &Self) -> Self;

    /// Multiply-assign: self = self * other
    fn multiply_assign(&mut self, other: &Self);

    /// Create a sign representing a non-deterministic measurement outcome.
    /// For traditional signs: creates +1 or -1 based on the random outcome.
    /// For symbolic signs: creates a set containing just this measurement's index.
    fn from_measurement(measurement_index: usize, outcome: bool) -> Self;

    /// Convert the sign to a measurement outcome.
    fn to_outcome(&self) -> Self::Outcome;

    /// Check if this represents a "negative" sign (for traditional: has minus sign).
    /// For symbolic signs, this is not meaningful and returns false.
    fn is_negative(&self) -> bool;

    /// Negate the sign (flip the minus component).
    /// For symbolic signs, this is a no-op since we don't track absolute phase.
    fn negate(&mut self);
}

/// Traditional phase sign representation for standard stabilizer simulation.
///
/// Represents signs as (-1)^minus * i^imag, giving the four phases: +1, -1, +i, -i.
///
/// This is used in the standard `StdSparseStab` simulator where measurements
/// collapse to concrete 0/1 outcomes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PhaseSign {
    /// True if the sign has a factor of -1
    pub minus: bool,
    /// True if the sign has a factor of i
    pub imag: bool,
}

impl PhaseSign {
    /// Create a new phase sign with the given components.
    #[inline]
    #[must_use]
    pub fn new(minus: bool, imag: bool) -> Self {
        Self { minus, imag }
    }

    /// Create the +1 phase.
    #[inline]
    #[must_use]
    pub fn plus_one() -> Self {
        Self {
            minus: false,
            imag: false,
        }
    }

    /// Create the -1 phase.
    #[inline]
    #[must_use]
    pub fn minus_one() -> Self {
        Self {
            minus: true,
            imag: false,
        }
    }

    /// Create the +i phase.
    #[inline]
    #[must_use]
    pub fn plus_i() -> Self {
        Self {
            minus: false,
            imag: true,
        }
    }

    /// Create the -i phase.
    #[inline]
    #[must_use]
    pub fn minus_i() -> Self {
        Self {
            minus: true,
            imag: true,
        }
    }
}

impl SignAlgebra for PhaseSign {
    type Outcome = bool;

    #[inline]
    fn identity() -> Self {
        Self::plus_one()
    }

    #[inline]
    fn multiply(&self, other: &Self) -> Self {
        // (-1)^a * i^b * (-1)^c * i^d = (-1)^(a+c) * i^(b+d)
        // But i^2 = -1, so if both have i, we get an extra -1
        let extra_minus = self.imag && other.imag;
        Self {
            minus: self.minus ^ other.minus ^ extra_minus,
            imag: self.imag ^ other.imag,
        }
    }

    #[inline]
    fn multiply_assign(&mut self, other: &Self) {
        let extra_minus = self.imag && other.imag;
        self.minus ^= other.minus ^ extra_minus;
        self.imag ^= other.imag;
    }

    #[inline]
    fn from_measurement(_measurement_index: usize, outcome: bool) -> Self {
        // For traditional simulation, we just record whether the outcome is 0 or 1
        // outcome=true means we measured |1⟩, which corresponds to -Z eigenvalue
        Self {
            minus: outcome,
            imag: false,
        }
    }

    #[inline]
    fn to_outcome(&self) -> bool {
        // The measurement outcome is determined by the minus sign
        // minus=true means -1 eigenvalue, which is |1⟩
        self.minus
    }

    #[inline]
    fn is_negative(&self) -> bool {
        self.minus
    }

    #[inline]
    fn negate(&mut self) {
        self.minus = !self.minus;
    }
}

/// Symbolic sign representation for deferred/symbolic measurement simulation.
///
/// Instead of collapsing to concrete +1/-1 values, this tracks which measurements
/// contribute to the sign via XOR (symmetric difference) of measurement indices.
///
/// An empty set corresponds to +1 (identity). A set {m1, m2, ...} means the sign
/// is the XOR of the outcomes of measurements m1, m2, etc.
///
/// This is useful for:
/// - Analyzing measurement dependency graphs
/// - Understanding which measurements affect which outcomes
/// - Pauli frame tracking / deferred measurement patterns
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolicSign {
    /// Set of measurement indices whose outcomes XOR together to give this sign.
    /// Empty set = +1 (deterministic 0 outcome).
    /// Uses `BitSet` for O(words) XOR operations instead of O(n+m) with `BTreeSet`.
    pub measurements: BitSet,
}

impl SymbolicSign {
    /// Create a new symbolic sign with the given measurement indices.
    #[inline]
    #[must_use]
    pub fn new(measurements: BitSet) -> Self {
        Self { measurements }
    }

    /// Create a new symbolic sign from a `BTreeSet` (for compatibility).
    #[inline]
    #[must_use]
    pub fn from_btree_set(measurements: &std::collections::BTreeSet<usize>) -> Self {
        Self {
            measurements: BitSet::from_btree_set(measurements),
        }
    }

    /// Create an empty (identity) symbolic sign.
    #[inline]
    #[must_use]
    pub fn empty() -> Self {
        Self {
            measurements: BitSet::new(),
        }
    }

    /// Create a symbolic sign from a single measurement index.
    #[inline]
    #[must_use]
    pub fn single(measurement_index: usize) -> Self {
        Self {
            measurements: BitSet::single(measurement_index),
        }
    }
}

impl SignAlgebra for SymbolicSign {
    type Outcome = BitSet;

    #[inline]
    fn identity() -> Self {
        Self::empty()
    }

    #[inline]
    fn multiply(&self, other: &Self) -> Self {
        // XOR / symmetric difference of the measurement sets
        // BitSet provides O(words) XOR instead of O(n+m) for BTreeSet
        Self {
            measurements: &self.measurements ^ &other.measurements,
        }
    }

    #[inline]
    fn multiply_assign(&mut self, other: &Self) {
        // In-place symmetric difference using BitSet's ^= operator
        self.measurements ^= &other.measurements;
    }

    #[inline]
    fn from_measurement(measurement_index: usize, _outcome: bool) -> Self {
        // For symbolic simulation, we ignore the concrete outcome and just
        // record which measurement this is
        Self::single(measurement_index)
    }

    #[inline]
    fn to_outcome(&self) -> BitSet {
        self.measurements.clone()
    }

    #[inline]
    fn is_negative(&self) -> bool {
        // Symbolic signs don't have a concrete "negative" state
        // The "sign" is the set of measurements
        false
    }

    #[inline]
    fn negate(&mut self) {
        // No-op for symbolic signs - we don't track absolute phase
        // The sign is purely relational (XOR of measurement outcomes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_sign_multiply() {
        // +1 * +1 = +1
        assert_eq!(
            PhaseSign::plus_one().multiply(&PhaseSign::plus_one()),
            PhaseSign::plus_one()
        );

        // +1 * -1 = -1
        assert_eq!(
            PhaseSign::plus_one().multiply(&PhaseSign::minus_one()),
            PhaseSign::minus_one()
        );

        // -1 * -1 = +1
        assert_eq!(
            PhaseSign::minus_one().multiply(&PhaseSign::minus_one()),
            PhaseSign::plus_one()
        );

        // +i * +i = -1
        assert_eq!(
            PhaseSign::plus_i().multiply(&PhaseSign::plus_i()),
            PhaseSign::minus_one()
        );

        // +i * -i = +1
        assert_eq!(
            PhaseSign::plus_i().multiply(&PhaseSign::minus_i()),
            PhaseSign::plus_one()
        );

        // -1 * +i = -i
        assert_eq!(
            PhaseSign::minus_one().multiply(&PhaseSign::plus_i()),
            PhaseSign::minus_i()
        );
    }

    #[test]
    fn test_phase_sign_outcome() {
        assert!(!PhaseSign::plus_one().to_outcome());
        assert!(PhaseSign::minus_one().to_outcome());
    }

    #[test]
    fn test_symbolic_sign_multiply() {
        let s1 = SymbolicSign::single(0);
        let s2 = SymbolicSign::single(1);
        let s3 = SymbolicSign::single(0);

        // {0} * {1} = {0, 1}
        let result = s1.multiply(&s2);
        assert_eq!(result.measurements.len(), 2);
        assert!(result.measurements.contains(0));
        assert!(result.measurements.contains(1));

        // {0} * {0} = {} (XOR cancels)
        let result = s1.multiply(&s3);
        assert!(result.measurements.is_empty());

        // {} * {0} = {0}
        let result = SymbolicSign::empty().multiply(&s1);
        assert_eq!(result.measurements.len(), 1);
        assert!(result.measurements.contains(0));
    }

    #[test]
    fn test_symbolic_sign_from_measurement() {
        let sign = SymbolicSign::from_measurement(42, true);
        assert_eq!(sign.measurements.len(), 1);
        assert!(sign.measurements.contains(42));

        // Outcome is ignored for symbolic signs
        let sign2 = SymbolicSign::from_measurement(42, false);
        assert_eq!(sign, sign2);
    }
}
