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

//! Symbolic generators for stabilizer states with measurement-indexed signs.
//!
//! This module provides [`SymbolicGensGeneric`], a variant of the generator storage that tracks
//! stabilizer signs in two parts:
//! 1. **Measurement dependencies** (`signs`): Sets of measurement indices that XOR together
//! 2. **Phase flips** (`signs_minus`, `signs_i`): Traditional phase tracking from unitary gates
//!
//! The final measurement outcome is: `XOR(measurement_outcomes) XOR phase_flip`
//!
//! # Set Type Parameter
//!
//! The `S` parameter controls which set implementation is used for Pauli operator storage:
//! - [`BitSet`](pecos_core::BitSet): O(1) toggle operations, better for large circuits
//! - [`VecSet<usize>`](pecos_core::VecSet): Lower overhead for small sets
//!
//! The default type alias [`SymbolicGens`] uses `BitSet` for optimal large-circuit performance.

use crate::sign_algebra::{SignAlgebra, SymbolicSign};
use pecos_core::{BitSet, IndexSet, VecSet};

/// Default symbolic generators using `BitSet` for optimal performance.
///
/// Uses O(1) toggle operations instead of O(n) linear search,
/// making it significantly faster for circuits with 100+ qubits.
pub type SymbolicGens = SymbolicGensGeneric<BitSet>;

/// Symbolic generators using `BitSet` (same as [`SymbolicGens`]).
pub type SymbolicGensBitSet = SymbolicGensGeneric<BitSet>;

/// Symbolic generators using `VecSet` for small circuits.
///
/// May have lower overhead for very small circuits (< 50 qubits),
/// but [`SymbolicGens`] (`BitSet`) is recommended for most use cases.
pub type SymbolicGensVecSet = SymbolicGensGeneric<VecSet<usize>>;

/// Generic generators for symbolic stabilizer simulation.
///
/// Tracks stabilizer signs in two parts:
/// 1. **Measurement dependencies** (`signs`): Per-generator sets of measurement indices
/// 2. **Phase tracking** (`signs_minus`, `signs_i`): Traditional phase from unitary gates
///
/// The final sign is: `{measurement_deps} ^ phase_flip` where `phase_flip` is computed
/// from `signs_minus` and `signs_i`.
///
/// # Type Parameter
///
/// - `S`: The set type used for Pauli operator storage (must implement [`IndexSet`])
#[derive(Clone, Debug)]
pub struct SymbolicGensGeneric<S: IndexSet> {
    num_qubits: usize,
    /// Column-wise storage of X operators: `col_x`[qubit] = set of generator indices with X on that qubit
    pub col_x: Vec<S>,
    /// Column-wise storage of Z operators: `col_z`[qubit] = set of generator indices with Z on that qubit
    pub col_z: Vec<S>,
    /// Row-wise storage of X operators: `row_x`[gen] = set of qubits where this generator has X
    pub row_x: Vec<S>,
    /// Row-wise storage of Z operators: `row_z`[gen] = set of qubits where this generator has Z
    pub row_z: Vec<S>,
    /// Symbolic signs for each generator: signs[gen] = set of measurement indices
    pub signs: Vec<SymbolicSign>,
    /// Traditional phase tracking: generators with a minus sign (from unitaries)
    pub signs_minus: S,
    /// Traditional phase tracking: generators with an imaginary component (from unitaries)
    pub signs_i: S,
}

impl<S: IndexSet> SymbolicGensGeneric<S> {
    /// Create new symbolic generators for the given number of qubits.
    #[must_use]
    #[inline]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            col_x: (0..num_qubits).map(|_| S::new()).collect(),
            col_z: (0..num_qubits).map(|_| S::new()).collect(),
            row_x: (0..num_qubits).map(|_| S::new()).collect(),
            row_z: (0..num_qubits).map(|_| S::new()).collect(),
            signs: vec![SymbolicSign::empty(); num_qubits],
            signs_minus: S::new(),
            signs_i: S::new(),
        }
    }

    /// Get the number of qubits.
    #[inline]
    #[must_use]
    pub fn get_num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Clear sign-related sets without reallocating.
    #[inline]
    fn clear_phase_signs(&mut self) {
        self.signs_minus.clear();
        self.signs_i.clear();
    }

    /// Clear all elements in a Vec of Sets, keeping the Vec's capacity.
    #[inline]
    fn clear_sets(sets: &mut [S]) {
        for set in sets.iter_mut() {
            set.clear();
        }
    }

    /// Initialize a Vec of Sets as identity (set[i] = {i}), reusing existing allocations.
    #[inline]
    fn init_as_identity(sets: &mut [S]) {
        for (i, set) in sets.iter_mut().enumerate() {
            set.clear();
            set.insert(i);
        }
    }

    /// Ensure the Vec of Sets has exactly `num_qubits` elements, reusing capacity when possible.
    #[inline]
    fn ensure_size(sets: &mut Vec<S>, num_qubits: usize) {
        match sets.len().cmp(&num_qubits) {
            std::cmp::Ordering::Less => {
                sets.reserve(num_qubits - sets.len());
                while sets.len() < num_qubits {
                    sets.push(S::new());
                }
            }
            std::cmp::Ordering::Greater => {
                sets.truncate(num_qubits);
            }
            std::cmp::Ordering::Equal => {}
        }
    }

    /// Ensure the Vec of `SymbolicSigns` has exactly `num_qubits` elements and clear them.
    #[inline]
    fn ensure_and_clear_signs(signs: &mut Vec<SymbolicSign>, num_qubits: usize) {
        match signs.len().cmp(&num_qubits) {
            std::cmp::Ordering::Less => {
                signs.reserve(num_qubits - signs.len());
                while signs.len() < num_qubits {
                    signs.push(SymbolicSign::empty());
                }
            }
            std::cmp::Ordering::Greater => {
                signs.truncate(num_qubits);
            }
            std::cmp::Ordering::Equal => {}
        }
        // Clear all signs to empty
        for sign in signs.iter_mut() {
            *sign = SymbolicSign::empty();
        }
    }

    /// Initialize all generators as Z operators (stabilizers start as `Z_i` for each qubit).
    #[inline]
    pub fn init_all_z(&mut self) {
        let n = self.num_qubits;

        // Ensure all Vecs have the right size
        Self::ensure_size(&mut self.col_x, n);
        Self::ensure_size(&mut self.col_z, n);
        Self::ensure_size(&mut self.row_x, n);
        Self::ensure_size(&mut self.row_z, n);
        Self::ensure_and_clear_signs(&mut self.signs, n);

        // Clear and initialize: col_x and row_x are empty, col_z and row_z are identity
        Self::clear_sets(&mut self.col_x);
        Self::init_as_identity(&mut self.col_z);
        Self::clear_sets(&mut self.row_x);
        Self::init_as_identity(&mut self.row_z);

        self.clear_phase_signs();
    }

    /// Initialize all generators as X operators (destabilizers start as `X_i` for each qubit).
    #[inline]
    pub fn init_all_x(&mut self) {
        let n = self.num_qubits;

        // Ensure all Vecs have the right size
        Self::ensure_size(&mut self.col_x, n);
        Self::ensure_size(&mut self.col_z, n);
        Self::ensure_size(&mut self.row_x, n);
        Self::ensure_size(&mut self.row_z, n);
        Self::ensure_and_clear_signs(&mut self.signs, n);

        // Clear and initialize: col_x and row_x are identity, col_z and row_z are empty
        Self::init_as_identity(&mut self.col_x);
        Self::clear_sets(&mut self.col_z);
        Self::init_as_identity(&mut self.row_x);
        Self::clear_sets(&mut self.row_z);

        self.clear_phase_signs();
    }

    /// Multiply the sign of generator `target` by the sign of generator `source`.
    /// This performs XOR (symmetric difference) of the measurement index sets.
    #[inline]
    pub fn multiply_signs(&mut self, target: usize, source: usize) {
        // We need to clone source sign to avoid borrow issues
        let source_sign = self.signs[source].clone();
        self.signs[target].multiply_assign(&source_sign);
    }

    /// Set the sign of a generator to a specific symbolic sign.
    #[inline]
    pub fn set_sign(&mut self, gen_idx: usize, sign: SymbolicSign) {
        self.signs[gen_idx] = sign;
    }

    /// Get the sign of a generator.
    #[inline]
    #[must_use]
    pub fn get_sign(&self, gen_idx: usize) -> &SymbolicSign {
        &self.signs[gen_idx]
    }

    /// Clear the sign of a generator (set to identity/empty set).
    #[inline]
    pub fn clear_sign(&mut self, gen_idx: usize) {
        self.signs[gen_idx] = SymbolicSign::empty();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_gens_new() {
        let gens = SymbolicGens::new(3);
        assert_eq!(gens.get_num_qubits(), 3);
        assert_eq!(gens.signs.len(), 3);
        for sign in &gens.signs {
            assert!(sign.measurements.is_empty());
        }
    }

    #[test]
    fn test_symbolic_gens_init_all_z() {
        let mut gens = SymbolicGens::new(2);
        gens.init_all_z();

        // col_z should have generator i in set for qubit i
        assert!(gens.col_z[0].contains(0));
        assert!(gens.col_z[1].contains(1));

        // col_x should be empty
        assert!(gens.col_x[0].is_empty());
        assert!(gens.col_x[1].is_empty());

        // All signs should be empty (identity)
        for sign in &gens.signs {
            assert!(sign.measurements.is_empty());
        }
    }

    #[test]
    fn test_symbolic_gens_multiply_signs() {
        let mut gens = SymbolicGens::new(3);

        // Set some signs
        gens.set_sign(0, SymbolicSign::single(0)); // {0}
        gens.set_sign(1, SymbolicSign::single(1)); // {1}
        gens.set_sign(2, SymbolicSign::empty()); // {}

        // Multiply sign[2] by sign[0]: {} * {0} = {0}
        gens.multiply_signs(2, 0);
        assert_eq!(gens.signs[2].measurements.len(), 1);
        assert!(gens.signs[2].measurements.contains(0));

        // Multiply sign[2] by sign[1]: {0} * {1} = {0, 1}
        gens.multiply_signs(2, 1);
        assert_eq!(gens.signs[2].measurements.len(), 2);
        assert!(gens.signs[2].measurements.contains(0));
        assert!(gens.signs[2].measurements.contains(1));

        // Multiply sign[2] by sign[0] again: {0, 1} * {0} = {1}
        gens.multiply_signs(2, 0);
        assert_eq!(gens.signs[2].measurements.len(), 1);
        assert!(gens.signs[2].measurements.contains(1));
    }

    #[test]
    fn test_symbolic_gens_vecset() {
        // Test that VecSet version also works
        let mut gens = SymbolicGensVecSet::new(2);
        gens.init_all_z();

        assert!(gens.col_z[0].contains(0));
        assert!(gens.col_z[1].contains(1));
        assert!(gens.col_x[0].is_empty());
        assert!(gens.col_x[1].is_empty());
    }
}
