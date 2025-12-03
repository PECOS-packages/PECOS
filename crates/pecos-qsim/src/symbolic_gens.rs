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
//! This module provides [`SymbolicGens`], a variant of the generator storage that tracks
//! stabilizer signs in two parts:
//! 1. **Measurement dependencies** (`signs`): Sets of measurement indices that XOR together
//! 2. **Phase flips** (`signs_minus`, `signs_i`): Traditional phase tracking from unitary gates
//!
//! The final measurement outcome is: `XOR(measurement_outcomes) XOR phase_flip`

use crate::sign_algebra::{SignAlgebra, SymbolicSign};
use core::fmt::Debug;
use core::marker::PhantomData;
use pecos_core::{IndexableElement, Set};

/// Generators for symbolic stabilizer simulation.
///
/// Tracks stabilizer signs in two parts:
/// 1. **Measurement dependencies** (`signs`): Per-generator sets of measurement indices
/// 2. **Phase tracking** (`signs_minus`, `signs_i`): Traditional phase from unitary gates
///
/// The final sign is: `{measurement_deps} ^ phase_flip` where `phase_flip` is computed
/// from `signs_minus` and `signs_i`.
#[derive(Clone, Debug)]
pub struct SymbolicGens<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    num_qubits: usize,
    /// Column-wise storage of X operators: `col_x`[qubit] = set of generator indices with X on that qubit
    pub col_x: Vec<T>,
    /// Column-wise storage of Z operators: `col_z`[qubit] = set of generator indices with Z on that qubit
    pub col_z: Vec<T>,
    /// Row-wise storage of X operators: `row_x`[gen] = set of qubits where this generator has X
    pub row_x: Vec<T>,
    /// Row-wise storage of Z operators: `row_z`[gen] = set of qubits where this generator has Z
    pub row_z: Vec<T>,
    /// Symbolic signs for each generator: signs[gen] = set of measurement indices
    pub signs: Vec<SymbolicSign>,
    /// Traditional phase tracking: generators with a minus sign (from unitaries)
    pub signs_minus: T,
    /// Traditional phase tracking: generators with an imaginary component (from unitaries)
    pub signs_i: T,
    _marker: PhantomData<E>,
}

impl<T, E> SymbolicGens<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    /// Create new symbolic generators for the given number of qubits.
    #[must_use]
    #[inline]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            col_x: vec![T::new(); num_qubits],
            col_z: vec![T::new(); num_qubits],
            row_x: vec![T::new(); num_qubits],
            row_z: vec![T::new(); num_qubits],
            signs: vec![SymbolicSign::empty(); num_qubits],
            signs_minus: T::new(),
            signs_i: T::new(),
            _marker: PhantomData,
        }
    }

    /// Get the number of qubits.
    #[inline]
    #[must_use]
    pub fn get_num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Clear all generator data.
    #[inline]
    fn clear(&mut self) {
        self.col_x.clear();
        self.col_z.clear();
        self.row_x.clear();
        self.row_z.clear();
        self.signs.clear();
        self.signs_minus.clear();
        self.signs_i.clear();
    }

    /// Initialize all generators as Z operators (stabilizers start as `Z_i` for each qubit).
    #[inline]
    pub fn init_all_z(&mut self) {
        self.clear();
        self.col_x = vec![T::new(); self.num_qubits];
        self.col_z = new_index_set::<T, E>(self.num_qubits);
        self.row_x = vec![T::new(); self.num_qubits];
        self.row_z = new_index_set::<T, E>(self.num_qubits);
        self.signs = vec![SymbolicSign::empty(); self.num_qubits];
        // signs_minus and signs_i are already cleared
    }

    /// Initialize all generators as X operators (destabilizers start as `X_i` for each qubit).
    #[inline]
    pub fn init_all_x(&mut self) {
        self.clear();
        self.col_x = new_index_set::<T, E>(self.num_qubits);
        self.col_z = vec![T::new(); self.num_qubits];
        self.row_x = new_index_set::<T, E>(self.num_qubits);
        self.row_z = vec![T::new(); self.num_qubits];
        self.signs = vec![SymbolicSign::empty(); self.num_qubits];
        // signs_minus and signs_i are already cleared
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

/// Helper function to create a vector of sets where set[i] contains just element i.
#[inline]
fn new_index_set<T, E>(num_qubits: usize) -> Vec<T>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    let mut sets = Vec::with_capacity(num_qubits);
    for i in 0..num_qubits {
        let mut set = T::new();
        set.insert(E::from_index(i));
        sets.push(set);
    }
    sets
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::VecSet;

    #[test]
    fn test_symbolic_gens_new() {
        let gens: SymbolicGens<VecSet<usize>, usize> = SymbolicGens::new(3);
        assert_eq!(gens.get_num_qubits(), 3);
        assert_eq!(gens.signs.len(), 3);
        for sign in &gens.signs {
            assert!(sign.measurements.is_empty());
        }
    }

    #[test]
    fn test_symbolic_gens_init_all_z() {
        let mut gens: SymbolicGens<VecSet<usize>, usize> = SymbolicGens::new(2);
        gens.init_all_z();

        // col_z should have generator i in set for qubit i
        assert!(gens.col_z[0].contains(&0));
        assert!(gens.col_z[1].contains(&1));

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
        let mut gens: SymbolicGens<VecSet<usize>, usize> = SymbolicGens::new(3);

        // Set some signs
        gens.set_sign(0, SymbolicSign::single(0)); // {0}
        gens.set_sign(1, SymbolicSign::single(1)); // {1}
        gens.set_sign(2, SymbolicSign::empty()); // {}

        // Multiply sign[2] by sign[0]: {} * {0} = {0}
        gens.multiply_signs(2, 0);
        assert_eq!(gens.signs[2].measurements.len(), 1);
        assert!(gens.signs[2].measurements.contains(&0));

        // Multiply sign[2] by sign[1]: {0} * {1} = {0, 1}
        gens.multiply_signs(2, 1);
        assert_eq!(gens.signs[2].measurements.len(), 2);
        assert!(gens.signs[2].measurements.contains(&0));
        assert!(gens.signs[2].measurements.contains(&1));

        // Multiply sign[2] by sign[0] again: {0, 1} * {0} = {1}
        gens.multiply_signs(2, 0);
        assert_eq!(gens.signs[2].measurements.len(), 1);
        assert!(gens.signs[2].measurements.contains(&1));
    }
}
