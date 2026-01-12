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

use pecos_core::{BitSet, IndexSet, VecSet};

/// Storage for stabilizer/destabilizer generators, generic over the set type.
///
/// Uses `IndexSet` trait to allow different implementations:
/// - [`BitSet`]: O(1) toggle operations, efficient for larger circuits
/// - [`VecSet<usize>`]: Lower overhead for small sets
#[derive(Clone, Debug)]
pub struct GensGeneric<S: IndexSet> {
    num_qubits: usize,
    pub col_x: Vec<S>,
    pub col_z: Vec<S>,
    pub row_x: Vec<S>,
    pub row_z: Vec<S>,
    pub signs_minus: S,
    pub signs_i: S,
}

/// Default generator storage using `BitSet` for O(1) toggle operations.
pub type Gens = GensGeneric<BitSet>;

/// Generator storage using `BitSet` (same as `Gens`).
pub type GensBitSet = GensGeneric<BitSet>;

/// Generator storage using `VecSet<usize>` for lower overhead on small sets.
pub type GensVecSet = GensGeneric<VecSet<usize>>;

/// Hybrid generator storage using `VecSet` for Pauli data and `BitSet` for signs.
///
/// This combines the benefits of both set types:
/// - `VecSet` is faster for small sets (typical in Pauli operations)
/// - `BitSet` is faster for membership checks on sign sets during measurements
#[derive(Clone, Debug)]
pub struct GensHybrid {
    num_qubits: usize,
    pub col_x: Vec<VecSet<usize>>,
    pub col_z: Vec<VecSet<usize>>,
    pub row_x: Vec<VecSet<usize>>,
    pub row_z: Vec<VecSet<usize>>,
    pub signs_minus: BitSet,
    pub signs_i: BitSet,
}

impl GensHybrid {
    #[must_use]
    #[inline]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            col_x: (0..num_qubits).map(|_| VecSet::new()).collect(),
            col_z: (0..num_qubits).map(|_| VecSet::new()).collect(),
            row_x: (0..num_qubits).map(|_| VecSet::new()).collect(),
            row_z: (0..num_qubits).map(|_| VecSet::new()).collect(),
            // Pre-allocate BitSets to avoid resizes during measurement
            signs_minus: BitSet::with_capacity(num_qubits),
            signs_i: BitSet::with_capacity(num_qubits),
        }
    }

    #[inline]
    #[must_use]
    pub fn get_num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Clear all sign sets without reallocating the Vec storage.
    #[inline]
    fn clear_signs(&mut self) {
        self.signs_minus.clear();
        self.signs_i.clear();
    }

    /// Clear all elements in a slice of VecSets, keeping the Vec's capacity.
    #[inline]
    fn clear_sets(sets: &mut [VecSet<usize>]) {
        for set in sets.iter_mut() {
            set.clear();
        }
    }

    /// Initialize a slice of VecSets as identity (set[i] = {i}), reusing existing allocations.
    #[inline]
    fn init_as_identity(sets: &mut [VecSet<usize>]) {
        for (i, set) in sets.iter_mut().enumerate() {
            set.set_single(i);
        }
    }

    /// Ensure the Vec has exactly `num_qubits` elements, reusing capacity when possible.
    #[inline]
    fn ensure_size(sets: &mut Vec<VecSet<usize>>, num_qubits: usize) {
        match sets.len().cmp(&num_qubits) {
            std::cmp::Ordering::Less => {
                sets.reserve(num_qubits - sets.len());
                while sets.len() < num_qubits {
                    sets.push(VecSet::new());
                }
            }
            std::cmp::Ordering::Greater => {
                sets.truncate(num_qubits);
            }
            std::cmp::Ordering::Equal => {}
        }
    }

    #[inline]
    pub fn init_all_z(&mut self) {
        let n = self.get_num_qubits();

        // Ensure all Vecs have the right size
        Self::ensure_size(&mut self.col_x, n);
        Self::ensure_size(&mut self.col_z, n);
        Self::ensure_size(&mut self.row_x, n);
        Self::ensure_size(&mut self.row_z, n);

        // Clear and initialize: col_x and row_x are empty, col_z and row_z are identity
        Self::clear_sets(&mut self.col_x);
        Self::init_as_identity(&mut self.col_z);
        Self::clear_sets(&mut self.row_x);
        Self::init_as_identity(&mut self.row_z);

        self.clear_signs();
    }

    #[inline]
    pub fn init_all_x(&mut self) {
        let n = self.get_num_qubits();

        // Ensure all Vecs have the right size
        Self::ensure_size(&mut self.col_x, n);
        Self::ensure_size(&mut self.col_z, n);
        Self::ensure_size(&mut self.row_x, n);
        Self::ensure_size(&mut self.row_z, n);

        // Clear and initialize: col_x and row_x are identity, col_z and row_z are empty
        Self::init_as_identity(&mut self.col_x);
        Self::clear_sets(&mut self.col_z);
        Self::init_as_identity(&mut self.row_x);
        Self::clear_sets(&mut self.row_z);

        self.clear_signs();
    }
}

impl<S: IndexSet> GensGeneric<S> {
    #[must_use]
    #[inline]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            col_x: (0..num_qubits).map(|_| S::new()).collect(),
            col_z: (0..num_qubits).map(|_| S::new()).collect(),
            row_x: (0..num_qubits).map(|_| S::new()).collect(),
            row_z: (0..num_qubits).map(|_| S::new()).collect(),
            signs_minus: S::new(),
            signs_i: S::new(),
        }
    }

    #[inline]
    #[must_use]
    pub fn get_num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Clear all sign sets without reallocating the Vec storage.
    #[inline]
    fn clear_signs(&mut self) {
        self.signs_minus.clear();
        self.signs_i.clear();
    }

    /// Clear all elements in a slice of Sets, keeping the Vec's capacity.
    #[inline]
    fn clear_sets(sets: &mut [S]) {
        for set in sets.iter_mut() {
            set.clear();
        }
    }

    /// Initialize a slice of Sets as identity (set[i] = {i}), reusing existing allocations.
    /// Uses `set_single` to avoid the `contains()` check since we know the set is empty.
    #[inline]
    fn init_as_identity(sets: &mut [S]) {
        for (i, set) in sets.iter_mut().enumerate() {
            set.set_single(i);
        }
    }

    /// Ensure the Vec has exactly `num_qubits` elements, reusing capacity when possible.
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

    #[inline]
    pub fn init_all_z(&mut self) {
        let n = self.get_num_qubits();

        // Ensure all Vecs have the right size
        Self::ensure_size(&mut self.col_x, n);
        Self::ensure_size(&mut self.col_z, n);
        Self::ensure_size(&mut self.row_x, n);
        Self::ensure_size(&mut self.row_z, n);

        // Clear and initialize: col_x and row_x are empty, col_z and row_z are identity
        Self::clear_sets(&mut self.col_x);
        Self::init_as_identity(&mut self.col_z);
        Self::clear_sets(&mut self.row_x);
        Self::init_as_identity(&mut self.row_z);

        self.clear_signs();
    }

    #[inline]
    pub fn init_all_x(&mut self) {
        let n = self.get_num_qubits();

        // Ensure all Vecs have the right size
        Self::ensure_size(&mut self.col_x, n);
        Self::ensure_size(&mut self.col_z, n);
        Self::ensure_size(&mut self.row_x, n);
        Self::ensure_size(&mut self.row_z, n);

        // Clear and initialize: col_x and row_x are identity, col_z and row_z are empty
        Self::init_as_identity(&mut self.col_x);
        Self::clear_sets(&mut self.col_z);
        Self::init_as_identity(&mut self.row_x);
        Self::clear_sets(&mut self.row_z);

        self.clear_signs();
    }
}
