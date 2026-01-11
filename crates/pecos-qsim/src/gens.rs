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

use pecos_core::{Set, VecSet};

/// Storage for stabilizer/destabilizer generators.
///
/// Uses `VecSet<usize>` for efficient sparse storage of Pauli operators.
#[derive(Clone, Debug)]
pub struct Gens {
    num_qubits: usize,
    pub col_x: Vec<VecSet<usize>>,
    pub col_z: Vec<VecSet<usize>>,
    pub row_x: Vec<VecSet<usize>>,
    pub row_z: Vec<VecSet<usize>>,
    pub sign: VecSet<usize>,
    pub signs_minus: VecSet<usize>,
    pub signs_i: VecSet<usize>,
}

impl Gens {
    #[must_use]
    #[inline]
    pub fn new(num_qubits: usize) -> Gens {
        Self {
            num_qubits,
            col_x: vec![VecSet::new(); num_qubits],
            col_z: vec![VecSet::new(); num_qubits],
            row_x: vec![VecSet::new(); num_qubits],
            row_z: vec![VecSet::new(); num_qubits],
            sign: VecSet::new(),
            signs_minus: VecSet::new(),
            signs_i: VecSet::new(),
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
        self.sign.clear();
        self.signs_minus.clear();
        self.signs_i.clear();
    }

    /// Clear all elements in a Vec of Sets, keeping the Vec's capacity.
    #[inline]
    fn clear_sets(sets: &mut [VecSet<usize>]) {
        for set in sets.iter_mut() {
            set.clear();
        }
    }

    /// Initialize a Vec of Sets as identity (set[i] = {i}), reusing existing allocations.
    /// Uses `set_single` to avoid the contains() check since we know the set is empty.
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
