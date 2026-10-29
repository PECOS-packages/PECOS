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

use pecos_core::{
    BitSet, IndexSet, Pauli, PauliOperator, PauliString, Phase, QuarterPhase, QubitId, VecSet,
};

/// Classification of a Pauli operator relative to a stabilizer state.
///
/// When checking if a Pauli operator is in the stabilizer group, there are three possibilities:
/// - It is a stabilizer (can be built from the generators)
/// - It is a logical operator (commutes with all stabilizers but is not in the group)
/// - It is an error (anticommutes with at least one stabilizer)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauliClassification {
    /// The operator is in the stabilizer group (can be built from generators).
    Stabilizer,
    /// The operator anticommutes with at least one stabilizer.
    Error,
    /// The operator commutes with all stabilizers but is not in the stabilizer group.
    /// This indicates a logical operator.
    Logical,
}

/// Meaning of overlapping X and Z support in a generator tableau.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauliConvention {
    /// X and Z on the same qubit mean XZ = -iY.
    W,
    /// X and Z on the same qubit mean Y = iXZ.
    Y,
}

/// Storage for stabilizer/destabilizer generators, generic over the set type.
///
/// Uses `IndexSet` trait to allow different implementations:
/// - [`AdaptiveIndexSet`](pecos_core::AdaptiveIndexSet): default, adaptive dense/sparse storage
/// - [`BitSet`]: dense words with O(1) toggle operations
/// - [`VecSet<usize>`]: Lower overhead for small sets
#[derive(Clone, Debug)]
pub struct GensGeneric<S: IndexSet> {
    num_qubits: usize,
    convention: PauliConvention,
    pub col_x: Vec<S>,
    pub col_z: Vec<S>,
    pub row_x: Vec<S>,
    pub row_z: Vec<S>,
    pub signs_minus: S,
    pub signs_i: S,
}

/// Default generator storage using adaptive dense/sparse index sets.
pub type Gens = GensGeneric<crate::SparseStabSet>;

/// Generator storage using explicit `BitSet` storage.
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

    /// Clear all elements in a slice of `VecSets`, keeping the Vec's capacity.
    #[inline]
    fn clear_sets(sets: &mut [VecSet<usize>]) {
        for set in sets.iter_mut() {
            set.clear();
        }
    }

    /// Initialize a slice of `VecSets` as identity (set[i] = {i}), reusing existing allocations.
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

    // ========================================================================
    // Generator extraction methods
    // ========================================================================

    /// Returns the number of generators stored.
    #[inline]
    #[must_use]
    pub fn num_generators(&self) -> usize {
        self.row_x.len()
    }

    /// Computes the phase of generator `i` from the sign bits.
    #[inline]
    #[must_use]
    pub fn generator_phase(&self, i: usize) -> QuarterPhase {
        match (self.signs_minus.contains(i), self.signs_i.contains(i)) {
            (false, false) => QuarterPhase::PlusOne,
            (true, false) => QuarterPhase::MinusOne,
            (false, true) => QuarterPhase::PlusI,
            (true, true) => QuarterPhase::MinusI,
        }
    }

    /// Extracts generator `i` as a [`PauliString`].
    ///
    /// # Panics
    /// Panics if `i >= num_generators()`.
    #[must_use]
    pub fn generator(&self, i: usize) -> PauliString {
        assert!(i < self.num_generators(), "generator index out of bounds");
        let mut phase = self.generator_phase(i);
        let mut paulis = Vec::new();
        let mut num_y_terms = 0usize;
        for q in 0..self.num_qubits {
            let has_x = self.row_x[i].contains(q);
            let has_z = self.row_z[i].contains(q);
            let pauli = match (has_x, has_z) {
                (false, false) => continue,
                (true, false) => Pauli::X,
                (false, true) => Pauli::Z,
                (true, true) => {
                    num_y_terms += 1;
                    Pauli::Y
                }
            };
            paulis.push((pauli, QubitId::new(q)));
        }
        for _ in 0..num_y_terms {
            phase = phase.multiply(&QuarterPhase::MinusI);
        }
        PauliString::with_phase_and_paulis(phase, paulis)
            .expect("tableau conversion emits one Pauli per qubit")
    }

    /// Extracts all generators as a `Vec<PauliString>`.
    #[must_use]
    pub fn generators(&self) -> Vec<PauliString> {
        (0..self.num_generators())
            .map(|i| self.generator(i))
            .collect()
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
    /// Create empty generators in the W convention (X and Z mean XZ).
    #[must_use]
    #[inline]
    pub fn new(num_qubits: usize) -> Self {
        Self::with_convention(num_qubits, PauliConvention::W)
    }

    /// Create empty generators with a fixed Pauli storage convention.
    #[must_use]
    #[inline]
    pub fn with_convention(num_qubits: usize, convention: PauliConvention) -> Self {
        Self {
            num_qubits,
            convention,
            col_x: (0..num_qubits).map(|_| S::new()).collect(),
            col_z: (0..num_qubits).map(|_| S::new()).collect(),
            row_x: (0..num_qubits).map(|_| S::new()).collect(),
            row_z: (0..num_qubits).map(|_| S::new()).collect(),
            signs_minus: S::new(),
            signs_i: S::new(),
        }
    }

    /// Create a new `GensGeneric` in the W convention from pre-populated parts.
    ///
    /// For Y-convention generators, build with [`Self::with_convention`] and
    /// fill the public fields.
    ///
    /// # Safety
    /// The caller must ensure all vectors have length `num_qubits`.
    #[must_use]
    #[inline]
    pub fn from_parts(
        num_qubits: usize,
        col_x: Vec<S>,
        col_z: Vec<S>,
        row_x: Vec<S>,
        row_z: Vec<S>,
        signs_minus: S,
        signs_i: S,
    ) -> Self {
        debug_assert_eq!(col_x.len(), num_qubits);
        debug_assert_eq!(col_z.len(), num_qubits);
        debug_assert_eq!(row_x.len(), num_qubits);
        debug_assert_eq!(row_z.len(), num_qubits);
        Self {
            num_qubits,
            convention: PauliConvention::W,
            col_x,
            col_z,
            row_x,
            row_z,
            signs_minus,
            signs_i,
        }
    }

    /// Returns the fixed convention used by this tableau's rows and sign bits.
    #[inline]
    #[must_use]
    pub fn convention(&self) -> PauliConvention {
        self.convention
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

    // ========================================================================
    // Generator extraction methods
    // ========================================================================

    /// Returns the number of generators stored.
    #[inline]
    #[must_use]
    pub fn num_generators(&self) -> usize {
        self.row_x.len()
    }

    /// Returns the raw phase bits of generator `i` in this tableau's convention.
    ///
    /// Use [`Self::generator`] for the phase of the Pauli written with explicit Ys.
    #[inline]
    #[must_use]
    pub fn generator_phase(&self, i: usize) -> QuarterPhase {
        match (self.signs_minus.contains(i), self.signs_i.contains(i)) {
            (false, false) => QuarterPhase::PlusOne,
            (true, false) => QuarterPhase::MinusOne,
            (false, true) => QuarterPhase::PlusI,
            (true, true) => QuarterPhase::MinusI,
        }
    }

    /// Extracts generator `i` as a `PauliString`.
    ///
    /// Returns the Pauli operator for the i-th generator (stabilizer or destabilizer,
    /// depending on which `Gens` this is), including its phase. W rows contribute
    /// a factor of -i per Y position; Y rows already store the explicit Pauli phase.
    ///
    /// In either convention, destabilizer phases are meaningful only while their
    /// signs are maintained: sign tracking must be on, with no measurement or
    /// refactor since.
    ///
    /// # Panics
    /// Panics if `i >= num_generators()`.
    #[must_use]
    pub fn generator(&self, i: usize) -> PauliString {
        assert!(i < self.num_generators(), "generator index out of bounds");

        let mut phase = self.generator_phase(i);

        // Collect non-identity Paulis
        let mut paulis = Vec::new();
        let mut num_y_terms = 0usize;

        // Iterate over all qubits and determine the Pauli at each position
        for q in 0..self.num_qubits {
            let has_x = self.row_x[i].contains(q);
            let has_z = self.row_z[i].contains(q);

            let pauli = match (has_x, has_z) {
                (false, false) => continue, // Identity, skip
                (true, false) => Pauli::X,
                (false, true) => Pauli::Z,
                (true, true) => {
                    num_y_terms += 1;
                    Pauli::Y
                }
            };

            paulis.push((pauli, QubitId::new(q)));
        }
        if self.convention == PauliConvention::W {
            for _ in 0..num_y_terms % 4 {
                phase = phase.multiply(&QuarterPhase::MinusI);
            }
        }

        PauliString::with_phase_and_paulis(phase, paulis)
            .expect("tableau conversion emits one Pauli per qubit")
    }

    /// Extracts all generators as explicit Paulis, converting this tableau's convention.
    ///
    /// In either convention, destabilizer phases are meaningful only while their
    /// signs are maintained: sign tracking must be on, with no measurement or
    /// refactor since. See [`Self::generator`].
    #[must_use]
    pub fn generators(&self) -> Vec<PauliString> {
        (0..self.num_generators())
            .map(|i| self.generator(i))
            .collect()
    }

    // ========================================================================
    // Commutation checking methods
    // ========================================================================

    /// Checks if a Pauli operator commutes with all generators.
    ///
    /// This is useful for determining if an operator is in the stabilizer group
    /// (if it commutes with all stabilizers).
    ///
    /// # Parameters
    /// - `x_positions`: Iterator of qubit indices with X component
    /// - `z_positions`: Iterator of qubit indices with Z component
    ///
    /// # Returns
    /// `true` if the operator commutes with all generators, `false` otherwise.
    pub fn commutes_with_all<I, J>(&self, x_positions: I, z_positions: J) -> bool
    where
        I: IntoIterator<Item = usize>,
        J: IntoIterator<Item = usize>,
    {
        // Collect positions that anticommute with generators
        // A Pauli P anticommutes with generator G if |P_X ∩ G_Z| + |P_Z ∩ G_X| is odd

        let x_pos: Vec<usize> = x_positions.into_iter().collect();
        let z_pos: Vec<usize> = z_positions.into_iter().collect();

        // Check each generator
        for i in 0..self.num_generators() {
            let mut anticommute_count = 0;

            // Count X positions of P that overlap with Z positions of G
            for &q in &x_pos {
                if q < self.num_qubits && self.row_z[i].contains(q) {
                    anticommute_count += 1;
                }
            }

            // Count Z positions of P that overlap with X positions of G
            for &q in &z_pos {
                if q < self.num_qubits && self.row_x[i].contains(q) {
                    anticommute_count += 1;
                }
            }

            // If odd, they anticommute
            if anticommute_count % 2 != 0 {
                return false;
            }
        }

        true
    }

    /// Checks if a `PauliString` commutes with all generators.
    pub fn commutes_with_pauli(&self, pauli: &PauliString) -> bool {
        self.commutes_with_all(pauli.x_positions(), pauli.z_positions())
    }

    /// Returns ascending indices of generators that anticommute with the given Pauli operator.
    ///
    /// # Parameters
    /// - `x_positions`: Iterator of qubit indices with X component
    /// - `z_positions`: Iterator of qubit indices with Z component
    pub fn anticommuting_indices<I, J>(&self, x_positions: I, z_positions: J) -> Vec<usize>
    where
        I: IntoIterator<Item = usize>,
        J: IntoIterator<Item = usize>,
    {
        let x_pos: Vec<usize> = x_positions.into_iter().collect();
        let z_pos: Vec<usize> = z_positions.into_iter().collect();

        let mut result = Vec::new();

        for i in 0..self.num_generators() {
            let mut anticommute_count = 0;

            for &q in &x_pos {
                if q < self.num_qubits && self.row_z[i].contains(q) {
                    anticommute_count += 1;
                }
            }

            for &q in &z_pos {
                if q < self.num_qubits && self.row_x[i].contains(q) {
                    anticommute_count += 1;
                }
            }

            if anticommute_count % 2 != 0 {
                result.push(i);
            }
        }

        result
    }

    // ========================================================================
    // Stabilizer group membership methods
    // ========================================================================

    /// Classifies a Pauli operator relative to this stabilizer state.
    ///
    /// Given the stabilizers (`self`) and destabilizers, determines whether the
    /// Pauli operator is:
    /// - A stabilizer (can be built from the generators)
    /// - A logical operator (commutes with all stabilizers but is not in the group)
    /// - An error (anticommutes with at least one stabilizer)
    ///
    /// # Parameters
    /// - `destabs`: The destabilizer generators (paired with these stabilizers)
    /// - `x_positions`: Iterator of qubit indices with X component
    /// - `z_positions`: Iterator of qubit indices with Z component
    ///
    /// # Returns
    /// The classification of the Pauli operator.
    pub fn classify_pauli<I, J>(
        &self,
        destabs: &GensGeneric<S>,
        x_positions: I,
        z_positions: J,
    ) -> PauliClassification
    where
        I: IntoIterator<Item = usize>,
        J: IntoIterator<Item = usize>,
    {
        let x_pos: Vec<usize> = x_positions.into_iter().collect();
        let z_pos: Vec<usize> = z_positions.into_iter().collect();

        // Step 1: Check if the operator commutes with all stabilizers using column lookup.
        // This is O(weight) instead of O(num_stabilizers * weight).
        //
        // For each X position q, XOR together col_z[q] (stabilizers with Z on q anticommute with X)
        // For each Z position q, XOR together col_x[q] (stabilizers with X on q anticommute with Z)
        // The XOR automatically handles the mod-2 counting.
        let mut anticom_stabs: S = S::new();

        for &q in &x_pos {
            if q < self.col_z.len() {
                for stab_id in self.col_z[q].iter() {
                    anticom_stabs.toggle(stab_id);
                }
            }
        }

        for &q in &z_pos {
            if q < self.col_x.len() {
                for stab_id in self.col_x[q].iter() {
                    anticom_stabs.toggle(stab_id);
                }
            }
        }

        // If any stabilizers remain after XOR, the operator anticommutes with them
        if !anticom_stabs.is_empty() {
            return PauliClassification::Error;
        }

        // Step 2: Try to build the operator from stabilizers using destabilizers.
        // For each X position q, collect stabilizers pointed to by destabs.col_z[q]
        // For each Z position q, collect stabilizers pointed to by destabs.col_x[q]
        let mut build_stabs: S = S::new();

        for &q in &x_pos {
            if q < destabs.col_z.len() {
                for stab_id in destabs.col_z[q].iter() {
                    build_stabs.toggle(stab_id);
                }
            }
        }

        for &q in &z_pos {
            if q < destabs.col_x.len() {
                for stab_id in destabs.col_x[q].iter() {
                    build_stabs.toggle(stab_id);
                }
            }
        }

        // Step 3: Build up the X and Z Paulis from those stabilizers and compare.
        // Instead of building and comparing, we can XOR directly with the input positions.
        // If the result is empty, they match.
        let mut diff_x: S = S::new();
        let mut diff_z: S = S::new();

        // Start with input positions
        for &q in &x_pos {
            diff_x.toggle(q);
        }
        for &q in &z_pos {
            diff_z.toggle(q);
        }

        // XOR with built stabilizers - if they match, result will be empty
        for stab_id in build_stabs.iter() {
            for q in self.row_x[stab_id].iter() {
                diff_x.toggle(q);
            }
            for q in self.row_z[stab_id].iter() {
                diff_z.toggle(q);
            }
        }

        if diff_x.is_empty() && diff_z.is_empty() {
            PauliClassification::Stabilizer
        } else {
            // Commutes but cannot be built from stabilizers -> logical operator
            PauliClassification::Logical
        }
    }

    /// Classifies a `PauliString` relative to this stabilizer state.
    ///
    /// Convenience method that extracts X and Z positions from a `PauliString`.
    pub fn classify_pauli_string(
        &self,
        destabs: &GensGeneric<S>,
        pauli: &PauliString,
    ) -> PauliClassification {
        self.classify_pauli(destabs, pauli.x_positions(), pauli.z_positions())
    }

    /// Checks if a Pauli operator is in the stabilizer group.
    ///
    /// Returns `true` if the operator can be built from the stabilizer generators.
    ///
    /// # Parameters
    /// - `destabs`: The destabilizer generators (paired with these stabilizers)
    /// - `x_positions`: Iterator of qubit indices with X component
    /// - `z_positions`: Iterator of qubit indices with Z component
    pub fn is_in_stabilizer_group<I, J>(
        &self,
        destabs: &GensGeneric<S>,
        x_positions: I,
        z_positions: J,
    ) -> bool
    where
        I: IntoIterator<Item = usize>,
        J: IntoIterator<Item = usize>,
    {
        self.classify_pauli(destabs, x_positions, z_positions) == PauliClassification::Stabilizer
    }

    /// Checks if a `PauliString` is in the stabilizer group.
    ///
    /// Convenience method that extracts X and Z positions from a `PauliString`.
    pub fn is_pauli_string_in_group(&self, destabs: &GensGeneric<S>, pauli: &PauliString) -> bool {
        self.classify_pauli_string(destabs, pauli) == PauliClassification::Stabilizer
    }

    // ========================================================================
    // Sign computation
    // ========================================================================

    /// Finds which stabilizer generators to multiply to produce a given Pauli operator.
    ///
    /// Uses destabilizer column lookup to efficiently find the required generators.
    /// The returned set contains the indices of stabilizer generators that, when multiplied
    /// together, produce the given Pauli operator (ignoring sign).
    ///
    /// Returns `None` if the operator is not in the stabilizer group.
    ///
    /// # Parameters
    /// - `destabs`: The destabilizer generators (paired with these stabilizers)
    /// - `x_positions`: Iterator of qubit indices with X component
    /// - `z_positions`: Iterator of qubit indices with Z component
    pub fn find_stabilizers_for_pauli<I, J>(
        &self,
        destabs: &GensGeneric<S>,
        x_positions: I,
        z_positions: J,
    ) -> Option<S>
    where
        I: IntoIterator<Item = usize> + Clone,
        J: IntoIterator<Item = usize> + Clone,
    {
        let x_pos: Vec<usize> = x_positions.clone().into_iter().collect();
        let z_pos: Vec<usize> = z_positions.clone().into_iter().collect();

        // First check if it's in the stabilizer group
        if self.classify_pauli(destabs, x_positions, z_positions) != PauliClassification::Stabilizer
        {
            return None;
        }

        // Build the set of stabilizers using destabilizer column lookup
        let mut build_stabs: S = S::new();

        for &q in &x_pos {
            if q < destabs.col_z.len() {
                for stab_id in destabs.col_z[q].iter() {
                    build_stabs.toggle(stab_id);
                }
            }
        }

        for &q in &z_pos {
            if q < destabs.col_x.len() {
                for stab_id in destabs.col_x[q].iter() {
                    build_stabs.toggle(stab_id);
                }
            }
        }

        Some(build_stabs)
    }

    /// Computes the sign when multiplying a set of stabilizer generators together.
    ///
    /// This accounts for:
    /// 1. The individual signs (±1, ±i) of each generator
    /// 2. The sign changes from reordering Paulis (ZX → -XZ)
    ///
    /// Factors are multiplied in `stab_indices` iteration order (irrelevant for
    /// commuting stabilizers). In the Y convention, each factor is converted to
    /// W form with `i^{n_Y}`, then the product is converted back with `(-i)^{n_Y}`.
    /// Stored imaginary phases contribute in both conventions.
    ///
    /// # Returns
    /// The product's phase in this tableau's storage convention: the sign bits
    /// needed for a generator whose rows are the XOR of the selected rows.
    #[must_use]
    pub fn compute_product_sign(&self, stab_indices: &S) -> QuarterPhase {
        // Count minus signs from individual stabilizers
        let mut num_minuses = 0usize;
        let mut num_is = 0usize;

        for stab_id in stab_indices.iter() {
            if self.signs_minus.contains(stab_id) {
                num_minuses += 1;
            }
            if self.signs_i.contains(stab_id) {
                num_is += 1;
            }
        }

        // In forward product order, the accumulated Zs cross the next factor's Xs.
        let mut cumulative_z: S = S::new();
        for stab_id in stab_indices.iter() {
            num_minuses += cumulative_z.intersection_count(&self.row_x[stab_id]);
            cumulative_z.xor_assign(&self.row_z[stab_id]);
        }

        if self.convention == PauliConvention::Y {
            let mut cumulative_x: S = S::new();
            for stab_id in stab_indices.iter() {
                num_is += self.row_x[stab_id].intersection_count(&self.row_z[stab_id]);
                cumulative_x.xor_assign(&self.row_x[stab_id]);
            }
            // Convert the W product back to Y storage, modulo four to avoid underflow.
            num_is += (4 - cumulative_x.intersection_count(&cumulative_z) % 4) % 4;
        }

        // Convert imaginary count to phase contribution
        // i^0 = 1, i^1 = i, i^2 = -1, i^3 = -i
        match num_is % 4 {
            2 | 3 => num_minuses += 1,
            _ => {}
        }

        // Compute final phase
        let is_minus = num_minuses % 2 == 1;
        let is_imag = num_is % 2 == 1;

        match (is_minus, is_imag) {
            (false, false) => QuarterPhase::PlusOne,
            (false, true) => QuarterPhase::PlusI,
            (true, false) => QuarterPhase::MinusOne,
            (true, true) => QuarterPhase::MinusI,
        }
    }

    /// Computes the sign of a Pauli operator that is in the stabilizer group.
    ///
    /// This combines finding which generators produce the operator with computing
    /// the resulting sign, converting from this tableau's storage convention
    /// to an operator with `num_ys` explicit Y factors. A Y-storage phase P is
    /// multiplied by `i^{|X intersection Z|} * (-i)^{num_ys}`; a W-storage phase
    /// is multiplied by `(-i)^{num_ys}`. When `num_ys` is the actual overlap count,
    /// both conventions return the same explicit Pauli phase.
    ///
    /// # Parameters
    /// - `destabs`: The destabilizer generators (paired with these stabilizers)
    /// - `x_positions`: Iterator of qubit indices with X component
    /// - `z_positions`: Iterator of qubit indices with Z component
    /// - `num_ys`: Number of Y positions in the original operator (needed for sign correction)
    ///
    /// # Returns
    /// `Some(phase)` if the operator is in the stabilizer group, `None` otherwise.
    pub fn find_pauli_sign<I, J>(
        &self,
        destabs: &GensGeneric<S>,
        x_positions: I,
        z_positions: J,
        num_ys: usize,
    ) -> Option<QuarterPhase>
    where
        I: IntoIterator<Item = usize> + Clone,
        J: IntoIterator<Item = usize> + Clone,
    {
        let stab_indices =
            self.find_stabilizers_for_pauli(destabs, x_positions.clone(), z_positions.clone())?;

        let mut phase = self.compute_product_sign(&stab_indices);

        // Convert a Y-storage product to W form before applying the explicit Y count.
        if self.convention == PauliConvention::Y {
            let mut xs = S::new();
            let mut zs = S::new();
            for q in x_positions {
                xs.insert(q);
            }
            for q in z_positions {
                zs.insert(q);
            }
            for _ in 0..xs.intersection_count(&zs) % 4 {
                phase = phase.multiply(&QuarterPhase::PlusI);
            }
        }
        // Each explicit Y = iXZ removes one i from the W-form coefficient.
        let y_phase_correction = num_ys % 4;
        for _ in 0..y_phase_correction {
            phase = phase.multiply(&QuarterPhase::MinusI);
        }

        Some(phase)
    }

    /// Computes the sign of a `PauliString` that is in the stabilizer group.
    ///
    /// Extracts positions and the true Y count, returning the same explicit Pauli
    /// phase in either storage convention. The input's phase is ignored.
    pub fn find_pauli_string_sign(
        &self,
        destabs: &GensGeneric<S>,
        pauli: &PauliString,
    ) -> Option<QuarterPhase> {
        // Y positions are where both X and Z are present
        let x_pos: std::collections::BTreeSet<usize> = pauli.x_positions().into_iter().collect();
        let z_pos: std::collections::BTreeSet<usize> = pauli.z_positions().into_iter().collect();
        let num_ys = x_pos.intersection(&z_pos).count();
        self.find_pauli_sign(destabs, pauli.x_positions(), pauli.z_positions(), num_ys)
    }

    // ========================================================================
    // Generator refactoring
    // ========================================================================

    /// Finds the stabilizer generators that anticommute with a given Pauli operator.
    ///
    /// This is useful for identifying which generators would need to be modified
    /// when measuring or refactoring.
    ///
    /// # Returns
    /// A set of generator indices that anticommute with the given operator.
    pub fn find_anticommuting_generators<I, J>(&self, x_positions: I, z_positions: J) -> S
    where
        I: IntoIterator<Item = usize>,
        J: IntoIterator<Item = usize>,
    {
        let mut anticom: S = S::new();

        for q in x_positions {
            if q < self.col_z.len() {
                for stab_id in self.col_z[q].iter() {
                    anticom.toggle(stab_id);
                }
            }
        }

        for q in z_positions {
            if q < self.col_x.len() {
                for stab_id in self.col_x[q].iter() {
                    anticom.toggle(stab_id);
                }
            }
        }

        anticom
    }

    /// Returns the weight (number of non-identity Paulis) of a generator.
    ///
    /// The weight is the total count of X and Z positions. Note that Y positions
    /// are counted twice (once for X, once for Z), which matches the standard
    /// definition of Pauli weight.
    #[must_use]
    #[inline]
    pub fn generator_weight(&self, gen_id: usize) -> usize {
        self.row_x[gen_id].len() + self.row_z[gen_id].len()
    }

    /// Finds the generator with minimum (weight, id) from a set of candidates.
    ///
    /// This is used to keep the tableau sparse by preferring low-weight generators
    /// when we have a choice of which to modify.
    ///
    /// # Panics
    /// Panics if `candidates` is empty.
    #[must_use]
    fn min_weight_generator(&self, candidates: &S) -> usize {
        candidates
            .iter()
            .min_by_key(|&gen_id| (self.generator_weight(gen_id), gen_id))
            .expect("candidates must not be empty")
    }

    /// Refactors the stabilizer generators so that a given Pauli becomes a generator.
    ///
    /// Given a Pauli operator that is in the stabilizer group, this modifies the
    /// generators so that the given operator becomes one of the generators.
    /// The new stabilizer's sign bits use this tableau's storage convention.
    /// Destabilizer signs are left unchanged, as during measurement; only gates
    /// maintain them when destabilizer sign tracking is enabled.
    ///
    /// # Parameters
    /// - `destabs`: The destabilizer generators (will be modified)
    /// - `x_positions`: Iterator of qubit indices with X component
    /// - `z_positions`: Iterator of qubit indices with Z component
    /// - `prefer`: Optional list of generator indices to prefer choosing
    /// - `protected`: Optional set of generator indices that must not be replaced
    ///
    /// # Returns
    /// `Some(gen_id)` with the index of the generator that was replaced, or `None`
    /// if the operator is not in the stabilizer group or all generators are protected.
    #[allow(clippy::missing_panics_doc)] // Panic is impossible: we return None if available is empty
    pub fn refactor<I, J>(
        &mut self,
        destabs: &mut GensGeneric<S>,
        x_positions: I,
        z_positions: J,
        prefer: Option<&[usize]>,
        protected: Option<&S>,
    ) -> Option<usize>
    where
        I: IntoIterator<Item = usize> + Clone,
        J: IntoIterator<Item = usize> + Clone,
    {
        // Find the stabilizers that build this operator
        let mut build_stabs = self.find_stabilizers_for_pauli(destabs, x_positions, z_positions)?;

        // Remove protected generators from consideration
        let mut available = build_stabs.clone();
        if let Some(prot) = protected {
            for p in prot.iter() {
                available.remove(p);
            }
        }

        if available.is_empty() {
            // All generators are protected
            return None;
        }

        // Choose which generator to replace, preferring minimum weight to keep tableau sparse
        let new_stab = if let Some(pref) = prefer {
            // Try preferred generators first
            let mut chosen = None;
            for &p in pref {
                if available.contains(p) {
                    chosen = Some(p);
                    break;
                }
            }
            // Fall back to minimum weight generator
            chosen.unwrap_or_else(|| self.min_weight_generator(&available))
        } else {
            // Choose minimum weight generator to keep tableau sparse
            self.min_weight_generator(&available)
        };

        // Compute the full product phase, including Pauli reordering, before
        // removing the chosen generator or mutating any of the source rows.
        let product_sign = self.compute_product_sign(&build_stabs);

        // Remove the chosen generator from the set (it will become the new operator)
        build_stabs.remove(new_stab);

        // Collect generator indices to avoid borrow issues
        let gens: Vec<usize> = build_stabs.iter().collect();

        // Update stabilizer generators:
        // For each remaining generator g in build_stabs: stab[new_stab] *= stab[g]
        for &g in &gens {
            // Collect positions first to avoid borrow conflicts
            let x_positions: Vec<usize> = self.row_x[g].iter().collect();
            let z_positions: Vec<usize> = self.row_z[g].iter().collect();

            // Update column indices
            for &q in &x_positions {
                self.col_x[q].toggle(new_stab);
            }
            for &q in &z_positions {
                self.col_z[q].toggle(new_stab);
            }
            // Update row (the actual Pauli)
            for &q in &x_positions {
                self.row_x[new_stab].toggle(q);
            }
            for &q in &z_positions {
                self.row_z[new_stab].toggle(q);
            }
        }

        // Update destabilizer generators:
        // For each g in build_stabs: destab[g] *= destab[new_stab]
        // First collect the positions from new_stab's destabilizer
        let destab_new_stab_x: Vec<usize> = destabs.row_x[new_stab].iter().collect();
        let destab_new_stab_z: Vec<usize> = destabs.row_z[new_stab].iter().collect();

        for &q in &destab_new_stab_x {
            for &g in &gens {
                destabs.col_x[q].toggle(g);
            }
        }
        for &q in &destab_new_stab_z {
            for &g in &gens {
                destabs.col_z[q].toggle(g);
            }
        }
        for &g in &gens {
            for &q in &destab_new_stab_x {
                destabs.row_x[g].toggle(q);
            }
            for &q in &destab_new_stab_z {
                destabs.row_z[g].toggle(q);
            }
        }

        // Assign the product phase in the tableau's storage convention.
        if matches!(product_sign, QuarterPhase::MinusOne | QuarterPhase::MinusI) {
            self.signs_minus.insert(new_stab);
        } else {
            self.signs_minus.remove(new_stab);
        }
        if matches!(product_sign, QuarterPhase::PlusI | QuarterPhase::MinusI) {
            self.signs_i.insert(new_stab);
        } else {
            self.signs_i.remove(new_stab);
        }

        Some(new_stab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CliffordGateable, SparseStab, SparseStabBitSet, SparseStabY};
    use pecos_random::PecosRng;
    use rand::RngExt;

    fn apply_convention_gate<T: CliffordGateable>(
        state: &mut T,
        gate: usize,
        first: QubitId,
        second: QubitId,
    ) {
        match gate {
            0 => state.h(&[first]),
            1 => state.sz(&[first]),
            2 => state.sy(&[first]),
            3 => state.cx(&[(first, second)]),
            4 => state.cy(&[(first, second)]),
            5 => state.cz(&[(first, second)]),
            6 => state.sxx(&[(first, second)]),
            7 => state.syy(&[(first, second)]),
            8 => state.szz(&[(first, second)]),
            _ => state.iswap(&[(first, second)]),
        };
    }

    fn assert_convention_rows(w: &SparseStabBitSet, y: &SparseStabY) {
        for (w_gens, y_gens) in [(w.stabs(), y.stabs()), (w.destabs(), y.destabs())] {
            assert_eq!(w_gens.row_x, y_gens.row_x, "X row oracle precondition");
            assert_eq!(w_gens.row_z, y_gens.row_z, "Z row oracle precondition");
            assert_eq!(w_gens.col_x, y_gens.col_x, "X column oracle precondition");
            assert_eq!(w_gens.col_z, y_gens.col_z, "Z column oracle precondition");
        }
    }

    fn convention_circuits() -> Vec<(SparseStabBitSet, SparseStabY)> {
        let mut rng = PecosRng::seed_from_u64(0x5eed_2026);
        (0..140)
            .map(|trial| {
                let num_qubits = 2 + trial % 7;
                let mut w = SparseStabBitSet::new(num_qubits);
                let mut y = SparseStabY::new(num_qubits);
                for step in 0..80 {
                    let first = QubitId(rng.random_range(0..num_qubits));
                    let second = QubitId((first.0 + rng.random_range(1..num_qubits)) % num_qubits);
                    let gate = rng.random_range(0..10);
                    apply_convention_gate(&mut w, gate, first, second);
                    apply_convention_gate(&mut y, gate, first, second);
                    if step % 5 == 0 {
                        let forced = rng.random();
                        assert_eq!(
                            w.mz_forced(first.0, forced).outcome,
                            y.mz_forced(first.0, forced).outcome
                        );
                    }
                    assert_convention_rows(&w, &y);
                }
                (w, y)
            })
            .collect()
    }

    fn convention_product(w: &SparseStabBitSet, rng: &mut PecosRng) -> (BitSet, PauliString) {
        let mut sources = BitSet::new();
        let generators = w.stabs().generators();
        for g in 0..generators.len() {
            if rng.random() {
                sources.insert(g);
            }
        }
        if sources.is_empty() {
            sources.insert(rng.random_range(0..generators.len()));
        }
        let product = sources
            .iter()
            .fold(PauliString::identity(), |p, g| p * &generators[g]);
        (sources, product)
    }

    #[test]
    fn test_convention_generators_random_cliffords() {
        for (w, y) in convention_circuits() {
            assert_eq!(w.stabs().generators(), y.stabs().generators());
        }
    }

    #[test]
    fn test_convention_product_sign_random_cliffords() {
        let mut rng = PecosRng::seed_from_u64(0xc0de_2026);
        for (w, y) in convention_circuits() {
            for _ in 0..8 {
                let (sources, product) = convention_product(&w, &mut rng);
                let mut w_phase = w.stabs().compute_product_sign(&sources);
                for _ in 0..product.y_qubits().len() % 4 {
                    w_phase = w_phase.multiply(&QuarterPhase::MinusI);
                }
                assert_eq!(w_phase, product.phase());
                assert_eq!(y.stabs().compute_product_sign(&sources), product.phase());
            }
        }
    }

    #[test]
    fn test_convention_find_sign_random_cliffords() {
        let mut rng = PecosRng::seed_from_u64(0xc0de_2026);
        for (w, y) in convention_circuits() {
            for _ in 0..8 {
                let (_, product) = convention_product(&w, &mut rng);
                let expected = Some(product.phase());
                assert_eq!(
                    w.stabs().find_pauli_string_sign(w.destabs(), &product),
                    expected
                );
                assert_eq!(
                    y.stabs().find_pauli_string_sign(y.destabs(), &product),
                    expected
                );
                // Explicit Y count may differ from the overlap: preserve the W API's rule.
                for num_ys in 0..4 {
                    assert_eq!(
                        w.stabs().find_pauli_sign(
                            w.destabs(),
                            product.x_positions(),
                            product.z_positions(),
                            num_ys
                        ),
                        y.stabs().find_pauli_sign(
                            y.destabs(),
                            product.x_positions(),
                            product.z_positions(),
                            num_ys
                        ),
                    );
                }
            }
        }
    }

    fn assert_convention_refactor(
        original_w: &SparseStabBitSet,
        original_y: &SparseStabY,
        sources: &BitSet,
        product: &PauliString,
        protected: Option<&BitSet>,
        chosen: usize,
    ) {
        let mut w = original_w.clone();
        let mut y = original_y.clone();
        let mut replaced = Vec::new();
        for (stabs, destabs) in [w.stabs_and_destabs_mut(), y.stabs_and_destabs_mut()] {
            let old_destab_minus = destabs.signs_minus.clone();
            let old_destab_i = destabs.signs_i.clone();
            let g = stabs
                .refactor(
                    destabs,
                    product.x_positions(),
                    product.z_positions(),
                    None,
                    protected,
                )
                .unwrap();
            replaced.push(g);
            assert_eq!(stabs.generator(g), *product);
            assert_eq!(destabs.signs_minus, old_destab_minus);
            assert_eq!(destabs.signs_i, old_destab_i);
            for generator in original_w.stabs().generators() {
                assert_eq!(
                    stabs.find_pauli_string_sign(destabs, &generator),
                    Some(generator.phase())
                );
            }
            if protected.is_some() {
                assert_eq!(g, chosen);
                for other in sources.iter().filter(|&other| other != chosen) {
                    assert_eq!(stabs.generator(other), original_w.stabs().generator(other));
                }
            }
        }
        assert_eq!(replaced[0], replaced[1]);
        assert_convention_rows(&w, &y);
        assert_eq!(w.stabs().generators(), y.stabs().generators());
        let qubits: Vec<_> = (0..w.num_qubits()).map(QubitId).collect();
        for forced in [false, true] {
            let mut measured_w = w.clone();
            let mut measured_y = y.clone();
            measured_w.h(&qubits).sz(&qubits);
            measured_y.h(&qubits).sz(&qubits);
            for q in &qubits {
                assert_eq!(
                    measured_w.mz_forced(q.0, forced).outcome,
                    measured_y.mz_forced(q.0, forced).outcome
                );
            }
        }
    }

    #[test]
    fn test_convention_refactor_random_cliffords() {
        let mut rng = PecosRng::seed_from_u64(0xc0de_2026);
        for (mut w, mut y) in convention_circuits() {
            for _ in 0..8 {
                let (sources, product) = convention_product(&w, &mut rng);
                let chosen = sources
                    .iter()
                    .nth(rng.random_range(0..sources.len()))
                    .unwrap();
                let mut protected = sources.clone();
                protected.remove(chosen);
                for protection in [None, Some(&protected)] {
                    assert_convention_refactor(&w, &y, &sources, &product, protection, chosen);
                }
                for (stabs, destabs) in [w.stabs_and_destabs_mut(), y.stabs_and_destabs_mut()] {
                    let before = stabs.generators();
                    assert_eq!(
                        stabs.refactor(
                            destabs,
                            product.x_positions(),
                            product.z_positions(),
                            None,
                            Some(&sources)
                        ),
                        None
                    );
                    assert_eq!(stabs.generators(), before);
                }
            }
        }
    }

    #[test]
    fn test_convention_stored_i_product() {
        let mut y = GensBitSet::with_convention(1, PauliConvention::Y);
        y.row_x = vec![BitSet::from_iter([0]), BitSet::new()];
        y.row_z = vec![BitSet::new(), BitSet::from_iter([0])];
        y.col_x[0].insert(0);
        y.col_z[0].insert(1);
        y.signs_i.insert(0);
        let mut w = GensBitSet::new(1);
        w.row_x.clone_from(&y.row_x);
        w.row_z.clone_from(&y.row_z);
        w.col_x.clone_from(&y.col_x);
        w.col_z.clone_from(&y.col_z);
        w.signs_i.clone_from(&y.signs_i);
        let sources = BitSet::from_iter([0, 1]);
        assert_eq!(y.compute_product_sign(&sources), QuarterPhase::PlusOne);
        assert_eq!(w.compute_product_sign(&sources), QuarterPhase::PlusI);
        assert_eq!(w.convention(), PauliConvention::W);
        assert_eq!(y.clone().convention(), PauliConvention::Y);

        // Raw phases remain raw, including imaginary coefficients on explicit Y rows.
        y.row_x[1].insert(0);
        y.col_x[0].insert(1);
        for (minus, imaginary, phase) in [
            (false, false, QuarterPhase::PlusOne),
            (true, false, QuarterPhase::MinusOne),
            (false, true, QuarterPhase::PlusI),
            (true, true, QuarterPhase::MinusI),
        ] {
            y.signs_minus.clear();
            y.signs_i.clear();
            if minus {
                y.signs_minus.insert(1);
            }
            if imaginary {
                y.signs_i.insert(1);
            }
            assert_eq!(y.generator_phase(1), phase);
            assert_eq!(y.generator(1).phase(), phase);
        }
    }

    #[test]
    fn test_convention_y_eigenstates() {
        for negative in [false, true] {
            let mut y = SparseStabY::new(1);
            y.h(&[QubitId(0)]).sz(&[QubitId(0)]);
            if negative {
                y.z(&[QubitId(0)]);
            }
            let phase = if negative {
                QuarterPhase::MinusOne
            } else {
                QuarterPhase::PlusOne
            };
            let expected =
                PauliString::with_phase_and_paulis(phase, vec![(Pauli::Y, QubitId(0))]).unwrap();
            assert_eq!(y.stabs().generator(0), expected);
            assert_eq!(
                y.stabs().find_pauli_string_sign(y.destabs(), &expected),
                Some(phase)
            );
        }
    }

    #[test]
    fn test_refactor_graph_state_preserves_sign_and_measurements() {
        let mut original = SparseStab::new(2);
        original.h(&[QubitId(0), QubitId(1)]);
        original.cz(&[(QubitId(0), QubitId(1))]);
        let sign = original
            .stabs()
            .find_pauli_sign(original.destabs(), [0, 1], [0, 1], 2)
            .unwrap();
        assert_eq!(sign, QuarterPhase::PlusOne);

        for forced in [false, true] {
            let mut expected = original.clone();
            let mut actual = original.clone();
            let (stabs, destabs) = actual.stabs_and_destabs_mut();
            let new_stab = stabs.refactor(destabs, [0, 1], [0, 1], None, None).unwrap();
            let generator = stabs.generator(new_stab);
            assert_eq!(generator.x_positions(), vec![0, 1]);
            assert_eq!(generator.z_positions(), vec![0, 1]);

            for state in [&mut expected, &mut actual] {
                state.szdg(&[QubitId(0), QubitId(1)]);
                state.h(&[QubitId(0), QubitId(1)]);
            }
            let expected_outcomes = [
                expected.mz_forced(0, forced).outcome,
                expected.mz_forced(1, forced).outcome,
            ];
            let actual_outcomes = [
                actual.mz_forced(0, forced).outcome,
                actual.mz_forced(1, forced).outcome,
            ];
            assert_eq!(expected_outcomes, [forced, forced]);
            assert_eq!(
                (generator.phase(), actual_outcomes),
                (sign, expected_outcomes)
            );
        }
    }

    fn random_clifford_state(num_qubits: usize, rng: &mut PecosRng) -> SparseStab {
        let mut state = SparseStab::new(num_qubits);
        for _ in 0..80 {
            let first = QubitId(rng.random_range(0..num_qubits));
            let second = QubitId((first.0 + rng.random_range(1..num_qubits)) % num_qubits);
            match rng.random_range(0..5) {
                0 => {
                    state.h(&[first]);
                }
                1 => {
                    state.sz(&[first]);
                }
                2 => {
                    state.cx(&[(first, second)]);
                }
                3 => {
                    state.cz(&[(first, second)]);
                }
                _ => {
                    state.mz_forced(first.0, rng.random());
                }
            }
        }
        state
    }

    #[test]
    fn test_refactor_random_cliffords_preserves_signed_group() {
        let mut rng = PecosRng::seed_from_u64(0x5eed_2026);
        for trial in 0..350 {
            let num_qubits = 2 + trial % 7;
            let original = random_clifford_state(num_qubits, &mut rng);
            let mut sources = crate::SparseStabSet::new();
            for g in 0..num_qubits {
                if rng.random() {
                    sources.insert(g);
                }
            }
            if sources.is_empty() {
                sources.insert(rng.random_range(0..num_qubits));
            }
            let mut xs = crate::SparseStabSet::new();
            let mut zs = crate::SparseStabSet::new();
            for g in &sources {
                for q in &original.stabs().row_x[g] {
                    xs.toggle(q);
                }
                for q in &original.stabs().row_z[g] {
                    zs.toggle(q);
                }
            }
            let phase = original.stabs().compute_product_sign(&sources);
            let chosen = sources
                .iter()
                .nth(rng.random_range(0..sources.len()))
                .unwrap();
            let mut protected = crate::SparseStabSet::new();
            for g in 0..num_qubits {
                if g != chosen {
                    protected.insert(g);
                }
            }

            for protection in [None, Some(&protected)] {
                let mut state = original.clone();
                let (stabs, destabs) = state.stabs_and_destabs_mut();
                let new_stab = stabs.refactor(destabs, &xs, &zs, None, protection).unwrap();
                for g in 0..num_qubits {
                    let generator = original.stabs().generator(g);
                    assert_eq!(
                        stabs.find_pauli_string_sign(destabs, &generator),
                        Some(generator.phase()),
                        "trial {trial}, generator {g}, protected {protection:?}"
                    );
                    if protection.is_some_and(|p| p.contains(g)) {
                        assert_eq!(stabs.generator(g), generator);
                    }
                }
                assert!(stabs.row_x[new_stab].iter().eq(xs.iter()));
                assert!(stabs.row_z[new_stab].iter().eq(zs.iter()));
                assert_eq!(stabs.generator_phase(new_stab), phase, "trial {trial}");
                if protection.is_some() {
                    assert_eq!(new_stab, chosen);
                }
            }
        }
    }

    /// Helper to create stabilizer/destabilizer generators for a simple 2-qubit state |00>.
    /// Stabilizers: Z0, Z1
    /// Destabilizers: X0, X1
    fn setup_two_qubit_z_state() -> (Gens, Gens) {
        let mut stabs = Gens::new(2);
        let mut destabs = Gens::new(2);

        // Stabilizer 0: Z on qubit 0
        stabs.row_z[0].insert(0);
        stabs.col_z[0].insert(0);

        // Stabilizer 1: Z on qubit 1
        stabs.row_z[1].insert(1);
        stabs.col_z[1].insert(1);

        // Destabilizer 0: X on qubit 0
        destabs.row_x[0].insert(0);
        destabs.col_x[0].insert(0);

        // Destabilizer 1: X on qubit 1
        destabs.row_x[1].insert(1);
        destabs.col_x[1].insert(1);

        (stabs, destabs)
    }

    #[test]
    fn test_classify_pauli_stabilizer() {
        let (stabs, destabs) = setup_two_qubit_z_state();

        // Z0 is a stabilizer
        assert_eq!(
            stabs.classify_pauli(&destabs, std::iter::empty(), [0].into_iter()),
            PauliClassification::Stabilizer
        );

        // Z1 is a stabilizer
        assert_eq!(
            stabs.classify_pauli(&destabs, std::iter::empty(), [1].into_iter()),
            PauliClassification::Stabilizer
        );

        // Z0*Z1 is a stabilizer (product of generators)
        assert_eq!(
            stabs.classify_pauli(&destabs, std::iter::empty(), [0, 1].into_iter()),
            PauliClassification::Stabilizer
        );
    }

    #[test]
    fn test_classify_pauli_error() {
        let (stabs, destabs) = setup_two_qubit_z_state();

        // X0 anticommutes with Z0, so it's an error
        assert_eq!(
            stabs.classify_pauli(&destabs, [0].into_iter(), std::iter::empty()),
            PauliClassification::Error
        );

        // Y0 = iX0Z0 anticommutes with Z0
        assert_eq!(
            stabs.classify_pauli(&destabs, [0].into_iter(), [0].into_iter()),
            PauliClassification::Error
        );
    }

    #[test]
    fn test_find_stabilizers_for_pauli() {
        let (stabs, destabs) = setup_two_qubit_z_state();

        // Z0 should be built from stabilizer 0
        let result = stabs.find_stabilizers_for_pauli(&destabs, std::iter::empty(), [0]);
        assert!(result.is_some());
        let stab_set = result.unwrap();
        assert!(stab_set.contains(0));
        assert!(!stab_set.contains(1));

        // Z0*Z1 should be built from stabilizers 0 and 1
        let result = stabs.find_stabilizers_for_pauli(&destabs, std::iter::empty(), [0, 1]);
        assert!(result.is_some());
        let stab_set = result.unwrap();
        assert!(stab_set.contains(0));
        assert!(stab_set.contains(1));
    }

    #[test]
    fn test_compute_product_sign_no_signs() {
        let (stabs, _destabs) = setup_two_qubit_z_state();

        // Product of generators with no signs should give +1
        let mut indices = crate::SparseStabSet::new();
        indices.insert(0);
        assert_eq!(stabs.compute_product_sign(&indices), QuarterPhase::PlusOne);

        indices.insert(1);
        assert_eq!(stabs.compute_product_sign(&indices), QuarterPhase::PlusOne);
    }

    #[test]
    fn test_compute_product_sign_with_minus() {
        let (mut stabs, _destabs) = setup_two_qubit_z_state();

        // Add a minus sign to stabilizer 0
        stabs.signs_minus.insert(0);

        let mut indices = crate::SparseStabSet::new();
        indices.insert(0);
        assert_eq!(stabs.compute_product_sign(&indices), QuarterPhase::MinusOne);

        // Product of -Z0 * Z1 = -Z0Z1
        indices.insert(1);
        assert_eq!(stabs.compute_product_sign(&indices), QuarterPhase::MinusOne);
    }

    #[test]
    fn test_find_anticommuting_generators() {
        let (stabs, _destabs) = setup_two_qubit_z_state();

        // X0 anticommutes with Z0 (stabilizer 0)
        let anticom = stabs.find_anticommuting_generators([0], std::iter::empty());
        assert!(anticom.contains(0));
        assert!(!anticom.contains(1));

        // X1 anticommutes with Z1 (stabilizer 1)
        let anticom = stabs.find_anticommuting_generators([1], std::iter::empty());
        assert!(!anticom.contains(0));
        assert!(anticom.contains(1));

        // X0X1 anticommutes with both
        let anticom = stabs.find_anticommuting_generators([0, 1], std::iter::empty());
        // Actually, X0X1 anticommutes with Z0 and Z1 individually, but the XOR gives empty
        // because each anticommutes once (odd), but together it's even per stabilizer
        // No wait - X0X1 anticommutes with Z0 (due to X0) and with Z1 (due to X1)
        // So both should be in the set
        assert!(anticom.contains(0));
        assert!(anticom.contains(1));
    }

    #[test]
    fn test_generator_weight() {
        let (stabs, _destabs) = setup_two_qubit_z_state();

        // Each stabilizer is just Z on one qubit, so weight = 1
        assert_eq!(stabs.generator_weight(0), 1);
        assert_eq!(stabs.generator_weight(1), 1);
    }

    #[test]
    fn test_min_weight_generator() {
        // Create a state with generators of different weights
        let mut stabs = Gens::new(3);

        // Generator 0: Z0 (weight 1)
        stabs.row_z[0].insert(0);
        stabs.col_z[0].insert(0);

        // Generator 1: Z1Z2 (weight 2)
        stabs.row_z[1].insert(1);
        stabs.row_z[1].insert(2);
        stabs.col_z[1].insert(1);
        stabs.col_z[2].insert(1);

        // Generator 2: X0Z0 = Y0 (weight 2, but counted as 2 since X and Z both present)
        stabs.row_x[2].insert(0);
        stabs.row_z[2].insert(0);
        stabs.col_x[0].insert(2);
        stabs.col_z[0].insert(2);

        // Build candidate set with all three
        let mut candidates = crate::SparseStabSet::new();
        candidates.insert(0);
        candidates.insert(1);
        candidates.insert(2);

        // Should choose generator 0 (weight 1)
        assert_eq!(stabs.min_weight_generator(&candidates), 0);

        // Remove generator 0; ties choose the lowest generator id.
        candidates.remove(0);
        assert_eq!(stabs.min_weight_generator(&candidates), 1);
    }

    #[test]
    fn test_min_weight_generator_unsorted_tie() {
        let mut stabs = GensGeneric::<VecSet<usize>>::new(3);
        stabs.row_z[1].insert(1);
        stabs.col_z[1].insert(1);
        stabs.row_z[2].insert(2);
        stabs.col_z[2].insert(2);
        let mut candidates = VecSet::new();
        candidates.insert(2);
        candidates.insert(1);
        assert_eq!(candidates.iter().copied().collect::<Vec<_>>(), [2, 1]);
        assert_eq!(stabs.min_weight_generator(&candidates), 1);
    }
}
