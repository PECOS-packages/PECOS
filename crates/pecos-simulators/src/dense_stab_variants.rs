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

// Inline(always) is intentional for these performance-critical bit manipulation functions.
#![allow(clippy::inline_always)]

//! Single-representation variants of the dense stabilizer simulator.
//!
//! These variants maintain only one representation (row-wise or column-wise) instead
//! of the dual representation used by [`super::DenseStab`]. This trades off:
//!
//! - **Memory**: Half the storage (n*words instead of 2*n*words per matrix)
//! - **Sync overhead**: No need to keep two representations consistent
//! - **Operation efficiency**: Some operations are faster, others slower
//!
//! # When to Use
//!
//! - [`DenseStabColOnly`]: Best when gate operations dominate (many gates, few measurements)
//! - [`DenseStabRowOnly`]: Best when row-based operations dominate (weight calculations,
//!   generator XORs)
//!
//! For balanced workloads like surface code syndrome extraction, the dual representation
//! in [`super::DenseStab`] is usually fastest.

use crate::{CliffordGateable, MeasurementResult, QuantumSimulator, StabilizerTableauSimulator};
use core::fmt::Debug;
use pecos_core::{QubitId, RngManageable};
use pecos_random::{PecosRng, Rng, RngExt, SeedableRng};
use smallvec::SmallVec;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_xor_si256};

// ========== Helper functions ==========

#[inline(always)]
fn set_bit_col(data: &mut [u64], words_per_col: usize, qubit: usize, row: usize) {
    let word_idx = qubit * words_per_col + row / 64;
    data[word_idx] |= 1u64 << (row % 64);
}

#[inline(always)]
fn set_bit_row(data: &mut [u64], words_per_row: usize, row: usize, qubit: usize) {
    let word_idx = row * words_per_row + qubit / 64;
    data[word_idx] |= 1u64 << (qubit % 64);
}

#[inline(always)]
fn toggle_sign(signs: &mut [u64], row: usize) {
    signs[row / 64] ^= 1u64 << (row % 64);
}

#[inline(always)]
fn get_sign(signs: &[u64], row: usize) -> bool {
    (signs[row / 64] >> (row % 64)) & 1 != 0
}

#[inline(always)]
fn set_sign(signs: &mut [u64], row: usize) {
    signs[row / 64] |= 1u64 << (row % 64);
}

#[inline(always)]
fn clear_sign(signs: &mut [u64], row: usize) {
    signs[row / 64] &= !(1u64 << (row % 64));
}

#[inline(always)]
fn xor_cols(data: &mut [u64], words_per_col: usize, col_a: usize, col_b: usize) {
    let base_a = col_a * words_per_col;
    let base_b = col_b * words_per_col;
    unsafe {
        match words_per_col {
            1 => {
                let a = *data.get_unchecked(base_a);
                *data.get_unchecked_mut(base_b) ^= a;
            }
            2 => {
                let a0 = *data.get_unchecked(base_a);
                let a1 = *data.get_unchecked(base_a + 1);
                *data.get_unchecked_mut(base_b) ^= a0;
                *data.get_unchecked_mut(base_b + 1) ^= a1;
            }
            _ => {
                // Use SIMD for larger sizes
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                if words_per_col >= 8 {
                    let chunks = words_per_col / 4;
                    let ptr_a = data.as_ptr().add(base_a) as *const __m256i;
                    let ptr_b = data.as_mut_ptr().add(base_b) as *mut __m256i;

                    for i in 0..chunks {
                        let a = _mm256_loadu_si256(ptr_a.add(i));
                        let b = _mm256_loadu_si256(ptr_b.add(i));
                        let result = _mm256_xor_si256(a, b);
                        _mm256_storeu_si256(ptr_b.add(i), result);
                    }

                    // Handle remainder
                    for w in (chunks * 4)..words_per_col {
                        let a = *data.get_unchecked(base_a + w);
                        *data.get_unchecked_mut(base_b + w) ^= a;
                    }
                    return;
                }

                // Scalar fallback
                for w in 0..words_per_col {
                    let a = *data.get_unchecked(base_a + w);
                    *data.get_unchecked_mut(base_b + w) ^= a;
                }
            }
        }
    }
}

#[inline(always)]
fn xor_rows(data: &mut [u64], words_per_row: usize, row_a: usize, row_b: usize) {
    let base_a = row_a * words_per_row;
    let base_b = row_b * words_per_row;
    unsafe {
        match words_per_row {
            1 => {
                let a = *data.get_unchecked(base_a);
                *data.get_unchecked_mut(base_b) ^= a;
            }
            2 => {
                let a0 = *data.get_unchecked(base_a);
                let a1 = *data.get_unchecked(base_a + 1);
                *data.get_unchecked_mut(base_b) ^= a0;
                *data.get_unchecked_mut(base_b + 1) ^= a1;
            }
            _ => {
                // Use SIMD for larger sizes
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                if words_per_row >= 8 {
                    let chunks = words_per_row / 4;
                    let ptr_a = data.as_ptr().add(base_a) as *const __m256i;
                    let ptr_b = data.as_mut_ptr().add(base_b) as *mut __m256i;

                    for i in 0..chunks {
                        let a = _mm256_loadu_si256(ptr_a.add(i));
                        let b = _mm256_loadu_si256(ptr_b.add(i));
                        let result = _mm256_xor_si256(a, b);
                        _mm256_storeu_si256(ptr_b.add(i), result);
                    }

                    // Handle remainder
                    for w in (chunks * 4)..words_per_row {
                        let a = *data.get_unchecked(base_a + w);
                        *data.get_unchecked_mut(base_b + w) ^= a;
                    }
                    return;
                }

                // Scalar fallback
                for w in 0..words_per_row {
                    let a = *data.get_unchecked(base_a + w);
                    *data.get_unchecked_mut(base_b + w) ^= a;
                }
            }
        }
    }
}

#[inline(always)]
fn row_weight(data: &[u64], words_per_row: usize, row: usize) -> usize {
    let base = row * words_per_row;
    let mut count = 0;
    for w in 0..words_per_row {
        count += data[base + w].count_ones() as usize;
    }
    count
}

// ========== Column-Only Variant ==========

/// Dense stabilizer simulator using only column-wise representation.
///
/// This variant stores generators in column-major order only, which makes
/// gate operations efficient but requires more work for row-based operations
/// like generator XORs during measurement.
///
/// # Memory Layout
///
/// For n qubits with w = ceil(n/64) words per column:
/// - `col_x[q*w..(q+1)*w]`: bit vector of which generators have X on qubit q
/// - `col_z[q*w..(q+1)*w]`: bit vector of which generators have Z on qubit q
#[derive(Debug, Clone)]
pub struct DenseStabColOnly<R: SeedableRng + Rng + Debug = PecosRng> {
    num_qubits: usize,
    words_per_col: usize,

    // Column-wise storage only
    stab_col_x: Vec<u64>,
    stab_col_z: Vec<u64>,
    destab_col_x: Vec<u64>,
    destab_col_z: Vec<u64>,

    // Signs
    stab_signs_minus: Vec<u64>,
    stab_signs_i: Vec<u64>,
    destab_signs_minus: Vec<u64>,
    destab_signs_i: Vec<u64>,

    // Scratch buffers for measurement (sparse iteration)
    scratch_pivot_x: Vec<usize>,
    scratch_pivot_z: Vec<usize>,

    rng: R,
}

impl DenseStabColOnly<PecosRng> {
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let rng = rand::make_rng();
        Self::with_rng(num_qubits, rng)
    }

    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let rng = PecosRng::seed_from_u64(seed);
        Self::with_rng(num_qubits, rng)
    }
}

impl<R: SeedableRng + Rng + Debug> DenseStabColOnly<R> {
    #[inline]
    #[must_use]
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        let words_per_col = num_qubits.div_ceil(64);
        let col_size = num_qubits * words_per_col;
        let sign_size = words_per_col;

        let mut sim = Self {
            num_qubits,
            words_per_col,
            stab_col_x: vec![0; col_size],
            stab_col_z: vec![0; col_size],
            destab_col_x: vec![0; col_size],
            destab_col_z: vec![0; col_size],
            stab_signs_minus: vec![0; sign_size],
            stab_signs_i: vec![0; sign_size],
            destab_signs_minus: vec![0; sign_size],
            destab_signs_i: vec![0; sign_size],
            scratch_pivot_x: Vec::with_capacity(num_qubits),
            scratch_pivot_z: Vec::with_capacity(num_qubits),
            rng,
        };
        sim.init_state();
        sim
    }

    fn init_state(&mut self) {
        self.stab_col_x.fill(0);
        self.stab_col_z.fill(0);
        self.destab_col_x.fill(0);
        self.destab_col_z.fill(0);
        self.stab_signs_minus.fill(0);
        self.stab_signs_i.fill(0);
        self.destab_signs_minus.fill(0);
        self.destab_signs_i.fill(0);

        // Initialize: stab[i] = Z_i, destab[i] = X_i
        for i in 0..self.num_qubits {
            set_bit_col(&mut self.stab_col_z, self.words_per_col, i, i);
            set_bit_col(&mut self.destab_col_x, self.words_per_col, i, i);
        }
    }

    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Extract qubits where pivot has X or Z into scratch buffers.
    fn extract_pivot_positions(&mut self, pivot_id: usize) {
        let words_per_col = self.words_per_col;
        let pivot_word = pivot_id / 64;
        let pivot_mask = 1u64 << (pivot_id % 64);

        self.scratch_pivot_x.clear();
        self.scratch_pivot_z.clear();

        for q in 0..self.num_qubits {
            let base = q * words_per_col + pivot_word;
            if self.stab_col_x[base] & pivot_mask != 0 {
                self.scratch_pivot_x.push(q);
            }
            if self.stab_col_z[base] & pivot_mask != 0 {
                self.scratch_pivot_z.push(q);
            }
        }
    }

    fn apply_cx(&mut self, control: usize, target: usize) {
        Self::apply_cx_to_gens(
            &mut self.stab_col_x,
            &mut self.stab_col_z,
            self.words_per_col,
            control,
            target,
        );
        Self::apply_cx_to_gens(
            &mut self.destab_col_x,
            &mut self.destab_col_z,
            self.words_per_col,
            control,
            target,
        );
    }

    #[inline(always)]
    fn apply_cx_to_gens(
        col_x: &mut [u64],
        col_z: &mut [u64],
        words_per_col: usize,
        control: usize,
        target: usize,
    ) {
        // CX: X_c -> X_c X_t, Z_t -> Z_c Z_t
        xor_cols(col_x, words_per_col, control, target);
        xor_cols(col_z, words_per_col, target, control);
    }

    fn apply_h(&mut self, qubit: usize) {
        Self::apply_h_to_gens(
            &mut self.stab_col_x,
            &mut self.stab_col_z,
            &mut self.stab_signs_minus,
            self.words_per_col,
            qubit,
        );
        Self::apply_h_to_gens(
            &mut self.destab_col_x,
            &mut self.destab_col_z,
            &mut self.destab_signs_minus,
            self.words_per_col,
            qubit,
        );
    }

    #[inline(always)]
    fn apply_h_to_gens(
        col_x: &mut [u64],
        col_z: &mut [u64],
        signs_minus: &mut [u64],
        words_per_col: usize,
        qubit: usize,
    ) {
        // H: X -> Z, Z -> X, Y -> -Y
        let col_base = qubit * words_per_col;
        for w in 0..words_per_col {
            let cx = col_x[col_base + w];
            let cz = col_z[col_base + w];
            signs_minus[w] ^= cx & cz;
            col_x[col_base + w] = cz;
            col_z[col_base + w] = cx;
        }
    }

    fn apply_sz(&mut self, qubit: usize) {
        // S/SZ gate: X -> iXZ (Y), Y -> -X, Z -> Z
        // The i-phase tracking handles all phase changes:
        // - X (no i) → iXZ: add i phase, add Z
        // - iXZ (Y, has i) → -X: i*i=-1 flips minus, remove i, remove Z
        let col_base = qubit * self.words_per_col;
        for w in 0..self.words_per_col {
            let x_gens = self.stab_col_x[col_base + w];
            // i*i = -1: toggle minus for generators with both i and X
            self.stab_signs_minus[w] ^= x_gens & self.stab_signs_i[w];
            // Toggle i for all X generators
            self.stab_signs_i[w] ^= x_gens;
            // Toggle Z for all X generators
            self.stab_col_z[col_base + w] ^= x_gens;
        }

        // For destabilizers, same logic
        for w in 0..self.words_per_col {
            let x_gens = self.destab_col_x[col_base + w];
            self.destab_signs_minus[w] ^= x_gens & self.destab_signs_i[w];
            self.destab_signs_i[w] ^= x_gens;
            self.destab_col_z[col_base + w] ^= x_gens;
        }
    }

    fn deterministic_meas(&self, qubit: usize) -> Option<MeasurementResult> {
        let words_per_col = self.words_per_col;
        let col_base = qubit * words_per_col;
        let n = self.num_qubits;

        // Check if any stabilizer has X on this qubit
        for w in 0..words_per_col {
            if self.stab_col_x[col_base + w] != 0 {
                return None;
            }
        }

        // Deterministic measurement: compute product of all destabilizers with X on this qubit
        // The outcome is determined by the phase of this product.
        //
        // We need to count:
        // 1. Individual minus signs from each destabilizer's sign
        // 2. Individual i phases from each destabilizer's i-sign
        // 3. Phase contributions from multiplying Paulis (Z*X = -i*Y, X*Z = i*Y)

        // Get mask of all destabilizers with X on the measured qubit
        let mut destab_mask = vec![0u64; words_per_col];
        let mut has_destab = false;
        for (w, dm) in destab_mask.iter_mut().enumerate() {
            *dm = self.destab_col_x[col_base + w];
            if *dm != 0 {
                has_destab = true;
            }
        }

        if !has_destab {
            // No destabilizer has X on this qubit - return default |0>
            return Some(MeasurementResult {
                outcome: false,
                is_deterministic: true,
            });
        }

        // Count minus signs from destabilizers
        let mut num_minuses: usize = 0;
        let mut num_is: usize = 0;
        for (w, &dm) in destab_mask.iter().enumerate() {
            num_minuses += (dm & self.stab_signs_minus[w]).count_ones() as usize;
            num_is += (dm & self.stab_signs_i[w]).count_ones() as usize;
        }

        // Collect destabilizer IDs with X on this qubit
        let mut destab_ids = Vec::new();
        for (w, &dm) in destab_mask.iter().enumerate() {
            let mut mask = dm;
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                destab_ids.push(w * 64 + bit);
                mask &= mask - 1;
            }
        }

        // Compute phase contributions from Pauli multiplication
        // For each destabilizer with X on the measured qubit, we use the STABILIZER
        // at that index to compute the product phase.
        //
        // We maintain a cumulative X row (XOR of stabilizer X rows seen so far)
        // For each new stabilizer, count overlap of its Z with cumulative X

        // Cumulative X row: cumulative_x[q] = 1 if odd number of cumulative stabilizers have X on q
        let mut cumulative_x = vec![false; n];

        for &stab_id in &destab_ids {
            let stab_word = stab_id / 64;
            let stab_bit = 1u64 << (stab_id % 64);

            // Count overlap: positions where this stabilizer has Z and cumulative has X
            // This gives phase contribution from X*Z = iY (previous X, current Z)
            for (q, cx) in cumulative_x.iter().enumerate() {
                let q_base = q * words_per_col;
                // Check if this stabilizer has Z on qubit q
                let has_z = self.stab_col_z[q_base + stab_word] & stab_bit != 0;
                if has_z && *cx {
                    num_minuses += 1; // XZ = iY contributes +i phase
                }
            }

            // XOR this stabilizer's X pattern into cumulative X
            for (q, cx) in cumulative_x.iter_mut().enumerate() {
                let q_base = q * words_per_col;
                let has_x = self.stab_col_x[q_base + stab_word] & stab_bit != 0;
                if has_x {
                    *cx = !*cx;
                }
            }
        }

        // Add i phase contribution
        if num_is & 3 != 0 {
            num_minuses += 1;
        }

        Some(MeasurementResult {
            outcome: num_minuses & 1 != 0,
            is_deterministic: true,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn nondeterministic_meas(&mut self, qubit: usize, outcome: bool) -> MeasurementResult {
        let words_per_col = self.words_per_col;
        let col_base = qubit * words_per_col;

        // Find minimum weight anti-commuting stabilizer
        // Weight is computed by counting X and Z positions
        let mut min_weight = usize::MAX;
        let mut pivot_id = 0;

        for w in 0..words_per_col {
            let mut mask = self.stab_col_x[col_base + w];
            while mask != 0 {
                let g = w * 64 + mask.trailing_zeros() as usize;
                let g_word = g / 64;
                let g_mask = 1u64 << (g % 64);

                // Count weight by scanning columns
                let mut weight = 0;
                for q in 0..self.num_qubits {
                    let base = q * words_per_col + g_word;
                    if self.stab_col_x[base] & g_mask != 0 {
                        weight += 1;
                    }
                    if self.stab_col_z[base] & g_mask != 0 {
                        weight += 1;
                    }
                }

                if weight < min_weight {
                    min_weight = weight;
                    pivot_id = g;
                    if weight == 1 {
                        break;
                    }
                }
                mask &= mask - 1;
            }
            if min_weight == 1 {
                break;
            }
        }

        // Extract pivot's X and Z positions for sparse iteration
        self.extract_pivot_positions(pivot_id);

        // Cache pivot sign and position
        let pivot_sign_minus = get_sign(&self.stab_signs_minus, pivot_id);
        let pivot_sign_i = get_sign(&self.stab_signs_i, pivot_id);
        let pivot_word = pivot_id / 64;
        let pivot_mask = 1u64 << (pivot_id % 64);

        // Handle pivot's i-phase contribution (bulk operation before per-generator loop)
        if pivot_sign_i {
            clear_sign(&mut self.stab_signs_i, pivot_id);
            for w in 0..words_per_col {
                let mut anticom = self.stab_col_x[col_base + w];
                if w == pivot_word {
                    anticom &= !pivot_mask;
                }
                // Toggle minus for anticom stabs that have i (i * i = -1)
                self.stab_signs_minus[w] ^= anticom & self.stab_signs_i[w];
                // Toggle i for all anticom stabs
                self.stab_signs_i[w] ^= anticom;
            }
        }

        // XOR pivot into other anti-commuting stabilizers
        for w in 0..words_per_col {
            let mut mask = self.stab_col_x[col_base + w];
            if w == pivot_word {
                mask &= !pivot_mask;
            }

            while mask != 0 {
                let g = w * 64 + mask.trailing_zeros() as usize;
                let g_word = g / 64;
                let g_mask = 1u64 << (g % 64);

                // Phase calculation: count Z_pivot & X_g overlaps (sparse iteration)
                let mut count = 0;
                for &q in &self.scratch_pivot_z {
                    let base = q * words_per_col + g_word;
                    if self.stab_col_x[base] & g_mask != 0 {
                        count += 1;
                    }
                }
                if count & 1 != 0 {
                    toggle_sign(&mut self.stab_signs_minus, g);
                }

                if pivot_sign_minus {
                    toggle_sign(&mut self.stab_signs_minus, g);
                }

                // XOR pivot into g (sparse iteration over pivot positions)
                for &q in &self.scratch_pivot_x {
                    let base = q * words_per_col + g_word;
                    self.stab_col_x[base] ^= g_mask;
                }
                for &q in &self.scratch_pivot_z {
                    let base = q * words_per_col + g_word;
                    self.stab_col_z[base] ^= g_mask;
                }

                mask &= mask - 1;
            }
        }

        // Step 2b (Aaronson-Gottesman): XOR pivot stab into anti-commuting destabilizers
        // Pre-compute anticom destab mask to avoid self-XOR when q == qubit
        let anticom_destab_mask: Vec<u64> = (0..words_per_col)
            .map(|w| {
                let mut m = self.destab_col_x[col_base + w];
                if w == pivot_word {
                    m &= !pivot_mask;
                }
                m
            })
            .collect();
        for &q in &self.scratch_pivot_x {
            let base = q * words_per_col;
            for (w, &mask) in anticom_destab_mask.iter().enumerate().take(words_per_col) {
                self.destab_col_x[base + w] ^= mask;
            }
        }
        for &q in &self.scratch_pivot_z {
            let base = q * words_per_col;
            for (w, &mask) in anticom_destab_mask.iter().enumerate().take(words_per_col) {
                self.destab_col_z[base + w] ^= mask;
            }
        }

        // Copy old stabilizer to destabilizer (sparse iteration)
        for &q in &self.scratch_pivot_x {
            let base = q * words_per_col + pivot_word;
            self.destab_col_x[base] |= pivot_mask;
        }
        for &q in &self.scratch_pivot_z {
            let base = q * words_per_col + pivot_word;
            self.destab_col_z[base] |= pivot_mask;
        }
        // Clear positions not in pivot (need to iterate all for this)
        for q in 0..self.num_qubits {
            let base = q * words_per_col + pivot_word;
            if self.stab_col_x[base] & pivot_mask == 0 {
                self.destab_col_x[base] &= !pivot_mask;
            }
            if self.stab_col_z[base] & pivot_mask == 0 {
                self.destab_col_z[base] &= !pivot_mask;
            }
        }

        // Copy stabilizer sign to destabilizer
        if get_sign(&self.stab_signs_minus, pivot_id) {
            set_sign(&mut self.destab_signs_minus, pivot_id);
        } else {
            clear_sign(&mut self.destab_signs_minus, pivot_id);
        }
        clear_sign(&mut self.destab_signs_i, pivot_id);

        // Replace pivot stabilizer with Z_qubit (sparse clear using scratch)
        for &q in &self.scratch_pivot_x {
            let base = q * words_per_col + pivot_word;
            self.stab_col_x[base] &= !pivot_mask;
        }
        for &q in &self.scratch_pivot_z {
            let base = q * words_per_col + pivot_word;
            self.stab_col_z[base] &= !pivot_mask;
        }
        set_bit_col(&mut self.stab_col_z, words_per_col, qubit, pivot_id);

        if outcome {
            set_sign(&mut self.stab_signs_minus, pivot_id);
        } else {
            clear_sign(&mut self.stab_signs_minus, pivot_id);
        }
        clear_sign(&mut self.stab_signs_i, pivot_id);

        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }

    fn mz_internal(&mut self, qubit: usize) -> MeasurementResult {
        if let Some(result) = self.deterministic_meas(qubit) {
            return result;
        }

        let outcome = self.rng.random_bool(0.5);
        self.nondeterministic_meas(qubit, outcome)
    }

    /// Measure qubit with forced outcome for non-deterministic cases.
    pub fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        if let Some(result) = self.deterministic_meas(qubit) {
            return result;
        }
        self.nondeterministic_meas(qubit, forced_outcome)
    }
}

impl<R: SeedableRng + Rng + Debug + Clone> QuantumSimulator for DenseStabColOnly<R> {
    fn reset(&mut self) -> &mut Self {
        self.init_state();
        self
    }
}

impl<R: SeedableRng + Rng + Debug + Clone> RngManageable for DenseStabColOnly<R> {
    type Rng = R;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.rng = rng;
    }

    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

impl<R: SeedableRng + Rng + Debug + Clone> CliffordGateable for DenseStabColOnly<R> {
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_h(q.index());
        }
        self
    }

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_sz(q.index());
        }
        self
    }

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(control, target) in pairs {
            self.apply_cx(control.index(), target.index());
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits.iter().map(|q| self.mz_internal(q.index())).collect()
    }
}

// ========== Row-Only Variant ==========

/// Dense stabilizer simulator using only row-wise representation.
///
/// This variant stores generators in row-major order only, which makes
/// generator XORs efficient but requires scanning all rows to find
/// generators with a specific Pauli on a qubit.
///
/// # Memory Layout
///
/// For n qubits with w = ceil(n/64) words per row:
/// - `row_x[g*w..(g+1)*w]`: bit vector of X components for generator g
/// - `row_z[g*w..(g+1)*w]`: bit vector of Z components for generator g
#[derive(Debug, Clone)]
pub struct DenseStabRowOnly<R: SeedableRng + Rng + Debug = PecosRng> {
    num_qubits: usize,
    words_per_row: usize,

    // Row-wise storage only
    stab_row_x: Vec<u64>,
    stab_row_z: Vec<u64>,
    destab_row_x: Vec<u64>,
    destab_row_z: Vec<u64>,

    // Signs
    stab_signs_minus: Vec<u64>,
    stab_signs_i: Vec<u64>,
    destab_signs_minus: Vec<u64>,
    destab_signs_i: Vec<u64>,

    // Scratch buffer for finding generators
    scratch_gens: Vec<usize>,

    rng: R,
}

impl DenseStabRowOnly<PecosRng> {
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let rng = rand::make_rng();
        Self::with_rng(num_qubits, rng)
    }

    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let rng = PecosRng::seed_from_u64(seed);
        Self::with_rng(num_qubits, rng)
    }
}

impl<R: SeedableRng + Rng + Debug> DenseStabRowOnly<R> {
    #[inline]
    #[must_use]
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        let words_per_row = num_qubits.div_ceil(64);
        let row_size = num_qubits * words_per_row;
        let sign_size = words_per_row;

        let mut sim = Self {
            num_qubits,
            words_per_row,
            stab_row_x: vec![0; row_size],
            stab_row_z: vec![0; row_size],
            destab_row_x: vec![0; row_size],
            destab_row_z: vec![0; row_size],
            stab_signs_minus: vec![0; sign_size],
            stab_signs_i: vec![0; sign_size],
            destab_signs_minus: vec![0; sign_size],
            destab_signs_i: vec![0; sign_size],
            scratch_gens: Vec::with_capacity(num_qubits),
            rng,
        };
        sim.init_state();
        sim
    }

    fn init_state(&mut self) {
        self.stab_row_x.fill(0);
        self.stab_row_z.fill(0);
        self.destab_row_x.fill(0);
        self.destab_row_z.fill(0);
        self.stab_signs_minus.fill(0);
        self.stab_signs_i.fill(0);
        self.destab_signs_minus.fill(0);
        self.destab_signs_i.fill(0);

        // Initialize: stab[i] = Z_i, destab[i] = X_i
        for i in 0..self.num_qubits {
            set_bit_row(&mut self.stab_row_z, self.words_per_row, i, i);
            set_bit_row(&mut self.destab_row_x, self.words_per_row, i, i);
        }
    }

    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn apply_cx(&mut self, control: usize, target: usize) {
        Self::apply_cx_to_gens(
            &mut self.stab_row_x,
            &mut self.stab_row_z,
            self.words_per_row,
            self.num_qubits,
            control,
            target,
        );
        Self::apply_cx_to_gens(
            &mut self.destab_row_x,
            &mut self.destab_row_z,
            self.words_per_row,
            self.num_qubits,
            control,
            target,
        );
    }

    #[inline(always)]
    fn apply_cx_to_gens(
        row_x: &mut [u64],
        row_z: &mut [u64],
        words_per_row: usize,
        num_rows: usize,
        control: usize,
        target: usize,
    ) {
        // CX: X_c -> X_c X_t, Z_t -> Z_c Z_t
        // Mask-based approach: compute masks once, use simple AND/XOR in loop
        let control_word = control / 64;
        let control_mask = 1u64 << (control % 64);
        let target_word = target / 64;
        let target_mask = 1u64 << (target % 64);

        if control_word == target_word {
            // Same word: can do both operations with single access
            for g in 0..num_rows {
                let idx = g * words_per_row + control_word;
                let x_word = row_x[idx];
                let z_word = row_z[idx];

                // X propagates: control -> target
                let x_toggle = if x_word & control_mask != 0 {
                    target_mask
                } else {
                    0
                };
                // Z propagates: target -> control
                let z_toggle = if z_word & target_mask != 0 {
                    control_mask
                } else {
                    0
                };

                row_x[idx] = x_word ^ x_toggle;
                row_z[idx] = z_word ^ z_toggle;
            }
        } else {
            // Different words
            for g in 0..num_rows {
                let base = g * words_per_row;
                let ctrl_idx = base + control_word;
                let tgt_idx = base + target_word;

                // X propagates: control -> target
                if row_x[ctrl_idx] & control_mask != 0 {
                    row_x[tgt_idx] ^= target_mask;
                }
                // Z propagates: target -> control
                if row_z[tgt_idx] & target_mask != 0 {
                    row_z[ctrl_idx] ^= control_mask;
                }
            }
        }
    }

    fn apply_h(&mut self, qubit: usize) {
        Self::apply_h_to_gens(
            &mut self.stab_row_x,
            &mut self.stab_row_z,
            &mut self.stab_signs_minus,
            self.words_per_row,
            self.num_qubits,
            qubit,
        );
        Self::apply_h_to_gens(
            &mut self.destab_row_x,
            &mut self.destab_row_z,
            &mut self.destab_signs_minus,
            self.words_per_row,
            self.num_qubits,
            qubit,
        );
    }

    #[inline(always)]
    fn apply_h_to_gens(
        row_x: &mut [u64],
        row_z: &mut [u64],
        signs_minus: &mut [u64],
        words_per_row: usize,
        num_rows: usize,
        qubit: usize,
    ) {
        // H: X -> Z, Z -> X, Y -> -Y
        // Mask-based implementation
        let qubit_word = qubit / 64;
        let qubit_mask = 1u64 << (qubit % 64);

        for g in 0..num_rows {
            let row_idx = g * words_per_row + qubit_word;
            let x_word = row_x[row_idx];
            let z_word = row_z[row_idx];

            let has_x = x_word & qubit_mask != 0;
            let has_z = z_word & qubit_mask != 0;

            // Toggle sign if both X and Z (Y)
            if has_x && has_z {
                signs_minus[g / 64] ^= 1u64 << (g % 64);
            }

            // Swap X and Z bits only if they differ
            if has_x != has_z {
                row_x[row_idx] = x_word ^ qubit_mask;
                row_z[row_idx] = z_word ^ qubit_mask;
            }
        }
    }

    fn apply_sz(&mut self, qubit: usize) {
        // S/SZ gate: X -> iXZ (Y), Y -> -X, Z -> Z
        // The i-phase tracking handles all phase changes
        let qubit_word = qubit / 64;
        let qubit_mask = 1u64 << (qubit % 64);

        for g in 0..self.num_qubits {
            let row_idx = g * self.words_per_row + qubit_word;

            // Only do work if generator has X on this qubit
            if self.stab_row_x[row_idx] & qubit_mask != 0 {
                let g_bit = 1u64 << (g % 64);
                let g_word = g / 64;
                // i*i = -1: toggle minus if has both i and X
                if self.stab_signs_i[g_word] & g_bit != 0 {
                    self.stab_signs_minus[g_word] ^= g_bit;
                }
                // Toggle i for X generators
                self.stab_signs_i[g_word] ^= g_bit;
                // Toggle Z
                self.stab_row_z[row_idx] ^= qubit_mask;
            }
        }

        // Destabilizers
        for g in 0..self.num_qubits {
            let row_idx = g * self.words_per_row + qubit_word;

            if self.destab_row_x[row_idx] & qubit_mask != 0 {
                let g_bit = 1u64 << (g % 64);
                let g_word = g / 64;
                if self.destab_signs_i[g_word] & g_bit != 0 {
                    self.destab_signs_minus[g_word] ^= g_bit;
                }
                self.destab_signs_i[g_word] ^= g_bit;
                self.destab_row_z[row_idx] ^= qubit_mask;
            }
        }
    }

    fn deterministic_meas(&self, qubit: usize) -> Option<MeasurementResult> {
        let words_per_row = self.words_per_row;
        let qubit_word = qubit / 64;
        let qubit_mask = 1u64 << (qubit % 64);

        // Check if any stabilizer has X on this qubit
        for g in 0..self.num_qubits {
            if self.stab_row_x[g * words_per_row + qubit_word] & qubit_mask != 0 {
                return None;
            }
        }

        // Deterministic measurement: compute product of stabilizers indexed by
        // destabilizers that have X on this qubit.
        //
        // Collect destabilizer indices with X on this qubit
        let mut destab_ids = Vec::new();
        for g in 0..self.num_qubits {
            if self.destab_row_x[g * words_per_row + qubit_word] & qubit_mask != 0 {
                destab_ids.push(g);
            }
        }

        if destab_ids.is_empty() {
            // No destabilizer has X on this qubit - return default |0>
            return Some(MeasurementResult {
                outcome: false,
                is_deterministic: true,
            });
        }

        // Count minus and i signs from the stabilizers at these indices
        let mut num_minuses: usize = 0;
        let mut num_is: usize = 0;
        for &g in &destab_ids {
            if get_sign(&self.stab_signs_minus, g) {
                num_minuses += 1;
            }
            if get_sign(&self.stab_signs_i, g) {
                num_is += 1;
            }
        }

        // Compute phase from multiplying stabilizers
        // Maintain cumulative X row
        let mut cumulative_x = vec![0u64; words_per_row];

        for &g in &destab_ids {
            let row_base = g * words_per_row;

            // Count overlap: positions where this stabilizer has Z and cumulative has X
            for (w, cx) in cumulative_x.iter().enumerate() {
                num_minuses += (self.stab_row_z[row_base + w] & *cx).count_ones() as usize;
            }

            // XOR this stabilizer's X row into cumulative
            for (w, cx) in cumulative_x.iter_mut().enumerate() {
                *cx ^= self.stab_row_x[row_base + w];
            }
        }

        // Add i phase contribution
        if num_is & 3 != 0 {
            num_minuses += 1;
        }

        Some(MeasurementResult {
            outcome: num_minuses & 1 != 0,
            is_deterministic: true,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn nondeterministic_meas(&mut self, qubit: usize, outcome: bool) -> MeasurementResult {
        let words_per_row = self.words_per_row;
        let qubit_word = qubit / 64;
        let qubit_mask = 1u64 << (qubit % 64);

        // Find anti-commuting stabilizers and minimum weight one
        self.scratch_gens.clear();
        let mut min_weight = usize::MAX;
        let mut pivot_id = 0;

        for g in 0..self.num_qubits {
            if self.stab_row_x[g * words_per_row + qubit_word] & qubit_mask != 0 {
                self.scratch_gens.push(g);
                let weight = row_weight(&self.stab_row_x, words_per_row, g)
                    + row_weight(&self.stab_row_z, words_per_row, g);
                if weight < min_weight {
                    min_weight = weight;
                    pivot_id = g;
                }
            }
        }

        // Cache pivot sign
        let pivot_sign_minus = get_sign(&self.stab_signs_minus, pivot_id);
        let pivot_sign_i = get_sign(&self.stab_signs_i, pivot_id);

        // Handle pivot's i-phase contribution
        if pivot_sign_i {
            clear_sign(&mut self.stab_signs_i, pivot_id);
            for &g in &self.scratch_gens {
                if g == pivot_id {
                    continue;
                }
                // Toggle minus for anticom stabs that have i (i * i = -1)
                if get_sign(&self.stab_signs_i, g) {
                    toggle_sign(&mut self.stab_signs_minus, g);
                }
                // Toggle i for all anticom stabs
                toggle_sign(&mut self.stab_signs_i, g);
            }
        }

        // XOR pivot into other anti-commuting stabilizers
        for &g in &self.scratch_gens {
            if g == pivot_id {
                continue;
            }

            // Phase calculation: count Z_pivot & X_g overlaps
            let base_p = pivot_id * words_per_row;
            let base_g = g * words_per_row;
            let mut count = 0;
            for w in 0..words_per_row {
                count += (self.stab_row_z[base_p + w] & self.stab_row_x[base_g + w]).count_ones();
            }
            if count & 1 != 0 {
                toggle_sign(&mut self.stab_signs_minus, g);
            }

            if pivot_sign_minus {
                toggle_sign(&mut self.stab_signs_minus, g);
            }

            // XOR rows
            xor_rows(&mut self.stab_row_x, words_per_row, pivot_id, g);
            xor_rows(&mut self.stab_row_z, words_per_row, pivot_id, g);
        }

        // Step 2b (Aaronson-Gottesman): XOR pivot stab into anti-commuting destabilizers
        for g in 0..self.num_qubits {
            if g == pivot_id {
                continue;
            }
            // Check if destab[g] anti-commutes with Z_qubit (has X on measured qubit)
            if self.destab_row_x[g * words_per_row + qubit_word] & qubit_mask != 0 {
                let base_p = pivot_id * words_per_row;
                let base_g = g * words_per_row;
                for w in 0..words_per_row {
                    self.destab_row_x[base_g + w] ^= self.stab_row_x[base_p + w];
                    self.destab_row_z[base_g + w] ^= self.stab_row_z[base_p + w];
                }
            }
        }

        // Copy old stabilizer to destabilizer before replacing
        let pivot_base = pivot_id * words_per_row;
        for w in 0..words_per_row {
            self.destab_row_x[pivot_base + w] = self.stab_row_x[pivot_base + w];
            self.destab_row_z[pivot_base + w] = self.stab_row_z[pivot_base + w];
        }
        // Copy stabilizer sign to destabilizer
        if get_sign(&self.stab_signs_minus, pivot_id) {
            set_sign(&mut self.destab_signs_minus, pivot_id);
        } else {
            clear_sign(&mut self.destab_signs_minus, pivot_id);
        }
        clear_sign(&mut self.destab_signs_i, pivot_id);

        // Replace pivot stabilizer with Z_qubit
        for w in 0..words_per_row {
            self.stab_row_x[pivot_base + w] = 0;
            self.stab_row_z[pivot_base + w] = 0;
        }
        set_bit_row(&mut self.stab_row_z, words_per_row, pivot_id, qubit);

        if outcome {
            set_sign(&mut self.stab_signs_minus, pivot_id);
        } else {
            clear_sign(&mut self.stab_signs_minus, pivot_id);
        }
        clear_sign(&mut self.stab_signs_i, pivot_id);

        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }

    fn mz_internal(&mut self, qubit: usize) -> MeasurementResult {
        if let Some(result) = self.deterministic_meas(qubit) {
            return result;
        }

        let outcome = self.rng.random_bool(0.5);
        self.nondeterministic_meas(qubit, outcome)
    }

    /// Measure qubit with forced outcome for non-deterministic cases.
    pub fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        if let Some(result) = self.deterministic_meas(qubit) {
            return result;
        }
        self.nondeterministic_meas(qubit, forced_outcome)
    }
}

impl<R: SeedableRng + Rng + Debug + Clone> QuantumSimulator for DenseStabRowOnly<R> {
    fn reset(&mut self) -> &mut Self {
        self.init_state();
        self
    }
}

impl<R: SeedableRng + Rng + Debug + Clone> RngManageable for DenseStabRowOnly<R> {
    type Rng = R;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.rng = rng;
    }

    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

impl<R: SeedableRng + Rng + Debug + Clone> CliffordGateable for DenseStabRowOnly<R> {
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_h(q.index());
        }
        self
    }

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_sz(q.index());
        }
        self
    }

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(control, target) in pairs {
            self.apply_cx(control.index(), target.index());
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits.iter().map(|q| self.mz_internal(q.index())).collect()
    }
}

// ========== SparseColOnly: Sparse column-only representation ==========
//
// Uses sparse vectors instead of dense bitvectors for column storage.
// Each column stores a sorted list of generator indices that have X/Z on that qubit.
// This is efficient when stabilizers are sparse (few qubits per stabilizer).

/// Sparse column-only stabilizer simulator.
///
/// Uses `SmallVec` to store which generators have X/Z on each qubit.
/// More efficient than dense representation when:
/// - Stabilizers are sparse (few qubits per stabilizer)
/// - Gate operations dominate over measurements
///
/// Column weight in surface code is ~4-8 (bounded by locality), so sparse
/// operations are O(8) instead of O(n/64) for dense.
pub struct SparseColOnly {
    num_qubits: usize,
    // For each qubit, which stabilizers have X (sorted)
    stab_col_x: Vec<SmallVec<[u16; 8]>>,
    stab_col_z: Vec<SmallVec<[u16; 8]>>,
    destab_col_x: Vec<SmallVec<[u16; 8]>>,
    destab_col_z: Vec<SmallVec<[u16; 8]>>,
    // Signs as dense bitvector (always need O(n) signs)
    stab_signs_minus: Vec<u64>,
    stab_signs_i: Vec<u64>,
    destab_signs_minus: Vec<u64>,
    destab_signs_i: Vec<u64>,
    rng: PecosRng,
}

impl Debug for SparseColOnly {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SparseColOnly")
            .field("num_qubits", &self.num_qubits)
            .finish_non_exhaustive()
    }
}

impl Clone for SparseColOnly {
    fn clone(&self) -> Self {
        Self {
            num_qubits: self.num_qubits,
            stab_col_x: self.stab_col_x.clone(),
            stab_col_z: self.stab_col_z.clone(),
            destab_col_x: self.destab_col_x.clone(),
            destab_col_z: self.destab_col_z.clone(),
            stab_signs_minus: self.stab_signs_minus.clone(),
            stab_signs_i: self.stab_signs_i.clone(),
            destab_signs_minus: self.destab_signs_minus.clone(),
            destab_signs_i: self.destab_signs_i.clone(),
            rng: self.rng.clone(),
        }
    }
}

impl SparseColOnly {
    /// Create a new sparse column-only simulator with `n` qubits.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self::with_seed(num_qubits, 0)
    }

    /// Create a new simulator with a specific RNG seed.
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let sign_words = num_qubits.div_ceil(64);
        let stab_col_x = vec![SmallVec::new(); num_qubits];
        let mut stab_col_z = vec![SmallVec::new(); num_qubits];
        let mut destab_col_x = vec![SmallVec::new(); num_qubits];
        let destab_col_z = vec![SmallVec::new(); num_qubits];

        // Initialize: stabilizer[i] = Z_i, destabilizer[i] = X_i
        for i in 0..num_qubits {
            stab_col_z[i].push(i as u16);
            destab_col_x[i].push(i as u16);
        }

        Self {
            num_qubits,
            stab_col_x,
            stab_col_z,
            destab_col_x,
            destab_col_z,
            stab_signs_minus: vec![0; sign_words],
            stab_signs_i: vec![0; sign_words],
            destab_signs_minus: vec![0; sign_words],
            destab_signs_i: vec![0; sign_words],
            rng: PecosRng::seed_from_u64(seed),
        }
    }

    /// Toggle generator `g` in a sorted column.
    #[inline(always)]
    fn toggle_in_col(col: &mut SmallVec<[u16; 8]>, g: u16) {
        match col.binary_search(&g) {
            Ok(pos) => {
                col.remove(pos);
            }
            Err(pos) => {
                col.insert(pos, g);
            }
        }
    }

    /// Check if generator `g` is in a sorted column.
    #[inline(always)]
    fn contains(col: &SmallVec<[u16; 8]>, g: u16) -> bool {
        col.binary_search(&g).is_ok()
    }

    fn apply_cx(&mut self, control: usize, target: usize) {
        // CX: X_c -> X_c X_t, Z_t -> Z_c Z_t
        // X propagates: control -> target
        let gens_x: SmallVec<[u16; 8]> = self.stab_col_x[control].clone();
        for &g in &gens_x {
            Self::toggle_in_col(&mut self.stab_col_x[target], g);
        }
        let gens_x: SmallVec<[u16; 8]> = self.destab_col_x[control].clone();
        for &g in &gens_x {
            Self::toggle_in_col(&mut self.destab_col_x[target], g);
        }

        // Z propagates: target -> control
        let gens_z: SmallVec<[u16; 8]> = self.stab_col_z[target].clone();
        for &g in &gens_z {
            Self::toggle_in_col(&mut self.stab_col_z[control], g);
        }
        let gens_z: SmallVec<[u16; 8]> = self.destab_col_z[target].clone();
        for &g in &gens_z {
            Self::toggle_in_col(&mut self.destab_col_z[control], g);
        }
    }

    fn apply_h(&mut self, qubit: usize) {
        // H: X -> Z, Z -> X, Y -> -Y
        // Swap X and Z columns
        std::mem::swap(&mut self.stab_col_x[qubit], &mut self.stab_col_z[qubit]);
        std::mem::swap(&mut self.destab_col_x[qubit], &mut self.destab_col_z[qubit]);

        // Toggle sign for generators that had both X and Z (Y)
        // After swap, these are now in both columns
        for &g in &self.stab_col_x[qubit] {
            if Self::contains(&self.stab_col_z[qubit], g) {
                toggle_sign(&mut self.stab_signs_minus, g as usize);
            }
        }
        for &g in &self.destab_col_x[qubit] {
            if Self::contains(&self.destab_col_z[qubit], g) {
                toggle_sign(&mut self.destab_signs_minus, g as usize);
            }
        }
    }

    fn apply_sz(&mut self, qubit: usize) {
        // S/SZ gate: X -> iXZ (Y), Y -> -X, Z -> Z
        // The i-phase tracking handles all phase changes
        let gens_x: SmallVec<[u16; 8]> = self.stab_col_x[qubit].clone();
        for &g in &gens_x {
            let g_usize = g as usize;
            // i*i = -1: toggle minus if has both i and X
            if get_sign(&self.stab_signs_i, g_usize) {
                toggle_sign(&mut self.stab_signs_minus, g_usize);
            }
            // Toggle i for X generators
            toggle_sign(&mut self.stab_signs_i, g_usize);
            // Toggle Z
            Self::toggle_in_col(&mut self.stab_col_z[qubit], g);
        }
        let gens_x: SmallVec<[u16; 8]> = self.destab_col_x[qubit].clone();
        for &g in &gens_x {
            let g_usize = g as usize;
            if get_sign(&self.destab_signs_i, g_usize) {
                toggle_sign(&mut self.destab_signs_minus, g_usize);
            }
            toggle_sign(&mut self.destab_signs_i, g_usize);
            Self::toggle_in_col(&mut self.destab_col_z[qubit], g);
        }
    }

    fn deterministic_meas(&self, qubit: usize) -> Option<MeasurementResult> {
        // Check if any stabilizer has X on this qubit
        if !self.stab_col_x[qubit].is_empty() {
            return None;
        }

        // Deterministic measurement: compute product of stabilizers indexed by
        // destabilizers that have X on this qubit.
        let destab_ids: SmallVec<[u16; 8]> = self.destab_col_x[qubit].clone();

        if destab_ids.is_empty() {
            // No destabilizer has X on this qubit - return default |0>
            return Some(MeasurementResult {
                outcome: false,
                is_deterministic: true,
            });
        }

        // Count minus and i signs from the stabilizers at these indices
        let mut num_minuses: usize = 0;
        let mut num_is: usize = 0;
        for &g in &destab_ids {
            if get_sign(&self.stab_signs_minus, g as usize) {
                num_minuses += 1;
            }
            if get_sign(&self.stab_signs_i, g as usize) {
                num_is += 1;
            }
        }

        // Compute phase from multiplying stabilizers
        // Maintain cumulative X: cumulative_x[q] = true if odd number of stabilizers have X on q
        let mut cumulative_x = vec![false; self.num_qubits];

        for &g in &destab_ids {
            // Count overlap: positions where this stabilizer has Z and cumulative has X
            for (q, cx) in cumulative_x.iter().enumerate() {
                let has_z = Self::contains(&self.stab_col_z[q], g);
                if has_z && *cx {
                    num_minuses += 1;
                }
            }

            // XOR this stabilizer's X pattern into cumulative X
            for (q, cx) in cumulative_x.iter_mut().enumerate() {
                let has_x = Self::contains(&self.stab_col_x[q], g);
                if has_x {
                    *cx = !*cx;
                }
            }
        }

        // Add i phase contribution (i^n where n = num_is mod 4)
        // i^1 = i, i^2 = -1, i^3 = -i, i^4 = 1
        // For measurement, we only care about the real part sign
        if num_is & 2 != 0 {
            num_minuses += 1;
        }

        Some(MeasurementResult {
            outcome: num_minuses & 1 != 0,
            is_deterministic: true,
        })
    }

    /// XOR generator `src` into generator `dst` in all columns.
    fn xor_generator(&mut self, src: u16, dst: u16) {
        for q in 0..self.num_qubits {
            if Self::contains(&self.stab_col_x[q], src) {
                Self::toggle_in_col(&mut self.stab_col_x[q], dst);
            }
            if Self::contains(&self.stab_col_z[q], src) {
                Self::toggle_in_col(&mut self.stab_col_z[q], dst);
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn nondeterministic_meas(&mut self, qubit: usize, outcome: bool) -> MeasurementResult {
        let pivot = self.stab_col_x[qubit][0];
        let pivot_id = pivot as usize;

        let pivot_sign_minus = get_sign(&self.stab_signs_minus, pivot_id);
        let pivot_sign_i = get_sign(&self.stab_signs_i, pivot_id);

        // Handle pivot's i-phase contribution
        if pivot_sign_i {
            clear_sign(&mut self.stab_signs_i, pivot_id);
            let gens_with_x: SmallVec<[u16; 8]> = self.stab_col_x[qubit].clone();
            for &g in &gens_with_x {
                if g == pivot {
                    continue;
                }
                let g_id = g as usize;
                // Toggle minus for anticom stabs that have i (i * i = -1)
                if get_sign(&self.stab_signs_i, g_id) {
                    toggle_sign(&mut self.stab_signs_minus, g_id);
                }
                // Toggle i for all anticom stabs
                toggle_sign(&mut self.stab_signs_i, g_id);
            }
        }

        // XOR other stabilizers with X on this qubit into pivot
        let gens_with_x: SmallVec<[u16; 8]> = self.stab_col_x[qubit].clone();
        for &g in &gens_with_x {
            if g != pivot {
                // Phase: count Z_pivot & X_g overlaps
                let mut count = 0usize;
                for q in 0..self.num_qubits {
                    if Self::contains(&self.stab_col_z[q], pivot)
                        && Self::contains(&self.stab_col_x[q], g)
                    {
                        count += 1;
                    }
                }
                if count & 1 != 0 {
                    toggle_sign(&mut self.stab_signs_minus, g as usize);
                }
                if pivot_sign_minus {
                    toggle_sign(&mut self.stab_signs_minus, g as usize);
                }
                self.xor_generator(pivot, g);
            }
        }

        // Step 2b (Aaronson-Gottesman): XOR pivot stab into anti-commuting destabilizers
        let anticom_destabs: SmallVec<[u16; 8]> = self.destab_col_x[qubit].clone();
        for &g in &anticom_destabs {
            if g != pivot {
                // XOR stab[pivot] into destab[g]
                for q in 0..self.num_qubits {
                    if Self::contains(&self.stab_col_x[q], pivot) {
                        Self::toggle_in_col(&mut self.destab_col_x[q], g);
                    }
                    if Self::contains(&self.stab_col_z[q], pivot) {
                        Self::toggle_in_col(&mut self.destab_col_z[q], g);
                    }
                }
            }
        }

        // Copy old pivot stabilizer to destabilizer before replacing
        // Clear destab[pivot]
        for q in 0..self.num_qubits {
            if Self::contains(&self.destab_col_x[q], pivot) {
                Self::toggle_in_col(&mut self.destab_col_x[q], pivot);
            }
            if Self::contains(&self.destab_col_z[q], pivot) {
                Self::toggle_in_col(&mut self.destab_col_z[q], pivot);
            }
        }
        // Copy stab[pivot] to destab[pivot]
        for q in 0..self.num_qubits {
            if Self::contains(&self.stab_col_x[q], pivot) {
                Self::toggle_in_col(&mut self.destab_col_x[q], pivot);
            }
            if Self::contains(&self.stab_col_z[q], pivot) {
                Self::toggle_in_col(&mut self.destab_col_z[q], pivot);
            }
        }
        // Copy sign and clear i
        if get_sign(&self.stab_signs_minus, pivot_id) {
            set_sign(&mut self.destab_signs_minus, pivot_id);
        } else {
            clear_sign(&mut self.destab_signs_minus, pivot_id);
        }
        clear_sign(&mut self.destab_signs_i, pivot_id);

        // Clear pivot stabilizer
        for q in 0..self.num_qubits {
            if Self::contains(&self.stab_col_x[q], pivot) {
                Self::toggle_in_col(&mut self.stab_col_x[q], pivot);
            }
            if Self::contains(&self.stab_col_z[q], pivot) {
                Self::toggle_in_col(&mut self.stab_col_z[q], pivot);
            }
        }

        // Set pivot stabilizer to Z_qubit
        Self::toggle_in_col(&mut self.stab_col_z[qubit], pivot);

        // Set outcome and clear i
        if outcome {
            set_sign(&mut self.stab_signs_minus, pivot_id);
        } else {
            clear_sign(&mut self.stab_signs_minus, pivot_id);
        }
        clear_sign(&mut self.stab_signs_i, pivot_id);

        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }

    fn mz_internal(&mut self, qubit: usize) -> MeasurementResult {
        if let Some(result) = self.deterministic_meas(qubit) {
            return result;
        }
        let outcome = self.rng.random_bool(0.5);
        self.nondeterministic_meas(qubit, outcome)
    }

    /// Measure qubit with forced outcome for non-deterministic cases.
    pub fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        if let Some(result) = self.deterministic_meas(qubit) {
            return result;
        }
        self.nondeterministic_meas(qubit, forced_outcome)
    }
}

impl QuantumSimulator for SparseColOnly {
    fn reset(&mut self) -> &mut Self {
        let n = self.num_qubits;
        for q in 0..n {
            self.stab_col_x[q].clear();
            self.stab_col_z[q].clear();
            self.stab_col_z[q].push(q as u16);
            self.destab_col_x[q].clear();
            self.destab_col_x[q].push(q as u16);
            self.destab_col_z[q].clear();
        }
        self.stab_signs_minus.fill(0);
        self.stab_signs_i.fill(0);
        self.destab_signs_minus.fill(0);
        self.destab_signs_i.fill(0);
        self
    }
}

impl RngManageable for SparseColOnly {
    type Rng = PecosRng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.rng = rng;
    }

    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

impl CliffordGateable for SparseColOnly {
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for q in qubits {
            self.apply_sz(q.index());
        }
        self
    }

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for q in qubits {
            self.apply_h(q.index());
        }
        self
    }

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(control, target) in pairs {
            self.apply_cx(control.index(), target.index());
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits.iter().map(|q| self.mz_internal(q.index())).collect()
    }
}

// ========== SparseRowOnly: Sparse row-only representation ==========
//
// Uses sparse vectors instead of dense bitvectors for row storage.
// Each row stores a sorted list of qubit indices that have X/Z in that generator.
// This is efficient when stabilizers are sparse (few qubits per stabilizer).

/// Sparse row-only stabilizer simulator.
///
/// Uses `SmallVec` to store which qubits have X/Z in each generator.
/// More efficient than dense representation when:
/// - Stabilizers are sparse (few qubits per stabilizer)
/// - Row-based operations dominate (generator XORs, weight calculations)
///
/// Row weight in surface code is ~4-8 (bounded by locality), so sparse
/// operations are O(8) instead of O(n/64) for dense.
pub struct SparseRowOnly {
    num_qubits: usize,
    // For each generator, which qubits have X/Z (sorted)
    stab_row_x: Vec<SmallVec<[u16; 8]>>,
    stab_row_z: Vec<SmallVec<[u16; 8]>>,
    destab_row_x: Vec<SmallVec<[u16; 8]>>,
    destab_row_z: Vec<SmallVec<[u16; 8]>>,
    // Signs as dense bitvector (always need O(n) signs)
    stab_signs_minus: Vec<u64>,
    stab_signs_i: Vec<u64>,
    destab_signs_minus: Vec<u64>,
    destab_signs_i: Vec<u64>,
    rng: PecosRng,
}

impl Debug for SparseRowOnly {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SparseRowOnly")
            .field("num_qubits", &self.num_qubits)
            .finish_non_exhaustive()
    }
}

impl Clone for SparseRowOnly {
    fn clone(&self) -> Self {
        Self {
            num_qubits: self.num_qubits,
            stab_row_x: self.stab_row_x.clone(),
            stab_row_z: self.stab_row_z.clone(),
            destab_row_x: self.destab_row_x.clone(),
            destab_row_z: self.destab_row_z.clone(),
            stab_signs_minus: self.stab_signs_minus.clone(),
            stab_signs_i: self.stab_signs_i.clone(),
            destab_signs_minus: self.destab_signs_minus.clone(),
            destab_signs_i: self.destab_signs_i.clone(),
            rng: self.rng.clone(),
        }
    }
}

impl SparseRowOnly {
    /// Create a new sparse row-only simulator with `n` qubits.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self::with_seed(num_qubits, 0)
    }

    /// Create a new simulator with a specific RNG seed.
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let sign_words = num_qubits.div_ceil(64);
        let stab_row_x = vec![SmallVec::new(); num_qubits];
        let mut stab_row_z = vec![SmallVec::new(); num_qubits];
        let mut destab_row_x = vec![SmallVec::new(); num_qubits];
        let destab_row_z = vec![SmallVec::new(); num_qubits];

        // Initialize: stabilizer[i] = Z_i, destabilizer[i] = X_i
        for i in 0..num_qubits {
            stab_row_z[i].push(i as u16);
            destab_row_x[i].push(i as u16);
        }

        Self {
            num_qubits,
            stab_row_x,
            stab_row_z,
            destab_row_x,
            destab_row_z,
            stab_signs_minus: vec![0; sign_words],
            stab_signs_i: vec![0; sign_words],
            destab_signs_minus: vec![0; sign_words],
            destab_signs_i: vec![0; sign_words],
            rng: PecosRng::seed_from_u64(seed),
        }
    }

    /// Toggle qubit `q` in a sorted row.
    #[inline(always)]
    fn toggle_in_row(row: &mut SmallVec<[u16; 8]>, q: u16) {
        match row.binary_search(&q) {
            Ok(pos) => {
                row.remove(pos);
            }
            Err(pos) => {
                row.insert(pos, q);
            }
        }
    }

    /// Check if qubit `q` is in a sorted row.
    #[inline(always)]
    fn contains(row: &SmallVec<[u16; 8]>, q: u16) -> bool {
        row.binary_search(&q).is_ok()
    }

    fn apply_h(&mut self, qubit: usize) {
        let q = qubit as u16;
        // H: X -> Z, Z -> X, Y -> -Y
        for g in 0..self.num_qubits {
            let has_x = Self::contains(&self.stab_row_x[g], q);
            let has_z = Self::contains(&self.stab_row_z[g], q);
            if has_x && has_z {
                toggle_sign(&mut self.stab_signs_minus, g);
            }
            if has_x != has_z {
                Self::toggle_in_row(&mut self.stab_row_x[g], q);
                Self::toggle_in_row(&mut self.stab_row_z[g], q);
            }
        }
        for g in 0..self.num_qubits {
            let has_x = Self::contains(&self.destab_row_x[g], q);
            let has_z = Self::contains(&self.destab_row_z[g], q);
            if has_x && has_z {
                toggle_sign(&mut self.destab_signs_minus, g);
            }
            if has_x != has_z {
                Self::toggle_in_row(&mut self.destab_row_x[g], q);
                Self::toggle_in_row(&mut self.destab_row_z[g], q);
            }
        }
    }

    fn apply_sz(&mut self, qubit: usize) {
        let q = qubit as u16;
        // S/SZ gate: X -> iXZ (Y), Y -> -X, Z -> Z
        for g in 0..self.num_qubits {
            if Self::contains(&self.stab_row_x[g], q) {
                if get_sign(&self.stab_signs_i, g) {
                    toggle_sign(&mut self.stab_signs_minus, g);
                }
                toggle_sign(&mut self.stab_signs_i, g);
                Self::toggle_in_row(&mut self.stab_row_z[g], q);
            }
        }
        for g in 0..self.num_qubits {
            if Self::contains(&self.destab_row_x[g], q) {
                if get_sign(&self.destab_signs_i, g) {
                    toggle_sign(&mut self.destab_signs_minus, g);
                }
                toggle_sign(&mut self.destab_signs_i, g);
                Self::toggle_in_row(&mut self.destab_row_z[g], q);
            }
        }
    }

    fn apply_cx(&mut self, control: usize, target: usize) {
        let c = control as u16;
        let t = target as u16;
        // CX: X_c -> X_c X_t, Z_t -> Z_c Z_t
        for g in 0..self.num_qubits {
            if Self::contains(&self.stab_row_x[g], c) {
                Self::toggle_in_row(&mut self.stab_row_x[g], t);
            }
            if Self::contains(&self.stab_row_z[g], t) {
                Self::toggle_in_row(&mut self.stab_row_z[g], c);
            }
        }
        for g in 0..self.num_qubits {
            if Self::contains(&self.destab_row_x[g], c) {
                Self::toggle_in_row(&mut self.destab_row_x[g], t);
            }
            if Self::contains(&self.destab_row_z[g], t) {
                Self::toggle_in_row(&mut self.destab_row_z[g], c);
            }
        }
    }

    fn deterministic_meas(&self, qubit: usize) -> Option<MeasurementResult> {
        let q = qubit as u16;

        // Check if any stabilizer has X on this qubit
        for g in 0..self.num_qubits {
            if Self::contains(&self.stab_row_x[g], q) {
                return None;
            }
        }

        // Collect destabilizer indices with X on this qubit
        let mut destab_ids: SmallVec<[usize; 8]> = SmallVec::new();
        for g in 0..self.num_qubits {
            if Self::contains(&self.destab_row_x[g], q) {
                destab_ids.push(g);
            }
        }

        if destab_ids.is_empty() {
            return Some(MeasurementResult {
                outcome: false,
                is_deterministic: true,
            });
        }

        // Count minus and i signs from the stabilizers at these indices
        let mut num_minuses: usize = 0;
        let mut num_is: usize = 0;
        for &g in &destab_ids {
            if get_sign(&self.stab_signs_minus, g) {
                num_minuses += 1;
            }
            if get_sign(&self.stab_signs_i, g) {
                num_is += 1;
            }
        }

        // Compute phase from multiplying stabilizers
        let mut cumulative_x = vec![false; self.num_qubits];

        for &g in &destab_ids {
            // Count overlap: where this stabilizer has Z and cumulative has X
            for &zq in &self.stab_row_z[g] {
                if cumulative_x[zq as usize] {
                    num_minuses += 1;
                }
            }

            // XOR this stabilizer's X into cumulative
            for &xq in &self.stab_row_x[g] {
                cumulative_x[xq as usize] = !cumulative_x[xq as usize];
            }
        }

        // Add i phase contribution (i^2 = -1, i^3 = -i)
        if num_is & 2 != 0 {
            num_minuses += 1;
        }

        Some(MeasurementResult {
            outcome: num_minuses & 1 != 0,
            is_deterministic: true,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn nondeterministic_meas(&mut self, qubit: usize, outcome: bool) -> MeasurementResult {
        let q = qubit as u16;

        // Find anti-commuting stabilizers and minimum weight one
        let mut anticom_stabs: SmallVec<[usize; 8]> = SmallVec::new();
        let mut min_weight = usize::MAX;
        let mut pivot_id = 0;

        for g in 0..self.num_qubits {
            if Self::contains(&self.stab_row_x[g], q) {
                anticom_stabs.push(g);
                let weight = self.stab_row_x[g].len() + self.stab_row_z[g].len();
                if weight < min_weight {
                    min_weight = weight;
                    pivot_id = g;
                }
            }
        }

        let pivot_sign_minus = get_sign(&self.stab_signs_minus, pivot_id);
        let pivot_sign_i = get_sign(&self.stab_signs_i, pivot_id);

        // Handle pivot's i-phase contribution
        if pivot_sign_i {
            clear_sign(&mut self.stab_signs_i, pivot_id);
            for &g in &anticom_stabs {
                if g == pivot_id {
                    continue;
                }
                if get_sign(&self.stab_signs_i, g) {
                    toggle_sign(&mut self.stab_signs_minus, g);
                }
                toggle_sign(&mut self.stab_signs_i, g);
            }
        }

        // Clone pivot rows for use in XOR operations
        let pivot_row_x: SmallVec<[u16; 8]> = self.stab_row_x[pivot_id].clone();
        let pivot_row_z: SmallVec<[u16; 8]> = self.stab_row_z[pivot_id].clone();

        // XOR pivot into other anti-commuting stabilizers
        for &g in &anticom_stabs {
            if g == pivot_id {
                continue;
            }

            // Phase calculation: count Z_pivot & X_g overlaps
            let mut count = 0usize;
            for &pz in &pivot_row_z {
                if Self::contains(&self.stab_row_x[g], pz) {
                    count += 1;
                }
            }
            if count & 1 != 0 {
                toggle_sign(&mut self.stab_signs_minus, g);
            }
            if pivot_sign_minus {
                toggle_sign(&mut self.stab_signs_minus, g);
            }

            // XOR rows
            for &pq in &pivot_row_x {
                Self::toggle_in_row(&mut self.stab_row_x[g], pq);
            }
            for &pq in &pivot_row_z {
                Self::toggle_in_row(&mut self.stab_row_z[g], pq);
            }
        }

        // Step 2b (Aaronson-Gottesman): XOR pivot stab into anti-commuting destabilizers
        for g in 0..self.num_qubits {
            if g == pivot_id {
                continue;
            }
            if Self::contains(&self.destab_row_x[g], q) {
                for &pq in &pivot_row_x {
                    Self::toggle_in_row(&mut self.destab_row_x[g], pq);
                }
                for &pq in &pivot_row_z {
                    Self::toggle_in_row(&mut self.destab_row_z[g], pq);
                }
            }
        }

        // Copy old pivot stabilizer to destabilizer before replacing
        self.destab_row_x[pivot_id].clone_from(&self.stab_row_x[pivot_id]);
        self.destab_row_z[pivot_id].clone_from(&self.stab_row_z[pivot_id]);
        if get_sign(&self.stab_signs_minus, pivot_id) {
            set_sign(&mut self.destab_signs_minus, pivot_id);
        } else {
            clear_sign(&mut self.destab_signs_minus, pivot_id);
        }
        clear_sign(&mut self.destab_signs_i, pivot_id);

        // Replace pivot stabilizer with Z_qubit
        self.stab_row_x[pivot_id].clear();
        self.stab_row_z[pivot_id].clear();
        self.stab_row_z[pivot_id].push(q);

        if outcome {
            set_sign(&mut self.stab_signs_minus, pivot_id);
        } else {
            clear_sign(&mut self.stab_signs_minus, pivot_id);
        }
        clear_sign(&mut self.stab_signs_i, pivot_id);

        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }

    fn mz_internal(&mut self, qubit: usize) -> MeasurementResult {
        if let Some(result) = self.deterministic_meas(qubit) {
            return result;
        }
        let outcome = self.rng.random_bool(0.5);
        self.nondeterministic_meas(qubit, outcome)
    }

    /// Measure qubit with forced outcome for non-deterministic cases.
    pub fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        if let Some(result) = self.deterministic_meas(qubit) {
            return result;
        }
        self.nondeterministic_meas(qubit, forced_outcome)
    }
}

impl QuantumSimulator for SparseRowOnly {
    fn reset(&mut self) -> &mut Self {
        let n = self.num_qubits;
        for g in 0..n {
            self.stab_row_x[g].clear();
            self.stab_row_z[g].clear();
            self.stab_row_z[g].push(g as u16);
            self.destab_row_x[g].clear();
            self.destab_row_x[g].push(g as u16);
            self.destab_row_z[g].clear();
        }
        self.stab_signs_minus.fill(0);
        self.stab_signs_i.fill(0);
        self.destab_signs_minus.fill(0);
        self.destab_signs_i.fill(0);
        self
    }
}

impl RngManageable for SparseRowOnly {
    type Rng = PecosRng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.rng = rng;
    }

    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

impl CliffordGateable for SparseRowOnly {
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for q in qubits {
            self.apply_sz(q.index());
        }
        self
    }

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for q in qubits {
            self.apply_h(q.index());
        }
        self
    }

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(control, target) in pairs {
            self.apply_cx(control.index(), target.index());
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits.iter().map(|q| self.mz_internal(q.index())).collect()
    }
}

// ========== StabilizerTableauSimulator implementations ==========

/// Build a tableau string from column-only u64 storage.
fn col_only_tableau_string(
    num_qubits: usize,
    words_per_col: usize,
    col_x: &[u64],
    col_z: &[u64],
    signs_minus: &[u64],
    signs_i: &[u64],
) -> String {
    let mut result = String::with_capacity(num_qubits * num_qubits + num_qubits + 2);
    for g in 0..num_qubits {
        if get_sign(signs_minus, g) {
            result.push('-');
        } else {
            result.push('+');
        }
        if get_sign(signs_i, g) {
            result.push('i');
        }

        for qubit in 0..num_qubits {
            let word_idx = qubit * words_per_col + g / 64;
            let bit_mask = 1u64 << (g % 64);
            let in_x = col_x[word_idx] & bit_mask != 0;
            let in_z = col_z[word_idx] & bit_mask != 0;
            let ch = match (in_x, in_z) {
                (false, false) => 'I',
                (true, false) => 'X',
                (false, true) => 'Z',
                (true, true) => 'Y',
            };
            result.push(ch);
        }
        result.push('\n');
    }
    result
}

/// Build a tableau string from row-only u64 storage.
fn row_only_tableau_string(
    num_qubits: usize,
    words_per_row: usize,
    row_x: &[u64],
    row_z: &[u64],
    signs_minus: &[u64],
    signs_i: &[u64],
) -> String {
    let mut result = String::with_capacity(num_qubits * num_qubits + num_qubits + 2);
    for g in 0..num_qubits {
        if get_sign(signs_minus, g) {
            result.push('-');
        } else {
            result.push('+');
        }
        if get_sign(signs_i, g) {
            result.push('i');
        }

        let base = g * words_per_row;
        for qubit in 0..num_qubits {
            let word_idx = base + qubit / 64;
            let bit_mask = 1u64 << (qubit % 64);
            let in_x = row_x[word_idx] & bit_mask != 0;
            let in_z = row_z[word_idx] & bit_mask != 0;
            let ch = match (in_x, in_z) {
                (false, false) => 'I',
                (true, false) => 'X',
                (false, true) => 'Z',
                (true, true) => 'Y',
            };
            result.push(ch);
        }
        result.push('\n');
    }
    result
}

impl<R: SeedableRng + Rng + Debug + Clone> StabilizerTableauSimulator for DenseStabColOnly<R> {
    fn stab_tableau(&self) -> String {
        col_only_tableau_string(
            self.num_qubits,
            self.words_per_col,
            &self.stab_col_x,
            &self.stab_col_z,
            &self.stab_signs_minus,
            &self.stab_signs_i,
        )
    }

    fn destab_tableau(&self) -> String {
        col_only_tableau_string(
            self.num_qubits,
            self.words_per_col,
            &self.destab_col_x,
            &self.destab_col_z,
            &self.destab_signs_minus,
            &self.destab_signs_i,
        )
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

impl<R: SeedableRng + Rng + Debug + Clone> StabilizerTableauSimulator for DenseStabRowOnly<R> {
    fn stab_tableau(&self) -> String {
        row_only_tableau_string(
            self.num_qubits,
            self.words_per_row,
            &self.stab_row_x,
            &self.stab_row_z,
            &self.stab_signs_minus,
            &self.stab_signs_i,
        )
    }

    fn destab_tableau(&self) -> String {
        row_only_tableau_string(
            self.num_qubits,
            self.words_per_row,
            &self.destab_row_x,
            &self.destab_row_z,
            &self.destab_signs_minus,
            &self.destab_signs_i,
        )
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

impl StabilizerTableauSimulator for SparseColOnly {
    fn stab_tableau(&self) -> String {
        sparse_col_tableau_string(
            self.num_qubits,
            &self.stab_col_x,
            &self.stab_col_z,
            &self.stab_signs_minus,
            &self.stab_signs_i,
        )
    }

    fn destab_tableau(&self) -> String {
        sparse_col_tableau_string(
            self.num_qubits,
            &self.destab_col_x,
            &self.destab_col_z,
            &self.destab_signs_minus,
            &self.destab_signs_i,
        )
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

/// Build a tableau string from sparse column storage (`SmallVec`<[u16; 8]>).
fn sparse_col_tableau_string(
    num_qubits: usize,
    col_x: &[SmallVec<[u16; 8]>],
    col_z: &[SmallVec<[u16; 8]>],
    signs_minus: &[u64],
    signs_i: &[u64],
) -> String {
    let mut result = String::with_capacity(num_qubits * num_qubits + num_qubits + 2);
    for g in 0..num_qubits {
        if get_sign(signs_minus, g) {
            result.push('-');
        } else {
            result.push('+');
        }
        if get_sign(signs_i, g) {
            result.push('i');
        }

        let g16 = g as u16;
        for qubit in 0..num_qubits {
            let in_x = col_x[qubit].binary_search(&g16).is_ok();
            let in_z = col_z[qubit].binary_search(&g16).is_ok();
            let ch = match (in_x, in_z) {
                (false, false) => 'I',
                (true, false) => 'X',
                (false, true) => 'Z',
                (true, true) => 'Y',
            };
            result.push(ch);
        }
        result.push('\n');
    }
    result
}

/// Build a tableau string from sparse row storage (`SmallVec`<[u16; 8]>).
fn sparse_row_tableau_string(
    num_qubits: usize,
    row_x: &[SmallVec<[u16; 8]>],
    row_z: &[SmallVec<[u16; 8]>],
    signs_minus: &[u64],
    signs_i: &[u64],
) -> String {
    let mut result = String::with_capacity(num_qubits * num_qubits + num_qubits + 2);
    for g in 0..num_qubits {
        if get_sign(signs_minus, g) {
            result.push('-');
        } else {
            result.push('+');
        }
        if get_sign(signs_i, g) {
            result.push('i');
        }

        for qubit in 0..num_qubits {
            let q = qubit as u16;
            let in_x = row_x[g].binary_search(&q).is_ok();
            let in_z = row_z[g].binary_search(&q).is_ok();
            let ch = match (in_x, in_z) {
                (false, false) => 'I',
                (true, false) => 'X',
                (false, true) => 'Z',
                (true, true) => 'Y',
            };
            result.push(ch);
        }
        result.push('\n');
    }
    result
}

impl StabilizerTableauSimulator for SparseRowOnly {
    fn stab_tableau(&self) -> String {
        sparse_row_tableau_string(
            self.num_qubits,
            &self.stab_row_x,
            &self.stab_row_z,
            &self.stab_signs_minus,
            &self.stab_signs_i,
        )
    }

    fn destab_tableau(&self) -> String {
        sparse_row_tableau_string(
            self.num_qubits,
            &self.destab_row_x,
            &self.destab_row_z,
            &self.destab_signs_minus,
            &self.destab_signs_i,
        )
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

// ========== ForcedMeasurement implementations ==========

use crate::stabilizer_test_utils::{ForcedMeasurement, StabilizerSimulator};

impl<R: SeedableRng + Rng + Debug + Clone> ForcedMeasurement for DenseStabColOnly<R> {
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        DenseStabColOnly::mz_forced(self, qubit, forced_outcome)
    }
}

impl<R: SeedableRng + Rng + Debug + Clone> ForcedMeasurement for DenseStabRowOnly<R> {
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        DenseStabRowOnly::mz_forced(self, qubit, forced_outcome)
    }
}

impl ForcedMeasurement for SparseColOnly {
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        SparseColOnly::mz_forced(self, qubit, forced_outcome)
    }
}

impl ForcedMeasurement for SparseRowOnly {
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        SparseRowOnly::mz_forced(self, qubit, forced_outcome)
    }
}

// ========== StabilizerSimulator implementations ==========

impl StabilizerSimulator for DenseStabColOnly<PecosRng> {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

impl StabilizerSimulator for DenseStabRowOnly<PecosRng> {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

impl StabilizerSimulator for SparseColOnly {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

impl StabilizerSimulator for SparseRowOnly {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_col_only_bell_state() {
        let mut sim: DenseStabColOnly = DenseStabColOnly::with_seed(2, 42);
        sim.h(&[QubitId(0)]);
        sim.cx(&[(QubitId(0), QubitId(1))]);
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert_eq!(results[0].outcome, results[1].outcome);
    }

    #[test]
    fn test_row_only_bell_state() {
        let mut sim: DenseStabRowOnly = DenseStabRowOnly::with_seed(2, 42);
        sim.h(&[QubitId(0)]);
        sim.cx(&[(QubitId(0), QubitId(1))]);
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert_eq!(results[0].outcome, results[1].outcome);
    }

    #[test]
    fn test_col_only_ghz() {
        let mut sim: DenseStabColOnly = DenseStabColOnly::with_seed(5, 123);
        sim.h(&[QubitId(0)]);
        for i in 0..4 {
            sim.cx(&[(QubitId(i), QubitId(i + 1))]);
        }
        let results = sim.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3), QubitId(4)]);
        let first = results[0].outcome;
        for r in &results {
            assert_eq!(r.outcome, first);
        }
    }

    #[test]
    fn test_row_only_ghz() {
        let mut sim: DenseStabRowOnly = DenseStabRowOnly::with_seed(5, 123);
        sim.h(&[QubitId(0)]);
        for i in 0..4 {
            sim.cx(&[(QubitId(i), QubitId(i + 1))]);
        }
        let results = sim.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3), QubitId(4)]);
        let first = results[0].outcome;
        for r in &results {
            assert_eq!(r.outcome, first);
        }
    }

    #[test]
    fn test_col_only_deterministic_z() {
        let mut sim: DenseStabColOnly = DenseStabColOnly::new(3);
        // In |0> state, Z measurement should be deterministic 0
        let results = sim.mz(&[QubitId(0), QubitId(1), QubitId(2)]);
        for r in &results {
            assert!(r.is_deterministic);
            assert!(!r.outcome);
        }
    }

    #[test]
    fn test_row_only_deterministic_z() {
        let mut sim: DenseStabRowOnly = DenseStabRowOnly::new(3);
        // In |0> state, Z measurement should be deterministic 0
        let results = sim.mz(&[QubitId(0), QubitId(1), QubitId(2)]);
        for r in &results {
            assert!(r.is_deterministic);
            assert!(!r.outcome);
        }
    }

    #[test]
    fn test_col_only_reset() {
        let mut sim: DenseStabColOnly = DenseStabColOnly::with_seed(2, 42);
        sim.h(&[QubitId(0)]);
        sim.cx(&[(QubitId(0), QubitId(1))]);
        sim.reset();
        // After reset, should be back to |00> state
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert!(results[0].is_deterministic);
        assert!(!results[0].outcome);
        assert!(results[1].is_deterministic);
        assert!(!results[1].outcome);
    }

    #[test]
    fn test_row_only_reset() {
        let mut sim: DenseStabRowOnly = DenseStabRowOnly::with_seed(2, 42);
        sim.h(&[QubitId(0)]);
        sim.cx(&[(QubitId(0), QubitId(1))]);
        sim.reset();
        // After reset, should be back to |00> state
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert!(results[0].is_deterministic);
        assert!(!results[0].outcome);
        assert!(results[1].is_deterministic);
        assert!(!results[1].outcome);
    }

    // SparseColOnly tests
    #[test]
    fn test_sparse_col_only_bell_state() {
        let mut sim: SparseColOnly = SparseColOnly::with_seed(2, 42);
        sim.h(&[QubitId(0)]);
        sim.cx(&[(QubitId(0), QubitId(1))]);
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert_eq!(results[0].outcome, results[1].outcome);
    }

    #[test]
    fn test_sparse_col_only_ghz() {
        let mut sim: SparseColOnly = SparseColOnly::with_seed(5, 123);
        sim.h(&[QubitId(0)]);
        for i in 0..4 {
            sim.cx(&[(QubitId(i), QubitId(i + 1))]);
        }
        let results = sim.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3), QubitId(4)]);
        let first = results[0].outcome;
        for r in &results {
            assert_eq!(r.outcome, first);
        }
    }

    #[test]
    fn test_sparse_col_only_deterministic_z() {
        let mut sim: SparseColOnly = SparseColOnly::new(3);
        // In |0> state, Z measurement should be deterministic 0
        let results = sim.mz(&[QubitId(0), QubitId(1), QubitId(2)]);
        for r in &results {
            assert!(r.is_deterministic);
            assert!(!r.outcome);
        }
    }

    #[test]
    fn test_sparse_col_only_reset() {
        let mut sim: SparseColOnly = SparseColOnly::with_seed(2, 42);
        sim.h(&[QubitId(0)]);
        sim.cx(&[(QubitId(0), QubitId(1))]);
        sim.reset();
        // After reset, should be back to |00> state
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert!(results[0].is_deterministic);
        assert!(!results[0].outcome);
        assert!(results[1].is_deterministic);
        assert!(!results[1].outcome);
    }

    // SparseRowOnly tests
    #[test]
    fn test_sparse_row_only_bell_state() {
        let mut sim: SparseRowOnly = SparseRowOnly::with_seed(2, 42);
        sim.h(&[QubitId(0)]);
        sim.cx(&[(QubitId(0), QubitId(1))]);
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert_eq!(results[0].outcome, results[1].outcome);
    }

    #[test]
    fn test_sparse_row_only_ghz() {
        let mut sim: SparseRowOnly = SparseRowOnly::with_seed(5, 123);
        sim.h(&[QubitId(0)]);
        for i in 0..4 {
            sim.cx(&[(QubitId(i), QubitId(i + 1))]);
        }
        let results = sim.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3), QubitId(4)]);
        let first = results[0].outcome;
        for r in &results {
            assert_eq!(r.outcome, first);
        }
    }

    #[test]
    fn test_sparse_row_only_deterministic_z() {
        let mut sim: SparseRowOnly = SparseRowOnly::new(3);
        // In |0> state, Z measurement should be deterministic 0
        let results = sim.mz(&[QubitId(0), QubitId(1), QubitId(2)]);
        for r in &results {
            assert!(r.is_deterministic);
            assert!(!r.outcome);
        }
    }

    #[test]
    fn test_sparse_row_only_reset() {
        let mut sim: SparseRowOnly = SparseRowOnly::with_seed(2, 42);
        sim.h(&[QubitId(0)]);
        sim.cx(&[(QubitId(0), QubitId(1))]);
        sim.reset();
        // After reset, should be back to |00> state
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert!(results[0].is_deterministic);
        assert!(!results[0].outcome);
        assert!(results[1].is_deterministic);
        assert!(!results[1].outcome);
    }

    // Full test suite tests
    use crate::SparseStab;
    use crate::stabilizer_test_utils::{
        compare_simulators_on_random_circuit_direct, run_full_stabilizer_test_suite,
    };

    #[test]
    fn test_col_only_vs_sparse_stab() {
        // Compare DenseStabColOnly with SparseStab (known good) on multiple random circuits
        for i in 0..20 {
            let seed = 12345u64.wrapping_add(i);
            let mut col_only: DenseStabColOnly = DenseStabColOnly::with_seed(8, 42);
            let mut sparse: SparseStab = SparseStab::with_seed(8, 42);
            compare_simulators_on_random_circuit_direct(&mut col_only, &mut sparse, 8, 20, seed);
        }
    }

    #[test]
    fn test_sparse_col_only_vs_sparse_stab() {
        use crate::stabilizer_test_utils::calculate_basis_probability;

        // Test case that would give 0.25 probability (2 qubits, both in superposition)
        let mut sparse_col: SparseColOnly = SparseColOnly::with_seed(2, 42);
        let mut sparse: SparseStab = SparseStab::with_seed(2, 42);

        // Apply H(0) and H(1) to create |++> state
        sparse_col.h(&[QubitId(0), QubitId(1)]);
        sparse.h(&[QubitId(0), QubitId(1)]);

        // Test probability calculation - each basis state should have 0.25 probability
        for basis in 0..4 {
            let prob_sparse_col = calculate_basis_probability(&sparse_col, basis, 2);
            let prob_sparse = calculate_basis_probability(&sparse, basis, 2);
            println!(
                "Probability |{basis:02b}>: sparse_col={prob_sparse_col}, sparse={prob_sparse}"
            );
            assert!(
                (prob_sparse_col - prob_sparse).abs() < 1e-10,
                "Probability mismatch for basis {basis}"
            );
        }

        // Test with more complex gates - SZ
        let mut sparse_col: SparseColOnly = SparseColOnly::with_seed(2, 42);
        let mut sparse: SparseStab = SparseStab::with_seed(2, 42);

        sparse_col.h(&[QubitId(0)]);
        sparse.h(&[QubitId(0)]);
        sparse_col.sz(&[QubitId(0)]);
        sparse.sz(&[QubitId(0)]);

        println!("\nAfter H(0), SZ(0):");
        for basis in 0..4 {
            let prob_sparse_col = calculate_basis_probability(&sparse_col, basis, 2);
            let prob_sparse = calculate_basis_probability(&sparse, basis, 2);
            println!(
                "Probability |{basis:02b}>: sparse_col={prob_sparse_col}, sparse={prob_sparse}"
            );
            assert!(
                (prob_sparse_col - prob_sparse).abs() < 1e-10,
                "Probability mismatch after SZ for basis {basis}"
            );
        }
    }

    #[test]
    fn test_sz_phase_tracking() {
        // Test SZ phase tracking by comparing with SparseStab
        // Initial state: S[0] = Z_0
        // After SZ(0): S[0] should become Y_0 = iXZ (X=[0], Z=[0], i=true)
        let mut col_only: DenseStabColOnly = DenseStabColOnly::with_seed(1, 42);
        let mut sparse: SparseStab = SparseStab::with_seed(1, 42);

        // Print initial state
        println!("Initial state:");
        let col_minus = col_only.stab_signs_minus[0] & 1 != 0;
        let col_i = col_only.stab_signs_i[0] & 1 != 0;
        let col_x = col_only.stab_col_x[0] & 1 != 0;
        let col_z = col_only.stab_col_z[0] & 1 != 0;
        println!("  col_only S[0]: X={col_x}, Z={col_z}, minus={col_minus}, i={col_i}");

        let sparse_minus = sparse.stabs.signs_minus.contains(0);
        let sparse_i = sparse.stabs.signs_i.contains(0);
        let sparse_x = sparse.stabs.row_x[0].contains(0);
        let sparse_z = sparse.stabs.row_z[0].contains(0);
        println!("  sparse S[0]: X={sparse_x}, Z={sparse_z}, minus={sparse_minus}, i={sparse_i}");

        // Apply SZ
        col_only.sz(&[QubitId(0)]);
        sparse.sz(&[QubitId(0)]);

        println!("After SZ(0):");
        let col_minus = col_only.stab_signs_minus[0] & 1 != 0;
        let col_i = col_only.stab_signs_i[0] & 1 != 0;
        let col_x = col_only.stab_col_x[0] & 1 != 0;
        let col_z = col_only.stab_col_z[0] & 1 != 0;
        println!("  col_only S[0]: X={col_x}, Z={col_z}, minus={col_minus}, i={col_i}");

        let sparse_minus = sparse.stabs.signs_minus.contains(0);
        let sparse_i = sparse.stabs.signs_i.contains(0);
        let sparse_x = sparse.stabs.row_x[0].contains(0);
        let sparse_z = sparse.stabs.row_z[0].contains(0);
        println!("  sparse S[0]: X={sparse_x}, Z={sparse_z}, minus={sparse_minus}, i={sparse_i}");

        // Expected after SZ(0) on Z_0: Z_0 → Z_0 (no change, since SZ doesn't affect Z)
        // SZ transforms: X -> iXZ, Z -> Z
        // So Z_0 stays Z_0

        // Now apply H first, then SZ
        let mut col_only: DenseStabColOnly = DenseStabColOnly::with_seed(1, 42);
        let mut sparse: SparseStab = SparseStab::with_seed(1, 42);

        col_only.h(&[QubitId(0)]);
        sparse.h(&[QubitId(0)]);
        println!("\nAfter H(0):");
        let col_minus = col_only.stab_signs_minus[0] & 1 != 0;
        let col_i = col_only.stab_signs_i[0] & 1 != 0;
        let col_x = col_only.stab_col_x[0] & 1 != 0;
        let col_z = col_only.stab_col_z[0] & 1 != 0;
        println!("  col_only S[0]: X={col_x}, Z={col_z}, minus={col_minus}, i={col_i}");

        let sparse_minus = sparse.stabs.signs_minus.contains(0);
        let sparse_i = sparse.stabs.signs_i.contains(0);
        let sparse_x = sparse.stabs.row_x[0].contains(0);
        let sparse_z = sparse.stabs.row_z[0].contains(0);
        println!("  sparse S[0]: X={sparse_x}, Z={sparse_z}, minus={sparse_minus}, i={sparse_i}");

        col_only.sz(&[QubitId(0)]);
        sparse.sz(&[QubitId(0)]);
        println!("After SZ(0):");
        let col_minus = col_only.stab_signs_minus[0] & 1 != 0;
        let col_i = col_only.stab_signs_i[0] & 1 != 0;
        let col_x = col_only.stab_col_x[0] & 1 != 0;
        let col_z = col_only.stab_col_z[0] & 1 != 0;
        println!("  col_only S[0]: X={col_x}, Z={col_z}, minus={col_minus}, i={col_i}");

        let sparse_minus = sparse.stabs.signs_minus.contains(0);
        let sparse_i = sparse.stabs.signs_i.contains(0);
        let sparse_x = sparse.stabs.row_x[0].contains(0);
        let sparse_z = sparse.stabs.row_z[0].contains(0);
        println!("  sparse S[0]: X={sparse_x}, Z={sparse_z}, minus={sparse_minus}, i={sparse_i}");

        // Check they match
        assert_eq!(col_x, sparse_x, "X mismatch");
        assert_eq!(col_z, sparse_z, "Z mismatch");
        assert_eq!(col_minus, sparse_minus, "minus mismatch");
        assert_eq!(col_i, sparse_i, "i mismatch");

        // Now test full SXdg: H * SZ * SZ * SZ * H
        let mut col_only: DenseStabColOnly = DenseStabColOnly::with_seed(1, 42);
        let mut sparse: SparseStab = SparseStab::with_seed(1, 42);

        println!("\n=== Testing SXdg manually (H * SZ * SZ * SZ * H) ===");
        println!("Initial: S[0] = Z");

        col_only.h(&[QubitId(0)]);
        sparse.h(&[QubitId(0)]);
        let col_minus = col_only.stab_signs_minus[0] & 1 != 0;
        let col_i = col_only.stab_signs_i[0] & 1 != 0;
        let col_x = col_only.stab_col_x[0] & 1 != 0;
        let col_z = col_only.stab_col_z[0] & 1 != 0;
        let sparse_minus = sparse.stabs.signs_minus.contains(0);
        let sparse_i = sparse.stabs.signs_i.contains(0);
        let sparse_x = sparse.stabs.row_x[0].contains(0);
        let sparse_z = sparse.stabs.row_z[0].contains(0);
        println!(
            "After H: col=(X={col_x}, Z={col_z}, minus={col_minus}, i={col_i}), sparse=(X={sparse_x}, Z={sparse_z}, minus={sparse_minus}, i={sparse_i})"
        );

        col_only.sz(&[QubitId(0)]);
        sparse.sz(&[QubitId(0)]);
        let col_minus = col_only.stab_signs_minus[0] & 1 != 0;
        let col_i = col_only.stab_signs_i[0] & 1 != 0;
        let col_x = col_only.stab_col_x[0] & 1 != 0;
        let col_z = col_only.stab_col_z[0] & 1 != 0;
        let sparse_minus = sparse.stabs.signs_minus.contains(0);
        let sparse_i = sparse.stabs.signs_i.contains(0);
        let sparse_x = sparse.stabs.row_x[0].contains(0);
        let sparse_z = sparse.stabs.row_z[0].contains(0);
        println!(
            "After SZ 1: col=(X={col_x}, Z={col_z}, minus={col_minus}, i={col_i}), sparse=(X={sparse_x}, Z={sparse_z}, minus={sparse_minus}, i={sparse_i})"
        );

        col_only.sz(&[QubitId(0)]);
        sparse.sz(&[QubitId(0)]);
        let col_minus = col_only.stab_signs_minus[0] & 1 != 0;
        let col_i = col_only.stab_signs_i[0] & 1 != 0;
        let col_x = col_only.stab_col_x[0] & 1 != 0;
        let col_z = col_only.stab_col_z[0] & 1 != 0;
        let sparse_minus = sparse.stabs.signs_minus.contains(0);
        let sparse_i = sparse.stabs.signs_i.contains(0);
        let sparse_x = sparse.stabs.row_x[0].contains(0);
        let sparse_z = sparse.stabs.row_z[0].contains(0);
        println!(
            "After SZ 2: col=(X={col_x}, Z={col_z}, minus={col_minus}, i={col_i}), sparse=(X={sparse_x}, Z={sparse_z}, minus={sparse_minus}, i={sparse_i})"
        );

        col_only.sz(&[QubitId(0)]);
        sparse.sz(&[QubitId(0)]);
        let col_minus = col_only.stab_signs_minus[0] & 1 != 0;
        let col_i = col_only.stab_signs_i[0] & 1 != 0;
        let col_x = col_only.stab_col_x[0] & 1 != 0;
        let col_z = col_only.stab_col_z[0] & 1 != 0;
        let sparse_minus = sparse.stabs.signs_minus.contains(0);
        let sparse_i = sparse.stabs.signs_i.contains(0);
        let sparse_x = sparse.stabs.row_x[0].contains(0);
        let sparse_z = sparse.stabs.row_z[0].contains(0);
        println!(
            "After SZ 3: col=(X={col_x}, Z={col_z}, minus={col_minus}, i={col_i}), sparse=(X={sparse_x}, Z={sparse_z}, minus={sparse_minus}, i={sparse_i})"
        );

        col_only.h(&[QubitId(0)]);
        sparse.h(&[QubitId(0)]);
        let col_minus = col_only.stab_signs_minus[0] & 1 != 0;
        let col_i = col_only.stab_signs_i[0] & 1 != 0;
        let col_x = col_only.stab_col_x[0] & 1 != 0;
        let col_z = col_only.stab_col_z[0] & 1 != 0;
        let sparse_minus = sparse.stabs.signs_minus.contains(0);
        let sparse_i = sparse.stabs.signs_i.contains(0);
        let sparse_x = sparse.stabs.row_x[0].contains(0);
        let sparse_z = sparse.stabs.row_z[0].contains(0);
        println!(
            "After H (final): col=(X={col_x}, Z={col_z}, minus={col_minus}, i={col_i}), sparse=(X={sparse_x}, Z={sparse_z}, minus={sparse_minus}, i={sparse_i})"
        );

        // Should be Y = iXZ
        assert!(col_x, "Expected X=true");
        assert!(col_z, "Expected Z=true");
        assert_eq!(col_minus, sparse_minus, "minus mismatch after SXdg");
        assert_eq!(col_i, sparse_i, "i mismatch after SXdg");
    }

    #[test]
    fn test_col_only_vs_sparse_minimal() {
        // Find minimal failing circuit
        use crate::stabilizer_test_utils::CliffordGate;

        // Minimal circuit that causes divergence - gates 0-14 from the failing circuit
        let gates = vec![
            CliffordGate::SXdg(7),
            CliffordGate::CY(2, 3),
            CliffordGate::CX(4, 1),
            CliffordGate::Y(0),
            CliffordGate::Y(2),
            CliffordGate::SYdg(2),
            CliffordGate::X(0),
            CliffordGate::Sdg(3),
            CliffordGate::CY(1, 2),
            CliffordGate::CZ(1, 6),
            CliffordGate::CZ(7, 0),
            CliffordGate::SX(0),
            CliffordGate::Z(1),
            CliffordGate::Sdg(5),
            CliffordGate::Sdg(2), // Gate 14 - this is when it diverges
        ];

        let mut col_only: DenseStabColOnly = DenseStabColOnly::with_seed(8, 42);
        let mut sparse: SparseStab = SparseStab::with_seed(8, 42);

        // Apply gates up to but not including gate 11 (SX(0)), tracing divergence
        for (i, gate) in gates.iter().take(11).enumerate() {
            crate::stabilizer_test_utils::apply_circuit(&mut col_only, &[*gate]);
            crate::stabilizer_test_utils::apply_circuit(&mut sparse, &[*gate]);

            // Check for any sign mismatches
            for g in 0..8 {
                let col_minus = col_only.stab_signs_minus[0] & (1u64 << g) != 0;
                let sparse_minus = sparse.stabs.signs_minus.contains(g);
                if col_minus != sparse_minus {
                    println!(
                        "DIVERGENCE at gate {i}: {gate:?} - S[{g}] col_only.minus={col_minus}, sparse.minus={sparse_minus}"
                    );
                }
            }
        }

        // Print S[7] state before SX(0)
        println!("Before SX(0):");
        let col_minus = col_only.stab_signs_minus[0] & (1u64 << 7) != 0;
        let sparse_minus = sparse.stabs.signs_minus.contains(7);
        let col_x7: Vec<usize> = (0..8)
            .filter(|&q| col_only.stab_col_x[q] & (1u64 << 7) != 0)
            .collect();
        let col_z7: Vec<usize> = (0..8)
            .filter(|&q| col_only.stab_col_z[q] & (1u64 << 7) != 0)
            .collect();
        let sparse_x7: Vec<usize> = sparse.stabs.row_x[7].iter().collect();
        let sparse_z7: Vec<usize> = sparse.stabs.row_z[7].iter().collect();
        println!("  S[7] col_only: minus={col_minus}, X={col_x7:?}, Z={col_z7:?}");
        println!("  S[7] sparse:   minus={sparse_minus}, X={sparse_x7:?}, Z={sparse_z7:?}");

        // Manually apply SX(0) = H(0), SZ(0), H(0)
        println!("\nApplying SX(0) step-by-step:");

        // H(0)
        col_only.h(&[QubitId(0)]);
        sparse.h(&[QubitId(0)]);
        let col_minus = col_only.stab_signs_minus[0] & (1u64 << 7) != 0;
        let sparse_minus = sparse.stabs.signs_minus.contains(7);
        let col_x7: Vec<usize> = (0..8)
            .filter(|&q| col_only.stab_col_x[q] & (1u64 << 7) != 0)
            .collect();
        let col_z7: Vec<usize> = (0..8)
            .filter(|&q| col_only.stab_col_z[q] & (1u64 << 7) != 0)
            .collect();
        let sparse_x7: Vec<usize> = sparse.stabs.row_x[7].iter().collect();
        let sparse_z7: Vec<usize> = sparse.stabs.row_z[7].iter().collect();
        println!("After H(0):");
        println!("  S[7] col_only: minus={col_minus}, X={col_x7:?}, Z={col_z7:?}");
        println!("  S[7] sparse:   minus={sparse_minus}, X={sparse_x7:?}, Z={sparse_z7:?}");

        // SZ(0)
        col_only.sz(&[QubitId(0)]);
        sparse.sz(&[QubitId(0)]);
        let col_minus = col_only.stab_signs_minus[0] & (1u64 << 7) != 0;
        let sparse_minus = sparse.stabs.signs_minus.contains(7);
        let col_x7: Vec<usize> = (0..8)
            .filter(|&q| col_only.stab_col_x[q] & (1u64 << 7) != 0)
            .collect();
        let col_z7: Vec<usize> = (0..8)
            .filter(|&q| col_only.stab_col_z[q] & (1u64 << 7) != 0)
            .collect();
        let sparse_x7: Vec<usize> = sparse.stabs.row_x[7].iter().collect();
        let sparse_z7: Vec<usize> = sparse.stabs.row_z[7].iter().collect();
        println!("After SZ(0):");
        println!("  S[7] col_only: minus={col_minus}, X={col_x7:?}, Z={col_z7:?}");
        println!("  S[7] sparse:   minus={sparse_minus}, X={sparse_x7:?}, Z={sparse_z7:?}");

        // H(0)
        col_only.h(&[QubitId(0)]);
        sparse.h(&[QubitId(0)]);
        let col_minus = col_only.stab_signs_minus[0] & (1u64 << 7) != 0;
        let sparse_minus = sparse.stabs.signs_minus.contains(7);
        let col_x7: Vec<usize> = (0..8)
            .filter(|&q| col_only.stab_col_x[q] & (1u64 << 7) != 0)
            .collect();
        let col_z7: Vec<usize> = (0..8)
            .filter(|&q| col_only.stab_col_z[q] & (1u64 << 7) != 0)
            .collect();
        let sparse_x7: Vec<usize> = sparse.stabs.row_x[7].iter().collect();
        let sparse_z7: Vec<usize> = sparse.stabs.row_z[7].iter().collect();
        println!("After H(0) (final):");
        println!("  S[7] col_only: minus={col_minus}, X={col_x7:?}, Z={col_z7:?}");
        println!("  S[7] sparse:   minus={sparse_minus}, X={sparse_x7:?}, Z={sparse_z7:?}");

        // Apply remaining gates
        for gate in gates.iter().skip(12) {
            crate::stabilizer_test_utils::apply_circuit(&mut col_only, &[*gate]);
            crate::stabilizer_test_utils::apply_circuit(&mut sparse, &[*gate]);
        }

        // Test Q3 deterministic measurement - should both return false but col_only returns true
        let r1 = col_only.clone().mz_forced(3, false);
        let r2 = sparse.clone().mz_forced(3, false);
        println!(
            "Q3: col_only=(det={}, out={}), sparse=(det={}, out={})",
            r1.is_deterministic, r1.outcome, r2.is_deterministic, r2.outcome
        );

        // Compare the full state between implementations
        let words_per_col = 8_usize.div_ceil(64);

        println!("\n=== STABILIZER COMPARISON ===");
        for g in 0..8 {
            let g_word = g / 64;
            let g_bit = 1u64 << (g % 64);

            // Get DenseStabColOnly stabilizer
            let col_minus = col_only.stab_signs_minus[g_word] & g_bit != 0;
            let mut col_x = Vec::new();
            let mut col_z = Vec::new();
            for q in 0..8 {
                let q_base = q * words_per_col;
                if col_only.stab_col_x[q_base + g_word] & g_bit != 0 {
                    col_x.push(q);
                }
                if col_only.stab_col_z[q_base + g_word] & g_bit != 0 {
                    col_z.push(q);
                }
            }

            // Get SparseStab stabilizer
            let sparse_minus = sparse.stabs.signs_minus.contains(g);
            let sparse_x: Vec<usize> = sparse.stabs.row_x[g].iter().collect();
            let sparse_z: Vec<usize> = sparse.stabs.row_z[g].iter().collect();

            let match_str = if col_minus == sparse_minus && col_x == sparse_x && col_z == sparse_z {
                "OK"
            } else {
                "MISMATCH"
            };

            println!(
                "  S[{g}]: col_only=(minus={col_minus}, X={col_x:?}, Z={col_z:?}), sparse=(minus={sparse_minus}, X={sparse_x:?}, Z={sparse_z:?}) {match_str}"
            );
        }

        println!("\n=== DESTABILIZER COMPARISON ===");
        for g in 0..8 {
            let g_word = g / 64;
            let g_bit = 1u64 << (g % 64);

            // Get DenseStabColOnly destabilizer
            let mut col_x = Vec::new();
            let mut col_z = Vec::new();
            for q in 0..8 {
                let q_base = q * words_per_col;
                if col_only.destab_col_x[q_base + g_word] & g_bit != 0 {
                    col_x.push(q);
                }
                if col_only.destab_col_z[q_base + g_word] & g_bit != 0 {
                    col_z.push(q);
                }
            }

            // Get SparseStab destabilizer
            let sparse_x: Vec<usize> = sparse.destabs.row_x[g].iter().collect();
            let sparse_z: Vec<usize> = sparse.destabs.row_z[g].iter().collect();

            let match_str = if col_x == sparse_x && col_z == sparse_z {
                "OK"
            } else {
                "MISMATCH"
            };

            println!(
                "  D[{g}]: col_only=(X={col_x:?}, Z={col_z:?}), sparse=(X={sparse_x:?}, Z={sparse_z:?}) {match_str}"
            );
        }

        // Which destabilizers have X on Q3?
        let qubit = 3;
        println!("\n=== DESTABS WITH X ON Q{qubit} ===");
        let col_base = qubit * words_per_col;
        let destab_mask = col_only.destab_col_x[col_base];
        let mut col_destab_ids: Vec<usize> = Vec::new();
        let mut mask = destab_mask;
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            col_destab_ids.push(bit);
            mask &= mask - 1;
        }
        let sparse_destab_ids: Vec<usize> = sparse.destabs.col_x[qubit].iter().collect();
        println!("  col_only destab_ids: {col_destab_ids:?}");
        println!("  sparse destab_ids: {sparse_destab_ids:?}");

        assert_eq!(
            r1.is_deterministic, r2.is_deterministic,
            "Q3 determinism mismatch"
        );
        assert_eq!(r1.outcome, r2.outcome, "Q3 outcome mismatch");
    }

    #[test]
    fn test_col_only_simple_probability() {
        use crate::DensityMatrix;
        use crate::stabilizer_test_utils::calculate_basis_probability;

        // Create 2-qubit simulator, apply H on qubit 0
        // After H(0), state is (|00> + |10>)/sqrt(2)
        // - basis_state 0 = |00> (q0=0, q1=0) -> probability 0.5
        // - basis_state 1 = |10> (q0=1, q1=0) -> probability 0.5
        // - basis_state 2 = |01> (q0=0, q1=1) -> probability 0
        // - basis_state 3 = |11> (q0=1, q1=1) -> probability 0
        let mut sim: DenseStabColOnly = DenseStabColOnly::with_seed(2, 42);
        sim.h(&[QubitId(0)]);

        // Compare with DensityMatrix
        let mut dm = DensityMatrix::new(2);
        dm.h(&[QubitId(0)]);

        for basis_state in 0..4 {
            let stab_prob = calculate_basis_probability(&sim, basis_state, 2);
            let dm_prob = dm.probability(basis_state);
            println!("basis_state {basis_state}: stabilizer={stab_prob}, density_matrix={dm_prob}");
            assert!(
                (stab_prob - dm_prob).abs() < 1e-10,
                "Mismatch for basis_state {basis_state}: stabilizer={stab_prob}, dm={dm_prob}"
            );
        }
    }

    #[test]
    fn test_col_only_full_stabilizer_suite() {
        let mut sim: DenseStabColOnly = DenseStabColOnly::with_seed(8, 42);
        run_full_stabilizer_test_suite(&mut sim, 8);
    }

    #[test]
    fn test_row_only_full_stabilizer_suite() {
        let mut sim: DenseStabRowOnly = DenseStabRowOnly::with_seed(8, 42);
        run_full_stabilizer_test_suite(&mut sim, 8);
    }

    #[test]
    fn test_sparse_col_only_full_stabilizer_suite() {
        let mut sim: SparseColOnly = SparseColOnly::with_seed(8, 42);
        run_full_stabilizer_test_suite(&mut sim, 8);
    }

    #[test]
    fn test_sparse_row_only_full_stabilizer_suite() {
        let mut sim: SparseRowOnly = SparseRowOnly::with_seed(8, 42);
        run_full_stabilizer_test_suite(&mut sim, 8);
    }

    /// Generate a random Clifford circuit using only H, SZ, CX (the universal generators)
    /// with mid-circuit forced measurements and init |0> operations, then compare
    /// the variant simulator against `SparseStab` (reference).
    fn mid_circuit_meas_test<S: CliffordGateable + ForcedMeasurement>(
        variant: &mut S,
        reference: &mut SparseStab,
        num_qubits: usize,
        num_gates: usize,
        seed: u64,
    ) {
        use pecos_random::{PecosRng, RngExt};
        let mut rng = PecosRng::seed_from_u64(seed);

        for gate_idx in 0..num_gates {
            let gate_type: u8 = rng.random_range(0..10);
            let q0 = rng.random_range(0..num_qubits);

            match gate_type {
                0 => {
                    variant.h(&[QubitId(q0)]);
                    reference.h(&[QubitId(q0)]);
                }
                1 => {
                    variant.sz(&[QubitId(q0)]);
                    reference.sz(&[QubitId(q0)]);
                }
                2..=4 if num_qubits >= 2 => {
                    let mut q1 = rng.random_range(0..num_qubits);
                    while q1 == q0 {
                        q1 = rng.random_range(0..num_qubits);
                    }
                    variant.cx(&[(QubitId(q0), QubitId(q1))]);
                    reference.cx(&[(QubitId(q0), QubitId(q1))]);
                }
                5..=7 => {
                    // Forced measurement (mid-circuit)
                    let forced: bool = rng.random();
                    let rv = variant.mz_forced(q0, forced);
                    let rr = reference.mz_forced(q0, forced);
                    assert_eq!(
                        rv.outcome, rr.outcome,
                        "seed {seed} gate {gate_idx}: mz_forced({q0}, {forced}) outcome mismatch"
                    );
                    assert_eq!(
                        rv.is_deterministic, rr.is_deterministic,
                        "seed {seed} gate {gate_idx}: mz_forced({q0}, {forced}) determinism mismatch"
                    );
                }
                8 => {
                    // Init |0> = mz_forced(false) + conditional X
                    let rv = variant.mz_forced(q0, false);
                    let rr = reference.mz_forced(q0, false);
                    assert_eq!(
                        rv.outcome, rr.outcome,
                        "seed {seed} gate {gate_idx}: init|0> mz_forced({q0}) outcome mismatch"
                    );
                    if rv.outcome {
                        variant.x(&[QubitId(q0)]);
                        reference.x(&[QubitId(q0)]);
                    }
                }
                _ => {
                    // SZ dagger = SZ^3
                    variant.sz(&[QubitId(q0)]);
                    variant.sz(&[QubitId(q0)]);
                    variant.sz(&[QubitId(q0)]);
                    reference.szdg(&[QubitId(q0)]);
                }
            }
        }

        // Final measurement of all qubits
        for q in 0..num_qubits {
            let forced: bool = PecosRng::seed_from_u64(seed + 1000 + q as u64).random();
            let rv = variant.mz_forced(q, forced);
            let rr = reference.mz_forced(q, forced);
            assert_eq!(
                rv.outcome, rr.outcome,
                "seed {seed}: final mz_forced({q}, {forced}) outcome mismatch"
            );
            assert_eq!(
                rv.is_deterministic, rr.is_deterministic,
                "seed {seed}: final mz_forced({q}, {forced}) determinism mismatch"
            );
        }
    }

    #[test]
    fn test_col_only_mid_circuit_meas() {
        use pecos_random::PecosRng;
        let num_qubits = 10;
        for i in 0..200 {
            let seed = 50_000 + i;
            let mut variant: DenseStabColOnly<PecosRng> = DenseStabColOnly::new(num_qubits);
            let mut reference = SparseStab::new(num_qubits);
            mid_circuit_meas_test(&mut variant, &mut reference, num_qubits, 50, seed);
        }
    }

    #[test]
    fn test_row_only_mid_circuit_meas() {
        use pecos_random::PecosRng;
        let num_qubits = 10;
        for i in 0..200 {
            let seed = 60_000 + i;
            let mut variant: DenseStabRowOnly<PecosRng> = DenseStabRowOnly::new(num_qubits);
            let mut reference = SparseStab::new(num_qubits);
            mid_circuit_meas_test(&mut variant, &mut reference, num_qubits, 50, seed);
        }
    }

    #[test]
    fn test_sparse_col_only_mid_circuit_meas() {
        let num_qubits = 10;
        for i in 0..200 {
            let seed = 70_000 + i;
            let mut variant = SparseColOnly::new(num_qubits);
            let mut reference = SparseStab::new(num_qubits);
            mid_circuit_meas_test(&mut variant, &mut reference, num_qubits, 50, seed);
        }
    }

    #[test]
    fn test_sparse_row_only_mid_circuit_meas() {
        let num_qubits = 10;
        for i in 0..200 {
            let seed = 80_000 + i;
            let mut variant = SparseRowOnly::new(num_qubits);
            let mut reference = SparseStab::new(num_qubits);
            mid_circuit_meas_test(&mut variant, &mut reference, num_qubits, 50, seed);
        }
    }
}
