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

//! Dense bit-matrix stabilizer simulator using Data-Oriented Design (DOD) principles.
//!
//! [`DenseStab`] is optimized for small-to-medium qubit counts (up to ~200 qubits)
//! where the O(n^2) memory usage is acceptable. It provides significant speedups
//! over sparse representations for typical stabilizer circuits.
//!
//! # Performance Characteristics
//!
//! - **d=3-7 (17-97 qubits)**: 2-3x faster than sparse `BitSet` representation
//! - **d=9-11 (161-241 qubits)**: 1.5-2x faster than sparse
//! - **d>=13 (337+ qubits)**: Similar to sparse (dense overhead dominates)
//!
//! # DOD Optimizations
//!
//! - **Dual representation**: Both row-wise and column-wise bit matrices for
//!   fast access patterns depending on the operation
//! - **Loop unrolling**: Manual unrolling for small word counts (1-4 words)
//! - **Batched operations**: CX gates batch stabilizer/destabilizer updates
//! - **AVX2 SIMD**: Explicit vectorization for large qubit counts (>=512 qubits)
//! - **Scratch buffers**: Pre-allocated buffers to avoid allocation in hot paths
//!
//! # Example
//!
//! ```
//! use pecos_simulators::{CliffordGateable, DenseStab};
//! use pecos_core::QubitId;
//!
//! // Use seeded RNG for determinism
//! let mut sim: DenseStab = DenseStab::with_seed(2, 42);
//! sim.h(&[QubitId(0)]);
//! sim.cx(&[(QubitId(0), QubitId(1))]);
//! let results = sim.mz(&[QubitId(0), QubitId(1)]);
//! // Bell state: both qubits measure the same
//! assert_eq!(results[0].outcome, results[1].outcome);
//! ```

use crate::{CliffordGateable, MeasurementResult, QuantumSimulator, StabilizerTableauSimulator};
use core::fmt::Debug;
use pecos_core::{QubitId, RngManageable};
use pecos_random::rng_ext::RngProbabilityExt;
use pecos_random::{PecosRng, Rng, SeedableRng};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_xor_si256};

// ========== Bit manipulation helpers (free functions) ==========

#[inline(always)]
fn set_bit_row(data: &mut [u64], words_per_row: usize, row: usize, qubit: usize) {
    let word_idx = row * words_per_row + qubit / 64;
    data[word_idx] |= 1u64 << (qubit % 64);
}

#[inline(always)]
fn set_bit_col(data: &mut [u64], words_per_col: usize, qubit: usize, row: usize) {
    let word_idx = qubit * words_per_col + row / 64;
    data[word_idx] |= 1u64 << (row % 64);
}

#[inline(always)]
fn clear_bit_col(data: &mut [u64], words_per_col: usize, qubit: usize, row: usize) {
    let word_idx = qubit * words_per_col + row / 64;
    data[word_idx] &= !(1u64 << (row % 64));
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

/// XOR column `col_a` into column `col_b`.
/// Uses explicit SIMD for larger sizes where it provides benefit.
#[inline(always)]
fn xor_cols(data: &mut [u64], words_per_col: usize, col_a: usize, col_b: usize) {
    let base_a = col_a * words_per_col;
    let base_b = col_b * words_per_col;
    debug_assert!(base_a + words_per_col <= data.len());
    debug_assert!(base_b + words_per_col <= data.len());

    // Manual unrolling for common small sizes to avoid loop overhead
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
            3 => {
                let a0 = *data.get_unchecked(base_a);
                let a1 = *data.get_unchecked(base_a + 1);
                let a2 = *data.get_unchecked(base_a + 2);
                *data.get_unchecked_mut(base_b) ^= a0;
                *data.get_unchecked_mut(base_b + 1) ^= a1;
                *data.get_unchecked_mut(base_b + 2) ^= a2;
            }
            4 => {
                let a0 = *data.get_unchecked(base_a);
                let a1 = *data.get_unchecked(base_a + 1);
                let a2 = *data.get_unchecked(base_a + 2);
                let a3 = *data.get_unchecked(base_a + 3);
                *data.get_unchecked_mut(base_b) ^= a0;
                *data.get_unchecked_mut(base_b + 1) ^= a1;
                *data.get_unchecked_mut(base_b + 2) ^= a2;
                *data.get_unchecked_mut(base_b + 3) ^= a3;
            }
            _ => {
                // Use SIMD for larger sizes
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                if words_per_col >= 8 {
                    let chunks = words_per_col / 4;
                    let remainder = words_per_col % 4;

                    let ptr_a = data.as_ptr().add(base_a) as *const __m256i;
                    let ptr_b = data.as_mut_ptr().add(base_b) as *mut __m256i;

                    for i in 0..chunks {
                        let a = _mm256_loadu_si256(ptr_a.add(i));
                        let b = _mm256_loadu_si256(ptr_b.add(i));
                        let result = _mm256_xor_si256(a, b);
                        _mm256_storeu_si256(ptr_b.add(i), result);
                    }

                    let start = chunks * 4;
                    for w in 0..remainder {
                        let a = *data.get_unchecked(base_a + start + w);
                        *data.get_unchecked_mut(base_b + start + w) ^= a;
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

/// XOR row `row_a` into row `row_b`.
/// Uses explicit SIMD for larger sizes where it provides benefit.
#[inline(always)]
fn xor_rows(data: &mut [u64], words_per_row: usize, row_a: usize, row_b: usize) {
    let base_a = row_a * words_per_row;
    let base_b = row_b * words_per_row;
    debug_assert!(base_a + words_per_row <= data.len());
    debug_assert!(base_b + words_per_row <= data.len());

    // Manual unrolling for common small sizes to avoid loop overhead
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
            3 => {
                let a0 = *data.get_unchecked(base_a);
                let a1 = *data.get_unchecked(base_a + 1);
                let a2 = *data.get_unchecked(base_a + 2);
                *data.get_unchecked_mut(base_b) ^= a0;
                *data.get_unchecked_mut(base_b + 1) ^= a1;
                *data.get_unchecked_mut(base_b + 2) ^= a2;
            }
            4 => {
                let a0 = *data.get_unchecked(base_a);
                let a1 = *data.get_unchecked(base_a + 1);
                let a2 = *data.get_unchecked(base_a + 2);
                let a3 = *data.get_unchecked(base_a + 3);
                *data.get_unchecked_mut(base_b) ^= a0;
                *data.get_unchecked_mut(base_b + 1) ^= a1;
                *data.get_unchecked_mut(base_b + 2) ^= a2;
                *data.get_unchecked_mut(base_b + 3) ^= a3;
            }
            _ => {
                // Use SIMD for larger sizes
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                if words_per_row >= 8 {
                    let chunks = words_per_row / 4;
                    let remainder = words_per_row % 4;

                    let ptr_a = data.as_ptr().add(base_a) as *const __m256i;
                    let ptr_b = data.as_mut_ptr().add(base_b) as *mut __m256i;

                    for i in 0..chunks {
                        let a = _mm256_loadu_si256(ptr_a.add(i));
                        let b = _mm256_loadu_si256(ptr_b.add(i));
                        let result = _mm256_xor_si256(a, b);
                        _mm256_storeu_si256(ptr_b.add(i), result);
                    }

                    let start = chunks * 4;
                    for w in 0..remainder {
                        let a = *data.get_unchecked(base_a + start + w);
                        *data.get_unchecked_mut(base_b + start + w) ^= a;
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
    debug_assert!(base + words_per_row <= data.len());
    let mut count = 0;
    for w in 0..words_per_row {
        // SAFETY: bounds are guaranteed by caller
        unsafe {
            count += data.get_unchecked(base + w).count_ones() as usize;
        }
    }
    count
}

#[inline(always)]
fn col_is_empty(data: &[u64], words_per_col: usize, qubit: usize) -> bool {
    let base = qubit * words_per_col;
    debug_assert!(base + words_per_col <= data.len());
    for w in 0..words_per_col {
        // SAFETY: bounds are guaranteed by caller
        unsafe {
            if *data.get_unchecked(base + w) != 0 {
                return false;
            }
        }
    }
    true
}

/// Dense bit-matrix stabilizer simulator.
#[derive(Clone, Debug)]
pub struct DenseStab<R: Rng = PecosRng> {
    num_qubits: usize,
    words_per_row: usize,
    words_per_col: usize,

    // Stabilizers - row-wise: row[row_idx * words_per_row + word]
    stab_row_x: Vec<u64>,
    stab_row_z: Vec<u64>,

    // Stabilizers - column-wise: col[qubit * words_per_col + word]
    stab_col_x: Vec<u64>,
    stab_col_z: Vec<u64>,

    // Destabilizers - row-wise
    destab_row_x: Vec<u64>,
    destab_row_z: Vec<u64>,

    // Destabilizers - column-wise
    destab_col_x: Vec<u64>,
    destab_col_z: Vec<u64>,

    // Signs: one bit per generator
    stab_signs_minus: Vec<u64>,
    stab_signs_i: Vec<u64>,
    destab_signs_minus: Vec<u64>,
    destab_signs_i: Vec<u64>,

    // Scratch buffers to avoid allocations in hot paths
    scratch_row: Vec<u64>,
    scratch_qubits: Vec<usize>,

    rng: R,
}

impl DenseStab<PecosRng> {
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

impl<R: SeedableRng + Rng + Debug> DenseStab<R> {
    #[inline]
    #[must_use]
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        let words_per_row = num_qubits.div_ceil(64);
        let words_per_col = num_qubits.div_ceil(64);

        let row_size = num_qubits * words_per_row;
        let col_size = num_qubits * words_per_col;
        let sign_size = words_per_col;

        let mut sim = Self {
            num_qubits,
            words_per_row,
            words_per_col,
            stab_row_x: vec![0; row_size],
            stab_row_z: vec![0; row_size],
            stab_col_x: vec![0; col_size],
            stab_col_z: vec![0; col_size],
            destab_row_x: vec![0; row_size],
            destab_row_z: vec![0; row_size],
            destab_col_x: vec![0; col_size],
            destab_col_z: vec![0; col_size],
            stab_signs_minus: vec![0; sign_size],
            stab_signs_i: vec![0; sign_size],
            destab_signs_minus: vec![0; sign_size],
            destab_signs_i: vec![0; sign_size],
            scratch_row: vec![0; words_per_row],
            scratch_qubits: Vec::with_capacity(num_qubits),
            rng,
        };
        sim.init_state();
        sim
    }

    /// Initialize to |0...0> state.
    fn init_state(&mut self) {
        self.stab_row_x.fill(0);
        self.stab_row_z.fill(0);
        self.stab_col_x.fill(0);
        self.stab_col_z.fill(0);
        self.destab_row_x.fill(0);
        self.destab_row_z.fill(0);
        self.destab_col_x.fill(0);
        self.destab_col_z.fill(0);
        self.stab_signs_minus.fill(0);
        self.stab_signs_i.fill(0);
        self.destab_signs_minus.fill(0);
        self.destab_signs_i.fill(0);

        // Initialize: stab[i] = Z_i, destab[i] = X_i
        for i in 0..self.num_qubits {
            set_bit_row(&mut self.stab_row_z, self.words_per_row, i, i);
            set_bit_col(&mut self.stab_col_z, self.words_per_col, i, i);
            set_bit_row(&mut self.destab_row_x, self.words_per_row, i, i);
            set_bit_col(&mut self.destab_col_x, self.words_per_col, i, i);
        }
    }

    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Apply CX to both stabs and destabs.
    fn apply_cx(&mut self, control: usize, target: usize) {
        let words_per_row = self.words_per_row;
        let words_per_col = self.words_per_col;

        // Apply to stabs
        Self::apply_cx_to_gens(
            &mut self.stab_row_x,
            &mut self.stab_row_z,
            &mut self.stab_col_x,
            &mut self.stab_col_z,
            words_per_row,
            words_per_col,
            control,
            target,
        );

        // Apply to destabs
        Self::apply_cx_to_gens(
            &mut self.destab_row_x,
            &mut self.destab_row_z,
            &mut self.destab_col_x,
            &mut self.destab_col_z,
            words_per_row,
            words_per_col,
            control,
            target,
        );
    }

    #[allow(clippy::too_many_arguments)] // tableau bitwise ops need separate X/Z arrays + dimensions
    #[inline(always)]
    fn apply_cx_to_gens(
        row_x: &mut [u64],
        row_z: &mut [u64],
        col_x: &mut [u64],
        col_z: &mut [u64],
        words_per_row: usize,
        words_per_col: usize,
        control: usize,
        target: usize,
    ) {
        let target_word = target / 64;
        let target_bit = target % 64;
        let target_mask = 1u64 << target_bit;
        let control_word = control / 64;
        let control_bit = control % 64;
        let control_mask = 1u64 << control_bit;

        // For each generator with X on control, toggle X on target
        let col_base = control * words_per_col;
        debug_assert!(col_base + words_per_col <= col_x.len());
        for w in 0..words_per_col {
            let mut gen_mask = unsafe { *col_x.get_unchecked(col_base + w) };
            while gen_mask != 0 {
                let g = w * 64 + gen_mask.trailing_zeros() as usize;
                debug_assert!(g * words_per_row + target_word < row_x.len());
                unsafe {
                    *row_x.get_unchecked_mut(g * words_per_row + target_word) ^= target_mask;
                }
                gen_mask &= gen_mask - 1;
            }
        }

        // XOR columns: col_x[target] ^= col_x[control]
        xor_cols(col_x, words_per_col, control, target);

        // For each generator with Z on target, toggle Z on control
        let col_base = target * words_per_col;
        debug_assert!(col_base + words_per_col <= col_z.len());
        for w in 0..words_per_col {
            let mut gen_mask = unsafe { *col_z.get_unchecked(col_base + w) };
            while gen_mask != 0 {
                let g = w * 64 + gen_mask.trailing_zeros() as usize;
                debug_assert!(g * words_per_row + control_word < row_z.len());
                unsafe {
                    *row_z.get_unchecked_mut(g * words_per_row + control_word) ^= control_mask;
                }
                gen_mask &= gen_mask - 1;
            }
        }

        // XOR columns: col_z[control] ^= col_z[target]
        xor_cols(col_z, words_per_col, target, control);
    }

    fn apply_h(&mut self, qubit: usize) {
        let words_per_row = self.words_per_row;
        let words_per_col = self.words_per_col;
        let num_qubits = self.num_qubits;

        // Apply to stabs
        Self::apply_h_to_gens(
            &mut self.stab_row_x,
            &mut self.stab_row_z,
            &mut self.stab_col_x,
            &mut self.stab_col_z,
            &mut self.stab_signs_minus,
            words_per_row,
            words_per_col,
            num_qubits,
            qubit,
        );

        // Apply to destabs
        Self::apply_h_to_gens(
            &mut self.destab_row_x,
            &mut self.destab_row_z,
            &mut self.destab_col_x,
            &mut self.destab_col_z,
            &mut self.destab_signs_minus,
            words_per_row,
            words_per_col,
            num_qubits,
            qubit,
        );
    }

    #[allow(clippy::too_many_arguments)] // tableau bitwise ops need separate X/Z arrays + dimensions
    #[inline(always)]
    fn apply_h_to_gens(
        row_x: &mut [u64],
        row_z: &mut [u64],
        col_x: &mut [u64],
        col_z: &mut [u64],
        signs_minus: &mut [u64],
        words_per_row: usize,
        words_per_col: usize,
        num_rows: usize,
        qubit: usize,
    ) {
        // H: X -> Z, Z -> X, Y -> -Y
        let col_base = qubit * words_per_col;
        debug_assert!(col_base + words_per_col <= col_x.len());
        debug_assert!(col_base + words_per_col <= col_z.len());
        debug_assert!(words_per_col <= signs_minus.len());

        // Update phases for Y -> -Y and swap X/Z columns in one pass
        for w in 0..words_per_col {
            unsafe {
                let cx = *col_x.get_unchecked(col_base + w);
                let cz = *col_z.get_unchecked(col_base + w);
                *signs_minus.get_unchecked_mut(w) ^= cx & cz;
                *col_x.get_unchecked_mut(col_base + w) = cz;
                *col_z.get_unchecked_mut(col_base + w) = cx;
            }
        }

        // Swap X and Z in rows using XOR swap
        let qubit_word = qubit / 64;
        let mask = 1u64 << (qubit % 64);
        for g in 0..num_rows {
            let row_idx = g * words_per_row + qubit_word;
            debug_assert!(row_idx < row_x.len());
            debug_assert!(row_idx < row_z.len());
            unsafe {
                let rx = *row_x.get_unchecked(row_idx);
                let rz = *row_z.get_unchecked(row_idx);
                let diff = (rx ^ rz) & mask;
                *row_x.get_unchecked_mut(row_idx) = rx ^ diff;
                *row_z.get_unchecked_mut(row_idx) = rz ^ diff;
            }
        }
    }

    fn apply_s(&mut self, qubit: usize) {
        let words_per_row = self.words_per_row;
        let words_per_col = self.words_per_col;
        let num_qubits = self.num_qubits;

        Self::apply_s_to_gens(
            &mut self.stab_row_x,
            &mut self.stab_row_z,
            &mut self.stab_col_x,
            &mut self.stab_col_z,
            &mut self.stab_signs_minus,
            &mut self.stab_signs_i,
            words_per_row,
            words_per_col,
            num_qubits,
            qubit,
        );

        Self::apply_s_to_gens(
            &mut self.destab_row_x,
            &mut self.destab_row_z,
            &mut self.destab_col_x,
            &mut self.destab_col_z,
            &mut self.destab_signs_minus,
            &mut self.destab_signs_i,
            words_per_row,
            words_per_col,
            num_qubits,
            qubit,
        );
    }

    #[allow(clippy::too_many_arguments)] // tableau bitwise ops need separate X/Z arrays + dimensions
    #[inline(always)]
    fn apply_s_to_gens(
        row_x: &mut [u64],
        row_z: &mut [u64],
        col_x: &mut [u64],
        col_z: &mut [u64],
        signs_minus: &mut [u64],
        signs_i: &mut [u64],
        words_per_row: usize,
        words_per_col: usize,
        num_rows: usize,
        qubit: usize,
    ) {
        // S: X -> Y = iXZ, Z -> Z
        // For generators with X on this qubit, multiply phase by i:
        // - i * i = -1, so toggle minus for generators with both i and X
        // - Toggle i for all X generators
        // - Toggle Z for all X generators
        let col_base = qubit * words_per_col;
        let qubit_word = qubit / 64;
        let qubit_bit = qubit % 64;
        let mask = 1u64 << qubit_bit;
        debug_assert!(col_base + words_per_col <= col_x.len());
        debug_assert!(col_base + words_per_col <= col_z.len());
        debug_assert!(words_per_col <= signs_minus.len());
        debug_assert!(words_per_col <= signs_i.len());

        for w in 0..words_per_col {
            unsafe {
                let x_gens = *col_x.get_unchecked(col_base + w);
                // i * i = -1: toggle minus for generators with both i and X
                *signs_minus.get_unchecked_mut(w) ^= x_gens & *signs_i.get_unchecked(w);
                // Toggle i for all X generators
                *signs_i.get_unchecked_mut(w) ^= x_gens;
                // Toggle Z for all X generators
                *col_z.get_unchecked_mut(col_base + w) ^= x_gens;
            }
        }

        // Update rows - branchless: XOR with (x_bit << qubit_bit)
        for g in 0..num_rows {
            let row_idx = g * words_per_row + qubit_word;
            debug_assert!(row_idx < row_x.len());
            debug_assert!(row_idx < row_z.len());
            unsafe {
                let rx = *row_x.get_unchecked(row_idx);
                let x_bit_at_pos = rx & mask;
                *row_z.get_unchecked_mut(row_idx) ^= x_bit_at_pos;
            }
        }
    }

    fn deterministic_meas(&mut self, qubit: usize) -> MeasurementResult {
        // Outcome is computed by considering destabilizers with X on the qubit
        let col_base = qubit * self.words_per_col;
        let words_per_row = self.words_per_row;
        let words_per_col = self.words_per_col;

        // Count intersection of destab_col_x[q] with stab_signs_minus and stab_signs_i
        let mut num_minuses: usize = 0;
        let mut num_is: usize = 0;
        for w in 0..words_per_col {
            let destab_mask = self.destab_col_x[col_base + w];
            num_minuses += (destab_mask & self.stab_signs_minus[w]).count_ones() as usize;
            num_is += (destab_mask & self.stab_signs_i[w]).count_ones() as usize;
        }

        // Use scratch buffer for cumulative_x (avoids allocation)
        self.scratch_row.fill(0);

        // For each destabilizer with X on q, count Z-X overlaps with accumulated X
        for w in 0..words_per_col {
            let mut mask = self.destab_col_x[col_base + w];
            while mask != 0 {
                let destab_id = w * 64 + mask.trailing_zeros() as usize;
                let row_base = destab_id * words_per_row;

                // Count overlap of stab_row_z[destab_id] with cumulative_x and XOR into scratch
                for ww in 0..words_per_row {
                    num_minuses += (self.stab_row_z[row_base + ww] & self.scratch_row[ww])
                        .count_ones() as usize;
                    self.scratch_row[ww] ^= self.stab_row_x[row_base + ww];
                }

                mask &= mask - 1;
            }
        }

        // Add i phase contribution
        if num_is & 3 != 0 {
            num_minuses += 1;
        }

        let outcome = num_minuses & 1 != 0;
        MeasurementResult {
            outcome,
            is_deterministic: true,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn nondeterministic_meas(&mut self, qubit: usize, outcome: bool) -> MeasurementResult {
        let words_per_row = self.words_per_row;
        let words_per_col = self.words_per_col;
        let col_base = qubit * words_per_col;

        // Find minimum weight anti-commuting stabilizer
        let mut min_weight = usize::MAX;
        let mut pivot_id = 0;

        for w in 0..words_per_col {
            let mut mask = self.stab_col_x[col_base + w];
            while mask != 0 {
                let g = w * 64 + mask.trailing_zeros() as usize;
                let weight = row_weight(&self.stab_row_x, words_per_row, g)
                    + row_weight(&self.stab_row_z, words_per_row, g);
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

        // Cache pivot sign lookups
        let pivot_sign_minus = get_sign(&self.stab_signs_minus, pivot_id);
        let pivot_sign_i = get_sign(&self.stab_signs_i, pivot_id);

        // Handle pivot's i-phase contribution (bulk operation before per-generator loop)
        // When multiplying by a Pauli with i phase:
        //   If target g also has i: i*i = -1, so toggle g's minus and clear g's i
        //   If target g doesn't have i: 1*i = i, so set g's i
        //   Net effect: toggle minus for anticom stabs WITH i, then toggle i for all anticom stabs
        if pivot_sign_i {
            clear_sign(&mut self.stab_signs_i, pivot_id);
            for w in 0..words_per_col {
                let mut anticom = self.stab_col_x[col_base + w];
                if w == pivot_id / 64 {
                    anticom &= !(1u64 << (pivot_id % 64));
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
            if w == pivot_id / 64 {
                mask &= !(1u64 << (pivot_id % 64));
            }

            while mask != 0 {
                let g = w * 64 + mask.trailing_zeros() as usize;

                // Phase calculation: count Z(pivot) & X(g) overlaps
                let base_p = pivot_id * words_per_row;
                let base_g = g * words_per_row;
                let mut count = 0;
                for ww in 0..words_per_row {
                    count += (self.stab_row_z[base_p + ww] & self.stab_row_x[base_g + ww])
                        .count_ones() as usize;
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

                mask &= mask - 1;
            }
        }

        // Save anticom_mask before modifying columns (it gets zeroed when q == qubit)
        // Use scratch_row to store anticom_mask since it's the right size (words_per_col)
        for w in 0..words_per_col {
            self.scratch_row[w] = self.stab_col_x[col_base + w];
        }

        // Update columns - use scratch buffer to avoid allocations
        // Process pivot_x_qubits: XOR anticom into stab_col_x and clear pivot bit
        self.scratch_qubits.clear();
        {
            let base = pivot_id * words_per_row;
            for w in 0..words_per_row {
                let mut word = self.stab_row_x[base + w];
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    self.scratch_qubits.push(w * 64 + bit);
                    word &= word - 1;
                }
            }
        }
        for &q in &self.scratch_qubits {
            for w in 0..words_per_col {
                let anticom_mask = self.scratch_row[w];
                self.stab_col_x[q * words_per_col + w] ^= anticom_mask;
            }
            clear_bit_col(&mut self.stab_col_x, words_per_col, q, pivot_id);
        }

        // Process pivot_z_qubits: XOR anticom into stab_col_z and clear pivot bit
        self.scratch_qubits.clear();
        {
            let base = pivot_id * words_per_row;
            for w in 0..words_per_row {
                let mut word = self.stab_row_z[base + w];
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    self.scratch_qubits.push(w * 64 + bit);
                    word &= word - 1;
                }
            }
        }
        for &q in &self.scratch_qubits {
            for w in 0..words_per_col {
                let anticom_mask = self.scratch_row[w];
                self.stab_col_z[q * words_per_col + w] ^= anticom_mask;
            }
            clear_bit_col(&mut self.stab_col_z, words_per_col, q, pivot_id);
        }

        // Step 2b (Aaronson-Gottesman): XOR pivot into anti-commuting destabilizers
        // Find destabilizers that anti-commute with Z_q (have X on measured qubit)
        // Reuse scratch_row for the destab anticom mask
        for w in 0..words_per_col {
            self.scratch_row[w] = self.destab_col_x[col_base + w];
        }
        // Exclude pivot from the anticom mask
        self.scratch_row[pivot_id / 64] &= !(1u64 << (pivot_id % 64));

        // XOR pivot's stab rows into anti-commuting destabilizer rows
        let pivot_row_base = pivot_id * words_per_row;
        for w in 0..words_per_col {
            let mut mask = self.scratch_row[w];
            while mask != 0 {
                let g = w * 64 + mask.trailing_zeros() as usize;
                let base_g = g * words_per_row;
                for ww in 0..words_per_row {
                    self.destab_row_x[base_g + ww] ^= self.stab_row_x[pivot_row_base + ww];
                    self.destab_row_z[base_g + ww] ^= self.stab_row_z[pivot_row_base + ww];
                }
                mask &= mask - 1;
            }
        }

        // Update destab columns for qubits where pivot has X
        self.scratch_qubits.clear();
        {
            for w in 0..words_per_row {
                let mut word = self.stab_row_x[pivot_row_base + w];
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    self.scratch_qubits.push(w * 64 + bit);
                    word &= word - 1;
                }
            }
        }
        for &q in &self.scratch_qubits {
            for w in 0..words_per_col {
                self.destab_col_x[q * words_per_col + w] ^= self.scratch_row[w];
            }
        }

        // Update destab columns for qubits where pivot has Z
        self.scratch_qubits.clear();
        {
            for w in 0..words_per_row {
                let mut word = self.stab_row_z[pivot_row_base + w];
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    self.scratch_qubits.push(w * 64 + bit);
                    word &= word - 1;
                }
            }
        }
        for &q in &self.scratch_qubits {
            for w in 0..words_per_col {
                self.destab_col_z[q * words_per_col + w] ^= self.scratch_row[w];
            }
        }

        // Copy old pivot stabilizer to destabilizer BEFORE replacing it
        // First clear old destab columns
        let pivot_base = pivot_id * words_per_row;
        self.scratch_qubits.clear();
        {
            for w in 0..words_per_row {
                let mut word = self.destab_row_x[pivot_base + w];
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    self.scratch_qubits.push(w * 64 + bit);
                    word &= word - 1;
                }
            }
        }
        for &q in &self.scratch_qubits {
            clear_bit_col(&mut self.destab_col_x, words_per_col, q, pivot_id);
        }

        self.scratch_qubits.clear();
        {
            for w in 0..words_per_row {
                let mut word = self.destab_row_z[pivot_base + w];
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    self.scratch_qubits.push(w * 64 + bit);
                    word &= word - 1;
                }
            }
        }
        for &q in &self.scratch_qubits {
            clear_bit_col(&mut self.destab_col_z, words_per_col, q, pivot_id);
        }

        // Copy stabilizer rows to destabilizer rows
        for w in 0..words_per_row {
            self.destab_row_x[pivot_base + w] = self.stab_row_x[pivot_base + w];
            self.destab_row_z[pivot_base + w] = self.stab_row_z[pivot_base + w];
        }

        // Set destab columns to match the copied rows
        self.scratch_qubits.clear();
        {
            for w in 0..words_per_row {
                let mut word = self.stab_row_x[pivot_base + w];
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    self.scratch_qubits.push(w * 64 + bit);
                    word &= word - 1;
                }
            }
        }
        for &q in &self.scratch_qubits {
            set_bit_col(&mut self.destab_col_x, words_per_col, q, pivot_id);
        }

        self.scratch_qubits.clear();
        {
            for w in 0..words_per_row {
                let mut word = self.stab_row_z[pivot_base + w];
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    self.scratch_qubits.push(w * 64 + bit);
                    word &= word - 1;
                }
            }
        }
        for &q in &self.scratch_qubits {
            set_bit_col(&mut self.destab_col_z, words_per_col, q, pivot_id);
        }

        // Copy stabilizer sign to destabilizer
        if get_sign(&self.stab_signs_minus, pivot_id) {
            set_sign(&mut self.destab_signs_minus, pivot_id);
        } else {
            clear_sign(&mut self.destab_signs_minus, pivot_id);
        }
        clear_sign(&mut self.destab_signs_i, pivot_id);

        // NOW replace pivot stabilizer with Z_qubit
        // First, clear old column entries for qubits where pivot had X
        self.scratch_qubits.clear();
        for w in 0..words_per_row {
            let mut word = self.stab_row_x[pivot_base + w];
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                self.scratch_qubits.push(w * 64 + bit);
                word &= word - 1;
            }
        }
        for &q in &self.scratch_qubits {
            clear_bit_col(&mut self.stab_col_x, words_per_col, q, pivot_id);
        }

        // Clear old column entries for qubits where pivot had Z
        self.scratch_qubits.clear();
        for w in 0..words_per_row {
            let mut word = self.stab_row_z[pivot_base + w];
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                self.scratch_qubits.push(w * 64 + bit);
                word &= word - 1;
            }
        }
        for &q in &self.scratch_qubits {
            clear_bit_col(&mut self.stab_col_z, words_per_col, q, pivot_id);
        }

        // Now clear rows and set new Z_qubit
        for w in 0..words_per_row {
            self.stab_row_x[pivot_base + w] = 0;
            self.stab_row_z[pivot_base + w] = 0;
        }
        set_bit_row(&mut self.stab_row_z, words_per_row, pivot_id, qubit);
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

    /// Measure a qubit with a forced outcome for non-deterministic measurements.
    ///
    /// This is used for testing probability calculations by forcing specific
    /// measurement outcomes when the result would otherwise be random.
    pub fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        let deterministic = col_is_empty(&self.stab_col_x, self.words_per_col, qubit);
        if deterministic {
            self.deterministic_meas(qubit)
        } else {
            self.nondeterministic_meas(qubit, forced_outcome)
        }
    }
}

impl<R: SeedableRng + Rng + Debug + Clone> QuantumSimulator for DenseStab<R> {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn reset(&mut self) -> &mut Self {
        self.init_state();
        self
    }
}

impl<R: SeedableRng + Rng + Debug + Clone> CliffordGateable for DenseStab<R> {
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_h(q.index());
        }
        self
    }

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_s(q.index());
        }
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        // S†: X → (-i)XZ, Z → Z
        // Multiply phase by -i for generators with X:
        // -i = (-1)*i, so first multiply by i (toggle i, and if i was set, i*i=-1 toggle minus)
        // then multiply by -1 (toggle minus)
        // Combined: toggle minus for X without i, then toggle i
        for &q in qubits {
            let qubit = q.index();
            let col_base = qubit * self.words_per_col;
            let qubit_word = qubit / 64;
            let mask = 1u64 << (qubit % 64);

            // Update columns and signs for stabilizers
            for w in 0..self.words_per_col {
                let x_gens = self.stab_col_x[col_base + w];
                // For -i multiplication: toggle minus for X without existing i
                self.stab_signs_minus[w] ^= x_gens & !self.stab_signs_i[w];
                // Toggle i for all X generators
                self.stab_signs_i[w] ^= x_gens;
                // Toggle Z for all X generators
                self.stab_col_z[col_base + w] ^= x_gens;
            }

            // Update rows for stabilizers
            for g in 0..self.num_qubits {
                let row_idx = g * self.words_per_row + qubit_word;
                debug_assert!(row_idx < self.stab_row_x.len());
                debug_assert!(row_idx < self.stab_row_z.len());
                unsafe {
                    let rx = *self.stab_row_x.get_unchecked(row_idx);
                    let x_bit = rx & mask;
                    *self.stab_row_z.get_unchecked_mut(row_idx) ^= x_bit;
                }
            }

            // Update columns and signs for destabilizers
            for w in 0..self.words_per_col {
                let x_gens = self.destab_col_x[col_base + w];
                self.destab_signs_minus[w] ^= x_gens & !self.destab_signs_i[w];
                self.destab_signs_i[w] ^= x_gens;
                self.destab_col_z[col_base + w] ^= x_gens;
            }

            // Update rows for destabilizers
            for g in 0..self.num_qubits {
                let row_idx = g * self.words_per_row + qubit_word;
                debug_assert!(row_idx < self.destab_row_x.len());
                debug_assert!(row_idx < self.destab_row_z.len());
                unsafe {
                    let rx = *self.destab_row_x.get_unchecked(row_idx);
                    let x_bit = rx & mask;
                    *self.destab_row_z.get_unchecked_mut(row_idx) ^= x_bit;
                }
            }
        }
        self
    }

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qubit = q.index();
            let col_base = qubit * self.words_per_col;
            for w in 0..self.words_per_col {
                self.stab_signs_minus[w] ^= self.stab_col_z[col_base + w];
                self.destab_signs_minus[w] ^= self.destab_col_z[col_base + w];
            }
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qubit = q.index();
            let col_base = qubit * self.words_per_col;
            for w in 0..self.words_per_col {
                let x_xor_z = self.stab_col_x[col_base + w] ^ self.stab_col_z[col_base + w];
                self.stab_signs_minus[w] ^= x_xor_z;
                let x_xor_z = self.destab_col_x[col_base + w] ^ self.destab_col_z[col_base + w];
                self.destab_signs_minus[w] ^= x_xor_z;
            }
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qubit = q.index();
            let col_base = qubit * self.words_per_col;
            for w in 0..self.words_per_col {
                self.stab_signs_minus[w] ^= self.stab_col_x[col_base + w];
                self.destab_signs_minus[w] ^= self.destab_col_x[col_base + w];
            }
        }
        self
    }

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // DOD optimization: for multiple gates, batch stab updates then destab updates
        // This keeps each data structure hot in cache
        if pairs.len() > 1 {
            let words_per_row = self.words_per_row;
            let words_per_col = self.words_per_col;

            // Process all stabilizer updates first
            for &(control_q, target_q) in pairs {
                let control = control_q.index();
                let target = target_q.index();
                Self::apply_cx_to_gens(
                    &mut self.stab_row_x,
                    &mut self.stab_row_z,
                    &mut self.stab_col_x,
                    &mut self.stab_col_z,
                    words_per_row,
                    words_per_col,
                    control,
                    target,
                );
            }

            // Then process all destabilizer updates
            for &(control_q, target_q) in pairs {
                let control = control_q.index();
                let target = target_q.index();
                Self::apply_cx_to_gens(
                    &mut self.destab_row_x,
                    &mut self.destab_row_z,
                    &mut self.destab_col_x,
                    &mut self.destab_col_z,
                    words_per_row,
                    words_per_col,
                    control,
                    target,
                );
            }
        } else {
            // Single gate: use normal path
            for &(control_q, target_q) in pairs {
                self.apply_cx(control_q.index(), target_q.index());
            }
        }
        self
    }

    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // CZ: X_1 -> X_1 Z_2, X_2 -> Z_1 X_2, Z -> Z
        // Sign: toggle minus for generators with X on both qubits
        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            let q1_word = q1 / 64;
            let q1_mask = 1u64 << (q1 % 64);
            let q2_word = q2 / 64;
            let q2_mask = 1u64 << (q2 % 64);
            let col_base_1 = q1 * self.words_per_col;
            let col_base_2 = q2 * self.words_per_col;

            // Process stabilizers
            // Sign update: toggle minus for generators with X on both qubits
            for w in 0..self.words_per_col {
                let x1 = self.stab_col_x[col_base_1 + w];
                let x2 = self.stab_col_x[col_base_2 + w];
                self.stab_signs_minus[w] ^= x1 & x2;
            }

            // For generators with X on q1, toggle Z on q2 (column update)
            for w in 0..self.words_per_col {
                let x1_gens = self.stab_col_x[col_base_1 + w];
                self.stab_col_z[col_base_2 + w] ^= x1_gens;
            }

            // For generators with X on q2, toggle Z on q1 (column update)
            for w in 0..self.words_per_col {
                let x2_gens = self.stab_col_x[col_base_2 + w];
                self.stab_col_z[col_base_1 + w] ^= x2_gens;
            }

            // Row updates for stabilizers
            for g in 0..self.num_qubits {
                let row_base = g * self.words_per_row;
                debug_assert!(row_base + q1_word < self.stab_row_x.len());
                debug_assert!(row_base + q2_word < self.stab_row_x.len());
                debug_assert!(row_base + q1_word < self.stab_row_z.len());
                debug_assert!(row_base + q2_word < self.stab_row_z.len());
                unsafe {
                    let x1_bit = *self.stab_row_x.get_unchecked(row_base + q1_word) & q1_mask;
                    let x2_bit = *self.stab_row_x.get_unchecked(row_base + q2_word) & q2_mask;

                    // If X on q1, toggle Z on q2
                    if x1_bit != 0 {
                        *self.stab_row_z.get_unchecked_mut(row_base + q2_word) ^= q2_mask;
                    }
                    // If X on q2, toggle Z on q1
                    if x2_bit != 0 {
                        *self.stab_row_z.get_unchecked_mut(row_base + q1_word) ^= q1_mask;
                    }
                }
            }

            // Process destabilizers (same pattern, no sign update needed for destabs)
            for w in 0..self.words_per_col {
                let x1_gens = self.destab_col_x[col_base_1 + w];
                self.destab_col_z[col_base_2 + w] ^= x1_gens;
            }

            for w in 0..self.words_per_col {
                let x2_gens = self.destab_col_x[col_base_2 + w];
                self.destab_col_z[col_base_1 + w] ^= x2_gens;
            }

            for g in 0..self.num_qubits {
                let row_base = g * self.words_per_row;
                debug_assert!(row_base + q1_word < self.destab_row_x.len());
                debug_assert!(row_base + q2_word < self.destab_row_x.len());
                debug_assert!(row_base + q1_word < self.destab_row_z.len());
                debug_assert!(row_base + q2_word < self.destab_row_z.len());
                unsafe {
                    let x1_bit = *self.destab_row_x.get_unchecked(row_base + q1_word) & q1_mask;
                    let x2_bit = *self.destab_row_x.get_unchecked(row_base + q2_word) & q2_mask;

                    if x1_bit != 0 {
                        *self.destab_row_z.get_unchecked_mut(row_base + q2_word) ^= q2_mask;
                    }
                    if x2_bit != 0 {
                        *self.destab_row_z.get_unchecked_mut(row_base + q1_word) ^= q1_mask;
                    }
                }
            }
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let mut results = Vec::with_capacity(qubits.len());

        for &q in qubits {
            let qubit = q.index();
            let deterministic = col_is_empty(&self.stab_col_x, self.words_per_col, qubit);

            let result = if deterministic {
                self.deterministic_meas(qubit)
            } else {
                let outcome = self.rng.coin_flip();
                self.nondeterministic_meas(qubit, outcome)
            };
            results.push(result);
        }

        results
    }
}

impl<R: SeedableRng + Rng + Debug> RngManageable for DenseStab<R> {
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

impl<R: SeedableRng + Rng + Debug + Clone> StabilizerTableauSimulator for DenseStab<R> {
    fn stab_tableau(&self) -> String {
        Self::gen_tableau_string(
            self.num_qubits,
            self.words_per_row,
            &self.stab_row_x,
            &self.stab_row_z,
            &self.stab_signs_minus,
            &self.stab_signs_i,
        )
    }

    fn destab_tableau(&self) -> String {
        Self::gen_tableau_string(
            self.num_qubits,
            self.words_per_row,
            &self.destab_row_x,
            &self.destab_row_z,
            &self.destab_signs_minus,
            &self.destab_signs_i,
        )
    }
}

impl<R: Rng> DenseStab<R> {
    /// Produces a tableau string from dense bit arrays.
    fn gen_tableau_string(
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

    /// Returns generator data as sparse index vectors, matching the format used by `PySparseSim::_gens_data()`.
    ///
    /// Returns `(col_x, col_z, row_x, row_z)` where each is a `Vec<Vec<usize>>`.
    pub fn gens_data(&self, is_stab: bool) -> crate::GensData {
        let (row_x, row_z, col_x, col_z) = if is_stab {
            (
                &self.stab_row_x,
                &self.stab_row_z,
                &self.stab_col_x,
                &self.stab_col_z,
            )
        } else {
            (
                &self.destab_row_x,
                &self.destab_row_z,
                &self.destab_col_x,
                &self.destab_col_z,
            )
        };

        let extract_rows = |data: &[u64]| -> Vec<Vec<usize>> {
            (0..self.num_qubits)
                .map(|row| {
                    let base = row * self.words_per_row;
                    let mut indices = Vec::new();
                    for w in 0..self.words_per_row {
                        let mut word = data[base + w];
                        while word != 0 {
                            let bit = word.trailing_zeros() as usize;
                            indices.push(w * 64 + bit);
                            word &= word - 1;
                        }
                    }
                    indices
                })
                .collect()
        };

        let extract_cols = |data: &[u64]| -> Vec<Vec<usize>> {
            (0..self.num_qubits)
                .map(|col| {
                    let base = col * self.words_per_col;
                    let mut indices = Vec::new();
                    for w in 0..self.words_per_col {
                        let mut word = data[base + w];
                        while word != 0 {
                            let bit = word.trailing_zeros() as usize;
                            indices.push(w * 64 + bit);
                            word &= word - 1;
                        }
                    }
                    indices
                })
                .collect()
        };

        (
            extract_cols(col_x),
            extract_cols(col_z),
            extract_rows(row_x),
            extract_rows(row_z),
        )
    }
}

use crate::stabilizer_test_utils::{ForcedMeasurement, StabilizerSimulator};

impl<R: SeedableRng + Rng + Debug + Clone> ForcedMeasurement for DenseStab<R> {
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        DenseStab::mz_forced(self, qubit, forced_outcome)
    }
}

impl StabilizerSimulator for DenseStab<PecosRng> {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_creation() {
        let sim = DenseStab::new(4);
        assert_eq!(sim.num_qubits(), 4);
    }

    #[test]
    fn test_x_gate() {
        let mut sim = DenseStab::with_seed(1, 42);
        sim.x(&[QubitId(0)]);
        let result = sim.mz(&[QubitId(0)]);
        assert!(result[0].outcome);
        assert!(result[0].is_deterministic);
    }

    #[test]
    fn test_bell_state() {
        let mut sim = DenseStab::with_seed(2, 42);
        sim.h(&[QubitId(0)]);
        sim.cx(&[(QubitId(0), QubitId(1))]);

        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert_eq!(results[0].outcome, results[1].outcome);
    }

    #[test]
    fn test_bell_state_sequential_meas() {
        // This mirrors the test suite's verify_bell_state_correlations
        let mut sim = DenseStab::with_seed(2, 42);
        sim.h(&[QubitId(0)]);
        sim.cx(&[(QubitId(0), QubitId(1))]);

        // Measure first qubit
        let r0 = sim.mz(&[QubitId(0)]);
        assert!(
            !r0[0].is_deterministic,
            "First Bell measurement should be non-deterministic"
        );

        // Measure second qubit - should be deterministic and correlated
        let r1 = sim.mz(&[QubitId(1)]);
        assert!(
            r1[0].is_deterministic,
            "Second Bell measurement should be deterministic"
        );
        assert_eq!(
            r0[0].outcome, r1[0].outcome,
            "Bell measurements should be correlated"
        );
    }

    #[test]
    fn test_bell_state_8_qubits() {
        // Test Bell state on first 2 qubits of an 8-qubit simulator
        let mut sim = DenseStab::with_seed(8, 42);
        sim.h(&[QubitId(0)]);
        sim.cx(&[(QubitId(0), QubitId(1))]);

        // Measure first qubit
        let r0 = sim.mz(&[QubitId(0)]);
        assert!(
            !r0[0].is_deterministic,
            "First Bell measurement should be non-deterministic"
        );

        // Measure second qubit - should be deterministic and correlated
        let r1 = sim.mz(&[QubitId(1)]);
        assert!(
            r1[0].is_deterministic,
            "Second Bell measurement should be deterministic"
        );
        assert_eq!(
            r0[0].outcome, r1[0].outcome,
            "Bell measurements should be correlated"
        );
    }

    #[test]
    fn test_z_on_plus() {
        let mut sim = DenseStab::with_seed(1, 42);
        sim.h(&[QubitId(0)]); // |+>
        sim.z(&[QubitId(0)]); // |->
        sim.h(&[QubitId(0)]); // |1>
        let result = sim.mz(&[QubitId(0)]);
        assert!(result[0].outcome);
        assert!(result[0].is_deterministic);
    }

    #[test]
    fn test_reset() {
        let mut sim = DenseStab::with_seed(2, 42);
        sim.x(&[QubitId(0), QubitId(1)]);
        sim.reset();
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert!(!results[0].outcome);
        assert!(!results[1].outcome);
    }

    #[test]
    fn test_sx_gate() {
        // SX^4 = I, so applying SX 4 times should return to initial state
        let mut sim: DenseStab = DenseStab::with_seed(1, 42);
        for _ in 0..4 {
            sim.sx(&[QubitId(0)]);
        }
        let result = sim.mz(&[QubitId(0)]);
        assert!(!result[0].outcome, "SX^4 should return to |0>");
        assert!(result[0].is_deterministic, "Should be deterministic");
    }

    #[test]
    fn test_sx_creates_superposition() {
        // SX on |0> should create a superposition
        let mut sim: DenseStab = DenseStab::with_seed(1, 42);
        sim.sx(&[QubitId(0)]);
        let result = sim.mz_forced(0, false);
        assert!(
            !result.is_deterministic,
            "SX|0> should be non-deterministic"
        );
    }

    #[test]
    fn test_dense_stab_vs_sparse_stab_random_circuits() {
        use crate::SparseStab;
        use crate::stabilizer_test_utils::compare_simulators_on_random_circuits;

        let mut dense: DenseStab = DenseStab::with_seed(8, 42);
        let mut sparse = SparseStab::new(8);
        // Run same random circuits on both and compare results
        compare_simulators_on_random_circuits(&mut dense, &mut sparse, 8, 20, 10, 12345);
    }

    #[test]
    fn test_dense_stab_full_stabilizer_suite() {
        use crate::stabilizer_test_utils::run_full_stabilizer_test_suite;
        let mut sim: DenseStab = DenseStab::with_seed(8, 42);
        run_full_stabilizer_test_suite(&mut sim, 8);
    }

    #[test]
    fn test_forced_meas_then_remeasure() {
        // Regression test: after a forced measurement, re-measuring the same qubit
        // should deterministically give the same result.
        let mut dense: DenseStab = DenseStab::new(2);
        dense.h(&[QubitId(0)]);
        dense.cx(&[(QubitId(0), QubitId(1))]);
        dense.h(&[QubitId(0)]);
        dense.sz(&[QubitId(0)]);

        // First forced measurement on qubit 1
        let r1 = dense.mz_forced(1, false);
        assert!(!r1.outcome, "Forced to 0 should return 0");

        // Second measurement should be deterministic and return 0
        let r2 = dense.mz_forced(1, false);
        assert!(
            r2.is_deterministic,
            "After measurement, qubit should be deterministic"
        );
        assert!(!r2.outcome, "After forced-0, re-measuring should give 0");
    }

    #[test]
    fn test_mid_circuit_meas_dense_vs_sparse() {
        // Compare DenseStab and SparseStab on random circuits with mid-circuit
        // measurements and init |0> operations (mz_forced + conditional X).
        // This catches bugs in nondeterministic_meas that pure Clifford tests miss.
        use crate::SparseStab;
        use pecos_random::{PecosRng, RngExt};

        let num_qubits = 10;
        let num_circuits = 200;
        let num_gates = 50;

        for circuit_idx in 0..num_circuits {
            let seed = 42_000 + circuit_idx;
            let mut rng = PecosRng::seed_from_u64(seed);

            let mut dense = DenseStab::<PecosRng>::new(num_qubits);
            let mut sparse = SparseStab::new(num_qubits);

            for gate_idx in 0..num_gates {
                let gate_type: u8 = rng.random_range(0..16);
                let q0 = rng.random_range(0..num_qubits);

                match gate_type {
                    0 => {
                        dense.h(&[QubitId(q0)]);
                        sparse.h(&[QubitId(q0)]);
                    }
                    1 => {
                        dense.sz(&[QubitId(q0)]);
                        sparse.sz(&[QubitId(q0)]);
                    }
                    2 => {
                        dense.szdg(&[QubitId(q0)]);
                        sparse.szdg(&[QubitId(q0)]);
                    }
                    3 => {
                        dense.x(&[QubitId(q0)]);
                        sparse.x(&[QubitId(q0)]);
                    }
                    4 => {
                        dense.y(&[QubitId(q0)]);
                        sparse.y(&[QubitId(q0)]);
                    }
                    5 => {
                        dense.z(&[QubitId(q0)]);
                        sparse.z(&[QubitId(q0)]);
                    }
                    6..=9 if num_qubits >= 2 => {
                        let mut q1 = rng.random_range(0..num_qubits);
                        while q1 == q0 {
                            q1 = rng.random_range(0..num_qubits);
                        }
                        let pair = &[(QubitId(q0), QubitId(q1))];
                        match gate_type {
                            6 => {
                                dense.cx(pair);
                                sparse.cx(pair);
                            }
                            7 => {
                                dense.cz(pair);
                                sparse.cz(pair);
                            }
                            8 => {
                                dense.cy(pair);
                                sparse.cy(pair);
                            }
                            _ => {
                                dense.swap(pair);
                                sparse.swap(pair);
                            }
                        }
                    }
                    10..=12 => {
                        // Forced measurement (mid-circuit)
                        let forced: bool = rng.random();
                        let rd = dense.mz_forced(q0, forced);
                        let rs = sparse.mz_forced(q0, forced);
                        assert_eq!(
                            rd.outcome, rs.outcome,
                            "circuit {seed} gate {gate_idx}: mz_forced({q0}, {forced}) outcome mismatch"
                        );
                        assert_eq!(
                            rd.is_deterministic, rs.is_deterministic,
                            "circuit {seed} gate {gate_idx}: mz_forced({q0}, {forced}) determinism mismatch"
                        );
                    }
                    13..=14 => {
                        // Init |0> = mz_forced + conditional X
                        let rd = dense.mz_forced(q0, false);
                        let rs = sparse.mz_forced(q0, false);
                        assert_eq!(
                            rd.outcome, rs.outcome,
                            "circuit {seed} gate {gate_idx}: init|0> mz_forced({q0}) outcome mismatch"
                        );
                        if rd.outcome {
                            dense.x(&[QubitId(q0)]);
                            sparse.x(&[QubitId(q0)]);
                        }
                    }
                    _ => {
                        // SX gate
                        dense.sx(&[QubitId(q0)]);
                        sparse.sx(&[QubitId(q0)]);
                    }
                }
            }

            // Final measurement of all qubits
            for q in 0..num_qubits {
                let forced: bool = PecosRng::seed_from_u64(seed + 1000 + q as u64).random();
                let rd = dense.mz_forced(q, forced);
                let rs = sparse.mz_forced(q, forced);
                assert_eq!(
                    rd.outcome, rs.outcome,
                    "circuit {seed}: final mz_forced({q}, {forced}) outcome mismatch"
                );
                assert_eq!(
                    rd.is_deterministic, rs.is_deterministic,
                    "circuit {seed}: final mz_forced({q}, {forced}) determinism mismatch"
                );
            }
        }
    }
}
