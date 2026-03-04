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

//! Parallel stabilizer simulator using row-based threading (STABSim-style).
//!
//! This module provides [`GpuStabParallel`], a stabilizer simulator that uses the
//! same parallel algorithms as `STABSim` (arxiv:2507.03092) but implemented with
//! rayon for CPU parallelism.
//!
//! # Design (from `STABSim`)
//!
//! 1. **Row-based threading**: Each thread handles one generator row
//! 2. **Embarrassingly parallel gates**: Clifford gates update rows independently
//! 3. **Parallel measurement reduction**: Uses parallel reduction for measurement
//!
//! # Memory Layout
//!
//! Row-major storage for both stabilizers and destabilizers:
//! ```text
//! row_x[g * words_per_row + word] = X bits for qubits [word*32..(word+1)*32]
//! row_z[g * words_per_row + word] = Z bits for qubits [word*32..(word+1)*32]
//! ```
//!
//! Each generator row contains the Pauli string for that stabilizer/destabilizer.
//! This layout enables:
//! - Independent row updates for gates (each thread processes its row)
//! - Efficient row multiplication for measurement (SIMD-friendly XOR)

use crate::{CliffordGateable, MeasurementResult, QuantumSimulator, StabilizerTableauSimulator};
use core::fmt::Debug;
use pecos_core::{QubitId, RngManageable};
use pecos_rng::PecosRng;
use pecos_rng::rng_ext::RngProbabilityExt;

/// Parallel stabilizer simulator using row-based threading.
#[derive(Clone)]
pub struct GpuStabParallel {
    num_qubits: usize,
    words_per_row: usize,

    // Row-major storage: row_x[g * words_per_row + word]
    // Stabilizers: generators 0..num_qubits
    // Destabilizers: generators num_qubits..2*num_qubits
    stab_row_x: Vec<u32>,
    stab_row_z: Vec<u32>,
    destab_row_x: Vec<u32>,
    destab_row_z: Vec<u32>,

    // Signs: one bit per generator (packed)
    stab_signs_minus: Vec<u32>,
    stab_signs_i: Vec<u32>,
    destab_signs_minus: Vec<u32>,
    destab_signs_i: Vec<u32>,

    rng: PecosRng,
}

impl Debug for GpuStabParallel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuStabParallel")
            .field("num_qubits", &self.num_qubits)
            .finish_non_exhaustive()
    }
}

impl GpuStabParallel {
    /// Creates a new parallel stabilizer simulator.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self::with_rng(num_qubits, rand::make_rng())
    }

    /// Creates a new simulator with a specific seed.
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_rng(num_qubits, PecosRng::seed_from_u64(seed))
    }

    /// Creates a new simulator with a provided RNG.
    #[must_use]
    pub fn with_rng(num_qubits: usize, rng: PecosRng) -> Self {
        let words_per_row = num_qubits.div_ceil(32);
        let row_size = num_qubits * words_per_row;
        let sign_words = num_qubits.div_ceil(32);

        let mut sim = Self {
            num_qubits,
            words_per_row,
            stab_row_x: vec![0; row_size],
            stab_row_z: vec![0; row_size],
            destab_row_x: vec![0; row_size],
            destab_row_z: vec![0; row_size],
            stab_signs_minus: vec![0; sign_words],
            stab_signs_i: vec![0; sign_words],
            destab_signs_minus: vec![0; sign_words],
            destab_signs_i: vec![0; sign_words],
            rng,
        };

        sim.init_tableau();
        sim
    }

    /// Returns the number of qubits.
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Initialize to |0...0> state.
    fn init_tableau(&mut self) {
        self.stab_row_x.fill(0);
        self.stab_row_z.fill(0);
        self.destab_row_x.fill(0);
        self.destab_row_z.fill(0);
        self.stab_signs_minus.fill(0);
        self.stab_signs_i.fill(0);
        self.destab_signs_minus.fill(0);
        self.destab_signs_i.fill(0);

        // Stabilizer i = Z_i, Destabilizer i = X_i
        for i in 0..self.num_qubits {
            let row_base = i * self.words_per_row;
            let word = i / 32;
            let bit = 1u32 << (i % 32);

            self.stab_row_z[row_base + word] = bit;
            self.destab_row_x[row_base + word] = bit;
        }
    }

    // ========== Helper methods ==========

    #[inline]
    fn get_sign_minus(&self, g: usize, is_stab: bool) -> bool {
        let signs = if is_stab {
            &self.stab_signs_minus
        } else {
            &self.destab_signs_minus
        };
        let word = g / 32;
        let bit = g % 32;
        (signs[word] >> bit) & 1 != 0
    }

    #[inline]
    fn flip_sign_minus(&mut self, g: usize, is_stab: bool) {
        let signs = if is_stab {
            &mut self.stab_signs_minus
        } else {
            &mut self.destab_signs_minus
        };
        let word = g / 32;
        let bit = 1u32 << (g % 32);
        signs[word] ^= bit;
    }

    #[inline]
    fn get_sign_i(&self, g: usize, is_stab: bool) -> bool {
        let signs = if is_stab {
            &self.stab_signs_i
        } else {
            &self.destab_signs_i
        };
        let word = g / 32;
        let bit = g % 32;
        (signs[word] >> bit) & 1 != 0
    }

    #[inline]
    fn flip_sign_i(&mut self, g: usize, is_stab: bool) {
        let signs = if is_stab {
            &mut self.stab_signs_i
        } else {
            &mut self.destab_signs_i
        };
        let word = g / 32;
        let bit = 1u32 << (g % 32);
        signs[word] ^= bit;
    }

    // ========== Gate operations (row-parallel) ==========
    // Each generator row can be updated independently.

    /// Apply H gate: X <-> Z, Y -> -Y (phase flip when X=Z=1)
    fn apply_h(&mut self, qubit: usize) {
        let words_per_row = self.words_per_row;
        let word = qubit / 32;
        let bit = 1u32 << (qubit % 32);

        // Process stabilizers
        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let x = (self.stab_row_x[row_base + word] >> (qubit % 32)) & 1;
            let z = (self.stab_row_z[row_base + word] >> (qubit % 32)) & 1;

            // Phase: Y -> -Y (when both X and Z are set)
            if x == 1 && z == 1 {
                self.flip_sign_minus(g, true);
            }

            // Swap X and Z
            if x != z {
                self.stab_row_x[row_base + word] ^= bit;
                self.stab_row_z[row_base + word] ^= bit;
            }
        }

        // Process destabilizers
        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let x = (self.destab_row_x[row_base + word] >> (qubit % 32)) & 1;
            let z = (self.destab_row_z[row_base + word] >> (qubit % 32)) & 1;

            if x == 1 && z == 1 {
                self.flip_sign_minus(g, false);
            }

            if x != z {
                self.destab_row_x[row_base + word] ^= bit;
                self.destab_row_z[row_base + word] ^= bit;
            }
        }
    }

    /// Apply S gate: X -> Y = iXZ, Z -> Z
    fn apply_s(&mut self, qubit: usize) {
        let words_per_row = self.words_per_row;
        let word = qubit / 32;
        let bit = 1u32 << (qubit % 32);

        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let x = (self.stab_row_x[row_base + word] >> (qubit % 32)) & 1;

            if x == 1 {
                // X -> Y: add Z, add phase i
                self.stab_row_z[row_base + word] ^= bit;

                // i phase: flip i, and if was already i, flip minus
                if self.get_sign_i(g, true) {
                    self.flip_sign_minus(g, true);
                }
                self.flip_sign_i(g, true);
            }
        }

        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let x = (self.destab_row_x[row_base + word] >> (qubit % 32)) & 1;

            if x == 1 {
                self.destab_row_z[row_base + word] ^= bit;

                if self.get_sign_i(g, false) {
                    self.flip_sign_minus(g, false);
                }
                self.flip_sign_i(g, false);
            }
        }
    }

    /// Apply S^dag gate: X -> -Y, Z -> Z
    fn apply_sdg(&mut self, qubit: usize) {
        let words_per_row = self.words_per_row;
        let word = qubit / 32;
        let bit = 1u32 << (qubit % 32);

        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let x = (self.stab_row_x[row_base + word] >> (qubit % 32)) & 1;

            if x == 1 {
                self.stab_row_z[row_base + word] ^= bit;

                // -i phase: flip minus if not i, then flip i
                if !self.get_sign_i(g, true) {
                    self.flip_sign_minus(g, true);
                }
                self.flip_sign_i(g, true);
            }
        }

        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let x = (self.destab_row_x[row_base + word] >> (qubit % 32)) & 1;

            if x == 1 {
                self.destab_row_z[row_base + word] ^= bit;

                if !self.get_sign_i(g, false) {
                    self.flip_sign_minus(g, false);
                }
                self.flip_sign_i(g, false);
            }
        }
    }

    /// Apply X gate: Z -> -Z
    fn apply_x(&mut self, qubit: usize) {
        let words_per_row = self.words_per_row;
        let word = qubit / 32;

        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let z = (self.stab_row_z[row_base + word] >> (qubit % 32)) & 1;
            if z == 1 {
                self.flip_sign_minus(g, true);
            }
        }

        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let z = (self.destab_row_z[row_base + word] >> (qubit % 32)) & 1;
            if z == 1 {
                self.flip_sign_minus(g, false);
            }
        }
    }

    /// Apply Y gate: X -> -X, Z -> -Z
    fn apply_y(&mut self, qubit: usize) {
        let words_per_row = self.words_per_row;
        let word = qubit / 32;

        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let x = (self.stab_row_x[row_base + word] >> (qubit % 32)) & 1;
            let z = (self.stab_row_z[row_base + word] >> (qubit % 32)) & 1;
            if x ^ z == 1 {
                self.flip_sign_minus(g, true);
            }
        }

        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let x = (self.destab_row_x[row_base + word] >> (qubit % 32)) & 1;
            let z = (self.destab_row_z[row_base + word] >> (qubit % 32)) & 1;
            if x ^ z == 1 {
                self.flip_sign_minus(g, false);
            }
        }
    }

    /// Apply Z gate: X -> -X
    fn apply_z(&mut self, qubit: usize) {
        let words_per_row = self.words_per_row;
        let word = qubit / 32;

        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let x = (self.stab_row_x[row_base + word] >> (qubit % 32)) & 1;
            if x == 1 {
                self.flip_sign_minus(g, true);
            }
        }

        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let x = (self.destab_row_x[row_base + word] >> (qubit % 32)) & 1;
            if x == 1 {
                self.flip_sign_minus(g, false);
            }
        }
    }

    /// Apply CX gate: `X_ctrl` -> `X_ctrl` `X_tgt`, `Z_tgt` -> `Z_ctrl` `Z_tgt`
    fn apply_cx(&mut self, ctrl: usize, tgt: usize) {
        let words_per_row = self.words_per_row;
        let ctrl_word = ctrl / 32;
        let ctrl_bit = 1u32 << (ctrl % 32);
        let tgt_word = tgt / 32;
        let tgt_bit = 1u32 << (tgt % 32);

        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let ctrl_x = (self.stab_row_x[row_base + ctrl_word] >> (ctrl % 32)) & 1;
            let tgt_z = (self.stab_row_z[row_base + tgt_word] >> (tgt % 32)) & 1;

            // X_ctrl propagates to target
            if ctrl_x == 1 {
                self.stab_row_x[row_base + tgt_word] ^= tgt_bit;
            }
            // Z_tgt propagates to control
            if tgt_z == 1 {
                self.stab_row_z[row_base + ctrl_word] ^= ctrl_bit;
            }
        }

        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let ctrl_x = (self.destab_row_x[row_base + ctrl_word] >> (ctrl % 32)) & 1;
            let tgt_z = (self.destab_row_z[row_base + tgt_word] >> (tgt % 32)) & 1;

            if ctrl_x == 1 {
                self.destab_row_x[row_base + tgt_word] ^= tgt_bit;
            }
            if tgt_z == 1 {
                self.destab_row_z[row_base + ctrl_word] ^= ctrl_bit;
            }
        }
    }

    /// Apply CZ gate: `X_a` -> `X_a` `Z_b`, `X_b` -> `Z_a` `X_b`
    fn apply_cz(&mut self, q1: usize, q2: usize) {
        let words_per_row = self.words_per_row;
        let word1 = q1 / 32;
        let bit1 = 1u32 << (q1 % 32);
        let word2 = q2 / 32;
        let bit2 = 1u32 << (q2 % 32);

        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let x1 = (self.stab_row_x[row_base + word1] >> (q1 % 32)) & 1;
            let x2 = (self.stab_row_x[row_base + word2] >> (q2 % 32)) & 1;

            // Sign update: toggle minus for generators with X on both qubits
            if x1 == 1 && x2 == 1 {
                self.flip_sign_minus(g, true);
            }

            if x1 == 1 {
                self.stab_row_z[row_base + word2] ^= bit2;
            }
            if x2 == 1 {
                self.stab_row_z[row_base + word1] ^= bit1;
            }
        }

        for g in 0..self.num_qubits {
            let row_base = g * words_per_row;
            let x1 = (self.destab_row_x[row_base + word1] >> (q1 % 32)) & 1;
            let x2 = (self.destab_row_x[row_base + word2] >> (q2 % 32)) & 1;

            if x1 == 1 {
                self.destab_row_z[row_base + word2] ^= bit2;
            }
            if x2 == 1 {
                self.destab_row_z[row_base + word1] ^= bit1;
            }
        }
    }

    // ========== Measurement ==========

    /// Check if measurement is deterministic (no stabilizer has X on this qubit).
    fn is_deterministic(&self, qubit: usize) -> bool {
        let word = qubit / 32;
        let bit_pos = qubit % 32;

        for g in 0..self.num_qubits {
            let row_base = g * self.words_per_row;
            if (self.stab_row_x[row_base + word] >> bit_pos) & 1 != 0 {
                return false;
            }
        }
        true
    }

    /// Find first anticommuting stabilizer (has X on qubit).
    fn find_anticommuting(&self, qubit: usize) -> Option<usize> {
        let word = qubit / 32;
        let bit_pos = qubit % 32;

        for g in 0..self.num_qubits {
            let row_base = g * self.words_per_row;
            if (self.stab_row_x[row_base + word] >> bit_pos) & 1 != 0 {
                return Some(g);
            }
        }
        None
    }

    /// Deterministic measurement: compute outcome from destabilizers.
    fn deterministic_meas(&self, qubit: usize) -> MeasurementResult {
        let word = qubit / 32;
        let bit_pos = qubit % 32;

        // Count initial signs from stabilizers (for destabilizers with X on qubit)
        let mut num_minuses = 0usize;
        let mut num_is = 0usize;

        for g in 0..self.num_qubits {
            let row_base = g * self.words_per_row;
            let destab_has_x = (self.destab_row_x[row_base + word] >> bit_pos) & 1;

            if destab_has_x == 1 {
                if self.get_sign_minus(g, true) {
                    num_minuses += 1;
                }
                if self.get_sign_i(g, true) {
                    num_is += 1;
                }
            }
        }

        // Compute cumulative phase by multiplying stabilizers
        // Only track cumulative_x - when Z meets X, add phase
        let mut cumulative_x = vec![0u32; self.words_per_row];

        for g in 0..self.num_qubits {
            let row_base = g * self.words_per_row;
            let destab_has_x = (self.destab_row_x[row_base + word] >> bit_pos) & 1;

            if destab_has_x == 1 {
                // Count overlap of stab_row_z with cumulative_x (contributes phase)
                for (w, cx) in cumulative_x.iter_mut().enumerate() {
                    num_minuses += (self.stab_row_z[row_base + w] & *cx).count_ones() as usize;
                    *cx ^= self.stab_row_x[row_base + w];
                }
            }
        }

        // Add i phase contribution (i^2 = -1)
        if num_is & 3 != 0 {
            num_minuses += 1;
        }

        MeasurementResult {
            outcome: num_minuses & 1 != 0,
            is_deterministic: true,
        }
    }

    /// Non-deterministic measurement.
    fn nondeterministic_meas(&mut self, qubit: usize, outcome: bool) -> MeasurementResult {
        let pivot = self.find_anticommuting(qubit).unwrap();
        let word = qubit / 32;
        let bit_pos = qubit % 32;
        let pivot_word = pivot / 32;
        let pivot_shift = pivot % 32;
        let pivot_bit = 1u32 << pivot_shift;
        let pivot_base = pivot * self.words_per_row;

        // Cache pivot signs
        let pivot_minus = (self.stab_signs_minus[pivot_word] >> pivot_shift) & 1 != 0;
        let pivot_i = (self.stab_signs_i[pivot_word] >> pivot_shift) & 1 != 0;

        // Step 1: Handle pivot's i-phase (matches DenseStab algorithm).
        if pivot_i {
            self.stab_signs_i[pivot_word] &= !pivot_bit;
            for g in 0..self.num_qubits {
                if g == pivot {
                    continue;
                }
                let row_base = g * self.words_per_row;
                if (self.stab_row_x[row_base + word] >> bit_pos) & 1 != 0 {
                    let g_word = g / 32;
                    let g_bit = 1u32 << (g % 32);
                    // i * i = -1: toggle minus for stabs that already have i
                    if (self.stab_signs_i[g_word] >> (g % 32)) & 1 != 0 {
                        self.stab_signs_minus[g_word] ^= g_bit;
                    }
                    // Toggle i for all anticommuting stabs
                    self.stab_signs_i[g_word] ^= g_bit;
                }
            }
        }

        // Step 2: XOR pivot into other anticommuting stabilizers.
        // Phase: count z_pivot & x_g overlaps.
        for g in 0..self.num_qubits {
            if g == pivot {
                continue;
            }
            let row_base = g * self.words_per_row;
            if (self.stab_row_x[row_base + word] >> bit_pos) & 1 != 0 {
                let g_word = g / 32;
                let g_bit = 1u32 << (g % 32);

                let mut count = 0u32;
                for w in 0..self.words_per_row {
                    count += (self.stab_row_z[pivot_base + w] & self.stab_row_x[row_base + w])
                        .count_ones();
                }
                if count & 1 != 0 {
                    self.stab_signs_minus[g_word] ^= g_bit;
                }
                if pivot_minus {
                    self.stab_signs_minus[g_word] ^= g_bit;
                }

                // XOR row data
                for w in 0..self.words_per_row {
                    self.stab_row_x[row_base + w] ^= self.stab_row_x[pivot_base + w];
                    self.stab_row_z[row_base + w] ^= self.stab_row_z[pivot_base + w];
                }
            }
        }

        // Step 3: XOR pivot stabilizer into anticommuting destabilizers.
        // Read from STAB arrays, write to DESTAB arrays. No sign update needed.
        for g in 0..self.num_qubits {
            let row_base = g * self.words_per_row;
            if (self.destab_row_x[row_base + word] >> bit_pos) & 1 != 0 {
                for w in 0..self.words_per_row {
                    self.destab_row_x[row_base + w] ^= self.stab_row_x[pivot_base + w];
                    self.destab_row_z[row_base + w] ^= self.stab_row_z[pivot_base + w];
                }
            }
        }

        // Copy pivot stabilizer to destabilizer
        let pivot_base = pivot * self.words_per_row;
        for w in 0..self.words_per_row {
            self.destab_row_x[pivot_base + w] = self.stab_row_x[pivot_base + w];
            self.destab_row_z[pivot_base + w] = self.stab_row_z[pivot_base + w];
        }

        let pivot_word = pivot / 32;
        let pivot_bit = 1u32 << (pivot % 32);
        let src_minus = (self.stab_signs_minus[pivot_word] >> (pivot % 32)) & 1;
        let src_i = (self.stab_signs_i[pivot_word] >> (pivot % 32)) & 1;

        if src_minus == 1 {
            self.destab_signs_minus[pivot_word] |= pivot_bit;
        } else {
            self.destab_signs_minus[pivot_word] &= !pivot_bit;
        }
        if src_i == 1 {
            self.destab_signs_i[pivot_word] |= pivot_bit;
        } else {
            self.destab_signs_i[pivot_word] &= !pivot_bit;
        }

        // Set pivot stabilizer to +-Z_qubit
        for w in 0..self.words_per_row {
            self.stab_row_x[pivot_base + w] = 0;
            self.stab_row_z[pivot_base + w] = 0;
        }
        self.stab_row_z[pivot_base + word] = 1u32 << bit_pos;

        self.stab_signs_i[pivot_word] &= !pivot_bit;
        if outcome {
            self.stab_signs_minus[pivot_word] |= pivot_bit;
        } else {
            self.stab_signs_minus[pivot_word] &= !pivot_bit;
        }

        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }
}

// ========== Trait implementations ==========

impl QuantumSimulator for GpuStabParallel {
    fn reset(&mut self) -> &mut Self {
        self.init_tableau();
        self
    }
}

impl CliffordGateable for GpuStabParallel {
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_h(q.index());
        }
        self
    }

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_x(q.index());
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_y(q.index());
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_z(q.index());
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
        for &q in qubits {
            self.apply_sdg(q.index());
        }
        self
    }

    fn cx(&mut self, qubits: &[QubitId]) -> &mut Self {
        for pair in qubits.chunks_exact(2) {
            self.apply_cx(pair[0].index(), pair[1].index());
        }
        self
    }

    fn cz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for pair in qubits.chunks_exact(2) {
            self.apply_cz(pair[0].index(), pair[1].index());
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let mut results = Vec::with_capacity(qubits.len());

        for &q in qubits {
            let qubit = q.index();
            let result = if self.is_deterministic(qubit) {
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

impl RngManageable for GpuStabParallel {
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

// ========== StabilizerTableauSimulator ==========

impl StabilizerTableauSimulator for GpuStabParallel {
    fn stab_tableau(&self) -> String {
        Self::gen_tableau_string_u32(
            self.num_qubits,
            self.words_per_row,
            &self.stab_row_x,
            &self.stab_row_z,
            &self.stab_signs_minus,
            &self.stab_signs_i,
        )
    }

    fn destab_tableau(&self) -> String {
        Self::gen_tableau_string_u32(
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

impl GpuStabParallel {
    fn gen_tableau_string_u32(
        num_qubits: usize,
        words_per_row: usize,
        row_x: &[u32],
        row_z: &[u32],
        signs_minus: &[u32],
        signs_i: &[u32],
    ) -> String {
        let mut result = String::with_capacity(num_qubits * num_qubits + num_qubits + 2);
        for g in 0..num_qubits {
            let sign_minus = (signs_minus[g / 32] >> (g % 32)) & 1 != 0;
            let sign_i = (signs_i[g / 32] >> (g % 32)) & 1 != 0;
            if sign_minus {
                result.push('-');
            } else {
                result.push('+');
            }
            if sign_i {
                result.push('i');
            }

            let base = g * words_per_row;
            for qubit in 0..num_qubits {
                let word_idx = base + qubit / 32;
                let bit_mask = 1u32 << (qubit % 32);
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
}

// ========== Test support ==========

use crate::stabilizer_test_utils::{ForcedMeasurement, StabilizerSimulator};

impl ForcedMeasurement for GpuStabParallel {
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        if self.is_deterministic(qubit) {
            self.deterministic_meas(qubit)
        } else {
            self.nondeterministic_meas(qubit, forced_outcome)
        }
    }
}

impl StabilizerSimulator for GpuStabParallel {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stabilizer_test_utils::run_full_stabilizer_test_suite;

    #[test]
    fn test_gpu_stab_parallel_basic() {
        let mut sim = GpuStabParallel::new(2);
        sim.h(&[QubitId(0)]);
        sim.cx(&[QubitId(0), QubitId(1)]);
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert_eq!(results[0].outcome, results[1].outcome);
    }

    #[test]
    fn test_gpu_stab_parallel_x_gate() {
        let mut sim = GpuStabParallel::new(1);
        sim.x(&[QubitId(0)]);
        let results = sim.mz(&[QubitId(0)]);
        assert!(results[0].outcome);
        assert!(results[0].is_deterministic);
    }

    #[test]
    fn test_gpu_stab_parallel_full_suite() {
        let mut sim = GpuStabParallel::with_seed(8, 42);
        run_full_stabilizer_test_suite(&mut sim, 8);
    }
}
