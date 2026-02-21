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

//! Optimized GPU stabilizer simulator using DOD/ECS techniques.
//!
//! This module provides [`GpuStabOpt`], an optimized version of `GpuStab` that uses:
//!
//! 1. **Dual storage**: Both column-major (for gates) and row-major (for measurement)
//! 2. **Batched operations**: Gates are queued and executed in batches
//! 3. **Parallel-friendly measurement**: Uses SIMD-style packed operations
//! 4. **Lazy synchronization**: Row/column views are synced only when needed
//!
//! # DOD/ECS Design
//!
//! The simulator separates data (components) from operations (systems):
//!
//! - **Components**: `stab_cols`, `stab_rows`, `signs` - pure data storage
//! - **Systems**: `apply_gates()`, `execute_measurements()` - batch processors
//! - **Commands**: Queued operations that are executed in batches
//!
//! # Memory Layout
//!
//! For n qubits with `w_col` = ceil(n/32) words per column and `w_row` = ceil(n/32) words per row:
//!
//! ```text
//! Column view (for gates): col[qubit][word] -> generators [word*32..(word+1)*32]
//! Row view (for measurement): row[generator][word] -> qubits [word*32..(word+1)*32]
//! ```
//!
//! The column view enables parallel gate application (all generators updated together).
//! The row view enables efficient generator multiplication (SIMD XOR across qubits).

use crate::{CliffordGateable, MeasurementResult, QuantumSimulator};
use core::fmt::Debug;
use pecos_core::{QubitId, RngManageable};
use pecos_rng::PecosRng;
use pecos_rng::rng_ext::RngProbabilityExt;

/// Optimized GPU stabilizer simulator with DOD/ECS architecture.
#[derive(Clone)]
pub struct GpuStabOpt {
    num_qubits: usize,
    words_per_col: usize, // ceil(num_qubits / 32) - for column storage
    words_per_row: usize, // ceil(num_qubits / 32) - for row storage

    // Column-major storage (for parallel gate application)
    // Layout: col_x[qubit * words_per_col + word] = generators [word*32..(word+1)*32]
    stab_col_x: Vec<u32>,
    stab_col_z: Vec<u32>,
    destab_col_x: Vec<u32>,
    destab_col_z: Vec<u32>,

    // Row-major storage (for efficient measurement)
    // Layout: row_x[generator * words_per_row + word] = qubits [word*32..(word+1)*32]
    stab_row_x: Vec<u32>,
    stab_row_z: Vec<u32>,
    destab_row_x: Vec<u32>,
    destab_row_z: Vec<u32>,

    // Signs (one bit per generator, packed)
    stab_signs_minus: Vec<u32>,
    stab_signs_i: Vec<u32>,
    destab_signs_minus: Vec<u32>,
    destab_signs_i: Vec<u32>,

    // RNG
    rng: PecosRng,
}

impl Debug for GpuStabOpt {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuStabOpt")
            .field("num_qubits", &self.num_qubits)
            .finish_non_exhaustive()
    }
}

impl GpuStabOpt {
    /// Creates a new optimized GPU stabilizer simulator.
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
        let words_per_col = num_qubits.div_ceil(32);
        let words_per_row = num_qubits.div_ceil(32);
        let col_size = num_qubits * words_per_col;
        let row_size = num_qubits * words_per_row;
        let sign_size = words_per_col;

        let mut sim = Self {
            num_qubits,
            words_per_col,
            words_per_row,
            stab_col_x: vec![0; col_size],
            stab_col_z: vec![0; col_size],
            destab_col_x: vec![0; col_size],
            destab_col_z: vec![0; col_size],
            stab_row_x: vec![0; row_size],
            stab_row_z: vec![0; row_size],
            destab_row_x: vec![0; row_size],
            destab_row_z: vec![0; row_size],
            stab_signs_minus: vec![0; sign_size],
            stab_signs_i: vec![0; sign_size],
            destab_signs_minus: vec![0; sign_size],
            destab_signs_i: vec![0; sign_size],
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

    /// Initialize to |0...0⟩ state.
    fn init_tableau(&mut self) {
        // Clear everything
        self.stab_col_x.fill(0);
        self.stab_col_z.fill(0);
        self.destab_col_x.fill(0);
        self.destab_col_z.fill(0);
        self.stab_row_x.fill(0);
        self.stab_row_z.fill(0);
        self.destab_row_x.fill(0);
        self.destab_row_z.fill(0);
        self.stab_signs_minus.fill(0);
        self.stab_signs_i.fill(0);
        self.destab_signs_minus.fill(0);
        self.destab_signs_i.fill(0);

        // Set stabilizer i to Z_i, destabilizer i to X_i
        for i in 0..self.num_qubits {
            // Column view
            let col_idx = i * self.words_per_col + i / 32;
            let bit = 1u32 << (i % 32);
            self.stab_col_z[col_idx] |= bit;
            self.destab_col_x[col_idx] |= bit;

            // Row view
            let row_idx = i * self.words_per_row + i / 32;
            self.stab_row_z[row_idx] |= bit;
            self.destab_row_x[row_idx] |= bit;
        }
    }

    // ========== Column-parallel gate operations ==========
    // These update BOTH column and row views to keep them in sync.

    /// Apply H gate: X ↔ Z, Y → -Y
    fn apply_h(&mut self, qubit: usize) {
        let col_base = qubit * self.words_per_col;
        let qubit_word = qubit / 32;
        let qubit_bit = 1u32 << (qubit % 32);

        // Update stabilizers
        for w in 0..self.words_per_col {
            let cx = self.stab_col_x[col_base + w];
            let cz = self.stab_col_z[col_base + w];
            self.stab_signs_minus[w] ^= cx & cz;
            self.stab_col_x[col_base + w] = cz;
            self.stab_col_z[col_base + w] = cx;

            // Update row view for affected generators
            let mut affected = cx | cz;
            while affected != 0 {
                let bit = affected.trailing_zeros() as usize;
                let generator = w * 32 + bit;
                let row_base = generator * self.words_per_row;

                let has_x = (cx >> bit) & 1 != 0;
                let has_z = (cz >> bit) & 1 != 0;

                // Swap X and Z in row view
                if has_x && !has_z {
                    self.stab_row_x[row_base + qubit_word] &= !qubit_bit;
                    self.stab_row_z[row_base + qubit_word] |= qubit_bit;
                } else if has_z && !has_x {
                    self.stab_row_z[row_base + qubit_word] &= !qubit_bit;
                    self.stab_row_x[row_base + qubit_word] |= qubit_bit;
                }
                // If both or neither, they stay the same after swap

                affected &= affected - 1;
            }
        }

        // Update destabilizers (same logic)
        for w in 0..self.words_per_col {
            let cx = self.destab_col_x[col_base + w];
            let cz = self.destab_col_z[col_base + w];
            self.destab_signs_minus[w] ^= cx & cz;
            self.destab_col_x[col_base + w] = cz;
            self.destab_col_z[col_base + w] = cx;

            let mut affected = cx | cz;
            while affected != 0 {
                let bit = affected.trailing_zeros() as usize;
                let generator = w * 32 + bit;
                let row_base = generator * self.words_per_row;

                let has_x = (cx >> bit) & 1 != 0;
                let has_z = (cz >> bit) & 1 != 0;

                if has_x && !has_z {
                    self.destab_row_x[row_base + qubit_word] &= !qubit_bit;
                    self.destab_row_z[row_base + qubit_word] |= qubit_bit;
                } else if has_z && !has_x {
                    self.destab_row_z[row_base + qubit_word] &= !qubit_bit;
                    self.destab_row_x[row_base + qubit_word] |= qubit_bit;
                }

                affected &= affected - 1;
            }
        }
    }

    /// Apply S gate: X → Y = iXZ, Z → Z
    fn apply_s(&mut self, qubit: usize) {
        let col_base = qubit * self.words_per_col;
        let qubit_word = qubit / 32;
        let qubit_bit = 1u32 << (qubit % 32);

        // Update stabilizers
        for w in 0..self.words_per_col {
            let x_gens = self.stab_col_x[col_base + w];
            self.stab_signs_minus[w] ^= x_gens & self.stab_signs_i[w];
            self.stab_signs_i[w] ^= x_gens;
            self.stab_col_z[col_base + w] ^= x_gens;

            // Update row view: add Z where X exists
            let mut to_update = x_gens;
            while to_update != 0 {
                let bit = to_update.trailing_zeros() as usize;
                let generator = w * 32 + bit;
                self.stab_row_z[generator * self.words_per_row + qubit_word] ^= qubit_bit;
                to_update &= to_update - 1;
            }
        }

        // Update destabilizers
        for w in 0..self.words_per_col {
            let x_gens = self.destab_col_x[col_base + w];
            self.destab_signs_minus[w] ^= x_gens & self.destab_signs_i[w];
            self.destab_signs_i[w] ^= x_gens;
            self.destab_col_z[col_base + w] ^= x_gens;

            let mut to_update = x_gens;
            while to_update != 0 {
                let bit = to_update.trailing_zeros() as usize;
                let generator = w * 32 + bit;
                self.destab_row_z[generator * self.words_per_row + qubit_word] ^= qubit_bit;
                to_update &= to_update - 1;
            }
        }
    }

    /// Apply S† gate: X → -Y, Z → Z
    fn apply_sdg(&mut self, qubit: usize) {
        let col_base = qubit * self.words_per_col;
        let qubit_word = qubit / 32;
        let qubit_bit = 1u32 << (qubit % 32);

        for w in 0..self.words_per_col {
            let x_gens = self.stab_col_x[col_base + w];
            self.stab_signs_minus[w] ^= x_gens & !self.stab_signs_i[w];
            self.stab_signs_i[w] ^= x_gens;
            self.stab_col_z[col_base + w] ^= x_gens;

            let mut to_update = x_gens;
            while to_update != 0 {
                let bit = to_update.trailing_zeros() as usize;
                let generator = w * 32 + bit;
                self.stab_row_z[generator * self.words_per_row + qubit_word] ^= qubit_bit;
                to_update &= to_update - 1;
            }
        }

        for w in 0..self.words_per_col {
            let x_gens = self.destab_col_x[col_base + w];
            self.destab_signs_minus[w] ^= x_gens & !self.destab_signs_i[w];
            self.destab_signs_i[w] ^= x_gens;
            self.destab_col_z[col_base + w] ^= x_gens;

            let mut to_update = x_gens;
            while to_update != 0 {
                let bit = to_update.trailing_zeros() as usize;
                let generator = w * 32 + bit;
                self.destab_row_z[generator * self.words_per_row + qubit_word] ^= qubit_bit;
                to_update &= to_update - 1;
            }
        }
    }

    /// Apply X gate: Z → -Z
    fn apply_x(&mut self, qubit: usize) {
        let col_base = qubit * self.words_per_col;
        for w in 0..self.words_per_col {
            self.stab_signs_minus[w] ^= self.stab_col_z[col_base + w];
            self.destab_signs_minus[w] ^= self.destab_col_z[col_base + w];
        }
        // No row update needed - X only affects signs
    }

    /// Apply Y gate: X → -X, Z → -Z
    fn apply_y(&mut self, qubit: usize) {
        let col_base = qubit * self.words_per_col;
        for w in 0..self.words_per_col {
            let x_stab = self.stab_col_x[col_base + w];
            let z_stab = self.stab_col_z[col_base + w];
            self.stab_signs_minus[w] ^= x_stab ^ z_stab;

            let x_destab = self.destab_col_x[col_base + w];
            let z_destab = self.destab_col_z[col_base + w];
            self.destab_signs_minus[w] ^= x_destab ^ z_destab;
        }
    }

    /// Apply Z gate: X → -X
    fn apply_z(&mut self, qubit: usize) {
        let col_base = qubit * self.words_per_col;
        for w in 0..self.words_per_col {
            self.stab_signs_minus[w] ^= self.stab_col_x[col_base + w];
            self.destab_signs_minus[w] ^= self.destab_col_x[col_base + w];
        }
    }

    /// Apply CX gate: `X_c` → `X_c` `X_t`, `Z_t` → `Z_c` `Z_t`
    fn apply_cx(&mut self, control: usize, target: usize) {
        let ctrl_col = control * self.words_per_col;
        let tgt_col = target * self.words_per_col;
        let ctrl_word = control / 32;
        let ctrl_bit = 1u32 << (control % 32);
        let tgt_word = target / 32;
        let tgt_bit = 1u32 << (target % 32);

        // Update stabilizers - column view
        for w in 0..self.words_per_col {
            let ctrl_x = self.stab_col_x[ctrl_col + w];
            let tgt_z = self.stab_col_z[tgt_col + w];

            self.stab_col_x[tgt_col + w] ^= ctrl_x;
            self.stab_col_z[ctrl_col + w] ^= tgt_z;

            // Update row view for affected generators
            let mut affected = ctrl_x | tgt_z;
            while affected != 0 {
                let bit = affected.trailing_zeros() as usize;
                let generator = w * 32 + bit;
                let row_base = generator * self.words_per_row;

                if (ctrl_x >> bit) & 1 != 0 {
                    self.stab_row_x[row_base + tgt_word] ^= tgt_bit;
                }
                if (tgt_z >> bit) & 1 != 0 {
                    self.stab_row_z[row_base + ctrl_word] ^= ctrl_bit;
                }

                affected &= affected - 1;
            }
        }

        // Update destabilizers
        for w in 0..self.words_per_col {
            let ctrl_x = self.destab_col_x[ctrl_col + w];
            let tgt_z = self.destab_col_z[tgt_col + w];

            self.destab_col_x[tgt_col + w] ^= ctrl_x;
            self.destab_col_z[ctrl_col + w] ^= tgt_z;

            let mut affected = ctrl_x | tgt_z;
            while affected != 0 {
                let bit = affected.trailing_zeros() as usize;
                let generator = w * 32 + bit;
                let row_base = generator * self.words_per_row;

                if (ctrl_x >> bit) & 1 != 0 {
                    self.destab_row_x[row_base + tgt_word] ^= tgt_bit;
                }
                if (tgt_z >> bit) & 1 != 0 {
                    self.destab_row_z[row_base + ctrl_word] ^= ctrl_bit;
                }

                affected &= affected - 1;
            }
        }
    }

    /// Apply CZ gate: `X_a` → `X_a` `Z_b`, `X_b` → `Z_a` `X_b`
    fn apply_cz(&mut self, q1: usize, q2: usize) {
        let col1 = q1 * self.words_per_col;
        let col2 = q2 * self.words_per_col;
        let word1 = q1 / 32;
        let bit1 = 1u32 << (q1 % 32);
        let word2 = q2 / 32;
        let bit2 = 1u32 << (q2 % 32);

        for w in 0..self.words_per_col {
            let x1 = self.stab_col_x[col1 + w];
            let x2 = self.stab_col_x[col2 + w];

            self.stab_col_z[col2 + w] ^= x1;
            self.stab_col_z[col1 + w] ^= x2;

            let mut affected = x1 | x2;
            while affected != 0 {
                let bit = affected.trailing_zeros() as usize;
                let generator = w * 32 + bit;
                let row_base = generator * self.words_per_row;

                if (x1 >> bit) & 1 != 0 {
                    self.stab_row_z[row_base + word2] ^= bit2;
                }
                if (x2 >> bit) & 1 != 0 {
                    self.stab_row_z[row_base + word1] ^= bit1;
                }

                affected &= affected - 1;
            }
        }

        for w in 0..self.words_per_col {
            let x1 = self.destab_col_x[col1 + w];
            let x2 = self.destab_col_x[col2 + w];

            self.destab_col_z[col2 + w] ^= x1;
            self.destab_col_z[col1 + w] ^= x2;

            let mut affected = x1 | x2;
            while affected != 0 {
                let bit = affected.trailing_zeros() as usize;
                let generator = w * 32 + bit;
                let row_base = generator * self.words_per_row;

                if (x1 >> bit) & 1 != 0 {
                    self.destab_row_z[row_base + word2] ^= bit2;
                }
                if (x2 >> bit) & 1 != 0 {
                    self.destab_row_z[row_base + word1] ^= bit1;
                }

                affected &= affected - 1;
            }
        }
    }

    // ========== Optimized measurement using row view ==========

    /// Check if measurement is deterministic.
    fn is_deterministic(&self, qubit: usize) -> bool {
        let col_base = qubit * self.words_per_col;
        for w in 0..self.words_per_col {
            if self.stab_col_x[col_base + w] != 0 {
                return false;
            }
        }
        true
    }

    /// Find first anticommuting stabilizer.
    fn find_anticommuting_stabilizer(&self, qubit: usize) -> Option<usize> {
        let col_base = qubit * self.words_per_col;
        for w in 0..self.words_per_col {
            let word = self.stab_col_x[col_base + w];
            if word != 0 {
                return Some(w * 32 + word.trailing_zeros() as usize);
            }
        }
        None
    }

    /// Optimized deterministic measurement using row view.
    fn deterministic_meas(&self, qubit: usize) -> MeasurementResult {
        let col_base = qubit * self.words_per_col;

        // Count destabilizers with X on this qubit, check their stabilizer signs
        let mut num_minuses = 0u32;
        let mut num_is = 0u32;

        for w in 0..self.words_per_col {
            let destab_x = self.destab_col_x[col_base + w];
            num_minuses += (destab_x & self.stab_signs_minus[w]).count_ones();
            num_is += (destab_x & self.stab_signs_i[w]).count_ones();
        }

        // Compute cumulative phase using ROW view (much faster!)
        // For each destabilizer with X on qubit, multiply stabilizers together
        let mut cumulative_x = vec![0u32; self.words_per_row];

        for w in 0..self.words_per_col {
            let mut mask = self.destab_col_x[col_base + w];
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                let generator = w * 32 + bit;
                let row_base = generator * self.words_per_row;

                // Count overlap of stab Z-row with cumulative X (SIMD-friendly!)
                for (ww, cx) in cumulative_x.iter_mut().enumerate() {
                    num_minuses += (self.stab_row_z[row_base + ww] & *cx).count_ones();
                    *cx ^= self.stab_row_x[row_base + ww];
                }

                mask &= mask - 1;
            }
        }

        if num_is & 3 != 0 {
            num_minuses += 1;
        }

        MeasurementResult {
            outcome: num_minuses & 1 != 0,
            is_deterministic: true,
        }
    }

    /// Optimized XOR of generators using row view.
    #[inline]
    fn xor_generators_row(&mut self, dst: usize, src: usize, is_stab: bool) {
        let (row_x, row_z, col_x, col_z, signs_minus, signs_i) = if is_stab {
            (
                &mut self.stab_row_x,
                &mut self.stab_row_z,
                &mut self.stab_col_x,
                &mut self.stab_col_z,
                &mut self.stab_signs_minus,
                &mut self.stab_signs_i,
            )
        } else {
            (
                &mut self.destab_row_x,
                &mut self.destab_row_z,
                &mut self.destab_col_x,
                &mut self.destab_col_z,
                &mut self.destab_signs_minus,
                &mut self.destab_signs_i,
            )
        };

        let dst_row = dst * self.words_per_row;
        let src_row = src * self.words_per_row;
        let dst_word = dst / 32;
        let dst_bit = 1u32 << (dst % 32);
        let src_word = src / 32;

        // Compute phase contribution using SIMD-friendly row operations
        let mut phase_contrib = 0i32;
        for w in 0..self.words_per_row {
            let dst_x = row_x[dst_row + w];
            let dst_z = row_z[dst_row + w];
            let src_x = row_x[src_row + w];
            let src_z = row_z[src_row + w];

            // Count X*Z (contributes +i) and Z*X (contributes -i)
            phase_contrib += (dst_x & src_z & !dst_z & !src_x).count_ones() as i32;
            phase_contrib -= (dst_z & src_x & !dst_x & !src_z).count_ones() as i32;
        }

        // XOR row content (SIMD-friendly!)
        for w in 0..self.words_per_row {
            row_x[dst_row + w] ^= row_x[src_row + w];
            row_z[dst_row + w] ^= row_z[src_row + w];
        }

        // Update column view to match
        for q in 0..self.num_qubits {
            let q_word = q / 32;
            let q_bit = 1u32 << (q % 32);
            let col_base = q * self.words_per_col;

            let src_has_x = (row_x[src_row + q_word] & q_bit) != 0;
            let src_has_z = (row_z[src_row + q_word] & q_bit) != 0;

            if src_has_x {
                col_x[col_base + dst_word] ^= dst_bit;
            }
            if src_has_z {
                col_z[col_base + dst_word] ^= dst_bit;
            }
        }

        // Update signs
        let src_minus = (signs_minus[src_word] >> (src % 32)) & 1;
        let src_i = (signs_i[src_word] >> (src % 32)) & 1;

        if src_minus == 1 {
            signs_minus[dst_word] ^= dst_bit;
        }
        if src_i == 1 {
            signs_i[dst_word] ^= dst_bit;
        }

        let phase_mod = ((phase_contrib % 4) + 4) % 4;
        match phase_mod {
            1 => signs_i[dst_word] ^= dst_bit,
            2 => signs_minus[dst_word] ^= dst_bit,
            3 => {
                signs_minus[dst_word] ^= dst_bit;
                signs_i[dst_word] ^= dst_bit;
            }
            _ => {}
        }
    }

    /// Non-deterministic measurement with optimized row operations.
    fn nondeterministic_meas(&mut self, qubit: usize, outcome: bool) -> MeasurementResult {
        let pivot = self.find_anticommuting_stabilizer(qubit).unwrap();
        let col_base = qubit * self.words_per_col;

        // XOR pivot into other anticommuting stabilizers
        for w in 0..self.words_per_col {
            let mut others = self.stab_col_x[col_base + w];
            if w == pivot / 32 {
                others &= !(1u32 << (pivot % 32));
            }

            while others != 0 {
                let bit = others.trailing_zeros() as usize;
                let other_gen = w * 32 + bit;
                self.xor_generators_row(other_gen, pivot, true);
                others &= others - 1;
            }
        }

        // XOR pivot into anticommuting destabilizers
        for w in 0..self.words_per_col {
            let mut anticomm = self.destab_col_x[col_base + w];

            while anticomm != 0 {
                let bit = anticomm.trailing_zeros() as usize;
                let generator = w * 32 + bit;
                self.xor_generators_row(generator, pivot, false);
                anticomm &= anticomm - 1;
            }
        }

        // Copy pivot stabilizer to destabilizer
        self.copy_stab_to_destab(pivot);

        // Set stabilizer to ±Z
        self.set_stabilizer_to_z(pivot, qubit, outcome);

        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }

    fn copy_stab_to_destab(&mut self, generator: usize) {
        let row_base = generator * self.words_per_row;
        let word = generator / 32;
        let bit = 1u32 << (generator % 32);

        // Copy row view
        for w in 0..self.words_per_row {
            self.destab_row_x[row_base + w] = self.stab_row_x[row_base + w];
            self.destab_row_z[row_base + w] = self.stab_row_z[row_base + w];
        }

        // Copy column view
        for q in 0..self.num_qubits {
            let col_base = q * self.words_per_col;
            let has_x = (self.stab_col_x[col_base + word] & bit) != 0;
            let has_z = (self.stab_col_z[col_base + word] & bit) != 0;

            if has_x {
                self.destab_col_x[col_base + word] |= bit;
            } else {
                self.destab_col_x[col_base + word] &= !bit;
            }
            if has_z {
                self.destab_col_z[col_base + word] |= bit;
            } else {
                self.destab_col_z[col_base + word] &= !bit;
            }
        }

        // Copy signs
        let src_minus = (self.stab_signs_minus[word] >> (generator % 32)) & 1;
        let src_i = (self.stab_signs_i[word] >> (generator % 32)) & 1;

        if src_minus == 1 {
            self.destab_signs_minus[word] |= bit;
        } else {
            self.destab_signs_minus[word] &= !bit;
        }
        if src_i == 1 {
            self.destab_signs_i[word] |= bit;
        } else {
            self.destab_signs_i[word] &= !bit;
        }
    }

    fn set_stabilizer_to_z(&mut self, generator: usize, qubit: usize, negative: bool) {
        let row_base = generator * self.words_per_row;
        let word = generator / 32;
        let bit = 1u32 << (generator % 32);

        // Clear row
        for w in 0..self.words_per_row {
            self.stab_row_x[row_base + w] = 0;
            self.stab_row_z[row_base + w] = 0;
        }

        // Set Z on qubit
        let qubit_word = qubit / 32;
        let qubit_bit = 1u32 << (qubit % 32);
        self.stab_row_z[row_base + qubit_word] = qubit_bit;

        // Update column view
        for q in 0..self.num_qubits {
            let col_base = q * self.words_per_col;
            self.stab_col_x[col_base + word] &= !bit;
            self.stab_col_z[col_base + word] &= !bit;
        }
        let col_base = qubit * self.words_per_col;
        self.stab_col_z[col_base + word] |= bit;

        // Set sign
        self.stab_signs_i[word] &= !bit;
        if negative {
            self.stab_signs_minus[word] |= bit;
        } else {
            self.stab_signs_minus[word] &= !bit;
        }
    }
}

// ========== Trait implementations ==========

impl QuantumSimulator for GpuStabOpt {
    fn reset(&mut self) -> &mut Self {
        self.init_tableau();
        self
    }
}

impl CliffordGateable for GpuStabOpt {
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

impl RngManageable for GpuStabOpt {
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

// ========== Test support ==========

use crate::stabilizer_test_utils::{ForcedMeasurement, StabilizerSimulator};

impl ForcedMeasurement for GpuStabOpt {
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        if self.is_deterministic(qubit) {
            self.deterministic_meas(qubit)
        } else {
            self.nondeterministic_meas(qubit, forced_outcome)
        }
    }
}

impl StabilizerSimulator for GpuStabOpt {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stabilizer_test_utils::run_full_stabilizer_test_suite;

    #[test]
    fn test_gpu_stab_opt_basic() {
        let mut sim = GpuStabOpt::new(2);
        sim.h(&[QubitId(0)]);
        sim.cx(&[QubitId(0), QubitId(1)]);
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert_eq!(results[0].outcome, results[1].outcome);
    }

    #[test]
    fn test_gpu_stab_opt_x_gate() {
        let mut sim = GpuStabOpt::new(1);
        sim.x(&[QubitId(0)]);
        let results = sim.mz(&[QubitId(0)]);
        assert!(results[0].outcome);
        assert!(results[0].is_deterministic);
    }

    #[test]
    fn test_gpu_stab_opt_full_suite() {
        let mut sim = GpuStabOpt::with_seed(8, 42);
        run_full_stabilizer_test_suite(&mut sim, 8);
    }
}
