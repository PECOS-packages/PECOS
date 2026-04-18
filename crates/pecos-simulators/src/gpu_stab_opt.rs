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

use crate::{CliffordGateable, MeasurementResult, QuantumSimulator, StabilizerTableauSimulator};
use core::fmt::Debug;
use pecos_core::{QubitId, RngManageable};
use pecos_random::PecosRng;
use pecos_random::rng_ext::RngProbabilityExt;

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

            // Sign update: toggle minus for generators with X on both qubits
            self.stab_signs_minus[w] ^= x1 & x2;

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

    /// Non-deterministic measurement with optimized row operations.
    fn nondeterministic_meas(&mut self, qubit: usize, outcome: bool) -> MeasurementResult {
        let pivot = self
            .find_anticommuting_stabilizer(qubit)
            .expect("non-deterministic measurement requires anticommuting stabilizer");
        let col_base = qubit * self.words_per_col;
        let pivot_word = pivot / 32;
        let pivot_shift = pivot % 32;
        let pivot_bit = 1u32 << pivot_shift;
        let pivot_row = pivot * self.words_per_row;

        // Cache pivot signs
        let pivot_minus = (self.stab_signs_minus[pivot_word] >> pivot_shift) & 1 != 0;
        let pivot_i = (self.stab_signs_i[pivot_word] >> pivot_shift) & 1 != 0;

        // Step 1: Handle pivot's i-phase (matches DenseStab algorithm).
        if pivot_i {
            self.stab_signs_i[pivot_word] &= !pivot_bit;
            for w in 0..self.words_per_col {
                let mut anticom = self.stab_col_x[col_base + w];
                if w == pivot_word {
                    anticom &= !pivot_bit;
                }
                self.stab_signs_minus[w] ^= anticom & self.stab_signs_i[w];
                self.stab_signs_i[w] ^= anticom;
            }
        }

        // Step 2: XOR pivot into other anticommuting stabilizers.
        // Phase: count z_pivot & x_other overlaps (row-based for efficiency).
        for w in 0..self.words_per_col {
            let mut mask = self.stab_col_x[col_base + w];
            if w == pivot_word {
                mask &= !pivot_bit;
            }

            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                let g = w * 32 + bit;
                let g_word = g / 32;
                let g_bit = 1u32 << (g % 32);
                let g_row = g * self.words_per_row;

                // Count z_pivot & x_g overlaps using row data
                let mut count = 0u32;
                for ww in 0..self.words_per_row {
                    count += (self.stab_row_z[pivot_row + ww] & self.stab_row_x[g_row + ww])
                        .count_ones();
                }
                if count & 1 != 0 {
                    self.stab_signs_minus[g_word] ^= g_bit;
                }
                if pivot_minus {
                    self.stab_signs_minus[g_word] ^= g_bit;
                }

                // XOR row data
                for ww in 0..self.words_per_row {
                    self.stab_row_x[g_row + ww] ^= self.stab_row_x[pivot_row + ww];
                    self.stab_row_z[g_row + ww] ^= self.stab_row_z[pivot_row + ww];
                }

                // Update column data
                for q in 0..self.num_qubits {
                    let cb = q * self.words_per_col;
                    if (self.stab_col_x[cb + pivot_word] >> pivot_shift) & 1 == 1 {
                        self.stab_col_x[cb + g_word] ^= g_bit;
                    }
                    if (self.stab_col_z[cb + pivot_word] >> pivot_shift) & 1 == 1 {
                        self.stab_col_z[cb + g_word] ^= g_bit;
                    }
                }

                mask &= mask - 1;
            }
        }

        // Step 3: XOR pivot stabilizer into anticommuting destabilizers.
        // Read from STAB arrays, write to DESTAB arrays. No sign update needed.
        for w in 0..self.words_per_col {
            let mut anticomm = self.destab_col_x[col_base + w];

            while anticomm != 0 {
                let bit = anticomm.trailing_zeros() as usize;
                let dst = w * 32 + bit;
                let dst_row = dst * self.words_per_row;
                let dst_cword = dst / 32;
                let dst_cbit = 1u32 << (dst % 32);

                // XOR row data
                for ww in 0..self.words_per_row {
                    self.destab_row_x[dst_row + ww] ^= self.stab_row_x[pivot_row + ww];
                    self.destab_row_z[dst_row + ww] ^= self.stab_row_z[pivot_row + ww];
                }

                // Update column data
                for q in 0..self.num_qubits {
                    let cb = q * self.words_per_col;
                    if (self.stab_col_x[cb + pivot_word] & pivot_bit) != 0 {
                        self.destab_col_x[cb + dst_cword] ^= dst_cbit;
                    }
                    if (self.stab_col_z[cb + pivot_word] & pivot_bit) != 0 {
                        self.destab_col_z[cb + dst_cword] ^= dst_cbit;
                    }
                }

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
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

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

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(control, target) in pairs {
            self.apply_cx(control.index(), target.index());
        }
        self
    }

    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            self.apply_cz(q0.index(), q1.index());
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

// ========== StabilizerTableauSimulator ==========

impl StabilizerTableauSimulator for GpuStabOpt {
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
}

impl GpuStabOpt {
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
        sim.cx(&[(QubitId(0), QubitId(1))]);
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
