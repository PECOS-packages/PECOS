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

//! GPU-optimized stabilizer simulator using column-only representation.
//!
//! This module provides [`GpuStab`], a stabilizer simulator designed for efficient
//! GPU execution. While this implementation runs on CPU, its memory layout and
//! algorithms are designed to be easily portable to GPU backends like wgpu, CUDA,
//! or Metal.
//!
//! # Design Principles
//!
//! 1. **32-bit words**: Uses `u32` instead of `u64` for better alignment with
//!    GPU warp sizes (32 threads per warp on NVIDIA).
//!
//! 2. **Column-only representation**: Stores only column-major data. For a gate
//!    on qubit q, all generators can be updated in parallel by reading column q.
//!
//! 3. **Batched operations**: Gates can be queued and executed in batches to
//!    amortize kernel launch overhead on actual GPU implementations.
//!
//! 4. **Parallel-friendly algorithms**: Measurement uses patterns amenable to
//!    GPU parallel reduction (counting, finding first set bit).
//!
//! # Memory Layout
//!
//! For n qubits (and n generators), with w = ceil(n/32) words per column:
//!
//! ```text
//! stab_col_x[q * w + i] = bits for generators [i*32 .. (i+1)*32] on qubit q
//! ```
//!
//! This layout ensures that when processing qubit q:
//! - All data for column q is contiguous in memory
//! - Threads processing adjacent generators access adjacent memory (coalesced)
//!
//! # Example
//!
//! ```rust
//! use pecos_qsim::{GpuStab, CliffordGateable, QuantumSimulator};
//! use pecos_core::qid;
//!
//! let mut sim = GpuStab::new(100);
//!
//! // Queue gates
//! sim.h(&qid(0));
//! for i in 1..100 {
//!     sim.cx(&[pecos_core::QubitId(0), pecos_core::QubitId(i)]);
//! }
//!
//! // Measure - this would trigger GPU execution in a real GPU impl
//! let results = sim.mz(&qid(0));
//! ```

use crate::{CliffordGateable, MeasurementResult, QuantumSimulator, StabilizerTableauSimulator};
use core::fmt::Debug;
use pecos_core::{QubitId, RngManageable};
use pecos_rng::PecosRng;
use pecos_rng::rng_ext::RngProbabilityExt;

/// GPU-optimized stabilizer simulator using column-only representation.
///
/// See the [module documentation](self) for details on the design and memory layout.
#[derive(Clone)]
pub struct GpuStab {
    num_qubits: usize,
    words_per_col: usize,

    // Column storage using u32 for 32-thread warp alignment
    // Layout: col_x[q * words_per_col + w] contains generators [w*32..(w+1)*32] for qubit q
    stab_col_x: Vec<u32>,
    stab_col_z: Vec<u32>,
    destab_col_x: Vec<u32>,
    destab_col_z: Vec<u32>,

    // Signs per generator (packed into u32 words)
    stab_signs_minus: Vec<u32>,
    stab_signs_i: Vec<u32>,
    destab_signs_minus: Vec<u32>,
    destab_signs_i: Vec<u32>,

    // RNG for non-deterministic measurements
    rng: PecosRng,
}

impl Debug for GpuStab {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuStab")
            .field("num_qubits", &self.num_qubits)
            .finish_non_exhaustive()
    }
}

impl GpuStab {
    /// Creates a new GPU-optimized stabilizer simulator.
    ///
    /// All qubits are initialized in the |0⟩ state.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self::with_rng(num_qubits, rand::make_rng())
    }

    /// Creates a new simulator with a specific RNG seed for reproducibility.
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_rng(num_qubits, PecosRng::seed_from_u64(seed))
    }

    /// Creates a new simulator with a provided RNG.
    #[must_use]
    pub fn with_rng(num_qubits: usize, rng: PecosRng) -> Self {
        let words_per_col = num_qubits.div_ceil(32);
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
    /// Stabilizers: `Z_i` for each qubit i
    /// Destabilizers: `X_i` for each qubit i
    fn init_tableau(&mut self) {
        // Clear everything
        self.stab_col_x.fill(0);
        self.stab_col_z.fill(0);
        self.destab_col_x.fill(0);
        self.destab_col_z.fill(0);
        self.stab_signs_minus.fill(0);
        self.stab_signs_i.fill(0);
        self.destab_signs_minus.fill(0);
        self.destab_signs_i.fill(0);

        // Set stabilizer i to have Z on qubit i
        // Set destabilizer i to have X on qubit i
        let words_per_col = self.words_per_col;
        for i in 0..self.num_qubits {
            let idx = i * words_per_col + i / 32;
            let bit = 1u32 << (i % 32);
            self.stab_col_z[idx] |= bit;
            self.destab_col_x[idx] |= bit;
        }
    }

    // ========== Gate implementations ==========
    // These are designed to be easily translatable to GPU kernels.
    // Each processes all generators in parallel (one thread per generator on GPU).

    /// Apply H gate to a qubit.
    /// H: X → Z, Z → X, Y → -Y
    ///
    /// GPU kernel: Each thread handles one u32 word (32 generators)
    fn apply_h(&mut self, qubit: usize) {
        let col_base = qubit * self.words_per_col;

        // Process stabilizers
        for w in 0..self.words_per_col {
            let cx = self.stab_col_x[col_base + w];
            let cz = self.stab_col_z[col_base + w];
            // Y = XZ gets sign flip: toggle minus where both X and Z are set
            self.stab_signs_minus[w] ^= cx & cz;
            // Swap X and Z
            self.stab_col_x[col_base + w] = cz;
            self.stab_col_z[col_base + w] = cx;
        }

        // Process destabilizers
        for w in 0..self.words_per_col {
            let cx = self.destab_col_x[col_base + w];
            let cz = self.destab_col_z[col_base + w];
            self.destab_signs_minus[w] ^= cx & cz;
            self.destab_col_x[col_base + w] = cz;
            self.destab_col_z[col_base + w] = cx;
        }
    }

    /// Apply S (SZ) gate to a qubit.
    /// S: X → Y = iXZ, Z → Z
    ///
    /// GPU kernel: Each thread handles one u32 word (32 generators)
    fn apply_s(&mut self, qubit: usize) {
        let col_base = qubit * self.words_per_col;

        // Process stabilizers
        for w in 0..self.words_per_col {
            let x_gens = self.stab_col_x[col_base + w];
            // i * i = -1: toggle minus for generators with both i and X
            self.stab_signs_minus[w] ^= x_gens & self.stab_signs_i[w];
            // Toggle i for all X generators
            self.stab_signs_i[w] ^= x_gens;
            // Toggle Z for all X generators
            self.stab_col_z[col_base + w] ^= x_gens;
        }

        // Process destabilizers
        for w in 0..self.words_per_col {
            let x_gens = self.destab_col_x[col_base + w];
            self.destab_signs_minus[w] ^= x_gens & self.destab_signs_i[w];
            self.destab_signs_i[w] ^= x_gens;
            self.destab_col_z[col_base + w] ^= x_gens;
        }
    }

    /// Apply S† (`SZdg`) gate to a qubit.
    /// S†: X → -iXZ = -Y, Z → Z
    ///
    /// GPU kernel: Each thread handles one u32 word (32 generators)
    fn apply_sdg(&mut self, qubit: usize) {
        let col_base = qubit * self.words_per_col;

        // Process stabilizers
        // -i multiplication: toggle minus for X without existing i, then toggle i
        for w in 0..self.words_per_col {
            let x_gens = self.stab_col_x[col_base + w];
            self.stab_signs_minus[w] ^= x_gens & !self.stab_signs_i[w];
            self.stab_signs_i[w] ^= x_gens;
            self.stab_col_z[col_base + w] ^= x_gens;
        }

        // Process destabilizers
        for w in 0..self.words_per_col {
            let x_gens = self.destab_col_x[col_base + w];
            self.destab_signs_minus[w] ^= x_gens & !self.destab_signs_i[w];
            self.destab_signs_i[w] ^= x_gens;
            self.destab_col_z[col_base + w] ^= x_gens;
        }
    }

    /// Apply CX (CNOT) gate.
    /// CX: `X_c` → `X_c` `X_t`, `Z_t` → `Z_c` `Z_t`
    ///
    /// GPU kernel: Each thread handles one u32 word, XORs two columns
    fn apply_cx(&mut self, control: usize, target: usize) {
        let ctrl_base = control * self.words_per_col;
        let tgt_base = target * self.words_per_col;

        // Process stabilizers
        for w in 0..self.words_per_col {
            // X_c → X_c X_t: XOR control X column into target X column
            self.stab_col_x[tgt_base + w] ^= self.stab_col_x[ctrl_base + w];
            // Z_t → Z_c Z_t: XOR target Z column into control Z column
            self.stab_col_z[ctrl_base + w] ^= self.stab_col_z[tgt_base + w];
        }

        // Process destabilizers
        for w in 0..self.words_per_col {
            self.destab_col_x[tgt_base + w] ^= self.destab_col_x[ctrl_base + w];
            self.destab_col_z[ctrl_base + w] ^= self.destab_col_z[tgt_base + w];
        }
    }

    /// Apply CZ gate.
    /// CZ: `X_a` → `X_a` `Z_b`, `X_b` → `Z_a` `X_b` (symmetric)
    ///
    /// GPU kernel: Each thread handles one u32 word
    fn apply_cz(&mut self, q1: usize, q2: usize) {
        let base1 = q1 * self.words_per_col;
        let base2 = q2 * self.words_per_col;

        // Process stabilizers
        for w in 0..self.words_per_col {
            let x1 = self.stab_col_x[base1 + w];
            let x2 = self.stab_col_x[base2 + w];
            // Sign update: toggle minus for generators with X on both qubits
            self.stab_signs_minus[w] ^= x1 & x2;
            // X on q1 adds Z on q2
            self.stab_col_z[base2 + w] ^= x1;
            // X on q2 adds Z on q1
            self.stab_col_z[base1 + w] ^= x2;
        }

        // Process destabilizers
        for w in 0..self.words_per_col {
            let x1 = self.destab_col_x[base1 + w];
            let x2 = self.destab_col_x[base2 + w];
            self.destab_col_z[base2 + w] ^= x1;
            self.destab_col_z[base1 + w] ^= x2;
        }
    }

    /// Apply X gate.
    /// X: Z → -Z (flips sign for generators with Z on this qubit)
    fn apply_x(&mut self, qubit: usize) {
        let col_base = qubit * self.words_per_col;

        for w in 0..self.words_per_col {
            self.stab_signs_minus[w] ^= self.stab_col_z[col_base + w];
            self.destab_signs_minus[w] ^= self.destab_col_z[col_base + w];
        }
    }

    /// Apply Y gate.
    /// Y: X → -X, Z → -Z
    fn apply_y(&mut self, qubit: usize) {
        let col_base = qubit * self.words_per_col;

        for w in 0..self.words_per_col {
            // Flip sign for generators with X or Z (but not both, since Y commutes with Y)
            let x_stab = self.stab_col_x[col_base + w];
            let z_stab = self.stab_col_z[col_base + w];
            self.stab_signs_minus[w] ^= x_stab ^ z_stab;

            let x_destab = self.destab_col_x[col_base + w];
            let z_destab = self.destab_col_z[col_base + w];
            self.destab_signs_minus[w] ^= x_destab ^ z_destab;
        }
    }

    /// Apply Z gate.
    /// Z: X → -X (flips sign for generators with X on this qubit)
    fn apply_z(&mut self, qubit: usize) {
        let col_base = qubit * self.words_per_col;

        for w in 0..self.words_per_col {
            self.stab_signs_minus[w] ^= self.stab_col_x[col_base + w];
            self.destab_signs_minus[w] ^= self.destab_col_x[col_base + w];
        }
    }

    // ========== Measurement ==========

    /// Check if measurement is deterministic (no stabilizer anticommutes with Z).
    /// Returns true if deterministic.
    ///
    /// GPU: This is a parallel OR reduction over the column.
    fn is_deterministic(&self, qubit: usize) -> bool {
        let col_base = qubit * self.words_per_col;
        for w in 0..self.words_per_col {
            if self.stab_col_x[col_base + w] != 0 {
                return false;
            }
        }
        true
    }

    /// Find the first stabilizer with X on the given qubit.
    /// Returns None if no such stabilizer exists.
    ///
    /// GPU: This is a parallel "find first set" operation.
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

    /// Perform deterministic measurement.
    /// Outcome is determined by the product of stabilizers corresponding to
    /// destabilizers that have X on the measured qubit.
    fn deterministic_meas(&self, qubit: usize) -> MeasurementResult {
        let col_base = qubit * self.words_per_col;

        // Count destabilizers with X on this qubit intersected with
        // the corresponding STABILIZER signs (not destabilizer signs!)
        let mut num_minuses = 0u32;
        let mut num_is = 0u32;

        for w in 0..self.words_per_col {
            let destab_x = self.destab_col_x[col_base + w];
            num_minuses += (destab_x & self.stab_signs_minus[w]).count_ones();
            num_is += (destab_x & self.stab_signs_i[w]).count_ones();
        }

        // Compute cumulative phase from Pauli multiplication
        // For each destabilizer with X on qubit q, we multiply together
        // the corresponding stabilizers. This requires tracking Z*X overlaps.
        let mut cumulative_x = vec![0u32; self.words_per_col];

        for w in 0..self.words_per_col {
            let mut mask = self.destab_col_x[col_base + w];
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                let generator = w * 32 + bit;

                // Count overlap of stab Z-row with cumulative X
                // We need to check each qubit: stab_col_z[q][generator] & cumulative_x[q]
                for q in 0..self.num_qubits {
                    let q_base = q * self.words_per_col;
                    let gen_word = generator / 32;
                    let gen_bit = 1u32 << (generator % 32);

                    let stab_has_z = (self.stab_col_z[q_base + gen_word] & gen_bit) != 0;
                    let cum_has_x = (cumulative_x[q / 32] & (1u32 << (q % 32))) != 0;

                    if stab_has_z && cum_has_x {
                        num_minuses += 1;
                    }

                    // XOR stab X into cumulative
                    let stab_has_x = (self.stab_col_x[q_base + gen_word] & gen_bit) != 0;
                    if stab_has_x {
                        cumulative_x[q / 32] ^= 1u32 << (q % 32);
                    }
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

    /// Perform non-deterministic measurement.
    /// Randomly chooses outcome and updates tableau.
    fn nondeterministic_meas(&mut self, qubit: usize, outcome: bool) -> MeasurementResult {
        let pivot = self.find_anticommuting_stabilizer(qubit).unwrap();
        let col_base = qubit * self.words_per_col;
        let pivot_word = pivot / 32;
        let pivot_shift = pivot % 32;
        let pivot_bit = 1u32 << pivot_shift;

        // Cache pivot signs
        let pivot_minus = (self.stab_signs_minus[pivot_word] >> pivot_shift) & 1 != 0;
        let pivot_i = (self.stab_signs_i[pivot_word] >> pivot_shift) & 1 != 0;

        // Step 1: Handle pivot's i-phase (matches DenseStab algorithm).
        // Multiply each anticommuting stab's sign by the pivot's i factor.
        if pivot_i {
            self.stab_signs_i[pivot_word] &= !pivot_bit;
            for w in 0..self.words_per_col {
                let mut anticom = self.stab_col_x[col_base + w];
                if w == pivot_word {
                    anticom &= !pivot_bit;
                }
                // i * i = -1: toggle minus for stabs that already have i
                self.stab_signs_minus[w] ^= anticom & self.stab_signs_i[w];
                // Toggle i for all anticommuting stabs
                self.stab_signs_i[w] ^= anticom;
            }
        }

        // Step 2: XOR pivot into other anticommuting stabilizers.
        // Phase: count z_pivot & x_other overlaps (simplified formula that works
        // because pivot's i-phase was already handled above).
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

                // Count z_pivot & x_g overlaps for phase
                let mut count = 0u32;
                for q in 0..self.num_qubits {
                    let base = q * self.words_per_col;
                    let pz = (self.stab_col_z[base + pivot_word] >> pivot_shift) & 1;
                    let gx = (self.stab_col_x[base + g_word] >> (g % 32)) & 1;
                    count += pz & gx;
                }
                if count & 1 != 0 {
                    self.stab_signs_minus[g_word] ^= g_bit;
                }
                if pivot_minus {
                    self.stab_signs_minus[g_word] ^= g_bit;
                }

                // XOR Pauli content of pivot into g
                for q in 0..self.num_qubits {
                    let base = q * self.words_per_col;
                    if (self.stab_col_x[base + pivot_word] >> pivot_shift) & 1 == 1 {
                        self.stab_col_x[base + g_word] ^= g_bit;
                    }
                    if (self.stab_col_z[base + pivot_word] >> pivot_shift) & 1 == 1 {
                        self.stab_col_z[base + g_word] ^= g_bit;
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
                let dst_word = dst / 32;
                let dst_bit = 1u32 << (dst % 32);

                for q in 0..self.num_qubits {
                    let base = q * self.words_per_col;
                    if (self.stab_col_x[base + pivot_word] >> pivot_shift) & 1 == 1 {
                        self.destab_col_x[base + dst_word] ^= dst_bit;
                    }
                    if (self.stab_col_z[base + pivot_word] >> pivot_shift) & 1 == 1 {
                        self.destab_col_z[base + dst_word] ^= dst_bit;
                    }
                }

                anticomm &= anticomm - 1;
            }
        }

        // Copy pivot stabilizer to destabilizer
        self.copy_stab_to_destab(pivot);

        // Set stabilizer to ±Z on measured qubit
        self.set_stabilizer_to_z(pivot, qubit, outcome);

        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }

    /// Copy stabilizer to destabilizer slot.
    fn copy_stab_to_destab(&mut self, generator: usize) {
        let word = generator / 32;
        let bit = 1u32 << (generator % 32);

        for q in 0..self.num_qubits {
            let base = q * self.words_per_col;
            let x_bit = (self.stab_col_x[base + word] >> (generator % 32)) & 1;
            let z_bit = (self.stab_col_z[base + word] >> (generator % 32)) & 1;

            if x_bit == 1 {
                self.destab_col_x[base + word] |= bit;
            } else {
                self.destab_col_x[base + word] &= !bit;
            }
            if z_bit == 1 {
                self.destab_col_z[base + word] |= bit;
            } else {
                self.destab_col_z[base + word] &= !bit;
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

    /// Set stabilizer to ±Z on the given qubit.
    fn set_stabilizer_to_z(&mut self, generator: usize, qubit: usize, negative: bool) {
        let word = generator / 32;
        let bit = 1u32 << (generator % 32);

        // Clear all X and Z bits for this generator
        for q in 0..self.num_qubits {
            let base = q * self.words_per_col;
            self.stab_col_x[base + word] &= !bit;
            self.stab_col_z[base + word] &= !bit;
        }

        // Set Z on the measured qubit
        let base = qubit * self.words_per_col;
        self.stab_col_z[base + word] |= bit;

        // Set sign
        self.stab_signs_i[word] &= !bit; // Clear i
        if negative {
            self.stab_signs_minus[word] |= bit;
        } else {
            self.stab_signs_minus[word] &= !bit;
        }
    }
}

// ========== Trait implementations ==========

impl QuantumSimulator for GpuStab {
    fn reset(&mut self) -> &mut Self {
        self.init_tableau();
        self
    }
}

impl CliffordGateable for GpuStab {
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

impl RngManageable for GpuStab {
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

impl StabilizerTableauSimulator for GpuStab {
    fn stab_tableau(&self) -> String {
        Self::col_only_tableau_string_u32(
            self.num_qubits,
            self.words_per_col,
            &self.stab_col_x,
            &self.stab_col_z,
            &self.stab_signs_minus,
            &self.stab_signs_i,
        )
    }

    fn destab_tableau(&self) -> String {
        Self::col_only_tableau_string_u32(
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

impl GpuStab {
    fn col_only_tableau_string_u32(
        num_qubits: usize,
        words_per_col: usize,
        col_x: &[u32],
        col_z: &[u32],
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

            for qubit in 0..num_qubits {
                let word_idx = qubit * words_per_col + g / 32;
                let bit_mask = 1u32 << (g % 32);
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
}

// ========== Test support ==========

use crate::stabilizer_test_utils::{ForcedMeasurement, StabilizerSimulator};

impl ForcedMeasurement for GpuStab {
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        if self.is_deterministic(qubit) {
            self.deterministic_meas(qubit)
        } else {
            self.nondeterministic_meas(qubit, forced_outcome)
        }
    }
}

impl StabilizerSimulator for GpuStab {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stabilizer_test_utils::run_full_stabilizer_test_suite;

    #[test]
    fn test_gpu_stab_basic() {
        let mut sim = GpuStab::new(2);
        sim.h(&[QubitId(0)]);
        sim.cx(&[QubitId(0), QubitId(1)]);
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert_eq!(results[0].outcome, results[1].outcome);
    }

    #[test]
    fn test_gpu_stab_deterministic_z() {
        let mut sim = GpuStab::new(1);
        // |0⟩ state, measure Z should give 0 deterministically
        let results = sim.mz(&[QubitId(0)]);
        assert!(!results[0].outcome);
        assert!(results[0].is_deterministic);
    }

    #[test]
    fn test_gpu_stab_x_gate() {
        let mut sim = GpuStab::new(1);
        sim.x(&[QubitId(0)]);
        let results = sim.mz(&[QubitId(0)]);
        assert!(results[0].outcome); // Should be |1⟩
        assert!(results[0].is_deterministic);
    }

    #[test]
    fn test_gpu_stab_full_suite() {
        let mut sim = GpuStab::with_seed(8, 42);
        run_full_stabilizer_test_suite(&mut sim, 8);
    }
}
