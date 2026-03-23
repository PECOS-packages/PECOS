// Copyright 2026 The PECOS Developers
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

//! Sparse State Vector Simulator (`AoS` Layout)
//!
//! Uses a sorted `Vec<(usize, Complex64)>` for efficient memory usage when
//! the state has few non-zero amplitudes. This `AoS` (Array of Structures) layout
//! is faster than `SoA` for typical sparse states due to better cache locality
//! and simpler code paths.
//!
//! ## When to use
//!
//! - States with limited superposition (e.g., after only X, Z, CX, CZ gates)
//! - Initial exploration before state becomes dense
//! - Very large qubit counts where dense is infeasible
//! - GHZ-like circuits that stay at 2 amplitudes
//! - Clifford-only circuits (X, Z, CX, CZ, SWAP) on computational basis
//!
//! ## Performance characteristics
//!
//! - GHZ-50 circuit: ~220ns
//! - 100 Clifford gates on 30 qubits: ~375ns
//! - Incremental H (8 qubits): ~4.2µs
//!
//! ## Memory comparison
//!
//! For k non-zero amplitudes out of 2^n total:
//! - Dense (f64): 2^n * 16 bytes
//! - Sparse: k * 24 bytes (index + complex)
//!
//! Sparse wins when k < 2^n * 0.67
//!
//! ## Design decisions
//!
//! 1. **Sorted Vec over `HashMap`**: 3x more memory efficient, better cache locality
//! 2. **Scratch buffer reuse**: Gates use a persistent scratch buffer, avoiding
//!    repeated allocation during gate sequences.
//! 3. **Binary search for pairs**: O(log k) pair lookup, acceptable for sparse states
//! 4. **Amplitude truncation**: Optional epsilon threshold to drop small amplitudes,
//!    keeping the state sparse longer (at cost of approximation error).

use crate::clifford_gateable::MeasurementResult;
use crate::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};
use num_complex::Complex64;
use pecos_core::{Angle64, QubitId};
use pecos_random::{PecosRng, Rng, RngExt, RngProbabilityExt};
use std::f64::consts::FRAC_1_SQRT_2;
use std::fmt::Debug;
use wide::f64x4;

/// Sparse state vector simulator using sorted amplitude vector.
///
/// Amplitudes are stored as `(basis_state_index, amplitude)` pairs,
/// sorted by index for efficient paired iteration during gate application.
#[derive(Debug)]
pub struct SparseStateVecAoS<R = PecosRng>
where
    R: Rng,
{
    /// Non-zero amplitudes sorted by basis state index.
    /// Invariant: indices are unique and sorted in ascending order (when `needs_sort` is false).
    amplitudes: Vec<(usize, Complex64)>,

    /// Number of qubits (determines max index: `2^num_qubits` - 1)
    num_qubits: usize,

    /// Random number generator for measurements
    rng: R,

    /// Amplitude truncation threshold. Amplitudes with |a|^2 < epsilon are dropped.
    /// Set to 0.0 to disable truncation (exact simulation).
    epsilon: f64,

    /// Scratch buffer for gate operations (avoids repeated allocation)
    scratch: Vec<(usize, Complex64)>,

    /// Index scratch buffers for H gate (avoids repeated allocation)
    /// Using u32 instead of usize to halve memory footprint (sufficient for up to 4B amplitudes)
    scratch_low: Vec<u32>,
    scratch_high: Vec<u32>,

    /// Deferred sorting flag. When true, amplitudes may not be sorted.
    /// Gates that modify indices set this to true; gates that need sorted
    /// order call `ensure_sorted()` first. This avoids redundant sorts when
    /// multiple index-modifying gates are applied in sequence.
    needs_sort: bool,
}

impl SparseStateVecAoS<PecosRng> {
    /// Create a new sparse state vector initialized to |0...0⟩
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self::with_rng(num_qubits, rand::make_rng())
    }

    /// Create with a specific seed for reproducibility
    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_rng(num_qubits, PecosRng::seed_from_u64(seed))
    }
}

impl<R: Rng> SparseStateVecAoS<R> {
    /// Create with a custom RNG
    #[must_use]
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        // Start in |0...0⟩ state - single amplitude at index 0
        let amplitudes = vec![(0usize, Complex64::new(1.0, 0.0))];

        Self {
            amplitudes,
            num_qubits,
            rng,
            epsilon: 0.0, // Exact by default
            scratch: Vec::new(),
            scratch_low: Vec::new(),
            scratch_high: Vec::new(),
            needs_sort: false, // Starts sorted (single amplitude at index 0)
        }
    }

    /// Set the amplitude truncation threshold.
    ///
    /// Amplitudes with |a|^2 < epsilon are dropped after gate operations.
    /// Set to 0.0 for exact simulation (default).
    #[inline]
    pub fn set_epsilon(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Get the number of qubits
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the number of non-zero amplitudes
    #[inline]
    #[must_use]
    pub fn num_amplitudes(&self) -> usize {
        self.amplitudes.len()
    }

    /// Get the sparsity ratio (`num_amplitudes` / `2^num_qubits`)
    #[inline]
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        self.amplitudes.len() as f64 / (1usize << self.num_qubits) as f64
    }

    /// Check if this state would be more efficient as dense
    #[inline]
    #[must_use]
    pub fn should_convert_to_dense(&self) -> bool {
        // Sparse entry: 24 bytes (usize + Complex64)
        // Dense entry: 16 bytes (Complex64)
        // Sparse is worse when: k * 24 > 2^n * 16
        // i.e., k > 2^n * 0.67
        self.amplitudes.len() > ((1usize << self.num_qubits) * 2 / 3)
    }

    /// Get amplitude at a specific basis state index (binary search)
    #[inline]
    #[must_use]
    pub fn get_amplitude(&mut self, index: usize) -> Complex64 {
        self.ensure_sorted();
        match self.amplitudes.binary_search_by_key(&index, |&(i, _)| i) {
            Ok(pos) => self.amplitudes[pos].1,
            Err(_) => Complex64::new(0.0, 0.0),
        }
    }

    /// Get probability of measuring a specific basis state
    #[inline]
    #[must_use]
    pub fn probability(&mut self, index: usize) -> f64 {
        self.get_amplitude(index).norm_sqr()
    }

    /// Iterate over all non-zero amplitudes
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &(usize, Complex64)> {
        self.amplitudes.iter()
    }

    /// Mutable iterator over amplitudes (for benchmarking/testing)
    #[inline]
    pub fn amplitudes_mut(&mut self) -> impl Iterator<Item = &mut (usize, Complex64)> {
        self.amplitudes.iter_mut()
    }

    /// Memory usage in bytes
    #[inline]
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        // Vec header (24) + entries + scratch
        24 + self.amplitudes.len() * 24 + 24 + self.scratch.capacity() * 24
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /// Normalize the state vector
    fn normalize(&mut self) {
        let norm_sq: f64 = self.amplitudes.iter().map(|&(_, a)| a.norm_sqr()).sum();
        if norm_sq > 0.0 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for (_, amp) in &mut self.amplitudes {
                *amp *= inv_norm;
            }
        }
    }

    /// Ensure amplitudes are sorted by index. Call this before operations that
    /// require sorted order (e.g., binary search, `partition_point`, two-pointer merge).
    #[inline]
    fn ensure_sorted(&mut self) {
        if self.needs_sort {
            self.amplitudes.sort_unstable_by_key(|&(idx, _)| idx);
            self.needs_sort = false;
        }
    }

    // =========================================================================
    // Single-qubit gate application
    // =========================================================================

    /// Apply X gate using O(k) partition-swap instead of O(k log k) sort.
    ///
    /// X gate flips bit q in each index. After XOR:
    /// - Indices with bit q=0 become bit q=1 (move to second half)
    /// - Indices with bit q=1 become bit q=0 (move to first half)
    ///
    /// Within each partition, relative order is preserved, so we can:
    /// 1. Partition into low (bit=0) and high (bit=1) groups
    /// 2. Concatenate: high group first (becomes low), then low group (becomes high)
    /// 3. XOR all indices
    ///
    /// For very small states (<= 8 amplitudes), uses simple XOR + sort which has
    /// lower overhead.
    #[inline]
    fn apply_x_inplace(&mut self, q: usize) {
        let mask = 1usize << q;
        let len = self.amplitudes.len();

        // For very small states, simple XOR + sort has lower overhead
        if len <= 8 {
            for (idx, _) in &mut self.amplitudes {
                *idx ^= mask;
            }
            self.amplitudes.sort_unstable_by_key(|&(idx, _)| idx);
            return;
        }

        // For larger states, use O(k) partition-swap
        // Find partition point: first index with bit q=1
        // Since array is sorted, all bit=0 come before bit=1
        let partition = self.amplitudes.partition_point(|&(idx, _)| idx & mask == 0);

        // Partition: [low_0..low_n | high_0..high_m]
        // After X:  [high_0'..high_m' | low_0'..low_n'] where ' means XOR with mask

        // Use scratch buffer to reorder
        self.scratch.clear();
        self.scratch.reserve(len);

        // High group first (becomes low indices after XOR)
        for i in partition..len {
            let (idx, amp) = self.amplitudes[i];
            self.scratch.push((idx ^ mask, amp));
        }

        // Low group second (becomes high indices after XOR)
        for i in 0..partition {
            let (idx, amp) = self.amplitudes[i];
            self.scratch.push((idx ^ mask, amp));
        }

        std::mem::swap(&mut self.amplitudes, &mut self.scratch);
    }

    /// Apply a single-qubit gate using O(k) two-pointer pair-finding.
    ///
    /// Strategy: Separate into low (bit=0) and high (bit=1) groups, then use
    /// two-pointer merge to find pairs. Output is produced in sorted order,
    /// avoiding the need for a final sort.
    ///
    /// For small states (<= 8 amplitudes), uses binary search which has lower overhead.
    #[inline]
    fn apply_single_qubit_gate(
        &mut self,
        q: usize,
        a: Complex64,
        b: Complex64,
        c: Complex64,
        d: Complex64,
    ) {
        let mask = 1usize << q;
        let len = self.amplitudes.len();

        // For very small states, binary search has lower overhead
        if len <= 8 {
            self.apply_single_qubit_gate_small(q, a, b, c, d);
            return;
        }

        // Separate into low (bit=0) and high (bit=1) groups using indices only
        // Note: We skip reserve() since these buffers are reused and maintain capacity
        self.scratch_low.clear();
        self.scratch_high.clear();

        for i in 0..len {
            if self.amplitudes[i].0 & mask == 0 {
                self.scratch_low.push(i as u32);
            } else {
                self.scratch_high.push(i as u32);
            }
        }

        self.scratch.clear();

        let low_len = self.scratch_low.len();
        let high_len = self.scratch_high.len();
        let mut low_ptr = 0;
        let mut high_ptr = 0;

        // Two-pointer merge with cached lookups
        // SAFETY: All indices are bounds-checked via have_low/have_high, and scratch indices
        // are valid because they were created from 0..len iteration over amplitudes.
        loop {
            let have_low = low_ptr < low_len;
            let have_high = high_ptr < high_len;

            if !have_low && !have_high {
                break;
            }

            // Cache lookups: get (basis_index, amplitude) for each side
            let (low_idx, low_amp) = if have_low {
                // SAFETY: low_ptr < low_len checked above, idx from 0..len
                let arr_idx = unsafe { *self.scratch_low.get_unchecked(low_ptr) } as usize;
                let entry = unsafe { self.amplitudes.get_unchecked(arr_idx) };
                (entry.0, entry.1)
            } else {
                (usize::MAX, Complex64::new(0.0, 0.0))
            };

            let (high_idx, high_amp) = if have_high {
                // SAFETY: high_ptr < high_len checked above, idx from 0..len
                let arr_idx = unsafe { *self.scratch_high.get_unchecked(high_ptr) } as usize;
                let entry = unsafe { self.amplitudes.get_unchecked(arr_idx) };
                (entry.0, entry.1)
            } else {
                (usize::MAX, Complex64::new(0.0, 0.0))
            };

            let high_partner = high_idx & !mask;

            match low_idx.cmp(&high_partner) {
                std::cmp::Ordering::Equal => {
                    // Paired: process both together
                    let new_low = a * low_amp + b * high_amp;
                    let new_high = c * low_amp + d * high_amp;

                    if new_low.norm_sqr() > self.epsilon {
                        self.scratch.push((low_idx, new_low));
                    }
                    if new_high.norm_sqr() > self.epsilon {
                        self.scratch.push((high_idx, new_high));
                    }
                    low_ptr += 1;
                    high_ptr += 1;
                }
                std::cmp::Ordering::Less => {
                    // Unpaired low
                    let new_low = a * low_amp;
                    let new_high = c * low_amp;

                    if new_low.norm_sqr() > self.epsilon {
                        self.scratch.push((low_idx, new_low));
                    }
                    if new_high.norm_sqr() > self.epsilon {
                        self.scratch.push((low_idx | mask, new_high));
                    }
                    low_ptr += 1;
                }
                std::cmp::Ordering::Greater => {
                    // Unpaired high
                    let new_low = b * high_amp;
                    let new_high = d * high_amp;

                    if new_low.norm_sqr() > self.epsilon {
                        self.scratch.push((high_partner, new_low));
                    }
                    if new_high.norm_sqr() > self.epsilon {
                        self.scratch.push((high_idx, new_high));
                    }
                    high_ptr += 1;
                }
            }
        }

        std::mem::swap(&mut self.amplitudes, &mut self.scratch);
    }

    /// Apply single-qubit gate for small states using binary search.
    /// Lower overhead than two-pointer for <= 8 amplitudes.
    #[inline]
    fn apply_single_qubit_gate_small(
        &mut self,
        q: usize,
        a: Complex64,
        b: Complex64,
        c: Complex64,
        d: Complex64,
    ) {
        let mask = 1usize << q;
        let len = self.amplitudes.len();

        self.scratch.clear();
        self.scratch.reserve(len * 2);

        // First pass: process all "low" indices (bit q=0)
        for i in 0..len {
            let (idx, amp) = self.amplitudes[i];
            if idx & mask != 0 {
                continue;
            }

            let paired_idx = idx | mask;
            let paired_amp = self.amplitudes[i + 1..]
                .binary_search_by_key(&paired_idx, |&(j, _)| j)
                .ok()
                .map_or(Complex64::new(0.0, 0.0), |offset| {
                    self.amplitudes[i + 1 + offset].1
                });

            let new_low = a * amp + b * paired_amp;
            let new_high = c * amp + d * paired_amp;

            if new_low.norm_sqr() > self.epsilon {
                self.scratch.push((idx, new_low));
            }
            if new_high.norm_sqr() > self.epsilon {
                self.scratch.push((paired_idx, new_high));
            }
        }

        // Second pass: handle unpaired "high" indices
        for i in 0..len {
            let (idx, amp) = self.amplitudes[i];
            if idx & mask == 0 {
                continue;
            }

            let paired_idx = idx & !mask;
            if self.amplitudes[..i]
                .binary_search_by_key(&paired_idx, |&(j, _)| j)
                .is_ok()
            {
                continue;
            }

            let new_low = b * amp;
            let new_high = d * amp;

            if new_low.norm_sqr() > self.epsilon {
                self.scratch.push((paired_idx, new_low));
            }
            if new_high.norm_sqr() > self.epsilon {
                self.scratch.push((idx, new_high));
            }
        }

        self.scratch.sort_unstable_by_key(|&(idx, _)| idx);
        std::mem::swap(&mut self.amplitudes, &mut self.scratch);
    }

    /// SIMD-optimized H gate implementation.
    ///
    /// Processes pairs in batches of 2 using f64x4 SIMD operations inline.
    /// For the H gate: `new_low` = (low + high) / √2, `new_high` = (low - high) / √2
    #[inline]
    fn apply_h_simd(&mut self, q: usize) {
        let mask = 1usize << q;
        let len = self.amplitudes.len();

        if len <= 8 {
            let h = Complex64::new(FRAC_1_SQRT_2, 0.0);
            let mh = Complex64::new(-FRAC_1_SQRT_2, 0.0);
            self.apply_single_qubit_gate_small(q, h, h, h, mh);
            return;
        }

        // Separate into low (bit=0) and high (bit=1) groups
        self.scratch_low.clear();
        self.scratch_high.clear();

        for i in 0..len {
            if self.amplitudes[i].0 & mask == 0 {
                self.scratch_low.push(i as u32);
            } else {
                self.scratch_high.push(i as u32);
            }
        }

        self.scratch.clear();

        let low_len = self.scratch_low.len();
        let high_len = self.scratch_high.len();
        let mut low_ptr = 0;
        let mut high_ptr = 0;

        // Batch buffer for SIMD processing
        let mut pair_batch: [(usize, usize, Complex64, Complex64); 2] =
            [(0, 0, Complex64::default(), Complex64::default()); 2];
        let mut batch_count = 0;
        let scale = f64x4::splat(FRAC_1_SQRT_2);

        loop {
            let have_low = low_ptr < low_len;
            let have_high = high_ptr < high_len;

            if !have_low && !have_high {
                break;
            }

            let (low_idx, low_amp) = if have_low {
                let arr_idx = unsafe { *self.scratch_low.get_unchecked(low_ptr) } as usize;
                let entry = unsafe { self.amplitudes.get_unchecked(arr_idx) };
                (entry.0, entry.1)
            } else {
                (usize::MAX, Complex64::new(0.0, 0.0))
            };

            let (high_idx, high_amp) = if have_high {
                let arr_idx = unsafe { *self.scratch_high.get_unchecked(high_ptr) } as usize;
                let entry = unsafe { self.amplitudes.get_unchecked(arr_idx) };
                (entry.0, entry.1)
            } else {
                (usize::MAX, Complex64::new(0.0, 0.0))
            };

            let high_partner = high_idx & !mask;

            match low_idx.cmp(&high_partner) {
                std::cmp::Ordering::Equal => {
                    // Paired: add to batch
                    pair_batch[batch_count] = (low_idx, high_idx, low_amp, high_amp);
                    batch_count += 1;

                    if batch_count == 2 {
                        // Process batch of 2 pairs with SIMD
                        let (li0, hi0, la0, ha0) = pair_batch[0];
                        let (li1, hi1, la1, ha1) = pair_batch[1];

                        let low_vec = f64x4::new([la0.re, la0.im, la1.re, la1.im]);
                        let high_vec = f64x4::new([ha0.re, ha0.im, ha1.re, ha1.im]);

                        let sum = (low_vec + high_vec) * scale;
                        let diff = (low_vec - high_vec) * scale;

                        let sum_arr = sum.to_array();
                        let diff_arr = diff.to_array();

                        let new_low0 = Complex64::new(sum_arr[0], sum_arr[1]);
                        let new_high0 = Complex64::new(diff_arr[0], diff_arr[1]);
                        let new_low1 = Complex64::new(sum_arr[2], sum_arr[3]);
                        let new_high1 = Complex64::new(diff_arr[2], diff_arr[3]);

                        if new_low0.norm_sqr() > self.epsilon {
                            self.scratch.push((li0, new_low0));
                        }
                        if new_high0.norm_sqr() > self.epsilon {
                            self.scratch.push((hi0, new_high0));
                        }
                        if new_low1.norm_sqr() > self.epsilon {
                            self.scratch.push((li1, new_low1));
                        }
                        if new_high1.norm_sqr() > self.epsilon {
                            self.scratch.push((hi1, new_high1));
                        }

                        batch_count = 0;
                    }

                    low_ptr += 1;
                    high_ptr += 1;
                }
                std::cmp::Ordering::Less => {
                    // Unpaired low
                    let scaled =
                        Complex64::new(low_amp.re * FRAC_1_SQRT_2, low_amp.im * FRAC_1_SQRT_2);

                    if scaled.norm_sqr() > self.epsilon {
                        self.scratch.push((low_idx, scaled));
                        self.scratch.push((low_idx | mask, scaled));
                    }
                    low_ptr += 1;
                }
                std::cmp::Ordering::Greater => {
                    // Unpaired high
                    let new_low =
                        Complex64::new(high_amp.re * FRAC_1_SQRT_2, high_amp.im * FRAC_1_SQRT_2);
                    let new_high =
                        Complex64::new(-high_amp.re * FRAC_1_SQRT_2, -high_amp.im * FRAC_1_SQRT_2);

                    if new_low.norm_sqr() > self.epsilon {
                        self.scratch.push((high_partner, new_low));
                    }
                    if new_high.norm_sqr() > self.epsilon {
                        self.scratch.push((high_idx, new_high));
                    }
                    high_ptr += 1;
                }
            }
        }

        // Process remaining pair in batch
        if batch_count == 1 {
            let (li, hi, la, ha) = pair_batch[0];
            let new_low = Complex64::new(
                (la.re + ha.re) * FRAC_1_SQRT_2,
                (la.im + ha.im) * FRAC_1_SQRT_2,
            );
            let new_high = Complex64::new(
                (la.re - ha.re) * FRAC_1_SQRT_2,
                (la.im - ha.im) * FRAC_1_SQRT_2,
            );

            if new_low.norm_sqr() > self.epsilon {
                self.scratch.push((li, new_low));
            }
            if new_high.norm_sqr() > self.epsilon {
                self.scratch.push((hi, new_high));
            }
        }

        std::mem::swap(&mut self.amplitudes, &mut self.scratch);
        // The two-pointer walk interleaves low and high results, producing
        // unsorted output. Mark for deferred sorting.
        self.needs_sort = true;
    }

    // =========================================================================
    // Two-qubit gate application
    // =========================================================================

    /// Apply a controlled gate (control must be |1⟩ for gate to apply)
    ///
    /// Uses binary search for pair lookup, no auxiliary allocation.
    /// Note: Most controlled gates now have direct optimized implementations,
    /// but this is kept for potential future use with non-standard controlled gates.
    #[inline]
    #[allow(dead_code)]
    fn apply_controlled_gate(
        &mut self,
        control: usize,
        target: usize,
        a: Complex64,
        b: Complex64,
        c: Complex64,
        d: Complex64,
    ) {
        let control_mask = 1usize << control;
        let target_mask = 1usize << target;
        let len = self.amplitudes.len();

        self.scratch.clear();
        self.scratch.reserve(len * 2);

        // Process in two passes to avoid needing a "processed" array:
        // 1. Pass through control=0 indices unchanged
        // 2. Process control=1, target=0 indices (the "low" side of pairs)
        // 3. Handle control=1, target=1 indices whose pair doesn't exist

        // First: pass through all control=0 indices
        for i in 0..len {
            let (idx, amp) = self.amplitudes[i];
            if idx & control_mask == 0 {
                self.scratch.push((idx, amp));
            }
        }

        // Second: process control=1, target=0 indices (low side of each pair)
        for i in 0..len {
            let (idx, amp) = self.amplitudes[i];
            if idx & control_mask == 0 || idx & target_mask != 0 {
                continue; // Skip control=0 and target=1 in this pass
            }

            // This is control=1, target=0 - find its pair (control=1, target=1)
            let paired_idx = idx | target_mask;
            let paired_amp = self.amplitudes[i + 1..]
                .binary_search_by_key(&paired_idx, |&(j, _)| j)
                .ok()
                .map_or(Complex64::new(0.0, 0.0), |offset| {
                    self.amplitudes[i + 1 + offset].1
                });

            // Apply gate
            let new_0 = a * amp + b * paired_amp;
            let new_1 = c * amp + d * paired_amp;

            if new_0.norm_sqr() > self.epsilon {
                self.scratch.push((idx, new_0));
            }
            if new_1.norm_sqr() > self.epsilon {
                self.scratch.push((paired_idx, new_1));
            }
        }

        // Third: handle control=1, target=1 indices whose low pair doesn't exist
        for i in 0..len {
            let (idx, amp) = self.amplitudes[i];
            if idx & control_mask == 0 || idx & target_mask == 0 {
                continue; // Skip control=0 and target=0
            }

            // This is control=1, target=1 - check if low pair exists
            let paired_idx = idx & !target_mask;
            if self.amplitudes[..i]
                .binary_search_by_key(&paired_idx, |&(j, _)| j)
                .is_ok()
            {
                continue; // Already processed
            }

            // Unpaired high: low amplitude is 0
            let new_0 = b * amp;
            let new_1 = d * amp;

            if new_0.norm_sqr() > self.epsilon {
                self.scratch.push((paired_idx, new_0));
            }
            if new_1.norm_sqr() > self.epsilon {
                self.scratch.push((idx, new_1));
            }
        }

        self.scratch.sort_unstable_by_key(|&(idx, _)| idx);
        std::mem::swap(&mut self.amplitudes, &mut self.scratch);
    }

    /// Apply CX gate using O(k) partition-swap instead of O(k log k) sort.
    ///
    /// CX flips the target bit only when control bit is 1.
    /// For control=0 indices: unchanged
    /// For control=1 indices: apply X-gate-like partition-swap on target bit
    ///
    /// For very small states (<= 8 amplitudes), uses simple XOR + sort which has
    /// lower overhead.
    #[inline]
    fn apply_cx_inplace(&mut self, control: usize, target: usize) {
        let control_mask = 1usize << control;
        let target_mask = 1usize << target;
        let len = self.amplitudes.len();

        // For very small states, simple XOR + sort has lower overhead
        if len <= 8 {
            for (idx, _) in &mut self.amplitudes {
                if *idx & control_mask != 0 {
                    *idx ^= target_mask;
                }
            }
            self.amplitudes.sort_unstable_by_key(|&(idx, _)| idx);
            return;
        }

        // For larger states, use O(k) partition-swap
        // Find the boundary between control=0 and control=1 indices
        // Since array is sorted, we can use partition_point
        let control_boundary = self
            .amplitudes
            .partition_point(|&(idx, _)| idx & control_mask == 0);

        // Control=0 indices stay unchanged
        // Control=1 indices need X applied on target bit

        // For control=1 indices, partition by target bit
        let control1_slice = &self.amplitudes[control_boundary..];
        let target_boundary_in_control1 =
            control1_slice.partition_point(|&(idx, _)| idx & target_mask == 0);
        let target_boundary = control_boundary + target_boundary_in_control1;

        // After CX on control=1:
        // - Indices with (c=1, t=0) become (c=1, t=1)
        // - Indices with (c=1, t=1) become (c=1, t=0)
        //
        // Original order in control=1 region: [c1_t0...] [c1_t1...]
        // After CX:                          [c1_t1'...] [c1_t0'...]  (where ' means XOR with target_mask)

        // Build result: control=0 indices unchanged, then merge transformed control=1 indices
        self.scratch.clear();
        self.scratch.reserve(len);

        // Control=0 indices (unchanged)
        for i in 0..control_boundary {
            self.scratch.push(self.amplitudes[i]);
        }

        // Control=1, originally t=1, now becomes t=0 (goes first in control=1 region)
        for i in target_boundary..len {
            let (idx, amp) = self.amplitudes[i];
            self.scratch.push((idx ^ target_mask, amp));
        }

        // Control=1, originally t=0, now becomes t=1 (goes second in control=1 region)
        for i in control_boundary..target_boundary {
            let (idx, amp) = self.amplitudes[i];
            self.scratch.push((idx ^ target_mask, amp));
        }

        std::mem::swap(&mut self.amplitudes, &mut self.scratch);
    }
}

// =============================================================================
// QuantumSimulator trait implementation
// =============================================================================

impl<R: Rng + Debug> QuantumSimulator for SparseStateVecAoS<R> {
    fn reset(&mut self) -> &mut Self {
        self.amplitudes.clear();
        self.amplitudes.push((0, Complex64::new(1.0, 0.0)));
        self.needs_sort = false; // Single element is trivially sorted
        self
    }
}

// =============================================================================
// CliffordGateable trait implementation
// =============================================================================

impl<R: Rng + Debug> CliffordGateable for SparseStateVecAoS<R> {
    // -------------------------------------------------------------------------
    // Single-qubit Clifford gates
    // -------------------------------------------------------------------------

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.ensure_sorted();
            self.apply_h_simd(q.0);
        }
        self
    }

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        if qubits.is_empty() {
            return self;
        }

        if qubits.len() == 1 {
            // Single qubit: use O(k) partition-swap (needs sorted input)
            self.ensure_sorted();
            self.apply_x_inplace(qubits[0].0);
        } else {
            // Multiple qubits: XOR with combined mask + sort
            // This avoids multiple buffer swaps
            let combined_mask: usize = qubits.iter().map(|q| 1usize << q.0).fold(0, |a, b| a | b);

            for (idx, _) in &mut self.amplitudes {
                *idx ^= combined_mask;
            }
            self.needs_sort = true;
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
        // Batched: flip bits (like X) + phase = i^k * (-1)^n_ones
        // where k = number of qubits, n_ones = count of bits that were 1
        if qubits.is_empty() {
            return self;
        }

        if qubits.len() == 1 {
            let bit_mask = 1usize << qubits[0].0;
            let i = Complex64::new(0.0, 1.0);
            let mi = Complex64::new(0.0, -1.0);
            for (idx, amp) in &mut self.amplitudes {
                let was_one = *idx & bit_mask != 0;
                *idx ^= bit_mask;
                *amp *= if was_one { mi } else { i };
            }
        } else {
            let combined_mask: usize = qubits.iter().map(|q| 1usize << q.0).fold(0, |a, b| a | b);
            let k = qubits.len();
            // Precompute i^k
            let i_to_k = [
                Complex64::new(1.0, 0.0),  // i^0
                Complex64::new(0.0, 1.0),  // i^1
                Complex64::new(-1.0, 0.0), // i^2
                Complex64::new(0.0, -1.0), // i^3
            ][k % 4];

            for (idx, amp) in &mut self.amplitudes {
                let n_ones = (*idx & combined_mask).count_ones();
                *idx ^= combined_mask;
                // Phase = i^k * (-1)^n_ones
                let phase = if n_ones.is_multiple_of(2) {
                    i_to_k
                } else {
                    -i_to_k
                };
                *amp *= phase;
            }
        }
        self.needs_sort = true;
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        // Z only affects phase of |1⟩ states - can do in-place
        // Optimization: single pass with combined mask instead of one pass per qubit
        // Z(q0) * Z(q1) * ... negates if odd number of specified qubits are |1⟩
        if qubits.is_empty() {
            return self;
        }

        if qubits.len() == 1 {
            // Single qubit: simple path
            let bit_mask = 1usize << qubits[0].0;
            for (idx, amp) in &mut self.amplitudes {
                if *idx & bit_mask != 0 {
                    *amp = -*amp;
                }
            }
        } else {
            // Multiple qubits: combine masks and count bits
            let combined_mask: usize = qubits.iter().map(|q| 1usize << q.0).fold(0, |a, b| a | b);
            for (idx, amp) in &mut self.amplitudes {
                // Negate if odd number of specified qubits have bit set
                if (*idx & combined_mask).count_ones() % 2 == 1 {
                    *amp = -*amp;
                }
            }
        }
        self
    }

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        // S gate: |1⟩ -> i|1⟩
        // Batched: i^(count_ones) where count is over combined mask
        // i^0=1, i^1=i, i^2=-1, i^3=-i (cycle of 4)
        if qubits.is_empty() {
            return self;
        }

        if qubits.len() == 1 {
            let bit_mask = 1usize << qubits[0].0;
            let i = Complex64::new(0.0, 1.0);
            for (idx, amp) in &mut self.amplitudes {
                if *idx & bit_mask != 0 {
                    *amp *= i;
                }
            }
        } else {
            let combined_mask: usize = qubits.iter().map(|q| 1usize << q.0).fold(0, |a, b| a | b);
            // Precompute i^n for n in 0..4
            let phases = [
                Complex64::new(1.0, 0.0),  // i^0
                Complex64::new(0.0, 1.0),  // i^1
                Complex64::new(-1.0, 0.0), // i^2
                Complex64::new(0.0, -1.0), // i^3
            ];
            for (idx, amp) in &mut self.amplitudes {
                let count = (*idx & combined_mask).count_ones() as usize;
                *amp *= phases[count % 4];
            }
        }
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        // S-dagger: |1⟩ -> -i|1⟩
        // Batched: (-i)^(count_ones)
        // (-i)^0=1, (-i)^1=-i, (-i)^2=-1, (-i)^3=i (cycle of 4)
        if qubits.is_empty() {
            return self;
        }

        if qubits.len() == 1 {
            let bit_mask = 1usize << qubits[0].0;
            let mi = Complex64::new(0.0, -1.0);
            for (idx, amp) in &mut self.amplitudes {
                if *idx & bit_mask != 0 {
                    *amp *= mi;
                }
            }
        } else {
            let combined_mask: usize = qubits.iter().map(|q| 1usize << q.0).fold(0, |a, b| a | b);
            let phases = [
                Complex64::new(1.0, 0.0),  // (-i)^0
                Complex64::new(0.0, -1.0), // (-i)^1
                Complex64::new(-1.0, 0.0), // (-i)^2
                Complex64::new(0.0, 1.0),  // (-i)^3
            ];
            for (idx, amp) in &mut self.amplitudes {
                let count = (*idx & combined_mask).count_ones() as usize;
                *amp *= phases[count % 4];
            }
        }
        self
    }

    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        let a = Complex64::new(0.5, 0.5);
        let b = Complex64::new(0.5, -0.5);

        for &q in qubits {
            self.ensure_sorted();
            self.apply_single_qubit_gate(q.0, a, b, b, a);
        }
        self
    }

    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        let a = Complex64::new(0.5, -0.5);
        let b = Complex64::new(0.5, 0.5);

        for &q in qubits {
            self.ensure_sorted();
            self.apply_single_qubit_gate(q.0, a, b, b, a);
        }
        self
    }

    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        let a = Complex64::new(0.5, 0.5);
        let b = Complex64::new(-0.5, -0.5);
        let c = Complex64::new(0.5, 0.5);

        for &q in qubits {
            self.ensure_sorted();
            self.apply_single_qubit_gate(q.0, a, b, c, a);
        }
        self
    }

    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        let a = Complex64::new(0.5, -0.5);
        let b = Complex64::new(0.5, -0.5);
        let c = Complex64::new(-0.5, 0.5);

        for &q in qubits {
            self.ensure_sorted();
            self.apply_single_qubit_gate(q.0, a, b, c, a);
        }
        self
    }

    // -------------------------------------------------------------------------
    // Two-qubit gates
    // -------------------------------------------------------------------------

    fn cx(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CX requires pairs of qubits"
        );

        if qubits.len() == 2 {
            // Single pair: use O(k) partition-swap (needs sorted input)
            self.ensure_sorted();
            self.apply_cx_inplace(qubits[0].0, qubits[1].0);
        } else {
            // Multiple pairs: compute combined XOR mask for each amplitude
            // This avoids multiple buffer swaps
            let pairs: Vec<(usize, usize)> = qubits
                .chunks_exact(2)
                .map(|pair| (1usize << pair[0].0, 1usize << pair[1].0))
                .collect();

            for (idx, _) in &mut self.amplitudes {
                let mut xor_mask = 0usize;
                for &(control_mask, target_mask) in &pairs {
                    if *idx & control_mask != 0 {
                        xor_mask ^= target_mask;
                    }
                }
                *idx ^= xor_mask;
            }
            self.needs_sort = true;
        }
        self
    }

    fn cz(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CZ requires pairs of qubits"
        );

        if qubits.len() == 2 {
            // Single pair: simple path
            let q1_mask = 1usize << qubits[0].0;
            let q2_mask = 1usize << qubits[1].0;
            for (idx, amp) in &mut self.amplitudes {
                if (*idx & q1_mask != 0) && (*idx & q2_mask != 0) {
                    *amp = -*amp;
                }
            }
        } else {
            // Multiple pairs: single pass checking all pairs
            // Precompute masks for efficiency
            let masks: Vec<(usize, usize)> = qubits
                .chunks_exact(2)
                .map(|pair| (1usize << pair[0].0, 1usize << pair[1].0))
                .collect();

            for (idx, amp) in &mut self.amplitudes {
                let mut flip = false;
                for &(q1_mask, q2_mask) in &masks {
                    if (*idx & q1_mask != 0) && (*idx & q2_mask != 0) {
                        flip = !flip;
                    }
                }
                if flip {
                    *amp = -*amp;
                }
            }
        }
        self
    }

    fn cy(&mut self, qubits: &[QubitId]) -> &mut Self {
        // CY: if control=1, apply Y to target
        // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CY requires pairs of qubits"
        );

        if qubits.len() == 2 {
            let control_mask = 1usize << qubits[0].0;
            let target_mask = 1usize << qubits[1].0;
            let i = Complex64::new(0.0, 1.0);
            let mi = Complex64::new(0.0, -1.0);

            for (idx, amp) in &mut self.amplitudes {
                if *idx & control_mask != 0 {
                    let target_was_one = *idx & target_mask != 0;
                    *idx ^= target_mask;
                    *amp *= if target_was_one { mi } else { i };
                }
            }
        } else {
            // Batched CY: for each pair, conditionally flip target and apply phase
            let pairs: Vec<(usize, usize)> = qubits
                .chunks_exact(2)
                .map(|pair| (1usize << pair[0].0, 1usize << pair[1].0))
                .collect();

            let i = Complex64::new(0.0, 1.0);
            let mi = Complex64::new(0.0, -1.0);

            for (idx, amp) in &mut self.amplitudes {
                let mut phase = Complex64::new(1.0, 0.0);
                let mut xor_mask = 0usize;

                for &(control_mask, target_mask) in &pairs {
                    if *idx & control_mask != 0 {
                        let target_was_one = *idx & target_mask != 0;
                        xor_mask ^= target_mask;
                        phase *= if target_was_one { mi } else { i };
                    }
                }

                *idx ^= xor_mask;
                *amp *= phase;
            }
        }
        self.needs_sort = true;
        self
    }

    fn swap(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "SWAP requires pairs of qubits"
        );

        if qubits.len() == 2 {
            // Single SWAP: simple path
            let mask1 = 1usize << qubits[0].0;
            let mask2 = 1usize << qubits[1].0;
            let combined = mask1 | mask2;

            for (idx, _) in &mut self.amplitudes {
                let bit1 = (*idx & mask1) != 0;
                let bit2 = (*idx & mask2) != 0;
                if bit1 != bit2 {
                    *idx ^= combined;
                }
            }
        } else {
            // Batched SWAP: compute all swaps in single pass
            let pairs: Vec<(usize, usize, usize)> = qubits
                .chunks_exact(2)
                .map(|pair| {
                    let m1 = 1usize << pair[0].0;
                    let m2 = 1usize << pair[1].0;
                    (m1, m2, m1 | m2)
                })
                .collect();

            for (idx, _) in &mut self.amplitudes {
                for &(mask1, mask2, combined) in &pairs {
                    let bit1 = (*idx & mask1) != 0;
                    let bit2 = (*idx & mask2) != 0;
                    if bit1 != bit2 {
                        *idx ^= combined;
                    }
                }
            }
        }

        self.needs_sort = true;
        self
    }

    fn iswap(&mut self, qubits: &[QubitId]) -> &mut Self {
        // iSWAP: |00⟩→|00⟩, |01⟩→i|10⟩, |10⟩→i|01⟩, |11⟩→|11⟩
        // Swaps bits when they differ, multiplies by i for each swap
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "iSWAP requires pairs of qubits"
        );

        if qubits.len() == 2 {
            let mask1 = 1usize << qubits[0].0;
            let mask2 = 1usize << qubits[1].0;
            let combined = mask1 | mask2;
            let i = Complex64::new(0.0, 1.0);

            for (idx, amp) in &mut self.amplitudes {
                let bit1 = (*idx & mask1) != 0;
                let bit2 = (*idx & mask2) != 0;
                if bit1 != bit2 {
                    *idx ^= combined;
                    *amp *= i;
                }
            }
        } else {
            // Batched iSWAP: swap bits and count swaps for phase
            let pairs: Vec<(usize, usize, usize)> = qubits
                .chunks_exact(2)
                .map(|pair| {
                    let m1 = 1usize << pair[0].0;
                    let m2 = 1usize << pair[1].0;
                    (m1, m2, m1 | m2)
                })
                .collect();

            // Precompute i^n for phase calculation
            let phases = [
                Complex64::new(1.0, 0.0),  // i^0
                Complex64::new(0.0, 1.0),  // i^1
                Complex64::new(-1.0, 0.0), // i^2
                Complex64::new(0.0, -1.0), // i^3
            ];

            for (idx, amp) in &mut self.amplitudes {
                let mut swap_count = 0usize;
                for &(mask1, mask2, combined) in &pairs {
                    let bit1 = (*idx & mask1) != 0;
                    let bit2 = (*idx & mask2) != 0;
                    if bit1 != bit2 {
                        *idx ^= combined;
                        swap_count += 1;
                    }
                }
                *amp *= phases[swap_count % 4];
            }
        }

        self.needs_sort = true;
        self
    }

    // -------------------------------------------------------------------------
    // Measurement
    // -------------------------------------------------------------------------

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        if qubits.is_empty() {
            return Vec::new();
        }

        if qubits.len() == 1 {
            // Single qubit: simple path
            let bit_mask = 1usize << qubits[0].0;
            let prob_one: f64 = self
                .amplitudes
                .iter()
                .filter(|&&(idx, _)| idx & bit_mask != 0)
                .map(|&(_, amp)| amp.norm_sqr())
                .sum();

            let is_deterministic = !(1e-10..=1.0 - 1e-10).contains(&prob_one);
            let outcome = self.rng.bernoulli(prob_one);

            let keep_mask_value = if outcome { bit_mask } else { 0 };
            self.amplitudes
                .retain(|&(idx, _)| (idx & bit_mask) == keep_mask_value);
            self.normalize();

            return vec![MeasurementResult {
                outcome,
                is_deterministic,
            }];
        }

        // Batched measurement: compute all outcome probabilities in single pass
        let n = qubits.len();
        if n > 16 {
            // Process in chunks of 16 to avoid 2^n bucket explosion
            let mut results = Vec::with_capacity(n);
            for chunk in qubits.chunks(16) {
                results.extend(self.mz(chunk));
            }
            return results;
        }

        // Build combined mask and per-qubit masks
        let qubit_masks: Vec<usize> = qubits.iter().map(|q| 1usize << q.0).collect();
        let combined_mask: usize = qubit_masks.iter().fold(0, |a, b| a | b);

        // Compute probability for each of 2^n outcomes and marginals in single pass
        // This is O(k * n) instead of O(2^n * n) for computing marginals afterwards
        let num_outcomes = 1usize << n;
        let mut probs = vec![0.0f64; num_outcomes];
        let mut marginals = vec![0.0f64; n];

        for &(idx, amp) in &self.amplitudes {
            let prob = amp.norm_sqr();
            // Extract the bits corresponding to measured qubits
            let mut outcome_idx = 0usize;
            for (i, &mask) in qubit_masks.iter().enumerate() {
                if idx & mask != 0 {
                    outcome_idx |= 1 << i;
                    marginals[i] += prob;
                }
            }
            probs[outcome_idx] += prob;
        }

        // Sample from the distribution
        // Use binary search for 6+ qubits (64+ outcomes) where it's faster
        let sampled_outcome = if n >= 6 {
            let cdf = self.rng.compute_cdf(&probs);
            self.rng.sample_discrete_cdf(&cdf)
        } else {
            // Linear scan for small distributions
            let rand_val = self.rng.random::<f64>();
            let mut cumulative = 0.0;
            let mut outcome = 0usize;
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if rand_val < cumulative {
                    outcome = i;
                    break;
                }
            }
            outcome
        };

        // Build results using pre-computed marginals
        let mut results = Vec::with_capacity(n);
        for (i, &prob_one) in marginals.iter().enumerate() {
            let outcome = (sampled_outcome >> i) & 1 == 1;
            // Use pre-computed marginal probability for determinism check
            let is_deterministic = !(1e-10..=1.0 - 1e-10).contains(&prob_one);
            results.push(MeasurementResult {
                outcome,
                is_deterministic,
            });
        }

        // Collapse: keep only amplitudes matching sampled outcome
        // Build the expected bit pattern for measured qubits
        let mut expected_pattern = 0usize;
        for (i, &mask) in qubit_masks.iter().enumerate() {
            if (sampled_outcome >> i) & 1 == 1 {
                expected_pattern |= mask;
            }
        }

        self.amplitudes
            .retain(|&(idx, _)| (idx & combined_mask) == expected_pattern);
        self.normalize();

        results
    }

    // -------------------------------------------------------------------------
    // Prep operations
    // -------------------------------------------------------------------------
    // pz/pnz use mz for measurement then apply corrections.
    // After mz collapses the state, corrections operate on few amplitudes.
    // XOR preserves sort order since all surviving amplitudes have identical
    // bits in the measured positions.

    fn pz(&mut self, qubits: &[QubitId]) -> &mut Self {
        if qubits.is_empty() {
            return self;
        }

        let results = self.mz(qubits);

        // Correction: flip bits that measured |1⟩ to get |0⟩
        let correction_mask: usize = qubits
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| r.outcome)
            .map(|(q, _)| 1usize << q.0)
            .fold(0, |a, b| a | b);

        if correction_mask != 0 {
            for (idx, _) in &mut self.amplitudes {
                *idx ^= correction_mask;
            }
        }

        self
    }

    fn pnz(&mut self, qubits: &[QubitId]) -> &mut Self {
        if qubits.is_empty() {
            return self;
        }

        let results = self.mz(qubits);

        // Correction: flip bits that measured |0⟩ to get |1⟩
        let correction_mask: usize = qubits
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| !r.outcome)
            .map(|(q, _)| 1usize << q.0)
            .fold(0, |a, b| a | b);

        if correction_mask != 0 {
            for (idx, _) in &mut self.amplitudes {
                *idx ^= correction_mask;
            }
        }

        self
    }

    // -------------------------------------------------------------------------
    // Measure-and-prep operations (mpz/mpnz - returns results)
    // -------------------------------------------------------------------------

    fn mpz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        if qubits.is_empty() {
            return Vec::new();
        }

        // Measure first - this collapses the state significantly
        let results = self.mz(qubits);

        // Build correction mask: qubits that measured |1⟩ need to be flipped to |0⟩
        let correction_mask: usize = qubits
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| r.outcome)
            .map(|(q, _)| 1usize << q.0)
            .fold(0, |a, b| a | b);

        if correction_mask != 0 {
            for (idx, _) in &mut self.amplitudes {
                *idx ^= correction_mask;
            }
        }

        results
    }

    fn mpnz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        if qubits.is_empty() {
            return Vec::new();
        }

        let results = self.mz(qubits);

        let correction_mask: usize = qubits
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| !r.outcome)
            .map(|(q, _)| 1usize << q.0)
            .fold(0, |a, b| a | b);

        if correction_mask != 0 {
            for (idx, _) in &mut self.amplitudes {
                *idx ^= correction_mask;
            }
        }

        results
    }
}

// =============================================================================
// ArbitraryRotationGateable trait implementation
// =============================================================================

impl<R: Rng + Debug> ArbitraryRotationGateable for SparseStateVecAoS<R> {
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        // RX(theta) = [[cos, -i*sin], [-i*sin, cos]]
        let a = Complex64::new(cos, 0.0);
        let b = Complex64::new(0.0, -sin);
        let c = Complex64::new(0.0, -sin);
        let d = Complex64::new(cos, 0.0);
        for &q in qubits {
            self.apply_single_qubit_gate(q.0, a, b, c, d);
        }
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let half = theta / 2.0;
        let cos = half.cos();
        let sin = half.sin();
        // RZ(theta) = [[e^{-i*theta/2}, 0], [0, e^{i*theta/2}]]
        let phase_low = Complex64::new(cos, -sin);
        let phase_high = Complex64::new(cos, sin);
        for &q in qubits {
            let mask = 1usize << q.0;
            for (idx, amp) in &mut self.amplitudes {
                if *idx & mask == 0 {
                    *amp *= phase_low;
                } else {
                    *amp *= phase_high;
                }
            }
        }
        self
    }

    fn rzz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "RZZ requires pairs of qubits"
        );
        let half = theta / 2.0;
        // Same parity: e^{-i*theta/2}, different parity: e^{i*theta/2}
        let phase_same = Complex64::new((-half).cos(), (-half).sin());
        let phase_diff = Complex64::new(half.cos(), half.sin());

        for pair in qubits.chunks_exact(2) {
            let mask1 = 1usize << pair[0].0;
            let mask2 = 1usize << pair[1].0;
            for (idx, amp) in &mut self.amplitudes {
                let bit1 = (*idx & mask1) != 0;
                let bit2 = (*idx & mask2) != 0;
                if bit1 == bit2 {
                    *amp *= phase_same;
                } else {
                    *amp *= phase_diff;
                }
            }
        }
        self
    }

    fn u(
        &mut self,
        theta: Angle64,
        phi: Angle64,
        lambda: Angle64,
        qubits: &[QubitId],
    ) -> &mut Self {
        let theta = theta.to_radians_signed();
        let phi = phi.to_radians_signed();
        let lambda = lambda.to_radians_signed();
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        // U gate matrix (matching StateVecSoA's direct implementation)
        let a = Complex64::new(cos, 0.0);
        let b = Complex64::new(-sin * lambda.cos(), -sin * lambda.sin());
        let c = Complex64::new(sin * phi.cos(), sin * phi.sin());
        let d = Complex64::new(cos * (phi + lambda).cos(), cos * (phi + lambda).sin());

        for &q in qubits {
            self.apply_single_qubit_gate(q.0, a, b, c, d);
        }
        self
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let mut sim = SparseStateVecAoS::new(4);
        assert_eq!(sim.num_qubits(), 4);
        assert_eq!(sim.num_amplitudes(), 1);
        assert_eq!(sim.get_amplitude(0), Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_x_gate() {
        let mut sim = SparseStateVecAoS::new(2);
        sim.x(&[QubitId(0)]);

        assert_eq!(sim.num_amplitudes(), 1);
        assert_eq!(sim.get_amplitude(1), Complex64::new(1.0, 0.0));
        assert_eq!(sim.get_amplitude(0), Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_h_gate() {
        let mut sim = SparseStateVecAoS::new(1);
        sim.h(&[QubitId(0)]);

        assert_eq!(sim.num_amplitudes(), 2);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sim.get_amplitude(0).re - inv_sqrt2).abs() < 1e-10);
        assert!((sim.get_amplitude(1).re - inv_sqrt2).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state() {
        let mut sim = SparseStateVecAoS::new(2);
        sim.h(&[QubitId(0)]);
        sim.cx(&[QubitId(0), QubitId(1)]);

        assert_eq!(sim.num_amplitudes(), 2);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sim.get_amplitude(0b00).re - inv_sqrt2).abs() < 1e-10);
        assert!((sim.get_amplitude(0b11).re - inv_sqrt2).abs() < 1e-10);
    }

    #[test]
    fn test_h_roundtrip() {
        // H^2 = I, so applying H twice should return to original state
        let mut sim = SparseStateVecAoS::new(4);
        sim.h(&[QubitId(0)]); // Creates 2 amplitudes
        sim.h(&[QubitId(1)]); // Creates 4 amplitudes

        let orig_amps = sim.num_amplitudes();
        assert_eq!(orig_amps, 4);

        // Apply H twice on qubit 2
        sim.h(&[QubitId(2)]);
        assert_eq!(sim.num_amplitudes(), 8);
        sim.h(&[QubitId(2)]);
        assert_eq!(sim.num_amplitudes(), 4); // Should return to 4
    }

    #[test]
    fn test_h_large_scale() {
        // Test H gate with 64 amplitudes (still uses two-pointer path)
        let mut sim = SparseStateVecAoS::new(10);
        for q in 0..6 {
            sim.h(&[QubitId(q)]);
        }
        assert_eq!(sim.num_amplitudes(), 64);

        // Apply H on next qubit
        sim.h(&[QubitId(6)]);
        assert_eq!(sim.num_amplitudes(), 128);

        // Apply H again - should return to 64
        sim.h(&[QubitId(6)]);
        assert_eq!(sim.num_amplitudes(), 64);
    }

    #[test]
    fn test_sparsity_tracking() {
        let mut sim = SparseStateVecAoS::new(4);
        assert!(sim.sparsity() < 0.1); // Very sparse initially

        // Apply H to all qubits - becomes dense
        for q in 0..4 {
            sim.h(&[QubitId(q)]);
        }

        assert_eq!(sim.num_amplitudes(), 16); // 2^4 = 16
        assert!((sim.sparsity() - 1.0).abs() < 1e-10); // Fully dense
    }

    #[test]
    fn test_cx_gate() {
        let mut sim = SparseStateVecAoS::new(2);
        sim.x(&[QubitId(0)]); // |01⟩
        sim.cx(&[QubitId(0), QubitId(1)]);

        // Should be |11⟩
        assert_eq!(sim.num_amplitudes(), 1);
        assert_eq!(sim.get_amplitude(0b11), Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_cz_gate() {
        let mut sim = SparseStateVecAoS::new(2);
        // Create |++⟩ state
        sim.h(&[QubitId(0), QubitId(1)]);

        // Apply CZ
        sim.cz(&[QubitId(0), QubitId(1)]);

        // |11⟩ component should have negative sign
        let amp_11 = sim.get_amplitude(0b11);
        assert!(amp_11.re < 0.0);
    }

    #[test]
    fn test_batched_z_gate() {
        // Verify batched Z equals individual Z gates
        let mut sim1 = SparseStateVecAoS::new(5);
        let mut sim2 = SparseStateVecAoS::new(5);

        // Create superposition
        for q in 0..3 {
            sim1.h(&[QubitId(q)]);
            sim2.h(&[QubitId(q)]);
        }

        // Individual Z gates
        sim1.z(&[QubitId(0)]);
        sim1.z(&[QubitId(1)]);
        sim1.z(&[QubitId(2)]);

        // Batched Z gates
        sim2.z(&[QubitId(0), QubitId(1), QubitId(2)]);

        // Compare states
        for i in 0..8 {
            let a1 = sim1.get_amplitude(i);
            let a2 = sim2.get_amplitude(i);
            assert!(
                (a1 - a2).norm() < 1e-10,
                "Mismatch at {i}: {a1:?} vs {a2:?}"
            );
        }
    }

    #[test]
    fn test_batched_measurement() {
        // Test that batched measurement works and collapses correctly
        let mut sim = SparseStateVecAoS::with_seed(20, 42);

        // Create superposition on 4 qubits
        for q in 0..4 {
            sim.h(&[QubitId(q)]);
        }
        assert_eq!(sim.num_amplitudes(), 16);

        // Batched measurement of all 4 qubits
        let results = sim.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);

        assert_eq!(results.len(), 4);
        // After measuring all qubits, should collapse to single amplitude
        assert_eq!(sim.num_amplitudes(), 1);
    }

    #[test]
    fn test_prep_operations() {
        // Test mpz (prep to |0⟩) and mpnz (prep to |1⟩)
        let mut sim = SparseStateVecAoS::with_seed(10, 42);

        // Create superposition on 4 qubits
        for q in 0..4 {
            sim.h(&[QubitId(q)]);
        }
        assert_eq!(sim.num_amplitudes(), 16);

        // Prep qubits 0,1 to |0⟩
        let results = sim.mpz(&[QubitId(0), QubitId(1)]);
        assert_eq!(results.len(), 2);
        // Should collapse to 4 amplitudes (qubits 2,3 still in superposition)
        assert_eq!(sim.num_amplitudes(), 4);
        // Qubits 0,1 should now be |0⟩ - check all amplitudes have bits 0,1 clear
        for &(idx, _) in sim.iter() {
            assert_eq!(idx & 0b11, 0, "Qubit 0,1 should be 0 after mpz");
        }

        // Prep qubit 2 to |1⟩
        sim.mpnz(&[QubitId(2)]);
        assert_eq!(sim.num_amplitudes(), 2);
        // Qubit 2 should now be |1⟩
        for &(idx, _) in sim.iter() {
            assert!(idx & 0b100 != 0, "Qubit 2 should be 1 after mpnz");
        }
    }

    #[test]
    fn test_batched_s_gate() {
        // Verify batched S equals individual S gates
        let mut sim1 = SparseStateVecAoS::new(5);
        let mut sim2 = SparseStateVecAoS::new(5);

        // Create superposition
        for q in 0..3 {
            sim1.h(&[QubitId(q)]);
            sim2.h(&[QubitId(q)]);
        }

        // Individual S gates
        sim1.sz(&[QubitId(0)]);
        sim1.sz(&[QubitId(1)]);
        sim1.sz(&[QubitId(2)]);

        // Batched S gates
        sim2.sz(&[QubitId(0), QubitId(1), QubitId(2)]);

        // Compare states
        for i in 0..8 {
            let a1 = sim1.get_amplitude(i);
            let a2 = sim2.get_amplitude(i);
            assert!(
                (a1 - a2).norm() < 1e-10,
                "S gate mismatch at {i}: {a1:?} vs {a2:?}"
            );
        }

        // Also test Sdg
        let mut sim3 = SparseStateVecAoS::new(5);
        let mut sim4 = SparseStateVecAoS::new(5);
        for q in 0..3 {
            sim3.h(&[QubitId(q)]);
            sim4.h(&[QubitId(q)]);
        }
        sim3.szdg(&[QubitId(0)]);
        sim3.szdg(&[QubitId(1)]);
        sim3.szdg(&[QubitId(2)]);
        sim4.szdg(&[QubitId(0), QubitId(1), QubitId(2)]);

        for i in 0..8 {
            let a1 = sim3.get_amplitude(i);
            let a2 = sim4.get_amplitude(i);
            assert!(
                (a1 - a2).norm() < 1e-10,
                "Sdg gate mismatch at {i}: {a1:?} vs {a2:?}"
            );
        }
    }

    #[test]
    fn test_batched_y_gate() {
        // Verify batched Y equals individual Y gates
        let mut sim1 = SparseStateVecAoS::new(5);
        let mut sim2 = SparseStateVecAoS::new(5);

        // Create superposition
        for q in 0..3 {
            sim1.h(&[QubitId(q)]);
            sim2.h(&[QubitId(q)]);
        }

        // Individual Y gates
        sim1.y(&[QubitId(0)]);
        sim1.y(&[QubitId(1)]);
        sim1.y(&[QubitId(2)]);

        // Batched Y gates
        sim2.y(&[QubitId(0), QubitId(1), QubitId(2)]);

        // Compare states
        for i in 0..8 {
            let a1 = sim1.get_amplitude(i);
            let a2 = sim2.get_amplitude(i);
            assert!(
                (a1 - a2).norm() < 1e-10,
                "Y gate mismatch at {i}: {a1:?} vs {a2:?}"
            );
        }
    }

    #[test]
    fn test_batched_cy_gate() {
        // Verify batched CY equals individual CY gates
        let mut sim1 = SparseStateVecAoS::new(8);
        let mut sim2 = SparseStateVecAoS::new(8);

        // Create superposition on control and target qubits
        for q in 0..4 {
            sim1.h(&[QubitId(q)]);
            sim2.h(&[QubitId(q)]);
        }

        // Individual CY gates
        sim1.cy(&[QubitId(0), QubitId(4)]);
        sim1.cy(&[QubitId(1), QubitId(5)]);

        // Batched CY gates
        sim2.cy(&[QubitId(0), QubitId(4), QubitId(1), QubitId(5)]);

        // Compare states
        for i in 0..64 {
            let a1 = sim1.get_amplitude(i);
            let a2 = sim2.get_amplitude(i);
            assert!(
                (a1 - a2).norm() < 1e-10,
                "CY gate mismatch at {i}: {a1:?} vs {a2:?}"
            );
        }
    }

    #[test]
    fn test_batched_iswap() {
        // Verify batched iSWAP equals individual iSWAP gates
        let mut sim1 = SparseStateVecAoS::new(6);
        let mut sim2 = SparseStateVecAoS::new(6);

        // Create superposition
        for q in 0..4 {
            sim1.h(&[QubitId(q)]);
            sim2.h(&[QubitId(q)]);
        }

        // Individual iSWAP gates
        sim1.iswap(&[QubitId(0), QubitId(1)]);
        sim1.iswap(&[QubitId(2), QubitId(3)]);

        // Batched iSWAP gates
        sim2.iswap(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);

        // Compare states
        for i in 0..64 {
            let a1 = sim1.get_amplitude(i);
            let a2 = sim2.get_amplitude(i);
            assert!(
                (a1 - a2).norm() < 1e-10,
                "iSWAP mismatch at {i}: {a1:?} vs {a2:?}"
            );
        }
    }
}
