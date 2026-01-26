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

//! Sparse State Vector Simulator (SoA Layout)
//!
//! Uses a Structure-of-Arrays layout with double buffering. This design
//! is intended for scenarios where the state may grow large but starts sparse.
//!
//! **Note**: For typical sparse states (<1K amplitudes), the AoS version
//! (`SparseStateVecAoS`) is significantly faster (up to 10x) due to simpler
//! code paths and better cache locality. Use this SoA version only when:
//! - You expect the state to grow to thousands of amplitudes
//! - You need the double-buffering pattern for other reasons
//! - Future SIMD optimizations are planned
//!
//! ## Architecture
//!
//! 1. **SoA Layout**: Separate arrays for indices, real, imag parts
//!    - Better for SIMD when amplitude count is large
//!    - More cache misses for small states
//!
//! 2. **Double Buffering**: Two sets of arrays (A and B), swap roles each gate
//!    - Avoids allocation during gate operations
//!    - Adds complexity and cache pressure
//!
//! 3. **Binary Search**: Uses same algorithm as AoS for pair lookup

use crate::clifford_gateable::MeasurementResult;
use crate::{CliffordGateable, QuantumSimulator};
use num_complex::Complex64;
use pecos_core::QubitId;
use pecos_rng::{PecosRng, Rng, RngProbabilityExt, SeedableRng};
use std::fmt::Debug;
use wide::f64x4;

/// DOD-optimized sparse state vector using SoA layout and double buffering.
#[derive(Debug)]
pub struct SparseStateVecSoA<R = PecosRng>
where
    R: Rng,
{
    // ===== HOT DATA - touched every gate operation =====
    /// Basis state indices (sorted) - buffer A
    indices_a: Vec<usize>,
    /// Real parts of amplitudes - buffer A
    real_a: Vec<f64>,
    /// Imaginary parts of amplitudes - buffer A
    imag_a: Vec<f64>,

    /// Basis state indices (sorted) - buffer B
    indices_b: Vec<usize>,
    /// Real parts of amplitudes - buffer B
    real_b: Vec<f64>,
    /// Imaginary parts of amplitudes - buffer B
    imag_b: Vec<f64>,

    /// Which buffer is active (true = A, false = B)
    active_a: bool,
    /// Number of valid amplitudes in active buffer
    len: usize,

    // ===== SCRATCH DATA - reused across gate operations =====
    /// Positions of amplitudes with target bit=0 (reused across gates)
    scratch_low: Vec<u32>,
    /// Positions of amplitudes with target bit=1 (reused across gates)
    scratch_high: Vec<u32>,

    // ===== MERGE BUFFERS - for sorted-merge gate output =====
    /// Temporary storage for one sorted stream during merge
    merge_idx: Vec<usize>,
    merge_re: Vec<f64>,
    merge_im: Vec<f64>,

    // ===== COLD DATA - rarely accessed =====
    /// Number of qubits
    num_qubits: usize,
    /// Random number generator
    rng: R,
    /// Amplitude truncation threshold (0 = exact)
    epsilon: f64,
}

impl SparseStateVecSoA<PecosRng> {
    /// Create a new sparse state vector initialized to |0...0⟩
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self::with_rng(num_qubits, PecosRng::from_os_rng())
    }

    /// Create with a specific seed for reproducibility
    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_rng(num_qubits, PecosRng::seed_from_u64(seed))
    }
}

impl<R: Rng> SparseStateVecSoA<R> {
    /// Initial capacity for buffers (can grow if needed)
    const INITIAL_CAPACITY: usize = 64;

    /// Create with a custom RNG
    #[must_use]
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        let cap = Self::INITIAL_CAPACITY;

        // Initialize buffer A with |0⟩ state
        let mut indices_a = Vec::with_capacity(cap);
        let mut real_a = Vec::with_capacity(cap);
        let mut imag_a = Vec::with_capacity(cap);
        indices_a.push(0);
        real_a.push(1.0);
        imag_a.push(0.0);

        Self {
            indices_a,
            real_a,
            imag_a,
            indices_b: Vec::with_capacity(cap),
            real_b: Vec::with_capacity(cap),
            imag_b: Vec::with_capacity(cap),
            active_a: true,
            len: 1,
            scratch_low: Vec::new(),
            scratch_high: Vec::new(),
            merge_idx: Vec::new(),
            merge_re: Vec::new(),
            merge_im: Vec::new(),
            num_qubits,
            rng,
            epsilon: 0.0,
        }
    }

    /// Set the amplitude truncation threshold
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
        self.len
    }

    /// Get the sparsity ratio
    #[inline]
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        self.len as f64 / (1usize << self.num_qubits) as f64
    }

    /// Get amplitude at a specific basis state index (binary search)
    #[must_use]
    pub fn get_amplitude(&self, index: usize) -> Complex64 {
        let (indices, real, imag) = self.active_buffers();
        match indices[..self.len].binary_search(&index) {
            Ok(pos) => Complex64::new(real[pos], imag[pos]),
            Err(_) => Complex64::new(0.0, 0.0),
        }
    }

    /// Get probability of measuring a specific basis state
    #[inline]
    #[must_use]
    pub fn probability(&self, index: usize) -> f64 {
        let amp = self.get_amplitude(index);
        amp.re * amp.re + amp.im * amp.im
    }

    // =========================================================================
    // Buffer management
    // =========================================================================

    /// Get references to active buffers
    #[inline]
    fn active_buffers(&self) -> (&[usize], &[f64], &[f64]) {
        if self.active_a {
            (&self.indices_a, &self.real_a, &self.imag_a)
        } else {
            (&self.indices_b, &self.real_b, &self.imag_b)
        }
    }


    // =========================================================================
    // Single-qubit gate application
    // =========================================================================

    /// Apply single-qubit gate using two-pointer merge with sorted output.
    ///
    /// For small states (<= 8 amplitudes), falls back to binary search which has
    /// lower overhead. For larger states, the two-pointer merge produces two
    /// sorted output streams (bit=0 and bit=1 results), which are merged in O(k)
    /// instead of requiring an O(k log k) sort.
    #[inline]
    fn apply_single_qubit_gate(
        &mut self,
        q: usize,
        a_re: f64, a_im: f64,  // Gate matrix element [0,0]
        b_re: f64, b_im: f64,  // Gate matrix element [0,1]
        c_re: f64, c_im: f64,  // Gate matrix element [1,0]
        d_re: f64, d_im: f64,  // Gate matrix element [1,1]
    ) {
        let mask = 1usize << q;
        let len = self.len;
        let epsilon = self.epsilon;

        if len <= 8 {
            // Small state: binary search + sort (sort cost is negligible)
            let (src_indices, src_real, src_imag, dst_indices, dst_real, dst_imag) =
                if self.active_a {
                    (
                        &self.indices_a[..len], &self.real_a[..len], &self.imag_a[..len],
                        &mut self.indices_b, &mut self.real_b, &mut self.imag_b,
                    )
                } else {
                    (
                        &self.indices_b[..len], &self.real_b[..len], &self.imag_b[..len],
                        &mut self.indices_a, &mut self.real_a, &mut self.imag_a,
                    )
                };

            dst_indices.clear();
            dst_real.clear();
            dst_imag.clear();
            dst_indices.reserve(len * 2);
            dst_real.reserve(len * 2);
            dst_imag.reserve(len * 2);

            Self::apply_gate_binary_search(
                src_indices, src_real, src_imag,
                dst_indices, dst_real, dst_imag,
                mask, epsilon,
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
            );

            self.len = dst_indices.len();
            self.active_a = !self.active_a;
            self.sort_active();
        } else {
            // Larger state: two-pointer with sorted-merge output.
            // Produces two sorted streams and merges them in O(k),
            // avoiding the O(k log k) sort.
            self.apply_gate_sorted_merge(
                mask, len, epsilon,
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
            );
        }
    }

    /// Apply gate using two-pointer merge with sorted output.
    ///
    /// The two-pointer processes pairs in order of their low-partner index,
    /// so bit=0 results and bit=1 results are each produced in sorted order.
    /// We write them to separate buffers and merge in O(k).
    ///
    /// Buffer flow:
    /// 1. Read from active buffer (source)
    /// 2. Two-pointer writes bit=0 results → merge buffers, bit=1 results → inactive buffer
    /// 3. Merge both sorted streams → active buffer (source is free after step 1)
    #[allow(clippy::too_many_arguments)]
    fn apply_gate_sorted_merge(
        &mut self,
        mask: usize, len: usize, epsilon: f64,
        a_re: f64, a_im: f64, b_re: f64, b_im: f64,
        c_re: f64, c_im: f64, d_re: f64, d_im: f64,
    ) {
        let active = self.active_a;

        // Phase 1: Partition source indices into low (bit=0) and high (bit=1) positions
        self.scratch_low.clear();
        self.scratch_high.clear();
        for i in 0..len {
            let idx = if active { self.indices_a[i] } else { self.indices_b[i] };
            if idx & mask == 0 {
                self.scratch_low.push(i as u32);
            } else {
                self.scratch_high.push(i as u32);
            }
        }

        // Phase 2: Two-pointer walk producing split sorted output
        // bit=0 results → merge buffers (sorted by construction)
        // bit=1 results → inactive buffer (sorted by construction)
        self.merge_idx.clear();
        self.merge_re.clear();
        self.merge_im.clear();

        if active {
            self.indices_b.clear();
            self.real_b.clear();
            self.imag_b.clear();
        } else {
            self.indices_a.clear();
            self.real_a.clear();
            self.imag_a.clear();
        }

        let low_len = self.scratch_low.len();
        let high_len = self.scratch_high.len();
        let mut low_ptr = 0;
        let mut high_ptr = 0;

        loop {
            let have_low = low_ptr < low_len;
            let have_high = high_ptr < high_len;

            if !have_low && !have_high {
                break;
            }

            // Read source amplitudes using indexed access
            let (low_idx, low_re, low_im) = if have_low {
                let pos = self.scratch_low[low_ptr] as usize;
                if active {
                    (self.indices_a[pos], self.real_a[pos], self.imag_a[pos])
                } else {
                    (self.indices_b[pos], self.real_b[pos], self.imag_b[pos])
                }
            } else {
                (usize::MAX, 0.0, 0.0)
            };

            let (high_idx, high_re, high_im) = if have_high {
                let pos = self.scratch_high[high_ptr] as usize;
                if active {
                    (self.indices_a[pos], self.real_a[pos], self.imag_a[pos])
                } else {
                    (self.indices_b[pos], self.real_b[pos], self.imag_b[pos])
                }
            } else {
                (usize::MAX, 0.0, 0.0)
            };

            let high_partner = high_idx & !mask;

            if low_idx == high_partner {
                // Paired: apply full 2x2 gate matrix
                let new_low_re = a_re * low_re - a_im * low_im + b_re * high_re - b_im * high_im;
                let new_low_im = a_re * low_im + a_im * low_re + b_re * high_im + b_im * high_re;
                let new_high_re = c_re * low_re - c_im * low_im + d_re * high_re - d_im * high_im;
                let new_high_im = c_re * low_im + c_im * low_re + d_re * high_im + d_im * high_re;

                let norm_low = new_low_re * new_low_re + new_low_im * new_low_im;
                let norm_high = new_high_re * new_high_re + new_high_im * new_high_im;

                if norm_low > epsilon {
                    self.merge_idx.push(low_idx);
                    self.merge_re.push(new_low_re);
                    self.merge_im.push(new_low_im);
                }
                if norm_high > epsilon {
                    if active {
                        self.indices_b.push(high_idx);
                        self.real_b.push(new_high_re);
                        self.imag_b.push(new_high_im);
                    } else {
                        self.indices_a.push(high_idx);
                        self.real_a.push(new_high_re);
                        self.imag_a.push(new_high_im);
                    }
                }
                low_ptr += 1;
                high_ptr += 1;
            } else if low_idx < high_partner {
                // Unpaired low: pair with implicit zero high
                let new_low_re = a_re * low_re - a_im * low_im;
                let new_low_im = a_re * low_im + a_im * low_re;
                let new_high_re = c_re * low_re - c_im * low_im;
                let new_high_im = c_re * low_im + c_im * low_re;

                let norm_low = new_low_re * new_low_re + new_low_im * new_low_im;
                let norm_high = new_high_re * new_high_re + new_high_im * new_high_im;

                if norm_low > epsilon {
                    self.merge_idx.push(low_idx);
                    self.merge_re.push(new_low_re);
                    self.merge_im.push(new_low_im);
                }
                if norm_high > epsilon {
                    let high_result_idx = low_idx | mask;
                    if active {
                        self.indices_b.push(high_result_idx);
                        self.real_b.push(new_high_re);
                        self.imag_b.push(new_high_im);
                    } else {
                        self.indices_a.push(high_result_idx);
                        self.real_a.push(new_high_re);
                        self.imag_a.push(new_high_im);
                    }
                }
                low_ptr += 1;
            } else {
                // Unpaired high: pair with implicit zero low
                let new_low_re = b_re * high_re - b_im * high_im;
                let new_low_im = b_re * high_im + b_im * high_re;
                let new_high_re = d_re * high_re - d_im * high_im;
                let new_high_im = d_re * high_im + d_im * high_re;

                let norm_low = new_low_re * new_low_re + new_low_im * new_low_im;
                let norm_high = new_high_re * new_high_re + new_high_im * new_high_im;

                if norm_low > epsilon {
                    self.merge_idx.push(high_partner);
                    self.merge_re.push(new_low_re);
                    self.merge_im.push(new_low_im);
                }
                if norm_high > epsilon {
                    if active {
                        self.indices_b.push(high_idx);
                        self.real_b.push(new_high_re);
                        self.imag_b.push(new_high_im);
                    } else {
                        self.indices_a.push(high_idx);
                        self.real_a.push(new_high_re);
                        self.imag_a.push(new_high_im);
                    }
                }
                high_ptr += 1;
            }
        }

        // Phase 3: Merge the two sorted streams into the active buffer
        // merge buffers = sorted bit=0 results
        // inactive buffer = sorted bit=1 results
        // active buffer = free (was source, now done reading)
        self.merge_streams_into_active();
    }

    /// Merge bit=0 results (in merge buffers) with bit=1 results (in inactive buffer)
    /// into the active buffer. Both input streams are sorted; output is sorted.
    fn merge_streams_into_active(&mut self) {
        let n0 = self.merge_idx.len();

        if self.active_a {
            let n1 = self.indices_b.len();
            self.indices_a.clear();
            self.real_a.clear();
            self.imag_a.clear();
            self.indices_a.reserve(n0 + n1);
            self.real_a.reserve(n0 + n1);
            self.imag_a.reserve(n0 + n1);

            let mut i = 0;
            let mut j = 0;
            while i < n0 && j < n1 {
                let m_idx = self.merge_idx[i];
                let b_idx = self.indices_b[j];
                if m_idx < b_idx {
                    self.indices_a.push(m_idx);
                    self.real_a.push(self.merge_re[i]);
                    self.imag_a.push(self.merge_im[i]);
                    i += 1;
                } else {
                    self.indices_a.push(b_idx);
                    self.real_a.push(self.real_b[j]);
                    self.imag_a.push(self.imag_b[j]);
                    j += 1;
                }
            }
            while i < n0 {
                self.indices_a.push(self.merge_idx[i]);
                self.real_a.push(self.merge_re[i]);
                self.imag_a.push(self.merge_im[i]);
                i += 1;
            }
            while j < n1 {
                self.indices_a.push(self.indices_b[j]);
                self.real_a.push(self.real_b[j]);
                self.imag_a.push(self.imag_b[j]);
                j += 1;
            }
            self.len = self.indices_a.len();
        } else {
            let n1 = self.indices_a.len();
            self.indices_b.clear();
            self.real_b.clear();
            self.imag_b.clear();
            self.indices_b.reserve(n0 + n1);
            self.real_b.reserve(n0 + n1);
            self.imag_b.reserve(n0 + n1);

            let mut i = 0;
            let mut j = 0;
            while i < n0 && j < n1 {
                let m_idx = self.merge_idx[i];
                let a_idx = self.indices_a[j];
                if m_idx < a_idx {
                    self.indices_b.push(m_idx);
                    self.real_b.push(self.merge_re[i]);
                    self.imag_b.push(self.merge_im[i]);
                    i += 1;
                } else {
                    self.indices_b.push(a_idx);
                    self.real_b.push(self.real_a[j]);
                    self.imag_b.push(self.imag_a[j]);
                    j += 1;
                }
            }
            while i < n0 {
                self.indices_b.push(self.merge_idx[i]);
                self.real_b.push(self.merge_re[i]);
                self.imag_b.push(self.merge_im[i]);
                i += 1;
            }
            while j < n1 {
                self.indices_b.push(self.indices_a[j]);
                self.real_b.push(self.real_a[j]);
                self.imag_b.push(self.imag_a[j]);
                j += 1;
            }
            self.len = self.indices_b.len();
        }
        // active_a stays the same (output is in the active buffer)
    }

    /// Binary search path for small states (<= 8 amplitudes).
    #[inline]
    fn apply_gate_binary_search(
        src_indices: &[usize], src_real: &[f64], src_imag: &[f64],
        dst_indices: &mut Vec<usize>, dst_real: &mut Vec<f64>, dst_imag: &mut Vec<f64>,
        mask: usize, epsilon: f64,
        a_re: f64, a_im: f64, b_re: f64, b_im: f64,
        c_re: f64, c_im: f64, d_re: f64, d_im: f64,
    ) {
        let len = src_indices.len();

        // First pass: process all "low" indices (bit q=0)
        for i in 0..len {
            let idx = src_indices[i];
            if idx & mask != 0 {
                continue;
            }

            let (amp_re, amp_im) = (src_real[i], src_imag[i]);
            let paired_idx = idx | mask;

            let (paired_re, paired_im) = src_indices[i + 1..]
                .binary_search(&paired_idx)
                .ok()
                .map(|offset| (src_real[i + 1 + offset], src_imag[i + 1 + offset]))
                .unwrap_or((0.0, 0.0));

            let new_low_re = a_re * amp_re - a_im * amp_im + b_re * paired_re - b_im * paired_im;
            let new_low_im = a_re * amp_im + a_im * amp_re + b_re * paired_im + b_im * paired_re;
            let new_high_re = c_re * amp_re - c_im * amp_im + d_re * paired_re - d_im * paired_im;
            let new_high_im = c_re * amp_im + c_im * amp_re + d_re * paired_im + d_im * paired_re;

            let norm_low = new_low_re * new_low_re + new_low_im * new_low_im;
            let norm_high = new_high_re * new_high_re + new_high_im * new_high_im;

            if norm_low > epsilon {
                dst_indices.push(idx);
                dst_real.push(new_low_re);
                dst_imag.push(new_low_im);
            }
            if norm_high > epsilon {
                dst_indices.push(paired_idx);
                dst_real.push(new_high_re);
                dst_imag.push(new_high_im);
            }
        }

        // Second pass: handle unpaired "high" indices
        for i in 0..len {
            let idx = src_indices[i];
            if idx & mask == 0 {
                continue;
            }

            let paired_idx = idx & !mask;
            if src_indices[..i].binary_search(&paired_idx).is_ok() {
                continue;
            }

            let (amp_re, amp_im) = (src_real[i], src_imag[i]);

            let new_low_re = b_re * amp_re - b_im * amp_im;
            let new_low_im = b_re * amp_im + b_im * amp_re;
            let new_high_re = d_re * amp_re - d_im * amp_im;
            let new_high_im = d_re * amp_im + d_im * amp_re;

            let norm_low = new_low_re * new_low_re + new_low_im * new_low_im;
            let norm_high = new_high_re * new_high_re + new_high_im * new_high_im;

            if norm_low > epsilon {
                dst_indices.push(paired_idx);
                dst_real.push(new_low_re);
                dst_imag.push(new_low_im);
            }
            if norm_high > epsilon {
                dst_indices.push(idx);
                dst_real.push(new_high_re);
                dst_imag.push(new_high_im);
            }
        }
    }

    /// Sort the active buffer by index.
    ///
    /// Builds a permutation sorted by index, then applies it in-place using
    /// cycle sort. Marks visited positions by setting `perm[j] = j` to avoid
    /// a separate visited-flags allocation.
    #[inline]
    fn sort_active(&mut self) {
        let len = self.len;
        let (indices, real, imag) = if self.active_a {
            (&mut self.indices_a[..len], &mut self.real_a[..len], &mut self.imag_a[..len])
        } else {
            (&mut self.indices_b[..len], &mut self.real_b[..len], &mut self.imag_b[..len])
        };

        // Create permutation sorted by index
        let mut perm: Vec<usize> = (0..len).collect();
        perm.sort_unstable_by_key(|&i| indices[i]);

        // Apply permutation in-place using cycle sort.
        // Mark visited by setting perm[j] = j (no separate visited vec needed).
        for i in 0..len {
            if perm[i] == i {
                continue;
            }

            let mut j = i;
            let tmp_idx = indices[i];
            let tmp_re = real[i];
            let tmp_im = imag[i];

            loop {
                let k = perm[j];
                perm[j] = j;
                if k == i {
                    indices[j] = tmp_idx;
                    real[j] = tmp_re;
                    imag[j] = tmp_im;
                    break;
                }
                indices[j] = indices[k];
                real[j] = real[k];
                imag[j] = imag[k];
                j = k;
            }
        }
    }

    // =========================================================================
    // Optimized in-place gates (Z, S, CZ) with SIMD
    // =========================================================================

    /// Apply Z gate in-place with SIMD optimization.
    ///
    /// Uses the identity: sign = 1.0 - 2.0 * ((idx >> q) & 1)
    /// This gives 1.0 for bit=0 and -1.0 for bit=1.
    fn apply_z_inplace(&mut self, q: usize) {
        let len = self.len;
        let (indices, real, imag) = if self.active_a {
            (
                &self.indices_a[..len],
                &mut self.real_a[..len],
                &mut self.imag_a[..len],
            )
        } else {
            (
                &self.indices_b[..len],
                &mut self.real_b[..len],
                &mut self.imag_b[..len],
            )
        };

        // SIMD path: process 4 elements at a time
        let chunks = len / 4;

        for c in 0..chunks {
            let i = c * 4;

            // Compute signs: 1.0 for bit=0, -1.0 for bit=1
            let signs = f64x4::new([
                1.0 - 2.0 * ((indices[i] >> q) & 1) as f64,
                1.0 - 2.0 * ((indices[i + 1] >> q) & 1) as f64,
                1.0 - 2.0 * ((indices[i + 2] >> q) & 1) as f64,
                1.0 - 2.0 * ((indices[i + 3] >> q) & 1) as f64,
            ]);

            // Load, multiply, store
            let re = f64x4::from(&real[i..i + 4]);
            let im = f64x4::from(&imag[i..i + 4]);
            let new_re: [f64; 4] = (re * signs).into();
            let new_im: [f64; 4] = (im * signs).into();
            real[i..i + 4].copy_from_slice(&new_re);
            imag[i..i + 4].copy_from_slice(&new_im);
        }

        // Scalar remainder
        for i in (chunks * 4)..len {
            if indices[i] >> q & 1 != 0 {
                real[i] = -real[i];
                imag[i] = -imag[i];
            }
        }
    }

    /// Apply S gate in-place (multiply by i where bit is set)
    fn apply_s_inplace(&mut self, q: usize) {
        let mask = 1usize << q;
        let (indices, real, imag) = if self.active_a {
            (
                &self.indices_a[..self.len],
                &mut self.real_a[..self.len],
                &mut self.imag_a[..self.len],
            )
        } else {
            (
                &self.indices_b[..self.len],
                &mut self.real_b[..self.len],
                &mut self.imag_b[..self.len],
            )
        };

        for i in 0..self.len {
            if indices[i] & mask != 0 {
                // Multiply by i: (re, im) -> (-im, re)
                let tmp = real[i];
                real[i] = -imag[i];
                imag[i] = tmp;
            }
        }
    }

    /// Apply S-dagger gate in-place (multiply by -i where bit is set)
    fn apply_sdg_inplace(&mut self, q: usize) {
        let mask = 1usize << q;
        let (indices, real, imag) = if self.active_a {
            (
                &self.indices_a[..self.len],
                &mut self.real_a[..self.len],
                &mut self.imag_a[..self.len],
            )
        } else {
            (
                &self.indices_b[..self.len],
                &mut self.real_b[..self.len],
                &mut self.imag_b[..self.len],
            )
        };

        for i in 0..self.len {
            if indices[i] & mask != 0 {
                // Multiply by -i: (re, im) -> (im, -re)
                let tmp = real[i];
                real[i] = imag[i];
                imag[i] = -tmp;
            }
        }
    }

    /// Apply CZ gate in-place (flip sign where both bits are set)
    fn apply_cz_inplace(&mut self, q1: usize, q2: usize) {
        let mask1 = 1usize << q1;
        let mask2 = 1usize << q2;
        let both_mask = mask1 | mask2;

        let (indices, real, imag) = if self.active_a {
            (
                &self.indices_a[..self.len],
                &mut self.real_a[..self.len],
                &mut self.imag_a[..self.len],
            )
        } else {
            (
                &self.indices_b[..self.len],
                &mut self.real_b[..self.len],
                &mut self.imag_b[..self.len],
            )
        };

        for i in 0..self.len {
            if indices[i] & both_mask == both_mask {
                real[i] = -real[i];
                imag[i] = -imag[i];
            }
        }
    }

    // =========================================================================
    // Two-qubit gates
    // =========================================================================

    /// Apply X gate in-place: flip bit q on all indices.
    ///
    /// XOR all indices with mask, then re-sort. The in-place approach with
    /// sort is faster than a merge-based approach because:
    /// - XOR is a single contiguous-memory pass
    /// - After XOR, data has 2 sorted runs; sort_active handles this efficiently
    /// - Avoids the cache-unfriendly indirect reads of a merge approach
    #[inline]
    fn apply_x_inplace(&mut self, q: usize) {
        let mask = 1usize << q;
        let len = self.len;

        let indices = if self.active_a {
            &mut self.indices_a[..len]
        } else {
            &mut self.indices_b[..len]
        };

        for i in 0..len {
            indices[i] ^= mask;
        }

        self.sort_active();
    }

    /// Apply CX (CNOT) gate in-place: if control=1, flip target bit.
    ///
    /// Modifies indices in the active buffer and re-sorts. Avoids copying
    /// all three arrays to the destination buffer.
    #[inline]
    fn apply_cx_gate(&mut self, control: usize, target: usize) {
        let control_mask = 1usize << control;
        let target_mask = 1usize << target;
        let len = self.len;

        let indices = if self.active_a {
            &mut self.indices_a[..len]
        } else {
            &mut self.indices_b[..len]
        };

        for i in 0..len {
            if indices[i] & control_mask != 0 {
                indices[i] ^= target_mask;
            }
        }

        self.sort_active();
    }

    /// Apply SWAP gate in-place: swap bits q1 and q2.
    #[inline]
    fn apply_swap_gate(&mut self, q1: usize, q2: usize) {
        let mask1 = 1usize << q1;
        let mask2 = 1usize << q2;
        let len = self.len;

        let indices = if self.active_a {
            &mut self.indices_a[..len]
        } else {
            &mut self.indices_b[..len]
        };

        for i in 0..len {
            let bit1 = (indices[i] & mask1) != 0;
            let bit2 = (indices[i] & mask2) != 0;
            if bit1 != bit2 {
                indices[i] ^= mask1 | mask2;
            }
        }

        self.sort_active();
    }

    /// Normalize the state
    fn normalize(&mut self) {
        let (_, real, imag) = if self.active_a {
            (
                &self.indices_a[..self.len],
                &mut self.real_a[..self.len],
                &mut self.imag_a[..self.len],
            )
        } else {
            (
                &self.indices_b[..self.len],
                &mut self.real_b[..self.len],
                &mut self.imag_b[..self.len],
            )
        };

        let mut norm_sq = 0.0;
        for i in 0..self.len {
            norm_sq += real[i] * real[i] + imag[i] * imag[i];
        }

        if norm_sq > 0.0 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for i in 0..self.len {
                real[i] *= inv_norm;
                imag[i] *= inv_norm;
            }
        }
    }
}

// =============================================================================
// QuantumSimulator trait implementation
// =============================================================================

impl<R: Rng + Debug> QuantumSimulator for SparseStateVecSoA<R> {
    fn reset(&mut self) -> &mut Self {
        // Reset to |0⟩ state in buffer A
        self.indices_a.clear();
        self.real_a.clear();
        self.imag_a.clear();
        self.indices_a.push(0);
        self.real_a.push(1.0);
        self.imag_a.push(0.0);
        self.active_a = true;
        self.len = 1;
        self
    }
}

// =============================================================================
// CliffordGateable trait implementation
// =============================================================================

impl<R: Rng + Debug> CliffordGateable for SparseStateVecSoA<R> {
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        for &q in qubits {
            self.apply_single_qubit_gate(
                q.0,
                inv_sqrt2, 0.0,
                inv_sqrt2, 0.0,
                inv_sqrt2, 0.0,
                -inv_sqrt2, 0.0,
            );
        }
        self
    }

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_x_inplace(q.0);
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_single_qubit_gate(
                q.0,
                0.0, 0.0,   // a
                0.0, -1.0,  // b = -i
                0.0, 1.0,   // c = i
                0.0, 0.0,   // d
            );
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_z_inplace(q.0);
        }
        self
    }

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_s_inplace(q.0);
        }
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_sdg_inplace(q.0);
        }
        self
    }

    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_single_qubit_gate(
                q.0,
                0.5, 0.5,   // a = (1+i)/2
                0.5, -0.5,  // b = (1-i)/2
                0.5, -0.5,  // c = (1-i)/2
                0.5, 0.5,   // d = (1+i)/2
            );
        }
        self
    }

    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_single_qubit_gate(
                q.0,
                0.5, -0.5,  // a = (1-i)/2
                0.5, 0.5,   // b = (1+i)/2
                0.5, 0.5,   // c = (1+i)/2
                0.5, -0.5,  // d = (1-i)/2
            );
        }
        self
    }

    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_single_qubit_gate(
                q.0,
                0.5, 0.5,    // a = (1+i)/2
                -0.5, -0.5,  // b = -(1+i)/2
                0.5, 0.5,    // c = (1+i)/2
                0.5, 0.5,    // d = (1+i)/2
            );
        }
        self
    }

    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_single_qubit_gate(
                q.0,
                0.5, -0.5,   // a = (1-i)/2
                0.5, -0.5,   // b = (1-i)/2
                -0.5, 0.5,   // c = -(1-i)/2
                0.5, -0.5,   // d = (1-i)/2
            );
        }
        self
    }

    fn cx(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(qubits.len() % 2 == 0, "CX requires pairs of qubits");
        for pair in qubits.chunks_exact(2) {
            self.apply_cx_gate(pair[0].0, pair[1].0);
        }
        self
    }

    fn cz(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(qubits.len() % 2 == 0, "CZ requires pairs of qubits");
        for pair in qubits.chunks_exact(2) {
            self.apply_cz_inplace(pair[0].0, pair[1].0);
        }
        self
    }

    fn cy(&mut self, qubits: &[QubitId]) -> &mut Self {
        // CY = (I ⊗ S†) CX (I ⊗ S)
        for pair in qubits.chunks_exact(2) {
            self.apply_s_inplace(pair[1].0);
            self.apply_cx_gate(pair[0].0, pair[1].0);
            self.apply_sdg_inplace(pair[1].0);
        }
        self
    }

    fn swap(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(qubits.len() % 2 == 0, "SWAP requires pairs of qubits");
        for pair in qubits.chunks_exact(2) {
            self.apply_swap_gate(pair[0].0, pair[1].0);
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let mut results = Vec::with_capacity(qubits.len());

        for &q in qubits {
            let mask = 1usize << q.0;
            let len = self.len;

            // Calculate probability of measuring |1⟩
            let prob_one = {
                let (indices, real, imag) = self.active_buffers();
                let mut p = 0.0;
                for i in 0..len {
                    if indices[i] & mask != 0 {
                        p += real[i] * real[i] + imag[i] * imag[i];
                    }
                }
                p
            };

            let is_deterministic = prob_one < 1e-10 || prob_one > 1.0 - 1e-10;
            let outcome = self.rng.bernoulli(prob_one);

            results.push(MeasurementResult {
                outcome,
                is_deterministic,
            });

            // Collapse in-place: keep only consistent amplitudes.
            // Since we're filtering a sorted sequence, the result stays sorted.
            let keep_value = if outcome { mask } else { 0 };

            {
                let (indices, real, imag) = if self.active_a {
                    (&mut self.indices_a[..len], &mut self.real_a[..len], &mut self.imag_a[..len])
                } else {
                    (&mut self.indices_b[..len], &mut self.real_b[..len], &mut self.imag_b[..len])
                };

                let mut write = 0;
                for read in 0..len {
                    if indices[read] & mask == keep_value {
                        if write != read {
                            indices[write] = indices[read];
                            real[write] = real[read];
                            imag[write] = imag[read];
                        }
                        write += 1;
                    }
                }
                self.len = write;
            }

            self.normalize();
        }

        results
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
        let sim = SparseStateVecSoA::new(4);
        assert_eq!(sim.num_qubits(), 4);
        assert_eq!(sim.num_amplitudes(), 1);
        assert_eq!(sim.get_amplitude(0), Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_x_gate() {
        let mut sim = SparseStateVecSoA::new(2);
        sim.x(&[QubitId(0)]);

        assert_eq!(sim.num_amplitudes(), 1);
        assert_eq!(sim.get_amplitude(1), Complex64::new(1.0, 0.0));
        assert_eq!(sim.get_amplitude(0), Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_h_gate() {
        let mut sim = SparseStateVecSoA::new(1);
        sim.h(&[QubitId(0)]);

        assert_eq!(sim.num_amplitudes(), 2);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sim.get_amplitude(0).re - inv_sqrt2).abs() < 1e-10);
        assert!((sim.get_amplitude(1).re - inv_sqrt2).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state() {
        let mut sim = SparseStateVecSoA::new(2);
        sim.h(&[QubitId(0)]);
        sim.cx(&[QubitId(0), QubitId(1)]);

        assert_eq!(sim.num_amplitudes(), 2);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sim.get_amplitude(0b00).re - inv_sqrt2).abs() < 1e-10);
        assert!((sim.get_amplitude(0b11).re - inv_sqrt2).abs() < 1e-10);
    }

    #[test]
    fn test_z_gate() {
        let mut sim = SparseStateVecSoA::new(1);
        sim.h(&[QubitId(0)]);
        sim.z(&[QubitId(0)]);

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sim.get_amplitude(0).re - inv_sqrt2).abs() < 1e-10);
        assert!((sim.get_amplitude(1).re + inv_sqrt2).abs() < 1e-10); // Sign flipped
    }

    #[test]
    fn test_cz_gate() {
        let mut sim = SparseStateVecSoA::new(2);
        sim.h(&[QubitId(0), QubitId(1)]);
        sim.cz(&[QubitId(0), QubitId(1)]);

        // |11⟩ should have negative amplitude
        assert!(sim.get_amplitude(0b11).re < 0.0);
    }

    #[test]
    fn test_cx_gate() {
        let mut sim = SparseStateVecSoA::new(2);
        sim.x(&[QubitId(0)]); // |01⟩
        sim.cx(&[QubitId(0), QubitId(1)]);

        // Should be |11⟩
        assert_eq!(sim.num_amplitudes(), 1);
        assert!((sim.get_amplitude(0b11).re - 1.0).abs() < 1e-10);
    }
}
