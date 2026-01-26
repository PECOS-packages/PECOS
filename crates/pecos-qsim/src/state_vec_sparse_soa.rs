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

use crate::clifford_frame::{CliffordFrame, PauliAxis, ELEMENT_MATRIX, PHASE_COCYCLE, PHASE_ROOTS};
use crate::clifford_gateable::MeasurementResult;
use crate::{CliffordGateable, QuantumSimulator};
use num_complex::Complex64;
use pecos_core::QubitId;
use pecos_rng::{PecosRng, Rng, RngProbabilityExt, SeedableRng};
use std::fmt::Debug;

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

    // ===== CLIFFORD FRAME - per-qubit deferred single-qubit Cliffords =====
    /// Per-qubit Clifford frame index (mod global phase) for Heisenberg lookups.
    frames: Vec<CliffordFrame>,
    /// Per-qubit accumulated phase as 8th-root-of-unity index (0-7).
    /// Tracks the exact global phase: actual_matrix = e^{i*phase*π/4} * ELEMENT_MATRIX[frame].
    frame_phases: Vec<u8>,

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
            frames: vec![CliffordFrame::IDENTITY; num_qubits],
            frame_phases: vec![0; num_qubits],
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

    /// Get the number of non-zero amplitudes.
    /// Flushes any deferred Clifford frames first.
    #[inline]
    pub fn num_amplitudes(&mut self) -> usize {
        self.flush_all_frames();
        self.len
    }

    /// Get the sparsity ratio
    #[inline]
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        self.len as f64 / (1usize << self.num_qubits) as f64
    }

    /// Flush all non-identity Clifford frames by physically applying them.
    pub fn flush_all_frames(&mut self) {
        for q in 0..self.num_qubits {
            self.flush_frame(q);
        }
    }

    /// Get amplitude at a specific basis state index (binary search).
    /// Flushes all Clifford frames first to ensure the physical state is current.
    #[must_use]
    pub fn get_amplitude(&mut self, index: usize) -> Complex64 {
        self.flush_all_frames();
        let (indices, real, imag) = self.active_buffers();
        match indices[..self.len].binary_search(&index) {
            Ok(pos) => Complex64::new(real[pos], imag[pos]),
            Err(_) => Complex64::new(0.0, 0.0),
        }
    }

    /// Get probability of measuring a specific basis state.
    /// Flushes all Clifford frames first.
    #[inline]
    #[must_use]
    pub fn probability(&mut self, index: usize) -> f64 {
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
// Frame flush methods
// =============================================================================

impl<R: Rng> SparseStateVecSoA<R> {
    /// Flush the Clifford frame on qubit `q` by physically applying the
    /// accumulated gate (reconstructed from frame index + phase).
    /// Resets the frame to identity afterwards.
    fn flush_frame(&mut self, q: usize) {
        let idx = self.frames[q].index() as usize;
        let phase = self.frame_phases[q];

        if idx == 0 && phase == 0 {
            return; // true identity, nothing to do
        }

        if idx == 0 {
            // Frame is identity with a global phase -- multiply all amplitudes
            // by the phase scalar. This is cheaper than a full gate application.
            let [cos_t, sin_t] = PHASE_ROOTS[phase as usize];
            let (real, imag) = if self.active_a {
                (&mut self.real_a[..self.len], &mut self.imag_a[..self.len])
            } else {
                (&mut self.real_b[..self.len], &mut self.imag_b[..self.len])
            };
            for i in 0..self.len {
                let r = real[i];
                let im = imag[i];
                real[i] = r * cos_t - im * sin_t;
                imag[i] = r * sin_t + im * cos_t;
            }
        } else {
            // Reconstruct the full 2x2 matrix: phase * ELEMENT_MATRIX[idx]
            let m = ELEMENT_MATRIX[idx];
            let [cos_t, sin_t] = PHASE_ROOTS[phase as usize];
            // Multiply each complex entry [re, im] by (cos_t + i*sin_t)
            let a_re = m[0] * cos_t - m[1] * sin_t;
            let a_im = m[0] * sin_t + m[1] * cos_t;
            let b_re = m[2] * cos_t - m[3] * sin_t;
            let b_im = m[2] * sin_t + m[3] * cos_t;
            let c_re = m[4] * cos_t - m[5] * sin_t;
            let c_im = m[4] * sin_t + m[5] * cos_t;
            let d_re = m[6] * cos_t - m[7] * sin_t;
            let d_im = m[6] * sin_t + m[7] * cos_t;
            self.apply_single_qubit_gate(q, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im);
        }

        self.frames[q] = CliffordFrame::IDENTITY;
        self.frame_phases[q] = 0;
    }

    /// Compose a gate into qubit q's frame using the phase cocycle table.
    /// `gate_idx` is the Clifford index of the gate. `gate_delta` is the
    /// phase correction from the standard gate matrix to the element matrix.
    #[inline]
    fn compose_frame(&mut self, q: usize, gate_idx: CliffordFrame, gate_delta: u8) {
        let old = self.frames[q].index() as usize;
        self.frames[q] = self.frames[q].compose(gate_idx);
        self.frame_phases[q] =
            (self.frame_phases[q] + PHASE_COCYCLE[old][gate_idx.index() as usize] + gate_delta) % 8;
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
        self.frames.fill(CliffordFrame::IDENTITY);
        self.frame_phases.fill(0);
        self
    }
}

// =============================================================================
// CliffordGateable trait implementation
// =============================================================================

impl<R: Rng + Debug> CliffordGateable for SparseStateVecSoA<R> {
    // ---- Single-qubit Clifford gates: O(1) frame composition ----

    // -- Pauli gates (delta: X=0, Y=6, Z=0) --

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::PAULI_X, 0);
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::PAULI_Y, 6);
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::PAULI_Z, 0);
        }
        self
    }

    // -- S-like gates (delta: S=0, Sdg=0, SX=0, SXdg=7, SY=1, SYdg=7) --

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::S_GATE, 0);
        }
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::SDG_GATE, 0);
        }
        self
    }

    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::SX_GATE, 0);
        }
        self
    }

    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::SX_DG_GATE, 7);
        }
        self
    }

    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::SY_GATE, 1);
        }
        self
    }

    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::SY_DG_GATE, 7);
        }
        self
    }

    // -- H-like gates (delta: H=0) --

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::H_GATE, 0);
        }
        self
    }

    // ---- Two-qubit gates: flush both frames, then apply physically ----

    fn cx(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(qubits.len() % 2 == 0, "CX requires pairs of qubits");
        for pair in qubits.chunks_exact(2) {
            let (c, t) = (pair[0].0, pair[1].0);
            self.flush_frame(c);
            self.flush_frame(t);
            self.apply_cx_gate(c, t);
        }
        self
    }

    fn cz(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(qubits.len() % 2 == 0, "CZ requires pairs of qubits");
        for pair in qubits.chunks_exact(2) {
            let (c, t) = (pair[0].0, pair[1].0);
            self.flush_frame(c);
            self.flush_frame(t);
            self.apply_cz_inplace(c, t);
        }
        self
    }

    fn cy(&mut self, qubits: &[QubitId]) -> &mut Self {
        // CY = (I tensor Sdg) . CX . (I tensor S)
        for pair in qubits.chunks_exact(2) {
            let (c, t) = (pair[0].0, pair[1].0);
            // Compose S on target frame, then flush both, apply CX, compose Sdg on target
            self.compose_frame(t, CliffordFrame::S_GATE, 0);
            self.flush_frame(c);
            self.flush_frame(t);
            self.apply_cx_gate(c, t);
            self.compose_frame(t, CliffordFrame::SDG_GATE, 0);
        }
        self
    }

    fn swap(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(qubits.len() % 2 == 0, "SWAP requires pairs of qubits");
        for pair in qubits.chunks_exact(2) {
            let (c, t) = (pair[0].0, pair[1].0);
            self.flush_frame(c);
            self.flush_frame(t);
            self.apply_swap_gate(c, t);
        }
        self
    }

    // ---- Measurement: check Z-image to avoid unnecessary flush ----

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let mut results = Vec::with_capacity(qubits.len());

        for &q in qubits {
            let z_img = self.frames[q.0].z_image();

            match z_img.axis {
                PauliAxis::Z => {
                    // Z maps to +/-Z: can measure physically without flush.
                    // The physical state collapses to a Z eigenstate, and the
                    // frame remains in place (logical state = frame * physical).
                    let result = self.physical_mz(q.0);
                    let result = if z_img.positive {
                        result
                    } else {
                        MeasurementResult {
                            outcome: !result.outcome,
                            is_deterministic: result.is_deterministic,
                        }
                    };
                    results.push(result);
                }
                _ => {
                    // Z maps to +/-X or +/-Y: must flush frame first
                    self.flush_frame(q.0);
                    results.push(self.physical_mz(q.0));
                }
            }
        }

        results
    }
}

impl<R: Rng> SparseStateVecSoA<R> {
    /// Physical Z-basis measurement (no frame logic).
    fn physical_mz(&mut self, q: usize) -> MeasurementResult {
        let mask = 1usize << q;
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

        let result = MeasurementResult {
            outcome,
            is_deterministic,
        };

        // Collapse in-place: keep only consistent amplitudes.
        let keep_value = if outcome { mask } else { 0 };

        {
            let (indices, real, imag) = if self.active_a {
                (
                    &mut self.indices_a[..len],
                    &mut self.real_a[..len],
                    &mut self.imag_a[..len],
                )
            } else {
                (
                    &mut self.indices_b[..len],
                    &mut self.real_b[..len],
                    &mut self.imag_b[..len],
                )
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
        result
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
        let mut sim = SparseStateVecSoA::new(4);
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
