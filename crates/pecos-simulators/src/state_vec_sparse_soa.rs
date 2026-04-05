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

//! Sparse State Vector Simulator (`SoA` Layout)
//!
//! Uses a Structure-of-Arrays layout with double buffering. This design
//! is intended for scenarios where the state may grow large but starts sparse.
//!
//! **Note**: For typical sparse states (<1K amplitudes), the `AoS` version
//! (`SparseStateVecAoS`) is significantly faster (up to 10x) due to simpler
//! code paths and better cache locality. Use this `SoA` version only when:
//! - You expect the state to grow to thousands of amplitudes
//! - You need the double-buffering pattern for other reasons
//! - Future SIMD optimizations are planned
//!
//! ## Architecture
//!
//! 1. **`SoA` Layout**: Separate arrays for indices, real, imag parts
//!    - Better for SIMD when amplitude count is large
//!    - More cache misses for small states
//!
//! 2. **Double Buffering**: Two sets of arrays (A and B), swap roles each gate
//!    - Avoids allocation during gate operations
//!    - Adds complexity and cache pressure
//!
//! 3. **Binary Search**: Uses same algorithm as `AoS` for pair lookup

use crate::clifford_frame::{CliffordFrame, ELEMENT_MATRIX, PHASE_COCYCLE, PHASE_ROOTS, PauliAxis};
use crate::clifford_gateable::MeasurementResult;
use crate::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};
use num_complex::Complex64;
use pecos_core::{Angle64, QubitId, RngManageable};
use pecos_random::{PecosRng, Rng, RngProbabilityExt, SeedableRng};
use std::fmt::Debug;
use wide::f64x4;

/// DOD-optimized sparse state vector using `SoA` layout and double buffering.
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
    /// Reusable permutation buffer for `sort_active`
    sort_perm: Vec<usize>,

    // ===== MERGE BUFFERS - for sorted-merge gate output =====
    /// Temporary storage for one sorted stream during merge
    merge_idx: Vec<usize>,
    merge_re: Vec<f64>,
    merge_im: Vec<f64>,

    // ===== CLIFFORD FRAME - per-qubit deferred single-qubit Cliffords =====
    /// Per-qubit Clifford frame index (mod global phase) for Heisenberg lookups.
    frames: Vec<CliffordFrame>,
    /// Per-qubit accumulated phase as 8th-root-of-unity index (0-7).
    /// Tracks the exact global phase: `actual_matrix` = e^{i*phase*π/4} * `ELEMENT_MATRIX`[frame].
    frame_phases: Vec<u8>,

    /// Deferred sorting flag. When true, indices may not be in sorted order.
    /// Index-permuting gates (iSWAP) set this; operations needing sorted order
    /// (merge, binary search) call `ensure_sorted()` first.
    needs_sort: bool,

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
        Self::with_rng(num_qubits, rand::make_rng())
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
            sort_perm: Vec::new(),
            merge_idx: Vec::new(),
            merge_re: Vec::new(),
            merge_im: Vec::new(),
            needs_sort: false,
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
    #[allow(clippy::cast_precision_loss)] // sparsity ratio
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
        self.ensure_sorted();
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
    // Rotation kernel helpers
    // =========================================================================

    /// Apply a diagonal RZ rotation in-place: e^{-i*theta/2} to |0> and e^{i*theta/2} to |1>.
    /// Takes precomputed cos(theta/2) and sin(theta/2).
    /// Branch-free with SIMD (f64x4) for the inner loop.
    #[inline]
    #[allow(clippy::cast_precision_loss)] // bit extraction (0 or 1) as f64
    fn apply_rz_kernel(&mut self, q: usize, cos: f64, sin: f64) {
        let shift = q;
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

        let cos_v = f64x4::splat(cos);
        let sin_v = f64x4::splat(sin);
        let chunks = len / 4;
        for c in 0..chunks {
            let base = c * 4;
            // Branch-free sign: +1.0 if bit==0, -1.0 if bit==1
            let s0 = 1.0 - 2.0 * ((indices[base] >> shift) & 1) as f64;
            let s1 = 1.0 - 2.0 * ((indices[base + 1] >> shift) & 1) as f64;
            let s2 = 1.0 - 2.0 * ((indices[base + 2] >> shift) & 1) as f64;
            let s3 = 1.0 - 2.0 * ((indices[base + 3] >> shift) & 1) as f64;
            let sign = f64x4::new([s0, s1, s2, s3]);
            let signed_sin = sign * sin_v;

            let r = f64x4::from(&real[base..base + 4]);
            let im = f64x4::from(&imag[base..base + 4]);
            // |0>: cos*r + sin*im,  cos*im - sin*r
            // |1>: cos*r - sin*im,  cos*im + sin*r
            // Unified: cos*r + sign*sin*im, cos*im - sign*sin*r
            let new_re: [f64; 4] = (cos_v * r + signed_sin * im).into();
            let new_im: [f64; 4] = (cos_v * im - signed_sin * r).into();
            real[base..base + 4].copy_from_slice(&new_re);
            imag[base..base + 4].copy_from_slice(&new_im);
        }
        // Scalar remainder
        for i in (chunks * 4)..len {
            let sign = 1.0 - 2.0 * ((indices[i] >> shift) & 1) as f64;
            let r = real[i];
            let im = imag[i];
            real[i] = cos * r + sign * sin * im;
            imag[i] = cos * im - sign * sin * r;
        }
    }

    /// Apply a diagonal RZZ rotation in-place. Takes precomputed cos(theta/2) and sin(theta/2).
    /// Same parity (bit1==bit2): multiply by e^{-i*theta/2}
    /// Different parity: multiply by e^{i*theta/2}
    #[inline]
    fn apply_rzz_kernel(&mut self, q1: usize, q2: usize, cos: f64, sin: f64) {
        // Precompute both phase options:
        //   same parity: e^{-i*theta/2} = (cos, -sin)
        //   diff parity: e^{i*theta/2}  = (cos, sin)
        let cos_same = cos;
        let sin_same = -sin;
        let cos_diff = cos;
        let sin_diff = sin;

        let mask1 = 1usize << q1;
        let mask2 = 1usize << q2;
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
        for i in 0..len {
            let bit1 = (indices[i] & mask1) != 0;
            let bit2 = (indices[i] & mask2) != 0;
            let r = real[i];
            let im = imag[i];
            if bit1 == bit2 {
                real[i] = cos_same * r - sin_same * im;
                imag[i] = sin_same * r + cos_same * im;
            } else {
                real[i] = cos_diff * r - sin_diff * im;
                imag[i] = sin_diff * r + cos_diff * im;
            }
        }
    }

    /// Apply a two-qubit parity-flip gate (RXX or RYY).
    ///
    /// These gates pair each amplitude at index `idx` with its partner at `idx ^ both_mask`
    /// where `both_mask = (1 << q1) | (1 << q2)`. The partner has both target qubit bits flipped.
    ///
    /// Parameters:
    /// - `same_sin_sign`: sign of the imaginary off-diagonal for same-parity pairs (00<->11).
    ///   For RXX: -1.0 (matrix uses -i*sin). For RYY: +1.0 (matrix uses +i*sin).
    /// - `diff_sin_sign`: sign for different-parity pairs (01<->10).
    ///   For RXX: -1.0. For RYY: -1.0.
    #[inline]
    fn apply_parity_flip_gate(
        &mut self,
        q1: usize,
        q2: usize,
        cos: f64,
        sin: f64,
        same_sin_sign: f64,
        diff_sin_sign: f64,
    ) {
        struct PairInfo {
            self_pos: u32,
            partner_pos: u32, // u32::MAX if partner doesn't exist
            self_idx: usize,
            partner_idx: usize,
            same_parity: bool,
        }

        self.ensure_sorted();
        let both_mask = (1usize << q1) | (1usize << q2);
        let mask1 = 1usize << q1;
        let mask2 = 1usize << q2;
        let len = self.len;
        let epsilon = self.epsilon;
        let active = self.active_a;

        // Read from active buffer, write to inactive buffer
        let src_idx = if active {
            &self.indices_a[..len]
        } else {
            &self.indices_b[..len]
        };

        // Build a quick lookup: for each amplitude, mark whether it's been processed.
        // We process each pair only once (from the side with lower index).
        // Use scratch_low as a bitset (repurposed; will be restored by clear).
        self.scratch_low.clear();
        self.scratch_low.resize(len, 0);

        // Phase 1: Identify pairs. For each amplitude, find its partner.
        // Process from lower index side only.
        let mut pairs: Vec<PairInfo> = Vec::with_capacity(len);

        for i in 0..len {
            if self.scratch_low[i] != 0 {
                continue; // Already processed as partner
            }

            let idx = src_idx[i];
            let partner_idx = idx ^ both_mask;
            let same = ((idx & mask1) != 0) == ((idx & mask2) != 0);

            if partner_idx > idx {
                // We're the lower index in the pair, look for partner
                let partner_pos = src_idx.binary_search(&partner_idx).ok();
                if let Some(pp) = partner_pos {
                    self.scratch_low[pp] = 1; // Mark partner as processed
                    #[allow(clippy::cast_possible_truncation)] // position index fits in u32
                    pairs.push(PairInfo {
                        self_pos: i as u32,
                        partner_pos: pp as u32,
                        self_idx: idx,
                        partner_idx,
                        same_parity: same,
                    });
                } else {
                    #[allow(clippy::cast_possible_truncation)] // position index fits in u32
                    pairs.push(PairInfo {
                        self_pos: i as u32,
                        partner_pos: u32::MAX,
                        self_idx: idx,
                        partner_idx,
                        same_parity: same,
                    });
                }
            } else {
                // partner_idx < idx: partner should have been processed first.
                // If partner doesn't exist, we're unpaired from the high side.
                let partner_pos = src_idx.binary_search(&partner_idx).ok();
                if partner_pos.is_some() {
                    // Partner exists and was processed (or will be) -- skip
                    // This shouldn't happen since partner has lower index and would
                    // have processed us. But guard against it.
                    continue;
                }
                // Unpaired from high side
                #[allow(clippy::cast_possible_truncation)] // position index fits in u32
                pairs.push(PairInfo {
                    self_pos: i as u32,
                    partner_pos: u32::MAX,
                    self_idx: idx,
                    partner_idx,
                    same_parity: same,
                });
            }
        }

        // Phase 2: Apply the gate and write to destination buffer
        if active {
            self.indices_b.clear();
            self.real_b.clear();
            self.imag_b.clear();
            self.indices_b.reserve(len * 2);
            self.real_b.reserve(len * 2);
            self.imag_b.reserve(len * 2);
        } else {
            self.indices_a.clear();
            self.real_a.clear();
            self.imag_a.clear();
            self.indices_a.reserve(len * 2);
            self.real_a.reserve(len * 2);
            self.imag_a.reserve(len * 2);
        }

        for pair in &pairs {
            let (r_self, im_self) = if active {
                (
                    self.real_a[pair.self_pos as usize],
                    self.imag_a[pair.self_pos as usize],
                )
            } else {
                (
                    self.real_b[pair.self_pos as usize],
                    self.imag_b[pair.self_pos as usize],
                )
            };

            let (r_partner, im_partner) = if pair.partner_pos == u32::MAX {
                (0.0, 0.0)
            } else if active {
                (
                    self.real_a[pair.partner_pos as usize],
                    self.imag_a[pair.partner_pos as usize],
                )
            } else {
                (
                    self.real_b[pair.partner_pos as usize],
                    self.imag_b[pair.partner_pos as usize],
                )
            };

            // Select sign for this parity group
            let s = if pair.same_parity {
                same_sin_sign
            } else {
                diff_sin_sign
            };
            let ss = s * sin; // signed sin

            // Matrix: [[cos, s*i*sin], [s*i*sin, cos]]  (symmetric)
            // (s*i*sin) * (r + i*im) = s*(-sin*im + i*sin*r) = (-s*sin*im, s*sin*r)
            let new_self_re = cos * r_self - ss * im_partner;
            let new_self_im = cos * im_self + ss * r_partner;
            let new_partner_re = -ss * im_self + cos * r_partner;
            let new_partner_im = ss * r_self + cos * im_partner;

            let norm_self = new_self_re * new_self_re + new_self_im * new_self_im;
            let norm_partner = new_partner_re * new_partner_re + new_partner_im * new_partner_im;

            if norm_self > epsilon {
                if active {
                    self.indices_b.push(pair.self_idx);
                    self.real_b.push(new_self_re);
                    self.imag_b.push(new_self_im);
                } else {
                    self.indices_a.push(pair.self_idx);
                    self.real_a.push(new_self_re);
                    self.imag_a.push(new_self_im);
                }
            }
            if norm_partner > epsilon {
                if active {
                    self.indices_b.push(pair.partner_idx);
                    self.real_b.push(new_partner_re);
                    self.imag_b.push(new_partner_im);
                } else {
                    self.indices_a.push(pair.partner_idx);
                    self.real_a.push(new_partner_re);
                    self.imag_a.push(new_partner_im);
                }
            }
        }

        self.len = if active {
            self.indices_b.len()
        } else {
            self.indices_a.len()
        };
        self.active_a = !active;
        self.sort_active();
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
    #[allow(clippy::too_many_arguments)] // 2x2 unitary matrix elements (re/im pairs)
    #[inline]
    fn apply_single_qubit_gate(
        &mut self,
        q: usize,
        a_re: f64,
        a_im: f64, // Gate matrix element [0,0]
        b_re: f64,
        b_im: f64, // Gate matrix element [0,1]
        c_re: f64,
        c_im: f64, // Gate matrix element [1,0]
        d_re: f64,
        d_im: f64, // Gate matrix element [1,1]
    ) {
        self.ensure_sorted();
        let mask = 1usize << q;
        let len = self.len;
        let epsilon = self.epsilon;

        if len <= 8 {
            // Small state: binary search + sort (sort cost is negligible)
            let (src_indices, src_real, src_imag, dst_indices, dst_real, dst_imag) =
                if self.active_a {
                    (
                        &self.indices_a[..len],
                        &self.real_a[..len],
                        &self.imag_a[..len],
                        &mut self.indices_b,
                        &mut self.real_b,
                        &mut self.imag_b,
                    )
                } else {
                    (
                        &self.indices_b[..len],
                        &self.real_b[..len],
                        &self.imag_b[..len],
                        &mut self.indices_a,
                        &mut self.real_a,
                        &mut self.imag_a,
                    )
                };

            dst_indices.clear();
            dst_real.clear();
            dst_imag.clear();
            dst_indices.reserve(len * 2);
            dst_real.reserve(len * 2);
            dst_imag.reserve(len * 2);

            Self::apply_gate_binary_search(
                src_indices,
                src_real,
                src_imag,
                dst_indices,
                dst_real,
                dst_imag,
                mask,
                epsilon,
                a_re,
                a_im,
                b_re,
                b_im,
                c_re,
                c_im,
                d_re,
                d_im,
            );

            self.len = dst_indices.len();
            self.active_a = !self.active_a;
            self.sort_active();
        } else {
            // Larger state: two-pointer with sorted-merge output.
            // Produces two sorted streams and merges them in O(k),
            // avoiding the O(k log k) sort.
            self.apply_gate_sorted_merge(
                mask, len, epsilon, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
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
        mask: usize,
        len: usize,
        epsilon: f64,
        a_re: f64,
        a_im: f64,
        b_re: f64,
        b_im: f64,
        c_re: f64,
        c_im: f64,
        d_re: f64,
        d_im: f64,
    ) {
        let active = self.active_a;

        // Phase 1: Partition source indices into low (bit=0) and high (bit=1) positions
        self.scratch_low.clear();
        self.scratch_high.clear();
        for i in 0..len {
            let idx = if active {
                self.indices_a[i]
            } else {
                self.indices_b[i]
            };
            if idx & mask == 0 {
                #[allow(clippy::cast_possible_truncation)] // amplitude index fits in u32
                self.scratch_low.push(i as u32);
            } else {
                #[allow(clippy::cast_possible_truncation)] // amplitude index fits in u32
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

            match low_idx.cmp(&high_partner) {
                std::cmp::Ordering::Equal => {
                    // Paired: apply full 2x2 gate matrix
                    let new_low_re =
                        a_re * low_re - a_im * low_im + b_re * high_re - b_im * high_im;
                    let new_low_im =
                        a_re * low_im + a_im * low_re + b_re * high_im + b_im * high_re;
                    let new_high_re =
                        c_re * low_re - c_im * low_im + d_re * high_re - d_im * high_im;
                    let new_high_im =
                        c_re * low_im + c_im * low_re + d_re * high_im + d_im * high_re;

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
                }
                std::cmp::Ordering::Less => {
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
                }
                std::cmp::Ordering::Greater => {
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
    #[allow(clippy::too_many_arguments)] // SoA hot path: src/dst arrays + 2x2 unitary elements
    #[inline]
    fn apply_gate_binary_search(
        src_indices: &[usize],
        src_real: &[f64],
        src_imag: &[f64],
        dst_indices: &mut Vec<usize>,
        dst_real: &mut Vec<f64>,
        dst_imag: &mut Vec<f64>,
        mask: usize,
        epsilon: f64,
        a_re: f64,
        a_im: f64,
        b_re: f64,
        b_im: f64,
        c_re: f64,
        c_im: f64,
        d_re: f64,
        d_im: f64,
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
                .map_or((0.0, 0.0), |offset| {
                    (src_real[i + 1 + offset], src_imag[i + 1 + offset])
                });

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

    /// Ensure the active buffer is in sorted order.
    /// Called before operations that depend on sorted indices (merges, binary search).
    #[inline]
    fn ensure_sorted(&mut self) {
        if self.needs_sort {
            self.sort_active();
            self.needs_sort = false;
        }
    }

    /// Sort the active buffer by index.
    ///
    /// Sort the active buffer only if it's not already sorted.
    /// Skips the full sort machinery when the indices are in order (common
    /// after CX/SWAP when indices happen to stay sorted).
    #[inline]
    fn sort_active_if_needed(&mut self) {
        let len = self.len;
        let indices = if self.active_a {
            &self.indices_a[..len]
        } else {
            &self.indices_b[..len]
        };
        let mut sorted = true;
        for i in 1..len {
            if indices[i - 1] >= indices[i] {
                sorted = false;
                break;
            }
        }
        if !sorted {
            self.sort_active();
        }
    }

    /// Builds a permutation sorted by index, then applies it in-place using
    /// cycle sort. Marks visited positions by setting `perm[j] = j` to avoid
    /// a separate visited-flags allocation.
    #[inline]
    fn sort_active(&mut self) {
        let len = self.len;
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

        // Reuse the permutation buffer across calls
        self.sort_perm.clear();
        self.sort_perm.extend(0..len);
        self.sort_perm.sort_unstable_by_key(|&i| indices[i]);
        let perm = &mut self.sort_perm;

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

    /// Apply SZZ diagonal phase gate in-place.
    ///
    /// SZZ = diag(e^{-iπ/4}, e^{iπ/4}, e^{iπ/4}, e^{-iπ/4}).
    /// Same parity (both bits equal) → multiply by e^{-iπ/4} = (1-i)/√2.
    /// Different parity → multiply by e^{iπ/4} = (1+i)/√2.
    /// No index changes; pure O(k) phase application.
    fn apply_szz_gate(&mut self, q1: usize, q2: usize) {
        let mask1 = 1usize << q1;
        let mask2 = 1usize << q2;
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
        let c = std::f64::consts::FRAC_1_SQRT_2;
        for i in 0..len {
            let bit1 = (indices[i] & mask1) != 0;
            let bit2 = (indices[i] & mask2) != 0;
            let r = real[i];
            let im = imag[i];
            if bit1 == bit2 {
                // Same parity: e^{-iπ/4} = (1-i)/√2
                real[i] = (r + im) * c;
                imag[i] = (im - r) * c;
            } else {
                // Different parity: e^{iπ/4} = (1+i)/√2
                real[i] = (r - im) * c;
                imag[i] = (im + r) * c;
            }
        }
    }

    /// Apply `SZZdg` diagonal phase gate in-place.
    ///
    /// `SZZdg` = diag(e^{iπ/4}, e^{-iπ/4}, e^{-iπ/4}, e^{iπ/4}).
    /// Same parity → multiply by e^{iπ/4} = (1+i)/√2.
    /// Different parity → multiply by e^{-iπ/4} = (1-i)/√2.
    fn apply_szzdg_gate(&mut self, q1: usize, q2: usize) {
        let mask1 = 1usize << q1;
        let mask2 = 1usize << q2;
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
        let c = std::f64::consts::FRAC_1_SQRT_2;
        for i in 0..len {
            let bit1 = (indices[i] & mask1) != 0;
            let bit2 = (indices[i] & mask2) != 0;
            let r = real[i];
            let im = imag[i];
            if bit1 == bit2 {
                // Same parity: e^{iπ/4} = (1+i)/√2
                real[i] = (r - im) * c;
                imag[i] = (im + r) * c;
            } else {
                // Different parity: e^{-iπ/4} = (1-i)/√2
                real[i] = (r + im) * c;
                imag[i] = (im - r) * c;
            }
        }
    }

    /// Apply iSWAP gate: swap bits when different, multiply by i.
    ///
    /// |00⟩→|00⟩, |01⟩→i|10⟩, |10⟩→i|01⟩, |11⟩→|11⟩.
    /// In-place XOR + phase. Sorting is deferred via `needs_sort` flag
    /// and only performed when an operation requiring sorted order is called.
    fn apply_iswap_gate(&mut self, q1: usize, q2: usize) {
        let mask1 = 1usize << q1;
        let mask2 = 1usize << q2;
        let swap_mask = mask1 | mask2;
        let len = self.len;

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
        for i in 0..len {
            let bit1 = (indices[i] & mask1) != 0;
            let bit2 = (indices[i] & mask2) != 0;
            if bit1 != bit2 {
                indices[i] ^= swap_mask;
                // Multiply by i: (r, im) -> (-im, r)
                let r = real[i];
                real[i] = -imag[i];
                imag[i] = r;
            }
        }
        self.needs_sort = true;
    }

    /// Apply CX (CNOT) gate: if control=1, flip target bit.
    ///
    /// For small states, XOR in-place + sort. For larger states, uses O(k)
    /// partition-merge: partition into 3 sorted groups (control=0, control=1
    /// with target flipped down, control=1 with target flipped up), then
    /// 3-way merge into inactive buffer.
    #[inline]
    fn apply_cx_gate(&mut self, control: usize, target: usize) {
        self.ensure_sorted();
        let control_mask = 1usize << control;
        let target_mask = 1usize << target;
        let len = self.len;

        // Small state: XOR + sort in-place (often stays sorted, low overhead)
        if len <= 16 {
            let indices = if self.active_a {
                &mut self.indices_a[..len]
            } else {
                &mut self.indices_b[..len]
            };
            for idx in indices.iter_mut() {
                if *idx & control_mask != 0 {
                    *idx ^= target_mask;
                }
            }
            self.sort_active_if_needed();
            return;
        }

        // Larger state: O(k) partition-merge into inactive buffer.
        //
        // Partition source into 3 groups (each sorted since source is sorted):
        //   A:  control=0 (unchanged)
        //   B1: control=1, target=1 → clear target bit (index -= target_mask)
        //   B0: control=1, target=0 → set target bit   (index += target_mask)
        // Then 3-way merge produces sorted output in O(k).
        self.scratch_low.clear(); // A positions
        self.scratch_high.clear(); // B1 positions
        self.sort_perm.clear(); // B0 positions

        {
            let indices = if self.active_a {
                &self.indices_a[..len]
            } else {
                &self.indices_b[..len]
            };
            for (i, &idx) in indices.iter().enumerate() {
                if idx & control_mask == 0 {
                    #[allow(clippy::cast_possible_truncation)] // amplitude index fits in u32
                    self.scratch_low.push(i as u32);
                } else if idx & target_mask != 0 {
                    #[allow(clippy::cast_possible_truncation)] // amplitude index fits in u32
                    self.scratch_high.push(i as u32);
                } else {
                    self.sort_perm.push(i);
                }
            }
        }

        // If no control=1 indices, nothing changes
        if self.scratch_high.is_empty() && self.sort_perm.is_empty() {
            return;
        }

        let a_len = self.scratch_low.len();
        let b0_len = self.sort_perm.len();
        let b1_len = self.scratch_high.len();
        let mut a = 0usize;
        let mut b0 = 0usize;
        let mut b1 = 0usize;

        // 3-way merge into inactive buffer
        if self.active_a {
            self.indices_b.clear();
            self.real_b.clear();
            self.imag_b.clear();
            self.indices_b.reserve(len);
            self.real_b.reserve(len);
            self.imag_b.reserve(len);

            while a < a_len || b0 < b0_len || b1 < b1_len {
                let a_val = if a < a_len {
                    self.indices_a[self.scratch_low[a] as usize]
                } else {
                    usize::MAX
                };
                let b1_val = if b1 < b1_len {
                    self.indices_a[self.scratch_high[b1] as usize] ^ target_mask
                } else {
                    usize::MAX
                };
                let b0_val = if b0 < b0_len {
                    self.indices_a[self.sort_perm[b0]] ^ target_mask
                } else {
                    usize::MAX
                };

                if a_val <= b1_val && a_val <= b0_val {
                    let pos = self.scratch_low[a] as usize;
                    self.indices_b.push(a_val);
                    self.real_b.push(self.real_a[pos]);
                    self.imag_b.push(self.imag_a[pos]);
                    a += 1;
                } else if b1_val <= b0_val {
                    let pos = self.scratch_high[b1] as usize;
                    self.indices_b.push(b1_val);
                    self.real_b.push(self.real_a[pos]);
                    self.imag_b.push(self.imag_a[pos]);
                    b1 += 1;
                } else {
                    let pos = self.sort_perm[b0];
                    self.indices_b.push(b0_val);
                    self.real_b.push(self.real_a[pos]);
                    self.imag_b.push(self.imag_a[pos]);
                    b0 += 1;
                }
            }
        } else {
            self.indices_a.clear();
            self.real_a.clear();
            self.imag_a.clear();
            self.indices_a.reserve(len);
            self.real_a.reserve(len);
            self.imag_a.reserve(len);

            while a < a_len || b0 < b0_len || b1 < b1_len {
                let a_val = if a < a_len {
                    self.indices_b[self.scratch_low[a] as usize]
                } else {
                    usize::MAX
                };
                let b1_val = if b1 < b1_len {
                    self.indices_b[self.scratch_high[b1] as usize] ^ target_mask
                } else {
                    usize::MAX
                };
                let b0_val = if b0 < b0_len {
                    self.indices_b[self.sort_perm[b0]] ^ target_mask
                } else {
                    usize::MAX
                };

                if a_val <= b1_val && a_val <= b0_val {
                    let pos = self.scratch_low[a] as usize;
                    self.indices_a.push(a_val);
                    self.real_a.push(self.real_b[pos]);
                    self.imag_a.push(self.imag_b[pos]);
                    a += 1;
                } else if b1_val <= b0_val {
                    let pos = self.scratch_high[b1] as usize;
                    self.indices_a.push(b1_val);
                    self.real_a.push(self.real_b[pos]);
                    self.imag_a.push(self.imag_b[pos]);
                    b1 += 1;
                } else {
                    let pos = self.sort_perm[b0];
                    self.indices_a.push(b0_val);
                    self.real_a.push(self.real_b[pos]);
                    self.imag_a.push(self.imag_b[pos]);
                    b0 += 1;
                }
            }
        }

        self.active_a = !self.active_a;
    }

    /// Apply SWAP gate: swap bits q1 and q2.
    /// For len <= 16: XOR + sort in-place.
    /// For len > 16: O(k) partition-merge into inactive buffer.
    #[inline]
    fn apply_swap_gate(&mut self, q1: usize, q2: usize) {
        let mask1 = 1usize << q1;
        let mask2 = 1usize << q2;
        let swap_mask = mask1 | mask2;
        let len = self.len;

        // Small state: XOR + sort in-place
        if len <= 16 {
            let indices = if self.active_a {
                &mut self.indices_a[..len]
            } else {
                &mut self.indices_b[..len]
            };
            for idx in indices.iter_mut() {
                let bit1 = (*idx & mask1) != 0;
                let bit2 = (*idx & mask2) != 0;
                if bit1 != bit2 {
                    *idx ^= swap_mask;
                }
            }
            self.sort_active_if_needed();
            return;
        }

        // Larger state: O(k) partition-merge into inactive buffer.
        //
        // Partition source into 3 groups (each sorted since source is sorted):
        //   A:   bits equal (00 or 11) -> unchanged
        //   B01: bit1=0, bit2=1 -> XOR swap_mask (becomes bit1=1, bit2=0)
        //   B10: bit1=1, bit2=0 -> XOR swap_mask (becomes bit1=0, bit2=1)
        // Within each group, all entries share the same q1/q2 bit pattern, so
        // XOR with swap_mask is equivalent to a fixed add/subtract (no carry
        // interference), preserving relative order. 3-way merge -> O(k).
        self.scratch_low.clear(); // A positions
        self.scratch_high.clear(); // B01 positions
        self.sort_perm.clear(); // B10 positions

        {
            let indices = if self.active_a {
                &self.indices_a[..len]
            } else {
                &self.indices_b[..len]
            };
            for (i, &idx) in indices.iter().enumerate() {
                let bit1 = (idx & mask1) != 0;
                let bit2 = (idx & mask2) != 0;
                if bit1 == bit2 {
                    #[allow(clippy::cast_possible_truncation)] // amplitude index fits in u32
                    self.scratch_low.push(i as u32); // A: unchanged
                } else if !bit1 {
                    #[allow(clippy::cast_possible_truncation)] // amplitude index fits in u32
                    self.scratch_high.push(i as u32); // B01: bit1=0, bit2=1
                } else {
                    self.sort_perm.push(i); // B10: bit1=1, bit2=0
                }
            }
        }

        // If no entries need swapping, nothing changes
        if self.scratch_high.is_empty() && self.sort_perm.is_empty() {
            return;
        }

        let a_len = self.scratch_low.len();
        let b01_len = self.scratch_high.len();
        let b10_len = self.sort_perm.len();
        let mut a = 0usize;
        let mut b01 = 0usize;
        let mut b10 = 0usize;

        // 3-way merge into inactive buffer
        if self.active_a {
            self.indices_b.clear();
            self.real_b.clear();
            self.imag_b.clear();
            self.indices_b.reserve(len);
            self.real_b.reserve(len);
            self.imag_b.reserve(len);

            while a < a_len || b01 < b01_len || b10 < b10_len {
                let a_val = if a < a_len {
                    self.indices_a[self.scratch_low[a] as usize]
                } else {
                    usize::MAX
                };
                let b01_val = if b01 < b01_len {
                    self.indices_a[self.scratch_high[b01] as usize] ^ swap_mask
                } else {
                    usize::MAX
                };
                let b10_val = if b10 < b10_len {
                    self.indices_a[self.sort_perm[b10]] ^ swap_mask
                } else {
                    usize::MAX
                };

                if a_val <= b01_val && a_val <= b10_val {
                    let pos = self.scratch_low[a] as usize;
                    self.indices_b.push(a_val);
                    self.real_b.push(self.real_a[pos]);
                    self.imag_b.push(self.imag_a[pos]);
                    a += 1;
                } else if b01_val <= b10_val {
                    let pos = self.scratch_high[b01] as usize;
                    self.indices_b.push(b01_val);
                    self.real_b.push(self.real_a[pos]);
                    self.imag_b.push(self.imag_a[pos]);
                    b01 += 1;
                } else {
                    let pos = self.sort_perm[b10];
                    self.indices_b.push(b10_val);
                    self.real_b.push(self.real_a[pos]);
                    self.imag_b.push(self.imag_a[pos]);
                    b10 += 1;
                }
            }
        } else {
            self.indices_a.clear();
            self.real_a.clear();
            self.imag_a.clear();
            self.indices_a.reserve(len);
            self.real_a.reserve(len);
            self.imag_a.reserve(len);

            while a < a_len || b01 < b01_len || b10 < b10_len {
                let a_val = if a < a_len {
                    self.indices_b[self.scratch_low[a] as usize]
                } else {
                    usize::MAX
                };
                let b01_val = if b01 < b01_len {
                    self.indices_b[self.scratch_high[b01] as usize] ^ swap_mask
                } else {
                    usize::MAX
                };
                let b10_val = if b10 < b10_len {
                    self.indices_b[self.sort_perm[b10]] ^ swap_mask
                } else {
                    usize::MAX
                };

                if a_val <= b01_val && a_val <= b10_val {
                    let pos = self.scratch_low[a] as usize;
                    self.indices_a.push(a_val);
                    self.real_a.push(self.real_b[pos]);
                    self.imag_a.push(self.imag_b[pos]);
                    a += 1;
                } else if b01_val <= b10_val {
                    let pos = self.scratch_high[b01] as usize;
                    self.indices_a.push(b01_val);
                    self.real_a.push(self.real_b[pos]);
                    self.imag_a.push(self.imag_b[pos]);
                    b01 += 1;
                } else {
                    let pos = self.sort_perm[b10];
                    self.indices_a.push(b10_val);
                    self.real_a.push(self.real_b[pos]);
                    self.imag_a.push(self.imag_b[pos]);
                    b10 += 1;
                }
            }
        }

        self.active_a = !self.active_a;
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
// Clone implementation
// =============================================================================

impl<R: Rng + Clone> Clone for SparseStateVecSoA<R> {
    fn clone(&self) -> Self {
        Self {
            indices_a: self.indices_a.clone(),
            real_a: self.real_a.clone(),
            imag_a: self.imag_a.clone(),
            indices_b: self.indices_b.clone(),
            real_b: self.real_b.clone(),
            imag_b: self.imag_b.clone(),
            active_a: self.active_a,
            len: self.len,
            // Don't clone scratch buffers - they're lazily reused
            scratch_low: Vec::new(),
            scratch_high: Vec::new(),
            sort_perm: Vec::new(),
            merge_idx: Vec::new(),
            merge_re: Vec::new(),
            merge_im: Vec::new(),
            frames: self.frames.clone(),
            frame_phases: self.frame_phases.clone(),
            needs_sort: self.needs_sort,
            num_qubits: self.num_qubits,
            rng: self.rng.clone(),
            epsilon: self.epsilon,
        }
    }
}

// =============================================================================
// RngManageable implementation
// =============================================================================

impl<R: Rng + SeedableRng> RngManageable for SparseStateVecSoA<R> {
    type Rng = R;

    fn set_rng(&mut self, rng: R) {
        self.rng = rng;
    }

    fn rng(&self) -> &R {
        &self.rng
    }

    fn rng_mut(&mut self) -> &mut R {
        &mut self.rng
    }
}

// =============================================================================
// State expansion
// =============================================================================

impl<R: Rng> SparseStateVecSoA<R> {
    /// Returns the full state vector as a dense Vec of Complex64.
    ///
    /// Flushes all Clifford frames and returns a 2^n-element vector with the
    /// sparse amplitudes placed at their corresponding indices.
    pub fn state(&mut self) -> Vec<Complex64> {
        self.flush_all_frames();
        self.ensure_sorted();
        let mut full = vec![Complex64::new(0.0, 0.0); 1 << self.num_qubits];
        let (indices, real, imag) = self.active_buffers();
        for i in 0..self.len {
            full[indices[i]] = Complex64::new(real[i], imag[i]);
        }
        full
    }

    /// Prepare a specific computational basis state |`basis_state`>.
    ///
    /// Resets the sparse state to contain exactly one amplitude at the given index.
    pub fn prepare_computational_basis(&mut self, basis_state: usize) -> &mut Self {
        // Reset all frames to identity
        for f in &mut self.frames {
            *f = CliffordFrame::IDENTITY;
        }
        for p in &mut self.frame_phases {
            *p = 0;
        }
        self.needs_sort = false;

        // Set single amplitude in active buffer
        if self.active_a {
            self.indices_a[0] = basis_state;
            self.real_a[0] = 1.0;
            self.imag_a[0] = 0.0;
        } else {
            self.indices_b[0] = basis_state;
            self.real_b[0] = 1.0;
            self.imag_b[0] = 0.0;
        }
        self.len = 1;
        self
    }

    /// Prepare all qubits in the |+> state, creating an equal superposition of all basis states.
    ///
    /// Creates a dense state with 2^n amplitudes, each equal to 1/sqrt(2^n).
    #[allow(clippy::cast_precision_loss)] // normalization factor
    pub fn prepare_plus_state(&mut self) -> &mut Self {
        // Reset all frames to identity
        for f in &mut self.frames {
            *f = CliffordFrame::IDENTITY;
        }
        for p in &mut self.frame_phases {
            *p = 0;
        }
        self.needs_sort = false;

        let size = 1 << self.num_qubits;
        let factor = 1.0 / (size as f64).sqrt();

        // Ensure buffers are large enough
        if self.indices_a.len() < size {
            self.indices_a.resize(size, 0);
            self.real_a.resize(size, 0.0);
            self.imag_a.resize(size, 0.0);
            self.indices_b.resize(size, 0);
            self.real_b.resize(size, 0.0);
            self.imag_b.resize(size, 0.0);
        }

        let (indices, real, imag) = if self.active_a {
            (
                &mut self.indices_a[..],
                &mut self.real_a[..],
                &mut self.imag_a[..],
            )
        } else {
            (
                &mut self.indices_b[..],
                &mut self.real_b[..],
                &mut self.imag_b[..],
            )
        };
        for i in 0..size {
            indices[i] = i;
            real[i] = factor;
            imag[i] = 0.0;
        }
        self.len = size;
        self
    }

    /// Apply a general 2-qubit unitary gate given by a 4x4 complex matrix.
    ///
    /// The matrix is indexed as `matrix[output_basis][input_basis]` where
    /// basis index = (`qubit1_bit` << 1) | `qubit2_bit`.
    pub fn two_qubit_unitary(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        matrix: [[Complex64; 4]; 4],
    ) -> &mut Self {
        self.flush_frame(qubit1);
        self.flush_frame(qubit2);

        let mask1 = 1usize << qubit1;
        let mask2 = 1usize << qubit2;
        let len = self.len;

        // Build input: collect (basis_index, amplitude) pairs
        let (indices, real, imag) = self.active_buffers();
        let input: Vec<(usize, Complex64)> = (0..len)
            .map(|i| (indices[i], Complex64::new(real[i], imag[i])))
            .collect();

        // Apply the unitary: for each input amplitude, distribute across all 4 output basis states
        // Use a map for accumulation since new indices may or may not overlap
        let mut output_map: std::collections::HashMap<usize, Complex64> =
            std::collections::HashMap::new();

        for &(idx, amp) in &input {
            let bit1 = usize::from((idx & mask1) != 0);
            let bit2 = usize::from((idx & mask2) != 0);
            let input_basis = (bit1 << 1) | bit2;

            // Clear the two qubit bits from the index
            let base_idx = idx & !mask1 & !mask2;

            for (out_basis, row) in matrix.iter().enumerate() {
                let m_elem = row[input_basis];
                if m_elem.norm_sqr() < 1e-30 {
                    continue;
                }
                let out_bit1 = (out_basis >> 1) & 1;
                let out_bit2 = out_basis & 1;
                let out_idx = base_idx | (out_bit1 * mask1) | (out_bit2 * mask2);
                let contribution = m_elem * amp;
                *output_map
                    .entry(out_idx)
                    .or_insert(Complex64::new(0.0, 0.0)) += contribution;
            }
        }

        // Filter out near-zero entries and write back
        let eps_sq = self.epsilon * self.epsilon;
        let results: Vec<(usize, Complex64)> = output_map
            .into_iter()
            .filter(|(_, c)| c.norm_sqr() > eps_sq)
            .collect();

        let new_len = results.len();
        // Ensure buffers are large enough
        if self.indices_a.len() < new_len {
            self.indices_a.resize(new_len, 0);
            self.real_a.resize(new_len, 0.0);
            self.imag_a.resize(new_len, 0.0);
            self.indices_b.resize(new_len, 0);
            self.real_b.resize(new_len, 0.0);
            self.imag_b.resize(new_len, 0.0);
        }

        let (out_indices, out_real, out_imag) = if self.active_a {
            (
                &mut self.indices_a[..],
                &mut self.real_a[..],
                &mut self.imag_a[..],
            )
        } else {
            (
                &mut self.indices_b[..],
                &mut self.real_b[..],
                &mut self.imag_b[..],
            )
        };
        for (i, (idx, c)) in results.iter().enumerate() {
            out_indices[i] = *idx;
            out_real[i] = c.re;
            out_imag[i] = c.im;
        }
        self.len = new_len;
        self.needs_sort = true;
        self
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
        self.needs_sort = false;
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
            self.compose_frame(q.0, CliffordFrame::X, 0);
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::Y, 6);
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::Z, 0);
        }
        self
    }

    // -- S-like gates (delta: S=0, Sdg=0, SX=0, SXdg=7, SY=1, SYdg=7) --

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::SZ, 0);
        }
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::SZDG, 0);
        }
        self
    }

    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::SX, 0);
        }
        self
    }

    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::SXDG, 7);
        }
        self
    }

    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::SY, 1);
        }
        self
    }

    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::SYDG, 7);
        }
        self
    }

    // -- H-like gates (delta: H=0) --

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.compose_frame(q.0, CliffordFrame::H, 0);
        }
        self
    }

    // ---- Two-qubit gates ----
    //
    // When both frames are Pauli, push them through the two-qubit gate
    // symbolically (O(1) frame update) instead of flushing (O(k) gate
    // application). CX is phase-free; CZ picks up (-1)^{xc*xt}.
    // SWAP can always swap frames without flushing.

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        if pairs.len() == 1 {
            // Single pair: fast path
            let (c, t) = (pairs[0].0.0, pairs[0].1.0);
            let fc = self.frames[c];
            let ft = self.frames[t];
            if fc.is_pauli() && ft.is_pauli() {
                let (new_c, new_t, _phase) = CliffordFrame::push_through_cx(fc, ft);
                self.frames[c] = new_c;
                self.frames[t] = new_t;
            } else {
                self.flush_frame(c);
                self.flush_frame(t);
            }
            self.apply_cx_gate(c, t);
        } else {
            // Multiple pairs: process all frames, then batch physical ops.
            for &(q0, q1) in pairs {
                let (c, t) = (q0.0, q1.0);
                let fc = self.frames[c];
                let ft = self.frames[t];
                if fc.is_pauli() && ft.is_pauli() {
                    let (new_c, new_t, _phase) = CliffordFrame::push_through_cx(fc, ft);
                    self.frames[c] = new_c;
                    self.frames[t] = new_t;
                } else {
                    self.flush_frame(c);
                    self.flush_frame(t);
                }
            }
            // Batched physical operation: combined XOR mask in single pass + single sort
            let len = self.len;
            let indices = if self.active_a {
                &mut self.indices_a[..len]
            } else {
                &mut self.indices_b[..len]
            };
            for idx in indices.iter_mut() {
                let mut xor_mask = 0usize;
                for &(q0, q1) in pairs {
                    let control_mask = 1usize << q0.0;
                    let target_mask = 1usize << q1.0;
                    if *idx & control_mask != 0 {
                        xor_mask ^= target_mask;
                    }
                }
                *idx ^= xor_mask;
            }
            self.sort_active_if_needed();
        }
        self
    }

    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        if pairs.len() == 1 {
            // Single pair: fast path
            let (c, t) = (pairs[0].0.0, pairs[0].1.0);
            let fc = self.frames[c];
            let ft = self.frames[t];
            if fc.is_pauli() && ft.is_pauli() {
                let (xc, _) = fc.pauli_xz_bits();
                let (xt, _) = ft.pauli_xz_bits();
                let (new_c, new_t, _phase) = CliffordFrame::push_through_cz(fc, ft);
                self.frames[c] = new_c;
                self.frames[t] = new_t;
                if xc && xt {
                    self.frame_phases[c] = (self.frame_phases[c] + 4) % 8;
                }
            } else {
                self.flush_frame(c);
                self.flush_frame(t);
            }
            self.apply_cz_inplace(c, t);
        } else {
            // Multiple pairs: process all frames, then batch physical sign flips
            for &(q0, q1) in pairs {
                let (c, t) = (q0.0, q1.0);
                let fc = self.frames[c];
                let ft = self.frames[t];
                if fc.is_pauli() && ft.is_pauli() {
                    let (xc, _) = fc.pauli_xz_bits();
                    let (xt, _) = ft.pauli_xz_bits();
                    let (new_c, new_t, _phase) = CliffordFrame::push_through_cz(fc, ft);
                    self.frames[c] = new_c;
                    self.frames[t] = new_t;
                    if xc && xt {
                        self.frame_phases[c] = (self.frame_phases[c] + 4) % 8;
                    }
                } else {
                    self.flush_frame(c);
                    self.flush_frame(t);
                }
            }
            // Batched physical operation: single pass with parity-based sign flip
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
            for i in 0..len {
                let mut flip = false;
                for &(q0, q1) in pairs {
                    let q1_mask = 1usize << q0.0;
                    let q2_mask = 1usize << q1.0;
                    if (indices[i] & q1_mask != 0) && (indices[i] & q2_mask != 0) {
                        flip = !flip;
                    }
                }
                if flip {
                    real[i] = -real[i];
                    imag[i] = -imag[i];
                }
            }
        }
        self
    }

    fn cy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // CY = (I tensor S) . CX . (I tensor Sdg)
        // Circuit order: Sdg on target, then CX, then S on target
        for &(q0, q1) in pairs {
            let (c, t) = (q0.0, q1.0);
            // Compose Sdg on target frame, then flush both, apply CX, compose S on target
            self.compose_frame(t, CliffordFrame::SZDG, 0);
            self.flush_frame(c);
            self.flush_frame(t);
            self.apply_cx_gate(c, t);
            self.compose_frame(t, CliffordFrame::SZ, 0);
        }
        self
    }

    fn swap(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        if pairs.len() == 1 {
            // Single pair: fast path
            let (c, t) = (pairs[0].0.0, pairs[0].1.0);
            self.apply_swap_gate(c, t);
            self.frames.swap(c, t);
            self.frame_phases.swap(c, t);
        } else {
            // Multiple pairs: swap all frames, then batch physical bit-swaps
            for &(q0, q1) in pairs {
                let (c, t) = (q0.0, q1.0);
                self.frames.swap(c, t);
                self.frame_phases.swap(c, t);
            }
            // Batched physical operation: all bit swaps in single pass + single sort
            let len = self.len;
            let indices = if self.active_a {
                &mut self.indices_a[..len]
            } else {
                &mut self.indices_b[..len]
            };
            for idx in indices.iter_mut() {
                for &(q0, q1) in pairs {
                    let mask1 = 1usize << q0.0;
                    let mask2 = 1usize << q1.0;
                    let bit1 = (*idx & mask1) != 0;
                    let bit2 = (*idx & mask2) != 0;
                    if bit1 != bit2 {
                        *idx ^= mask1 | mask2;
                    }
                }
            }
            self.sort_active_if_needed();
        }
        self
    }

    // ---- Optimized two-qubit Clifford gates with frame push-through ----
    //
    // When both frames are Pauli, push them through the gate symbolically
    // (O(1) frame update + phase correction) instead of flushing (O(k)).
    // Then apply the physical gate directly.
    //
    // Note: push_through_* functions return the Heisenberg-picture phase
    // (G† P G = phase · P'), but we need the Schrödinger-picture phase
    // (G P G† = phase* · P'). For 8th-root phases, conjugation is (8-k)%8.

    fn szz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        if pairs.len() == 1 {
            let (q1, q2) = (pairs[0].0.0, pairs[0].1.0);
            let f1 = self.frames[q1];
            let f2 = self.frames[q2];
            if f1.is_pauli() && f2.is_pauli() {
                let (new_f1, new_f2, heis_phase) = CliffordFrame::push_through_szz(f1, f2);
                self.frames[q1] = new_f1;
                self.frames[q2] = new_f2;
                let schrod_phase = (8 - heis_phase) % 8;
                self.frame_phases[q1] = (self.frame_phases[q1] + schrod_phase) % 8;
            } else {
                self.flush_frame(q1);
                self.flush_frame(q2);
            }
            self.apply_szz_gate(q1, q2);
        } else {
            // Multiple pairs: process frames, then batch physical diagonal phase
            for &(qa, qb) in pairs {
                let (q1, q2) = (qa.0, qb.0);
                let f1 = self.frames[q1];
                let f2 = self.frames[q2];
                if f1.is_pauli() && f2.is_pauli() {
                    let (new_f1, new_f2, heis_phase) = CliffordFrame::push_through_szz(f1, f2);
                    self.frames[q1] = new_f1;
                    self.frames[q2] = new_f2;
                    let schrod_phase = (8 - heis_phase) % 8;
                    self.frame_phases[q1] = (self.frame_phases[q1] + schrod_phase) % 8;
                } else {
                    self.flush_frame(q1);
                    self.flush_frame(q2);
                }
            }
            // Batched physical SZZ: single pass, combined parity phase
            let n_pairs = pairs.len();
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
            for i in 0..len {
                let mut n_diff: u32 = 0;
                for &(q0, q1) in pairs {
                    let mask1 = 1usize << q0.0;
                    let mask2 = 1usize << q1.0;
                    let bit1 = (indices[i] & mask1) != 0;
                    let bit2 = (indices[i] & mask2) != 0;
                    if bit1 != bit2 {
                        n_diff += 1;
                    }
                }
                // Phase index = (2*n_diff - n_pairs) mod 8
                // n_diff different-parity pairs contribute e^{iπ/4} each,
                // (n_pairs - n_diff) same-parity pairs contribute e^{-iπ/4} each.
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                // n_diff and n_pairs are small counts
                let k = ((2 * n_diff as i32 - n_pairs as i32).rem_euclid(8)) as usize;
                if k != 0 {
                    let [cos_k, sin_k] = PHASE_ROOTS[k];
                    let r = real[i];
                    let im = imag[i];
                    real[i] = r * cos_k - im * sin_k;
                    imag[i] = r * sin_k + im * cos_k;
                }
            }
        }
        self
    }

    fn szzdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        if pairs.len() == 1 {
            let (q1, q2) = (pairs[0].0.0, pairs[0].1.0);
            let f1 = self.frames[q1];
            let f2 = self.frames[q2];
            if f1.is_pauli() && f2.is_pauli() {
                let (new_f1, new_f2, heis_phase) = CliffordFrame::push_through_szz(f1, f2);
                self.frames[q1] = new_f1;
                self.frames[q2] = new_f2;
                // For SZZdg, Schrödinger phase = Heisenberg phase of SZZ (no conjugation),
                // because SZZdg Schrödinger = SZZ† · P · SZZ = SZZ Heisenberg.
                self.frame_phases[q1] = (self.frame_phases[q1] + heis_phase) % 8;
            } else {
                self.flush_frame(q1);
                self.flush_frame(q2);
            }
            self.apply_szzdg_gate(q1, q2);
        } else {
            // Multiple pairs: process frames, then batch physical diagonal phase
            for &(qa, qb) in pairs {
                let (q1, q2) = (qa.0, qb.0);
                let f1 = self.frames[q1];
                let f2 = self.frames[q2];
                if f1.is_pauli() && f2.is_pauli() {
                    let (new_f1, new_f2, heis_phase) = CliffordFrame::push_through_szz(f1, f2);
                    self.frames[q1] = new_f1;
                    self.frames[q2] = new_f2;
                    self.frame_phases[q1] = (self.frame_phases[q1] + heis_phase) % 8;
                } else {
                    self.flush_frame(q1);
                    self.flush_frame(q2);
                }
            }
            // Batched physical SZZdg: single pass, conjugated parity phase
            let n_pairs = pairs.len();
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
            for i in 0..len {
                let mut n_diff: u32 = 0;
                for &(q0, q1) in pairs {
                    let mask1 = 1usize << q0.0;
                    let mask2 = 1usize << q1.0;
                    let bit1 = (indices[i] & mask1) != 0;
                    let bit2 = (indices[i] & mask2) != 0;
                    if bit1 != bit2 {
                        n_diff += 1;
                    }
                }
                // SZZdg: conjugated phase = (8 - szz_phase) % 8
                // SZZ phase = (2*n_diff - n_pairs) mod 8
                // SZZdg phase = (n_pairs - 2*n_diff) mod 8
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                // n_pairs and n_diff are small counts
                let k = ((n_pairs as i32 - 2 * n_diff as i32).rem_euclid(8)) as usize;
                if k != 0 {
                    let [cos_k, sin_k] = PHASE_ROOTS[k];
                    let r = real[i];
                    let im = imag[i];
                    real[i] = r * cos_k - im * sin_k;
                    imag[i] = r * sin_k + im * cos_k;
                }
            }
        }
        self
    }

    fn iswap(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        if pairs.len() == 1 {
            let (q1, q2) = (pairs[0].0.0, pairs[0].1.0);
            let f1 = self.frames[q1];
            let f2 = self.frames[q2];
            if f1.is_pauli() && f2.is_pauli() {
                let (new_f1, new_f2, heis_phase) = CliffordFrame::push_through_iswap(f1, f2);
                self.frames[q1] = new_f1;
                self.frames[q2] = new_f2;
                let schrod_phase = (8 - heis_phase) % 8;
                self.frame_phases[q1] = (self.frame_phases[q1] + schrod_phase) % 8;
            } else {
                self.flush_frame(q1);
                self.flush_frame(q2);
            }
            self.apply_iswap_gate(q1, q2);
        } else {
            // Multiple pairs: process frames, then batch physical iSWAP
            for &(qa, qb) in pairs {
                let (q1, q2) = (qa.0, qb.0);
                let f1 = self.frames[q1];
                let f2 = self.frames[q2];
                if f1.is_pauli() && f2.is_pauli() {
                    let (new_f1, new_f2, heis_phase) = CliffordFrame::push_through_iswap(f1, f2);
                    self.frames[q1] = new_f1;
                    self.frames[q2] = new_f2;
                    let schrod_phase = (8 - heis_phase) % 8;
                    self.frame_phases[q1] = (self.frame_phases[q1] + schrod_phase) % 8;
                } else {
                    self.flush_frame(q1);
                    self.flush_frame(q2);
                }
            }
            // Batched physical iSWAP: combined bit-swaps + i^count phase
            let len = self.len;
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
            for i in 0..len {
                let mut swap_count: u32 = 0;
                for &(q0, q1) in pairs {
                    let mask1 = 1usize << q0.0;
                    let mask2 = 1usize << q1.0;
                    let bit1 = (indices[i] & mask1) != 0;
                    let bit2 = (indices[i] & mask2) != 0;
                    if bit1 != bit2 {
                        indices[i] ^= mask1 | mask2;
                        swap_count += 1;
                    }
                }
                // Multiply by i^swap_count
                match swap_count % 4 {
                    0 => {}
                    1 => {
                        let r = real[i];
                        real[i] = -imag[i];
                        imag[i] = r;
                    }
                    2 => {
                        real[i] = -real[i];
                        imag[i] = -imag[i];
                    }
                    3 => {
                        let r = real[i];
                        real[i] = imag[i];
                        imag[i] = -r;
                    }
                    _ => unreachable!(),
                }
            }
            self.sort_active_if_needed();
        }
        self
    }

    // SXX, SYY, G: trait defaults are already optimal with the frame system.
    // SXX/SYY decompose into single-qubit frame compositions + CX (one flush),
    // and G chains CZ push-through with H compositions. Overriding would not
    // improve performance since the decomposition requires non-Pauli flushes
    // regardless of outer frame push-through.

    // ---- Measurement: check Z-image to avoid unnecessary flush ----

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let mut results = Vec::with_capacity(qubits.len());

        for &q in qubits {
            let z_img = self.frames[q.0].z_image();

            if z_img.axis == PauliAxis::Z {
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
            } else {
                // Z maps to +/-X or +/-Y: must flush frame first
                self.flush_frame(q.0);
                results.push(self.physical_mz(q.0));
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

        let is_deterministic = !(1e-10..=1.0 - 1e-10).contains(&prob_one);
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
// ArbitraryRotationGateable trait implementation
// =============================================================================

impl<R: Rng + Debug> ArbitraryRotationGateable for SparseStateVecSoA<R> {
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        // RX(theta) = exp(-i*theta*X/2) = [[cos, -i*sin], [-i*sin, cos]]
        //
        // Push-through: Z and Y anticommute with X, so negate theta when has_z.
        for &q in qubits {
            let effective_theta = if self.frames[q.0].is_pauli() {
                let (_, has_z) = self.frames[q.0].pauli_xz_bits();
                if has_z { -theta } else { theta }
            } else {
                self.flush_frame(q.0);
                theta
            };
            let half = effective_theta / 2.0;
            let cos = half.cos();
            let sin = half.sin();
            self.apply_single_qubit_gate(
                q.0, cos, 0.0, // a = cos
                0.0, -sin, // b = -i*sin
                0.0, -sin, // c = -i*sin
                cos, 0.0, // d = cos
            );
        }
        self
    }

    fn ry(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        // RY(theta) = exp(-i*theta*Y/2) = [[cos, -sin], [sin, cos]]
        //
        // Push-through: X anticommutes with Y, Z anticommutes with Y, but Y commutes.
        // Negate when has_x XOR has_z (i.e., frame is X or Z but not I or Y).
        for &q in qubits {
            let effective_theta = if self.frames[q.0].is_pauli() {
                let (has_x, has_z) = self.frames[q.0].pauli_xz_bits();
                if has_x == has_z { theta } else { -theta }
            } else {
                self.flush_frame(q.0);
                theta
            };
            let half = effective_theta / 2.0;
            let cos = half.cos();
            let sin = half.sin();
            self.apply_single_qubit_gate(
                q.0, cos, 0.0, // a = cos
                -sin, 0.0, // b = -sin
                sin, 0.0, // c = sin
                cos, 0.0, // d = cos
            );
        }
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        // RZ(theta) = diag(e^{-i*theta/2}, e^{i*theta/2})
        //
        // Push-through: X and Y anticommute with Z, so negate theta when has_x.
        for &q in qubits {
            let effective_theta = if self.frames[q.0].is_pauli() {
                let (has_x, _) = self.frames[q.0].pauli_xz_bits();
                if has_x { -theta } else { theta }
            } else {
                self.flush_frame(q.0);
                theta
            };

            let half = effective_theta / 2.0;
            let cos = half.cos();
            let sin = half.sin();
            self.apply_rz_kernel(q.0, cos, sin);
        }
        self
    }

    fn t(&mut self, qubits: &[QubitId]) -> &mut Self {
        // T = RZ(pi/4). cos(pi/8) and sin(pi/8) are compile-time constants.
        const COS_PI_8: f64 = 0.923_879_532_511_286_7;
        const SIN_PI_8: f64 = 0.382_683_432_365_089_8;
        for &q in qubits {
            let (cos, sin) = if self.frames[q.0].is_pauli() {
                let (has_x, _) = self.frames[q.0].pauli_xz_bits();
                if has_x {
                    (COS_PI_8, -SIN_PI_8)
                } else {
                    (COS_PI_8, SIN_PI_8)
                }
            } else {
                self.flush_frame(q.0);
                (COS_PI_8, SIN_PI_8)
            };
            self.apply_rz_kernel(q.0, cos, sin);
        }
        self
    }

    fn tdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        // Tdg = RZ(-pi/4). cos(-pi/8) = cos(pi/8), sin(-pi/8) = -sin(pi/8).
        const COS_PI_8: f64 = 0.923_879_532_511_286_7;
        const SIN_PI_8: f64 = 0.382_683_432_365_089_8;
        for &q in qubits {
            let (cos, sin) = if self.frames[q.0].is_pauli() {
                let (has_x, _) = self.frames[q.0].pauli_xz_bits();
                if has_x {
                    (COS_PI_8, SIN_PI_8)
                } else {
                    (COS_PI_8, -SIN_PI_8)
                }
            } else {
                self.flush_frame(q.0);
                (COS_PI_8, -SIN_PI_8)
            };
            self.apply_rz_kernel(q.0, cos, sin);
        }
        self
    }

    fn rxx(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta_rad = theta.to_radians_signed();
        // RXX(theta) = exp(-i*theta*XX/2)
        // Pairs (00<->11) and (01<->10) with matrix [[cos, -i*sin], [-i*sin, cos]].
        //
        // Push-through: Z and Y anticommute with X, so each qubit with Z component
        // in its Pauli frame contributes a flip (mirror of RZZ).
        //
        // Hybrid: use direct gate for Pauli frames (push-through saves flush),
        // decomposition H*H*RZZ*H*H for non-Pauli (frame cancellation is O(k)).
        for &(qa, qb) in pairs {
            let q1 = qa.0;
            let q2 = qb.0;

            if self.frames[q1].is_pauli() && self.frames[q2].is_pauli() {
                let mut flips = 0u32;
                let (_, has_z) = self.frames[q1].pauli_xz_bits();
                flips += u32::from(has_z);
                let (_, has_z) = self.frames[q2].pauli_xz_bits();
                flips += u32::from(has_z);

                let effective_theta = if flips & 1 == 1 {
                    -theta_rad
                } else {
                    theta_rad
                };
                let half = effective_theta / 2.0;
                let cos = half.cos();
                let sin = half.sin();

                // RXX: off-diagonal is -i*sin for both parity groups -> sin_sign = -1.0
                self.apply_parity_flip_gate(q1, q2, cos, sin, -1.0, -1.0);
            } else {
                // Non-Pauli frame: decompose as H*H*RZZ*H*H.
                // H updates the frame cheaply, and RZZ benefits from push-through.
                self.h(&[qa])
                    .h(&[qb])
                    .rzz(theta, &[(qa, qb)])
                    .h(&[qa])
                    .h(&[qb]);
            }
        }
        self
    }

    fn ryy(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta_rad = theta.to_radians_signed();
        // RYY(theta) = exp(-i*theta*YY/2)
        // Same parity (00<->11): matrix [[cos, +i*sin], [+i*sin, cos]]
        // Diff parity (01<->10): matrix [[cos, -i*sin], [-i*sin, cos]]
        //
        // Push-through: X and Z individually anticommute with Y, but Y commutes.
        // Each qubit with has_x != has_z (X or Z, not I or Y) contributes a flip.
        //
        // Hybrid: use direct gate for Pauli frames (push-through saves flush),
        // decomposition SX*SX*RZZ*SXdg*SXdg for non-Pauli (frame cancellation is O(k)).
        for &(qa, qb) in pairs {
            let q1 = qa.0;
            let q2 = qb.0;

            if self.frames[q1].is_pauli() && self.frames[q2].is_pauli() {
                let mut flips = 0u32;
                let (has_x, has_z) = self.frames[q1].pauli_xz_bits();
                flips += u32::from(has_x != has_z);
                let (has_x, has_z) = self.frames[q2].pauli_xz_bits();
                flips += u32::from(has_x != has_z);

                let effective_theta = if flips & 1 == 1 {
                    -theta_rad
                } else {
                    theta_rad
                };
                let half = effective_theta / 2.0;
                let cos = half.cos();
                let sin = half.sin();

                // RYY: same-parity off-diagonal is +i*sin (sign=+1),
                //      diff-parity off-diagonal is -i*sin (sign=-1)
                self.apply_parity_flip_gate(q1, q2, cos, sin, 1.0, -1.0);
            } else {
                // Non-Pauli frame: decompose as SX*SX*RZZ*SXdg*SXdg.
                // SX/SXdg update frames cheaply, and RZZ benefits from push-through.
                self.sx(&[qa])
                    .sx(&[qb])
                    .rzz(theta, &[(qa, qb)])
                    .sxdg(&[qa])
                    .sxdg(&[qb]);
            }
        }
        self
    }

    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed();
        // RZZ(theta) = diag(e^{-i*theta/2}, e^{i*theta/2}, e^{i*theta/2}, e^{-i*theta/2})
        //
        // Push-through: X and Y anticommute with Z, so each qubit with X component
        // in its Pauli frame contributes a flip.
        for &(qa, qb) in pairs {
            let q1 = qa.0;
            let q2 = qb.0;

            let mut flips = 0u32;
            if self.frames[q1].is_pauli() {
                let (has_x, _) = self.frames[q1].pauli_xz_bits();
                flips += u32::from(has_x);
            } else {
                self.flush_frame(q1);
            }
            if self.frames[q2].is_pauli() {
                let (has_x, _) = self.frames[q2].pauli_xz_bits();
                flips += u32::from(has_x);
            } else {
                self.flush_frame(q2);
            }

            let effective_theta = if flips & 1 == 1 { -theta } else { theta };
            let half = effective_theta / 2.0;
            let cos = half.cos();
            let sin = half.sin();

            self.apply_rzz_kernel(q1, q2, cos, sin);
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

        // U gate matrix elements (matching StateVecSoA's direct implementation):
        // U = [[cos(theta/2),            -e^{i*lambda}*sin(theta/2)],
        //      [e^{i*phi}*sin(theta/2),   e^{i*(phi+lambda)}*cos(theta/2)]]
        let u00_re = cos;
        let u00_im = 0.0;
        let u01_re = -sin * lambda.cos();
        let u01_im = -sin * lambda.sin();
        let u10_re = sin * phi.cos();
        let u10_im = sin * phi.sin();
        let u11_re = cos * (phi + lambda).cos();
        let u11_im = cos * (phi + lambda).sin();

        for &q in qubits {
            self.flush_frame(q.0);
            self.apply_single_qubit_gate(
                q.0, u00_re, u00_im, u01_re, u01_im, u10_re, u10_im, u11_re, u11_im,
            );
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
        sim.cx(&[(QubitId(0), QubitId(1))]);

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
        sim.cz(&[(QubitId(0), QubitId(1))]);

        // |11⟩ should have negative amplitude
        assert!(sim.get_amplitude(0b11).re < 0.0);
    }

    #[test]
    fn test_cx_gate() {
        let mut sim = SparseStateVecSoA::new(2);
        sim.x(&[QubitId(0)]); // |01⟩
        sim.cx(&[(QubitId(0), QubitId(1))]);

        // Should be |11⟩
        assert_eq!(sim.num_amplitudes(), 1);
        assert!((sim.get_amplitude(0b11).re - 1.0).abs() < 1e-10);
    }
}
