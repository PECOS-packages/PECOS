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
use pecos_rng::{PecosRng, Rng, SeedableRng};
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

    /// Apply single-qubit gate using binary search.
    ///
    /// Uses direct field access to avoid borrow conflicts and eliminate temp allocations.
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

        // Get pointers to source (active) and dest (inactive) buffers
        let (src_indices, src_real, src_imag, dst_indices, dst_real, dst_imag) = if self.active_a {
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

        // Clear and prepare destination
        dst_indices.clear();
        dst_real.clear();
        dst_imag.clear();
        dst_indices.reserve(len * 2);
        dst_real.reserve(len * 2);
        dst_imag.reserve(len * 2);

        // First pass: process all "low" indices (bit q=0)
        for i in 0..len {
            let idx = src_indices[i];
            if idx & mask != 0 {
                continue;
            }

            let (amp_re, amp_im) = (src_real[i], src_imag[i]);
            let paired_idx = idx | mask;

            // Binary search for pair
            let (paired_re, paired_im) = src_indices[i + 1..]
                .binary_search(&paired_idx)
                .ok()
                .map(|offset| (src_real[i + 1 + offset], src_imag[i + 1 + offset]))
                .unwrap_or((0.0, 0.0));

            // Apply gate
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

        // Sort and update state
        self.len = dst_indices.len();
        self.active_a = !self.active_a;
        self.sort_active();
    }

    /// Sort the active buffer by index
    #[inline]
    fn sort_active(&mut self) {
        let len = self.len;
        let (indices, real, imag) = if self.active_a {
            (&mut self.indices_a[..len], &mut self.real_a[..len], &mut self.imag_a[..len])
        } else {
            (&mut self.indices_b[..len], &mut self.real_b[..len], &mut self.imag_b[..len])
        };

        // Create permutation
        let mut perm: Vec<usize> = (0..len).collect();
        perm.sort_unstable_by_key(|&i| indices[i]);

        // Apply permutation using cycle sort
        let mut visited = vec![false; len];
        for i in 0..len {
            if visited[i] || perm[i] == i {
                continue;
            }

            let mut j = i;
            let tmp_idx = indices[i];
            let tmp_re = real[i];
            let tmp_im = imag[i];

            loop {
                visited[j] = true;
                let k = perm[j];
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

    /// Apply CX (CNOT) gate.
    ///
    /// CX is a permutation: if control=1, flip target bit.
    #[inline]
    fn apply_cx_gate(&mut self, control: usize, target: usize) {
        let control_mask = 1usize << control;
        let target_mask = 1usize << target;
        let len = self.len;

        // Get source and dest buffers
        let (src_indices, src_real, src_imag, dst_indices, dst_real, dst_imag) = if self.active_a {
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
        dst_indices.reserve(len);
        dst_real.reserve(len);
        dst_imag.reserve(len);

        for i in 0..len {
            let idx = src_indices[i];
            let new_idx = if idx & control_mask != 0 {
                idx ^ target_mask
            } else {
                idx
            };
            dst_indices.push(new_idx);
            dst_real.push(src_real[i]);
            dst_imag.push(src_imag[i]);
        }

        self.active_a = !self.active_a;
        self.sort_active();
    }

    /// Apply SWAP gate.
    #[inline]
    fn apply_swap_gate(&mut self, q1: usize, q2: usize) {
        let mask1 = 1usize << q1;
        let mask2 = 1usize << q2;
        let len = self.len;

        let (src_indices, src_real, src_imag, dst_indices, dst_real, dst_imag) = if self.active_a {
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
        dst_indices.reserve(len);
        dst_real.reserve(len);
        dst_imag.reserve(len);

        for i in 0..len {
            let idx = src_indices[i];
            let bit1 = (idx & mask1) != 0;
            let bit2 = (idx & mask2) != 0;
            let new_idx = if bit1 != bit2 {
                idx ^ (mask1 | mask2)
            } else {
                idx
            };
            dst_indices.push(new_idx);
            dst_real.push(src_real[i]);
            dst_imag.push(src_imag[i]);
        }

        self.active_a = !self.active_a;
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
                inv_sqrt2, 0.0,   // a
                inv_sqrt2, 0.0,   // b
                inv_sqrt2, 0.0,   // c
                -inv_sqrt2, 0.0,  // d
            );
        }
        self
    }

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_single_qubit_gate(
                q.0,
                0.0, 0.0,  // a
                1.0, 0.0,  // b
                1.0, 0.0,  // c
                0.0, 0.0,  // d
            );
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
                0.5, 0.5,    // b = (1+i)/2
                -0.5, -0.5,  // c = -(1+i)/2
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
            let outcome = self.rng.random::<f64>() < prob_one;

            results.push(MeasurementResult {
                outcome,
                is_deterministic,
            });

            // Collapse: keep only consistent amplitudes
            let keep_value = if outcome { mask } else { 0 };

            // Get source and dest buffers
            let new_len = {
                let (src_indices, src_real, src_imag, dst_indices, dst_real, dst_imag) = if self.active_a {
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

                for i in 0..len {
                    if src_indices[i] & mask == keep_value {
                        dst_indices.push(src_indices[i]);
                        dst_real.push(src_real[i]);
                        dst_imag.push(src_imag[i]);
                    }
                }

                dst_indices.len()
            };

            self.len = new_len;
            self.active_a = !self.active_a;
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
