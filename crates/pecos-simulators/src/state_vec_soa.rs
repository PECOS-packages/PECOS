// Copyright 2025 The PECOS Developers
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

//! Optimized State Vector Simulator combining multiple optimization strategies:
//!
//! 1. **`SoA` Layout**: Separate real and imaginary arrays for SIMD-friendly math
//! 2. **Strided Iteration**: Cache-efficient access patterns for two-qubit gates
//!
//! This simulator prioritizes simple, clean code that the compiler can optimize well.

use crate::clifford_gateable::MeasurementResult;
use crate::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};
use num_complex::Complex64;
use pecos_core::{Angle64, QubitId, RngManageable};
use pecos_random::{PecosRng, Rng, RngProbabilityExt, SeedableRng};
use std::fmt::Debug;
use wide::f64x4;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Wrapper for raw pointer to allow Send+Sync for parallel iteration.
/// SAFETY: This is only safe when the parallel access pattern guarantees
/// non-overlapping memory regions for each thread.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct SendPtr(*mut f64);

#[cfg(feature = "parallel")]
impl SendPtr {
    #[inline]
    fn ptr(self) -> *mut f64 {
        self.0
    }
}

#[cfg(feature = "parallel")]
unsafe impl Send for SendPtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for SendPtr {}

// =============================================================================
// Gate Fusion Support
// =============================================================================

/// 2x2 complex matrix for gate fusion.
/// Stored as [[a, b], [c, d]] where each element is (real, imag).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Complex2x2 {
    a_re: f64,
    a_im: f64,
    b_re: f64,
    b_im: f64,
    c_re: f64,
    c_im: f64,
    d_re: f64,
    d_im: f64,
}

impl Complex2x2 {
    /// Check if this is the identity matrix (no-op)
    #[inline]
    fn is_identity(&self) -> bool {
        const EPS: f64 = 1e-15;
        (self.a_re - 1.0).abs() < EPS
            && self.a_im.abs() < EPS
            && self.b_re.abs() < EPS
            && self.b_im.abs() < EPS
            && self.c_re.abs() < EPS
            && self.c_im.abs() < EPS
            && (self.d_re - 1.0).abs() < EPS
            && self.d_im.abs() < EPS
    }

    /// Matrix multiplication: self * other
    /// Computes the product of two 2x2 complex matrices.
    #[inline]
    fn mul(&self, other: &Self) -> Self {
        // (a1*a2 + b1*c2, a1*b2 + b1*d2)
        // (c1*a2 + d1*c2, c1*b2 + d1*d2)
        Self {
            a_re: self.a_re * other.a_re - self.a_im * other.a_im + self.b_re * other.c_re
                - self.b_im * other.c_im,
            a_im: self.a_re * other.a_im
                + self.a_im * other.a_re
                + self.b_re * other.c_im
                + self.b_im * other.c_re,
            b_re: self.a_re * other.b_re - self.a_im * other.b_im + self.b_re * other.d_re
                - self.b_im * other.d_im,
            b_im: self.a_re * other.b_im
                + self.a_im * other.b_re
                + self.b_re * other.d_im
                + self.b_im * other.d_re,
            c_re: self.c_re * other.a_re - self.c_im * other.a_im + self.d_re * other.c_re
                - self.d_im * other.c_im,
            c_im: self.c_re * other.a_im
                + self.c_im * other.a_re
                + self.d_re * other.c_im
                + self.d_im * other.c_re,
            d_re: self.c_re * other.b_re - self.c_im * other.b_im + self.d_re * other.d_re
                - self.d_im * other.d_im,
            d_im: self.c_re * other.b_im
                + self.c_im * other.b_re
                + self.d_re * other.d_im
                + self.d_im * other.d_re,
        }
    }
}

/// Pre-defined gate matrices for fusion
mod gate_matrices {
    use super::Complex2x2;

    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

    /// Hadamard gate: (1/sqrt(2)) * [[1, 1], [1, -1]]
    pub const H: Complex2x2 = Complex2x2 {
        a_re: INV_SQRT2,
        a_im: 0.0,
        b_re: INV_SQRT2,
        b_im: 0.0,
        c_re: INV_SQRT2,
        c_im: 0.0,
        d_re: -INV_SQRT2,
        d_im: 0.0,
    };

    /// X gate: [[0, 1], [1, 0]]
    pub const X: Complex2x2 = Complex2x2 {
        a_re: 0.0,
        a_im: 0.0,
        b_re: 1.0,
        b_im: 0.0,
        c_re: 1.0,
        c_im: 0.0,
        d_re: 0.0,
        d_im: 0.0,
    };

    /// Y gate: [[0, -i], [i, 0]]
    pub const Y: Complex2x2 = Complex2x2 {
        a_re: 0.0,
        a_im: 0.0,
        b_re: 0.0,
        b_im: -1.0,
        c_re: 0.0,
        c_im: 1.0,
        d_re: 0.0,
        d_im: 0.0,
    };

    /// Z gate: [[1, 0], [0, -1]]
    pub const Z: Complex2x2 = Complex2x2 {
        a_re: 1.0,
        a_im: 0.0,
        b_re: 0.0,
        b_im: 0.0,
        c_re: 0.0,
        c_im: 0.0,
        d_re: -1.0,
        d_im: 0.0,
    };

    /// S gate (SZ): [[1, 0], [0, i]]
    pub const SZ: Complex2x2 = Complex2x2 {
        a_re: 1.0,
        a_im: 0.0,
        b_re: 0.0,
        b_im: 0.0,
        c_re: 0.0,
        c_im: 0.0,
        d_re: 0.0,
        d_im: 1.0,
    };

    /// S-dagger gate (SZDG): [[1, 0], [0, -i]]
    pub const SZDG: Complex2x2 = Complex2x2 {
        a_re: 1.0,
        a_im: 0.0,
        b_re: 0.0,
        b_im: 0.0,
        c_re: 0.0,
        c_im: 0.0,
        d_re: 0.0,
        d_im: -1.0,
    };

    /// SX gate: (1/2)[[1+i, 1-i], [1-i, 1+i]]
    pub const SX: Complex2x2 = Complex2x2 {
        a_re: 0.5,
        a_im: 0.5,
        b_re: 0.5,
        b_im: -0.5,
        c_re: 0.5,
        c_im: -0.5,
        d_re: 0.5,
        d_im: 0.5,
    };

    /// SXDG gate: (1/2)[[1-i, 1+i], [1+i, 1-i]]
    pub const SXDG: Complex2x2 = Complex2x2 {
        a_re: 0.5,
        a_im: -0.5,
        b_re: 0.5,
        b_im: 0.5,
        c_re: 0.5,
        c_im: 0.5,
        d_re: 0.5,
        d_im: -0.5,
    };

    /// SY gate: (1/2)[[1+i, -1-i], [1+i, 1+i]]
    pub const SY: Complex2x2 = Complex2x2 {
        a_re: 0.5,
        a_im: 0.5,
        b_re: -0.5,
        b_im: -0.5,
        c_re: 0.5,
        c_im: 0.5,
        d_re: 0.5,
        d_im: 0.5,
    };

    /// SYDG gate: (1/2)[[1-i, 1-i], [-1+i, 1-i]]
    pub const SYDG: Complex2x2 = Complex2x2 {
        a_re: 0.5,
        a_im: -0.5,
        b_re: 0.5,
        b_im: -0.5,
        c_re: -0.5,
        c_im: 0.5,
        d_re: 0.5,
        d_im: -0.5,
    };

    /// F gate = SZ * SX: (1/2)[[1+i, 1-i], [1+i, -1+i]]
    pub const F: Complex2x2 = Complex2x2 {
        a_re: 0.5,
        a_im: 0.5,
        b_re: 0.5,
        b_im: -0.5,
        c_re: 0.5,
        c_im: 0.5,
        d_re: -0.5,
        d_im: 0.5,
    };

    /// FDG gate = SXDG * SZDG: (1/2)[[1-i, 1-i], [1+i, -1-i]]
    pub const FDG: Complex2x2 = Complex2x2 {
        a_re: 0.5,
        a_im: -0.5,
        b_re: 0.5,
        b_im: -0.5,
        c_re: 0.5,
        c_im: 0.5,
        d_re: -0.5,
        d_im: -0.5,
    };

    /// H2 gate: Z * SY = (1/2)[[1+i, -(1+i)], [-(1+i), -(1+i)]]
    pub const H2: Complex2x2 = Complex2x2 {
        a_re: 0.5,
        a_im: 0.5,
        b_re: -0.5,
        b_im: -0.5,
        c_re: -0.5,
        c_im: -0.5,
        d_re: -0.5,
        d_im: -0.5,
    };

    /// H3 gate: Y * SZ = [[0, 1], [i, 0]]
    pub const H3: Complex2x2 = Complex2x2 {
        a_re: 0.0,
        a_im: 0.0,
        b_re: 1.0,
        b_im: 0.0,
        c_re: 0.0,
        c_im: 1.0,
        d_re: 0.0,
        d_im: 0.0,
    };

    /// H4 gate: X * SZ = [[0, i], [1, 0]]
    pub const H4: Complex2x2 = Complex2x2 {
        a_re: 0.0,
        a_im: 0.0,
        b_re: 0.0,
        b_im: 1.0,
        c_re: 1.0,
        c_im: 0.0,
        d_re: 0.0,
        d_im: 0.0,
    };

    /// H5 gate: Z * SX = (1/2)[[1+i, 1-i], [-(1-i), -(1+i)]]
    pub const H5: Complex2x2 = Complex2x2 {
        a_re: 0.5,
        a_im: 0.5,
        b_re: 0.5,
        b_im: -0.5,
        c_re: -0.5,
        c_im: 0.5,
        d_re: -0.5,
        d_im: -0.5,
    };

    /// H6 gate: Y * SX = (1/2)[[-1-i, 1-i], [-1+i, 1+i]]
    pub const H6: Complex2x2 = Complex2x2 {
        a_re: -0.5,
        a_im: -0.5,
        b_re: 0.5,
        b_im: -0.5,
        c_re: -0.5,
        c_im: 0.5,
        d_re: 0.5,
        d_im: 0.5,
    };

    /// F2 gate: SY * SXDG = (1/2)[[1-i, -1+i], [1+i, 1+i]]
    pub const F2: Complex2x2 = Complex2x2 {
        a_re: 0.5,
        a_im: -0.5,
        b_re: -0.5,
        b_im: 0.5,
        c_re: 0.5,
        c_im: 0.5,
        d_re: 0.5,
        d_im: 0.5,
    };

    /// F2DG gate: SX * SYDG = (1/2)[[1+i, 1-i], [-1-i, 1-i]]
    pub const F2DG: Complex2x2 = Complex2x2 {
        a_re: 0.5,
        a_im: 0.5,
        b_re: 0.5,
        b_im: -0.5,
        c_re: -0.5,
        c_im: -0.5,
        d_re: 0.5,
        d_im: -0.5,
    };

    /// F3 gate: SZ * SXDG = (1/2)[[1-i, 1+i], [-1+i, 1+i]]
    pub const F3: Complex2x2 = Complex2x2 {
        a_re: 0.5,
        a_im: -0.5,
        b_re: 0.5,
        b_im: 0.5,
        c_re: -0.5,
        c_im: 0.5,
        d_re: 0.5,
        d_im: 0.5,
    };

    /// F3DG gate: SX * SZDG = (1/2)[[1+i, -1-i], [1-i, 1-i]]
    pub const F3DG: Complex2x2 = Complex2x2 {
        a_re: 0.5,
        a_im: 0.5,
        b_re: -0.5,
        b_im: -0.5,
        c_re: 0.5,
        c_im: -0.5,
        d_re: 0.5,
        d_im: -0.5,
    };

    /// F4 gate: SX * SZ = (1/2)[[1+i, 1+i], [1-i, -1+i]]
    pub const F4: Complex2x2 = Complex2x2 {
        a_re: 0.5,
        a_im: 0.5,
        b_re: 0.5,
        b_im: 0.5,
        c_re: 0.5,
        c_im: -0.5,
        d_re: -0.5,
        d_im: 0.5,
    };

    /// F4DG gate: SZDG * SXDG = (1/2)[[1-i, 1+i], [1-i, -1-i]]
    pub const F4DG: Complex2x2 = Complex2x2 {
        a_re: 0.5,
        a_im: -0.5,
        b_re: 0.5,
        b_im: 0.5,
        c_re: 0.5,
        c_im: -0.5,
        d_re: -0.5,
        d_im: -0.5,
    };
}

/// Optimized state vector simulator with `SoA` layout.
#[derive(Debug)]
pub struct StateVecSoA<R = PecosRng>
where
    R: Rng,
{
    /// Real components of the state vector
    pub(crate) real: Vec<f64>,
    /// Imaginary components of the state vector
    pub(crate) imag: Vec<f64>,
    /// Number of qubits
    num_qubits: usize,
    /// Random number generator for measurements
    rng: R,
    /// Scratch buffer for real components (lazily allocated, used by `two_qubit_unitary`)
    scratch_real: Vec<f64>,
    /// Scratch buffer for imaginary components (lazily allocated, used by `two_qubit_unitary`)
    scratch_imag: Vec<f64>,
    /// Gate fusion: accumulated matrix per qubit (None = identity/no pending gates)
    pending_gates: Vec<Option<Complex2x2>>,
    /// Whether gate fusion is enabled (default: true)
    fusion_enabled: bool,
    /// Whether parallel execution is enabled (default: false).
    /// When enabled and state vector is large enough (14+ qubits), gate operations
    /// are parallelized across multiple threads using rayon.
    parallel_enabled: bool,
    /// Number of threads for parallel execution (None = use all available).
    num_threads: Option<usize>,
}

impl<R: Rng + Clone> Clone for StateVecSoA<R> {
    fn clone(&self) -> Self {
        Self {
            real: self.real.clone(),
            imag: self.imag.clone(),
            num_qubits: self.num_qubits,
            rng: self.rng.clone(),
            // Don't clone scratch buffers - they're lazily allocated as needed
            scratch_real: Vec::new(),
            scratch_imag: Vec::new(),
            pending_gates: self.pending_gates.clone(),
            fusion_enabled: self.fusion_enabled,
            parallel_enabled: self.parallel_enabled,
            num_threads: self.num_threads,
        }
    }
}

// Constructors that use the default PecosRng
impl StateVecSoA {
    /// Creates a new state vector initialized to |0...0⟩.
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> StateVecSoA<PecosRng> {
        let rng = rand::make_rng();
        StateVecSoA::with_rng(num_qubits, rng)
    }

    /// Creates a new state vector with a specific seed for reproducibility.
    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> StateVecSoA<PecosRng> {
        let rng = PecosRng::seed_from_u64(seed);
        StateVecSoA::with_rng(num_qubits, rng)
    }
}

impl StateVecSoA<PecosRng> {
    /// Sets the random seed for measurements.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = PecosRng::seed_from_u64(seed);
    }
}

impl<R> StateVecSoA<R>
where
    R: Rng,
{
    /// Creates a new state vector with a custom RNG.
    #[must_use]
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        let size = 1 << num_qubits;
        let mut real = vec![0.0; size];
        let imag = vec![0.0; size];
        real[0] = 1.0; // |0...0⟩ state

        Self {
            real,
            imag,
            num_qubits,
            rng,
            scratch_real: Vec::new(),
            scratch_imag: Vec::new(),
            pending_gates: vec![None; num_qubits],
            fusion_enabled: true,
            parallel_enabled: false,
            num_threads: None,
        }
    }

    // =========================================================================
    // Gate Fusion Methods
    // =========================================================================

    /// Enable or disable gate fusion.
    ///
    /// When enabled, consecutive single-qubit gates on the same qubit are
    /// accumulated into a single fused matrix and applied together, reducing
    /// memory passes. This can provide significant speedups (up to 10x) for
    /// circuits with many consecutive single-qubit gates on the same qubit.
    ///
    /// When disabled (default), gates are applied immediately. This is better
    /// for typical circuits with frequent two-qubit gates, which cause flushes
    /// that negate fusion benefits.
    #[inline]
    pub fn set_fusion(&mut self, enabled: bool) {
        if !enabled && self.fusion_enabled {
            // Flush all pending gates before disabling
            self.flush();
        }
        self.fusion_enabled = enabled;
    }

    /// Returns whether gate fusion is enabled.
    #[inline]
    #[must_use]
    pub fn fusion_enabled(&self) -> bool {
        self.fusion_enabled
    }

    /// Enable or disable parallel execution for large state vectors.
    ///
    /// When enabled, gate operations are parallelized across multiple threads
    /// using rayon for state vectors with 14+ qubits (16K+ amplitudes).
    /// For smaller state vectors, parallelism overhead exceeds benefits.
    ///
    /// **Default: disabled.** This is appropriate for most use cases since:
    /// - `MonteCarloEngine` already parallelizes at the shot level
    /// - Single-shot scenarios with amplitude inspection benefit from parallelism
    ///
    /// Requires the `parallel` feature to be enabled at compile time.
    ///
    /// # Example
    /// ```
    /// use pecos_simulators::StateVecSoA;
    ///
    /// let mut sim = StateVecSoA::new(4);
    /// sim.set_parallel(true);
    /// ```
    #[inline]
    pub fn set_parallel(&mut self, enabled: bool) -> &mut Self {
        self.parallel_enabled = enabled;
        self
    }

    /// Enable parallel execution (builder pattern).
    ///
    /// This is equivalent to `set_parallel(true)` but provides a more fluent API.
    ///
    /// # Example
    /// ```
    /// use pecos_simulators::StateVecSoA;
    ///
    /// let mut sim = StateVecSoA::new(4);
    /// sim.parallel(true).num_threads(Some(4));
    /// ```
    #[inline]
    pub fn parallel(&mut self, enabled: bool) -> &mut Self {
        self.set_parallel(enabled)
    }

    /// Returns whether parallel execution is enabled.
    #[inline]
    #[must_use]
    pub fn parallel_enabled(&self) -> bool {
        self.parallel_enabled
    }

    /// Minimum number of qubits for parallel execution to be beneficial.
    /// Below this threshold, parallelism overhead exceeds benefits.
    #[cfg(feature = "parallel")]
    const PARALLEL_THRESHOLD_QUBITS: usize = 14;

    /// Set the number of threads for parallel execution.
    ///
    /// - `None` (default): Use all available threads (rayon's default behavior)
    /// - `Some(n)`: Use exactly `n` threads for parallel operations
    ///
    /// This creates a custom thread pool when parallel operations are executed.
    /// Only takes effect when parallel execution is enabled via `set_parallel(true)`.
    ///
    /// # Example
    /// ```
    /// use pecos_simulators::StateVecSoA;
    ///
    /// let mut sim = StateVecSoA::new(4);
    /// sim.parallel(true).num_threads(Some(4));
    /// ```
    #[inline]
    pub fn set_num_threads(&mut self, num_threads: Option<usize>) -> &mut Self {
        self.num_threads = num_threads;
        self
    }

    /// Set the number of threads for parallel execution (builder pattern).
    ///
    /// Alias for `set_num_threads` for fluent API.
    #[inline]
    pub fn num_threads(&mut self, num_threads: Option<usize>) -> &mut Self {
        self.set_num_threads(num_threads)
    }

    /// Returns the configured number of threads for parallel execution.
    #[inline]
    #[must_use]
    pub fn get_num_threads(&self) -> Option<usize> {
        self.num_threads
    }

    /// Queue a single-qubit gate for fusion.
    /// If fusion is disabled, applies the gate immediately.
    #[inline]
    fn queue_gate(&mut self, qubit: usize, gate: &Complex2x2) {
        if !self.fusion_enabled {
            // Apply immediately
            self.apply_fused_matrix(qubit, gate);
            return;
        }

        // Accumulate the gate
        match &mut self.pending_gates[qubit] {
            Some(accumulated) => {
                // Multiply: accumulated = gate * accumulated
                // Gates are applied right-to-left, so new gate multiplies from the left
                *accumulated = gate.mul(accumulated);
            }
            None => {
                // First gate for this qubit
                self.pending_gates[qubit] = Some(*gate);
            }
        }
    }

    /// Flush pending gates for a specific qubit.
    #[inline]
    fn flush_qubit(&mut self, qubit: usize) {
        if let Some(matrix) = self.pending_gates[qubit].take()
            && !matrix.is_identity()
        {
            self.apply_fused_matrix(qubit, &matrix);
        }
    }

    /// Flush all pending gates for all qubits.
    ///
    /// This is called automatically before two-qubit gates and measurements.
    /// You can also call it manually to ensure all pending gates are applied.
    ///
    /// For large state vectors (16+ qubits), uses cache-blocked iteration:
    /// low-stride qubits (stride < block size) are processed together per
    /// cache block, so each block is loaded from memory once instead of once
    /// per gate. High-stride qubits are flushed individually.
    pub fn flush(&mut self) {
        // Block size in amplitudes: 2^14 = 16384 × 16 bytes = 256KB (fits in L2 cache)
        const BLOCK_BITS: usize = 14;

        // Only use blocking when the state vector is large enough for cache effects
        // to matter (> 2^16 amplitudes = 1MB) and there are multiple pending gates
        let pending_count = self.pending_gates[..self.num_qubits]
            .iter()
            .filter(|g| g.is_some())
            .count();

        // Only use blocking when the state vector exceeds L3 cache (~16MB).
        // At 21+ qubits (32MB+), multiple flush passes cause real cache thrashing.
        // Below that, the simple approach is faster because data stays in cache.
        if self.num_qubits < 21 || pending_count < 3 {
            for qubit in 0..self.num_qubits {
                self.flush_qubit(qubit);
            }
            return;
        }

        self.flush_blocked(BLOCK_BITS);
    }

    /// Cache-blocked flush: apply low-stride pending gates per block, then
    /// high-stride gates individually.
    fn flush_blocked(&mut self, block_bits: usize) {
        let n = self.real.len();
        let block_size = 1usize << block_bits;
        let max_low_qubit = block_bits.min(self.num_qubits);

        // Collect low-stride pending gates (stride fits within one block)
        let mut low_gates: Vec<(usize, Complex2x2)> = Vec::new();
        for q in 0..max_low_qubit {
            if let Some(matrix) = self.pending_gates[q].take()
                && !matrix.is_identity()
            {
                low_gates.push((q, matrix));
            }
        }

        // Apply low-stride gates in blocked fashion: one block loaded into L2,
        // all gates applied before moving to next block
        if !low_gates.is_empty() {
            for block_start in (0..n).step_by(block_size) {
                for &(q, ref m) in &low_gates {
                    let step = 1 << q;
                    let block_end = block_start + block_size;

                    if step >= 4 {
                        let a_re = f64x4::splat(m.a_re);
                        let a_im = f64x4::splat(m.a_im);
                        let b_re = f64x4::splat(m.b_re);
                        let b_im = f64x4::splat(m.b_im);
                        let c_re = f64x4::splat(m.c_re);
                        let c_im = f64x4::splat(m.c_im);
                        let d_re = f64x4::splat(m.d_re);
                        let d_im = f64x4::splat(m.d_im);

                        for i in (block_start..block_end).step_by(step * 2) {
                            let mut j = i;
                            while j + 4 <= i + step {
                                let pj = j + step;

                                let ar = f64x4::from(&self.real[j..j + 4]);
                                let ai = f64x4::from(&self.imag[j..j + 4]);
                                let br = f64x4::from(&self.real[pj..pj + 4]);
                                let bi = f64x4::from(&self.imag[pj..pj + 4]);

                                let nr: [f64; 4] =
                                    ((a_re * ar - a_im * ai) + (b_re * br - b_im * bi)).into();
                                let ni: [f64; 4] =
                                    ((a_re * ai + a_im * ar) + (b_re * bi + b_im * br)).into();
                                let pr: [f64; 4] =
                                    ((c_re * ar - c_im * ai) + (d_re * br - d_im * bi)).into();
                                let pi: [f64; 4] =
                                    ((c_re * ai + c_im * ar) + (d_re * bi + d_im * br)).into();

                                self.real[j..j + 4].copy_from_slice(&nr);
                                self.imag[j..j + 4].copy_from_slice(&ni);
                                self.real[pj..pj + 4].copy_from_slice(&pr);
                                self.imag[pj..pj + 4].copy_from_slice(&pi);

                                j += 4;
                            }
                        }
                    } else {
                        for i in (block_start..block_end).step_by(step * 2) {
                            for j in i..(i + step) {
                                let pj = j + step;
                                let ar = self.real[j];
                                let ai = self.imag[j];
                                let br = self.real[pj];
                                let bi = self.imag[pj];

                                self.real[j] =
                                    (m.a_re * ar - m.a_im * ai) + (m.b_re * br - m.b_im * bi);
                                self.imag[j] =
                                    (m.a_re * ai + m.a_im * ar) + (m.b_re * bi + m.b_im * br);
                                self.real[pj] =
                                    (m.c_re * ar - m.c_im * ai) + (m.d_re * br - m.d_im * bi);
                                self.imag[pj] =
                                    (m.c_re * ai + m.c_im * ar) + (m.d_re * bi + m.d_im * br);
                            }
                        }
                    }
                }
            }
        }

        // Flush remaining high-stride qubits, pairing adjacent qubits to halve
        // memory passes. For each pair (q, q+1), both gates are applied in one
        // pass: load 4 amplitude groups, apply M_lo then M_hi, store.
        let mut q = max_low_qubit;
        while q + 1 < self.num_qubits {
            let have_lo = self.pending_gates[q].is_some();
            let have_hi = self.pending_gates[q + 1].is_some();

            if have_lo && have_hi {
                let m_lo = self.pending_gates[q].take().unwrap();
                let m_hi = self.pending_gates[q + 1].take().unwrap();
                let lo_id = m_lo.is_identity();
                let hi_id = m_hi.is_identity();

                if !lo_id && !hi_id {
                    self.flush_pair(q, q + 1, &m_lo, &m_hi);
                } else if !lo_id {
                    self.apply_fused_matrix(q, &m_lo);
                } else if !hi_id {
                    self.apply_fused_matrix(q + 1, &m_hi);
                }
                q += 2;
            } else {
                if have_lo {
                    self.flush_qubit(q);
                }
                if have_hi {
                    self.flush_qubit(q + 1);
                }
                q += 2;
            }
        }
        // Handle leftover odd qubit
        if q < self.num_qubits {
            self.flush_qubit(q);
        }
    }

    /// Apply two independent single-qubit gates in one pass over the state vector.
    /// For adjacent qubits (`q_lo`, `q_hi` = `q_lo` + 1), loads groups of 4 amplitudes,
    /// applies both matrices, and stores — one pass instead of two.
    fn flush_pair(&mut self, q_lo: usize, q_hi: usize, m_lo: &Complex2x2, m_hi: &Complex2x2) {
        let n = self.real.len();
        let step_lo = 1usize << q_lo;
        let step_hi = 1usize << q_hi;

        // SIMD splats for M_lo
        let la_re = f64x4::splat(m_lo.a_re);
        let la_im = f64x4::splat(m_lo.a_im);
        let lb_re = f64x4::splat(m_lo.b_re);
        let lb_im = f64x4::splat(m_lo.b_im);
        let lc_re = f64x4::splat(m_lo.c_re);
        let lc_im = f64x4::splat(m_lo.c_im);
        let ld_re = f64x4::splat(m_lo.d_re);
        let ld_im = f64x4::splat(m_lo.d_im);
        // SIMD splats for M_hi
        let ha_re = f64x4::splat(m_hi.a_re);
        let ha_im = f64x4::splat(m_hi.a_im);
        let hb_re = f64x4::splat(m_hi.b_re);
        let hb_im = f64x4::splat(m_hi.b_im);
        let hc_re = f64x4::splat(m_hi.c_re);
        let hc_im = f64x4::splat(m_hi.c_im);
        let hd_re = f64x4::splat(m_hi.d_re);
        let hd_im = f64x4::splat(m_hi.d_im);

        for i_hi in (0..n).step_by(step_hi * 2) {
            for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                let mut off = 0;
                while off + 4 <= step_lo {
                    let base = i_lo + off;
                    let i00 = base;
                    let i01 = base + step_lo;
                    let i10 = base + step_hi;
                    let i11 = base + step_lo + step_hi;

                    // Load 4 amplitude groups
                    let r00 = f64x4::from(&self.real[i00..i00 + 4]);
                    let m00 = f64x4::from(&self.imag[i00..i00 + 4]);
                    let r01 = f64x4::from(&self.real[i01..i01 + 4]);
                    let m01 = f64x4::from(&self.imag[i01..i01 + 4]);
                    let r10 = f64x4::from(&self.real[i10..i10 + 4]);
                    let m10 = f64x4::from(&self.imag[i10..i10 + 4]);
                    let r11 = f64x4::from(&self.real[i11..i11 + 4]);
                    let m11 = f64x4::from(&self.imag[i11..i11 + 4]);

                    // Apply M_lo to (00,01) pair and (10,11) pair
                    let t00r = (la_re * r00 - la_im * m00) + (lb_re * r01 - lb_im * m01);
                    let t00i = (la_re * m00 + la_im * r00) + (lb_re * m01 + lb_im * r01);
                    let t01r = (lc_re * r00 - lc_im * m00) + (ld_re * r01 - ld_im * m01);
                    let t01i = (lc_re * m00 + lc_im * r00) + (ld_re * m01 + ld_im * r01);
                    let t10r = (la_re * r10 - la_im * m10) + (lb_re * r11 - lb_im * m11);
                    let t10i = (la_re * m10 + la_im * r10) + (lb_re * m11 + lb_im * r11);
                    let t11r = (lc_re * r10 - lc_im * m10) + (ld_re * r11 - ld_im * m11);
                    let t11i = (lc_re * m10 + lc_im * r10) + (ld_re * m11 + ld_im * r11);

                    // Apply M_hi to (00,10) pair and (01,11) pair
                    let f00r: [f64; 4] =
                        ((ha_re * t00r - ha_im * t00i) + (hb_re * t10r - hb_im * t10i)).into();
                    let f00i: [f64; 4] =
                        ((ha_re * t00i + ha_im * t00r) + (hb_re * t10i + hb_im * t10r)).into();
                    let f10r: [f64; 4] =
                        ((hc_re * t00r - hc_im * t00i) + (hd_re * t10r - hd_im * t10i)).into();
                    let f10i: [f64; 4] =
                        ((hc_re * t00i + hc_im * t00r) + (hd_re * t10i + hd_im * t10r)).into();
                    let f01r: [f64; 4] =
                        ((ha_re * t01r - ha_im * t01i) + (hb_re * t11r - hb_im * t11i)).into();
                    let f01i: [f64; 4] =
                        ((ha_re * t01i + ha_im * t01r) + (hb_re * t11i + hb_im * t11r)).into();
                    let f11r: [f64; 4] =
                        ((hc_re * t01r - hc_im * t01i) + (hd_re * t11r - hd_im * t11i)).into();
                    let f11i: [f64; 4] =
                        ((hc_re * t01i + hc_im * t01r) + (hd_re * t11i + hd_im * t11r)).into();

                    // Store
                    self.real[i00..i00 + 4].copy_from_slice(&f00r);
                    self.imag[i00..i00 + 4].copy_from_slice(&f00i);
                    self.real[i01..i01 + 4].copy_from_slice(&f01r);
                    self.imag[i01..i01 + 4].copy_from_slice(&f01i);
                    self.real[i10..i10 + 4].copy_from_slice(&f10r);
                    self.imag[i10..i10 + 4].copy_from_slice(&f10i);
                    self.real[i11..i11 + 4].copy_from_slice(&f11r);
                    self.imag[i11..i11 + 4].copy_from_slice(&f11i);

                    off += 4;
                }
            }
        }
    }

    /// Flush pending gates for qubits involved in a two-qubit operation.
    #[inline]
    fn flush_two_qubit(&mut self, q1: usize, q2: usize) {
        self.flush_qubit(q1);
        self.flush_qubit(q2);
    }

    /// Apply a fused 2x2 complex matrix to a single qubit using SIMD.
    /// When parallel execution is enabled and the state vector is large enough,
    /// this operation is parallelized across multiple threads.
    fn apply_fused_matrix(&mut self, q: usize, m: &Complex2x2) {
        let step = 1 << q;
        let n = self.real.len();

        // Check if we should use parallel execution.
        // Conditions:
        // 1. Parallel feature enabled at compile time
        // 2. parallel_enabled flag set at runtime
        // 3. State vector is large enough (>= threshold qubits)
        // 4. Step is large enough for SIMD (>= 4)
        // 5. There are enough blocks to parallelize (>= 4 blocks)
        //    This ensures we don't pay thread overhead for single-block operations.
        #[cfg(feature = "parallel")]
        if self.parallel_enabled
            && self.num_qubits >= Self::PARALLEL_THRESHOLD_QUBITS
            && step >= 4
            && (n / (step * 2)) >= 4
        {
            self.apply_fused_matrix_parallel(q, m);
            return;
        }

        if step < 4 {
            // Scalar fallback for small steps
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;

                    let alpha_re = self.real[j];
                    let alpha_im = self.imag[j];
                    let beta_re = self.real[paired_j];
                    let beta_im = self.imag[paired_j];

                    // new_alpha = a * alpha + b * beta
                    self.real[j] = (m.a_re * alpha_re - m.a_im * alpha_im)
                        + (m.b_re * beta_re - m.b_im * beta_im);
                    self.imag[j] = (m.a_re * alpha_im + m.a_im * alpha_re)
                        + (m.b_re * beta_im + m.b_im * beta_re);

                    // new_beta = c * alpha + d * beta
                    self.real[paired_j] = (m.c_re * alpha_re - m.c_im * alpha_im)
                        + (m.d_re * beta_re - m.d_im * beta_im);
                    self.imag[paired_j] = (m.c_re * alpha_im + m.c_im * alpha_re)
                        + (m.d_re * beta_im + m.d_im * beta_re);
                }
            }
        } else {
            // SIMD path
            let a_re = f64x4::splat(m.a_re);
            let a_im = f64x4::splat(m.a_im);
            let b_re = f64x4::splat(m.b_re);
            let b_im = f64x4::splat(m.b_im);
            let c_re = f64x4::splat(m.c_re);
            let c_im = f64x4::splat(m.c_im);
            let d_re = f64x4::splat(m.d_re);
            let d_im = f64x4::splat(m.d_im);

            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let paired_j = j + step;

                    let alpha_re_v = f64x4::from(&self.real[j..j + 4]);
                    let alpha_im_v = f64x4::from(&self.imag[j..j + 4]);
                    let beta_re_v = f64x4::from(&self.real[paired_j..paired_j + 4]);
                    let beta_im_v = f64x4::from(&self.imag[paired_j..paired_j + 4]);

                    // new_alpha = a * alpha + b * beta
                    let new_alpha_re = (a_re * alpha_re_v - a_im * alpha_im_v)
                        + (b_re * beta_re_v - b_im * beta_im_v);
                    let new_alpha_im = (a_re * alpha_im_v + a_im * alpha_re_v)
                        + (b_re * beta_im_v + b_im * beta_re_v);

                    // new_beta = c * alpha + d * beta
                    let new_beta_re = (c_re * alpha_re_v - c_im * alpha_im_v)
                        + (d_re * beta_re_v - d_im * beta_im_v);
                    let new_beta_im = (c_re * alpha_im_v + c_im * alpha_re_v)
                        + (d_re * beta_im_v + d_im * beta_re_v);

                    let arr_alpha_re: [f64; 4] = new_alpha_re.into();
                    let arr_alpha_im: [f64; 4] = new_alpha_im.into();
                    let arr_beta_re: [f64; 4] = new_beta_re.into();
                    let arr_beta_im: [f64; 4] = new_beta_im.into();

                    self.real[j..j + 4].copy_from_slice(&arr_alpha_re);
                    self.imag[j..j + 4].copy_from_slice(&arr_alpha_im);
                    self.real[paired_j..paired_j + 4].copy_from_slice(&arr_beta_re);
                    self.imag[paired_j..paired_j + 4].copy_from_slice(&arr_beta_im);

                    j += 4;
                }
            }
        }
    }

    /// Parallel version of `apply_fused_matrix` using rayon.
    /// Each block of size `step * 2` is processed independently.
    /// Uses a custom thread pool if `num_threads` is set, otherwise uses rayon's global pool.
    #[cfg(feature = "parallel")]
    fn apply_fused_matrix_parallel(&mut self, q: usize, m: &Complex2x2) {
        let step = 1 << q;
        let n = self.real.len();
        let block_size = step * 2;
        let num_blocks = n / block_size;

        // Wrap raw pointers in SendPtr for parallel access
        // SAFETY: Each parallel iteration accesses a disjoint block of indices.
        // Block i accesses indices [i*block_size .. (i+1)*block_size], which are non-overlapping.
        let real_ptr = SendPtr(self.real.as_mut_ptr());
        let imag_ptr = SendPtr(self.imag.as_mut_ptr());

        // Create the parallel work closure
        let work = || {
            (0..num_blocks).into_par_iter().for_each(|block_idx| {
                let block_start = block_idx * block_size;

                // SIMD constants
                let a_re = f64x4::splat(m.a_re);
                let a_im = f64x4::splat(m.a_im);
                let b_re = f64x4::splat(m.b_re);
                let b_im = f64x4::splat(m.b_im);
                let c_re = f64x4::splat(m.c_re);
                let c_im = f64x4::splat(m.c_im);
                let d_re = f64x4::splat(m.d_re);
                let d_im = f64x4::splat(m.d_im);

                // Get raw pointers from SendPtr wrappers
                let rp = real_ptr.ptr();
                let ip = imag_ptr.ptr();

                let mut j = block_start;
                while j + 4 <= block_start + step {
                    let paired_j = j + step;

                    // SAFETY: j and paired_j are within bounds and each block is disjoint
                    unsafe {
                        let alpha_re_v = f64x4::from(std::slice::from_raw_parts(rp.add(j), 4));
                        let alpha_im_v = f64x4::from(std::slice::from_raw_parts(ip.add(j), 4));
                        let beta_re_v =
                            f64x4::from(std::slice::from_raw_parts(rp.add(paired_j), 4));
                        let beta_im_v =
                            f64x4::from(std::slice::from_raw_parts(ip.add(paired_j), 4));

                        // new_alpha = a * alpha + b * beta
                        let new_alpha_re = (a_re * alpha_re_v - a_im * alpha_im_v)
                            + (b_re * beta_re_v - b_im * beta_im_v);
                        let new_alpha_im = (a_re * alpha_im_v + a_im * alpha_re_v)
                            + (b_re * beta_im_v + b_im * beta_re_v);

                        // new_beta = c * alpha + d * beta
                        let new_beta_re = (c_re * alpha_re_v - c_im * alpha_im_v)
                            + (d_re * beta_re_v - d_im * beta_im_v);
                        let new_beta_im = (c_re * alpha_im_v + c_im * alpha_re_v)
                            + (d_re * beta_im_v + d_im * beta_re_v);

                        let arr_alpha_re: [f64; 4] = new_alpha_re.into();
                        let arr_alpha_im: [f64; 4] = new_alpha_im.into();
                        let arr_beta_re: [f64; 4] = new_beta_re.into();
                        let arr_beta_im: [f64; 4] = new_beta_im.into();

                        std::ptr::copy_nonoverlapping(arr_alpha_re.as_ptr(), rp.add(j), 4);
                        std::ptr::copy_nonoverlapping(arr_alpha_im.as_ptr(), ip.add(j), 4);
                        std::ptr::copy_nonoverlapping(arr_beta_re.as_ptr(), rp.add(paired_j), 4);
                        std::ptr::copy_nonoverlapping(arr_beta_im.as_ptr(), ip.add(paired_j), 4);
                    }

                    j += 4;
                }
            });
        };

        // Execute with custom thread pool if num_threads is specified
        if let Some(num_threads) = self.num_threads {
            // Build a custom thread pool with the specified number of threads
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("Failed to build thread pool");
            pool.install(work);
        } else {
            // Use rayon's global thread pool (all available threads)
            work();
        }
    }

    // =========================================================================
    // Specialized Gate Implementations (used when fusion is disabled)
    // =========================================================================

    /// Specialized Z gate: negate amplitudes where qubit bit is 1.
    /// Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    #[inline]
    fn apply_z_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        if step >= 4 {
            // SIMD path: negate 4 elements at a time
            let neg_one = f64x4::splat(-1.0);
            for i in (0..n).step_by(step * 2) {
                let mut j = i + step;
                while j + 4 <= i + step * 2 {
                    let re = f64x4::from(&self.real[j..j + 4]);
                    let im = f64x4::from(&self.imag[j..j + 4]);
                    let neg_re: [f64; 4] = (re * neg_one).into();
                    let neg_im: [f64; 4] = (im * neg_one).into();
                    self.real[j..j + 4].copy_from_slice(&neg_re);
                    self.imag[j..j + 4].copy_from_slice(&neg_im);
                    j += 4;
                }
            }
        } else {
            // Scalar fallback
            for i in (0..n).step_by(step * 2) {
                for j in (i + step)..(i + step * 2) {
                    self.real[j] = -self.real[j];
                    self.imag[j] = -self.imag[j];
                }
            }
        }
    }

    /// Specialized X gate: swap amplitude pairs.
    /// X|0⟩ = |1⟩, X|1⟩ = |0⟩
    #[inline]
    fn apply_x_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        // Use slice swap for better performance
        for i in (0..n).step_by(step * 2) {
            let (left_re, right_re) = self.real[i..i + step * 2].split_at_mut(step);
            left_re.swap_with_slice(right_re);

            let (left_im, right_im) = self.imag[i..i + step * 2].split_at_mut(step);
            left_im.swap_with_slice(right_im);
        }
    }

    /// Specialized Y gate: swap with phase factors.
    /// Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
    #[inline]
    fn apply_y_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        if step >= 4 {
            // SIMD path
            for i in (0..n).step_by(step * 2) {
                let mut j = 0;
                while j + 4 <= step {
                    let idx0 = i + j;
                    let idx1 = i + j + step;

                    let re0 = f64x4::from(&self.real[idx0..idx0 + 4]);
                    let im0 = f64x4::from(&self.imag[idx0..idx0 + 4]);
                    let re1 = f64x4::from(&self.real[idx1..idx1 + 4]);
                    let im1 = f64x4::from(&self.imag[idx1..idx1 + 4]);

                    // New |0⟩ = -i * old|1⟩: (re1, im1) * -i = (im1, -re1)
                    // New |1⟩ = i * old|0⟩: (re0, im0) * i = (-im0, re0)
                    let new_re0: [f64; 4] = im1.into();
                    let new_im0: [f64; 4] = (-re1).into();
                    let new_re1: [f64; 4] = (-im0).into();
                    let new_im1: [f64; 4] = re0.into();

                    self.real[idx0..idx0 + 4].copy_from_slice(&new_re0);
                    self.imag[idx0..idx0 + 4].copy_from_slice(&new_im0);
                    self.real[idx1..idx1 + 4].copy_from_slice(&new_re1);
                    self.imag[idx1..idx1 + 4].copy_from_slice(&new_im1);

                    j += 4;
                }
            }
        } else {
            // Scalar fallback
            for i in (0..n).step_by(step * 2) {
                for j in 0..step {
                    let idx0 = i + j;
                    let idx1 = i + j + step;

                    let re0 = self.real[idx0];
                    let im0 = self.imag[idx0];
                    let re1 = self.real[idx1];
                    let im1 = self.imag[idx1];

                    // New |0⟩ = -i * old|1⟩
                    self.real[idx0] = im1;
                    self.imag[idx0] = -re1;
                    // New |1⟩ = i * old|0⟩
                    self.real[idx1] = -im0;
                    self.imag[idx1] = re0;
                }
            }
        }
    }

    /// Specialized SZ (S) gate: multiply by i where qubit bit is 1.
    /// S|0⟩ = |0⟩, S|1⟩ = i|1⟩
    #[inline]
    fn apply_sz_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        // Multiply by i: (re, im) -> (-im, re)
        if step >= 4 {
            // SIMD path
            for i in (0..n).step_by(step * 2) {
                let mut j = i + step;
                while j + 4 <= i + step * 2 {
                    let re = f64x4::from(&self.real[j..j + 4]);
                    let im = f64x4::from(&self.imag[j..j + 4]);
                    let new_re: [f64; 4] = (-im).into();
                    let new_im: [f64; 4] = re.into();
                    self.real[j..j + 4].copy_from_slice(&new_re);
                    self.imag[j..j + 4].copy_from_slice(&new_im);
                    j += 4;
                }
            }
        } else {
            // Scalar fallback
            for i in (0..n).step_by(step * 2) {
                for j in (i + step)..(i + step * 2) {
                    let re = self.real[j];
                    let im = self.imag[j];
                    self.real[j] = -im;
                    self.imag[j] = re;
                }
            }
        }
    }

    /// Specialized SZDG (S†) gate: multiply by -i where qubit bit is 1.
    /// S†|0⟩ = |0⟩, S†|1⟩ = -i|1⟩
    #[inline]
    fn apply_szdg_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        // Multiply by -i: (re, im) -> (im, -re)
        if step >= 4 {
            // SIMD path
            for i in (0..n).step_by(step * 2) {
                let mut j = i + step;
                while j + 4 <= i + step * 2 {
                    let re = f64x4::from(&self.real[j..j + 4]);
                    let im = f64x4::from(&self.imag[j..j + 4]);
                    let new_re: [f64; 4] = im.into();
                    let new_im: [f64; 4] = (-re).into();
                    self.real[j..j + 4].copy_from_slice(&new_re);
                    self.imag[j..j + 4].copy_from_slice(&new_im);
                    j += 4;
                }
            }
        } else {
            // Scalar fallback
            for i in (0..n).step_by(step * 2) {
                for j in (i + step)..(i + step * 2) {
                    let re = self.real[j];
                    let im = self.imag[j];
                    self.real[j] = im;
                    self.imag[j] = -re;
                }
            }
        }
    }

    /// Specialized H gate using SIMD.
    /// H|0⟩ = (|0⟩ + |1⟩)/√2, H|1⟩ = (|0⟩ - |1⟩)/√2
    #[inline]
    fn apply_h_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        if step >= 4 {
            // SIMD path
            let factor = f64x4::splat(inv_sqrt2);
            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let paired_j = j + step;

                    let a_re = f64x4::from(&self.real[j..j + 4]);
                    let a_im = f64x4::from(&self.imag[j..j + 4]);
                    let b_re = f64x4::from(&self.real[paired_j..paired_j + 4]);
                    let b_im = f64x4::from(&self.imag[paired_j..paired_j + 4]);

                    // new_a = (a + b) / sqrt(2)
                    // new_b = (a - b) / sqrt(2)
                    let new_a_re: [f64; 4] = ((a_re + b_re) * factor).into();
                    let new_a_im: [f64; 4] = ((a_im + b_im) * factor).into();
                    let new_b_re: [f64; 4] = ((a_re - b_re) * factor).into();
                    let new_b_im: [f64; 4] = ((a_im - b_im) * factor).into();

                    self.real[j..j + 4].copy_from_slice(&new_a_re);
                    self.imag[j..j + 4].copy_from_slice(&new_a_im);
                    self.real[paired_j..paired_j + 4].copy_from_slice(&new_b_re);
                    self.imag[paired_j..paired_j + 4].copy_from_slice(&new_b_im);

                    j += 4;
                }
            }
        } else {
            // Scalar fallback
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;

                    let a_re = self.real[j];
                    let a_im = self.imag[j];
                    let b_re = self.real[paired_j];
                    let b_im = self.imag[paired_j];

                    self.real[j] = (a_re + b_re) * inv_sqrt2;
                    self.imag[j] = (a_im + b_im) * inv_sqrt2;
                    self.real[paired_j] = (a_re - b_re) * inv_sqrt2;
                    self.imag[paired_j] = (a_im - b_im) * inv_sqrt2;
                }
            }
        }
    }

    /// Specialized SX gate: (1+i)/2 on diagonal, (1-i)/2 off-diagonal
    #[inline]
    fn apply_sx_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        // SX = (1/2) * [[1+i, 1-i], [1-i, 1+i]]
        if step >= 4 {
            // SIMD path
            let half = f64x4::splat(0.5);
            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let paired_j = j + step;

                    let a_re = f64x4::from(&self.real[j..j + 4]);
                    let a_im = f64x4::from(&self.imag[j..j + 4]);
                    let b_re = f64x4::from(&self.real[paired_j..paired_j + 4]);
                    let b_im = f64x4::from(&self.imag[paired_j..paired_j + 4]);

                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let diff_re = a_re - b_re;
                    let diff_im = a_im - b_im;

                    let new_a_re: [f64; 4] = ((sum_re - diff_im) * half).into();
                    let new_a_im: [f64; 4] = ((sum_im + diff_re) * half).into();
                    let new_b_re: [f64; 4] = ((sum_re + diff_im) * half).into();
                    let new_b_im: [f64; 4] = ((sum_im - diff_re) * half).into();

                    self.real[j..j + 4].copy_from_slice(&new_a_re);
                    self.imag[j..j + 4].copy_from_slice(&new_a_im);
                    self.real[paired_j..paired_j + 4].copy_from_slice(&new_b_re);
                    self.imag[paired_j..paired_j + 4].copy_from_slice(&new_b_im);

                    j += 4;
                }
            }
        } else {
            // Scalar fallback
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;

                    let a_re = self.real[j];
                    let a_im = self.imag[j];
                    let b_re = self.real[paired_j];
                    let b_im = self.imag[paired_j];

                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let diff_re = a_re - b_re;
                    let diff_im = a_im - b_im;

                    self.real[j] = (sum_re - diff_im) * 0.5;
                    self.imag[j] = (sum_im + diff_re) * 0.5;
                    self.real[paired_j] = (sum_re + diff_im) * 0.5;
                    self.imag[paired_j] = (sum_im - diff_re) * 0.5;
                }
            }
        }
    }

    /// Specialized SXDG gate: (1-i)/2 on diagonal, (1+i)/2 off-diagonal
    #[inline]
    fn apply_sxdg_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        // SXDG = (1/2) * [[1-i, 1+i], [1+i, 1-i]]
        if step >= 4 {
            // SIMD path
            let half = f64x4::splat(0.5);
            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let paired_j = j + step;

                    let a_re = f64x4::from(&self.real[j..j + 4]);
                    let a_im = f64x4::from(&self.imag[j..j + 4]);
                    let b_re = f64x4::from(&self.real[paired_j..paired_j + 4]);
                    let b_im = f64x4::from(&self.imag[paired_j..paired_j + 4]);

                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let diff_re = a_re - b_re;
                    let diff_im = a_im - b_im;

                    // SXDG is conjugate of SX: swap signs on i terms
                    let new_a_re: [f64; 4] = ((sum_re + diff_im) * half).into();
                    let new_a_im: [f64; 4] = ((sum_im - diff_re) * half).into();
                    let new_b_re: [f64; 4] = ((sum_re - diff_im) * half).into();
                    let new_b_im: [f64; 4] = ((sum_im + diff_re) * half).into();

                    self.real[j..j + 4].copy_from_slice(&new_a_re);
                    self.imag[j..j + 4].copy_from_slice(&new_a_im);
                    self.real[paired_j..paired_j + 4].copy_from_slice(&new_b_re);
                    self.imag[paired_j..paired_j + 4].copy_from_slice(&new_b_im);

                    j += 4;
                }
            }
        } else {
            // Scalar fallback
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;

                    let a_re = self.real[j];
                    let a_im = self.imag[j];
                    let b_re = self.real[paired_j];
                    let b_im = self.imag[paired_j];

                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let diff_re = a_re - b_re;
                    let diff_im = a_im - b_im;

                    // SXDG is conjugate of SX: swap signs on i terms
                    self.real[j] = (sum_re + diff_im) * 0.5;
                    self.imag[j] = (sum_im - diff_re) * 0.5;
                    self.real[paired_j] = (sum_re - diff_im) * 0.5;
                    self.imag[paired_j] = (sum_im + diff_re) * 0.5;
                }
            }
        }
    }

    /// Specialized SY gate: (1/2)[[1+i, -1-i], [1+i, 1+i]]
    #[inline]
    fn apply_sy_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        if step >= 4 {
            // SIMD path
            let half = f64x4::splat(0.5);
            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let paired_j = j + step;

                    let a_re = f64x4::from(&self.real[j..j + 4]);
                    let a_im = f64x4::from(&self.imag[j..j + 4]);
                    let b_re = f64x4::from(&self.real[paired_j..paired_j + 4]);
                    let b_im = f64x4::from(&self.imag[paired_j..paired_j + 4]);

                    // new_a = (1+i)/2 * a + (-1-i)/2 * b
                    let new_a_re: [f64; 4] = ((a_re - a_im - b_re + b_im) * half).into();
                    let new_a_im: [f64; 4] = ((a_re + a_im - b_re - b_im) * half).into();

                    // new_b = (1+i)/2 * (a + b)
                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let new_b_re: [f64; 4] = ((sum_re - sum_im) * half).into();
                    let new_b_im: [f64; 4] = ((sum_re + sum_im) * half).into();

                    self.real[j..j + 4].copy_from_slice(&new_a_re);
                    self.imag[j..j + 4].copy_from_slice(&new_a_im);
                    self.real[paired_j..paired_j + 4].copy_from_slice(&new_b_re);
                    self.imag[paired_j..paired_j + 4].copy_from_slice(&new_b_im);

                    j += 4;
                }
            }
        } else {
            // Scalar fallback
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;

                    let a_re = self.real[j];
                    let a_im = self.imag[j];
                    let b_re = self.real[paired_j];
                    let b_im = self.imag[paired_j];

                    let new_a_re = (a_re - a_im - b_re + b_im) * 0.5;
                    let new_a_im = (a_re + a_im - b_re - b_im) * 0.5;

                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let new_b_re = (sum_re - sum_im) * 0.5;
                    let new_b_im = (sum_re + sum_im) * 0.5;

                    self.real[j] = new_a_re;
                    self.imag[j] = new_a_im;
                    self.real[paired_j] = new_b_re;
                    self.imag[paired_j] = new_b_im;
                }
            }
        }
    }

    /// Specialized SYDG gate: (1/2)[[1-i, 1-i], [-1+i, 1-i]]
    #[inline]
    fn apply_sydg_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        if step >= 4 {
            // SIMD path
            let half = f64x4::splat(0.5);
            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let paired_j = j + step;

                    let a_re = f64x4::from(&self.real[j..j + 4]);
                    let a_im = f64x4::from(&self.imag[j..j + 4]);
                    let b_re = f64x4::from(&self.real[paired_j..paired_j + 4]);
                    let b_im = f64x4::from(&self.imag[paired_j..paired_j + 4]);

                    // new_a = (1-i)/2 * (a + b)
                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let new_a_re: [f64; 4] = ((sum_re + sum_im) * half).into();
                    let new_a_im: [f64; 4] = ((sum_im - sum_re) * half).into();

                    // new_b = (-1+i)/2 * a + (1-i)/2 * b
                    let new_b_re: [f64; 4] = ((-a_re - a_im + b_re + b_im) * half).into();
                    let new_b_im: [f64; 4] = ((a_re - a_im - b_re + b_im) * half).into();

                    self.real[j..j + 4].copy_from_slice(&new_a_re);
                    self.imag[j..j + 4].copy_from_slice(&new_a_im);
                    self.real[paired_j..paired_j + 4].copy_from_slice(&new_b_re);
                    self.imag[paired_j..paired_j + 4].copy_from_slice(&new_b_im);

                    j += 4;
                }
            }
        } else {
            // Scalar fallback
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;

                    let a_re = self.real[j];
                    let a_im = self.imag[j];
                    let b_re = self.real[paired_j];
                    let b_im = self.imag[paired_j];

                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let new_a_re = (sum_re + sum_im) * 0.5;
                    let new_a_im = (-sum_re + sum_im) * 0.5;

                    let new_b_re = (-a_re - a_im + b_re + b_im) * 0.5;
                    let new_b_im = (a_re - a_im - b_re + b_im) * 0.5;

                    self.real[j] = new_a_re;
                    self.imag[j] = new_a_im;
                    self.real[paired_j] = new_b_re;
                    self.imag[paired_j] = new_b_im;
                }
            }
        }
    }

    /// Specialized H3 gate: [[0, 1], [i, 0]] - swap with i on lower
    #[inline]
    fn apply_h3_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        for i in (0..n).step_by(step * 2) {
            for j in i..(i + step) {
                let paired_j = j + step;

                let a_re = self.real[j];
                let a_im = self.imag[j];
                let b_re = self.real[paired_j];
                let b_im = self.imag[paired_j];

                // new_a = b
                // new_b = i * a = (-a_im, a_re)
                self.real[j] = b_re;
                self.imag[j] = b_im;
                self.real[paired_j] = -a_im;
                self.imag[paired_j] = a_re;
            }
        }
    }

    /// Specialized H4 gate: [[0, i], [1, 0]] - swap with i on upper
    #[inline]
    fn apply_h4_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        for i in (0..n).step_by(step * 2) {
            for j in i..(i + step) {
                let paired_j = j + step;

                let a_re = self.real[j];
                let a_im = self.imag[j];
                let b_re = self.real[paired_j];
                let b_im = self.imag[paired_j];

                // new_a = i * b = (-b_im, b_re)
                // new_b = a
                self.real[j] = -b_im;
                self.imag[j] = b_re;
                self.real[paired_j] = a_re;
                self.imag[paired_j] = a_im;
            }
        }
    }

    /// Apply a general 2x2 unitary matrix (for gates without special structure)
    #[inline]
    fn apply_general_gate(&mut self, q: usize, m: &Complex2x2) {
        let step = 1 << q;
        let n = self.real.len();

        for i in (0..n).step_by(step * 2) {
            for j in i..(i + step) {
                let paired_j = j + step;

                let a_re = self.real[j];
                let a_im = self.imag[j];
                let b_re = self.real[paired_j];
                let b_im = self.imag[paired_j];

                // new_a = m.a * a + m.b * b
                self.real[j] = (m.a_re * a_re - m.a_im * a_im) + (m.b_re * b_re - m.b_im * b_im);
                self.imag[j] = (m.a_re * a_im + m.a_im * a_re) + (m.b_re * b_im + m.b_im * b_re);

                // new_b = m.c * a + m.d * b
                self.real[paired_j] =
                    (m.c_re * a_re - m.c_im * a_im) + (m.d_re * b_re - m.d_im * b_im);
                self.imag[paired_j] =
                    (m.c_re * a_im + m.c_im * a_re) + (m.d_re * b_im + m.d_im * b_re);
            }
        }
    }

    /// Specialized RZ gate: diagonal rotation [[e^(-i*theta/2), 0], [0, e^(i*theta/2)]]
    #[inline]
    fn apply_rz_gate(&mut self, q: usize, theta: f64) {
        let step = 1 << q;
        let n = self.real.len();

        let half_theta = theta * 0.5;
        let cos_t = half_theta.cos();
        let sin_t = half_theta.sin();

        // |0⟩ component: multiply by e^(-i*theta/2) = cos - i*sin
        // |1⟩ component: multiply by e^(i*theta/2) = cos + i*sin
        if step >= 4 {
            // SIMD path
            let cos_v = f64x4::splat(cos_t);
            let sin_v = f64x4::splat(sin_t);
            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let paired_j = j + step;

                    let a_re = f64x4::from(&self.real[j..j + 4]);
                    let a_im = f64x4::from(&self.imag[j..j + 4]);
                    let b_re = f64x4::from(&self.real[paired_j..paired_j + 4]);
                    let b_im = f64x4::from(&self.imag[paired_j..paired_j + 4]);

                    let new_a_re: [f64; 4] = (a_re * cos_v + a_im * sin_v).into();
                    let new_a_im: [f64; 4] = (a_im * cos_v - a_re * sin_v).into();
                    let new_b_re: [f64; 4] = (b_re * cos_v - b_im * sin_v).into();
                    let new_b_im: [f64; 4] = (b_im * cos_v + b_re * sin_v).into();

                    self.real[j..j + 4].copy_from_slice(&new_a_re);
                    self.imag[j..j + 4].copy_from_slice(&new_a_im);
                    self.real[paired_j..paired_j + 4].copy_from_slice(&new_b_re);
                    self.imag[paired_j..paired_j + 4].copy_from_slice(&new_b_im);

                    j += 4;
                }
            }
        } else {
            // Scalar fallback
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;

                    let a_re = self.real[j];
                    let a_im = self.imag[j];
                    self.real[j] = a_re * cos_t + a_im * sin_t;
                    self.imag[j] = a_im * cos_t - a_re * sin_t;

                    let b_re = self.real[paired_j];
                    let b_im = self.imag[paired_j];
                    self.real[paired_j] = b_re * cos_t - b_im * sin_t;
                    self.imag[paired_j] = b_im * cos_t + b_re * sin_t;
                }
            }
        }
    }

    /// Specialized RX gate: [[cos(t/2), -i*sin(t/2)], [-i*sin(t/2), cos(t/2)]]
    #[inline]
    fn apply_rx_gate(&mut self, q: usize, theta: f64) {
        let step = 1 << q;
        let n = self.real.len();

        let half_theta = theta * 0.5;
        let cos_t = half_theta.cos();
        let sin_t = half_theta.sin();

        if step >= 4 {
            // SIMD path
            let cos_v = f64x4::splat(cos_t);
            let sin_v = f64x4::splat(sin_t);
            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let paired_j = j + step;

                    let a_re = f64x4::from(&self.real[j..j + 4]);
                    let a_im = f64x4::from(&self.imag[j..j + 4]);
                    let b_re = f64x4::from(&self.real[paired_j..paired_j + 4]);
                    let b_im = f64x4::from(&self.imag[paired_j..paired_j + 4]);

                    let new_a_re: [f64; 4] = (cos_v * a_re + sin_v * b_im).into();
                    let new_a_im: [f64; 4] = (cos_v * a_im - sin_v * b_re).into();
                    let new_b_re: [f64; 4] = (sin_v * a_im + cos_v * b_re).into();
                    let new_b_im: [f64; 4] = (cos_v * b_im - sin_v * a_re).into();

                    self.real[j..j + 4].copy_from_slice(&new_a_re);
                    self.imag[j..j + 4].copy_from_slice(&new_a_im);
                    self.real[paired_j..paired_j + 4].copy_from_slice(&new_b_re);
                    self.imag[paired_j..paired_j + 4].copy_from_slice(&new_b_im);

                    j += 4;
                }
            }
        } else {
            // Scalar fallback
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;

                    let a_re = self.real[j];
                    let a_im = self.imag[j];
                    let b_re = self.real[paired_j];
                    let b_im = self.imag[paired_j];

                    self.real[j] = cos_t * a_re + sin_t * b_im;
                    self.imag[j] = cos_t * a_im - sin_t * b_re;
                    self.real[paired_j] = sin_t * a_im + cos_t * b_re;
                    self.imag[paired_j] = -sin_t * a_re + cos_t * b_im;
                }
            }
        }
    }

    /// Specialized RY gate: [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
    #[inline]
    fn apply_ry_gate(&mut self, q: usize, theta: f64) {
        let step = 1 << q;
        let n = self.real.len();

        let half_theta = theta * 0.5;
        let cos_t = half_theta.cos();
        let sin_t = half_theta.sin();

        if step >= 4 {
            // SIMD path
            let cos_v = f64x4::splat(cos_t);
            let sin_v = f64x4::splat(sin_t);
            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let paired_j = j + step;

                    let a_re = f64x4::from(&self.real[j..j + 4]);
                    let a_im = f64x4::from(&self.imag[j..j + 4]);
                    let b_re = f64x4::from(&self.real[paired_j..paired_j + 4]);
                    let b_im = f64x4::from(&self.imag[paired_j..paired_j + 4]);

                    let new_a_re: [f64; 4] = (cos_v * a_re - sin_v * b_re).into();
                    let new_a_im: [f64; 4] = (cos_v * a_im - sin_v * b_im).into();
                    let new_b_re: [f64; 4] = (sin_v * a_re + cos_v * b_re).into();
                    let new_b_im: [f64; 4] = (sin_v * a_im + cos_v * b_im).into();

                    self.real[j..j + 4].copy_from_slice(&new_a_re);
                    self.imag[j..j + 4].copy_from_slice(&new_a_im);
                    self.real[paired_j..paired_j + 4].copy_from_slice(&new_b_re);
                    self.imag[paired_j..paired_j + 4].copy_from_slice(&new_b_im);

                    j += 4;
                }
            }
        } else {
            // Scalar fallback
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;

                    let a_re = self.real[j];
                    let a_im = self.imag[j];
                    let b_re = self.real[paired_j];
                    let b_im = self.imag[paired_j];

                    self.real[j] = cos_t * a_re - sin_t * b_re;
                    self.imag[j] = cos_t * a_im - sin_t * b_im;
                    self.real[paired_j] = sin_t * a_re + cos_t * b_re;
                    self.imag[paired_j] = sin_t * a_im + cos_t * b_im;
                }
            }
        }
    }

    /// Returns a reference to the real components.
    #[inline]
    #[must_use]
    pub fn real(&mut self) -> &[f64] {
        self.flush();
        &self.real
    }

    /// Returns a reference to the imaginary components.
    #[inline]
    #[must_use]
    pub fn imag(&mut self) -> &[f64] {
        self.flush();
        &self.imag
    }

    /// Prepare a specific computational basis state |n⟩.
    #[inline]
    pub fn prepare_computational_basis(&mut self, basis_state: usize) -> &mut Self {
        // Clear pending gates (state is being reset anyway)
        for pg in &mut self.pending_gates {
            *pg = None;
        }
        for r in &mut self.real {
            *r = 0.0;
        }
        for i in &mut self.imag {
            *i = 0.0;
        }
        self.real[basis_state] = 1.0;
        self
    }

    /// Returns the number of qubits in the state vector.
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the probability of measuring a specific computational basis state.
    ///
    /// The probability is calculated as |amplitude|^2 for the given basis state.
    /// This method auto-flushes any pending gates before returning the probability.
    #[inline]
    #[must_use]
    pub fn probability(&mut self, basis_state: usize) -> f64 {
        self.flush();
        let re = self.real[basis_state];
        let im = self.imag[basis_state];
        re * re + im * im
    }

    /// Returns the amplitude at the given basis state index as a Complex64.
    /// This method auto-flushes any pending gates before returning the amplitude.
    #[inline]
    #[must_use]
    pub fn get_amplitude(&mut self, index: usize) -> Complex64 {
        self.flush();
        Complex64::new(self.real[index], self.imag[index])
    }

    /// Sets the amplitude at the given basis state index.
    /// This method auto-flushes any pending gates before setting the amplitude.
    #[inline]
    pub fn set_amplitude(&mut self, index: usize, value: Complex64) {
        self.flush();
        self.real[index] = value.re;
        self.imag[index] = value.im;
    }

    /// Returns the state vector as a Vec of Complex64 for inspection.
    ///
    /// This creates a new vector by combining the real and imaginary arrays.
    /// This method auto-flushes any pending gates before returning the state.
    #[must_use]
    pub fn to_complex_vec(&mut self) -> Vec<Complex64> {
        self.flush();
        self.real
            .iter()
            .zip(&self.imag)
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect()
    }

    /// Creates a state vector from a Vec of Complex64.
    ///
    /// The length of the state vector must be a power of 2.
    ///
    /// # Panics
    /// Panics if `state.len()` is not a power of 2.
    #[must_use]
    pub fn from_complex_state(state: &[Complex64], rng: R) -> Self {
        let num_qubits = state.len().trailing_zeros() as usize;
        let size = state.len();
        assert_eq!(1 << num_qubits, size, "Invalid state vector size");

        let real: Vec<f64> = state.iter().map(|c| c.re).collect();
        let imag: Vec<f64> = state.iter().map(|c| c.im).collect();

        Self {
            real,
            imag,
            num_qubits,
            rng,
            scratch_real: Vec::new(),
            scratch_imag: Vec::new(),
            pending_gates: vec![None; num_qubits],
            fusion_enabled: true,
            parallel_enabled: false,
            num_threads: None,
        }
    }

    /// Creates a state vector from a Vec of Complex64.
    ///
    /// Alias for `from_complex_state` for API compatibility.
    #[must_use]
    pub fn from_state(state: &[Complex64], rng: R) -> Self {
        Self::from_complex_state(state, rng)
    }

    /// Returns the state vector as a Vec of Complex64.
    ///
    /// This creates a new vector by combining the real and imaginary arrays.
    /// This method auto-flushes any pending gates before returning the state.
    #[must_use]
    pub fn state(&mut self) -> Vec<Complex64> {
        self.flush();
        self.real
            .iter()
            .zip(self.imag.iter())
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect()
    }

    /// Returns the state vector as a Vec of Complex64 without flushing pending gates.
    ///
    /// WARNING: If gate fusion is enabled and gates are pending, this returns stale data.
    /// Use `state()` instead unless you're sure no gates are pending (e.g., after
    /// construction, reset, or with fusion disabled via `set_fusion(false)`).
    #[must_use]
    pub fn state_no_flush(&self) -> Vec<Complex64> {
        self.real
            .iter()
            .zip(self.imag.iter())
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect()
    }

    /// Prepare all qubits in the |+⟩ state, creating an equal superposition of all basis states.
    ///
    /// This operation prepares the state (1/√2^n)(|0...0⟩ + |0...1⟩ + ... + |1...1⟩)
    /// where n is the number of qubits.
    #[inline]
    pub fn prepare_plus_state(&mut self) -> &mut Self {
        let factor = 1.0 / f64::from(1 << self.num_qubits).sqrt();
        self.real.fill(factor);
        self.imag.fill(0.0);
        self
    }

    /// Apply a general single-qubit unitary gate given by a 2x2 complex matrix.
    ///
    /// The matrix elements are:
    /// ```text
    /// U = [[u00, u01],
    ///      [u10, u11]]
    /// ```
    ///
    /// # Example
    /// ```
    /// use pecos_simulators::StateVecSoA;
    /// use num_complex::Complex64;
    /// use std::f64::consts::FRAC_1_SQRT_2;
    ///
    /// let mut sim = StateVecSoA::new(1);
    /// // Apply Hadamard gate
    /// sim.single_qubit_unitary(0,
    ///     Complex64::new(FRAC_1_SQRT_2, 0.0),  // u00
    ///     Complex64::new(FRAC_1_SQRT_2, 0.0),  // u01
    ///     Complex64::new(FRAC_1_SQRT_2, 0.0),  // u10
    ///     Complex64::new(-FRAC_1_SQRT_2, 0.0), // u11
    /// );
    /// ```
    #[inline]
    pub fn single_qubit_unitary(
        &mut self,
        qubit: usize,
        u00: Complex64,
        u01: Complex64,
        u10: Complex64,
        u11: Complex64,
    ) -> &mut Self {
        let step = 1 << qubit;
        for i in (0..self.real.len()).step_by(2 * step) {
            for offset in 0..step {
                let j = i + offset;
                let k = j ^ step;

                let a_re = self.real[j];
                let a_im = self.imag[j];
                let b_re = self.real[k];
                let b_im = self.imag[k];

                // new_j = u00 * a + u01 * b
                self.real[j] = u00.re * a_re - u00.im * a_im + u01.re * b_re - u01.im * b_im;
                self.imag[j] = u00.re * a_im + u00.im * a_re + u01.re * b_im + u01.im * b_re;

                // new_k = u10 * a + u11 * b
                self.real[k] = u10.re * a_re - u10.im * a_im + u11.re * b_re - u11.im * b_im;
                self.imag[k] = u10.re * a_im + u10.im * a_re + u11.re * b_im + u11.im * b_re;
            }
        }
        self
    }

    /// Apply a general two-qubit unitary gate given by a 4x4 complex matrix.
    ///
    /// The matrix is indexed as:
    /// ```text
    /// U = [[u[0][0], u[0][1], u[0][2], u[0][3]],
    ///      [u[1][0], u[1][1], u[1][2], u[1][3]],
    ///      [u[2][0], u[2][1], u[2][2], u[2][3]],
    ///      [u[3][0], u[3][1], u[3][2], u[3][3]]]
    /// ```
    ///
    /// where rows/columns correspond to basis states |00⟩, |01⟩, |10⟩, |11⟩.
    #[inline]
    pub fn two_qubit_unitary(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        matrix: [[Complex64; 4]; 4],
    ) -> &mut Self {
        // Flush pending gates for both qubits
        self.flush_two_qubit(qubit1, qubit2);

        let size = self.real.len();

        // Lazily allocate scratch buffers on first use
        if self.scratch_real.len() < size {
            self.scratch_real = vec![0.0; size];
            self.scratch_imag = vec![0.0; size];
        }

        // Ensure consistent ordering for strided iteration
        let (lo, hi) = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };
        let step_lo = 1 << lo;
        let step_hi = 1 << hi;

        // The matrix is indexed as matrix[output_basis][input_basis]
        // where basis_idx = (qubit1_bit << 1) | qubit2_bit
        //
        // Our iteration uses (lo_bit, hi_bit) ordering:
        // - idx 0: lo=0, hi=0
        // - idx 1: lo=1, hi=0
        // - idx 2: lo=0, hi=1
        // - idx 3: lo=1, hi=1
        //
        // When qubit1 < qubit2 (qubit1 is lo, qubit2 is hi):
        //   lo_bit = qubit1_bit, hi_bit = qubit2_bit
        //   our_idx -> basis_idx: 0->0, 1->2, 2->1, 3->3
        //
        // When qubit1 > qubit2 (qubit2 is lo, qubit1 is hi):
        //   lo_bit = qubit2_bit, hi_bit = qubit1_bit
        //   our_idx -> basis_idx: 0->0, 1->1, 2->2, 3->3 (identity)

        // Permutation from our iteration order to matrix basis order
        let perm: [usize; 4] = if qubit1 < qubit2 {
            [0, 2, 1, 3] // swap indices 1 and 2
        } else {
            [0, 1, 2, 3] // identity
        };

        // Process groups of 4 basis states that share the same "frame" bits
        for outer in (0..size).step_by(step_hi * 2) {
            for mid in (0..step_hi).step_by(step_lo * 2) {
                for inner_idx in 0..step_lo {
                    let base = outer + mid + inner_idx;

                    // The 4 indices in (lo_bit, hi_bit) order
                    let indices = [
                        base,                     // lo=0, hi=0
                        base + step_lo,           // lo=1, hi=0
                        base + step_hi,           // lo=0, hi=1
                        base + step_hi + step_lo, // lo=1, hi=1
                    ];

                    // Load the 4 amplitudes in matrix basis order
                    let a = [
                        (self.real[indices[perm[0]]], self.imag[indices[perm[0]]]),
                        (self.real[indices[perm[1]]], self.imag[indices[perm[1]]]),
                        (self.real[indices[perm[2]]], self.imag[indices[perm[2]]]),
                        (self.real[indices[perm[3]]], self.imag[indices[perm[3]]]),
                    ];

                    // Apply the 4x4 matrix: new[j] = sum_k matrix[j][k] * old[k]
                    for (j, row) in matrix.iter().enumerate() {
                        let mut new_re = 0.0;
                        let mut new_im = 0.0;
                        for (k, &(amp_re, amp_im)) in a.iter().enumerate() {
                            let m = row[k];
                            new_re += m.re * amp_re - m.im * amp_im;
                            new_im += m.re * amp_im + m.im * amp_re;
                        }
                        // Write to the correct index using inverse permutation
                        self.scratch_real[indices[perm[j]]] = new_re;
                        self.scratch_imag[indices[perm[j]]] = new_im;
                    }
                }
            }
        }

        // Swap buffers (avoids copying)
        std::mem::swap(&mut self.real, &mut self.scratch_real);
        std::mem::swap(&mut self.imag, &mut self.scratch_imag);
        self
    }

    /// Joint measurement of ALL qubits via CDF sampling.
    ///
    /// Instead of 2n passes (probability + collapse per qubit), this does:
    /// 1. One pass to build CDF, sample outcome, and compute marginal probabilities
    /// 2. One pass to collapse to the sampled basis state
    #[allow(clippy::cast_precision_loss)] // bit extraction (0 or 1) as f64
    fn mz_joint_all(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let n = self.real.len();
        let num_qubits = self.num_qubits;

        // Pass 1: SIMD probability computation + CDF sampling + marginal probs.
        //
        // For chunks of 4 consecutive amplitudes (i, i+1, i+2, i+3):
        // - Qubit 0 bits: always [0, 1, 0, 1] (constant SIMD mask)
        // - Qubit 1 bits: always [0, 0, 1, 1] (constant SIMD mask)
        // - Qubit q>=2 bits: all same value (i >> q) & 1 (use scalar chunk_sum)
        let r: f64 = rand::RngExt::random(&mut self.rng);
        let mut cumsum = 0.0f64;
        let mut sampled_idx = n - 1;
        let mut marginal_probs = vec![0.0f64; num_qubits];

        // SIMD accumulators for qubits 0 and 1
        let q0_mask = f64x4::from([0.0, 1.0, 0.0, 1.0]);
        let q1_mask = f64x4::from([0.0, 0.0, 1.0, 1.0]);
        let mut q0_acc = f64x4::ZERO;
        let mut q1_acc = f64x4::ZERO;

        let mut i = 0;
        while i + 4 <= n {
            let re = f64x4::from(&self.real[i..i + 4]);
            let im = f64x4::from(&self.imag[i..i + 4]);
            let probs = re * re + im * im;

            // Qubits 0, 1: constant SIMD masks
            q0_acc += probs * q0_mask;
            q1_acc += probs * q1_mask;

            // Qubits 2+: all 4 amplitudes share the same bit at position q,
            // so use the horizontal sum × scalar bit
            let vals: [f64; 4] = probs.into();
            let chunk_sum = vals[0] + vals[1] + vals[2] + vals[3];
            for (q, mp) in marginal_probs.iter_mut().enumerate().skip(2) {
                *mp += chunk_sum * (((i >> q) & 1) as f64);
            }

            // CDF sampling
            cumsum += chunk_sum;
            if sampled_idx == n - 1 && cumsum >= r {
                // Threshold crossed in this chunk — find exact amplitude
                cumsum -= chunk_sum;
                for (j, &p) in vals.iter().enumerate() {
                    cumsum += p;
                    if cumsum >= r {
                        sampled_idx = i + j;
                        break;
                    }
                }
            }

            i += 4;
        }

        // Remainder (n not divisible by 4 — rare for power-of-two state vectors)
        while i < n {
            let prob = self.real[i] * self.real[i] + self.imag[i] * self.imag[i];
            for (q, mp) in marginal_probs.iter_mut().enumerate() {
                *mp += prob * (((i >> q) & 1) as f64);
            }
            cumsum += prob;
            if sampled_idx == n - 1 && cumsum >= r {
                sampled_idx = i;
            }
            i += 1;
        }

        // Reduce SIMD accumulators
        let v0: [f64; 4] = q0_acc.into();
        marginal_probs[0] = v0[0] + v0[1] + v0[2] + v0[3];
        if num_qubits > 1 {
            let v1: [f64; 4] = q1_acc.into();
            marginal_probs[1] = v1[0] + v1[1] + v1[2] + v1[3];
        }

        // Build results in caller's qubit order
        let mut results = Vec::with_capacity(qubits.len());
        for &q in qubits {
            let outcome = (sampled_idx >> q.index()) & 1 == 1;
            let prob_one = marginal_probs[q.index()];
            let is_deterministic = !(1e-10..=1.0 - 1e-10).contains(&prob_one);
            results.push(MeasurementResult {
                outcome,
                is_deterministic,
            });
        }

        // Pass 2: collapse to |sampled_idx⟩ using fill + set (avoids per-element branch)
        let norm = (self.real[sampled_idx] * self.real[sampled_idx]
            + self.imag[sampled_idx] * self.imag[sampled_idx])
            .sqrt();
        let final_re = self.real[sampled_idx] / norm;
        let final_im = self.imag[sampled_idx] / norm;
        self.real.fill(0.0);
        self.imag.fill(0.0);
        self.real[sampled_idx] = final_re;
        self.imag[sampled_idx] = final_im;

        results
    }

    /// Joint measurement of a SUBSET of qubits (4 <= k <= 20) via probability table.
    ///
    /// Builds a 2^k probability table in one pass over the state vector, samples
    /// a joint outcome, then collapses the state in one pass using bitmask matching.
    #[allow(clippy::cast_precision_loss)] // bit extraction (0 or 1) as f64
    fn mz_joint_subset(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let k = qubits.len();
        let n = self.real.len();

        let qubit_indices: Vec<usize> = qubits.iter().map(pecos_core::QubitId::index).collect();
        let table_size = 1usize << k;
        let mut prob_table = vec![0.0f64; table_size];
        let mut marginal_probs = vec![0.0f64; k];

        // Classify measured qubits by position for SIMD optimization.
        // For chunks of 4 consecutive amplitudes: bits at positions 0,1 vary
        // within the chunk; bits at positions >= 2 are constant.
        let q0_mask = f64x4::from([0.0, 1.0, 0.0, 1.0]);
        let q1_mask = f64x4::from([0.0, 0.0, 1.0, 1.0]);
        let mut acc_q0 = f64x4::ZERO;
        let mut acc_q1 = f64x4::ZERO;
        let mut q0_j: Option<usize> = None; // j-index for measured qubit at position 0
        let mut q1_j: Option<usize> = None;
        let mut high_qubits: Vec<(usize, usize)> = Vec::new();

        for (j, &q_idx) in qubit_indices.iter().enumerate() {
            match q_idx {
                0 => q0_j = Some(j),
                1 => q1_j = Some(j),
                _ => high_qubits.push((j, q_idx)),
            }
        }

        let q0_bit = q0_j.map_or(0, |j| 1usize << j);
        let q1_bit = q1_j.map_or(0, |j| 1usize << j);

        // Pass 1: SIMD probability + table accumulation + marginal probs
        let mut i = 0;
        while i + 4 <= n {
            let re = f64x4::from(&self.real[i..i + 4]);
            let im = f64x4::from(&self.imag[i..i + 4]);
            let probs = re * re + im * im;

            // SIMD marginal probs for qubits at positions 0 and 1
            if q0_j.is_some() {
                acc_q0 += probs * q0_mask;
            }
            if q1_j.is_some() {
                acc_q1 += probs * q1_mask;
            }

            let vals: [f64; 4] = probs.into();
            let chunk_sum = vals[0] + vals[1] + vals[2] + vals[3];

            // Scalar marginal probs for high qubits (same bit for all 4)
            for &(j, q_idx) in &high_qubits {
                marginal_probs[j] += chunk_sum * (((i >> q_idx) & 1) as f64);
            }

            // Table accumulation: base pattern from high qubits is shared
            let mut base = 0usize;
            for &(j, q_idx) in &high_qubits {
                base |= ((i >> q_idx) & 1) << j;
            }

            match (q0_j.is_some(), q1_j.is_some()) {
                (false, false) => {
                    prob_table[base] += chunk_sum;
                }
                (true, false) => {
                    prob_table[base] += vals[0] + vals[2];
                    prob_table[base | q0_bit] += vals[1] + vals[3];
                }
                (false, true) => {
                    prob_table[base] += vals[0] + vals[1];
                    prob_table[base | q1_bit] += vals[2] + vals[3];
                }
                (true, true) => {
                    prob_table[base] += vals[0];
                    prob_table[base | q0_bit] += vals[1];
                    prob_table[base | q1_bit] += vals[2];
                    prob_table[base | q0_bit | q1_bit] += vals[3];
                }
            }

            i += 4;
        }

        // Remainder
        while i < n {
            let prob = self.real[i] * self.real[i] + self.imag[i] * self.imag[i];
            let mut pattern = 0usize;
            for (j, &q_idx) in qubit_indices.iter().enumerate() {
                let bit = (i >> q_idx) & 1;
                pattern |= bit << j;
                marginal_probs[j] += prob * (bit as f64);
            }
            prob_table[pattern] += prob;
            i += 1;
        }

        // Reduce SIMD accumulators
        if let Some(j) = q0_j {
            let v: [f64; 4] = acc_q0.into();
            marginal_probs[j] = v[0] + v[1] + v[2] + v[3];
        }
        if let Some(j) = q1_j {
            let v: [f64; 4] = acc_q1.into();
            marginal_probs[j] = v[0] + v[1] + v[2] + v[3];
        }

        // Sample from probability table
        let r: f64 = rand::RngExt::random(&mut self.rng);
        let mut cumsum = 0.0;
        let mut sampled_pattern = table_size - 1;
        for (pat, &prob) in prob_table.iter().enumerate() {
            cumsum += prob;
            if cumsum >= r {
                sampled_pattern = pat;
                break;
            }
        }

        // Build results
        let mut results = Vec::with_capacity(k);
        for (j, &prob_one) in marginal_probs.iter().enumerate().take(k) {
            let outcome = (sampled_pattern >> j) & 1 == 1;
            let is_deterministic = !(1e-10..=1.0 - 1e-10).contains(&prob_one);
            results.push(MeasurementResult {
                outcome,
                is_deterministic,
            });
        }

        // Pass 2: SIMD collapse using precomputed bitmask factors.
        // For 4 consecutive indices, bits at positions >= 2 are constant,
        // so we precompute a SIMD mask for the varying low bits.
        let mut measured_mask = 0usize;
        let mut expected_bits = 0usize;
        for (j, &q_idx) in qubit_indices.iter().enumerate() {
            measured_mask |= 1 << q_idx;
            if (sampled_pattern >> j) & 1 == 1 {
                expected_bits |= 1 << q_idx;
            }
        }

        let pattern_prob = prob_table[sampled_pattern];
        let norm_factor = 1.0 / pattern_prob.sqrt();

        // Precompute per-element factors for the 4 varying low-bit combinations
        let high_mask = measured_mask & !3usize;
        let high_expected = expected_bits & !3usize;
        let mut d_factors = [0.0f64; 4];
        for (d, factor) in d_factors.iter_mut().enumerate() {
            let low_match = (d & measured_mask & 3) == (expected_bits & 3);
            *factor = if low_match { 1.0 } else { 0.0 };
        }
        let d_fv = f64x4::from(d_factors);

        let mut ci = 0;
        while ci + 4 <= n {
            let base_match = (ci & high_mask) == high_expected;
            let factor = if base_match { norm_factor } else { 0.0 };
            let fv = f64x4::splat(factor) * d_fv;

            let re = f64x4::from(&self.real[ci..ci + 4]);
            let im = f64x4::from(&self.imag[ci..ci + 4]);
            let nr: [f64; 4] = (re * fv).into();
            let ni: [f64; 4] = (im * fv).into();
            self.real[ci..ci + 4].copy_from_slice(&nr);
            self.imag[ci..ci + 4].copy_from_slice(&ni);

            ci += 4;
        }
        while ci < n {
            if (ci & measured_mask) == expected_bits {
                self.real[ci] *= norm_factor;
                self.imag[ci] *= norm_factor;
            } else {
                self.real[ci] = 0.0;
                self.imag[ci] = 0.0;
            }
            ci += 1;
        }

        results
    }

    /// Internal helper for computing probability of |1⟩.
    #[inline]
    fn probability_one(&self, qubit: usize) -> f64 {
        let step = 1 << qubit;

        // For small step sizes, use scalar to avoid SIMD overhead
        if step < 4 {
            let mut prob = 0.0;
            for i in (0..self.real.len()).step_by(step * 2) {
                for j in (i + step)..(i + 2 * step) {
                    prob += self.real[j] * self.real[j] + self.imag[j] * self.imag[j];
                }
            }
            return prob;
        }

        // SIMD accumulator
        let mut acc = f64x4::ZERO;

        for i in (0..self.real.len()).step_by(step * 2) {
            let mut j = i + step;
            // Process 4 elements at a time
            while j + 4 <= i + 2 * step {
                let re = f64x4::from(&self.real[j..j + 4]);
                let im = f64x4::from(&self.imag[j..j + 4]);
                acc += re * re + im * im;
                j += 4;
            }
            // Handle remainder (step is power of 2 and >= 4, so remainder is 0)
        }

        // Horizontal sum
        let vals: [f64; 4] = acc.into();
        vals[0] + vals[1] + vals[2] + vals[3]
    }
}

impl<R> QuantumSimulator for StateVecSoA<R>
where
    R: Rng,
{
    fn reset(&mut self) -> &mut Self {
        // Clear pending gates (state is being reset anyway)
        for pg in &mut self.pending_gates {
            *pg = None;
        }
        for r in &mut self.real {
            *r = 0.0;
        }
        for i in &mut self.imag {
            *i = 0.0;
        }
        self.real[0] = 1.0;
        self
    }
}

impl<R> CliffordGateable for StateVecSoA<R>
where
    R: Rng,
{
    #[inline]
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::H);
            }
        } else {
            for &q in qubits {
                self.apply_h_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn h2(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::H2);
            }
        } else {
            for &q in qubits {
                self.apply_general_gate(q.index(), &gate_matrices::H2);
            }
        }
        self
    }

    #[inline]
    fn h3(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::H3);
            }
        } else {
            for &q in qubits {
                self.apply_h3_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn h4(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::H4);
            }
        } else {
            for &q in qubits {
                self.apply_h4_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn h5(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::H5);
            }
        } else {
            for &q in qubits {
                self.apply_general_gate(q.index(), &gate_matrices::H5);
            }
        }
        self
    }

    #[inline]
    fn h6(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::H6);
            }
        } else {
            for &q in qubits {
                self.apply_general_gate(q.index(), &gate_matrices::H6);
            }
        }
        self
    }

    #[inline]
    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::X);
            }
        } else {
            for &q in qubits {
                self.apply_x_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::Y);
            }
        } else {
            for &q in qubits {
                self.apply_y_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::Z);
            }
        } else {
            for &q in qubits {
                self.apply_z_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::SZ);
            }
        } else {
            for &q in qubits {
                self.apply_sz_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::SZDG);
            }
        } else {
            for &q in qubits {
                self.apply_szdg_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::SX);
            }
        } else {
            for &q in qubits {
                self.apply_sx_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::SXDG);
            }
        } else {
            for &q in qubits {
                self.apply_sxdg_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::SY);
            }
        } else {
            for &q in qubits {
                self.apply_sy_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::SYDG);
            }
        } else {
            for &q in qubits {
                self.apply_sydg_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn f(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::F);
            }
        } else {
            for &q in qubits {
                self.apply_general_gate(q.index(), &gate_matrices::F);
            }
        }
        self
    }

    #[inline]
    fn fdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::FDG);
            }
        } else {
            for &q in qubits {
                self.apply_general_gate(q.index(), &gate_matrices::FDG);
            }
        }
        self
    }

    #[inline]
    fn f2(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::F2);
            }
        } else {
            for &q in qubits {
                self.apply_general_gate(q.index(), &gate_matrices::F2);
            }
        }
        self
    }

    #[inline]
    fn f2dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::F2DG);
            }
        } else {
            for &q in qubits {
                self.apply_general_gate(q.index(), &gate_matrices::F2DG);
            }
        }
        self
    }

    #[inline]
    fn f3(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::F3);
            }
        } else {
            for &q in qubits {
                self.apply_general_gate(q.index(), &gate_matrices::F3);
            }
        }
        self
    }

    #[inline]
    fn f3dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::F3DG);
            }
        } else {
            for &q in qubits {
                self.apply_general_gate(q.index(), &gate_matrices::F3DG);
            }
        }
        self
    }

    #[inline]
    fn f4(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::F4);
            }
        } else {
            for &q in qubits {
                self.apply_general_gate(q.index(), &gate_matrices::F4);
            }
        }
        self
    }

    #[inline]
    fn f4dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        if self.fusion_enabled {
            for &q in qubits {
                self.queue_gate(q.index(), &gate_matrices::F4DG);
            }
        } else {
            for &q in qubits {
                self.apply_general_gate(q.index(), &gate_matrices::F4DG);
            }
        }
        self
    }

    #[inline]
    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            let control = q0.index();
            let target = q1.index();

            self.flush_two_qubit(control, target);

            let n = self.real.len();
            let (q_lo, q_hi) = if control < target {
                (control, target)
            } else {
                (target, control)
            };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let control_mask = 1 << control;
            let target_mask = 1 << target;

            // When q_lo >= 2, indices are contiguous and we can use SIMD
            if step_lo >= 4 {
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let base = i_lo + offset;
                            let idx0 = base | control_mask;
                            let idx1 = idx0 | target_mask;

                            // Load both sets
                            let re0 = f64x4::from(&self.real[idx0..idx0 + 4]);
                            let im0 = f64x4::from(&self.imag[idx0..idx0 + 4]);
                            let re1 = f64x4::from(&self.real[idx1..idx1 + 4]);
                            let im1 = f64x4::from(&self.imag[idx1..idx1 + 4]);

                            // Swap by storing in opposite locations
                            let arr_re0: [f64; 4] = re1.into();
                            let arr_im0: [f64; 4] = im1.into();
                            let arr_re1: [f64; 4] = re0.into();
                            let arr_im1: [f64; 4] = im0.into();

                            self.real[idx0..idx0 + 4].copy_from_slice(&arr_re0);
                            self.imag[idx0..idx0 + 4].copy_from_slice(&arr_im0);
                            self.real[idx1..idx1 + 4].copy_from_slice(&arr_re1);
                            self.imag[idx1..idx1 + 4].copy_from_slice(&arr_im1);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step_lo
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_c1_t0 = base | control_mask;
                            let idx_c1_t1 = idx_c1_t0 | target_mask;

                            self.real.swap(idx_c1_t0, idx_c1_t1);
                            self.imag.swap(idx_c1_t0, idx_c1_t1);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            self.flush_two_qubit(q1, q2);

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask_11 = (1 << q1) | (1 << q2);

            // When q_lo >= 2, indices are contiguous and we can use SIMD
            if step_lo >= 4 {
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let base = i_lo + offset;
                            let idx = base | mask_11;
                            let re = f64x4::from(&self.real[idx..idx + 4]);
                            let im = f64x4::from(&self.imag[idx..idx + 4]);
                            let neg_re: [f64; 4] = (-re).into();
                            let neg_im: [f64; 4] = (-im).into();
                            self.real[idx..idx + 4].copy_from_slice(&neg_re);
                            self.imag[idx..idx + 4].copy_from_slice(&neg_im);
                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step_lo
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_11 = base | mask_11;
                            self.real[idx_11] = -self.real[idx_11];
                            self.imag[idx_11] = -self.imag[idx_11];
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn swap(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            self.flush_two_qubit(q1, q2);

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask_01 = 1 << q2;
            let mask_10 = 1 << q1;

            // When q_lo >= 2, indices are contiguous and we can use SIMD
            if step_lo >= 4 {
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let base = i_lo + offset;
                            let idx_01 = base | mask_01;
                            let idx_10 = base | mask_10;

                            let re01 = f64x4::from(&self.real[idx_01..idx_01 + 4]);
                            let im01 = f64x4::from(&self.imag[idx_01..idx_01 + 4]);
                            let re10 = f64x4::from(&self.real[idx_10..idx_10 + 4]);
                            let im10 = f64x4::from(&self.imag[idx_10..idx_10 + 4]);

                            let arr_re01: [f64; 4] = re10.into();
                            let arr_im01: [f64; 4] = im10.into();
                            let arr_re10: [f64; 4] = re01.into();
                            let arr_im10: [f64; 4] = im01.into();

                            self.real[idx_01..idx_01 + 4].copy_from_slice(&arr_re01);
                            self.imag[idx_01..idx_01 + 4].copy_from_slice(&arr_im01);
                            self.real[idx_10..idx_10 + 4].copy_from_slice(&arr_re10);
                            self.imag[idx_10..idx_10 + 4].copy_from_slice(&arr_im10);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_01 = base | mask_01;
                            let idx_10 = base | mask_10;
                            self.real.swap(idx_01, idx_10);
                            self.imag.swap(idx_01, idx_10);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn cy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            let control = q0.index();
            let target = q1.index();

            self.flush_two_qubit(control, target);

            let n = self.real.len();
            let (q_lo, q_hi) = if control < target {
                (control, target)
            } else {
                (target, control)
            };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let control_mask = 1 << control;
            let target_mask = 1 << target;

            // CY = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ Y
            // When control=1: apply Y to target
            // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩

            if step_lo >= 4 {
                // SIMD version: process 4 elements at a time
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        // Check if target bit is set in i_lo - if so, skip this entire block
                        let test_idx = i_lo | control_mask;
                        if (test_idx & target_mask) != 0 {
                            continue;
                        }

                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let base = i_lo + offset;
                            let idx_c1_t0 = base | control_mask;
                            let idx_c1_t1 = idx_c1_t0 | target_mask;

                            let re_t0 = f64x4::from(&self.real[idx_c1_t0..idx_c1_t0 + 4]);
                            let im_t0 = f64x4::from(&self.imag[idx_c1_t0..idx_c1_t0 + 4]);
                            let re_t1 = f64x4::from(&self.real[idx_c1_t1..idx_c1_t1 + 4]);
                            let im_t1 = f64x4::from(&self.imag[idx_c1_t1..idx_c1_t1 + 4]);

                            // new |t0⟩ = -i * old |t1⟩: -i * (re, im) = (im, -re)
                            let new_re_t0: [f64; 4] = im_t1.into();
                            let new_im_t0: [f64; 4] = (-re_t1).into();

                            // new |t1⟩ = i * old |t0⟩: i * (re, im) = (-im, re)
                            let new_re_t1: [f64; 4] = (-im_t0).into();
                            let new_im_t1: [f64; 4] = re_t0.into();

                            self.real[idx_c1_t0..idx_c1_t0 + 4].copy_from_slice(&new_re_t0);
                            self.imag[idx_c1_t0..idx_c1_t0 + 4].copy_from_slice(&new_im_t0);
                            self.real[idx_c1_t1..idx_c1_t1 + 4].copy_from_slice(&new_re_t1);
                            self.imag[idx_c1_t1..idx_c1_t1 + 4].copy_from_slice(&new_im_t1);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_c1_t0 = base | control_mask;

                            // Skip if target bit already set (we handle pairs)
                            if (idx_c1_t0 & target_mask) != 0 {
                                continue;
                            }

                            let idx_c1_t1 = idx_c1_t0 | target_mask;

                            let re_t0 = self.real[idx_c1_t0];
                            let im_t0 = self.imag[idx_c1_t0];
                            let re_t1 = self.real[idx_c1_t1];
                            let im_t1 = self.imag[idx_c1_t1];

                            // new |t0⟩ = -i * old |t1⟩
                            self.real[idx_c1_t0] = im_t1;
                            self.imag[idx_c1_t0] = -re_t1;

                            // new |t1⟩ = i * old |t0⟩
                            self.real[idx_c1_t1] = -im_t0;
                            self.imag[idx_c1_t1] = re_t0;
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn sxx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        const K: f64 = std::f64::consts::FRAC_1_SQRT_2;

        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            self.flush_two_qubit(q1, q2);

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask1 = 1 << q1;
            let mask2 = 1 << q2;

            // SXX = exp(-i * π/4 * X⊗X) = (1/√2)(I - i*X⊗X)
            // Matrix: (1/√2) * [[1, 0, 0, -i], [0, 1, -i, 0], [0, -i, 1, 0], [-i, 0, 0, 1]]

            if step_lo >= 4 {
                // SIMD version
                let k_v = f64x4::splat(K);
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let idx_00 = i_lo + offset;
                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let re_00 = f64x4::from(&self.real[idx_00..idx_00 + 4]);
                            let im_00 = f64x4::from(&self.imag[idx_00..idx_00 + 4]);
                            let re_01 = f64x4::from(&self.real[idx_01..idx_01 + 4]);
                            let im_01 = f64x4::from(&self.imag[idx_01..idx_01 + 4]);
                            let re_10 = f64x4::from(&self.real[idx_10..idx_10 + 4]);
                            let im_10 = f64x4::from(&self.imag[idx_10..idx_10 + 4]);
                            let re_11 = f64x4::from(&self.real[idx_11..idx_11 + 4]);
                            let im_11 = f64x4::from(&self.imag[idx_11..idx_11 + 4]);

                            // new_00 = K * (|00⟩ - i*|11⟩)
                            let new_re_00: [f64; 4] = (k_v * (re_00 + im_11)).into();
                            let new_im_00: [f64; 4] = (k_v * (im_00 - re_11)).into();

                            // new_01 = K * (|01⟩ - i*|10⟩)
                            let new_re_01: [f64; 4] = (k_v * (re_01 + im_10)).into();
                            let new_im_01: [f64; 4] = (k_v * (im_01 - re_10)).into();

                            // new_10 = K * (|10⟩ - i*|01⟩)
                            let new_re_10: [f64; 4] = (k_v * (re_10 + im_01)).into();
                            let new_im_10: [f64; 4] = (k_v * (im_10 - re_01)).into();

                            // new_11 = K * (|11⟩ - i*|00⟩)
                            let new_re_11: [f64; 4] = (k_v * (re_11 + im_00)).into();
                            let new_im_11: [f64; 4] = (k_v * (im_11 - re_00)).into();

                            self.real[idx_00..idx_00 + 4].copy_from_slice(&new_re_00);
                            self.imag[idx_00..idx_00 + 4].copy_from_slice(&new_im_00);
                            self.real[idx_01..idx_01 + 4].copy_from_slice(&new_re_01);
                            self.imag[idx_01..idx_01 + 4].copy_from_slice(&new_im_01);
                            self.real[idx_10..idx_10 + 4].copy_from_slice(&new_re_10);
                            self.imag[idx_10..idx_10 + 4].copy_from_slice(&new_im_10);
                            self.real[idx_11..idx_11 + 4].copy_from_slice(&new_re_11);
                            self.imag[idx_11..idx_11 + 4].copy_from_slice(&new_im_11);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_00 = base & !(mask1 | mask2);

                            // Only process each quartet once
                            if base != idx_00 {
                                continue;
                            }

                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let (re_00, im_00) = (self.real[idx_00], self.imag[idx_00]);
                            let (re_01, im_01) = (self.real[idx_01], self.imag[idx_01]);
                            let (re_10, im_10) = (self.real[idx_10], self.imag[idx_10]);
                            let (re_11, im_11) = (self.real[idx_11], self.imag[idx_11]);

                            // new_00 = K * (|00⟩ - i*|11⟩)
                            self.real[idx_00] = K * (re_00 + im_11);
                            self.imag[idx_00] = K * (im_00 - re_11);

                            // new_01 = K * (|01⟩ - i*|10⟩)
                            self.real[idx_01] = K * (re_01 + im_10);
                            self.imag[idx_01] = K * (im_01 - re_10);

                            // new_10 = K * (|10⟩ - i*|01⟩)
                            self.real[idx_10] = K * (re_10 + im_01);
                            self.imag[idx_10] = K * (im_10 - re_01);

                            // new_11 = K * (|11⟩ - i*|00⟩)
                            self.real[idx_11] = K * (re_11 + im_00);
                            self.imag[idx_11] = K * (im_11 - re_00);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn sxxdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        const K: f64 = std::f64::consts::FRAC_1_SQRT_2;

        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            self.flush_two_qubit(q1, q2);

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask1 = 1 << q1;
            let mask2 = 1 << q2;

            // SXXDG = exp(+i * π/4 * X⊗X) = (1/√2)(I + i*X⊗X)
            // Matrix: (1/√2) * [[1, 0, 0, i], [0, 1, i, 0], [0, i, 1, 0], [i, 0, 0, 1]]

            if step_lo >= 4 {
                // SIMD version
                let k_v = f64x4::splat(K);
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let idx_00 = i_lo + offset;
                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let re_00 = f64x4::from(&self.real[idx_00..idx_00 + 4]);
                            let im_00 = f64x4::from(&self.imag[idx_00..idx_00 + 4]);
                            let re_01 = f64x4::from(&self.real[idx_01..idx_01 + 4]);
                            let im_01 = f64x4::from(&self.imag[idx_01..idx_01 + 4]);
                            let re_10 = f64x4::from(&self.real[idx_10..idx_10 + 4]);
                            let im_10 = f64x4::from(&self.imag[idx_10..idx_10 + 4]);
                            let re_11 = f64x4::from(&self.real[idx_11..idx_11 + 4]);
                            let im_11 = f64x4::from(&self.imag[idx_11..idx_11 + 4]);

                            // new_00 = K * (|00⟩ + i*|11⟩)
                            let new_re_00: [f64; 4] = (k_v * (re_00 - im_11)).into();
                            let new_im_00: [f64; 4] = (k_v * (im_00 + re_11)).into();

                            // new_01 = K * (|01⟩ + i*|10⟩)
                            let new_re_01: [f64; 4] = (k_v * (re_01 - im_10)).into();
                            let new_im_01: [f64; 4] = (k_v * (im_01 + re_10)).into();

                            // new_10 = K * (|10⟩ + i*|01⟩)
                            let new_re_10: [f64; 4] = (k_v * (re_10 - im_01)).into();
                            let new_im_10: [f64; 4] = (k_v * (im_10 + re_01)).into();

                            // new_11 = K * (|11⟩ + i*|00⟩)
                            let new_re_11: [f64; 4] = (k_v * (re_11 - im_00)).into();
                            let new_im_11: [f64; 4] = (k_v * (im_11 + re_00)).into();

                            self.real[idx_00..idx_00 + 4].copy_from_slice(&new_re_00);
                            self.imag[idx_00..idx_00 + 4].copy_from_slice(&new_im_00);
                            self.real[idx_01..idx_01 + 4].copy_from_slice(&new_re_01);
                            self.imag[idx_01..idx_01 + 4].copy_from_slice(&new_im_01);
                            self.real[idx_10..idx_10 + 4].copy_from_slice(&new_re_10);
                            self.imag[idx_10..idx_10 + 4].copy_from_slice(&new_im_10);
                            self.real[idx_11..idx_11 + 4].copy_from_slice(&new_re_11);
                            self.imag[idx_11..idx_11 + 4].copy_from_slice(&new_im_11);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_00 = base & !(mask1 | mask2);

                            if base != idx_00 {
                                continue;
                            }

                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let (re_00, im_00) = (self.real[idx_00], self.imag[idx_00]);
                            let (re_01, im_01) = (self.real[idx_01], self.imag[idx_01]);
                            let (re_10, im_10) = (self.real[idx_10], self.imag[idx_10]);
                            let (re_11, im_11) = (self.real[idx_11], self.imag[idx_11]);

                            // new_00 = K * (|00⟩ + i*|11⟩)
                            self.real[idx_00] = K * (re_00 - im_11);
                            self.imag[idx_00] = K * (im_00 + re_11);

                            // new_01 = K * (|01⟩ + i*|10⟩)
                            self.real[idx_01] = K * (re_01 - im_10);
                            self.imag[idx_01] = K * (im_01 + re_10);

                            // new_10 = K * (|10⟩ + i*|01⟩)
                            self.real[idx_10] = K * (re_10 - im_01);
                            self.imag[idx_10] = K * (im_10 + re_01);

                            // new_11 = K * (|11⟩ + i*|00⟩)
                            self.real[idx_11] = K * (re_11 - im_00);
                            self.imag[idx_11] = K * (im_11 + re_00);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn syy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        const K: f64 = std::f64::consts::FRAC_1_SQRT_2;

        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            self.flush_two_qubit(q1, q2);

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask1 = 1 << q1;
            let mask2 = 1 << q2;

            // SYY = exp(-i * π/4 * Y⊗Y) = (1/√2)(I - i*Y⊗Y)
            // Y⊗Y swaps |00⟩↔-|11⟩ and |01⟩↔|10⟩
            // Matrix: (1/√2) * [[1, 0, 0, i], [0, 1, -i, 0], [0, -i, 1, 0], [i, 0, 0, 1]]

            if step_lo >= 4 {
                // SIMD version
                let k_v = f64x4::splat(K);
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let idx_00 = i_lo + offset;
                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let re_00 = f64x4::from(&self.real[idx_00..idx_00 + 4]);
                            let im_00 = f64x4::from(&self.imag[idx_00..idx_00 + 4]);
                            let re_01 = f64x4::from(&self.real[idx_01..idx_01 + 4]);
                            let im_01 = f64x4::from(&self.imag[idx_01..idx_01 + 4]);
                            let re_10 = f64x4::from(&self.real[idx_10..idx_10 + 4]);
                            let im_10 = f64x4::from(&self.imag[idx_10..idx_10 + 4]);
                            let re_11 = f64x4::from(&self.real[idx_11..idx_11 + 4]);
                            let im_11 = f64x4::from(&self.imag[idx_11..idx_11 + 4]);

                            // new_00 = K * (|00⟩ + i*|11⟩)
                            let new_re_00: [f64; 4] = (k_v * (re_00 - im_11)).into();
                            let new_im_00: [f64; 4] = (k_v * (im_00 + re_11)).into();

                            // new_01 = K * (|01⟩ - i*|10⟩)
                            let new_re_01: [f64; 4] = (k_v * (re_01 + im_10)).into();
                            let new_im_01: [f64; 4] = (k_v * (im_01 - re_10)).into();

                            // new_10 = K * (|10⟩ - i*|01⟩)
                            let new_re_10: [f64; 4] = (k_v * (re_10 + im_01)).into();
                            let new_im_10: [f64; 4] = (k_v * (im_10 - re_01)).into();

                            // new_11 = K * (|11⟩ + i*|00⟩)
                            let new_re_11: [f64; 4] = (k_v * (re_11 - im_00)).into();
                            let new_im_11: [f64; 4] = (k_v * (im_11 + re_00)).into();

                            self.real[idx_00..idx_00 + 4].copy_from_slice(&new_re_00);
                            self.imag[idx_00..idx_00 + 4].copy_from_slice(&new_im_00);
                            self.real[idx_01..idx_01 + 4].copy_from_slice(&new_re_01);
                            self.imag[idx_01..idx_01 + 4].copy_from_slice(&new_im_01);
                            self.real[idx_10..idx_10 + 4].copy_from_slice(&new_re_10);
                            self.imag[idx_10..idx_10 + 4].copy_from_slice(&new_im_10);
                            self.real[idx_11..idx_11 + 4].copy_from_slice(&new_re_11);
                            self.imag[idx_11..idx_11 + 4].copy_from_slice(&new_im_11);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_00 = base & !(mask1 | mask2);

                            if base != idx_00 {
                                continue;
                            }

                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let (re_00, im_00) = (self.real[idx_00], self.imag[idx_00]);
                            let (re_01, im_01) = (self.real[idx_01], self.imag[idx_01]);
                            let (re_10, im_10) = (self.real[idx_10], self.imag[idx_10]);
                            let (re_11, im_11) = (self.real[idx_11], self.imag[idx_11]);

                            // new_00 = K * (|00⟩ + i*|11⟩)
                            self.real[idx_00] = K * (re_00 - im_11);
                            self.imag[idx_00] = K * (im_00 + re_11);

                            // new_01 = K * (|01⟩ - i*|10⟩)
                            self.real[idx_01] = K * (re_01 + im_10);
                            self.imag[idx_01] = K * (im_01 - re_10);

                            // new_10 = K * (|10⟩ - i*|01⟩)
                            self.real[idx_10] = K * (re_10 + im_01);
                            self.imag[idx_10] = K * (im_10 - re_01);

                            // new_11 = K * (|11⟩ + i*|00⟩)
                            self.real[idx_11] = K * (re_11 - im_00);
                            self.imag[idx_11] = K * (im_11 + re_00);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn syydg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        const K: f64 = std::f64::consts::FRAC_1_SQRT_2;

        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            self.flush_two_qubit(q1, q2);

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask1 = 1 << q1;
            let mask2 = 1 << q2;

            // SYYDG = exp(+i * π/4 * Y⊗Y) = (1/√2)(I + i*Y⊗Y)
            // Matrix: (1/√2) * [[1, 0, 0, -i], [0, 1, i, 0], [0, i, 1, 0], [-i, 0, 0, 1]]

            if step_lo >= 4 {
                // SIMD version
                let k_v = f64x4::splat(K);
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let idx_00 = i_lo + offset;
                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let re_00 = f64x4::from(&self.real[idx_00..idx_00 + 4]);
                            let im_00 = f64x4::from(&self.imag[idx_00..idx_00 + 4]);
                            let re_01 = f64x4::from(&self.real[idx_01..idx_01 + 4]);
                            let im_01 = f64x4::from(&self.imag[idx_01..idx_01 + 4]);
                            let re_10 = f64x4::from(&self.real[idx_10..idx_10 + 4]);
                            let im_10 = f64x4::from(&self.imag[idx_10..idx_10 + 4]);
                            let re_11 = f64x4::from(&self.real[idx_11..idx_11 + 4]);
                            let im_11 = f64x4::from(&self.imag[idx_11..idx_11 + 4]);

                            // new_00 = K * (|00⟩ - i*|11⟩)
                            let new_re_00: [f64; 4] = (k_v * (re_00 + im_11)).into();
                            let new_im_00: [f64; 4] = (k_v * (im_00 - re_11)).into();

                            // new_01 = K * (|01⟩ + i*|10⟩)
                            let new_re_01: [f64; 4] = (k_v * (re_01 - im_10)).into();
                            let new_im_01: [f64; 4] = (k_v * (im_01 + re_10)).into();

                            // new_10 = K * (|10⟩ + i*|01⟩)
                            let new_re_10: [f64; 4] = (k_v * (re_10 - im_01)).into();
                            let new_im_10: [f64; 4] = (k_v * (im_10 + re_01)).into();

                            // new_11 = K * (|11⟩ - i*|00⟩)
                            let new_re_11: [f64; 4] = (k_v * (re_11 + im_00)).into();
                            let new_im_11: [f64; 4] = (k_v * (im_11 - re_00)).into();

                            self.real[idx_00..idx_00 + 4].copy_from_slice(&new_re_00);
                            self.imag[idx_00..idx_00 + 4].copy_from_slice(&new_im_00);
                            self.real[idx_01..idx_01 + 4].copy_from_slice(&new_re_01);
                            self.imag[idx_01..idx_01 + 4].copy_from_slice(&new_im_01);
                            self.real[idx_10..idx_10 + 4].copy_from_slice(&new_re_10);
                            self.imag[idx_10..idx_10 + 4].copy_from_slice(&new_im_10);
                            self.real[idx_11..idx_11 + 4].copy_from_slice(&new_re_11);
                            self.imag[idx_11..idx_11 + 4].copy_from_slice(&new_im_11);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_00 = base & !(mask1 | mask2);

                            if base != idx_00 {
                                continue;
                            }

                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let (re_00, im_00) = (self.real[idx_00], self.imag[idx_00]);
                            let (re_01, im_01) = (self.real[idx_01], self.imag[idx_01]);
                            let (re_10, im_10) = (self.real[idx_10], self.imag[idx_10]);
                            let (re_11, im_11) = (self.real[idx_11], self.imag[idx_11]);

                            // new_00 = K * (|00⟩ - i*|11⟩)
                            self.real[idx_00] = K * (re_00 + im_11);
                            self.imag[idx_00] = K * (im_00 - re_11);

                            // new_01 = K * (|01⟩ + i*|10⟩)
                            self.real[idx_01] = K * (re_01 - im_10);
                            self.imag[idx_01] = K * (im_01 + re_10);

                            // new_10 = K * (|10⟩ + i*|01⟩)
                            self.real[idx_10] = K * (re_10 - im_01);
                            self.imag[idx_10] = K * (im_10 + re_01);

                            // new_11 = K * (|11⟩ - i*|00⟩)
                            self.real[idx_11] = K * (re_11 + im_00);
                            self.imag[idx_11] = K * (im_11 - re_00);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn szz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // SZZ = exp(-i * π/4 * Z⊗Z)
        // Z⊗Z is diagonal: diag(1, -1, -1, 1)
        // SZZ = diag(e^{-iπ/4}, e^{iπ/4}, e^{iπ/4}, e^{-iπ/4})
        // e^{-iπ/4} = (1-i)/√2: (re,im) -> K*(re+im, -re+im)
        // e^{iπ/4} = (1+i)/√2: (re,im) -> K*(re-im, re+im)
        const K: f64 = std::f64::consts::FRAC_1_SQRT_2;

        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            self.flush_two_qubit(q1, q2);

            let q_lo = q1.min(q2);

            // When both qubits >= 2, consecutive indices share the same phase
            if q_lo >= 2 {
                let n = self.real.len();
                let k_v = f64x4::splat(K);
                let mut i = 0;
                while i + 4 <= n {
                    let bit1 = (i >> q1) & 1;
                    let bit2 = (i >> q2) & 1;

                    let re = f64x4::from(&self.real[i..i + 4]);
                    let im = f64x4::from(&self.imag[i..i + 4]);

                    let (new_re, new_im) = if bit1 == bit2 {
                        // e^{-iπ/4}: (re,im) -> K*(re+im, -re+im)
                        (k_v * (re + im), k_v * (im - re))
                    } else {
                        // e^{iπ/4}: (re,im) -> K*(re-im, re+im)
                        (k_v * (re - im), k_v * (re + im))
                    };
                    let arr_re: [f64; 4] = new_re.into();
                    let arr_im: [f64; 4] = new_im.into();
                    self.real[i..i + 4].copy_from_slice(&arr_re);
                    self.imag[i..i + 4].copy_from_slice(&arr_im);
                    i += 4;
                }
            } else {
                // Scalar fallback
                let n = self.real.len();
                let mask1 = 1 << q1;
                let mask2 = 1 << q2;
                let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
                let step_lo = 1 << q_lo;
                let step_hi = 1 << q_hi;

                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_00 = base & !(mask1 | mask2);

                            if base != idx_00 {
                                continue;
                            }

                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            // |00⟩ → (1-i)/√2 |00⟩
                            let (re, im) = (self.real[idx_00], self.imag[idx_00]);
                            self.real[idx_00] = K * (re + im);
                            self.imag[idx_00] = K * (-re + im);

                            // |01⟩ → (1+i)/√2 |01⟩
                            let (re, im) = (self.real[idx_01], self.imag[idx_01]);
                            self.real[idx_01] = K * (re - im);
                            self.imag[idx_01] = K * (re + im);

                            // |10⟩ → (1+i)/√2 |10⟩
                            let (re, im) = (self.real[idx_10], self.imag[idx_10]);
                            self.real[idx_10] = K * (re - im);
                            self.imag[idx_10] = K * (re + im);

                            // |11⟩ → (1-i)/√2 |11⟩
                            let (re, im) = (self.real[idx_11], self.imag[idx_11]);
                            self.real[idx_11] = K * (re + im);
                            self.imag[idx_11] = K * (-re + im);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn szzdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // SZZDG = exp(+i * π/4 * Z⊗Z)
        // SZZDG = diag(e^{iπ/4}, e^{-iπ/4}, e^{-iπ/4}, e^{iπ/4})
        // e^{iπ/4} = (1+i)/√2: (re,im) -> K*(re-im, re+im)
        // e^{-iπ/4} = (1-i)/√2: (re,im) -> K*(re+im, -re+im)
        const K: f64 = std::f64::consts::FRAC_1_SQRT_2;

        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            self.flush_two_qubit(q1, q2);

            let q_lo = q1.min(q2);

            // When both qubits >= 2, consecutive indices share the same phase
            if q_lo >= 2 {
                let n = self.real.len();
                let k_v = f64x4::splat(K);
                let mut i = 0;
                while i + 4 <= n {
                    let bit1 = (i >> q1) & 1;
                    let bit2 = (i >> q2) & 1;

                    let re = f64x4::from(&self.real[i..i + 4]);
                    let im = f64x4::from(&self.imag[i..i + 4]);

                    let (new_re, new_im) = if bit1 == bit2 {
                        // e^{iπ/4}: (re,im) -> K*(re-im, re+im)
                        (k_v * (re - im), k_v * (re + im))
                    } else {
                        // e^{-iπ/4}: (re,im) -> K*(re+im, -re+im)
                        (k_v * (re + im), k_v * (im - re))
                    };
                    let arr_re: [f64; 4] = new_re.into();
                    let arr_im: [f64; 4] = new_im.into();
                    self.real[i..i + 4].copy_from_slice(&arr_re);
                    self.imag[i..i + 4].copy_from_slice(&arr_im);
                    i += 4;
                }
            } else {
                // Scalar fallback
                let n = self.real.len();
                let mask1 = 1 << q1;
                let mask2 = 1 << q2;
                let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
                let step_lo = 1 << q_lo;
                let step_hi = 1 << q_hi;

                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_00 = base & !(mask1 | mask2);

                            if base != idx_00 {
                                continue;
                            }

                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            // |00⟩ → (1+i)/√2 |00⟩
                            let (re, im) = (self.real[idx_00], self.imag[idx_00]);
                            self.real[idx_00] = K * (re - im);
                            self.imag[idx_00] = K * (re + im);

                            // |01⟩ → (1-i)/√2 |01⟩
                            let (re, im) = (self.real[idx_01], self.imag[idx_01]);
                            self.real[idx_01] = K * (re + im);
                            self.imag[idx_01] = K * (-re + im);

                            // |10⟩ → (1-i)/√2 |10⟩
                            let (re, im) = (self.real[idx_10], self.imag[idx_10]);
                            self.real[idx_10] = K * (re + im);
                            self.imag[idx_10] = K * (-re + im);

                            // |11⟩ → (1+i)/√2 |11⟩
                            let (re, im) = (self.real[idx_11], self.imag[idx_11]);
                            self.real[idx_11] = K * (re - im);
                            self.imag[idx_11] = K * (re + im);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn iswap(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // iSWAP matrix:
        // [[1, 0, 0, 0],
        //  [0, 0, i, 0],
        //  [0, i, 0, 0],
        //  [0, 0, 0, 1]]
        // |00⟩ → |00⟩, |01⟩ → i|10⟩, |10⟩ → i|01⟩, |11⟩ → |11⟩

        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            self.flush_two_qubit(q1, q2);

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask1 = 1 << q1;
            let mask2 = 1 << q2;

            if step_lo >= 4 {
                // SIMD version: when q_lo >= 2, consecutive base indices have
                // consecutive idx_01 and idx_10 values
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let base = i_lo + offset;
                            // Since we only process base indices where both qubit bits are 0,
                            // and step_lo >= 4, consecutive bases have consecutive idx values
                            let idx_01 = base | mask2;
                            let idx_10 = base | mask1;

                            // Load 4 consecutive values for |01⟩ and |10⟩ states
                            let re_01 = f64x4::from(&self.real[idx_01..idx_01 + 4]);
                            let im_01 = f64x4::from(&self.imag[idx_01..idx_01 + 4]);
                            let re_10 = f64x4::from(&self.real[idx_10..idx_10 + 4]);
                            let im_10 = f64x4::from(&self.imag[idx_10..idx_10 + 4]);

                            // new |01⟩ = i * old |10⟩: i * (re, im) = (-im, re)
                            let new_re_01: [f64; 4] = (-im_10).into();
                            let new_im_01: [f64; 4] = re_10.into();

                            // new |10⟩ = i * old |01⟩
                            let new_re_10: [f64; 4] = (-im_01).into();
                            let new_im_10: [f64; 4] = re_01.into();

                            self.real[idx_01..idx_01 + 4].copy_from_slice(&new_re_01);
                            self.imag[idx_01..idx_01 + 4].copy_from_slice(&new_im_01);
                            self.real[idx_10..idx_10 + 4].copy_from_slice(&new_re_10);
                            self.imag[idx_10..idx_10 + 4].copy_from_slice(&new_im_10);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_01 = (base & !(mask1 | mask2)) | mask2;
                            let idx_10 = (base & !(mask1 | mask2)) | mask1;

                            // Skip if we've already processed this pair
                            if base != (base & !(mask1 | mask2)) {
                                continue;
                            }

                            // Swap |01⟩ ↔ |10⟩ and multiply both by i
                            // i * (re, im) = (-im, re)
                            let (re_01, im_01) = (self.real[idx_01], self.imag[idx_01]);
                            let (re_10, im_10) = (self.real[idx_10], self.imag[idx_10]);

                            // new |01⟩ = i * old |10⟩
                            self.real[idx_01] = -im_10;
                            self.imag[idx_01] = re_10;

                            // new |10⟩ = i * old |01⟩
                            self.real[idx_10] = -im_01;
                            self.imag[idx_10] = re_01;
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn g(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // G = CZ.H(q1).H(q2).CZ
        // Traced through the decomposition, the actual matrix is:
        // [[1,  1,  1, -1],
        //  [1, -1,  1,  1],
        //  [1,  1, -1,  1],
        //  [-1, 1,  1,  1]] / 2
        //
        // new_00 = (|00⟩ + |01⟩ + |10⟩ - |11⟩) / 2
        // new_01 = (|00⟩ - |01⟩ + |10⟩ + |11⟩) / 2
        // new_10 = (|00⟩ + |01⟩ - |10⟩ + |11⟩) / 2
        // new_11 = (-|00⟩ + |01⟩ + |10⟩ + |11⟩) / 2

        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask1 = 1 << q1;
            let mask2 = 1 << q2;

            if step_lo >= 4 {
                // SIMD version: when q_lo >= 2, consecutive base indices have consecutive idx values
                let half_v = f64x4::splat(0.5);
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let idx_00 = i_lo + offset;
                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let re_00 = f64x4::from(&self.real[idx_00..idx_00 + 4]);
                            let im_00 = f64x4::from(&self.imag[idx_00..idx_00 + 4]);
                            let re_01 = f64x4::from(&self.real[idx_01..idx_01 + 4]);
                            let im_01 = f64x4::from(&self.imag[idx_01..idx_01 + 4]);
                            let re_10 = f64x4::from(&self.real[idx_10..idx_10 + 4]);
                            let im_10 = f64x4::from(&self.imag[idx_10..idx_10 + 4]);
                            let re_11 = f64x4::from(&self.real[idx_11..idx_11 + 4]);
                            let im_11 = f64x4::from(&self.imag[idx_11..idx_11 + 4]);

                            // new_00 = (|00⟩ + |01⟩ + |10⟩ - |11⟩) / 2
                            let new_re_00: [f64; 4] =
                                (half_v * (re_00 + re_01 + re_10 - re_11)).into();
                            let new_im_00: [f64; 4] =
                                (half_v * (im_00 + im_01 + im_10 - im_11)).into();

                            // new_01 = (|00⟩ - |01⟩ + |10⟩ + |11⟩) / 2
                            let new_re_01: [f64; 4] =
                                (half_v * (re_00 - re_01 + re_10 + re_11)).into();
                            let new_im_01: [f64; 4] =
                                (half_v * (im_00 - im_01 + im_10 + im_11)).into();

                            // new_10 = (|00⟩ + |01⟩ - |10⟩ + |11⟩) / 2
                            let new_re_10: [f64; 4] =
                                (half_v * (re_00 + re_01 - re_10 + re_11)).into();
                            let new_im_10: [f64; 4] =
                                (half_v * (im_00 + im_01 - im_10 + im_11)).into();

                            // new_11 = (-|00⟩ + |01⟩ + |10⟩ + |11⟩) / 2
                            let new_re_11: [f64; 4] =
                                (half_v * (-re_00 + re_01 + re_10 + re_11)).into();
                            let new_im_11: [f64; 4] =
                                (half_v * (-im_00 + im_01 + im_10 + im_11)).into();

                            self.real[idx_00..idx_00 + 4].copy_from_slice(&new_re_00);
                            self.imag[idx_00..idx_00 + 4].copy_from_slice(&new_im_00);
                            self.real[idx_01..idx_01 + 4].copy_from_slice(&new_re_01);
                            self.imag[idx_01..idx_01 + 4].copy_from_slice(&new_im_01);
                            self.real[idx_10..idx_10 + 4].copy_from_slice(&new_re_10);
                            self.imag[idx_10..idx_10 + 4].copy_from_slice(&new_im_10);
                            self.real[idx_11..idx_11 + 4].copy_from_slice(&new_re_11);
                            self.imag[idx_11..idx_11 + 4].copy_from_slice(&new_im_11);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_00 = base & !(mask1 | mask2);
                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            // Skip if we've already processed this quartet
                            if base != idx_00 {
                                continue;
                            }

                            let (re_00, im_00) = (self.real[idx_00], self.imag[idx_00]);
                            let (re_01, im_01) = (self.real[idx_01], self.imag[idx_01]);
                            let (re_10, im_10) = (self.real[idx_10], self.imag[idx_10]);
                            let (re_11, im_11) = (self.real[idx_11], self.imag[idx_11]);

                            // new_00 = (|00⟩ + |01⟩ + |10⟩ - |11⟩) / 2
                            self.real[idx_00] = 0.5 * (re_00 + re_01 + re_10 - re_11);
                            self.imag[idx_00] = 0.5 * (im_00 + im_01 + im_10 - im_11);

                            // new_01 = (|00⟩ - |01⟩ + |10⟩ + |11⟩) / 2
                            self.real[idx_01] = 0.5 * (re_00 - re_01 + re_10 + re_11);
                            self.imag[idx_01] = 0.5 * (im_00 - im_01 + im_10 + im_11);

                            // new_10 = (|00⟩ + |01⟩ - |10⟩ + |11⟩) / 2
                            self.real[idx_10] = 0.5 * (re_00 + re_01 - re_10 + re_11);
                            self.imag[idx_10] = 0.5 * (im_00 + im_01 - im_10 + im_11);

                            // new_11 = (-|00⟩ + |01⟩ + |10⟩ + |11⟩) / 2
                            self.real[idx_11] = 0.5 * (-re_00 + re_01 + re_10 + re_11);
                            self.imag[idx_11] = 0.5 * (-im_00 + im_01 + im_10 + im_11);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        // Flush all pending gates before measurement
        self.flush();

        // When measuring multiple qubits, use joint sampling to reduce
        // memory passes from 2k to 2. Each sequential measurement does
        // a probability pass + collapse pass over the full state vector.
        // Joint sampling computes all probabilities in one pass and
        // collapses in one pass.
        let k = qubits.len();
        if k >= 4 && k == self.num_qubits {
            return self.mz_joint_all(qubits);
        }
        if (4..=20).contains(&k) {
            return self.mz_joint_subset(qubits);
        }

        let mut results = Vec::with_capacity(qubits.len());
        for &q in qubits {
            let q_idx = q.index();
            let step = 1 << q_idx;

            // Calculate probability of measuring |1⟩ using SIMD
            let prob_one = self.probability_one(q_idx);

            // Sample outcome
            let outcome = self.rng.bernoulli(prob_one);
            let is_deterministic = !(1e-10..=1.0 - 1e-10).contains(&prob_one);

            // Collapse and renormalize
            let norm_factor = if outcome {
                1.0 / prob_one.sqrt()
            } else {
                1.0 / (1.0 - prob_one).sqrt()
            };

            // For small steps, use scalar collapse
            if step < 4 {
                for i in (0..self.real.len()).step_by(step * 2) {
                    if outcome {
                        for j in i..(i + step) {
                            self.real[j] = 0.0;
                            self.imag[j] = 0.0;
                        }
                        for j in (i + step)..(i + 2 * step) {
                            self.real[j] *= norm_factor;
                            self.imag[j] *= norm_factor;
                        }
                    } else {
                        for j in i..(i + step) {
                            self.real[j] *= norm_factor;
                            self.imag[j] *= norm_factor;
                        }
                        for j in (i + step)..(i + 2 * step) {
                            self.real[j] = 0.0;
                            self.imag[j] = 0.0;
                        }
                    }
                }
            } else {
                // SIMD collapse and renormalize
                let norm_vec = f64x4::splat(norm_factor);

                for i in (0..self.real.len()).step_by(step * 2) {
                    if outcome {
                        // Zero |0⟩ states, normalize |1⟩ states
                        let mut j = i;
                        while j + 4 <= i + step {
                            self.real[j..j + 4].copy_from_slice(&[0.0; 4]);
                            self.imag[j..j + 4].copy_from_slice(&[0.0; 4]);
                            j += 4;
                        }
                        let mut j = i + step;
                        while j + 4 <= i + 2 * step {
                            let re = f64x4::from(&self.real[j..j + 4]);
                            let im = f64x4::from(&self.imag[j..j + 4]);
                            let scaled_re: [f64; 4] = (norm_vec * re).into();
                            let scaled_im: [f64; 4] = (norm_vec * im).into();
                            self.real[j..j + 4].copy_from_slice(&scaled_re);
                            self.imag[j..j + 4].copy_from_slice(&scaled_im);
                            j += 4;
                        }
                    } else {
                        // Normalize |0⟩ states, zero |1⟩ states
                        let mut j = i;
                        while j + 4 <= i + step {
                            let re = f64x4::from(&self.real[j..j + 4]);
                            let im = f64x4::from(&self.imag[j..j + 4]);
                            let scaled_re: [f64; 4] = (norm_vec * re).into();
                            let scaled_im: [f64; 4] = (norm_vec * im).into();
                            self.real[j..j + 4].copy_from_slice(&scaled_re);
                            self.imag[j..j + 4].copy_from_slice(&scaled_im);
                            j += 4;
                        }
                        let mut j = i + step;
                        while j + 4 <= i + 2 * step {
                            self.real[j..j + 4].copy_from_slice(&[0.0; 4]);
                            self.imag[j..j + 4].copy_from_slice(&[0.0; 4]);
                            j += 4;
                        }
                    }
                }
            }

            results.push(MeasurementResult {
                outcome,
                is_deterministic,
            });
        }
        results
    }

    /// Optimized measure-and-prepare-Z: always prepares |0⟩ state.
    ///
    /// This is more efficient than the default implementation because it:
    /// 1. Always collapses to |0⟩ regardless of measurement outcome
    /// 2. Avoids the conditional X correction step
    #[inline]
    fn mpz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.flush();

        let mut results = Vec::with_capacity(qubits.len());
        for &q in qubits {
            let q_idx = q.index();
            let step = 1 << q_idx;

            // Calculate probability of measuring |1⟩
            let prob_one = self.probability_one(q_idx);

            // Sample outcome (for the measurement result)
            let outcome = self.rng.bernoulli(prob_one);
            let is_deterministic = !(1e-10..=1.0 - 1e-10).contains(&prob_one);

            // Always prepare |0⟩: zero the |1⟩ amplitudes and normalize |0⟩
            let norm_factor = 1.0 / (1.0 - prob_one).sqrt();

            if step < 4 {
                // Scalar path
                for i in (0..self.real.len()).step_by(step * 2) {
                    // Normalize |0⟩ states
                    for j in i..(i + step) {
                        self.real[j] *= norm_factor;
                        self.imag[j] *= norm_factor;
                    }
                    // Zero |1⟩ states
                    for j in (i + step)..(i + 2 * step) {
                        self.real[j] = 0.0;
                        self.imag[j] = 0.0;
                    }
                }
            } else {
                // SIMD path
                let norm_vec = f64x4::splat(norm_factor);

                for i in (0..self.real.len()).step_by(step * 2) {
                    // Normalize |0⟩ states
                    let mut j = i;
                    while j + 4 <= i + step {
                        let re = f64x4::from(&self.real[j..j + 4]);
                        let im = f64x4::from(&self.imag[j..j + 4]);
                        let scaled_re: [f64; 4] = (norm_vec * re).into();
                        let scaled_im: [f64; 4] = (norm_vec * im).into();
                        self.real[j..j + 4].copy_from_slice(&scaled_re);
                        self.imag[j..j + 4].copy_from_slice(&scaled_im);
                        j += 4;
                    }
                    // Zero |1⟩ states
                    let mut j = i + step;
                    while j + 4 <= i + 2 * step {
                        self.real[j..j + 4].copy_from_slice(&[0.0; 4]);
                        self.imag[j..j + 4].copy_from_slice(&[0.0; 4]);
                        j += 4;
                    }
                }
            }

            results.push(MeasurementResult {
                outcome,
                is_deterministic,
            });
        }
        results
    }
}

impl<R> ArbitraryRotationGateable for StateVecSoA<R>
where
    R: Rng,
{
    #[inline]
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        if self.fusion_enabled {
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            let m = Complex2x2 {
                a_re: cos,
                a_im: 0.0,
                b_re: 0.0,
                b_im: -sin,
                c_re: 0.0,
                c_im: -sin,
                d_re: cos,
                d_im: 0.0,
            };
            for &q in qubits {
                self.queue_gate(q.index(), &m);
            }
        } else {
            for &q in qubits {
                self.apply_rx_gate(q.index(), theta);
            }
        }
        self
    }

    #[inline]
    fn ry(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        if self.fusion_enabled {
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            let m = Complex2x2 {
                a_re: cos,
                a_im: 0.0,
                b_re: -sin,
                b_im: 0.0,
                c_re: sin,
                c_im: 0.0,
                d_re: cos,
                d_im: 0.0,
            };
            for &q in qubits {
                self.queue_gate(q.index(), &m);
            }
        } else {
            for &q in qubits {
                self.apply_ry_gate(q.index(), theta);
            }
        }
        self
    }

    #[inline]
    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        if self.fusion_enabled {
            let half = theta / 2.0;
            let cos = half.cos();
            let sin = half.sin();
            let m = Complex2x2 {
                a_re: cos,
                a_im: -sin,
                b_re: 0.0,
                b_im: 0.0,
                c_re: 0.0,
                c_im: 0.0,
                d_re: cos,
                d_im: sin,
            };
            for &q in qubits {
                self.queue_gate(q.index(), &m);
            }
        } else {
            for &q in qubits {
                self.apply_rz_gate(q.index(), theta);
            }
        }
        self
    }

    #[inline]
    fn r1xy(&mut self, theta: Angle64, phi: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let phi = phi.to_radians_signed();
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        // R1XY: [[cos, r01], [r10, cos]]
        // r01 = -i*sin*e^(-iφ) = -sin*sinφ - i*sin*cosφ
        // r10 = -i*sin*e^(iφ)  = sin*sinφ - i*sin*cosφ
        let m = Complex2x2 {
            a_re: cos,
            a_im: 0.0,
            b_re: -sin * phi.sin(),
            b_im: -sin * phi.cos(),
            c_re: sin * phi.sin(),
            c_im: -sin * phi.cos(),
            d_re: cos,
            d_im: 0.0,
        };
        for &q in qubits {
            self.queue_gate(q.index(), &m);
        }
        self
    }

    #[inline]
    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let cos_pos = (theta / 2.0).cos();
        let sin_pos = (theta / 2.0).sin();
        let cos_neg = (-theta / 2.0).cos();
        let sin_neg = (-theta / 2.0).sin();

        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            self.flush_two_qubit(q1, q2);

            let q_lo = q1.min(q2);

            // When both qubits >= 2, consecutive indices share the same phase
            if q_lo >= 2 {
                let n = self.real.len();
                let mut i = 0;
                while i + 4 <= n {
                    let bit1 = (i >> q1) & 1;
                    let bit2 = (i >> q2) & 1;
                    let (cos, sin) = if bit1 == bit2 {
                        (cos_neg, sin_neg)
                    } else {
                        (cos_pos, sin_pos)
                    };
                    let cos_v = f64x4::splat(cos);
                    let sin_v = f64x4::splat(sin);

                    let re = f64x4::from(&self.real[i..i + 4]);
                    let im = f64x4::from(&self.imag[i..i + 4]);
                    let new_re: [f64; 4] = (cos_v * re - sin_v * im).into();
                    let new_im: [f64; 4] = (sin_v * re + cos_v * im).into();
                    self.real[i..i + 4].copy_from_slice(&new_re);
                    self.imag[i..i + 4].copy_from_slice(&new_im);
                    i += 4;
                }
            } else {
                // Scalar fallback for small qubit indices
                for i in 0..self.real.len() {
                    let bit1 = (i >> q1) & 1;
                    let bit2 = (i >> q2) & 1;
                    let (cos, sin) = if bit1 == bit2 {
                        (cos_neg, sin_neg)
                    } else {
                        (cos_pos, sin_pos)
                    };
                    let re = self.real[i];
                    let im = self.imag[i];
                    self.real[i] = cos * re - sin * im;
                    self.imag[i] = sin * re + cos * im;
                }
            }
        }
        self
    }

    #[inline]
    fn rxx(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            self.flush_two_qubit(q1, q2);

            // Use strided iteration for cache efficiency
            let (lo, hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
            let step_lo = 1 << lo;
            let step_hi = 1 << hi;

            // RXX matrix (in computational basis):
            // |00⟩ -> cos|00⟩ - i*sin|11⟩
            // |01⟩ -> cos|01⟩ - i*sin|10⟩
            // |10⟩ -> -i*sin|01⟩ + cos|10⟩
            // |11⟩ -> -i*sin|00⟩ + cos|11⟩

            for outer in (0..self.real.len()).step_by(step_hi * 2) {
                for mid in (0..step_hi).step_by(step_lo * 2) {
                    for inner_idx in 0..step_lo {
                        let base = outer + mid + inner_idx;
                        let i00 = base;
                        let i01 = base + step_lo;
                        let i10 = base + step_hi;
                        let i11 = base + step_hi + step_lo;

                        // Load amplitudes
                        let (r00, m00) = (self.real[i00], self.imag[i00]);
                        let (r01, m01) = (self.real[i01], self.imag[i01]);
                        let (r10, m10) = (self.real[i10], self.imag[i10]);
                        let (r11, m11) = (self.real[i11], self.imag[i11]);

                        // Apply RXX: multiply by -i*sin means (re, im) -> (sin*im, -sin*re)
                        // new|00⟩ = cos*|00⟩ - i*sin*|11⟩
                        self.real[i00] = cos * r00 + sin * m11;
                        self.imag[i00] = cos * m00 - sin * r11;

                        // new|01⟩ = cos*|01⟩ - i*sin*|10⟩
                        self.real[i01] = cos * r01 + sin * m10;
                        self.imag[i01] = cos * m01 - sin * r10;

                        // new|10⟩ = -i*sin*|01⟩ + cos*|10⟩
                        self.real[i10] = sin * m01 + cos * r10;
                        self.imag[i10] = -sin * r01 + cos * m10;

                        // new|11⟩ = -i*sin*|00⟩ + cos*|11⟩
                        self.real[i11] = sin * m00 + cos * r11;
                        self.imag[i11] = -sin * r00 + cos * m11;
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn ryy(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            self.flush_two_qubit(q1, q2);

            // Use strided iteration for cache efficiency
            let (lo, hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
            let step_lo = 1 << lo;
            let step_hi = 1 << hi;

            // RYY matrix (in computational basis):
            // |00⟩ -> cos|00⟩ + i*sin|11⟩
            // |01⟩ -> cos|01⟩ - i*sin|10⟩
            // |10⟩ -> -i*sin|01⟩ + cos|10⟩
            // |11⟩ -> i*sin|00⟩ + cos|11⟩

            for outer in (0..self.real.len()).step_by(step_hi * 2) {
                for mid in (0..step_hi).step_by(step_lo * 2) {
                    for inner_idx in 0..step_lo {
                        let base = outer + mid + inner_idx;
                        let i00 = base;
                        let i01 = base + step_lo;
                        let i10 = base + step_hi;
                        let i11 = base + step_hi + step_lo;

                        // Load amplitudes
                        let (r00, m00) = (self.real[i00], self.imag[i00]);
                        let (r01, m01) = (self.real[i01], self.imag[i01]);
                        let (r10, m10) = (self.real[i10], self.imag[i10]);
                        let (r11, m11) = (self.real[i11], self.imag[i11]);

                        // Apply RYY: multiply by i*sin means (re, im) -> (-sin*im, sin*re)
                        // new|00⟩ = cos*|00⟩ + i*sin*|11⟩
                        self.real[i00] = cos * r00 - sin * m11;
                        self.imag[i00] = cos * m00 + sin * r11;

                        // new|01⟩ = cos*|01⟩ - i*sin*|10⟩
                        self.real[i01] = cos * r01 + sin * m10;
                        self.imag[i01] = cos * m01 - sin * r10;

                        // new|10⟩ = -i*sin*|01⟩ + cos*|10⟩
                        self.real[i10] = sin * m01 + cos * r10;
                        self.imag[i10] = -sin * r01 + cos * m10;

                        // new|11⟩ = i*sin*|00⟩ + cos*|11⟩
                        self.real[i11] = -sin * m00 + cos * r11;
                        self.imag[i11] = sin * r00 + cos * m11;
                    }
                }
            }
        }
        self
    }

    #[inline]
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

        // U gate matrix elements
        let m = Complex2x2 {
            a_re: cos,
            a_im: 0.0,
            b_re: -sin * lambda.cos(),
            b_im: -sin * lambda.sin(),
            c_re: sin * phi.cos(),
            c_im: sin * phi.sin(),
            d_re: cos * (phi + lambda).cos(),
            d_im: cos * (phi + lambda).sin(),
        };

        for &q in qubits {
            self.queue_gate(q.index(), &m);
        }
        self
    }
}

// ============================================================================
// RNG Management
// ============================================================================

impl<R> RngManageable for StateVecSoA<R>
where
    R: Rng + SeedableRng + Debug,
{
    type Rng = R;

    #[inline]
    fn set_rng(&mut self, rng: R) {
        self.rng = rng;
    }

    #[inline]
    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    #[inline]
    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

// ============================================================================
// Fused Gate Operations
// ============================================================================
//
// These fused gates combine two operations into a single pass over memory,
// reducing memory bandwidth requirements by ~50% compared to separate gates.

impl<R> StateVecSoA<R>
where
    R: Rng,
{
    /// Fused H-Z gate: applies H then Z in a single memory pass.
    ///
    /// Matrix: Z*H = 1/√2 [[1, 1], [-1, 1]] (rightmost applied first)
    ///
    /// This is ~1.5x faster than calling `h()` then `z()` separately.
    #[inline]
    pub fn hz(&mut self, qubits: &[QubitId]) -> &mut Self {
        let k = std::f64::consts::FRAC_1_SQRT_2;
        let m = Complex2x2 {
            a_re: k,
            a_im: 0.0,
            b_re: k,
            b_im: 0.0,
            c_re: -k,
            c_im: 0.0,
            d_re: k,
            d_im: 0.0,
        };
        for &q in qubits {
            self.flush_qubit(q.index());
            self.apply_fused_matrix(q.index(), &m);
        }
        self
    }

    /// Fused Z-H gate: applies Z then H in a single memory pass.
    ///
    /// Matrix: H*Z = 1/√2 [[1, -1], [1, 1]] (rightmost applied first)
    #[inline]
    pub fn zh(&mut self, qubits: &[QubitId]) -> &mut Self {
        let k = std::f64::consts::FRAC_1_SQRT_2;
        let m = Complex2x2 {
            a_re: k,
            a_im: 0.0,
            b_re: -k,
            b_im: 0.0,
            c_re: k,
            c_im: 0.0,
            d_re: k,
            d_im: 0.0,
        };
        for &q in qubits {
            self.flush_qubit(q.index());
            self.apply_fused_matrix(q.index(), &m);
        }
        self
    }

    /// Fused H-S gate: applies H then S in a single memory pass.
    ///
    /// Matrix: S*H = 1/√2 [[1, 1], [i, -i]] (rightmost applied first)
    #[inline]
    pub fn hs(&mut self, qubits: &[QubitId]) -> &mut Self {
        let k = std::f64::consts::FRAC_1_SQRT_2;
        let m = Complex2x2 {
            a_re: k,
            a_im: 0.0,
            b_re: k,
            b_im: 0.0,
            c_re: 0.0,
            c_im: k,
            d_re: 0.0,
            d_im: -k,
        };
        for &q in qubits {
            self.flush_qubit(q.index());
            self.apply_fused_matrix(q.index(), &m);
        }
        self
    }

    /// Fused S-H gate: applies S then H in a single memory pass.
    ///
    /// Matrix: H*S = 1/√2 [[1, i], [1, -i]] (rightmost applied first)
    #[inline]
    pub fn sh(&mut self, qubits: &[QubitId]) -> &mut Self {
        let k = std::f64::consts::FRAC_1_SQRT_2;
        let m = Complex2x2 {
            a_re: k,
            a_im: 0.0,
            b_re: 0.0,
            b_im: k,
            c_re: k,
            c_im: 0.0,
            d_re: 0.0,
            d_im: -k,
        };
        for &q in qubits {
            self.flush_qubit(q.index());
            self.apply_fused_matrix(q.index(), &m);
        }
        self
    }

    /// Fused H-X gate: applies H then X in a single memory pass.
    ///
    /// Matrix: X*H = 1/√2 [[1, -1], [1, 1]] (rightmost applied first)
    #[inline]
    pub fn hx(&mut self, qubits: &[QubitId]) -> &mut Self {
        let k = std::f64::consts::FRAC_1_SQRT_2;
        // Same as zh
        let m = Complex2x2 {
            a_re: k,
            a_im: 0.0,
            b_re: -k,
            b_im: 0.0,
            c_re: k,
            c_im: 0.0,
            d_re: k,
            d_im: 0.0,
        };
        for &q in qubits {
            self.flush_qubit(q.index());
            self.apply_fused_matrix(q.index(), &m);
        }
        self
    }

    /// Fused X-H gate: applies X then H in a single memory pass.
    ///
    /// Matrix: H*X = 1/√2 [[1, 1], [-1, 1]] (rightmost applied first)
    #[inline]
    pub fn xh(&mut self, qubits: &[QubitId]) -> &mut Self {
        let k = std::f64::consts::FRAC_1_SQRT_2;
        // Same as hz
        let m = Complex2x2 {
            a_re: k,
            a_im: 0.0,
            b_re: k,
            b_im: 0.0,
            c_re: -k,
            c_im: 0.0,
            d_re: k,
            d_im: 0.0,
        };
        for &q in qubits {
            self.flush_qubit(q.index());
            self.apply_fused_matrix(q.index(), &m);
        }
        self
    }

    /// Fused H on target then CX: applies H(target) then CX(control, target) in optimized passes.
    ///
    /// This pattern is common for creating entanglement after preparing superposition.
    /// The H and CX operate on the same target qubit, allowing some optimization.
    #[inline]
    pub fn h_then_cx(&mut self, control: QubitId, target: QubitId) -> &mut Self {
        // Apply H to target first
        self.h(&[target]);
        // Then apply CX - these can't be fully fused since they have different structure
        self.cx(&[(control, target)]);
        self
    }

    /// Fused CX then H on target: applies CX(control, target) then H(target).
    ///
    /// This pattern is common in measurement preparation.
    #[inline]
    pub fn cx_then_h(&mut self, control: QubitId, target: QubitId) -> &mut Self {
        self.cx(&[(control, target)]);
        self.h(&[target]);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateVecSoA;
    use num_complex::Complex64;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, PI};

    fn states_match(sv: &mut StateVecSoA, opt: &mut StateVecSoA, tolerance: f64) -> bool {
        sv.state().iter().enumerate().all(|(i, c)| {
            let opt_c = opt.get_amplitude(i);
            (*c - opt_c).norm() < tolerance
        })
    }

    fn assert_states_match(sv: &mut StateVecSoA, opt: &mut StateVecSoA, context: &str) {
        const TOLERANCE: f64 = 1e-10;
        assert!(
            states_match(sv, opt, TOLERANCE),
            "States don't match for {context}"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_new_state() {
        let mut opt: StateVecSoA = StateVecSoA::new(3);
        assert_eq!(opt.num_qubits(), 3);
        assert_eq!(opt.real().len(), 8);
        assert_eq!(opt.real()[0], 1.0);
        assert_eq!(opt.imag()[0], 0.0);
        for i in 1..8 {
            assert_eq!(opt.real()[i], 0.0);
            assert_eq!(opt.imag()[i], 0.0);
        }
    }

    #[test]
    fn test_h_gate() {
        for num_qubits in 1..=5 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                sv.h(&[QubitId(target)]);
                opt.h(&[QubitId(target)]);

                assert_states_match(
                    &mut sv,
                    &mut opt,
                    &format!("H on qubit {target} of {num_qubits}"),
                );
            }
        }
    }

    #[test]
    fn test_x_gate() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                sv.x(&[QubitId(target)]);
                opt.x(&[QubitId(target)]);

                assert_states_match(&mut sv, &mut opt, &format!("X on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_y_gate() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                sv.y(&[QubitId(target)]);
                opt.y(&[QubitId(target)]);

                assert_states_match(&mut sv, &mut opt, &format!("Y on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_z_gate() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                sv.h(&[QubitId(target)]);
                opt.h(&[QubitId(target)]);
                sv.z(&[QubitId(target)]);
                opt.z(&[QubitId(target)]);

                assert_states_match(&mut sv, &mut opt, &format!("Z on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_cx_gate() {
        for num_qubits in 2..=4 {
            for control in 0..num_qubits {
                for target in 0..num_qubits {
                    if control == target {
                        continue;
                    }

                    let mut sv = StateVecSoA::new(num_qubits);
                    let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                    sv.h(&[QubitId(control)]);
                    opt.h(&[QubitId(control)]);

                    sv.cx(&[(QubitId(control), QubitId(target))]);
                    opt.cx(&[(QubitId(control), QubitId(target))]);

                    assert_states_match(
                        &mut sv,
                        &mut opt,
                        &format!("CX({control},{target}) in {num_qubits}q"),
                    );
                }
            }
        }
    }

    #[test]
    fn test_cz_gate() {
        for num_qubits in 2..=4 {
            let mut sv = StateVecSoA::new(num_qubits);
            let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

            for q in 0..num_qubits {
                sv.h(&[QubitId(q)]);
                opt.h(&[QubitId(q)]);
            }

            sv.cz(&[(QubitId(0), QubitId(1))]);
            opt.cz(&[(QubitId(0), QubitId(1))]);

            assert_states_match(&mut sv, &mut opt, &format!("CZ in {num_qubits}q"));
        }
    }

    #[test]
    fn test_swap_gate() {
        for num_qubits in 2..=4 {
            let mut sv = StateVecSoA::new(num_qubits);
            let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

            sv.x(&[QubitId(0)]);
            opt.x(&[QubitId(0)]);
            sv.h(&[QubitId(1)]);
            opt.h(&[QubitId(1)]);

            sv.swap(&[(QubitId(0), QubitId(1))]);
            opt.swap(&[(QubitId(0), QubitId(1))]);

            assert_states_match(&mut sv, &mut opt, &format!("SWAP in {num_qubits}q"));
        }
    }

    #[test]
    fn test_rx_gate() {
        let angles = [0.0, FRAC_PI_2, PI, 1.234];
        for &theta in &angles {
            let mut sv = StateVecSoA::new(2);
            let mut opt: StateVecSoA = StateVecSoA::new(2);

            sv.rx(Angle64::from_radians(theta), &[QubitId(0)]);
            opt.rx(Angle64::from_radians(theta), &[QubitId(0)]);

            assert_states_match(&mut sv, &mut opt, &format!("RX({theta})"));
        }
    }

    #[test]
    fn test_u_gate() {
        let mut sv = StateVecSoA::new(2);
        let mut opt: StateVecSoA = StateVecSoA::new(2);

        sv.u(
            Angle64::from_radians(PI / 3.0),
            Angle64::from_radians(PI / 4.0),
            Angle64::from_radians(PI / 5.0),
            &[QubitId(0)],
        );
        opt.u(
            Angle64::from_radians(PI / 3.0),
            Angle64::from_radians(PI / 4.0),
            Angle64::from_radians(PI / 5.0),
            &[QubitId(0)],
        );

        assert_states_match(&mut sv, &mut opt, "U gate");
    }

    #[test]
    fn test_ghz_state() {
        let num_qubits = 4;
        let mut sv = StateVecSoA::new(num_qubits);
        let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

        sv.h(&[QubitId(0)]);
        opt.h(&[QubitId(0)]);
        for i in 0..(num_qubits - 1) {
            sv.cx(&[(QubitId(i), QubitId(i + 1))]);
            opt.cx(&[(QubitId(i), QubitId(i + 1))]);
        }

        assert_states_match(&mut sv, &mut opt, "GHZ state");
    }

    #[test]
    fn test_mz_deterministic() {
        // Test deterministic measurement of |0⟩
        let mut opt: StateVecSoA = StateVecSoA::new(1);
        let result = opt.mz(&[QubitId(0)]);
        assert!(!result[0].outcome, "Expected 0 outcome for |0> state");

        // Test deterministic measurement of |1⟩
        let mut opt: StateVecSoA = StateVecSoA::new(1);
        opt.x(&[QubitId(0)]);
        let result = opt.mz(&[QubitId(0)]);
        assert!(result[0].outcome, "Expected 1 outcome for |1> state");
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_reset() {
        let mut opt: StateVecSoA = StateVecSoA::new(3);
        opt.h(&[QubitId(0), QubitId(1), QubitId(2)]);
        opt.cx(&[(QubitId(0), QubitId(1))]);
        opt.reset();

        assert_eq!(opt.real()[0], 1.0);
        for i in 1..opt.real().len() {
            assert_eq!(opt.real()[i], 0.0);
            assert_eq!(opt.imag()[i], 0.0);
        }
    }

    // Helper to compare two StateVecSoA instances
    fn opts_match(a: &mut StateVecSoA, b: &mut StateVecSoA, tolerance: f64) -> bool {
        a.flush();
        b.flush();
        let a_real = a.real().to_vec();
        let a_imag = a.imag().to_vec();
        let b_real = b.real().to_vec();
        let b_imag = b.imag().to_vec();
        a_real
            .iter()
            .zip(&b_real)
            .all(|(x, y)| (x - y).abs() < tolerance)
            && a_imag
                .iter()
                .zip(&b_imag)
                .all(|(x, y)| (x - y).abs() < tolerance)
    }

    fn assert_opts_match(a: &mut StateVecSoA, b: &mut StateVecSoA, context: &str) {
        const TOLERANCE: f64 = 1e-10;
        assert!(
            opts_match(a, b, TOLERANCE),
            "States don't match for {context}"
        );
    }

    #[test]
    fn test_fused_hz() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                // Prepare non-trivial state first
                let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                // Put into superposition
                separate.h(&[QubitId(target)]);
                fused.h(&[QubitId(target)]);

                // Apply H then Z separately
                separate.h(&[QubitId(target)]);
                separate.z(&[QubitId(target)]);

                // Apply fused H-Z
                fused.hz(&[QubitId(target)]);

                assert_opts_match(
                    &mut separate,
                    &mut fused,
                    &format!("HZ fused on qubit {target}"),
                );
            }
        }
    }

    #[test]
    fn test_fused_zh() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                separate.h(&[QubitId(target)]);
                fused.h(&[QubitId(target)]);

                // Apply Z then H separately
                separate.z(&[QubitId(target)]);
                separate.h(&[QubitId(target)]);

                // Apply fused Z-H
                fused.zh(&[QubitId(target)]);

                assert_opts_match(
                    &mut separate,
                    &mut fused,
                    &format!("ZH fused on qubit {target}"),
                );
            }
        }
    }

    #[test]
    fn test_fused_hs() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                separate.h(&[QubitId(target)]);
                fused.h(&[QubitId(target)]);

                // Apply H then S separately
                separate.h(&[QubitId(target)]);
                separate.sz(&[QubitId(target)]);

                // Apply fused H-S
                fused.hs(&[QubitId(target)]);

                assert_opts_match(
                    &mut separate,
                    &mut fused,
                    &format!("HS fused on qubit {target}"),
                );
            }
        }
    }

    #[test]
    fn test_fused_sh() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                separate.h(&[QubitId(target)]);
                fused.h(&[QubitId(target)]);

                // Apply S then H separately
                separate.sz(&[QubitId(target)]);
                separate.h(&[QubitId(target)]);

                // Apply fused S-H
                fused.sh(&[QubitId(target)]);

                assert_opts_match(
                    &mut separate,
                    &mut fused,
                    &format!("SH fused on qubit {target}"),
                );
            }
        }
    }

    #[test]
    fn test_fused_hx() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                separate.h(&[QubitId(target)]);
                fused.h(&[QubitId(target)]);

                // Apply H then X separately
                separate.h(&[QubitId(target)]);
                separate.x(&[QubitId(target)]);

                // Apply fused H-X
                fused.hx(&[QubitId(target)]);

                assert_opts_match(
                    &mut separate,
                    &mut fused,
                    &format!("HX fused on qubit {target}"),
                );
            }
        }
    }

    #[test]
    fn test_fused_xh() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                separate.h(&[QubitId(target)]);
                fused.h(&[QubitId(target)]);

                // Apply X then H separately
                separate.x(&[QubitId(target)]);
                separate.h(&[QubitId(target)]);

                // Apply fused X-H
                fused.xh(&[QubitId(target)]);

                assert_opts_match(
                    &mut separate,
                    &mut fused,
                    &format!("XH fused on qubit {target}"),
                );
            }
        }
    }

    #[test]
    fn test_fused_h_then_cx() {
        for num_qubits in 2..=4 {
            for control in 0..num_qubits {
                for target in 0..num_qubits {
                    if control == target {
                        continue;
                    }

                    let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                    let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                    // Apply H then CX separately
                    separate.h(&[QubitId(target)]);
                    separate.cx(&[(QubitId(control), QubitId(target))]);

                    // Apply fused H-CX
                    fused.h_then_cx(QubitId(control), QubitId(target));

                    assert_opts_match(
                        &mut separate,
                        &mut fused,
                        &format!("H-CX fused c={control} t={target}"),
                    );
                }
            }
        }
    }

    #[test]
    fn test_fused_cx_then_h() {
        for num_qubits in 2..=4 {
            for control in 0..num_qubits {
                for target in 0..num_qubits {
                    if control == target {
                        continue;
                    }

                    let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                    let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                    // Prepare entangled state first
                    separate.h(&[QubitId(control)]);
                    fused.h(&[QubitId(control)]);
                    separate.cx(&[(QubitId(control), QubitId(target))]);
                    fused.cx(&[(QubitId(control), QubitId(target))]);

                    // Apply CX then H separately
                    separate.cx(&[(QubitId(control), QubitId(target))]);
                    separate.h(&[QubitId(target)]);

                    // Apply fused CX-H
                    fused.cx_then_h(QubitId(control), QubitId(target));

                    assert_opts_match(
                        &mut separate,
                        &mut fused,
                        &format!("CX-H fused c={control} t={target}"),
                    );
                }
            }
        }
    }

    // Additional tests for parity with StateVecSoA test coverage

    #[test]
    fn test_probability() {
        let mut sv = StateVecSoA::new(1);
        let mut opt: StateVecSoA = StateVecSoA::new(1);

        // Prepare |+⟩ state
        sv.h(&[QubitId(0)]);
        opt.h(&[QubitId(0)]);

        let sv_prob_zero = sv.probability(0);
        let sv_prob_one = sv.probability(1);

        let opt_prob_zero = opt.probability(0);
        let opt_prob_one = opt.probability(1);

        assert!((sv_prob_zero - opt_prob_zero).abs() < 1e-10);
        assert!((sv_prob_one - opt_prob_one).abs() < 1e-10);
        assert!((opt_prob_zero - 0.5).abs() < 1e-10);
        assert!((opt_prob_one - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_prepare_computational_basis_all_states() {
        let num_qubits = 3;

        for basis_state in 0..(1 << num_qubits) {
            let mut sv = StateVecSoA::new(num_qubits);
            let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

            sv.prepare_computational_basis(basis_state);
            opt.prepare_computational_basis(basis_state);

            assert_states_match(
                &mut sv,
                &mut opt,
                &format!("prepare_computational_basis({basis_state})"),
            );
        }
    }

    #[test]
    fn test_sz_gate() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                // Put into superposition first to see effect
                sv.h(&[QubitId(target)]);
                opt.h(&[QubitId(target)]);
                sv.sz(&[QubitId(target)]);
                opt.sz(&[QubitId(target)]);

                assert_states_match(&mut sv, &mut opt, &format!("SZ on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_cy_gate() {
        for num_qubits in 2..=4 {
            for control in 0..num_qubits {
                for target in 0..num_qubits {
                    if control == target {
                        continue;
                    }

                    let mut sv = StateVecSoA::new(num_qubits);
                    let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                    // Create |+0⟩ state
                    sv.h(&[QubitId(control)]);
                    opt.h(&[QubitId(control)]);

                    sv.cy(&[(QubitId(control), QubitId(target))]);
                    opt.cy(&[(QubitId(control), QubitId(target))]);

                    assert_states_match(
                        &mut sv,
                        &mut opt,
                        &format!("CY({control},{target}) in {num_qubits}q"),
                    );
                }
            }
        }
    }

    #[test]
    fn test_measurement_consistency() {
        // Measuring a deterministic state should always give the same result
        let mut opt: StateVecSoA = StateVecSoA::new(1);
        opt.x(&[QubitId(0)]); // Put in |1⟩

        let result1 = opt.mz(&[QubitId(0)]);
        let result2 = opt.mz(&[QubitId(0)]);

        assert!(result1[0].outcome);
        assert!(result2[0].outcome);
    }

    #[test]
    fn test_measurement_collapse() {
        let mut opt: StateVecSoA = StateVecSoA::new(1);

        // Prepare |+⟩ = (|0⟩ + |1⟩) / √2
        opt.h(&[QubitId(0)]);

        // Simulate a measurement
        let result = opt.mz(&[QubitId(0)]);

        // State should collapse to |0⟩ or |1⟩
        if result[0].outcome {
            assert!((opt.probability(1) - 1.0).abs() < 1e-10);
        } else {
            assert!((opt.probability(0) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pz() {
        // Use same seed for both to ensure deterministic comparison
        let seed = 42;
        let mut sv = StateVecSoA::with_seed(1, seed);
        let mut opt: StateVecSoA = StateVecSoA::with_seed(1, seed);

        sv.h(&[QubitId(0)]);
        opt.h(&[QubitId(0)]);

        sv.pz(&[QubitId(0)]);
        opt.pz(&[QubitId(0)]);

        assert_states_match(&mut sv, &mut opt, "PZ on single qubit");
    }

    #[test]
    fn test_pz_multiple_qubits() {
        // Use same seed for both to ensure deterministic comparison
        let seed = 42;
        let mut sv = StateVecSoA::with_seed(2, seed);
        let mut opt: StateVecSoA = StateVecSoA::with_seed(2, seed);

        sv.h(&[QubitId(0)]);
        opt.h(&[QubitId(0)]);
        sv.cx(&[(QubitId(0), QubitId(1))]);
        opt.cx(&[(QubitId(0), QubitId(1))]);

        sv.pz(&[QubitId(0)]);
        opt.pz(&[QubitId(0)]);

        assert_states_match(&mut sv, &mut opt, "PZ on entangled state");
    }

    #[test]
    fn test_ry_gate() {
        let angles = [0.0, FRAC_PI_2, PI, 1.234];
        for &theta in &angles {
            let mut sv = StateVecSoA::new(2);
            let mut opt: StateVecSoA = StateVecSoA::new(2);

            sv.ry(Angle64::from_radians(theta), &[QubitId(0)]);
            opt.ry(Angle64::from_radians(theta), &[QubitId(0)]);

            assert_states_match(&mut sv, &mut opt, &format!("RY({theta})"));
        }
    }

    #[test]
    fn test_rz_gate() {
        let angles = [0.0, FRAC_PI_2, PI, 1.234];
        for &theta in &angles {
            let mut sv = StateVecSoA::new(2);
            let mut opt: StateVecSoA = StateVecSoA::new(2);

            // Put in superposition to see phase effects
            sv.h(&[QubitId(0)]);
            opt.h(&[QubitId(0)]);
            sv.rz(Angle64::from_radians(theta), &[QubitId(0)]);
            opt.rz(Angle64::from_radians(theta), &[QubitId(0)]);

            assert_states_match(&mut sv, &mut opt, &format!("RZ({theta})"));
        }
    }

    #[test]
    fn test_r1xy_gate() {
        let mut sv = StateVecSoA::new(1);
        let mut opt: StateVecSoA = StateVecSoA::new(1);

        let theta = FRAC_PI_3;
        let phi = FRAC_PI_4;

        sv.r1xy(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            &[QubitId(0)],
        );
        opt.r1xy(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            &[QubitId(0)],
        );

        assert_states_match(&mut sv, &mut opt, "R1XY gate");
    }

    #[test]
    fn test_rxx_gate() {
        let angles = [FRAC_PI_2, PI, FRAC_PI_4];
        for &theta in &angles {
            let mut sv = StateVecSoA::new(2);
            let mut opt: StateVecSoA = StateVecSoA::new(2);

            sv.rxx(Angle64::from_radians(theta), &[(QubitId(0), QubitId(1))]);
            opt.rxx(Angle64::from_radians(theta), &[(QubitId(0), QubitId(1))]);

            assert_states_match(&mut sv, &mut opt, &format!("RXX({theta})"));
        }
    }

    #[test]
    fn test_ryy_gate() {
        let angles = [FRAC_PI_2, PI, FRAC_PI_4];
        for &theta in &angles {
            let mut sv = StateVecSoA::new(2);
            let mut opt: StateVecSoA = StateVecSoA::new(2);

            sv.ryy(Angle64::from_radians(theta), &[(QubitId(0), QubitId(1))]);
            opt.ryy(Angle64::from_radians(theta), &[(QubitId(0), QubitId(1))]);

            assert_states_match(&mut sv, &mut opt, &format!("RYY({theta})"));
        }
    }

    #[test]
    fn test_rzz_gate() {
        let angles = [FRAC_PI_2, PI, FRAC_PI_4];
        for &theta in &angles {
            let mut sv = StateVecSoA::new(2);
            let mut opt: StateVecSoA = StateVecSoA::new(2);

            // Create non-trivial state
            sv.h(&[QubitId(0)]);
            opt.h(&[QubitId(0)]);
            sv.h(&[QubitId(1)]);
            opt.h(&[QubitId(1)]);

            sv.rzz(Angle64::from_radians(theta), &[(QubitId(0), QubitId(1))]);
            opt.rzz(Angle64::from_radians(theta), &[(QubitId(0), QubitId(1))]);

            assert_states_match(&mut sv, &mut opt, &format!("RZZ({theta})"));
        }
    }

    #[test]
    fn test_normalization() {
        let mut opt: StateVecSoA = StateVecSoA::new(2);
        opt.h(&[QubitId(0)])
            .sz(&[QubitId(0)])
            .cx(&[(QubitId(0), QubitId(1))]);

        let real_copy = opt.real().to_vec();
        let imag_copy = opt.imag().to_vec();
        let norm: f64 = real_copy
            .iter()
            .zip(&imag_copy)
            .map(|(r, i)| r * r + i * i)
            .sum();
        assert!((norm - 1.0).abs() < 1e-10, "State should be normalized");
    }

    #[test]
    fn test_unitarity() {
        let mut opt: StateVecSoA = StateVecSoA::new(1);
        opt.h(&[QubitId(0)]);
        let initial_real = opt.real().to_vec();
        let initial_imag = opt.imag().to_vec();

        // H^2 = I
        opt.h(&[QubitId(0)]).h(&[QubitId(0)]);

        let final_real = opt.real().to_vec();
        let final_imag = opt.imag().to_vec();

        for i in 0..final_real.len() {
            assert!(
                (final_real[i] - initial_real[i]).abs() < 1e-10,
                "H^2 should equal I (real part)"
            );
            assert!(
                (final_imag[i] - initial_imag[i]).abs() < 1e-10,
                "H^2 should equal I (imag part)"
            );
        }
    }

    #[test]
    fn test_pauli_relations() {
        // XYZ = iI (up to global phase)
        let mut sv = StateVecSoA::new(1);
        let mut opt: StateVecSoA = StateVecSoA::new(1);

        sv.x(&[QubitId(0)]).y(&[QubitId(0)]).z(&[QubitId(0)]);
        opt.x(&[QubitId(0)]).y(&[QubitId(0)]).z(&[QubitId(0)]);

        assert_states_match(&mut sv, &mut opt, "XYZ sequence");

        // Also verify YZX gives same result
        let mut sv2 = StateVecSoA::new(1);
        let mut opt2: StateVecSoA = StateVecSoA::new(1);

        sv2.y(&[QubitId(0)]).z(&[QubitId(0)]).x(&[QubitId(0)]);
        opt2.y(&[QubitId(0)]).z(&[QubitId(0)]).x(&[QubitId(0)]);

        assert_states_match(&mut sv2, &mut opt2, "YZX sequence");
    }

    #[test]
    fn test_bell_state_correlations() {
        // Create Bell state and verify measurement correlations
        let mut opt: StateVecSoA = StateVecSoA::new(2);
        opt.h(&[QubitId(0)]);
        opt.cx(&[(QubitId(0), QubitId(1))]);

        // Measure first qubit
        let result1 = opt.mz(&[QubitId(0)]);
        // Measure second qubit - should match first
        let result2 = opt.mz(&[QubitId(1)]);

        assert_eq!(
            result1[0].outcome, result2[0].outcome,
            "Bell state measurements should be correlated"
        );
    }

    #[test]
    fn test_sx_gate() {
        for num_qubits in 1..=3 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                sv.sx(&[QubitId(target)]);
                opt.sx(&[QubitId(target)]);

                assert_states_match(&mut sv, &mut opt, &format!("SX on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_sxdg_gate() {
        for num_qubits in 1..=3 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                sv.sxdg(&[QubitId(target)]);
                opt.sxdg(&[QubitId(target)]);

                assert_states_match(&mut sv, &mut opt, &format!("SXDG on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_szdg_gate() {
        for num_qubits in 1..=3 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                sv.h(&[QubitId(target)]);
                opt.h(&[QubitId(target)]);
                sv.szdg(&[QubitId(target)]);
                opt.szdg(&[QubitId(target)]);

                assert_states_match(&mut sv, &mut opt, &format!("SZDG on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_iswap_gate() {
        for num_qubits in 2..=3 {
            let mut sv = StateVecSoA::new(num_qubits);
            let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

            sv.x(&[QubitId(0)]);
            opt.x(&[QubitId(0)]);

            sv.iswap(&[(QubitId(0), QubitId(1))]);
            opt.iswap(&[(QubitId(0), QubitId(1))]);

            assert_states_match(&mut sv, &mut opt, &format!("ISWAP in {num_qubits}q"));
        }
    }

    #[test]
    fn test_measurement_superposition_statistics() {
        // Test that superposition measurements are roughly 50/50
        let mut zeros = 0;
        let trials = 1000;

        for _ in 0..trials {
            let mut opt: StateVecSoA = StateVecSoA::new(1);
            opt.h(&[QubitId(0)]);
            let result = opt.mz(&[QubitId(0)]);
            if !result[0].outcome {
                zeros += 1;
            }
        }

        let ratio = f64::from(zeros) / f64::from(trials);
        assert!(
            (ratio - 0.5).abs() < 0.1,
            "Superposition measurements should be roughly 50/50, got {ratio}"
        );
    }

    // Tests for new convenience and inspection methods

    #[test]
    fn test_get_set_amplitude() {
        let mut opt: StateVecSoA = StateVecSoA::new(2);

        // Initial state should be |00⟩
        assert!((opt.get_amplitude(0) - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!(opt.get_amplitude(1).norm() < 1e-10);

        // Set a specific amplitude
        opt.set_amplitude(1, Complex64::new(0.5, 0.5));
        let amp = opt.get_amplitude(1);
        assert!((amp.re - 0.5).abs() < 1e-10);
        assert!((amp.im - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_to_complex_vec() {
        let mut opt: StateVecSoA = StateVecSoA::new(2);
        opt.h(&[QubitId(0)]);

        let complex_state = opt.to_complex_vec();
        assert_eq!(complex_state.len(), 4);

        // After H on qubit 0: (|00⟩ + |01⟩)/√2
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((complex_state[0].re - expected).abs() < 1e-10);
        assert!((complex_state[1].re - expected).abs() < 1e-10);
        assert!(complex_state[2].norm() < 1e-10);
        assert!(complex_state[3].norm() < 1e-10);
    }

    #[test]
    fn test_from_complex_state() {
        let state = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ];

        let mut opt: StateVecSoA = StateVecSoA::from_complex_state(&state, rand::make_rng());

        for (i, expected) in state.iter().enumerate() {
            let actual = opt.get_amplitude(i);
            assert!((actual - expected).norm() < 1e-10);
        }
    }

    #[test]
    fn test_prepare_plus_state() {
        let mut opt: StateVecSoA = StateVecSoA::new(3);
        opt.prepare_plus_state();

        // Verify all amplitudes are equal to 1/sqrt(2^n) for n qubits
        let expected = 1.0 / (8.0_f64).sqrt();
        for i in 0..8 {
            let amp = opt.get_amplitude(i);
            assert!(
                (amp.re - expected).abs() < 1e-10,
                "Real part mismatch at index {i}"
            );
            assert!(
                amp.im.abs() < 1e-10,
                "Imaginary part should be zero at index {i}"
            );
        }

        // Verify normalization (sum of probabilities = 1)
        let total_prob: f64 = (0..8).map(|i| opt.probability(i)).sum();
        assert!(
            (total_prob - 1.0).abs() < 1e-10,
            "State should be normalized"
        );
    }

    #[test]
    fn test_single_qubit_unitary() {
        use std::f64::consts::FRAC_1_SQRT_2;

        // Test Hadamard gate via single_qubit_unitary
        let mut opt: StateVecSoA = StateVecSoA::new(1);
        opt.single_qubit_unitary(
            0,
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(-FRAC_1_SQRT_2, 0.0),
        );

        // Compare with H gate
        let mut opt2: StateVecSoA = StateVecSoA::new(1);
        opt2.h(&[QubitId(0)]);

        assert_opts_match(&mut opt, &mut opt2, "single_qubit_unitary (Hadamard)");

        // Test X gate via single_qubit_unitary
        let mut opt: StateVecSoA = StateVecSoA::new(1);
        opt.single_qubit_unitary(
            0,
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        );

        let mut opt2: StateVecSoA = StateVecSoA::new(1);
        opt2.x(&[QubitId(0)]);

        assert_opts_match(&mut opt, &mut opt2, "single_qubit_unitary (X)");
    }

    #[test]
    fn test_two_qubit_unitary() {
        // Test CNOT gate via two_qubit_unitary
        let cnot = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        ];

        // Create Bell state using two_qubit_unitary
        let mut opt: StateVecSoA = StateVecSoA::new(2);
        opt.h(&[QubitId(0)]);
        opt.two_qubit_unitary(0, 1, cnot);

        // Create Bell state using CX
        let mut opt2: StateVecSoA = StateVecSoA::new(2);
        opt2.h(&[QubitId(0)]);
        opt2.cx(&[(QubitId(0), QubitId(1))]);

        assert_opts_match(&mut opt, &mut opt2, "two_qubit_unitary (CNOT)");

        // Test SWAP gate via two_qubit_unitary
        let swap = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ];

        let mut opt: StateVecSoA = StateVecSoA::new(2);
        opt.x(&[QubitId(0)]);
        opt.two_qubit_unitary(0, 1, swap);

        let mut opt2: StateVecSoA = StateVecSoA::new(2);
        opt2.x(&[QubitId(0)]);
        opt2.swap(&[(QubitId(0), QubitId(1))]);

        assert_opts_match(&mut opt, &mut opt2, "two_qubit_unitary (SWAP)");
    }

    #[test]
    fn test_roundtrip_complex_state() {
        // Create a non-trivial state
        let mut opt: StateVecSoA = StateVecSoA::new(2);
        opt.h(&[QubitId(0)]);
        opt.cx(&[(QubitId(0), QubitId(1))]);

        // Convert to complex vec and back
        let complex_state = opt.to_complex_vec();
        let mut opt2: StateVecSoA =
            StateVecSoA::from_complex_state(&complex_state, rand::make_rng());

        assert_opts_match(&mut opt, &mut opt2, "roundtrip complex state");
    }

    /// Test that parallel execution produces the same results as sequential.
    /// This test is only run when the `parallel` feature is enabled.
    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_execution_correctness() {
        // Use 16 qubits to exceed the parallel threshold (14)
        let num_qubits = 16;

        // Create two simulators: one sequential, one parallel
        let mut seq = StateVecSoA::with_seed(num_qubits, 12345);
        let mut par = StateVecSoA::with_seed(num_qubits, 12345);
        par.set_parallel(true);

        // Apply a variety of gates
        for q in 0..num_qubits {
            seq.h(&[QubitId(q)]);
            par.h(&[QubitId(q)]);
        }
        for q in 0..num_qubits {
            seq.sz(&[QubitId(q)]);
            par.sz(&[QubitId(q)]);
        }
        for q in 0..num_qubits {
            seq.sx(&[QubitId(q)]);
            par.sx(&[QubitId(q)]);
        }

        // Compare states
        let seq_state = seq.state();
        let par_state = par.state();

        for (i, (s, p)) in seq_state.iter().zip(par_state.iter()).enumerate() {
            let diff = (*s - *p).norm();
            assert!(
                diff < 1e-10,
                "State mismatch at index {i}: seq={s}, par={p}, diff={diff}"
            );
        }
    }

    /// Test that parallel execution with custom thread count produces correct results.
    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_with_custom_threads() {
        let num_qubits = 16;

        // Create two simulators: one with default threads, one with limited threads
        let mut default_threads = StateVecSoA::with_seed(num_qubits, 12345);
        default_threads.parallel(true);

        let mut limited_threads = StateVecSoA::with_seed(num_qubits, 12345);
        limited_threads.parallel(true).num_threads(Some(2));

        // Apply gates
        for q in 0..num_qubits {
            default_threads.h(&[QubitId(q)]);
            limited_threads.h(&[QubitId(q)]);
        }

        // Compare states - they should be identical
        let state1 = default_threads.state();
        let state2 = limited_threads.state();

        for (i, (s1, s2)) in state1.iter().zip(state2.iter()).enumerate() {
            let diff = (*s1 - *s2).norm();
            assert!(
                diff < 1e-10,
                "State mismatch at index {i}: default={s1}, limited={s2}, diff={diff}"
            );
        }
    }

    /// Test the builder-style API for parallel configuration.
    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_builder_api() {
        let mut sim = StateVecSoA::new(10);

        // Test chaining
        sim.parallel(true).num_threads(Some(4));

        assert!(sim.parallel_enabled());
        assert_eq!(sim.get_num_threads(), Some(4));

        // Test changing settings
        sim.parallel(false).num_threads(None);

        assert!(!sim.parallel_enabled());
        assert_eq!(sim.get_num_threads(), None);
    }
}
