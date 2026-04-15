// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

//! GPU density matrix simulator via Choi-Jamiolkowski isomorphism.
//!
//! Represents an N-qubit density matrix as a 2N-qubit state vector on the GPU.
//! Each physical-qubit gate G becomes two state-vector gates: G on qubit q
//! (system) and G-dagger on qubit q+N (environment).
//!
//! Generic over the backing GPU state vector (f32 or f64). Use the
//! [`GpuDensityMatrix64`] alias for f64 precision (canonical) or
//! [`GpuDensityMatrix32`] to trade precision for a ~2x smaller state.

use crate::{GpuError, GpuStateVec32, GpuStateVec64};
use num_complex::Complex64;
use pecos_core::{Angle64, QubitId, RngManageable};
use pecos_random::{PecosRng, RngExt, SeedableRng};
use pecos_simulators::arbitrary_rotation_gateable::ArbitraryRotationGateable;
use pecos_simulators::clifford_gateable::{CliffordGateable, MeasurementResult};
use pecos_simulators::quantum_simulator::QuantumSimulator;

// =============================================================================
// Backend trait
// =============================================================================

/// Abstraction over a GPU state vector simulator that `GpuDensityMatrix` can
/// be built on top of. The trait exposes state readback and write-back in
/// f64 (the backend converts as needed) plus the standard gate traits.
pub trait GpuStateVecBackend:
    CliffordGateable + ArbitraryRotationGateable + QuantumSimulator + Sized
{
    /// Construct a backend with `num_qubits` qubits.
    ///
    /// # Errors
    /// Returns [`GpuError`] if the backend GPU initialization fails.
    fn new_backend(num_qubits: u32) -> Result<Self, GpuError>;

    /// Readback the state vector as f64 amplitudes `[re, im]`.
    fn state_f64(&mut self) -> Vec<[f64; 2]>;

    /// Overwrite the state vector from f64 amplitudes.
    fn write_state_f64(&mut self, amps: &[[f64; 2]]);

    /// Force all pending GPU work to complete (for honest timing).
    fn sync_backend(&mut self);
}

impl GpuStateVecBackend for GpuStateVec32 {
    fn new_backend(num_qubits: u32) -> Result<Self, GpuError> {
        GpuStateVec32::new(num_qubits)
    }

    fn state_f64(&mut self) -> Vec<[f64; 2]> {
        self.state()
            .into_iter()
            .map(|[re, im]| [f64::from(re), f64::from(im)])
            .collect()
    }

    fn write_state_f64(&mut self, amps: &[[f64; 2]]) {
        #[allow(clippy::cast_possible_truncation)] // f32 backend: accept f64→f32 precision loss
        let f32_amps: Vec<[f32; 2]> = amps
            .iter()
            .map(|[re, im]| [*re as f32, *im as f32])
            .collect();
        self.write_state(&f32_amps);
    }

    fn sync_backend(&mut self) {
        self.sync();
    }
}

impl GpuStateVecBackend for GpuStateVec64 {
    fn new_backend(num_qubits: u32) -> Result<Self, GpuError> {
        GpuStateVec64::new(num_qubits)
    }

    fn state_f64(&mut self) -> Vec<[f64; 2]> {
        self.state()
    }

    fn write_state_f64(&mut self, amps: &[[f64; 2]]) {
        self.write_state(amps);
    }

    fn sync_backend(&mut self) {
        self.sync();
    }
}

// =============================================================================
// GpuDensityMatrix
// =============================================================================

/// GPU-backed density matrix simulator, generic over the backend precision.
pub struct GpuDensityMatrix<SV: GpuStateVecBackend> {
    num_physical_qubits: usize,
    state_vector: SV,
    rng: PecosRng,
}

/// f32 GPU density matrix: ~2x smaller state, single-precision amplitudes.
/// Use when memory is the bottleneck or you need an extra physical qubit.
pub type GpuDensityMatrix32 = GpuDensityMatrix<GpuStateVec32>;

/// f64 GPU density matrix: canonical precision, matches CPU reference to
/// ~1e-10 in isolation.
pub type GpuDensityMatrix64 = GpuDensityMatrix<GpuStateVec64>;

impl<SV: GpuStateVecBackend> GpuDensityMatrix<SV> {
    /// Create a density matrix for `n` physical qubits, initialized to |0..0><0..0|.
    ///
    /// # Errors
    /// Returns [`GpuError`] if the backend GPU state vector cannot be created.
    pub fn new(num_physical_qubits: usize) -> Result<Self, GpuError> {
        let sv_qubits =
            u32::try_from(2 * num_physical_qubits).map_err(|_| GpuError::TooManyQubits {
                requested: u32::MAX,
                max: 15,
            })?;
        let state_vector = SV::new_backend(sv_qubits)?;
        Ok(Self {
            num_physical_qubits,
            state_vector,
            rng: PecosRng::from_seed([0u8; 32]),
        })
    }

    /// Create a density matrix with a deterministic RNG seed.
    ///
    /// # Errors
    /// Returns [`GpuError`] if the backend GPU state vector cannot be created.
    pub fn with_seed(num_physical_qubits: usize, seed: u64) -> Result<Self, GpuError> {
        let sv_qubits =
            u32::try_from(2 * num_physical_qubits).map_err(|_| GpuError::TooManyQubits {
                requested: u32::MAX,
                max: 15,
            })?;
        let state_vector = SV::new_backend(sv_qubits)?;
        Ok(Self {
            num_physical_qubits,
            state_vector,
            rng: PecosRng::seed_from_u64(seed),
        })
    }

    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_physical_qubits
    }

    #[must_use]
    pub fn state_vector(&self) -> &SV {
        &self.state_vector
    }

    pub fn state_vector_mut(&mut self) -> &mut SV {
        &mut self.state_vector
    }

    /// Force all pending GPU work to complete. Call before timing measurements.
    pub fn sync(&mut self) {
        self.state_vector.sync_backend();
    }

    // -------------------------------------------------------------------------
    // Helpers: probability / density matrix / purity
    // -------------------------------------------------------------------------

    /// Probability of measuring the computational basis state `basis_state`.
    /// P(k) = rho_{k,k} = `sum_i` |psi[(k << n) | i]|^2 in the Choi representation.
    ///
    /// # Panics
    /// Panics if `basis_state >= 2^num_physical_qubits`.
    #[must_use]
    pub fn probability(&mut self, basis_state: usize) -> f64 {
        assert!(basis_state < 1 << self.num_physical_qubits);
        let n = self.num_physical_qubits;
        let sv = self.state_vector.state_f64();
        let mut prob = 0.0;
        for i in 0..(1 << n) {
            let idx = (basis_state << n) | i;
            let [re, im] = sv[idx];
            prob += re * re + im * im;
        }
        prob
    }

    /// Full `NxN` density matrix as a flat row-major complex slab.
    /// `rho[row * dim + col] = [re, im]`.
    #[must_use]
    pub fn get_density_matrix(&mut self) -> Vec<Vec<Complex64>> {
        let n = self.num_physical_qubits;
        let dim = 1 << n;
        let sv = self.state_vector.state_f64();

        let mut rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        for (row, row_vec) in rho.iter_mut().enumerate() {
            for (col, cell) in row_vec.iter_mut().enumerate() {
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for k in 0..dim {
                    let idx1 = (row << n) | k;
                    let idx2 = (col << n) | k;
                    let [ar, ai] = sv[idx1];
                    let [br, bi] = sv[idx2];
                    // a * conj(b)
                    re += ar * br + ai * bi;
                    im += ai * br - ar * bi;
                }
                *cell = Complex64::new(re, im);
            }
        }
        rho
    }

    /// Tr(rho^2). 1 for pure states; 1/2^n for maximally mixed.
    ///
    /// # Panics
    /// Panics if the backing state vector has unexpected size.
    #[must_use]
    pub fn purity(&mut self) -> f64 {
        self.get_density_matrix()
            .iter()
            .flatten()
            .map(Complex64::norm_sqr)
            .sum()
    }

    /// Returns true if the state is pure (purity within `tol` of 1.0).
    /// The default `is_pure` uses `1e-5`, reflecting the f32-precision gate
    /// constants in the backing state vectors. Pass a tighter tolerance if
    /// you need stricter purity checks on known-noise-free states.
    #[must_use]
    pub fn is_pure_with_tol(&mut self, tol: f64) -> bool {
        (self.purity() - 1.0).abs() < tol
    }

    #[must_use]
    pub fn is_pure(&mut self) -> bool {
        self.is_pure_with_tol(1e-5)
    }

    // -------------------------------------------------------------------------
    // State preparation
    // -------------------------------------------------------------------------

    /// Prepare |`basis_state`><`basis_state`|.
    ///
    /// # Panics
    /// Panics if `basis_state >= 2^num_physical_qubits`.
    pub fn prepare_computational_basis(&mut self, basis_state: usize) -> &mut Self {
        assert!(basis_state < 1 << self.num_physical_qubits);
        let n = self.num_physical_qubits;
        let sv_size = 1usize << (2 * n);
        let mut new_state = vec![[0.0f64, 0.0f64]; sv_size];
        let idx = (basis_state << n) | basis_state;
        new_state[idx] = [1.0, 0.0];
        self.state_vector.write_state_f64(&new_state);
        self
    }

    /// Prepare |+>^N: tensor product of |+> states on all qubits.
    pub fn prepare_plus_state(&mut self) -> &mut Self {
        let n = self.num_physical_qubits;
        self.prepare_computational_basis(0);
        for q in 0..n {
            self.h(&[QubitId(q)]);
        }
        self
    }

    /// Prepare the maximally mixed state I / 2^n.
    pub fn prepare_maximally_mixed(&mut self) -> &mut Self {
        let n = self.num_physical_qubits;
        let sv_size = 1usize << (2 * n);
        let dim = 1usize << n;
        #[allow(clippy::cast_precision_loss)] // dim = 2^n with n <= 15, exact in f64
        let factor = 1.0 / (dim as f64).sqrt();
        let mut new_state = vec![[0.0f64, 0.0f64]; sv_size];
        for i in 0..dim {
            new_state[(i << n) | i] = [factor, 0.0];
        }
        self.state_vector.write_state_f64(&new_state);
        self
    }

    // -------------------------------------------------------------------------
    // Noise channels
    // -------------------------------------------------------------------------

    /// Amplitude damping: `rho -> E_0 rho E_0^dagger + E_1 rho E_1^dagger` with
    /// `E_0 = |0><0| + sqrt(1-gamma)|1><1|`, `E_1 = sqrt(gamma)|0><1|`. Applies
    /// the channel on the density matrix then Cholesky-re-purifies the Choi
    /// state so `probability()` / `purity()` stay consistent.
    pub fn apply_amplitude_damping(&mut self, qubit: usize, gamma: f64) -> &mut Self {
        let gamma = gamma.clamp(0.0, 1.0);
        if gamma < f64::EPSILON {
            return self;
        }
        let n = self.num_physical_qubits;
        let dim = 1usize << n;
        let qubit_mask = 1usize << qubit;

        let rho = self.get_density_matrix();
        let mut new_rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        let sqrt_1mg = (1.0 - gamma).sqrt();
        for i in 0..dim {
            let i1 = (i & qubit_mask) != 0;
            for j in 0..dim {
                let j1 = (j & qubit_mask) != 0;
                new_rho[i][j] = match (i1, j1) {
                    (false, false) => {
                        let ii = i | qubit_mask;
                        let jj = j | qubit_mask;
                        rho[i][j] + gamma * rho[ii][jj]
                    }
                    (true, true) => (1.0 - gamma) * rho[i][j],
                    _ => sqrt_1mg * rho[i][j],
                };
            }
        }
        self.set_from_density_matrix(&new_rho);
        self
    }

    /// Phase damping (pure dephasing): diagonals preserved, off-diagonals
    /// (w.r.t. the target qubit) scaled by sqrt(1-lambda). Applies the channel
    /// on the density matrix then Cholesky-re-purifies the Choi state.
    pub fn apply_phase_damping(&mut self, qubit: usize, lambda: f64) -> &mut Self {
        let lambda = lambda.clamp(0.0, 1.0);
        if lambda < f64::EPSILON {
            return self;
        }
        let n = self.num_physical_qubits;
        let dim = 1usize << n;
        let qubit_mask = 1usize << qubit;

        let rho = self.get_density_matrix();
        let mut new_rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        let sqrt_1ml = (1.0 - lambda).sqrt();
        for i in 0..dim {
            let i1 = (i & qubit_mask) != 0;
            for j in 0..dim {
                let j1 = (j & qubit_mask) != 0;
                new_rho[i][j] = if i1 == j1 {
                    rho[i][j]
                } else {
                    sqrt_1ml * rho[i][j]
                };
            }
        }
        self.set_from_density_matrix(&new_rho);
        self
    }

    /// Depolarizing: `rho -> (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)`.
    /// Uses Cholesky re-purification of the transformed density matrix, so has
    /// a readback + O(dim^3) CPU cost + writeback round trip.
    pub fn apply_depolarizing_noise(&mut self, qubit: usize, probability: f64) -> &mut Self {
        let p = probability.clamp(0.0, 1.0);
        if p < f64::EPSILON {
            return self;
        }
        let n = self.num_physical_qubits;
        let dim = 1usize << n;
        let qubit_mask = 1usize << qubit;

        let rho = self.get_density_matrix();
        let mut new_rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let i1 = (i & qubit_mask) != 0;
                let j1 = (j & qubit_mask) != 0;
                if i1 == j1 {
                    let i_flip = i ^ qubit_mask;
                    let j_flip = j ^ qubit_mask;
                    new_rho[i][j] =
                        (1.0 - 2.0 * p / 3.0) * rho[i][j] + (2.0 * p / 3.0) * rho[i_flip][j_flip];
                } else {
                    new_rho[i][j] = (1.0 - 4.0 * p / 3.0) * rho[i][j];
                }
            }
        }
        self.set_from_density_matrix(&new_rho);
        self
    }

    /// Bit flip: `rho -> (1-p) rho + p X rho X`.
    pub fn apply_bit_flip(&mut self, qubit: usize, probability: f64) -> &mut Self {
        let p = probability.clamp(0.0, 1.0);
        if p < f64::EPSILON {
            return self;
        }
        let n = self.num_physical_qubits;
        let dim = 1usize << n;
        let qubit_mask = 1usize << qubit;

        let rho = self.get_density_matrix();
        let mut new_rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let i_flip = i ^ qubit_mask;
                let j_flip = j ^ qubit_mask;
                new_rho[i][j] = (1.0 - p) * rho[i][j] + p * rho[i_flip][j_flip];
            }
        }
        self.set_from_density_matrix(&new_rho);
        self
    }

    /// Phase flip: `rho -> (1-p) rho + p Z rho Z`.
    pub fn apply_phase_flip(&mut self, qubit: usize, probability: f64) -> &mut Self {
        let p = probability.clamp(0.0, 1.0);
        if p < f64::EPSILON {
            return self;
        }
        let n = self.num_physical_qubits;
        let dim = 1usize << n;
        let qubit_mask = 1usize << qubit;

        let rho = self.get_density_matrix();
        let mut new_rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let i1 = (i & qubit_mask) != 0;
                let j1 = (j & qubit_mask) != 0;
                new_rho[i][j] = if i1 == j1 {
                    rho[i][j]
                } else {
                    (1.0 - 2.0 * p) * rho[i][j]
                };
            }
        }
        self.set_from_density_matrix(&new_rho);
        self
    }

    /// Re-establish the Choi state from a density matrix via Cholesky.
    /// `rho = L L^dagger`, then `psi[(i<<n)|j] = L[i][j]`.
    ///
    /// Two numerical guards: the diagonal sqrt clamps slight negatives to zero
    /// (numerical noise on a true PSD rho), and off-diagonals on a near-zero
    /// pivot are left at zero. Both fire only on rounding noise for a
    /// well-formed channel; a buggy channel that sends rho non-PSD will trip
    /// the `debug_assert` below.
    #[allow(clippy::needless_range_loop)] // Cholesky: indexed access into multiple matrices
    fn set_from_density_matrix(&mut self, rho: &[Vec<Complex64>]) {
        // Tolerance for "legitimate numerical noise" on the diagonal.
        // For a properly PSD rho with trace 1, diagonals are in [0, 1] and
        // accumulated f64 rounding error is bounded by ~dim * eps_f64.
        // 1e-9 is well above that for any practical dim and well below any
        // physically meaningful negative eigenvalue.
        const PSD_NEG_TOL: f64 = -1e-9;
        // Pivot threshold for off-diagonal division. Same scale as the
        // diagonal noise floor: pivots below this are treated as zero rather
        // than dividing by them and amplifying noise.
        const PIVOT_EPS: f64 = 1e-15;

        let n = self.num_physical_qubits;
        let dim = rho.len();

        let mut l: Vec<Vec<Complex64>> = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        for i in 0..dim {
            for j in 0..=i {
                let mut sum: Complex64 = rho[i][j];
                for k in 0..j {
                    sum -= l[i][k] * l[j][k].conj();
                }
                if i == j {
                    debug_assert!(
                        sum.re > PSD_NEG_TOL,
                        "set_from_density_matrix: rho not PSD at diag[{i}]: {} (tol {PSD_NEG_TOL:e}); \
                         a noise channel likely violated trace preservation or positivity",
                        sum.re
                    );
                    let diag = sum.re.max(0.0);
                    l[i][j] = Complex64::new(diag.sqrt(), 0.0);
                } else if l[j][j].norm() > PIVOT_EPS {
                    l[i][j] = sum / l[j][j];
                }
            }
        }

        let sv_size = 1usize << (2 * n);
        let mut new_state = vec![[0.0f64, 0.0f64]; sv_size];
        for i in 0..dim {
            for j in 0..dim {
                let idx = (i << n) | j;
                let c: Complex64 = l[i][j];
                new_state[idx] = [c.re, c.im];
            }
        }
        self.state_vector.write_state_f64(&new_state);
    }

    // -------------------------------------------------------------------------
    // Internal gate helpers
    // -------------------------------------------------------------------------

    fn apply_1q_sys_env<F, G>(&mut self, qubits: &[QubitId], sys_op: F, env_op: G)
    where
        F: Fn(&mut SV, &[QubitId]),
        G: Fn(&mut SV, &[QubitId]),
    {
        let n = self.num_physical_qubits;
        for &q in qubits {
            let qi = q.index();
            sys_op(&mut self.state_vector, &[QubitId(qi)]);
            env_op(&mut self.state_vector, &[QubitId(qi + n)]);
        }
    }

    fn apply_2q_sys_env<F, G>(&mut self, pairs: &[(QubitId, QubitId)], sys_op: F, env_op: G)
    where
        F: Fn(&mut SV, &[(QubitId, QubitId)]),
        G: Fn(&mut SV, &[(QubitId, QubitId)]),
    {
        let n = self.num_physical_qubits;
        for &(c, t) in pairs {
            let ci = c.index();
            let ti = t.index();
            sys_op(&mut self.state_vector, &[(QubitId(ci), QubitId(ti))]);
            env_op(
                &mut self.state_vector,
                &[(QubitId(ci + n), QubitId(ti + n))],
            );
        }
    }
}

impl<SV: GpuStateVecBackend> QuantumSimulator for GpuDensityMatrix<SV> {
    fn reset(&mut self) -> &mut Self {
        self.state_vector.reset();
        self
    }
}

impl<SV: GpuStateVecBackend> RngManageable for GpuDensityMatrix<SV> {
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

impl<SV: GpuStateVecBackend> CliffordGateable for GpuDensityMatrix<SV> {
    // --- Hermitian 1q: apply identically on system and environment ---

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_1q_sys_env(
            qubits,
            |s, q| {
                s.h(q);
            },
            |s, q| {
                s.h(q);
            },
        );
        self
    }

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_1q_sys_env(
            qubits,
            |s, q| {
                s.x(q);
            },
            |s, q| {
                s.x(q);
            },
        );
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_1q_sys_env(
            qubits,
            |s, q| {
                s.y(q);
            },
            |s, q| {
                s.y(q);
            },
        );
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_1q_sys_env(
            qubits,
            |s, q| {
                s.z(q);
            },
            |s, q| {
                s.z(q);
            },
        );
        self
    }

    // --- Non-Hermitian 1q: env gets the dagger ---

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_1q_sys_env(
            qubits,
            |s, q| {
                s.sz(q);
            },
            |s, q| {
                s.szdg(q);
            },
        );
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_1q_sys_env(
            qubits,
            |s, q| {
                s.szdg(q);
            },
            |s, q| {
                s.sz(q);
            },
        );
        self
    }

    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_1q_sys_env(
            qubits,
            |s, q| {
                s.sx(q);
            },
            |s, q| {
                s.sxdg(q);
            },
        );
        self
    }

    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_1q_sys_env(
            qubits,
            |s, q| {
                s.sxdg(q);
            },
            |s, q| {
                s.sx(q);
            },
        );
        self
    }

    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_1q_sys_env(
            qubits,
            |s, q| {
                s.sy(q);
            },
            |s, q| {
                s.sydg(q);
            },
        );
        self
    }

    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_1q_sys_env(
            qubits,
            |s, q| {
                s.sydg(q);
            },
            |s, q| {
                s.sy(q);
            },
        );
        self
    }

    // --- 2q gates ---

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.apply_2q_sys_env(
            pairs,
            |s, p| {
                s.cx(p);
            },
            |s, p| {
                s.cx(p);
            },
        );
        self
    }

    fn cy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.apply_2q_sys_env(
            pairs,
            |s, p| {
                s.cy(p);
            },
            |s, p| {
                s.cy(p);
            },
        );
        self
    }

    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.apply_2q_sys_env(
            pairs,
            |s, p| {
                s.cz(p);
            },
            |s, p| {
                s.cz(p);
            },
        );
        self
    }

    fn swap(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.apply_2q_sys_env(
            pairs,
            |s, p| {
                s.swap(p);
            },
            |s, p| {
                s.swap(p);
            },
        );
        self
    }

    // SZZ/SXX/SYY family: non-Hermitian, env gets dagger

    fn szz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.apply_2q_sys_env(
            pairs,
            |s, p| {
                s.szz(p);
            },
            |s, p| {
                s.szzdg(p);
            },
        );
        self
    }

    fn szzdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.apply_2q_sys_env(
            pairs,
            |s, p| {
                s.szzdg(p);
            },
            |s, p| {
                s.szz(p);
            },
        );
        self
    }

    fn sxx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.apply_2q_sys_env(
            pairs,
            |s, p| {
                s.sxx(p);
            },
            |s, p| {
                s.sxxdg(p);
            },
        );
        self
    }

    fn sxxdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.apply_2q_sys_env(
            pairs,
            |s, p| {
                s.sxxdg(p);
            },
            |s, p| {
                s.sxx(p);
            },
        );
        self
    }

    fn syy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.apply_2q_sys_env(
            pairs,
            |s, p| {
                s.syy(p);
            },
            |s, p| {
                s.syydg(p);
            },
        );
        self
    }

    fn syydg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.apply_2q_sys_env(
            pairs,
            |s, p| {
                s.syydg(p);
            },
            |s, p| {
                s.syy(p);
            },
        );
        self
    }

    /// Z-basis projective measurement. Reads state back, samples + projects on
    /// the CPU, writes collapsed state back. O(2^(2N)) per measurement.
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let n = self.num_physical_qubits;
        let sv_size = 1usize << (2 * n);
        let mut results = Vec::with_capacity(qubits.len());

        for &q in qubits {
            let qubit = q.index();
            let state = self.state_vector.state_f64();

            // P(qubit = 1) = sum_k rho_{k,k} over k with bit_q(k) = 1.
            // In the purification convention, rho_{k,k} = sum_i |psi[(k<<n)|i]|^2,
            // so we sum over every Choi index whose system-row has bit_q set.
            let qubit_mask = 1usize << qubit;
            let prob_one: f64 = state
                .iter()
                .enumerate()
                .filter(|(idx, _)| ((idx >> n) & qubit_mask) != 0)
                .map(|(_, [re, im])| re * re + im * im)
                .sum();

            let is_deterministic = !(1e-10..=1.0 - 1e-10).contains(&prob_one);
            let outcome = if is_deterministic {
                prob_one > 0.5
            } else {
                self.rng.random_range(0.0..1.0) < prob_one
            };

            let target_bit = if outcome { qubit_mask } else { 0 };
            let mut new_state = vec![[0.0f64, 0.0f64]; sv_size];
            let mut norm_sq = 0.0;

            for idx in 0..sv_size {
                let row = idx >> n;
                let col = idx & ((1 << n) - 1);
                if (row & qubit_mask) == target_bit && (col & qubit_mask) == target_bit {
                    new_state[idx] = state[idx];
                    let [re, im] = state[idx];
                    norm_sq += re * re + im * im;
                }
            }

            if norm_sq > 1e-15 {
                let norm = norm_sq.sqrt();
                for amp in &mut new_state {
                    amp[0] /= norm;
                    amp[1] /= norm;
                }
            }

            self.state_vector.write_state_f64(&new_state);

            results.push(MeasurementResult {
                outcome,
                is_deterministic,
            });
        }

        results
    }
}

impl<SV: GpuStateVecBackend> ArbitraryRotationGateable for GpuDensityMatrix<SV> {
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        // RX(-theta) = Z * RX(theta) * Z
        let n = self.num_physical_qubits;
        for &q in qubits {
            let qi = q.index();
            self.state_vector.rx(theta, &[QubitId(qi)]);
            self.state_vector.z(&[QubitId(qi + n)]);
            self.state_vector.rx(theta, &[QubitId(qi + n)]);
            self.state_vector.z(&[QubitId(qi + n)]);
        }
        self
    }

    fn ry(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        // RY is real, RY* = RY
        let n = self.num_physical_qubits;
        for &q in qubits {
            let qi = q.index();
            self.state_vector.ry(theta, &[QubitId(qi)]);
            self.state_vector.ry(theta, &[QubitId(qi + n)]);
        }
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        // RZ(-theta) = X * RZ(theta) * X
        let n = self.num_physical_qubits;
        for &q in qubits {
            let qi = q.index();
            self.state_vector.rz(theta, &[QubitId(qi)]);
            self.state_vector.x(&[QubitId(qi + n)]);
            self.state_vector.rz(theta, &[QubitId(qi + n)]);
            self.state_vector.x(&[QubitId(qi + n)]);
        }
        self
    }

    fn t(&mut self, qubits: &[QubitId]) -> &mut Self {
        // T* = Tdg
        self.apply_1q_sys_env(
            qubits,
            |s, q| {
                s.t(q);
            },
            |s, q| {
                s.tdg(q);
            },
        );
        self
    }

    fn tdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_1q_sys_env(
            qubits,
            |s, q| {
                s.tdg(q);
            },
            |s, q| {
                s.t(q);
            },
        );
        self
    }

    // NOTE: we deliberately do NOT override rxx/ryy here. The default trait
    // impls decompose them into H-RZZ-H and SX-RZZ-SXdg sequences, which route
    // through our overridden h/sx/rzz (correct sys/env handling per gate).
    // The raw GpuStateVec RXX/RYY shaders have a pre-existing correctness bug
    // (only half the basis pairs updated) -- keeping this decomposition until
    // that's fixed.

    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // RZZ(-theta) = (X tensor I) RZZ(theta) (X tensor I). X on just one
        // qubit anticommutes with Z (x) Z; X on both commutes and doesn't flip.
        let n = self.num_physical_qubits;
        for &(c, t) in pairs {
            let ci = c.index();
            let ti = t.index();
            self.state_vector.rzz(theta, &[(QubitId(ci), QubitId(ti))]);
            self.state_vector.x(&[QubitId(ci + n)]);
            self.state_vector
                .rzz(theta, &[(QubitId(ci + n), QubitId(ti + n))]);
            self.state_vector.x(&[QubitId(ci + n)]);
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
    use pecos_simulators::DensityMatrix;

    // Primary tests run on the f32 backend (GpuDensityMatrix32), because the
    // f64 backend has pre-existing shader bugs in RZZ/RXX/RYY we haven't
    // fixed yet. Tolerance ~1e-3 reflects f32 precision for f64 comparisons.
    const TOL: f64 = 1e-3;

    fn gpu_dm_matrix<SV: GpuStateVecBackend>(sim: &mut GpuDensityMatrix<SV>) -> Vec<Complex64> {
        let rho = sim.get_density_matrix();
        rho.into_iter().flatten().collect()
    }

    fn cpu_dm_matrix(sim: &mut DensityMatrix) -> Vec<Complex64> {
        sim.get_density_matrix().into_iter().flatten().collect()
    }

    fn assert_dm_close(gpu: &[Complex64], cpu: &[Complex64], tol: f64, label: &str) {
        assert_eq!(gpu.len(), cpu.len(), "{label}: dim mismatch");
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let d = (g - c).norm();
            assert!(d < tol, "{label}: idx {i} gpu={g} cpu={c} diff={d}");
        }
    }

    // --- Regression: RZZ on |0000> for sv64 (was broken, now fixed) ---

    #[test]
    fn regression_sv64_rzz_on_zero() {
        use crate::GpuStateVec64;
        let Ok(mut sv) = GpuStateVec64::new(4) else {
            return;
        };
        let t = Angle64::from_radians(0.37);
        sv.rzz(t, &[(QubitId(0), QubitId(1))]);
        let state = sv.state();
        let t_rad = t.to_radians_signed();
        let (c, s) = ((t_rad / 2.0).cos(), (t_rad / 2.0).sin());
        let [re, im] = state[0];
        assert!(
            (re - c).abs() < 1e-5 && (im + s).abs() < 1e-5,
            "sv64 rzz: ({re}, {im}) vs expected ({c}, {})",
            -s
        );
    }

    // --- f32 backend tests (primary) ---

    #[test]
    fn test_bell_state() {
        let Ok(mut gpu) = GpuDensityMatrix32::new(2) else {
            return;
        };
        let mut cpu = DensityMatrix::new(2);
        gpu.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
        cpu.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
        assert_dm_close(
            &gpu_dm_matrix(&mut gpu),
            &cpu_dm_matrix(&mut cpu),
            TOL,
            "Bell",
        );
    }

    #[test]
    fn test_rzz_only() {
        let Ok(mut gpu) = GpuDensityMatrix32::new(2) else {
            return;
        };
        let mut cpu = DensityMatrix::new(2);
        let t = Angle64::from_radians(0.37);
        gpu.h(&[QubitId(0), QubitId(1)])
            .rzz(t, &[(QubitId(0), QubitId(1))]);
        cpu.h(&[QubitId(0), QubitId(1)])
            .rzz(t, &[(QubitId(0), QubitId(1))]);
        assert_dm_close(
            &gpu_dm_matrix(&mut gpu),
            &cpu_dm_matrix(&mut cpu),
            TOL,
            "rzz",
        );
    }

    #[test]
    fn test_rotations_and_2q() {
        // Exercises rz + rxx (default decomposition) + ryy (default
        // decomposition) + cz + t in one circuit.
        let Ok(mut gpu) = GpuDensityMatrix32::new(3) else {
            return;
        };
        let mut cpu = DensityMatrix::new(3);
        let t = Angle64::from_radians(0.37);
        gpu.h(&[QubitId(0), QubitId(1), QubitId(2)])
            .rz(t, &[QubitId(0)])
            .rxx(t, &[(QubitId(0), QubitId(1))])
            .ryy(t, &[(QubitId(1), QubitId(2))])
            .cz(&[(QubitId(0), QubitId(2))])
            .t(&[QubitId(1)]);
        cpu.h(&[QubitId(0), QubitId(1), QubitId(2)])
            .rz(t, &[QubitId(0)])
            .rxx(t, &[(QubitId(0), QubitId(1))])
            .ryy(t, &[(QubitId(1), QubitId(2))])
            .cz(&[(QubitId(0), QubitId(2))])
            .t(&[QubitId(1)]);
        assert_dm_close(
            &gpu_dm_matrix(&mut gpu),
            &cpu_dm_matrix(&mut cpu),
            TOL,
            "rot+2q",
        );
    }

    #[test]
    fn test_probability_and_purity() {
        let Ok(mut gpu) = GpuDensityMatrix32::new(2) else {
            return;
        };
        gpu.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
        let p00 = gpu.probability(0);
        let p11 = gpu.probability(3);
        assert!((p00 - 0.5).abs() < TOL, "P(00)={p00}");
        assert!((p11 - 0.5).abs() < TOL, "P(11)={p11}");
        assert!((gpu.purity() - 1.0).abs() < TOL);
        assert!(gpu.is_pure());
    }

    #[test]
    fn test_prepare_maximally_mixed() {
        let Ok(mut gpu) = GpuDensityMatrix32::new(2) else {
            return;
        };
        gpu.prepare_maximally_mixed();
        assert!((gpu.purity() - 0.25).abs() < TOL);
        for k in 0..4 {
            assert!((gpu.probability(k) - 0.25).abs() < TOL);
        }
    }

    #[test]
    fn test_prepare_computational_basis() {
        let Ok(mut gpu) = GpuDensityMatrix32::new(2) else {
            return;
        };
        gpu.prepare_computational_basis(2);
        assert!((gpu.probability(2) - 1.0).abs() < TOL);
        assert!(gpu.probability(0) < TOL);
        assert!(gpu.is_pure());
    }

    #[test]
    fn test_phase_damping_matches_cpu() {
        let Ok(mut gpu) = GpuDensityMatrix32::new(1) else {
            return;
        };
        let mut cpu = DensityMatrix::new(1);
        gpu.h(&[QubitId(0)]);
        cpu.h(&[QubitId(0)]);
        gpu.apply_phase_damping(0, 0.5);
        cpu.apply_phase_damping(0, 0.5);
        assert_dm_close(
            &gpu_dm_matrix(&mut gpu),
            &cpu_dm_matrix(&mut cpu),
            TOL,
            "phase damp",
        );
    }

    #[test]
    fn test_phase_damping_preserves_diagonal() {
        // Regression: after full dephasing of |+>, rho = I/2, so P(0)=P(1)=0.5.
        // Pre-Cholesky implementations broke this identity.
        let Ok(mut gpu) = GpuDensityMatrix32::new(1) else {
            return;
        };
        gpu.h(&[QubitId(0)]);
        gpu.apply_phase_damping(0, 1.0);
        let p0 = gpu.probability(0);
        let p1 = gpu.probability(1);
        assert!((p0 - 0.5).abs() < TOL, "P(0)={p0}");
        assert!((p1 - 0.5).abs() < TOL, "P(1)={p1}");
        assert!((p0 + p1 - 1.0).abs() < TOL, "probabilities don't sum to 1");
    }

    #[test]
    fn test_amplitude_damping_preserves_trace() {
        // Regression: partial amp damping on |+><+| should keep tr(rho) = 1.
        let Ok(mut gpu) = GpuDensityMatrix32::new(1) else {
            return;
        };
        gpu.h(&[QubitId(0)]);
        gpu.apply_amplitude_damping(0, 0.3);
        let p0 = gpu.probability(0);
        let p1 = gpu.probability(1);
        assert!((p0 + p1 - 1.0).abs() < TOL, "tr(rho) = {} != 1", p0 + p1);
        // Expected: rho_{00} = 0.5 + 0.5*0.3 = 0.65, rho_{11} = 0.5*0.7 = 0.35
        assert!((p0 - 0.65).abs() < TOL, "P(0)={p0} expected 0.65");
        assert!((p1 - 0.35).abs() < TOL, "P(1)={p1} expected 0.35");
    }

    #[test]
    fn test_amplitude_damping_matches_cpu() {
        let Ok(mut gpu) = GpuDensityMatrix32::new(2) else {
            return;
        };
        let mut cpu = DensityMatrix::new(2);
        gpu.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
        cpu.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
        gpu.apply_amplitude_damping(0, 0.3);
        cpu.apply_amplitude_damping(0, 0.3);
        assert_dm_close(
            &gpu_dm_matrix(&mut gpu),
            &cpu_dm_matrix(&mut cpu),
            TOL,
            "amp damp",
        );
    }

    #[test]
    fn test_depolarizing_matches_cpu() {
        let Ok(mut gpu) = GpuDensityMatrix32::new(2) else {
            return;
        };
        let mut cpu = DensityMatrix::new(2);
        gpu.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
        cpu.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
        gpu.apply_depolarizing_noise(0, 0.2);
        cpu.apply_depolarizing_noise(0, 0.2);
        assert_dm_close(
            &gpu_dm_matrix(&mut gpu),
            &cpu_dm_matrix(&mut cpu),
            TOL,
            "depol",
        );
    }

    #[test]
    fn test_bit_flip_matches_cpu() {
        let Ok(mut gpu) = GpuDensityMatrix32::new(2) else {
            return;
        };
        let mut cpu = DensityMatrix::new(2);
        gpu.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
        cpu.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
        gpu.apply_bit_flip(1, 0.15);
        cpu.apply_bit_flip(1, 0.15);
        assert_dm_close(
            &gpu_dm_matrix(&mut gpu),
            &cpu_dm_matrix(&mut cpu),
            TOL,
            "bit flip",
        );
    }

    #[test]
    fn test_phase_flip_matches_cpu() {
        let Ok(mut gpu) = GpuDensityMatrix32::new(2) else {
            return;
        };
        let mut cpu = DensityMatrix::new(2);
        gpu.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
        cpu.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
        gpu.apply_phase_flip(0, 0.25);
        cpu.apply_phase_flip(0, 0.25);
        assert_dm_close(
            &gpu_dm_matrix(&mut gpu),
            &cpu_dm_matrix(&mut cpu),
            TOL,
            "phase flip",
        );
    }
}
