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

//! Time-dependent (time-convolutionless, TCL) Lindbladian support for
//! non-Markovian dynamics.
//!
//! # Scope
//!
//! This module supports **time-local** non-Markovian master equations of
//! the form
//!
//! ```text
//! drho/dt = -i [H_delta(t), rho] + sum_j gamma_j(t) * D[c_j] rho
//! ```
//!
//! where rates `gamma_j(t)` and coherent noise `H_delta(t)` can vary
//! arbitrarily with time. This covers:
//!
//! - **1/f dephasing**: `gamma_phi(t) = gamma_0 * A / (A + t/t_c)` (leads to
//!   non-exponential T_2 decay).
//! - **Gaussian coherence decay**: `gamma_phi(t) ∝ t` gives `exp(-(t/T_2)^2)`.
//! - **Pulse-shape-dependent dephasing**: `gamma_phi(t) ∝ |pulse(t)|^2`.
//! - **Coloured coherent noise**: `H_delta(t) = (delta_0 cos(omega_d t) / 2) Z`.
//!
//! # Out of scope
//!
//! Time-nonlocal (true memory-kernel) master equations like
//! Nakajima-Zwanzig `drho/dt = int_0^t K(t-s) rho(s) ds` require
//! convolution integrals over the history of rho and are not supported.
//! This is a genuine structural limit of the TCL framework.

use std::sync::Arc;

use num_complex::Complex64;

use crate::lindbladian::Lindbladian;
use crate::matrix::Matrix;

/// Closure type for time-dependent scalar rates.
pub type RateFn = Arc<dyn Fn(f64) -> f64 + Send + Sync>;

/// Closure type for time-dependent Hermitian operators (d x d matrix).
pub type HermitianFn = Arc<dyn Fn(f64) -> Matrix + Send + Sync>;

/// Time-convolutionless (TCL) non-Markovian Lindbladian.
///
/// Stores a Hilbert-space dimension, a time-dependent coherent noise
/// Hamiltonian (which may be constant zero), and a list of collapse
/// operators with time-dependent rates. Evaluation at time `t` returns a
/// standard [`Lindbladian`] snapshot.
#[derive(Clone)]
pub struct TimeDepLindbladian {
    pub d: usize,
    pub hamiltonian_fn: HermitianFn,
    pub collapse_fns: Vec<(Matrix, RateFn)>,
}

impl TimeDepLindbladian {
    /// Construct with a constant Hamiltonian and per-operator time-dep rates.
    pub fn with_static_hamiltonian(
        d: usize,
        hamiltonian: Matrix,
        collapse_fns: Vec<(Matrix, RateFn)>,
    ) -> Self {
        assert_eq!(hamiltonian.len(), d * d, "hamiltonian wrong shape");
        let h_clone = hamiltonian.clone();
        let hamiltonian_fn: HermitianFn = Arc::new(move |_t| h_clone.clone());
        Self {
            d,
            hamiltonian_fn,
            collapse_fns,
        }
    }

    /// Construct with fully time-dependent Hamiltonian and rates.
    pub fn new(d: usize, hamiltonian_fn: HermitianFn, collapse_fns: Vec<(Matrix, RateFn)>) -> Self {
        Self {
            d,
            hamiltonian_fn,
            collapse_fns,
        }
    }

    /// Evaluate at time `t`, producing a static [`Lindbladian`] snapshot.
    pub fn at(&self, t: f64) -> Lindbladian {
        let h = (self.hamiltonian_fn)(t);
        let collapse: Vec<(Matrix, f64)> = self
            .collapse_fns
            .iter()
            .map(|(op, rate_fn)| (op.clone(), rate_fn(t)))
            .collect();
        Lindbladian::new(self.d, h, collapse)
    }
}

impl std::fmt::Debug for TimeDepLindbladian {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimeDepLindbladian")
            .field("d", &self.d)
            .field("num_collapse", &self.collapse_fns.len())
            .finish()
    }
}

// ============================================================================
// Synthesis entry point
// ============================================================================

use crate::basis::PauliString;
use crate::matrix;
use crate::pauli_lindblad::PauliLindbladModel;

/// Synthesize a Pauli-Lindblad model from a time-dependent noise
/// Lindbladian via time-slicing. Each slice's superoperator is the
/// interaction-frame-transformed L(t_mid) snapshot.
///
/// `num_qubits` gives the Pauli enumeration dimension; `ideal_h` is the
/// (time-independent) ideal gate Hamiltonian used for the interaction
/// frame; `noise_td` is the time-dependent noise.
pub fn synthesize_superop_time_dep(
    num_qubits: usize,
    ideal_h: &Matrix,
    noise_td: &TimeDepLindbladian,
    tau_g: f64,
    n_slices: usize,
) -> PauliLindbladModel {
    assert!(n_slices >= 1);
    let d = 1usize << num_qubits;
    assert_eq!(ideal_h.len(), d * d);
    assert_eq!(noise_td.d, d);
    let d2 = d * d;
    let dt = tau_g / n_slices as f64;

    // Product of per-slice propagators, newest on left.
    let mut u_total = matrix::identity(d2);
    for k in 0..n_slices {
        let t_mid = (k as f64 + 0.5) * dt;

        // Evaluate time-dependent noise at this slice.
        let lind_snap = noise_td.at(t_mid);

        // Interaction-frame transform.
        let u_g = matrix::exp_minus_i_h_t(ideal_h, d, t_mid);
        let u_g_dag = matrix::dag(&u_g, d);
        let h_i = matrix::matmul(
            &matrix::matmul(&u_g_dag, &lind_snap.hamiltonian, d),
            &u_g,
            d,
        );
        let collapse_i: Vec<(Matrix, f64)> = lind_snap
            .collapse
            .iter()
            .map(|(c, g)| (matrix::matmul(&matrix::matmul(&u_g_dag, c, d), &u_g, d), *g))
            .collect();

        let lind_i = Lindbladian::new(d, h_i, collapse_i);
        let l_super = lind_i.superoperator();
        let u_slice = matrix::expm(&matrix::scale(&l_super, Complex64::new(dt, 0.0)), d2);
        u_total = matrix::matmul(&u_slice, &u_total, d2);
    }

    // Apply to each Pauli, extract fidelities, Walsh-Hadamard inversion.
    let paulis = PauliString::enumerate_nonidentity(num_qubits);
    let alphas: Vec<f64> = paulis
        .iter()
        .map(|p| {
            let p_mat = matrix::pauli_string_mat(p);
            let vec_p = matrix::vec_of(&p_mat, d);
            let vec_applied = matrix::matvec(&u_total, &vec_p, d2);
            let applied = matrix::unvec(&vec_applied, d);
            let inner = matrix::trace(&matrix::matmul(&p_mat, &applied, d), d);
            let f_b = inner.re / d as f64;
            assert!(
                f_b > 0.1,
                "Pauli fidelity {} too low for {:?}; noise outside weak regime",
                f_b,
                p,
            );
            -f_b.ln()
        })
        .collect();
    walsh_hadamard_invert(paulis, alphas, num_qubits)
}

fn walsh_hadamard_invert(
    paulis: Vec<PauliString>,
    alphas: Vec<f64>,
    n_qubits: usize,
) -> PauliLindbladModel {
    let norm = 1.0 / (1usize << (2 * n_qubits)) as f64;
    let rates: Vec<f64> = paulis
        .iter()
        .map(|k| {
            let mut s = 0.0;
            for (b, &a) in paulis.iter().zip(alphas.iter()) {
                let sign = if k.symplectic_product(b) == 0 {
                    1.0
                } else {
                    -1.0
                };
                s += sign * a;
            }
            (-norm * s).max(0.0)
        })
        .collect();
    PauliLindbladModel::new(paulis, rates)
}
