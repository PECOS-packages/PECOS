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

//! Phases 1-3: numerical Pauli-Lindblad synthesis for arbitrary-qubit
//! gates via interaction-frame transform + Simpson's rule + Walsh-Hadamard
//! inversion.
//!
//! Entry points:
//! - [`synthesize_identity_1q`]: fast path for 1Q `H_g = 0` (Phase 1;
//!   exact + non-perturbative under AD + PD).
//! - [`synthesize_numerical_1q`]: 1Q, general `H_g`, Simpson integrand.
//! - [`synthesize_numerical`]: n-qubit, general `H_g`, Simpson integrand,
//!   Walsh-Hadamard rate recovery.
//!
//! See `design/lindblad_magnus_algorithm.md` for the math spec and paper
//! arXiv:2502.03462 for closed-form fixtures.

use num_complex::Complex64;

use crate::basis::{Pauli1, PauliString};
use crate::gate::Gate;
use crate::lindbladian::Lindbladian;
use crate::matrix::{self, Matrix};
use crate::pauli_lindblad::PauliLindbladModel;

const PHASE1_PAULIS: [Pauli1; 3] = [Pauli1::X, Pauli1::Y, Pauli1::Z];

/// Default number of Simpson intervals for time integration. Composite
/// Simpson's 1/3 rule, order-4 accurate. 1024 gives ~1e-12 for smooth
/// integrands on a bounded interval (sinusoidal at gate frequency up to a
/// few cycles).
pub const DEFAULT_N_STEPS: usize = 1024;

/// Synthesize a 1-qubit Pauli-Lindblad model from an identity gate. Fast
/// path: identity gate (`H_g = 0`) => interaction-frame Lindbladian is
/// constant and `Omega_1 = L * tau_g`, so no time integration needed.
pub fn synthesize_identity_1q(gate: &Gate) -> PauliLindbladModel {
    assert_eq!(
        gate.num_qubits, 1,
        "synthesize_identity_1q requires 1 qubit"
    );
    assert!(
        is_zero_matrix(&gate.ideal.hamiltonian),
        "synthesize_identity_1q requires H_g = 0",
    );
    let tau = gate.tau_g;
    let paulis: Vec<PauliString> = PHASE1_PAULIS
        .iter()
        .map(|&p| PauliString::single(p))
        .collect();
    let alphas: Vec<f64> = PHASE1_PAULIS
        .iter()
        .map(|&p| constant_alpha(&gate.noise, p) * tau)
        .collect();
    model_from_alphas_walsh(paulis, alphas, 1)
}

/// Synthesize a 1-qubit Pauli-Lindblad model from an arbitrary 1-qubit
/// gate via Simpson's rule on `Omega_1 = int_0^{tau_g} L_I(t) dt`. Works
/// for identity (reduces to the Phase 1 result) and for gates like
/// `X_theta`, `Y_theta`, `Z_theta`.
pub fn synthesize_numerical_1q(gate: &Gate, n_steps: usize) -> PauliLindbladModel {
    assert_eq!(
        gate.num_qubits, 1,
        "synthesize_numerical_1q requires 1 qubit"
    );
    synthesize_numerical(gate, n_steps)
}

/// Default number of time slices for `synthesize_superop`. Midpoint-rule
/// propagator per slice is second-order accurate; `N=128` gives ~`1e-10`
/// precision for single-oscillation gates.
pub const DEFAULT_N_SLICES: usize = 128;

/// **General** synthesis for any gate with any combination of coherent
/// and dissipative noise via time-slicing of the interaction-frame
/// Lindblad superoperator.
///
/// For each midpoint `t_k`, builds `L_I(t_k) = U_g^dag(t_k) L_noise
/// U_g(t_k)`, exponentiates to get a per-slice propagator
/// `exp(L_I(t_k) * dt)` and multiplies them left-to-right to form
/// `U_total`. Applies to each `vec(P_b)`, extracts Pauli fidelity, inverts
/// via Walsh-Hadamard.
///
/// This is the only path that handles **gates with `H_g != 0` AND
/// simultaneous coherent + dissipative noise** -- the general case that
/// `synthesize_numerical` (dissipative leading-order only) and
/// `synthesize_exact_unitary` (coherent, asserts no c_ops) don't cover.
///
/// Cost: `n_slices` per-slice superoperator builds + exps at size
/// `d^2 x d^2`. For 2Q gates `d^2=16`, comfortably sub-second at
/// `n_slices=128`.
pub fn synthesize_superop(gate: &Gate, n_slices: usize) -> PauliLindbladModel {
    assert!(n_slices >= 1, "n_slices must be >= 1");
    let n = gate.num_qubits;
    let d = 1usize << n;
    let d2 = d * d;
    let tau = gate.tau_g;
    let dt = tau / n_slices as f64;

    let h_g = &gate.ideal.hamiltonian;
    let h_delta = &gate.noise.hamiltonian;

    // U_total = prod_k exp(L_I(t_k) * dt), built left-to-right (newest leftmost).
    let mut u_total = matrix::identity(d2);
    for k in 0..n_slices {
        let t_mid = (k as f64 + 0.5) * dt;
        let u_g = matrix::exp_minus_i_h_t(h_g, d, t_mid);
        let u_g_dag = matrix::dag(&u_g, d);

        // Transform noise operators to the interaction frame at t_mid.
        let h_i = matrix::matmul(&matrix::matmul(&u_g_dag, h_delta, d), &u_g, d);
        let collapse_i: Vec<(Matrix, f64)> = gate
            .noise
            .collapse
            .iter()
            .map(|(c, g)| (matrix::matmul(&matrix::matmul(&u_g_dag, c, d), &u_g, d), *g))
            .collect();

        // L_I(t_mid) superop.
        let lind_i = Lindbladian::new(d, h_i, collapse_i);
        let l_super = lind_i.superoperator();

        // Per-slice propagator and accumulate.
        let u_slice = matrix::expm(&matrix::scale(&l_super, Complex64::new(dt, 0.0)), d2);
        u_total = matrix::matmul(&u_slice, &u_total, d2);
    }

    // Apply to each Pauli, extract fidelities, invert.
    let paulis = PauliString::enumerate_nonidentity(n);
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
    model_from_alphas_walsh(paulis, alphas, n)
}

/// Synthesize a Pauli-Lindblad model from **mixed coherent + dissipative
/// noise** on a time-independent ideal Hamiltonian (currently: identity
/// gate, `H_g = 0`) via the full Lindblad superoperator path.
///
/// Builds `L_super` (`d^2 x d^2`), exponentiates to get the channel
/// `exp(L_super * tau_g)`, applies to each `vec(P_b)`, extracts Pauli
/// fidelity `f_b = (1/d) tr(P_b * Lambda(P_b))`, then inverts via
/// Walsh-Hadamard. Unifies the coherent and dissipative paths for the
/// identity case: matches [`synthesize_identity_1q`] for pure AD+PD,
/// matches [`synthesize_exact_unitary`] for pure coherent noise on
/// identity, and handles **both at once** (the case the other paths
/// reject).
///
/// Requires `gate.ideal.hamiltonian` to be (numerically) zero -- for
/// non-trivial `H_g` the interaction-frame Lindbladian is time-dependent
/// and requires either Magnus order >= 1 time-ordering (existing
/// `synthesize_numerical` path for linear-order dissipative) or
/// time-slicing (future work).
pub fn synthesize_superop_identity(gate: &Gate) -> PauliLindbladModel {
    let n = gate.num_qubits;
    let d = 1usize << n;
    assert!(
        is_zero_matrix(&gate.ideal.hamiltonian),
        "synthesize_superop_identity requires H_g = 0 (time-independent L_I)"
    );
    let tau = gate.tau_g;
    let l_super = gate.noise.superoperator();
    // channel = exp(L_super * tau_g)
    let channel = matrix::expm(&matrix::scale(&l_super, Complex64::new(tau, 0.0)), d * d);

    let paulis = PauliString::enumerate_nonidentity(n);
    let alphas: Vec<f64> = paulis
        .iter()
        .map(|p| {
            let p_mat = matrix::pauli_string_mat(p);
            let vec_p = matrix::vec_of(&p_mat, d);
            let vec_applied = matrix::matvec(&channel, &vec_p, d * d);
            let applied = matrix::unvec(&vec_applied, d);
            let inner = matrix::trace(&matrix::matmul(&p_mat, &applied, d), d);
            let f_b = inner.re / d as f64;
            assert!(
                f_b > 0.1,
                "Pauli fidelity {} too low for {:?}; noise outside weak-coupling regime",
                f_b,
                p,
            );
            -f_b.ln()
        })
        .collect();
    model_from_alphas_walsh(paulis, alphas, n)
}

/// Synthesize a Pauli-Lindblad model for a gate with **purely coherent
/// noise** (no collapse operators) via the exact error-unitary path.
///
/// For coherent noise the Pauli rates are quadratic in the perturbation
/// strength (see `design/lindblad_magnus_algorithm.md` section 4.5). The
/// linear-order [`synthesize_numerical`] path gives `alpha_b = 0` for
/// coherent noise because `Tr(P_b L(P_b)) = 0` when `L` is a single
/// commutator. This function computes the exact error unitary
/// `U_err = U_ideal^dag * U_full` and extracts Pauli fidelities directly.
///
/// Requires `gate.noise.collapse` to be empty. Use
/// [`synthesize_numerical`] for dissipative noise (AD, PD).
pub fn synthesize_exact_unitary(gate: &Gate) -> PauliLindbladModel {
    assert!(
        gate.noise.collapse.is_empty(),
        "synthesize_exact_unitary requires purely coherent noise (no c_ops)"
    );
    let n = gate.num_qubits;
    let d = 1usize << n;
    let tau = gate.tau_g;

    let h_g = &gate.ideal.hamiltonian;
    let h_delta = &gate.noise.hamiltonian;
    let h_full = matrix::add(h_g, h_delta);

    let u_full = matrix::expm(&matrix::scale(&h_full, Complex64::new(0.0, -tau)), d);
    let u_ideal = matrix::expm(&matrix::scale(h_g, Complex64::new(0.0, -tau)), d);
    let u_ideal_dag = matrix::dag(&u_ideal, d);
    let u_err = matrix::matmul(&u_ideal_dag, &u_full, d);
    let u_err_dag = matrix::dag(&u_err, d);

    let paulis = PauliString::enumerate_nonidentity(n);
    let alphas: Vec<f64> = paulis
        .iter()
        .map(|p| {
            let p_mat = matrix::pauli_string_mat(p);
            let up = matrix::matmul(&u_err, &p_mat, d);
            let upudag = matrix::matmul(&up, &u_err_dag, d);
            let inner = matrix::trace(&matrix::matmul(&p_mat, &upudag, d), d);
            let f_b = inner.re / d as f64;
            // For weak noise f_b ~ 1. Use alpha_b = -ln(f_b); equal to
            // (1 - f_b) at leading order. Panic if f_b drifts out of the
            // weak-noise regime (< 0.1 means you are outside the Magnus
            // convergence radius and the PL model is not a good fit).
            assert!(
                f_b > 0.1,
                "Pauli fidelity {} for {:?} below weak-noise threshold; noise too strong for PL model",
                f_b,
                p,
            );
            -f_b.ln()
        })
        .collect();

    model_from_alphas_walsh(paulis, alphas, n)
}

/// Synthesize a Pauli-Lindblad model from an arbitrary gate. Enumerates all
/// non-identity Paulis on `gate.num_qubits`, integrates `alpha_b * tau_g`
/// for each via Simpson's rule on the interaction-frame Lindbladian, and
/// inverts via Walsh-Hadamard.
pub fn synthesize_numerical(gate: &Gate, n_steps: usize) -> PauliLindbladModel {
    assert!(
        n_steps >= 2 && n_steps.is_multiple_of(2),
        "n_steps must be even and >= 2, got {}",
        n_steps
    );
    let n = gate.num_qubits;
    let paulis = PauliString::enumerate_nonidentity(n);
    if is_zero_matrix(&gate.ideal.hamiltonian) {
        let tau = gate.tau_g;
        let alphas: Vec<f64> = paulis
            .iter()
            .map(|p| constant_alpha_pauli_string(&gate.noise, p) * tau)
            .collect();
        return model_from_alphas_walsh(paulis, alphas, n);
    }

    let alphas: Vec<f64> = paulis
        .iter()
        .map(|p| integrated_alpha(gate, p, n_steps))
        .collect();
    model_from_alphas_walsh(paulis, alphas, n)
}

/// `alpha_b = -Tr(P_b L(P_b)) / d` for time-independent L. Units: 1/time.
fn constant_alpha(noise: &Lindbladian, p: Pauli1) -> f64 {
    let d = noise.d;
    let p_mat = matrix::pauli_1q(p);
    let l_p = noise.apply(&p_mat);
    let inner = matrix::trace(&matrix::matmul(&p_mat, &l_p, d), d);
    -inner.re / d as f64
}

/// `alpha_b = -Tr(P_b L(P_b)) / d` for a time-independent Lindbladian and
/// arbitrary-qubit Pauli string. Units: 1/time.
fn constant_alpha_pauli_string(noise: &Lindbladian, p: &PauliString) -> f64 {
    let d = noise.d;
    let p_mat = matrix::pauli_string_mat(p);
    let l_p = noise.apply(&p_mat);
    let inner = matrix::trace(&matrix::matmul(&p_mat, &l_p, d), d);
    -inner.re / d as f64
}

/// Integrated `alpha_b * tau_g = -Tr(P_b * Omega_1(P_b)) / d` via Simpson's
/// rule on `[0, tau_g]`. Works for any `n_qubits`.
fn integrated_alpha(gate: &Gate, p: &PauliString, n_steps: usize) -> f64 {
    let n = gate.num_qubits;
    assert_eq!(p.num_qubits(), n);
    let d = 1usize << n;
    let p_mat = matrix::pauli_string_mat(p);
    let h_g = &gate.ideal.hamiltonian;
    let tau = gate.tau_g;
    let h_step = tau / n_steps as f64;

    // integrand(t) = -Tr(P_b * L_I(t)(P_b)).re / d
    //              = -Tr(P_b * U_g^†(t) L(U_g(t) P_b U_g^†(t)) U_g(t)) / d
    let integrand = |t: f64| -> f64 {
        let u = matrix::exp_minus_i_h_t(h_g, d, t);
        let u_dag = matrix::dag(&u, d);
        let rotated = matrix::matmul(&matrix::matmul(&u, &p_mat, d), &u_dag, d);
        let l_rotated = gate.noise.apply(&rotated);
        let l_i_pb = matrix::matmul(&matrix::matmul(&u_dag, &l_rotated, d), &u, d);
        let inner = matrix::trace(&matrix::matmul(&p_mat, &l_i_pb, d), d);
        -inner.re / d as f64
    };

    // Composite Simpson's 1/3 rule. Weights: 1, 4, 2, 4, 2, ..., 4, 1.
    let mut s = integrand(0.0) + integrand(tau);
    for k in 1..n_steps {
        let t = k as f64 * h_step;
        let w = if k % 2 == 1 { 4.0 } else { 2.0 };
        s += w * integrand(t);
    }
    s * h_step / 3.0
}

/// Walsh-Hadamard inversion:
///   `lambda_k = -(1/4^n) * sum_{b non-identity} (-1)^{<k,b>_sp} alpha_b`
/// (see `design/lindblad_magnus_algorithm.md` step 4). alpha_I = 0 is
/// implicit.
fn model_from_alphas_walsh(
    paulis: Vec<PauliString>,
    alphas: Vec<f64>,
    n_qubits: usize,
) -> PauliLindbladModel {
    assert_eq!(paulis.len(), alphas.len());
    let norm = 1.0 / (1usize << (2 * n_qubits)) as f64;
    let rates: Vec<f64> = paulis
        .iter()
        .map(|k| {
            let mut s = 0.0;
            for (b, &alpha_b) in paulis.iter().zip(alphas.iter()) {
                let sign = if k.symplectic_product(b) == 0 {
                    1.0
                } else {
                    -1.0
                };
                s += sign * alpha_b;
            }
            clip_negative(-norm * s)
        })
        .collect();
    PauliLindbladModel::new(paulis, rates)
}

fn is_zero_matrix(m: &Matrix) -> bool {
    m.iter()
        .all(|c: &Complex64| c.re.abs() < 1e-14 && c.im.abs() < 1e-14)
}

/// Phase 1 positivity policy: clip tiny negatives to zero; panic on large
/// negatives so bugs surface. Revisit in Phase 3 with per-user policy.
fn clip_negative(lambda: f64) -> f64 {
    if lambda < -1e-8 {
        panic!("PauliLindbladModel rate unexpectedly negative: {}", lambda);
    }
    lambda.max(0.0)
}
