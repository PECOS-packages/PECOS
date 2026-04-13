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

//! Phase 1-2: numerical Pauli-Lindblad synthesis for 1-qubit gates.
//!
//! - `synthesize_identity_1q`: fast path for `H_g = 0` (Phase 1; identity
//!   under amplitude damping + pure dephasing is exact, non-perturbative).
//! - `synthesize_numerical_1q`: general 1-qubit path (Phase 2) via
//!   interaction-frame transform + Simpson's rule on Omega_1.
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

/// Default number of Simpson intervals for 1-qubit time integration.
/// Composite Simpson's 1/3 rule, order-4 accurate. 1024 gives ~1e-12 for
/// smooth integrands on a bounded interval (sinusoidal at frequency
/// `omega_x` up to a few cycles).
pub const DEFAULT_N_STEPS: usize = 1024;

/// Synthesize a 1-qubit Pauli-Lindblad model from an identity gate. Fast
/// path: identity gate (`H_g = 0`) => interaction-frame Lindbladian is
/// constant and `Omega_1 = L * tau_g`.
pub fn synthesize_identity_1q(gate: &Gate) -> PauliLindbladModel {
    assert_eq!(gate.num_qubits, 1, "synthesize_identity_1q requires 1 qubit");
    assert!(
        is_zero_matrix(&gate.ideal.hamiltonian),
        "synthesize_identity_1q requires H_g = 0",
    );
    let tau = gate.tau_g;
    let alphas = PHASE1_PAULIS.map(|p| constant_alpha(&gate.noise, p) * tau);
    model_from_alphas(alphas)
}

/// Synthesize a 1-qubit Pauli-Lindblad model from an arbitrary 1-qubit
/// gate via Simpson's rule on `Omega_1 = int_0^{tau_g} L_I(t) dt`. Works
/// for identity (reduces to the Phase 1 result) and for gates like
/// `X_theta`, `Y_theta`, `Z_theta`.
pub fn synthesize_numerical_1q(gate: &Gate, n_steps: usize) -> PauliLindbladModel {
    assert_eq!(gate.num_qubits, 1, "synthesize_numerical_1q requires 1 qubit");
    assert!(n_steps >= 2 && n_steps % 2 == 0, "n_steps must be even and >= 2, got {}", n_steps);
    let alphas = PHASE1_PAULIS.map(|p| integrated_alpha_1q(gate, p, n_steps));
    model_from_alphas(alphas)
}

/// `alpha_b = -Tr(P_b L(P_b)) / d` for time-independent L. Units: 1/time.
fn constant_alpha(noise: &Lindbladian, p: Pauli1) -> f64 {
    let d = noise.d;
    let p_mat = matrix::pauli_1q(p);
    let l_p = noise.apply(&p_mat);
    let inner = matrix::trace(&matrix::matmul(&p_mat, &l_p, d), d);
    -inner.re / d as f64
}

/// Integrated `alpha_b * tau_g = -Tr(P_b * Omega_1(P_b)) / d` via Simpson's
/// rule on `[0, tau_g]`.
fn integrated_alpha_1q(gate: &Gate, p: Pauli1, n_steps: usize) -> f64 {
    let d = 2;
    let p_mat = matrix::pauli_1q(p);
    let h_g = &gate.ideal.hamiltonian;
    let tau = gate.tau_g;
    let h_step = tau / n_steps as f64;

    // integrand(t) = -Tr(P_b * L_I(t)(P_b)).re / d
    //              = -Tr(P_b * U_g^†(t) L(U_g(t) P_b U_g^†(t)) U_g(t)) / d
    let integrand = |t: f64| -> f64 {
        let u = matrix::exp_minus_i_h_t_1q(h_g, t);
        let u_dag = matrix::dag(&u, d);
        // rotated = U_g(t) P_b U_g^†(t)
        let rotated = matrix::matmul(&matrix::matmul(&u, &p_mat, d), &u_dag, d);
        // L applied to rotated
        let l_rotated = gate.noise.apply(&rotated);
        // Conjugate back: U_g^†(t) L(rotated) U_g(t)
        let l_i_pb = matrix::matmul(&matrix::matmul(&u_dag, &l_rotated, d), &u, d);
        // -Tr(P_b * L_I(t)(P_b)) / d
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

fn model_from_alphas(alphas: [f64; 3]) -> PauliLindbladModel {
    // alphas = [alpha_X * tau_g, alpha_Y * tau_g, alpha_Z * tau_g].
    // Linear solve of alpha_b = 2 sum_{k != I} lambda_k <b,k>_sp:
    //   alpha_X = 2(lambda_Y + lambda_Z)
    //   alpha_Y = 2(lambda_X + lambda_Z)
    //   alpha_Z = 2(lambda_X + lambda_Y)
    let [ax, ay, az] = alphas;
    let lambda_x = (ay + az - ax) / 4.0;
    let lambda_y = (ax + az - ay) / 4.0;
    let lambda_z = (ax + ay - az) / 4.0;

    let supports = vec![
        PauliString::single(Pauli1::X),
        PauliString::single(Pauli1::Y),
        PauliString::single(Pauli1::Z),
    ];
    let rates = vec![clip_negative(lambda_x), clip_negative(lambda_y), clip_negative(lambda_z)];
    PauliLindbladModel::new(supports, rates)
}

fn is_zero_matrix(m: &Matrix) -> bool {
    m.iter().all(|c: &Complex64| c.re.abs() < 1e-14 && c.im.abs() < 1e-14)
}

/// Phase 1 positivity policy: clip tiny negatives to zero; panic on large
/// negatives so bugs surface. Revisit in Phase 3 with per-user policy.
fn clip_negative(lambda: f64) -> f64 {
    if lambda < -1e-8 {
        panic!("PauliLindbladModel rate unexpectedly negative: {}", lambda);
    }
    lambda.max(0.0)
}
