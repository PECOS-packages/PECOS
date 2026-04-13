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

//! Phase 1: numerical Pauli-Lindblad synthesis for identity gates.
//!
//! For an identity gate (`H_g = 0`) the interaction-frame Lindbladian is
//! time-independent and `Omega_1 = L * tau_g` is exact to first order in the
//! Magnus expansion. For 1-qubit identity with amplitude damping plus pure
//! dephasing the first-order result is already **exact and
//! non-perturbative** (paper arXiv:2502.03462 line 812); see
//! `design/lindblad_magnus_algorithm.md` section 3.

use crate::basis::{Pauli1, PauliString};
use crate::gate::Gate;
use crate::matrix::{self, Matrix};
use crate::pauli_lindblad::PauliLindbladModel;

/// Synthesize a 1-qubit Pauli-Lindblad model from an identity gate.
/// Supports are `{X, Y, Z}`; rates come from the linear system
/// `alpha_b * tau_g = 2 sum_{k != I} lambda_k * <b,k>_sp`.
pub fn synthesize_identity_1q(gate: &Gate) -> PauliLindbladModel {
    assert_eq!(gate.num_qubits, 1, "synthesize_identity_1q requires 1 qubit");
    assert!(
        is_zero_hamiltonian(&gate.ideal.hamiltonian),
        "synthesize_identity_1q requires H_g = 0 (identity gate)",
    );

    let alpha_x = alpha_rate(&gate.noise, Pauli1::X);
    let alpha_y = alpha_rate(&gate.noise, Pauli1::Y);
    let alpha_z = alpha_rate(&gate.noise, Pauli1::Z);

    let tau = gate.tau_g;
    let lambda_x = (alpha_y + alpha_z - alpha_x) * tau / 4.0;
    let lambda_y = (alpha_x + alpha_z - alpha_y) * tau / 4.0;
    let lambda_z = (alpha_x + alpha_y - alpha_z) * tau / 4.0;

    let supports = vec![
        PauliString::single(Pauli1::X),
        PauliString::single(Pauli1::Y),
        PauliString::single(Pauli1::Z),
    ];
    let rates = vec![clip_negative(lambda_x), clip_negative(lambda_y), clip_negative(lambda_z)];
    PauliLindbladModel::new(supports, rates)
}

/// Compute the Pauli-basis rate `alpha_b = -Tr(P_b * L(P_b)) / d` for a
/// 1-qubit Pauli. Units: 1/time.
fn alpha_rate(noise: &crate::lindbladian::Lindbladian, p: Pauli1) -> f64 {
    let d = noise.d;
    let p_mat = matrix::pauli_1q(p);
    let l_p = noise.apply(&p_mat);
    let inner = matrix::trace(&matrix::matmul(&p_mat, &l_p, d), d);
    // For Hermitian P and CPTP L, alpha_b is real.
    -inner.re / d as f64
}

fn is_zero_hamiltonian(h: &Matrix) -> bool {
    h.iter().all(|c| c.re.abs() < 1e-14 && c.im.abs() < 1e-14)
}

/// Phase 1 positivity policy: clip tiny negatives to zero; panic on large
/// negatives to surface bugs. Revisit in Phase 2.
fn clip_negative(lambda: f64) -> f64 {
    if lambda < -1e-10 {
        panic!("PauliLindbladModel rate unexpectedly negative: {}", lambda);
    }
    lambda.max(0.0)
}
