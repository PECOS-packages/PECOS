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

//! Smoke test for `Gate::from_hamiltonian` — the general escape hatch
//! that lets users build gates not in the named catalog. Exercises the
//! `matrix::expm` fallback for a non-structured 4x4 Hamiltonian.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use pecos_lindblad::matrix::{self, Matrix};
use pecos_lindblad::{
    DEFAULT_N_STEPS, Gate, Lindbladian, Pauli1, PauliString, synthesize_exact_unitary,
    synthesize_numerical,
};

/// iSWAP_theta generator: `H_g = (omega/2)(XX + YY)` (4x4 Hermitian,
/// non-diagonal, non-block-diagonal in computational basis -- hits
/// the `expm` fallback path).
fn iswap_hamiltonian(omega: f64) -> Matrix {
    let x = matrix::pauli_1q(Pauli1::X);
    let y = matrix::pauli_1q(Pauli1::Y);
    let xx = matrix::kron(&x, &x, 2, 2);
    let yy = matrix::kron(&y, &y, 2, 2);
    matrix::scale(&matrix::add(&xx, &yy), Complex64::new(omega / 2.0, 0.0))
}

#[test]
fn iswap_theta_reduces_to_identity_at_zero_theta() {
    // At theta=0 (or tau_g=0), the gate Hamiltonian has no time to act,
    // so the result should match identity+AD+PD rates exactly.
    let d = 2;
    let i2 = matrix::identity(2);
    let sm = matrix::sigma_minus();
    let z1 = matrix::pauli_1q(Pauli1::Z);
    let sm_l = matrix::kron(&sm, &i2, 2, 2);
    let sm_r = matrix::kron(&i2, &sm, 2, 2);
    let z_l = matrix::kron(&z1, &i2, 2, 2);
    let z_r = matrix::kron(&i2, &z1, 2, 2);
    let beta_down = 1e-4;
    let beta_phi = 2e-4;
    let _ = d;

    let collapse: Vec<(Matrix, f64)> = vec![
        (sm_l, beta_down),
        (sm_r, beta_down),
        (z_l, beta_phi / 2.0),
        (z_r, beta_phi / 2.0),
    ];
    let noise = Lindbladian::new(4, matrix::zeros(4), collapse);
    let tau_g = 0.0; // Degenerate duration.
    let h_iswap = iswap_hamiltonian(1.0);
    let gate = Gate::from_hamiltonian("iswap_0", 2, h_iswap, noise, tau_g);

    let pl = synthesize_numerical(&gate, DEFAULT_N_STEPS);
    // At tau_g=0 everything should vanish.
    for ps in PauliString::enumerate_nonidentity(2) {
        assert_abs_diff_eq!(pl.rate(&ps), 0.0, epsilon = 1e-14);
    }
}

#[test]
fn custom_hamiltonian_coherent_noise_produces_nonzero_rates() {
    // Construct a non-structured 2Q Hamiltonian H_g = XX + YY (iSWAP
    // generator) with coherent IZ phase noise. Verify synthesis runs
    // and produces some non-zero rate (exercises expm fallback).
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_4;
    let tau_g = theta / omega;
    let h_g = iswap_hamiltonian(omega);

    let i2 = matrix::identity(2);
    let z1 = matrix::pauli_1q(Pauli1::Z);
    let iz = matrix::kron(&i2, &z1, 2, 2);
    let delta = 1e-4;
    let h_delta = matrix::scale(&iz, Complex64::new(delta / 2.0, 0.0));
    let noise = Lindbladian::new(4, h_delta, Vec::new());

    let gate = Gate::from_hamiltonian("iswap_xy", 2, h_g, noise, tau_g);
    let pl = synthesize_exact_unitary(&gate);

    // Some rate should be non-zero; total rate in correct order of magnitude.
    let total = pl.total_rate();
    let expected_scale = (delta / omega).powi(2);
    assert!(total > 0.0, "expected non-zero rates, got total={}", total);
    assert!(
        total < 10.0 * expected_scale,
        "total rate {} exceeds scale {} by >10x -- higher-order term leak?",
        total,
        expected_scale,
    );
}
