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

//! Tests for `synthesize_superop_identity` -- the unified path for
//! identity gates with mixed coherent + dissipative noise. Validates
//! consistency with:
//!
//! - `synthesize_identity_1q` (pure dissipative AD+PD)
//! - `synthesize_exact_unitary` (pure coherent)
//!
//! and exercises the **new** mixed case (both at once) that the other two
//! entry points reject or under-model.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use pecos_lindblad::matrix::{self, Matrix};
use pecos_lindblad::noise_models::{ad_pd_1q, coherent_phase_2q};
use pecos_lindblad::{
    Gate, Lindbladian, Pauli1, PauliString, synthesize_exact_unitary, synthesize_identity_1q,
    synthesize_superop_identity,
};

#[test]
fn superop_identity_matches_fast_ad_pd_1q() {
    // Pure AD+PD, 1Q identity. Superop path should match fast closed-form
    // path to machine precision.
    let t1 = 100.0;
    let t2 = 80.0;
    let tau_g = 5.0;
    let noise = ad_pd_1q(t1, t2);
    let gate = Gate::identity(1, noise, tau_g);

    let fast = synthesize_identity_1q(&gate);
    let superop = synthesize_superop_identity(&gate);

    for p in [Pauli1::X, Pauli1::Y, Pauli1::Z] {
        let key = PauliString::single(p);
        assert_abs_diff_eq!(fast.rate(&key), superop.rate(&key), epsilon = 1e-10);
    }
}

#[test]
fn superop_identity_matches_exact_unitary_for_pure_coherent() {
    // 2Q identity with pure coherent phase noise: both exact_unitary and
    // superop paths should agree.
    let tau_g = 10.0;
    let delta_iz = 1e-5;
    let delta_zi = 2e-5;
    let delta_zz = 5e-6;
    let noise = coherent_phase_2q(delta_iz, delta_zi, delta_zz);
    let gate = Gate::identity(2, noise, tau_g);

    let exact = synthesize_exact_unitary(&gate);
    let superop = synthesize_superop_identity(&gate);

    for ps in PauliString::enumerate_nonidentity(2) {
        assert_abs_diff_eq!(exact.rate(&ps), superop.rate(&ps), epsilon = 1e-10);
    }
}

#[test]
fn superop_identity_handles_mixed_coherent_and_dissipative_2q() {
    // THE new capability: simultaneous AD+PD AND coherent crosstalk on
    // an identity gate. Neither `synthesize_identity_1q` (1Q only) nor
    // `synthesize_exact_unitary` (coherent only) can handle this;
    // `synthesize_numerical` catches only the Omega_1 dissipative part.
    // Only `synthesize_superop_identity` gives the full answer.
    let d = 4;
    let tau_g = 1.0;
    let beta_down = 1e-4;
    let beta_phi = 2e-4;
    let delta_zz = 1e-3;

    // Hamiltonian part: (delta_zz / 2) ZZ coherent
    let i2 = matrix::identity(2);
    let z = matrix::pauli_1q(Pauli1::Z);
    let zz = matrix::kron(&z, &z, 2, 2);
    let h_delta = matrix::scale(&zz, Complex64::new(delta_zz / 2.0, 0.0));

    // Dissipator part: AD+PD on both qubits
    let sm = matrix::sigma_minus();
    let sm_l = matrix::kron(&sm, &i2, 2, 2);
    let sm_r = matrix::kron(&i2, &sm, 2, 2);
    let z_l = matrix::kron(&z, &i2, 2, 2);
    let z_r = matrix::kron(&i2, &z, 2, 2);
    let collapse: Vec<(Matrix, f64)> = vec![
        (sm_l, beta_down),
        (sm_r, beta_down),
        (z_l, beta_phi / 2.0),
        (z_r, beta_phi / 2.0),
    ];
    let mixed = Lindbladian::new(d, h_delta, collapse);
    let gate = Gate::identity(2, mixed, tau_g);
    let pl = synthesize_superop_identity(&gate);

    let rate = |s: &str| pl.rate(&PauliString::from_label(s).unwrap());

    // Expected (leading-order superposition of independent contributions):
    //   Dissipative (AD+PD) on each qubit contributes single-qubit rates
    //   (paper line 812): lambda_{i·x,y} = beta_down * tau / 4,
    //                     lambda_{i·z} = beta_phi * tau / 2 (mirror for l).
    //   Coherent ZZ contributes lambda_ZZ = (delta_zz * tau)^2 / 4
    //   (paper eq. 981 identity case).
    // Cross terms between dissipative and coherent are O(beta * delta * tau^2)
    // and small for these parameter values.
    let expected_ix_iy = beta_down * tau_g / 4.0;
    let expected_iz = beta_phi * tau_g / 2.0;
    let expected_zz = (delta_zz * tau_g).powi(2) / 4.0;
    let tol_dissipative = 1e-10;
    let tol_coherent = 1e-10;

    assert_abs_diff_eq!(rate("IX"), expected_ix_iy, epsilon = tol_dissipative);
    assert_abs_diff_eq!(rate("IY"), expected_ix_iy, epsilon = tol_dissipative);
    assert_abs_diff_eq!(rate("XI"), expected_ix_iy, epsilon = tol_dissipative);
    assert_abs_diff_eq!(rate("YI"), expected_ix_iy, epsilon = tol_dissipative);
    assert_abs_diff_eq!(rate("IZ"), expected_iz, epsilon = tol_dissipative);
    assert_abs_diff_eq!(rate("ZI"), expected_iz, epsilon = tol_dissipative);
    assert_abs_diff_eq!(rate("ZZ"), expected_zz, epsilon = tol_coherent);
}
