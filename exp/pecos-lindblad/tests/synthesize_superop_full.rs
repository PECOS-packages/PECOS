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

//! Tests for `synthesize_superop` -- the fully-general time-sliced path
//! that handles `H_g != 0` AND simultaneous coherent + dissipative noise.
//!
//! Three validation modes:
//! 1. Matches `synthesize_numerical` on pure dissipative inputs (CX+AD+PD).
//! 2. Matches `synthesize_exact_unitary` on pure coherent inputs (CX+phase).
//! 3. NEW: handles mixed input that neither of the above covers
//!    (CX_theta + AD+PD + coherent ZZ phase, all at once).

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use pecos_lindblad::matrix::{self, Matrix};
use pecos_lindblad::noise_models::{ad_pd_2q, coherent_phase_2q};
use pecos_lindblad::{
    DEFAULT_N_SLICES, DEFAULT_N_STEPS, Gate, Lindbladian, Pauli1, PauliString,
    synthesize_exact_unitary, synthesize_numerical, synthesize_superop,
};

#[test]
fn superop_matches_synthesize_numerical_cx_ad_pd() {
    // Weak dissipative noise on CX_theta: superop (all orders) and
    // synthesize_numerical (leading order) should agree to high precision.
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_4;
    let noise = ad_pd_2q(1e5, 1e5, 8e4, 8e4); // very weak
    let gate = Gate::cx_theta(omega, theta, noise);

    let simpson = synthesize_numerical(&gate, DEFAULT_N_STEPS);
    let superop = synthesize_superop(&gate, DEFAULT_N_SLICES);

    for ps in PauliString::enumerate_nonidentity(2) {
        // At this noise level (~1e-5), O(beta^2) corrections are ~1e-10
        // and negligible. Expect agreement to ~1e-9.
        assert_abs_diff_eq!(simpson.rate(&ps), superop.rate(&ps), epsilon = 1e-9);
    }
}

#[test]
fn superop_matches_synthesize_exact_unitary_cx_coherent() {
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_4;
    let noise = coherent_phase_2q(1e-5, 2e-5, 5e-6);
    let gate = Gate::cx_theta(omega, theta, noise);

    let exact = synthesize_exact_unitary(&gate);
    let superop = synthesize_superop(&gate, DEFAULT_N_SLICES);

    for ps in PauliString::enumerate_nonidentity(2) {
        assert_abs_diff_eq!(exact.rate(&ps), superop.rate(&ps), epsilon = 1e-10);
    }
}

#[test]
fn superop_handles_cx_mixed_ad_pd_plus_coherent_zz() {
    // THE new capability: CX_theta with simultaneous AD+PD dissipators
    // AND coherent ZZ phase noise. No other synthesis path can do this:
    // - synthesize_numerical: dissipative leading-order, loses coherent
    //   quadratic contribution.
    // - synthesize_exact_unitary: asserts no c_ops, refuses mixed input.
    // - synthesize_superop_identity: requires H_g = 0.
    let d = 4;
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_4;
    let t1 = 1e5; // very weak AD
    let t2 = 8e4; // weak PD
    let delta_zz = 1e-4; // weak coherent ZZ

    let i2 = matrix::identity(2);
    let sm = matrix::sigma_minus();
    let z = matrix::pauli_1q(Pauli1::Z);
    let sm_l = matrix::kron(&sm, &i2, 2, 2);
    let sm_r = matrix::kron(&i2, &sm, 2, 2);
    let z_l = matrix::kron(&z, &i2, 2, 2);
    let z_r = matrix::kron(&i2, &z, 2, 2);
    let zz = matrix::kron(&z, &z, 2, 2);

    let beta_down = 1.0 / t1;
    let beta_phi = 1.0 / t2 - 1.0 / (2.0 * t1);
    let h_delta = matrix::scale(&zz, Complex64::new(delta_zz / 2.0, 0.0));
    let collapse: Vec<(Matrix, f64)> = vec![
        (sm_l, beta_down),
        (sm_r, beta_down),
        (z_l, beta_phi / 2.0),
        (z_r, beta_phi / 2.0),
    ];
    let mixed = Lindbladian::new(d, h_delta, collapse);
    let mixed_gate = Gate::cx_theta(omega, theta, mixed);

    // Also build pure-dissipative and pure-coherent variants for
    // superposition comparison.
    let pure_diss = ad_pd_2q(t1, t1, t2, t2);
    let diss_gate = Gate::cx_theta(omega, theta, pure_diss);
    let pure_coh = coherent_phase_2q(0.0, 0.0, delta_zz);
    let coh_gate = Gate::cx_theta(omega, theta, pure_coh);

    let pl_mixed = synthesize_superop(&mixed_gate, DEFAULT_N_SLICES);
    let pl_diss = synthesize_superop(&diss_gate, DEFAULT_N_SLICES);
    let pl_coh = synthesize_superop(&coh_gate, DEFAULT_N_SLICES);

    // At weak coupling, the mixed rates should equal the superposition of
    // the two individual-noise rates (cross-terms are second-order small).
    for ps in PauliString::enumerate_nonidentity(2) {
        let expected = pl_diss.rate(&ps) + pl_coh.rate(&ps);
        let got = pl_mixed.rate(&ps);
        // Cross-term O(beta * delta * tau^2) ~ 1e-5 * 1e-4 * 1 = 1e-9.
        // Use tolerance slightly above that to account for MC/numerical noise.
        assert_abs_diff_eq!(got, expected, epsilon = 1e-8);
    }

    // Additional sanity: mixed has non-trivial rates from both sources.
    let rate_mixed = |s: &str| pl_mixed.rate(&PauliString::from_label(s).unwrap());
    assert!(
        rate_mixed("IX") > 1e-8,
        "dissipative contribution should be present"
    );
    assert!(
        rate_mixed("ZZ") > 1e-10,
        "coherent ZZ contribution should be present"
    );
}
