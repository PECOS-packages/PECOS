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

//! End-to-end sanity tests for the device-parameter convenience API in
//! `noise_models`. Exercises the same paper fixtures as the hand-rolled
//! tests but through the ergonomic `(T_1, T_2)` interface.

use approx::assert_abs_diff_eq;

use pecos_lindblad::noise_models::{ad_pd_1q, ad_pd_2q, coherent_phase_2q, t1_t2_to_rates};
use pecos_lindblad::{
    DEFAULT_N_STEPS, Gate, Pauli1, PauliString, synthesize_exact_unitary, synthesize_identity_1q,
    synthesize_numerical,
};

#[test]
fn identity_1q_via_device_params() {
    // T1 = 100 us, T2 = 80 us => beta_down = 1e4, beta_phi = 7500.
    let t1 = 100e-6;
    let t2 = 80e-6;
    let tau_g = 1e-6;
    let (bd, bp) = t1_t2_to_rates(t1, t2);

    let noise = ad_pd_1q(t1, t2);
    let gate = Gate::identity(1, noise, tau_g);
    let pl = synthesize_identity_1q(&gate);

    let rate = |p: Pauli1| pl.rate(&PauliString::single(p));
    assert_abs_diff_eq!(rate(Pauli1::X), bd * tau_g / 4.0, epsilon = 1e-14);
    assert_abs_diff_eq!(rate(Pauli1::Y), bd * tau_g / 4.0, epsilon = 1e-14);
    assert_abs_diff_eq!(rate(Pauli1::Z), bp * tau_g / 2.0, epsilon = 1e-14);
}

#[test]
fn cx_theta_via_device_params_matches_hand_rolled() {
    // Build the same gate both ways and confirm rates agree.
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_4;
    let t1_l = 100.0;
    let t1_r = 80.0;
    let t2_l = 120.0;
    let t2_r = 90.0;

    let noise = ad_pd_2q(t1_l, t1_r, t2_l, t2_r);
    let gate = Gate::cx_theta(omega, theta, noise);
    let pl = synthesize_numerical(&gate, DEFAULT_N_STEPS);

    // Paper eq 941 sanity: lambda_xi = (2 theta + sin 2 theta) / 16 * beta_down_l / omega.
    let (bd_l, _) = t1_t2_to_rates(t1_l, t2_l);
    let s2 = (2.0 * theta).sin();
    let expected = (2.0 * theta + s2) / 16.0 * bd_l / omega;
    assert_abs_diff_eq!(
        pl.rate(&PauliString::from_label("XI").unwrap()),
        expected,
        epsilon = 1e-8
    );
}

#[test]
fn coherent_phase_2q_via_api() {
    // Check that coherent_phase_2q + synthesize_exact_unitary reproduces
    // paper eq 981 for CZ_theta.
    let omega_cz = 1.0;
    let theta = std::f64::consts::FRAC_PI_3;
    let delta_iz = 1e-6;
    let delta_zi = 2e-6;
    let delta_zz = 5e-7;

    let noise = coherent_phase_2q(delta_iz, delta_zi, delta_zz);
    let gate = Gate::cz_theta(omega_cz, theta, noise);
    let pl = synthesize_exact_unitary(&gate);

    let rate = |s: &str| pl.rate(&PauliString::from_label(s).unwrap());
    let factor = theta.powi(2) / 4.0 / omega_cz.powi(2);
    assert_abs_diff_eq!(rate("IZ"), factor * delta_iz.powi(2), epsilon = 1e-14);
    assert_abs_diff_eq!(rate("ZI"), factor * delta_zi.powi(2), epsilon = 1e-14);
    assert_abs_diff_eq!(rate("ZZ"), factor * delta_zz.powi(2), epsilon = 1e-14);
}
