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

//! Parity test: 1-qubit X_theta gate under amplitude damping + pure
//! dephasing vs closed-form leading-order results from arXiv:2502.03462
//! eqs. 869-874 (appendix SubApp:X_th+AD+PD).
//!
//! Paper closed forms (lambda_k are dimensionless, integrated over the gate):
//!   lambda_x = (theta / 4) * (beta_down / omega_x)
//!   lambda_y = ((2 theta + sin 2 theta) / 16) * (beta_down / omega_x)
//!            + ((2 theta - sin 2 theta) / 8)  * (beta_phi  / omega_x)
//!   lambda_z = ((2 theta - sin 2 theta) / 16) * (beta_down / omega_x)
//!            + ((2 theta + sin 2 theta) / 8)  * (beta_phi  / omega_x)
//!
//! The paper's approximation is "leading order in beta/omega"; at
//! `beta/omega ~ 1e-2` deviation should be ~`O(1e-5)` per
//! Appendix `App:LindPertPrecision` (line 1078).

use approx::assert_abs_diff_eq;

use pecos_lindblad::matrix::{self, Matrix};
use pecos_lindblad::{
    DEFAULT_N_STEPS, Gate, Lindbladian, Pauli1, PauliString, synthesize_numerical_1q,
};

fn ad_plus_pd_noise(beta_down: f64, beta_phi: f64) -> Lindbladian {
    let d = 2;
    let hamiltonian = matrix::zeros(d);
    let collapse: Vec<(Matrix, f64)> = vec![
        (matrix::sigma_minus(), beta_down),
        (matrix::pauli_1q(Pauli1::Z), beta_phi / 2.0),
    ];
    Lindbladian::new(d, hamiltonian, collapse)
}

fn paper_closed_form(theta: f64, omega_x: f64, beta_down: f64, beta_phi: f64) -> (f64, f64, f64) {
    let two_t = 2.0 * theta;
    let sin_2t = two_t.sin();
    let lambda_x = (theta / 4.0) * (beta_down / omega_x);
    let lambda_y = ((two_t + sin_2t) / 16.0) * (beta_down / omega_x)
        + ((two_t - sin_2t) / 8.0) * (beta_phi / omega_x);
    let lambda_z = ((two_t - sin_2t) / 16.0) * (beta_down / omega_x)
        + ((two_t + sin_2t) / 8.0) * (beta_phi / omega_x);
    (lambda_x, lambda_y, lambda_z)
}

fn run_and_compare(theta: f64, omega_x: f64, beta_down: f64, beta_phi: f64, tol: f64) {
    let noise = ad_plus_pd_noise(beta_down, beta_phi);
    let gate = Gate::x_theta(omega_x, theta, noise);
    let pl = synthesize_numerical_1q(&gate, DEFAULT_N_STEPS);

    let (expected_x, expected_y, expected_z) =
        paper_closed_form(theta, omega_x, beta_down, beta_phi);

    let got_x = pl.rate(&PauliString::single(Pauli1::X));
    let got_y = pl.rate(&PauliString::single(Pauli1::Y));
    let got_z = pl.rate(&PauliString::single(Pauli1::Z));

    assert_abs_diff_eq!(got_x, expected_x, epsilon = tol);
    assert_abs_diff_eq!(got_y, expected_y, epsilon = tol);
    assert_abs_diff_eq!(got_z, expected_z, epsilon = tol);
}

#[test]
fn x_theta_ad_plus_pd_pi_over_4() {
    // Weak noise => numerical integrand matches leading-order closed form
    // to within ~O(beta^2 / omega^2). Use beta/omega = 1e-4 => tol ~1e-8.
    let omega_x = 1.0;
    let beta_down = 1e-4;
    let beta_phi = 2e-4;
    let theta = std::f64::consts::FRAC_PI_4;
    run_and_compare(theta, omega_x, beta_down, beta_phi, 1e-8);
}

#[test]
fn x_theta_ad_plus_pd_pi_over_2() {
    let omega_x = 1.0;
    let beta_down = 1e-4;
    let beta_phi = 5e-5;
    let theta = std::f64::consts::FRAC_PI_2;
    run_and_compare(theta, omega_x, beta_down, beta_phi, 1e-8);
}

#[test]
fn x_theta_ad_only_pi_over_3() {
    // lambda_x = (theta/4)(beta_down/omega_x), lambda_y and lambda_z from
    // beta_down only.
    let omega_x = 2.0;
    let beta_down = 3e-4;
    let beta_phi = 0.0;
    let theta = std::f64::consts::FRAC_PI_3;
    run_and_compare(theta, omega_x, beta_down, beta_phi, 1e-8);
}

#[test]
fn x_theta_pd_only_pi_over_2() {
    // lambda_x = 0 (no AD).
    let omega_x = 1.0;
    let beta_down = 0.0;
    let beta_phi = 4e-4;
    let theta = std::f64::consts::FRAC_PI_2;
    run_and_compare(theta, omega_x, beta_down, beta_phi, 1e-8);
}

#[test]
fn numerical_1q_reduces_to_identity() {
    // synthesize_numerical_1q on an identity gate (H_g = 0) should match
    // the fast-path identity synthesis to high precision. Exercises the
    // time-integral path on a constant integrand.
    use pecos_lindblad::synthesize_identity_1q;

    let beta_down = 1e-4;
    let beta_phi = 3e-4;
    let tau_g = 50.0;
    let noise = ad_plus_pd_noise(beta_down, beta_phi);
    let gate = Gate::identity(1, noise, tau_g);

    let fast = synthesize_identity_1q(&gate);
    let numerical = synthesize_numerical_1q(&gate, DEFAULT_N_STEPS);

    for p in [Pauli1::X, Pauli1::Y, Pauli1::Z] {
        let key = PauliString::single(p);
        assert_abs_diff_eq!(fast.rate(&key), numerical.rate(&key), epsilon = 1e-12);
    }
}
