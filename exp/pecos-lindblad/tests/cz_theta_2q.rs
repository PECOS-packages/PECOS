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

//! Parity test: 2-qubit CZ_theta gate under independent AD + PD on each
//! qubit vs closed-form leading-order results from arXiv:2502.03462
//! eqs. 896-906 (appendix SubApp:CZ_th+AD+PD).
//!
//! String index convention: leftmost factor is the "l" (left) qubit.
//!   "iz" == I (x) Z
//!   "zx" == Z (x) X
//!
//! Paper closed forms (all others vanish to leading order):
//!   lambda_iz = (theta/2) * beta_phi_r / omega_cz
//!   lambda_zi = (theta/2) * beta_phi_l / omega_cz
//!   lambda_ix = lambda_iy = (2t+sin2t)/16 * beta_down_r / omega_cz
//!   lambda_xi = lambda_yi = (2t+sin2t)/16 * beta_down_l / omega_cz
//!   lambda_zx = lambda_zy = (2t-sin2t)/16 * beta_down_r / omega_cz
//!   lambda_xz = lambda_yz = (2t-sin2t)/16 * beta_down_l / omega_cz

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use pecos_lindblad::matrix::{self, Matrix};
use pecos_lindblad::{
    DEFAULT_N_STEPS, Gate, Lindbladian, Pauli1, PauliString, synthesize_numerical,
};

fn two_qubit_ad_plus_pd(
    beta_down_l: f64,
    beta_down_r: f64,
    beta_phi_l: f64,
    beta_phi_r: f64,
) -> Lindbladian {
    let d = 4;
    let i2 = matrix::identity(2);
    let sm = matrix::sigma_minus();
    let z = matrix::pauli_1q(Pauli1::Z);
    // Kronecker: (l) ⊗ (r) with l = left qubit = index 0.
    let sm_l = matrix::kron(&sm, &i2, 2, 2);
    let sm_r = matrix::kron(&i2, &sm, 2, 2);
    let z_l = matrix::kron(&z, &i2, 2, 2);
    let z_r = matrix::kron(&i2, &z, 2, 2);

    let collapse: Vec<(Matrix, f64)> = vec![
        (sm_l, beta_down_l),
        (sm_r, beta_down_r),
        (z_l, beta_phi_l / 2.0),
        (z_r, beta_phi_r / 2.0),
    ];
    let zero_ham: Matrix = vec![Complex64::new(0.0, 0.0); d * d];
    Lindbladian::new(d, zero_ham, collapse)
}

#[derive(Debug, Clone, Copy)]
struct CzExpected {
    iz: f64,
    zi: f64,
    ix: f64,
    iy: f64,
    xi: f64,
    yi: f64,
    zx: f64,
    zy: f64,
    xz: f64,
    yz: f64,
}

fn paper_closed_form(
    theta: f64,
    omega_cz: f64,
    beta_down_l: f64,
    beta_down_r: f64,
    beta_phi_l: f64,
    beta_phi_r: f64,
) -> CzExpected {
    let two_t = 2.0 * theta;
    let sin_2t = two_t.sin();
    let amp_r = (two_t + sin_2t) / 16.0 * (beta_down_r / omega_cz);
    let amp_l = (two_t + sin_2t) / 16.0 * (beta_down_l / omega_cz);
    let anti_r = (two_t - sin_2t) / 16.0 * (beta_down_r / omega_cz);
    let anti_l = (two_t - sin_2t) / 16.0 * (beta_down_l / omega_cz);
    CzExpected {
        iz: (theta / 2.0) * (beta_phi_r / omega_cz),
        zi: (theta / 2.0) * (beta_phi_l / omega_cz),
        ix: amp_r,
        iy: amp_r,
        xi: amp_l,
        yi: amp_l,
        zx: anti_r,
        zy: anti_r,
        xz: anti_l,
        yz: anti_l,
    }
}

fn run_cz(
    theta: f64,
    omega_cz: f64,
    beta_down_l: f64,
    beta_down_r: f64,
    beta_phi_l: f64,
    beta_phi_r: f64,
    tol: f64,
) {
    let noise = two_qubit_ad_plus_pd(beta_down_l, beta_down_r, beta_phi_l, beta_phi_r);
    let gate = Gate::cz_theta(omega_cz, theta, noise);
    let pl = synthesize_numerical(&gate, DEFAULT_N_STEPS);

    let exp = paper_closed_form(
        theta,
        omega_cz,
        beta_down_l,
        beta_down_r,
        beta_phi_l,
        beta_phi_r,
    );
    let rate = |s: &str| pl.rate(&PauliString::from_label(s).unwrap());

    assert_abs_diff_eq!(rate("IZ"), exp.iz, epsilon = tol);
    assert_abs_diff_eq!(rate("ZI"), exp.zi, epsilon = tol);
    assert_abs_diff_eq!(rate("IX"), exp.ix, epsilon = tol);
    assert_abs_diff_eq!(rate("IY"), exp.iy, epsilon = tol);
    assert_abs_diff_eq!(rate("XI"), exp.xi, epsilon = tol);
    assert_abs_diff_eq!(rate("YI"), exp.yi, epsilon = tol);
    assert_abs_diff_eq!(rate("ZX"), exp.zx, epsilon = tol);
    assert_abs_diff_eq!(rate("ZY"), exp.zy, epsilon = tol);
    assert_abs_diff_eq!(rate("XZ"), exp.xz, epsilon = tol);
    assert_abs_diff_eq!(rate("YZ"), exp.yz, epsilon = tol);

    // All remaining 5 Paulis should be (numerically) zero at leading order.
    for label in ["XX", "XY", "YX", "YY", "ZZ"] {
        assert_abs_diff_eq!(rate(label), 0.0, epsilon = tol);
    }
}

#[test]
fn cz_theta_ad_plus_pd_pi_over_4() {
    // Weak noise beta/omega ~ 1e-4 => leading-order match to ~1e-8.
    run_cz(
        std::f64::consts::FRAC_PI_4,
        1.0,
        1e-4, // AD l
        2e-4, // AD r
        5e-5, // PD l
        7e-5, // PD r
        1e-8,
    );
}

#[test]
fn cz_theta_ad_plus_pd_pi_over_2() {
    // theta=pi/2 is Clifford: paper predicts 4-fold degeneracies.
    run_cz(std::f64::consts::FRAC_PI_2, 1.0, 1e-4, 1e-4, 0.0, 0.0, 1e-8);
}

#[test]
fn cz_theta_ad_only() {
    run_cz(std::f64::consts::FRAC_PI_3, 1.5, 3e-4, 1e-4, 0.0, 0.0, 1e-8);
}

#[test]
fn cz_theta_pd_only() {
    run_cz(std::f64::consts::FRAC_PI_4, 1.0, 0.0, 0.0, 2e-4, 3e-4, 1e-8);
}
