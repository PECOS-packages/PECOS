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

//! Parity test: 2-qubit CX_theta gate under independent AD + PD on each
//! qubit vs closed-form leading-order results from arXiv:2502.03462
//! eqs. 929-956 (appendix SubApp:CX_th+AD+PD).
//!
//! CX_theta is the showcase gate of the paper. Unlike CZ_theta, AD and PD
//! contributions *mix* on `lambda_{iy, iz, zy, zz}` (each has both a
//! `beta_down_r/omega` and `beta_phi_r/omega` term).
//!
//! Paper closed forms (10 non-zero rates):
//!   lambda_ix = (theta/4)(beta_down_r/omega)
//!   lambda_iy = [(12t+8s2+s4)/128] beta_down_r/omega
//!             + [(4t-s4)/64] beta_phi_r/omega
//!   lambda_iz = [(4t-s4)/128] beta_down_r/omega
//!             + [(12t+8s2+s4)/64] beta_phi_r/omega
//!   lambda_xi = lambda_yi = [(2t+s2)/16] beta_down_l/omega
//!   lambda_xx = lambda_yx = [(2t-s2)/16] beta_down_l/omega
//!   lambda_zi = (theta/2)(beta_phi_l/omega)
//!   lambda_zy = [(12t-8s2+s4)/128] beta_down_r/omega
//!             + [(4t-s4)/64] beta_phi_r/omega
//!   lambda_zz = [(4t-s4)/128] beta_down_r/omega
//!             + [(12t-8s2+s4)/64] beta_phi_r/omega
//!
//! where s2 = sin(2 theta), s4 = sin(4 theta).
//!
//! 5 rates are zero to leading order: XY, XZ, YY, YZ, ZX.

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

#[allow(clippy::too_many_arguments)]
fn paper_cx_rate(
    label: &str,
    theta: f64,
    omega: f64,
    bd_l: f64,
    bd_r: f64,
    bp_l: f64,
    bp_r: f64,
) -> f64 {
    let s2 = (2.0 * theta).sin();
    let s4 = (4.0 * theta).sin();
    let f_amp_plus = (2.0 * theta + s2) / 16.0;
    let f_amp_minus = (2.0 * theta - s2) / 16.0;
    let f_dbl_plus = (12.0 * theta + 8.0 * s2 + s4) / 128.0;
    let f_dbl_minus = (12.0 * theta - 8.0 * s2 + s4) / 128.0;
    let f_anti_4 = (4.0 * theta - s4) / 64.0;
    let f_anti_128 = (4.0 * theta - s4) / 128.0;
    match label {
        "IX" => (theta / 4.0) * (bd_r / omega),
        "IY" => f_dbl_plus * (bd_r / omega) + f_anti_4 * (bp_r / omega),
        "IZ" => {
            f_anti_128 * (bd_r / omega) + (12.0 * theta + 8.0 * s2 + s4) / 64.0 * (bp_r / omega)
        }
        "XI" | "YI" => f_amp_plus * (bd_l / omega),
        "XX" | "YX" => f_amp_minus * (bd_l / omega),
        "ZI" => (theta / 2.0) * (bp_l / omega),
        "ZY" => f_dbl_minus * (bd_r / omega) + f_anti_4 * (bp_r / omega),
        "ZZ" => {
            f_anti_128 * (bd_r / omega) + (12.0 * theta - 8.0 * s2 + s4) / 64.0 * (bp_r / omega)
        }
        "XY" | "XZ" | "YY" | "YZ" | "ZX" => 0.0,
        _ => panic!("unknown label {}", label),
    }
}

fn run_cx(theta: f64, omega: f64, bd_l: f64, bd_r: f64, bp_l: f64, bp_r: f64, tol: f64) {
    let noise = two_qubit_ad_plus_pd(bd_l, bd_r, bp_l, bp_r);
    let gate = Gate::cx_theta(omega, theta, noise);
    let pl = synthesize_numerical(&gate, DEFAULT_N_STEPS);

    let all_labels = [
        "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ",
    ];
    for label in all_labels {
        let got = pl.rate(&PauliString::from_label(label).unwrap());
        let expected = paper_cx_rate(label, theta, omega, bd_l, bd_r, bp_l, bp_r);
        assert_abs_diff_eq!(got, expected, epsilon = tol);
    }
}

#[test]
fn cx_theta_ad_plus_pd_pi_over_4() {
    // Paper's showcase angle (sqrt(CX) = CX_{pi/4}).
    run_cx(
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
fn cx_theta_ad_plus_pd_pi_over_2() {
    // Clifford: full CNOT. Exercises sin(4 theta) = sin(2pi) = 0 terms.
    run_cx(
        std::f64::consts::FRAC_PI_2,
        1.0,
        1e-4,
        1.5e-4,
        3e-5,
        4e-5,
        1e-8,
    );
}

#[test]
fn cx_theta_ad_only() {
    run_cx(std::f64::consts::FRAC_PI_3, 1.5, 3e-4, 1e-4, 0.0, 0.0, 1e-8);
}

#[test]
fn cx_theta_pd_only() {
    run_cx(std::f64::consts::FRAC_PI_4, 1.0, 0.0, 0.0, 2e-4, 3e-4, 1e-8);
}

#[test]
fn cx_theta_symmetric_beta_down_symmetric_pd() {
    // Exercise all non-zero rates simultaneously, symmetric noise case.
    run_cx(
        std::f64::consts::FRAC_PI_4,
        1.0,
        2e-4,
        2e-4,
        1e-4,
        1e-4,
        1e-8,
    );
}
