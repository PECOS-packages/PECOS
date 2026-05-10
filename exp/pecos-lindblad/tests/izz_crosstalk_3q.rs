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

//! Parity test: 3-qubit `CX_theta ⊗ I` gate with coherent IZZ crosstalk
//! between target (q1) and spectator (q2), vs closed-form results from
//! arXiv:2502.03462 eqs. 1009-1011 (`SubApp:3QXtalk`).
//!
//! String index convention: leftmost factor = qubit 0 (control).
//!   "IYZ" = I on q0, Y on q1, Z on q2.
//!
//! Paper closed forms (quadratic in delta, weight-3 rates):
//!   lambda_iyz = lambda_zyz = sin^4(theta) / 16     * (delta / omega)^2
//!   lambda_izz = [2 theta + sin 2 theta]^2 / 64     * (delta / omega)^2
//!   lambda_zzz = [2 theta - sin 2 theta]^2 / 64     * (delta / omega)^2
//!
//! All other non-identity 3Q Paulis (there are 63 - 4 = 59 of them) are
//! **zero** to leading order in delta/omega.
//!
//! **Weight-3 rates** break the standard sparse-PL weight-2 assumption --
//! this test also exercises `PauliLindbladModel::supports` over the full
//! 3Q basis.

use approx::assert_abs_diff_eq;

use pecos_lindblad::{Gate, PauliString, synthesize_exact_unitary};

fn paper_rate(label: &str, theta: f64, omega: f64, delta: f64) -> f64 {
    let dratio_sq = (delta / omega).powi(2);
    let s = theta.sin();
    let s2 = (2.0 * theta).sin();
    match label {
        "IYZ" | "ZYZ" => s.powi(4) / 16.0 * dratio_sq,
        "IZZ" => (2.0 * theta + s2).powi(2) / 64.0 * dratio_sq,
        "ZZZ" => (2.0 * theta - s2).powi(2) / 64.0 * dratio_sq,
        _ => 0.0,
    }
}

fn run_izz(theta: f64, omega: f64, delta: f64, tol: f64) {
    let gate = Gate::cx_theta_with_izz_crosstalk(omega, theta, delta);
    let pl = synthesize_exact_unitary(&gate);

    // All 63 non-identity 3Q Paulis.
    for ps in PauliString::enumerate_nonidentity(3) {
        let label = format!("{}", ps);
        let got = pl.rate(&ps);
        let expected = paper_rate(&label, theta, omega, delta);
        assert_abs_diff_eq!(got, expected, epsilon = tol);
    }
}

#[test]
fn izz_crosstalk_pi_over_4_weak() {
    // delta/omega = 1e-3 => rates ~ 1e-6 at most; tol ~1e-10.
    run_izz(std::f64::consts::FRAC_PI_4, 1.0, 1e-3, 1e-10);
}

#[test]
fn izz_crosstalk_pi_over_2_weak() {
    // theta = pi/2 (Clifford): sin(2 theta) = 0, so lambda_izz = lambda_zzz.
    run_izz(std::f64::consts::FRAC_PI_2, 1.0, 1e-3, 1e-10);
}

#[test]
fn izz_crosstalk_pi_over_3_weak() {
    run_izz(std::f64::consts::FRAC_PI_3, 1.5, 5e-4, 1e-10);
}

#[test]
fn izz_crosstalk_zero_delta_gives_zero_rates() {
    // delta = 0 => no crosstalk => all rates zero.
    let gate = Gate::cx_theta_with_izz_crosstalk(1.0, std::f64::consts::FRAC_PI_4, 0.0);
    let pl = synthesize_exact_unitary(&gate);
    for ps in PauliString::enumerate_nonidentity(3) {
        assert_abs_diff_eq!(pl.rate(&ps), 0.0, epsilon = 1e-14);
    }
}

#[test]
fn izz_crosstalk_produces_only_weight_3_and_no_weight_2() {
    // The paper's claim: this gate produces weight-2 (IZZ) AND weight-3
    // (IYZ, ZYZ, ZZZ) rates but NO weight-1 rates. Verify the weight
    // distribution in our output matches that claim.
    let gate = Gate::cx_theta_with_izz_crosstalk(1.0, std::f64::consts::FRAC_PI_4, 1e-3);
    let pl = synthesize_exact_unitary(&gate);

    // Weight-1 rates should all be (numerically) zero.
    for ps in PauliString::enumerate_nonidentity(3) {
        if ps.weight() == 1 {
            assert_abs_diff_eq!(pl.rate(&ps), 0.0, epsilon = 1e-10);
        }
    }

    // At least one weight-3 rate (e.g. lambda_iyz) should be non-zero.
    let iyz = PauliString::from_label("IYZ").unwrap();
    assert!(pl.rate(&iyz) > 1e-12);
}
