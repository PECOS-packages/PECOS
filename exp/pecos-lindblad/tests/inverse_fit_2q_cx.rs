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

//! 2-qubit inverse-fit tests for `CX_theta + AD+PD`. Trickier than the CZ
//! case because `beta_down_r` and `beta_phi_r` mix in `lambda_iz` /
//! `lambda_zz`. We use the `lambda_iz - lambda_zz` identity to decouple.

use approx::assert_abs_diff_eq;

use pecos_lindblad::noise_models::{ad_pd_2q, recover_ad_pd_2q_from_cx_theta};
use pecos_lindblad::{DEFAULT_N_STEPS, Gate, PauliLindbladModel, synthesize_numerical};

fn synth_cx(
    t1_l: f64,
    t1_r: f64,
    t2_l: f64,
    t2_r: f64,
    omega: f64,
    theta: f64,
) -> PauliLindbladModel {
    let noise = ad_pd_2q(t1_l, t1_r, t2_l, t2_r);
    let gate = Gate::cx_theta(omega, theta, noise);
    synthesize_numerical(&gate, DEFAULT_N_STEPS)
}

#[test]
fn round_trip_cx_pi_over_4() {
    let t1_l = 150.0;
    let t1_r = 100.0;
    let t2_l = 100.0;
    let t2_r = 70.0;
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_4;

    let pl = synth_cx(t1_l, t1_r, t2_l, t2_r, omega, theta);
    let rec = recover_ad_pd_2q_from_cx_theta(&pl, omega, theta).unwrap();
    // At beta/omega ~ 1e-2, next-order corrections ~ 1e-4 * 1e-2 = 1e-6.
    // Use 1e-5 tolerance.
    assert_abs_diff_eq!(rec.t1_l, t1_l, epsilon = 1e-5);
    assert_abs_diff_eq!(rec.t1_r, t1_r, epsilon = 1e-5);
    assert_abs_diff_eq!(rec.t2_l, t2_l, epsilon = 1e-5);
    assert_abs_diff_eq!(rec.t2_r, t2_r, epsilon = 1e-5);
}

#[test]
fn round_trip_cx_pi_over_3() {
    let pl = synth_cx(200.0, 300.0, 150.0, 200.0, 1.5, std::f64::consts::FRAC_PI_3);
    let rec = recover_ad_pd_2q_from_cx_theta(&pl, 1.5, std::f64::consts::FRAC_PI_3).unwrap();
    assert_abs_diff_eq!(rec.t1_l, 200.0, epsilon = 1e-5);
    assert_abs_diff_eq!(rec.t1_r, 300.0, epsilon = 1e-5);
    assert_abs_diff_eq!(rec.t2_l, 150.0, epsilon = 1e-5);
    assert_abs_diff_eq!(rec.t2_r, 200.0, epsilon = 1e-5);
}

#[test]
fn recovery_returns_none_at_degenerate_angle_pi_over_2() {
    // At theta = pi/2, sin(2 theta) = 0 -> beta_down_r and beta_phi_r
    // are not independently recoverable.
    let pl = synth_cx(100.0, 100.0, 80.0, 80.0, 1.0, std::f64::consts::FRAC_PI_2);
    assert!(recover_ad_pd_2q_from_cx_theta(&pl, 1.0, std::f64::consts::FRAC_PI_2).is_none());
}

#[test]
fn recovery_returns_none_at_zero_theta() {
    // At theta = 0, the gate is trivial and rates vanish. Also degenerate.
    let pl = synth_cx(100.0, 100.0, 80.0, 80.0, 1.0, 0.0);
    assert!(recover_ad_pd_2q_from_cx_theta(&pl, 1.0, 0.0).is_none());
}

#[test]
fn cx_round_trip_end_to_end() {
    // Device-style story: dev data (T1, T2) per qubit -> rates ->
    // recover -> compare.
    let t1_l = 300.0;
    let t2_l = 200.0;
    let t1_r = 280.0;
    let t2_r = 190.0;
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_4;

    let pl = synth_cx(t1_l, t1_r, t2_l, t2_r, omega, theta);
    let rec = recover_ad_pd_2q_from_cx_theta(&pl, omega, theta).unwrap();

    // Print-style check: all 4 params recovered with ~5 decimal digits accuracy.
    for (got, want, name) in [
        (rec.t1_l, t1_l, "T1_l"),
        (rec.t2_l, t2_l, "T2_l"),
        (rec.t1_r, t1_r, "T1_r"),
        (rec.t2_r, t2_r, "T2_r"),
    ] {
        let rel_err = (got - want).abs() / want;
        assert!(
            rel_err < 1e-4,
            "{}: recovered {} vs true {} (rel_err={})",
            name,
            got,
            want,
            rel_err,
        );
    }
}
