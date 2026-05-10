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

//! 2-qubit inverse-fit tests for `CZ_theta + AD+PD`. Round-trip:
//!   (T_1, T_2)_{l,r} -> synthesize -> PL rates -> recover -> (T_1, T_2)_{l,r}.
//! Must be bit-close on clean (noiseless) synthetic data.

use approx::assert_abs_diff_eq;

use pecos_lindblad::noise_models::{
    ad_pd_2q, cz_recovery_residual, recover_ad_pd_2q_from_cz_theta,
};
use pecos_lindblad::{
    DEFAULT_N_STEPS, Gate, PauliLindbladModel, PauliString, synthesize_numerical,
};

fn synth_cz(
    t1_l: f64,
    t1_r: f64,
    t2_l: f64,
    t2_r: f64,
    omega: f64,
    theta: f64,
) -> PauliLindbladModel {
    let noise = ad_pd_2q(t1_l, t1_r, t2_l, t2_r);
    let gate = Gate::cz_theta(omega, theta, noise);
    synthesize_numerical(&gate, DEFAULT_N_STEPS)
}

#[test]
fn round_trip_2q_asymmetric_params() {
    // Four independent parameters; recovery should find all four.
    let t1_l = 120.0;
    let t1_r = 80.0;
    let t2_l = 90.0;
    let t2_r = 60.0;
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_4;

    let pl = synth_cz(t1_l, t1_r, t2_l, t2_r, omega, theta);
    let rec = recover_ad_pd_2q_from_cz_theta(&pl, omega, theta).unwrap();

    assert_abs_diff_eq!(rec.t1_l, t1_l, epsilon = 1e-10);
    assert_abs_diff_eq!(rec.t1_r, t1_r, epsilon = 1e-10);
    assert_abs_diff_eq!(rec.t2_l, t2_l, epsilon = 1e-10);
    assert_abs_diff_eq!(rec.t2_r, t2_r, epsilon = 1e-10);
}

#[test]
fn round_trip_2q_at_pi_over_3() {
    let pl = synth_cz(200.0, 200.0, 150.0, 150.0, 1.5, std::f64::consts::FRAC_PI_3);
    let rec = recover_ad_pd_2q_from_cz_theta(&pl, 1.5, std::f64::consts::FRAC_PI_3).unwrap();
    assert_abs_diff_eq!(rec.t1_l, 200.0, epsilon = 1e-10);
    assert_abs_diff_eq!(rec.t1_r, 200.0, epsilon = 1e-10);
    assert_abs_diff_eq!(rec.t2_l, 150.0, epsilon = 1e-10);
    assert_abs_diff_eq!(rec.t2_r, 150.0, epsilon = 1e-10);
}

#[test]
fn recovery_residual_small_for_pure_ad_pd() {
    // On clean AD+PD synthesis, the degenerate rate pairs should
    // coincide to machine precision -> residual ~ 0.
    let pl = synth_cz(100.0, 80.0, 80.0, 60.0, 1.0, std::f64::consts::FRAC_PI_4);
    let residual = cz_recovery_residual(&pl);
    assert!(
        residual < 1e-10,
        "residual for clean AD+PD should be near zero: {}",
        residual
    );
}

#[test]
fn recovery_residual_nonzero_under_model_mismatch() {
    // Build a model where the degenerate pairs explicitly *differ* --
    // simulating measured rates that don't fit the pure-AD+PD form.
    let supports: Vec<_> = ["IX", "IY", "XI", "YI"]
        .iter()
        .map(|s| PauliString::from_label(s).unwrap())
        .collect();
    // IX, IY differ by 20% (not allowed under pure AD+PD for CZ).
    let rates = vec![0.003, 0.0036, 0.002, 0.002];
    let pl = PauliLindbladModel::new(supports, rates);
    let residual = cz_recovery_residual(&pl);
    assert!(
        residual > 1e-4,
        "expected residual to flag the mismatch, got {}",
        residual
    );
}

#[test]
fn recovery_returns_none_on_negative_rates() {
    let supports = vec![
        PauliString::from_label("IX").unwrap(),
        PauliString::from_label("IY").unwrap(),
        PauliString::from_label("XI").unwrap(),
        PauliString::from_label("YI").unwrap(),
        PauliString::from_label("IZ").unwrap(),
        PauliString::from_label("ZI").unwrap(),
    ];
    // lambda_ix is zero -> recovery can't infer beta_down_r.
    let rates = vec![0.0, 0.0, 0.001, 0.001, 0.002, 0.002];
    let pl = PauliLindbladModel::new(supports, rates);
    assert!(recover_ad_pd_2q_from_cz_theta(&pl, 1.0, std::f64::consts::FRAC_PI_4).is_none());
}

#[test]
fn round_trip_2q_validation_workflow() {
    // End-to-end validation story: 4 device params -> 15 PL rates ->
    // recover 4 params -> compare.
    let t1_l = 250.0;
    let t2_l = 180.0;
    let t1_r = 300.0;
    let t2_r = 220.0;
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_4;

    // 1. Forward: physics -> rates.
    let pl = synth_cz(t1_l, t1_r, t2_l, t2_r, omega, theta);

    // 2. Consistency check: on clean data the 2-fold pair residual = 0.
    assert!(cz_recovery_residual(&pl) < 1e-10);

    // 3. Inverse.
    let rec = recover_ad_pd_2q_from_cz_theta(&pl, omega, theta).unwrap();

    // 4. Closure.
    assert_abs_diff_eq!(rec.t1_l, t1_l, epsilon = 1e-10);
    assert_abs_diff_eq!(rec.t2_l, t2_l, epsilon = 1e-10);
    assert_abs_diff_eq!(rec.t1_r, t1_r, epsilon = 1e-10);
    assert_abs_diff_eq!(rec.t2_r, t2_r, epsilon = 1e-10);
}
