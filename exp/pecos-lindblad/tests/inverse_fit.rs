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

//! Inverse fit / parameter recovery tests. Closes the "validation loop":
//! forward synthesis produces rates; inverse recovery back-solves physical
//! parameters from rates. At the 1Q identity level, the recovery is
//! analytic and must be a bit-exact round-trip of
//! [`noise_models::ad_pd_1q`] -> [`synthesize_identity_1q`].

use approx::assert_abs_diff_eq;

use pecos_lindblad::noise_models::{ad_pd_1q, recover_t1_t2_from_identity_1q};
use pecos_lindblad::{Gate, Pauli1, PauliLindbladModel, PauliString, synthesize_identity_1q};

fn synth_1q(t1: f64, t2: f64, tau_g: f64) -> PauliLindbladModel {
    synthesize_identity_1q(&Gate::identity(1, ad_pd_1q(t1, t2), tau_g))
}

#[test]
fn round_trip_recovery_1q_ad_pd() {
    for (t1, t2, tau) in [
        (100.0, 80.0, 1.0),
        (300.0, 200.0, 0.5),
        (50.0, 50.0, 2.0), // T_2 = T_1 case
    ] {
        let pl = synth_1q(t1, t2, tau);
        let (t1_rec, t2_rec) = recover_t1_t2_from_identity_1q(&pl, tau).unwrap();
        assert_abs_diff_eq!(t1_rec, t1, epsilon = 1e-10);
        assert_abs_diff_eq!(t2_rec, t2, epsilon = 1e-10);
    }
}

#[test]
fn recovery_handles_t2_equals_2_t1_limit() {
    // Pure-T1 limit (no dephasing): T_2 = 2 T_1, so lambda_z = 0 exactly.
    let t1 = 150.0;
    let t2 = 2.0 * t1;
    let tau = 1.0;
    let pl = synth_1q(t1, t2, tau);
    let (t1_rec, t2_rec) = recover_t1_t2_from_identity_1q(&pl, tau).unwrap();
    assert_abs_diff_eq!(t1_rec, t1, epsilon = 1e-10);
    assert_abs_diff_eq!(t2_rec, t2, epsilon = 1e-10);
}

#[test]
fn recovery_returns_none_on_inconsistent_rates() {
    // lambda_x=0 => cannot determine T_1.
    let pl = PauliLindbladModel::new(
        vec![
            PauliString::single(Pauli1::X),
            PauliString::single(Pauli1::Z),
        ],
        vec![0.0, 0.01],
    );
    assert!(recover_t1_t2_from_identity_1q(&pl, 1.0).is_none());
}

#[test]
fn recovery_returns_none_on_zero_tau() {
    let pl = synth_1q(100.0, 80.0, 1.0);
    assert!(recover_t1_t2_from_identity_1q(&pl, 0.0).is_none());
}

#[test]
fn compose_independent_sums_rates() {
    let a = PauliLindbladModel::new(
        vec![
            PauliString::from_label("IX").unwrap(),
            PauliString::from_label("ZZ").unwrap(),
        ],
        vec![0.001, 0.005],
    );
    let b = PauliLindbladModel::new(
        vec![
            PauliString::from_label("IX").unwrap(),
            PauliString::from_label("XI").unwrap(),
        ],
        vec![0.002, 0.003],
    );
    let ab = a.compose_independent(&b);
    // Expected support: IX (merged), XI, ZZ. Rates: IX=0.003, XI=0.003, ZZ=0.005.
    assert_abs_diff_eq!(
        ab.rate(&PauliString::from_label("IX").unwrap()),
        0.003,
        epsilon = 1e-14
    );
    assert_abs_diff_eq!(
        ab.rate(&PauliString::from_label("XI").unwrap()),
        0.003,
        epsilon = 1e-14
    );
    assert_abs_diff_eq!(
        ab.rate(&PauliString::from_label("ZZ").unwrap()),
        0.005,
        epsilon = 1e-14
    );
    assert_eq!(ab.supports.len(), 3);
}

#[test]
fn compose_commutes() {
    let a = PauliLindbladModel::new(vec![PauliString::single(Pauli1::X)], vec![0.01]);
    let b = PauliLindbladModel::new(vec![PauliString::single(Pauli1::Z)], vec![0.02]);
    let ab = a.compose_independent(&b);
    let ba = b.compose_independent(&a);
    for p in [Pauli1::X, Pauli1::Y, Pauli1::Z] {
        let k = PauliString::single(p);
        assert_abs_diff_eq!(ab.rate(&k), ba.rate(&k), epsilon = 1e-14);
    }
}

#[test]
fn round_trip_validation_workflow() {
    // End-to-end: predict rates from (T1, T2), then recover T1/T2 from
    // those rates, verify the loop closes.
    let t1_nominal = 250.0;
    let t2_nominal = 180.0;
    let tau = 0.5;

    // 1. Forward: physics -> rates.
    let predicted = synth_1q(t1_nominal, t2_nominal, tau);

    // 2. Inverse: rates -> physics.
    let (t1_back, t2_back) = recover_t1_t2_from_identity_1q(&predicted, tau).unwrap();

    // 3. Closure.
    assert_abs_diff_eq!(t1_back, t1_nominal, epsilon = 1e-10);
    assert_abs_diff_eq!(t2_back, t2_nominal, epsilon = 1e-10);
}
