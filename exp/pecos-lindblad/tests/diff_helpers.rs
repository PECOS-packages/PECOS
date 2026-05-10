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

//! Tests for validation-oriented diff helpers on `PauliLindbladModel`.

use approx::assert_abs_diff_eq;

use pecos_lindblad::{PauliLindbladModel, PauliString};

fn model(entries: &[(&str, f64)]) -> PauliLindbladModel {
    let supports: Vec<_> = entries
        .iter()
        .map(|(s, _)| PauliString::from_label(s).unwrap())
        .collect();
    let rates: Vec<_> = entries.iter().map(|(_, r)| *r).collect();
    PauliLindbladModel::new(supports, rates)
}

#[test]
fn diff_is_sorted_by_absolute_residual() {
    let pred = model(&[("IX", 0.001), ("IZ", 0.005), ("XI", 0.002)]);
    let meas = model(&[("IX", 0.0012), ("IZ", 0.008), ("XI", 0.002)]);
    let d = pred.diff(&meas);
    // IZ has largest absolute residual (0.003), IX next (0.0002), XI last (0).
    assert_eq!(format!("{}", d[0].0), "IZ");
    assert_eq!(format!("{}", d[1].0), "IX");
    assert_eq!(format!("{}", d[2].0), "XI");
    assert_abs_diff_eq!(d[0].3, -0.003, epsilon = 1e-12);
}

#[test]
fn residual_l2_and_max() {
    let a = model(&[("X", 0.01), ("Y", 0.02), ("Z", 0.03)]);
    let b = model(&[("X", 0.02), ("Y", 0.00), ("Z", 0.03)]);
    // Differences: X=-0.01, Y=+0.02, Z=0. L2 = sqrt(0.0001 + 0.0004) = sqrt(5)*0.01.
    assert_abs_diff_eq!(a.residual_l2(&b), (5.0_f64).sqrt() * 0.01, epsilon = 1e-12);
    let (worst_p, worst_r) = a.max_residual(&b).unwrap();
    assert_eq!(format!("{}", worst_p), "Y");
    assert_abs_diff_eq!(worst_r, 0.02, epsilon = 1e-12);
}

#[test]
fn residual_by_weight_classifies_correctly() {
    let a = model(&[("IX", 0.001), ("IY", 0.001), ("ZZ", 0.005), ("IZZ", 0.002)]);
    let b = model(&[("IX", 0.002), ("IY", 0.001), ("ZZ", 0.009), ("IZZ", 0.0025)]);
    let by_w = a.residual_by_weight(&b);
    // Differences: IX=-0.001 (wt 1), IY=0 (wt 1), ZZ=-0.004 (wt 2), IZZ=-0.0005 (wt 2).
    let w1 = by_w.iter().find(|(w, _)| *w == 1).unwrap().1;
    let w2 = by_w.iter().find(|(w, _)| *w == 2).unwrap().1;
    assert_abs_diff_eq!(w1, 0.001, epsilon = 1e-12);
    assert_abs_diff_eq!(w2, 0.0045, epsilon = 1e-12);
}

#[test]
fn diagnose_flags_large_weight_2_residual() {
    let pred = model(&[("IX", 0.001), ("IZ", 0.005)]);
    let meas = model(&[("IX", 0.001), ("IZ", 0.005), ("ZZ", 0.01)]);
    // Predicted model missing ZZ -- diagnose should flag weight-2 residual.
    let diagnoses = pred.diagnose_gap(&meas, 1e-5);
    assert!(
        diagnoses.iter().any(|m| m.contains("weight-2")),
        "expected weight-2 diagnosis in {:?}",
        diagnoses
    );
    assert!(
        diagnoses.iter().any(|m| m.contains("ZZ")),
        "expected ZZ mention in {:?}",
        diagnoses
    );
}

#[test]
fn diagnose_quiet_when_models_agree() {
    let pred = model(&[("X", 0.001), ("Z", 0.002)]);
    let meas = model(&[("X", 0.001), ("Z", 0.002)]);
    let diagnoses = pred.diagnose_gap(&meas, 1e-10);
    assert!(
        diagnoses.is_empty(),
        "expected no diagnoses, got {:?}",
        diagnoses
    );
}

#[test]
fn union_of_supports_considered() {
    // a has X, b has Z; diff should include both with appropriate zero.
    let a = model(&[("X", 0.003)]);
    let b = model(&[("Z", 0.005)]);
    let d = a.diff(&b);
    assert_eq!(d.len(), 2);
    // Largest |residual| is Z = 0 - 0.005 = -0.005.
    assert_eq!(format!("{}", d[0].0), "Z");
    assert_abs_diff_eq!(d[0].3, -0.005, epsilon = 1e-12);
}
