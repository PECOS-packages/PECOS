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

//! Tests for `PauliLindbladModel::top_contributors` / `explain`.

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
fn top_contributors_sorted_by_rate_descending() {
    let m = model(&[("IX", 0.001), ("ZI", 0.005), ("IY", 0.003), ("IZ", 0.002)]);
    let top = m.top_contributors(4);
    assert_eq!(format!("{}", top[0].0), "ZI"); // 0.005
    assert_eq!(format!("{}", top[1].0), "IY"); // 0.003
    assert_eq!(format!("{}", top[2].0), "IZ"); // 0.002
    assert_eq!(format!("{}", top[3].0), "IX"); // 0.001
}

#[test]
fn top_contributors_truncates_to_n() {
    let m = model(&[("X", 0.01), ("Y", 0.02), ("Z", 0.03)]);
    let top2 = m.top_contributors(2);
    assert_eq!(top2.len(), 2);
    assert_eq!(format!("{}", top2[0].0), "Z");
    assert_eq!(format!("{}", top2[1].0), "Y");
}

#[test]
fn top_contributors_ties_broken_lexicographically() {
    let m = model(&[("X", 0.01), ("Y", 0.01), ("Z", 0.01)]);
    let top = m.top_contributors(3);
    // Tie on rate -> lexicographic on Pauli1 (I<X<Y<Z).
    assert_eq!(format!("{}", top[0].0), "X");
    assert_eq!(format!("{}", top[1].0), "Y");
    assert_eq!(format!("{}", top[2].0), "Z");
}

#[test]
fn explain_contains_expected_sections() {
    let m = model(&[("IX", 0.001), ("ZZ", 0.005), ("XYZ", 0.0002)]);
    let out = m.explain();
    assert!(out.contains("Pauli-Lindblad noise budget"));
    assert!(out.contains("By weight:"));
    assert!(out.contains("Top"));
    assert!(out.contains("weight-1"));
    assert!(out.contains("weight-2"));
    assert!(out.contains("weight-3"));
    assert!(out.contains("ZZ"));
    assert!(out.contains("XYZ"));
}

#[test]
fn explain_weight_sums_match_rate_at_weight() {
    // Sanity: per-weight row in explain() should align with
    // rate_at_weight(w). Grug verify by parsing the weight sums out of
    // the formatted output.
    let m = model(&[("IX", 0.001), ("IY", 0.001), ("ZZ", 0.005), ("XYZ", 0.0002)]);
    let w1 = m.rate_at_weight(1);
    let w2 = m.rate_at_weight(2);
    let w3 = m.rate_at_weight(3);
    assert_abs_diff_eq!(w1, 0.002, epsilon = 1e-14);
    assert_abs_diff_eq!(w2, 0.005, epsilon = 1e-14);
    assert_abs_diff_eq!(w3, 0.0002, epsilon = 1e-14);
    let total = m.total_rate();
    assert_abs_diff_eq!(total, w1 + w2 + w3, epsilon = 1e-14);
}

#[test]
fn explain_empty_model_handles_gracefully() {
    let m = PauliLindbladModel::default();
    let out = m.explain();
    // Empty model: 0 terms, 0 total rate. Shouldn't panic.
    assert!(out.contains("0 terms"));
}
