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

//! Serde round-trip tests (gated on the `serde` feature). Users can cache
//! expensive synthesis results and reload them later.

#![cfg(feature = "serde")]

use pecos_lindblad::noise_models::ad_pd_2q;
use pecos_lindblad::{
    DEFAULT_N_STEPS, Gate, Pauli1, PauliLindbladModel, PauliString, synthesize_numerical,
};

#[test]
fn pauli1_round_trip() {
    for p in [Pauli1::I, Pauli1::X, Pauli1::Y, Pauli1::Z] {
        let json = serde_json::to_string(&p).unwrap();
        let restored: Pauli1 = serde_json::from_str(&json).unwrap();
        assert_eq!(p, restored);
    }
}

#[test]
fn pauli_string_round_trip() {
    for s in ["I", "X", "Y", "Z", "IX", "XYZI", "ZZZZZ"] {
        let ps = PauliString::from_label(s).unwrap();
        let json = serde_json::to_string(&ps).unwrap();
        let restored: PauliString = serde_json::from_str(&json).unwrap();
        assert_eq!(ps, restored);
    }
}

#[test]
fn pauli_lindblad_model_round_trip_via_cx_theta() {
    // Synthesize a non-trivial 2Q CX_theta model, serialize, and verify
    // round-trip is bit-exact.
    let t1 = 100.0;
    let t2 = 80.0;
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_4;
    let noise = ad_pd_2q(t1, t1, t2, t2);
    let gate = Gate::cx_theta(omega, theta, noise);
    let pl = synthesize_numerical(&gate, DEFAULT_N_STEPS);

    let json = serde_json::to_string(&pl).unwrap();
    let restored: PauliLindbladModel = serde_json::from_str(&json).unwrap();

    assert_eq!(pl.supports.len(), restored.supports.len());
    for (a, b) in pl.supports.iter().zip(&restored.supports) {
        assert_eq!(a, b);
    }
    for (a, b) in pl.rates.iter().zip(&restored.rates) {
        assert_eq!(a.to_bits(), b.to_bits(), "rate mismatch {} vs {}", a, b);
    }
}

#[test]
fn pauli_lindblad_model_json_is_human_readable() {
    // Sanity: verify the JSON shape is predictable enough for users to
    // inspect / hand-edit.
    let pl = PauliLindbladModel::new(
        vec![
            PauliString::from_label("X").unwrap(),
            PauliString::from_label("Z").unwrap(),
        ],
        vec![0.001, 0.002],
    );
    let json = serde_json::to_string(&pl).unwrap();
    // Expect something like: {"supports":[...],"rates":[0.001,0.002]}
    assert!(json.contains("\"rates\""));
    assert!(json.contains("\"supports\""));
    assert!(json.contains("0.001") || json.contains("1e-3"));
}
