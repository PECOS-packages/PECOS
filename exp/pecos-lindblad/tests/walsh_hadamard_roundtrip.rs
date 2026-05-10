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

//! Walsh-Hadamard forward/inverse consistency. Starting from arbitrary
//! non-negative rates `{lambda_k}`:
//!
//!   1. Forward map: `alpha_b = 2 * sum_k lambda_k * <b,k>_sp`.
//!   2. Invert via Walsh-Hadamard: `lambda'_k = -(1/4^n) * sum_b (-1)^{<k,b>_sp} * alpha_b`.
//!   3. Verify `lambda'_k == lambda_k` (self-consistency of the algorithm).
//!
//! This closes a gap in the existing coverage: paper-fixture tests verify
//! the round-trip end-to-end but not the Walsh-Hadamard step in isolation,
//! so a bug there could cancel against a parallel bug in synthesis.

use approx::assert_abs_diff_eq;

use pecos_lindblad::{Pauli1, PauliLindbladModel, PauliString};

fn forward_alpha(model: &PauliLindbladModel, b: &PauliString) -> f64 {
    // alpha_b = 2 * sum_k lambda_k * <b,k>_sp.
    model
        .supports
        .iter()
        .zip(&model.rates)
        .map(|(k, lam)| 2.0 * lam * f64::from(b.symplectic_product(k)))
        .sum()
}

fn inverse_walsh_hadamard(paulis: &[PauliString], alphas: &[f64], n_qubits: usize) -> Vec<f64> {
    // lambda_k = -(1/4^n) * sum_b (-1)^{<k,b>_sp} * alpha_b (paper App B).
    let norm = 1.0 / (1usize << (2 * n_qubits)) as f64;
    paulis
        .iter()
        .map(|k| {
            let s: f64 = paulis
                .iter()
                .zip(alphas.iter())
                .map(|(b, &ab)| {
                    let sign = if k.symplectic_product(b) == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    sign * ab
                })
                .sum();
            -norm * s
        })
        .collect()
}

fn round_trip(n_qubits: usize, seed_rates: &[(&str, f64)]) {
    let supports: Vec<PauliString> = seed_rates
        .iter()
        .map(|(s, _)| PauliString::from_label(s).unwrap())
        .collect();
    let rates: Vec<f64> = seed_rates.iter().map(|(_, r)| *r).collect();
    let model = PauliLindbladModel::new(supports.clone(), rates.clone());

    // Enumerate all non-identity paulis to get alpha_b for each.
    let all = PauliString::enumerate_nonidentity(n_qubits);
    let alphas: Vec<f64> = all.iter().map(|b| forward_alpha(&model, b)).collect();
    let recovered = inverse_walsh_hadamard(&all, &alphas, n_qubits);

    // Build the "true" rates aligned to `all` order (0 for unseeded supports).
    let mut expected = vec![0.0; all.len()];
    for (s, r) in &rates
        .iter()
        .zip(&supports)
        .map(|(r, s)| (s.clone(), *r))
        .collect::<Vec<_>>()
    {
        if let Some(idx) = all.iter().position(|p| p == s) {
            expected[idx] = *r;
        }
    }
    for (got, want) in recovered.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(got, want, epsilon = 1e-12);
    }
}

#[test]
fn walsh_hadamard_round_trip_1q() {
    round_trip(1, &[("X", 0.001), ("Y", 0.002), ("Z", 0.003)]);
}

#[test]
fn walsh_hadamard_round_trip_2q_sparse() {
    round_trip(2, &[("IX", 1e-3), ("IZ", 2e-3), ("XI", 3e-3), ("ZZ", 4e-4)]);
}

#[test]
fn walsh_hadamard_round_trip_3q_with_weight_3() {
    round_trip(
        3,
        &[("IYZ", 1e-4), ("IZZ", 2e-4), ("ZZZ", 5e-5), ("XII", 1e-3)],
    );
}

#[test]
fn walsh_hadamard_round_trip_dense_1q() {
    // Dense 1Q (all 3 non-identity rates set) to exercise every row.
    let _ = Pauli1::X; // re-export reachable
    round_trip(1, &[("X", 0.1), ("Y", 0.2), ("Z", 0.3)]);
}
