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

//! Statistical validation of [`PauliLindbladModel::sample`]. Build a
//! known PL model and verify empirical flip probabilities match the
//! analytical per-term formula `p_k = (1 - exp(-2 lambda_k t)) / 2`.
//!
//! Uses a binomial-CI tolerance of `5 sigma = 5 * sqrt(p(1-p)/N)` to keep
//! flakiness very low.

use rand::SeedableRng;
use rand::rngs::StdRng;

use pecos_lindblad::{Pauli1, PauliLindbladModel, PauliString};

#[test]
fn single_term_flip_probability_matches_formula() {
    // 1Q PL with only lambda_x non-zero.
    let lambda_x = 0.05;
    let t_scale = 1.0;
    let expected_p: f64 = 0.5 * (1.0 - f64::exp(-2.0 * lambda_x * t_scale));
    let model = PauliLindbladModel::new(
        vec![
            PauliString::single(Pauli1::X),
            PauliString::single(Pauli1::Y),
            PauliString::single(Pauli1::Z),
        ],
        vec![lambda_x, 0.0, 0.0],
    );

    let n_samples: usize = 100_000;
    let mut rng = StdRng::seed_from_u64(12345);
    let mut x_hits = 0usize;
    let mut y_hits = 0usize;
    let mut z_hits = 0usize;
    for _ in 0..n_samples {
        let s = model.sample(t_scale, &mut rng);
        let ch = s.0[0];
        match ch {
            Pauli1::X => x_hits += 1,
            Pauli1::Y => y_hits += 1,
            Pauli1::Z => z_hits += 1,
            Pauli1::I => {}
        }
    }

    let p_hat_x = x_hits as f64 / n_samples as f64;
    let sigma = (expected_p * (1.0 - expected_p) / n_samples as f64).sqrt();
    let tol = 5.0 * sigma;
    assert!(
        (p_hat_x - expected_p).abs() < tol,
        "X flip rate: got {}, expected {}, diff {} > 5 sigma {}",
        p_hat_x,
        expected_p,
        (p_hat_x - expected_p).abs(),
        tol,
    );
    // Y/Z rates are 0; allow a small count due to X*X*X*X etc. not firing.
    assert_eq!(y_hits, 0, "Y should never fire (lambda_y = 0)");
    assert_eq!(z_hits, 0, "Z should never fire (lambda_z = 0)");
}

#[test]
fn multi_term_sample_respects_independent_bernoullis() {
    // Two independent Pauli terms. Empirical joint distribution over
    // {I, X, Z, XZ=Y} should match the 2x2 independent Bernoulli table.
    let lambda_x = 0.02;
    let lambda_z = 0.03;
    let t_scale = 1.0;
    let p_x: f64 = 0.5 * (1.0 - f64::exp(-2.0 * lambda_x * t_scale));
    let p_z: f64 = 0.5 * (1.0 - f64::exp(-2.0 * lambda_z * t_scale));

    let model = PauliLindbladModel::new(
        vec![
            PauliString::single(Pauli1::X),
            PauliString::single(Pauli1::Z),
        ],
        vec![lambda_x, lambda_z],
    );

    let n_samples: usize = 200_000;
    let mut rng = StdRng::seed_from_u64(7);
    // Bucket counts: [I, X, Y, Z]. Note X*Z (unordered) = Y in our
    // sign-dropped multiplication table.
    let mut counts = [0usize; 4];
    for _ in 0..n_samples {
        let s = model.sample(t_scale, &mut rng);
        let idx = match s.0[0] {
            Pauli1::I => 0,
            Pauli1::X => 1,
            Pauli1::Y => 2,
            Pauli1::Z => 3,
        };
        counts[idx] += 1;
    }
    let emp = |k: usize| counts[k] as f64 / n_samples as f64;

    // Analytical:
    //   P(I) = (1-p_x)(1-p_z)
    //   P(X) = p_x (1 - p_z)
    //   P(Z) = (1 - p_x) p_z
    //   P(Y) = p_x p_z
    let e_i = (1.0 - p_x) * (1.0 - p_z);
    let e_x = p_x * (1.0 - p_z);
    let e_z = (1.0 - p_x) * p_z;
    let e_y = p_x * p_z;

    for (idx, expected) in [(0, e_i), (1, e_x), (2, e_y), (3, e_z)] {
        let got = emp(idx);
        let sigma = (expected * (1.0 - expected) / n_samples as f64).sqrt();
        let tol = 5.0 * sigma;
        assert!(
            (got - expected).abs() < tol,
            "bucket {}: got {}, expected {}, diff {} > 5 sigma {}",
            idx,
            got,
            expected,
            (got - expected).abs(),
            tol,
        );
    }
}
