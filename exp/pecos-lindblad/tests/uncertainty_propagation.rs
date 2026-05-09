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

//! Monte-Carlo uncertainty propagation tests. For 1Q identity + AD+PD the
//! rates are exactly `lambda_x = lambda_y = beta_down tau/4` and
//! `lambda_z = beta_phi tau/2`, so uncertainty propagation is analytic:
//!
//!   d lambda_x / d T_1 = -tau / (4 T_1^2)
//!   d lambda_z / d T_1 = -tau / (2 * 2 T_1^2) = -tau / (4 T_1^2) (from 1/(2T_1))
//!   d lambda_z / d T_2 = -tau / (2 T_2^2)
//!
//! For small Gaussian jitter, first-order MC std should match these
//! derivatives times the input sigma, to within Monte-Carlo error.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use pecos_lindblad::noise_models::{ad_pd_1q, propagate_uncertainty};
use pecos_lindblad::{Gate, Pauli1, PauliString, synthesize_identity_1q};

fn gaussian(rng: &mut StdRng, mean: f64, sigma: f64) -> f64 {
    // Box-Muller.
    let u1: f64 = rng.random_range(1e-15f64..1.0f64);
    let u2: f64 = rng.random_range(0.0f64..1.0f64);
    mean + sigma * f64::sqrt(-2.0 * f64::ln(u1)) * f64::cos(2.0 * std::f64::consts::PI * u2)
}

#[test]
fn uncertainty_propagation_gives_expected_std_scale() {
    let t1_mean = 100.0;
    let t2_mean = 80.0;
    let t1_sigma = 5.0; // 5% relative
    let t2_sigma = 4.0; // 5% relative
    let tau_g = 1.0;
    let n_samples = 2000;

    let mut rng = StdRng::seed_from_u64(42);
    let unc = propagate_uncertainty(n_samples, |_k| {
        let t1 = gaussian(&mut rng, t1_mean, t1_sigma).max(t2_mean / 2.0 + 1e-3);
        let t2 = gaussian(&mut rng, t2_mean, t2_sigma)
            .min(2.0 * t1)
            .max(1e-3);
        let noise = ad_pd_1q(t1, t2);
        let gate = Gate::identity(1, noise, tau_g);
        synthesize_identity_1q(&gate)
    });

    // Sanity: means should be close to central-parameter prediction.
    let beta_down = 1.0 / t1_mean;
    let beta_phi = 1.0 / t2_mean - 1.0 / (2.0 * t1_mean);
    let expected_mean_x = beta_down * tau_g / 4.0;
    let expected_mean_z = beta_phi * tau_g / 2.0;

    let got_mean_x = unc.mean(&PauliString::single(Pauli1::X));
    let got_mean_z = unc.mean(&PauliString::single(Pauli1::Z));
    // Monte-Carlo sample-mean error ~ std / sqrt(N); allow 5 sigma.
    let got_std_x = unc.std(&PauliString::single(Pauli1::X));
    let got_std_z = unc.std(&PauliString::single(Pauli1::Z));
    let mean_err_tol_x = 5.0 * got_std_x / (n_samples as f64).sqrt();
    let mean_err_tol_z = 5.0 * got_std_z / (n_samples as f64).sqrt();
    // Add small bias tolerance for the boundary clamping above.
    assert!(
        (got_mean_x - expected_mean_x).abs() < 5.0 * mean_err_tol_x + 0.05 * expected_mean_x,
        "mean_x: got {}, expected {}, diff {}",
        got_mean_x,
        expected_mean_x,
        (got_mean_x - expected_mean_x).abs(),
    );
    assert!(
        (got_mean_z - expected_mean_z).abs() < 5.0 * mean_err_tol_z + 0.05 * expected_mean_z,
        "mean_z: got {}, expected {}, diff {}",
        got_mean_z,
        expected_mean_z,
        (got_mean_z - expected_mean_z).abs(),
    );

    // Std should be non-trivial (not collapsed to zero) and in the right ballpark
    // via first-order propagation: sigma_lambda_x ~ tau/(4 T_1^2) * sigma_T1.
    let predicted_sigma_x = tau_g / (4.0 * t1_mean.powi(2)) * t1_sigma;
    assert!(
        got_std_x > 0.3 * predicted_sigma_x && got_std_x < 3.0 * predicted_sigma_x,
        "sigma_x: got {}, predicted (1st-order) {}, out of factor-3 range",
        got_std_x,
        predicted_sigma_x,
    );
}

#[test]
fn uncertainty_std_scales_linearly_with_input_sigma() {
    // Double the input sigma -> output sigma should roughly double
    // (first-order Taylor expansion).
    let t1_mean = 100.0;
    let t2_mean = 80.0;
    let tau_g = 1.0;

    let run = |sigma_t1: f64, seed: u64| -> f64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let unc = propagate_uncertainty(1000, |_k| {
            let t1 = gaussian(&mut rng, t1_mean, sigma_t1).max(t2_mean / 2.0 + 1e-3);
            let noise = ad_pd_1q(t1, t2_mean);
            let gate = Gate::identity(1, noise, tau_g);
            synthesize_identity_1q(&gate)
        });
        unc.std(&PauliString::single(Pauli1::X))
    };

    let s1 = run(2.0, 11);
    let s2 = run(4.0, 11);
    // Ratio should be ~2, allow 25% slack for MC noise.
    let ratio = s2 / s1;
    assert!(
        (ratio - 2.0).abs() < 0.5,
        "std did not scale linearly: s1={}, s2={}, ratio={}",
        s1,
        s2,
        ratio
    );
}
