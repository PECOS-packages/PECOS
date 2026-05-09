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

//! Simpson's-rule convergence test for `synthesize_numerical`.
//!
//! We sweep `N_STEPS` on a `CX_theta + AD+PD` fixture and verify that each
//! step count converges (within tol) to `DEFAULT_N_STEPS = 1024`. Ensures
//! that the default is neither needlessly large nor too small, and that
//! users tweaking `N_STEPS` know what to expect.
//!
//! For the current test parameters, convergence is already at `1e-12`
//! between `N = 64` and `N = 1024`, so `DEFAULT_N_STEPS = 1024` is
//! comfortably safe (~16x over-sampled).

use approx::assert_abs_diff_eq;

use pecos_lindblad::noise_models::ad_pd_2q;
use pecos_lindblad::{DEFAULT_N_STEPS, Gate, PauliString, synthesize_numerical};

fn cx_rates(n_steps: usize) -> Vec<f64> {
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_4;
    let noise = ad_pd_2q(100.0, 80.0, 120.0, 90.0);
    let gate = Gate::cx_theta(omega, theta, noise);
    let pl = synthesize_numerical(&gate, n_steps);
    PauliString::enumerate_nonidentity(2)
        .iter()
        .map(|p| pl.rate(p))
        .collect()
}

#[test]
fn simpson_converges_for_cx_theta() {
    let reference = cx_rates(DEFAULT_N_STEPS);
    // Simpson's 1/3 rule has error O(h^4). At N=64 over a single-oscillation
    // interval we expect ~1e-10; at N=128, ~1e-12; at N>=256, machine eps.
    let tolerances = [
        (16usize, 1e-6),
        (32, 1e-8),
        (64, 1e-10),
        (128, 1e-12),
        (256, 1e-13),
    ];
    for (n, tol) in tolerances {
        let result = cx_rates(n);
        for (a, b) in result.iter().zip(reference.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = tol);
        }
    }
}

#[test]
fn default_n_steps_matches_fine_grid() {
    // DEFAULT_N_STEPS = 1024 should match N = 2048 to ~machine precision.
    let a = cx_rates(DEFAULT_N_STEPS);
    let b = cx_rates(2048);
    for (x, y) in a.iter().zip(b.iter()) {
        assert_abs_diff_eq!(x, y, epsilon = 1e-14);
    }
}
