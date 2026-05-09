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

//! Positivity stress test: scan noise strength `beta/omega` across a range
//! and verify `synthesize_numerical` never panics and produces only
//! non-negative rates. Documents the regime where the first-order PL model
//! stays self-consistent (leading order of Magnus assumes weak coupling).

use pecos_lindblad::noise_models::ad_pd_2q;
use pecos_lindblad::{DEFAULT_N_STEPS, Gate, PauliString, synthesize_numerical};

fn max_negative_rate(rates: &[f64]) -> f64 {
    rates
        .iter()
        .copied()
        .filter(|&r| r < 0.0)
        .map(|r| -r)
        .fold(0.0, f64::max)
}

#[test]
fn cx_theta_rates_non_negative_across_weak_regime() {
    // beta/omega from 1e-5 to 1e-1 (a factor of 10^4 sweep). This brackets
    // realistic device regimes: T1/tau_g >> 1 on IBM-like hardware maps to
    // beta/omega ~ 1e-4 or smaller.
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_4;
    let tau_g = theta / omega;

    // Run the sweep by holding T2 = 2*T1 (pure-dephasing-free) and varying T1.
    // Ratio beta_down * tau_g tracks inverse T1. Values: 0.1/tau_g ... 1e-5/tau_g.
    let ratios = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1];
    for &r in &ratios {
        let t1 = tau_g / r;
        let t2 = 2.0 * t1;
        let noise = ad_pd_2q(t1, t1, t2, t2);
        let gate = Gate::cx_theta(omega, theta, noise);
        let pl = synthesize_numerical(&gate, DEFAULT_N_STEPS);
        let rates: Vec<f64> = PauliString::enumerate_nonidentity(2)
            .iter()
            .map(|p| pl.rate(p))
            .collect();
        assert!(
            max_negative_rate(&rates) < 1e-12,
            "negative rate at beta/omega={}: max negative = {}",
            r,
            max_negative_rate(&rates),
        );
        // Sanity: highest rate should scale linearly with r to leading order.
        let total: f64 = rates.iter().sum();
        assert!(
            total > 0.0 && total < 2.0,
            "total rate out of range at beta/omega={}: total={}",
            r,
            total,
        );
    }
}

#[test]
fn identity_2q_rates_non_negative_across_weak_regime() {
    // Same sweep on identity + AD+PD (Phase 1 exact-fixture regime).
    let tau_g = 10.0;
    let ratios = [1e-6, 1e-4, 1e-2, 1e-1];
    for &r in &ratios {
        let t1 = tau_g / r;
        let t2 = 2.0 * t1;
        let noise = ad_pd_2q(t1, t1, t2, t2);
        let gate = Gate::identity(2, noise, tau_g);
        let pl = synthesize_numerical(&gate, DEFAULT_N_STEPS);
        let rates: Vec<f64> = PauliString::enumerate_nonidentity(2)
            .iter()
            .map(|p| pl.rate(p))
            .collect();
        assert!(
            max_negative_rate(&rates) < 1e-12,
            "negative rate at beta/omega={}: max negative = {}",
            r,
            max_negative_rate(&rates),
        );
    }
}
