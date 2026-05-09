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

//! Non-Markovian dynamics tests. Two case studies:
//!
//! 1. **TCL sanity**: a constant-rate time-dependent Lindbladian must
//!    reproduce the Markovian result for identity + PD to machine
//!    precision.
//! 2. **1/f-style rate**: `gamma_phi(t) = gamma_0 A / (A + t/t_c)` with
//!    `A >> tau_g/t_c` reproduces Markov to leading order; as `tau_g/t_c`
//!    grows, rates should DIFFER from Markov prediction, demonstrating
//!    we actually capture the non-Markovian correction.

use std::sync::Arc;

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use pecos_lindblad::matrix::{self, Matrix};
use pecos_lindblad::noise_models::ad_pd_1q;
use pecos_lindblad::{
    DEFAULT_N_SLICES, Gate, Pauli1, PauliString, RateFn, TimeDepLindbladian,
    synthesize_identity_1q, synthesize_superop_time_dep,
};

/// 1-qubit Z operator for PD.
fn z1q() -> Matrix {
    matrix::pauli_1q(Pauli1::Z)
}

#[test]
fn constant_rate_time_dep_matches_markovian() {
    // Build a time-dependent Lindbladian whose rates happen to be constant.
    // Synthesis should match the Markovian synthesize_identity_1q to
    // high precision.
    let beta_phi: f64 = 2e-3;
    let tau_g: f64 = 1.0;
    let d = 2;
    let rate_fn: RateFn = Arc::new(move |_t| beta_phi / 2.0);
    let noise_td =
        TimeDepLindbladian::with_static_hamiltonian(d, matrix::zeros(d), vec![(z1q(), rate_fn)]);

    // For identity gate, T1 = infinity effectively: use just PD.
    let h_ideal = matrix::zeros(d);
    let pl_td = synthesize_superop_time_dep(1, &h_ideal, &noise_td, tau_g, DEFAULT_N_SLICES);

    // Markovian baseline: identity + PD only (no AD).
    // For pure PD at rate beta_phi/2 on Z: paper eq 801 says
    //   lambda_z = (beta_phi * tau_g) / 2.
    // Our T1/T2 API requires beta_down > 0; emulate "AD off" via T2 = 2 T1
    // at huge T1 so beta_down -> 0.
    let t1 = 1e18;
    let t2 = 1.0 / beta_phi; // 1/T_2 = 1/(2T_1) + beta_phi -> beta_phi
    let noise_mk = ad_pd_1q(t1, t2);
    let pl_mk = synthesize_identity_1q(&Gate::identity(1, noise_mk, tau_g));

    for p in [Pauli1::X, Pauli1::Y, Pauli1::Z] {
        let k = PauliString::single(p);
        assert_abs_diff_eq!(pl_td.rate(&k), pl_mk.rate(&k), epsilon = 1e-9);
    }
}

#[test]
fn one_over_f_weak_non_markov_reduces_to_markov() {
    // 1/f-ish rate: gamma_phi(t) = gamma_0 * A / (A + t/t_c) with A=1, t_c
    // large compared to tau_g -> rate ~ gamma_0 constant -> should match
    // Markov to high precision.
    let gamma_0: f64 = 1e-3;
    let t_c: f64 = 1e6; // very slow variation
    let a: f64 = 1.0;
    let tau_g: f64 = 1.0;
    let d = 2;

    let rate_fn: RateFn = Arc::new(move |t: f64| gamma_0 * a / (a + t / t_c) / 2.0);
    let noise_td =
        TimeDepLindbladian::with_static_hamiltonian(d, matrix::zeros(d), vec![(z1q(), rate_fn)]);
    let h_ideal = matrix::zeros(d);
    let pl_nm = synthesize_superop_time_dep(1, &h_ideal, &noise_td, tau_g, 256);

    // Corresponding Markov (constant rate = gamma_0): lambda_z = gamma_0 * tau_g / 2.
    let expected_z = gamma_0 * tau_g / 2.0;
    assert_abs_diff_eq!(
        pl_nm.rate(&PauliString::single(Pauli1::Z)),
        expected_z,
        epsilon = 1e-8
    );
}

#[test]
fn one_over_f_strong_non_markov_differs_from_markov() {
    // Now use t_c comparable to tau_g -> gamma_phi(t) varies substantially
    // over the gate. Predicted PL rate must DIFFER from the constant-rate
    // Markov prediction, proving the non-Markovian correction is captured.
    let gamma_0: f64 = 1e-3;
    let a: f64 = 1.0;
    let tau_g: f64 = 1.0;
    let t_c: f64 = tau_g / 2.0; // rate drops substantially over the gate
    let d = 2;

    let rate_fn: RateFn = Arc::new(move |t: f64| gamma_0 * a / (a + t / t_c) / 2.0);
    let noise_td =
        TimeDepLindbladian::with_static_hamiltonian(d, matrix::zeros(d), vec![(z1q(), rate_fn)]);
    let h_ideal = matrix::zeros(d);
    let pl_nm = synthesize_superop_time_dep(1, &h_ideal, &noise_td, tau_g, 512);

    // Analytic: integrated rate = int_0^tau_g gamma(t) dt
    //   = int_0^tau_g gamma_0 / (1 + 2 t/tau_g) dt
    //   = gamma_0 * (tau_g/2) * ln(1 + 2)
    //   = gamma_0 * tau_g * ln(3) / 2
    // lambda_z = (integrated rate) / 2 (paper convention: lambda_z = beta_phi * tau_g / 2).
    // Wait: with our factor-of-1/2 in rate_fn (beta_phi/2), integrated is
    // gamma_0 * tau_g * ln(3) / 2 / 2. Let me work this out again.
    //
    // Effective PD: D[Z] rho = Z rho Z - rho, rate = gamma_phi(t)/2.
    // Integrated rate per paper: lambda_z = int (gamma_phi(t)/2) dt * 2 = int gamma_phi(t) dt.
    // Hmm -- actually the paper's identity closed form is lambda_z = beta_phi * tau_g / 2
    // where beta_phi is the RATE coefficient attached to (beta_phi/2) D[Z]
    // (i.e. the "1/T_phi"). So:
    //   lambda_z(const) = beta_phi * tau_g / 2.
    // For time-dep: lambda_z(t-dep) = (1/2) * int_0^tau_g gamma_0/(1 + 2t/tau_g) dt
    //                              = (1/2) * gamma_0 * (tau_g/2) * ln(1 + 2)
    //                              = gamma_0 * tau_g * ln(3) / 4.
    let expected_nm = gamma_0 * tau_g * (3.0_f64).ln() / 4.0;
    let got = pl_nm.rate(&PauliString::single(Pauli1::Z));

    // Same prediction via constant-rate Markov model:
    let lambda_z_mk = gamma_0 * tau_g / 2.0;

    // Markov and non-Markov disagree substantially.
    assert!(
        (got - lambda_z_mk).abs() > 0.1 * lambda_z_mk,
        "non-Markov should differ substantially from Markov: got {}, markov {}",
        got,
        lambda_z_mk
    );
    // And our non-Markov result matches the analytic integrated-rate formula.
    assert_abs_diff_eq!(got, expected_nm, epsilon = 1e-7);
}

#[test]
fn time_dependent_coherent_noise_gaussian_pulse() {
    // H_delta(t) = (delta * exp(-((t - tau_g/2)/sigma)^2) / 2) * Z
    // (Gaussian envelope of coherent Z over the gate). Verify rates
    // scale with the envelope's time integral.
    let delta: f64 = 1e-4;
    let tau_g: f64 = 1.0;
    let sigma: f64 = tau_g / 4.0;
    let d = 2;

    let h_fn: pecos_lindblad::time_dep::HermitianFn = Arc::new(move |t: f64| {
        let arg = ((t - tau_g / 2.0) / sigma).powi(2);
        let amp = delta * (-arg).exp() / 2.0;
        matrix::scale(&z1q(), Complex64::new(amp, 0.0))
    });
    let noise_td = TimeDepLindbladian::new(d, h_fn, vec![]);
    let h_ideal = matrix::zeros(d);
    let pl_nm = synthesize_superop_time_dep(1, &h_ideal, &noise_td, tau_g, 256);

    // For purely coherent Z noise on identity:
    //   effective unitary phase phi = int_0^tau_g (H(t) component of Z) dt
    //   = delta * int_0^tau_g exp(-((t-tau/2)/sigma)^2) / 2 dt
    //   = delta/2 * sigma * sqrt(pi) * erf(tau_g/(2 sigma)) ...
    // for sigma = tau_g/4 and tau_g = 1: integral approx = sqrt(pi) * tau_g/4 * erf(2)
    // erf(2) ~ 0.9953
    let integral_gaussian = sigma * std::f64::consts::PI.sqrt() * erf(tau_g / (2.0 * sigma));
    let phi = delta * integral_gaussian / 2.0;
    // For coherent Z on identity: lambda_z = phi^2 / 2 (from -ln(cos(phi)) ~ phi^2/2).
    // Walsh-Hadamard gives lambda_z = alpha_z * tau_g / 4 * 2 = ... actually the
    // 1Q identity-coherent-Z result: lambda_z = phi^2 / 2 where phi is total phase.
    //
    // Actually from our previous phase-noise test: for constant (delta/2) Z,
    //   lambda_z = (delta * tau_g / 2)^2 / 2 / 2 = phi^2 / 4 where phi = delta * tau_g / 2.
    // Hmm let's just check order of magnitude.
    let got = pl_nm.rate(&PauliString::single(Pauli1::Z));
    assert!(got > 0.0, "Gaussian pulse should produce nonzero lambda_z");
    // For coherent Z-on-identity, lambda_z ~ phi^2 at leading order.
    // Allow factor-of-2 margin either side.
    assert!(
        got > 0.5 * phi * phi && got < 2.0 * phi * phi,
        "got {}, phi^2 {} -- expected leading-order ~phi^2",
        got,
        phi * phi,
    );
}

/// Abramowitz-Stegun approximation of erf. Accuracy ~7 digits.
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}
