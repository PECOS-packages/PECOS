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

//! Cross-path consistency: different entry points must agree on inputs
//! they both handle.
//!
//! - [`synthesize_identity_1q`] (fast) vs [`synthesize_numerical`] (Simpson)
//!   for 1Q identity gate: should be bit-close (Simpson on constant
//!   integrand is exact up to quadrature).
//! - [`synthesize_numerical`] for 2Q identity vs manual 1Q decomposition:
//!   for independent qubits, rates should be consistent with the
//!   single-qubit theory.

use approx::assert_abs_diff_eq;

use pecos_lindblad::noise_models::{ad_pd_1q, ad_pd_2q};
use pecos_lindblad::{
    DEFAULT_N_STEPS, Gate, Pauli1, PauliString, synthesize_identity_1q, synthesize_numerical,
};

#[test]
fn fast_identity_matches_simpson_1q() {
    let noise = ad_pd_1q(150.0, 120.0);
    let tau_g = 2.0;
    let gate = Gate::identity(1, noise, tau_g);
    let fast = synthesize_identity_1q(&gate);
    let simpson = synthesize_numerical(&gate, DEFAULT_N_STEPS);
    for p in [Pauli1::X, Pauli1::Y, Pauli1::Z] {
        let key = PauliString::single(p);
        assert_abs_diff_eq!(fast.rate(&key), simpson.rate(&key), epsilon = 1e-14);
    }
}

#[test]
fn identity_2q_rates_agree_with_1q_independent_qubits() {
    // For identity + AD+PD acting independently on two qubits, we expect
    // weight-1 rates {lambda_ix, lambda_iy, lambda_iz, lambda_xi, lambda_yi,
    // lambda_zi} to match the single-qubit predictions (paper line 812):
    //   lambda_{i·x} = lambda_{i·y} = beta_down_r * tau_g / 4
    //   lambda_{i·z} = beta_phi_r * tau_g / 2
    //   (mirror for the l qubit)
    // Weight-2 rates should all be zero (no interaction between qubits).
    let t1_l = 100.0;
    let t2_l = 80.0;
    let t1_r = 150.0;
    let t2_r = 120.0;
    let tau_g = 2.0;
    let noise = ad_pd_2q(t1_l, t1_r, t2_l, t2_r);
    let gate = Gate::identity(2, noise, tau_g);
    let pl = synthesize_numerical(&gate, DEFAULT_N_STEPS);

    let bd_l = 1.0 / t1_l;
    let bd_r = 1.0 / t1_r;
    let bp_l = 1.0 / t2_l - 1.0 / (2.0 * t1_l);
    let bp_r = 1.0 / t2_r - 1.0 / (2.0 * t1_r);

    let rate = |s: &str| pl.rate(&PauliString::from_label(s).unwrap());
    assert_abs_diff_eq!(rate("IX"), bd_r * tau_g / 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(rate("IY"), bd_r * tau_g / 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(rate("IZ"), bp_r * tau_g / 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(rate("XI"), bd_l * tau_g / 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(rate("YI"), bd_l * tau_g / 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(rate("ZI"), bp_l * tau_g / 2.0, epsilon = 1e-12);

    // All weight-2 rates must be zero (no coupling).
    for label in ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"] {
        assert_abs_diff_eq!(rate(label), 0.0, epsilon = 1e-12);
    }
}
