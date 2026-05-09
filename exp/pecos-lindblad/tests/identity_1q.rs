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

//! Golden-fixture test for 1-qubit identity gate under amplitude damping
//! plus pure dephasing (arXiv:2502.03462 line 812, exact non-perturbative):
//!
//!   lambda_x = lambda_y = (beta_down * tau_g) / 4
//!   lambda_z = (beta_phi * tau_g) / 2

use approx::assert_abs_diff_eq;

use pecos_lindblad::matrix::{self, Matrix};
use pecos_lindblad::{Gate, Lindbladian, Pauli1, PauliString, synthesize_identity_1q};

fn amplitude_damping_plus_dephasing(beta_down: f64, beta_phi: f64) -> Lindbladian {
    let d = 2;
    let hamiltonian = matrix::zeros(d);
    let collapse: Vec<(Matrix, f64)> = vec![
        (matrix::sigma_minus(), beta_down),
        (matrix::pauli_1q(Pauli1::Z), beta_phi / 2.0),
    ];
    Lindbladian::new(d, hamiltonian, collapse)
}

#[test]
fn identity_ad_plus_pd_matches_paper() {
    let beta_down = 2e-4;
    let beta_phi = 5e-4;
    let tau_g = 50.0;
    let noise = amplitude_damping_plus_dephasing(beta_down, beta_phi);
    let gate = Gate::identity(1, noise, tau_g);

    let pl = synthesize_identity_1q(&gate);

    let expected_x = beta_down * tau_g / 4.0;
    let expected_y = beta_down * tau_g / 4.0;
    let expected_z = beta_phi * tau_g / 2.0;

    assert_abs_diff_eq!(
        pl.rate(&PauliString::single(Pauli1::X)),
        expected_x,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        pl.rate(&PauliString::single(Pauli1::Y)),
        expected_y,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        pl.rate(&PauliString::single(Pauli1::Z)),
        expected_z,
        epsilon = 1e-12
    );
}

#[test]
fn identity_ad_only() {
    let beta_down = 1e-3;
    let tau_g = 100.0;
    let noise = amplitude_damping_plus_dephasing(beta_down, 0.0);
    let gate = Gate::identity(1, noise, tau_g);

    let pl = synthesize_identity_1q(&gate);

    let expected_xy = beta_down * tau_g / 4.0;
    assert_abs_diff_eq!(
        pl.rate(&PauliString::single(Pauli1::X)),
        expected_xy,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        pl.rate(&PauliString::single(Pauli1::Y)),
        expected_xy,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        pl.rate(&PauliString::single(Pauli1::Z)),
        0.0,
        epsilon = 1e-12
    );
}

#[test]
fn identity_pd_only() {
    let beta_phi = 7e-4;
    let tau_g = 40.0;
    let noise = amplitude_damping_plus_dephasing(0.0, beta_phi);
    let gate = Gate::identity(1, noise, tau_g);

    let pl = synthesize_identity_1q(&gate);

    let expected_z = beta_phi * tau_g / 2.0;
    assert_abs_diff_eq!(
        pl.rate(&PauliString::single(Pauli1::X)),
        0.0,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        pl.rate(&PauliString::single(Pauli1::Y)),
        0.0,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        pl.rate(&PauliString::single(Pauli1::Z)),
        expected_z,
        epsilon = 1e-12
    );
}

#[test]
fn identity_zero_noise_gives_zero_rates() {
    let noise = amplitude_damping_plus_dephasing(0.0, 0.0);
    let gate = Gate::identity(1, noise, 1.0);

    let pl = synthesize_identity_1q(&gate);

    for p in [Pauli1::X, Pauli1::Y, Pauli1::Z] {
        assert_abs_diff_eq!(pl.rate(&PauliString::single(p)), 0.0, epsilon = 1e-14);
    }
}
