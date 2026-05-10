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

//! Parity tests: 2-qubit gates under coherent phase noise from
//! arXiv:2502.03462 SubApp:2QPhNoise (lines 962-1001).
//!
//! Noise Hamiltonian:
//!   H_delta = (delta_iz/2) IZ + (delta_zi/2) ZI + (delta_zz/2) ZZ
//!
//! Cases tested:
//! - (i)   Identity (H_g = 0): all three delta components commute with H_g,
//!   so rates are quadratic-in-delta and decoupled (eq. 981).
//! - (iii) CX_theta: phase noise doesn't commute with X_target, producing
//!   mixing between delta_iz and delta_zz into lambda_iy, lambda_zy,
//!   lambda_iz, lambda_zz (eqs. 986-990).
//!
//! Synthesis path: `synthesize_exact_unitary` (coherent noise, no c_ops).

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use pecos_lindblad::matrix::{self, Matrix};
use pecos_lindblad::{Gate, Lindbladian, Pauli1, PauliString, synthesize_exact_unitary};

fn phase_noise_2q(delta_iz: f64, delta_zi: f64, delta_zz: f64) -> Lindbladian {
    let d = 4;
    let i2 = matrix::identity(2);
    let z = matrix::pauli_1q(Pauli1::Z);
    let iz = matrix::kron(&i2, &z, 2, 2);
    let zi = matrix::kron(&z, &i2, 2, 2);
    let zz = matrix::kron(&z, &z, 2, 2);
    let half = Complex64::new(0.5, 0.0);
    let h_delta: Matrix = matrix::add(
        &matrix::add(
            &matrix::scale(&iz, Complex64::new(delta_iz, 0.0) * half),
            &matrix::scale(&zi, Complex64::new(delta_zi, 0.0) * half),
        ),
        &matrix::scale(&zz, Complex64::new(delta_zz, 0.0) * half),
    );
    Lindbladian::new(d, h_delta, Vec::new())
}

#[test]
fn identity_2q_phase_noise_commuting_case() {
    // Paper eq. 981: lambda_iz = (tau_g * delta_iz)^2 / 4, etc.
    // (obtained by setting theta_cz = omega_cz * tau_g and dividing).
    // Use weak noise so O(g^4) corrections stay below the 1e-10 tolerance.
    // At g = delta * tau ~ 1e-5 the next-order correction ~g^4/24 ~ 4e-22.
    let tau_g = 10.0;
    let delta_iz = 1e-6;
    let delta_zi = 2e-6;
    let delta_zz = 5e-7;
    let noise = phase_noise_2q(delta_iz, delta_zi, delta_zz);
    let gate = Gate::identity(2, noise, tau_g);
    let pl = synthesize_exact_unitary(&gate);

    let rate = |s: &str| pl.rate(&PauliString::from_label(s).unwrap());
    assert_abs_diff_eq!(
        rate("IZ"),
        (delta_iz * tau_g).powi(2) / 4.0,
        epsilon = 1e-10
    );
    assert_abs_diff_eq!(
        rate("ZI"),
        (delta_zi * tau_g).powi(2) / 4.0,
        epsilon = 1e-10
    );
    assert_abs_diff_eq!(
        rate("ZZ"),
        (delta_zz * tau_g).powi(2) / 4.0,
        epsilon = 1e-10
    );

    // All others should be zero (phase noise commutes, so only Z-basis
    // rates appear).
    for label in [
        "IX", "IY", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZX", "ZY",
    ] {
        assert_abs_diff_eq!(rate(label), 0.0, epsilon = 1e-10);
    }
}

#[test]
fn cx_theta_phase_noise_mixing_case() {
    // Paper eqs. 986-990: CX_theta with H_delta = IZ, ZI, ZZ phase noise.
    //   lambda_zi = theta^2 / 4 * (delta_zi / omega)^2
    //   lambda_iy = lambda_zy = sin^4(theta) / 16 * ((delta_iz - delta_zz) / omega)^2
    //   lambda_iz = [2 theta (delta_iz + delta_zz)
    //               + sin(2 theta)(delta_iz - delta_zz)]^2 / (64 omega^2)
    //   lambda_zz = [2 theta (delta_iz + delta_zz)
    //               + sin(2 theta)(delta_zz - delta_iz)]^2 / (64 omega^2)
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_4;
    let delta_iz = 1e-3;
    let delta_zi = 2e-3;
    let delta_zz = 5e-4;

    let noise = phase_noise_2q(delta_iz, delta_zi, delta_zz);
    let gate = Gate::cx_theta(omega, theta, noise);
    let pl = synthesize_exact_unitary(&gate);

    let rate = |s: &str| pl.rate(&PauliString::from_label(s).unwrap());

    let s2t = (2.0 * theta).sin();
    let sin4 = theta.sin().powi(4);
    let sum = delta_iz + delta_zz;
    let diff_iz_zz = delta_iz - delta_zz;

    let expected_zi = theta.powi(2) / 4.0 * (delta_zi / omega).powi(2);
    let expected_iy_zy = sin4 / 16.0 * (diff_iz_zz / omega).powi(2);
    let expected_iz = (2.0 * theta * sum + s2t * diff_iz_zz).powi(2) / (64.0 * omega.powi(2));
    let expected_zz = (2.0 * theta * sum + s2t * (-diff_iz_zz)).powi(2) / (64.0 * omega.powi(2));

    assert_abs_diff_eq!(rate("ZI"), expected_zi, epsilon = 1e-9);
    assert_abs_diff_eq!(rate("IY"), expected_iy_zy, epsilon = 1e-9);
    assert_abs_diff_eq!(rate("ZY"), expected_iy_zy, epsilon = 1e-9);
    assert_abs_diff_eq!(rate("IZ"), expected_iz, epsilon = 1e-9);
    assert_abs_diff_eq!(rate("ZZ"), expected_zz, epsilon = 1e-9);

    // Other 10 rates should be zero to leading order.
    for label in ["IX", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZX"] {
        assert_abs_diff_eq!(rate(label), 0.0, epsilon = 1e-9);
    }
}

#[test]
fn cz_theta_phase_noise_commuting_case() {
    // Paper eq. 981 case (ii): CZ_theta with phase noise.
    // Since H_g = (omega_cz/2)(II-IZ-ZI+ZZ) is diagonal and phase noise is
    // also diagonal, the Hamiltonians commute. Leading-order rates:
    //   lambda_iz = theta_cz^2 / 4 * (delta_iz / omega_cz)^2
    //   lambda_zi = same with delta_zi
    //   lambda_zz = same with delta_zz
    let omega_cz = 1.0;
    let theta = std::f64::consts::FRAC_PI_3;
    let delta_iz = 1e-6;
    let delta_zi = 2e-6;
    let delta_zz = 5e-7;
    let noise = phase_noise_2q(delta_iz, delta_zi, delta_zz);
    let gate = Gate::cz_theta(omega_cz, theta, noise);
    let pl = synthesize_exact_unitary(&gate);

    let rate = |s: &str| pl.rate(&PauliString::from_label(s).unwrap());
    let factor = theta.powi(2) / 4.0 / omega_cz.powi(2);
    assert_abs_diff_eq!(rate("IZ"), factor * delta_iz.powi(2), epsilon = 1e-14);
    assert_abs_diff_eq!(rate("ZI"), factor * delta_zi.powi(2), epsilon = 1e-14);
    assert_abs_diff_eq!(rate("ZZ"), factor * delta_zz.powi(2), epsilon = 1e-14);

    // All non-Z-basis rates should be zero (commuting case, no mixing).
    for label in [
        "IX", "IY", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZX", "ZY",
    ] {
        assert_abs_diff_eq!(rate(label), 0.0, epsilon = 1e-14);
    }
}

#[test]
fn cx_theta_phase_noise_pi_over_2() {
    // theta = pi/2 => sin(2 theta) = 0; the mixing term vanishes and
    // lambda_iz = lambda_zz.
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_2;
    let delta_iz = 1e-3;
    let delta_zz = 2e-3;
    let noise = phase_noise_2q(delta_iz, 0.0, delta_zz);
    let gate = Gate::cx_theta(omega, theta, noise);
    let pl = synthesize_exact_unitary(&gate);

    let rate = |s: &str| pl.rate(&PauliString::from_label(s).unwrap());
    let expected = (2.0 * theta * (delta_iz + delta_zz)).powi(2) / (64.0 * omega.powi(2));
    assert_abs_diff_eq!(rate("IZ"), expected, epsilon = 1e-9);
    assert_abs_diff_eq!(rate("ZZ"), expected, epsilon = 1e-9);
}
