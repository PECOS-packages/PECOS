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

//! Smoke tests for `matrix::expm` (general Taylor + scaling + squaring).

use num_complex::Complex64;

use pecos_lindblad::Pauli1;
use pecos_lindblad::matrix::{self, Matrix};

fn assert_close(a: &Matrix, b: &Matrix, tol: f64) {
    assert_eq!(a.len(), b.len(), "matrix size mismatch");
    for i in 0..a.len() {
        let delta = (a[i] - b[i]).norm();
        assert!(
            delta < tol,
            "entry {}: |{:?} - {:?}| = {} > {}",
            i,
            a[i],
            b[i],
            delta,
            tol
        );
    }
}

#[test]
fn expm_of_zero_is_identity() {
    for d in [2, 3, 4, 8] {
        let z = matrix::zeros(d);
        let result = matrix::expm(&z, d);
        assert_close(&result, &matrix::identity(d), 1e-14);
    }
}

#[test]
fn expm_of_diagonal_is_elementwise_exp() {
    let d = 4;
    let mut m = matrix::zeros(d);
    m[0] = Complex64::new(0.5, 0.0);
    m[d + 1] = Complex64::new(-0.3, 0.0);
    m[2 * d + 2] = Complex64::new(0.0, 1.2);
    m[3 * d + 3] = Complex64::new(-0.1, -0.4);
    let result = matrix::expm(&m, d);
    let mut expected = matrix::zeros(d);
    for i in 0..d {
        expected[i * d + i] = m[i * d + i].exp();
    }
    assert_close(&result, &expected, 1e-12);
}

#[test]
fn expm_agrees_with_1q_bloch_on_traceless() {
    // H = 1.3 * X + 0.7 * Y - 0.4 * Z (traceless Hermitian 2x2).
    // Compare expm(-i * H * t) against exp_minus_i_h_t_1q_traceless.
    let d = 2;
    let x = matrix::pauli_1q(Pauli1::X);
    let y = matrix::pauli_1q(Pauli1::Y);
    let z = matrix::pauli_1q(Pauli1::Z);
    let h = matrix::add(
        &matrix::add(
            &matrix::scale(&x, Complex64::new(1.3, 0.0)),
            &matrix::scale(&y, Complex64::new(0.7, 0.0)),
        ),
        &matrix::scale(&z, Complex64::new(-0.4, 0.0)),
    );
    for t in [0.1, 0.7, 1.5] {
        let bloch = matrix::exp_minus_i_h_t_1q_traceless(&h, t);
        let via_expm = matrix::expm(&matrix::scale(&h, Complex64::new(0.0, -t)), d);
        assert_close(&bloch, &via_expm, 1e-11);
    }
}

#[test]
fn expm_preserves_unitarity_for_hermitian_input() {
    // For any Hermitian H, U = exp(-i H t) should satisfy U U^dag = I.
    let d = 4;
    let i2 = matrix::identity(2);
    let x = matrix::pauli_1q(Pauli1::X);
    let z = matrix::pauli_1q(Pauli1::Z);
    let ix = matrix::kron(&i2, &x, 2, 2);
    let zx = matrix::kron(&z, &x, 2, 2);
    let h = matrix::sub(&ix, &zx); // CX-style Hermitian
    let t = 0.47;
    let u = matrix::expm(&matrix::scale(&h, Complex64::new(0.0, -t)), d);
    let u_udag = matrix::matmul(&u, &matrix::dag(&u, d), d);
    assert_close(&u_udag, &matrix::identity(d), 1e-11);
}

#[test]
fn expm_fallback_used_for_non_structured_4x4() {
    // Construct a 4x4 Hermitian that is NOT diagonal and NOT 2x2-block-diag.
    // H = IX + XI + YZ (mixes all quadrants).
    let d = 4;
    let i2 = matrix::identity(2);
    let x = matrix::pauli_1q(Pauli1::X);
    let y = matrix::pauli_1q(Pauli1::Y);
    let z = matrix::pauli_1q(Pauli1::Z);
    let ix = matrix::kron(&i2, &x, 2, 2);
    let xi = matrix::kron(&x, &i2, 2, 2);
    let yz = matrix::kron(&y, &z, 2, 2);
    let h = matrix::add(&matrix::add(&ix, &xi), &yz);

    assert!(!matrix::is_diagonal(&h, d, 1e-14));
    assert!(!matrix::is_2x2_block_diagonal(&h, 1e-14));

    // Should not panic (falls through to general expm).
    let u = matrix::exp_minus_i_h_t(&h, d, 0.3);
    // Unitarity check.
    let u_udag = matrix::matmul(&u, &matrix::dag(&u, d), d);
    assert_close(&u_udag, &matrix::identity(d), 1e-11);
}
