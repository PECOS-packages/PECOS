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

//! Input-validation tests. Bad inputs must produce immediate panics at
//! construction time rather than silently returning wrong results.

use num_complex::Complex64;

use pecos_lindblad::matrix::{self, Matrix};
use pecos_lindblad::{Gate, Lindbladian};

#[test]
#[should_panic(expected = "Hermitian")]
fn non_hermitian_hamiltonian_in_lindbladian_panics() {
    let d = 2;
    let mut h: Matrix = vec![Complex64::new(0.0, 0.0); d * d];
    // Asymmetric imaginary entry makes this non-Hermitian.
    h[1] = Complex64::new(1.0, 0.0);
    h[2] = Complex64::new(2.0, 0.0); // Should be 1.0 for Hermitian.
    let _ = Lindbladian::new(d, h, Vec::new());
}

#[test]
#[should_panic(expected = "Hermitian")]
fn non_hermitian_ideal_hamiltonian_in_gate_panics() {
    let d = 2;
    let mut h: Matrix = vec![Complex64::new(0.0, 0.0); d * d];
    h[1] = Complex64::new(0.0, 1.0); // pure imaginary, not Hermitian paired with h[2]=0
    let noise = Lindbladian::zero(d);
    let _ = Gate::from_hamiltonian("bad", 1, h, noise, 1.0);
}

#[test]
#[should_panic(expected = "non-negative")]
fn negative_collapse_rate_panics() {
    let d = 2;
    let _ = Lindbladian::new(d, matrix::zeros(d), vec![(matrix::sigma_minus(), -1e-3)]);
}

#[test]
#[should_panic(expected = "tau_g")]
fn negative_tau_g_panics() {
    let h = matrix::zeros(2);
    let noise = Lindbladian::zero(2);
    let _ = Gate::from_hamiltonian("bad", 1, h, noise, -1.0);
}

#[test]
#[should_panic(expected = "wrong shape")]
fn wrong_matrix_size_panics() {
    // 3-element matrix is neither 1x1 nor any d*d.
    let h: Matrix = vec![Complex64::new(0.0, 0.0); 3];
    let _ = Lindbladian::new(2, h, Vec::new());
}

#[test]
fn hermitian_traceless_is_accepted() {
    // Pauli X is Hermitian, traceless.
    let x = matrix::pauli_1q(pecos_lindblad::Pauli1::X);
    let _ = Lindbladian::new(2, x, Vec::new());
}

#[test]
fn hermitian_with_real_diagonal_is_accepted() {
    // diag(1, -1, 0.5, -0.5) is Hermitian.
    let d = 4;
    let mut h: Matrix = vec![Complex64::new(0.0, 0.0); d * d];
    h[0] = Complex64::new(1.0, 0.0);
    h[d + 1] = Complex64::new(-1.0, 0.0);
    h[2 * d + 2] = Complex64::new(0.5, 0.0);
    h[3 * d + 3] = Complex64::new(-0.5, 0.0);
    let _ = Lindbladian::new(d, h, Vec::new());
}
