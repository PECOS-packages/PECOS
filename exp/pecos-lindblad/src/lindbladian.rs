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

//! Lindbladian type: Hermitian Hamiltonian plus rate-weighted collapse operators.

use num_complex::Complex64;

use crate::matrix::{self, Matrix};

/// Time-independent Lindbladian of form
/// `drho/dt = -i[H, rho] + sum_j gamma_j * D[c_j] rho`
/// where `D[c] rho = c rho c^dag - 1/2 {c^dag c, rho}`.
#[derive(Clone, Debug)]
pub struct Lindbladian {
    pub d: usize,
    pub hamiltonian: Matrix,
    pub collapse: Vec<(Matrix, f64)>,
}

impl Lindbladian {
    pub fn new(d: usize, hamiltonian: Matrix, collapse: Vec<(Matrix, f64)>) -> Self {
        assert_eq!(hamiltonian.len(), d * d, "hamiltonian wrong shape");
        assert!(
            matrix::is_hermitian(&hamiltonian, d, 1e-10),
            "Lindbladian Hamiltonian must be Hermitian",
        );
        for (c, gamma) in &collapse {
            assert_eq!(c.len(), d * d, "collapse op wrong shape");
            assert!(*gamma >= 0.0, "collapse rate must be non-negative, got {}", gamma);
        }
        Self { d, hamiltonian, collapse }
    }

    /// Zero Hamiltonian with no collapse ops (no-op).
    pub fn zero(d: usize) -> Self {
        Self { d, hamiltonian: matrix::zeros(d), collapse: Vec::new() }
    }

    /// Apply `L` to a matrix `rho`. Returns `L(rho)`.
    pub fn apply(&self, rho: &Matrix) -> Matrix {
        let d = self.d;
        let neg_i = Complex64::new(0.0, -1.0);
        let mut out = matrix::scale(&matrix::commutator(&self.hamiltonian, rho, d), neg_i);
        for (c, gamma) in &self.collapse {
            let cdag = matrix::dag(c, d);
            let c_rho_cdag = matrix::matmul(&matrix::matmul(c, rho, d), &cdag, d);
            let cdag_c = matrix::matmul(&cdag, c, d);
            let acom = matrix::anticommutator(&cdag_c, rho, d);
            let diss = matrix::sub(&c_rho_cdag, &matrix::scale(&acom, Complex64::new(0.5, 0.0)));
            out = matrix::add(&out, &matrix::scale(&diss, Complex64::new(*gamma, 0.0)));
        }
        out
    }
}
