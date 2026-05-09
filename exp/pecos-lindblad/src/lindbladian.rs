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
            assert!(
                *gamma >= 0.0,
                "collapse rate must be non-negative, got {}",
                gamma
            );
        }
        Self {
            d,
            hamiltonian,
            collapse,
        }
    }

    /// Zero Hamiltonian with no collapse ops (no-op).
    pub fn zero(d: usize) -> Self {
        Self {
            d,
            hamiltonian: matrix::zeros(d),
            collapse: Vec::new(),
        }
    }

    /// Build the `d^2 x d^2` Liouville-superoperator matrix representation
    /// of `L`. Column-stacking convention: `vec(L(rho)) = L_super * vec(rho)`.
    ///
    /// `L(rho) = -i[H, rho] + sum_j gamma_j * ( c_j rho c_j^dag
    ///                                         - 1/2 {c_j^dag c_j, rho} )`
    /// translates to:
    ///
    /// `L_super = -i (I ⊗ H - H^T ⊗ I)
    ///            + sum_j gamma_j * ( conj(c_j) ⊗ c_j
    ///                               - 1/2 I ⊗ c_j^dag c_j
    ///                               - 1/2 (c_j^dag c_j)^T ⊗ I )`
    ///
    /// (Note `(c^dag)^T = conj(c)`.)
    pub fn superoperator(&self) -> Matrix {
        let d = self.d;
        let d2 = d * d;
        let id = matrix::identity(d);
        let neg_i = Complex64::new(0.0, -1.0);

        // Hamiltonian part: -i (I ⊗ H - H^T ⊗ I).
        let h_t = matrix::transpose(&self.hamiltonian, d);
        let coh = matrix::sub(
            &matrix::kron(&id, &self.hamiltonian, d, d),
            &matrix::kron(&h_t, &id, d, d),
        );
        let mut l_super = matrix::scale(&coh, neg_i);

        for (c, gamma) in &self.collapse {
            let c_bar = matrix::conj(c);
            let c_dag = matrix::dag(c, d);
            let cdag_c = matrix::matmul(&c_dag, c, d);
            let cdag_c_t = matrix::transpose(&cdag_c, d);

            // gamma * ( c_bar ⊗ c - 1/2 I ⊗ c^dag c - 1/2 (c^dag c)^T ⊗ I )
            let term_a = matrix::kron(&c_bar, c, d, d);
            let term_b = matrix::kron(&id, &cdag_c, d, d);
            let term_c = matrix::kron(&cdag_c_t, &id, d, d);
            let inner = matrix::sub(
                &term_a,
                &matrix::add(
                    &matrix::scale(&term_b, Complex64::new(0.5, 0.0)),
                    &matrix::scale(&term_c, Complex64::new(0.5, 0.0)),
                ),
            );
            let scaled = matrix::scale(&inner, Complex64::new(*gamma, 0.0));
            l_super = matrix::add(&l_super, &scaled);
        }

        assert_eq!(l_super.len(), d2 * d2);
        l_super
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
