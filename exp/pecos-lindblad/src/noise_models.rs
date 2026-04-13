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

//! Device-parameter convenience constructors for noise [`Lindbladian`]s.
//!
//! Real QEC experiments are typically specified in terms of coherence
//! times `(T_1, T_2)`, not raw Lindblad rates. This module converts
//! between them and builds the tensor-product Lindbladians that the
//! paper-fixture tests would otherwise hand-roll.
//!
//! # T1/T2 convention
//!
//! Standard textbook relation:
//!
//! ```text
//! beta_down = 1 / T_1
//! 1 / T_2   = 1 / (2 T_1) + 1 / T_phi
//! beta_phi  = 1 / T_phi = 1/T_2 - 1/(2 T_1)
//! ```
//!
//! `T_2 >= 2 T_1 / (1 + 2 T_1 / T_phi)`; pure-dephasing-free limit is
//! `T_2 = 2 T_1` with `beta_phi = 0`.

use num_complex::Complex64;

use crate::basis::Pauli1;
use crate::lindbladian::Lindbladian;
use crate::matrix::{self, Matrix};

/// Convert `(T_1, T_2)` to `(beta_down, beta_phi)`. Panics if `T_2 > 2 T_1`
/// (unphysical -- dephasing would be negative).
pub fn t1_t2_to_rates(t1: f64, t2: f64) -> (f64, f64) {
    assert!(t1 > 0.0, "T_1 must be positive");
    assert!(t2 > 0.0, "T_2 must be positive");
    let beta_down = 1.0 / t1;
    let inv_tphi = 1.0 / t2 - 1.0 / (2.0 * t1);
    assert!(
        inv_tphi >= -1e-15,
        "T_2 ({}) > 2 T_1 ({}) violates 1/T_phi = 1/T_2 - 1/(2 T_1) >= 0",
        t2,
        2.0 * t1,
    );
    (beta_down, inv_tphi.max(0.0))
}

/// 1-qubit amplitude-damping + pure-dephasing Lindbladian from `(T_1, T_2)`.
///
/// Collapse operators: `sigma_- with rate 1/T_1`, `Z with rate beta_phi/2`
/// where `beta_phi = 1/T_2 - 1/(2 T_1)`.
pub fn ad_pd_1q(t1: f64, t2: f64) -> Lindbladian {
    let (beta_down, beta_phi) = t1_t2_to_rates(t1, t2);
    let d = 2;
    let hamiltonian = matrix::zeros(d);
    let collapse: Vec<(Matrix, f64)> = vec![
        (matrix::sigma_minus(), beta_down),
        (matrix::pauli_1q(Pauli1::Z), beta_phi / 2.0),
    ];
    Lindbladian::new(d, hamiltonian, collapse)
}

/// 2-qubit amplitude-damping + pure-dephasing, independently parameterised
/// on left (`l`) and right (`r`) qubits.
pub fn ad_pd_2q(t1_l: f64, t1_r: f64, t2_l: f64, t2_r: f64) -> Lindbladian {
    let (bd_l, bp_l) = t1_t2_to_rates(t1_l, t2_l);
    let (bd_r, bp_r) = t1_t2_to_rates(t1_r, t2_r);
    let d = 4;
    let i2 = matrix::identity(2);
    let sm = matrix::sigma_minus();
    let z = matrix::pauli_1q(Pauli1::Z);
    let sm_l = matrix::kron(&sm, &i2, 2, 2);
    let sm_r = matrix::kron(&i2, &sm, 2, 2);
    let z_l = matrix::kron(&z, &i2, 2, 2);
    let z_r = matrix::kron(&i2, &z, 2, 2);
    let collapse: Vec<(Matrix, f64)> = vec![
        (sm_l, bd_l),
        (sm_r, bd_r),
        (z_l, bp_l / 2.0),
        (z_r, bp_r / 2.0),
    ];
    Lindbladian::new(d, matrix::zeros(d), collapse)
}

/// 2-qubit coherent phase noise:
/// `H_delta = (delta_iz/2) IZ + (delta_zi/2) ZI + (delta_zz/2) ZZ`,
/// no collapse operators (use [`crate::synthesize_exact_unitary`]).
pub fn coherent_phase_2q(delta_iz: f64, delta_zi: f64, delta_zz: f64) -> Lindbladian {
    let d = 4;
    let i2 = matrix::identity(2);
    let z = matrix::pauli_1q(Pauli1::Z);
    let iz = matrix::kron(&i2, &z, 2, 2);
    let zi = matrix::kron(&z, &i2, 2, 2);
    let zz = matrix::kron(&z, &z, 2, 2);
    let half = Complex64::new(0.5, 0.0);
    let h_delta = matrix::add(
        &matrix::add(
            &matrix::scale(&iz, Complex64::new(delta_iz, 0.0) * half),
            &matrix::scale(&zi, Complex64::new(delta_zi, 0.0) * half),
        ),
        &matrix::scale(&zz, Complex64::new(delta_zz, 0.0) * half),
    );
    Lindbladian::new(d, h_delta, Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t1_t2_round_trip() {
        let t1 = 100e-6;
        let t2 = 80e-6; // < 2 T_1 so physical
        let (bd, bp) = t1_t2_to_rates(t1, t2);
        assert!((bd - 1.0 / t1).abs() < 1e-15);
        assert!((bp - (1.0 / t2 - 1.0 / (2.0 * t1))).abs() < 1e-15);
    }

    #[test]
    fn t2_equals_2_t1_gives_zero_dephasing() {
        // Dephasing-free limit.
        let t1 = 100e-6;
        let (bd, bp) = t1_t2_to_rates(t1, 2.0 * t1);
        assert!((bd - 1.0 / t1).abs() < 1e-15);
        assert!(bp < 1e-15, "bp should be ~0, got {}", bp);
    }

    #[test]
    #[should_panic(expected = "T_2")]
    fn unphysical_t2_panics() {
        let _ = t1_t2_to_rates(100e-6, 300e-6); // T_2 > 2 T_1
    }
}
