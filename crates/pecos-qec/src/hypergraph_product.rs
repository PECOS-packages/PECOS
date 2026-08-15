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

//! Hypergraph-product CSS codes built from two classical parity-check matrices.
//!
//! The construction follows the hypergraph product of Tillich and Zemor
//! (arXiv:0903.0566): from classical checks `H1` (`r1 x n1`) and `H2`
//! (`r2 x n2`),
//!
//! ```text
//! Hx = [ H1 (x) I_n2 | I_r1 (x) H2^T ]
//! Hz = [ I_n1 (x) H2 | H1^T (x) I_r2 ]
//! ```
//!
//! on `n = n1*n2 + r1*r2` data qubits. CSS orthogonality holds identically;
//! the constructor asserts it anyway to fail fast on implementation error.

use crate::memory_circuit::discover_css_logical_operators;
use crate::parity_check_matrix::ParityCheckMatrix;
use pecos_quantum::F2Matrix;
use thiserror::Error;

/// Errors constructing a [`HypergraphProductCode`].
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum HypergraphProductError {
    /// A classical input has no rows or no columns.
    #[error("classical input {which} must be nonempty")]
    EmptyInput {
        /// Which input was empty.
        which: &'static str,
    },
}

/// A hypergraph-product CSS code with discovered logical bases.
#[derive(Clone, Debug)]
pub struct HypergraphProductCode {
    hx: ParityCheckMatrix,
    hz: ParityCheckMatrix,
    logical_x: ParityCheckMatrix,
    logical_z: ParityCheckMatrix,
}

impl HypergraphProductCode {
    /// Builds the hypergraph product of two classical parity-check matrices.
    ///
    /// # Errors
    ///
    /// Returns an error if either classical input is empty.
    ///
    /// # Panics
    ///
    /// Panics only if the internally generated rectangular binary matrices are
    /// rejected by [`ParityCheckMatrix`] or violate CSS orthogonality, either
    /// of which would break this module's construction invariant.
    pub fn new(
        h1: &ParityCheckMatrix,
        h2: &ParityCheckMatrix,
    ) -> Result<Self, HypergraphProductError> {
        let (r1, n1) = (h1.num_checks(), h1.num_qubits());
        let (r2, n2) = (h2.num_checks(), h2.num_qubits());
        if r1 == 0 || n1 == 0 {
            return Err(HypergraphProductError::EmptyInput { which: "H1" });
        }
        if r2 == 0 || n2 == 0 {
            return Err(HypergraphProductError::EmptyInput { which: "H2" });
        }

        let h1m = h1.matrix();
        let h2m = h2.matrix();
        let hx_left = h1m.kronecker(&F2Matrix::identity(n2));
        let hx_right = F2Matrix::identity(r1).kronecker(&h2m.transpose());
        let hz_left = F2Matrix::identity(n1).kronecker(h2m);
        let hz_right = h1m.transpose().kronecker(&F2Matrix::identity(r2));

        let num_qubits = n1 * n2 + r1 * r2;
        let mut hx = F2Matrix::zeros(r1 * n2, num_qubits);
        let mut hz = F2Matrix::zeros(n1 * r2, num_qubits);
        for row in 0..r1 * n2 {
            for col in 0..n1 * n2 {
                hx.set(row, col, hx_left.get(row, col));
            }
            for col in 0..r1 * r2 {
                hx.set(row, n1 * n2 + col, hx_right.get(row, col));
            }
        }
        for row in 0..n1 * r2 {
            for col in 0..n1 * n2 {
                hz.set(row, col, hz_left.get(row, col));
            }
            for col in 0..r1 * r2 {
                hz.set(row, n1 * n2 + col, hz_right.get(row, col));
            }
        }
        assert_eq!(
            hx.mul(&hz.transpose()),
            F2Matrix::zeros(r1 * n2, n1 * r2),
            "hypergraph-product orthogonality is guaranteed by construction",
        );

        let hx = ParityCheckMatrix::from_dense(hx.rows())
            .expect("a nonempty rectangular binary matrix was generated");
        let hz = ParityCheckMatrix::from_dense(hz.rows())
            .expect("a nonempty rectangular binary matrix was generated");
        let (logical_x, logical_z) = discover_css_logical_operators(&hx, &hz);

        Ok(Self {
            hx,
            hz,
            logical_x,
            logical_z,
        })
    }

    /// Number of data qubits, `n1*n2 + r1*r2`.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.hx.num_qubits()
    }

    /// Number of logical qubits, `n - rank(Hx) - rank(Hz)`.
    #[must_use]
    pub fn num_logical_qubits(&self) -> usize {
        self.num_qubits() - self.hx.rank() - self.hz.rank()
    }

    /// The X-type check matrix.
    #[must_use]
    pub fn hx(&self) -> &ParityCheckMatrix {
        &self.hx
    }

    /// The Z-type check matrix.
    #[must_use]
    pub fn hz(&self) -> &ParityCheckMatrix {
        &self.hz
    }

    /// A basis of X-type logical representatives.
    #[must_use]
    pub fn logical_x(&self) -> &ParityCheckMatrix {
        &self.logical_x
    }

    /// A basis of Z-type logical representatives.
    #[must_use]
    pub fn logical_z(&self) -> &ParityCheckMatrix {
        &self.logical_z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BoundedEnumerationDistance, bounded_enumeration_code_distance,
        connected_cluster_code_distance,
    };

    fn repetition3() -> ParityCheckMatrix {
        ParityCheckMatrix::from_dense(vec![vec![1, 1, 0], vec![0, 1, 1]]).unwrap()
    }

    fn hamming() -> ParityCheckMatrix {
        ParityCheckMatrix::from_dense(vec![
            vec![1, 0, 1, 0, 1, 0, 1],
            vec![0, 1, 1, 0, 0, 1, 1],
            vec![0, 0, 0, 1, 1, 1, 1],
        ])
        .unwrap()
    }

    #[test]
    fn repetition_square_is_the_thirteen_qubit_surface_type_code() {
        let code = HypergraphProductCode::new(&repetition3(), &repetition3()).unwrap();
        // n = 3*3 + 2*2 = 13; k = k1*k2 + k1T*k2T = 1*1 + 0*0 = 1.
        assert_eq!(code.num_qubits(), 13);
        assert_eq!(code.num_logical_qubits(), 1);

        let cc = connected_cluster_code_distance(code.hx(), code.logical_x(), 13)
            .expect("distance within budget");
        assert_eq!(cc.distance, 3);
        match bounded_enumeration_code_distance(code.hx(), code.logical_x(), 13)
            .expect("kernel is nonempty")
        {
            BoundedEnumerationDistance::CertifiedByBounds { distance, .. } => {
                assert_eq!(distance, 3);
            }
            BoundedEnumerationDistance::LevelLimitReached { .. } => {
                panic!("thirteen-qubit search must certify")
            }
        }
    }

    #[test]
    fn hamming_times_repetition_has_expected_parameters_and_distance() {
        let code = HypergraphProductCode::new(&hamming(), &repetition3()).unwrap();
        // n = 7*3 + 3*2 = 27; k = k1*k2 + k1T*k2T = 4*1 + 0*0 = 4
        // (both transpose codes are trivial: the inputs have full row rank).
        assert_eq!(code.num_qubits(), 27);
        assert_eq!(code.num_logical_qubits(), 4);

        // Expected distance: min(d1, d2) = min(3, 3) = 3 for full-rank inputs;
        // measured rather than assumed.
        let z_side = connected_cluster_code_distance(code.hx(), code.logical_x(), 27)
            .expect("distance within budget");
        let x_side = connected_cluster_code_distance(code.hz(), code.logical_z(), 27)
            .expect("distance within budget");
        assert_eq!(z_side.distance.min(x_side.distance), 3);
    }

    #[test]
    fn empty_inputs_are_rejected() {
        let ok = repetition3();
        let empty = ParityCheckMatrix::zeros(0, 3);
        assert_eq!(
            HypergraphProductCode::new(&empty, &ok).unwrap_err(),
            HypergraphProductError::EmptyInput { which: "H1" }
        );
        assert_eq!(
            HypergraphProductCode::new(&ok, &empty).unwrap_err(),
            HypergraphProductError::EmptyInput { which: "H2" }
        );
    }

    #[test]
    fn construction_is_deterministic() {
        let a = HypergraphProductCode::new(&hamming(), &repetition3()).unwrap();
        let b = HypergraphProductCode::new(&hamming(), &repetition3()).unwrap();
        assert_eq!(a.hx().rows(), b.hx().rows());
        assert_eq!(a.logical_z().rows(), b.logical_z().rows());
    }
}
