// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Exact one-qubit matrices and gate words.
//!
//! Gate words are stored in **execution order**: index zero is applied first.
//! Consequently, the matrix of `[A, B]` is `B * A`. This is the reverse of the
//! operator-string convention used in the Matsumoto--Amano literature.

use std::ops::Mul;

use num_traits::{One, Zero};

use crate::{DOmega, ZOmega};

/// A one-qubit gate token.
///
/// The declaration order is the fixed canonical alphabet order used by exact
/// synthesis. `S` is PECOS's `SZ` gate.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Gate {
    /// Identity.
    I,
    /// Pauli X.
    X,
    /// Pauli Y.
    Y,
    /// Pauli Z.
    Z,
    /// Hadamard.
    H,
    /// Phase gate `diag(1, i)` (PECOS `SZ`).
    S,
    /// Adjoint phase gate `diag(1, -i)` (PECOS `SZdg`).
    Sdg,
    /// Pi/8 gate `diag(1, omega)`.
    T,
    /// Adjoint pi/8 gate `diag(1, omega^-1)`.
    Tdg,
}

/// An exact two-by-two matrix over `D[omega]`.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Matrix {
    entries: [[DOmega; 2]; 2],
}

impl Matrix {
    /// Constructs a matrix from row-major entries.
    #[must_use]
    pub const fn new(entries: [[DOmega; 2]; 2]) -> Self {
        Self { entries }
    }

    /// Returns the row-major entries.
    #[must_use]
    pub const fn entries(&self) -> &[[DOmega; 2]; 2] {
        &self.entries
    }

    /// Returns the identity matrix.
    #[must_use]
    pub fn identity() -> Self {
        Self::new([
            [DOmega::one(), DOmega::zero()],
            [DOmega::zero(), DOmega::one()],
        ])
    }

    /// Returns the exact matrix of a gate token.
    #[must_use]
    pub fn from_gate(gate: Gate) -> Self {
        let zero = DOmega::zero();
        let one = DOmega::one();
        let minus_one = -&one;
        let i = DOmega::from(ZOmega::i());
        let minus_i = -&i;
        let omega = DOmega::from(ZOmega::omega());

        match gate {
            Gate::I => Self::identity(),
            Gate::X => Self::new([[zero.clone(), one.clone()], [one, zero]]),
            Gate::Y => Self::new([[zero.clone(), minus_i], [i, zero]]),
            Gate::Z => Self::new([[one, zero.clone()], [zero, minus_one]]),
            Gate::H => {
                let inverse_sqrt2 = DOmega::new(ZOmega::one(), 1);
                Self::new([
                    [inverse_sqrt2.clone(), inverse_sqrt2.clone()],
                    [inverse_sqrt2.clone(), -inverse_sqrt2],
                ])
            }
            Gate::S => Self::new([[one, zero.clone()], [zero, i]]),
            Gate::Sdg => Self::new([[one, zero.clone()], [zero, minus_i]]),
            Gate::T => Self::new([[one, zero.clone()], [zero, omega]]),
            Gate::Tdg => Self::new([
                [one, zero.clone()],
                [zero, DOmega::from(ZOmega::omega().conjugate())],
            ]),
        }
    }

    /// Reconstructs the exact matrix of a word in execution order.
    #[must_use]
    pub fn from_word(word: &[Gate]) -> Self {
        word.iter().fold(Self::identity(), |matrix, gate| {
            &Self::from_gate(*gate) * &matrix
        })
    }

    /// Returns the conjugate transpose.
    #[must_use]
    pub fn adjoint(&self) -> Self {
        Self::new([
            [
                self.entries[0][0].conjugate(),
                self.entries[1][0].conjugate(),
            ],
            [
                self.entries[0][1].conjugate(),
                self.entries[1][1].conjugate(),
            ],
        ])
    }

    /// Tests exact unitarity by comparing `self^dagger * self` with identity.
    #[must_use]
    pub fn is_unitary(&self) -> bool {
        &self.adjoint() * self == Self::identity()
    }

    /// Multiplies every entry by the scalar `omega^exponent`.
    #[must_use]
    pub fn with_global_phase(&self, exponent: u8) -> Self {
        let mut scalar = ZOmega::one();
        for _ in 0..exponent % 8 {
            scalar = &scalar * &ZOmega::omega();
        }
        let scalar = DOmega::from(scalar);
        Self::new([
            [&self.entries[0][0] * &scalar, &self.entries[0][1] * &scalar],
            [&self.entries[1][0] * &scalar, &self.entries[1][1] * &scalar],
        ])
    }
}

impl Mul for Matrix {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
        Matrix::new([
            [
                &(&self.entries[0][0] * &rhs.entries[0][0])
                    + &(&self.entries[0][1] * &rhs.entries[1][0]),
                &(&self.entries[0][0] * &rhs.entries[0][1])
                    + &(&self.entries[0][1] * &rhs.entries[1][1]),
            ],
            [
                &(&self.entries[1][0] * &rhs.entries[0][0])
                    + &(&self.entries[1][1] * &rhs.entries[1][0]),
                &(&self.entries[1][0] * &rhs.entries[0][1])
                    + &(&self.entries[1][1] * &rhs.entries[1][1]),
            ],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::{Gate, Matrix};

    /// Absolute fixtures pinning the ORIENTATION of the convention.
    ///
    /// Every other test in this crate is relational: `omega^2 == i`,
    /// `T^2 == S`, the automorphism identities, the round trip and the
    /// enumeration count all hold equally under the coordinated relabelling
    /// `omega -> omega^{-1}`, `i -> -i`. That mirror image is a DIFFERENT
    /// convention -- it makes `T` the conjugate `diag(1, e^{-i pi/4})`, i.e.
    /// PECOS's `Tdg` -- so it must be pinned by absolute coordinates, not by
    /// relations. Do not rewrite these as relational checks.
    #[test]
    fn generator_convention_is_pinned_by_absolute_coordinates() {
        use crate::ring::ZOmega;
        use num_bigint::BigInt;

        fn coords(a: i64, b: i64, c: i64, d: i64) -> [BigInt; 4] {
            [
                BigInt::from(a),
                BigInt::from(b),
                BigInt::from(c),
                BigInt::from(d),
            ]
        }

        // omega is the first basis vector after the constant term, and i is
        // the second: omega = [0,1,0,0], i = omega^2 = [0,0,1,0].
        assert_eq!(ZOmega::omega().coordinates(), &coords(0, 1, 0, 0));
        assert_eq!(ZOmega::i().coordinates(), &coords(0, 0, 1, 0));
        // sqrt2 = omega - omega^3 = [0,1,0,-1].
        assert_eq!(ZOmega::sqrt2().coordinates(), &coords(0, 1, 0, -1));

        // S = diag(1, i) and T = diag(1, omega), with the POSITIVE angle in
        // the lower-right entry. Under the mirrored convention these would be
        // [0,0,-1,0] and [0,0,0,-1] respectively.
        let s = Matrix::from_gate(Gate::S);
        assert_eq!(
            s.entries()[1][1].numerator().coordinates(),
            &coords(0, 0, 1, 0)
        );
        let t = Matrix::from_gate(Gate::T);
        assert_eq!(
            t.entries()[1][1].numerator().coordinates(),
            &coords(0, 1, 0, 0)
        );

        // One asymmetric multi-gate product, computed by hand. In execution
        // order [T, H] the matrix is H * T:
        //   H = (1/sqrt2) [[1, 1], [1, -1]],  T = diag(1, omega)
        //   H * T = (1/sqrt2) [[1, omega], [1, -omega]]
        // The (0,1) entry is omega/sqrt2. Its numerator distinguishes omega
        // from omega^{-1} = -omega^3, which would be [0,0,0,-1].
        let ht = Matrix::from_word(&[Gate::T, Gate::H]);
        let upper_right = &ht.entries()[0][1];
        assert_eq!(upper_right.least_denominator_exponent(), 1);
        assert_eq!(upper_right.numerator().coordinates(), &coords(0, 1, 0, 0));
        let lower_right = &ht.entries()[1][1];
        assert_eq!(lower_right.least_denominator_exponent(), 1);
        assert_eq!(lower_right.numerator().coordinates(), &coords(0, -1, 0, 0));
    }

    #[test]
    fn execution_order_is_reverse_matrix_order() {
        assert_eq!(
            Matrix::from_word(&[Gate::T, Gate::H]),
            Matrix::from_gate(Gate::H) * Matrix::from_gate(Gate::T)
        );
    }

    #[test]
    fn generators_and_scalars_are_exactly_unitary() {
        for gate in [
            Gate::I,
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::H,
            Gate::S,
            Gate::Sdg,
            Gate::T,
            Gate::Tdg,
        ] {
            assert!(Matrix::from_gate(gate).is_unitary(), "{gate:?}");
        }

        for exponent in 0..8 {
            assert!(Matrix::identity().with_global_phase(exponent).is_unitary());
        }
    }

    #[test]
    fn adjoints_match_inverse_tokens() {
        assert_eq!(
            Matrix::from_gate(Gate::S).adjoint(),
            Matrix::from_gate(Gate::Sdg)
        );
        assert_eq!(
            Matrix::from_gate(Gate::T).adjoint(),
            Matrix::from_gate(Gate::Tdg)
        );
    }

    #[test]
    fn phase_generators_use_the_conventional_positive_angle() {
        assert_eq!(
            Matrix::from_gate(Gate::T) * Matrix::from_gate(Gate::T),
            Matrix::from_gate(Gate::S)
        );
        assert_eq!(
            Matrix::from_gate(Gate::S) * Matrix::from_gate(Gate::S),
            Matrix::from_gate(Gate::Z)
        );
    }
}
