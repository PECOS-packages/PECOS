// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Binomial bivariate tricycle codes and their depth-two cup-product CCZ.
//!
//! The matrices follow arXiv:2510.07269, equation (`3D_BT_codes`) and the
//! table "classical code choice", and arXiv:2508.10714, "Tricycle codes".
//! Monomial matrices use the same forward-shift convention as bivariate bicycle
//! codes: group element `(x, y)` has index `x*m + y`.

use pecos_quantum::F2Matrix;
use thiserror::Error;

use crate::bivariate_bicycle::{
    TorusValidationError, forward_shift, monomial_matrices, sum_matrices, transpose_shift,
    validate_terms, validated_block_size,
};
use crate::memory_circuit::discover_css_logical_operators;
use crate::{BbMonomial, ParityCheckMatrix};

/// Errors reported while constructing a binomial bivariate tricycle code.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BivariateTricycleError {
    /// A torus dimension was zero.
    #[error("bivariate-tricycle dimensions must be positive, got l={l}, m={m}")]
    ZeroDimension { l: usize, m: usize },
    /// The block or code size overflowed `usize`.
    #[error("bivariate-tricycle dimensions l={l}, m={m} overflow the supported size")]
    SizeOverflow { l: usize, m: usize },
    /// A binomial did not contain exactly two terms.
    #[error("{polynomial} must contain exactly two monomials, got {actual}")]
    WrongTermCount {
        polynomial: &'static str,
        actual: usize,
    },
    /// A monomial exponent was outside the canonical torus range.
    #[error(
        "{polynomial} monomial {term_index} has exponent ({x_power}, {y_power}) outside Z_{l} x Z_{m}"
    )]
    ExponentOutOfRange {
        polynomial: &'static str,
        term_index: usize,
        x_power: usize,
        y_power: usize,
        l: usize,
        m: usize,
    },
    /// Two terms describe the same permutation matrix.
    #[error("{polynomial} monomials {first} and {second} describe the same permutation matrix")]
    DuplicateMonomial {
        polynomial: &'static str,
        first: usize,
        second: usize,
    },
    /// A generated monomial was not a permutation matrix.
    #[error("{polynomial} monomial {term_index} is not a permutation matrix")]
    NonPermutationMonomial {
        polynomial: &'static str,
        term_index: usize,
    },
    /// The CSS commutation condition failed.
    #[error("bivariate-tricycle checks do not commute: Hx * Hz^T is nonzero")]
    NonCommutingChecks,
}

impl From<TorusValidationError> for BivariateTricycleError {
    fn from(error: TorusValidationError) -> Self {
        match error {
            TorusValidationError::ZeroDimension { l, m } => Self::ZeroDimension { l, m },
            TorusValidationError::SizeOverflow { l, m } => Self::SizeOverflow { l, m },
            TorusValidationError::WrongTermCount { polynomial, actual } => {
                Self::WrongTermCount { polynomial, actual }
            }
            TorusValidationError::ExponentOutOfRange {
                polynomial,
                term_index,
                x_power,
                y_power,
                l,
                m,
            } => Self::ExponentOutOfRange {
                polynomial,
                term_index,
                x_power,
                y_power,
                l,
                m,
            },
            TorusValidationError::DuplicateMonomial {
                polynomial,
                first,
                second,
            } => Self::DuplicateMonomial {
                polynomial,
                first,
                second,
            },
            TorusValidationError::NonPermutationMonomial {
                polynomial,
                term_index,
            } => Self::NonPermutationMonomial {
                polynomial,
                term_index,
            },
        }
    }
}

/// A validated three-sector CSS code over `F2[x,y]/(x^l+1,y^m+1)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BivariateTricycleCode {
    l: usize,
    m: usize,
    terms: [[BbMonomial; 2]; 3],
    hx: ParityCheckMatrix,
    hz: ParityCheckMatrix,
    meta_z: ParityCheckMatrix,
    logical_x: ParityCheckMatrix,
    logical_z: ParityCheckMatrix,
}

impl BivariateTricycleCode {
    /// Construct the 3D BT code of arXiv:2510.07269, equation (`3D_BT_codes`).
    ///
    /// Each polynomial has two distinct canonical terms, with exponents in
    /// `0..l` and `0..m`. A monomial's one in row `(x,y)` is in column
    /// `(x+x_power mod l, y+y_power mod m)`. All `3lm` Z-check rows are retained.
    ///
    /// # Errors
    /// Rejects zero dimensions, size overflow, invalid binomials, non-permutation
    /// monomial matrices, or failure of `Hx * Hz^T = 0`.
    ///
    /// # Panics
    /// Panics only if an internally generated nonempty rectangular binary matrix
    /// is rejected by [`ParityCheckMatrix`].
    pub fn new(
        l: usize,
        m: usize,
        a_terms: &[BbMonomial],
        b_terms: &[BbMonomial],
        c_terms: &[BbMonomial],
    ) -> Result<Self, BivariateTricycleError> {
        let size = validated_block_size(l, m)?;
        let n = size
            .checked_mul(3)
            .ok_or(BivariateTricycleError::SizeOverflow { l, m })?;
        let terms = [
            validate_terms("A", l, m, a_terms)?,
            validate_terms("B", l, m, b_terms)?,
            validate_terms("C", l, m, c_terms)?,
        ];
        let matrices = [
            sum_matrices(&monomial_matrices("A", l, m, &terms[0])?),
            sum_matrices(&monomial_matrices("B", l, m, &terms[1])?),
            sum_matrices(&monomial_matrices("C", l, m, &terms[2])?),
        ];
        let mut hx = F2Matrix::zeros(size, n);
        let mut hz = F2Matrix::zeros(n, n);
        let mut meta_z = F2Matrix::zeros(size, n);
        for row in 0..size {
            for column in 0..size {
                for (sector, matrix) in matrices.iter().enumerate() {
                    hx.set(row, sector * size + column, matrix.get(row, column));
                    meta_z.set(row, (2 - sector) * size + column, matrix.get(column, row));
                }
                for (group, (i, j)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
                    hz.set(
                        group * size + row,
                        i * size + column,
                        matrices[j].get(column, row),
                    );
                    hz.set(
                        group * size + row,
                        j * size + column,
                        matrices[i].get(column, row),
                    );
                }
            }
        }
        if hx.mul(&hz.transpose()) != F2Matrix::zeros(size, n) {
            return Err(BivariateTricycleError::NonCommutingChecks);
        }
        let parity = |matrix: F2Matrix| {
            ParityCheckMatrix::from_dense(matrix.rows())
                .expect("a nonempty rectangular binary matrix was generated")
        };
        let hx = parity(hx);
        let hz = parity(hz);
        let (logical_x, logical_z) = discover_css_logical_operators(&hx, &hz);
        Ok(Self {
            l,
            m,
            terms,
            hx,
            hz,
            meta_z: parity(meta_z),
            logical_x,
            logical_z,
        })
    }

    /// Number of data qubits, `n = 3lm`.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.hx.num_qubits()
    }

    /// Number of encoded qubits, `k = n - rank(Hx) - rank(Hz)`.
    #[must_use]
    pub fn num_logical_qubits(&self) -> usize {
        self.num_qubits() - self.hx.rank() - self.hz.rank()
    }

    /// X checks `Hx = [A | B | C]`.
    #[must_use]
    pub fn hx(&self) -> &ParityCheckMatrix {
        &self.hx
    }

    /// All Z checks, in groups `[B^T | A^T | 0]`, `[C^T | 0 | A^T]`,
    /// `[0 | C^T | B^T]`, as in arXiv:2510.07269, equation (`3D_BT_codes`).
    #[must_use]
    pub fn hz(&self) -> &ParityCheckMatrix {
        &self.hz
    }

    /// Z-check dependencies `M_Z = [C^T | B^T | A^T]`, with `M_Z H_Z = 0`.
    ///
    /// Rows are dependencies; columns index the `3lm` **rows** of `H_Z`, so no
    /// transpose of `H_Z` is taken in this identity. This is the next boundary
    /// map of arXiv:2510.07269, equation (`3D_BT_codes`).
    #[must_use]
    pub fn meta_z(&self) -> &ParityCheckMatrix {
        &self.meta_z
    }

    /// A basis of logical X operators, represented as binary rows.
    #[must_use]
    pub fn logical_x(&self) -> &ParityCheckMatrix {
        &self.logical_x
    }

    /// A basis of logical Z operators, represented as binary rows.
    #[must_use]
    pub fn logical_z(&self) -> &ParityCheckMatrix {
        &self.logical_z
    }

    /// Torus dimensions `(l,m)`.
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.l, self.m)
    }

    /// The two canonical terms of A, in input order.
    #[must_use]
    pub fn a_terms(&self) -> &[BbMonomial; 2] {
        &self.terms[0]
    }

    /// The two canonical terms of B, in input order.
    #[must_use]
    pub fn b_terms(&self) -> &[BbMonomial; 2] {
        &self.terms[1]
    }

    /// The two canonical terms of C, in input order.
    #[must_use]
    pub fn c_terms(&self) -> &[BbMonomial; 2] {
        &self.terms[2]
    }

    /// Two disjoint CCZ layers between three copies, with block-local indices.
    ///
    /// Uses arXiv:2508.10714, Proposition (`fccz_def`), the symmetric triple cup
    /// product. For binomials, singleton in/out partitions satisfy the
    /// non-overlapping-bits condition of arXiv:2410.16250, Section 5, and give
    /// its cup product; see also arXiv:2510.07269, appendix "Cup product and
    /// Symmetric Triple Cup Product".
    ///
    /// The first input term is `in` and the second is `out`. For distinct
    /// sectors `(i,j,k)`, solve `r in_i in_j = q in_i out_k = p out_j out_k`.
    /// Qubit `(s,g)` is `s*lm + idx(g)`. Even permutations form layer zero,
    /// odd permutations layer one; each layer touches every qubit once per block.
    #[must_use]
    pub fn cup_product_ccz(&self) -> [Vec<[usize; 3]>; 2] {
        let size = self.l * self.m;
        [
            [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
            [[0, 2, 1], [2, 1, 0], [1, 0, 2]],
        ]
        .map(|permutations| {
            let mut gates = Vec::with_capacity(self.num_qubits());
            for [i, j, k] in permutations {
                for p in 0..size {
                    let q = transpose_shift(
                        forward_shift(p, self.l, self.m, self.terms[j][1]),
                        self.l,
                        self.m,
                        self.terms[i][0],
                    );
                    let r = transpose_shift(
                        forward_shift(q, self.l, self.m, self.terms[k][1]),
                        self.l,
                        self.m,
                        self.terms[j][0],
                    );
                    gates.push([i * size + p, j * size + q, k * size + r]);
                }
            }
            gates
        })
    }
}

#[cfg(test)]
mod tests;
