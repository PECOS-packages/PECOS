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

//! Role-neutral binary parity-check matrices for QEC codes.

use pecos_core::{Pauli, PauliString, QuarterPhase, QubitId};
use pecos_quantum::F2Matrix;
use thiserror::Error;

/// Errors that can occur when constructing a [`ParityCheckMatrix`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParityCheckMatrixError {
    /// No rows were supplied, so the matrix width cannot be inferred.
    #[error(
        "cannot infer parity-check matrix width from empty input; use ParityCheckMatrix::zeros"
    )]
    EmptyRows,
    /// A row has a different width from the first row.
    #[error("parity-check matrix row {row} has {actual} columns, expected {expected}")]
    RaggedRows {
        /// Index of the mismatched row.
        row: usize,
        /// Width inferred from the first row.
        expected: usize,
        /// Actual width of the mismatched row.
        actual: usize,
    },
    /// A dense entry was not binary.
    #[error("parity-check matrix entry at row {row}, column {column} is {value}, expected 0 or 1")]
    InvalidEntry {
        /// Row containing the invalid entry.
        row: usize,
        /// Column containing the invalid entry.
        column: usize,
        /// Invalid value.
        value: u8,
    },
}

/// A role-neutral binary matrix whose rows are checks and columns are qubits.
///
/// Whether rows become X-type or Z-type stabilizers is chosen only when
/// converting the matrix; that role is not stored in this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParityCheckMatrix {
    matrix: F2Matrix,
}

impl ParityCheckMatrix {
    /// Constructs a validated parity-check matrix from dense binary rows.
    ///
    /// # Errors
    ///
    /// Returns an error for empty input, ragged rows, or non-binary entries.
    pub fn from_dense(rows: Vec<Vec<u8>>) -> Result<Self, ParityCheckMatrixError> {
        let Some(first) = rows.first() else {
            return Err(ParityCheckMatrixError::EmptyRows);
        };
        let num_qubits = first.len();
        for (row_index, row) in rows.iter().enumerate() {
            if row.len() != num_qubits {
                return Err(ParityCheckMatrixError::RaggedRows {
                    row: row_index,
                    expected: num_qubits,
                    actual: row.len(),
                });
            }
            for (column, &value) in row.iter().enumerate() {
                if value > 1 {
                    return Err(ParityCheckMatrixError::InvalidEntry {
                        row: row_index,
                        column,
                        value,
                    });
                }
            }
        }
        Ok(Self {
            matrix: F2Matrix::from_rows(rows),
        })
    }

    /// Constructs an all-zero matrix with an explicit number of qubits.
    #[must_use]
    pub fn zeros(num_checks: usize, num_qubits: usize) -> Self {
        Self {
            matrix: F2Matrix::zeros(num_checks, num_qubits),
        }
    }

    /// Returns the number of checks (rows).
    #[must_use]
    pub fn num_checks(&self) -> usize {
        self.matrix.num_rows()
    }

    /// Returns the number of qubits (columns).
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.matrix.num_cols()
    }

    /// Returns the rank over GF(2).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.matrix.row_reduce().1.len()
    }

    /// Returns dense copies of all rows.
    #[must_use]
    pub fn rows(&self) -> Vec<Vec<u8>> {
        self.matrix.rows()
    }

    /// Returns a dense copy of one row, or `None` if the index is out of range.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<Vec<u8>> {
        (index < self.num_checks()).then(|| self.matrix.row(index))
    }

    /// Converts rows to stabilizers made of Pauli X operators, with phase `+1`.
    ///
    /// “X stabilizers” means stabilizers made of X, not stabilizers that detect
    /// X errors.
    #[must_use]
    pub fn to_x_stabilizers(&self) -> Vec<PauliString> {
        self.to_stabilizers(Pauli::X)
    }

    /// Converts rows to stabilizers made of Pauli Z operators, with phase `+1`.
    ///
    /// “Z stabilizers” means stabilizers made of Z, not stabilizers that detect
    /// Z errors.
    #[must_use]
    pub fn to_z_stabilizers(&self) -> Vec<PauliString> {
        self.to_stabilizers(Pauli::Z)
    }

    pub(crate) fn matrix(&self) -> &F2Matrix {
        &self.matrix
    }

    fn to_stabilizers(&self, pauli: Pauli) -> Vec<PauliString> {
        (0..self.num_checks())
            .map(|row| {
                let paulis = (0..self.num_qubits())
                    .filter(|&qubit| self.matrix.get(row, qubit) == 1)
                    .map(|qubit| (pauli, QubitId::new(qubit)))
                    .collect();
                PauliString::with_phase_and_paulis(QuarterPhase::PlusOne, paulis)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_detects_dependent_rows() {
        let matrix =
            ParityCheckMatrix::from_dense(vec![vec![1, 1, 0], vec![0, 1, 1], vec![1, 0, 1]])
                .unwrap();

        assert_eq!(matrix.num_checks(), 3);
        assert_eq!(matrix.rank(), 2);
    }

    #[test]
    fn zero_row_matrix_preserves_width() {
        let matrix = ParityCheckMatrix::zeros(0, 9);
        assert_eq!(matrix.num_checks(), 0);
        assert_eq!(matrix.num_qubits(), 9);
        assert!(matrix.rows().is_empty());
    }
}
