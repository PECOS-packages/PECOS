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

//! Validated binary symplectic matrices for Pauli operators.

use crate::{F2Matrix, PauliSequence};
use pecos_core::{Pauli, PauliString, QuarterPhase, QubitId};
use std::fmt;

/// Errors that can occur when constructing a [`SymplecticMatrix`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymplecticMatrixError {
    /// No rows were supplied, so the matrix width cannot be inferred.
    EmptyRows,
    /// A row has a different width from the first row.
    RaggedRows {
        /// Index of the mismatched row.
        row: usize,
        /// Width inferred from the first row.
        expected: usize,
        /// Actual width of the mismatched row.
        actual: usize,
    },
    /// A dense entry was not binary.
    InvalidEntry {
        /// Row containing the invalid entry.
        row: usize,
        /// Column containing the invalid entry.
        column: usize,
        /// Invalid value.
        value: u8,
    },
    /// A symplectic matrix must have equally sized X and Z blocks.
    OddColumnCount {
        /// Number of supplied columns.
        columns: usize,
    },
    /// A Pauli operator acts beyond the requested explicit width.
    OperatorExceedsWidth {
        /// Index of the offending operator.
        row: usize,
        /// Offending qubit index.
        qubit: usize,
        /// Requested number of qubits.
        num_qubits: usize,
    },
}

impl fmt::Display for SymplecticMatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRows => write!(
                f,
                "cannot infer symplectic matrix width from empty input; use SymplecticMatrix::zeros"
            ),
            Self::RaggedRows {
                row,
                expected,
                actual,
            } => write!(
                f,
                "symplectic matrix row {row} has {actual} columns, expected {expected}"
            ),
            Self::InvalidEntry { row, column, value } => write!(
                f,
                "symplectic matrix entry at row {row}, column {column} is {value}, expected 0 or 1"
            ),
            Self::OddColumnCount { columns } => write!(
                f,
                "symplectic matrix has {columns} columns, expected an even column count"
            ),
            Self::OperatorExceedsWidth {
                row,
                qubit,
                num_qubits,
            } => write!(
                f,
                "Pauli operator at row {row} acts on qubit {qubit}, outside the explicit width of {num_qubits} qubits"
            ),
        }
    }
}

impl std::error::Error for SymplecticMatrixError {}

/// A binary symplectic matrix whose rows are Pauli operators.
///
/// For `n` qubits, columns are ordered as
/// `[x_0, ..., x_{n-1} | z_0, ..., z_{n-1}]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymplecticMatrix {
    matrix: F2Matrix,
}

impl SymplecticMatrix {
    /// Constructs a validated symplectic matrix from dense binary rows.
    ///
    /// # Errors
    ///
    /// Returns an error for empty input, ragged rows, non-binary entries, or
    /// an odd number of columns.
    pub fn from_dense(rows: Vec<Vec<u8>>) -> Result<Self, SymplecticMatrixError> {
        let Some(first) = rows.first() else {
            return Err(SymplecticMatrixError::EmptyRows);
        };
        let num_cols = first.len();
        if num_cols % 2 != 0 {
            return Err(SymplecticMatrixError::OddColumnCount { columns: num_cols });
        }
        for (row_index, row) in rows.iter().enumerate() {
            if row.len() != num_cols {
                return Err(SymplecticMatrixError::RaggedRows {
                    row: row_index,
                    expected: num_cols,
                    actual: row.len(),
                });
            }
            for (column, &value) in row.iter().enumerate() {
                if value > 1 {
                    return Err(SymplecticMatrixError::InvalidEntry {
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
    pub fn zeros(num_rows: usize, num_qubits: usize) -> Self {
        Self {
            matrix: F2Matrix::zeros(num_rows, 2 * num_qubits),
        }
    }

    /// Converts a Pauli sequence to symplectic form at an explicit width.
    ///
    /// Pauli phases are deliberately ignored. Use [`to_positive_paulis`](Self::to_positive_paulis)
    /// to recover operators with phase `+1`.
    ///
    /// # Errors
    ///
    /// Returns an error if any operator acts on a qubit outside `num_qubits`.
    pub fn from_pauli_sequence_ignoring_phase(
        sequence: &PauliSequence,
        num_qubits: usize,
    ) -> Result<Self, SymplecticMatrixError> {
        for (row, operator) in sequence.iter().enumerate() {
            if let Some(qubit) = operator
                .qubits()
                .into_iter()
                .find(|&qubit| qubit >= num_qubits)
            {
                return Err(SymplecticMatrixError::OperatorExceedsWidth {
                    row,
                    qubit,
                    num_qubits,
                });
            }
        }

        let inferred_num_qubits = sequence.num_qubits();
        let inferred = sequence.to_symplectic_matrix();
        let mut matrix = F2Matrix::zeros(sequence.len(), 2 * num_qubits);
        for row in 0..sequence.len() {
            for qubit in 0..inferred_num_qubits {
                matrix.set(row, qubit, inferred.get(row, qubit));
                matrix.set(
                    row,
                    num_qubits + qubit,
                    inferred.get(row, inferred_num_qubits + qubit),
                );
            }
        }
        Ok(Self { matrix })
    }

    /// Converts each row to a phase-`+1` Pauli operator.
    ///
    /// Symplectic matrices contain no sign or quarter-phase information, so
    /// every returned operator necessarily has positive phase.
    #[must_use]
    pub fn to_positive_paulis(&self) -> Vec<PauliString> {
        let num_qubits = self.num_qubits();
        (0..self.num_rows())
            .map(|row| {
                let mut paulis = Vec::new();
                for qubit in 0..num_qubits {
                    let x = self.matrix.get(row, qubit);
                    let z = self.matrix.get(row, num_qubits + qubit);
                    let pauli = match (x, z) {
                        (1, 0) => Some(Pauli::X),
                        (0, 1) => Some(Pauli::Z),
                        (1, 1) => Some(Pauli::Y),
                        _ => None,
                    };
                    if let Some(pauli) = pauli {
                        paulis.push((pauli, QubitId::new(qubit)));
                    }
                }
                PauliString::with_phase_and_paulis(QuarterPhase::PlusOne, paulis)
            })
            .collect()
    }

    /// Returns the number of matrix rows.
    #[must_use]
    pub fn num_rows(&self) -> usize {
        self.matrix.num_rows()
    }

    /// Returns the number of represented qubits.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.matrix.num_cols() / 2
    }

    /// Returns a copy of the X block.
    #[must_use]
    pub fn x_block(&self) -> F2Matrix {
        let num_qubits = self.num_qubits();
        let mut block = F2Matrix::zeros(self.num_rows(), num_qubits);
        for row in 0..self.num_rows() {
            for qubit in 0..num_qubits {
                block.set(row, qubit, self.matrix.get(row, qubit));
            }
        }
        block
    }

    /// Returns a copy of the Z block.
    #[must_use]
    pub fn z_block(&self) -> F2Matrix {
        let num_qubits = self.num_qubits();
        let mut block = F2Matrix::zeros(self.num_rows(), num_qubits);
        for row in 0..self.num_rows() {
            for qubit in 0..num_qubits {
                block.set(row, qubit, self.matrix.get(row, num_qubits + qubit));
            }
        }
        block
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
        (index < self.num_rows()).then(|| self.matrix.row(index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pauli_round_trip_ignores_phase_and_preserves_explicit_width() {
        let negative_y: PauliString = "-Y0".parse().unwrap();
        let imaginary_xz: PauliString = "+i X1 Z3".parse().unwrap();
        let sequence = PauliSequence::new(vec![negative_y, imaginary_xz]);

        let matrix = SymplecticMatrix::from_pauli_sequence_ignoring_phase(&sequence, 5).unwrap();
        assert_eq!(matrix.num_rows(), 2);
        assert_eq!(matrix.num_qubits(), 5);
        assert_eq!(matrix.x_block().rows()[0], vec![1, 0, 0, 0, 0]);
        assert_eq!(matrix.z_block().rows()[0], vec![1, 0, 0, 0, 0]);

        let positive = matrix.to_positive_paulis();
        assert_eq!(positive[0].get_phase(), QuarterPhase::PlusOne);
        assert_eq!(positive[1].get_phase(), QuarterPhase::PlusOne);
        assert_eq!(positive[0].to_dense_str(Some(5)), "+YIIII");
        assert_eq!(positive[1].to_dense_str(Some(5)), "+IXIZI");
    }

    #[test]
    fn empty_dense_input_requires_explicit_zeros_constructor() {
        assert_eq!(
            SymplecticMatrix::from_dense(Vec::new()).unwrap_err(),
            SymplecticMatrixError::EmptyRows
        );
        assert_eq!(SymplecticMatrix::zeros(0, 7).num_qubits(), 7);
    }

    #[test]
    fn dense_input_rejects_ragged_rows() {
        assert_eq!(
            SymplecticMatrix::from_dense(vec![vec![1, 0], vec![1]]).unwrap_err(),
            SymplecticMatrixError::RaggedRows {
                row: 1,
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn dense_input_rejects_non_binary_entries() {
        assert_eq!(
            SymplecticMatrix::from_dense(vec![vec![0, 2]]).unwrap_err(),
            SymplecticMatrixError::InvalidEntry {
                row: 0,
                column: 1,
                value: 2,
            }
        );
    }

    #[test]
    fn pauli_sequence_rejects_operator_beyond_explicit_width() {
        let sequence = PauliSequence::new(vec![PauliString::x(9)]);

        assert_eq!(
            SymplecticMatrix::from_pauli_sequence_ignoring_phase(&sequence, 3).unwrap_err(),
            SymplecticMatrixError::OperatorExceedsWidth {
                row: 0,
                qubit: 9,
                num_qubits: 3,
            }
        );
    }
}
