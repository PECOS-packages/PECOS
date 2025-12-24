//! Sparse matrix representation for LDPC codes

#![allow(clippy::similar_names)]

use ndarray::{Array2, ArrayView2};
use std::collections::BTreeSet;

/// Sparse matrix in COO (Coordinate) format
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_indices: Vec<u32>,
    pub col_indices: Vec<u32>,
}

impl SparseMatrix {
    /// Create a new empty sparse matrix
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
        }
    }

    /// Create from a dense matrix
    #[must_use]
    pub fn from_dense(dense: &ArrayView2<u8>) -> Self {
        let (rows, cols) = dense.dim();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();

        for ((i, j), &val) in dense.indexed_iter() {
            if val != 0 {
                row_indices.push(u32::try_from(i).unwrap_or(0));
                col_indices.push(u32::try_from(j).unwrap_or(0));
            }
        }

        Self {
            rows,
            cols,
            row_indices,
            col_indices,
        }
    }

    /// Create from COO format arrays
    ///
    /// # Errors
    ///
    /// Returns an error if the row and column index arrays have different lengths or if indices are out of bounds.
    pub fn from_coo(
        rows: usize,
        cols: usize,
        row_indices: Vec<u32>,
        col_indices: Vec<u32>,
    ) -> Result<Self, String> {
        if row_indices.len() != col_indices.len() {
            return Err("Row and column indices must have the same length".to_string());
        }

        // Validate indices
        for (&r, &c) in row_indices.iter().zip(col_indices.iter()) {
            if r as usize >= rows || c as usize >= cols {
                return Err(format!(
                    "Index ({r}, {c}) out of bounds for {rows}x{cols} matrix"
                ));
            }
        }

        Ok(Self {
            rows,
            cols,
            row_indices,
            col_indices,
        })
    }

    /// Get the number of non-zero elements
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.row_indices.len()
    }

    /// Convert to dense matrix
    #[must_use]
    pub fn to_dense(&self) -> Array2<u8> {
        let mut dense = Array2::zeros((self.rows, self.cols));
        for (&r, &c) in self.row_indices.iter().zip(self.col_indices.iter()) {
            dense[[r as usize, c as usize]] = 1;
        }
        dense
    }

    /// Check if the matrix has duplicate entries
    #[must_use]
    pub fn has_duplicates(&self) -> bool {
        let mut seen = BTreeSet::new();
        for (&r, &c) in self.row_indices.iter().zip(self.col_indices.iter()) {
            if !seen.insert((r, c)) {
                return true;
            }
        }
        false
    }

    /// Remove duplicate entries
    pub fn remove_duplicates(&mut self) {
        let mut seen = BTreeSet::new();
        let mut new_row_indices = Vec::new();
        let mut new_col_indices = Vec::new();

        for (&r, &c) in self.row_indices.iter().zip(self.col_indices.iter()) {
            if seen.insert((r, c)) {
                new_row_indices.push(r);
                new_col_indices.push(c);
            }
        }

        self.row_indices = new_row_indices;
        self.col_indices = new_col_indices;
    }

    /// Convert to FFI representation
    pub(crate) fn to_ffi_repr(&self) -> super::bridge::ffi::SparseMatrixRepr {
        super::bridge::ffi::SparseMatrixRepr {
            rows: u32::try_from(self.rows).unwrap_or(0),
            cols: u32::try_from(self.cols).unwrap_or(0),
            row_indices: self.row_indices.clone(),
            col_indices: self.col_indices.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_sparse_from_dense() {
        let dense = arr2(&[[1, 0, 1], [0, 1, 0], [1, 1, 0]]);

        let sparse = SparseMatrix::from_dense(&dense.view());
        assert_eq!(sparse.rows, 3);
        assert_eq!(sparse.cols, 3);
        assert_eq!(sparse.nnz(), 5);
    }

    #[test]
    fn test_sparse_to_dense() {
        let sparse =
            SparseMatrix::from_coo(3, 3, vec![0, 0, 1, 2, 2], vec![0, 2, 1, 0, 1]).unwrap();

        let dense = sparse.to_dense();
        assert_eq!(dense[[0, 0]], 1);
        assert_eq!(dense[[0, 1]], 0);
        assert_eq!(dense[[0, 2]], 1);
        assert_eq!(dense[[1, 1]], 1);
        assert_eq!(dense[[2, 0]], 1);
        assert_eq!(dense[[2, 1]], 1);
    }
}
