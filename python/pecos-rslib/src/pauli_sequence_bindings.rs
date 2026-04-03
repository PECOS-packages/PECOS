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

//! Python bindings for `PauliSequence` — ordered collection with GF(2) analysis.

use pecos_quantum::PauliSequence as RustPauliSequence;
use pyo3::prelude::*;

use crate::pauli_bindings::PauliString;

/// An ordered sequence of Pauli strings with symplectic GF(2) analysis.
///
/// Unlike PauliStabilizerGroup, this does NOT require commutativity or
/// real phases. Useful for general Pauli algebra: rank, membership,
/// independence, centralizer computation, and row reduction.
///
/// Examples:
///     >>> from pecos_rslib import PauliSequence
///     >>> seq = PauliSequence.from_str("ZZI\nIZZ")
///     >>> seq.rank()
///     2
///     >>> seq.is_abelian()
///     True
#[allow(clippy::doc_markdown)]
#[pyclass(name = "PauliSequence", module = "pecos_rslib", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyPauliSequence {
    pub(crate) inner: RustPauliSequence,
}

unsafe impl Send for PyPauliSequence {}
unsafe impl Sync for PyPauliSequence {}

#[pymethods]
#[allow(clippy::doc_markdown)]
impl PyPauliSequence {
    /// Create a PauliSequence from a list of PauliString objects.
    ///
    /// Args:
    ///     paulis: List of PauliString objects
    #[new]
    #[pyo3(signature = (paulis))]
    fn new(paulis: Vec<PauliString>) -> Self {
        let rust_paulis = paulis.into_iter().map(|p| p.to_rust()).collect();
        Self {
            inner: RustPauliSequence::new(rust_paulis),
        }
    }

    /// Create from newline-delimited Pauli strings.
    ///
    /// Supports both dense ("ZZI") and sparse ("Z0 Z1") formats.
    ///
    /// Args:
    ///     s: Newline-delimited string of Pauli operators
    ///
    /// Returns:
    ///     PauliSequence
    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        let seq: RustPauliSequence =
            s.parse().map_err(|e: pecos_core::ParsePauliStringError| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to parse: {e}"))
            })?;
        Ok(Self { inner: seq })
    }

    // ========================================================================
    // Basic properties
    // ========================================================================

    /// Number of Pauli strings in the sequence.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of physical qubits.
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Whether the sequence is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the list of Pauli strings.
    fn paulis(&self) -> Vec<PauliString> {
        self.inner
            .paulis()
            .iter()
            .map(|p| PauliString::from_rust(p.clone()))
            .collect()
    }

    // ========================================================================
    // GF(2) analysis methods
    // ========================================================================

    /// Rank of the symplectic matrix (number of linearly independent Paulis).
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Check if a Pauli string is in the GF(2) span (ignoring phase).
    fn contains(&self, pauli: &PauliString) -> bool {
        self.inner.contains(&pauli.to_rust())
    }

    /// Check if a Pauli string is in the GF(2) span (exact phase match).
    fn contains_with_phase(&self, pauli: &PauliString) -> bool {
        self.inner.contains_with_phase(&pauli.to_rust())
    }

    /// Check if all Pauli strings mutually commute.
    fn is_abelian(&self) -> bool {
        self.inner.is_abelian()
    }

    /// Commutation matrix: result[i][j] is True if i and j commute.
    fn commutation_matrix(&self) -> Vec<Vec<bool>> {
        self.inner.commutation_matrix()
    }

    /// Row-reduced form: independent Pauli strings in echelon form.
    ///
    /// Returns a new PauliSequence with redundant elements removed and
    /// phases tracked correctly through the GF(2) row operations.
    fn row_reduce(&self) -> Self {
        Self {
            inner: self.inner.row_reduce(),
        }
    }

    /// Centralizer basis as symplectic vectors.
    ///
    /// Returns the basis for all n-qubit Pauli strings (ignoring phase)
    /// that commute with every element in this sequence. Each vector has
    /// length 2*num_qubits: (x_0..x_{n-1} | z_0..z_{n-1}).
    fn centralizer(&self) -> Vec<Vec<u8>> {
        self.inner.centralizer()
    }

    // ========================================================================
    // String representations
    // ========================================================================

    /// Dense string representation (one Pauli per line, padded to num_qubits).
    fn to_dense_str(&self) -> String {
        self.inner.to_dense_str()
    }

    /// Sparse string representation (one Pauli per line, with phase prefix).
    fn to_sparse_str(&self) -> String {
        self.inner.to_sparse_str()
    }

    fn __str__(&self) -> String {
        self.inner.to_dense_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "PauliSequence(len={}, num_qubits={})",
            self.inner.len(),
            self.inner.num_qubits()
        )
    }
}

/// Register `PauliSequence` types with Python module.
pub fn register_pauli_sequence_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPauliSequence>()?;
    Ok(())
}
