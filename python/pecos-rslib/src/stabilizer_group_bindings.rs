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

//! Python bindings for `PauliStabilizerGroup`.

use crate::prelude::PauliString as RustPauliString;
use pecos_quantum::PauliStabilizerGroup as RustGroup;
use pyo3::prelude::*;

use crate::pauli_bindings::PauliString;

/// A validated Pauli stabilizer group.
///
/// All generators mutually commute and have real phases (+1 or -1).
/// Provides algebraic analysis: rank, group membership, element
/// enumeration, and Clifford conjugation.
///
/// Examples:
///     >>> from pecos_rslib import PauliStabilizerGroup
///     >>> # 3-qubit repetition code stabilizers
///     >>> code = PauliStabilizerGroup.from_str("ZZI\nIZZ")
///     >>> code.rank()
///     2
#[allow(clippy::doc_markdown)]
#[pyclass(name = "PauliStabilizerGroup", module = "pecos_rslib", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyPauliStabilizerGroup {
    pub(crate) inner: RustGroup,
}

unsafe impl Send for PyPauliStabilizerGroup {}
unsafe impl Sync for PyPauliStabilizerGroup {}

#[pymethods]
#[allow(clippy::doc_markdown)]
impl PyPauliStabilizerGroup {
    /// Create a stabilizer group from a list of `PauliString` generators.
    ///
    /// Args:
    ///     generators: List of PauliString stabilizer generators
    ///
    /// Raises:
    ///     ValueError: If generators don't commute or have non-real phases
    #[new]
    #[pyo3(signature = (generators))]
    fn new(generators: Vec<PauliString>) -> PyResult<Self> {
        let rust_gens: Vec<RustPauliString> = generators.into_iter().map(|g| g.to_rust()).collect();
        let group = RustGroup::new(rust_gens).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid stabilizer group: {e}"))
        })?;
        Ok(Self { inner: group })
    }

    /// Create from newline-delimited Pauli strings.
    ///
    /// Supports both dense ("ZZI") and sparse ("Z0 Z1") formats.
    ///
    /// Args:
    ///     s: Newline-delimited string of Pauli generators
    ///
    /// Returns:
    ///     PauliStabilizerGroup
    ///
    /// Examples:
    ///     >>> PauliStabilizerGroup.from_str("ZZI\nIZZ")
    ///     >>> PauliStabilizerGroup.from_str("Z0 Z1\nZ1 Z2")
    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        let group: RustGroup = s.parse().map_err(|e: Box<dyn std::error::Error>| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to parse: {e}"))
        })?;
        Ok(Self { inner: group })
    }

    // ========================================================================
    // Algebraic analysis methods
    // ========================================================================

    /// Number of independent generators (rank of the symplectic matrix).
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Number of physical qubits.
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Number of generators.
    fn num_generators(&self) -> usize {
        self.inner.num_generators()
    }

    /// Whether all generators are linearly independent.
    fn is_independent(&self) -> bool {
        self.inner.is_independent()
    }

    /// Check if a Pauli string is in the stabilizer group (ignoring phase).
    fn contains(&self, pauli: &PauliString) -> bool {
        self.inner.contains(&pauli.to_rust())
    }

    /// Check if a Pauli string is in the stabilizer group (exact phase match).
    fn contains_with_phase(&self, pauli: &PauliString) -> bool {
        self.inner.contains_with_phase(&pauli.to_rust())
    }

    /// Get the list of stabilizer generators.
    fn stabilizers(&self) -> Vec<PauliString> {
        self.inner
            .stabilizers()
            .iter()
            .map(|p| PauliString::from_rust(p.clone()))
            .collect()
    }

    // ========================================================================
    // Mutation methods
    // ========================================================================

    /// Add a generator to the stabilizer group.
    ///
    /// The new generator must commute with all existing generators and
    /// have a real phase (+1 or -1).
    ///
    /// Args:
    ///     generator: PauliString to add
    ///
    /// Raises:
    ///     ValueError: If generator doesn't commute or has non-real phase
    fn add_generator(&mut self, generator: &PauliString) -> PyResult<()> {
        self.inner
            .add_generator(generator.to_rust())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    /// Remove and return the generator at the given index.
    ///
    /// Args:
    ///     index: Index of the generator to remove
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn remove_generator(&mut self, index: usize) -> PyResult<PauliString> {
        if index >= self.inner.num_generators() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {index} out of range for {} generators",
                self.inner.num_generators()
            )));
        }
        Ok(PauliString::from_rust(self.inner.remove_generator(index)))
    }

    /// Merge another stabilizer group into this one.
    ///
    /// All generators from other must commute with all generators in self.
    ///
    /// Args:
    ///     other: PauliStabilizerGroup to merge
    ///
    /// Raises:
    ///     ValueError: If any generators anticommute
    fn merge(&mut self, other: &PyPauliStabilizerGroup) -> PyResult<()> {
        self.inner
            .merge(&other.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    // ========================================================================
    // String representations
    // ========================================================================

    /// Dense string representation (one stabilizer per line).
    fn to_dense_str(&self) -> String {
        self.inner.to_dense_str()
    }

    /// Sparse string representation (one stabilizer per line).
    fn to_sparse_str(&self) -> String {
        self.inner.to_sparse_str()
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        let n = self.inner.num_generators();
        format!("PauliStabilizerGroup({n} generators)")
    }

    fn __len__(&self) -> usize {
        self.inner.num_generators()
    }
}

impl PauliString {
    /// Convert from Python `PauliString` to Rust `PauliString`.
    pub(crate) fn to_rust(&self) -> RustPauliString {
        self.inner.clone()
    }

    /// Convert from Rust `PauliString` to Python `PauliString`.
    pub(crate) fn from_rust(p: RustPauliString) -> Self {
        Self { inner: p }
    }
}

/// Register stabilizer group types with Python module.
pub fn register_stabilizer_group_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPauliStabilizerGroup>()?;
    Ok(())
}
