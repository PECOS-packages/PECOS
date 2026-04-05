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

//! Python bindings for `StabilizerCode`.

use pecos_qec::StabilizerCode as RustCode;
use pyo3::prelude::*;

use crate::pauli_bindings::PauliString;
use crate::stabilizer_group_bindings::PyPauliStabilizerGroup;

/// A stabilizer code: a stabilizer group with an explicit qubit count.
///
/// Provides QEC analysis: logical qubits, distance, syndrome, and
/// logical operators. Also includes standard code constructors.
///
/// Examples:
///
/// ```python
/// >>> from pecos_rslib import StabilizerCode
/// >>> code = StabilizerCode.repetition(3)
/// >>> code.num_logical_qubits()
/// 1
/// >>> code.distance()
/// 1
///
/// >>> steane = StabilizerCode.steane()
/// >>> steane.distance()
/// 3
/// ```
#[allow(clippy::doc_markdown)]
#[pyclass(name = "StabilizerCode", module = "pecos_rslib", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyStabilizerCode {
    inner: RustCode,
}

unsafe impl Send for PyStabilizerCode {}
unsafe impl Sync for PyStabilizerCode {}

#[pymethods]
#[allow(clippy::doc_markdown)]
impl PyStabilizerCode {
    /// Create a stabilizer code from a PauliStabilizerGroup.
    ///
    /// Args:
    ///     group: PauliStabilizerGroup
    ///     num_qubits: Optional explicit qubit count (defaults to group.num_qubits())
    #[new]
    #[pyo3(signature = (group, num_qubits=None))]
    fn new(group: &PyPauliStabilizerGroup, num_qubits: Option<usize>) -> PyResult<Self> {
        let code = match num_qubits {
            Some(n) => {
                if n < group.inner.num_qubits() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "num_qubits ({n}) must be >= group.num_qubits() ({})",
                        group.inner.num_qubits()
                    )));
                }
                RustCode::new(group.inner.clone(), n)
            }
            None => RustCode::from_group(group.inner.clone()),
        };
        Ok(Self { inner: code })
    }

    // ========================================================================
    // Standard code constructors
    // ========================================================================

    /// Create the [[n, 1, n]] bit-flip repetition code.
    ///
    /// Args:
    ///     n: Number of qubits (>= 2)
    #[staticmethod]
    fn repetition(n: usize) -> PyResult<Self> {
        if n < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "repetition code requires at least 2 qubits",
            ));
        }
        Ok(Self {
            inner: RustCode::repetition(n),
        })
    }

    /// Create the [[7, 1, 3]] Steane code.
    #[staticmethod]
    fn steane() -> Self {
        Self {
            inner: RustCode::steane(),
        }
    }

    /// Create the [[5, 1, 3]] perfect code.
    #[staticmethod]
    fn five_qubit() -> Self {
        Self {
            inner: RustCode::five_qubit(),
        }
    }

    /// Create the [[9, 1, 3]] Shor code.
    #[staticmethod]
    fn shor() -> Self {
        Self {
            inner: RustCode::shor(),
        }
    }

    /// Create the [[4, 2, 2]] error-detecting code.
    #[staticmethod]
    fn four_two_two() -> Self {
        Self {
            inner: RustCode::four_two_two(),
        }
    }

    /// Create the toric code on an L x L torus.
    ///
    /// Args:
    ///     l: Lattice dimension (>= 2)
    #[staticmethod]
    fn toric(l: usize) -> PyResult<Self> {
        if l < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "toric code requires L >= 2",
            ));
        }
        Ok(Self {
            inner: RustCode::toric(l),
        })
    }

    // ========================================================================
    // Code parameter methods
    // ========================================================================

    /// Number of physical qubits.
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Number of encoded logical qubits (n - rank).
    fn num_logical_qubits(&self) -> usize {
        self.inner.num_logical_qubits()
    }

    /// Code parameters as "[[n, k]]" string.
    fn code_parameters(&self) -> String {
        self.inner.code_parameters()
    }

    // ========================================================================
    // Analysis methods
    // ========================================================================

    /// Compute the code distance (minimum weight non-trivial logical).
    ///
    /// Returns None if there are no logical qubits.
    /// Only suitable for small codes (k + rank <= 30).
    fn distance(&self) -> Option<usize> {
        self.inner.distance()
    }

    /// Compute the syndrome of an error against the generators.
    ///
    /// Returns a list of booleans where True means the error
    /// anticommutes with that generator.
    fn syndrome(&self, error: &PauliString) -> Vec<bool> {
        self.inner.syndrome(&error.to_rust())
    }

    /// Get a basis for the logical operators of this code.
    ///
    /// Returns a list of PauliStrings that commute with all stabilizers
    /// but are not in the stabilizer group.
    fn logical_operators(&self) -> Vec<PauliString> {
        self.inner
            .logical_operators()
            .into_iter()
            .map(PauliString::from_rust)
            .collect()
    }

    /// Get the underlying stabilizer group.
    fn group(&self) -> PyPauliStabilizerGroup {
        PyPauliStabilizerGroup {
            inner: self.inner.group().clone(),
        }
    }

    // ========================================================================
    // String representations
    // ========================================================================

    fn __str__(&self) -> String {
        format!("{}", self.inner.group())
    }

    fn __repr__(&self) -> String {
        let params = self.inner.code_parameters();
        format!("StabilizerCode({params})")
    }
}

/// Register stabilizer code types with Python module.
pub fn register_stabilizer_code_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStabilizerCode>()?;
    Ok(())
}
