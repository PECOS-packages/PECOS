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

//! Python bindings for binary QEC code matrices.

use pecos_qec::ParityCheckMatrix as RustParityCheckMatrix;
use pecos_quantum::SymplecticMatrix as RustSymplecticMatrix;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::pauli_bindings::PauliString;

fn validated_binary_rows(rows: Vec<Vec<i64>>, name: &str) -> PyResult<Vec<Vec<u8>>> {
    rows.into_iter()
        .enumerate()
        .map(|(row_index, row)| {
            row.into_iter()
                .map(|value| match value {
                    0 | 1 => Ok(u8::from(value == 1)),
                    _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{name} row {row_index} contains invalid value {value}; expected 0 or 1"
                    ))),
                })
                .collect()
        })
        .collect()
}

fn python_binary_rows(rows: Vec<Vec<u8>>) -> Vec<Vec<usize>> {
    rows.into_iter()
        .map(|row| row.into_iter().map(usize::from).collect())
        .collect()
}

/// A role-neutral binary parity-check matrix.
#[pyclass(name = "ParityCheckMatrix", module = "pecos_rslib", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyParityCheckMatrix {
    pub(crate) inner: RustParityCheckMatrix,
}

#[pymethods]
impl PyParityCheckMatrix {
    #[new]
    fn new(rows: Vec<Vec<i64>>) -> PyResult<Self> {
        Self::from_rows(rows)
    }

    #[classmethod]
    fn from_dense(_cls: &Bound<'_, PyType>, rows: Vec<Vec<i64>>) -> PyResult<Self> {
        Self::from_rows(rows)
    }

    #[classmethod]
    fn zeros(_cls: &Bound<'_, PyType>, num_checks: usize, num_qubits: usize) -> Self {
        Self {
            inner: RustParityCheckMatrix::zeros(num_checks, num_qubits),
        }
    }

    fn num_checks(&self) -> usize {
        self.inner.num_checks()
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn rank(&self) -> usize {
        self.inner.rank()
    }

    fn rows(&self) -> Vec<Vec<usize>> {
        python_binary_rows(self.inner.rows())
    }

    fn to_x_stabilizers(&self) -> Vec<PauliString> {
        self.inner
            .to_x_stabilizers()
            .into_iter()
            .map(PauliString::from_rust)
            .collect()
    }

    fn to_z_stabilizers(&self) -> Vec<PauliString> {
        self.inner
            .to_z_stabilizers()
            .into_iter()
            .map(PauliString::from_rust)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ParityCheckMatrix(shape=({}, {}))",
            self.inner.num_checks(),
            self.inner.num_qubits()
        )
    }
}

impl PyParityCheckMatrix {
    fn from_rows(rows: Vec<Vec<i64>>) -> PyResult<Self> {
        let rows = validated_binary_rows(rows, "parity-check matrix")?;
        let inner = RustParityCheckMatrix::from_dense(rows)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }
}

/// A binary symplectic matrix whose rows represent Pauli operators.
#[pyclass(name = "SymplecticMatrix", module = "pecos_rslib", from_py_object)]
#[derive(Clone, Debug)]
pub struct PySymplecticMatrix {
    pub(crate) inner: RustSymplecticMatrix,
}

#[pymethods]
impl PySymplecticMatrix {
    #[new]
    fn new(rows: Vec<Vec<i64>>) -> PyResult<Self> {
        Self::from_rows(rows)
    }

    #[classmethod]
    fn from_dense(_cls: &Bound<'_, PyType>, rows: Vec<Vec<i64>>) -> PyResult<Self> {
        Self::from_rows(rows)
    }

    #[classmethod]
    fn zeros(_cls: &Bound<'_, PyType>, num_rows: usize, num_qubits: usize) -> Self {
        Self {
            inner: RustSymplecticMatrix::zeros(num_rows, num_qubits),
        }
    }

    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn rank(&self) -> usize {
        self.inner.rank()
    }

    fn rows(&self) -> Vec<Vec<usize>> {
        python_binary_rows(self.inner.rows())
    }

    fn x_block(&self) -> Vec<Vec<usize>> {
        python_binary_rows(self.inner.x_block().rows())
    }

    fn z_block(&self) -> Vec<Vec<usize>> {
        python_binary_rows(self.inner.z_block().rows())
    }

    fn to_positive_paulis(&self) -> Vec<PauliString> {
        self.inner
            .to_positive_paulis()
            .into_iter()
            .map(PauliString::from_rust)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "SymplecticMatrix(shape=({}, {}))",
            self.inner.num_rows(),
            self.inner.num_qubits()
        )
    }
}

impl PySymplecticMatrix {
    fn from_rows(rows: Vec<Vec<i64>>) -> PyResult<Self> {
        let rows = validated_binary_rows(rows, "symplectic matrix")?;
        let inner = RustSymplecticMatrix::from_dense(rows)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }
}

pub fn register_code_matrix_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyParityCheckMatrix>()?;
    m.add_class::<PySymplecticMatrix>()?;
    Ok(())
}
