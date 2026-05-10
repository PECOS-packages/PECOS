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

//! Thin Python bindings for PECOS quantum-information primitives.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use pecos_core::PauliBitmaskSmall;
use pecos_quantum::{
    ChoiMatrix as RustChoiMatrix, KrausOps as RustKrausOps, PauliChannel as RustPauliChannel,
    Ptm as RustPtm, average_gate_fidelity as rust_average_gate_fidelity,
    gate_error as rust_gate_error, pauli_basis_len, process_fidelity as rust_process_fidelity,
    purity as rust_purity, random_density_matrix as rust_random_density_matrix,
    random_quantum_channel as rust_random_quantum_channel, state_fidelity as rust_state_fidelity,
    state_fidelity_with_density_matrix as rust_state_fidelity_with_density_matrix,
};
use pecos_random::PecosRng;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

fn py_value_err(err: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

fn real_matrix_from_rows(rows: Vec<Vec<f64>>) -> PyResult<DMatrix<f64>> {
    let row_count = rows.len();
    let col_count = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != col_count) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "matrix rows must all have the same length",
        ));
    }
    let data: Vec<f64> = rows.into_iter().flatten().collect();
    Ok(DMatrix::from_row_slice(row_count, col_count, &data))
}

fn complex_matrix_from_rows(rows: Vec<Vec<Complex64>>) -> PyResult<DMatrix<Complex64>> {
    let row_count = rows.len();
    let col_count = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != col_count) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "matrix rows must all have the same length",
        ));
    }
    let data: Vec<Complex64> = rows.into_iter().flatten().collect();
    Ok(DMatrix::from_row_slice(row_count, col_count, &data))
}

fn real_matrix_to_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
        .collect()
}

fn complex_matrix_to_rows(matrix: &DMatrix<Complex64>) -> Vec<Vec<Complex64>> {
    (0..matrix.nrows())
        .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
        .collect()
}

fn parse_pauli_label(num_qubits: usize, label: &str) -> PyResult<PauliBitmaskSmall> {
    let label = label.trim();
    if label.len() != num_qubits {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Pauli label '{label}' has length {}, expected {num_qubits}",
            label.len()
        )));
    }
    let mut index = 0usize;
    for (qubit, ch) in label.chars().rev().enumerate() {
        let digit = match ch {
            'I' => 0,
            'X' => 1,
            'Y' => 2,
            'Z' => 3,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid Pauli label '{label}'; expected only I, X, Y, Z"
                )));
            }
        };
        index |= digit << (2 * qubit);
    }
    pecos_quantum::basis_bitmask(num_qubits, index).map_err(py_value_err)
}

fn pauli_probabilities_from_py(
    num_qubits: usize,
    probabilities: &Bound<'_, PyAny>,
) -> PyResult<BTreeMap<PauliBitmaskSmall, f64>> {
    let items: Vec<(String, f64)> = if let Ok(dict) = probabilities.cast::<PyDict>() {
        dict.iter()
            .map(|(key, value)| Ok((key.extract()?, value.extract()?)))
            .collect::<PyResult<_>>()?
    } else {
        probabilities.extract()?
    };
    items
        .into_iter()
        .map(|(label, probability)| Ok((parse_pauli_label(num_qubits, &label)?, probability)))
        .collect()
}

#[pyclass(name = "PauliChannel", module = "pecos_rslib.quantum_info")]
pub struct PyPauliChannel {
    inner: RustPauliChannel,
}

#[pymethods]
impl PyPauliChannel {
    #[staticmethod]
    fn one_qubit(px: f64, py: f64, pz: f64) -> PyResult<Self> {
        Ok(Self {
            inner: RustPauliChannel::one_qubit(px, py, pz).map_err(py_value_err)?,
        })
    }

    #[staticmethod]
    fn from_probabilities(num_qubits: usize, probabilities: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RustPauliChannel::try_new(
                num_qubits,
                pauli_probabilities_from_py(num_qubits, probabilities)?,
            )
            .map_err(py_value_err)?,
        })
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn probabilities(&self) -> PyResult<BTreeMap<String, f64>> {
        let mut out = BTreeMap::new();
        let basis_len = pauli_basis_len(self.inner.num_qubits()).map_err(py_value_err)?;
        for basis_index in 0..basis_len {
            let pauli = pecos_quantum::basis_bitmask(self.inner.num_qubits(), basis_index)
                .map_err(py_value_err)?;
            let probability = self.inner.probability(&pauli);
            if probability > 0.0 {
                out.insert(
                    pecos_quantum::basis_label(self.inner.num_qubits(), basis_index)
                        .map_err(py_value_err)?,
                    probability,
                );
            }
        }
        Ok(out)
    }

    fn total_error_rate(&self) -> f64 {
        self.inner.total_error_rate()
    }

    fn to_ptm(&self) -> PyResult<PyPtm> {
        Ok(PyPtm {
            inner: self.inner.to_ptm().map_err(py_value_err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("PauliChannel(num_qubits={})", self.inner.num_qubits())
    }
}

#[pyclass(name = "Ptm", module = "pecos_rslib.quantum_info")]
pub struct PyPtm {
    inner: RustPtm,
}

#[pymethods]
impl PyPtm {
    #[new]
    fn new(num_qubits: usize, matrix: Vec<Vec<f64>>) -> PyResult<Self> {
        Ok(Self {
            inner: RustPtm::try_new(num_qubits, real_matrix_from_rows(matrix)?)
                .map_err(py_value_err)?,
        })
    }

    #[staticmethod]
    fn identity(num_qubits: usize) -> PyResult<Self> {
        Ok(Self {
            inner: RustPtm::identity(num_qubits).map_err(py_value_err)?,
        })
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn matrix(&self) -> Vec<Vec<f64>> {
        real_matrix_to_rows(self.inner.matrix())
    }

    fn entry(&self, output: usize, input: usize) -> f64 {
        self.inner.entry(output, input)
    }

    fn to_choi(&self) -> PyResult<PyChoiMatrix> {
        Ok(PyChoiMatrix {
            inner: self.inner.to_choi().map_err(py_value_err)?,
        })
    }

    fn to_kraus(&self) -> PyResult<PyKrausOps> {
        Ok(PyKrausOps {
            inner: self.inner.to_kraus().map_err(py_value_err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("Ptm(num_qubits={})", self.inner.num_qubits())
    }
}

#[pyclass(name = "KrausOps", module = "pecos_rslib.quantum_info")]
pub struct PyKrausOps {
    inner: RustKrausOps,
}

#[pymethods]
impl PyKrausOps {
    #[new]
    fn new(num_qubits: usize, operators: Vec<Vec<Vec<Complex64>>>) -> PyResult<Self> {
        let operators = operators
            .into_iter()
            .map(complex_matrix_from_rows)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: RustKrausOps::try_new(num_qubits, operators).map_err(py_value_err)?,
        })
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn operators(&self) -> Vec<Vec<Vec<Complex64>>> {
        self.inner
            .operators()
            .iter()
            .map(complex_matrix_to_rows)
            .collect()
    }

    fn is_trace_preserving(&self) -> bool {
        self.inner.is_trace_preserving()
    }

    fn to_ptm(&self) -> PyResult<PyPtm> {
        Ok(PyPtm {
            inner: self.inner.to_ptm().map_err(py_value_err)?,
        })
    }

    fn to_choi(&self) -> PyResult<PyChoiMatrix> {
        Ok(PyChoiMatrix {
            inner: self.inner.to_choi().map_err(py_value_err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("KrausOps(num_qubits={})", self.inner.num_qubits())
    }
}

#[pyclass(name = "ChoiMatrix", module = "pecos_rslib.quantum_info")]
pub struct PyChoiMatrix {
    inner: RustChoiMatrix,
}

#[pymethods]
impl PyChoiMatrix {
    #[new]
    fn new(num_qubits: usize, matrix: Vec<Vec<Complex64>>) -> PyResult<Self> {
        Ok(Self {
            inner: RustChoiMatrix::try_new(num_qubits, complex_matrix_from_rows(matrix)?)
                .map_err(py_value_err)?,
        })
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn matrix(&self) -> Vec<Vec<Complex64>> {
        complex_matrix_to_rows(self.inner.matrix())
    }

    fn apply_to_operator(&self, operator: Vec<Vec<Complex64>>) -> PyResult<Vec<Vec<Complex64>>> {
        Ok(complex_matrix_to_rows(
            &self
                .inner
                .apply_to_operator(&complex_matrix_from_rows(operator)?)
                .map_err(py_value_err)?,
        ))
    }

    fn partial_trace_output(&self) -> PyResult<Vec<Vec<Complex64>>> {
        Ok(complex_matrix_to_rows(
            &self.inner.partial_trace_output().map_err(py_value_err)?,
        ))
    }

    fn partial_trace_input(&self) -> PyResult<Vec<Vec<Complex64>>> {
        Ok(complex_matrix_to_rows(
            &self.inner.partial_trace_input().map_err(py_value_err)?,
        ))
    }

    fn is_completely_positive(&self) -> bool {
        self.inner.is_completely_positive()
    }

    fn is_trace_preserving(&self) -> bool {
        self.inner.is_trace_preserving()
    }

    fn is_cptp(&self) -> bool {
        self.inner.is_cptp()
    }

    fn is_unital(&self) -> bool {
        self.inner.is_unital()
    }

    fn to_ptm(&self) -> PyResult<PyPtm> {
        Ok(PyPtm {
            inner: self.inner.to_ptm().map_err(py_value_err)?,
        })
    }

    fn to_kraus(&self) -> PyResult<PyKrausOps> {
        Ok(PyKrausOps {
            inner: self.inner.to_kraus().map_err(py_value_err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("ChoiMatrix(num_qubits={})", self.inner.num_qubits())
    }
}

#[pyfunction]
fn state_fidelity(left: Vec<Complex64>, right: Vec<Complex64>) -> PyResult<f64> {
    rust_state_fidelity(&DVector::from_vec(left), &DVector::from_vec(right)).map_err(py_value_err)
}

#[pyfunction]
fn state_fidelity_with_density_matrix(
    rho: Vec<Vec<Complex64>>,
    psi: Vec<Complex64>,
) -> PyResult<f64> {
    rust_state_fidelity_with_density_matrix(
        &complex_matrix_from_rows(rho)?,
        &DVector::from_vec(psi),
    )
    .map_err(py_value_err)
}

#[pyfunction]
fn purity(rho: Vec<Vec<Complex64>>) -> PyResult<f64> {
    rust_purity(&complex_matrix_from_rows(rho)?).map_err(py_value_err)
}

#[pyfunction]
fn process_fidelity(left: &PyPtm, right: &PyPtm) -> PyResult<f64> {
    rust_process_fidelity(&left.inner, &right.inner).map_err(py_value_err)
}

#[pyfunction]
fn average_gate_fidelity(left: &PyPtm, right: &PyPtm) -> PyResult<f64> {
    rust_average_gate_fidelity(&left.inner, &right.inner).map_err(py_value_err)
}

#[pyfunction]
fn gate_error(left: &PyPtm, right: &PyPtm) -> PyResult<f64> {
    rust_gate_error(&left.inner, &right.inner).map_err(py_value_err)
}

#[pyfunction]
fn pauli_channel_diamond_norm(left: &PyPauliChannel, right: &PyPauliChannel) -> PyResult<f64> {
    pecos_quantum::pauli_channel_diamond_norm(&left.inner, &right.inner).map_err(py_value_err)
}

#[pyfunction]
fn pauli_channel_diamond_distance(left: &PyPauliChannel, right: &PyPauliChannel) -> PyResult<f64> {
    pecos_quantum::pauli_channel_diamond_distance(&left.inner, &right.inner).map_err(py_value_err)
}

#[pyfunction]
fn random_density_matrix(num_qubits: usize, seed: u64) -> PyResult<Vec<Vec<Complex64>>> {
    let mut rng = PecosRng::seed_from_u64(seed);
    Ok(complex_matrix_to_rows(
        &rust_random_density_matrix(&mut rng, num_qubits).map_err(py_value_err)?,
    ))
}

#[pyfunction]
fn random_quantum_channel(num_qubits: usize, num_kraus: usize, seed: u64) -> PyResult<PyKrausOps> {
    let mut rng = PecosRng::seed_from_u64(seed);
    Ok(PyKrausOps {
        inner: rust_random_quantum_channel(&mut rng, num_qubits, num_kraus)
            .map_err(py_value_err)?,
    })
}

pub fn register_quantum_info_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyPauliChannel>()?;
    parent.add_class::<PyPtm>()?;
    parent.add_class::<PyKrausOps>()?;
    parent.add_class::<PyChoiMatrix>()?;

    parent.add_function(wrap_pyfunction!(state_fidelity, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        state_fidelity_with_density_matrix,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(purity, parent)?)?;
    parent.add_function(wrap_pyfunction!(process_fidelity, parent)?)?;
    parent.add_function(wrap_pyfunction!(average_gate_fidelity, parent)?)?;
    parent.add_function(wrap_pyfunction!(gate_error, parent)?)?;
    parent.add_function(wrap_pyfunction!(pauli_channel_diamond_norm, parent)?)?;
    parent.add_function(wrap_pyfunction!(pauli_channel_diamond_distance, parent)?)?;
    parent.add_function(wrap_pyfunction!(random_density_matrix, parent)?)?;
    parent.add_function(wrap_pyfunction!(random_quantum_channel, parent)?)?;

    let py = parent.py();
    let module = PyModule::new(py, "quantum_info")?;
    for name in [
        "PauliChannel",
        "Ptm",
        "KrausOps",
        "ChoiMatrix",
        "state_fidelity",
        "state_fidelity_with_density_matrix",
        "purity",
        "process_fidelity",
        "average_gate_fidelity",
        "gate_error",
        "pauli_channel_diamond_norm",
        "pauli_channel_diamond_distance",
        "random_density_matrix",
        "random_quantum_channel",
    ] {
        module.add(name, parent.getattr(name)?)?;
    }

    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.quantum_info", &module)?;
    parent.add_submodule(&module)?;
    Ok(())
}
