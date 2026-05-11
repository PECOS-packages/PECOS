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

use crate::pauli_bindings::PauliString as PyPauliString;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use pecos_core::{Pauli as RustPauli, PauliBitmaskSmall, QuarterPhase};
use pecos_quantum::{
    ChiMatrix as RustChiMatrix, ChoiMatrix as RustChoiMatrix, KrausOps as RustKrausOps,
    PauliChannel as RustPauliChannel, ProcessTomographyDesign as RustProcessTomographyDesign,
    Ptm as RustPtm, Stinespring as RustStinespring, SuperOp as RustSuperOp,
    average_gate_fidelity as rust_average_gate_fidelity, entropy as rust_entropy,
    gate_error as rust_gate_error, hellinger_distance as rust_hellinger_distance,
    hellinger_fidelity as rust_hellinger_fidelity,
    logarithmic_negativity as rust_logarithmic_negativity, negativity as rust_negativity,
    partial_trace_qubits as rust_partial_trace_qubits,
    partial_trace_subsystems as rust_partial_trace_subsystems, pauli_basis_len,
    process_fidelity as rust_process_fidelity, purity as rust_purity,
    random_density_matrix as rust_random_density_matrix,
    random_quantum_channel as rust_random_quantum_channel, state_fidelity as rust_state_fidelity,
    state_fidelity_with_density_matrix as rust_state_fidelity_with_density_matrix,
};
use pecos_random::PecosRng;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};

type PySchmidtTerm = (f64, Vec<Complex64>, Vec<Complex64>);

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

fn complex_matrices_from_rows(
    matrices: Vec<Vec<Vec<Complex64>>>,
) -> PyResult<Vec<DMatrix<Complex64>>> {
    matrices.into_iter().map(complex_matrix_from_rows).collect()
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

fn complex_matrices_to_rows(matrices: &[DMatrix<Complex64>]) -> Vec<Vec<Vec<Complex64>>> {
    matrices.iter().map(complex_matrix_to_rows).collect()
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

fn parse_pauli_string_key(
    num_qubits: usize,
    pauli_string: &PyPauliString,
) -> PyResult<PauliBitmaskSmall> {
    if pauli_string.inner.get_phase() != QuarterPhase::PlusOne {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "PauliChannel probabilities require unphased PauliString keys",
        ));
    }

    let mut out = PauliBitmaskSmall::identity();
    for (pauli, qubit) in pauli_string.inner.get_paulis() {
        let qubit = qubit.index();
        if qubit >= num_qubits {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "PauliString key acts on qubit {qubit}, outside num_qubits={num_qubits}"
            )));
        }
        let single = match pauli {
            RustPauli::I => PauliBitmaskSmall::identity(),
            RustPauli::X => PauliBitmaskSmall::x(qubit),
            RustPauli::Y => PauliBitmaskSmall::y(qubit),
            RustPauli::Z => PauliBitmaskSmall::z(qubit),
        };
        out = out.multiply(&single);
    }
    Ok(out)
}

fn parse_pauli_probability_key(
    num_qubits: usize,
    key: &Bound<'_, PyAny>,
) -> PyResult<PauliBitmaskSmall> {
    if let Ok(label) = key.extract::<String>() {
        return parse_pauli_label(num_qubits, &label);
    }
    if let Ok(pauli_string) = key.extract::<PyRef<'_, PyPauliString>>() {
        return parse_pauli_string_key(num_qubits, &pauli_string);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "PauliChannel probability keys must be dense Pauli labels or PauliString objects",
    ))
}

fn insert_probability(
    probabilities: &mut BTreeMap<PauliBitmaskSmall, f64>,
    pauli: PauliBitmaskSmall,
    probability: f64,
) -> PyResult<()> {
    if probabilities.insert(pauli, probability).is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "duplicate PauliChannel probability key",
        ));
    }
    Ok(())
}

fn pauli_probabilities_from_py(
    num_qubits: usize,
    probabilities: &Bound<'_, PyAny>,
) -> PyResult<BTreeMap<PauliBitmaskSmall, f64>> {
    let mut out = BTreeMap::new();
    if let Ok(dict) = probabilities.cast::<PyDict>() {
        for (key, value) in dict.iter() {
            let pauli = parse_pauli_probability_key(num_qubits, &key)?;
            insert_probability(&mut out, pauli, value.extract()?)?;
        }
    } else {
        for item in probabilities.try_iter()? {
            let tuple: Bound<'_, PyTuple> = item?.cast_into()?;
            if tuple.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "PauliChannel probability sequences must contain (pauli, probability) pairs",
                ));
            }
            let pauli = parse_pauli_probability_key(num_qubits, &tuple.get_item(0)?)?;
            insert_probability(&mut out, pauli, tuple.get_item(1)?.extract()?)?;
        }
    }
    Ok(out)
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

    fn to_superop(&self) -> PyResult<PySuperOp> {
        Ok(PySuperOp {
            inner: self.inner.to_superop().map_err(py_value_err)?,
        })
    }

    fn to_chi(&self) -> PyResult<PyChiMatrix> {
        Ok(PyChiMatrix {
            inner: self.inner.to_chi().map_err(py_value_err)?,
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

    fn to_superop(&self) -> PyResult<PySuperOp> {
        Ok(PySuperOp {
            inner: self.inner.to_superop().map_err(py_value_err)?,
        })
    }

    fn to_chi(&self) -> PyResult<PyChiMatrix> {
        Ok(PyChiMatrix {
            inner: self.inner.to_chi().map_err(py_value_err)?,
        })
    }

    fn to_stinespring(&self) -> PyResult<PyStinespring> {
        Ok(PyStinespring {
            inner: self.inner.to_stinespring().map_err(py_value_err)?,
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

    #[staticmethod]
    fn from_matrix_unit_outputs(
        num_qubits: usize,
        outputs: Vec<Vec<Vec<Complex64>>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RustChoiMatrix::from_matrix_unit_outputs(
                num_qubits,
                &complex_matrices_from_rows(outputs)?,
            )
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

    fn to_superop(&self) -> PyResult<PySuperOp> {
        Ok(PySuperOp {
            inner: self.inner.to_superop().map_err(py_value_err)?,
        })
    }

    fn to_chi(&self) -> PyResult<PyChiMatrix> {
        Ok(PyChiMatrix {
            inner: self.inner.to_chi().map_err(py_value_err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("ChoiMatrix(num_qubits={})", self.inner.num_qubits())
    }
}

#[pyclass(name = "ProcessTomographyDesign", module = "pecos_rslib.quantum_info")]
pub struct PyProcessTomographyDesign {
    inner: RustProcessTomographyDesign,
}

#[pymethods]
impl PyProcessTomographyDesign {
    #[staticmethod]
    fn matrix_unit(num_qubits: usize) -> PyResult<Self> {
        Ok(Self {
            inner: RustProcessTomographyDesign::matrix_unit(num_qubits).map_err(py_value_err)?,
        })
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn num_inputs(&self) -> usize {
        self.inner.num_inputs()
    }

    fn input_index(&self, row: usize, col: usize) -> PyResult<usize> {
        self.inner.input_index(row, col).map_err(py_value_err)
    }

    fn input_metadata(&self, index: usize) -> PyResult<(usize, usize, usize)> {
        let input = self.inner.input_metadata(index).map_err(py_value_err)?;
        Ok((input.index, input.row, input.col))
    }

    fn input_metadata_all(&self) -> Vec<(usize, usize, usize)> {
        self.inner
            .input_metadata_all()
            .into_iter()
            .map(|input| (input.index, input.row, input.col))
            .collect()
    }

    fn input_operator(&self, index: usize) -> PyResult<Vec<Vec<Complex64>>> {
        Ok(complex_matrix_to_rows(
            &self.inner.input_operator(index).map_err(py_value_err)?,
        ))
    }

    fn input_operators(&self) -> Vec<Vec<Vec<Complex64>>> {
        complex_matrices_to_rows(&self.inner.input_operators())
    }

    fn simulate_outputs(&self, channel: &PyChoiMatrix) -> PyResult<Vec<Vec<Vec<Complex64>>>> {
        Ok(complex_matrices_to_rows(
            &self
                .inner
                .simulate_outputs(&channel.inner)
                .map_err(py_value_err)?,
        ))
    }

    fn reconstruct_choi(&self, outputs: Vec<Vec<Vec<Complex64>>>) -> PyResult<PyChoiMatrix> {
        Ok(PyChoiMatrix {
            inner: self
                .inner
                .reconstruct_choi(&complex_matrices_from_rows(outputs)?)
                .map_err(py_value_err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ProcessTomographyDesign(num_qubits={}, num_inputs={})",
            self.inner.num_qubits(),
            self.inner.num_inputs()
        )
    }
}

#[pyclass(name = "SuperOp", module = "pecos_rslib.quantum_info")]
pub struct PySuperOp {
    inner: RustSuperOp,
}

#[pymethods]
impl PySuperOp {
    #[new]
    fn new(num_qubits: usize, matrix: Vec<Vec<Complex64>>) -> PyResult<Self> {
        Ok(Self {
            inner: RustSuperOp::try_new(num_qubits, complex_matrix_from_rows(matrix)?)
                .map_err(py_value_err)?,
        })
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn matrix(&self) -> Vec<Vec<Complex64>> {
        complex_matrix_to_rows(self.inner.matrix())
    }

    fn to_choi(&self) -> PyResult<PyChoiMatrix> {
        Ok(PyChoiMatrix {
            inner: self.inner.to_choi().map_err(py_value_err)?,
        })
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

    fn compose(&self, other: &PySuperOp) -> PyResult<PySuperOp> {
        Ok(PySuperOp {
            inner: self.inner.compose(&other.inner).map_err(py_value_err)?,
        })
    }

    fn tensor(&self, other: &PySuperOp) -> PyResult<PySuperOp> {
        Ok(PySuperOp {
            inner: self.inner.tensor(&other.inner).map_err(py_value_err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("SuperOp(num_qubits={})", self.inner.num_qubits())
    }
}

#[pyclass(name = "ChiMatrix", module = "pecos_rslib.quantum_info")]
pub struct PyChiMatrix {
    inner: RustChiMatrix,
}

#[pymethods]
impl PyChiMatrix {
    #[new]
    fn new(num_qubits: usize, matrix: Vec<Vec<Complex64>>) -> PyResult<Self> {
        Ok(Self {
            inner: RustChiMatrix::try_new(num_qubits, complex_matrix_from_rows(matrix)?)
                .map_err(py_value_err)?,
        })
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn matrix(&self) -> Vec<Vec<Complex64>> {
        complex_matrix_to_rows(self.inner.matrix())
    }

    fn to_choi(&self) -> PyResult<PyChoiMatrix> {
        Ok(PyChoiMatrix {
            inner: self.inner.to_choi().map_err(py_value_err)?,
        })
    }

    fn to_ptm(&self) -> PyResult<PyPtm> {
        Ok(PyPtm {
            inner: self.inner.to_ptm().map_err(py_value_err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("ChiMatrix(num_qubits={})", self.inner.num_qubits())
    }
}

#[pyclass(name = "Stinespring", module = "pecos_rslib.quantum_info")]
pub struct PyStinespring {
    inner: RustStinespring,
}

#[pymethods]
impl PyStinespring {
    #[new]
    fn new(num_qubits: usize, isometry: Vec<Vec<Complex64>>) -> PyResult<Self> {
        Ok(Self {
            inner: RustStinespring::try_new(num_qubits, complex_matrix_from_rows(isometry)?)
                .map_err(py_value_err)?,
        })
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn environment_dim(&self) -> usize {
        self.inner.environment_dim()
    }

    fn isometry(&self) -> Vec<Vec<Complex64>> {
        complex_matrix_to_rows(self.inner.isometry())
    }

    fn to_kraus(&self) -> PyResult<PyKrausOps> {
        Ok(PyKrausOps {
            inner: self.inner.to_kraus().map_err(py_value_err)?,
        })
    }

    fn to_choi(&self) -> PyResult<PyChoiMatrix> {
        Ok(PyChoiMatrix {
            inner: self.inner.to_choi().map_err(py_value_err)?,
        })
    }

    fn to_superop(&self) -> PyResult<PySuperOp> {
        Ok(PySuperOp {
            inner: self.inner.to_superop().map_err(py_value_err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Stinespring(num_qubits={}, environment_dim={})",
            self.inner.num_qubits(),
            self.inner.environment_dim()
        )
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
fn entropy(rho: Vec<Vec<Complex64>>) -> PyResult<f64> {
    rust_entropy(&complex_matrix_from_rows(rho)?).map_err(py_value_err)
}

#[pyfunction]
fn shannon_entropy(probabilities: Vec<f64>, base: f64) -> PyResult<f64> {
    pecos_quantum::shannon_entropy(&probabilities, base).map_err(py_value_err)
}

#[pyfunction]
fn negativity(rho: Vec<Vec<Complex64>>, dims: Vec<usize>, subsystem: usize) -> PyResult<f64> {
    rust_negativity(&complex_matrix_from_rows(rho)?, &dims, subsystem).map_err(py_value_err)
}

#[pyfunction]
fn logarithmic_negativity(
    rho: Vec<Vec<Complex64>>,
    dims: Vec<usize>,
    subsystem: usize,
) -> PyResult<f64> {
    rust_logarithmic_negativity(&complex_matrix_from_rows(rho)?, &dims, subsystem)
        .map_err(py_value_err)
}

#[pyfunction]
fn schmidt_decomposition(
    state: Vec<Complex64>,
    dims: Vec<usize>,
    left_subsystems: Vec<usize>,
) -> PyResult<Vec<PySchmidtTerm>> {
    pecos_quantum::schmidt_decomposition(&DVector::from_vec(state), &dims, &left_subsystems)
        .map_err(py_value_err)
}

#[pyfunction]
fn partial_trace_subsystems(
    rho: Vec<Vec<Complex64>>,
    dims: Vec<usize>,
    traced_subsystems: Vec<usize>,
) -> PyResult<Vec<Vec<Complex64>>> {
    Ok(complex_matrix_to_rows(
        &rust_partial_trace_subsystems(&complex_matrix_from_rows(rho)?, &dims, &traced_subsystems)
            .map_err(py_value_err)?,
    ))
}

#[pyfunction]
fn partial_trace_qubits(
    rho: Vec<Vec<Complex64>>,
    num_qubits: usize,
    traced_qubits: Vec<usize>,
) -> PyResult<Vec<Vec<Complex64>>> {
    Ok(complex_matrix_to_rows(
        &rust_partial_trace_qubits(&complex_matrix_from_rows(rho)?, num_qubits, &traced_qubits)
            .map_err(py_value_err)?,
    ))
}

#[pyfunction]
fn hellinger_distance(left: Vec<f64>, right: Vec<f64>) -> PyResult<f64> {
    rust_hellinger_distance(&left, &right).map_err(py_value_err)
}

#[pyfunction]
fn hellinger_fidelity(left: Vec<f64>, right: Vec<f64>) -> PyResult<f64> {
    rust_hellinger_fidelity(&left, &right).map_err(py_value_err)
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
fn matrix_unit_basis(num_qubits: usize) -> PyResult<Vec<Vec<Vec<Complex64>>>> {
    Ok(complex_matrices_to_rows(
        &pecos_quantum::matrix_unit_basis(num_qubits).map_err(py_value_err)?,
    ))
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
    parent.add_class::<PyProcessTomographyDesign>()?;
    parent.add_class::<PySuperOp>()?;
    parent.add_class::<PyChiMatrix>()?;
    parent.add_class::<PyStinespring>()?;

    parent.add_function(wrap_pyfunction!(state_fidelity, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        state_fidelity_with_density_matrix,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(purity, parent)?)?;
    parent.add_function(wrap_pyfunction!(entropy, parent)?)?;
    parent.add_function(wrap_pyfunction!(shannon_entropy, parent)?)?;
    parent.add_function(wrap_pyfunction!(negativity, parent)?)?;
    parent.add_function(wrap_pyfunction!(logarithmic_negativity, parent)?)?;
    parent.add_function(wrap_pyfunction!(schmidt_decomposition, parent)?)?;
    parent.add_function(wrap_pyfunction!(partial_trace_subsystems, parent)?)?;
    parent.add_function(wrap_pyfunction!(partial_trace_qubits, parent)?)?;
    parent.add_function(wrap_pyfunction!(hellinger_distance, parent)?)?;
    parent.add_function(wrap_pyfunction!(hellinger_fidelity, parent)?)?;
    parent.add_function(wrap_pyfunction!(process_fidelity, parent)?)?;
    parent.add_function(wrap_pyfunction!(average_gate_fidelity, parent)?)?;
    parent.add_function(wrap_pyfunction!(gate_error, parent)?)?;
    parent.add_function(wrap_pyfunction!(pauli_channel_diamond_norm, parent)?)?;
    parent.add_function(wrap_pyfunction!(pauli_channel_diamond_distance, parent)?)?;
    parent.add_function(wrap_pyfunction!(matrix_unit_basis, parent)?)?;
    parent.add_function(wrap_pyfunction!(random_density_matrix, parent)?)?;
    parent.add_function(wrap_pyfunction!(random_quantum_channel, parent)?)?;

    let py = parent.py();
    let module = PyModule::new(py, "quantum_info")?;
    for name in [
        "PauliChannel",
        "Ptm",
        "KrausOps",
        "ChoiMatrix",
        "ProcessTomographyDesign",
        "SuperOp",
        "ChiMatrix",
        "Stinespring",
        "state_fidelity",
        "state_fidelity_with_density_matrix",
        "purity",
        "entropy",
        "shannon_entropy",
        "negativity",
        "logarithmic_negativity",
        "schmidt_decomposition",
        "partial_trace_subsystems",
        "partial_trace_qubits",
        "hellinger_distance",
        "hellinger_fidelity",
        "process_fidelity",
        "average_gate_fidelity",
        "gate_error",
        "pauli_channel_diamond_norm",
        "pauli_channel_diamond_distance",
        "matrix_unit_basis",
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
