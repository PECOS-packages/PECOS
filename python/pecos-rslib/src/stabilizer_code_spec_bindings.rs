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

//! Python bindings for stabilizer-code specifications and distance search.

use pecos_qec::{
    DistanceResult as RustDistanceResult, DistanceSearchConfig,
    LogicalOperatorInfo as RustLogicalOperatorInfo, StabilizerCodeSpec as RustCodeSpec,
    StabilizerCodeSpecBuilder as RustCodeSpecBuilder, calculate_distance, find_shortest_logicals,
};
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::code_matrix_bindings::{PyParityCheckMatrix, PySymplecticMatrix};
use crate::pauli_bindings::PauliString;
use crate::stabilizer_code_bindings::PyStabilizerCode;

/// Result of a stabilizer-code distance search.
#[pyclass(name = "DistanceResult", module = "pecos_rslib", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyDistanceResult {
    inner: RustDistanceResult,
}

#[pymethods]
impl PyDistanceResult {
    /// The code distance.
    #[getter]
    fn distance(&self) -> usize {
        self.inner.distance
    }

    /// A logical operator achieving the code distance.
    #[getter]
    fn min_weight_operator(&self) -> PauliString {
        PauliString::from_rust(self.inner.min_weight_operator.clone())
    }

    fn __repr__(&self) -> String {
        let operator = PauliString::from_rust(self.inner.min_weight_operator.clone());
        format!(
            "DistanceResult(distance={}, min_weight_operator={})",
            self.inner.distance,
            operator.__str__()
        )
    }
}

impl From<RustDistanceResult> for PyDistanceResult {
    fn from(inner: RustDistanceResult) -> Self {
        Self { inner }
    }
}

/// A minimum-weight logical operator and its logical equivalence information.
#[pyclass(
    name = "LogicalOperatorInfo",
    module = "pecos_rslib",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyLogicalOperatorInfo {
    inner: RustLogicalOperatorInfo,
}

#[pymethods]
impl PyLogicalOperatorInfo {
    /// The physical Pauli operator.
    #[getter]
    fn operator(&self) -> PauliString {
        PauliString::from_rust(self.inner.operator.clone())
    }

    /// The physical weight of the operator.
    #[getter]
    fn weight(&self) -> usize {
        self.inner.weight
    }

    /// Logical operations implemented by the operator.
    #[getter]
    fn equivalent_logicals(&self) -> Vec<(String, usize)> {
        self.inner
            .equivalent_logicals
            .iter()
            .map(|(logical_type, index)| (logical_type.to_string(), *index))
            .collect()
    }

    /// Return the logical equivalence as a compact string such as ``X0*Z1``.
    fn equivalence_string(&self) -> String {
        self.inner.equivalence_string()
    }

    fn __repr__(&self) -> String {
        let operator = PauliString::from_rust(self.inner.operator.clone());
        format!(
            "LogicalOperatorInfo(operator={}, weight={}, equivalence={})",
            operator.__str__(),
            self.inner.weight,
            self.inner.equivalence_string()
        )
    }
}

impl From<RustLogicalOperatorInfo> for PyLogicalOperatorInfo {
    fn from(inner: RustLogicalOperatorInfo) -> Self {
        Self { inner }
    }
}

/// Builder for a stabilizer-code specification.
#[pyclass(
    name = "StabilizerCodeSpecBuilder",
    module = "pecos_rslib",
    skip_from_py_object
)]
pub struct PyStabilizerCodeSpecBuilder {
    inner: Option<RustCodeSpecBuilder>,
}

impl PyStabilizerCodeSpecBuilder {
    fn new(num_qubits: usize) -> Self {
        Self {
            inner: Some(RustCodeSpecBuilder::new(num_qubits)),
        }
    }

    fn take_inner(&mut self) -> PyResult<RustCodeSpecBuilder> {
        self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "StabilizerCodeSpecBuilder has already been consumed",
            )
        })
    }
}

#[pymethods]
impl PyStabilizerCodeSpecBuilder {
    /// Add a stabilizer generator.
    fn check(&mut self, op: &PauliString) -> PyResult<()> {
        let builder = self.take_inner()?;
        self.inner = Some(builder.check(op.to_rust()));
        Ok(())
    }

    /// Add X-type and Z-type stabilizers from CSS parity-check matrices.
    fn checks_from_css(
        &mut self,
        x_stabilizers: &PyParityCheckMatrix,
        z_stabilizers: &PyParityCheckMatrix,
    ) -> PyResult<()> {
        let builder = self.take_inner()?;
        self.inner = Some(
            builder
                .checks_from_css(&x_stabilizers.inner, &z_stabilizers.inner)
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
        );
        Ok(())
    }

    /// Add stabilizers from the rows of a symplectic matrix.
    fn checks_from_symplectic(&mut self, matrix: &PySymplecticMatrix) -> PyResult<()> {
        let builder = self.take_inner()?;
        self.inner = Some(
            builder
                .checks_from_symplectic(&matrix.inner)
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
        );
        Ok(())
    }

    /// Add a logical Z operator.
    fn logical_z(&mut self, op: &PauliString) -> PyResult<()> {
        let builder = self.take_inner()?;
        self.inner = Some(builder.logical_z(op.to_rust()));
        Ok(())
    }

    /// Add a logical X operator.
    fn logical_x(&mut self, op: &PauliString) -> PyResult<()> {
        let builder = self.take_inner()?;
        self.inner = Some(builder.logical_x(op.to_rust()));
        Ok(())
    }

    /// Build with count validation only.
    fn build(&mut self) -> PyResult<PyStabilizerCodeSpec> {
        let inner = self
            .take_inner()?
            .build()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(PyStabilizerCodeSpec { inner })
    }

    /// Build and fully verify all commutation relations.
    fn build_verified(&mut self) -> PyResult<PyStabilizerCodeSpec> {
        let inner = self
            .take_inner()?
            .build_verified()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(PyStabilizerCodeSpec { inner })
    }

    /// Build and automatically discover paired logical operators.
    fn build_with_discovered_logicals(&mut self) -> PyResult<PyStabilizerCodeSpec> {
        let inner = self
            .take_inner()?
            .build_with_discovered_logicals()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(PyStabilizerCodeSpec { inner })
    }
}

/// A complete stabilizer-code specification with paired logical operators.
#[pyclass(name = "StabilizerCodeSpec", module = "pecos_rslib", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStabilizerCodeSpec {
    pub(crate) inner: RustCodeSpec,
}

#[pymethods]
impl PyStabilizerCodeSpec {
    /// Create a builder for a code with the specified number of qubits.
    #[staticmethod]
    fn builder(num_qubits: usize) -> PyStabilizerCodeSpecBuilder {
        PyStabilizerCodeSpecBuilder::new(num_qubits)
    }

    /// Create a stabilizer-code specification.
    #[new]
    fn new(
        num_qubits: usize,
        stabilizers: Vec<PauliString>,
        logical_zs: Vec<PauliString>,
        logical_xs: Vec<PauliString>,
    ) -> PyResult<Self> {
        let stabilizers = stabilizers
            .into_iter()
            .map(|pauli| pauli.to_rust())
            .collect();
        let logical_zs = logical_zs
            .into_iter()
            .map(|pauli| pauli.to_rust())
            .collect();
        let logical_xs = logical_xs
            .into_iter()
            .map(|pauli| pauli.to_rust())
            .collect();
        let inner = RustCodeSpec::new(num_qubits, stabilizers, logical_zs, logical_xs)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    /// Create a full specification from a ``StabilizerCode``.
    #[classmethod]
    fn from_stabilizer_code(_cls: &Bound<'_, PyType>, code: &PyStabilizerCode) -> PyResult<Self> {
        let inner = RustCodeSpec::from_stabilizer_code(&code.inner)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of physical qubits.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Number of encoded logical qubits.
    #[getter]
    fn num_logical_qubits(&self) -> usize {
        self.inner.num_logical_qubits()
    }

    /// Stabilizer generators.
    #[getter]
    fn stabilizers(&self) -> Vec<PauliString> {
        self.inner
            .stabilizers()
            .iter()
            .cloned()
            .map(PauliString::from_rust)
            .collect()
    }

    /// Destabilizer generators.
    #[getter]
    fn destabilizers(&self) -> Vec<PauliString> {
        self.inner
            .destabilizers()
            .iter()
            .cloned()
            .map(PauliString::from_rust)
            .collect()
    }

    /// Logical Z operators.
    #[getter]
    fn logical_zs(&self) -> Vec<PauliString> {
        self.inner
            .logical_zs()
            .iter()
            .cloned()
            .map(PauliString::from_rust)
            .collect()
    }

    /// Logical X operators.
    #[getter]
    fn logical_xs(&self) -> Vec<PauliString> {
        self.inner
            .logical_xs()
            .iter()
            .cloned()
            .map(PauliString::from_rust)
            .collect()
    }

    /// Verify all stabilizer and logical commutation relations.
    fn verify(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    /// Find the code distance and one minimum-weight logical operator.
    #[pyo3(signature = (max_weight=None, css=false, verbose=false))]
    fn distance(
        &self,
        max_weight: Option<usize>,
        css: bool,
        verbose: bool,
    ) -> PyResult<Option<PyDistanceResult>> {
        let config = DistanceSearchConfig {
            max_weight,
            css_only: css,
            verbose,
        };
        calculate_distance(&self.inner, &config)
            .map(|result| result.map(PyDistanceResult::from))
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    /// Find all logical operators at the minimum weight searched.
    #[pyo3(signature = (max_weight=None, css=false, verbose=false))]
    fn min_weight_logicals(
        &self,
        max_weight: Option<usize>,
        css: bool,
        verbose: bool,
    ) -> PyResult<Vec<PyLogicalOperatorInfo>> {
        let config = DistanceSearchConfig {
            max_weight,
            css_only: css,
            verbose,
        };
        find_shortest_logicals(&self.inner, &config, 0)
            .map(|logicals| {
                logicals
                    .into_iter()
                    .map(PyLogicalOperatorInfo::from)
                    .collect()
            })
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    /// Find logical operators through ``delta`` weights above the minimum.
    #[pyo3(signature = (delta=0, max_weight=None, css=false, verbose=false))]
    fn shortest_logicals(
        &self,
        delta: usize,
        max_weight: Option<usize>,
        css: bool,
        verbose: bool,
    ) -> PyResult<Vec<PyLogicalOperatorInfo>> {
        let config = DistanceSearchConfig {
            max_weight,
            css_only: css,
            verbose,
        };
        find_shortest_logicals(&self.inner, &config, delta)
            .map(|logicals| {
                logicals
                    .into_iter()
                    .map(PyLogicalOperatorInfo::from)
                    .collect()
            })
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "StabilizerCodeSpec([[{}, {}]])",
            self.inner.num_qubits(),
            self.inner.num_logical_qubits()
        )
    }
}

/// Register stabilizer-code specification and distance-result types.
pub fn register_stabilizer_code_spec_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStabilizerCodeSpec>()?;
    m.add_class::<PyStabilizerCodeSpecBuilder>()?;
    m.add_class::<PyDistanceResult>()?;
    m.add_class::<PyLogicalOperatorInfo>()?;
    Ok(())
}
