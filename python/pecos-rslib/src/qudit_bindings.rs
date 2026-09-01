// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Thin Python bindings for the Rust qudit reference simulators.

use num_complex::Complex64;
use pecos_random::PecosRng;
use pecos_simulators::{
    DensityMatrixDiagnostics, InstrumentSample, KrausSample, MeasurementSample, QuditDensityMatrix,
    QuditError as CoreQuditError, QuditStateVec, QutritDensityMatrix, QutritStateVec, basis_swap,
    embedded_qubit_unitary, qutrit_leakage_channel, qutrit_seepage_channel,
};
use pyo3::exceptions::{PyException, PyIndexError, PyMemoryError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple, PyType};

pyo3::create_exception!(
    pecos_rslib,
    QuditError,
    PyException,
    "Base class for every error raised by the qudit reference simulators.\n\n\
     Each instance carries a `kind` attribute holding a stable machine-readable\n\
     tag for the underlying condition, such as `\"LeakagePopulation\"`."
);

static QUDIT_VALUE_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static QUDIT_INDEX_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static QUDIT_MEMORY_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// Build (once) an exception class deriving from both `QuditError` and a builtin.
///
/// Deriving from the builtin keeps every existing `except ValueError` /
/// `except IndexError` / `except MemoryError` working unchanged, while the
/// `QuditError` base makes the whole family catchable in one clause.
fn derived_exception<'py>(
    py: Python<'py>,
    cell: &'static PyOnceLock<Py<PyType>>,
    name: &str,
    builtin: &Bound<'py, PyType>,
) -> PyResult<Bound<'py, PyType>> {
    let class = cell.get_or_try_init(py, || {
        let bases = PyTuple::new(
            py,
            [
                py.get_type::<QuditError>().into_any(),
                builtin.clone().into_any(),
            ],
        )?;
        let namespace = PyDict::new(py);
        namespace.set_item("__module__", "pecos_rslib.simulators")?;
        let metaclass = py.import("builtins")?.getattr("type")?;
        let class = metaclass.call1((name, bases, namespace))?;
        Ok::<_, PyErr>(class.cast_into::<PyType>()?.unbind())
    })?;
    Ok(class.bind(py).clone())
}

/// Every exception class this module publishes, in registration order.
pub fn exception_classes(py: Python<'_>) -> PyResult<Vec<(&'static str, Bound<'_, PyAny>)>> {
    Ok(vec![
        ("QuditError", py.get_type::<QuditError>().into_any()),
        ("QuditValueError", value_error_class(py)?.into_any()),
        ("QuditIndexError", index_error_class(py)?.into_any()),
        ("QuditMemoryError", memory_error_class(py)?.into_any()),
    ])
}

fn value_error_class(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    derived_exception(
        py,
        &QUDIT_VALUE_ERROR,
        "QuditValueError",
        &py.get_type::<PyValueError>(),
    )
}

fn index_error_class(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    derived_exception(
        py,
        &QUDIT_INDEX_ERROR,
        "QuditIndexError",
        &py.get_type::<PyIndexError>(),
    )
}

fn memory_error_class(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    derived_exception(
        py,
        &QUDIT_MEMORY_ERROR,
        "QuditMemoryError",
        &py.get_type::<PyMemoryError>(),
    )
}

/// Stable machine-readable tag for a core error.
///
/// Deliberately exhaustive: adding a `QuditError` variant must fail this match
/// rather than silently produce an unlabelled Python exception.
fn error_kind(error: &CoreQuditError) -> &'static str {
    match error {
        CoreQuditError::InvalidLocalDimension(_) => "InvalidLocalDimension",
        CoreQuditError::DimensionOverflow => "DimensionOverflow",
        CoreQuditError::InvalidStateLength { .. } => "InvalidStateLength",
        CoreQuditError::TargetOutOfRange { .. } => "TargetOutOfRange",
        CoreQuditError::DuplicateTarget(_) => "DuplicateTarget",
        CoreQuditError::EmptyTargets => "EmptyTargets",
        CoreQuditError::InvalidOperatorLength { .. } => "InvalidOperatorLength",
        CoreQuditError::InvalidBasisState { .. } => "InvalidBasisState",
        CoreQuditError::ZeroNorm => "ZeroNorm",
        CoreQuditError::NonFiniteValue => "NonFiniteValue",
        CoreQuditError::InvalidProbability(_) => "InvalidProbability",
        CoreQuditError::NotNormalized { .. } => "NotNormalized",
        CoreQuditError::EmptyKrausChannel => "EmptyKrausChannel",
        CoreQuditError::NonUnitary { .. } => "NonUnitary",
        CoreQuditError::NotTracePreserving { .. } => "NotTracePreserving",
        CoreQuditError::LeakagePopulation { .. } => "LeakagePopulation",
        CoreQuditError::InvalidMeasurementPartition => "InvalidMeasurementPartition",
        CoreQuditError::InvalidMeasurementInstrument => "InvalidMeasurementInstrument",
        CoreQuditError::NonHermitian { .. } => "NonHermitian",
        CoreQuditError::NotPositiveSemidefinite { .. } => "NotPositiveSemidefinite",
        CoreQuditError::AllocationFailed { .. } => "AllocationFailed",
        CoreQuditError::InvalidTolerance(_) => "InvalidTolerance",
    }
}

fn python_error(error: CoreQuditError) -> PyErr {
    let kind = error_kind(&error);
    let message = error.to_string();
    Python::attach(|py| {
        let class = match error {
            CoreQuditError::AllocationFailed { .. } => memory_error_class(py),
            CoreQuditError::TargetOutOfRange { .. } | CoreQuditError::InvalidBasisState { .. } => {
                index_error_class(py)
            }
            _ => value_error_class(py),
        };
        match class.and_then(|class| {
            let error = PyErr::from_type(class, message);
            error.value(py).setattr("kind", kind)?;
            Ok(error)
        }) {
            Ok(error) | Err(error) => error,
        }
    })
}

fn seeded_rng(seed: Option<u64>) -> PecosRng {
    seed.map_or_else(rand::make_rng, PecosRng::seed_from_u64)
}

/// Return a local basis-state swap in row-major order.
#[pyfunction(name = "basis_swap")]
pub fn py_basis_swap(
    local_dimension: usize,
    first: usize,
    second: usize,
) -> PyResult<Vec<Complex64>> {
    basis_swap(local_dimension, first, second).map_err(python_error)
}

/// Embed a 2x2 qubit unitary into a larger local Hilbert space.
#[pyfunction(name = "embedded_qubit_unitary")]
pub fn py_embedded_qubit_unitary(
    local_dimension: usize,
    qubit_unitary: [Complex64; 4],
) -> PyResult<Vec<Complex64>> {
    embedded_qubit_unitary(local_dimension, &qubit_unitary).map_err(python_error)
}

/// Return the generic qutrit leakage channel with probability `probability`.
#[pyfunction(name = "qutrit_leakage_channel")]
pub fn py_qutrit_leakage_channel(probability: f64) -> PyResult<Vec<Vec<Complex64>>> {
    qutrit_leakage_channel(probability).map_err(python_error)
}

/// Return a qutrit seepage channel.
///
/// `zero_fraction` is the conditional probability that seeped population returns
/// to `|0>` rather than `|1>`, matching the Rust core's parameter of the same name.
#[pyfunction(name = "qutrit_seepage_channel")]
pub fn py_qutrit_seepage_channel(
    probability: f64,
    zero_fraction: f64,
) -> PyResult<Vec<Vec<Complex64>>> {
    qutrit_seepage_channel(probability, zero_fraction).map_err(python_error)
}

/// Result of sampling a Kraus trajectory branch.
#[pyclass(
    name = "KrausSample",
    module = "pecos_rslib.simulators",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyKrausSample {
    pub operator_index: usize,
    pub probability: f64,
}

impl From<KrausSample> for PyKrausSample {
    fn from(sample: KrausSample) -> Self {
        Self {
            operator_index: sample.operator_index,
            probability: sample.probability,
        }
    }
}

/// Result of sampling a projective measurement.
#[pyclass(
    name = "MeasurementSample",
    module = "pecos_rslib.simulators",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyMeasurementSample {
    pub outcome: usize,
    pub probability: f64,
}

impl From<MeasurementSample> for PyMeasurementSample {
    fn from(sample: MeasurementSample) -> Self {
        Self {
            outcome: sample.outcome,
            probability: sample.probability,
        }
    }
}

/// Result of sampling a generalized-measurement trajectory.
#[pyclass(
    name = "InstrumentSample",
    module = "pecos_rslib.simulators",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyInstrumentSample {
    pub outcome: usize,
    pub operator_index: usize,
    pub outcome_probability: f64,
    pub branch_probability: f64,
}

impl From<InstrumentSample> for PyInstrumentSample {
    fn from(sample: InstrumentSample) -> Self {
        Self {
            outcome: sample.outcome,
            operator_index: sample.operator_index,
            outcome_probability: sample.outcome_probability,
            branch_probability: sample.branch_probability,
        }
    }
}

/// Numerical diagnostics for an exact density operator.
#[pyclass(
    name = "DensityMatrixDiagnostics",
    module = "pecos_rslib.simulators",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyDensityMatrixDiagnostics {
    pub trace: f64,
    pub trace_imaginary_error: f64,
    pub hermiticity_error: f64,
    pub minimum_eigenvalue: f64,
}

#[pymethods]
impl PyDensityMatrixDiagnostics {
    fn is_physical(&self, tolerance: f64) -> bool {
        DensityMatrixDiagnostics {
            trace: self.trace,
            trace_imaginary_error: self.trace_imaginary_error,
            hermiticity_error: self.hermiticity_error,
            minimum_eigenvalue: self.minimum_eigenvalue,
        }
        .is_physical(tolerance)
    }
}

impl From<DensityMatrixDiagnostics> for PyDensityMatrixDiagnostics {
    fn from(diagnostics: DensityMatrixDiagnostics) -> Self {
        Self {
            trace: diagnostics.trace,
            trace_imaginary_error: diagnostics.trace_imaginary_error,
            hermiticity_error: diagnostics.hermiticity_error,
            minimum_eigenvalue: diagnostics.minimum_eigenvalue,
        }
    }
}

/// Dense state-vector simulation with a uniform local dimension.
///
/// Index conventions, matching the Rust core:
///
/// - Site 0 is the least-significant radix digit of a global basis index.
/// - For a local operator, ``targets[0]`` is the least-significant digit of the
///   operator's row and column indices, so `[0, 1]` and `[1, 0]` are different
///   operations.
/// - Operators, Kraus operators, and reduced density matrices are flat
///   row-major sequences, never nested rows.
#[pyclass(name = "QuditStateVec", module = "pecos_rslib.simulators", subclass)]
pub struct PyQuditStateVec {
    inner: QuditStateVec,
}

#[pymethods]
impl PyQuditStateVec {
    #[new]
    #[pyo3(signature = (num_sites, local_dimension, seed=None))]
    fn new(num_sites: usize, local_dimension: usize, seed: Option<u64>) -> PyResult<Self> {
        Ok(Self {
            inner: QuditStateVec::with_rng(num_sites, local_dimension, seeded_rng(seed))
                .map_err(python_error)?,
        })
    }

    #[classmethod]
    #[pyo3(signature = (num_sites, local_dimension, state, seed=None))]
    fn from_state(
        _cls: &Bound<'_, PyType>,
        num_sites: usize,
        local_dimension: usize,
        state: Vec<Complex64>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: QuditStateVec::from_state(num_sites, local_dimension, state, seeded_rng(seed))
                .map_err(python_error)?,
        })
    }

    #[staticmethod]
    fn required_memory_bytes(num_sites: usize, local_dimension: usize) -> PyResult<usize> {
        QuditStateVec::required_memory_bytes(num_sites, local_dimension).map_err(python_error)
    }

    #[getter]
    fn num_sites(&self) -> usize {
        self.inner.num_sites()
    }

    #[getter]
    fn local_dimension(&self) -> usize {
        self.inner.local_dimension()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn state(&self) -> Vec<Complex64> {
        self.inner.state().to_vec()
    }

    fn probability(&self, basis_state: usize) -> PyResult<f64> {
        self.inner.probability(basis_state).map_err(python_error)
    }

    fn outcome_probabilities(&self, target: usize) -> PyResult<Vec<f64>> {
        self.inner
            .outcome_probabilities(target)
            .map_err(python_error)
    }

    /// Joint outcome distribution over the ordered target sites.
    ///
    /// `targets[0]` is the least-significant digit of the returned outcome index.
    fn joint_outcome_probabilities(&self, targets: Vec<usize>) -> PyResult<Vec<f64>> {
        self.inner
            .joint_outcome_probabilities(&targets)
            .map_err(python_error)
    }

    /// Apply a flat row-major local operator of length `local_dimension ** (2 * len(targets))`.
    ///
    /// `targets[0]` is the least-significant digit of the operator's row and
    /// column indices. The operator must be unitary.
    fn apply_operator(&mut self, targets: Vec<usize>, operator: Vec<Complex64>) -> PyResult<()> {
        self.inner
            .apply_operator(&targets, &operator)
            .map(|_| ())
            .map_err(python_error)
    }

    fn apply_embedded_qubit_unitary(
        &mut self,
        target: usize,
        qubit_unitary: [Complex64; 4],
    ) -> PyResult<()> {
        self.inner
            .apply_embedded_qubit_unitary(target, &qubit_unitary)
            .map(|_| ())
            .map_err(python_error)
    }

    fn apply_kraus(
        &mut self,
        targets: Vec<usize>,
        operators: Vec<Vec<Complex64>>,
    ) -> PyResult<PyKrausSample> {
        self.inner
            .apply_kraus(&targets, &operators)
            .map(Into::into)
            .map_err(python_error)
    }

    fn instrument_probabilities(
        &self,
        targets: Vec<usize>,
        outcomes: Vec<Vec<Vec<Complex64>>>,
    ) -> PyResult<Vec<f64>> {
        self.inner
            .instrument_probabilities(&targets, &outcomes)
            .map_err(python_error)
    }

    fn measure_instrument(
        &mut self,
        targets: Vec<usize>,
        outcomes: Vec<Vec<Vec<Complex64>>>,
    ) -> PyResult<PyInstrumentSample> {
        self.inner
            .measure_instrument(&targets, &outcomes)
            .map(Into::into)
            .map_err(python_error)
    }

    fn measure(&mut self, target: usize) -> PyResult<usize> {
        self.inner.measure(target).map_err(python_error)
    }

    fn measure_joint(&mut self, targets: Vec<usize>) -> PyResult<PyMeasurementSample> {
        self.inner
            .measure_joint(&targets)
            .map(Into::into)
            .map_err(python_error)
    }

    /// Measure a coarse-grained partition of the targets' joint local basis.
    ///
    /// Each group lists joint basis indices in the `targets[0]`-least-significant
    /// ordering. The groups must cover every joint basis index exactly once.
    fn measure_partition(
        &mut self,
        targets: Vec<usize>,
        groups: Vec<Vec<usize>>,
    ) -> PyResult<PyMeasurementSample> {
        self.inner
            .measure_partition(&targets, &groups)
            .map(Into::into)
            .map_err(python_error)
    }

    /// Measure `|0>` versus `|1>` on a site with no population above `|1>`.
    ///
    /// Raises `ValueError` when the site carries leakage population, rather than
    /// binning it into a detector outcome.
    fn measure_computational(&mut self, target: usize) -> PyResult<bool> {
        self.inner
            .measure_computational(target)
            .map_err(python_error)
    }

    /// Reset one site to local basis state zero.
    ///
    /// This resets a single site, not the whole simulator. On the state-vector
    /// backend it samples a trajectory branch and so consumes randomness; the
    /// density-matrix backend applies the exact reset channel.
    fn reset_site(&mut self, target: usize) -> PyResult<()> {
        self.inner
            .reset_site(target)
            .map(|_| ())
            .map_err(python_error)
    }

    fn prepare_basis(&mut self, target: usize, basis_state: usize) -> PyResult<()> {
        self.inner
            .prepare_basis(target, basis_state)
            .map(|_| ())
            .map_err(python_error)
    }

    /// Reduced density matrix over the ordered target sites.
    ///
    /// Returned flat and row-major, with `targets[0]` the least-significant
    /// digit of both the row and the column index.
    fn reduced_density_matrix(&self, targets: Vec<usize>) -> PyResult<Vec<Complex64>> {
        self.inner
            .reduced_density_matrix(&targets)
            .map_err(python_error)
    }
}

/// Qutrit state-vector simulation in the basis `|0>, |1>, |L>`.
#[pyclass(name = "QutritStateVec", module = "pecos_rslib.simulators", extends = PyQuditStateVec)]
pub struct PyQutritStateVec;

#[pymethods]
impl PyQutritStateVec {
    #[new]
    #[pyo3(signature = (num_sites, seed=None))]
    fn new(num_sites: usize, seed: Option<u64>) -> PyResult<PyClassInitializer<Self>> {
        let base = PyQuditStateVec {
            inner: QutritStateVec::with_rng(num_sites, seeded_rng(seed))
                .map_err(python_error)?
                .into_inner(),
        };
        Ok(PyClassInitializer::from(base).add_subclass(Self))
    }

    #[classmethod]
    #[pyo3(signature = (num_sites, state, seed=None))]
    fn from_state(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        num_sites: usize,
        state: Vec<Complex64>,
        seed: Option<u64>,
    ) -> PyResult<Py<Self>> {
        let base = PyQuditStateVec {
            inner: QutritStateVec::from_state(num_sites, state, seeded_rng(seed))
                .map_err(python_error)?
                .into_inner(),
        };
        Py::new(py, PyClassInitializer::from(base).add_subclass(Self))
    }

    #[staticmethod]
    fn required_memory_bytes(num_sites: usize) -> PyResult<usize> {
        QutritStateVec::required_memory_bytes(num_sites).map_err(python_error)
    }
}

/// Exact dense density-matrix simulation with a uniform local dimension.
///
/// Index conventions, matching the Rust core:
///
/// - Site 0 is the least-significant radix digit of a global basis index.
/// - For a local operator, ``targets[0]`` is the least-significant digit of the
///   operator's row and column indices, so `[0, 1]` and `[1, 0]` are different
///   operations.
/// - Operators, Kraus operators, and reduced density matrices are flat
///   row-major sequences, never nested rows.
#[pyclass(
    name = "QuditDensityMatrix",
    module = "pecos_rslib.simulators",
    subclass
)]
pub struct PyQuditDensityMatrix {
    inner: QuditDensityMatrix,
}

#[pymethods]
impl PyQuditDensityMatrix {
    #[new]
    #[pyo3(signature = (num_sites, local_dimension, seed=None))]
    fn new(num_sites: usize, local_dimension: usize, seed: Option<u64>) -> PyResult<Self> {
        Ok(Self {
            inner: QuditDensityMatrix::with_rng(num_sites, local_dimension, seeded_rng(seed))
                .map_err(python_error)?,
        })
    }

    #[classmethod]
    #[pyo3(signature = (num_sites, local_dimension, density_matrix, seed=None))]
    fn from_density_matrix(
        _cls: &Bound<'_, PyType>,
        num_sites: usize,
        local_dimension: usize,
        density_matrix: Vec<Complex64>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: QuditDensityMatrix::from_density_matrix(
                num_sites,
                local_dimension,
                density_matrix,
                seeded_rng(seed),
            )
            .map_err(python_error)?,
        })
    }

    #[staticmethod]
    fn required_memory_bytes(num_sites: usize, local_dimension: usize) -> PyResult<usize> {
        QuditDensityMatrix::required_memory_bytes(num_sites, local_dimension).map_err(python_error)
    }

    #[getter]
    fn num_sites(&self) -> usize {
        self.inner.num_sites()
    }

    #[getter]
    fn local_dimension(&self) -> usize {
        self.inner.local_dimension()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn density_matrix(&self) -> Vec<Complex64> {
        self.inner.density_matrix().to_vec()
    }

    fn probability(&self, basis_state: usize) -> PyResult<f64> {
        self.inner.probability(basis_state).map_err(python_error)
    }

    fn trace(&self) -> Complex64 {
        self.inner.trace()
    }

    fn purity(&self) -> f64 {
        self.inner.purity()
    }

    fn diagnostics(&self) -> PyDensityMatrixDiagnostics {
        self.inner.diagnostics().into()
    }

    fn validate_physicality(&self, tolerance: f64) -> PyResult<()> {
        self.inner
            .validate_physicality(tolerance)
            .map_err(python_error)
    }

    fn outcome_probabilities(&self, target: usize) -> PyResult<Vec<f64>> {
        self.inner
            .outcome_probabilities(target)
            .map_err(python_error)
    }

    /// Joint outcome distribution over the ordered target sites.
    ///
    /// `targets[0]` is the least-significant digit of the returned outcome index.
    fn joint_outcome_probabilities(&self, targets: Vec<usize>) -> PyResult<Vec<f64>> {
        self.inner
            .joint_outcome_probabilities(&targets)
            .map_err(python_error)
    }

    /// Apply a flat row-major local operator of length `local_dimension ** (2 * len(targets))`.
    ///
    /// `targets[0]` is the least-significant digit of the operator's row and
    /// column indices. The operator must be unitary.
    fn apply_operator(&mut self, targets: Vec<usize>, operator: Vec<Complex64>) -> PyResult<()> {
        self.inner
            .apply_operator(&targets, &operator)
            .map(|_| ())
            .map_err(python_error)
    }

    fn apply_embedded_qubit_unitary(
        &mut self,
        target: usize,
        qubit_unitary: [Complex64; 4],
    ) -> PyResult<()> {
        self.inner
            .apply_embedded_qubit_unitary(target, &qubit_unitary)
            .map(|_| ())
            .map_err(python_error)
    }

    fn apply_kraus(&mut self, targets: Vec<usize>, operators: Vec<Vec<Complex64>>) -> PyResult<()> {
        self.inner
            .apply_kraus(&targets, &operators)
            .map(|_| ())
            .map_err(python_error)
    }

    fn instrument_probabilities(
        &self,
        targets: Vec<usize>,
        outcomes: Vec<Vec<Vec<Complex64>>>,
    ) -> PyResult<Vec<f64>> {
        self.inner
            .instrument_probabilities(&targets, &outcomes)
            .map_err(python_error)
    }

    fn measure_instrument(
        &mut self,
        targets: Vec<usize>,
        outcomes: Vec<Vec<Vec<Complex64>>>,
    ) -> PyResult<PyMeasurementSample> {
        self.inner
            .measure_instrument(&targets, &outcomes)
            .map(Into::into)
            .map_err(python_error)
    }

    fn measure(&mut self, target: usize) -> PyResult<usize> {
        self.inner.measure(target).map_err(python_error)
    }

    fn measure_joint(&mut self, targets: Vec<usize>) -> PyResult<PyMeasurementSample> {
        self.inner
            .measure_joint(&targets)
            .map(Into::into)
            .map_err(python_error)
    }

    /// Measure a coarse-grained partition of the targets' joint local basis.
    ///
    /// Each group lists joint basis indices in the `targets[0]`-least-significant
    /// ordering. The groups must cover every joint basis index exactly once.
    fn measure_partition(
        &mut self,
        targets: Vec<usize>,
        groups: Vec<Vec<usize>>,
    ) -> PyResult<PyMeasurementSample> {
        self.inner
            .measure_partition(&targets, &groups)
            .map(Into::into)
            .map_err(python_error)
    }

    /// Measure `|0>` versus `|1>` on a site with no population above `|1>`.
    ///
    /// Raises `ValueError` when the site carries leakage population, rather than
    /// binning it into a detector outcome.
    fn measure_computational(&mut self, target: usize) -> PyResult<bool> {
        self.inner
            .measure_computational(target)
            .map_err(python_error)
    }

    /// Reset one site to local basis state zero.
    ///
    /// This resets a single site, not the whole simulator. On the state-vector
    /// backend it samples a trajectory branch and so consumes randomness; the
    /// density-matrix backend applies the exact reset channel.
    fn reset_site(&mut self, target: usize) -> PyResult<()> {
        self.inner
            .reset_site(target)
            .map(|_| ())
            .map_err(python_error)
    }

    fn prepare_basis(&mut self, target: usize, basis_state: usize) -> PyResult<()> {
        self.inner
            .prepare_basis(target, basis_state)
            .map(|_| ())
            .map_err(python_error)
    }

    /// Reduced density matrix over the ordered target sites.
    ///
    /// Returned flat and row-major, with `targets[0]` the least-significant
    /// digit of both the row and the column index.
    fn reduced_density_matrix(&self, targets: Vec<usize>) -> PyResult<Vec<Complex64>> {
        self.inner
            .reduced_density_matrix(&targets)
            .map_err(python_error)
    }
}

/// Qutrit density-matrix simulation in the basis `|0>, |1>, |L>`.
#[pyclass(
    name = "QutritDensityMatrix",
    module = "pecos_rslib.simulators",
    extends = PyQuditDensityMatrix
)]
pub struct PyQutritDensityMatrix;

#[pymethods]
impl PyQutritDensityMatrix {
    #[new]
    #[pyo3(signature = (num_sites, seed=None))]
    fn new(num_sites: usize, seed: Option<u64>) -> PyResult<PyClassInitializer<Self>> {
        let base = PyQuditDensityMatrix {
            inner: QutritDensityMatrix::with_rng(num_sites, seeded_rng(seed))
                .map_err(python_error)?
                .into_inner(),
        };
        Ok(PyClassInitializer::from(base).add_subclass(Self))
    }

    #[classmethod]
    #[pyo3(signature = (num_sites, density_matrix, seed=None))]
    fn from_density_matrix(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        num_sites: usize,
        density_matrix: Vec<Complex64>,
        seed: Option<u64>,
    ) -> PyResult<Py<Self>> {
        let base = PyQuditDensityMatrix {
            inner: QutritDensityMatrix::from_density_matrix(
                num_sites,
                density_matrix,
                seeded_rng(seed),
            )
            .map_err(python_error)?
            .into_inner(),
        };
        Py::new(py, PyClassInitializer::from(base).add_subclass(Self))
    }

    #[staticmethod]
    fn required_memory_bytes(num_sites: usize) -> PyResult<usize> {
        QutritDensityMatrix::required_memory_bytes(num_sites).map_err(python_error)
    }
}
