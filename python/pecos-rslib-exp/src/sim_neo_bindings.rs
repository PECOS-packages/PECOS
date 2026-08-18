// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Python bindings for `sim_neo` with builder pattern.
//!
//! Mirrors the Rust-side API:
//! ```python
//! results = (sim_neo(tc)
//!     .quantum(stab_mps().lazy_measure().max_bond_dim(128))
//!     .noise(depolarizing().p1(0.003).p2(0.003).p_meas(0.003).p_prep(0.003).idle_rz(0.05))
//!     .sampling(monte_carlo(5000))
//!     .seed(42)
//!     .run())
//! ```

use pecos_core::{Angle64, ChannelExpr, Gate, Pauli, PauliString, QuarterPhase, QubitId};
use pecos_neo::command::CommandBuilder;
use pecos_neo::noise::{
    ComposableNoiseModel, MeasurementChannel, PreparationChannel, SingleQubitChannel,
    TwoQubitChannel,
};
use pecos_neo::tool::sim_neo;
use pecos_simulators::measurement_sampler::SampleResult;
use pyo3::prelude::*;

#[derive(serde::Deserialize)]
struct RecDef {
    /// DEM-style negative record offsets. Alternative to `meas_ids`.
    #[serde(default)]
    records: Vec<i32>,
    /// Stable measurement ids. Alternative to `records`; the traced-QIS
    /// pipeline emits only this field.
    #[serde(default)]
    meas_ids: Vec<usize>,
}

fn measurement_record_index(record: i32, num_measurements: usize) -> Option<usize> {
    let idx = if record < 0 {
        i32::try_from(num_measurements).ok()?.checked_add(record)?
    } else {
        record
    };
    usize::try_from(idx)
        .ok()
        .filter(|&idx| idx < num_measurements)
}

// ============================================================================
// Columnar raw measurement result (stays in Rust memory)
// ============================================================================

/// Raw measurement batch — common result type for all sim_neo backends.
///
/// Stores either columnar bit-packed data (meas_sampling) or row-major
/// data (stabilizer/statevec). The Python API is identical regardless of
/// storage: `result[shot]`, `result.get(shot, meas)`, iteration, `len()`.
#[pyclass(name = "RawMeasurementResult", module = "pecos_rslib_exp")]
pub struct PyRawMeasurementResult {
    storage: RawMeasurementStorage,
    /// Per-row weights (path probabilities for path enumeration,
    /// importance weights for importance sampling). None for plain
    /// Monte Carlo runs.
    weights: Option<Vec<f64>>,
    /// Rare-event estimate (subset simulation only; rows are empty for
    /// subset runs). Mirrors Rust `SimulationResults::subset`.
    subset: Option<Py<PySubsetResult>>,
}

enum RawMeasurementStorage {
    /// Columnar bit-packed (from meas_sampling geometric sampler).
    Columnar(SampleResult),
    /// Row-major (from gate-by-gate stabilizer/statevec simulation).
    RowMajor {
        rows: Vec<Vec<u8>>,
        num_measurements: usize,
    },
}

impl RawMeasurementStorage {
    fn num_shots(&self) -> usize {
        match self {
            Self::Columnar(s) => s.shots(),
            Self::RowMajor { rows, .. } => rows.len(),
        }
    }

    fn num_measurements(&self) -> usize {
        match self {
            Self::Columnar(s) => s.num_measurements(),
            Self::RowMajor {
                num_measurements, ..
            } => *num_measurements,
        }
    }

    fn get(&self, shot: usize, measurement: usize) -> u8 {
        match self {
            Self::Columnar(s) => u8::from(s.get(shot, measurement).0),
            Self::RowMajor { rows, .. } => rows[shot][measurement],
        }
    }

    fn get_shot(&self, shot: usize) -> Vec<u8> {
        match self {
            Self::Columnar(s) => {
                let n = s.num_measurements();
                let mut row = Vec::with_capacity(n);
                for meas in 0..n {
                    row.push(u8::from(s.get(shot, meas).0));
                }
                row
            }
            Self::RowMajor { rows, .. } => rows[shot].clone(),
        }
    }
}

impl PyRawMeasurementResult {
    /// Convert a signed Python index to a checked usize.
    /// Negative indices raise IndexError (no Python-list-style wrapping).
    fn check_index(idx: isize, len: usize, name: &str) -> PyResult<usize> {
        if idx < 0 {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "negative {name} index {idx}"
            )));
        }
        let u = usize::try_from(idx).map_err(|_| {
            pyo3::exceptions::PyIndexError::new_err(format!("invalid {name} index {idx}"))
        })?;
        if u >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "{name} {u} out of range ({len})"
            )));
        }
        Ok(u)
    }

    /// Construct from columnar SampleResult (meas_sampling path).
    pub fn from_columnar(result: SampleResult) -> Self {
        Self {
            storage: RawMeasurementStorage::Columnar(result),
            weights: None,
            subset: None,
        }
    }

    /// Construct from row-major data (stabilizer/statevec path).
    pub fn from_rows(rows: Vec<Vec<u8>>) -> Self {
        let num_measurements = rows.first().map_or(0, Vec::len);
        Self {
            storage: RawMeasurementStorage::RowMajor {
                rows,
                num_measurements,
            },
            weights: None,
            subset: None,
        }
    }

    /// Construct from row-major data with per-row weights.
    pub fn from_rows_weighted(rows: Vec<Vec<u8>>, weights: Vec<f64>) -> Self {
        let mut result = Self::from_rows(rows);
        result.weights = Some(weights);
        result
    }

    /// Construct a subset-simulation result (no rows; estimate only).
    pub fn from_subset(subset: Py<PySubsetResult>) -> Self {
        let mut result = Self::from_rows(Vec::new());
        result.subset = Some(subset);
        result
    }
}

#[pymethods]
impl PyRawMeasurementResult {
    /// Number of shots.
    #[getter]
    fn num_shots(&self) -> usize {
        self.storage.num_shots()
    }

    /// Number of measurements per shot.
    #[getter]
    fn num_measurements(&self) -> usize {
        self.storage.num_measurements()
    }

    /// Per-row weights, or None for plain Monte Carlo runs.
    ///
    /// Path enumeration: exact path probabilities (sum to 1 for complete
    /// enumeration).
    #[getter]
    fn weights(&self) -> Option<Vec<f64>> {
        self.weights.clone()
    }

    /// Rare-event estimate, or None unless run with subset simulation.
    #[getter]
    fn subset(&self, py: Python<'_>) -> Option<Py<PySubsetResult>> {
        self.subset.as_ref().map(|s| s.clone_ref(py))
    }

    /// Get a single measurement bit (0 or 1).
    fn get(&self, shot: isize, measurement: isize) -> PyResult<u8> {
        let s = Self::check_index(shot, self.storage.num_shots(), "shot")?;
        let m = Self::check_index(measurement, self.storage.num_measurements(), "measurement")?;
        Ok(self.storage.get(s, m))
    }

    /// Get one full shot as a list of u8.
    fn get_shot(&self, shot: isize) -> PyResult<Vec<u8>> {
        let s = Self::check_index(shot, self.storage.num_shots(), "shot")?;
        Ok(self.storage.get_shot(s))
    }

    /// Materialize all shots as list[list[int]].
    fn to_list(&self) -> Vec<Vec<u8>> {
        let n = self.storage.num_shots();
        (0..n).map(|i| self.storage.get_shot(i)).collect()
    }

    fn __len__(&self) -> usize {
        self.storage.num_shots()
    }

    fn __getitem__(&self, shot: isize) -> PyResult<Vec<u8>> {
        let s = Self::check_index(shot, self.storage.num_shots(), "index")?;
        Ok(self.storage.get_shot(s))
    }
}

// ============================================================================
// Noise model builder
// ============================================================================

/// Builder for composable noise models.
///
/// Example:
///     depolarizing().p1(0.003).p2(0.003).p_meas(0.003).p_prep(0.003).idle_rz(0.05)
#[pyclass(
    name = "NoiseModelBuilder",
    skip_from_py_object,
    module = "pecos_rslib_exp"
)]
#[derive(Clone, Default)]
pub struct PyNoiseModelBuilder {
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
    idle_rz_angle: f64,
}

#[pymethods]
impl PyNoiseModelBuilder {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Single-qubit depolarizing rate (X/Y/Z each with p/3 after unitary 1q gates).
    fn p1(&self, p: f64) -> Self {
        Self {
            p1: p,
            ..self.clone()
        }
    }

    /// Two-qubit depolarizing rate (15 Paulis each with p/15 after unitary 2q gates).
    fn p2(&self, p: f64) -> Self {
        Self {
            p2: p,
            ..self.clone()
        }
    }

    /// Measurement bit-flip rate (symmetric, after MZ).
    fn p_meas(&self, p: f64) -> Self {
        Self {
            p_meas: p,
            ..self.clone()
        }
    }

    /// Preparation error rate (X flip after PZ/QAlloc).
    fn p_prep(&self, p: f64) -> Self {
        Self {
            p_prep: p,
            ..self.clone()
        }
    }

    /// Coherent idle RZ angle (radians) applied to both qubits after each CX.
    fn idle_rz(&self, angle: f64) -> Self {
        Self {
            idle_rz_angle: angle,
            ..self.clone()
        }
    }
}

impl PyNoiseModelBuilder {
    fn build_noise(&self) -> Option<ComposableNoiseModel> {
        let has_noise = self.p1 > 0.0
            || self.p2 > 0.0
            || self.p_meas > 0.0
            || self.p_prep > 0.0
            || self.idle_rz_angle > 0.0;

        if !has_noise {
            return None;
        }

        let mut noise = ComposableNoiseModel::new();
        if self.p1 > 0.0 {
            noise = noise.add_channel(SingleQubitChannel::depolarizing(self.p1));
        }
        if self.p2 > 0.0 {
            noise = noise.add_channel(TwoQubitChannel::depolarizing(self.p2));
        }
        if self.p_meas > 0.0 {
            noise = noise.add_channel(MeasurementChannel::symmetric(self.p_meas));
        }
        if self.p_prep > 0.0 {
            noise = noise.add_channel(PreparationChannel::new(self.p_prep));
        }
        if self.idle_rz_angle > 0.0 {
            noise = noise.add_channel(crate::coherent_idle_channel::CoherentIdleChannel::new(
                self.idle_rz_angle,
            ));
        }
        Some(noise)
    }
}

/// Create a noise model builder.
#[pyfunction]
pub fn depolarizing() -> PyNoiseModelBuilder {
    PyNoiseModelBuilder::new()
}

/// Marker type for the stabilizer (SparseStab) backend.
///
/// Pass to `.quantum()` to select the stabilizer simulator.
///
/// Example:
///     sim_neo(tc).quantum(stabilizer()).noise(depolarizing().p2(0.01)).sampling(monte_carlo(10000)).run()
#[pyclass(
    name = "StabilizerBuilder",
    skip_from_py_object,
    module = "pecos_rslib_exp"
)]
#[derive(Clone)]
pub struct PyStabilizerBuilder;

#[pymethods]
impl PyStabilizerBuilder {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Marker type for the state vector backend.
///
/// Pass to `.quantum()` to select the state vector simulator.
/// Supports arbitrary gates including non-Clifford (T, RZ, etc.).
///
/// Example:
///     sim_neo(tc).quantum(statevec()).noise(depolarizing().idle_rz(0.05)).sampling(monte_carlo(10000)).run()
#[pyclass(
    name = "StateVecBuilder",
    skip_from_py_object,
    module = "pecos_rslib_exp"
)]
#[derive(Clone)]
pub struct PyStateVecBuilder;

#[pymethods]
impl PyStateVecBuilder {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Create a state vector backend builder.
///
/// Example:
///     sim_neo(tc).quantum(statevec()).noise(...).sampling(monte_carlo(10000)).run()
#[pyfunction]
pub fn statevec() -> PyStateVecBuilder {
    PyStateVecBuilder
}

/// Create a stabilizer (SparseStab) backend builder.
///
/// Example:
///     sim_neo(tc).quantum(stabilizer()).noise(...).sampling(monte_carlo(10000)).run()
#[pyfunction]
pub fn stabilizer() -> PyStabilizerBuilder {
    PyStabilizerBuilder
}

// ============================================================================
// StabMps backend builder
// ============================================================================

/// Builder for StabMps backend configuration.
///
/// Example:
///     stab_mps().lazy_measure().max_bond_dim(128)
#[pyclass(
    name = "StabMpsBuilder",
    skip_from_py_object,
    module = "pecos_rslib_exp"
)]
#[derive(Clone)]
pub struct PyStabMpsBuilder {
    pub(crate) inner: crate::stabmps_builder::StabMpsBuilder,
}

#[pymethods]
impl PyStabMpsBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: crate::stabmps_builder::StabMpsBuilder::new(),
        }
    }

    /// Set whether measurements are deferred for non-Clifford states.
    ///
    /// Calling without an argument enables lazy measurement; an explicit bool
    /// sets the option exactly.
    #[pyo3(signature = (enabled=true))]
    fn lazy_measure(mut slf: PyRefMut<'_, Self>, enabled: bool) -> PyRefMut<'_, Self> {
        slf.inner.lazy_measure = enabled;
        slf
    }

    fn max_bond_dim(mut slf: PyRefMut<'_, Self>, bd: usize) -> PyRefMut<'_, Self> {
        slf.inner.max_bond_dim = bd;
        slf
    }

    /// Set the adaptive truncation bound. Negative and non-finite values raise
    /// `ValueError`.
    fn max_truncation_error(mut slf: PyRefMut<'_, Self>, err: f64) -> PyResult<PyRefMut<'_, Self>> {
        if !err.is_finite() || err < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_truncation_error must be finite and non-negative",
            ));
        }
        slf.inner.max_truncation_error = Some(err);
        Ok(slf)
    }

    /// Set whether consecutive RZ rotations on the same qubit are merged.
    ///
    /// Calling without an argument enables RZ merging; an explicit bool sets
    /// the option exactly.
    #[pyo3(signature = (enabled=true))]
    fn merge_rz(mut slf: PyRefMut<'_, Self>, enabled: bool) -> PyRefMut<'_, Self> {
        slf.inner.merge_rz = enabled;
        slf
    }
}

/// Create a StabMps backend builder.
#[pyfunction]
pub fn stab_mps() -> PyStabMpsBuilder {
    PyStabMpsBuilder::new()
}

// ============================================================================
// sim_neo builder
// ============================================================================

/// Measurement sampling backend builder.
#[pyclass(
    name = "MeasSamplingBuilder",
    skip_from_py_object,
    module = "pecos_rslib_exp"
)]
#[derive(Clone)]
pub struct PyMeasSamplingBuilder {
    method: String,
}

#[pymethods]
impl PyMeasSamplingBuilder {
    #[new]
    #[pyo3(signature = (method="auto"))]
    fn new(method: &str) -> Self {
        Self {
            method: method.to_string(),
        }
    }
}

/// Create a measurement sampling backend builder.
///
/// Samples raw measurement rows from a whole-circuit measurement model. Fast, handles coherent noise at any distance.
///
/// Methods:
///   - "auto": uses coherent_dem if idle_rz > 0, else stochastic (default)
///   - "stochastic": DEM from backward Pauli propagation
///   - "coherent": DEM from EEG backward Heisenberg walk
#[pyfunction]
#[pyo3(signature = (method="auto"))]
pub fn meas_sampling(method: &str) -> PyMeasSamplingBuilder {
    PyMeasSamplingBuilder::new(method)
}

/// Monte Carlo sampling strategy builder. Mirrors Rust `monte_carlo(shots)`.
///
/// Example:
///     sim_neo(tc).sampling(monte_carlo(1000).workers(4)).run()
#[pyclass(
    name = "MonteCarloBuilder",
    skip_from_py_object,
    module = "pecos_rslib_exp"
)]
#[derive(Clone)]
pub struct PyMonteCarloBuilder {
    pub(crate) shots: usize,
    pub(crate) workers: usize,
}

#[pymethods]
impl PyMonteCarloBuilder {
    /// Set the number of parallel workers (1 = sequential).
    fn workers(&self, n: usize) -> Self {
        let mut c = self.clone();
        c.workers = n;
        c
    }
}

/// Create a Monte Carlo sampling strategy running `shots` shots.
///
/// Sequential by default; chain `.workers(n)` for parallel execution.
#[pyfunction]
pub fn monte_carlo(shots: usize) -> PyMonteCarloBuilder {
    PyMonteCarloBuilder { shots, workers: 1 }
}

/// Path enumeration strategy builder. Mirrors Rust `path_enumeration(k)`.
///
/// Exhaustively enumerates the measurement branches of a noiseless Clifford
/// circuit. Each distinct realized path becomes one result row; exact path
/// probabilities are exposed via `result.weights`.
///
/// Example:
///     result = sim_neo(tc).quantum(stabilizer()).sampling(path_enumeration(2)).run()
///     for row, p in zip(result, result.weights): ...
#[pyclass(
    name = "PathEnumerationBuilder",
    skip_from_py_object,
    module = "pecos_rslib_exp"
)]
#[derive(Clone)]
pub struct PyPathEnumerationBuilder {
    pub(crate) max_measurements: usize,
}

/// Create a path enumeration strategy covering up to `max_measurements`
/// random measurement branches.
#[pyfunction]
pub fn path_enumeration(max_measurements: usize) -> PyPathEnumerationBuilder {
    PyPathEnumerationBuilder { max_measurements }
}

/// Subset simulation strategy builder. Mirrors Rust `subset_simulation(n)`.
///
/// Estimates rare event probabilities by decomposing them into conditional
/// probabilities across adaptive levels. Requires a `.score(fn)` (how close
/// is this outcome to failure?) and a `.failure(fn)` predicate; both receive
/// the measurement bits of one sample as `list[int]` and are called once per
/// sample (Python-callable cost applies).
///
/// Example:
///     result = (sim_neo(tc)
///         .quantum(stabilizer())
///         .sampling(subset_simulation(1000)
///             .score(lambda bits: float(sum(bits)))
///             .failure(lambda bits: all(bits)))
///         .seed(42)
///         .run())
///     print(result.subset.probability)
#[pyclass(
    name = "SubsetSimulationBuilder",
    skip_from_py_object,
    module = "pecos_rslib_exp"
)]
pub struct PySubsetSimulationBuilder {
    samples_per_level: usize,
    threshold_fraction: f64,
    max_levels: usize,
    min_conditional_prob: f64,
    allow_biased_multilevel: bool,
    score: Option<Py<PyAny>>,
    failure: Option<Py<PyAny>>,
}

impl Clone for PySubsetSimulationBuilder {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            samples_per_level: self.samples_per_level,
            threshold_fraction: self.threshold_fraction,
            max_levels: self.max_levels,
            min_conditional_prob: self.min_conditional_prob,
            allow_biased_multilevel: self.allow_biased_multilevel,
            score: self.score.as_ref().map(|f| f.clone_ref(py)),
            failure: self.failure.as_ref().map(|f| f.clone_ref(py)),
        })
    }
}

#[pymethods]
impl PySubsetSimulationBuilder {
    /// Set the score function: bits (`list[int]`) -> float.
    ///
    /// Higher scores advance to the next level; failing outcomes should
    /// score at least as high as any non-failing outcome.
    fn score(&self, f: Py<PyAny>) -> Self {
        let mut c = self.clone();
        c.score = Some(f);
        c
    }

    /// Set the failure predicate: bits (`list[int]`) -> bool.
    fn failure(&self, f: Py<PyAny>) -> Self {
        let mut c = self.clone();
        c.failure = Some(f);
        c
    }

    /// Fraction of samples that advances past each threshold (default 0.1).
    fn threshold_fraction(&self, fraction: f64) -> Self {
        let mut c = self.clone();
        c.threshold_fraction = fraction;
        c
    }

    /// Maximum number of levels before giving up.
    ///
    /// Defaults to 1 (a single, unbiased direct-Monte-Carlo level).
    /// Setting more than one level engages the multi-level estimator,
    /// which is currently biased upward, and therefore also requires an
    /// explicit `.allow_biased_multilevel()` acknowledgment, or `.run()`
    /// raises.
    fn max_levels(&self, levels: usize) -> Self {
        let mut c = self.clone();
        c.max_levels = levels;
        c
    }

    /// Acknowledge and accept the known upward bias of the multi-level
    /// subset estimator, enabling `max_levels > 1`. Without it, subset
    /// simulation runs a single unbiased level (direct Monte Carlo).
    fn allow_biased_multilevel(&self) -> Self {
        let mut c = self.clone();
        c.allow_biased_multilevel = true;
        c
    }

    /// Minimum conditional probability before declaring the failure event
    /// unreachable (default 1e-6).
    fn min_conditional_prob(&self, p: f64) -> Self {
        let mut c = self.clone();
        c.min_conditional_prob = p;
        c
    }
}

/// Create a subset simulation strategy running `samples_per_level` samples
/// at each level. `.score(..)` and `.failure(..)` are required.
#[pyfunction]
pub fn subset_simulation(samples_per_level: usize) -> PySubsetSimulationBuilder {
    let defaults = pecos_neo::sampling::subset::SubsetConfig::default();
    PySubsetSimulationBuilder {
        samples_per_level,
        threshold_fraction: defaults.threshold_fraction,
        // Default to a single, unbiased level; the biased multi-level path
        // requires an explicit .allow_biased_multilevel() opt-in.
        max_levels: 1,
        min_conditional_prob: defaults.min_conditional_prob,
        allow_biased_multilevel: false,
        score: None,
        failure: None,
    }
}

/// Rare-event estimate from subset simulation. Mirrors Rust `SubsetResult`.
#[pyclass(name = "SubsetResult", module = "pecos_rslib_exp")]
pub struct PySubsetResult {
    inner: pecos_neo::sampling::subset::SubsetResult,
}

#[pymethods]
impl PySubsetResult {
    /// Overall probability estimate.
    #[getter]
    fn probability(&self) -> f64 {
        self.inner.probability()
    }

    /// Coefficient of variation (standard error / estimate).
    #[getter]
    fn coefficient_of_variation(&self) -> f64 {
        self.inner.coefficient_of_variation
    }

    /// Total number of samples run across all levels.
    #[getter]
    fn total_samples(&self) -> usize {
        self.inner.total_samples
    }

    /// Number of failures observed directly.
    #[getter]
    fn direct_failures(&self) -> usize {
        self.inner.direct_failures
    }

    /// 95% confidence interval (assuming log-normal): (lower, upper).
    fn confidence_interval_95(&self) -> (f64, f64) {
        self.inner.confidence_interval_95()
    }

    /// Per-level statistics as a list of dicts.
    fn levels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        use pyo3::types::{PyDict, PyList};
        let list = PyList::empty(py);
        for level in &self.inner.levels {
            let d = PyDict::new(py);
            d.set_item("level", level.level)?;
            d.set_item("threshold", level.threshold)?;
            d.set_item("num_samples", level.num_samples)?;
            d.set_item("num_exceeded", level.num_exceeded)?;
            d.set_item("conditional_prob", level.conditional_prob)?;
            d.set_item("num_failures", level.num_failures)?;
            list.append(d)?;
        }
        Ok(list)
    }
}

/// Sampling strategy selected on the Python builder.
#[derive(Clone)]
enum PySampling {
    MonteCarlo { shots: usize, workers: usize },
    PathEnumeration { max_measurements: usize },
    SubsetSimulation { config: PySubsetSimulationBuilder },
}

/// Builder for sim_neo simulations. Mirrors the Rust-side `SimNeoBuilder`.
#[pyclass(
    name = "SimNeoBuilder",
    skip_from_py_object,
    module = "pecos_rslib_exp"
)]
#[derive(Clone)]
pub struct PySimNeoBuilder {
    commands: pecos_neo::command::CommandQueue,
    /// Original Rust TickCircuit for meas_sampling (avoids reconstruction).
    /// Wrapped in Arc for Clone compatibility with pyo3.
    tick_circuit: std::sync::Arc<pecos_quantum::TickCircuit>,
    /// Sampling strategy. None until `.sampling()`.
    sampling: Option<PySampling>,
    /// Shot count from the deprecated top-level `.shots()` forwarder.
    legacy_shots: Option<usize>,
    /// Random seed. None = nondeterministic, mirroring the Rust builder.
    seed: Option<u64>,
    /// Backend auto-selection opt-in from `.auto()`.
    auto: bool,
    noise_config: Option<PyNoiseModelBuilder>,
    /// Backend name. None until `.quantum()` is called; `.auto()` opts into
    /// automatic selection at run time.
    backend: Option<String>,
    stabmps_config: Option<crate::stabmps_builder::StabMpsBuilder>,
    meas_sampling_method: Option<String>,
}

#[pymethods]
impl PySimNeoBuilder {
    /// Set the quantum backend.
    ///
    /// Accepts:
    ///   - `state_vec()` — state vector (exact, supports non-Clifford gates)
    ///   - `stabilizer()` — SparseStab (fast Clifford-only)
    ///   - `stab_mps()` — hybrid stabilizer-MPS (Clifford + T gates)
    ///
    /// Example:
    ///     sim_neo(tc).quantum(state_vec()).noise(...).run()
    ///     sim_neo(tc).quantum(stabilizer()).noise(...).run()
    ///     sim_neo(tc).quantum(stab_mps().lazy_measure()).noise(...).run()
    fn quantum(&self, builder: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut c = self.clone();
        if builder.is_instance_of::<PyMeasSamplingBuilder>() {
            let b: PyRef<'_, PyMeasSamplingBuilder> = builder.extract()?;
            c.backend = Some("meas_sampling".to_string());
            c.meas_sampling_method = Some(b.method.clone());
            c.stabmps_config = None;
        } else if builder.is_instance_of::<PyStabMpsBuilder>() {
            let b: PyRef<'_, PyStabMpsBuilder> = builder.extract()?;
            c.backend = Some("stabmps".to_string());
            c.stabmps_config = Some(b.inner.clone());
            c.meas_sampling_method = None;
        } else if builder.is_instance_of::<PyStabilizerBuilder>() {
            c.backend = Some("stabilizer".to_string());
            c.stabmps_config = None;
            c.meas_sampling_method = None;
        } else if builder.is_instance_of::<PyStateVecBuilder>() {
            c.backend = Some("statevec".to_string());
            c.stabmps_config = None;
            c.meas_sampling_method = None;
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "quantum() expects statevec(), stabilizer(), stab_mps(), or meas_sampling()",
            ));
        }
        Ok(c)
    }

    /// Opt into automatic selection of unset components.
    ///
    /// Mirrors the Rust builder's `.auto()`: explicit-about-being-implicit.
    /// If `.quantum()` was not called, the backend is selected automatically
    /// (the stabilizer backend; circuits with inline channel operations route
    /// to the density-matrix path instead, since the stabilizer cannot
    /// execute arbitrary channels). The sampling strategy is never
    /// auto-selected: `.sampling(monte_carlo(shots))` is always required.
    fn auto(&self) -> Self {
        let mut c = self.clone();
        c.auto = true;
        c
    }

    /// Set the noise model.
    fn noise(&self, noise_builder: &PyNoiseModelBuilder) -> Self {
        let mut c = self.clone();
        c.noise_config = Some(noise_builder.clone());
        c
    }

    /// Set the sampling strategy (shots and workers live on the sampler).
    ///
    /// Accepts `monte_carlo(shots)` or `path_enumeration(max_measurements)`.
    ///
    /// Example:
    ///     sim_neo(tc).sampling(monte_carlo(1000).workers(4)).run()
    ///     sim_neo(tc).sampling(path_enumeration(2)).run()
    fn sampling(&self, sampler: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut c = self.clone();
        if sampler.is_instance_of::<PyMonteCarloBuilder>() {
            let s: PyRef<'_, PyMonteCarloBuilder> = sampler.extract()?;
            c.sampling = Some(PySampling::MonteCarlo {
                shots: s.shots,
                workers: s.workers,
            });
        } else if sampler.is_instance_of::<PyPathEnumerationBuilder>() {
            let s: PyRef<'_, PyPathEnumerationBuilder> = sampler.extract()?;
            c.sampling = Some(PySampling::PathEnumeration {
                max_measurements: s.max_measurements,
            });
        } else if sampler.is_instance_of::<PySubsetSimulationBuilder>() {
            let s: PyRef<'_, PySubsetSimulationBuilder> = sampler.extract()?;
            c.sampling = Some(PySampling::SubsetSimulation { config: s.clone() });
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "sampling() expects monte_carlo(shots), path_enumeration(max_measurements), \
                 or subset_simulation(samples_per_level)",
            ));
        }
        Ok(c)
    }

    /// Set number of shots.
    ///
    /// Deprecated: use `.sampling(monte_carlo(shots))` instead.
    fn shots(&self, py: Python<'_>, n: usize) -> PyResult<Self> {
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
            c"sim_neo(...).shots(n) is deprecated; use .sampling(monte_carlo(n))",
            1,
        )?;
        let mut c = self.clone();
        c.legacy_shots = Some(n);
        Ok(c)
    }

    /// Set random seed.
    fn seed(&self, s: u64) -> Self {
        let mut c = self.clone();
        c.seed = Some(s);
        c
    }

    /// Run the simulation and return per-shot measurement outcomes.
    ///
    /// All backends return `RawMeasurementResult` which supports:
    /// `result[shot]`, `result.get(shot, meas)`, `len(result)`, iteration.
    fn run(&self, py: Python<'_>) -> PyResult<PyRawMeasurementResult> {
        if self.tick_circuit.has_channel_operations() {
            return self.run_inline_channel_circuit();
        }

        let backend = self.resolved_backend()?;
        if backend == "meas_sampling" {
            return self.run_meas_sampling();
        }

        match self.resolved_sampling()? {
            PySampling::PathEnumeration { max_measurements } => {
                return self.run_path_enumeration(&backend, max_measurements);
            }
            PySampling::SubsetSimulation { config } => {
                return self.run_subset_simulation(py, &backend, &config);
            }
            PySampling::MonteCarlo { .. } => {}
        }
        let (shots, workers) = self.resolved_monte_carlo("this backend")?;

        let noise = self
            .noise_config
            .as_ref()
            .and_then(PyNoiseModelBuilder::build_noise);

        let mut builder = sim_neo(self.commands.clone())
            .sampling(pecos_neo::tool::monte_carlo(shots).workers(workers));

        if let Some(seed) = self.seed {
            builder = builder.seed(seed);
        }

        if let Some(n) = noise {
            builder = builder.noise(n);
        }

        match backend.as_str() {
            "stabmps" => {
                let config = self.stabmps_config.clone().unwrap_or_default();
                builder = builder.quantum(pecos_neo::tool::custom_backend_from_factory(config));
            }
            "statevec" => {
                builder = builder.quantum(pecos_neo::tool::state_vector());
            }
            "stabilizer" => {
                builder = builder.quantum(pecos_neo::tool::sparse_stab());
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown backend: {backend}"
                )));
            }
        }

        let mut sim = builder.build();
        let results = sim.run();

        let mut all_shots = Vec::with_capacity(shots);
        for shot_outcomes in &results.outcomes {
            let meas: Vec<u8> = shot_outcomes.iter().map(|o| u8::from(o.outcome)).collect();
            all_shots.push(meas);
        }

        Ok(PyRawMeasurementResult::from_rows(all_shots))
    }
}

impl PySimNeoBuilder {
    /// Path enumeration: exhaustively enumerate measurement branches.
    ///
    /// Pre-validates with ValueError mirroring the Rust builder's
    /// build-time checks, then runs through the Rust sim_neo builder.
    fn run_path_enumeration(
        &self,
        backend: &str,
        max_measurements: usize,
    ) -> PyResult<PyRawMeasurementResult> {
        if backend != "stabilizer" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Path enumeration currently supports only the stabilizer() backend \
                 (or .auto()).",
            ));
        }
        if self.noise_config.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Path enumeration enumerates measurement branches of the noiseless \
                 circuit; remove .noise().",
            ));
        }
        if max_measurements > 24 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Path enumeration covers 2^max_measurements paths; \
                 max_measurements = {max_measurements} would enumerate more than 16M paths."
            )));
        }

        let results = sim_neo(self.commands.clone())
            .quantum(pecos_neo::tool::sparse_stab())
            .sampling(pecos_neo::tool::path_enumeration(max_measurements))
            .build()
            .run();

        let rows: Vec<Vec<u8>> = results
            .outcomes
            .iter()
            .map(|shot| shot.iter().map(|o| u8::from(o.outcome)).collect())
            .collect();
        let weights: Vec<f64> = results
            .weights
            .as_ref()
            .map(|ws| {
                ws.iter()
                    .map(pecos_neo::sampling::weight::SampleWeight::weight)
                    .collect()
            })
            .unwrap_or_default();

        Ok(PyRawMeasurementResult::from_rows_weighted(rows, weights))
    }

    /// Subset simulation with Python score/failure callables.
    ///
    /// Each callable receives the sample's measurement bits as `list[int]`
    /// and is invoked once per sample on the calling thread. The first
    /// Python exception raised by a callable aborts the run and propagates.
    fn run_subset_simulation(
        &self,
        py: Python<'_>,
        backend: &str,
        config: &PySubsetSimulationBuilder,
    ) -> PyResult<PyRawMeasurementResult> {
        use pecos_neo::outcome::MeasurementOutcomes;
        use pecos_neo::sampling::subset::{SubsetConfig, SubsetSimulation};
        use std::sync::{Arc, Mutex};

        if backend != "stabilizer" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Subset simulation currently supports only the stabilizer() backend \
                 (or .auto()).",
            ));
        }
        let (Some(score), Some(failure)) = (&config.score, &config.failure) else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Subset simulation requires both .score(..) and .failure(..) on the \
                 subset_simulation(..) builder; neither has a sensible default.",
            ));
        };
        if config.max_levels > 1 && !config.allow_biased_multilevel {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "subset_simulation with max_levels > 1 engages the multi-level estimator, \
                 which is currently biased upward (the resample is unconditioned). Either \
                 keep a single level (an unbiased direct-Monte-Carlo failure-fraction \
                 estimate) or call .allow_biased_multilevel() to accept the documented bias.",
            ));
        }

        let noise = self
            .noise_config
            .as_ref()
            .and_then(PyNoiseModelBuilder::build_noise);

        let num_qubits = self
            .commands
            .iter()
            .flat_map(|cmd| cmd.qubits.iter())
            .map(|q| q.0)
            .max()
            .map_or(1, |max| max + 1);

        // Bridge Python callables into Fn closures. Errors cannot propagate
        // through the closure signature, so capture the first one and check
        // after the run. After an error, the closures stop calling Python
        // and report every sample as a failure, which trips the algorithm's
        // all-samples-failed termination at the end of the current level —
        // bounding the wasted work without changing the library API.
        let captured_err: Arc<Mutex<Option<PyErr>>> = Arc::new(Mutex::new(None));
        let bits_of = |outcomes: &MeasurementOutcomes| -> Vec<u8> {
            outcomes.iter().map(|o| u8::from(o.outcome)).collect()
        };
        let has_err = |slot: &Arc<Mutex<Option<PyErr>>>| {
            slot.lock()
                .expect("subset callable error slot poisoned")
                .is_some()
        };

        let score_fn = score.clone_ref(py);
        let score_err = Arc::clone(&captured_err);
        let score_closure = move |outcomes: &MeasurementOutcomes| -> f64 {
            if has_err(&score_err) {
                return 0.0;
            }
            Python::attach(|py| {
                match score_fn
                    .call1(py, (bits_of(outcomes),))
                    .and_then(|v| v.extract::<f64>(py))
                {
                    Ok(v) => v,
                    Err(e) => {
                        score_err
                            .lock()
                            .expect("subset callable error slot poisoned")
                            .get_or_insert(e);
                        0.0
                    }
                }
            })
        };

        let failure_fn = failure.clone_ref(py);
        let failure_err = Arc::clone(&captured_err);
        let failure_closure = move |outcomes: &MeasurementOutcomes| -> bool {
            if has_err(&failure_err) {
                // Steer the run to its all-failed termination condition.
                return true;
            }
            Python::attach(|py| {
                match failure_fn
                    .call1(py, (bits_of(outcomes),))
                    .and_then(|v| v.extract::<bool>(py))
                {
                    Ok(v) => v,
                    Err(e) => {
                        failure_err
                            .lock()
                            .expect("subset callable error slot poisoned")
                            .get_or_insert(e);
                        true
                    }
                }
            })
        };

        let subset_config = SubsetConfig {
            samples_per_level: config.samples_per_level,
            threshold_fraction: config.threshold_fraction,
            max_levels: config.max_levels,
            min_conditional_prob: config.min_conditional_prob,
            seed: self.seed,
        };

        let result = SubsetSimulation::new(
            self.commands.clone(),
            num_qubits,
            score_closure,
            failure_closure,
        )
        .with_noise_builder(move || noise.clone())
        .with_config(subset_config)
        .run();

        if let Some(err) = captured_err
            .lock()
            .expect("subset callable error slot poisoned")
            .take()
        {
            return Err(err);
        }

        let subset = Py::new(py, PySubsetResult { inner: result })?;
        Ok(PyRawMeasurementResult::from_subset(subset))
    }

    /// Resolve the sampling strategy, mirroring the Rust builder's rules
    /// and error messages.
    fn resolved_sampling(&self) -> PyResult<PySampling> {
        match (&self.sampling, self.legacy_shots) {
            (Some(sampling), None) => Ok(sampling.clone()),
            (Some(_), Some(_)) => Err(pyo3::exceptions::PyValueError::new_err(
                "Conflicting sampling configuration: deprecated .shots() cannot be combined \
                 with .sampling(). Set shots on the sampler builder, e.g. \
                 .sampling(monte_carlo(1000)).",
            )),
            (None, Some(shots)) => Ok(PySampling::MonteCarlo { shots, workers: 1 }),
            (None, None) => Err(pyo3::exceptions::PyValueError::new_err(
                "No sampling strategy set. Use .sampling(monte_carlo(shots)).",
            )),
        }
    }

    /// Resolve to (shots, workers) for execution paths that only support
    /// Monte Carlo sampling.
    fn resolved_monte_carlo(&self, path_name: &str) -> PyResult<(usize, usize)> {
        match self.resolved_sampling()? {
            PySampling::MonteCarlo { shots, workers } => Ok((shots, workers)),
            PySampling::PathEnumeration { .. } => {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{path_name} does not support path enumeration; use \
                     .sampling(monte_carlo(shots)) instead."
                )))
            }
            PySampling::SubsetSimulation { .. } => {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{path_name} does not support subset simulation; use \
                     .sampling(monte_carlo(shots)) instead."
                )))
            }
        }
    }

    /// Resolve the quantum backend, mirroring the Rust builder's rules:
    /// explicit `.quantum()` wins; `.auto()` opts into automatic selection;
    /// otherwise fail fast. Auto selects the stabilizer backend, except for
    /// circuits with inline channel operations, which route to the
    /// density-matrix path (the stabilizer cannot execute arbitrary
    /// channels).
    fn resolved_backend(&self) -> PyResult<String> {
        if let Some(backend) = &self.backend {
            return Ok(backend.clone());
        }
        if self.auto {
            let auto_backend = if self.tick_circuit.has_channel_operations() {
                "statevec"
            } else {
                "stabilizer"
            };
            return Ok(auto_backend.to_string());
        }
        Err(pyo3::exceptions::PyValueError::new_err(
            "No quantum backend set. Use .quantum(stabilizer()) or .quantum(statevec()), \
             or call .auto() to let sim_neo choose.",
        ))
    }

    /// Concrete seed for execution paths that require one. Unset seed means
    /// nondeterministic (mirroring the Rust builder), so draw fresh entropy.
    fn resolved_seed_u64(&self) -> u64 {
        use std::hash::{BuildHasher, Hasher};
        self.seed.unwrap_or_else(|| {
            std::collections::hash_map::RandomState::new()
                .build_hasher()
                .finish()
        })
    }

    fn run_inline_channel_circuit(&self) -> PyResult<PyRawMeasurementResult> {
        if self.noise_config.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sim_neo received a TickCircuit with inline channel operations; do not also pass .noise()",
            ));
        }
        let backend = self.resolved_backend()?;
        match backend.as_str() {
            "statevec" | "stabilizer" => {}
            "stabmps" => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "stab_mps backend does not support inline channel operations; use statevec() for density-matrix execution",
                ));
            }
            "meas_sampling" => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "meas_sampling backend builds its own measurement model and does not consume inline channel operations",
                ));
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown backend: {other}"
                )));
            }
        }
        let (shots, workers) = self.resolved_monte_carlo("inline-channel execution")?;
        if workers > 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "inline-channel execution does not support parallel workers; use monte_carlo(shots) without .workers()",
            ));
        }
        let seed = self.resolved_seed_u64();

        match backend.as_str() {
            "statevec" => self.run_inline_channel_density_matrix(shots, seed),
            _ => self.run_inline_pauli_channel_stabilizer(shots, seed),
        }
    }

    fn run_inline_channel_density_matrix(
        &self,
        shots: usize,
        seed: u64,
    ) -> PyResult<PyRawMeasurementResult> {
        let rows = pecos_neo::inline_channel::run_inline_channels_density_matrix(
            &self.tick_circuit,
            shots,
            seed,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyRawMeasurementResult::from_rows(rows))
    }

    fn run_inline_pauli_channel_stabilizer(
        &self,
        shots: usize,
        seed: u64,
    ) -> PyResult<PyRawMeasurementResult> {
        let rows = pecos_neo::inline_channel::run_inline_pauli_channels_stabilizer(
            &self.tick_circuit,
            shots,
            seed,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyRawMeasurementResult::from_rows(rows))
    }

    /// DEM sampling backend: dispatches to stochastic or coherent path based on method.
    fn run_meas_sampling(&self) -> PyResult<PyRawMeasurementResult> {
        let (_, workers) = self.resolved_monte_carlo("meas_sampling")?;
        if workers > 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "meas_sampling does its own batch sampling and does not support parallel workers; use monte_carlo(shots) without .workers()",
            ));
        }
        let noise_config = self.noise_config.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("DEM sampling requires .noise() to be set")
        })?;

        let method = self.meas_sampling_method.as_deref().unwrap_or("auto");

        let has_coherent = noise_config.idle_rz_angle.abs() > 1e-15;

        match method {
            "stochastic" => {
                if has_coherent {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "DEM sampling method='stochastic' cannot handle idle_rz noise. \
                         Use method='coherent' or method='auto'.",
                    ));
                }
                self.run_stochastic_meas_columnar()
            }
            "coherent" | "coherent_approx" | "coherent_exact" => {
                let rows = self.run_coherent_meas_sampling(noise_config, method)?;
                Ok(PyRawMeasurementResult::from_rows(rows))
            }
            "auto" => {
                if has_coherent {
                    let rows = self.run_coherent_meas_sampling(noise_config, "coherent_approx")?;
                    Ok(PyRawMeasurementResult::from_rows(rows))
                } else {
                    self.run_stochastic_meas_columnar()
                }
            }
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown DEM sampling method: {other:?}. \
                     Use 'auto', 'stochastic', 'coherent', 'coherent_approx', or 'coherent_exact'."
            ))),
        }
    }

    /// Stochastic path: columnar raw-measurement sampling.
    fn run_stochastic_meas_columnar(&self) -> PyResult<PyRawMeasurementResult> {
        use pecos_qec::fault_tolerance::fault_sampler::{
            self, RawMeasurementPlan, StochasticNoiseParams,
        };

        let noise_config = self.noise_config.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("DEM sampling requires .noise() to be set")
        })?;

        let history = run_symbolic_sim_with_pz(&self.tick_circuit)?;

        let noise = StochasticNoiseParams {
            p1: noise_config.p1,
            p2: noise_config.p2,
            p_meas: noise_config.p_meas,
            p_prep: noise_config.p_prep,
        };
        let mechanisms = fault_sampler::build_fault_table(&self.tick_circuit, &noise)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let plan = RawMeasurementPlan::new(&history, mechanisms);
        let (shots, _) = self.resolved_monte_carlo("meas_sampling")?;
        let result = plan.sample(shots, self.resolved_seed_u64());

        Ok(PyRawMeasurementResult::from_columnar(result))
    }

    /// Coherent path: EEG DemGenerator with measurement synthesis.
    fn run_coherent_meas_sampling(
        &self,
        noise_config: &PyNoiseModelBuilder,
        method: &str,
    ) -> PyResult<Vec<Vec<u8>>> {
        use pecos_eeg::dem_generator::select_generator;
        use pecos_eeg::dem_simulator::{CircuitMeasurementMeta, run_dem_simulation};

        // Extract metadata from stored TickCircuit
        let num_meas_attr = self
            .tick_circuit
            .get_meta("num_measurements")
            .and_then(|a| {
                if let pecos_quantum::Attribute::String(s) = a {
                    s.parse::<usize>().ok()
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "TickCircuit missing num_measurements metadata",
                )
            })?;
        let det_json = self
            .tick_circuit
            .get_meta("detectors")
            .and_then(|a| {
                if let pecos_quantum::Attribute::String(s) = a {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "[]".to_string());
        let obs_json = self
            .tick_circuit
            .get_meta("observables")
            .and_then(|a| {
                if let pecos_quantum::Attribute::String(s) = a {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "[]".to_string());

        let det_records: Vec<Vec<i32>> = serde_json::from_str::<Vec<RecDef>>(&det_json)
            .map(|defs| defs.iter().map(|d| d.records.clone()).collect())
            .unwrap_or_default();
        let obs_records: Vec<Vec<i32>> = serde_json::from_str::<Vec<RecDef>>(&obs_json)
            .map(|defs| defs.iter().map(|d| d.records.clone()).collect())
            .unwrap_or_default();

        let meta = CircuitMeasurementMeta {
            num_measurements: num_meas_attr,
            detector_records: det_records,
            observable_records: obs_records,
        };

        let noise = pecos_eeg::noise::UniformNoise {
            idle_rz: noise_config.idle_rz_angle,
            p1: noise_config.p1,
            p2: noise_config.p2,
            p_meas: noise_config.p_meas,
            p_prep: noise_config.p_prep,
        };

        let gates = commands_to_gates(&self.commands);
        let generator = select_generator(method, noise_config.idle_rz_angle);

        let (shots, _) = self.resolved_monte_carlo("meas_sampling")?;
        let result = run_dem_simulation(
            &gates,
            &noise,
            &meta,
            generator.as_ref(),
            shots,
            self.resolved_seed_u64(),
        )
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
        Ok(result.measurements)
    }
}

/// Convert CommandQueue to Vec<Gate> for EEG analysis.
fn commands_to_gates(commands: &pecos_neo::command::CommandQueue) -> Vec<pecos_core::Gate> {
    use pecos_core::{GateAngles, GateMeasIds, GateParams};

    commands
        .iter()
        .map(|cmd| {
            let qubits = cmd.qubits.iter().copied().collect();
            let mut angles = GateAngles::new();
            for &a in &cmd.angles {
                angles.push(a);
            }
            // Convert pecos_neo::GateType to pecos_core::GateType
            let gate_type: pecos_core::gate_type::GateType = cmd.gate_type.into();
            Gate {
                gate_type,
                qubits,
                angles,
                params: GateParams::new(),
                meas_ids: GateMeasIds::new(),
                channel: None,
            }
        })
        .collect()
}

// ============================================================================
// Entry point
// ============================================================================

/// Create a sim_neo simulation builder from a TickCircuit.
///
/// Explicit-by-default, mirroring the Rust builder: a quantum backend
/// (`.quantum(..)` or `.auto()`) and a sampling strategy
/// (`.sampling(monte_carlo(shots))`) are required; missing either is an
/// error at `.run()`. The seed is optional; unset means nondeterministic.
///
/// Example:
///     results = (sim_neo(tc)
///         .quantum(stab_mps().lazy_measure().max_bond_dim(128))
///         .noise(depolarizing().p1(0.003).p2(0.003).p_meas(0.003).idle_rz(0.05))
///         .sampling(monte_carlo(5000))
///         .seed(42)
///         .run())
#[pyfunction]
#[pyo3(name = "sim_neo")]
pub fn py_sim_neo(tick_circuit: &Bound<'_, PyAny>) -> PyResult<PySimNeoBuilder> {
    // Build a Rust TickCircuit from the Python object.
    // This is the canonical circuit representation used by DemSampler.
    let tc = build_rust_tick_circuit(tick_circuit)?;
    let commands = if tc.has_channel_operations() {
        pecos_neo::command::CommandQueue::new()
    } else {
        extract_commands(tick_circuit)?
    };

    Ok(PySimNeoBuilder {
        commands,
        tick_circuit: std::sync::Arc::new(tc),
        sampling: None,
        legacy_shots: None,
        seed: None,
        auto: false,
        noise_config: None,
        backend: None,
        stabmps_config: None,
        meas_sampling_method: None,
    })
}

/// Build a proper Rust TickCircuit from a Python TickCircuit object.
///
/// First tries to extract the inner Rust TickCircuit directly (fast path).
/// Falls back to rebuilding from Python gate iteration (slow path).
fn build_rust_tick_circuit(py_tc: &Bound<'_, PyAny>) -> PyResult<pecos_quantum::TickCircuit> {
    // Fast path: try to access the inner TickCircuit directly.
    // The Python TickCircuit wraps `pub inner: TickCircuit` — access via
    // the `_inner_tick_circuit()` method if available, or via serialization.
    if let Ok(tc_bytes) = py_tc.call_method0("_serialize_inner")
        && let Ok(bytes) = tc_bytes.extract::<Vec<u8>>()
    {
        // Deserialize — but TickCircuit doesn't impl serde. Skip.
        let _ = bytes;
    }

    // The only reliable fast path: call `to_dag_circuit()` on the Python TC,
    // then use DemSampler::from_circuit on that DagCircuit. But we can't get
    // the DagCircuit across crate boundaries easily.
    //
    // For now: reconstruct via gate iteration (matches original structure if
    // we respect tick boundaries from the Python object).
    build_rust_tick_circuit_from_gates(py_tc)
}

/// Reconstruct TickCircuit from Python gate iteration, preserving tick structure.
///
/// Respects the original tick boundaries: all gates from the same Python tick
/// go into the same Rust tick. Uses typed .mz() for measurements and .pz() for
/// prep within each tick (these consume the TickHandle, so we process them after
/// other gates in the tick).
fn build_rust_tick_circuit_from_gates(
    py_tc: &Bound<'_, PyAny>,
) -> PyResult<pecos_quantum::TickCircuit> {
    use pecos_quantum::{Attribute, TickMeasRef};

    let num_ticks: usize = py_tc.call_method0("num_ticks")?.extract()?;
    let mut tc = pecos_quantum::TickCircuit::default();
    let mut all_meas_refs: Vec<TickMeasRef> = Vec::new();

    for tick_idx in 0..num_ticks {
        let py_tick = py_tc.call_method1("get_tick", (tick_idx,))?;
        let py_gates = py_tick.call_method0("gate_batches")?;
        let gates: Vec<Bound<'_, PyAny>> = py_gates.extract()?;

        // Separate gates by type: MZ, PZ, and other
        let mut mz_qubits: Vec<pecos_core::QubitId> = Vec::new();
        let mut pz_qubits: Vec<pecos_core::QubitId> = Vec::new();
        let mut other_gates: Vec<pecos_core::Gate> = Vec::new();

        for gate in &gates {
            let gate_type_obj = gate.getattr("gate_type")?;
            let gate_name: String = format!("{gate_type_obj:?}");
            let gate_name = gate_name
                .split('.')
                .next_back()
                .unwrap_or(&gate_name)
                .to_string();
            let py_qubits = gate.getattr("qubits")?;
            let qubits: Vec<usize> = py_qubits.extract()?;
            let qubit_ids: Vec<pecos_core::QubitId> =
                qubits.iter().map(|&q| pecos_core::QubitId(q)).collect();

            match gate_name.as_str() {
                // MeasureFree lowers to MZ here: record-bearing, and the free
                // has no stabilizer effect. The expansion accepts it either way.
                "MZ" | "Measure" | "MeasureFree" => {
                    mz_qubits.extend(qubit_ids);
                }

                "QAlloc" | "PZ" | "Prep" => {
                    pz_qubits.extend(qubit_ids);
                }
                "TrackedPauli" | "TrackedPauliMeta" => {}
                _ => {
                    let core_gate = build_gate_from_python(gate, &gate_name, &qubit_ids)?;
                    other_gates.push(core_gate);
                }
            }
        }

        // Add PZ first (prep before other gates)
        if !pz_qubits.is_empty() {
            tc.tick().pz(&pz_qubits);
        }

        // Add other gates in one tick (error on qubit conflicts)
        if !other_gates.is_empty() {
            let mut tick_handle = tc.tick();
            for g in &other_gates {
                if let Err(e) = tick_handle.try_add_gate(g.clone()) {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Gate conflict in tick {tick_idx}: {e}"
                    )));
                }
            }
        }

        // Add MZ last (measure after other gates)
        if !mz_qubits.is_empty() {
            let refs = tc.tick().mz(&mz_qubits);
            all_meas_refs.extend(refs);
        }
    }

    // Copy metadata from Python TickCircuit
    if let Ok(num_meas) = py_tc.call_method1("get_meta", ("num_measurements",))
        && let Ok(s) = num_meas.extract::<String>()
    {
        tc.set_meta("num_measurements", Attribute::String(s));
    }
    if let Ok(det_json) = py_tc.call_method1("get_meta", ("detectors",))
        && let Ok(s) = det_json.extract::<String>()
    {
        // Create structured annotations from JSON
        create_annotations_from_json(&mut tc, &s, &all_meas_refs, true)?;
        tc.set_meta("detectors", Attribute::String(s));
    }
    if let Ok(obs_json) = py_tc.call_method1("get_meta", ("observables",))
        && let Ok(s) = obs_json.extract::<String>()
    {
        create_annotations_from_json(&mut tc, &s, &all_meas_refs, false)?;
        tc.set_meta("observables", Attribute::String(s));
    }
    copy_tracked_pauli_annotations_from_python(py_tc, &mut tc)?;

    // Compact for performance
    tc.compact_ticks();

    Ok(tc)
}

fn copy_tracked_pauli_annotations_from_python(
    py_tc: &pyo3::Bound<'_, pyo3::PyAny>,
    tc: &mut pecos_quantum::TickCircuit,
) -> PyResult<()> {
    let Ok(annotations) = py_tc.call_method0("annotations") else {
        return Ok(());
    };

    for ann in annotations.try_iter()? {
        let ann = ann?;
        let kind: String = ann.get_item("kind")?.extract()?;
        if kind != "tracked_pauli" {
            continue;
        }
        let pauli_obj = ann.get_item("pauli")?;
        let pauli_text = pauli_obj.str()?.to_string();
        let pauli = parse_python_pauli_string(&pauli_text).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Could not parse tracked Pauli annotation: {pauli_text}"
            ))
        })?;
        let label: Option<String> = ann.get_item("label")?.extract()?;
        if let Some(label) = label {
            tc.tracked_pauli_labeled(&label, pauli);
        } else {
            tc.tracked_pauli(pauli);
        }
    }

    Ok(())
}

fn parse_python_pauli_string(text: &str) -> Option<PauliString> {
    let text = text.trim();
    let text = text
        .strip_prefix("+i*")
        .or_else(|| text.strip_prefix("-i*"))
        .or_else(|| text.strip_prefix('-'))
        .unwrap_or(text)
        .trim();
    if text.is_empty() || text == "I" {
        return Some(PauliString::new());
    }

    let mut paulis = Vec::new();
    for token in text.split_whitespace() {
        let mut chars = token.chars();
        let p = match chars.next()? {
            'X' | 'x' => pecos_core::Pauli::X,
            'Y' | 'y' => pecos_core::Pauli::Y,
            'Z' | 'z' => pecos_core::Pauli::Z,
            'I' | 'i' => continue,
            _ => return None,
        };
        let rest = chars.as_str().strip_prefix('_').unwrap_or(chars.as_str());
        let qubit = rest.parse::<usize>().ok()?;
        paulis.push((p, pecos_core::QubitId(qubit)));
    }

    Some(PauliString::with_phase_and_paulis(
        pecos_core::QuarterPhase::PlusOne,
        paulis,
    ))
}

fn channel_expr_from_python_gate(gate: &Bound<'_, PyAny>) -> PyResult<ChannelExpr> {
    let terms: Vec<(f64, Vec<(String, usize)>)> =
        gate.call_method0("channel_mixed_pauli_terms")?.extract()?;
    let mut ops = Vec::with_capacity(terms.len());
    for (probability, terms) in terms {
        let mut paulis = Vec::with_capacity(terms.len());
        for (label, qubit) in terms {
            let pauli = match label.as_str() {
                "I" => continue,
                "X" => Pauli::X,
                "Y" => Pauli::Y,
                "Z" => Pauli::Z,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unsupported channel Pauli label {other:?}"
                    )));
                }
            };
            paulis.push((pauli, QubitId(qubit)));
        }
        let pauli_string = PauliString::with_phase_and_paulis(QuarterPhase::PlusOne, paulis);
        ops.push((probability, pecos_core::UnitaryRep::from(pauli_string)));
    }
    Ok(ChannelExpr::MixedUnitary(ops))
}

/// Create detector or observable annotations from JSON metadata.
/// Malformed JSON and unresolvable record offsets are errors: a dropped or
/// thinned annotation silently weakens the DEM.
fn create_annotations_from_json(
    tc: &mut pecos_quantum::TickCircuit,
    json_str: &str,
    all_meas_refs: &[pecos_quantum::TickMeasRef],
    is_detector: bool,
) -> PyResult<()> {
    let kind = if is_detector {
        "detector"
    } else {
        "observable"
    };
    let num_meas = all_meas_refs.len();
    let defs: Vec<RecDef> = serde_json::from_str(json_str).map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!("malformed {kind} JSON metadata: {err}"))
    })?;
    let ref_by_id: std::collections::BTreeMap<usize, pecos_quantum::TickMeasRef> = all_meas_refs
        .iter()
        .map(|r| (r.meas_id.index(), *r))
        .collect();
    for (def_idx, def) in defs.iter().enumerate() {
        if def.records.is_empty() && def.meas_ids.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{kind} {def_idx} carries neither records nor meas_ids"
            )));
        }
        let mut refs: Vec<pecos_quantum::TickMeasRef> =
            Vec::with_capacity(def.records.len() + def.meas_ids.len());
        for &rec in &def.records {
            let mref = measurement_record_index(rec, num_meas)
                .and_then(|abs_idx| all_meas_refs.get(abs_idx).copied())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "{kind} {def_idx} references record offset {rec}, which does \
                         not resolve among the circuit's {num_meas} measurement(s)"
                    ))
                })?;
            refs.push(mref);
        }
        for &meas_id in &def.meas_ids {
            let mref = ref_by_id.get(&meas_id).copied().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "{kind} {def_idx} references MeasId({meas_id}), which no \
                     measurement in the circuit holds"
                ))
            })?;
            refs.push(mref);
        }
        let result = if is_detector {
            tc.detector(&refs)
        } else {
            tc.observable(&refs)
        };
        result.map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    }
    Ok(())
}

/// Build a pecos_core::Gate from a Python gate object.
fn build_gate_from_python(
    gate: &Bound<'_, PyAny>,
    gate_name: &str,
    qubit_ids: &[pecos_core::QubitId],
) -> PyResult<pecos_core::Gate> {
    use pecos_core::gate_type::GateType;
    use pecos_core::{Gate, GateAngles, GateMeasIds, GateParams};

    if gate_name == "Channel" {
        return Ok(Gate::channel(channel_expr_from_python_gate(gate)?));
    }

    let gate_type = match gate_name {
        "H" => GateType::H,
        "X" => GateType::X,
        "Y" => GateType::Y,
        "Z" => GateType::Z,
        "F" => GateType::F,
        "Fdg" => GateType::Fdg,
        "CX" | "CNOT" => GateType::CX,
        "CY" => GateType::CY,
        "CZ" => GateType::CZ,
        "SZ" | "S" => GateType::SZ,
        "SZdg" | "Sdg" => GateType::SZdg,
        "SX" => GateType::SX,
        "SXdg" => GateType::SXdg,
        "SY" => GateType::SY,
        "SYdg" => GateType::SYdg,
        "T" => GateType::T,
        "Tdg" => GateType::Tdg,
        "SWAP" => GateType::SWAP,
        "RZ" => GateType::RZ,
        "RX" => GateType::RX,
        "RY" => GateType::RY,
        "RZZ" => GateType::RZZ,
        "RXX" => GateType::RXX,
        "RYY" => GateType::RYY,
        "SZZ" => GateType::SZZ,
        "SZZdg" => GateType::SZZdg,
        "SXX" => GateType::SXX,
        "SXXdg" => GateType::SXXdg,
        "SYY" => GateType::SYY,
        "SYYdg" => GateType::SYYdg,
        "R1XY" => GateType::R1XY,
        "I" | "Idle" => GateType::I,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported gate type for meas_sampling simulation: {other}"
            )));
        }
    };

    let mut angles = GateAngles::new();
    if let Ok(py_angle) = gate.getattr("angle")
        && let Ok(a) = py_angle.extract::<f64>()
    {
        angles.push(pecos_core::Angle64::from_radians(a));
    }
    if let Ok(py_angles) = gate.getattr("angles")
        && let Ok(a_list) = py_angles.extract::<Vec<f64>>()
    {
        for a in a_list {
            angles.push(pecos_core::Angle64::from_radians(a));
        }
    }

    Ok(Gate {
        gate_type,
        qubits: qubit_ids.iter().copied().collect(),
        angles,
        params: GateParams::new(),
        meas_ids: GateMeasIds::new(),
        channel: None,
    })
}

// ============================================================================
// Circuit extraction
// ============================================================================

/// Extract a CommandQueue from a Python TickCircuit by iterating its stored gate batches.
fn extract_commands(py_tc: &Bound<'_, PyAny>) -> PyResult<pecos_neo::command::CommandQueue> {
    let num_ticks: usize = py_tc.call_method0("num_ticks")?.extract()?;
    let mut cb = CommandBuilder::new();

    for tick_idx in 0..num_ticks {
        let py_tick = py_tc.call_method1("get_tick", (tick_idx,))?;
        let py_gates = py_tick.call_method0("gate_batches")?;
        let gates: Vec<Bound<'_, PyAny>> = py_gates.extract()?;

        for gate in &gates {
            let gate_type_obj = gate.getattr("gate_type")?;
            let name: String = gate_type_obj.getattr("name")?.extract()?;
            let qubits: Vec<usize> = gate.getattr("qubits")?.extract()?;

            match name.as_str() {
                "QAlloc" | "PZ" => {
                    cb = cb.pz(&qubits);
                }
                "H" => {
                    cb = cb.h(&qubits);
                }
                "F" => {
                    cb = cb.f(&qubits);
                }
                "Fdg" => {
                    cb = cb.fdg(&qubits);
                }
                "X" => {
                    cb = cb.x(&qubits);
                }
                "Y" => {
                    cb = cb.y(&qubits);
                }
                "Z" => {
                    cb = cb.z(&qubits);
                }
                "SZ" => {
                    cb = cb.sz(&qubits);
                }
                "SZdg" => {
                    cb = cb.szdg(&qubits);
                }
                "SX" => {
                    cb = cb.sx(&qubits);
                }
                "SXdg" => {
                    cb = cb.sxdg(&qubits);
                }
                "SY" => {
                    cb = cb.sy(&qubits);
                }
                "SYdg" => {
                    cb = cb.sydg(&qubits);
                }
                "CX" => {
                    let pairs: Vec<(usize, usize)> =
                        qubits.chunks(2).map(|c| (c[0], c[1])).collect();
                    cb = cb.cx(&pairs);
                }
                "CY" => {
                    let pairs: Vec<(usize, usize)> =
                        qubits.chunks(2).map(|c| (c[0], c[1])).collect();
                    cb = cb.cy(&pairs);
                }
                "CZ" => {
                    let pairs: Vec<(usize, usize)> =
                        qubits.chunks(2).map(|c| (c[0], c[1])).collect();
                    cb = cb.cz(&pairs);
                }
                "SZZ" => {
                    let pairs: Vec<(usize, usize)> =
                        qubits.chunks(2).map(|c| (c[0], c[1])).collect();
                    cb = cb.szz(&pairs);
                }
                "SZZdg" => {
                    let pairs: Vec<(usize, usize)> =
                        qubits.chunks(2).map(|c| (c[0], c[1])).collect();
                    cb = cb.szzdg(&pairs);
                }
                "SXX" => {
                    let pairs: Vec<(usize, usize)> =
                        qubits.chunks(2).map(|c| (c[0], c[1])).collect();
                    cb = cb.sxx(&pairs);
                }
                "SXXdg" => {
                    let pairs: Vec<(usize, usize)> =
                        qubits.chunks(2).map(|c| (c[0], c[1])).collect();
                    cb = cb.sxxdg(&pairs);
                }
                "SYY" => {
                    let pairs: Vec<(usize, usize)> =
                        qubits.chunks(2).map(|c| (c[0], c[1])).collect();
                    cb = cb.syy(&pairs);
                }
                "SYYdg" => {
                    let pairs: Vec<(usize, usize)> =
                        qubits.chunks(2).map(|c| (c[0], c[1])).collect();
                    cb = cb.syydg(&pairs);
                }
                "SWAP" => {
                    let pairs: Vec<(usize, usize)> =
                        qubits.chunks(2).map(|c| (c[0], c[1])).collect();
                    cb = cb.swap(&pairs);
                }
                "T" => {
                    cb = cb.t(&qubits);
                }
                "Tdg" => {
                    cb = cb.tdg(&qubits);
                }
                "MZ" | "Measure" | "MeasureFree" => {
                    // Z-basis measurement; ``MeasureFree`` additionally frees the
                    // qubit, which is a no-op for the fixed-width simulator. Kept
                    // consistent with build_rust_tick_circuit_from_gates and the
                    // surface DEM path, which treat all three as Z measurements.
                    cb = cb.mz(&qubits);
                }
                "RX" => {
                    let angles: Vec<f64> = gate.getattr("angles")?.extract().unwrap_or_default();
                    if let Some(&angle) = angles.first() {
                        cb = cb.rx(&qubits, Angle64::from_radians(angle));
                    }
                }
                "RY" => {
                    let angles: Vec<f64> = gate.getattr("angles")?.extract().unwrap_or_default();
                    if let Some(&angle) = angles.first() {
                        cb = cb.ry(&qubits, Angle64::from_radians(angle));
                    }
                }
                "RZ" => {
                    let angles: Vec<f64> = gate.getattr("angles")?.extract().unwrap_or_default();
                    if let Some(&angle) = angles.first() {
                        cb = cb.rz(&qubits, Angle64::from_radians(angle));
                    }
                }
                "R1XY" => {
                    let angles: Vec<f64> = gate.getattr("angles")?.extract().unwrap_or_default();
                    if angles.len() >= 2 {
                        cb = cb.r1xy(
                            &qubits,
                            Angle64::from_radians(angles[0]),
                            Angle64::from_radians(angles[1]),
                        );
                    }
                }
                "RZZ" => {
                    let angles: Vec<f64> = gate.getattr("angles")?.extract().unwrap_or_default();
                    if let Some(&angle) = angles.first() {
                        let pairs: Vec<(usize, usize)> =
                            qubits.chunks(2).map(|c| (c[0], c[1])).collect();
                        cb = cb.rzz(&pairs, Angle64::from_radians(angle));
                    }
                }
                "RXX" => {
                    let angles: Vec<f64> = gate.getattr("angles")?.extract().unwrap_or_default();
                    if let Some(&angle) = angles.first() {
                        let pairs: Vec<(usize, usize)> =
                            qubits.chunks(2).map(|c| (c[0], c[1])).collect();
                        cb = cb.rxx(&pairs, Angle64::from_radians(angle));
                    }
                }
                "RYY" => {
                    let angles: Vec<f64> = gate.getattr("angles")?.extract().unwrap_or_default();
                    if let Some(&angle) = angles.first() {
                        let pairs: Vec<(usize, usize)> =
                            qubits.chunks(2).map(|c| (c[0], c[1])).collect();
                        cb = cb.ryy(&pairs, Angle64::from_radians(angle));
                    }
                }
                "I" | "Idle" | "TrackedPauli" | "TrackedPauliMeta" => {
                    // Identity/Idle and tracked-Pauli metadata gates: skip (no-op for simulation)
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unsupported gate type '{name}' in extract_commands. \
                         Add support in sim_neo_bindings.rs or lower to supported gates \
                         with tc.lower_clifford_rotations()."
                    )));
                }
            }
        }
    }

    Ok(cb.build())
}

/// Run SymbolicSparseStab through a TickCircuit with proper PZ (reset) semantics.
///
/// Iterates tick-by-tick to match the TickCircuit's measurement numbering,
/// which is what detector and observable definitions reference.
fn run_symbolic_sim_with_pz(
    tc: &pecos_quantum::TickCircuit,
) -> PyResult<pecos_simulators::symbolic_sparse_stab::MeasurementHistory> {
    pecos_qec::fault_tolerance::fault_sampler::symbolic_measurement_history(tc)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

// ============================================================================
// Fault Catalog Python API
// ============================================================================

/// One fault alternative at a physical location.
#[pyclass(name = "FaultAlternative", module = "pecos_rslib_exp")]
pub struct PyFaultAlternative {
    /// "pauli", "measurement_flip", or "prep_flip"
    #[pyo3(get)]
    kind: String,
    /// PauliString object (from pecos.quantum) for Pauli faults, None otherwise
    pauli_obj: Py<PyAny>,
    /// Measurement indices flipped
    #[pyo3(get)]
    measurements: Vec<usize>,
    /// Detector indices flipped
    #[pyo3(get)]
    detectors: Vec<usize>,
    /// Observable indices flipped
    #[pyo3(get)]
    observables: Vec<usize>,
    /// Tracked-Pauli indices flipped
    #[pyo3(get)]
    tracked_paulis: Vec<usize>,
    /// Probability of this alternative given the mechanism fires (1/k)
    #[pyo3(get)]
    conditional_probability: f64,
    /// Marginal per-location alternative probability: p_i / k_i.
    /// This is NOT "probability of this fault and no others." Full configuration
    /// probabilities require multiplying by no_fault_probability for all other locations.
    #[pyo3(get)]
    absolute_probability: f64,
    /// Total channel probability (same as parent location)
    #[pyo3(get)]
    channel_probability: f64,
}

#[pymethods]
impl PyFaultAlternative {
    #[getter]
    fn pauli(&self, py: Python<'_>) -> Py<PyAny> {
        self.pauli_obj.clone_ref(py)
    }
}

/// A physical fault location in the circuit.
#[pyclass(name = "FaultLocation", module = "pecos_rslib_exp")]
pub struct PyFaultLocation {
    #[pyo3(get)]
    tick: usize,
    #[pyo3(get)]
    gate_index: usize,
    #[pyo3(get)]
    gate_type: String,
    #[pyo3(get)]
    qubits: Vec<usize>,
    /// "p1", "p2", "p_meas", or "p_prep"
    #[pyo3(get)]
    channel: String,
    #[pyo3(get)]
    channel_probability: f64,
    /// 1 - channel_probability
    #[pyo3(get)]
    no_fault_probability: f64,
    #[pyo3(get)]
    num_alternatives: usize,
    #[pyo3(get)]
    faults: Vec<Py<PyFaultAlternative>>,
}

/// A k-fault configuration yielded by `catalog.fault_configurations(k)`.
#[pyclass(name = "FaultConfiguration", module = "pecos_rslib_exp")]
pub struct PyFaultConfiguration {
    #[pyo3(get)]
    location_indices: Vec<usize>,
    #[pyo3(get)]
    alternative_indices: Vec<usize>,
    /// The FaultLocation objects for selected locations.
    #[pyo3(get)]
    locations: Vec<Py<PyFaultLocation>>,
    /// The FaultAlternative objects for selected alternatives.
    #[pyo3(get)]
    faults: Vec<Py<PyFaultAlternative>>,
    #[pyo3(get)]
    measurements: Vec<usize>,
    #[pyo3(get)]
    detectors: Vec<usize>,
    #[pyo3(get)]
    observables: Vec<usize>,
    #[pyo3(get)]
    tracked_paulis: Vec<usize>,
    #[pyo3(get)]
    selected_probability: f64,
    #[pyo3(get)]
    configuration_probability: f64,
}

/// Lazy Python iterator over k-fault configurations.
#[pyclass(name = "FaultConfigurationIter", module = "pecos_rslib_exp")]
pub struct PyFaultConfigurationIter {
    /// Owned Rust iterator (self-contained, no borrows).
    inner: pecos_qec::fault_tolerance::fault_sampler::OwnedFaultConfigIter,
    /// Python-side location objects for building yielded configs.
    py_locations: Vec<Py<PyFaultLocation>>,
}

#[pymethods]
impl PyFaultConfigurationIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<PyFaultConfiguration> {
        let config = self.inner.next()?;

        // Build .locations and .faults references
        let locations: Vec<Py<PyFaultLocation>> = config
            .location_indices
            .iter()
            .map(|&i| self.py_locations[i].clone_ref(py))
            .collect();
        let faults: Vec<Py<PyFaultAlternative>> = config
            .location_indices
            .iter()
            .zip(config.alternative_indices.iter())
            .map(|(&loc_i, &alt_i)| {
                let loc = self.py_locations[loc_i].borrow(py);
                loc.faults[alt_i].clone_ref(py)
            })
            .collect();

        Some(PyFaultConfiguration {
            location_indices: config.location_indices,
            alternative_indices: config.alternative_indices,
            locations,
            faults,
            measurements: config.affected_measurements,
            detectors: config.affected_detectors,
            observables: config.affected_observables,
            tracked_paulis: config.affected_tracked_paulis,
            selected_probability: config.selected_probability,
            configuration_probability: config.configuration_probability,
        })
    }
}

/// Complete fault catalog for a circuit and noise model.
#[pyclass(name = "FaultCatalog", module = "pecos_rslib_exp")]
pub struct PyFaultCatalog {
    /// Physical fault locations in the structural catalog.
    #[pyo3(get)]
    locations: Vec<Py<PyFaultLocation>>,
    /// Rust-side catalog for iterator support.
    rust_catalog: pecos_qec::fault_tolerance::fault_sampler::FaultCatalog,
}

fn stochastic_params_from_inputs(
    noise: Option<&PyNoiseModelBuilder>,
    p1: Option<f64>,
    p2: Option<f64>,
    p_meas: Option<f64>,
    p_prep: Option<f64>,
) -> pecos_qec::fault_tolerance::fault_sampler::StochasticNoiseParams {
    let mut params = noise.map_or(
        pecos_qec::fault_tolerance::fault_sampler::StochasticNoiseParams {
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.0,
        },
        |noise| pecos_qec::fault_tolerance::fault_sampler::StochasticNoiseParams {
            p1: noise.p1,
            p2: noise.p2,
            p_meas: noise.p_meas,
            p_prep: noise.p_prep,
        },
    );
    if let Some(p) = p1 {
        params.p1 = p;
    }
    if let Some(p) = p2 {
        params.p2 = p;
    }
    if let Some(p) = p_meas {
        params.p_meas = p;
    }
    if let Some(p) = p_prep {
        params.p_prep = p;
    }
    params
}

fn py_locations_from_catalog(
    py: Python<'_>,
    catalog: &pecos_qec::fault_tolerance::fault_sampler::FaultCatalog,
) -> PyResult<Vec<Py<PyFaultLocation>>> {
    use pecos_qec::fault_tolerance::fault_sampler::{FaultChannel, FaultKind};

    let quantum_mod = py.import("pecos.quantum")?;
    let ps_class = quantum_mod.getattr("PauliString")?;
    let pauli_enum = quantum_mod.getattr("Pauli")?;
    let pauli_x = pauli_enum.getattr("X")?;
    let pauli_y = pauli_enum.getattr("Y")?;
    let pauli_z = pauli_enum.getattr("Z")?;

    let mut locations = Vec::with_capacity(catalog.locations.len());
    for loc in &catalog.locations {
        let mut faults = Vec::with_capacity(loc.faults.len());
        for fault in &loc.faults {
            let pauli_obj: Py<PyAny> = if let Some(ps) = &fault.pauli {
                let mut pair_list = Vec::new();
                for (p, q) in ps.iter_pairs() {
                    let py_pauli = match p {
                        pecos_core::Pauli::X => &pauli_x,
                        pecos_core::Pauli::Y => &pauli_y,
                        pecos_core::Pauli::Z => &pauli_z,
                        pecos_core::Pauli::I => continue,
                    };
                    let pair = pyo3::types::PyTuple::new(
                        py,
                        [py_pauli.as_any(), &q.index().into_pyobject(py)?.into_any()],
                    )?;
                    pair_list.push(pair.unbind());
                }
                let py_list = pyo3::types::PyList::new(py, pair_list.iter().map(|p| p.bind(py)))?;
                ps_class.call1((py_list,))?.unbind()
            } else {
                py.None()
            };

            faults.push(Py::new(
                py,
                PyFaultAlternative {
                    kind: match fault.kind {
                        FaultKind::Pauli => "pauli".to_string(),
                        FaultKind::MeasurementFlip => "measurement_flip".to_string(),
                        FaultKind::PrepFlip => "prep_flip".to_string(),
                    },
                    pauli_obj,
                    measurements: fault.affected_measurements.clone(),
                    detectors: fault.affected_detectors.clone(),
                    observables: fault.affected_observables.clone(),
                    tracked_paulis: fault.affected_tracked_paulis.clone(),
                    conditional_probability: fault.conditional_probability,
                    absolute_probability: fault.absolute_probability,
                    channel_probability: loc.channel_probability,
                },
            )?);
        }

        locations.push(Py::new(
            py,
            PyFaultLocation {
                tick: loc.tick,
                gate_index: loc.gate_index,
                gate_type: format!("{:?}", loc.gate_type),
                qubits: loc.qubits.clone(),
                channel: match loc.channel {
                    FaultChannel::P1 => "p1",
                    FaultChannel::P2 => "p2",
                    FaultChannel::PMeas => "p_meas",
                    FaultChannel::PPrep => "p_prep",
                }
                .to_string(),
                channel_probability: loc.channel_probability,
                no_fault_probability: loc.no_fault_probability,
                num_alternatives: loc.num_alternatives,
                faults,
            },
        )?);
    }
    Ok(locations)
}

fn sync_py_catalog_probabilities(py: Python<'_>, catalog: &mut PyFaultCatalog) -> PyResult<()> {
    if catalog.locations.len() != catalog.rust_catalog.locations.len() {
        catalog.locations = py_locations_from_catalog(py, &catalog.rust_catalog)?;
        return Ok(());
    }

    let fault_lengths_match = catalog
        .locations
        .iter()
        .zip(&catalog.rust_catalog.locations)
        .all(|(py_loc, rust_loc)| py_loc.borrow(py).faults.len() == rust_loc.faults.len());
    if !fault_lengths_match {
        catalog.locations = py_locations_from_catalog(py, &catalog.rust_catalog)?;
        return Ok(());
    }

    for (py_loc, rust_loc) in catalog
        .locations
        .iter()
        .zip(&catalog.rust_catalog.locations)
    {
        let mut loc = py_loc.borrow_mut(py);
        loc.channel_probability = rust_loc.channel_probability;
        loc.no_fault_probability = rust_loc.no_fault_probability;
        loc.num_alternatives = rust_loc.num_alternatives;
        for (py_fault, rust_fault) in loc.faults.iter().zip(&rust_loc.faults) {
            let mut fault = py_fault.borrow_mut(py);
            fault.conditional_probability = rust_fault.conditional_probability;
            fault.absolute_probability = rust_fault.absolute_probability;
            fault.channel_probability = rust_loc.channel_probability;
        }
    }
    Ok(())
}

fn py_fault_catalog_from_rust(
    py: Python<'_>,
    catalog: pecos_qec::fault_tolerance::fault_sampler::FaultCatalog,
) -> PyResult<PyFaultCatalog> {
    let locations = py_locations_from_catalog(py, &catalog)?;
    Ok(PyFaultCatalog {
        locations,
        rust_catalog: catalog,
    })
}

#[pymethods]
impl PyFaultCatalog {
    fn __len__(&self) -> usize {
        self.locations.len()
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyFaultLocation>> {
        let len = isize::try_from(self.locations.len()).map_err(|_| {
            pyo3::exceptions::PyIndexError::new_err("fault catalog is too large to index")
        })?;
        let index = if index < 0 { len + index } else { index };
        if index < 0 || index >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "fault catalog index out of range",
            ));
        }
        let index = usize::try_from(index).map_err(|_| {
            pyo3::exceptions::PyIndexError::new_err("fault catalog index out of range")
        })?;
        Ok(self.locations[index].clone_ref(py))
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let locations = pyo3::types::PyList::new(py, slf.locations.iter().map(|loc| loc.bind(py)))?;
        Ok(locations.call_method0("__iter__")?.unbind())
    }

    /// Recompute catalog probabilities for a new stochastic noise point.
    #[pyo3(signature = (noise=None, *, p1=None, p2=None, p_meas=None, p_prep=None))]
    fn with_noise(
        &mut self,
        py: Python<'_>,
        noise: Option<&PyNoiseModelBuilder>,
        p1: Option<f64>,
        p2: Option<f64>,
        p_meas: Option<f64>,
        p_prep: Option<f64>,
    ) -> PyResult<()> {
        let params = stochastic_params_from_inputs(noise, p1, p2, p_meas, p_prep);
        self.rust_catalog.with_noise(&params);
        sync_py_catalog_probabilities(py, self)
    }

    /// Return a cloned catalog parameterized at a new stochastic noise point.
    #[pyo3(signature = (noise=None, *, p1=None, p2=None, p_meas=None, p_prep=None))]
    fn parameterized(
        &self,
        py: Python<'_>,
        noise: Option<&PyNoiseModelBuilder>,
        p1: Option<f64>,
        p2: Option<f64>,
        p_meas: Option<f64>,
        p_prep: Option<f64>,
    ) -> PyResult<PyFaultCatalog> {
        let params = stochastic_params_from_inputs(noise, p1, p2, p_meas, p_prep);
        py_fault_catalog_from_rust(py, self.rust_catalog.parameterized(&params))
    }

    /// Lazily iterate all k-fault configurations.
    ///
    /// Returns an iterator yielding `FaultConfiguration` objects one at a time.
    fn fault_configurations(
        &self,
        py: Python<'_>,
        k: usize,
    ) -> PyResult<Py<PyFaultConfigurationIter>> {
        use pecos_qec::fault_tolerance::fault_sampler::OwnedFaultConfigIter;
        let inner = OwnedFaultConfigIter::new(self.rust_catalog.clone(), k);
        let py_locations: Vec<Py<PyFaultLocation>> =
            self.locations.iter().map(|l| l.clone_ref(py)).collect();
        Py::new(
            py,
            PyFaultConfigurationIter {
                inner,
                py_locations,
            },
        )
    }
}

/// Build a fault catalog for a circuit, optionally parameterized by a noise model.
///
/// Returns a ``FaultCatalog`` object with ``catalog.locations``. The catalog
/// also supports direct iteration, indexing, and ``len(catalog)``.
///
/// Each location has attribute access: ``loc.tick``, ``loc.gate_type``,
/// ``loc.qubits``, ``loc.faults``.
///
/// Each ``FaultAlternative`` has: ``fault.kind``, ``fault.pauli`` (a real
/// PECOS ``PauliString`` or ``None``), ``fault.detectors``, ``fault.observables``,
/// ``fault.tracked_paulis``, ``fault.measurements``, ``fault.conditional_probability``,
/// ``fault.absolute_probability``, ``fault.channel_probability``.
///
/// When noise is omitted, returns a structural catalog with zero probabilities.
/// The catalog includes all structurally supported physical fault locations.
#[pyfunction]
#[pyo3(signature = (tick_circuit, noise=None, *, p1=None, p2=None, p_meas=None, p_prep=None))]
pub fn fault_catalog(
    tick_circuit: &Bound<'_, PyAny>,
    noise: Option<&PyNoiseModelBuilder>,
    py: Python<'_>,
    p1: Option<f64>,
    p2: Option<f64>,
    p_meas: Option<f64>,
    p_prep: Option<f64>,
) -> PyResult<PyFaultCatalog> {
    use pecos_qec::fault_tolerance::fault_sampler::FaultCatalog;

    let tc = build_rust_tick_circuit(tick_circuit)?;
    let mut catalog = FaultCatalog::from_circuit(&tc)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    if noise.is_some() || p1.is_some() || p2.is_some() || p_meas.is_some() || p_prep.is_some() {
        let noise_params = stochastic_params_from_inputs(noise, p1, p2, p_meas, p_prep);
        catalog.with_noise(&noise_params);
    }
    py_fault_catalog_from_rust(py, catalog)
}
