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
//!     .shots(5000)
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
    records: Vec<i32>,
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
        }
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
///     sim_neo(tc).quantum(stabilizer()).noise(depolarizing().p2(0.01)).shots(10000).run()
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
///     sim_neo(tc).quantum(statevec()).noise(depolarizing().idle_rz(0.05)).shots(10000).run()
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
///     sim_neo(tc).quantum(statevec()).noise(...).shots(10000).run()
#[pyfunction]
pub fn statevec() -> PyStateVecBuilder {
    PyStateVecBuilder
}

/// Create a stabilizer (SparseStab) backend builder.
///
/// Example:
///     sim_neo(tc).quantum(stabilizer()).noise(...).shots(10000).run()
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

    fn lazy_measure(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.lazy_measure = true;
        slf
    }

    fn max_bond_dim(mut slf: PyRefMut<'_, Self>, bd: usize) -> PyRefMut<'_, Self> {
        slf.inner.max_bond_dim = bd;
        slf
    }

    fn max_truncation_error(mut slf: PyRefMut<'_, Self>, err: f64) -> PyRefMut<'_, Self> {
        slf.inner.max_truncation_error = Some(err);
        slf
    }

    fn merge_rz(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.merge_rz = true;
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
    shots: usize,
    seed: u64,
    noise_config: Option<PyNoiseModelBuilder>,
    backend: String,
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
            c.backend = "meas_sampling".to_string();
            c.meas_sampling_method = Some(b.method.clone());
            c.stabmps_config = None;
        } else if builder.is_instance_of::<PyStabMpsBuilder>() {
            let b: PyRef<'_, PyStabMpsBuilder> = builder.extract()?;
            c.backend = "stabmps".to_string();
            c.stabmps_config = Some(b.inner.clone());
            c.meas_sampling_method = None;
        } else if builder.is_instance_of::<PyStabilizerBuilder>() {
            c.backend = "stabilizer".to_string();
            c.stabmps_config = None;
            c.meas_sampling_method = None;
        } else if builder.is_instance_of::<PyStateVecBuilder>() {
            c.backend = "statevec".to_string();
            c.stabmps_config = None;
            c.meas_sampling_method = None;
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "quantum() expects statevec(), stabilizer(), stab_mps(), or meas_sampling()",
            ));
        }
        Ok(c)
    }

    /// Set the noise model.
    fn noise(&self, noise_builder: &PyNoiseModelBuilder) -> Self {
        let mut c = self.clone();
        c.noise_config = Some(noise_builder.clone());
        c
    }

    /// Set number of shots.
    fn shots(&self, n: usize) -> Self {
        let mut c = self.clone();
        c.shots = n;
        c
    }

    /// Set random seed.
    fn seed(&self, s: u64) -> Self {
        let mut c = self.clone();
        c.seed = s;
        c
    }

    /// Run the simulation and return per-shot measurement outcomes.
    ///
    /// All backends return `RawMeasurementResult` which supports:
    /// `result[shot]`, `result.get(shot, meas)`, `len(result)`, iteration.
    fn run(&self) -> PyResult<PyRawMeasurementResult> {
        if self.tick_circuit.has_channel_operations() {
            return self.run_inline_channel_circuit();
        }

        if self.backend == "meas_sampling" {
            return self.run_meas_sampling();
        }

        let noise = self
            .noise_config
            .as_ref()
            .and_then(PyNoiseModelBuilder::build_noise);

        let mut builder = sim_neo(self.commands.clone())
            .shots(self.shots)
            .seed(self.seed);

        if let Some(n) = noise {
            builder = builder.noise(n);
        }

        match self.backend.as_str() {
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
                    "Unknown backend: {}",
                    self.backend
                )));
            }
        }

        let mut sim = builder.build();
        let results = sim.run();

        let mut all_shots = Vec::with_capacity(self.shots);
        for shot_outcomes in &results.outcomes {
            let meas: Vec<u8> = shot_outcomes.iter().map(|o| u8::from(o.outcome)).collect();
            all_shots.push(meas);
        }

        Ok(PyRawMeasurementResult::from_rows(all_shots))
    }
}

impl PySimNeoBuilder {
    fn run_inline_channel_circuit(&self) -> PyResult<PyRawMeasurementResult> {
        if self.noise_config.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sim_neo received a TickCircuit with inline channel operations; do not also pass .noise()",
            ));
        }

        match self.backend.as_str() {
            "statevec" => self.run_inline_channel_density_matrix(),
            "stabilizer" => self.run_inline_pauli_channel_stabilizer(),
            "stabmps" => Err(pyo3::exceptions::PyValueError::new_err(
                "stab_mps backend does not support inline channel operations; use statevec()/default for density-matrix execution",
            )),
            "meas_sampling" => Err(pyo3::exceptions::PyValueError::new_err(
                "meas_sampling backend builds its own measurement model and does not consume inline channel operations",
            )),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown backend: {other}"
            ))),
        }
    }

    fn run_inline_channel_density_matrix(&self) -> PyResult<PyRawMeasurementResult> {
        let rows = pecos_neo::inline_channel::run_inline_channels_density_matrix(
            &self.tick_circuit,
            self.shots,
            self.seed,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyRawMeasurementResult::from_rows(rows))
    }

    fn run_inline_pauli_channel_stabilizer(&self) -> PyResult<PyRawMeasurementResult> {
        let rows = pecos_neo::inline_channel::run_inline_pauli_channels_stabilizer(
            &self.tick_circuit,
            self.shots,
            self.seed,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyRawMeasurementResult::from_rows(rows))
    }

    /// DEM sampling backend: dispatches to stochastic or coherent path based on method.
    fn run_meas_sampling(&self) -> PyResult<PyRawMeasurementResult> {
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
        let result = plan.sample(self.shots, self.seed);

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

        let result = run_dem_simulation(
            &gates,
            &noise,
            &meta,
            generator.as_ref(),
            self.shots,
            self.seed,
        );
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
/// Example:
///     results = (sim_neo(tc)
///         .quantum(stab_mps().lazy_measure().max_bond_dim(128))
///         .noise(depolarizing().p1(0.003).p2(0.003).p_meas(0.003).idle_rz(0.05))
///         .shots(5000)
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
        shots: 1,
        seed: 42,
        noise_config: None,
        backend: "statevec".to_string(),
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
                "MZ" | "Measure" | "MeasureFree" => {
                    mz_qubits.extend(qubit_ids);
                }
                "QAlloc" | "PZ" | "Prep" => {
                    pz_qubits.extend(qubit_ids);
                }
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
        create_annotations_from_json(&mut tc, &s, &all_meas_refs, true);
        tc.set_meta("detectors", Attribute::String(s));
    }
    if let Ok(obs_json) = py_tc.call_method1("get_meta", ("observables",))
        && let Ok(s) = obs_json.extract::<String>()
    {
        create_annotations_from_json(&mut tc, &s, &all_meas_refs, false);
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
fn create_annotations_from_json(
    tc: &mut pecos_quantum::TickCircuit,
    json_str: &str,
    all_meas_refs: &[pecos_quantum::TickMeasRef],
    is_detector: bool,
) {
    let num_meas = all_meas_refs.len();
    if let Ok(defs) = serde_json::from_str::<Vec<RecDef>>(json_str) {
        for def in &defs {
            let refs: Vec<pecos_quantum::TickMeasRef> = def
                .records
                .iter()
                .filter_map(|&rec| {
                    let abs_idx = measurement_record_index(rec, num_meas)?;
                    all_meas_refs.get(abs_idx).copied()
                })
                .collect();
            if !refs.is_empty() {
                if is_detector {
                    tc.detector(&refs);
                } else {
                    tc.observable(&refs);
                }
            }
        }
    }
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
                "MZ" => {
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
                "I" | "Idle" => {
                    // Identity/Idle gates: skip (no-op for simulation)
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
