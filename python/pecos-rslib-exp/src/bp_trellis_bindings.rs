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

use crate::decoder_specs::{
    decoder_error_to_py, indexed_decoder_error_to_py, validate_batch_workers,
};
use pecos_bp_trellis::{
    BpTrellisConfig as RustBpTrellisConfig, BpTrellisDecoder as RustBpTrellisDecoder,
    BpTrellisOutcome, EscalationRung, NoPathCause, NoPathReport,
    TrellisOrdering as RustTrellisOrdering,
};
use pecos_trellis::{ObsMask, SparseDem, TrellisResult, TrellisStatus};
use pyo3::Borrowed;
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyInt, PyList};

pub(crate) enum TrellisOrderArgument {
    Name(String),
    Explicit(Vec<usize>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for TrellisOrderArgument {
    type Error = PyErr;

    fn extract(object: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(name) = object.extract::<String>() {
            return Ok(Self::Name(name));
        }
        if let Ok(order) = object.extract::<Vec<usize>>() {
            return Ok(Self::Explicit(order));
        }
        Err(PyValueError::new_err(
            "ordering must be 'deadline', 'backward_deadline', 'time_order', \
             or a list of mechanism indices",
        ))
    }
}

impl Default for TrellisOrderArgument {
    fn default() -> Self {
        Self::Name("deadline".to_owned())
    }
}

fn sparse_index_error(index: u64, num_detectors: usize) -> PyErr {
    PyRuntimeError::new_err(format!(
        "Invalid node index {index}: must be < {num_detectors}"
    ))
}

fn sparse_to_dense(indices: &[u64], num_detectors: usize) -> PyResult<Vec<u8>> {
    let mut syndrome = vec![0; num_detectors];
    for &index in indices {
        let detector =
            usize::try_from(index).map_err(|_| sparse_index_error(index, num_detectors))?;
        let bit = syndrome
            .get_mut(detector)
            .ok_or_else(|| sparse_index_error(index, num_detectors))?;
        *bit = 1;
    }
    Ok(syndrome)
}

fn parse_dem_and_config(
    dem_str: &str,
    k: usize,
    delta: f64,
    score_alpha: f64,
    bp_score_iterations: usize,
    merge_indistinguishable: bool,
    ordering: TrellisOrderArgument,
    escalation_ks: Option<Vec<usize>>,
    escalation: Option<Vec<(usize, f64)>>,
) -> PyResult<(SparseDem, RustBpTrellisConfig)> {
    let config = RustBpTrellisConfig {
        k,
        delta,
        score_alpha,
        bp_score_iterations,
        merge_indistinguishable,
        ordering: parse_ordering(ordering)?,
        escalation: resolve_escalation(escalation_ks, escalation, delta)?,
    };
    config
        .validate()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let dem = SparseDem::from_dem_str(dem_str).map_err(|error| decoder_error_to_py(&error))?;
    Ok((dem, config))
}

pub(crate) fn resolve_escalation(
    escalation_ks: Option<Vec<usize>>,
    escalation: Option<Vec<(usize, f64)>>,
    delta: f64,
) -> PyResult<Vec<EscalationRung>> {
    match (escalation_ks, escalation) {
        (Some(_), Some(_)) => Err(PyValueError::new_err(
            "escalation_ks and escalation cannot both be supplied",
        )),
        (Some(ks), None) => Ok(ks
            .into_iter()
            .map(|k| EscalationRung { k, delta })
            .collect()),
        (None, rungs) => Ok(rungs
            .unwrap_or_default()
            .into_iter()
            .map(|(k, delta)| EscalationRung { k, delta })
            .collect()),
    }
}

fn report_no_path(on_no_path: &str) -> PyResult<bool> {
    match on_no_path {
        "raise" => Ok(false),
        "report" => Ok(true),
        _ => Err(PyValueError::new_err(
            "on_no_path must be 'raise' or 'report'",
        )),
    }
}

fn outcome_to_py(
    py: Python<'_>,
    outcome: BpTrellisOutcome,
    num_observables: usize,
) -> PyResult<Py<PyAny>> {
    match outcome {
        BpTrellisOutcome::Decoded(inner) => Ok(Py::new(
            py,
            PyBpTrellisResult {
                inner,
                num_observables,
            },
        )?
        .into_any()),
        BpTrellisOutcome::NoPath(inner) => Ok(Py::new(
            py,
            PyBpTrellisNoPath {
                inner,
                num_observables,
            },
        )?
        .into_any()),
    }
}

pub(crate) fn parse_ordering(ordering: TrellisOrderArgument) -> PyResult<RustTrellisOrdering> {
    match ordering {
        TrellisOrderArgument::Name(name) => match name.as_str() {
            "deadline" => Ok(RustTrellisOrdering::Deadline),
            "backward_deadline" => Ok(RustTrellisOrdering::BackwardDeadline),
            "time_order" => Ok(RustTrellisOrdering::TimeOrder),
            _ => Err(PyValueError::new_err(format!(
                "invalid ordering {name:?}; expected 'deadline', 'backward_deadline', \
                     'time_order', or a list of mechanism indices"
            ))),
        },
        TrellisOrderArgument::Explicit(order) => Ok(RustTrellisOrdering::Explicit(order)),
    }
}

fn obs_mask_to_py(py: Python<'_>, mask: &ObsMask) -> PyResult<Py<PyAny>> {
    if let Some(value) = mask.to_u64() {
        return Ok(value.into_pyobject(py)?.into_any().unbind());
    }

    let mut bytes = Vec::with_capacity(std::mem::size_of_val(mask.words()));
    for &word in mask.words() {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let py_bytes = PyBytes::new(py, &bytes);
    Ok(py
        .get_type::<PyInt>()
        .call_method1("from_bytes", (py_bytes, "little"))?
        .unbind())
}

/// Logical-observable flips returned by the BP-trellis decoder.
#[pyclass(
    name = "BpTrellisObservableFlips",
    module = "pecos_rslib_exp",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyBpTrellisObservableFlips {
    mask: ObsMask,
    num_observables: usize,
}

#[pymethods]
impl PyBpTrellisObservableFlips {
    fn __len__(&self) -> usize {
        self.num_observables
    }

    fn __getitem__(&self, index: isize) -> PyResult<bool> {
        let normalized = if index < 0 {
            self.num_observables.checked_add_signed(index)
        } else {
            usize::try_from(index).ok()
        };
        normalized
            .filter(|&normalized_index| normalized_index < self.num_observables)
            .map(|normalized_index| self.mask.get(normalized_index))
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "Observable index {index} out of range (num_observables={})",
                    self.num_observables
                ))
            })
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let bits = (0..self.num_observables).map(|index| self.mask.get(index));
        Ok(PyList::new(py, bits)?.call_method0("__iter__")?.unbind())
    }

    fn indices(&self) -> Vec<usize> {
        self.mask.iter_set_bits().collect()
    }

    #[getter]
    fn mask(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        obs_mask_to_py(py, &self.mask)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mask = obs_mask_to_py(py, &self.mask)?;
        Ok(format!(
            "BpTrellisObservableFlips(num_observables={}, mask={})",
            self.num_observables,
            mask.bind(py).str()?.to_str()?
        ))
    }
}

/// Result returned by the native experimental BP-trellis decoder.
#[pyclass(name = "BpTrellisResult", module = "pecos_rslib_exp")]
pub struct PyBpTrellisResult {
    inner: TrellisResult,
    num_observables: usize,
}

#[pymethods]
impl PyBpTrellisResult {
    /// Predicted logical-observable flips with their intrinsic width.
    #[getter]
    fn observable_flips(&self) -> PyBpTrellisObservableFlips {
        PyBpTrellisObservableFlips {
            mask: self.inner.predicted.clone(),
            num_observables: self.num_observables,
        }
    }

    /// A successful decode is never a no-path placeholder.
    #[getter]
    fn no_path(&self) -> bool {
        false
    }

    /// Number of BP refreshes run for this shot.
    #[getter]
    fn bp_runs(&self) -> u32 {
        self.inner.bp_runs
    }

    /// Total retained log evidence.
    #[getter]
    fn log_evidence(&self) -> f64 {
        self.inner.log_evidence
    }

    /// Winner-to-runner-up retained-mass gap, if a runner-up survives pruning.
    #[getter]
    fn runner_up_gap(&self) -> Option<f64> {
        self.inner.runner_up_gap
    }

    /// Peak number of retained dynamic-programming states.
    #[getter]
    fn peak_retained_states(&self) -> usize {
        self.inner.peak_retained_states
    }

    /// Number of probabilistic columns processed.
    #[getter]
    fn processed_columns(&self) -> usize {
        self.inner.processed_columns
    }

    /// Number of candidate branch evaluations across all attempted rungs.
    #[getter]
    fn transitions(&self) -> u64 {
        self.inner.transitions
    }

    /// Number of merged states discarded by the successful rung.
    #[getter]
    fn dropped_states(&self) -> u64 {
        self.inner.dropped_states
    }

    /// Log-sum-exp of masses discarded by the successful rung at pruning time.
    #[getter]
    fn dropped_log_mass(&self) -> f64 {
        self.inner.dropped_log_mass
    }

    /// Wall-clock seconds spent preparing BP-informed scores once for this shot.
    #[getter]
    fn bp_seconds(&self) -> f64 {
        self.inner.bp_seconds
    }

    /// Number of no-path escalation rungs attempted before success.
    #[getter]
    fn escalation_rungs_used(&self) -> u32 {
        self.inner.escalation_rungs_used
    }

    /// Completeness status of the successful rung.
    #[getter]
    fn status(&self) -> &'static str {
        match self.inner.status {
            TrellisStatus::Exact => "exact",
            TrellisStatus::Pruned {
                k_capped: true,
                delta_pruned: false,
            } => "pruned:k",
            TrellisStatus::Pruned {
                k_capped: false,
                delta_pruned: true,
            } => "pruned:delta",
            TrellisStatus::Pruned {
                k_capped: true,
                delta_pruned: true,
            } => "pruned:k+delta",
            TrellisStatus::Pruned {
                k_capped: false,
                delta_pruned: false,
            } => unreachable!("pruned status must name at least one pruning mechanism"),
        }
    }

    /// Terminal logical labels and their unnormalized joint log masses.
    #[getter]
    fn logical_masses(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, f64)>> {
        self.inner
            .logical_masses
            .iter()
            .map(|mass| Ok((obs_mask_to_py(py, &mass.logical)?, mass.log_mass)))
            .collect()
    }
}

/// Explicit no-path outcome with the initial forced observable contribution.
#[pyclass(name = "BpTrellisNoPath", module = "pecos_rslib_exp")]
pub struct PyBpTrellisNoPath {
    inner: NoPathReport,
    num_observables: usize,
}

#[pymethods]
impl PyBpTrellisNoPath {
    /// Reason no path exists: "residual", "infeasible", or "exhausted".
    #[getter]
    fn cause(&self) -> &'static str {
        match self.inner.cause {
            NoPathCause::Residual { .. } => "residual",
            NoPathCause::Infeasible => "infeasible",
            NoPathCause::Exhausted => "exhausted",
        }
    }
    /// Lowest detector with an unchangeable residual, or None for other causes.
    #[getter]
    fn detector(&self) -> Option<usize> {
        match self.inner.cause {
            NoPathCause::Residual { detector } => Some(detector),
            _ => None,
        }
    }
    /// Initial forced observable flips with their intrinsic width; a placeholder, not a correction.
    #[getter]
    fn observable_flips(&self) -> PyBpTrellisObservableFlips {
        PyBpTrellisObservableFlips {
            mask: self.inner.placeholder.clone(),
            num_observables: self.num_observables,
        }
    }
    /// Number of escalation rungs attempted after the base attempt.
    #[getter]
    fn rungs_tried(&self) -> u32 {
        self.inner.rungs_tried
    }
    /// Number of candidate branch evaluations across the base and all attempted rungs.
    #[getter]
    fn transitions(&self) -> u64 {
        self.inner.transitions
    }
    /// Number of BP refreshes run for this shot; zero when the residual precheck fails.
    #[getter]
    fn bp_runs(&self) -> u32 {
        self.inner.bp_runs
    }
    /// Wall-clock seconds spent preparing BP-informed scores once for this shot.
    #[getter]
    fn bp_seconds(&self) -> f64 {
        self.inner.bp_seconds
    }
    /// Always True: this outcome is a no-path placeholder, not a successful decode.
    #[getter]
    fn no_path(&self) -> bool {
        true
    }
    fn __repr__(&self) -> String {
        let detector = self
            .detector()
            .map_or_else(String::new, |detector| format!(", detector={detector}"));
        format!(
            "BpTrellisNoPath(cause='{}'{detector}, rungs_tried={}, transitions={})",
            self.cause(),
            self.inner.rungs_tried,
            self.inner.transitions
        )
    }
}

/// PECOS's BP-guided trellis-class decoder.
#[pyclass(name = "BpTrellisDecoder", module = "pecos_rslib_exp", unsendable)]
pub struct PyBpTrellisDecoder {
    inner: RustBpTrellisDecoder,
    num_detectors: usize,
    num_observables: usize,
}

#[pymethods]
impl PyBpTrellisDecoder {
    /// Construct a PECOS BP-guided trellis decoder from a Stim-format DEM.
    #[staticmethod]
    #[pyo3(
        signature = (dem, *, k=8, delta=100.0, score_alpha=0.8, bp_score_iterations=5, merge_indistinguishable=true, ordering=TrellisOrderArgument::default(), escalation_ks=None, escalation=None),
        text_signature = "(dem, *, k=8, delta=100.0, score_alpha=0.8, bp_score_iterations=5, merge_indistinguishable=True, ordering='deadline', escalation_ks=None, escalation=None)"
    )]
    fn from_dem(
        dem: &str,
        k: usize,
        delta: f64,
        score_alpha: f64,
        bp_score_iterations: usize,
        merge_indistinguishable: bool,
        ordering: TrellisOrderArgument,
        escalation_ks: Option<Vec<usize>>,
        escalation: Option<Vec<(usize, f64)>>,
    ) -> PyResult<Self> {
        let (dem, config) = parse_dem_and_config(
            dem,
            k,
            delta,
            score_alpha,
            bp_score_iterations,
            merge_indistinguishable,
            ordering,
            escalation_ks,
            escalation,
        )?;
        let num_detectors = dem.num_detectors;
        let num_observables = dem.num_observables;
        let inner = RustBpTrellisDecoder::from_sparse_dem(&dem, config)
            .map_err(|error| decoder_error_to_py(&error))?;
        Ok(Self {
            inner,
            num_detectors,
            num_observables,
        })
    }

    /// Wall-clock seconds spent constructing the decoder model.
    #[getter]
    fn build_seconds(&self) -> f64 {
        self.inner.build_seconds()
    }

    /// Decode sparse fired-detector indices, optionally reporting no-path outcomes.
    /// on_no_path="raise" (default) raises for no-path; "report" returns a
    /// BpTrellisNoPath placeholder. Successful shots return BpTrellisResult.
    /// Other decoding errors still raise; unknown on_no_path values raise ValueError.
    #[pyo3(signature = (defects, *, on_no_path="raise"))]
    fn decode_from_defects(
        &mut self,
        py: Python<'_>,
        defects: Vec<u64>,
        on_no_path: &str,
    ) -> PyResult<Py<PyAny>> {
        report_no_path(on_no_path)?;
        let syndrome = sparse_to_dense(&defects, self.num_detectors)?;
        self.decode_syndrome(py, syndrome, on_no_path)
    }

    /// Decode one dense detector syndrome, optionally reporting a no-path outcome.
    /// on_no_path="raise" (default) raises for no-path; "report" returns a
    /// BpTrellisNoPath placeholder. Successful shots return BpTrellisResult.
    /// Other decoding errors still raise; unknown on_no_path values raise ValueError.
    #[pyo3(signature = (syndrome, *, on_no_path="raise"))]
    fn decode_syndrome(
        &mut self,
        py: Python<'_>,
        syndrome: Vec<u8>,
        on_no_path: &str,
    ) -> PyResult<Py<PyAny>> {
        let outcome = if report_no_path(on_no_path)? {
            self.inner.decode_outcome(&syndrome)
        } else {
            self.inner.decode(&syndrome).map(BpTrellisOutcome::Decoded)
        }
        .map_err(|error| decoder_error_to_py(&error))?;
        outcome_to_py(py, outcome, self.num_observables)
    }

    /// Decode dense shots in input order. Defaults to sequential execution.
    /// workers must be positive and is capped at one per shot (one for an empty batch).
    /// Each worker owns independent scratch; workers > 1 releases the GIL.
    /// on_no_path="raise" (default) raises on the first no-path with its shot index.
    /// on_no_path="report" returns a list that can mix BpTrellisResult and
    /// BpTrellisNoPath objects. Other shot errors still raise with their index.
    /// Unknown on_no_path values raise ValueError, including for an empty batch.
    #[pyo3(signature = (shots, *, workers=1, on_no_path="raise"))]
    fn decode_batch(
        &self,
        py: Python<'_>,
        shots: Vec<Vec<u8>>,
        workers: i64,
        on_no_path: &str,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let report = report_no_path(on_no_path)?;
        let workers = validate_batch_workers(workers)?;
        let decode = || {
            if report {
                self.inner.decode_batch_outcomes(&shots, workers)
            } else {
                self.inner.decode_batch(&shots, workers).map(|results| {
                    results
                        .into_iter()
                        .map(|result| result.map(BpTrellisOutcome::Decoded))
                        .collect()
                })
            }
        };
        let results = if workers > 1 {
            py.detach(decode)
        } else {
            decode()
        }
        .map_err(|error| decoder_error_to_py(&error))?;
        results
            .into_iter()
            .enumerate()
            .map(|(shot_index, result)| {
                let outcome =
                    result.map_err(|error| indexed_decoder_error_to_py(shot_index, &error))?;
                outcome_to_py(py, outcome, self.num_observables)
            })
            .collect()
    }

    fn __getattr__(&self, name: &str) -> PyResult<()> {
        if name == "decode" {
            Err(PyAttributeError::new_err(
                "BpTrellisDecoder has no attribute 'decode'; use decode_syndrome(...) for a dense \
                 vector or decode_from_defects(...) for sparse detector indices",
            ))
        } else {
            Err(PyAttributeError::new_err(format!(
                "'BpTrellisDecoder' object has no attribute '{name}'"
            )))
        }
    }
}
