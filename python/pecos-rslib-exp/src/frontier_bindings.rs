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

use pecos_frontier::{
    CommitteeDirection, CommitteeMember, CommitteeStatus, DecoderError,
    FrontierCommittee as RustFrontierCommittee,
    FrontierCommitteeResult as RustFrontierCommitteeResult, FrontierConfig as RustFrontierConfig,
    FrontierDecoder as RustFrontierDecoder, FrontierResult as RustFrontierResult, FrontierStatus,
    ObsMask, SparseDem, backward_deadline_column_order, deadline_column_order,
};
use pyo3::Borrowed;
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyInt, PyList};

enum ColumnOrderArgument {
    Name(String),
    Explicit(Vec<usize>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for ColumnOrderArgument {
    type Error = PyErr;

    fn extract(object: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(name) = object.extract::<String>() {
            return Ok(Self::Name(name));
        }
        if let Ok(order) = object.extract::<Vec<usize>>() {
            return Ok(Self::Explicit(order));
        }
        Err(PyValueError::new_err(
            "column_order must be 'deadline_reorder', 'time_order', \
             'backward_deadline_reorder', or a list of mechanism indices",
        ))
    }
}

impl Default for ColumnOrderArgument {
    fn default() -> Self {
        Self::Name("deadline_reorder".to_owned())
    }
}

fn runtime_error(error: &DecoderError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
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

fn resolve_column_order(
    dem: &SparseDem,
    column_order: ColumnOrderArgument,
) -> PyResult<Option<Vec<usize>>> {
    match column_order {
        ColumnOrderArgument::Name(name) => match name.as_str() {
            "deadline_reorder" => deadline_column_order(dem)
                .map(Some)
                .map_err(|e| runtime_error(&e)),
            "time_order" => Ok(None),
            "backward_deadline_reorder" => backward_deadline_column_order(dem)
                .map(Some)
                .map_err(|e| runtime_error(&e)),
            _ => Err(PyValueError::new_err(format!(
                "invalid column_order {name:?}; expected 'deadline_reorder', 'time_order', \
                 'backward_deadline_reorder', or a list of mechanism indices"
            ))),
        },
        ColumnOrderArgument::Explicit(order) => Ok(Some(order)),
    }
}

fn parse_dem_and_config(
    dem_str: &str,
    k: usize,
    delta: f64,
    score_alpha: f64,
    bp_score_iterations: usize,
    column_order: ColumnOrderArgument,
    merge_indistinguishable: bool,
) -> PyResult<(SparseDem, RustFrontierConfig)> {
    let dem = SparseDem::from_dem_str(dem_str).map_err(|e| runtime_error(&e))?;
    let column_order = resolve_column_order(&dem, column_order)?;
    Ok((
        dem,
        RustFrontierConfig {
            k,
            delta,
            score_alpha,
            column_order,
            merge_indistinguishable,
            bp_score_iterations,
        },
    ))
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

/// Logical-observable flips returned by a Frontier decoder.
#[pyclass(
    name = "FrontierObservableFlips",
    module = "pecos_rslib_exp",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFrontierObservableFlips {
    mask: ObsMask,
    num_observables: usize,
}

#[pymethods]
impl PyFrontierObservableFlips {
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
            "FrontierObservableFlips(num_observables={}, mask={})",
            self.num_observables,
            mask.bind(py).str()?.to_str()?
        ))
    }
}

fn logical_masses(py: Python<'_>, result: &RustFrontierResult) -> PyResult<Vec<(Py<PyAny>, f64)>> {
    result
        .logical_masses
        .iter()
        .map(|mass| Ok((obs_mask_to_py(py, &mass.logical)?, mass.log_mass)))
        .collect()
}

fn member_log_evidence(member: CommitteeMember) -> Option<f64> {
    match member.status {
        CommitteeStatus::Ok => Some(member.log_evidence),
        CommitteeStatus::NoPath => None,
    }
}

fn frontier_status(status: FrontierStatus) -> &'static str {
    match status {
        FrontierStatus::Exact => "exact",
        FrontierStatus::Pruned {
            k_capped: true,
            delta_pruned: false,
        } => "pruned:k",
        FrontierStatus::Pruned {
            k_capped: false,
            delta_pruned: true,
        } => "pruned:delta",
        FrontierStatus::Pruned {
            k_capped: true,
            delta_pruned: true,
        } => "pruned:k+delta",
        FrontierStatus::Pruned {
            k_capped: false,
            delta_pruned: false,
        } => unreachable!("pruned status must name at least one pruning mechanism"),
    }
}

/// Result returned by the native experimental Frontier decoder.
#[pyclass(name = "FrontierResult", module = "pecos_rslib_exp")]
pub struct PyFrontierResult {
    inner: RustFrontierResult,
    num_observables: usize,
}

#[pymethods]
impl PyFrontierResult {
    /// Predicted logical-observable flips with their intrinsic width.
    #[getter]
    fn observable_flips(&self) -> PyFrontierObservableFlips {
        PyFrontierObservableFlips {
            mask: self.inner.predicted.clone(),
            num_observables: self.num_observables,
        }
    }

    /// Total retained log evidence.
    #[getter]
    fn log_evidence(&self) -> f64 {
        self.inner.log_evidence
    }

    /// Winner-to-runner-up retained-mass gap, if a runner-up survives pruning.
    /// At small K this is often absent and is not an escalation trigger.
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

    /// Number of candidate branch evaluations, including all attempted rungs
    /// for an escalated BP-trellis result.
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

    /// Wall-clock seconds spent producing BP-informed pruning scores,
    /// including all attempted rungs for an escalated BP-trellis result.
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
        frontier_status(self.inner.status)
    }

    /// Terminal logical labels and their unnormalized joint log masses.
    #[getter]
    fn logical_masses(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, f64)>> {
        logical_masses(py, &self.inner)
    }
}

/// Result returned by the native experimental forward/backward committee.
#[pyclass(name = "FrontierCommitteeResult", module = "pecos_rslib_exp")]
pub struct PyFrontierCommitteeResult {
    inner: RustFrontierCommitteeResult,
    num_observables: usize,
}

#[pymethods]
impl PyFrontierCommitteeResult {
    /// Predicted logical-observable flips from the selected leg.
    #[getter]
    fn observable_flips(&self) -> PyFrontierObservableFlips {
        PyFrontierObservableFlips {
            mask: self.inner.selected.predicted.clone(),
            num_observables: self.num_observables,
        }
    }

    /// Total retained log evidence from the selected leg.
    #[getter]
    fn log_evidence(&self) -> f64 {
        self.inner.selected.log_evidence
    }

    /// Winner-to-runner-up retained-mass gap from the selected leg, if a
    /// runner-up survives pruning. At small K this is often absent and is not
    /// an escalation trigger.
    #[getter]
    fn runner_up_gap(&self) -> Option<f64> {
        self.inner.selected.runner_up_gap
    }

    /// Peak number of states retained by the selected leg.
    #[getter]
    fn peak_retained_states(&self) -> usize {
        self.inner.selected.peak_retained_states
    }

    /// Number of probabilistic columns processed by the selected leg.
    #[getter]
    fn processed_columns(&self) -> usize {
        self.inner.selected.processed_columns
    }

    /// Number of candidate branch evaluations in the selected leg.
    #[getter]
    fn transitions(&self) -> u64 {
        self.inner.selected.transitions
    }

    /// Number of states discarded by the selected leg.
    #[getter]
    fn dropped_states(&self) -> u64 {
        self.inner.selected.dropped_states
    }

    /// Log-sum-exp of masses discarded by the selected leg at pruning time.
    #[getter]
    fn dropped_log_mass(&self) -> f64 {
        self.inner.selected.dropped_log_mass
    }

    /// Wall-clock seconds spent by the selected leg producing BP-informed
    /// pruning scores.
    #[getter]
    fn bp_seconds(&self) -> f64 {
        self.inner.selected.bp_seconds
    }

    /// Completeness status of the selected leg.
    #[getter]
    fn status(&self) -> &'static str {
        frontier_status(self.inner.selected.status)
    }

    /// Terminal logical labels and log masses from the selected leg.
    #[getter]
    fn logical_masses(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, f64)>> {
        logical_masses(py, &self.inner.selected)
    }

    /// Selected committee direction: `"forward"` or `"backward"`.
    #[getter]
    fn direction(&self) -> &'static str {
        match self.inner.direction {
            CommitteeDirection::Forward => "forward",
            CommitteeDirection::Backward => "backward",
        }
    }

    /// Forward-leg evidence, or `None` when that leg found no path.
    #[getter]
    fn forward_log_evidence(&self) -> Option<f64> {
        member_log_evidence(self.inner.forward)
    }

    /// Backward-leg evidence, or `None` when that leg found no path.
    #[getter]
    fn backward_log_evidence(&self) -> Option<f64> {
        member_log_evidence(self.inner.backward)
    }
}

/// Native implementation of the Frontier decoder (Leverrier & Urbanke,
/// arXiv:2606.20513). This experimental decoder is implemented in Rust and does
/// not wrap the upstream `frontier` package.
#[pyclass(name = "FrontierDecoder", module = "pecos_rslib_exp", unsendable)]
pub struct PyFrontierDecoder {
    inner: RustFrontierDecoder,
    num_detectors: usize,
    num_observables: usize,
}

#[pymethods]
impl PyFrontierDecoder {
    /// Construct a native Frontier decoder from a Stim-format DEM string.
    #[staticmethod]
    #[pyo3(
        signature = (dem, *, k=64, delta=50.0, score_alpha=0.8, bp_score_iterations=0, column_order=ColumnOrderArgument::default(), merge_indistinguishable=false),
        text_signature = "(dem, *, k=64, delta=50.0, score_alpha=0.8, bp_score_iterations=0, column_order='deadline_reorder', merge_indistinguishable=False)"
    )]
    fn from_dem(
        dem: &str,
        k: usize,
        delta: f64,
        score_alpha: f64,
        bp_score_iterations: usize,
        column_order: ColumnOrderArgument,
        merge_indistinguishable: bool,
    ) -> PyResult<Self> {
        let (dem, config) = parse_dem_and_config(
            dem,
            k,
            delta,
            score_alpha,
            bp_score_iterations,
            column_order,
            merge_indistinguishable,
        )?;
        let num_detectors = dem.num_detectors;
        let num_observables = dem.num_observables;
        let inner = RustFrontierDecoder::from_sparse_dem(&dem, config)
            .map_err(|error| runtime_error(&error))?;
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

    /// Decode sparse fired-detector indices.
    fn decode_from_defects(&mut self, defects: Vec<u64>) -> PyResult<PyFrontierResult> {
        let syndrome = sparse_to_dense(&defects, self.num_detectors)?;
        self.decode_syndrome(syndrome)
    }

    /// Decode one dense detector syndrome.
    fn decode_syndrome(&mut self, syndrome: Vec<u8>) -> PyResult<PyFrontierResult> {
        self.inner
            .decode(&syndrome)
            .map(|inner| PyFrontierResult {
                inner,
                num_observables: self.num_observables,
            })
            .map_err(|error| runtime_error(&error))
    }

    /// Decode a batch of dense detector syndromes in input order.
    fn decode_batch(&mut self, shots: Vec<Vec<u8>>) -> PyResult<Vec<PyFrontierResult>> {
        shots
            .into_iter()
            .map(|syndrome| self.decode_syndrome(syndrome))
            .collect()
    }

    fn __getattr__(&self, name: &str) -> PyResult<()> {
        if name == "decode" {
            Err(PyAttributeError::new_err(
                "FrontierDecoder has no attribute 'decode'; use decode_syndrome(...) for a dense \
                 vector or decode_from_defects(...) for sparse detector indices",
            ))
        } else {
            Err(PyAttributeError::new_err(format!(
                "'FrontierDecoder' object has no attribute '{name}'"
            )))
        }
    }
}

/// Native experimental forward/backward committee of two Frontier decoders.
/// The implementation is Rust-native and does not wrap the upstream
/// `frontier` package (Leverrier & Urbanke, arXiv:2606.20513).
#[pyclass(
    name = "FrontierCommitteeDecoder",
    module = "pecos_rslib_exp",
    unsendable
)]
pub struct PyFrontierCommitteeDecoder {
    inner: RustFrontierCommittee,
    num_detectors: usize,
    num_observables: usize,
}

#[pymethods]
impl PyFrontierCommitteeDecoder {
    /// Construct a native forward/backward committee from a Stim-format DEM.
    #[staticmethod]
    #[pyo3(
        signature = (dem, *, k=64, delta=50.0, score_alpha=0.8, bp_score_iterations=0, column_order=ColumnOrderArgument::default(), merge_indistinguishable=false),
        text_signature = "(dem, *, k=64, delta=50.0, score_alpha=0.8, bp_score_iterations=0, column_order='deadline_reorder', merge_indistinguishable=False)"
    )]
    fn from_dem(
        dem: &str,
        k: usize,
        delta: f64,
        score_alpha: f64,
        bp_score_iterations: usize,
        column_order: ColumnOrderArgument,
        merge_indistinguishable: bool,
    ) -> PyResult<Self> {
        let (dem, config) = parse_dem_and_config(
            dem,
            k,
            delta,
            score_alpha,
            bp_score_iterations,
            column_order,
            merge_indistinguishable,
        )?;
        let num_detectors = dem.num_detectors;
        let num_observables = dem.num_observables;
        let inner = RustFrontierCommittee::from_sparse_dem(&dem, config)
            .map_err(|error| runtime_error(&error))?;
        Ok(Self {
            inner,
            num_detectors,
            num_observables,
        })
    }

    /// Wall-clock seconds spent constructing both committee legs.
    #[getter]
    fn build_seconds(&self) -> f64 {
        self.inner.build_seconds()
    }

    /// Decode sparse fired-detector indices with both committee legs.
    fn decode_from_defects(&mut self, defects: Vec<u64>) -> PyResult<PyFrontierCommitteeResult> {
        let syndrome = sparse_to_dense(&defects, self.num_detectors)?;
        self.decode_syndrome(syndrome)
    }

    /// Decode one dense detector syndrome with both committee legs.
    fn decode_syndrome(&mut self, syndrome: Vec<u8>) -> PyResult<PyFrontierCommitteeResult> {
        self.inner
            .decode(&syndrome)
            .map(|inner| PyFrontierCommitteeResult {
                inner,
                num_observables: self.num_observables,
            })
            .map_err(|error| runtime_error(&error))
    }

    /// Decode a batch of dense detector syndromes in input order.
    fn decode_batch(&mut self, shots: Vec<Vec<u8>>) -> PyResult<Vec<PyFrontierCommitteeResult>> {
        shots
            .into_iter()
            .map(|syndrome| self.decode_syndrome(syndrome))
            .collect()
    }

    fn __getattr__(&self, name: &str) -> PyResult<()> {
        if name == "decode" {
            Err(PyAttributeError::new_err(
                "FrontierCommitteeDecoder has no attribute 'decode'; use decode_syndrome(...) for \
                 a dense vector or decode_from_defects(...) for sparse detector indices",
            ))
        } else {
            Err(PyAttributeError::new_err(format!(
                "'FrontierCommitteeDecoder' object has no attribute '{name}'"
            )))
        }
    }
}
