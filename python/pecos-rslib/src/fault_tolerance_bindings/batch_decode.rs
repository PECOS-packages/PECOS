//! Execution core and Python result for `SampleBatch.decode`.

use super::{PyDecodeStats, PyProjectionReport, decoder_build_error_to_py};
use crate::batch_decoder_spec::DecoderBuildError;
use pecos_decoder_core::obs_mask::ObsMask;
use pecos_decoders::batch::ExecutionPlan;
use pecos_decoders::batch::{BatchDecodeError, DecodeResult, ShotDecodeError};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub(super) struct BatchExecutionOutput {
    /// Shared projection diagnostics. `None` means no projection was needed, or a
    /// composition without a shared policy projected inside members with no shared report.
    pub(super) projection: Option<pecos_decoder_core::graphlike_projection::ProjectionReport>,
    pub(super) num_errors: usize,
    pub(super) predictions: Option<Vec<ObsMask>>,
    pub(super) partial_predictions: Option<Vec<Option<ObsMask>>>,
    pub(super) recovery: Vec<pecos_decoder_core::outcome::ShotRecovery>,
    pub(super) per_shot_seconds: Option<Vec<f64>>,
    pub(super) wall_elapsed: f64,
}

pub(super) enum BatchExecutionError {
    Build(DecoderBuildError),
    SamplerDimension {
        sampler_detectors: usize,
        decoder_detectors: usize,
    },
    Decode(ShotDecodeError),
    Runtime(String),
}

impl BatchExecutionError {
    pub(super) fn into_pyerr(self) -> PyErr {
        match self {
            Self::Build(DecoderBuildError::Decoder(error)) => decoder_build_error_to_py(error),
            Self::Build(DecoderBuildError::Python(error)) => error,
            Self::SamplerDimension {
                sampler_detectors,
                decoder_detectors,
            } => PyValueError::new_err(format!(
                "DemSampler has {sampler_detectors} detectors, but the decoder model has \
                 {decoder_detectors}"
            )),
            Self::Decode(error) => PyRuntimeError::new_err(error.to_string()),
            Self::Runtime(message) => PyRuntimeError::new_err(message),
        }
    }
}

/// Per-shot diagnostics for retries, undecoded shots, and fallbacks.
#[pyclass(
    name = "DecodeRecovery",
    module = "pecos_rslib.qec",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyDecodeRecovery {
    #[pyo3(get)]
    shot_index: usize,
    #[pyo3(get)]
    undecoded: bool,
    #[pyo3(get)]
    logical_error: Option<bool>,
    #[pyo3(get)]
    fallback_used: bool,
    #[pyo3(get)]
    syndrome_consistent: Option<bool>,
    #[pyo3(get)]
    deltas_tried: Vec<f64>,
}
impl From<pecos_decoder_core::outcome::ShotRecovery> for PyDecodeRecovery {
    fn from(value: pecos_decoder_core::outcome::ShotRecovery) -> Self {
        Self {
            shot_index: value.shot_index,
            undecoded: value.undecoded,
            logical_error: value.logical_error,
            fallback_used: value.fallback_used,
            syndrome_consistent: value.syndrome_consistent,
            deltas_tried: value.deltas_tried,
        }
    }
}

/// Result of decoding and scoring one `SampleBatch`.
#[pyclass(name = "DecodeResult", module = "pecos_rslib.qec", skip_from_py_object)]
pub struct PyDecodeResult {
    /// Shared projection diagnostics. `None` means no projection was needed, or a
    /// composition without a shared policy projected inside members with no shared report.
    #[pyo3(get)]
    projection: Option<PyProjectionReport>,
    #[pyo3(get)]
    num_shots: usize,
    #[pyo3(get)]
    num_errors: usize,
    #[pyo3(get)]
    num_undecoded: usize,
    #[pyo3(get)]
    num_fallback: usize,
    #[pyo3(get)]
    num_scored: usize,
    #[pyo3(get)]
    recovery: Vec<PyDecodeRecovery>,
    #[pyo3(get)]
    logical_error_rate: f64,
    #[pyo3(get)]
    execution_path: String,
    #[pyo3(get)]
    workers_used: usize,
    #[pyo3(get)]
    reproducibility_warnings: Vec<String>,
    #[pyo3(get)]
    sampling_seed_used: Option<u64>,
    predictions: Option<Vec<Py<PyAny>>>,
    stats: Option<Py<PyDecodeStats>>,
}

impl PyDecodeResult {
    pub(super) fn from_result(py: Python<'_>, result: DecodeResult) -> PyResult<Self> {
        Self::from_execution_with_seed(
            py,
            result.num_shots,
            ExecutionPlan {
                path: result.execution_path,
                workers_used: result.workers_used,
                reproducibility_warnings: result.reproducibility_warnings,
            },
            BatchExecutionOutput {
                projection: result.projection,
                num_errors: result.num_errors,
                predictions: result.predictions,
                partial_predictions: result.partial_predictions,
                recovery: result.recovery,
                per_shot_seconds: result.per_shot_seconds,
                wall_elapsed: result.wall_elapsed,
            },
            None,
        )
    }

    pub(super) fn from_sampler_execution(
        py: Python<'_>,
        num_shots: usize,
        plan: ExecutionPlan,
        output: BatchExecutionOutput,
        sampling_seed_used: u64,
    ) -> PyResult<Self> {
        Self::from_execution_with_seed(py, num_shots, plan, output, Some(sampling_seed_used))
    }

    fn from_execution_with_seed(
        py: Python<'_>,
        num_shots: usize,
        plan: ExecutionPlan,
        output: BatchExecutionOutput,
        sampling_seed_used: Option<u64>,
    ) -> PyResult<Self> {
        let num_undecoded = output.recovery.iter().filter(|r| r.undecoded).count();
        let num_fallback = output.recovery.iter().filter(|r| r.fallback_used).count();
        let logical_error_rate = if num_undecoded > 0 {
            f64::NAN
        } else {
            pecos_decoders::batch::logical_error_rate(output.num_errors, num_shots)
        };
        let predictions = output
            .partial_predictions
            .or_else(|| {
                output
                    .predictions
                    .map(|masks| masks.into_iter().map(Some).collect())
            })
            .map(|masks| {
                masks
                    .iter()
                    .map(|mask| match mask {
                        Some(mask) => crate::observable_flips_bindings::obsmask_to_py(py, mask),
                        None => Ok(py.None()),
                    })
                    .collect()
            })
            .transpose()?;
        let stats = output
            .per_shot_seconds
            .map(|times| {
                let summed_decode_elapsed = times.iter().sum();
                let mut stats = PyDecodeStats::from_times_with_elapsed(
                    num_shots,
                    output.num_errors,
                    times,
                    output.wall_elapsed,
                    summed_decode_elapsed,
                );
                stats.logical_error_rate = logical_error_rate;
                Py::new(py, stats)
            })
            .transpose()?;
        Ok(Self {
            projection: output.projection.map(Into::into),
            num_shots,
            num_errors: output.num_errors,
            num_undecoded,
            num_fallback,
            num_scored: num_shots - num_undecoded,
            recovery: output.recovery.into_iter().map(Into::into).collect(),
            logical_error_rate,
            execution_path: plan.path.as_str().to_string(),
            workers_used: plan.workers_used,
            reproducibility_warnings: plan.reproducibility_warnings,
            sampling_seed_used,
            predictions,
            stats,
        })
    }
}

#[pymethods]
impl PyDecodeResult {
    #[getter]
    fn predictions(&self, py: Python<'_>) -> Option<Vec<Py<PyAny>>> {
        self.predictions.as_ref().map(|predictions| {
            predictions
                .iter()
                .map(|prediction| prediction.clone_ref(py))
                .collect()
        })
    }

    #[getter]
    fn stats(&self, py: Python<'_>) -> Option<Py<PyDecodeStats>> {
        self.stats.as_ref().map(|stats| stats.clone_ref(py))
    }

    /// Return the equal-tailed Jeffreys interval `(lo, hi)`.
    ///
    /// The interval helper's internal point estimate is the Jeffreys posterior
    /// mean `(k + 0.5) / (n + 1)`, distinct from this result's empirical `k / n`.
    #[pyo3(signature = (alpha=0.05))]
    fn interval(&self, alpha: f64) -> PyResult<(f64, f64)> {
        if self.num_undecoded > 0 {
            return Err(PyValueError::new_err(
                "logical-error interval is undefined with undecoded shots",
            ));
        }
        pecos_decoders::batch::validate_decode_interval(self.num_shots, alpha)
            .map_err(PyValueError::new_err)?;
        let interval = pecos_num::stats::jeffreys_interval(
            u64::try_from(self.num_errors).unwrap_or(u64::MAX),
            u64::try_from(self.num_shots).unwrap_or(u64::MAX),
            alpha,
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok((interval.lo, interval.hi))
    }

    fn __repr__(&self) -> String {
        format!(
            "DecodeResult(shots={}, errors={}, rate={:.6}, execution_path='{}')",
            self.num_shots, self.num_errors, self.logical_error_rate, self.execution_path
        )
    }
}

pub(super) fn batch_error_to_py(error: BatchDecodeError<DecoderBuildError>) -> PyErr {
    match error {
        BatchDecodeError::Build(DecoderBuildError::Decoder(error)) => {
            decoder_build_error_to_py(error)
        }
        BatchDecodeError::Build(DecoderBuildError::Python(error)) => error,
        BatchDecodeError::Plan(_) | BatchDecodeError::Dimension { .. } => {
            PyValueError::new_err(error.to_string())
        }
        // Missing detector dimension, indexed decode failures, runtime
        // failures, and any executor failure added later.
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}
