//! Execution core and Python result for `SampleBatch.decode`.

use super::decoder_scoring::{DecodeRangeResult, ShotDecodeError, decode_and_score_range};
use super::{PyDecodeStats, PySampleBatch, decoder_build_error_to_py};
use pecos_decoder_core::DecoderError;
use pecos_decoder_core::obs_mask::ObsMask;
use pecos_decoders::batch::{ExecutionPath, ExecutionPlan, IndexedChunk, native_sub_batches};
use pecos_decoders::{DecodeModel, DecoderSpec};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

pub(super) struct BatchExecutionOutput {
    pub(super) num_errors: usize,
    pub(super) predictions: Option<Vec<ObsMask>>,
    pub(super) per_shot_seconds: Option<Vec<f64>>,
    pub(super) wall_elapsed: f64,
}

pub(super) enum BatchExecutionError {
    Build(DecoderError),
    Dimension {
        batch_detectors: usize,
        decoder_detectors: usize,
    },
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
            Self::Build(error) => decoder_build_error_to_py(error),
            Self::Dimension {
                batch_detectors,
                decoder_detectors,
            } => PyValueError::new_err(format!(
                "SampleBatch has {batch_detectors} detectors, but the decoder model has \
                 {decoder_detectors}"
            )),
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

pub(super) fn decode_model(spec: &DecoderSpec, dem: &str) -> DecodeModel {
    spec.embedded_hybrid_full_dem().map_or_else(
        || DecodeModel::SingleDem(dem.to_string()),
        |full| DecodeModel::HybridDem {
            full: full.to_string(),
            decomposed: dem.to_string(),
        },
    )
}

fn preflight_dimensions(
    batch: &PySampleBatch,
    decoder: &dyn pecos_decoders::ObservableDecoder,
) -> Result<(), BatchExecutionError> {
    let decoder_detectors = decoder.num_detectors().ok_or_else(|| {
        BatchExecutionError::Runtime(
            "decoder specification did not report its detector dimension".to_string(),
        )
    })?;
    if decoder_detectors != batch.num_detectors {
        return Err(BatchExecutionError::Dimension {
            batch_detectors: batch.num_detectors,
            decoder_detectors,
        });
    }
    Ok(())
}

fn sequential(
    batch: &PySampleBatch,
    spec: &DecoderSpec,
    model: &DecodeModel,
    predictions: bool,
    timing: bool,
) -> Result<DecodeRangeResult, BatchExecutionError> {
    let mut decoder = spec.build(model).map_err(BatchExecutionError::Build)?;
    preflight_dimensions(batch, decoder.as_ref())?;
    let mut syndrome = vec![0u8; batch.num_detectors];
    decode_and_score_range(
        0..batch.num_shots,
        &mut syndrome,
        |shot, buffer| {
            batch.extract_syndrome(shot, buffer);
            batch.extract_obs_mask_wide(shot)
        },
        decoder.as_mut(),
        predictions,
        timing,
    )
    .map_err(BatchExecutionError::Decode)
}

fn parallel(
    batch: &PySampleBatch,
    spec: &DecoderSpec,
    model: &DecodeModel,
    workers: usize,
    predictions: bool,
    timing: bool,
) -> Result<DecodeRangeResult, BatchExecutionError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|error| BatchExecutionError::Runtime(error.to_string()))?;
    // Workers pull small chunks from a shared cursor instead of taking one fixed
    // contiguous slice each. Per-shot decode cost varies by orders of magnitude
    // for search-based decoders, so a static split leaves workers idle behind a
    // straggler; dynamic chunks let them steal the remaining work. Each worker
    // still builds exactly one decoder, and results carry their chunk index so
    // shot order is restored independently of completion order.
    let next_chunk = std::sync::atomic::AtomicUsize::new(0);
    let num_chunks = batch
        .num_shots
        .div_ceil(pecos_decoders::batch::PARALLEL_CHUNK_SHOTS);
    let worker_results: Vec<Result<Vec<IndexedChunk<DecodeRangeResult>>, BatchExecutionError>> =
        pool.install(|| {
            (0..workers)
                .into_par_iter()
                .map(|_| {
                    // Build even when this worker wins no chunk: an explicit
                    // worker count means exactly that many decoder instances.
                    let mut decoder = spec.build(model).map_err(BatchExecutionError::Build)?;
                    preflight_dimensions(batch, decoder.as_ref())?;
                    let mut syndrome = vec![0u8; batch.num_detectors];
                    let mut mine = Vec::new();
                    loop {
                        let chunk_index =
                            next_chunk.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if chunk_index >= num_chunks {
                            break;
                        }
                        let start = chunk_index * pecos_decoders::batch::PARALLEL_CHUNK_SHOTS;
                        let end = (start + pecos_decoders::batch::PARALLEL_CHUNK_SHOTS)
                            .min(batch.num_shots);
                        let scored = decode_and_score_range(
                            start..end,
                            &mut syndrome,
                            |shot, buffer| {
                                batch.extract_syndrome(shot, buffer);
                                batch.extract_obs_mask_wide(shot)
                            },
                            decoder.as_mut(),
                            predictions,
                            timing,
                        )
                        .map_err(BatchExecutionError::Decode)?;
                        mine.push(IndexedChunk {
                            chunk_index,
                            value: scored,
                        });
                    }
                    Ok(mine)
                })
                .collect()
        });

    // Indexed parallel collection preserves worker order. Inspect every result
    // and choose the lowest failing shot explicitly so scheduler timing cannot
    // affect the reported error.
    let mut chunks = Vec::new();
    let mut decode_errors = Vec::new();
    let mut build_error = None;
    let mut runtime_error = None;
    for result in worker_results {
        match result {
            Ok(mut worker_chunks) => chunks.append(&mut worker_chunks),
            Err(BatchExecutionError::Decode(error)) => decode_errors.push(error),
            Err(BatchExecutionError::Build(error)) if build_error.is_none() => {
                build_error = Some(error);
            }
            Err(BatchExecutionError::Runtime(message)) if runtime_error.is_none() => {
                runtime_error = Some(message);
            }
            Err(BatchExecutionError::Dimension {
                batch_detectors,
                decoder_detectors,
            }) => {
                return Err(BatchExecutionError::Dimension {
                    batch_detectors,
                    decoder_detectors,
                });
            }
            Err(BatchExecutionError::SamplerDimension {
                sampler_detectors,
                decoder_detectors,
            }) => {
                return Err(BatchExecutionError::SamplerDimension {
                    sampler_detectors,
                    decoder_detectors,
                });
            }
            Err(BatchExecutionError::Build(_) | BatchExecutionError::Runtime(_)) => {}
        }
    }
    if let Some(error) = build_error {
        return Err(BatchExecutionError::Build(error));
    }
    if let Some(message) = runtime_error {
        return Err(BatchExecutionError::Runtime(message));
    }
    let indexed_errors =
        decode_errors
            .into_iter()
            .map(|error| pecos_decoders::batch::IndexedDecodeError {
                shot_index: error.shot_index,
                error,
            });
    if let Some(error) = pecos_decoders::batch::lowest_indexed_error(indexed_errors) {
        return Err(BatchExecutionError::Decode(error.error));
    }

    let mut combined = DecodeRangeResult {
        mismatches: 0,
        predictions: if predictions {
            Vec::with_capacity(batch.num_shots)
        } else {
            Vec::new()
        },
        per_shot_seconds: if timing {
            Vec::with_capacity(batch.num_shots)
        } else {
            Vec::new()
        },
    };
    // Restore shot order from the canonical chunk index: workers finish in
    // whatever order the scheduler chose, but the caller sees shot order.
    let expected_chunks = batch
        .num_shots
        .div_ceil(pecos_decoders::batch::PARALLEL_CHUNK_SHOTS);
    let ordered = pecos_decoders::batch::assemble_indexed_chunks(chunks, expected_chunks)
        .map_err(|error| BatchExecutionError::Runtime(error.to_string()))?;
    for mut result in ordered {
        combined.mismatches += result.mismatches;
        combined.predictions.append(&mut result.predictions);
        combined
            .per_shot_seconds
            .append(&mut result.per_shot_seconds);
    }
    Ok(combined)
}

fn native_batch(
    batch: &PySampleBatch,
    spec: &DecoderSpec,
    model: &DecodeModel,
    predictions: bool,
) -> Result<DecodeRangeResult, BatchExecutionError> {
    let mut decoder = spec.build(model).map_err(BatchExecutionError::Build)?;
    preflight_dimensions(batch, decoder.as_ref())?;
    let scratch_len =
        pecos_decoders::batch::native_scratch_len(batch.num_shots, batch.num_detectors)
            .ok_or_else(|| {
                BatchExecutionError::Runtime(
                    "native batch scratch-buffer dimensions overflow usize".to_string(),
                )
            })?;
    let mut scratch = vec![0u8; scratch_len];
    let mut result = DecodeRangeResult {
        mismatches: 0,
        predictions: if predictions {
            Vec::with_capacity(batch.num_shots)
        } else {
            Vec::new()
        },
        per_shot_seconds: Vec::new(),
    };

    for range in native_sub_batches(batch.num_shots) {
        let used = range.len() * batch.num_detectors;
        for (local_shot, shot) in range.clone().enumerate() {
            let row_start = local_shot * batch.num_detectors;
            let row_end = row_start + batch.num_detectors;
            batch.extract_syndrome(shot, &mut scratch[row_start..row_end]);
        }
        let decoded = match decoder.decode_batch_to_observables(
            &scratch[..used],
            range.len(),
            batch.num_detectors,
        ) {
            Ok(decoded) => decoded,
            Err(batch_source) => {
                // A native backend reports one error for the sub-batch. Replay
                // its bounded rows through a fresh instance only on failure to
                // identify the lowest actual failing shot deterministically.
                let mut diagnostic = spec.build(model).map_err(BatchExecutionError::Build)?;
                preflight_dimensions(batch, diagnostic.as_ref())?;
                for (local_shot, shot) in range.clone().enumerate() {
                    let row_start = local_shot * batch.num_detectors;
                    let row_end = row_start + batch.num_detectors;
                    if let Err(source) = diagnostic.decode_obs(&scratch[row_start..row_end]) {
                        return Err(BatchExecutionError::Decode(ShotDecodeError::new(
                            shot, source,
                        )));
                    }
                }
                return Err(BatchExecutionError::Runtime(format!(
                    "native batch decode failed over shots {}..{} (no single shot reproduces the failure): {}",
                    range.start, range.end, batch_source
                )));
            }
        };
        if decoded.len() != range.len() {
            return Err(BatchExecutionError::Decode(ShotDecodeError::new(
                range.start,
                DecoderError::DecodingFailed(format!(
                    "native batch decoder returned {} predictions for {} shots",
                    decoded.len(),
                    range.len()
                )),
            )));
        }
        for (shot, prediction) in range.zip(decoded) {
            result.mismatches += usize::from(prediction != batch.extract_obs_mask_wide(shot));
            if predictions {
                result.predictions.push(prediction);
            }
        }
    }
    Ok(result)
}

pub(super) fn execute(
    batch: &PySampleBatch,
    dem: &str,
    spec: &DecoderSpec,
    plan: &ExecutionPlan,
    predictions: bool,
    timing: bool,
) -> Result<BatchExecutionOutput, BatchExecutionError> {
    let model = decode_model(spec, dem);
    let wall_start = std::time::Instant::now();
    let scored = match plan.path {
        ExecutionPath::Sequential => sequential(batch, spec, &model, predictions, timing)?,
        ExecutionPath::Parallel => {
            parallel(batch, spec, &model, plan.workers_used, predictions, timing)?
        }
        ExecutionPath::NativeBatch => {
            // The planner never selects the native path with timing requested
            // (it cannot produce per-shot samples); assert the cross-crate
            // invariant where we rely on it.
            debug_assert!(
                !timing,
                "planner selected native batch with timing requested"
            );
            native_batch(batch, spec, &model, predictions)?
        }
    };
    Ok(BatchExecutionOutput {
        num_errors: scored.mismatches,
        predictions: predictions.then_some(scored.predictions),
        per_shot_seconds: timing.then_some(scored.per_shot_seconds),
        wall_elapsed: wall_start.elapsed().as_secs_f64(),
    })
}

#[allow(clippy::cast_precision_loss)]
fn logical_error_rate(num_errors: usize, num_shots: usize) -> f64 {
    if num_shots == 0 {
        0.0
    } else {
        num_errors as f64 / num_shots as f64
    }
}

/// Result of decoding and scoring one `SampleBatch`.
#[pyclass(name = "DecodeResult", module = "pecos_rslib.qec", skip_from_py_object)]
pub struct PyDecodeResult {
    #[pyo3(get)]
    num_shots: usize,
    #[pyo3(get)]
    num_errors: usize,
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
    pub(super) fn from_execution(
        py: Python<'_>,
        num_shots: usize,
        plan: ExecutionPlan,
        output: BatchExecutionOutput,
    ) -> PyResult<Self> {
        Self::from_execution_with_seed(py, num_shots, plan, output, None)
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
        let predictions = output
            .predictions
            .map(|masks| {
                masks
                    .iter()
                    .map(|mask| crate::observable_flips_bindings::obsmask_to_py(py, mask))
                    .collect()
            })
            .transpose()?;
        let stats = output
            .per_shot_seconds
            .map(|times| {
                let summed_decode_elapsed = times.iter().sum();
                Py::new(
                    py,
                    PyDecodeStats::from_times_with_elapsed(
                        num_shots,
                        output.num_errors,
                        times,
                        output.wall_elapsed,
                        summed_decode_elapsed,
                    ),
                )
            })
            .transpose()?;
        Ok(Self {
            num_shots,
            num_errors: output.num_errors,
            logical_error_rate: logical_error_rate(output.num_errors, num_shots),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_rate_uses_zero_for_empty_results() {
        assert!(logical_error_rate(0, 0).abs() < f64::EPSILON);
        assert!((logical_error_rate(2, 5) - 0.4).abs() < f64::EPSILON);
    }
}
