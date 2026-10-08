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

//! Batch execution with one local decoder per worker.
use super::{
    DecodeRangeResult, ExecutionPath, ExecutionPlan, ExecutionPlanError, ExecutionPlanInputs,
    IndexedChunk, SampleBatch, ShotDecodeError, batch_worker_cap, decode_and_score_range,
    native_sub_batches, plan_execution,
};
use crate::{DecodeModel, DecoderError, DecoderSpec, ExecutionTraits, ObservableDecoder};
use pecos_decoder_core::obs_mask::ObsMask;
use rayon::prelude::*;
use std::fmt;

/// Builds worker-local decoders without requiring the decoders to be `Send`.
///
/// The factory is `Sync` because parallel workers share it while building their
/// own decoders. Each decoder is built and used on the same worker.
pub trait DecoderFactory: Sync {
    /// Original construction failure, transferable from a worker to the caller.
    type BuildError: Send;
    /// Cross-shot state and wall-clock dependencies used by the planner.
    fn execution_traits(&self) -> ExecutionTraits;
    /// Whether built decoders implement native observable batch decoding.
    ///
    /// Returning true permits the planner to call `decode_batch_to_observables`
    /// instead of `decode_obs`. It must return exactly one prediction per shot,
    /// in input order, or a decoding error.
    fn native_batch_capable(&self) -> bool;
    /// Select the model before execution timing starts; defaults to `SingleDem`.
    fn decode_model(&self, dem: &str) -> DecodeModel {
        DecodeModel::SingleDem(dem.to_string())
    }
    /// Construct one decoder.
    ///
    /// # Errors
    /// Returns the factory's original construction error.
    fn build(&self, model: &DecodeModel) -> Result<Box<dyn ObservableDecoder>, Self::BuildError>;
}

impl DecoderFactory for DecoderSpec {
    type BuildError = DecoderError;
    fn execution_traits(&self) -> ExecutionTraits {
        self.execution_traits()
    }
    fn native_batch_capable(&self) -> bool {
        self.native_batch_capable()
    }
    fn decode_model(&self, dem: &str) -> DecodeModel {
        self.embedded_hybrid_full_dem().map_or_else(
            || DecodeModel::SingleDem(dem.to_string()),
            |full| DecodeModel::HybridDem {
                full: full.to_string(),
                decomposed: dem.to_string(),
            },
        )
    }
    fn build(&self, model: &DecodeModel) -> Result<Box<dyn ObservableDecoder>, Self::BuildError> {
        self.build(model)
    }
}

/// Execution and retention options for a batch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct DecodeOptions {
    /// Requested worker count; `None` selects automatic planning. Zero is invalid.
    pub workers: Option<usize>,
    /// Retain one wide prediction per shot; disabled by default.
    pub predictions: bool,
    /// Retain per-shot decode seconds, disabling native batch dispatch; false by default.
    pub timing: bool,
}
impl DecodeOptions {
    /// Request a worker count; zero is rejected when decoding.
    #[must_use]
    pub const fn workers(mut self, workers: usize) -> Self {
        self.workers = Some(workers);
        self
    }
    /// Choose whether to retain predictions.
    #[must_use]
    pub const fn predictions(mut self, predictions: bool) -> Self {
        self.predictions = predictions;
        self
    }
    /// Choose whether to retain per-shot decode timings.
    #[must_use]
    pub const fn timing(mut self, timing: bool) -> Self {
        self.timing = timing;
        self
    }
}

/// Scored shots in their original order.
#[derive(Debug)]
#[non_exhaustive]
pub struct DecodeResult {
    /// Number of decoded shots.
    pub num_shots: usize,
    /// Number of predictions that differ from their true observable flips.
    pub num_errors: usize,
    /// Execution mechanism selected by the planner.
    pub execution_path: ExecutionPath,
    /// Actual worker count, at most one per shot; an empty batch keeps one worker.
    pub workers_used: usize,
    /// Planner warnings about execution-dependent reproducibility.
    pub reproducibility_warnings: Vec<String>,
    /// Predictions in shot order, or `None` when retention was not requested.
    pub predictions: Option<Vec<ObsMask>>,
    /// Seconds spent in each `decode_obs` call, excluding extraction and scoring.
    pub per_shot_seconds: Option<Vec<f64>>,
    /// Wall seconds after model selection, including decoder construction and execution.
    pub wall_elapsed: f64,
}
impl DecodeResult {
    /// Empirical mismatch fraction, or zero for an empty batch.
    #[must_use]
    pub fn logical_error_rate(&self) -> f64 {
        logical_error_rate(self.num_errors, self.num_shots)
    }
}

/// Planning, construction, dimension, or execution failure.
#[derive(Debug)]
pub enum BatchDecodeError<E> {
    /// Invalid planner inputs.
    Plan(ExecutionPlanError),
    /// Original factory build error, preserved unchanged.
    Build(E),
    /// Batch and decoder detector dimensions differ.
    Dimension {
        /// Detector columns in the batch.
        batch_detectors: usize,
        /// Detector dimension reported by this decoder.
        decoder_detectors: usize,
    },
    /// The decoder does not report its detector dimension.
    MissingDetectorDimension,
    /// Per-shot decode failure with its absolute shot index.
    Decode(ShotDecodeError),
    /// Thread-pool or other execution failure.
    Runtime(String),
}
impl<E: fmt::Display> fmt::Display for BatchDecodeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(error) => error.fmt(f),
            Self::Build(error) => error.fmt(f),
            Self::Dimension {
                batch_detectors,
                decoder_detectors,
            } => write!(
                f,
                "SampleBatch has {batch_detectors} detectors, but the decoder model has {decoder_detectors}"
            ),
            Self::MissingDetectorDimension => {
                f.write_str("decoder specification did not report its detector dimension")
            }
            Self::Decode(error) => error.fmt(f),
            Self::Runtime(message) => f.write_str(message),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for BatchDecodeError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Build(error) => error.source(),
            Self::Decode(error) => error.source(),
            Self::Plan(error) => error.source(),
            Self::Dimension { .. } | Self::MissingDetectorDimension | Self::Runtime(_) => None,
        }
    }
}

impl SampleBatch {
    /// Decode and score using automatic execution and no retained per-shot data.
    ///
    /// # Errors
    /// Returns planning, factory, dimension, or indexed decoding errors.
    pub fn decode<F: DecoderFactory + ?Sized>(
        &self,
        dem: &str,
        factory: &F,
    ) -> Result<DecodeResult, BatchDecodeError<F::BuildError>> {
        self.decode_with(dem, factory, &DecodeOptions::default())
    }
    /// Decode and score with explicit execution and retention options.
    ///
    /// # Errors
    /// Returns planning, factory, dimension, or indexed decoding errors.
    pub fn decode_with<F: DecoderFactory + ?Sized>(
        &self,
        dem: &str,
        factory: &F,
        options: &DecodeOptions,
    ) -> Result<DecodeResult, BatchDecodeError<F::BuildError>> {
        let mut plan = plan_execution(ExecutionPlanInputs {
            traits: factory.execution_traits(),
            num_shots: self.num_shots(),
            native_batch_capable: factory.native_batch_capable(),
            timing: options.timing,
            explicit_workers: options.workers,
            available_threads: rayon::current_num_threads(),
        })
        .map_err(BatchDecodeError::Plan)?;
        // A worker needs at least one shot; additional workers would only idle.
        // An empty batch keeps one worker, which builds its decoder and decodes nothing.
        if plan.path == ExecutionPath::Parallel {
            plan.workers_used = plan.workers_used.min(batch_worker_cap(self.num_shots()));
        }
        execute(
            self,
            dem,
            factory,
            &plan,
            options.predictions,
            options.timing,
        )
    }
}

fn preflight_dimensions<E>(
    batch: &SampleBatch,
    decoder: &dyn crate::ObservableDecoder,
) -> Result<(), BatchDecodeError<E>> {
    let decoder_detectors = decoder
        .num_detectors()
        .ok_or(BatchDecodeError::MissingDetectorDimension)?;
    if decoder_detectors != batch.num_detectors() {
        return Err(BatchDecodeError::Dimension {
            batch_detectors: batch.num_detectors(),
            decoder_detectors,
        });
    }
    Ok(())
}

fn sequential<F: DecoderFactory + ?Sized>(
    batch: &SampleBatch,
    spec: &F,
    model: &DecodeModel,
    predictions: bool,
    timing: bool,
) -> Result<DecodeRangeResult, BatchDecodeError<F::BuildError>> {
    let mut decoder = spec.build(model).map_err(BatchDecodeError::Build)?;
    preflight_dimensions(batch, decoder.as_ref())?;
    let mut syndrome = vec![0u8; batch.num_detectors()];
    decode_and_score_range(
        0..batch.num_shots(),
        &mut syndrome,
        |shot, buffer| {
            batch.syndrome_into(shot, buffer);
            batch.observable_flips(shot)
        },
        decoder.as_mut(),
        predictions,
        timing,
    )
    .map_err(BatchDecodeError::Decode)
}

type WorkerResult<E> = Result<Vec<IndexedChunk<DecodeRangeResult>>, BatchDecodeError<E>>;

fn parallel<F: DecoderFactory + ?Sized>(
    batch: &SampleBatch,
    spec: &F,
    model: &DecodeModel,
    workers: usize,
    predictions: bool,
    timing: bool,
) -> Result<DecodeRangeResult, BatchDecodeError<F::BuildError>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|error| BatchDecodeError::Runtime(error.to_string()))?;
    // Workers pull small chunks from a shared cursor instead of taking one fixed
    // contiguous slice each. Per-shot decode cost varies by orders of magnitude
    // for search-based decoders, so a static split leaves workers idle behind a
    // straggler; dynamic chunks let them steal the remaining work. Each worker
    // still builds exactly one decoder, and results carry their chunk index so
    // shot order is restored independently of completion order.
    let next_chunk = std::sync::atomic::AtomicUsize::new(0);
    let chunk_shots = crate::batch::parallel_chunk_shots(batch.num_shots(), workers);
    let num_chunks = batch.num_shots().div_ceil(chunk_shots);
    let worker_results: Vec<WorkerResult<F::BuildError>> = pool.install(|| {
        (0..workers)
            .into_par_iter()
            .map(|_| {
                // Build even when this worker wins no chunk: the planned
                // worker count means exactly that many decoder instances.
                let mut decoder = spec.build(model).map_err(BatchDecodeError::Build)?;
                preflight_dimensions(batch, decoder.as_ref())?;
                let mut syndrome = vec![0u8; batch.num_detectors()];
                let mut mine = Vec::new();
                loop {
                    let chunk_index = next_chunk.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if chunk_index >= num_chunks {
                        break;
                    }
                    let start = chunk_index * chunk_shots;
                    let end = (start + chunk_shots).min(batch.num_shots());
                    let scored = decode_and_score_range(
                        start..end,
                        &mut syndrome,
                        |shot, buffer| {
                            batch.syndrome_into(shot, buffer);
                            batch.observable_flips(shot)
                        },
                        decoder.as_mut(),
                        predictions,
                        timing,
                    )
                    .map_err(BatchDecodeError::Decode)?;
                    mine.push(IndexedChunk {
                        chunk_index,
                        value: scored,
                    });
                }
                Ok(mine)
            })
            .collect()
    });

    reduce_worker_results(
        worker_results,
        batch.num_shots(),
        num_chunks,
        predictions,
        timing,
    )
}

/// Reduce results in logical worker order, restoring shot order on success.
fn reduce_worker_results<E>(
    worker_results: Vec<WorkerResult<E>>,
    num_shots: usize,
    num_chunks: usize,
    predictions: bool,
    timing: bool,
) -> Result<DecodeRangeResult, BatchDecodeError<E>> {
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
            Err(BatchDecodeError::Decode(error)) => decode_errors.push(error),
            Err(BatchDecodeError::Build(error)) if build_error.is_none() => {
                build_error = Some(error);
            }
            Err(BatchDecodeError::Runtime(message)) if runtime_error.is_none() => {
                runtime_error = Some(BatchDecodeError::Runtime(message));
            }
            Err(BatchDecodeError::Dimension {
                batch_detectors,
                decoder_detectors,
            }) => {
                return Err(BatchDecodeError::Dimension {
                    batch_detectors,
                    decoder_detectors,
                });
            }
            Err(BatchDecodeError::MissingDetectorDimension) if runtime_error.is_none() => {
                runtime_error = Some(BatchDecodeError::MissingDetectorDimension);
            }
            Err(BatchDecodeError::Plan(error)) => return Err(BatchDecodeError::Plan(error)),
            Err(
                BatchDecodeError::Build(_)
                | BatchDecodeError::Runtime(_)
                | BatchDecodeError::MissingDetectorDimension,
            ) => {}
        }
    }
    if let Some(error) = build_error {
        return Err(BatchDecodeError::Build(error));
    }
    if let Some(message) = runtime_error {
        return Err(message);
    }
    let indexed_errors = decode_errors
        .into_iter()
        .map(|error| crate::batch::IndexedDecodeError {
            shot_index: error.shot_index,
            error,
        });
    if let Some(error) = crate::batch::lowest_indexed_error(indexed_errors) {
        return Err(BatchDecodeError::Decode(error.error));
    }

    let mut combined = DecodeRangeResult {
        mismatches: 0,
        predictions: if predictions {
            Vec::with_capacity(num_shots)
        } else {
            Vec::new()
        },
        per_shot_seconds: if timing {
            Vec::with_capacity(num_shots)
        } else {
            Vec::new()
        },
    };
    // Restore shot order from the canonical chunk index: workers finish in
    // whatever order the scheduler chose, but the caller sees shot order.
    let ordered = crate::batch::assemble_indexed_chunks(chunks, num_chunks)
        .map_err(|error| BatchDecodeError::Runtime(error.to_string()))?;
    for mut result in ordered {
        combined.mismatches += result.mismatches;
        combined.predictions.append(&mut result.predictions);
        combined
            .per_shot_seconds
            .append(&mut result.per_shot_seconds);
    }
    Ok(combined)
}

fn native_batch<F: DecoderFactory + ?Sized>(
    batch: &SampleBatch,
    spec: &F,
    model: &DecodeModel,
    predictions: bool,
) -> Result<DecodeRangeResult, BatchDecodeError<F::BuildError>> {
    let mut decoder = spec.build(model).map_err(BatchDecodeError::Build)?;
    preflight_dimensions(batch, decoder.as_ref())?;
    let scratch_len = crate::batch::native_scratch_len(batch.num_shots(), batch.num_detectors())
        .ok_or_else(|| {
            BatchDecodeError::Runtime(
                "native batch scratch-buffer dimensions overflow usize".to_string(),
            )
        })?;
    let mut scratch = vec![0u8; scratch_len];
    let mut result = DecodeRangeResult {
        mismatches: 0,
        predictions: if predictions {
            Vec::with_capacity(batch.num_shots())
        } else {
            Vec::new()
        },
        per_shot_seconds: Vec::new(),
    };

    for range in native_sub_batches(batch.num_shots()) {
        let used = range.len() * batch.num_detectors();
        for (local_shot, shot) in range.clone().enumerate() {
            let row_start = local_shot * batch.num_detectors();
            let row_end = row_start + batch.num_detectors();
            batch.syndrome_into(shot, &mut scratch[row_start..row_end]);
        }
        let decoded = match decoder.decode_batch_to_observables(
            &scratch[..used],
            range.len(),
            batch.num_detectors(),
        ) {
            Ok(decoded) => decoded,
            Err(batch_source) => {
                // A native backend reports one error for the sub-batch. Replay
                // its bounded rows through a fresh instance only on failure to
                // identify the lowest actual failing shot deterministically.
                let mut diagnostic = spec.build(model).map_err(BatchDecodeError::Build)?;
                preflight_dimensions(batch, diagnostic.as_ref())?;
                for (local_shot, shot) in range.clone().enumerate() {
                    let row_start = local_shot * batch.num_detectors();
                    let row_end = row_start + batch.num_detectors();
                    if let Err(source) = diagnostic.decode_obs(&scratch[row_start..row_end]) {
                        return Err(BatchDecodeError::Decode(ShotDecodeError::new(shot, source)));
                    }
                }
                return Err(BatchDecodeError::Runtime(format!(
                    "native batch decode failed over shots {}..{} (no single shot reproduces the failure): {}",
                    range.start, range.end, batch_source
                )));
            }
        };
        if decoded.len() != range.len() {
            return Err(BatchDecodeError::Decode(ShotDecodeError::new(
                range.start,
                DecoderError::DecodingFailed(format!(
                    "native batch decoder returned {} predictions for {} shots",
                    decoded.len(),
                    range.len()
                )),
            )));
        }
        for (shot, prediction) in range.zip(decoded) {
            result.mismatches += usize::from(prediction != batch.observable_flips(shot));
            if predictions {
                result.predictions.push(prediction);
            }
        }
    }
    Ok(result)
}

pub(super) fn execute<F: DecoderFactory + ?Sized>(
    batch: &SampleBatch,
    dem: &str,
    spec: &F,
    plan: &ExecutionPlan,
    predictions: bool,
    timing: bool,
) -> Result<DecodeResult, BatchDecodeError<F::BuildError>> {
    let model = spec.decode_model(dem);
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
    Ok(DecodeResult {
        num_shots: batch.num_shots(),
        execution_path: plan.path,
        workers_used: plan.workers_used,
        reproducibility_warnings: plan.reproducibility_warnings.clone(),
        num_errors: scored.mismatches,
        predictions: predictions.then_some(scored.predictions),
        per_shot_seconds: timing.then_some(scored.per_shot_seconds),
        wall_elapsed: wall_start.elapsed().as_secs_f64(),
    })
}

/// Empirical mismatch fraction from scored counts, or zero for an empty batch.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn logical_error_rate(num_errors: usize, num_shots: usize) -> f64 {
    if num_shots == 0 {
        0.0
    } else {
        num_errors as f64 / num_shots as f64
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

    fn decode_failure(shot_index: usize) -> BatchDecodeError<&'static str> {
        BatchDecodeError::Decode(ShotDecodeError::new(
            shot_index,
            DecoderError::DecodingFailed(format!("shot {shot_index}")),
        ))
    }

    #[test]
    fn reduction_selects_lowest_decode_error_in_every_worker_position() {
        // These are the logical worker results passed to the SAME reduction
        // function called by parallel(), not inputs to its selection helper.
        // Position the lowest shot after earlier worker errors explicitly, so
        // changing this reduction to report its first error fails deterministically.
        for workers in [2, 3, 7] {
            for lowest_worker in 0..workers {
                let results = (0..workers)
                    .map(|worker| {
                        Err(decode_failure(if worker == lowest_worker {
                            17
                        } else {
                            64 * (worker + 1)
                        }))
                    })
                    .collect();
                let error = reduce_worker_results(results, 1301, 21, false, false).unwrap_err();
                let BatchDecodeError::Decode(error) = error else {
                    panic!("expected indexed decode failure");
                };
                assert_eq!(
                    error.shot_index, 17,
                    "workers={workers}, lowest_worker={lowest_worker}"
                );
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Failure {
        Dimension(usize),
        Plan(usize),
        Build(&'static str),
        Runtime(&'static str),
        Missing,
        Decode(usize),
    }

    impl Failure {
        fn error(self) -> BatchDecodeError<&'static str> {
            match self {
                Self::Dimension(decoder_detectors) => BatchDecodeError::Dimension {
                    batch_detectors: 1,
                    decoder_detectors,
                },
                Self::Plan(workers) => {
                    BatchDecodeError::Plan(ExecutionPlanError::HistoryDependentParallel { workers })
                }
                Self::Build(message) => BatchDecodeError::Build(message),
                Self::Runtime(message) => BatchDecodeError::Runtime(message.into()),
                Self::Missing => BatchDecodeError::MissingDetectorDimension,
                Self::Decode(shot) => decode_failure(shot),
            }
        }
    }

    #[test]
    fn reduction_preserves_error_precedence_and_first_worker_ties() {
        use Failure::{Build, Decode, Dimension, Missing, Plan, Runtime};
        let cases = [
            // Dimension and Plan immediately return whichever occurs first,
            // even if deferred build/runtime/decode errors were seen before it.
            (
                vec![
                    Decode(0),
                    Build("first"),
                    Runtime("first"),
                    Missing,
                    Dimension(2),
                    Plan(3),
                ],
                Dimension(2),
            ),
            (
                vec![
                    Decode(0),
                    Build("first"),
                    Runtime("first"),
                    Missing,
                    Plan(3),
                    Dimension(2),
                ],
                Plan(3),
            ),
            (
                vec![Dimension(2), Dimension(3), Build("first")],
                Dimension(2),
            ),
            (vec![Plan(2), Plan(3), Build("first")], Plan(2)),
            // Keep the first build error, regardless of surrounding lower priorities.
            (
                vec![
                    Decode(0),
                    Missing,
                    Runtime("first"),
                    Build("first"),
                    Build("second"),
                ],
                Build("first"),
            ),
            (
                vec![Build("first"), Runtime("first"), Build("second"), Decode(0)],
                Build("first"),
            ),
            // Runtime and Missing share a priority and keep the first worker's error.
            (
                vec![Decode(0), Runtime("first"), Missing, Runtime("second")],
                Runtime("first"),
            ),
            (vec![Decode(0), Missing, Runtime("first"), Missing], Missing),
            (
                vec![Runtime("first"), Runtime("second"), Decode(0)],
                Runtime("first"),
            ),
            (vec![Decode(64), Decode(17)], Decode(17)),
        ];
        for (failures, expected) in cases {
            let results = failures
                .iter()
                .copied()
                .map(|failure| Err(failure.error()))
                .collect();
            let actual = reduce_worker_results(results, 1301, 21, false, false).unwrap_err();
            assert_eq!(
                format!("{actual:?}"),
                format!("{:?}", expected.error()),
                "{failures:?}"
            );
        }
    }

    #[test]
    fn reduction_restores_chunk_order_and_combines_scoring() {
        let chunk = |chunk_index, bit, mismatches, seconds| {
            let mut mask = ObsMask::new();
            mask.set(bit);
            IndexedChunk {
                chunk_index,
                value: DecodeRangeResult {
                    mismatches,
                    predictions: vec![mask],
                    per_shot_seconds: vec![seconds],
                },
            }
        };
        let results: Vec<WorkerResult<&str>> = vec![
            Ok(vec![chunk(2, 129, 1, 0.3), chunk(0, 0, 1, 0.1)]),
            Ok(vec![]),
            Ok(vec![chunk(1, 64, 0, 0.2)]),
        ];
        let combined = reduce_worker_results(results, 3, 3, true, true).unwrap();
        assert_eq!(combined.mismatches, 2);
        let bits: Vec<_> = combined
            .predictions
            .iter()
            .map(|mask| mask.iter_set_bits().collect::<Vec<_>>())
            .collect();
        assert_eq!(bits, vec![vec![0], vec![64], vec![129]]);
        assert_eq!(combined.per_shot_seconds, vec![0.1, 0.2, 0.3]);
    }
}
