//! Fused sampling and planned decoding for `DemSampler.decode`.

use super::batch_decode::{BatchExecutionError, BatchExecutionOutput, decode_model};
use super::decoder_scoring::{DecodeRangeResult, ShotDecodeError};
use pecos_decoder_core::obs_mask::ObsMask;
use pecos_decoder_core::{DecoderError, ObservableDecoder};
use pecos_decoders::batch::{
    ExecutionPath, ExecutionPlan, IndexedChunk, SAMPLING_CHUNK_SHOTS, assemble_indexed_chunks,
    for_each_canonical_sample, sampling_chunks,
};
use pecos_decoders::{DecodeModel, DecoderSpec};
use pecos_qec::fault_tolerance::dem_builder::DemSampler;
use pecos_random::PecosRng;
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub(super) struct DecodeOptions {
    predictions: bool,
    timing: bool,
}

impl DecodeOptions {
    pub(super) const fn new(predictions: bool, timing: bool) -> Self {
        Self {
            predictions,
            timing,
        }
    }
}

struct ShotSampler<'a> {
    sampler: &'a DemSampler,
    observable_mask: ObsMask,
}

impl<'a> ShotSampler<'a> {
    fn new(sampler: &'a DemSampler) -> Self {
        Self {
            sampler,
            observable_mask: sampler.observable_dem_output_mask(),
        }
    }

    fn sample_shot(
        &self,
        rng: &mut PecosRng,
        syndrome: &mut [u8],
        shot_index: usize,
    ) -> Result<ObsMask, ShotDecodeError> {
        // Sampling ABI v1 requires exactly this one single-shot call. In
        // particular, neither `sample_batch` nor `sample_batch_geometric` may be
        // substituted here.
        let (detector_events, dem_output_flips) = self.sampler.sample(rng);
        if detector_events.len() != syndrome.len() {
            return Err(ShotDecodeError::new(
                shot_index,
                DecoderError::DecodingFailed(format!(
                    "sampler returned {} detector events for a {}-detector decoder",
                    detector_events.len(),
                    syndrome.len()
                )),
            ));
        }
        if dem_output_flips.len() != self.sampler.num_dem_outputs() {
            return Err(ShotDecodeError::new(
                shot_index,
                DecoderError::DecodingFailed(format!(
                    "sampler returned {} DEM-output flips for a {}-output sampler",
                    dem_output_flips.len(),
                    self.sampler.num_dem_outputs()
                )),
            ));
        }
        for (value, event) in syndrome.iter_mut().zip(detector_events) {
            *value = u8::from(event);
        }

        // `DemSampler` stores truth as `Vec<bool>`; this helper converts it
        // directly to wide `ObsMask` without narrowing through `u64`.
        Ok(self
            .sampler
            .observable_mask_from_dem_output_flips(&dem_output_flips, &self.observable_mask))
    }

    fn decode_one(
        &self,
        rng: &mut PecosRng,
        syndrome: &mut [u8],
        decoder: &mut dyn ObservableDecoder,
        shot_index: usize,
        result: &mut DecodeRangeResult,
        options: DecodeOptions,
    ) -> Result<(), ShotDecodeError> {
        let truth = self.sample_shot(rng, syndrome, shot_index)?;
        let decode_start = options.timing.then(std::time::Instant::now);
        let decoded = decoder.decode_obs(syndrome);
        if let Some(decode_start) = decode_start {
            result
                .per_shot_seconds
                .push(decode_start.elapsed().as_secs_f64());
        }
        let mut prediction = decoded.map_err(|source| ShotDecodeError::new(shot_index, source))?;
        prediction &= &self.observable_mask;
        result.mismatches += usize::from(prediction != truth);
        if options.predictions {
            result.predictions.push(prediction);
        }
        Ok(())
    }
}

fn preflight_dimensions(
    sampler: &DemSampler,
    decoder: &dyn ObservableDecoder,
) -> Result<(), BatchExecutionError> {
    let decoder_detectors = decoder.num_detectors().ok_or_else(|| {
        BatchExecutionError::Runtime(
            "decoder specification did not report its detector dimension".to_string(),
        )
    })?;
    if decoder_detectors != sampler.num_detectors() {
        return Err(BatchExecutionError::SamplerDimension {
            sampler_detectors: sampler.num_detectors(),
            decoder_detectors,
        });
    }
    Ok(())
}

fn empty_result(num_shots: usize, options: DecodeOptions) -> DecodeRangeResult {
    DecodeRangeResult {
        mismatches: 0,
        predictions: if options.predictions {
            Vec::with_capacity(num_shots)
        } else {
            Vec::new()
        },
        per_shot_seconds: if options.timing {
            Vec::with_capacity(num_shots)
        } else {
            Vec::new()
        },
    }
}

fn sequential(
    sampler: &DemSampler,
    mut decoder: Box<dyn ObservableDecoder>,
    num_shots: usize,
    seed: u64,
    options: DecodeOptions,
) -> Result<DecodeRangeResult, BatchExecutionError> {
    let shot_sampler = ShotSampler::new(sampler);
    let mut syndrome = vec![0u8; sampler.num_detectors()];
    let mut result = empty_result(num_shots, options);

    // The decoder lives outside the canonical sampling callback, so its state
    // advances continuously across the 1024-shot RNG chunk boundaries.
    for_each_canonical_sample(num_shots, seed, |shot_index, rng| {
        shot_sampler.decode_one(
            rng,
            &mut syndrome,
            decoder.as_mut(),
            shot_index,
            &mut result,
            options,
        )
    })
    .map_err(BatchExecutionError::Decode)?;
    Ok(result)
}

fn decode_chunk(
    shot_sampler: &ShotSampler<'_>,
    decoder: &mut dyn ObservableDecoder,
    range: std::ops::Range<usize>,
    chunk_index: usize,
    seed: u64,
    options: DecodeOptions,
) -> Result<DecodeRangeResult, BatchExecutionError> {
    let mut rng = pecos_decoders::batch::sampling_chunk_rng(seed, chunk_index);
    let mut syndrome = vec![0u8; shot_sampler.sampler.num_detectors()];
    let mut result = empty_result(range.len(), options);
    for shot_index in range {
        shot_sampler
            .decode_one(
                &mut rng,
                &mut syndrome,
                decoder,
                shot_index,
                &mut result,
                options,
            )
            .map_err(BatchExecutionError::Decode)?;
    }
    Ok(result)
}

fn combine_chunk_results(
    chunk_results: Vec<Result<DecodeRangeResult, BatchExecutionError>>,
    num_shots: usize,
    options: DecodeOptions,
) -> Result<DecodeRangeResult, BatchExecutionError> {
    let mut successes = Vec::with_capacity(chunk_results.len());
    let mut decode_errors = Vec::new();
    let mut build_error = None;
    let mut runtime_error = None;
    let mut dimension_error = None;

    for result in chunk_results {
        match result {
            Ok(result) => successes.push(result),
            Err(BatchExecutionError::Decode(error)) => decode_errors.push(error),
            Err(BatchExecutionError::Build(error)) if build_error.is_none() => {
                build_error = Some(error);
            }
            Err(BatchExecutionError::Runtime(message)) if runtime_error.is_none() => {
                runtime_error = Some(message);
            }
            Err(BatchExecutionError::SamplerDimension {
                sampler_detectors,
                decoder_detectors,
            }) if dimension_error.is_none() => {
                dimension_error = Some((sampler_detectors, decoder_detectors));
            }
            Err(
                BatchExecutionError::Build(_)
                | BatchExecutionError::Runtime(_)
                | BatchExecutionError::SamplerDimension { .. },
            ) => {}
            Err(BatchExecutionError::Dimension {
                batch_detectors,
                decoder_detectors,
            }) => {
                return Err(BatchExecutionError::Dimension {
                    batch_detectors,
                    decoder_detectors,
                });
            }
        }
    }
    if let Some(error) = build_error {
        return Err(BatchExecutionError::Build(error));
    }
    if let Some((sampler_detectors, decoder_detectors)) = dimension_error {
        return Err(BatchExecutionError::SamplerDimension {
            sampler_detectors,
            decoder_detectors,
        });
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

    let mut combined = empty_result(num_shots, options);
    for mut result in successes {
        combined.mismatches += result.mismatches;
        combined.predictions.append(&mut result.predictions);
        combined
            .per_shot_seconds
            .append(&mut result.per_shot_seconds);
    }
    Ok(combined)
}

fn parallel(
    sampler: &DemSampler,
    spec: &DecoderSpec,
    model: &DecodeModel,
    num_shots: usize,
    seed: u64,
    workers: usize,
    options: DecodeOptions,
) -> Result<DecodeRangeResult, BatchExecutionError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|error| BatchExecutionError::Runtime(error.to_string()))?;
    let chunks = sampling_chunks(num_shots).enumerate().collect::<Vec<_>>();
    let expected_chunks = chunks.len();
    let shot_sampler = ShotSampler::new(sampler);

    let scheduled = pool.install(|| {
        chunks
            .into_par_iter()
            .map_init(
                || spec.build(model).map_err(|error| error.to_string()),
                |decoder, (chunk_index, range)| {
                    // `map_init` state belongs to one Rayon job, not one OS
                    // worker. That is sufficient: stateless chunk results depend
                    // only on the spec and canonical chunk seed, never on which
                    // engine instance Rayon assigns to the chunk.
                    let value = match decoder {
                        Ok(decoder) => decode_chunk(
                            &shot_sampler,
                            decoder.as_mut(),
                            range,
                            chunk_index,
                            seed,
                            options,
                        ),
                        Err(message) => Err(BatchExecutionError::Runtime(format!(
                            "parallel decoder construction failed: {message}"
                        ))),
                    };
                    IndexedChunk { chunk_index, value }
                },
            )
            .collect::<Vec<_>>()
    });

    let ordered = assemble_indexed_chunks(scheduled, expected_chunks)
        .map_err(|error| BatchExecutionError::Runtime(error.to_string()))?;
    combine_chunk_results(ordered, num_shots, options)
}

fn native_batch(
    sampler: &DemSampler,
    spec: &DecoderSpec,
    model: &DecodeModel,
    mut decoder: Box<dyn ObservableDecoder>,
    num_shots: usize,
    seed: u64,
    predictions: bool,
) -> Result<DecodeRangeResult, BatchExecutionError> {
    let num_detectors = sampler.num_detectors();
    let scratch_len = SAMPLING_CHUNK_SHOTS
        .min(num_shots)
        .checked_mul(num_detectors)
        .ok_or_else(|| {
            BatchExecutionError::Runtime(
                "native batch scratch-buffer dimensions overflow usize".to_string(),
            )
        })?;
    let mut scratch = vec![0u8; scratch_len];
    let shot_sampler = ShotSampler::new(sampler);
    let mut result = empty_result(
        num_shots,
        DecodeOptions {
            predictions,
            timing: false,
        },
    );

    for (chunk_index, range) in sampling_chunks(num_shots).enumerate() {
        let mut rng = pecos_decoders::batch::sampling_chunk_rng(seed, chunk_index);
        let used = range.len().checked_mul(num_detectors).ok_or_else(|| {
            BatchExecutionError::Runtime(
                "native batch scratch-buffer dimensions overflow usize".to_string(),
            )
        })?;
        let mut truth_masks = Vec::with_capacity(range.len());
        for (local_shot, shot_index) in range.clone().enumerate() {
            let row_start = local_shot * num_detectors;
            let row_end = row_start + num_detectors;
            truth_masks.push(
                shot_sampler
                    .sample_shot(&mut rng, &mut scratch[row_start..row_end], shot_index)
                    .map_err(BatchExecutionError::Decode)?,
            );
        }

        let decoded = match decoder.decode_batch_to_observables(
            &scratch[..used],
            range.len(),
            num_detectors,
        ) {
            Ok(decoded) => decoded,
            Err(batch_source) => {
                // Reuse the already sampled, bounded rows: diagnostics must not
                // resample because that would consume a different ABI stream.
                let mut diagnostic = spec.build(model).map_err(BatchExecutionError::Build)?;
                for (local_shot, shot_index) in range.clone().enumerate() {
                    let row_start = local_shot * num_detectors;
                    let row_end = row_start + num_detectors;
                    if let Err(source) = diagnostic.decode_obs(&scratch[row_start..row_end]) {
                        return Err(BatchExecutionError::Decode(ShotDecodeError::new(
                            shot_index, source,
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
        for (mut prediction, truth) in decoded.into_iter().zip(truth_masks) {
            prediction &= &shot_sampler.observable_mask;
            result.mismatches += usize::from(prediction != truth);
            if predictions {
                result.predictions.push(prediction);
            }
        }
    }
    Ok(result)
}

pub(super) fn execute(
    sampler: &DemSampler,
    dem: &str,
    spec: &DecoderSpec,
    plan: &ExecutionPlan,
    num_shots: usize,
    seed: u64,
    options: DecodeOptions,
) -> Result<BatchExecutionOutput, BatchExecutionError> {
    let model = decode_model(spec, dem);
    // The wall clock deliberately starts immediately before decoder
    // construction, and therefore includes preflight, sampling, decoding, and
    // scoring while excluding the later conversion into Python objects.
    let wall_start = std::time::Instant::now();
    let decoder = spec.build(&model).map_err(BatchExecutionError::Build)?;
    preflight_dimensions(sampler, decoder.as_ref())?;

    let scored = match plan.path {
        ExecutionPath::Sequential => sequential(sampler, decoder, num_shots, seed, options)?,
        ExecutionPath::Parallel => {
            // The unconditional preflight engine above is intentionally not one
            // of the per-job engines used for independent stateless chunks.
            drop(decoder);
            parallel(
                sampler,
                spec,
                &model,
                num_shots,
                seed,
                plan.workers_used,
                options,
            )?
        }
        ExecutionPath::NativeBatch => {
            // The planner never selects the native path with timing requested
            // (it cannot produce per-shot samples); assert the cross-crate
            // invariant where we rely on it so a future planner change fails
            // loudly instead of silently returning empty stats.
            debug_assert!(
                !options.timing,
                "planner selected native batch with timing requested"
            );
            native_batch(
                sampler,
                spec,
                &model,
                decoder,
                num_shots,
                seed,
                options.predictions,
            )?
        }
    };
    Ok(BatchExecutionOutput {
        num_errors: scored.mismatches,
        predictions: options.predictions.then_some(scored.predictions),
        per_shot_seconds: options.timing.then_some(scored.per_shot_seconds),
        wall_elapsed: wall_start.elapsed().as_secs_f64(),
    })
}
