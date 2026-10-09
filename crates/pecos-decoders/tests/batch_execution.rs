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

//! Backend-independent contracts for the public batch executor.
use pecos_decoders::ObsMask;
use pecos_decoders::batch::{
    BatchDecodeError, DecodeOptions, DecoderFactory, ExecutionPath, SampleBatch,
};
use pecos_decoders::{DecodeModel, DecoderError, ExecutionTraits, ObservableDecoder};
use std::rc::Rc;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
const WIDTH: usize = 12;
#[derive(Clone, Copy, Default)]
enum Behavior {
    #[default]
    Healthy,
    Dimension,
    Missing,
    Build,
    FailShots,
    NativeFailure,
    Unreproducible,
    WrongCount,
}
#[derive(Default)]
struct Factory {
    native: bool,
    behaviors: Vec<Behavior>,
    builds: AtomicUsize,
    calls: Arc<AtomicUsize>,
    native_calls: Arc<AtomicUsize>,
    traits: ExecutionTraits,
}
impl Factory {
    fn with(behavior: Behavior) -> Self {
        Self {
            behaviors: vec![behavior],
            ..Self::default()
        }
    }
}
struct Decoder {
    behavior: Behavior,
    calls: Arc<AtomicUsize>,
    native_calls: Arc<AtomicUsize>,
    native_failed: bool,
    // Compile-time proof that built decoders need not be Send.
    _local: Rc<()>,
}
impl DecoderFactory for Factory {
    type BuildError = DecoderError;
    fn execution_traits(&self) -> ExecutionTraits {
        self.traits
    }
    fn native_batch_capable(&self) -> bool {
        self.native
    }
    fn build(&self, model: &DecodeModel) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
        assert_eq!(model, &DecodeModel::SingleDem("test DEM".into()));
        let index = self.builds.fetch_add(1, Ordering::SeqCst);
        let behavior = self
            .behaviors
            .get(index % self.behaviors.len().max(1))
            .copied()
            .unwrap_or_default();
        if matches!(behavior, Behavior::Build) {
            return Err(DecoderError::InvalidConfiguration(
                "original build error".into(),
            ));
        }
        Ok(Box::new(Decoder {
            behavior,
            calls: self.calls.clone(),
            native_calls: self.native_calls.clone(),
            native_failed: false,
            _local: Rc::new(()),
        }))
    }
}
fn index(syndrome: &[u8]) -> usize {
    syndrome
        .iter()
        .enumerate()
        .fold(0, |n, (bit, &v)| n | (usize::from(v) << bit))
}
fn prediction(n: usize) -> ObsMask {
    let mut mask = ObsMask::new();
    mask.set(n % 130);
    mask
}
impl ObservableDecoder for Decoder {
    fn num_detectors(&self) -> Option<usize> {
        match self.behavior {
            Behavior::Dimension => Some(WIDTH + 1),
            Behavior::Missing => None,
            _ => Some(WIDTH),
        }
    }
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        assert!(!self.native_failed, "diagnosis must use a fresh decoder");
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(
            !matches!(self.behavior, Behavior::Dimension | Behavior::Missing),
            "preflight must reject before decoding"
        );
        let n = index(syndrome);
        if matches!(self.behavior, Behavior::FailShots | Behavior::NativeFailure) && n >= 17 {
            return Err(DecoderError::DecodingFailed(format!("failure {n}")));
        }
        Ok(prediction(n))
    }
    fn decode_batch_to_observables(
        &mut self,
        syndromes: &[u8],
        num_shots: usize,
        num_detectors: usize,
    ) -> Result<Vec<ObsMask>, DecoderError> {
        self.native_calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(num_detectors, WIDTH);
        assert_eq!(syndromes.len(), num_shots * WIDTH);
        match self.behavior {
            Behavior::NativeFailure | Behavior::Unreproducible => {
                self.native_failed = true;
                Err(DecoderError::DecodingFailed("native failure".into()))
            }
            Behavior::WrongCount => Ok(Vec::new()),
            _ => Ok(syndromes
                .chunks(WIDTH)
                .map(|row| prediction(index(row)))
                .collect()),
        }
    }
}
fn batch(n: usize) -> SampleBatch {
    if n == 0 {
        return SampleBatch::from_columnar(vec![vec![]; WIDTH], vec![vec![]; 130], 0).unwrap();
    }
    let rows: Vec<Vec<u8>> = (0..n)
        .map(|shot| {
            (0..WIDTH)
                .map(|bit| u8::from(shot & (1 << bit) != 0))
                .collect()
        })
        .collect();
    let truths: Vec<_> = (0..n)
        .map(|shot| {
            if shot % 3 == 0 {
                ObsMask::new()
            } else {
                prediction(shot)
            }
        })
        .collect();
    SampleBatch::from_row_major(&rows, &truths, 130).unwrap()
}
fn options(workers: Option<usize>) -> DecodeOptions {
    let mut options = DecodeOptions::default();
    options.workers = workers;
    options
}
fn run(
    n: usize,
    factory: &Factory,
    options: DecodeOptions,
) -> Result<pecos_decoders::batch::DecodeResult, BatchDecodeError<DecoderError>> {
    batch(n).decode_with("test DEM", factory, &options)
}
#[test]
fn paths_preserve_order_counts_and_wide_predictions() {
    let expected: Vec<_> = (0..1301).map(prediction).collect();
    for workers in [None, Some(1), Some(2), Some(3), Some(7)] {
        let factory = Factory {
            native: workers.is_none(),
            ..Factory::default()
        };
        let result = run(1301, &factory, options(workers).predictions(true)).unwrap();
        assert_eq!(result.num_shots, 1301);
        assert_eq!(result.num_errors, 434);
        assert_eq!(result.predictions, Some(expected.clone()));
        assert_eq!(
            result.execution_path,
            match workers {
                None => ExecutionPath::NativeBatch,
                Some(1) => ExecutionPath::Sequential,
                _ => ExecutionPath::Parallel,
            }
        );
        assert_eq!(result.workers_used, workers.unwrap_or(1));
        assert!(result.per_shot_seconds.is_none());
        assert_eq!(result.reproducibility_warnings, Vec::<String>::new());
        assert_eq!(factory.builds.load(Ordering::SeqCst), workers.unwrap_or(1));
        assert_eq!(
            factory.native_calls.load(Ordering::SeqCst),
            if workers.is_none() { 2 } else { 0 }
        );
        assert_eq!(
            factory.calls.load(Ordering::SeqCst),
            if workers.is_none() { 0 } else { 1301 }
        );
    }
}
#[test]
fn lowest_failure_is_independent_of_workers() {
    // End-to-end coverage of actual decoder workers and absolute shot indices.
    // Adversarial logical worker orders are tested deterministically against
    // reduce_worker_results in execution.rs, independently of Rayon scheduling.
    for workers in [1, 2, 3, 7] {
        let factory = Factory::with(Behavior::FailShots);
        let error = run(1301, &factory, DecodeOptions::default().workers(workers)).unwrap_err();
        let BatchDecodeError::Decode(error) = error else {
            panic!("expected indexed decode failure")
        };
        assert_eq!(error.shot_index, 17);
        assert_eq!(
            error.to_string(),
            "decoder failed on shot 17: Decoding failed: failure 17"
        );
        assert_eq!(factory.builds.load(Ordering::SeqCst), workers);
    }
}
#[test]
fn dimensions_are_checked_per_instance_before_decoding() {
    for behavior in [Behavior::Dimension, Behavior::Missing] {
        for workers in [None, Some(1), Some(3)] {
            let factory = Factory {
                native: workers.is_none(),
                ..Factory::with(behavior)
            };
            let error = run(5, &factory, options(workers)).unwrap_err();
            match behavior {
                Behavior::Dimension => assert!(matches!(
                    error,
                    BatchDecodeError::Dimension {
                        batch_detectors: WIDTH,
                        decoder_detectors: 13
                    }
                )),
                Behavior::Missing => {
                    assert!(matches!(error, BatchDecodeError::MissingDetectorDimension));
                }
                _ => unreachable!(),
            }
            assert_eq!(factory.calls.load(Ordering::SeqCst), 0);
            assert_eq!(factory.native_calls.load(Ordering::SeqCst), 0);
            assert_eq!(factory.builds.load(Ordering::SeqCst), workers.unwrap_or(1));
        }
    }
}
#[test]
fn empty_batches_build_preflight_and_retain_only_when_requested() {
    for workers in [None, Some(1), Some(4)] {
        for retain in [false, true] {
            let factory = Factory {
                native: workers.is_none(),
                ..Factory::default()
            };
            let result = run(
                0,
                &factory,
                options(workers).predictions(retain).timing(retain),
            )
            .unwrap();
            assert_eq!(factory.builds.load(Ordering::SeqCst), 1);
            assert_eq!(result.workers_used, 1);
            assert_eq!(result.num_errors, 0);
            assert_eq!(result.num_shots, 0);
            assert!(result.logical_error_rate().abs() < f64::EPSILON);
            assert_eq!(result.predictions, retain.then(Vec::new));
            assert_eq!(result.per_shot_seconds, retain.then(Vec::new));
            assert_eq!(factory.calls.load(Ordering::SeqCst), 0);
            assert_eq!(factory.native_calls.load(Ordering::SeqCst), 0);
            let bad = Factory {
                native: workers.is_none(),
                ..Factory::with(Behavior::Dimension)
            };
            assert!(matches!(
                run(0, &bad, options(workers)),
                Err(BatchDecodeError::Dimension { .. })
            ));
        }
    }
}
#[test]
fn parallel_workers_are_clamped_to_shots() {
    let factory = Factory::default();
    let result = run(2, &factory, DecodeOptions::default().workers(9)).unwrap();
    assert_eq!(result.execution_path, ExecutionPath::Parallel);
    assert_eq!(result.workers_used, 2);
    assert_eq!(factory.builds.load(Ordering::SeqCst), 2);
}
#[test]
fn native_failure_builds_a_fresh_diagnostic_decoder() {
    let factory = Factory {
        native: true,
        ..Factory::with(Behavior::NativeFailure)
    };
    let error = batch(1100).decode("test DEM", &factory).unwrap_err();
    assert!(matches!(error, BatchDecodeError::Decode(ref e) if e.shot_index == 17));
    assert_eq!(factory.builds.load(Ordering::SeqCst), 2);
    assert_eq!(factory.calls.load(Ordering::SeqCst), 18);
    assert_eq!(factory.native_calls.load(Ordering::SeqCst), 1);
}
#[test]
fn unreproducible_native_failure_is_runtime_error() {
    let factory = Factory {
        native: true,
        ..Factory::with(Behavior::Unreproducible)
    };
    let error = batch(5).decode("test DEM", &factory).unwrap_err();
    assert!(
        matches!(error, BatchDecodeError::Runtime(ref message) if message == "native batch decode failed over shots 0..5 (no single shot reproduces the failure): Decoding failed: native failure")
    );
    assert_eq!(factory.builds.load(Ordering::SeqCst), 2);
    assert_eq!(factory.calls.load(Ordering::SeqCst), 5);
}
#[test]
fn wrong_native_prediction_count_is_indexed() {
    let factory = Factory {
        native: true,
        ..Factory::with(Behavior::WrongCount)
    };
    let error = batch(5).decode("test DEM", &factory).unwrap_err();
    assert!(
        matches!(error, BatchDecodeError::Decode(ref e) if e.shot_index == 0 && e.source.to_string() == "Decoding failed: native batch decoder returned 0 predictions for 5 shots")
    );
    assert_eq!(factory.builds.load(Ordering::SeqCst), 1);
    assert_eq!(factory.calls.load(Ordering::SeqCst), 0);
}
#[test]
fn parallel_error_precedence_is_dimension_build_runtime_decode() {
    for behaviors in [
        vec![
            Behavior::FailShots,
            Behavior::Missing,
            Behavior::Build,
            Behavior::Dimension,
        ],
        vec![Behavior::FailShots, Behavior::Missing, Behavior::Build],
        vec![Behavior::FailShots, Behavior::Missing],
    ] {
        let workers = behaviors.len();
        let factory = Factory {
            behaviors,
            ..Factory::default()
        };
        let error = run(1301, &factory, DecodeOptions::default().workers(workers)).unwrap_err();
        match workers {
            4 => assert!(matches!(error, BatchDecodeError::Dimension { .. })),
            3 => assert!(
                matches!(error, BatchDecodeError::Build(DecoderError::InvalidConfiguration(ref s)) if s == "original build error")
            ),
            2 => assert!(matches!(error, BatchDecodeError::MissingDetectorDimension)),
            _ => unreachable!(),
        }
        assert_eq!(factory.builds.load(Ordering::SeqCst), workers);
    }
}
#[test]
fn timings_force_per_shot_dispatch_and_retention() {
    for workers in [1, 3] {
        let factory = Factory {
            native: true,
            ..Factory::default()
        };
        let result = run(
            90,
            &factory,
            DecodeOptions::default()
                .workers(workers)
                .predictions(true)
                .timing(true),
        )
        .unwrap();
        let times = result.per_shot_seconds.unwrap();
        assert_eq!(times.len(), 90);
        assert!(times.iter().all(|&t| t >= 0.0 && t <= result.wall_elapsed));
        assert_eq!(result.predictions.unwrap().len(), 90);
        assert_eq!(factory.native_calls.load(Ordering::SeqCst), 0);
        assert_eq!(factory.calls.load(Ordering::SeqCst), 90);
    }
    let factory = Factory {
        native: true,
        ..Factory::default()
    };
    let result = run(90, &factory, DecodeOptions::default().timing(true)).unwrap();
    assert_eq!(result.execution_path, ExecutionPath::Sequential);
    assert_eq!(result.per_shot_seconds.unwrap().len(), 90);
    assert!(result.predictions.is_none());
}
#[test]
fn constructors_validate_before_transposing_and_round_trip_wide_masks() {
    for (det, obs, shots) in [
        (vec![vec![]], vec![], 1),
        (vec![], vec![vec![0, 0]], 1),
        (vec![vec![0]], vec![], 0),
    ] {
        assert!(SampleBatch::from_columnar(det, obs, shots).is_err());
    }
    assert!(SampleBatch::from_row_major(&[vec![1u8]], &[], 1).is_err());
    assert!(SampleBatch::from_row_major::<Vec<u8>>(&[], &[ObsMask::new()], 0).is_err());
    assert!(
        SampleBatch::from_row_major(
            &[vec![1u8], vec![1, 1]],
            &[ObsMask::new(), ObsMask::new()],
            0
        )
        .is_err()
    );
    assert!(SampleBatch::from_row_major(&[vec![1u8]], &[prediction(64)], 64).is_err());
    let rows: Vec<_> = (0..130).map(|i| vec![u8::from(i % 2 == 0), 255]).collect();
    let masks: Vec<_> = (0..130).map(prediction).collect();
    let batch = SampleBatch::from_row_major(&rows, &masks, 130).unwrap();
    assert_eq!(batch.num_shots(), 130);
    assert_eq!(batch.num_detectors(), 2);
    assert_eq!(batch.num_observables(), 130);
    assert_eq!(
        batch.det_columns()[0],
        vec![0x5555_5555_5555_5555, 0x5555_5555_5555_5555, 1]
    );
    assert_eq!(batch.obs_columns()[64], vec![0, 1, 0]);
    for (i, shot) in batch.shots().enumerate() {
        assert_eq!(shot.syndrome, vec![rows[i][0], 1]);
        assert_eq!(shot.observable_flips, masks[i]);
        let mut buffer = vec![9; 4];
        batch.syndrome_into(i, &mut buffer);
        assert_eq!(buffer, vec![rows[i][0], 1, 0, 0]);
    }
    assert_eq!(
        batch,
        SampleBatch::from_columnar(
            batch.det_columns().to_vec(),
            batch.obs_columns().to_vec(),
            130
        )
        .unwrap()
    );
}
#[test]
fn planning_errors_and_warnings_are_preserved() {
    let factory = Factory::default();
    assert!(matches!(
        run(1, &factory, DecodeOptions::default().workers(0)),
        Err(BatchDecodeError::Plan(_))
    ));
    assert_eq!(factory.builds.load(Ordering::SeqCst), 0);
    let factory = Factory {
        traits: ExecutionTraits {
            history_dependent: true,
            wall_clock_dependent: false,
        },
        ..Factory::default()
    };
    assert!(matches!(
        run(1, &factory, DecodeOptions::default().workers(2)),
        Err(BatchDecodeError::Plan(_))
    ));
    let factory = Factory {
        traits: ExecutionTraits {
            history_dependent: false,
            wall_clock_dependent: true,
        },
        ..Factory::default()
    };
    let result = run(2, &factory, DecodeOptions::default().workers(2)).unwrap();
    assert_eq!(
        result.reproducibility_warnings,
        vec![
            "parallel wall-clock-limited decoding may not be reproducible because CPU contention can change which shots reach the solver time limit"
        ]
    );
}

#[test]
fn diagnostic_decoder_is_also_preflighted() {
    for behavior in [Behavior::Dimension, Behavior::Missing, Behavior::Build] {
        let factory = Factory {
            native: true,
            behaviors: vec![Behavior::NativeFailure, behavior],
            ..Factory::default()
        };
        let error = batch(20).decode("test DEM", &factory).unwrap_err();
        match behavior {
            Behavior::Dimension => assert!(matches!(error, BatchDecodeError::Dimension { .. })),
            Behavior::Missing => {
                assert!(matches!(error, BatchDecodeError::MissingDetectorDimension));
            }
            Behavior::Build => assert!(matches!(error, BatchDecodeError::Build(_))),
            _ => unreachable!(),
        }
        assert_eq!(factory.builds.load(Ordering::SeqCst), 2);
        assert_eq!(factory.calls.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn builtin_factory_selects_embedded_hybrid_model() {
    use pecos_decoders::DecoderSpec;
    use pecos_decoders::spec::{BeliefMatchingConfig, BeliefMatchingMode};
    assert_eq!(
        DecoderSpec::AStar.decode_model("single"),
        DecodeModel::SingleDem("single".into())
    );
    let spec = DecoderSpec::BeliefMatching(BeliefMatchingConfig {
        mode: BeliefMatchingMode::Hybrid,
        embedded_full_dem: Some("full".into()),
    });
    assert_eq!(
        spec.decode_model("decomposed"),
        DecodeModel::HybridDem {
            full: "full".into(),
            decomposed: "decomposed".into()
        }
    );
}

fn assert_shot_panic(panic: &(dyn std::any::Any + Send), shot: usize, count: usize) {
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied());
    assert_eq!(
        message,
        Some(format!("shot index {shot} out of range (num_shots={count})").as_str())
    );
}

#[test]
fn syndrome_rejects_every_out_of_range_shot() {
    for batch in [
        batch(3),
        SampleBatch::from_columnar(vec![], vec![], 3).unwrap(),
        batch(0),
    ] {
        for shot in [batch.num_shots(), 10, 64, usize::MAX] {
            let panic = std::panic::catch_unwind(|| {
                let mut syndrome = vec![0; batch.num_detectors()];
                batch.syndrome_into(shot, &mut syndrome);
            })
            .expect_err("invalid shot must panic, even inside a padding word or without columns");
            assert_shot_panic(panic.as_ref(), shot, batch.num_shots());
        }
    }
}

#[test]
fn row_constructor_accepts_borrowed_row_slices() {
    // Callers holding borrowed rows must not have to allocate owned vectors.
    let rows: [&[u8]; 2] = [&[1, 0], &[0, 3]];
    let batch = SampleBatch::from_row_major(&rows, &[ObsMask::new(), ObsMask::new()], 0).unwrap();
    let syndromes: Vec<_> = batch.shots().map(|shot| shot.syndrome).collect();
    assert_eq!(syndromes, vec![vec![1, 0], vec![0, 1]]);
}

#[test]
fn syndrome_rejects_a_short_buffer_even_without_set_detectors() {
    // All detectors are clear, so a missing length check would index nothing
    // out of bounds and silently return a truncated syndrome.
    let batch = SampleBatch::from_row_major(&[vec![0u8; 3]], &[ObsMask::new()], 0).unwrap();
    let panic = std::panic::catch_unwind(|| batch.syndrome_into(0, &mut [0u8; 1]))
        .expect_err("a buffer shorter than the detector count must panic");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied());
    assert_eq!(
        message,
        Some("syndrome buffer length 1 is smaller than the detector count 3")
    );
}

#[test]
fn observables_reject_every_out_of_range_shot() {
    for batch in [
        batch(3),
        SampleBatch::from_columnar(vec![], vec![], 3).unwrap(),
        batch(0),
    ] {
        for shot in [batch.num_shots(), 10, 64, usize::MAX] {
            let panic = std::panic::catch_unwind(|| batch.observable_flips(shot)).expect_err(
                "invalid shot must panic, even inside a padding word or without columns",
            );
            assert_shot_panic(panic.as_ref(), shot, batch.num_shots());
        }
    }
}

#[test]
fn shots_iterator_excludes_padding() {
    for count in [0, 3, 64, 65] {
        let batch = batch(count);
        let mut shots = batch.shots();
        assert_eq!(shots.len(), count);
        for i in 0..count {
            let shot = shots.next().unwrap();
            assert_eq!(index(&shot.syndrome), i);
            assert_eq!(shot.observable_flips, batch.observable_flips(i));
        }
        assert_eq!(shots.next(), None);
        assert_eq!(batch.shots().nth(count), None);
    }
}

#[test]
fn packed_columns_reject_nonzero_padding() {
    for shots in [1usize, 3, 63, 65, 127] {
        let mut bad = vec![0; shots.div_ceil(64)];
        *bad.last_mut().unwrap() = 1 << (shots % 64);
        for (kind, detectors, observables) in [
            ("detector", vec![vec![0; bad.len()], bad.clone()], vec![]),
            ("observable", vec![], vec![vec![0; bad.len()], bad.clone()]),
        ] {
            assert_eq!(
                SampleBatch::from_columnar(detectors, observables, shots)
                    .unwrap_err()
                    .to_string(),
                format!("{kind} column 1 has nonzero padding bits above num_shots={shots}")
            );
        }
        // All valid bits may be set, including bit 63 in preceding full words.
        let mut good = vec![u64::MAX; shots.div_ceil(64)];
        *good.last_mut().unwrap() = (1 << (shots % 64)) - 1;
        assert!(SampleBatch::from_columnar(vec![good.clone()], vec![good], shots).is_ok());
    }
    for shots in [0usize, 64, 128] {
        assert!(
            SampleBatch::from_columnar(vec![vec![u64::MAX; shots / 64]], vec![], shots).is_ok()
        );
    }
}

#[test]
fn batch_errors_preserve_source_chains() {
    use pecos_decoders::batch::{ExecutionPlanError, ShotDecodeError};
    use std::error::Error;

    #[derive(Debug)]
    struct BuildFailure(DecoderError);
    impl std::fmt::Display for BuildFailure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("build context")
        }
    }
    impl Error for BuildFailure {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            Some(&self.0)
        }
    }

    let build = BatchDecodeError::Build(BuildFailure(DecoderError::InternalError(
        "build cause".into(),
    )));
    assert_eq!(build.to_string(), "build context");
    assert!(
        matches!(build.source().unwrap().downcast_ref::<DecoderError>(),
        Some(DecoderError::InternalError(message)) if message == "build cause")
    );
    let leaf_build = BatchDecodeError::Build(DecoderError::InternalError("leaf build".into()));
    assert!(leaf_build.source().is_none());

    let decode: BatchDecodeError<DecoderError> = BatchDecodeError::Decode(ShotDecodeError::new(
        17,
        DecoderError::DecodingFailed("decode cause".into()),
    ));
    assert_eq!(
        decode.to_string(),
        "decoder failed on shot 17: Decoding failed: decode cause"
    );
    assert!(
        matches!(decode.source().unwrap().downcast_ref::<DecoderError>(),
        Some(DecoderError::DecodingFailed(message)) if message == "decode cause")
    );
    // ShotDecodeError adds context, so it retains its immediate decoder cause.
    let BatchDecodeError::Decode(shot) = decode else {
        unreachable!()
    };
    assert!(
        matches!(shot.source().unwrap().downcast_ref::<DecoderError>(),
        Some(DecoderError::DecodingFailed(message)) if message == "decode cause")
    );

    let plan: BatchDecodeError<DecoderError> =
        BatchDecodeError::Plan(ExecutionPlanError::InvalidWorkerCount);
    assert_eq!(plan.to_string(), "worker count must be at least 1");
    assert!(plan.source().is_none());
    for error in [
        BatchDecodeError::<DecoderError>::MissingDetectorDimension,
        BatchDecodeError::Dimension {
            batch_detectors: 1,
            decoder_detectors: 2,
        },
        BatchDecodeError::Runtime("runtime".into()),
    ] {
        assert!(error.source().is_none());
    }
}
