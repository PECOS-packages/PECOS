//! Deterministic planning and canonical-sampling primitives for batch decoding.

use std::fmt;
use std::ops::Range;

use pecos_random::{PecosRng, nth_derived_seed};

/// Frozen version of the worker-count-independent fused-sampling protocol.
pub const SAMPLING_ABI_VERSION: u32 = 1;

/// Number of shots assigned to each reproducible sampling RNG stream.
///
/// ABI v1 always performs exactly one `DemSampler::sample` call per shot;
/// geometric and bulk samplers must not be substituted for these chunks.
/// A final partial chunk consumes only its stream prefix, so the stream for
/// `N` shots is a prefix of the stream for any larger shot count at one seed.
pub const SAMPLING_CHUNK_SHOTS: usize = 1024;

/// `SplitMix` derivation-domain tag for fused DEM sampling (ASCII `PECOSDEM`).
///
/// Applying XOR with this tag guarantees distinct derivation inputs at equal
/// indices and pseudorandom decorrelation across domains. It does not prove global
/// cross-domain non-collision: different domain offsets share one `SplitMix`
/// orbit. Within one run, derived chunk seeds are collision-free because the
/// fixed-base derivation is bijective in the chunk index.
pub const DEM_SAMPLING_DOMAIN_TAG: u64 = 0x5045_434F_5344_454D;

/// Derive the canonical ABI-v1 RNG seed for one fused-sampling chunk.
///
/// A chunk uses `PecosRng::seed_from_u64` on this value. `PecosRng` expands it
/// into four `ParallelRapidRng` streams through `SplitMix`; the resulting streams
/// are deterministic, unique per chunk index, and pseudorandomly decorrelated,
/// but this is not a formal proof that their generated sequences never overlap.
#[must_use]
pub fn sampling_chunk_seed(seed: u64, chunk_index: u64) -> u64 {
    nth_derived_seed(seed ^ DEM_SAMPLING_DOMAIN_TAG, chunk_index)
}

/// Below this many shots, setting up worker threads dominates generic decoding.
pub const SMALL_BATCH_SEQUENTIAL_THRESHOLD: usize = 1000;

/// Auto-planning gives each generic decoder worker at least this many shots.
pub const MIN_SHOTS_PER_WORKER: usize = 256;

/// Shots a generic parallel worker claims per trip to the shared cursor.
///
/// Search-based decoders vary by orders of magnitude from shot to shot, so
/// workers pull small chunks dynamically rather than splitting the batch into
/// one fixed slice each. Small enough to balance a heavy tail, large enough
/// that the atomic fetch is not the bottleneck.
pub const PARALLEL_CHUNK_SHOTS: usize = 64;

/// Effective chunk size for one parallel run.
///
/// Caps [`PARALLEL_CHUNK_SHOTS`] so a batch smaller than
/// `workers * PARALLEL_CHUNK_SHOTS` still spreads across every worker an
/// explicit worker count asked for, instead of leaving the late workers with
/// no chunk to claim. Always at least 1 so `div_ceil` is safe on empty
/// batches.
#[must_use]
pub fn parallel_chunk_shots(num_shots: usize, workers: usize) -> usize {
    PARALLEL_CHUNK_SHOTS
        .min(num_shots.div_ceil(workers.max(1)))
        .max(1)
}

/// Native decoders receive at most this many transposed shots at once.
pub const NATIVE_SUB_BATCH_SHOTS: usize = 1024;

const MAX_INTERVAL_SHOTS: usize = 100_000_000;
const MIN_INTERVAL_ALPHA: f64 = 1e-6;
const MAX_INTERVAL_ALPHA: f64 = 0.5;

/// An error paired with the absolute shot index that produced it.
#[derive(Debug)]
pub struct IndexedDecodeError<E> {
    pub shot_index: usize,
    pub error: E,
}

/// Select the lowest indexed failure, independent of worker completion order.
pub fn lowest_indexed_error<E>(
    errors: impl IntoIterator<Item = IndexedDecodeError<E>>,
) -> Option<IndexedDecodeError<E>> {
    errors.into_iter().min_by_key(|error| error.shot_index)
}

/// Inputs to the deterministic batch execution planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionPlanInputs {
    pub traits: crate::ExecutionTraits,
    pub num_shots: usize,
    pub native_batch_capable: bool,
    pub timing: bool,
    pub explicit_workers: Option<usize>,
    pub available_threads: usize,
}

/// Concrete execution mechanism selected by the planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionPath {
    Sequential,
    Parallel,
    NativeBatch,
}

impl ExecutionPath {
    /// Stable Python-facing spelling for this path.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Parallel => "parallel",
            Self::NativeBatch => "native_batch",
        }
    }
}

/// Fully resolved batch execution plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionPlan {
    pub path: ExecutionPath,
    pub workers_used: usize,
    pub reproducibility_warnings: Vec<String>,
}

impl ExecutionPlan {
    fn sequential() -> Self {
        Self {
            path: ExecutionPath::Sequential,
            workers_used: 1,
            reproducibility_warnings: Vec::new(),
        }
    }

    fn parallel(workers_used: usize, reproducibility_warnings: Vec<String>) -> Self {
        Self {
            path: ExecutionPath::Parallel,
            workers_used,
            reproducibility_warnings,
        }
    }

    fn native_batch() -> Self {
        Self {
            path: ExecutionPath::NativeBatch,
            workers_used: 1,
            reproducibility_warnings: Vec::new(),
        }
    }
}

/// Invalid combinations rejected by [`plan_execution`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionPlanError {
    InvalidWorkerCount,
    HistoryDependentParallel { workers: usize },
}

impl fmt::Display for ExecutionPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWorkerCount => formatter.write_str("worker count must be at least 1"),
            Self::HistoryDependentParallel { workers } => write!(
                formatter,
                "worker count {workers} is invalid for a history-dependent decoder"
            ),
        }
    }
}

impl std::error::Error for ExecutionPlanError {}

/// Validate the numerically verified domain of the decode-result interval.
///
/// # Errors
///
/// Returns a message when there are no shots, the shot count exceeds the
/// verified limit, or `alpha` is non-finite or outside `[1e-6, 0.5]`.
pub fn validate_decode_interval(num_shots: usize, alpha: f64) -> Result<(), String> {
    if !alpha.is_finite() || !(MIN_INTERVAL_ALPHA..=MAX_INTERVAL_ALPHA).contains(&alpha) {
        return Err(format!(
            "alpha must be finite and in [{MIN_INTERVAL_ALPHA}, {MAX_INTERVAL_ALPHA}]"
        ));
    }
    if num_shots == 0 {
        return Err("num_shots must be greater than zero".to_string());
    }
    if num_shots > MAX_INTERVAL_SHOTS {
        return Err(format!(
            "num_shots must be at most {MAX_INTERVAL_SHOTS}; got {num_shots}"
        ));
    }
    Ok(())
}

/// Select a batch execution path without reading process or thread-pool globals.
///
/// # Errors
///
/// Returns [`ExecutionPlanError`] for a zero explicit worker count or explicit
/// parallelism of a history-dependent decoder.
pub fn plan_execution(inputs: ExecutionPlanInputs) -> Result<ExecutionPlan, ExecutionPlanError> {
    if let Some(workers) = inputs.explicit_workers {
        if workers == 0 {
            return Err(ExecutionPlanError::InvalidWorkerCount);
        }
        if workers == 1 {
            return Ok(ExecutionPlan::sequential());
        }
        if inputs.traits.history_dependent {
            return Err(ExecutionPlanError::HistoryDependentParallel { workers });
        }
        let warnings = if inputs.traits.wall_clock_dependent {
            vec![
                "parallel wall-clock-limited decoding may not be reproducible because CPU \
                 contention can change which shots reach the solver time limit"
                    .to_string(),
            ]
        } else {
            Vec::new()
        };
        return Ok(ExecutionPlan::parallel(workers, warnings));
    }

    if inputs.traits.history_dependent || inputs.traits.wall_clock_dependent {
        return Ok(ExecutionPlan::sequential());
    }
    if inputs.native_batch_capable && !inputs.timing {
        return Ok(ExecutionPlan::native_batch());
    }
    if inputs.num_shots < SMALL_BATCH_SEQUENTIAL_THRESHOLD {
        return Ok(ExecutionPlan::sequential());
    }

    let workers = inputs
        .available_threads
        .max(1)
        .min(inputs.num_shots.div_ceil(MIN_SHOTS_PER_WORKER));
    Ok(ExecutionPlan::parallel(workers, Vec::new()))
}

/// Iterator over bounded native-batch shot ranges.
#[derive(Clone, Debug)]
pub struct NativeSubBatches {
    next: usize,
    num_shots: usize,
}

impl Iterator for NativeSubBatches {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.num_shots {
            return None;
        }
        let start = self.next;
        let end = start
            .saturating_add(NATIVE_SUB_BATCH_SHOTS)
            .min(self.num_shots);
        self.next = end;
        Some(start..end)
    }
}

/// Split `num_shots` into native sub-batches of at most
/// [`NATIVE_SUB_BATCH_SHOTS`].
#[must_use]
pub const fn native_sub_batches(num_shots: usize) -> NativeSubBatches {
    NativeSubBatches { next: 0, num_shots }
}

/// Number of bytes required by the reusable native transpose scratch buffer.
///
/// At most one native sub-batch is allocated, while a smaller batch allocates
/// only the rows it can use. Returns `None` if the dimensions overflow.
#[must_use]
pub fn native_scratch_len(num_shots: usize, num_detectors: usize) -> Option<usize> {
    NATIVE_SUB_BATCH_SHOTS
        .min(num_shots)
        .checked_mul(num_detectors)
}

/// Iterator over the frozen ABI-v1 fused-sampling chunk partition.
#[derive(Clone, Debug)]
pub struct SamplingChunks {
    next: usize,
    num_shots: usize,
}

impl Iterator for SamplingChunks {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.num_shots {
            return None;
        }
        let start = self.next;
        let end = start
            .saturating_add(SAMPLING_CHUNK_SHOTS)
            .min(self.num_shots);
        self.next = end;
        Some(start..end)
    }
}

/// Partition shots as `[0..1024), [1024..2048), ...` for sampling ABI v1.
#[must_use]
pub const fn sampling_chunks(num_shots: usize) -> SamplingChunks {
    SamplingChunks { next: 0, num_shots }
}

/// Upper bound on useful workers for fused sampling.
///
/// A sampling chunk is the unit of parallel work, so threads beyond one per
/// chunk can never be given anything to do. Both the thread pool and the
/// reported `workers_used` derive from this, so the number a caller sees is
/// the number that could actually run.
#[must_use]
pub fn fused_worker_cap(num_shots: usize) -> usize {
    num_shots.div_ceil(SAMPLING_CHUNK_SHOTS).max(1)
}

/// The single seam constructing the canonical per-chunk RNG for sampling ABI v1.
///
/// Every execution path — sequential, parallel, native — must obtain its chunk
/// RNG here so the seed derivation and the index-width policy can never
/// diverge between paths.
///
/// # Panics
///
/// Panics if `chunk_index` does not fit `u64`, which cannot happen on any
/// supported target (pointer width is at most 64 bits).
#[must_use]
pub fn sampling_chunk_rng(seed: u64, chunk_index: usize) -> PecosRng {
    let chunk_index =
        u64::try_from(chunk_index).expect("chunk index fits u64 on all supported targets");
    PecosRng::seed_from_u64(sampling_chunk_seed(seed, chunk_index))
}

/// Run a sequential fused loop with one fresh canonical RNG per sampling chunk.
///
/// The callback is invoked strictly in absolute shot order. State captured by
/// it—most importantly a history-dependent decoder—is preserved across chunk
/// boundaries. The callback must consume exactly one `DemSampler::sample` call
/// per invocation to conform to sampling ABI v1.
///
/// # Errors
///
/// Returns the first callback error.
pub fn for_each_canonical_sample<E>(
    num_shots: usize,
    seed: u64,
    mut sample_shot: impl FnMut(usize, &mut PecosRng) -> Result<(), E>,
) -> Result<(), E> {
    for (chunk_index, range) in sampling_chunks(num_shots).enumerate() {
        let mut rng = sampling_chunk_rng(seed, chunk_index);
        for shot_index in range {
            sample_shot(shot_index, &mut rng)?;
        }
    }
    Ok(())
}

/// One chunk result tagged independently of scheduler completion order.
#[derive(Debug, Eq, PartialEq)]
pub struct IndexedChunk<T> {
    pub chunk_index: usize,
    pub value: T,
}

/// A missing, duplicated, or out-of-range chunk index during final assembly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkAssemblyError {
    pub expected_index: usize,
    pub actual_index: Option<usize>,
}

impl fmt::Display for ChunkAssemblyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "sampling chunk assembly expected index {}, got {:?}",
            self.expected_index, self.actual_index
        )
    }
}

impl std::error::Error for ChunkAssemblyError {}

/// Sort scheduler results into canonical chunk order and verify `0..N` exactly.
///
/// # Errors
///
/// Returns [`ChunkAssemblyError`] if an index is missing, duplicated, or extra.
pub fn assemble_indexed_chunks<T>(
    mut chunks: Vec<IndexedChunk<T>>,
    expected_chunks: usize,
) -> Result<Vec<T>, ChunkAssemblyError> {
    chunks.sort_unstable_by_key(|chunk| chunk.chunk_index);
    for expected_index in 0..expected_chunks {
        let actual_index = chunks.get(expected_index).map(|chunk| chunk.chunk_index);
        if actual_index != Some(expected_index) {
            return Err(ChunkAssemblyError {
                expected_index,
                actual_index,
            });
        }
    }
    if chunks.len() != expected_chunks {
        return Err(ChunkAssemblyError {
            expected_index: expected_chunks,
            actual_index: chunks.get(expected_chunks).map(|chunk| chunk.chunk_index),
        });
    }
    Ok(chunks.into_iter().map(|chunk| chunk.value).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_seed_pins_are_frozen_for_reproducibility_abi_v1() {
        // Reproducibility ABI v1 — never update these literals without a
        // deliberate version bump and release note.
        assert_eq!(SAMPLING_ABI_VERSION, 1);
        assert_eq!(sampling_chunk_seed(0, 0), 0x20D0_F923_7F7F_D397);
        assert_eq!(sampling_chunk_seed(0, 1), 0x4D24_F3DD_891B_E6D1);
        assert_eq!(sampling_chunk_seed(42, 7), 0x4C26_E1A7_0BF0_C0AE);
    }

    #[test]
    fn sampling_chunk_partition_is_frozen() {
        assert!(sampling_chunks(0).next().is_none());
        for (num_shots, expected) in [(1, 0..1), (1023, 0..1023), (1024, 0..1024)] {
            let mut chunks = sampling_chunks(num_shots);
            assert_eq!(chunks.next(), Some(expected));
            assert_eq!(chunks.next(), None);
        }
        assert_eq!(
            sampling_chunks(1025).collect::<Vec<_>>(),
            [0..1024, 1024..1025]
        );
        assert_eq!(
            sampling_chunks(2065).collect::<Vec<_>>(),
            [0..1024, 1024..2048, 2048..2065]
        );
    }

    #[test]
    fn assembly_uses_chunk_indices_not_completion_order() {
        use std::sync::mpsc;

        let (completed_tx, completed_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let slow_tx = completed_tx.clone();
            scope.spawn(move || {
                release_rx.recv().unwrap();
                slow_tx
                    .send(IndexedChunk {
                        chunk_index: 0,
                        value: vec![0, 1],
                    })
                    .unwrap();
            });
            let fast_tx = completed_tx.clone();
            scope.spawn(move || {
                fast_tx
                    .send(IndexedChunk {
                        chunk_index: 1,
                        value: vec![1024, 1025],
                    })
                    .unwrap();
                release_tx.send(()).unwrap();
            });
        });
        drop(completed_tx);

        let completion_order = completed_rx.into_iter().collect::<Vec<_>>();
        assert_eq!(completion_order[0].chunk_index, 1);
        assert_eq!(
            assemble_indexed_chunks(completion_order, 2).unwrap(),
            vec![vec![0, 1], vec![1024, 1025]]
        );
    }

    #[test]
    fn history_dependent_state_crosses_sampling_chunk_boundary() {
        #[derive(Default)]
        struct StatefulDecoder {
            calls: u64,
        }

        impl StatefulDecoder {
            fn decode(&mut self, sample: u64) -> u64 {
                self.calls += 1;
                sample ^ self.calls
            }
        }

        const SHOTS: usize = SAMPLING_CHUNK_SHOTS + 17;
        const SEED: u64 = 91;
        let mut fused_decoder = StatefulDecoder::default();
        let mut fused = Vec::with_capacity(SHOTS);
        for_each_canonical_sample(SHOTS, SEED, |_, rng| {
            fused.push(fused_decoder.decode(rng.next_u64()));
            Ok::<_, std::convert::Infallible>(())
        })
        .unwrap();

        let mut manual_decoder = StatefulDecoder::default();
        let mut manual = Vec::with_capacity(SHOTS);
        for (chunk_index, range) in sampling_chunks(SHOTS).enumerate() {
            let mut rng = PecosRng::seed_from_u64(sampling_chunk_seed(
                SEED,
                u64::try_from(chunk_index).unwrap(),
            ));
            for _ in range {
                manual.push(manual_decoder.decode(rng.next_u64()));
            }
        }

        assert_eq!(fused, manual);
        assert_eq!(fused_decoder.calls, u64::try_from(SHOTS).unwrap());
        assert_eq!(manual_decoder.calls, u64::try_from(SHOTS).unwrap());
    }

    fn inputs() -> ExecutionPlanInputs {
        ExecutionPlanInputs {
            traits: crate::ExecutionTraits::default(),
            num_shots: 10_000,
            native_batch_capable: false,
            timing: false,
            explicit_workers: None,
            available_threads: 8,
        }
    }

    #[test]
    fn explicit_worker_rules_cover_all_traits() {
        for history_dependent in [false, true] {
            for wall_clock_dependent in [false, true] {
                let mut case = inputs();
                case.traits.history_dependent = history_dependent;
                case.traits.wall_clock_dependent = wall_clock_dependent;

                case.explicit_workers = Some(1);
                assert_eq!(plan_execution(case).unwrap(), ExecutionPlan::sequential());

                case.explicit_workers = Some(4);
                if history_dependent {
                    assert_eq!(
                        plan_execution(case),
                        Err(ExecutionPlanError::HistoryDependentParallel { workers: 4 })
                    );
                } else {
                    let plan = plan_execution(case).unwrap();
                    assert_eq!(plan.path, ExecutionPath::Parallel);
                    assert_eq!(plan.workers_used, 4);
                    assert_eq!(
                        plan.reproducibility_warnings.is_empty(),
                        !wall_clock_dependent
                    );
                }
            }
        }

        let mut case = inputs();
        case.explicit_workers = Some(0);
        assert_eq!(
            plan_execution(case),
            Err(ExecutionPlanError::InvalidWorkerCount)
        );
    }

    #[test]
    fn auto_trait_precedence_is_sequential() {
        for (history_dependent, wall_clock_dependent) in
            [(true, false), (false, true), (true, true)]
        {
            let mut case = inputs();
            case.traits.history_dependent = history_dependent;
            case.traits.wall_clock_dependent = wall_clock_dependent;
            case.native_batch_capable = true;
            assert_eq!(plan_execution(case).unwrap(), ExecutionPlan::sequential());
        }
    }

    #[test]
    fn auto_native_precedes_small_batch_and_timing_disables_native() {
        let mut case = inputs();
        case.num_shots = 1;
        case.native_batch_capable = true;
        assert_eq!(plan_execution(case).unwrap(), ExecutionPlan::native_batch());

        case.timing = true;
        assert_eq!(plan_execution(case).unwrap(), ExecutionPlan::sequential());
    }

    #[test]
    fn auto_generic_threshold_and_worker_cap() {
        let mut case = inputs();
        case.num_shots = SMALL_BATCH_SEQUENTIAL_THRESHOLD - 1;
        assert_eq!(plan_execution(case).unwrap(), ExecutionPlan::sequential());

        case.num_shots = SMALL_BATCH_SEQUENTIAL_THRESHOLD;
        let plan = plan_execution(case).unwrap();
        assert_eq!(plan.path, ExecutionPath::Parallel);
        assert_eq!(plan.workers_used, 4);

        case.num_shots = 10_000;
        case.available_threads = 3;
        assert_eq!(plan_execution(case).unwrap().workers_used, 3);
    }

    #[test]
    fn native_sub_batches_bound_reusable_scratch() {
        const NUM_DETECTORS: usize = 7;
        const NUM_SHOTS: usize = NATIVE_SUB_BATCH_SHOTS * 2 + 17;
        let mut scratch = vec![0u8; native_scratch_len(NUM_SHOTS, NUM_DETECTORS).unwrap()];
        let initial_capacity = scratch.capacity();
        let ranges = native_sub_batches(NUM_SHOTS).collect::<Vec<_>>();
        for range in &ranges {
            let used = range.len() * NUM_DETECTORS;
            scratch[..used].fill(1);
            assert!(used <= NATIVE_SUB_BATCH_SHOTS * NUM_DETECTORS);
            assert_eq!(scratch.capacity(), initial_capacity);
        }
        assert_eq!(ranges, vec![0..1024, 1024..2048, 2048..2065]);

        assert_eq!(native_scratch_len(17, NUM_DETECTORS), Some(119));
        assert_eq!(native_scratch_len(0, NUM_DETECTORS), Some(0));
    }

    #[test]
    fn parallel_chunk_shots_spreads_small_batches_across_all_workers() {
        // Big batch: the fixed chunk size stands.
        assert_eq!(parallel_chunk_shots(100_000, 8), PARALLEL_CHUNK_SHOTS);
        // Small batch with many workers: shrink chunks so every worker can
        // claim at least one (500 shots / 32 workers -> 16-shot chunks).
        assert_eq!(parallel_chunk_shots(500, 32), 16);
        // Degenerate inputs stay safe for div_ceil.
        assert_eq!(parallel_chunk_shots(0, 4), 1);
        assert_eq!(parallel_chunk_shots(5, 0), 5);
    }

    #[test]
    fn parallel_fake_decoder_reports_lowest_absolute_failure() {
        use pecos_decoder_core::obs_mask::ObsMask;
        use pecos_decoder_core::{DecoderError, ObservableDecoder};

        struct FakeDecoder;

        impl ObservableDecoder for FakeDecoder {
            fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
                match syndrome[0] {
                    2 | 9 => Err(DecoderError::DecodingFailed("selected failure".into())),
                    _ => Ok(ObsMask::new()),
                }
            }
        }

        let worker_errors = std::thread::scope(|scope| {
            let handles = (0..4usize)
                .map(|worker| {
                    scope.spawn(move || {
                        let mut decoder = FakeDecoder;
                        (worker * 4..(worker + 1) * 4).find_map(|shot_index| {
                            let syndrome = [u8::try_from(shot_index).unwrap()];
                            decoder
                                .decode_obs(&syndrome)
                                .err()
                                .map(|error| IndexedDecodeError { shot_index, error })
                        })
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .filter_map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        assert_eq!(lowest_indexed_error(worker_errors).unwrap().shot_index, 2);
    }

    #[test]
    fn sequential_fake_decoder_reports_absolute_failure() {
        use pecos_decoder_core::obs_mask::ObsMask;
        use pecos_decoder_core::{DecoderError, ObservableDecoder};

        struct FailsAtFive;

        impl ObservableDecoder for FailsAtFive {
            fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
                if syndrome[0] == 5 {
                    Err(DecoderError::DecodingFailed("selected failure".into()))
                } else {
                    Ok(ObsMask::new())
                }
            }
        }

        let mut decoder = FailsAtFive;
        let failure = (3..8).find_map(|shot_index| {
            let syndrome = [u8::try_from(shot_index).unwrap()];
            decoder
                .decode_obs(&syndrome)
                .err()
                .map(|error| IndexedDecodeError { shot_index, error })
        });
        assert_eq!(failure.unwrap().shot_index, 5);
    }

    #[test]
    fn interval_prevalidation_enforces_verified_domain() {
        for alpha in [MIN_INTERVAL_ALPHA, MAX_INTERVAL_ALPHA] {
            assert!(validate_decode_interval(10, alpha).is_ok());
        }
        for alpha in [
            0.0,
            MIN_INTERVAL_ALPHA / 2.0,
            0.500_001,
            f64::NAN,
            f64::INFINITY,
        ] {
            assert!(validate_decode_interval(10, alpha).is_err());
        }
        assert!(validate_decode_interval(0, 0.05).is_err());
        assert!(validate_decode_interval(MAX_INTERVAL_SHOTS + 1, 0.05).is_err());
    }
}
