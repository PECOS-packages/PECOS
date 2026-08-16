//! Pure planning primitives for batch decoder execution.

use std::fmt;
use std::ops::Range;

/// Below this many shots, setting up worker threads dominates generic decoding.
pub const SMALL_BATCH_SEQUENTIAL_THRESHOLD: usize = 1000;

/// Auto-planning gives each generic decoder worker at least this many shots.
pub const MIN_SHOTS_PER_WORKER: usize = 256;

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

/// Return the contiguous shot range assigned to one parallel worker.
///
/// Invalid worker indices and a zero worker count produce an empty range at
/// the end of the batch; execution planning rejects zero workers before this
/// helper is reached.
#[must_use]
pub fn parallel_worker_range(
    num_shots: usize,
    workers: usize,
    worker_index: usize,
) -> Range<usize> {
    if workers == 0 || worker_index >= workers {
        return num_shots..num_shots;
    }
    let chunk_size = num_shots.div_ceil(workers);
    let start = worker_index.saturating_mul(chunk_size).min(num_shots);
    let end = start.saturating_add(chunk_size).min(num_shots);
    start..end
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn parallel_worker_ranges_cover_empty_and_oversubscribed_batches() {
        let oversubscribed = (0..4)
            .map(|worker| parallel_worker_range(2, 4, worker))
            .collect::<Vec<_>>();
        assert_eq!(oversubscribed, vec![0..1, 1..2, 2..2, 2..2]);

        let empty = (0..4)
            .map(|worker| parallel_worker_range(0, 4, worker))
            .collect::<Vec<_>>();
        assert_eq!(empty, vec![0..0, 0..0, 0..0, 0..0]);
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
                        parallel_worker_range(16, 4, worker).find_map(|shot_index| {
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
