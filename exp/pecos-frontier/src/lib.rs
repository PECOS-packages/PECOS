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

//! Frontier approximate logical maximum-likelihood decoding.
//!
//! The decoder performs ordered dynamic programming over independent binary
//! fault mechanisms. Prefixes with identical active detector boundary and
//! logical labels are merged by log-sum-exp, preserving degeneracy mass. The
//! configured frontier width and log-mass window provide deterministic pruning,
//! and the engine's `libm` transcendentals make outputs bit-identical across
//! platforms.
//! [`FrontierDecoder`] is the parity port of Leverrier and Urbanke's Frontier
//! decoder. Its input-order defaults remain those of that port.

pub mod recovery;

use pecos_decoder_core::ObservableDecoder;
pub use pecos_trellis::factor::{Factor, FactorModel, Outcome};
pub use pecos_trellis::{
    DecoderError, MetricMode, ObsMask, SparseDem, StreamingMode, TrellisBoundedProgress,
    TrellisOrdering, TrellisResumableDecoder, TrellisStreamingDecoder,
    backward_deadline_column_order, backward_deadline_column_order_for_factors,
    deadline_column_order, deadline_column_order_for_factors,
};
use std::cmp::Ordering;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::time::Instant;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type TimerStart = Instant;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
struct TimerStart;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn timer_start() -> TimerStart {
    Instant::now()
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn timer_start() -> TimerStart {
    TimerStart
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn timer_seconds(start: TimerStart) -> f64 {
    start.elapsed().as_secs_f64()
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn timer_seconds(_: TimerStart) -> f64 {
    0.0
}

/// Parity-port name for [`pecos_trellis::TrellisDecoder`].
pub type FrontierDecoder = pecos_trellis::TrellisDecoder;
/// Parity-port name for [`pecos_trellis::TrellisConfig`].
pub type FrontierConfig = pecos_trellis::TrellisConfig;
/// Parity-port name for [`pecos_trellis::TrellisResult`].
pub type FrontierResult = pecos_trellis::TrellisResult;
/// Parity-port name for [`pecos_trellis::TrellisStatus`].
pub type FrontierStatus = pecos_trellis::TrellisStatus;
/// Parity-port name for [`pecos_trellis::TrellisDecodeAttempt`].
pub type FrontierDecodeAttempt = pecos_trellis::TrellisDecodeAttempt;
/// Parity-port name for [`pecos_trellis::TrellisLogicalMass`].
pub type FrontierLogicalMass = pecos_trellis::TrellisLogicalMass;

/// Direction selected by [`FrontierCommittee`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitteeDirection {
    /// The configured processing order.
    Forward,
    /// The plain reverse of the configured processing order.
    Backward,
}

/// Decode status for one committee leg.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitteeStatus {
    /// The leg retained at least one terminal state.
    Ok,
    /// The leg found no retained path for the syndrome. Engine faults never
    /// appear here: they abort the whole committee decode instead.
    NoPath,
}

/// Summary of one committee leg.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CommitteeMember {
    /// Decode status for this leg.
    pub status: CommitteeStatus,
    /// Total retained log evidence, or negative infinity for no path.
    pub log_evidence: f64,
}

/// Result selected from the forward/backward committee.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontierCommitteeResult {
    /// Full result from the selected leg.
    pub selected: FrontierResult,
    /// Direction of the selected leg.
    pub direction: CommitteeDirection,
    /// Forward-leg status and evidence.
    pub forward: CommitteeMember,
    /// Backward-leg status and evidence.
    pub backward: CommitteeMember,
}

/// Two-leg Frontier decoder using a processing order and its plain reverse.
#[derive(Clone, Debug)]
pub struct FrontierCommittee {
    forward: FrontierDecoder,
    backward: FrontierDecoder,
    build_seconds: f64,
}

/// Phase of an exact window committee split across bounded invocations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontierBoundedProgress {
    /// The configured-order leg is pending; a ready leg needs one more step to
    /// construct its result and prepare the backward leg, without column work.
    Forward(TrellisBoundedProgress),
    /// The reversed-order leg is pending; a ready leg needs one more step to
    /// construct its result, without column work.
    Backward(TrellisBoundedProgress),
    /// Both structured attempts are ready for the unchanged committee resolver.
    Ready,
}

/// Exact full-input committee decoding resumed across bounded calls.
///
/// Every begin starts the same independent forward/backward window attempts as
/// [`FrontierCommittee::decode`]. No beam is carried between windows. Both legs
/// always run, including after a first-leg failure, and the existing structured
/// resolver alone decides fallback, error precedence, ranking and direction.
///
/// A step does at most one leg's column budget, or a phase transition that
/// constructs one terminal result and prepares the next leg. Begin, transitions
/// and finish must also be timed: column budgets do not bound wall-clock time.
pub struct FrontierResumableCommittee {
    forward: TrellisResumableDecoder,
    backward: TrellisResumableDecoder,
    build_seconds: f64,
    attempt: Option<BoundedCommitteeAttempt>,
}

struct BoundedCommitteeAttempt {
    syndrome: Vec<u8>,
    without_dropped_mass: bool,
    phase: BoundedCommitteePhase,
    forward: Option<FrontierDecodeAttempt>,
    backward: Option<FrontierDecodeAttempt>,
}

#[derive(Clone, Copy)]
enum BoundedCommitteePhase {
    Forward,
    Backward,
    Ready,
}

impl FrontierCommittee {
    /// Construct a forward/backward committee from a sparse DEM.
    ///
    /// The forward leg uses `config.column_order` (or DEM order when absent).
    /// The backward leg uses the plain reverse of that same sequence.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] when the configuration or
    /// DEM is invalid.
    pub fn from_sparse_dem(dem: &SparseDem, config: FrontierConfig) -> Result<Self, DecoderError> {
        let build_started = timer_start();
        if config.metric_mode == MetricMode::MaxLogInt {
            return Err(DecoderError::InvalidConfiguration(
                "FrontierCommittee arbitration ranks on coset-mass posteriors; the integer max-log metric is not supported"
                    .into(),
            ));
        }
        let FrontierConfig {
            k,
            delta,
            score_alpha,
            column_order,
            merge_indistinguishable,
            bp_score_iterations,
            metric_mode,
            int_metric_scale,
        } = config;
        let mut forward_order = column_order.unwrap_or_else(|| (0..dem.mechanisms.len()).collect());
        let forward = FrontierDecoder::from_sparse_dem(
            dem,
            FrontierConfig {
                k,
                delta,
                score_alpha,
                column_order: Some(forward_order.clone()),
                merge_indistinguishable,
                bp_score_iterations,
                metric_mode,
                int_metric_scale,
            },
        )?;
        forward_order.reverse();
        let backward_config = FrontierConfig {
            k,
            delta,
            score_alpha,
            column_order: Some(forward_order),
            merge_indistinguishable,
            bp_score_iterations,
            metric_mode,
            int_metric_scale,
        };
        let backward = FrontierDecoder::from_sparse_dem(dem, backward_config)?;
        let build_seconds = timer_seconds(build_started);
        Ok(Self {
            forward,
            backward,
            build_seconds,
        })
    }

    /// Wall-clock seconds spent constructing both committee legs in
    /// [`Self::from_sparse_dem`].
    #[must_use]
    pub fn build_seconds(&self) -> f64 {
        self.build_seconds
    }

    /// Consume this committee into an owning exact resumable wrapper.
    ///
    /// Infallible: every committee is built from a sparse DEM (binary kernel)
    /// and its constructor rejects the integer metric.
    ///
    /// # Panics
    /// Panics if a committee leg is not a binary float kernel, which the
    /// committee constructor rules out.
    #[must_use]
    pub fn into_resumable(self) -> FrontierResumableCommittee {
        let resumable = |leg: FrontierDecoder| {
            leg.into_resumable().unwrap_or_else(|(error, _)| {
                panic!("committee constructor guarantees binary float legs: {error}")
            })
        };
        FrontierResumableCommittee {
            forward: resumable(self.forward),
            backward: resumable(self.backward),
            build_seconds: self.build_seconds,
            attempt: None,
        }
    }

    /// Parse a Stim-format detector error model and construct a committee.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if parsing or committee construction fails.
    pub fn from_dem_str(dem_str: &str, config: FrontierConfig) -> Result<Self, DecoderError> {
        let dem = SparseDem::from_dem_str(dem_str)?;
        Self::from_sparse_dem(&dem, config)
    }

    /// Decode dense shots in input order with shared models and worker-local scratch.
    ///
    /// Workers are capped at one per shot, with one worker for an empty batch.
    ///
    /// # Errors
    /// Returns `InvalidConfiguration` for zero workers and `InternalError` for pool creation failure.
    /// Individual shot errors are retained in input order.
    pub fn decode_batch(
        &self,
        shots: &[Vec<u8>],
        workers: usize,
    ) -> Result<Vec<Result<FrontierCommitteeResult, DecoderError>>, DecoderError> {
        pecos_trellis::batch::decode_batch(
            shots,
            workers,
            || Self {
                forward: self.forward.fresh_worker(),
                backward: self.backward.fresh_worker(),
                build_seconds: self.build_seconds,
            },
            Self::decode,
        )
    }

    /// Decode with both processing directions and select the stronger result.
    ///
    /// Failure provenance comes from the engine's structured
    /// [`TrellisDecodeAttempt`](FrontierDecodeAttempt): only a no-path outcome
    /// lets the committee fall back to the other leg. Any other failure --
    /// dimension mismatch, configuration or internal fault -- aborts the whole
    /// committee even if the other leg produced a valid decode, because both
    /// legs saw the same input and such a fault is a property of the call, not
    /// of the processing direction.
    ///
    /// # Errors
    ///
    /// Returns the engine's own no-path error when both legs find no retained
    /// path, and propagates any non-no-path leg failure unchanged. When both
    /// legs fail with engine faults, the forward leg's error is returned.
    pub fn decode(&mut self, syndrome: &[u8]) -> Result<FrontierCommitteeResult, DecoderError> {
        // Each leg owns its BP graph and scratch and runs independently on the
        // same syndrome. Their beliefs are expected to match, but are never
        // shared by reference between the two code paths.
        let forward_attempt = self.forward.decode_attempt(syndrome);
        let backward_attempt = self.backward.decode_attempt(syndrome);
        resolve_committee_attempts(forward_attempt, backward_attempt)
    }

    /// Decode with full retained class masses and confidence, without the
    /// discarded-prefix-mass diagnostic.
    ///
    /// This explicit opt-in uses the same configured pruning parameters,
    /// retained states, floating-point reduction order, and committee arbitration
    /// as [`Self::decode`]. It preserves the predicted class, complete retained
    /// logical-mass vector, retained log evidence, class gap, directions, statuses,
    /// transition counts, and dropped-state counts. The selected result's
    /// [`FrontierResult::dropped_log_mass`] is unavailable and is set to `NaN`.
    ///
    /// Skipping that diagnostic avoids sorting discarded states and summing their
    /// masses at every pruning step. Discarded mass is not used to score, prune,
    /// merge, normalize the retained class masses, or select a committee leg.
    /// Call [`Self::decode`] when the discarded-mass diagnostic is required;
    /// its behavior and default remain unchanged.
    ///
    /// # Errors
    ///
    /// Returns the same errors, including genuine no-path and non-no-path engine
    /// failures, as [`Self::decode`]. This does not enable retries or recovery.
    pub fn decode_without_dropped_mass(
        &mut self,
        syndrome: &[u8],
    ) -> Result<FrontierCommitteeResult, DecoderError> {
        let forward_attempt = self.forward.decode_attempt_without_dropped_mass(syndrome);
        let backward_attempt = self.backward.decode_attempt_without_dropped_mass(syndrome);
        resolve_committee_attempts(forward_attempt, backward_attempt)
    }

    /// Decode like [`Self::decode`], running the backward leg on a second thread.
    ///
    /// The legs share no state, so the result is identical to [`Self::decode`];
    /// only the wall-clock latency changes. This spawns one scoped thread per
    /// call, so combining it with multi-worker [`Self::decode_batch`] would use
    /// two threads per batch worker.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::decode`], plus `InternalError` when
    /// the second thread cannot be spawned.
    pub fn decode_parallel(
        &mut self,
        syndrome: &[u8],
    ) -> Result<FrontierCommitteeResult, DecoderError> {
        let Self {
            forward, backward, ..
        } = self;
        let (forward_attempt, backward_attempt) = std::thread::scope(|scope| {
            let backward_leg = std::thread::Builder::new()
                .spawn_scoped(scope, || backward.decode_attempt(syndrome))
                .map_err(|error| DecoderError::InternalError(error.to_string()))?;
            let forward_attempt = forward.decode_attempt(syndrome);
            let backward_attempt = backward_leg
                .join()
                .unwrap_or_else(|payload| std::panic::resume_unwind(payload));
            Ok::<_, DecoderError>((forward_attempt, backward_attempt))
        })?;
        resolve_committee_attempts(forward_attempt, backward_attempt)
    }
}

impl FrontierResumableCommittee {
    /// Begin the configured-order leg of a new full-syndrome window.
    ///
    /// Preparation errors remain structured until both legs reach the resolver.
    /// This performs no DP columns or terminal class aggregation.
    ///
    /// # Errors
    /// Returns `InvalidConfiguration` while another window is pending.
    pub fn begin(&mut self, syndrome: &[u8]) -> Result<FrontierBoundedProgress, DecoderError> {
        self.begin_with_mode(syndrome, false)
    }

    /// Begin the same exact committee without discarded-mass bookkeeping.
    ///
    /// All retained outputs and counts are preserved. Only the selected result's
    /// `dropped_log_mass` is unavailable (NaN), as in the unbounded opt-in API.
    ///
    /// # Errors
    /// Returns the same protocol error as [`Self::begin`].
    pub fn begin_without_dropped_mass(
        &mut self,
        syndrome: &[u8],
    ) -> Result<FrontierBoundedProgress, DecoderError> {
        self.begin_with_mode(syndrome, true)
    }

    fn begin_with_mode(
        &mut self,
        syndrome: &[u8],
        without_dropped_mass: bool,
    ) -> Result<FrontierBoundedProgress, DecoderError> {
        if self.attempt.is_some() {
            return Err(DecoderError::InvalidConfiguration(
                "finish or cancel the pending resumable committee before beginning another".into(),
            ));
        }
        let progress = if without_dropped_mass {
            self.forward.begin_without_dropped_mass(syndrome)?
        } else {
            self.forward.begin(syndrome)?
        };
        self.attempt = Some(BoundedCommitteeAttempt {
            syndrome: syndrome.to_vec(),
            without_dropped_mass,
            phase: BoundedCommitteePhase::Forward,
            forward: None,
            backward: None,
        });
        Ok(FrontierBoundedProgress::Forward(progress))
    }

    /// Advance one leg by at most `max_columns`, or perform one phase boundary.
    ///
    /// A phase-boundary call performs no column work. It constructs a ready
    /// leg's original result and, after forward, prepares backward on the same
    /// stored input. Both legs run even when forward produced `NoPath` or `Error`.
    /// Positive-budget steps on a ready committee are idempotent.
    ///
    /// # Errors
    /// Returns `InvalidConfiguration` for zero budget or a missing begin.
    ///
    /// # Panics
    /// Panics if the private phase and its owned leg attempt disagree,
    /// indicating an internal coordinator invariant violation.
    pub fn step(&mut self, max_columns: usize) -> Result<FrontierBoundedProgress, DecoderError> {
        if max_columns == 0 {
            return Err(DecoderError::InvalidConfiguration(
                "resumable committees require a positive column budget".into(),
            ));
        }
        let state = self.attempt.as_mut().ok_or_else(|| {
            DecoderError::InvalidConfiguration("begin a resumable committee before stepping".into())
        })?;
        match state.phase {
            BoundedCommitteePhase::Forward => {
                if !self
                    .forward
                    .progress()
                    .expect("coordinator owns the forward attempt")
                    .ready_to_finish
                {
                    return self
                        .forward
                        .step(max_columns)
                        .map(FrontierBoundedProgress::Forward);
                }
                state.forward = Some(self.forward.finish()?);
                let progress = if state.without_dropped_mass {
                    self.backward.begin_without_dropped_mass(&state.syndrome)?
                } else {
                    self.backward.begin(&state.syndrome)?
                };
                state.phase = BoundedCommitteePhase::Backward;
                Ok(FrontierBoundedProgress::Backward(progress))
            }
            BoundedCommitteePhase::Backward => {
                if !self
                    .backward
                    .progress()
                    .expect("coordinator owns the backward attempt")
                    .ready_to_finish
                {
                    return self
                        .backward
                        .step(max_columns)
                        .map(FrontierBoundedProgress::Backward);
                }
                state.backward = Some(self.backward.finish()?);
                state.phase = BoundedCommitteePhase::Ready;
                Ok(FrontierBoundedProgress::Ready)
            }
            BoundedCommitteePhase::Ready => Ok(FrontierBoundedProgress::Ready),
        }
    }

    /// Inspect the pending phase without doing DP or terminal work.
    ///
    /// # Panics
    /// Panics if a pending private phase lacks its owned leg attempt,
    /// indicating an internal coordinator invariant violation.
    #[must_use]
    pub fn progress(&self) -> Option<FrontierBoundedProgress> {
        self.attempt.as_ref().map(|state| match state.phase {
            BoundedCommitteePhase::Forward => FrontierBoundedProgress::Forward(
                self.forward
                    .progress()
                    .expect("coordinator owns the forward attempt"),
            ),
            BoundedCommitteePhase::Backward => FrontierBoundedProgress::Backward(
                self.backward
                    .progress()
                    .expect("coordinator owns the backward attempt"),
            ),
            BoundedCommitteePhase::Ready => FrontierBoundedProgress::Ready,
        })
    }

    /// Consume both ready attempts with the unchanged committee resolver.
    ///
    /// This does no DP or terminal class construction. Protocol failures leave
    /// the pending window intact; an engine error consumes the completed window.
    ///
    /// # Errors
    /// Returns `InvalidConfiguration` before both legs are ready; otherwise
    /// returns exactly the structured resolver's engine error, if any.
    ///
    /// # Panics
    /// Panics if the checked ready phase lacks its stored leg attempts,
    /// indicating an internal coordinator invariant violation.
    pub fn finish(&mut self) -> Result<FrontierCommitteeResult, DecoderError> {
        if !matches!(self.progress(), Some(FrontierBoundedProgress::Ready)) {
            return Err(DecoderError::InvalidConfiguration(
                "drain both resumable committee legs before finishing".into(),
            ));
        }
        let state = self.attempt.take().expect("checked pending committee");
        resolve_committee_attempts(
            state.forward.expect("ready forward attempt"),
            state.backward.expect("ready backward attempt"),
        )
    }

    /// Cancel both pending legs. The next begin prepares a fresh window.
    pub fn cancel(&mut self) {
        self.forward.cancel();
        self.backward.cancel();
        self.attempt = None;
    }

    /// The original committee's model-construction cost.
    #[must_use]
    pub fn build_seconds(&self) -> f64 {
        self.build_seconds
    }
}

impl ObservableDecoder for FrontierCommittee {
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        Ok(self.decode(syndrome)?.selected.predicted)
    }
}

/// Turn the two legs' structured attempts into a committee result.
///
/// Provenance is the engine's own: [`FrontierDecodeAttempt::NoPath`] is the
/// only failure the committee absorbs, and [`FrontierDecodeAttempt::Error`]
/// always propagates -- checked before anything else, forward leg first --
/// because an engine fault on a shared input can never be repaired by
/// preferring the other processing direction.
fn resolve_committee_attempts(
    forward_attempt: FrontierDecodeAttempt,
    backward_attempt: FrontierDecodeAttempt,
) -> Result<FrontierCommitteeResult, DecoderError> {
    use pecos_trellis::TrellisDecodeAttempt::{Error, NoPath, Success};

    let forward = committee_member(&forward_attempt);
    let backward = committee_member(&backward_attempt);
    let (selected, direction) = match (forward_attempt, backward_attempt) {
        // Pattern order is the precedence: a forward fault, then a backward
        // fault, and only then the double no-path with the engine's own error.
        (Error(error), _) | (_, Error(error)) | (NoPath { error, .. }, NoPath { .. }) => {
            return Err(error);
        }
        (Success(selected), NoPath { .. }) => (selected, CommitteeDirection::Forward),
        (NoPath { .. }, Success(selected)) => (selected, CommitteeDirection::Backward),
        (Success(forward_result), Success(backward_result)) => {
            if compare_committee_legs(Some(&forward_result), Some(&backward_result))
                == Ordering::Less
            {
                (backward_result, CommitteeDirection::Backward)
            } else {
                (forward_result, CommitteeDirection::Forward)
            }
        }
    };

    Ok(FrontierCommitteeResult {
        selected,
        direction,
        forward,
        backward,
    })
}

/// Summarize one leg for the committee result.
///
/// A non-`Ok` status here is always a genuine no-path:
/// [`resolve_committee_attempts`] propagates every engine fault before a
/// member summary is returned.
fn committee_member(attempt: &FrontierDecodeAttempt) -> CommitteeMember {
    match attempt {
        FrontierDecodeAttempt::Success(decoded) => CommitteeMember {
            status: CommitteeStatus::Ok,
            log_evidence: decoded.log_evidence,
        },
        FrontierDecodeAttempt::NoPath { .. } | FrontierDecodeAttempt::Error(_) => CommitteeMember {
            status: CommitteeStatus::NoPath,
            log_evidence: f64::NEG_INFINITY,
        },
    }
}

fn compare_committee_legs(
    forward: Option<&FrontierResult>,
    backward: Option<&FrontierResult>,
) -> Ordering {
    let forward_rank = committee_rank(forward, true);
    let backward_rank = committee_rank(backward, false);
    forward_rank
        .iter()
        .zip(backward_rank)
        .find_map(|(forward_component, backward_component)| {
            let ordering = forward_component.total_cmp(&backward_component);
            (ordering != Ordering::Equal).then_some(ordering)
        })
        .unwrap_or(Ordering::Equal)
}

fn committee_rank(result: Option<&FrontierResult>, is_forward: bool) -> [f64; 6] {
    let forward_bonus = if is_forward { 1.0 } else { 0.0 };
    let Some(result) = result else {
        return [
            1.0,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            0.0,
            forward_bonus,
        ];
    };

    let terminal_gap = result.runner_up_gap.unwrap_or(f64::INFINITY);
    let terminal_gap = if terminal_gap.is_nan() {
        f64::NEG_INFINITY
    } else {
        terminal_gap
    };
    let top_one_posterior = result
        .logical_masses
        .first()
        .map_or(f64::NEG_INFINITY, |winner| {
            winner.log_mass - result.log_evidence
        });
    let top_one_posterior = if top_one_posterior.is_finite() {
        top_one_posterior
    } else {
        f64::NEG_INFINITY
    };
    [
        2.0,
        result.log_evidence,
        terminal_gap,
        top_one_posterior,
        0.0,
        forward_bonus,
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        CommitteeStatus, DecoderError, FrontierCommittee, FrontierConfig, FrontierDecodeAttempt,
        FrontierDecoder, FrontierLogicalMass, FrontierResult, FrontierStatus, MetricMode,
        SparseDem, committee_rank, resolve_committee_attempts,
    };
    use pecos_decoder_core::obs_mask::ObsMask;
    use std::collections::BTreeMap;

    #[test]
    fn resumable_committee_preserves_both_complete_leg_attempts() {
        fn fingerprint(attempt: &FrontierDecodeAttempt) -> String {
            match attempt {
                FrontierDecodeAttempt::Success(result) => {
                    let mut result = result.clone();
                    assert!(result.bp_seconds.is_finite() && result.bp_seconds >= 0.0);
                    result.bp_seconds = 0.0;
                    format!("{result:?}")
                }
                FrontierDecodeAttempt::NoPath {
                    error,
                    transitions,
                    dropped_states,
                    bp_seconds,
                } => {
                    assert!(bp_seconds.is_finite() && *bp_seconds >= 0.0);
                    format!("{error:?}/{transitions}/{dropped_states}")
                }
                FrontierDecodeAttempt::Error(error) => format!("Error({error:?})"),
            }
        }
        let dem = SparseDem {
            mechanisms: vec![
                (0.15, vec![0], vec![]),
                (0.15, vec![0, 1], vec![130]),
                (0.08, vec![1], vec![]),
                (0.21, vec![0], vec![0]),
            ],
            detector_coords: BTreeMap::new(),
            num_detectors: 2,
            num_observables: 131,
        };
        for bp_score_iterations in [0, 2] {
            let base = FrontierCommittee::from_sparse_dem(
                &dem,
                FrontierConfig {
                    k: 1,
                    bp_score_iterations,
                    merge_indistinguishable: false,
                    ..FrontierConfig::default()
                },
            )
            .unwrap();
            let mut batch = base.clone();
            let mut bounded = base.into_resumable();
            for budget in [1, 2, 7, 64] {
                for shot in [&[0, 0][..], &[1, 0][..], &[0, 1][..], &[1, 1][..], &[0][..]] {
                    let forward = batch.forward.decode_attempt(shot);
                    let backward = batch.backward.decode_attempt(shot);
                    bounded.begin(shot).unwrap();
                    while bounded.progress() != Some(super::FrontierBoundedProgress::Ready) {
                        bounded.step(budget).unwrap();
                    }
                    let state = bounded.attempt.as_ref().unwrap();
                    assert_eq!(
                        fingerprint(&forward),
                        fingerprint(state.forward.as_ref().unwrap())
                    );
                    assert_eq!(
                        fingerprint(&backward),
                        fingerprint(state.backward.as_ref().unwrap())
                    );
                    assert_eq!(batch.forward.bp_refreshes(), bounded.forward.bp_refreshes());
                    assert_eq!(
                        batch.backward.bp_refreshes(),
                        bounded.backward.bp_refreshes()
                    );
                    let _ = bounded.finish();
                }
            }
        }
    }

    #[test]
    fn resumable_committee_runs_both_legs_through_one_sided_engine_faults() {
        // Legs expecting different detector counts make exactly one leg fail
        // with an engine error on the same input, in either phase.
        let config = FrontierConfig::default();
        let two =
            FrontierDecoder::from_dem_str("error(0.2) D0 L0\nerror(0.1) D0 D1", config.clone())
                .unwrap();
        let three = FrontierDecoder::from_dem_str(
            "error(0.2) D0 L0\nerror(0.1) D0 D1\ndetector D2",
            config,
        )
        .unwrap();
        for (forward, backward) in [(two.clone(), three.clone()), (three, two)] {
            let mut batch_forward = forward.clone();
            let mut batch_backward = backward.clone();
            let mut bounded = super::FrontierResumableCommittee {
                forward: forward.into_resumable().unwrap(),
                backward: backward.into_resumable().unwrap(),
                build_seconds: 0.0,
                attempt: None,
            };
            for shot in [&[1, 0][..], &[1, 0, 0][..]] {
                let expected = resolve_committee_attempts(
                    batch_forward.decode_attempt(shot),
                    batch_backward.decode_attempt(shot),
                );
                bounded.begin(shot).unwrap();
                while bounded.progress() != Some(super::FrontierBoundedProgress::Ready) {
                    bounded.step(1).unwrap();
                }
                let state = bounded.attempt.as_ref().unwrap();
                let legs = [state.forward.as_ref(), state.backward.as_ref()];
                assert_eq!(
                    legs.iter()
                        .filter(|leg| matches!(leg, Some(FrontierDecodeAttempt::Error(_))))
                        .count(),
                    1,
                    "exactly one leg faults"
                );
                assert!(
                    legs.iter()
                        .any(|leg| matches!(leg, Some(FrontierDecodeAttempt::Success(_)))),
                    "the other leg still runs to completion"
                );
                let actual = bounded.finish();
                assert!(matches!(
                    actual,
                    Err(DecoderError::InvalidDimensions { .. })
                ));
                assert_eq!(format!("{expected:?}"), format!("{actual:?}"));
            }
        }
    }

    #[test]
    fn committee_batch_matches_sequential() {
        let dem =
            SparseDem::from_dem_str("error(0.2) D0 D1 L0\nerror(0.1) D0\ndetector D2\n").unwrap();
        let mut decoder = FrontierCommittee::from_sparse_dem(
            &dem,
            FrontierConfig {
                k: 1,
                bp_score_iterations: 5,
                ..FrontierConfig::default()
            },
        )
        .unwrap();
        let shots: Vec<Vec<u8>> = (0..257)
            .map(|i| (0..3).map(|bit| u8::from(i & (1 << bit) != 0)).collect())
            .collect();
        for workers in [1, 4] {
            for (actual, shot) in decoder
                .decode_batch(&shots, workers)
                .unwrap()
                .into_iter()
                .zip(&shots)
            {
                match (actual, decoder.decode(shot)) {
                    (Ok(mut actual), Ok(mut expected)) => {
                        actual.selected.bp_seconds = 0.0;
                        expected.selected.bp_seconds = 0.0;
                        assert_eq!(actual, expected);
                    }
                    (Err(actual), Err(expected)) => {
                        assert_eq!(actual.to_string(), expected.to_string());
                    }
                    _ => panic!("committee batch outcome differs"),
                }
            }
        }
        assert!(matches!(
            decoder.decode_batch(&[], 0),
            Err(DecoderError::InvalidConfiguration(_))
        ));
    }

    fn no_path_attempt() -> FrontierDecodeAttempt {
        FrontierDecodeAttempt::NoPath {
            error: DecoderError::DecodingFailed(
                "syndrome is unexplainable at the given pruning parameters".into(),
            ),
            transitions: 0,
            bp_seconds: 0.0,
            dropped_states: 0,
        }
    }

    fn fault_attempt(expected: usize, actual: usize) -> FrontierDecodeAttempt {
        FrontierDecodeAttempt::Error(DecoderError::InvalidDimensions { expected, actual })
    }

    /// The full failure-arbitration matrix, on synthesized attempts so every
    /// combination is reachable: an engine fault always propagates (forward
    /// first), and only a double no-path returns the engine's no-path error.
    #[test]
    fn committee_arbitration_covers_every_failure_combination() {
        // fault x fault -> the forward fault.
        let error = resolve_committee_attempts(fault_attempt(1, 0), fault_attempt(2, 0))
            .expect_err("two faults must not produce a result");
        assert!(matches!(
            error,
            DecoderError::InvalidDimensions { expected: 1, .. }
        ));

        // fault x no-path (either order) -> the fault, never the no-path.
        for (forward, backward) in [
            (fault_attempt(1, 0), no_path_attempt()),
            (no_path_attempt(), fault_attempt(1, 0)),
        ] {
            let error = resolve_committee_attempts(forward, backward)
                .expect_err("a fault beside a no-path must not produce a result");
            assert!(
                matches!(error, DecoderError::InvalidDimensions { .. }),
                "fault must win over no-path, got {error:?}"
            );
        }

        // no-path x no-path -> the engine's own no-path error.
        let error = resolve_committee_attempts(no_path_attempt(), no_path_attempt())
            .expect_err("two no-paths must not produce a result");
        assert!(matches!(error, DecoderError::DecodingFailed(_)));
        assert!(error.to_string().contains("unexplainable"));
    }

    /// A leg fault aborts the committee even when the other leg SUCCEEDED:
    /// both legs saw the same input, so the fault is the call's, and a valid
    /// result from one direction must not hide it.
    #[test]
    fn committee_fault_beside_a_success_still_aborts() {
        let dem = SparseDem {
            mechanisms: vec![(0.2, vec![0], vec![0])],
            detector_coords: BTreeMap::new(),
            num_detectors: 1,
            num_observables: 1,
        };
        let mut leg = super::FrontierDecoder::from_sparse_dem(
            &dem,
            FrontierConfig {
                k: usize::MAX,
                delta: f64::INFINITY,
                score_alpha: 0.8,
                column_order: None,
                merge_indistinguishable: false,
                bp_score_iterations: 0,
                metric_mode: MetricMode::default(),
                int_metric_scale: 1024,
            },
        )
        .unwrap();
        let success = leg.decode_attempt(&[1]);
        assert!(matches!(success, FrontierDecodeAttempt::Success(_)));

        let error = resolve_committee_attempts(success, fault_attempt(1, 0))
            .expect_err("a fault must abort even beside a success");
        assert!(matches!(error, DecoderError::InvalidDimensions { .. }));

        // A success beside a genuine no-path is still selected, with the
        // failed leg reported honestly.
        let success = leg.decode_attempt(&[1]);
        let result = resolve_committee_attempts(success, no_path_attempt())
            .expect("a success beside a no-path must be selected");
        assert_eq!(result.backward.status, CommitteeStatus::NoPath);
        assert!(result.backward.log_evidence.is_infinite() && result.backward.log_evidence < 0.0);
    }

    #[test]
    fn committee_merges_both_legs() {
        let dem = SparseDem {
            mechanisms: vec![
                (0.3, vec![0], vec![0]),
                (0.2, vec![1], vec![]),
                (0.4, vec![0], vec![0]),
            ],
            detector_coords: BTreeMap::new(),
            num_detectors: 2,
            num_observables: 1,
        };
        let mut committee = FrontierCommittee::from_sparse_dem(
            &dem,
            FrontierConfig {
                k: usize::MAX,
                delta: f64::INFINITY,
                score_alpha: 0.8,
                column_order: None,
                merge_indistinguishable: true,
                bp_score_iterations: 0,
                metric_mode: MetricMode::default(),
                int_metric_scale: 1024,
            },
        )
        .unwrap();
        let forward = committee.forward.decode(&[0, 0]).unwrap();
        let backward = committee.backward.decode(&[0, 0]).unwrap();

        assert_eq!(forward.processed_columns, 2);
        assert_eq!(backward.processed_columns, 2);
        assert_eq!(forward.processed_columns, backward.processed_columns);
    }

    #[test]
    fn committee_rejects_maxlog_metric() {
        let dem = SparseDem {
            mechanisms: vec![(0.2, vec![0], vec![0])],
            detector_coords: BTreeMap::new(),
            num_detectors: 1,
            num_observables: 1,
        };
        let error = FrontierCommittee::from_sparse_dem(
            &dem,
            FrontierConfig {
                metric_mode: MetricMode::MaxLogInt,
                ..FrontierConfig::default()
            },
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Invalid configuration: FrontierCommittee arbitration ranks on coset-mass posteriors; the integer max-log metric is not supported"
        );
    }

    #[test]
    fn committee_bp_legs_do_not_share_bp_state() {
        let dem = SparseDem {
            mechanisms: vec![
                (0.15, vec![0], vec![]),
                (0.15, vec![0, 1], vec![0]),
                (0.08, vec![1], vec![]),
            ],
            detector_coords: BTreeMap::new(),
            num_detectors: 2,
            num_observables: 1,
        };
        let committee = FrontierCommittee::from_sparse_dem(
            &dem,
            FrontierConfig {
                k: 1,
                delta: f64::INFINITY,
                score_alpha: 0.8,
                column_order: None,
                merge_indistinguishable: false,
                bp_score_iterations: 5,
                metric_mode: MetricMode::default(),
                int_metric_scale: 1024,
            },
        )
        .unwrap();
        let (forward_graph, forward_scratch) = committee.forward.bp_state_addrs().unwrap();
        let (backward_graph, backward_scratch) = committee.backward.bp_state_addrs().unwrap();

        assert_ne!(forward_graph, forward_scratch);
        assert_ne!(forward_graph, backward_graph);
        assert_ne!(forward_graph, backward_scratch);
        assert_ne!(forward_scratch, backward_graph);
        assert_ne!(forward_scratch, backward_scratch);
        assert_ne!(backward_graph, backward_scratch);
    }

    #[test]
    fn committee_rank_maps_special_terminal_statistics() {
        let no_runner_up = FrontierResult {
            predicted: ObsMask::new(),
            log_evidence: -1.0,
            runner_up_gap: None,
            peak_retained_states: 1,
            processed_columns: 0,
            transitions: 0,
            dropped_states: 0,
            dropped_log_mass: f64::NEG_INFINITY,
            bp_seconds: 0.0,
            bp_runs: 0,
            escalation_rungs_used: 0,
            status: FrontierStatus::Exact,
            logical_masses: vec![FrontierLogicalMass {
                logical: ObsMask::new(),
                log_mass: f64::NAN,
            }],
        };
        let rank = committee_rank(Some(&no_runner_up), true);
        assert_eq!(rank[2].to_bits(), f64::INFINITY.to_bits());
        assert_eq!(rank[3].to_bits(), f64::NEG_INFINITY.to_bits());
        assert_eq!(rank[5].to_bits(), 1.0_f64.to_bits());

        let nan_gap = FrontierResult {
            runner_up_gap: Some(f64::NAN),
            logical_masses: vec![FrontierLogicalMass {
                logical: ObsMask::new(),
                log_mass: -1.5,
            }],
            ..no_runner_up
        };
        let rank = committee_rank(Some(&nan_gap), false);
        assert_eq!(rank[2].to_bits(), f64::NEG_INFINITY.to_bits());
        assert_eq!(rank[3].to_bits(), (-0.5_f64).to_bits());

        let no_path_rank = committee_rank(None, false);
        assert_eq!(no_path_rank[0].to_bits(), 1.0_f64.to_bits());
        assert_eq!(no_path_rank[1].to_bits(), f64::NEG_INFINITY.to_bits());
    }
}
