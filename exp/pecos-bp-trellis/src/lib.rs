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

//! PECOS's BP-guided trellis decoder for logical coset posterior mass.
//!
//! [`BpTrellisDecoder`] is a degeneracy-aware approximate logical
//! maximum-likelihood decoder, exact in the unpruned limit. It is optimal
//! relative to the supplied detector error model (DEM), not the underlying
//! physics. Pruned results have no certified bound on discarded posterior
//! mass. Belief propagation (BP) guides only which states pruning retains and
//! never changes branch probabilities or mass arithmetic. It is not a wrap or
//! port of an external project. The shared engine is PECOS-native; its
//! bitwise parity pinning against an external reference implementation is
//! maintained elsewhere and is not a constraint on this decoder.
//!
//! The facade owns one [`TrellisDecoder`] configured with PECOS's
//! defaults, ordering semantics, and optional no-path escalation ladder. The
//! trellis engine lives in `pecos-trellis`.

use pecos_decoder_core::ObservableDecoder;
pub use pecos_trellis::TrellisOrdering;
use pecos_trellis::{
    DecoderError, MetricMode, ObsMask, PruneParams, SparseDem, TrellisConfig, TrellisDecodeAttempt,
    TrellisDecoder, TrellisPrepared, TrellisResult,
};
use std::time::Instant;

/// A retry's pruning parameters on the shared model and prepared shot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EscalationRung {
    /// Maximum number of retained boundary states.
    pub k: usize,
    /// Non-negative log-score window.
    pub delta: f64,
}

/// Why a shot has no retained path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoPathCause {
    /// Lowest detector with a residual no mechanism can change.
    Residual { detector: usize },
    /// An attempt found no path without dropping any states.
    Infeasible,
    /// Every attempt found no path after dropping states.
    Exhausted,
}

/// Per-shot no-path information, including an explicitly marked placeholder.
#[derive(Clone, Debug, PartialEq)]
pub struct NoPathReport {
    /// Reason the shot has no path.
    pub cause: NoPathCause,
    /// Initial forced observable contribution, not a decoded correction.
    pub placeholder: ObsMask,
    /// Number of attempts after the base attempt.
    pub rungs_tried: u32,
    /// Candidate branch evaluations across all attempts.
    pub transitions: u64,
    /// BP refreshes actually run for this shot.
    pub bp_runs: u32,
    /// Preparation's BP time, counted once.
    pub bp_seconds: f64,
}

impl NoPathReport {
    fn into_error(self) -> DecoderError {
        DecoderError::DecodingFailed(match self.cause {
            NoPathCause::Residual { detector } => format!(
                "syndrome is unexplainable: detector {detector} has a residual no mechanism can change"
            ),
            NoPathCause::Infeasible => {
                "syndrome is unexplainable under the detector error model".into()
            }
            NoPathCause::Exhausted => format!(
                "syndrome is unexplainable at the given pruning parameters after {} escalation rungs",
                self.rungs_tried
            ),
        })
    }
}

/// A decoded correction or an explicit per-shot no-path report.
#[derive(Clone, Debug, PartialEq)]
pub enum BpTrellisOutcome {
    /// At least one terminal state survived.
    Decoded(TrellisResult),
    /// No correction was found; the report carries a placeholder.
    NoPath(NoPathReport),
}

/// Configuration for PECOS's [`BpTrellisDecoder`].
///
/// These defaults are provisional. In particular, `k = 8` was validated as
/// near-floor on the BB144 benchmark used to select BP-guided retention, but
/// broader code and noise-model validation is still pending.
#[derive(Clone, Debug, PartialEq)]
pub struct BpTrellisConfig {
    /// Maximum number of boundary states retained after each column.
    pub k: usize,
    /// Log-mass window below the best boundary state retained after each
    /// column.
    pub delta: f64,
    /// Weight applied to the suffix-compatibility score during pruning.
    pub score_alpha: f64,
    /// Number of min-sum BP iterations used only to score pruning candidates.
    pub bp_score_iterations: usize,
    /// Merge probabilistic mechanisms with identical detector and observable
    /// sets using their exact XOR-combined probability.
    pub merge_indistinguishable: bool,
    /// Mechanism processing order.
    pub ordering: TrellisOrdering,
    /// Escalation ladder used only after a no-path decode.
    ///
    /// Rungs reuse one model and one BP preparation. They need not be monotone.
    /// An empty ladder disables escalation and is the default.
    pub escalation: Vec<EscalationRung>,
}

impl BpTrellisConfig {
    /// Validate the base configuration and every rung without a detector error model.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] for an oversized ladder, an invalid
    /// rung, or a non-empty ladder on an exact base (an exact base never prunes, so no
    /// rung could help).
    pub fn validate(&self) -> Result<(), DecoderError> {
        if u32::try_from(self.escalation.len()).is_err() {
            return Err(DecoderError::InvalidConfiguration(
                "escalation ladder has more rungs than escalation_rungs_used can represent".into(),
            ));
        }
        let mut config = self.trellis_config();
        config.validate()?;
        if !self.escalation.is_empty() && self.k == usize::MAX && self.delta.is_infinite() {
            return Err(DecoderError::InvalidConfiguration(
                "an exact base never prunes so no rung can help".into(),
            ));
        }
        for (rung, params) in self.escalation.iter().enumerate() {
            config.k = params.k;
            config.delta = params.delta;
            config.validate().map_err(|error| match error {
                DecoderError::InvalidConfiguration(message) => {
                    DecoderError::InvalidConfiguration(format!("escalation[{rung}]: {message}"))
                }
                other => other,
            })?;
        }
        Ok(())
    }

    fn trellis_config(&self) -> TrellisConfig {
        TrellisConfig {
            k: self.k,
            delta: self.delta,
            score_alpha: self.score_alpha,
            column_order: None,
            merge_indistinguishable: self.merge_indistinguishable,
            bp_score_iterations: self.bp_score_iterations,
            // BpTrellis escalation is defined over coset masses, so max-log is deliberately absent from its config.
            metric_mode: MetricMode::LogSumExpFloat,
            int_metric_scale: 1024,
        }
    }
}

impl Default for BpTrellisConfig {
    fn default() -> Self {
        Self {
            k: 8,
            delta: 100.0,
            score_alpha: 0.8,
            bp_score_iterations: 5,
            merge_indistinguishable: true,
            ordering: TrellisOrdering::Deadline,
            escalation: Vec::new(),
        }
    }
}

/// PECOS's BP-guided trellis decoder for logical coset posterior mass.
///
/// This is a degeneracy-aware approximate logical maximum-likelihood decoder,
/// exact in the unpruned limit. It is optimal relative to the supplied DEM,
/// not the underlying physics. Pruned results have no certified bound on
/// discarded posterior mass. BP guides only which states pruning retains and
/// never changes the engine's branch probabilities or mass arithmetic.
///
/// This PECOS decoder is not a wrap or port of an external project. It owns a
/// [`TrellisDecoder`] configured with PECOS's defaults and ordering semantics
/// and shares that engine's [`TrellisResult`].
#[derive(Clone, Debug)]
pub struct BpTrellisDecoder {
    inner: TrellisDecoder,
    escalation: Vec<EscalationRung>,
    has_wide_observables: bool,
    build_seconds: f64,
}

impl BpTrellisDecoder {
    /// Construct a decoder from a sparse detector error model.
    ///
    /// Unlike [`TrellisDecoder`], the default ordering is the explicitly
    /// computed [`pecos_trellis::deadline_column_order`], not input order.
    /// Every escalation rung reuses the same immutable engine model.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if ordering generation or the mapped trellis
    /// configuration fails validation.
    pub fn from_sparse_dem(dem: &SparseDem, config: BpTrellisConfig) -> Result<Self, DecoderError> {
        config.validate()?;
        let build_started = Instant::now();
        let mut trellis_config = config.trellis_config();
        trellis_config.column_order = config.ordering.resolve(dem)?;
        let inner = TrellisDecoder::from_sparse_dem(dem, trellis_config)?;
        let escalation = config.escalation;
        let build_seconds = build_started.elapsed().as_secs_f64();
        Ok(Self {
            inner,
            escalation,
            has_wide_observables: dem.num_observables > u64::BITS as usize,
            build_seconds,
        })
    }

    /// Parse a Stim-format detector error model and construct a decoder.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if parsing, ordering generation, or decoder
    /// validation fails.
    pub fn from_dem_str(dem_str: &str, config: BpTrellisConfig) -> Result<Self, DecoderError> {
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
    ) -> Result<Vec<Result<TrellisResult, DecoderError>>, DecoderError> {
        pecos_trellis::batch::decode_batch(shots, workers, || self.fresh_worker(), Self::decode)
    }

    fn fresh_worker(&self) -> Self {
        Self {
            inner: self.inner.fresh_worker(),
            escalation: self.escalation.clone(),
            has_wide_observables: self.has_wide_observables,
            build_seconds: self.build_seconds,
        }
    }

    /// Decode dense shots in input order, retaining no-path reports per shot.
    ///
    /// # Errors
    /// Returns an error for zero workers or pool creation failure. Shot errors
    /// remain in input order, separately from no-path reports.
    pub fn decode_batch_outcomes(
        &self,
        shots: &[Vec<u8>],
        workers: usize,
    ) -> Result<Vec<Result<BpTrellisOutcome, DecoderError>>, DecoderError> {
        pecos_trellis::batch::decode_batch(
            shots,
            workers,
            || self.fresh_worker(),
            Self::decode_outcome,
        )
    }

    /// Decode a shot, retrying only no-path attempts that dropped states.
    ///
    /// # Errors
    /// Returns dimension and engine errors. A no-path is an outcome, never an error.
    pub fn decode_outcome(&mut self, syndrome: &[u8]) -> Result<BpTrellisOutcome, DecoderError> {
        self.decode_with_attempt(syndrome, TrellisDecoder::attempt)
    }

    // The callable keeps attempt-count instrumentation in tests, without adding
    // a counter to either decoder or changing the public result contract.
    fn decode_with_attempt(
        &mut self,
        syndrome: &[u8],
        mut attempt: impl FnMut(&mut TrellisDecoder, PruneParams) -> TrellisDecodeAttempt,
    ) -> Result<BpTrellisOutcome, DecoderError> {
        let refreshes = self.inner.bp_refreshes();
        let prepared = self.inner.prepare(syndrome)?;
        let mut report = NoPathReport {
            cause: NoPathCause::Exhausted,
            placeholder: self.inner.forced_observables(),
            rungs_tried: 0,
            transitions: 0,
            bp_runs: u32::try_from(self.inner.bp_refreshes() - refreshes)
                .expect("one shot's BP refreshes fit u32"),
            bp_seconds: 0.0,
        };
        match prepared {
            TrellisPrepared::Residual { detector } => {
                report.cause = NoPathCause::Residual { detector };
                return Ok(BpTrellisOutcome::NoPath(report));
            }
            TrellisPrepared::Ready { bp_seconds, .. } => report.bp_seconds = bp_seconds,
        }
        let params =
            std::iter::once(self.inner.prune_params()).chain(self.escalation.iter().map(|rung| {
                PruneParams {
                    k: rung.k,
                    delta: rung.delta,
                }
            }));
        for (index, params) in params.enumerate() {
            report.rungs_tried = u32::try_from(index).expect("validated ladder length");
            let outcome = attempt(&mut self.inner, params);
            report.bp_runs = u32::try_from(self.inner.bp_refreshes() - refreshes)
                .expect("one shot's BP refreshes fit u32");
            match outcome {
                TrellisDecodeAttempt::Success(mut result) => {
                    result.transitions += report.transitions;
                    result.bp_seconds = report.bp_seconds;
                    result.bp_runs = report.bp_runs;
                    result.escalation_rungs_used = report.rungs_tried;
                    return Ok(BpTrellisOutcome::Decoded(result));
                }
                TrellisDecodeAttempt::NoPath {
                    transitions,
                    dropped_states,
                    ..
                } => {
                    report.transitions += transitions;
                    if dropped_states == 0 {
                        report.cause = NoPathCause::Infeasible;
                        break;
                    }
                }
                TrellisDecodeAttempt::Error(error) => return Err(error),
            }
        }
        Ok(BpTrellisOutcome::NoPath(report))
    }

    /// Decode a shot, returning an error with its cause when no path exists.
    ///
    /// # Errors
    /// Returns dimension and engine errors, or `DecodingFailed` on no-path.
    pub fn decode(&mut self, syndrome: &[u8]) -> Result<TrellisResult, DecoderError> {
        match self.decode_outcome(syndrome)? {
            BpTrellisOutcome::Decoded(result) => Ok(result),
            BpTrellisOutcome::NoPath(report) => Err(report.into_error()),
        }
    }

    /// Total wall-clock seconds spent constructing the shared model.
    #[must_use]
    pub fn build_seconds(&self) -> f64 {
        self.build_seconds
    }
}

impl ObservableDecoder for BpTrellisDecoder {
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        Ok(self.decode(syndrome)?.predicted)
    }

    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        if self.has_wide_observables {
            return Err(DecoderError::InvalidConfiguration(
                "decoder has more than 64 observables; use decode_obs() for the wide mask".into(),
            ));
        }
        let decoded = self.decode(syndrome)?.predicted;
        Ok(decoded.words().first().copied().unwrap_or(0))
    }
}

#[cfg(test)]
mod outcome_tests;
