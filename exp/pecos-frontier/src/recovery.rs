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

//! Opt-in recovery around Frontier. The underlying decoder's defaults are unchanged.
//!
//! # Example
//!
//! ```
//! use pecos_frontier::{FrontierConfig, SparseDem};
//! use pecos_frontier::recovery::{
//!     FallbackPolicy, FrontierRecoveryDecoder, FrontierRecoveryOutcome,
//!     NoPathAction, RecoveryOptions,
//! };
//!
//! let dem = SparseDem::from_dem_str("error(0.1) D0 D1 L0")?;
//! let mut decoder = FrontierRecoveryDecoder::from_sparse_dem(
//!     &dem,
//!     FrontierConfig { k: 256, delta: 16.0, ..Default::default() },
//!     Some(FallbackPolicy::FeasibleThenZero),
//! )?;
//! let outcome = decoder.decode(&[1, 0], RecoveryOptions {
//!     delta_max: Some(64.0),
//!     on_no_path: NoPathAction::Fallback,
//! })?;
//! assert!(matches!(outcome, FrontierRecoveryOutcome::Fallback {
//!     syndrome_consistent: Some(false), ..
//! }));
//! # Ok::<(), pecos_frontier::DecoderError>(())
//! ```

use crate::{FactorModel, FrontierConfig, FrontierDecoder, FrontierResult};
use pecos_decoder_core::{
    DecoderError, dem::SparseDem, feasibility::DemFeasibility, obs_mask::ObsMask,
};
pub use pecos_trellis::bp_trellis::NoPathCause;
use pecos_trellis::{PruneParams, TrellisDecodeAttempt, TrellisPrepared};
use std::sync::Arc;

/// What to do when configured attempts retain no path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NoPathAction {
    /// Preserve Frontier's strict error behavior.
    #[default]
    Raise,
    /// Return a no-path report with no prediction.
    Report,
    /// Return a prediction from the explicitly configured fallback.
    Fallback,
}

/// Explicit last-resort prediction policy, never enabled by default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackPolicy {
    /// Predict no observable flips, without claiming syndrome consistency.
    Zero,
    /// Solve in a probability-ordered GF(2) basis of a binary DEM; if impossible,
    /// return zero observable flips with `syndrome_consistent=false`.
    FeasibleThenZero,
}

/// Per-call recovery options, also used for every shot in a batch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RecoveryOptions {
    /// Double delta after a pruned no-path, capped at this finite value.
    /// Zero grows first to min(1, max); K is unchanged. None disables retries.
    pub delta_max: Option<f64>,
    /// Strict, reporting, or explicitly configured fallback behavior.
    pub on_no_path: NoPathAction,
}

impl RecoveryOptions {
    /// Validate against the configured base delta and fallback policy.
    ///
    /// # Errors
    /// Rejects a non-finite or below-base maximum and an unconfigured fallback.
    pub fn validate(
        self,
        base_delta: f64,
        fallback: Option<FallbackPolicy>,
    ) -> Result<(), DecoderError> {
        if let Some(max) = self.delta_max
            && (!max.is_finite() || max < base_delta)
        {
            return Err(DecoderError::InvalidConfiguration(format!(
                "delta_max {max:?} must be finite and at least the base delta {base_delta:?}"
            )));
        }
        if self.on_no_path == NoPathAction::Fallback && fallback.is_none() {
            return Err(DecoderError::InvalidConfiguration(
                "on_no_path='fallback' requires a configured fallback".into(),
            ));
        }
        Ok(())
    }
}

/// Diagnostics retained after all attempts fail.
#[derive(Clone, Debug)]
pub struct FrontierNoPath {
    /// Residual/infeasible is a proof; exhausted means pruning may be responsible.
    pub cause: NoPathCause,
    /// Actual DP deltas attempted, empty for an impossible residual precheck.
    pub deltas_tried: Vec<f64>,
    /// Total candidate transitions across attempts.
    pub transitions: u64,
    /// Total states discarded across attempts.
    pub dropped_states: u64,
    /// BP preparation time, counted only once.
    pub bp_seconds: f64,
}

/// Distinct successful, undecoded, and fallback outcomes.
#[derive(Debug)]
pub enum FrontierRecoveryOutcome {
    /// A normal Frontier result. Transitions include all attempted deltas;
    /// evidence and pruning status describe the successful attempt only.
    Decoded {
        /// Successful result.
        result: FrontierResult,
        /// Actual DP deltas attempted.
        deltas_tried: Vec<f64>,
    },
    /// No prediction was produced.
    NoPath(FrontierNoPath),
    /// An explicitly marked fallback prediction, without fabricated confidence.
    Fallback {
        /// Predicted observable flips.
        predicted: ObsMask,
        /// Policy selected by the caller (`FeasibleThenZero` may end in a zero sentinel).
        policy: FallbackPolicy,
        /// True for a verified explanation, false for a proved impossible DEM,
        /// None for the unverified Zero policy.
        syndrome_consistent: Option<bool>,
        /// Original mechanism indices of a verified explanation, if available.
        mechanisms: Option<Vec<usize>>,
        /// The original Frontier no-path diagnostics.
        no_path: FrontierNoPath,
    },
}

/// Shared-model Frontier facade with opt-in bounded retries and fallback.
/// Binary feasibility preprocessing is prepared once and shared across workers.
#[derive(Clone, Debug)]
pub struct FrontierRecoveryDecoder {
    inner: FrontierDecoder,
    num_detectors: usize,
    fallback: Option<FallbackPolicy>,
    feasibility: Option<Arc<DemFeasibility>>,
}

impl FrontierRecoveryDecoder {
    /// Construct from a binary DEM, optionally preparing a fallback.
    ///
    /// # Errors
    /// Propagates model/configuration or feasibility preparation errors.
    pub fn from_sparse_dem(
        dem: &SparseDem,
        config: FrontierConfig,
        fallback: Option<FallbackPolicy>,
    ) -> Result<Self, DecoderError> {
        let inner = FrontierDecoder::from_sparse_dem(dem, config)?;
        let feasibility = if fallback == Some(FallbackPolicy::FeasibleThenZero) {
            Some(Arc::new(DemFeasibility::new(dem)?))
        } else {
            None
        };
        Ok(Self {
            inner,
            num_detectors: dem.num_detectors,
            fallback,
            feasibility,
        })
    }

    /// Construct from categorical factors. Only the Zero fallback is supported;
    /// GF(2) columns cannot enforce mutually exclusive categorical outcomes.
    ///
    /// # Errors
    /// Rejects the feasibility fallback or invalid model/configuration.
    pub fn from_factor_model(
        model: &FactorModel,
        config: FrontierConfig,
        fallback: Option<FallbackPolicy>,
    ) -> Result<Self, DecoderError> {
        if fallback == Some(FallbackPolicy::FeasibleThenZero) {
            return Err(DecoderError::InvalidConfiguration(
                "feasible fallback requires an independent binary DEM".into(),
            ));
        }
        Ok(Self {
            inner: FrontierDecoder::from_factor_model(model, config)?,
            num_detectors: model.num_detectors(),
            fallback,
            feasibility: None,
        })
    }

    /// Bind recovery options for use through the shared observable-decoder API.
    ///
    /// # Errors
    /// Rejects invalid recovery options before any shots are decoded.
    pub fn with_options(
        self,
        options: RecoveryOptions,
    ) -> Result<ConfiguredFrontierDecoder, DecoderError> {
        self.validate_options(options)?;
        Ok(ConfiguredFrontierDecoder {
            decoder: self,
            options,
        })
    }

    /// Frontier model construction time; excludes optional fallback preparation.
    #[must_use]
    pub fn build_seconds(&self) -> f64 {
        self.inner.build_seconds()
    }

    /// Validate recovery options even for an empty batch.
    ///
    /// # Errors
    /// Rejects a non-finite maximum, a maximum below the base delta, or a
    /// requested fallback that was not configured at construction.
    pub fn validate_options(&self, options: RecoveryOptions) -> Result<(), DecoderError> {
        options.validate(self.inner.prune_params().delta, self.fallback)
    }

    /// Decode a single shot with one BP preparation and fresh DP attempts.
    /// Retries stop on success or proof of incompatibility. No failure policy
    /// suppresses malformed-input or internal errors.
    ///
    /// # Errors
    /// Propagates invalid inputs/configuration and engine errors; Raise also
    /// returns a decoding error when no configured attempt succeeds.
    pub fn decode(
        &mut self,
        syndrome: &[u8],
        options: RecoveryOptions,
    ) -> Result<FrontierRecoveryOutcome, DecoderError> {
        self.decode_with_attempt(syndrome, options, FrontierDecoder::attempt)
    }

    fn decode_with_attempt(
        &mut self,
        syndrome: &[u8],
        options: RecoveryOptions,
        mut attempt: impl FnMut(&mut FrontierDecoder, PruneParams) -> TrellisDecodeAttempt,
    ) -> Result<FrontierRecoveryOutcome, DecoderError> {
        self.validate_options(options)?;
        let mut report = FrontierNoPath {
            cause: NoPathCause::Exhausted,
            deltas_tried: Vec::new(),
            transitions: 0,
            dropped_states: 0,
            bp_seconds: 0.0,
        };
        let bp_ran = match self.inner.prepare(syndrome)? {
            TrellisPrepared::Residual { detector } => {
                report.cause = NoPathCause::Residual { detector };
                return self.finish_no_path(syndrome, options, report);
            }
            TrellisPrepared::Ready { bp_ran, bp_seconds } => {
                report.bp_seconds = bp_seconds;
                bp_ran
            }
        };
        let mut params = self.inner.prune_params();
        loop {
            report.deltas_tried.push(params.delta);
            match attempt(&mut self.inner, params) {
                TrellisDecodeAttempt::Success(mut result) => {
                    result.transitions += report.transitions;
                    result.bp_seconds = report.bp_seconds;
                    result.bp_runs = u32::from(bp_ran);
                    result.escalation_rungs_used = u32::try_from(report.deltas_tried.len() - 1)
                        .map_err(|_| {
                            DecoderError::InternalError("delta retry count overflow".into())
                        })?;
                    return Ok(FrontierRecoveryOutcome::Decoded {
                        result,
                        deltas_tried: report.deltas_tried,
                    });
                }
                TrellisDecodeAttempt::Error(error) => return Err(error),
                TrellisDecodeAttempt::NoPath {
                    transitions,
                    dropped_states,
                    ..
                } => {
                    report.transitions += transitions;
                    report.dropped_states += dropped_states;
                    if dropped_states == 0 {
                        report.cause = NoPathCause::Infeasible;
                        break;
                    }
                }
            }
            let Some(max) = options.delta_max else {
                break;
            };
            if params.delta >= max {
                break;
            }
            params = PruneParams {
                k: params.k,
                delta: if params.delta == 0.0 {
                    1.0_f64.min(max)
                } else {
                    (params.delta * 2.0).min(max)
                },
            };
        }
        self.finish_no_path(syndrome, options, report)
    }

    fn finish_no_path(
        &self,
        syndrome: &[u8],
        options: RecoveryOptions,
        report: FrontierNoPath,
    ) -> Result<FrontierRecoveryOutcome, DecoderError> {
        match options.on_no_path {
            NoPathAction::Raise => Err(DecoderError::DecodingFailed(
                "syndrome is unexplainable at the given pruning parameters".into(),
            )),
            NoPathAction::Report => Ok(FrontierRecoveryOutcome::NoPath(report)),
            NoPathAction::Fallback => {
                let policy = self.fallback.expect("validated fallback");
                let explanation = self
                    .feasibility
                    .as_ref()
                    .map(|prepared| prepared.solve(syndrome))
                    .transpose()?
                    .flatten();
                let syndrome_consistent =
                    (policy == FallbackPolicy::FeasibleThenZero).then_some(explanation.is_some());
                let (predicted, mechanisms) = explanation.map_or_else(
                    || (ObsMask::new(), None),
                    |value| (value.observables, Some(value.mechanisms)),
                );
                Ok(FrontierRecoveryOutcome::Fallback {
                    predicted,
                    policy,
                    syndrome_consistent,
                    mechanisms,
                    no_path: report,
                })
            }
        }
    }

    /// Share immutable Frontier and feasibility models with fresh worker scratch.
    #[must_use]
    pub fn fresh_worker(&self) -> Self {
        Self {
            inner: self.inner.fresh_worker(),
            num_detectors: self.num_detectors,
            fallback: self.fallback,
            feasibility: self.feasibility.clone(),
        }
    }

    /// Decode in input order with independent scratch and shared prepared models.
    /// Per-shot errors are retained; global option/pool errors abort the call.
    ///
    /// # Errors
    /// Rejects invalid recovery options/workers or a failed worker pool.
    pub fn decode_batch(
        &self,
        shots: &[Vec<u8>],
        workers: usize,
        options: RecoveryOptions,
    ) -> Result<Vec<Result<FrontierRecoveryOutcome, DecoderError>>, DecoderError> {
        self.validate_options(options)?;
        pecos_trellis::batch::decode_batch(
            shots,
            workers,
            || self.fresh_worker(),
            |decoder, shot| decoder.decode(shot, options),
        )
    }
}

/// Frontier with validated recovery options bound to every observable decode call.
/// Use this adapter from a shared batch `DecoderFactory`; native batch dispatch
/// must remain disabled so per-shot outcomes are preserved.
#[derive(Clone, Debug)]
pub struct ConfiguredFrontierDecoder {
    decoder: FrontierRecoveryDecoder,
    options: RecoveryOptions,
}
impl ConfiguredFrontierDecoder {
    /// Share prepared models and options with fresh worker scratch.
    #[must_use]
    pub fn fresh_worker(&self) -> Self {
        Self {
            decoder: self.decoder.fresh_worker(),
            options: self.options,
        }
    }

    // Observable callers consume only the predicted mask, so the attempts skip
    // dropped-mass accounting; recovery decisions are identical to full attempts.
    fn decode_prediction(
        &mut self,
        syndrome: &[u8],
    ) -> Result<FrontierRecoveryOutcome, DecoderError> {
        self.decoder.decode_with_attempt(
            syndrome,
            self.options,
            FrontierDecoder::attempt_prediction_only,
        )
    }
}
impl pecos_decoder_core::ObservableDecoder for ConfiguredFrontierDecoder {
    fn num_detectors(&self) -> Option<usize> {
        Some(self.decoder.num_detectors)
    }
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        self.decode_outcome(syndrome)?.prediction.ok_or_else(|| {
            DecoderError::DecodingFailed(
                "no prediction; use decode_outcome() for reporting outcomes".into(),
            )
        })
    }
    fn decode_outcome(
        &mut self,
        syndrome: &[u8],
    ) -> Result<pecos_decoder_core::outcome::ObservableDecodeOutcome, DecoderError> {
        use pecos_decoder_core::outcome::ObservableDecodeOutcome;
        Ok(match self.decode_prediction(syndrome)? {
            FrontierRecoveryOutcome::Decoded {
                result,
                deltas_tried,
            } => ObservableDecodeOutcome {
                prediction: Some(result.predicted),
                fallback_used: false,
                syndrome_consistent: None,
                deltas_tried,
            },
            FrontierRecoveryOutcome::NoPath(report) => ObservableDecodeOutcome {
                prediction: None,
                fallback_used: false,
                syndrome_consistent: None,
                deltas_tried: report.deltas_tried,
            },
            FrontierRecoveryOutcome::Fallback {
                predicted,
                syndrome_consistent,
                no_path,
                ..
            } => ObservableDecodeOutcome {
                prediction: Some(predicted),
                fallback_used: true,
                syndrome_consistent,
                deltas_tried: no_path.deltas_tried,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEM: &str = "error(0.4) D0\nerror(0.4) D1\nerror(0.1) D0 D1 D2 L70";

    #[test]
    fn bp_refreshes_once_across_retries() {
        let dem = SparseDem::from_dem_str(DEM).unwrap();
        let mut decoder = FrontierRecoveryDecoder::from_sparse_dem(
            &dem,
            FrontierConfig {
                k: 8,
                delta: 0.0,
                score_alpha: 0.0,
                column_order: Some(vec![0, 1, 2]),
                bp_score_iterations: 2,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        for _ in 0..2 {
            let before = decoder.inner.bp_refreshes();
            let FrontierRecoveryOutcome::Decoded { deltas_tried, .. } = decoder
                .decode(
                    &[0, 0, 1],
                    RecoveryOptions {
                        delta_max: Some(32.0),
                        on_no_path: NoPathAction::Raise,
                    },
                )
                .unwrap()
            else {
                panic!("retry must succeed");
            };
            assert!(deltas_tried.len() > 1);
            assert_eq!(decoder.inner.bp_refreshes() - before, 1);
        }
    }

    #[test]
    fn observable_adapter_decodes_prediction_only() {
        let dem = SparseDem::from_dem_str(DEM).unwrap();
        let config = FrontierConfig {
            k: 8,
            delta: 0.0,
            score_alpha: 0.0,
            column_order: Some(vec![0, 1, 2]),
            ..Default::default()
        };
        let options = RecoveryOptions {
            delta_max: Some(32.0),
            on_no_path: NoPathAction::Raise,
        };
        let mut full = FrontierRecoveryDecoder::from_sparse_dem(&dem, config, None).unwrap();
        let mut adapter = full.fresh_worker().with_options(options).unwrap();
        for shot in [[0, 0, 0], [0, 0, 1]] {
            let FrontierRecoveryOutcome::Decoded { result: full, .. } =
                full.decode(&shot, options).unwrap()
            else {
                panic!("full decode must succeed");
            };
            let FrontierRecoveryOutcome::Decoded { result, .. } =
                adapter.decode_prediction(&shot).unwrap()
            else {
                panic!("adapter decode must succeed");
            };
            // Prediction-only success marks the skipped accounting as NaN.
            assert!(!full.dropped_log_mass.is_nan());
            assert!(result.dropped_log_mass.is_nan());
            assert_eq!(result.predicted, full.predicted);
        }
    }

    #[test]
    fn fresh_worker_shares_feasibility_and_decodes_identically() {
        let dem = SparseDem::from_dem_str(DEM).unwrap();
        let mut decoder = FrontierRecoveryDecoder::from_sparse_dem(
            &dem,
            FrontierConfig {
                k: 1,
                delta: 0.0,
                score_alpha: 0.0,
                column_order: Some(vec![0, 1, 2]),
                ..Default::default()
            },
            Some(FallbackPolicy::FeasibleThenZero),
        )
        .unwrap();
        let mut worker = decoder.fresh_worker();
        assert!(Arc::ptr_eq(
            decoder.feasibility.as_ref().unwrap(),
            worker.feasibility.as_ref().unwrap(),
        ));
        let options = RecoveryOptions {
            delta_max: Some(3.0),
            on_no_path: NoPathAction::Fallback,
        };
        for shot in [[0, 0, 0], [0, 0, 1]] {
            let expected = decoder.decode(&shot, options).unwrap();
            let actual = worker.decode(&shot, options).unwrap();
            // BP is disabled, so these results contain no nondeterministic timing.
            assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
        }
    }
}
