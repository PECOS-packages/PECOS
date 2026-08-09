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
//! port of an external project.
//!
//! The facade owns [`TrellisDecoder`] instances configured with PECOS's
//! defaults, ordering semantics, and optional no-path escalation ladder. The
//! trellis engine lives in `pecos-trellis`.

use pecos_decoder_core::ObservableDecoder;
use pecos_trellis::{
    DecoderError, ObsMask, SparseDem, TrellisConfig, TrellisDecodeAttempt, TrellisDecoder,
    TrellisResult, backward_deadline_column_order, deadline_column_order,
};
use std::time::Instant;

/// Processing order used by [`BpTrellisDecoder`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum TrellisOrdering {
    /// Compute the deadline-optimized order with [`deadline_column_order`].
    #[default]
    Deadline,
    /// Compute the backward deadline-optimized order with
    /// [`backward_deadline_column_order`].
    BackwardDeadline,
    /// Preserve the detector error model's mechanism order.
    TimeOrder,
    /// Use an explicit permutation mapping target positions to source
    /// mechanism indices.
    Explicit(Vec<usize>),
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
    /// Each entry pre-builds another decoder with that `k` and otherwise the
    /// same configuration. Construction cost therefore grows with the number
    /// of rungs. An empty ladder disables escalation and is the explicit
    /// default because escalation changes per-shot work; whether a future
    /// default should enable a ladder is deferred to the evaluation campaign.
    pub escalation_ks: Vec<usize>,
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
            escalation_ks: Vec::new(),
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
    escalation: Vec<TrellisDecoder>,
    has_wide_observables: bool,
    build_seconds: f64,
}

impl BpTrellisDecoder {
    /// Construct a decoder from a sparse detector error model.
    ///
    /// Unlike [`TrellisDecoder`], the default ordering is the explicitly
    /// computed [`deadline_column_order`], not input order. Every configured
    /// escalation rung is constructed here as an independent
    /// [`TrellisDecoder`], so construction cost scales with the full ladder
    /// and decode-time escalation performs no model building.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if ordering generation or the mapped trellis
    /// configuration fails validation.
    pub fn from_sparse_dem(dem: &SparseDem, config: BpTrellisConfig) -> Result<Self, DecoderError> {
        let build_started = Instant::now();
        let BpTrellisConfig {
            k,
            delta,
            score_alpha,
            bp_score_iterations,
            merge_indistinguishable,
            ordering,
            escalation_ks,
        } = config;
        if u32::try_from(escalation_ks.len()).is_err() {
            return Err(DecoderError::InvalidConfiguration(
                "escalation ladder has more rungs than escalation_rungs_used can represent".into(),
            ));
        }
        let column_order = match ordering {
            TrellisOrdering::Deadline => Some(deadline_column_order(dem)?),
            TrellisOrdering::BackwardDeadline => Some(backward_deadline_column_order(dem)?),
            TrellisOrdering::TimeOrder => None,
            TrellisOrdering::Explicit(order) => Some(order),
        };
        let trellis_config = TrellisConfig {
            k,
            delta,
            score_alpha,
            column_order,
            merge_indistinguishable,
            bp_score_iterations,
        };
        let inner = TrellisDecoder::from_sparse_dem(dem, trellis_config.clone())?;
        let escalation = escalation_ks
            .into_iter()
            .map(|rung_k| {
                TrellisDecoder::from_sparse_dem(
                    dem,
                    TrellisConfig {
                        k: rung_k,
                        ..trellis_config.clone()
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
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

    /// Decode a dense detector syndrome with the shared trellis engine.
    ///
    /// Every nonzero byte is treated as a fired detector. A no-path base
    /// attempt is retried at each pre-built escalation rung until one
    /// succeeds. Any returned prediction, including a pruned or incorrect
    /// prediction, is final because production decoding has no truth oracle.
    /// Other errors do not escalate. On success, `transitions` and
    /// `bp_seconds` cover the full attempted sequence; pruning telemetry and
    /// status come from the successful rung.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] for a dimension mismatch or when the syndrome
    /// is unexplainable with the retained frontier.
    pub fn decode(&mut self, syndrome: &[u8]) -> Result<TrellisResult, DecoderError> {
        let (mut final_error, mut transitions, mut bp_seconds) =
            match self.inner.decode_attempt(syndrome) {
                TrellisDecodeAttempt::Success(result) => return Ok(result),
                TrellisDecodeAttempt::NoPath {
                    error,
                    transitions,
                    bp_seconds,
                } => (error, transitions, bp_seconds),
                TrellisDecodeAttempt::Error(error) => return Err(error),
            };

        let mut escalation_rungs_used = 0_u32;
        for decoder in &mut self.escalation {
            escalation_rungs_used = escalation_rungs_used.saturating_add(1);
            match decoder.decode_attempt(syndrome) {
                TrellisDecodeAttempt::Success(mut result) => {
                    result.transitions += transitions;
                    result.bp_seconds += bp_seconds;
                    result.escalation_rungs_used = escalation_rungs_used;
                    return Ok(result);
                }
                TrellisDecodeAttempt::NoPath {
                    error,
                    transitions: rung_transitions,
                    bp_seconds: rung_bp_seconds,
                } => {
                    final_error = error;
                    transitions += rung_transitions;
                    bp_seconds += rung_bp_seconds;
                }
                TrellisDecodeAttempt::Error(error) => return Err(error),
            }
        }

        Err(final_error)
    }

    /// Total wall-clock seconds spent constructing the base decoder and every
    /// escalation-rung model.
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
