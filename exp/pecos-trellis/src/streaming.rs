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

//! Incremental binary float decoding with zero-regret logical commitments.

use crate::{
    BinaryFailure, BinaryProgress, DecoderError, Kernel, MetricMode, ObsMask, SparseDem,
    TrellisConfig, TrellisDecoder, TrellisModel, TrellisResult, WORD_BITS, factor::FactorModel,
    set_bit, set_bits,
};
use std::sync::Arc;

/// Scheduling policy. Dependency-driven is the backward-compatible default.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum StreamingMode {
    #[default]
    DependencyDriven,
    /// ZIP-style persistent streaming, with `window_size - 1` rounds of lag.
    /// Bounds include the terminal detector block, not only extraction rounds.
    FixedLag {
        detector_round_bounds: Vec<usize>,
        window_size: usize,
    },
}

/// Snapshot after processing all currently ready columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamingProgress {
    /// Total number of columns processed in this shot.
    pub columns_processed: usize,
    /// Commitments first reported by this advance, including construction/reset commitments.
    pub newly_committed: Vec<(usize, bool)>,
    /// Values of committed logical bits, using [`pecos_decoder_core::obs_mask::ObsMask`].
    /// Consult `committed_mask` to distinguish an uncommitted bit from a committed zero.
    pub committed: ObsMask,
    /// Bit j is set exactly when logical j has been committed.
    pub committed_mask: ObsMask,
}

#[derive(Debug)]
struct ScoringPrefixPlan {
    known_count: usize,
    start: usize,
    ends: Vec<usize>,
}

impl ScoringPrefixPlan {
    fn for_range(&self, known_count: usize, range: std::ops::Range<usize>) -> Option<&[usize]> {
        if known_count != self.known_count {
            return None;
        }
        let start = range.start.checked_sub(self.start)?;
        let end = range.end.checked_sub(self.start)?;
        self.ends.get(start..end)
    }
}

/// Opt-in streaming decoder for detector blocks arriving in index order.
///
/// In the default dependency-driven mode, a successful flush is bit-identical to batch
/// [`TrellisDecoder::decode`], including result telemetry. Every committed logical
/// bit equals the corresponding bit of that final prediction. Commitment requires
/// unanimity of the retained frontier and no remaining column toggling that bit.
///
/// Streaming v1 requires the binary float kernel and `bp_score_iterations == 0`:
/// BP-guided scoring needs the whole syndrome before the column walk. Failures
/// persist until [`Self::reset`]. Flushing does not reset the shot.
#[derive(Debug)]
pub struct TrellisStreamingDecoder {
    model: Arc<TrellisModel>,
    /// Prefix maximum of the highest detector in each close/active mask union.
    /// Columns execute in order, so earlier requirements remain prerequisites.
    /// Use the masks, not touch order: probability-one mechanisms are folded into
    /// the initial forced syndrome, making future touched detectors active early.
    ready: Vec<Option<usize>>,
    lookahead: Vec<usize>,
    last_toggle_column: Vec<Option<usize>>,
    progress: BinaryProgress,
    observed: Vec<u64>,
    arrived_count: usize,
    next_column: usize,
    committed: Vec<u64>,
    committed_mask: Vec<u64>,
    pending_commitments: Vec<(usize, bool)>,
    failure: Option<BinaryFailure>,
    mode: StreamingMode,
    column_rounds: Vec<usize>,
    scoring_prefix_plans: Vec<ScoringPrefixPlan>,
    arrived_round: usize,
    evidence_omitted: bool,
}

impl TrellisStreamingDecoder {
    /// Construct a stream using the batch decoder's model construction.
    ///
    /// # Errors
    /// Returns `InvalidConfiguration` for BP scoring, non-float metrics, or invalid models.
    pub fn from_sparse_dem(dem: &SparseDem, config: TrellisConfig) -> Result<Self, DecoderError> {
        validate_streaming_config(&config)?;
        let decoder = TrellisDecoder::from_sparse_dem(dem, config)?;
        Ok(Self::from_binary_decoder(decoder, dem.num_observables))
    }

    /// Parse a Stim-format DEM and construct a stream.
    ///
    /// # Errors
    /// Returns parsing errors or the configuration errors of [`Self::from_sparse_dem`].
    pub fn from_dem_str(dem_str: &str, config: TrellisConfig) -> Result<Self, DecoderError> {
        Self::from_sparse_dem(&SparseDem::from_dem_str(dem_str)?, config)
    }

    /// Construct a stream from a binary-shaped factor model.
    ///
    /// # Errors
    /// Returns `InvalidConfiguration` for genuinely N-ary models, BP scoring,
    /// non-float metrics, or invalid configuration.
    pub fn from_factor_model(
        model: &FactorModel,
        config: TrellisConfig,
    ) -> Result<Self, DecoderError> {
        validate_streaming_config(&config)?;
        let decoder = TrellisDecoder::from_factor_model(model, config)?;
        if !matches!(decoder.model.kernel, Kernel::Binary(_)) {
            return Err(binary_kernel_error());
        }
        Ok(Self::from_binary_decoder(decoder, model.num_observables()))
    }

    fn from_binary_decoder(decoder: TrellisDecoder, num_observables: usize) -> Self {
        let model = decoder.model;
        let Kernel::Binary(columns) = &model.kernel else {
            unreachable!("streaming constructors validate the binary kernel");
        };
        let mut ready = Vec::with_capacity(columns.len());
        let mut lookahead = Vec::with_capacity(columns.len());
        let mut last_toggle_column = vec![None; num_observables];
        let mut prefix_requirement = None;
        for (index, column) in columns.iter().enumerate() {
            let required = set_bits(&column.close_mask)
                .chain(set_bits(&column.active_mask))
                .max();
            prefix_requirement = prefix_requirement.max(required);
            ready.push(prefix_requirement);
            // Adding one expresses None as -1 without signed index conversions.
            let own_detector_count = set_bits(&column.detector_toggle).max().map_or(0, |d| d + 1);
            lookahead.push(
                prefix_requirement
                    .map_or(0, |d| d + 1)
                    .checked_sub(own_detector_count)
                    .expect("toggled detectors are a subset of close | active rows"),
            );
            for logical in set_bits(&column.logical_toggle) {
                last_toggle_column[logical] = Some(index);
            }
        }
        let mut stream = Self {
            observed: vec![0; model.detector_words],
            committed: vec![0; model.logical_words],
            committed_mask: vec![0; model.logical_words],
            model,
            ready,
            lookahead,
            last_toggle_column,
            progress: decoder.scratch.float_progress,
            arrived_count: 0,
            next_column: 0,
            pending_commitments: Vec::new(),
            failure: None,
            mode: StreamingMode::DependencyDriven,
            column_rounds: Vec::new(),
            scoring_prefix_plans: Vec::new(),
            arrived_round: 0,
            evidence_omitted: false,
        };
        stream.reset();
        stream
    }

    /// Select a schedule before starting a shot. Cached compact model and
    /// prediction-only fast paths are reused. Fixed-lag changes finite-width
    /// pruning and is not promised to equal batch/dependency-driven decoding.
    ///
    /// # Errors
    /// Returns `InvalidConfiguration` after shot processing has started or for
    /// invalid bounds/window size. A failed selection preserves the old mode.
    pub fn set_streaming_mode(&mut self, mode: StreamingMode) -> Result<(), DecoderError> {
        if self.arrived_count != 0 || self.next_column != 0 || self.arrived_round != 0 {
            return Err(DecoderError::InvalidConfiguration(
                "reset before changing streaming mode".into(),
            ));
        }
        let mut requirements = Vec::new();
        let mut scoring_prefix_plans = Vec::new();
        if let StreamingMode::FixedLag {
            detector_round_bounds: bounds,
            window_size,
        } = &mode
        {
            if bounds.len() < 3
                || bounds.first() != Some(&0)
                || bounds.last() != Some(&self.model.num_detectors)
                || bounds.windows(2).any(|pair| pair[0] > pair[1])
                || *window_size < 2
                || *window_size >= bounds.len()
            {
                return Err(DecoderError::InvalidConfiguration(
                    "invalid fixed-lag bounds or window size".into(),
                ));
            }
            let Kernel::Binary(columns) = &self.model.kernel else {
                unreachable!("streaming constructors validate the binary kernel");
            };
            let row_round = |row: usize| bounds.partition_point(|&edge| edge <= row);
            let mut next_release = bounds.len() - 1;
            requirements.resize(columns.len(), next_release);
            for (index, column) in columns.iter().enumerate().rev() {
                if let Some(row) = set_bits(&column.detector_toggle).min() {
                    next_release = row_round(row);
                }
                requirements[index] = next_release;
            }
            let mut prefix = 0;
            for (required, column) in requirements.iter_mut().zip(columns) {
                let closing = set_bits(&column.close_mask)
                    .map(row_round)
                    .max()
                    .unwrap_or(0);
                prefix = prefix.max((*required).max(closing));
                *required = prefix;
            }
            // Cache only columns normally released at each arrival round. Each
            // column contributes one end index, with no per-shot preparation.
            let mut start = 0;
            for (round, &known_count) in bounds.iter().enumerate() {
                let target = if round == bounds.len() - 1 {
                    round
                } else {
                    round.saturating_sub(window_size - 1)
                };
                let end = requirements.partition_point(|&release| release <= target);
                let ends = columns[start..end]
                    .iter()
                    .map(|column| {
                        column.suffix_compatibility.partition_point(|row| {
                            row.word as usize * WORD_BITS + usize::from(row.bit) < known_count
                        })
                    })
                    .collect();
                scoring_prefix_plans.push(ScoringPrefixPlan {
                    known_count,
                    start,
                    ends,
                });
                start = end;
            }
            debug_assert_eq!(start, columns.len());
        }
        self.scoring_prefix_plans = scoring_prefix_plans;
        self.column_rounds = requirements;
        self.mode = mode;
        Ok(())
    }

    /// Per-column lookahead in detector indices: readiness minus the column's
    /// highest toggled detector. A missing detector index is interpreted as -1.
    /// This diagnostic describes the dependency-driven schedule only; fixed
    /// lag is specified separately by its round bounds and window size.
    #[must_use]
    pub fn column_lookahead(&self) -> &[usize] {
        &self.lookahead
    }

    /// Append the NEXT block of detector values. Every nonzero byte means fired.
    /// Only newly arrived detectors are checked against the forced contribution
    /// on rows untouched by probabilistic columns.
    /// In fixed-lag mode this must be exactly the next configured round,
    /// including an explicit empty slice for an empty detector round.
    ///
    /// # Errors
    /// Returns `InvalidDimensions` for overflow (without changing the shot), or
    /// the batch decoder's `DecodingFailed` error as soon as an inconsistent
    /// untouched detector arrives. A previous shot failure is returned again.
    pub fn feed_prefix(&mut self, detectors: &[u8]) -> Result<(), DecoderError> {
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        let total = self.arrived_count.saturating_add(detectors.len());
        if let StreamingMode::FixedLag {
            detector_round_bounds: bounds,
            ..
        } = &self.mode
            && (self.arrived_round + 1 >= bounds.len() || total != bounds[self.arrived_round + 1])
        {
            return Err(DecoderError::InvalidConfiguration(
                "fixed-lag feed_prefix must contain exactly the next detector round".into(),
            ));
        }
        if total > self.model.num_detectors {
            return Err(DecoderError::InvalidDimensions {
                expected: self.model.num_detectors,
                actual: total,
            });
        }
        self.append_detectors(detectors);
        if matches!(self.mode, StreamingMode::FixedLag { .. }) {
            self.arrived_round += 1;
        }
        self.failure.map_or(Ok(()), |failure| Err(failure.error()))
    }

    /// Record already validated detector values after the current prefix.
    fn append_detectors(&mut self, detectors: &[u8]) {
        for (offset, &value) in detectors.iter().enumerate() {
            let detector = self.arrived_count + offset;
            let word = detector / WORD_BITS;
            let mask = 1_u64 << (detector % WORD_BITS);
            let seen = if value == 0 { 0 } else { mask };
            self.observed[word] |= seen;
            if (seen ^ self.model.forced_syndrome[word])
                & !self.model.touched_detectors[word]
                & mask
                != 0
            {
                self.failure = Some(BinaryFailure::NoPath);
            }
        }
        self.arrived_count += detectors.len();
    }

    /// Feed the whole syndrome in one block before any detectors have arrived.
    ///
    /// In fixed-lag mode this delivers every configured round at once, so no
    /// column is processed with unknown outcomes and the decode matches batch.
    /// Fixed lag changes results only when rounds are fed incrementally.
    ///
    /// # Errors
    /// Returns a stored failure first, then `InvalidDimensions` for a syndrome
    /// whose length differs from the model's detector count, then the errors of
    /// [`Self::feed_prefix`]: `InvalidDimensions` reporting the cumulative count
    /// when any detectors have already arrived, and `DecodingFailed` for an
    /// inconsistent untouched detector. In fixed-lag mode, feeding after any
    /// round has arrived returns `InvalidConfiguration`.
    pub fn feed_dense(&mut self, syndrome: &[u8]) -> Result<(), DecoderError> {
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        if syndrome.len() != self.model.num_detectors {
            return Err(DecoderError::InvalidDimensions {
                expected: self.model.num_detectors,
                actual: syndrome.len(),
            });
        }
        if let StreamingMode::FixedLag {
            detector_round_bounds,
            ..
        } = &self.mode
        {
            if self.arrived_round != 0 {
                return Err(DecoderError::InvalidConfiguration(
                    "feed_dense requires a fresh shot".into(),
                ));
            }
            self.arrived_round = detector_round_bounds.len() - 1;
            self.append_detectors(syndrome);
            self.failure.map_or(Ok(()), |failure| Err(failure.error()))
        } else {
            self.feed_prefix(syndrome)
        }
    }

    /// Process all ready columns, then check uncommitted logicals for unanimity.
    /// Initial forced commitments are reported on the first successful advance.
    ///
    /// # Errors
    /// Returns the same no-path or internal error as the batch column walk.
    pub fn advance(&mut self) -> Result<StreamingProgress, DecoderError> {
        self.advance_prediction()?;
        self.check_commitments();
        let newly_committed = self.pending_commitments.clone();
        self.pending_commitments.clear();
        Ok(StreamingProgress {
            columns_processed: self.next_column,
            newly_committed,
            committed: ObsMask::from_words(&self.committed),
            committed_mask: ObsMask::from_words(&self.committed_mask),
        })
    }

    /// Process ready columns without constructing commitment snapshots.
    ///
    /// This path retains the same candidate set and correction as [`Self::advance`],
    /// including dropped-mass telemetry, so it can safely be mixed with `flush`.
    /// It is intended
    /// for latency-sensitive runtimes that need only the final prediction.
    ///
    /// # Errors
    /// Returns the same no-path or internal error as [`Self::advance`].
    pub fn advance_prediction(&mut self) -> Result<usize, DecoderError> {
        self.advance_prediction_mode(crate::DecodeMode::Full)
    }

    /// Prediction-only streaming without discarded-mass evidence telemetry.
    /// Candidate scoring and retention are unchanged. Do not use full-result
    /// evidence/diagnostics after this call; reset before returning to full mode.
    ///
    /// # Errors
    /// Returns the same no-path or internal failure as [`Self::advance_prediction`].
    pub fn advance_prediction_without_evidence(&mut self) -> Result<usize, DecoderError> {
        self.advance_prediction_mode(crate::DecodeMode::PredictionOnly)
    }

    fn advance_prediction_mode(&mut self, mode: crate::DecodeMode) -> Result<usize, DecoderError> {
        if mode == crate::DecodeMode::Full && self.evidence_omitted {
            return Err(DecoderError::InvalidConfiguration(
                "reset before requesting evidence after prediction-only streaming".into(),
            ));
        }
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        let (end, known_count) = match &self.mode {
            StreamingMode::DependencyDriven => (
                self.ready
                    .partition_point(|required| required.is_none_or(|d| d < self.arrived_count)),
                None,
            ),
            StreamingMode::FixedLag {
                detector_round_bounds,
                window_size,
            } => {
                let target = if self.arrived_round == detector_round_bounds.len() - 1 {
                    self.arrived_round
                } else {
                    self.arrived_round.saturating_sub(window_size - 1)
                };
                (
                    self.column_rounds.partition_point(|&round| round <= target),
                    Some(self.arrived_count),
                )
            }
        };
        self.evidence_omitted |= mode == crate::DecodeMode::PredictionOnly;
        // Delayed advances can include columns from earlier rounds. Use a plan
        // only when both the range and the known detector count match exactly.
        let prefix_ends = self
            .scoring_prefix_plans
            .get(self.arrived_round)
            .and_then(|plan| plan.for_range(self.arrived_count, self.next_column..end));
        if let Err(failure) = self.model.process_binary_range_with_prefixes(
            &mut self.progress,
            &self.observed,
            &self.model.suffix_values,
            self.next_column..end,
            crate::PruneParams {
                k: self.model.config.k,
                delta: self.model.config.delta,
            },
            crate::BinaryRangeOptions { mode, known_count },
            prefix_ends,
        ) {
            self.failure = Some(failure);
            return Err(failure.error());
        }
        self.next_column = end;
        Ok(self.next_column)
    }

    /// Current commitment values and mask, including commitments discovered by flush.
    #[must_use]
    pub fn committed(&self) -> (ObsMask, ObsMask) {
        (
            ObsMask::from_words(&self.committed),
            ObsMask::from_words(&self.committed_mask),
        )
    }

    /// Finish a fully fed shot using the batch terminal-result construction.
    ///
    /// Remaining commitments discovered here are visible through [`Self::committed`].
    ///
    /// # Errors
    /// Returns `InvalidDimensions` until all detectors arrive, `InvalidConfiguration`
    /// until every fixed-lag round arrives, or the same no-path or internal error
    /// as [`Self::advance`].
    ///
    /// # Panics
    /// Panics if any column remains unprocessed or a committed bit disagrees with
    /// the final prediction; both indicate an engine invariant violation.
    pub fn flush(&mut self) -> Result<TrellisResult, DecoderError> {
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }

        self.ensure_shot_complete()?;
        self.advance()?;
        let Kernel::Binary(columns) = &self.model.kernel else {
            unreachable!("streaming constructors validate the binary kernel");
        };
        assert_eq!(
            self.next_column,
            columns.len(),
            "flush must process every column"
        );
        let result = TrellisModel::finish_binary(&self.progress, self.next_column, 0.0);
        for logical in set_bits(&self.committed_mask) {
            let value = self.committed[logical / WORD_BITS] & (1_u64 << (logical % WORD_BITS)) != 0;
            assert!(
                result.predicted.get(logical) == value,
                "committed logical bit {logical} differs from the terminal prediction"
            );
        }
        Ok(result)
    }

    /// Finish a fully fed shot and return only the predicted logical mask.
    ///
    /// This skips terminal evidence, runner-up, commitment, and logical-mass
    /// construction. The prediction is identical to [`Self::flush`] on the same
    /// build and platform.
    ///
    /// # Errors
    /// Returns `InvalidDimensions` until all detectors arrive, `InvalidConfiguration`
    /// until every fixed-lag round arrives, or the same no-path or internal error
    /// as [`Self::advance_prediction`].
    ///
    /// # Panics
    /// Panics if the validated streaming model fails to process every column
    /// after the complete syndrome has arrived.
    pub fn flush_prediction(&mut self) -> Result<ObsMask, DecoderError> {
        self.flush_prediction_mode(crate::DecodeMode::Full)
    }

    /// Terminal correction only, with discarded-mass telemetry disabled.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::flush_prediction`], with no-path or
    /// internal failures from [`Self::advance_prediction_without_evidence`].
    pub fn flush_prediction_without_evidence(&mut self) -> Result<ObsMask, DecoderError> {
        self.flush_prediction_mode(crate::DecodeMode::PredictionOnly)
    }

    fn flush_prediction_mode(&mut self, mode: crate::DecodeMode) -> Result<ObsMask, DecoderError> {
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        self.ensure_shot_complete()?;
        self.advance_prediction_mode(mode)?;
        let Kernel::Binary(columns) = &self.model.kernel else {
            unreachable!("streaming constructors validate the binary kernel");
        };
        assert_eq!(
            self.next_column,
            columns.len(),
            "flush_prediction must process every column"
        );
        Ok(TrellisModel::finish_binary_prediction(&self.progress))
    }

    /// A shot is complete once every detector arrives. Fixed lag also requires
    /// every configured round, including trailing empty rounds, because its
    /// terminal release follows the final round.
    fn ensure_shot_complete(&self) -> Result<(), DecoderError> {
        if self.arrived_count != self.model.num_detectors {
            return Err(DecoderError::InvalidDimensions {
                expected: self.model.num_detectors,
                actual: self.arrived_count,
            });
        }
        if let StreamingMode::FixedLag {
            detector_round_bounds,
            ..
        } = &self.mode
            && self.arrived_round + 1 != detector_round_bounds.len()
        {
            return Err(DecoderError::InvalidConfiguration(
                "fixed-lag flush requires every configured round, including empty rounds".into(),
            ));
        }
        Ok(())
    }

    /// Begin another shot, reusing buffers and committing bits that never toggle.
    pub fn reset(&mut self) {
        self.arrived_count = 0;
        self.arrived_round = 0;
        self.next_column = 0;
        self.observed.fill(0);
        self.committed.fill(0);
        self.committed_mask.fill(0);
        self.pending_commitments.clear();
        self.failure = None;
        self.evidence_omitted = false;
        self.progress.reset(&self.model);
        self.check_commitments();
    }

    fn check_commitments(&mut self) {
        let frontier = &self.progress.frontier;
        for (logical, &last) in self.last_toggle_column.iter().enumerate() {
            let word = logical / WORD_BITS;
            let mask = 1_u64 << (logical % WORD_BITS);
            if self.committed_mask[word] & mask != 0
                || last.is_some_and(|column| column >= self.next_column)
            {
                continue;
            }
            let state_word = frontier.detector_words + word;
            let first = frontier.parent.key(0, frontier.stride)[state_word] & mask;
            if (1..frontier.parent.masses.len()).all(|index| {
                frontier.parent.key(index, frontier.stride)[state_word] & mask == first
            }) {
                set_bit(&mut self.committed_mask, logical);
                self.committed[word] |= first;
                self.pending_commitments.push((logical, first != 0));
            }
        }
    }
}

fn binary_kernel_error() -> DecoderError {
    DecoderError::InvalidConfiguration("streaming v1 supports the binary float kernel".into())
}

fn validate_streaming_config(config: &TrellisConfig) -> Result<(), DecoderError> {
    if config.metric_mode != MetricMode::LogSumExpFloat {
        return Err(DecoderError::InvalidConfiguration(
            "streaming v1 supports the LogSumExpFloat metric".into(),
        ));
    }
    if config.bp_score_iterations != 0 {
        return Err(DecoderError::InvalidConfiguration(
            "streaming requires bp_score_iterations == 0 because BP-guided scoring reads the whole syndrome before the column walk".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod fixed_lag_tests {
    use super::*;
    fn stream(bounds: Vec<usize>) -> TrellisStreamingDecoder {
        stream_with(
            "error(0.1) D0 D2 L0\nerror(0.2) D1\nerror(0.3) D2",
            bounds,
            2,
        )
    }
    fn stream_with(dem: &str, bounds: Vec<usize>, window_size: usize) -> TrellisStreamingDecoder {
        let mut decoder = TrellisStreamingDecoder::from_dem_str(
            dem,
            TrellisConfig {
                k: 256,
                delta: 14.0,
                score_alpha: 0.8,
                merge_indistinguishable: false,
                ..TrellisConfig::default()
            },
        )
        .unwrap();
        decoder
            .set_streaming_mode(StreamingMode::FixedLag {
                detector_round_bounds: bounds,
                window_size,
            })
            .unwrap();
        decoder
    }
    fn columns_after_each_round(
        decoder: &mut TrellisStreamingDecoder,
        rounds: &[&[u8]],
    ) -> Vec<usize> {
        rounds
            .iter()
            .map(|round| {
                decoder.feed_prefix(round).unwrap();
                decoder.advance_prediction().unwrap()
            })
            .collect()
    }
    #[test]
    fn one_round_startup_and_terminal_flush() {
        let mut decoder = stream(vec![0, 1, 2, 3]);
        decoder.feed_prefix(&[1]).unwrap();
        assert_eq!(decoder.advance_prediction().unwrap(), 0);
        decoder.feed_prefix(&[0]).unwrap();
        // Active D2 is not known yet, but D0 can close.
        assert_eq!(decoder.advance_prediction().unwrap(), 1);
        decoder.feed_prefix(&[1]).unwrap();
        assert_eq!(decoder.advance_prediction().unwrap(), 3);
        assert!(decoder.flush_prediction().unwrap().get(0));
        decoder.reset();
        decoder.feed_dense(&[0, 1, 0]).unwrap();
        assert!(!decoder.flush_prediction().unwrap().get(0));
    }
    #[test]
    fn repeated_empty_rounds_are_explicit_arrivals() {
        let mut decoder = stream(vec![0, 0, 1, 2, 3]);
        decoder.feed_prefix(&[]).unwrap();
        assert_eq!(decoder.advance_prediction().unwrap(), 0);
        decoder.feed_prefix(&[1]).unwrap();
        assert_eq!(decoder.advance_prediction().unwrap(), 0);
        decoder.feed_prefix(&[0]).unwrap();
        assert_eq!(decoder.advance_prediction().unwrap(), 1);
        decoder.feed_prefix(&[1]).unwrap();
        assert!(decoder.flush_prediction().unwrap().get(0));
    }
    #[test]
    fn invalid_bounds_and_partial_round_leave_shot_unchanged() {
        let mut decoder = stream(vec![0, 1, 2, 3]);
        assert!(decoder.feed_prefix(&[]).is_err());
        assert_eq!(decoder.arrived_count, 0);
        assert_eq!(decoder.arrived_round, 0);
        assert!(
            decoder
                .set_streaming_mode(StreamingMode::FixedLag {
                    detector_round_bounds: vec![0, 2, 1, 3],
                    window_size: 2
                })
                .is_err()
        );
        decoder.feed_prefix(&[0]).unwrap();
        assert!(
            decoder
                .set_streaming_mode(StreamingMode::DependencyDriven)
                .is_err()
        );
    }
    #[test]
    fn flush_waits_for_trailing_empty_round() {
        let mut decoder = stream(vec![0, 3, 3]);
        decoder.feed_prefix(&[1, 0, 1]).unwrap();
        assert!(decoder.flush().is_err());
        assert!(decoder.flush_prediction().is_err());
        assert!(decoder.flush_prediction_without_evidence().is_err());
        decoder.feed_prefix(&[]).unwrap();
        assert!(decoder.flush().unwrap().predicted.get(0));
    }
    #[test]
    fn column_waits_for_the_round_of_its_last_closing_row() {
        // The only column touching D2 closes it, two rounds after its first row.
        let mut decoder = stream_with("error(0.1) D0 D2 L0\nerror(0.2) D1", vec![0, 1, 2, 3], 2);
        assert_eq!(
            columns_after_each_round(&mut decoder, &[&[1], &[0], &[1]]),
            [0, 0, 2]
        );
        assert!(decoder.flush_prediction().unwrap().get(0));
    }
    #[test]
    fn window_three_keeps_two_rounds_of_lookahead() {
        // Round 1's column is released by round 3, before the terminal round.
        let mut decoder = stream_with(
            "error(0.1) D0 L0\nerror(0.1) D1\nerror(0.1) D2\nerror(0.1) D3",
            vec![0, 1, 2, 3, 4],
            3,
        );
        assert_eq!(
            columns_after_each_round(&mut decoder, &[&[1], &[0], &[0], &[0]]),
            [0, 0, 1, 4]
        );
    }
    #[test]
    fn a_late_closing_column_holds_back_every_later_column() {
        // Release rounds before the prefix maximum are 1, 3, 1, 1: the second
        // column closes D1 and D2, so no later column may run before round 3.
        let mut decoder = stream_with(
            "error(0.1) D0 L0\nerror(0.1) D0 D1 D2\nerror(0.2) D0\nerror(0.3) D0",
            vec![0, 1, 2, 3],
            2,
        );
        assert_eq!(
            columns_after_each_round(&mut decoder, &[&[1], &[1], &[1]]),
            [0, 1, 4]
        );
    }
}

#[cfg(test)]
mod scoring_prefix_plan_tests {
    use super::*;

    fn model() -> SparseDem {
        SparseDem {
            mechanisms: (0_u32..24)
                .map(|i| (0.03 + f64::from(i % 3) * 0.02, vec![i % 8], vec![i % 2]))
                .collect(),
            detector_coords: std::collections::BTreeMap::new(),
            num_detectors: 8,
            num_observables: 2,
        }
    }
    fn compare(a: &TrellisStreamingDecoder, b: &TrellisStreamingDecoder) {
        let a = &a.progress;
        let b = &b.progress;
        assert_eq!(a.frontier.parent.words, b.frontier.parent.words);
        assert!(
            a.frontier.parent.masses.iter().map(|m| m.to_bits()).eq(b
                .frontier
                .parent
                .masses
                .iter()
                .map(|m| m.to_bits()))
        );
        assert_eq!(a.transitions, b.transitions);
        assert_eq!(a.dropped_states, b.dropped_states);
        assert_eq!(a.dropped_log_mass.to_bits(), b.dropped_log_mass.to_bits());
        assert_eq!(a.k_capped, b.k_capped);
        assert_eq!(a.delta_pruned, b.delta_pruned);
        assert_eq!(a.peak_retained_states, b.peak_retained_states);
    }
    #[test]
    fn cache_matches_fallback_with_delayed_and_repeated_advances() {
        let dem = model();
        for bounds in [vec![0, 2, 4, 6, 8], vec![0, 0, 2, 5, 8]] {
            for window_size in [2, 3] {
                for full in [false, true] {
                    for schedule in [0, 1, 2] {
                        let mode = StreamingMode::FixedLag {
                            detector_round_bounds: bounds.clone(),
                            window_size,
                        };
                        let mut a = TrellisStreamingDecoder::from_sparse_dem(
                            &dem,
                            TrellisConfig::default(),
                        )
                        .unwrap();
                        let mut b = TrellisStreamingDecoder::from_sparse_dem(
                            &dem,
                            TrellisConfig::default(),
                        )
                        .unwrap();
                        a.set_streaming_mode(mode.clone()).unwrap();
                        b.set_streaming_mode(mode).unwrap();
                        b.scoring_prefix_plans.clear();
                        for bits in 0..256 {
                            a.reset();
                            b.reset();
                            for round in 1..bounds.len() {
                                let block: Vec<u8> = (bounds[round - 1]..bounds[round])
                                    .map(|i| u8::from((bits >> i) & 1 != 0))
                                    .collect();
                                assert_eq!(
                                    format!("{:?}", a.feed_prefix(&block)),
                                    format!("{:?}", b.feed_prefix(&block))
                                );
                                if schedule == 0
                                    || (schedule == 1 && round % 2 == 0)
                                    || round == bounds.len() - 1
                                {
                                    for _ in 0..2 {
                                        let mode = if full {
                                            crate::DecodeMode::Full
                                        } else {
                                            crate::DecodeMode::PredictionOnly
                                        };
                                        assert_eq!(
                                            format!("{:?}", a.advance_prediction_mode(mode)),
                                            format!("{:?}", b.advance_prediction_mode(mode))
                                        );
                                        compare(&a, &b);
                                    }
                                }
                            }
                            let mode = if full {
                                crate::DecodeMode::Full
                            } else {
                                crate::DecodeMode::PredictionOnly
                            };
                            assert_eq!(
                                format!("{:?}", a.flush_prediction_mode(mode)),
                                format!("{:?}", b.flush_prediction_mode(mode))
                            );
                            compare(&a, &b);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn plans_cover_columns_preserve_failed_mode_and_clear_on_dependency_mode() {
        let mut a =
            TrellisStreamingDecoder::from_sparse_dem(&model(), TrellisConfig::default()).unwrap();
        a.set_streaming_mode(StreamingMode::FixedLag {
            detector_round_bounds: vec![0, 2, 4, 6, 8],
            window_size: 2,
        })
        .unwrap();
        let Kernel::Binary(columns) = &a.model.kernel else {
            panic!()
        };
        assert_eq!(
            a.scoring_prefix_plans
                .iter()
                .map(|p| p.ends.len())
                .sum::<usize>(),
            columns.len()
        );
        for plan in &a.scoring_prefix_plans {
            for (i, &end) in plan.ends.iter().enumerate() {
                assert_eq!(
                    end,
                    columns[plan.start + i]
                        .suffix_compatibility
                        .partition_point(|row| row.word as usize * WORD_BITS
                            + usize::from(row.bit)
                            < plan.known_count)
                );
            }
        }
        let before = format!("{:?}", a.scoring_prefix_plans);
        assert!(
            a.set_streaming_mode(StreamingMode::FixedLag {
                detector_round_bounds: vec![0, 8],
                window_size: 2
            })
            .is_err()
        );
        assert_eq!(before, format!("{:?}", a.scoring_prefix_plans));
        a.reset();
        assert_eq!(before, format!("{:?}", a.scoring_prefix_plans));
        a.set_streaming_mode(StreamingMode::DependencyDriven)
            .unwrap();
        assert!(a.scoring_prefix_plans.is_empty());
    }
    #[test]
    fn mismatched_prefix_and_outside_plan_use_fallback() {
        let plan = ScoringPrefixPlan {
            known_count: 64,
            start: 5,
            ends: vec![1, 2, 3],
        };
        assert_eq!(plan.for_range(64, 5..8), Some(&[1, 2, 3][..]));
        assert_eq!(plan.for_range(64, 6..7), Some(&[2][..]));
        assert_eq!(plan.for_range(64, 8..8), Some(&[][..]));
        assert_eq!(plan.for_range(63, 5..8), None);
        assert_eq!(plan.for_range(64, 4..8), None);
        assert_eq!(plan.for_range(64, 5..9), None);
    }
}
