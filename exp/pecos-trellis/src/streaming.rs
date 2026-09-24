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

/// Opt-in streaming decoder for detector blocks arriving in index order.
///
/// On the same build and platform, a successful flush is bit-identical to batch
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
        };
        stream.reset();
        stream
    }

    /// Per-column lookahead in detector indices: readiness minus the column's
    /// highest toggled detector. A missing detector index is interpreted as -1.
    #[must_use]
    pub fn column_lookahead(&self) -> &[usize] {
        &self.lookahead
    }

    /// Append the NEXT block of detector values. Every nonzero byte means fired.
    /// Only newly arrived detectors are checked against the forced contribution
    /// on rows untouched by probabilistic columns.
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
        if total > self.model.num_detectors {
            return Err(DecoderError::InvalidDimensions {
                expected: self.model.num_detectors,
                actual: total,
            });
        }
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
        self.arrived_count = total;
        self.failure.map_or(Ok(()), |failure| Err(failure.error()))
    }

    /// Feed the whole syndrome in one block before any detectors have arrived.
    ///
    /// # Errors
    /// Returns a stored failure first, then `InvalidDimensions` for a syndrome
    /// whose length differs from the model's detector count. Appends through
    /// [`Self::feed_prefix`], which rejects overflow after a partial feed.
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
        self.feed_prefix(syndrome)
    }

    /// Process all ready columns, then check uncommitted logicals for unanimity.
    /// Initial forced commitments are reported on the first successful advance.
    ///
    /// # Errors
    /// Returns the same no-path or internal error as the batch column walk.
    pub fn advance(&mut self) -> Result<StreamingProgress, DecoderError> {
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        let end = self
            .ready
            .partition_point(|required| required.is_none_or(|d| d < self.arrived_count));
        if let Err(failure) = self.model.process_binary_range(
            &mut self.progress,
            &self.observed,
            &self.model.suffix_values,
            self.next_column..end,
        ) {
            self.failure = Some(failure);
            return Err(failure.error());
        }
        self.next_column = end;
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
    /// Returns `InvalidDimensions` until all detectors arrive, or the same no-path
    /// or internal error as [`Self::advance`].
    ///
    /// # Panics
    /// Panics if any column remains unprocessed or a committed bit disagrees with
    /// the final prediction; both indicate an engine invariant violation.
    pub fn flush(&mut self) -> Result<TrellisResult, DecoderError> {
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }

        if self.arrived_count != self.model.num_detectors {
            return Err(DecoderError::InvalidDimensions {
                expected: self.model.num_detectors,
                actual: self.arrived_count,
            });
        }
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

    /// Begin another shot, reusing buffers and committing bits that never toggle.
    pub fn reset(&mut self) {
        self.arrived_count = 0;
        self.next_column = 0;
        self.observed.fill(0);
        self.committed.fill(0);
        self.committed_mask.fill(0);
        self.pending_commitments.clear();
        self.failure = None;
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
