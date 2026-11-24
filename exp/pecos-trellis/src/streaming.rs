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
#[cfg(feature = "experimental-conditional-retirement")]
use {
    crate::{Candidate, StateKey, TrellisLogicalMass, logaddexp, sort_candidates},
    std::collections::BTreeMap,
};

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

/// Opt-in irreversible conditioning of an already completed logical class.
///
/// Masses describe the current retained frontier, without normalization. This
/// operation changes inference: later results and commitments are conditional
/// on its selected class, not the original unconditioned batch posterior.
#[cfg(feature = "experimental-conditional-retirement")]
#[derive(Clone, Debug)]
pub struct ConditionalLogicalRetirement {
    /// Selected values at the requested original logical indices; other bits are zero.
    pub selected_class: ObsMask,
    /// Masked classes in descending mass, with unsigned numeric mask tie order.
    pub class_log_masses: Vec<TrellisLogicalMass>,
    /// Stable current-key-order sum of all retained masses before conditioning.
    pub log_evidence_before: f64,
    /// Stable sum of the selected class; surviving individual masses are unchanged.
    pub log_evidence_after: f64,
    /// Selected class mass minus runner-up class mass, or `None` for one class.
    pub runner_up_gap: Option<f64>,
    /// Retained states before conditioning.
    pub states_before: usize,
    /// Retained states after conditioning.
    pub states_after: usize,
    /// Removed states; separate from ordinary K/Delta pruning telemetry.
    pub removed_states: usize,
    /// Existing commitment snapshot, evaluated once after the stable filter.
    pub progress: StreamingProgress,
}

/// Read-only analysis of the completed masked class in the retained frontier.
///
/// These are base masses conditional on prior retirements and any K/Delta
/// truncation. They are not a calibrated probability for the original model.
#[cfg(feature = "experimental-conditional-retirement")]
#[derive(Clone, Debug)]
pub struct ConditionalLogicalClassPreview {
    /// Winner at the requested original logical indices; other bits are zero.
    pub selected_class: ObsMask,
    /// Descending base masses, with unsigned numeric mask ties.
    pub class_log_masses: Vec<TrellisLogicalMass>,
    /// Existing retained-key-order evidence sum before any conditioning.
    pub log_evidence_before: f64,
    /// Base log mass of the selected masked class.
    pub selected_log_mass: f64,
    /// Winner minus runner-up mass, or `None` for a sole class.
    pub runner_up_gap: Option<f64>,
    /// Number of retained states before the proposed conditioning.
    pub states_before: usize,
    /// Retained states belonging to the selected masked class.
    pub selected_state_count: usize,
}

/// Atomic confidence decision. A deferred operation does not consume progress
/// or mutate the beam, pruning counters, commitments, or any decoder state.
#[cfg(feature = "experimental-conditional-retirement")]
#[derive(Clone, Debug)]
pub enum ConditionalLogicalRetirementDecision {
    Deferred(ConditionalLogicalClassPreview),
    Retired(ConditionalLogicalRetirement),
}

/// Read-only retained-mass analysis of an XOR-linear logical projection.
///
/// Output bit `j` is the parity of the intersection between projection row `j`
/// and the original full logical key. Returned masks use these compact row
/// coordinates, not the model's original logical indices. Dependent and zero
/// rows are valid; the caller need not supply an independent basis.
#[cfg(feature = "experimental-conditional-retirement")]
#[derive(Clone, Debug)]
pub struct ConditionalLinearClassPreview {
    /// Winning projected class, in compact row coordinates.
    pub selected_class: ObsMask,
    /// Projected classes in descending base mass, with unsigned numeric ties.
    pub class_log_masses: Vec<TrellisLogicalMass>,
    /// Stable current-key-order sum before any conditioning.
    pub log_evidence_before: f64,
    /// Base log mass of the winning projected class.
    pub selected_log_mass: f64,
    /// Winner minus runner-up mass, or `None` for a sole class.
    pub runner_up_gap: Option<f64>,
    /// Retained states before the proposed conditioning.
    pub states_before: usize,
    /// Retained states belonging to the winning projected class.
    pub selected_state_count: usize,
}

/// Irreversible conditioning on a completed XOR-linear logical class.
///
/// Full original keys and surviving individual mass bits remain unchanged.
/// This is conditional inference over the retained frontier, not a zero-regret
/// claim, full untruncated posterior, or physical Pauli-frame interpretation.
#[cfg(feature = "experimental-conditional-retirement")]
#[derive(Clone, Debug)]
pub struct ConditionalLinearRetirement {
    /// Winning class in compact row coordinates, never original bit indices.
    pub selected_class: ObsMask,
    /// Projected class masses, preserving the preview's exact fold/tie order.
    pub class_log_masses: Vec<TrellisLogicalMass>,
    /// Stable sum of all retained masses before conditioning.
    pub log_evidence_before: f64,
    /// Winning class's stable sum; survivors are not renormalized.
    pub log_evidence_after: f64,
    /// Winner minus runner-up mass, or `None` for a sole class.
    pub runner_up_gap: Option<f64>,
    /// Retained states before conditioning.
    pub states_before: usize,
    /// Retained states after conditioning.
    pub states_after: usize,
    /// Removed states, separate from ordinary K/Delta pruning telemetry.
    pub removed_states: usize,
    /// Ordinary commitments of original logical bits after filtering. Projected
    /// parity values are never reported as individual-bit commitments.
    pub progress: StreamingProgress,
}

#[derive(Debug)]
struct ScoringPrefixPlan {
    known_count: usize,
    start: usize,
    ends: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct BoundedAdvance {
    end: usize,
    known_count: Option<usize>,
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
/// With the optional `experimental-conditional-retirement` feature, explicitly
/// conditioning a completed logical class changes these results and subsequent
/// commitments to the selected conditional frontier; batch parity then does not
/// apply. Merely enabling the feature changes no decoding behavior.
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
    bounded_advance: Option<BoundedAdvance>,
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
            bounded_advance: None,
        };
        stream.reset();
        stream
    }

    /// Select a schedule before starting a shot. Cached compact model and
    /// prediction methods are reused. Fixed-lag changes finite-width
    /// pruning and is not promised to equal batch/dependency-driven decoding.
    ///
    /// # Errors
    /// Returns `InvalidConfiguration` after shot processing has started or for
    /// invalid bounds/window size, or when `incremental_score` is enabled with
    /// `FixedLag`. A failed selection preserves the old mode.
    pub fn set_streaming_mode(&mut self, mode: StreamingMode) -> Result<(), DecoderError> {
        if self.model.config.incremental_score && matches!(mode, StreamingMode::FixedLag { .. }) {
            return Err(DecoderError::InvalidConfiguration(
                "incremental_score cannot use FixedLag masked rows; use DependencyDriven or disable incremental_score".into(),
            ));
        }
        self.ensure_no_bounded_advance()?;
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
        self.ensure_no_bounded_advance()?;
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
        self.ensure_no_bounded_advance()?;
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
        Ok(self.take_progress())
    }

    /// Begin a resumable advance at the current detector prefix.
    ///
    /// This captures exactly the target of [`Self::advance`], without doing column
    /// work. Call [`Self::step_bounded_advance`] until
    /// [`Self::bounded_columns_remaining`] is `Some(0)`, then call
    /// [`Self::finish_bounded_advance`] for its identical commitment snapshot, or
    /// [`Self::finish_bounded_flush`] for a fully fed shot's full terminal result.
    /// Commitments are checked only at finish, preserving the unbounded callback's
    /// check cadence and the order of `newly_committed`.
    /// In fixed-lag mode, the current arrival round determines both the target
    /// column and the detector prefix visible to scoring. These are captured
    /// together, so yielding does not expose later arrivals or change pruning.
    ///
    /// Detectors, schedule, and scoring inputs cannot change
    /// during the drain. Feed and unbounded processing methods reject that state.
    /// Reset cancels the drain and starts a fresh shot. A column budget bounds
    /// actual columns of DP work, not wall-clock time or total decision latency.
    ///
    /// # Errors
    /// Returns `InvalidConfiguration` for reentry;
    /// returns a stored decoding failure unchanged.
    pub fn begin_bounded_advance(&mut self) -> Result<usize, DecoderError> {
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        self.ensure_no_bounded_advance()?;
        let (end, known_count) = self.advance_target();
        self.bounded_advance = Some(BoundedAdvance { end, known_count });
        Ok(end - self.next_column)
    }

    /// Number of captured ready columns still to process, or `None` outside a drain.
    #[must_use]
    pub fn bounded_columns_remaining(&self) -> Option<usize> {
        self.bounded_advance
            .map(|advance| advance.end - self.next_column)
    }

    /// Process at most `max_columns` of real DP work in the captured advance.
    ///
    /// Returns cumulative columns processed and whether the captured target is
    /// drained. Each column uses the unmodified scoring, merge, prune, mass-fold,
    /// and telemetry path. The captured detector visibility is unchanged across
    /// every substep: unrestricted for dependency-driven scoring, or the arrived
    /// prefix for fixed lag. No commitments are scanned or reported
    /// until finish. A drained step performs no column work and is idempotent.
    ///
    /// # Errors
    /// Returns `InvalidConfiguration` for a zero budget or missing begin, or the
    /// same sticky no-path/internal failure as the unbounded column walk.
    pub fn step_bounded_advance(
        &mut self,
        max_columns: usize,
    ) -> Result<(usize, bool), DecoderError> {
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        if max_columns == 0 {
            return Err(DecoderError::InvalidConfiguration(
                "bounded advance requires a positive column budget".into(),
            ));
        }
        let advance = self.bounded_advance.ok_or_else(|| {
            DecoderError::InvalidConfiguration("begin a bounded advance before stepping".into())
        })?;
        let end = self
            .next_column
            .saturating_add(max_columns)
            .min(advance.end);
        let prefix_ends = advance.known_count.and_then(|known_count| {
            self.scoring_prefix_plans
                .get(self.arrived_round)
                .and_then(|plan| plan.for_range(known_count, self.next_column..end))
        });
        if let Err(failure) = self.model.dispatch_binary_range_with_prefixes(
            &mut self.progress,
            (&self.observed, &self.model.suffix_values),
            self.next_column..end,
            crate::PruneParams {
                k: self.model.config.k,
                delta: self.model.config.delta,
            },
            crate::BinaryRangeOptions {
                mode: self.model.decode_mode(),
                known_count: advance.known_count,
            },
            prefix_ends,
        ) {
            self.failure = Some(failure);
            return Err(failure.error());
        }
        self.next_column = end;
        Ok((self.next_column, end == advance.end))
    }

    /// Finish a drained advance without doing any additional column work.
    ///
    /// Returns exactly the snapshot from the corresponding unbounded advance.
    /// After this call the next detector prefix may be fed. For a terminal block,
    /// [`Self::finish_bounded_flush`] can instead construct the retained result.
    ///
    /// # Errors
    /// Returns `InvalidConfiguration` for a missing or undrained begin, or a stored
    /// decoding failure. An invalid finish leaves the pending drain unchanged.
    pub fn finish_bounded_advance(&mut self) -> Result<StreamingProgress, DecoderError> {
        self.ensure_bounded_advance_drained()?;
        self.bounded_advance = None;
        Ok(self.take_progress())
    }

    /// Finish a fully fed, drained terminal advance with the full retained result.
    ///
    /// This does no column work. Commitment checks and the terminal constructor
    /// are identical to `flush`, including the configured dropped-mass diagnostic.
    /// This invocation and the preceding step invocations have separate
    /// costs; a column budget is not a terminal-construction time bound.
    ///
    /// # Errors
    /// Returns a stored failure first, then the unbounded incomplete-shot error,
    /// or `InvalidConfiguration` for a missing/undrained begin. No failed protocol
    /// check consumes the drain or changes the frontier.
    ///
    /// # Panics
    /// Panics if a complete drain has not processed every model column, or a
    /// committed bit differs from the terminal prediction. Both indicate an
    /// internal engine invariant violation.
    pub fn finish_bounded_flush(&mut self) -> Result<TrellisResult, DecoderError> {
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        self.ensure_shot_complete()?;
        self.ensure_bounded_advance_drained()?;
        self.finish_bounded_advance()?;
        Ok(self.terminal_result())
    }

    /// Check uncommitted logicals and report commitments found since the last snapshot.
    fn take_progress(&mut self) -> StreamingProgress {
        self.check_commitments();
        let newly_committed = self.pending_commitments.clone();
        self.pending_commitments.clear();
        StreamingProgress {
            columns_processed: self.next_column,
            newly_committed,
            committed: ObsMask::from_words(&self.committed),
            committed_mask: ObsMask::from_words(&self.committed_mask),
        }
    }

    /// Build the batch terminal result once every column has been processed.
    ///
    /// # Panics
    /// Panics if any column remains unprocessed or a committed bit disagrees with
    /// the final prediction; both indicate an engine invariant violation.
    fn terminal_result(&self) -> TrellisResult {
        let Kernel::Binary(columns) = &self.model.kernel else {
            unreachable!("streaming constructors validate the binary kernel");
        };
        assert_eq!(
            self.next_column,
            columns.len(),
            "flush must process every column"
        );
        let result = self
            .model
            .finish_binary(&self.progress, self.next_column, 0.0);
        for logical in set_bits(&self.committed_mask) {
            let value = self.committed[logical / WORD_BITS] & (1_u64 << (logical % WORD_BITS)) != 0;
            assert!(
                result.predicted.get(logical) == value,
                "committed logical bit {logical} differs from the terminal prediction"
            );
        }
        result
    }

    fn ensure_bounded_advance_drained(&self) -> Result<BoundedAdvance, DecoderError> {
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        let advance = self.bounded_advance.ok_or_else(|| {
            DecoderError::InvalidConfiguration("begin a bounded advance before finishing".into())
        })?;
        if advance.end != self.next_column {
            return Err(DecoderError::InvalidConfiguration(
                "drain every captured column before finishing the bounded advance".into(),
            ));
        }
        Ok(advance)
    }

    fn ensure_no_bounded_advance(&self) -> Result<(), DecoderError> {
        if self.bounded_advance.is_some() {
            return Err(DecoderError::InvalidConfiguration(
                "finish or reset the pending bounded advance before feeding or unbounded processing".into(),
            ));
        }
        Ok(())
    }

    /// Process ready columns without constructing commitment snapshots.
    ///
    /// This path retains the same candidate set and correction as [`Self::advance`],
    /// with dropped-mass telemetry controlled by the config. It is intended for
    /// latency-sensitive runtimes that need only the final prediction.
    ///
    /// # Errors
    /// Returns the same no-path or internal error as [`Self::advance`].
    pub fn advance_prediction(&mut self) -> Result<usize, DecoderError> {
        // A stored failure outranks protocol errors, as in feed and flush. A
        // failed bounded step leaves its drain pending until reset.
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        self.ensure_no_bounded_advance()?;
        let (end, known_count) = self.advance_target();
        // Delayed advances can include columns from earlier rounds. Use a plan
        // only when both the range and the known detector count match exactly.
        let prefix_ends = self
            .scoring_prefix_plans
            .get(self.arrived_round)
            .and_then(|plan| plan.for_range(self.arrived_count, self.next_column..end));
        if let Err(failure) = self.model.dispatch_binary_range_with_prefixes(
            &mut self.progress,
            (&self.observed, &self.model.suffix_values),
            self.next_column..end,
            crate::PruneParams {
                k: self.model.config.k,
                delta: self.model.config.delta,
            },
            crate::BinaryRangeOptions {
                mode: self.model.decode_mode(),
                known_count,
            },
            prefix_ends,
        ) {
            self.failure = Some(failure);
            return Err(failure.error());
        }
        self.next_column = end;
        Ok(self.next_column)
    }

    /// Capture the unbounded callback's release target and scoring visibility.
    fn advance_target(&self) -> (usize, Option<usize>) {
        match &self.mode {
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
        }
    }

    /// Irreversibly retain the maximum-mass class of completed logical bits.
    ///
    /// The caller chooses only logical indices, never a desired class. Requested
    /// indices must have no unprocessed stochastic toggle. Fixed model offsets
    /// are already present in the frontier. This method does not enforce a time
    /// horizon or mechanism ownership; an application must validate those rules.
    ///
    /// Classes aggregate **base log masses**, in existing retained-key order,
    /// using the ordinary exact log-sum-exp routine. Selection uses the existing
    /// descending `f64::total_cmp` ordering and unsigned numeric logical-mask tie
    /// rule. Stable filtering preserves every surviving full key and mass bit.
    /// There is no clearing, relabeling, merging, normalization, column work or
    /// change to ordinary pruning counters. Repeating a completed conditioning
    /// succeeds with the same sole class. Commitments are checked only after
    /// filtering, and are relative to this now-conditional frontier.
    ///
    /// This is experimental, regretful conditioning, not a zero-regret promise,
    /// an unconditioned global posterior, or physical Pauli-frame inference.
    ///
    /// # Errors
    /// Returns a stored decoding failure first; otherwise `InvalidConfiguration`
    /// for a pending bounded phase or empty/duplicate/out-of-range/unfinished
    /// indices. Invalid requests leave the stream and pending progress unchanged.
    /// Returns `InternalError` for an invalid empty/non-finite retained frontier.
    #[cfg(feature = "experimental-conditional-retirement")]
    pub fn retire_completed_logical_class(
        &mut self,
        logicals: &[usize],
    ) -> Result<ConditionalLogicalRetirement, DecoderError> {
        let mask = self.completed_retirement_mask(logicals)?;
        let preview = self.analyze_completed_class(&mask)?;
        Ok(self.apply_completed_class(&mask, preview))
    }

    /// Analyze a completed logical class without changing any decoder state.
    /// Validation, aggregation order, and tie rules match unconditional retirement.
    /// Pending commitments are not consumed by this read-only call.
    ///
    /// # Errors
    /// Returns the same protocol/frontier errors as
    /// [`Self::retire_completed_logical_class`], without mutation.
    #[cfg(feature = "experimental-conditional-retirement")]
    pub fn preview_completed_logical_class(
        &self,
        logicals: &[usize],
    ) -> Result<ConditionalLogicalClassPreview, DecoderError> {
        let mask = self.completed_retirement_mask(logicals)?;
        self.analyze_completed_class(&mask)
    }

    /// Atomically analyze and, only above the requested retained-mass threshold,
    /// condition a completed logical class. No caller-supplied winner is accepted.
    ///
    /// Zero reproduces unconditional retirement. One accepts only a sole class,
    /// even if a tiny runner-up would round the winning probability to one.
    /// Intermediate thresholds compare log masses using deterministic `libm::log`.
    /// Deferred returns before any mutable operation or progress consumption.
    ///
    /// # Errors
    /// Returns stored failures first, then the usual retirement validation errors
    /// or `InvalidConfiguration` for a non-finite/out-of-range threshold.
    #[cfg(feature = "experimental-conditional-retirement")]
    pub fn try_retire_completed_logical_class(
        &mut self,
        logicals: &[usize],
        minimum_retained_probability: f64,
    ) -> Result<ConditionalLogicalRetirementDecision, DecoderError> {
        let mask = self.completed_retirement_mask(logicals)?;
        if !minimum_retained_probability.is_finite()
            || !(0.0..=1.0).contains(&minimum_retained_probability)
        {
            return Err(DecoderError::InvalidConfiguration(
                "retained class probability threshold must be finite and within [0, 1]".into(),
            ));
        }
        let preview = self.analyze_completed_class(&mask)?;
        let accepts = minimum_retained_probability == 0.0
            || (minimum_retained_probability.to_bits() == 1.0_f64.to_bits()
                && preview.class_log_masses.len() == 1)
            || (minimum_retained_probability > 0.0
                && minimum_retained_probability < 1.0
                && preview.selected_log_mass - preview.log_evidence_before
                    >= libm::log(minimum_retained_probability));
        if !accepts {
            return Ok(ConditionalLogicalRetirementDecision::Deferred(preview));
        }
        Ok(ConditionalLogicalRetirementDecision::Retired(
            self.apply_completed_class(&mask, preview),
        ))
    }

    /// Preview the class defined by XOR-linear rows over original logical bits.
    ///
    /// Row `j` contributes output bit `j`; each parity may reference any number
    /// of original logical words. Classes use retained base masses and the same
    /// exact log-sum-exp fold and unsigned numeric tie rule as ordinary logical
    /// retirement. This method does not consume progress or mutate any state.
    ///
    /// # Errors
    /// Returns a stored decoding failure first, then `InvalidConfiguration` for
    /// an unfinished bounded phase, an empty row list, or any referenced source
    /// bit that is out of range or still has an unprocessed normalized toggle.
    /// Every referenced bit must be complete, even if unfinished contributions
    /// would cancel across a row. Zero/dependent rows are valid. Returns
    /// `InternalError` for an empty or non-finite retained frontier.
    #[cfg(feature = "experimental-conditional-retirement")]
    pub fn preview_completed_linear_class(
        &self,
        rows: &[ObsMask],
    ) -> Result<ConditionalLinearClassPreview, DecoderError> {
        self.validate_completed_linear_rows(rows)?;
        self.analyze_completed_linear_class(rows)
    }

    /// Atomically choose and condition a completed XOR-linear logical class.
    ///
    /// Validation and analysis complete before the first mutation. Selection is
    /// over projected class sums, not the most likely full logical tuple. The
    /// stable filter preserves each survivor's entire original key and mass
    /// bits, including correlations with all unselected logical/active-detector
    /// components. It does not clear tags, merge states, renormalize, replay
    /// columns, or change K/Delta pruning counters. Existing commitment logic
    /// runs once afterward; projection bits are not synthetic commitments.
    ///
    /// # Errors
    /// Returns the same protocol/frontier errors as the read-only preview.
    /// All invalid requests leave the stream and pending progress unchanged.
    #[cfg(feature = "experimental-conditional-retirement")]
    pub fn retire_completed_linear_class(
        &mut self,
        rows: &[ObsMask],
    ) -> Result<ConditionalLinearRetirement, DecoderError> {
        self.validate_completed_linear_rows(rows)?;
        let preview = self.analyze_completed_linear_class(rows)?;
        let frontier = &mut self.progress.frontier;
        frontier.retained.clear();
        for index in 0..preview.states_before {
            let key = frontier.parent.key(index, frontier.stride);
            let keep = Self::linear_class_words(&key[frontier.detector_words..], rows).as_slice()
                == preview.selected_class.words();
            frontier.retained.push(keep);
        }
        frontier.retain_in_place();
        let states_after = frontier.parent.masses.len();
        debug_assert_eq!(states_after, preview.selected_state_count);
        let progress = self.take_progress();
        Ok(ConditionalLinearRetirement {
            selected_class: preview.selected_class,
            class_log_masses: preview.class_log_masses,
            log_evidence_before: preview.log_evidence_before,
            log_evidence_after: preview.selected_log_mass,
            runner_up_gap: preview.runner_up_gap,
            states_before: preview.states_before,
            states_after,
            removed_states: preview.states_before - states_after,
            progress,
        })
    }

    #[cfg(feature = "experimental-conditional-retirement")]
    fn validate_completed_linear_rows(&self, rows: &[ObsMask]) -> Result<(), DecoderError> {
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        self.ensure_no_bounded_advance()?;
        if rows.is_empty() {
            return Err(DecoderError::InvalidConfiguration(
                "project at least one logical row".into(),
            ));
        }
        for row in rows {
            for logical in row.iter_set_bits() {
                let Some(last) = self.last_toggle_column.get(logical) else {
                    return Err(DecoderError::InvalidConfiguration(
                        "projection source logical index out of range".into(),
                    ));
                };
                if last.is_some_and(|column| column >= self.next_column) {
                    return Err(DecoderError::InvalidConfiguration(
                        "projection source logical still has an unprocessed toggle".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "experimental-conditional-retirement")]
    fn linear_class_words(logical: &[u64], rows: &[ObsMask]) -> Vec<u64> {
        let mut projected = vec![0; rows.len().div_ceil(WORD_BITS)];
        for (index, row) in rows.iter().enumerate() {
            let parity = logical
                .iter()
                .zip(row.words())
                .fold(0, |parity, (&word, &mask)| {
                    parity ^ (word & mask).count_ones()
                });
            if parity & 1 != 0 {
                set_bit(&mut projected, index);
            }
        }
        projected
    }

    #[cfg(feature = "experimental-conditional-retirement")]
    fn analyze_completed_linear_class(
        &self,
        rows: &[ObsMask],
    ) -> Result<ConditionalLinearClassPreview, DecoderError> {
        let frontier = &self.progress.frontier;
        let states_before = frontier.parent.masses.len();
        if states_before == 0 || frontier.parent.masses.iter().any(|mass| !mass.is_finite()) {
            return Err(DecoderError::InternalError(
                "conditional linear retirement requires a nonempty finite frontier".into(),
            ));
        }
        let mut classes = BTreeMap::<StateKey, f64>::new();
        let mut log_evidence_before = f64::NEG_INFINITY;
        for (index, &mass) in frontier.parent.masses.iter().enumerate() {
            let key = frontier.parent.key(index, frontier.stride);
            let logical = Self::linear_class_words(&key[frontier.detector_words..], rows);
            let class = classes
                .entry(StateKey {
                    active_syndrome: Vec::new(),
                    logical,
                })
                .or_insert(f64::NEG_INFINITY);
            *class = logaddexp(*class, mass);
            log_evidence_before = logaddexp(log_evidence_before, mass);
        }
        let mut classes: Vec<_> = classes
            .into_iter()
            .map(|(key, log_mass)| Candidate { key, log_mass })
            .collect();
        sort_candidates(&mut classes);
        let selected = &classes[0];
        let selected_class = ObsMask::from_words(&selected.key.logical);
        let selected_log_mass = selected.log_mass;
        let runner_up_gap = classes
            .get(1)
            .map(|runner| selected.log_mass - runner.log_mass);
        let selected_state_count = (0..states_before)
            .filter(|&index| {
                let key = frontier.parent.key(index, frontier.stride);
                Self::linear_class_words(&key[frontier.detector_words..], rows)
                    == selected.key.logical
            })
            .count();
        let class_log_masses = classes
            .into_iter()
            .map(|candidate| TrellisLogicalMass {
                logical: ObsMask::from_words(&candidate.key.logical),
                log_mass: candidate.log_mass,
            })
            .collect();
        Ok(ConditionalLinearClassPreview {
            selected_class,
            class_log_masses,
            log_evidence_before,
            selected_log_mass,
            runner_up_gap,
            states_before,
            selected_state_count,
        })
    }

    #[cfg(feature = "experimental-conditional-retirement")]
    fn analyze_completed_class(
        &self,
        mask: &[u64],
    ) -> Result<ConditionalLogicalClassPreview, DecoderError> {
        let frontier = &self.progress.frontier;
        let states_before = frontier.parent.masses.len();
        if states_before == 0 || frontier.parent.masses.iter().any(|mass| !mass.is_finite()) {
            return Err(DecoderError::InternalError(
                "conditional retirement requires a nonempty finite frontier".into(),
            ));
        }
        let mut classes = BTreeMap::<StateKey, f64>::new();
        let mut log_evidence_before = f64::NEG_INFINITY;
        for (index, &mass) in frontier.parent.masses.iter().enumerate() {
            let key = frontier.parent.key(index, frontier.stride);
            let logical = key[frontier.detector_words..]
                .iter()
                .zip(mask)
                .map(|(&word, &selected)| word & selected)
                .collect();
            let class = classes
                .entry(StateKey {
                    active_syndrome: Vec::new(),
                    logical,
                })
                .or_insert(f64::NEG_INFINITY);
            *class = logaddexp(*class, mass);
            log_evidence_before = logaddexp(log_evidence_before, mass);
        }
        let mut classes: Vec<_> = classes
            .into_iter()
            .map(|(key, log_mass)| Candidate { key, log_mass })
            .collect();
        sort_candidates(&mut classes);
        let selected = &classes[0];
        let selected_class = ObsMask::from_words(&selected.key.logical);
        let selected_log_mass = selected.log_mass;
        let runner_up_gap = classes
            .get(1)
            .map(|runner| selected.log_mass - runner.log_mass);
        let selected_state_count = (0..states_before)
            .filter(|&index| {
                let key = frontier.parent.key(index, frontier.stride);
                key[frontier.detector_words..]
                    .iter()
                    .zip(mask)
                    .zip(&selected.key.logical)
                    .all(|((&word, &selected_bits), &value)| word & selected_bits == value)
            })
            .count();
        let class_log_masses = classes
            .into_iter()
            .map(|candidate| TrellisLogicalMass {
                logical: ObsMask::from_words(&candidate.key.logical),
                log_mass: candidate.log_mass,
            })
            .collect();
        Ok(ConditionalLogicalClassPreview {
            selected_class,
            class_log_masses,
            log_evidence_before,
            selected_log_mass,
            runner_up_gap,
            states_before,
            selected_state_count,
        })
    }

    #[cfg(feature = "experimental-conditional-retirement")]
    fn apply_completed_class(
        &mut self,
        mask: &[u64],
        preview: ConditionalLogicalClassPreview,
    ) -> ConditionalLogicalRetirement {
        let frontier = &mut self.progress.frontier;
        frontier.retained.clear();
        for index in 0..preview.states_before {
            let key = frontier.parent.key(index, frontier.stride);
            let keep = key[frontier.detector_words..]
                .iter()
                .zip(mask)
                .zip(preview.selected_class.words())
                .all(|((&word, &selected_bits), &value)| word & selected_bits == value);
            frontier.retained.push(keep);
        }
        frontier.retain_in_place();
        let states_after = frontier.parent.masses.len();
        debug_assert_eq!(states_after, preview.selected_state_count);
        let progress = self.take_progress();
        ConditionalLogicalRetirement {
            selected_class: preview.selected_class,
            class_log_masses: preview.class_log_masses,
            log_evidence_before: preview.log_evidence_before,
            log_evidence_after: preview.selected_log_mass,
            runner_up_gap: preview.runner_up_gap,
            states_before: preview.states_before,
            states_after,
            removed_states: preview.states_before - states_after,
            progress,
        }
    }

    #[cfg(feature = "experimental-conditional-retirement")]
    fn completed_retirement_mask(&self, logicals: &[usize]) -> Result<Vec<u64>, DecoderError> {
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        self.ensure_no_bounded_advance()?;
        if logicals.is_empty() {
            return Err(DecoderError::InvalidConfiguration(
                "retire at least one logical bit".into(),
            ));
        }
        let mut mask = vec![0; self.model.logical_words];
        for &logical in logicals {
            let Some(last) = self.last_toggle_column.get(logical) else {
                return Err(DecoderError::InvalidConfiguration(
                    "retirement logical index out of range".into(),
                ));
            };
            let word = logical / WORD_BITS;
            let bit = 1_u64 << (logical % WORD_BITS);
            if mask[word] & bit != 0 {
                return Err(DecoderError::InvalidConfiguration(
                    "duplicate retirement logical index".into(),
                ));
            }
            if last.is_some_and(|column| column >= self.next_column) {
                return Err(DecoderError::InvalidConfiguration(
                    "retirement logical still has an unprocessed toggle".into(),
                ));
            }
            mask[word] |= bit;
        }
        Ok(mask)
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
        Ok(self.terminal_result())
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
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        self.ensure_shot_complete()?;
        self.advance_prediction()?;
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
        self.bounded_advance = None;
        self.progress.reset(
            &self.model,
            crate::PruneParams {
                k: self.model.config.k,
                delta: self.model.config.delta,
            },
        );
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
                for flag_on in [false, true] {
                    for schedule in [0, 1, 2] {
                        let mode = StreamingMode::FixedLag {
                            detector_round_bounds: bounds.clone(),
                            window_size,
                        };
                        let mut a = TrellisStreamingDecoder::from_sparse_dem(
                            &dem,
                            TrellisConfig {
                                dropped_mass: flag_on,
                                ..TrellisConfig::default()
                            },
                        )
                        .unwrap();
                        let mut b = TrellisStreamingDecoder::from_sparse_dem(
                            &dem,
                            TrellisConfig {
                                dropped_mass: flag_on,
                                ..TrellisConfig::default()
                            },
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
                                        assert_eq!(
                                            format!("{:?}", a.advance_prediction()),
                                            format!("{:?}", b.advance_prediction())
                                        );
                                        compare(&a, &b);
                                    }
                                }
                            }
                            assert_eq!(
                                format!("{:?}", a.flush_prediction()),
                                format!("{:?}", b.flush_prediction())
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

#[cfg(test)]
mod dropped_mass_tests {
    use super::*;

    fn check_accumulation(bounded: bool) {
        for dropped_mass in [false, true] {
            let mut decoder = TrellisStreamingDecoder::from_dem_str(
                "error(0.25) L0\nerror(0.2) L1",
                TrellisConfig {
                    dropped_mass,
                    k: 1,
                    ..TrellisConfig::default()
                },
            )
            .unwrap();
            decoder.feed_dense(&[]).unwrap();
            let result = if bounded {
                decoder.begin_bounded_advance().unwrap();
                while !decoder.step_bounded_advance(1).unwrap().1 {}
                decoder.finish_bounded_flush().unwrap()
            } else {
                decoder.flush().unwrap()
            };
            assert!(decoder.progress.dropped_states > 0);
            assert_eq!(result.dropped_states, decoder.progress.dropped_states);
            if dropped_mass {
                assert!(decoder.progress.dropped_log_mass.is_finite());
                assert_eq!(
                    result.dropped_log_mass.expect("enabled").to_bits(),
                    decoder.progress.dropped_log_mass.to_bits()
                );
            } else {
                assert_eq!(
                    decoder.progress.dropped_log_mass.to_bits(),
                    f64::NEG_INFINITY.to_bits()
                );
                assert_eq!(result.dropped_log_mass, None);
            }
        }
    }

    #[test]
    fn streaming_flush_dropped_mass_flag_controls_accumulation() {
        check_accumulation(false);
    }

    #[test]
    fn bounded_streaming_flush_dropped_mass_flag_controls_accumulation() {
        check_accumulation(true);
    }
}

#[cfg(test)]
mod incremental_score_tests {
    use super::*;

    #[test]
    fn integer_incremental_preserves_streaming_metric_error() {
        let dem = SparseDem::from_dem_str("error(0.1) D0 L0").unwrap();
        let error = TrellisStreamingDecoder::from_sparse_dem(
            &dem,
            TrellisConfig {
                incremental_score: true,
                metric_mode: MetricMode::MaxLogInt,
                ..TrellisConfig::default()
            },
        )
        .unwrap_err();
        let DecoderError::InvalidConfiguration(message) = error else {
            panic!("expected InvalidConfiguration");
        };
        assert_eq!(message, "streaming v1 supports the LogSumExpFloat metric");
    }

    #[test]
    fn fixed_lag_rejection_preserves_dependency_driven_and_split_bits() {
        let dem = SparseDem::from_dem_str("error(1) D130\nerror(0.1) D0 L0\nerror(0.2) D0 D70\nerror(0.3) D70 D130 L1\nerror(0.15) D130\n").unwrap();
        let config = TrellisConfig {
            incremental_score: true,
            k: 2,
            delta: 12.0,
            dropped_mass: true,
            ..TrellisConfig::default()
        };
        let mut batch = TrellisDecoder::from_sparse_dem(&dem, config.clone()).unwrap();
        let mut stream = TrellisStreamingDecoder::from_sparse_dem(&dem, config).unwrap();
        for bounds in [vec![], vec![0, 70, 131]] {
            let error = stream
                .set_streaming_mode(StreamingMode::FixedLag {
                    detector_round_bounds: bounds,
                    window_size: 2,
                })
                .unwrap_err();
            assert!(matches!(error, DecoderError::InvalidConfiguration(_)));
            assert!(error.to_string().contains("incremental_score"));
            assert!(
                error
                    .to_string()
                    .contains("DependencyDriven or disable incremental_score")
            );
            assert_eq!(stream.mode, StreamingMode::DependencyDriven);
        }
        for bits in 0..8 {
            let mut shot = vec![0; dem.num_detectors];
            for (i, d) in [0, 70, 130].into_iter().enumerate() {
                shot[d] = u8::from(bits & (1 << i) != 0);
            }
            let expected = batch.decode(&shot).unwrap();
            for bounded in [false, true] {
                stream.reset();
                for chunk in shot.chunks(17) {
                    stream.feed_prefix(chunk).unwrap();
                    if bounded {
                        stream.begin_bounded_advance().unwrap();
                        while stream.bounded_columns_remaining().unwrap() > 0 {
                            stream.step_bounded_advance(1).unwrap();
                        }
                        stream.finish_bounded_advance().unwrap();
                    } else {
                        stream.advance_prediction().unwrap();
                    }
                }
                let actual = stream.flush().unwrap();
                crate::incremental_score_tests::assert_bits(&expected, &actual);
                assert_eq!(
                    stream.progress.frontier.parent.suffix_sums,
                    batch.scratch.float_progress.frontier.parent.suffix_sums
                );
            }
        }
    }
}
