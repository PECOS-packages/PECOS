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

//! Trellis dynamic-programming engine for coset-mass decoding.
//!
//! The decoder performs ordered dynamic programming over independent binary
//! fault mechanisms or mutually exclusive multi-outcome factors. Prefixes with
//! identical active detector boundary and logical labels are merged by the
//! configured metric: log-sum-exp preserves degeneracy mass by default, while
//! integer max-log retains the best route. The configured frontier width and
//! log-mass window provide deterministic pruning for a fixed build and
//! platform; underlying `ln`/`exp` implementations may differ across platforms.
//! This engine is PECOS-native code. Its numerics are additionally held to a
//! bitwise parity contract with an external reference implementation of the
//! same algorithm class; that contract is maintained by a separate crate and
//! is not a constraint this crate imposes on its callers.

pub mod batch;
pub mod factor;
pub mod streaming;

pub use streaming::{StreamingProgress, TrellisStreamingDecoder};

use factor::{FactorModel, NormalizedFactor, Outcome};
use pecos_bp::{BpGraph, BpScratch, min_sum_bp_into};
use pecos_decoder_core::ObservableDecoder;
pub use pecos_decoder_core::dem::SparseDem;
pub use pecos_decoder_core::errors::DecoderError;
pub use pecos_decoder_core::obs_mask::ObsMask;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

const WORD_BITS: usize = u64::BITS as usize;
const BP_MIN_SUM_SCALE: f64 = 0.625;
const BP_SCORE_PROBABILITY_MIN: f64 = 1e-6;
const INT_METRIC_NEG_INF: i64 = i64::MIN / 4;
const INT_METRIC_MAX: i64 = i64::MAX / 4;

/// Arithmetic used to merge routes and rank trellis states.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MetricMode {
    /// Sum route masses with floating-point log-sum-exp. Exact unpruned
    /// results are logical-coset masses.
    #[default]
    LogSumExpFloat,
    /// Keep the best route with quantized integer max-log arithmetic.
    /// Unpruned results are Viterbi route masses, not logical-coset masses.
    MaxLogInt,
}

/// Processing order used by the trellis decoders.
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

impl TrellisOrdering {
    /// Resolve the processing order for a sparse detector error model.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if ordering generation fails.
    pub fn resolve(&self, dem: &SparseDem) -> Result<Option<Vec<usize>>, DecoderError> {
        match self {
            Self::Deadline => deadline_column_order(dem).map(Some),
            Self::BackwardDeadline => backward_deadline_column_order(dem).map(Some),
            Self::TimeOrder => Ok(None),
            Self::Explicit(order) => Ok(Some(order.clone())),
        }
    }
}

/// Pruning and column-order configuration for the trellis engine.
///
/// The [`Default`] pruning values are provisional pending benchmarking.
/// Pruning ranks accumulated prefix log mass plus a `score_alpha`-weighted
/// suffix-compatibility estimate. Unpruned results are exact and
/// upstream-verified.
#[derive(Clone, Debug, PartialEq)]
pub struct TrellisConfig {
    /// Maximum number of boundary states retained after each column.
    pub k: usize,
    /// Log-mass window below the best boundary state retained after each column.
    pub delta: f64,
    /// Weight applied to the suffix-compatibility score during pruning.
    /// Defaults to `0.8`, chosen to match the parity contract.
    pub score_alpha: f64,
    /// Optional permutation of the DEM mechanism or factor indices.
    pub column_order: Option<Vec<usize>>,
    /// Merge probabilistic mechanisms with identical detector and observable
    /// sets using their XOR-combined probability.
    ///
    /// This merge is mathematically exact under the default float metric and is
    /// rejected under `maxlog_int`. It takes a different floating-point path and
    /// the external parity contract on this engine is bitwise, so it is disabled
    /// by default.
    /// Zero-probability mechanisms are already discarded, while probability-one
    /// mechanisms remain separate in the forced layer and are not merged with
    /// otherwise identical probabilistic mechanisms.
    pub merge_indistinguishable: bool,
    /// Number of min-sum BP iterations used only to score pruning candidates.
    /// Zero disables BP-informed scoring.
    pub bp_score_iterations: usize,
    /// Arithmetic used for route merging and pruning scores.
    pub metric_mode: MetricMode,
    /// Quantization units per natural-log unit for [`MetricMode::MaxLogInt`].
    /// This must be positive in every mode and is ignored by the float metric.
    pub int_metric_scale: i32,
}

/// Pruning parameters for one attempt, validated against its decoder's model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PruneParams {
    /// Maximum retained boundary states; must be at least one.
    pub k: usize,
    /// Non-negative log-score window; must be finite for max-log arithmetic.
    pub delta: f64,
}

impl PruneParams {
    fn validate(self, metric_mode: MetricMode) -> Result<(), DecoderError> {
        if self.k == 0 {
            return Err(DecoderError::InvalidConfiguration(
                "TrellisConfig.k must be at least 1".into(),
            ));
        }
        if self.delta.is_nan() || self.delta < 0.0 {
            return Err(DecoderError::InvalidConfiguration(format!(
                "TrellisConfig.delta must be non-negative and not NaN, got {}",
                self.delta
            )));
        }
        if metric_mode == MetricMode::MaxLogInt && !self.delta.is_finite() {
            return Err(DecoderError::InvalidConfiguration(
                "delta must be finite under maxlog_int; infinite delta would quantize to zero and prune to score-ties"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Preparation of a shot, before any frontier work.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrellisPrepared {
    /// Precheck and optional BP scoring completed.
    Ready {
        /// Whether BP ran for this shot.
        bp_ran: bool,
        /// Wall-clock seconds spent on BP scoring.
        bp_seconds: f64,
    },
    /// Lowest detector with a residual no probabilistic mechanism can change.
    Residual { detector: usize },
}

impl TrellisConfig {
    /// Validate configuration rules that do not depend on a detector error model.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] for invalid pruning or metric options.
    pub fn validate(&self) -> Result<(), DecoderError> {
        PruneParams {
            k: self.k,
            delta: self.delta,
        }
        .validate(self.metric_mode)?;
        if self.metric_mode == MetricMode::MaxLogInt && self.merge_indistinguishable {
            return Err(DecoderError::InvalidConfiguration(
                "indistinguishable-mechanism merging sums coset mass and is incompatible with the max-log route metric"
                    .into(),
            ));
        }
        if self.int_metric_scale <= 0 {
            return Err(DecoderError::InvalidConfiguration(
                "TrellisConfig.int_metric_scale must be positive".into(),
            ));
        }
        if self.metric_mode == MetricMode::MaxLogInt
            && self.score_alpha > 0.0
            && quantize_metric(self.score_alpha, self.int_metric_scale) == 0
        {
            return Err(DecoderError::InvalidConfiguration(format!(
                "score_alpha {} quantizes to zero at int_metric_scale {} and would silently disable suffix scoring; pass score_alpha 0.0 to disable it explicitly or use a larger scale",
                self.score_alpha, self.int_metric_scale
            )));
        }
        if !self.score_alpha.is_finite() || self.score_alpha < 0.0 {
            return Err(DecoderError::InvalidConfiguration(format!(
                "TrellisConfig.score_alpha must be finite and non-negative, got {}",
                self.score_alpha
            )));
        }
        Ok(())
    }
}

impl Default for TrellisConfig {
    fn default() -> Self {
        // Provisional defaults pending benchmarking.
        Self {
            k: 64,
            delta: 50.0,
            score_alpha: 0.8,
            column_order: None,
            merge_indistinguishable: false,
            bp_score_iterations: 0,
            metric_mode: MetricMode::LogSumExpFloat,
            int_metric_scale: 1024,
        }
    }
}

/// Generate the deadline-optimized processing order for a sparse DEM.
///
/// The input mechanism order is treated as time order. Mechanisms that can
/// close detectors earlier are placed first; detector-free mechanisms sort
/// last. The returned permutation maps target positions to source mechanism
/// indices and can be assigned directly to [`TrellisConfig::column_order`].
///
/// # Errors
///
/// Returns [`DecoderError::InvalidConfiguration`] if a mechanism contains an
/// out-of-range or duplicate detector index.
pub fn deadline_column_order(dem: &SparseDem) -> Result<Vec<usize>, DecoderError> {
    let supports: Vec<Vec<u32>> = dem
        .mechanisms
        .iter()
        .map(|(_, detectors, _)| detectors.clone())
        .collect();
    let time_order: Vec<usize> = (0..dem.mechanisms.len()).collect();
    deadline_order_for_sequence(&supports, dem.num_detectors, "mechanism", &time_order)
}

/// Generate the backward deadline-optimized processing order for a sparse DEM.
///
/// This first computes the forward deadline order, reverses that ordered
/// sequence, reruns deadline optimization in the reversed time coordinates,
/// and composes the result back to original mechanism indices.
///
/// # Errors
///
/// Returns [`DecoderError::InvalidConfiguration`] if a mechanism contains an
/// out-of-range or duplicate detector index.
pub fn backward_deadline_column_order(dem: &SparseDem) -> Result<Vec<usize>, DecoderError> {
    let supports: Vec<Vec<u32>> = dem
        .mechanisms
        .iter()
        .map(|(_, detectors, _)| detectors.clone())
        .collect();
    let mut reversed_forward = deadline_column_order(dem)?;
    reversed_forward.reverse();
    deadline_order_for_sequence(&supports, dem.num_detectors, "mechanism", &reversed_forward)
}

/// Generate the deadline-optimized processing order for a factor model.
///
/// A factor's support is the sorted union of the detectors in all of its raw
/// outcomes, including zero-probability outcomes.
///
/// # Errors
///
/// Returns [`DecoderError::InvalidConfiguration`] if a support contains an
/// out-of-range detector index.
pub fn deadline_column_order_for_factors(model: &FactorModel) -> Result<Vec<usize>, DecoderError> {
    let supports = factor_supports(model);
    let time_order: Vec<usize> = (0..model.factors().len()).collect();
    deadline_order_for_sequence(&supports, model.num_detectors(), "factor", &time_order)
}

/// Generate the backward deadline-optimized processing order for a factor model.
///
/// # Errors
///
/// Returns [`DecoderError::InvalidConfiguration`] if a support contains an
/// out-of-range detector index.
pub fn backward_deadline_column_order_for_factors(
    model: &FactorModel,
) -> Result<Vec<usize>, DecoderError> {
    let supports = factor_supports(model);
    let mut reversed_forward = deadline_column_order_for_factors(model)?;
    reversed_forward.reverse();
    deadline_order_for_sequence(
        &supports,
        model.num_detectors(),
        "factor",
        &reversed_forward,
    )
}

/// Retained unnormalized joint log mass for one logical label.
#[derive(Clone, Debug, PartialEq)]
pub struct TrellisLogicalMass {
    /// Logical-observable flip label.
    pub logical: ObsMask,
    /// Under the float metric, unnormalized joint mass
    /// `ln P(logical class, observed syndrome)`. Under `maxlog_int`, the
    /// quantized best-route mass for this logical label divided by the metric
    /// scale.
    ///
    /// In float mode, subtract [`TrellisResult::log_evidence`] to obtain this
    /// label's log posterior probability within the retained terminal mass.
    pub log_mass: f64,
}

/// Completeness status of one successful trellis decode.
///
/// `NoPath` remains a [`DecoderError`]. This envelope will gain a budget arm
/// only when the decoder has an actual budget mechanism.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrellisStatus {
    /// No state was discarded by pruning, so the retained result is exact.
    Exact,
    /// At least one state was discarded by pruning.
    Pruned {
        /// Whether the configured frontier-width cap discarded any state.
        k_capped: bool,
        /// Whether the configured log-mass window discarded any state.
        delta_pruned: bool,
    },
}

/// Result of one trellis decode.
#[derive(Clone, Debug, PartialEq)]
pub struct TrellisResult {
    /// Predicted logical-observable flip mask.
    pub predicted: ObsMask,
    /// Under the float metric, the logarithm of the total retained joint mass
    /// over all terminal logical labels, approximating
    /// `ln P(observed syndrome)` when pruning is enabled. Under `maxlog_int`,
    /// the winning label's quantized best-route mass divided by the scale.
    ///
    /// The winning label's own log mass is [`Self::logical_masses`]'s first
    /// entry.
    pub log_evidence: f64,
    /// Difference between the winning and runner-up terminal masses, if a
    /// runner-up exists. Under `maxlog_int`, this is a best-route margin.
    ///
    /// This is retained-mass telemetry, not a certified confidence measure.
    /// In the M6 BB144 experiment, none of 300 shots retained a runner-up at
    /// `k = 2`, so the gap is not used to trigger escalation.
    pub runner_up_gap: Option<f64>,
    /// Largest retained frontier size, including the initial boundary state.
    pub peak_retained_states: usize,
    /// Number of probabilistic binary mechanisms or non-forced factors processed.
    pub processed_columns: usize,
    /// Number of candidate branch evaluations before closing-detector checks:
    /// two per retained state for a binary column, or one per outcome and
    /// retained state for an N-ary column.
    /// For an escalated `BpTrellis` result, this is the total across
    /// the base attempt and every attempted rung.
    pub transitions: u64,
    /// Number of merged boundary states discarded across all pruning calls in
    /// the successful rung.
    pub dropped_states: u64,
    /// Log-sum-exp of the log masses of all states discarded by float pruning,
    /// or negative infinity when no state was discarded. Under `maxlog_int`,
    /// this is the largest dropped quantized route mass divided by the scale.
    ///
    /// This accounts for retained prefix mass discarded at pruning time. It is
    /// not a bound on true lost posterior mass: a state dropped early would
    /// otherwise have branched through later columns.
    /// For an escalated `BpTrellis` result, this covers only the
    /// successful rung.
    pub dropped_log_mass: f64,
    /// Wall-clock seconds spent producing BP-informed suffix scores for this
    /// shot. This is zero when BP scoring is disabled or pruning cannot run.
    /// Escalation reuses these scores, so this time is attached once.
    pub bp_seconds: f64,
    /// Number of BP refreshes run for this shot.
    pub bp_runs: u32,
    /// Number of escalation rungs attempted before success.
    ///
    /// Zero means the base decode succeeded. [`TrellisDecoder`] results and
    /// successful ladder-free `BpTrellis` decodes also report zero.
    pub escalation_rungs_used: u32,
    /// Whether the successful decode was exact or which pruning mechanisms
    /// discarded at least one state. For an escalated `BpTrellis` result, this
    /// is the successful rung's status.
    pub status: TrellisStatus,
    /// Retained terminal masses, ordered by mass descending and numeric label
    /// ascending. Under `maxlog_int`, each entry is the label's best-route
    /// mass rather than a sum over routes.
    pub logical_masses: Vec<TrellisLogicalMass>,
}

#[derive(Clone, Debug)]
struct Column {
    detector_toggle: Vec<u64>,
    logical_toggle: Vec<u64>,
    close_mask: Vec<u64>,
    active_mask: Vec<u64>,
    suffix_compatibility: Vec<SuffixRow>,
    log_odds: f64,
    log_one_minus_probability: f64,
    log_odds_int: i64,
    log_one_minus_probability_int: i64,
}

#[derive(Clone, Debug)]
enum Kernel {
    Binary(Vec<Column>),
    Nary(Vec<FactorColumn>),
}

#[derive(Clone, Debug)]
struct FactorColumn {
    outcomes: Vec<ColumnOutcome>,
    close_mask: Vec<u64>,
    active_mask: Vec<u64>,
    suffix_compatibility: Vec<SuffixRow>,
}

#[derive(Clone, Debug)]
struct ColumnOutcome {
    detector_toggle: Vec<u64>,
    logical_toggle: Vec<u64>,
    probability: f64,
    log_prior: f64,
    log_prior_int: i64,
}

/// A 12-byte word/bit record keeps epoch indices at 32 bits.
/// Columns retain these references in ascending detector order for scoring.
#[derive(Clone, Debug)]
struct SuffixRow {
    word: u32,
    epoch: u32,
    bit: u8,
}

/// Initial epochs precede updates in reverse-column, ascending-detector order.
/// A column's rows reference epochs BEFORE that column's toggle is applied.
#[derive(Clone, Debug)]
struct SuffixEpoch {
    detector: usize,
    column_index: Option<usize>,
}

#[derive(Clone, Debug)]
struct SuffixLogProbabilities {
    zero: f64,
    one: f64,
    zero_int: i64,
    one_int: i64,
}

impl SuffixLogProbabilities {
    fn new(eta: f64, int_metric_scale: Option<i32>) -> Self {
        let log_probability_zero = libm::log(1.0_f64.midpoint(eta));
        let log_probability_one = libm::log(1.0_f64.midpoint(-eta));
        Self {
            zero: log_probability_zero,
            one: log_probability_one,
            // Float tables never read these sentinels; the buffer's scale tag
            // is asserted before integer suffix scoring.
            zero_int: int_metric_scale.map_or(INT_METRIC_NEG_INF, |scale| {
                quantize_metric(log_probability_zero, scale)
            }),
            one_int: int_metric_scale.map_or(INT_METRIC_NEG_INF, |scale| {
                quantize_metric(log_probability_one, scale)
            }),
        }
    }
}

#[derive(Clone, Debug)]
struct SuffixValues {
    probabilities: Vec<SuffixLogProbabilities>,
    int_metric_scale: Option<i32>,
}

impl SuffixValues {
    fn new(epoch_count: usize, int_metric_scale: Option<i32>) -> Self {
        Self {
            probabilities: vec![SuffixLogProbabilities::new(1.0, int_metric_scale); epoch_count],
            int_metric_scale,
        }
    }

    fn fill(
        &mut self,
        epochs: &[SuffixEpoch],
        row_moments: &mut [f64],
        moment: impl Fn(usize, usize) -> f64,
    ) {
        assert_eq!(self.probabilities.len(), epochs.len());
        row_moments.fill(1.0);
        // Epoch order encodes the original reverse scan, including the order
        // of detector updates within each column. Never regroup products.
        for (epoch, probability) in epochs.iter().zip(&mut self.probabilities) {
            if let Some(column_index) = epoch.column_index {
                row_moments[epoch.detector] *= moment(column_index, epoch.detector);
            }
            *probability =
                SuffixLogProbabilities::new(row_moments[epoch.detector], self.int_metric_scale);
        }
    }
}

#[derive(Clone, Copy)]
struct SuffixCompatibility<'a> {
    rows: &'a [SuffixRow],
    values: &'a SuffixValues,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StateKey {
    active_syndrome: Vec<u64>,
    logical: Vec<u64>,
}

impl Ord for StateKey {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_words_as_unsigned(&self.active_syndrome, &other.active_syndrome)
            .then_with(|| compare_words_as_unsigned(&self.logical, &other.logical))
    }
}

impl PartialOrd for StateKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
struct Candidate {
    key: StateKey,
    log_mass: f64,
}

struct PruneResult<M> {
    dropped_states: u64,
    dropped_log_mass: M,
    k_capped: bool,
    delta_pruned: bool,
}

/// Fixed-stride keys: detector words followed by logical words. Mass count is
/// the state count, including when both word counts (and the stride) are zero.
#[derive(Clone, Debug, Default)]
struct StateBuffer<M> {
    words: Vec<u64>,
    masses: Vec<M>,
}

impl<M: Copy> StateBuffer<M> {
    fn clear(&mut self, stride: usize) {
        debug_assert_eq!(self.words.len(), self.masses.len() * stride);
        self.words.clear();
        self.masses.clear();
    }

    fn key(&self, index: usize, stride: usize) -> &[u64] {
        debug_assert_eq!(self.words.len(), self.masses.len() * stride);
        &self.words[index * stride..(index + 1) * stride]
    }

    fn copy_state(&mut self, source: &Self, index: usize, stride: usize) {
        debug_assert_eq!(self.words.len(), self.masses.len() * stride);
        self.words.extend_from_slice(source.key(index, stride));
        self.masses.push(source.masses[index]);
        debug_assert_eq!(self.words.len(), self.masses.len() * stride);
    }
}

/// Both arenas and all sorting/scoring workspaces survive columns and shots.
/// Their buffers keep their high-water capacity for the decoder's lifetime by design.
/// `parent` is always in ascending key order at a column boundary: expansion
/// order determines the load-bearing left-to-right floating-point merge fold.
#[derive(Clone, Debug, Default)]
struct FrontierScratch<M> {
    parent: StateBuffer<M>,
    branches: StateBuffer<M>,
    indices: Vec<usize>,
    scores: Vec<M>,
    transposed: Vec<u64>,
    retained: Vec<bool>,
    detector_words: usize,
    stride: usize,
}

impl<M: Copy> FrontierScratch<M> {
    fn reset(&mut self, syndrome: &[u64], logical: &[u64], touched: &[u64], mass: M) {
        self.parent.clear(self.stride);
        self.branches.clear(self.stride);
        self.detector_words = syndrome.len();
        self.stride = syndrome.len() + logical.len();
        self.parent.words.extend_from_slice(syndrome);
        and_assign(&mut self.parent.words, touched);
        self.parent.words.extend_from_slice(logical);
        self.parent.masses.push(mass);
        debug_assert_eq!(
            self.parent.words.len(),
            self.parent.masses.len() * self.stride
        );
    }

    /// Stable sorting preserves arrival order within each equal-key run. The
    /// first arrival is installed verbatim, then each subsequent arrival applies
    /// `fold(accumulated, next)` in arrival order.
    fn merge(&mut self, fold: impl Fn(M, M) -> M) {
        self.indices.clear();
        self.indices.extend(0..self.branches.masses.len());
        self.indices.sort_by(|&left, &right| {
            compare_state_words(
                self.branches.key(left, self.stride),
                self.branches.key(right, self.stride),
                self.detector_words,
            )
        });
        self.parent.clear(self.stride);
        for &index in &self.indices {
            let count = self.parent.masses.len();
            if count != 0
                && self.parent.key(count - 1, self.stride) == self.branches.key(index, self.stride)
            {
                self.parent.masses[count - 1] =
                    fold(self.parent.masses[count - 1], self.branches.masses[index]);
            } else {
                self.parent.copy_state(&self.branches, index, self.stride);
            }
        }
    }

    /// Transpose the inclusive detector-word span read by the scoring rows.
    fn transpose_detectors(&mut self, first_word: usize, last_word: usize) {
        debug_assert!(first_word <= last_word);
        debug_assert!(last_word < self.detector_words);
        debug_assert_eq!(
            self.parent.words.len(),
            self.parent.masses.len() * self.stride
        );
        let count = self.parent.masses.len();
        self.transposed
            .resize((last_word - first_word + 1) * count, 0);
        if count == 0 {
            return;
        }
        for (word, candidates) in self.transposed.chunks_exact_mut(count).enumerate() {
            for (destination, key) in candidates
                .iter_mut()
                .zip(self.parent.words.chunks_exact(self.stride))
            {
                *destination = key[first_word + word];
            }
        }
    }

    /// Copy in merged key order, never score order, for the next expansion.
    fn retain(&mut self) {
        self.branches.clear(self.stride);
        for (index, &keep) in self.retained.iter().enumerate() {
            if keep {
                self.branches.copy_state(&self.parent, index, self.stride);
            }
        }
        std::mem::swap(&mut self.parent, &mut self.branches);
    }
}

fn compare_state_words(left: &[u64], right: &[u64], detector_words: usize) -> Ordering {
    compare_words_as_unsigned(&left[..detector_words], &right[..detector_words])
        .then_with(|| compare_words_as_unsigned(&left[detector_words..], &right[detector_words..]))
}

/// A column's compatibility masks and syndrome are shared by every branch.
struct BranchContext<'a> {
    detector_words: usize,
    close_mask: &'a [u64],
    active_mask: &'a [u64],
    observed: &'a [u64],
}

impl BranchContext<'_> {
    fn emit<M: Copy>(
        &self,
        destination: &mut StateBuffer<M>,
        parent: &[u64],
        toggles: Option<(&[u64], &[u64])>,
        mass: M,
        transitions: &mut u64,
    ) {
        debug_assert_eq!(self.close_mask.len(), self.detector_words);
        debug_assert_eq!(self.active_mask.len(), self.detector_words);
        debug_assert_eq!(self.observed.len(), self.detector_words);
        debug_assert_eq!(
            destination.words.len(),
            destination.masses.len() * parent.len()
        );
        *transitions += 1;
        let start = destination.words.len();
        destination.words.extend_from_slice(parent);
        let (syndrome, logical) = destination.words[start..].split_at_mut(self.detector_words);
        if let Some((detector_toggle, logical_toggle)) = toggles {
            xor_assign(syndrome, detector_toggle);
            xor_assign(logical, logical_toggle);
        }
        if syndrome
            .iter()
            .zip(self.observed)
            .zip(self.close_mask)
            .any(|((&accumulated, &expected), &closing)| (accumulated ^ expected) & closing != 0)
        {
            destination.words.truncate(start);
            debug_assert_eq!(
                destination.words.len(),
                destination.masses.len() * parent.len()
            );
            return;
        }
        and_assign(syndrome, self.active_mask);
        destination.masses.push(mass);
        debug_assert_eq!(
            destination.words.len(),
            destination.masses.len() * parent.len()
        );
    }
}

#[derive(Clone, Copy)]
struct MaxLogDecodeStats {
    peak_retained_states: usize,
    processed_columns: usize,
    transitions: u64,
    dropped_states: u64,
    dropped_log_mass: i64,
    bp_seconds: f64,
    k_capped: bool,
    delta_pruned: bool,
}

#[derive(Clone, Debug)]
struct BpScoreState {
    scratch: BpScratch,
    posterior: Vec<f64>,
    residual_syndrome: Vec<u8>,
    suffix_values: SuffixValues,
    row_moments: Vec<f64>,
    column_moments: Vec<f64>,
}

impl BpScoreState {
    fn new(graph: &BpGraph, suffix_values: SuffixValues, num_detectors: usize) -> Self {
        debug_assert_eq!(graph.check_count(), num_detectors);
        let scratch = BpScratch::new(graph);
        let posterior = vec![0.0; graph.mechanism_count()];
        let residual_syndrome = vec![0; graph.check_count()];
        let row_moments = vec![1.0; num_detectors];
        let column_moments = vec![0.0; graph.mechanism_count()];
        Self {
            scratch,
            posterior,
            residual_syndrome,
            suffix_values,
            row_moments,
            column_moments,
        }
    }
}

type RawColumn = (Vec<u64>, Vec<u64>, f64);

/// Structured outcome of one trellis engine attempt.
///
/// This preserves work telemetry for a no-path outcome so higher-level decode
/// policies can retry without duplicating engine logic.
pub enum TrellisDecodeAttempt {
    /// The attempt retained at least one terminal state.
    Success(TrellisResult),
    /// The attempt retained no path for the observed syndrome.
    NoPath {
        /// Error reported when no higher-level retry succeeds.
        error: DecoderError,
        /// Candidate branch evaluations performed by this attempt.
        transitions: u64,
        /// States discarded before the empty column.
        dropped_states: u64,
        /// Wall-clock seconds the shot's single BP refresh took, repeated on
        /// every attempt of that shot; never sum it across attempts.
        bp_seconds: f64,
    },
    /// A non-retryable decoding error.
    Error(DecoderError),
}

impl TrellisDecodeAttempt {
    fn into_result(self) -> Result<TrellisResult, DecoderError> {
        match self {
            Self::Success(result) => Ok(result),
            Self::NoPath { error, .. } | Self::Error(error) => Err(error),
        }
    }
}

/// Ordered, pruned dynamic-programming decoder for sparse DEMs and factor models.
#[derive(Clone, Debug)]
pub struct TrellisDecoder {
    model: Arc<TrellisModel>,
    scratch: TrellisScratch,
}

#[derive(Clone, Debug)]
struct TrellisScratch {
    observed: Vec<u64>,
    prepared: Option<(bool, f64)>,
    bp_refreshes: u64,
    bp_score: Option<BpScoreState>,
    float_progress: BinaryProgress,
    int_frontier: FrontierScratch<i64>,
}

impl TrellisScratch {
    fn new(model: &TrellisModel) -> Self {
        Self {
            observed: Vec::new(),
            prepared: None,
            bp_refreshes: 0,
            bp_score: model.bp_graph.as_ref().map(|graph| {
                BpScoreState::new(
                    graph,
                    SuffixValues::new(
                        model.suffix_epochs.len(),
                        model.suffix_values.int_metric_scale,
                    ),
                    model.num_detectors,
                )
            }),
            float_progress: BinaryProgress::default(),
            int_frontier: FrontierScratch::default(),
        }
    }
}

/// Scratch and cumulative telemetry across one or more binary column ranges.
#[derive(Clone, Debug)]
struct BinaryProgress {
    frontier: FrontierScratch<f64>,
    transitions: u64,
    dropped_states: u64,
    dropped_log_mass: f64,
    k_capped: bool,
    delta_pruned: bool,
    peak_retained_states: usize,
}

impl Default for BinaryProgress {
    fn default() -> Self {
        let frontier = FrontierScratch::default();
        let peak_retained_states = frontier.parent.masses.len();
        Self {
            frontier,
            transitions: 0,
            dropped_states: 0,
            dropped_log_mass: f64::NEG_INFINITY,
            k_capped: false,
            delta_pruned: false,
            peak_retained_states,
        }
    }
}

impl BinaryProgress {
    fn reset(&mut self, model: &TrellisModel) {
        self.frontier.reset(
            &model.forced_syndrome,
            &model.forced_logical,
            &model.touched_detectors,
            0.0,
        );
        self.transitions = 0;
        self.dropped_states = 0;
        self.dropped_log_mass = f64::NEG_INFINITY;
        self.k_capped = false;
        self.delta_pruned = false;
        self.peak_retained_states = self.frontier.parent.masses.len();
    }
}

#[derive(Clone, Copy, Debug)]
enum BinaryFailure {
    NoPath,
    UnusableScores,
}

impl BinaryFailure {
    fn error(self) -> DecoderError {
        match self {
            Self::NoPath => unexplainable_error(),
            Self::UnusableScores => DecoderError::InternalError(
                "pruning emptied a nonempty frontier; candidate scores were not finite".into(),
            ),
        }
    }
}

#[derive(Debug)]
struct TrellisModel {
    bp_graph: Option<BpGraph>,
    config: TrellisConfig,
    kernel: Kernel,
    num_detectors: usize,
    detector_words: usize,
    logical_words: usize,
    touched_detectors: Vec<u64>,
    forced_syndrome: Vec<u64>,
    forced_logical: Vec<u64>,
    suffix_epochs: Vec<SuffixEpoch>,
    suffix_values: SuffixValues,
    build_seconds: f64,
}

impl TrellisDecoder {
    /// Construct a decoder from a sparse detector error model.
    ///
    /// Zero-probability mechanisms are discarded and probability-one mechanisms
    /// are folded into the initial state after validating the optional ordering
    /// permutation. When configured, indistinguishable probabilistic mechanisms
    /// are merged in their ordered sequence before deadline and suffix data are
    /// constructed. All indices and probabilities are checked.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] for invalid pruning
    /// parameters, probabilities, indices, column order, or suffix-record widths.
    ///
    /// # Panics
    ///
    /// Panics if the internal post-filter BP graph and DP column counts differ.
    pub fn from_sparse_dem(dem: &SparseDem, config: TrellisConfig) -> Result<Self, DecoderError> {
        let build_started = Instant::now();
        validate_config(&config, dem.mechanisms.len())?;

        let detector_words = checked_detector_words(dem.num_detectors)? as usize;
        let logical_words = words_for(dem.num_observables);
        let order = config
            .column_order
            .clone()
            .unwrap_or_else(|| (0..dem.mechanisms.len()).collect());
        let mut raw_columns: Vec<RawColumn> = Vec::with_capacity(dem.mechanisms.len());
        #[cfg(debug_assertions)]
        let mut probabilistic_order = Vec::with_capacity(dem.mechanisms.len());
        let mut forced_syndrome = vec![0; detector_words];
        let mut forced_logical = vec![0; logical_words];

        for mechanism_index in order {
            let (probability, detectors, observables) = &dem.mechanisms[mechanism_index];
            validate_probability(*probability, mechanism_index)?;
            validate_indices(
                detectors,
                dem.num_detectors,
                "detector",
                "mechanism",
                mechanism_index,
            )?;
            validate_indices(
                observables,
                dem.num_observables,
                "observable",
                "mechanism",
                mechanism_index,
            )?;
            if *probability == 0.0 {
                continue;
            }

            let detector_toggle = indices_to_words(detectors, detector_words);
            let logical_toggle = indices_to_words(observables, logical_words);
            if probability.to_bits() == 1.0_f64.to_bits() {
                xor_assign(&mut forced_syndrome, &detector_toggle);
                xor_assign(&mut forced_logical, &logical_toggle);
                continue;
            }

            raw_columns.push((detector_toggle, logical_toggle, *probability));
            #[cfg(debug_assertions)]
            probabilistic_order.push(mechanism_index);
        }

        #[cfg(debug_assertions)]
        {
            let expected_probabilistic_order: Vec<usize> = dem
                .mechanisms
                .iter()
                .enumerate()
                .filter_map(|(index, (probability, _, _))| {
                    (*probability != 0.0 && probability.to_bits() != 1.0_f64.to_bits())
                        .then_some(index)
                })
                .collect();
            let mut sorted_probabilistic_order = probabilistic_order;
            sorted_probabilistic_order.sort_unstable();
            debug_assert_eq!(sorted_probabilistic_order, expected_probabilistic_order);
        }

        if config.merge_indistinguishable {
            raw_columns = merge_indistinguishable_columns(raw_columns);
        }

        let bp_graph = if config.bp_score_iterations > 0
            && !(config.k == usize::MAX && config.delta.is_infinite())
        {
            // This graph is deliberately built from exactly the post-order,
            // post-zero/one-filter, post-merge column sequence consumed by the
            // DP, never from the raw DEM mechanisms. Posterior index j therefore
            // corresponds to DP column j.
            let bp_dem = SparseDem {
                mechanisms: raw_columns
                    .iter()
                    .map(|(detector_words, _, probability)| {
                        let detectors = set_bits(detector_words)
                            .map(|detector| {
                                u32::try_from(detector).map_err(|_| {
                                    DecoderError::InvalidConfiguration(format!(
                                        "detector index {detector} does not fit u32"
                                    ))
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok((*probability, detectors, Vec::new()))
                    })
                    .collect::<Result<Vec<_>, DecoderError>>()?,
                detector_coords: BTreeMap::new(),
                num_detectors: dem.num_detectors,
                num_observables: 0,
            };
            let graph = BpGraph::from_sparse_dem(&bp_dem)?;
            assert_eq!(
                graph.mechanism_count(),
                raw_columns.len(),
                "BP mechanisms must correspond one-for-one with DP columns"
            );
            Some(graph)
        } else {
            None
        };

        let mut touched_detectors = vec![0; detector_words];
        let mut last_touch = vec![None; dem.num_detectors];
        for (column_index, (detectors, _, _)) in raw_columns.iter().enumerate() {
            or_assign(&mut touched_detectors, detectors);
            for detector in set_bits(detectors) {
                last_touch[detector] = Some(column_index);
            }
        }

        // Seed with the forced contribution: detectors carrying a forced bit
        // must stay in every active mask until their closing column, or the
        // per-step projection would erase the bit before that column arrives.
        // Forced-only detectors have no probabilistic closing column and are
        // handled by the precheck instead, so they are not active DP state.
        let mut open_detectors = forced_syndrome.clone();
        and_assign(&mut open_detectors, &touched_detectors);
        let mut columns = Vec::with_capacity(raw_columns.len());
        let mut column_moments = Vec::with_capacity(raw_columns.len());
        for (column_index, (detector_toggle, logical_toggle, probability)) in
            raw_columns.into_iter().enumerate()
        {
            or_assign(&mut open_detectors, &detector_toggle);
            let mut close_mask = vec![0; detector_words];
            for (detector, &last) in last_touch.iter().enumerate() {
                if last == Some(column_index) {
                    set_bit(&mut close_mask, detector);
                }
            }
            and_not_assign(&mut open_detectors, &close_mask);

            column_moments.push(1.0 - 2.0 * probability);
            let log_odds = libm::log(probability / (1.0 - probability));
            let log_one_minus_probability = libm::log(1.0 - probability);
            columns.push(Column {
                detector_toggle,
                logical_toggle,
                close_mask,
                active_mask: open_detectors.clone(),
                suffix_compatibility: Vec::new(),
                log_odds,
                log_one_minus_probability,
                log_odds_int: quantize_metric(log_odds, config.int_metric_scale),
                log_one_minus_probability_int: quantize_metric(
                    log_one_minus_probability,
                    config.int_metric_scale,
                ),
            });
        }

        let (suffix_epochs, suffix_tables) = build_suffix_epochs(
            columns
                .iter()
                .map(|column| (&column.active_mask[..], &column.detector_toggle[..])),
            dem.num_detectors,
        )?;
        for (column, rows) in columns.iter_mut().zip(suffix_tables) {
            column.suffix_compatibility = rows;
        }
        let mut suffix_values = SuffixValues::new(
            suffix_epochs.len(),
            (config.metric_mode == MetricMode::MaxLogInt).then_some(config.int_metric_scale),
        );
        suffix_values.fill(
            &suffix_epochs,
            &mut vec![1.0; dem.num_detectors],
            |column, _| column_moments[column],
        );

        debug_assert_model_invariants(&columns, &touched_detectors);
        let mut model = TrellisModel {
            config,
            kernel: Kernel::Binary(columns),
            num_detectors: dem.num_detectors,
            detector_words,
            logical_words,
            touched_detectors,
            forced_syndrome,
            forced_logical,
            bp_graph,
            suffix_epochs,
            suffix_values,
            build_seconds: 0.0,
        };
        let scratch = TrellisScratch::new(&model);
        model.build_seconds = build_started.elapsed().as_secs_f64();
        Ok(Self {
            scratch,
            model: Arc::new(model),
        })
    }

    /// Construct a decoder from a validated multi-outcome factor model.
    ///
    /// Binary-shaped models delegate to [`Self::from_sparse_dem`] and are
    /// bitwise-identical to the equivalent sparse DEM parameterized by each
    /// toggle probability. A stored baseline may differ from that DEM's implied
    /// complement only when the induced relative baseline log-mass error is
    /// within the engine's `1e-9` acceptance tolerance. Models containing any
    /// genuinely multi-outcome factor use the N-ary kernel.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] for invalid pruning or
    /// ordering configuration, or when binary-only BP scoring or mechanism
    /// merging is requested for a genuinely N-ary model, or if suffix-record
    /// widths exceed their limits.
    pub fn from_factor_model(
        model: &FactorModel,
        config: TrellisConfig,
    ) -> Result<Self, DecoderError> {
        let normalized_factors = model.normalized_factors();
        if normalized_factors
            .iter()
            .all(|factor| !matches!(factor, NormalizedFactor::Nary(_)))
        {
            let mechanisms = normalized_factors
                .into_iter()
                .map(|factor| match factor {
                    NormalizedFactor::Forced(outcome) => {
                        (1.0, outcome.detectors, outcome.observables)
                    }
                    NormalizedFactor::Binary { toggle, .. } => {
                        (toggle.probability, toggle.detectors, toggle.observables)
                    }
                    NormalizedFactor::Nary(_) => unreachable!("model was classified binary-shaped"),
                })
                .collect();
            let dem = SparseDem {
                mechanisms,
                detector_coords: BTreeMap::new(),
                num_detectors: model.num_detectors(),
                num_observables: model.num_observables(),
            };
            return Self::from_sparse_dem(&dem, config);
        }

        if config.bp_score_iterations > 0 && !(config.k == usize::MAX && config.delta.is_infinite())
        {
            return Err(DecoderError::InvalidConfiguration(
                "BP-guided pruning requires a binary model".into(),
            ));
        }
        if config.merge_indistinguishable {
            return Err(DecoderError::InvalidConfiguration(
                "indistinguishable-mechanism merging is defined for binary mechanisms only".into(),
            ));
        }
        validate_config(&config, model.factors().len())?;

        let build_started = Instant::now();
        let detector_words = checked_detector_words(model.num_detectors())? as usize;
        let logical_words = words_for(model.num_observables());
        let order = config
            .column_order
            .clone()
            .unwrap_or_else(|| (0..model.factors().len()).collect());
        let mut forced_syndrome = vec![0; detector_words];
        let mut forced_logical = vec![0; logical_words];
        let mut raw_columns: Vec<Vec<ColumnOutcome>> = Vec::with_capacity(model.factors().len());
        let mut normalized_factors: Vec<Option<NormalizedFactor>> =
            normalized_factors.into_iter().map(Some).collect();

        for factor_index in order {
            let factor = normalized_factors
                .get_mut(factor_index)
                .and_then(Option::take)
                .ok_or_else(|| {
                    DecoderError::InternalError(
                        "validated column_order did not select each normalized factor once".into(),
                    )
                })?;
            match factor {
                NormalizedFactor::Forced(outcome) => {
                    let detector_toggle = indices_to_words(&outcome.detectors, detector_words);
                    let logical_toggle = indices_to_words(&outcome.observables, logical_words);
                    xor_assign(&mut forced_syndrome, &detector_toggle);
                    xor_assign(&mut forced_logical, &logical_toggle);
                }
                NormalizedFactor::Binary { outcomes, .. } | NormalizedFactor::Nary(outcomes) => {
                    raw_columns.push(
                        outcomes
                            .into_iter()
                            .map(|outcome| {
                                column_outcome(
                                    &outcome,
                                    detector_words,
                                    logical_words,
                                    config.int_metric_scale,
                                )
                            })
                            .collect(),
                    );
                }
            }
        }

        let mut touched_detectors = vec![0; detector_words];
        let mut last_touch = vec![None; model.num_detectors()];
        let supports: Vec<Vec<u64>> = raw_columns
            .iter()
            .enumerate()
            .map(|(column_index, outcomes)| {
                let mut support = vec![0; detector_words];
                for outcome in outcomes {
                    or_assign(&mut support, &outcome.detector_toggle);
                }
                or_assign(&mut touched_detectors, &support);
                for detector in set_bits(&support) {
                    last_touch[detector] = Some(column_index);
                }
                support
            })
            .collect();

        let mut open_detectors = forced_syndrome.clone();
        and_assign(&mut open_detectors, &touched_detectors);
        let mut columns = Vec::with_capacity(raw_columns.len());
        for (column_index, (outcomes, support)) in
            raw_columns.into_iter().zip(&supports).enumerate()
        {
            or_assign(&mut open_detectors, support);
            let mut close_mask = vec![0; detector_words];
            for (detector, &last) in last_touch.iter().enumerate() {
                if last == Some(column_index) {
                    set_bit(&mut close_mask, detector);
                }
            }
            and_not_assign(&mut open_detectors, &close_mask);
            columns.push(FactorColumn {
                outcomes,
                close_mask,
                active_mask: open_detectors.clone(),
                suffix_compatibility: Vec::new(),
            });
        }

        let (suffix_epochs, suffix_tables) = build_suffix_epochs(
            columns
                .iter()
                .zip(&supports)
                .map(|(column, support)| (&column.active_mask[..], &support[..])),
            model.num_detectors(),
        )?;
        for (column, rows) in columns.iter_mut().zip(suffix_tables) {
            column.suffix_compatibility = rows;
        }
        let mut suffix_values = SuffixValues::new(
            suffix_epochs.len(),
            (config.metric_mode == MetricMode::MaxLogInt).then_some(config.int_metric_scale),
        );
        suffix_values.fill(
            &suffix_epochs,
            &mut vec![1.0; model.num_detectors()],
            |column, detector| {
                let word_index = detector / WORD_BITS;
                let bit_mask = 1 << (detector % WORD_BITS);
                let toggle_probability = columns[column]
                    .outcomes
                    .iter()
                    .filter(|outcome| outcome.detector_toggle[word_index] & bit_mask != 0)
                    .map(|outcome| outcome.probability)
                    .sum::<f64>()
                    .min(1.0);
                1.0 - 2.0 * toggle_probability
            },
        );
        debug_assert_factor_model_invariants(&columns, &touched_detectors);
        let mut model = TrellisModel {
            config,
            kernel: Kernel::Nary(columns),
            num_detectors: model.num_detectors(),
            detector_words,
            logical_words,
            touched_detectors,
            forced_syndrome,
            forced_logical,
            bp_graph: None,
            suffix_epochs,
            suffix_values,
            build_seconds: 0.0,
        };
        let scratch = TrellisScratch::new(&model);
        model.build_seconds = build_started.elapsed().as_secs_f64();
        Ok(Self {
            scratch,
            model: Arc::new(model),
        })
    }

    /// Wall-clock seconds spent constructing this model.
    #[must_use]
    pub fn build_seconds(&self) -> f64 {
        self.model.build_seconds
    }

    /// Addresses of this decoder's BP scoring state, if BP scoring is enabled.
    /// Exists so downstream crates can assert two decoders do not share state.
    #[doc(hidden)]
    #[must_use]
    pub fn bp_state_addrs(&self) -> Option<(usize, usize)> {
        self.scratch.bp_score.as_ref().map(|bp_score| {
            let graph = self
                .model
                .bp_graph
                .as_ref()
                .expect("BP scratch has a graph");
            let scratch: &BpScratch = &bp_score.scratch;
            (
                std::ptr::from_ref(graph).addr(),
                std::ptr::from_ref(scratch).addr(),
            )
        })
    }

    /// Parse a Stim-format detector error model and construct a decoder.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if parsing or decoder validation fails.
    pub fn from_dem_str(dem_str: &str, config: TrellisConfig) -> Result<Self, DecoderError> {
        let dem = SparseDem::from_dem_str(dem_str)?;
        Self::from_sparse_dem(&dem, config)
    }

    /// Decode a dense detector syndrome.
    ///
    /// Every nonzero byte is treated as a fired detector.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] for a dimension mismatch or when the syndrome
    /// is unexplainable with the retained frontier.
    pub fn decode(&mut self, syndrome: &[u8]) -> Result<TrellisResult, DecoderError> {
        self.decode_attempt(syndrome).into_result()
    }

    /// Decode while preserving no-path work telemetry for higher-level retry
    /// policies.
    #[must_use]
    pub fn decode_attempt(&mut self, syndrome: &[u8]) -> TrellisDecodeAttempt {
        match self.prepare(syndrome) {
            Ok(TrellisPrepared::Ready { .. }) => self.attempt(PruneParams {
                k: self.model.config.k,
                delta: self.model.config.delta,
            }),
            Ok(TrellisPrepared::Residual { .. }) => TrellisDecodeAttempt::NoPath {
                error: unexplainable_error(),
                transitions: 0,
                dropped_states: 0,
                bp_seconds: 0.0,
            },
            Err(error) => TrellisDecodeAttempt::Error(error),
        }
    }

    /// Check the syndrome and refresh BP once; the DP runs later in `attempt`.
    ///
    /// # Errors
    /// Returns a dimension error or a BP engine error. Any unsuccessful prepare
    /// invalidates the previous shot, including a residual outcome.
    pub fn prepare(&mut self, syndrome: &[u8]) -> Result<TrellisPrepared, DecoderError> {
        self.scratch.prepared = None;
        if syndrome.len() != self.model.num_detectors {
            return Err(DecoderError::InvalidDimensions {
                expected: self.model.num_detectors,
                actual: syndrome.len(),
            });
        }
        let observed = syndrome_to_words(syndrome, self.model.detector_words);
        for (word, ((&seen, &forced), &touched)) in observed
            .iter()
            .zip(&self.model.forced_syndrome)
            .zip(&self.model.touched_detectors)
            .enumerate()
        {
            let residual = (seen ^ forced) & !touched;
            if residual != 0 {
                return Ok(TrellisPrepared::Residual {
                    detector: word * WORD_BITS + residual.trailing_zeros() as usize,
                });
            }
        }
        let seconds = self
            .model
            .refresh_bp_suffix_values(&mut self.scratch, &observed)?;
        let bp_ran = seconds.is_some();
        let bp_seconds = seconds.unwrap_or(0.0);
        self.scratch.observed = observed;
        self.scratch.prepared = Some((bp_ran, bp_seconds));
        Ok(TrellisPrepared::Ready { bp_ran, bp_seconds })
    }

    /// Run the DP on the prepared shot with independently chosen pruning parameters.
    /// Invalid parameters return `Error(InvalidConfiguration)` before checking readiness.
    ///
    /// # Panics
    /// Panics unless the latest prepare completed `Ready`.
    #[must_use]
    pub fn attempt(&mut self, params: PruneParams) -> TrellisDecodeAttempt {
        if let Err(error) = params.validate(self.model.config.metric_mode) {
            return TrellisDecodeAttempt::Error(error);
        }
        if self.model.config.bp_score_iterations > 0
            && self.model.bp_graph.is_none()
            && !(params.k == usize::MAX && params.delta.is_infinite())
        {
            return TrellisDecodeAttempt::Error(DecoderError::InvalidConfiguration(
                "non-exact pruning parameters require a BP graph when bp_score_iterations > 0"
                    .into(),
            ));
        }
        let (bp_ran, _) = self
            .scratch
            .prepared
            .expect("attempt requires a Ready prepare");
        let mut attempt = self.model.decode_attempt(&mut self.scratch, params);
        if let TrellisDecodeAttempt::Success(result) = &mut attempt {
            result.bp_runs = u32::from(bp_ran);
        }
        attempt
    }

    /// Configured pruning parameters used by `decode_attempt`.
    #[must_use]
    pub fn prune_params(&self) -> PruneParams {
        PruneParams {
            k: self.model.config.k,
            delta: self.model.config.delta,
        }
    }

    /// Initial observable contribution of probability-one mechanisms.
    #[must_use]
    pub fn forced_observables(&self) -> ObsMask {
        ObsMask::from_words(&self.model.forced_logical)
    }

    /// Lifetime count of BP refreshes run by this decoder's scratch.
    #[must_use]
    pub fn bp_refreshes(&self) -> u64 {
        self.scratch.bp_refreshes
    }

    /// Share the immutable model with a decoder owning fresh scratch.
    #[must_use]
    pub fn fresh_worker(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            scratch: TrellisScratch::new(&self.model),
        }
    }

    /// Decode dense shots in input order using independent worker scratch.
    ///
    /// Workers are capped at one per shot, with one worker for an empty batch.
    ///
    /// # Errors
    /// Returns `InvalidConfiguration` for zero workers and `InternalError` for a pool creation failure.
    pub fn decode_batch(
        &self,
        shots: &[Vec<u8>],
        workers: usize,
    ) -> Result<Vec<TrellisDecodeAttempt>, DecoderError> {
        batch::decode_batch(shots, workers, || self.fresh_worker(), Self::decode_attempt)
    }
}

impl TrellisModel {
    fn decode_attempt(
        &self,
        scratch: &mut TrellisScratch,
        params: PruneParams,
    ) -> TrellisDecodeAttempt {
        match (self.config.metric_mode, &self.kernel) {
            (MetricMode::LogSumExpFloat, Kernel::Binary(_)) => {
                self.decode_attempt_binary(scratch, params)
            }
            (MetricMode::LogSumExpFloat, Kernel::Nary(_)) => {
                self.decode_attempt_nary(scratch, params)
            }
            (MetricMode::MaxLogInt, Kernel::Binary(_)) => {
                self.decode_attempt_binary_maxlog(scratch, params)
            }
            (MetricMode::MaxLogInt, Kernel::Nary(_)) => {
                self.decode_attempt_nary_maxlog(scratch, params)
            }
        }
    }

    fn decode_attempt_binary(
        &self,
        scratch: &mut TrellisScratch,
        params: PruneParams,
    ) -> TrellisDecodeAttempt {
        let observed = &scratch.observed;
        let bp_seconds = scratch.prepared.expect("prepared shot").1;
        let progress = &mut scratch.float_progress;
        progress.reset(self);
        let Kernel::Binary(columns) = &self.kernel else {
            unreachable!("binary decode called with N-ary kernel");
        };
        let suffix_values = scratch
            .bp_score
            .as_ref()
            .map_or(&self.suffix_values, |bp| &bp.suffix_values);
        if let Err(failure) =
            self.process_binary_range(progress, observed, suffix_values, 0..columns.len(), params)
        {
            return match failure {
                BinaryFailure::NoPath => TrellisDecodeAttempt::NoPath {
                    error: failure.error(),
                    transitions: progress.transitions,
                    dropped_states: progress.dropped_states,
                    bp_seconds,
                },
                BinaryFailure::UnusableScores => TrellisDecodeAttempt::Error(failure.error()),
            };
        }
        TrellisDecodeAttempt::Success(Self::finish_binary(progress, columns.len(), bp_seconds))
    }

    /// The single binary float column walk, shared by batch and streaming.
    fn process_binary_range(
        &self,
        progress: &mut BinaryProgress,
        observed: &[u64],
        suffix_values: &SuffixValues,
        range: std::ops::Range<usize>,
        params: PruneParams,
    ) -> Result<(), BinaryFailure> {
        let BinaryProgress {
            frontier,
            transitions,
            dropped_states,
            dropped_log_mass,
            k_capped,
            delta_pruned,
            peak_retained_states,
        } = progress;
        let Kernel::Binary(columns) = &self.kernel else {
            unreachable!("binary decode called with N-ary kernel");
        };
        for column in &columns[range] {
            debug_assert!((1..frontier.parent.masses.len()).all(|index| {
                compare_state_words(
                    frontier.parent.key(index - 1, frontier.stride),
                    frontier.parent.key(index, frontier.stride),
                    frontier.detector_words,
                ) == Ordering::Less
            }));
            frontier.branches.clear(frontier.stride);
            let branch_context = BranchContext {
                detector_words: frontier.detector_words,
                close_mask: &column.close_mask,
                active_mask: &column.active_mask,
                observed,
            };
            for (index, &log_mass) in frontier.parent.masses.iter().enumerate() {
                let state = frontier.parent.key(index, frontier.stride);
                let branch_base = log_mass + column.log_one_minus_probability;
                branch_context.emit(
                    &mut frontier.branches,
                    state,
                    None,
                    branch_base,
                    transitions,
                );
                branch_context.emit(
                    &mut frontier.branches,
                    state,
                    Some((&column.detector_toggle, &column.logical_toggle)),
                    branch_base + column.log_odds,
                    transitions,
                );
            }
            frontier.merge(logaddexp);
            if frontier.parent.masses.is_empty() {
                return Err(BinaryFailure::NoPath);
            }
            let suffix_compatibility = SuffixCompatibility {
                rows: &column.suffix_compatibility,
                values: suffix_values,
            };
            let pruned = prune(
                frontier,
                params.k,
                params.delta,
                self.config.score_alpha,
                suffix_compatibility,
                observed,
            );
            *dropped_states += pruned.dropped_states;
            *dropped_log_mass = logaddexp(*dropped_log_mass, pruned.dropped_log_mass);
            *k_capped |= pruned.k_capped;
            *delta_pruned |= pruned.delta_pruned;
            if frontier.parent.masses.is_empty() {
                // Pruning always retains the best-scoring candidate of a
                // nonempty set, so an empty frontier here means the scores
                // themselves were unusable (non-finite) -- an engine fault,
                // not an unexplainable syndrome. Genuine no-path exits happen
                // above, before pruning, when no branch is compatible.
                return Err(BinaryFailure::UnusableScores);
            }
            *peak_retained_states = (*peak_retained_states).max(frontier.parent.masses.len());
        }

        Ok(())
    }

    fn finish_binary(
        progress: &BinaryProgress,
        processed_columns: usize,
        bp_seconds: f64,
    ) -> TrellisResult {
        let BinaryProgress {
            ref frontier,
            transitions,
            dropped_states,
            dropped_log_mass,
            k_capped,
            delta_pruned,
            peak_retained_states,
        } = *progress;

        let mut terminal: Vec<Candidate> = frontier
            .parent
            .masses
            .iter()
            .enumerate()
            .map(|(index, &log_mass)| {
                let words = frontier.parent.key(index, frontier.stride);
                Candidate {
                    key: StateKey {
                        active_syndrome: words[..frontier.detector_words].to_vec(),
                        logical: words[frontier.detector_words..].to_vec(),
                    },
                    log_mass,
                }
            })
            .collect();
        sort_candidates(&mut terminal);
        let winner = &terminal[0];
        let log_evidence = terminal.iter().fold(f64::NEG_INFINITY, |total, candidate| {
            logaddexp(total, candidate.log_mass)
        });
        let logical_masses = terminal
            .iter()
            .map(|candidate| TrellisLogicalMass {
                logical: ObsMask::from_words(&candidate.key.logical),
                log_mass: candidate.log_mass,
            })
            .collect();
        let status = if dropped_states == 0 {
            TrellisStatus::Exact
        } else {
            TrellisStatus::Pruned {
                k_capped,
                delta_pruned,
            }
        };

        TrellisResult {
            predicted: ObsMask::from_words(&winner.key.logical),
            log_evidence,
            runner_up_gap: terminal
                .get(1)
                .map(|runner_up| winner.log_mass - runner_up.log_mass),
            peak_retained_states,
            processed_columns,
            transitions,
            dropped_states,
            dropped_log_mass,
            bp_seconds,
            bp_runs: 0,
            escalation_rungs_used: 0,
            status,
            logical_masses,
        }
    }

    fn decode_attempt_nary(
        &self,
        scratch: &mut TrellisScratch,
        params: PruneParams,
    ) -> TrellisDecodeAttempt {
        let observed = &scratch.observed;
        let frontier = &mut scratch.float_progress.frontier;
        frontier.reset(
            &self.forced_syndrome,
            &self.forced_logical,
            &self.touched_detectors,
            0.0,
        );
        let mut peak_retained_states = frontier.parent.masses.len();
        let mut transitions = 0;
        let mut dropped_states = 0;
        let mut dropped_log_mass = f64::NEG_INFINITY;
        let mut k_capped = false;
        let mut delta_pruned = false;

        let Kernel::Nary(columns) = &self.kernel else {
            unreachable!("N-ary decode called with binary kernel");
        };
        for column in columns {
            debug_assert!((1..frontier.parent.masses.len()).all(|index| {
                compare_state_words(
                    frontier.parent.key(index - 1, frontier.stride),
                    frontier.parent.key(index, frontier.stride),
                    frontier.detector_words,
                ) == Ordering::Less
            }));
            frontier.branches.clear(frontier.stride);
            let branch_context = BranchContext {
                detector_words: frontier.detector_words,
                close_mask: &column.close_mask,
                active_mask: &column.active_mask,
                observed,
            };
            for (index, &log_mass) in frontier.parent.masses.iter().enumerate() {
                let state = frontier.parent.key(index, frontier.stride);
                for outcome in &column.outcomes {
                    branch_context.emit(
                        &mut frontier.branches,
                        state,
                        Some((&outcome.detector_toggle, &outcome.logical_toggle)),
                        log_mass + outcome.log_prior,
                        &mut transitions,
                    );
                }
            }
            frontier.merge(logaddexp);
            if frontier.parent.masses.is_empty() {
                return TrellisDecodeAttempt::NoPath {
                    error: unexplainable_error(),
                    transitions,
                    dropped_states,
                    bp_seconds: 0.0,
                };
            }
            let pruned = prune(
                frontier,
                params.k,
                params.delta,
                self.config.score_alpha,
                SuffixCompatibility {
                    rows: &column.suffix_compatibility,
                    values: &self.suffix_values,
                },
                observed,
            );
            dropped_states += pruned.dropped_states;
            dropped_log_mass = logaddexp(dropped_log_mass, pruned.dropped_log_mass);
            k_capped |= pruned.k_capped;
            delta_pruned |= pruned.delta_pruned;
            if frontier.parent.masses.is_empty() {
                return TrellisDecodeAttempt::Error(DecoderError::InternalError(
                    "pruning emptied a nonempty frontier; candidate scores were not finite".into(),
                ));
            }
            peak_retained_states = peak_retained_states.max(frontier.parent.masses.len());
        }

        let mut terminal: Vec<Candidate> = frontier
            .parent
            .masses
            .iter()
            .enumerate()
            .map(|(index, &log_mass)| {
                let words = frontier.parent.key(index, frontier.stride);
                Candidate {
                    key: StateKey {
                        active_syndrome: words[..frontier.detector_words].to_vec(),
                        logical: words[frontier.detector_words..].to_vec(),
                    },
                    log_mass,
                }
            })
            .collect();
        sort_candidates(&mut terminal);
        let winner = &terminal[0];
        let log_evidence = terminal.iter().fold(f64::NEG_INFINITY, |total, candidate| {
            logaddexp(total, candidate.log_mass)
        });
        let logical_masses = terminal
            .iter()
            .map(|candidate| TrellisLogicalMass {
                logical: ObsMask::from_words(&candidate.key.logical),
                log_mass: candidate.log_mass,
            })
            .collect();
        let status = if dropped_states == 0 {
            TrellisStatus::Exact
        } else {
            TrellisStatus::Pruned {
                k_capped,
                delta_pruned,
            }
        };

        TrellisDecodeAttempt::Success(TrellisResult {
            predicted: ObsMask::from_words(&winner.key.logical),
            log_evidence,
            runner_up_gap: terminal
                .get(1)
                .map(|runner_up| winner.log_mass - runner_up.log_mass),
            peak_retained_states,
            processed_columns: columns.len(),
            transitions,
            dropped_states,
            dropped_log_mass,
            bp_seconds: 0.0,
            bp_runs: 0,
            escalation_rungs_used: 0,
            status,
            logical_masses,
        })
    }

    fn decode_attempt_binary_maxlog(
        &self,
        scratch: &mut TrellisScratch,
        params: PruneParams,
    ) -> TrellisDecodeAttempt {
        let observed = &scratch.observed;
        let bp_seconds = scratch.prepared.expect("prepared shot").1;
        let frontier = &mut scratch.int_frontier;
        frontier.reset(
            &self.forced_syndrome,
            &self.forced_logical,
            &self.touched_detectors,
            0_i64,
        );
        let mut peak_retained_states = frontier.parent.masses.len();
        let mut transitions = 0;
        let mut dropped_states = 0;
        let mut dropped_log_mass = INT_METRIC_NEG_INF;
        let mut k_capped = false;
        let mut delta_pruned = false;
        let scale = self.config.int_metric_scale;
        let delta_int = quantize_metric(params.delta, scale);
        debug_assert!(
            delta_int >= 0,
            "validate_config rejects negative and non-finite delta under maxlog_int"
        );
        let alpha_int = quantize_metric(self.config.score_alpha, scale);

        let suffix_values = scratch
            .bp_score
            .as_ref()
            .map_or(&self.suffix_values, |bp| &bp.suffix_values);
        let Kernel::Binary(columns) = &self.kernel else {
            unreachable!("binary max-log decode called with N-ary kernel");
        };
        for column in columns {
            debug_assert!((1..frontier.parent.masses.len()).all(|index| {
                compare_state_words(
                    frontier.parent.key(index - 1, frontier.stride),
                    frontier.parent.key(index, frontier.stride),
                    frontier.detector_words,
                ) == Ordering::Less
            }));
            frontier.branches.clear(frontier.stride);
            let branch_context = BranchContext {
                detector_words: frontier.detector_words,
                close_mask: &column.close_mask,
                active_mask: &column.active_mask,
                observed,
            };
            for (index, &log_mass) in frontier.parent.masses.iter().enumerate() {
                let state = frontier.parent.key(index, frontier.stride);
                let branch_base = int_metric_add(log_mass, column.log_one_minus_probability_int);
                branch_context.emit(
                    &mut frontier.branches,
                    state,
                    None,
                    branch_base,
                    &mut transitions,
                );
                branch_context.emit(
                    &mut frontier.branches,
                    state,
                    Some((&column.detector_toggle, &column.logical_toggle)),
                    int_metric_add(branch_base, column.log_odds_int),
                    &mut transitions,
                );
            }
            frontier.merge(i64::max);
            if frontier.parent.masses.is_empty() {
                return TrellisDecodeAttempt::NoPath {
                    error: unexplainable_error(),
                    transitions,
                    dropped_states,
                    bp_seconds,
                };
            }
            let suffix_compatibility = SuffixCompatibility {
                rows: &column.suffix_compatibility,
                values: suffix_values,
            };
            let pruned = prune_maxlog(
                frontier,
                params.k,
                delta_int,
                alpha_int,
                scale,
                suffix_compatibility,
                observed,
            );
            dropped_states += pruned.dropped_states;
            dropped_log_mass = dropped_log_mass.max(pruned.dropped_log_mass);
            k_capped |= pruned.k_capped;
            delta_pruned |= pruned.delta_pruned;
            peak_retained_states = peak_retained_states.max(frontier.parent.masses.len());
        }

        finish_maxlog_decode(
            frontier,
            scale,
            MaxLogDecodeStats {
                peak_retained_states,
                processed_columns: columns.len(),
                transitions,
                dropped_states,
                dropped_log_mass,
                bp_seconds,
                k_capped,
                delta_pruned,
            },
        )
    }

    fn decode_attempt_nary_maxlog(
        &self,
        scratch: &mut TrellisScratch,
        params: PruneParams,
    ) -> TrellisDecodeAttempt {
        let observed = &scratch.observed;
        let frontier = &mut scratch.int_frontier;
        frontier.reset(
            &self.forced_syndrome,
            &self.forced_logical,
            &self.touched_detectors,
            0_i64,
        );
        let mut peak_retained_states = frontier.parent.masses.len();
        let mut transitions = 0;
        let mut dropped_states = 0;
        let mut dropped_log_mass = INT_METRIC_NEG_INF;
        let mut k_capped = false;
        let mut delta_pruned = false;
        let scale = self.config.int_metric_scale;
        let delta_int = quantize_metric(params.delta, scale);
        debug_assert!(
            delta_int >= 0,
            "validate_config rejects negative and non-finite delta under maxlog_int"
        );
        let alpha_int = quantize_metric(self.config.score_alpha, scale);

        let Kernel::Nary(columns) = &self.kernel else {
            unreachable!("N-ary max-log decode called with binary kernel");
        };
        for column in columns {
            debug_assert!((1..frontier.parent.masses.len()).all(|index| {
                compare_state_words(
                    frontier.parent.key(index - 1, frontier.stride),
                    frontier.parent.key(index, frontier.stride),
                    frontier.detector_words,
                ) == Ordering::Less
            }));
            frontier.branches.clear(frontier.stride);
            let branch_context = BranchContext {
                detector_words: frontier.detector_words,
                close_mask: &column.close_mask,
                active_mask: &column.active_mask,
                observed,
            };
            for (index, &log_mass) in frontier.parent.masses.iter().enumerate() {
                let state = frontier.parent.key(index, frontier.stride);
                for outcome in &column.outcomes {
                    branch_context.emit(
                        &mut frontier.branches,
                        state,
                        Some((&outcome.detector_toggle, &outcome.logical_toggle)),
                        int_metric_add(log_mass, outcome.log_prior_int),
                        &mut transitions,
                    );
                }
            }
            frontier.merge(i64::max);
            if frontier.parent.masses.is_empty() {
                return TrellisDecodeAttempt::NoPath {
                    error: unexplainable_error(),
                    transitions,
                    dropped_states,
                    bp_seconds: 0.0,
                };
            }
            let pruned = prune_maxlog(
                frontier,
                params.k,
                delta_int,
                alpha_int,
                scale,
                SuffixCompatibility {
                    rows: &column.suffix_compatibility,
                    values: &self.suffix_values,
                },
                observed,
            );
            dropped_states += pruned.dropped_states;
            dropped_log_mass = dropped_log_mass.max(pruned.dropped_log_mass);
            k_capped |= pruned.k_capped;
            delta_pruned |= pruned.delta_pruned;
            peak_retained_states = peak_retained_states.max(frontier.parent.masses.len());
        }

        finish_maxlog_decode(
            frontier,
            scale,
            MaxLogDecodeStats {
                peak_retained_states,
                processed_columns: columns.len(),
                transitions,
                dropped_states,
                dropped_log_mass,
                bp_seconds: 0.0,
                k_capped,
                delta_pruned,
            },
        )
    }

    fn refresh_bp_suffix_values(
        &self,
        scratch: &mut TrellisScratch,
        observed: &[u64],
    ) -> Result<Option<f64>, DecoderError> {
        let Kernel::Binary(columns) = &self.kernel else {
            return Ok(None);
        };
        let Some(bp_score) = &mut scratch.bp_score else {
            return Ok(None);
        };

        let started = Instant::now();
        for (detector, residual) in bp_score.residual_syndrome.iter_mut().enumerate() {
            let word_index = detector / WORD_BITS;
            let bit_mask = 1 << (detector % WORD_BITS);
            *residual =
                u8::from((observed[word_index] ^ self.forced_syndrome[word_index]) & bit_mask != 0);
        }
        scratch.bp_refreshes += 1;
        min_sum_bp_into(
            self.bp_graph.as_ref().expect("BP scratch has a graph"),
            &bp_score.residual_syndrome,
            self.config.bp_score_iterations,
            BP_MIN_SUM_SCALE,
            true,
            &mut bp_score.scratch,
            &mut bp_score.posterior,
        )?;
        assert_eq!(
            bp_score.posterior.len(),
            columns.len(),
            "BP beliefs must correspond one-for-one with DP columns"
        );

        // These clamped probabilities are a heuristic for score arithmetic
        // only. The BP output never replaces the DEM probabilities used by
        // branch mass arithmetic.
        for (moment, &llr) in bp_score.column_moments.iter_mut().zip(&bp_score.posterior) {
            *moment = 1.0 - 2.0 * bp_score_probability(llr);
        }
        bp_score.suffix_values.fill(
            &self.suffix_epochs,
            &mut bp_score.row_moments,
            |column, _| bp_score.column_moments[column],
        );
        Ok(Some(started.elapsed().as_secs_f64()))
    }
}

impl ObservableDecoder for TrellisDecoder {
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        Ok(self.decode(syndrome)?.predicted)
    }

    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        if self.model.logical_words > 1 {
            return Err(DecoderError::InvalidConfiguration(
                "decoder has more than 64 observables; use decode_obs() for the wide mask".into(),
            ));
        }
        let decoded = self.decode(syndrome)?.predicted;
        Ok(decoded.words().first().copied().unwrap_or(0))
    }
}

type DeadlineKey = (usize, usize, usize, usize, usize);

fn deadline_order_for_sequence(
    supports: &[Vec<u32>],
    num_detectors: usize,
    item_noun: &str,
    sequence: &[usize],
) -> Result<Vec<usize>, DecoderError> {
    let sentinel = supports.len() + 1;
    let mut first_touch = vec![sentinel; num_detectors];
    let mut last_touch = vec![sentinel; num_detectors];

    for (position, &mechanism_index) in sequence.iter().enumerate() {
        let detectors = &supports[mechanism_index];
        validate_indices(
            detectors,
            num_detectors,
            "detector",
            item_noun,
            mechanism_index,
        )?;
        for &detector in detectors {
            let detector = detector as usize;
            first_touch[detector] = first_touch[detector].min(position);
            last_touch[detector] = position;
        }
    }

    let mut positions: Vec<usize> = (0..sequence.len()).collect();
    positions.sort_by_key(|&position| -> DeadlineKey {
        let mechanism_index = sequence[position];
        let detectors = &supports[mechanism_index];
        if detectors.is_empty() {
            return (sentinel, sentinel, sentinel, mechanism_index, position);
        }

        let (earliest_last, latest_last, earliest_first) = detectors.iter().fold(
            (sentinel, 0, sentinel),
            |(min_last, max_last, min_first), &detector| {
                let detector = detector as usize;
                (
                    min_last.min(last_touch[detector]),
                    max_last.max(last_touch[detector]),
                    min_first.min(first_touch[detector]),
                )
            },
        );
        (
            earliest_last,
            latest_last,
            earliest_first,
            mechanism_index,
            position,
        )
    });
    let ordered_sequence: Vec<usize> = positions
        .into_iter()
        .map(|position| sequence[position])
        .collect();
    #[cfg(debug_assertions)]
    {
        let mut sorted_input = sequence.to_vec();
        sorted_input.sort_unstable();
        let mut sorted_output = ordered_sequence.clone();
        sorted_output.sort_unstable();
        debug_assert_eq!(
            sorted_output, sorted_input,
            "generated order must permute input"
        );
    }
    Ok(ordered_sequence)
}

fn factor_supports(model: &FactorModel) -> Vec<Vec<u32>> {
    model
        .factors()
        .iter()
        .map(|factor| {
            factor
                .outcomes
                .iter()
                .flat_map(|outcome| outcome.detectors.iter().copied())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect()
        })
        .collect()
}

fn validate_config(config: &TrellisConfig, mechanism_count: usize) -> Result<(), DecoderError> {
    config.validate()?;
    if let Some(order) = &config.column_order {
        if order.len() != mechanism_count {
            return Err(DecoderError::InvalidConfiguration(format!(
                "column_order must be a permutation of 0..{mechanism_count}"
            )));
        }
        let mut seen = vec![false; mechanism_count];
        for &index in order {
            if index >= mechanism_count || seen[index] {
                return Err(DecoderError::InvalidConfiguration(format!(
                    "column_order must be a permutation of 0..{mechanism_count}"
                )));
            }
            seen[index] = true;
        }
    }
    Ok(())
}

fn validate_probability(probability: f64, index: usize) -> Result<(), DecoderError> {
    if !(0.0..=1.0).contains(&probability) {
        return Err(DecoderError::InvalidConfiguration(format!(
            "mechanism {index} probability must satisfy 0 <= p <= 1, got {probability}"
        )));
    }
    Ok(())
}

fn merge_indistinguishable_columns(raw_columns: Vec<RawColumn>) -> Vec<RawColumn> {
    let mut first_positions: BTreeMap<(Vec<u64>, Vec<u64>), usize> = BTreeMap::new();
    let mut merged_columns: Vec<RawColumn> = Vec::with_capacity(raw_columns.len());

    for (detectors, observables, probability) in raw_columns {
        let symptoms = (detectors.clone(), observables.clone());
        if let Some(&first_position) = first_positions.get(&symptoms) {
            let first_probability = &mut merged_columns[first_position].2;
            *first_probability = xor_combined_probability(*first_probability, probability);
        } else {
            first_positions.insert(symptoms, merged_columns.len());
            merged_columns.push((detectors, observables, probability));
        }
    }

    merged_columns
}

fn xor_combined_probability(first: f64, second: f64) -> f64 {
    debug_assert!(first > 0.0 && first < 1.0);
    debug_assert!(second > 0.0 && second < 1.0);
    let combined = first * (1.0 - second) + second * (1.0 - first);
    debug_assert!(
        combined > 0.0 && combined < 1.0,
        "the XOR probability of two probabilities in (0, 1) must remain in (0, 1)"
    );
    combined
}

fn column_outcome(
    outcome: &Outcome,
    detector_words: usize,
    logical_words: usize,
    int_metric_scale: i32,
) -> ColumnOutcome {
    let log_prior = libm::log(outcome.probability);
    ColumnOutcome {
        detector_toggle: indices_to_words(&outcome.detectors, detector_words),
        logical_toggle: indices_to_words(&outcome.observables, logical_words),
        probability: outcome.probability,
        log_prior,
        log_prior_int: quantize_metric(log_prior, int_metric_scale),
    }
}

fn validate_indices(
    indices: &[u32],
    upper_bound: usize,
    kind: &str,
    item_noun: &str,
    item_index: usize,
) -> Result<(), DecoderError> {
    let mut seen = std::collections::BTreeSet::new();
    for &index in indices {
        if index as usize >= upper_bound {
            return Err(DecoderError::InvalidConfiguration(format!(
                "{item_noun} {item_index} {kind} index {index} is out of range 0..{upper_bound}"
            )));
        }
        if !seen.insert(index) {
            return Err(DecoderError::InvalidConfiguration(format!(
                "{item_noun} {item_index} repeats {kind} index {index}"
            )));
        }
    }
    Ok(())
}

fn compare_words_as_unsigned(left: &[u64], right: &[u64]) -> Ordering {
    left.iter().rev().cmp(right.iter().rev())
}

/// Quantizes with an `f64` product and round-half-away-from-zero. Upstream's
/// intermediate `long double` width is x86-specific. Divergence is confined to
/// non-power-of-two scales whose exact product lands within one ulp of a half
/// boundary; PECOS owns these numerics, and fixture parity covers the shipped
/// power-of-two scales.
fn quantize_metric(value: f64, scale: i32) -> i64 {
    debug_assert!(
        value != f64::INFINITY,
        "positive-infinite metric input would saturate to the negative sentinel"
    );
    if !value.is_finite() {
        return INT_METRIC_NEG_INF;
    }
    let scaled = value * f64::from(scale);
    let lo = i64_to_f64(INT_METRIC_NEG_INF + 1);
    let hi = i64_to_f64(INT_METRIC_MAX);
    if scaled <= lo {
        INT_METRIC_NEG_INF
    } else if scaled >= hi {
        INT_METRIC_MAX
    } else {
        integral_f64_to_i64(scaled.round())
    }
}

fn i64_to_f64(value: i64) -> f64 {
    let high = i32::try_from(value >> 32).expect("the high i64 word must fit i32");
    let low =
        u32::try_from(value & i64::from(u32::MAX)).expect("the masked low i64 word must fit u32");
    f64::from(high) * 4_294_967_296.0 + f64::from(low)
}

fn integral_f64_to_i64(value: f64) -> i64 {
    debug_assert!(
        value.is_finite() && value.fract() == 0.0 && value.abs() <= i64_to_f64(INT_METRIC_MAX),
        "integral metric conversion requires a finite integer within the metric saturation range"
    );
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let biased_exponent = i32::try_from((bits >> 52) & 0x7ff).expect("exponent fits i32");
    let exponent = biased_exponent - 1023;
    if exponent < 0 {
        return 0;
    }
    let significand = (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52);
    let magnitude = if exponent >= 52 {
        significand << u32::try_from(exponent - 52).expect("nonnegative shift fits u32")
    } else {
        significand >> u32::try_from(52 - exponent).expect("nonnegative shift fits u32")
    };
    let magnitude = i64::try_from(magnitude).expect("quantized metric magnitude fits i64");
    if negative { -magnitude } else { magnitude }
}

fn saturating_i128_to_i64(value: i128) -> i64 {
    if value > i128::from(i64::MAX) {
        i64::MAX
    } else if value < i128::from(i64::MIN) {
        i64::MIN
    } else {
        i64::try_from(value).expect("range was checked above")
    }
}

fn fixed_mul_round(value: i64, multiplier: i64, scale: i64) -> i64 {
    if value <= INT_METRIC_NEG_INF / 2 {
        return INT_METRIC_NEG_INF;
    }
    if multiplier == 0 {
        return 0;
    }
    if multiplier == scale {
        return value;
    }
    let mut product = i128::from(value) * i128::from(multiplier);
    let divisor = i128::from(scale);
    let rounded = if product >= 0 {
        product += divisor / 2;
        product / divisor
    } else {
        product = -product + divisor / 2;
        -(product / divisor)
    };
    saturating_i128_to_i64(rounded)
}

fn fixed_mul_round_fast(value: i64, multiplier: i64, scale: i64) -> i64 {
    if value <= INT_METRIC_NEG_INF / 2 {
        return INT_METRIC_NEG_INF;
    }
    if multiplier == 0 {
        return 0;
    }
    if multiplier == scale {
        return value;
    }
    if scale == 1024 {
        let mut product = i128::from(value) * i128::from(multiplier);
        let rounded = if product >= 0 {
            product += 512;
            product >> 10
        } else {
            product = -product + 512;
            -(product >> 10)
        };
        return saturating_i128_to_i64(rounded);
    }
    fixed_mul_round(value, multiplier, scale)
}

fn int_metric_add(left: i64, right: i64) -> i64 {
    if left <= INT_METRIC_NEG_INF / 2 || right <= INT_METRIC_NEG_INF / 2 {
        INT_METRIC_NEG_INF
    } else {
        left.saturating_add(right)
    }
}

fn score_int_metric(log_mass: i64, parity: i64, alpha_int: i64, scale: i32) -> i64 {
    int_metric_add(
        log_mass,
        fixed_mul_round_fast(parity, alpha_int, i64::from(scale)),
    )
}

/// Rows must remain in ascending detector order: each candidate performs the
/// original left-to-right sum. Only the independent candidates may vectorize.
fn score_candidates(
    frontier: &mut FrontierScratch<f64>,
    score_alpha: f64,
    suffix_compatibility: SuffixCompatibility<'_>,
    observed: &[u64],
) {
    frontier.scores.clear();
    if score_alpha == 0.0 {
        frontier.scores.extend_from_slice(&frontier.parent.masses);
        return;
    }
    // Bit-identity requires the standard library's Sum neutral element and
    // left-fold order; the cfg(test) scalar oracle catches changes to either.
    let neutral = std::iter::empty::<f64>().sum::<f64>();
    let rows = suffix_compatibility.rows;
    let (Some(first), Some(last)) = (rows.first(), rows.last()) else {
        frontier.scores.extend(
            frontier
                .parent
                .masses
                .iter()
                .map(|&log_mass| log_mass + score_alpha * neutral),
        );
        return;
    };
    let first_word = first.word as usize;
    frontier.transpose_detectors(first_word, last.word as usize);
    let count = frontier.parent.masses.len();
    frontier.scores.resize(count, neutral);
    for row in suffix_compatibility.rows {
        let probabilities = &suffix_compatibility.values.probabilities[row.epoch as usize];
        let (zero, one) = (probabilities.zero, probabilities.one);
        let word = row.word as usize;
        let observed_word = observed[word];
        let mask = 1_u64 << row.bit;
        let offset = word - first_word;
        let candidates = &frontier.transposed[offset * count..(offset + 1) * count];
        for (accumulator, &candidate) in frontier.scores.iter_mut().zip(candidates) {
            let mismatch = (candidate ^ observed_word) & mask != 0;
            *accumulator += if mismatch { one } else { zero };
        }
    }
    for (score, &log_mass) in frontier.scores.iter_mut().zip(&frontier.parent.masses) {
        *score = log_mass + score_alpha * *score;
    }
}

fn score_candidates_int(
    frontier: &mut FrontierScratch<i64>,
    alpha_int: i64,
    scale: i32,
    suffix_compatibility: SuffixCompatibility<'_>,
    observed: &[u64],
) {
    frontier.scores.clear();
    // Skip entirely: fixed_mul_round* handles sentinel parity before a zero
    // multiplier, so multiplying an ln(0) suffix by zero would poison the score.
    if alpha_int == 0 {
        frontier.scores.extend_from_slice(&frontier.parent.masses);
        return;
    }
    let rows = suffix_compatibility.rows;
    let (Some(first), Some(last)) = (rows.first(), rows.last()) else {
        frontier.scores.extend(
            frontier
                .parent
                .masses
                .iter()
                .map(|&log_mass| score_int_metric(log_mass, 0, alpha_int, scale)),
        );
        return;
    };
    let first_word = first.word as usize;
    frontier.transpose_detectors(first_word, last.word as usize);
    let count = frontier.parent.masses.len();
    frontier.scores.resize(count, 0);
    for row in suffix_compatibility.rows {
        let probabilities = &suffix_compatibility.values.probabilities[row.epoch as usize];
        let (zero, one) = (probabilities.zero_int, probabilities.one_int);
        let word = row.word as usize;
        let observed_word = observed[word];
        let mask = 1_u64 << row.bit;
        let offset = word - first_word;
        let candidates = &frontier.transposed[offset * count..(offset + 1) * count];
        for (accumulator, &candidate) in frontier.scores.iter_mut().zip(candidates) {
            let mismatch = (candidate ^ observed_word) & mask != 0;
            *accumulator = int_metric_add(*accumulator, if mismatch { one } else { zero });
        }
    }
    for (score, &log_mass) in frontier.scores.iter_mut().zip(&frontier.parent.masses) {
        *score = score_int_metric(log_mass, *score, alpha_int, scale);
    }
}

fn prune(
    frontier: &mut FrontierScratch<f64>,
    k: usize,
    delta: f64,
    score_alpha: f64,
    suffix_compatibility: SuffixCompatibility<'_>,
    observed: &[u64],
) -> PruneResult<f64> {
    if k == usize::MAX && delta.is_infinite() {
        return PruneResult {
            dropped_states: 0,
            dropped_log_mass: f64::NEG_INFINITY,
            k_capped: false,
            delta_pruned: false,
        };
    }

    score_candidates(frontier, score_alpha, suffix_compatibility, observed);
    frontier.indices.clear();
    frontier.indices.extend(0..frontier.parent.masses.len());
    frontier.indices.sort_by(|&left, &right| {
        frontier.scores[right]
            .total_cmp(&frontier.scores[left])
            .then_with(|| {
                compare_state_words(
                    frontier.parent.key(left, frontier.stride),
                    frontier.parent.key(right, frontier.stride),
                    frontier.detector_words,
                )
            })
    });
    let cutoff = frontier.scores[frontier.indices[0]] - delta;
    frontier.retained.clear();
    frontier
        .retained
        .resize(frontier.parent.masses.len(), false);
    let mut dropped_states = 0;
    let mut dropped_log_mass = f64::NEG_INFINITY;
    let mut k_capped = false;
    let mut delta_pruned = false;

    for (index, &candidate) in frontier.indices.iter().enumerate() {
        let within_k = index < k;
        let within_delta = frontier.scores[candidate] >= cutoff;
        if within_k && within_delta {
            frontier.retained[candidate] = true;
        } else {
            dropped_states += 1;
            dropped_log_mass = logaddexp(dropped_log_mass, frontier.parent.masses[candidate]);
            k_capped |= !within_k;
            delta_pruned |= within_k && !within_delta;
        }
    }

    frontier.retain();
    PruneResult {
        dropped_states,
        dropped_log_mass,
        k_capped,
        delta_pruned,
    }
}

fn prune_maxlog(
    frontier: &mut FrontierScratch<i64>,
    k: usize,
    delta_int: i64,
    alpha_int: i64,
    scale: i32,
    suffix_compatibility: SuffixCompatibility<'_>,
    observed: &[u64],
) -> PruneResult<i64> {
    assert!(
        suffix_compatibility.values.int_metric_scale == Some(scale),
        "integer suffix scoring requires a table quantized at the decoder's metric scale"
    );
    score_candidates_int(frontier, alpha_int, scale, suffix_compatibility, observed);
    frontier.indices.clear();
    frontier.indices.extend(0..frontier.parent.masses.len());
    // Preserve the integer score tie-break: mass descending, then key.
    frontier.indices.sort_by(|&left, &right| {
        frontier.scores[right]
            .cmp(&frontier.scores[left])
            .then_with(|| frontier.parent.masses[right].cmp(&frontier.parent.masses[left]))
            .then_with(|| {
                compare_state_words(
                    frontier.parent.key(left, frontier.stride),
                    frontier.parent.key(right, frontier.stride),
                    frontier.detector_words,
                )
            })
    });
    let cutoff = frontier.scores[frontier.indices[0]].saturating_sub(delta_int);
    frontier.retained.clear();
    frontier
        .retained
        .resize(frontier.parent.masses.len(), false);
    let mut dropped_states = 0;
    let mut dropped_log_mass = INT_METRIC_NEG_INF;
    let mut k_capped = false;
    let mut delta_pruned = false;

    for (index, &candidate) in frontier.indices.iter().enumerate() {
        let within_k = index < k;
        let within_delta = frontier.scores[candidate] >= cutoff;
        if within_k && within_delta {
            frontier.retained[candidate] = true;
        } else {
            dropped_states += 1;
            dropped_log_mass = dropped_log_mass.max(frontier.parent.masses[candidate]);
            k_capped |= !within_k;
            delta_pruned |= within_k && !within_delta;
        }
    }

    frontier.retain();
    PruneResult {
        dropped_states,
        dropped_log_mass,
        k_capped,
        delta_pruned,
    }
}

fn build_suffix_epochs<'a>(
    columns: impl DoubleEndedIterator<Item = (&'a [u64], &'a [u64])> + ExactSizeIterator,
    num_detectors: usize,
) -> Result<(Vec<SuffixEpoch>, Vec<Vec<SuffixRow>>), DecoderError> {
    let detector_words = checked_detector_words(num_detectors)?;
    let mut tables = vec![Vec::new(); columns.len()];
    let mut epochs: Vec<SuffixEpoch> = (0..num_detectors)
        .map(|detector| SuffixEpoch {
            detector,
            column_index: None,
        })
        .collect();
    let mut current_epochs: Vec<usize> = (0..num_detectors).collect();
    for (column_index, (active_mask, detector_toggle)) in columns.enumerate().rev() {
        debug_assert_eq!(active_mask.len(), detector_words as usize);
        for (word, mask) in (0..detector_words).zip(active_mask) {
            for bit in set_bits(std::slice::from_ref(mask)) {
                let detector = word as usize * WORD_BITS + bit;
                // Only referenced epochs must fit; final first-toggle updates
                // can create epochs that no column ever reads.
                let epoch = u32::try_from(current_epochs[detector]).map_err(|_| {
                    DecoderError::InvalidConfiguration(
                        "referenced suffix epoch must fit u32".into(),
                    )
                })?;
                tables[column_index].push(SuffixRow {
                    word,
                    epoch,
                    bit: u8::try_from(bit).expect("detector bit must fit u8"),
                });
            }
        }
        for detector in set_bits(detector_toggle) {
            current_epochs[detector] = epochs.len();
            epochs.push(SuffixEpoch {
                detector,
                column_index: Some(column_index),
            });
        }
    }
    Ok((epochs, tables))
}

fn checked_detector_words(num_detectors: usize) -> Result<u32, DecoderError> {
    u32::try_from(words_for(num_detectors))
        .map_err(|_| DecoderError::InvalidConfiguration("detector word count must fit u32".into()))
}

fn bp_score_probability(posterior_llr: f64) -> f64 {
    let probability = 1.0 / (1.0 + libm::exp(posterior_llr));
    probability.clamp(BP_SCORE_PROBABILITY_MIN, 1.0 - BP_SCORE_PROBABILITY_MIN)
}

#[cfg(not(debug_assertions))]
fn debug_assert_model_invariants(_columns: &[Column], _touched_detectors: &[u64]) {}

#[cfg(debug_assertions)]
fn debug_assert_model_invariants(columns: &[Column], touched_detectors: &[u64]) {
    let mut closed_detectors = vec![0; touched_detectors.len()];
    for column in columns {
        debug_assert!(
            closed_detectors
                .iter()
                .zip(&column.close_mask)
                .all(|(&closed, &closing)| closed & closing == 0),
            "close masks must be disjoint"
        );
        or_assign(&mut closed_detectors, &column.close_mask);
        debug_assert!(
            closed_detectors
                .iter()
                .zip(&column.active_mask)
                .all(|(&closed, &active)| closed & active == 0),
            "a detector must not remain active after its closing column"
        );
    }
    debug_assert_eq!(
        closed_detectors, touched_detectors,
        "close masks must partition touched detectors"
    );
    debug_assert!(
        columns
            .last()
            .is_none_or(|column| column.active_mask.iter().all(|&word| word == 0)),
        "the final column must have an empty active mask"
    );
}

#[cfg(not(debug_assertions))]
fn debug_assert_factor_model_invariants(_columns: &[FactorColumn], _touched_detectors: &[u64]) {}

#[cfg(debug_assertions)]
fn debug_assert_factor_model_invariants(columns: &[FactorColumn], touched_detectors: &[u64]) {
    let mut closed_detectors = vec![0; touched_detectors.len()];
    for column in columns {
        debug_assert!(
            closed_detectors
                .iter()
                .zip(&column.close_mask)
                .all(|(&closed, &closing)| closed & closing == 0),
            "close masks must be disjoint"
        );
        or_assign(&mut closed_detectors, &column.close_mask);
        debug_assert!(
            closed_detectors
                .iter()
                .zip(&column.active_mask)
                .all(|(&closed, &active)| closed & active == 0),
            "a detector must not remain active after its closing column"
        );
    }
    debug_assert_eq!(
        closed_detectors, touched_detectors,
        "close masks must partition touched detectors"
    );
    debug_assert!(
        columns
            .last()
            .is_none_or(|column| column.active_mask.iter().all(|&word| word == 0)),
        "the final column must have an empty active mask"
    );
}

#[cfg(test)]
fn suffix_compatibility_score(
    active_syndrome: &[u64],
    observed: &[u64],
    suffix_compatibility: SuffixCompatibility<'_>,
) -> f64 {
    suffix_compatibility
        .rows
        .iter()
        .map(|reference| {
            let row = &suffix_compatibility.values.probabilities[reference.epoch as usize];
            let word_index = reference.word as usize;
            let bit_mask = 1 << reference.bit;
            if (active_syndrome[word_index] ^ observed[word_index]) & bit_mask == 0 {
                row.zero
            } else {
                row.one
            }
        })
        .sum()
}

#[cfg(test)]
fn suffix_compatibility_score_int(
    active_syndrome: &[u64],
    observed: &[u64],
    suffix_compatibility: SuffixCompatibility<'_>,
) -> i64 {
    suffix_compatibility
        .rows
        .iter()
        .fold(0, |total, reference| {
            let row = &suffix_compatibility.values.probabilities[reference.epoch as usize];
            let word_index = reference.word as usize;
            let bit_mask = 1 << reference.bit;
            let term = if (active_syndrome[word_index] ^ observed[word_index]) & bit_mask == 0 {
                row.zero_int
            } else {
                row.one_int
            };
            int_metric_add(total, term)
        })
}

fn finish_maxlog_decode(
    frontier: &FrontierScratch<i64>,
    scale: i32,
    stats: MaxLogDecodeStats,
) -> TrellisDecodeAttempt {
    let mut terminal_by_logical = BTreeMap::<Vec<u64>, i64>::new();
    for (index, &log_mass) in frontier.parent.masses.iter().enumerate() {
        let logical =
            frontier.parent.key(index, frontier.stride)[frontier.detector_words..].to_vec();
        // The final column closes every active detector, so StateKey uniqueness
        // already implies one terminal entry per logical label. Upstream's
        // per-label MAX fold is therefore a no-op in this representation.
        assert!(
            terminal_by_logical.insert(logical, log_mass).is_none(),
            "terminal boundary states must be unique per logical label"
        );
    }
    let mut terminal: Vec<(Vec<u64>, i64)> = terminal_by_logical.into_iter().collect();
    terminal.sort_by(|(left_logical, left_mass), (right_logical, right_mass)| {
        right_mass
            .cmp(left_mass)
            .then_with(|| compare_words_as_unsigned(left_logical, right_logical))
    });
    let (winner_logical, winner_mass) = &terminal[0];
    let scale_f64 = f64::from(scale);
    let runner_up_gap = terminal
        .get(1)
        .map(|(_, runner_up_mass)| i64_to_f64(*winner_mass - *runner_up_mass) / scale_f64);
    let logical_masses = terminal
        .iter()
        .map(|(logical, log_mass)| TrellisLogicalMass {
            logical: ObsMask::from_words(logical),
            log_mass: i64_to_f64(*log_mass) / scale_f64,
        })
        .collect();
    let status = if stats.dropped_states == 0 {
        TrellisStatus::Exact
    } else {
        TrellisStatus::Pruned {
            k_capped: stats.k_capped,
            delta_pruned: stats.delta_pruned,
        }
    };

    TrellisDecodeAttempt::Success(TrellisResult {
        predicted: ObsMask::from_words(winner_logical),
        log_evidence: i64_to_f64(*winner_mass) / scale_f64,
        runner_up_gap,
        peak_retained_states: stats.peak_retained_states,
        processed_columns: stats.processed_columns,
        transitions: stats.transitions,
        dropped_states: stats.dropped_states,
        dropped_log_mass: if stats.dropped_log_mass == INT_METRIC_NEG_INF {
            f64::NEG_INFINITY
        } else {
            i64_to_f64(stats.dropped_log_mass) / scale_f64
        },
        bp_seconds: stats.bp_seconds,
        bp_runs: 0,
        escalation_rungs_used: 0,
        status,
        logical_masses,
    })
}

fn sort_candidates(candidates: &mut [Candidate]) {
    candidates.sort_by(|left, right| {
        right
            .log_mass
            .total_cmp(&left.log_mass)
            .then_with(|| left.key.cmp(&right.key))
    });
}

fn logaddexp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let (high, low) = if left.total_cmp(&right) == Ordering::Less {
        (right, left)
    } else {
        (left, right)
    };
    high + libm::log1p(libm::exp(low - high))
}

fn unexplainable_error() -> DecoderError {
    DecoderError::DecodingFailed("syndrome is unexplainable at the given pruning parameters".into())
}

const fn words_for(bits: usize) -> usize {
    bits.div_ceil(WORD_BITS)
}

fn indices_to_words(indices: &[u32], word_count: usize) -> Vec<u64> {
    let mut words = vec![0; word_count];
    for &index in indices {
        set_bit(&mut words, index as usize);
    }
    words
}

fn syndrome_to_words(syndrome: &[u8], word_count: usize) -> Vec<u64> {
    let mut words = vec![0; word_count];
    for (index, &value) in syndrome.iter().enumerate() {
        if value != 0 {
            set_bit(&mut words, index);
        }
    }
    words
}

fn set_bit(words: &mut [u64], index: usize) {
    words[index / WORD_BITS] |= 1 << (index % WORD_BITS);
}

fn set_bits(words: &[u64]) -> impl Iterator<Item = usize> + '_ {
    // Ascending visit order is load-bearing: downstream float reductions sum in
    // this order and the bitwise parity contract pins it. Skipping zero words
    // and clearing lowest set bits preserves that order exactly.
    words.iter().enumerate().flat_map(|(word_index, &word)| {
        let mut remaining = word;
        std::iter::from_fn(move || {
            if remaining == 0 {
                return None;
            }
            let bit = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            Some(word_index * WORD_BITS + bit)
        })
    })
}

fn xor_assign(left: &mut [u64], right: &[u64]) {
    for (left_word, &right_word) in left.iter_mut().zip(right) {
        *left_word ^= right_word;
    }
}

fn or_assign(left: &mut [u64], right: &[u64]) {
    for (left_word, &right_word) in left.iter_mut().zip(right) {
        *left_word |= right_word;
    }
}

fn and_assign(left: &mut [u64], right: &[u64]) {
    for (left_word, &right_word) in left.iter_mut().zip(right) {
        *left_word &= right_word;
    }
}

fn and_not_assign(left: &mut [u64], right: &[u64]) {
    for (left_word, &right_word) in left.iter_mut().zip(right) {
        *left_word &= !right_word;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        INT_METRIC_MAX, INT_METRIC_NEG_INF, MetricMode, SparseDem, TrellisConfig, TrellisDecoder,
        bp_score_probability, fixed_mul_round, fixed_mul_round_fast, i64_to_f64, logaddexp,
        merge_indistinguishable_columns, quantize_metric,
    };
    use std::collections::BTreeMap;

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn detector_word_width_is_checked_before_allocation() {
        let largest_width = u32::MAX as usize * super::WORD_BITS;
        assert_eq!(
            super::checked_detector_words(largest_width).unwrap(),
            u32::MAX
        );
        let dem = SparseDem {
            mechanisms: Vec::new(),
            detector_coords: BTreeMap::new(),
            num_detectors: largest_width + 1,
            num_observables: 0,
        };
        assert!(matches!(
            TrellisDecoder::from_sparse_dem(&dem, TrellisConfig::default()),
            Err(super::DecoderError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn flat_merge_preserves_branch_arrival_fold_order() {
        let (first, second, third) = (-0.1, -0.2, -1.0);
        let expected = logaddexp(logaddexp(first, second), third);
        let right_associated = logaddexp(first, logaddexp(second, third));
        let reordered = logaddexp(logaddexp(first, third), second);
        assert_ne!(expected.to_bits(), right_associated.to_bits());
        assert_ne!(expected.to_bits(), reordered.to_bits());

        let mut frontier = super::FrontierScratch::<f64>::default();
        frontier.reset(&[0, 0], &[0], &[0, 0], 0.0);
        let context = super::BranchContext {
            detector_words: frontier.detector_words,
            close_mask: &[0, 0],
            active_mask: &[u64::MAX, u64::MAX],
            observed: &[0, 0],
        };
        let mut transitions = 0;
        // Interleave other keys so sorting must move the colliding arrivals.
        // Detector words compare most-significant first, before logical words.
        for (key, mass) in [
            ([0, 1, 0], -4.0),
            ([1, 0, 1], first),
            ([1, 0, 0], -5.0),
            ([1, 0, 1], second),
            ([0, 1, 0], -6.0),
            ([1, 0, 1], third),
        ] {
            context.emit(&mut frontier.branches, &key, None, mass, &mut transitions);
        }
        frontier.merge(logaddexp);
        assert_eq!(transitions, 6);
        assert_eq!(frontier.parent.masses.len(), 3);
        assert_eq!(frontier.parent.words, [1, 0, 0, 1, 0, 1, 0, 1, 0]);
        assert_eq!(frontier.parent.masses[1].to_bits(), expected.to_bits());
        assert_ne!(
            frontier.parent.masses[1].to_bits(),
            right_associated.to_bits()
        );
        assert_ne!(frontier.parent.masses[1].to_bits(), reordered.to_bits());
    }

    #[test]
    fn candidate_scores_match_scalar_suffix_folds() {
        use rand::{RngExt, SeedableRng};
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5343_4f52_4553);
        for (num_detectors, window) in [(70, 0..70), (200, 70..190)] {
            let dem = SparseDem {
                mechanisms: (0..24)
                    .map(|_| {
                        (
                            rng.random_range(0.001..0.999),
                            window.clone().filter(|_| rng.random_bool(0.2)).collect(),
                            vec![0],
                        )
                    })
                    .collect(),
                detector_coords: BTreeMap::new(),
                num_detectors,
                num_observables: 1,
            };
            let decoder = TrellisDecoder::from_sparse_dem(
                &dem,
                TrellisConfig {
                    metric_mode: MetricMode::MaxLogInt,
                    ..TrellisConfig::default()
                },
            )
            .unwrap();
            let super::Kernel::Binary(columns) = &decoder.model.kernel else {
                panic!("binary DEM");
            };
            assert!(columns.last().unwrap().suffix_compatibility.is_empty());
            let mut float = super::FrontierScratch::<f64>::default();
            let mut integer = super::FrontierScratch::<i64>::default();
            let detector_words = super::words_for(num_detectors);
            let zeros = vec![0; detector_words];
            let mut nonzero_origins = 0;
            let mut nonzero_multiword_spans = 0;
            for column in columns {
                let rows = &column.suffix_compatibility;
                if let (Some(first), Some(last)) = (rows.first(), rows.last()) {
                    nonzero_origins += usize::from(first.word > 0);
                    nonzero_multiword_spans +=
                        usize::from(first.word > 0 && last.word > first.word);
                }
                for count in [1, 3, 16, 33] {
                    float.reset(&zeros, &[0], &zeros, 0.0);
                    float.branches.clear(float.stride);
                    // Include duplicate arrivals so the oracle sees a merged set.
                    for candidate in 0..count {
                        let mut key: Vec<_> = column
                            .active_mask
                            .iter()
                            .map(|mask| rng.random::<u64>() & mask)
                            .collect();
                        key.push(candidate);
                        float.branches.words.extend_from_slice(&key);
                        float.branches.masses.push(if candidate == 0 {
                            // Empty suffix sums must also preserve negative zero.
                            if count == 1 { -0.0 } else { 0.0 }
                        } else {
                            rng.random_range(-100.0..0.0)
                        });
                        if candidate != 0 {
                            float.branches.words.extend_from_slice(&key);
                            float.branches.masses.push(-200.0);
                        }
                    }
                    float.merge(logaddexp);
                    assert!(float.parent.masses.contains(&0.0));
                    integer.reset(&zeros, &[0], &zeros, 0);
                    integer.parent.words.clone_from(&float.parent.words);
                    integer.parent.masses = float
                        .parent
                        .masses
                        .iter()
                        .map(|&mass| quantize_metric(mass, 1024))
                        .collect();
                    let compatibility = super::SuffixCompatibility {
                        rows: &column.suffix_compatibility,
                        values: &decoder.model.suffix_values,
                    };
                    for _ in 0..4 {
                        let observed: Vec<u64> =
                            (0..detector_words).map(|_| rng.random()).collect();
                        for alpha in [0.0, 0.8, 1.0] {
                            super::score_candidates(&mut float, alpha, compatibility, &observed);
                            for (index, &mass) in float.parent.masses.iter().enumerate() {
                                let expected = if alpha == 0.0 {
                                    mass
                                } else {
                                    mass + alpha
                                        * super::suffix_compatibility_score(
                                            &float.parent.key(index, float.stride)
                                                [..detector_words],
                                            &observed,
                                            compatibility,
                                        )
                                };
                                assert_eq!(float.scores[index].to_bits(), expected.to_bits());
                            }
                        }
                        for alpha in [0, 819, 1024] {
                            super::score_candidates_int(
                                &mut integer,
                                alpha,
                                1024,
                                compatibility,
                                &observed,
                            );
                            for (index, &mass) in integer.parent.masses.iter().enumerate() {
                                let expected = if alpha == 0 {
                                    mass
                                } else {
                                    super::score_int_metric(
                                        mass,
                                        super::suffix_compatibility_score_int(
                                            &integer.parent.key(index, integer.stride)
                                                [..detector_words],
                                            &observed,
                                            compatibility,
                                        ),
                                        alpha,
                                        1024,
                                    )
                                };
                                assert_eq!(integer.scores[index], expected);
                            }
                        }
                    }
                }
            }
            if window.start > 0 {
                assert!(
                    nonzero_origins > 0,
                    "windowed model must exercise nonzero origins"
                );
                assert!(
                    nonzero_multiword_spans > 0,
                    "windowed model must exercise nonzero multiword spans"
                );
            }
        }
    }

    #[test]
    fn suffix_epochs_match_direct_recomputation() {
        use rand::{RngExt, SeedableRng};
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4550_4f43_4853);
        // Cross a word boundary and include positive, negative, and zero moments.
        let dem = SparseDem {
            mechanisms: (0..96)
                .map(|column| {
                    let probability = if column % 11 == 0 {
                        0.5
                    } else {
                        rng.random_range(0.001..0.999)
                    };
                    let detectors = (0..70).filter(|_| rng.random_bool(0.12)).collect();
                    (probability, detectors, Vec::new())
                })
                .collect(),
            detector_coords: BTreeMap::new(),
            num_detectors: 70,
            num_observables: 0,
        };
        let prior_moments: Vec<f64> = dem
            .mechanisms
            .iter()
            .map(|(p, _, _)| 1.0 - 2.0 * p)
            .collect();
        for metric_mode in [MetricMode::default(), MetricMode::MaxLogInt] {
            let mut decoder = TrellisDecoder::from_sparse_dem(
                &dem,
                TrellisConfig {
                    k: 2,
                    bp_score_iterations: 5,
                    metric_mode,
                    ..TrellisConfig::default()
                },
            )
            .unwrap();
            let super::Kernel::Binary(columns) = &decoder.model.kernel else {
                panic!("binary DEM");
            };
            assert_eq!(
                decoder.model.suffix_epochs.len(),
                dem.num_detectors
                    + columns
                        .iter()
                        .map(|column| super::set_bits(&column.detector_toggle).count())
                        .sum::<usize>()
            );
            assert_direct_suffix_values(
                columns,
                &decoder.model.suffix_values,
                &prior_moments,
                dem.num_detectors,
            );
            let allocation = decoder
                .scratch
                .bp_score
                .as_ref()
                .unwrap()
                .suffix_values
                .probabilities
                .as_ptr();
            for _ in 0..4 {
                let observed = super::indices_to_words(
                    &(0..70).filter(|_| rng.random_bool(0.5)).collect::<Vec<_>>(),
                    decoder.model.detector_words,
                );
                assert!(
                    decoder
                        .model
                        .refresh_bp_suffix_values(&mut decoder.scratch, &observed)
                        .unwrap()
                        .is_some()
                );
                let bp = decoder.scratch.bp_score.as_ref().unwrap();
                assert_eq!(allocation, bp.suffix_values.probabilities.as_ptr());
                let moments: Vec<f64> = bp
                    .posterior
                    .iter()
                    .map(|&llr| 1.0 - 2.0 * bp_score_probability(llr))
                    .collect();
                let super::Kernel::Binary(columns) = &decoder.model.kernel else {
                    panic!("binary DEM");
                };
                assert_direct_suffix_values(
                    columns,
                    &bp.suffix_values,
                    &moments,
                    dem.num_detectors,
                );
            }
        }
    }

    /// Independent old per-(column, row) formula, retained only as a test oracle.
    fn assert_direct_suffix_values(
        columns: &[super::Column],
        values: &super::SuffixValues,
        moments: &[f64],
        num_detectors: usize,
    ) {
        let mut row_moments = vec![1.0; num_detectors];
        for (column, &moment) in columns.iter().zip(moments).rev() {
            let detectors: Vec<usize> = super::set_bits(&column.active_mask).collect();
            assert_eq!(column.suffix_compatibility.len(), detectors.len());
            for (reference, detector) in column.suffix_compatibility.iter().zip(detectors) {
                assert_eq!(reference.word as usize, detector / super::WORD_BITS);
                assert_eq!(usize::from(reference.bit), detector % super::WORD_BITS);
                let eta = row_moments[detector];
                let zero = libm::log(1.0_f64.midpoint(eta));
                let one = libm::log(1.0_f64.midpoint(-eta));
                let pair = &values.probabilities[reference.epoch as usize];
                assert_eq!(pair.zero.to_bits(), zero.to_bits());
                assert_eq!(pair.one.to_bits(), one.to_bits());
                if let Some(scale) = values.int_metric_scale {
                    assert_eq!(pair.zero_int, quantize_metric(zero, scale));
                    assert_eq!(pair.one_int, quantize_metric(one, scale));
                }
            }
            for detector in super::set_bits(&column.detector_toggle) {
                row_moments[detector] *= moment;
            }
        }
    }

    #[test]
    fn logaddexp_handles_negative_infinity_on_either_side() {
        assert_eq!(
            logaddexp(f64::NEG_INFINITY, -2.5).to_bits(),
            (-2.5_f64).to_bits()
        );
        assert_eq!(
            logaddexp(-2.5, f64::NEG_INFINITY).to_bits(),
            (-2.5_f64).to_bits()
        );
    }

    #[test]
    fn integer_metric_quantization_saturates_at_its_boundaries() {
        for non_finite in [f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(quantize_metric(non_finite, 1024), INT_METRIC_NEG_INF);
        }
        #[cfg(debug_assertions)]
        assert!(
            std::panic::catch_unwind(|| quantize_metric(f64::INFINITY, 1024)).is_err(),
            "positive infinity must trip the debug-only caller-contract assertion"
        );
        // Release builds compile the caller-contract assertion out; the
        // upstream-faithful fallback (saturate to the negative sentinel)
        // must hold there.
        #[cfg(not(debug_assertions))]
        assert_eq!(quantize_metric(f64::INFINITY, 1024), INT_METRIC_NEG_INF);
        assert_eq!(quantize_metric(f64::MIN, 1024), INT_METRIC_NEG_INF);
        assert_eq!(quantize_metric(f64::MAX, 1024), INT_METRIC_MAX);
        assert_eq!(quantize_metric(-3e18, 1), INT_METRIC_NEG_INF);
        // `INT_METRIC_NEG_INF + 1` rounds back to the same f64 as the sentinel,
        // so upstream's `+ 1` low-bound detail is inert in this f64 port.
        let just_inside_low = i64_to_f64(INT_METRIC_NEG_INF) + 1024.0;
        assert_eq!(
            quantize_metric(just_inside_low, 1),
            INT_METRIC_NEG_INF + 1024
        );
        assert_eq!(
            quantize_metric(i64_to_f64(INT_METRIC_MAX), 1),
            INT_METRIC_MAX
        );
        assert_eq!(quantize_metric(1.5 / 1024.0, 1024), 2);
        assert_eq!(quantize_metric(-1.5 / 1024.0, 1024), -2);
        // Unlike 1.5, 2.5 distinguishes half-away from ties-to-even.
        assert_eq!(quantize_metric(2.5 / 1024.0, 1024), 3);
        assert_eq!(quantize_metric(-2.5 / 1024.0, 1024), -3);
    }

    #[test]
    fn integer_fixed_multiply_fast_path_matches_general_rounding() {
        for value in [
            1,
            -1,
            1_025,
            -1_025,
            987_654_321,
            -987_654_321,
            INT_METRIC_NEG_INF,
        ] {
            for multiplier in [0, 1, 512, 800, 1024] {
                assert_eq!(
                    fixed_mul_round_fast(value, multiplier, 1024),
                    fixed_mul_round(value, multiplier, 1024)
                );
            }
        }
        assert_eq!(fixed_mul_round(1, 512, 1024), 1);
        assert_eq!(fixed_mul_round(-1, 512, 1024), -1);
        assert_eq!(fixed_mul_round_fast(1, 512, 1024), 1);
        assert_eq!(fixed_mul_round_fast(-1, 512, 1024), -1);
    }

    #[test]
    fn xor_probability_arithmetic_is_pinned() {
        let two_copies =
            merge_indistinguishable_columns(vec![(vec![1], vec![2], 0.3), (vec![1], vec![2], 0.3)]);
        // Exactly one of two p=0.3 variables fires with probability
        // 0.3*0.7 + 0.3*0.7 = 0.3*0.7*2 = 0.42.
        assert_eq!(two_copies.len(), 1);
        assert_eq!(two_copies[0].2.to_bits(), (0.3_f64 * 0.7 * 2.0).to_bits());

        let three_copies = merge_indistinguishable_columns(vec![
            (vec![1], vec![2], 0.5),
            (vec![1], vec![2], 0.5),
            (vec![1], vec![2], 0.5),
        ]);
        // XOR with a fair bit is fair. The first pair folds to 0.5, and
        // folding the third fair bit therefore remains exactly 0.5.
        assert_eq!(three_copies.len(), 1);
        assert_eq!(three_copies[0].2.to_bits(), 0.5_f64.to_bits());

        let dem = SparseDem {
            mechanisms: vec![
                (0.5, vec![0], vec![0]),
                (0.5, vec![0], vec![0]),
                (0.5, vec![0], vec![0]),
            ],
            detector_coords: BTreeMap::new(),
            num_detectors: 1,
            num_observables: 1,
        };
        let mut decoder = TrellisDecoder::from_sparse_dem(
            &dem,
            TrellisConfig {
                merge_indistinguishable: true,
                ..TrellisConfig::default()
            },
        )
        .unwrap();
        let result = decoder.decode(&[1]).unwrap();
        assert_eq!(result.processed_columns, 1);
        assert_eq!(result.log_evidence.to_bits(), libm::log(0.5).to_bits());
    }

    #[test]
    fn bp_score_probability_clamps_saturated_llrs() {
        assert_eq!(bp_score_probability(1_000.0).to_bits(), 1e-6_f64.to_bits());
        assert_eq!(
            bp_score_probability(-1_000.0).to_bits(),
            (1.0 - 1e-6_f64).to_bits()
        );
    }

    #[test]
    fn bp_enabled_decoders_do_not_share_state() {
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
        let config = TrellisConfig {
            k: 1,
            delta: f64::INFINITY,
            score_alpha: 0.8,
            column_order: None,
            merge_indistinguishable: false,
            bp_score_iterations: 5,
            metric_mode: MetricMode::default(),
            int_metric_scale: 1024,
        };
        let first = TrellisDecoder::from_sparse_dem(&dem, config.clone()).unwrap();
        let second = TrellisDecoder::from_sparse_dem(&dem, config).unwrap();
        let (first_graph, first_scratch) = first.bp_state_addrs().unwrap();
        let (second_graph, second_scratch) = second.bp_state_addrs().unwrap();

        assert_ne!(first_graph, first_scratch);
        assert_ne!(first_graph, second_graph);
        assert_ne!(first_graph, second_scratch);
        assert_ne!(first_scratch, second_graph);
        assert_ne!(first_scratch, second_scratch);
        assert_ne!(second_graph, second_scratch);
    }
}
