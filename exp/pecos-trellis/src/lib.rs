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

pub mod factor;

use factor::{FactorModel, NormalizedFactor, Outcome};
use pecos_decoder_core::ObservableDecoder;
use pecos_decoder_core::bp::{BpGraph, BpScratch, min_sum_bp_into};
pub use pecos_decoder_core::dem::SparseDem;
pub use pecos_decoder_core::errors::DecoderError;
pub use pecos_decoder_core::obs_mask::ObsMask;
use std::cmp::Ordering;
use std::collections::BTreeMap;
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
    /// This merge is mathematically exact, but it takes a different
    /// floating-point path and the external parity contract on this engine is
    /// bitwise, so it is disabled by default.
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
    /// Number of candidate branch evaluations, counted at entry to
    /// `merge_branch` (two per retained state for a binary column, or one per
    /// outcome and retained state for an N-ary column).
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
    /// For an escalated `BpTrellis` result, this is the total across
    /// the base attempt and every attempted rung.
    pub bp_seconds: f64,
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
    suffix_compatibility: Vec<SuffixCompatibility>,
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
    suffix_compatibility: Vec<SuffixCompatibility>,
}

#[derive(Clone, Debug)]
struct ColumnOutcome {
    detector_toggle: Vec<u64>,
    logical_toggle: Vec<u64>,
    probability: f64,
    log_prior: f64,
    log_prior_int: i64,
}

#[derive(Clone, Debug)]
struct SuffixCompatibility {
    word_index: usize,
    bit_mask: u64,
    log_probability_zero: f64,
    log_probability_one: f64,
    log_probability_zero_int: i64,
    log_probability_one_int: i64,
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

#[derive(Clone, Debug)]
struct ScoredCandidate {
    candidate: Candidate,
    score: f64,
}

struct PruneResult {
    retained: BTreeMap<StateKey, f64>,
    dropped_states: u64,
    dropped_log_mass: f64,
    k_capped: bool,
    delta_pruned: bool,
}

#[derive(Clone, Debug)]
struct IntCandidate {
    key: StateKey,
    log_mass: i64,
}

#[derive(Clone, Debug)]
struct ScoredIntCandidate {
    candidate: IntCandidate,
    score: i64,
}

struct IntPruneResult {
    retained: BTreeMap<StateKey, i64>,
    dropped_states: u64,
    dropped_log_mass: i64,
    k_capped: bool,
    delta_pruned: bool,
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
    graph: BpGraph,
    scratch: BpScratch,
    posterior: Vec<f64>,
    residual_syndrome: Vec<u8>,
}

impl BpScoreState {
    fn new(graph: BpGraph) -> Self {
        let scratch = BpScratch::new(&graph);
        let posterior = vec![0.0; graph.mechanism_count()];
        let residual_syndrome = vec![0; graph.check_count()];
        Self {
            graph,
            scratch,
            posterior,
            residual_syndrome,
        }
    }
}

type RawColumn = (Vec<u64>, Vec<u64>, f64);
type SuffixCompatibilityTables = Vec<Vec<SuffixCompatibility>>;
type BpSuffixPreparation = (Option<SuffixCompatibilityTables>, f64);

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
        /// Wall-clock seconds spent on BP suffix scoring in this attempt.
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
    config: TrellisConfig,
    kernel: Kernel,
    num_detectors: usize,
    detector_words: usize,
    logical_words: usize,
    touched_detectors: Vec<u64>,
    forced_syndrome: Vec<u64>,
    forced_logical: Vec<u64>,
    bp_score: Option<BpScoreState>,
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
    /// parameters, probabilities, indices, or column order.
    ///
    /// # Panics
    ///
    /// Panics if the internal post-filter BP graph and DP column counts differ.
    pub fn from_sparse_dem(dem: &SparseDem, config: TrellisConfig) -> Result<Self, DecoderError> {
        let build_started = Instant::now();
        validate_config(&config, dem.mechanisms.len())?;

        let detector_words = words_for(dem.num_detectors);
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

        let bp_score = if config.bp_score_iterations > 0
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
            Some(BpScoreState::new(graph))
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

        let suffix_tables = build_suffix_compatibility_tables(
            &columns,
            &column_moments,
            dem.num_detectors,
            config.int_metric_scale,
        );
        for (column, suffix_compatibility) in columns.iter_mut().zip(suffix_tables) {
            column.suffix_compatibility = suffix_compatibility;
        }

        debug_assert_model_invariants(&columns, &touched_detectors);
        let build_seconds = build_started.elapsed().as_secs_f64();

        Ok(Self {
            config,
            kernel: Kernel::Binary(columns),
            num_detectors: dem.num_detectors,
            detector_words,
            logical_words,
            touched_detectors,
            forced_syndrome,
            forced_logical,
            bp_score,
            build_seconds,
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
    /// merging is requested for a genuinely N-ary model.
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
        let detector_words = words_for(model.num_detectors());
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
        for (column_index, (outcomes, support)) in raw_columns.into_iter().zip(supports).enumerate()
        {
            or_assign(&mut open_detectors, &support);
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

        let suffix_tables = build_suffix_compatibility_tables_nary(
            &columns,
            model.num_detectors(),
            config.int_metric_scale,
        );
        for (column, suffix_compatibility) in columns.iter_mut().zip(suffix_tables) {
            column.suffix_compatibility = suffix_compatibility;
        }
        debug_assert_factor_model_invariants(&columns, &touched_detectors);
        let build_seconds = build_started.elapsed().as_secs_f64();

        Ok(Self {
            config,
            kernel: Kernel::Nary(columns),
            num_detectors: model.num_detectors(),
            detector_words,
            logical_words,
            touched_detectors,
            forced_syndrome,
            forced_logical,
            bp_score: None,
            build_seconds,
        })
    }

    /// Wall-clock seconds spent constructing this model.
    #[must_use]
    pub fn build_seconds(&self) -> f64 {
        self.build_seconds
    }

    /// Addresses of this decoder's BP scoring state, if BP scoring is enabled.
    /// Exists so downstream crates can assert two decoders do not share state.
    #[doc(hidden)]
    #[must_use]
    pub fn bp_state_addrs(&self) -> Option<(usize, usize)> {
        self.bp_score.as_ref().map(|bp_score| {
            let graph: &BpGraph = &bp_score.graph;
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
        match (self.config.metric_mode, &self.kernel) {
            (MetricMode::LogSumExpFloat, Kernel::Binary(_)) => self.decode_attempt_binary(syndrome),
            (MetricMode::LogSumExpFloat, Kernel::Nary(_)) => self.decode_attempt_nary(syndrome),
            (MetricMode::MaxLogInt, Kernel::Binary(_)) => {
                self.decode_attempt_binary_maxlog(syndrome)
            }
            (MetricMode::MaxLogInt, Kernel::Nary(_)) => self.decode_attempt_nary_maxlog(syndrome),
        }
    }

    fn decode_attempt_binary(&mut self, syndrome: &[u8]) -> TrellisDecodeAttempt {
        if syndrome.len() != self.num_detectors {
            return TrellisDecodeAttempt::Error(DecoderError::InvalidDimensions {
                expected: self.num_detectors,
                actual: syndrome.len(),
            });
        }

        let observed = syndrome_to_words(syndrome, self.detector_words);
        if observed
            .iter()
            .zip(&self.forced_syndrome)
            .zip(&self.touched_detectors)
            .any(|((&seen, &forced), &touched)| (seen ^ forced) & !touched != 0)
        {
            return TrellisDecodeAttempt::NoPath {
                error: unexplainable_error(),
                transitions: 0,
                bp_seconds: 0.0,
            };
        }

        let (bp_suffix_compatibility, bp_seconds) = match self.bp_suffix_compatibility(&observed) {
            Ok(preparation) => preparation,
            Err(error) => return TrellisDecodeAttempt::Error(error),
        };

        let mut initial_syndrome = self.forced_syndrome.clone();
        and_assign(&mut initial_syndrome, &self.touched_detectors);
        let initial = StateKey {
            active_syndrome: initial_syndrome,
            logical: self.forced_logical.clone(),
        };
        let mut frontier = BTreeMap::from([(initial, 0.0)]);
        let mut peak_retained_states = frontier.len();
        let mut transitions = 0;
        let mut dropped_states = 0;
        let mut dropped_log_mass = f64::NEG_INFINITY;
        let mut k_capped = false;
        let mut delta_pruned = false;

        let Kernel::Binary(columns) = &self.kernel else {
            unreachable!("binary decode called with N-ary kernel");
        };
        for (column_index, column) in columns.iter().enumerate() {
            let mut merged = BTreeMap::new();
            for (state, &log_mass) in &frontier {
                let branch_base = log_mass + column.log_one_minus_probability;
                merge_branch(
                    &mut merged,
                    state.clone(),
                    branch_base,
                    &column.close_mask,
                    &column.active_mask,
                    &observed,
                    &mut transitions,
                );

                let mut taken = state.clone();
                xor_assign(&mut taken.active_syndrome, &column.detector_toggle);
                xor_assign(&mut taken.logical, &column.logical_toggle);
                merge_branch(
                    &mut merged,
                    taken,
                    branch_base + column.log_odds,
                    &column.close_mask,
                    &column.active_mask,
                    &observed,
                    &mut transitions,
                );
            }

            if merged.is_empty() {
                return TrellisDecodeAttempt::NoPath {
                    error: unexplainable_error(),
                    transitions,
                    bp_seconds,
                };
            }
            let suffix_compatibility = bp_suffix_compatibility
                .as_ref()
                .map_or(&column.suffix_compatibility, |tables| &tables[column_index]);
            let pruned = prune(
                merged,
                self.config.k,
                self.config.delta,
                self.config.score_alpha,
                suffix_compatibility,
                &observed,
            );
            frontier = pruned.retained;
            dropped_states += pruned.dropped_states;
            dropped_log_mass = logaddexp(dropped_log_mass, pruned.dropped_log_mass);
            k_capped |= pruned.k_capped;
            delta_pruned |= pruned.delta_pruned;
            if frontier.is_empty() {
                // Pruning always retains the best-scoring candidate of a
                // nonempty set, so an empty frontier here means the scores
                // themselves were unusable (non-finite) -- an engine fault,
                // not an unexplainable syndrome. Genuine no-path exits happen
                // above, before pruning, when no branch is compatible.
                return TrellisDecodeAttempt::Error(DecoderError::InternalError(
                    "pruning emptied a nonempty frontier; candidate scores were not finite".into(),
                ));
            }
            peak_retained_states = peak_retained_states.max(frontier.len());
        }

        let mut terminal: Vec<Candidate> = frontier
            .into_iter()
            .map(|(key, log_mass)| Candidate { key, log_mass })
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
            bp_seconds,
            escalation_rungs_used: 0,
            status,
            logical_masses,
        })
    }

    fn decode_attempt_nary(&mut self, syndrome: &[u8]) -> TrellisDecodeAttempt {
        if syndrome.len() != self.num_detectors {
            return TrellisDecodeAttempt::Error(DecoderError::InvalidDimensions {
                expected: self.num_detectors,
                actual: syndrome.len(),
            });
        }

        let observed = syndrome_to_words(syndrome, self.detector_words);
        if observed
            .iter()
            .zip(&self.forced_syndrome)
            .zip(&self.touched_detectors)
            .any(|((&seen, &forced), &touched)| (seen ^ forced) & !touched != 0)
        {
            return TrellisDecodeAttempt::NoPath {
                error: unexplainable_error(),
                transitions: 0,
                bp_seconds: 0.0,
            };
        }

        let mut initial_syndrome = self.forced_syndrome.clone();
        and_assign(&mut initial_syndrome, &self.touched_detectors);
        let initial = StateKey {
            active_syndrome: initial_syndrome,
            logical: self.forced_logical.clone(),
        };
        let mut frontier = BTreeMap::from([(initial, 0.0)]);
        let mut peak_retained_states = frontier.len();
        let mut transitions = 0;
        let mut dropped_states = 0;
        let mut dropped_log_mass = f64::NEG_INFINITY;
        let mut k_capped = false;
        let mut delta_pruned = false;

        let Kernel::Nary(columns) = &self.kernel else {
            unreachable!("N-ary decode called with binary kernel");
        };
        for column in columns {
            let mut merged = BTreeMap::new();
            for (state, &log_mass) in &frontier {
                for outcome in &column.outcomes {
                    let mut taken = state.clone();
                    xor_assign(&mut taken.active_syndrome, &outcome.detector_toggle);
                    xor_assign(&mut taken.logical, &outcome.logical_toggle);
                    merge_branch(
                        &mut merged,
                        taken,
                        log_mass + outcome.log_prior,
                        &column.close_mask,
                        &column.active_mask,
                        &observed,
                        &mut transitions,
                    );
                }
            }

            if merged.is_empty() {
                return TrellisDecodeAttempt::NoPath {
                    error: unexplainable_error(),
                    transitions,
                    bp_seconds: 0.0,
                };
            }
            let pruned = prune(
                merged,
                self.config.k,
                self.config.delta,
                self.config.score_alpha,
                &column.suffix_compatibility,
                &observed,
            );
            frontier = pruned.retained;
            dropped_states += pruned.dropped_states;
            dropped_log_mass = logaddexp(dropped_log_mass, pruned.dropped_log_mass);
            k_capped |= pruned.k_capped;
            delta_pruned |= pruned.delta_pruned;
            if frontier.is_empty() {
                return TrellisDecodeAttempt::Error(DecoderError::InternalError(
                    "pruning emptied a nonempty frontier; candidate scores were not finite".into(),
                ));
            }
            peak_retained_states = peak_retained_states.max(frontier.len());
        }

        let mut terminal: Vec<Candidate> = frontier
            .into_iter()
            .map(|(key, log_mass)| Candidate { key, log_mass })
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
            escalation_rungs_used: 0,
            status,
            logical_masses,
        })
    }

    fn decode_attempt_binary_maxlog(&mut self, syndrome: &[u8]) -> TrellisDecodeAttempt {
        if syndrome.len() != self.num_detectors {
            return TrellisDecodeAttempt::Error(DecoderError::InvalidDimensions {
                expected: self.num_detectors,
                actual: syndrome.len(),
            });
        }

        let observed = syndrome_to_words(syndrome, self.detector_words);
        if observed
            .iter()
            .zip(&self.forced_syndrome)
            .zip(&self.touched_detectors)
            .any(|((&seen, &forced), &touched)| (seen ^ forced) & !touched != 0)
        {
            return TrellisDecodeAttempt::NoPath {
                error: unexplainable_error(),
                transitions: 0,
                bp_seconds: 0.0,
            };
        }

        let (bp_suffix_compatibility, bp_seconds) = match self.bp_suffix_compatibility(&observed) {
            Ok(preparation) => preparation,
            Err(error) => return TrellisDecodeAttempt::Error(error),
        };
        let mut initial_syndrome = self.forced_syndrome.clone();
        and_assign(&mut initial_syndrome, &self.touched_detectors);
        let initial = StateKey {
            active_syndrome: initial_syndrome,
            logical: self.forced_logical.clone(),
        };
        let mut frontier = BTreeMap::from([(initial, 0_i64)]);
        let mut peak_retained_states = frontier.len();
        let mut transitions = 0;
        let mut dropped_states = 0;
        let mut dropped_log_mass = INT_METRIC_NEG_INF;
        let mut k_capped = false;
        let mut delta_pruned = false;
        let scale = self.config.int_metric_scale;
        let delta_int = quantize_metric(self.config.delta, scale).max(0);
        let alpha_int = quantize_metric(self.config.score_alpha, scale);

        let Kernel::Binary(columns) = &self.kernel else {
            unreachable!("binary max-log decode called with N-ary kernel");
        };
        for (column_index, column) in columns.iter().enumerate() {
            let mut merged = BTreeMap::new();
            for (state, &log_mass) in &frontier {
                let branch_base = int_metric_add(log_mass, column.log_one_minus_probability_int);
                merge_branch_maxlog(
                    &mut merged,
                    state.clone(),
                    branch_base,
                    &column.close_mask,
                    &column.active_mask,
                    &observed,
                    &mut transitions,
                );

                let mut taken = state.clone();
                xor_assign(&mut taken.active_syndrome, &column.detector_toggle);
                xor_assign(&mut taken.logical, &column.logical_toggle);
                merge_branch_maxlog(
                    &mut merged,
                    taken,
                    int_metric_add(branch_base, column.log_odds_int),
                    &column.close_mask,
                    &column.active_mask,
                    &observed,
                    &mut transitions,
                );
            }
            if merged.is_empty() {
                return TrellisDecodeAttempt::NoPath {
                    error: unexplainable_error(),
                    transitions,
                    bp_seconds,
                };
            }
            let suffix_compatibility = bp_suffix_compatibility
                .as_ref()
                .map_or(&column.suffix_compatibility, |tables| &tables[column_index]);
            let pruned = prune_maxlog(
                merged,
                self.config.k,
                delta_int,
                alpha_int,
                scale,
                suffix_compatibility,
                &observed,
            );
            frontier = pruned.retained;
            dropped_states += pruned.dropped_states;
            dropped_log_mass = dropped_log_mass.max(pruned.dropped_log_mass);
            k_capped |= pruned.k_capped;
            delta_pruned |= pruned.delta_pruned;
            peak_retained_states = peak_retained_states.max(frontier.len());
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

    fn decode_attempt_nary_maxlog(&mut self, syndrome: &[u8]) -> TrellisDecodeAttempt {
        if syndrome.len() != self.num_detectors {
            return TrellisDecodeAttempt::Error(DecoderError::InvalidDimensions {
                expected: self.num_detectors,
                actual: syndrome.len(),
            });
        }

        let observed = syndrome_to_words(syndrome, self.detector_words);
        if observed
            .iter()
            .zip(&self.forced_syndrome)
            .zip(&self.touched_detectors)
            .any(|((&seen, &forced), &touched)| (seen ^ forced) & !touched != 0)
        {
            return TrellisDecodeAttempt::NoPath {
                error: unexplainable_error(),
                transitions: 0,
                bp_seconds: 0.0,
            };
        }

        let mut initial_syndrome = self.forced_syndrome.clone();
        and_assign(&mut initial_syndrome, &self.touched_detectors);
        let initial = StateKey {
            active_syndrome: initial_syndrome,
            logical: self.forced_logical.clone(),
        };
        let mut frontier = BTreeMap::from([(initial, 0_i64)]);
        let mut peak_retained_states = frontier.len();
        let mut transitions = 0;
        let mut dropped_states = 0;
        let mut dropped_log_mass = INT_METRIC_NEG_INF;
        let mut k_capped = false;
        let mut delta_pruned = false;
        let scale = self.config.int_metric_scale;
        let delta_int = quantize_metric(self.config.delta, scale).max(0);
        let alpha_int = quantize_metric(self.config.score_alpha, scale);

        let Kernel::Nary(columns) = &self.kernel else {
            unreachable!("N-ary max-log decode called with binary kernel");
        };
        for column in columns {
            let mut merged = BTreeMap::new();
            for (state, &log_mass) in &frontier {
                for outcome in &column.outcomes {
                    let mut taken = state.clone();
                    xor_assign(&mut taken.active_syndrome, &outcome.detector_toggle);
                    xor_assign(&mut taken.logical, &outcome.logical_toggle);
                    merge_branch_maxlog(
                        &mut merged,
                        taken,
                        int_metric_add(log_mass, outcome.log_prior_int),
                        &column.close_mask,
                        &column.active_mask,
                        &observed,
                        &mut transitions,
                    );
                }
            }
            if merged.is_empty() {
                return TrellisDecodeAttempt::NoPath {
                    error: unexplainable_error(),
                    transitions,
                    bp_seconds: 0.0,
                };
            }
            let pruned = prune_maxlog(
                merged,
                self.config.k,
                delta_int,
                alpha_int,
                scale,
                &column.suffix_compatibility,
                &observed,
            );
            frontier = pruned.retained;
            dropped_states += pruned.dropped_states;
            dropped_log_mass = dropped_log_mass.max(pruned.dropped_log_mass);
            k_capped |= pruned.k_capped;
            delta_pruned |= pruned.delta_pruned;
            peak_retained_states = peak_retained_states.max(frontier.len());
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

    fn bp_suffix_compatibility(
        &mut self,
        observed: &[u64],
    ) -> Result<BpSuffixPreparation, DecoderError> {
        let Kernel::Binary(columns) = &self.kernel else {
            return Ok((None, 0.0));
        };
        let Some(bp_score) = &mut self.bp_score else {
            return Ok((None, 0.0));
        };

        let started = Instant::now();
        for (detector, residual) in bp_score.residual_syndrome.iter_mut().enumerate() {
            let word_index = detector / WORD_BITS;
            let bit_mask = 1 << (detector % WORD_BITS);
            *residual =
                u8::from((observed[word_index] ^ self.forced_syndrome[word_index]) & bit_mask != 0);
        }
        min_sum_bp_into(
            &bp_score.graph,
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
        let moments: Vec<f64> = bp_score
            .posterior
            .iter()
            .map(|&llr| 1.0 - 2.0 * bp_score_probability(llr))
            .collect();
        let tables = build_suffix_compatibility_tables(
            columns,
            &moments,
            self.num_detectors,
            self.config.int_metric_scale,
        );
        Ok((Some(tables), started.elapsed().as_secs_f64()))
    }
}

impl ObservableDecoder for TrellisDecoder {
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        Ok(self.decode(syndrome)?.predicted)
    }

    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        if self.logical_words > 1 {
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
    if config.k == 0 {
        return Err(DecoderError::InvalidConfiguration(
            "TrellisConfig.k must be at least 1".into(),
        ));
    }
    if config.delta.is_nan() || config.delta < 0.0 {
        return Err(DecoderError::InvalidConfiguration(format!(
            "TrellisConfig.delta must be non-negative and not NaN, got {}",
            config.delta
        )));
    }
    if config.metric_mode == MetricMode::MaxLogInt && !config.delta.is_finite() {
        return Err(DecoderError::InvalidConfiguration(
            "delta must be finite under maxlog_int; infinite delta would quantize to zero and prune to score-ties"
                .into(),
        ));
    }
    if config.int_metric_scale <= 0 {
        return Err(DecoderError::InvalidConfiguration(
            "TrellisConfig.int_metric_scale must be positive".into(),
        ));
    }
    if !config.score_alpha.is_finite() || config.score_alpha < 0.0 {
        return Err(DecoderError::InvalidConfiguration(format!(
            "TrellisConfig.score_alpha must be finite and non-negative, got {}",
            config.score_alpha
        )));
    }
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
/// intermediate `long double` width is x86-specific; PECOS requires exact
/// agreement at the fixture values rather than bit-replicating that type.
fn quantize_metric(value: f64, scale: i32) -> i64 {
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
    debug_assert!(value.is_finite() && value.fract() == 0.0);
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

fn merge_branch(
    merged: &mut BTreeMap<StateKey, f64>,
    mut state: StateKey,
    log_mass: f64,
    close_mask: &[u64],
    active_mask: &[u64],
    observed: &[u64],
    transitions: &mut u64,
) {
    *transitions += 1;
    if state
        .active_syndrome
        .iter()
        .zip(observed)
        .zip(close_mask)
        .any(|((&accumulated, &expected), &closing)| (accumulated ^ expected) & closing != 0)
    {
        return;
    }
    and_assign(&mut state.active_syndrome, active_mask);
    merged
        .entry(state)
        .and_modify(|mass| *mass = logaddexp(*mass, log_mass))
        .or_insert(log_mass);
}

fn merge_branch_maxlog(
    merged: &mut BTreeMap<StateKey, i64>,
    mut state: StateKey,
    log_mass: i64,
    close_mask: &[u64],
    active_mask: &[u64],
    observed: &[u64],
    transitions: &mut u64,
) {
    *transitions += 1;
    if state
        .active_syndrome
        .iter()
        .zip(observed)
        .zip(close_mask)
        .any(|((&accumulated, &expected), &closing)| (accumulated ^ expected) & closing != 0)
    {
        return;
    }
    and_assign(&mut state.active_syndrome, active_mask);
    merged
        .entry(state)
        .and_modify(|mass| *mass = (*mass).max(log_mass))
        .or_insert(log_mass);
}

fn prune(
    frontier: BTreeMap<StateKey, f64>,
    k: usize,
    delta: f64,
    score_alpha: f64,
    suffix_compatibility: &[SuffixCompatibility],
    observed: &[u64],
) -> PruneResult {
    if k == usize::MAX && delta.is_infinite() {
        return PruneResult {
            retained: frontier,
            dropped_states: 0,
            dropped_log_mass: f64::NEG_INFINITY,
            k_capped: false,
            delta_pruned: false,
        };
    }

    let mut candidates: Vec<ScoredCandidate> = frontier
        .into_iter()
        .map(|(key, log_mass)| {
            let score = if score_alpha == 0.0 {
                log_mass
            } else {
                log_mass
                    + score_alpha
                        * suffix_compatibility_score(
                            &key.active_syndrome,
                            observed,
                            suffix_compatibility,
                        )
            };
            ScoredCandidate {
                candidate: Candidate { key, log_mass },
                score,
            }
        })
        .collect();
    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.candidate.key.cmp(&right.candidate.key))
    });
    let cutoff = candidates[0].score - delta;
    let mut retained = BTreeMap::new();
    let mut dropped_states = 0;
    let mut dropped_log_mass = f64::NEG_INFINITY;
    let mut k_capped = false;
    let mut delta_pruned = false;

    for (index, scored) in candidates.into_iter().enumerate() {
        let within_k = index < k;
        let within_delta = scored.score >= cutoff;
        if within_k && within_delta {
            retained.insert(scored.candidate.key, scored.candidate.log_mass);
        } else {
            dropped_states += 1;
            dropped_log_mass = logaddexp(dropped_log_mass, scored.candidate.log_mass);
            k_capped |= !within_k;
            delta_pruned |= within_k && !within_delta;
        }
    }

    PruneResult {
        retained,
        dropped_states,
        dropped_log_mass,
        k_capped,
        delta_pruned,
    }
}

fn prune_maxlog(
    frontier: BTreeMap<StateKey, i64>,
    k: usize,
    delta_int: i64,
    alpha_int: i64,
    scale: i32,
    suffix_compatibility: &[SuffixCompatibility],
    observed: &[u64],
) -> IntPruneResult {
    let mut candidates: Vec<ScoredIntCandidate> = frontier
        .into_iter()
        .map(|(key, log_mass)| {
            let parity = suffix_compatibility_score_int(
                &key.active_syndrome,
                observed,
                suffix_compatibility,
            );
            ScoredIntCandidate {
                candidate: IntCandidate { key, log_mass },
                score: score_int_metric(log_mass, parity, alpha_int, scale),
            }
        })
        .collect();
    candidates.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.candidate.key.cmp(&right.candidate.key))
    });
    let cutoff = candidates[0].score.saturating_sub(delta_int);
    let mut retained = BTreeMap::new();
    let mut dropped_states = 0;
    let mut dropped_log_mass = INT_METRIC_NEG_INF;
    let mut k_capped = false;
    let mut delta_pruned = false;

    for (index, scored) in candidates.into_iter().enumerate() {
        let within_k = index < k;
        let within_delta = scored.score >= cutoff;
        if within_k && within_delta {
            retained.insert(scored.candidate.key, scored.candidate.log_mass);
        } else {
            dropped_states += 1;
            dropped_log_mass = dropped_log_mass.max(scored.candidate.log_mass);
            k_capped |= !within_k;
            delta_pruned |= within_k && !within_delta;
        }
    }

    IntPruneResult {
        retained,
        dropped_states,
        dropped_log_mass,
        k_capped,
        delta_pruned,
    }
}

fn build_suffix_compatibility_tables(
    columns: &[Column],
    column_moments: &[f64],
    num_detectors: usize,
    int_metric_scale: i32,
) -> Vec<Vec<SuffixCompatibility>> {
    assert_eq!(columns.len(), column_moments.len());
    let mut tables = vec![Vec::new(); columns.len()];
    let mut row_moments = vec![1.0; num_detectors];
    for ((column, table), &moment) in columns
        .iter()
        .rev()
        .zip(tables.iter_mut().rev())
        .zip(column_moments.iter().rev())
    {
        *table = set_bits(&column.active_mask)
            .map(|detector| {
                let eta = row_moments[detector];
                let log_probability_zero = libm::log(1.0_f64.midpoint(eta));
                let log_probability_one = libm::log(1.0_f64.midpoint(-eta));
                SuffixCompatibility {
                    word_index: detector / WORD_BITS,
                    bit_mask: 1 << (detector % WORD_BITS),
                    log_probability_zero,
                    log_probability_one,
                    log_probability_zero_int: quantize_metric(
                        log_probability_zero,
                        int_metric_scale,
                    ),
                    log_probability_one_int: quantize_metric(log_probability_one, int_metric_scale),
                }
            })
            .collect();
        for detector in set_bits(&column.detector_toggle) {
            row_moments[detector] *= moment;
        }
    }
    tables
}

fn build_suffix_compatibility_tables_nary(
    columns: &[FactorColumn],
    num_detectors: usize,
    int_metric_scale: i32,
) -> Vec<Vec<SuffixCompatibility>> {
    let mut tables = vec![Vec::new(); columns.len()];
    let mut row_moments = vec![1.0; num_detectors];
    for (column, table) in columns.iter().rev().zip(tables.iter_mut().rev()) {
        *table = set_bits(&column.active_mask)
            .map(|detector| {
                let eta = row_moments[detector];
                let log_probability_zero = libm::log(1.0_f64.midpoint(eta));
                let log_probability_one = libm::log(1.0_f64.midpoint(-eta));
                SuffixCompatibility {
                    word_index: detector / WORD_BITS,
                    bit_mask: 1 << (detector % WORD_BITS),
                    log_probability_zero,
                    log_probability_one,
                    log_probability_zero_int: quantize_metric(
                        log_probability_zero,
                        int_metric_scale,
                    ),
                    log_probability_one_int: quantize_metric(log_probability_one, int_metric_scale),
                }
            })
            .collect();

        let mut support = vec![0; words_for(num_detectors)];
        for outcome in &column.outcomes {
            or_assign(&mut support, &outcome.detector_toggle);
        }
        for detector in set_bits(&support) {
            let word_index = detector / WORD_BITS;
            let bit_mask = 1 << (detector % WORD_BITS);
            let toggle_probability = column
                .outcomes
                .iter()
                .filter(|outcome| outcome.detector_toggle[word_index] & bit_mask != 0)
                .map(|outcome| outcome.probability)
                .sum::<f64>()
                .min(1.0);
            row_moments[detector] *= 1.0 - 2.0 * toggle_probability;
        }
    }
    tables
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

fn suffix_compatibility_score(
    active_syndrome: &[u64],
    observed: &[u64],
    suffix_compatibility: &[SuffixCompatibility],
) -> f64 {
    suffix_compatibility
        .iter()
        .map(|row| {
            if (active_syndrome[row.word_index] ^ observed[row.word_index]) & row.bit_mask == 0 {
                row.log_probability_zero
            } else {
                row.log_probability_one
            }
        })
        .sum()
}

fn suffix_compatibility_score_int(
    active_syndrome: &[u64],
    observed: &[u64],
    suffix_compatibility: &[SuffixCompatibility],
) -> i64 {
    suffix_compatibility.iter().fold(0, |total, row| {
        let term =
            if (active_syndrome[row.word_index] ^ observed[row.word_index]) & row.bit_mask == 0 {
                row.log_probability_zero_int
            } else {
                row.log_probability_one_int
            };
        int_metric_add(total, term)
    })
}

fn finish_maxlog_decode(
    frontier: BTreeMap<StateKey, i64>,
    scale: i32,
    stats: MaxLogDecodeStats,
) -> TrellisDecodeAttempt {
    let mut terminal_by_logical = BTreeMap::<Vec<u64>, i64>::new();
    for (key, log_mass) in frontier {
        terminal_by_logical
            .entry(key.logical)
            .and_modify(|mass| *mass = (*mass).max(log_mass))
            .or_insert(log_mass);
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
        for non_finite in [f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
            assert_eq!(quantize_metric(non_finite, 1024), INT_METRIC_NEG_INF);
        }
        assert_eq!(quantize_metric(f64::MIN, 1024), INT_METRIC_NEG_INF);
        assert_eq!(quantize_metric(f64::MAX, 1024), INT_METRIC_MAX);
        assert_eq!(
            quantize_metric(i64_to_f64(INT_METRIC_NEG_INF), 1),
            INT_METRIC_NEG_INF
        );
        assert_eq!(
            quantize_metric(i64_to_f64(INT_METRIC_MAX), 1),
            INT_METRIC_MAX
        );
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
