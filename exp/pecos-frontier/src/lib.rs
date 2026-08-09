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
//! configured frontier width and log-mass window provide deterministic pruning
//! for a fixed build and platform; underlying `ln`/`exp` implementations may
//! differ across platforms.
//!
//! Pruning ranks accumulated prefix log mass plus a `score_alpha`-weighted
//! suffix-compatibility estimate. Unpruned results are exact and
//! upstream-verified.

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

/// Frontier pruning and column-order configuration.
///
/// The [`Default`] pruning values are provisional pending benchmarking.
/// Pruning ranks accumulated prefix log mass plus a `score_alpha`-weighted
/// suffix-compatibility estimate. Unpruned results are exact and
/// upstream-verified.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontierConfig {
    /// Maximum number of boundary states retained after each column.
    pub k: usize,
    /// Log-mass window below the best boundary state retained after each column.
    pub delta: f64,
    /// Weight applied to the suffix-compatibility score during pruning.
    /// Defaults to `0.8`, matching upstream Frontier.
    pub score_alpha: f64,
    /// Optional permutation of the DEM mechanism indices.
    pub column_order: Option<Vec<usize>>,
    /// Merge probabilistic mechanisms with identical detector and observable
    /// sets using their XOR-combined probability.
    ///
    /// This merge is mathematically exact, but it takes a different
    /// floating-point path and Frontier's parity contract is bitwise, so it is
    /// disabled by default. Zero-probability mechanisms are already discarded,
    /// while probability-one mechanisms remain separate in the forced layer
    /// and are not merged with otherwise identical probabilistic mechanisms.
    pub merge_indistinguishable: bool,
    /// Number of min-sum BP iterations used only to score pruning candidates.
    /// Zero disables BP-informed scoring.
    pub bp_score_iterations: usize,
}

impl Default for FrontierConfig {
    fn default() -> Self {
        // Provisional defaults pending benchmarking.
        Self {
            k: 64,
            delta: 50.0,
            score_alpha: 0.8,
            column_order: None,
            merge_indistinguishable: false,
            bp_score_iterations: 0,
        }
    }
}

/// Generate the deadline-optimized processing order for a sparse DEM.
///
/// The input mechanism order is treated as time order. Mechanisms that can
/// close detectors earlier are placed first; detector-free mechanisms sort
/// last. The returned permutation maps target positions to source mechanism
/// indices and can be assigned directly to [`FrontierConfig::column_order`].
///
/// # Errors
///
/// Returns [`DecoderError::InvalidConfiguration`] if a mechanism contains an
/// out-of-range or duplicate detector index.
pub fn deadline_column_order(dem: &SparseDem) -> Result<Vec<usize>, DecoderError> {
    let time_order: Vec<usize> = (0..dem.mechanisms.len()).collect();
    deadline_order_for_sequence(dem, &time_order)
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
    let mut reversed_forward = deadline_column_order(dem)?;
    reversed_forward.reverse();
    deadline_order_for_sequence(dem, &reversed_forward)
}

/// Retained unnormalized joint log mass for one logical label.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontierLogicalMass {
    /// Logical-observable flip label.
    pub logical: ObsMask,
    /// Unnormalized joint mass `ln P(logical class, observed syndrome)`.
    /// Subtract [`FrontierResult::log_evidence`] to obtain the label's log
    /// posterior probability within the retained terminal mass.
    pub log_mass: f64,
}

/// Completeness status of one successful Frontier decode.
///
/// `NoPath` remains a [`DecoderError`]. This envelope will gain a budget arm
/// only when the decoder has an actual budget mechanism.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontierStatus {
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

/// Result of one Frontier decode.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontierResult {
    /// Predicted logical-observable flip mask.
    pub predicted: ObsMask,
    /// Log evidence: the logarithm of the total retained joint mass over all
    /// terminal logical labels, approximating `ln P(observed syndrome)` when
    /// pruning is enabled.
    ///
    /// The winning label's own log mass is [`Self::logical_masses`]'s first
    /// entry.
    pub log_evidence: f64,
    /// Difference between the winning and runner-up unnormalized joint log
    /// masses, if a runner-up exists.
    pub runner_up_gap: Option<f64>,
    /// Largest retained frontier size, including the initial boundary state.
    pub peak_retained_states: usize,
    /// Number of probabilistic columns processed (`0 < p < 1`).
    pub processed_columns: usize,
    /// Number of candidate branch evaluations, counted at entry to
    /// `merge_branch` (two per retained state for every processed column).
    pub transitions: u64,
    /// Number of merged boundary states discarded across all pruning calls.
    pub dropped_states: u64,
    /// Log-sum-exp of the log masses of all states discarded by pruning, or
    /// negative infinity when no state was discarded.
    ///
    /// This accounts for retained prefix mass discarded at pruning time. It is
    /// not a bound on true lost posterior mass: a state dropped early would
    /// otherwise have branched through later columns.
    pub dropped_log_mass: f64,
    /// Wall-clock seconds spent producing BP-informed suffix scores for this
    /// shot. This is zero when BP scoring is disabled or pruning cannot run.
    pub bp_seconds: f64,
    /// Whether the successful decode was exact or which pruning mechanisms
    /// discarded at least one state.
    pub status: FrontierStatus,
    /// Retained unnormalized joint terminal masses, ordered by mass descending
    /// and numeric label ascending. The first entry is the winning label and
    /// its retained log mass.
    pub logical_masses: Vec<FrontierLogicalMass>,
}

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
    /// The leg found no retained path for the syndrome.
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

#[derive(Clone, Debug)]
struct Column {
    detector_toggle: Vec<u64>,
    logical_toggle: Vec<u64>,
    close_mask: Vec<u64>,
    active_mask: Vec<u64>,
    suffix_compatibility: Vec<SuffixCompatibility>,
    log_odds: f64,
    log_one_minus_probability: f64,
}

#[derive(Clone, Debug)]
struct SuffixCompatibility {
    word_index: usize,
    bit_mask: u64,
    log_probability_zero: f64,
    log_probability_one: f64,
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

/// Ordered, pruned dynamic-programming decoder for sparse detector error models.
#[derive(Clone, Debug)]
pub struct FrontierDecoder {
    config: FrontierConfig,
    columns: Vec<Column>,
    num_detectors: usize,
    detector_words: usize,
    logical_words: usize,
    touched_detectors: Vec<u64>,
    forced_syndrome: Vec<u64>,
    forced_logical: Vec<u64>,
    bp_score: Option<BpScoreState>,
    build_seconds: f64,
}

/// Two-leg Frontier decoder using a processing order and its plain reverse.
#[derive(Clone, Debug)]
pub struct FrontierCommittee {
    forward: FrontierDecoder,
    backward: FrontierDecoder,
    build_seconds: f64,
}

impl FrontierDecoder {
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
    pub fn from_sparse_dem(dem: &SparseDem, config: FrontierConfig) -> Result<Self, DecoderError> {
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
            validate_indices(detectors, dem.num_detectors, "detector", mechanism_index)?;
            validate_indices(
                observables,
                dem.num_observables,
                "observable",
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
            columns.push(Column {
                detector_toggle,
                logical_toggle,
                close_mask,
                active_mask: open_detectors.clone(),
                suffix_compatibility: Vec::new(),
                log_odds: (probability / (1.0 - probability)).ln(),
                log_one_minus_probability: (1.0 - probability).ln(),
            });
        }

        let suffix_tables =
            build_suffix_compatibility_tables(&columns, &column_moments, dem.num_detectors);
        for (column, suffix_compatibility) in columns.iter_mut().zip(suffix_tables) {
            column.suffix_compatibility = suffix_compatibility;
        }

        debug_assert_model_invariants(&columns, &touched_detectors);
        let build_seconds = build_started.elapsed().as_secs_f64();

        Ok(Self {
            config,
            columns,
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

    /// Wall-clock seconds spent constructing this model in
    /// [`Self::from_sparse_dem`].
    #[must_use]
    pub fn build_seconds(&self) -> f64 {
        self.build_seconds
    }

    /// Parse a Stim-format detector error model and construct a decoder.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if parsing or decoder validation fails.
    pub fn from_dem_str(dem_str: &str, config: FrontierConfig) -> Result<Self, DecoderError> {
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
    pub fn decode(&mut self, syndrome: &[u8]) -> Result<FrontierResult, DecoderError> {
        if syndrome.len() != self.num_detectors {
            return Err(DecoderError::InvalidDimensions {
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
            return Err(unexplainable_error());
        }

        let (bp_suffix_compatibility, bp_seconds) = self.bp_suffix_compatibility(&observed)?;

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

        for (column_index, column) in self.columns.iter().enumerate() {
            let mut merged = BTreeMap::new();
            for (state, &log_mass) in &frontier {
                let branch_base = log_mass + column.log_one_minus_probability;
                merge_branch(
                    &mut merged,
                    state.clone(),
                    branch_base,
                    column,
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
                    column,
                    &observed,
                    &mut transitions,
                );
            }

            if merged.is_empty() {
                return Err(unexplainable_error());
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
                return Err(unexplainable_error());
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
            .map(|candidate| FrontierLogicalMass {
                logical: ObsMask::from_words(&candidate.key.logical),
                log_mass: candidate.log_mass,
            })
            .collect();
        let status = if dropped_states == 0 {
            FrontierStatus::Exact
        } else {
            FrontierStatus::Pruned {
                k_capped,
                delta_pruned,
            }
        };

        Ok(FrontierResult {
            predicted: ObsMask::from_words(&winner.key.logical),
            log_evidence,
            runner_up_gap: terminal
                .get(1)
                .map(|runner_up| winner.log_mass - runner_up.log_mass),
            peak_retained_states,
            processed_columns: self.columns.len(),
            transitions,
            dropped_states,
            dropped_log_mass,
            bp_seconds,
            status,
            logical_masses,
        })
    }

    fn bp_suffix_compatibility(
        &mut self,
        observed: &[u64],
    ) -> Result<BpSuffixPreparation, DecoderError> {
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
            self.columns.len(),
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
        let tables = build_suffix_compatibility_tables(&self.columns, &moments, self.num_detectors);
        Ok((Some(tables), started.elapsed().as_secs_f64()))
    }
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
        let build_started = Instant::now();
        let FrontierConfig {
            k,
            delta,
            score_alpha,
            column_order,
            merge_indistinguishable,
            bp_score_iterations,
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
        };
        let backward = FrontierDecoder::from_sparse_dem(dem, backward_config)?;
        let build_seconds = build_started.elapsed().as_secs_f64();
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

    /// Parse a Stim-format detector error model and construct a committee.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if parsing or committee construction fails.
    pub fn from_dem_str(dem_str: &str, config: FrontierConfig) -> Result<Self, DecoderError> {
        let dem = SparseDem::from_dem_str(dem_str)?;
        Self::from_sparse_dem(&dem, config)
    }

    /// Decode with both processing directions and select the stronger result.
    ///
    /// # Errors
    ///
    /// Returns the standard unexplainable-syndrome error if both legs find no
    /// retained path.
    pub fn decode(&mut self, syndrome: &[u8]) -> Result<FrontierCommitteeResult, DecoderError> {
        // Each leg owns its BP graph and scratch and runs independently on the
        // same syndrome. Their beliefs are expected to match, but are never
        // shared by reference between the two code paths.
        let forward_result = self.forward.decode(syndrome);
        let backward_result = self.backward.decode(syndrome);
        let forward = committee_member(&forward_result);
        let backward = committee_member(&backward_result);
        let (selected, direction) = match (forward_result, backward_result) {
            (Err(_), Err(_)) => return Err(unexplainable_error()),
            (Ok(selected), Err(_)) => (selected, CommitteeDirection::Forward),
            (Err(_), Ok(selected)) => (selected, CommitteeDirection::Backward),
            (Ok(forward_result), Ok(backward_result)) => {
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
}

impl ObservableDecoder for FrontierDecoder {
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

impl ObservableDecoder for FrontierCommittee {
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        Ok(self.decode(syndrome)?.selected.predicted)
    }
}

type DeadlineKey = (usize, usize, usize, usize, usize);

fn deadline_order_for_sequence(
    dem: &SparseDem,
    sequence: &[usize],
) -> Result<Vec<usize>, DecoderError> {
    let sentinel = dem.mechanisms.len() + 1;
    let mut first_touch = vec![sentinel; dem.num_detectors];
    let mut last_touch = vec![sentinel; dem.num_detectors];

    for (position, &mechanism_index) in sequence.iter().enumerate() {
        let detectors = &dem.mechanisms[mechanism_index].1;
        validate_indices(detectors, dem.num_detectors, "detector", mechanism_index)?;
        for &detector in detectors {
            let detector = detector as usize;
            first_touch[detector] = first_touch[detector].min(position);
            last_touch[detector] = position;
        }
    }

    let mut positions: Vec<usize> = (0..sequence.len()).collect();
    positions.sort_by_key(|&position| -> DeadlineKey {
        let mechanism_index = sequence[position];
        let detectors = &dem.mechanisms[mechanism_index].1;
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

fn committee_member(result: &Result<FrontierResult, DecoderError>) -> CommitteeMember {
    match result {
        Ok(decoded) => CommitteeMember {
            status: CommitteeStatus::Ok,
            log_evidence: decoded.log_evidence,
        },
        Err(_) => CommitteeMember {
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

fn validate_config(config: &FrontierConfig, mechanism_count: usize) -> Result<(), DecoderError> {
    if config.k == 0 {
        return Err(DecoderError::InvalidConfiguration(
            "FrontierConfig.k must be at least 1".into(),
        ));
    }
    if config.delta.is_nan() || config.delta < 0.0 {
        return Err(DecoderError::InvalidConfiguration(format!(
            "FrontierConfig.delta must be non-negative and not NaN, got {}",
            config.delta
        )));
    }
    if !config.score_alpha.is_finite() || config.score_alpha < 0.0 {
        return Err(DecoderError::InvalidConfiguration(format!(
            "FrontierConfig.score_alpha must be finite and non-negative, got {}",
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

fn validate_indices(
    indices: &[u32],
    upper_bound: usize,
    kind: &str,
    mechanism_index: usize,
) -> Result<(), DecoderError> {
    let mut seen = std::collections::BTreeSet::new();
    for &index in indices {
        if index as usize >= upper_bound {
            return Err(DecoderError::InvalidConfiguration(format!(
                "mechanism {mechanism_index} {kind} index {index} is out of range 0..{upper_bound}"
            )));
        }
        if !seen.insert(index) {
            return Err(DecoderError::InvalidConfiguration(format!(
                "mechanism {mechanism_index} repeats {kind} index {index}"
            )));
        }
    }
    Ok(())
}

fn compare_words_as_unsigned(left: &[u64], right: &[u64]) -> Ordering {
    left.iter().rev().cmp(right.iter().rev())
}

fn merge_branch(
    merged: &mut BTreeMap<StateKey, f64>,
    mut state: StateKey,
    log_mass: f64,
    column: &Column,
    observed: &[u64],
    transitions: &mut u64,
) {
    *transitions += 1;
    if state
        .active_syndrome
        .iter()
        .zip(observed)
        .zip(&column.close_mask)
        .any(|((&accumulated, &expected), &closing)| (accumulated ^ expected) & closing != 0)
    {
        return;
    }
    and_assign(&mut state.active_syndrome, &column.active_mask);
    merged
        .entry(state)
        .and_modify(|mass| *mass = logaddexp(*mass, log_mass))
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

fn build_suffix_compatibility_tables(
    columns: &[Column],
    column_moments: &[f64],
    num_detectors: usize,
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
                SuffixCompatibility {
                    word_index: detector / WORD_BITS,
                    bit_mask: 1 << (detector % WORD_BITS),
                    log_probability_zero: 1.0_f64.midpoint(eta).ln(),
                    log_probability_one: 1.0_f64.midpoint(-eta).ln(),
                }
            })
            .collect();
        for detector in set_bits(&column.detector_toggle) {
            row_moments[detector] *= moment;
        }
    }
    tables
}

fn bp_score_probability(posterior_llr: f64) -> f64 {
    let probability = 1.0 / (1.0 + posterior_llr.exp());
    probability.clamp(BP_SCORE_PROBABILITY_MIN, 1.0 - BP_SCORE_PROBABILITY_MIN)
}

fn debug_assert_model_invariants(columns: &[Column], touched_detectors: &[u64]) {
    #[cfg(debug_assertions)]
    {
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
    high + (low - high).exp().ln_1p()
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
    words.iter().enumerate().flat_map(|(word_index, &word)| {
        (0..WORD_BITS)
            .filter(move |&bit| word & (1 << bit) != 0)
            .map(move |bit| word_index * WORD_BITS + bit)
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
        FrontierCommittee, FrontierConfig, FrontierDecoder, FrontierLogicalMass, FrontierResult,
        FrontierStatus, SparseDem, bp_score_probability, committee_rank, logaddexp,
        merge_indistinguishable_columns,
    };
    use pecos_decoder_core::obs_mask::ObsMask;
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
        let mut decoder = FrontierDecoder::from_sparse_dem(
            &dem,
            FrontierConfig {
                merge_indistinguishable: true,
                ..FrontierConfig::default()
            },
        )
        .unwrap();
        let result = decoder.decode(&[1]).unwrap();
        assert_eq!(result.processed_columns, 1);
        assert_eq!(result.log_evidence.to_bits(), 0.5_f64.ln().to_bits());
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
    fn committee_bp_legs_own_and_run_independent_state() {
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
        let mut committee = FrontierCommittee::from_sparse_dem(
            &dem,
            FrontierConfig {
                k: 1,
                delta: f64::INFINITY,
                score_alpha: 0.8,
                column_order: None,
                merge_indistinguishable: false,
                bp_score_iterations: 5,
            },
        )
        .unwrap();
        let forward_graph = &committee.forward.bp_score.as_ref().unwrap().graph;
        let backward_graph = &committee.backward.bp_score.as_ref().unwrap().graph;
        assert!(!std::ptr::eq(forward_graph, backward_graph));

        let forward = committee.forward.decode(&[1, 0]).unwrap();
        let backward = committee.backward.decode(&[1, 0]).unwrap();
        assert!(forward.bp_seconds > 0.0);
        assert!(backward.bp_seconds > 0.0);
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
