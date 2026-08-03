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
//! configured frontier width and log-mass window provide deterministic pruning.

use pecos_decoder_core::ObservableDecoder;
use pecos_decoder_core::dem::SparseDem;
use pecos_decoder_core::errors::DecoderError;
use pecos_decoder_core::obs_mask::ObsMask;
use std::cmp::Ordering;
use std::collections::BTreeMap;

const WORD_BITS: usize = u64::BITS as usize;

/// Frontier pruning and column-order configuration.
///
/// The [`Default`] pruning values are provisional pending benchmarking.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontierConfig {
    /// Maximum number of boundary states retained after each column.
    pub k: usize,
    /// Log-mass window below the best boundary state retained after each column.
    pub delta: f64,
    /// Optional permutation of the DEM mechanism indices.
    pub column_order: Option<Vec<usize>>,
}

impl Default for FrontierConfig {
    fn default() -> Self {
        // Provisional defaults pending benchmarking.
        Self {
            k: 64,
            delta: 50.0,
            column_order: None,
        }
    }
}

/// Retained posterior log mass for one logical label.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontierLogicalMass {
    /// Logical-observable flip label.
    pub logical: ObsMask,
    /// Natural logarithm of the retained probability mass for this label.
    pub log_mass: f64,
}

/// Result of one Frontier decode.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontierResult {
    /// Predicted logical-observable flip mask.
    pub predicted: ObsMask,
    /// Natural logarithm of the retained probability mass of `predicted`.
    pub log_evidence: f64,
    /// Difference between the winning and runner-up log masses, if one exists.
    pub runner_up_gap: Option<f64>,
    /// Largest retained frontier size, including the initial boundary state.
    pub peak_retained_states: usize,
    /// Number of nonzero-probability columns processed.
    pub processed_columns: usize,
    /// Retained terminal masses, ordered by mass descending and label ascending.
    pub logical_masses: Vec<FrontierLogicalMass>,
}

#[derive(Clone, Debug)]
struct Column {
    detector_toggle: Vec<u64>,
    logical_toggle: Vec<u64>,
    close_mask: Vec<u64>,
    active_mask: Vec<u64>,
    log_odds: f64,
    log_one_minus_probability: f64,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct StateKey {
    active_syndrome: Vec<u64>,
    logical: Vec<u64>,
}

#[derive(Clone, Debug)]
struct Candidate {
    key: StateKey,
    log_mass: f64,
}

/// Ordered, pruned dynamic-programming decoder for sparse detector error models.
#[derive(Clone, Debug)]
pub struct FrontierDecoder {
    config: FrontierConfig,
    columns: Vec<Column>,
    num_detectors: usize,
    detector_words: usize,
    logical_words: usize,
    touched_detectors: Vec<u64>,
}

impl FrontierDecoder {
    /// Construct a decoder from a sparse detector error model.
    ///
    /// Zero-probability mechanisms are discarded after validating the optional
    /// ordering permutation. All indices and nonzero probabilities are checked.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] for invalid probabilities,
    /// indices, or column order.
    pub fn from_sparse_dem(dem: &SparseDem, config: FrontierConfig) -> Result<Self, DecoderError> {
        validate_column_order(&config, dem.mechanisms.len())?;

        let detector_words = words_for(dem.num_detectors);
        let logical_words = words_for(dem.num_observables);
        let order = config
            .column_order
            .clone()
            .unwrap_or_else(|| (0..dem.mechanisms.len()).collect());
        let mut raw_columns = Vec::with_capacity(dem.mechanisms.len());

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

            raw_columns.push((
                indices_to_words(detectors, detector_words),
                indices_to_words(observables, logical_words),
                *probability,
            ));
        }

        let mut touched_detectors = vec![0; detector_words];
        let mut last_touch = vec![None; dem.num_detectors];
        for (column_index, (detectors, _, _)) in raw_columns.iter().enumerate() {
            or_assign(&mut touched_detectors, detectors);
            for detector in set_bits(detectors) {
                last_touch[detector] = Some(column_index);
            }
        }

        let mut open_detectors = vec![0; detector_words];
        let mut columns = Vec::with_capacity(raw_columns.len());
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

            columns.push(Column {
                detector_toggle,
                logical_toggle,
                close_mask,
                active_mask: open_detectors.clone(),
                log_odds: (probability / (1.0 - probability)).ln(),
                log_one_minus_probability: (1.0 - probability).ln(),
            });
        }

        Ok(Self {
            config,
            columns,
            num_detectors: dem.num_detectors,
            detector_words,
            logical_words,
            touched_detectors,
        })
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
            .zip(&self.touched_detectors)
            .any(|(&seen, &touched)| seen & !touched != 0)
        {
            return Err(unexplainable_error());
        }

        let initial = StateKey {
            active_syndrome: vec![0; self.detector_words],
            logical: vec![0; self.logical_words],
        };
        let mut frontier = BTreeMap::from([(initial, 0.0)]);
        let mut peak_retained_states = frontier.len();

        for column in &self.columns {
            let mut merged = BTreeMap::new();
            for (state, &log_mass) in &frontier {
                let branch_base = log_mass + column.log_one_minus_probability;
                merge_branch(&mut merged, state.clone(), branch_base, column, &observed);

                let mut taken = state.clone();
                xor_assign(&mut taken.active_syndrome, &column.detector_toggle);
                xor_assign(&mut taken.logical, &column.logical_toggle);
                merge_branch(
                    &mut merged,
                    taken,
                    branch_base + column.log_odds,
                    column,
                    &observed,
                );
            }

            if merged.is_empty() {
                return Err(unexplainable_error());
            }
            frontier = prune(merged, self.config.k, self.config.delta);
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
        let logical_masses = terminal
            .iter()
            .map(|candidate| FrontierLogicalMass {
                logical: ObsMask::from_words(&candidate.key.logical),
                log_mass: candidate.log_mass,
            })
            .collect();

        Ok(FrontierResult {
            predicted: ObsMask::from_words(&winner.key.logical),
            log_evidence: winner.log_mass,
            runner_up_gap: terminal
                .get(1)
                .map(|runner_up| winner.log_mass - runner_up.log_mass),
            peak_retained_states,
            processed_columns: self.columns.len(),
            logical_masses,
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

fn validate_column_order(
    config: &FrontierConfig,
    mechanism_count: usize,
) -> Result<(), DecoderError> {
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
    if !(0.0..1.0).contains(&probability) {
        return Err(DecoderError::InvalidConfiguration(format!(
            "mechanism {index} probability must satisfy 0 <= p < 1, got {probability}"
        )));
    }
    Ok(())
}

fn validate_indices(
    indices: &[u32],
    upper_bound: usize,
    kind: &str,
    mechanism_index: usize,
) -> Result<(), DecoderError> {
    if let Some(&index) = indices.iter().find(|&&index| index as usize >= upper_bound) {
        return Err(DecoderError::InvalidConfiguration(format!(
            "mechanism {mechanism_index} {kind} index {index} is out of range 0..{upper_bound}"
        )));
    }
    Ok(())
}

fn merge_branch(
    merged: &mut BTreeMap<StateKey, f64>,
    mut state: StateKey,
    log_mass: f64,
    column: &Column,
    observed: &[u64],
) {
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

fn prune(frontier: BTreeMap<StateKey, f64>, k: usize, delta: f64) -> BTreeMap<StateKey, f64> {
    let mut candidates: Vec<Candidate> = frontier
        .into_iter()
        .map(|(key, log_mass)| Candidate { key, log_mass })
        .collect();
    sort_candidates(&mut candidates);
    let cutoff = candidates[0].log_mass - delta;
    candidates
        .into_iter()
        .take(k)
        .take_while(|candidate| candidate.log_mass >= cutoff)
        .map(|candidate| (candidate.key, candidate.log_mass))
        .collect()
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
    use super::logaddexp;

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
}
