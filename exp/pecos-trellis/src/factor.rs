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

//! Multi-outcome factor-model inputs for the trellis decoder.

use crate::{DecoderError, SparseDem};
use std::collections::BTreeSet;

const PROBABILITY_SUM_TOLERANCE: f64 = 1e-10;

/// One mutually exclusive outcome of a factor.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// Probability of this outcome.
    pub probability: f64,
    /// Detector indices toggled by this outcome.
    pub detectors: Vec<u32>,
    /// Observable indices toggled by this outcome.
    pub observables: Vec<u32>,
}

impl Outcome {
    pub(crate) fn is_empty(&self) -> bool {
        self.detectors.is_empty() && self.observables.is_empty()
    }
}

/// A collection of mutually exclusive outcomes.
#[derive(Clone, Debug, PartialEq)]
pub struct Factor {
    /// Outcomes belonging to this factor.
    pub outcomes: Vec<Outcome>,
}

/// A validated collection of independent factors.
#[derive(Clone, Debug, PartialEq)]
pub struct FactorModel {
    factors: Vec<Factor>,
    num_detectors: usize,
    num_observables: usize,
}

#[derive(Clone, Debug)]
pub(crate) enum NormalizedFactor {
    Forced(Outcome),
    Binary(Outcome),
    Nary(Vec<Outcome>),
}

impl FactorModel {
    /// Validate and construct a factor model.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] if a factor has no
    /// outcomes, an outcome probability or index is invalid, an outcome
    /// repeats an index, or a factor's probabilities do not sum to one within
    /// `1e-10`.
    pub fn new(
        factors: Vec<Factor>,
        num_detectors: usize,
        num_observables: usize,
    ) -> Result<Self, DecoderError> {
        for (factor_index, factor) in factors.iter().enumerate() {
            if factor.outcomes.is_empty() {
                return Err(DecoderError::InvalidConfiguration(format!(
                    "factor {factor_index} must have at least one outcome"
                )));
            }

            let mut probability_sum = 0.0;
            for (outcome_index, outcome) in factor.outcomes.iter().enumerate() {
                if !(0.0..=1.0).contains(&outcome.probability) {
                    return Err(DecoderError::InvalidConfiguration(format!(
                        "factor {factor_index} outcome {outcome_index} probability must satisfy 0 <= p <= 1, got {}",
                        outcome.probability
                    )));
                }
                validate_outcome_indices(
                    &outcome.detectors,
                    num_detectors,
                    "detector",
                    factor_index,
                    outcome_index,
                )?;
                validate_outcome_indices(
                    &outcome.observables,
                    num_observables,
                    "observable",
                    factor_index,
                    outcome_index,
                )?;
                probability_sum += outcome.probability;
            }
            if (probability_sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
                return Err(DecoderError::InvalidConfiguration(format!(
                    "factor {factor_index} outcome probabilities must sum to 1 within {PROBABILITY_SUM_TOLERANCE}, got {probability_sum}"
                )));
            }
        }

        Ok(Self {
            factors,
            num_detectors,
            num_observables,
        })
    }

    /// Factors in their input processing order.
    #[must_use]
    pub fn factors(&self) -> &[Factor] {
        &self.factors
    }

    /// Number of detector bits in the model.
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Number of logical-observable bits in the model.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    pub(crate) fn normalized_factor(&self, factor_index: usize) -> NormalizedFactor {
        normalize_factor(&self.factors[factor_index])
    }

    pub(crate) fn is_binary_shaped(&self) -> bool {
        self.factors
            .iter()
            .all(|factor| !matches!(normalize_factor(factor), NormalizedFactor::Nary(_)))
    }
}

impl From<&SparseDem> for FactorModel {
    fn from(dem: &SparseDem) -> Self {
        let factors = dem
            .mechanisms
            .iter()
            .map(|(probability, detectors, observables)| Factor {
                outcomes: vec![
                    Outcome {
                        probability: 1.0 - probability,
                        detectors: Vec::new(),
                        observables: Vec::new(),
                    },
                    Outcome {
                        probability: *probability,
                        detectors: detectors.clone(),
                        observables: observables.clone(),
                    },
                ],
            })
            .collect();
        Self {
            factors,
            num_detectors: dem.num_detectors,
            num_observables: dem.num_observables,
        }
    }
}

fn normalize_factor(factor: &Factor) -> NormalizedFactor {
    let outcomes: Vec<Outcome> = factor
        .outcomes
        .iter()
        .filter(|outcome| outcome.probability != 0.0)
        .cloned()
        .collect();
    debug_assert!(!outcomes.is_empty());

    if outcomes.len() == 1 {
        return NormalizedFactor::Forced(outcomes.into_iter().next().unwrap());
    }
    if outcomes.len() == 2 && outcomes.iter().any(Outcome::is_empty) {
        // When both outcomes are empty, outcome 1 is the deterministic toggle
        // choice. This pins normalization even though either choice is
        // mathematically equivalent.
        let toggle_index = usize::from(outcomes[0].is_empty());
        return NormalizedFactor::Binary(outcomes[toggle_index].clone());
    }
    NormalizedFactor::Nary(outcomes)
}

fn validate_outcome_indices(
    indices: &[u32],
    upper_bound: usize,
    kind: &str,
    factor_index: usize,
    outcome_index: usize,
) -> Result<(), DecoderError> {
    let mut seen = BTreeSet::new();
    for &index in indices {
        if index as usize >= upper_bound {
            return Err(DecoderError::InvalidConfiguration(format!(
                "factor {factor_index} outcome {outcome_index} {kind} index {index} is out of range 0..{upper_bound}"
            )));
        }
        if !seen.insert(index) {
            return Err(DecoderError::InvalidConfiguration(format!(
                "factor {factor_index} outcome {outcome_index} repeats {kind} index {index}"
            )));
        }
    }
    Ok(())
}
