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
const BINARY_COMPLEMENT_RELATIVE_TOLERANCE: f64 = 1e-9;

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
    Binary {
        outcomes: Vec<Outcome>,
        toggle: Outcome,
    },
    Nary(Vec<Outcome>),
}

impl FactorModel {
    /// Validate and construct a factor model.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] if a declared width
    /// exceeds the `u32` outcome-index range, a factor has no outcomes, an
    /// outcome probability or index is invalid, an outcome repeats an index,
    /// or a factor's probabilities do not sum to one within `1e-10`.
    pub fn new(
        factors: Vec<Factor>,
        num_detectors: usize,
        num_observables: usize,
    ) -> Result<Self, DecoderError> {
        // Indices are u32, so the largest addressable width is u32::MAX + 1
        // (index u32::MAX exists) -- the same contract as the DEM parser,
        // which accepts `D4294967295` and reports that width.
        const MAX_WIDTH: usize = u32::MAX as usize + 1;
        if num_detectors > MAX_WIDTH {
            return Err(DecoderError::InvalidConfiguration(format!(
                "num_detectors {num_detectors} exceeds the u32 index-addressable width {MAX_WIDTH}"
            )));
        }
        if num_observables > MAX_WIDTH {
            return Err(DecoderError::InvalidConfiguration(format!(
                "num_observables {num_observables} exceeds the u32 index-addressable width {MAX_WIDTH}"
            )));
        }

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

    pub(crate) fn normalized_factors(&self) -> Vec<NormalizedFactor> {
        self.factors.iter().map(normalize_factor).collect()
    }
}

impl TryFrom<&SparseDem> for FactorModel {
    type Error = DecoderError;

    fn try_from(dem: &SparseDem) -> Result<Self, Self::Error> {
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
        Self::new(factors, dem.num_detectors, dem.num_observables)
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
        // When exactly one outcome is empty the toggle is the non-empty one,
        // whatever its position. When BOTH are empty, position must not
        // decide (the gate divides by the baseline, so a position-dependent
        // choice made classification depend on listing order): take the
        // larger probability as the baseline, which is order-independent
        // (an exact tie is symmetric, so either choice yields identical
        // arithmetic).
        let toggle_index = if outcomes[0].is_empty() && outcomes[1].is_empty() {
            usize::from(outcomes[0].probability >= outcomes[1].probability)
        } else {
            usize::from(outcomes[0].is_empty())
        };
        let baseline_index = 1 - toggle_index;
        let complement = 1.0 - outcomes[toggle_index].probability;
        let baseline = outcomes[baseline_index].probability;
        // Delegation substitutes fl(1 - p) for the stored baseline, changing
        // that factor's log mass by approximately |q - fl(1 - p)| / q. The
        // gate bounds this PER-FACTOR error at 1e-9; the worst-case model
        // total scales with the number of delegated factors. In practice
        // representation-noise pairs sit at ~1e-16 relative, so realistic
        // accumulation is far below any test bar, while a model built from
        // many maximally-drifted-yet-passing pairs is deliberate
        // parameterization at scale, not noise. Pairs delegate in either
        // listing order; deliberate drift stays faithful on the N-ary
        // kernel. Below roughly 2e-7, even an exact-sum pair may not
        // delegate because ulp(1)-scale subtraction noise alone can exceed
        // the per-factor bound; N-ary evaluation is then the faithful
        // result, not a missed optimization.
        if (baseline - complement).abs() <= BINARY_COMPLEMENT_RELATIVE_TOLERANCE * baseline {
            let toggle = outcomes[toggle_index].clone();
            return NormalizedFactor::Binary { outcomes, toggle };
        }
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

#[cfg(test)]
mod tests {
    use super::{Factor, FactorModel, Outcome};
    use crate::deadline_column_order_for_factors;

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn factor_model_widths_are_bounded_by_the_u32_addressable_width() {
        // Index u32::MAX exists, so width u32::MAX + 1 is addressable -- the
        // DEM parser accepts `D4294967295` and reports exactly this width, and
        // TryFrom<&SparseDem> must not reject a DEM the parser accepts.
        let max_width = u32::MAX as usize + 1;
        FactorModel::new(Vec::new(), max_width, max_width).unwrap();
        let outcome = Outcome {
            probability: 1.0,
            detectors: vec![u32::MAX],
            observables: Vec::new(),
        };
        FactorModel::new(
            vec![Factor {
                outcomes: vec![outcome],
            }],
            max_width,
            0,
        )
        .unwrap();

        let too_wide = max_width + 1;
        let detector_error = FactorModel::new(Vec::new(), too_wide, 0).unwrap_err();
        assert_eq!(
            detector_error.to_string(),
            format!(
                "Invalid configuration: num_detectors {too_wide} exceeds the u32 index-addressable width {max_width}"
            )
        );

        let observable_error = FactorModel::new(Vec::new(), 0, too_wide).unwrap_err();
        assert_eq!(
            observable_error.to_string(),
            format!(
                "Invalid configuration: num_observables {too_wide} exceeds the u32 index-addressable width {max_width}"
            )
        );
    }

    #[test]
    fn factor_ordering_errors_name_the_factor() {
        // Public constructors reject this model. Construct it here only to
        // exercise the ordering helper's factor-specific diagnostic.
        let model = FactorModel {
            factors: vec![Factor {
                outcomes: vec![Outcome {
                    probability: 1.0,
                    detectors: vec![1],
                    observables: Vec::new(),
                }],
            }],
            num_detectors: 1,
            num_observables: 0,
        };
        let error = deadline_column_order_for_factors(&model).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("factor 0 detector index 1 is out of range 0..1")
        );
    }
}
