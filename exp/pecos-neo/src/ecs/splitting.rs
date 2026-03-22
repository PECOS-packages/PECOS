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

//! Splitting criteria for rare event simulation.
//!
//! This module provides traits and implementations for deciding when to split
//! trajectories in rare event simulation algorithms like multilevel splitting
//! and subset simulation.
//!
//! ## Overview
//!
//! In rare event simulation, we want to estimate the probability of rare events
//! (like logical errors in quantum error correction) more efficiently than
//! standard Monte Carlo. Splitting methods work by:
//!
//! 1. Running trajectories until they reach a "promising" state
//! 2. Splitting (cloning) promising trajectories
//! 3. Pruning trajectories that don't progress
//! 4. Weighting results to correct for the biased sampling
//!
//! ## Criteria
//!
//! - [`SyndromeWeightCriterion`]: Split based on syndrome weight (for QEC)
//! - [`ThresholdCriterion`]: Split when a score crosses a threshold
//! - [`LevelCriterion`]: Subset simulation with discrete levels

use super::entity::EntityId;
use super::world::World;
use crate::sampling::SampleWeight;
use pecos_simulators::CliffordGateable;

/// Trait for splitting criteria that decide when to clone trajectories.
///
/// Implementations should be deterministic given the same entity state.
pub trait SplittingCriterion<S: CliffordGateable>: Send + Sync {
    /// Evaluate whether this entity should be split and how many copies.
    ///
    /// Returns:
    /// - `None` if the entity should not be split
    /// - `Some(n)` if the entity should be split into `n` copies (including original)
    ///
    /// A return value of `Some(1)` means keep the entity as-is (no actual split).
    /// A return value of `Some(0)` means prune (remove) the entity.
    fn should_split(&self, entity: EntityId, world: &World<S>) -> Option<usize>;

    /// Get a score for this entity (used for sorting/selection).
    ///
    /// Higher scores indicate more "promising" trajectories that are
    /// closer to the rare event.
    fn score(&self, entity: EntityId, world: &World<S>) -> f64;

    /// Name of this criterion for debugging/logging.
    fn name(&self) -> &'static str;
}

/// Configuration for subset simulation levels.
#[derive(Debug, Clone)]
pub struct SubsetLevel {
    /// Score threshold for this level.
    pub threshold: f64,
    /// Target number of entities that should exceed this threshold.
    pub target_count: usize,
}

impl SubsetLevel {
    /// Create a new subset level.
    #[must_use]
    pub fn new(threshold: f64, target_count: usize) -> Self {
        Self {
            threshold,
            target_count,
        }
    }
}

/// Criterion based on a simple score threshold.
///
/// Entities with score >= threshold are kept and potentially split.
/// Entities with score < threshold are pruned.
#[derive(Debug, Clone)]
pub struct ThresholdCriterion {
    /// Score threshold for splitting.
    threshold: f64,
    /// Number of copies to create when splitting.
    split_factor: usize,
}

impl ThresholdCriterion {
    /// Create a new threshold criterion.
    #[must_use]
    pub fn new(threshold: f64, split_factor: usize) -> Self {
        Self {
            threshold,
            split_factor,
        }
    }
}

impl<S: CliffordGateable> SplittingCriterion<S> for ThresholdCriterion {
    fn should_split(&self, entity: EntityId, world: &World<S>) -> Option<usize> {
        let score = self.score(entity, world);
        if score >= self.threshold {
            Some(self.split_factor)
        } else {
            Some(0) // Prune
        }
    }

    fn score(&self, entity: EntityId, world: &World<S>) -> f64 {
        // Default score is the log weight
        world
            .weights
            .get(entity)
            .map_or(0.0, |w| w.weight.log_weight())
    }

    fn name(&self) -> &'static str {
        "ThresholdCriterion"
    }
}

/// Score function type for custom scoring.
pub type ScoreFn<S> = Box<dyn Fn(EntityId, &World<S>) -> f64 + Send + Sync>;

/// Criterion with custom score function.
pub struct CustomScoreCriterion<S: CliffordGateable> {
    /// Score function.
    score_fn: ScoreFn<S>,
    /// Threshold for splitting.
    threshold: f64,
    /// Split factor.
    split_factor: usize,
}

impl<S: CliffordGateable> CustomScoreCriterion<S> {
    /// Create a criterion with a custom score function.
    #[must_use]
    pub fn new(score_fn: ScoreFn<S>, threshold: f64, split_factor: usize) -> Self {
        Self {
            score_fn,
            threshold,
            split_factor,
        }
    }
}

impl<S: CliffordGateable> SplittingCriterion<S> for CustomScoreCriterion<S> {
    fn should_split(&self, entity: EntityId, world: &World<S>) -> Option<usize> {
        let score = self.score(entity, world);
        if score >= self.threshold {
            Some(self.split_factor)
        } else {
            Some(0) // Prune
        }
    }

    fn score(&self, entity: EntityId, world: &World<S>) -> f64 {
        (self.score_fn)(entity, world)
    }

    fn name(&self) -> &'static str {
        "CustomScoreCriterion"
    }
}

/// Result of a splitting decision.
#[derive(Debug, Clone)]
pub struct SplitDecision {
    /// Entity to split.
    pub entity: EntityId,
    /// Number of copies to create (0 = prune, 1 = keep, n > 1 = split).
    pub copies: usize,
    /// Adjusted weight for each copy.
    pub new_weight: SampleWeight,
}

impl SplitDecision {
    /// Create a decision to prune an entity.
    #[must_use]
    pub fn prune(entity: EntityId) -> Self {
        Self {
            entity,
            copies: 0,
            new_weight: SampleWeight::one(),
        }
    }

    /// Create a decision to keep an entity unchanged.
    #[must_use]
    pub fn keep(entity: EntityId) -> Self {
        Self {
            entity,
            copies: 1,
            new_weight: SampleWeight::one(),
        }
    }

    /// Create a decision to split an entity.
    #[must_use]
    pub fn split(entity: EntityId, copies: usize, weight_per_copy: SampleWeight) -> Self {
        Self {
            entity,
            copies,
            new_weight: weight_per_copy,
        }
    }
}

/// Statistics from a splitting operation.
#[derive(Debug, Clone, Default)]
pub struct SplitStats {
    /// Number of entities before splitting.
    pub entities_before: usize,
    /// Number of entities after splitting.
    pub entities_after: usize,
    /// Number of entities pruned.
    pub pruned: usize,
    /// Number of entities split (copies created).
    pub split: usize,
    /// Total weight before splitting.
    pub total_weight_before: f64,
    /// Total weight after splitting.
    pub total_weight_after: f64,
}

impl SplitStats {
    /// Check if splitting preserved total weight (for validation).
    #[must_use]
    pub fn weight_preserved(&self, tolerance: f64) -> bool {
        (self.total_weight_before - self.total_weight_after).abs() < tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subset_level() {
        let level = SubsetLevel::new(0.5, 100);
        assert!((level.threshold - 0.5).abs() < 1e-10);
        assert_eq!(level.target_count, 100);
    }

    #[test]
    fn test_split_decision() {
        let prune = SplitDecision::prune(EntityId(1));
        assert_eq!(prune.copies, 0);

        let keep = SplitDecision::keep(EntityId(2));
        assert_eq!(keep.copies, 1);

        let split = SplitDecision::split(EntityId(3), 4, SampleWeight::from_linear(0.25));
        assert_eq!(split.copies, 4);
    }

    #[test]
    fn test_split_stats() {
        let stats = SplitStats {
            entities_before: 100,
            entities_after: 150,
            pruned: 20,
            split: 70,
            total_weight_before: 100.0,
            total_weight_after: 100.0,
        };

        assert!(stats.weight_preserved(1e-10));
    }
}
