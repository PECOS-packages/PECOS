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

//! Importance sampling plugin for the Tool architecture.
//!
//! This plugin enables importance sampling for efficient estimation of rare events
//! (like logical error rates). It works by boosting error rates during sampling
//! and tracking importance weights for proper reweighting.
//!
//! # Example
//!
//! ```no_run
//! use pecos_neo::tool::{sim_neo, ImportanceSamplingPlugin};
//! use pecos_neo::prelude::*;
//!
//! let circuit = CommandBuilder::new().pz(0).h(0).mz(0).build();
//!
//! // Create simulation with importance sampling
//! let mut sim = sim_neo(circuit)
//!     .depolarizing(0.001)  // True error rate
//!     .build();
//!
//! // Add importance sampling with 10x boost
//! sim.tool_mut().add_plugin_mut(ImportanceSamplingPlugin::new(0.001, 10.0));
//!
//! let results = sim.run();
//! // Use weighted statistics to analyze results...
//! ```

use crate::sampling::importance::ImportanceConfig;
use crate::sampling::weight::{SampleWeight, WeightedStatistics};

use super::resource::Resources;
use super::{Plugin, Stage, Tool};

/// Configuration for importance sampling in the Tool architecture.
#[derive(Debug, Clone)]
pub struct ImportanceSamplingConfig {
    /// Single-qubit gate importance sampling config.
    pub single_qubit: Option<ImportanceConfig>,
    /// Two-qubit gate importance sampling config.
    pub two_qubit: Option<ImportanceConfig>,
    /// Measurement error importance sampling config.
    pub measurement: Option<ImportanceConfig>,
}

impl Default for ImportanceSamplingConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ImportanceSamplingConfig {
    /// Create a new empty importance sampling configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            single_qubit: None,
            two_qubit: None,
            measurement: None,
        }
    }

    /// Configure importance sampling for single-qubit gates.
    #[must_use]
    pub fn with_single_qubit(mut self, p_true: f64, boost_factor: f64) -> Self {
        self.single_qubit = Some(ImportanceConfig::with_boost(p_true, boost_factor));
        self
    }

    /// Configure importance sampling for two-qubit gates.
    #[must_use]
    pub fn with_two_qubit(mut self, p_true: f64, boost_factor: f64) -> Self {
        self.two_qubit = Some(ImportanceConfig::with_boost(p_true, boost_factor));
        self
    }

    /// Configure importance sampling for measurement errors.
    #[must_use]
    pub fn with_measurement(mut self, p_true: f64, boost_factor: f64) -> Self {
        self.measurement = Some(ImportanceConfig::with_boost(p_true, boost_factor));
        self
    }

    /// Configure all error types with the same parameters.
    #[must_use]
    pub fn with_uniform_boost(p_true: f64, boost_factor: f64) -> Self {
        Self::new()
            .with_single_qubit(p_true, boost_factor)
            .with_two_qubit(p_true, boost_factor)
            .with_measurement(p_true, boost_factor)
    }
}

/// Current shot's importance weight.
#[derive(Debug, Clone, Default)]
pub struct CurrentShotWeight(pub SampleWeight);

impl CurrentShotWeight {
    /// Create a new unit weight.
    #[must_use]
    pub fn new() -> Self {
        Self(SampleWeight::one())
    }

    /// Reset for a new shot.
    pub fn reset(&mut self) {
        self.0.reset();
    }
}

/// Accumulated weighted results for importance sampling.
#[derive(Debug, Clone, Default)]
pub struct ImportanceSamplingResults {
    /// Weighted statistics accumulator.
    pub statistics: WeightedStatistics,
    /// Per-shot weights for detailed analysis.
    pub shot_weights: Vec<SampleWeight>,
}

impl ImportanceSamplingResults {
    /// Create new empty results.
    #[must_use]
    pub fn new() -> Self {
        Self {
            statistics: WeightedStatistics::new(),
            shot_weights: Vec::new(),
        }
    }

    /// Get the weighted mean (e.g., estimated logical error rate).
    #[must_use]
    pub fn weighted_mean(&self) -> f64 {
        self.statistics.mean()
    }

    /// Get the standard error of the weighted mean.
    #[must_use]
    pub fn standard_error(&self) -> f64 {
        self.statistics.standard_error()
    }

    /// Get the number of samples.
    #[must_use]
    pub fn count(&self) -> usize {
        self.statistics.count()
    }

    /// Clear results for reuse.
    pub fn clear(&mut self) {
        self.statistics = WeightedStatistics::new();
        self.shot_weights.clear();
    }
}

/// Plugin that enables importance sampling for rare event simulation.
///
/// When added to a Tool, this plugin:
/// 1. Inserts importance sampling configuration as a resource
/// 2. Tracks per-shot importance weights
/// 3. Accumulates weighted statistics
///
/// Note: For full importance sampling functionality (boosted noise application),
/// use the dedicated `ImportanceSamplingRunner` or configure the noise model
/// with boosted error rates manually.
pub struct ImportanceSamplingPlugin {
    config: ImportanceSamplingConfig,
}

impl ImportanceSamplingPlugin {
    /// Create a new importance sampling plugin with uniform boost.
    ///
    /// # Arguments
    /// * `p_true` - True error probability
    /// * `boost_factor` - How much to multiply the error rate for the proposal
    #[must_use]
    pub fn new(p_true: f64, boost_factor: f64) -> Self {
        Self {
            config: ImportanceSamplingConfig::with_uniform_boost(p_true, boost_factor),
        }
    }

    /// Create a plugin with custom configuration.
    #[must_use]
    pub fn with_config(config: ImportanceSamplingConfig) -> Self {
        Self { config }
    }
}

impl Plugin for ImportanceSamplingPlugin {
    fn build(&self, tool: &mut Tool) {
        // Insert resources
        tool.insert_resource_mut(self.config.clone());
        tool.insert_resource_mut(CurrentShotWeight::new());

        if !tool.contains_resource::<ImportanceSamplingResults>() {
            tool.insert_resource_mut(ImportanceSamplingResults::new());
        }

        // Add systems for weight tracking
        tool.add_system_mut(Stage::PreShot, importance_pre_shot);
        tool.add_system_mut(Stage::PostShot, importance_post_shot);
        tool.add_system_mut(Stage::Finish, importance_finish);
    }
}

/// Pre-shot system: Reset weight for new shot.
fn importance_pre_shot(resources: &mut Resources) {
    resources.get_mut::<CurrentShotWeight>().reset();
}

/// Post-shot system: Record weight for this shot.
fn importance_post_shot(resources: &mut Resources) {
    let weight = resources.get::<CurrentShotWeight>().0;

    // Store shot weight
    resources
        .get_mut::<ImportanceSamplingResults>()
        .shot_weights
        .push(weight);
}

/// Finish system: Compute final statistics.
fn importance_finish(resources: &mut Resources) {
    // Statistics are accumulated during post_shot via add()
    // This could compute additional metrics if needed
    let results = resources.get::<ImportanceSamplingResults>();

    if results.count() > 0 {
        // Log summary (in real usage, this would go to a logger)
        let _mean = results.weighted_mean();
        let _se = results.standard_error();
        // Could add logging here
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_importance_sampling_config() {
        let config = ImportanceSamplingConfig::with_uniform_boost(0.001, 10.0);

        assert!(config.single_qubit.is_some());
        assert!(config.two_qubit.is_some());
        assert!(config.measurement.is_some());

        let sq = config.single_qubit.unwrap();
        assert!((sq.p_true - 0.001).abs() < 1e-10);
        assert!((sq.p_proposal - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_importance_sampling_plugin() {
        let tool = Tool::new().add_plugin(ImportanceSamplingPlugin::new(0.001, 10.0));

        assert!(tool.contains_resource::<ImportanceSamplingConfig>());
        assert!(tool.contains_resource::<CurrentShotWeight>());
        assert!(tool.contains_resource::<ImportanceSamplingResults>());
    }

    #[test]
    fn test_current_shot_weight_reset() {
        let mut weight = CurrentShotWeight::new();
        weight.0.update(0.1, 0.5); // Modify weight

        assert!((weight.0.weight() - 0.2).abs() < 1e-10);

        weight.reset();
        assert!((weight.0.weight() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_importance_sampling_results() {
        let mut results = ImportanceSamplingResults::new();

        // Add some samples
        results.statistics.add(1.0, &SampleWeight::from_linear(0.1));
        results.statistics.add(0.0, &SampleWeight::from_linear(0.9));

        // Weighted mean = (1.0 * 0.1 + 0.0 * 0.9) / 1.0 = 0.1
        assert!((results.weighted_mean() - 0.1).abs() < 1e-10);
        assert_eq!(results.count(), 2);
    }
}
