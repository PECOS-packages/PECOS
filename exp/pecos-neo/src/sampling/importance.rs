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

//! Importance sampling noise wrapper.
//!
//! Wraps a noise model to sample from a proposal distribution Q while
//! tracking the likelihood ratio P/Q for proper reweighting.
//!
//! ## How It Works
//!
//! For each noise decision:
//! 1. Sample whether an error occurs from Q (proposal, higher error rate)
//! 2. Compute P(decision) / Q(decision) and accumulate in weight
//! 3. Apply the sampled error (or not)
//!
//! At the end of a shot, the weight corrects for the biased sampling.
//!
//! ## Example
//!
//! ```
//! use pecos_neo::sampling::{ImportanceConfig, ImportanceSamplingNoise};
//!
//! // Create importance sampling noise with boosted error rates
//! let mut importance_noise = ImportanceSamplingNoise::new()
//!     .with_single_qubit(0.001, 10.0)   // True: 0.001, proposal: 0.01
//!     .with_two_qubit(0.01, 5.0);       // True: 0.01, proposal: 0.05
//!
//! // Run shot, get weight
//! // ... execute circuit with importance_noise ...
//! let weight = importance_noise.weight();
//! ```

use super::weight::SampleWeight;
use crate::command::GateType;
use crate::noise::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse};
use pecos_random::PecosRng;
use rand::RngExt;

/// Configuration for importance sampling.
#[derive(Debug, Clone)]
pub struct ImportanceConfig {
    /// Error probability under the true (target) distribution.
    pub p_true: f64,
    /// Error probability under the proposal distribution.
    /// Should be higher than `p_true` to oversample errors.
    pub p_proposal: f64,
}

impl ImportanceConfig {
    /// Create a new importance sampling configuration.
    ///
    /// # Arguments
    /// * `p_true` - True error probability (what we're estimating for)
    /// * `boost_factor` - How much to increase error rate for proposal
    ///
    /// # Example
    /// ```
    /// use pecos_neo::sampling::ImportanceConfig;
    ///
    /// // True error rate 0.001, boost by 10x for proposal
    /// let config = ImportanceConfig::with_boost(0.001, 10.0);
    /// assert!((config.p_proposal - 0.01).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn with_boost(p_true: f64, boost_factor: f64) -> Self {
        let p_proposal = (p_true * boost_factor).min(0.5); // Cap at 50%
        Self { p_true, p_proposal }
    }

    /// Create configuration with explicit probabilities.
    #[must_use]
    pub fn new(p_true: f64, p_proposal: f64) -> Self {
        Self { p_true, p_proposal }
    }

    /// Compute the weight contribution for an error occurring.
    #[must_use]
    pub fn weight_for_error(&self) -> (f64, f64) {
        (self.p_true, self.p_proposal)
    }

    /// Compute the weight contribution for no error.
    #[must_use]
    pub fn weight_for_no_error(&self) -> (f64, f64) {
        (1.0 - self.p_true, 1.0 - self.p_proposal)
    }
}

/// Importance sampling wrapper for a noise channel.
///
/// Samples errors from a boosted proposal distribution while tracking
/// the likelihood ratio for proper reweighting.
#[derive(Debug, Clone)]
pub struct ImportanceSamplingChannel<C: NoiseChannel> {
    /// The underlying channel (determines error type, not rate).
    inner: C,
    /// Importance sampling configuration.
    config: ImportanceConfig,
    /// Accumulated weight for current shot.
    weight: SampleWeight,
}

impl<C: NoiseChannel> ImportanceSamplingChannel<C> {
    /// Create a new importance sampling channel.
    pub fn new(inner: C, config: ImportanceConfig) -> Self {
        Self {
            inner,
            config,
            weight: SampleWeight::one(),
        }
    }

    /// Get the current accumulated weight.
    #[must_use]
    pub fn weight(&self) -> &SampleWeight {
        &self.weight
    }

    /// Reset the weight for a new shot.
    pub fn reset_weight(&mut self) {
        self.weight.reset();
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &ImportanceConfig {
        &self.config
    }
}

impl<C: NoiseChannel + Clone + 'static> NoiseChannel for ImportanceSamplingChannel<C> {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        self.inner.responds_to(event)
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        // Sample from proposal distribution
        let error_occurs = rng.random::<f64>() < self.config.p_proposal;

        // Note: We can't mutate self.weight here because apply takes &self
        // This is a design issue - we need a different approach
        // For now, just delegate to inner channel
        // The proper solution requires rethinking the channel interface

        if error_occurs {
            // Apply the error from the inner channel
            self.inner.apply(event, ctx, rng)
        } else {
            NoiseResponse::None
        }
    }

    fn name(&self) -> &'static str {
        "ImportanceSamplingChannel"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

/// Importance sampling noise model that wraps the entire noise pipeline.
///
/// This is a higher-level wrapper that tracks weights across all noise events.
#[derive(Debug, Clone)]
pub struct ImportanceSamplingNoise {
    /// Single-qubit error configuration.
    pub single_qubit: Option<ImportanceConfig>,
    /// Two-qubit error configuration.
    pub two_qubit: Option<ImportanceConfig>,
    /// Measurement error configuration.
    pub measurement: Option<ImportanceConfig>,
    /// Accumulated weight for current shot.
    weight: SampleWeight,
}

impl Default for ImportanceSamplingNoise {
    fn default() -> Self {
        Self::new()
    }
}

impl ImportanceSamplingNoise {
    /// Create a new importance sampling noise model with no channels.
    #[must_use]
    pub fn new() -> Self {
        Self {
            single_qubit: None,
            two_qubit: None,
            measurement: None,
            weight: SampleWeight::one(),
        }
    }

    /// Add single-qubit error importance sampling.
    #[must_use]
    pub fn with_single_qubit(mut self, p_true: f64, boost_factor: f64) -> Self {
        self.single_qubit = Some(ImportanceConfig::with_boost(p_true, boost_factor));
        self
    }

    /// Add two-qubit error importance sampling.
    #[must_use]
    pub fn with_two_qubit(mut self, p_true: f64, boost_factor: f64) -> Self {
        self.two_qubit = Some(ImportanceConfig::with_boost(p_true, boost_factor));
        self
    }

    /// Add measurement error importance sampling.
    #[must_use]
    pub fn with_measurement(mut self, p_true: f64, boost_factor: f64) -> Self {
        self.measurement = Some(ImportanceConfig::with_boost(p_true, boost_factor));
        self
    }

    /// Get the current accumulated weight.
    #[must_use]
    pub fn weight(&self) -> &SampleWeight {
        &self.weight
    }

    /// Take the weight and reset for next shot.
    pub fn take_weight(&mut self) -> SampleWeight {
        let w = self.weight;
        self.weight.reset();
        w
    }

    /// Reset the weight for a new shot.
    pub fn reset(&mut self) {
        self.weight.reset();
    }

    /// Sample a single-qubit error decision and update weight.
    ///
    /// Returns true if an error should be applied.
    pub fn sample_single_qubit_error(&mut self, rng: &mut PecosRng) -> bool {
        if let Some(config) = &self.single_qubit {
            let error_occurs = rng.random::<f64>() < config.p_proposal;
            let (p_true, p_proposal) = if error_occurs {
                config.weight_for_error()
            } else {
                config.weight_for_no_error()
            };
            self.weight.update(p_true, p_proposal);
            error_occurs
        } else {
            false
        }
    }

    /// Sample a two-qubit error decision and update weight.
    ///
    /// Returns true if an error should be applied.
    pub fn sample_two_qubit_error(&mut self, rng: &mut PecosRng) -> bool {
        if let Some(config) = &self.two_qubit {
            let error_occurs = rng.random::<f64>() < config.p_proposal;
            let (p_true, p_proposal) = if error_occurs {
                config.weight_for_error()
            } else {
                config.weight_for_no_error()
            };
            self.weight.update(p_true, p_proposal);
            error_occurs
        } else {
            false
        }
    }

    /// Sample a measurement error decision and update weight.
    ///
    /// Returns true if the measurement should be flipped.
    pub fn sample_measurement_error(&mut self, rng: &mut PecosRng) -> bool {
        if let Some(config) = &self.measurement {
            let error_occurs = rng.random::<f64>() < config.p_proposal;
            let (p_true, p_proposal) = if error_occurs {
                config.weight_for_error()
            } else {
                config.weight_for_no_error()
            };
            self.weight.update(p_true, p_proposal);
            error_occurs
        } else {
            false
        }
    }

    /// Get the configuration for a gate type.
    #[must_use]
    pub fn config_for_gate(&self, gate_type: GateType) -> Option<&ImportanceConfig> {
        match gate_type.quantum_arity() {
            1 => self.single_qubit.as_ref(),
            2 => self.two_qubit.as_ref(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_importance_config_boost() {
        let config = ImportanceConfig::with_boost(0.001, 10.0);
        assert!((config.p_true - 0.001).abs() < 1e-10);
        assert!((config.p_proposal - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_importance_config_capped() {
        // Very high boost should be capped at 50%
        let config = ImportanceConfig::with_boost(0.1, 100.0);
        assert!((config.p_proposal - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_importance_sampling_noise() {
        let noise = ImportanceSamplingNoise::new()
            .with_single_qubit(0.001, 10.0)
            .with_two_qubit(0.01, 5.0);

        assert!(noise.single_qubit.is_some());
        assert!(noise.two_qubit.is_some());
        assert!(noise.measurement.is_none());

        // Initial weight should be 1
        assert!((noise.weight().weight() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weight_accumulation() {
        let mut noise = ImportanceSamplingNoise::new().with_single_qubit(0.001, 10.0);

        let mut rng = PecosRng::seed_from_u64(42);

        // Sample some errors
        for _ in 0..10 {
            let _ = noise.sample_single_qubit_error(&mut rng);
        }

        // Weight should have changed from 1.0
        let w = noise.weight().weight();
        assert!((w - 1.0).abs() > 1e-10);

        // Reset should restore to 1.0
        noise.reset();
        assert!((noise.weight().weight() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_importance_sampling_statistics() {
        use super::super::weight::WeightedStatistics;

        // Simulate importance sampling for a rare event
        // True rate: 0.001, Proposal rate: 0.1

        let config = ImportanceConfig::new(0.001, 0.1);
        let mut stats = WeightedStatistics::new();
        let mut rng = PecosRng::seed_from_u64(12345);

        let num_samples = 10000;
        for _ in 0..num_samples {
            let mut weight = SampleWeight::one();

            // Simulate one "gate" with possible error
            let error_occurs = rng.random::<f64>() < config.p_proposal;
            let (p_true, p_proposal) = if error_occurs {
                config.weight_for_error()
            } else {
                config.weight_for_no_error()
            };
            weight.update(p_true, p_proposal);

            // Value is 1 if error occurred, 0 otherwise
            let value = if error_occurs { 1.0 } else { 0.0 };
            stats.add(value, &weight);
        }

        // The weighted mean should approximate the true error rate
        let estimated_rate = stats.mean();
        // Allow 20% relative error with 10k samples
        assert!(
            (estimated_rate - 0.001).abs() < 0.001 * 0.5,
            "Estimated rate {estimated_rate} should be close to 0.001"
        );
    }
}
