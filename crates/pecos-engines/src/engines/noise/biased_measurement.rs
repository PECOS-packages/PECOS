// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use crate::byte_message::ByteMessage;
use crate::engines::noise::{NoiseModel, NoiseRng, ProbabilityValidator, RngManageable};
use crate::engines::{ControlEngine, EngineStage};
use pecos_core::errors::PecosError;
use rand_chacha::ChaCha8Rng;
use std::any::Any;

/// A noise model that biases qubit measurements
///
/// This noise model introduces bias to measurement results, changing
/// the probability of measuring 0 vs 1. It leaves quantum operations
/// unchanged.
///
/// # Usage
///
/// ```rust
/// use pecos_engines::engines::noise::BiasedMeasurementNoiseModel;
/// use pecos_engines::engines::noise::{NoiseModel, RngManageable};
///
/// // Create directly
/// let mut noise_model = BiasedMeasurementNoiseModel::new(0.01, 0.02);
/// noise_model.set_seed(42).unwrap(); // For reproducibility
///
/// // Or use builder pattern
/// let noise_model = BiasedMeasurementNoiseModel::builder()
///     .with_prob_flip_from_0(0.01)
///     .with_prob_flip_from_1(0.02)
///     .with_seed(42)
///     .build();
/// ```
#[derive(Clone)]
pub struct BiasedMeasurementNoiseModel {
    /// The probability of flipping a 0 measurement to 1
    prob_flip_from_0: f64,
    /// The probability of flipping a 1 measurement to 0
    prob_flip_from_1: f64,
    /// Random number generator
    rng: NoiseRng<ChaCha8Rng>,
}

impl ProbabilityValidator for BiasedMeasurementNoiseModel {}

impl BiasedMeasurementNoiseModel {
    /// Creates a new biased measurement noise model
    ///
    /// # Arguments
    /// * `prob_flip_from_0` - Probability of flipping a 0 measurement to 1
    /// * `prob_flip_from_1` - Probability of flipping a 1 measurement to 0
    ///
    /// # Panics
    /// Panics if either probability is not in the range [0, 1]
    #[must_use]
    pub fn new(prob_flip_from_0: f64, prob_flip_from_1: f64) -> Self {
        // Validate probabilities
        Self::validate_named_probability(prob_flip_from_0, "prob_flip_from_0");
        Self::validate_named_probability(prob_flip_from_1, "prob_flip_from_1");

        Self {
            prob_flip_from_0,
            prob_flip_from_1,
            rng: NoiseRng::default(),
        }
    }

    /// Creates a new biased measurement noise model with a specific seed
    ///
    /// # Arguments
    /// * `prob_flip_from_0` - Probability of flipping a 0 measurement to 1
    /// * `prob_flip_from_1` - Probability of flipping a 1 measurement to 0
    /// * `seed` - Seed for the random number generator
    ///
    /// # Panics
    /// Panics if either probability is not in the range [0, 1]
    #[must_use]
    pub fn with_seed(prob_flip_from_0: f64, prob_flip_from_1: f64, seed: u64) -> Self {
        // Validate probabilities
        Self::validate_named_probability(prob_flip_from_0, "prob_flip_from_0");
        Self::validate_named_probability(prob_flip_from_1, "prob_flip_from_1");

        Self {
            prob_flip_from_0,
            prob_flip_from_1,
            rng: NoiseRng::with_seed(seed),
        }
    }

    /// Create a new builder for the biased measurement noise model
    #[must_use]
    pub fn builder() -> BiasedMeasurementNoiseModelBuilder {
        BiasedMeasurementNoiseModelBuilder::new()
    }

    /// Get the probability of flipping a 0 measurement to a 1
    #[must_use]
    pub fn prob_flip_from_0(&self) -> f64 {
        self.prob_flip_from_0
    }

    /// Get the probability of flipping a 1 measurement to a 0
    #[must_use]
    pub fn prob_flip_from_1(&self) -> f64 {
        self.prob_flip_from_1
    }

    /// Apply bias to a single measurement result
    ///
    /// # Arguments
    /// * `result_id` - The result ID of the measurement
    /// * `outcome` - The outcome of the measurement (0 or 1)
    ///
    /// # Returns
    /// The potentially biased measurement outcome
    fn apply_bias_to_measurement(&mut self, result_id: u32, outcome: u32) -> (u32, u32) {
        // Generate a random number to determine if we should flip
        let should_flip = if outcome == 0 {
            // Flip from 0 to 1 with probability prob_flip_from_0
            self.rng.occurs(self.prob_flip_from_0)
        } else {
            // Flip from 1 to 0 with probability prob_flip_from_1
            self.rng.occurs(self.prob_flip_from_1)
        };

        if should_flip {
            // Flip the measurement outcome
            (result_id, 1 - outcome)
        } else {
            // Keep the original measurement
            (result_id, outcome)
        }
    }

    /// Apply bias to a `ByteMessage` containing measurement results
    ///
    /// # Arguments
    /// * `message` - The `ByteMessage` containing measurement results
    ///
    /// # Returns
    /// A new `ByteMessage` with biased measurement results
    ///
    /// # Errors
    /// Returns a `PecosError` if applying bias fails
    fn apply_bias_to_message(&mut self, message: ByteMessage) -> Result<ByteMessage, PecosError> {
        // Parse the message to extract the measurement results
        let measurements = message.parse_measurements()?;

        // If the message doesn't contain measurements, return it unchanged
        if measurements.is_empty() {
            return Ok(message);
        }

        // Apply bias to each measurement
        let biased_measurements: Vec<(usize, u32)> = measurements
            .iter()
            .map(|(result_id, outcome)| {
                let (biased_result_id, biased_outcome) =
                    self.apply_bias_to_measurement(*result_id, *outcome);
                (biased_result_id as usize, biased_outcome)
            })
            .collect();

        // Create a new ByteMessage with the biased measurements
        Ok(ByteMessage::record_measurement_results(
            &biased_measurements,
        ))
    }
}

/// Builder for creating biased measurement noise models
pub struct BiasedMeasurementNoiseModelBuilder {
    /// The probability of flipping a 0 measurement to 1
    prob_flip_from_0: Option<f64>,
    /// The probability of flipping a 1 measurement to 0
    prob_flip_from_1: Option<f64>,
    /// Optional seed for the RNG
    seed: Option<u64>,
}

impl Default for BiasedMeasurementNoiseModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BiasedMeasurementNoiseModelBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            prob_flip_from_0: None,
            prob_flip_from_1: None,
            seed: None,
        }
    }

    /// Set the probability of flipping a 0 measurement to 1
    #[must_use]
    pub fn with_prob_flip_from_0(mut self, probability: f64) -> Self {
        self.prob_flip_from_0 = Some(probability);
        self
    }

    /// Set the probability of flipping a 1 measurement to 0
    #[must_use]
    pub fn with_prob_flip_from_1(mut self, probability: f64) -> Self {
        self.prob_flip_from_1 = Some(probability);
        self
    }

    /// Set the seed for the random number generator
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the biased measurement noise model
    ///
    /// # Returns
    /// A boxed noise model
    ///
    /// # Panics
    /// Panics if probabilities are not set or are not between 0 and 1.
    #[must_use]
    pub fn build(self) -> Box<dyn NoiseModel> {
        let prob_flip_from_0 = self
            .prob_flip_from_0
            .expect("Probability of flipping from 0 to 1 must be set");
        let prob_flip_from_1 = self
            .prob_flip_from_1
            .expect("Probability of flipping from 1 to 0 must be set");

        // Create the noise model
        let mut noise = BiasedMeasurementNoiseModel::new(prob_flip_from_0, prob_flip_from_1);

        // Set the seed if provided
        if let Some(seed) = self.seed {
            // Use RngManageable::set_seed directly
            noise.set_seed(seed).expect("Failed to set seed");
        }

        Box::new(noise)
    }
}

impl ControlEngine for BiasedMeasurementNoiseModel {
    type Input = ByteMessage;
    type Output = ByteMessage;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(
        &mut self,
        input: Self::Input,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        // Quantum operations pass through unchanged
        Ok(EngineStage::NeedsProcessing(input))
    }

    fn continue_processing(
        &mut self,
        result: Self::EngineOutput,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        // Apply bias to measurement results
        let biased_result = self.apply_bias_to_message(result)?;
        Ok(EngineStage::Complete(biased_result))
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Nothing to reset
        Ok(())
    }
}

impl NoiseModel for BiasedMeasurementNoiseModel {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl RngManageable for BiasedMeasurementNoiseModel {
    type Rng = ChaCha8Rng;

    fn set_rng(&mut self, rng: ChaCha8Rng) -> Result<(), PecosError> {
        self.rng = NoiseRng::new(rng);
        Ok(())
    }

    fn rng(&self) -> &Self::Rng {
        self.rng.inner()
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.rng.inner_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_pattern() {
        // Create with builder
        let mut noise1 = BiasedMeasurementNoiseModel::builder()
            .with_prob_flip_from_0(0.1)
            .with_prob_flip_from_1(0.2)
            .with_seed(42)
            .build();

        // Create directly
        let mut noise2 = BiasedMeasurementNoiseModel::with_seed(0.1, 0.2, 42);

        // Verify the builder works by checking they produce the same randomness sequence
        let noise1_ref = noise1
            .as_any_mut()
            .downcast_mut::<BiasedMeasurementNoiseModel>()
            .unwrap();

        for _ in 0..10 {
            let flip_test = noise1_ref.apply_bias_to_measurement(0, 0);
            let flip_test2 = noise2.apply_bias_to_measurement(0, 0);
            assert_eq!(flip_test, flip_test2);
        }
    }

    #[test]
    #[should_panic(
        expected = "Probability prob_flip_from_0 must be between 0.0 and 1.0, but was 1.1"
    )]
    fn test_invalid_probability() {
        let _ = BiasedMeasurementNoiseModel::new(1.1, 0.5);
    }

    #[test]
    fn test_apply_bias() {
        let mut noise = BiasedMeasurementNoiseModel::new(1.0, 0.0);

        // With prob_flip_from_0 = 1.0, all 0s should be flipped to 1s
        assert_eq!(noise.apply_bias_to_measurement(0, 0), (0, 1));

        // With prob_flip_from_1 = 0.0, all 1s should remain 1s
        assert_eq!(noise.apply_bias_to_measurement(0, 1), (0, 1));

        // Test with different probabilities
        noise = BiasedMeasurementNoiseModel::new(0.0, 1.0);

        // With prob_flip_from_0 = 0.0, all 0s should remain 0s
        assert_eq!(noise.apply_bias_to_measurement(0, 0), (0, 0));

        // With prob_flip_from_1 = 1.0, all 1s should be flipped to 0s
        assert_eq!(noise.apply_bias_to_measurement(0, 1), (0, 0));
    }
}
