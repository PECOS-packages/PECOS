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

//! Quantum noise models for realistic quantum computation simulation
//!
//! This module provides various noise models that can be used to simulate
//! realistic quantum computation with errors. Each noise model implements
//! the `NoiseModel` trait and can be used with the quantum engines.

pub mod biased_depolarizing;
pub mod depolarizing;
pub mod general;
pub mod noise_rng;
pub mod pass_through;
pub mod utils;
pub mod weighted_sampler;

pub use self::biased_depolarizing::{
    BiasedDepolarizingNoiseModel, BiasedDepolarizingNoiseModelBuilder,
};
pub use self::depolarizing::{DepolarizingNoiseModel, DepolarizingNoiseModelBuilder};
pub use self::general::{GeneralNoiseModel, GeneralNoiseModelBuilder};
pub use self::noise_rng::NoiseRng;
pub use self::pass_through::{PassThroughNoiseModel, PassThroughNoiseModelBuilder};
pub use self::utils::{NoiseUtils, ProbabilityValidator};
pub use self::weighted_sampler::{
    CrosstalkWeightedSampler, SingleQubitWeightedSampler, TwoQubitWeightedSampler, WeightedSampler,
};

use crate::byte_message::ByteMessage;
use crate::engine_system::{ControlEngine, EngineStage};
use dyn_clone::DynClone;
use pecos_core::errors::PecosError;
use pecos_random::PecosRng;
use std::any::Any;

// Re-export RngManageable to ensure consistent trait resolution
// This helps solve Windows-specific dependency issues
pub use pecos_core::RngManageable;

/// Trait defining interface for quantum noise models
///
/// Noise models are a special kind of controller that transform
/// quantum operations before they're executed and potentially transform measurement
/// results after they're produced.
pub trait NoiseModel:
    ControlEngine<
        Input = ByteMessage,
        Output = ByteMessage,
        EngineInput = ByteMessage,
        EngineOutput = ByteMessage,
    > + DynClone
    + Send
    + Sync
    + Any
    + RngManageable<Rng = PecosRng>
{
    /// Returns a reference to self as Any
    ///
    /// This allows for type-checking and downcasting without requiring
    /// experimental trait upcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns a mutable reference to self as Any
    ///
    /// This allows for type-checking and downcasting without requiring
    /// experimental trait upcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Trait for types that can be converted into a noise model
///
/// This trait enables lazy evaluation by storing a closure that builds
/// the noise model when needed.
pub trait IntoNoiseModel: Send + Sync {
    /// Convert into a boxed noise model
    fn into_noise_model(self) -> Box<dyn NoiseModel>;
}

// Register traits with dyn_clone
dyn_clone::clone_trait_object!(NoiseModel);

/// Base implementation for noise models
///
/// This struct provides common functionality for all noise models,
/// reducing code duplication and improving maintainability.
pub struct BaseNoiseModel {
    /// The random number generator for the noise model
    rng: NoiseRng<PecosRng>,
}

impl BaseNoiseModel {
    /// Create a new `BaseNoiseModel` with a random seed
    #[must_use]
    pub fn new() -> Self {
        Self {
            rng: NoiseRng::default(),
        }
    }

    /// Create a new `BaseNoiseModel` with a specific seed
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: NoiseRng::with_seed(seed),
        }
    }

    /// Get a reference to the random number generator
    #[must_use]
    pub fn rng(&self) -> &NoiseRng<PecosRng> {
        &self.rng
    }

    /// Get a mutable reference to the random number generator
    #[must_use]
    pub fn rng_mut(&mut self) -> &mut NoiseRng<PecosRng> {
        &mut self.rng
    }

    /// Check if a message contains measurement results
    ///
    /// # Arguments
    /// * `message` - The `ByteMessage` to check
    ///
    /// # Returns
    /// true if the message contains measurement results, false otherwise
    #[must_use]
    pub fn has_measurements(&self, message: &ByteMessage) -> bool {
        NoiseUtils::has_measurements(message)
    }
}

impl Default for BaseNoiseModel {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for BaseNoiseModel {
    fn clone(&self) -> Self {
        Self {
            rng: self.rng.clone(),
        }
    }
}

impl RngManageable for BaseNoiseModel {
    type Rng = PecosRng;

    fn set_rng(&mut self, rng: PecosRng) {
        self.rng = NoiseRng::new(rng);
    }

    fn rng(&self) -> &Self::Rng {
        self.rng.inner()
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.rng.inner_mut()
    }
}

impl ControlEngine for Box<dyn NoiseModel> {
    type Input = ByteMessage;
    type Output = ByteMessage;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(
        &mut self,
        input: Self::Input,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        (**self).start(input)
    }

    fn continue_processing(
        &mut self,
        result: Self::EngineOutput,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        (**self).continue_processing(result)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        (**self).reset()
    }
}

// Add tests for the BaseNoiseModel
#[cfg(test)]
mod base_tests {
    use super::*;

    #[test]
    fn test_base_noise_model_construction() {
        let model = BaseNoiseModel::new();
        // Verify RNG is initialized, not checking for null since from_ref is never null
        assert!(
            model.rng().inner() != &PecosRng::seed_from_u64(0),
            "Default RNG should be randomly seeded"
        );

        let model = BaseNoiseModel::with_seed(42);
        // Check the model has a properly seeded RNG
        assert_eq!(
            *model.rng().inner(),
            PecosRng::seed_from_u64(42),
            "RNG should be initialized with seed 42"
        );
    }

    #[test]
    fn test_base_noise_model_has_measurements() {
        let model = BaseNoiseModel::new();

        // Test with a message that has no measurements
        let empty_msg = ByteMessage::new(&[]);
        assert!(!model.has_measurements(&empty_msg));

        // Test with a message that has measurements
        let mut builder = ByteMessage::outcomes_builder();
        builder.add_outcomes(&[0]);
        let measure_msg = builder.build();
        assert!(model.has_measurements(&measure_msg));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::byte_message::ByteMessageBuilder;

    #[test]
    fn test_noise_model_biased_depolarizing() {
        // Create a biased depolarizing noise model with a fixed seed
        let mut noise_model = BiasedDepolarizingNoiseModel::new_uniform(0.1);
        noise_model.set_seed(42); // Set seed for deterministic behavior

        // Create a quantum operation message
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.add_x(&[0]);
        let quantum_message = builder.build();

        // Create a measurement result message
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();
        builder.add_outcomes(&[0]);
        let measurement_message = builder.build();

        // Operations may be modified
        let operation_result = noise_model.start(quantum_message.clone()).unwrap();
        if let EngineStage::NeedsProcessing(_) = operation_result {
            // Expected - operations should be processed
        } else {
            panic!("Expected NeedsProcessing stage");
        }

        // Measurements may be modified by biased depolarizing noise
        let measurement_result = noise_model
            .continue_processing(measurement_message.clone())
            .unwrap();
        if let EngineStage::Complete(output) = measurement_result {
            // BiasedDepolarizingNoiseModel applies measurement bias
            // With p=0.1, there's a 10% chance of flipping each measurement
            // We can't assert exact equality, but we should get valid measurement results
            let outcomes = output.outcomes().unwrap();
            assert_eq!(outcomes.len(), 1, "Should have one measurement outcome");
            assert!(outcomes[0] <= 1, "Outcome should be 0 or 1");
        } else {
            panic!("Expected Complete stage");
        }
    }

    #[test]
    fn test_noise_model_depolarizing() {
        // Create a depolarizing noise model
        let mut noise_model = DepolarizingNoiseModel::new_uniform(0.1);

        // Create a quantum operation message
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.add_x(&[0]);
        let quantum_message = builder.build();

        // Create a measurement result message
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();
        builder.add_outcomes(&[0]);
        let measurement_message = builder.build();

        // Operations should be modified
        let operation_result = noise_model.start(quantum_message.clone()).unwrap();
        if let EngineStage::NeedsProcessing(output) = operation_result {
            // Can't check for exact output due to randomness
            let gates = output.quantum_ops().unwrap();
            assert!(!gates.is_empty(), "Output should contain at least one gate");
        } else {
            panic!("Expected NeedsProcessing stage");
        }

        // Measurements should pass through unchanged
        let measurement_result = noise_model
            .continue_processing(measurement_message.clone())
            .unwrap();
        if let EngineStage::Complete(output) = measurement_result {
            assert_eq!(
                output.as_bytes(),
                measurement_message.as_bytes(),
                "Measurement results should pass through depolarizing noise unchanged"
            );
        } else {
            panic!("Expected Complete stage");
        }
    }
}
