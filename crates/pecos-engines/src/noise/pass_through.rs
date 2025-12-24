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

use super::NoiseModel;
use crate::byte_message::ByteMessage;
use crate::engine_system::{ControlEngine, EngineStage};
use pecos_core::RngManageable;
use pecos_core::errors::PecosError;
use pecos_rng::PecosRng;
use std::any::Any;

/// A noise model that passes through messages unchanged
///
/// This is useful as a default for systems that don't need noise.
#[derive(Clone, Debug)]
pub struct PassThroughNoiseModel {
    /// Dummy RNG field to satisfy the `RngManageable` trait
    /// `PassThroughNoiseModel` doesn't actually use randomness
    rng: PecosRng,
}

impl PassThroughNoiseModel {
    /// Create a new pass-through noise model
    #[must_use]
    pub fn new() -> Self {
        Self {
            rng: PecosRng::seed_from_u64(0), // Default seed, not used
        }
    }

    /// Create a new builder for pass-through noise model
    #[must_use]
    pub fn builder() -> PassThroughNoiseModelBuilder {
        PassThroughNoiseModelBuilder::new()
    }
}

impl Default for PassThroughNoiseModel {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseModel for PassThroughNoiseModel {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// Implement RngManageable for PassThroughNoise
impl RngManageable for PassThroughNoiseModel {
    type Rng = PecosRng;

    fn set_rng(&mut self, rng: Self::Rng) {
        // PassThroughNoise doesn't use randomness, but we store it to satisfy the trait
        self.rng = rng;
    }

    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

impl ControlEngine for PassThroughNoiseModel {
    type Input = ByteMessage;
    type Output = ByteMessage;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(
        &mut self,
        input: Self::Input,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        // Simply pass through the input message unchanged
        Ok(EngineStage::NeedsProcessing(input))
    }

    fn continue_processing(
        &mut self,
        result: Self::EngineOutput,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        // Simply pass through the result message unchanged
        Ok(EngineStage::Complete(result))
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // No state to reset
        Ok(())
    }
}

/// Builder for creating pass-through (no noise) models
///
/// This builder exists for API consistency, allowing all noise models
/// to be created through the same builder pattern.
#[derive(Debug, Clone, Default)]
pub struct PassThroughNoiseModelBuilder;

impl PassThroughNoiseModelBuilder {
    /// Create a new pass-through noise model builder
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Build the pass-through noise model
    ///
    /// Since this is a no-op noise model, the builder has no configuration options.
    #[must_use]
    pub fn build(self) -> PassThroughNoiseModel {
        PassThroughNoiseModel {
            rng: PecosRng::seed_from_u64(0), // Default seed, not used
        }
    }
}

impl crate::noise::IntoNoiseModel for PassThroughNoiseModelBuilder {
    fn into_noise_model(self) -> Box<dyn crate::noise::NoiseModel> {
        Box::new(self.build())
    }
}
