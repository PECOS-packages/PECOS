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
use crate::engines::{ControlEngine, EngineStage};
use pecos_core::RngManageable;
use pecos_core::errors::PecosError;
use rand_chacha::ChaCha8Rng;
use std::any::Any;

/// A noise model that passes through messages unchanged
///
/// This is useful as a default for systems that don't need noise.
#[derive(Clone, Debug)]
pub struct PassThroughNoiseModel;

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
    type Rng = ChaCha8Rng;

    fn set_rng(&mut self, _rng: Self::Rng) -> Result<(), PecosError> {
        // PassThroughNoise doesn't use randomness, so just ignore the RNG
        Ok(())
    }

    fn rng(&self) -> &Self::Rng {
        // This is a placeholder implementation since we don't actually have an RNG
        panic!("PassThroughNoise doesn't have an RNG")
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        // This is a placeholder implementation since we don't actually have an RNG
        panic!("PassThroughNoise doesn't have an RNG")
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
