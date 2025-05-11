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

use super::engine::HybridEngine;
use crate::engines::noise::{DepolarizingNoiseModel, NoiseModel, PassThroughNoiseModel};
use crate::engines::quantum_system::QuantumSystem;
use crate::engines::{ClassicalEngine, QuantumEngine};
use pecos_core::errors::PecosError;

/// Builder for creating a `HybridEngine` with customizable configuration
///
/// This builder provides a fluent interface for constructing a `HybridEngine`
/// with various configuration options. It simplifies the creation process
/// and makes the code more maintainable by centralizing configuration logic.
///
/// # Examples
///
/// ```
/// use pecos_engines::engines::hybrid::HybridEngineBuilder;
/// use pecos_engines::engines::quantum;
/// use pecos_engines::engines::monte_carlo::engine::ExternalClassicalEngine;
///
/// // Create a HybridEngine with default settings (no noise)
/// let engine = HybridEngineBuilder::new()
///     .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
///     .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 42))
///     .build();
///
/// // Create a HybridEngine with depolarizing noise
/// let engine_with_noise = HybridEngineBuilder::new()
///     .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
///     .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 42))
///     .with_depolarizing_noise(0.01)
///     .build();
///
/// // Create a HybridEngine with a specific seed
/// let seeded_engine = HybridEngineBuilder::new()
///     .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
///     .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 0))
///     .with_seed(42)
///     .build();
/// ```
#[derive(Clone)]
pub struct HybridEngineBuilder {
    classical_engine: Option<Box<dyn ClassicalEngine>>,
    quantum_engine: Option<Box<dyn QuantumEngine>>,
    noise_model: Option<Box<dyn NoiseModel>>,
    quantum_system: Option<QuantumSystem>,
    seed: Option<u64>,
}

impl HybridEngineBuilder {
    /// Create a new `HybridEngineBuilder` with default settings
    ///
    /// By default, no components are set. You must at minimum provide:
    /// - A classical engine via `with_classical_engine`
    /// - Either a quantum engine via `with_quantum_engine` or a quantum system via `with_quantum_system`
    ///
    /// # Returns
    /// A new `HybridEngineBuilder` with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            classical_engine: None,
            quantum_engine: None,
            noise_model: None,
            quantum_system: None,
            seed: None,
        }
    }

    /// Set the classical engine component
    ///
    /// # Arguments
    /// * `engine` - The classical engine to use
    ///
    /// # Returns
    /// The builder for method chaining
    #[must_use]
    pub fn with_classical_engine(mut self, engine: Box<dyn ClassicalEngine>) -> Self {
        self.classical_engine = Some(engine);
        self
    }

    /// Set the quantum engine component
    ///
    /// This will be combined with the noise model (if set) to create a quantum system.
    /// If a quantum system is already set, this will replace it.
    ///
    /// # Arguments
    /// * `engine` - The quantum engine to use
    ///
    /// # Returns
    /// The builder for method chaining
    #[must_use]
    pub fn with_quantum_engine(mut self, engine: Box<dyn QuantumEngine>) -> Self {
        self.quantum_engine = Some(engine);
        self.quantum_system = None; // Reset quantum_system as it's now invalid
        self
    }

    /// Set a custom noise model
    ///
    /// This will be combined with the quantum engine (if set) to create a quantum system.
    /// If a quantum system is already set, this will replace it.
    ///
    /// # Arguments
    /// * `model` - The noise model to use
    ///
    /// # Returns
    /// The builder for method chaining
    #[must_use]
    pub fn with_noise_model(mut self, model: Box<dyn NoiseModel>) -> Self {
        self.noise_model = Some(model);
        self.quantum_system = None; // Reset quantum_system as it's now invalid
        self
    }

    /// Set depolarizing noise with the given probability
    ///
    /// This is a convenience method that creates a `DepolarizingNoise` model
    /// with the specified probability.
    ///
    /// # Arguments
    /// * `probability` - The probability parameter for depolarizing noise (between 0.0 and 1.0)
    ///
    /// # Returns
    /// The builder for method chaining
    #[must_use]
    pub fn with_depolarizing_noise(mut self, probability: f64) -> Self {
        self.noise_model = Some(Box::new(DepolarizingNoiseModel::new_uniform(probability)));
        self.quantum_system = None; // Reset quantum_system as it's now invalid
        self
    }

    /// Set a pre-configured quantum system
    ///
    /// This can be used when you need precise control over the quantum system configuration.
    /// If quantum engine or noise model are already set, this will replace them.
    ///
    /// # Arguments
    /// * `system` - The pre-configured quantum system
    ///
    /// # Returns
    /// The builder for method chaining
    #[must_use]
    pub fn with_quantum_system(mut self, system: QuantumSystem) -> Self {
        self.quantum_system = Some(system);
        self.quantum_engine = None; // Reset as they're now managed by quantum_system
        self.noise_model = None;
        self
    }

    /// Set a seed value for deterministic randomness
    ///
    /// The seed will be applied to all components after building.
    ///
    /// # Arguments
    /// * `seed` - The seed value
    ///
    /// # Returns
    /// The builder for method chaining
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the `HybridEngine` with the configured components
    ///
    /// # Returns
    /// A new `HybridEngine` configured according to the builder settings
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - No classical engine has been set
    /// - Neither a quantum system nor a quantum engine has been set
    #[must_use]
    pub fn build(self) -> HybridEngine {
        // Get the classical engine or panic if not set
        let classical_engine = self
            .classical_engine
            .expect("Classical engine is required. Use with_classical_engine() to set one.");

        // Determine the quantum system
        let quantum_system = if let Some(system) = self.quantum_system {
            system
        } else {
            // Get the quantum engine or panic if not set
            let quantum_engine = self.quantum_engine.expect(
                "Either quantum engine or quantum system is required. Use with_quantum_engine() or with_quantum_system() to set one.",
            );

            // Create a noise model (default to PassThroughNoise if not set)
            let noise_model = self
                .noise_model
                .unwrap_or_else(|| Box::new(PassThroughNoiseModel));

            // Create the quantum system
            QuantumSystem::new(noise_model, quantum_engine)
        };

        // Create the HybridEngine
        let mut engine = HybridEngine {
            classical_engine,
            quantum_system,
        };

        // If a seed is set, apply it (and ignore errors since this is a convenience builder)
        if let Some(seed) = self.seed {
            let _ = engine.set_seed(seed);
        }

        engine
    }

    /// Build the `HybridEngine` with the configured components and set the seed
    ///
    /// This is similar to `build()` but returns a Result to handle seed setting errors.
    ///
    /// # Returns
    /// A new `HybridEngine` configured according to the builder settings with the seed set
    ///
    /// # Errors
    /// Returns a `PecosError` if setting the seed fails
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - No classical engine has been set
    /// - Neither a quantum system nor a quantum engine has been set
    /// - No seed has been set
    pub fn build_with_seed(self) -> Result<HybridEngine, PecosError> {
        // Get the seed or panic if not set
        let seed = self.seed.expect(
            "Seed is required for build_with_seed(). Use with_seed() to set one or use build() instead.",
        );

        // Build the engine
        let mut engine = self.build();

        // Set the seed and return the result
        engine.set_seed(seed)?;
        Ok(engine)
    }
}

impl Default for HybridEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::monte_carlo::engine::ExternalClassicalEngine;
    use crate::engines::quantum;

    #[test]
    fn test_basic_builder() {
        // Create a basic engine with no noise
        let mut engine = HybridEngineBuilder::new()
            .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
            .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 42))
            .build();

        // Verify the engine was created successfully
        assert!(engine.run_shot().is_ok());
    }

    #[test]
    fn test_with_depolarizing_noise() {
        // Create an engine with depolarizing noise
        let mut engine = HybridEngineBuilder::new()
            .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
            .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 42))
            .with_depolarizing_noise(0.01)
            .build();

        // Verify the engine was created successfully
        assert!(engine.run_shot().is_ok());
    }

    #[test]
    fn test_with_seed() {
        // Create an engine with a seed
        let result = HybridEngineBuilder::new()
            .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
            .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 0))
            .with_seed(42)
            .build_with_seed();

        // Verify the engine was created and seed was set successfully
        assert!(result.is_ok());
    }

    #[test]
    #[should_panic(expected = "Classical engine is required")]
    fn test_missing_classical_engine() {
        // Try to build without setting a classical engine
        let _ = HybridEngineBuilder::new()
            .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 42))
            .build();
    }

    #[test]
    #[should_panic(expected = "Either quantum engine or quantum system is required")]
    fn test_missing_quantum_components() {
        // Try to build without setting a quantum engine or system
        let _ = HybridEngineBuilder::new()
            .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
            .build();
    }

    #[test]
    fn test_with_quantum_system() {
        // Create a quantum system
        let quantum_system = QuantumSystem::new(
            Box::new(PassThroughNoiseModel),
            quantum::new_quantum_engine_with_seed(2, 42),
        );

        // Create an engine with the quantum system
        let mut engine = HybridEngineBuilder::new()
            .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
            .with_quantum_system(quantum_system)
            .build();

        // Verify the engine was created successfully
        assert!(engine.run_shot().is_ok());
    }
}
