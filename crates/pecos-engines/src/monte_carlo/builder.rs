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

use crate::engine_system::{ClassicalControlEngine, ClassicalEngine, HybridEngine};
use crate::hybrid::HybridEngineBuilder;
use crate::monte_carlo::engine::MonteCarloEngine;
use crate::noise::{DepolarizingNoiseModel, NoiseModel};
use crate::quantum::QuantumEngine;
use crate::quantum_system::QuantumSystem;
use pecos_core::errors::PecosError;
use pecos_random::{PecosRng, resolve_seed};

/// Builder for creating a `MonteCarloEngine` with customizable configuration
///
/// This builder provides a fluent interface for constructing a `MonteCarloEngine`
/// with various configuration options. It simplifies the creation process
/// and makes the code more maintainable by centralizing configuration logic.
///
/// # Examples
///
/// ```
/// use pecos_engines::monte_carlo::MonteCarloEngineBuilder;
/// use pecos_engines::quantum;
/// use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
///
/// # fn main() -> Result<(), pecos_core::errors::PecosError> {
/// // Create a basic Monte Carlo engine
/// let engine = MonteCarloEngineBuilder::new()
///     .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
///     .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 42))
///     .build();
///
/// // Create a Monte Carlo engine with depolarizing noise
/// let engine_with_noise = MonteCarloEngineBuilder::new()
///     .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
///     .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 42))
///     .with_depolarizing_noise(0.01)?
///     .build();
///
/// // Create a Monte Carlo engine with a specific seed
/// let seeded_engine = MonteCarloEngineBuilder::new()
///     .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
///     .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 0))
///     .with_seed(42)
///     .build();
/// # Ok(())
/// # }
/// ```
#[derive(Default)]
pub struct MonteCarloEngineBuilder {
    /// Builder for the hybrid engine template
    hybrid_engine_builder: Option<HybridEngineBuilder>,
    /// Pre-built hybrid engine template (overrides builder if provided)
    hybrid_engine: Option<HybridEngine>,
    /// Optional seed for the `MonteCarloEngine`'s RNG
    seed: Option<u64>,
    /// Default number of worker threads
    default_workers: usize,
    explicit_num_qubits: Option<usize>,
}

impl MonteCarloEngineBuilder {
    /// Create a new `MonteCarloEngineBuilder` with default settings
    ///
    /// # Returns
    /// A new `MonteCarloEngineBuilder` with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            hybrid_engine_builder: Some(HybridEngineBuilder::new()),
            hybrid_engine: None,
            seed: None,
            default_workers: 1,
            explicit_num_qubits: None,
        }
    }

    /// Helper method to update the hybrid engine builder or create a new one
    ///
    /// # Arguments
    /// * `hybrid_engine` - The current hybrid engine (if any)
    /// * `hybrid_engine_builder` - The current hybrid engine builder (if any)
    /// * `update_builder` - A function that updates an existing builder
    ///
    /// # Returns
    /// (updated builder, updated engine option)
    fn update_hybrid_builder_with<F>(
        hybrid_engine: Option<HybridEngine>,
        hybrid_engine_builder: Option<HybridEngineBuilder>,
        update_builder: F,
    ) -> (Option<HybridEngineBuilder>, Option<HybridEngine>)
    where
        F: FnOnce(HybridEngineBuilder) -> HybridEngineBuilder,
    {
        if let Some(builder) = hybrid_engine_builder {
            // Create a new clone of the builder with the update applied
            (Some(update_builder(builder)), None)
        } else if let Some(engine) = hybrid_engine {
            // Move all components and their state into the builder before replacing one.
            (
                Some(update_builder(HybridEngineBuilder::from(engine))),
                None,
            )
        } else {
            // If no engine exists, create a new builder and apply the update
            (Some(update_builder(HybridEngineBuilder::new())), None)
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
    pub fn with_classical_engine(mut self, engine: Box<dyn ClassicalControlEngine>) -> Self {
        let update_fn = move |builder: HybridEngineBuilder| builder.with_classical_engine(engine);

        let (builder, engine) = Self::update_hybrid_builder_with(
            self.hybrid_engine,
            self.hybrid_engine_builder,
            update_fn,
        );

        self.hybrid_engine_builder = builder;
        self.hybrid_engine = engine;
        self
    }

    /// Set the quantum engine component
    ///
    /// # Arguments
    /// * `engine` - The quantum engine to use
    ///
    /// # Returns
    /// The builder for method chaining
    #[must_use]
    pub fn with_quantum_engine(mut self, engine: Box<dyn QuantumEngine>) -> Self {
        self.explicit_num_qubits = None;
        let update_fn = move |builder: HybridEngineBuilder| builder.with_quantum_engine(engine);

        let (builder, engine) = Self::update_hybrid_builder_with(
            self.hybrid_engine,
            self.hybrid_engine_builder,
            update_fn,
        );

        self.hybrid_engine_builder = builder;
        self.hybrid_engine = engine;
        self
    }

    /// Set a custom noise model
    ///
    /// # Arguments
    /// * `model` - The noise model to use
    ///
    /// # Returns
    /// The builder for method chaining
    #[must_use]
    pub fn with_noise_model(mut self, model: Box<dyn NoiseModel>) -> Self {
        let update_fn = move |builder: HybridEngineBuilder| builder.with_noise_model(model);

        let (builder, engine) = Self::update_hybrid_builder_with(
            self.hybrid_engine,
            self.hybrid_engine_builder,
            update_fn,
        );

        self.hybrid_engine_builder = builder;
        self.hybrid_engine = engine;
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
    /// `Ok(builder)` with the noise model installed for further method chaining,
    /// or `Err(PecosError)` when the probability is invalid.
    ///
    /// # Errors
    /// Returns an error for a non-finite probability or one outside `0.0..=1.0`.
    pub fn with_depolarizing_noise(self, probability: f64) -> Result<Self, PecosError> {
        Ok(
            self.with_noise_model(Box::new(DepolarizingNoiseModel::try_new_uniform(
                probability,
            )?)),
        )
    }

    /// Set the quantum system
    ///
    /// # Arguments
    /// * `system` - The quantum system to use
    ///
    /// # Returns
    /// The builder for method chaining
    #[must_use]
    pub fn with_quantum_system(mut self, system: QuantumSystem) -> Self {
        self.explicit_num_qubits = None;
        let update_fn = move |builder: HybridEngineBuilder| builder.with_quantum_system(system);

        let (builder, engine) = Self::update_hybrid_builder_with(
            self.hybrid_engine,
            self.hybrid_engine_builder,
            update_fn,
        );

        self.hybrid_engine_builder = builder;
        self.hybrid_engine = engine;
        self
    }

    /// Set a pre-built hybrid engine to use directly
    ///
    /// This overrides any configuration done with the other methods.
    ///
    /// # Arguments
    /// * `engine` - The pre-built hybrid engine
    ///
    /// # Returns
    /// The builder for method chaining
    #[must_use]
    pub fn with_hybrid_engine(mut self, engine: HybridEngine) -> Self {
        self.explicit_num_qubits = None;
        self.hybrid_engine = Some(engine);
        self.hybrid_engine_builder = None;
        self
    }

    /// Set a seed value for the Monte Carlo engine
    ///
    /// This seeds the internal random number generator that draws a base seed for
    /// each run. Every shot reseeds all components from that base seed and its
    /// global index, independently of worker count and draws in earlier shots.
    /// Component seeds use `DefaultHasher`, so seeded samples are not promised
    /// stable across Rust releases.
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

    /// Set the default number of worker threads
    ///
    /// # Arguments
    /// * `workers` - The default number of worker threads to use
    ///
    /// # Returns
    /// The builder for method chaining
    #[must_use]
    pub fn with_default_workers(mut self, workers: usize) -> Self {
        self.default_workers = workers;
        self
    }

    /// Set the fixed number of qubits and the classical runtime allocation hint.
    ///
    /// Replacing the quantum engine/system or the entire hybrid engine supersedes
    /// this count. This replaces the quantum engine with `StateVecEngine::new(num_qubits)`
    /// and keeps the existing noise model and classical engine.
    ///
    /// # Arguments
    /// * `num_qubits` - The number of qubits
    ///
    /// # Returns
    /// The builder for method chaining
    #[must_use]
    pub fn with_num_qubits(mut self, num_qubits: usize) -> Self {
        self.explicit_num_qubits = Some(num_qubits);
        let update_fn = move |builder: HybridEngineBuilder| {
            builder.with_quantum_engine(Box::new(crate::quantum::StateVecEngine::new(num_qubits)))
        };

        let (builder, engine) = Self::update_hybrid_builder_with(
            self.hybrid_engine,
            self.hybrid_engine_builder,
            update_fn,
        );

        self.hybrid_engine_builder = builder;
        self.hybrid_engine = engine;
        self
    }

    /// Build the `MonteCarloEngine` with the configured components
    ///
    /// # Returns
    /// A new `MonteCarloEngine` configured according to the builder settings
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - No hybrid engine has been configured
    /// - Required components like classical engine are missing
    #[must_use]
    pub fn build(self) -> MonteCarloEngine {
        // Determine the hybrid engine to use
        let mut hybrid_engine = if let Some(engine) = self.hybrid_engine {
            engine
        } else if let Some(builder) = self.hybrid_engine_builder {
            builder.build()
        } else {
            panic!(
                "No hybrid engine has been configured. Use either with_hybrid_engine() or other configuration methods."
            );
        };

        if let Some(count) = self.explicit_num_qubits {
            hybrid_engine.classical_engine.set_num_qubits_hint(count);
        }

        // Create a new Monte Carlo engine with the hybrid engine
        let seed = resolve_seed(self.seed);
        let rng = PecosRng::seed_from_u64(seed);

        MonteCarloEngine {
            hybrid_engine_template: hybrid_engine,
            rng,
            seed,
            default_workers: self.default_workers,
        }
    }

    /// Build the `MonteCarloEngine` with the configured components and set the seed
    ///
    /// This is similar to `build()` but also sets the seed.
    ///
    /// # Returns
    /// A new `MonteCarloEngine` configured according to the builder settings with the seed set
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - No hybrid engine has been configured
    /// - Required components like classical engine are missing
    /// - No seed has been set
    #[must_use]
    pub fn build_with_seed(self) -> MonteCarloEngine {
        // Get the seed or panic if not set
        let seed = self.seed.expect(
            "Seed is required for build_with_seed(). Use with_seed() to set one or use build() instead.",
        );

        // Build a hybrid engine with the seed
        let mut hybrid_engine = if let Some(engine) = self.hybrid_engine {
            let mut engine_copy = engine.clone();
            engine_copy.set_seed(seed);
            engine_copy
        } else if let Some(builder) = self.hybrid_engine_builder {
            builder.with_seed(seed).build_with_seed()
        } else {
            panic!(
                "No hybrid engine has been configured. Use either with_hybrid_engine() or other configuration methods."
            );
        };

        if let Some(count) = self.explicit_num_qubits {
            hybrid_engine.classical_engine.set_num_qubits_hint(count);
        }

        // Create a new Monte Carlo engine with the hybrid engine and seed
        MonteCarloEngine {
            hybrid_engine_template: hybrid_engine,
            rng: PecosRng::seed_from_u64(seed),
            seed,
            default_workers: self.default_workers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::engine::ExternalClassicalEngine;
    use crate::quantum;

    #[test]
    fn test_basic_builder() {
        // Create a basic engine
        let engine = MonteCarloEngineBuilder::new()
            .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
            .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 42))
            .build();

        // Just verify that it was created without panic
        assert_eq!(
            engine.hybrid_engine_template.classical_engine.num_qubits(),
            2
        );
    }

    #[test]
    fn test_with_depolarizing_noise() {
        // Create an engine with depolarizing noise
        let engine = MonteCarloEngineBuilder::new()
            .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
            .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 42))
            .with_depolarizing_noise(0.01)
            .unwrap()
            .build();

        // Just verify that it was created without panic
        assert_eq!(
            engine.hybrid_engine_template.classical_engine.num_qubits(),
            2
        );
    }

    #[test]
    fn test_with_seed() {
        // Create an engine with a specific seed
        let engine = MonteCarloEngineBuilder::new()
            .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
            .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 42))
            .with_seed(42)
            .build();

        // Verify the seed was stored correctly
        assert_eq!(engine.seed, 42);
    }

    #[test]
    #[should_panic(expected = "Classical engine is required")]
    fn test_empty_builder() {
        // Create an engine without any configuration
        let _ = MonteCarloEngineBuilder::new().build();
    }

    #[test]
    fn test_with_hybrid_engine() {
        // Create a hybrid engine
        let hybrid_engine = HybridEngineBuilder::new()
            .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
            .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 42))
            .build();

        // Create a Monte Carlo engine with the hybrid engine
        let engine = MonteCarloEngineBuilder::new()
            .with_hybrid_engine(hybrid_engine)
            .build();

        // Just verify that it was created without panic
        assert_eq!(
            engine.hybrid_engine_template.classical_engine.num_qubits(),
            2
        );
    }

    #[test]
    fn test_with_num_qubits() {
        // Create an engine with a specified number of qubits
        let engine = MonteCarloEngineBuilder::new()
            .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
            .with_num_qubits(3)
            .build();

        // Just verify that it was created without panic
        assert_eq!(
            engine.hybrid_engine_template.classical_engine.num_qubits(),
            2
        );
    }

    #[test]
    fn test_change_quantum_engine_after_hybrid_engine() {
        // Create a hybrid engine
        let hybrid_engine = HybridEngineBuilder::new()
            .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
            .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 42))
            .build();

        // Create a Monte Carlo engine with the hybrid engine and then change the quantum engine
        let engine = MonteCarloEngineBuilder::new()
            .with_hybrid_engine(hybrid_engine)
            .with_quantum_engine(quantum::new_quantum_engine_with_seed(3, 42))
            .build();

        // Just verify that it was created without panic
        assert_eq!(
            engine.hybrid_engine_template.classical_engine.num_qubits(),
            2
        );
    }
}
