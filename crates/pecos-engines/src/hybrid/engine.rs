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

use crate::Engine;
use crate::byte_message::ByteMessage;
use crate::engine_system::{
    ClassicalControlEngine, ClassicalEngine, ControlEngine, EngineStage, EngineSystem,
};
use crate::quantum_system::QuantumSystem;
use crate::shot_results::Shot;
use dyn_clone;
use log::debug;
use pecos_core::errors::PecosError;
use pecos_core::rng::rng_manageable::derive_seed;

/// Coordinates between classical control and quantum simulation components
///
/// Serves as the central coordination point in the quantum simulation pipeline,
/// managing communication between classical program flow and quantum execution.
///
/// # Noise Application Flow
///
/// ```text
/// HybridEngine
///   +- ClassicalEngine (program control)
///   +- QuantumSystem (quantum execution)
///       +- NoiseModel (transforms operations)
///       +- QuantumEngine (executes operations)
/// ```
///
/// When a classical engine generates quantum commands, they flow through the
/// `QuantumSystem` where noise is applied before execution, producing realistic
/// results that reflect the effects of quantum noise.
///
/// # Role in Monte Carlo Simulations
///
/// In the `MonteCarloEngine`, this engine is:
/// - Cloned for each worker thread
/// - Assigned a unique derived seed
/// - Reset between shots to ensure clean state
///
/// # Example
///
/// ```rust
/// use pecos_engines::hybrid::builder::HybridEngineBuilder;
/// use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
/// use pecos_engines::quantum::StateVecEngine;
///
/// // Create sample engines
/// let classical_engine = Box::new(ExternalClassicalEngine::new());
/// let quantum_engine = Box::new(StateVecEngine::new(2));
///
/// let mut engine = HybridEngineBuilder::new()
///     .with_classical_engine(classical_engine)
///     .with_quantum_engine(quantum_engine)
///     .with_depolarizing_noise(0.01)
///     .build();
///
/// // This would run a single shot but we won't actually run it in the doctest
/// # let _result = engine.run_shot();
/// ```
pub struct HybridEngine {
    /// The classical engine component responsible for program flow and measurement processing
    pub classical_engine: Box<dyn ClassicalControlEngine>,
    /// The quantum system component responsible for executing quantum operations
    pub quantum_system: QuantumSystem,
}

impl HybridEngine {
    /// Set a specific seed for all components of the `HybridEngine`
    ///
    /// This method sets different but deterministic seeds for each component:
    /// - Classical engine (if it implements a seed setting method)
    /// - Quantum system (which further sets seeds for both the quantum engine and noise model)
    ///
    /// # Arguments
    /// * `seed` - Base seed value for random number generators
    pub fn set_seed(&mut self, seed: u64) {
        // Derive seeds for each component
        let classical_seed = derive_seed(seed, "classical_engine");
        let quantum_seed = derive_seed(seed, "quantum_system");

        // Set seed for quantum system (this sets seeds for both quantum engine and noise model)
        self.quantum_system.set_seed(quantum_seed);

        // Set seed for classical engine
        self.classical_engine.set_seed(classical_seed);
    }

    /// Resets the state of the hybrid engine, including classical, quantum, and noise model components.
    ///
    /// This function ensures all components are returned to their initial states,
    /// allowing for reuse in subsequent operations.
    ///
    /// # Errors
    /// Returns a `PecosError` if:
    /// - Resetting the classical engine fails.
    /// - Resetting the engine fails.
    pub fn reset(&mut self) -> Result<(), PecosError> {
        debug!("HybridEngine::reset() being called!");
        // Use the fully qualified path to disambiguate which reset to call
        ClassicalEngine::reset(&mut *self.classical_engine)?;
        self.quantum_system.reset()
    }

    /// Executes a single quantum circuit shot and returns the result.
    ///
    /// # Errors
    /// This function returns a `PecosError` if:
    /// - Resetting the quantum or classical engine fails.
    /// - Generating commands through the classical engine fails.
    /// - Processing commands through the quantum engine fails.
    /// - Handling measurements through the classical engine fails.
    pub fn run_shot(&mut self) -> Result<Shot, PecosError> {
        debug!(
            "HybridEngine::run_shot() starting - Thread {:?}",
            std::thread::current().id()
        );
        let mut stage = self.classical_engine.start(())?;

        // Handle the case where the engine immediately completes with no processing needed
        if let EngineStage::Complete(results) = stage {
            debug!(
                "HybridEngine::run_shot() completed immediately (no commands) - Thread {:?}",
                std::thread::current().id()
            );
            return Ok(results);
        }

        let mut iteration_count = 0;
        while let EngineStage::NeedsProcessing(command_message) = stage {
            iteration_count += 1;
            debug!(
                "HybridEngine::run_shot() iteration {} - Thread {:?}",
                iteration_count,
                std::thread::current().id()
            );
            debug!("Processing quantum commands, iteration {iteration_count}");

            // Process through engine (could be QuantumEngine or EngineSystem)
            let measurement_message = self.quantum_system.process(command_message)?;

            debug!("Calling continue_processing with measurements");
            // Continue classical processing with measurements
            stage = self
                .classical_engine
                .continue_processing(measurement_message)?;
            debug!(
                "continue_processing returned stage: {:?}",
                match &stage {
                    EngineStage::Complete(_) => "Complete",
                    EngineStage::NeedsProcessing(_) => "NeedsProcessing",
                }
            );
        }

        match stage {
            EngineStage::Complete(results) => {
                debug!(
                    "HybridEngine::run_shot() completed after {} iterations - Thread {:?}",
                    iteration_count,
                    std::thread::current().id()
                );
                Ok(results)
            }
            EngineStage::NeedsProcessing(_) => unreachable!(),
        }
    }
}

impl Engine for HybridEngine {
    type Input = ();
    type Output = Shot;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, PecosError> {
        // Delegate to process_as_system for standard implementation
        self.process_as_system(input)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Reset both controller and engine components by using fully qualified path
        ClassicalEngine::reset(&mut *self.classical_engine)?;
        self.quantum_system.reset()
    }
}

impl EngineSystem for HybridEngine {
    type Controller = Box<dyn ClassicalControlEngine>;
    type ControlledEngine = QuantumSystem;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn controller(&self) -> &Self::Controller {
        &self.classical_engine
    }

    fn controller_mut(&mut self) -> &mut Self::Controller {
        &mut self.classical_engine
    }

    fn engine(&self) -> &Self::ControlledEngine {
        &self.quantum_system
    }

    fn engine_mut(&mut self) -> &mut Self::ControlledEngine {
        &mut self.quantum_system
    }
}

impl Clone for HybridEngine {
    fn clone(&self) -> Self {
        Self {
            classical_engine: dyn_clone::clone_box(&*self.classical_engine),
            quantum_system: self.quantum_system.clone(),
        }
    }
}
