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
use crate::core::shot_results::{ShotResult, ShotResults};
use crate::engines::hybrid::HybridEngineBuilder;
use crate::engines::noise::NoiseModel;
use crate::engines::quantum::{QuantumEngine, StateVecEngine};
use crate::engines::{ClassicalEngine, ControlEngine, Engine, EngineStage, HybridEngine};
use crate::errors::QueueError;
use log::{debug, info};
use pecos_core::rng::RngManageable;
use pecos_core::rng::rng_manageable::derive_seed;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::builder::MonteCarloEngineBuilder;

/// Orchestrates parallel Monte Carlo simulations of quantum programs with noise
///
/// # Architecture
///
/// ```text
/// MonteCarloEngine
///   +- HybridEngine (template, cloned for each worker)
///       +- ClassicalEngine (controls execution flow)
///       +- QuantumSystem (performs quantum operations)
///           +- NoiseModel (applies noise to operations)
///           +- QuantumEngine (simulates quantum operations)
/// ```
///
/// # Key Features
///
/// - **Parallelization**: Distributes shots across multiple worker threads
/// - **Seed Management**: Hierarchical seeding for reproducible results
///   - Base seed → Worker seeds → Component seeds
/// - **Noise Integration**: Applies noise before quantum operations
///
/// # Best Practices
///
/// - **Noise Levels**: 0.001-0.01 (0.1-1%) for hardware-like simulations
/// - **Shot Count**: 1000+ for noisy simulations
/// - **Workers**: Set to (CPU cores - 1) for optimal performance
/// - **Testing**: Use higher noise (~0.3) and fixed seeds
///
/// # Example
///
/// ```rust
/// use pecos_engines::engines::monte_carlo::MonteCarloEngine;
/// use pecos_engines::engines::monte_carlo::engine::ExternalClassicalEngine;
/// use pecos_engines::engines::quantum::StateVecEngine;
///
/// // Create sample engines
/// let classical_engine = Box::new(ExternalClassicalEngine::new());
/// let quantum_engine = Box::new(StateVecEngine::new(2));
///
/// // Build the Monte Carlo engine
/// let mut engine = MonteCarloEngine::builder()
///     .with_classical_engine(classical_engine)
///     .with_quantum_engine(quantum_engine)
///     .with_depolarizing_noise(0.01)
///     .build();
///
/// // For reproducibility
/// engine.set_seed(42);
///
/// // This would run the simulation but we won't actually run it in the doctest
/// # let num_shots = 10; // Using a small number for the doctest
/// # let num_workers = 1; // Using a single worker for the doctest
/// # let _results = engine.run(num_shots, num_workers);
/// ```
pub struct MonteCarloEngine {
    /// Template `HybridEngine` that is cloned for each worker
    pub hybrid_engine_template: HybridEngine,
    /// Random number generator for seed generation
    pub rng: ChaCha8Rng,
}

impl MonteCarloEngine {
    /// Create a new Monte Carlo engine with default settings.
    ///
    /// This method returns a builder that can be used to configure the engine.
    /// See [`MonteCarloEngineBuilder`] for configuration options.
    ///
    /// # Examples
    ///
    /// ```
    /// // Import necessary types for the example
    /// use pecos_engines::engines::monte_carlo::MonteCarloEngine;
    /// use pecos_engines::engines::monte_carlo::engine::ExternalClassicalEngine;
    /// use pecos_engines::engines::quantum;
    ///
    /// // Create a Monte Carlo engine with default settings
    /// let classical_engine = Box::new(ExternalClassicalEngine::new());
    /// let mut engine = MonteCarloEngine::builder()
    ///     .with_classical_engine(classical_engine)
    ///     .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 42))
    ///     .build();
    /// ```
    #[must_use]
    pub fn builder() -> MonteCarloEngineBuilder {
        MonteCarloEngineBuilder::new()
    }

    /// Convenience method to create a Monte Carlo engine with a classical engine and default components.
    ///
    /// This is the simplest way to create a Monte Carlo engine when you only have a classical engine.
    /// It will automatically create a state vector quantum engine and a pass-through noise model.
    ///
    /// # Parameters
    /// - `classical_engine`: The classical engine to use for the simulation.
    ///
    /// # Returns
    /// A configured `MonteCarloEngine` ready for use.
    ///
    /// # Examples
    ///
    /// ```
    /// // Import necessary types for the example
    /// use pecos_engines::engines::monte_carlo::MonteCarloEngine;
    /// use pecos_engines::engines::monte_carlo::engine::ExternalClassicalEngine;
    /// use pecos_engines::engines::quantum;
    ///
    /// // Create a Monte Carlo engine with default settings
    /// let classical_engine = Box::new(ExternalClassicalEngine::new());
    /// let mut engine = MonteCarloEngine::new_with_defaults(classical_engine);
    /// ```
    #[must_use]
    pub fn new_with_defaults(classical_engine: Box<dyn ClassicalEngine>) -> Self {
        // Use the builder pattern
        let num_qubits = classical_engine.num_qubits();
        Self::builder()
            .with_classical_engine(classical_engine)
            .with_quantum_engine(Box::new(StateVecEngine::new(num_qubits)))
            .build()
    }

    /// Create a Monte Carlo engine with a classical engine and a depolarizing noise model.
    ///
    /// This is a convenience method that sets up a `MonteCarloEngine` with a state vector
    /// quantum engine and a depolarizing noise model with the specified probability.
    ///
    /// # Parameters
    /// - `classical_engine`: The classical engine to use for the simulation.
    /// - `p`: The probability parameter for the depolarizing noise model (between 0.0 and 1.0).
    ///
    /// # Returns
    /// A configured `MonteCarloEngine` ready for use.
    ///
    /// # Examples
    ///
    /// ```
    /// // Import necessary types for the example
    /// use pecos_engines::engines::monte_carlo::MonteCarloEngine;
    /// use pecos_engines::engines::monte_carlo::engine::ExternalClassicalEngine;
    /// use pecos_engines::engines::quantum;
    ///
    /// // Create a Monte Carlo engine with depolarizing noise
    /// let classical_engine = Box::new(ExternalClassicalEngine::new());
    /// let mut engine = MonteCarloEngine::builder()
    ///     .with_classical_engine(classical_engine)
    ///     .with_quantum_engine(quantum::new_quantum_engine_with_seed(2, 42))
    ///     .with_depolarizing_noise(0.01)
    ///     .build();
    /// ```
    #[must_use]
    pub fn new_with_depolarizing_noise(classical_engine: Box<dyn ClassicalEngine>, p: f64) -> Self {
        // Use the builder pattern
        Self::builder()
            .with_classical_engine(classical_engine)
            .with_depolarizing_noise(p)
            .build()
    }

    /// Set a specific seed for the random number generator.
    ///
    /// Setting a seed ensures deterministic behavior across runs with the same seed.
    /// This method sets the seed for:
    /// - The internal `ChaCha8Rng` used for shot distribution
    /// - The template `HybridEngine` (which sets seeds for the noise model and quantum engine)
    ///
    /// # Arguments
    /// * `seed` - The seed value for the random number generators
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Errors
    /// Returns a `QueueError` if setting the seed fails for any component
    pub fn set_seed(&mut self, seed: u64) -> Result<(), QueueError> {
        // Set the seed for the internal RNG
        self.rng = ChaCha8Rng::seed_from_u64(seed);

        // Set the seed for the hybrid engine template
        self.hybrid_engine_template.set_seed(seed)?;

        Ok(())
    }

    /// Run a Monte Carlo simulation with the specified number of shots and worker threads.
    ///
    /// This method executes multiple shots of the quantum program in parallel using
    /// the configured components. It distributes the shots across the specified number
    /// of workers and collects the results.
    ///
    /// # Parameters
    /// - `num_shots`: The total number of circuit executions to perform.
    /// - `num_workers`: The number of worker threads to use for parallel execution.
    ///
    /// # Returns
    /// Aggregated results from all shots.
    ///
    /// # Errors
    /// Returns a `QueueError` if any part of the simulation fails.
    ///
    /// # Panics
    /// - If `num_shots` is zero.
    /// - If `num_workers` is zero.
    pub fn run(&mut self, num_shots: usize, num_workers: usize) -> Result<ShotResults, QueueError> {
        assert!((num_shots != 0), "num_shots cannot be zero");

        assert!((num_workers != 0), "num_workers cannot be zero");

        debug!(
            "Running Monte Carlo simulation with {} shots on {} workers",
            num_shots, num_workers
        );

        // Create a vector to hold the results with worker ID and shot index information
        // (worker_idx, shot_idx, result)
        let results_vec = Arc::new(Mutex::new(
            Vec::<(usize, usize, ShotResult)>::with_capacity(num_shots),
        ));

        // Calculate work distribution (shots per worker)
        let shots_per_worker = distribute_shots(num_shots, num_workers);

        // Seed management: derive seeds for each worker deterministically from the base seed
        let base_seed = self.rng.next_u64();
        let worker_seeds: Vec<u64> = (0..num_workers)
            .map(|idx| {
                let context = format!("worker_{idx}");
                derive_seed(base_seed, &context)
            })
            .collect();

        info!(
            "Distributing {} shots across {} workers",
            num_shots, num_workers
        );

        // Run the shots in parallel
        let _ = (0..num_workers)
            .into_par_iter()
            .map(|worker_idx| {
                let shots_this_worker = shots_per_worker[worker_idx];
                if shots_this_worker == 0 {
                    return Ok(());
                }

                // Create a copy of the template engine and set its seed
                let mut engine = self.hybrid_engine_template.clone();
                let worker_seed = worker_seeds[worker_idx];

                // Set seed for this worker's engine
                if let Err(e) = engine.set_seed(worker_seed) {
                    return Err(QueueError::OperationError(format!(
                        "Failed to set seed for worker {worker_idx}: {e}"
                    )));
                }

                // Run assigned shots
                debug!(
                    "Worker {} running {} shots with seed {}",
                    worker_idx, shots_this_worker, worker_seed
                );
                for shot_idx in 0..shots_this_worker {
                    // Reset the engine state before each shot
                    engine.reset()?;

                    let shot_result = engine.run_shot()?;

                    // Store the result with the worker index and shot index for deterministic ordering
                    let mut results = results_vec.lock().unwrap();
                    results.push((worker_idx, shot_idx, shot_result));
                }

                Ok(())
            })
            .collect::<Result<Vec<()>, QueueError>>()?;

        // Sort the results by worker ID and then by shot index within each worker
        // This ensures a completely deterministic ordering regardless of execution timing
        let mut results = results_vec.lock().unwrap();
        results.sort_by(|(w1, s1, _), (w2, s2, _)| w1.cmp(w2).then(s1.cmp(s2)));

        // Extract just the shot results in the sorted order
        let shot_results: Vec<ShotResult> =
            results.iter().map(|(_, _, shot)| shot.clone()).collect();

        // Convert the results to a ShotResults object
        let combined_results = ShotResults::from_measurements(&shot_results);

        debug!("Monte Carlo simulation completed successfully");
        Ok(combined_results)
    }

    /// Run a simulation using the provided engines directly.
    ///
    /// This convenience method creates a `HybridEngine` from the provided components
    /// and then runs the Monte Carlo simulation.
    ///
    /// # Parameters
    /// - `classical_engine`: The classical engine to use for the simulation.
    /// - `noise_model`: The noise model to apply during the simulation.
    /// - `quantum_engine`: The quantum engine to use for the simulation.
    /// - `num_shots`: The number of shots to execute in the simulation.
    /// - `num_workers`: The number of parallel workers to use.
    /// - `seed`: Optional seed for deterministic behavior.
    ///
    /// # Returns
    /// - `Ok(ShotResults)`: The results from the simulation.
    /// - `Err(QueueError)`: If an error occurs during the configuration or simulation.
    ///
    /// # Errors
    /// This function will return a `QueueError` if:
    /// - There is an error during the execution of the simulation.
    pub fn run_with_engines(
        classical_engine: Box<dyn ClassicalEngine>,
        noise_model: Box<dyn NoiseModel>,
        quantum_engine: Box<dyn QuantumEngine>,
        num_shots: usize,
        num_workers: usize,
        seed: Option<u64>,
    ) -> Result<ShotResults, QueueError> {
        // Create a HybridEngine from the components
        let hybrid_engine = HybridEngineBuilder::new()
            .with_classical_engine(classical_engine)
            .with_quantum_engine(quantum_engine)
            .with_noise_model(noise_model)
            .build();

        // Use the new method to run with the hybrid engine
        Self::run_with_hybrid_engine(hybrid_engine, num_shots, num_workers, seed)
    }

    /// Static method to run a simulation with a pre-configured hybrid engine.
    ///
    /// This method is useful when you have a hybrid engine that you want to use
    /// for Monte Carlo simulation without creating a full `MonteCarloEngine` instance.
    ///
    /// # Parameters
    /// - `hybrid_engine`: The pre-configured hybrid engine to use.
    /// - `num_shots`: The total number of circuit executions to perform.
    /// - `num_workers`: The number of worker threads to use for parallel execution.
    /// - `seed`: Optional seed for deterministic behavior.
    ///
    /// # Returns
    /// Aggregated results from all shots.
    ///
    /// # Errors
    /// Returns a `QueueError` if any part of the simulation fails.
    pub fn run_with_hybrid_engine(
        hybrid_engine: HybridEngine,
        num_shots: usize,
        num_workers: usize,
        seed: Option<u64>,
    ) -> Result<ShotResults, QueueError> {
        // Create a Monte Carlo engine with the provided hybrid engine
        let mut engine = MonteCarloEngineBuilder::new()
            .with_hybrid_engine(hybrid_engine)
            .build();

        // Set the seed if provided
        if let Some(s) = seed {
            engine.set_seed(s)?;
        }

        // Run the simulation
        engine.run(num_shots, num_workers)
    }

    /// Static method to run a simulation with a classical engine and any noise model.
    ///
    /// This is a generic method that sets up a `MonteCarloEngine` with a state vector
    /// quantum engine and any provided noise model. This is a more flexible approach
    /// than the specialized methods for specific noise models.
    ///
    /// # Parameters
    /// - `classical_engine`: The classical engine to use.
    /// - `noise_model`: The noise model to apply during simulation.
    /// - `num_shots`: The total number of circuit executions to perform.
    /// - `num_workers`: The number of worker threads to use for parallel execution.
    /// - `seed`: Optional seed for deterministic behavior.
    ///
    /// # Returns
    /// Aggregated results from all shots.
    ///
    /// # Errors
    /// Returns a `QueueError` if any part of the simulation fails.
    pub fn run_with_noise_model(
        classical_engine: Box<dyn ClassicalEngine>,
        noise_model: Box<dyn NoiseModel>,
        num_shots: usize,
        num_workers: usize,
        seed: Option<u64>,
    ) -> Result<ShotResults, QueueError> {
        // Create a quantum engine with the same number of qubits as the classical engine
        let num_qubits = classical_engine.num_qubits();
        let quantum_engine = Box::new(StateVecEngine::new(num_qubits));

        // Create a hybrid engine with the provided components
        let mut hybrid_engine = HybridEngineBuilder::new()
            .with_classical_engine(classical_engine)
            .with_quantum_engine(quantum_engine)
            .with_noise_model(noise_model)
            .build();

        // If a seed is provided, explicitly set it on the hybrid engine
        if let Some(s) = seed {
            hybrid_engine.set_seed(s)?;
        }

        Self::run_with_hybrid_engine(hybrid_engine, num_shots, num_workers, seed)
    }

    /// Static method to run a simulation based on a configuration string.
    ///
    /// This method is intended for use with configuration management systems where
    /// the engine configuration is specified as a string.
    ///
    /// # Parameters
    /// - `config`: Configuration string specifying the engine components.
    /// - `num_shots`: The total number of circuit executions to perform.
    /// - `num_workers`: The number of worker threads to use for parallel execution.
    /// - `seed`: Optional seed for deterministic behavior.
    ///
    /// # Returns
    /// Aggregated results from all shots.
    ///
    /// # Errors
    /// Returns a `QueueError` if any part of the simulation fails.
    pub fn run_with_config(
        config: &str,
        num_shots: usize,
        num_workers: usize,
        seed: Option<u64>,
    ) -> Result<ShotResults, QueueError> {
        // Parse the configuration string and create the engine
        // For now, we'll treat it as a simple noise probability
        let p = config.parse::<f64>().map_err(|e| {
            QueueError::OperationError(format!("Failed to parse config string as float: {e}"))
        })?;

        let classical_engine = Box::new(ExternalClassicalEngine::new());

        // Create a depolarizing noise model with the parsed probability
        let mut noise_model = crate::engines::noise::DepolarizingNoiseModel::new_uniform(p);

        // If a seed is provided, set it on the noise model
        if let Some(s) = seed {
            let noise_seed = pecos_core::rng::rng_manageable::derive_seed(s, "noise_model");
            noise_model.set_seed(noise_seed)?;
        }

        Self::run_with_noise_model(
            classical_engine,
            Box::new(noise_model),
            num_shots,
            num_workers,
            seed,
        )
    }
}

impl Clone for MonteCarloEngine {
    fn clone(&self) -> Self {
        Self {
            hybrid_engine_template: self.hybrid_engine_template.clone(),
            rng: self.rng.clone(),
        }
    }
}

/// Utility function to distribute shots across workers
///
/// This function calculates how many shots each worker should execute
/// based on the total number of shots and workers.
///
/// # Arguments
/// * `num_shots` - The total number of shots to distribute
/// * `num_workers` - The number of workers available
///
/// # Returns
/// A vector where each element is the number of shots for a worker
fn distribute_shots(num_shots: usize, num_workers: usize) -> Vec<usize> {
    let mut shots_per_worker = vec![num_shots / num_workers; num_workers];
    let remainder = num_shots % num_workers;

    // Distribute the remainder shots among the first few workers
    shots_per_worker
        .iter_mut()
        .take(remainder)
        .for_each(|shots| *shots += 1);

    shots_per_worker
}

/// An external classical engine implementation used for testing and examples.
///
/// This implementation provides a basic classical engine that returns predetermined results
/// for demonstration and testing purposes.
#[derive(Debug, Clone)]
pub struct ExternalClassicalEngine {
    results: HashMap<String, i64>,
}

impl Default for ExternalClassicalEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ExternalClassicalEngine {
    /// Create a new `ExternalClassicalEngine` with default results.
    #[must_use]
    pub fn new() -> Self {
        // Initialize with a default results map
        let mut results = HashMap::new();
        results.insert("result".to_string(), 0);

        Self { results }
    }
}

impl Engine for ExternalClassicalEngine {
    type Input = ();
    type Output = ShotResult;

    fn process(&mut self, _input: Self::Input) -> Result<Self::Output, QueueError> {
        // Generate a ByteMessage with a simple circuit
        let _message = self.generate_commands()?;

        // Process it somehow (in a real engine, this would run the quantum simulation)
        // For this stub, we'll just return the stored results
        let mut shot_result = ShotResult::default();

        // Convert the HashMap<String, i64> to HashMap<String, u32>
        let measurements: HashMap<String, u32> = self
            .results
            .iter()
            .map(|(k, v)| {
                // For a test utility, simply clamp values that are out of bounds
                let value = u32::try_from(*v).unwrap_or(0);
                (k.clone(), value)
            })
            .collect();

        shot_result.measurements = measurements;

        Ok(shot_result)
    }

    fn reset(&mut self) -> Result<(), QueueError> {
        // Reset all results to 0
        for value in self.results.values_mut() {
            *value = 0;
        }

        Ok(())
    }
}

impl ClassicalEngine for ExternalClassicalEngine {
    fn num_qubits(&self) -> usize {
        // Default to 2 qubits for testing
        2
    }

    fn generate_commands(&mut self) -> Result<ByteMessage, QueueError> {
        // Create a simple command that prepares and measures a qubit
        Ok(ByteMessage::builder().build())
    }

    fn handle_measurements(&mut self, _: ByteMessage) -> Result<(), QueueError> {
        // Store a random result
        Ok(())
    }

    fn get_results(&self) -> Result<ShotResult, QueueError> {
        // Create a ShotResult with the stored results
        let mut shot_result = ShotResult::default();

        // Convert the HashMap<String, i64> to HashMap<String, u32>
        let measurements: HashMap<String, u32> = self
            .results
            .iter()
            .map(|(k, v)| {
                // For a test utility, simply clamp values that are out of bounds
                let value = u32::try_from(*v).unwrap_or(0);
                (k.clone(), value)
            })
            .collect();

        shot_result.measurements = measurements;

        Ok(shot_result)
    }

    fn compile(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Nothing to compile for this stub
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ControlEngine for ExternalClassicalEngine {
    type Input = ();
    type Output = ShotResult;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, (): ()) -> Result<EngineStage<ByteMessage, ShotResult>, QueueError> {
        // Generate commands and return NeedsProcessing
        let commands = self.generate_commands()?;
        Ok(EngineStage::NeedsProcessing(commands))
    }

    fn continue_processing(
        &mut self,
        results: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ShotResult>, QueueError> {
        // Process the results and return Complete
        self.handle_measurements(results)?;
        let shot_result = self.get_results()?;
        Ok(EngineStage::Complete(shot_result))
    }

    fn reset(&mut self) -> Result<(), QueueError> {
        Engine::reset(self)
    }
}
