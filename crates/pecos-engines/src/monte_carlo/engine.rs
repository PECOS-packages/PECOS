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
    ClassicalControlEngine, ClassicalEngine, ControlEngine, EngineStage, HybridEngine,
};
use crate::hybrid::HybridEngineBuilder;
use crate::noise::depolarizing::DepolarizingFaultCatalog;
use crate::noise::{DepolarizingNoiseModel, DepolarizingSampledFault, NoiseModel};
use crate::quantum::{QuantumEngine, StateVecEngine};
use crate::shot_results::{Data, Shot, ShotVec};
use log::debug;
use pecos_core::errors::PecosError;
use pecos_core::rng::RngManageable;
use pecos_core::rng::rng_manageable::derive_seed;
use pecos_random::PecosRng;
use rayon::{
    ThreadPoolBuilder,
    iter::{IntoParallelIterator, ParallelIterator},
};
use std::any::Any;
use std::collections::BTreeMap;
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
/// # Tips
///
/// - **Noise Levels**: 0.001-0.01 (0.1-1%) for hardware-like simulations
/// - **Shot Count**: 1000+ for noisy simulations
/// - **Workers**: Set to (CPU cores - 1) for optimal performance
/// - **Testing**: Use higher noise (~0.3) and fixed seeds
///
/// # Example
///
/// ```rust
/// use pecos_engines::monte_carlo::MonteCarloEngine;
/// use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
/// use pecos_engines::quantum::StateVecEngine;
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
/// # let _results = engine.run(num_shots);
/// ```
pub struct MonteCarloEngine {
    /// Template `HybridEngine` that is cloned for each worker
    pub hybrid_engine_template: HybridEngine,
    /// Random number generator for seed generation
    pub rng: PecosRng,
    /// The seed used to initialize the RNG
    pub seed: u64,
    /// Default number of worker threads
    pub default_workers: usize,
    /// If true, collect sampled fault histories during `run`.
    pub fault_history_enabled: bool,
}

/// Per-shot sampled-fault history for depolarizing runs.
pub type DepolarizingFaultHistory = Vec<DepolarizingSampledFault>;

/// Result payload for Monte Carlo runs.
///
/// When fault-history tracking is disabled, `fault_histories` is empty.
#[derive(Debug, Clone)]
pub struct MonteCarloRunResult {
    /// Aggregated measurement results.
    pub results: ShotVec,
    /// One history vector per shot, in shot order.
    pub fault_histories: Vec<DepolarizingFaultHistory>,
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
    /// use pecos_engines::monte_carlo::MonteCarloEngine;
    /// use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
    /// use pecos_engines::quantum;
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
    /// use pecos_engines::monte_carlo::MonteCarloEngine;
    /// use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
    /// use pecos_engines::quantum;
    ///
    /// // Create a Monte Carlo engine with default settings
    /// let classical_engine = Box::new(ExternalClassicalEngine::new());
    /// let mut engine = MonteCarloEngine::new_with_defaults(classical_engine);
    /// ```
    #[must_use]
    pub fn new_with_defaults(classical_engine: Box<dyn ClassicalControlEngine>) -> Self {
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
    /// use pecos_engines::monte_carlo::MonteCarloEngine;
    /// use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
    /// use pecos_engines::quantum;
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
    pub fn new_with_depolarizing_noise(
        classical_engine: Box<dyn ClassicalControlEngine>,
        p: f64,
    ) -> Self {
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
    /// - The internal `PecosRng` used for shot distribution
    /// - The template `HybridEngine` (which sets seeds for the noise model and quantum engine)
    ///
    /// # Arguments
    /// * `seed` - The seed value for the random number generators
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = PecosRng::seed_from_u64(seed);
        self.hybrid_engine_template.set_seed(seed);
    }

    /// Reset the Monte Carlo engine to its initial state.
    ///
    /// This resets the hybrid engine template (including the quantum state back to |0⟩)
    /// and re-seeds the RNG with the original seed for reproducibility.
    ///
    /// # Returns
    /// The engine itself for method chaining.
    ///
    /// # Errors
    /// Returns a `PecosError` if resetting the hybrid engine fails.
    pub fn reset(&mut self) -> Result<&mut Self, PecosError> {
        // Reset the hybrid engine template (resets quantum state to |0⟩)
        self.hybrid_engine_template.reset()?;
        // Re-seed the RNG with the original seed for reproducibility
        self.rng = PecosRng::seed_from_u64(self.seed);
        Ok(self)
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
    /// Returns a `PecosError` if any part of the simulation fails.
    ///
    /// # Panics
    /// - If `num_shots` is zero.
    pub fn run(&mut self, num_shots: usize) -> Result<MonteCarloRunResult, PecosError> {
        self.run_with_workers(num_shots, self.default_workers)
    }

    /// Run the Monte Carlo simulation with a specified number of worker threads.
    ///
    /// This method runs the simulation with the specified number of shots and worker threads,
    /// overriding the default worker count configured during construction.
    ///
    /// # Arguments
    /// * `num_shots` - The number of shots to run
    /// * `num_workers` - The number of parallel worker threads to use
    ///
    /// # Returns
    /// Aggregated results from all shots.
    ///
    /// # Errors
    /// Returns a `PecosError` if any part of the simulation fails.
    ///
    /// # Panics
    /// - If `num_shots` is zero.
    /// - If `num_workers` is zero.
    pub fn run_with_workers(
        &mut self,
        num_shots: usize,
        num_workers: usize,
    ) -> Result<MonteCarloRunResult, PecosError> {
        assert!(num_shots > 0, "num_shots cannot be zero");
        assert!(num_workers > 0, "num_workers cannot be zero");

        debug!("Running Monte Carlo simulation: {num_shots} shots, {num_workers} workers");

        // Shared results collection
        let results_vec = Arc::new(Mutex::new(Vec::<(usize, usize, Shot)>::with_capacity(
            num_shots,
        )));

        // Shared fault history collection if enabled
        let fault_histories_vec = if self.fault_history_enabled {
            Some(Arc::new(Mutex::new(Vec::<(
                usize,
                usize,
                DepolarizingFaultHistory,
            )>::with_capacity(
                num_shots
            ))))
        } else {
            None
        };

        // Determine shots per worker and generate deterministic seeds
        let shots_per_worker = distribute_shots(num_shots, num_workers);
        let base_seed = self.rng.next_u64();

        // CRITICAL: Pre-create worker engines on the main thread before parallel execution.
        // This avoids potential deadlocks when worker threads try to clone engines
        // simultaneously, which can trigger concurrent library loading operations
        // that contend with each other or the dynamic linker.
        let worker_engines: Vec<_> = (0..num_workers)
            .map(|worker_idx| {
                let mut engine = self.hybrid_engine_template.clone();
                let worker_seed = derive_seed(base_seed, &format!("worker_{worker_idx}"));
                engine.set_seed(worker_seed);
                (worker_idx, shots_per_worker[worker_idx], engine)
            })
            .collect();

        // Create a dedicated thread pool for this simulation to avoid contention
        // with global Rayon thread pool when multiple simulations run concurrently.
        // CRITICAL: For QIS programs, we need to ensure each test gets its own
        // isolated thread pool to prevent TLS conflicts during library cleanup.
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .thread_name(|index| format!("pecos-mc-worker-{index}"))
            .build()
            .map_err(|e| PecosError::Processing(format!("Failed to create thread pool: {e}")))?;

        // Run shots in parallel across workers using dedicated thread pool
        // CRITICAL: Use install() to ensure all work completes before thread pool cleanup
        let parallel_result = thread_pool.install(|| {
            worker_engines
                .into_par_iter()
                .map(|(worker_idx, shots_this_worker, mut engine)| {
                    if shots_this_worker == 0 {
                        return Ok(());
                    }

                    // Check that fault history is enabled only if using depolarizing noise
                    if self.fault_history_enabled {
                        let noise_model = engine.quantum_system.noise_model_mut();
                        let depolarizing = noise_model
                            .as_any_mut()
                            .downcast_mut::<DepolarizingNoiseModel>()
                            .ok_or_else(|| {
                                PecosError::Input(
                                    "fault_history_enabled requires DepolarizingNoiseModel"
                                        .to_string(),
                                )
                            })?;

                        // Turn on fault history collection
                        depolarizing.set_sampled_fault_history_enabled(true);
                    }

                    // Process all shots for this worker
                    debug!("Worker {worker_idx} running {shots_this_worker} shots");

                    for shot_idx in 0..shots_this_worker {
                        engine.reset()?;

                        // Catch panics during shot execution and convert to PecosError
                        let shot_result =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                engine.run_shot()
                            }));

                        let shot_result = match shot_result {
                            Ok(Ok(result)) => result,
                            Ok(Err(e)) => return Err(e),
                            Err(panic_payload) => {
                                // Convert panic to PecosError
                                let panic_msg =
                                    if let Some(s) = panic_payload.downcast_ref::<String>() {
                                        s.clone()
                                    } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
                                        (*s).to_string()
                                    } else {
                                        "Unknown panic occurred during shot execution".to_string()
                                    };

                                return Err(PecosError::Processing(format!(
                                    "Shot execution failed: {panic_msg}"
                                )));
                            }
                        };

                        // Store with worker/shot indices for deterministic ordering
                        results_vec.lock().expect("results mutex poisoned").push((
                            worker_idx,
                            shot_idx,
                            shot_result,
                        ));

                        // Extract the fault history from the noise model
                        if let Some(histories_shared) = &fault_histories_vec {
                            let history = {
                                let noise_model = engine.quantum_system.noise_model_mut();
                                let depolarizing = noise_model
                                    .as_any_mut()
                                    .downcast_mut::<DepolarizingNoiseModel>()
                                    .ok_or_else(|| {
                                        PecosError::Input(
                                            "fault_history_enabled requires DepolarizingNoiseModel"
                                                .to_string(),
                                        )
                                    })?;
                                depolarizing
                                    .sampled_fault_history()
                                    .map_or_else(Vec::new, |history| history.to_vec())
                            };

                            // Add history to the history vector
                            histories_shared
                                .lock()
                                .expect("fault histories mutex poisoned")
                                .push((worker_idx, shot_idx, history));
                        }
                    }

                    Ok(())
                })
                .collect::<Result<Vec<()>, PecosError>>()
        });

        // Handle the parallel execution result
        parallel_result?;

        // CRITICAL: Explicitly drop the thread pool to ensure clean shutdown
        // This helps prevent TLS issues during test cleanup
        drop(thread_pool);

        // Ensure deterministic ordering of results
        let mut results = results_vec.lock().expect("results mutex poisoned");
        results.sort_by(|(w1, s1, _), (w2, s2, _)| w1.cmp(w2).then(s1.cmp(s2)));

        // Convert to final results format
        let shot_results: Vec<Shot> = results.iter().map(|(_, _, shot)| shot.clone()).collect();
        let combined_results = ShotVec::from_measurements(&shot_results);

        // Sort the fault histories into the shot order
        let combined_histories = if let Some(histories_shared) = fault_histories_vec {
            // Sort the histories by worker and shot index to ensure deterministic ordering
            let mut histories = histories_shared
                .lock()
                .expect("fault histories mutex poisoned");
            histories.sort_by(|(w1, s1, _), (w2, s2, _)| w1.cmp(w2).then(s1.cmp(s2)));
            histories
                .iter()
                .map(|(_, _, history)| history.clone())
                .collect::<Vec<DepolarizingFaultHistory>>()
        } else {
            Vec::new()
        };

        debug!("Monte Carlo simulation completed successfully");
        Ok(MonteCarloRunResult {
            results: combined_results,
            fault_histories: combined_histories,
        })
    }

    /// Performs a "dry run" of the Monte Carlo simulation to collect
    /// the fault locations without actually doing any sampling
    pub fn return_fault_catalog(&mut self) -> Result<DepolarizingFaultCatalog, PecosError> {
        if !self.fault_history_enabled {
            return Err(PecosError::Input(
                "catalog_faults requires fault_history_enabled to be true".to_string(),
            ));
        }
        let mut engine = self.hybrid_engine_template.clone();
        let noise_model = engine.quantum_system.noise_model_mut();
        let depolarizing = noise_model
            .as_any_mut()
            .downcast_mut::<DepolarizingNoiseModel>()
            .ok_or_else(|| {
                PecosError::Input("catalog_faults requires DepolarizingNoiseModel".to_string())
            })?;
        let msg = engine
            .classical_engine
            .generate_commands()
            .unwrap_or_else(|e| {
                panic!("Failed to generate commands for fault catalog: {e}");
            });
        depolarizing.build_fault_catalog_from_message(&msg)
    }

    /// Runs by specifying a set of fault histories. There should be one
    /// fault history per shot.
    ///
    /// # Errors
    /// Returns [`PecosError::Input`] when the configured noise model is not
    /// [`DepolarizingNoiseModel`].
    pub fn run_with_fault_histories(
        &mut self,
        fault_histories: Vec<DepolarizingFaultHistory>,
    ) -> Result<MonteCarloRunResult, PecosError> {
        if fault_histories.is_empty() {
            return Err(PecosError::Input(
                "run_with_fault_histories requires at least one fault history".to_string(),
            ));
        }

        let num_shots = fault_histories.len();
        let mut worker_engine = self.hybrid_engine_template.clone();
        let worker_seed = derive_seed(self.rng.next_u64(), "specified_history_worker_0");
        worker_engine.set_seed(worker_seed);

        let mut shots = Vec::with_capacity(num_shots);
        let mut applied_histories = Vec::with_capacity(num_shots);

        for history in fault_histories {
            worker_engine.reset()?;

            // Set the fault history in the noise model for this shot
            {
                let noise_model = worker_engine.quantum_system.noise_model_mut();
                // Throw an error for now if it is not depolarizing noise
                let depolarizing = noise_model
                    .as_any_mut()
                    .downcast_mut::<DepolarizingNoiseModel>()
                    .ok_or_else(|| {
                        PecosError::Input(
                            "run_with_specified_fault_histories requires DepolarizingNoiseModel"
                                .to_string(),
                        )
                    })?;
                depolarizing.set_sampled_fault_history_enabled(true);
                depolarizing.set_replay_fault_history(Some(history));
            }

            // Run the simulation for this shot
            let shot = worker_engine.run_shot()?;

            // Extract the applied fault history from the noise model after execution
            let applied_history = {
                let noise_model = worker_engine.quantum_system.noise_model_mut();
                let depolarizing = noise_model
                    .as_any_mut()
                    .downcast_mut::<DepolarizingNoiseModel>()
                    .ok_or_else(|| {
                        PecosError::Input(
                            "run_with_specified_fault_histories requires DepolarizingNoiseModel"
                                .to_string(),
                        )
                    })?;
                let applied = depolarizing
                    .sampled_fault_history()
                    .map_or_else(Vec::new, |faults| faults.to_vec());
                depolarizing.clear_replay_fault_history();
                applied
            };

            shots.push(shot);
            applied_histories.push(applied_history);
        }

        Ok(MonteCarloRunResult {
            results: ShotVec::from_measurements(&shots),
            fault_histories: applied_histories,
        })
    }

    // Simple function to run with a single fault history
    pub fn run_with_fault_history(
        &mut self,
        fault_history: &DepolarizingFaultHistory,
    ) -> Result<MonteCarloRunResult, PecosError> {
        self.run_with_fault_histories(vec![fault_history.clone()])
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
    /// - `Ok(ShotVec)`: The results from the simulation.
    /// - `Err(PecosError)`: If an error occurs during the configuration or simulation.
    ///
    /// # Errors
    /// This function will return a `PecosError` if:
    /// - There is an error during the execution of the simulation.
    pub fn run_with_engines(
        classical_engine: Box<dyn ClassicalControlEngine>,
        noise_model: Box<dyn NoiseModel>,
        quantum_engine: Box<dyn QuantumEngine>,
        num_shots: usize,
        num_workers: usize,
        seed: Option<u64>,
    ) -> Result<ShotVec, PecosError> {
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
    /// Returns a `PecosError` if any part of the simulation fails.
    pub fn run_with_hybrid_engine(
        hybrid_engine: HybridEngine,
        num_shots: usize,
        num_workers: usize,
        seed: Option<u64>,
    ) -> Result<ShotVec, PecosError> {
        let mut engine = MonteCarloEngineBuilder::new()
            .with_hybrid_engine(hybrid_engine)
            .build();

        if let Some(s) = seed {
            engine.set_seed(s);
        }

        Ok(engine.run_with_workers(num_shots, num_workers)?.results)
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
    /// Returns a `PecosError` if any part of the simulation fails.
    pub fn run_with_noise_model(
        classical_engine: Box<dyn ClassicalControlEngine>,
        noise_model: Box<dyn NoiseModel>,
        num_shots: usize,
        num_workers: usize,
        seed: Option<u64>,
    ) -> Result<ShotVec, PecosError> {
        // Create a hybrid engine with the state vector quantum engine
        let num_qubits = classical_engine.num_qubits();
        debug!(
            "MonteCarloEngine::run_with_noise_model: Creating StateVecEngine with {num_qubits} qubits"
        );
        let quantum_engine = Box::new(StateVecEngine::new(num_qubits));
        let mut hybrid_engine = HybridEngineBuilder::new()
            .with_classical_engine(classical_engine)
            .with_quantum_engine(quantum_engine)
            .with_noise_model(noise_model)
            .build();

        // Set seed if provided
        if let Some(s) = seed {
            hybrid_engine.set_seed(s);
        }

        Self::run_with_hybrid_engine(hybrid_engine, num_shots, num_workers, seed)
    }

    /// Static method to run a simulation with a classical engine, noise model, and max qubits.
    ///
    /// This method allows specifying the maximum number of qubits for the quantum engine,
    /// which is necessary for programs with dynamic qubit allocation in loops.
    ///
    /// # Parameters
    /// - `classical_engine`: The classical engine to use.
    /// - `noise_model`: The noise model to apply during simulation.
    /// - `num_qubits`: Number of qubits for the quantum engine (also sets allocation limit).
    /// - `num_shots`: The total number of circuit executions to perform.
    /// - `num_workers`: The number of worker threads to use for parallel execution.
    /// - `seed`: Optional seed for deterministic behavior.
    ///
    /// # Returns
    /// Aggregated results from all shots.
    ///
    /// # Errors
    /// Returns a `PecosError` if any part of the simulation fails.
    pub fn run_with_noise_model_and_max_qubits(
        classical_engine: Box<dyn ClassicalControlEngine>,
        noise_model: Box<dyn NoiseModel>,
        num_qubits: usize,
        num_shots: usize,
        num_workers: usize,
        seed: Option<u64>,
    ) -> Result<ShotVec, PecosError> {
        debug!(
            "MonteCarloEngine::run_with_noise_model_and_max_qubits: Creating StateVecEngine with {num_qubits} qubits"
        );
        let quantum_engine = Box::new(StateVecEngine::new(num_qubits));
        let mut hybrid_engine = HybridEngineBuilder::new()
            .with_classical_engine(classical_engine)
            .with_quantum_engine(quantum_engine)
            .with_noise_model(noise_model)
            .build();

        // Set seed if provided
        if let Some(s) = seed {
            hybrid_engine.set_seed(s);
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
    /// Returns a `PecosError` if any part of the simulation fails.
    pub fn run_with_config(
        config: &str,
        num_shots: usize,
        num_workers: usize,
        seed: Option<u64>,
    ) -> Result<ShotVec, PecosError> {
        // Parse the configuration string as a noise probability
        let p = config.parse::<f64>().map_err(|e| {
            PecosError::Input(format!("Failed to parse config string as float: {e}"))
        })?;

        // Create and seed a depolarizing noise model
        let mut noise_model = crate::noise::DepolarizingNoiseModel::new_uniform(p);

        if let Some(s) = seed {
            noise_model.set_seed(derive_seed(s, "noise_model"));
        }

        // Run simulation with external classical engine
        Self::run_with_noise_model(
            Box::new(ExternalClassicalEngine::new()),
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
            seed: self.seed,
            default_workers: self.default_workers,
            fault_history_enabled: self.fault_history_enabled,
        }
    }
}

/// Distributes shots evenly across workers with any remainder going to initial workers
///
/// # Returns
/// A vector containing the number of shots for each worker
fn distribute_shots(num_shots: usize, num_workers: usize) -> Vec<usize> {
    let base = num_shots / num_workers;
    let remainder = num_shots % num_workers;

    // Create vector with base shots per worker
    let mut result = vec![base; num_workers];

    // Add remainder shots to first 'remainder' workers
    result
        .iter_mut()
        .take(remainder)
        .for_each(|shots| *shots += 1);

    result
}

/// An external classical engine implementation used for testing and examples.
///
/// This implementation provides a basic classical engine that returns predetermined results
/// for demonstration and testing purposes.
#[derive(Debug, Clone)]
pub struct ExternalClassicalEngine {
    results: BTreeMap<String, i64>,
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
        let mut results = BTreeMap::new();
        results.insert("result".to_string(), 0);

        Self { results }
    }
}

impl Engine for ExternalClassicalEngine {
    type Input = ();
    type Output = Shot;

    fn process(&mut self, _input: Self::Input) -> Result<Self::Output, PecosError> {
        // For this stub implementation, just generate commands and return results
        let _message = self.generate_commands()?;
        self.get_results()
    }

    fn reset(&mut self) -> Result<(), PecosError> {
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

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        // Create a simple command that prepares and measures a qubit
        Ok(ByteMessage::builder().build())
    }

    fn handle_measurements(&mut self, _: ByteMessage) -> Result<(), PecosError> {
        // Store a random result
        Ok(())
    }

    fn get_results(&self) -> Result<Shot, PecosError> {
        // Create Shot with converted results
        let mut shot_result = Shot::default();

        // Add results to data field
        for (k, v) in &self.results {
            if *v >= 0 {
                // Handle positive values
                if let Ok(value) = u32::try_from(*v) {
                    shot_result.data.insert(k.clone(), Data::U32(value));
                } else if let Ok(value) = u64::try_from(*v) {
                    shot_result.data.insert(k.clone(), Data::U64(value));
                } else {
                    shot_result.data.insert(k.clone(), Data::I64(*v));
                }
            } else {
                // Handle negative values
                shot_result.data.insert(k.clone(), Data::I64(*v));
            }
        }

        Ok(shot_result)
    }

    fn compile(&self) -> Result<(), PecosError> {
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
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, (): ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        // Generate commands and return NeedsProcessing
        let commands = self.generate_commands()?;

        // If the message is empty and we're in compatibility mode, still return NeedsProcessing
        // to ensure MonteCarloEngine receives at least one batch
        let is_empty = commands.is_empty().unwrap_or(true);
        if is_empty {
            // Decide whether to return Complete or continue with an empty message
            // For empty messages, we'll check if it's the first batch (just after reset)
            let shot_result = self.get_results()?;
            Ok(EngineStage::Complete(shot_result))
        } else {
            Ok(EngineStage::NeedsProcessing(commands))
        }
    }

    fn continue_processing(
        &mut self,
        results: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        // Process the results and return Complete
        self.handle_measurements(results)?;
        let shot_result = self.get_results()?;
        Ok(EngineStage::Complete(shot_result))
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        Engine::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::byte_message::ByteMessageBuilder;

    #[derive(Clone)]
    struct FixedCircuitClassicalEngine {
        commands: ByteMessage,
    }

    impl FixedCircuitClassicalEngine {
        fn new() -> Self {
            let mut builder = ByteMessageBuilder::new();
            let _ = builder.for_quantum_operations();
            builder.pz(&[0]);
            builder.x(&[0]);
            builder.cx(&[(0, 1)]);
            builder.mz(&[1]);

            Self {
                commands: builder.build(),
            }
        }
    }

    impl Engine for FixedCircuitClassicalEngine {
        type Input = ();
        type Output = Shot;

        fn process(&mut self, _input: Self::Input) -> Result<Self::Output, PecosError> {
            Ok(Shot::default())
        }

        fn reset(&mut self) -> Result<(), PecosError> {
            Ok(())
        }
    }

    impl ClassicalEngine for FixedCircuitClassicalEngine {
        fn num_qubits(&self) -> usize {
            2
        }

        fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
            Ok(self.commands.clone())
        }

        fn handle_measurements(&mut self, _message: ByteMessage) -> Result<(), PecosError> {
            Ok(())
        }

        fn get_results(&self) -> Result<Shot, PecosError> {
            Ok(Shot::default())
        }

        fn compile(&self) -> Result<(), PecosError> {
            Ok(())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    impl ControlEngine for FixedCircuitClassicalEngine {
        type Input = ();
        type Output = Shot;
        type EngineInput = ByteMessage;
        type EngineOutput = ByteMessage;

        fn start(&mut self, (): ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
            Ok(EngineStage::NeedsProcessing(self.commands.clone()))
        }

        fn continue_processing(
            &mut self,
            _results: ByteMessage,
        ) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
            Ok(EngineStage::Complete(Shot::default()))
        }

        fn reset(&mut self) -> Result<(), PecosError> {
            Engine::reset(self)
        }
    }

    #[test]
    fn test_run_returns_results_and_fault_histories_when_enabled() {
        // Basic test to ensure that when the fault history is enabled
        // the API looks and works correctly.
        let classical_engine = Box::new(FixedCircuitClassicalEngine::new());
        let mut mc = MonteCarloEngine::builder()
            .with_classical_engine(classical_engine)
            .with_quantum_engine(Box::new(StateVecEngine::new(2)))
            .with_depolarizing_noise(1.0)
            .fault_history_enabled()
            .build();

        mc.set_seed(1234);

        let run = mc.run(16).expect("run should succeed");

        assert_eq!(run.results.shots.len(), 16);
        assert_eq!(run.fault_histories.len(), 16);

        for per_shot in &run.fault_histories {
            assert_eq!(per_shot.len(), 4);

            assert_eq!(per_shot[0].site_uid, 0);
            assert_eq!(per_shot[0].outcome_index, 1);
            assert_eq!(per_shot[0].outcome_label, "X");

            assert_eq!(per_shot[1].site_uid, 1);
            assert!((1..=3).contains(&per_shot[1].outcome_index));

            assert_eq!(per_shot[2].site_uid, 2);
            assert!((1..=15).contains(&per_shot[2].outcome_index));

            assert_eq!(per_shot[3].site_uid, 3);
            assert_eq!(per_shot[3].outcome_index, 1);
            assert_eq!(per_shot[3].outcome_label, "X");
        }
    }

    #[test]
    fn test_run_returns_empty_fault_histories_when_disabled() {
        let classical_engine = Box::new(FixedCircuitClassicalEngine::new());
        let mut mc = MonteCarloEngine::builder()
            .with_classical_engine(classical_engine)
            .with_quantum_engine(Box::new(StateVecEngine::new(2)))
            .with_depolarizing_noise(0.5)
            .build();

        mc.set_seed(1234);

        let run = mc.run(8).expect("run should succeed");

        assert_eq!(run.results.shots.len(), 8);
        assert!(run.fault_histories.is_empty());
    }

    #[test]
    fn test_run_fault_histories_are_seed_deterministic() {
        let build_engine = || {
            MonteCarloEngine::builder()
                .with_classical_engine(Box::new(FixedCircuitClassicalEngine::new()))
                .with_quantum_engine(Box::new(StateVecEngine::new(2)))
                .with_depolarizing_noise(0.5)
                .fault_history_enabled()
                .with_default_workers(2)
                .build()
        };

        let mut mc1 = build_engine();
        mc1.set_seed(1234);
        let run1 = mc1.run(20).expect("first run should succeed");

        let mut mc2 = build_engine();
        mc2.set_seed(1234);
        let run2 = mc2.run(20).expect("second run should succeed");

        assert_eq!(run1.fault_histories, run2.fault_histories);
        assert_eq!(run1.results.shots.len(), run2.results.shots.len());
    }

    #[test]
    fn test_run_with_fault_histories_replays_sampled_histories() {
        // Test that runs with specified fault histories produce the same
        // results as the original sampled run.
        let mut sampler_mc = MonteCarloEngine::builder()
            .with_classical_engine(Box::new(FixedCircuitClassicalEngine::new()))
            .with_quantum_engine(Box::new(StateVecEngine::new(2)))
            .with_depolarizing_noise(0.5)
            .fault_history_enabled()
            .build();

        sampler_mc.set_seed(0);

        let sampled_run = sampler_mc.run(12).expect("initial run should succeed");
        let sampled_histories = sampled_run.fault_histories.clone();

        assert_eq!(sampled_histories.len(), 12);

        // Replay in a zero-noise engine so only forced faults are applied.
        let mut replay_mc = MonteCarloEngine::builder()
            .with_classical_engine(Box::new(FixedCircuitClassicalEngine::new()))
            .with_quantum_engine(Box::new(StateVecEngine::new(2)))
            .with_depolarizing_noise(0.0)
            .fault_history_enabled()
            .build();

        replay_mc.set_seed(1);

        let replay_run = replay_mc
            .run_with_fault_histories(sampled_histories.clone())
            .expect("replay run should succeed");

        assert_eq!(replay_run.results.shots.len(), 12);
        assert_eq!(replay_run.fault_histories, sampled_histories);
    }

    #[test]
    fn test_run_with_fault_history_replays_sampled_history() {
        // Test that runs with specified fault history produce the same
        // results as the original sampled run.
        let mut sampler_mc = MonteCarloEngine::builder()
            .with_classical_engine(Box::new(FixedCircuitClassicalEngine::new()))
            .with_quantum_engine(Box::new(StateVecEngine::new(2)))
            .with_depolarizing_noise(0.5)
            .fault_history_enabled()
            .build();

        sampler_mc.set_seed(0);

        let sampled_run = sampler_mc.run(2).expect("initial run should succeed");
        let sampled_histories = sampled_run.fault_histories.clone();

        assert_eq!(sampled_histories.len(), 2);

        // Replay in a zero-noise engine so only forced faults are applied.
        let mut replay_mc = MonteCarloEngine::builder()
            .with_classical_engine(Box::new(FixedCircuitClassicalEngine::new()))
            .with_quantum_engine(Box::new(StateVecEngine::new(2)))
            .with_depolarizing_noise(0.0)
            .fault_history_enabled()
            .build();

        replay_mc.set_seed(1);

        let replay_run = replay_mc
            .run_with_fault_history(&sampled_histories[0].clone())
            .expect("replay run should succeed");

        assert_eq!(replay_run.results.shots.len(), 1);
        assert_eq!(replay_run.fault_histories[0], sampled_histories[0]);
    }
}
