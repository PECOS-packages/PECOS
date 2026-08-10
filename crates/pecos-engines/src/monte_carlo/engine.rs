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
use crate::noise::NoiseModel;
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
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
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
        let num_qubits = classical_engine.num_qubits();
        Self::builder()
            .with_classical_engine(classical_engine)
            .with_quantum_engine(Box::new(StateVecEngine::new(num_qubits)))
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
    pub fn run(&mut self, num_shots: usize) -> Result<ShotVec, PecosError> {
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
    ) -> Result<ShotVec, PecosError> {
        let (shots, _) = self.run_with_workers_report_seeds(num_shots, num_workers)?;
        Ok(shots)
    }

    /// Runs a Monte Carlo simulation and returns the seeds used by each worker.
    ///
    /// The seed report records the engine root seed, the base seed drawn for the run,
    /// and each worker's deterministic seed and shot count so the run can be reproduced
    /// or audited.
    /// This method runs the simulation with the specified number of shots and worker threads,
    /// overriding the default worker count configured during construction.
    ///
    /// # Arguments
    /// * `num_shots` - The number of shots to run
    /// * `num_workers` - The number of parallel worker threads to use
    ///
    /// # Returns
    /// A tuple containing the aggregated shot results and the seed report for the run.
    ///
    /// # Errors
    /// Returns a `PecosError` if any part of the simulation fails.
    ///
    /// # Panics
    /// - If `num_shots` is zero.
    /// - If `num_workers` is zero.
    pub fn run_with_workers_report_seeds(
        &mut self,
        num_shots: usize,
        num_workers: usize,
    ) -> Result<(ShotVec, SeedReport), PecosError> {
        assert!(num_shots > 0, "num_shots cannot be zero");
        assert!(num_workers > 0, "num_workers cannot be zero");

        debug!("Running Monte Carlo simulation: {num_shots} shots, {num_workers} workers");

        // Determine shots per worker and generate deterministic seeds
        let shots_per_worker = distribute_shots(num_shots, num_workers);
        let base_seed = self.rng.next_u64();

        // Create the seed report for this run
        let seed_report = SeedReport {
            root_seed: self.seed,
            base_seed,
            num_shots,
            num_workers,
            workers: (0..num_workers)
                .map(|worker_idx| {
                    let seed = derive_seed(base_seed, &format!("worker_{worker_idx}"));
                    WorkerSeedRecord {
                        worker_idx,
                        shots: shots_per_worker[worker_idx],
                        seed,
                    }
                })
                .collect(),
        };

        let shotvec = self.run_with_workers_from_seed_report(&seed_report)?;
        Ok((shotvec, seed_report))
    }

    /// Runs a Monte Carlo simulation using the worker configuration and seeds
    /// recorded in `seed_report`.
    ///
    /// The returned shots are ordered deterministically by worker and shot index.
    ///
    /// # Arguments
    /// * `seed_report` - The shot count, worker count, and worker seeds to replay
    ///
    /// # Returns
    /// The aggregated shot results.
    ///
    /// # Errors
    /// Returns `PecosError::Input` if the report contains fewer worker seed records
    /// than its configured worker count. Returns a `PecosError` if the worker pool
    /// cannot be created or any shot fails.
    ///
    /// # Panics
    /// Panics if the report specifies zero shots or workers, or if a worker record's
    /// index or shot count does not match the report configuration.
    pub fn run_with_workers_from_seed_report(
        &mut self,
        seed_report: &SeedReport,
    ) -> Result<ShotVec, PecosError> {
        // Import shot count, worker count, and all seeds from seed report.
        let num_shots = seed_report.num_shots;
        let num_workers = seed_report.num_workers;

        // check for invalid num_shots or num_workers
        assert!(num_shots > 0, "num_shots cannot be zero");
        assert!(num_workers > 0, "num_workers cannot be zero");
        if seed_report.workers.len() < num_workers {
            return Err(PecosError::Input(format!(
                "Seed report contains {} worker records, but num_workers is {num_workers}",
                seed_report.workers.len()
            )));
        }

        let shots_per_worker = distribute_shots(num_shots, num_workers);
        self.set_seed(seed_report.root_seed); // make sure to update root seed.

        debug!("Running Monte Carlo simulation: {num_shots} shots, {num_workers} workers");

        // Shared results collection
        let results_vec = Arc::new(Mutex::new(Vec::<(usize, usize, Shot)>::with_capacity(
            num_shots,
        )));

        // CRITICAL: Pre-create worker engines on the main thread before parallel execution.
        // This avoids potential deadlocks when worker threads try to clone engines
        // simultaneously, which can trigger concurrent library loading operations
        // that contend with each other or the dynamic linker.
        let worker_engines: Vec<_> = (0..num_workers)
            .map(|worker_idx| {
                let mut engine = self.hybrid_engine_template.clone();
                engine.set_seed(seed_report.workers[worker_idx].seed);
                (worker_idx, shots_per_worker[worker_idx], engine)
            })
            .collect();

        // Verify that worker indices and shots per worker match the seed report
        for worker_index in 0..num_workers {
            // check that worker indices agree
            assert!(
                seed_report.workers[worker_index].worker_idx == worker_engines[worker_index].0,
                ".."
            );
            // check that worker shot counts agree
            assert!(
                seed_report.workers[worker_index].shots == worker_engines[worker_index].1,
                ".."
            );
        }

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

        engine.run_with_workers(num_shots, num_workers)
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

/// Seed metadata for one Monte Carlo worker.
///
/// Each record captures the worker index, the number of shots assigned to that
/// worker, and the deterministic seed used to initialize its cloned engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkerSeedRecord {
    pub worker_idx: usize,
    pub shots: usize,
    pub seed: u64,
}

/// Reproducibility metadata captured for a Monte Carlo simulation run.
///
/// The report records the engine's root seed, the base seed drawn for this run,
/// the shot and worker configuration, and the deterministic seed assigned to
/// each worker.
///
/// # Note
/// `root_seed` alone cannot reproduce `base_seed` if the engine has a job history
/// before this one. The per-worker seeds are the pieces that ensure deterministic replay.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SeedReport {
    pub root_seed: u64,
    pub base_seed: u64,
    pub num_shots: usize,
    pub num_workers: usize,
    pub workers: Vec<WorkerSeedRecord>,
}

/// JSON serialization helpers for `SeedReport`.
///
/// Use these methods to save or reload reproducibility metadata when rerunning
/// or investigating specific worker seeds.
impl SeedReport {
    /// Deserializes a `SeedReport` from a JSON string.
    ///
    /// # Returns
    /// `SeedReport` imported from the JSON string.
    ///
    /// # Errors
    /// Returns `PecosError::Input` when the JSON is malformed or does not match
    /// the expected seed report schema.
    pub fn from_json_str(json: &str) -> Result<Self, PecosError> {
        serde_json::from_str(json)
            .map_err(|err| PecosError::Input(format!("Failed to parse seed report JSON: {err}")))
    }

    /// Reads and deserializes a `SeedReport` from a JSON file.
    ///
    /// # Returns
    /// `SeedReport` imported from a JSON file.
    ///
    /// # Errors
    /// Returns `PecosError::Input` when the file cannot be read or the file
    /// contents cannot be parsed as a seed report.
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self, PecosError> {
        let json = fs::read_to_string(path)
            .map_err(|err| PecosError::Input(format!("Failed to read seed report JSON: {err}")))?;
        Self::from_json_str(&json)
    }

    /// Serializes this `SeedReport` to the given JSON file.
    ///
    /// # Errors
    /// Returns `PecosError::Input` if the file cannot be created or written, or
    /// if the report cannot be serialized.
    pub fn to_json_file<P: AsRef<Path>>(&self, path: P) -> Result<(), PecosError> {
        let file = fs::File::create(path)
            .map_err(|err| PecosError::Input(format!("Failed to write seed report JSON: {err}")))?;
        let mut writer = BufWriter::new(file);

        serde_json::to_writer(&mut writer, self).map_err(|err| {
            PecosError::Input(format!("Failed to serialize seed report JSON: {err}"))
        })?;
        writer
            .flush()
            .map_err(|err| PecosError::Input(format!("Failed to write seed report JSON: {err}")))
    }
}

/// A minimal classical controller for testing and examples.
///
/// When driven through [`ControlEngine`] (for example, by [`HybridEngine`]), the
/// default controller has no quantum commands and completes each shot with
/// `result = 0`. Use [`Self::new_with_circuit`] to supply a fixed batch of quantum
/// commands that is emitted once per shot. Returned measurement outcomes are exposed
/// in message order as `result` for the first measurement and `result_1`, `result_2`,
/// and so on for subsequent measurements.
///
/// The controller reports a fixed capacity of two qubits, so configured circuits
/// should target only qubits 0 and 1. Calling [`Engine::process`] directly does not
/// provide a quantum backend; it returns the current result values without executing
/// the configured circuit.
#[derive(Clone)]
pub struct ExternalClassicalEngine {
    results: BTreeMap<String, i64>,
    circuit: ByteMessage,
}

impl std::fmt::Debug for ExternalClassicalEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalClassicalEngine")
            .field("results", &self.results)
            .field("circuit_bytes", &self.circuit.as_bytes().len())
            .finish()
    }
}

impl Default for ExternalClassicalEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ExternalClassicalEngine {
    /// Create a controller with an empty circuit and `result` initialized to zero.
    ///
    /// Because the circuit is empty, the controller completes without invoking a
    /// quantum engine.
    #[must_use]
    pub fn new() -> Self {
        Self::new_with_circuit(ByteMessage::create_empty())
    }

    /// Create a controller that emits `circuit` once per shot.
    ///
    /// Measurement outcomes returned by the quantum engine are recorded in message order.
    /// The first is named `result`; later outcomes are named `result_1`, `result_2`,
    /// and so on. Calling `reset` retains the existing result fields and sets every
    /// value back to zero before the next shot.
    ///
    /// This test controller reports two available qubits, so `circuit` should target
    /// only qubits 0 and 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_engines::ByteMessage;
    /// use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
    ///
    /// let circuit = ByteMessage::quantum_operations_builder()
    ///     .pz(&[0])
    ///     .h(&[0])
    ///     .mz(&[0])
    ///     .build();
    /// let controller = ExternalClassicalEngine::new_with_circuit(circuit);
    /// ```
    #[must_use]
    pub fn new_with_circuit(circuit: ByteMessage) -> Self {
        // Initialize with a default results map
        let mut results = BTreeMap::new();
        results.insert("result".to_string(), 0);

        Self { results, circuit }
    }
}

impl Engine for ExternalClassicalEngine {
    type Input = ();
    type Output = Shot;

    fn process(&mut self, _input: Self::Input) -> Result<Self::Output, PecosError> {
        // Direct processing has no quantum backend, so retrieve the configured batch
        // and return the controller's current results without executing it.
        let _message = self.generate_commands()?;
        self.get_results()
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Retain every result field while resetting its value to zero.
        for value in self.results.values_mut() {
            *value = 0;
        }

        Ok(())
    }
}

impl ClassicalEngine for ExternalClassicalEngine {
    fn num_qubits(&self) -> usize {
        // This fixed-circuit test controller exposes a hard-coded two-qubit capacity.
        2
    }

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        Ok(self.circuit.clone())
    }

    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError> {
        for (index, outcome) in message.outcomes()?.into_iter().enumerate() {
            let name = if index == 0 {
                "result".to_string()
            } else {
                format!("result_{index}")
            };
            self.results.insert(name, i64::from(outcome));
        }
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
        // The fixed ByteMessage circuit needs no compilation.
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
        // Retrieve the configured command batch for this shot.
        let commands = self.generate_commands()?;

        // Empty circuits complete immediately with the current result values.
        let is_empty = commands.is_empty().unwrap_or(true);
        if is_empty {
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
