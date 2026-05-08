//! Simulation builder using data-oriented design
//!
//! This module provides a builder pattern that collects all simulation configuration
//! data and constructs a `MonteCarloEngine` for the "build once, run multiple times" pattern.

use crate::ClassicalControlEngine;
use crate::engine_builder::ClassicalControlEngineBuilder;
use crate::hybrid::HybridEngineBuilder;
use crate::monte_carlo::builder::MonteCarloEngineBuilder;
use crate::monte_carlo::engine::MonteCarloEngine;
use crate::noise::{IntoNoiseModel, NoiseModel};
use crate::quantum_engine_builder::{IntoQuantumEngineBuilder, QuantumEngineBuilder};
use crate::shot_results::ShotVec;
use pecos_core::errors::PecosError;

/// Configuration for simulations
#[derive(Debug, Clone)]
pub struct SimConfig {
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Number of worker threads
    pub workers: usize,
    /// Verbose output
    pub verbose: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            seed: None,
            workers: 1,
            verbose: false,
        }
    }
}

/// Trait for building boxed classical control engines
///
/// This internal trait allows storing different engine builders uniformly as trait objects.
trait BoxedClassicalEngineBuilder: Send {
    fn build_boxed(self: Box<Self>) -> Result<Box<dyn ClassicalControlEngine>, PecosError>;
}

/// Trait for building boxed quantum engines
trait BoxedQuantumEngineBuilder: Send {
    fn build_boxed(self: Box<Self>) -> Result<Box<dyn crate::quantum::QuantumEngine>, PecosError>;
    fn set_qubits_if_needed(&mut self, num_qubits: usize);
}

/// Trait for building boxed noise models
trait BoxedNoiseModelBuilder: Send {
    fn build_boxed(self: Box<Self>) -> Box<dyn NoiseModel>;
}

/// Wrapper that converts any `ClassicalControlEngineBuilder` to `BoxedClassicalEngineBuilder`
struct ClassicalBuilderWrapper<B: ClassicalControlEngineBuilder> {
    builder: B,
}

impl<B> BoxedClassicalEngineBuilder for ClassicalBuilderWrapper<B>
where
    B: ClassicalControlEngineBuilder + Send,
    B::Engine: 'static,
{
    fn build_boxed(self: Box<Self>) -> Result<Box<dyn ClassicalControlEngine>, PecosError> {
        Ok(Box::new(self.builder.build()?))
    }
}

/// Wrapper for quantum engine builders
struct QuantumBuilderWrapper<B: QuantumEngineBuilder> {
    builder: B,
}

impl<B> BoxedQuantumEngineBuilder for QuantumBuilderWrapper<B>
where
    B: QuantumEngineBuilder + Send + 'static,
{
    fn build_boxed(
        mut self: Box<Self>,
    ) -> Result<Box<dyn crate::quantum::QuantumEngine>, PecosError> {
        self.builder.build()
    }

    fn set_qubits_if_needed(&mut self, num_qubits: usize) {
        self.builder.set_qubits_if_needed(num_qubits);
    }
}

/// Wrapper for noise model builders
struct NoiseModelWrapper<N: IntoNoiseModel> {
    noise: N,
}

impl<N> BoxedNoiseModelBuilder for NoiseModelWrapper<N>
where
    N: IntoNoiseModel + Send + 'static,
{
    fn build_boxed(self: Box<Self>) -> Box<dyn NoiseModel> {
        self.noise.into_noise_model()
    }
}

/// A simulation builder using data-oriented design principles
///
/// This builder collects all simulation configuration data and builds a `MonteCarloEngine`
/// that can be run multiple times. It treats all components (classical engine, quantum engine,
/// noise model) equally and validates everything at build time.
///
/// # Design Philosophy
///
/// - **Data Collection**: The builder is just a data collector - POD-like configuration
/// - **Ownership**: The builder owns all its data and consumes itself on build
/// - **Validation**: All validation happens at build time, not during collection
/// - **Flexibility**: Supports runtime component selection via trait objects
///
/// # Example
///
/// ```rust
/// # use pecos_engines::{sim_builder, ClassicalControlEngineBuilder};
/// # use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
/// # struct MyEngineBuilder;
/// # impl ClassicalControlEngineBuilder for MyEngineBuilder {
/// #     type Engine = ExternalClassicalEngine;
/// #     fn build(self) -> Result<Self::Engine, pecos_core::errors::PecosError> {
/// #         Ok(ExternalClassicalEngine::new())
/// #     }
/// # }
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Pattern 1: Direct run
/// let results = sim_builder()
///     .classical(MyEngineBuilder)
///     .seed(42)
///     .run(100)?;
///
/// // Pattern 2: Build once, run multiple times
/// let mut engine = sim_builder()
///     .classical(MyEngineBuilder)
///     .seed(42)
///     .build()?;
///
/// let results1 = engine.run(100)?;  // 100 shots
/// let results2 = engine.run_with_workers(200, 4)?;  // 200 shots, 4 workers
/// # Ok(())
/// # }
/// ```
pub struct SimBuilder {
    // Store builders as trait objects for runtime flexibility
    classical_builder: Option<Box<dyn BoxedClassicalEngineBuilder>>,
    quantum_builder: Option<Box<dyn BoxedQuantumEngineBuilder>>,
    noise_builder: Option<Box<dyn BoxedNoiseModelBuilder>>,
    config: SimConfig,
    explicit_num_qubits: Option<usize>,
}

impl SimBuilder {
    /// Create a new unified simulation builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            classical_builder: None,
            quantum_builder: None,
            noise_builder: None,
            config: SimConfig::default(),
            explicit_num_qubits: None,
        }
    }

    /// Set the classical control engine builder
    #[must_use]
    pub fn classical<B>(mut self, engine_builder: B) -> Self
    where
        B: ClassicalControlEngineBuilder + Send + 'static,
        B::Engine: 'static,
    {
        self.classical_builder = Some(Box::new(ClassicalBuilderWrapper {
            builder: engine_builder,
        }));
        self
    }

    /// Set the random seed
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Set the number of worker threads
    #[must_use]
    pub fn workers(mut self, workers: usize) -> Self {
        self.config.workers = workers;
        self
    }

    /// Use automatic worker count based on available CPUs
    #[must_use]
    pub fn auto_workers(mut self) -> Self {
        self.config.workers =
            std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
        self
    }

    /// Enable verbose output
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Set the noise model
    #[must_use]
    pub fn noise<N>(mut self, noise: N) -> Self
    where
        N: IntoNoiseModel + Send + 'static,
    {
        self.noise_builder = Some(Box::new(NoiseModelWrapper { noise }));
        self
    }

    /// Set the quantum engine
    #[must_use]
    pub fn quantum<Q>(mut self, quantum_builder: Q) -> Self
    where
        Q: IntoQuantumEngineBuilder + 'static,
        Q::Builder: Send + 'static,
    {
        let builder = quantum_builder.into_quantum_engine_builder();
        self.quantum_builder = Some(Box::new(QuantumBuilderWrapper { builder }));
        self
    }

    /// Alias for `quantum` method
    #[must_use]
    pub fn quantum_engine<Q>(self, quantum_builder: Q) -> Self
    where
        Q: IntoQuantumEngineBuilder + 'static,
        Q::Builder: Send + 'static,
    {
        self.quantum(quantum_builder)
    }

    /// Set the number of qubits explicitly
    ///
    /// This is useful when the engine needs to know the number of qubits
    /// before program execution.
    #[must_use]
    pub fn qubits(mut self, num_qubits: usize) -> Self {
        self.explicit_num_qubits = Some(num_qubits);
        self
    }

    /// Build the `MonteCarloEngine`
    ///
    /// This consumes the builder and all its data to create a `MonteCarloEngine`
    /// that can be run multiple times.
    ///
    /// # Errors
    ///
    /// Returns an error if required components are missing:
    /// - Classical engine (always required)
    /// - Number of qubits (if not provided by engine)
    pub fn build(self) -> Result<MonteCarloEngine, PecosError> {
        use crate::noise::PassThroughNoiseModel;
        use crate::quantum::SparseStabEngine;

        // Build classical engine (required)
        let classical_engine = match self.classical_builder {
            Some(builder) => builder.build_boxed()?,
            None => {
                return Err(PecosError::Input(
                    "Classical control engine not set. Use .classical() to set one.".to_string(),
                ));
            }
        };

        // Determine number of qubits
        let num_qubits = self
            .explicit_num_qubits
            .or_else(|| Some(classical_engine.num_qubits()))
            .ok_or_else(|| {
                PecosError::Input(
                    "Number of qubits not specified and cannot be inferred from engine".to_string(),
                )
            })?;

        // Build quantum engine (require explicit qubit specification)
        let quantum_engine = if let Some(mut builder) = self.quantum_builder {
            // Set qubits on the quantum engine builder if explicitly specified
            builder.set_qubits_if_needed(num_qubits);
            builder.build_boxed()?
        } else {
            // Default: sparse stabilizer
            Box::new(SparseStabEngine::new(num_qubits))
        };

        // Build noise model (with default if not set)
        let noise_model = if let Some(builder) = self.noise_builder {
            builder.build_boxed()
        } else {
            // Default: no noise
            Box::new(PassThroughNoiseModel::new())
        };

        // Build HybridEngine
        let hybrid_engine = HybridEngineBuilder::new()
            .with_classical_engine(classical_engine)
            .with_quantum_engine(quantum_engine)
            .with_noise_model(noise_model)
            .build();

        // Build MonteCarloEngine
        let mut monte_carlo = MonteCarloEngineBuilder::new()
            .with_hybrid_engine(hybrid_engine)
            .with_default_workers(self.config.workers)
            .build();

        // Set seed if configured
        if let Some(seed) = self.config.seed {
            monte_carlo.set_seed(seed);
        }

        Ok(monte_carlo)
    }

    /// Build and run the simulation
    ///
    /// This is a convenience method that builds and runs in one step.
    /// Uses the configured number of workers (default: 1).
    ///
    /// # Errors
    ///
    /// Returns an error if the simulation cannot be built or if execution fails
    pub fn run(self, shots: usize) -> Result<ShotVec, PecosError> {
        let mut engine = self.build()?;
        engine.run(shots)
    }
}

impl Default for SimBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a new simulation builder without a classical engine
///
/// This function returns a builder that requires setting the classical engine
/// via the `.classical()` method, providing a flexible API for simulation setup.
///
/// The builder supports two usage patterns:
/// - Direct: `.run(shots)` - builds and runs in one step
/// - Reusable: `.build()` then `.run(shots)` multiple times
///
/// # Example
///
/// ```rust
/// # use pecos_engines::{sim_builder, ClassicalControlEngineBuilder};
/// # use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
/// # struct MyEngineBuilder;
/// # impl ClassicalControlEngineBuilder for MyEngineBuilder {
/// #     type Engine = ExternalClassicalEngine;
/// #     fn build(self) -> Result<Self::Engine, pecos_core::errors::PecosError> {
/// #         Ok(ExternalClassicalEngine::new())
/// #     }
/// # }
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use pecos_engines::{sim_builder, sparse_stab, DepolarizingNoise};
///
/// // Direct usage
/// let results = sim_builder()
///     .classical(MyEngineBuilder)
///     .quantum(sparse_stab())
///     .noise(DepolarizingNoise { p: 0.01 })
///     .seed(42)
///     .run(100)?;
///
/// // Reusable pattern
/// let mut sim = sim_builder()
///     .classical(MyEngineBuilder)
///     .quantum(sparse_stab())
///     .build()?;
///
/// let batch1 = sim.run(100)?;  // 100 shots
/// let batch2 = sim.run_with_workers(200, 4)?;  // 200 shots, 4 workers
/// # Ok(())
/// # }
/// ```
#[must_use]
pub fn sim_builder() -> SimBuilder {
    SimBuilder::new()
}

/// Create a simulation builder from any `SimInput`
///
/// This function accepts any type that can be converted into a `SimBuilder`,
/// including engine builders, programs, or other custom types implementing `SimInput`.
///
/// # Example
/// ```rust
/// # use pecos_engines::{sim, ClassicalControlEngineBuilder};
/// # use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
/// # struct MyEngineBuilder;
/// # impl ClassicalControlEngineBuilder for MyEngineBuilder {
/// #     type Engine = ExternalClassicalEngine;
/// #     fn build(self) -> Result<Self::Engine, pecos_core::errors::PecosError> {
/// #         Ok(ExternalClassicalEngine::new())
/// #     }
/// # }
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // With an engine builder
/// let results = sim(MyEngineBuilder).seed(42).run(10)?;
/// assert_eq!(results.len(), 10);
/// # Ok(())
/// # }
/// ```
pub fn sim<I: crate::engine_builder::SimInput>(input: I) -> SimBuilder {
    input.into_sim_builder()
}

// ============================================================================
// Noise Model Structs for Ergonomic API
// ============================================================================

/// Pass-through noise configuration (no noise)
#[derive(Debug, Clone, Copy)]
pub struct PassThroughNoise;

/// Depolarizing noise configuration
#[derive(Debug, Clone, Copy)]
pub struct DepolarizingNoise {
    /// The depolarizing probability
    pub p: f64,
}

/// Biased depolarizing noise configuration
#[derive(Debug, Clone, Copy)]
pub struct BiasedDepolarizingNoise {
    /// The depolarizing probability
    pub p: f64,
}

// ============================================================================
// IntoNoiseModel implementations for convenience structs
// ============================================================================

impl crate::noise::IntoNoiseModel for PassThroughNoise {
    fn into_noise_model(self) -> Box<dyn crate::noise::NoiseModel> {
        Box::new(crate::noise::PassThroughNoiseModel::new())
    }
}

impl crate::noise::IntoNoiseModel for DepolarizingNoise {
    fn into_noise_model(self) -> Box<dyn crate::noise::NoiseModel> {
        Box::new(crate::noise::DepolarizingNoiseModel::new_uniform(self.p))
    }
}

impl crate::noise::IntoNoiseModel for BiasedDepolarizingNoise {
    fn into_noise_model(self) -> Box<dyn crate::noise::NoiseModel> {
        Box::new(crate::noise::BiasedDepolarizingNoiseModel::new_uniform(
            self.p,
        ))
    }
}

/// Convert `ShotVec` to columnar format
///
/// This is a helper for engines that need to return `HashMap`<String, Vec<i64>>
///
/// # Panics
///
/// Panics if a register name exists in the first shot but not in subsequent shots
#[must_use]
pub fn shots_to_columnar(
    shots: &crate::shot_results::ShotVec,
) -> std::collections::BTreeMap<String, Vec<i64>> {
    use std::collections::BTreeMap;

    let mut columnar = BTreeMap::new();

    if shots.is_empty() {
        return columnar;
    }

    // Get all register names from first shot
    let register_names: Vec<String> = if let Some(first_shot) = shots.shots.first() {
        first_shot.data.keys().cloned().collect()
    } else {
        return columnar;
    };

    // Initialize columns
    for name in &register_names {
        columnar.insert(name.clone(), Vec::with_capacity(shots.len()));
    }

    // Fill columns
    for shot in &shots.shots {
        for name in &register_names {
            if let Some(col) = columnar.get_mut(name) {
                if let Some(data) = shot.data.get(name) {
                    use crate::shot_results::Data;
                    let value = match data {
                        Data::U32(v) => i64::from(*v),
                        Data::I64(v) => *v,
                        #[allow(clippy::cast_possible_truncation)]
                        Data::F64(v) => *v as i64,
                        Data::Bool(v) => i64::from(*v),
                        _ => 0,
                    };
                    col.push(value);
                } else {
                    col.push(0);
                }
            }
        }
    }

    // If no named registers, create a default "_result" register
    if columnar.is_empty() {
        let values: Vec<i64> = shots.shots.iter().map(|_| 0).collect();
        columnar.insert("_result".to_string(), values);
    }

    columnar
}
