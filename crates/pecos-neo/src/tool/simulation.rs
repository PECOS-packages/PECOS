// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Simulation builder and handle for the Tool architecture.
//!
//! This module provides:
//! - [`sim_neo()`] - Universal entry point accepting any program type
//! - [`sim_neo_builder()`] - Empty builder for advanced configuration
//! - [`SimNeoBuilder`] - Builder for configuring simulation tools
//! - [`Simulation`] - Reusable simulation handle
//! - [`SimNeoInput`] - Trait for types that can be simulated
//!
//! # Usage Patterns
//!
//! The `sim_neo()` function accepts any program type, similar to `sim()`:
//!
//! ## Static Circuits
//!
//! For circuits without mid-circuit classical control:
//!
//! ```ignore
//! use pecos_neo::tool::sim_neo;
//! use pecos_neo::prelude::*;
//!
//! let circuit = CommandBuilder::new()
//!     .prep(0).h(0).measure(0)
//!     .build();
//!
//! let results = sim_neo(circuit)
//!     .depolarizing(0.01)
//!     .shots(1000)
//!     .seed(42)
//!     .build()
//!     .run();
//! ```
//!
//! ## QASM Programs
//!
//! For QASM programs with classical control flow:
//!
//! ```ignore
//! use pecos_neo::tool::sim_neo;
//! use pecos_qasm::qasm_engine;
//!
//! let qasm = r#"
//!     OPENQASM 2.0;
//!     include "qelib1.inc";
//!     qreg q[2];
//!     creg c[2];
//!     h q[0];
//!     measure q[0] -> c[0];
//!     if (c[0] == 1) x q[1];  // Conditional!
//!     measure q[1] -> c[1];
//! "#;
//!
//! // Pass engine builder directly to sim_neo()
//! let results = sim_neo(qasm_engine().qasm(qasm))
//!     .depolarizing(0.01)
//!     .shots(1000)
//!     .seed(42)
//!     .build()
//!     .run();
//! ```
//!
//! ## Other Program Types
//!
//! Any `ClassicalControlEngineBuilder` works with `sim_neo()`:
//!
//! ```ignore
//! use pecos_neo::tool::sim_neo;
//! use pecos_hugr::hugr_engine;
//! use pecos_qis::qis_engine;
//!
//! // HUGR programs
//! let results = sim_neo(hugr_engine().hugr(&hugr_module))
//!     .shots(1000)
//!     .build()
//!     .run();
//!
//! // QIS programs
//! let results = sim_neo(qis_engine().qis(&qis_program))
//!     .shots(1000)
//!     .build()
//!     .run();
//! ```
//!
//! ## Reusable Simulations
//!
//! Build once, run multiple times:
//!
//! ```ignore
//! let mut sim = sim_neo(circuit)
//!     .shots(1000)
//!     .build();
//!
//! let results1 = sim.run();
//! let results2 = sim.seed(123).run();  // Different seed
//! let results3 = sim.shots(5000).run(); // More shots
//! ```

use crate::command::CommandQueue;
use crate::noise::ComposableNoiseModel;
use crate::outcome::MeasurementOutcomes;
use crate::program::{CommandSource, ProgramRunner, StaticProgram};
use crate::runner::ShotRunner;
use pecos_core::rng::rng_manageable::derive_seed;
use pecos_qsim::{CliffordGateable, SparseStab, StateVec};
use rayon::prelude::*;

use super::resource::Resources;
use super::{Plugin, Stage, Tool};

// ============================================================================
// Quantum Backend Builders (builder-of-builders pattern)
// ============================================================================

/// Configuration for a quantum backend, stored as data in the builder.
///
/// This enum represents the choice of quantum simulator. The actual simulator
/// is constructed at build time, following the builder-of-builders pattern.
#[derive(Debug, Clone, Default)]
pub enum QuantumBackend {
    /// Sparse stabilizer simulator (default).
    ///
    /// Efficient for Clifford circuits and QEC simulations.
    /// Only supports Clifford gates (H, S, CNOT, CZ, etc.).
    #[default]
    SparseStab,

    /// State vector simulator.
    ///
    /// Supports arbitrary gates including non-Clifford (T, rotations).
    /// Memory scales as 2^n for n qubits.
    StateVec,
}

/// Builder for sparse stabilizer backend configuration.
///
/// Currently a simple marker type; future versions may add configuration options.
#[derive(Debug, Clone, Default)]
pub struct SparseStabBuilder;

impl SparseStabBuilder {
    /// Create a new sparse stabilizer builder.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl From<SparseStabBuilder> for QuantumBackend {
    fn from(_: SparseStabBuilder) -> Self {
        QuantumBackend::SparseStab
    }
}

/// Builder for state vector backend configuration.
///
/// Currently a simple marker type; future versions may add configuration options
/// like precision (f32 vs f64) or sparse vs dense representation.
#[derive(Debug, Clone, Default)]
pub struct StateVecBuilder;

impl StateVecBuilder {
    /// Create a new state vector builder.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl From<StateVecBuilder> for QuantumBackend {
    fn from(_: StateVecBuilder) -> Self {
        QuantumBackend::StateVec
    }
}

/// Create a sparse stabilizer backend builder.
///
/// The sparse stabilizer is the default backend, efficient for Clifford circuits
/// and quantum error correction simulations.
///
/// # Example
///
/// ```ignore
/// use pecos_neo::tool::{sim_neo, sparse_stab};
///
/// let results = sim_neo(circuit)
///     .quantum(sparse_stab())
///     .shots(1000)
///     .run();
/// ```
#[must_use]
pub fn sparse_stab() -> SparseStabBuilder {
    SparseStabBuilder::new()
}

/// Create a state vector backend builder.
///
/// The state vector simulator supports arbitrary gates including non-Clifford
/// operations like T gates and arbitrary rotations.
///
/// # Example
///
/// ```ignore
/// use pecos_neo::tool::{sim_neo, state_vector};
///
/// let results = sim_neo(circuit)
///     .quantum(state_vector())
///     .shots(1000)
///     .run();
/// ```
#[must_use]
pub fn state_vector() -> StateVecBuilder {
    StateVecBuilder::new()
}

// ============================================================================
// SimNeoInput Trait
// ============================================================================

/// Trait for types that can be used as input to [`sim_neo()`].
///
/// This trait enables `sim_neo()` to accept various program types:
/// - Static circuits (`CommandQueue`)
/// - Classical engine builders (QASM, HUGR, PHIR, QIS, etc.)
///
/// # Implementing for Custom Types
///
/// To make a custom type work with `sim_neo()`, implement this trait:
///
/// ```ignore
/// impl SimNeoInput for MyProgramType {
///     fn into_sim_neo_builder(self) -> SimNeoBuilder {
///         // Convert to SimNeoBuilder
///         sim_neo_builder().classical(my_engine_builder(self))
///     }
/// }
/// ```
pub trait SimNeoInput {
    /// Convert this input into a `SimNeoBuilder`.
    fn into_sim_neo_builder(self) -> SimNeoBuilder;
}

/// Implementation for `CommandQueue` (static circuits).
impl SimNeoInput for CommandQueue {
    fn into_sim_neo_builder(self) -> SimNeoBuilder {
        SimNeoBuilder::with_circuit(self)
    }
}

/// Implementation for `TickCircuit`.
impl SimNeoInput for pecos_quantum::TickCircuit {
    fn into_sim_neo_builder(self) -> SimNeoBuilder {
        SimNeoBuilder::with_circuit(self.into())
    }
}

/// Implementation for `&TickCircuit`.
impl SimNeoInput for &pecos_quantum::TickCircuit {
    fn into_sim_neo_builder(self) -> SimNeoBuilder {
        SimNeoBuilder::with_circuit(self.into())
    }
}

/// Implementation for `DagCircuit`.
impl SimNeoInput for pecos_quantum::DagCircuit {
    fn into_sim_neo_builder(self) -> SimNeoBuilder {
        SimNeoBuilder::with_circuit(self.into())
    }
}

/// Implementation for `&DagCircuit`.
impl SimNeoInput for &pecos_quantum::DagCircuit {
    fn into_sim_neo_builder(self) -> SimNeoBuilder {
        SimNeoBuilder::with_circuit(self.into())
    }
}

/// Implementation for `&str` (program source code like QASM).
///
/// When passing a string, use `.classical(engine)` to specify how to interpret it:
///
/// ```ignore
/// sim_neo(qasm_code)
///     .classical(qasm_engine())
///     .shots(1000)
///     .build()
///     .run();
/// ```
#[cfg(feature = "engines-adapter")]
impl SimNeoInput for &str {
    fn into_sim_neo_builder(self) -> SimNeoBuilder {
        SimNeoBuilder::with_program_source(self.to_string())
    }
}

/// Implementation for `String` (program source code).
#[cfg(feature = "engines-adapter")]
impl SimNeoInput for String {
    fn into_sim_neo_builder(self) -> SimNeoBuilder {
        SimNeoBuilder::with_program_source(self)
    }
}

/// Implementation for `Qasm` program type.
///
/// Use `.auto()` to automatically select the QASM engine, or
/// `.classical(engine)` for explicit control:
///
/// ```ignore
/// // Auto mode - uses qasm_engine() automatically
/// sim_neo(Qasm::from_string(qasm_code))
///     .auto()
///     .shots(1000)
///     .build()
///     .run();
///
/// // Explicit mode
/// sim_neo(Qasm::from_string(qasm_code))
///     .classical(qasm_engine())
///     .shots(1000)
///     .build()
///     .run();
/// ```
#[cfg(feature = "engines-adapter")]
impl SimNeoInput for pecos_programs::Qasm {
    fn into_sim_neo_builder(self) -> SimNeoBuilder {
        SimNeoBuilder::with_typed_program(TypedProgram::Qasm(self))
    }
}

/// Implementation for the unified `Program` enum.
///
/// Use `.auto()` to automatically select the appropriate engine based on
/// the program type:
///
/// ```ignore
/// sim_neo(Program::Qasm(qasm))
///     .auto()
///     .shots(1000)
///     .build()
///     .run();
/// ```
#[cfg(feature = "engines-adapter")]
impl SimNeoInput for pecos_programs::Program {
    fn into_sim_neo_builder(self) -> SimNeoBuilder {
        let typed = match self {
            pecos_programs::Program::Qasm(p) => TypedProgram::Qasm(p),
            // Add other program types as support is added
            _ => TypedProgram::Unsupported(format!("{}", self.program_type())),
        };
        SimNeoBuilder::with_typed_program(typed)
    }
}

// ============================================================================
// Resources
// ============================================================================

/// The circuit to execute.
#[derive(Clone)]
pub struct Circuit(pub CommandQueue);

/// Simulation configuration.
#[derive(Debug, Clone)]
pub struct SimConfig {
    /// Number of shots to run.
    pub shots: usize,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            shots: 1000,
            seed: None,
        }
    }
}

/// Orchestration strategy for simulation execution.
///
/// This enum defines how shots are executed. Different strategies offer
/// trade-offs between simplicity, parallelism, and specialized sampling.
///
/// Stored as data in the builder, the actual execution is set up at run time.
#[derive(Debug, Clone)]
pub enum Orchestrator {
    /// Sequential execution via Tool (default for simplicity).
    Sequential,

    /// Parallel Monte Carlo execution using rayon.
    ///
    /// Each worker runs a batch of shots independently with deterministic seeding.
    /// Best for noiseless simulations or when noise model is cheap to clone.
    MonteCarlo {
        /// Number of parallel workers.
        workers: usize,
    },

    /// Importance sampling for rare event estimation.
    ///
    /// Biases sampling toward rare events and reweights results.
    /// Use when estimating probabilities of rare outcomes.
    ImportanceSampling {
        /// Boost factor for rare events.
        boost: f64,
        /// Number of parallel workers.
        workers: usize,
    },
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::Sequential
    }
}

impl Orchestrator {
    /// Create a Monte Carlo orchestrator with specified workers.
    #[must_use]
    pub fn monte_carlo(workers: usize) -> Self {
        Self::MonteCarlo { workers }
    }

    /// Create a Monte Carlo orchestrator with auto-detected worker count.
    #[must_use]
    pub fn monte_carlo_auto() -> Self {
        let workers = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        Self::MonteCarlo { workers }
    }

    /// Create an importance sampling orchestrator.
    #[must_use]
    pub fn importance_sampling(boost: f64, workers: usize) -> Self {
        Self::ImportanceSampling { boost, workers }
    }
}

/// Accumulated simulation results.
#[derive(Debug, Clone, Default)]
pub struct SimulationResults {
    /// Per-shot measurement outcomes.
    pub outcomes: Vec<MeasurementOutcomes>,
}

impl SimulationResults {
    /// Create new empty results.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of shots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }

    /// Clear results for reuse.
    pub fn clear(&mut self) {
        self.outcomes.clear();
    }
}

/// Current shot state during execution.
pub struct ShotState {
    /// The shot runner (with noise already configured).
    pub runner: ShotRunner<SparseStab>,
    /// Current shot index.
    pub shot_index: usize,
    /// Cached circuit for execution.
    pub circuit: CommandQueue,
}

/// Wrapper for noise model resource.
pub struct NoiseResource(pub ComposableNoiseModel);

// ============================================================================
// Classical Engine Support
// ============================================================================

/// Trait for type-erased engine building.
///
/// This allows storing different engine builder types uniformly.
#[cfg(feature = "engines-adapter")]
pub trait BoxedEngineBuilder: Send + Sync {
    /// Build the classical engine and wrap it in an adapter.
    ///
    /// # Errors
    ///
    /// Returns an error if the engine cannot be built.
    fn build_adapter(
        self: Box<Self>,
    ) -> Result<Box<dyn CommandSource + Send + Sync>, pecos_core::errors::PecosError>;

    /// Get the number of qubits (if known before building).
    ///
    /// This is optional - most builders don't know the qubit count until built.
    /// Use `.qubits(n)` on `SimNeoBuilder` to set explicitly if needed.
    #[allow(dead_code)]
    fn num_qubits_hint(&self) -> Option<usize>;
}

/// Wrapper for concrete classical engine builders.
#[cfg(feature = "engines-adapter")]
struct EngineBuilderWrapper<B>
where
    B: pecos_engines::ClassicalControlEngineBuilder + Send + Sync,
    B::Engine: 'static,
{
    builder: B,
}

#[cfg(feature = "engines-adapter")]
impl<B> BoxedEngineBuilder for EngineBuilderWrapper<B>
where
    B: pecos_engines::ClassicalControlEngineBuilder + Send + Sync,
    B::Engine: 'static,
{
    fn build_adapter(
        self: Box<Self>,
    ) -> Result<Box<dyn CommandSource + Send + Sync>, pecos_core::errors::PecosError> {
        let engine = self.builder.build()?;
        Ok(Box::new(crate::adapter::ClassicalEngineAdapter::new(engine)))
    }

    fn num_qubits_hint(&self) -> Option<usize> {
        // Most builders don't know num_qubits until built
        None
    }
}

/// Engine builder stored as data, waiting for source to be configured at build time.
///
/// This enum holds engine builders in their unconfigured state. At `.build()` time,
/// the source code is injected and the builder is configured. This follows the
/// "everything is data" principle - we store configuration as data and defer
/// actual construction to build time.
#[cfg(feature = "engines-adapter")]
pub enum PendingEngineBuilder {
    /// QASM engine builder (requires `qasm` feature)
    #[cfg(feature = "qasm")]
    Qasm(pecos_qasm::QasmEngineBuilder),
    // Future: Add variants for Hugr, PhirJson, Qis as support is added
}

#[cfg(feature = "engines-adapter")]
impl PendingEngineBuilder {
    /// Configure this builder with source and return a boxed engine builder.
    ///
    /// Called at `.build()` time to inject the source into the stored builder.
    fn configure_with_source(self, source: String) -> Box<dyn BoxedEngineBuilder> {
        match self {
            #[cfg(feature = "qasm")]
            Self::Qasm(builder) => {
                let configured = builder.qasm(source);
                Box::new(EngineBuilderWrapper { builder: configured })
            }
        }
    }
}

// Conversion from QasmEngineBuilder to PendingEngineBuilder
#[cfg(feature = "qasm")]
impl From<pecos_qasm::QasmEngineBuilder> for PendingEngineBuilder {
    fn from(builder: pecos_qasm::QasmEngineBuilder) -> Self {
        Self::Qasm(builder)
    }
}

/// The source of quantum operations for simulation.
pub enum ProgramSource {
    /// A static circuit (no mid-circuit feedback).
    Static(CommandQueue),
    /// Raw program source code (needs engine factory to interpret).
    #[cfg(feature = "engines-adapter")]
    RawSource(String),
    /// A typed program (knows its type, can use `.auto()` for engine selection).
    #[cfg(feature = "engines-adapter")]
    Typed(TypedProgram),
    /// A classical engine builder (supports mid-circuit feedback).
    #[cfg(feature = "engines-adapter")]
    Classical(Box<dyn BoxedEngineBuilder>),
}

/// Typed program variants for automatic engine selection.
///
/// When using `.auto()`, the appropriate engine is selected based on the variant.
#[cfg(feature = "engines-adapter")]
#[derive(Debug, Clone)]
pub enum TypedProgram {
    /// QASM program - uses `qasm_engine()`
    Qasm(pecos_programs::Qasm),
    /// Unsupported program type (for error messages)
    Unsupported(String),
    // Future: Add Hugr, PhirJson, Qis as support is added
}

/// Resource to hold the program source.
pub struct ProgramSourceResource(pub ProgramSource);


// ============================================================================
// Simulation Plugin
// ============================================================================

/// Plugin that provides core simulation functionality.
///
/// This plugin sets up the systems for running quantum circuit simulations.
pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, tool: &mut Tool) {
        // Insert default resources if not present
        if !tool.contains_resource::<SimConfig>() {
            tool.insert_resource_mut(SimConfig::default());
        }
        if !tool.contains_resource::<SimulationResults>() {
            tool.insert_resource_mut(SimulationResults::new());
        }

        // Add simulation systems
        tool.add_system_mut(Stage::Startup, simulation_startup);
        tool.add_system_mut(Stage::PreShot, simulation_pre_shot);
        tool.add_system_mut(Stage::Execute, simulation_execute);
        tool.add_system_mut(Stage::PostShot, simulation_post_shot);
    }
}

/// Startup system: Initialize the simulator and runner.
fn simulation_startup(resources: &mut Resources) {
    // Get configuration (cloned to avoid borrow issues)
    let config = resources.get::<SimConfig>().clone();
    let circuit = resources.get::<Circuit>().0.clone();

    // Determine number of qubits from circuit
    let num_qubits = circuit
        .iter()
        .flat_map(|cmd| cmd.qubits.iter())
        .map(|q| q.0)
        .max()
        .map_or(1, |max| max + 1);

    // Create simulator and runner
    let sim = SparseStab::new(num_qubits);
    let mut runner = ShotRunner::new(sim);

    // Apply noise if present - take ownership since we can't borrow
    if let Some(noise) = resources.try_remove::<NoiseResource>() {
        runner = runner.with_noise(noise.0);
    }

    // Apply seed if configured
    if let Some(seed) = config.seed {
        runner = runner.with_full_seed(seed);
    }

    // Store shot state with circuit
    resources.insert(ShotState {
        runner,
        shot_index: 0,
        circuit,
    });

    // Clear previous results
    resources.get_mut::<SimulationResults>().clear();
}

/// Pre-shot system: Prepare for next shot.
fn simulation_pre_shot(resources: &mut Resources) {
    let config = resources.get::<SimConfig>().clone();
    let state = resources.get_mut::<ShotState>();

    // Derive per-shot seed if configured
    if let Some(base_seed) = config.seed {
        let shot_seed = derive_seed(base_seed, &format!("shot_{}", state.shot_index));
        state.runner.set_full_seed(shot_seed);
    }
}

/// Execute system: Run the circuit.
fn simulation_execute(resources: &mut Resources) {
    let state = resources.get_mut::<ShotState>();

    // Run the shot (resets simulator internally)
    let outcomes = state.runner.run_shot_fresh(&state.circuit);

    // Store outcomes temporarily for post-shot processing
    resources.insert(CurrentOutcomes(outcomes));
}

/// Post-shot system: Collect results.
fn simulation_post_shot(resources: &mut Resources) {
    // Move outcomes to results
    let outcomes = resources.remove::<CurrentOutcomes>();
    resources
        .get_mut::<SimulationResults>()
        .outcomes
        .push(outcomes.0);

    // Increment shot counter
    resources.get_mut::<ShotState>().shot_index += 1;
}

/// Temporary storage for current shot outcomes.
struct CurrentOutcomes(MeasurementOutcomes);

// ============================================================================
// SimNeoBuilder
// ============================================================================

/// Builder for configuring simulation tools (builder-of-builders pattern).
///
/// This builder collects configuration data and sub-builders, then assembles
/// everything into a [`Tool`] at build time.
///
/// Created via [`sim_neo()`] or [`sim_neo_builder()`], this builder provides
/// a fluent API for configuring quantum circuit simulations.
///
/// # Usage Patterns
///
/// ## Static Circuit
///
/// ```ignore
/// use pecos_neo::tool::sim_neo;
///
/// let results = sim_neo(circuit)
///     .depolarizing(0.01)
///     .shots(1000)
///     .seed(42)
///     .build()
///     .run();
/// ```
///
/// ## QASM Program (builder-of-builders pattern)
///
/// ```ignore
/// use pecos_neo::tool::sim_neo;
/// use pecos_qasm::qasm_engine;
///
/// // Pass program source first, then engine factory
/// let results = sim_neo(qasm_code)
///     .classical(qasm_engine())  // Engine configured with source at build time
///     .shots(1000)
///     .seed(42)
///     .build()
///     .run();
/// ```
///
/// ## Pre-configured Engine Builder
///
/// ```ignore
/// use pecos_neo::tool::sim_neo_builder;
/// use pecos_qasm::qasm_engine;
///
/// // Or pass already-configured engine builder
/// let results = sim_neo_builder()
///     .classical(qasm_engine().qasm(qasm_code))
///     .shots(1000)
///     .build()
///     .run();
/// ```
pub struct SimNeoBuilder {
    /// The program source (circuit, raw source, or engine builder).
    source: Option<ProgramSource>,
    /// Engine builder stored as data, waiting for source at build time.
    #[cfg(feature = "engines-adapter")]
    pending_builder: Option<PendingEngineBuilder>,
    /// Noise model (collected as data, used at build time).
    noise: Option<ComposableNoiseModel>,
    /// Simulation configuration (data).
    config: SimConfig,
    /// Orchestration strategy (data).
    orchestrator: Orchestrator,
    /// Quantum backend configuration (data).
    quantum_backend: QuantumBackend,
    /// Explicit qubit count override (data).
    explicit_num_qubits: Option<usize>,
}

impl SimNeoBuilder {
    /// Create a new simulation builder for a circuit.
    #[must_use]
    pub fn with_circuit(circuit: CommandQueue) -> Self {
        Self {
            source: Some(ProgramSource::Static(circuit)),
            #[cfg(feature = "engines-adapter")]
            pending_builder: None,
            noise: None,
            config: SimConfig::default(),
            orchestrator: Orchestrator::default(),
            quantum_backend: QuantumBackend::default(),
            explicit_num_qubits: None,
        }
    }

    /// Create a simulation builder with raw program source.
    ///
    /// Use `.classical(builder)` to specify how to interpret the source.
    #[must_use]
    #[cfg(feature = "engines-adapter")]
    pub fn with_program_source(source: String) -> Self {
        Self {
            source: Some(ProgramSource::RawSource(source)),
            pending_builder: None,
            noise: None,
            config: SimConfig::default(),
            orchestrator: Orchestrator::default(),
            quantum_backend: QuantumBackend::default(),
            explicit_num_qubits: None,
        }
    }

    /// Create a simulation builder with a typed program.
    ///
    /// Use `.auto()` to automatically select the engine, or
    /// `.classical(builder)` for explicit control.
    #[must_use]
    #[cfg(feature = "engines-adapter")]
    pub fn with_typed_program(program: TypedProgram) -> Self {
        Self {
            source: Some(ProgramSource::Typed(program)),
            pending_builder: None,
            noise: None,
            config: SimConfig::default(),
            orchestrator: Orchestrator::default(),
            quantum_backend: QuantumBackend::default(),
            explicit_num_qubits: None,
        }
    }

    /// Create a new simulation builder for a circuit (legacy alias).
    #[must_use]
    pub fn new(circuit: CommandQueue) -> Self {
        Self::with_circuit(circuit)
    }

    /// Create an empty simulation builder.
    ///
    /// Use this when you want to set the program source via `.classical()`.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            source: None,
            #[cfg(feature = "engines-adapter")]
            pending_builder: None,
            noise: None,
            config: SimConfig::default(),
            orchestrator: Orchestrator::default(),
            quantum_backend: QuantumBackend::default(),
            explicit_num_qubits: None,
        }
    }

    /// Set the classical control engine builder (builder-of-builders pattern).
    ///
    /// The builder is stored as data and configured with source at `.build()` time.
    /// This follows "everything is data" - we collect configuration, then wire
    /// it all together when building the Tool.
    ///
    /// ```ignore
    /// use pecos_neo::tool::sim_neo;
    /// use pecos_qasm::qasm_engine;
    ///
    /// // Builder is stored as data, source injected at build time
    /// let results = sim_neo(qasm_code)
    ///     .classical(qasm_engine())  // stores builder as data
    ///     .shots(1000)
    ///     .build()  // orchestrates: configures builder, builds engine, creates Tool
    ///     .run();
    /// ```
    ///
    /// For pre-configured engine builders, use `.with_engine()` instead:
    ///
    /// ```ignore
    /// let results = sim_neo_builder()
    ///     .with_engine(qasm_engine().qasm(qasm_code))
    ///     .shots(1000)
    ///     .build()
    ///     .run();
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if no raw source was provided via `sim_neo(source_code)`.
    #[cfg(feature = "engines-adapter")]
    #[must_use]
    pub fn classical<B>(mut self, builder: B) -> Self
    where
        B: Into<PendingEngineBuilder>,
    {
        // Check if we have source to configure the builder with later
        match self.source.take() {
            Some(ProgramSource::RawSource(source)) => {
                // Store source and builder as data; they'll be combined at build time
                self.source = Some(ProgramSource::RawSource(source));
                self.pending_builder = Some(builder.into());
            }
            Some(ProgramSource::Typed(typed)) => {
                // Extract source from typed program
                let source = match typed {
                    TypedProgram::Qasm(qasm) => qasm.source,
                    TypedProgram::Unsupported(name) => {
                        panic!("Unsupported program type: {name}");
                    }
                };
                self.source = Some(ProgramSource::RawSource(source));
                self.pending_builder = Some(builder.into());
            }
            Some(ProgramSource::Static(_)) => {
                panic!(
                    "Cannot use .classical() with a static circuit. \
                     Use sim_neo(source_code).classical(builder) for classical engines."
                );
            }
            Some(ProgramSource::Classical(_)) => {
                panic!(
                    "Classical engine already set. \
                     Use .classical() only once."
                );
            }
            None => {
                panic!(
                    "No program source provided. \
                     Use sim_neo(source_code).classical(builder) or \
                     sim_neo_builder().with_engine(configured_builder)"
                );
            }
        }
        self
    }

    /// Set the classical control engine with a pre-configured builder.
    ///
    /// Use this when you've already configured the engine builder with its program.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pecos_neo::tool::sim_neo_builder;
    /// use pecos_qasm::qasm_engine;
    ///
    /// let results = sim_neo_builder()
    ///     .with_engine(qasm_engine().qasm(qasm_code))
    ///     .shots(1000)
    ///     .build()
    ///     .run();
    /// ```
    #[cfg(feature = "engines-adapter")]
    #[must_use]
    pub fn with_engine<B>(mut self, engine_builder: B) -> Self
    where
        B: pecos_engines::ClassicalControlEngineBuilder + Send + Sync + 'static,
        B::Engine: 'static,
    {
        self.source = Some(ProgramSource::Classical(Box::new(EngineBuilderWrapper {
            builder: engine_builder,
        })));
        self
    }

    /// Automatically select the appropriate engine based on program type.
    ///
    /// This is a convenience method that selects good defaults:
    /// - `Qasm` programs use `qasm_engine()`
    /// - Future: `Hugr`, `PhirJson`, `Qis` will use their respective engines
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pecos_neo::tool::sim_neo;
    /// use pecos_programs::Qasm;
    ///
    /// // Auto-select engine based on program type
    /// let results = sim_neo(Qasm::from_string(qasm_code))
    ///     .auto()
    ///     .shots(1000)
    ///     .build()
    ///     .run();
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - No typed program was provided (use `sim_neo(Qasm::from_string(...))`)
    /// - The program type is not yet supported for auto-selection
    ///
    /// Note: `.auto()` also sets Monte Carlo orchestration with auto-detected workers
    /// as the default execution strategy.
    #[cfg(feature = "engines-adapter")]
    #[must_use]
    pub fn auto(mut self) -> Self {
        match self.source.take() {
            Some(ProgramSource::Typed(typed)) => {
                match typed {
                    TypedProgram::Qasm(qasm) => {
                        // Auto-select qasm_engine() and configure with the program
                        let builder = pecos_qasm::qasm_engine().qasm(qasm.source);
                        self.source = Some(ProgramSource::Classical(Box::new(
                            EngineBuilderWrapper { builder },
                        )));
                    }
                    TypedProgram::Unsupported(type_name) => {
                        panic!(
                            "Program type '{type_name}' is not yet supported for auto-selection. \
                             Use .classical(engine) to specify the engine explicitly."
                        );
                    }
                }
            }
            Some(ProgramSource::RawSource(_)) => {
                panic!(
                    "Cannot use .auto() with raw string source. \
                     Use sim_neo(Qasm::from_string(...)).auto() or \
                     sim_neo(source).classical(engine) instead."
                );
            }
            Some(ProgramSource::Static(_)) => {
                panic!(
                    "Cannot use .auto() with static circuits. \
                     Static circuits don't need an engine - just call .build() directly."
                );
            }
            Some(ProgramSource::Classical(_)) => {
                panic!(
                    "Engine already configured. \
                     Don't use both .auto() and .classical()/.with_engine()."
                );
            }
            None => {
                panic!(
                    "No program provided. \
                     Use sim_neo(Qasm::from_string(...)).auto() or similar."
                );
            }
        }

        // Auto mode defaults to Monte Carlo with auto-detected workers
        self.orchestrator = Orchestrator::monte_carlo_auto();
        self
    }

    /// Set the number of qubits explicitly.
    ///
    /// This is required when using `.classical()` with engines that don't
    /// report their qubit count until built.
    #[must_use]
    pub fn qubits(mut self, num_qubits: usize) -> Self {
        self.explicit_num_qubits = Some(num_qubits);
        self
    }

    /// Set the number of shots.
    #[must_use]
    pub fn shots(mut self, shots: usize) -> Self {
        self.config.shots = shots;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Set the orchestration strategy for simulation execution.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pecos_neo::tool::{sim_neo, Orchestrator};
    ///
    /// // Parallel Monte Carlo with 4 workers
    /// let results = sim_neo(circuit)
    ///     .orchestrator(Orchestrator::monte_carlo(4))
    ///     .shots(1000)
    ///     .run();
    ///
    /// // Auto-detect worker count
    /// let results = sim_neo(circuit)
    ///     .orchestrator(Orchestrator::monte_carlo_auto())
    ///     .shots(1000)
    ///     .run();
    /// ```
    #[must_use]
    pub fn orchestrator(mut self, orchestrator: Orchestrator) -> Self {
        self.orchestrator = orchestrator;
        self
    }

    /// Convenience method for parallel Monte Carlo with specified workers.
    #[must_use]
    pub fn workers(mut self, workers: usize) -> Self {
        self.orchestrator = Orchestrator::monte_carlo(workers);
        self
    }

    /// Convenience method for parallel Monte Carlo with auto-detected workers.
    #[must_use]
    pub fn auto_workers(mut self) -> Self {
        self.orchestrator = Orchestrator::monte_carlo_auto();
        self
    }

    /// Set the quantum backend for simulation.
    ///
    /// This selects which quantum simulator to use. Different backends have
    /// different capabilities and performance characteristics:
    ///
    /// - `sparse_stab()` - Sparse stabilizer (default), efficient for Clifford circuits
    /// - `state_vector()` - State vector, supports arbitrary gates including T and rotations
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pecos_neo::tool::{sim_neo, sparse_stab, state_vector};
    ///
    /// // Use sparse stabilizer (default, Clifford-only)
    /// let results = sim_neo(circuit)
    ///     .quantum(sparse_stab())
    ///     .shots(1000)
    ///     .run();
    ///
    /// // Use state vector (supports T gates, rotations)
    /// let results = sim_neo(circuit)
    ///     .quantum(state_vector())
    ///     .shots(1000)
    ///     .run();
    /// ```
    #[must_use]
    pub fn quantum<B: Into<QuantumBackend>>(mut self, backend: B) -> Self {
        self.quantum_backend = backend.into();
        self
    }

    /// Set the noise model.
    ///
    /// Accepts any type that implements `Into<ComposableNoiseModel>`:
    /// - `ComposableNoiseModel` directly
    /// - `GeneralNoiseModelBuilder` (without calling `.build()`)
    /// - Any single `NoiseChannel`
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pecos_neo::tool::sim_neo;
    /// use pecos_neo::noise::{GeneralNoiseModelBuilder, SingleQubitChannel};
    ///
    /// // Using GeneralNoiseModelBuilder (no .build() needed)
    /// sim_neo(circuit.clone())
    ///     .noise(GeneralNoiseModelBuilder::new().with_p1(0.01).with_p2(0.02))
    ///     .build();
    ///
    /// // Using a single channel directly
    /// sim_neo(circuit.clone())
    ///     .noise(SingleQubitChannel::depolarizing(0.01))
    ///     .build();
    /// ```
    #[must_use]
    pub fn noise(mut self, noise: impl Into<ComposableNoiseModel>) -> Self {
        self.noise = Some(noise.into());
        self
    }

    /// Add uniform depolarizing noise to all operations.
    ///
    /// This is a convenience method equivalent to:
    /// ```text
    /// .noise(GeneralNoiseModelBuilder::new()
    ///     .with_p1(p)
    ///     .with_p2(p)
    ///     .with_p_prep(p)
    ///     .with_p_meas_symmetric(p))
    /// ```
    ///
    /// # Arguments
    /// * `p` - Error probability for gates, preparation, and measurements
    #[must_use]
    pub fn depolarizing(self, p: f64) -> Self {
        self.noise(
            crate::noise::GeneralNoiseModelBuilder::new()
                .with_p1(p)
                .with_p2(p)
                .with_p_prep(p)
                .with_p_meas_symmetric(p),
        )
    }

    /// Build the simulation handle.
    ///
    /// This is where all the collected builders and configuration come together:
    /// - Program source is wired with engine factory (if applicable)
    /// - Noise model is built
    /// - Tool is constructed with all plugins and systems
    ///
    /// # Panics
    ///
    /// Panics if no program source is set (neither circuit nor classical engine).
    #[must_use]
    pub fn build(self) -> Simulation {
        // Resolve the program source - configure pending builder with source if needed
        #[cfg(feature = "engines-adapter")]
        let source = {
            match (self.source, self.pending_builder) {
                // Raw source + pending builder = configure and use
                (Some(ProgramSource::RawSource(source)), Some(builder)) => {
                    let configured = builder.configure_with_source(source);
                    ProgramSource::Classical(configured)
                }
                // Raw source without builder - error
                (Some(ProgramSource::RawSource(_)), None) => {
                    panic!(
                        "Program source provided but no engine builder. \
                         Use .classical(builder) to specify how to interpret the source."
                    );
                }
                // Typed program without .auto() - error with helpful message
                (Some(ProgramSource::Typed(typed)), _) => {
                    let type_name = match &typed {
                        TypedProgram::Qasm(_) => "Qasm",
                        TypedProgram::Unsupported(name) => name,
                    };
                    panic!(
                        "Typed program ({type_name}) provided but engine not selected. \
                         Use .auto() for automatic engine selection or \
                         .classical(builder) for explicit control."
                    );
                }
                // Already resolved source
                (Some(source), _) => source,
                // No source - error
                (None, _) => {
                    panic!(
                        "No program source set. Use sim_neo(circuit) or \
                         sim_neo(source).classical(builder) or \
                         sim_neo_builder().with_engine(configured_builder)"
                    );
                }
            }
        };

        #[cfg(not(feature = "engines-adapter"))]
        let source = self.source.expect(
            "No program source set. Use sim_neo(circuit) to provide a circuit."
        );

        // Extract parallel execution data for static circuits without noise
        // (noise models contain trait objects that aren't Clone)
        let parallel_data = match (&source, &self.noise) {
            (ProgramSource::Static(circuit), None) => {
                // Compute num_qubits for parallel workers
                let inferred_qubits = circuit
                    .iter()
                    .flat_map(|cmd| cmd.qubits.iter())
                    .map(|q| q.0)
                    .max()
                    .map_or(1, |max| max + 1);
                let num_qubits = self.explicit_num_qubits.unwrap_or(inferred_qubits);

                Some(ParallelExecutionData {
                    circuit: circuit.clone(),
                    num_qubits,
                    quantum_backend: self.quantum_backend.clone(),
                })
            }
            _ => None, // Classical engines or noise models don't support parallel execution yet
        };

        let mut tool = Tool::new()
            .insert_resource(ProgramSourceResource(source))
            .insert_resource(self.config)
            .add_plugin(UnifiedSimulationPlugin {
                explicit_num_qubits: self.explicit_num_qubits,
                quantum_backend: self.quantum_backend,
            });

        // Add noise if configured
        if let Some(noise) = self.noise {
            tool = tool.insert_resource(NoiseResource(noise));
        }

        Simulation {
            tool,
            orchestrator: self.orchestrator,
            parallel_data,
        }
    }

    /// Build and run the simulation in one step.
    ///
    /// This is a convenience method equivalent to `.build().run()`.
    /// Use `.build()` instead if you need to run multiple times or reconfigure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pecos_neo::tool::sim_neo;
    /// use pecos_qasm::qasm_engine;
    ///
    /// let results = sim_neo(qasm_code)
    ///     .classical(qasm_engine())
    ///     .shots(1000)
    ///     .run();  // builds and runs
    /// ```
    #[must_use]
    pub fn run(self) -> SimulationResults {
        self.build().run()
    }
}

// ============================================================================
// Unified Simulation Plugin
// ============================================================================

/// Plugin that handles both static circuits and classical engines.
struct UnifiedSimulationPlugin {
    explicit_num_qubits: Option<usize>,
    quantum_backend: QuantumBackend,
}

/// Resource to store explicit qubit count.
struct ExplicitNumQubits(Option<usize>);

/// Resource to store quantum backend choice.
struct QuantumBackendResource(QuantumBackend);

impl Plugin for UnifiedSimulationPlugin {
    fn build(&self, tool: &mut Tool) {
        // Insert default resources if not present
        if !tool.contains_resource::<SimConfig>() {
            tool.insert_resource_mut(SimConfig::default());
        }
        if !tool.contains_resource::<SimulationResults>() {
            tool.insert_resource_mut(SimulationResults::new());
        }

        // Store explicit num_qubits for startup
        tool.insert_resource_mut(ExplicitNumQubits(self.explicit_num_qubits));

        // Store quantum backend choice for startup
        tool.insert_resource_mut(QuantumBackendResource(self.quantum_backend.clone()));

        // Add simulation systems
        tool.add_system_mut(Stage::Startup, unified_simulation_startup);
        tool.add_system_mut(Stage::PreShot, unified_simulation_pre_shot);
        tool.add_system_mut(Stage::Execute, unified_simulation_execute);
        tool.add_system_mut(Stage::PostShot, unified_simulation_post_shot);
    }
}

/// Quantum runner that dispatches to different simulator backends.
///
/// This enum allows runtime selection of quantum simulators while maintaining
/// type safety. Each variant wraps a `ProgramRunner<S>` for the appropriate
/// simulator type.
pub enum QuantumRunner {
    /// Sparse stabilizer simulator (Clifford-only).
    SparseStab(ProgramRunner<SparseStab>),
    /// State vector simulator (supports arbitrary gates).
    StateVec(ProgramRunner<StateVec>),
}

impl QuantumRunner {
    /// Run a shot and return the result.
    pub fn run_shot(&mut self, source: &mut dyn CommandSource) -> crate::program::ProgramResult {
        match self {
            Self::SparseStab(runner) => runner.run_shot(source),
            Self::StateVec(runner) => runner.run_shot(source),
        }
    }

    /// Get mutable access to the shot runner for seeding.
    pub fn shot_runner_mut(&mut self) -> &mut dyn ShotRunnerOps {
        match self {
            Self::SparseStab(runner) => runner.shot_runner_mut(),
            Self::StateVec(runner) => runner.shot_runner_mut(),
        }
    }
}

/// Trait for common shot runner operations needed by the simulation systems.
pub trait ShotRunnerOps {
    /// Set the full seed for deterministic execution.
    fn set_full_seed(&mut self, seed: u64);
}

impl<S: CliffordGateable + pecos_core::rng::RngManageable<Rng = pecos_rng::PecosRng>> ShotRunnerOps
    for ShotRunner<S>
{
    fn set_full_seed(&mut self, seed: u64) {
        ShotRunner::set_full_seed(self, seed);
    }
}

/// Unified shot state that works with both static circuits and dynamic programs.
pub struct UnifiedShotState {
    /// Quantum runner for execution (dispatches to appropriate backend).
    pub quantum_runner: QuantumRunner,
    /// The command source (static or from classical engine).
    pub command_source: Box<dyn CommandSource + Send + Sync>,
    /// Current shot index.
    pub shot_index: usize,
}

/// Startup system for unified simulation.
fn unified_simulation_startup(resources: &mut Resources) {
    let config = resources.get::<SimConfig>().clone();
    let explicit_qubits = resources.get::<ExplicitNumQubits>().0;

    // Check if we already have a UnifiedShotState (from a previous run)
    // If so, just reset it instead of rebuilding
    if resources.contains::<UnifiedShotState>() {
        let state = resources.get_mut::<UnifiedShotState>();
        state.shot_index = 0;
        state.command_source.reset();

        // Clear previous results
        resources.get_mut::<SimulationResults>().clear();
        return;
    }

    // First run - take the program source and build
    let source_resource = resources.remove::<ProgramSourceResource>();

    // Build the command source and determine num_qubits
    let (command_source, num_qubits): (Box<dyn CommandSource + Send + Sync>, usize) =
        match source_resource.0 {
            ProgramSource::Static(circuit) => {
                // Determine num_qubits from circuit
                let inferred_qubits = circuit
                    .iter()
                    .flat_map(|cmd| cmd.qubits.iter())
                    .map(|q| q.0)
                    .max()
                    .map_or(1, |max| max + 1);

                let num_qubits = explicit_qubits.unwrap_or(inferred_qubits);
                let program = StaticProgram::new(circuit, num_qubits);
                (Box::new(program), num_qubits)
            }
            #[cfg(feature = "engines-adapter")]
            ProgramSource::RawSource(_) => {
                // This should never happen - build() resolves RawSource with engine factory
                unreachable!(
                    "RawSource should be resolved to Classical by SimNeoBuilder::build(). \
                     This is a bug in the simulation builder."
                );
            }
            #[cfg(feature = "engines-adapter")]
            ProgramSource::Typed(_) => {
                // This should never happen - build() catches Typed without .auto()
                unreachable!(
                    "Typed program should be resolved by .auto() or caught by build(). \
                     This is a bug in the simulation builder."
                );
            }
            #[cfg(feature = "engines-adapter")]
            ProgramSource::Classical(engine_builder) => {
                // Build the engine adapter
                let adapter = engine_builder
                    .build_adapter()
                    .expect("Failed to build classical engine");

                let num_qubits = explicit_qubits.unwrap_or_else(|| adapter.num_qubits());
                (adapter, num_qubits)
            }
        };

    // Get quantum backend choice
    let backend = resources.get::<QuantumBackendResource>().0.clone();

    // Create quantum runner based on backend choice
    let noise = resources.try_remove::<NoiseResource>();
    let quantum_runner = match backend {
        QuantumBackend::SparseStab => {
            let sim = SparseStab::new(num_qubits);
            let mut runner = ProgramRunner::new(sim);
            if let Some(n) = noise {
                runner = runner.with_noise(n.0);
            }
            if let Some(seed) = config.seed {
                runner = runner.with_seed(seed);
            }
            QuantumRunner::SparseStab(runner)
        }
        QuantumBackend::StateVec => {
            let sim = StateVec::new(num_qubits);
            let mut runner = ProgramRunner::new(sim);
            if let Some(n) = noise {
                runner = runner.with_noise(n.0);
            }
            if let Some(seed) = config.seed {
                runner = runner.with_seed(seed);
            }
            QuantumRunner::StateVec(runner)
        }
    };

    // Store unified shot state
    resources.insert(UnifiedShotState {
        quantum_runner,
        command_source,
        shot_index: 0,
    });

    // Clear previous results
    resources.get_mut::<SimulationResults>().clear();
}

/// Pre-shot system for unified simulation.
fn unified_simulation_pre_shot(resources: &mut Resources) {
    let config = resources.get::<SimConfig>().clone();
    let state = resources.get_mut::<UnifiedShotState>();

    // Derive per-shot seed if configured
    if let Some(base_seed) = config.seed {
        let shot_seed = derive_seed(base_seed, &format!("shot_{}", state.shot_index));
        state.quantum_runner.shot_runner_mut().set_full_seed(shot_seed);
    }
}

/// Execute system for unified simulation.
fn unified_simulation_execute(resources: &mut Resources) {
    let state = resources.get_mut::<UnifiedShotState>();

    // Run the program (handles both static and dynamic programs)
    let result = state.quantum_runner.run_shot(&mut *state.command_source);

    // Store outcomes temporarily for post-shot processing
    resources.insert(CurrentOutcomes(result.outcomes));
}

/// Post-shot system for unified simulation.
fn unified_simulation_post_shot(resources: &mut Resources) {
    // Move outcomes to results
    let outcomes = resources.remove::<CurrentOutcomes>();
    resources
        .get_mut::<SimulationResults>()
        .outcomes
        .push(outcomes.0);

    // Increment shot counter
    resources.get_mut::<UnifiedShotState>().shot_index += 1;
}

// ============================================================================
// Simulation Handle
// ============================================================================

/// Reusable simulation handle.
///
/// Created via [`SimNeoBuilder::build()`], this handle can be run multiple
/// times with different configurations.
///
/// # Example
///
/// ```ignore
/// let mut sim = sim_neo(circuit).shots(1000).build();
///
/// let results1 = sim.run();
///
/// // Reconfigure and run again
/// sim.shots(2000).seed(123);
/// let results2 = sim.run();
/// ```
pub struct Simulation {
    tool: Tool,
    /// Orchestration strategy (stored as data).
    orchestrator: Orchestrator,
    /// Data for parallel execution (if applicable).
    /// Stored separately from Tool to allow cloning for workers.
    parallel_data: Option<ParallelExecutionData>,
}

/// Data stored for parallel execution support.
///
/// This is populated for static circuits without noise (which can be cloned for workers).
/// For classical engines or circuits with noise, this is None and execution falls back to sequential.
struct ParallelExecutionData {
    /// The circuit to execute (cloned for each worker).
    circuit: CommandQueue,
    /// Number of qubits for simulators.
    num_qubits: usize,
    /// Quantum backend to use.
    quantum_backend: QuantumBackend,
}

impl Simulation {
    /// Set the number of shots for the next run.
    pub fn shots(&mut self, shots: usize) -> &mut Self {
        self.tool.resource_mut::<SimConfig>().shots = shots;
        self
    }

    /// Set the seed for the next run.
    pub fn seed(&mut self, seed: u64) -> &mut Self {
        self.tool.resource_mut::<SimConfig>().seed = Some(seed);
        self
    }

    /// Run the simulation with current configuration.
    ///
    /// Returns the simulation results. The simulation can be run again
    /// after reconfiguring with [`shots()`](Self::shots) or [`seed()`](Self::seed).
    ///
    /// Execution strategy depends on the orchestrator:
    /// - `Sequential`: Runs shots one at a time via the Tool
    /// - `MonteCarlo`: Parallelizes shots across workers (for static noiseless circuits)
    /// - `ImportanceSampling`: Biased sampling for rare events (not yet implemented)
    pub fn run(&mut self) -> SimulationResults {
        let config = self.tool.resource::<SimConfig>().clone();

        // Dispatch based on orchestration strategy
        match &self.orchestrator {
            Orchestrator::MonteCarlo { workers } if *workers > 1 => {
                if let Some(ref parallel_data) = self.parallel_data {
                    return self.run_parallel(&config, parallel_data, *workers);
                }
                // Fall through to sequential for classical engines or noisy circuits
            }
            Orchestrator::ImportanceSampling { .. } => {
                // TODO: Implement importance sampling via ImportanceSamplingPlugin
                unimplemented!("Importance sampling orchestration not yet implemented");
            }
            _ => {} // Sequential falls through
        }

        // Sequential execution via Tool
        self.tool.reset();
        self.tool.run_shots(config.shots);

        // Take results and re-insert empty for next run
        let results = self.tool.take_resource::<SimulationResults>();
        self.tool.insert_resource_mut(SimulationResults::new());
        results
    }

    /// Run shots in parallel using rayon (noiseless static circuits only).
    fn run_parallel(
        &self,
        config: &SimConfig,
        data: &ParallelExecutionData,
        num_workers: usize,
    ) -> SimulationResults {
        let shots = config.shots;
        let base_seed = config.seed.unwrap_or(0);

        // Distribute shots among workers and compute starting indices
        let shots_per_worker = distribute_shots(shots, num_workers);
        let mut start_indices = vec![0usize; num_workers];
        for i in 1..num_workers {
            start_indices[i] = start_indices[i - 1] + shots_per_worker[i - 1];
        }

        // Run in parallel, each worker with its own state
        let all_outcomes: Vec<Vec<MeasurementOutcomes>> = (0..num_workers)
            .into_par_iter()
            .map(|worker_id| {
                let worker_shots = shots_per_worker[worker_id];
                if worker_shots == 0 {
                    return Vec::new();
                }

                let start_index = start_indices[worker_id];

                // Run shots for this worker, using per-shot seeding (same as sequential)
                let mut outcomes = Vec::with_capacity(worker_shots);
                for local_shot in 0..worker_shots {
                    let global_shot = start_index + local_shot;

                    // Derive per-shot seed (matches sequential mode's set_full_seed)
                    let shot_seed = derive_seed(base_seed, &format!("shot_{global_shot}"));
                    // Further derive separate seeds for simulator and noise (matches set_full_seed)
                    let sim_seed = derive_seed(shot_seed, "simulator");
                    let noise_seed = derive_seed(shot_seed, "noise");

                    // Create fresh simulator and runner for each shot based on backend
                    let mut program = StaticProgram::new(data.circuit.clone(), data.num_qubits);
                    let result_outcomes = match &data.quantum_backend {
                        QuantumBackend::SparseStab => {
                            let sim = SparseStab::with_seed(data.num_qubits, sim_seed);
                            let mut runner = ProgramRunner::new(sim).with_seed(noise_seed);
                            runner.run_shot(&mut program).outcomes
                        }
                        QuantumBackend::StateVec => {
                            let sim = StateVec::with_seed(data.num_qubits, sim_seed);
                            let mut runner = ProgramRunner::new(sim).with_seed(noise_seed);
                            runner.run_shot(&mut program).outcomes
                        }
                    };
                    outcomes.push(result_outcomes);
                }
                outcomes
            })
            .collect();

        // Flatten results in deterministic order
        let outcomes: Vec<MeasurementOutcomes> = all_outcomes.into_iter().flatten().collect();

        SimulationResults { outcomes }
    }

    /// Get a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &SimConfig {
        self.tool.resource::<SimConfig>()
    }

    /// Get access to the underlying tool (for advanced use).
    #[must_use]
    pub fn tool(&self) -> &Tool {
        &self.tool
    }

    /// Get mutable access to the underlying tool (for advanced use).
    #[must_use]
    pub fn tool_mut(&mut self) -> &mut Tool {
        &mut self.tool
    }
}

// ============================================================================
// Convenience Entry Point
// ============================================================================

/// Create a simulation builder for any program type.
///
/// This is the primary entry point for creating quantum simulations using
/// the Tool/ECS architecture. It accepts any type that implements [`SimNeoInput`]:
///
/// - **Static circuits**: `CommandQueue`, `TickCircuit`, `DagCircuit`
/// - **Classical engines**: Any `ClassicalControlEngineBuilder` (QASM, HUGR, PHIR, QIS)
///
/// # Examples
///
/// ## Static Circuit
///
/// ```ignore
/// use pecos_neo::tool::sim_neo;
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new()
///     .prep(0).h(0).measure(0)
///     .build();
///
/// let results = sim_neo(circuit)
///     .depolarizing(0.01)
///     .shots(1000)
///     .seed(42)
///     .build()
///     .run();
/// ```
///
/// ## QASM Program
///
/// ```ignore
/// use pecos_neo::tool::sim_neo;
/// use pecos_qasm::qasm_engine;
///
/// let qasm = r#"
///     OPENQASM 2.0;
///     include "qelib1.inc";
///     qreg q[2];
///     creg c[2];
///     h q[0];
///     measure q[0] -> c[0];
///     if (c[0] == 1) x q[1];
///     measure q[1] -> c[1];
/// "#;
///
/// let results = sim_neo(qasm_engine().qasm(qasm))
///     .depolarizing(0.01)
///     .shots(1000)
///     .seed(42)
///     .build()
///     .run();
/// ```
///
/// ## Reusable Simulation
///
/// ```ignore
/// let mut sim = sim_neo(circuit)
///     .shots(1000)
///     .build();
///
/// let results1 = sim.run();
/// let results2 = sim.seed(123).shots(2000).run();
/// ```
#[must_use]
pub fn sim_neo<I: SimNeoInput>(input: I) -> SimNeoBuilder {
    input.into_sim_neo_builder()
}

/// Create an empty simulation builder for use with classical engines.
///
/// This entry point is for programs with classical control flow (conditionals,
/// loops, etc.). Use `.classical()` to set the engine builder.
///
/// # Example
///
/// ```ignore
/// use pecos_neo::tool::sim_neo_builder;
/// use pecos_qasm::qasm_engine;
///
/// let qasm = r#"
///     OPENQASM 2.0;
///     include "qelib1.inc";
///     qreg q[2];
///     creg c[2];
///     h q[0];
///     measure q[0] -> c[0];
///     if (c[0] == 1) x q[1];  // Conditional!
///     measure q[1] -> c[1];
/// "#;
///
/// let results = sim_neo_builder()
///     .classical(qasm_engine().program(qasm))
///     .depolarizing(0.01)
///     .shots(1000)
///     .seed(42)
///     .build()
///     .run();
/// ```
#[must_use]
pub fn sim_neo_builder() -> SimNeoBuilder {
    SimNeoBuilder::empty()
}

// ============================================================================
// Parallel Execution Helpers
// ============================================================================

/// Distribute shots evenly across workers with remainder going to initial workers.
fn distribute_shots(num_shots: usize, num_workers: usize) -> Vec<usize> {
    let base = num_shots / num_workers;
    let remainder = num_shots % num_workers;

    let mut result = vec![base; num_workers];
    result
        .iter_mut()
        .take(remainder)
        .for_each(|shots| *shots += 1);

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::CommandBuilder;
    use crate::noise::{ComposableNoiseModel, SingleQubitChannel};
    use pecos_core::QubitId;

    #[test]
    fn test_sim_neo_basic() {
        let circuit = CommandBuilder::new()
            .prep(0)
            .x(0) // Flip to |1>
            .measure(0)
            .build();

        let mut sim = sim_neo(circuit).shots(10).seed(42).build();

        let results = sim.run();

        assert_eq!(results.len(), 10);

        // All outcomes should be 1 (X gate flips |0> to |1>)
        for outcome in &results.outcomes {
            assert!(
                outcome.get_bit(QubitId(0)).unwrap(),
                "X gate should produce |1>"
            );
        }
    }

    #[test]
    fn test_sim_neo_rerun() {
        let circuit = CommandBuilder::new()
            .prep(0)
            .x(0)
            .measure(0)
            .build();

        let mut sim = sim_neo(circuit).shots(5).build();

        let results1 = sim.run();
        assert_eq!(results1.len(), 5);

        // Reconfigure and run again
        sim.shots(10);
        let results2 = sim.run();
        assert_eq!(results2.len(), 10);
    }

    #[test]
    fn test_sim_neo_deterministic() {
        let circuit = CommandBuilder::new()
            .prep(0)
            .h(0) // Superposition - outcome depends on RNG
            .measure(0)
            .build();

        // Same seed should produce same results
        let results1 = sim_neo(circuit.clone())
            .shots(20)
            .seed(42)
            .build()
            .run();

        let results2 = sim_neo(circuit).shots(20).seed(42).build().run();

        assert_eq!(results1.outcomes.len(), results2.outcomes.len());
        for (o1, o2) in results1.outcomes.iter().zip(results2.outcomes.iter()) {
            assert_eq!(
                o1.get_bit(QubitId(0)),
                o2.get_bit(QubitId(0)),
                "Same seed should produce identical results"
            );
        }
    }

    #[test]
    fn test_sim_neo_with_noise() {
        // Circuit: prep |0>, Z gate, measure
        // Z|0> = |0>, so without noise we'd always measure 0
        // But with depolarizing noise on the Z gate, we'll see errors
        let circuit = CommandBuilder::new()
            .prep(0)
            .z(0) // Single-qubit gate to trigger noise
            .measure(0)
            .build();

        // Very high error rate - this will definitely flip some outcomes
        let noise = ComposableNoiseModel::new()
            .add_channel(SingleQubitChannel::depolarizing(0.5));

        let results = sim_neo(circuit)
            .noise(noise)
            .shots(100)
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 100);

        // With 50% depolarizing noise, we should see a mix of outcomes
        // X and Y errors flip the qubit, Z error keeps it at 0
        // So ~1/3 of errors flip the qubit (when X or Y is sampled)
        let ones: usize = results
            .outcomes
            .iter()
            .filter(|o| o.get_bit(QubitId(0)).unwrap_or(false))
            .count();

        // With noise, we should see some 1s (very unlikely to have 0 or 100)
        assert!(
            ones > 0 && ones < 100,
            "With 50% depolarizing noise, expected mix of outcomes but got {ones} ones",
        );
    }

    #[test]
    fn test_sim_neo_noise_deterministic() {
        // Verify noise is deterministic with same seed
        let circuit = CommandBuilder::new()
            .prep(0)
            .z(0) // Single-qubit gate to trigger noise
            .measure(0)
            .build();

        let noise1 = ComposableNoiseModel::new()
            .add_channel(SingleQubitChannel::depolarizing(0.5));
        let noise2 = ComposableNoiseModel::new()
            .add_channel(SingleQubitChannel::depolarizing(0.5));

        let results1 = sim_neo(circuit.clone())
            .noise(noise1)
            .shots(20)
            .seed(42)
            .build()
            .run();

        let results2 = sim_neo(circuit)
            .noise(noise2)
            .shots(20)
            .seed(42)
            .build()
            .run();

        for (o1, o2) in results1.outcomes.iter().zip(results2.outcomes.iter()) {
            assert_eq!(
                o1.get_bit(QubitId(0)),
                o2.get_bit(QubitId(0)),
                "Noise should be deterministic with same seed"
            );
        }
    }

    #[test]
    fn test_sim_neo_ergonomic_noise() {
        // Test the ergonomic .noise(channel) syntax (without explicit ComposableNoiseModel)
        let circuit = CommandBuilder::new()
            .prep(0)
            .z(0)
            .measure(0)
            .build();

        // This uses the From<C: NoiseChannel> impl for ComposableNoiseModel
        let results = sim_neo(circuit)
            .noise(SingleQubitChannel::depolarizing(0.5))
            .shots(50)
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 50);

        // Should see some noise effect
        let ones: usize = results
            .outcomes
            .iter()
            .filter(|o| o.get_bit(QubitId(0)).unwrap_or(false))
            .count();

        assert!(ones > 0, "Expected some errors from 50% depolarizing noise");
    }

    #[test]
    fn test_sim_neo_builder_without_build() {
        // Test that GeneralNoiseModelBuilder can be passed directly without .build()
        use crate::noise::GeneralNoiseModelBuilder;

        let circuit = CommandBuilder::new()
            .prep(0)
            .x(0)
            .measure(0)
            .build();

        // Pass builder directly - no .build() needed!
        let results = sim_neo(circuit)
            .noise(GeneralNoiseModelBuilder::new().with_p1(0.3))
            .shots(100)
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 100);

        // With 30% error rate, we should see some errors
        let zeros: usize = results
            .outcomes
            .iter()
            .filter(|o| !o.get_bit(QubitId(0)).unwrap_or(true))
            .count();

        assert!(zeros > 0, "Expected some errors from 30% depolarizing noise");
    }

    #[test]
    fn test_sim_neo_convenience_depolarizing() {
        // Test the .depolarizing(p) convenience method
        let circuit = CommandBuilder::new()
            .prep(0)
            .prep(1)
            .x(0)
            .cx(0, 1)
            .measure(0)
            .measure(1)
            .build();

        let results = sim_neo(circuit)
            .depolarizing(0.2) // 20% on both 1Q and 2Q gates
            .shots(100)
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 100);

        // Should see some errors from high depolarizing rate
        let correct: usize = results
            .outcomes
            .iter()
            .filter(|o| {
                o.get_bit(QubitId(0)).unwrap_or(false)
                    && o.get_bit(QubitId(1)).unwrap_or(false)
            })
            .count();

        assert!(
            correct < 100,
            "Expected some errors from 20% depolarizing noise"
        );
    }

    #[test]
    fn test_sim_neo_measurement_noise() {
        // Test measurement noise via GeneralNoiseModelBuilder
        use crate::noise::GeneralNoiseModelBuilder;

        let circuit = CommandBuilder::new()
            .prep(0)
            .measure(0)
            .build();

        let results = sim_neo(circuit)
            .noise(GeneralNoiseModelBuilder::new().with_p_meas_symmetric(0.15))
            .shots(200)
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 200);

        // Should see ~15% flips
        let ones: usize = results
            .outcomes
            .iter()
            .filter(|o| o.get_bit(QubitId(0)).unwrap_or(false))
            .count();

        let rate = ones as f64 / 200.0;
        assert!(
            (rate - 0.15).abs() < 0.10,
            "Measurement noise rate should be ~15%: got {rate:.2}"
        );
    }

    #[test]
    fn test_sim_neo_prep_noise() {
        // Test prep noise via GeneralNoiseModelBuilder
        use crate::noise::GeneralNoiseModelBuilder;

        let circuit = CommandBuilder::new()
            .prep(0)
            .measure(0)
            .build();

        let results = sim_neo(circuit)
            .noise(GeneralNoiseModelBuilder::new().with_p_prep(0.20))
            .shots(200)
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 200);

        // Should see ~20% errors
        let ones: usize = results
            .outcomes
            .iter()
            .filter(|o| o.get_bit(QubitId(0)).unwrap_or(false))
            .count();

        let rate = ones as f64 / 200.0;
        assert!(
            (rate - 0.20).abs() < 0.10,
            "Prep noise rate should be ~20%: got {rate:.2}"
        );
    }

    #[cfg(feature = "qasm")]
    #[test]
    fn test_sim_neo_auto_with_qasm() {
        // Test the .auto() pattern with a Qasm typed program
        let qasm_source = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            x q[0];
            measure q[0] -> c[0];
        "#;

        let qasm = pecos_programs::Qasm::from_string(qasm_source);

        // .auto() should automatically select qasm_engine()
        // Using .run() shortcut (equivalent to .build().run())
        let results = sim_neo(qasm).auto().shots(10).seed(42).run();

        assert_eq!(results.len(), 10);

        // All outcomes should be 1 (X gate flips |0> to |1>)
        for outcome in &results.outcomes {
            assert!(
                outcome.get_bit(QubitId(0)).unwrap_or(false),
                "X gate should produce |1>"
            );
        }
    }

    #[cfg(feature = "qasm")]
    #[test]
    fn test_sim_neo_classical_with_run_shortcut() {
        // Test .classical() with .run() shortcut
        let qasm_source = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            x q[0];
            measure q[0] -> c[0];
        "#;

        // Direct .run() without explicit .build()
        let results = sim_neo(qasm_source)
            .classical(pecos_qasm::qasm_engine())
            .shots(10)
            .seed(42)
            .run();

        assert_eq!(results.len(), 10);

        for outcome in &results.outcomes {
            assert!(
                outcome.get_bit(QubitId(0)).unwrap_or(false),
                "X gate should produce |1>"
            );
        }
    }

    #[cfg(feature = "qasm")]
    #[test]
    fn test_sim_neo_auto_with_program_enum() {
        // Test .auto() with the Program enum wrapper
        let qasm_source = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0], q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
        "#;

        let program = pecos_programs::Program::Qasm(pecos_programs::Qasm::from_string(qasm_source));

        // .auto() should detect Qasm variant and use qasm_engine()
        let results = sim_neo(program).auto().shots(50).seed(42).build().run();

        assert_eq!(results.len(), 50);

        // Bell state: both qubits should be correlated
        for outcome in &results.outcomes {
            let q0 = outcome.get_bit(QubitId(0)).unwrap_or(false);
            let q1 = outcome.get_bit(QubitId(1)).unwrap_or(false);
            assert_eq!(q0, q1, "Bell state qubits should be correlated");
        }
    }

    #[test]
    fn test_sim_neo_monte_carlo_orchestrator() {
        // Test Monte Carlo orchestration with multiple workers
        let circuit = CommandBuilder::new()
            .prep(0)
            .x(0) // Flip to |1>
            .measure(0)
            .build();

        // Use .workers() convenience method for Monte Carlo
        let results = sim_neo(circuit)
            .workers(4)
            .shots(100)
            .seed(42)
            .run();

        assert_eq!(results.len(), 100);

        // All outcomes should be 1 (X gate flips |0> to |1>)
        for outcome in &results.outcomes {
            assert!(
                outcome.get_bit(QubitId(0)).unwrap(),
                "X gate should produce |1>"
            );
        }
    }

    #[test]
    fn test_sim_neo_monte_carlo_deterministic() {
        // Test that Monte Carlo with same seed produces same results
        let circuit = CommandBuilder::new()
            .prep(0)
            .h(0) // Superposition
            .measure(0)
            .build();

        let results1 = sim_neo(circuit.clone())
            .workers(4)
            .shots(50)
            .seed(42)
            .run();

        let results2 = sim_neo(circuit)
            .workers(4)
            .shots(50)
            .seed(42)
            .run();

        assert_eq!(results1.outcomes.len(), results2.outcomes.len());
        for (o1, o2) in results1.outcomes.iter().zip(results2.outcomes.iter()) {
            assert_eq!(
                o1.get_bit(QubitId(0)),
                o2.get_bit(QubitId(0)),
                "Same seed should produce identical results"
            );
        }
    }

    #[test]
    fn test_sim_neo_orchestrator_explicit() {
        // Test explicit orchestrator configuration
        use super::Orchestrator;

        let circuit = CommandBuilder::new()
            .prep(0)
            .x(0)
            .measure(0)
            .build();

        // Use explicit Orchestrator enum
        let results = sim_neo(circuit)
            .orchestrator(Orchestrator::monte_carlo(2))
            .shots(20)
            .seed(42)
            .run();

        assert_eq!(results.len(), 20);

        for outcome in &results.outcomes {
            assert!(
                outcome.get_bit(QubitId(0)).unwrap(),
                "X gate should produce |1>"
            );
        }
    }

    #[test]
    fn test_sim_neo_sequential_orchestrator() {
        // Test sequential orchestrator (default)
        use super::Orchestrator;

        let circuit = CommandBuilder::new()
            .prep(0)
            .x(0)
            .measure(0)
            .build();

        let results = sim_neo(circuit)
            .orchestrator(Orchestrator::Sequential)
            .shots(10)
            .seed(42)
            .run();

        assert_eq!(results.len(), 10);

        for outcome in &results.outcomes {
            assert!(
                outcome.get_bit(QubitId(0)).unwrap(),
                "X gate should produce |1>"
            );
        }
    }

    #[test]
    fn test_sim_neo_parallel_matches_sequential() {
        // Critical test: parallel and sequential should produce identical results
        // with the same seed (they use the same per-shot seeding scheme)
        use super::Orchestrator;

        let circuit = CommandBuilder::new()
            .prep(0)
            .h(0) // Superposition - outcome depends on RNG
            .measure(0)
            .build();

        // Run with sequential orchestrator
        let sequential_results = sim_neo(circuit.clone())
            .orchestrator(Orchestrator::Sequential)
            .shots(50)
            .seed(42)
            .run();

        // Run with parallel Monte Carlo orchestrator
        let parallel_results = sim_neo(circuit)
            .orchestrator(Orchestrator::monte_carlo(4))
            .shots(50)
            .seed(42)
            .run();

        // Results should be identical
        assert_eq!(sequential_results.outcomes.len(), parallel_results.outcomes.len());
        for (i, (seq, par)) in sequential_results
            .outcomes
            .iter()
            .zip(parallel_results.outcomes.iter())
            .enumerate()
        {
            assert_eq!(
                seq.get_bit(QubitId(0)),
                par.get_bit(QubitId(0)),
                "Sequential and parallel should produce identical results at shot {i}"
            );
        }
    }

    #[test]
    fn test_sim_neo_quantum_sparse_stab() {
        // Test explicitly selecting sparse stabilizer backend
        use super::sparse_stab;

        let circuit = CommandBuilder::new()
            .prep(0)
            .x(0)
            .measure(0)
            .build();

        let results = sim_neo(circuit)
            .quantum(sparse_stab())
            .shots(10)
            .seed(42)
            .run();

        assert_eq!(results.len(), 10);

        for outcome in &results.outcomes {
            assert!(
                outcome.get_bit(QubitId(0)).unwrap(),
                "X gate should produce |1>"
            );
        }
    }

    #[test]
    fn test_sim_neo_quantum_state_vector() {
        // Test state vector backend
        use super::state_vector;

        let circuit = CommandBuilder::new()
            .prep(0)
            .x(0)
            .measure(0)
            .build();

        let results = sim_neo(circuit)
            .quantum(state_vector())
            .shots(10)
            .seed(42)
            .run();

        assert_eq!(results.len(), 10);

        for outcome in &results.outcomes {
            assert!(
                outcome.get_bit(QubitId(0)).unwrap(),
                "X gate should produce |1>"
            );
        }
    }

    #[test]
    fn test_sim_neo_quantum_backends_deterministic() {
        // Test that each backend is internally deterministic (same seed = same results)
        use super::{sparse_stab, state_vector};

        let circuit = CommandBuilder::new()
            .prep(0)
            .h(0)
            .measure(0)
            .build();

        // Test sparse_stab determinism
        let sparse1 = sim_neo(circuit.clone())
            .quantum(sparse_stab())
            .shots(20)
            .seed(42)
            .run();

        let sparse2 = sim_neo(circuit.clone())
            .quantum(sparse_stab())
            .shots(20)
            .seed(42)
            .run();

        for (o1, o2) in sparse1.outcomes.iter().zip(sparse2.outcomes.iter()) {
            assert_eq!(
                o1.get_bit(QubitId(0)),
                o2.get_bit(QubitId(0)),
                "SparseStab should be deterministic with same seed"
            );
        }

        // Test state_vector determinism
        let sv1 = sim_neo(circuit.clone())
            .quantum(state_vector())
            .shots(20)
            .seed(42)
            .run();

        let sv2 = sim_neo(circuit)
            .quantum(state_vector())
            .shots(20)
            .seed(42)
            .run();

        for (o1, o2) in sv1.outcomes.iter().zip(sv2.outcomes.iter()) {
            assert_eq!(
                o1.get_bit(QubitId(0)),
                o2.get_bit(QubitId(0)),
                "StateVec should be deterministic with same seed"
            );
        }
    }

    #[test]
    fn test_sim_neo_state_vector_parallel() {
        // Test state vector with parallel Monte Carlo
        use super::state_vector;

        let circuit = CommandBuilder::new()
            .prep(0)
            .x(0)
            .measure(0)
            .build();

        let results = sim_neo(circuit)
            .quantum(state_vector())
            .workers(4)
            .shots(100)
            .seed(42)
            .run();

        assert_eq!(results.len(), 100);

        for outcome in &results.outcomes {
            assert!(
                outcome.get_bit(QubitId(0)).unwrap(),
                "X gate should produce |1>"
            );
        }
    }
}
