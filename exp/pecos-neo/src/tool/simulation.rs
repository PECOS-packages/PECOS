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
//! ```no_run
//! use pecos_neo::tool::{monte_carlo, sim_neo};
//! use pecos_neo::prelude::*;
//!
//! let circuit = CommandBuilder::new()
//!     .pz(&[0]).h(&[0]).mz(&[0])
//!     .build();
//!
//! let results = sim_neo(circuit).auto()
//!     .depolarizing(0.01)
//!     .sampling(monte_carlo(1000))
//!     .seed(42)
//!     .build()
//!     .run();
//! ```
//!
//! ## QASM Programs
//!
//! For QASM programs with classical control flow:
//!
//! ```no_run
//! use pecos_neo::tool::{monte_carlo, sim_neo};
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
//! // Pass QASM source, then set the engine
//! let results = sim_neo(qasm).auto()
//!     .classical(qasm_engine())
//!     .depolarizing(0.01)
//!     .sampling(monte_carlo(1000))
//!     .seed(42)
//!     .build()
//!     .run();
//! ```
//!
//! ## Other Program Types
//!
//! Any `ClassicalControlEngineBuilder` works with `sim_neo()`:
//!
//! ```text
//! use pecos_neo::tool::{monte_carlo, sim_neo};
//! use pecos_hugr::hugr_engine;
//! use pecos_qis::qis_engine;
//!
//! // HUGR programs
//! let results = sim_neo(hugr_engine().hugr(&hugr_module)).auto()
//!     .sampling(monte_carlo(1000))
//!     .build()
//!     .run();
//!
//! // QIS programs
//! let results = sim_neo(qis_engine().qis(&qis_program)).auto()
//!     .sampling(monte_carlo(1000))
//!     .build()
//!     .run();
//! ```
//!
//! ## Reusable Simulations
//!
//! Build once, run multiple times:
//!
//! ```no_run
//! use pecos_neo::tool::{monte_carlo, sim_neo};
//! use pecos_neo::prelude::*;
//!
//! let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
//! let mut sim = sim_neo(circuit).auto()
//!     .sampling(monte_carlo(1000))
//!     .build();
//!
//! let results1 = sim.run();
//! let results2 = sim.seed(123).run();  // Different seed
//! let results3 = sim.shots(5000).run(); // More shots
//! ```

use crate::command::CommandQueue;
use crate::extensible::GateDefinitions;
use crate::noise::ComposableNoiseModel;
use crate::outcome::{MeasurementOutcomes, RegisterMap};
use crate::program::{CommandSource, DynProgramRunner, ProgramRunner, StaticProgram};
use crate::runner::{EventHandlers, GateOverrides};
use crate::sampling::importance_runner::ImportanceSamplingRunner;
use crate::sampling::path::{PathEnumerator, PathExplorer};
use crate::sampling::subset::{SubsetConfig, SubsetResult, SubsetSimulation};
use pecos_core::rng::RngManageable;
use pecos_core::rng::rng_manageable::derive_seed;
use pecos_random::PecosRng;
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, SparseStab, Stabilizer, StateVec,
};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::resource::Resources;
use super::{Plugin, Stage, Tool};

// --- Quantum Backend Builders (builder-of-builders pattern) ---

/// Configuration for a quantum backend, stored as data in the builder.
///
/// This enum represents the choice of quantum simulator. The actual simulator
/// is constructed at build time, following the builder-of-builders pattern.
///
/// There is no default: select a backend explicitly via
/// [`SimNeoBuilder::quantum()`](SimNeoBuilder::quantum), or call
/// [`SimNeoBuilder::auto()`](SimNeoBuilder::auto) to opt into automatic
/// selection (currently `SparseStab`).
pub enum QuantumBackend {
    /// Sparse stabilizer simulator.
    ///
    /// Efficient for Clifford circuits and QEC simulations.
    /// Only supports Clifford gates (H, S, CNOT, CZ, etc.).
    /// This is what `.auto()` selects.
    SparseStab,

    /// Public stabilizer simulator.
    ///
    /// Uses PECOS's stable stabilizer simulator interface while preserving
    /// Clifford-only semantics.
    Stabilizer,

    /// State vector simulator.
    ///
    /// Supports arbitrary gates including non-Clifford (T, rotations).
    /// Memory scales as 2^n for n qubits.
    StateVec,

    /// Adapted `pecos-engines` quantum-engine builder.
    ///
    /// This path uses `QuantumEngineProgramRunner` to execute `sim_neo` command
    /// batches through the gate-by-gate `QuantumEngine` protocol.
    AdaptedQuantumEngine(Box<dyn AdaptedQuantumEngineFactory>),

    /// Custom simulator backend via factory function.
    ///
    /// Allows any simulator implementing `CliffordGateable + RngManageable<Rng = PecosRng>`
    /// to be used through the `sim_neo().auto()` API. Use [`custom_backend()`] to create.
    ///
    /// The factory is invoked once per worker, so parallel Monte Carlo
    /// works like the built-in backends (per-shot seeding from global shot
    /// indices keeps results identical for any worker count).
    Custom(Arc<dyn SimulatorFactory>),
}

impl std::fmt::Debug for QuantumBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SparseStab => write!(f, "SparseStab"),
            Self::Stabilizer => write!(f, "Stabilizer"),
            Self::StateVec => write!(f, "StateVec"),
            Self::AdaptedQuantumEngine(_) => write!(f, "AdaptedQuantumEngine(...)"),
            Self::Custom(_) => write!(f, "Custom(...)"),
        }
    }
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

/// Builder for the public stabilizer backend configuration.
///
/// Currently a simple marker type; future versions may add configuration
/// options while preserving the stable simulator interface.
#[derive(Debug, Clone, Default)]
pub struct StabilizerBuilder;

impl StabilizerBuilder {
    /// Create a new stabilizer builder.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl From<StabilizerBuilder> for QuantumBackend {
    fn from(_: StabilizerBuilder) -> Self {
        QuantumBackend::Stabilizer
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
/// The sparse stabilizer is the backend `.auto()` selects, efficient for Clifford circuits
/// and quantum error correction simulations.
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo, sparse_stab};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
/// let results = sim_neo(circuit)
///     .quantum(sparse_stab())
///     .sampling(monte_carlo(1000))
///     .build()
///     .run();
/// ```
#[must_use]
pub fn sparse_stab() -> SparseStabBuilder {
    SparseStabBuilder::new()
}

/// Create a stabilizer backend builder.
///
/// This is the stable public stabilizer backend for Clifford circuits. Use
/// [`sparse_stab()`] when you specifically want the current sparse-tableau
/// implementation.
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo, stabilizer};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
/// let results = sim_neo(circuit)
///     .quantum(stabilizer())
///     .sampling(monte_carlo(1000))
///     .build()
///     .run();
/// ```
#[must_use]
pub fn stabilizer() -> StabilizerBuilder {
    StabilizerBuilder::new()
}

/// Create a state vector backend builder.
///
/// The state vector simulator supports arbitrary gates including non-Clifford
/// operations like T gates and arbitrary rotations.
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo, state_vector};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
/// let results = sim_neo(circuit)
///     .quantum(state_vector())
///     .sampling(monte_carlo(1000))
///     .build()
///     .run();
/// ```
#[must_use]
pub fn state_vector() -> StateVecBuilder {
    StateVecBuilder::new()
}

// --- Custom Backend Support ---

/// Factory for creating simulator instances.
///
/// This trait allows custom simulators to be used with `sim_neo()` by providing
/// a way to create new instances. A blanket implementation is provided for
/// closures `Fn(usize) -> S` where `S: CliffordGateable + RngManageable<Rng = PecosRng>`,
/// so most users should prefer [`custom_backend()`] over implementing this directly.
///
/// Implement this directly only for advanced use cases (e.g., simulators that
/// need custom noise injection or seed handling).
pub trait SimulatorFactory: Send + Sync {
    /// Short diagnostic label for error messages.
    ///
    /// This is not used for dispatch; execution is selected by the trait
    /// object itself. The label only keeps unsupported-configuration errors
    /// readable after type erasure.
    fn diagnostic_label(&self) -> &'static str {
        "custom backend"
    }

    /// Create a program runner for the given number of qubits.
    ///
    /// Called once during simulation startup. The returned runner handles
    /// all shots for the simulation.
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits inferred from the circuit
    /// * `noise` - The noise model (if configured via `.noise()` or `.depolarizing()`)
    /// * `seed` - The base seed (if configured via `.seed()`)
    fn create_runner(
        &self,
        num_qubits: usize,
        noise: Option<ComposableNoiseModel>,
        seed: Option<u64>,
    ) -> Box<dyn DynProgramRunner>;
}
#[doc(hidden)]
pub trait AdaptedQuantumEngineFactory: Send + Sync {
    fn create_runner(&self, num_qubits: usize, seed: Option<u64>) -> Box<dyn DynProgramRunner>;

    fn create_parallel_runner_factory(
        &self,
        num_qubits: usize,
    ) -> Box<dyn ParallelQuantumRunnerFactory>;
}
struct QuantumEngineSimulatorFactory<B>
where
    B: pecos_engines::QuantumEngineBuilder + Clone + 'static,
{
    builder: B,
}
impl<B> AdaptedQuantumEngineFactory for QuantumEngineSimulatorFactory<B>
where
    B: pecos_engines::QuantumEngineBuilder + Clone + 'static,
{
    fn create_runner(&self, num_qubits: usize, seed: Option<u64>) -> Box<dyn DynProgramRunner> {
        let mut builder = self.builder.clone();
        builder.set_qubits_if_needed(num_qubits);
        let mut engine = builder
            .build()
            .expect("Failed to build quantum engine backend");
        if let Some(seed) = seed {
            engine.set_seed(seed);
        }
        Box::new(crate::adapter::QuantumEngineProgramRunner::new(engine))
    }

    fn create_parallel_runner_factory(
        &self,
        num_qubits: usize,
    ) -> Box<dyn ParallelQuantumRunnerFactory> {
        Box::new(AdaptedQuantumEngineRunnerFactory {
            builder: self.builder.clone(),
            num_qubits,
        })
    }
}
impl<B> From<B> for QuantumBackend
where
    B: pecos_engines::IntoQuantumEngineBuilder + 'static,
    B::Builder: Clone + 'static,
{
    fn from(builder: B) -> Self {
        QuantumBackend::AdaptedQuantumEngine(Box::new(QuantumEngineSimulatorFactory {
            builder: builder.into_quantum_engine_builder(),
        }))
    }
}

/// Blanket implementation for closures that create simulators.
///
/// This allows `custom_backend(|n| MySimulator::new(n))` to work.
impl<S, F> SimulatorFactory for F
where
    S: CliffordGateable + RngManageable<Rng = PecosRng> + Send + Sync + 'static,
    F: Fn(usize) -> S + Send + Sync,
{
    fn create_runner(
        &self,
        num_qubits: usize,
        noise: Option<ComposableNoiseModel>,
        seed: Option<u64>,
    ) -> Box<dyn DynProgramRunner> {
        let sim = (self)(num_qubits);
        let mut runner = ProgramRunner::new(sim);
        if let Some(n) = noise {
            runner = runner.with_noise(n);
        }
        if let Some(s) = seed {
            runner = runner.with_seed(s);
        }
        Box::new(runner)
    }
}

/// Builder for custom simulator backends.
///
/// Created via [`custom_backend()`]. Converts into [`QuantumBackend::Custom`].
pub struct CustomBackendBuilder {
    factory: Arc<dyn SimulatorFactory>,
}

impl From<CustomBackendBuilder> for QuantumBackend {
    fn from(builder: CustomBackendBuilder) -> Self {
        QuantumBackend::Custom(builder.factory)
    }
}

/// Create a custom backend from a factory closure.
///
/// This allows any simulator implementing `CliffordGateable + RngManageable<Rng = PecosRng>`
/// to be used through `sim_neo().auto()`. The closure receives the number of qubits
/// and should return a new simulator instance.
///
/// The factory is invoked once per worker for parallel Monte Carlo, so
/// `.workers(n)` works like the built-in backends. (Importance sampling
/// always runs on its internal sparse stabilizer and ignores the backend
/// choice.)
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo, custom_backend};
/// use pecos_neo::prelude::*;
/// use pecos_simulators::SparseStab;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
///
/// // Use a custom simulator backend
/// let results = sim_neo(circuit)
///     .quantum(custom_backend(|n| SparseStab::new(n)))
///     .sampling(monte_carlo(100))
///     .seed(42)
///     .build()
///     .run();
/// ```
#[must_use]
pub fn custom_backend<S, F>(factory: F) -> CustomBackendBuilder
where
    S: CliffordGateable + RngManageable<Rng = PecosRng> + Send + Sync + 'static,
    F: Fn(usize) -> S + Send + Sync + 'static,
{
    CustomBackendBuilder {
        factory: Arc::new(factory),
    }
}

/// Create a custom backend from a `SimulatorFactory` implementation.
///
/// Unlike [`custom_backend()`] which takes a closure, this accepts any type
/// implementing `SimulatorFactory` directly. Use this when the factory needs
/// configuration state (e.g., `StabMpsBackend` with bond dimension settings).
#[must_use]
pub fn custom_backend_from_factory(
    factory: impl SimulatorFactory + 'static,
) -> CustomBackendBuilder {
    CustomBackendBuilder {
        factory: Arc::new(factory),
    }
}

/// Create a custom backend with rotation support from a factory closure.
///
/// Like [`custom_backend()`], but enables rotation gates (T, RZ, etc.) for
/// simulators implementing `ArbitraryRotationGateable`. Use this instead of
/// `custom_backend()` when your simulator supports non-Clifford gates.
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo, custom_backend_with_rotations};
/// use pecos_neo::prelude::*;
/// use pecos_simulators::StateVec;
///
/// let circuit = CommandBuilder::new().pz(&[0]).t(&[0]).mz(&[0]).build();
///
/// let results = sim_neo(circuit)
///     .quantum(custom_backend_with_rotations(|n| StateVec::new(n)))
///     .sampling(monte_carlo(100))
///     .seed(42)
///     .build()
///     .run();
/// ```
#[must_use]
pub fn custom_backend_with_rotations<S, F>(factory: F) -> CustomBackendBuilder
where
    S: CliffordGateable
        + ArbitraryRotationGateable
        + RngManageable<Rng = PecosRng>
        + Send
        + Sync
        + 'static,
    F: Fn(usize) -> S + Send + Sync + 'static,
{
    CustomBackendBuilder {
        factory: Arc::new(RotationSimulatorFactory(factory)),
    }
}

/// Factory wrapper that creates rotation-enabled runners.
struct RotationSimulatorFactory<F>(F);

impl<S, F> SimulatorFactory for RotationSimulatorFactory<F>
where
    S: CliffordGateable
        + ArbitraryRotationGateable
        + RngManageable<Rng = PecosRng>
        + Send
        + Sync
        + 'static,
    F: Fn(usize) -> S + Send + Sync,
{
    fn create_runner(
        &self,
        num_qubits: usize,
        noise: Option<ComposableNoiseModel>,
        seed: Option<u64>,
    ) -> Box<dyn DynProgramRunner> {
        let sim = (self.0)(num_qubits);
        let mut runner = ProgramRunner::rotations(sim);
        if let Some(n) = noise {
            runner = runner.with_noise(n);
        }
        if let Some(s) = seed {
            runner = runner.with_seed(s);
        }
        Box::new(runner)
    }
}

// --- SimNeoInput Trait ---

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
/// ```no_run
/// use pecos_neo::tool::{SimNeoInput, SimNeoBuilder};
///
/// struct MyProgramType;
///
/// impl SimNeoInput for MyProgramType {
///     fn into_sim_neo_builder(self) -> SimNeoBuilder {
///         // Convert to SimNeoBuilder
///         SimNeoBuilder::empty()
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

/// Implementation for boxed dynamic command sources.
impl SimNeoInput for Box<dyn CommandSource + Send + Sync> {
    fn into_sim_neo_builder(self) -> SimNeoBuilder {
        SimNeoBuilder::with_command_source(self)
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
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo};
/// use pecos_qasm::qasm_engine;
///
/// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];";
/// sim_neo(qasm_code).auto()
///     .classical(qasm_engine())
///     .sampling(monte_carlo(1000))
///     .build()
///     .run();
/// ```
impl SimNeoInput for &str {
    fn into_sim_neo_builder(self) -> SimNeoBuilder {
        SimNeoBuilder::with_program_source(self.to_string())
    }
}

/// Implementation for `String` (program source code).
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
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo};
/// use pecos_programs::Qasm;
/// use pecos_qasm::qasm_engine;
///
/// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];".to_string();
///
/// // Auto mode - uses qasm_engine() automatically
/// sim_neo(Qasm::from_string(qasm_code.clone()))
///     .auto()
///     .sampling(monte_carlo(1000))
///     .build()
///     .run();
///
/// // Explicit mode
/// sim_neo(Qasm::from_string(qasm_code)).auto()
///     .classical(qasm_engine())
///     .sampling(monte_carlo(1000))
///     .build()
///     .run();
/// ```
impl SimNeoInput for pecos_programs::Qasm {
    fn into_sim_neo_builder(self) -> SimNeoBuilder {
        SimNeoBuilder::with_typed_program(TypedProgram::Qasm(self))
    }
}

/// Implementation for HUGR programs.
///
/// Use `.auto()` to automatically select the HUGR interpreter engine:
///
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo};
/// use pecos_programs::Hugr;
///
/// let hugr = Hugr::from_file("program.hugr").unwrap();
/// sim_neo(hugr)
///     .auto()
///     .sampling(monte_carlo(1000))
///     .build()
///     .run();
/// ```
impl SimNeoInput for pecos_programs::Hugr {
    fn into_sim_neo_builder(self) -> SimNeoBuilder {
        SimNeoBuilder::with_typed_program(TypedProgram::Hugr(self))
    }
}

/// Implementation for the unified `Program` enum.
///
/// Use `.auto()` to automatically select the appropriate engine based on
/// the program type:
///
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo};
/// use pecos_programs::{Program, Qasm};
///
/// let qasm = Qasm::from_string("OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];".to_string());
/// sim_neo(Program::Qasm(qasm))
///     .auto()
///     .sampling(monte_carlo(1000))
///     .build()
///     .run();
/// ```
impl SimNeoInput for pecos_programs::Program {
    fn into_sim_neo_builder(self) -> SimNeoBuilder {
        let typed = match self {
            pecos_programs::Program::Qasm(p) => TypedProgram::Qasm(p),
            pecos_programs::Program::Hugr(p) => TypedProgram::Hugr(p),
            _ => TypedProgram::Unsupported(self.program_type().to_string()),
        };
        SimNeoBuilder::with_typed_program(typed)
    }
}

// --- Resources ---

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

/// Builder for importance sampling configuration.
///
/// Specifies the shot count, true error rates, and boost factor for biased
/// sampling. Use the [`importance_sampling()`] function to create an instance.
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{sim_neo, importance_sampling};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
/// let results = sim_neo(circuit).auto()
///     .sampling(importance_sampling(10000)
///         .with_p1(0.001)
///         .with_p2(0.01)
///         .with_p_meas(0.001)
///         .with_boost(10.0))
///     .build()
///     .run();
/// ```
#[derive(Debug, Clone)]
pub struct ImportanceSamplingBuilder {
    /// Number of (boosted) trials to run.
    shots: usize,
    /// Number of parallel workers (1 = sequential).
    workers: usize,
    /// Single-qubit gate error rate (true distribution).
    p1: f64,
    /// Two-qubit gate error rate (true distribution).
    p2: f64,
    /// Measurement error rate (true distribution).
    p_meas: f64,
    /// Boost factor for proposal distribution.
    boost: f64,
}

impl ImportanceSamplingBuilder {
    /// Create a new importance sampling builder running `shots` trials.
    ///
    /// Default rates: p1=0.001, p2=0.01, `p_meas=0.001`, boost=10.0
    #[must_use]
    pub fn new(shots: usize) -> Self {
        Self {
            shots,
            workers: 1,
            p1: 0.001,
            p2: 0.01,
            p_meas: 0.001,
            boost: 10.0,
        }
    }

    /// Set the number of parallel workers (1 = sequential).
    ///
    /// Trials are seeded per global shot index, so results are identical
    /// for any worker count.
    #[must_use]
    pub fn workers(mut self, workers: usize) -> Self {
        self.workers = workers;
        self
    }

    /// Set the worker count from available parallelism.
    #[must_use]
    pub fn auto_workers(mut self) -> Self {
        self.workers = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        self
    }

    /// Set the single-qubit gate error rate.
    #[must_use]
    pub fn with_p1(mut self, p: f64) -> Self {
        self.p1 = p;
        self
    }

    /// Set the two-qubit gate error rate.
    #[must_use]
    pub fn with_p2(mut self, p: f64) -> Self {
        self.p2 = p;
        self
    }

    /// Set the measurement error rate.
    #[must_use]
    pub fn with_p_meas(mut self, p: f64) -> Self {
        self.p_meas = p;
        self
    }

    /// Set all error rates to the same value.
    #[must_use]
    pub fn with_uniform_error(mut self, p: f64) -> Self {
        self.p1 = p;
        self.p2 = p;
        self.p_meas = p;
        self
    }

    /// Set the boost factor for the proposal distribution.
    ///
    /// The proposal distribution samples errors at rate `p * boost`,
    /// capped at 50%.
    #[must_use]
    pub fn with_boost(mut self, boost: f64) -> Self {
        self.boost = boost;
        self
    }

    /// Build the sampling strategy.
    #[must_use]
    pub fn build(self) -> Sampling {
        Sampling::ImportanceSampling { config: self }
    }

    /// Get the single-qubit error rate.
    #[must_use]
    pub fn p1(&self) -> f64 {
        self.p1
    }

    /// Get the two-qubit error rate.
    #[must_use]
    pub fn p2(&self) -> f64 {
        self.p2
    }

    /// Get the measurement error rate.
    #[must_use]
    pub fn p_meas(&self) -> f64 {
        self.p_meas
    }

    /// Get the boost factor.
    #[must_use]
    pub fn boost(&self) -> f64 {
        self.boost
    }

    /// Get the number of (boosted) trials to run.
    #[must_use]
    pub fn shots(&self) -> usize {
        self.shots
    }
}

impl From<ImportanceSamplingBuilder> for Sampling {
    fn from(builder: ImportanceSamplingBuilder) -> Self {
        builder.build()
    }
}

/// Create an importance sampling strategy builder running `shots` trials.
///
/// Importance sampling biases noise toward higher error rates to observe
/// rare events more frequently, then reweights results for unbiased estimates.
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{sim_neo, importance_sampling};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
/// let results = sim_neo(circuit).auto()
///     .sampling(importance_sampling(10000)
///         .with_p1(0.001)
///         .with_p2(0.01)
///         .with_boost(10.0))
///     .build()
///     .run();
///
/// // Compute weighted statistics
/// if let Some(rate) = results.weighted_mean(|outcome| {
///     // Replace with actual logical error check
///     0.0
/// }) {
///     println!("Estimated error rate: {:.2e}", rate);
/// }
/// ```
#[must_use]
pub fn importance_sampling(shots: usize) -> ImportanceSamplingBuilder {
    ImportanceSamplingBuilder::new(shots)
}

// The Monte Carlo sampling vocabulary (`monte_carlo()` + `MonteCarloBuilder`)
// is shared with the engines stack and lives in `pecos-engines` so BOTH stacks
// accept the same `monte_carlo(n).workers(m)` spelling (the engines
// `MonteCarloEngine`'s run-spec). Re-exported here so `pecos_neo::tool::*`
// paths are unchanged; neo maps it into its own `Sampling` strategy below.
pub use pecos_engines::sampling::{MonteCarloBuilder, monte_carlo};

impl From<MonteCarloBuilder> for Sampling {
    fn from(builder: MonteCarloBuilder) -> Self {
        Sampling::MonteCarlo {
            shots: builder.shots(),
            workers: builder.resolved_workers(),
        }
    }
}

/// Builder for the path enumeration sampling strategy.
///
/// Created by [`path_enumeration()`].
#[derive(Debug, Clone)]
pub struct PathEnumerationBuilder {
    max_measurements: usize,
}

impl From<PathEnumerationBuilder> for Sampling {
    fn from(builder: PathEnumerationBuilder) -> Self {
        Sampling::PathEnumeration { config: builder }
    }
}

/// Create a path enumeration strategy covering up to `max_measurements`
/// random measurement branches.
///
/// Instead of sampling, this systematically enumerates the measurement
/// branches of a (noiseless) Clifford circuit: every random measurement
/// splits the execution into two equal-probability paths. Each distinct
/// realized path becomes one entry in `SimulationResults::outcomes`, with
/// its exact probability in `SimulationResults::weights`
/// (`2^-{number of random measurements}` along that path; deterministic
/// measurements do not branch).
///
/// If `max_measurements` covers every random measurement in the circuit,
/// the enumeration is complete and the weights sum to 1. If the circuit
/// has more random measurements than `max_measurements`, uncovered
/// branches default to outcome 0 and the weights sum to less than 1.
///
/// Requires a static circuit on the `sparse_stab()` backend with no
/// `.noise()` (noise makes branching stochastic beyond measurements);
/// checked at `.build()`. The rerun override `Simulation::shots()` has no
/// effect on enumeration size.
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{path_enumeration, sim_neo, sparse_stab};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new()
///     .pz(&[0, 1])
///     .h(&[0])
///     .cx(&[(0, 1)])
///     .mz(&[0, 1])
///     .build();
///
/// let results = sim_neo(circuit)
///     .quantum(sparse_stab())
///     .sampling(path_enumeration(1))
///     .run();
///
/// // Two paths (00 and 11), each with probability 0.5.
/// for (outcome, weight) in results
///     .outcomes
///     .iter()
///     .zip(results.weights.as_ref().unwrap())
/// {
///     println!("p = {:.3}: {:?}", weight.weight(), outcome);
/// }
/// ```
#[must_use]
pub fn path_enumeration(max_measurements: usize) -> PathEnumerationBuilder {
    PathEnumerationBuilder { max_measurements }
}

/// Score function for subset simulation: how "close" an outcome is to the
/// failure event (higher = closer).
pub type SubsetScoreFn = Arc<dyn Fn(&MeasurementOutcomes) -> f64 + Send + Sync>;

/// Failure predicate for subset simulation: did this outcome reach the
/// rare event?
pub type SubsetFailureFn = Arc<dyn Fn(&MeasurementOutcomes) -> bool + Send + Sync>;

/// Builder for the subset simulation sampling strategy.
///
/// Created by [`subset_simulation()`]. `samples_per_level` is the defining
/// argument; `.score()` and `.failure()` are required (there is no sensible
/// default for either), checked at `.build()`.
#[derive(Clone)]
pub struct SubsetSimulationBuilder {
    samples_per_level: usize,
    threshold_fraction: f64,
    max_levels: usize,
    min_conditional_prob: f64,
    /// Explicit acknowledgment that the multi-level estimator is biased.
    /// Required before `max_levels > 1` is allowed (see
    /// [`SubsetSimulationBuilder::allow_biased_multilevel`]).
    allow_biased_multilevel: bool,
    score: Option<SubsetScoreFn>,
    failure: Option<SubsetFailureFn>,
}

impl std::fmt::Debug for SubsetSimulationBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubsetSimulationBuilder")
            .field("samples_per_level", &self.samples_per_level)
            .field("threshold_fraction", &self.threshold_fraction)
            .field("max_levels", &self.max_levels)
            .field("allow_biased_multilevel", &self.allow_biased_multilevel)
            .field("min_conditional_prob", &self.min_conditional_prob)
            .field("score", &self.score.as_ref().map(|_| "Fn(..) -> f64"))
            .field("failure", &self.failure.as_ref().map(|_| "Fn(..) -> bool"))
            .finish()
    }
}

impl SubsetSimulationBuilder {
    /// Set the score function: how "close" is this outcome to failure?
    ///
    /// Higher scores advance to the next level. The score must be
    /// consistent with `.failure()`: failing outcomes should score at
    /// least as high as any non-failing outcome.
    #[must_use]
    pub fn score<F>(mut self, score: F) -> Self
    where
        F: Fn(&MeasurementOutcomes) -> f64 + Send + Sync + 'static,
    {
        self.score = Some(Arc::new(score));
        self
    }

    /// Set the failure predicate: did this outcome reach the rare event?
    #[must_use]
    pub fn failure<F>(mut self, failure: F) -> Self
    where
        F: Fn(&MeasurementOutcomes) -> bool + Send + Sync + 'static,
    {
        self.failure = Some(Arc::new(failure));
        self
    }

    /// Set the fraction of samples that advances past each threshold
    /// (typically 0.1-0.2; default 0.1).
    #[must_use]
    pub fn threshold_fraction(mut self, fraction: f64) -> Self {
        self.threshold_fraction = fraction;
        self
    }

    /// Set the maximum number of levels before giving up.
    ///
    /// Defaults to 1 (a single level, which is an unbiased direct-Monte-
    /// Carlo estimate of the failure fraction). Setting more than one level
    /// engages the multi-level estimator, which is currently BIASED upward
    /// (the resample is unconditioned; see the `sampling::subset` module
    /// docs). Requesting `levels > 1` therefore also requires an explicit
    /// [`Self::allow_biased_multilevel`] acknowledgment, or `.run()` fails
    /// at build time.
    #[must_use]
    pub fn max_levels(mut self, levels: usize) -> Self {
        self.max_levels = levels;
        self
    }

    /// Acknowledge and accept the known upward bias of the multi-level
    /// subset estimator, enabling `max_levels > 1`.
    ///
    /// Without this, [`subset_simulation`] runs a single unbiased level
    /// (direct Monte Carlo). The multi-level path provides rare-event
    /// reach but is currently biased (see the `sampling::subset` module
    /// docs and the estimator-overhaul follow-up); call this only when an
    /// approximate, biased estimate is acceptable.
    #[must_use]
    pub fn allow_biased_multilevel(mut self) -> Self {
        self.allow_biased_multilevel = true;
        self
    }

    /// Set the minimum conditional probability before declaring the
    /// failure event unreachable (default 1e-6).
    #[must_use]
    pub fn min_conditional_prob(mut self, p: f64) -> Self {
        self.min_conditional_prob = p;
        self
    }
}

impl From<SubsetSimulationBuilder> for Sampling {
    fn from(builder: SubsetSimulationBuilder) -> Self {
        Sampling::SubsetSimulation { config: builder }
    }
}

/// Create a subset simulation strategy builder running `samples_per_level`
/// samples at each level.
///
/// Subset simulation estimates rare event probabilities by decomposing
/// them into a product of conditional probabilities across adaptive
/// levels. It needs a `.score()` function (how close an outcome is to
/// failure) and a `.failure()` predicate (did the rare event occur);
/// both are required.
///
/// Accuracy caveat: the current multi-level estimator biases upward once
/// more than one level engages (see `sampling::subset` module docs);
/// treat deep-rare-event estimates as approximate until the estimator
/// overhaul lands.
///
/// The result arrives in [`SimulationResults::subset`]; per-shot
/// `outcomes` are empty for subset runs.
///
/// Currently supports static circuits on the `sparse_stab()` backend only.
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{sim_neo, sparse_stab, subset_simulation};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new()
///     .pz(&[0, 1, 2])
///     .h(&[0, 1, 2])
///     .mz(&[0, 1, 2])
///     .build();
///
/// let results = sim_neo(circuit)
///     .quantum(sparse_stab())
///     .sampling(
///         subset_simulation(1000)
///             .score(|o| o.iter().filter(|m| m.outcome).count() as f64)
///             .failure(|o| o.iter().all(|m| m.outcome)),
///     )
///     .seed(42)
///     .run();
///
/// let subset = results.subset.expect("subset strategy produces an estimate");
/// println!("P(failure) = {:.2e}", subset.probability());
/// ```
#[must_use]
pub fn subset_simulation(samples_per_level: usize) -> SubsetSimulationBuilder {
    let defaults = SubsetConfig::default();
    SubsetSimulationBuilder {
        samples_per_level,
        threshold_fraction: defaults.threshold_fraction,
        // Default to a single, unbiased level. The multi-level estimator is
        // biased upward, so engaging it (`max_levels > 1`) requires an
        // explicit `.allow_biased_multilevel()` acknowledgment.
        max_levels: 1,
        min_conditional_prob: defaults.min_conditional_prob,
        allow_biased_multilevel: false,
        score: None,
        failure: None,
    }
}

/// Sampling strategy for simulation execution.
///
/// This enum defines how shots are executed. Different strategies offer
/// trade-offs between simplicity, parallelism, and specialized sampling.
///
/// Stored as data in the builder, the actual execution is set up at run time.
///
/// Construct via the builder functions [`monte_carlo()`] and
/// [`importance_sampling()`] and pass to
/// [`SimNeoBuilder::sampling()`](SimNeoBuilder::sampling). There is no
/// default: the shot count is part of the strategy and must be explicit.
#[derive(Clone)]
pub enum Sampling {
    /// Monte Carlo execution (sequential with 1 worker, parallel with >1).
    ///
    /// Each worker runs a batch of shots independently with deterministic seeding.
    /// Supports both noiseless and noisy circuits (noise model is cloned per worker).
    /// With 1 worker, runs via the Tool's schedule directly.
    ///
    /// Use the [`monte_carlo()`] builder function to create this variant.
    MonteCarlo {
        /// Number of shots to run.
        shots: usize,
        /// Number of parallel workers (1 = sequential).
        workers: usize,
    },

    /// Importance sampling for rare event estimation.
    ///
    /// Biases sampling toward rare events and reweights results.
    /// Use when estimating probabilities of rare outcomes (~1e-3 to 1e-6).
    ///
    /// Use the [`importance_sampling()`] builder function to create this variant.
    ImportanceSampling {
        /// Configuration for importance sampling.
        config: ImportanceSamplingBuilder,
    },

    /// Subset simulation: decompose a rare-event probability into a product
    /// of conditional probabilities across adaptive levels. Produces an
    /// estimate in [`SimulationResults::subset`] instead of per-shot
    /// outcomes.
    ///
    /// Accuracy caveat: the current multi-level estimator biases upward
    /// once more than one level engages (see the `sampling::subset` module
    /// docs); treat deep-rare-event estimates as approximate until the
    /// estimator overhaul lands. Single-level runs behave like direct
    /// Monte Carlo and are unbiased.
    ///
    /// Use the [`subset_simulation()`] builder function to create this variant.
    SubsetSimulation {
        /// Configuration for subset simulation.
        config: SubsetSimulationBuilder,
    },

    /// Exhaustive enumeration of measurement branches (noiseless circuits).
    ///
    /// Each distinct realized path becomes one outcome entry with its exact
    /// probability as the weight.
    ///
    /// Use the [`path_enumeration()`] builder function to create this variant.
    PathEnumeration {
        /// Configuration for path enumeration.
        config: PathEnumerationBuilder,
    },
}

impl std::fmt::Debug for Sampling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MonteCarlo { shots, workers } => f
                .debug_struct("MonteCarlo")
                .field("shots", shots)
                .field("workers", workers)
                .finish(),
            Self::ImportanceSampling { config } => f
                .debug_struct("ImportanceSampling")
                .field("config", config)
                .finish(),
            Self::SubsetSimulation { config } => f
                .debug_struct("SubsetSimulation")
                .field("config", config)
                .finish(),
            Self::PathEnumeration { config } => f
                .debug_struct("PathEnumeration")
                .field("config", config)
                .finish(),
        }
    }
}

impl Sampling {
    /// Number of shots/trials this strategy will run.
    ///
    /// For subset simulation this is the samples per level; the total
    /// sample count depends on how many levels the run needs. For path
    /// enumeration this is the number of enumerated forced paths
    /// (`2^max_measurements`); distinct realized paths may be fewer.
    #[must_use]
    pub fn shots(&self) -> usize {
        match self {
            Self::MonteCarlo { shots, .. } => *shots,
            Self::ImportanceSampling { config } => config.shots(),
            Self::SubsetSimulation { config } => config.samples_per_level,
            // Saturate rather than overflow the shift: an out-of-range
            // max_measurements is rejected with a friendly message at
            // build time (the <= 24 path-enumeration assert), and this
            // accessor must not panic with a raw shift overflow first.
            Self::PathEnumeration { config } => u32::try_from(config.max_measurements)
                .ok()
                .and_then(|bits| 1usize.checked_shl(bits))
                .unwrap_or(usize::MAX),
        }
    }
}

/// Accumulated simulation results.
#[derive(Debug, Clone, Default)]
pub struct SimulationResults {
    /// Per-shot measurement outcomes.
    pub outcomes: Vec<MeasurementOutcomes>,
    /// Per-shot importance weights (only for importance sampling).
    pub weights: Option<Vec<crate::sampling::weight::SampleWeight>>,
    /// Rare-event estimate with per-level statistics (only for subset
    /// simulation; `outcomes` is empty for subset runs).
    pub subset: Option<SubsetResult>,
    /// Per-shot named-register results, populated when the program source
    /// produces them (classical engines: QASM cregs, PHIR variables).
    /// None for sources without register data; see
    /// [`to_shot_vec()`](Self::to_shot_vec) to synthesize from outcomes.
    pub shots: Option<pecos_results::ShotVec>,
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
        if let Some(ref mut weights) = self.weights {
            weights.clear();
        }
        self.subset = None;
        self.shots = None;
    }

    /// View the results as a [`pecos_results::ShotVec`].
    ///
    /// Program sources with named-register data (classical engines)
    /// populate [`shots`](Self::shots) directly; that is returned as-is.
    /// Otherwise one [`pecos_results::Shot`] per outcome is synthesized:
    /// with a `register_map`, one `Data::BitVec` register per named
    /// register (bits in the register's qubit order); without one, a
    /// single register named `"meas"` with bits in ascending qubit order.
    #[must_use]
    pub fn to_shot_vec(&self, register_map: Option<&RegisterMap>) -> pecos_results::ShotVec {
        if let Some(shots) = &self.shots {
            return shots.clone();
        }

        let mut shot_vec = pecos_results::ShotVec::new();
        for outcomes in &self.outcomes {
            let mut shot = pecos_results::Shot::default();
            if let Some(map) = register_map {
                for name in map.register_names() {
                    if let Some(bits) = outcomes.register_bitstring(map, name) {
                        let bitstring: String =
                            bits.iter().map(|&b| if b { '1' } else { '0' }).collect();
                        if let Some(data) = pecos_results::Data::from_bitstring(&bitstring) {
                            shot.data.insert(name.to_string(), data);
                        }
                    }
                }
            } else {
                let bitstring: String = outcomes
                    .iter()
                    .map(|o| if o.outcome { '1' } else { '0' })
                    .collect();
                if let Some(data) = pecos_results::Data::from_bitstring(&bitstring) {
                    shot.data.insert("meas".to_string(), data);
                }
            }
            shot_vec.shots.push(shot);
        }
        shot_vec
    }

    /// Check if this result has importance weights.
    #[must_use]
    pub fn has_weights(&self) -> bool {
        self.weights.is_some()
    }

    /// Compute weighted statistics for a binary indicator function.
    ///
    /// Returns `None` if no importance weights are present.
    ///
    /// # Arguments
    /// * `indicator` - Function that returns 1.0 for "success" outcomes, 0.0 otherwise
    #[must_use]
    pub fn weighted_mean<F>(&self, indicator: F) -> Option<f64>
    where
        F: Fn(&MeasurementOutcomes) -> f64,
    {
        let weights = self.weights.as_ref()?;
        if weights.is_empty() {
            return None;
        }

        let mut stats = crate::sampling::weight::WeightedStatistics::new();
        for (outcome, weight) in self.outcomes.iter().zip(weights.iter()) {
            stats.add(indicator(outcome), weight);
        }
        Some(stats.mean())
    }

    /// Compute weighted statistics with standard error.
    ///
    /// Returns `(mean, standard_error)` or `None` if no weights.
    #[must_use]
    pub fn weighted_stats<F>(&self, indicator: F) -> Option<(f64, f64)>
    where
        F: Fn(&MeasurementOutcomes) -> f64,
    {
        let weights = self.weights.as_ref()?;
        if weights.is_empty() {
            return None;
        }

        let mut stats = crate::sampling::weight::WeightedStatistics::new();
        for (outcome, weight) in self.outcomes.iter().zip(weights.iter()) {
            stats.add(indicator(outcome), weight);
        }
        Some((stats.mean(), stats.standard_error()))
    }

    // --- Register-based accessors ---

    /// Convert to columnar format: `register_name` -> `Vec<Vec<bool>>` (one bitstring per shot).
    ///
    /// Registers where any qubit hasn't been measured (in any shot) are omitted.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::tool::{monte_carlo, sim_neo};
    /// use pecos_neo::outcome::RegisterMap;
    /// use pecos_neo::prelude::*;
    ///
    /// let circuit = CommandBuilder::new()
    ///     .pz(&[0]).pz(&[1]).h(&[0]).mz(&[0]).mz(&[1])
    ///     .build();
    ///
    /// let mut reg = RegisterMap::new();
    /// reg.add_register("c", &[QubitId(0), QubitId(1)]);
    ///
    /// let results = sim_neo(circuit).auto().sampling(monte_carlo(100)).seed(42).run();
    /// let columns = results.as_register_columns(&reg);
    /// assert_eq!(columns["c"].len(), 100);
    /// ```
    #[must_use]
    pub fn as_register_columns(&self, register: &RegisterMap) -> BTreeMap<String, Vec<Vec<bool>>> {
        let mut columns: BTreeMap<String, Vec<Vec<bool>>> = BTreeMap::new();

        for name in register.register_names() {
            let mut col = Vec::with_capacity(self.outcomes.len());
            let mut all_valid = true;

            for outcome in &self.outcomes {
                if let Some(bits) = outcome.register_bitstring(register, name) {
                    col.push(bits);
                } else {
                    all_valid = false;
                    break;
                }
            }

            if all_valid {
                columns.insert(name.to_string(), col);
            }
        }

        columns
    }

    /// Count unique bitstring occurrences for a named register.
    ///
    /// Returns a map from bitstring -> count. Returns an empty map if
    /// the register doesn't exist or any qubit hasn't been measured.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::tool::{monte_carlo, sim_neo};
    /// use pecos_neo::outcome::RegisterMap;
    /// use pecos_neo::prelude::*;
    ///
    /// let circuit = CommandBuilder::new()
    ///     .pz(&[0]).h(&[0]).mz(&[0])
    ///     .build();
    ///
    /// let mut reg = RegisterMap::new();
    /// reg.add_register("c", &[QubitId(0)]);
    ///
    /// let results = sim_neo(circuit).auto().sampling(monte_carlo(1000)).seed(42).run();
    /// let counts = results.register_counts(&reg, "c");
    /// // Should have entries for [false] and [true]
    /// ```
    #[must_use]
    pub fn register_counts(
        &self,
        register: &RegisterMap,
        name: &str,
    ) -> BTreeMap<Vec<bool>, usize> {
        let mut counts = BTreeMap::new();

        for outcome in &self.outcomes {
            if let Some(bits) = outcome.register_bitstring(register, name) {
                *counts.entry(bits).or_insert(0) += 1;
            }
        }

        counts
    }
}

/// Wrapper for noise model resource.
pub struct NoiseResource(pub ComposableNoiseModel);

/// Wrapper for gate definitions resource.
struct GateDefinitionsResource(GateDefinitions);

/// Wrapper for max decomposition depth resource.
struct MaxDecompDepthResource(usize);

/// Type-erased storage for gate overrides.
///
/// Gate overrides are generic over the simulator type `S`, but the builder
/// doesn't know `S` until build time. This enum carries the typed overrides
/// as data, deferring application to startup when the backend is known.
#[derive(Clone)]
pub enum StoredOverrides {
    /// Overrides for the sparse stabilizer backend.
    SparseStab(GateOverrides<SparseStab>),
    /// Overrides for the public stabilizer backend.
    Stabilizer(GateOverrides<Stabilizer>),
    /// Overrides for the state vector backend.
    StateVec(GateOverrides<StateVec>),
}

impl From<GateOverrides<SparseStab>> for StoredOverrides {
    fn from(overrides: GateOverrides<SparseStab>) -> Self {
        Self::SparseStab(overrides)
    }
}

impl From<GateOverrides<Stabilizer>> for StoredOverrides {
    fn from(overrides: GateOverrides<Stabilizer>) -> Self {
        Self::Stabilizer(overrides)
    }
}

impl From<GateOverrides<StateVec>> for StoredOverrides {
    fn from(overrides: GateOverrides<StateVec>) -> Self {
        Self::StateVec(overrides)
    }
}

/// Wrapper for gate overrides resource.
struct GateOverridesResource(StoredOverrides);

/// Wrapper for event handlers resource.
struct EventHandlersResource(EventHandlers);

// --- Classical Engine Support ---

/// Trait for type-erased engine building.
///
/// This allows storing different engine builder types uniformly.
pub trait BoxedEngineBuilder: Send + Sync {
    /// Clone this builder into a boxed trait object for independent workers.
    fn clone_box(&self) -> Box<dyn BoxedEngineBuilder>;

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
impl Clone for Box<dyn BoxedEngineBuilder> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Wrapper for concrete classical engine builders.
struct EngineBuilderWrapper<B>
where
    B: pecos_engines::ClassicalControlEngineBuilder + Clone + Send + Sync,
    B::Engine: 'static,
{
    builder: B,
}
impl<B> BoxedEngineBuilder for EngineBuilderWrapper<B>
where
    B: pecos_engines::ClassicalControlEngineBuilder + Clone + Send + Sync + 'static,
    B::Engine: 'static,
{
    fn clone_box(&self) -> Box<dyn BoxedEngineBuilder> {
        Box::new(EngineBuilderWrapper {
            builder: self.builder.clone(),
        })
    }

    fn build_adapter(
        self: Box<Self>,
    ) -> Result<Box<dyn CommandSource + Send + Sync>, pecos_core::errors::PecosError> {
        let engine = self.builder.build()?;
        Ok(Box::new(crate::adapter::ClassicalEngineAdapter::new(
            engine,
        )))
    }

    fn num_qubits_hint(&self) -> Option<usize> {
        // Most builders don't know num_qubits until built
        None
    }
}

/// Engine builder stored as data, waiting for source text to be configured at build time.
///
/// This keeps `.classical(builder)` shape-based instead of tying it to a closed
/// list of built-in language frontends. Built-in QASM/HUGR builders provide
/// `From` impls when those optional frontend features are enabled, and external
/// crates can construct this wrapper with [`PendingEngineBuilder::from_source_builder`].
pub struct PendingEngineBuilder {
    configure: Box<dyn FnOnce(String) -> Box<dyn BoxedEngineBuilder> + Send + Sync>,
}

impl PendingEngineBuilder {
    /// Create a pending source builder from a function that accepts raw source
    /// and returns a configured classical-engine builder.
    pub fn from_source_builder<B, F>(configure: F) -> Self
    where
        B: pecos_engines::ClassicalControlEngineBuilder + Clone + Send + Sync + 'static,
        B::Engine: 'static,
        F: FnOnce(String) -> B + Send + Sync + 'static,
    {
        Self {
            configure: Box::new(move |source| {
                Box::new(EngineBuilderWrapper {
                    builder: configure(source),
                })
            }),
        }
    }

    /// Configure this builder with source and return a boxed engine builder.
    ///
    /// Called at `.build()` time to inject the source into the stored builder.
    fn configure_with_source(self, source: String) -> Box<dyn BoxedEngineBuilder> {
        (self.configure)(source)
    }
}

// Conversion from QasmEngineBuilder to PendingEngineBuilder
#[cfg(feature = "qasm")]
impl From<pecos_qasm::QasmEngineBuilder> for PendingEngineBuilder {
    fn from(builder: pecos_qasm::QasmEngineBuilder) -> Self {
        Self::from_source_builder(move |source| builder.qasm(source))
    }
}

// Conversion from HugrEngineBuilder to PendingEngineBuilder
#[cfg(feature = "hugr")]
impl From<pecos_hugr::HugrEngineBuilder> for PendingEngineBuilder {
    fn from(builder: pecos_hugr::HugrEngineBuilder) -> Self {
        Self::from_source_builder(move |source| builder.hugr_bytes(source.into_bytes()))
    }
}

/// The source of quantum operations for simulation.
pub enum ProgramSource {
    /// A static circuit (no mid-circuit feedback).
    Static(CommandQueue),
    /// A dynamic command source.
    Dynamic(Box<dyn CommandSource + Send + Sync>),
    /// Raw program source code (needs engine factory to interpret).
    RawSource(String),
    /// A typed program (knows its type, can use `.auto()` for engine selection).
    Typed(TypedProgram),
    /// A classical engine builder (supports mid-circuit feedback).
    Classical(Box<dyn BoxedEngineBuilder>),
}

/// Typed program variants for automatic engine selection.
///
/// When using `.auto()`, the appropriate engine is selected based on the variant.
#[derive(Debug, Clone)]
pub enum TypedProgram {
    /// QASM program - uses `qasm_engine()`
    Qasm(pecos_programs::Qasm),
    /// HUGR program - uses `hugr_engine()`
    Hugr(pecos_programs::Hugr),
    /// Unsupported program type (for error messages)
    Unsupported(String),
}

/// Resource to hold the program source.
pub struct ProgramSourceResource(pub ProgramSource);

/// Temporary storage for current shot outcomes.
struct CurrentOutcomes(MeasurementOutcomes);

fn infer_num_qubits_from_circuit(circuit: &CommandQueue) -> usize {
    circuit
        .iter()
        .flat_map(|cmd| cmd.qubits.iter())
        .map(|q| q.0)
        .max()
        .map_or(1, |max| max + 1)
}

// --- SimNeoBuilder ---

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
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
/// let results = sim_neo(circuit).auto()
///     .depolarizing(0.01)
///     .sampling(monte_carlo(1000))
///     .seed(42)
///     .build()
///     .run();
/// ```
///
/// ## QASM Program (builder-of-builders pattern)
///
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo};
/// use pecos_qasm::qasm_engine;
///
/// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];";
/// // Pass program source first, then engine factory
/// let results = sim_neo(qasm_code).auto()
///     .classical(qasm_engine())  // Engine configured with source at build time
///     .sampling(monte_carlo(1000))
///     .seed(42)
///     .build()
///     .run();
/// ```
///
/// ## Pre-configured Engine Builder
///
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo_builder};
/// use pecos_qasm::qasm_engine;
///
/// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];";
/// // Or pass already-configured engine builder
/// let results = sim_neo_builder()
///     .with_engine(qasm_engine().qasm(qasm_code))
///     .sampling(monte_carlo(1000))
///     .build()
///     .run();
/// ```
pub struct SimNeoBuilder {
    /// The program source (circuit, raw source, or engine builder).
    source: Option<ProgramSource>,
    /// Engine builder stored as data, waiting for source at build time.
    pending_builder: Option<PendingEngineBuilder>,
    /// Noise model (collected as data, used at build time).
    noise: Option<ComposableNoiseModel>,
    /// Gate definitions for custom/decomposed gates.
    definitions: Option<GateDefinitions>,
    /// Simulation configuration (data).
    config: SimConfig,
    /// Sampling strategy (data). None until `.sampling()` is called.
    sampling: Option<Sampling>,
    /// Shot count from the top-level `.shots()` shortcut (sugar for the
    /// default Monte Carlo sampler). Mutually exclusive with `.sampling()`.
    shots_shortcut: Option<usize>,
    /// Worker count from the deprecated top-level `.workers()` forwarder.
    legacy_workers: Option<usize>,
    /// Auto worker-count request from `.auto()`/deprecated `.auto_workers()`,
    /// honored only on the top-level `.shots()` shortcut path.
    auto_workers_hint: bool,
    /// Backend auto-selection opt-in from `.auto()`.
    auto_backend: bool,
    /// Quantum backend configuration (data). None until `.quantum()` is
    /// called; `.auto()` opts into automatic selection at build time.
    quantum_backend: Option<QuantumBackend>,
    /// Explicit qubit count override (data).
    explicit_num_qubits: Option<usize>,
    /// Maximum decomposition depth for gate resolution.
    max_decomp_depth: Option<usize>,
    /// Gate overrides (type-erased, applied at startup).
    overrides: Option<StoredOverrides>,
    /// Event handlers (cloned per worker for parallel execution).
    event_handlers: Option<EventHandlers>,
}

impl SimNeoBuilder {
    /// Create a builder with the given source and all other fields unset.
    fn from_source(source: Option<ProgramSource>) -> Self {
        Self {
            source,
            pending_builder: None,
            noise: None,
            definitions: None,
            config: SimConfig::default(),
            sampling: None,
            shots_shortcut: None,
            legacy_workers: None,
            auto_workers_hint: false,
            auto_backend: false,
            quantum_backend: None,
            explicit_num_qubits: None,
            max_decomp_depth: None,
            overrides: None,
            event_handlers: None,
        }
    }

    /// Create a new simulation builder for a circuit.
    #[must_use]
    pub fn with_circuit(circuit: CommandQueue) -> Self {
        Self::from_source(Some(ProgramSource::Static(circuit)))
    }

    /// Create a simulation builder for a dynamic command source.
    #[must_use]
    pub fn with_command_source(source: Box<dyn CommandSource + Send + Sync>) -> Self {
        Self::from_source(Some(ProgramSource::Dynamic(source)))
    }

    /// Create a simulation builder with raw program source.
    ///
    /// Use `.classical(builder)` to specify how to interpret the source.
    #[must_use]
    pub fn with_program_source(source: String) -> Self {
        Self::from_source(Some(ProgramSource::RawSource(source)))
    }

    /// Create a simulation builder with a typed program.
    ///
    /// Use `.auto()` to automatically select the engine, or
    /// `.classical(builder)` for explicit control.
    #[must_use]
    pub fn with_typed_program(program: TypedProgram) -> Self {
        Self::from_source(Some(ProgramSource::Typed(program)))
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
        Self::from_source(None)
    }

    /// Set the classical control engine builder (builder-of-builders pattern).
    ///
    /// The builder is stored as data and configured with source at `.build()` time.
    /// This follows "everything is data" - we collect configuration, then wire
    /// it all together when building the Tool.
    ///
    /// ```no_run
    /// use pecos_neo::tool::{monte_carlo, sim_neo};
    /// use pecos_qasm::qasm_engine;
    ///
    /// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];";
    /// // Builder is stored as data, source injected at build time
    /// let results = sim_neo(qasm_code).auto()
    ///     .classical(qasm_engine())  // stores builder as data
    ///     .sampling(monte_carlo(1000))
    ///     .build()  // configures builder, builds engine, creates Tool
    ///     .run();
    /// ```
    ///
    /// For pre-configured engine builders, use `.with_engine()` instead:
    ///
    /// ```no_run
    /// use pecos_neo::tool::{monte_carlo, sim_neo_builder};
    /// use pecos_qasm::qasm_engine;
    ///
    /// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];";
    /// let results = sim_neo_builder()
    ///     .with_engine(qasm_engine().qasm(qasm_code))
    ///     .sampling(monte_carlo(1000))
    ///     .build()
    ///     .run();
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if no raw source was provided via `sim_neo(source_code)`.
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
                    TypedProgram::Hugr(_) => {
                        panic!(
                            "HUGR programs cannot be used with .classical(engine_builder). \
                             Use .auto() or pass the HUGR bytes directly to the engine builder."
                        );
                    }
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
            Some(ProgramSource::Dynamic(_)) => {
                panic!(
                    "Cannot use .classical() with an existing dynamic command source. \
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
    /// ```no_run
    /// use pecos_neo::tool::{monte_carlo, sim_neo_builder};
    /// use pecos_qasm::qasm_engine;
    ///
    /// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];";
    /// let results = sim_neo_builder()
    ///     .with_engine(qasm_engine().qasm(qasm_code))
    ///     .sampling(monte_carlo(1000))
    ///     .build()
    ///     .run();
    /// ```
    #[must_use]
    pub fn with_engine<B>(mut self, engine_builder: B) -> Self
    where
        B: pecos_engines::ClassicalControlEngineBuilder + Clone + Send + Sync + 'static,
        B::Engine: 'static,
    {
        self.source = Some(ProgramSource::Classical(Box::new(EngineBuilderWrapper {
            builder: engine_builder,
        })));
        self
    }

    /// Opt into automatic selection of unset components.
    ///
    /// `.auto()` is explicit-about-being-implicit: it lets the builder fill
    /// in components you did not set, instead of failing at build time.
    /// Currently it selects:
    /// - The classical engine for typed programs (`Qasm` uses `qasm_engine()`,
    ///   `Hugr` uses `hugr_engine()`); other sources are left unchanged.
    /// - The quantum backend, if `.quantum()` was not called
    ///   (currently `SparseStab`).
    ///
    /// The sampling strategy is never auto-selected: a shot count cannot be
    /// guessed, so `.sampling(monte_carlo(shots))` is always required. (On
    /// the deprecated top-level `.shots()` path, `.auto()` additionally
    /// requests an auto-detected worker count to preserve legacy behavior.)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::tool::{monte_carlo, sim_neo};
    /// use pecos_programs::Qasm;
    ///
    /// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];".to_string();
    /// // Auto-select engine and backend
    /// let results = sim_neo(Qasm::from_string(qasm_code))
    ///     .auto()
    ///     .sampling(monte_carlo(1000))
    ///     .build()
    ///     .run();
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if a typed program's type is not yet supported for
    /// auto-selection, or its engine's cargo feature is disabled.
    #[must_use]
    pub fn auto(mut self) -> Self {
        self.auto_backend = true;
        self.auto_workers_hint = true;
        self.source = match self.source.take() {
            Some(ProgramSource::Typed(typed)) => match typed {
                #[cfg(feature = "qasm")]
                TypedProgram::Qasm(qasm) => {
                    // Auto-select qasm_engine() and configure with the program.
                    let builder = pecos_qasm::qasm_engine().qasm(qasm.source);
                    Some(ProgramSource::Classical(Box::new(EngineBuilderWrapper {
                        builder,
                    })))
                }
                #[cfg(not(feature = "qasm"))]
                TypedProgram::Qasm(_) => {
                    panic!(
                        "QASM auto-selection requires the 'qasm' feature. \
                         Enable it with: features = [\"qasm\"]"
                    );
                }
                #[cfg(feature = "hugr")]
                TypedProgram::Hugr(hugr) => {
                    // Auto-select hugr_engine() and configure with the program.
                    let builder = pecos_hugr::hugr_engine().hugr_bytes(hugr.hugr);
                    Some(ProgramSource::Classical(Box::new(EngineBuilderWrapper {
                        builder,
                    })))
                }
                #[cfg(not(feature = "hugr"))]
                TypedProgram::Hugr(_) => {
                    panic!(
                        "HUGR auto-selection requires the 'hugr' feature. \
                         Enable it with: features = [\"hugr\"]"
                    );
                }
                TypedProgram::Unsupported(type_name) => {
                    panic!(
                        "Program type '{type_name}' is not yet supported for auto-selection. \
                         Use .classical(engine) to specify the engine explicitly."
                    );
                }
            },
            other => other,
        };
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

    /// Set the number of Monte Carlo shots to run.
    ///
    /// This is the shorthand for the common case: `.shots(n)` is exactly
    /// `.sampling(monte_carlo(n))`. Reach for [`sampling()`](Self::sampling)
    /// directly when you need worker parallelism or a non-Monte-Carlo strategy
    /// (e.g. `.sampling(monte_carlo(n).workers(8))` or
    /// `.sampling(importance_sampling(n))`). Setting both `.shots()` and
    /// `.sampling()` is rejected at build time rather than silently picking one.
    ///
    /// The facade (`pecos::sim().stack(...)`) exposes the same `.shots(n)`, so
    /// both stacks share the `.shots(n).run()` shape.
    #[must_use]
    pub fn shots(mut self, shots: usize) -> Self {
        self.shots_shortcut = Some(shots);
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Set the sampling strategy for simulation execution.
    ///
    /// The strategy carries its own shot count and execution knobs; build
    /// it with [`monte_carlo()`] or [`importance_sampling()`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::tool::{monte_carlo, sim_neo};
    /// use pecos_neo::prelude::*;
    ///
    /// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
    ///
    /// // Parallel Monte Carlo with 4 workers
    /// let results = sim_neo(circuit.clone()).auto()
    ///     .sampling(monte_carlo(1000).workers(4))
    ///     .build()
    ///     .run();
    ///
    /// // Auto-detect worker count
    /// let results = sim_neo(circuit).auto()
    ///     .sampling(monte_carlo(1000).auto_workers())
    ///     .build()
    ///     .run();
    /// ```
    #[must_use]
    pub fn sampling(mut self, sampling: impl Into<Sampling>) -> Self {
        self.sampling = Some(sampling.into());
        self
    }

    /// Convenience method for parallel Monte Carlo with specified workers.
    #[deprecated(
        since = "0.2.0",
        note = "workers lives on the sampler builder: use .sampling(monte_carlo(shots).workers(n))"
    )]
    #[must_use]
    pub fn workers(mut self, workers: usize) -> Self {
        self.legacy_workers = Some(workers);
        self
    }

    /// Convenience method for parallel Monte Carlo with auto-detected workers.
    #[deprecated(
        since = "0.2.0",
        note = "workers lives on the sampler builder: use .sampling(monte_carlo(shots).auto_workers())"
    )]
    #[must_use]
    pub fn auto_workers(mut self) -> Self {
        self.auto_workers_hint = true;
        self
    }

    /// Set the quantum backend for simulation.
    ///
    /// This selects which quantum simulator to use. Different backends have
    /// different capabilities and performance characteristics:
    ///
    /// - `sparse_stab()` - Sparse stabilizer, efficient for Clifford circuits
    /// - `state_vector()` - State vector, supports arbitrary gates including T and rotations
    ///
    /// A backend must be chosen: either call `.quantum()` explicitly or opt
    /// into automatic selection with `.auto()`. A missing backend is a
    /// build-time error.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::tool::{monte_carlo, sim_neo, sparse_stab, state_vector};
    /// use pecos_neo::prelude::*;
    ///
    /// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
    ///
    /// // Use sparse stabilizer (Clifford-only)
    /// let results = sim_neo(circuit.clone())
    ///     .quantum(sparse_stab())
    ///     .sampling(monte_carlo(1000))
    ///     .build()
    ///     .run();
    ///
    /// // Use state vector (supports T gates, rotations)
    /// let results = sim_neo(circuit)
    ///     .quantum(state_vector())
    ///     .sampling(monte_carlo(1000))
    ///     .build()
    ///     .run();
    /// ```
    #[must_use]
    pub fn quantum<B: Into<QuantumBackend>>(mut self, backend: B) -> Self {
        self.quantum_backend = Some(backend.into());
        self
    }

    /// Set the `sim_neo` noise model.
    ///
    /// This configures `sim_neo`'s noise-modeling layer. It is intentionally
    /// separate from the quantum-engine builder protocol; backends that only
    /// provide quantum execution reject this configuration instead of silently
    /// ignoring it.
    ///
    /// Accepts any type that implements `Into<ComposableNoiseModel>`:
    /// - `ComposableNoiseModel` directly
    /// - `GeneralNoiseModelBuilder` (without calling `.build()`)
    /// - Any single `NoiseChannel`
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pecos_neo::tool::{monte_carlo, sim_neo};
    /// use pecos_neo::prelude::*;
    /// use pecos_neo::noise::GeneralNoiseModelBuilder;
    ///
    /// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
    ///
    /// // Using GeneralNoiseModelBuilder (no .build() needed)
    /// sim_neo(circuit.clone()).auto()
    ///     .noise(GeneralNoiseModelBuilder::new().with_p1(0.01).with_p2(0.02))
    ///     .build();
    ///
    /// // Using a single channel directly
    /// sim_neo(circuit.clone()).auto()
    ///     .noise(SingleQubitChannel::depolarizing(0.01))
    ///     .build();
    /// ```
    #[must_use]
    pub fn noise(mut self, noise: impl Into<ComposableNoiseModel>) -> Self {
        self.noise = Some(noise.into());
        self
    }

    /// Set custom gate definitions (decompositions, user-defined gates).
    ///
    /// Gate definitions control how gate identifiers are mapped to simulator
    /// operations. Use this to add custom gates or override built-in gate
    /// decompositions.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::tool::{monte_carlo, sim_neo};
    /// use pecos_neo::prelude::*;
    ///
    /// let defs = GateDefinitions::new(); // core gates included by default
    ///
    /// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
    /// let results = sim_neo(circuit).auto()
    ///     .gate_definitions(defs)
    ///     .sampling(monte_carlo(100))
    ///     .seed(42)
    ///     .build()
    ///     .run();
    /// ```
    #[must_use]
    pub fn gate_definitions(mut self, definitions: GateDefinitions) -> Self {
        self.definitions = Some(definitions);
        self
    }

    /// Set the maximum decomposition depth for gate resolution.
    ///
    /// Custom gates can decompose into other gates, which may themselves
    /// decompose further. This setting limits the recursion depth to prevent
    /// infinite loops from circular definitions. The default is 10.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::tool::{monte_carlo, sim_neo};
    /// use pecos_neo::prelude::*;
    ///
    /// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
    /// let results = sim_neo(circuit).auto()
    ///     .max_decomp_depth(20)
    ///     .sampling(monte_carlo(100))
    ///     .seed(42)
    ///     .run();
    /// ```
    #[must_use]
    pub fn max_decomp_depth(mut self, depth: usize) -> Self {
        self.max_decomp_depth = Some(depth);
        self
    }

    /// Set custom gate overrides for built-in backends.
    ///
    /// Gate overrides replace the default implementation of specific gates
    /// with custom executor functions. The overrides must match the selected
    /// backend type (`SparseStab` or `StateVec`).
    ///
    /// Type inference selects the correct `StoredOverrides` variant automatically
    /// via `From` impls, so just pass `GateOverrides<SparseStab>` or
    /// `GateOverrides<StateVec>` directly.
    ///
    /// # Panics
    ///
    /// Panics at run time if the overrides don't match the selected backend
    /// (e.g., `SparseStab` overrides with `state_vector()` backend).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::tool::{monte_carlo, sim_neo};
    /// use pecos_neo::prelude::*;
    /// use pecos_simulators::SparseStab;
    ///
    /// let overrides = GateOverrides::<SparseStab>::new()
    ///     .register(gates::X, |_sim, _angles, _qubits| {
    ///         // Custom X gate implementation
    ///         true
    ///     });
    ///
    /// let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();
    /// let results = sim_neo(circuit).auto()
    ///     .gate_overrides(overrides)
    ///     .sampling(monte_carlo(100))
    ///     .seed(42)
    ///     .run();
    /// ```
    #[must_use]
    pub fn gate_overrides(mut self, overrides: impl Into<StoredOverrides>) -> Self {
        self.overrides = Some(overrides.into());
        self
    }

    /// Set event handlers (gate and signal handlers) for the simulation.
    ///
    /// Event handlers are cloned per worker in parallel execution, so they
    /// work correctly with `.workers(n)` and `.auto_workers()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::tool::{monte_carlo, sim_neo};
    /// use pecos_neo::prelude::*;
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// use std::sync::Arc;
    ///
    /// let gate_count = Arc::new(AtomicUsize::new(0));
    /// let c = gate_count.clone();
    ///
    /// let handlers = EventHandlers::new()
    ///     .on_before_gate(move |_ctx| {
    ///         c.fetch_add(1, Ordering::Relaxed);
    ///         NoiseResponse::None
    ///     });
    ///
    /// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
    /// let results = sim_neo(circuit).auto()
    ///     .event_handlers(handlers)
    ///     .sampling(monte_carlo(100))
    ///     .seed(42)
    ///     .run();
    /// ```
    #[must_use]
    pub fn event_handlers(mut self, handlers: EventHandlers) -> Self {
        self.event_handlers = Some(handlers);
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
    /// - Sampling strategy is resolved and validated
    /// - Noise model is built
    /// - Tool is constructed with all plugins and systems
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - No program source is set (neither circuit nor classical engine)
    /// - No sampling strategy is set (use `.sampling(monte_carlo(shots))`)
    /// - No quantum backend is set (use `.quantum(..)` or `.auto()`)
    /// - Deprecated `.shots()`/`.workers()` are combined with `.sampling()`
    /// - Parallel Monte Carlo (`workers > 1`) is requested for a
    ///   configuration that cannot build per-worker state (pre-built dynamic
    ///   command sources)
    /// - Subset simulation is missing `.score()`/`.failure()`, or is used
    ///   with a non-static source or a backend other than `sparse_stab()`
    #[must_use]
    pub fn build(self) -> Simulation {
        // Resolve the program source - configure pending builder with source if needed
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
                        TypedProgram::Hugr(_) => "Hugr",
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

        // Resolve the sampling strategy: the .sampling() path carries its own
        // shot count; the top-level .shots() shortcut (and the deprecated
        // .workers() forwarder) map onto Monte Carlo. Mixing .shots()/.workers()
        // with .sampling() is ambiguous and rejected.
        let sampling = match (self.sampling, self.shots_shortcut) {
            (Some(sampling), None) => {
                assert!(
                    self.legacy_workers.is_none(),
                    "Conflicting sampling configuration: deprecated .workers() cannot be \
                     combined with .sampling(). Set workers on the sampler builder, e.g. \
                     .sampling(monte_carlo(1000).workers(8))."
                );
                sampling
            }
            (Some(_), Some(_)) => panic!(
                "Conflicting sampling configuration: .shots() cannot be combined with \
                 .sampling() (both set the shot count). Use one: .shots(1000) for the common \
                 case, or .sampling(monte_carlo(1000).workers(8)) for parallel/other strategies."
            ),
            (None, Some(shots)) => {
                let workers = self.legacy_workers.unwrap_or_else(|| {
                    if self.auto_workers_hint {
                        std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
                    } else {
                        1
                    }
                });
                Sampling::MonteCarlo { shots, workers }
            }
            (None, None) => panic!(
                "No sampling strategy set. Use .sampling(monte_carlo(shots)) for Monte Carlo \
                 or .sampling(importance_sampling(shots)) for rare-event estimation."
            ),
        };

        // The shot count drives the Tool's run loop via the SimConfig resource.
        let mut config = self.config;
        config.shots = sampling.shots();

        // Resolve the quantum backend: explicit .quantum() wins; .auto() opts
        // into automatic selection; otherwise fail fast.
        let auto_backend = self.auto_backend;
        let quantum_backend = self.quantum_backend.unwrap_or_else(|| {
            assert!(
                auto_backend,
                "No quantum backend set. Use .quantum(sparse_stab()) or \
                 .quantum(state_vector()), or call .auto() to let sim_neo choose."
            );
            QuantumBackend::SparseStab
        });

        // Configuration/backend mismatches are knowable now; fail at build
        // instead of at startup. The startup-time checks remain as defensive
        // duplicates for direct Tool users.
        if let Some(overrides) = &self.overrides {
            validate_overrides_backend(overrides, &quantum_backend);
        }
        match &quantum_backend {
            QuantumBackend::AdaptedQuantumEngine(_) => {
                reject_dynamic_runner_config(
                    "QuantumEngineBuilder backend",
                    self.definitions.as_ref(),
                    self.max_decomp_depth.as_ref(),
                    self.overrides.as_ref(),
                    self.event_handlers.as_ref(),
                );
                assert!(
                    self.noise.is_none(),
                    "QuantumEngineBuilder backends do not support sim_neo noise modeling. \
                     Use a noise-modeling runner/backend instead."
                );
            }
            QuantumBackend::Custom(factory) => {
                reject_dynamic_runner_config(
                    factory.diagnostic_label(),
                    self.definitions.as_ref(),
                    self.max_decomp_depth.as_ref(),
                    self.overrides.as_ref(),
                    self.event_handlers.as_ref(),
                );
            }
            _ => {}
        }

        let parallel_plan = match &sampling {
            Sampling::MonteCarlo { workers, .. } if *workers > 1 => {
                let plan = build_parallel_execution_plan(
                    &source,
                    &quantum_backend,
                    self.explicit_num_qubits,
                    self.noise.clone(),
                    self.definitions.clone(),
                    self.max_decomp_depth,
                    self.overrides.clone(),
                    self.event_handlers.clone(),
                );
                assert!(
                    plan.is_some(),
                    "Parallel Monte Carlo (workers > 1) requires per-worker construction: \
                     a static circuit or classical engine builder source. Pre-built dynamic \
                     command sources cannot build per-worker state; remove .workers(..) for \
                     sequential execution."
                );
                plan
            }
            _ => None,
        };

        // Importance sampling requires a static circuit; this is knowable
        // now, so fail at build time. Parallel IS runs outside the Tool
        // schedule and needs the circuit captured.
        let is_parallel_spec = match &sampling {
            Sampling::ImportanceSampling { config: is_config } => {
                let ProgramSource::Static(circuit) = &source else {
                    panic!(
                        "Importance sampling requires a static circuit. \
                         Classical engines are not supported."
                    )
                };
                if is_config.workers > 1 {
                    let circuit = circuit.clone();
                    let num_qubits = self
                        .explicit_num_qubits
                        .unwrap_or_else(|| infer_num_qubits_from_circuit(&circuit));
                    Some(StaticCircuitSpec {
                        circuit,
                        num_qubits,
                    })
                } else {
                    None
                }
            }
            _ => None,
        };

        // Path enumeration runs outside the Tool schedule; validate its
        // requirements here and capture what the run needs.
        let path_spec = match &sampling {
            Sampling::PathEnumeration { config: pe_config } => {
                assert!(
                    pe_config.max_measurements <= 24,
                    "Path enumeration covers 2^max_measurements paths; \
                     max_measurements = {} would enumerate more than 16M paths. \
                     Use subset_simulation or importance_sampling for larger spaces.",
                    pe_config.max_measurements
                );
                let circuit = match &source {
                    ProgramSource::Static(circuit) => circuit.clone(),
                    _ => panic!(
                        "Path enumeration requires a static circuit. Classical engines \
                         and dynamic command sources are not supported."
                    ),
                };
                assert!(
                    matches!(quantum_backend, QuantumBackend::SparseStab),
                    "Path enumeration currently supports only the sparse_stab() backend \
                     (or .auto())."
                );
                assert!(
                    self.noise.is_none(),
                    "Path enumeration enumerates measurement branches of the noiseless \
                     circuit; remove .noise()."
                );
                let num_qubits = self
                    .explicit_num_qubits
                    .unwrap_or_else(|| infer_num_qubits_from_circuit(&circuit));
                Some(StaticCircuitSpec {
                    circuit,
                    num_qubits,
                })
            }
            _ => None,
        };

        // Subset simulation runs outside the Tool schedule, driving
        // CircuitRunner directly; validate its requirements here and capture
        // what the run needs.
        let subset_spec = match &sampling {
            Sampling::SubsetSimulation { config: ss_config } => {
                assert!(
                    ss_config.score.is_some() && ss_config.failure.is_some(),
                    "Subset simulation requires both .score(..) and .failure(..) on the \
                     subset_simulation(..) builder; neither has a sensible default."
                );
                assert!(
                    ss_config.max_levels <= 1 || ss_config.allow_biased_multilevel,
                    "subset_simulation with max_levels > 1 engages the multi-level estimator, \
                     which is currently biased upward (the resample is unconditioned). Either \
                     keep a single level (an unbiased direct-Monte-Carlo failure-fraction \
                     estimate) or call .allow_biased_multilevel() to accept the documented bias."
                );
                let circuit = match &source {
                    ProgramSource::Static(circuit) => circuit.clone(),
                    _ => panic!(
                        "Subset simulation requires a static circuit. Classical engines \
                         and dynamic command sources are not supported."
                    ),
                };
                assert!(
                    matches!(quantum_backend, QuantumBackend::SparseStab),
                    "Subset simulation currently supports only the sparse_stab() backend \
                     (or .auto())."
                );
                let num_qubits = self
                    .explicit_num_qubits
                    .unwrap_or_else(|| infer_num_qubits_from_circuit(&circuit));
                Some(SubsetRunSpec {
                    circuit,
                    num_qubits,
                    noise: self.noise.clone(),
                })
            }
            _ => None,
        };

        let mut tool = Tool::new()
            .insert_resource(ProgramSourceResource(source))
            .insert_resource(config)
            .insert_resource(QuantumBackendResource(quantum_backend));

        match &sampling {
            Sampling::ImportanceSampling { config: is_config } => {
                tool = tool.add_plugin(&ImportanceSamplingSimPlugin {
                    is_config: is_config.clone(),
                    explicit_num_qubits: self.explicit_num_qubits,
                });
            }
            Sampling::MonteCarlo { .. } => {
                tool = tool.add_plugin(&UnifiedSimulationPlugin {
                    explicit_num_qubits: self.explicit_num_qubits,
                });
            }
            // Subset simulation and path enumeration do not use the Tool
            // schedule; no plugin.
            Sampling::SubsetSimulation { .. } | Sampling::PathEnumeration { .. } => {}
        }

        // Add noise if configured
        if let Some(noise) = self.noise {
            tool = tool.insert_resource(NoiseResource(noise));
        }

        // Add gate definitions if configured
        if let Some(definitions) = self.definitions {
            tool = tool.insert_resource(GateDefinitionsResource(definitions));
        }

        // Add max decomposition depth if configured
        if let Some(depth) = self.max_decomp_depth {
            tool = tool.insert_resource(MaxDecompDepthResource(depth));
        }

        // Add gate overrides if configured
        if let Some(overrides) = self.overrides.clone() {
            tool = tool.insert_resource(GateOverridesResource(overrides));
        }

        // Add event handlers if configured
        if let Some(handlers) = self.event_handlers {
            tool = tool.insert_resource(EventHandlersResource(handlers));
        }

        Simulation {
            tool,
            sampling,
            parallel_plan,
            subset_spec,
            is_parallel_spec,
            path_spec,
        }
    }

    /// Build and run the simulation in one step.
    ///
    /// This is a convenience method equivalent to `.build().run()`.
    /// Use `.build()` instead if you need to run multiple times or reconfigure.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::tool::{monte_carlo, sim_neo};
    /// use pecos_qasm::qasm_engine;
    ///
    /// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];";
    /// let results = sim_neo(qasm_code).auto()
    ///     .classical(qasm_engine())
    ///     .sampling(monte_carlo(1000))
    ///     .run();  // builds and runs
    /// ```
    #[must_use]
    pub fn run(self) -> SimulationResults {
        self.build().run()
    }
}

// --- Unified Simulation Plugin ---

/// Plugin that handles both static circuits and classical engines.
struct UnifiedSimulationPlugin {
    explicit_num_qubits: Option<usize>,
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

        // QuantumBackendResource is inserted directly by SimNeoBuilder::build()

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
/// simulator type, or a type-erased `DynProgramRunner` for custom backends.
pub enum QuantumRunner {
    /// Sparse stabilizer simulator (Clifford-only).
    SparseStab(ProgramRunner<SparseStab>),
    /// Public stabilizer simulator (Clifford-only).
    Stabilizer(ProgramRunner<Stabilizer>),
    /// State vector simulator (supports arbitrary gates).
    StateVec(ProgramRunner<StateVec>),
    /// Custom simulator backend via dynamic dispatch.
    Custom(Box<dyn DynProgramRunner>),
}

impl QuantumRunner {
    /// Run a shot and return the result.
    pub fn run_shot(&mut self, source: &mut dyn CommandSource) -> crate::program::ProgramResult {
        match self {
            Self::SparseStab(runner) => runner.run_shot(source),
            Self::Stabilizer(runner) => runner.run_shot(source),
            Self::StateVec(runner) => runner.run_shot(source),
            Self::Custom(runner) => runner.run_shot(source),
        }
    }

    /// Set the full seed for deterministic execution.
    pub fn set_full_seed(&mut self, seed: u64) {
        match self {
            Self::SparseStab(pr) => pr.set_full_seed(seed),
            Self::Stabilizer(pr) => pr.set_full_seed(seed),
            Self::StateVec(pr) => pr.set_full_seed(seed),
            Self::Custom(runner) => runner.set_full_seed(seed),
        }
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

/// Validate that stored gate overrides match the resolved backend.
///
/// The mismatch is knowable at build time; messages mirror the startup-time
/// checks for each backend.
fn validate_overrides_backend(overrides: &StoredOverrides, backend: &QuantumBackend) {
    let override_kind = match overrides {
        StoredOverrides::SparseStab(_) => "SparseStab",
        StoredOverrides::Stabilizer(_) => "Stabilizer",
        StoredOverrides::StateVec(_) => "StateVec",
    };
    let backend_kind = match backend {
        QuantumBackend::SparseStab => "SparseStab",
        QuantumBackend::Stabilizer => "Stabilizer",
        QuantumBackend::StateVec => "StateVec",
        // Adapted/custom backends reject overrides wholesale elsewhere.
        QuantumBackend::AdaptedQuantumEngine(_) | QuantumBackend::Custom(_) => return,
    };
    assert!(
        override_kind == backend_kind,
        "{override_kind} gate overrides used with {backend_kind} backend. \
         Use GateOverrides::<{backend_kind}> instead."
    );
}

fn reject_dynamic_runner_config(
    backend_name: &str,
    definitions: Option<&GateDefinitions>,
    max_depth: Option<&usize>,
    overrides: Option<&StoredOverrides>,
    event_handlers: Option<&EventHandlers>,
) {
    assert!(
        definitions.is_none(),
        "{backend_name} does not support sim_neo gate definitions. \
         Put custom gate handling inside the backend runner/factory instead."
    );
    assert!(
        max_depth.is_none(),
        "{backend_name} does not support sim_neo gate decomposition depth. \
         Put decomposition handling inside the backend runner/factory instead."
    );
    assert!(
        overrides.is_none(),
        "{backend_name} does not support sim_neo gate overrides. \
         Put override handling inside the backend runner/factory instead."
    );
    assert!(
        event_handlers.is_none(),
        "{backend_name} does not support sim_neo event handlers. \
         Use a ProgramRunner-based backend when event hooks are required."
    );
}
fn reject_parallel_adapted_engine_config(
    noise: Option<&ComposableNoiseModel>,
    definitions: Option<&GateDefinitions>,
    max_depth: Option<&usize>,
    overrides: Option<&StoredOverrides>,
    event_handlers: Option<&EventHandlers>,
) {
    assert!(
        noise.is_none(),
        "QuantumEngineBuilder backends do not support sim_neo noise modeling. \
         Use a noise-modeling runner/backend instead."
    );
    assert!(
        definitions.is_none(),
        "QuantumEngineBuilder backend does not support sim_neo gate definitions. \
         Put custom gate handling inside the backend runner/factory instead."
    );
    assert!(
        max_depth.is_none(),
        "QuantumEngineBuilder backend does not support sim_neo gate decomposition depth. \
         Put decomposition handling inside the backend runner/factory instead."
    );
    assert!(
        overrides.is_none(),
        "QuantumEngineBuilder backend does not support sim_neo gate overrides. \
         Put override handling inside the backend runner/factory instead."
    );
    assert!(
        event_handlers.is_none(),
        "QuantumEngineBuilder backend does not support sim_neo event handlers. \
         Use a ProgramRunner-based backend when event hooks are required."
    );
}

fn apply_standard_runner_config<S>(
    mut runner: ProgramRunner<S>,
    noise: Option<NoiseResource>,
    seed: Option<u64>,
    max_depth: Option<MaxDecompDepthResource>,
) -> ProgramRunner<S>
where
    S: CliffordGateable,
{
    if let Some(n) = noise {
        runner = runner.with_noise(n.0);
    }
    if let Some(seed) = seed {
        runner = runner.with_seed(seed);
    }
    if let Some(d) = max_depth {
        runner = runner.with_max_decomp_depth(d.0);
    }
    runner
}

fn apply_event_handlers<S>(
    mut runner: ProgramRunner<S>,
    event_handlers: Option<EventHandlersResource>,
) -> ProgramRunner<S>
where
    S: CliffordGateable,
{
    if let Some(eh) = event_handlers {
        runner = runner.with_event_handlers(eh.0);
    }
    runner
}

fn clifford_runner<S>(
    simulator: S,
    definitions: Option<GateDefinitionsResource>,
    noise: Option<NoiseResource>,
    seed: Option<u64>,
    max_depth: Option<MaxDecompDepthResource>,
) -> ProgramRunner<S>
where
    S: CliffordGateable,
{
    let runner = if let Some(defs) = definitions {
        ProgramRunner::with_definitions(simulator, defs.0)
    } else {
        ProgramRunner::new(simulator)
    };
    apply_standard_runner_config(runner, noise, seed, max_depth)
}

fn rotation_runner<S>(
    simulator: S,
    definitions: Option<GateDefinitionsResource>,
    noise: Option<NoiseResource>,
    seed: Option<u64>,
    max_depth: Option<MaxDecompDepthResource>,
) -> ProgramRunner<S>
where
    S: CliffordGateable + ArbitraryRotationGateable,
{
    let runner = if let Some(defs) = definitions {
        ProgramRunner::rotations_with_definitions(simulator, defs.0)
    } else {
        ProgramRunner::rotations(simulator)
    };
    apply_standard_runner_config(runner, noise, seed, max_depth)
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
            ProgramSource::Dynamic(source) => {
                let num_qubits = explicit_qubits.unwrap_or_else(|| source.num_qubits());
                (source, num_qubits)
            }
            ProgramSource::RawSource(_) => {
                // This should never happen - build() resolves RawSource with engine factory
                unreachable!(
                    "RawSource should be resolved to Classical by SimNeoBuilder::build(). \
                     This is a bug in the simulation builder."
                );
            }
            ProgramSource::Typed(_) => {
                // This should never happen - build() catches Typed without .auto()
                unreachable!(
                    "Typed program should be resolved by .auto() or caught by build(). \
                     This is a bug in the simulation builder."
                );
            }
            ProgramSource::Classical(engine_builder) => {
                // Build the engine adapter
                let adapter = engine_builder
                    .build_adapter()
                    .expect("Failed to build classical engine");

                let num_qubits = explicit_qubits.unwrap_or_else(|| adapter.num_qubits());
                (adapter, num_qubits)
            }
        };

    // Take quantum backend choice (take ownership for Custom variant)
    let backend = resources.remove::<QuantumBackendResource>().0;

    // Create quantum runner based on backend choice
    let noise = resources.try_remove::<NoiseResource>();
    let definitions = resources.try_remove::<GateDefinitionsResource>();
    let max_depth = resources.try_remove::<MaxDecompDepthResource>();
    let overrides = resources.try_remove::<GateOverridesResource>();
    let event_handlers = resources.try_remove::<EventHandlersResource>();
    let quantum_runner = match backend {
        QuantumBackend::SparseStab => {
            let mut runner = clifford_runner(
                SparseStab::new(num_qubits),
                definitions,
                noise,
                config.seed,
                max_depth,
            );
            if let Some(o) = overrides {
                match o.0 {
                    StoredOverrides::SparseStab(ov) => {
                        runner = runner.with_overrides(ov);
                    }
                    StoredOverrides::Stabilizer(_) => {
                        panic!(
                            "Stabilizer gate overrides used with SparseStab backend. \
                             Use GateOverrides::<SparseStab> instead."
                        );
                    }
                    StoredOverrides::StateVec(_) => {
                        panic!(
                            "StateVec gate overrides used with SparseStab backend. \
                             Use GateOverrides::<SparseStab> instead."
                        );
                    }
                }
            }
            runner = apply_event_handlers(runner, event_handlers);
            QuantumRunner::SparseStab(runner)
        }
        QuantumBackend::Stabilizer => {
            let mut runner = clifford_runner(
                Stabilizer::new(num_qubits),
                definitions,
                noise,
                config.seed,
                max_depth,
            );
            if let Some(o) = overrides {
                match o.0 {
                    StoredOverrides::Stabilizer(ov) => {
                        runner = runner.with_overrides(ov);
                    }
                    StoredOverrides::SparseStab(_) => {
                        panic!(
                            "SparseStab gate overrides used with Stabilizer backend. \
                             Use GateOverrides::<Stabilizer> instead."
                        );
                    }
                    StoredOverrides::StateVec(_) => {
                        panic!(
                            "StateVec gate overrides used with Stabilizer backend. \
                             Use GateOverrides::<Stabilizer> instead."
                        );
                    }
                }
            }
            runner = apply_event_handlers(runner, event_handlers);
            QuantumRunner::Stabilizer(runner)
        }
        QuantumBackend::StateVec => {
            let mut runner = rotation_runner(
                StateVec::new(num_qubits),
                definitions,
                noise,
                config.seed,
                max_depth,
            );
            if let Some(o) = overrides {
                match o.0 {
                    StoredOverrides::StateVec(ov) => {
                        runner = runner.with_overrides(ov);
                    }
                    StoredOverrides::Stabilizer(_) => {
                        panic!(
                            "Stabilizer gate overrides used with StateVec backend. \
                             Use GateOverrides::<StateVec> instead."
                        );
                    }
                    StoredOverrides::SparseStab(_) => {
                        panic!(
                            "SparseStab gate overrides used with StateVec backend. \
                             Use GateOverrides::<StateVec> instead."
                        );
                    }
                }
            }
            runner = apply_event_handlers(runner, event_handlers);
            QuantumRunner::StateVec(runner)
        }
        QuantumBackend::AdaptedQuantumEngine(factory) => {
            reject_dynamic_runner_config(
                "QuantumEngineBuilder backend",
                definitions.as_ref().map(|d| &d.0),
                max_depth.as_ref().map(|d| &d.0),
                overrides.as_ref().map(|o| &o.0),
                event_handlers.as_ref().map(|h| &h.0),
            );
            assert!(
                noise.is_none(),
                "QuantumEngineBuilder backends do not support sim_neo noise modeling. \
                 Use a noise-modeling runner/backend instead."
            );
            let runner = factory.create_runner(num_qubits, config.seed);
            QuantumRunner::Custom(runner)
        }
        QuantumBackend::Custom(factory) => {
            reject_dynamic_runner_config(
                factory.diagnostic_label(),
                definitions.as_ref().map(|d| &d.0),
                max_depth.as_ref().map(|d| &d.0),
                overrides.as_ref().map(|o| &o.0),
                event_handlers.as_ref().map(|h| &h.0),
            );
            // Custom backends create their own runner; gate definitions
            // should be captured in the factory closure if needed.
            let runner = factory.create_runner(num_qubits, noise.map(|n| n.0), config.seed);
            QuantumRunner::Custom(runner)
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
        state.quantum_runner.set_full_seed(shot_seed);
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

    // Collect rich register results when the source produces them
    // (classical engines: QASM cregs, PHIR variables).
    let shot = resources
        .get::<UnifiedShotState>()
        .command_source
        .shot_results();
    if let Some(shot) = shot {
        resources
            .get_mut::<SimulationResults>()
            .shots
            .get_or_insert_with(pecos_results::ShotVec::new)
            .shots
            .push(shot);
    }

    // `shots[i]` is consumed as the register view of `outcomes[i]`, so the
    // two must stay index-aligned: a source must produce a shot record for
    // every shot or for none. Anything in between (a source that returns
    // Some on some shots and None on others) silently misaligns them.
    let results = resources.get::<SimulationResults>();
    debug_assert!(
        results
            .shots
            .as_ref()
            .is_none_or(|shots| shots.shots.len() == results.outcomes.len()),
        "shot records and outcomes are misaligned ({} shot records vs {} outcomes); \
         a CommandSource must return shot_results() for every shot or for none",
        results.shots.as_ref().map_or(0, |s| s.shots.len()),
        results.outcomes.len(),
    );

    // Increment shot counter
    resources.get_mut::<UnifiedShotState>().shot_index += 1;
}

// --- Importance Sampling Simulation Plugin ---

/// Plugin for importance-sampling simulation.
///
/// Replaces [`UnifiedSimulationPlugin`] when importance sampling is selected.
/// Uses [`ImportanceSamplingRunner`] for biased noise with weight tracking.
struct ImportanceSamplingSimPlugin {
    is_config: ImportanceSamplingBuilder,
    explicit_num_qubits: Option<usize>,
}

impl Plugin for ImportanceSamplingSimPlugin {
    fn build(&self, tool: &mut Tool) {
        if !tool.contains_resource::<SimConfig>() {
            tool.insert_resource_mut(SimConfig::default());
        }
        if !tool.contains_resource::<SimulationResults>() {
            tool.insert_resource_mut(SimulationResults::new());
        }

        tool.insert_resource_mut(ExplicitNumQubits(self.explicit_num_qubits));
        tool.insert_resource_mut(ISConfigResource(self.is_config.clone()));

        tool.add_system_mut(Stage::Startup, is_sim_startup);
        tool.add_system_mut(Stage::PreShot, is_sim_pre_shot);
        tool.add_system_mut(Stage::Execute, is_sim_execute);
        tool.add_system_mut(Stage::PostShot, is_sim_post_shot);
    }
}

/// Resource holding IS configuration, consumed at startup.
struct ISConfigResource(ImportanceSamplingBuilder);

/// State for importance sampling simulation shots.
struct ISShotState {
    /// The importance sampling runner.
    runner: ImportanceSamplingRunner<SparseStab>,
    /// The circuit to execute.
    circuit: CommandQueue,
    /// Current shot index.
    shot_index: usize,
}

/// Temporary result from IS execution, passed from Execute to `PostShot`.
struct ISCurrentResult {
    outcomes: MeasurementOutcomes,
    weight: crate::sampling::weight::SampleWeight,
}

/// Build an importance sampling runner from builder config.
fn build_importance_runner(
    is_config: &ImportanceSamplingBuilder,
    num_qubits: usize,
) -> ImportanceSamplingRunner<SparseStab> {
    ImportanceSamplingRunner::new(SparseStab::new(num_qubits))
        .with_single_qubit_boost(is_config.p1(), is_config.boost())
        .with_two_qubit_boost(is_config.p2(), is_config.boost())
        .with_measurement_boost(is_config.p_meas(), is_config.boost())
}

/// Seed an importance sampling runner for a specific global shot index.
///
/// Seeds derive from (`base_seed`, `shot_index`) only, so results are
/// identical whether shots run sequentially or partitioned across workers.
fn seed_importance_runner(
    runner: &mut ImportanceSamplingRunner<SparseStab>,
    base_seed: u64,
    shot_index: usize,
) {
    let shot_seed = derive_seed(base_seed, &format!("shot_{shot_index}"));
    runner.rng = PecosRng::seed_from_u64(derive_seed(shot_seed, "noise"));
    runner
        .simulator
        .set_seed(derive_seed(shot_seed, "simulator"));
}

/// Startup system for importance sampling simulation.
fn is_sim_startup(resources: &mut Resources) {
    let explicit_qubits = resources.get::<ExplicitNumQubits>().0;

    // Re-run: reset state instead of rebuilding
    if resources.contains::<ISShotState>() {
        resources.get_mut::<ISShotState>().shot_index = 0;
        let results = resources.get_mut::<SimulationResults>();
        results.clear();
        results.weights = Some(Vec::new());
        return;
    }

    // First run - consume resources and build the runner
    let source_resource = resources.remove::<ProgramSourceResource>();
    let is_config = resources.remove::<ISConfigResource>().0;

    let circuit = match source_resource.0 {
        ProgramSource::Static(circuit) => circuit,
        ProgramSource::Dynamic(_)
        | ProgramSource::RawSource(_)
        | ProgramSource::Typed(_)
        | ProgramSource::Classical(_) => {
            panic!(
                "Importance sampling requires a static circuit. \
                 Classical engines are not supported."
            )
        }
    };

    let num_qubits = explicit_qubits.unwrap_or_else(|| {
        circuit
            .iter()
            .flat_map(|cmd| cmd.qubits.iter())
            .map(|q| q.0)
            .max()
            .map_or(1, |max| max + 1)
    });

    // Consume QuantumBackendResource (IS always uses SparseStab internally)
    let _ = resources.remove::<QuantumBackendResource>();

    // Also consume NoiseResource if present (IS uses its own boosted noise)
    let _ = resources.try_remove::<NoiseResource>();

    let runner = build_importance_runner(&is_config, num_qubits);

    resources.insert(ISShotState {
        runner,
        circuit,
        shot_index: 0,
    });

    // Initialize results with weight tracking
    let results = resources.get_mut::<SimulationResults>();
    results.clear();
    results.weights = Some(Vec::new());
}

/// Pre-shot system for importance sampling: derive and set per-shot seeds.
///
/// Only reseeds when a base seed was configured, mirroring the unified
/// Monte Carlo path ([`unified_simulation_pre_shot`]): an unseeded run
/// keeps the runner's entropy-seeded RNG rather than silently forcing a
/// deterministic stream.
fn is_sim_pre_shot(resources: &mut Resources) {
    let config = resources.get::<SimConfig>().clone();
    let state = resources.get_mut::<ISShotState>();

    if let Some(base_seed) = config.seed {
        let shot_index = state.shot_index;
        seed_importance_runner(&mut state.runner, base_seed, shot_index);
    }
}

/// Execute system for importance sampling: run one shot with biased noise.
fn is_sim_execute(resources: &mut Resources) {
    let state = resources.get_mut::<ISShotState>();
    let result = state.runner.run_shot_fresh(&state.circuit);

    resources.insert(ISCurrentResult {
        outcomes: result.outcomes,
        weight: result.weight,
    });
}

/// Post-shot system for importance sampling: collect outcomes and weights.
fn is_sim_post_shot(resources: &mut Resources) {
    let result = resources.remove::<ISCurrentResult>();
    let results = resources.get_mut::<SimulationResults>();

    results.outcomes.push(result.outcomes);
    if let Some(ref mut weights) = results.weights {
        weights.push(result.weight);
    }

    resources.get_mut::<ISShotState>().shot_index += 1;
}

// --- Simulation Handle ---

/// Reusable simulation handle.
///
/// Created via [`SimNeoBuilder::build()`], this handle can be run multiple
/// times with different configurations.
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
/// let mut sim = sim_neo(circuit).auto().sampling(monte_carlo(1000)).build();
///
/// let results1 = sim.run();
///
/// // Reconfigure and run again
/// sim.shots(2000).seed(123);
/// let results2 = sim.run();
/// ```
pub struct Simulation {
    tool: Tool,
    /// Sampling strategy (stored as data).
    sampling: Sampling,
    /// Data-oriented plan for parallel execution (if applicable).
    parallel_plan: Option<ParallelExecutionPlan>,
    /// Captured inputs for subset simulation (if applicable).
    subset_spec: Option<SubsetRunSpec>,
    /// Captured inputs for parallel importance sampling (if applicable).
    is_parallel_spec: Option<StaticCircuitSpec>,
    /// Captured inputs for path enumeration (if applicable).
    path_spec: Option<StaticCircuitSpec>,
}

/// Inputs captured at build time for a subset simulation run.
struct SubsetRunSpec {
    circuit: CommandQueue,
    num_qubits: usize,
    noise: Option<ComposableNoiseModel>,
}

/// Inputs captured at build time for a parallel importance sampling run.
struct StaticCircuitSpec {
    circuit: CommandQueue,
    num_qubits: usize,
}

/// Native backend used by the internal parallel runner factory.
#[derive(Debug, Clone, Copy)]
enum NativeParallelBackend {
    SparseStab,
    Stabilizer,
    StateVec,
}

trait ParallelCommandSourceFactory: Send + Sync {
    fn create_source(&self) -> Box<dyn CommandSource + Send + Sync>;
}

#[doc(hidden)]
pub trait ParallelQuantumRunnerFactory: Send + Sync {
    fn create_runner(&self, seed: Option<u64>) -> QuantumRunner;
}

struct ParallelExecutionPlan {
    command_source_factory: Box<dyn ParallelCommandSourceFactory>,
    quantum_runner_factory: Box<dyn ParallelQuantumRunnerFactory>,
}

struct StaticCommandSourceFactory {
    circuit: CommandQueue,
    num_qubits: usize,
}

impl ParallelCommandSourceFactory for StaticCommandSourceFactory {
    fn create_source(&self) -> Box<dyn CommandSource + Send + Sync> {
        Box::new(StaticProgram::new(self.circuit.clone(), self.num_qubits))
    }
}
struct ClassicalCommandSourceFactory {
    builder: Box<dyn BoxedEngineBuilder>,
}
impl ParallelCommandSourceFactory for ClassicalCommandSourceFactory {
    fn create_source(&self) -> Box<dyn CommandSource + Send + Sync> {
        self.builder
            .clone()
            .build_adapter()
            .expect("Failed to build classical engine for worker")
    }
}

struct NativeQuantumRunnerFactory {
    backend: NativeParallelBackend,
    num_qubits: usize,
    noise: Option<ComposableNoiseModel>,
    definitions: Option<GateDefinitions>,
    max_decomp_depth: Option<usize>,
    overrides: Option<StoredOverrides>,
    event_handlers: Option<EventHandlers>,
}

impl ParallelQuantumRunnerFactory for NativeQuantumRunnerFactory {
    fn create_runner(&self, seed: Option<u64>) -> QuantumRunner {
        let noise = self.noise.clone().map(NoiseResource);
        let definitions = self.definitions.clone().map(GateDefinitionsResource);
        let max_depth = self.max_decomp_depth.map(MaxDecompDepthResource);
        let event_handlers = self.event_handlers.clone().map(EventHandlersResource);

        match self.backend {
            NativeParallelBackend::SparseStab => {
                let mut runner = clifford_runner(
                    SparseStab::new(self.num_qubits),
                    definitions,
                    noise,
                    seed,
                    max_depth,
                );
                if let Some(overrides) = self.overrides.clone() {
                    match overrides {
                        StoredOverrides::SparseStab(ov) => runner = runner.with_overrides(ov),
                        StoredOverrides::Stabilizer(_) => {
                            panic!(
                                "Stabilizer gate overrides used with SparseStab backend. \
                                 Use GateOverrides::<SparseStab> instead."
                            );
                        }
                        StoredOverrides::StateVec(_) => {
                            panic!(
                                "StateVec gate overrides used with SparseStab backend. \
                                 Use GateOverrides::<SparseStab> instead."
                            );
                        }
                    }
                }
                runner = apply_event_handlers(runner, event_handlers);
                QuantumRunner::SparseStab(runner)
            }
            NativeParallelBackend::Stabilizer => {
                let mut runner = clifford_runner(
                    Stabilizer::new(self.num_qubits),
                    definitions,
                    noise,
                    seed,
                    max_depth,
                );
                if let Some(overrides) = self.overrides.clone() {
                    match overrides {
                        StoredOverrides::Stabilizer(ov) => runner = runner.with_overrides(ov),
                        StoredOverrides::SparseStab(_) => {
                            panic!(
                                "SparseStab gate overrides used with Stabilizer backend. \
                                 Use GateOverrides::<Stabilizer> instead."
                            );
                        }
                        StoredOverrides::StateVec(_) => {
                            panic!(
                                "StateVec gate overrides used with Stabilizer backend. \
                                 Use GateOverrides::<Stabilizer> instead."
                            );
                        }
                    }
                }
                runner = apply_event_handlers(runner, event_handlers);
                QuantumRunner::Stabilizer(runner)
            }
            NativeParallelBackend::StateVec => {
                let mut runner = rotation_runner(
                    StateVec::new(self.num_qubits),
                    definitions,
                    noise,
                    seed,
                    max_depth,
                );
                if let Some(overrides) = self.overrides.clone() {
                    match overrides {
                        StoredOverrides::StateVec(ov) => runner = runner.with_overrides(ov),
                        StoredOverrides::SparseStab(_) => {
                            panic!(
                                "SparseStab gate overrides used with StateVec backend. \
                                 Use GateOverrides::<StateVec> instead."
                            );
                        }
                        StoredOverrides::Stabilizer(_) => {
                            panic!(
                                "Stabilizer gate overrides used with StateVec backend. \
                                 Use GateOverrides::<StateVec> instead."
                            );
                        }
                    }
                }
                runner = apply_event_handlers(runner, event_handlers);
                QuantumRunner::StateVec(runner)
            }
        }
    }
}
/// Per-worker runner factory for custom `SimulatorFactory` backends.
///
/// The user's factory is invoked once per worker with a clone of the noise
/// model; per-shot seeding from global shot indices happens in the shared
/// schedule, exactly as for built-in backends.
struct CustomRunnerFactory {
    factory: Arc<dyn SimulatorFactory>,
    num_qubits: usize,
    noise: Option<ComposableNoiseModel>,
}

impl ParallelQuantumRunnerFactory for CustomRunnerFactory {
    fn create_runner(&self, seed: Option<u64>) -> QuantumRunner {
        QuantumRunner::Custom(
            self.factory
                .create_runner(self.num_qubits, self.noise.clone(), seed),
        )
    }
}

struct AdaptedQuantumEngineRunnerFactory<B>
where
    B: pecos_engines::QuantumEngineBuilder + Clone + 'static,
{
    builder: B,
    num_qubits: usize,
}
impl<B> ParallelQuantumRunnerFactory for AdaptedQuantumEngineRunnerFactory<B>
where
    B: pecos_engines::QuantumEngineBuilder + Clone + 'static,
{
    fn create_runner(&self, seed: Option<u64>) -> QuantumRunner {
        let mut builder = self.builder.clone();
        builder.set_qubits_if_needed(self.num_qubits);
        let mut engine = builder
            .build()
            .expect("Failed to build quantum engine backend for worker");
        if let Some(seed) = seed {
            engine.set_seed(seed);
        }
        QuantumRunner::Custom(Box::new(crate::adapter::QuantumEngineProgramRunner::new(
            engine,
        )))
    }
}

#[allow(clippy::too_many_arguments)]
fn build_parallel_execution_plan(
    source: &ProgramSource,
    backend: &QuantumBackend,
    explicit_num_qubits: Option<usize>,
    noise: Option<ComposableNoiseModel>,
    definitions: Option<GateDefinitions>,
    max_decomp_depth: Option<usize>,
    overrides: Option<StoredOverrides>,
    event_handlers: Option<EventHandlers>,
) -> Option<ParallelExecutionPlan> {
    let (source_factory, num_qubits): (Box<dyn ParallelCommandSourceFactory>, usize) = match source
    {
        ProgramSource::Static(circuit) => {
            let num_qubits =
                explicit_num_qubits.unwrap_or_else(|| infer_num_qubits_from_circuit(circuit));
            (
                Box::new(StaticCommandSourceFactory {
                    circuit: circuit.clone(),
                    num_qubits,
                }),
                num_qubits,
            )
        }
        ProgramSource::Dynamic(_) => return None,
        ProgramSource::Classical(engine_builder) => {
            let probe = engine_builder
                .clone()
                .build_adapter()
                .expect("Failed to build classical engine while preparing parallel plan");
            let num_qubits = explicit_num_qubits.unwrap_or_else(|| probe.num_qubits());
            (
                Box::new(ClassicalCommandSourceFactory {
                    builder: engine_builder.clone(),
                }),
                num_qubits,
            )
        }
        ProgramSource::RawSource(_) | ProgramSource::Typed(_) => {
            unreachable!("raw and typed sources should be resolved before plan construction")
        }
    };

    let runner_factory: Box<dyn ParallelQuantumRunnerFactory> = match backend {
        QuantumBackend::SparseStab => Box::new(NativeQuantumRunnerFactory {
            backend: NativeParallelBackend::SparseStab,
            num_qubits,
            noise,
            definitions,
            max_decomp_depth,
            overrides,
            event_handlers,
        }),
        QuantumBackend::Stabilizer => Box::new(NativeQuantumRunnerFactory {
            backend: NativeParallelBackend::Stabilizer,
            num_qubits,
            noise,
            definitions,
            max_decomp_depth,
            overrides,
            event_handlers,
        }),
        QuantumBackend::StateVec => Box::new(NativeQuantumRunnerFactory {
            backend: NativeParallelBackend::StateVec,
            num_qubits,
            noise,
            definitions,
            max_decomp_depth,
            overrides,
            event_handlers,
        }),
        QuantumBackend::AdaptedQuantumEngine(factory) => {
            reject_parallel_adapted_engine_config(
                noise.as_ref(),
                definitions.as_ref(),
                max_decomp_depth.as_ref(),
                overrides.as_ref(),
                event_handlers.as_ref(),
            );
            factory.create_parallel_runner_factory(num_qubits)
        }
        QuantumBackend::Custom(factory) => Box::new(CustomRunnerFactory {
            factory: Arc::clone(factory),
            num_qubits,
            noise,
        }),
    };

    Some(ParallelExecutionPlan {
        command_source_factory: source_factory,
        quantum_runner_factory: runner_factory,
    })
}

impl Simulation {
    /// Override the number of shots for the next run.
    ///
    /// Rerun convenience: adjusts the shot count of the already-built
    /// simulation without rebuilding. The sampling strategy (and its worker
    /// count) is fixed at build time.
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
    /// Execution strategy depends on the sampling strategy:
    /// - `MonteCarlo { workers: 1, .. }`: Runs shots via the Tool
    /// - `MonteCarlo { workers: n, .. }`: Parallelizes shots across n workers
    /// - `ImportanceSampling`: Runs via the Tool with `ImportanceSamplingSimPlugin`
    /// - `SubsetSimulation`: Runs the level-adaptive subset algorithm
    ///   directly; the estimate lands in [`SimulationResults::subset`]
    ///
    /// # Panics
    ///
    /// Panics if the parallel execution plan or subset spec is missing for
    /// the corresponding strategy; `SimNeoBuilder::build()` validates both,
    /// so it cannot happen for simulations constructed through the builder.
    pub fn run(&mut self) -> SimulationResults {
        let config = self.tool.resource::<SimConfig>().clone();

        // Dispatch based on sampling strategy
        match &self.sampling {
            Sampling::MonteCarlo { workers, .. } if *workers > 1 => {
                let plan = self
                    .parallel_plan
                    .as_ref()
                    .expect("parallel plan validated at build time for workers > 1");
                self.run_parallel(&config, plan, *workers)
            }
            Sampling::ImportanceSampling { config: is_config } if is_config.workers > 1 => {
                let spec = self
                    .is_parallel_spec
                    .as_ref()
                    .expect("parallel IS spec captured at build time");
                Self::run_parallel_importance(&config, is_config, spec)
            }
            Sampling::PathEnumeration { config: pe_config } => {
                let spec = self
                    .path_spec
                    .as_ref()
                    .expect("path spec validated at build time");
                let mut explorer = PathExplorer::new(SparseStab::new(spec.num_qubits));

                let mut seen = std::collections::BTreeSet::new();
                let mut outcomes = Vec::new();
                let mut weights = Vec::new();
                for forced_path in PathEnumerator::new(pe_config.max_measurements) {
                    let result = explorer.run_with_path(&spec.circuit, &forced_path);
                    // Different forced paths can realize the same actual path
                    // (deterministic measurements ignore forced bits); keep
                    // each distinct realized path once with its exact
                    // probability.
                    if seen.insert(result.path.signature().to_binary_string()) {
                        weights.push(result.path.probability());
                        outcomes.push(result.outcomes);
                    }
                }

                SimulationResults {
                    outcomes,
                    weights: Some(weights),
                    subset: None,
                    shots: None,
                }
            }
            Sampling::SubsetSimulation { config: ss_config } => {
                let spec = self
                    .subset_spec
                    .as_ref()
                    .expect("subset spec validated at build time");
                let score = ss_config
                    .score
                    .clone()
                    .expect("score fn validated at build time");
                let failure = ss_config
                    .failure
                    .clone()
                    .expect("failure fn validated at build time");
                // config.shots carries samples_per_level, so the rerun
                // override Simulation::shots() applies to it naturally.
                let subset_config = SubsetConfig {
                    samples_per_level: config.shots,
                    threshold_fraction: ss_config.threshold_fraction,
                    max_levels: ss_config.max_levels,
                    min_conditional_prob: ss_config.min_conditional_prob,
                    seed: config.seed,
                };
                let noise = spec.noise.clone();
                let result = SubsetSimulation::new(
                    spec.circuit.clone(),
                    spec.num_qubits,
                    move |o: &MeasurementOutcomes| score(o),
                    move |o: &MeasurementOutcomes| failure(o),
                )
                .with_noise_builder(move || noise.clone())
                .with_config(subset_config)
                .run();
                SimulationResults {
                    outcomes: Vec::new(),
                    weights: None,
                    subset: Some(result),
                    shots: None,
                }
            }
            _ => {
                // Both MonteCarlo{workers:1} and ImportanceSampling run via the Tool.
                // IS uses ImportanceSamplingSimPlugin instead of UnifiedSimulationPlugin.
                self.tool.reset();
                self.tool.run_shots(config.shots);

                // Take results and re-insert empty for next run
                let results = self.tool.take_resource::<SimulationResults>();
                self.tool.insert_resource_mut(SimulationResults::new());
                results
            }
        }
    }

    /// Run importance sampling trials in parallel using rayon.
    ///
    /// Each worker builds its own boosted runner and processes a contiguous
    /// range of global shot indices. Seeds derive from (base seed, global
    /// shot index) alone — the same scheme as the sequential IS systems —
    /// so outcomes and weights are identical for any worker count.
    fn run_parallel_importance(
        config: &SimConfig,
        is_config: &ImportanceSamplingBuilder,
        spec: &StaticCircuitSpec,
    ) -> SimulationResults {
        let shots = config.shots;
        let num_workers = is_config.workers;
        // Reseed per shot only when a base seed was configured; an
        // unseeded run keeps each worker's entropy-seeded runner, matching
        // the Monte Carlo path rather than forcing a deterministic stream.
        let base_seed = config.seed;

        let shots_per_worker = distribute_shots(shots, num_workers);
        let mut start_indices = vec![0usize; num_workers];
        for i in 1..num_workers {
            start_indices[i] = start_indices[i - 1] + shots_per_worker[i - 1];
        }

        let per_worker: Vec<(
            Vec<MeasurementOutcomes>,
            Vec<crate::sampling::weight::SampleWeight>,
        )> = (0..num_workers)
            .into_par_iter()
            .map(|worker_id| {
                let worker_shots = shots_per_worker[worker_id];
                let mut outcomes = Vec::with_capacity(worker_shots);
                let mut weights = Vec::with_capacity(worker_shots);
                if worker_shots == 0 {
                    return (outcomes, weights);
                }

                let mut runner = build_importance_runner(is_config, spec.num_qubits);
                let start = start_indices[worker_id];
                for shot_index in start..start + worker_shots {
                    if let Some(base_seed) = base_seed {
                        seed_importance_runner(&mut runner, base_seed, shot_index);
                    }
                    let result = runner.run_shot_fresh(&spec.circuit);
                    outcomes.push(result.outcomes);
                    weights.push(result.weight);
                }
                (outcomes, weights)
            })
            .collect();

        // Flatten in worker order = global shot order.
        let mut outcomes = Vec::with_capacity(shots);
        let mut weights = Vec::with_capacity(shots);
        for (o, w) in per_worker {
            outcomes.extend(o);
            weights.extend(w);
        }

        SimulationResults {
            outcomes,
            weights: Some(weights),
            subset: None,
            shots: None,
        }
    }

    /// Run shots in parallel using rayon (static circuits with built-in backends).
    ///
    /// Each worker gets its own `Resources` and runs the shared schedule,
    /// so user-registered plugins/hooks fire correctly per worker.
    /// Per-shot seeding is preserved via global shot indices.
    fn run_parallel(
        &self,
        config: &SimConfig,
        plan: &ParallelExecutionPlan,
        num_workers: usize,
    ) -> SimulationResults {
        let shots = config.shots;

        // Distribute shots among workers and compute starting indices
        let shots_per_worker = distribute_shots(shots, num_workers);
        let mut start_indices = vec![0usize; num_workers];
        for i in 1..num_workers {
            start_indices[i] = start_indices[i - 1] + shots_per_worker[i - 1];
        }

        let schedule = self.tool.schedule();

        // Run in parallel, each worker with its own Resources
        let all_results: Vec<SimulationResults> = (0..num_workers)
            .into_par_iter()
            .map(|worker_id| {
                let worker_shots = shots_per_worker[worker_id];
                if worker_shots == 0 {
                    return SimulationResults::new();
                }

                // Build per-worker Resources with the same configuration
                let mut resources = Resources::new();
                resources.insert(SimConfig {
                    shots: worker_shots,
                    seed: config.seed,
                });
                resources.insert(ExplicitNumQubits(None));
                resources.insert(SimulationResults::new());
                resources.insert(UnifiedShotState {
                    quantum_runner: plan.quantum_runner_factory.create_runner(config.seed),
                    command_source: plan.command_source_factory.create_source(),
                    shot_index: 0,
                });

                // Run Startup. Since the worker state is already assembled, the
                // unified startup system only resets the command source and clears results.
                schedule.run_stage(Stage::Startup, &mut resources);

                // Set global starting shot index so per-shot seeding matches sequential
                resources.get_mut::<UnifiedShotState>().shot_index = start_indices[worker_id];

                // Run shot loop (PreShot/Execute/PostShot per shot)
                for _ in 0..worker_shots {
                    schedule.run_stage(Stage::PreShot, &mut resources);
                    schedule.run_stage(Stage::Execute, &mut resources);
                    schedule.run_stage(Stage::PostShot, &mut resources);
                }

                // Run Finish
                schedule.run_stage(Stage::Finish, &mut resources);

                // Extract results
                resources.remove::<SimulationResults>()
            })
            .collect();

        // Flatten in deterministic order, merging per-worker register shots
        // when the source produced them.
        let mut outcomes = Vec::new();
        let mut shots: Option<pecos_results::ShotVec> = None;
        for worker_results in all_results {
            outcomes.extend(worker_results.outcomes);
            if let Some(worker_shots) = worker_results.shots {
                shots
                    .get_or_insert_with(pecos_results::ShotVec::new)
                    .shots
                    .extend(worker_shots.shots);
            }
        }

        // Cross-worker alignment: each worker's records are index-aligned
        // (the post-shot debug-assert enforces it per worker), but merging
        // a worker that produced shot records with one that did not would
        // desync the flattened vectors. Guard the merged invariant too.
        debug_assert!(
            shots
                .as_ref()
                .is_none_or(|s| s.shots.len() == outcomes.len()),
            "merged shot records and outcomes are misaligned ({} shot records vs {} \
             outcomes); all workers must agree on whether the source produces shot records",
            shots.as_ref().map_or(0, |s| s.shots.len()),
            outcomes.len(),
        );

        SimulationResults {
            outcomes,
            weights: None,
            subset: None,
            shots,
        }
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

// --- Convenience Entry Point ---

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
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new()
///     .pz(&[0]).h(&[0]).mz(&[0])
///     .build();
///
/// let results = sim_neo(circuit).auto()
///     .depolarizing(0.01)
///     .sampling(monte_carlo(1000))
///     .seed(42)
///     .build()
///     .run();
/// ```
///
/// ## QASM Program
///
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo};
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
/// let results = sim_neo(qasm).auto()
///     .classical(qasm_engine())
///     .depolarizing(0.01)
///     .sampling(monte_carlo(1000))
///     .seed(42)
///     .build()
///     .run();
/// ```
///
/// ## Reusable Simulation
///
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
/// let mut sim = sim_neo(circuit).auto()
///     .sampling(monte_carlo(1000))
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
/// ```no_run
/// use pecos_neo::tool::{monte_carlo, sim_neo_builder};
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
///     .with_engine(qasm_engine().qasm(qasm))
///     .depolarizing(0.01)
///     .sampling(monte_carlo(1000))
///     .seed(42)
///     .build()
///     .run();
/// ```
#[must_use]
pub fn sim_neo_builder() -> SimNeoBuilder {
    SimNeoBuilder::empty()
}

// --- Parallel Execution Helpers ---

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
#[allow(clippy::cast_precision_loss)] // statistical tests use count as f64
mod tests {
    use super::*;
    use crate::command::CommandBuilder;
    use crate::noise::{ComposableNoiseModel, SingleQubitChannel};
    use crate::program::ConditionalProgram;
    use pecos_core::QubitId;

    #[test]
    fn test_sim_neo_basic() {
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .x(&[0]) // Flip to |1>
            .mz(&[0])
            .build();

        let mut sim = sim_neo(circuit)
            .auto()
            .sampling(monte_carlo(10))
            .seed(42)
            .build();

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
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let mut sim = sim_neo(circuit).auto().sampling(monte_carlo(5)).build();

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
            .pz(&[0])
            .h(&[0]) // Superposition - outcome depends on RNG
            .mz(&[0])
            .build();

        // Same seed should produce same results
        let results1 = sim_neo(circuit.clone())
            .auto()
            .sampling(monte_carlo(20))
            .seed(42)
            .build()
            .run();

        let results2 = sim_neo(circuit)
            .auto()
            .sampling(monte_carlo(20))
            .seed(42)
            .build()
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
    fn test_sim_neo_with_noise() {
        // Circuit: prep |0>, Z gate, measure
        // Z|0> = |0>, so without noise we'd always measure 0
        // But with depolarizing noise on the Z gate, we'll see errors
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .z(&[0]) // Single-qubit gate to trigger noise
            .mz(&[0])
            .build();

        // Very high error rate - this will definitely flip some outcomes
        let noise = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.5));

        let results = sim_neo(circuit)
            .auto()
            .noise(noise)
            .sampling(monte_carlo(100))
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
            .pz(&[0])
            .z(&[0]) // Single-qubit gate to trigger noise
            .mz(&[0])
            .build();

        let noise1 = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.5));
        let noise2 = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.5));

        let results1 = sim_neo(circuit.clone())
            .auto()
            .noise(noise1)
            .sampling(monte_carlo(20))
            .seed(42)
            .build()
            .run();

        let results2 = sim_neo(circuit)
            .auto()
            .noise(noise2)
            .sampling(monte_carlo(20))
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
        let circuit = CommandBuilder::new().pz(&[0]).z(&[0]).mz(&[0]).build();

        // This uses the From<C: NoiseChannel> impl for ComposableNoiseModel
        let results = sim_neo(circuit)
            .auto()
            .noise(SingleQubitChannel::depolarizing(0.5))
            .sampling(monte_carlo(50))
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

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        // Pass builder directly - no .build() needed!
        let results = sim_neo(circuit)
            .auto()
            .noise(GeneralNoiseModelBuilder::new().with_p1(0.3))
            .sampling(monte_carlo(100))
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

        assert!(
            zeros > 0,
            "Expected some errors from 30% depolarizing noise"
        );
    }

    #[test]
    fn test_sim_neo_convenience_depolarizing() {
        // Test the .depolarizing(p) convenience method
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .x(&[0])
            .cx(&[(0, 1)])
            .mz(&[0])
            .mz(&[1])
            .build();

        let results = sim_neo(circuit)
            .auto()
            .depolarizing(0.2) // 20% on both 1Q and 2Q gates
            .sampling(monte_carlo(100))
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 100);

        // Should see some errors from high depolarizing rate
        let correct: usize = results
            .outcomes
            .iter()
            .filter(|o| {
                o.get_bit(QubitId(0)).unwrap_or(false) && o.get_bit(QubitId(1)).unwrap_or(false)
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

        let circuit = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .auto()
            .noise(GeneralNoiseModelBuilder::new().with_p_meas_symmetric(0.15))
            .sampling(monte_carlo(200))
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

        let circuit = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .auto()
            .noise(GeneralNoiseModelBuilder::new().with_p_prep(0.20))
            .sampling(monte_carlo(200))
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
        let results = sim_neo(qasm)
            .auto()
            .sampling(monte_carlo(10))
            .seed(42)
            .run();

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
            .auto()
            .classical(pecos_qasm::qasm_engine())
            .sampling(monte_carlo(10))
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
        let results = sim_neo(program)
            .auto()
            .sampling(monte_carlo(50))
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 50);

        // Bell state: both qubits should be correlated
        for outcome in &results.outcomes {
            let q0 = outcome.get_bit(QubitId(0)).unwrap_or(false);
            let q1 = outcome.get_bit(QubitId(1)).unwrap_or(false);
            assert_eq!(q0, q1, "Bell state qubits should be correlated");
        }
    }

    #[test]
    fn test_sim_neo_monte_carlo_sampling() {
        // Test Monte Carlo sampling with multiple workers
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .x(&[0]) // Flip to |1>
            .mz(&[0])
            .build();

        // Use .workers() convenience method for Monte Carlo
        let results = sim_neo(circuit)
            .auto()
            .sampling(monte_carlo(100).workers(4))
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
            .pz(&[0])
            .h(&[0]) // Superposition
            .mz(&[0])
            .build();

        let results1 = sim_neo(circuit.clone())
            .auto()
            .sampling(monte_carlo(50).workers(4))
            .seed(42)
            .run();

        let results2 = sim_neo(circuit)
            .auto()
            .sampling(monte_carlo(50).workers(4))
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
    fn test_sim_neo_sampling_explicit() {
        // Test explicit sampling configuration with workers on the builder
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .auto()
            .sampling(monte_carlo(20).workers(2))
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
    fn test_sim_neo_sampling_order_independent() {
        // Regression for the old top-level .workers() footgun: builder calls
        // must commute. .sampling() before or after other config gives the
        // same results.
        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let r1 = sim_neo(circuit.clone())
            .auto()
            .sampling(monte_carlo(30))
            .seed(7)
            .run();
        let r2 = sim_neo(circuit)
            .auto()
            .seed(7)
            .sampling(monte_carlo(30))
            .run();

        assert_eq!(r1.outcomes.len(), r2.outcomes.len());
        for (o1, o2) in r1.outcomes.iter().zip(r2.outcomes.iter()) {
            assert_eq!(o1.get_bit(QubitId(0)), o2.get_bit(QubitId(0)));
        }
    }

    #[test]
    #[should_panic(expected = "No sampling strategy set")]
    fn test_sim_neo_missing_sampling_is_build_error() {
        let circuit = CommandBuilder::new().pz(&[0]).mz(&[0]).build();
        let _ = sim_neo(circuit).auto().build();
    }

    #[test]
    #[should_panic(expected = "No quantum backend set")]
    fn test_sim_neo_missing_quantum_backend_is_build_error() {
        // Explicit-by-default: no silent SparseStab. Either .quantum(..) or
        // .auto() must be called.
        let circuit = CommandBuilder::new().pz(&[0]).mz(&[0]).build();
        let _ = sim_neo(circuit).sampling(monte_carlo(10)).build();
    }

    #[test]
    fn test_sim_neo_auto_selects_backend_for_static_circuit() {
        // .auto() opts into automatic backend selection (SparseStab) for
        // static circuits, which previously rejected .auto() entirely.
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();
        let results = sim_neo(circuit)
            .auto()
            .sampling(monte_carlo(10))
            .seed(1)
            .run();
        assert_eq!(results.len(), 10);
        for outcome in &results.outcomes {
            assert!(outcome.get_bit(QubitId(0)).unwrap());
        }
    }

    #[test]
    fn test_sim_neo_explicit_quantum_overrides_auto() {
        // .auto() plus explicit .quantum() is allowed; the explicit choice
        // wins regardless of call order.
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .x(&[0])
            .t(&[0])
            .mz(&[0])
            .build();
        // T gate requires the state-vector backend; if auto's SparseStab
        // choice won, this would fail to execute.
        let results = sim_neo(circuit)
            .auto()
            .quantum(state_vector())
            .sampling(monte_carlo(5))
            .seed(1)
            .run();
        assert_eq!(results.len(), 5);
    }

    #[test]
    #[should_panic(expected = ".shots() cannot be combined with")]
    fn test_sim_neo_shots_conflicts_with_sampling() {
        let circuit = CommandBuilder::new().pz(&[0]).mz(&[0]).build();
        let _ = sim_neo(circuit)
            .auto()
            .sampling(monte_carlo(10))
            .shots(20)
            .build();
    }

    #[test]
    #[should_panic(expected = "deprecated .workers() cannot be combined")]
    fn test_sim_neo_legacy_workers_conflicts_with_sampling() {
        // The old footgun: .sampling(importance_sampling(..)).workers(n)
        // silently discarded the importance-sampling config. Now it fails
        // loudly at build time.
        let circuit = CommandBuilder::new().pz(&[0]).mz(&[0]).build();
        #[allow(deprecated)]
        let _ = sim_neo(circuit)
            .auto()
            .sampling(importance_sampling(10))
            .workers(4)
            .build();
    }

    #[test]
    #[should_panic(
        expected = "Parallel Monte Carlo (workers > 1) requires per-worker construction"
    )]
    fn test_sim_neo_parallel_dynamic_source_fails_at_build_not_run() {
        // The parallel-incapable combination must be rejected by .build(),
        // before any shot executes.
        let _ = sim_neo(deterministic_conditional_program())
            .auto()
            .sampling(monte_carlo(2).workers(2))
            .build();
    }

    #[test]
    fn test_sim_neo_shots_shortcut_matches_sampling() {
        // .shots(n) must behave exactly like .sampling(monte_carlo(n)) -- it is
        // the blessed shorthand for that common case.
        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let shortcut = sim_neo(circuit.clone()).auto().shots(40).seed(11).run();
        let explicit = sim_neo(circuit)
            .auto()
            .sampling(monte_carlo(40))
            .seed(11)
            .run();

        assert_eq!(shortcut.outcomes.len(), 40);
        assert_eq!(shortcut.outcomes.len(), explicit.outcomes.len());
        for (o1, o2) in shortcut.outcomes.iter().zip(explicit.outcomes.iter()) {
            assert_eq!(o1.get_bit(QubitId(0)), o2.get_bit(QubitId(0)));
        }
    }

    #[test]
    fn test_sim_neo_shots_with_deprecated_workers_still_parallel() {
        // Blessed .shots(m) combined with the deprecated .workers(n) forwarder
        // maps onto MonteCarlo { shots: m, workers: n } and still runs.
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        #[allow(deprecated)]
        let results = sim_neo(circuit).auto().workers(2).shots(30).seed(5).run();

        assert_eq!(results.len(), 30);
        for outcome in &results.outcomes {
            assert!(outcome.get_bit(QubitId(0)).unwrap());
        }
    }

    #[test]
    fn test_sim_neo_importance_sampling_shot_count_on_builder() {
        // importance_sampling(shots) drives the trial count directly.
        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .auto()
            .sampling(importance_sampling(35).with_uniform_error(0.01))
            .seed(3)
            .run();

        assert_eq!(results.len(), 35);
        assert!(results.has_weights());
    }

    #[test]
    fn test_sim_neo_importance_sampling_parallel_matches_sequential() {
        // Per-shot seeding from global indices: any worker count gives
        // identical outcomes AND weights.
        let circuit = CommandBuilder::new()
            .pz(&[0, 1])
            .h(&[0])
            .cx(&[(0, 1)])
            .mz(&[0, 1])
            .build();
        let run = |workers: usize| {
            sim_neo(circuit.clone())
                .auto()
                .sampling(
                    importance_sampling(60)
                        .with_uniform_error(0.01)
                        .with_boost(10.0)
                        .workers(workers),
                )
                .seed(42)
                .run()
        };

        let sequential = run(1);
        let parallel = run(4);

        assert_eq!(sequential.outcomes.len(), 60);
        assert_eq!(sequential.outcomes.len(), parallel.outcomes.len());
        for (i, (s, p)) in sequential
            .outcomes
            .iter()
            .zip(parallel.outcomes.iter())
            .enumerate()
        {
            assert_eq!(
                s.get_bit(QubitId(0)),
                p.get_bit(QubitId(0)),
                "Shot {i} qubit 0 should match"
            );
            assert_eq!(
                s.get_bit(QubitId(1)),
                p.get_bit(QubitId(1)),
                "Shot {i} qubit 1 should match"
            );
        }

        let sw = sequential.weights.as_ref().unwrap();
        let pw = parallel.weights.as_ref().unwrap();
        assert_eq!(sw.len(), pw.len());
        for (i, (a, b)) in sw.iter().zip(pw.iter()).enumerate() {
            assert!(
                (a.weight() - b.weight()).abs() < 1e-12,
                "Weight at shot {i} should match"
            );
        }
    }

    #[test]
    fn test_sim_neo_importance_sampling_parallel_deterministic() {
        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
        let run = || {
            sim_neo(circuit.clone())
                .auto()
                .sampling(importance_sampling(30).with_uniform_error(0.01).workers(3))
                .seed(5)
                .run()
        };
        let r1 = run();
        let r2 = run();
        for (o1, o2) in r1.outcomes.iter().zip(r2.outcomes.iter()) {
            assert_eq!(o1.get_bit(QubitId(0)), o2.get_bit(QubitId(0)));
        }
    }

    #[test]
    fn test_sim_neo_clifford_angle_rotation_on_stabilizer_backend() {
        // QASM compiles s -> u1(pi/2) -> rz(pi/2); Clifford backends must
        // execute Clifford-angle rotations via the CliffordRotation
        // decomposition instead of failing with NoDecomposition.
        // h . rz(pi/2) . rz(pi/2) . h = h z h = x, so the outcome is
        // deterministically 1.
        use pecos_core::Angle64;
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .h(&[0])
            .rz(&[0], Angle64::QUARTER_TURN)
            .rz(&[0], Angle64::QUARTER_TURN)
            .h(&[0])
            .mz(&[0])
            .build();

        let results = sim_neo(circuit)
            .quantum(sparse_stab())
            .sampling(monte_carlo(10))
            .seed(1)
            .run();

        for outcome in &results.outcomes {
            assert_eq!(
                outcome.get_bit(QubitId(0)),
                Some(true),
                "h z h = x must deterministically flip"
            );
        }
    }

    // --- Shot/ShotVec Production Tests ---

    #[test]
    fn test_sim_neo_static_circuit_shot_vec_synthesis() {
        let circuit = CommandBuilder::new()
            .pz(&[0, 1])
            .x(&[0])
            .mz(&[0, 1])
            .build();
        let results = sim_neo(circuit)
            .auto()
            .sampling(monte_carlo(3))
            .seed(1)
            .run();

        assert!(
            results.shots.is_none(),
            "Static circuits have no register data"
        );

        // Without a map: single "meas" register, bits in ascending qubit order.
        let synthesized = results.to_shot_vec(None);
        assert_eq!(synthesized.shots.len(), 3);
        for shot in &synthesized.shots {
            assert_eq!(shot.data["meas"].to_bitstring().unwrap(), "10");
        }

        // With a map: one BitVec register per name.
        let mut map = RegisterMap::new();
        map.add_register("a", &[QubitId(0)]);
        map.add_register("b", &[QubitId(1)]);
        let named = results.to_shot_vec(Some(&map));
        for shot in &named.shots {
            assert_eq!(shot.data["a"].to_bitstring().unwrap(), "1");
            assert_eq!(shot.data["b"].to_bitstring().unwrap(), "0");
        }
    }

    #[cfg(feature = "qasm")]
    #[test]
    fn test_sim_neo_qasm_produces_register_shots() {
        // The classical engine's named cregs flow through the adapter into
        // SimulationResults::shots — including the feedback-conditioned bit.
        let program = pecos_programs::Qasm::from_string(deterministic_conditional_qasm());
        let results = sim_neo(program)
            .auto()
            .quantum(sparse_stab())
            .sampling(monte_carlo(5))
            .seed(42)
            .run();

        let shots = results
            .shots
            .as_ref()
            .expect("classical engines produce register shots");
        assert_eq!(shots.shots.len(), 5);
        for shot in &shots.shots {
            assert_eq!(shot.data["c"].to_bitstring().unwrap(), "11");
        }
        // to_shot_vec returns the engine-produced registers as-is.
        assert_eq!(results.to_shot_vec(None).shots.len(), 5);
    }

    #[cfg(feature = "qasm")]
    #[test]
    fn test_sim_neo_qasm_parallel_register_shots() {
        let program = pecos_programs::Qasm::from_string(deterministic_conditional_qasm());
        let results = sim_neo(program)
            .auto()
            .quantum(sparse_stab())
            .sampling(monte_carlo(6).workers(2))
            .seed(42)
            .run();

        let shots = results
            .shots
            .as_ref()
            .expect("parallel adapter runs merge register shots");
        assert_eq!(shots.shots.len(), 6);
        for shot in &shots.shots {
            assert_eq!(shot.data["c"].to_bitstring().unwrap(), "11");
        }
    }

    // --- Path Enumeration Strategy Tests ---

    #[test]
    fn test_sim_neo_path_enumeration_bell_pair() {
        // H + CX: first measurement is random (two branches), second is
        // deterministic given the first. Exactly two paths, p = 0.5 each,
        // with perfectly correlated outcomes.
        let circuit = CommandBuilder::new()
            .pz(&[0, 1])
            .h(&[0])
            .cx(&[(0, 1)])
            .mz(&[0, 1])
            .build();

        let results = sim_neo(circuit)
            .quantum(sparse_stab())
            .sampling(path_enumeration(1))
            .run();

        assert_eq!(results.outcomes.len(), 2, "Two measurement branches");
        let weights = results.weights.as_ref().unwrap();
        let total: f64 = weights
            .iter()
            .map(crate::sampling::weight::SampleWeight::weight)
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "Complete enumeration sums to 1"
        );
        for (outcome, weight) in results.outcomes.iter().zip(weights) {
            assert!((weight.weight() - 0.5).abs() < 1e-12);
            assert_eq!(
                outcome.get_bit(QubitId(0)),
                outcome.get_bit(QubitId(1)),
                "Bell pair outcomes must be correlated"
            );
        }
    }

    #[test]
    fn test_sim_neo_path_enumeration_dedupes_deterministic_circuit() {
        // X then measure: fully deterministic, one realized path with p = 1
        // even though 2^2 forced paths are enumerated.
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit).auto().sampling(path_enumeration(2)).run();

        assert_eq!(results.outcomes.len(), 1, "One distinct path");
        let weights = results.weights.as_ref().unwrap();
        assert!((weights[0].weight() - 1.0).abs() < 1e-12);
        assert_eq!(results.outcomes[0].get_bit(QubitId(0)), Some(true));
    }

    #[test]
    fn test_sim_neo_path_enumeration_three_qubit_uniform() {
        // Three independent H measurements: 8 paths, p = 1/8 each.
        let results = sim_neo(three_qubit_h_circuit())
            .auto()
            .sampling(path_enumeration(3))
            .run();

        assert_eq!(results.outcomes.len(), 8);
        let weights = results.weights.as_ref().unwrap();
        let total: f64 = weights
            .iter()
            .map(crate::sampling::weight::SampleWeight::weight)
            .sum();
        assert!((total - 1.0).abs() < 1e-12);
        for weight in weights {
            assert!((weight.weight() - 0.125).abs() < 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "remove .noise()")]
    fn test_sim_neo_path_enumeration_rejects_noise() {
        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
        let _ = sim_neo(circuit)
            .auto()
            .noise(SingleQubitChannel::depolarizing(0.1))
            .sampling(path_enumeration(1))
            .build();
    }

    #[test]
    #[should_panic(expected = "Path enumeration currently supports only the sparse_stab() backend")]
    fn test_sim_neo_path_enumeration_rejects_state_vector_backend() {
        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
        let _ = sim_neo(circuit)
            .quantum(state_vector())
            .sampling(path_enumeration(1))
            .build();
    }

    #[test]
    #[should_panic(expected = "more than 16M paths")]
    fn test_sim_neo_path_enumeration_rejects_huge_enumeration() {
        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
        let _ = sim_neo(circuit)
            .auto()
            .sampling(path_enumeration(25))
            .build();
    }

    // --- Subset Simulation Strategy Tests ---

    fn three_qubit_h_circuit() -> CommandQueue {
        CommandBuilder::new()
            .pz(&[0, 1, 2])
            .h(&[0, 1, 2])
            .mz(&[0, 1, 2])
            .build()
    }

    fn count_ones(outcomes: &MeasurementOutcomes) -> f64 {
        outcomes.iter().filter(|m| m.outcome).count() as f64
    }

    fn all_ones(outcomes: &MeasurementOutcomes) -> bool {
        outcomes.iter().count() > 0 && outcomes.iter().all(|m| m.outcome)
    }

    #[test]
    fn test_sim_neo_subset_simulation_estimates_known_probability() {
        // Three H gates: P(all three measure 1) = 1/8.
        let results = sim_neo(three_qubit_h_circuit())
            .quantum(sparse_stab())
            .sampling(subset_simulation(2000).score(count_ones).failure(all_ones))
            .seed(42)
            .run();

        assert!(results.outcomes.is_empty());
        let subset = results.subset.expect("subset strategy returns an estimate");
        let p = subset.probability();
        assert!(
            (0.08..=0.20).contains(&p),
            "Expected estimate near 1/8, got {p:.4}"
        );
        assert!(subset.total_samples >= 2000);
    }

    #[test]
    fn test_sim_neo_subset_simulation_certain_event() {
        // X gate makes the failure event certain.
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();
        let results = sim_neo(circuit)
            .auto()
            .sampling(subset_simulation(200).score(count_ones).failure(all_ones))
            .seed(7)
            .run();

        let subset = results.subset.expect("subset estimate");
        assert!(
            (subset.probability() - 1.0).abs() < 1e-9,
            "Certain event should estimate ~1.0, got {}",
            subset.probability()
        );
    }

    #[test]
    fn test_sim_neo_subset_simulation_with_noise() {
        // Depolarizing(0.3) after Z: P(flip) = 2/3 * 0.3 = 0.2.
        let circuit = CommandBuilder::new().pz(&[0]).z(&[0]).mz(&[0]).build();
        let noise = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.3));

        let results = sim_neo(circuit)
            .quantum(sparse_stab())
            .noise(noise)
            .sampling(subset_simulation(2000).score(count_ones).failure(all_ones))
            .seed(11)
            .run();

        let p = results.subset.expect("subset estimate").probability();
        assert!(
            (0.12..=0.28).contains(&p),
            "Expected estimate near 0.2, got {p:.4}"
        );
    }

    #[test]
    fn test_sim_neo_subset_simulation_deterministic() {
        let run = || {
            sim_neo(three_qubit_h_circuit())
                .auto()
                .sampling(subset_simulation(500).score(count_ones).failure(all_ones))
                .seed(99)
                .run()
                .subset
                .expect("subset estimate")
                .probability()
        };
        assert!((run() - run()).abs() < 1e-15, "Same seed, same estimate");
    }

    #[test]
    #[should_panic(expected = "requires both .score(..) and .failure(..)")]
    fn test_sim_neo_subset_simulation_missing_fns_is_build_error() {
        let _ = sim_neo(three_qubit_h_circuit())
            .auto()
            .sampling(subset_simulation(100))
            .build();
    }

    #[test]
    #[should_panic(expected = "Subset simulation requires a static circuit")]
    fn test_sim_neo_subset_simulation_rejects_dynamic_source() {
        let _ = sim_neo(deterministic_conditional_program())
            .auto()
            .sampling(subset_simulation(100).score(count_ones).failure(all_ones))
            .build();
    }

    #[test]
    #[should_panic(expected = "supports only the sparse_stab() backend")]
    fn test_sim_neo_subset_simulation_rejects_state_vector_backend() {
        let _ = sim_neo(three_qubit_h_circuit())
            .quantum(state_vector())
            .sampling(subset_simulation(100).score(count_ones).failure(all_ones))
            .build();
    }

    #[test]
    #[should_panic(expected = "biased upward")]
    fn test_sim_neo_subset_multilevel_without_opt_in_is_build_error() {
        // max_levels > 1 engages the biased multi-level estimator and must
        // be refused unless explicitly acknowledged.
        let _ = sim_neo(three_qubit_h_circuit())
            .auto()
            .sampling(
                subset_simulation(100)
                    .score(count_ones)
                    .failure(all_ones)
                    .max_levels(5),
            )
            .build();
    }

    #[test]
    fn test_sim_neo_subset_multilevel_runs_with_opt_in() {
        // With the explicit acknowledgment, the multi-level path runs.
        let results = sim_neo(three_qubit_h_circuit())
            .auto()
            .sampling(
                subset_simulation(500)
                    .score(count_ones)
                    .failure(all_ones)
                    .max_levels(5)
                    .allow_biased_multilevel(),
            )
            .seed(42)
            .run();
        assert!(
            results.subset.is_some(),
            "multi-level opt-in must produce an estimate"
        );
    }

    #[test]
    fn test_sim_neo_subset_default_is_single_level() {
        // The default (no .max_levels) is a single unbiased level: exactly
        // one level in the result.
        let results = sim_neo(three_qubit_h_circuit())
            .auto()
            .sampling(subset_simulation(500).score(count_ones).failure(all_ones))
            .seed(42)
            .run();
        let subset = results.subset.expect("subset estimate");
        assert_eq!(
            subset.levels.len(),
            1,
            "default subset_simulation must run a single (unbiased) level"
        );
    }

    #[test]
    fn test_sim_neo_single_worker_matches_parallel() {
        // Critical test: 1 worker and multiple workers should produce identical
        // results with the same seed (they use the same per-shot seeding scheme)
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .h(&[0]) // Superposition - outcome depends on RNG
            .mz(&[0])
            .build();

        // Run with default (1 worker)
        let single_results = sim_neo(circuit.clone())
            .auto()
            .sampling(monte_carlo(50))
            .seed(42)
            .run();

        // Run with parallel Monte Carlo sampling (4 workers)
        let parallel_results = sim_neo(circuit)
            .auto()
            .sampling(monte_carlo(50).workers(4))
            .seed(42)
            .run();

        // Results should be identical
        assert_eq!(
            single_results.outcomes.len(),
            parallel_results.outcomes.len()
        );
        for (i, (single, par)) in single_results
            .outcomes
            .iter()
            .zip(parallel_results.outcomes.iter())
            .enumerate()
        {
            assert_eq!(
                single.get_bit(QubitId(0)),
                par.get_bit(QubitId(0)),
                "Single-worker and parallel should produce identical results at shot {i}"
            );
        }
    }

    #[test]
    fn test_sim_neo_noisy_single_worker_matches_parallel() {
        // Critical test: parallel noisy execution should produce identical results
        // to single-worker noisy execution with the same seed.
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .h(&[0])
            .z(&[0]) // Trigger single-qubit noise
            .mz(&[0])
            .build();

        let noise_single =
            ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.3));
        let noise_par =
            ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.3));

        // Run with single worker (default)
        let single_results = sim_neo(circuit.clone())
            .auto()
            .noise(noise_single)
            .sampling(monte_carlo(50))
            .seed(42)
            .run();

        // Run with parallel Monte Carlo sampling
        let parallel_results = sim_neo(circuit)
            .auto()
            .noise(noise_par)
            .sampling(monte_carlo(50).workers(4))
            .seed(42)
            .run();

        // Results should be identical shot-for-shot
        assert_eq!(
            single_results.outcomes.len(),
            parallel_results.outcomes.len()
        );
        for (i, (single, par)) in single_results
            .outcomes
            .iter()
            .zip(parallel_results.outcomes.iter())
            .enumerate()
        {
            assert_eq!(
                single.get_bit(QubitId(0)),
                par.get_bit(QubitId(0)),
                "Noisy single-worker and parallel should produce identical results at shot {i}"
            );
        }
    }

    #[test]
    fn test_sim_neo_noisy_parallel_deterministic() {
        // Two parallel noisy runs with the same seed should produce identical results.

        let circuit = CommandBuilder::new()
            .pz(&[0])
            .h(&[0])
            .z(&[0])
            .mz(&[0])
            .build();

        let noise1 = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.3));
        let noise2 = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.3));

        let results1 = sim_neo(circuit.clone())
            .auto()
            .noise(noise1)
            .sampling(monte_carlo(50).workers(4))
            .seed(42)
            .run();

        let results2 = sim_neo(circuit)
            .auto()
            .noise(noise2)
            .sampling(monte_carlo(50).workers(4))
            .seed(42)
            .run();

        for (i, (r1, r2)) in results1
            .outcomes
            .iter()
            .zip(results2.outcomes.iter())
            .enumerate()
        {
            assert_eq!(
                r1.get_bit(QubitId(0)),
                r2.get_bit(QubitId(0)),
                "Parallel noisy runs with same seed should be deterministic at shot {i}"
            );
        }
    }

    #[test]
    fn test_sim_neo_quantum_sparse_stab() {
        // Test explicitly selecting sparse stabilizer backend
        use super::sparse_stab;

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .quantum(sparse_stab())
            .sampling(monte_carlo(10))
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
    fn test_sim_neo_quantum_stabilizer() {
        // Test explicitly selecting the stable public stabilizer backend.
        use super::stabilizer;

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .quantum(stabilizer())
            .sampling(monte_carlo(10))
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
    fn test_sim_neo_quantum_engine_builder_adapter() {
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .x(&[0])
            .mz(&[0])
            .pz(&[1])
            .h(&[1])
            .mz(&[1])
            .build();

        let results = sim_neo(circuit)
            .quantum(pecos_engines::stabilizer())
            .sampling(monte_carlo(12))
            .seed(42)
            .run();

        assert_eq!(results.len(), 12);
        for outcome in &results.outcomes {
            assert!(
                outcome.get_bit(QubitId(0)).unwrap(),
                "X gate should produce |1>"
            );
            assert!(
                outcome.get_bit(QubitId(1)).is_some(),
                "QuantumEngine adapter should return measurement outcomes by qubit"
            );
        }
    }
    #[test]
    fn test_sim_neo_quantum_engine_builder_adapter_preserves_engine_gate_capabilities() {
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .h(&[0])
            .t(&[0])
            .mz(&[0])
            .build();

        let results = sim_neo(circuit)
            .quantum(pecos_engines::state_vector())
            .sampling(monte_carlo(8))
            .seed(123)
            .run();

        assert_eq!(results.len(), 8);
        for outcome in &results.outcomes {
            assert!(
                outcome.get_bit(QubitId(0)).is_some(),
                "QuantumEngine adapter should preserve state-vector support for T gates"
            );
        }
    }
    #[test]
    #[should_panic(
        expected = "QuantumEngineBuilder backends do not support sim_neo noise modeling"
    )]
    fn test_sim_neo_quantum_engine_builder_rejects_composable_noise() {
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();
        let noise = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.1));

        let _ = sim_neo(circuit)
            .quantum(pecos_engines::stabilizer())
            .noise(noise)
            .sampling(monte_carlo(1))
            .run();
    }
    #[test]
    #[should_panic(
        expected = "QuantumEngineBuilder backends do not support sim_neo noise modeling"
    )]
    fn test_sim_neo_quantum_engine_builder_parallel_rejects_composable_noise() {
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();
        let noise = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.1));

        let _ = sim_neo(circuit)
            .quantum(pecos_engines::stabilizer())
            .noise(noise)
            .sampling(monte_carlo(2).workers(2))
            .run();
    }
    #[test]
    #[should_panic(
        expected = "QuantumEngineBuilder backend does not support sim_neo gate definitions"
    )]
    fn test_sim_neo_quantum_engine_builder_rejects_gate_definitions() {
        use crate::extensible::GateDefinitions;

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let _ = sim_neo(circuit)
            .quantum(pecos_engines::stabilizer())
            .gate_definitions(GateDefinitions::new())
            .sampling(monte_carlo(1))
            .run();
    }
    #[test]
    #[should_panic(
        expected = "QuantumEngineBuilder backend does not support sim_neo gate decomposition depth"
    )]
    fn test_sim_neo_quantum_engine_builder_rejects_max_decomp_depth() {
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let _ = sim_neo(circuit)
            .quantum(pecos_engines::stabilizer())
            .max_decomp_depth(20)
            .sampling(monte_carlo(1))
            .run();
    }
    #[test]
    #[should_panic(
        expected = "QuantumEngineBuilder backend does not support sim_neo gate overrides"
    )]
    fn test_sim_neo_quantum_engine_builder_rejects_gate_overrides() {
        use crate::extensible::gates;
        use crate::runner::GateOverrides;

        let overrides =
            GateOverrides::<SparseStab>::new().register(gates::X, |_sim, _angles, _qubits| true);
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let _ = sim_neo(circuit)
            .quantum(pecos_engines::stabilizer())
            .gate_overrides(overrides)
            .sampling(monte_carlo(1))
            .run();
    }
    #[test]
    #[should_panic(
        expected = "QuantumEngineBuilder backend does not support sim_neo event handlers"
    )]
    fn test_sim_neo_quantum_engine_builder_rejects_event_handlers() {
        let handlers =
            EventHandlers::new().on_before_gate(|_ctx| crate::noise::NoiseResponse::None);
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let _ = sim_neo(circuit)
            .quantum(pecos_engines::stabilizer())
            .event_handlers(handlers)
            .sampling(monte_carlo(1))
            .run();
    }
    #[test]
    fn test_sim_neo_quantum_engine_builder_parallel_static_circuit() {
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .quantum(pecos_engines::stabilizer())
            .sampling(monte_carlo(6).workers(2))
            .seed(42)
            .run();

        assert_eq!(results.len(), 6);
        for outcome in &results.outcomes {
            assert_eq!(outcome.get_bit(QubitId(0)), Some(true));
            assert_eq!(outcome.len(), 1);
        }
    }
    #[test]
    fn test_sim_neo_quantum_engine_builder_parallel_preserves_gate_capabilities() {
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .x(&[0])
            .t(&[0])
            .mz(&[0])
            .build();

        let results = sim_neo(circuit)
            .quantum(pecos_engines::state_vector())
            .sampling(monte_carlo(6).workers(2))
            .seed(42)
            .run();

        assert_eq!(results.len(), 6);
        for outcome in &results.outcomes {
            assert_eq!(outcome.get_bit(QubitId(0)), Some(true));
            assert_eq!(outcome.len(), 1);
        }
    }

    fn deterministic_conditional_program() -> Box<dyn CommandSource + Send + Sync> {
        let initial = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();
        let branch = |outcomes: &MeasurementOutcomes| {
            if outcomes.get_bit(QubitId(0)) == Some(true) {
                Some(CommandBuilder::new().x(&[1]).mz(&[1]).build())
            } else {
                Some(CommandBuilder::new().mz(&[1]).build())
            }
        };
        Box::new(ConditionalProgram::new(initial, branch, 2))
    }

    #[cfg(feature = "qasm")]
    fn deterministic_conditional_qasm() -> &'static str {
        r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            x q[0];
            measure q[0] -> c[0];
            if (c[0] == 1) x q[1];
            measure q[1] -> c[1];
        "#
    }

    #[test]
    fn test_sim_neo_dynamic_command_source_native_stabilizer() {
        let results = sim_neo(deterministic_conditional_program())
            .quantum(stabilizer())
            .sampling(monte_carlo(6))
            .seed(42)
            .run();

        assert_eq!(results.len(), 6);
        for outcome in &results.outcomes {
            assert_eq!(outcome.get_bit(QubitId(0)), Some(true));
            assert_eq!(outcome.get_bit(QubitId(1)), Some(true));
            assert_eq!(outcome.len(), 2);
        }
    }

    #[test]
    fn test_sim_neo_dynamic_command_source_rerun() {
        let mut sim = sim_neo(deterministic_conditional_program())
            .quantum(stabilizer())
            .sampling(monte_carlo(2))
            .seed(42)
            .build();

        let first = sim.run();
        assert_eq!(first.len(), 2);
        for outcome in &first.outcomes {
            assert_eq!(outcome.get_bit(QubitId(0)), Some(true));
            assert_eq!(outcome.get_bit(QubitId(1)), Some(true));
        }

        sim.shots(4);
        let second = sim.run();
        assert_eq!(second.len(), 4);
        for outcome in &second.outcomes {
            assert_eq!(outcome.get_bit(QubitId(0)), Some(true));
            assert_eq!(outcome.get_bit(QubitId(1)), Some(true));
        }
    }

    #[cfg(feature = "qasm")]
    #[test]
    fn test_sim_neo_qasm_conditional_native_stabilizer() {
        let results = sim_neo(deterministic_conditional_qasm())
            .classical(pecos_qasm::qasm_engine())
            .quantum(stabilizer())
            .sampling(monte_carlo(6))
            .seed(42)
            .run();

        assert_eq!(results.len(), 6);
        for outcome in &results.outcomes {
            assert_eq!(outcome.get_bit(QubitId(0)), Some(true));
            assert_eq!(outcome.get_bit(QubitId(1)), Some(true));
            assert_eq!(outcome.len(), 2);
        }
    }
    #[test]
    fn test_sim_neo_dynamic_command_source_quantum_engine_adapter() {
        let results = sim_neo(deterministic_conditional_program())
            .quantum(pecos_engines::stabilizer())
            .sampling(monte_carlo(6))
            .seed(42)
            .run();

        assert_eq!(results.len(), 6);
        for outcome in &results.outcomes {
            assert_eq!(outcome.get_bit(QubitId(0)), Some(true));
            assert_eq!(outcome.get_bit(QubitId(1)), Some(true));
            assert_eq!(outcome.len(), 2);
        }
    }

    #[cfg(feature = "qasm")]
    #[test]
    fn test_sim_neo_qasm_conditional_quantum_engine_adapter() {
        let results = sim_neo(deterministic_conditional_qasm())
            .classical(pecos_qasm::qasm_engine())
            .quantum(pecos_engines::stabilizer())
            .sampling(monte_carlo(6))
            .seed(42)
            .run();

        assert_eq!(results.len(), 6);
        for outcome in &results.outcomes {
            assert_eq!(outcome.get_bit(QubitId(0)), Some(true));
            assert_eq!(outcome.get_bit(QubitId(1)), Some(true));
            assert_eq!(outcome.len(), 2);
        }
    }

    #[cfg(feature = "qasm")]
    #[test]
    fn test_sim_neo_qasm_conditional_native_stabilizer_parallel() {
        let results = sim_neo(deterministic_conditional_qasm())
            .classical(pecos_qasm::qasm_engine())
            .quantum(stabilizer())
            .sampling(monte_carlo(6).workers(2))
            .seed(42)
            .run();

        assert_eq!(results.len(), 6);
        for outcome in &results.outcomes {
            assert_eq!(outcome.get_bit(QubitId(0)), Some(true));
            assert_eq!(outcome.get_bit(QubitId(1)), Some(true));
            assert_eq!(outcome.len(), 2);
        }
    }

    #[cfg(feature = "qasm")]
    #[test]
    fn test_sim_neo_qasm_conditional_quantum_engine_adapter_parallel() {
        let results = sim_neo(deterministic_conditional_qasm())
            .classical(pecos_qasm::qasm_engine())
            .quantum(pecos_engines::stabilizer())
            .sampling(monte_carlo(6).workers(2))
            .seed(42)
            .run();

        assert_eq!(results.len(), 6);
        for outcome in &results.outcomes {
            assert_eq!(outcome.get_bit(QubitId(0)), Some(true));
            assert_eq!(outcome.get_bit(QubitId(1)), Some(true));
            assert_eq!(outcome.len(), 2);
        }
    }

    #[cfg(feature = "qasm")]
    #[test]
    fn test_sim_neo_qasm_auto_conditional_parallel_after_worker_selection() {
        let program = pecos_programs::Qasm::from_string(deterministic_conditional_qasm());
        let results = sim_neo(program)
            .auto()
            .quantum(pecos_engines::stabilizer())
            .sampling(monte_carlo(6).workers(2))
            .seed(42)
            .run();

        assert_eq!(results.len(), 6);
        for outcome in &results.outcomes {
            assert_eq!(outcome.get_bit(QubitId(0)), Some(true));
            assert_eq!(outcome.get_bit(QubitId(1)), Some(true));
            assert_eq!(outcome.len(), 2);
        }
    }
    #[test]
    #[should_panic(
        expected = "Parallel Monte Carlo (workers > 1) requires per-worker construction"
    )]
    fn test_sim_neo_dynamic_command_source_quantum_engine_adapter_rejects_parallel_workers() {
        let _ = sim_neo(deterministic_conditional_program())
            .quantum(pecos_engines::stabilizer())
            .sampling(monte_carlo(2).workers(2))
            .run();
    }

    #[test]
    fn test_sim_neo_quantum_state_vector() {
        // Test state vector backend
        use super::state_vector;

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .quantum(state_vector())
            .sampling(monte_carlo(10))
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

        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        // Test sparse_stab determinism
        let sparse1 = sim_neo(circuit.clone())
            .quantum(sparse_stab())
            .sampling(monte_carlo(20))
            .seed(42)
            .run();

        let sparse2 = sim_neo(circuit.clone())
            .quantum(sparse_stab())
            .sampling(monte_carlo(20))
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
            .sampling(monte_carlo(20))
            .seed(42)
            .run();

        let sv2 = sim_neo(circuit)
            .quantum(state_vector())
            .sampling(monte_carlo(20))
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

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .quantum(state_vector())
            .sampling(monte_carlo(100).workers(4))
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

    // --- Importance Sampling Strategy Tests ---

    #[test]
    #[should_panic(expected = "Importance sampling requires a static circuit")]
    fn test_sim_neo_importance_sampling_rejects_dynamic_command_source() {
        use super::importance_sampling;

        let _ = sim_neo(deterministic_conditional_program())
            .auto()
            .sampling(importance_sampling(1))
            .run();
    }

    #[test]
    fn test_sim_neo_importance_sampling_basic() {
        // Basic test: importance sampling should return results with weights
        use super::importance_sampling;

        let circuit = CommandBuilder::new()
            .pz(&[0])
            .h(&[0]) // Single-qubit gate triggers importance sampling
            .mz(&[0])
            .build();

        let results = sim_neo(circuit)
            .auto()
            .sampling(
                importance_sampling(100)
                    .with_p1(0.01)
                    .with_p2(0.02)
                    .with_p_meas(0.01)
                    .with_boost(5.0),
            )
            .seed(42)
            .run();

        assert_eq!(results.len(), 100);
        assert!(
            results.has_weights(),
            "Importance sampling should produce weights"
        );

        let weights = results.weights.as_ref().unwrap();
        assert_eq!(weights.len(), 100);
    }

    #[test]
    fn test_sim_neo_importance_sampling_uniform() {
        // Test the convenience method for uniform error rates
        use super::importance_sampling;

        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .auto()
            .sampling(
                importance_sampling(50)
                    .with_uniform_error(0.01)
                    .with_boost(10.0),
            )
            .seed(123)
            .run();

        assert_eq!(results.len(), 50);
        assert!(results.has_weights());
    }

    #[test]
    fn test_sim_neo_importance_sampling_produces_unbiased_estimates() {
        // The weighted mean should approximate the true expectation
        // For H then measure: P(0) = P(1) = 0.5 (without errors)
        use super::importance_sampling;

        let circuit = CommandBuilder::new()
            .pz(&[0])
            .h(&[0]) // Creates |+> = (|0> + |1>)/sqrt(2)
            .mz(&[0])
            .build();

        // Run with importance sampling (boosting noise that doesn't affect this test)
        let results = sim_neo(circuit)
            .auto()
            .sampling(
                importance_sampling(2000)
                    .with_uniform_error(0.001)
                    .with_boost(100.0), // Very aggressive boost
            )
            .seed(42)
            .run();

        // Compute weighted mean of outcome
        let weighted_one_rate = results
            .weighted_mean(|outcome| {
                if outcome.get_bit(QubitId(0)).unwrap_or(false) {
                    1.0
                } else {
                    0.0
                }
            })
            .expect("Should have weights");

        // Should be approximately 0.5
        assert!(
            (weighted_one_rate - 0.5).abs() < 0.1,
            "Weighted mean should be ~0.5, got {weighted_one_rate:.4}"
        );
    }

    #[test]
    fn test_sim_neo_importance_sampling_deterministic() {
        // Same seed should produce same results
        use super::importance_sampling;

        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let is_builder = importance_sampling(20)
            .with_uniform_error(0.01)
            .with_boost(10.0);

        let results1 = sim_neo(circuit.clone())
            .auto()
            .sampling(is_builder.clone())
            .seed(42)
            .run();

        let results2 = sim_neo(circuit).auto().sampling(is_builder).seed(42).run();

        assert_eq!(results1.outcomes.len(), results2.outcomes.len());
        for (i, (o1, o2)) in results1
            .outcomes
            .iter()
            .zip(results2.outcomes.iter())
            .enumerate()
        {
            assert_eq!(
                o1.get_bit(QubitId(0)),
                o2.get_bit(QubitId(0)),
                "Shot {i} should be deterministic"
            );
        }

        // Weights should also match
        let w1 = results1.weights.as_ref().unwrap();
        let w2 = results2.weights.as_ref().unwrap();
        for (i, (a, b)) in w1.iter().zip(w2.iter()).enumerate() {
            assert!(
                (a.weight() - b.weight()).abs() < 1e-10,
                "Weight at shot {i} should match"
            );
        }
    }

    #[test]
    fn test_sim_neo_importance_sampling_with_two_qubit_gate() {
        // Test that two-qubit gates also trigger importance sampling
        use super::importance_sampling;

        let circuit = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .h(&[0])
            .cx(&[(0, 1)]) // Two-qubit gate
            .mz(&[0])
            .mz(&[1])
            .build();

        let results = sim_neo(circuit)
            .auto()
            .sampling(
                importance_sampling(100)
                    .with_p1(0.001)
                    .with_p2(0.01)
                    .with_p_meas(0.001)
                    .with_boost(10.0),
            )
            .seed(42)
            .run();

        assert_eq!(results.len(), 100);
        assert!(results.has_weights());
    }

    #[test]
    fn test_sim_neo_importance_sampling_weighted_stats() {
        // Test the weighted_stats helper method
        use super::importance_sampling;

        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .auto()
            .sampling(
                importance_sampling(500)
                    .with_uniform_error(0.01)
                    .with_boost(10.0),
            )
            .seed(42)
            .run();

        // Use weighted_stats to compute mean and variance
        let stats = results.weighted_stats(|outcome| {
            if outcome.get_bit(QubitId(0)).unwrap_or(false) {
                1.0
            } else {
                0.0
            }
        });

        assert!(stats.is_some(), "Should have weights for stats");
        let (mean, _std_error) = stats.unwrap();

        // Mean should be approximately 0.5
        assert!(
            (mean - 0.5).abs() < 0.15,
            "Mean should be ~0.5, got {mean:.4}"
        );
    }

    // --- Custom Backend Tests ---

    #[test]
    fn test_custom_backend_basic() {
        // Use SparseStab via custom_backend and verify correct results
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .x(&[0]) // Flip to |1>
            .mz(&[0])
            .build();

        let results = sim_neo(circuit)
            .quantum(custom_backend(SparseStab::new))
            .sampling(monte_carlo(10))
            .seed(42)
            .build()
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
    fn test_custom_backend_matches_builtin() {
        // Custom backend with SparseStab should produce identical results
        // to the built-in SparseStab backend with the same seed
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .h(&[0]) // Superposition - outcome depends on RNG
            .mz(&[0])
            .build();

        let builtin_results = sim_neo(circuit.clone())
            .quantum(sparse_stab())
            .sampling(monte_carlo(50))
            .seed(42)
            .run();

        let custom_results = sim_neo(circuit)
            .quantum(custom_backend(SparseStab::new))
            .sampling(monte_carlo(50))
            .seed(42)
            .run();

        assert_eq!(
            builtin_results.outcomes.len(),
            custom_results.outcomes.len()
        );
        for (i, (builtin, custom)) in builtin_results
            .outcomes
            .iter()
            .zip(custom_results.outcomes.iter())
            .enumerate()
        {
            assert_eq!(
                builtin.get_bit(QubitId(0)),
                custom.get_bit(QubitId(0)),
                "Custom backend should match built-in at shot {i}"
            );
        }
    }

    #[test]
    fn test_custom_backend_with_noise() {
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .z(&[0]) // Single-qubit gate triggers noise
            .mz(&[0])
            .build();

        let noise = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.5));

        let results = sim_neo(circuit)
            .quantum(custom_backend(SparseStab::new))
            .noise(noise)
            .sampling(monte_carlo(100))
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 100);

        // With 50% depolarizing, should see a mix of outcomes
        let ones: usize = results
            .outcomes
            .iter()
            .filter(|o| o.get_bit(QubitId(0)).unwrap_or(false))
            .count();

        assert!(
            ones > 0 && ones < 100,
            "Expected mix of outcomes with 50% noise, got {ones} ones"
        );
    }

    #[test]
    fn test_custom_backend_deterministic() {
        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let results1 = sim_neo(circuit.clone())
            .quantum(custom_backend(SparseStab::new))
            .sampling(monte_carlo(20))
            .seed(42)
            .run();

        let results2 = sim_neo(circuit)
            .quantum(custom_backend(SparseStab::new))
            .sampling(monte_carlo(20))
            .seed(42)
            .run();

        for (o1, o2) in results1.outcomes.iter().zip(results2.outcomes.iter()) {
            assert_eq!(
                o1.get_bit(QubitId(0)),
                o2.get_bit(QubitId(0)),
                "Custom backend should be deterministic with same seed"
            );
        }
    }

    #[test]
    fn test_custom_backend_parallel_matches_sequential() {
        // The factory builds one runner per worker; per-shot seeding from
        // global shot indices makes results identical for any worker count.
        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
        let run = |workers: usize| {
            sim_neo(circuit.clone())
                .quantum(custom_backend(SparseStab::new))
                .sampling(monte_carlo(50).workers(workers))
                .seed(42)
                .run()
        };

        let sequential = run(1);
        let parallel = run(4);

        assert_eq!(sequential.outcomes.len(), 50);
        assert_eq!(sequential.outcomes.len(), parallel.outcomes.len());
        for (i, (s, p)) in sequential
            .outcomes
            .iter()
            .zip(parallel.outcomes.iter())
            .enumerate()
        {
            assert_eq!(
                s.get_bit(QubitId(0)),
                p.get_bit(QubitId(0)),
                "Shot {i} should match across worker counts"
            );
        }
    }

    #[test]
    fn test_custom_backend_parallel_with_noise() {
        // Noise is cloned per worker; results stay worker-count invariant.
        let circuit = CommandBuilder::new().pz(&[0]).z(&[0]).mz(&[0]).build();
        let run = |workers: usize| {
            sim_neo(circuit.clone())
                .quantum(custom_backend(SparseStab::new))
                .noise(SingleQubitChannel::depolarizing(0.3))
                .sampling(monte_carlo(40).workers(workers))
                .seed(7)
                .run()
        };

        let sequential = run(1);
        let parallel = run(3);

        for (i, (s, p)) in sequential
            .outcomes
            .iter()
            .zip(parallel.outcomes.iter())
            .enumerate()
        {
            assert_eq!(
                s.get_bit(QubitId(0)),
                p.get_bit(QubitId(0)),
                "Noisy shot {i} should match across worker counts"
            );
        }
    }

    #[test]
    fn test_custom_backend_state_vector() {
        // Verify StateVec also works via custom_backend
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .quantum(custom_backend(StateVec::new))
            .sampling(monte_carlo(10))
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

    // --- Register-based Results Tests ---

    #[test]
    fn test_register_counts() {
        // H gate should produce roughly 50/50 split
        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let mut reg = RegisterMap::new();
        reg.add_register("c", &[QubitId(0)]);

        let results = sim_neo(circuit)
            .auto()
            .sampling(monte_carlo(200))
            .seed(42)
            .run();
        let counts = results.register_counts(&reg, "c");

        // Should have entries for both [false] and [true]
        assert!(
            counts.contains_key(&vec![false]),
            "Should have |0> outcomes"
        );
        assert!(counts.contains_key(&vec![true]), "Should have |1> outcomes");

        let total: usize = counts.values().sum();
        assert_eq!(total, 200, "Total counts should equal number of shots");
    }

    #[test]
    fn test_register_counts_bell_state() {
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .h(&[0])
            .cx(&[(0, 1)])
            .mz(&[0])
            .mz(&[1])
            .build();

        let mut reg = RegisterMap::new();
        reg.add_register("c", &[QubitId(0), QubitId(1)]);

        let results = sim_neo(circuit)
            .auto()
            .sampling(monte_carlo(100))
            .seed(42)
            .run();
        let counts = results.register_counts(&reg, "c");

        // Bell state: only |00> and |11> should appear
        for bitstring in counts.keys() {
            assert_eq!(
                bitstring[0], bitstring[1],
                "Bell state qubits must be correlated: got {bitstring:?}"
            );
        }
    }

    #[test]
    fn test_as_register_columns() {
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .x(&[0]) // qubit 0 -> |1>
            .mz(&[0])
            .mz(&[1])
            .build();

        let mut reg = RegisterMap::new();
        reg.add_register("a", &[QubitId(0)]);
        reg.add_register("b", &[QubitId(1)]);

        let results = sim_neo(circuit)
            .auto()
            .sampling(monte_carlo(5))
            .seed(42)
            .run();
        let columns = results.as_register_columns(&reg);

        assert_eq!(columns.len(), 2);
        assert_eq!(columns["a"].len(), 5);
        assert_eq!(columns["b"].len(), 5);

        // qubit 0 is always |1>, qubit 1 is always |0>
        for shot in &columns["a"] {
            assert_eq!(shot, &[true]);
        }
        for shot in &columns["b"] {
            assert_eq!(shot, &[false]);
        }
    }

    #[test]
    fn test_register_counts_missing_register() {
        let circuit = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

        let mut reg = RegisterMap::new();
        reg.add_register("missing", &[QubitId(5)]); // never measured

        let results = sim_neo(circuit)
            .auto()
            .sampling(monte_carlo(10))
            .seed(42)
            .run();
        let counts = results.register_counts(&reg, "missing");

        assert!(
            counts.is_empty(),
            "Unmeasured register should have no counts"
        );
    }

    #[test]
    fn test_sim_neo_with_gate_definitions() {
        use crate::extensible::GateDefinitions;

        let circuit = CommandBuilder::new()
            .pz(&[0])
            .x(&[0]) // Flip to |1>
            .mz(&[0])
            .build();

        let defs = GateDefinitions::new();

        let results = sim_neo(circuit)
            .auto()
            .gate_definitions(defs)
            .sampling(monte_carlo(10))
            .seed(42)
            .build()
            .run();

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
    fn test_sim_neo_gate_definitions_with_statevec() {
        use crate::extensible::GateDefinitions;

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let defs = GateDefinitions::new();

        let results = sim_neo(circuit)
            .quantum(state_vector())
            .gate_definitions(defs)
            .sampling(monte_carlo(10))
            .seed(42)
            .build()
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
    fn test_sim_neo_statevec_t_gate() {
        // T gate is non-Clifford -- needs rotation support.
        // Verify T gate runs without error on StateVec.
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .t(&[0]) // Non-Clifford gate
            .mz(&[0])
            .build();

        // This would fail with ProgramRunner::new() (Clifford-only)
        let results = sim_neo(circuit)
            .quantum(state_vector())
            .sampling(monte_carlo(10))
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 10);

        // T|0> = |0> (up to phase), so measurement should always be 0
        for outcome in &results.outcomes {
            assert!(
                !outcome.get_bit(QubitId(0)).unwrap(),
                "T|0> should measure as |0>"
            );
        }
    }

    #[test]
    fn test_sim_neo_statevec_rz_gate() {
        use pecos_core::Angle64;
        use std::f64::consts::PI;

        // RZ(pi) on |+> should flip phase, so H then RZ(pi) then H = X (up to global phase)
        // Instead, simpler test: RZ on |0> leaves |0>
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .rz(&[0], Angle64::from_radians(PI / 4.0)) // arbitrary rotation
            .mz(&[0])
            .build();

        let results = sim_neo(circuit)
            .quantum(state_vector())
            .sampling(monte_carlo(10))
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 10);

        // RZ on |0> gives e^{-i*pi/8}|0> -- still |0> when measured
        for outcome in &results.outcomes {
            assert!(
                !outcome.get_bit(QubitId(0)).unwrap(),
                "RZ|0> should measure as |0>"
            );
        }
    }

    #[test]
    fn test_sim_neo_statevec_parallel() {
        // Verify StateVec rotation support works with parallel workers too
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .t(&[0]) // Non-Clifford
            .mz(&[0])
            .build();

        let results = sim_neo(circuit)
            .quantum(state_vector())
            .sampling(monte_carlo(10).workers(2))
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 10);

        for outcome in &results.outcomes {
            assert!(
                !outcome.get_bit(QubitId(0)).unwrap(),
                "T|0> should measure as |0>"
            );
        }
    }

    #[test]
    fn test_sim_neo_max_decomp_depth() {
        // Verify that max_decomp_depth builder method works without error
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .auto()
            .max_decomp_depth(20)
            .sampling(monte_carlo(10))
            .seed(42)
            .build()
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
    fn test_sim_neo_max_decomp_depth_parallel() {
        // Verify max_decomp_depth works with parallel workers
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .auto()
            .max_decomp_depth(20)
            .sampling(monte_carlo(10).workers(2))
            .seed(42)
            .build()
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
    fn test_sim_neo_custom_backend_with_rotations() {
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .t(&[0]) // Non-Clifford
            .mz(&[0])
            .build();

        let results = sim_neo(circuit)
            .quantum(custom_backend_with_rotations(StateVec::new))
            .sampling(monte_carlo(10))
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 10);

        for outcome in &results.outcomes {
            assert!(
                !outcome.get_bit(QubitId(0)).unwrap(),
                "T|0> should measure as |0>"
            );
        }
    }

    #[test]
    fn test_sim_neo_gate_overrides() {
        use crate::extensible::gates;
        use crate::runner::GateOverrides;

        // Override X gate to be identity (do nothing) -- measurement should stay 0
        let overrides =
            GateOverrides::<SparseStab>::new().register(gates::X, |_sim, _angles, _qubits| true);

        let circuit = CommandBuilder::new()
            .pz(&[0])
            .x(&[0]) // Would flip to |1>, but override makes it a no-op
            .mz(&[0])
            .build();

        let results = sim_neo(circuit)
            .auto()
            .gate_overrides(overrides)
            .sampling(monte_carlo(10))
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 10);

        // X was overridden to be identity, so all measurements should be 0
        for outcome in &results.outcomes {
            assert!(
                !outcome.get_bit(QubitId(0)).unwrap(),
                "Overridden X should be identity, measuring |0>"
            );
        }
    }

    #[test]
    fn test_sim_neo_gate_overrides_parallel() {
        use crate::extensible::gates;
        use crate::runner::GateOverrides;

        // Override X to be identity -- verify it works with parallel workers
        let overrides =
            GateOverrides::<SparseStab>::new().register(gates::X, |_sim, _angles, _qubits| true);

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .auto()
            .gate_overrides(overrides)
            .sampling(monte_carlo(10).workers(2))
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 10);

        for outcome in &results.outcomes {
            assert!(
                !outcome.get_bit(QubitId(0)).unwrap(),
                "Overridden X should be identity, measuring |0>"
            );
        }
    }

    #[test]
    fn test_sim_neo_gate_overrides_statevec() {
        use crate::extensible::gates;
        use crate::runner::GateOverrides;

        // Override X to be identity on StateVec backend
        let overrides =
            GateOverrides::<StateVec>::new().register(gates::X, |_sim, _angles, _qubits| true);

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .quantum(state_vector())
            .gate_overrides(overrides)
            .sampling(monte_carlo(10))
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 10);

        for outcome in &results.outcomes {
            assert!(
                !outcome.get_bit(QubitId(0)).unwrap(),
                "Overridden X should be identity, measuring |0>"
            );
        }
    }

    #[test]
    #[should_panic(expected = "StateVec gate overrides used with SparseStab backend")]
    fn test_sim_neo_gate_overrides_backend_mismatch_statevec_on_sparsestab() {
        use crate::extensible::gates;
        use crate::runner::GateOverrides;

        let overrides =
            GateOverrides::<StateVec>::new().register(gates::X, |_sim, _angles, _qubits| true);

        // .auto() selects SparseStab -- StateVec overrides should panic
        sim_neo(CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build())
            .auto()
            .gate_overrides(overrides)
            .sampling(monte_carlo(1))
            .seed(42)
            .build()
            .run();
    }

    #[test]
    #[should_panic(expected = "SparseStab gate overrides used with StateVec backend")]
    fn test_sim_neo_gate_overrides_backend_mismatch_sparsestab_on_statevec() {
        use crate::extensible::gates;
        use crate::runner::GateOverrides;

        let overrides =
            GateOverrides::<SparseStab>::new().register(gates::X, |_sim, _angles, _qubits| true);

        sim_neo(CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build())
            .quantum(state_vector())
            .gate_overrides(overrides)
            .sampling(monte_carlo(1))
            .seed(42)
            .build()
            .run();
    }

    #[test]
    fn test_sim_neo_gate_overrides_observable_effect() {
        use crate::extensible::gates;
        use crate::runner::GateOverrides;
        use pecos_simulators::CliffordGateable;

        // Override X to apply Z instead. Z|0> = |0>, so measurement stays 0
        // (without override, X|0> = |1>)
        let overrides =
            GateOverrides::<SparseStab>::new().register(gates::X, |sim, _angles, qubits| {
                sim.z(qubits);
                true
            });

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .auto()
            .gate_overrides(overrides)
            .sampling(monte_carlo(10))
            .seed(42)
            .build()
            .run();

        // Z|0> = |0>, so all outcomes should be 0
        for outcome in &results.outcomes {
            assert!(
                !outcome.get_bit(QubitId(0)).unwrap(),
                "X overridden to Z: Z|0> should measure |0>"
            );
        }
    }

    #[test]
    fn test_sim_neo_gate_definitions_parallel() {
        use crate::extensible::GateDefinitions;

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let defs = GateDefinitions::new();

        let results = sim_neo(circuit)
            .auto()
            .gate_definitions(defs)
            .sampling(monte_carlo(10).workers(2))
            .seed(42)
            .build()
            .run();

        assert_eq!(results.len(), 10);

        for outcome in &results.outcomes {
            assert!(
                outcome.get_bit(QubitId(0)).unwrap(),
                "X gate should produce |1>"
            );
        }
    }
}
