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
//! use pecos_neo::tool::sim_neo;
//! use pecos_neo::prelude::*;
//!
//! let circuit = CommandBuilder::new()
//!     .pz(&[0]).h(&[0]).mz(&[0])
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
//! ```no_run
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
//! // Pass QASM source, then set the engine
//! let results = sim_neo(qasm)
//!     .classical(qasm_engine())
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
//! ```text
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
//! ```no_run
//! use pecos_neo::tool::sim_neo;
//! use pecos_neo::prelude::*;
//!
//! let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
//! let mut sim = sim_neo(circuit)
//!     .shots(1000)
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
use pecos_core::rng::RngManageable;
use pecos_core::rng::rng_manageable::derive_seed;
use pecos_random::PecosRng;
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, SparseStab, Stabilizer, StateVec,
};
use rayon::prelude::*;
use std::collections::BTreeMap;

use super::resource::Resources;
use super::{Plugin, Stage, Tool};

// --- Quantum Backend Builders (builder-of-builders pattern) ---

/// Configuration for a quantum backend, stored as data in the builder.
///
/// This enum represents the choice of quantum simulator. The actual simulator
/// is constructed at build time, following the builder-of-builders pattern.
#[derive(Default)]
pub enum QuantumBackend {
    /// Sparse stabilizer simulator (default).
    ///
    /// Efficient for Clifford circuits and QEC simulations.
    /// Only supports Clifford gates (H, S, CNOT, CZ, etc.).
    #[default]
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
    /// to be used through the `sim_neo()` API. Use [`custom_backend()`] to create.
    ///
    /// Custom backends only support sequential execution. Using `.workers()`,
    /// `.auto_workers()`, or importance sampling will panic at `.run()` time.
    Custom(Box<dyn SimulatorFactory>),
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
/// The sparse stabilizer is the default backend, efficient for Clifford circuits
/// and quantum error correction simulations.
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{sim_neo, sparse_stab};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
/// let results = sim_neo(circuit)
///     .quantum(sparse_stab())
///     .shots(1000)
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
/// use pecos_neo::tool::{sim_neo, stabilizer};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
/// let results = sim_neo(circuit)
///     .quantum(stabilizer())
///     .shots(1000)
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
/// use pecos_neo::tool::{sim_neo, state_vector};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
/// let results = sim_neo(circuit)
///     .quantum(state_vector())
///     .shots(1000)
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
    factory: Box<dyn SimulatorFactory>,
}

impl From<CustomBackendBuilder> for QuantumBackend {
    fn from(builder: CustomBackendBuilder) -> Self {
        QuantumBackend::Custom(builder.factory)
    }
}

/// Create a custom backend from a factory closure.
///
/// This allows any simulator implementing `CliffordGateable + RngManageable<Rng = PecosRng>`
/// to be used through `sim_neo()`. The closure receives the number of qubits
/// and should return a new simulator instance.
///
/// **Note:** Custom backends only support sequential execution. Using `.workers()`,
/// `.auto_workers()`, or importance sampling with a custom backend will panic
/// at `.run()` time.
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{sim_neo, custom_backend};
/// use pecos_neo::prelude::*;
/// use pecos_simulators::SparseStab;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
///
/// // Use a custom simulator backend
/// let results = sim_neo(circuit)
///     .quantum(custom_backend(|n| SparseStab::new(n)))
///     .shots(100)
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
        factory: Box::new(factory),
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
        factory: Box::new(factory),
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
/// use pecos_neo::tool::{sim_neo, custom_backend_with_rotations};
/// use pecos_neo::prelude::*;
/// use pecos_simulators::StateVec;
///
/// let circuit = CommandBuilder::new().pz(&[0]).t(&[0]).mz(&[0]).build();
///
/// let results = sim_neo(circuit)
///     .quantum(custom_backend_with_rotations(|n| StateVec::new(n)))
///     .shots(100)
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
        factory: Box::new(RotationSimulatorFactory(factory)),
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
/// use pecos_neo::tool::sim_neo;
/// use pecos_qasm::qasm_engine;
///
/// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];";
/// sim_neo(qasm_code)
///     .classical(qasm_engine())
///     .shots(1000)
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
/// use pecos_neo::tool::sim_neo;
/// use pecos_programs::Qasm;
/// use pecos_qasm::qasm_engine;
///
/// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];".to_string();
///
/// // Auto mode - uses qasm_engine() automatically
/// sim_neo(Qasm::from_string(qasm_code.clone()))
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
/// use pecos_neo::tool::sim_neo;
/// use pecos_programs::Hugr;
///
/// let hugr = Hugr::from_file("program.hugr").unwrap();
/// sim_neo(hugr)
///     .auto()
///     .shots(1000)
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
/// use pecos_neo::tool::sim_neo;
/// use pecos_programs::{Program, Qasm};
///
/// let qasm = Qasm::from_string("OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];".to_string());
/// sim_neo(Program::Qasm(qasm))
///     .auto()
///     .shots(1000)
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
/// Specifies the true error rates and boost factor for biased sampling.
/// Use the [`importance_sampling()`] function to create an instance.
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{sim_neo, importance_sampling};
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
/// let results = sim_neo(circuit)
///     .sampling(importance_sampling()
///         .with_p1(0.001)
///         .with_p2(0.01)
///         .with_p_meas(0.001)
///         .with_boost(10.0))
///     .shots(10000)
///     .build()
///     .run();
/// ```
#[derive(Debug, Clone)]
pub struct ImportanceSamplingBuilder {
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
    /// Create a new importance sampling builder with default values.
    ///
    /// Default: p1=0.001, p2=0.01, `p_meas=0.001`, boost=10.0
    #[must_use]
    pub fn new() -> Self {
        Self {
            p1: 0.001,
            p2: 0.01,
            p_meas: 0.001,
            boost: 10.0,
        }
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
}

impl Default for ImportanceSamplingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl From<ImportanceSamplingBuilder> for Sampling {
    fn from(builder: ImportanceSamplingBuilder) -> Self {
        builder.build()
    }
}

/// Create an importance sampling strategy builder.
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
/// let results = sim_neo(circuit)
///     .sampling(importance_sampling()
///         .with_p1(0.001)
///         .with_p2(0.01)
///         .with_boost(10.0))
///     .shots(10000)
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
pub fn importance_sampling() -> ImportanceSamplingBuilder {
    ImportanceSamplingBuilder::new()
}

/// Sampling strategy for simulation execution.
///
/// This enum defines how shots are executed. Different strategies offer
/// trade-offs between simplicity, parallelism, and specialized sampling.
///
/// Stored as data in the builder, the actual execution is set up at run time.
///
/// The default is `MonteCarlo { workers: 1 }`, which runs shots sequentially
/// using the Tool/Schedule/Plugin system. Use `.workers(n)` or `.auto_workers()`
/// for parallel execution.
#[derive(Debug, Clone)]
pub enum Sampling {
    /// Monte Carlo execution (sequential with 1 worker, parallel with >1).
    ///
    /// Each worker runs a batch of shots independently with deterministic seeding.
    /// Supports both noiseless and noisy circuits (noise model is cloned per worker).
    /// With 1 worker, runs via the Tool's schedule directly.
    MonteCarlo {
        /// Number of parallel workers (default: 1).
        workers: usize,
    },

    /// Importance sampling for rare event estimation.
    ///
    /// Biases sampling toward rare events and reweights results.
    /// Use when estimating probabilities of rare outcomes.
    ///
    /// Use the [`importance_sampling()`] builder function to create this variant.
    ImportanceSampling {
        /// Configuration for importance sampling.
        config: ImportanceSamplingBuilder,
    },
}

impl Default for Sampling {
    fn default() -> Self {
        Self::MonteCarlo { workers: 1 }
    }
}

impl Sampling {
    /// Create a Monte Carlo sampling strategy with specified workers.
    #[must_use]
    pub fn monte_carlo(workers: usize) -> Self {
        Self::MonteCarlo { workers }
    }

    /// Create a Monte Carlo sampling strategy with auto-detected worker count.
    #[must_use]
    pub fn monte_carlo_auto() -> Self {
        let workers = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        Self::MonteCarlo { workers }
    }
}

/// Accumulated simulation results.
#[derive(Debug, Clone, Default)]
pub struct SimulationResults {
    /// Per-shot measurement outcomes.
    pub outcomes: Vec<MeasurementOutcomes>,
    /// Per-shot importance weights (only for importance sampling).
    pub weights: Option<Vec<crate::sampling::weight::SampleWeight>>,
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
    /// use pecos_neo::tool::sim_neo;
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
    /// let results = sim_neo(circuit).shots(100).seed(42).run();
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
    /// use pecos_neo::tool::sim_neo;
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
    /// let results = sim_neo(circuit).shots(1000).seed(42).run();
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
/// use pecos_neo::tool::sim_neo;
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
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
/// ```no_run
/// use pecos_neo::tool::sim_neo;
/// use pecos_qasm::qasm_engine;
///
/// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];";
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
/// ```no_run
/// use pecos_neo::tool::sim_neo_builder;
/// use pecos_qasm::qasm_engine;
///
/// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];";
/// // Or pass already-configured engine builder
/// let results = sim_neo_builder()
///     .with_engine(qasm_engine().qasm(qasm_code))
///     .shots(1000)
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
    /// Sampling strategy (data).
    sampling: Sampling,
    /// Quantum backend configuration (data).
    quantum_backend: QuantumBackend,
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
    /// Create a new simulation builder for a circuit.
    #[must_use]
    pub fn with_circuit(circuit: CommandQueue) -> Self {
        Self {
            source: Some(ProgramSource::Static(circuit)),
            pending_builder: None,
            noise: None,
            definitions: None,
            config: SimConfig::default(),
            sampling: Sampling::default(),
            quantum_backend: QuantumBackend::default(),
            explicit_num_qubits: None,
            max_decomp_depth: None,
            overrides: None,
            event_handlers: None,
        }
    }

    /// Create a simulation builder for a dynamic command source.
    #[must_use]
    pub fn with_command_source(source: Box<dyn CommandSource + Send + Sync>) -> Self {
        Self {
            source: Some(ProgramSource::Dynamic(source)),
            pending_builder: None,
            noise: None,
            definitions: None,
            config: SimConfig::default(),
            sampling: Sampling::default(),
            quantum_backend: QuantumBackend::default(),
            explicit_num_qubits: None,
            max_decomp_depth: None,
            overrides: None,
            event_handlers: None,
        }
    }

    /// Create a simulation builder with raw program source.
    ///
    /// Use `.classical(builder)` to specify how to interpret the source.
    #[must_use]
    pub fn with_program_source(source: String) -> Self {
        Self {
            source: Some(ProgramSource::RawSource(source)),
            pending_builder: None,
            noise: None,
            definitions: None,
            config: SimConfig::default(),
            sampling: Sampling::default(),
            quantum_backend: QuantumBackend::default(),
            explicit_num_qubits: None,
            max_decomp_depth: None,
            overrides: None,
            event_handlers: None,
        }
    }

    /// Create a simulation builder with a typed program.
    ///
    /// Use `.auto()` to automatically select the engine, or
    /// `.classical(builder)` for explicit control.
    #[must_use]
    pub fn with_typed_program(program: TypedProgram) -> Self {
        Self {
            source: Some(ProgramSource::Typed(program)),
            pending_builder: None,
            noise: None,
            definitions: None,
            config: SimConfig::default(),
            sampling: Sampling::default(),
            quantum_backend: QuantumBackend::default(),
            explicit_num_qubits: None,
            max_decomp_depth: None,
            overrides: None,
            event_handlers: None,
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
            pending_builder: None,
            noise: None,
            definitions: None,
            config: SimConfig::default(),
            sampling: Sampling::default(),
            quantum_backend: QuantumBackend::default(),
            explicit_num_qubits: None,
            max_decomp_depth: None,
            overrides: None,
            event_handlers: None,
        }
    }

    /// Set the classical control engine builder (builder-of-builders pattern).
    ///
    /// The builder is stored as data and configured with source at `.build()` time.
    /// This follows "everything is data" - we collect configuration, then wire
    /// it all together when building the Tool.
    ///
    /// ```no_run
    /// use pecos_neo::tool::sim_neo;
    /// use pecos_qasm::qasm_engine;
    ///
    /// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];";
    /// // Builder is stored as data, source injected at build time
    /// let results = sim_neo(qasm_code)
    ///     .classical(qasm_engine())  // stores builder as data
    ///     .shots(1000)
    ///     .build()  // configures builder, builds engine, creates Tool
    ///     .run();
    /// ```
    ///
    /// For pre-configured engine builders, use `.with_engine()` instead:
    ///
    /// ```no_run
    /// use pecos_neo::tool::sim_neo_builder;
    /// use pecos_qasm::qasm_engine;
    ///
    /// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];";
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
    /// use pecos_neo::tool::sim_neo_builder;
    /// use pecos_qasm::qasm_engine;
    ///
    /// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];";
    /// let results = sim_neo_builder()
    ///     .with_engine(qasm_engine().qasm(qasm_code))
    ///     .shots(1000)
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

    /// Automatically select the appropriate engine based on program type.
    ///
    /// This is a convenience method that selects good defaults:
    /// - `Qasm` programs use `qasm_engine()`
    /// - Future: `Hugr`, `PhirJson`, `Qis` will use their respective engines
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::tool::sim_neo;
    /// use pecos_programs::Qasm;
    ///
    /// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];".to_string();
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
    /// Note: `.auto()` also selects the default Monte Carlo sampling strategy. The
    /// parallel execution plan decides whether the selected command source and
    /// quantum backend can safely build independent worker state.
    #[must_use]
    pub fn auto(mut self) -> Self {
        match self.source.take() {
            Some(ProgramSource::Typed(typed)) => match typed {
                #[cfg(feature = "qasm")]
                TypedProgram::Qasm(qasm) => {
                    // Auto-select qasm_engine() and configure with the program.
                    let builder = pecos_qasm::qasm_engine().qasm(qasm.source);
                    self.source = Some(ProgramSource::Classical(Box::new(EngineBuilderWrapper {
                        builder,
                    })));
                    self.sampling = Sampling::monte_carlo_auto();
                    self
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
                    self.source = Some(ProgramSource::Classical(Box::new(EngineBuilderWrapper {
                        builder,
                    })));
                    self.sampling = Sampling::monte_carlo_auto();
                    self
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
            Some(ProgramSource::Dynamic(_)) => {
                panic!(
                    "Cannot use .auto() with an existing dynamic command source. \
                     Command sources are already executable."
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

    /// Set the sampling strategy for simulation execution.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::tool::{sim_neo, Sampling};
    /// use pecos_neo::prelude::*;
    ///
    /// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
    ///
    /// // Parallel Monte Carlo with 4 workers
    /// let results = sim_neo(circuit.clone())
    ///     .sampling(Sampling::monte_carlo(4))
    ///     .shots(1000)
    ///     .build()
    ///     .run();
    ///
    /// // Auto-detect worker count
    /// let results = sim_neo(circuit)
    ///     .sampling(Sampling::monte_carlo_auto())
    ///     .shots(1000)
    ///     .build()
    ///     .run();
    /// ```
    #[must_use]
    pub fn sampling(mut self, sampling: impl Into<Sampling>) -> Self {
        self.sampling = sampling.into();
        self
    }

    /// Convenience method for parallel Monte Carlo with specified workers.
    ///
    /// Parallel execution distributes shots across workers using rayon,
    /// with each worker getting its own simulator and noise model clone.
    /// Works with both noiseless and noisy circuits.
    ///
    /// # Panics
    ///
    /// Panics at `.run()` time if parallel execution is not possible.
    /// Parallel execution requires a static circuit using a built-in
    /// backend (`SparseStab` or `StateVec`). Classical engines and custom
    /// backends are not supported.
    #[must_use]
    pub fn workers(mut self, workers: usize) -> Self {
        self.sampling = Sampling::monte_carlo(workers);
        self
    }

    /// Convenience method for parallel Monte Carlo with auto-detected workers.
    ///
    /// See [`workers()`](Self::workers) for requirements and panics.
    #[must_use]
    pub fn auto_workers(mut self) -> Self {
        self.sampling = Sampling::monte_carlo_auto();
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
    /// ```no_run
    /// use pecos_neo::tool::{sim_neo, sparse_stab, state_vector};
    /// use pecos_neo::prelude::*;
    ///
    /// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
    ///
    /// // Use sparse stabilizer (default, Clifford-only)
    /// let results = sim_neo(circuit.clone())
    ///     .quantum(sparse_stab())
    ///     .shots(1000)
    ///     .build()
    ///     .run();
    ///
    /// // Use state vector (supports T gates, rotations)
    /// let results = sim_neo(circuit)
    ///     .quantum(state_vector())
    ///     .shots(1000)
    ///     .build()
    ///     .run();
    /// ```
    #[must_use]
    pub fn quantum<B: Into<QuantumBackend>>(mut self, backend: B) -> Self {
        self.quantum_backend = backend.into();
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
    /// use pecos_neo::tool::sim_neo;
    /// use pecos_neo::prelude::*;
    /// use pecos_neo::noise::GeneralNoiseModelBuilder;
    ///
    /// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
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

    /// Set custom gate definitions (decompositions, user-defined gates).
    ///
    /// Gate definitions control how gate identifiers are mapped to simulator
    /// operations. Use this to add custom gates or override built-in gate
    /// decompositions.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::tool::sim_neo;
    /// use pecos_neo::prelude::*;
    ///
    /// let defs = GateDefinitions::new(); // core gates included by default
    ///
    /// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
    /// let results = sim_neo(circuit)
    ///     .gate_definitions(defs)
    ///     .shots(100)
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
    /// use pecos_neo::tool::sim_neo;
    /// use pecos_neo::prelude::*;
    ///
    /// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
    /// let results = sim_neo(circuit)
    ///     .max_decomp_depth(20)
    ///     .shots(100)
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
    /// use pecos_neo::tool::sim_neo;
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
    /// let results = sim_neo(circuit)
    ///     .gate_overrides(overrides)
    ///     .shots(100)
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
    /// use pecos_neo::tool::sim_neo;
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
    /// let results = sim_neo(circuit)
    ///     .event_handlers(handlers)
    ///     .shots(100)
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
    /// - Noise model is built
    /// - Tool is constructed with all plugins and systems
    ///
    /// # Panics
    ///
    /// Panics if no program source is set (neither circuit nor classical engine).
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

        let parallel_plan = match self.sampling {
            Sampling::MonteCarlo { workers } if workers > 1 => build_parallel_execution_plan(
                &source,
                &self.quantum_backend,
                self.explicit_num_qubits,
                self.noise.clone(),
                self.definitions.clone(),
                self.max_decomp_depth,
                self.overrides.clone(),
                self.event_handlers.clone(),
            ),
            _ => None,
        };

        let mut tool = Tool::new()
            .insert_resource(ProgramSourceResource(source))
            .insert_resource(self.config)
            .insert_resource(QuantumBackendResource(self.quantum_backend));

        match &self.sampling {
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
            sampling: self.sampling,
            parallel_plan,
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
    /// use pecos_neo::tool::sim_neo;
    /// use pecos_qasm::qasm_engine;
    ///
    /// let qasm_code = "OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0];";
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

fn reject_dynamic_runner_config(
    backend_name: &str,
    definitions: Option<&GateDefinitionsResource>,
    max_depth: Option<&MaxDecompDepthResource>,
    overrides: Option<&GateOverridesResource>,
    event_handlers: Option<&EventHandlersResource>,
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
                definitions.as_ref(),
                max_depth.as_ref(),
                overrides.as_ref(),
                event_handlers.as_ref(),
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
                definitions.as_ref(),
                max_depth.as_ref(),
                overrides.as_ref(),
                event_handlers.as_ref(),
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

    let sim = SparseStab::new(num_qubits);
    let runner = ImportanceSamplingRunner::new(sim)
        .with_single_qubit_boost(is_config.p1(), is_config.boost())
        .with_two_qubit_boost(is_config.p2(), is_config.boost())
        .with_measurement_boost(is_config.p_meas(), is_config.boost());

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
fn is_sim_pre_shot(resources: &mut Resources) {
    let config = resources.get::<SimConfig>().clone();
    let state = resources.get_mut::<ISShotState>();

    let base_seed = config.seed.unwrap_or(0);
    let shot_seed = derive_seed(base_seed, &format!("shot_{}", state.shot_index));
    let sim_seed = derive_seed(shot_seed, "simulator");
    let noise_seed = derive_seed(shot_seed, "noise");

    state.runner.rng = PecosRng::seed_from_u64(noise_seed);
    state.runner.simulator.set_seed(sim_seed);
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
/// use pecos_neo::tool::sim_neo;
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
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
    /// Sampling strategy (stored as data).
    sampling: Sampling,
    /// Data-oriented plan for parallel execution (if applicable).
    parallel_plan: Option<ParallelExecutionPlan>,
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
        QuantumBackend::Custom(_) => return None,
    };

    Some(ParallelExecutionPlan {
        command_source_factory: source_factory,
        quantum_runner_factory: runner_factory,
    })
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
    /// Execution strategy depends on the sampling strategy:
    /// - `MonteCarlo { workers: 1 }`: Runs shots via the Tool (default)
    /// - `MonteCarlo { workers: n }`: Parallelizes shots across n workers
    /// - `ImportanceSampling`: Runs via the Tool with `ImportanceSamplingSimPlugin`
    ///
    /// # Panics
    /// Panics if parallel Monte Carlo is used without per-worker runner construction support.
    pub fn run(&mut self) -> SimulationResults {
        let config = self.tool.resource::<SimConfig>().clone();

        // Dispatch based on sampling strategy
        match &self.sampling {
            Sampling::MonteCarlo { workers } if *workers > 1 => {
                let plan = self.parallel_plan.as_ref().unwrap_or_else(|| {
                    panic!(
                        "Parallel Monte Carlo requires per-worker runner construction support. \
                         Dynamic programs need a command-source factory, and custom backends \
                         need a quantum-runner factory. Remove .workers() / .auto_workers() \
                         for single-worker execution, or use a backend/source path with \
                         explicit per-worker construction."
                    )
                });
                self.run_parallel(&config, plan, *workers)
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

        // Flatten in deterministic order
        let outcomes = all_results.into_iter().flat_map(|r| r.outcomes).collect();

        SimulationResults {
            outcomes,
            weights: None,
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
/// use pecos_neo::tool::sim_neo;
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new()
///     .pz(&[0]).h(&[0]).mz(&[0])
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
/// ```no_run
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
/// let results = sim_neo(qasm)
///     .classical(qasm_engine())
///     .depolarizing(0.01)
///     .shots(1000)
///     .seed(42)
///     .build()
///     .run();
/// ```
///
/// ## Reusable Simulation
///
/// ```no_run
/// use pecos_neo::tool::sim_neo;
/// use pecos_neo::prelude::*;
///
/// let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
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
/// ```no_run
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
///     .with_engine(qasm_engine().qasm(qasm))
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
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

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
            .pz(&[0])
            .h(&[0]) // Superposition - outcome depends on RNG
            .mz(&[0])
            .build();

        // Same seed should produce same results
        let results1 = sim_neo(circuit.clone()).shots(20).seed(42).build().run();

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
            .pz(&[0])
            .z(&[0]) // Single-qubit gate to trigger noise
            .mz(&[0])
            .build();

        // Very high error rate - this will definitely flip some outcomes
        let noise = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.5));

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
            .pz(&[0])
            .z(&[0]) // Single-qubit gate to trigger noise
            .mz(&[0])
            .build();

        let noise1 = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.5));
        let noise2 = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.5));

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
        let circuit = CommandBuilder::new().pz(&[0]).z(&[0]).mz(&[0]).build();

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

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

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

        let circuit = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

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
    fn test_sim_neo_monte_carlo_sampling() {
        // Test Monte Carlo sampling with multiple workers
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .x(&[0]) // Flip to |1>
            .mz(&[0])
            .build();

        // Use .workers() convenience method for Monte Carlo
        let results = sim_neo(circuit).workers(4).shots(100).seed(42).run();

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

        let results1 = sim_neo(circuit.clone()).workers(4).shots(50).seed(42).run();

        let results2 = sim_neo(circuit).workers(4).shots(50).seed(42).run();

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
        // Test explicit sampling configuration
        use super::Sampling;

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        // Use explicit Sampling enum
        let results = sim_neo(circuit)
            .sampling(Sampling::monte_carlo(2))
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
    fn test_sim_neo_single_worker_matches_parallel() {
        // Critical test: 1 worker and multiple workers should produce identical
        // results with the same seed (they use the same per-shot seeding scheme)
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .h(&[0]) // Superposition - outcome depends on RNG
            .mz(&[0])
            .build();

        // Run with default (1 worker)
        let single_results = sim_neo(circuit.clone()).shots(50).seed(42).run();

        // Run with parallel Monte Carlo sampling (4 workers)
        let parallel_results = sim_neo(circuit).workers(4).shots(50).seed(42).run();

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
            .noise(noise_single)
            .shots(50)
            .seed(42)
            .run();

        // Run with parallel Monte Carlo sampling
        let parallel_results = sim_neo(circuit)
            .noise(noise_par)
            .workers(4)
            .shots(50)
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
            .noise(noise1)
            .workers(4)
            .shots(50)
            .seed(42)
            .run();

        let results2 = sim_neo(circuit)
            .noise(noise2)
            .workers(4)
            .shots(50)
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
    fn test_sim_neo_quantum_stabilizer() {
        // Test explicitly selecting the stable public stabilizer backend.
        use super::stabilizer;

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .quantum(stabilizer())
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
            .shots(12)
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
            .shots(8)
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
            .shots(1)
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
            .workers(2)
            .shots(2)
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
            .shots(1)
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
            .shots(1)
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
            .shots(1)
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
            .shots(1)
            .run();
    }
    #[test]
    fn test_sim_neo_quantum_engine_builder_parallel_static_circuit() {
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .quantum(pecos_engines::stabilizer())
            .workers(2)
            .shots(6)
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
            .workers(2)
            .shots(6)
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
            .shots(6)
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
            .shots(2)
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
            .shots(6)
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
            .shots(6)
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
            .shots(6)
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
            .workers(2)
            .shots(6)
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
            .workers(2)
            .shots(6)
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
            .workers(2)
            .quantum(pecos_engines::stabilizer())
            .shots(6)
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
        expected = "Parallel Monte Carlo requires per-worker runner construction support"
    )]
    fn test_sim_neo_dynamic_command_source_quantum_engine_adapter_rejects_parallel_workers() {
        let _ = sim_neo(deterministic_conditional_program())
            .quantum(pecos_engines::stabilizer())
            .workers(2)
            .shots(2)
            .run();
    }

    #[test]
    fn test_sim_neo_quantum_state_vector() {
        // Test state vector backend
        use super::state_vector;

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

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

        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

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

        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

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

    // --- Importance Sampling Strategy Tests ---

    #[test]
    #[should_panic(expected = "Importance sampling requires a static circuit")]
    fn test_sim_neo_importance_sampling_rejects_dynamic_command_source() {
        use super::importance_sampling;

        let _ = sim_neo(deterministic_conditional_program())
            .sampling(importance_sampling())
            .shots(1)
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
            .sampling(
                importance_sampling()
                    .with_p1(0.01)
                    .with_p2(0.02)
                    .with_p_meas(0.01)
                    .with_boost(5.0),
            )
            .shots(100)
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
            .sampling(
                importance_sampling()
                    .with_uniform_error(0.01)
                    .with_boost(10.0),
            )
            .shots(50)
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
            .sampling(
                importance_sampling()
                    .with_uniform_error(0.001)
                    .with_boost(100.0),
            ) // Very aggressive boost
            .shots(2000)
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

        let is_builder = importance_sampling()
            .with_uniform_error(0.01)
            .with_boost(10.0);

        let results1 = sim_neo(circuit.clone())
            .sampling(is_builder.clone())
            .shots(20)
            .seed(42)
            .run();

        let results2 = sim_neo(circuit)
            .sampling(is_builder)
            .shots(20)
            .seed(42)
            .run();

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
            .sampling(
                importance_sampling()
                    .with_p1(0.001)
                    .with_p2(0.01)
                    .with_p_meas(0.001)
                    .with_boost(10.0),
            )
            .shots(100)
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
            .sampling(
                importance_sampling()
                    .with_uniform_error(0.01)
                    .with_boost(10.0),
            )
            .shots(500)
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
            .shots(10)
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
            .shots(50)
            .seed(42)
            .run();

        let custom_results = sim_neo(circuit)
            .quantum(custom_backend(SparseStab::new))
            .shots(50)
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
            .shots(100)
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
            .shots(20)
            .seed(42)
            .run();

        let results2 = sim_neo(circuit)
            .quantum(custom_backend(SparseStab::new))
            .shots(20)
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
    fn test_custom_backend_state_vector() {
        // Verify StateVec also works via custom_backend
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let results = sim_neo(circuit)
            .quantum(custom_backend(StateVec::new))
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

    // --- Register-based Results Tests ---

    #[test]
    fn test_register_counts() {
        // H gate should produce roughly 50/50 split
        let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let mut reg = RegisterMap::new();
        reg.add_register("c", &[QubitId(0)]);

        let results = sim_neo(circuit).shots(200).seed(42).run();
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

        let results = sim_neo(circuit).shots(100).seed(42).run();
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

        let results = sim_neo(circuit).shots(5).seed(42).run();
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

        let results = sim_neo(circuit).shots(10).seed(42).run();
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
            .gate_definitions(defs)
            .shots(10)
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
            .shots(10)
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
            .shots(10)
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
            .shots(10)
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
            .workers(2)
            .shots(10)
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
            .max_decomp_depth(20)
            .shots(10)
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
            .max_decomp_depth(20)
            .workers(2)
            .shots(10)
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
            .shots(10)
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
            .gate_overrides(overrides)
            .shots(10)
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
            .gate_overrides(overrides)
            .workers(2)
            .shots(10)
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
            .shots(10)
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

        // SparseStab is the default backend -- StateVec overrides should panic
        sim_neo(CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build())
            .gate_overrides(overrides)
            .shots(1)
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
            .shots(1)
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
            .gate_overrides(overrides)
            .shots(10)
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
            .gate_definitions(defs)
            .workers(2)
            .shots(10)
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
