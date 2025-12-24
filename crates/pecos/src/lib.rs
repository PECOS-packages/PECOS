//! # PECOS - Performance Estimator of Codes On Surfaces
//!
//! PECOS is a framework for simulating and evaluating quantum error correction codes.
//! It provides a comprehensive set of tools for quantum simulation, noise modeling,
//! and error correction analysis.
//!
//! ## Features
//!
//! The library functionality is gated behind feature flags:
//!
//! ### Core Features
//! - **`core`**: Core types and error handling (very lightweight)
//! - **`num`**: Numerical computing utilities (scipy-like functions, random numbers)
//! - **`sim`**: Quantum simulation library (includes core and num)
//! - **`runtime`**: Full simulation with QASM and PHIR format support
//!
//! ### Format/Language Support
//! - **`qasm`**: `OpenQASM` 2.0 support (includes sim)
//! - **`phir`**: PHIR JSON format support (includes sim)
//! - **`qis`**: QIS/LLVM IR execution (includes llvm, requires LLVM 14)
//! - **`hugr`**: HUGR program support (includes qis)
//!
//! ### Backends and Extensions
//! - **`llvm`**: LLVM infrastructure (required by qis)
//! - **`quest`**: `QuEST` quantum simulator backend
//! - **`cppsparsesim`**: C++ sparse stabilizer simulator
//! - **`qulacs`**: Qulacs quantum simulator backend
//! - **`wasm`**: WebAssembly foreign object support
//! - **`ldpc`**: LDPC decoder support
//!
//! ## Quick Start
//!
//! Enable the `qasm` feature to simulate QASM programs:
//!
//! ```toml
//! [dependencies]
//! pecos = { version = "...", features = ["qasm"] }
//! ```
//!
//! For PHIR support, add `phir`. For both, use `features = ["runtime"]`.
//!
//! Then use the unified simulation API (see examples in the prelude module).
//!
//! ## Running Examples
//!
//! Most examples require the `runtime` feature (QASM + PHIR). Run them with:
//!
//! ```sh
//! cargo run --example sim_api_final --features runtime
//! cargo run --example unified_sim_demo --features runtime
//! ```
//!
//! For `QuEST` examples, also enable the `quest` feature:
//!
//! ```sh
//! cargo run --example quest_example --features quest
//! ```
//!
//! ## Organized Namespaces
//!
//! PECOS exports functionality through organized namespaces for easy discovery:
//!
//! ### Quantum Simulation (requires `sim` feature)
//! - `engines` - Classical control engines (QASM, QIS, PHIR)
//! - `quantum` - Quantum simulation backends (state vector, sparse stabilizer)
//! - `noise` - Noise models (depolarizing, general, etc.)
//! - `programs` - Program types (QASM, QIS, HUGR, etc.)
//! - `runtime` - QIS runtime implementations
//! - `results` - Result types (Shot, `ShotVec`, `ShotMap`)
//!
//! ### Numerical Computing (requires `num` feature)
//! - `linalg` - Linear algebra operations (norm, etc.)
//! - `random` - Random number generation (NumPy-compatible)
//! - `optimize` - Optimization algorithms (root finding, curve fitting)
//! - `polynomial` - Polynomial fitting and evaluation
//! - `stats` - Statistical functions (mean, std, etc.)
//! - `math` - Mathematical functions (sin, cos, exp, etc.)
//! - `compare` - Comparison utilities (allclose, isclose, etc.)
//!
//! Commonly used functions are also re-exported at the crate root for convenience.
//!
//! ## Program Types
//!
//! PECOS supports multiple quantum program formats:
//! - QASM (`OpenQASM` 2.0)
//! - QIS (Quantum Instruction Set - LLVM IR)
//! - HUGR (Hierarchical Unified Graph Representation)
//! - PHIR JSON (PECOS High-level IR in JSON format)

// ============================================================================
// Core re-exports (available with just the `core` feature)
// ============================================================================

/// Core types and error handling from pecos-core
#[cfg(feature = "core")]
pub mod core {
    pub use pecos_core::*;
}

// Re-export commonly used core types at crate root for convenience
#[cfg(feature = "core")]
pub use pecos_core::{QubitId, errors::PecosError};

// ============================================================================
// Internal modules
// ============================================================================

// Engine type support (requires sim for core simulation types)
#[cfg(feature = "sim")]
pub mod engine_type;

// Full prelude and unified API (require runtime for format support)
#[cfg(feature = "runtime")]
pub mod prelude;
#[cfg(feature = "runtime")]
pub mod program;
#[cfg(feature = "runtime")]
pub mod unified_sim;

// ============================================================================
// Namespace modules for organized exports (require sim feature)
// ============================================================================

/// Classical control engines for quantum program execution
///
/// This module provides builders and types for different classical control engines
/// that parse and execute quantum programs.
///
/// # Available Engines
///
/// - **QASM**: `OpenQASM` 2.0 support via [`qasm_engine()`](qasm_engine)
/// - **QIS**: LLVM IR quantum programs via [`qis_engine()`](qis_engine)
/// - **PHIR JSON**: PHIR JSON format via [`phir_json_engine()`](phir_json_engine)
///
/// # Example
///
/// ```rust
/// use pecos::engines;
/// use pecos_programs::Qasm;
///
/// let program = Qasm::from_string("OPENQASM 2.0; qreg q[1]; h q[0];");
/// let engine = engines::qasm_engine().program(program);
/// ```
#[cfg(feature = "sim")]
pub mod engines {
    #[cfg(feature = "qasm")]
    pub use pecos_qasm::{QASMEngine, QasmEngineBuilder, qasm_engine};

    #[cfg(feature = "qis")]
    pub use pecos_qis_core::{
        QisEngine, QisEngineBuilder, qis_engine, setup_qis_engine_with_runtime,
    };

    #[cfg(feature = "phir")]
    pub use pecos_phir_json::{PhirJsonEngine, PhirJsonEngineBuilder, phir_json_engine};
}

/// Quantum simulation backends and circuit representation
///
/// This module provides builders and types for quantum state simulation backends
/// as well as quantum circuit representation types.
///
/// # Circuit Representation
///
/// - **`DagCircuit`**: DAG-based quantum circuit (nodes=gates, edges=qubit wires)
/// - **`Gate`**: Quantum gate representation
/// - **`GateType`**: Enum of supported gate types
/// - **`QubitId`**: Qubit identifier
///
/// # Simulation Backends
///
/// - **State Vector**: Full quantum state simulation via [`state_vector()`](state_vector)
/// - **Sparse Stabilizer**: Efficient Clifford simulation via [`sparse_stabilizer()`](sparse_stabilizer)
///
/// # Example
///
/// ```rust
/// use pecos::quantum::{DagCircuit, Gate, QubitId};
///
/// // Build a Bell state circuit
/// let mut circuit = DagCircuit::new();
/// let h = circuit.add_gate(Gate::h(&[0]));
/// let cx = circuit.add_gate(Gate::cx(&[(0, 1)]));
/// circuit.connect(h, cx, QubitId::from(0)).unwrap();
///
/// // Or use simulation backends
/// let qengine = pecos::quantum::state_vector();
/// ```
#[cfg(feature = "quantum")]
pub mod quantum {
    // Circuit representation from pecos-quantum
    pub use pecos_quantum::{
        Attribute, DagCircuit, DagWouldCycleError, Gate, GateType, QubitId, Tick, TickCircuit,
    };

    // HUGR conversion (requires hugr feature)
    #[cfg(feature = "hugr")]
    pub use pecos_quantum::hugr_convert::{
        HugrConvertError, gate_type_to_hugr_op, hugr_op_to_gate_type, hugr_to_dag_circuit,
        is_quantum_operation,
    };

    // Re-export read_hugr_envelope for parsing HUGR bytes
    #[cfg(feature = "hugr")]
    pub use pecos_hugr_qis::read_hugr_envelope;

    // Simulation backends (require sim feature)
    #[cfg(feature = "sim")]
    pub use pecos_engines::quantum::{
        QuantumEngine, SparseStabEngine, StateVecEngine, new_quantum_engine_arbitrary_qgate,
    };
    #[cfg(feature = "sim")]
    pub use pecos_engines::quantum_engine_builder::{
        IntoQuantumEngineBuilder, SparseStabilizerEngineBuilder, StateVectorEngineBuilder,
        sparse_stabilizer, state_vector,
    };

    // Re-export feature-gated backends
    #[cfg(feature = "cppsparsesim")]
    pub use pecos_cppsparsesim::CppSparseStab;

    #[cfg(feature = "quest")]
    pub use pecos_quest::{
        QuestDensityMatrix, QuestDensityMatrixEngine, QuestDensityMatrixEngineBuilder,
        QuestStateVec, QuestStateVecEngine, QuestStateVectorEngineBuilder, quest_density_matrix,
        quest_state_vec,
    };

    #[cfg(feature = "qulacs")]
    pub use pecos_qulacs::QulacsStateVec;
}

/// Noise models for quantum simulations
///
/// This module provides noise models and builders for realistic quantum simulations.
///
/// # Available Models
///
/// - **Depolarizing**: Symmetric depolarizing noise
/// - **Biased Depolarizing**: Asymmetric noise with configurable bias
/// - **General**: Flexible noise model for arbitrary noise channels
/// - **Pass-through**: No noise (ideal simulation)
///
/// # Example
///
/// ```rust
/// use pecos::noise::DepolarizingNoise;
///
/// let noise_model = DepolarizingNoise { p: 0.01 };
/// ```
#[cfg(feature = "sim")]
pub mod noise {
    pub use pecos_engines::noise::{
        BiasedDepolarizingNoiseModelBuilder, DepolarizingNoiseModel, DepolarizingNoiseModelBuilder,
        GeneralNoiseModelBuilder, IntoNoiseModel, NoiseModel, PassThroughNoiseModel,
        general::GeneralNoiseModel,
    };

    pub use pecos_engines::{BiasedDepolarizingNoise, DepolarizingNoise, PassThroughNoise};
}

/// Program types for quantum circuits
///
/// This module provides program representations for different quantum computing frameworks.
///
/// # Available Program Types
///
/// - **`Qasm`**: `OpenQASM` 2.0 programs
/// - **`Qis`**: LLVM IR based quantum programs
/// - **`Hugr`**: HUGR-based quantum programs
///
/// # Example
///
/// ```rust
/// use pecos::programs::Qasm;
///
/// let program = Qasm::from_string("OPENQASM 2.0; qreg q[1]; h q[0];");
/// ```
#[cfg(feature = "sim")]
pub mod programs {
    pub use pecos_programs::{Hugr, Program, Qasm, Qis};
}

/// QIS runtime implementations
///
/// This module provides Selene-based QIS interface and runtime implementations.
///
/// # Available Runtimes
///
/// - **Selene**: Selene-based runtime via [`SeleneRuntime`] (requires `qis` feature)
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "qis")]
/// # {
/// use pecos::runtime::selene_simple_runtime;
///
/// let runtime = selene_simple_runtime();
/// # }
/// ```
#[cfg(feature = "qis")]
pub mod runtime {
    // Re-export Selene interface
    pub use pecos_qis_selene::{
        HeliosInterfaceBuilder, QisHeliosInterface, SeleneRuntime, helios_interface_builder,
        selene_runtime_auto, selene_simple_runtime,
    };

    // Re-export core runtime types
    pub use pecos_qis_core::{ClassicalState, QisRuntime, RuntimeError};
}

/// Simulation results and data types
///
/// This module provides types for representing simulation results.
///
/// # Main Types
///
/// - [`Shot`] - A single measurement shot result
/// - [`ShotVec`] - A vector of shots
/// - [`ShotMap`] - A map of register names to measurement results
/// - [`Data`] - Measurement data representation
///
/// # Example
///
/// ```rust
/// use pecos::results::{ShotVec, ShotMap};
///
/// // Results from simulation
/// fn process_results(results: ShotVec) {
///     let shot_map = results.try_as_shot_map().unwrap();
///     // Process the shot map...
/// }
/// ```
#[cfg(feature = "sim")]
pub mod results {
    pub use pecos_engines::shot_results::{Data, Shot, ShotMap, ShotVec};
    pub use pecos_engines::{
        BitVecDisplayFormat, ShotMapDisplay, ShotMapDisplayExt, ShotMapDisplayOptions,
    };
}

/// WebAssembly foreign object support
///
/// This module provides WebAssembly execution support for classical computations
/// within PECOS quantum programs (QASM and PHIR).
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "wasm")]
/// # {
/// use pecos::wasm::WasmForeignObject;
/// use std::path::Path;
///
/// // Load a WASM module
/// let wasm_obj = WasmForeignObject::new(Path::new("module.wasm")).unwrap();
/// # }
/// ```
#[cfg(feature = "wasm")]
pub mod wasm {
    pub use pecos_wasm::{ForeignObject, WasmForeignObject};
}

// ============================================================================
// Numerical computing namespace modules (pecos-num) - require sim
// ============================================================================

/// Linear algebra operations
///
/// This module provides linear algebra operations for vectors and matrices.
///
/// # Available Functions
///
/// - **`norm()`** - Vector/matrix norm calculation (L2 norm by default)
///
/// # Example
///
/// ```rust
/// use pecos::linalg;
/// use pecos::prelude::*;
///
/// let vec = Array1::from_vec(vec![3.0, 4.0]);
/// let norm = linalg::norm(&vec.view(), None); // None = L2 norm
/// assert!((norm - 5.0).abs() < 1e-10);
/// ```
#[cfg(feature = "num")]
pub mod linalg {
    pub use pecos_num::linalg::*;
}

/// Random number generation
///
/// This module provides NumPy-compatible random number generation functions.
///
/// # Available Functions
///
/// - **`seed()`** - Set the random seed for reproducibility
/// - **`randint()`** - Generate random integers in range [low, high)
/// - **`random()`** - Generate random floats in [0, 1)
/// - **`choice()`** - Random sampling from arrays
/// - **`shuffle()`** - Shuffle arrays in-place
/// - And more...
///
/// # Example
///
/// ```rust
/// use pecos::random;
///
/// // Set seed for reproducibility
/// random::seed(42);
///
/// // Generate random integers in range [0, 10), size 100
/// let samples = random::randint(0, Some(10), 100);
/// assert_eq!(samples.len(), 100);
/// ```
#[cfg(feature = "num")]
pub mod random {
    pub use pecos_num::random::*;
}

/// Optimization algorithms
///
/// This module provides root finding and optimization algorithms.
///
/// # Available Functions
///
/// - **`brentq()`** - Brent's method for root finding
/// - **`newton()`** - Newton-Raphson method for root finding
///
/// # Example
///
/// ```rust
/// use pecos::optimize;
///
/// // Find root of x^2 - 2 = 0 in range [0, 2]
/// let root = optimize::brentq(|x| x * x - 2.0, 0.0, 2.0, None).unwrap();
/// assert!((root - std::f64::consts::SQRT_2).abs() < 1e-10);
/// ```
#[cfg(feature = "num")]
pub mod optimize {
    pub use pecos_num::optimize::*;
}

/// Polynomial operations
///
/// This module provides polynomial fitting and evaluation.
///
/// # Available Functions
///
/// - **`polyfit()`** - Fit polynomial to data
/// - **`Poly1d`** - Polynomial evaluation and manipulation
///
/// # Example
///
/// ```rust
/// use pecos::polynomial;
/// use pecos::prelude::*;
///
/// let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
/// let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);
///
/// // Fit linear polynomial (degree 1): y = mx + b
/// let coeffs = polynomial::polyfit(x.view(), y.view(), 1).unwrap();
/// assert_eq!(coeffs.len(), 2); // [b, m]
/// ```
#[cfg(feature = "num")]
pub mod polynomial {
    pub use pecos_num::polynomial::*;
}

/// Statistical functions
///
/// This module provides statistical analysis functions.
///
/// # Available Functions
///
/// - **`mean()`** - Calculate mean/average
/// - **`std()`** - Calculate standard deviation
/// - And more...
///
/// # Example
///
/// ```rust
/// use pecos::stats;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let avg = stats::mean(&data);
/// assert_eq!(avg, 3.0);
/// ```
#[cfg(feature = "num")]
pub mod stats {
    pub use pecos_num::stats::*;
}

/// Mathematical functions
///
/// This module provides mathematical functions for arrays and scalars.
///
/// # Available Functions
///
/// - Trigonometric: `sin()`, `cos()`, `tan()`, etc.
/// - Hyperbolic: `sinh()`, `cosh()`, `tanh()`, etc.
/// - Exponential: `exp()`, `log()`, `ln()`, etc.
/// - Power: `sqrt()`, `power()`, etc.
///
/// # Example
///
/// ```rust
/// use pecos::math;
///
/// let x = std::f64::consts::PI / 2.0;
/// let result = math::sin(x);
/// assert!((result - 1.0).abs() < 1e-10);
/// ```
#[cfg(feature = "num")]
pub mod math {
    pub use pecos_num::math::*;
}

/// Comparison and logical operations
///
/// This module provides comparison utilities for floating-point values.
///
/// # Available Functions
///
/// - **`isclose()`** - Element-wise approximate equality
/// - **`allclose()`** - Array approximate equality with tolerance
/// - **`isnan()`** - Check for NaN values
///
/// # Example
///
/// ```rust
/// use pecos::compare;
/// use pecos::prelude::*;
///
/// let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array1::from_vec(vec![1.0 + 1e-9, 2.0, 3.0]);
///
/// // allclose(a, b, rtol, atol, equal_nan)
/// assert!(compare::allclose(&a.view(), &b.view(), 1e-8, 1e-8, false));
/// ```
#[cfg(feature = "num")]
pub mod compare {
    pub use pecos_num::compare::*;
}

/// Graph algorithms for quantum error correction
///
/// This module provides graph data structures and algorithms for quantum error
/// correction, particularly the MWPM (Minimum Weight Perfect Matching) decoder.
///
/// # Main Types
///
/// - **`Graph`** - Undirected graph with weighted edges
/// - **`DiGraph`** - Directed graph
/// - **`DAG`** - Directed acyclic graph with cycle checking
///
/// # Available Functions
///
/// - **`max_weight_matching()`** - Compute maximum weight matching (used in MWPM decoder)
///
/// # Example
///
/// ```rust
/// use pecos::graph::Graph;
///
/// let mut graph = Graph::new();
/// let n0 = graph.add_node();
/// let n1 = graph.add_node();
/// let n2 = graph.add_node();
/// let n3 = graph.add_node();
///
/// graph.add_edge(n0, n1).weight(10.0);
/// graph.add_edge(n2, n3).weight(20.0);
///
/// let matching = graph.max_weight_matching(false);
/// assert_eq!(matching.len(), 4); // Two pairs, each appearing twice
/// ```
#[cfg(feature = "num")]
pub mod graph {
    pub use pecos_num::graph::*;
}

/// Directed graph data structure
///
/// This module provides the `DiGraph` type for directed graphs.
/// Unlike `DAG`, this type allows cycles.
///
/// # Example
///
/// ```rust
/// use pecos::digraph::DiGraph;
///
/// let mut g = DiGraph::new();
/// let n0 = g.add_node();
/// let n1 = g.add_node();
/// g.add_edge(n0, n1);
///
/// assert_eq!(g.successors(n0), vec![n1]);
/// assert_eq!(g.predecessors(n1), vec![n0]);
/// ```
#[cfg(feature = "num")]
pub mod digraph {
    pub use pecos_num::digraph::*;
}

/// Directed acyclic graph (DAG) data structure
///
/// This module provides the `DAG` type which enforces acyclicity at runtime.
/// Adding an edge that would create a cycle returns an error.
///
/// # Example
///
/// ```rust
/// use pecos::dag::DAG;
///
/// let mut g = DAG::new();
/// let n0 = g.add_node();
/// let n1 = g.add_node();
/// g.add_edge(n0, n1).unwrap();
///
/// // Topological sort always succeeds for a DAG
/// let order = g.topological_sort();
/// ```
#[cfg(feature = "num")]
pub mod dag {
    pub use pecos_num::dag::*;
}

/// Quantum error correction decoders
///
/// This module provides decoders for quantum error correction codes.
///
/// # Available Decoders (feature-gated)
///
/// With `ldpc` feature:
/// - **`BpOsdDecoder`** - Belief propagation with ordered statistics decoding
/// - **`BpLsdDecoder`** - Belief propagation with localized statistics decoding
/// - **`UnionFindDecoder`** - Union-find decoder
/// - **`BeliefFindDecoder`** - Belief-find decoder
/// - **`FlipDecoder`** - Flip decoder
/// - **`MbpDecoder`** - Modified belief propagation decoder
/// - **`SoftInfoBpDecoder`** - Soft information BP decoder
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "ldpc")]
/// # {
/// use pecos::decoders::{Decoder, BpOsdDecoder};
/// # }
/// ```
#[cfg(any(feature = "ldpc", feature = "all-decoders"))]
pub mod decoders {
    pub use pecos_decoders::*;
}

/// Quantum simulation implementations
///
/// This module provides low-level quantum simulation implementations and utilities
/// from pecos-qsim, including stabilizer simulators, state vectors, and measurement
/// samplers.
///
/// # Available Types
///
/// - **Simulators**: `SparseStab`, `StateVec`, `SymbolicSparseStab`
/// - **Measurement Sampling**: `MeasurementSampler`
/// - **Utilities**: `CliffordGateable`, `ArbitraryRotationGateable`
///
/// # Example
///
/// ```rust
/// use pecos::qsim::measurement_sampler::MeasurementSampler;
/// use pecos::prelude::*;
///
/// let mut sim = StdSymbolicSparseStab::new(2);
/// sim.h(0).cx(0, 1);
/// sim.mz(0);
/// sim.mz(1);
///
/// let sampler = MeasurementSampler::new(sim.measurement_history());
/// let samples = sampler.sample(1000);
/// ```
#[cfg(feature = "sim")]
pub mod qsim {
    pub use pecos_qsim::*;
}

// ============================================================================
// Top-level re-exports for convenience and backward compatibility
// (require sim feature unless otherwise noted)
// ============================================================================

// Engine builders
#[cfg(feature = "qasm")]
pub use pecos_qasm::{QasmEngineBuilder, qasm_engine, run_qasm};

#[cfg(feature = "qis")]
pub use pecos_qis_core::{QisEngineBuilder, qis_engine, setup_qis_engine_with_runtime};

#[cfg(feature = "phir")]
pub use pecos_phir::PhirConfig;
#[cfg(feature = "phir")]
pub use pecos_phir_json::{PhirJsonEngineBuilder, phir_json_engine};

// Quantum backends
#[cfg(feature = "sim")]
pub use pecos_engines::{sparse_stabilizer, state_vector};

// Noise models
#[cfg(feature = "sim")]
pub use pecos_engines::{
    BiasedDepolarizingNoise, DepolarizingNoise, GeneralNoiseModelBuilder, PassThroughNoiseModel,
};

// Program types
#[cfg(feature = "sim")]
pub use pecos_programs::{Hugr, Program, Qasm, Qis};

// Selene interface (when feature is enabled)
#[cfg(feature = "qis")]
pub use pecos_qis_selene::{
    HeliosInterfaceBuilder, QisHeliosInterface, SeleneRuntime, helios_interface_builder,
    selene_runtime_auto, selene_simple_runtime,
};

// Simulation API
#[cfg(feature = "sim")]
pub use pecos_engines::{SimInput, sim_builder};
#[cfg(feature = "runtime")]
pub use unified_sim::{ProgrammedSimBuilder, SimBuilderExt, sim};

// Engine type support
#[cfg(feature = "sim")]
pub use engine_type::{DynamicEngineBuilder, EngineType, sim_dynamic};

// Feature-gated quantum backends
#[cfg(feature = "cppsparsesim")]
pub use pecos_cppsparsesim::CppSparseStab;

#[cfg(feature = "quest")]
pub use pecos_quest::{
    QuestDensityMatrix, QuestDensityMatrixEngine, QuestDensityMatrixEngineBuilder, QuestStateVec,
    QuestStateVecEngine, QuestStateVectorEngineBuilder, quest_density_matrix, quest_state_vec,
};

#[cfg(feature = "qulacs")]
pub use pecos_qulacs::QulacsStateVec;

// WebAssembly foreign object support
#[cfg(feature = "wasm")]
pub use pecos_wasm::{ForeignObject, WasmForeignObject};

// Numerical computing - commonly used functions at top level for convenience
#[cfg(feature = "num")]
pub use pecos_num::{
    Poly1d,
    // Comparison utilities
    allclose,
    // Optimization algorithms
    brentq,
    curve_fit,
    // Statistical functions
    mean,
    newton,
    // Polynomial operations
    polyfit,
};
