//! # PECOS - Performance Estimator of Codes On Surfaces
//!
//! PECOS is a framework for simulating and evaluating quantum error correction codes.
//! It provides a comprehensive set of tools for quantum simulation, noise modeling,
//! and error correction analysis.
//!
//! ## Quick Start
//!
//! The easiest way to use PECOS is through the unified simulation API:
//!
//! ```rust,no_run
//! use pecos::prelude::*;
//! use pecos::quantum::sparse_stabilizer;
//!
//! // Create a QASM program
//! let qasm_code = r#"
//!     OPENQASM 2.0;
//!     include "qelib1.inc";
//!     qreg q[2];
//!     creg c[2];
//!     h q[0];
//!     cx q[0], q[1];
//!     measure q -> c;
//! "#;
//!
//! let program = Qasm::from_string(qasm_code);
//!
//! // Run simulation
//! let results = sim(program)
//!     .quantum(sparse_stabilizer())
//!     .seed(42)
//!     .run(1000)?;
//!
//! println!("Got {} shots", results.len());
//! # Ok::<(), pecos_core::errors::PecosError>(())
//! ```
//!
//! ## Organized Namespaces
//!
//! PECOS exports functionality through organized namespaces for easy discovery:
//!
//! ### Quantum Simulation
//! - [`engines`] - Classical control engines (QASM, QIS, PHIR)
//! - [`quantum`] - Quantum simulation backends (state vector, sparse stabilizer)
//! - [`noise`] - Noise models (depolarizing, general, etc.)
//! - [`programs`] - Program types (QASM, QIS, HUGR, etc.)
//! - [`runtime`] - QIS runtime implementations
//! - [`results`] - Result types (Shot, `ShotVec`, `ShotMap`)
//!
//! ### Numerical Computing
//! - [`linalg`] - Linear algebra operations (norm, etc.)
//! - [`random`] - Random number generation (NumPy-compatible)
//! - [`optimize`] - Optimization algorithms (root finding, curve fitting)
//! - [`polynomial`] - Polynomial fitting and evaluation
//! - [`stats`] - Statistical functions (mean, std, etc.)
//! - [`math`] - Mathematical functions (sin, cos, exp, etc.)
//! - [`compare`] - Comparison utilities (allclose, isclose, etc.)
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
//!
//! ## Features
//!
//! PECOS supports a variety of noise models and quantum simulators. Check the documentation
//! for the simulation builders and noise models for more details on the available options.

// ============================================================================
// Internal modules
// ============================================================================

pub mod engine_type;
pub mod prelude;
pub mod program;
pub mod unified_sim;

// ============================================================================
// Namespace modules for organized exports
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
/// ```rust,no_run
/// # use pecos_core::errors::PecosError;
/// # fn example() -> Result<(), PecosError> {
/// use pecos::engines;
/// use pecos_programs::Qasm;
///
/// let program = Qasm::from_string("OPENQASM 2.0; qreg q[1]; h q[0];");
/// let engine = engines::qasm_engine().program(program);
/// # Ok(())
/// # }
/// ```
pub mod engines {
    #[cfg(feature = "qasm")]
    pub use pecos_qasm::{QASMEngine, QasmEngineBuilder, qasm_engine};

    pub use pecos_qis_core::{
        QisEngine, QisEngineBuilder, qis_engine, setup_qis_engine_with_runtime,
    };

    #[cfg(feature = "phir")]
    pub use pecos_phir_json::{PhirJsonEngine, PhirJsonEngineBuilder, phir_json_engine};
}

/// Quantum simulation backends
///
/// This module provides builders and types for different quantum state simulation backends.
///
/// # Available Backends
///
/// - **State Vector**: Full quantum state simulation via [`state_vector()`](state_vector)
/// - **Sparse Stabilizer**: Efficient Clifford simulation via [`sparse_stabilizer()`](sparse_stabilizer)
///
/// # Example
///
/// ```rust
/// use pecos::quantum;
///
/// // Create a state vector quantum backend
/// let qengine = quantum::state_vector();
///
/// // Or use sparse stabilizer for efficient Clifford simulation
/// let qengine = quantum::sparse_stabilizer();
/// ```
pub mod quantum {
    pub use pecos_engines::quantum::{
        QuantumEngine, SparseStabEngine, StateVecEngine, new_quantum_engine_arbitrary_qgate,
    };
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
pub mod programs {
    pub use pecos_programs::{Hugr, Program, Qasm, Qis};
}

/// QIS runtime implementations
///
/// This module provides Selene-based QIS interface and runtime implementations.
///
/// # Available Runtimes
///
/// - **Selene**: Selene-based runtime via [`SeleneRuntime`] (requires `selene` feature)
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "selene")]
/// # {
/// use pecos::runtime::selene_simple_runtime;
///
/// let runtime = selene_simple_runtime();
/// # }
/// ```
pub mod runtime {
    // Re-export Selene interface when feature is enabled
    #[cfg(feature = "selene")]
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
// Numerical computing namespace modules (pecos-num)
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
pub mod graph {
    pub use pecos_num::graph::*;
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
pub mod qsim {
    pub use pecos_qsim::*;
}

// ============================================================================
// Top-level re-exports for convenience and backward compatibility
// ============================================================================

// Engine builders
#[cfg(feature = "qasm")]
pub use pecos_qasm::{QasmEngineBuilder, qasm_engine, run_qasm};

pub use pecos_qis_core::{QisEngineBuilder, qis_engine, setup_qis_engine_with_runtime};

#[cfg(feature = "phir")]
pub use pecos_phir::PhirConfig;
#[cfg(feature = "phir")]
pub use pecos_phir_json::{PhirJsonEngineBuilder, phir_json_engine};

// Quantum backends
pub use pecos_engines::{sparse_stabilizer, state_vector};

// Noise models
pub use pecos_engines::{
    BiasedDepolarizingNoise, DepolarizingNoise, GeneralNoiseModelBuilder, PassThroughNoiseModel,
};

// Program types
pub use pecos_programs::{Hugr, Program, Qasm, Qis};

// Selene interface (when feature is enabled)
#[cfg(feature = "selene")]
pub use pecos_qis_selene::{
    HeliosInterfaceBuilder, QisHeliosInterface, SeleneRuntime, helios_interface_builder,
    selene_runtime_auto, selene_simple_runtime,
};

// Simulation API
pub use pecos_engines::{SimInput, sim_builder};
pub use unified_sim::{ProgrammedSimBuilder, SimBuilderExt, sim};

// Engine type support
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
