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
//! let program = QasmProgram::from_string(qasm_code);
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
//! - [`engines`] - Classical control engines (QASM, QIS, PHIR)
//! - [`quantum`] - Quantum simulation backends (state vector, sparse stabilizer)
//! - [`noise`] - Noise models (depolarizing, general, etc.)
//! - [`programs`] - Program types (QASM, QIS, HUGR, etc.)
//! - [`runtime`] - QIS runtime implementations
//! - [`results`] - Result types (Shot, `ShotVec`, `ShotMap`)
//!
//! All types are also re-exported at the crate root for convenience.
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
/// use pecos_programs::QasmProgram;
///
/// let program = QasmProgram::from_string("OPENQASM 2.0; qreg q[1]; h q[0];");
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
/// - **`QasmProgram`**: `OpenQASM` 2.0 programs
/// - **`QisProgram`**: LLVM IR based quantum programs
/// - **`HugrProgram`**: HUGR-based quantum programs
///
/// # Example
///
/// ```rust
/// use pecos::programs::QasmProgram;
///
/// let program = QasmProgram::from_string("OPENQASM 2.0; qreg q[1]; h q[0];");
/// ```
pub mod programs {
    pub use pecos_programs::{HugrProgram, Program, QasmProgram, QisProgram};
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
pub use pecos_programs::{HugrProgram, Program, QasmProgram, QisProgram};

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
