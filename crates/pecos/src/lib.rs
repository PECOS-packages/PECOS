//! # PECOS - Performance Estimator of Codes On Surfaces
//!
//! Quantum error correction simulation framework. Feature-gated:
//!
//! - **`core`**: Core types and error handling
//! - **`sim`**: Quantum simulation (includes core + num)
//! - **`runtime`**: Full simulation with QASM + PHIR support
//! - **`qis`**: QIS/LLVM IR execution (requires LLVM 14)
//! - **`hugr`**: HUGR program support
//! - **`quest`/`qulacs`/`cppsparsestab`**: Simulator backends
//! - **`num`**: Numerical computing (scipy-like)
//!
//! See `Cargo.toml` for the full feature list.

// Core re-exports
#[cfg(feature = "core")]
pub mod core {
    pub use pecos_core::*;
}
#[cfg(feature = "core")]
pub use pecos_core::{QubitId, errors::PecosError};

// Internal modules
#[cfg(feature = "sim")]
pub mod engine_type;
#[cfg(feature = "runtime")]
pub mod prelude;
#[cfg(feature = "runtime")]
pub mod program;
#[cfg(feature = "runtime")]
pub mod unified_sim;

/// Classical control engines (QASM, QIS, PHIR).
#[cfg(feature = "sim")]
pub mod engines {
    #[cfg(feature = "hugr")]
    pub use pecos_hugr::{HugrEngine, HugrEngineBuilder, hugr_engine, hugr_sim};
    #[cfg(feature = "phir")]
    pub use pecos_phir::{PhirEngine, PhirEngineBuilder, phir_engine};
    #[cfg(feature = "phir")]
    pub use pecos_phir_json::{PhirJsonEngine, PhirJsonEngineBuilder, phir_json_engine};
    #[cfg(feature = "qasm")]
    pub use pecos_qasm::{QASMEngine, QasmEngineBuilder, qasm_engine};
    #[cfg(feature = "qis")]
    pub use pecos_qis::{QisEngine, QisEngineBuilder, qis_engine, setup_qis_engine_with_runtime};
}

/// Quantum circuit representation and Pauli algebra.
#[cfg(feature = "quantum")]
pub mod quantum {
    #[cfg(feature = "hugr")]
    pub use pecos_hugr_qis::read_hugr_envelope;
    #[cfg(feature = "hugr")]
    pub use pecos_quantum::hugr_convert::{
        HugrConvertError, NotSimpleError, SimpleHugr, dag_circuit_to_hugr, gate_type_to_hugr_op,
        hugr_op_to_gate_type, hugr_to_dag_circuit, is_quantum_operation,
    };
    pub use pecos_quantum::{
        Attribute, Circuit, CircuitMut, CustomGateError, DagCircuit, DagWouldCycleError, Gate,
        GateHandle, GateType, GateView, QubitId, Tick, TickCircuit,
    };
    pub use pecos_quantum::{F2Matrix, PauliSequence, PauliSet, PauliStabilizerGroup};
}

/// Quantum simulator backends and engine builders.
#[cfg(feature = "sim")]
pub mod simulators {
    #[cfg(feature = "cppsparsestab")]
    pub use pecos_cppsparsestab::CppSparseStab;
    pub use pecos_engines::quantum::{
        DensityMatrixEngine, QuantumEngine, SparseStabEngine, StabVecEngine, StabilizerEngine,
        StateVecEngine, new_quantum_engine_arbitrary_qgate,
    };
    pub use pecos_engines::quantum_engine_builder::{
        DensityMatrixEngineBuilder, IntoQuantumEngineBuilder, SparseStabEngineBuilder,
        StabVecEngineBuilder, StabilizerEngineBuilder, StateVectorEngineBuilder, density_matrix,
        sparse_stab, stab_vec, stabilizer, state_vector,
    };
    pub use pecos_simulators::*;
}

/// Noise models for quantum simulations.
#[cfg(feature = "sim")]
pub mod noise {
    pub use pecos_engines::noise::{
        BiasedDepolarizingNoiseModelBuilder, DepolarizingNoiseModel, DepolarizingNoiseModelBuilder,
        GeneralNoiseModelBuilder, IntoNoiseModel, NoiseModel, PassThroughNoiseModel,
        general::GeneralNoiseModel,
    };
    pub use pecos_engines::{BiasedDepolarizingNoise, DepolarizingNoise, PassThroughNoise};
}

/// Program types (Qasm, Qis, Hugr).
#[cfg(feature = "sim")]
pub mod programs {
    pub use pecos_programs::{Hugr, Program, Qasm, Qis};
}

/// QIS runtime (Selene + Helios interface).
#[cfg(feature = "qis")]
pub mod runtime {
    pub use pecos_qis::{ClassicalState, QisRuntime, RuntimeError};
    pub use pecos_qis::{
        HeliosInterfaceBuilder, QisHeliosInterface, SeleneRuntime, helios_interface_builder,
        selene_runtime_auto, selene_simple_runtime,
    };
}

/// Simulation result types (Shot, `ShotVec`, `ShotMap`).
#[cfg(feature = "sim")]
pub mod results {
    pub use pecos_engines::shot_results::{Data, DataVec, Shot, ShotMap, ShotVec};
    pub use pecos_engines::{
        BitVecDisplayFormat, ShotMapDisplay, ShotMapDisplayExt, ShotMapDisplayOptions,
    };
}

/// WebAssembly foreign object support.
#[cfg(feature = "wasm")]
pub mod wasm {
    pub use pecos_wasm::{ForeignObject, WasmForeignObject};
}

// Numerical computing modules
#[cfg(feature = "num")]
pub mod linalg {
    pub use pecos_num::linalg::*;
}
#[cfg(feature = "num")]
pub mod array {
    pub use pecos_num::array::*;
}
#[cfg(feature = "num")]
pub mod random {
    pub use pecos_num::random::*;
}
#[cfg(feature = "num")]
pub mod optimize {
    pub use pecos_num::optimize::*;
}
#[cfg(feature = "num")]
pub mod polynomial {
    pub use pecos_num::polynomial::*;
}
#[cfg(feature = "num")]
pub mod stats {
    pub use pecos_num::stats::*;
}
#[cfg(feature = "num")]
pub mod math {
    pub use pecos_num::math::*;
}
#[cfg(feature = "num")]
pub mod compare {
    pub use pecos_num::compare::*;
}
#[cfg(feature = "num")]
pub mod graph {
    pub use pecos_num::graph::*;
}
#[cfg(feature = "num")]
pub mod digraph {
    pub use pecos_num::digraph::*;
}
#[cfg(feature = "num")]
pub mod dag {
    pub use pecos_num::dag::*;
}

/// Quantum error correction decoders.
#[cfg(any(
    feature = "ldpc",
    feature = "pymatching",
    feature = "fusion-blossom",
    feature = "tesseract",
    feature = "chromobius",
    feature = "relay-bp",
    feature = "all-decoders"
))]
pub mod decoders {
    pub use pecos_decoders::*;
}

/// Quantum error correction and fault tolerance analysis.
#[cfg(feature = "qec")]
pub mod qec {
    pub use pecos_qec::*;
}

// Top-level re-exports for convenience

#[cfg(feature = "sim")]
pub use engine_type::{DynamicEngineBuilder, EngineType, sim_dynamic};
#[cfg(feature = "cppsparsestab")]
pub use pecos_cppsparsestab::CppSparseStab;
#[cfg(feature = "sim")]
pub use pecos_engines::{
    BiasedDepolarizingNoise, DepolarizingNoise, GeneralNoiseModelBuilder, PassThroughNoiseModel,
};
#[cfg(feature = "sim")]
pub use pecos_engines::{SimInput, sim_builder};
#[cfg(feature = "sim")]
pub use pecos_engines::{
    coin_toss, density_matrix, sparse_stab, stab_vec, stabilizer, state_vector,
};
#[cfg(feature = "hugr")]
pub use pecos_hugr::{HugrEngine, HugrEngineBuilder, hugr_engine, hugr_sim};
#[cfg(feature = "num")]
pub use pecos_num::{Poly1d, allclose, brentq, curve_fit, mean, newton, polyfit};
#[cfg(feature = "phir")]
pub use pecos_phir::{PhirConfig, PhirEngineBuilder, phir_engine};
#[cfg(feature = "phir")]
pub use pecos_phir_json::{PhirJsonEngineBuilder, phir_json_engine};
#[cfg(feature = "sim")]
pub use pecos_programs::{Hugr, Program, Qasm, Qis};
#[cfg(feature = "qasm")]
pub use pecos_qasm::{QasmEngineBuilder, qasm_engine, run_qasm};
#[cfg(feature = "qis")]
pub use pecos_qis::{
    HeliosInterfaceBuilder, QisHeliosInterface, SeleneRuntime, helios_interface_builder,
    selene_runtime_auto, selene_simple_runtime,
};
#[cfg(feature = "qis")]
pub use pecos_qis::{QisEngineBuilder, qis_engine, setup_qis_engine_with_runtime};
#[cfg(feature = "wasm")]
pub use pecos_wasm::{ForeignObject, WasmForeignObject};
#[cfg(feature = "runtime")]
pub use unified_sim::{ProgrammedSimBuilder, SimBuilderExt, sim};
