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

// Core traits - these are fundamental to using the unified API
pub use crate::{
    ClassicalControlEngine,
    ClassicalControlEngineBuilder, // For .to_sim() method (sim_builder() preferred)
    ClassicalEngine,
    ControlEngine,
    Engine,
};

// Quantum engines and builders
pub use crate::quantum::{
    CliffordRzEngine, CoinTossEngine, DensityMatrixEngine, QuantumEngine, SparseStabEngine,
    StabilizerEngine, StateVecEngine, new_quantum_engine_arbitrary_qgate,
};
pub use crate::quantum_engine_builder::{
    CliffordRzEngineBuilder, CoinTossEngineBuilder, DensityMatrixEngineBuilder,
    IntoQuantumEngineBuilder, SparseStabEngineBuilder, StabilizerEngineBuilder,
    StateVectorEngineBuilder, clifford_rz, coin_toss, density_matrix, sparse_stab, stabilizer,
    state_vector,
};

// Noise models - both traits and common implementations
pub use crate::noise::{
    BiasedDepolarizingNoiseModelBuilder,
    DepolarizingNoiseModel,
    DepolarizingNoiseModelBuilder,
    GeneralNoiseModelBuilder,
    IntoNoiseModel, // Needed for .noise() method to work smoothly
    NoiseModel,
    PassThroughNoiseModel,
    general::GeneralNoiseModel,
};

// Convenience structs for noise configuration
pub use crate::{BiasedDepolarizingNoise, DepolarizingNoise, PassThroughNoise};

// Engine system and stages
pub use crate::{EngineStage, EngineSystem, HybridEngine, MonteCarloEngine, QuantumSystem};

// Message passing
pub use crate::{ByteMessage, ByteMessageBuilder, byte_message::dump_batch};

// Results and data structures
pub use crate::shot_results::{Data, Shot, ShotMap, ShotVec};
pub use crate::{BitVecDisplayFormat, ShotMapDisplay, ShotMapDisplayExt, ShotMapDisplayOptions};

// Simulation builders
pub use crate::sim_builder::{SimBuilder, sim, sim_builder}; // For unified API

pub use serde_json::Value;
