// Copyright 2024 The PECOS Developers
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

// re-exporting pecos-core
pub use pecos_core::{IndexableElement, Set, VecSet};

// re-exporting pecos-engines
pub use pecos_engines::{
    ByteMessage, ByteMessageBuilder, ClassicalEngine, ControlEngine, DepolarizingNoiseModel,
    Engine, EngineStage, EngineSystem, HybridEngine, MonteCarloEngine, NoiseModel, PHIREngine,
    QirEngine, QuantumEngine, QuantumSystem, QueueError, ShotResult, ShotResults,
};

// Re-exporting noise models
pub use pecos_core::rng::RngManageable;
pub use pecos_core::rng::rng_manageable::derive_seed;
pub use pecos_engines::engines::noise::general::GeneralNoiseModel;

// Re-exporting specific implementations that aren't at the crate root
pub use pecos_engines::engines::{
    classical::{ProgramType, detect_program_type, get_program_path, setup_engine},
    quantum::{SparseStabEngine, StateVecEngine, new_quantum_engine_arbitrary_qgate},
};

// Re-exporting byte_message functions
pub use pecos_engines::byte_message::dump_batch;

// re-exporting pecos-qsim
pub use pecos_qsim::{
    ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, SparseStab, StateVec,
};
