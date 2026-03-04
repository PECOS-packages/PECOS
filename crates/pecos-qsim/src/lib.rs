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

pub mod arbitrary_rotation_gateable;
pub mod batched_ops;
pub mod circuit_executor;
pub mod clifford_frame;
pub mod clifford_gateable;
pub mod clifford_test_utils;
pub mod coin_toss;
pub mod dense_stab;
pub mod dense_stab_variants;
pub mod density_matrix;
pub mod density_matrix_test_utils;
pub mod gens;
pub mod gpu_stab;
pub mod gpu_stab_opt;
pub mod gpu_stab_parallel;
pub mod graph_state;
pub mod graph_state_repr;
pub mod measurement_sampler;
pub mod pauli_prop;
// pub mod paulis;
pub mod prelude;
pub mod quantum_simulator;
pub mod rotation_test_utils;
pub mod sign_algebra;
pub mod sparse_stab;
pub mod stab;
pub mod stabilizer_tableau;
pub mod stabilizer_test_utils;
pub mod state_vec;
pub mod state_vec_aos;
pub mod state_vec_soa;
pub mod state_vec_soa32;
pub mod state_vec_sparse_aos;
pub mod state_vec_sparse_soa;
pub mod state_vector_test_utils;
pub mod symbolic_gens;
pub mod symbolic_sparse_stab;
pub mod symbolic_sparse_stab_bitset;

pub use arbitrary_rotation_gateable::ArbitraryRotationGateable;
pub use batched_ops::{BatchedOps, CommandBuffer, RawOps};
pub use circuit_executor::{CircuitExecutor, GateSystem, GateSystemRegistry, execute_batched};
pub use clifford_gateable::{CliffordGateable, MeasurementResult};
pub use coin_toss::CoinToss;
/// Sparse index representation of stabilizer/destabilizer generators.
///
/// Returns `(col_x, col_z, row_x, row_z)` where each is a `Vec<Vec<usize>>`.
pub type GensData = (
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
);

pub use dense_stab::DenseStab;
pub use dense_stab_variants::{DenseStabColOnly, DenseStabRowOnly, SparseColOnly, SparseRowOnly};
pub use density_matrix::DensityMatrix;
pub use gens::{Gens, GensBitSet, GensGeneric, GensHybrid, GensVecSet, PauliClassification};
pub use gpu_stab::GpuStab;
pub use gpu_stab_opt::GpuStabOpt;
pub use gpu_stab_parallel::GpuStabParallel;
pub use graph_state::GraphStateSim;
pub use graph_state_repr::{GraphState, GraphStateRenderer};
// pub use paulis::Paulis;
pub use measurement_sampler::{
    MeasurementKind, MeasurementSampler, MeasurementValidationError, SampleResult,
    SequentialMeasurementSampler,
};
pub use pauli_prop::PauliProp;
pub use pecos_core::{VecSet, qid, qid2, qids, qids2};
pub use quantum_simulator::QuantumSimulator;
pub use sign_algebra::{PhaseSign, SignAlgebra, SymbolicSign};
pub use sparse_stab::{
    SparseStab, SparseStabBitSet, SparseStabGeneric, SparseStabHybrid, SparseStabSortedVecSet,
    SparseStabUnsortedVecSet, SparseStabVecSet,
};
pub use stab::Stab;
pub use stabilizer_tableau::StabilizerTableauSimulator;
// StateVec uses the sparse SoA implementation optimized for QEC workloads.
// The dense implementation is available as DenseStateVec / StateVecSoA.
pub use state_vec::StateVec as StateVecOld;
pub use state_vec_aos::StateVecAoS;
pub use state_vec_soa::StateVecSoA as DenseStateVec;
pub use state_vec_soa::StateVecSoA;
pub use state_vec_soa32::StateVecSoA32;
pub use state_vec_sparse_aos::SparseStateVecAoS;
pub use state_vec_sparse_soa::SparseStateVecSoA as StateVec;
pub use state_vec_sparse_soa::SparseStateVecSoA;
// Alias for backwards compatibility and common usage
pub use state_vec_sparse_aos::SparseStateVecAoS as SparseStateVec;
pub use symbolic_gens::{
    SymbolicGens, SymbolicGensBitSet, SymbolicGensGeneric, SymbolicGensVecSet,
};
pub use symbolic_sparse_stab::{
    MeasurementHistory, SymbolicMeasurementResult, SymbolicSparseStabVecSet,
};
pub use symbolic_sparse_stab_bitset::SymbolicSparseStab;

// Re-export stabilizer testing utilities
pub use stabilizer_test_utils::{ForcedMeasurement, StabilizerSimulator};

// Re-export state vector testing utilities
pub use state_vector_test_utils::StateVectorSimulator;

// Re-export density matrix testing utilities
pub use density_matrix_test_utils::DensityMatrixSimulator;
