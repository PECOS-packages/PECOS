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
pub use pecos_core::VecSet;

// re-exporting pecos-qsims
pub use pecos_qsims::CliffordSimulator;
pub use pecos_qsims::SparseStab;
// TODO: add the following in the future as makes sense...
// pub use pecos_qsims::clifford_simulator::CliffordSimulator;
// pub use pecos_qsims::gens::Gens;
// pub use pecos_qsims::measurement::{MeasBitValue, MeasValue, Measurement}; // TODO: Distinguish between trait and struct/enum
// pub use pecos_qsims::nonclifford_simulator::NonCliffordSimulator;
// pub use pecos_qsims::pauli_prop::{PauliProp, StdPauliProp};
// pub use pecos_qsims::paulis::Paulis;
// pub use pecos_qsims::quantum_simulator::QuantumSimulator;
// pub use pecos_qsims::sparse_stab::SparseStab;
