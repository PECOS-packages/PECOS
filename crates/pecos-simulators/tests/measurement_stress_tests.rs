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

//! Measurement stress tests for all rotation-capable simulators.

use pecos_simulators::{
    DensityMatrix, SparseStateVecAoS, SparseStateVecSoA, StabVec, StateVecAoS, StateVecSoA,
    StateVecSoA32,
};

// State vector simulators
pecos_simulators::measurement_stress_test_suite!(StateVecSoA, 4, StateVecSoA::with_seed(4, 42));
pecos_simulators::measurement_stress_test_suite!(StateVecSoA32, 4, StateVecSoA32::with_seed(4, 42));
pecos_simulators::measurement_stress_test_suite!(StateVecAoS, 4, StateVecAoS::with_seed(4, 42));
pecos_simulators::measurement_stress_test_suite!(
    SparseStateVecAoS,
    4,
    SparseStateVecAoS::with_seed(4, 42)
);
pecos_simulators::measurement_stress_test_suite!(
    SparseStateVecSoA,
    4,
    SparseStateVecSoA::with_seed(4, 42)
);

// Density matrix
pecos_simulators::measurement_stress_test_suite!(DensityMatrix, 4, DensityMatrix::with_seed(4, 42));

// Clifford+Rz (exact mode)
pecos_simulators::measurement_stress_test_suite!(
    StabVec,
    4,
    StabVec::builder(4).seed(42).pruning_threshold(0.0).build()
);
