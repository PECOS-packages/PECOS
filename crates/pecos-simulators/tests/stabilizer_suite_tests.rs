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

//! Comprehensive test suite for all stabilizer simulator implementations.
//!
//! This file uses the `stabilizer_test_suite!` macro to generate standardized
//! tests for all types implementing `StabilizerSimulator`.

use pecos_simulators::{
    CHForm, DenseStab, DenseStabColOnly, DenseStabRowOnly, GpuStab, GpuStabOpt, GpuStabParallel,
    SparseColOnly, SparseStab, SparseStabHybrid, SparseStabUnsortedVecSet, SparseStabVecSet,
    Stabilizer,
};

// Generate test suites for all stabilizer simulator implementations

// Sparse stabilizer simulators
pecos_simulators::stabilizer_test_suite!(SparseStab);
pecos_simulators::stabilizer_test_suite!(SparseStabVecSet);
pecos_simulators::stabilizer_test_suite!(SparseStabUnsortedVecSet);
pecos_simulators::stabilizer_test_suite!(SparseStabHybrid);

// Dense stabilizer simulators
pecos_simulators::stabilizer_test_suite!(DenseStab);
pecos_simulators::stabilizer_test_suite!(DenseStabColOnly);
pecos_simulators::stabilizer_test_suite!(DenseStabRowOnly);
pecos_simulators::stabilizer_test_suite!(SparseColOnly);

// Default wrapper
pecos_simulators::stabilizer_test_suite!(Stabilizer);

// CH-form stabilizer simulator
pecos_simulators::stabilizer_test_suite!(CHForm);

// GPU-optimized stabilizer simulators
pecos_simulators::stabilizer_test_suite!(GpuStab);
pecos_simulators::stabilizer_test_suite!(GpuStabOpt);
pecos_simulators::stabilizer_test_suite!(GpuStabParallel);
