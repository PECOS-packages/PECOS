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

//! Test suite for all stabilizer simulator implementations.
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
pecos_simulators::stabilizer_test_suite!(sparse_stab, SparseStab);
pecos_simulators::stabilizer_test_suite!(sparse_stab_vec_set, SparseStabVecSet);
pecos_simulators::stabilizer_test_suite!(sparse_stab_unsorted_vec_set, SparseStabUnsortedVecSet);
pecos_simulators::stabilizer_test_suite!(sparse_stab_hybrid, SparseStabHybrid);

// Dense stabilizer simulators
pecos_simulators::stabilizer_test_suite!(dense_stab, DenseStab);
pecos_simulators::stabilizer_test_suite!(dense_stab_col_only, DenseStabColOnly);
pecos_simulators::stabilizer_test_suite!(dense_stab_row_only, DenseStabRowOnly);
pecos_simulators::stabilizer_test_suite!(sparse_col_only, SparseColOnly);

// Default wrapper
pecos_simulators::stabilizer_test_suite!(stabilizer, Stabilizer);

// CH-form stabilizer simulator
pecos_simulators::stabilizer_test_suite!(ch_form, CHForm);

// GPU-optimized stabilizer simulators
pecos_simulators::stabilizer_test_suite!(gpu_stab, GpuStab);
pecos_simulators::stabilizer_test_suite!(gpu_stab_opt, GpuStabOpt);
pecos_simulators::stabilizer_test_suite!(gpu_stab_parallel, GpuStabParallel);
