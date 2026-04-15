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

// Benchmark code casts qubit indices to i32/u32 for FFI and GPU APIs
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use criterion::{Criterion, criterion_group, criterion_main};

mod modules {
    pub mod allocation_overhead;
    pub mod clifford_rz;
    pub mod cpu_stabilizer_comparison;
    pub mod dem_builder;
    pub mod dem_sampler;
    pub mod dod_statevec;
    pub mod quizx_eval;
    // TODO: pub mod hadamard_ops;
    #[cfg(feature = "cuquantum")]
    pub mod cuquantum;
    #[cfg(feature = "gpu-sims")]
    pub mod gpu_influence_sampler;
    pub mod measurement_sampling;
    pub mod native_statevec_comparison;
    pub mod noise_models;
    #[cfg(feature = "cppsparsestab")]
    pub mod sparse_stab_vs_cpp;
    pub mod sparse_stab_w_vs_y;
    // TODO: pub mod pauli_ops;
    pub mod pecos_neo_comparison;
    pub mod rng;
    pub mod set_ops;
    pub mod sparse_state_vec;
    pub mod stabilizer_sims;
    pub mod state_vec_sims;
    pub mod surface_code;
    pub mod trig;
}

#[cfg(feature = "cuquantum")]
use modules::cuquantum;
#[cfg(feature = "gpu-sims")]
use modules::gpu_influence_sampler;
#[cfg(feature = "cppsparsestab")]
use modules::sparse_stab_vs_cpp;
use modules::{
    allocation_overhead, clifford_rz, cpu_stabilizer_comparison, dem_builder, dem_sampler,
    dod_statevec, measurement_sampling, native_statevec_comparison, noise_models,
    pecos_neo_comparison, quizx_eval, rng, set_ops, sparse_stab_w_vs_y, sparse_state_vec,
    stabilizer_sims, state_vec_sims, surface_code, trig,
};

fn all_benchmarks(c: &mut Criterion) {
    allocation_overhead::benchmarks(c);
    clifford_rz::benchmarks(c);
    cpu_stabilizer_comparison::benchmarks(c);
    quizx_eval::benchmarks(c);
    #[cfg(feature = "cuquantum")]
    cuquantum::benchmarks(c);
    dem_builder::benchmarks(c);
    dem_sampler::benchmarks(c);
    dod_statevec::benchmarks(c);
    #[cfg(feature = "gpu-sims")]
    gpu_influence_sampler::benchmarks(c);
    measurement_sampling::benchmarks(c);
    native_statevec_comparison::benchmarks(c);
    noise_models::benchmarks(c);
    pecos_neo_comparison::benchmarks(c);
    rng::benchmarks(c);
    set_ops::benchmarks(c);
    sparse_state_vec::benchmarks(c);
    stabilizer_sims::benchmarks(c);
    state_vec_sims::benchmarks(c);
    #[cfg(feature = "cppsparsestab")]
    sparse_stab_vs_cpp::benchmarks(c);
    sparse_stab_w_vs_y::benchmarks(c);
    surface_code::benchmarks(c);
    trig::benchmarks(c);
    // TODO: pauli_ops::benchmarks(c);
    // TODO: hadamard_ops::benchmarks(c);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(100).measurement_time(core::time::Duration::from_secs(10));
    targets = all_benchmarks
}
criterion_main!(benches);
