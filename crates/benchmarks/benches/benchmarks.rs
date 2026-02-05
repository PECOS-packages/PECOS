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

use criterion::{Criterion, criterion_group, criterion_main};

mod modules {
    pub mod allocation_overhead;
    pub mod dem_sampler;
    pub mod dod_statevec;
    // TODO: pub mod hadamard_ops;
    #[cfg(feature = "gpu-sims")]
    pub mod gpu_influence_sampler;
    pub mod measurement_sampling;
    pub mod noise_models;
    // TODO: pub mod pauli_ops;
    pub mod rng;
    pub mod set_ops;
    pub mod sparse_state_vec;
    pub mod stabilizer_sims;
    pub mod state_vec_sims;
    pub mod surface_code;
    pub mod trig;
}

#[cfg(feature = "gpu-sims")]
use modules::gpu_influence_sampler;
use modules::{
    allocation_overhead, dem_sampler, dod_statevec, measurement_sampling, noise_models, rng,
    set_ops, sparse_state_vec, stabilizer_sims, state_vec_sims, surface_code, trig,
};

fn all_benchmarks(c: &mut Criterion) {
    allocation_overhead::benchmarks(c);
    dem_sampler::benchmarks(c);
    dod_statevec::benchmarks(c);
    #[cfg(feature = "gpu-sims")]
    gpu_influence_sampler::benchmarks(c);
    measurement_sampling::benchmarks(c);
    noise_models::benchmarks(c);
    rng::benchmarks(c);
    set_ops::benchmarks(c);
    sparse_state_vec::benchmarks(c);
    stabilizer_sims::benchmarks(c);
    state_vec_sims::benchmarks(c);
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
