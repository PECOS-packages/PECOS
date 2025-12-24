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
    pub mod element_ops;
    // TODO: pub mod hadamard_ops;
    pub mod measurement_sampling;
    pub mod noise_models;
    // TODO: pub mod pauli_ops;
    pub mod rng;
    pub mod set_ops;
    pub mod surface_code;
}

use modules::{element_ops, measurement_sampling, noise_models, rng, set_ops, surface_code};

fn all_benchmarks(c: &mut Criterion) {
    element_ops::benchmarks(c);
    measurement_sampling::benchmarks(c);
    noise_models::benchmarks(c);
    rng::benchmarks(c);
    set_ops::benchmarks(c);
    surface_code::benchmarks(c);
    // TODO: pauli_ops::benchmarks(c);
    // TODO: hadamard_ops::benchmarks(c);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(100).measurement_time(core::time::Duration::from_secs(10));
    targets = all_benchmarks
}
criterion_main!(benches);
