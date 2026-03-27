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

//! Profiling binary for pecos-neo hot paths.
//!
//! Run with: `samply record cargo run --release --example profile_hotpath -p pecos-neo`

use pecos_neo::prelude::{
    CircuitRunner, CommandBuilder, ComposableNoiseModel, CorePlugin, SingleQubitChannel,
    TwoQubitChannel,
};
use pecos_simulators::SparseStab;
use std::hint::black_box;

fn main() {
    let iterations = 100_000;

    // Build commands once
    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .h(&[0])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    println!("Running {iterations} iterations of shot execution with noise...");

    for _ in 0..iterations {
        let noise = ComposableNoiseModel::new()
            .add_plugin(CorePlugin)
            .add_channel(SingleQubitChannel::depolarizing(0.001))
            .add_channel(TwoQubitChannel::depolarizing(0.001));
        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);
        state.reset();
        let result = runner.apply_circuit(&mut state, &commands).unwrap();
        black_box(result);
    }

    println!("Done!");
}
