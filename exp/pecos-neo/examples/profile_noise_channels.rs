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

//! Profiling binary for noise channel overhead analysis.

use pecos_core::QubitId;
use pecos_neo::GateType;
use pecos_neo::noise::{ComposableNoiseModel, NoiseEvent, SingleQubitChannel};
use pecos_neo::prelude::CorePlugin;
use pecos_random::PecosRng;
use std::hint::black_box;

fn main() {
    let iterations = 1_000_000;
    let qubit = QubitId(0);
    let qubits = [qubit];
    let angles = [];

    println!("Profiling noise channel overhead ({iterations} iterations each)...\n");

    // Baseline: just RNG sampling
    let mut rng = PecosRng::seed_from_u64(42);
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let sample: f64 = rand::RngExt::random(&mut rng);
        black_box(sample);
    }
    let rng_time = start.elapsed();
    println!(
        "RNG sampling only: {:?} ({:.1} ns/iter)",
        rng_time,
        rng_time.as_nanos() as f64 / f64::from(iterations)
    );

    // Single channel with event check
    let mut noise = ComposableNoiseModel::new()
        .add_plugin(CorePlugin)
        .add_channel(SingleQubitChannel::depolarizing(0.001));
    let mut rng = PecosRng::seed_from_u64(42);

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };
        let response = noise.emit(event, &mut rng);
        black_box(response);
    }
    let single_channel_time = start.elapsed();
    println!(
        "Single channel emit: {:?} ({:.1} ns/iter)",
        single_channel_time,
        single_channel_time.as_nanos() as f64 / f64::from(iterations)
    );

    // Multiple channels
    let mut noise = ComposableNoiseModel::new()
        .add_plugin(CorePlugin)
        .add_channel(SingleQubitChannel::depolarizing(0.001))
        .add_channel(SingleQubitChannel::depolarizing(0.001))
        .add_channel(SingleQubitChannel::depolarizing(0.001));
    let mut rng = PecosRng::seed_from_u64(42);

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };
        let response = noise.emit(event, &mut rng);
        black_box(response);
    }
    let multi_channel_time = start.elapsed();
    println!(
        "Three channels emit: {:?} ({:.1} ns/iter)",
        multi_channel_time,
        multi_channel_time.as_nanos() as f64 / f64::from(iterations)
    );

    // Noise model creation overhead
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let noise = ComposableNoiseModel::new()
            .add_plugin(CorePlugin)
            .add_channel(SingleQubitChannel::depolarizing(0.001));
        black_box(noise);
    }
    let creation_time = start.elapsed();
    println!(
        "Noise model creation: {:?} ({:.1} ns/iter)",
        creation_time,
        creation_time.as_nanos() as f64 / f64::from(iterations)
    );

    println!("\nDone!");
}
