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

//! Noise model configuration examples for pecos-neo.
//!
//! This example demonstrates:
//! - Composing noise channels directly
//! - Using the `GeneralNoiseModelBuilder`
//! - Configuring different noise types
//! - Measuring noise effects on circuits
//!
//! Run with: cargo run --example `noise_models`

use pecos_core::TimeScale;
use pecos_neo::noise::GeneralNoiseModelBuilder;
use pecos_neo::prelude::*;
use pecos_simulators::SparseStab;
use std::collections::HashMap;

fn main() {
    println!("=== pecos-neo Noise Model Examples ===\n");

    example_depolarizing_noise();
    example_asymmetric_measurement();
    example_multi_channel();
    example_builder_api();
    example_idle_noise();
    example_z_biased_noise();
}

/// Simple depolarizing noise on single-qubit gates
fn example_depolarizing_noise() {
    println!("--- Depolarizing Noise ---");

    // Circuit: |0⟩ → X → measure (should give |1⟩)
    let commands = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

    // Test different error rates
    for error_rate in [0.0, 0.05, 0.10, 0.20] {
        let noise = ComposableNoiseModel::new()
            .add_plugin(CorePlugin)
            .add_channel(SingleQubitChannel::depolarizing(error_rate));

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);

        let mut errors = 0;
        let shots = 1000;

        for _ in 0..shots {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            // X gate should give |1⟩, so |0⟩ is an error
            if !outcomes.get_bit(QubitId(0)).unwrap_or(true) {
                errors += 1;
            }
        }

        let measured_rate = f64::from(errors) / f64::from(shots);
        println!(
            "  p={:.0}%: measured error rate = {:.1}%",
            error_rate * 100.0,
            measured_rate * 100.0
        );
    }
    println!();
}

/// Asymmetric measurement errors (different 0→1 and 1→0 rates)
fn example_asymmetric_measurement() {
    println!("--- Asymmetric Measurement Errors ---");

    // Test measurement of |0⟩
    let commands_0 = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

    // Test measurement of |1⟩
    let commands_1 = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

    // Asymmetric: 10% chance of 0→1, 5% chance of 1→0
    let p_0_to_1 = 0.10;
    let p_1_to_0 = 0.05;

    let shots = 2000;

    // Measure |0⟩ state
    let noise_0 = ComposableNoiseModel::new()
        .add_plugin(CorePlugin)
        .add_channel(MeasurementChannel::asymmetric(p_0_to_1, p_1_to_0));

    let mut state = SparseStab::new(1);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise_0)
        .with_seed(42);

    let mut flips_0 = 0;
    for _ in 0..shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands_0).unwrap();
        if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            flips_0 += 1; // Got |1⟩ when should be |0⟩
        }
    }

    // Measure |1⟩ state
    let noise_1 = ComposableNoiseModel::new()
        .add_plugin(CorePlugin)
        .add_channel(MeasurementChannel::asymmetric(p_0_to_1, p_1_to_0));

    let mut state = SparseStab::new(1);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise_1)
        .with_seed(43);

    let mut flips_1 = 0;
    for _ in 0..shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands_1).unwrap();
        if !outcomes.get_bit(QubitId(0)).unwrap_or(true) {
            flips_1 += 1; // Got |0⟩ when should be |1⟩
        }
    }

    println!(
        "  Configured: p(0→1)={:.0}%, p(1→0)={:.0}%",
        p_0_to_1 * 100.0,
        p_1_to_0 * 100.0
    );
    println!(
        "  Measured |0⟩: {:.1}% flipped to |1⟩",
        f64::from(flips_0) / f64::from(shots) * 100.0
    );
    println!(
        "  Measured |1⟩: {:.1}% flipped to |0⟩",
        f64::from(flips_1) / f64::from(shots) * 100.0
    );
    println!();
}

/// Multiple noise channels combined
fn example_multi_channel() {
    println!("--- Multi-Channel Noise ---");

    // Bell state circuit
    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .h(&[0])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    // Combine multiple noise sources
    let noise = ComposableNoiseModel::new()
        .add_plugin(CorePlugin)
        .add_channel(PreparationChannel::new(0.02)) // 2% prep error
        .add_channel(SingleQubitChannel::depolarizing(0.01)) // 1% 1Q error
        .add_channel(TwoQubitChannel::depolarizing(0.05)) // 5% 2Q error
        .add_channel(MeasurementChannel::symmetric(0.02)); // 2% meas error

    let mut state = SparseStab::new(2);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise)
        .with_seed(42);

    let mut counts: HashMap<String, usize> = HashMap::new();
    let shots = 2000;

    for _ in 0..shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        let q0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
        let q1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);
        let key = format!("{}{}", u8::from(q0), u8::from(q1));
        *counts.entry(key).or_insert(0) += 1;
    }

    println!("  Bell state with multi-channel noise ({shots} shots):");
    let correlated = counts.get("00").unwrap_or(&0) + counts.get("11").unwrap_or(&0);
    let anti_correlated = counts.get("01").unwrap_or(&0) + counts.get("10").unwrap_or(&0);
    println!(
        "    Correlated (|00⟩ + |11⟩): {:.1}%",
        correlated as f64 / f64::from(shots) * 100.0
    );
    println!(
        "    Anti-correlated (|01⟩ + |10⟩): {:.1}%",
        anti_correlated as f64 / f64::from(shots) * 100.0
    );
    println!();
}

/// Using the `GeneralNoiseModelBuilder` for familiar API
fn example_builder_api() {
    println!("--- GeneralNoiseModelBuilder API ---");

    // Bell state circuit
    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .h(&[0])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    // Build noise model with familiar parameters
    let noise = GeneralNoiseModelBuilder::new()
        .with_p_prep(0.01) // 1% preparation error
        .with_p1(0.005) // 0.5% single-qubit error
        .with_p2(0.02) // 2% two-qubit error
        .with_p_meas(0.01, 0.01) // 1% symmetric measurement error
        .build();

    let mut state = SparseStab::new(2);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise)
        .with_seed(42);

    let mut counts: HashMap<String, usize> = HashMap::new();
    let shots = 2000;

    for _ in 0..shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        let q0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
        let q1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);
        let key = format!("{}{}", u8::from(q0), u8::from(q1));
        *counts.entry(key).or_insert(0) += 1;
    }

    println!("  Bell state with builder-configured noise ({shots} shots):");
    for (outcome, count) in &counts {
        println!(
            "    |{}⟩: {:.1}%",
            outcome,
            *count as f64 / f64::from(shots) * 100.0
        );
    }
    println!();
}

/// Idle noise with T1/T2 using `TimeScale`
fn example_idle_noise() {
    println!("--- Idle Noise (T1/T2 Decoherence) ---");

    // Circuit: prep |+⟩, idle, then H to detect Z errors, measure
    // Z errors during idle will flip the measurement outcome
    let commands = CommandBuilder::new()
        .pz(&[0])
        .h(&[0]) // Prepare |+⟩
        .idle(&[0], 1000) // Idle for 1000 time units
        .h(&[0]) // Convert Z errors to bit flips
        .mz(&[0])
        .build();

    // Configure with nanosecond time scale
    // T1 = 10us, T2 = 5us (typical superconducting qubit)
    let noise = GeneralNoiseModelBuilder::new()
        .with_time_scale(TimeScale::NANOSECONDS)
        .with_idle_t1_t2(10e-6, 5e-6) // T1=10us, T2=5us
        .build();

    let mut state = SparseStab::new(1);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise)
        .with_seed(42);

    let shots = 5000;
    let mut errors = 0;

    for _ in 0..shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            errors += 1; // Z error during idle caused bit flip
        }
    }

    let error_rate = f64::from(errors) / f64::from(shots);
    println!("  T1=10us, T2=5us, idle=1us (1000 time units)");
    println!(
        "  Measured decoherence error rate: {:.1}%",
        error_rate * 100.0
    );
    println!("  (Linear/T1 contribution expected: ~10%)");
    println!();
}

/// Z-biased noise (common in superconducting qubits)
fn example_z_biased_noise() {
    println!("--- Z-Biased Noise ---");

    // Circuit: H to create superposition, then gate, then H+measure
    // This makes Z errors detectable
    let commands = CommandBuilder::new()
        .pz(&[0])
        .h(&[0])
        .sz(&[0]) // Apply gate with noise
        .h(&[0])
        .mz(&[0])
        .build();

    // Test uniform vs Z-biased noise
    let shots = 5000;

    // Uniform depolarizing
    let uniform_noise = ComposableNoiseModel::new()
        .add_plugin(CorePlugin)
        .add_channel(SingleQubitChannel::depolarizing(0.10));

    let mut state = SparseStab::new(1);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(uniform_noise)
        .with_seed(42);

    let mut uniform_errors = 0;
    for _ in 0..shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        // S gate on |+⟩ gives |+i⟩, after H should give specific outcome
        // Errors will deviate from this
        if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            uniform_errors += 1;
        }
    }

    // Z-biased (90% Z, 5% X, 5% Y)
    let z_biased_channel =
        SingleQubitChannel::depolarizing(0.10).with_pauli_weights(PauliWeights::z_biased(0.9));

    let z_biased_noise = ComposableNoiseModel::new()
        .add_plugin(CorePlugin)
        .add_channel(z_biased_channel);

    let mut state = SparseStab::new(1);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(z_biased_noise)
        .with_seed(42);

    let mut z_biased_errors = 0;
    for _ in 0..shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            z_biased_errors += 1;
        }
    }

    println!("  10% error rate, {shots} shots:");
    println!(
        "    Uniform depolarizing: {:.1}% |1⟩",
        f64::from(uniform_errors) / f64::from(shots) * 100.0
    );
    println!(
        "    Z-biased (90% Z): {:.1}% |1⟩",
        f64::from(z_biased_errors) / f64::from(shots) * 100.0
    );
    println!("  (Z-biased has higher detectable error rate in X basis)");
    println!();
}
