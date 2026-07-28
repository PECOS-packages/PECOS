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

// statistical calculations use count as f64
#![allow(clippy::cast_precision_loss)]
//! Noise Cookbook: Common patterns for the unified noise system.
//!
//! This example demonstrates:
//! - Using pre-built noise patterns for quick setup
//! - Building custom noise with the composite primitive system
//! - Configuring topology-aware crosstalk
//! - Creating realistic device noise models
//! - Integration with `sim_neo()` builder API
//!
//! Run with: cargo run --example `noise_cookbook`

use pecos_neo::noise::prelude::*;
use pecos_neo::prelude::*;
use pecos_neo::tool::{monte_carlo, sim_neo};
use pecos_simulators::SparseStab;

fn main() {
    println!("=== Noise Cookbook ===\n");

    recipe_quick_start();
    recipe_sim_neo_integration();
    recipe_measurement_noise();
    recipe_leakage_model();
    recipe_custom_composite();
    recipe_crosstalk();
    recipe_realistic_device();
}

/// Recipe 1: Quick Start with Pre-built Patterns
///
/// The easiest way to add noise - use ready-made configurations.
fn recipe_quick_start() {
    println!("--- Recipe 1: Quick Start ---");

    // Simple X gate circuit
    let commands = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

    // Method 1: One-liner with pre-built pattern
    let error_rate = run_with_noise(
        &commands,
        || depolarizing_with_measurement(0.01, 0.05, 0.02),
        1000,
    );
    println!(
        "  depolarizing_with_measurement: {:.1}% error rate",
        error_rate * 100.0
    );

    // Method 2: Using the builder
    let error_rate = run_with_noise(
        &commands,
        || {
            NoiseModelBuilder::new()
                .with_depolarizing(0.01, 0.05)
                .with_measurement_error(0.02)
                .build()
        },
        1000,
    );
    println!("  NoiseModelBuilder: {:.1}% error rate", error_rate * 100.0);

    println!();
}

/// Recipe 2: `sim_neo()` Integration
///
/// Use noise patterns with the high-level `sim_neo()` builder API.
fn recipe_sim_neo_integration() {
    println!("--- Recipe 2: sim_neo() Integration ---");

    let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

    // Method 1: Pass pre-built pattern directly to .noise()
    let results = sim_neo(circuit.clone())
        .auto()
        .noise(depolarizing_with_measurement(0.01, 0.05, 0.02))
        .sampling(monte_carlo(1000))
        .seed(42)
        .build()
        .run();

    let error_rate = results
        .outcomes
        .iter()
        .filter(|o| !o.get_bit(pecos_core::QubitId(0)).unwrap_or(true))
        .count() as f64
        / 1000.0;
    println!("  Pre-built pattern: {:.1}% error rate", error_rate * 100.0);

    // Method 2: Use the convenience .depolarizing() method
    let results = sim_neo(circuit.clone())
        .auto()
        .depolarizing(0.01)
        .sampling(monte_carlo(1000))
        .seed(42)
        .build()
        .run();

    let error_rate = results
        .outcomes
        .iter()
        .filter(|o| !o.get_bit(pecos_core::QubitId(0)).unwrap_or(true))
        .count() as f64
        / 1000.0;
    println!(
        "  .depolarizing() convenience: {:.1}% error rate",
        error_rate * 100.0
    );

    // Method 3: Pass NoiseModelBuilder (auto-converts via Into<ComposableNoiseModel>)
    let results = sim_neo(circuit.clone())
        .auto()
        .noise(
            NoiseModelBuilder::new()
                .with_depolarizing(0.01, 0.05)
                .with_measurement_error(0.02)
                .build(),
        )
        .sampling(monte_carlo(1000))
        .seed(42)
        .build()
        .run();

    let error_rate = results
        .outcomes
        .iter()
        .filter(|o| !o.get_bit(pecos_core::QubitId(0)).unwrap_or(true))
        .count() as f64
        / 1000.0;
    println!("  NoiseModelBuilder: {:.1}% error rate", error_rate * 100.0);

    // Method 4: Reusable simulation with noise
    let mut sim = sim_neo(circuit)
        .auto()
        .noise(realistic_device_noise(
            &DeviceNoiseParams::new()
                .with_p1(0.001)
                .with_p2(0.01)
                .with_measurement_error(0.02),
        ))
        .sampling(monte_carlo(500))
        .build();

    // Run multiple times with different seeds
    let results1 = sim.seed(100).run();
    let results2 = sim.seed(200).run();

    let err1 = results1
        .outcomes
        .iter()
        .filter(|o| !o.get_bit(pecos_core::QubitId(0)).unwrap_or(true))
        .count();
    let err2 = results2
        .outcomes
        .iter()
        .filter(|o| !o.get_bit(pecos_core::QubitId(0)).unwrap_or(true))
        .count();
    println!("  Reusable sim (seed 100): {err1} errors, (seed 200): {err2} errors");

    println!();
}

/// Recipe 3: Measurement Noise Variations
///
/// Different ways to model measurement errors.
fn recipe_measurement_noise() {
    println!("--- Recipe 3: Measurement Noise ---");

    let commands = CommandBuilder::new()
        .pz(&[0]) // Prepare |0>
        .mz(&[0])
        .build();

    // Symmetric measurement error (same rate for 0->1 and 1->0)
    let error_rate = run_with_noise(&commands, || measurement_only(0.05, 0.05), 1000);
    println!(
        "  Symmetric (5%/5%): {:.1}% flipped to 1",
        error_rate * 100.0
    );

    // Asymmetric: easier to flip 0->1 than 1->0
    let error_rate = run_with_noise(&commands, || measurement_only(0.10, 0.02), 1000);
    println!(
        "  Asymmetric (10%/2%): {:.1}% flipped to 1",
        error_rate * 100.0
    );

    // Using composite primitives for outcome-dependent noise
    let error_rate = run_with_noise(
        &commands,
        || {
            let meas_noise = seq![
                on_zero(prob(0.10, flip_outcome())), // 10% flip when measuring 0
                on_one(prob(0.02, flip_outcome())),  // 2% flip when measuring 1
            ];
            NoiseModelBuilder::new()
                .with_measurement_noise(meas_noise)
                .build()
        },
        1000,
    );
    println!("  Flow primitives: {:.1}% flipped to 1", error_rate * 100.0);

    println!();
}

/// Recipe 4: Leakage Model
///
/// Model qubits that leak outside the computational basis.
fn recipe_leakage_model() {
    println!("--- Recipe 4: Leakage Model ---");

    // Multiple gates to accumulate leakage effects
    let commands = CommandBuilder::new()
        .pz(&[0])
        .h(&[0])
        .h(&[0])
        .h(&[0])
        .h(&[0])
        .h(&[0]) // 5 Hadamard gates
        .mz(&[0])
        .build();

    // Using pre-built leakage pattern
    // 10% of errors cause leakage, 50% seepage rate
    let error_rate = run_with_noise(&commands, || with_leakage(0.1, 0.2, 0.1, 0.5), 1000);
    println!(
        "  with_leakage pattern: {:.1}% error rate",
        error_rate * 100.0
    );

    // Custom leakage with composite primitives
    let error_rate = run_with_noise(
        &commands,
        || {
            let sq_noise = seq![
                skip_if_leaked(), // Leaked qubits skip the gate
                prob(
                    0.1, // 10% fault probability
                    when_leaked(
                        seep(), // If leaked: seepage back to computational basis
                        sample![
                            (0.1, leak()),  // 10% of faults cause leakage
                            (0.9, pauli()), // 90% are Pauli errors
                        ]
                    )
                ),
            ];

            NoiseModelBuilder::new()
                .with_single_qubit_noise(sq_noise)
                .build()
        },
        1000,
    );
    println!(
        "  Custom composite primitives: {:.1}% error rate",
        error_rate * 100.0
    );

    println!();
}

/// Recipe 5: Custom Composite Primitives
///
/// Build exactly the noise model you need with composite primitives.
fn recipe_custom_composite() {
    println!("--- Recipe 5: Custom Composite Primitives ---");

    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    // Custom two-qubit noise: biased towards Z errors (dephasing)
    let error_rate = run_with_noise_2q(
        &commands,
        || {
            let tq_noise = seq![
                skip_if_leaked(),
                prob(
                    0.05, // 5% error rate
                    sample![
                        (0.5, inject_z()), // 50% Z errors (dephasing)
                        (0.3, inject_x()), // 30% X errors (bit flip)
                        (0.2, inject_y()), // 20% Y errors
                    ]
                ),
            ];

            NoiseModelBuilder::new()
                .with_two_qubit_noise(tq_noise)
                .build()
        },
        1000,
    );
    println!(
        "  Z-biased two-qubit noise: {:.1}% error rate",
        error_rate * 100.0
    );

    // Pure dephasing model (Z errors only)
    let error_rate = run_with_noise_2q(&commands, || dephasing_only(0.01, 0.05), 1000);
    println!("  Pure dephasing: {:.1}% error rate", error_rate * 100.0);

    println!();
}

/// Recipe 6: Topology-Aware Crosstalk
///
/// Model errors that spread to neighboring qubits.
fn recipe_crosstalk() {
    println!("--- Recipe 6: Crosstalk ---");

    // 3-qubit chain: measure middle qubit, observe neighbors
    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .pz(&[2])
        .mz(&[1]) // Measure middle qubit
        .mz(&[0]) // Check if neighbors were affected
        .mz(&[2])
        .build();

    // Chain crosstalk: measuring qubit 1 can flip qubits 0 and 2
    let mut neighbor_errors = 0;
    for seed in 0..1000 {
        let crosstalk = CompositeCrosstalkChannel::new("chain_xt", prob(0.1, inject_x()))
            .responds_to_measurement()
            .local(chain_neighbors);

        let noise = ComposableNoiseModel::new().add_channel(crosstalk);

        let mut state = SparseStab::new(3);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(seed);
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        // Count errors on neighbor qubits (should be 0 without crosstalk)
        if outcomes
            .get(pecos_core::QubitId(0))
            .is_some_and(|o| o.outcome)
        {
            neighbor_errors += 1;
        }
        if outcomes
            .get(pecos_core::QubitId(2))
            .is_some_and(|o| o.outcome)
        {
            neighbor_errors += 1;
        }
    }

    println!(
        "  Chain crosstalk: {:.1}% neighbor errors (2000 opportunities)",
        f64::from(neighbor_errors) / 20.0
    );

    // Demonstrate topology helpers
    println!("\n  Topology helpers available:");
    println!("    chain_neighbors - 1D: qubit i has neighbors i-1, i+1");
    println!("    grid_neighbors(cols) - 2D grid with given columns");
    println!("    chain_distance(a, b) - distance on 1D chain");
    println!("    grid_distance(cols)(a, b) - Manhattan distance on grid");

    println!();
}

/// Recipe 7: Realistic Device Noise
///
/// Comprehensive noise model for real quantum hardware.
fn recipe_realistic_device() {
    println!("--- Recipe 7: Realistic Device Noise ---");

    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .h(&[0])
        .cx(&[(0, 1)]) // Create Bell state
        .mz(&[0])
        .mz(&[1])
        .build();

    // Full device noise with all parameters
    let same_outcome = count_bell_correlation(
        &commands,
        || {
            realistic_device_noise(
                &DeviceNoiseParams::new()
                    .with_p1(0.001) // 0.1% single-qubit gate error
                    .with_p2(0.01) // 1% two-qubit gate error
                    .with_measurement_error(0.02) // 2% readout error
                    .with_prep_error(0.001), // 0.1% preparation error
            )
        },
        1000,
    );

    println!(
        "  Realistic device: {:.1}% same outcome (expect ~95-99% with noise)",
        same_outcome as f64 / 10.0
    );

    // Surface code optimized noise
    let same_outcome = count_bell_correlation(&commands, || surface_code_noise(0.001, false), 1000);
    println!(
        "  Surface code noise (p=0.1%): {:.1}% same outcome",
        same_outcome as f64 / 10.0
    );

    println!();
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Run circuit with noise and return error rate (fraction of 1 outcomes on qubit 0).
fn run_with_noise<F>(commands: &CommandQueue, make_noise: F, shots: u64) -> f64
where
    F: Fn() -> ComposableNoiseModel,
{
    let mut ones = 0;
    for seed in 0..shots {
        let noise = make_noise();
        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(seed);
        let outcomes = runner.apply_circuit(&mut state, commands).unwrap();
        if outcomes
            .get(pecos_core::QubitId(0))
            .is_some_and(|o| o.outcome)
        {
            ones += 1;
        }
    }
    f64::from(ones) / shots as f64
}

/// Run 2-qubit circuit and return error rate (either qubit has wrong outcome).
fn run_with_noise_2q<F>(commands: &CommandQueue, make_noise: F, shots: u64) -> f64
where
    F: Fn() -> ComposableNoiseModel,
{
    let mut errors = 0;
    for seed in 0..shots {
        let noise = make_noise();
        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(seed);
        let outcomes = runner.apply_circuit(&mut state, commands).unwrap();
        // For a CX on |00>, expect |00>
        let o0 = outcomes.get(pecos_core::QubitId(0)).map(|o| o.outcome);
        let o1 = outcomes.get(pecos_core::QubitId(1)).map(|o| o.outcome);
        if o0 == Some(true) || o1 == Some(true) {
            errors += 1;
        }
    }
    f64::from(errors) / shots as f64
}

/// Count how many times Bell state qubits have same outcome.
fn count_bell_correlation<F>(commands: &CommandQueue, make_noise: F, shots: u64) -> u64
where
    F: Fn() -> ComposableNoiseModel,
{
    let mut same = 0;
    for seed in 0..shots {
        let noise = make_noise();
        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(seed);
        let outcomes = runner.apply_circuit(&mut state, commands).unwrap();
        let o0 = outcomes.get(pecos_core::QubitId(0)).map(|o| o.outcome);
        let o1 = outcomes.get(pecos_core::QubitId(1)).map(|o| o.outcome);
        if o0 == o1 {
            same += 1;
        }
    }
    same
}
