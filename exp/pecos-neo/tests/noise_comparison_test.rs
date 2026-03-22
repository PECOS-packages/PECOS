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

//! Comparison tests between `ComposableNoiseModel` and `GeneralNoiseModel`.
//!
//! These tests verify that the ECS-inspired noise system produces statistically
//! similar results to the existing `GeneralNoiseModel` implementation.

use pecos_core::QubitId;
use pecos_engines::byte_message::ByteMessageBuilder;
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{Engine, QuantumSystem};
use pecos_neo::prelude::*;
use pecos_simulators::SparseStab;
use std::collections::BTreeMap;

const NUM_SHOTS: usize = 5000;
const TOLERANCE_PERCENT: f64 = 5.0; // Allow 5% difference in error rates

/// Run a circuit with `GeneralNoiseModel` and count results.
fn run_general_noise_model(
    noise_model: GeneralNoiseModel,
    circuit_builder: impl Fn(&mut ByteMessageBuilder),
    num_qubits: usize,
    num_shots: usize,
) -> BTreeMap<String, usize> {
    let quantum = Box::new(StateVecEngine::new(num_qubits));
    let mut system = QuantumSystem::new(Box::new(noise_model), quantum);
    system.set_seed(42);

    let mut counts = BTreeMap::new();

    for _ in 0..num_shots {
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        circuit_builder(&mut builder);
        let circ = builder.build();

        system.reset().expect("Failed to reset system");
        let output = system.process(circ).expect("Processing failed");

        let mut result = String::new();
        if let Ok(outcomes) = output.outcomes().map(|outcomes| {
            outcomes
                .into_iter()
                .enumerate()
                .collect::<Vec<(usize, u32)>>()
        }) {
            let mut bits = vec!['0'; num_qubits];
            for (result_id, outcome) in outcomes {
                if result_id < num_qubits {
                    bits[result_id] = if outcome != 0 { '1' } else { '0' };
                }
            }
            result = bits.into_iter().collect();
        }

        if result.is_empty() {
            result = "0".repeat(num_qubits);
        }

        *counts.entry(result).or_insert(0) += 1;
    }

    counts
}

/// Run a circuit with `ComposableNoiseModel` and count results.
fn run_composable_noise_model(
    noise_model: ComposableNoiseModel,
    commands: CommandQueue,
    num_qubits: usize,
    num_shots: usize,
) -> BTreeMap<String, usize> {
    let mut state = SparseStab::new(num_qubits);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise_model)
        .with_seed(42);

    // Create qubit IDs for bitstring extraction
    let qubits: Vec<QubitId> = (0..num_qubits).map(QubitId).collect();

    let mut counts = BTreeMap::new();

    for _ in 0..num_shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        // Convert outcomes to bitstring
        let result = if let Some(bits) = outcomes.bitstring(&qubits) {
            bits.iter()
                .map(|&b| if b { '1' } else { '0' })
                .collect::<String>()
        } else {
            "0".repeat(num_qubits)
        };

        *counts.entry(result).or_insert(0) += 1;
    }

    counts
}

/// Calculate the percentage of a specific outcome.
fn outcome_percentage(counts: &BTreeMap<String, usize>, outcome: &str, total: usize) -> f64 {
    let count = *counts.get(outcome).unwrap_or(&0);
    (count as f64 / total as f64) * 100.0
}

/// Compare two error rates and check if they're within tolerance.
fn rates_match(rate1: f64, rate2: f64, tolerance: f64) -> bool {
    (rate1 - rate2).abs() <= tolerance
}

#[test]
fn test_single_qubit_depolarizing_comparison() {
    // Test that single-qubit depolarizing noise produces similar error rates
    // in both noise model implementations.
    //
    // Circuit: |0⟩ → X → measure
    // Expected (no noise): 100% |1⟩
    // With 30% depolarizing: ~30% chance of error on X gate

    let p1 = 0.30; // 30% error rate on single-qubit gates

    // Note: GeneralNoiseModel uses "average" probability which scales by 3/2
    // So to get p1=0.30, we set average_p1 = 0.30 / 1.5 = 0.20
    let average_p1 = p1 / 1.5;

    // GeneralNoiseModel setup
    let general_model = GeneralNoiseModel::builder()
        .with_prep_probability(0.0)
        .with_meas_0_probability(0.0)
        .with_meas_1_probability(0.0)
        .with_average_p1_probability(average_p1)
        .with_average_p2_probability(0.0)
        .with_p1_emission_ratio(0.0) // No leakage
        .with_seed(42)
        .build();

    // ComposableNoiseModel setup
    let composable_model =
        ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(p1));

    // Circuit for GeneralNoiseModel
    let general_counts = run_general_noise_model(
        general_model,
        |builder| {
            builder.add_x(&[0]);
            builder.add_measurements(&[0]);
        },
        1,
        NUM_SHOTS,
    );

    // Circuit for ComposableNoiseModel
    let commands = CommandBuilder::new().pz(0).x(0).mz(0).build();

    let composable_counts = run_composable_noise_model(composable_model, commands, 1, NUM_SHOTS);

    // Calculate |0⟩ percentages (errors, since X gate should give |1⟩)
    let general_zero = outcome_percentage(&general_counts, "0", NUM_SHOTS);
    let composable_zero = outcome_percentage(&composable_counts, "0", NUM_SHOTS);

    println!("Single-qubit depolarizing comparison (p1={p1}):");
    println!("  GeneralNoiseModel: {general_zero:.1}% |0⟩ (errors)");
    println!("  ComposableNoiseModel: {composable_zero:.1}% |0⟩ (errors)");

    // Both should have similar error rates
    // With uniform depolarizing, X error cancels X gate (gives |0⟩)
    // Y error: Y·X = iZ, still gives |0⟩ for measurement
    // Z error: Z·X = -X, gives |1⟩
    // So ~2/3 of errors result in |0⟩, ~1/3 result in |1⟩

    assert!(
        rates_match(general_zero, composable_zero, TOLERANCE_PERCENT),
        "Error rates should match within {TOLERANCE_PERCENT}%: general={general_zero:.1}%, composable={composable_zero:.1}%"
    );
}

#[test]
fn test_two_qubit_depolarizing_comparison() {
    // Test that two-qubit depolarizing noise produces similar error rates.
    //
    // Circuit: |00⟩ → X(0) → CX(0,1) → measure
    // Expected (no noise): 100% |11⟩
    // With 20% depolarizing on CX: some errors

    let p2 = 0.20; // 20% error rate on two-qubit gates

    // GeneralNoiseModel uses average probability which scales by 5/4
    let average_p2 = p2 / 1.25;

    // GeneralNoiseModel setup
    let general_model = GeneralNoiseModel::builder()
        .with_prep_probability(0.0)
        .with_meas_0_probability(0.0)
        .with_meas_1_probability(0.0)
        .with_average_p1_probability(0.0)
        .with_average_p2_probability(average_p2)
        .with_p2_emission_ratio(0.0) // No leakage
        .with_seed(42)
        .build();

    // ComposableNoiseModel setup
    let composable_model =
        ComposableNoiseModel::new().add_channel(TwoQubitChannel::depolarizing(p2));

    // Circuit for GeneralNoiseModel
    let general_counts = run_general_noise_model(
        general_model,
        |builder| {
            builder.add_x(&[0]);
            builder.add_cx(&[0], &[1]);
            builder.add_measurements(&[0, 1]);
        },
        2,
        NUM_SHOTS,
    );

    // Circuit for ComposableNoiseModel
    let commands = CommandBuilder::new()
        .pz(0)
        .pz(1)
        .x(0)
        .cx(0, 1)
        .mz(0)
        .mz(1)
        .build();

    let composable_counts = run_composable_noise_model(composable_model, commands, 2, NUM_SHOTS);

    // Calculate |11⟩ percentages (correct outcome)
    let general_correct = outcome_percentage(&general_counts, "11", NUM_SHOTS);
    let composable_correct = outcome_percentage(&composable_counts, "11", NUM_SHOTS);

    println!("Two-qubit depolarizing comparison (p2={p2}):");
    println!("  GeneralNoiseModel: {general_correct:.1}% |11⟩ (correct)");
    println!("  ComposableNoiseModel: {composable_correct:.1}% |11⟩ (correct)");

    // Error rate = 1 - correct rate
    let general_error = 100.0 - general_correct;
    let composable_error = 100.0 - composable_correct;

    println!("  GeneralNoiseModel: {general_error:.1}% errors");
    println!("  ComposableNoiseModel: {composable_error:.1}% errors");

    assert!(
        rates_match(general_error, composable_error, TOLERANCE_PERCENT),
        "Error rates should match within {TOLERANCE_PERCENT}%: general={general_error:.1}%, composable={composable_error:.1}%"
    );
}

#[test]
fn test_measurement_error_comparison() {
    // Test that measurement errors produce similar flip rates.
    //
    // Circuit: |0⟩ → measure (should give 0, but measurement error may flip to 1)
    // With 10% measurement error on 0→1 flip

    let p_meas_0 = 0.10; // 10% chance of flipping 0 to 1

    // GeneralNoiseModel setup
    let general_model = GeneralNoiseModel::builder()
        .with_prep_probability(0.0)
        .with_meas_0_probability(p_meas_0)
        .with_meas_1_probability(0.0)
        .with_average_p1_probability(0.0)
        .with_average_p2_probability(0.0)
        .with_seed(42)
        .build();

    // ComposableNoiseModel setup
    let composable_model =
        ComposableNoiseModel::new().add_channel(MeasurementChannel::asymmetric(p_meas_0, 0.0));

    // Circuit for GeneralNoiseModel (just measure |0⟩)
    let general_counts = run_general_noise_model(
        general_model,
        |builder| {
            builder.add_measurements(&[0]);
        },
        1,
        NUM_SHOTS,
    );

    // Circuit for ComposableNoiseModel
    let commands = CommandBuilder::new().pz(0).mz(0).build();

    let composable_counts = run_composable_noise_model(composable_model, commands, 1, NUM_SHOTS);

    // Calculate |1⟩ percentages (measurement errors)
    let general_one = outcome_percentage(&general_counts, "1", NUM_SHOTS);
    let composable_one = outcome_percentage(&composable_counts, "1", NUM_SHOTS);

    println!("Measurement error comparison (p_meas_0={p_meas_0}):");
    println!("  GeneralNoiseModel: {general_one:.1}% |1⟩ (errors)");
    println!("  ComposableNoiseModel: {composable_one:.1}% |1⟩ (errors)");

    // Both should be close to 10%
    assert!(
        rates_match(general_one, composable_one, TOLERANCE_PERCENT),
        "Measurement error rates should match within {TOLERANCE_PERCENT}%: general={general_one:.1}%, composable={composable_one:.1}%"
    );

    // Also verify they're close to expected value
    assert!(
        (general_one - p_meas_0 * 100.0).abs() < TOLERANCE_PERCENT,
        "GeneralNoiseModel measurement error rate should be close to {}: got {general_one:.1}%",
        p_meas_0 * 100.0
    );
    assert!(
        (composable_one - p_meas_0 * 100.0).abs() < TOLERANCE_PERCENT,
        "ComposableNoiseModel measurement error rate should be close to {}: got {composable_one:.1}%",
        p_meas_0 * 100.0
    );
}

#[test]
fn test_preparation_error_comparison() {
    // Test that preparation errors produce similar bit flip rates.
    //
    // Circuit: prep |0⟩ → measure
    // With 15% preparation error (bit flip)

    let p_prep = 0.15;

    // GeneralNoiseModel setup
    let general_model = GeneralNoiseModel::builder()
        .with_prep_probability(p_prep)
        .with_prep_leak_ratio(0.0) // No leakage
        .with_meas_0_probability(0.0)
        .with_meas_1_probability(0.0)
        .with_average_p1_probability(0.0)
        .with_average_p2_probability(0.0)
        .with_seed(42)
        .build();

    // ComposableNoiseModel setup
    let composable_model = ComposableNoiseModel::new().add_channel(PreparationChannel::new(p_prep));

    // Circuit for GeneralNoiseModel
    // Note: GeneralNoiseModel may handle prep differently - it applies prep error after Prep gate
    let general_counts = run_general_noise_model(
        general_model,
        |builder| {
            builder.add_prep(&[0]);
            builder.add_measurements(&[0]);
        },
        1,
        NUM_SHOTS,
    );

    // Circuit for ComposableNoiseModel
    let commands = CommandBuilder::new().pz(0).mz(0).build();

    let composable_counts = run_composable_noise_model(composable_model, commands, 1, NUM_SHOTS);

    // Calculate |1⟩ percentages (preparation errors)
    let general_one = outcome_percentage(&general_counts, "1", NUM_SHOTS);
    let composable_one = outcome_percentage(&composable_counts, "1", NUM_SHOTS);

    println!("Preparation error comparison (p_prep={p_prep}):");
    println!("  GeneralNoiseModel: {general_one:.1}% |1⟩ (errors)");
    println!("  ComposableNoiseModel: {composable_one:.1}% |1⟩ (errors)");

    assert!(
        rates_match(general_one, composable_one, TOLERANCE_PERCENT),
        "Preparation error rates should match within {TOLERANCE_PERCENT}%: general={general_one:.1}%, composable={composable_one:.1}%"
    );
}

#[test]
fn test_combined_noise_comparison() {
    // Test with multiple noise sources active at once.
    //
    // Circuit: prep |0⟩ → H → CX → measure
    // Bell state creation with various noise sources

    let p_prep = 0.02;
    let p1 = 0.05;
    let p2 = 0.10;
    let p_meas = 0.02;

    // GeneralNoiseModel setup
    let general_model = GeneralNoiseModel::builder()
        .with_prep_probability(p_prep)
        .with_prep_leak_ratio(0.0)
        .with_meas_0_probability(p_meas)
        .with_meas_1_probability(p_meas)
        .with_average_p1_probability(p1 / 1.5)
        .with_average_p2_probability(p2 / 1.25)
        .with_p1_emission_ratio(0.0)
        .with_p2_emission_ratio(0.0)
        .with_seed(42)
        .build();

    // ComposableNoiseModel setup
    let composable_model = ComposableNoiseModel::new()
        .add_channel(PreparationChannel::new(p_prep))
        .add_channel(SingleQubitChannel::depolarizing(p1))
        .add_channel(TwoQubitChannel::depolarizing(p2))
        .add_channel(MeasurementChannel::symmetric(p_meas));

    // Bell state circuit for GeneralNoiseModel
    let general_counts = run_general_noise_model(
        general_model,
        |builder| {
            builder.add_prep(&[0]);
            builder.add_prep(&[1]);
            builder.add_h(&[0]);
            builder.add_cx(&[0], &[1]);
            builder.add_measurements(&[0, 1]);
        },
        2,
        NUM_SHOTS,
    );

    // Bell state circuit for ComposableNoiseModel
    let commands = CommandBuilder::new()
        .pz(0)
        .pz(1)
        .h(0)
        .cx(0, 1)
        .mz(0)
        .mz(1)
        .build();

    let composable_counts = run_composable_noise_model(composable_model, commands, 2, NUM_SHOTS);

    // For Bell state, ideal is 50% |00⟩ and 50% |11⟩
    // With noise, we expect some |01⟩ and |10⟩ (anti-correlated)

    let general_00 = outcome_percentage(&general_counts, "00", NUM_SHOTS);
    let general_11 = outcome_percentage(&general_counts, "11", NUM_SHOTS);
    let general_correlated = general_00 + general_11;

    let composable_00 = outcome_percentage(&composable_counts, "00", NUM_SHOTS);
    let composable_11 = outcome_percentage(&composable_counts, "11", NUM_SHOTS);
    let composable_correlated = composable_00 + composable_11;

    println!("Combined noise Bell state comparison:");
    println!(
        "  GeneralNoiseModel: {general_00:.1}% |00⟩, {general_11:.1}% |11⟩ ({general_correlated:.1}% correlated)"
    );
    println!(
        "  ComposableNoiseModel: {composable_00:.1}% |00⟩, {composable_11:.1}% |11⟩ ({composable_correlated:.1}% correlated)"
    );

    // The correlated outcome rate should be similar
    assert!(
        rates_match(
            general_correlated,
            composable_correlated,
            TOLERANCE_PERCENT * 2.0
        ),
        "Correlated rates should match within {}%: general={general_correlated:.1}%, composable={composable_correlated:.1}%",
        TOLERANCE_PERCENT * 2.0
    );
}

#[test]
fn test_general_noise_model_builder_comparison() {
    // Test that GeneralNoiseModelBuilder produces equivalent results to
    // GeneralNoiseModel from pecos-engines.
    //
    // This validates the builder API is correctly translating parameters.

    let p_prep = 0.02;
    let p1 = 0.05;
    let p2 = 0.10;
    let p_meas_0 = 0.03;
    let p_meas_1 = 0.02;

    // Original GeneralNoiseModel from pecos-engines
    let general_model = GeneralNoiseModel::builder()
        .with_prep_probability(p_prep)
        .with_prep_leak_ratio(0.0)
        .with_meas_0_probability(p_meas_0)
        .with_meas_1_probability(p_meas_1)
        .with_average_p1_probability(p1 / 1.5)
        .with_average_p2_probability(p2 / 1.25)
        .with_p1_emission_ratio(0.0)
        .with_p2_emission_ratio(0.0)
        .with_seed(42)
        .build();

    // New GeneralNoiseModelBuilder from pecos-neo
    let builder_model = GeneralNoiseModelBuilder::new()
        .with_p_prep(p_prep)
        .with_p1(p1)
        .with_p2(p2)
        .with_p_meas(p_meas_0, p_meas_1)
        .build();

    // Run circuit: prep → X → CX → measure
    let general_counts = run_general_noise_model(
        general_model,
        |builder| {
            builder.add_prep(&[0]);
            builder.add_prep(&[1]);
            builder.add_x(&[0]);
            builder.add_cx(&[0], &[1]);
            builder.add_measurements(&[0, 1]);
        },
        2,
        NUM_SHOTS,
    );

    let commands = CommandBuilder::new()
        .pz(0)
        .pz(1)
        .x(0)
        .cx(0, 1)
        .mz(0)
        .mz(1)
        .build();

    let composable_counts = run_composable_noise_model(builder_model, commands, 2, NUM_SHOTS);

    // Expected: |11⟩ with some errors
    let general_11 = outcome_percentage(&general_counts, "11", NUM_SHOTS);
    let composable_11 = outcome_percentage(&composable_counts, "11", NUM_SHOTS);

    println!("GeneralNoiseModelBuilder comparison:");
    println!("  pecos-engines GeneralNoiseModel: {general_11:.1}% |11⟩");
    println!("  pecos-neo GeneralNoiseModelBuilder: {composable_11:.1}% |11⟩");

    assert!(
        rates_match(general_11, composable_11, TOLERANCE_PERCENT * 2.0),
        "Builder should produce equivalent results: general={general_11:.1}%, builder={composable_11:.1}%"
    );
}

#[test]
fn test_idle_noise_with_time_scale() {
    // Test that idle noise with TimeScale produces expected decoherence.
    //
    // Circuit: prep |0> → X (to get |1>) → H → idle → H → measure
    // The H gates convert Z errors (dephasing) to bit flip errors for detection.
    // With T1=10us, T2=5us, and 1us idle, we expect ~10% error rate.
    //
    // Note: IdleChannel by default produces Z-only errors (dephasing model),
    // so we use H-basis measurement to detect them.

    use pecos_core::TimeScale;

    // Create model with nanosecond time scale
    // T1=10us=10000ns, T2=5us=5000ns
    let model = GeneralNoiseModelBuilder::new()
        .with_time_scale(TimeScale::NANOSECONDS)
        .with_idle_t1_t2(10e-6, 5e-6) // 10us T1, 5us T2
        .build();

    // Verify time scale is set
    assert!(model.time_scale().is_some());
    assert_eq!(model.channel_count(), 1);

    // Circuit with idle - use H gates to make Z errors detectable
    // H|+> = |0>, H|-> = |1>, so Z|+> = |-> gives different outcome after H
    let commands = CommandBuilder::new()
        .pz(0)
        .h(0) // Prepare |+> state
        .idle(0, 1000) // 1000 ns idle = 1 us (Z errors here)
        .h(0) // Convert Z errors to bit flips
        .mz(0)
        .build();

    let mut state = SparseStab::new(1);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(model)
        .with_seed(42);

    let qubits = [QubitId(0)];
    let mut error_count = 0;

    for _ in 0..NUM_SHOTS {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        if let Some(bits) = outcomes.bitstring(&qubits)
            && bits[0]
        {
            error_count += 1; // Z error during idle will cause |1> outcome
        }
    }

    let error_rate = (f64::from(error_count) / NUM_SHOTS as f64) * 100.0;

    println!("Idle noise with TimeScale:");
    println!("  T1=10us, T2=5us, idle=1us");
    println!("  Error rate: {error_rate:.1}% (expected ~10% from linear/T1 dephasing)");

    // With T1=10us and idle=1us, linear rate gives ~10% error probability
    // (quadratic rate is negligible at this scale)
    // Allow for statistical variation
    assert!(
        error_rate > 5.0 && error_rate < 20.0,
        "Error rate {error_rate:.1}% should be in reasonable range for T1/T2 dephasing"
    );
}
