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

//! Circuit integration example for pecos-neo.
//!
//! This example demonstrates:
//! - Converting `TickCircuit` to `CommandQueue`
//! - Converting `DagCircuit` to `CommandQueue`
//! - Executing circuits directly with `CircuitRunner`
//! - Round-trip conversions
//!
//! Run with: cargo run --example `circuit_integration`

use pecos_neo::prelude::*;
use pecos_qsim::SparseStab;
use pecos_quantum::{DagCircuit, TickCircuit};
use std::collections::HashMap;

fn main() {
    println!("=== Circuit Integration Examples ===\n");

    example_tick_circuit_execution();
    example_dag_circuit_execution();
    example_tick_with_noise();
    example_round_trip();
    example_qec_style_circuit();
}

/// Execute a `TickCircuit` directly
fn example_tick_circuit_execution() {
    println!("--- TickCircuit Execution ---");

    // Build a Bell state circuit using TickCircuit
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0]);
    circuit.tick().pz(&[1]);
    circuit.tick().h(&[0]);
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().mz(&[0]);
    circuit.tick().mz(&[1]);

    println!("  TickCircuit with {} ticks", circuit.num_ticks());

    // Convert to CommandQueue and execute
    let commands = pecos_neo::command::CommandQueue::from(&circuit);
    let mut state = SparseStab::new(2);
    let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);

    let mut counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..1000 {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        let q0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
        let q1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);
        let key = format!("{}{}", u8::from(q0), u8::from(q1));
        *counts.entry(key).or_insert(0) += 1;
    }

    println!("  Bell state results (1000 shots):");
    for (outcome, count) in &counts {
        println!("    |{}⟩: {:.1}%", outcome, *count as f64 / 10.0);
    }
    println!();
}

/// Execute a `DagCircuit` directly
fn example_dag_circuit_execution() {
    println!("--- DagCircuit Execution ---");

    // Build a GHZ state circuit using DagCircuit
    let mut dag = DagCircuit::new();
    dag.pz(0);
    dag.pz(1);
    dag.pz(2);
    dag.h(0);
    dag.cx(0, 1);
    dag.cx(1, 2);
    dag.mz(0);
    dag.mz(1);
    dag.mz(2);

    println!("  DagCircuit with {} gates", dag.topological_order().len());

    // Convert to CommandQueue and execute
    let commands = pecos_neo::command::CommandQueue::from(&dag);
    let mut state = SparseStab::new(3);
    let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);

    let mut counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..1000 {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        let mut key = String::new();
        for i in 0..3 {
            let bit = outcomes.get_bit(QubitId(i)).unwrap_or(false);
            key.push(if bit { '1' } else { '0' });
        }
        *counts.entry(key).or_insert(0) += 1;
    }

    println!("  GHZ state results (1000 shots):");
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by_key(|(k, _)| *k);
    for (outcome, count) in sorted {
        if *count > 10 {
            println!("    |{}⟩: {:.1}%", outcome, *count as f64 / 10.0);
        }
    }
    println!();
}

/// Execute `TickCircuit` with noise
fn example_tick_with_noise() {
    println!("--- TickCircuit with Noise ---");

    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0]);
    circuit.tick().pz(&[1]);
    circuit.tick().h(&[0]);
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().mz(&[0]);
    circuit.tick().mz(&[1]);

    // Add depolarizing noise
    let noise = ComposableNoiseModel::new()
        .add_plugin(CorePlugin)
        .add_channel(SingleQubitChannel::depolarizing(0.01))
        .add_channel(TwoQubitChannel::depolarizing(0.02));

    let mut state = SparseStab::new(2);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise)
        .with_seed(42);

    let mut correlated = 0;
    let mut anti_correlated = 0;

    let commands = pecos_neo::command::CommandQueue::from(&circuit);
    for _ in 0..1000 {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        let q0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
        let q1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);

        if q0 == q1 {
            correlated += 1;
        } else {
            anti_correlated += 1;
        }
    }

    println!("  Bell state with 1%/2% depolarizing noise:");
    println!("    Correlated: {:.1}%", f64::from(correlated) / 10.0);
    println!(
        "    Anti-correlated: {:.1}%",
        f64::from(anti_correlated) / 10.0
    );
    println!();
}

/// Demonstrate round-trip conversion
fn example_round_trip() {
    println!("--- Round-Trip Conversion ---");

    // Start with CommandBuilder
    let original = CommandBuilder::new()
        .pz(0)
        .pz(1)
        .h(0)
        .cx(0, 1)
        .mz(0)
        .mz(1)
        .build();

    println!("  Original CommandQueue: {} commands", original.len());

    // Convert to TickCircuit
    let tick_circuit = TickCircuit::from(&original);
    println!(
        "  Converted to TickCircuit: {} ticks",
        tick_circuit.num_ticks()
    );

    // Convert back to CommandQueue
    let back = pecos_neo::command::CommandQueue::from(&tick_circuit);
    println!("  Converted back to CommandQueue: {} commands", back.len());

    // Both should produce statistically identical results
    let mut state1 = SparseStab::new(2);
    let mut runner1 = CircuitRunner::<SparseStab>::new().with_seed(42);
    let mut state2 = SparseStab::new(2);
    let mut runner2 = CircuitRunner::<SparseStab>::new().with_seed(42);

    let mut corr1 = 0;
    let mut corr2 = 0;
    let shots = 1000;

    for _ in 0..shots {
        state1.reset();
        let outcomes1 = runner1.apply_circuit(&mut state1, &original).unwrap();
        state2.reset();
        let outcomes2 = runner2.apply_circuit(&mut state2, &back).unwrap();

        if outcomes1.get_bit(QubitId(0)) == outcomes1.get_bit(QubitId(1)) {
            corr1 += 1;
        }
        if outcomes2.get_bit(QubitId(0)) == outcomes2.get_bit(QubitId(1)) {
            corr2 += 1;
        }
    }

    println!("  Bell state correlation (should be ~100% for both):");
    println!(
        "    Original: {:.1}%",
        f64::from(corr1) / f64::from(shots) * 100.0
    );
    println!(
        "    Round-trip: {:.1}%",
        f64::from(corr2) / f64::from(shots) * 100.0
    );
    println!();
}

/// QEC-style circuit using `TickCircuit`'s parallel structure
fn example_qec_style_circuit() {
    println!("--- QEC-Style Syndrome Extraction ---");

    // Build a simple repetition code syndrome extraction round
    // Data qubits: 0, 1, 2
    // Ancilla qubits: 3, 4

    let mut circuit = TickCircuit::new();

    // Tick 0: Prepare all qubits
    circuit.tick().pz(&[0, 1, 2, 3, 4]);

    // Tick 1: Prepare data in |+⟩ state
    circuit.tick().h(&[0, 1, 2]);

    // Tick 2: First round of CNOTs (parallel)
    circuit.tick().cx(&[(0, 3), (1, 4)]);

    // Tick 3: Second round of CNOTs
    circuit.tick().cx(&[(1, 3), (2, 4)]);

    // Tick 4: Measure ancillas
    circuit.tick().mz(&[3, 4]);

    // Tick 5: Measure data (return to Z basis first)
    circuit.tick().h(&[0, 1, 2]);

    // Tick 6: Measure data qubits
    circuit.tick().mz(&[0, 1, 2]);

    println!(
        "  Syndrome extraction circuit: {} ticks",
        circuit.num_ticks()
    );

    // Run with noise
    let noise = ComposableNoiseModel::new()
        .add_plugin(CorePlugin)
        .add_channel(SingleQubitChannel::depolarizing(0.005))
        .add_channel(TwoQubitChannel::depolarizing(0.01))
        .add_channel(MeasurementChannel::symmetric(0.005));

    let mut state = SparseStab::new(5);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise)
        .with_seed(42);

    let mut syndrome_counts: HashMap<String, usize> = HashMap::new();

    let commands = pecos_neo::command::CommandQueue::from(&circuit);
    for _ in 0..1000 {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        // Extract syndrome (ancilla measurements)
        let s0 = outcomes.get_bit(QubitId(3)).unwrap_or(false);
        let s1 = outcomes.get_bit(QubitId(4)).unwrap_or(false);
        let key = format!("{}{}", u8::from(s0), u8::from(s1));
        *syndrome_counts.entry(key).or_insert(0) += 1;
    }

    println!("  Syndrome distribution (1000 shots):");
    let mut sorted: Vec<_> = syndrome_counts.iter().collect();
    sorted.sort_by(|(_, a), (_, b)| b.cmp(a));
    for (syndrome, count) in sorted {
        println!("    {}: {:.1}%", syndrome, *count as f64 / 10.0);
    }
    println!("  (00 = no detected errors, others indicate errors)");
    println!();
}
