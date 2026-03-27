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

//! Basic usage examples for pecos-neo.
//!
//! This example demonstrates:
//! - Building quantum circuits with `CommandBuilder`
//! - Running simulations with `CircuitRunner`
//! - Collecting and analyzing measurement outcomes
//!
//! Run with: cargo run --example basic

use pecos_neo::prelude::*;
use pecos_simulators::SparseStab;
use std::collections::HashMap;

fn main() {
    println!("=== pecos-neo Basic Examples ===\n");

    example_bell_state();
    example_ghz_state();
    example_random_circuit();
    example_shot_statistics();
}

/// Create and measure a Bell state |00⟩ + |11⟩
fn example_bell_state() {
    println!("--- Bell State ---");

    // Build the circuit
    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .h(&[0]) // Create superposition on qubit 0
        .cx(&[(0, 1)]) // Entangle with qubit 1
        .mz(&[0])
        .mz(&[1])
        .build();

    // Create a simulator and a stateless runner
    let mut state = SparseStab::new(2);
    let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);

    // Run 1000 shots and collect statistics
    let mut counts: HashMap<String, usize> = HashMap::new();

    for _ in 0..1000 {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        let q0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
        let q1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);

        let key = format!("{}{}", u8::from(q0), u8::from(q1));
        *counts.entry(key).or_insert(0) += 1;
    }

    println!("Results (1000 shots):");
    for (outcome, count) in &counts {
        println!("  |{}⟩: {} ({:.1}%)", outcome, count, *count as f64 / 10.0);
    }

    // Bell state should only produce |00⟩ and |11⟩
    let correlated = counts.get("00").unwrap_or(&0) + counts.get("11").unwrap_or(&0);
    println!("  Correlation: {:.1}%\n", correlated as f64 / 10.0);
}

/// Create and measure a 4-qubit GHZ state |0000⟩ + |1111⟩
fn example_ghz_state() {
    println!("--- GHZ State (4 qubits) ---");

    // Build the GHZ circuit
    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .pz(&[2])
        .pz(&[3])
        .h(&[0])
        .cx(&[(0, 1)])
        .cx(&[(1, 2)])
        .cx(&[(2, 3)])
        .mz(&[0])
        .mz(&[1])
        .mz(&[2])
        .mz(&[3])
        .build();

    let mut state = SparseStab::new(4);
    let mut runner = CircuitRunner::<SparseStab>::new().with_seed(123);

    let mut counts: HashMap<String, usize> = HashMap::new();

    for _ in 0..1000 {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        let mut key = String::new();
        for i in 0..4 {
            let bit = outcomes.get_bit(QubitId(i)).unwrap_or(false);
            key.push(if bit { '1' } else { '0' });
        }
        *counts.entry(key).or_insert(0) += 1;
    }

    println!("Results (1000 shots):");
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by_key(|(k, _)| *k);
    for (outcome, count) in sorted {
        if *count > 10 {
            // Only show significant outcomes
            println!("  |{}⟩: {} ({:.1}%)", outcome, count, *count as f64 / 10.0);
        }
    }
    println!();
}

/// Run a random Clifford circuit
fn example_random_circuit() {
    println!("--- Random Clifford Circuit ---");

    // Build a circuit with various Clifford gates
    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .pz(&[2])
        .h(&[0])
        .sz(&[1])
        .cx(&[(0, 1)])
        .cz(&[(1, 2)])
        .h(&[2])
        .szdg(&[0])
        .cx(&[(2, 0)])
        .mz(&[0])
        .mz(&[1])
        .mz(&[2])
        .build();

    let mut state = SparseStab::new(3);
    let mut runner = CircuitRunner::<SparseStab>::new().with_seed(456);

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

    println!("Results (1000 shots):");
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by(|(_, a), (_, b)| b.cmp(a)); // Sort by count descending
    for (outcome, count) in sorted.iter().take(5) {
        println!("  |{}⟩: {} ({:.1}%)", outcome, count, **count as f64 / 10.0);
    }
    println!();
}

/// Demonstrate shot statistics collection
fn example_shot_statistics() {
    println!("--- Shot Statistics ---");

    // Simple Hadamard circuit - should give 50/50 distribution
    let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

    let mut state = SparseStab::new(1);
    let mut runner = CircuitRunner::<SparseStab>::new().with_seed(789);

    let num_shots = 10000;
    let mut count_0 = 0;
    let mut count_1 = 0;

    for _ in 0..num_shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            count_1 += 1;
        } else {
            count_0 += 1;
        }
    }

    println!("Hadamard gate statistics ({num_shots} shots):");
    println!(
        "  |0⟩: {} ({:.2}%)",
        count_0,
        f64::from(count_0) / f64::from(num_shots) * 100.0
    );
    println!(
        "  |1⟩: {} ({:.2}%)",
        count_1,
        f64::from(count_1) / f64::from(num_shots) * 100.0
    );

    // Calculate chi-squared statistic for uniformity
    let expected = f64::from(num_shots) / 2.0;
    let chi_sq = (f64::from(count_0) - expected).powi(2) / expected
        + (f64::from(count_1) - expected).powi(2) / expected;
    println!("  Chi-squared: {chi_sq:.2} (should be < 3.84 for p=0.05)");
    println!();
}
