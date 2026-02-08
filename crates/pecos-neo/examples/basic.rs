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
//! - Building quantum circuits with CommandBuilder
//! - Running simulations with ShotRunner
//! - Collecting and analyzing measurement outcomes
//!
//! Run with: cargo run --example basic

use pecos_neo::prelude::*;
use pecos_qsim::SparseStab;
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
        .prep(0)
        .prep(1)
        .h(0)       // Create superposition on qubit 0
        .cx(0, 1)   // Entangle with qubit 1
        .measure(0)
        .measure(1)
        .build();

    // Create a runner with the stabilizer simulator
    let mut runner = ShotRunner::new(SparseStab::new(2)).with_seed(42);

    // Run 1000 shots and collect statistics
    let mut counts: HashMap<String, usize> = HashMap::new();

    for _ in 0..1000 {
        let outcomes = runner.run_shot(&commands);
        let q0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
        let q1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);

        let key = format!("{}{}", q0 as u8, q1 as u8);
        *counts.entry(key).or_insert(0) += 1;
    }

    println!("Results (1000 shots):");
    for (outcome, count) in counts.iter() {
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
        .prep(0).prep(1).prep(2).prep(3)
        .h(0)
        .cx(0, 1)
        .cx(1, 2)
        .cx(2, 3)
        .measure(0).measure(1).measure(2).measure(3)
        .build();

    let mut runner = ShotRunner::new(SparseStab::new(4)).with_seed(123);

    let mut counts: HashMap<String, usize> = HashMap::new();

    for _ in 0..1000 {
        let outcomes = runner.run_shot(&commands);

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
        if *count > 10 {  // Only show significant outcomes
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
        .prep(0).prep(1).prep(2)
        .h(0)
        .sz(1)
        .cx(0, 1)
        .cz(1, 2)
        .h(2)
        .szdg(0)
        .cx(2, 0)
        .measure(0).measure(1).measure(2)
        .build();

    let mut runner = ShotRunner::new(SparseStab::new(3)).with_seed(456);

    let mut counts: HashMap<String, usize> = HashMap::new();

    for _ in 0..1000 {
        let outcomes = runner.run_shot(&commands);

        let mut key = String::new();
        for i in 0..3 {
            let bit = outcomes.get_bit(QubitId(i)).unwrap_or(false);
            key.push(if bit { '1' } else { '0' });
        }
        *counts.entry(key).or_insert(0) += 1;
    }

    println!("Results (1000 shots):");
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by(|(_, a), (_, b)| b.cmp(a));  // Sort by count descending
    for (outcome, count) in sorted.iter().take(5) {
        println!("  |{}⟩: {} ({:.1}%)", outcome, count, **count as f64 / 10.0);
    }
    println!();
}

/// Demonstrate shot statistics collection
fn example_shot_statistics() {
    println!("--- Shot Statistics ---");

    // Simple Hadamard circuit - should give 50/50 distribution
    let commands = CommandBuilder::new()
        .prep(0)
        .h(0)
        .measure(0)
        .build();

    let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(789);

    let num_shots = 10000;
    let mut count_0 = 0;
    let mut count_1 = 0;

    for _ in 0..num_shots {
        let outcomes = runner.run_shot(&commands);
        if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            count_1 += 1;
        } else {
            count_0 += 1;
        }
    }

    println!("Hadamard gate statistics ({} shots):", num_shots);
    println!("  |0⟩: {} ({:.2}%)", count_0, count_0 as f64 / num_shots as f64 * 100.0);
    println!("  |1⟩: {} ({:.2}%)", count_1, count_1 as f64 / num_shots as f64 * 100.0);

    // Calculate chi-squared statistic for uniformity
    let expected = num_shots as f64 / 2.0;
    let chi_sq = (count_0 as f64 - expected).powi(2) / expected
               + (count_1 as f64 - expected).powi(2) / expected;
    println!("  Chi-squared: {:.2} (should be < 3.84 for p=0.05)", chi_sq);
    println!();
}
