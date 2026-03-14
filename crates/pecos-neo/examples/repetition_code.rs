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

//! Repetition code example demonstrating quantum error correction.
//!
//! This example shows:
//! - Building a distance-3 repetition code
//! - Syndrome extraction circuits
//! - Running noisy simulations
//! - Analyzing logical error rates
//!
//! The repetition code is a simple 1D error correction code that protects
//! against bit-flip (X) errors. It uses:
//! - 3 data qubits (forming a logical qubit)
//! - 2 ancilla qubits (for syndrome measurement)
//!
//! Run with: cargo run --example `repetition_code`

use pecos_neo::prelude::*;
use pecos_qsim::SparseStab;
use std::collections::HashMap;

/// A distance-3 repetition code.
///
/// Layout:
/// ```text
///   D0 --- A0 --- D1 --- A1 --- D2
/// ```
///
/// - D0, D1, D2 are data qubits (indices 0, 1, 2)
/// - A0, A1 are ancilla qubits (indices 3, 4)
/// - A0 measures Z0 * Z1 parity
/// - A1 measures Z1 * Z2 parity
struct RepetitionCode {
    data_qubits: [usize; 3],
    ancilla_qubits: [usize; 2],
}

impl RepetitionCode {
    fn new() -> Self {
        Self {
            data_qubits: [0, 1, 2],
            ancilla_qubits: [3, 4],
        }
    }

    fn num_qubits(&self) -> usize {
        5
    }

    /// Build syndrome extraction circuit for one round.
    fn build_syndrome_round(&self, builder: CommandBuilder) -> CommandBuilder {
        let mut b = builder;

        // Prepare ancillas in |0⟩
        for &a in &self.ancilla_qubits {
            b = b.pz(a);
        }

        // Parity check for A0 = Z0 * Z1
        b = b.cx(self.data_qubits[0], self.ancilla_qubits[0]);
        b = b.cx(self.data_qubits[1], self.ancilla_qubits[0]);

        // Parity check for A1 = Z1 * Z2
        b = b.cx(self.data_qubits[1], self.ancilla_qubits[1]);
        b = b.cx(self.data_qubits[2], self.ancilla_qubits[1]);

        // Measure ancillas
        for &a in &self.ancilla_qubits {
            b = b.mz(a);
        }

        b
    }

    /// Build full circuit with initialization, syndrome rounds, and final measurement.
    fn build_circuit(&self, num_rounds: usize) -> CommandQueue {
        let mut builder = CommandBuilder::new();

        // Initialize data qubits in |000⟩ (logical |0⟩)
        for &d in &self.data_qubits {
            builder = builder.pz(d);
        }

        // Syndrome extraction rounds
        for _ in 0..num_rounds {
            builder = self.build_syndrome_round(builder);
        }

        // Final data measurement
        for &d in &self.data_qubits {
            builder = builder.mz(d);
        }

        builder.build()
    }

    /// Decode syndromes using majority voting.
    ///
    /// Returns the estimated logical measurement outcome.
    fn decode(&self, data_outcomes: &[bool], syndromes: &[Vec<bool>]) -> bool {
        // Simple decoder: use majority vote on final data
        // A more sophisticated decoder would use syndrome history
        let _ = syndromes; // Could be used for better decoding

        let ones = data_outcomes.iter().filter(|&&b| b).count();
        ones > data_outcomes.len() / 2
    }
}

/// Simulation results
#[derive(Default)]
struct SimulationResults {
    total_shots: usize,
    logical_errors: usize,
    syndrome_counts: HashMap<String, usize>,
}

impl SimulationResults {
    fn logical_error_rate(&self) -> f64 {
        if self.total_shots == 0 {
            0.0
        } else {
            self.logical_errors as f64 / self.total_shots as f64
        }
    }
}

fn main() {
    println!("=== Repetition Code Example ===\n");

    let code = RepetitionCode::new();

    example_noiseless(&code);
    example_with_noise(&code);
    example_error_scaling(&code);
    example_round_scaling(&code);
}

/// Run the code without noise (sanity check)
fn example_noiseless(code: &RepetitionCode) {
    println!("--- Noiseless Operation ---");

    let commands = code.build_circuit(3);
    let mut state = SparseStab::new(code.num_qubits());
    let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);

    let mut all_zero = true;
    for _ in 0..100 {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        // Check final data qubits
        for &d in &code.data_qubits {
            if outcomes.get_bit(QubitId(d)).unwrap_or(true) {
                all_zero = false;
            }
        }

        // Syndromes should all be 0 in noiseless case
        // (we don't need to check each one here)
    }

    println!("  100 shots with 3 rounds: all data qubits = 0? {all_zero}");
    println!("  (Expected: true - no noise means no errors)\n");
}

/// Run with depolarizing noise
fn example_with_noise(code: &RepetitionCode) {
    println!("--- With Depolarizing Noise ---");

    let p_error = 0.02; // 2% error rate

    let noise = ComposableNoiseModel::new()
        .add_plugin(CorePlugin)
        .add_channel(SingleQubitChannel::depolarizing(p_error))
        .add_channel(TwoQubitChannel::depolarizing(p_error * 2.0))
        .add_channel(MeasurementChannel::symmetric(p_error));

    let commands = code.build_circuit(3);

    let mut state = SparseStab::new(code.num_qubits());
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise)
        .with_seed(42);

    let mut results = SimulationResults::default();
    let shots = 2000;

    for _ in 0..shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        // Extract data outcomes (last 3 measurements)
        let all_outcomes: Vec<bool> = outcomes.iter().map(|o| o.outcome).collect();
        let num_ancilla_meas = 3 * 2; // 3 rounds * 2 ancillas
        let data_outcomes: Vec<bool> = all_outcomes[num_ancilla_meas..].to_vec();

        // Extract syndromes
        let syndromes: Vec<Vec<bool>> = (0..3)
            .map(|r| {
                let start = r * 2;
                vec![
                    all_outcomes.get(start).copied().unwrap_or(false),
                    all_outcomes.get(start + 1).copied().unwrap_or(false),
                ]
            })
            .collect();

        // Decode
        let logical = code.decode(&data_outcomes, &syndromes);

        results.total_shots += 1;
        if logical {
            results.logical_errors += 1; // We initialized in |0⟩, so |1⟩ is error
        }

        // Track syndrome patterns
        let syndrome_key: String = syndromes
            .iter()
            .flat_map(|s| s.iter().map(|&b| if b { '1' } else { '0' }))
            .collect();
        *results.syndrome_counts.entry(syndrome_key).or_insert(0) += 1;
    }

    println!("  Physical error rate: {:.1}%", p_error * 100.0);
    println!(
        "  Logical error rate: {:.2}%",
        results.logical_error_rate() * 100.0
    );
    println!("  (Logical rate should be lower than physical due to error correction)");

    // Show most common syndrome patterns
    let mut sorted: Vec<_> = results.syndrome_counts.iter().collect();
    sorted.sort_by(|(_, a), (_, b)| b.cmp(a));

    println!("\n  Most common syndrome patterns:");
    for (pattern, count) in sorted.iter().take(5) {
        println!(
            "    {}: {:.1}%",
            pattern,
            **count as f64 / f64::from(shots) * 100.0
        );
    }
    println!();
}

/// Show how logical error rate scales with physical error rate
fn example_error_scaling(code: &RepetitionCode) {
    println!("--- Error Rate Scaling ---");
    println!("  Physical vs Logical error rates (3 rounds):\n");
    println!("  {:>10} {:>15} {:>12}", "Physical", "Logical", "Reduction");
    println!("  {:->10} {:->15} {:->12}", "", "", "");

    let commands = code.build_circuit(3);

    for p_phys in [0.005, 0.01, 0.02, 0.03, 0.05] {
        let noise = ComposableNoiseModel::new()
            .add_plugin(CorePlugin)
            .add_channel(SingleQubitChannel::depolarizing(p_phys))
            .add_channel(TwoQubitChannel::depolarizing(p_phys * 2.0))
            .add_channel(MeasurementChannel::symmetric(p_phys));

        let mut state = SparseStab::new(code.num_qubits());
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);

        let shots = 3000;
        let mut errors = 0;

        for _ in 0..shots {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            let all_outcomes: Vec<bool> = outcomes.iter().map(|o| o.outcome).collect();
            let num_ancilla_meas = 3 * 2;
            let data_outcomes: Vec<bool> = all_outcomes[num_ancilla_meas..].to_vec();

            // Simple majority vote
            let ones = data_outcomes.iter().filter(|&&b| b).count();
            if ones > 1 {
                errors += 1;
            }
        }

        let p_log = f64::from(errors) / f64::from(shots);
        let reduction = if p_log > 0.0 {
            p_phys / p_log
        } else {
            f64::INFINITY
        };

        println!(
            "  {:>9.1}% {:>14.2}% {:>11.1}x",
            p_phys * 100.0,
            p_log * 100.0,
            reduction
        );
    }
    println!();
}

/// Show how logical error rate changes with number of rounds
fn example_round_scaling(code: &RepetitionCode) {
    println!("--- Round Scaling ---");
    println!("  Logical error rate vs number of syndrome rounds:\n");
    println!("  Physical error rate: 2%\n");
    println!("  {:>8} {:>15}", "Rounds", "Logical Error");
    println!("  {:->8} {:->15}", "", "");

    let p_phys = 0.02;

    for num_rounds in [1, 2, 3, 5, 8, 10] {
        let commands = code.build_circuit(num_rounds);

        let noise = ComposableNoiseModel::new()
            .add_plugin(CorePlugin)
            .add_channel(SingleQubitChannel::depolarizing(p_phys))
            .add_channel(TwoQubitChannel::depolarizing(p_phys * 2.0))
            .add_channel(MeasurementChannel::symmetric(p_phys));

        let mut state = SparseStab::new(code.num_qubits());
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);

        let shots = 2000;
        let mut errors = 0;

        for _ in 0..shots {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            let all_outcomes: Vec<bool> = outcomes.iter().map(|o| o.outcome).collect();
            let num_ancilla_meas = num_rounds * 2;
            let data_outcomes: Vec<bool> = all_outcomes[num_ancilla_meas..].to_vec();

            let ones = data_outcomes.iter().filter(|&&b| b).count();
            if ones > 1 {
                errors += 1;
            }
        }

        let p_log = f64::from(errors) / f64::from(shots);
        println!("  {:>8} {:>14.2}%", num_rounds, p_log * 100.0);
    }

    println!("\n  (More rounds = more opportunities for errors, but also better correction)");
    println!();
}
