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

//! Custom noise channel example for pecos-neo.
//!
//! This example demonstrates:
//! - Implementing the `NoiseChannel` trait
//! - Creating gate-specific noise
//! - Building correlated noise models
//! - Using context for stateful noise
//!
//! Run with: cargo run --example `custom_channel`

use pecos_neo::command::GateCommand;
use pecos_neo::prelude::*;
use pecos_random::PecosRng;
use pecos_simulators::SparseStab;
use rand::RngExt;
use std::collections::HashMap;

fn main() {
    println!("=== Custom Noise Channel Examples ===\n");

    example_amplitude_damping();
    example_gate_specific_noise();
    example_correlated_noise();
    example_context_aware_noise();
    example_builtin_gate_dependent();
    example_builtin_correlated();
}

// --- Example 1: Amplitude Damping Channel ---

/// A simple amplitude damping channel.
///
/// Models T1 decay: |1⟩ → |0⟩ with probability gamma.
/// This is a simplified model that applies X when damping occurs.
#[derive(Debug, Clone)]
struct AmplitudeDampingChannel {
    /// Damping probability per gate
    gamma: f64,
}

impl AmplitudeDampingChannel {
    fn new(gamma: f64) -> Self {
        Self { gamma }
    }
}

impl NoiseChannel for AmplitudeDampingChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        // Apply after any gate
        matches!(event, NoiseEvent::AfterGate { .. })
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        _ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let NoiseEvent::AfterGate { qubits, .. } = event else {
            return NoiseResponse::None;
        };

        let mut gates = Vec::new();

        for &qubit in *qubits {
            if rng.random::<f64>() < self.gamma {
                // Apply X to simulate |1⟩ → |0⟩ decay
                // (This is a simplification; true amplitude damping is non-unitary)
                gates.push(GateCommand::x(qubit));
            }
        }

        if gates.is_empty() {
            NoiseResponse::None
        } else {
            NoiseResponse::inject_gates(gates.into())
        }
    }

    fn name(&self) -> &'static str {
        "AmplitudeDampingChannel"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

fn example_amplitude_damping() {
    println!("--- Amplitude Damping Channel ---");

    // Circuit: prepare |1⟩ and measure
    let commands = CommandBuilder::new()
        .pz(&[0])
        .x(&[0]) // Prepare |1⟩
        .mz(&[0])
        .build();

    for gamma in [0.0, 0.1, 0.3, 0.5] {
        let noise = ComposableNoiseModel::new()
            .add_plugin(&CorePlugin)
            .add_channel(AmplitudeDampingChannel::new(gamma));

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);

        let shots = 1000;
        let mut decays = 0;

        for _ in 0..shots {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            // If we get |0⟩, decay occurred
            if !outcomes.get_bit(QubitId(0)).unwrap_or(true) {
                decays += 1;
            }
        }

        println!(
            "  gamma={:.1}: {:.1}% decayed to |0⟩",
            gamma,
            f64::from(decays) / f64::from(shots) * 100.0
        );
    }
    println!();
}

// --- Example 2: Gate-Specific Noise ---

/// Noise channel that applies different error rates for different gates.
#[derive(Debug, Clone)]
struct GateSpecificChannel {
    /// Error rates per gate type
    rates: HashMap<GateType, f64>,
    /// Default rate for unlisted gates
    default_rate: f64,
}

impl GateSpecificChannel {
    fn new() -> Self {
        Self {
            rates: HashMap::new(),
            default_rate: 0.0,
        }
    }

    #[allow(dead_code)]
    fn with_rate(mut self, gate: GateType, rate: f64) -> Self {
        self.rates.insert(gate, rate);
        self
    }

    fn with_default(mut self, rate: f64) -> Self {
        self.default_rate = rate;
        self
    }
}

impl NoiseChannel for GateSpecificChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        matches!(event, NoiseEvent::AfterGate { .. })
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        _ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let NoiseEvent::AfterGate {
            gate_type, qubits, ..
        } = event
        else {
            return NoiseResponse::None;
        };

        let rate = self
            .rates
            .get(gate_type)
            .copied()
            .unwrap_or(self.default_rate);

        if rate <= 0.0 {
            return NoiseResponse::None;
        }

        let mut gates = Vec::new();

        for &qubit in *qubits {
            if rng.random::<f64>() < rate {
                // Apply random Pauli
                let pauli = match rng.random_range(0..3) {
                    0 => GateType::X,
                    1 => GateType::Y,
                    _ => GateType::Z,
                };
                gates.push(GateCommand::new(pauli, smallvec::smallvec![qubit]));
            }
        }

        if gates.is_empty() {
            NoiseResponse::None
        } else {
            NoiseResponse::inject_gates(gates.into())
        }
    }

    fn name(&self) -> &'static str {
        "GateSpecificChannel"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

fn example_gate_specific_noise() {
    println!("--- Gate-Specific Noise ---");

    let shots = 1000;

    // Test H gate
    let h_commands = CommandBuilder::new()
        .pz(&[0])
        .h(&[0])
        .h(&[0]) // H^2 = I, should return to |0⟩
        .mz(&[0])
        .build();

    let noise_h = ComposableNoiseModel::new()
        .add_plugin(&CorePlugin)
        .add_channel(
            GateSpecificChannel::new()
                .with_rate(GateType::H, 0.20)
                .with_rate(GateType::SZ, 0.02)
                .with_default(0.05),
        );

    let mut state = SparseStab::new(1);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise_h)
        .with_seed(42);

    let mut h_errors = 0;
    for _ in 0..shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &h_commands).unwrap();
        if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            h_errors += 1;
        }
    }

    // Test SZ gate
    let sz_commands = CommandBuilder::new()
        .pz(&[0])
        .sz(&[0])
        .szdg(&[0]) // SZ * SZ^dag = I
        .mz(&[0])
        .build();

    let noise_sz = ComposableNoiseModel::new()
        .add_plugin(&CorePlugin)
        .add_channel(
            GateSpecificChannel::new()
                .with_rate(GateType::H, 0.20)
                .with_rate(GateType::SZ, 0.02)
                .with_default(0.05),
        );

    let mut state = SparseStab::new(1);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise_sz)
        .with_seed(42);

    let mut sz_errors = 0;
    for _ in 0..shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &sz_commands).unwrap();
        if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            sz_errors += 1;
        }
    }

    println!(
        "  H gate (20% rate): {:.1}% logical errors",
        f64::from(h_errors) / f64::from(shots) * 100.0
    );
    println!(
        "  SZ gate (2% rate): {:.1}% logical errors",
        f64::from(sz_errors) / f64::from(shots) * 100.0
    );
    println!();
}

// --- Example 3: Correlated Noise ---

/// Noise channel with spatial correlations.
///
/// When an error occurs on one qubit, neighboring qubits have
/// increased probability of error.
#[derive(Debug, Clone)]
struct CorrelatedNoiseChannel {
    /// Base error probability
    base_rate: f64,
    /// Correlation factor (0 = independent, 1 = fully correlated)
    correlation: f64,
}

impl CorrelatedNoiseChannel {
    fn new(base_rate: f64, correlation: f64) -> Self {
        Self {
            base_rate,
            correlation: correlation.clamp(0.0, 1.0),
        }
    }
}

impl NoiseChannel for CorrelatedNoiseChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        matches!(event, NoiseEvent::AfterGate { .. })
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        _ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let NoiseEvent::AfterGate { qubits, .. } = event else {
            return NoiseResponse::None;
        };

        if qubits.len() < 2 {
            // For single-qubit gates, just use base rate
            if rng.random::<f64>() < self.base_rate {
                return NoiseResponse::inject_gate(GateCommand::z(qubits[0]));
            }
            return NoiseResponse::None;
        }

        // For two-qubit gates, apply correlated errors
        let mut gates = Vec::new();

        // First qubit
        let first_error = rng.random::<f64>() < self.base_rate;
        if first_error {
            gates.push(GateCommand::z(qubits[0]));
        }

        // Second qubit: higher probability if first had error
        let second_rate = if first_error {
            self.base_rate + (1.0 - self.base_rate) * self.correlation
        } else {
            self.base_rate * (1.0 - self.correlation)
        };

        if rng.random::<f64>() < second_rate {
            gates.push(GateCommand::z(qubits[1]));
        }

        if gates.is_empty() {
            NoiseResponse::None
        } else {
            NoiseResponse::inject_gates(gates.into())
        }
    }

    fn name(&self) -> &'static str {
        "CorrelatedNoiseChannel"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

fn example_correlated_noise() {
    println!("--- Correlated Noise ---");

    let shots = 2000;

    for correlation in [0.0, 0.5, 0.9] {
        let noise = ComposableNoiseModel::new()
            .add_plugin(&CorePlugin)
            .add_channel(CorrelatedNoiseChannel::new(0.1, correlation));

        // Bell state circuit
        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .h(&[0])
            .cx(&[(0, 1)])
            .mz(&[0])
            .mz(&[1])
            .build();

        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);

        let mut both_same = 0;
        let mut both_different = 0;

        for _ in 0..shots {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            let q0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
            let q1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);

            if q0 == q1 {
                both_same += 1;
            } else {
                both_different += 1;
            }
        }

        println!(
            "  correlation={:.1}: {:.1}% correlated, {:.1}% anti-correlated",
            correlation,
            f64::from(both_same) / f64::from(shots) * 100.0,
            f64::from(both_different) / f64::from(shots) * 100.0
        );
    }
    println!("  (Higher correlation → more correlated errors → less anti-correlation)");
    println!();
}

// --- Example 4: Context-Aware Noise ---

/// Noise channel that uses context to apply different noise
/// based on qubit state (leaked qubits get different treatment).
#[derive(Debug, Clone)]
struct ContextAwareChannel {
    /// Error rate for active qubits
    active_rate: f64,
    /// Error rate for leaked qubits (might be higher due to heating)
    leaked_rate: f64,
}

impl ContextAwareChannel {
    fn new(active_rate: f64, leaked_rate: f64) -> Self {
        Self {
            active_rate,
            leaked_rate,
        }
    }
}

impl NoiseChannel for ContextAwareChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        matches!(event, NoiseEvent::AfterGate { .. })
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let NoiseEvent::AfterGate { qubits, .. } = event else {
            return NoiseResponse::None;
        };

        let mut gates = Vec::new();

        for &qubit in *qubits {
            // Use qubit state from context
            let state = ctx.qubit_state(qubit);

            let rate = match state {
                Some(QubitState::Leaked) => self.leaked_rate,
                _ => self.active_rate,
            };

            if rng.random::<f64>() < rate {
                gates.push(GateCommand::z(qubit));
            }
        }

        if gates.is_empty() {
            NoiseResponse::None
        } else {
            NoiseResponse::inject_gates(gates.into())
        }
    }

    fn name(&self) -> &'static str {
        "ContextAwareChannel"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

fn example_context_aware_noise() {
    println!("--- Context-Aware Noise ---");

    // Different error rates based on qubit state (active vs leaked)
    let shots = 1000;

    // Test with active qubits (normal circuit)
    let active_noise = ComposableNoiseModel::new()
        .add_plugin(&CorePlugin)
        .add_channel(ContextAwareChannel::new(0.05, 0.50)); // 5% active, 50% leaked

    let commands = CommandBuilder::new()
        .pz(&[0])
        .h(&[0])
        .h(&[0]) // Should return to |0⟩
        .mz(&[0])
        .build();

    let mut state = SparseStab::new(1);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(active_noise)
        .with_seed(42);

    let mut active_errors = 0;
    for _ in 0..shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            active_errors += 1;
        }
    }

    // Demonstrate the channel checks context - here we just show the active case
    // In a real scenario, leaked qubits would get the higher error rate
    println!(
        "  Active qubits (5% rate): {:.1}% errors",
        f64::from(active_errors) / f64::from(shots) * 100.0
    );
    println!("  (Leaked qubits would get 50% rate - higher error due to heating)");
    println!("  (This demonstrates using NoiseContext to make state-dependent decisions)");
    println!();
}

// --- Example 5: Built-in GateDependentChannel ---

/// Demonstrate the built-in `GateDependentChannel` for gate-specific error rates.
fn example_builtin_gate_dependent() {
    use pecos_neo::noise::{GateDependentChannel, GateNoiseConfig};

    println!("--- Built-in GateDependentChannel ---");

    let shots = 1000;

    // Create a gate-dependent channel with different error rates per gate
    let gate_noise = GateDependentChannel::new()
        .with_gate_error(GateType::H, 0.20) // 20% error on H gates
        .with_gate_error(GateType::SZ, 0.02) // 2% error on SZ gates
        .with_gate(
            GateType::CX,
            GateNoiseConfig::new(0.10).with_pauli_weights(PauliWeights::z_biased(0.8)),
        ) // 10% Z-biased error on CX
        .with_default(0.05); // 5% default for unlisted gates

    // Test H gate (high error rate)
    let h_commands = CommandBuilder::new()
        .pz(&[0])
        .h(&[0])
        .h(&[0]) // H^2 = I
        .mz(&[0])
        .build();

    let noise = ComposableNoiseModel::new()
        .add_plugin(&CorePlugin)
        .add_channel(gate_noise.clone());

    let mut state = SparseStab::new(1);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise)
        .with_seed(42);

    let mut h_errors = 0;
    for _ in 0..shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &h_commands).unwrap();
        if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            h_errors += 1;
        }
    }

    // Test SZ gate (low error rate)
    let sz_commands = CommandBuilder::new()
        .pz(&[0])
        .sz(&[0])
        .szdg(&[0]) // SZ * SZ^dag = I
        .mz(&[0])
        .build();

    let noise = ComposableNoiseModel::new()
        .add_plugin(&CorePlugin)
        .add_channel(gate_noise);

    let mut state = SparseStab::new(1);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise)
        .with_seed(42);

    let mut sz_errors = 0;
    for _ in 0..shots {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &sz_commands).unwrap();
        if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            sz_errors += 1;
        }
    }

    println!("  Using built-in GateDependentChannel:");
    println!(
        "  H gate (20% rate): {:.1}% errors",
        f64::from(h_errors) / f64::from(shots) * 100.0
    );
    println!(
        "  SZ gate (2% rate): {:.1}% errors",
        f64::from(sz_errors) / f64::from(shots) * 100.0
    );
    println!("  (Built-in channel provides same functionality with cleaner API)");
    println!();
}

// --- Example 6: Built-in CorrelatedNoiseChannel ---

/// Demonstrate the built-in `CorrelatedNoiseChannel` for spatially correlated errors.
fn example_builtin_correlated() {
    use pecos_neo::noise::CorrelatedNoiseChannel;

    println!("--- Built-in CorrelatedNoiseChannel ---");

    let shots = 2000;

    for correlation in [0.0, 0.5, 0.9] {
        let noise = ComposableNoiseModel::new()
            .add_plugin(&CorePlugin)
            .add_channel(CorrelatedNoiseChannel::new(0.1, correlation));

        // Bell state circuit
        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .h(&[0])
            .cx(&[(0, 1)])
            .mz(&[0])
            .mz(&[1])
            .build();

        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);

        let mut both_same = 0;
        let mut both_different = 0;

        for _ in 0..shots {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            let q0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
            let q1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);

            if q0 == q1 {
                both_same += 1;
            } else {
                both_different += 1;
            }
        }

        println!(
            "  correlation={:.1}: {:.1}% correlated, {:.1}% anti-correlated",
            correlation,
            f64::from(both_same) / f64::from(shots) * 100.0,
            f64::from(both_different) / f64::from(shots) * 100.0
        );
    }
    println!("  (Higher correlation -> more correlated errors -> less anti-correlation)");
    println!("  (Built-in CorrelatedNoiseChannel provides easy spatial correlation)");
    println!();
}
