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

//! Differential tests for `PerGateTypeNoise::to_neo_channel`.
//!
//! Each test configures noise through the pecos-qec type (qec orderings
//! and conventions), converts, runs circuit-level Monte Carlo on the neo
//! stack, and checks the measured rates against the analytic values the
//! qec configuration implies. The two-qubit cells are chosen to fail
//! loudly if the `PAULI_2Q_ORDER` -> `TWO_QUBIT_PAULIS` permutation or
//! the qubit-pair orientation ever drifts.

#![cfg(feature = "neo")]

use pecos_core::QubitId;
use pecos_core::gate_type::GateType;
use pecos_neo::noise::PerGatePauliChannel;
use pecos_neo::prelude::*;
use pecos_qec::fault_tolerance::dem_builder::{NoiseConfig, PerGateTypeNoise};
use pecos_simulators::SparseStab;

const SHOTS: usize = 20_000;

/// Rate of outcome-1 on one qubit over `SHOTS` runs of a circuit.
#[allow(clippy::cast_precision_loss)]
fn one_rate(noise: &PerGateTypeNoise, commands: &CommandQueue, qubit: usize) -> f64 {
    let model = ComposableNoiseModel::new().add_channel(noise.to_neo_channel());
    let mut state = SparseStab::new(2);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(model)
        .with_seed(42);
    let qubits = [QubitId(qubit)];
    let mut ones = 0usize;
    for _ in 0..SHOTS {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, commands).unwrap();
        if let Some(bits) = outcomes.bitstring(&qubits)
            && bits[0]
        {
            ones += 1;
        }
    }
    ones as f64 / SHOTS as f64
}

#[allow(clippy::cast_precision_loss)]
fn five_sigma(p: f64) -> f64 {
    5.0 * (p * (1.0 - p) / SHOTS as f64).sqrt()
}

/// Index of a Pauli-pair label in qec's `PAULI_2Q_ORDER`.
fn qec_2q_index(label: &str) -> usize {
    pecos_qec::fault_tolerance::dem_builder::PAULI_2Q_ORDER
        .iter()
        .position(|&entry| entry == label)
        .expect("valid Pauli pair label")
}

#[test]
fn per_gate_1q_rates_map_with_analytic_flip_rate() {
    // 30% X-error on X gates (qec [X, Y, Z] ordering): the injected X
    // cancels the gate, so P(outcome = 0) = 0.3.
    let noise = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.0))
        .with_1q_rates(GateType::X, [0.3, 0.0, 0.0]);

    let commands = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();
    let rate = 1.0 - one_rate(&noise, &commands, 0);
    assert!(
        (rate - 0.3).abs() < five_sigma(0.3),
        "1q gate rate: got flip rate {rate}, expected 0.3"
    );
}

#[test]
fn per_qubit_1q_rates_override_per_gate() {
    let noise = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.0))
        .with_1q_rates(GateType::X, [0.3, 0.0, 0.0])
        .with_1q_rates_for_qubit(GateType::X, QubitId(0), [0.0; 3]);

    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .x(&[0])
        .x(&[1])
        .mz(&[0])
        .mz(&[1])
        .build();

    let flip0 = 1.0 - one_rate(&noise, &commands, 0);
    let flip1 = 1.0 - one_rate(&noise, &commands, 1);
    assert!(
        flip0.abs() < f64::EPSILON,
        "qubit 0 override must be noiseless, got {flip0}"
    );
    assert!(
        (flip1 - 0.3).abs() < five_sigma(0.3),
        "qubit 1 keeps the per-gate rate: got {flip1}, expected 0.3"
    );
}

#[test]
fn qec_2q_ordering_is_permuted_correctly() {
    // Configure, IN QEC ORDERING, a 25% "IX" error on CX: identity on the
    // first (control) qubit, X on the second (target). If the permutation
    // into neo's ordering drifted, the error would land on the wrong
    // Pauli pair and the wrong qubit would flip.
    let mut rates = [0.0; 15];
    rates[qec_2q_index("IX")] = 0.25;
    let noise = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.0))
        .with_2q_rates(GateType::CX, rates);

    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    let rate0 = one_rate(&noise, &commands, 0);
    let rate1 = one_rate(&noise, &commands, 1);
    assert!(
        rate0.abs() < f64::EPSILON,
        "control qubit must be untouched by IX, got {rate0}"
    );
    assert!(
        (rate1 - 0.25).abs() < five_sigma(0.25),
        "target qubit must flip at the IX rate, got {rate1}"
    );
}

#[test]
fn qec_2q_per_pair_rates_override_per_gate() {
    // Per-gate: 20% "XI" (control flips). Per-pair override for (0, 1):
    // 20% "IX" (target flips). The override must win for that pair.
    let mut gate_rates = [0.0; 15];
    gate_rates[qec_2q_index("XI")] = 0.2;
    let mut pair_rates = [0.0; 15];
    pair_rates[qec_2q_index("IX")] = 0.2;
    let noise = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.0))
        .with_2q_rates(GateType::CX, gate_rates)
        .with_2q_rates_for_qubits(GateType::CX, QubitId(0), QubitId(1), pair_rates);

    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    let rate0 = one_rate(&noise, &commands, 0);
    let rate1 = one_rate(&noise, &commands, 1);
    assert!(
        rate0.abs() < f64::EPSILON,
        "pair override replaces the per-gate XI error, got control rate {rate0}"
    );
    assert!(
        (rate1 - 0.2).abs() < five_sigma(0.2),
        "pair override applies IX to the target: got {rate1}, expected 0.2"
    );
}

#[test]
fn measurement_and_init_rates_map_with_per_qubit_overrides() {
    // p_meas/p_init are seeded from the base config's p_meas/p_prep.
    let base = NoiseConfig {
        p_meas: 0.1,
        ..NoiseConfig::uniform(0.0)
    };
    let noise = PerGateTypeNoise::from_base_noise(base).with_measurement_rate(QubitId(1), 0.4);

    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .mz(&[0])
        .mz(&[1])
        .build();

    let rate0 = one_rate(&noise, &commands, 0);
    let rate1 = one_rate(&noise, &commands, 1);
    assert!(
        (rate0 - 0.1).abs() < five_sigma(0.1),
        "default meas rate: got {rate0}, expected 0.1"
    );
    assert!(
        (rate1 - 0.4).abs() < five_sigma(0.4),
        "per-qubit meas rate: got {rate1}, expected 0.4"
    );
}

#[test]
fn base_noise_back_fills_unlisted_gates() {
    // base p1 = 0.3 -> uniform per-Pauli 0.1; X and Y flip the outcome
    // after an X gate: P(outcome = 1) = 0.8.
    let noise = PerGateTypeNoise::from_base_noise(NoiseConfig {
        p1: 0.3,
        ..NoiseConfig::uniform(0.0)
    });

    let commands = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();
    let rate = one_rate(&noise, &commands, 0);
    assert!(
        (rate - 0.8).abs() < five_sigma(0.8),
        "base fallback: got outcome-1 rate {rate}, expected 0.8"
    );
}

#[test]
#[should_panic(expected = "cannot carry idle noise")]
fn idle_configuration_is_rejected_not_dropped() {
    // NoiseConfig::uniform sets p_idle = p, and the DEM built from this
    // config includes idle contributions; silently dropping them in the
    // conversion would change the physics.
    let noise = PerGateTypeNoise::from_base_noise(NoiseConfig {
        p_idle: 0.001,
        ..NoiseConfig::uniform(0.0)
    });
    let _ = noise.to_neo_channel();
}

#[test]
#[should_panic(expected = "cannot carry idle noise")]
fn idle_gate_entries_are_rejected_not_dropped() {
    let noise = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.0))
        .with_1q_rates(GateType::Idle, [0.001, 0.0, 0.0]);
    let _ = noise.to_neo_channel();
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn default_noise_config_carries_realistic_base_rates_without_idle_panic() {
    // The realistic-nonzero-defaults trap: NoiseConfig::default() is 0.01
    // EVERYWHERE (p1/p2/p_meas/p_prep), not off — it bit the GNM and qec
    // mappings before. Lock that to_neo_channel on the default config
    // (a) does NOT trip the idle guard (default has p_idle = 0, t1/t2 = None,
    // so this call would panic if it did) and (b) carries the 0.01
    // base/meas/init rates EXACTLY — bit-identical to the neo channel built
    // by hand with those values (a mishandling that dropped or rescaled the
    // defaults would diverge).
    let from_default = PerGateTypeNoise::from_base_noise(NoiseConfig::default()).to_neo_channel();
    let by_hand = PerGatePauliChannel::new()
        .with_base(0.01, 0.01)
        .with_meas_init(0.01, 0.01);

    let commands = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

    let count_ones = |channel: PerGatePauliChannel| -> usize {
        let model = ComposableNoiseModel::new().add_channel(channel);
        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(model)
            .with_seed(42);
        let qubits = [QubitId(0)];
        let mut ones = 0usize;
        for _ in 0..SHOTS {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            if let Some(bits) = outcomes.bitstring(&qubits)
                && bits[0]
            {
                ones += 1;
            }
        }
        ones
    };

    let default_ones = count_ones(from_default);
    let hand_ones = count_ones(by_hand);

    assert_eq!(
        default_ones, hand_ones,
        "to_neo_channel(NoiseConfig::default()) must carry the 0.01 base/meas/init rates \
         exactly (got {default_ones} vs hand-built {hand_ones})"
    );
    // The defaults are NOT silently dropped: the circuit's nominal outcome
    // is 1 (X flips |0> to |1>), so the error rate is the fraction reading
    // 0 — a small but nonzero value from the combined 0.01 prep/gate/meas
    // sources (~0.027), confirming the defaults carry rather than vanish.
    let error_rate = 1.0 - default_ones as f64 / SHOTS as f64;
    assert!(
        error_rate > 0.0 && error_rate < 0.1,
        "the realistic 0.01 defaults must produce a small nonzero error rate, got {error_rate}"
    );
}
