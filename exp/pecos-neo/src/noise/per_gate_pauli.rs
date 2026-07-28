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

//! Per-gate-type, optionally per-qubit Pauli noise.
//!
//! Mirrors pecos-qec's `PerGateTypeNoise` layered lookup so circuit-level
//! Monte Carlo on this stack can run the same noise that drives DEM
//! generation:
//!
//! ```text
//! 1. per-(gate, qubit) rates        // most specific
//! 2. per-gate-type rates
//! 3. base p1/3 (or p2/15) uniform   // fallback
//! ```
//!
//! Rates are absolute per-Pauli probabilities (NOT normalized weights):
//! a `[f64; 3]` entry is `[P(X), P(Y), P(Z)]` and the gate's total error
//! probability is the sum. Two-qubit arrays follow [`TWO_QUBIT_PAULIS`]
//! ordering. Idle noise is out of scope here — compose with
//! [`super::idle::IdleChannel`] for T1/T2 effects.

use super::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse, TWO_QUBIT_PAULIS};
use crate::command::{GateCommand, GateType};
use pecos_core::QubitId;
use pecos_random::PecosRng;
use rand::RngExt;
use smallvec::SmallVec;
use std::collections::BTreeMap;

/// Per-gate-type Pauli channel with per-qubit overrides.
///
/// See the module docs for the lookup order and rate conventions.
#[derive(Debug, Clone, Default)]
pub struct PerGatePauliChannel {
    rates_1q: BTreeMap<GateType, [f64; 3]>,
    rates_2q: BTreeMap<GateType, [f64; 15]>,
    rates_1q_per_qubit: BTreeMap<(GateType, QubitId), [f64; 3]>,
    rates_2q_per_qubits: BTreeMap<(GateType, QubitId, QubitId), [f64; 15]>,
    measurement_rates: BTreeMap<QubitId, f64>,
    init_rates: BTreeMap<QubitId, f64>,
    p_meas: f64,
    p_init: f64,
    base_p1: f64,
    base_p2: f64,
}

fn assert_probability_vector(rates: &[f64], what: &str) {
    let total: f64 = rates.iter().sum();
    assert!(
        rates.iter().all(|&r| r >= 0.0) && total <= 1.0 + 1e-9,
        "{what} must be non-negative probabilities summing to at most 1.0, got total {total}"
    );
}

impl PerGatePauliChannel {
    /// Create an empty channel: no per-gate entries, zero base rates.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the uniform base rates used for gates with no explicit entry.
    ///
    /// `p1`/`p2` are total error probabilities, split uniformly over the
    /// 3 (15) Paulis, matching pecos-qec's `NoiseConfig` fallback.
    #[must_use]
    pub fn with_base(mut self, p1: f64, p2: f64) -> Self {
        assert_probability_vector(&[p1], "base p1");
        assert_probability_vector(&[p2], "base p2");
        self.base_p1 = p1;
        self.base_p2 = p2;
        self
    }

    /// Set the default measurement and preparation error rates.
    ///
    /// The measurement error is a physical X injected before readout (it
    /// propagates into the post-measurement state, matching the DEM's
    /// measurement-fault convention), and the preparation error is a
    /// physical X after prep.
    #[must_use]
    pub fn with_meas_init(mut self, p_meas: f64, p_init: f64) -> Self {
        assert_probability_vector(&[p_meas], "p_meas");
        assert_probability_vector(&[p_init], "p_init");
        self.p_meas = p_meas;
        self.p_init = p_init;
        self
    }

    /// Attach `[P(X), P(Y), P(Z)]` rates for a 1q gate type on any qubit.
    #[must_use]
    pub fn with_1q_rates(mut self, gate: GateType, rates: [f64; 3]) -> Self {
        assert_probability_vector(&rates, "1q rates");
        self.rates_1q.insert(gate, rates);
        self
    }

    /// Attach rates for a 2q gate type on any qubit pair, ordered by
    /// [`TWO_QUBIT_PAULIS`].
    #[must_use]
    pub fn with_2q_rates(mut self, gate: GateType, rates: [f64; 15]) -> Self {
        assert_probability_vector(&rates, "2q rates");
        self.rates_2q.insert(gate, rates);
        self
    }

    /// Attach `[P(X), P(Y), P(Z)]` rates for a 1q gate type on one qubit.
    /// Takes precedence over [`Self::with_1q_rates`] for that pair.
    #[must_use]
    pub fn with_1q_rates_for_qubit(
        mut self,
        gate: GateType,
        qubit: QubitId,
        rates: [f64; 3],
    ) -> Self {
        assert_probability_vector(&rates, "1q per-qubit rates");
        self.rates_1q_per_qubit.insert((gate, qubit), rates);
        self
    }

    /// Attach rates for a 2q gate type on one ordered qubit pair, ordered
    /// by [`TWO_QUBIT_PAULIS`]. Takes precedence over
    /// [`Self::with_2q_rates`] for that combination.
    #[must_use]
    pub fn with_2q_rates_for_qubits(
        mut self,
        gate: GateType,
        first: QubitId,
        second: QubitId,
        rates: [f64; 15],
    ) -> Self {
        assert_probability_vector(&rates, "2q per-qubit rates");
        self.rates_2q_per_qubits
            .insert((gate, first, second), rates);
        self
    }

    /// Set a per-qubit measurement X-flip probability (overrides the
    /// default from [`Self::with_meas_init`] for that qubit).
    #[must_use]
    pub fn with_meas_rate_for_qubit(mut self, qubit: QubitId, p: f64) -> Self {
        assert_probability_vector(&[p], "per-qubit meas rate");
        self.measurement_rates.insert(qubit, p);
        self
    }

    /// Set a per-qubit preparation X-error probability (overrides the
    /// default from [`Self::with_meas_init`] for that qubit).
    #[must_use]
    pub fn with_init_rate_for_qubit(mut self, qubit: QubitId, p: f64) -> Self {
        assert_probability_vector(&[p], "per-qubit init rate");
        self.init_rates.insert(qubit, p);
        self
    }

    /// Resolve the 1q rates for a gate on a qubit via the layered lookup.
    fn rates_1q_for(&self, gate: GateType, qubit: QubitId) -> [f64; 3] {
        if let Some(rates) = self.rates_1q_per_qubit.get(&(gate, qubit)) {
            return *rates;
        }
        if let Some(rates) = self.rates_1q.get(&gate) {
            return *rates;
        }
        [self.base_p1 / 3.0; 3]
    }

    /// Resolve the 2q rates for a gate on an ordered pair.
    fn rates_2q_for(&self, gate: GateType, first: QubitId, second: QubitId) -> [f64; 15] {
        if let Some(rates) = self.rates_2q_per_qubits.get(&(gate, first, second)) {
            return *rates;
        }
        if let Some(rates) = self.rates_2q.get(&gate) {
            return *rates;
        }
        [self.base_p2 / 15.0; 15]
    }

    fn has_any_gate_noise(&self) -> bool {
        self.base_p1 > 0.0
            || self.base_p2 > 0.0
            || !self.rates_1q.is_empty()
            || !self.rates_2q.is_empty()
            || !self.rates_1q_per_qubit.is_empty()
            || !self.rates_2q_per_qubits.is_empty()
    }

    fn apply_after_gate(
        &self,
        gate_type: GateType,
        qubits: &[QubitId],
        ctx: &NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        if ctx.is_noiseless(gate_type) {
            return NoiseResponse::None;
        }

        let mut gates: SmallVec<[GateCommand; 4]> = SmallVec::new();
        if gate_type.is_single_qubit() {
            for &qubit in qubits {
                if ctx.is_leaked(qubit) {
                    continue;
                }
                let [px, py, pz] = self.rates_1q_for(gate_type, qubit);
                let r = rng.random::<f64>();
                let pauli = if r < px {
                    GateType::X
                } else if r < px + py {
                    GateType::Y
                } else if r < px + py + pz {
                    GateType::Z
                } else {
                    continue;
                };
                gates.push(GateCommand::new(pauli, smallvec::smallvec![qubit]));
            }
        } else if qubits.len() == 2 {
            let (first, second) = (qubits[0], qubits[1]);
            if !ctx.is_leaked(first) && !ctx.is_leaked(second) {
                let rates = self.rates_2q_for(gate_type, first, second);
                let r = rng.random::<f64>();
                let mut cumulative = 0.0;
                for (idx, &p) in rates.iter().enumerate() {
                    cumulative += p;
                    if r < cumulative {
                        let (pauli0, pauli1) = TWO_QUBIT_PAULIS[idx];
                        if pauli0 != GateType::I {
                            gates.push(GateCommand::new(pauli0, smallvec::smallvec![first]));
                        }
                        if pauli1 != GateType::I {
                            gates.push(GateCommand::new(pauli1, smallvec::smallvec![second]));
                        }
                        break;
                    }
                }
            }
        }

        if gates.is_empty() {
            NoiseResponse::None
        } else {
            NoiseResponse::inject_gates(gates)
        }
    }

    fn apply_before_measurement(
        &self,
        qubits: &[QubitId],
        ctx: &NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        // The DEM models a measurement error as a physical X at the
        // measurement location (see pecos-qec's
        // `process_meas_fault_source_tracked`: "Measurement error is a
        // bit flip (X error)"), which propagates into the post-measurement
        // state. Inject X before readout to match that convention, so this
        // channel runs the same measurement physics the DEM encodes —
        // a re-measured (un-reset) qubit flips at 2p(1-p), not p.
        let mut gates: SmallVec<[GateCommand; 4]> = SmallVec::new();
        for &qubit in qubits {
            if ctx.is_leaked(qubit) {
                continue;
            }
            let p = *self.measurement_rates.get(&qubit).unwrap_or(&self.p_meas);
            if p > 0.0 && rng.random::<f64>() < p {
                gates.push(GateCommand::new(GateType::X, smallvec::smallvec![qubit]));
            }
        }
        if gates.is_empty() {
            NoiseResponse::None
        } else {
            NoiseResponse::inject_gates(gates)
        }
    }

    fn apply_after_preparation(
        &self,
        qubits: &[QubitId],
        ctx: &NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let mut gates: SmallVec<[GateCommand; 4]> = SmallVec::new();
        for &qubit in qubits {
            if ctx.is_leaked(qubit) {
                continue;
            }
            let p = *self.init_rates.get(&qubit).unwrap_or(&self.p_init);
            if p > 0.0 && rng.random::<f64>() < p {
                gates.push(GateCommand::new(GateType::X, smallvec::smallvec![qubit]));
            }
        }
        if gates.is_empty() {
            NoiseResponse::None
        } else {
            NoiseResponse::inject_gates(gates)
        }
    }
}

impl NoiseChannel for PerGatePauliChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        match event {
            NoiseEvent::AfterGate { gate_type, .. } => {
                gate_type.is_unitary_gate()
                    && (gate_type.is_single_qubit() || gate_type.is_two_qubit())
                    && self.has_any_gate_noise()
            }
            NoiseEvent::BeforeMeasurement { .. } => {
                self.p_meas > 0.0 || !self.measurement_rates.is_empty()
            }
            NoiseEvent::AfterPreparation { .. } => self.p_init > 0.0 || !self.init_rates.is_empty(),
            _ => false,
        }
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        match event {
            NoiseEvent::AfterGate {
                gate_type, qubits, ..
            } => self.apply_after_gate(*gate_type, qubits, ctx, rng),
            NoiseEvent::BeforeMeasurement { qubits } => {
                self.apply_before_measurement(qubits, ctx, rng)
            }
            NoiseEvent::AfterPreparation { qubits } => {
                self.apply_after_preparation(qubits, ctx, rng)
            }
            _ => NoiseResponse::None,
        }
    }

    fn name(&self) -> &'static str {
        "PerGatePauliChannel"
    }

    fn priority(&self) -> i32 {
        10
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
// statistical tests use count as f64
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::noise::ComposableNoiseModel;
    use crate::prelude::*;
    use pecos_simulators::SparseStab;

    const SHOTS: usize = 20_000;

    fn flip_rate(model: ComposableNoiseModel, commands: &CommandQueue, qubit: usize) -> f64 {
        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(model)
            .with_seed(42);
        let qubits = [QubitId(qubit)];
        let mut flips = 0usize;
        for _ in 0..SHOTS {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, commands).unwrap();
            if let Some(bits) = outcomes.bitstring(&qubits)
                && bits[0]
            {
                flips += 1;
            }
        }
        flips as f64 / SHOTS as f64
    }

    fn five_sigma(p: f64) -> f64 {
        5.0 * (p * (1.0 - p) / SHOTS as f64).sqrt()
    }

    #[test]
    fn per_gate_rates_apply_to_that_gate_only() {
        // X-only error on H; the X gate stays noiseless.
        let channel = PerGatePauliChannel::new().with_1q_rates(GateType::H, [0.2, 0.0, 0.0]);
        let model = ComposableNoiseModel::new().add_channel(channel);

        // H twice returns to |0>; an injected X after either H flips the
        // outcome unless both Hs get one (prob 0.2*0.2 cancels via parity:
        // P(flip) = 2*0.2*0.8 = 0.32).
        let commands = CommandBuilder::new()
            .pz(&[0])
            .h(&[0])
            .h(&[0])
            .mz(&[0])
            .build();
        let rate = flip_rate(model.clone(), &commands, 0);
        // X after the second H flips Z-outcome directly; X after the first H
        // becomes Z through the second H... careful: track exactly.
        // |0> -H-> |+> -X-> |+> (X|+> = |+>): an X after the FIRST H does
        // nothing to the final outcome. After the second H the state is |0>,
        // X flips it. So P(flip) = 0.2 exactly.
        assert!(
            (rate - 0.2).abs() < five_sigma(0.2),
            "H-specific X rate: got {rate}, expected 0.2"
        );

        // The X gate has no entry and base is zero: noiseless.
        let channel = PerGatePauliChannel::new().with_1q_rates(GateType::H, [0.2, 0.0, 0.0]);
        let model = ComposableNoiseModel::new().add_channel(channel);
        let commands = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();
        let rate = flip_rate(model, &commands, 0);
        assert!(
            (rate - 1.0).abs() < f64::EPSILON,
            "X gate must stay noiseless, got outcome-1 rate {rate}"
        );
    }

    #[test]
    fn per_qubit_rates_override_per_gate_rates() {
        // Gate-level: 30% X error on X gates; qubit 0 override: 0%.
        let channel = PerGatePauliChannel::new()
            .with_1q_rates(GateType::X, [0.3, 0.0, 0.0])
            .with_1q_rates_for_qubit(GateType::X, QubitId(0), [0.0; 3]);
        let model = ComposableNoiseModel::new().add_channel(channel);

        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .x(&[0])
            .x(&[1])
            .mz(&[0])
            .mz(&[1])
            .build();

        // Qubit 0: override says noiseless -> outcome always 1.
        let rate0 = 1.0 - flip_rate(model.clone(), &commands, 0);
        assert!(
            rate0.abs() < f64::EPSILON,
            "qubit 0 override must be noiseless, got flip rate {rate0}"
        );
        // Qubit 1: per-gate 30% X error cancels the X gate -> outcome 0.
        let rate1 = 1.0 - flip_rate(model, &commands, 1);
        assert!(
            (rate1 - 0.3).abs() < five_sigma(0.3),
            "qubit 1 per-gate rate: got {rate1}, expected 0.3"
        );
    }

    #[test]
    fn two_qubit_orientation_first_pauli_hits_first_qubit() {
        // XI-only error on CX(0, 1): qubit 0 flips at 25%, qubit 1 never
        // (after CX from |00>, X on control would propagate only if it
        // happened BEFORE the gate; injection is after, so it stays local).
        let mut rates = [0.0; 15];
        rates[0] = 0.25; // (X, I) in TWO_QUBIT_PAULIS order
        let channel = PerGatePauliChannel::new().with_2q_rates(GateType::CX, rates);
        let model = ComposableNoiseModel::new().add_channel(channel);

        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .cx(&[(0, 1)])
            .mz(&[0])
            .mz(&[1])
            .build();

        let rate0 = flip_rate(model.clone(), &commands, 0);
        let rate1 = flip_rate(model, &commands, 1);
        assert!(
            (rate0 - 0.25).abs() < five_sigma(0.25),
            "first qubit must flip at the XI rate, got {rate0}"
        );
        assert!(
            rate1.abs() < f64::EPSILON,
            "second qubit must be untouched by XI, got {rate1}"
        );
    }

    #[test]
    fn measurement_and_init_rates_with_per_qubit_overrides() {
        let channel = PerGatePauliChannel::new()
            .with_meas_init(0.1, 0.0)
            .with_meas_rate_for_qubit(QubitId(1), 0.4);
        let model = ComposableNoiseModel::new().add_channel(channel);

        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .mz(&[0])
            .mz(&[1])
            .build();

        let rate0 = flip_rate(model.clone(), &commands, 0);
        let rate1 = flip_rate(model, &commands, 1);
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
    fn measurement_error_propagates_to_a_re_measured_qubit() {
        // The measurement error is a physical X before readout (DEM
        // convention), so measuring the same qubit twice without a reset
        // sees the second outcome flipped at 2p(1-p), not p. A record-only
        // flip (the bug this guards) would give p.
        let p = 0.25;
        let channel = PerGatePauliChannel::new().with_meas_init(p, 0.0);
        let model = ComposableNoiseModel::new().add_channel(channel);

        let commands = CommandBuilder::new().pz(&[0]).mz(&[0]).mz(&[0]).build();
        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(model)
            .with_seed(42);
        let mut second_ones = 0usize;
        for _ in 0..SHOTS {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            let bits: Vec<bool> = outcomes.iter().map(|o| o.outcome).collect();
            assert_eq!(bits.len(), 2);
            if bits[1] {
                second_ones += 1;
            }
        }
        let rate = second_ones as f64 / SHOTS as f64;
        let expected = 2.0 * p * (1.0 - p);
        assert!(
            (rate - expected).abs() < five_sigma(expected),
            "second-measure rate: got {rate}, expected {expected}"
        );
    }

    #[test]
    fn base_rates_back_fill_unlisted_gates() {
        // base p1 = 0.3 -> uniform X/Y/Z at 0.1 each; X and Y flip the
        // Z-basis outcome after an X gate, so P(outcome = 1) = 0.8.
        let channel = PerGatePauliChannel::new().with_base(0.3, 0.0);
        let model = ComposableNoiseModel::new().add_channel(channel);
        let commands = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();
        let rate = flip_rate(model, &commands, 0);
        assert!(
            (rate - 0.8).abs() < five_sigma(0.8),
            "base fallback: got outcome-1 rate {rate}, expected 0.8"
        );
    }

    #[test]
    #[should_panic(expected = "must be non-negative probabilities")]
    fn rejects_rates_summing_above_one() {
        let _ = PerGatePauliChannel::new().with_1q_rates(GateType::H, [0.5, 0.4, 0.3]);
    }
}
