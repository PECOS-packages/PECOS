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

//! Leakage tracking channel.
//!
//! Suppresses gates on leaked operands, including noiseless gates.
//! Optional partner Pauli faults default to zero and respect noiseless operations.
//! Leaked readout is owned by `CircuitRunner::record_measurements`, which reports 1.
//!
//! This channel should ALWAYS be added to a noise model that tracks leakage,
//! as it handles fundamental leakage behavior independent of any noise rates.

use super::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse};
use crate::command::{GateCommand, GateType};
use crate::noise::validation::{ValidationResult, validate_probability};
use pecos_random::PecosRng;
use rand::RngExt;
use smallvec::SmallVec;

/// Channel that handles the effects of leaked qubits.
///
/// This channel handles gate suppression and optional partner faults:
///
/// 1. **`BeforeGate`**: Skip the operand groups that contain a leaked qubit
/// 2. **`AfterGate`**: For two-qubit gates where one qubit is leaked,
///    optionally apply Pauli noise to the non-leaked partner (default probability 0).
///
/// It never changes measurement records. `CircuitRunner::record_measurements`
/// reports 1 for leaked readout. Noiseless gates still get suppressed, but never
/// produce partner faults. `AfterGate` is subscribed only with a positive partner rate.
///
/// ## Important
///
/// This channel should be added to any noise model that uses leakage,
/// even if no other noise is desired. Without it, gates on leaked qubits
/// would incorrectly be applied to the quantum state.
///
/// ## Leakage Scale
///
/// The `leakage_scale` parameter controls how `MarkLeaked` responses from
/// other channels are interpreted:
/// - 1.0: All leakage events remain as leakage
/// - 0.0: All leakage events become depolarizing noise instead
/// - Between: Probabilistic conversion
#[derive(Debug, Clone)]
pub struct LeakageChannel {
    /// Scale leakage events to depolarizing events.
    ///
    /// 0.0 = all leakage becomes depolarizing, 1.0 = all leakage remains leakage.
    leakage_scale: f64,
    partner_pauli_probability: f64,
    partner_pauli_threshold: u64,
}

impl Default for LeakageChannel {
    fn default() -> Self {
        Self {
            leakage_scale: 1.0,
            partner_pauli_probability: 0.0,
            partner_pauli_threshold: 0,
        }
    }
}

impl LeakageChannel {
    /// Return the configured `leakage_scale`.
    #[must_use]
    pub fn leakage_scale(&self) -> f64 {
        self.leakage_scale
    }

    /// Create a leakage channel with full leakage (no scaling).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a leakage channel that converts all leakage to depolarizing.
    #[must_use]
    pub fn no_leakage() -> Self {
        Self {
            leakage_scale: 0.0,
            ..Self::default()
        }
    }

    /// Opt into Pauli noise on the healthy partner of a leaked pair.
    ///
    /// Defaults to zero. Noiseless operations produce no partner fault or draw.
    ///
    /// # Errors
    /// Returns an error if `probability` is not finite or outside `[0, 1]`.
    pub fn with_partner_pauli_probability(mut self, probability: f64) -> ValidationResult<Self> {
        validate_probability(probability, "partner_pauli_probability")?;
        self.partner_pauli_probability = probability;
        self.partner_pauli_threshold = PecosRng::probability_threshold(probability);
        Ok(self)
    }

    /// Set the leakage scale factor.
    ///
    /// # Errors
    /// Returns an error for invalid probabilities or weights.
    pub fn with_scale(mut self, scale: f64) -> ValidationResult<Self> {
        validate_probability(scale, "scale")?;
        self.leakage_scale = scale;
        Ok(self)
    }
}

impl NoiseChannel for LeakageChannel {
    fn event_kinds(&self) -> super::EventKinds {
        let kinds = super::EventKinds::of(super::NoiseEventKind::BeforeGate);
        if self.partner_pauli_probability > 0.0 {
            kinds.with(super::NoiseEventKind::AfterGate)
        } else {
            kinds
        }
    }

    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        matches!(event, NoiseEvent::BeforeGate { .. })
            || (self.partner_pauli_probability > 0.0
                && matches!(event, NoiseEvent::AfterGate { .. }))
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        match event {
            NoiseEvent::BeforeGate {
                gate_type, qubits, ..
            } => Self::handle_before_gate(*gate_type, qubits, ctx),
            NoiseEvent::AfterGate { gate_type, .. } => {
                if self.partner_pauli_probability <= 0.0 || ctx.is_noiseless_operation(event) {
                    return NoiseResponse::None;
                }
                self.handle_after_gate(*gate_type, &event.gated_qubits(), ctx, rng)
            }
            _ => NoiseResponse::None,
        }
    }

    fn name(&self) -> &'static str {
        "LeakageChannel"
    }

    fn priority(&self) -> i32 {
        // High priority - leakage checks should happen before other noise channels
        100
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

impl LeakageChannel {
    /// Handle `BeforeGate` - skip operand groups containing leaked qubits.
    ///
    /// Note: Measurements and preparations are NOT skipped for leaked qubits.
    /// Measurements have special handling in the runner to return appropriate
    /// outcomes for leaked qubits.
    fn handle_before_gate(
        gate_type: GateType,
        qubits: &[pecos_core::QubitId],
        ctx: &NoiseContext,
    ) -> NoiseResponse {
        // Don't skip measurements or preparations - they have special handling
        if gate_type.is_measurement() || gate_type.is_preparation() {
            return NoiseResponse::None;
        }

        // If any qubit is leaked, skip its affected operand groups
        // Uses optimized any_leaked which has O(1) fast path when leaked_count == 0
        if ctx.any_leaked(qubits) {
            return NoiseResponse::SkipGate(
                qubits
                    .iter()
                    .copied()
                    .filter(|&q| ctx.is_leaked(q))
                    .collect(),
            );
        }
        NoiseResponse::None
    }

    /// Handle `AfterGate` - for two-qubit gates where one qubit is leaked,
    /// apply depolarizing noise to the non-leaked partner.
    ///
    /// Uses leakage present when the after-gate event runs, normally on pairs
    /// suppressed before the gate. Decomposition children can also leak a qubit
    /// before the parent's after-gate event. State changes returned by channels
    /// during this event are applied only after all channels have run.
    fn handle_after_gate(
        &self,
        gate_type: GateType,
        qubits: &[pecos_core::QubitId],
        ctx: &NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        // Fast path: if no qubits are leaked at all, return immediately
        if ctx.leaked_count() == 0 {
            return NoiseResponse::None;
        }

        if gate_type.is_two_qubit() {
            let mut gates = SmallVec::new();

            for &[q0, q1] in qubits.as_chunks::<2>().0 {
                let partner = match (ctx.is_leaked(q0), ctx.is_leaked(q1)) {
                    (true, false) => q1,
                    (false, true) => q0,
                    _ => continue,
                };
                if !rng.check_probability(self.partner_pauli_threshold) {
                    continue;
                }
                let pauli = match rng.random_range(0..3) {
                    0 => GateType::X,
                    1 => GateType::Y,
                    _ => GateType::Z,
                };
                gates.push(GateCommand::new(pauli, smallvec::smallvec![partner]));
            }

            if !gates.is_empty() {
                return NoiseResponse::inject_gates(gates);
            }
        }

        NoiseResponse::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::QubitId;

    #[test]
    fn test_batched_partner_noise_is_pair_local() {
        let qubits = [QubitId(0), QubitId(1), QubitId(2), QubitId(3)];
        for (leaked, partners) in [
            (vec![0], vec![QubitId(1)]),
            (vec![0, 1], vec![]),
            (vec![0, 3], vec![QubitId(1), QubitId(2)]),
            (vec![], vec![]),
        ] {
            let mut ctx = NoiseContext::new();
            for q in leaked {
                ctx.mark_leaked(QubitId(q));
            }
            let mut rng = PecosRng::seed_from_u64(42);
            let mut expected_rng = rng.clone();
            let expected_gates: Vec<_> = partners
                .iter()
                .map(|&q| {
                    assert!(expected_rng.check_probability(PecosRng::probability_threshold(1.0)));
                    let pauli = match expected_rng.random_range(0..3) {
                        0 => GateType::X,
                        1 => GateType::Y,
                        _ => GateType::Z,
                    };
                    GateCommand::new(pauli, smallvec::smallvec![q])
                })
                .collect();
            let response = LeakageChannel::new()
                .with_partner_pauli_probability(1.0)
                .unwrap()
                .handle_after_gate(GateType::CX, &qubits, &ctx, &mut rng);
            if partners.is_empty() {
                assert!(response.is_none());
            } else {
                let NoiseResponse::InjectGates(gates) = response else {
                    panic!("expected partner Paulis");
                };
                assert_eq!(gates.as_slice(), expected_gates.as_slice());
            }
            assert_eq!(rng.random::<u64>(), expected_rng.random::<u64>());
        }
    }

    #[test]
    fn test_batched_partner_noise_default_has_no_gates_or_draws() {
        let qubits = [QubitId(0), QubitId(1), QubitId(2), QubitId(3)];
        let event = NoiseEvent::after_gate(GateType::CX, &qubits, &[]);
        for leaked in [vec![0], vec![0, 1], vec![0, 3], vec![]] {
            let mut ctx = NoiseContext::new();
            for q in leaked {
                ctx.mark_leaked(QubitId(q));
            }
            let mut rng = PecosRng::seed_from_u64(42);
            let mut expected_rng = rng.clone();
            assert!(
                LeakageChannel::new()
                    .apply(&event, &mut ctx, &mut rng)
                    .is_none()
            );
            assert_eq!(rng.random::<u64>(), expected_rng.random::<u64>());
        }
    }

    #[test]
    fn test_skip_gate_for_leaked_qubit() {
        let channel = LeakageChannel::new();

        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];
        let event = NoiseEvent::BeforeGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
            group_size: crate::noise::gate_group_size(GateType::H.quantum_arity(), qubits.len()),
        };

        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(0));

        let mut rng = PecosRng::seed_from_u64(42);
        let response = channel.apply(&event, &mut ctx, &mut rng);

        assert_eq!(response.skipped_qubits().as_slice(), &[QubitId(0)]);
    }

    #[test]
    fn test_skip_two_qubit_gate_if_any_leaked() {
        let channel = LeakageChannel::new();

        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];
        let event = NoiseEvent::BeforeGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
            group_size: crate::noise::gate_group_size(GateType::CX.quantum_arity(), qubits.len()),
        };

        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(1)); // Only target is leaked

        let mut rng = PecosRng::seed_from_u64(42);
        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Should skip even if only one qubit is leaked
        assert_eq!(response.skipped_qubits().as_slice(), &[QubitId(1)]);
    }

    #[test]
    fn test_no_skip_without_leakage() {
        let channel = LeakageChannel::new();

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::BeforeGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
            group_size: crate::noise::gate_group_size(GateType::H.quantum_arity(), qubits.len()),
        };

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);
        assert!(response.is_none());
    }

    #[test]
    fn test_no_skip_measurement_for_leaked_qubit() {
        let channel = LeakageChannel::new();

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::BeforeGate {
            gate_type: GateType::MZ,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
            group_size: crate::noise::gate_group_size(GateType::MZ.quantum_arity(), qubits.len()),
        };

        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(0));

        let mut rng = PecosRng::seed_from_u64(42);
        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Measurements should NOT be skipped - they have special handling
        assert!(response.skipped_qubits().is_empty());
    }

    #[test]
    fn test_leaked_qubit_measurement() {
        let channel = LeakageChannel::new();

        let qubits = [QubitId(0)];
        let event = NoiseEvent::BeforeMeasurement {
            gate_type: GateType::MZ,
            qubits: &qubits,
        };

        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(0));

        let mut rng = PecosRng::seed_from_u64(42);
        let mut expected_rng = rng.clone();
        assert!(!channel.responds_to(&event));
        assert!(channel.apply(&event, &mut ctx, &mut rng).is_none());
        assert_eq!(rng.random::<u64>(), expected_rng.random::<u64>());
    }

    #[test]
    fn test_two_qubit_gate_with_leaked_partner() {
        let channel = LeakageChannel::new()
            .with_partner_pauli_probability(1.0)
            .unwrap();

        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
            group_size: crate::noise::gate_group_size(GateType::CX.quantum_arity(), qubits.len()),
            leaked_suppressed: &[],
        };

        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(0)); // Control is leaked

        let mut rng = PecosRng::seed_from_u64(42);
        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Should apply depolarizing to non-leaked qubit
        if let NoiseResponse::InjectGates(gates) = response {
            assert_eq!(gates.len(), 1);
            assert_eq!(gates[0].qubits[0], QubitId(1)); // Target qubit
        } else {
            panic!("Expected InjectGates response");
        }
    }

    #[test]
    fn test_no_effect_without_leakage() {
        let channel = LeakageChannel::new();

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
            group_size: crate::noise::gate_group_size(GateType::H.quantum_arity(), qubits.len()),
            leaked_suppressed: &[],
        };

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);
        assert!(response.is_none());
    }
}
