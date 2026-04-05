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
//! Handles ALL effects of leaked qubits on the simulation:
//! - **Gate skipping**: Gates on leaked qubits are not applied to the quantum state
//! - **Partner depolarizing**: For two-qubit gates, non-leaked partners get random Pauli errors
//! - **Measurement handling**: Leaked qubits produce random measurement results
//!
//! This channel should ALWAYS be added to a noise model that tracks leakage,
//! as it handles fundamental leakage behavior independent of any noise rates.

use super::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse};
use crate::command::{GateCommand, GateType};
use pecos_random::PecosRng;
use rand::RngExt;
use smallvec::SmallVec;

/// Channel that handles the effects of leaked qubits.
///
/// This channel is responsible for ALL leakage-related effects:
///
/// 1. **`BeforeGate`**: Skip gates if any involved qubit is leaked
/// 2. **`AfterGate`**: For two-qubit gates where one qubit is leaked,
///    apply depolarizing noise to the non-leaked partner
/// 3. **`BeforeMeasurement`**: Leaked qubits produce random results
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
    pub leakage_scale: f64,
}

impl Default for LeakageChannel {
    fn default() -> Self {
        Self { leakage_scale: 1.0 }
    }
}

impl LeakageChannel {
    /// Create a leakage channel with full leakage (no scaling).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a leakage channel that converts all leakage to depolarizing.
    #[must_use]
    pub fn no_leakage() -> Self {
        Self { leakage_scale: 0.0 }
    }

    /// Set the leakage scale factor.
    #[must_use]
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.leakage_scale = scale;
        self
    }
}

impl NoiseChannel for LeakageChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        // Always respond to gate and measurement events - we need to check for leakage
        matches!(
            event,
            NoiseEvent::BeforeGate { .. }
                | NoiseEvent::AfterGate { .. }
                | NoiseEvent::BeforeMeasurement { .. }
        )
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
            NoiseEvent::AfterGate {
                gate_type, qubits, ..
            } => Self::handle_after_gate(*gate_type, qubits, ctx, rng),
            NoiseEvent::BeforeMeasurement { qubits } => Self::handle_measurement(qubits, ctx, rng),
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
    /// Handle `BeforeGate` - skip the gate if any qubit is leaked.
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

        // If any qubit is leaked, skip the gate entirely
        // Uses optimized any_leaked which has O(1) fast path when leaked_count == 0
        if ctx.any_leaked(qubits) {
            return NoiseResponse::SkipGate;
        }
        NoiseResponse::None
    }

    /// Handle `AfterGate` - for two-qubit gates where one qubit is leaked,
    /// apply depolarizing noise to the non-leaked partner.
    ///
    /// Note: This only triggers if the gate wasn't skipped (i.e., if leakage
    /// happened during the gate itself, not before).
    fn handle_after_gate(
        gate_type: GateType,
        qubits: &[pecos_core::QubitId],
        ctx: &NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        // Fast path: if no qubits are leaked at all, return immediately
        if ctx.leaked_count() == 0 {
            return NoiseResponse::None;
        }

        // Check if any involved qubit is leaked
        let leaked: Vec<_> = qubits.iter().filter(|&&q| ctx.is_leaked(q)).collect();

        if leaked.is_empty() {
            return NoiseResponse::None;
        }

        // For two-qubit gates, if one qubit is leaked, apply depolarizing to the other
        if gate_type.is_two_qubit() && qubits.len() >= 2 {
            let mut gates = SmallVec::new();

            for &qubit in qubits {
                if !ctx.is_leaked(qubit) {
                    // Apply random Pauli to non-leaked partner
                    let pauli = match rng.random_range(0..3) {
                        0 => GateType::X,
                        1 => GateType::Y,
                        _ => GateType::Z,
                    };
                    gates.push(GateCommand::new(pauli, smallvec::smallvec![qubit]));
                }
            }

            if !gates.is_empty() {
                return NoiseResponse::inject_gates(gates);
            }
        }

        NoiseResponse::None
    }

    fn handle_measurement(
        qubits: &[pecos_core::QubitId],
        ctx: &NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        // Leaked qubits produce random measurement results
        let mut flips = SmallVec::new();

        for &qubit in qubits {
            if ctx.is_leaked(qubit) {
                // 50% chance to flip (resulting in random outcome)
                if rng.random::<bool>() {
                    flips.push(qubit);
                }
            }
        }

        if flips.is_empty() {
            NoiseResponse::None
        } else {
            NoiseResponse::FlipOutcomes(flips)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::QubitId;

    #[test]
    fn test_skip_gate_for_leaked_qubit() {
        let channel = LeakageChannel::new();

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::BeforeGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(0));

        let mut rng = PecosRng::seed_from_u64(42);
        let response = channel.apply(&event, &mut ctx, &mut rng);

        assert!(response.should_skip_gate());
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
        };

        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(1)); // Only target is leaked

        let mut rng = PecosRng::seed_from_u64(42);
        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Should skip even if only one qubit is leaked
        assert!(response.should_skip_gate());
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
        };

        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(0));

        let mut rng = PecosRng::seed_from_u64(42);
        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Measurements should NOT be skipped - they have special handling
        assert!(!response.should_skip_gate());
    }

    #[test]
    fn test_leaked_qubit_measurement() {
        let channel = LeakageChannel::new();

        let qubits = [QubitId(0)];
        let event = NoiseEvent::BeforeMeasurement { qubits: &qubits };

        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(0));

        // Run many times to check randomness
        let mut flipped = 0;
        let mut not_flipped = 0;
        let mut rng = PecosRng::seed_from_u64(42);
        for _ in 0..100 {
            match channel.apply(&event, &mut ctx, &mut rng) {
                NoiseResponse::FlipOutcomes(_) => flipped += 1,
                NoiseResponse::None => not_flipped += 1,
                _ => panic!("Unexpected response"),
            }
        }

        // Should be roughly 50/50
        assert!(flipped > 20 && not_flipped > 20);
    }

    #[test]
    fn test_two_qubit_gate_with_leaked_partner() {
        let channel = LeakageChannel::new();

        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
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
        };

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);
        assert!(response.is_none());
    }
}
