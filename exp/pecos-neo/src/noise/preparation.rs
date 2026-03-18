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

//! Preparation noise channel.
//!
//! Handles errors during state preparation (initialization).

use super::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse};
use crate::command::{GateCommand, GateType};
use pecos_rng::PecosRng;
use smallvec::SmallVec;

/// Noise channel for state preparation operations.
///
/// Applies bit-flip errors (X gates) after preparation, modeling
/// imperfect initialization of qubits to |0⟩.
#[derive(Debug, Clone)]
pub struct PreparationChannel {
    /// Probability of preparation error.
    pub error_probability: f64,

    /// Probability that a preparation error causes leakage.
    pub leakage_ratio: f64,

    // Precomputed probability thresholds for fast sampling
    error_threshold: u64,
    leakage_threshold: u64,
}

impl Default for PreparationChannel {
    fn default() -> Self {
        Self {
            error_probability: 0.0,
            leakage_ratio: 0.0,
            error_threshold: 0,
            leakage_threshold: 0,
        }
    }
}

impl PreparationChannel {
    /// Create a preparation error channel.
    #[must_use]
    pub fn new(p: f64) -> Self {
        Self {
            error_probability: p,
            error_threshold: PecosRng::probability_threshold(p),
            ..Default::default()
        }
    }

    /// Set the leakage ratio.
    #[must_use]
    pub fn with_leakage(mut self, ratio: f64) -> Self {
        self.leakage_ratio = ratio;
        self.leakage_threshold = PecosRng::probability_threshold(ratio);
        self
    }
}

impl NoiseChannel for PreparationChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        if self.error_probability <= 0.0 {
            return false;
        }
        matches!(event, NoiseEvent::AfterPreparation { .. })
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        _ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let NoiseEvent::AfterPreparation { qubits } = event else {
            return NoiseResponse::None;
        };

        let mut gates = SmallVec::new();
        let mut leaked = SmallVec::new();

        for &qubit in *qubits {
            // Note: mark_prepared() is called by the composer, not here,
            // so that state tracking happens even without this channel.

            // Apply preparation error with probability (using precomputed threshold)
            if rng.check_probability(self.error_threshold) {
                if self.leakage_ratio > 0.0 && rng.check_probability(self.leakage_threshold) {
                    leaked.push(qubit);
                } else {
                    // Preparation error is modeled as bit flip (X gate)
                    gates.push(GateCommand::new(GateType::X, smallvec::smallvec![qubit]));
                }
            }
        }

        let mut response = NoiseResponse::None;

        if !gates.is_empty() {
            response = response.combine(NoiseResponse::inject_gates(gates));
        }
        if !leaked.is_empty() {
            response = response.combine(NoiseResponse::MarkLeaked(leaked));
        }

        response
    }

    fn name(&self) -> &'static str {
        "PreparationChannel"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::QubitId;

    #[test]
    fn test_preparation_error() {
        let channel = PreparationChannel::new(1.0); // Always error

        let qubits = [QubitId(0)];
        let event = NoiseEvent::AfterPreparation { qubits: &qubits };

        assert!(channel.responds_to(&event));

        let mut ctx = NoiseContext::new();
        // State updates are handled by the composer via event.apply_state_updates()
        // In isolated channel tests, we call it manually:
        event.apply_state_updates(&mut ctx);

        let mut rng = PecosRng::seed_from_u64(42);
        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Should produce an X gate (bit flip)
        if let NoiseResponse::InjectGates(gates) = response {
            assert_eq!(gates.len(), 1);
            assert_eq!(gates[0].gate_type, GateType::X);
        } else {
            panic!("Expected InjectGates response");
        }

        // Qubit should be marked as prepared (done by apply_state_updates)
        assert!(ctx.exists(QubitId(0)));
    }

    #[test]
    fn test_preparation_marks_prepared() {
        let channel = PreparationChannel::new(0.0); // No error

        let qubits = [QubitId(0)];
        let event = NoiseEvent::AfterPreparation { qubits: &qubits };

        let mut ctx = NoiseContext::new();
        // State updates are handled by event.apply_state_updates()
        event.apply_state_updates(&mut ctx);

        let mut rng = PecosRng::seed_from_u64(42);
        let _response = channel.apply(&event, &mut ctx, &mut rng);

        // Qubit should be marked as prepared (done by apply_state_updates)
        assert!(ctx.exists(QubitId(0)));
    }

    #[test]
    fn test_preparation_with_leakage() {
        let channel = PreparationChannel::new(1.0).with_leakage(1.0); // Always leak

        let qubits = [QubitId(0)];
        let event = NoiseEvent::AfterPreparation { qubits: &qubits };

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);
        assert!(matches!(response, NoiseResponse::MarkLeaked(_)));
    }
}
