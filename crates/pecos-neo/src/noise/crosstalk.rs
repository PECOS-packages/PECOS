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

//! Crosstalk noise channel.
//!
//! Models crosstalk errors that occur when operations on some qubits
//! affect other nearby (local) or distant (global) qubits.
//!
//! ## Crosstalk Types
//!
//! - **Local crosstalk**: Affects only qubits near the gated qubits (e.g., nearest neighbors)
//! - **Global crosstalk**: Affects all other active qubits in the system
//!
//! ## Physical Motivation
//!
//! In ion trap systems, crosstalk can occur due to:
//! - Scattered light during optical pumping or measurement
//! - Stray fields affecting nearby ions
//! - Phonon-mediated interactions
//!
//! ## Usage
//!
//! ```
//! use pecos_neo::noise::crosstalk::CrosstalkChannel;
//!
//! // Create a crosstalk channel with different local and global rates
//! let crosstalk = CrosstalkChannel::new()
//!     .with_local_rate(0.01)    // 1% chance per local qubit
//!     .with_global_rate(0.001); // 0.1% chance per global qubit
//! ```

use super::{
    CrosstalkResult, CrosstalkTransitions, NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse,
};
use crate::command::{GateCommand, GateType};
use pecos_core::QubitId;
use pecos_rng::PecosRng;
use rand::RngExt;
use smallvec::SmallVec;

/// Crosstalk noise channel.
///
/// Applies errors to qubits that are not directly involved in an operation
/// but may be affected due to physical proximity or shared resources.
#[derive(Debug, Clone)]
pub struct CrosstalkChannel {
    /// Probability of crosstalk error per local (neighbor) qubit.
    pub local_rate: f64,

    /// Probability of crosstalk error per global (non-neighbor) qubit.
    pub global_rate: f64,

    /// Function to determine which qubits are neighbors of the gated qubits.
    /// If None, all non-gated qubits are treated as global.
    neighbor_fn: Option<NeighborFunction>,

    /// Probability that a crosstalk error causes leakage (simple model).
    ///
    /// Used when `transitions` is None. Ignored when using transition model.
    pub leakage_probability: f64,

    /// State-dependent transition probabilities for measurement crosstalk.
    ///
    /// When set, crosstalk during measurements uses this model which
    /// accounts for the target qubit's current state. When None, uses
    /// simple random Pauli + leakage model.
    pub transitions: Option<CrosstalkTransitions>,
}

/// Function type for determining neighbor qubits.
///
/// Given the gated qubits, returns the list of their neighbors.
#[derive(Clone)]
pub struct NeighborFunction {
    /// The function itself.
    func: fn(&[QubitId]) -> Vec<QubitId>,
}

impl std::fmt::Debug for NeighborFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeighborFunction").finish()
    }
}

impl Default for CrosstalkChannel {
    fn default() -> Self {
        Self {
            local_rate: 0.0,
            global_rate: 0.0,
            neighbor_fn: None,
            leakage_probability: 0.0,
            transitions: None,
        }
    }
}

impl CrosstalkChannel {
    /// Create a new crosstalk channel with default (zero) rates.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the local crosstalk rate (affects neighbor qubits).
    #[must_use]
    pub fn with_local_rate(mut self, rate: f64) -> Self {
        self.local_rate = rate;
        self
    }

    /// Set the global crosstalk rate (affects non-neighbor qubits).
    #[must_use]
    pub fn with_global_rate(mut self, rate: f64) -> Self {
        self.global_rate = rate;
        self
    }

    /// Set the function that determines neighbor qubits.
    ///
    /// Without this, all non-gated qubits are treated as global.
    #[must_use]
    pub fn with_neighbor_function(mut self, func: fn(&[QubitId]) -> Vec<QubitId>) -> Self {
        self.neighbor_fn = Some(NeighborFunction { func });
        self
    }

    /// Set the probability that a crosstalk error causes leakage.
    ///
    /// This is used with the simple crosstalk model. For state-dependent
    /// transitions (including state-dependent leakage), use `with_transitions`.
    #[must_use]
    pub fn with_leakage(mut self, probability: f64) -> Self {
        self.leakage_probability = probability;
        self
    }

    /// Set state-dependent transition probabilities for measurement crosstalk.
    ///
    /// The transition model specifies what happens based on the target qubit's
    /// current state (0 or 1):
    /// - Stay at current value
    /// - Flip to opposite value
    /// - Leak out of computational subspace
    ///
    /// When this is set, the simple `leakage_probability` is ignored.
    #[must_use]
    pub fn with_transitions(mut self, transitions: CrosstalkTransitions) -> Self {
        self.transitions = Some(transitions);
        self
    }

    /// Create a simple crosstalk channel with uniform global rate.
    #[must_use]
    pub fn global_only(rate: f64) -> Self {
        Self {
            global_rate: rate,
            ..Default::default()
        }
    }

    /// Get the neighbor qubits for a set of gated qubits.
    fn get_neighbors(&self, gated_qubits: &[QubitId]) -> Vec<QubitId> {
        match &self.neighbor_fn {
            Some(nf) => (nf.func)(gated_qubits),
            None => Vec::new(), // No neighbors defined, all qubits are global
        }
    }

    /// Apply crosstalk to a single qubit.
    ///
    /// If `known_state` is provided, uses state-dependent transitions.
    /// Otherwise, uses random Pauli + leakage model.
    fn apply_crosstalk_to_qubit(
        &self,
        qubit: QubitId,
        known_state: Option<bool>,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        // Use transition model if available
        if let Some(ref transitions) = self.transitions {
            // If we know the state, use it; otherwise sample randomly
            let state = known_state.unwrap_or_else(|| rng.random::<bool>());
            let result = transitions.sample(state, rng.random::<f64>());

            return match result {
                CrosstalkResult::NoChange => NoiseResponse::None,
                CrosstalkResult::Flip => NoiseResponse::inject_gate(GateCommand::new(
                    GateType::X,
                    smallvec::smallvec![qubit],
                )),
                CrosstalkResult::Leak => NoiseResponse::MarkLeaked(smallvec::smallvec![qubit]),
            };
        }

        // Simple model: random Pauli or leakage
        if rng.random::<f64>() < self.leakage_probability {
            NoiseResponse::MarkLeaked(smallvec::smallvec![qubit])
        } else {
            let pauli = match rng.random_range(0..3) {
                0 => GateType::X,
                1 => GateType::Y,
                _ => GateType::Z,
            };
            NoiseResponse::inject_gate(GateCommand::new(pauli, smallvec::smallvec![qubit]))
        }
    }
}

impl NoiseChannel for CrosstalkChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        // Only respond if we have non-zero crosstalk rates
        if self.local_rate <= 0.0 && self.global_rate <= 0.0 {
            return false;
        }

        // Respond to preparation and measurement operations
        // (These are the main sources of crosstalk in ion traps)
        matches!(
            event,
            NoiseEvent::AfterPreparation { .. }
                | NoiseEvent::AfterMeasurement { .. }
                | NoiseEvent::AfterGate { .. }
        )
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let gated_qubits = event.qubits();

        let mut responses: Vec<NoiseResponse> = Vec::new();
        let mut leaked: SmallVec<[QubitId; 4]> = SmallVec::new();
        let mut gates: SmallVec<[GateCommand; 4]> = SmallVec::new();

        // Get neighbors for local crosstalk
        let neighbors = self.get_neighbors(gated_qubits);

        // Apply local crosstalk to neighbor qubits
        if self.local_rate > 0.0 {
            let local_targets = ctx.local_crosstalk_targets(gated_qubits, &neighbors);
            for target in local_targets {
                if rng.random::<f64>() < self.local_rate {
                    match self.apply_crosstalk_to_qubit(target, None, rng) {
                        NoiseResponse::MarkLeaked(qubits) => leaked.extend(qubits),
                        NoiseResponse::InjectGates(g) => gates.extend(g.iter().cloned()),
                        response => responses.push(response),
                    }
                }
            }
        }

        // Apply global crosstalk to non-neighbor qubits
        if self.global_rate > 0.0 {
            // Global targets are all active qubits except gated qubits and neighbors
            let mut exclude: Vec<QubitId> = gated_qubits.to_vec();
            exclude.extend(neighbors);

            let global_targets = ctx.crosstalk_targets(&exclude);
            for target in global_targets {
                if rng.random::<f64>() < self.global_rate {
                    match self.apply_crosstalk_to_qubit(target, None, rng) {
                        NoiseResponse::MarkLeaked(qubits) => leaked.extend(qubits),
                        NoiseResponse::InjectGates(g) => gates.extend(g.iter().cloned()),
                        response => responses.push(response),
                    }
                }
            }
        }

        // Combine all responses
        let mut result = NoiseResponse::None;

        if !gates.is_empty() {
            result = result.combine(NoiseResponse::inject_gates(gates));
        }
        if !leaked.is_empty() {
            result = result.combine(NoiseResponse::MarkLeaked(leaked));
        }
        for r in responses {
            result = result.combine(r);
        }

        result
    }

    fn name(&self) -> &'static str {
        "CrosstalkChannel"
    }

    fn priority(&self) -> i32 {
        // Lower priority - apply after main gate noise
        -10
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crosstalk_channel_creation() {
        let channel = CrosstalkChannel::new()
            .with_local_rate(0.01)
            .with_global_rate(0.001)
            .with_leakage(0.1);

        assert!((channel.local_rate - 0.01).abs() < 1e-10);
        assert!((channel.global_rate - 0.001).abs() < 1e-10);
        assert!((channel.leakage_probability - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_crosstalk_global_only() {
        let channel = CrosstalkChannel::global_only(0.1);

        let qubits = [QubitId(0)];
        let event = NoiseEvent::AfterPreparation { qubits: &qubits };

        assert!(channel.responds_to(&event));

        // Create context with multiple active qubits
        let mut ctx = NoiseContext::new();
        ctx.mark_prepared(QubitId(0));
        ctx.mark_prepared(QubitId(1));
        ctx.mark_prepared(QubitId(2));

        // Run many times to check that crosstalk occurs
        let mut had_response = false;
        let mut rng = PecosRng::seed_from_u64(42);
        for _ in 0..100 {
            let response = channel.apply(&event, &mut ctx, &mut rng);
            if !response.is_none() {
                had_response = true;
                break;
            }
        }

        assert!(had_response, "Should have crosstalk effects with 10% rate");
    }

    #[test]
    fn test_no_crosstalk_on_inactive_qubits() {
        let channel = CrosstalkChannel::global_only(1.0); // 100% rate

        let qubits = [QubitId(0)];
        let event = NoiseEvent::AfterPreparation { qubits: &qubits };

        // Create context where qubit 1 is measured (inactive)
        let mut ctx = NoiseContext::new();
        ctx.mark_prepared(QubitId(0));
        ctx.mark_prepared(QubitId(1));
        ctx.mark_measured(QubitId(1)); // Now inactive

        let mut rng = PecosRng::seed_from_u64(42);
        let response = channel.apply(&event, &mut ctx, &mut rng);

        // No targets (qubit 1 is inactive), so no crosstalk
        assert!(response.is_none());
    }

    #[test]
    fn test_local_vs_global_crosstalk() {
        // Define a neighbor function: qubit i has neighbors i-1 and i+1
        fn neighbors(gated: &[QubitId]) -> Vec<QubitId> {
            let mut result = Vec::new();
            for &QubitId(q) in gated {
                if q > 0 {
                    result.push(QubitId(q - 1));
                }
                result.push(QubitId(q + 1));
            }
            result
        }

        let channel = CrosstalkChannel::new()
            .with_local_rate(1.0) // Always hit local
            .with_global_rate(0.0) // Never hit global
            .with_neighbor_function(neighbors);

        let qubits = [QubitId(5)];
        let event = NoiseEvent::AfterPreparation { qubits: &qubits };

        // Prepare qubits 0-10
        let mut ctx = NoiseContext::new();
        for i in 0..=10 {
            ctx.mark_prepared(QubitId(i));
        }

        let mut rng = PecosRng::seed_from_u64(42);
        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Should only affect neighbors (4 and 6)
        if let NoiseResponse::InjectGates(gates) = response {
            for gate in gates.iter() {
                let qubit = gate.qubits[0];
                assert!(
                    qubit == QubitId(4) || qubit == QubitId(6),
                    "Only neighbors should be affected"
                );
            }
        }
    }

    #[test]
    fn test_crosstalk_with_transitions() {
        // Create transition model: always flip, never leak
        let transitions = CrosstalkTransitions::custom(
            0.0, 1.0, 0.0, // from 0: always flip
            0.0, 1.0, 0.0, // from 1: always flip
        );

        let channel = CrosstalkChannel::new()
            .with_global_rate(1.0)
            .with_transitions(transitions);

        let qubits = [QubitId(0)];
        let event = NoiseEvent::AfterPreparation { qubits: &qubits };

        let mut ctx = NoiseContext::new();
        ctx.mark_prepared(QubitId(0));
        ctx.mark_prepared(QubitId(1));

        let mut rng = PecosRng::seed_from_u64(42);
        let response = channel.apply(&event, &mut ctx, &mut rng);

        // With transitions that always flip, should produce X gate
        if let NoiseResponse::InjectGates(gates) = response {
            assert_eq!(gates.len(), 1);
            assert_eq!(gates[0].gate_type, GateType::X);
            assert_eq!(gates[0].qubits[0], QubitId(1));
        } else {
            panic!("Expected InjectGates response");
        }
    }

    #[test]
    fn test_crosstalk_transitions_with_leakage() {
        // Create transition model: always leak
        let transitions = CrosstalkTransitions::custom(
            0.0, 0.0, 1.0, // from 0: always leak
            0.0, 0.0, 1.0, // from 1: always leak
        );

        let channel = CrosstalkChannel::new()
            .with_global_rate(1.0)
            .with_transitions(transitions);

        let qubits = [QubitId(0)];
        let event = NoiseEvent::AfterPreparation { qubits: &qubits };

        let mut ctx = NoiseContext::new();
        ctx.mark_prepared(QubitId(0));
        ctx.mark_prepared(QubitId(1));

        let mut rng = PecosRng::seed_from_u64(42);
        let response = channel.apply(&event, &mut ctx, &mut rng);

        // With transitions that always leak, should mark qubit as leaked
        if let NoiseResponse::MarkLeaked(leaked) = response {
            assert_eq!(leaked.len(), 1);
            assert_eq!(leaked[0], QubitId(1));
        } else {
            panic!("Expected MarkLeaked response, got {response:?}");
        }
    }
}
