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

//! Single-qubit gate noise channel.
//!
//! This is a traditional standalone channel implementation. For composable,
//! declarative noise models with conditional logic, see `CompositeChannel` in
//! `pecos_neo::noise::composite::prelude`.
//!
//! ## When to use this vs `CompositeChannel`
//!
//! **Use `SingleQubitChannel` when:**
//! - You want a simple, direct noise model
//! - Performance is critical (no primitive tree traversal)
//! - The built-in options (depolarizing, emission, seepage) suffice
//!
//! **Use `CompositeChannel` when:**
//! - You need complex conditional logic (when leaked, when partner fired, etc.)
//! - You want to compose reusable noise primitives
//! - You need custom branching or sampling behavior
//!
//! Handles depolarizing and Pauli noise on single-qubit gates, with support for:
//! - Configurable Pauli error distribution (uniform or biased)
//! - Emission errors that can cause leakage
//! - Seepage for leaked qubits

use super::{
    NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse, PauliWeights, SingleQubitEmissionResult,
    SingleQubitEmissionWeights,
};
use crate::command::GateCommand;
use pecos_random::PecosRng;
use rand::RngExt;
use smallvec::SmallVec;

/// Noise channel for single-qubit gates.
///
/// Models two types of errors:
/// 1. Pauli errors (X, Y, Z) with configurable distribution
/// 2. Emission errors that can cause Pauli errors AND/OR leakage
///
/// This matches the structure of `GeneralNoiseModel`'s single-qubit noise.
///
/// See also: `CompositeChannel` for composable, primitive-based noise models.
#[derive(Debug, Clone)]
pub struct SingleQubitChannel {
    /// Probability of any error occurring (total error rate).
    pub error_probability: f64,

    /// Distribution of Pauli errors when a Pauli error occurs.
    pub pauli_weights: PauliWeights,

    /// Fraction of errors that are emission errors (vs Pauli errors).
    ///
    /// When an error occurs:
    /// - With probability `emission_ratio`, it's an emission error
    /// - Otherwise, it's a Pauli error from `pauli_weights`
    pub emission_ratio: f64,

    /// Distribution of emission errors (Pauli gates and/or leakage).
    ///
    /// When an emission error occurs on a non-leaked qubit, sample from this
    /// distribution to determine whether to apply a Pauli error or cause leakage.
    pub emission_weights: SingleQubitEmissionWeights,

    /// Probability of seepage when an emission error occurs on a leaked qubit.
    ///
    /// Seepage returns a leaked qubit to the computational subspace.
    pub seepage_probability: f64,

    // Precomputed probability thresholds for fast sampling (avoids f64 conversion)
    error_threshold: u64,
    emission_threshold: u64,
    seepage_threshold: u64,
}

impl Default for SingleQubitChannel {
    fn default() -> Self {
        Self {
            error_probability: 0.0,
            pauli_weights: PauliWeights::uniform(),
            emission_ratio: 0.0,
            emission_weights: SingleQubitEmissionWeights::uniform(),
            seepage_probability: 0.0,
            error_threshold: 0,
            emission_threshold: 0,
            seepage_threshold: 0,
        }
    }
}

impl SingleQubitChannel {
    /// Create a new channel with all parameters specified.
    ///
    /// Precomputes probability thresholds for faster sampling.
    #[must_use]
    pub fn new(
        error_probability: f64,
        pauli_weights: PauliWeights,
        emission_ratio: f64,
        emission_weights: SingleQubitEmissionWeights,
        seepage_probability: f64,
    ) -> Self {
        Self {
            error_probability,
            pauli_weights,
            emission_ratio,
            emission_weights,
            seepage_probability,
            error_threshold: PecosRng::probability_threshold(error_probability),
            emission_threshold: PecosRng::probability_threshold(emission_ratio),
            seepage_threshold: PecosRng::probability_threshold(seepage_probability),
        }
    }

    /// Create a uniform depolarizing noise channel.
    ///
    /// With probability `p`, applies a uniformly random Pauli (X, Y, or Z).
    #[must_use]
    pub fn depolarizing(p: f64) -> Self {
        Self {
            error_probability: p,
            error_threshold: PecosRng::probability_threshold(p),
            pauli_weights: PauliWeights::uniform(),
            ..Default::default()
        }
    }

    /// Create a channel with a specific Pauli distribution.
    #[must_use]
    pub fn with_pauli_weights(mut self, weights: PauliWeights) -> Self {
        self.pauli_weights = weights;
        self
    }

    /// Set the emission error ratio with simple leakage probability.
    ///
    /// This is a convenience method that configures emission weights
    /// as uniform Pauli errors with the specified leakage probability.
    #[must_use]
    pub fn with_emission(mut self, ratio: f64, leakage_prob: f64) -> Self {
        self.emission_ratio = ratio;
        self.emission_threshold = PecosRng::probability_threshold(ratio);
        // Distribute remaining probability among X, Y, Z
        let pauli_prob = (1.0 - leakage_prob) / 3.0;
        self.emission_weights =
            SingleQubitEmissionWeights::custom(pauli_prob, pauli_prob, pauli_prob, leakage_prob);
        self
    }

    /// Set the emission error ratio with custom emission weights.
    #[must_use]
    pub fn with_emission_weights(
        mut self,
        ratio: f64,
        weights: SingleQubitEmissionWeights,
    ) -> Self {
        self.emission_ratio = ratio;
        self.emission_threshold = PecosRng::probability_threshold(ratio);
        self.emission_weights = weights;
        self
    }

    /// Set the seepage probability.
    #[must_use]
    pub fn with_seepage(mut self, p: f64) -> Self {
        self.seepage_probability = p;
        self.seepage_threshold = PecosRng::probability_threshold(p);
        self
    }

    /// Create a Z-biased dephasing channel.
    #[must_use]
    pub fn dephasing(p: f64) -> Self {
        Self {
            error_probability: p,
            error_threshold: PecosRng::probability_threshold(p),
            pauli_weights: PauliWeights::custom(0.0, 0.0, 1.0),
            ..Default::default()
        }
    }

    /// Create a bit-flip channel (X errors only).
    #[must_use]
    pub fn bit_flip(p: f64) -> Self {
        Self {
            error_probability: p,
            error_threshold: PecosRng::probability_threshold(p),
            pauli_weights: PauliWeights::custom(1.0, 0.0, 0.0),
            ..Default::default()
        }
    }

    /// Scale the error probability by a factor.
    ///
    /// This multiplies the current error probability by `scale`.
    /// Useful for globally adjusting noise levels.
    #[must_use]
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.error_probability *= scale;
        self.error_threshold = PecosRng::probability_threshold(self.error_probability);
        self
    }
}

impl NoiseChannel for SingleQubitChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        if self.error_probability <= 0.0 {
            return false;
        }
        // Respond to BeforeGate for leaked qubit handling and AfterGate for noise
        match event {
            NoiseEvent::BeforeGate { gate_type, .. } | NoiseEvent::AfterGate { gate_type, .. } => {
                gate_type.is_single_qubit()
            }
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
            NoiseEvent::BeforeGate {
                gate_type, qubits, ..
            } => {
                // Skip noise for noiseless gates (but still check leakage)
                if ctx.is_noiseless(*gate_type) {
                    return NoiseResponse::None;
                }
                self.handle_before_gate(qubits, ctx, rng)
            }
            NoiseEvent::AfterGate {
                gate_type, qubits, ..
            } => {
                // Skip noise for noiseless gates
                if ctx.is_noiseless(*gate_type) {
                    return NoiseResponse::None;
                }
                self.handle_after_gate(qubits, ctx, rng)
            }
            _ => NoiseResponse::None,
        }
    }

    /// Optimized combined check + apply that avoids redundant event matching.
    #[inline]
    fn try_apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> Option<NoiseResponse> {
        // Early exit if no errors configured
        if self.error_probability <= 0.0 {
            return None;
        }

        match event {
            NoiseEvent::BeforeGate {
                gate_type, qubits, ..
            } => {
                if !gate_type.is_single_qubit() {
                    return None;
                }
                // Skip noise for noiseless gates (but still check leakage)
                if ctx.is_noiseless(*gate_type) {
                    return Some(NoiseResponse::None);
                }
                Some(self.handle_before_gate(qubits, ctx, rng))
            }
            NoiseEvent::AfterGate {
                gate_type, qubits, ..
            } => {
                if !gate_type.is_single_qubit() {
                    return None;
                }
                // Skip noise for noiseless gates
                if ctx.is_noiseless(*gate_type) {
                    return Some(NoiseResponse::None);
                }
                Some(self.handle_after_gate(qubits, ctx, rng))
            }
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        "SingleQubitChannel"
    }

    fn priority(&self) -> i32 {
        // Higher priority to handle leakage checks first
        10
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

impl SingleQubitChannel {
    /// Handle `BeforeGate` event - check for leaked qubits and skip gate if needed.
    fn handle_before_gate(
        &self,
        qubits: &[pecos_core::QubitId],
        ctx: &NoiseContext,
        _rng: &mut PecosRng,
    ) -> NoiseResponse {
        // If any qubit is leaked, skip the gate
        // Uses optimized any_leaked which has O(1) fast path when leaked_count == 0
        if ctx.any_leaked(qubits) {
            return NoiseResponse::SkipGate;
        }
        NoiseResponse::None
    }

    /// Handle `AfterGate` event - apply Pauli or emission errors.
    fn handle_after_gate(
        &self,
        qubits: &[pecos_core::QubitId],
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let mut gates = SmallVec::new();
        let mut leaked = SmallVec::new();
        let mut unleaked = SmallVec::new();

        // Fast path: check if any leakage exists at all
        let has_any_leakage = ctx.leaked_count() > 0;

        for &qubit in qubits {
            // Skip expensive is_leaked call if we know no qubits are leaked
            let is_leaked = has_any_leakage && ctx.is_leaked(qubit);

            // For leaked qubits, only emission errors can cause seepage
            if is_leaked {
                if self.emission_ratio > 0.0
                    && rng.check_probability(self.error_threshold)
                    && rng.check_probability(self.emission_threshold)
                    && rng.check_probability(self.seepage_threshold)
                {
                    unleaked.push(qubit);
                }
                continue;
            }

            // Apply error with probability (using precomputed threshold for speed)
            if rng.check_probability(self.error_threshold) {
                // Determine if this is an emission or Pauli error
                if self.emission_ratio > 0.0 && rng.check_probability(self.emission_threshold) {
                    // Emission error - sample from emission weights
                    match self.emission_weights.sample(rng.random::<f64>()) {
                        SingleQubitEmissionResult::Pauli(pauli) => {
                            gates.push(GateCommand::new(pauli, smallvec::smallvec![qubit]));
                        }
                        SingleQubitEmissionResult::Leaked => {
                            leaked.push(qubit);
                        }
                    }
                } else {
                    // Pauli error - sample from pauli weights
                    let pauli = self.pauli_weights.sample(rng.random::<f64>());
                    gates.push(GateCommand::new(pauli, smallvec::smallvec![qubit]));
                }
            }
        }

        // Build combined response
        let mut response = NoiseResponse::None;

        if !gates.is_empty() {
            response = response.combine(NoiseResponse::inject_gates(gates));
        }
        if !leaked.is_empty() {
            response = response.combine(NoiseResponse::MarkLeaked(leaked));
        }
        if !unleaked.is_empty() {
            response = response.combine(NoiseResponse::MarkUnleaked(unleaked));
        }

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::GateType;
    use pecos_core::QubitId;

    #[test]
    fn test_depolarizing_channel() {
        let channel = SingleQubitChannel::depolarizing(1.0);

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        assert!(channel.responds_to(&event));

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);
        // Should produce some gates (depolarizing with p=1.0)
        assert!(!response.is_none());
    }

    #[test]
    fn test_no_error_on_two_qubit_gate() {
        let channel = SingleQubitChannel::depolarizing(1.0);

        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        assert!(!channel.responds_to(&event));
    }

    #[test]
    fn test_leakage() {
        let channel = SingleQubitChannel::depolarizing(1.0).with_emission(1.0, 1.0);

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
        // With emission_ratio=1.0 and leakage_probability=1.0, should cause leakage
        assert!(matches!(response, NoiseResponse::MarkLeaked(_)));
    }

    #[test]
    fn test_skip_gate_for_leaked_qubit() {
        let channel = SingleQubitChannel::depolarizing(1.0);

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
    fn test_z_biased_channel() {
        let channel = SingleQubitChannel::dephasing(1.0);

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

        // Should always produce Z gate
        if let NoiseResponse::InjectGates(gates) = response {
            assert_eq!(gates.len(), 1);
            assert_eq!(gates[0].gate_type, GateType::Z);
        } else {
            panic!("Expected InjectGates response");
        }
    }
}
