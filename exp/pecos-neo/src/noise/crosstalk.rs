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
//! - **Global crosstalk**: Affects other non-leaked qubits prepared this shot, measured or not
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
//! let crosstalk = CrosstalkChannel::measurement()
//!     .with_local_rate(0.01)?    // 1% chance per local qubit
//!     .with_global_rate(0.001); // 0.1% chance per global qubit
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use super::{
    CrosstalkResult, CrosstalkTransitions, NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse,
};
use crate::command::{GateCommand, GateType};
use crate::noise::validation::{ValidationError, ValidationResult, validate_probability};
use pecos_core::QubitId;
use pecos_random::PecosRng;
use rand::RngExt;
use smallvec::SmallVec;

/// Crosstalk noise channel.
///
/// Applies errors to qubits that are not directly involved in an operation
/// but may be affected due to physical proximity or shared resources.
/// Targets exist this shot and are not leaked; measurement does not remove them.
/// Each channel subscribes to either preparation (noisy `PZ` only) or measurement.
/// Without a transition table, a fired target receives Z with probability 1/2,
/// after the optional leakage draw. Tables retain uniform sampled input states.
#[derive(Debug, Clone)]
pub struct CrosstalkChannel {
    kind: CrosstalkKind,

    /// Probability of crosstalk error per local (neighbor) qubit.
    local_rate: f64,

    /// Probability of crosstalk error per global (non-neighbor) qubit.
    global_rate: f64,

    /// Function to determine which qubits are neighbors of the gated qubits.
    /// If None, all non-gated qubits are treated as global.
    neighbor_fn: Option<NeighborFunction>,

    /// Probability that a crosstalk error causes leakage (simple model).
    ///
    /// Used when `transitions` is None. Ignored when using transition model.
    leakage_probability: f64,

    /// State-dependent transition probabilities for measurement crosstalk.
    ///
    /// When set, measurement crosstalk samples a state uniformly for this table.
    /// When None, uses the dephasing + leakage model.
    transitions: Option<CrosstalkTransitions>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrosstalkKind {
    Preparation,
    Measurement,
}

impl CrosstalkChannel {
    /// Return the configured `local_rate`.
    #[must_use]
    pub fn local_rate(&self) -> f64 {
        self.local_rate
    }

    /// Return the configured `global_rate`.
    #[must_use]
    pub fn global_rate(&self) -> f64 {
        self.global_rate
    }

    /// Return the configured `leakage_probability`.
    #[must_use]
    pub fn leakage_probability(&self) -> f64 {
        self.leakage_probability
    }

    /// Return the configured `transitions`.
    #[must_use]
    pub fn transitions(&self) -> &Option<CrosstalkTransitions> {
        &self.transitions
    }

    /// Create global crosstalk triggered only by noisy `PZ` preparation.
    ///
    /// # Errors
    /// Returns an error if `rate` is not a valid probability.
    pub fn preparation(rate: f64) -> ValidationResult<Self> {
        validate_probability(rate, "rate")?;
        Ok(Self {
            kind: CrosstalkKind::Preparation,
            global_rate: rate,
            ..Self::measurement()
        })
    }

    /// Create measurement crosstalk with zero local and global rates.
    #[must_use]
    pub fn measurement() -> Self {
        Self {
            kind: CrosstalkKind::Measurement,
            local_rate: 0.0,
            global_rate: 0.0,
            neighbor_fn: None,
            leakage_probability: 0.0,
            transitions: None,
        }
    }

    fn require_measurement(&self, field: &str) -> ValidationResult<()> {
        if self.kind == CrosstalkKind::Preparation {
            return Err(ValidationError::new(
                field,
                "the engines have no local or table-driven preparation crosstalk",
            )
            .with_suggestion(
                "Use CrosstalkChannel::measurement() for local, neighbor-function or table-driven crosstalk",
            ));
        }
        Ok(())
    }

    /// Set the local crosstalk rate (affects neighbor qubits).
    ///
    /// # Errors
    /// Returns an error for invalid probabilities or a preparation channel.
    pub fn with_local_rate(mut self, rate: f64) -> ValidationResult<Self> {
        self.require_measurement("local_rate")?;
        validate_probability(rate, "rate")?;
        self.local_rate = rate;
        Ok(self)
    }

    /// Set the global crosstalk rate (affects non-neighbor qubits).
    ///
    /// # Errors
    /// Returns an error for invalid probabilities or weights.
    pub fn with_global_rate(mut self, rate: f64) -> ValidationResult<Self> {
        validate_probability(rate, "rate")?;
        self.global_rate = rate;
        Ok(self)
    }

    /// Set the function that determines neighbor qubits.
    ///
    /// Without this, all non-gated qubits are treated as global.
    ///
    /// # Errors
    /// Returns an error for a preparation channel: engines have no local preparation crosstalk.
    pub fn with_neighbor_function(
        mut self,
        func: fn(&[QubitId]) -> Vec<QubitId>,
    ) -> ValidationResult<Self> {
        self.require_measurement("neighbor_function")?;
        self.neighbor_fn = Some(NeighborFunction { func });
        Ok(self)
    }

    /// Set the probability that a crosstalk error causes leakage.
    ///
    /// After the target fires at its rate, draw leakage first. If it does not
    /// leak, draw an independent fair coin: apply Z on heads, otherwise nothing.
    /// This is used with the simple crosstalk model. For state-dependent
    /// transitions (including state-dependent leakage), use `with_transitions`.
    ///
    /// # Errors
    /// Returns an error for invalid probabilities or weights.
    pub fn with_leakage(mut self, probability: f64) -> ValidationResult<Self> {
        validate_probability(probability, "probability")?;
        self.leakage_probability = probability;
        Ok(self)
    }

    /// Set state-dependent transition probabilities for measurement crosstalk.
    ///
    /// The transition model specifies what happens based on the target qubit's
    /// sampled state (0 or 1), currently drawn uniformly rather than measured:
    /// - Stay at current value
    /// - Flip to opposite value
    /// - Leak out of computational subspace
    ///
    /// When this is set, the simple `leakage_probability` is ignored.
    ///
    /// # Errors
    /// Returns an error for an invalid table or a preparation channel.
    pub fn with_transitions(mut self, transitions: CrosstalkTransitions) -> ValidationResult<Self> {
        self.require_measurement("transitions")?;
        transitions.validate()?;
        self.transitions = Some(transitions);
        Ok(self)
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
    /// With a table, use `known_state` or sample a state uniformly.
    /// Without a table, draw leakage first and otherwise dephase.
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

        // Simple model: leakage first, then dephasing
        if rng.random::<f64>() < self.leakage_probability {
            NoiseResponse::MarkLeaked(smallvec::smallvec![qubit])
        } else if rng.random::<bool>() {
            NoiseResponse::inject_gate(GateCommand::new(GateType::Z, smallvec::smallvec![qubit]))
        } else {
            NoiseResponse::None
        }
    }
}

impl NoiseChannel for CrosstalkChannel {
    fn event_kinds(&self) -> super::EventKinds {
        super::EventKinds::of(match self.kind {
            CrosstalkKind::Preparation => super::NoiseEventKind::AfterPreparation,
            CrosstalkKind::Measurement => super::NoiseEventKind::AfterMeasurement,
        })
    }

    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        (self.local_rate > 0.0 || self.global_rate > 0.0)
            && self.event_kinds().contains(event.kind())
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        if ctx.is_noiseless_operation(event) {
            return NoiseResponse::None;
        }
        if matches!(event, NoiseEvent::AfterPreparation { gate_type, .. } if *gate_type != GateType::PZ)
        {
            return NoiseResponse::None;
        }
        let gated_qubits = event.gated_qubits();

        let mut responses: Vec<NoiseResponse> = Vec::new();
        let mut leaked: SmallVec<[QubitId; 4]> = SmallVec::new();
        let mut gates: SmallVec<[GateCommand; 4]> = SmallVec::new();

        // Get neighbors for local crosstalk
        let neighbors = self.get_neighbors(&gated_qubits);

        // Apply local crosstalk to neighbor qubits
        if self.local_rate > 0.0 {
            let local_targets = ctx.local_crosstalk_targets(&gated_qubits, &neighbors);
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
            // Global targets are all existing non-leaked qubits except gated qubits and neighbors
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
        match self.kind {
            CrosstalkKind::Preparation => "PreparationCrosstalkChannel",
            CrosstalkKind::Measurement => "MeasurementCrosstalkChannel",
        }
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
        let channel = CrosstalkChannel::measurement()
            .with_local_rate(0.01)
            .unwrap()
            .with_global_rate(0.001)
            .unwrap()
            .with_leakage(0.1)
            .unwrap();

        assert_ne!(
            channel.name(),
            CrosstalkChannel::preparation(0.01).unwrap().name()
        );
        assert!((channel.local_rate - 0.01).abs() < 1e-10);
        assert!((channel.global_rate - 0.001).abs() < 1e-10);
        assert!((channel.leakage_probability - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_preparation_crosstalk() {
        let channel = CrosstalkChannel::preparation(0.1).unwrap();

        let qubits = [QubitId(0)];
        let event = NoiseEvent::AfterPreparation {
            gate_type: GateType::PZ,
            qubits: &qubits,
        };

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
    fn test_measured_targets_remain_but_leaked_targets_do_not() {
        let channel = CrosstalkChannel::preparation(1.0).unwrap();
        let event = NoiseEvent::AfterPreparation {
            gate_type: GateType::PZ,
            qubits: &[QubitId(0)],
        };
        let mut ctx = NoiseContext::new();
        ctx.mark_prepared(QubitId(0));
        ctx.mark_prepared(QubitId(1));
        ctx.mark_measured(QubitId(1));
        let mut rng = PecosRng::seed_from_u64(42);
        let count = (0..1000)
            .filter(|_| !channel.apply(&event, &mut ctx, &mut rng).is_none())
            .count();
        assert!((421..=579).contains(&count), "{count}");
        ctx.mark_leaked(QubitId(1));
        for _ in 0..1000 {
            assert!(channel.apply(&event, &mut ctx, &mut rng).is_none());
        }
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

        let channel = CrosstalkChannel::measurement()
            .with_local_rate(1.0)
            .unwrap() // Always hit local
            .with_global_rate(0.0)
            .unwrap() // Never hit global
            .with_neighbor_function(neighbors)
            .unwrap();

        let qubits = [QubitId(5)];
        let event = NoiseEvent::AfterMeasurement {
            gate_type: GateType::MZ,
            qubits: &qubits,
            outcomes: &[false],
        };

        // Prepare qubits 0-10
        let mut ctx = NoiseContext::new();
        for i in 0..=10 {
            ctx.mark_prepared(QubitId(i));
        }

        ctx.mark_measured(QubitId(4));
        ctx.mark_leaked(QubitId(6));
        let mut rng = PecosRng::seed_from_u64(42);
        let mut count = 0;
        for _ in 0..1000 {
            if let NoiseResponse::InjectGates(gates) = channel.apply(&event, &mut ctx, &mut rng) {
                for gate in gates.iter() {
                    assert_eq!(gate.qubits.as_slice(), &[QubitId(4)]);
                    assert_eq!(gate.gate_type, GateType::Z);
                    count += 1;
                }
            }
        }
        assert!((421..=579).contains(&count), "{count}");
    }

    #[test]
    fn test_crosstalk_with_transitions() {
        // Create transition model: always flip, never leak
        let transitions = CrosstalkTransitions::custom(
            0.0, 1.0, 0.0, // from 0: always flip
            0.0, 1.0, 0.0, // from 1: always flip
        )
        .unwrap();

        let channel = CrosstalkChannel::measurement()
            .with_global_rate(1.0)
            .unwrap()
            .with_transitions(transitions)
            .unwrap();

        let qubits = [QubitId(0)];
        let event = NoiseEvent::AfterMeasurement {
            gate_type: GateType::MZ,
            qubits: &qubits,
            outcomes: &[false],
        };

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
        )
        .unwrap();

        let channel = CrosstalkChannel::measurement()
            .with_global_rate(1.0)
            .unwrap()
            .with_transitions(transitions)
            .unwrap();

        let qubits = [QubitId(0)];
        let event = NoiseEvent::AfterMeasurement {
            gate_type: GateType::MZ,
            qubits: &qubits,
            outcomes: &[false],
        };

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

    #[test]
    fn leakage_precedes_dephasing_for_fired_targets() {
        for channel in [
            CrosstalkChannel::preparation(1.0).unwrap(),
            CrosstalkChannel::measurement()
                .with_global_rate(1.0)
                .unwrap(),
        ] {
            let channel = channel.with_leakage(0.25).unwrap();
            let mut rng = PecosRng::seed_from_u64(42);
            let mut expected = rng.clone();
            let mut counts = [0; 3];
            for _ in 0..1000 {
                let leaks = expected.random::<f64>() < 0.25;
                let z = !leaks && expected.random::<bool>();
                let response = channel.apply_crosstalk_to_qubit(QubitId(1), None, &mut rng);
                match response {
                    NoiseResponse::MarkLeaked(qubits) => {
                        assert!(leaks);
                        assert_eq!(qubits.as_slice(), &[QubitId(1)]);
                        counts[0] += 1;
                    }
                    NoiseResponse::InjectGates(gates) => {
                        assert!(z);
                        for gate in gates.iter() {
                            assert_eq!(gate.gate_type, GateType::Z);
                            assert_eq!(gate.qubits.as_slice(), &[QubitId(1)]);
                            counts[1] += 1;
                        }
                    }
                    NoiseResponse::None => {
                        assert!(!leaks && !z);
                        counts[2] += 1;
                    }
                    other => panic!("Unexpected response: {other:?}"),
                }
            }
            assert!((180..=320).contains(&counts[0]), "{counts:?}");
            assert!((298..=452).contains(&counts[1]), "{counts:?}");
            assert!((298..=452).contains(&counts[2]), "{counts:?}");
            assert_eq!(rng.random::<u64>(), expected.random::<u64>());
        }
    }
}
