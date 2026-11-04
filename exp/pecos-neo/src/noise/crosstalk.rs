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

//! Preparation dephasing and payload-driven measurement crosstalk.
//!
//! Global payload operands exclude victims; local payload operands name victims.
//! Targets exist this shot and are not leaked, including measured qubits.
//! Measurement crosstalk collapses each fired victim and uses its raw outcome
//! for optional transitions. Preparation crosstalk uses a fair Z coin.

use super::{
    CrosstalkResult, CrosstalkTransitions, NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse,
};
use crate::command::{GateCommand, GateType};
use crate::noise::validation::{ValidationError, ValidationResult, validate_probability};
use pecos_core::QubitId;
use pecos_random::PecosRng;
use rand::RngExt;
use smallvec::SmallVec;

/// Which measurement-crosstalk payload a channel owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadKind {
    /// Operands exclude victims from the existing, non-leaked qubits.
    Global,
    /// Operands name potential victims, filtered by existence and leakage.
    Local,
}

impl PayloadKind {
    /// The metadata gate associated with this kind.
    #[must_use]
    pub const fn gate_type(self) -> GateType {
        match self {
            Self::Global => GateType::MeasCrosstalkGlobalPayload,
            Self::Local => GateType::MeasCrosstalkLocalPayload,
        }
    }

    pub(crate) fn from_gate_type(gate_type: GateType) -> Option<Self> {
        match gate_type {
            GateType::MeasCrosstalkGlobalPayload => Some(Self::Global),
            GateType::MeasCrosstalkLocalPayload => Some(Self::Local),
            _ => None,
        }
    }

    pub(crate) fn measure(
        self,
        qubits: &[QubitId],
        rate: f64,
        ctx: &NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        if rate == 0.0 {
            return NoiseResponse::None;
        }

        let targets = match self {
            Self::Global => ctx.global_crosstalk_targets(qubits),
            Self::Local => ctx.local_crosstalk_targets(&[], qubits),
        };
        let qubits: SmallVec<[QubitId; 4]> = targets
            .into_iter()
            .filter(|_| rng.random::<f64>() < rate)
            .collect();
        if qubits.is_empty() {
            NoiseResponse::None
        } else {
            NoiseResponse::MeasureHidden {
                gate_type: self.gate_type(),
                qubits,
            }
        }
    }
}

/// Preparation dephasing or measurement crosstalk on both payload kinds.
///
/// Preparation fires only on noisy PZ. Measurement crosstalk selects victims
/// at separate global/local rates, measures them invisibly, and applies an
/// optional stay/flip/leak table using each actual outcome. Without a table,
/// measurement only collapses the victim.
#[derive(Debug, Clone)]
pub struct CrosstalkChannel {
    kind: CrosstalkKind,
    local_rate: f64,
    global_rate: f64,
    transitions: Option<CrosstalkTransitions>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrosstalkKind {
    Preparation,
    Measurement,
}

impl CrosstalkChannel {
    /// Return the local payload rate.
    #[must_use]
    pub fn local_rate(&self) -> f64 {
        self.local_rate
    }

    /// Return the global rate.
    #[must_use]
    pub fn global_rate(&self) -> f64 {
        self.global_rate
    }

    /// Return the optional measurement transition table.
    #[must_use]
    pub fn transitions(&self) -> &Option<CrosstalkTransitions> {
        &self.transitions
    }

    /// Create global crosstalk triggered only by noisy PZ preparation.
    ///
    /// # Errors
    /// Returns an error if the rate is not a probability.
    pub fn preparation(rate: f64) -> ValidationResult<Self> {
        validate_probability(rate, "rate")?;
        Ok(Self {
            kind: CrosstalkKind::Preparation,
            global_rate: rate,
            ..Self::measurement()
        })
    }

    /// Create a channel owning both measurement payload kinds, initially at zero rates.
    #[must_use]
    pub fn measurement() -> Self {
        Self {
            kind: CrosstalkKind::Measurement,
            local_rate: 0.0,
            global_rate: 0.0,
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
                "Use CrosstalkChannel::measurement() for local or table-driven crosstalk",
            ));
        }
        Ok(())
    }

    /// Set the local payload rate.
    ///
    /// # Errors
    /// Returns an error for an invalid probability or a preparation channel.
    pub fn with_local_rate(mut self, rate: f64) -> ValidationResult<Self> {
        self.require_measurement("local_rate")?;
        validate_probability(rate, "rate")?;
        self.local_rate = rate;
        Ok(self)
    }

    /// Set the global rate.
    ///
    /// # Errors
    /// Returns an error for an invalid probability.
    pub fn with_global_rate(mut self, rate: f64) -> ValidationResult<Self> {
        validate_probability(rate, "rate")?;
        self.global_rate = rate;
        Ok(self)
    }

    /// Set the stay/flip/leak table selected by each hidden measurement outcome.
    ///
    /// # Errors
    /// Returns an error for an invalid table or a preparation channel.
    pub fn with_transitions(mut self, transitions: CrosstalkTransitions) -> ValidationResult<Self> {
        self.require_measurement("transitions")?;
        transitions.validate()?;
        self.transitions = Some(transitions);
        Ok(self)
    }

    /// Dephase one fired preparation target.
    fn apply_crosstalk_to_qubit(qubit: QubitId, rng: &mut PecosRng) -> NoiseResponse {
        if rng.random::<bool>() {
            NoiseResponse::inject_gate(GateCommand::z(qubit))
        } else {
            NoiseResponse::None
        }
    }
}

impl NoiseChannel for CrosstalkChannel {
    fn event_kinds(&self) -> super::EventKinds {
        use super::{EventKinds, NoiseEventKind};
        match self.kind {
            CrosstalkKind::Preparation => EventKinds::of(NoiseEventKind::AfterPreparation),
            CrosstalkKind::Measurement => EventKinds::of(NoiseEventKind::CrosstalkPayload)
                .with(NoiseEventKind::AfterHiddenMeasurement),
        }
    }

    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        self.event_kinds().contains(event.kind())
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
        let mut response = NoiseResponse::None;
        match (self.kind, event) {
            (
                CrosstalkKind::Preparation,
                NoiseEvent::AfterPreparation {
                    gate_type: GateType::PZ,
                    qubits,
                },
            ) => {
                for target in ctx.global_crosstalk_targets(qubits) {
                    if rng.random::<f64>() < self.global_rate {
                        response = response.combine(Self::apply_crosstalk_to_qubit(target, rng));
                    }
                }
            }
            (CrosstalkKind::Measurement, NoiseEvent::CrosstalkPayload { gate_type, qubits }) => {
                if let Some(kind) = PayloadKind::from_gate_type(*gate_type) {
                    let rate = match kind {
                        PayloadKind::Global => self.global_rate,
                        PayloadKind::Local => self.local_rate,
                    };
                    return kind.measure(qubits, rate, ctx, rng);
                }
            }
            (
                CrosstalkKind::Measurement,
                NoiseEvent::AfterHiddenMeasurement {
                    gate_type,
                    qubits,
                    outcomes,
                },
            ) if gate_type.is_crosstalk_payload() => {
                if let Some(transitions) = self.transitions {
                    for (&qubit, &outcome) in qubits.iter().zip(*outcomes) {
                        let transition = match transitions.sample(outcome, rng.random::<f64>()) {
                            CrosstalkResult::NoChange => NoiseResponse::None,
                            CrosstalkResult::Flip => {
                                NoiseResponse::inject_gate(GateCommand::x(qubit))
                            }
                            CrosstalkResult::Leak => {
                                NoiseResponse::MarkLeaked(smallvec::smallvec![qubit])
                            }
                        };
                        response = response.combine(transition);
                    }
                }
            }
            _ => {}
        }
        response
    }

    fn name(&self) -> &'static str {
        match self.kind {
            CrosstalkKind::Preparation => "PreparationCrosstalkChannel",
            CrosstalkKind::Measurement => "MeasurementCrosstalkChannel",
        }
    }

    fn priority(&self) -> i32 {
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
            .unwrap();

        assert_ne!(
            channel.name(),
            CrosstalkChannel::preparation(0.01).unwrap().name()
        );
        assert!((channel.local_rate - 0.01).abs() < 1e-10);
        assert!((channel.global_rate - 0.001).abs() < 1e-10);
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

        let qubits = [QubitId(1)];
        let event = NoiseEvent::AfterHiddenMeasurement {
            gate_type: GateType::MeasCrosstalkGlobalPayload,
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

        let qubits = [QubitId(1)];
        let event = NoiseEvent::AfterHiddenMeasurement {
            gate_type: GateType::MeasCrosstalkGlobalPayload,
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
    fn preparation_fired_targets_only_dephase() {
        let mut rng = PecosRng::seed_from_u64(42);
        let mut count = 0;
        for _ in 0..1000 {
            match CrosstalkChannel::apply_crosstalk_to_qubit(QubitId(1), &mut rng) {
                NoiseResponse::InjectGates(gates) => {
                    assert_eq!(gates.len(), 1);
                    assert_eq!(gates[0].gate_type, GateType::Z);
                    count += 1;
                }
                NoiseResponse::None => {}
                other => panic!("Unexpected preparation response {other:?}"),
            }
        }
        assert!((421..=579).contains(&count));
    }
}
