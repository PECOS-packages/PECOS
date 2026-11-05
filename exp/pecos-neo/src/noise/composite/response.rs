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

//! Noise response types for the composite-based noise system.

use crate::command::GateCommand;
use crate::noise::NoiseResponse;
use pecos_core::QubitId;
use smallvec::SmallVec;

/// Result of applying a noise primitive.
///
/// This represents what action(s) to take after evaluating a noise decision tree.
#[derive(Debug, Clone, Default)]
pub enum CompositeResponse {
    /// No noise applied - continue normally
    #[default]
    None,

    /// Inject these gates (typically after the current gate)
    InjectGates(Vec<GateCommand>),

    /// Skip/remove the current gate for this qubit
    SkipGate,

    /// Stop this member's sequence for a skip already emitted by a wrapped channel.
    ///
    /// This carries no additional runner effect, avoiding duplicate skip targets.
    SkipGateAlreadyReported,

    /// Mark the qubit as leaked
    Leak,

    /// Mark the qubit as unleaked (seeped back)
    Unleak,

    /// Flip the measurement outcome (0 <-> 1)
    FlipOutcome,

    /// Force the measurement outcome to a specific value
    ForceOutcome(bool),

    /// Mark measurement as coming from a leaked qubit.
    ///
    /// This corresponds to `MeasureLeaked` behavior where the outcome
    /// should be reported as 2 (or the special leaked indicator) rather
    /// than 0 or 1.
    LeakedMeasurement,

    /// A wrapped channel response, retaining every named qubit.
    Channel(Box<NoiseResponse>),

    /// Multiple responses to combine
    Multiple(Vec<CompositeResponse>),
}

impl CompositeResponse {
    /// Check if this response has any effect.
    #[must_use]
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Check if this response skips any qubit, independent of the current member.
    #[must_use]
    pub fn skips_gate(&self) -> bool {
        match self {
            Self::SkipGate | Self::SkipGateAlreadyReported => true,
            Self::Channel(response) => channel_any(
                response,
                |r| matches!(r, NoiseResponse::SkipGate(qubits) if !qubits.is_empty()),
            ),
            Self::Multiple(responses) => responses.iter().any(Self::skips_gate),
            _ => false,
        }
    }

    /// Check if this response skips the current member identified by `qubit`.
    ///
    /// Ordinary skip responses refer to the current member. Wrapped channel
    /// responses skip only their explicitly named targets.
    #[must_use]
    pub fn skips_qubit(&self, qubit: QubitId) -> bool {
        match self {
            Self::SkipGate | Self::SkipGateAlreadyReported => true,
            Self::Channel(response) => channel_any(
                response,
                |r| matches!(r, NoiseResponse::SkipGate(qubits) if qubits.contains(&qubit)),
            ),
            Self::Multiple(responses) => responses.iter().any(|r| r.skips_qubit(qubit)),
            _ => false,
        }
    }

    /// Targets reported by wrapped channels, including through enclosing sequences.
    pub(crate) fn channel_skipped_qubits(&self) -> SmallVec<[QubitId; 4]> {
        match self {
            Self::Channel(response) => response.skipped_qubits(),
            Self::Multiple(responses) => responses
                .iter()
                .flat_map(Self::channel_skipped_qubits)
                .collect(),
            _ => SmallVec::new(),
        }
    }

    /// Check if this response causes leakage.
    #[must_use]
    pub fn causes_leak(&self) -> bool {
        match self {
            Self::Leak => true,
            Self::Channel(response) => channel_any(
                response,
                |r| matches!(r, NoiseResponse::MarkLeaked(qubits) if !qubits.is_empty()),
            ),
            Self::Multiple(responses) => responses.iter().any(Self::causes_leak),
            _ => false,
        }
    }

    /// Check if this response clears leakage for any qubit.
    #[must_use]
    pub fn causes_unleak(&self) -> bool {
        match self {
            Self::Unleak => true,
            Self::Channel(response) => channel_any(
                response,
                |r| matches!(r, NoiseResponse::MarkUnleaked(qubits) if !qubits.is_empty()),
            ),
            Self::Multiple(responses) => responses.iter().any(Self::causes_unleak),
            _ => false,
        }
    }

    /// Check if this response flips the outcome.
    #[must_use]
    pub fn flips_outcome(&self) -> bool {
        match self {
            Self::FlipOutcome => true,
            Self::Channel(response) => channel_any(
                response,
                |r| matches!(r, NoiseResponse::FlipOutcomes(qubits) if !qubits.is_empty()),
            ),
            Self::Multiple(responses) => responses.iter().any(Self::flips_outcome),
            _ => false,
        }
    }

    /// Check if this response forces an outcome.
    #[must_use]
    pub fn forces_outcome(&self) -> Option<bool> {
        match self {
            Self::ForceOutcome(value) => Some(*value),
            Self::Channel(response) => channel_forced_outcome(response),
            Self::Multiple(responses) => {
                // Return the last forced outcome (in case of multiple)
                responses
                    .iter()
                    .filter_map(Self::forces_outcome)
                    .next_back()
            }
            _ => None,
        }
    }

    /// Check if this response indicates a leaked measurement (outcome = 2).
    #[must_use]
    pub fn is_leaked_measurement(&self) -> bool {
        match self {
            Self::LeakedMeasurement => true,
            Self::Channel(response) => channel_any(
                response,
                |r| matches!(r, NoiseResponse::LeakedMeasurement(qubits) if !qubits.is_empty()),
            ),
            Self::Multiple(responses) => responses.iter().any(Self::is_leaked_measurement),
            _ => false,
        }
    }

    /// Combine two responses.
    #[must_use]
    pub fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::None, other) => other,
            (this, Self::None) => this,
            (Self::Multiple(mut a), Self::Multiple(b)) => {
                a.extend(b);
                Self::Multiple(a)
            }
            (Self::Multiple(mut a), other) => {
                a.push(other);
                Self::Multiple(a)
            }
            (this, Self::Multiple(mut b)) => {
                b.insert(0, this);
                Self::Multiple(b)
            }
            (this, other) => Self::Multiple(vec![this, other]),
        }
    }

    /// Collect all injected gates from this response.
    #[must_use]
    pub fn collect_gates(&self) -> Vec<GateCommand> {
        match self {
            Self::InjectGates(gates) => gates.clone(),
            Self::Channel(response) => channel_gates(response),
            Self::Multiple(responses) => responses.iter().flat_map(Self::collect_gates).collect(),
            _ => Vec::new(),
        }
    }
}

/// Query all leaves without allocating or discarding their qubit targets.
fn channel_any(
    response: &NoiseResponse,
    predicate: impl Fn(&NoiseResponse) -> bool + Copy,
) -> bool {
    match response {
        NoiseResponse::Multiple(responses) => responses.iter().any(|r| channel_any(r, predicate)),
        response => predicate(response),
    }
}

fn channel_forced_outcome(response: &NoiseResponse) -> Option<bool> {
    match response {
        NoiseResponse::ForceOutcomes(outcomes) => outcomes.last().map(|(_, value)| *value),
        NoiseResponse::Multiple(responses) => responses
            .iter()
            .filter_map(channel_forced_outcome)
            .next_back(),
        _ => None,
    }
}

fn channel_gates(response: &NoiseResponse) -> Vec<GateCommand> {
    match response {
        NoiseResponse::InjectGates(gates) => gates.to_vec(),
        NoiseResponse::Multiple(responses) => responses.iter().flat_map(channel_gates).collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skip_queries_distinguish_the_current_member_from_other_targets() {
        use smallvec::smallvec;
        let response = CompositeResponse::Multiple(vec![
            CompositeResponse::None,
            CompositeResponse::Channel(Box::new(NoiseResponse::Multiple(vec![
                NoiseResponse::MarkLeaked(smallvec![QubitId(0)]),
                NoiseResponse::Multiple(vec![NoiseResponse::SkipGate(smallvec![QubitId(1)])]),
            ]))),
        ]);
        assert!(response.skips_gate());
        assert!(!response.skips_qubit(QubitId(0)));
        assert!(response.skips_qubit(QubitId(1)));
        assert_eq!(response.channel_skipped_qubits().as_slice(), &[QubitId(1)]);
        for response in [
            CompositeResponse::SkipGate,
            CompositeResponse::SkipGateAlreadyReported,
        ] {
            assert!(response.skips_qubit(QubitId(0)));
            assert!(response.skips_qubit(QubitId(1)));
            assert!(response.channel_skipped_qubits().is_empty());
        }
        let empty = CompositeResponse::Channel(Box::new(NoiseResponse::SkipGate(smallvec![])));
        assert!(!empty.skips_qubit(QubitId(0)));
        assert!(!CompositeResponse::None.skips_qubit(QubitId(0)));
    }

    #[test]
    fn channel_helpers_answer_for_every_named_qubit() {
        use pecos_core::QubitId;
        use smallvec::smallvec;
        let qubits = smallvec![QubitId(0), QubitId(1)];
        let cases = [
            NoiseResponse::SkipGate(qubits.clone()),
            NoiseResponse::MarkLeaked(qubits.clone()),
            NoiseResponse::MarkUnleaked(qubits.clone()),
            NoiseResponse::FlipOutcomes(qubits.clone()),
            NoiseResponse::LeakedMeasurement(qubits),
            NoiseResponse::ForceOutcomes(smallvec![(QubitId(0), false), (QubitId(1), true)]),
            NoiseResponse::inject_gates(smallvec![
                GateCommand::x(QubitId(0)),
                GateCommand::z(QubitId(1))
            ]),
        ];
        for (i, response) in cases.into_iter().enumerate() {
            let r = CompositeResponse::Channel(Box::new(NoiseResponse::Multiple(vec![
                NoiseResponse::ForceOutcomes(smallvec![(QubitId(0), false)]),
                NoiseResponse::Multiple(vec![response]),
            ])));
            assert!(!r.is_none());
            assert_eq!(r.skips_gate(), i == 0);
            assert_eq!(r.causes_leak(), i == 1);
            assert_eq!(r.causes_unleak(), i == 2);
            assert_eq!(r.flips_outcome(), i == 3);
            assert_eq!(r.is_leaked_measurement(), i == 4);
            assert_eq!(r.forces_outcome(), Some(i == 5));
            let gates = if i == 6 {
                vec![GateCommand::x(QubitId(0)), GateCommand::z(QubitId(1))]
            } else {
                vec![]
            };
            assert_eq!(r.collect_gates(), gates);
            let combined = CompositeResponse::None
                .combine(r)
                .combine(CompositeResponse::Unleak);
            assert_eq!(combined.skips_gate(), i == 0);
            assert_eq!(combined.forces_outcome(), Some(i == 5));
            assert_eq!(combined.collect_gates(), gates);
        }
        let empty = CompositeResponse::Channel(Box::new(NoiseResponse::Multiple(vec![
            NoiseResponse::SkipGate(smallvec![]),
            NoiseResponse::MarkLeaked(smallvec![]),
            NoiseResponse::MarkUnleaked(smallvec![]),
            NoiseResponse::FlipOutcomes(smallvec![]),
            NoiseResponse::LeakedMeasurement(smallvec![]),
            NoiseResponse::ForceOutcomes(smallvec![]),
        ])));
        assert!(!empty.skips_gate());
        assert!(!empty.causes_leak());
        assert!(!empty.causes_unleak());
        assert!(!empty.flips_outcome());
        assert!(!empty.is_leaked_measurement());
        assert_eq!(empty.forces_outcome(), None);
        assert_eq!(empty.collect_gates(), Vec::<GateCommand>::new());
    }

    #[test]
    fn test_response_none() {
        let r = CompositeResponse::None;
        assert!(r.is_none());
        assert!(!r.skips_gate());
        assert!(!r.causes_leak());
    }

    #[test]
    fn test_response_skip_gate() {
        let r = CompositeResponse::SkipGate;
        assert!(!r.is_none());
        assert!(r.skips_gate());
    }

    #[test]
    fn test_response_leak() {
        let r = CompositeResponse::Leak;
        assert!(r.causes_leak());
    }

    #[test]
    fn test_response_combine() {
        let r1 = CompositeResponse::SkipGate;
        let r2 = CompositeResponse::Leak;
        let combined = r1.combine(r2);

        assert!(combined.skips_gate());
        assert!(combined.causes_leak());
    }

    #[test]
    fn test_response_combine_with_none() {
        let r1 = CompositeResponse::None;
        let r2 = CompositeResponse::SkipGate;
        let combined = r1.combine(r2);

        assert!(matches!(combined, CompositeResponse::SkipGate));
    }
}
