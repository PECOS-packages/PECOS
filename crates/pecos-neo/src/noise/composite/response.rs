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

    /// Multiple responses to combine
    Multiple(Vec<CompositeResponse>),
}

impl CompositeResponse {
    /// Check if this response has any effect.
    #[must_use]
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Check if this response skips the gate.
    #[must_use]
    pub fn skips_gate(&self) -> bool {
        match self {
            Self::SkipGate => true,
            Self::Multiple(responses) => responses.iter().any(Self::skips_gate),
            _ => false,
        }
    }

    /// Check if this response causes leakage.
    #[must_use]
    pub fn causes_leak(&self) -> bool {
        match self {
            Self::Leak => true,
            Self::Multiple(responses) => responses.iter().any(Self::causes_leak),
            _ => false,
        }
    }

    /// Check if this response flips the outcome.
    #[must_use]
    pub fn flips_outcome(&self) -> bool {
        match self {
            Self::FlipOutcome => true,
            Self::Multiple(responses) => responses.iter().any(Self::flips_outcome),
            _ => false,
        }
    }

    /// Check if this response forces an outcome.
    #[must_use]
    pub fn forces_outcome(&self) -> Option<bool> {
        match self {
            Self::ForceOutcome(value) => Some(*value),
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
            Self::Multiple(responses) => responses.iter().flat_map(Self::collect_gates).collect(),
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
