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

//! Compiled primitives for fast enum-based dispatch.
//!
//! This module provides an optimized representation of noise primitives that uses
//! enum dispatch instead of trait objects. The [`CompiledPrimitive`] enum covers
//! common primitive patterns with direct match-based dispatch, falling back to
//! Arc-wrapped trait objects only for custom user-defined primitives.
//!
//! # Performance
//!
//! Enum dispatch is typically 2-4x faster than trait object dispatch because:
//! - No vtable lookup
//! - Better branch prediction
//! - Better cache locality (no pointer chasing)
//!
//! # Usage
//!
//! The builder automatically compiles primitives at build time. Users don't need
//! to interact with this module directly.

use super::Primitive;
use super::action::PauliWeights;
use super::response::CompositeResponse;
use crate::command::{GateCommand, GateType};
use crate::noise::{
    NoiseContext, SingleQubitEmissionWeights, TwoQubitEmissionWeights, TwoQubitPauliWeights,
};
use pecos_core::QubitId;
use pecos_rng::PecosRng;
use rand::RngExt;
use smallvec::smallvec;
use std::sync::Arc;

// ============================================================================
// Compiled Action Enum
// ============================================================================

/// Compiled action for fast enum dispatch.
///
/// Covers common action types with direct implementation, avoiding trait dispatch.
#[derive(Clone)]
pub enum CompiledAction {
    /// No action.
    Nothing,
    /// Skip the current gate.
    SkipGate,
    /// Mark qubit as leaked.
    Leak,
    /// Mark qubit as unleaked.
    Unleak,
    /// Seep: unleak and apply random Pauli.
    Seep(PauliWeights),
    /// Inject a specific gate.
    Inject(GateType),
    /// Random Pauli with weights.
    Pauli(PauliWeights),
    /// Single-qubit emission (Pauli or leak).
    Emission(SingleQubitEmissionWeights),
    /// Two-qubit correlated Pauli.
    TwoQubitPauli(TwoQubitPauliWeights),
    /// Two-qubit emission.
    TwoQubitEmission(TwoQubitEmissionWeights),
    /// Flip measurement outcome.
    FlipOutcome,
    /// Force measurement outcome to specific value.
    ForceOutcome(bool),
    /// Mark measurement as leaked.
    LeakedMeasurement,
    /// Crosstalk with transitions.
    CrosstalkTransitions(crate::noise::CrosstalkTransitions),
    /// Fallback to Arc-wrapped trait object for custom actions.
    Custom(Arc<dyn Primitive>),
}

impl CompiledAction {
    /// Apply this action.
    #[inline]
    pub fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        match self {
            Self::Nothing => CompositeResponse::None,
            Self::SkipGate => CompositeResponse::SkipGate,
            Self::Leak => {
                ctx.mark_leaked(qubit);
                CompositeResponse::Leak
            }
            Self::Unleak => {
                ctx.mark_unleaked(qubit);
                CompositeResponse::Unleak
            }
            Self::Seep(weights) => apply_seep(qubit, ctx, rng, weights),
            Self::Inject(gate_type) => {
                let cmd = GateCommand {
                    gate_type: *gate_type,
                    qubits: smallvec![qubit],
                    angles: smallvec![],
                };
                CompositeResponse::InjectGates(vec![cmd])
            }
            Self::Pauli(weights) => apply_pauli(qubit, rng, weights),
            Self::Emission(weights) => apply_emission(qubit, ctx, rng, weights),
            Self::TwoQubitPauli(weights) => apply_two_qubit_pauli(qubit, ctx, rng, weights),
            Self::TwoQubitEmission(weights) => apply_two_qubit_emission(qubit, ctx, rng, weights),
            Self::FlipOutcome => CompositeResponse::FlipOutcome,
            Self::ForceOutcome(value) => CompositeResponse::ForceOutcome(*value),
            Self::LeakedMeasurement => CompositeResponse::LeakedMeasurement,
            Self::CrosstalkTransitions(transitions) => {
                apply_crosstalk_transitions(qubit, ctx, rng, transitions)
            }
            Self::Custom(prim) => prim.apply(qubit, ctx, rng),
        }
    }
}

// ============================================================================
// Compiled Condition Enum
// ============================================================================

/// Compiled condition for fast enum dispatch.
#[derive(Clone, Copy)]
pub enum CompiledCondition {
    /// Qubit is leaked.
    Leaked,
    /// Qubit is not leaked.
    NotLeaked,
    /// Qubit is active.
    Active,
    /// Always true.
    Always,
    /// Always false.
    Never,
    /// Outcome equals value.
    OutcomeIs(bool),
    /// Gate type matches.
    GateTypeIs(GateType),
}

impl CompiledCondition {
    /// Evaluate this condition.
    #[inline]
    #[must_use]
    pub fn evaluate(&self, qubit: QubitId, ctx: &NoiseContext) -> bool {
        match self {
            Self::Leaked => ctx.is_leaked(qubit),
            Self::NotLeaked => !ctx.is_leaked(qubit),
            Self::Active => ctx.is_active(qubit),
            Self::Always => true,
            Self::Never => false,
            Self::OutcomeIs(expected) => ctx.current_outcome() == Some(*expected),
            Self::GateTypeIs(expected) => ctx
                .current_gate()
                .is_some_and(|info| info.gate_type == *expected),
        }
    }
}

// ============================================================================
// Compiled Primitive Enum
// ============================================================================

/// Compiled primitive for fast enum dispatch.
///
/// This enum covers common primitive patterns with direct match-based dispatch.
/// Complex or custom primitives fall back to Arc-wrapped trait objects.
#[derive(Clone)]
pub enum CompiledPrimitive {
    /// Terminal action.
    Action(CompiledAction),
    /// Probability gate: with probability p, execute inner.
    Prob {
        probability: f64,
        inner: Box<CompiledPrimitive>,
    },
    /// Conditional: if condition, then else.
    When {
        condition: CompiledCondition,
        then_branch: Box<CompiledPrimitive>,
        else_branch: Box<CompiledPrimitive>,
    },
    /// Weighted sample from branches.
    Sample {
        branches: Vec<(f64, CompiledPrimitive)>,
        cumulative_weights: Vec<f64>,
    },
    /// Sequential execution.
    Seq(Vec<CompiledPrimitive>),
    /// Skip if condition is true.
    SkipIf(CompiledCondition),
    /// Fallback to Arc-wrapped trait object.
    Custom(Arc<dyn Primitive>),
}

impl CompiledPrimitive {
    /// Apply this primitive.
    #[inline]
    pub fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        match self {
            Self::Action(action) => action.apply(qubit, ctx, rng),
            Self::Prob { probability, inner } => {
                if rng.random::<f64>() < *probability {
                    inner.apply(qubit, ctx, rng)
                } else {
                    CompositeResponse::None
                }
            }
            Self::When {
                condition,
                then_branch,
                else_branch,
            } => {
                if condition.evaluate(qubit, ctx) {
                    then_branch.apply(qubit, ctx, rng)
                } else {
                    else_branch.apply(qubit, ctx, rng)
                }
            }
            Self::Sample {
                branches,
                cumulative_weights,
            } => {
                if branches.is_empty() {
                    return CompositeResponse::None;
                }
                let r: f64 = rng.random();
                for (i, &threshold) in cumulative_weights.iter().enumerate() {
                    if r < threshold {
                        return branches[i].1.apply(qubit, ctx, rng);
                    }
                }
                branches
                    .last()
                    .expect("CompiledPrimitive::Sample must have at least one branch")
                    .1
                    .apply(qubit, ctx, rng)
            }
            Self::Seq(primitives) => {
                let mut combined = CompositeResponse::None;
                for prim in primitives {
                    let response = prim.apply(qubit, ctx, rng);
                    if response.skips_gate() {
                        return combined.combine(response);
                    }
                    combined = combined.combine(response);
                }
                combined
            }
            Self::SkipIf(condition) => {
                if condition.evaluate(qubit, ctx) {
                    CompositeResponse::SkipGate
                } else {
                    CompositeResponse::None
                }
            }
            Self::Custom(prim) => prim.apply(qubit, ctx, rng),
        }
    }
}

impl Primitive for CompiledPrimitive {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        CompiledPrimitive::apply(self, qubit, ctx, rng)
    }

    fn describe(&self) -> String {
        match self {
            Self::Action(_) => "action".to_string(),
            Self::Prob { probability, .. } => format!("prob({probability:.4})"),
            Self::When { .. } => "when(...)".to_string(),
            Self::Sample { branches, .. } => format!("sample([{} branches])", branches.len()),
            Self::Seq(prims) => format!("seq([{} items])", prims.len()),
            Self::SkipIf(_) => "skip_if(...)".to_string(),
            Self::Custom(p) => p.describe(),
        }
    }

    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

#[inline]
fn apply_pauli(qubit: QubitId, rng: &mut PecosRng, weights: &PauliWeights) -> CompositeResponse {
    let r: f64 = rng.random();
    let normalized = weights.normalized();

    let gate_type = if r < normalized.x {
        GateType::X
    } else if r < normalized.x + normalized.y {
        GateType::Y
    } else {
        GateType::Z
    };

    let cmd = GateCommand {
        gate_type,
        qubits: smallvec![qubit],
        angles: smallvec![],
    };
    CompositeResponse::InjectGates(vec![cmd])
}

#[inline]
fn apply_seep(
    qubit: QubitId,
    ctx: &mut NoiseContext,
    rng: &mut PecosRng,
    weights: &PauliWeights,
) -> CompositeResponse {
    ctx.mark_unleaked(qubit);

    let r: f64 = rng.random();
    if r < 0.25 {
        return CompositeResponse::Unleak;
    }

    let r_scaled = (r - 0.25) / 0.75;
    let normalized = weights.normalized();

    let gate_type = if r_scaled < normalized.x {
        GateType::X
    } else if r_scaled < normalized.x + normalized.y {
        GateType::Y
    } else {
        GateType::Z
    };

    let cmd = GateCommand {
        gate_type,
        qubits: smallvec![qubit],
        angles: smallvec![],
    };

    CompositeResponse::Unleak.combine(CompositeResponse::InjectGates(vec![cmd]))
}

#[inline]
fn apply_emission(
    qubit: QubitId,
    ctx: &mut NoiseContext,
    rng: &mut PecosRng,
    weights: &SingleQubitEmissionWeights,
) -> CompositeResponse {
    let r: f64 = rng.random();
    let result = weights.sample(r);

    match result {
        crate::noise::SingleQubitEmissionResult::Pauli(gate_type) => {
            let cmd = GateCommand {
                gate_type,
                qubits: smallvec![qubit],
                angles: smallvec![],
            };
            CompositeResponse::InjectGates(vec![cmd])
        }
        crate::noise::SingleQubitEmissionResult::Leaked => {
            ctx.mark_leaked(qubit);
            CompositeResponse::Leak
        }
    }
}

#[inline]
fn apply_two_qubit_pauli(
    qubit: QubitId,
    ctx: &mut NoiseContext,
    rng: &mut PecosRng,
    weights: &TwoQubitPauliWeights,
) -> CompositeResponse {
    let qubit_index = ctx.current_qubit_index();

    let idx = if qubit_index == 0 {
        let r: f64 = rng.random();
        let sampled_idx = weights.sample(r);
        ctx.set_sampled_correlation(sampled_idx);
        sampled_idx
    } else {
        ctx.sampled_correlation().unwrap_or_else(|| {
            let r: f64 = rng.random();
            weights.sample(r)
        })
    };

    let (first_pauli, second_pauli) = TwoQubitPauliWeights::get_paulis(idx);
    let my_pauli = if qubit_index == 0 {
        first_pauli
    } else {
        second_pauli
    };

    if my_pauli == GateType::I {
        CompositeResponse::None
    } else {
        let cmd = GateCommand {
            gate_type: my_pauli,
            qubits: smallvec![qubit],
            angles: smallvec![],
        };
        CompositeResponse::InjectGates(vec![cmd])
    }
}

#[inline]
fn apply_two_qubit_emission(
    qubit: QubitId,
    ctx: &mut NoiseContext,
    rng: &mut PecosRng,
    weights: &TwoQubitEmissionWeights,
) -> CompositeResponse {
    let qubit_index = ctx.current_qubit_index();

    let idx = if qubit_index == 0 {
        let r: f64 = rng.random();
        let sampled_idx = weights.sample(r);
        ctx.set_sampled_correlation(sampled_idx);
        sampled_idx
    } else {
        ctx.sampled_correlation().unwrap_or_else(|| {
            let r: f64 = rng.random();
            weights.sample(r)
        })
    };

    let result = TwoQubitEmissionWeights::get_result(idx);

    // Get the effect for this qubit based on index
    let (my_pauli, my_leaked) = if qubit_index == 0 {
        (result.first, result.first_leaked)
    } else {
        (result.second, result.second_leaked)
    };

    // Handle leakage
    if my_leaked {
        ctx.mark_leaked(qubit);
        return CompositeResponse::Leak;
    }

    // Handle Pauli (if any)
    match my_pauli {
        Some(gate_type) if gate_type != GateType::I => {
            let cmd = GateCommand {
                gate_type,
                qubits: smallvec![qubit],
                angles: smallvec![],
            };
            CompositeResponse::InjectGates(vec![cmd])
        }
        _ => CompositeResponse::None,
    }
}

#[inline]
fn apply_crosstalk_transitions(
    qubit: QubitId,
    ctx: &mut NoiseContext,
    rng: &mut PecosRng,
    transitions: &crate::noise::CrosstalkTransitions,
) -> CompositeResponse {
    // Determine current state from outcome
    let is_one = ctx.current_outcome().unwrap_or(false);

    let r: f64 = rng.random();
    let result = transitions.sample(is_one, r);

    match result {
        crate::noise::CrosstalkResult::NoChange => CompositeResponse::None,
        crate::noise::CrosstalkResult::Flip => {
            let cmd = GateCommand {
                gate_type: GateType::X,
                qubits: smallvec![qubit],
                angles: smallvec![],
            };
            CompositeResponse::InjectGates(vec![cmd])
        }
        crate::noise::CrosstalkResult::Leak => {
            ctx.mark_leaked(qubit);
            CompositeResponse::Leak
        }
    }
}

// ============================================================================
// Builder for Compiled Primitives
// ============================================================================

impl CompiledPrimitive {
    /// Create a compiled action.
    #[must_use]
    pub fn action(action: CompiledAction) -> Self {
        Self::Action(action)
    }

    /// Create a compiled probability gate.
    #[must_use]
    pub fn prob(probability: f64, inner: Self) -> Self {
        Self::Prob {
            probability,
            inner: Box::new(inner),
        }
    }

    /// Create a compiled conditional.
    #[must_use]
    pub fn when(condition: CompiledCondition, then_branch: Self, else_branch: Self) -> Self {
        Self::When {
            condition,
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        }
    }

    /// Create a compiled sequence.
    #[must_use]
    pub fn seq(primitives: Vec<Self>) -> Self {
        // Flatten nested sequences
        let flattened: Vec<Self> = primitives
            .into_iter()
            .flat_map(|p| {
                if let Self::Seq(inner) = p {
                    inner
                } else {
                    vec![p]
                }
            })
            .collect();

        // Optimize single-element sequences
        if flattened.len() == 1 {
            return flattened
                .into_iter()
                .next()
                .expect("flattened has exactly 1 element");
        }

        Self::Seq(flattened)
    }

    /// Create a compiled weighted sample.
    #[must_use]
    pub fn sample(branches: Vec<(f64, Self)>) -> Self {
        if branches.is_empty() {
            return Self::Action(CompiledAction::Nothing);
        }

        // Compute cumulative weights
        let total: f64 = branches.iter().map(|(w, _)| w).sum();
        let mut cumulative = 0.0;
        let cumulative_weights: Vec<f64> = branches
            .iter()
            .map(|(w, _)| {
                cumulative += w / total;
                cumulative
            })
            .collect();

        Self::Sample {
            branches,
            cumulative_weights,
        }
    }

    /// Create a skip-if condition.
    #[must_use]
    pub fn skip_if(condition: CompiledCondition) -> Self {
        Self::SkipIf(condition)
    }

    /// Create from an Arc-wrapped primitive (fallback for custom types).
    pub fn custom(prim: Arc<dyn Primitive>) -> Self {
        Self::Custom(prim)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiled_action_nothing() {
        let action = CompiledAction::Nothing;
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);
        let response = action.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(matches!(response, CompositeResponse::None));
    }

    #[test]
    fn test_compiled_action_pauli() {
        let action = CompiledAction::Pauli(PauliWeights::uniform());
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);
        let response = action.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(matches!(response, CompositeResponse::InjectGates(_)));
    }

    #[test]
    fn test_compiled_primitive_prob() {
        let prim = CompiledPrimitive::Prob {
            probability: 1.0,
            inner: Box::new(CompiledPrimitive::Action(CompiledAction::Pauli(
                PauliWeights::uniform(),
            ))),
        };
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);
        let response = prim.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(matches!(response, CompositeResponse::InjectGates(_)));
    }

    #[test]
    fn test_compiled_primitive_seq() {
        let prim = CompiledPrimitive::Seq(vec![
            CompiledPrimitive::Action(CompiledAction::Nothing),
            CompiledPrimitive::Action(CompiledAction::Inject(GateType::X)),
        ]);
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);
        let response = prim.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(matches!(response, CompositeResponse::InjectGates(_)));
    }

    #[test]
    fn test_compiled_condition_leaked() {
        let mut ctx = NoiseContext::new();
        let cond = CompiledCondition::Leaked;

        assert!(!cond.evaluate(QubitId(0), &ctx));
        ctx.mark_leaked(QubitId(0));
        assert!(cond.evaluate(QubitId(0), &ctx));
    }

    #[test]
    fn test_compiled_primitive_builder() {
        // Test the builder methods
        let prim = CompiledPrimitive::seq(vec![
            CompiledPrimitive::skip_if(CompiledCondition::Leaked),
            CompiledPrimitive::prob(
                0.5,
                CompiledPrimitive::when(
                    CompiledCondition::NotLeaked,
                    CompiledPrimitive::action(CompiledAction::Pauli(PauliWeights::uniform())),
                    CompiledPrimitive::action(CompiledAction::Nothing),
                ),
            ),
        ]);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Should work without panic
        let _response = prim.apply(QubitId(0), &mut ctx, &mut rng);
    }

    #[test]
    fn test_compiled_sample() {
        let prim = CompiledPrimitive::sample(vec![
            (
                0.5,
                CompiledPrimitive::action(CompiledAction::Inject(GateType::X)),
            ),
            (
                0.5,
                CompiledPrimitive::action(CompiledAction::Inject(GateType::Z)),
            ),
        ]);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Run multiple times to test distribution
        let mut x_count = 0;
        for _ in 0..1000 {
            let response = prim.apply(QubitId(0), &mut ctx, &mut rng);
            if let CompositeResponse::InjectGates(gates) = response
                && gates[0].gate_type == GateType::X
            {
                x_count += 1;
            }
        }

        // Should be roughly 50/50
        let x_rate = f64::from(x_count) / 1000.0;
        assert!((x_rate - 0.5).abs() < 0.1, "Expected ~50% X, got {x_rate}");
    }
}
